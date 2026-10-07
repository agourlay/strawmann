//! A text index scored with BM25, Qdrant dev's `scoring` text index
//! (decisions.md, 2026-10-07). One per indexed text field.
//!
//! ## Exact statistics
//!
//! BM25 reads `N` (documents), `df` (documents per term), each document's
//! length and `avgdl`, and all four stay exact under overwrite and delete: a
//! point's term counts are kept, so rewriting or deleting it subtracts what it
//! added. This is where strawmANN departs from Qdrant on purpose: Qdrant's
//! immutable and on-disk indexes keep deleted points in their postings until a
//! rebuild, so its `df` over-counts (decisions.md, the deletions gap).
//!
//! Postings are not rewritten on delete: an entry carries the generation of
//! the document it came from, and one whose point has since moved on is
//! skipped. The statistics never read the postings, so stale entries cost a
//! skip at query time and nothing in a score.
//!
//! ## Qdrant's semantics, mirrored
//!
//! - A point is a document when its values yield a token, or when it holds an
//!   array of two or more values (Qdrant's phrase boundary token,
//!   qdrant/qdrant#11010); its values' tokens concatenate.
//! - The score is Lucene's BM25 over the query's distinct terms, in `f32` in
//!   the order Qdrant computes it (`bm25/mod.rs`), with `idf` clamped at 0.
//! - A point is a candidate when it holds any query term (OR).

const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const heap = @import("../index/heap.zig");

pub const default_k1: f32 = 1.2;
pub const default_b: f32 = 0.75;

const Posting = struct { point: u32, tf: u32, gen: u32 };

const TermCount = struct { term: u32, tf: u32 };

const Doc = struct {
    /// Bumped whenever the point's document changes, so its old postings are
    /// recognisable as stale.
    gen: u32 = 0,
    live: bool = false,
    len: u32 = 0,
    terms: []TermCount = &.{},
};

/// A query worker's reusable state for `TextIndex.search`, so that a search
/// allocates nothing once its buffers have grown to the collection (§6.3).
pub const SearchScratch = struct {
    gpa: std.mem.Allocator,
    /// Per point, the score accumulated so far. A point holding a query term
    /// scores above zero (exact statistics keep `df <= N`, so `idf > 0`), so
    /// zero means untouched, and every touched entry is zeroed again after.
    acc: []f32 = &.{},
    touched: std.ArrayList(u32) = .empty,
    terms: std.ArrayList(u32) = .empty,
    lowered: std.ArrayList(u8) = .empty,
    stemmed: std.ArrayList(u8) = .empty,

    pub fn init(gpa: std.mem.Allocator) SearchScratch {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *SearchScratch) void {
        self.gpa.free(self.acc);
        self.touched.deinit(self.gpa);
        self.terms.deinit(self.gpa);
        self.lowered.deinit(self.gpa);
        self.stemmed.deinit(self.gpa);
    }

    fn cover(self: *SearchScratch, points: usize) !void {
        if (self.acc.len >= points) return;
        const grown = try self.gpa.alloc(f32, @max(points, self.acc.len * 2));
        @memset(grown, 0);
        self.gpa.free(self.acc);
        self.acc = grown;
    }
};

pub const TextIndex = struct {
    gpa: std.mem.Allocator,
    opts: tokenizer.Options,
    term_ids: std.StringHashMapUnmanaged(u32) = .empty,
    postings: std.ArrayList(std.ArrayList(Posting)) = .empty,
    df: std.ArrayList(u32) = .empty,
    docs: std.ArrayList(Doc) = .empty,
    documents: u32 = 0,
    total_tokens: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, opts: tokenizer.Options) TextIndex {
        return .{ .gpa = gpa, .opts = opts };
    }

    pub fn deinit(self: *TextIndex) void {
        var it = self.term_ids.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.term_ids.deinit(self.gpa);
        for (self.postings.items) |*p| p.deinit(self.gpa);
        self.postings.deinit(self.gpa);
        self.df.deinit(self.gpa);
        for (self.docs.items) |d| self.gpa.free(d.terms);
        self.docs.deinit(self.gpa);
    }

    fn termId(self: *TextIndex, token: []const u8) !u32 {
        const gop = try self.term_ids.getOrPut(self.gpa, token);
        if (gop.found_existing) return gop.value_ptr.*;
        errdefer self.term_ids.removeByPtr(gop.key_ptr);
        const id: u32 = @intCast(self.postings.items.len);
        try self.postings.append(self.gpa, .empty);
        errdefer _ = self.postings.pop();
        try self.df.append(self.gpa, 0);
        errdefer _ = self.df.pop();
        // The map borrows the key: own it before anything can look it up.
        gop.key_ptr.* = try self.gpa.dupe(u8, token);
        gop.value_ptr.* = id;
        return id;
    }

    /// Drop `point`'s document from the statistics, if it has one.
    pub fn remove(self: *TextIndex, point: u32) void {
        if (point >= self.docs.items.len) return;
        const d = &self.docs.items[point];
        if (!d.live) return;
        for (d.terms) |tc| self.df.items[tc.term] -= 1;
        self.documents -= 1;
        self.total_tokens -= d.len;
        self.gpa.free(d.terms);
        d.* = .{ .gen = d.gen +% 1 };
    }

    /// Replace `point`'s document with the tokens of `values`: one string, an
    /// array's elements, or none.
    pub fn set(self: *TextIndex, point: u32, values: []const []const u8) !void {
        self.remove(point);
        if (point >= self.docs.items.len) {
            try self.docs.appendNTimes(self.gpa, .{}, point + 1 - self.docs.items.len);
        }
        var counts: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
        defer counts.deinit(self.gpa);
        var len: u32 = 0;
        for (values) |value| {
            var it = tokenizer.TokenIterator.init(self.gpa, self.opts, value);
            defer it.deinit();
            while (try it.next()) |token| {
                const id = try self.termId(token);
                const gop = try counts.getOrPut(self.gpa, id);
                gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
                len += 1;
            }
        }
        // Qdrant's rule: a boundary token between an array's values makes it a
        // (possibly empty) document.
        if (len == 0 and values.len < 2) return;
        const terms = try self.gpa.alloc(TermCount, counts.count());
        errdefer self.gpa.free(terms);
        const d = &self.docs.items[point];
        for (counts.keys(), counts.values(), 0..) |term, tf, i| {
            terms[i] = .{ .term = term, .tf = tf };
            try self.postings.items[term].append(self.gpa, .{ .point = point, .tf = tf, .gen = d.gen });
        }
        for (terms) |tc| self.df.items[tc.term] += 1;
        d.* = .{ .gen = d.gen, .live = true, .len = len, .terms = terms };
        self.documents += 1;
        self.total_tokens += len;
    }

    pub fn avgdl(self: *const TextIndex) ?f32 {
        if (self.documents == 0) return null;
        return @as(f32, @floatFromInt(self.total_tokens)) / @as(f32, @floatFromInt(self.documents));
    }

    fn idf(self: *const TextIndex, term: u32) f32 {
        const n: f32 = @floatFromInt(self.documents);
        const df: f32 = @floatFromInt(self.df.items[term]);
        return @max(@log((n - df + 0.5) / (df + 0.5) + 1.0), 0.0);
    }

    /// The best points for `query` into `top`, which orders them by score and
    /// then by point. A point counts only if `allowed` admits it (when given);
    /// the statistics stay the whole index's.
    pub fn search(
        self: *const TextIndex,
        scratch: *SearchScratch,
        query: []const u8,
        k1: f32,
        b: f32,
        top: *heap.TopK,
        allowed: anytype,
    ) !void {
        // The query's distinct terms the index holds; the others score nothing.
        scratch.terms.clearRetainingCapacity();
        {
            var it = tokenizer.TokenIterator.init(scratch.gpa, self.opts, query);
            it.lowered = scratch.lowered;
            it.stemmed = scratch.stemmed;
            defer {
                scratch.lowered = it.lowered;
                scratch.stemmed = it.stemmed;
            }
            while (try it.next()) |token| {
                const id = self.term_ids.get(token) orelse continue;
                if (std.mem.indexOfScalar(u32, scratch.terms.items, id) == null) {
                    try scratch.terms.append(scratch.gpa, id);
                }
            }
        }
        const avg = self.avgdl() orelse return;
        try scratch.cover(self.docs.items.len);
        scratch.touched.clearRetainingCapacity();
        for (scratch.terms.items) |term| {
            const term_idf = self.idf(term);
            for (self.postings.items[term].items) |p| {
                const d = self.docs.items[p.point];
                if (!d.live or d.gen != p.gen) continue;
                if (@TypeOf(allowed) != @TypeOf(null) and !allowed.admits(p.point)) continue;
                const tf: f32 = @floatFromInt(p.tf);
                const len: f32 = @floatFromInt(d.len);
                const norm = if (b > 0) k1 * (1 - b + b * len / avg) else k1;
                const s = term_idf * tf * (k1 + 1) / (tf + norm);
                if (scratch.acc[p.point] == 0) try scratch.touched.append(scratch.gpa, p.point);
                scratch.acc[p.point] += s;
            }
        }
        for (scratch.touched.items) |point| {
            top.push(.{ .id = point, .score = scratch.acc[point] });
            scratch.acc[point] = 0;
        }
    }
};

const testing = std.testing;

fn texts(comptime xs: []const []const u8) []const []const u8 {
    return xs;
}

var test_results: [16]heap.Candidate = undefined;

/// `search` at the defaults into a fresh heap of ten.
fn testSearch(idx: *const TextIndex, query: []const u8, allowed: anytype) ![]heap.Candidate {
    var scratch = SearchScratch.init(testing.allocator);
    defer scratch.deinit();
    var top = heap.TopK.init(&test_results, 10);
    try idx.search(&scratch, query, default_k1, default_b, &top, allowed);
    return top.finish();
}

test "statistics are exact through overwrite and delete" {
    var idx = TextIndex.init(testing.allocator, .{});
    defer idx.deinit();
    try idx.set(0, texts(&.{"alpha beta"}));
    try idx.set(1, texts(&.{"alpha alpha gamma delta"}));
    try idx.set(2, texts(&.{ "beta", "gamma gamma" }));
    try testing.expectEqual(@as(u32, 3), idx.documents);
    try testing.expectEqual(@as(?f32, 3.0), idx.avgdl());
    // Overwrite: point 1 now holds one token.
    try idx.set(1, texts(&.{"delta"}));
    try testing.expectEqual(@as(u32, 3), idx.documents);
    try testing.expectEqual(@as(u64, 2 + 1 + 3), idx.total_tokens);
    try testing.expectEqual(@as(u32, 1), idx.df.items[idx.term_ids.get("alpha").?]);
    // Delete: point 0 leaves N, df and the total.
    idx.remove(0);
    try testing.expectEqual(@as(u32, 2), idx.documents);
    try testing.expectEqual(@as(u32, 0), idx.df.items[idx.term_ids.get("alpha").?]);
    // Its stale postings never come back.
    try testing.expectEqual(@as(usize, 0), (try testSearch(&idx, "alpha", null)).len);
}

test "an array of two values is a document even without tokens, as in Qdrant" {
    var idx = TextIndex.init(testing.allocator, .{ .english_stopwords = true });
    defer idx.deinit();
    try idx.set(0, texts(&.{"alpha"}));
    try idx.set(1, texts(&.{ "", "the" }));
    try idx.set(2, texts(&.{"the"}));
    try idx.set(3, texts(&.{""}));
    try testing.expectEqual(@as(u32, 2), idx.documents);
    try testing.expectEqual(@as(?f32, 0.5), idx.avgdl());
}

test "scores are Lucene BM25, and a filter narrows without moving statistics" {
    var idx = TextIndex.init(testing.allocator, .{});
    defer idx.deinit();
    try idx.set(0, texts(&.{"alpha beta"}));
    try idx.set(1, texts(&.{"gamma"}));
    // Qdrant scores "alpha" on this corpus at 0.609970 (qdrant/qdrant#11010).
    const hits = try testSearch(&idx, "alpha alpha", null);
    try testing.expectEqual(@as(usize, 1), hits.len);
    try testing.expectApproxEqRel(@as(f32, 0.609970), hits[0].score, 1e-5);
    const score = hits[0].score;
    const Only = struct {
        p: u32,
        fn admits(self: @This(), point: u32) bool {
            return point == self.p;
        }
    };
    try testing.expectEqual(@as(usize, 0), (try testSearch(&idx, "alpha", Only{ .p = 1 })).len);
    const kept = try testSearch(&idx, "alpha", Only{ .p = 0 });
    try testing.expectEqual(score, kept[0].score);
}
