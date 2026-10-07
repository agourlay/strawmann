//! Qdrant's `word` tokenizer and token pipeline, the subset decisions.md
//! (2026-10-07) puts in scope: split on every character that is not
//! alphanumeric, then lowercase, then drop English stopwords (checked on the
//! lowercased token, before stemming), then stem with the English Snowball
//! stemmer (`full_text_index/tokenizers/{mod,tokens_processor}.rs` at
//! 850859ec9).
//!
//! "Alphanumeric" and "lowercase" are Rust's (`char::is_alphanumeric`,
//! `str::to_lowercase`, final sigma included), from `unicode_tables.zig`,
//! generated from the standard library of the rustc that builds the Qdrant
//! under comparison.

const std = @import("std");
const tables = @import("unicode_tables.zig");
const stopwords_english = @import("stopwords_english.zig");
const snowball = @import("snowball.zig");
const stem_english = @import("stem_english.zig");

pub const Options = struct {
    /// Qdrant's default is on.
    lowercase: bool = true,
    english_stopwords: bool = false,
    english_stemmer: bool = false,
};

const stopword_set = std.StaticStringMap(void).initComptime(blk: {
    @setEvalBranchQuota(100_000);
    var kvs: [stopwords_english.english.len]struct { []const u8 } = undefined;
    for (stopwords_english.english, 0..) |w, i| {
        // Qdrant lowercases the list for a lowercasing index; it is lowercase
        // already, so one set serves both.
        for (w) |ch| std.debug.assert(!std.ascii.isUpper(ch));
        kvs[i] = .{w};
    }
    break :blk kvs;
});

fn inRanges(ranges: []const [2]u21, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid][0]) {
            hi = mid;
        } else if (cp > ranges[mid][1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

/// `char::is_alphanumeric`.
pub fn isAlphanumeric(cp: u21) bool {
    if (cp < 0x80) return std.ascii.isAlphanumeric(@intCast(cp));
    return inRanges(&tables.alphanumeric, cp);
}

fn lowerOf(cp: u21) ?tables.Lower {
    var lo: usize = 0;
    var hi: usize = tables.lowercase.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const e = tables.lowercase[mid];
        if (cp < e.cp) {
            hi = mid;
        } else if (cp > e.cp) {
            lo = mid + 1;
        } else {
            return e;
        }
    }
    return null;
}

/// The code point that ends at byte `end` of `s`, and where it starts.
fn decodeBefore(s: []const u8, end: usize) struct { cp: u21, start: usize } {
    var start = end - 1;
    while (start > 0 and (s[start] & 0xC0) == 0x80) start -= 1;
    return .{ .cp = std.unicode.utf8Decode(s[start..end]) catch 0xFFFD, .start = start };
}

/// Rust's `case_ignorable_then_cased`: skip what final sigma skips, then is
/// the next character cased? `backward` walks from `at` to the start of `s`.
fn ignorableThenCased(s: []const u8, at: usize, backward: bool) bool {
    var i = at;
    while (true) {
        var cp: u21 = undefined;
        if (backward) {
            if (i == 0) return false;
            const d = decodeBefore(s, i);
            cp = d.cp;
            i = d.start;
        } else {
            if (i >= s.len) return false;
            const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return false;
            cp = std.unicode.utf8Decode(s[i..][0..n]) catch return false;
            i += n;
        }
        if (inRanges(&tables.sigma_skip, cp)) continue;
        return inRanges(&tables.sigma_cased, cp);
    }
}

/// `str::to_lowercase` of `s` (valid UTF-8), appended to `out`.
pub fn appendLowercase(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    var i: usize = 0;
    while (i < s.len) {
        const b = s[i];
        if (b < 0x80) {
            try out.append(gpa, std.ascii.toLower(b));
            i += 1;
            continue;
        }
        const n = try std.unicode.utf8ByteSequenceLength(b);
        const cp = try std.unicode.utf8Decode(s[i..][0..n]);
        if (cp == 0x3A3) {
            // `map_uppercase_sigma`: final when a cased character precedes it
            // and none follows, across what final sigma skips.
            const final = ignorableThenCased(s, i, true) and !ignorableThenCased(s, i + n, false);
            try appendCodePoint(gpa, out, if (final) 0x3C2 else 0x3C3);
        } else if (lowerOf(cp)) |e| {
            for (e.to[0..e.len]) |t| try appendCodePoint(gpa, out, t);
        } else {
            try out.appendSlice(gpa, s[i..][0..n]);
        }
        i += n;
    }
}

fn appendCodePoint(gpa: std.mem.Allocator, out: *std.ArrayList(u8), cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
    try out.appendSlice(gpa, buf[0..n]);
}

/// The tokens of one text value, in order and with repeats, each valid until
/// the next call to `next`.
pub const TokenIterator = struct {
    gpa: std.mem.Allocator,
    opts: Options,
    text: []const u8,
    pos: usize = 0,
    lowered: std.ArrayList(u8) = .empty,
    stemmed: std.ArrayList(u8) = .empty,

    pub fn init(gpa: std.mem.Allocator, opts: Options, text: []const u8) TokenIterator {
        return .{ .gpa = gpa, .opts = opts, .text = text };
    }

    pub fn deinit(self: *TokenIterator) void {
        self.lowered.deinit(self.gpa);
        self.stemmed.deinit(self.gpa);
    }

    /// The next maximal run of alphanumeric characters, as bytes of `text`.
    fn nextPiece(self: *TokenIterator) !?[]const u8 {
        const s = self.text;
        // Skip separators.
        while (self.pos < s.len) {
            const n = try std.unicode.utf8ByteSequenceLength(s[self.pos]);
            const cp = try std.unicode.utf8Decode(s[self.pos..][0..n]);
            if (isAlphanumeric(cp)) break;
            self.pos += n;
        }
        if (self.pos >= s.len) return null;
        const start = self.pos;
        while (self.pos < s.len) {
            const n = try std.unicode.utf8ByteSequenceLength(s[self.pos]);
            const cp = try std.unicode.utf8Decode(s[self.pos..][0..n]);
            if (!isAlphanumeric(cp)) break;
            self.pos += n;
        }
        return s[start..self.pos];
    }

    pub fn next(self: *TokenIterator) !?[]const u8 {
        while (try self.nextPiece()) |piece| {
            var token: []const u8 = piece;
            if (self.opts.lowercase) {
                self.lowered.clearRetainingCapacity();
                try appendLowercase(self.gpa, &self.lowered, piece);
                token = self.lowered.items;
            }
            if (self.opts.english_stopwords and stopword_set.has(token)) continue;
            if (self.opts.english_stemmer) {
                try self.stemmed.resize(self.gpa, token.len + snowball.capacity_slack);
                var env = snowball.Env.init(self.stemmed.items, token);
                _ = stem_english.stem(&env);
                token = env.current();
            }
            return token;
        }
        return null;
    }
};

const testing = std.testing;

fn expectTokens(opts: Options, text: []const u8, want: []const []const u8) !void {
    var it = TokenIterator.init(testing.allocator, opts, text);
    defer it.deinit();
    var i: usize = 0;
    while (try it.next()) |t| : (i += 1) {
        try testing.expect(i < want.len);
        try testing.expectEqualStrings(want[i], t);
    }
    try testing.expectEqual(want.len, i);
}

test "the word tokenizer splits on anything not alphanumeric" {
    try expectTokens(.{}, "Hello, World! foo_bar x42 -- naïve café", &.{
        "hello", "world", "foo", "bar", "x42",
        "naïve",
        "café",
    });
    try expectTokens(.{}, "日本語 Ⅻ", &.{ "日本語", "ⅻ" });
    try expectTokens(.{ .lowercase = false }, "Hello hello", &.{ "Hello", "hello" });
    try expectTokens(.{}, "", &.{});
    try expectTokens(.{}, " ,;- ", &.{});
}

test "lowercasing is Rust's, final sigma and expanding mappings included" {
    // Σ is final after a cased letter with none following, as `str::to_lowercase` has it.
    try expectTokens(.{}, "ΑΣ ΑΣΑ Σ", &.{ "ας", "ασα", "σ" });
    // İ lowercases to two code points.
    try expectTokens(.{}, "İ", &.{"i\u{307}"});
}

test "stopwords are dropped before stemming" {
    try expectTokens(.{ .english_stopwords = true, .english_stemmer = true }, "The runners were running to THE dogs", &.{
        "runner", "run", "dog",
    });
}

/// Text values with the oracle's tokens for them (`conformance token-sample`,
/// lowercase only): `testdata/token_edge.txt`'s edge cases, then 600 values
/// of scifact and fiqa that hold a non-ASCII character, cut at 240 characters.
const token_sample = @embedFile("testdata/token_sample.jsonl");

/// Every line of `sample` the tokenizer splits or lowercases differently.
fn tokenMismatches(gpa: std.mem.Allocator, opts: Options, sample: []const u8) !usize {
    const Line = struct { text: []const u8, tokens: []const []const u8 };
    var mismatches: usize = 0;
    var lines = std.mem.splitScalar(u8, sample, '\n');
    while (lines.next()) |raw| {
        if (raw.len == 0) continue;
        const parsed = try std.json.parseFromSlice(Line, gpa, raw, .{});
        defer parsed.deinit();
        var it = TokenIterator.init(gpa, opts, parsed.value.text);
        defer it.deinit();
        var i: usize = 0;
        var same = true;
        while (try it.next()) |tok| : (i += 1) {
            if (i >= parsed.value.tokens.len or !std.mem.eql(u8, tok, parsed.value.tokens[i])) same = false;
        }
        if (i != parsed.value.tokens.len) same = false;
        if (!same) {
            if (mismatches < 10) std.debug.print("tokens differ for {s}\n", .{parsed.value.text});
            mismatches += 1;
        }
    }
    return mismatches;
}

test "the tokenizer agrees with the oracle on Unicode text" {
    try testing.expectEqual(@as(usize, 0), try tokenMismatches(testing.allocator, .{}, token_sample));
}
