//! Top-k result collection.
//!
//! §6.5: "Candidate and result heaps: flat arrays, `ef`-bounded, branch-light
//! sift. At `ef ≤ 512` the heaps live comfortably in L1."
//!
//! §6.3: "No allocation on the query path. Per-worker preallocated: visited
//! set, candidate heap, result heap, rescore buffer, distance scratch." So the
//! heap borrows its storage and never grows; the caller sizes it from `ef` once
//! per worker.
//!
//! ## The ordering rule
//!
//! §8.7: "**Total order on results:** score descending, then internal ID
//! ascending. Our own output is then stable even under ties. We do *not*
//! require Qdrant to match that order, hence T2."
//!
//! That total order is what makes §8.7's determinism claim reachable. Ties are
//! not rare, §8.5's T2 tier exists because "Equal scores occur constantly
//! (duplicate vectors, quantized encodings, small dims)", and a heap that
//! breaks them by insertion order produces thread-count-dependent output, which
//! would make the differ unable to distinguish "we disagree with Qdrant" from
//! "we disagree with ourselves".
//!
//! Every kernel in `dist/` returns an *internal similarity* where higher is
//! better, including the negated distances of §8.3. So this file only ever
//! implements one comparison direction, and no caller has to know which metric
//! produced the score.

const std = @import("std");

pub const Candidate = struct {
    /// Internal offset (§3: "the currency of the entire engine").
    id: u32,
    /// Internal similarity, higher is better.
    score: f32,

    /// The §8.7 total order: score descending, then id ascending.
    ///
    /// Returns true when `a` should rank before `b`. Scores are assumed
    /// finite: a NaN compares unequal to everything, so it would fall through
    /// to the id tie-break and corrupt the heap order. The API layer rejects
    /// non-finite vectors before a score can be produced from them.
    pub fn better(a: Candidate, b: Candidate) bool {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    }

    /// Strictly worse than `b` under the same total order.
    pub fn worse(a: Candidate, b: Candidate) bool {
        return better(b, a);
    }
};

/// A bounded max-`k` collection: keeps the `k` best candidates seen.
///
/// Implemented as a **min-heap of the survivors**, so the element easiest to
/// reach is the worst one, which is the one a new candidate must beat. That
/// makes the reject path (the common case, since most candidates are worse than
/// the current k-th) a single comparison against `peekWorst`.
pub const TopK = struct {
    items: []Candidate,
    len: usize = 0,
    /// Capacity actually in use, which may be below `items.len` when one
    /// preallocated buffer serves several `k` values across queries.
    k: usize,

    pub fn init(storage: []Candidate, k: usize) TopK {
        // `k` past the storage means `push` writes past the buffer. The heaps
        // are preallocated per §6.3, so this is the boundary between a client's
        // `limit` and our own memory.
        std.debug.assert(k <= storage.len);
        return .{ .items = storage, .k = k };
    }

    pub fn reset(self: *TopK, k: usize) void {
        std.debug.assert(k <= self.items.len);
        self.len = 0;
        self.k = k;
    }

    pub fn isFull(self: *const TopK) bool {
        return self.len >= self.k;
    }

    /// The worst survivor. Only valid when full; this is the early-termination
    /// threshold of §6.5 ("worst candidate is worse than current k-th best").
    pub fn peekWorst(self: *const TopK) Candidate {
        std.debug.assert(self.len > 0);
        return self.items[0];
    }

    /// Would this candidate be accepted? Pure, so the caller can test before
    /// paying for a full distance evaluation in a rescore loop.
    pub fn wouldAccept(self: *const TopK, c: Candidate) bool {
        if (self.len < self.k) return true;
        return c.better(self.items[0]);
    }

    pub fn push(self: *TopK, c: Candidate) void {
        if (self.len < self.k) {
            self.items[self.len] = c;
            self.len += 1;
            self.siftUp(self.len - 1);
            return;
        }
        if (self.k == 0) return;
        // Full: replace the worst if this beats it, then restore the heap.
        if (!c.better(self.items[0])) return;
        self.items[0] = c;
        self.siftDown(0);
    }

    /// Min-heap on `better`: the root is the *worst* survivor.
    fn siftUp(self: *TopK, start: usize) void {
        var i = start;
        while (i > 0) {
            const parent = (i - 1) / 2;
            // Parent must be worse than the child in a min-heap of quality.
            if (self.items[parent].worse(self.items[i])) break;
            std.mem.swap(Candidate, &self.items[parent], &self.items[i]);
            i = parent;
        }
    }

    fn siftDown(self: *TopK, start: usize) void {
        var i = start;
        while (true) {
            const l = 2 * i + 1;
            const r = l + 1;
            var worst = i;
            if (l < self.len and self.items[l].worse(self.items[worst])) worst = l;
            if (r < self.len and self.items[r].worse(self.items[worst])) worst = r;
            if (worst == i) return;
            std.mem.swap(Candidate, &self.items[i], &self.items[worst]);
            i = worst;
        }
    }

    /// Sort the survivors best-first, in place, and return them.
    ///
    /// After this the structure is no longer a heap, it is a finished result
    /// list, which is all the caller wants at that point.
    pub fn finish(self: *TopK) []Candidate {
        const out = self.items[0..self.len];
        std.mem.sort(Candidate, out, {}, struct {
            fn lt(_: void, a: Candidate, b: Candidate) bool {
                return a.better(b);
            }
        }.lt);
        return out;
    }
};

/// A min-heap used as the HNSW candidate frontier.
///
/// Distinct from `TopK` because it is unbounded-in-quality but bounded-in-size,
/// and because the traversal pops the *best* candidate to expand next while
/// `TopK` discards the worst. Merging them into one type would need a
/// comparison-direction parameter on every operation for no gain.
pub const Frontier = struct {
    items: []Candidate,
    len: usize = 0,

    pub fn init(storage: []Candidate) Frontier {
        return .{ .items = storage };
    }

    pub fn reset(self: *Frontier) void {
        self.len = 0;
    }

    pub fn isEmpty(self: *const Frontier) bool {
        return self.len == 0;
    }

    /// Push. When the frontier is full, the **worst** candidate makes way, and
    /// only if the newcomer beats it.
    ///
    /// This used to drop whatever arrived last, justified as "a dropped
    /// candidate is by construction worse than `ef` others, so it could not
    /// have entered the result set". That is not true of a heap that is merely
    /// *full*: arrival order says nothing about rank, and `searchLayer` only
    /// pushes candidates it has already checked are better than the current
    /// worst result — so every dropped push was a promising candidate thrown
    /// away.
    ///
    /// It was not rare. Building the 50k SIFT graph at `ef_construct = 100`
    /// dropped 178,027 of 11,021,631 pushes, 1.6%. Because the frontier and
    /// the result heap both scale with `ef_construct`, the drop *rate* stayed
    /// put as it was raised — which is why recall was identical at
    /// `ef_construct` 100, 200 and 400. The graph could not improve, because
    /// the same fraction of good candidates was discarded at every setting.
    ///
    /// Still no allocation on the query path (§6.3): the storage is fixed, and
    /// the eviction scan covers the heap's leaf half, which is where a
    /// max-heap's minimum must be.
    pub fn push(self: *Frontier, c: Candidate) void {
        if (self.len == self.items.len) {
            if (self.len == 0) return;
            var worst: usize = self.len / 2;
            var i: usize = worst + 1;
            while (i < self.len) : (i += 1) {
                if (self.items[worst].better(self.items[i])) worst = i;
            }
            if (!c.better(self.items[worst])) return;
            self.items[worst] = c;
            // The replacement beats what it replaced, so it can only move up.
            var j = worst;
            while (j > 0) {
                const parent = (j - 1) / 2;
                if (self.items[parent].better(self.items[j])) break;
                std.mem.swap(Candidate, &self.items[parent], &self.items[j]);
                j = parent;
            }
            return;
        }
        self.items[self.len] = c;
        self.len += 1;
        var i = self.len - 1;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (self.items[parent].better(self.items[i])) break;
            std.mem.swap(Candidate, &self.items[parent], &self.items[i]);
            i = parent;
        }
    }

    /// Remove and return the best candidate.
    pub fn pop(self: *Frontier) ?Candidate {
        if (self.len == 0) return null;
        const top = self.items[0];
        self.len -= 1;
        if (self.len > 0) {
            self.items[0] = self.items[self.len];
            var i: usize = 0;
            while (true) {
                const l = 2 * i + 1;
                const r = l + 1;
                var best = i;
                if (l < self.len and self.items[l].better(self.items[best])) best = l;
                if (r < self.len and self.items[r].better(self.items[best])) best = r;
                if (best == i) break;
                std.mem.swap(Candidate, &self.items[i], &self.items[best]);
                i = best;
            }
        }
        return top;
    }

    pub fn peek(self: *const Frontier) ?Candidate {
        if (self.len == 0) return null;
        return self.items[0];
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "§8.7 total order: score descending, then id ascending" {
    try testing.expect(Candidate.better(.{ .id = 5, .score = 2.0 }, .{ .id = 1, .score = 1.0 }));
    // Tie on score: lower id wins, regardless of insertion order.
    try testing.expect(Candidate.better(.{ .id = 1, .score = 1.0 }, .{ .id = 2, .score = 1.0 }));
    try testing.expect(!Candidate.better(.{ .id = 2, .score = 1.0 }, .{ .id = 1, .score = 1.0 }));
    // Irreflexive.
    try testing.expect(!Candidate.better(.{ .id = 1, .score = 1.0 }, .{ .id = 1, .score = 1.0 }));
}

test "topk keeps the k best and returns them sorted" {
    var storage: [8]Candidate = undefined;
    var h = TopK.init(&storage, 3);

    const input = [_]Candidate{
        .{ .id = 0, .score = 0.5 },
        .{ .id = 1, .score = 0.9 },
        .{ .id = 2, .score = 0.1 },
        .{ .id = 3, .score = 0.7 },
        .{ .id = 4, .score = 0.3 },
    };
    for (input) |c| h.push(c);

    const out = h.finish();
    try testing.expectEqual(@as(usize, 3), out.len);
    try testing.expectEqual(@as(u32, 1), out[0].id);
    try testing.expectEqual(@as(u32, 3), out[1].id);
    try testing.expectEqual(@as(u32, 0), out[2].id);
}

test "topk output is independent of insertion order, including under ties" {
    // §8.7's determinism requirement made concrete: the same set of candidates
    // must produce the same list whatever order they arrive in. Without the
    // id tie-break this fails as soon as two scores are equal.
    var pool: [64]Candidate = undefined;
    for (&pool, 0..) |*c, i| {
        c.* = .{ .id = @intCast(i), .score = @as(f32, @floatFromInt(i % 5)) };
    }

    var storage_a: [10]Candidate = undefined;
    var a = TopK.init(&storage_a, 10);
    for (pool) |c| a.push(c);
    var expect: [10]Candidate = undefined;
    @memcpy(&expect, a.finish());

    var prng = std.Random.DefaultPrng.init(0x70c);
    const rnd = prng.random();
    for (0..50) |_| {
        var shuffled = pool;
        rnd.shuffle(Candidate, &shuffled);
        var storage_b: [10]Candidate = undefined;
        var b = TopK.init(&storage_b, 10);
        for (shuffled) |c| b.push(c);
        const got = b.finish();
        for (expect, got) |e, g| {
            try testing.expectEqual(e.id, g.id);
            try testing.expectEqual(e.score, g.score);
        }
    }
}

test "topk agrees with a full sort" {
    var prng = std.Random.DefaultPrng.init(0x5017);
    const rnd = prng.random();

    for ([_]usize{ 1, 2, 10, 64 }) |k| {
        const n = 500;
        const all = try testing.allocator.alloc(Candidate, n);
        defer testing.allocator.free(all);
        for (all, 0..) |*c, i| c.* = .{ .id = @intCast(i), .score = rnd.float(f32) };

        const storage = try testing.allocator.alloc(Candidate, k);
        defer testing.allocator.free(storage);
        var h = TopK.init(storage, k);
        for (all) |c| h.push(c);
        const got = h.finish();

        const sorted = try testing.allocator.dupe(Candidate, all);
        defer testing.allocator.free(sorted);
        std.mem.sort(Candidate, sorted, {}, struct {
            fn lt(_: void, x: Candidate, y: Candidate) bool {
                return x.better(y);
            }
        }.lt);

        try testing.expectEqual(k, got.len);
        for (0..k) |i| {
            try testing.expectEqual(sorted[i].id, got[i].id);
            try testing.expectEqual(sorted[i].score, got[i].score);
        }
    }
}

test "peekWorst is the early-termination threshold" {
    var storage: [4]Candidate = undefined;
    var h = TopK.init(&storage, 4);
    for ([_]f32{ 0.9, 0.8, 0.7, 0.6 }) |s| {
        h.push(.{ .id = 0, .score = s });
    }
    try testing.expect(h.isFull());
    try testing.expectEqual(@as(f32, 0.6), h.peekWorst().score);

    // Anything worse than the k-th is rejected without disturbing the heap -
    // §6.5's "worst candidate is worse than current k-th best".
    try testing.expect(!h.wouldAccept(.{ .id = 99, .score = 0.5 }));
    h.push(.{ .id = 99, .score = 0.5 });
    try testing.expectEqual(@as(f32, 0.6), h.peekWorst().score);

    try testing.expect(h.wouldAccept(.{ .id = 99, .score = 0.65 }));
    h.push(.{ .id = 99, .score = 0.65 });
    try testing.expectEqual(@as(f32, 0.65), h.peekWorst().score);
}

test "topk with k=0 accepts nothing and does not fault" {
    var storage: [4]Candidate = undefined;
    var h = TopK.init(&storage, 0);
    h.push(.{ .id = 1, .score = 1.0 });
    try testing.expectEqual(@as(usize, 0), h.finish().len);
}

test "topk handles fewer candidates than k" {
    var storage: [10]Candidate = undefined;
    var h = TopK.init(&storage, 10);
    h.push(.{ .id = 2, .score = 0.2 });
    h.push(.{ .id = 1, .score = 0.9 });
    const out = h.finish();
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqual(@as(u32, 1), out[0].id);
}

test "frontier pops best-first" {
    var storage: [16]Candidate = undefined;
    var f = Frontier.init(&storage);
    for ([_]f32{ 0.1, 0.9, 0.5, 0.7, 0.3 }) |s| {
        f.push(.{ .id = 0, .score = s });
    }
    var last: f32 = std.math.inf(f32);
    var n: usize = 0;
    while (f.pop()) |c| {
        try testing.expect(c.score <= last);
        last = c.score;
        n += 1;
    }
    try testing.expectEqual(@as(usize, 5), n);
    try testing.expect(f.isEmpty());
}

test "frontier evicts its worst rather than allocating when full" {
    // §6.3 forbids allocation on the query path. What goes is the *worst*
    // candidate held, by rank; the earlier rule, drop the arrival, assumed
    // arrival order said something about rank, which `Frontier.push`'s doc
    // measures it does not.
    var storage: [4]Candidate = undefined;
    var f = Frontier.init(&storage);
    for (0..10) |i| f.push(.{ .id = @intCast(i), .score = @floatFromInt(i) });
    try testing.expectEqual(@as(usize, 4), f.len);
    // The four it kept are the four *best* (scores 6..9): a full frontier
    // evicts its worst, not the newest, see the eviction test below. The heap
    // property still holds.
    var last: f32 = std.math.inf(f32);
    var popped: usize = 0;
    while (f.pop()) |c| {
        try testing.expect(c.score <= last);
        try testing.expect(c.score >= 6.0);
        last = c.score;
        popped += 1;
    }
    try testing.expectEqual(@as(usize, 4), popped);
}

test "frontier breaks ties by id, like TopK" {
    var storage: [8]Candidate = undefined;
    var f = Frontier.init(&storage);
    f.push(.{ .id = 7, .score = 1.0 });
    f.push(.{ .id = 3, .score = 1.0 });
    f.push(.{ .id = 5, .score = 1.0 });
    try testing.expectEqual(@as(u32, 3), f.pop().?.id);
    try testing.expectEqual(@as(u32, 5), f.pop().?.id);
    try testing.expectEqual(@as(u32, 7), f.pop().?.id);
}

test "a full frontier evicts its worst candidate, not the newest one" {
    // The defect this replaced: `push` returned early when full, so a candidate
    // was discarded for *arriving late* rather than for being poor. Building
    // the 50k SIFT graph dropped 178,027 of 11,021,631 pushes that way, every
    // one of them already checked by `searchLayer` as better than the current
    // worst result (findings 22).
    var storage: [4]Candidate = undefined;
    var f = Frontier.init(&storage);

    for ([_]f32{ 0.5, 0.4, 0.3, 0.2 }) |sc| {
        f.push(.{ .id = @intFromFloat(sc * 10), .score = sc });
    }
    try std.testing.expect(f.len == 4);

    // Full. A better candidate than the worst (0.2) must displace it.
    f.push(.{ .id = 99, .score = 0.9 });
    try std.testing.expect(f.len == 4);
    // And it must come out first, which the old code made impossible: it was
    // never stored at all.
    try std.testing.expectEqual(@as(u32, 99), f.pop().?.id);

    // A candidate worse than everything present is correctly refused.
    var g = Frontier.init(&storage);
    for ([_]f32{ 0.5, 0.4, 0.3, 0.2 }) |sc| {
        g.push(.{ .id = @intFromFloat(sc * 10), .score = sc });
    }
    g.push(.{ .id = 77, .score = 0.01 });
    var seen_77 = false;
    while (g.pop()) |c| {
        if (c.id == 77) seen_77 = true;
    }
    try std.testing.expect(!seen_77);

    // The heap property survives eviction: everything comes out in order.
    var h = Frontier.init(&storage);
    for ([_]f32{ 0.1, 0.9, 0.5, 0.3 }) |sc| {
        h.push(.{ .id = @intFromFloat(sc * 10), .score = sc });
    }
    h.push(.{ .id = 60, .score = 0.6 }); // displaces 0.1
    var last: f32 = std.math.inf(f32);
    while (h.pop()) |c| {
        try std.testing.expect(c.score <= last);
        last = c.score;
    }
}
