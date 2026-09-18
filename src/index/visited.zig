//! The visited set.
//!
//! §6.5: "Visited set: **generation-stamped `u32` array**, sized to point
//! count, with a monotonically incrementing per-query epoch. No clearing
//! between queries, no allocation, no hashing. 4 MB per worker at 1M points -
//! measure against a bitmap-plus-dirty-list alternative, which is smaller but
//! requires clearing."
//!
//! §11 open question 3: "Generation-stamped array vs bitmap-plus-dirty-list for
//! the visited set, at what point count does the 4 MB/worker footprint start
//! hurting?"
//!
//! Both are implemented here behind the same interface so the microbenchmark
//! can answer that question with numbers rather than argument. They are not
//! interchangeable in cost:
//!
//! | | generation-stamped | bitmap + dirty list |
//! |---|---|---|
//! | footprint at 1M pts | 4 MB | 128 KB + dirty list |
//! | per-query reset | free (epoch++) | O(visited) clear |
//! | per-probe cost | 1 load + compare | 1 load + shift + mask |
//! | cache pressure | 4 B/point touched | 1 bit/point touched |
//!
//! The trade is footprint against reset cost, and which side wins depends on
//! the ratio of `Nd` (nodes actually visited, §5.3: expect 1500–4000) to the
//! collection size. At 1M points a query touches ~0.3% of the array, so the
//! bitmap's 32× smaller footprint means 32× fewer cache lines pulled in, but
//! it pays a clear proportional to what it touched. §11's question is exactly
//! where those cross.
//!
//! ## The epoch wraparound
//!
//! A `u32` epoch wraps after 4 billion queries. That is reachable in a long
//! benchmark run, at 40k QPS it is about 30 hours, and a wrap would make
//! stale stamps from the previous cycle read as visited, silently truncating
//! every subsequent search. `beginQuery` handles it by clearing the array on
//! wrap, which costs one memset every 4 billion queries.

const std = @import("std");

/// §6.5's default: a generation-stamped u32 array.
pub const Generation = struct {
    stamps: []u32,
    epoch: u32 = 0,

    pub fn init(alloc: std.mem.Allocator, capacity: usize) !Generation {
        const stamps = try alloc.alloc(u32, capacity);
        @memset(stamps, 0);
        return .{ .stamps = stamps };
    }

    pub fn deinit(self: *Generation, alloc: std.mem.Allocator) void {
        alloc.free(self.stamps);
    }

    /// Start a new query. O(1) except on the wraparound.
    pub fn beginQuery(self: *Generation) void {
        self.epoch +%= 1;
        if (self.epoch == 0) {
            // Wrapped. Every stamp is now potentially a false positive, so the
            // array must be cleared before epoch 0 is reused. Skipping this
            // would truncate results in a way that only appears after billions
            // of queries and would be attributed to anything but its cause.
            @memset(self.stamps, 0);
            self.epoch = 1;
        }
    }

    /// Mark `id` visited. Returns true if it was *not* already visited.
    pub fn testAndSet(self: *Generation, id: u32) bool {
        if (self.stamps[id] == self.epoch) return false;
        self.stamps[id] = self.epoch;
        return true;
    }

    /// Bring `id`'s stamp towards the cache ahead of `testAndSet`. Write
    /// intent, since the common outcome is to stamp it.
    pub inline fn prefetch(self: *const Generation, id: u32) void {
        @prefetch(&self.stamps[id], .{ .rw = .write, .locality = 3, .cache = .data });
    }

    pub fn isVisited(self: *const Generation, id: u32) bool {
        return self.stamps[id] == self.epoch;
    }

    pub fn footprintBytes(capacity: usize) usize {
        return capacity * @sizeOf(u32);
    }
};

/// The alternative §6.5 asks to measure against: a bitmap plus a dirty list.
///
/// The dirty list is what makes the reset proportional to what was visited
/// rather than to the collection size. Without it, clearing a 1M-point bitmap
/// between queries would cost 128 KB of memset per query, far more than the
/// ~2500 bits a query actually sets.
pub const Bitmap = struct {
    bits: []u64,
    /// Indices of words that have any bit set this query.
    dirty: []u32,
    dirty_len: usize = 0,
    /// Set when the dirty list overflowed, in which case the reset falls back
    /// to a full clear. Sized generously, so this is a safety valve rather
    /// than an expected path.
    overflowed: bool = false,

    pub fn init(alloc: std.mem.Allocator, capacity: usize, max_visited: usize) !Bitmap {
        const words = (capacity + 63) / 64;
        const bits = try alloc.alloc(u64, words);
        errdefer alloc.free(bits);
        @memset(bits, 0);
        const dirty = try alloc.alloc(u32, max_visited);
        return .{ .bits = bits, .dirty = dirty };
    }

    pub fn deinit(self: *Bitmap, alloc: std.mem.Allocator) void {
        alloc.free(self.bits);
        alloc.free(self.dirty);
    }

    pub fn beginQuery(self: *Bitmap) void {
        if (self.overflowed) {
            @memset(self.bits, 0);
        } else {
            for (self.dirty[0..self.dirty_len]) |w| self.bits[w] = 0;
        }
        self.dirty_len = 0;
        self.overflowed = false;
    }

    pub fn testAndSet(self: *Bitmap, id: u32) bool {
        const word = id / 64;
        const bit = @as(u64, 1) << @intCast(id % 64);
        const was = self.bits[word];
        if (was & bit != 0) return false;
        // Record the word the first time any of its bits is set. Tracking words
        // rather than ids keeps the list 64× shorter in the common case where a
        // traversal visits clustered ids.
        if (was == 0) {
            if (self.dirty_len < self.dirty.len) {
                self.dirty[self.dirty_len] = word;
                self.dirty_len += 1;
            } else {
                self.overflowed = true;
            }
        }
        self.bits[word] = was | bit;
        return true;
    }

    pub fn isVisited(self: *const Bitmap, id: u32) bool {
        return self.bits[id / 64] & (@as(u64, 1) << @intCast(id % 64)) != 0;
    }

    pub fn footprintBytes(capacity: usize, max_visited: usize) usize {
        return ((capacity + 63) / 64) * @sizeOf(u64) + max_visited * @sizeOf(u32);
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "generation: first visit returns true, repeats false" {
    var v = try Generation.init(testing.allocator, 128);
    defer v.deinit(testing.allocator);

    v.beginQuery();
    try testing.expect(v.testAndSet(5));
    try testing.expect(!v.testAndSet(5));
    try testing.expect(v.isVisited(5));
    try testing.expect(!v.isVisited(6));
}

test "generation: a new query forgets the previous one without clearing" {
    var v = try Generation.init(testing.allocator, 128);
    defer v.deinit(testing.allocator);

    v.beginQuery();
    _ = v.testAndSet(5);
    v.beginQuery();
    try testing.expect(!v.isVisited(5));
    try testing.expect(v.testAndSet(5));
}

test "generation: epoch wraparound does not produce false positives" {
    // The bug this guards against appears only after 2^32 queries and would
    // silently truncate every search thereafter.
    var v = try Generation.init(testing.allocator, 8);
    defer v.deinit(testing.allocator);

    // Drive the epoch to just below the wrap and stamp a node.
    v.epoch = std.math.maxInt(u32) - 1;
    v.beginQuery(); // epoch = maxInt
    try testing.expect(v.testAndSet(3));
    try testing.expectEqual(std.math.maxInt(u32), v.stamps[3]);

    // The next query wraps. Without the clear, epoch would become 0 and every
    // never-visited node (stamp 0) would read as visited.
    v.beginQuery();
    try testing.expectEqual(@as(u32, 1), v.epoch);
    for (0..8) |i| {
        try testing.expect(!v.isVisited(@intCast(i)));
        try testing.expect(v.testAndSet(@intCast(i)));
    }
}

test "generation: node 0 is not confused with 'never visited'" {
    // Stamps start at 0 and epochs start at 1, so a fresh array reads as
    // unvisited for every node including node 0.
    var v = try Generation.init(testing.allocator, 4);
    defer v.deinit(testing.allocator);
    v.beginQuery();
    try testing.expectEqual(@as(u32, 1), v.epoch);
    try testing.expect(!v.isVisited(0));
    try testing.expect(v.testAndSet(0));
}

test "bitmap: matches the generation-stamped behaviour exactly" {
    // Both implementations must be interchangeable, or the §11 comparison
    // measures two different algorithms rather than two data structures.
    const n = 4096;
    var g = try Generation.init(testing.allocator, n);
    defer g.deinit(testing.allocator);
    var b = try Bitmap.init(testing.allocator, n, 1024);
    defer b.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(0x1517);
    const rnd = prng.random();

    for (0..50) |_| {
        g.beginQuery();
        b.beginQuery();
        for (0..500) |_| {
            const id = rnd.uintLessThan(u32, n);
            try testing.expectEqual(g.testAndSet(id), b.testAndSet(id));
            try testing.expectEqual(g.isVisited(id), b.isVisited(id));
        }
    }
}

test "bitmap: dirty-list reset clears only what was touched" {
    var b = try Bitmap.init(testing.allocator, 1 << 16, 64);
    defer b.deinit(testing.allocator);

    b.beginQuery();
    _ = b.testAndSet(10);
    _ = b.testAndSet(11); // same word, must not be listed twice
    _ = b.testAndSet(5000);
    try testing.expectEqual(@as(usize, 2), b.dirty_len);
    try testing.expect(!b.overflowed);

    b.beginQuery();
    try testing.expect(!b.isVisited(10));
    try testing.expect(!b.isVisited(11));
    try testing.expect(!b.isVisited(5000));
}

test "bitmap: dirty-list overflow falls back to a full clear, still correct" {
    // The safety valve. Correctness must not depend on the list being big
    // enough, only performance.
    var b = try Bitmap.init(testing.allocator, 1 << 14, 4);
    defer b.deinit(testing.allocator);

    b.beginQuery();
    // Touch 20 distinct words with a list sized for 4.
    for (0..20) |i| _ = b.testAndSet(@intCast(i * 64));
    try testing.expect(b.overflowed);

    b.beginQuery();
    for (0..20) |i| try testing.expect(!b.isVisited(@intCast(i * 64)));
}

test "§11 open question 3: the footprints the comparison is about" {
    // 1M points, as §6.5 quotes.
    const n = 1_000_000;
    const gen = Generation.footprintBytes(n);
    const bmp = Bitmap.footprintBytes(n, 4000);

    // §6.5: "4 MB per worker at 1M points".
    try testing.expectEqual(@as(usize, 4_000_000), gen);
    // The bitmap is ~32x smaller even counting a generous dirty list.
    try testing.expect(bmp * 20 < gen);
}
