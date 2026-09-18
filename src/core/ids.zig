//! §3, external point IDs to dense internal offsets.
//!
//!   "**Point**: external ID (u64 or u128 UUID) → internal `u32` offset,
//!    assigned densely and monotonically. Internal offsets are the currency of
//!    the entire engine; nothing below the API layer sees external IDs."
//!
//! Two structures, matching §6.4's file layout:
//!
//!   `ids.bin`   internal u32 → external id      (a flat array)
//!   `idmap.bin` external id → internal u32      (open-addressed table, rebuildable)
//!
//! §2 constrains the behaviour in two ways that are easy to miss:
//!
//!   "**Point IDs** may be `num` (u64) or `uuid` (string), `--uuids` switches.
//!    Both must map to a dense internal `u32` offset."
//!
//!   "**`--max-id`** means upserts overwrite existing IDs. The ID map must
//!    handle update-in-place."
//!
//! The second is why `insert` returns whether the ID was new: an upsert of an
//! existing ID must reuse its offset and overwrite the vector in place, not
//! append a duplicate. Appending would inflate `points_count`, corrupt recall
//! against ground truth (§4.3 requires point ID to equal base-file row index),
//! and quietly change what W1 measures.

const std = @import("std");

/// An external point ID.
///
/// UUIDs are stored as the u128 they parse to rather than as a string. A UUID
/// is 16 bytes of entropy; keeping it as 36 bytes of text would triple the
/// table's footprint and make comparison a memcmp instead of a register
/// compare. The textual form is reconstructed on the way out.
pub const ExternalId = union(enum) {
    num: u64,
    uuid: u128,

    pub fn hash(self: ExternalId) u64 {
        // Both variants go through the same finaliser so that a `num` and a
        // `uuid` that happen to share bits still land in different buckets:
        // the tag participates in the hash.
        return switch (self) {
            .num => |v| mix(v ^ 0x9e3779b97f4a7c15),
            .uuid => |v| mix(@as(u64, @truncate(v)) ^ @as(u64, @truncate(v >> 64)) ^ 0xc2b2ae3d27d4eb4f),
        };
    }

    pub fn eql(a: ExternalId, b: ExternalId) bool {
        return switch (a) {
            .num => |x| switch (b) {
                .num => |y| x == y,
                .uuid => false,
            },
            .uuid => |x| switch (b) {
                .num => false,
                .uuid => |y| x == y,
            },
        };
    }

    /// Parse the canonical 8-4-4-4-12 hyphenated form.
    ///
    /// Rejects anything else rather than guessing: a malformed UUID that
    /// silently hashed to *something* would produce a collection whose IDs do
    /// not round-trip, and §8.5's T0 tier compares ID variants on the wire.
    pub fn parseUuid(text: []const u8) ?ExternalId {
        if (text.len != 36) return null;
        if (text[8] != '-' or text[13] != '-' or text[18] != '-' or text[23] != '-') return null;
        var v: u128 = 0;
        for (text, 0..) |ch, i| {
            if (i == 8 or i == 13 or i == 18 or i == 23) continue;
            const nib: u8 = switch (ch) {
                '0'...'9' => ch - '0',
                'a'...'f' => ch - 'a' + 10,
                'A'...'F' => ch - 'A' + 10,
                else => return null,
            };
            v = (v << 4) | nib;
        }
        return .{ .uuid = v };
    }

    /// Render back to the canonical hyphenated form, lowercase.
    pub fn formatUuid(self: ExternalId, out: *[36]u8) void {
        const hex = "0123456789abcdef";
        const v = switch (self) {
            .uuid => |x| x,
            .num => |x| @as(u128, x),
        };
        var i: usize = 0;
        var nibble: u7 = 32;
        while (i < 36) {
            if (i == 8 or i == 13 or i == 18 or i == 23) {
                out[i] = '-';
                i += 1;
                continue;
            }
            nibble -= 1;
            out[i] = hex[@as(u4, @truncate(v >> (@as(u7, nibble) * 4)))];
            i += 1;
        }
    }
};

fn mix(x: u64) u64 {
    var h = x;
    h ^= h >> 33;
    h *%= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *%= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
}

/// Sentinel for an empty slot. `maxInt(u32)` rather than 0 so that internal
/// offset 0, a perfectly ordinary point, is not mistaken for "absent".
pub const empty_slot: u32 = std.math.maxInt(u32);

/// Open-addressed external→internal map with linear probing.
///
/// §6.4: "idmap.bin  external id → internal u32 (open-addressed table,
/// rebuildable)". Rebuildable is the important word: the map holds no
/// information that `ids.bin` does not, so a corrupt or absent map costs a
/// linear rebuild rather than the collection.
///
/// Capacity is preallocated per §3 ("Capacity is preallocated... and never
/// grown mid-run. Growth policy is a benchmark artefact we choose not to pay
/// for"), sized to keep the load factor under 0.7.
pub const IdMap = struct {
    /// Parallel arrays rather than an array of structs: probing touches only
    /// `slots`, so keeping the 4-byte offsets separate from the 24-byte keys
    /// puts ~16 probe candidates in a cache line instead of ~2.
    slots: []u32,
    keys: []ExternalId,
    mask: usize,
    count: usize = 0,

    pub fn capacityFor(max_points: usize) usize {
        // Load factor 0.7, rounded up to a power of two so the modulo is a mask.
        const want = (max_points * 10) / 7 + 1;
        return std.math.ceilPowerOfTwo(usize, @max(16, want)) catch unreachable;
    }

    pub fn init(alloc: std.mem.Allocator, max_points: usize) !IdMap {
        const cap = capacityFor(max_points);
        const slots = try alloc.alloc(u32, cap);
        errdefer alloc.free(slots);
        const keys = try alloc.alloc(ExternalId, cap);
        @memset(slots, empty_slot);
        return .{ .slots = slots, .keys = keys, .mask = cap - 1 };
    }

    pub fn deinit(self: *IdMap, alloc: std.mem.Allocator) void {
        alloc.free(self.slots);
        alloc.free(self.keys);
    }

    pub fn clear(self: *IdMap) void {
        @memset(self.slots, empty_slot);
        self.count = 0;
    }

    pub fn get(self: *const IdMap, id: ExternalId) ?u32 {
        var i = id.hash() & self.mask;
        while (true) {
            const slot = self.slots[i];
            if (slot == empty_slot) return null;
            if (self.keys[i].eql(id)) return slot;
            i = (i + 1) & self.mask;
        }
    }

    pub const InsertResult = struct {
        offset: u32,
        /// False when the ID was already present, i.e. this upsert is an
        /// update-in-place (§2, `--max-id`).
        is_new: bool,
    };

    /// Look up `id`, inserting `next_offset` if absent.
    ///
    /// Returns the existing offset when the ID is known, so the caller
    /// overwrites that slot rather than appending. The caller only advances its
    /// offset counter when `is_new`.
    pub fn getOrInsert(self: *IdMap, id: ExternalId, next_offset: u32) error{MapFull}!InsertResult {
        // One slot always stays empty: `get` terminates on an empty slot and
        // nothing else, so a table filled to the last slot would loop forever
        // on an absent key. `capacityFor` keeps the load under 0.7 through
        // `IdSpace`; this is the guard for a direct caller.
        if (self.count + 1 >= self.slots.len) return error.MapFull;
        var i = id.hash() & self.mask;
        while (true) {
            const slot = self.slots[i];
            if (slot == empty_slot) {
                self.slots[i] = next_offset;
                self.keys[i] = id;
                self.count += 1;
                return .{ .offset = next_offset, .is_new = true };
            }
            if (self.keys[i].eql(id)) return .{ .offset = slot, .is_new = false };
            i = (i + 1) & self.mask;
        }
    }
};

/// Internal offset → external ID. A flat array, indexed directly.
pub const IdTable = struct {
    ids: []ExternalId,
    len: usize = 0,

    pub fn init(alloc: std.mem.Allocator, capacity: usize) !IdTable {
        return .{ .ids = try alloc.alloc(ExternalId, capacity) };
    }

    pub fn deinit(self: *IdTable, alloc: std.mem.Allocator) void {
        alloc.free(self.ids);
    }

    pub fn set(self: *IdTable, offset: u32, id: ExternalId) void {
        self.ids[offset] = id;
        if (offset + 1 > self.len) self.len = offset + 1;
    }

    pub fn get(self: *const IdTable, offset: u32) ExternalId {
        return self.ids[offset];
    }

    pub fn clear(self: *IdTable) void {
        self.len = 0;
    }
};

/// Both directions together, which is how callers always use them.
pub const IdSpace = struct {
    map: IdMap,
    table: IdTable,
    /// Points published so far, and the bound every reader scans to. Dense and
    /// monotonic (§3). Atomic because searches read it without the write lock:
    /// it is the handle by which a row becomes visible, so its ordering is the
    /// ordering of the row's contents.
    next: std.atomic.Value(usize) = .init(0),
    capacity: usize,

    pub fn init(alloc: std.mem.Allocator, capacity: usize) !IdSpace {
        var map = try IdMap.init(alloc, capacity);
        errdefer map.deinit(alloc);
        const table = try IdTable.init(alloc, capacity);
        return .{ .map = map, .table = table, .capacity = capacity };
    }

    pub fn deinit(self: *IdSpace, alloc: std.mem.Allocator) void {
        self.map.deinit(alloc);
        self.table.deinit(alloc);
    }

    pub fn clear(self: *IdSpace) void {
        self.map.clear();
        self.table.clear();
        self.next.store(0, .release);
    }

    pub const Error = error{ MapFull, CapacityExceeded };

    /// Resolve an upsert to an internal offset, *without publishing it*.
    ///
    /// Publication is split from resolution because a reader's only bound is
    /// `count()`: the moment an offset is inside it, a search may score that
    /// row and a build may wire it into the graph. Bumping `next` here, before
    /// the caller has written the vector, left a window where both read a row
    /// that is still zeroed — a point permanently mis-wired into the graph
    /// from a zero vector, with no error anywhere. The caller writes the row
    /// and then calls `publish`.
    ///
    /// The id map and the id table are only touched for a *new* point, and
    /// only in `publish`, for the same reason: nothing may name a row that has
    /// no contents yet.
    pub fn reserve(self: *IdSpace, id: ExternalId) Error!IdMap.InsertResult {
        if (self.map.get(id)) |off| return .{ .offset = off, .is_new = false };
        const n = self.next.load(.monotonic);
        if (n >= self.capacity) {
            // "The ID is known" is distinguished from "we are full" above: an
            // update to an existing point must still succeed at full capacity.
            return Error.CapacityExceeded;
        }
        return .{ .offset = @intCast(n), .is_new = true };
    }

    /// Make a reserved offset visible, after its row has been written.
    ///
    /// The release store pairs with the acquire load in `count`, so a reader
    /// that can see the offset can also see every byte of its vector.
    pub fn publish(self: *IdSpace, id: ExternalId, offset: u32) Error!void {
        const r = try self.map.getOrInsert(id, offset);
        std.debug.assert(r.offset == offset);
        self.table.set(offset, id);
        self.next.store(@as(usize, offset) + 1, .release);
    }

    pub fn lookup(self: *const IdSpace, id: ExternalId) ?u32 {
        return self.map.get(id);
    }

    pub fn external(self: *const IdSpace, offset: u32) ExternalId {
        return self.table.get(offset);
    }

    /// Points whose rows are fully written. Acquire, so the vector's bytes
    /// are visible to whoever can see the count.
    pub fn count(self: *const IdSpace) usize {
        return self.next.load(.acquire);
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

/// The two-step reserve/publish, as one call. Tests want the old shape; the
/// engine deliberately does not, because the split is what orders a row's
/// contents against its visibility.
fn upsertBoth(sp: *IdSpace, id: ExternalId) !IdMap.InsertResult {
    const r = try sp.reserve(id);
    if (r.is_new) try sp.publish(id, r.offset);
    return r;
}

test "numeric ids map densely and monotonically" {
    var sp = try IdSpace.init(testing.allocator, 1000);
    defer sp.deinit(testing.allocator);

    for (0..100) |i| {
        const r = try sp.reserve(.{ .num = i * 7 });
        try testing.expect(r.is_new);
        try testing.expectEqual(@as(u32, @intCast(i)), r.offset);
        try sp.publish(.{ .num = i * 7 }, r.offset);
    }
    try testing.expectEqual(@as(usize, 100), sp.count());

    for (0..100) |i| {
        try testing.expectEqual(@as(?u32, @intCast(i)), sp.lookup(.{ .num = i * 7 }));
        try testing.expect(sp.external(@intCast(i)).eql(.{ .num = i * 7 }));
    }
}

test "§2 --max-id: re-upserting an existing id updates in place" {
    var sp = try IdSpace.init(testing.allocator, 1000);
    defer sp.deinit(testing.allocator);

    const first = try upsertBoth(&sp, .{ .num = 42 });
    try testing.expect(first.is_new);
    try testing.expectEqual(@as(u32, 0), first.offset);

    // The same ID again must reuse offset 0 and not advance the counter.
    // Appending here would inflate points_count and break the ID-equals-row
    // requirement §4.3 puts on relevance runs.
    const second = try upsertBoth(&sp, .{ .num = 42 });
    try testing.expect(!second.is_new);
    try testing.expectEqual(@as(u32, 0), second.offset);
    try testing.expectEqual(@as(usize, 1), sp.count());
}

test "the map keeps one slot empty so a miss terminates" {
    // `get` stops on an empty slot and nothing else; a table filled to the
    // last slot looped forever on an absent key. The guard admits
    // `slots.len - 1` keys and refuses the one that would fill it.
    var map = try IdMap.init(testing.allocator, 4);
    defer map.deinit(testing.allocator);
    const room = map.slots.len - 1;
    for (0..room) |i| _ = try map.getOrInsert(.{ .num = i * 7919 }, @intCast(i));
    try testing.expectError(error.MapFull, map.getOrInsert(.{ .num = 999_999 }, 0));
    try testing.expectEqual(@as(?u32, null), map.get(.{ .num = 999_999 }));
}

test "uuid ids round-trip through parse and format" {
    const text = "550e8400-e29b-41d4-a716-446655440000";
    const id = ExternalId.parseUuid(text).?;
    var out: [36]u8 = undefined;
    id.formatUuid(&out);
    try testing.expectEqualStrings(text, &out);

    // Uppercase input normalises to lowercase output.
    const upper = "550E8400-E29B-41D4-A716-446655440000";
    const id2 = ExternalId.parseUuid(upper).?;
    try testing.expect(id.eql(id2));
    id2.formatUuid(&out);
    try testing.expectEqualStrings(text, &out);
}

test "malformed uuids are rejected rather than guessed" {
    try testing.expectEqual(@as(?ExternalId, null), ExternalId.parseUuid(""));
    try testing.expectEqual(@as(?ExternalId, null), ExternalId.parseUuid("550e8400"));
    // Right length, wrong separators.
    try testing.expectEqual(@as(?ExternalId, null), ExternalId.parseUuid("550e8400xe29bx41d4xa716x446655440000"));
    // Right shape, non-hex digit.
    try testing.expectEqual(@as(?ExternalId, null), ExternalId.parseUuid("550e8400-e29b-41d4-a716-44665544000g"));
    // One character too long.
    try testing.expectEqual(@as(?ExternalId, null), ExternalId.parseUuid("550e8400-e29b-41d4-a716-4466554400000"));
}

test "a num and a uuid with the same bits are distinct keys" {
    var sp = try IdSpace.init(testing.allocator, 100);
    defer sp.deinit(testing.allocator);

    const a = try upsertBoth(&sp, .{ .num = 12345 });
    const b = try upsertBoth(&sp, .{ .uuid = 12345 });
    try testing.expect(a.is_new);
    try testing.expect(b.is_new);
    try testing.expect(a.offset != b.offset);
    try testing.expectEqual(@as(usize, 2), sp.count());
}

test "uuid ids map densely, mixed with numeric ones" {
    var sp = try IdSpace.init(testing.allocator, 1000);
    defer sp.deinit(testing.allocator);

    var expected: u32 = 0;
    for (0..50) |i| {
        const u = try upsertBoth(&sp, .{ .uuid = @as(u128, i) << 64 | 0xabcdef });
        try testing.expectEqual(expected, u.offset);
        expected += 1;
        const n = try upsertBoth(&sp, .{ .num = i });
        try testing.expectEqual(expected, n.offset);
        expected += 1;
    }
    try testing.expectEqual(@as(usize, 100), sp.count());
}

test "capacity is enforced but updates still succeed when full" {
    var sp = try IdSpace.init(testing.allocator, 4);
    defer sp.deinit(testing.allocator);

    for (0..4) |i| _ = try upsertBoth(&sp, .{ .num = i });
    try testing.expectEqual(@as(usize, 4), sp.count());

    // A new ID has nowhere to go.
    try testing.expectError(IdSpace.Error.CapacityExceeded, upsertBoth(&sp, .{ .num = 999 }));

    // But an existing one must still resolve, that is the `--max-id` workload
    // at steady state, where every upsert is an overwrite.
    const r = try upsertBoth(&sp, .{ .num = 2 });
    try testing.expect(!r.is_new);
    try testing.expectEqual(@as(u32, 2), r.offset);
}

test "offset zero is never confused with an empty slot" {
    // The reason `empty_slot` is maxInt rather than 0. Point 0 is ordinary and
    // must be findable.
    var sp = try IdSpace.init(testing.allocator, 16);
    defer sp.deinit(testing.allocator);
    const r = try upsertBoth(&sp, .{ .num = 0 });
    try testing.expectEqual(@as(u32, 0), r.offset);
    try testing.expectEqual(@as(?u32, 0), sp.lookup(.{ .num = 0 }));
    try testing.expectEqual(@as(?u32, null), sp.lookup(.{ .num = 1 }));
}

test "map survives heavy collision pressure" {
    // Keys chosen to land in *one* bucket, so every probe chain is long. The
    // hash avalanches (`mix`), so stride multiples do not collide -- the test
    // used to insert 200 of those at load 0.2 and exercise the ordinary path;
    // now the keys are found by their hash.
    var map = try IdMap.init(testing.allocator, 512);
    defer map.deinit(testing.allocator);

    var keys: [200]u64 = undefined;
    var found: usize = 0;
    var candidate: u64 = 1;
    while (found < keys.len) : (candidate += 1) {
        const id = ExternalId{ .num = candidate };
        if (id.hash() & map.mask == 0) {
            keys[found] = candidate;
            found += 1;
        }
    }
    var next: u32 = 0;
    for (keys) |k| {
        const r = try map.getOrInsert(.{ .num = k }, next);
        try testing.expect(r.is_new);
        next += 1;
    }
    for (keys, 0..) |k, i| {
        try testing.expectEqual(@as(?u32, @intCast(i)), map.get(.{ .num = k }));
    }
    try testing.expectEqual(@as(?u32, null), map.get(.{ .num = 999_999_999 }));
}

test "capacityFor keeps the load factor under 0.7" {
    for ([_]usize{ 1, 10, 1000, 1_000_000 }) |n| {
        const cap = IdMap.capacityFor(n);
        try testing.expect(std.math.isPowerOfTwo(cap));
        const load = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(cap));
        try testing.expect(load <= 0.7);
    }
}

test "clear resets both directions" {
    var sp = try IdSpace.init(testing.allocator, 100);
    defer sp.deinit(testing.allocator);
    for (0..10) |i| _ = try upsertBoth(&sp, .{ .num = i });
    sp.clear();
    try testing.expectEqual(@as(usize, 0), sp.count());
    try testing.expectEqual(@as(?u32, null), sp.lookup(.{ .num = 5 }));
    // And offsets restart from zero.
    const r = try upsertBoth(&sp, .{ .num = 77 });
    try testing.expectEqual(@as(u32, 0), r.offset);
}
