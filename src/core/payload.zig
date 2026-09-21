//! §3: "**Payload**: opaque, append-only blob per point, phase 2." And M7's
//! keyword index, which is what turns W12 from a row this engine declines into
//! a filtered-search row.
//!
//! ## What is stored
//!
//! The map entries of `PointStruct.payload` (`map<string, Value>`, field 3),
//! **as they arrived on the wire**, framed one after another as
//! `[varint len][entry bytes]`. An entry is a protobuf map entry — a nested
//! message with `key = 1` (string) and `value = 2` (`qdrant.Value`) — and the
//! response messages carry the same map under a different field number
//! (`ScoredPoint.payload = 2`, `RetrievedPoint.payload = 2`), so `with_payload`
//! is a re-tag and a copy, never a decode and re-encode. Opaque, as §3 says:
//! nothing here knows what a `Value` *means* beyond the two kinds a filter can
//! ask about.
//!
//! ## Where it lives
//!
//! An append-only chunk arena. Chunks are allocated as they fill and never
//! moved, so a reader holding a slice into one is safe for as long as the
//! store exists; a point's slot is a single `u64` — chunk-relative offset and
//! length packed — published with release ordering *after* the bytes are
//! written, so a reader that can see the slot can see the blob. Overwriting a
//! point's payload appends a new blob and repoints the slot; the old bytes are
//! garbage until the store is freed, which is §3's "no compaction" applied to
//! payloads. The write side runs under `Collection.write_lock`; the read side
//! takes no lock (§6.3).
//!
//! ## The index
//!
//! Per indexed field, a posting list per value: `keyword` maps string to
//! offsets, `integer` maps `i64` to offsets. Postings are **hints, never
//! truth**: an overwrite leaves the old value's posting in place rather than
//! searching it out (a linear scan of a posting list on every overwrite is the
//! cost W11 would pay for an index it does not use), so every offset a posting
//! yields is re-checked against the point's actual blob before it is admitted.
//! The index can only over-approximate, never miss, and an unindexed filter is
//! answered by evaluating the blob directly, so the answer does not depend on
//! whether `CreateFieldIndex` was ever called, only its cost does. That is
//! Qdrant's contract too, and decisions.md records what the unindexed cost is:
//! a scan of every point per query.
//!
//! Only the postings take a lock — a `RwLock`, read for the length of one
//! `select` and written for one insert — because the maps behind them are
//! rehashed as they grow, and a reader on a map mid-rehash is a use-after-free.

const std = @import("std");
const wire = @import("../proto/wire.zig");
const futex_lock = @import("../lock.zig");

// `qdrant.Value { oneof kind { NullValue null_value = 1; double double_value = 2;
//   int64 integer_value = 3; string string_value = 4; bool bool_value = 5;
//   Struct struct_value = 6; ListValue list_value = 7; } }` (VERIFIED,
// json_with_int.proto in qdrant-client 1.19.0). `ListValue { repeated Value
// values = 1; }`.
pub const value_integer: u32 = 3;
pub const value_string: u32 = 4;
pub const value_bool: u32 = 5;
pub const value_list: u32 = 7;

/// A map entry, on the wire: `key = 1`, `value = 2`.
const entry_key: u32 = 1;
const entry_value: u32 = 2;

// =========================================================================
// Filters
// =========================================================================

/// One `Match`, the part of `FieldCondition` this engine evaluates.
///
/// `Match { oneof match_value { string keyword = 1; int64 integer = 2;
/// bool boolean = 3; string text = 4; RepeatedStrings keywords = 5;
/// RepeatedIntegers integers = 6; ... } }` (VERIFIED, points.proto). `text`,
/// `phrase`, `prefix` and the `except_*` forms are refused by name at decode.
pub const Match = union(enum) {
    keyword: []const u8,
    integer: i64,
    boolean: bool,
    /// The bytes of a `RepeatedStrings { repeated string strings = 1; }`:
    /// any one of them matching is a match (`match_any`).
    keywords: []const u8,
    /// The bytes of a `RepeatedIntegers { repeated int64 integers = 1; }`.
    integers: []const u8,
};

pub const Condition = struct {
    key: []const u8,
    match: Match,
};

/// Conditions per clause. bfb's W12 sends one; the bound exists so a
/// decoded filter is a fixed-size value on the handler's stack (§6.3: no
/// allocation on the query path) rather than a list.
pub const max_conditions = 16;

/// A decoded `Filter { should = 1; must = 2; must_not = 3; }`, borrowed from
/// the request body it was decoded from.
///
/// Semantics are Qdrant's: every `must` holds, no `must_not` holds, and when
/// `should` is non-empty at least one of them holds.
pub const Filter = struct {
    must: [max_conditions]Condition = undefined,
    must_len: usize = 0,
    must_not: [max_conditions]Condition = undefined,
    must_not_len: usize = 0,
    should: [max_conditions]Condition = undefined,
    should_len: usize = 0,

    pub const Clause = enum { must, must_not, should };

    pub fn add(self: *Filter, clause: Clause, c: Condition) error{TooManyConditions}!void {
        const arr, const len = switch (clause) {
            .must => .{ &self.must, &self.must_len },
            .must_not => .{ &self.must_not, &self.must_not_len },
            .should => .{ &self.should, &self.should_len },
        };
        if (len.* >= max_conditions) return error.TooManyConditions;
        arr[len.*] = c;
        len.* += 1;
    }

    pub fn isEmpty(self: *const Filter) bool {
        return self.must_len == 0 and self.must_not_len == 0 and self.should_len == 0;
    }

    pub fn musts(self: *const Filter) []const Condition {
        return self.must[0..self.must_len];
    }
    pub fn mustNots(self: *const Filter) []const Condition {
        return self.must_not[0..self.must_not_len];
    }
    pub fn shoulds(self: *const Filter) []const Condition {
        return self.should[0..self.should_len];
    }

    /// Whether a framed blob satisfies the filter.
    pub fn matchesBlob(self: *const Filter, blob: []const u8) bool {
        for (self.musts()) |c| if (!conditionMatches(blob, c)) return false;
        for (self.mustNots()) |c| if (conditionMatches(blob, c)) return false;
        if (self.should_len > 0) {
            for (self.shoulds()) |c| if (conditionMatches(blob, c)) return true;
            return false;
        }
        return true;
    }
};

/// Whether any entry under `c.key` in the blob matches `c.match`.
///
/// A key absent from the payload matches nothing, which is what makes
/// `must_not` on an absent key true and `must` on one false, as in Qdrant.
fn conditionMatches(blob: []const u8, c: Condition) bool {
    var it = BlobIterator.init(blob);
    while (it.next()) |entry| {
        const kv = splitEntry(entry) orelse continue;
        if (!std.mem.eql(u8, kv.key, c.key)) continue;
        if (valueMatches(kv.value, c.match)) return true;
    }
    return false;
}

/// Whether one `Value` matches, descending one level into a `ListValue` so a
/// multi-valued keyword field (`["a", "b"]`) matches either.
fn valueMatches(value: []const u8, m: Match) bool {
    return valueMatchesAt(value, m, true);
}

/// One walk over a `Value`'s scalar arms, with the list descent as a flag
/// rather than a second copy of the arms: a list of lists is not a keyword
/// field, so the descent stops after one level.
fn valueMatchesAt(value: []const u8, m: Match, descend: bool) bool {
    var r = wire.Reader.init(value);
    while (!r.atEnd()) {
        const t = r.tag() catch return false;
        switch (t.field) {
            value_string => {
                const s = r.bytes() catch return false;
                switch (m) {
                    .keyword => |k| if (std.mem.eql(u8, s, k)) return true,
                    .keywords => |list| if (repeatedStringsContain(list, s)) return true,
                    else => {},
                }
            },
            value_integer => {
                const v: i64 = @bitCast(r.varint() catch return false);
                switch (m) {
                    .integer => |i| if (v == i) return true,
                    .integers => |list| if (repeatedIntegersContain(list, v)) return true,
                    else => {},
                }
            },
            value_bool => {
                const b = r.boolean() catch return false;
                switch (m) {
                    .boolean => |want| if (b == want) return true,
                    else => {},
                }
            },
            value_list => {
                var list = r.nested() catch return false;
                while (!list.atEnd()) {
                    const lt = list.tag() catch return false;
                    if (lt.field != 1) {
                        list.skip(lt.wire_type) catch return false;
                        continue;
                    }
                    const inner = list.bytes() catch return false;
                    if (descend and valueMatchesAt(inner, m, false)) return true;
                }
            },
            else => r.skip(t.wire_type) catch return false,
        }
    }
    return false;
}

fn repeatedStringsContain(list: []const u8, s: []const u8) bool {
    var r = wire.Reader.init(list);
    while (!r.atEnd()) {
        const t = r.tag() catch return false;
        if (t.field == 1 and t.wire_type == .length_delimited) {
            const x = r.bytes() catch return false;
            if (std.mem.eql(u8, x, s)) return true;
        } else r.skip(t.wire_type) catch return false;
    }
    return false;
}

/// `RepeatedIntegers { repeated int64 integers = 1; }`.
///
/// The field is checked *before* the payload is consumed, as
/// `repeatedStringsContain` and `PostingLists` do. Reading the nested blob
/// first and testing the field inside it meant a length-delimited field under
/// some other number was parsed as a run of varints, so malformed bytes in a
/// field this has no interest in abandoned the whole condition (`catch return
/// false`) instead of being skipped. Four walks over this message shape exist
/// in this file and two of them did it the other way round; they now agree.
fn repeatedIntegersContain(list: []const u8, v: i64) bool {
    var r = wire.Reader.init(list);
    while (!r.atEnd()) {
        const t = r.tag() catch return false;
        if (t.field != 1) {
            r.skip(t.wire_type) catch return false;
            continue;
        }
        switch (t.wire_type) {
            .varint => {
                const x: i64 = @bitCast(r.varint() catch return false);
                if (x == v) return true;
            },
            // `repeated int64` is packed in proto3.
            .length_delimited => {
                var packed_r = r.nested() catch return false;
                while (!packed_r.atEnd()) {
                    const x: i64 = @bitCast(packed_r.varint() catch return false);
                    if (x == v) return true;
                }
            },
            else => r.skip(t.wire_type) catch return false,
        }
    }
    return false;
}

// =========================================================================
// Blob framing
// =========================================================================

/// Walks the framed entries of one point's blob.
pub const BlobIterator = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn init(blob: []const u8) BlobIterator {
        return .{ .buf = blob };
    }

    pub fn next(self: *BlobIterator) ?[]const u8 {
        if (self.pos >= self.buf.len) return null;
        var r = wire.Reader.init(self.buf[self.pos..]);
        const len = r.varint() catch return null;
        const start = self.pos + r.pos;
        const end = start + @as(usize, @intCast(len));
        if (end > self.buf.len) return null;
        self.pos = end;
        return self.buf[start..end];
    }
};

/// Walks the map entries of a `map<string, Value>` field in a request body:
/// every `field`-numbered length-delimited value in `bytes`, in order. The
/// bytes may hold other fields between entries; they are skipped.
pub const WireEntries = struct {
    r: wire.Reader,
    field: u32,

    pub fn init(bytes: []const u8, field: u32) WireEntries {
        return .{ .r = wire.Reader.init(bytes), .field = field };
    }

    pub fn next(self: *WireEntries) wire.Error!?[]const u8 {
        while (!self.r.atEnd()) {
            const t = try self.r.tag();
            if (t.field == self.field and t.wire_type == .length_delimited) return try self.r.bytes();
            try self.r.skip(t.wire_type);
        }
        return null;
    }
};

const Entry = struct { key: []const u8, value: []const u8 };

/// The key and the `Value` bytes of one map entry, or null when it has no key.
pub fn splitEntry(entry: []const u8) ?Entry {
    var r = wire.Reader.init(entry);
    var key: ?[]const u8 = null;
    var value: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = r.tag() catch return null;
        switch (t.field) {
            entry_key => key = r.bytes() catch return null,
            entry_value => value = r.bytes() catch return null,
            else => r.skip(t.wire_type) catch return null,
        }
    }
    return .{ .key = key orelse return null, .value = value };
}

fn varintLen(v: u64) usize {
    var n: usize = 1;
    var x = v >> 7;
    while (x != 0) : (x >>= 7) n += 1;
    return n;
}

fn writeVarint(buf: []u8, v: u64) usize {
    var x = v;
    var i: usize = 0;
    while (x >= 0x80) : (x >>= 7) {
        buf[i] = @as(u8, @truncate(x & 0x7f)) | 0x80;
        i += 1;
    }
    buf[i] = @truncate(x);
    return i + 1;
}

/// Bytes the framed form of `entry` occupies.
pub fn framedLen(entry: []const u8) usize {
    return varintLen(entry.len) + entry.len;
}

/// Append `entry` in framed form.
pub fn frameInto(out: *std.ArrayList(u8), alloc: std.mem.Allocator, entry: []const u8) !void {
    var lenbuf: [10]u8 = undefined;
    const n = writeVarint(&lenbuf, entry.len);
    try out.appendSlice(alloc, lenbuf[0..n]);
    try out.appendSlice(alloc, entry);
}

// =========================================================================
// The index
// =========================================================================

/// `FieldType` values this engine indexes (`FieldTypeKeyword = 0`,
/// `FieldTypeInteger = 1`, VERIFIED collections.proto). Stored as the
/// `PayloadSchemaType` the collection info reports them as (`Keyword = 1`,
/// `Integer = 2`), which is a different enum with different numbers.
pub const Kind = enum(u8) {
    keyword = 1,
    integer = 2,

    pub fn fromFieldType(v: u64) ?Kind {
        return switch (v) {
            0 => .keyword,
            1 => .integer,
            else => null,
        };
    }

    /// `PayloadSchemaType`.
    pub fn schemaType(self: Kind) u32 {
        return @intFromEnum(self);
    }
};

const Postings = std.ArrayList(u32);

pub const Field = struct {
    name: []u8,
    kind: Kind,
    keywords: std.StringHashMapUnmanaged(Postings) = .empty,
    integers: std.AutoHashMapUnmanaged(i64, Postings) = .empty,
    /// Postings written, which over-counts overwritten points the same way
    /// the postings do. Reported as `PayloadSchemaInfo.points`.
    points: usize = 0,

    fn deinit(self: *Field, alloc: std.mem.Allocator) void {
        var ki = self.keywords.iterator();
        while (ki.next()) |e| {
            alloc.free(e.key_ptr.*);
            e.value_ptr.deinit(alloc);
        }
        self.keywords.deinit(alloc);
        var ii = self.integers.iterator();
        while (ii.next()) |e| e.value_ptr.deinit(alloc);
        self.integers.deinit(alloc);
        alloc.free(self.name);
    }

    /// Index one point's value(s) for this field.
    fn add(self: *Field, alloc: std.mem.Allocator, offset: u32, value: []const u8) !void {
        var r = wire.Reader.init(value);
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                value_string => {
                    const s = try r.bytes();
                    if (self.kind == .keyword) try self.addKeyword(alloc, offset, s);
                },
                value_integer => {
                    const v: i64 = @bitCast(try r.varint());
                    if (self.kind == .integer) try self.addInteger(alloc, offset, v);
                },
                value_list => {
                    var list = try r.nested();
                    while (!list.atEnd()) {
                        const lt = try list.tag();
                        if (lt.field != 1) {
                            try list.skip(lt.wire_type);
                            continue;
                        }
                        // One level, like `valueMatches`.
                        var inner = wire.Reader.init(try list.bytes());
                        while (!inner.atEnd()) {
                            const it = try inner.tag();
                            switch (it.field) {
                                value_string => {
                                    const s = try inner.bytes();
                                    if (self.kind == .keyword) try self.addKeyword(alloc, offset, s);
                                },
                                value_integer => {
                                    const v: i64 = @bitCast(try inner.varint());
                                    if (self.kind == .integer) try self.addInteger(alloc, offset, v);
                                },
                                else => try inner.skip(it.wire_type),
                            }
                        }
                    }
                },
                else => try r.skip(t.wire_type),
            }
        }
    }

    /// Every value under this field's key in a framed blob.
    fn addBlob(self: *Field, alloc: std.mem.Allocator, offset: u32, blob: []const u8) !void {
        var it = BlobIterator.init(blob);
        while (it.next()) |entry| {
            const kv = splitEntry(entry) orelse continue;
            if (std.mem.eql(u8, self.name, kv.key)) try self.add(alloc, offset, kv.value);
        }
    }

    fn addKeyword(self: *Field, alloc: std.mem.Allocator, offset: u32, s: []const u8) !void {
        const gop = try self.keywords.getOrPut(alloc, s);
        if (!gop.found_existing) {
            // The map borrows its key: own it, and repoint the entry at the
            // copy before anything can look it up.
            const owned = try alloc.dupe(u8, s);
            gop.key_ptr.* = owned;
            gop.value_ptr.* = .empty;
        }
        try gop.value_ptr.append(alloc, offset);
        self.points += 1;
    }

    fn addInteger(self: *Field, alloc: std.mem.Allocator, offset: u32, v: i64) !void {
        const gop = try self.integers.getOrPut(alloc, v);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(alloc, offset);
        self.points += 1;
    }
};

// =========================================================================
// The store
// =========================================================================

/// A slot packs the blob's arena offset and length: 40 bits of offset
/// (1 TiB of arena) and 24 of length (16 MiB per point). Zero is "no
/// payload", so an empty map and an absent one are the same thing, which is
/// what Qdrant returns for both.
const len_bits = 24;
const len_mask: u64 = (1 << len_bits) - 1;
pub const max_blob = len_mask;
/// 16 MiB chunks: two of them hold a million bfb keyword payloads.
pub const chunk_size: usize = 1 << 24;
/// Never moved, so a reader's slice stays valid; bounded so the pointer table
/// is a fixed array the reader indexes without a lock.
pub const max_chunks = 4096;

/// Readers-writer lock over the postings. `std.Io.RwLock` needs an `Io`
/// instance this server does not carry, so this is the project's futex mutex
/// plus a reader count: a reader takes the mutex for one increment, a writer
/// holds it and waits for the count to drain. Readers are one `select` long
/// and writers one posting append, so neither side waits for much.
const RwLock = struct {
    mutex: futex_lock.Mutex = .{},
    readers: std.atomic.Value(u32) = .init(0),

    fn lockShared(self: *RwLock) void {
        self.mutex.lock();
        _ = self.readers.fetchAdd(1, .acquire);
        self.mutex.unlock();
    }

    fn unlockShared(self: *RwLock) void {
        _ = self.readers.fetchSub(1, .release);
    }

    fn lock(self: *RwLock) void {
        self.mutex.lock();
        while (self.readers.load(.acquire) != 0) std.atomic.spinLoopHint();
    }

    fn unlock(self: *RwLock) void {
        self.mutex.unlock();
    }
};

pub const Selection = struct {
    /// Points the filter admits, verified against their blobs.
    count: usize,
};

pub const Store = struct {
    slots: []std.atomic.Value(u64),
    chunks: []std.atomic.Value(?[*]u8),
    chunk_count: usize = 0,
    /// Bytes used in the last chunk.
    used: usize = 0,
    /// Points whose slot is non-zero.
    with_payload: usize = 0,
    fields: std.ArrayList(Field) = .empty,
    index_lock: RwLock = .{},

    pub fn init(alloc: std.mem.Allocator, capacity: usize) !Store {
        const slots = try alloc.alloc(std.atomic.Value(u64), capacity);
        errdefer alloc.free(slots);
        for (slots) |*s| s.* = .init(0);
        const chunks = try alloc.alloc(std.atomic.Value(?[*]u8), max_chunks);
        for (chunks) |*c| c.* = .init(null);
        return .{ .slots = slots, .chunks = chunks };
    }

    pub fn deinit(self: *Store, alloc: std.mem.Allocator) void {
        for (self.chunks[0..self.chunk_count]) |*c| {
            if (c.load(.monotonic)) |p| {
                const whole: []u8 = p[0..chunk_size];
                alloc.free(whole);
            }
        }
        alloc.free(self.chunks);
        alloc.free(self.slots);
        for (self.fields.items) |*f| f.deinit(alloc);
        self.fields.deinit(alloc);
    }

    /// The framed blob of `offset`, empty when it has none. Lock-free.
    pub fn get(self: *const Store, offset: u32) []const u8 {
        const slot = self.slots[offset].load(.acquire);
        if (slot == 0) return &.{};
        const len: usize = @intCast(slot & len_mask);
        const off: usize = @intCast(slot >> len_bits);
        const chunk = self.chunks[off / chunk_size].load(.acquire) orelse return &.{};
        return chunk[off % chunk_size ..][0..len];
    }

    pub fn has(self: *const Store, offset: u32) bool {
        return self.slots[offset].load(.acquire) != 0;
    }

    /// Whether `offset`'s payload satisfies `filter`. Lock-free.
    pub fn matches(self: *const Store, offset: u32, filter: *const Filter) bool {
        return filter.matchesBlob(self.get(offset));
    }

    /// Replace `offset`'s payload with the map entries in `entries` (wire
    /// bytes holding repeated `field`-numbered entries). Under the
    /// collection's write lock. Empty entries clear the payload, which is what
    /// an upsert without one means.
    pub fn set(self: *Store, alloc: std.mem.Allocator, offset: u32, entries: []const u8, map_field: u32) !void {
        var total: usize = 0;
        var it = WireEntries.init(entries, map_field);
        while (try it.next()) |e| total += framedLen(e);
        if (total == 0) {
            self.publish(offset, 0);
            return;
        }
        if (total > max_blob) return error.PayloadTooLarge;
        const dst = try self.reserve(alloc, total);
        var pos: usize = 0;
        it = WireEntries.init(entries, map_field);
        while (try it.next()) |e| {
            pos += writeVarint(dst.bytes[pos..], e.len);
            @memcpy(dst.bytes[pos..][0..e.len], e);
            pos += e.len;
        }
        std.debug.assert(pos == total);
        // Index first, publish second: a slot published before its posting
        // left a point the bitset path could not find if the insert then
        // failed, and nothing repaired it.
        try self.indexBlob(alloc, offset, dst.bytes);
        self.publish(offset, (@as(u64, dst.offset) << len_bits) | @as(u64, total));
    }

    /// `SetPayload`: merge the entries in `entries` over the existing payload,
    /// replacing values under the same key and keeping the rest.
    pub fn merge(self: *Store, alloc: std.mem.Allocator, offset: u32, entries: []const u8, map_field: u32) !void {
        var framed: std.ArrayList(u8) = .empty;
        defer framed.deinit(alloc);
        // Existing entries whose key is not being set.
        var old = BlobIterator.init(self.get(offset));
        while (old.next()) |entry| {
            const kv = splitEntry(entry) orelse continue;
            var replaced = false;
            var it = WireEntries.init(entries, map_field);
            while (try it.next()) |e| {
                const nkv = splitEntry(e) orelse continue;
                if (std.mem.eql(u8, nkv.key, kv.key)) {
                    replaced = true;
                    break;
                }
            }
            if (!replaced) try frameInto(&framed, alloc, entry);
        }
        var it = WireEntries.init(entries, map_field);
        while (try it.next()) |e| try frameInto(&framed, alloc, e);
        try self.setFramed(alloc, offset, framed.items);
    }

    /// Replace `offset`'s payload with an already framed blob.
    pub fn setFramed(self: *Store, alloc: std.mem.Allocator, offset: u32, framed: []const u8) !void {
        if (framed.len == 0) {
            self.publish(offset, 0);
            return;
        }
        if (framed.len > max_blob) return error.PayloadTooLarge;
        const dst = try self.reserve(alloc, framed.len);
        @memcpy(dst.bytes, framed);
        try self.indexBlob(alloc, offset, dst.bytes);
        self.publish(offset, (@as(u64, dst.offset) << len_bits) | @as(u64, framed.len));
    }

    const Reserved = struct { bytes: []u8, offset: u64 };

    fn reserve(self: *Store, alloc: std.mem.Allocator, n: usize) !Reserved {
        std.debug.assert(n <= chunk_size);
        if (self.chunk_count == 0 or self.used + n > chunk_size) {
            if (self.chunk_count >= max_chunks) return error.PayloadArenaFull;
            const chunk = try alloc.alloc(u8, chunk_size);
            self.chunks[self.chunk_count].store(chunk.ptr, .release);
            self.chunk_count += 1;
            self.used = 0;
        }
        const chunk = self.chunks[self.chunk_count - 1].load(.monotonic).?;
        const start = self.used;
        self.used += n;
        return .{
            .bytes = chunk[start..][0..n],
            .offset = @as(u64, (self.chunk_count - 1)) * chunk_size + start,
        };
    }

    fn publish(self: *Store, offset: u32, slot: u64) void {
        const had = self.slots[offset].load(.monotonic) != 0;
        self.slots[offset].store(slot, .release);
        if (had and slot == 0) self.with_payload -= 1;
        if (!had and slot != 0) self.with_payload += 1;
    }

    /// Add a blob's values to every indexed field. Under the exclusive lock
    /// for the whole blob: `fields.items` is read here and appended to by
    /// `createIndex`, and the postings behind it are rehashed as they grow.
    fn indexBlob(self: *Store, alloc: std.mem.Allocator, offset: u32, blob: []const u8) !void {
        if (self.fields.items.len == 0) return;
        self.index_lock.lock();
        defer self.index_lock.unlock();
        for (self.fields.items) |*f| try f.addBlob(alloc, offset, blob);
    }

    /// Hold the shared lock while reading `fields.items` from outside:
    /// `createIndex` appends under the exclusive half.
    pub fn lockFields(self: *const Store) void {
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.lockShared();
    }

    pub fn unlockFields(self: *const Store) void {
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.unlockShared();
    }

    pub fn field(self: *const Store, name: []const u8) ?*const Field {
        for (self.fields.items) |*f| if (std.mem.eql(u8, f.name, name)) return f;
        return null;
    }

    /// `CreateFieldIndex`: index `name` as `kind` over the `count` points
    /// stored so far, and every point written from now on. Idempotent for the
    /// same kind; a different kind on an existing name is refused.
    pub fn createIndex(self: *Store, alloc: std.mem.Allocator, name: []const u8, kind: Kind, count: usize) !void {
        for (self.fields.items) |f| {
            if (std.mem.eql(u8, f.name, name)) {
                if (f.kind != kind) return error.IndexKindMismatch;
                return;
            }
        }
        const owned = try alloc.dupe(u8, name);
        var f = Field{ .name = owned, .kind = kind };
        errdefer f.deinit(alloc);
        // Populate first, publish second. The field is private until it is
        // appended, so the backfill needs no lock; appending it under the
        // exclusive half is what makes it visible to `select`, which walks
        // `fields.items`. Published before its postings existed, a concurrent
        // filtered query found the new field, saw a partial posting list,
        // chose it as cheapest and answered a short page from it.
        var off: u32 = 0;
        while (off < count) : (off += 1) {
            const blob = self.get(off);
            if (blob.len == 0) continue;
            try f.addBlob(alloc, off, blob);
        }
        self.index_lock.lock();
        defer self.index_lock.unlock();
        try self.fields.append(alloc, f);
    }

    /// Whether some `must` condition can be answered from a posting list.
    pub fn indexedMust(self: *const Store, filter: *const Filter) bool {
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.lockShared();
        defer rw.unlockShared();
        return self.cheapestMust(filter) != null;
    }

    const Pick = struct { f: *const Field, c: Condition };

    /// The indexed `must` condition with the shortest posting list, if any.
    /// Under the shared lock.
    fn cheapestMust(self: *const Store, filter: *const Filter) ?Pick {
        var best: ?Pick = null;
        var best_len: usize = std.math.maxInt(usize);
        for (filter.musts()) |c| {
            const f = self.field(c.key) orelse continue;
            const n = postingLen(f, c.match) orelse continue;
            if (n < best_len) {
                best_len = n;
                best = .{ .f = f, .c = c };
            }
        }
        return best;
    }

    fn postingLen(f: *const Field, m: Match) ?usize {
        switch (m) {
            .keyword => |k| {
                if (f.kind != .keyword) return null;
                return if (f.keywords.get(k)) |p| p.items.len else 0;
            },
            .integer => |i| {
                if (f.kind != .integer) return null;
                return if (f.integers.get(i)) |p| p.items.len else 0;
            },
            .keywords => |list| {
                if (f.kind != .keyword) return null;
                var n: usize = 0;
                var r = wire.Reader.init(list);
                while (!r.atEnd()) {
                    const t = r.tag() catch return null;
                    if (t.field == 1 and t.wire_type == .length_delimited) {
                        const s = r.bytes() catch return null;
                        if (f.keywords.get(s)) |p| n += p.items.len;
                    } else r.skip(t.wire_type) catch return null;
                }
                return n;
            },
            .integers => |list| {
                if (f.kind != .integer) return null;
                var n: usize = 0;
                var r = wire.Reader.init(list);
                while (!r.atEnd()) {
                    const t = r.tag() catch return null;
                    // Field first, then the payload. See `repeatedIntegersContain`.
                    if (t.field != 1) {
                        r.skip(t.wire_type) catch return null;
                        continue;
                    }
                    switch (t.wire_type) {
                        .varint => {
                            const v: i64 = @bitCast(r.varint() catch return null);
                            if (f.integers.get(v)) |p| n += p.items.len;
                        },
                        .length_delimited => {
                            var pr = r.nested() catch return null;
                            while (!pr.atEnd()) {
                                const v: i64 = @bitCast(pr.varint() catch return null);
                                if (f.integers.get(v)) |p| n += p.items.len;
                            }
                        },
                        else => r.skip(t.wire_type) catch return null,
                    }
                }
                return n;
            },
            .boolean => return null,
        }
    }

    /// Build the set of points `filter` admits, from the index.
    ///
    /// `bits` is the caller's scratch, one bit per offset up to `bound`
    /// (`Collection.id_space.count()` at the moment of the query -- the
    /// published offset count, *not* `Collection.count()`, which subtracts
    /// tombstones and would exclude the last live offsets: a posting may name
    /// an offset whose row is written but not yet published, and nothing may
    /// score that). Every offset a posting yields is verified against the
    /// point's blob and the whole filter, so the result is exact whatever the
    /// postings have accumulated. Returns null when no `must` condition is
    /// indexed, in which case the caller evaluates blobs directly.
    pub fn select(self: *const Store, filter: *const Filter, bits: []u64, bound: usize) ?Selection {
        // The lock is the one thing a *reader* mutates, like `SearchGuard`.
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.lockShared();
        defer rw.unlockShared();
        const pick = self.cheapestMust(filter) orelse return null;
        const words = (bound + 63) / 64;
        std.debug.assert(words <= bits.len);
        @memset(bits[0..words], 0);
        var count: usize = 0;
        var lists = PostingLists.init(pick.f, pick.c.match);
        while (lists.next()) |p| {
            for (p.items) |off| {
                if (off >= bound) continue;
                const w = off / 64;
                const m = @as(u64, 1) << @intCast(off % 64);
                if (bits[w] & m != 0) continue;
                if (!self.matches(off, filter)) continue;
                bits[w] |= m;
                count += 1;
            }
        }
        return .{ .count = count };
    }

    /// The posting lists one `Match` reads: one for a scalar, several for a
    /// `keywords`/`integers` list.
    const PostingLists = struct {
        f: *const Field,
        m: Match,
        r: wire.Reader,
        packed_r: ?wire.Reader = null,
        done: bool = false,

        fn init(f: *const Field, m: Match) PostingLists {
            const raw: []const u8 = switch (m) {
                .keywords, .integers => |l| l,
                else => &.{},
            };
            return .{ .f = f, .m = m, .r = wire.Reader.init(raw) };
        }

        fn next(self: *PostingLists) ?*const Postings {
            switch (self.m) {
                .keyword => |k| {
                    if (self.done) return null;
                    self.done = true;
                    return if (self.f.keywords.getPtr(k)) |p| p else null;
                },
                .integer => |i| {
                    if (self.done) return null;
                    self.done = true;
                    return if (self.f.integers.getPtr(i)) |p| p else null;
                },
                .boolean => return null,
                .keywords => {
                    while (!self.r.atEnd()) {
                        const t = self.r.tag() catch return null;
                        if (t.field == 1 and t.wire_type == .length_delimited) {
                            const s = self.r.bytes() catch return null;
                            if (self.f.keywords.getPtr(s)) |p| return p;
                        } else self.r.skip(t.wire_type) catch return null;
                    }
                    return null;
                },
                .integers => {
                    while (true) {
                        if (self.packed_r) |*pr| {
                            while (!pr.atEnd()) {
                                const v: i64 = @bitCast(pr.varint() catch return null);
                                if (self.f.integers.getPtr(v)) |p| return p;
                            }
                            self.packed_r = null;
                        }
                        if (self.r.atEnd()) return null;
                        const t = self.r.tag() catch return null;
                        switch (t.wire_type) {
                            .varint => {
                                const v: i64 = @bitCast(self.r.varint() catch return null);
                                if (t.field == 1) if (self.f.integers.getPtr(v)) |p| return p;
                            },
                            .length_delimited => {
                                if (t.field == 1) {
                                    self.packed_r = self.r.nested() catch return null;
                                } else self.r.skip(t.wire_type) catch return null;
                            },
                            else => self.r.skip(t.wire_type) catch return null,
                        }
                    }
                },
            }
        }
    };
};

// =========================================================================
// Predicates, in the shape `hnsw.Index.Filter` takes
// =========================================================================

/// A selection bitset: admits the offsets whose bit is set, below `bound`.
///
/// `bound` is the published offset count (`id_space.count()`) the selection
/// was built against. A point published after that (W11 writes during
/// searches) has no bit, and the word it would land in may hold a stale one
/// from an earlier query, so anything past the bound is refused rather than
/// read.
pub const BitsCtx = struct {
    bits: []const u64,
    bound: usize,

    pub fn pred(ctx: *const anyopaque, node: u32) bool {
        const self: *const BitsCtx = @ptrCast(@alignCast(ctx));
        return node < self.bound and testBit(self.bits, node);
    }
};

pub fn testBit(bits: []const u64, node: u32) bool {
    return bits[node / 64] & (@as(u64, 1) << @intCast(node % 64)) != 0;
}

/// Direct evaluation against the blob, for a filter no posting list answers.
pub const EvalCtx = struct {
    store: *const Store,
    filter: *const Filter,

    pub fn pred(ctx: *const anyopaque, node: u32) bool {
        const self: *const EvalCtx = @ptrCast(@alignCast(ctx));
        return self.store.matches(node, self.filter);
    }
};

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

/// `{ key: Value }` as a map entry, `Value` being one of the scalar kinds.
fn entryBytes(buf: []u8, key: []const u8, comptime value_field: u32, value: anytype) []const u8 {
    var w = wire.Writer.init(buf);
    w.writeStringField(entry_key, key) catch unreachable;
    const val = w.beginNested(entry_value, 2) catch unreachable;
    if (comptime value_field == value_string) {
        w.writeStringField(value_string, value) catch unreachable;
    } else if (comptime value_field == value_integer) {
        w.writeVarintFieldAlways(value_integer, @bitCast(@as(i64, value))) catch unreachable;
    } else if (comptime value_field == value_bool) {
        w.writeBoolField(value_bool, value) catch unreachable;
    } else {
        @compileError("unsupported value kind");
    }
    w.endNested(val) catch unreachable;
    return w.written();
}

/// A payload map with the entries in `entries`, each tagged as field 3.
fn mapBytes(buf: []u8, entries: []const []const u8) []const u8 {
    var w = wire.Writer.init(buf);
    for (entries) |e| {
        w.tag(3, .length_delimited) catch unreachable;
        w.varint(e.len) catch unreachable;
        w.raw(e) catch unreachable;
    }
    return w.written();
}

fn keywordFilter(key: []const u8, value: []const u8) Filter {
    var f = Filter{};
    f.add(.must, .{ .key = key, .match = .{ .keyword = value } }) catch unreachable;
    return f;
}

test "a payload round-trips as the bytes it arrived as, and a bare upsert clears it" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    var m: [256]u8 = undefined;
    const a = entryBytes(&e1, "a", value_string, "keyword_7");
    const b = entryBytes(&e2, "n", value_integer, -3);
    try store.set(testing.allocator, 2, mapBytes(&m, &.{ a, b }), 3);
    try testing.expect(store.has(2));
    try testing.expect(!store.has(1));
    try testing.expectEqual(@as(usize, 1), store.with_payload);

    var it = BlobIterator.init(store.get(2));
    try testing.expectEqualSlices(u8, a, it.next().?);
    try testing.expectEqualSlices(u8, b, it.next().?);
    try testing.expect(it.next() == null);

    // Overwrite with nothing: gone, as a Qdrant upsert without a payload does.
    try store.set(testing.allocator, 2, &.{}, 3);
    try testing.expect(!store.has(2));
    try testing.expectEqual(@as(usize, 0), store.with_payload);
    try testing.expectEqual(@as(usize, 0), store.get(2).len);
}

test "must, must_not and should evaluate against the blob with Qdrant's semantics" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    var e3: [64]u8 = undefined;
    var m: [256]u8 = undefined;
    const a = entryBytes(&e1, "a", value_string, "red");
    const n = entryBytes(&e2, "n", value_integer, 42);
    const b = entryBytes(&e3, "b", value_bool, true);
    try store.set(testing.allocator, 0, mapBytes(&m, &.{ a, n, b }), 3);

    try testing.expect(store.matches(0, &keywordFilter("a", "red")));
    try testing.expect(!store.matches(0, &keywordFilter("a", "blue")));
    // A key the point does not carry matches nothing.
    try testing.expect(!store.matches(0, &keywordFilter("zzz", "red")));
    // Types do not coerce: the integer 42 is not the keyword "42".
    try testing.expect(!store.matches(0, &keywordFilter("n", "42")));

    var f = Filter{};
    try f.add(.must, .{ .key = "n", .match = .{ .integer = 42 } });
    try f.add(.must, .{ .key = "b", .match = .{ .boolean = true } });
    try testing.expect(store.matches(0, &f));
    try f.add(.must_not, .{ .key = "a", .match = .{ .keyword = "red" } });
    try testing.expect(!store.matches(0, &f));

    // `should`: at least one when any is given.
    var s = Filter{};
    try s.add(.should, .{ .key = "a", .match = .{ .keyword = "blue" } });
    try testing.expect(!store.matches(0, &s));
    try s.add(.should, .{ .key = "a", .match = .{ .keyword = "red" } });
    try testing.expect(store.matches(0, &s));

    // `must_not` alone on an absent key holds; an empty filter admits all.
    var mn = Filter{};
    try mn.add(.must_not, .{ .key = "zzz", .match = .{ .keyword = "x" } });
    try testing.expect(store.matches(0, &mn));
    try testing.expect(store.matches(1, &Filter{}));
    try testing.expect(!store.matches(1, &keywordFilter("a", "red")));
}

test "match_any over a RepeatedStrings and a list-valued field" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    // `{ tags: ["x", "y"] }`: a ListValue of two string Values.
    var ebuf: [128]u8 = undefined;
    var w = wire.Writer.init(&ebuf);
    try w.writeStringField(entry_key, "tags");
    const val = try w.beginNested(entry_value, 2);
    const list = try w.beginNested(value_list, 2);
    for ([_][]const u8{ "x", "y" }) |s| {
        const v = try w.beginNested(1, 2);
        try w.writeStringField(value_string, s);
        try w.endNested(v);
    }
    try w.endNested(list);
    try w.endNested(val);
    var m: [256]u8 = undefined;
    try store.set(testing.allocator, 0, mapBytes(&m, &.{w.written()}), 3);

    try testing.expect(store.matches(0, &keywordFilter("tags", "y")));
    try testing.expect(!store.matches(0, &keywordFilter("tags", "z")));

    // `RepeatedStrings { strings = ["q", "x"] }`.
    var rs: [32]u8 = undefined;
    var rw = wire.Writer.init(&rs);
    try rw.writeStringField(1, "q");
    try rw.writeStringField(1, "x");
    var f = Filter{};
    try f.add(.must, .{ .key = "tags", .match = .{ .keywords = rw.written() } });
    try testing.expect(store.matches(0, &f));

    // And the index sees both list elements.
    try store.createIndex(testing.allocator, "tags", .keyword, 1);
    var bits: [1]u64 = undefined;
    const sel = store.select(&keywordFilter("tags", "x"), &bits, 1).?;
    try testing.expectEqual(@as(usize, 1), sel.count);
    try testing.expect(testBit(&bits, 0));
    try testing.expectEqual(@as(usize, 1), store.select(&f, &bits, 1).?.count);
}

test "the index selects exactly the matching points, whether built before or after the writes" {
    var store = try Store.init(testing.allocator, 128);
    defer store.deinit(testing.allocator);
    var m: [128]u8 = undefined;
    var e: [64]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    // 64 points, keyword `a` = "k<i % 4>", indexed after the first half.
    for (0..64) |i| {
        if (i == 32) try store.createIndex(testing.allocator, "a", .keyword, 32);
        const s = try std.fmt.bufPrint(&nbuf, "k{d}", .{i % 4});
        try store.set(testing.allocator, @intCast(i), mapBytes(&m, &.{entryBytes(&e, "a", value_string, s)}), 3);
    }
    var bits: [2]u64 = undefined;
    const sel = store.select(&keywordFilter("a", "k1"), &bits, 64).?;
    try testing.expectEqual(@as(usize, 16), sel.count);
    for (0..64) |i| try testing.expectEqual(i % 4 == 1, testBit(&bits, @intCast(i)));

    // A value nobody has: an empty selection, not null.
    try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "k9"), &bits, 64).?.count);
    // An unindexed key: null, so the caller evaluates blobs.
    try testing.expect(store.select(&keywordFilter("zzz", "k1"), &bits, 64) == null);
    // A bound below the postings: offsets past it are not admitted.
    try testing.expectEqual(@as(usize, 4), store.select(&keywordFilter("a", "k1"), &bits, 16).?.count);

    // The same kind again is a no-op; another kind on the name is refused.
    try store.createIndex(testing.allocator, "a", .keyword, 64);
    try testing.expectError(error.IndexKindMismatch, store.createIndex(testing.allocator, "a", .integer, 64));
    try testing.expectEqual(@as(usize, 1), store.fields.items.len);
}

test "an overwrite leaves a stale posting that verification drops" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "a", .keyword, 0);
    var m: [128]u8 = undefined;
    var e: [64]u8 = undefined;
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "old")}), 3);
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "new")}), 3);
    var bits: [1]u64 = undefined;
    // The "old" posting still names offset 0; the blob says otherwise.
    try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "old"), &bits, 1).?.count);
    try testing.expectEqual(@as(usize, 1), store.select(&keywordFilter("a", "new"), &bits, 1).?.count);
    // A point whose payload was cleared drops out of every selection.
    try store.set(testing.allocator, 0, &.{}, 3);
    try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "new"), &bits, 1).?.count);
}

test "SetPayload merges by key and indexes the merged values" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "n", .integer, 0);
    var m: [256]u8 = undefined;
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    try store.set(testing.allocator, 3, mapBytes(&m, &.{
        entryBytes(&e1, "a", value_string, "red"),
        entryBytes(&e2, "n", value_integer, 1),
    }), 3);
    // Set `n` to 2 and add `c`: `a` survives, `n` is replaced, `c` appended.
    var e3: [64]u8 = undefined;
    var e4: [64]u8 = undefined;
    try store.merge(testing.allocator, 3, mapBytes(&m, &.{
        entryBytes(&e3, "n", value_integer, 2),
        entryBytes(&e4, "c", value_string, "z"),
    }), 3);
    try testing.expect(store.matches(3, &keywordFilter("a", "red")));
    try testing.expect(store.matches(3, &keywordFilter("c", "z")));
    var f1 = Filter{};
    try f1.add(.must, .{ .key = "n", .match = .{ .integer = 1 } });
    var f2 = Filter{};
    try f2.add(.must, .{ .key = "n", .match = .{ .integer = 2 } });
    try testing.expect(!store.matches(3, &f1));
    try testing.expect(store.matches(3, &f2));
    var bits: [1]u64 = undefined;
    try testing.expectEqual(@as(usize, 1), store.select(&f2, &bits, 4).?.count);
    try testing.expectEqual(@as(usize, 0), store.select(&f1, &bits, 4).?.count);
    // Exactly three entries, `a` first as it was stored.
    var it = BlobIterator.init(store.get(3));
    var n: usize = 0;
    while (it.next()) |entry| : (n += 1) {
        const kv = splitEntry(entry).?;
        const want: []const u8 = switch (n) {
            0 => "a",
            1 => "n",
            2 => "c",
            else => unreachable,
        };
        try testing.expectEqualStrings(want, kv.key);
    }
    try testing.expectEqual(@as(usize, 3), n);
}

test "blobs cross chunk boundaries without moving what readers hold" {
    var store = try Store.init(testing.allocator, 4);
    defer store.deinit(testing.allocator);
    // One entry of ~half a chunk, written three times: the third opens a
    // second chunk, and the first two stay readable where they were.
    const big = try testing.allocator.alloc(u8, chunk_size / 2 - 64);
    defer testing.allocator.free(big);
    @memset(big, 'x');
    const ebuf = try testing.allocator.alloc(u8, big.len + 64);
    defer testing.allocator.free(ebuf);
    var w = wire.Writer.init(ebuf);
    try w.writeStringField(entry_key, "blob");
    const val = try w.beginNested(entry_value, 4);
    try w.writeStringField(value_string, big);
    try w.endNested(val);
    const entry = w.written();
    const mbuf = try testing.allocator.alloc(u8, entry.len + 16);
    defer testing.allocator.free(mbuf);
    const map = mapBytes(mbuf, &.{entry});
    try store.set(testing.allocator, 0, map, 3);
    const first = store.get(0);
    try store.set(testing.allocator, 1, map, 3);
    try store.set(testing.allocator, 2, map, 3);
    try testing.expectEqual(@as(usize, 2), store.chunk_count);
    try testing.expectEqual(first.ptr, store.get(0).ptr);
    var it = BlobIterator.init(store.get(2));
    try testing.expectEqualSlices(u8, entry, it.next().?);
}
