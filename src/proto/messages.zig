//! §12, the minimal Qdrant proto subset, hand-decoded.
//!
//! §6.2: "Hand-written codecs for the ~30 messages actually needed. No generic
//! reflection, no code generator dependency, no `unknown_fields` retention.
//! Zig's comptime makes this pleasant: describe each message as a comptime
//! field table and generate encode/decode functions from it, so the fast path
//! is a switch over field tags with no indirection."
//!
//! Field numbers below are transcribed from the Qdrant 1.19 `.proto` files
//! (vendored under `proto/` for reference, deliberately not compiled). Getting
//! one wrong is a silent wire incompatibility, so each struct carries the
//! numbers as comments beside the fields and `T0` of §8.5 checks the resulting
//! bytes against a real Qdrant response.
//!
//! ## What "not supported" means here
//!
//! §1's non-goals list is long, and §2 is emphatic about how the gaps behave:
//!
//!   "Anything in this list that `bfb` can be told to emit is handled by
//!    returning a clean `UNIMPLEMENTED`, never by silently degrading."
//!
//! So decoding distinguishes three outcomes, and the type system carries the
//! distinction rather than a comment:
//!
//!  - a field we implement            -> decoded
//!  - a field we do not implement, set -> `error.Unimplemented`, mapped to a
//!                                       gRPC UNIMPLEMENTED naming the field
//!  - a field we have never heard of   -> skipped (forward compatibility)
//!
//! The middle case is the one that matters. Ignoring a `sparse` vector or a
//! `formula` query would produce a fast, wrong benchmark, which §11 calls "the
//! classic way benchmark projects become useless".

const std = @import("std");
const wire = @import("wire.zig");
const payload = @import("../core/payload.zig");
const Reader = wire.Reader;
const Writer = wire.Writer;

pub const DecodeError = wire.Error || error{
    /// A field is set that names a feature outside the scope of §1.
    Unimplemented,
    /// A required field is absent or a value is out of range.
    InvalidArgument,
};

/// Names the specific construct that was unimplemented, so the gRPC status
/// message can say which one rather than "unimplemented".
///
/// §2: "Return proper gRPC status codes in trailers, with `grpc-message` set,
/// so failures are legible rather than showing up as timeouts."
pub threadlocal var unimplemented_detail: []const u8 = "";

fn unimplemented(what: []const u8) DecodeError {
    unimplemented_detail = what;
    return DecodeError.Unimplemented;
}

/// The same for INVALID_ARGUMENT: which field, and why. Empty when the decoder
/// had nothing more specific to say than "missing or invalid field". The API
/// layer reads it once and clears it, so a detail cannot outlive the request
/// that set it and attach itself to a later, unrelated failure.
pub threadlocal var invalid_detail: []const u8 = "";

fn invalidArgument(what: []const u8) DecodeError {
    invalid_detail = what;
    return DecodeError.InvalidArgument;
}

/// The detail behind the last `InvalidArgument`, cleared on the way out so a
/// request whose decoder set none cannot be answered with a stale one from
/// an earlier request on the same worker. Every reader goes through here.
pub fn takeInvalidDetail() []const u8 {
    const detail = invalid_detail;
    invalid_detail = "";
    return detail;
}

// =========================================================================
// Health
// =========================================================================

pub const HealthCheckReply = struct {
    title: []const u8, // 1
    version: []const u8, // 2
    commit: ?[]const u8 = null, // 3, optional

    pub fn encode(self: HealthCheckReply, w: *Writer) Writer.Error!void {
        try w.writeStringField(1, self.title);
        try w.writeStringField(2, self.version);
        if (self.commit) |c| try w.writeStringField(3, c);
    }
};

// =========================================================================
// Point IDs
// =========================================================================

/// `PointId { oneof point_id_options { uint64 num = 1; string uuid = 2; } }`
///
/// §2: "**Point IDs** may be `num` (u64) or `uuid` (string), `--uuids`
/// switches. Both must map to a dense internal `u32` offset."
pub const PointId = union(enum) {
    num: u64,
    uuid: []const u8,

    pub fn decode(r: *Reader) DecodeError!PointId {
        var result: ?PointId = null;
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => result = .{ .num = try r.varint() },
                2 => result = .{ .uuid = try r.bytes() },
                else => try r.skip(t.wire_type),
            }
        }
        return result orelse DecodeError.InvalidArgument;
    }

    pub fn encode(self: PointId, w: *Writer) Writer.Error!void {
        switch (self) {
            // `Always` because a oneof member at its default value must still
            // be present on the wire, `PointId{num: 0}` is point zero, not an
            // absent ID.
            .num => |v| try w.writeVarintFieldAlways(1, v),
            .uuid => |s| {
                try w.tag(2, .length_delimited);
                try w.varint(s.len);
                try w.raw(s);
            },
        }
    }

    pub fn eql(a: PointId, b: PointId) bool {
        return switch (a) {
            .num => |x| switch (b) {
                .num => |y| x == y,
                .uuid => false,
            },
            .uuid => |x| switch (b) {
                .num => false,
                .uuid => |y| std.mem.eql(u8, x, y),
            },
        };
    }
};

// =========================================================================
// Vectors
// =========================================================================

/// `DenseVector { repeated float data = 1; }`
///
/// The decode is the fp32 fast path of §6.2, see `wire.packedFloatsInto`.
pub const DenseVector = struct {
    /// Borrowed raw bytes of the packed float run, still in the frame buffer.
    /// Kept as bytes rather than `[]const f32` because §2 warns the payload is
    /// not guaranteed 4-byte aligned.
    raw: []const u8,

    pub fn decode(r: *Reader) DecodeError!DenseVector {
        var raw: []const u8 = &.{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => {
                    if (t.wire_type != .length_delimited) {
                        // Unpacked repeated float is legal proto3 but no client
                        // emits it for this field; treat it as malformed rather
                        // than growing a slow path nothing exercises.
                        return DecodeError.InvalidArgument;
                    }
                    raw = try r.bytes();
                    if (raw.len % 4 != 0) return wire.Error.InvalidPackedLength;
                },
                else => try r.skip(t.wire_type),
            }
        }
        return .{ .raw = raw };
    }

    pub fn dim(self: DenseVector) usize {
        return self.raw.len / 4;
    }

    /// Copy into an aligned destination. One memcpy (§6.2).
    pub fn copyInto(self: DenseVector, dst: []f32) DecodeError!usize {
        const count = self.dim();
        if (count > dst.len) return DecodeError.InvalidArgument;
        @memcpy(std.mem.sliceAsBytes(dst[0..count]), self.raw);
        return count;
    }
};

/// `Vector { oneof { DenseVector dense = 1; SparseVector sparse = 2;
///                   MultiDenseVector multi_dense = 3; Document/Image/... } }`
///
/// Everything except `dense` is a §1 non-goal.
pub const Vector = struct {
    dense: DenseVector,

    // Field numbers transcribed from qdrant 1.19's `Vector` message and
    // VERIFIED against the generated `qdrant-client` code, not guessed.
    //
    // The dense variant lives at **101**, not 1. `Vector` carries a legacy
    // `repeated float data = 1` from before the oneof existed, and the modern
    // oneof was given a high tag range to avoid colliding with it. Decoding
    // tag 1 as the dense message, the obvious reading of the .proto, makes
    // every upsert from a current client fail with "vector is required", and
    // decoding tag 101 as unknown makes it fail silently instead. Both were
    // reachable here before the numbers were checked against the client.
    const f_legacy_data = 1;
    const f_legacy_indices = 2;
    const f_legacy_vectors_count = 3;
    const f_dense = 101;
    const f_sparse = 102;
    const f_multi_dense = 103;
    const f_document = 104;
    const f_image = 105;
    const f_object = 106;

    pub fn decode(r: *Reader) DecodeError!Vector {
        var dense: ?DenseVector = null;
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                // The legacy form: `repeated float data = 1`. prost packs
                // proto3 repeated scalars, so a client that still sends it
                // sends the length-delimited form; proto3 requires readers
                // accept both, so both wire types are handled.
                f_legacy_data => switch (t.wire_type) {
                    .length_delimited => {
                        const raw = try r.bytes();
                        if (raw.len % 4 != 0) return wire.Error.InvalidPackedLength;
                        dense = .{ .raw = raw };
                    },
                    .fixed32 => {
                        // Unpacked: one tag per element. Collecting these would
                        // need a scratch buffer the decoder does not own, and
                        // no current client emits them for this field, so it is
                        // refused rather than half-supported.
                        return unimplemented("unpacked repeated float in Vector.data");
                    },
                    else => return DecodeError.InvalidArgument,
                },
                f_legacy_indices, f_legacy_vectors_count => try r.skip(t.wire_type),
                f_dense => {
                    var sub = try r.nested();
                    dense = try DenseVector.decode(&sub);
                },
                // §1 non-goals: "sparse vectors (phase 2 at best), multivectors,
                // ColBERT, inference/`Document`/`Image` query variants".
                f_sparse => return unimplemented("sparse vectors"),
                f_multi_dense => return unimplemented("multi-dense (ColBERT) vectors"),
                f_document => return unimplemented("Document inference vectors"),
                f_image => return unimplemented("Image inference vectors"),
                f_object => return unimplemented("Object inference vectors"),
                else => try r.skip(t.wire_type),
            }
        }
        return .{ .dense = dense orelse return DecodeError.InvalidArgument };
    }
};

/// `Vectors { oneof { Vector vector = 1; NamedVectors vectors = 2; } }`
///
/// §3: "Named vectors: support the empty-name default plus N named spaces
/// (bfb's `--vectors-per-point` > 1 needs this), but optimise the single-space
/// case." The single unnamed vector is field 1 and is the fast path; named
/// vectors arrive as a map in field 2.
pub const NamedVector = struct {
    name: []const u8,
    vector: Vector,
};

pub const Vectors = struct {
    /// The unnamed default vector, when field 1 was used.
    single: ?Vector = null,
    /// Named vectors, when field 2 was used. Borrowed from the frame; the
    /// caller iterates and copies into arenas.
    named_raw: ?[]const u8 = null,

    pub fn decode(r: *Reader) DecodeError!Vectors {
        var out: Vectors = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => {
                    var sub = try r.nested();
                    out.single = try Vector.decode(&sub);
                },
                2 => out.named_raw = try r.bytes(),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }

    /// Iterate the `map<string, Vector>` in field 2.
    ///
    /// A protobuf map entry is a nested message with `key = 1`, `value = 2`.
    pub const NamedIterator = struct {
        r: Reader,

        pub fn next(self: *NamedIterator) DecodeError!?NamedVector {
            if (self.r.atEnd()) return null;
            const t = try self.r.tag();
            if (t.field != 1) {
                try self.r.skip(t.wire_type);
                return self.next();
            }
            var entry = try self.r.nested();
            var name: []const u8 = &.{};
            var vec: ?Vector = null;
            while (!entry.atEnd()) {
                const et = try entry.tag();
                switch (et.field) {
                    1 => name = try entry.bytes(),
                    2 => {
                        var sub = try entry.nested();
                        vec = try Vector.decode(&sub);
                    },
                    else => try entry.skip(et.wire_type),
                }
            }
            return .{ .name = name, .vector = vec orelse return DecodeError.InvalidArgument };
        }
    };

    pub fn namedIterator(self: Vectors) NamedIterator {
        return .{ .r = Reader.init(self.named_raw orelse &.{}) };
    }
};

// =========================================================================
// Repeated fields
// =========================================================================

/// Walks one repeated, length-delimited field of a message body: the
/// `points` of an upsert, the `query_points` of a batch, the `ids` of a
/// selector. Four iterators and two counters used to be five copies of this
/// loop, differing only in the field number and what they did with the bytes.
pub const RepeatedField = struct {
    r: Reader,
    field: u32,

    pub fn init(body: []const u8, field: u32) RepeatedField {
        return .{ .r = Reader.init(body), .field = field };
    }

    /// The next entry as a nested reader (depth-tracked), or null at the end.
    pub fn nextNested(self: *RepeatedField) DecodeError!?Reader {
        while (!self.r.atEnd()) {
            const t = try self.r.tag();
            if (t.field == self.field and t.wire_type == .length_delimited) return try self.r.nested();
            try self.r.skip(t.wire_type);
        }
        return null;
    }

    /// Pass over `n` entries by their length, without decoding them: reaching
    /// entry `i` costs `i` length reads rather than `i` decodes, which is
    /// what keeps a fanned-out batch linear (`api.handlers` `BatchJob`).
    pub fn skip(self: *RepeatedField, n: usize) DecodeError!void {
        var left = n;
        while (left > 0 and !self.r.atEnd()) {
            const t = try self.r.tag();
            if (t.field == self.field and t.wire_type == .length_delimited) {
                _ = try self.r.bytes();
                left -= 1;
                continue;
            }
            try self.r.skip(t.wire_type);
        }
    }

    /// How many entries there are, without decoding them.
    pub fn count(self: *RepeatedField) DecodeError!usize {
        var n: usize = 0;
        while (!self.r.atEnd()) {
            const t = try self.r.tag();
            if (t.field == self.field and t.wire_type == .length_delimited) {
                _ = try self.r.bytes();
                n += 1;
                continue;
            }
            try self.r.skip(t.wire_type);
        }
        return n;
    }
};

// =========================================================================
// Upsert
// =========================================================================

/// `PointStruct { PointId id = 1; map<string, Value> payload = 3; Vectors vectors = 4; }`
pub const PointStruct = struct {
    id: PointId,
    vectors: Vectors,
    /// Borrowed, undecoded: the bytes of the message from the first `payload`
    /// map entry to the end of the last, tags included, so
    /// `payload.WireEntries` can walk them. The store keeps entries as they
    /// arrived (§3: an opaque blob), so nothing here decodes a `Value`. It
    /// used to be `r.bytes()` of *one* entry, which for a two-key payload was
    /// the second key only.
    payload_raw: ?[]const u8 = null,

    /// `map<string, Value> payload = 3`.
    pub const payload_field: u32 = 3;

    pub fn decode(r: *Reader) DecodeError!PointStruct {
        var id: ?PointId = null;
        var vectors: ?Vectors = null;
        var payload_start: ?usize = null;
        var payload_end: usize = 0;

        while (!r.atEnd()) {
            const before = r.pos;
            const t = try r.tag();
            switch (t.field) {
                1 => {
                    var sub = try r.nested();
                    id = try PointId.decode(&sub);
                },
                payload_field => {
                    _ = try r.bytes();
                    if (payload_start == null) payload_start = before;
                    payload_end = r.pos;
                },
                4 => {
                    var sub = try r.nested();
                    vectors = try Vectors.decode(&sub);
                },
                else => try r.skip(t.wire_type),
            }
        }
        return .{
            .id = id orelse return DecodeError.InvalidArgument,
            .vectors = vectors orelse return DecodeError.InvalidArgument,
            .payload_raw = if (payload_start) |s| r.buf[s..payload_end] else null,
        };
    }
};

// =========================================================================
// Filters (§12 phase 2+: `Filter`/`Condition`/`FieldCondition`/`Match`)
// =========================================================================

/// Decode a `Filter { should = 1; must = 2; must_not = 3; MinShould
/// min_should = 4; }` (VERIFIED, points.proto) into the shape the payload
/// store evaluates. Borrows the bytes.
///
/// Every construct outside the keyword/integer/boolean `Match` on a
/// `FieldCondition` is refused *by name* rather than ignored: a filter
/// condition the engine skipped would return points the client asked not to
/// see, which is the silent degradation §1 forbids. Nested `Filter`s in a
/// `Condition` are flattened one level when they carry only `must`, since
/// `must(must(a), must(b))` is `must(a, b)`.
pub fn decodeFilter(bytes: []const u8) DecodeError!payload.Filter {
    var out = payload.Filter{};
    var r = Reader.init(bytes);
    try decodeFilterInto(&r, &out, null);
    return out;
}

fn decodeFilterInto(r: *Reader, out: *payload.Filter, forced: ?payload.Filter.Clause) DecodeError!void {
    while (!r.atEnd()) {
        const t = try r.tag();
        const clause: payload.Filter.Clause = switch (t.field) {
            1 => .should,
            2 => .must,
            3 => .must_not,
            4 => return unimplemented("Filter.min_should"),
            else => {
                try r.skip(t.wire_type);
                continue;
            },
        };
        var cond = try r.nested();
        // A nested `must` inside a `must` is the same clause; anything else
        // would need real boolean composition.
        const target = if (forced) |f| blk: {
            if (clause != .must) return unimplemented("a nested Filter with should/must_not conditions");
            break :blk f;
        } else clause;
        try decodeCondition(&cond, out, target);
    }
}

fn decodeCondition(r: *Reader, out: *payload.Filter, clause: payload.Filter.Clause) DecodeError!void {
    while (!r.atEnd()) {
        const t = try r.tag();
        switch (t.field) {
            1 => {
                var fc = try r.nested();
                const c = try decodeFieldCondition(&fc);
                out.add(clause, c) catch return invalidArgument("more than 16 conditions in one clause");
            },
            2 => return unimplemented("is_empty conditions"),
            3 => return unimplemented("has_id conditions"),
            4 => {
                if (clause != .must) return unimplemented("a nested Filter under should/must_not");
                var inner = try r.nested();
                try decodeFilterInto(&inner, out, .must);
            },
            5 => return unimplemented("is_null conditions"),
            6 => return unimplemented("nested (array element) conditions"),
            7 => return unimplemented("has_vector conditions"),
            8 => return unimplemented("slice conditions"),
            else => try r.skip(t.wire_type),
        }
    }
}

/// `FieldCondition { key = 1; Match match = 2; Range range = 3; ... }`.
fn decodeFieldCondition(r: *Reader) DecodeError!payload.Condition {
    var key: ?[]const u8 = null;
    var match: ?payload.Match = null;
    while (!r.atEnd()) {
        const t = try r.tag();
        switch (t.field) {
            1 => key = try r.bytes(),
            2 => {
                var m = try r.nested();
                match = try decodeMatch(&m);
            },
            3 => return unimplemented("range conditions"),
            4, 5, 7 => return unimplemented("geo conditions (§1 non-goal)"),
            6 => return unimplemented("values_count conditions"),
            8 => return unimplemented("datetime_range conditions"),
            9 => return unimplemented("FieldCondition.is_empty"),
            10 => return unimplemented("FieldCondition.is_null"),
            else => try r.skip(t.wire_type),
        }
    }
    return .{
        .key = key orelse return invalidArgument("FieldCondition without a key"),
        .match = match orelse return invalidArgument("FieldCondition without a match (only match conditions are implemented)"),
    };
}

/// `Match { keyword = 1; integer = 2; boolean = 3; text = 4; keywords = 5;
/// integers = 6; except_integers = 7; except_keywords = 8; phrase = 9;
/// text_any = 10; prefix = 11; }`.
fn decodeMatch(r: *Reader) DecodeError!payload.Match {
    var out: ?payload.Match = null;
    while (!r.atEnd()) {
        const t = try r.tag();
        out = switch (t.field) {
            1 => .{ .keyword = try r.bytes() },
            2 => .{ .integer = @bitCast(try r.varint()) },
            3 => .{ .boolean = try r.boolean() },
            4, 9, 10 => return unimplemented("full-text match (§1 non-goal)"),
            5 => .{ .keywords = try r.bytes() },
            6 => .{ .integers = try r.bytes() },
            7, 8 => return unimplemented("except_* matches"),
            11 => return unimplemented("prefix match"),
            else => {
                try r.skip(t.wire_type);
                continue;
            },
        };
    }
    return out orelse invalidArgument("Match without a value");
}

/// `UpsertPoints { string collection_name = 1; bool wait = 2;
///                 repeated PointStruct points = 3; WriteOrdering ordering = 4;
///                 ShardKeySelector shard_key_selector = 5; }`
///
/// Points are **not** decoded eagerly into a list. `points_raw` is the
/// remaining frame bytes and the handler iterates them, decoding each point
/// straight into the arena, §6.2: "Upsert decode writes **directly into final
/// storage**, the point's vector lands in `vectors.bin`'s mapped arena, not in
/// an intermediate `Vec<f32>`."
/// §12 phase 2: `ScrollPoints`.
///
/// Field numbers verified against qdrant-client 1.19's generated `qdrant.rs`
/// rather than recalled, `offset` is **3** and `limit` is **4**, which is one
/// lower than the ordering in the `.proto` text suggests. A wrong number here
/// decodes as an unknown field and is skipped, so the request silently loses
/// its cursor and every page is page one. That failure looks like a working
/// scroll on a small collection.
pub const ScrollPoints = struct {
    collection_name: []const u8 = &.{},
    /// Cursor: start at this id, inclusive. Absent means start at the
    /// beginning.
    offset: ?PointId = null,
    /// `optional uint32 limit = 4`, `#[validate(range(min = 1))]`, and
    /// `ScrollRequestInternal::default_limit` is 10 (VERIFIED, qdrant 1.19
    /// `lib/api/src/grpc/qdrant.rs`, `lib/shard/src/scroll.rs`).
    /// An explicit 0 used to be accepted and answered with an empty page that
    /// still carried `next_page_offset`, which is a pager that never ends.
    limit: u32 = 10,
    /// Borrowed filter bytes; `filter()` decodes them.
    filter_raw: ?[]const u8 = null,
    with_payload: bool = false,
    with_vectors: bool = false,
    has_order_by: bool = false,
    has_shard_key: bool = false,

    pub fn decode(r: *Reader) DecodeError!ScrollPoints {
        var out: ScrollPoints = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.collection_name = try r.bytes(),
                2 => out.filter_raw = try r.bytes(),
                3 => {
                    var sub = try r.nested();
                    out.offset = try PointId.decode(&sub);
                },
                4 => {
                    // Range-check *before* narrowing: `@truncate` turned a
                    // varint of 2^32 into exactly the zero the line before
                    // it had just refused.
                    const v = try r.varint();
                    if (v == 0) return invalidArgument("limit must be at least 1 (Qdrant validates `range(min = 1)`)");
                    if (v > std.math.maxInt(u32)) return invalidArgument("limit exceeds uint32");
                    out.limit = @intCast(v);
                },
                // The same rule as `QueryPoints`: `enable` answers, an
                // include/exclude selector is refused by name rather than
                // answered with every field.
                6 => out.with_payload = try decodeWithPayloadSelector(r),
                7 => out.with_vectors = try decodeWithVectorsSelector(r),
                9 => {
                    _ = try r.bytes();
                    out.has_shard_key = true;
                },
                10 => {
                    _ = try r.bytes();
                    out.has_order_by = true;
                },
                // 8 read_consistency, 11 timeout: accepted and ignored.
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }

    /// As `QueryPoints.filter`.
    pub fn filter(self: ScrollPoints) DecodeError!?payload.Filter {
        const raw = self.filter_raw orelse return null;
        if (raw.len == 0) return null;
        const f = try decodeFilter(raw);
        return if (f.isEmpty()) null else f;
    }
};

/// `WithPayloadSelector` / `WithVectorsSelector` both put a bool `enable` at
/// field 1; the include/exclude variants are separate fields. Any of them means
/// the client wants data we do not store, so "was anything requested" is the
/// only distinction that matters here.
fn decodeSelectorEnabled(r: *Reader) DecodeError!bool {
    var any = false;
    while (!r.atEnd()) {
        const t = try r.tag();
        switch (t.field) {
            1 => any = try r.boolean(),
            else => {
                try r.skip(t.wire_type);
                any = true;
            },
        }
    }
    return any;
}

/// §12 phase 2: `ScrollResponse{next_page_offset=1, result=2, time=3}`.
pub const RetrievedPoint = struct {
    id: PointId,
    /// The point's framed payload blob (`payload.Store.get`), re-tagged as
    /// `map<string, Value> payload = 2`. Empty for none.
    payload: []const u8 = &.{},

    pub fn encode(self: RetrievedPoint, w: *Writer) Writer.Error!void {
        const n = try w.beginNested(1, 2);
        try self.id.encode(w);
        try w.endNested(n);
        try writePayloadEntries(w, 2, self.payload);
    }
};

/// Write a stored blob's entries as the map field `field`: one
/// length-delimited entry per stored entry, bytes as they arrived.
pub fn writePayloadEntries(w: *Writer, field: u32, blob: []const u8) Writer.Error!void {
    var it = payload.BlobIterator.init(blob);
    while (it.next()) |entry| {
        try w.tag(field, .length_delimited);
        try w.varint(entry.len);
        try w.raw(entry);
    }
}

/// Upper bound on what `writePayloadEntries` emits for `blob`: each entry's
/// framed length plus one tag byte, and an entry is at least one byte long.
pub fn payloadEncodedBound(blob: []const u8) usize {
    return 2 * blob.len;
}

pub const UpsertPoints = struct {
    collection_name: []const u8 = &.{},
    /// §2: "**`wait` on upsert.** `--wait-on-upsert` sets `wait: true`. Honour
    /// it as 'visible to subsequent reads', not as 'fsynced'."
    wait: bool = false,
    body: []const u8 = &.{},

    pub fn decode(r: *Reader) DecodeError!UpsertPoints {
        var out: UpsertPoints = .{};
        out.body = r.buf;
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.collection_name = try r.bytes(),
                2 => out.wait = try r.boolean(),
                3 => try r.skip(t.wire_type), // iterated separately
                4 => try r.skip(t.wire_type), // WriteOrdering: accepted, ignored
                // Fields that change what an upsert *does*, refused by name
                // rather than skipped: `update_mode: update_only` answered
                // with an unconditional upsert and `Completed` inserted the
                // points the client asked not to insert (VERIFIED numbers,
                // points.proto: shard_key_selector = 5, update_filter = 6,
                // update_mode = 8).
                5 => return unimplemented("sharding (§1 non-goal)"),
                6 => return unimplemented("UpsertPoints.update_filter"),
                8 => return unimplemented("UpsertPoints.update_mode"),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }

    /// Iterate the `points` field without materialising a list.
    pub const PointIterator = struct {
        inner: RepeatedField,

        pub fn next(self: *PointIterator) DecodeError!?PointStruct {
            var sub = (try self.inner.nextNested()) orelse return null;
            return try PointStruct.decode(&sub);
        }
    };

    pub fn pointIterator(self: UpsertPoints) PointIterator {
        return .{ .inner = RepeatedField.init(self.body, 3) };
    }
};

// =========================================================================
// Query
// =========================================================================

/// `SearchParams { uint64 hnsw_ef = 1; bool exact = 2;
///                 QuantizationSearchParams quantization = 3;
///                 bool indexed_only = 4; }`
pub const SearchParams = struct {
    hnsw_ef: ?u64 = null,
    exact: bool = false,
    indexed_only: bool = false,
    quantization: QuantizationSearchParams = .{},

    pub fn decode(r: *Reader) DecodeError!SearchParams {
        var out: SearchParams = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.hnsw_ef = try r.varint(),
                2 => out.exact = try r.boolean(),
                3 => {
                    var sub = try r.nested();
                    out.quantization = try QuantizationSearchParams.decode(&sub);
                },
                4 => out.indexed_only = try r.boolean(),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }
};

/// `QuantizationSearchParams { bool ignore = 1; bool rescore = 2;
///                             double oversampling = 3; }`
///
/// Note `rescore` and `oversampling` are proto3 `optional` in Qdrant, so
/// absence is meaningful: absent means "use the collection default", not
/// "false"/"1.0". Collapsing that distinction would silently change what
/// `--quantization-rescore` does.
pub const QuantizationSearchParams = struct {
    ignore: ?bool = null,
    rescore: ?bool = null,
    oversampling: ?f64 = null,

    pub fn decode(r: *Reader) DecodeError!QuantizationSearchParams {
        var out: QuantizationSearchParams = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.ignore = try r.boolean(),
                2 => out.rescore = try r.boolean(),
                3 => out.oversampling = try r.double(),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }
};

/// `VectorInput { oneof { PointId id = 1; DenseVector dense = 2;
///                        SparseVector sparse = 3; MultiDenseVector multi_dense = 4;
///                        Document = 5; Image = 6; InferenceObject = 7; } }`
///
/// VERIFIED against the generated client: `id` is 1 and `dense` is **2**. The
/// natural guess, dense first, is wrong, and gets every query rejected as
/// "query.nearest.dense is required" while the vector sits in the frame under
/// a tag the decoder skipped.
pub const VectorInput = struct {
    dense: ?DenseVector = null,
    /// Query-by-existing-point-id. bfb's `--uuid-query` path pre-fetches IDs
    /// via Scroll and then queries by vector, so this is not on the critical
    /// path, but recognising it lets us reject it by name.
    id: ?PointId = null,

    pub fn decode(r: *Reader) DecodeError!VectorInput {
        var out: VectorInput = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => {
                    var sub = try r.nested();
                    out.id = try PointId.decode(&sub);
                },
                2 => {
                    var sub = try r.nested();
                    out.dense = try DenseVector.decode(&sub);
                },
                3 => return unimplemented("sparse query vectors"),
                4 => return unimplemented("multi-dense query vectors"),
                5 => return unimplemented("Document inference queries"),
                6 => return unimplemented("Image inference queries"),
                7 => return unimplemented("Object inference queries"),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }
};

/// `Query { oneof variant { VectorInput nearest = 1; RecommendInput recommend = 2;
///          DiscoverInput discover = 3; ContextInput context = 4;
///          OrderByInput order_by = 5; Fusion fusion = 6; uint32 sample = 7;
///          Formula formula = 8; NearestInputWithMmr nearest_with_mmr = 9; } }`
///
/// §2: "We need `QueryPoints` with `query = Query{ nearest: VectorInput{ dense } }`
/// ... Everything else in the `Query` oneof → `UNIMPLEMENTED`."
pub const Query = struct {
    nearest: VectorInput,

    pub fn decode(r: *Reader) DecodeError!Query {
        var nearest: ?VectorInput = null;
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => {
                    var sub = try r.nested();
                    nearest = try VectorInput.decode(&sub);
                },
                2 => return unimplemented("recommend queries"),
                3 => return unimplemented("discover queries"),
                4 => return unimplemented("context queries"),
                5 => return unimplemented("order_by queries"),
                // §1 non-goals: "RRF/fusion/MMR/formula queries".
                6 => return unimplemented("fusion (RRF) queries"),
                7 => return unimplemented("sample queries"),
                8 => return unimplemented("formula queries"),
                9 => return unimplemented("MMR queries"),
                10 => return unimplemented("RRF fusion queries"),
                11 => return unimplemented("relevance feedback queries"),
                else => try r.skip(t.wire_type),
            }
        }
        return .{ .nearest = nearest orelse return DecodeError.InvalidArgument };
    }
};

/// `QueryPoints { string collection_name = 1; repeated PrefetchQuery prefetch = 2;
///                Query query = 3; string using = 4; Filter filter = 5;
///                SearchParams params = 6; float score_threshold = 7; uint64 limit = 8;
///                uint64 offset = 9; WithVectorsSelector with_vectors = 10;
///                WithPayloadSelector with_payload = 11; ... shard_key_selector = 13; }`
/// (VERIFIED, points.proto; the constants below are the source of truth and
/// this comment used to disagree with them.)
pub const QueryPoints = struct {
    collection_name: []const u8 = &.{},
    query: ?Query = null,
    using: ?[]const u8 = null,
    /// Borrowed filter bytes; `filter()` decodes them (§2 phase 3, M7). A
    /// condition the decoder cannot evaluate is refused by name rather than
    /// ignored, because ignoring it would return more results than asked for
    /// and quietly invalidate W12.
    filter_raw: ?[]const u8 = null,
    params: SearchParams = .{},
    limit: u64 = 10,
    offset: u64 = 0,
    with_payload: bool = false,
    with_vectors: bool = false,
    score_threshold: ?f32 = null,

    // VERIFIED against the generated client. The numbers are not contiguous
    // and not in the order the message reads: `filter` is 5 (not 2),
    // `score_threshold` is 7, `limit` is 8, `offset` is 9, `with_vectors` is
    // 10 and `with_payload` is 11. Guessing them produced a decoder that
    // accepted every request and silently used a default limit of 10 with no
    // filter, fast, plausible, and wrong.
    const f_collection_name = 1;
    const f_prefetch = 2;
    const f_query = 3;
    const f_using = 4;
    const f_filter = 5;
    const f_params = 6;
    const f_score_threshold = 7;
    const f_limit = 8;
    const f_offset = 9;
    const f_with_vectors = 10;
    const f_with_payload = 11;
    const f_shard_key_selector = 13;

    pub fn decode(r: *Reader) DecodeError!QueryPoints {
        var out: QueryPoints = .{};
        var saw_limit = false;
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                f_collection_name => out.collection_name = try r.bytes(),
                // §1 non-goals: prefetch drives fusion/multi-stage queries.
                f_prefetch => return unimplemented("prefetch (multi-stage) queries"),
                f_query => {
                    var sub = try r.nested();
                    out.query = try Query.decode(&sub);
                },
                f_using => out.using = try r.bytes(),
                f_filter => out.filter_raw = try r.bytes(),
                f_params => {
                    var sub = try r.nested();
                    out.params = try SearchParams.decode(&sub);
                },
                f_score_threshold => out.score_threshold = try r.float(),
                f_limit => {
                    out.limit = try r.varint();
                    saw_limit = true;
                },
                f_offset => out.offset = try r.varint(),
                f_with_vectors => out.with_vectors = try decodeWithVectorsSelector(r),
                f_with_payload => out.with_payload = try decodeWithPayloadSelector(r),
                // Refused as `Scroll` refuses it: a query scoped to a shard
                // answered from all data is the silent degradation §1 forbids.
                f_shard_key_selector => return unimplemented("sharding (§1 non-goal)"),
                else => try r.skip(t.wire_type),
            }
        }
        // `limit` is `optional uint64` (VERIFIED in points.proto), so it *has*
        // presence: an absent field is Qdrant's default of 10, but an explicit
        // zero is a value the client sent, and Qdrant validates it with
        // `range(min = 1)` and answers INVALID_ARGUMENT. An earlier version
        // mapped 0 to 10 on the theory that proto3 scalars have no presence,
        // which is true of plain scalars and false of `optional` ones, and it
        // turned a request Qdrant refuses into one that quietly returned ten
        // results.
        if (!saw_limit) out.limit = 10;
        if (out.limit == 0) return invalidArgument("limit must be at least 1 (Qdrant validates `range(min = 1)`)");
        return out;
    }

    /// The query's filter, or null for none. An empty `Filter{}` is no
    /// filter, as Qdrant reads it, and so is one whose clauses are all empty.
    pub fn filter(self: QueryPoints) DecodeError!?payload.Filter {
        const raw = self.filter_raw orelse return null;
        if (raw.len == 0) return null;
        const f = try decodeFilter(raw);
        return if (f.isEmpty()) null else f;
    }
};

/// `WithPayloadSelector { oneof { bool enable = 1; PayloadIncludeSelector include = 2;
///                                PayloadExcludeSelector exclude = 3; } }`
fn decodeWithPayloadSelector(r: *Reader) DecodeError!bool {
    var sub = try r.nested();
    var enable = false;
    while (!sub.atEnd()) {
        const t = try sub.tag();
        switch (t.field) {
            1 => enable = try sub.boolean(),
            2, 3 => return unimplemented("payload include/exclude selectors"),
            else => try sub.skip(t.wire_type),
        }
    }
    return enable;
}

fn decodeWithVectorsSelector(r: *Reader) DecodeError!bool {
    var sub = try r.nested();
    var enable = false;
    while (!sub.atEnd()) {
        const t = try sub.tag();
        switch (t.field) {
            1 => enable = try sub.boolean(),
            2 => return unimplemented("named vector output selectors"),
            else => try sub.skip(t.wire_type),
        }
    }
    return enable;
}

/// `QueryBatchPoints { string collection_name = 1; repeated QueryPoints query_points = 2;
///                     ReadConsistency read_consistency = 3; uint64 timeout = 4; }`
///
/// §2: "**`bfb` searches via `QueryBatch`, not `Search`.**"
pub const QueryBatchPoints = struct {
    collection_name: []const u8 = &.{},
    body: []const u8 = &.{},

    pub fn decode(r: *Reader) DecodeError!QueryBatchPoints {
        var out: QueryBatchPoints = .{};
        out.body = r.buf;
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.collection_name = try r.bytes(),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }

    pub const QueryIterator = struct {
        inner: RepeatedField,

        pub fn next(self: *QueryIterator) DecodeError!?QueryPoints {
            var sub = (try self.inner.nextNested()) orelse return null;
            return try QueryPoints.decode(&sub);
        }

        /// Skip `n` queries without decoding them (`RepeatedField.skip`).
        pub fn skip(self: *QueryIterator, n: usize) DecodeError!void {
            return self.inner.skip(n);
        }
    };

    pub fn queryIterator(self: QueryBatchPoints) QueryIterator {
        return .{ .inner = RepeatedField.init(self.body, 2) };
    }

    /// Count the queries without decoding them, so the response buffer can be
    /// pre-sized. §6.2: "pre-size it from `limit × batch_size`."
    pub fn countQueries(self: QueryBatchPoints) DecodeError!usize {
        var it = RepeatedField.init(self.body, 2);
        return it.count();
    }
};

// =========================================================================
// Payload index and SetPayload (§2 phases 2 and 3)
// =========================================================================

/// `CreateFieldIndexCollection { collection_name = 1; wait = 2; field_name = 3;
/// FieldType field_type = 4; PayloadIndexParams field_index_params = 5;
/// ordering = 6; timeout = 7; }` (VERIFIED, points.proto).
pub const CreateFieldIndexCollection = struct {
    collection_name: []const u8 = &.{},
    wait: bool = false,
    field_name: []const u8 = &.{},
    /// `FieldType`: absent means keyword, which is what bfb sends for `-k`.
    field_type: u64 = 0,

    pub fn decode(r: *Reader) DecodeError!CreateFieldIndexCollection {
        var out: CreateFieldIndexCollection = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.collection_name = try r.bytes(),
                2 => out.wait = try r.boolean(),
                3 => out.field_name = try r.bytes(),
                4 => out.field_type = try r.varint(),
                // `KeywordIndexParams { is_tenant, on_disk, enable_hnsw,
                // prefix, memory }` and the integer twin: tuning of an index
                // whose only representation here is a posting list. Accepted
                // and ignored, like `WriteOrdering`.
                5, 6, 7 => try r.skip(t.wire_type),
                else => try r.skip(t.wire_type),
            }
        }
        if (out.field_name.len == 0) return invalidArgument("field_name is required");
        return out;
    }
};

/// `SetPayloadPoints { collection_name = 1; wait = 2; map<string, Value>
/// payload = 3; PointsSelector points_selector = 5; ordering = 6;
/// shard_key_selector = 7; key = 8; timeout = 9; }` (VERIFIED, points.proto).
/// `PointsSelector { PointsIdsList points = 1; Filter filter = 2; }`,
/// `PointsIdsList { repeated PointId ids = 1; }`.
pub const SetPayloadPoints = struct {
    collection_name: []const u8 = &.{},
    wait: bool = false,
    /// As `PointStruct.payload_raw`: the span of the map entries.
    payload_raw: ?[]const u8 = null,
    /// `PointsIdsList` bytes, when the selector is a list of ids.
    ids_raw: ?[]const u8 = null,
    /// `Filter` bytes, when the selector is a filter.
    filter_raw: ?[]const u8 = null,

    pub const payload_field: u32 = 3;

    pub fn decode(r: *Reader) DecodeError!SetPayloadPoints {
        var out: SetPayloadPoints = .{};
        var payload_start: ?usize = null;
        var payload_end: usize = 0;
        while (!r.atEnd()) {
            const before = r.pos;
            const t = try r.tag();
            switch (t.field) {
                1 => out.collection_name = try r.bytes(),
                2 => out.wait = try r.boolean(),
                payload_field => {
                    _ = try r.bytes();
                    if (payload_start == null) payload_start = before;
                    payload_end = r.pos;
                },
                5 => {
                    var sel = try r.nested();
                    while (!sel.atEnd()) {
                        const st = try sel.tag();
                        switch (st.field) {
                            1 => out.ids_raw = try sel.bytes(),
                            2 => out.filter_raw = try sel.bytes(),
                            else => try sel.skip(st.wire_type),
                        }
                    }
                },
                7 => return unimplemented("sharding (§1 non-goal)"),
                // A JSON path into the payload: this store is opaque blobs.
                8 => return unimplemented("SetPayload.key (a nested payload path)"),
                else => try r.skip(t.wire_type),
            }
        }
        out.payload_raw = if (payload_start) |s| r.buf[s..payload_end] else null;
        return out;
    }

    /// The ids of a `PointsIdsList` selector.
    pub const IdIterator = struct {
        inner: RepeatedField,

        pub fn next(self: *IdIterator) DecodeError!?PointId {
            var sub = (try self.inner.nextNested()) orelse return null;
            return try PointId.decode(&sub);
        }
    };

    pub fn idIterator(self: SetPayloadPoints) IdIterator {
        return .{ .inner = RepeatedField.init(self.ids_raw orelse &.{}, 1) };
    }
};

// =========================================================================
// Responses
// =========================================================================

/// `ScoredPoint { PointId id = 1; map<string,Value> payload = 2; float score = 3;
///                uint64 version = 5; Vectors vectors = 6; ... }`
pub const ScoredPoint = struct {
    id: PointId,
    score: f32,
    version: u64 = 0,
    /// As `RetrievedPoint.payload`; `with_payload` fills it.
    payload: []const u8 = &.{},

    pub fn encode(self: ScoredPoint, w: *Writer) Writer.Error!void {
        // `maxEncodedSize` is a *promise* the batch sizer in `handlers.zig`
        // spends: `queryBatchFits` multiplies it by the point count to decide
        // whether a response can be encoded at all, and answers before a byte
        // is written. If this function ever writes more than the promise, that
        // sizer says yes to what does not fit and the encode fails mid-response
        // — after the headers, where there is no status left to send. The two
        // are hand-kept in different scopes, so the relation is stated here
        // rather than trusted.
        const start = w.pos;
        defer std.debug.assert(w.pos - start <= maxEncodedSize() + payloadEncodedBound(self.payload));
        {
            const n = try w.beginNested(1, 2);
            try self.id.encode(w);
            try w.endNested(n);
        }
        try writePayloadEntries(w, 2, self.payload);
        // `Always`: a score of exactly 0.0 is a legitimate result (orthogonal
        // vectors under dot, or an exact match under Euclid), and eliding it
        // would make the client read it as a missing field.
        try w.writeFloatFieldAlways(3, self.score);
        try w.writeVarintField(5, self.version);
    }

    /// Upper bound on encoded size, for response buffer sizing.
    pub fn maxEncodedSize() usize {
        // tag+len for the nested id (2+2), id payload (1 tag + 10 varint, or a
        // UUID string up to 36 bytes + 2), score (1+4), version (1+10).
        return 4 + 38 + 5 + 11;
    }
};

/// `BatchResult { repeated ScoredPoint result = 1; }`
/// `QueryBatchResponse { repeated BatchResult result = 1; double time = 2; }`
///
/// §2: "the response is `QueryBatchResponse { repeated BatchResult result }`,
/// and `time` (seconds, f64) is read by bfb as the *server-side* timing, it
/// feeds the 'server_timings' histogram, so it must be measured honestly at the
/// RPC boundary."
pub const QueryBatchResponseWriter = struct {
    w: *Writer,

    pub fn init(w: *Writer) QueryBatchResponseWriter {
        return .{ .w = w };
    }

    /// Begin one `BatchResult`. A 4-byte length reservation covers up to
    /// 2^28 bytes, far beyond any single batch result.
    pub fn beginBatch(self: *QueryBatchResponseWriter) Writer.Error!Writer.Nested {
        return self.w.beginNested(1, 4);
    }

    pub fn endBatch(self: *QueryBatchResponseWriter, n: Writer.Nested) Writer.Error!void {
        return self.w.endNested(n);
    }

    pub fn point(self: *QueryBatchResponseWriter, p: ScoredPoint) Writer.Error!void {
        // Two length bytes hold 16 KiB; a payload can be larger, and a body
        // that outgrows its reservation is an encode failure mid-response.
        const n = try self.w.beginNested(1, if (p.payload.len > 8000) 3 else 2);
        try p.encode(self.w);
        try self.w.endNested(n);
    }

    /// Write the trailing `time` field. Seconds, as f64.
    pub fn finishTime(self: *QueryBatchResponseWriter, seconds: f64) Writer.Error!void {
        try self.w.writeDoubleField(2, seconds);
    }
};

/// `UpdateResult { uint64 operation_id = 1; UpdateStatus status = 2; }`
/// `PointsOperationResponse { UpdateResult result = 1; double time = 2; }`
pub const UpdateStatus = enum(u32) {
    unknown = 0,
    acknowledged = 1,
    completed = 2,
    clock_rejected = 3,
};

pub const PointsOperationResponse = struct {
    operation_id: u64,
    status: UpdateStatus,
    time: f64,

    pub fn encode(self: PointsOperationResponse, w: *Writer) Writer.Error!void {
        {
            const n = try w.beginNested(1, 2);
            try w.writeVarintField(1, self.operation_id);
            try w.writeVarintField(2, @intFromEnum(self.status));
            try w.endNested(n);
        }
        try w.writeDoubleField(2, self.time);
    }
};

// =========================================================================
// Collections
// =========================================================================

pub const CollectionStatus = enum(u32) {
    unknown = 0,
    /// §2: "After upload, bfb polls `collection_info` once per second and
    /// requires `status == Green` **three consecutive times**."
    green = 1,
    yellow = 2,
    red = 3,
    grey = 4,
};

// There is no `OptimizerStatus` enum here. One was declared, and it was wrong:
// qdrant's `optimizer_status` is a *message*, `{ bool ok = 1; string error = 2; }`,
// which is what `getCollectionInfo` writes. Nothing referenced the enum, so
// nothing ever contradicted it, and a declaration that misdescribes the wire
// format is worse than no declaration at all.

// `ProductQuantization.CompressionRatio` lives in `quant/quant.zig` beside the
// mode it parameterises, with its wire mapping on it, exactly as `Metric` keeps
// `fromProto`/`toProto` in `dist/metric.zig`. Nothing in `quant/` may depend on
// this file.

/// `VectorParams { uint64 size = 1; Distance distance = 2; HnswConfigDiff hnsw_config = 3;
///                 QuantizationConfig quantization_config = 4; bool on_disk = 5 [deprecated];
///                 Datatype datatype = 6; MultiVectorConfig multivector_config = 7;
///                 Memory memory = 8; }`
///
/// Both placement fields are decoded, because Qdrant still accepts both and
/// bfb's client version decides which one it sends. `on_disk` is deprecated
/// upstream in favour of `memory`, and when both arrive `memory` wins
/// (`Memory::resolve`: "the explicit parameter always wins").
///
/// Both cross this boundary as they arrive on the wire, like `Distance`:
/// `storage.Placement.resolve` interprets them, and nothing in `proto/` depends
/// on `core/`.
pub const VectorParams = struct {
    size: u64 = 0,
    distance: i32 = 0,
    /// Deprecated upstream. `null` means the client did not send it, which is
    /// not the same as `false` — absent means "use the default placement",
    /// `false` means "Cached", and collapsing the two would silently move
    /// every collection off the pinned default.
    on_disk: ?bool = null,
    /// `Memory` enum value as received; 0 (`MemoryUnknown`) is treated as
    /// absent, exactly as an unset optional proto field would be.
    memory: i32 = 0,
    datatype: i32 = 0,
    hnsw_m: ?u64 = null,
    hnsw_ef_construct: ?u64 = null,
    hnsw_full_scan_threshold: ?u64 = null,
    hnsw_placement_requested: bool = false,
    quantization_raw: ?[]const u8 = null,

    pub fn decode(r: *Reader) DecodeError!VectorParams {
        var out: VectorParams = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.size = try r.varint(),
                2 => out.distance = @bitCast(try r.varint32()),
                3 => {
                    var sub = try r.nested();
                    const h = try HnswConfigDiff.decode(&sub);
                    out.hnsw_m = h.m;
                    out.hnsw_ef_construct = h.ef_construct;
                    out.hnsw_full_scan_threshold = h.full_scan_threshold;
                    out.hnsw_placement_requested = h.placementRequested();
                },
                4 => out.quantization_raw = try r.bytes(),
                5 => out.on_disk = try r.boolean(),
                6 => out.datatype = @bitCast(try r.varint32()),
                7 => return unimplemented("multivector config"),
                8 => out.memory = @bitCast(try r.varint32()),
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }
};

pub const HnswConfigDiff = struct {
    m: ?u64 = null, // 1
    ef_construct: ?u64 = null, // 2
    full_scan_threshold: ?u64 = null, // 3
    /// `bool on_disk = 5 [deprecated]` and `Memory memory = 8`: where the graph
    /// itself lives. Decoded so a request that asks for it can be refused by
    /// name; strawmANN's graph is always pinned. §1's rule is that a construct
    /// we do not implement returns UNIMPLEMENTED rather than being ignored, and
    /// ignoring this one would report an on-disk-graph number measured on an
    /// in-memory graph.
    on_disk: ?bool = null, // 5
    memory: i32 = 0, // 8

    /// Whether the client asked for a graph placement at all. An explicit
    /// `on_disk: false` counts: it asks for `Cached`, an evictable mmap, which
    /// is not what the pinned graph does either. `memory == 0` is
    /// `MemoryUnknown`, which the proto uses for "unset".
    pub fn placementRequested(self: HnswConfigDiff) bool {
        return self.on_disk != null or self.memory != 0;
    }

    pub fn decode(r: *Reader) DecodeError!HnswConfigDiff {
        var out: HnswConfigDiff = .{};
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                1 => out.m = try r.varint(),
                2 => out.ef_construct = try r.varint(),
                3 => out.full_scan_threshold = try r.varint(),
                5 => out.on_disk = try r.boolean(),
                8 => out.memory = @bitCast(try r.varint32()),
                // Accepted and inert, each for a stated reason rather than by
                // default (§1's rule is that a knob we do not honour must not
                // change the answer):
                //   4 `max_indexing_threads`: how many cores the build uses,
                //     which is `--build-threads` here (a server flag, not a
                //     per-collection one) and does not change what a query
                //     returns;
                //   6 `payload_m`: M for the per-payload-block graphs, and
                //     there are no payloads (§2 phase 2, unbuilt), so no such graphs;
                //   7 `inline_storage`: whether vectors are stored inline
                //     with the links, a graph layout knob, and §6.5 fixes the
                //     layout.
                else => try r.skip(t.wire_type),
            }
        }
        return out;
    }
};

/// `Datatype { Default = 0; Float32 = 1; Uint8 = 2; Float16 = 3; }`
pub const Datatype = enum(i32) {
    default = 0,
    float32 = 1,
    uint8 = 2,
    float16 = 3,

    pub fn fromProto(v: i32) ?Datatype {
        return switch (v) {
            0 => .default,
            1 => .float32,
            2 => .uint8,
            3 => .float16,
            else => null,
        };
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "PointId round-trips both variants including the zero case" {
    var buf: [64]u8 = undefined;

    // num = 0 must survive: it is point zero, not an absent field.
    for ([_]PointId{ .{ .num = 0 }, .{ .num = 1 }, .{ .num = std.math.maxInt(u64) } }) |id| {
        var w = Writer.init(&buf);
        try id.encode(&w);
        var r = Reader.init(w.written());
        const got = try PointId.decode(&r);
        try testing.expect(id.eql(got));
    }

    const uuid = PointId{ .uuid = "550e8400-e29b-41d4-a716-446655440000" };
    var w2 = Writer.init(&buf);
    try uuid.encode(&w2);
    var r2 = Reader.init(w2.written());
    const got2 = try PointId.decode(&r2);
    try testing.expect(uuid.eql(got2));
}

test "DenseVector decodes via one memcpy and reports its dimension" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var buf: [256]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writePackedFloats(1, &src);

    var r = Reader.init(w.written());
    const dv = try DenseVector.decode(&r);
    try testing.expectEqual(@as(usize, 5), dv.dim());

    var dst: [8]f32 = undefined;
    const n = try dv.copyInto(&dst);
    try testing.expectEqual(@as(usize, 5), n);
    try testing.expectEqualSlices(f32, &src, dst[0..5]);

    // A vector larger than the collection's dimension must be refused.
    var tiny: [2]f32 = undefined;
    try testing.expectError(DecodeError.InvalidArgument, dv.copyInto(&tiny));
}

test "§1 non-goals produce Unimplemented naming the construct, not silence" {
    // A Vector carrying a sparse variant. VERIFIED: sparse is field 102 in the
    // `Vector` oneof, not 2, the low tags belong to the legacy flat encoding.
    var buf: [64]u8 = undefined;
    var w = Writer.init(&buf);
    const n = try w.beginNested(102, 2);
    try w.writeVarintField(1, 7);
    try w.endNested(n);

    var r = Reader.init(w.written());
    try testing.expectError(DecodeError.Unimplemented, Vector.decode(&r));
    try testing.expectEqualStrings("sparse vectors", unimplemented_detail);
}

test "every rejected Query variant names itself" {
    const cases = [_]struct { u32, []const u8 }{
        .{ 2, "recommend queries" },
        .{ 3, "discover queries" },
        .{ 4, "context queries" },
        .{ 5, "order_by queries" },
        .{ 6, "fusion (RRF) queries" },
        .{ 7, "sample queries" },
        .{ 8, "formula queries" },
        .{ 9, "MMR queries" },
    };
    var buf: [64]u8 = undefined;
    for (cases) |c| {
        var w = Writer.init(&buf);
        if (c[0] == 7) {
            // `sample` is a uint32, not a message.
            try w.writeVarintFieldAlways(c[0], 3);
        } else {
            const n = try w.beginNested(c[0], 2);
            try w.writeVarintField(1, 1);
            try w.endNested(n);
        }
        var r = Reader.init(w.written());
        try testing.expectError(DecodeError.Unimplemented, Query.decode(&r));
        try testing.expectEqualStrings(c[1], unimplemented_detail);
    }
}

test "unknown fields are skipped for forward compatibility" {
    // A DenseVector with a field 99 the client invented. Must decode fine.
    var buf: [256]u8 = undefined;
    var w = Writer.init(&buf);
    const src = [_]f32{ 1.0, 2.0 };
    try w.writePackedFloats(1, &src);
    try w.writeVarintFieldAlways(99, 12345);
    try w.writeStringField(100, "future");

    var r = Reader.init(w.written());
    const dv = try DenseVector.decode(&r);
    try testing.expectEqual(@as(usize, 2), dv.dim());
}

test "QueryPoints defaults limit to 10 when absent, and refuses an explicit zero" {
    var buf: [128]u8 = undefined;

    var w = Writer.init(&buf);
    try w.writeStringField(1, "bench");
    var r = Reader.init(w.written());
    const q = try QueryPoints.decode(&r);
    try testing.expectEqual(@as(u64, 10), q.limit);
    try testing.expectEqualStrings("bench", q.collection_name);

    // `optional uint64`: an explicit zero is present, and Qdrant's
    // `range(min = 1)` rejects it. It used to be silently mapped to 10.
    var w2 = Writer.init(&buf);
    try w2.writeVarintFieldAlways(8, 0);
    var r2 = Reader.init(w2.written());
    try testing.expectError(DecodeError.InvalidArgument, QueryPoints.decode(&r2));
    try testing.expect(std.mem.indexOf(u8, invalid_detail, "limit") != null);
    invalid_detail = "";

    // An explicit non-zero limit is honoured.
    var w3 = Writer.init(&buf);
    try w3.writeVarintFieldAlways(8, 100);
    var r3 = Reader.init(w3.written());
    const q3 = try QueryPoints.decode(&r3);
    try testing.expectEqual(@as(u64, 100), q3.limit);
}

test "QuantizationSearchParams distinguishes absent from false" {
    var buf: [64]u8 = undefined;

    var w = Writer.init(&buf);
    var r = Reader.init(w.written());
    const empty = try QuantizationSearchParams.decode(&r);
    try testing.expectEqual(@as(?bool, null), empty.rescore);
    try testing.expectEqual(@as(?f64, null), empty.oversampling);

    // rescore = false explicitly set is *not* the same as absent.
    var w2 = Writer.init(&buf);
    try w2.writeVarintFieldAlways(2, 0);
    var r2 = Reader.init(w2.written());
    const explicit = try QuantizationSearchParams.decode(&r2);
    try testing.expectEqual(@as(?bool, false), explicit.rescore);
}

test "SearchParams decodes the flags bfb sets" {
    var buf: [128]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writeVarintField(1, 128); // hnsw_ef
    try w.writeBoolField(2, true); // exact
    try w.writeBoolField(4, true); // indexed_only
    const n = try w.beginNested(3, 2);
    try w.writeBoolField(2, true); // rescore
    try w.writeDoubleField(3, 4.0); // oversampling
    try w.endNested(n);

    var r = Reader.init(w.written());
    const p = try SearchParams.decode(&r);
    try testing.expectEqual(@as(?u64, 128), p.hnsw_ef);
    try testing.expect(p.exact);
    try testing.expect(p.indexed_only);
    try testing.expectEqual(@as(?bool, true), p.quantization.rescore);
    try testing.expectEqual(@as(?f64, 4.0), p.quantization.oversampling);
}

test "ScoredPoint encodes a zero score rather than eliding it" {
    var buf: [128]u8 = undefined;
    var w = Writer.init(&buf);
    const p = ScoredPoint{ .id = .{ .num = 7 }, .score = 0.0, .version = 0 };
    try p.encode(&w);

    var r = Reader.init(w.written());
    var saw_score = false;
    var saw_id = false;
    while (!r.atEnd()) {
        const t = try r.tag();
        switch (t.field) {
            1 => {
                var sub = try r.nested();
                const id = try PointId.decode(&sub);
                try testing.expect(id.eql(.{ .num = 7 }));
                saw_id = true;
            },
            3 => {
                try testing.expectEqual(@as(f32, 0.0), try r.float());
                saw_score = true;
            },
            else => try r.skip(t.wire_type),
        }
    }
    try testing.expect(saw_id);
    try testing.expect(saw_score);
}

test "UpsertPoints iterates points without materialising a list" {
    var buf: [1024]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writeStringField(1, "c");
    try w.writeBoolField(2, true);

    const vecs = [_][3]f32{ .{ 1, 2, 3 }, .{ 4, 5, 6 }, .{ 7, 8, 9 } };
    for (vecs, 0..) |v, i| {
        const pt = try w.beginNested(3, 2);
        {
            const idn = try w.beginNested(1, 2);
            try w.writeVarintFieldAlways(1, i);
            try w.endNested(idn);
        }
        {
            const vsn = try w.beginNested(4, 2); // Vectors
            const vn = try w.beginNested(1, 2); // Vectors.vector
            const dn = try w.beginNested(101, 2); // Vector.dense (VERIFIED)
            try w.writePackedFloats(1, &v);
            try w.endNested(dn);
            try w.endNested(vn);
            try w.endNested(vsn);
        }
        try w.endNested(pt);
    }

    var r = Reader.init(w.written());
    const up = try UpsertPoints.decode(&r);
    try testing.expectEqualStrings("c", up.collection_name);
    try testing.expect(up.wait);

    var it = up.pointIterator();
    var count: usize = 0;
    while (try it.next()) |p| {
        try testing.expect(p.id.eql(.{ .num = count }));
        const dv = p.vectors.single.?.dense;
        try testing.expectEqual(@as(usize, 3), dv.dim());
        var dst: [3]f32 = undefined;
        _ = try dv.copyInto(&dst);
        try testing.expectEqualSlices(f32, &vecs[count], &dst);
        count += 1;
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "QueryBatchPoints counts and iterates queries" {
    var buf: [1024]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writeStringField(1, "bench");
    for (0..4) |i| {
        const qn = try w.beginNested(2, 2);
        try w.writeVarintFieldAlways(8, 10 + i); // limit (VERIFIED: field 8)
        try w.endNested(qn);
    }

    var r = Reader.init(w.written());
    const qb = try QueryBatchPoints.decode(&r);
    try testing.expectEqualStrings("bench", qb.collection_name);
    try testing.expectEqual(@as(usize, 4), try qb.countQueries());

    var it = qb.queryIterator();
    var i: u64 = 0;
    while (try it.next()) |q| : (i += 1) {
        try testing.expectEqual(10 + i, q.limit);
    }
    try testing.expectEqual(@as(u64, 4), i);
}

test "VectorParams decodes size, distance and hnsw config" {
    var buf: [256]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writeVarintField(1, 768);
    try w.writeVarintField(2, 1); // Cosine
    const n = try w.beginNested(3, 2);
    try w.writeVarintField(1, 16); // m
    try w.writeVarintField(2, 100); // ef_construct
    try w.endNested(n);
    try w.writeVarintField(6, 1); // Float32

    var r = Reader.init(w.written());
    const vp = try VectorParams.decode(&r);
    try testing.expectEqual(@as(u64, 768), vp.size);
    try testing.expectEqual(@as(i32, 1), vp.distance);
    try testing.expectEqual(@as(?u64, 16), vp.hnsw_m);
    try testing.expectEqual(@as(?u64, 100), vp.hnsw_ef_construct);
    try testing.expectEqual(Datatype.float32, Datatype.fromProto(vp.datatype).?);
}

test "Vectors decodes named vectors as a map" {
    var buf: [512]u8 = undefined;
    var w = Writer.init(&buf);
    // Vectors.vectors = 2 (NamedVectors), whose `vectors` map is field 1.
    const outer = try w.beginNested(2, 2);
    for ([_][]const u8{ "text", "image" }, 0..) |name, i| {
        const entry = try w.beginNested(1, 2);
        try w.writeStringField(1, name);
        const vn = try w.beginNested(2, 2); // map entry value: Vector
        const dn = try w.beginNested(101, 2); // Vector.dense (VERIFIED)
        const data = [_]f32{ @floatFromInt(i), 1.0 };
        try w.writePackedFloats(1, &data);
        try w.endNested(dn);
        try w.endNested(vn);
        try w.endNested(entry);
    }
    try w.endNested(outer);

    var r = Reader.init(w.written());
    const vs = try Vectors.decode(&r);
    try testing.expect(vs.single == null);
    try testing.expect(vs.named_raw != null);

    var it = vs.namedIterator();
    var seen: usize = 0;
    while (try it.next()) |nv| : (seen += 1) {
        try testing.expectEqual(@as(usize, 2), nv.vector.dense.dim());
    }
    try testing.expectEqual(@as(usize, 2), seen);
}

test "VERIFIED field numbers: the ones that were wrong before checking" {
    // These four numbers were each guessed wrong on the first pass, and every
    // one of them produces a plausible-looking failure rather than a crash:
    //
    //   Vector.dense         101, not 1   -> "vector is required" on every upsert
    //   VectorInput.dense      2, not 1   -> "query.nearest.dense is required"
    //   QueryPoints.limit      8, not 7   -> silently defaults to 10
    //   QueryPoints.filter     5, not 2   -> a set filter is silently ignored
    //
    // The last two are the dangerous ones: they do not fail, they produce a
    // fast wrong answer. §11 calls that "the classic way benchmark projects
    // become useless". This test pins them against the generated client.
    var buf: [512]u8 = undefined;

    // Vector.dense at 101 decodes; at 1 it does not.
    {
        var w = Writer.init(&buf);
        const vn = try w.beginNested(101, 2);
        try w.writePackedFloats(1, &[_]f32{ 1, 2, 3, 4 });
        try w.endNested(vn);
        var r = Reader.init(w.written());
        const v = try Vector.decode(&r);
        try testing.expectEqual(@as(usize, 4), v.dense.dim());
    }

    // VectorInput: field 1 is the point id, field 2 is the dense vector.
    {
        var w = Writer.init(&buf);
        const dn = try w.beginNested(2, 2);
        try w.writePackedFloats(1, &[_]f32{ 5, 6 });
        try w.endNested(dn);
        var r = Reader.init(w.written());
        const vi = try VectorInput.decode(&r);
        try testing.expect(vi.dense != null);
        try testing.expectEqual(@as(usize, 2), vi.dense.?.dim());
        try testing.expect(vi.id == null);
    }
    {
        var w = Writer.init(&buf);
        const idn = try w.beginNested(1, 2);
        try w.writeVarintFieldAlways(1, 42);
        try w.endNested(idn);
        var r = Reader.init(w.written());
        const vi = try VectorInput.decode(&r);
        try testing.expect(vi.id != null);
        try testing.expect(vi.dense == null);
    }

    // QueryPoints: limit at 8, filter at 5, score_threshold at 7, offset at 9.
    {
        var w = Writer.init(&buf);
        try w.writeVarintFieldAlways(8, 25); // limit
        try w.writeVarintFieldAlways(9, 5); // offset
        try w.writeFloatFieldAlways(7, 0.5); // score_threshold
        const f = try w.beginNested(5, 2); // filter
        try w.writeVarintField(1, 1);
        try w.endNested(f);

        var r = Reader.init(w.written());
        const q = try QueryPoints.decode(&r);
        try testing.expectEqual(@as(u64, 25), q.limit);
        try testing.expectEqual(@as(u64, 5), q.offset);
        try testing.expectEqual(@as(?f32, 0.5), q.score_threshold);
        // The filter must be *seen*, so the handler can refuse it. Silently
        // ignoring it would return unfiltered results for a filtered query.
        try testing.expect(q.filter_raw != null);
    }

    // A prefetch field means a multi-stage query, which §1 excludes.
    {
        var w = Writer.init(&buf);
        const pn = try w.beginNested(2, 2);
        try w.writeVarintField(1, 1);
        try w.endNested(pn);
        var r = Reader.init(w.written());
        try testing.expectError(DecodeError.Unimplemented, QueryPoints.decode(&r));
    }
}

test "ScrollPoints defaults limit to 10 when absent, and refuses an explicit zero" {
    // `optional uint32 limit = 4` with `range(min = 1)`; an explicit 0 used
    // to decode to an empty page that still carried a cursor.
    var buf: [64]u8 = undefined;
    var w = wire.Writer.init(&buf);
    try w.writeStringField(1, "s");
    var r = wire.Reader.init(w.written());
    const sp = try ScrollPoints.decode(&r);
    try testing.expectEqual(@as(u32, 10), sp.limit);

    var w2 = wire.Writer.init(&buf);
    try w2.writeStringField(1, "s");
    try w2.writeVarintFieldAlways(4, 0);
    var r2 = wire.Reader.init(w2.written());
    try testing.expectError(DecodeError.InvalidArgument, ScrollPoints.decode(&r2));
    try testing.expect(std.mem.indexOf(u8, invalid_detail, "limit") != null);
    invalid_detail = "";

    var w3 = wire.Writer.init(&buf);
    try w3.writeStringField(1, "s");
    try w3.writeVarintFieldAlways(4, 7);
    var r3 = wire.Reader.init(w3.written());
    try testing.expectEqual(@as(u32, 7), (try ScrollPoints.decode(&r3)).limit);
}

test "ScrollPoints refuses a limit above uint32 rather than truncating it to zero" {
    var buf: [64]u8 = undefined;
    var w = wire.Writer.init(&buf);
    try w.writeStringField(1, "s");
    try w.writeVarintFieldAlways(4, 1 << 32);
    var r = wire.Reader.init(w.written());
    try testing.expectError(DecodeError.InvalidArgument, ScrollPoints.decode(&r));
    try testing.expect(std.mem.indexOf(u8, invalid_detail, "limit") != null);
    invalid_detail = "";

    // The last value that fits is still honoured.
    var w2 = wire.Writer.init(&buf);
    try w2.writeStringField(1, "s");
    try w2.writeVarintFieldAlways(4, std.math.maxInt(u32));
    var r2 = wire.Reader.init(w2.written());
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), (try ScrollPoints.decode(&r2)).limit);
}

test "RepeatedField's next, skip and count agree on the same body" {
    // Three entries of field 2 with a field-1 string between them.
    var buf: [128]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writeStringField(1, "name");
    for ([_]u64{ 7, 8, 9 }) |v| {
        const n = try w.beginNested(2, 2);
        try w.writeVarintFieldAlways(1, v);
        try w.endNested(n);
        try w.writeStringField(3, "x");
    }
    var counter = RepeatedField.init(w.written(), 2);
    try testing.expectEqual(@as(usize, 3), try counter.count());
    var it = RepeatedField.init(w.written(), 2);
    try it.skip(2);
    var sub = (try it.nextNested()).?;
    _ = try sub.tag();
    try testing.expectEqual(@as(u64, 9), try sub.varint());
    try testing.expect((try it.nextNested()) == null);
    // A field nobody sent: nothing, not an error.
    var none = RepeatedField.init(w.written(), 5);
    try testing.expectEqual(@as(usize, 0), try none.count());
}

test "an InvalidArgument detail is taken once and does not outlive its request" {
    // `searchOne` read the threadlocal without clearing it, so an upsert with
    // no vectors on the same worker was answered with the previous query's
    // "FieldCondition without a match".
    var buf: [64]u8 = undefined;
    var w = Writer.init(&buf);
    {
        const cond = try w.beginNested(2, 2);
        const fc = try w.beginNested(1, 2);
        try w.writeStringField(1, "k");
        try w.endNested(fc);
        try w.endNested(cond);
    }
    try testing.expectError(DecodeError.InvalidArgument, decodeFilter(w.written()));
    try testing.expect(std.mem.indexOf(u8, takeInvalidDetail(), "without a match") != null);
    try testing.expectEqualStrings("", takeInvalidDetail());
}

test "a Filter decodes must/should/must_not by field number, and refuses the rest by name" {
    var buf: [256]u8 = undefined;
    var w = Writer.init(&buf);
    // must: a = "x"; must_not: n = 3 (integer); should: a in ["p", "q"].
    {
        const cond = try w.beginNested(2, 2);
        const fc = try w.beginNested(1, 2);
        try w.writeStringField(1, "a");
        const m = try w.beginNested(2, 2);
        try w.writeStringField(1, "x");
        try w.endNested(m);
        try w.endNested(fc);
        try w.endNested(cond);
    }
    {
        const cond = try w.beginNested(3, 2);
        const fc = try w.beginNested(1, 2);
        try w.writeStringField(1, "n");
        const m = try w.beginNested(2, 2);
        try w.writeVarintFieldAlways(2, 3);
        try w.endNested(m);
        try w.endNested(fc);
        try w.endNested(cond);
    }
    {
        const cond = try w.beginNested(1, 2);
        const fc = try w.beginNested(1, 2);
        try w.writeStringField(1, "a");
        const m = try w.beginNested(2, 2);
        const rs = try w.beginNested(5, 2);
        try w.writeStringField(1, "p");
        try w.writeStringField(1, "q");
        try w.endNested(rs);
        try w.endNested(m);
        try w.endNested(fc);
        try w.endNested(cond);
    }
    const f = try decodeFilter(w.written());
    try testing.expectEqual(@as(usize, 1), f.must_len);
    try testing.expectEqualStrings("a", f.must[0].key);
    try testing.expectEqualStrings("x", f.must[0].match.keyword);
    try testing.expectEqual(@as(usize, 1), f.must_not_len);
    try testing.expectEqual(@as(i64, 3), f.must_not[0].match.integer);
    try testing.expectEqual(@as(usize, 1), f.should_len);
    try testing.expect(f.should[0].match == .keywords);
    // An empty filter is empty.
    try testing.expect((try decodeFilter(&.{})).isEmpty());

    // `Match.text` (4) is full-text search: refused, naming it.
    var w2 = Writer.init(&buf);
    {
        const cond = try w2.beginNested(2, 2);
        const fc = try w2.beginNested(1, 2);
        try w2.writeStringField(1, "a");
        const m = try w2.beginNested(2, 2);
        try w2.writeStringField(4, "hello");
        try w2.endNested(m);
        try w2.endNested(fc);
        try w2.endNested(cond);
    }
    try testing.expectError(DecodeError.Unimplemented, decodeFilter(w2.written()));
    try testing.expectEqualStrings("full-text match (§1 non-goal)", unimplemented_detail);
    // `has_id` (Condition = 3).
    var w3 = Writer.init(&buf);
    {
        const cond = try w3.beginNested(2, 2);
        const hi = try w3.beginNested(3, 2);
        try w3.writeVarintField(1, 1);
        try w3.endNested(hi);
        try w3.endNested(cond);
    }
    try testing.expectError(DecodeError.Unimplemented, decodeFilter(w3.written()));
    try testing.expectEqualStrings("has_id conditions", unimplemented_detail);
}

test "PointStruct.payload_raw spans every map entry, not the last one" {
    var buf: [256]u8 = undefined;
    var w = Writer.init(&buf);
    {
        const idn = try w.beginNested(1, 2);
        try w.writeVarintFieldAlways(1, 1);
        try w.endNested(idn);
    }
    for ([_][]const u8{ "k1", "k2", "k3" }) |k| {
        const entry = try w.beginNested(3, 2);
        try w.writeStringField(1, k);
        const val = try w.beginNested(2, 2);
        try w.writeStringField(payload.value_string, "v");
        try w.endNested(val);
        try w.endNested(entry);
    }
    {
        const vs = try w.beginNested(4, 3);
        const vec = try w.beginNested(1, 3);
        const dv = try w.beginNested(101, 3);
        try w.writePackedFloats(1, &[_]f32{ 1, 2 });
        try w.endNested(dv);
        try w.endNested(vec);
        try w.endNested(vs);
    }
    var r = Reader.init(w.written());
    const pt = try PointStruct.decode(&r);
    var it = payload.WireEntries.init(pt.payload_raw.?, PointStruct.payload_field);
    var n: usize = 0;
    while (try it.next()) |e| : (n += 1) {
        const kv = payload.splitEntry(e).?;
        const want: []const u8 = switch (n) {
            0 => "k1",
            1 => "k2",
            2 => "k3",
            else => unreachable,
        };
        try testing.expectEqualStrings(want, kv.key);
    }
    try testing.expectEqual(@as(usize, 3), n);
}
