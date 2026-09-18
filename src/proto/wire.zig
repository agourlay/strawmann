//! Protobuf wire-format primitives.
//!
//! §6.2: "Hand-written codecs for the ~30 messages actually needed. No generic
//! reflection, no code generator dependency, no `unknown_fields` retention."
//!
//! This file is the layer below the message definitions: varints, tags, wire
//! types, and the length-delimited framing. It knows nothing about Qdrant.
//!
//! Two decisions worth stating, because both are load-bearing for §6.2's fast
//! path:
//!
//!  1. **The reader borrows, it does not copy.** `bytes()` returns a subslice of
//!     the input buffer. An upsert's vector data is then a `[]const u8` pointing
//!     into the network frame, which is what lets the decode write "directly
//!     into final storage, the point's vector lands in `vectors.bin`'s mapped
//!     arena, not in an intermediate `Vec<f32>`."
//!
//!  2. **Unknown fields are skipped by wire type and discarded.** proto3 allows
//!     retaining them for round-tripping; we do not, because nothing in the
//!     compatibility surface of §2 re-emits a message it received.

const std = @import("std");
const builtin = @import("builtin");

pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    length_delimited = 2,
    start_group = 3, // proto2 groups; rejected
    end_group = 4, // proto2 groups; rejected
    fixed32 = 5,
    _,
};

pub const Error = error{
    /// Ran off the end of the buffer.
    Truncated,
    /// A varint longer than 10 bytes, or one that overflows its target type.
    InvalidVarint,
    /// A wire type we do not implement (proto2 groups) or that is not a valid
    /// wire type at all.
    InvalidWireType,
    /// A length-delimited field whose length exceeds the remaining buffer.
    InvalidLength,
    /// A packed repeated fixed-width field whose byte length is not a multiple
    /// of the element size.
    InvalidPackedLength,
    /// Field number 0, which is never legal.
    InvalidFieldNumber,
    /// Nesting deeper than `max_depth`.
    TooDeep,
};

/// Maximum message nesting depth.
///
/// §8.8 asks protocol fuzzing to cover "deeply nested messages" with the
/// requirement "no crash, no hang, no UB, a clean gRPC error". A depth limit
/// is how a recursive-descent decoder gets that guarantee; without it a crafted
/// message overflows the stack. The Qdrant messages we implement nest at most
/// 5 deep (QueryBatchPoints > QueryPoints > Query > VectorInput > DenseVector),
/// so 32 is generous while still bounded.
pub const max_depth = 32;

pub const Tag = struct {
    field: u32,
    wire_type: WireType,
};

/// A cursor over an encoded message.
pub const Reader = struct {
    buf: []const u8,
    pos: usize = 0,
    depth: u8 = 0,

    pub fn init(buf: []const u8) Reader {
        return .{ .buf = buf };
    }

    pub fn atEnd(self: *const Reader) bool {
        return self.pos >= self.buf.len;
    }

    pub fn remaining(self: *const Reader) usize {
        return self.buf.len - self.pos;
    }

    /// Read a base-128 varint, up to 10 bytes.
    pub fn varint(self: *Reader) Error!u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        var i: usize = 0;
        while (i < 10) : (i += 1) {
            if (self.pos >= self.buf.len) return Error.Truncated;
            const b = self.buf[self.pos];
            self.pos += 1;
            if (i == 9) {
                // The 10th byte may only carry the single remaining bit; any
                // other value overflows u64. Rejecting it rather than silently
                // truncating is what keeps a malformed frame from decoding into
                // a plausible-looking number.
                if (b > 1) return Error.InvalidVarint;
                result |= @as(u64, b) << 63;
                return result;
            }
            result |= @as(u64, b & 0x7f) << shift;
            if (b & 0x80 == 0) return result;
            shift += 7;
        }
        return Error.InvalidVarint;
    }

    pub fn varint32(self: *Reader) Error!u32 {
        const v = try self.varint();
        // proto3 encodes negative int32 as a 10-byte varint (sign-extended to
        // 64 bits), so truncating to the low 32 bits is correct rather than
        // lossy. Callers wanting the signed value use `@bitCast`.
        return @truncate(v);
    }

    pub fn boolean(self: *Reader) Error!bool {
        return (try self.varint()) != 0;
    }

    /// ZigZag-decoded sint32/sint64.
    pub fn zigzag64(self: *Reader) Error!i64 {
        const v = try self.varint();
        return @bitCast((v >> 1) ^ (~(v & 1) +% 1));
    }

    pub fn fixed32(self: *Reader) Error!u32 {
        if (self.remaining() < 4) return Error.Truncated;
        const v = std.mem.readInt(u32, self.buf[self.pos..][0..4], .little);
        self.pos += 4;
        return v;
    }

    pub fn fixed64(self: *Reader) Error!u64 {
        if (self.remaining() < 8) return Error.Truncated;
        const v = std.mem.readInt(u64, self.buf[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }

    pub fn float(self: *Reader) Error!f32 {
        return @bitCast(try self.fixed32());
    }

    pub fn double(self: *Reader) Error!f64 {
        return @bitCast(try self.fixed64());
    }

    /// A length-delimited field, returned as a **borrowed** subslice.
    pub fn bytes(self: *Reader) Error![]const u8 {
        const len = try self.varint();
        if (len > self.remaining()) return Error.InvalidLength;
        const start = self.pos;
        self.pos += @intCast(len);
        return self.buf[start..self.pos];
    }

    /// A nested message, as a sub-reader carrying the depth counter.
    pub fn nested(self: *Reader) Error!Reader {
        if (self.depth + 1 >= max_depth) return Error.TooDeep;
        const b = try self.bytes();
        return .{ .buf = b, .pos = 0, .depth = self.depth + 1 };
    }

    pub fn tag(self: *Reader) Error!Tag {
        const t = try self.varint32();
        const field = t >> 3;
        if (field == 0) return Error.InvalidFieldNumber;
        const wt: WireType = @enumFromInt(@as(u3, @truncate(t)));
        switch (wt) {
            .varint, .fixed64, .length_delimited, .fixed32 => {},
            else => return Error.InvalidWireType,
        }
        return .{ .field = field, .wire_type = wt };
    }

    /// Skip a field of unknown number, by wire type.
    ///
    /// §6.2: "Strict-but-cheap validation: unknown field → skip by wire type".
    /// Forward compatibility depends on this: a client from a newer Qdrant
    /// version sending a field we have never heard of must not fail the request.
    pub fn skip(self: *Reader, wt: WireType) Error!void {
        switch (wt) {
            .varint => _ = try self.varint(),
            .fixed64 => _ = try self.fixed64(),
            .fixed32 => _ = try self.fixed32(),
            .length_delimited => _ = try self.bytes(),
            else => return Error.InvalidWireType,
        }
    }

    /// **The fp32 fast path.** §6.2:
    ///
    ///   "`DenseVector.data` decode: bounds-check, then one memcpy into the
    ///    destination slot in the vector arena. Never element-wise. Handle
    ///    unaligned source with unaligned loads."
    ///
    /// §2 explains why a memcpy is sufficient and not merely convenient:
    /// "`repeated float data = 1` inside `DenseVector` is packed in proto3 →
    /// arrives as one length-delimited little-endian `f32` run. On x86-64 this
    /// is byte-identical to our in-memory layout."
    ///
    /// Returns the number of floats written. `dst` must be large enough; the
    /// caller knows the collection's dimension and checks the count against it.
    pub fn packedFloatsInto(self: *Reader, dst: []f32) Error!usize {
        // The memcpy below is only valid because the wire's little-endian f32
        // run is byte-identical to our in-memory layout. On a big-endian target
        // it would silently produce reversed floats: no error, no crash, just
        // wrong vectors and therefore wrong neighbours. Assert it at compile
        // time so that build fails instead of that run lying.
        comptime std.debug.assert(builtin.cpu.arch.endian() == .little);
        const raw = try self.bytes();
        if (raw.len % 4 != 0) return Error.InvalidPackedLength;
        const count = raw.len / 4;
        if (count > dst.len) return Error.InvalidLength;
        // One memcpy. `dst` is 64 B aligned arena storage, `raw` points into a
        // network frame with no alignment guarantee, so this is an unaligned
        // load / aligned store, which is what `@memcpy` compiles to.
        @memcpy(std.mem.sliceAsBytes(dst[0..count]), raw);
        return count;
    }

    /// Borrow a packed float run without copying, when the alignment happens to
    /// permit it. Returns null when the source is not 4-byte aligned, in which
    /// case the caller must use `packedFloatsInto`.
    ///
    /// Used on the *query* path, where the vector is read once and discarded:
    /// there is no arena slot to land in, so a copy would be pure overhead.
    /// §2 warns "the payload is not guaranteed 4-byte aligned in the frame
    /// buffer", hence the null case rather than an unchecked cast.
    pub fn packedFloatsBorrow(self: *Reader) Error!?[]const f32 {
        const raw = try self.bytes();
        if (raw.len % 4 != 0) return Error.InvalidPackedLength;
        if (@intFromPtr(raw.ptr) % @alignOf(f32) != 0) return null;
        return @alignCast(std.mem.bytesAsSlice(f32, raw));
    }
};

// -------------------------------------------------------------------------
// Encoding
// -------------------------------------------------------------------------

/// A writer over a caller-provided buffer.
///
/// §6.2: "Response encode for `QueryBatchResponse` writes directly into the
/// output frame buffer; pre-size it from `limit × batch_size`." So the writer
/// never allocates and never grows, the caller sizes the buffer from the
/// request, and an overflow is a bug in that sizing rather than a runtime
/// condition to recover from.
pub const Writer = struct {
    buf: []u8,
    pos: usize = 0,

    pub const Error = error{BufferTooSmall};

    pub fn init(buf: []u8) Writer {
        return .{ .buf = buf };
    }

    pub fn written(self: *const Writer) []u8 {
        return self.buf[0..self.pos];
    }

    pub fn reset(self: *Writer) void {
        self.pos = 0;
    }

    fn need(self: *Writer, n: usize) Writer.Error!void {
        if (self.pos + n > self.buf.len) return Writer.Error.BufferTooSmall;
    }

    pub fn varint(self: *Writer, v: u64) Writer.Error!void {
        var x = v;
        while (true) {
            try self.need(1);
            const b: u8 = @truncate(x & 0x7f);
            x >>= 7;
            if (x == 0) {
                self.buf[self.pos] = b;
                self.pos += 1;
                return;
            }
            self.buf[self.pos] = b | 0x80;
            self.pos += 1;
        }
    }

    pub fn tag(self: *Writer, field: u32, wt: WireType) Writer.Error!void {
        // Field 0 is not a legal protobuf field number and encodes as a tag
        // byte of 0, which a decoder reads as end-of-message. Every field
        // written after it would be silently dropped.
        std.debug.assert(field > 0);
        try self.varint((@as(u64, field) << 3) | @intFromEnum(wt));
    }

    pub fn fixed32(self: *Writer, v: u32) Writer.Error!void {
        try self.need(4);
        std.mem.writeInt(u32, self.buf[self.pos..][0..4], v, .little);
        self.pos += 4;
    }

    pub fn fixed64(self: *Writer, v: u64) Writer.Error!void {
        try self.need(8);
        std.mem.writeInt(u64, self.buf[self.pos..][0..8], v, .little);
        self.pos += 8;
    }

    pub fn raw(self: *Writer, b: []const u8) Writer.Error!void {
        try self.need(b.len);
        @memcpy(self.buf[self.pos..][0..b.len], b);
        self.pos += b.len;
    }

    // --- typed field writers ---
    //
    // proto3 default-value elision: a field equal to its type's default is not
    // written at all. Every writer below honours that, because Qdrant's own
    // encoder does and the T0 wire-conformance tier of §8.5 compares "field
    // presence" between the two engines. Emitting an explicit zero where Qdrant
    // omits the field is a T0 failure even though every value is identical.

    pub fn writeVarintField(self: *Writer, field: u32, v: u64) Writer.Error!void {
        if (v == 0) return;
        try self.tag(field, .varint);
        try self.varint(v);
    }

    pub fn writeBoolField(self: *Writer, field: u32, v: bool) Writer.Error!void {
        if (!v) return;
        try self.tag(field, .varint);
        try self.varint(1);
    }

    pub fn writeFloatField(self: *Writer, field: u32, v: f32) Writer.Error!void {
        // Note: `-0.0 == 0.0` is true in IEEE, so a negative zero is elided as
        // a default. That matches protobuf's own behaviour.
        if (v == 0.0) return;
        try self.tag(field, .fixed32);
        try self.fixed32(@bitCast(v));
    }

    pub fn writeDoubleField(self: *Writer, field: u32, v: f64) Writer.Error!void {
        if (v == 0.0) return;
        try self.tag(field, .fixed64);
        try self.fixed64(@bitCast(v));
    }

    pub fn writeStringField(self: *Writer, field: u32, v: []const u8) Writer.Error!void {
        if (v.len == 0) return;
        try self.tag(field, .length_delimited);
        try self.varint(v.len);
        try self.raw(v);
    }

    /// Always writes, even when the value is the default.
    ///
    /// Needed for fields inside a `oneof`, where presence is the whole point:
    /// `PointId{num: 0}` must serialise as a present field or the ID is lost.
    pub fn writeVarintFieldAlways(self: *Writer, field: u32, v: u64) Writer.Error!void {
        try self.tag(field, .varint);
        try self.varint(v);
    }

    pub fn writeFloatFieldAlways(self: *Writer, field: u32, v: f32) Writer.Error!void {
        try self.tag(field, .fixed32);
        try self.fixed32(@bitCast(v));
    }

    /// Packed repeated float, the encode-side counterpart of
    /// `packedFloatsInto`, and one memcpy for the same reason.
    pub fn writePackedFloats(self: *Writer, field: u32, v: []const f32) Writer.Error!void {
        std.debug.assert(field > 0);
        if (v.len == 0) return;
        try self.tag(field, .length_delimited);
        try self.varint(v.len * 4);
        try self.raw(std.mem.sliceAsBytes(v));
    }

    /// Begin a nested message whose length is not yet known.
    ///
    /// Reserves `reserve` bytes for the length varint, writes the body, then
    /// backfills. Sub-messages in the responses we emit are small and bounded,
    /// so a 2-byte reservation (up to 16383 bytes) covers `ScoredPoint` and
    /// friends; `BatchResult` needs more and asks for it explicitly.
    pub fn beginNested(self: *Writer, field: u32, reserve: usize) Writer.Error!Nested {
        try self.tag(field, .length_delimited);
        try self.need(reserve);
        const len_pos = self.pos;
        self.pos += reserve;
        return .{ .len_pos = len_pos, .reserve = reserve, .body_start = self.pos };
    }

    pub const Nested = struct {
        len_pos: usize,
        reserve: usize,
        body_start: usize,
    };

    pub fn endNested(self: *Writer, n: Nested) Writer.Error!void {
        const body_len = self.pos - n.body_start;

        // Write the length into the reserved space, padded with redundant
        // continuation bytes so the body never has to move. A varint with
        // non-minimal encoding is legal on the wire and every decoder accepts
        // it; the alternative, memmove the body to close the gap, would cost
        // a copy of the entire response.
        var v = body_len;
        var i: usize = 0;
        while (i < n.reserve - 1) : (i += 1) {
            self.buf[n.len_pos + i] = @as(u8, @truncate(v & 0x7f)) | 0x80;
            v >>= 7;
        }
        if (v > 0x7f) return Writer.Error.BufferTooSmall; // body outgrew the reservation
        self.buf[n.len_pos + n.reserve - 1] = @truncate(v);
    }
};

/// Bytes a varint of this value occupies.
pub fn varintSize(v: u64) usize {
    if (v == 0) return 1;
    return (63 - @clz(v)) / 7 + 1;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "varint round-trips across the interesting magnitudes" {
    const values = [_]u64{
        0,                    1,                    127,
        128,                  300,                  16383,
        16384,                std.math.maxInt(u32), std.math.maxInt(u32) + 1,
        std.math.maxInt(u64),
    };
    var buf: [64]u8 = undefined;
    for (values) |v| {
        var w = Writer.init(&buf);
        try w.varint(v);
        try std.testing.expectEqual(varintSize(v), w.pos);
        var r = Reader.init(w.written());
        try std.testing.expectEqual(v, try r.varint());
        try std.testing.expect(r.atEnd());
    }
}

test "varint rejects overlong and overflowing encodings" {
    // 11 continuation bytes: never valid.
    const too_long = [_]u8{0xff} ** 11;
    var r = Reader.init(&too_long);
    try std.testing.expectError(Error.InvalidVarint, r.varint());

    // 10 bytes where the last carries more than one bit: overflows u64.
    const overflow = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02 };
    var r2 = Reader.init(&overflow);
    try std.testing.expectError(Error.InvalidVarint, r2.varint());

    // Truncated mid-varint.
    const truncated = [_]u8{ 0xff, 0xff };
    var r3 = Reader.init(&truncated);
    try std.testing.expectError(Error.Truncated, r3.varint());
}

test "tag encodes field number and wire type" {
    var buf: [16]u8 = undefined;
    var w = Writer.init(&buf);
    try w.tag(1, .length_delimited);
    try w.tag(1000, .varint);

    var r = Reader.init(w.written());
    const t1 = try r.tag();
    try std.testing.expectEqual(@as(u32, 1), t1.field);
    try std.testing.expectEqual(WireType.length_delimited, t1.wire_type);
    const t2 = try r.tag();
    try std.testing.expectEqual(@as(u32, 1000), t2.field);
    try std.testing.expectEqual(WireType.varint, t2.wire_type);
}

test "tag rejects field 0 and proto2 group wire types" {
    // Field 0 with wire type 0 encodes as a single zero byte.
    const zero_field = [_]u8{0x00};
    var r = Reader.init(&zero_field);
    try std.testing.expectError(Error.InvalidFieldNumber, r.tag());

    // Field 1, wire type 3 (start_group).
    const group = [_]u8{(1 << 3) | 3};
    var r2 = Reader.init(&group);
    try std.testing.expectError(Error.InvalidWireType, r2.tag());
}

test "packed floats decode with a single memcpy and reject bad lengths" {
    const src = [_]f32{ 1.0, -2.5, 3.25, 1e30, -0.0 };
    var buf: [128]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writePackedFloats(1, &src);

    var r = Reader.init(w.written());
    const t = try r.tag();
    try std.testing.expectEqual(WireType.length_delimited, t.wire_type);

    var dst: [8]f32 = undefined;
    const n = try r.packedFloatsInto(&dst);
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqualSlices(f32, &src, dst[0..5]);

    // A byte length not divisible by 4 is malformed.
    const bad = [_]u8{ (1 << 3) | 2, 3, 0x00, 0x00, 0x00 };
    var rb = Reader.init(&bad);
    _ = try rb.tag();
    try std.testing.expectError(Error.InvalidPackedLength, rb.packedFloatsInto(&dst));

    // More floats than the destination holds must be refused, not truncated:
    // silently accepting a 1536-dim vector into a 768-dim collection would
    // corrupt the neighbouring arena slot.
    var tiny: [2]f32 = undefined;
    var r2 = Reader.init(w.written());
    _ = try r2.tag();
    try std.testing.expectError(Error.InvalidLength, r2.packedFloatsInto(&tiny));
}

test "packedFloatsBorrow returns null on unaligned input rather than misreading" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    // Build a buffer where the float payload lands at an odd offset.
    var backing: [64]u8 align(4) = undefined;
    var w = Writer.init(backing[1..]); // deliberately offset by 1
    try w.writePackedFloats(1, &src);

    var r = Reader.init(w.written());
    _ = try r.tag();
    const borrowed = try r.packedFloatsBorrow();
    // Payload starts at backing[1] + 1 tag byte + 1 length byte = backing[3],
    // which is not 4-byte aligned, so borrowing must decline.
    try std.testing.expectEqual(@as(?[]const f32, null), borrowed);
}

test "skip advances past every wire type" {
    var buf: [128]u8 = undefined;
    var w = Writer.init(&buf);
    try w.tag(1, .varint);
    try w.varint(123456);
    try w.tag(2, .fixed64);
    try w.fixed64(0xdeadbeefcafe);
    try w.tag(3, .length_delimited);
    try w.varint(3);
    try w.raw("abc");
    try w.tag(4, .fixed32);
    try w.fixed32(42);
    try w.tag(5, .varint);
    try w.varint(7);

    var r = Reader.init(w.written());
    // Skip the first four fields blind, then read the fifth.
    for (0..4) |_| {
        const t = try r.tag();
        try r.skip(t.wire_type);
    }
    const t5 = try r.tag();
    try std.testing.expectEqual(@as(u32, 5), t5.field);
    try std.testing.expectEqual(@as(u64, 7), try r.varint());
    try std.testing.expect(r.atEnd());
}

test "proto3 default elision matches what Qdrant emits" {
    var buf: [64]u8 = undefined;
    var w = Writer.init(&buf);
    try w.writeVarintField(1, 0); // elided
    try w.writeBoolField(2, false); // elided
    try w.writeFloatField(3, 0.0); // elided
    try w.writeStringField(4, ""); // elided
    try std.testing.expectEqual(@as(usize, 0), w.pos);

    try w.writeVarintField(1, 5);
    try std.testing.expect(w.pos > 0);

    // But a oneof member must be written even at its default.
    var w2 = Writer.init(&buf);
    try w2.writeVarintFieldAlways(1, 0);
    try std.testing.expectEqual(@as(usize, 2), w2.pos);
}

test "nested length backfill handles a body that grows past one length byte" {
    var buf: [4096]u8 = undefined;
    var w = Writer.init(&buf);
    const n = try w.beginNested(1, 2);
    // 200 bytes of body: needs 2 length bytes in minimal encoding, and our
    // reservation is exactly 2.
    const body = [_]u8{0xab} ** 200;
    try w.raw(&body);
    try w.endNested(n);

    var r = Reader.init(w.written());
    const t = try r.tag();
    try std.testing.expectEqual(@as(u32, 1), t.field);
    const got = try r.bytes();
    try std.testing.expectEqual(@as(usize, 200), got.len);
    try std.testing.expectEqualSlices(u8, &body, got);
}

test "nested length backfill produces a decodable non-minimal varint" {
    // A 3-byte reservation holding a small length must still decode correctly.
    var buf: [256]u8 = undefined;
    var w = Writer.init(&buf);
    const n = try w.beginNested(7, 3);
    try w.raw("hi");
    try w.endNested(n);

    var r = Reader.init(w.written());
    const t = try r.tag();
    try std.testing.expectEqual(@as(u32, 7), t.field);
    try std.testing.expectEqualSlices(u8, "hi", try r.bytes());
    try std.testing.expect(r.atEnd());
}

test "writer refuses to overflow its buffer" {
    var buf: [4]u8 = undefined;
    var w = Writer.init(&buf);
    try w.raw("abcd");
    try std.testing.expectError(Writer.Error.BufferTooSmall, w.raw("e"));
}

test "nesting depth is bounded" {
    // Build a chain of nested length-delimited fields deeper than max_depth and
    // confirm the reader refuses rather than recursing. §8.8: "deeply nested
    // messages. Target: no crash, no hang, no UB."
    // `max_depth + 1` nested field-1 messages, innermost first: each level
    // is a tag, a length and the level below it.
    // Level i is 2i + 2 bytes, so the chain needs the sum of those.
    var storage_buf: [(max_depth + 2) * (max_depth + 3)]u8 = undefined;
    var used: usize = 0;
    var inner: []u8 = storage_buf[0..0];
    for (0..max_depth + 1) |_| {
        const buf = storage_buf[used..][0 .. inner.len + 2];
        buf[0] = (1 << 3) | 2; // field 1, length-delimited
        buf[1] = @intCast(inner.len);
        @memcpy(buf[2..], inner);
        used += buf.len;
        inner = buf;
    }
    var r = Reader.init(inner);
    var depth: usize = 0;
    const result = blk: while (true) {
        _ = r.tag() catch |e| break :blk e;
        r = r.nested() catch |e| break :blk e;
        depth += 1;
        if (r.atEnd()) break :blk error.ChainEnded;
    };
    try std.testing.expectEqual(Error.TooDeep, result);
    try std.testing.expect(depth < max_depth + 1);
}

test "zigzag decodes signed values" {
    var buf: [16]u8 = undefined;
    // ZigZag: 0->0, -1->1, 1->2, -2->3
    const cases = [_]struct { u64, i64 }{
        .{ 0, 0 }, .{ 1, -1 }, .{ 2, 1 }, .{ 3, -2 }, .{ 4294967294, 2147483647 },
    };
    for (cases) |c| {
        var w = Writer.init(&buf);
        try w.varint(c[0]);
        var r = Reader.init(w.written());
        try std.testing.expectEqual(c[1], try r.zigzag64());
    }
}
