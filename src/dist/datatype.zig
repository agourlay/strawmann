//! Storage element types, matching Qdrant's `VectorParams.datatype`.
//!
//! §5.4's argument is that quantization's real win is "moving the working set
//! up a level of the memory hierarchy". A storage datatype is the same win
//! without a codebook: f16 halves the arena, u8 quarters it, and the graph
//! traversal touches proportionally fewer cache lines. §5.2's table already
//! prices fp16 at 1536 B/vector against fp32's 3072 at d=768.
//!
//! ## The semantics are Qdrant's, and they were read rather than assumed
//!
//! Verified against a `dev` checkout at `0e3397469`, because this is exactly
//! the kind of thing this project has invented before: "8 to 13 segments" was
//! inferred from behaviour and then cited in six files as though it were read.
//!
//! | | ingest | dot | euclid | manhattan | cosine |
//! |---|---|---|---|---|---|
//! | `float32` | metric preprocess | Σab | −Σ(a−b)² | −Σ\|a−b\| | dot, normalised at ingest |
//! | `float16` | preprocess, then `f16::from_f32` | as f32, widened | as f32 | as f32 | dot, normalised at ingest |
//! | `uint8` | **no normalisation**, `x as u8` | Σ i32(a)i32(b) | −Σ(a−b)² | −Σ\|a−b\| | **dot / √(Σa²·Σb²) at query time** |
//!
//! Two of those cells are the interesting ones.
//!
//! **`uint8` does not normalise cosine at ingest**, unlike every other
//! combination. `impl Metric<VectorElementTypeByte> for CosineMetric` has
//! `fn preprocess(vector) -> vector`, the identity, and
//! `cosine_similarity_bytes` divides by the norms at query time instead. It
//! could hardly do otherwise: normalised components lie in [−1, 1] and would
//! truncate to 0 or 1, destroying the vector. This contradicts §8.3's rule that
//! "Cosine is normalised at ingest and stored normalised, so the search path
//! only ever runs dot product", so the rule holds per datatype rather than
//! globally, and `Datatype.normalisesAtIngest` is where that is written down.
//!
//! **`uint8` conversion saturates, it does not validate.** Qdrant's cast is
//! Rust's `x as u8`, which clamps and truncates toward zero rather than
//! rejecting; their own test pins it:
//!
//! ```text
//! [-10.0, 1.0, 2.0, 3.0, 255.0, 300.0] -> [0, 1, 2, 3, 255, 255]
//! ```
//!
//! So a client sending fp32 data into a `uint8` collection gets a quietly
//! mangled vector rather than an error. We match it, because §8.5's T1 tier
//! compares values with the real server and "we are stricter" is still a
//! difference. `toU8` carries the same test.

const std = @import("std");
const Metric = @import("metric.zig").Metric;

/// The element a vector is stored as. `enum(u8)` because it goes in a 4 KiB
/// file header (`core/storage.zig`) where the width is part of the format.
pub const Datatype = enum(u8) {
    /// The default, and what every earlier build stored. Its numeric value is
    /// 0 so that a header written before this field existed, whose reserved
    /// bytes were zeroed, loads as fp32 rather than as garbage.
    float32 = 0,
    float16 = 1,
    uint8 = 2,

    pub fn elemSize(self: Datatype) usize {
        return switch (self) {
            .float32 => @sizeOf(f32),
            .float16 => @sizeOf(f16),
            .uint8 => @sizeOf(u8),
        };
    }

    pub fn name(self: Datatype) []const u8 {
        return @tagName(self);
    }

    /// qdrant `Datatype`: Default=0, Float32=1, Uint8=2, Float16=3.
    ///
    /// Their numbering, not ours: the wire value and the storage value are
    /// deliberately separate constants, for the same reason `Metric` keeps
    /// `fromProto` apart from its own tag values. A proto renumbering must not
    /// silently change what a file header means.
    pub fn fromProto(v: i32) ?Datatype {
        return switch (v) {
            0, 1 => .float32,
            2 => .uint8,
            3 => .float16,
            else => null,
        };
    }

    pub fn toProto(self: Datatype) i32 {
        return switch (self) {
            .float32 => 1,
            .uint8 => 2,
            .float16 => 3,
        };
    }

    /// Whether cosine is normalised at ingest for this datatype (§8.3).
    ///
    /// True for the float types, false for `uint8`, which is Qdrant's
    /// behaviour and the reason §8.3's rule is per datatype. A `uint8` cosine
    /// collection therefore stores raw components and pays for the norms on
    /// every comparison.
    pub fn normalisesAtIngest(self: Datatype, metric: Metric) bool {
        if (metric != .cosine) return false;
        return switch (self) {
            .float32, .float16 => true,
            .uint8 => false,
        };
    }

    /// Whether the search path can treat cosine as a plain dot product.
    /// The inverse of the above, phrased the way the search path asks it.
    pub fn cosineIsDot(self: Datatype, metric: Metric) bool {
        return metric != .cosine or self.normalisesAtIngest(metric);
    }
};

/// f32 -> f16, round to nearest even, the hardware default.
pub inline fn toF16(x: f32) f16 {
    return @floatCast(x);
}

/// The width the conversions work in. One `vcvtps2ph` covers 8 lanes on F16C
/// and 16 on AVX-512, and the u8 path folds to a min/max/convert of the same
/// shape, so the loops below are one instruction group per iteration.
const conv_lanes = 16;

/// Convert a whole vector to f16.
///
/// Ingest converts `count × dim` components per collection and every search
/// converts `dim` more, and both were scalar loops: the same arithmetic at a
/// sixteenth of the width.
pub fn convertToF16(dst: []f16, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    const V = @Vector(conv_lanes, f32);
    const H = @Vector(conv_lanes, f16);
    var i: usize = 0;
    while (i + conv_lanes <= src.len) : (i += conv_lanes) {
        const v: V = src[i..][0..conv_lanes].*;
        dst[i..][0..conv_lanes].* = @as(H, @floatCast(v));
    }
    while (i < src.len) : (i += 1) dst[i] = toF16(src[i]);
}

/// Convert a whole vector to u8, with the same saturating truncation `toU8`
/// performs one element at a time.
///
/// NaN is selected to zero explicitly rather than left to `@min`/`@max`:
/// whether a NaN survives a vector min depends on the instruction the backend
/// picks, and this conversion has to match Qdrant's on every input, not on the
/// inputs a test happens to try.
pub fn convertToU8(dst: []u8, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    const V = @Vector(conv_lanes, f32);
    const B = @Vector(conv_lanes, u8);
    const zero: V = @splat(0);
    const top: V = @splat(255);
    var i: usize = 0;
    while (i + conv_lanes <= src.len) : (i += conv_lanes) {
        const v: V = src[i..][0..conv_lanes].*;
        const t = @trunc(v);
        const finite = @select(f32, t == t, t, zero);
        const clamped = @min(@max(finite, zero), top);
        dst[i..][0..conv_lanes].* = @as(B, @intFromFloat(clamped));
    }
    while (i < src.len) : (i += 1) dst[i] = toU8(src[i]);
}

/// f32 -> u8 the way Rust's `as` does it, which is what Qdrant relies on.
///
/// Saturating and truncating toward zero, NaN to 0. Zig's `@intFromFloat` is
/// illegal behaviour outside the destination range rather than saturating, so
/// the clamp is explicit and is the whole point of this function existing.
pub fn toU8(x: f32) u8 {
    // Ordered comparisons are false for NaN, so `!(x >= 1)` covers NaN,
    // negatives, and everything that truncates to zero in one test, and
    // guarantees the `@intFromFloat` below sees a value inside u8's range.
    // Getting that wrong is illegal behaviour in ReleaseFast rather than a
    // wrong number, which is why the range is established by construction
    // rather than checked afterwards.
    if (!(x >= 1)) return 0;
    if (x >= 255) return 255;
    return @intFromFloat(@trunc(x));
}

test "uint8 conversion matches qdrant's own pinned test vector" {
    // `lib/segment/src/spaces/metric_uint/simple_euclid.rs`,
    // `test_conversion_to_bytes`, verbatim.
    var in = [_]f32{ -10.0, 1.0, 2.0, 3.0, 255.0, 300.0 };
    const want = [_]u8{ 0, 1, 2, 3, 255, 255 };
    for (&in, want) |*x, w| try std.testing.expectEqual(w, toU8(x.*));
}

test "uint8 conversion truncates toward zero and takes NaN to zero" {
    // Through a `var` so these are runtime values: a comptime-folded NaN takes
    // a different path through the compiler than the one the ingest loop will.
    var v: f32 = 3.9;
    try std.testing.expectEqual(@as(u8, 3), toU8(v));
    v = 0.9;
    try std.testing.expectEqual(@as(u8, 0), toU8(v));
    v = -0.9;
    try std.testing.expectEqual(@as(u8, 0), toU8(v));
    v = std.math.nan(f32);
    try std.testing.expectEqual(@as(u8, 0), toU8(v));
    v = std.math.inf(f32);
    try std.testing.expectEqual(@as(u8, 255), toU8(v));
    v = -std.math.inf(f32);
    try std.testing.expectEqual(@as(u8, 0), toU8(v));
    v = 254.999;
    try std.testing.expectEqual(@as(u8, 254), toU8(v));
}

test "the wire numbering is qdrant's and the storage numbering is ours" {
    // Default and Float32 both mean fp32 storage; they are distinct on the
    // wire and identical here.
    try std.testing.expectEqual(Datatype.float32, Datatype.fromProto(0).?);
    try std.testing.expectEqual(Datatype.float32, Datatype.fromProto(1).?);
    try std.testing.expectEqual(Datatype.uint8, Datatype.fromProto(2).?);
    try std.testing.expectEqual(Datatype.float16, Datatype.fromProto(3).?);
    try std.testing.expectEqual(@as(?Datatype, null), Datatype.fromProto(4));

    // Round trip, but only for the values that name a storage type: Default
    // has no counterpart here by construction.
    for ([_]Datatype{ .float32, .float16, .uint8 }) |d| {
        try std.testing.expectEqual(d, Datatype.fromProto(d.toProto()).?);
    }
    // The tag values are the on-disk format and must not drift.
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(Datatype.float32));
}

test "§8.3's ingest normalisation rule is per datatype, not global" {
    try std.testing.expect(Datatype.float32.normalisesAtIngest(.cosine));
    try std.testing.expect(Datatype.float16.normalisesAtIngest(.cosine));
    // The divergence, verified against qdrant's source rather than assumed.
    try std.testing.expect(!Datatype.uint8.normalisesAtIngest(.cosine));

    try std.testing.expect(Datatype.float32.cosineIsDot(.cosine));
    try std.testing.expect(!Datatype.uint8.cosineIsDot(.cosine));
    // Every non-cosine metric is its own kernel regardless of storage.
    for ([_]Datatype{ .float32, .float16, .uint8 }) |d| {
        try std.testing.expect(d.cosineIsDot(.dot));
        try std.testing.expect(d.cosineIsDot(.euclid));
        try std.testing.expect(!d.normalisesAtIngest(.euclid));
    }
}

test "element sizes are what the arena arithmetic assumes" {
    try std.testing.expectEqual(@as(usize, 4), Datatype.float32.elemSize());
    try std.testing.expectEqual(@as(usize, 2), Datatype.float16.elemSize());
    try std.testing.expectEqual(@as(usize, 1), Datatype.uint8.elemSize());
}

test "the vector conversions agree with the element-at-a-time ones" {
    // Including the ragged tail past the last full vector, and the values that
    // decide the clamp, since the whole point of the scalar version is that it
    // reproduces Qdrant exactly.
    var prng = std.Random.DefaultPrng.init(0x77aa);
    const rnd = prng.random();
    for ([_]usize{ 1, 15, 16, 17, 33, 128, 1536 }) |dim| {
        const src = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(src);
        for (src, 0..) |*x, i| {
            x.* = switch (i % 7) {
                0 => rnd.floatNorm(f32) * 400,
                1 => -rnd.float(f32) * 10,
                2 => 255.0,
                3 => 300.0,
                4 => std.math.nan(f32),
                5 => 0.4,
                else => rnd.float(f32) * 255,
            };
        }

        const halves = try std.testing.allocator.alloc(f16, dim);
        defer std.testing.allocator.free(halves);
        convertToF16(halves, src);
        for (src, halves) |x, h| {
            // NaN converts to NaN, and NaN never equals itself, so the
            // comparison is on the property rather than the value. f16 keeps
            // NaN a NaN, which is what `f16::from_f32` does too.
            if (std.math.isNan(x)) {
                try std.testing.expect(std.math.isNan(h));
            } else {
                try std.testing.expectEqual(toF16(x), h);
            }
        }

        const bytes = try std.testing.allocator.alloc(u8, dim);
        defer std.testing.allocator.free(bytes);
        convertToU8(bytes, src);
        for (src, bytes) |x, b| try std.testing.expectEqual(toU8(x), b);
    }
}
