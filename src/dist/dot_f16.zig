//! `dot_f16`, half-precision distance kernels.
//!
//! §6.6.2, the fp16 row: "AVX2: `vcvtph2ps` (F16C) convert-then-compute /
//! AVX-512: AVX512-FP16 native, or `vdpbf16ps` for bf16, relevant to bfb's
//! `--datatype Float16`."
//!
//! §5.2 gives fp16 its place in the cost model: 1536 B/vector at d=768, 24 cache
//! lines, ~180 ns modelled cost cold, exactly half fp32's. It sits above the
//! ~12-line threshold where a single dependent fetch still fills the line fill
//! buffers, so unlike binary and PQ it stays bandwidth bound and should show
//! *little* benefit from wider SIMD. That makes it a second test of §6.6.1's
//! prediction, at a different point on the working-set curve than fp32.
//!
//! ## Convert-then-compute, deliberately
//!
//! Both kernels widen f16 to f32 and accumulate in f32, rather than
//! accumulating in f16 natively even where AVX512-FP16 would allow it.
//!
//! That is an accuracy decision, not an oversight. f16 has an 11-bit
//! significand; summing 768 products into an f16 accumulator loses roughly
//! `log₂(768) ≈ 10` bits to accumulation error alone, which is the entire
//! significand. The result would not be within any tolerance §8.4 could
//! calibrate, and the §8.5 T1 tier would fail on the storage format rather than
//! on anything we did wrong.
//!
//! Qdrant makes the same choice for the same reason, so matching it is also
//! what conformance requires. Native f16 accumulation is available behind
//! `DotF16Native` for the microbenchmark, so the cost of the accuracy can be
//! quoted rather than assumed.

const std = @import("std");
const common = @import("common.zig");

/// f16 dot product with f32 accumulation. The default.
pub fn DotF16(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const H = @Vector(L, f16);
        const block = L * NACC;

        pub fn call(a: []const f16, b: []const f16) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const ah: H = a[off..][0..L].*;
                    const bh: H = b[off..][0..L].*;
                    // `vcvtph2ps` on F16C, or a native widening on AVX512-FP16.
                    const av: V = @floatCast(ah);
                    const bv: V = @floatCast(bh);
                    acc[k] = common.mulAdd(V, av, bv, acc[k]);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const ah: H = a[i..][0..L].*;
                const bh: H = b[i..][0..L].*;
                const av: V = @floatCast(ah);
                const bv: V = @floatCast(bh);
                acc[0] = common.mulAdd(V, av, bv, acc[0]);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                s += @as(f32, @floatCast(a[i])) * @as(f32, @floatCast(b[i]));
            }
            return s;
        }
    };
}

/// f16 Euclid with f32 accumulation. `−Σ(a−b)²`, computed directly per §8.3.
pub fn EuclidF16(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const H = @Vector(L, f16);
        const block = L * NACC;

        pub fn call(a: []const f16, b: []const f16) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const ah: H = a[off..][0..L].*;
                    const bh: H = b[off..][0..L].*;
                    const av: V = @floatCast(ah);
                    const bv: V = @floatCast(bh);
                    // Subtract *after* widening. Subtracting in f16 first would
                    // round the difference to f16 precision, which for near
                    // neighbours is where all the information is.
                    const d = av - bv;
                    acc[k] = common.mulAdd(V, d, d, acc[k]);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const ah: H = a[i..][0..L].*;
                const bh: H = b[i..][0..L].*;
                const av: V = @floatCast(ah);
                const bv: V = @floatCast(bh);
                const d = av - bv;
                acc[0] = common.mulAdd(V, d, d, acc[0]);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                const d = @as(f32, @floatCast(a[i])) - @as(f32, @floatCast(b[i]));
                s += d * d;
            }
            return -s;
        }
    };
}

/// `-Σ |a[i] - b[i]|` over f16, widened to f32 first.
///
/// The third metric the storage datatypes need. Same widen-then-compute shape
/// as `EuclidF16` and for the same reason: the difference of two near
/// neighbours is where the information is, and taking it in f16 rounds it away
/// before the absolute value ever sees it.
pub fn ManhattanF16(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const H = @Vector(L, f16);
        const block = L * NACC;

        pub fn call(a: []const f16, b: []const f16) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const ah: H = a[off..][0..L].*;
                    const bh: H = b[off..][0..L].*;
                    const av: V = @floatCast(ah);
                    const bv: V = @floatCast(bh);
                    acc[k] += @abs(av - bv);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const ah: H = a[i..][0..L].*;
                const bh: H = b[i..][0..L].*;
                const av: V = @floatCast(ah);
                const bv: V = @floatCast(bh);
                acc[0] += @abs(av - bv);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                s += @abs(@as(f32, @floatCast(a[i])) - @as(f32, @floatCast(b[i])));
            }
            return -s;
        }
    };
}

/// **Not on the conformance path.** f16 dot with f16 accumulation.
///
/// Exists so the microbenchmark can quote the throughput of native AVX512-FP16
/// accumulation against the accuracy it costs, rather than leaving the
/// trade-off as an assertion in a comment. See the module doc: at d=768 the
/// accumulation error consumes essentially the whole significand.
pub fn DotF16Native(comptime L: usize, comptime NACC: usize) type {
    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;
        pub const on_conformance_path = false;

        const H = @Vector(L, f16);
        const block = L * NACC;

        pub fn call(a: []const f16, b: []const f16) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]H = @splat(@as(H, @splat(0.0)));
            var i: usize = 0;
            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const av: H = a[off..][0..L].*;
                    const bv: H = b[off..][0..L].*;
                    acc[k] = common.mulAdd(H, av, bv, acc[k]);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: H = a[i..][0..L].*;
                const bv: H = b[i..][0..L].*;
                acc[0] = common.mulAdd(H, av, bv, acc[0]);
            }
            const folded = common.treeReduce(H, NACC, acc);
            var s: f16 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) s += a[i] * b[i];
            return @floatCast(s);
        }
    };
}

pub const dot_native = DotF16(common.nativeLanes(f32), common.default_accumulators);
pub const euclid_native = EuclidF16(common.nativeLanes(f32), common.default_accumulators);
pub const manhattan_native = ManhattanF16(common.nativeLanes(f32), common.default_accumulators);

/// Convert an f32 slice to f16 storage. Ingest-time only.
pub fn fromF32(dst: []f16, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*d, s| d.* = @floatCast(s);
}

pub fn toF32(dst: []f32, src: []const f16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*d, s| d.* = @floatCast(s);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

fn refDotF16(a: []const f16, b: []const f16) f64 {
    var acc: f64 = 0;
    for (a, b) |x, y| acc += @as(f64, @floatCast(x)) * @as(f64, @floatCast(y));
    return acc;
}

test "f16 dot matches the widened reference across widths and dims" {
    var prng = std.Random.DefaultPrng.init(0xf16d);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 7, 8, 16, 33, 128, 384, 768, 1536 };

    inline for ([_]usize{ 4, 8, 16 }) |L| {
        const K = DotF16(L, 8);
        for (dims) |d| {
            const a = try std.testing.allocator.alloc(f16, d);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(f16, d);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = @floatCast(rnd.floatNorm(f32));
                y.* = @floatCast(rnd.floatNorm(f32));
            }
            const got = K.call(a, b);
            const want = refDotF16(a, b);
            const scale = @max(1.0, @abs(want));
            try std.testing.expect(@abs(@as(f64, got) - want) / scale < 1e-5);
        }
    }
}

test "f16 euclid matches the widened reference" {
    var prng = std.Random.DefaultPrng.init(0xf16e);
    const rnd = prng.random();
    const d = 768;
    var a: [d]f16 = undefined;
    var b: [d]f16 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = @floatCast(rnd.floatNorm(f32));
        y.* = @floatCast(rnd.floatNorm(f32));
    }
    var want: f64 = 0;
    for (a, b) |x, y| {
        const df = @as(f64, @floatCast(x)) - @as(f64, @floatCast(y));
        want += df * df;
    }
    const got = euclid_native.call(&a, &b);
    try std.testing.expect(@abs(@as(f64, got) + want) / @max(1.0, want) < 1e-5);
}

test "f16 self-distance is exactly zero" {
    var prng = std.Random.DefaultPrng.init(2);
    const rnd = prng.random();
    var v: [768]f16 = undefined;
    for (&v) |*x| x.* = @floatCast(rnd.floatNorm(f32));
    try std.testing.expectEqual(@as(f32, 0.0), euclid_native.call(&v, &v));
}

test "native f16 accumulation is measurably worse, which is why it is off the path" {
    // Quantifies the module doc's claim rather than asserting it. At d=1536 the
    // f16 accumulator should lose several orders of magnitude of relative
    // accuracy against the f32-accumulating kernel.
    try std.testing.expect(!DotF16Native(8, 8).on_conformance_path);

    var prng = std.Random.DefaultPrng.init(0xacc);
    const rnd = prng.random();
    const d = 1536;
    var a: [d]f16 = undefined;
    var b: [d]f16 = undefined;
    // All-positive inputs so the errors accumulate rather than cancelling,
    // which is the realistic worst case for a normalised embedding's squared
    // terms and makes the test deterministic in its verdict.
    for (&a, &b) |*x, *y| {
        x.* = @floatCast(@abs(rnd.floatNorm(f32)) + 0.5);
        y.* = @floatCast(@abs(rnd.floatNorm(f32)) + 0.5);
    }
    const truth = refDotF16(&a, &b);
    const wide_acc = dot_native.call(&a, &b);
    const narrow_acc = DotF16Native(8, 8).call(&a, &b);

    const err_wide = @abs(@as(f64, wide_acc) - truth) / truth;
    const err_narrow = @abs(@as(f64, narrow_acc) - truth) / truth;

    try std.testing.expect(err_wide < 1e-5);
    try std.testing.expect(err_narrow > err_wide * 10.0);
}

test "fromF32/toF32 round-trip within f16 resolution" {
    var prng = std.Random.DefaultPrng.init(0x16a7);
    const rnd = prng.random();
    const d = 384;
    var src: [d]f32 = undefined;
    for (&src) |*x| x.* = rnd.floatNorm(f32);
    var half: [d]f16 = undefined;
    var back: [d]f32 = undefined;
    fromF32(&half, &src);
    toF32(&back, &half);
    for (src, back) |s, b| {
        // f16 has ~3 decimal digits; a relative tolerance of 1e-3 is the
        // format's resolution, not a slack allowance.
        try std.testing.expect(@abs(s - b) <= @max(@abs(s) * 1e-3, 1e-4));
    }
}

test "f16 manhattan matches a widened scalar reference" {
    var prng = std.Random.DefaultPrng.init(0x5a1);
    const rnd = prng.random();
    for ([_]usize{ 1, 8, 33, 128, 768 }) |dim| {
        const a = try std.testing.allocator.alloc(f16, dim);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(f16, dim);
        defer std.testing.allocator.free(b);
        for (a, b) |*x, *y| {
            x.* = @floatCast(rnd.floatNorm(f32));
            y.* = @floatCast(rnd.floatNorm(f32));
        }

        var want: f32 = 0;
        for (a, b) |x, y| want += @abs(@as(f32, @floatCast(x)) - @as(f32, @floatCast(y)));
        // Summation order differs between the tree reduction and the scalar
        // loop, so this is an fp32 accumulation tolerance, not a semantic one.
        try std.testing.expectApproxEqRel(-want, manhattan_native.call(a, b), 1e-5);
    }
}
