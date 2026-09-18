//! `dot_f32`, `@Vector(L, f32)`, 8 accumulators, fixed-order tree reduction.
//!
//! §6.6.1 predicts this kernel is the one where wider SIMD helps *least*: cold
//! fp32 over a 3.07 GB working set is bandwidth bound by ~25× (§5.1), so the
//! ALU is not the limiter. That prediction is the point. §6.6.1: "The AVX-512
//! delta is itself a diagnostic. If the cold fp32 kernel shows ~0% improvement
//! from 512-bit while the L3-resident binary kernel shows +40%, the §5 cost
//! model is validated."
//!
//! The same kernel is *also* the hot path for `--search-exact` (W9), which
//! streams with perfect hardware prefetching and is therefore much closer to
//! compute bound, the one fp32 cell where width should pay. Both cells are
//! measured; see `bench/micro`.

const std = @import("std");
const common = @import("common.zig");

/// A dot-product kernel at a fixed lane count and accumulator count.
///
/// `L`   , elements per vector register (4 = SSE, 8 = AVX2, 16 = AVX-512 for f32)
/// `NACC`, independent accumulator chains, power of two, ≥8 to cover FMA latency
pub fn Dot(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const block = L * NACC;

        /// `Σ aᵢbᵢ`.
        ///
        /// `@setFloatMode(.optimized)` is scoped to this function only, per
        /// §6.6.3: "so FMA contraction and reassociation are permitted where
        /// they're safe and nowhere else." The reduction is written as a
        /// fixed tree (`treeReduce`), but under this float mode reassociation
        /// is licensed, so LLVM may re-tree it; what §8.7 guarantees is that
        /// the result is deterministic *per build* (same binary, same input,
        /// same bits), not that the emitted shape is the source's. What the
        /// float mode buys is contraction and the freedom to keep the
        /// accumulators in registers.
        pub fn call(a: []const f32, b: []const f32) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            // Main body: NACC independent FMA chains, no cross-chain dependency.
            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const av: V = a[off..][0..L].*;
                    const bv: V = b[off..][0..L].*;
                    // common.mulAdd is an explicit FMA on FMA hardware rather
                    // than a hope that the optimiser contracts
                    // `av * bv + acc[k]`. §11 lists "fails to contract FMA" as
                    // a real, invisible-without-checking risk; being explicit
                    // removes it from the list. On the no-FMA baseline arm it
                    // is a plain mul+add instead of a per-lane libcall (see
                    // the helper's doc).
                    acc[k] = common.mulAdd(V, av, bv, acc[k]);
                }
            }

            // Vector tail: whole registers that did not fill a full block.
            // Folded into chain 0, a fixed choice, so the shape stays
            // deterministic for every length.
            while (i + L <= a.len) : (i += L) {
                const av: V = a[i..][0..L].*;
                const bv: V = b[i..][0..L].*;
                acc[0] = common.mulAdd(V, av, bv, acc[0]);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);

            // Scalar epilogue for dims that are not a multiple of the lane
            // count. §6.6.2 notes AVX-512 could predicate this away with mask
            // registers at zero cost; for the dims we care about (128/384/768/
            // 1536) this loop never executes, so it is correctness scaffolding
            // rather than a hot path.
            while (i < a.len) : (i += 1) s += a[i] * b[i];
            return s;
        }
    };
}

/// The instantiation for the compiled target's native width.
pub const native = Dot(common.nativeLanes(f32), common.default_accumulators);

// -------------------------------------------------------------------------
// Tests, every instantiation is checked against the strict-order reference.
// -------------------------------------------------------------------------

const reference = @import("reference.zig");

test "dot matches the strict reference across widths, dims, and tails" {
    var prng = std.Random.DefaultPrng.init(0xd07f32);
    const rnd = prng.random();

    // Dims deliberately include the matrix dims (128/384/768/1536), values that
    // exercise every tail path, and small awkward ones.
    const dims = [_]usize{ 1, 3, 4, 7, 8, 15, 16, 31, 33, 64, 100, 128, 384, 768, 960, 1536 };

    inline for ([_]usize{ 4, 8, 16 }) |L| {
        const K = Dot(L, 8);
        for (dims) |d| {
            const a = try std.testing.allocator.alloc(f32, d);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(f32, d);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = rnd.floatNorm(f32);
                y.* = rnd.floatNorm(f32);
            }

            const got = K.call(a, b);
            const want = reference.dotWide(a, b);
            // Blocked summation is *more* accurate than the strict left-to-right
            // reference, so compare against the fp64 accumulation. Tolerance is
            // relative to the magnitude, which grows as √d.
            const scale = @max(1.0, @abs(want));
            const rel = @abs(@as(f64, got) - want) / scale;
            try std.testing.expect(rel < 1e-5);
        }
    }
}

test "dot is exact on integer-valued inputs" {
    // Integers below 2²⁴ are exactly representable and their products and sums
    // stay exact for these sizes, so any reassociation is invisible and we can
    // assert exact equality. This catches indexing bugs that a tolerance-based
    // test would let through.
    const d = 768;
    var a: [d]f32 = undefined;
    var b: [d]f32 = undefined;
    for (0..d) |i| {
        a[i] = @floatFromInt(i % 7);
        b[i] = @floatFromInt(i % 5);
    }
    var expect: f64 = 0;
    for (0..d) |i| expect += @as(f64, a[i]) * @as(f64, b[i]);

    inline for ([_]usize{ 4, 8, 16 }) |L| {
        inline for ([_]usize{ 1, 2, 8 }) |NACC| {
            const got = Dot(L, NACC).call(&a, &b);
            try std.testing.expectEqual(@as(f32, @floatCast(expect)), got);
        }
    }
}

test "dot is deterministic across repeated calls" {
    // §8.7: "Same input + same build + same thread count → bit-identical
    // output." Weak but non-vacuous: catches accumulator state leaking between
    // calls.
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var a: [768]f32 = undefined;
    var b: [768]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }
    const first = native.call(&a, &b);
    for (0..64) |_| try std.testing.expectEqual(first, native.call(&a, &b));
}

test "dot of a unit vector with itself is 1" {
    const d = 1536;
    var v: [d]f32 = undefined;
    const val: f32 = 1.0 / @sqrt(@as(f32, d));
    @memset(&v, val);
    const got = native.call(&v, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got, 1e-6);
}
