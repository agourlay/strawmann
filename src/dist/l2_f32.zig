//! `l2_f32`, Euclid and Manhattan internal similarities.
//!
//! §6.6.3 offers "fused sub+fma, or |a|²+|b|²−2ab on normalised data" as the
//! two implementation choices. This file implements the first and *deliberately
//! does not* implement the second on the conformance path. §8.3:
//!
//!   "`(a−b)²` is computed directly, not via the `|a|² + |b|² − 2ab` expansion.
//!    The expansion is tempting because it turns L2 into a dot product with
//!    precomputed norms, but it suffers catastrophic cancellation exactly
//!    where it matters, since the error scales with `|a|²` rather than with the
//!    (small) distance between near neighbours. For close neighbours the
//!    relative error can be several orders of magnitude worse. If strawmann
//!    wants the expansion for speed, it must be a separate, separately-
//!    validated mode with its own tolerance, never silently substituted into
//!    the conformance path."
//!
//! So the expansion lives here too, as `EuclidExpanded`, but it is a distinct
//! type with its own name, its own tolerance, and a doc comment saying what it
//! costs. It can never be reached by accident from the default path.

const std = @import("std");
const common = @import("common.zig");

/// Euclid internal similarity: `−Σ(aᵢ−bᵢ)²`, computed directly.
pub fn Euclid(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const block = L * NACC;

        pub fn call(a: []const f32, b: []const f32) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const av: V = a[off..][0..L].*;
                    const bv: V = b[off..][0..L].*;
                    const d = av - bv;
                    // sub then FMA: two instructions, no cancellation risk.
                    acc[k] = common.mulAdd(V, d, d, acc[k]);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: V = a[i..][0..L].*;
                const bv: V = b[i..][0..L].*;
                const d = av - bv;
                acc[0] = common.mulAdd(V, d, d, acc[0]);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                const d = a[i] - b[i];
                s += d * d;
            }
            // §8.3: internal similarity is the *negated* squared distance, so
            // that "higher is better" holds uniformly across every metric and
            // the top-k heap never needs to know which metric it is serving.
            return -s;
        }
    };
}

/// Manhattan internal similarity: `−Σ|aᵢ−bᵢ|`.
pub fn Manhattan(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const block = L * NACC;

        pub fn call(a: []const f32, b: []const f32) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(a.len == b.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const av: V = a[off..][0..L].*;
                    const bv: V = b[off..][0..L].*;
                    // @abs on a float vector lowers to an andps/andpd with the
                    // sign-bit mask, one instruction, no branch.
                    acc[k] += @abs(av - bv);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: V = a[i..][0..L].*;
                const bv: V = b[i..][0..L].*;
                acc[0] += @abs(av - bv);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) s += @abs(a[i] - b[i]);
            return -s;
        }
    };
}

/// **Not on the conformance path.** `−(|a|² + |b|² − 2·a·b)`, using a
/// precomputed `|a|²` for the stored vector.
///
/// This turns L2 into a dot product, which matters because it lets the Euclid
/// path reuse the (VNNI-accelerated, quantization-friendly) dot kernel instead
/// of needing its own. The cost is catastrophic cancellation for near
/// neighbours, precisely the vectors a k-NN search cares about, because it
/// subtracts two large nearly-equal quantities to obtain a small one.
///
/// §8.3 permits this "as a separate, separately-validated mode with its own
/// tolerance". Callers must opt in explicitly and must not compare its output
/// against Qdrant's Euclid scores under the §8.4 ε calibrated for the direct
/// form. `reference.zig`'s test quantifies the gap: several orders of magnitude
/// of relative error on vectors far from the origin but close to each other.
pub fn EuclidExpanded(comptime L: usize, comptime NACC: usize) type {
    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;
        pub const on_conformance_path = false;

        const DotK = @import("dot_f32.zig").Dot(L, NACC);

        /// `a_sq` is `Σaᵢ²` for the *query*, `b_sq` for the stored vector -
        /// normally precomputed once at ingest and kept beside the vector.
        pub fn call(a: []const f32, b: []const f32, a_sq: f32, b_sq: f32) f32 {
            @setFloatMode(.optimized);
            return -(a_sq + b_sq - 2.0 * DotK.call(a, b));
        }
    };
}

pub const euclid_native = Euclid(common.nativeLanes(f32), common.default_accumulators);
pub const manhattan_native = Manhattan(common.nativeLanes(f32), common.default_accumulators);

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const reference = @import("reference.zig");
const Metric = @import("metric.zig").Metric;

test "euclid matches the strict reference across widths, dims, and tails" {
    var prng = std.Random.DefaultPrng.init(0xe0c11d);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 3, 4, 7, 8, 15, 16, 31, 33, 100, 128, 384, 768, 960, 1536 };

    inline for ([_]usize{ 4, 8, 16 }) |L| {
        const K = Euclid(L, 8);
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
            const want = reference.euclidWide(a, b);
            const scale = @max(1.0, @abs(want));
            try std.testing.expect(@abs(@as(f64, got) - want) / scale < 1e-5);
        }
    }
}

test "manhattan matches the strict reference across widths, dims, and tails" {
    var prng = std.Random.DefaultPrng.init(0x11a11a);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 3, 7, 8, 15, 16, 33, 128, 384, 768, 1536 };

    inline for ([_]usize{ 4, 8, 16 }) |L| {
        const K = Manhattan(L, 8);
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
            const want = reference.manhattanWide(a, b);
            const scale = @max(1.0, @abs(want));
            try std.testing.expect(@abs(@as(f64, got) - want) / scale < 1e-5);
        }
    }
}

test "euclid is exact on integer-valued inputs" {
    const d = 384;
    var a: [d]f32 = undefined;
    var b: [d]f32 = undefined;
    for (0..d) |i| {
        a[i] = @floatFromInt(i % 11);
        b[i] = @floatFromInt(i % 3);
    }
    var expect: f64 = 0;
    for (0..d) |i| {
        const df = @as(f64, a[i]) - @as(f64, b[i]);
        expect += df * df;
    }
    inline for ([_]usize{ 4, 8, 16 }) |L| {
        const got = Euclid(L, 8).call(&a, &b);
        try std.testing.expectEqual(@as(f32, @floatCast(-expect)), got);
    }
}

test "§8.3: the full Euclid pipeline reproduces the documented score" {
    // Internal similarity ranks; postprocess produces the returned score.
    const a = [_]f32{ 0, 0, 0, 0 };
    const b = [_]f32{ 3, 4, 0, 0 };
    const sim = euclid_native.call(&a, &b);
    try std.testing.expectEqual(@as(f32, -25.0), sim);
    try std.testing.expectEqual(@as(f32, 5.0), Metric.euclid.postprocess(sim));
}

test "§8.3: the full Manhattan pipeline reproduces the documented score" {
    const a = [_]f32{ 0, 0, 0, 0 };
    const b = [_]f32{ 3, -4, 1, 0 };
    const sim = manhattan_native.call(&a, &b);
    try std.testing.expectEqual(@as(f32, -8.0), sim);
    try std.testing.expectEqual(@as(f32, 8.0), Metric.manhattan.postprocess(sim));
}

test "self-distance is zero for euclid and manhattan" {
    // §8.6 metamorphic property: score(x, x) is maximal. For the negated
    // distances that maximum is exactly 0.
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var v: [768]f32 = undefined;
    for (&v) |*x| x.* = rnd.floatNorm(f32);
    try std.testing.expectEqual(@as(f32, 0.0), euclid_native.call(&v, &v));
    try std.testing.expectEqual(@as(f32, 0.0), manhattan_native.call(&v, &v));
}

test "EuclidExpanded is off the conformance path and measurably worse" {
    try std.testing.expect(!EuclidExpanded(8, 8).on_conformance_path);

    // Reproduce §8.3's warning as a number: far from origin, close together.
    const n = 256;
    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    for (0..n) |i| {
        a[i] = 500.0 + @as(f32, @floatFromInt(i)) * 0.5;
        b[i] = a[i] + 0.002;
    }
    const a_sq = reference.squaredLength(&a);
    const b_sq = reference.squaredLength(&b);

    const truth = reference.euclidWide(&a, &b);
    const direct = euclid_native.call(&a, &b);
    const expanded = EuclidExpanded(8, 8).call(&a, &b, a_sq, b_sq);

    const err_direct = @abs(@as(f64, direct) - truth) / @abs(truth);
    const err_expanded = @abs(@as(f64, expanded) - truth) / @abs(truth);
    try std.testing.expect(err_direct < 1e-4);
    try std.testing.expect(err_expanded > err_direct * 50.0);
}
