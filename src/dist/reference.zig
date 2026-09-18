//! §8.2, oracle tier 2, strawmann's own strict-order fp32 reference.
//!
//! "Scalar, single accumulator, no FMA, no reassociation. Isolates 'our SIMD
//! kernel is wrong' from 'our algorithm is wrong.'"
//!
//! Nothing in this file may be optimised for speed, ever. Its entire value is
//! that it is obviously correct by inspection and that its rounding behaviour
//! is the textbook one. If a SIMD kernel disagrees with this by more than the
//! calibrated ε of §8.4, the SIMD kernel is the suspect.
//!
//! Deliberately absent: `@setFloatMode(.optimized)`. Zig's default float mode
//! is `.strict`, which forbids both reassociation and mul+add contraction into
//! FMA. That default is load-bearing here, do not "tidy" it by adding a float
//! mode annotation.
//!
//! Also provided: an fp64 accumulation variant. This is *not* the fp64 oracle
//! of §8.2 tier 1 (that one is exhaustive, lives in `conformance/oracle/`, and
//! computes in double throughout). This is the narrower question "how much of
//! the delta is accumulation error in fp32?", answered by keeping the inputs
//! in fp32 and only widening the accumulator.

const std = @import("std");

// -------------------------------------------------------------------------
// fp32, single accumulator, strict order. The reference.
// -------------------------------------------------------------------------

/// `Σ aᵢbᵢ`, the internal similarity for Dot and (on normalised data) Cosine.
pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: f32 = 0.0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

/// `−Σ(aᵢ−bᵢ)²`, the internal similarity for Euclid.
///
/// §8.3: computed directly as `(a−b)²`, never as `|a|² + |b|² − 2ab`. The
/// expansion is tempting because it turns L2 into a dot product with
/// precomputed norms, but it suffers catastrophic cancellation exactly where it
/// matters: the error scales with `|a|²` rather than with the (small) distance
/// between near neighbours.
pub fn euclid(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: f32 = 0.0;
    for (a, b) |x, y| {
        const d = x - y;
        acc += d * d;
    }
    return -acc;
}

/// `−Σ|aᵢ−bᵢ|`, the internal similarity for Manhattan.
pub fn manhattan(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: f32 = 0.0;
    for (a, b) |x, y| acc += @abs(x - y);
    return -acc;
}

/// `Σ xᵢ²`, the squared length, as Qdrant computes it before deciding whether
/// to normalise. See `norm.zig` for why the *squared* length is the thing being
/// tested against 1.0.
pub fn squaredLength(v: []const f32) f32 {
    var acc: f32 = 0.0;
    for (v) |x| acc += x * x;
    return acc;
}

// -------------------------------------------------------------------------
// fp64 accumulation over fp32 inputs.
//
// Used by §8.4's tolerance calibration to separate "accumulation error" from
// "kernel bug": if a SIMD kernel matches `dotWide` more closely than it matches
// `dot`, the SIMD kernel is not wrong, it is simply summing in a better order
// than the strict left-to-right reference, which blocked/tree summation does.
// -------------------------------------------------------------------------

pub fn dotWide(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var acc: f64 = 0.0;
    for (a, b) |x, y| acc += @as(f64, x) * @as(f64, y);
    return acc;
}

pub fn euclidWide(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var acc: f64 = 0.0;
    for (a, b) |x, y| {
        const d = @as(f64, x) - @as(f64, y);
        acc += d * d;
    }
    return -acc;
}

pub fn manhattanWide(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var acc: f64 = 0.0;
    for (a, b) |x, y| acc += @abs(@as(f64, x) - @as(f64, y));
    return -acc;
}

pub fn squaredLengthWide(v: []const f32) f64 {
    var acc: f64 = 0.0;
    for (v) |x| acc += @as(f64, x) * @as(f64, x);
    return acc;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "dot of orthogonal basis vectors is zero" {
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };
    try std.testing.expectEqual(@as(f32, 0.0), dot(&a, &b));
}

test "euclid returns the negated squared distance" {
    const a = [_]f32{ 0, 0, 0 };
    const b = [_]f32{ 3, 4, 0 };
    // 3² + 4² = 25, negated.
    try std.testing.expectEqual(@as(f32, -25.0), euclid(&a, &b));
}

test "manhattan returns the negated L1 distance" {
    const a = [_]f32{ 0, 0, 0 };
    const b = [_]f32{ 3, -4, 1 };
    try std.testing.expectEqual(@as(f32, -8.0), manhattan(&a, &b));
}

test "§8.3 trap 2: the |a|²+|b|²−2ab expansion loses precision on near neighbours" {
    // This is the concrete demonstration of why the expansion is banned from
    // the conformance path. Two vectors that are far from the origin but very
    // close to each other: the direct form keeps full relative precision, the
    // expansion cancels catastrophically because it subtracts two large,
    // nearly-equal quantities.
    const n = 128;
    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    for (0..n) |i| {
        a[i] = 1000.0 + @as(f32, @floatFromInt(i));
        b[i] = a[i] + 0.001; // a tiny, exactly representable-ish offset
    }

    const direct = -euclid(&a, &b); // Σ(a−b)²
    const expansion = squaredLength(&a) + squaredLength(&b) - 2.0 * dot(&a, &b);

    // Ground truth in double precision.
    const truth = -euclidWide(&a, &b);

    const err_direct = @abs(@as(f64, direct) - truth) / truth;
    const err_expansion = @abs(@as(f64, expansion) - truth) / truth;

    // The direct form should be accurate to near fp32 epsilon; the expansion
    // should be orders of magnitude worse. We assert the *ordering* rather than
    // absolute magnitudes so the test is not brittle across hosts.
    try std.testing.expect(err_direct < 1e-4);
    try std.testing.expect(err_expansion > err_direct * 100.0);
}

test "wide and narrow references agree closely on well-conditioned input" {
    var prng = std.Random.DefaultPrng.init(0xf00d);
    const rnd = prng.random();
    const n = 768;
    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    for (0..n) |i| {
        a[i] = rnd.floatNorm(f32);
        b[i] = rnd.floatNorm(f32);
    }
    const narrow = dot(&a, &b);
    const wide = dotWide(&a, &b);
    // d=768 normal-distributed: |dot| is O(√768) ≈ 28, so an absolute
    // tolerance of 1e-2 is very loose and only catches gross errors.
    try std.testing.expect(@abs(@as(f64, narrow) - wide) < 1e-2);
}
