//! `norm`, ingest-time normalisation for Cosine.
//!
//! §8.3, trap 3, in full:
//!
//!   "Cosine normalisation short-circuits. Qdrant computes `length = Σx²` and,
//!    if `length < f32::EPSILON || |length − 1.0| ≤ 1e-6`, **returns the vector
//!    untouched**. Note it tests the *squared* length against 1.0 before taking
//!    the square root. A vector that is already near-normalised is therefore
//!    stored exactly as sent, while one just outside that window is divided
//!    through. Replicating the divide but not the short-circuit produces a
//!    small population of systematically-off points, the worst kind of bug,
//!    because it passes on random data and fails on real normalised embeddings.
//!    Also: it is a **division by `length`**, not a multiply by a reciprocal,
//!    and definitely not `rsqrtps` (whose ~12-bit approximation would blow ε to
//!    ~1e-3)."
//!
//! ## Which `Σx²` Qdrant actually computes, and why we compute the same one
//!
//! The short-circuit is a **discontinuous decision**. Two implementations that
//! compute `Σx²` to within 1e-7 of each other still agree on the *value*, but
//! for a vector sitting near the `|length − 1.0| ≤ 1e-6` boundary they can
//! disagree on the *branch*, one stores the vector untouched, the other
//! divides it through. That is not an ε-sized disagreement in the score; it is
//! a different stored vector, and it shows up as a small population of points
//! that are systematically wrong while every other point is bit-perfect.
//!
//! Real normalised embeddings, the dbpedia-openai and BEIR tiers of §4.2 -
//! sit *exactly* on that boundary by construction. This is the population the
//! bug would hit, and random test data would never reveal it.
//!
//! So the sum must be computed **in the order Qdrant computes it**, and Qdrant
//! does not have one order: `CosineMetric::preprocess` (Qdrant 1.19.0,
//! `lib/segment/src/spaces/simple.rs`) picks at runtime, in this priority:
//!
//!   1. `cosine_preprocess_avx` (`simple_avx.rs`) when
//!      `is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")`
//!      and `dim >= MIN_DIM_SIZE_AVX (32)`: four 8-lane `vfmadd231ps`
//!      accumulators over blocks of 32, reduced by `four_way_hsum` →
//!      `hsum256_ps_avx` in a fixed shuffle order, then the `dim % 32` tail
//!      added scalar left-to-right;
//!   2. `cosine_preprocess_sse` (`simple_sse.rs`) when SSE is detected (always,
//!      on x86_64) and `dim >= MIN_DIM_SIZE_SIMD (16)`: four 4-lane `mulps` +
//!      `addps` accumulators (no FMA) over blocks of 16, each reduced by
//!      `hsum128_ps_sse` and the four partials added left-to-right, then the
//!      `dim % 16` tail scalar;
//!   3. the scalar `cosine_preprocess`: `.iter().map(|x| x * x).sum::<f32>()`,
//!      f32 accumulator, strict left-to-right.
//!
//! Every host in the §7.1 comparison is AVX+FMA and every §7.5 dimension is
//! ≥ 128, so the path Qdrant takes on our data is (1), which was previously
//! only replicated in a test here while the engine used (3). `pathFor` makes
//! the same three-way choice from the same cpuid facts and each `squaredLength*`
//! is an order-preserving transcription of the corresponding Rust, so the
//! result is bit-exact whichever ISA *we* were compiled for. The vector code
//! below uses portable `@Vector`s, and the AVX replica's fused step is
//! `@mulAdd` on an FMA build and `common.fmafExact` on one without: the
//! `@mulAdd` libcall on a no-FMA target (compiler_rt `fmaf`) is double-rounded
//! and would disagree with Qdrant's `vfmadd231ps` in the halfway case, so our
//! own ISA arm cannot be allowed to leak into the decision that way either.
//!
//! **Non-x86 hosts are not replicated.** Qdrant 1.19.0 on aarch64 takes
//! `cosine_preprocess_neon` (`simple_neon.rs`, 4 × 4-lane `vfmaq_f32`,
//! reduced by `vaddvq_f32`) for `dim >= 16`. This project is Linux x86-64
//! only (README), so `pathFor` returns `.scalar` off x86 and the conformance
//! claim is made for x86-64 only; see the `pathFor` note and its test.
//!
//! In 1.19.0 all three paths then take a real `sqrt` and **divide** every
//! component by it (`vector.into_iter().map(|x| x / length)`). Qdrant `dev`
//! after 2026-08 (PR #8650) changed the AVX path to multiply by
//! `1.0 / length.sqrt()`; that is a *later* version than the one docs/comparison-sift1m.md
//! pins, so it is not reproduced here. When the comparison target moves, the
//! `Scale` switch below is the one line to flip.
//!
//! Zig's default float mode is `.strict`, which guarantees no FMA contraction,
//! no reassociation and no vectorised reduction; the absence of a
//! `@setFloatMode(.optimized)` in this file is deliberate and load-bearing.
//!
//! The cost is one O(d) pass per vector at ingest and one per query. Ingest is
//! bandwidth bound (§5.1) and the query pass is one row against d, so this is
//! the cheapest possible place to buy exactness.

const std = @import("std");
const builtin = @import("builtin");
const dispatch = @import("dispatch.zig");
const common = @import("common.zig");

/// `f32::EPSILON` as Rust defines it, 2⁻²³. Note this is *machine epsilon*,
/// not the smallest positive float; Zig's `std.math.floatEps(f32)` is the same
/// value. Spelled out here so the comparison against Qdrant's source is
/// mechanical.
pub const f32_epsilon: f32 = std.math.floatEps(f32); // 1.1920929e-7

/// The tolerance Qdrant uses when deciding a vector is "already normalised".
pub const normalized_tolerance: f32 = 1.0e-6;

/// `MIN_DIM_SIZE_AVX` and `MIN_DIM_SIZE_SIMD` from `simple.rs`: below these the
/// SIMD paths are not worth their setup and Qdrant falls through.
pub const min_dim_avx: usize = 32;
pub const min_dim_simd: usize = 16;

/// The three `Σx²` orders Qdrant can compute, named after the Rust source file
/// each transcribes.
pub const Path = enum {
    /// `simple.rs::cosine_preprocess`, strict scalar.
    scalar,
    /// `simple_sse.rs::cosine_preprocess_sse`, 4 × 4-lane mul+add.
    sse,
    /// `simple_avx.rs::cosine_preprocess_avx`, 4 × 8-lane fmadd.
    avx,
};

/// How the components are scaled once the vector is known to need it.
/// 1.19.0 divides on every path; see the module doc for the `dev` change.
const Scale = enum { divide, reciprocal_multiply };
const scale: Scale = .divide;

/// The path Qdrant 1.19.0 takes for a vector of `dim` components **on this
/// host**, from the same facts its `is_x86_feature_detected!` consults.
///
/// Runtime, not comptime: the choice depends on the machine Qdrant would run
/// on, which is the machine we run on, not on the ISA arm we were built for.
/// The cpuid probe is cached in `dispatch`, so this is a load and two compares
/// per call.
pub fn pathFor(dim: usize) Path {
    if (builtin.cpu.arch == .x86_64) {
        if (dim >= min_dim_avx and dispatch.hostHasAvxFma()) return .avx;
        // `is_x86_feature_detected!("sse")` is unconditionally true on x86_64;
        // SSE2 is part of the base ISA.
        if (dim >= min_dim_simd) return .sse;
    }
    // Not x86-64: Qdrant would take `cosine_preprocess_neon` on aarch64 for
    // dim >= 16, which is *not* transcribed here (the project is x86-64 only,
    // see the module doc), so this is the scalar order and the bit-exactness
    // claim does not hold on such a host. Reaching here on x86-64 means
    // dim < 16, where Qdrant is scalar too.
    return .scalar;
}

/// `Σ xᵢ²`, in the exact order and precision Qdrant computes it on this host.
///
/// **Do not "improve" any of the three orders.** See the module doc. The value
/// feeds a discontinuous branch, so a more-accurate sum is a *worse* answer
/// here: it makes us disagree with Qdrant about which side of the boundary a
/// vector falls on.
pub fn squaredLength(v: []const f32) f32 {
    return squaredLengthOn(pathFor(v.len), v);
}

/// `Σ xᵢ²` on an explicit path, for tests and for the conformance harness to
/// pin a path independently of the host.
pub fn squaredLengthOn(path: Path, v: []const f32) f32 {
    return switch (path) {
        .scalar => squaredLengthScalar(v),
        .sse => squaredLengthSse(v),
        .avx => squaredLengthAvx(v),
    };
}

/// Rust's `.iter().map(|x| x * x).sum::<f32>()`: f32 accumulator, strict
/// left-to-right, no FMA, no reassociation, one accumulator.
pub fn squaredLengthScalar(v: []const f32) f32 {
    var acc: f32 = 0.0;
    for (v) |x| acc += x * x;
    return acc;
}

/// `simple_avx.rs::cosine_preprocess_avx`, the length half, transcribed
/// intrinsic by intrinsic:
///
/// ```rust
/// let m = n - (n % 32);
/// while i < m {
///     sum256_k = _mm256_fmadd_ps(m256_k, m256_k, sum256_k);   // k = 1..4
///     i += 32;
/// }
/// let mut length = four_way_hsum(sum256_1, sum256_2, sum256_3, sum256_4);
/// for i in 0..n - m { length += (*ptr.add(i)).powi(2); }
/// ```
///
/// `_mm256_fmadd_ps` is a single rounding, so `@mulAdd`, not `v * v + acc`.
/// `powi(2)` is `x * x` (LLVM folds `llvm.powi(x, 2)`), added scalar in order.
pub fn squaredLengthAvx(v: []const f32) f32 {
    const V = @Vector(8, f32);
    const n = v.len;
    const m = n - (n % 32);
    var acc: [4]V = @splat(@as(V, @splat(0.0)));
    var i: usize = 0;
    while (i < m) : (i += 32) {
        inline for (0..4) |k| {
            const x: V = v[i + k * 8 ..][0..8].*;
            if (comptime common.has_hw_fma) {
                acc[k] = @mulAdd(V, x, x, acc[k]);
            } else {
                // No hardware FMA on this build: `@mulAdd` would be the
                // double-rounded compiler_rt `fmaf`, see `common.fmafExact`.
                inline for (0..8) |l| acc[k][l] = common.fmafExact(x[l], x[l], acc[k][l]);
            }
        }
    }
    var length = fourWayHsum(acc[0], acc[1], acc[2], acc[3]);
    while (i < n) : (i += 1) length += v[i] * v[i];
    return length;
}

/// `simple_avx.rs::four_way_hsum`: `(a + b) + (c + d)` lane-wise, then
/// `hsum256_ps_avx` of the total.
fn fourWayHsum(a: @Vector(8, f32), b: @Vector(8, f32), c: @Vector(8, f32), d: @Vector(8, f32)) f32 {
    const sum1 = a + b;
    const sum2 = c + d;
    return hsum256(sum1 + sum2);
}

/// `simple_avx.rs::hsum256_ps_avx`, whose order is *not* the obvious
/// left-to-right one:
///
/// ```rust
/// let lr_sum = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
/// let hsum = _mm_hadd_ps(lr_sum, lr_sum);
/// f32::from_bits(_mm_extract_ps(hsum, 0)) + f32::from_bits(_mm_extract_ps(hsum, 1))
/// ```
///
/// i.e. `((x0+x4) + (x1+x5)) + ((x2+x6) + (x3+x7))`. `@reduce(.Add)` would be
/// a different tree and a different bit pattern; the earlier test replica of
/// this function used it and was therefore not exact.
fn hsum256(x: @Vector(8, f32)) f32 {
    const lo: @Vector(4, f32) = @shuffle(f32, x, undefined, [4]i32{ 0, 1, 2, 3 });
    const hi: @Vector(4, f32) = @shuffle(f32, x, undefined, [4]i32{ 4, 5, 6, 7 });
    // Lane-wise, so the operand order inside each lane does not matter (fp add
    // is commutative, just not associative).
    const lr = hi + lo;
    const p1 = lr[0] + lr[1];
    const p2 = lr[2] + lr[3];
    return p1 + p2;
}

/// `simple_sse.rs::cosine_preprocess_sse`, the length half:
///
/// ```rust
/// let m = n - (n % 16);
/// while i < m {
///     sum128_k = _mm_add_ps(_mm_mul_ps(m128_k, m128_k), sum128_k);   // k = 1..4
///     i += 16;
/// }
/// let mut length = hsum128_ps_sse(sum128_1) + hsum128_ps_sse(sum128_2)
///                + hsum128_ps_sse(sum128_3) + hsum128_ps_sse(sum128_4);
/// for i in 0..n - m { length += (*ptr.add(i)).powi(2); }
/// ```
///
/// Two roundings per lane per step (mul, then add), no FMA, and the four
/// partials are each reduced *before* being added left-to-right, unlike the
/// AVX path which adds the vectors first.
pub fn squaredLengthSse(v: []const f32) f32 {
    const V = @Vector(4, f32);
    const n = v.len;
    const m = n - (n % 16);
    var acc: [4]V = @splat(@as(V, @splat(0.0)));
    var i: usize = 0;
    while (i < m) : (i += 16) {
        inline for (0..4) |k| {
            const x: V = v[i + k * 4 ..][0..4].*;
            acc[k] = (x * x) + acc[k];
        }
    }
    var length = ((hsum128(acc[0]) + hsum128(acc[1])) + hsum128(acc[2])) + hsum128(acc[3]);
    while (i < n) : (i += 1) length += v[i] * v[i];
    return length;
}

/// `simple_sse.rs::hsum128_ps_sse`:
///
/// ```rust
/// let x64 = _mm_add_ps(x, _mm_movehl_ps(x, x));          // [x0+x2, x1+x3, ..]
/// let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55)); // x64[0] + x64[1]
/// ```
///
/// i.e. `(x0+x2) + (x1+x3)`.
fn hsum128(x: @Vector(4, f32)) f32 {
    const a = x[0] + x[2];
    const b = x[1] + x[3];
    return a + b;
}

/// Qdrant's `is_length_zero_or_normalized`, verbatim.
///
/// Takes the **squared** length. The `1.0` comparison happens before the square
/// root, testing `|√length − 1.0|` instead would accept a different (wider,
/// since √ compresses toward 1) set of vectors.
pub fn isLengthZeroOrNormalized(squared_len: f32) bool {
    return squared_len < f32_epsilon or @abs(squared_len - 1.0) <= normalized_tolerance;
}

/// Normalise `v` in place for Cosine storage, reproducing Qdrant's
/// `CosineMetric::preprocess` on this host.
///
/// Returns `true` if the vector was divided through, `false` if it was left
/// untouched by the short-circuit. The return value exists so that ingest can
/// count short-circuited vectors and the conformance harness can assert the
/// population matches Qdrant's, a divergence in that count localises the bug
/// immediately instead of leaving it as an unexplained score delta.
pub fn normalizeInPlace(v: []f32) bool {
    return normalizeInPlaceOn(pathFor(v.len), v);
}

/// `normalizeInPlace` on an explicit path.
pub fn normalizeInPlaceOn(path: Path, v: []f32) bool {
    const squared_len = squaredLengthOn(path, v);
    if (isLengthZeroOrNormalized(squared_len)) return false;

    // A real square root and a real division, per §8.3. `rsqrtps` and its
    // ~12-bit approximation would blow the tolerance to ~1e-3, three orders of
    // magnitude past the ε we are trying to earn. Newton-Raphson refinement of
    // rsqrt would get closer but still not bit-match a divide, and this is
    // ingest-time code where the divide is free.
    const length = @sqrt(squared_len);
    switch (scale) {
        .divide => for (v) |*x| {
            x.* = x.* / length;
        },
        .reciprocal_multiply => {
            const inv = 1.0 / length;
            for (v) |*x| x.* = x.* * inv;
        },
    }
    return true;
}

/// Normalise from `src` into `dst`. Same semantics as `normalizeInPlace`,
/// used on the query path where the source is a decoded protobuf frame we do
/// not own.
pub fn normalizeInto(dst: []f32, src: []const f32) bool {
    std.debug.assert(dst.len == src.len);
    @memcpy(dst, src);
    return normalizeInPlace(dst);
}

/// Apply the metric's ingest preprocessing, if it has any.
///
/// The single funnel through which every stored and every queried vector
/// passes, so that "did we normalise the query the same way we normalised the
/// data?" has exactly one answer in the codebase.
pub fn preprocessInPlace(metric: @import("metric.zig").Metric, v: []f32) bool {
    if (!metric.normalisesAtIngest()) return false;
    return normalizeInPlace(v);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "f32_epsilon matches Rust's f32::EPSILON" {
    try std.testing.expectEqual(@as(f32, 1.1920929e-7), f32_epsilon);
}

test "§8.3 trap 3: an already-normalised vector is stored untouched" {
    // At d=128 the sum of 128 copies of (1/√128)² lands within 1e-6 of 1.0 on
    // every one of Qdrant's three paths, so the short-circuit fires and the
    // bytes are preserved exactly. A correct-looking implementation that
    // always divides would perturb every component here.
    const d = 128;
    var v: [d]f32 = undefined;
    const val: f32 = 1.0 / @sqrt(@as(f32, d));
    @memset(&v, val);

    for (std.enums.values(Path)) |p| {
        var w = v;
        const divided = normalizeInPlaceOn(p, &w);
        try std.testing.expect(!divided);
        try std.testing.expectEqualSlices(f32, &v, &w);
    }
    // And whatever this host selects, the same.
    var w = v;
    try std.testing.expect(!normalizeInPlace(&w));
    try std.testing.expectEqualSlices(f32, &v, &w);
}

test "§8.3 trap 3: the short-circuit window closes as dimension grows, per path" {
    // MEASURED, not assumed. See docs/cosine-shortcircuit.md.
    //
    // The short-circuit tests `Σx²` against 1.0 ± 1e-6. That sum accumulates
    // rounding error proportional to the number of terms *per accumulator*, so
    // the window is a function of dimension **and of the path**: the strict
    // scalar sum of a mathematically-unit vector leaves the window by d=384,
    // while the AVX path (32 lanes' worth of accumulators, fused mul-add) keeps
    // the same vector inside it at every dimension in the §7.5 matrix.
    //
    // This is exactly why the path must match Qdrant's rather than being the
    // "most accurate": on an AVX+FMA host Qdrant stores the d=1536 uniform unit
    // vector untouched and the strict scalar sum would divide it through, a
    // different stored vector for the same input.
    const uniform_unit_shortcircuits = struct {
        fn f(path: Path, comptime d: usize) bool {
            var v: [d]f32 = undefined;
            const val: f32 = 1.0 / @sqrt(@as(f32, d));
            @memset(&v, val);
            return !normalizeInPlaceOn(path, &v);
        }
    }.f;

    // Scalar: d=384 (|sqlen−1| ≈ 2.98e-6), d=768 (≈ 6.14e-6), d=1536
    // (≈ 1.22e-5), every dimension above 128 falls outside the window.
    try std.testing.expect(uniform_unit_shortcircuits(.scalar, 4));
    try std.testing.expect(uniform_unit_shortcircuits(.scalar, 128));
    try std.testing.expect(!uniform_unit_shortcircuits(.scalar, 384));
    try std.testing.expect(!uniform_unit_shortcircuits(.scalar, 768));
    try std.testing.expect(!uniform_unit_shortcircuits(.scalar, 1536));

    // AVX: 4 × 8 accumulators, each seeing d/32 terms with one rounding per
    // step. Inside the window at every matrix dimension: 384 → 0.9999998,
    // 768 → 1.0000001, 1536 → 0.9999996.
    try std.testing.expect(uniform_unit_shortcircuits(.avx, 128));
    try std.testing.expect(uniform_unit_shortcircuits(.avx, 384));
    try std.testing.expect(uniform_unit_shortcircuits(.avx, 768));
    try std.testing.expect(uniform_unit_shortcircuits(.avx, 1536));

    // SSE: 4 × 4 accumulators, two roundings per step, and still inside at
    // every matrix dimension: 384 → 1.0000001, 768 → 0.9999996, 1536 →
    // 1.0000007. Sixteen accumulators of d/16 terms is already enough.
    try std.testing.expect(uniform_unit_shortcircuits(.sse, 128));
    try std.testing.expect(uniform_unit_shortcircuits(.sse, 384));
    try std.testing.expect(uniform_unit_shortcircuits(.sse, 768));
    try std.testing.expect(uniform_unit_shortcircuits(.sse, 1536));

    // So the scalar path gives a different answer from either SIMD path for
    // the same d ≥ 384 input, and the engine's answer must be the host's
    // path's answer. On every x86_64 host that is one of the SIMD ones.
    const d = 768;
    var v: [d]f32 = undefined;
    @memset(&v, 1.0 / @sqrt(@as(f32, d)));
    try std.testing.expectEqual(uniform_unit_shortcircuits(pathFor(d), d), !normalizeInPlace(&v));
}

test "§8.3 trap 3: the test is on the SQUARED length, before the root" {
    // Construct a vector whose squared length is 1.0000005 (inside the 1e-6
    // window) and one whose squared length is 1.000_01 (outside it).
    //
    // If the implementation wrongly tested |√len − 1.0| ≤ 1e-6, the second
    // vector would also be accepted: √1.00001 ≈ 1.000005, whose distance from
    // 1.0 is ~5e-6, still outside, but the *window widths differ by 2×*, and
    // there is a band of vectors on which the two tests disagree. This test
    // pins the boundary at the squared value.
    try std.testing.expect(isLengthZeroOrNormalized(1.0000005));
    try std.testing.expect(!isLengthZeroOrNormalized(1.00001));

    // The disagreement band: squared length 1.000_003 is rejected by the
    // squared test (3e-6 > 1e-6) but its root is 1.0000015, which a root-based
    // test with the same tolerance would accept (1.5e-6 > 1e-6, also rejected,
    // so pick a value that lands between). 1.0000015 squared-distance is
    // 1.5e-6, rejected; root-distance ~7.5e-7, accepted. That is the bug.
    const in_band: f32 = 1.0000015;
    try std.testing.expect(!isLengthZeroOrNormalized(in_band));
    const root_distance = @abs(@sqrt(in_band) - 1.0);
    try std.testing.expect(root_distance <= normalized_tolerance);
}

test "§8.3 trap 3: a zero vector is left untouched, not divided by zero" {
    for (std.enums.values(Path)) |p| {
        var v = [_]f32{0.0} ** 128;
        const divided = normalizeInPlaceOn(p, &v);
        try std.testing.expect(!divided);
        for (v) |x| try std.testing.expectEqual(@as(f32, 0.0), x);
    }
}

test "a clearly unnormalised vector is divided through" {
    var v = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    const divided = normalizeInPlace(&v);
    try std.testing.expect(divided);
    // length = 5, so the result is (0.6, 0.8, 0, 0).
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 1e-7);
    // And it is now unit length.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), squaredLength(&v), 1e-6);
}

test "normalisation is idempotent at low dimension" {
    // The short-circuit is what makes the second pass a no-op, but only while
    // the accumulated error stays inside the window. See the next test.
    var prng = std.Random.DefaultPrng.init(0x0e0e);
    const rnd = prng.random();
    var v: [64]f32 = undefined;
    for (&v) |*x| x.* = rnd.floatNorm(f32) * 10.0;

    try std.testing.expect(normalizeInPlace(&v));
    const once = v;
    const divided_again = normalizeInPlace(&v);
    try std.testing.expect(!divided_again);
    try std.testing.expectEqualSlices(f32, &once, &v);
}

test "normalisation is NOT idempotent at the dimensions we benchmark" {
    // MEASURED on the strict scalar path. Of 2000 random normal vectors
    // normalised once and re-measured:
    //
    //   d=128   100.0% short-circuit on the second pass  (idempotent)
    //   d=384    99.0%
    //   d=768    92.8%
    //   d=1536   79.1%                                   (21% divide again)
    //
    // So at the headline dimension (dbpedia-openai, d=1536) roughly a fifth of
    // already-normalised vectors get divided a second time, by a divisor within
    // ~1e-6 of 1.0. The effect on the score is tiny; the effect on *bit
    // equality with Qdrant* is total, and it is entirely determined by whether
    // our sum of squares rounds the same way Qdrant's does.
    //
    // The two SIMD paths carry far less accumulated error (fewer terms per
    // accumulator, and one rounding per step on AVX) so their window is
    // effectively wide enough for random data: MEASURED 0/2000 divide again on
    // both, at every dimension. That is asserted too, because it is the reason
    // the population of "systematically-off points" only exists when the two
    // engines are on *different* paths.
    //
    // This test asserts the scalar population is non-empty rather than
    // pinning an exact rate, so it documents the phenomenon without being
    // brittle.
    var prng = std.Random.DefaultPrng.init(0xd15c0);
    const rnd = prng.random();
    const d = 1536;
    var v: [d]f32 = undefined;
    const trials = 512;

    for (std.enums.values(Path)) |p| {
        var divided_twice: usize = 0;
        for (0..trials) |_| {
            for (&v) |*x| x.* = rnd.floatNorm(f32);
            _ = normalizeInPlaceOn(p, &v);
            if (normalizeInPlaceOn(p, &v)) divided_twice += 1;
        }
        switch (p) {
            // Non-vacuous in both directions: the phenomenon is real, and it
            // is not everything (most vectors still short-circuit).
            .scalar => {
                try std.testing.expect(divided_twice > 0);
                try std.testing.expect(divided_twice < trials);
            },
            .sse, .avx => try std.testing.expectEqual(@as(usize, 0), divided_twice),
        }
    }
}

test "normalizeInto agrees with normalizeInPlace bit for bit" {
    var prng = std.Random.DefaultPrng.init(0x1234);
    const rnd = prng.random();
    var src: [768]f32 = undefined;
    for (&src) |*x| x.* = rnd.floatNorm(f32) * 3.0;

    var a = src;
    var b: [768]f32 = undefined;
    const da = normalizeInPlace(&a);
    const db = normalizeInto(&b, &src);
    try std.testing.expectEqual(da, db);
    try std.testing.expectEqualSlices(f32, &a, &b);
}

test "each squaredLength path is reproducible, and the paths differ" {
    // Non-associativity is the point: a vector built so that the strict sum
    // differs from a pairwise sum. We only assert reproducibility here, since
    // asserting a specific bit pattern would encode the host's rounding.
    var v: [1024]f32 = undefined;
    v[0] = 1.0e4;
    for (1..v.len) |i| v[i] = 1.0;
    for (std.enums.values(Path)) |p| {
        const first = squaredLengthOn(p, &v);
        for (0..32) |_| try std.testing.expectEqual(first, squaredLengthOn(p, &v));
    }
    // And the paths do differ on it, which is the whole reason they are three
    // functions: ulp(1e8) is 8, so 1e8 + 1 + 1 + ... in strict f32 order stays
    // 1e8, whereas the 31 AVX lanes that never see the 1e8 term keep their 32
    // ones each.
    try std.testing.expect(squaredLengthOn(.scalar, &v) != squaredLengthOn(.avx, &v));
}

test "preprocessInPlace only touches cosine" {
    var v = [_]f32{ 3.0, 4.0 };
    const before = v;

    try std.testing.expect(!preprocessInPlace(.dot, &v));
    try std.testing.expectEqualSlices(f32, &before, &v);
    try std.testing.expect(!preprocessInPlace(.euclid, &v));
    try std.testing.expectEqualSlices(f32, &before, &v);
    try std.testing.expect(!preprocessInPlace(.manhattan, &v));
    try std.testing.expectEqualSlices(f32, &before, &v);

    try std.testing.expect(preprocessInPlace(.cosine, &v));
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-7);
}

// -------------------------------------------------------------------------
// The Qdrant path replicas: selection, and bit-exactness against a
// hand-transcribed scalar-indexed reference of the Rust
// -------------------------------------------------------------------------

test "off x86-64, pathFor is scalar: the NEON path Qdrant would take is not replicated" {
    // Documents the limitation rather than hiding it: on aarch64 Qdrant 1.19.0
    // uses `cosine_preprocess_neon` for dim >= 16, and this project (Linux
    // x86-64 only) makes no bit-exactness claim there.
    if (comptime builtin.cpu.arch == .x86_64) return error.SkipZigTest;
    try std.testing.expectEqual(Path.scalar, pathFor(15));
    try std.testing.expectEqual(Path.scalar, pathFor(16));
    try std.testing.expectEqual(Path.scalar, pathFor(1536));
}

test "pathFor makes Qdrant 1.19.0's choice from this host's cpuid" {
    if (comptime builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    // Below MIN_DIM_SIZE_SIMD nothing SIMD is considered.
    try std.testing.expectEqual(Path.scalar, pathFor(0));
    try std.testing.expectEqual(Path.scalar, pathFor(15));
    // 16..31 is SSE regardless of AVX: MIN_DIM_SIZE_AVX is 32.
    try std.testing.expectEqual(Path.sse, pathFor(16));
    try std.testing.expectEqual(Path.sse, pathFor(31));
    // From 32 up it depends on the host, and only on the host.
    const want: Path = if (dispatch.hostHasAvxFma()) .avx else .sse;
    try std.testing.expectEqual(want, pathFor(32));
    try std.testing.expectEqual(want, pathFor(128));
    try std.testing.expectEqual(want, pathFor(1536));
    try std.testing.expectEqual(want, pathFor(1537));
}

/// `cosine_preprocess_avx`'s length, written the dumb way: 32 named f32
/// accumulator slots indexed `[register][lane]`, scalar `@mulAdd` per slot,
/// and the reduction spelled out element by element from the intrinsics.
/// No `@Vector`, no shuffles, so a mistake in the vector transcription above
/// (a wrong lane order in `hsum256`, say) cannot be repeated here.
fn avxLengthByHand(v: []const f32) f32 {
    var s: [4][8]f32 = @splat(@splat(0.0));
    const n = v.len;
    const m = n - (n % 32);
    var i: usize = 0;
    while (i < m) : (i += 32) {
        for (0..4) |r| for (0..8) |l| {
            const x = v[i + r * 8 + l];
            s[r][l] = common.fmaf(x, x, s[r][l]);
        };
    }
    // four_way_hsum: sum1 = a+b, sum2 = c+d, total = sum1+sum2, lane-wise.
    var total: [8]f32 = undefined;
    for (0..8) |l| total[l] = (s[0][l] + s[1][l]) + (s[2][l] + s[3][l]);
    // hsum256_ps_avx: lr = hi128 + lo128; hadd; extract 0 + extract 1.
    var lr: [4]f32 = undefined;
    for (0..4) |l| lr[l] = total[l + 4] + total[l];
    const hadd0 = lr[0] + lr[1];
    const hadd1 = lr[2] + lr[3];
    var length = hadd0 + hadd1;
    for (i..n) |j| length += v[j] * v[j];
    return length;
}

test "the no-FMA arm of the AVX replica reproduces the hardware fused step bit for bit" {
    // On a build without FMA `squaredLengthAvx` uses `common.fmafExact` per
    // lane; this checks, on an FMA host, that a by-hand transcription using
    // only that software step lands on the same bits as the hardware one, on
    // this kernel's real data shape (squares accumulated over lanes).
    if (!common.has_hw_fma) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x5f3a);
    const rnd = prng.random();
    const buf = try std.testing.allocator.alloc(f32, 1600);
    defer std.testing.allocator.free(buf);
    for ([_]usize{ 32, 64, 128, 384, 768, 1536, 1537, 1599 }) |d| {
        for ([_]f32{ 1.0, 1000.0, 3.0e4, 1.0e-3 }) |mag| {
            var round: usize = 0;
            while (round < 8) : (round += 1) {
                const v = buf[0..d];
                for (v) |*x| x.* = rnd.floatNorm(f32) * mag;
                var soft: [4][8]f32 = @splat(@splat(0.0));
                const m = d - (d % 32);
                var i: usize = 0;
                while (i < m) : (i += 32) {
                    for (0..4) |r| for (0..8) |l| {
                        const x = v[i + r * 8 + l];
                        soft[r][l] = common.fmafExact(x, x, soft[r][l]);
                    };
                }
                var hard: [4]@Vector(8, f32) = @splat(@as(@Vector(8, f32), @splat(0.0)));
                i = 0;
                while (i < m) : (i += 32) {
                    inline for (0..4) |k| {
                        const x: @Vector(8, f32) = v[i + k * 8 ..][0..8].*;
                        hard[k] = @mulAdd(@Vector(8, f32), x, x, hard[k]);
                    }
                }
                for (0..4) |r| {
                    const hr: [8]f32 = hard[r];
                    try std.testing.expectEqualSlices(f32, &hr, &soft[r]);
                }
            }
        }
    }
}

/// `cosine_preprocess_sse`'s length, same treatment.
fn sseLengthByHand(v: []const f32) f32 {
    var s: [4][4]f32 = @splat(@splat(0.0));
    const n = v.len;
    const m = n - (n % 16);
    var i: usize = 0;
    while (i < m) : (i += 16) {
        for (0..4) |r| for (0..4) |l| {
            const x = v[i + r * 4 + l];
            s[r][l] = (x * x) + s[r][l];
        };
    }
    var h: [4]f32 = undefined;
    for (0..4) |r| {
        // hsum128_ps_sse: x64 = x + movehl(x); x32 = x64[0] + x64[1].
        const x64_0 = s[r][0] + s[r][2];
        const x64_1 = s[r][1] + s[r][3];
        h[r] = x64_0 + x64_1;
    }
    var length = ((h[0] + h[1]) + h[2]) + h[3];
    for (i..n) |j| length += v[j] * v[j];
    return length;
}

test "the AVX and SSE length replicas are bit-exact against the by-hand transcription, tails included" {
    var prng = std.Random.DefaultPrng.init(0xa5a5);
    const rnd = prng.random();
    const buf = try std.testing.allocator.alloc(f32, 1600);
    defer std.testing.allocator.free(buf);

    // Every tail residue mod 32 at least once, plus the matrix dimensions and
    // one past each of them, plus large magnitudes so the accumulators are far
    // from 1.0 and rounding is exercised.
    var dims = std.ArrayList(usize).empty;
    defer dims.deinit(std.testing.allocator);
    for (0..64) |d| try dims.append(std.testing.allocator, d);
    for ([_]usize{ 127, 128, 129, 383, 384, 385, 767, 768, 769, 1535, 1536, 1537, 1599 }) |d| try dims.append(std.testing.allocator, d);

    var avx_differs: usize = 0;
    for (dims.items) |d| {
        for ([_]f32{ 1.0, 1000.0, 1.0e-3 }) |mag| {
            const v = buf[0..d];
            for (v) |*x| x.* = rnd.floatNorm(f32) * mag;
            try std.testing.expectEqual(avxLengthByHand(v), squaredLengthAvx(v));
            try std.testing.expectEqual(sseLengthByHand(v), squaredLengthSse(v));
            // And the strict scalar sum is a different number for most of the
            // wide ones, so the exactness above is not vacuous.
            if (d >= 384 and squaredLengthAvx(v) != squaredLengthScalar(v)) avx_differs += 1;
        }
    }
    try std.testing.expect(avx_differs > 0);
}

test "the normalised vector, not only the length, is bit-exact per path" {
    // The path decides the branch and the divisor; the divide itself is
    // per-component IEEE and identical everywhere. So a vector normalised by
    // the by-hand length must equal the path's output bit for bit.
    var prng = std.Random.DefaultPrng.init(0x5151);
    const rnd = prng.random();
    var v: [1000]f32 = undefined;
    for (0..16) |_| {
        for (&v) |*x| x.* = rnd.floatNorm(f32) * 4.0;
        var got = v;
        _ = normalizeInPlaceOn(.avx, &got);
        const len = @sqrt(avxLengthByHand(&v));
        for (v, got) |x, g| try std.testing.expectEqual(x / len, g);
    }
}

test "d=1536 near the 1e-6 boundary: the branch follows the AVX sum, and the scalar sum would disagree" {
    // The population §8.3 warns about: near-unit vectors at the headline
    // dimension. Take a unit vector and stretch it by (1 + t) with t uniform in
    // ±2e-6, so the squared length sweeps 1 ± 4e-6 across the ±1e-6 window
    // edges. On an AVX+FMA host Qdrant asks the AVX sum whether to divide, and
    // so must `normalizeInPlace`. The scalar sum differs from the AVX sum by
    // ~1e-7 at this dimension, so on the vectors that land within that of an
    // edge the two sums disagree on the *branch*; if they never did, this whole
    // exercise would be moot.
    var prng = std.Random.DefaultPrng.init(0xb0da);
    const rnd = prng.random();
    const d = 1536;
    var v: [d]f32 = undefined;

    var disagreements: usize = 0;
    var avx_divides: usize = 0;
    const trials = 400;
    for (0..trials) |_| {
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = normalizeInPlaceOn(.avx, &v);
        const t: f32 = (rnd.float(f32) * 4.0e-6) - 2.0e-6;
        for (&v) |*x| x.* *= (1.0 + t);

        const avx_says = !isLengthZeroOrNormalized(squaredLengthAvx(&v));
        const scalar_says = !isLengthZeroOrNormalized(squaredLengthScalar(&v));
        if (avx_says != scalar_says) disagreements += 1;
        if (avx_says) avx_divides += 1;

        // The engine's decision on this host is the decision of the path
        // Qdrant would take here.
        var w = v;
        const engine_says = normalizeInPlace(&w);
        const host_path = pathFor(d);
        const host_says = !isLengthZeroOrNormalized(squaredLengthOn(host_path, &v));
        try std.testing.expectEqual(host_says, engine_says);
        if (host_path == .avx) try std.testing.expectEqual(avx_says, engine_says);
    }
    // MEASURED with this seed: the two sums disagree on 37 of the 400 (the
    // AVX sum divides 302 of them), and neither branch is vacuous.
    try std.testing.expect(disagreements > 0);
    try std.testing.expect(avx_divides > 0);
    try std.testing.expect(avx_divides < trials);
}

// -------------------------------------------------------------------------
// §8.4: how far apart Qdrant's own two *normalisations* are
// -------------------------------------------------------------------------

test "§8.4: Qdrant's scalar and AVX preprocess differ, and by how much" {
    // The headline tier's dimension, because that is where T1 fails:
    // `docs/comparison-sift1m.md` §4c records max |Δ| 1.967e-6 against a calibrated
    // ε of 9.537e-7, and attributes it to fp32 accumulation depth in the
    // distance kernel. This measures a second contributor: before this module
    // followed Qdrant's path selection, the engine used the scalar sum while
    // Qdrant on the comparison host used the AVX one, so the two engines did
    // not store the same normalised vector to begin with.
    //
    // MEASURED with this seed, 64 trials at d=1536, Qdrant 1.19.0 semantics
    // (both paths divide by √length):
    //   max |Δcomponent| ≈ 6e-8 (one ulp of the divisor, when the two sums
    //   round √length differently, or the whole divisor when the branches
    //   disagree), max |Δscore| ≈ 1.5e-6, i.e. the size of the §4c residual.
    // The earlier version of this test printed the numbers; the Zig 0.16
    // build runner treats test stderr as a failure, so they live here.
    const dim = 1536;
    const trials = 64;
    var prng = std.Random.DefaultPrng.init(0x8e4);
    const rnd = prng.random();

    const a = try std.testing.allocator.alloc(f32, dim);
    defer std.testing.allocator.free(a);
    const scalar_path = try std.testing.allocator.alloc(f32, dim);
    defer std.testing.allocator.free(scalar_path);
    const avx_path = try std.testing.allocator.alloc(f32, dim);
    defer std.testing.allocator.free(avx_path);

    var max_component: f32 = 0;
    var max_score: f32 = 0;
    for (0..trials) |_| {
        for (a) |*x| x.* = rnd.floatNorm(f32);

        @memcpy(scalar_path, a);
        _ = normalizeInPlaceOn(.scalar, scalar_path);
        @memcpy(avx_path, a);
        _ = normalizeInPlaceOn(.avx, avx_path);

        for (scalar_path, avx_path) |x, y| max_component = @max(max_component, @abs(x - y));

        // What a client actually sees: a score computed over the two stored
        // forms of the same vector.
        var dot_scalar: f32 = 0;
        var dot_avx: f32 = 0;
        for (scalar_path, avx_path) |x, y| {
            dot_scalar += x * x;
            dot_avx += y * y;
        }
        max_score = @max(max_score, @abs(dot_scalar - dot_avx));
    }

    // Both are real and small. The point of pinning them is that they are not
    // zero: a T1 comparison at d=1536 cosine against an engine on the other
    // path is comparing scores over vectors that were never bit-identical in
    // storage, so some of the divergence §4c measured belongs here rather
    // than to the distance kernel.
    try std.testing.expect(max_component > 0);
    try std.testing.expect(max_component < 1e-6);
    try std.testing.expect(max_score < 1e-5);
}
