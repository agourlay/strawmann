//! §6.7, scalar quantization (SQ8).
//!
//! | mode | encoding | distance | notes |
//! |---|---|---|---|
//! | `scalar` | int8, global quantile bounds (bfb uses 0.99) | VNNI `vpdpbusd` where available, else `vpmaddubsw`+`vpmaddwd` | symmetric (query also quantized) and asymmetric variants, both measured |
//!
//! §5.2 places SQ8 at the interesting point of the memory curve: 768 B/vector
//! at d=768 is **12 cache lines, which exactly saturates the fill buffers**.
//! Below that (PQ, binary) a single dependent fetch stops filling the memory
//! pipeline and cost collapses to raw latency; above it (fp32, fp16) the fetch
//! is bandwidth bound. SQ8 sits on the boundary, which makes it the encoding
//! where the §5.2 model is most sensitive to being right.
//!
//! ## Quantile bounds, not min/max
//!
//! §6.7 says "global quantile bounds (bfb uses 0.99)". Using the true min and
//! max instead is the obvious simplification and it is wrong on real
//! embeddings: a single outlier component stretches the range so that the
//! other 99.99% of values compress into a fraction of the 256 codes. Clipping
//! at the 0.5%/99.5% quantiles costs a little accuracy on the outliers and buys
//! a great deal on everything else.
//!
//! ## How this SQ8 differs from Qdrant 1.19's
//!
//! A recall-matched SQ8 comparison against Qdrant is **not** like-for-like;
//! two choices differ, both deliberate (§6.7 asks for quantile bounds and
//! for both query-side variants to be measured), neither numerically equal to
//! Qdrant's:
//!
//! | | strawmann | Qdrant 1.19 (`lib/quantization/src/encoded_vectors_u8.rs`) |
//! |---|---|---|
//! | levels | 256: `alpha = (hi − lo) / 255`, codes 0..255 | 127: `alpha = (hi − lo) / 127`, codes 0..127 (an `i8`-safe range) |
//! | query side | `SymmetricQuery`: the query is quantized once with the same bounds and stage 1 is integer (`vpdpbusd`), which is what the store serves; `AsymmetricQuery` keeps the query in fp32 and is the measured alternative | query is quantized with the same bounds, so both sides carry quantization error |
//! | bounds | true order statistics: the `0.5%`/`99.5%` values of the training sample | `quantile.rs::find_quantile_interval` cuts `⌊vectors·(1−q)/2⌋` *values* off each end of a `vectors × dim` sample (5 of ~768k at q=0.99, 1000 vectors, d=768), then min/max of the rest, so the bounds are effectively min/max |
//!
//! So per comparison strawmann's SQ8 carries less rounding than Qdrant's
//! (twice the levels) and its bounds are tighter on heavy-tailed data. Read an
//! SQ8-vs-SQ8 row as "each engine's scalar quantization at the same recall",
//! not as the same algorithm on two engines. The query side used to differ
//! too (fp32 asymmetric); it was switched (findings 29), and
//! the SQ8 recall curve did not move.
//!
//! ## The reconstruction identity
//!
//! Codes decode as `x ≈ lo + alpha·c`. Substituting into a dot product:
//!
//! ```
//! Σ qᵢ·xᵢ ≈ Σ qᵢ·(lo + alpha·cᵢ) = lo·Σqᵢ + alpha·Σ qᵢ·cᵢ
//! ```
//!
//! so the asymmetric score needs one precomputed `Σqᵢ` per query plus the
//! `f32 × u8` dot. For the symmetric variant both sides are coded and the
//! expansion has four terms, which is why `RowStats` precomputes the
//! per-vector code sums at ingest, recomputing them per comparison would cost
//! more than the quantization saves.

const std = @import("std");
const dist = @import("../dist/dist.zig");

pub const Metric = dist.Metric;

/// Global quantization bounds for one collection.
pub const Params = struct {
    /// Value mapped to code 0.
    lo: f32,
    /// Quantization step: `x ≈ lo + alpha * code`.
    alpha: f32,
    /// Dimension the bounds were trained for, so a mismatch is catchable.
    dim: usize,

    pub fn dequantize(self: Params, code: u8) f32 {
        return self.lo + self.alpha * @as(f32, @floatFromInt(code));
    }

    /// The reconstruction rule, as the kernels take it. One value rather than
    /// two arguments that can be swapped.
    pub fn affine(self: Params) dist.dot_i8.Affine {
        return .{ .lo = self.lo, .alpha = self.alpha };
    }

    pub fn quantizeOne(self: Params, x: f32) u8 {
        const t = (x - self.lo) / self.alpha;
        return @intFromFloat(std.math.clamp(@round(t), 0.0, 255.0));
    }

    /// Worst-case absolute error per component: half a quantization step.
    pub fn maxComponentError(self: Params) f32 {
        return self.alpha * 0.5;
    }
};

/// bfb's quantile, per §6.7.
pub const default_quantile: f32 = 0.99;

/// Train global bounds from a sample of vectors.
///
/// `quantile` is the *central* fraction retained: 0.99 clips the extreme 0.5%
/// at each end. `scratch` must hold at least `sample.len` floats and is used
/// for the selection; the caller owns it so training allocates nothing.
pub fn train(sample: []const f32, quantile: f32, dim: usize, scratch: []f32) Params {
    std.debug.assert(scratch.len >= sample.len);
    @memcpy(scratch[0..sample.len], sample);
    const s = scratch[0..sample.len];
    std.mem.sort(f32, s, {}, std.sort.asc(f32));

    // One `clip` count applied at both ends, so `quantile` means what the doc
    // says: deriving `hi_idx` separately as `⌈(1 − tail)·n⌉` clipped one fewer
    // at the top than `⌊tail·n⌋` did at the bottom. Rounded, in f64, because
    // the f32 `0.99` sits slightly above 0.99 and a floor of `tail·n` computed
    // in f32 lands just under 1 at n = 200, clipping nothing.
    const tail: f64 = (1.0 - @as(f64, quantile)) / 2.0;
    const clip: usize = @intFromFloat(@max(0.0, @round(tail * @as(f64, @floatFromInt(s.len)))));
    var lo_idx: usize = clip;
    var hi_idx: usize = s.len - 1 -| clip;
    if (lo_idx >= hi_idx) {
        lo_idx = 0;
        hi_idx = s.len - 1;
    }

    const lo = s[lo_idx];
    const hi = s[hi_idx];
    // A degenerate range (every sampled value identical) would give alpha = 0
    // and a division by zero in `quantizeOne`. Mapping everything to code 0 is
    // the correct behaviour there and costs nothing.
    const alpha = if (hi > lo) (hi - lo) / 255.0 else 1.0;
    return .{ .lo = lo, .alpha = alpha, .dim = dim };
}

/// Encode one vector into `dst`.
pub fn encode(params: Params, dst: []u8, v: []const f32) void {
    std.debug.assert(dst.len == v.len);
    for (dst, v) |*d, x| d.* = params.quantizeOne(x);
}

/// Decode back to f32, for rescoring and for the fidelity measurements of
/// §8.5's T4 tier.
pub fn decode(params: Params, dst: []f32, codes: []const u8) void {
    std.debug.assert(dst.len == codes.len);
    for (dst, codes) |*d, c| d.* = params.dequantize(c);
}

/// Per-query state for the asymmetric path.
///
/// §6.7 wants both variants measured. Asymmetric keeps the query in fp32, so
/// it carries no query-side quantization error at all, often the better
/// accuracy/speed trade at low `ef`, where the rescore stage of §5.3 dominates
/// and shaving code-side error matters more than shaving cycles.
pub const AsymmetricQuery = struct {
    params: Params,
    /// `Σ qᵢ`, so the `lo` term of the reconstruction is a single multiply.
    query_sum: f32,

    pub fn init(params: Params, query: []const f32) AsymmetricQuery {
        var s: f32 = 0;
        for (query) |x| s += x;
        return .{ .params = params, .query_sum = s };
    }

    /// `Σ qᵢ·xᵢ` against the reconstruction of `codes`.
    pub fn dot(self: AsymmetricQuery, query: []const f32, codes: []const u8) f32 {
        const raw = dist.dot_i8.f32u8_native.call(query, codes);
        return self.params.lo * self.query_sum + self.params.alpha * raw;
    }

    /// `−Σ|qᵢ−xᵢ|` against the reconstruction.
    ///
    /// §8.3 gives Manhattan its own internal similarity; reusing the Euclid arm
    /// would return a squared L2 distance under an L1 label.
    pub fn manhattan(self: AsymmetricQuery, query: []const f32, codes: []const u8) f32 {
        return dist.dot_i8.manhattan_f32u8_native.call(query, codes, self.params.affine());
    }

    /// `−Σ(qᵢ−xᵢ)²` against the reconstruction.
    ///
    /// Computed directly by `EuclidF32U8`, which dequantizes each code in
    /// register (`lo + α·c`) and squares the difference, so it does *not* go
    /// through the `|a|² + |b|² − 2ab` expansion §8.3 warns about. It is still
    /// not ε-comparable with Qdrant's fp32 Euclid, because it compares against
    /// a *quantized* vector; §8.5's T4 tier measures that combined error rather
    /// than attributing it to one stage.
    pub fn euclid(self: AsymmetricQuery, query: []const f32, codes: []const u8) f32 {
        return dist.dot_i8.euclid_f32u8_native.call(query, codes, self.params.affine());
    }
};

/// The two integer sums the symmetric kernels need per stored row, kept
/// beside the codes (`core.quantized.Store.Scalar.stats`) and refreshed by
/// the same call that writes the codes. Eight bytes per row.
pub const RowStats = struct {
    /// `Σ codeᵢ`
    sum: u32,
    /// `Σ codeᵢ²`; 255² × 1536 is 10⁸, so a u32 holds any dimension the
    /// engine accepts.
    sq: u32,

    pub fn of(codes: []const u8) RowStats {
        var sum: u32 = 0;
        var sq: u32 = 0;
        for (codes) |c| {
            const w: u32 = c;
            sum += w;
            sq += w * w;
        }
        return .{ .sum = sum, .sq = sq };
    }
};

/// Per-query state for the symmetric path: the query quantised once, so
/// every comparison is integer.
///
/// This is Qdrant's SQ8 shape (`encoded_vectors_u8.rs`: the query is encoded
/// with the same bounds and scored `u8 × u8` with precomputed per-vector
/// offsets), and it is what the store's stage 1 uses. The asymmetric
/// `AsymmetricQuery` above keeps the fp32 query and dequantises every code
/// in register: 5 instructions per 16 components against one `vpdpbusd` per
/// 64 here. Profiled on W6 (SIFT1M, ef 128, rescore on) it was 46% of the
/// server's CPU, and the SQ8 row served 4,270 q/s to Qdrant's 4,804 with a
/// server-side p50 only 15% under the fp32 row's; Qdrant's SQ8 row was 34%
/// under its fp32 one. Both variants stay measured (`bench-micro`); this one
/// is served.
///
/// Reconstruction, with `q = lo + α·a` and `x = lo + α·b` per component and
/// `d` the dimension:
/// ```
///   Σ qx        = d·lo² + lo·α·(Σa + Σb) + α²·Σab
///   Σ (q − x)²  = α²·(Σa² − 2Σab + Σb²)
///   Σ |q − x|   = α·Σ|a − b|
/// ```
/// `Σab` comes from `vpdpbusd` on the stored `u8` and the query biased into
/// `i8` (`encodeQuerySigned`), so `Σab = raw + 128·Σb`; `Σb` and `Σb²` are
/// the row's `RowStats`.
pub const SymmetricQuery = struct {
    params: Params,
    /// Query codes biased by −128, for `vpdpbusd`.
    signed: []const i8,
    /// The same codes unbiased, for the L1 kernel.
    unsigned: []const u8,
    /// `Σ (a − 128)`, what `encodeQuerySigned` returns.
    sum_biased: i32,
    /// `Σ a²`
    sq: u32,

    pub fn init(params: Params, signed: []i8, unsigned: []u8, q: []const f32) SymmetricQuery {
        std.debug.assert(signed.len == q.len and unsigned.len == q.len);
        const sum_biased = encodeQuerySigned(params, signed, q);
        var sq: u32 = 0;
        for (unsigned, signed) |*u, sgn| {
            const c: u8 = @intCast(@as(i16, sgn) + 128);
            u.* = c;
            sq += @as(u32, c) * @as(u32, c);
        }
        return .{ .params = params, .signed = signed, .unsigned = unsigned, .sum_biased = sum_biased, .sq = sq };
    }

    inline fn dotCodes(self: SymmetricQuery, codes: []const u8, st: RowStats) i64 {
        const raw: i64 = dist.dot_i8.u8i8_native.call(codes, self.signed);
        return raw + 128 * @as(i64, st.sum);
    }

    /// `Σ qᵢ·xᵢ` against the reconstruction.
    pub fn dot(self: SymmetricQuery, codes: []const u8, st: RowStats) f32 {
        const d: f32 = @floatFromInt(codes.len);
        const lo = self.params.lo;
        const a = self.params.alpha;
        const sum_q: f32 = @floatFromInt(self.sum_biased + 128 * @as(i32, @intCast(codes.len)));
        const sum_x: f32 = @floatFromInt(st.sum);
        const ab: f32 = @floatFromInt(self.dotCodes(codes, st));
        return d * lo * lo + lo * a * (sum_q + sum_x) + a * a * ab;
    }

    /// `−Σ(qᵢ−xᵢ)²` against the reconstruction, the sign `AsymmetricQuery.euclid`
    /// returns so the two are interchangeable in stage 1.
    pub fn euclid(self: SymmetricQuery, codes: []const u8, st: RowStats) f32 {
        const ab = self.dotCodes(codes, st);
        const sq: i64 = @as(i64, self.sq) - 2 * ab + @as(i64, st.sq);
        const a = self.params.alpha;
        return -(a * a) * @as(f32, @floatFromInt(sq));
    }

    /// `−Σ|qᵢ−xᵢ|` against the reconstruction. `ManhattanU8` already
    /// returns the negated sum (the similarity convention every kernel
    /// keeps), so only the scale is applied here.
    pub fn manhattan(self: SymmetricQuery, codes: []const u8) f32 {
        const neg_l1: i32 = dist.dot_i8.manhattan_u8_native.call(self.unsigned, codes);
        return self.params.alpha * @as(f32, @floatFromInt(neg_l1));
    }
};

/// Encode a query into signed codes for the `vpdpbusd` kernel.
///
/// `vpdpbusd` is unsigned × signed, so one operand must be `i8`. The data codes
/// stay `u8` (they are the ones streamed from memory, and an unsigned range
/// costs nothing there); the query is biased by −128 into `i8` and the bias is
/// corrected analytically in `SymmetricQuery.dotCodes`. Pretending the instruction
/// is symmetric is the standard way to get a kernel that is correct below 128
/// and silently wrong above it.
///
/// Returns `Σ (code − 128)`, the biased sum the reconstruction needs.
pub fn encodeQuerySigned(params: Params, dst: []i8, q: []const f32) i32 {
    std.debug.assert(dst.len == q.len);
    var sum: i32 = 0;
    for (dst, q) |*d, x| {
        const c = params.quantizeOne(x);
        const biased: i8 = @intCast(@as(i16, c) - 128);
        d.* = biased;
        sum += biased;
    }
    return sum;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

fn trainOn(alloc: std.mem.Allocator, values: []const f32, dim: usize) !Params {
    const scratch = try alloc.alloc(f32, values.len);
    defer alloc.free(scratch);
    return train(values, default_quantile, dim, scratch);
}

test "training clips outliers rather than letting them stretch the range" {
    // The failure mode this guards: one outlier at 1000 with everything else in
    // [-1, 1] would make alpha ~4, so the entire real distribution collapses
    // into a single code.
    var values: [10000]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5c1);
    const rnd = prng.random();
    for (&values) |*v| v.* = rnd.floatNorm(f32);
    values[0] = 1000.0;
    values[1] = -1000.0;

    const p = try trainOn(testing.allocator, &values, 128);

    // The bounds must reflect the bulk of the distribution, not the outliers.
    try testing.expect(p.lo > -10.0);
    try testing.expect(p.lo + p.alpha * 255.0 < 10.0);
    // And the step must be fine enough to resolve a standard normal.
    try testing.expect(p.maxComponentError() < 0.05);
}

test "training clips the same number of values at each end" {
    // Ascending ramp, so the index of the chosen bound is readable off the
    // value. `hi` must sit as far from the top as `lo` sits from the bottom;
    // the old `⌈(1 − tail)·n⌉` clipped one fewer at the top.
    inline for (.{ .{ 200, 1 }, .{ 10000, 50 } }) |case| {
        const n = case[0];
        const expect_clip = case[1];
        const values = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(values);
        const scratch = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(scratch);
        for (values, 0..) |*v, i| v.* = @floatFromInt(i);
        const p = train(values, 0.99, 8, scratch);
        try testing.expectEqual(@as(f32, expect_clip), p.lo);
        try testing.expectApproxEqAbs(@as(f32, n - 1 - expect_clip), p.lo + p.alpha * 255.0, 1e-3);
    }
}

test "min/max bounds would be far worse, which is why quantiles are used" {
    var values: [10000]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5c2);
    const rnd = prng.random();
    for (&values) |*v| v.* = rnd.floatNorm(f32);
    values[0] = 500.0;

    const scratch = try testing.allocator.alloc(f32, values.len);
    defer testing.allocator.free(scratch);

    const quantile_params = train(&values, 0.99, 8, scratch);
    // quantile = 1.0 degenerates to plain min/max.
    const minmax_params = train(&values, 1.0, 8, scratch);

    // Quantifies the claim rather than asserting it: min/max bounds give a
    // step over an order of magnitude coarser.
    try testing.expect(minmax_params.alpha > quantile_params.alpha * 10.0);
}

test "encode/decode round-trips within half a step" {
    var values: [4096]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5c3);
    const rnd = prng.random();
    for (&values) |*v| v.* = rnd.floatNorm(f32);

    const p = try trainOn(testing.allocator, &values, 64);

    const dim = 64;
    var codes: [dim]u8 = undefined;
    var back: [dim]f32 = undefined;
    // Clamp the source to the trained range so the test measures quantization
    // error rather than clipping error, which is a separate (intended) effect.
    var src: [dim]f32 = undefined;
    const hi = p.lo + p.alpha * 255.0;
    for (&src, 0..) |*x, i| x.* = std.math.clamp(values[i], p.lo, hi);

    encode(p, &codes, &src);
    decode(p, &back, &codes);
    for (src, back) |a, b| {
        try testing.expect(@abs(a - b) <= p.maxComponentError() + 1e-6);
    }
}

test "quantizeOne clamps rather than wrapping" {
    const p = Params{ .lo = -1.0, .alpha = 2.0 / 255.0, .dim = 4 };
    try testing.expectEqual(@as(u8, 0), p.quantizeOne(-100.0));
    try testing.expectEqual(@as(u8, 255), p.quantizeOne(100.0));
    try testing.expectEqual(@as(u8, 0), p.quantizeOne(-1.0));
    try testing.expectEqual(@as(u8, 255), p.quantizeOne(1.0));
}

test "degenerate training data does not divide by zero" {
    var values = [_]f32{3.5} ** 100;
    const p = try trainOn(testing.allocator, &values, 8);
    try testing.expect(p.alpha > 0);
    // Everything maps to one code and decodes back to the constant.
    try testing.expectEqual(@as(u8, 0), p.quantizeOne(3.5));
    try testing.expectEqual(@as(f32, 3.5), p.dequantize(0));
}

test "asymmetric dot approximates the fp32 dot" {
    const dim = 768;
    var prng = std.Random.DefaultPrng.init(0x5c4);
    const rnd = prng.random();

    var data: [dim]f32 = undefined;
    var query: [dim]f32 = undefined;
    for (&data, &query) |*a, *b| {
        a.* = rnd.floatNorm(f32);
        b.* = rnd.floatNorm(f32);
    }

    const p = try trainOn(testing.allocator, &data, dim);
    var codes: [dim]u8 = undefined;
    encode(p, &codes, &data);

    // Ground truth is the dot against the *reconstruction*, which isolates the
    // kernel's correctness from the quantizer's error.
    var recon: [dim]f32 = undefined;
    decode(p, &recon, &codes);
    const truth = dist.reference.dotWide(&query, &recon);

    const aq = AsymmetricQuery.init(p, &query);
    const got = aq.dot(&query, &codes);
    try testing.expect(@abs(@as(f64, got) - truth) / @max(1.0, @abs(truth)) < 1e-4);
}

test "symmetric dot approximates the fp32 dot, and vpdpbusd bias is corrected" {
    const dim = 768;
    var prng = std.Random.DefaultPrng.init(0x5c5);
    const rnd = prng.random();

    var data: [dim]f32 = undefined;
    var query: [dim]f32 = undefined;
    for (&data, &query) |*a, *b| {
        a.* = rnd.floatNorm(f32);
        b.* = rnd.floatNorm(f32);
    }

    const p = try trainOn(testing.allocator, &data, dim);
    var codes: [dim]u8 = undefined;
    encode(p, &codes, &data);

    // The shipped path: `SymmetricQuery` over the row's `RowStats`. The
    // parallel `SymmetricParams` this used to exercise was a second copy of
    // the same bias algebra that nothing served.
    var signed: [dim]i8 = undefined;
    var unsigned: [dim]u8 = undefined;
    const sq = SymmetricQuery.init(p, &signed, &unsigned, &query);

    // Truth: dot between both reconstructions.
    var recon_a: [dim]f32 = undefined;
    decode(p, &recon_a, &codes);
    var recon_b: [dim]f32 = undefined;
    decode(p, &recon_b, &unsigned);
    const truth = dist.reference.dotWide(&recon_a, &recon_b);

    const got = sq.dot(&codes, RowStats.of(&codes));
    // The identity is exact in real arithmetic; the tolerance covers f32
    // accumulation over 768 terms with values up to 255².
    try testing.expect(@abs(@as(f64, got) - truth) / @max(1.0, @abs(truth)) < 1e-3);
}

test "symmetric encoding is correct above code 127, where a naive kernel breaks" {
    // The specific hazard `encodeQuerySigned` exists to avoid: treating u8
    // codes as i8 silently reinterprets everything above 127 as negative.
    const dim = 64;
    const p = Params{ .lo = 0.0, .alpha = 1.0, .dim = dim };

    var q: [dim]f32 = undefined;
    @memset(&q, 250.0); // code 250, well above 127
    var signed: [dim]i8 = undefined;
    var unsigned: [dim]u8 = undefined;
    const sq = SymmetricQuery.init(p, &signed, &unsigned, &q);

    // 250 - 128 = 122, which fits i8.
    for (signed) |c| try testing.expectEqual(@as(i8, 122), c);
    try testing.expectEqual(@as(i32, 122 * dim), sq.sum_biased);

    var data: [dim]f32 = undefined;
    @memset(&data, 200.0);
    var codes: [dim]u8 = undefined;
    encode(p, &codes, &data);
    for (codes) |c| try testing.expectEqual(@as(u8, 200), c);

    const got = sq.dot(&codes, RowStats.of(&codes));
    // Reconstructions are exactly 200 and 250, so the dot is 64 * 200 * 250.
    try testing.expectApproxEqRel(@as(f32, 64.0 * 200.0 * 250.0), got, 1e-5);
}

test "quantization preserves ranking well enough to be useful" {
    // The property that actually matters for search: the top-k under the
    // quantized score should mostly match the top-k under fp32. §8.5's T4 tier
    // measures this properly; here it is a sanity floor.
    const dim = 128;
    const n = 500;
    var prng = std.Random.DefaultPrng.init(0x5c6);
    const rnd = prng.random();

    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = rnd.floatNorm(f32);

    const p = try trainOn(testing.allocator, data, dim);
    const codes = try testing.allocator.alloc(u8, n * dim);
    defer testing.allocator.free(codes);
    for (0..n) |i| encode(p, codes[i * dim ..][0..dim], data[i * dim ..][0..dim]);

    var query: [dim]f32 = undefined;
    for (&query) |*x| x.* = rnd.floatNorm(f32);
    const aq = AsymmetricQuery.init(p, &query);

    const Pair = struct { id: u32, s: f32 };
    const exact = try testing.allocator.alloc(Pair, n);
    defer testing.allocator.free(exact);
    const approx = try testing.allocator.alloc(Pair, n);
    defer testing.allocator.free(approx);

    for (0..n) |i| {
        exact[i] = .{ .id = @intCast(i), .s = dist.native.dot(&query, data[i * dim ..][0..dim]) };
        approx[i] = .{ .id = @intCast(i), .s = aq.dot(&query, codes[i * dim ..][0..dim]) };
    }
    const lt = struct {
        fn f(_: void, a: Pair, b: Pair) bool {
            return a.s > b.s;
        }
    }.f;
    std.mem.sort(Pair, exact, {}, lt);
    std.mem.sort(Pair, approx, {}, lt);

    const k = 10;
    var overlap: usize = 0;
    for (approx[0..k]) |a| {
        for (exact[0..k]) |e| {
            if (a.id == e.id) {
                overlap += 1;
                break;
            }
        }
    }
    // SQ8 on random normal data should recover nearly all of the true top-10.
    try testing.expect(overlap >= 8);
}

test "the quantized Manhattan score is L1, not squared L2" {
    // `quantized.Query.score` shared the Euclid arm for `.manhattan`, so a
    // client that asked for Manhattan got a *squared Euclidean* distance back -
    // and with `rescore = false` that value goes out on the wire as the score.
    // §8.3 fixes what each metric returns; a quantized path is not licensed to
    // return a different quantity.
    //
    // The two agree in sign and often in ranking, which is why this survived:
    // the failure is in the returned magnitude, not usually in the order.
    var scratch: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5151);
    const rnd = prng.random();

    const dim = 64;
    var sample: [dim * 4]f32 = undefined;
    for (&sample) |*x| x.* = rnd.floatNorm(f32);
    const params = train(&sample, default_quantile, dim, &scratch);

    var v: [dim]f32 = undefined;
    var q: [dim]f32 = undefined;
    for (&v) |*x| x.* = rnd.floatNorm(f32);
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    var codes: [dim]u8 = undefined;
    encode(params, &codes, &v);
    const aq = AsymmetricQuery.init(params, &q);

    // Against the reconstruction, computed independently.
    var want_l1: f32 = 0;
    var want_l2: f32 = 0;
    for (q, codes) |qi, c| {
        const d = qi - params.dequantize(c);
        want_l1 += @abs(d);
        want_l2 += d * d;
    }

    try testing.expectApproxEqAbs(-want_l1, aq.manhattan(&q, &codes), 1e-3);
    try testing.expectApproxEqAbs(-want_l2, aq.euclid(&q, &codes), 1e-3);

    // And the two are genuinely different quantities: at d=64 with unit-scale
    // components the L1 sum is several times the L2 sum, so returning one for
    // the other is not a rounding-level substitution.
    try testing.expect(@abs(want_l1 - want_l2) > 1.0);
}

test "symmetric query agrees with the asymmetric one to the quantisation step, all three metrics" {
    // Both are approximations of the fp32 value; the difference between them
    // is the query's own quantisation, at most half a step per component.
    // The point pinned: same sign convention, same scale, so stage 1 can use
    // either and rank the same way.
    const dim = 96;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var values: [dim * 64]f32 = undefined;
    for (&values) |*v| v.* = rnd.floatNorm(f32);
    const p = try trainOn(testing.allocator, &values, dim);

    var codes: [dim]u8 = undefined;
    var signed: [dim]i8 = undefined;
    var unsigned: [dim]u8 = undefined;
    var worst_l2: f32 = 0;
    for (0..32) |i| {
        const q = values[i * dim ..][0..dim];
        const x = values[(i + 32) * dim ..][0..dim];
        encode(p, &codes, x);
        const st = RowStats.of(&codes);
        const asym = AsymmetricQuery.init(p, q);
        const sym = SymmetricQuery.init(p, &signed, &unsigned, q);
        // Exact values over the reconstruction of `x` and the *quantised* q.
        var exact_dot: f64 = 0;
        var exact_l2: f64 = 0;
        var exact_l1: f64 = 0;
        for (0..dim) |k| {
            const qk: f64 = p.lo + p.alpha * @as(f32, @floatFromInt(unsigned[k]));
            const xk: f64 = p.lo + p.alpha * @as(f32, @floatFromInt(codes[k]));
            exact_dot += qk * xk;
            exact_l2 += (qk - xk) * (qk - xk);
            exact_l1 += @abs(qk - xk);
        }
        // Symmetric is exact over the two reconstructions, to fp32 rounding.
        // The dot is a small number left by cancelling terms of order
        // `d·lo²` (~600 here), so it is judged absolutely at that scale.
        try testing.expectApproxEqAbs(@as(f32, @floatCast(exact_dot)), sym.dot(&codes, st), 1e-2);
        try testing.expectApproxEqRel(@as(f32, @floatCast(-exact_l2)), sym.euclid(&codes, st), 1e-4);
        try testing.expectApproxEqRel(@as(f32, @floatCast(-exact_l1)), sym.manhattan(&codes), 1e-4);
        // And close to the asymmetric value: the two differ by the query's
        // own quantisation, half a step per component except where the
        // 0.5% quantile clipped it, so a few percent of the distance, never
        // more. The dot is judged on the scale of its terms.
        const l2 = @abs(asym.euclid(q, &codes));
        try testing.expect(@abs(sym.euclid(&codes, st) - asym.euclid(q, &codes)) <= 0.05 * l2);
        const l1 = @abs(asym.manhattan(q, &codes));
        try testing.expect(@abs(sym.manhattan(&codes) - asym.manhattan(q, &codes)) <= 0.05 * l1);
        const dot_scale: f32 = @as(f32, @floatFromInt(dim)) * (p.lo * p.lo + 1.0);
        try testing.expect(@abs(sym.dot(&codes, st) - asym.dot(q, &codes)) <= 0.02 * dot_scale);
        worst_l2 = @max(worst_l2, @abs(sym.euclid(&codes, st) - asym.euclid(q, &codes)));
    }
    try testing.expect(worst_l2 > 0); // the two really are different estimators
}

test "RowStats sums codes and their squares" {
    const codes = [_]u8{ 0, 1, 2, 255, 128 };
    const st = RowStats.of(&codes);
    try testing.expectEqual(@as(u32, 386), st.sum);
    try testing.expectEqual(@as(u32, 1 + 4 + 65025 + 16384), st.sq);
}
