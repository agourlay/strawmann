//! §6.7, binary quantization, and the rescore stage.
//!
//! | mode | encoding | distance | notes |
//! |---|---|---|---|
//! | `binary` | 1 bit/dim, sign-based | XOR + `@popCount` over `@Vector(u64)` | fastest; needs oversampling + rescore |
//!
//! ## Why the rescore stage is the interesting part
//!
//! §5.3's per-query model, at `Nd = 2500`:
//!
//! | encoding | modelled latency | modelled QPS/core |
//! |---|---|---|
//! | fp32 | ~900 µs | ~1.1k |
//! | int8 | ~225 µs | ~4.4k |
//! | binary + rescore top-100 fp32 | ~37 µs + ~36 µs | ~13k |
//!
//! and the observation that follows it:
//!
//!   "Note how, in the binary row, **rescoring becomes the dominant term**.
//!    That immediately makes oversampling factor a first-class tuning parameter
//!    and suggests rescoring should itself run against a cheaper-than-fp32
//!    representation."
//!
//! So `Rescorer` is parameterised by the representation it rescores *against*,
//! not hardcoded to fp32. §6.7 asks for "the **combined** curve of (recall,
//! latency) across `(ef, oversampling, rescore-encoding)` rather than tuning
//! stages independently", and that sweep is only expressible if the third axis
//! exists in the code.
//!
//! ## Why binary is latency bound, not bandwidth bound
//!
//! §5.2's table gives binary 96 B/vector at d=768, **2 cache lines**, against
//! the ~12 needed to saturate the fill buffers. Its modelled cost is therefore
//! a range, "~15–90 ns", depending entirely on whether several vectors are in
//! flight:
//!
//!   "Below ~12 cache lines per vector, a single dependent fetch no longer
//!    fills the memory pipeline, and measured cost collapses to raw DRAM
//!    latency unless we explicitly issue neighbour prefetches. **Software
//!    prefetch of the neighbour list's vectors, issued before scoring the
//!    current candidate, is the single highest-leverage optimisation in the
//!    search path**, and it only becomes visible once quantization has shrunk
//!    vectors below the LFB threshold."
//!
//! `scoreBatch` exists for that reason: it takes a *list* of candidates and
//! prefetches all of their codes before scoring any, which is the only way the
//! 6× between the two ends of that range is reachable.

const std = @import("std");
const dist = @import("../dist/dist.zig");
const hamming = dist.hamming;

/// Packed-word count for `dim` bits, the *logical* size.
pub fn wordsFor(dim: usize) usize {
    return hamming.wordsFor(dim);
}

/// u64 lanes in the widest vector register the engine targets (512-bit).
pub const max_u64_lanes = 8;

/// Stored word count: `wordsFor` rounded up to a whole number of vector
/// registers.
///
/// ## Why the padding exists, measured
///
/// The Hamming kernel processes `L` words per vector operation and falls back
/// to a scalar loop for the remainder. That remainder gets *worse* as the
/// register gets wider, which inverts the ISA comparison §7.5 is trying to make:
///
/// | dim | words | AVX2 (L=4) | AVX-512 (L=8) |
/// |---|--:|---|---|
/// | 128 | 2 | 2 scalar | 2 scalar |
/// | 384 | 6 | 4 vector + 2 scalar | **6 scalar** |
/// | 768 | 12 | 12 vector | 8 vector + **4 scalar** |
/// | 1536 | 24 | 16+8 vector | 24 vector |
///
/// So at d=384 the 512-bit build did *no* vector work at all, and at d=768 it
/// did a third of the work scalar while the 256-bit build did none. That is why
/// `vpopcntq` measured at 0.99× against `vpshufb` at d=768, most of the loop
/// was not using either.
///
/// Padding to a fixed 8 words makes every dimension fully vectorised on every
/// arm, and keeps the stored layout identical across arms so §7.5 compares
/// instructions rather than tail handling.
///
/// The cost is real and worth stating: at d=768 a row grows from 96 B to 128 B,
/// a 33% increase over §5.4's quoted binary working set. That is acceptable
/// *because binary is latency bound, not bandwidth bound*, the measured cold
/// throughput is 3.22 GB/s against a 25.3 GB/s per-core ceiling, so the extra
/// bytes ride along in cache lines already being fetched.
///
/// Measured effect, `hamming` cycles/vector, AVX-512 ÷ AVX2, DRAM-cold:
///
/// | d | before | after |
/// |---|--:|--:|
/// | 128 | 1.48x | **1.97x** |
/// | 384 | 1.05x | **2.42x** |
/// | 768 | 1.57x | **2.72x** |
/// | 1536 | 2.20x | 2.44x |
///
/// The instruction count is what exposed the problem: at d=768 the 512-bit
/// build was emitting 97 instructions per vector against the 256-bit build's
/// 101, no reduction at all, where every other kernel showed 1.7x. After
/// padding it is 62 against 102.
pub fn paddedWordsFor(dim: usize) usize {
    return std.mem.alignForward(usize, wordsFor(dim), max_u64_lanes);
}

/// Encode one vector's sign bits.
///
/// §6.7: "1 bit/dim, sign-based". The tie-break, a bit is set only when the
/// component is **strictly positive**, lives in `dist/hamming.zig` and is
/// tested there, because getting it wrong diverges from Qdrant only on exact
/// zeros, which random test data essentially never produces and real sparse-ish
/// embeddings produce constantly.
pub fn encode(dst: []u64, v: []const f32) void {
    hamming.packSigns(dst, v);
}

/// Ranking-only similarity for binary codes: `−hamming`.
///
/// Negated so "higher is better" holds uniformly (§8.3's convention), which is
/// what lets the same `TopK` serve every encoding. It orders candidates
/// exactly as `signDot` does, but it is **not** the number Qdrant puts on the
/// wire, so a path whose stage-1 score can reach a client (`rescore = false`)
/// must score with `signDot` instead.
pub fn similarity(a: []const u64, b: []const u64) f32 {
    return -@as(f32, @floatFromInt(hamming.native.call(a, b)));
}

/// The `±1` dot product the Hamming distance implies, `dim − 2·hamming`, which
/// is Qdrant's binary score for every metric.
///
/// Qdrant 1.19 `encoded_vectors_binary.rs` (`calculate_metric`) returns
/// `zeros − xor = dim − 2·xor` for Dot/Cosine, and for L1/L2 with
/// `invert = true`, which `quantized_vectors.rs` sets for Euclid and Manhattan,
/// so the same value is what its `postprocess` then sees. Handing this to
/// `Metric.postprocess` therefore reproduces Qdrant's wire score for all four
/// metrics: identity for dot/cosine, `√|dim − 2h|` for euclid, `|dim − 2h|`
/// for manhattan. It is monotone in `−hamming`, so ranking on it is ranking on
/// `similarity`.
pub fn signDot(dim: usize, a: []const u64, b: []const u64) f32 {
    return hamming.signDotFromHamming(dim, hamming.native.call(a, b));
}

/// A binary code store.
pub const Codes = struct {
    words: usize,
    data: []u64,

    pub fn init(alloc: std.mem.Allocator, dim: usize, capacity: usize) !Codes {
        const w = paddedWordsFor(dim);
        const data = try alloc.alloc(u64, w * capacity);
        @memset(data, 0);
        return .{ .words = w, .data = data };
    }

    pub fn deinit(self: *Codes, alloc: std.mem.Allocator) void {
        alloc.free(self.data);
    }

    pub fn row(self: *const Codes, offset: u32) []u64 {
        return self.data[@as(usize, offset) * self.words ..][0..self.words];
    }

    pub fn rowConst(self: *const Codes, offset: u32) []const u64 {
        return self.row(offset);
    }

    pub fn set(self: *Codes, offset: u32, v: []const f32) void {
        encode(self.row(offset), v);
    }

    /// Score a batch of candidates, prefetching every code row first.
    ///
    /// This is §5.2's "single highest-leverage optimisation" made concrete. At
    /// 2 cache lines per vector, scoring candidates one at a time serialises on
    /// DRAM latency (~90 ns each); issuing all the prefetches up front lets the
    /// fill buffers overlap them, which is the difference between the two ends
    /// of §5.2's "~15–90 ns" range for this encoding.
    pub fn scoreBatch(
        self: *const Codes,
        query: []const u64,
        candidates: []const u32,
        out: []f32,
    ) void {
        std.debug.assert(out.len >= candidates.len);

        // Pass 1: touch every row we are about to need.
        for (candidates) |c| {
            const r = self.rowConst(c);
            dist.common.prefetchRow(@ptrCast(r.ptr), r.len * @sizeOf(u64));
        }
        // Pass 2: score. By now the loads are in flight or landed.
        for (candidates, 0..) |c, i| {
            out[i] = similarity(query, self.rowConst(c));
        }
    }
};

// =========================================================================
// Oversampling and rescore
// =========================================================================

/// What the rescore stage scores against.
///
/// §5.3: "rescoring should itself run against a cheaper-than-fp32
/// representation". Making this an axis rather than a constant is what lets
/// §6.7's `(ef, oversampling, rescore-encoding)` sweep exist.
pub const RescoreEncoding = enum {
    /// No rescore. Binary scores are returned directly, fastest and least
    /// accurate, and the baseline the others are measured against.
    none,
    /// Rescore against SQ8 codes. Roughly 4× cheaper than fp32 per §5.2's
    /// bytes/vector, at some accuracy cost.
    sq8,
    /// Rescore against the full fp32 vectors. Most accurate, and the term
    /// §5.3 predicts will dominate the binary row's latency.
    fp32,
};

/// §6.7: "quantized search produces `limit × oversampling` candidates,
/// rescored with the full-precision (or next-tier) representation."
///
/// `oversampling` arrives from the wire as an unvalidated `double`, so this has
/// to be total over every bit pattern a client can send, not just over the
/// sensible range:
///
///   * **NaN.** `NaN <= 1.0` is *false*, so a naive guard passes NaN straight
///     through to `@intFromFloat`, which is illegal behaviour on a non-finite
///     value: an assertion failure in Debug and a garbage limit in ReleaseFast.
///     The comparison has to be written so NaN takes the reject branch.
///   * **Huge and infinite.** `1e30` is finite and greater than 1, and
///     `@intFromFloat` on a value past `maxInt(usize)` is equally illegal.
///     `max_oversampling` caps it; anything above that would exhaust the
///     candidate buffer long before it mattered anyway.
///
/// Clamping rather than erroring matches Qdrant, which treats oversampling as a
/// hint: the result stays correct, only slower or cheaper than asked.
pub fn oversampledLimit(limit: usize, oversampling: f64) usize {
    // Written as "is it in range" rather than "is it out of range" so NaN,
    // which compares false against everything, falls through to `limit`.
    if (!(oversampling > 1.0)) return limit;
    const factor = @min(oversampling, max_oversampling);
    const scaled = @ceil(@as(f64, @floatFromInt(limit)) * factor);
    if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(usize)))) return std.math.maxInt(usize);
    return @intFromFloat(scaled);
}

/// A ceiling on the oversampling factor. §6.7 sweeps this axis to 64×; beyond a
/// few thousand the candidate set is the whole collection and the knob is
/// meaningless.
pub const max_oversampling: f64 = 65536.0;

/// The rescore stage.
///
/// Takes the candidate ids a cheap encoding produced, rescores them with a more
/// accurate one, and returns the top `limit`.
pub const Rescorer = struct {
    encoding: RescoreEncoding,
    /// Called to rescore one candidate. Supplied by the collection, which owns
    /// the fp32 arena and any SQ8 codes.
    score: *const fn (ctx: *const anyopaque, query: []const f32, node: u32) f32,
    ctx: *const anyopaque,

    /// Rescore `candidates` in place into `out`, returning the top `limit`.
    ///
    /// `out` must hold at least `candidates.len` entries. The result is sorted
    /// by the §8.7 total order, so ties break identically to every other path.
    pub fn run(
        self: Rescorer,
        query: []const f32,
        candidates: []const u32,
        limit: usize,
        out: []Scored,
    ) []Scored {
        std.debug.assert(out.len >= candidates.len);
        for (candidates, 0..) |c, i| {
            out[i] = .{ .id = c, .score = self.score(self.ctx, query, c) };
        }
        const slice = out[0..candidates.len];
        std.mem.sort(Scored, slice, {}, struct {
            fn lt(_: void, a: Scored, b: Scored) bool {
                if (a.score != b.score) return a.score > b.score;
                return a.id < b.id;
            }
        }.lt);
        return slice[0..@min(limit, slice.len)];
    }
};

pub const Scored = struct {
    id: u32,
    score: f32,
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "encode then similarity reproduces the sign dot product" {
    const dim = 768;
    var prng = std.Random.DefaultPrng.init(0xb1a);
    const rnd = prng.random();
    var a: [dim]f32 = undefined;
    var b: [dim]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }

    var ca: [12]u64 = undefined;
    var cb: [12]u64 = undefined;
    encode(&ca, &a);
    encode(&cb, &b);

    var expect: f32 = 0;
    for (a, b) |x, y| {
        const sx: f32 = if (x > 0) 1 else -1;
        const sy: f32 = if (y > 0) 1 else -1;
        expect += sx * sy;
    }
    try testing.expectEqual(expect, signDot(dim, &ca, &cb));
}

test "similarity is negated so higher is better" {
    var a = [_]u64{0} ** 4;
    var b = [_]u64{0} ** 4;
    try testing.expectEqual(@as(f32, 0.0), similarity(&a, &b));
    b[0] = 0xff;
    try testing.expectEqual(@as(f32, -8.0), similarity(&a, &b));
    try testing.expect(similarity(&a, &a) > similarity(&a, &b));
}

test "signDot fed to Metric.postprocess reproduces Qdrant's binary wire score" {
    // d = 768 with 100 differing bits: Qdrant returns dim − 2·xor = 568 for
    // dot/cosine, √568 for euclid and 568 for manhattan (its L1/L2 arm is
    // built inverted, so the value its postprocess sees is the same 568).
    // The ranking-only `similarity` (−100) would put −100, 10 and 100 on the
    // wire instead.
    const dim = 768;
    var a = [_]u64{0} ** paddedWordsFor(dim);
    var b = [_]u64{0} ** paddedWordsFor(dim);
    // 100 differing bits: one full word (64) plus 36 in the next.
    b[0] = std.math.maxInt(u64);
    b[1] = (@as(u64, 1) << 36) - 1;
    try testing.expectEqual(@as(u32, 100), hamming.native.call(&a, &b));

    const s = signDot(dim, &a, &b);
    try testing.expectEqual(@as(f32, 568.0), s);
    try testing.expectEqual(@as(f32, 568.0), dist.Metric.dot.postprocess(s));
    try testing.expectEqual(@as(f32, 568.0), dist.Metric.cosine.postprocess(s));
    try testing.expectEqual(@sqrt(@as(f32, 568.0)), dist.Metric.euclid.postprocess(s));
    try testing.expectEqual(@as(f32, 568.0), dist.Metric.manhattan.postprocess(s));

    // And it ranks exactly as `similarity` does: fewer differing bits, higher
    // score, on both.
    var c = [_]u64{0} ** paddedWordsFor(dim);
    c[0] = 0xff;
    try testing.expect(signDot(dim, &a, &c) > signDot(dim, &a, &b));
    try testing.expect(similarity(&a, &c) > similarity(&a, &b));
}

test "code store rows are padded to a whole vector register" {
    var c = try Codes.init(testing.allocator, 768, 16);
    defer c.deinit(testing.allocator);

    // §5.2's table quotes 96 B/vector at d=768, that is the *logical* size.
    try testing.expectEqual(@as(usize, 12), wordsFor(768));
    try testing.expectEqual(@as(usize, 96), wordsFor(768) * @sizeOf(u64));

    // Stored, rows are padded to 8 words so the kernel never falls back to a
    // scalar tail. See `paddedWordsFor` for the measurement that motivated it.
    try testing.expectEqual(@as(usize, 16), c.words);
    try testing.expectEqual(@as(usize, 128), c.words * @sizeOf(u64));

    const r0 = c.row(0);
    const r1 = c.row(1);
    try testing.expectEqual(@intFromPtr(r0.ptr) + 128, @intFromPtr(r1.ptr));
}

test "padding is zeroed, so it contributes nothing to any distance" {
    // The property the whole scheme rests on: padding words XOR to zero and
    // popcount to zero, so a padded comparison equals an unpadded one.
    var prng = std.Random.DefaultPrng.init(0x9ad);
    const rnd = prng.random();
    const dim = 768; // 12 logical words, 16 stored

    var c = try Codes.init(testing.allocator, dim, 2);
    defer c.deinit(testing.allocator);

    var a: [dim]f32 = undefined;
    var b: [dim]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }
    c.set(0, &a);
    c.set(1, &b);

    // Padding words are zero in both rows.
    for (c.rowConst(0)[wordsFor(dim)..]) |wd| try testing.expectEqual(@as(u64, 0), wd);
    for (c.rowConst(1)[wordsFor(dim)..]) |wd| try testing.expectEqual(@as(u64, 0), wd);

    // And the padded distance equals the logical one.
    const padded = hamming.native.call(c.rowConst(0), c.rowConst(1));
    const logical = hamming.native.call(
        c.rowConst(0)[0..wordsFor(dim)],
        c.rowConst(1)[0..wordsFor(dim)],
    );
    try testing.expectEqual(logical, padded);

    // And it still equals the true sign dot product.
    var expect: f32 = 0;
    for (a, b) |x, y| {
        const sx: f32 = if (x > 0) 1 else -1;
        const sy: f32 = if (y > 0) 1 else -1;
        expect += sx * sy;
    }
    try testing.expectEqual(expect, signDot(dim, c.rowConst(0), c.rowConst(1)));
}

test "every matrix dimension is a whole number of vector registers when stored" {
    // The point of the padding: no dimension falls back to a scalar tail on
    // any arm the §7.5 matrix builds.
    for ([_]usize{ 128, 384, 768, 960, 1536 }) |d| {
        try testing.expectEqual(@as(usize, 0), paddedWordsFor(d) % max_u64_lanes);
        try testing.expect(paddedWordsFor(d) >= wordsFor(d));
    }
}

test "scoreBatch matches scoring one at a time" {
    // The prefetch pass must not change the answer, only when the loads issue.
    const dim = 256;
    const n = 64;
    var c = try Codes.init(testing.allocator, dim, n);
    defer c.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(0xb1b);
    const rnd = prng.random();
    var v: [dim]f32 = undefined;
    for (0..n) |i| {
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        c.set(@intCast(i), &v);
    }
    for (&v) |*x| x.* = rnd.floatNorm(f32);
    var q: [paddedWordsFor(dim)]u64 = undefined;
    encode(&q, &v);

    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    var batched: [n]f32 = undefined;
    c.scoreBatch(&q, &ids, &batched);

    for (0..n) |i| {
        try testing.expectEqual(similarity(&q, c.rowConst(@intCast(i))), batched[i]);
    }
}

test "§6.7 oversampling widens the candidate set" {
    try testing.expectEqual(@as(usize, 10), oversampledLimit(10, 1.0));
    try testing.expectEqual(@as(usize, 40), oversampledLimit(10, 4.0));
    try testing.expectEqual(@as(usize, 15), oversampledLimit(10, 1.5));
    // Values at or below 1 must not shrink the result below what was asked for.
    try testing.expectEqual(@as(usize, 10), oversampledLimit(10, 0.5));
}

test "§8.6: recall is monotonically non-decreasing in oversampling" {
    // "Quantization dominance: with `rescore = true` and oversampling `n`,
    // recall is monotonically non-decreasing in `n`." This is the metamorphic
    // property, checked end-to-end through binary search plus fp32 rescore.
    const dim = 128;
    const n = 800;
    const k = 10;

    var prng = std.Random.DefaultPrng.init(0xb1c);
    const rnd = prng.random();
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = rnd.floatNorm(f32);

    var codes = try Codes.init(testing.allocator, dim, n);
    defer codes.deinit(testing.allocator);
    for (0..n) |i| codes.set(@intCast(i), data[i * dim ..][0..dim]);

    const Ctx = struct {
        data: []const f32,
        dim: usize,
        fn score(ctx: *const anyopaque, q: []const f32, node: u32) f32 {
            const self: *const @This() = @ptrCast(@alignCast(ctx));
            return dist.native.dot(q, self.data[@as(usize, node) * self.dim ..][0..self.dim]);
        }
    };
    var ctx = Ctx{ .data = data, .dim = dim };
    const rescorer = Rescorer{ .encoding = .fp32, .score = Ctx.score, .ctx = @ptrCast(&ctx) };

    const queries = 40;
    var last_recall: f64 = -1.0;

    for ([_]f64{ 1.0, 2.0, 4.0, 8.0 }) |ov| {
        var hits: usize = 0;
        for (0..queries) |_| {
            var q: [dim]f32 = undefined;
            for (&q) |*x| x.* = rnd.floatNorm(f32);

            // Exact top-k by fp32 dot.
            const exact = try testing.allocator.alloc(Scored, n);
            defer testing.allocator.free(exact);
            for (0..n) |i| {
                exact[i] = .{ .id = @intCast(i), .score = Ctx.score(@ptrCast(&ctx), &q, @intCast(i)) };
            }
            std.mem.sort(Scored, exact, {}, struct {
                fn lt(_: void, a: Scored, b: Scored) bool {
                    return a.score > b.score;
                }
            }.lt);

            // Binary stage: take the oversampled top by Hamming.
            var qc: [paddedWordsFor(dim)]u64 = undefined;
            encode(&qc, &q);
            const bin = try testing.allocator.alloc(Scored, n);
            defer testing.allocator.free(bin);
            for (0..n) |i| {
                bin[i] = .{ .id = @intCast(i), .score = similarity(&qc, codes.rowConst(@intCast(i))) };
            }
            std.mem.sort(Scored, bin, {}, struct {
                fn lt(_: void, a: Scored, b: Scored) bool {
                    if (a.score != b.score) return a.score > b.score;
                    return a.id < b.id;
                }
            }.lt);

            const take = @min(oversampledLimit(k, ov), n);
            const cand_ids = try testing.allocator.alloc(u32, take);
            defer testing.allocator.free(cand_ids);
            for (0..take) |i| cand_ids[i] = bin[i].id;

            const out = try testing.allocator.alloc(Scored, take);
            defer testing.allocator.free(out);
            const final = rescorer.run(&q, cand_ids, k, out);

            for (final) |f| {
                for (exact[0..k]) |e| {
                    if (e.id == f.id) {
                        hits += 1;
                        break;
                    }
                }
            }
        }
        const recall = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(queries * k));
        // Monotone: more candidates can only add true neighbours, never remove
        // one that a smaller set already contained.
        try testing.expect(recall >= last_recall - 1e-9);
        last_recall = recall;
    }

    // And at 8x oversampling binary + fp32 rescore should recover most of the
    // true top-10, the trade §6.7 exists to make.
    try testing.expect(last_recall > 0.5);
}

test "rescorer returns the §8.7 total order" {
    const Ctx = struct {
        fn score(_: *const anyopaque, _: []const f32, node: u32) f32 {
            // Deliberately tie every pair so the id tie-break is what orders
            // the output.
            return if (node % 2 == 0) 1.0 else 0.5;
        }
    };
    var dummy: u8 = 0;
    const r = Rescorer{ .encoding = .fp32, .score = Ctx.score, .ctx = @ptrCast(&dummy) };

    const ids = [_]u32{ 7, 2, 5, 4, 9, 0 };
    var out: [6]Scored = undefined;
    const got = r.run(&.{}, &ids, 6, &out);

    // Score descending, then id ascending: evens (score 1.0) first in id order,
    // then odds.
    try testing.expectEqual(@as(u32, 0), got[0].id);
    try testing.expectEqual(@as(u32, 2), got[1].id);
    try testing.expectEqual(@as(u32, 4), got[2].id);
    try testing.expectEqual(@as(u32, 5), got[3].id);
    try testing.expectEqual(@as(u32, 7), got[4].id);
    try testing.expectEqual(@as(u32, 9), got[5].id);
}

test "rescorer honours the limit" {
    const Ctx = struct {
        fn score(_: *const anyopaque, _: []const f32, node: u32) f32 {
            return @floatFromInt(node);
        }
    };
    var dummy: u8 = 0;
    const r = Rescorer{ .encoding = .fp32, .score = Ctx.score, .ctx = @ptrCast(&dummy) };
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    var out: [5]Scored = undefined;
    const got = r.run(&.{}, &ids, 2, &out);
    try testing.expectEqual(@as(usize, 2), got.len);
    try testing.expectEqual(@as(u32, 5), got[0].id);
    try testing.expectEqual(@as(u32, 4), got[1].id);
}

test "oversampledLimit is total over the wire's f64 domain" {
    // The ordinary cases.
    try testing.expectEqual(@as(usize, 10), oversampledLimit(10, 1.0));
    try testing.expectEqual(@as(usize, 10), oversampledLimit(10, 0.5));
    try testing.expectEqual(@as(usize, 40), oversampledLimit(10, 4.0));
    try testing.expectEqual(@as(usize, 25), oversampledLimit(10, 2.5)); // ceil

    // The ones a client can actually send. Each of these reached
    // `@intFromFloat` with a value it cannot represent before the guard was
    // rewritten; in Debug that is a panic, which is a remotely triggerable
    // crash from a well-formed `QueryPoints`.
    const hostile = [_]f64{
        std.math.nan(f64),
        -std.math.nan(f64),
        std.math.inf(f64),
        -std.math.inf(f64),
        1e30,
        1e300,
        std.math.floatMax(f64),
        -1.0,
        0.0,
    };
    for (hostile) |o| {
        const got = oversampledLimit(100, o);
        try testing.expect(got >= 100);
    }

    // NaN specifically must behave as "no oversampling", not as "some huge
    // number". This is the assertion that fails if the guard is written the
    // obvious way round.
    try testing.expectEqual(@as(usize, 100), oversampledLimit(100, std.math.nan(f64)));
    try testing.expectEqual(@as(usize, 100 * 65536), oversampledLimit(100, std.math.inf(f64)));

    // No overflow at the top of the usize range.
    _ = oversampledLimit(std.math.maxInt(usize), 4.0);
    _ = oversampledLimit(std.math.maxInt(usize) / 2, 1e9);
}
