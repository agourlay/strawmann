//! §6.7 / M5, product quantization.
//!
//! | mode | encoding | distance | notes |
//! |---|---|---|---|
//! | `product-x{4,8,16,32,64}` | PQ, 8 bit/subquantizer | LUT ADC, scalar gather baseline | k-means codebooks trained on a sample |
//! | PQ4 / FastScan | 4 bit, `vpshufb`-based LUT | in-register table lookup | stretch goal; the version that actually competes |
//!
//! ## What `product-xN` means
//!
//! bfb's `--quantization product-x16` names a *compression ratio*, not a
//! subquantizer count. `xN` compresses fp32 by N×, so at 4 bytes per component
//! the code must be `4/N` bytes per component, which at 8 bits per
//! subquantizer means each subquantizer covers `N/4` dimensions.
//!
//! Getting this backwards is easy and produces a working index with the wrong
//! memory footprint, which would make every §5.4 working-set argument wrong
//! while recall still looked fine. `subquantizerCount` is the single place the
//! mapping lives.
//!
//! ## The ADC table
//!
//! The distance kernels live in `dist/pq_adc.zig`; this file owns the
//! codebooks, the training, and the encoding. §6.6.1 places PQ in the
//! "partially L3-resident, LUT lookup throughput is the kernel" regime, at
//! 192 MB for 1M×768 at x16 the codes fit a large L3, so unlike fp32 the memory
//! system stops hiding the kernel and the lookup itself becomes the limiter.

const std = @import("std");
const dist = @import("../dist/dist.zig");
const pq_adc = dist.pq_adc;

//: k-means passes per subspace. 25 was a literal at the one call site, beside
//: five other integers; it is the only training knob that was not named.
pub const default_iterations: usize = 25;

pub const centroids_8bit = 256;
pub const centroids_4bit = 16;

/// Subquantizers for a `product-xN` compression ratio at 8 bits per code.
///
/// `xN` means N× smaller than fp32. fp32 is `4·dim` bytes; the code is
/// `m` bytes at 8 bits each, so `m = 4·dim/N`.
pub fn subquantizerCount(dim: usize, ratio: usize) usize {
    std.debug.assert(ratio > 0);
    const m = (4 * dim) / ratio;
    return @max(1, m);
}

/// The subquantizer count a store *actually* builds for `(dim, ratio)`.
///
/// PQ needs equal subvectors, so when `subquantizerCount` does not divide
/// `dim` the store falls back to the largest `m` that does (`dim = 100` at
/// x64 asks for 6 and builds 5). This is the single definition of that rule:
/// `Mode.bytesPerVector` used to report the requested `m` while the store
/// applied the decrement privately, so a results row could quote a working set
/// the build never produced.
pub fn effectiveSubquantizerCount(dim: usize, ratio: usize) usize {
    var m = subquantizerCount(dim, ratio);
    while (m > 1 and dim % m != 0) m -= 1;
    return m;
}

/// Dimensions covered by each subquantizer.
pub fn subDim(dim: usize, m: usize) usize {
    return dim / m;
}

pub const Error = error{
    /// `dim` is not divisible by the subquantizer count. PQ requires equal
    /// subvectors; an uneven split is a configuration error rather than
    /// something to paper over with a ragged last block.
    UnevenSplit,
    OutOfMemory,
};

/// The shape of a codebook: how the vector splits, and how finely each part is
/// coded. One type because the three travelled as three adjacent `usize`
/// arguments through `Codebook.init` and `train`, where `m` and `centroids` are
/// interchangeable to the compiler and produce a codebook either way.
///
/// The same argument `quantized_search.Request` was written for, one directory
/// over: "`ef` and `limit` are both `usize` and adjacent. They are one thing,
/// so they are one type."
pub const Geometry = struct {
    dim: usize,
    /// Subquantizers: how many parts the vector splits into.
    m: usize,
    /// Centroids per subquantizer, i.e. the code alphabet.
    centroids: usize,

    pub fn subDim(self: Geometry) usize {
        return self.dim / self.m;
    }

    /// PQ requires equal subvectors; an uneven split is a configuration error
    /// rather than something to paper over with a ragged last block.
    pub fn validate(self: Geometry) Error!void {
        if (self.m == 0 or self.dim % self.m != 0) return Error.UnevenSplit;
    }
};

/// What to train the codebook *on*, as opposed to what shape it is.
pub const Training = struct {
    /// Sample vectors in the flat `sample` slice.
    samples: usize,
    iterations: usize,
    /// Up to one OS thread per subquantizer. The codebook is bit-identical for
    /// any value: each subspace's k-means is deterministic on its own, so only
    /// the wall clock moves.
    threads: usize = 1,
};

/// A trained product-quantization codebook.
pub const Codebook = struct {
    dim: usize,
    m: usize,
    sub_dim: usize,
    centroids: usize,
    /// `m × centroids × sub_dim` f32, row-major.
    data: []f32,

    pub fn init(alloc: std.mem.Allocator, geom: Geometry) Error!Codebook {
        try geom.validate();
        const sd = geom.subDim();
        const data = alloc.alloc(f32, geom.m * geom.centroids * sd) catch
            return Error.OutOfMemory;
        @memset(data, 0);
        return .{ .dim = geom.dim, .m = geom.m, .sub_dim = sd, .centroids = geom.centroids, .data = data };
    }

    pub fn geometry(self: *const Codebook) Geometry {
        return .{ .dim = self.dim, .m = self.m, .centroids = self.centroids };
    }

    pub fn deinit(self: *Codebook, alloc: std.mem.Allocator) void {
        alloc.free(self.data);
    }

    pub fn centroid(self: *const Codebook, sub: usize, c: usize) []f32 {
        const start = (sub * self.centroids + c) * self.sub_dim;
        return self.data[start..][0..self.sub_dim];
    }

    pub fn centroidConst(self: *const Codebook, sub: usize, c: usize) []const f32 {
        return self.centroid(sub, c);
    }

    /// Bytes per encoded vector: one per subquantizer. `encode` and `decode`
    /// write and read a byte per code whatever `centroids` is; a 4-bit
    /// packing is `pq_adc.packFastScanBlock`'s interleaved block layout, which
    /// no `Store` uses, and this accessor used to claim it for any codebook
    /// of 16 centroids, halving the footprint `compressionRatio` reported.
    pub fn codeBytes(self: *const Codebook) usize {
        return self.m;
    }

    /// Compression ratio against fp32, for the §5.4 working-set table.
    pub fn compressionRatio(self: *const Codebook) f64 {
        return @as(f64, @floatFromInt(self.dim * 4)) / @as(f64, @floatFromInt(self.codeBytes()));
    }
};

/// Deterministic k-means over one subspace.
///
/// §6.7: "k-means codebooks trained on a sample". Determinism matters here for
/// the same reason it matters for the graph (§8.7): a codebook that differs run
/// to run makes every quantized score differ run to run, and the differ could
/// not then distinguish "we disagree with Qdrant" from "we disagree with
/// ourselves".
///
/// So initialisation is seeded and by strided selection rather than by random
/// sampling, and the assignment tie-break is lowest centroid index.
/// The buffers one k-means run works in, owned by the caller so the allocator
/// is never asked to be thread-safe. Named for the same reason
/// `hnsw.Index.Scratch` and `quantized.QueryScratch` are: four bare slices in a
/// nine-argument signature say nothing about which is which.
pub const Scratch = struct {
    assign: []u32,
    sums: []f64,
    counts: []u32,
};

fn kmeansSubspace(
    sample: []const f32,
    n: usize,
    sub_dim: usize,
    k: usize,
    iterations: usize,
    out: []f32,
    scratch: Scratch,
) void {
    const assign, const sums, const counts = .{ scratch.assign, scratch.sums, scratch.counts };
    std.debug.assert(out.len == k * sub_dim);
    std.debug.assert(assign.len >= n and sums.len == k * sub_dim and counts.len == k);

    // Initialise from evenly spaced sample points. With fewer samples than
    // centroids the extras duplicate, which is harmless: duplicate centroids
    // simply never win an assignment.
    // Before the seeding loop, which reads `sample`: with no rows the slice
    // is one float long and any `sub_dim > 1` read past it.
    if (n == 0) {
        @memset(out, 0);
        return;
    }
    for (0..k) |c| {
        const src = (c * n) / k % n;
        @memcpy(out[c * sub_dim ..][0..sub_dim], sample[src * sub_dim ..][0..sub_dim]);
    }

    // Initialised, not left as the previous subspace left it. The convergence
    // check compares each new assignment against the previous one, so a
    // stale or uninitialised value on the first pass makes `moved == 0`, and
    // therefore how many iterations run, depend on what was there before.
    // That contradicts the determinism this function's doc comment promises,
    // and (when the scratch was allocated per call) the dependence was on
    // allocator *reuse across subspaces*, so it did not reproduce in
    // isolation. `maxInt` rather than 0 so the first pass always counts as
    // movement.
    @memset(assign[0..n], std.math.maxInt(u32));

    for (0..iterations) |_| {
        var moved: usize = 0;
        for (0..n) |i| {
            const v = sample[i * sub_dim ..][0..sub_dim];
            var best: u32 = 0;
            var best_d: f32 = std.math.inf(f32);
            for (0..k) |c| {
                const cent = out[c * sub_dim ..][0..sub_dim];
                var d: f32 = 0;
                for (v, cent) |x, y| {
                    const t = x - y;
                    d += t * t;
                }
                // Strict `<` makes the lowest index win ties, which is what
                // keeps the assignment reproducible.
                if (d < best_d) {
                    best_d = d;
                    best = @intCast(c);
                }
            }
            if (assign[i] != best) moved += 1;
            assign[i] = best;
        }

        @memset(sums, 0);
        @memset(counts, 0);
        for (0..n) |i| {
            const c = assign[i];
            counts[c] += 1;
            const v = sample[i * sub_dim ..][0..sub_dim];
            for (v, 0..) |x, d| sums[@as(usize, c) * sub_dim + d] += x;
        }
        for (0..k) |c| {
            if (counts[c] == 0) continue; // keep an empty centroid where it is
            const inv = 1.0 / @as(f64, @floatFromInt(counts[c]));
            for (0..sub_dim) |d| {
                out[c * sub_dim + d] = @floatCast(sums[c * sub_dim + d] * inv);
            }
        }
        if (moved == 0) break; // converged
    }
}

/// Train a codebook from a sample of full vectors, over up to `how.threads` OS
/// threads, one subspace at a time each. `sample` is `how.samples × geom.dim`
/// row-major.
///
/// The `m` k-means runs are independent and each is deterministic on its
/// own (`kmeansSubspace`), so the codebook is bit-identical for any thread
/// count; only the wall clock moves. It moved a lot: at d=128, m=8, 65,536
/// samples, 256 centroids and 25 iterations the serial loop was ~40 s of the
/// 107 s W8-upload measured, against Qdrant's 88 s for the
/// same row. Every thread owns its scratch (allocated here, before spawn), so
/// the allocator is not asked to be thread-safe.
///
/// `trainThreaded` was the same function with `threads` as its eighth
/// positional argument; it is a field with a default now, so the two spellings
/// had become one call.
pub fn train(
    alloc: std.mem.Allocator,
    sample: []const f32,
    geom: Geometry,
    how: Training,
) Error!Codebook {
    var cb = try Codebook.init(alloc, geom);
    errdefer cb.deinit(alloc);

    const n = how.samples;
    const t = @max(1, @min(how.threads, geom.m));
    const workers = alloc.alloc(SubspaceWorker, t) catch return Error.OutOfMemory;
    defer alloc.free(workers);
    var made: usize = 0;
    defer for (workers[0..made]) |*w| w.deinit(alloc);
    for (workers) |*w| {
        w.* = try SubspaceWorker.init(alloc, n, cb.sub_dim, geom.centroids);
        made += 1;
    }
    for (workers, 0..) |*w, i| {
        w.* = .{
            .sub = w.sub,
            .assign = w.assign,
            .sums = w.sums,
            .counts = w.counts,
            .sample = sample,
            .n = n,
            .dim = geom.dim,
            .cb = &cb,
            .iterations = how.iterations,
            .first = i,
            .step = t,
        };
    }
    if (t == 1) {
        workers[0].run();
        return cb;
    }
    const handles = alloc.alloc(std.Thread, t - 1) catch return Error.OutOfMemory;
    defer alloc.free(handles);
    var spawned: usize = 0;
    for (workers[1..], 0..) |*w, i| {
        handles[i] = std.Thread.spawn(.{}, SubspaceWorker.run, .{w}) catch break;
        spawned += 1;
    }
    // Whatever did not get a thread is done here, after this thread's own
    // share: a spawn failure costs wall clock, never a subspace.
    workers[0].run();
    for (workers[1 + spawned ..]) |*w| w.run();
    for (handles[0..spawned]) |h| h.join();
    return cb;
}

/// One thread's share of the subspaces, and the scratch it works in.
const SubspaceWorker = struct {
    /// The subspace gathered contiguously, `n × sub_dim`.
    sub: []f32,
    assign: []u32,
    sums: []f64,
    counts: []u32,
    sample: []const f32 = &.{},
    n: usize = 0,
    dim: usize = 0,
    cb: *Codebook = undefined,
    iterations: usize = 0,
    first: usize = 0,
    step: usize = 1,

    fn init(alloc: std.mem.Allocator, n: usize, sub_dim: usize, centroids: usize) Error!SubspaceWorker {
        const sub = alloc.alloc(f32, @max(1, n * sub_dim)) catch return Error.OutOfMemory;
        errdefer alloc.free(sub);
        const assign = alloc.alloc(u32, @max(1, n)) catch return Error.OutOfMemory;
        errdefer alloc.free(assign);
        const sums = alloc.alloc(f64, centroids * sub_dim) catch return Error.OutOfMemory;
        errdefer alloc.free(sums);
        const counts = alloc.alloc(u32, centroids) catch return Error.OutOfMemory;
        return .{ .sub = sub, .assign = assign, .sums = sums, .counts = counts };
    }

    fn deinit(self: *SubspaceWorker, alloc: std.mem.Allocator) void {
        alloc.free(self.sub);
        alloc.free(self.assign);
        alloc.free(self.sums);
        alloc.free(self.counts);
    }

    fn run(self: *SubspaceWorker) void {
        const cb = self.cb;
        var s = self.first;
        while (s < cb.m) : (s += self.step) {
            // Gather the subspace contiguously so k-means sees a dense array.
            // Doing this once costs one pass; strided access inside the
            // k-means inner loop would cost a cache miss per component.
            for (0..self.n) |i| {
                const src = self.sample[i * self.dim + s * cb.sub_dim ..][0..cb.sub_dim];
                @memcpy(self.sub[i * cb.sub_dim ..][0..cb.sub_dim], src);
            }
            kmeansSubspace(
                self.sub,
                self.n,
                cb.sub_dim,
                cb.centroids,
                self.iterations,
                cb.data[s * cb.centroids * cb.sub_dim ..][0 .. cb.centroids * cb.sub_dim],
                .{ .assign = self.assign, .sums = self.sums, .counts = self.counts },
            );
        }
    }
};

/// Encode one vector to `m` byte codes.
pub fn encode(cb: *const Codebook, dst: []u8, v: []const f32) void {
    std.debug.assert(dst.len == cb.m);
    std.debug.assert(v.len == cb.dim);
    for (0..cb.m) |s| {
        const sv = v[s * cb.sub_dim ..][0..cb.sub_dim];
        var best: u8 = 0;
        var best_d: f32 = std.math.inf(f32);
        for (0..cb.centroids) |c| {
            const cent = cb.centroidConst(s, c);
            var d: f32 = 0;
            for (sv, cent) |x, y| {
                const t = x - y;
                d += t * t;
            }
            if (d < best_d) {
                best_d = d;
                best = @intCast(c);
            }
        }
        dst[s] = best;
    }
}

/// Reconstruct a vector from its codes, for rescoring and for T4's fidelity
/// distributions.
pub fn decode(cb: *const Codebook, dst: []f32, codes: []const u8) void {
    std.debug.assert(dst.len == cb.dim);
    std.debug.assert(codes.len == cb.m);
    for (0..cb.m) |s| {
        @memcpy(dst[s * cb.sub_dim ..][0..cb.sub_dim], cb.centroidConst(s, codes[s]));
    }
}

/// Per-query ADC lookup table.
pub const QueryTable = struct {
    table: []f32,
    m: usize,
    centroids: usize,

    pub fn init(alloc: std.mem.Allocator, cb: *const Codebook) Error!QueryTable {
        const t = alloc.alloc(f32, cb.m * cb.centroids) catch return Error.OutOfMemory;
        return .{ .table = t, .m = cb.m, .centroids = cb.centroids };
    }

    pub fn deinit(self: *QueryTable, alloc: std.mem.Allocator) void {
        alloc.free(self.table);
    }

    /// Build the table for `query` under `kernel`.
    ///
    /// Both arms produce an *internal similarity* (higher is better), so the
    /// caller never needs to know which metric it is using, the same sign
    /// convention §8.3 establishes for the fp32 kernels.
    pub fn build(self: *QueryTable, cb: *const Codebook, kernel: dist.Kernel, query: []const f32) void {
        switch (kernel) {
            .dot => pq_adc.Pq8Adc.buildTableDot(self.table, query, cb.data, cb.m),
            .euclid => pq_adc.Pq8Adc.buildTableEuclid(self.table, query, cb.data, cb.m),
            .manhattan => {
                // No ADC form for L1 in `pq_adc`, so it is computed here.
                // Manhattan is not on bfb's quantization path, but leaving it
                // silently wrong would be worse than the few lines it costs.
                for (0..cb.m) |s| {
                    const qsub = query[s * cb.sub_dim ..][0..cb.sub_dim];
                    for (0..cb.centroids) |c| {
                        const cent = cb.centroidConst(s, c);
                        var acc: f32 = 0;
                        for (qsub, cent) |q, x| acc += @abs(q - x);
                        self.table[s * cb.centroids + c] = -acc;
                    }
                }
            },
        }
    }

    pub fn score(self: *const QueryTable, codes: []const u8) f32 {
        return pq_adc.Pq8Adc.score(self.table, codes);
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "product-xN maps to the subquantizer count that yields that ratio" {
    // x16 at d=768: 4*768/16 = 192 subquantizers, 192 bytes/vector.
    // §5.4's table says PQ x16 is 192 MB for 1M x 768, exactly this.
    try testing.expectEqual(@as(usize, 192), subquantizerCount(768, 16));
    try testing.expectEqual(@as(usize, 384), subquantizerCount(768, 8));
    try testing.expectEqual(@as(usize, 96), subquantizerCount(768, 32));
    try testing.expectEqual(@as(usize, 48), subquantizerCount(768, 64));
    try testing.expectEqual(@as(usize, 768), subquantizerCount(768, 4));
}

test "codebook geometry and the §5.4 compression ratio" {
    var cb = try Codebook.init(testing.allocator, .{ .dim = 768, .m = 192, .centroids = 256 });
    defer cb.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), cb.sub_dim);
    try testing.expectEqual(@as(usize, 192), cb.codeBytes());
    try testing.expectApproxEqAbs(@as(f64, 16.0), cb.compressionRatio(), 1e-9);

    // 16 centroids still encode to a byte per subquantizer: nothing packs
    // nibbles, so the ratio must say 16x, not the 32x it used to claim.
    var cb4 = try Codebook.init(testing.allocator, .{ .dim = 768, .m = 192, .centroids = 16 });
    defer cb4.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 192), cb4.codeBytes());
    try testing.expectApproxEqAbs(@as(f64, 16.0), cb4.compressionRatio(), 1e-9);
}

test "an uneven split is refused rather than silently truncated" {
    try testing.expectError(Error.UnevenSplit, Codebook.init(testing.allocator, .{ .dim = 100, .m = 7, .centroids = 16 }));
    try testing.expectError(Error.UnevenSplit, Codebook.init(testing.allocator, .{ .dim = 100, .m = 0, .centroids = 16 }));
}

test "training is deterministic" {
    // §8.7's requirement applied to codebooks: a codebook that varies run to
    // run makes every quantized score vary run to run.
    const n = 400;
    const dim = 16;
    var prng = std.Random.DefaultPrng.init(0x9c0);
    const rnd = prng.random();
    const sample = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(sample);
    for (sample) |*x| x.* = rnd.floatNorm(f32);

    var first: ?[]f32 = null;
    defer if (first) |f| testing.allocator.free(f);

    for (0..3) |run| {
        var cb = try train(testing.allocator, sample, .{ .dim = dim, .m = 4, .centroids = 16 }, .{ .samples = n, .iterations = 10 });
        defer cb.deinit(testing.allocator);
        if (run == 0) {
            first = try testing.allocator.dupe(f32, cb.data);
        } else {
            try testing.expectEqualSlices(f32, first.?, cb.data);
        }
    }
}

test "k-means reduces reconstruction error below random centroids" {
    // The property that makes training worth doing at all.
    const n = 600;
    const dim = 8;
    var prng = std.Random.DefaultPrng.init(0x9c1);
    const rnd = prng.random();
    const sample = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(sample);
    // Clustered data, so k-means has structure to find.
    for (0..n) |i| {
        const cluster: f32 = @floatFromInt(i % 4);
        for (0..dim) |d| sample[i * dim + d] = cluster * 5.0 + rnd.floatNorm(f32) * 0.2;
    }

    var trained = try train(testing.allocator, sample, .{ .dim = dim, .m = 2, .centroids = 4 }, .{ .samples = n, .iterations = 25 });
    defer trained.deinit(testing.allocator);

    var untrained = try Codebook.init(testing.allocator, .{ .dim = dim, .m = 2, .centroids = 4 });
    defer untrained.deinit(testing.allocator);
    for (untrained.data) |*x| x.* = rnd.floatNorm(f32);

    const err = struct {
        fn f(cb: *const Codebook, s: []const f32, count: usize, d: usize, alloc: std.mem.Allocator) !f64 {
            const codes = try alloc.alloc(u8, cb.m);
            defer alloc.free(codes);
            const recon = try alloc.alloc(f32, d);
            defer alloc.free(recon);
            var acc: f64 = 0;
            for (0..count) |i| {
                const v = s[i * d ..][0..d];
                encode(cb, codes, v);
                decode(cb, recon, codes);
                for (v, recon) |a, b| acc += (a - b) * (a - b);
            }
            return acc;
        }
    }.f;

    const e_trained = try err(&trained, sample, n, dim, testing.allocator);
    const e_untrained = try err(&untrained, sample, n, dim, testing.allocator);
    try testing.expect(e_trained < e_untrained * 0.1);
}

test "ADC score matches the dot product against the reconstruction" {
    // The defining property of ADC: the table sum must equal the exact score
    // against the vector the codes name.
    const n = 300;
    const dim = 32;
    var prng = std.Random.DefaultPrng.init(0x9c2);
    const rnd = prng.random();
    const sample = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(sample);
    for (sample) |*x| x.* = rnd.floatNorm(f32);

    var cb = try train(testing.allocator, sample, .{ .dim = dim, .m = 8, .centroids = 32 }, .{ .samples = n, .iterations = 15 });
    defer cb.deinit(testing.allocator);

    var codes: [8]u8 = undefined;
    encode(&cb, &codes, sample[0..dim]);
    var recon: [dim]f32 = undefined;
    decode(&cb, &recon, &codes);

    var query: [dim]f32 = undefined;
    for (&query) |*x| x.* = rnd.floatNorm(f32);

    var qt = try QueryTable.init(testing.allocator, &cb);
    defer qt.deinit(testing.allocator);

    qt.build(&cb, .dot, &query);
    const via_adc = qt.score(&codes);
    const direct = dist.reference.dotWide(&query, &recon);
    try testing.expect(@abs(@as(f64, via_adc) - direct) < 1e-4);

    // And the same for Euclid, where the sign convention also has to hold.
    qt.build(&cb, .euclid, &query);
    const euc_adc = qt.score(&codes);
    const euc_direct = dist.reference.euclidWide(&query, &recon);
    try testing.expect(@abs(@as(f64, euc_adc) - euc_direct) < 1e-4);
    try testing.expect(euc_adc <= 0.0); // higher is better
}

test "manhattan ADC has the right sign and value" {
    const dim = 8;
    var cb = try Codebook.init(testing.allocator, .{ .dim = dim, .m = 2, .centroids = 4 });
    defer cb.deinit(testing.allocator);
    // Centroid 0 of each subspace is all zeros; make centroid 1 all ones.
    for (0..2) |s| {
        @memset(cb.centroid(s, 1), 1.0);
    }

    var qt = try QueryTable.init(testing.allocator, &cb);
    defer qt.deinit(testing.allocator);
    var q = [_]f32{0.0} ** dim;
    qt.build(&cb, .manhattan, &q);

    // Codes (0,0): reconstruction is all zeros, L1 distance 0.
    try testing.expectEqual(@as(f32, 0.0), qt.score(&[_]u8{ 0, 0 }));
    // Codes (1,1): reconstruction is all ones, L1 distance 8, negated.
    try testing.expectEqual(@as(f32, -8.0), qt.score(&[_]u8{ 1, 1 }));
}

test "§6.7: PQ generates candidates for rescore, it does not rank directly" {
    // The property PQ is actually used for. §6.7: "quantized search produces
    // `limit × oversampling` candidates, rescored with the full-precision (or
    // next-tier) representation."
    //
    // Measured behaviour: PQ recovers only ~6 of the true top-10 *directly*,
    // because its centroids resolve which region a vector is in but not its
    // rank within that region, the residual inside a cell is exactly what the
    // encoding discards. Asserting a high direct top-10 overlap would be
    // asserting something false about product quantization.
    //
    // What *is* true, and what the rescore stage depends on, is that the true
    // neighbours land inside a modestly oversampled candidate set. That is the
    // number this test pins.
    const n = 500;
    const dim = 32;
    const k = 10;
    const m = 8;
    const oversample = 5;

    var prng = std.Random.DefaultPrng.init(0x9c3);
    const rnd = prng.random();

    // Clustered data, which is what a real embedding space looks like, §4.1
    // is explicit that isotropic noise is unrepresentative for exactly this
    // kind of measurement.
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    const n_clusters = 24;
    const centres = try testing.allocator.alloc(f32, n_clusters * dim);
    defer testing.allocator.free(centres);
    for (centres) |*x| x.* = rnd.floatNorm(f32) * 6.0;
    for (0..n) |i| {
        const c = i % n_clusters;
        for (0..dim) |d| data[i * dim + d] = centres[c * dim + d] + rnd.floatNorm(f32) * 0.35;
    }

    var cb = try train(testing.allocator, data, .{ .dim = dim, .m = m, .centroids = 64 }, .{ .samples = n, .iterations = 20 });
    defer cb.deinit(testing.allocator);

    const codes = try testing.allocator.alloc(u8, n * cb.m);
    defer testing.allocator.free(codes);
    for (0..n) |i| encode(&cb, codes[i * cb.m ..][0..cb.m], data[i * dim ..][0..dim]);

    var qt = try QueryTable.init(testing.allocator, &cb);
    defer qt.deinit(testing.allocator);

    const Pair = struct { id: u32, s: f32 };
    const exact = try testing.allocator.alloc(Pair, n);
    defer testing.allocator.free(exact);
    const approx = try testing.allocator.alloc(Pair, n);
    defer testing.allocator.free(approx);
    const lt = struct {
        fn f(_: void, a: Pair, b: Pair) bool {
            return a.s > b.s;
        }
    }.f;

    const queries = 25;
    var contained: usize = 0;
    const query = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(query);

    for (0..queries) |_| {
        // Held out per §4.3: drawn from the data's distribution, never a data
        // point itself.
        const pick = rnd.uintLessThan(usize, n);
        for (query, 0..) |*x, j| x.* = data[pick * dim + j] + rnd.floatNorm(f32) * 0.15;

        qt.build(&cb, .euclid, query);
        for (0..n) |i| {
            exact[i] = .{ .id = @intCast(i), .s = dist.native.euclid(query, data[i * dim ..][0..dim]) };
            approx[i] = .{ .id = @intCast(i), .s = qt.score(codes[i * cb.m ..][0..cb.m]) };
        }
        std.mem.sort(Pair, exact, {}, lt);
        std.mem.sort(Pair, approx, {}, lt);

        const cand = approx[0 .. k * oversample];
        for (exact[0..k]) |e| {
            for (cand) |a| {
                if (a.id == e.id) {
                    contained += 1;
                    break;
                }
            }
        }
    }

    const candidate_recall = @as(f64, @floatFromInt(contained)) /
        @as(f64, @floatFromInt(queries * k));
    // With 5x oversampling the candidate set must contain nearly all the true
    // neighbours, or the rescore stage has nothing to recover and §6.7's
    // (recall, latency) frontier collapses.
    try testing.expect(candidate_recall > 0.9);
}

test "PQ direct top-k overlap is modest, which is why rescore exists" {
    // Documents the measured behaviour above as an explicit, checkable fact
    // rather than leaving it as a comment: PQ alone recovers well under the
    // full top-10, so any design that skips rescore is trading far more recall
    // than §6.7's table implies.
    const n = 400;
    const dim = 32;
    const k = 10;
    var prng = std.Random.DefaultPrng.init(0x9c7);
    const rnd = prng.random();

    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = rnd.floatNorm(f32);

    var cb = try train(testing.allocator, data, .{ .dim = dim, .m = 8, .centroids = 64 }, .{ .samples = n, .iterations = 20 });
    defer cb.deinit(testing.allocator);
    const codes = try testing.allocator.alloc(u8, n * cb.m);
    defer testing.allocator.free(codes);
    for (0..n) |i| encode(&cb, codes[i * cb.m ..][0..cb.m], data[i * dim ..][0..dim]);

    var query: [dim]f32 = undefined;
    for (&query) |*x| x.* = rnd.floatNorm(f32);
    var qt = try QueryTable.init(testing.allocator, &cb);
    defer qt.deinit(testing.allocator);
    qt.build(&cb, .euclid, &query);

    const Pair = struct { id: u32, s: f32 };
    const exact = try testing.allocator.alloc(Pair, n);
    defer testing.allocator.free(exact);
    const approx = try testing.allocator.alloc(Pair, n);
    defer testing.allocator.free(approx);
    for (0..n) |i| {
        exact[i] = .{ .id = @intCast(i), .s = dist.native.euclid(&query, data[i * dim ..][0..dim]) };
        approx[i] = .{ .id = @intCast(i), .s = qt.score(codes[i * cb.m ..][0..cb.m]) };
    }
    const lt = struct {
        fn f(_: void, a: Pair, b: Pair) bool {
            return a.s > b.s;
        }
    }.f;
    std.mem.sort(Pair, exact, {}, lt);
    std.mem.sort(Pair, approx, {}, lt);

    var overlap: usize = 0;
    for (approx[0..k]) |a| {
        for (exact[0..k]) |e| {
            if (a.id == e.id) {
                overlap += 1;
                break;
            }
        }
    }
    // Non-vacuous in both directions: PQ carries real signal (not zero) but is
    // not a substitute for the exact score (not ten).
    try testing.expect(overlap >= 3);
    try testing.expect(overlap < k);
}
