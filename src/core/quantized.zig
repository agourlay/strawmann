//! §6.7, quantized storage attached to a collection, and the two-stage
//! search it enables.
//!
//! §6.7: "Rescoring: quantized search produces `limit × oversampling`
//! candidates, rescored with the full-precision (or next-tier) representation.
//! Given §5.3, the rescore stage is often the dominant cost at low `ef`, so
//! measure the **combined** curve of (recall, latency) across
//! `(ef, oversampling, rescore-encoding)` rather than tuning stages
//! independently."
//!
//! That sentence is the design. The search is two stages with three knobs, and
//! all three have to be reachable from the request:
//!
//!   `ef`            , how wide the graph traversal goes      (SearchParams.hnsw_ef)
//!   `oversampling`  , how many candidates survive to rescore (QuantizationSearchParams.oversampling)
//!   rescore encoding, what the second stage scores against   (§5.3's suggestion)
//!
//! §5.3's binary row, "~37 µs + ~36 µs", is what makes this worth the
//! complexity: at that point the rescore stage costs as much as the entire
//! graph traversal, so treating it as a fixed postlude rather than a tuned
//! stage would leave half the latency unexamined.

const std = @import("std");
const dist = @import("../dist/dist.zig");
const quant = @import("../quant/quant.zig");
const heap = @import("../index/heap.zig");

pub const Mode = quant.Mode;
pub const Candidate = heap.Candidate;

pub const Error = error{ OutOfMemory, UnevenSplit, Unsupported };

/// Quantized codes for one collection, alongside the fp32 arena.
///
/// The fp32 vectors are *not* discarded: §6.7's rescore stage needs them, and
/// §1's non-goals do not include memory efficiency. Quantization here buys
/// traversal speed by shrinking the *hot* working set, which is §5.4's whole
/// argument, "Quantization's real win is not fewer ALU ops and not even fewer
/// bytes, it is **moving the working set up a level of the memory hierarchy**."
pub const Store = union(enum) {
    none,
    scalar: Scalar,
    binary: Binary,
    product: Product,

    pub const Scalar = struct {
        params: quant.scalar.Params,
        /// `capacity × dim` codes.
        codes: []u8,
        /// `capacity` per-row sums (`Σcode`, `Σcode²`) for the symmetric
        /// kernels, written by the same call that writes the row's codes.
        stats: []quant.scalar.RowStats,
        dim: usize,

        pub fn row(self: *const Scalar, offset: u32) []const u8 {
            return self.codes[@as(usize, offset) * self.dim ..][0..self.dim];
        }

        fn setRow(self: *Scalar, offset: u32, vec: []const f32) void {
            const codes = self.codes[@as(usize, offset) * self.dim ..][0..self.dim];
            quant.scalar.encode(self.params, codes, vec);
            self.stats[offset] = quant.scalar.RowStats.of(codes);
        }
    };

    pub const Binary = struct {
        codes: quant.binary.Codes,
        dim: usize,
    };

    pub const Product = struct {
        codebook: quant.pq.Codebook,
        /// `capacity × m` codes.
        codes: []u8,

        pub fn row(self: *const Product, offset: u32) []const u8 {
            return self.codes[@as(usize, offset) * self.codebook.m ..][0..self.codebook.m];
        }
    };

    /// Re-encode one row in place, for a point overwritten after the store
    /// was built (`Collection.noteOverwrite`).
    ///
    /// The store's parameters (SQ8 bounds, PQ codebook) are not retrained: one
    /// row does not move the quantiles, and retraining would re-encode every
    /// row. Called with the collection's write lock held, on a row the store
    /// covers.
    pub fn encodeRow(self: *Store, offset: u32, vec: []const f32) void {
        switch (self.*) {
            .none => {},
            .scalar => |*s| s.setRow(offset, vec),
            .binary => |*b| b.codes.set(offset, vec),
            .product => |*p| quant.pq.encode(&p.codebook, p.codes[@as(usize, offset) * p.codebook.m ..][0..p.codebook.m], vec),
        }
    }

    pub fn deinit(self: *Store, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .none => {},
            .scalar => |*s| {
                alloc.free(s.codes);
                alloc.free(s.stats);
            },
            .binary => |*b| b.codes.deinit(alloc),
            .product => |*p| {
                p.codebook.deinit(alloc);
                alloc.free(p.codes);
            },
        }
        self.* = .none;
    }

    /// What this store holds. Deliberately not a `Mode`: a `Mode` is what was
    /// *requested*, and `buildProduct` reduces `m` until the subvectors divide
    /// evenly, so the ratio it achieved need not be one of the five a request
    /// can name. This used to return `.{ .product = round(compressionRatio()) }`,
    /// which manufactured a request that nothing would have accepted.
    pub fn encoding(self: *const Store) quant.Encoding {
        return switch (self.*) {
            .none => .none,
            .scalar => .scalar,
            .binary => .binary,
            .product => .product,
        };
    }

    /// The compression a built PQ store actually achieved, which is the number
    /// a results row should quote. Null for the encodings whose ratio is fixed
    /// by the encoding itself.
    pub fn achievedCompression(self: *const Store) ?f64 {
        return switch (self.*) {
            .none, .scalar, .binary => null,
            .product => |p| p.codebook.compressionRatio(),
        };
    }

    /// Bytes per vector, for the §5.4 working-set table and for §5.7's
    /// "bytes touched (derived) and % of modelled roofline".
    pub fn bytesPerVector(self: *const Store) usize {
        return switch (self.*) {
            .none => 0,
            .scalar => |s| s.dim,
            .binary => |b| b.codes.words * @sizeOf(u64),
            .product => |p| p.codebook.m,
        };
    }
};

/// Vector accessor the builder needs, kept abstract so this file does not
/// depend on `Collection` (which depends on this one).
pub const Source = struct {
    ctx: *const anyopaque,
    /// Stored fp32 vector for an offset.
    row: *const fn (ctx: *const anyopaque, offset: u32) []const f32,
    /// The same, widening into the caller's `scratch` (of `dim` f32) when
    /// the store is narrower than fp32, so several threads can read rows at
    /// once. Null means `row` is safe to call from any thread as it is
    /// (fp32 storage) or the build is serial.
    row_into: ?*const fn (ctx: *const anyopaque, offset: u32, scratch: []f32) []const f32 = null,
    count: usize,
    dim: usize,
};

/// Sample size for training.
///
/// §6.7 says codebooks are "trained on a sample". 65536 is far more than any
/// quantizer here needs to converge and costs a fraction of a second, while
/// scanning a full 1M-point collection for bounds would add meaningfully to
/// the W2 build-time measurement for no accuracy gain.
pub const train_sample = 65_536;

/// The knobs a `QuantizationConfig` carries beyond the mode.
pub const BuildOptions = struct {
    /// `ScalarQuantization.quantile`: the central fraction of the component
    /// distribution the SQ8 range covers. 1.0 is plain min/max, which is
    /// Qdrant's behaviour when the field is absent; bfb sends 0.99 (§6.7).
    quantile: f32 = quant.scalar.default_quantile,
    /// Threads for the parts of a build that parallelise without changing
    /// the result: PQ codebook training (one subspace per task) and PQ
    /// encoding (rows in ranges). The store is bit-identical for any value.
    threads: usize = 1,
};

/// Build the quantized store for a collection, with the default options.
pub fn build(alloc: std.mem.Allocator, mode: Mode, src: Source, capacity: usize) Error!Store {
    return buildWith(alloc, mode, src, capacity, .{});
}

pub fn buildWith(alloc: std.mem.Allocator, mode: Mode, src: Source, capacity: usize, opts: BuildOptions) Error!Store {
    return switch (mode) {
        .none => .none,
        .scalar => try buildScalar(alloc, src, capacity, opts.quantile),
        .binary => try buildBinary(alloc, src, capacity),
        .product => |cr| try buildProduct(alloc, src, capacity, cr, opts.threads),
    };
}

fn sampleStride(count: usize) usize {
    return @max(1, count / train_sample);
}

fn buildScalar(alloc: std.mem.Allocator, src: Source, capacity: usize, quantile: f32) Error!Store {
    // Gather a strided sample of components for the quantile bounds. Strided
    // rather than the first N points, because ingest order is often correlated
    // with content and the first N would train on one region of the space.
    const stride = sampleStride(src.count);
    var sample_len: usize = 0;
    var i: usize = 0;
    while (i < src.count) : (i += stride) sample_len += src.dim;
    if (sample_len == 0) sample_len = 1;

    const sample = alloc.alloc(f32, sample_len) catch return Error.OutOfMemory;
    defer alloc.free(sample);
    var w: usize = 0;
    i = 0;
    while (i < src.count and w + src.dim <= sample.len) : (i += stride) {
        @memcpy(sample[w..][0..src.dim], src.row(src.ctx, @intCast(i)));
        w += src.dim;
    }
    if (w == 0) {
        sample[0] = 0;
        w = 1;
    }

    const scratch = alloc.alloc(f32, w) catch return Error.OutOfMemory;
    defer alloc.free(scratch);
    // The requested quantile, not the constant: `ScalarQuantization.quantile`
    // used to be read off the wire and then ignored here, so a client asking
    // for 0.9 or 1.0 got 0.99 and a store whose bounds it did not choose.
    const params = quant.scalar.train(sample[0..w], quantile, src.dim, scratch);

    const codes = alloc.alloc(u8, capacity * src.dim) catch return Error.OutOfMemory;
    errdefer alloc.free(codes);
    @memset(codes, 0);
    const stats = alloc.alloc(quant.scalar.RowStats, capacity) catch return Error.OutOfMemory;
    // Rows the build did not write score as all-zero codes, `Σ = Σ² = 0`,
    // which is what `RowStats.of` says of a zero row.
    @memset(stats, .{ .sum = 0, .sq = 0 });
    var store = Store.Scalar{ .params = params, .codes = codes, .stats = stats, .dim = src.dim };
    for (0..src.count) |n| store.setRow(@intCast(n), src.row(src.ctx, @intCast(n)));
    return .{ .scalar = store };
}

fn buildBinary(alloc: std.mem.Allocator, src: Source, capacity: usize) Error!Store {
    var codes = quant.binary.Codes.init(alloc, src.dim, capacity) catch return Error.OutOfMemory;
    errdefer codes.deinit(alloc);
    for (0..src.count) |n| codes.set(@intCast(n), src.row(src.ctx, @intCast(n)));
    return .{ .binary = .{ .codes = codes, .dim = src.dim } };
}

fn buildProduct(alloc: std.mem.Allocator, src: Source, capacity: usize, cr: quant.CompressionRatio, threads: usize) Error!Store {
    // PQ needs equal subvectors. Rather than reject a dimension that does not
    // divide evenly, fall back to the largest m that does. The rule lives in
    // `pq.effectiveSubquantizerCount` so `Mode.bytesPerVector` reports the
    // count that is actually built rather than the nominal one.
    const m = quant.pq.effectiveSubquantizerCount(src.dim, cr.ratio());
    if (src.dim % m != 0) return Error.UnevenSplit;

    const stride = sampleStride(src.count);
    var n_sample: usize = 0;
    var i: usize = 0;
    while (i < src.count) : (i += stride) n_sample += 1;
    if (n_sample == 0) n_sample = 1;

    const sample = alloc.alloc(f32, n_sample * src.dim) catch return Error.OutOfMemory;
    defer alloc.free(sample);
    var w: usize = 0;
    i = 0;
    while (i < src.count and w < n_sample) : (i += stride) {
        @memcpy(sample[w * src.dim ..][0..src.dim], src.row(src.ctx, @intCast(i)));
        w += 1;
    }
    if (w == 0) {
        @memset(sample, 0);
        w = 1;
    }

    var cb = try quant.pq.train(alloc, sample[0 .. w * src.dim], .{ .dim = src.dim, .m = m, .centroids = quant.pq.centroids_8bit }, .{ .samples = w, .iterations = quant.pq.default_iterations, .threads = threads });
    errdefer cb.deinit(alloc);

    const codes = alloc.alloc(u8, capacity * cb.m) catch return Error.OutOfMemory;
    errdefer alloc.free(codes);
    @memset(codes, 0);
    try encodeAllProduct(alloc, &cb, codes, src, threads);
    return .{ .product = .{ .codebook = cb, .codes = codes } };
}

/// Encode every row of `src` into `codes`, over up to `threads` threads when
/// the source can be read from several at once (`Source.row_into`), in
/// contiguous row ranges. Row `n` is encoded from its own vector by a
/// deterministic function, so the codes do not depend on the split.
fn encodeAllProduct(alloc: std.mem.Allocator, cb: *const quant.pq.Codebook, codes: []u8, src: Source, threads: usize) Error!void {
    const Task = struct {
        cb: *const quant.pq.Codebook,
        codes: []u8,
        src: Source,
        scratch: []f32,
        lo: usize,
        hi: usize,

        fn run(self: *@This()) void {
            var n = self.lo;
            while (n < self.hi) : (n += 1) {
                const off: u32 = @intCast(n);
                const v = if (self.src.row_into) |ri| ri(self.src.ctx, off, self.scratch) else self.src.row(self.src.ctx, off);
                quant.pq.encode(self.cb, self.codes[n * self.cb.m ..][0..self.cb.m], v);
            }
        }
    };
    // Rows are cheap enough that under a few tens of thousands one thread is
    // faster than spawning; and without `row_into` the source is not known
    // to be safe to read concurrently.
    const t = if (src.row_into == null or src.count < 32_768) 1 else @max(1, @min(threads, 64));
    if (t == 1) {
        var one = Task{ .cb = cb, .codes = codes, .src = src, .scratch = &.{}, .lo = 0, .hi = src.count };
        if (src.row_into != null) {
            one.scratch = alloc.alloc(f32, src.dim) catch return Error.OutOfMemory;
        }
        defer if (one.scratch.len > 0) alloc.free(one.scratch);
        one.run();
        return;
    }
    const tasks = alloc.alloc(Task, t) catch return Error.OutOfMemory;
    defer alloc.free(tasks);
    const scratch = alloc.alloc(f32, t * src.dim) catch return Error.OutOfMemory;
    defer alloc.free(scratch);
    const per = (src.count + t - 1) / t;
    for (tasks, 0..) |*task, i| {
        task.* = .{
            .cb = cb,
            .codes = codes,
            .src = src,
            .scratch = scratch[i * src.dim ..][0..src.dim],
            .lo = @min(i * per, src.count),
            .hi = @min((i + 1) * per, src.count),
        };
    }
    const handles = alloc.alloc(std.Thread, t - 1) catch return Error.OutOfMemory;
    defer alloc.free(handles);
    var spawned: usize = 0;
    for (tasks[1..], 0..) |*task, i| {
        handles[i] = std.Thread.spawn(.{}, Task.run, .{task}) catch break;
        spawned += 1;
    }
    tasks[0].run();
    for (tasks[1 + spawned ..]) |*task| task.run();
    for (handles[0..spawned]) |h| h.join();
}

// =========================================================================
// Query-time state
// =========================================================================

/// Per-query encoded form of the query vector, plus whatever the score needs.
///
/// Built once per query and pointed at by the traversal's score callback, so
/// the encoding cost is paid once rather than per distance evaluation.
pub const Query = struct {
    store: *const Store,
    kernel: dist.Kernel,

    // Scalar: the query quantised once (`SymmetricQuery`), so stage 1 is
    // integer throughout. The fp32 query stays for the rescore stage.
    sym: ?quant.scalar.SymmetricQuery = null,
    fp32: []const f32 = &.{},

    // Binary
    bin_codes: []u64 = &.{},

    // Product
    table: ?*quant.pq.QueryTable = null,

    /// Score one node with the quantized representation.
    pub fn score(ctx: *const anyopaque, node: u32) f32 {
        const self: *const Query = @ptrCast(@alignCast(ctx));
        return switch (self.store.*) {
            .none => 0,
            .scalar => |s| switch (self.kernel) {
                .dot => self.sym.?.dot(s.row(node), s.stats[node]),
                .euclid => self.sym.?.euclid(s.row(node), s.stats[node]),
                // L1, not L2. Sharing the Euclid arm returned a *squared
                // Euclidean* distance to a client that asked for Manhattan -
                // and with `rescore = false` that value goes out on the wire as
                // the score. §8.3 fixes what each metric returns; a quantized
                // path is not licensed to return a different quantity.
                .manhattan => self.sym.?.manhattan(s.row(node)),
            },
            .binary => |b| quant.binary.signDot(b.dim, self.bin_codes, b.codes.rowConst(node)),
            .product => |p| self.table.?.score(p.row(node)),
        };
    }
};

/// Scratch a worker needs to run a quantized query.
pub const QueryScratch = struct {
    bin_codes: []u64,
    /// The scalar query's codes, biased for `vpdpbusd` and plain for L1.
    sq_signed: []i8,
    sq_unsigned: []u8,
    table: quant.pq.QueryTable,
    table_valid: bool = false,
    /// Candidate ids surviving the first stage.
    candidates: []u32,
    /// Rescored results.
    rescored: []quant.binary.Scored,

    pub fn init(alloc: std.mem.Allocator, max_dim: usize, max_candidates: usize) !QueryScratch {
        const bin_codes = try alloc.alloc(u64, quant.binary.paddedWordsFor(max_dim));
        errdefer alloc.free(bin_codes);
        const sq_signed = try alloc.alloc(i8, max_dim);
        errdefer alloc.free(sq_signed);
        const sq_unsigned = try alloc.alloc(u8, max_dim);
        errdefer alloc.free(sq_unsigned);
        const candidates = try alloc.alloc(u32, max_candidates);
        errdefer alloc.free(candidates);
        const rescored = try alloc.alloc(quant.binary.Scored, max_candidates);
        return .{
            .bin_codes = bin_codes,
            .sq_signed = sq_signed,
            .sq_unsigned = sq_unsigned,
            .table = .{ .table = &.{}, .m = 0, .centroids = 0 },
            .candidates = candidates,
            .rescored = rescored,
        };
    }

    pub fn deinit(self: *QueryScratch, alloc: std.mem.Allocator) void {
        alloc.free(self.bin_codes);
        alloc.free(self.sq_signed);
        alloc.free(self.sq_unsigned);
        if (self.table_valid) self.table.deinit(alloc);
        alloc.free(self.candidates);
        alloc.free(self.rescored);
    }

    pub fn ensureTable(self: *QueryScratch, alloc: std.mem.Allocator, cb: *const quant.pq.Codebook) !void {
        if (self.table_valid and self.table.m == cb.m and self.table.centroids == cb.centroids) return;
        if (self.table_valid) self.table.deinit(alloc);
        self.table = try quant.pq.QueryTable.init(alloc, cb);
        self.table_valid = true;
    }
};

/// Prepare a `Query` for the given store and query vector.
pub fn prepareQuery(
    store: *const Store,
    kernel: dist.Kernel,
    query: []const f32,
    scratch: *QueryScratch,
    alloc: std.mem.Allocator,
) !Query {
    var q = Query{ .store = store, .kernel = kernel, .fp32 = query };
    switch (store.*) {
        .none => {},
        .scalar => |s| q.sym = quant.scalar.SymmetricQuery.init(
            s.params,
            scratch.sq_signed[0..query.len],
            scratch.sq_unsigned[0..query.len],
            query,
        ),
        .binary => {
            // Padded, so the query and the stored rows are the same length and
            // the kernel is fully vectorised on both.
            const words = quant.binary.paddedWordsFor(query.len);
            quant.binary.encode(scratch.bin_codes[0..words], query);
            q.bin_codes = scratch.bin_codes[0..words];
        },
        .product => |p| {
            try scratch.ensureTable(alloc, &p.codebook);
            scratch.table.build(&p.codebook, kernel, query);
            q.table = &scratch.table;
        },
    }
    return q;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

const TestSource = struct {
    data: []f32,
    dim: usize,
    n: usize,

    fn row(ctx: *const anyopaque, offset: u32) []const f32 {
        const self: *const TestSource = @ptrCast(@alignCast(ctx));
        return self.data[@as(usize, offset) * self.dim ..][0..self.dim];
    }

    fn rowInto(ctx: *const anyopaque, offset: u32, scratch: []f32) []const f32 {
        _ = scratch; // fp32 in memory: nothing to widen, the row itself is returned
        return row(ctx, offset);
    }

    fn source(self: *const TestSource) Source {
        return .{ .ctx = @ptrCast(self), .row = row, .row_into = rowInto, .count = self.n, .dim = self.dim };
    }
};

test "a product store is bit-identical whatever the thread count (§8.7)" {
    // Training is one k-means per subspace and encoding is one row at a time
    // from its own vector, so neither may depend on how they were split
    // across threads. 40,000 rows clears `encodeAllProduct`'s serial cutoff.
    // Small subspaces keep the Debug-mode k-means affordable.
    const dim = 8;
    const n = 40_000;
    var src = try makeSource(testing.allocator, n, dim, 0x9e7);
    defer testing.allocator.free(src.data);
    var serial = try buildWith(testing.allocator, .{ .product = .x4 }, src.source(), n, .{ .threads = 1 });
    defer serial.deinit(testing.allocator);
    var threaded = try buildWith(testing.allocator, .{ .product = .x4 }, src.source(), n, .{ .threads = 5 });
    defer threaded.deinit(testing.allocator);
    try testing.expectEqualSlices(f32, serial.product.codebook.data, threaded.product.codebook.data);
    try testing.expectEqualSlices(u8, serial.product.codes, threaded.product.codes);
    // And a thread count above the subspace count is clamped, not an error.
    var many = try buildWith(testing.allocator, .{ .product = .x4 }, src.source(), n, .{ .threads = 64 });
    defer many.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, serial.product.codes, many.product.codes);
}

fn makeSource(alloc: std.mem.Allocator, n: usize, dim: usize, seed: u64) !TestSource {
    const data = try alloc.alloc(f32, n * dim);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (data) |*x| x.* = rnd.floatNorm(f32);
    return .{ .data = data, .dim = dim, .n = n };
}

test "§5.4 bytes per vector matches the working-set table" {
    const dim = 768;
    var src = try makeSource(testing.allocator, 100, dim, 1);
    defer testing.allocator.free(src.data);

    for ([_]struct { Mode, usize }{
        .{ .scalar, 768 },
        // Stored, not logical: padded to a whole vector register.
        .{ .binary, 128 },
        .{ .{ .product = .x16 }, 192 },
    }) |case| {
        var store = try build(testing.allocator, case[0], src.source(), 100);
        defer store.deinit(testing.allocator);
        try testing.expectEqual(case[1], store.bytesPerVector());
    }
}

test "scalar store round-trips and scores close to fp32" {
    const dim = 128;
    const n = 200;
    var src = try makeSource(testing.allocator, n, dim, 2);
    defer testing.allocator.free(src.data);

    var store = try build(testing.allocator, .scalar, src.source(), n);
    defer store.deinit(testing.allocator);

    var scratch = try QueryScratch.init(testing.allocator, dim, 64);
    defer scratch.deinit(testing.allocator);

    const q = src.data[0..dim];
    const query = try prepareQuery(&store, .dot, q, &scratch, testing.allocator);

    // Quantized and exact scores must correlate strongly; a sign flip or a
    // wrong `lo` term would show up as a near-zero or negative correlation.
    var agree: usize = 0;
    for (0..n) |i| {
        for (0..n) |j| {
            if (i >= j) continue;
            const qi = Query.score(@ptrCast(&query), @intCast(i));
            const qj = Query.score(@ptrCast(&query), @intCast(j));
            const ei = dist.native.dot(q, src.data[i * dim ..][0..dim]);
            const ej = dist.native.dot(q, src.data[j * dim ..][0..dim]);
            if ((qi > qj) == (ei > ej)) agree += 1;
        }
    }
    const pairs = n * (n - 1) / 2;
    const concordance = @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(pairs));
    try testing.expect(concordance > 0.95);
}

test "binary store scores are consistent with the sign dot product" {
    const dim = 256;
    const n = 100;
    var src = try makeSource(testing.allocator, n, dim, 3);
    defer testing.allocator.free(src.data);

    var store = try build(testing.allocator, .binary, src.source(), n);
    defer store.deinit(testing.allocator);
    var scratch = try QueryScratch.init(testing.allocator, dim, 64);
    defer scratch.deinit(testing.allocator);

    const q = src.data[0..dim];
    const query = try prepareQuery(&store, .dot, q, &scratch, testing.allocator);

    // A vector scored against itself must be the maximum: Hamming 0, which
    // on Qdrant's sign-dot scale (`dim - 2*hamming`, the value that reaches
    // the wire when `rescore = false`) is `dim`, not the old `-hamming` 0.
    const self_score = Query.score(@ptrCast(&query), 0);
    try testing.expectEqual(@as(f32, @floatFromInt(dim)), self_score);
    for (1..n) |i| {
        try testing.expect(Query.score(@ptrCast(&query), @intCast(i)) <= self_score);
    }
}

test "product store builds and scores, falling back to a divisible m" {
    // d=100 with ratio 16 wants m=25, which does divide 100. d=768/x16 wants
    // 192, which divides. Pick a case that does *not* divide to exercise the
    // fallback: d=100, ratio 8 wants m=50 (divides), ratio 64 wants m=6 (does
    // not divide 100) -> falls back to 5.
    const dim = 100;
    const n = 300;
    var src = try makeSource(testing.allocator, n, dim, 4);
    defer testing.allocator.free(src.data);

    var store = try build(testing.allocator, .{ .product = .x64 }, src.source(), n);
    defer store.deinit(testing.allocator);

    const m = store.product.codebook.m;
    try testing.expect(m >= 1);
    try testing.expectEqual(@as(usize, 0), dim % m);

    var scratch = try QueryScratch.init(testing.allocator, dim, 64);
    defer scratch.deinit(testing.allocator);
    const q = src.data[0..dim];
    const query = try prepareQuery(&store, .euclid, q, &scratch, testing.allocator);

    // Euclid scores are negated distances, so all non-positive.
    for (0..n) |i| {
        try testing.expect(Query.score(@ptrCast(&query), @intCast(i)) <= 1e-4);
    }
}

test "the scalar quantile is honoured, not pinned to the constant" {
    // At quantile 1.0 the SQ8 range is the sample's exact min/max, so the
    // extreme value round-trips to the top code; at 0.5 the range covers only
    // the central half and the same value clips. A store that ignored the
    // option would produce identical codes for both.
    const dim = 8;
    const n = 400;
    var src = try makeSource(testing.allocator, n, dim, 7);
    defer testing.allocator.free(src.data);
    // Plant one extreme component so the outlier is unambiguous.
    src.data[0] = 50.0;

    var wide = try buildWith(testing.allocator, .scalar, src.source(), n, .{ .quantile = 1.0 });
    defer wide.deinit(testing.allocator);
    var narrow = try buildWith(testing.allocator, .scalar, src.source(), n, .{ .quantile = 0.5 });
    defer narrow.deinit(testing.allocator);

    // Full range: the outlier maps to the top code and the range is wide.
    try testing.expectEqual(@as(u8, 255), wide.scalar.row(0)[0]);
    // Central half: the range is far narrower, so `alpha` (the step) is
    // smaller, and the outlier still clips to the top code while an ordinary
    // component lands on a *different* code than under the wide range.
    try testing.expect(narrow.scalar.params.alpha < wide.scalar.params.alpha);
    try testing.expectEqual(@as(u8, 255), narrow.scalar.row(0)[0]);
    try testing.expect(!std.mem.eql(u8, wide.scalar.row(1), narrow.scalar.row(1)));
}

test "encodeRow rewrites exactly one row's codes" {
    const dim = 16;
    const n = 64;
    var src = try makeSource(testing.allocator, n, dim, 8);
    defer testing.allocator.free(src.data);

    for ([_]Mode{ .scalar, .binary, .{ .product = .x4 } }) |m| {
        var store = try build(testing.allocator, m, src.source(), n);
        defer store.deinit(testing.allocator);

        var scratch = try QueryScratch.init(testing.allocator, dim, 16);
        defer scratch.deinit(testing.allocator);

        // Score row 5 against row 7's vector, before and after row 5 is
        // re-encoded as row 7's vector: afterwards it must score as row 7
        // does, and row 6 must be untouched.
        const q = src.data[7 * dim ..][0..dim];
        const query = try prepareQuery(&store, .dot, q, &scratch, testing.allocator);
        const six_before = Query.score(@ptrCast(&query), 6);
        const seven = Query.score(@ptrCast(&query), 7);

        store.encodeRow(5, q);
        try testing.expectEqual(seven, Query.score(@ptrCast(&query), 5));
        try testing.expectEqual(six_before, Query.score(@ptrCast(&query), 6));
    }
}

test "none mode produces an empty store" {
    var src = try makeSource(testing.allocator, 10, 8, 5);
    defer testing.allocator.free(src.data);
    var store = try build(testing.allocator, .none, src.source(), 10);
    defer store.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), store.bytesPerVector());
    try testing.expectEqual(quant.Encoding.none, store.encoding());
}

test "building over an empty collection does not fault" {
    var src = TestSource{ .data = &.{}, .dim = 16, .n = 0 };
    for ([_]Mode{ .scalar, .binary, .{ .product = .x16 } }) |m| {
        var store = try build(testing.allocator, m, src.source(), 4);
        defer store.deinit(testing.allocator);
    }
}
