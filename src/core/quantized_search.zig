//! §6.7, quantized two-stage search.
//!
//! Split out of `collection.zig`, which had grown to 2200 lines spanning four
//! unrelated concerns: the collection type and its ingest path, §6.5's index
//! construction, this, and §6.4's persistence. They share only the `Collection`
//! type, so keeping them together bought nothing and made the file impossible
//! to hold in one's head.
//!
//! Everything here operates on a `*Collection` rather than being a method on
//! it, which is what makes the split mechanical: no behaviour moved, only text.

const std = @import("std");
const testing = std.testing;

const collection = @import("collection.zig");
const Collection = collection.Collection;
const makeCollection = collection.makeCollection;
const buildIndex = collection.buildIndex;
const search = collection.search;
const bruteForce = collection.bruteForce;
const quantize = collection.quantize;
const quantized = @import("quantized.zig");
const quant_mod = @import("../quant/quant.zig");
const heap = @import("../index/heap.zig");
const hnsw = @import("../index/hnsw.zig");
const Candidate = heap.Candidate;

// =========================================================================
// §6.7, quantized two-stage search
// =========================================================================

/// What a search is being asked for.
///
/// The four values that describe the *request* travelled as four positional
/// arguments beside three describing the *worker*, in a nine-parameter
/// signature where `ef` and `limit` are both `usize` and adjacent. They are
/// one thing, so they are one type.
pub const Request = struct {
    query: []const f32,
    ef: usize,
    limit: usize,
    quant: QuantParams = .{},
    /// A payload filter, applied where the fp32 path applies it: stage 1's
    /// admission and the pending tail. Stage 2 only reranks stage 1's
    /// survivors, so it needs none.
    filter: ?hnsw.Index.Filter = null,
};

/// The per-worker scratch a search needs, which §6.3 requires be preallocated.
///
/// `api.Workspace` owns these and used to unbundle them at every call site so
/// the callee could rebundle them mentally. `core` cannot name `Workspace`
/// (that would invert the layering), so this is the shape `Workspace` hands
/// over.
pub const Scratch = struct {
    hnsw: *hnsw.Index.Scratch,
    quant: *quantized.QueryScratch,
    /// §6.7's codebook table is built per query and is the one allocation the
    /// quantized path still makes; it is here so the caller decides where it
    /// comes from.
    alloc: std.mem.Allocator,
};

/// §6.7's knobs, resolved from `SearchParams.quantization` per request.
pub const QuantParams = struct {
    /// `params.quantization.ignore`: search the fp32 representation even
    /// though a quantized store exists.
    ignore: bool = false,
    /// `params.quantization.rescore`. Absent means "collection default", which
    /// for us is on, §6.7 pairs binary with "oversampling + rescore" and
    /// silently skipping it would report a recall the configuration cannot
    /// actually deliver.
    rescore: bool = true,
    /// `params.quantization.oversampling`.
    oversampling: f64 = 1.0,
};

/// Two-stage search: traverse with the quantized representation, then rescore.
///
/// §6.7: "quantized search produces `limit × oversampling` candidates,
/// rescored with the full-precision (or next-tier) representation."
///
/// Read as a floor rather than as the count: with rescore on, stage 2 receives
/// `max(limit × oversampling, ef)` candidates, because the traversal has
/// already scored `ef` of them and discarding the surplus costs recall for
/// nothing saved in stage 1. `oversampling` still asks for more than the walk
/// kept. See the note at the stage-1 heap for what that was measured to be
/// worth at d=1536, and for the cost it trades against.
///
/// Returns the top `limit` by the *rescored* score, so the caller sees scores
/// on the same scale as an unquantized search, which is what makes §7.4's
/// "compare at equal recall" possible without a per-encoding correction.
pub fn searchQuantized(
    coll: *const Collection,
    req: Request,
    scratch: Scratch,
    out: *heap.TopK,
) !void {
    const query = req.query;
    const ef = req.ef;
    const limit = req.limit;
    const qp = req.quant;
    const hnsw_scratch = scratch.hnsw;
    const qscratch = scratch.quant;
    const alloc = scratch.alloc;

    // The quantized store is retired and replaced by a rebuild exactly as the
    // graph is, so this path takes the same guard.
    const guard = collection.SearchGuard.begin(coll);
    defer guard.end();

    switch (collection.choosePath(coll, .approximate, if (qp.ignore) .ignore else .use)) {
        // Without a graph there is nothing for the quantized traversal to
        // accelerate, and brute-forcing quantized codes then rescoring would be
        // strictly worse than brute-forcing fp32 directly.
        // `.approximate` above: this path never chooses the exact scan.
        .exact => unreachable,
        .brute => {
            collection.bruteForceRangeFiltered(coll, query, 0, @intCast(coll.id_space.count()), req.filter, out);
            return;
        },
        // No store, or the request asked to ignore it: the ordinary path.
        .graph => {
            collection.searchFiltered(coll, query, ef, .approximate, hnsw_scratch, req.filter, out);
            return;
        },
        .quantized => {},
    }

    const g = collection.publishedGraph(coll).?;
    // Loaded once, under the guard, and used for the whole query.
    // `choosePath` saw a store, and a store, once published, is only ever
    // retired (never freed) while a guard is held, so this cannot be null and
    // cannot dangle. Re-reading `coll.quant` later could observe a
    // replacement mid-query and score stage 1 against two different codebooks.
    const store = coll.quant.load(.seq_cst) orelse {
        // A `quantize` that unpublished between `choosePath` and here: fall
        // back to the fp32 path, which answers the same question.
        search(coll, query, ef, .approximate, hnsw_scratch, out);
        return;
    };
    // Points written since the last build are in neither the graph nor the
    // codes, so they are scanned in fp32 at the end exactly as `search` does.
    //
    // Leaving that out was not a slower answer, it was a wrong one. Since the
    // W11 fix an upsert keeps the graph and leaves the collection `.ready`
    // until the tail passes `rebuild_ratio`, so on a quantized collection every
    // non-exact query silently omitted up to 10% of the most recently written
    // points. The fp32 path scanned them; this one did not; nothing compared
    // the two, and no workload in §4 writes to a quantized collection.
    // One atomic load, as `collection.searchFiltered` reads it: the same
    // quantity, for the same tail boundary, spelled the same way.
    const covered: u32 = @intCast(@atomicLoad(usize, &g.count, .acquire));
    const pending = coll.id_space.count() > covered;

    // A pending tail forces the rescore on.
    //
    // The tail has no codes, so fp32 is the only representation it can be
    // scored in, and with `rescore = false` stage 1's scores are on the
    // quantized scale: for binary that is a popcount similarity, which shares
    // no scale with a dot product. Merging the two would produce a ranking in
    // which the tail either always wins or always loses. Rescoring puts every
    // candidate on the fp32 scale, which is the only way the merge means
    // anything, and it is paid only when a tail exists.
    const rescore = qp.rescore or pending;

    const q = try quantized.prepareQuery(store, coll.config.metric.kernel(), query, qscratch, alloc);

    // Stage 1: traverse using the quantized score.
    //
    // With rescore on, stage 2 sees everything the traversal already scored,
    // not `limit × oversampling` of it. §6.7 words the candidate count as
    // "limit × oversampling", which reads as the whole rule and is not: the
    // walk visits `ef` nodes and scores every one of them, so keeping `limit`
    // throws away work already paid for, and it throws it away *ranked by the
    // quantized score* — which is the ranking rescoring exists to correct.
    //
    // MEASURED, dbpedia-openai-100K (d=1536, cosine), recall@10 at
    // ef=128 against the fp64 oracle:
    //
    //     SQ8, 8 bits/dim, oversampling 1 (pool 10)     0.8062
    //     binary, 1 bit/dim, oversampling 4 (pool 40)   0.9077
    //
    // One bit per dimension beating eight by ten points, on the same graph and
    // the same collection, is not an encoding result: the pool is what differs.
    // At d=128 the quantized top-10 is already the true top-10 (SQ8 there is
    // 0.9624, and Qdrant 0.9627), so nothing was visibly lost and this went
    // unnoticed until a tier with 12x the dimensionality was run.
    //
    // VERIFIED by a bench6 sweep on the same corpus after the change
    // (`recall.py`, 2000 queries against the fp64 oracle), recall@10:
    //
    //     ef          32      64     128     256     512
    //     before  0.7619  0.7924  0.8062  0.8104  0.8145
    //     after   0.8873  0.9450  0.9757  0.9879  0.9954
    //
    // The shape is the confirmation, not the size: before, sixteen times the
    // search bought five points, because the pool was pinned at `limit`
    // whatever the walk found. After, recall tracks `ef` the way it should.
    // MRDE falls from 2.9e-3 to 6.8e-5 across that sweep.
    //
    // The cost is real and is a trade: stage 2 now rescores up to `ef` vectors
    // in fp32 rather than `limit`. It scales with the `ef` the caller chose,
    // who by asking for it has already accepted `ef` nodes of traversal.
    // `oversampling` keeps its meaning for asking for *more* than the walk kept.
    // The conformance harness's single-client rate — a smoke test, not W6 —
    // fell 40-51% at fixed `ef`. At *equal recall*, which is the comparison
    // §7.4 makes, it is not a cost at all: 0.8873 now comes at ef=32 and
    // 11,323/s, where the old pool could not reach 0.89 at any `ef` and topped
    // out at 0.8145 and 3,355/s. W6's throughput still has to be measured.
    const take = quant_mod.binary.oversampledLimit(limit, qp.oversampling);
    const asked = @max(take, limit);
    const stage1 = @min(if (rescore) @max(asked, ef) else asked, qscratch.candidates.len);

    const idx = hnsw.Index{ .graph = g, .scorer = .of(&q) };
    // Stage 1 builds its heap in the *caller's* result buffer, which is the
    // whole backing array rather than `storage[0..limit]`. So the caller's
    // buffer, not `stage1`, is what finally sizes the pool — and a buffer
    // smaller than `ef` silently walks decisions.md §5's `max(asked, ef)` back
    // toward the `limit`-sized pool that section says we do not use.
    //
    // Asserted on the realised size for that reason: the three predicates that
    // used to sit at the sizing line above were algebraic restatements of it
    // and could not fire for any input. This one can, and did — an in-tree
    // caller passed a 64-slot buffer at `ef` 128 and got a pool of 64, which
    // cost recall@10 at ef=256 0.072.
    //
    // In production it holds because `Workspace.max_limit` and `max_ef`
    // (handlers.zig) are equal. They are two independent constants, so raising
    // `max_limit` alone would trip this — which is the point of asserting it
    // here rather than trusting the coincidence.
    //
    // What this pool is and is not evidence for: decisions.md §5 keeps the
    // divergence from Qdrant's `limit`-sized pool because rescoring a walk
    // already taken costs nothing, and §7.4 then refuses the quantized rows a
    // ratio on *measured recall inequality* — the pool is the cause of that
    // inequality, not the rule. It is not evidence that Qdrant recalls worse:
    // findings 42 measured `--quantization-oversampling 2` closing the gap to
    // 0.9888 against 0.9891, so what its default costs is a default, not a
    // ceiling.
    const cand_storage = out.items;
    const pool = @min(stage1, cand_storage.len);
    if (rescore) std.debug.assert(pool >= @min(ef, stage1));
    var cand = heap.TopK.init(cand_storage, pool);
    var adm: collection.Admission = undefined;
    const filter = collection.admission(coll, req.filter, &adm);
    // The same bound the fp32 path applies, for the same reason and against the
    // same two failures: with live insertion the graph can hold nodes past this
    // query's `covered`, and those belong to its tail scan below, not to its
    // traversal. Without this the quantized path returns a point twice (it is
    // in both regions) or not at all (in neither), which the differential test
    // against the exact scan catches. Costs nothing when the graph holds no
    // more than `covered`, which is every query with the flag off.
    var bnd: collection.Bounded = undefined;
    const traversal_filter = collection.bounded(filter, covered, @atomicLoad(usize, &g.count, .acquire), &bnd);
    // Tombstones are dropped as candidates are admitted, so stage 1 yields
    // `stage1` *live* candidates rather than `stage1` minus however many
    // happened to be deleted, otherwise oversampling silently buys less than
    // it claims and the rescore stage has fewer rows to choose from.
    idx.searchFiltered(@max(ef, pool), hnsw_scratch, &cand, traversal_filter);
    const found = cand.finish();

    var n: usize = 0;
    for (found) |c| {
        if (n >= qscratch.candidates.len) break;
        qscratch.candidates[n] = c.id;
        n += 1;
    }

    // Stage 2: rescore. §5.3 predicts this term dominates at low `ef` for
    // binary, which is why it is measured separately rather than folded in.
    if (!rescore) {
        // `found` aliases `out.items` (stage 1 built its heap there), so the
        // survivors must be copied out before `out` is rewritten, pushing
        // while reading would overwrite entries not yet consumed.
        var m: usize = 0;
        for (found) |c| {
            if (m >= qscratch.rescored.len) break;
            qscratch.rescored[m] = .{ .id = c.id, .score = c.score };
            m += 1;
        }
        out.reset(limit);
        for (qscratch.rescored[0..m]) |c| out.push(.{ .id = c.id, .score = c.score });
        return;
    }

    // The rescore stage compares against *stored* vectors, so it goes through
    // the same converted probe as every other path. On an fp32 collection this
    // is the previous behaviour exactly; on a f16 or uint8 one it is the
    // difference between rescoring and reading a f16 row as if it were f32.
    var probe_buf: [collection.Probe.max_buffer]u8 align(collection.Probe.buffer_align) = undefined;
    const probe = collection.Probe.init(coll, query, &probe_buf);
    const rescorer = quant_mod.binary.Rescorer{
        .encoding = .fp32,
        .ctx = @ptrCast(&probe),
        .score = struct {
            fn f(ctx: *const anyopaque, _: []const f32, node: u32) f32 {
                const pr: *const collection.Probe = @ptrCast(@alignCast(ctx));
                return pr.scoreNode(node);
            }
        }.f,
    };
    const final = rescorer.run(query, qscratch.candidates[0..n], limit, qscratch.rescored);

    out.reset(limit);
    for (final) |f| out.push(.{ .id = f.id, .score = f.score });

    // Both sets are now fp32 similarities, so `out` can merge them.
    collection.scanPendingTailFiltered(coll, query, covered, req.filter, out);
}

test "§6.7 two-stage search: quantized traversal plus rescore matches fp32 closely" {
    const dim = 64;
    const n = 1500;
    var c = try makeCollection(dim, .dot, 2048);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0x977);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try quantize(&c, .scalar);

    var hs = try hnsw.Index.Scratch.init(testing.allocator, 2048, 512);
    defer hs.deinit(testing.allocator);
    var qs = try quantized.QueryScratch.init(testing.allocator, dim, 512);
    defer qs.deinit(testing.allocator);

    const k = 10;
    const queries = 60;
    var hits: usize = 0;
    for (0..queries) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);

        var tb: [k]Candidate = undefined;
        var truth = heap.TopK.init(&tb, k);
        bruteForce(&c, &q, &truth);
        const t = truth.finish();

        var gb: [128]Candidate = undefined;
        var got = heap.TopK.init(&gb, k);
        try searchQuantized(
            &c,
            .{ .query = &q, .ef = 128, .limit = k, .quant = .{ .oversampling = 4.0 } },
            .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator },
            &got,
        );
        for (got.finish()) |gc| {
            for (t) |tc| {
                if (tc.id == gc.id) {
                    hits += 1;
                    break;
                }
            }
        }
    }
    const recall = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(queries * k));
    // SQ8 traversal with 4x oversampling and fp32 rescore should land very
    // close to the unquantized result, that is the trade §6.7 is making.
    try testing.expect(recall > 0.9);
}

test "§6.7: with rescore on, a wide `ef` reaches the fp32 answer" {
    // findings 42's property, asserted as the *ceiling* rather than the slope.
    //
    // The slope does not discriminate: `searchFiltered(@max(ef, pool))` widens
    // the walk with `ef` whatever the pool is, so recall@10 climbs under a
    // `limit`-sized pool too. Mutating the sizing to Qdrant's and running the
    // first version of this test left it green at 291 -> 371 hits of 400,
    // against 294 -> 400 unmutated. What separates them is where they *stop*:
    // a pool that grows with `ef` reaches the exact answer, and one fixed at
    // `limit` tops out short of it -- 0.928 in that measurement.
    //
    // So: at a wide `ef` on a small corpus, stage 2 rescores enough of the
    // walk to return the fp32 top-10 exactly. That is the assertion a
    // `limit`-sized pool fails.
    const dim = 64;
    const n = 2000;
    var c = try makeCollection(dim, .dot, 2560);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x42f2);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try quantize(&c, .scalar);

    var hs = try hnsw.Index.Scratch.init(testing.allocator, 2560, 1024);
    defer hs.deinit(testing.allocator);
    var qs = try quantized.QueryScratch.init(testing.allocator, dim, 1024);
    defer qs.deinit(testing.allocator);

    const k = 10;
    const queries = 40;
    const ef = 256;
    var hits: usize = 0;
    for (0..queries) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);

        var tb: [k]Candidate = undefined;
        var truth = heap.TopK.init(&tb, k);
        bruteForce(&c, &q, &truth);
        const want = truth.finish();

        // Sized to `ef`: stage 1 builds its heap in this buffer, so anything
        // smaller caps the pool below `ef` and measures the mutation instead
        // of the engine.
        var sb: [ef]Candidate = undefined;
        var got_h = heap.TopK.init(&sb, k);
        try searchQuantized(
            &c,
            .{ .query = &q, .ef = ef, .limit = k },
            .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator },
            &got_h,
        );
        for (want) |w| {
            for (got_h.finish()) |g| {
                if (g.id == w.id) {
                    hits += 1;
                    break;
                }
            }
        }
    }
    // Exact, not "better than the narrow arm". A `limit`-sized pool cannot
    // reach this, which is the whole point of the assertion.
    try testing.expectEqual(queries * k, hits);
}

test "§6.7: with rescore on, oversampling below ef changes nothing" {
    // The invariant the stage-1 sizing buys: stage 2 rescores what the walk
    // scored, so asking for a pool smaller than `ef` cannot make the answer
    // worse. Before that, oversampling 1 rescored 10 of the 128 nodes visited
    // and oversampling 4 rescored 40 of them, and the two disagreed — which at
    // d=1536 was worth ten points of recall@10.
    const dim = 256;
    const n = 1500;
    var c = try makeCollection(dim, .dot, 2048);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x5c8a);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try quantize(&c, .scalar);

    var hs = try hnsw.Index.Scratch.init(testing.allocator, 2048, 512);
    defer hs.deinit(testing.allocator);
    var qs = try quantized.QueryScratch.init(testing.allocator, dim, 512);
    defer qs.deinit(testing.allocator);

    const k = 10;
    const ef = 128;
    for (0..25) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);

        var ab: [ef]Candidate = undefined;
        var none = heap.TopK.init(&ab, k);
        try searchQuantized(
            &c,
            .{ .query = &q, .ef = ef, .limit = k, .quant = .{ .oversampling = 1.0 } },
            .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator },
            &none,
        );

        var bb: [ef]Candidate = undefined;
        var over = heap.TopK.init(&bb, k);
        try searchQuantized(
            &c,
            .{ .query = &q, .ef = ef, .limit = k, .quant = .{ .oversampling = 4.0 } },
            .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator },
            &over,
        );

        const a = none.finish();
        const b = over.finish();
        try testing.expectEqual(a.len, b.len);
        for (a, b) |x, y| try testing.expectEqual(x.id, y.id);
    }
}

test "§6.7: quantization.ignore falls back to the fp32 path" {
    const dim = 32;
    const n = 400;
    var c = try makeCollection(dim, .dot, 512);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x1a1a);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try quantize(&c, .binary);

    var hs = try hnsw.Index.Scratch.init(testing.allocator, 512, 256);
    defer hs.deinit(testing.allocator);
    var qs = try quantized.QueryScratch.init(testing.allocator, dim, 256);
    defer qs.deinit(testing.allocator);

    var q: [dim]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    var ab: [32]Candidate = undefined;
    var ignored = heap.TopK.init(&ab, 10);
    try searchQuantized(&c, .{ .query = &q, .ef = 128, .limit = 10, .quant = .{ .ignore = true } }, .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator }, &ignored);

    var bb: [32]Candidate = undefined;
    var plain = heap.TopK.init(&bb, 10);
    search(&c, &q, 128, .approximate, &hs, &plain);

    const a = ignored.finish();
    const b = plain.finish();
    try testing.expectEqual(b.len, a.len);
    for (a, b) |x, y| try testing.expectEqual(y.id, x.id);
}

test "points written after the build are found on the quantized path too" {
    // The W11 fix keeps the graph across writes and scans whatever is past
    // `graph.count`. That scan lived only in `search`, so this exact sequence -
    // build, quantize, write, query - dropped the tail on the quantized path
    // and returned a result set missing the best answer. No §4 workload writes
    // to a quantized collection, so nothing measured it.
    for ([_]quant_mod.Mode{ .scalar, .binary, .{ .product = .x16 } }) |mode| {
        const dim = 32;
        const indexed = 600;
        var c = try makeCollection(dim, .dot, 1024);
        defer c.deinit();

        var prng = std.Random.DefaultPrng.init(0x7a11);
        const rnd = prng.random();
        for (0..indexed) |i| {
            var v: [dim]f32 = undefined;
            for (&v) |*x| x.* = rnd.floatNorm(f32) * 0.1;
            _ = try c.upsert(.{ .num = i }, &v);
        }
        try buildIndex(&c, .serial, 1);
        try quantize(&c, mode);

        // A tail well under `rebuild_ratio`, so the collection stays `.ready`
        // and the quantized traversal is still the chosen path.
        for (0..20) |i| {
            var v: [dim]f32 = undefined;
            for (&v) |*x| x.* = rnd.floatNorm(f32) * 0.1;
            _ = try c.upsert(.{ .num = indexed + i }, &v);
        }
        var target: [dim]f32 = @splat(1.0);
        const answer = try c.upsert(.{ .num = 9999 }, &target);
        collection.invalidateIndex(&c);
        try testing.expectEqual(collection.IndexState.ready, c.index_state.load(.acquire));
        try testing.expectEqual(collection.SearchPath.quantized, collection.choosePath(&c, .approximate, .use));

        var hs = try hnsw.Index.Scratch.init(testing.allocator, 1024, 256);
        defer hs.deinit(testing.allocator);
        var qs = try quantized.QueryScratch.init(testing.allocator, dim, 256);
        defer qs.deinit(testing.allocator);

        // The target is its own nearest neighbour by a wide margin, and it is
        // in the tail, so it must come back first whatever the encoding.
        var gb: [64]Candidate = undefined;
        var got = heap.TopK.init(&gb, 10);
        try searchQuantized(&c, .{ .query = &target, .ef = 64, .limit = 10, .quant = .{} }, .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator }, &got);
        const ranked = got.finish();
        try testing.expect(ranked.len > 0);
        try testing.expectEqual(answer, ranked[0].id);

        // And with rescore off: the tail forces it back on rather than merging
        // two scales, so the answer still surfaces.
        var nb: [64]Candidate = undefined;
        var no_rescore = heap.TopK.init(&nb, 10);
        try searchQuantized(&c, .{ .query = &target, .ef = 64, .limit = 10, .quant = .{ .rescore = false } }, .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator }, &no_rescore);
        const nr = no_rescore.finish();
        try testing.expect(nr.len > 0);
        try testing.expectEqual(answer, nr[0].id);
    }
}

test "§6.7: the (oversampling, rescore) frontier for binary quantization" {
    // §6.7 asks for "the **combined** curve of (recall, latency) across
    // `(ef, oversampling, rescore-encoding)` rather than tuning stages
    // independently". This is that curve's recall axis, measured rather than
    // assumed, and the numbers are the point.
    //
    // MEASURED on 1200 random normal vectors at d=64, ef=128, recall@10:
    //
    //   oversampling |  no rescore | with fp32 rescore
    //   -------------|-------------|------------------
    //             1x |       0.173 |             0.173
    //             4x |       0.173 |             0.410
    //            16x |       0.173 |             0.768
    //            64x |       0.178 |             1.000
    //
    // Three things follow, all of which matter for how results get reported:
    //
    //  1. **Oversampling without rescore does nothing.** The no-rescore column
    //     is flat. Oversampling widens the candidate set, but if the final
    //     ranking is still by binary score then the same ten codes win however
    //     many were considered. The two knobs are not independent, one is
    //     inert without the other, which is exactly why §6.7 insists on "the
    //     **combined** curve ... rather than tuning stages independently".
    //
    //  2. **Rescore is not polish, it is the entire recall story.** At 64x it
    //     is the difference between 0.18 and 1.00. §5.3 prices the rescore
    //     stage at roughly the cost of the whole traversal for binary; these
    //     numbers say that is a trade worth making, and that a "binary
    //     quantization is Nx faster" claim measured without rescore is
    //     describing a configuration nobody should run.
    //
    //  3. **Binary needs large oversampling on data like this.** §4.1 warns
    //     that isotropic noise is unrepresentative and 1 bit/dim is where that
    //     bites hardest: the sign vector discards magnitude entirely, and in
    //     random high-dimensional data magnitude is most of what separates near
    //     neighbours. Real embeddings compress far better; §4.2's dataset tiers
    //     exist so the headline numbers are not measured here.
    const dim = 64;
    const n = 1200;
    var c = try makeCollection(dim, .dot, 2048);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x2b2b);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try quantize(&c, .binary);

    var hs = try hnsw.Index.Scratch.init(testing.allocator, 2048, 2048);
    defer hs.deinit(testing.allocator);
    var qs = try quantized.QueryScratch.init(testing.allocator, dim, 2048);
    defer qs.deinit(testing.allocator);

    const k = 10;
    const queries = 40;

    // Fixed query set, so every cell of the sweep sees identical work.
    const qbuf = try testing.allocator.alloc(f32, queries * dim);
    defer testing.allocator.free(qbuf);
    for (qbuf) |*x| x.* = rnd.floatNorm(f32);

    var last_with: f64 = -1.0;
    var last_without: f64 = -1.0;
    var best_with: f64 = 0;

    for ([_]f64{ 1.0, 4.0, 16.0, 64.0 }) |ov| {
        var hits_with: usize = 0;
        var hits_without: usize = 0;

        for (0..queries) |qi| {
            const q = qbuf[qi * dim ..][0..dim];

            var tb: [k]Candidate = undefined;
            var truth = heap.TopK.init(&tb, k);
            bruteForce(&c, q, &truth);
            const t = truth.finish();

            inline for ([_]bool{ true, false }) |do_rescore| {
                var gb: [1024]Candidate = undefined;
                var got = heap.TopK.init(&gb, k);
                try searchQuantized(
                    &c,
                    .{ .query = q, .ef = 128, .limit = k, .quant = .{
                        .rescore = do_rescore,
                        .oversampling = ov,
                    } },
                    .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator },
                    &got,
                );
                for (got.finish()) |gc| {
                    for (t) |tc| {
                        if (tc.id == gc.id) {
                            if (do_rescore) hits_with += 1 else hits_without += 1;
                            break;
                        }
                    }
                }
            }
        }

        const with = @as(f64, @floatFromInt(hits_with)) / @as(f64, @floatFromInt(queries * k));
        const without = @as(f64, @floatFromInt(hits_without)) / @as(f64, @floatFromInt(queries * k));

        // §8.6's "quantization dominance": recall is monotonically
        // non-decreasing in oversampling, for both settings.
        try testing.expect(with >= last_with - 1e-9);
        try testing.expect(without >= last_without - 1e-9);
        // Rescore is never worse, and in practice is much better.
        try testing.expect(with >= without);

        last_with = with;
        last_without = without;
        best_with = @max(best_with, with);
    }

    // At the top of the sweep, binary + rescore recovers most of the true
    // top-10, the configuration §6.7 actually recommends.
    try testing.expect(best_with > 0.85);
}

// =========================================================================
// Rebuilds racing writes and reads
// =========================================================================

test "rebuilds racing overwrites and searches never touch a freed store, and end consistent" {
    // Three threads on one quantized collection: one rebuilds in a loop, one
    // overwrites indexed rows in a loop, one runs quantized searches in a
    // loop. Two things used to go wrong here. `noteOverwrite` loaded the
    // store without `SearchGuard` and encoded into it while the build thread
    // swapped and reclaimed it, a write into freed memory (the codes are a
    // large allocation, so the test allocator unmaps them and that write is
    // a fault). And an overwrite that landed after the builder read its row
    // was encoded into the store about to be retired, so the published store
    // described the old vector.
    //
    // The end-state check is what pins the second: once every thread has
    // stopped, every indexed row's codes must equal the encoding of the
    // vector the arena holds now, whichever of build, log-replay or
    // `noteOverwrite` was responsible for writing them.
    const dim = 16;
    const n = 500;
    var c = try makeCollection(dim, .dot, 2048);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x5afe);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    c.quant_mode = .scalar;
    try buildIndex(&c, .serial, 1);

    const Shared = struct {
        coll: *Collection,
        stop: std.atomic.Value(bool) = .init(false),
        builds: usize = 0,
        writes: usize = 0,
        searches: usize = 0,
        failed: std.atomic.Value(bool) = .init(false),
    };
    var sh = Shared{ .coll = &c };

    const Builder = struct {
        fn run(s: *Shared) void {
            for (0..25) |_| {
                buildIndex(s.coll, .serial, 1) catch {
                    s.failed.store(true, .release);
                    break;
                };
                s.builds += 1;
            }
            s.stop.store(true, .release);
        }
    };
    const Writer = struct {
        fn run(s: *Shared) void {
            var p = std.Random.DefaultPrng.init(0x11);
            const r = p.random();
            while (!s.stop.load(.acquire)) {
                var v: [dim]f32 = undefined;
                for (&v) |*x| x.* = r.floatNorm(f32);
                _ = s.coll.upsert(.{ .num = r.uintLessThan(usize, n) }, &v) catch {
                    s.failed.store(true, .release);
                    return;
                };
                s.writes += 1;
            }
        }
    };
    const Searcher = struct {
        fn run(s: *Shared) void {
            var hs = hnsw.Index.Scratch.init(testing.allocator, 2048, 128) catch return;
            defer hs.deinit(testing.allocator);
            var qs = quantized.QueryScratch.init(testing.allocator, dim, 128) catch return;
            defer qs.deinit(testing.allocator);
            var p = std.Random.DefaultPrng.init(0x22);
            const r = p.random();
            while (!s.stop.load(.acquire)) {
                var q: [dim]f32 = undefined;
                for (&q) |*x| x.* = r.floatNorm(f32);
                var gb: [64]Candidate = undefined;
                var got = heap.TopK.init(&gb, 10);
                searchQuantized(
                    s.coll,
                    .{ .query = &q, .ef = 64, .limit = 10, .quant = .{ .oversampling = 2.0 } },
                    .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator },
                    &got,
                ) catch {
                    s.failed.store(true, .release);
                    return;
                };
                if (got.len != 10) s.failed.store(true, .release);
                s.searches += 1;
            }
        }
    };

    const tb = try std.Thread.spawn(.{}, Builder.run, .{&sh});
    const tw = try std.Thread.spawn(.{}, Writer.run, .{&sh});
    const ts = try std.Thread.spawn(.{}, Searcher.run, .{&sh});
    tb.join();
    tw.join();
    ts.join();

    try testing.expect(!sh.failed.load(.acquire));
    try testing.expectEqual(@as(usize, 25), sh.builds);
    try testing.expect(sh.writes > 0);
    try testing.expect(sh.searches > 0);

    // End state: codes are the codes of what is stored, for every indexed row.
    const store = c.quant.load(.acquire).?;
    var widen: [dim]f32 = undefined;
    var want: [dim]u8 = undefined;
    const covered = c.graph_count.load(.acquire);
    try testing.expectEqual(@as(usize, n), covered);
    for (0..covered) |off| {
        const row = c.space.readInto(@intCast(off), &widen);
        quant_mod.scalar.encode(store.scalar.params, &want, row);
        try testing.expectEqualSlices(u8, &want, store.scalar.row(@intCast(off)));
    }
    // Nothing is left mid-flight: the log is closed and empty, and whatever
    // was retired under a reader is reclaimable now that none is running.
    try testing.expect(!c.overwrite_log_active);
    try testing.expectEqual(@as(usize, 0), c.overwrite_log.count());
    _ = collection.reclaimRetired(&c);
    try testing.expectEqual(@as(usize, 0), c.retired_quant.items.len);
    try testing.expectEqual(@as(usize, 0), c.retired_graphs.items.len);

    // And a search now agrees with the exact scan on the moved points: the
    // published store and graph describe the collection as it is.
    var hs = try hnsw.Index.Scratch.init(testing.allocator, 2048, 128);
    defer hs.deinit(testing.allocator);
    var qs = try quantized.QueryScratch.init(testing.allocator, dim, 128);
    defer qs.deinit(testing.allocator);
    var hits: usize = 0;
    const queries = 40;
    for (0..queries) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);
        var tbuf: [10]Candidate = undefined;
        var truth = heap.TopK.init(&tbuf, 10);
        bruteForce(&c, &q, &truth);
        // Sized to `ef`, not to `limit`: stage 1 builds its heap in this
        // buffer, so a 64-slot array at `ef` 128 halves the rescore pool and
        // measures something other than what this test is about.
        var gb: [128]Candidate = undefined;
        var got = heap.TopK.init(&gb, 10);
        try searchQuantized(&c, .{ .query = &q, .ef = 128, .limit = 10, .quant = .{ .oversampling = 4.0 } }, .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator }, &got);
        for (got.finish()) |g| {
            for (truth.finish()) |t| {
                if (t.id == g.id) {
                    hits += 1;
                    break;
                }
            }
        }
    }
    const recall = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(queries * 10));
    try testing.expect(recall > 0.85);
}

// =========================================================================
// Randomised differential test: approximate against exact
// =========================================================================

// The invariants every approximate answer must satisfy against the exact
// scan, on a collection that has been deleted from, overwritten and appended
// to since its build. Ran across seeds, metrics, and fp32 / SQ8.
//
//  1. No tombstoned point is returned.
//  2. The page is full: `min(k, live)` results, however many of the true
//     nearest are deleted.
//  3. A point in the pending tail that is among the exact top-k is in the
//     result: the tail is scanned exhaustively, so this is not a matter of
//     recall.
//  4. Recall against the exact scan is high in aggregate.
test "randomised differential: approximate search against the exact scan under mutation" {
    const k = 10;
    var seed: u64 = 0;
    var total_hits: usize = 0;
    var total_wanted: usize = 0;
    while (seed < 10) : (seed += 1) {
        var prng = std.Random.DefaultPrng.init(0xd1ff_0000 + seed);
        const rnd = prng.random();
        const dim: usize = 16 + 8 * rnd.uintLessThan(usize, 3); // 16, 24, 32
        const n: usize = 300 + rnd.uintLessThan(usize, 500);
        const metric: collection.Metric = ([_]collection.Metric{ .dot, .euclid, .cosine })[rnd.uintLessThan(usize, 3)];
        const mode: quant_mod.Mode = if (seed % 2 == 0) .none else .scalar;

        var c = try collection.Collection.init(testing.allocator, "diff", .{
            .dim = dim,
            .metric = metric,
            .capacity = 1024,
        });
        defer c.deinit();
        var vec: [32]f32 = undefined;
        for (0..n) |i| {
            for (vec[0..dim]) |*x| x.* = rnd.floatNorm(f32);
            _ = try c.upsert(.{ .num = i }, vec[0..dim]);
        }
        c.quant_mode = mode;
        try buildIndex(&c, .serial, 1);

        // Mutate: ~10% deleted, ~5% overwritten in place, a tail under the
        // rebuild ratio appended.
        for (0..n / 10) |_| _ = c.delete(.{ .num = rnd.uintLessThan(usize, n) });
        for (0..n / 20) |_| {
            for (vec[0..dim]) |*x| x.* = rnd.floatNorm(f32);
            _ = try c.upsert(.{ .num = rnd.uintLessThan(usize, n) }, vec[0..dim]);
        }
        const tail = n / 20;
        for (0..tail) |i| {
            for (vec[0..dim]) |*x| x.* = rnd.floatNorm(f32);
            _ = try c.upsert(.{ .num = n + i }, vec[0..dim]);
        }
        try testing.expectEqual(collection.IndexState.ready, c.index_state.load(.acquire));
        const covered: u32 = @intCast(c.graph_count.load(.acquire));
        // Either the appended tail is pending (the default), or `-Dlive-insert`
        // put it in the graph and there is no tail. Both are correct states and
        // this test is about the *answers*, which must match the exact scan
        // either way, so it asserts the state it is in rather than one of them.
        try testing.expect(covered == n or covered == n + tail);

        var hs = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
        defer hs.deinit(testing.allocator);
        var qs = try quantized.QueryScratch.init(testing.allocator, dim, 128);
        defer qs.deinit(testing.allocator);

        var scratch: [32]f32 = undefined;
        for (0..20) |_| {
            var raw: [32]f32 = undefined;
            for (raw[0..dim]) |*x| x.* = rnd.floatNorm(f32);
            const q = try c.prepareQuery(&scratch, raw[0..dim]);

            var tb: [k]Candidate = undefined;
            var truth = heap.TopK.init(&tb, k);
            bruteForce(&c, q, &truth);
            const t = truth.finish();

            var gb: [128]Candidate = undefined;
            var got = heap.TopK.init(&gb, k);
            if (mode == .none) {
                search(&c, q, 128, .approximate, &hs, &got);
            } else {
                try searchQuantized(&c, .{ .query = q, .ef = 128, .limit = k, .quant = .{ .oversampling = 4.0 } }, .{ .hnsw = &hs, .quant = &qs, .alloc = testing.allocator }, &got);
            }
            const g = got.finish();

            // 1, 2.
            try testing.expectEqual(@min(k, c.count()), g.len);
            for (g) |cand| try testing.expect(!c.deleted.isSet(cand.id));
            // 3, 4.
            for (t) |tc| {
                var found = false;
                for (g) |gc| {
                    if (gc.id == tc.id) {
                        found = true;
                        break;
                    }
                }
                if (found) total_hits += 1;
                if (tc.id >= covered) try testing.expect(found);
            }
            total_wanted += t.len;
        }
    }
    const recall = @as(f64, @floatFromInt(total_hits)) / @as(f64, @floatFromInt(total_wanted));
    try testing.expect(recall > 0.9);
}
