//! §6.5, HNSW.
//!
//! ## Layout, which is where a from-scratch implementation differs most
//!
//! §6.5: "**Level 0**: a single flat array, stride `M0 = 2×M` `u32`s, indexed
//! by internal offset. For `M=16`: 128 B/node = exactly two cache lines, 128 MB
//! for 1M points. No per-node allocation, no pointer chasing, no indirection.
//! **Upper levels**: CSR (offsets + neighbours), tiny (~1/M of level 0), and
//! hot."
//!
//! The flat level-0 array is the single most consequential choice here. An
//! incrementally-grown implementation stores each node's neighbour list
//! separately, so expanding a node costs a pointer dereference *before* the
//! neighbour ids are even known, a dependent miss that must complete before
//! the M independent vector fetches can be issued. A flat array computes the
//! address, which means the neighbour ids arrive in one or two cache lines and
//! the vector prefetches can be issued immediately.
//!
//! ## Build
//!
//! §6.5: "Bulk, post-ingest, all cores. Triggered when a collection transitions
//! out of the ingest phase (status `Yellow` → build → `Green`, which is exactly
//! what bfb's poll loop wants)." §2 explains why this is legitimate rather than
//! a shortcut: bfb polls `collection_info` and waits for Green, so there is no
//! requirement to index concurrently with ingest.
//!
//! ## Determinism
//!
//! §8.7 is the hard constraint on the build:
//!
//!   "**Reproducible graph builds.** This is the hard one: a lock-based
//!    parallel build with neighbour lists mutated in arrival order produces a
//!    thread-count-dependent graph. Options, in preference order: (a)
//!    deterministic build order with parallelism only inside each insertion's
//!    candidate search, (b) accept nondeterminism but checksum the resulting
//!    graph and require the checksum to be stable for a given (seed, thread
//!    count, dataset), (c) a slow deterministic build mode used only for
//!    conformance. Pick (b) as the default with (c) available."
//!
//! Both (b) and (c) are implemented. `buildSerial` is (c): insertion in
//! ascending offset order, single-threaded, bit-reproducible for a given seed
//! regardless of anything else. `buildParallel` is (b): work is partitioned by
//! node and neighbour lists are mutated under per-node locks, so the graph
//! depends on thread count, and `checksum` exists precisely so that dependence
//! can be asserted stable rather than hoped for.

const std = @import("std");
const heap = @import("heap.zig");
const visited = @import("visited.zig");

pub const Candidate = heap.Candidate;

/// Neighbour slot meaning "empty". Internal offsets are dense from 0, so
/// `maxInt` can never collide with a real one.
pub const empty_neighbour: u32 = std.math.maxInt(u32);

pub const Params = struct {
    /// Neighbours per node on upper levels.
    m: usize = 16,
    /// Neighbours per node on level 0. §6.5: "stride `M0 = 2×M`".
    m0: usize = 32,
    ef_construct: usize = 100,
    /// §8.7: "Seeded RNG for HNSW level assignment, with the seed recorded in
    /// `meta.json`."
    seed: u64 = 0x517a_7e6d,

    pub fn fromM(m: usize, ef_construct: usize, seed: u64) Params {
        return .{ .m = m, .m0 = 2 * m, .ef_construct = ef_construct, .seed = seed };
    }

    /// §6.5: "Level assignment with the standard exponential decay,
    /// `mL = 1/ln(M)`."
    pub fn levelMultiplier(self: Params) f64 {
        return 1.0 / @log(@as(f64, @floatFromInt(self.m)));
    }
};

/// The graph.
pub const Graph = struct {
    alloc: std.mem.Allocator,
    params: Params,
    capacity: usize,

    /// §6.5: level 0 as one flat array, `capacity * m0` u32s.
    level0: []u32,

    /// Upper levels in CSR form. `upper_offsets[node]` indexes into
    /// `upper_neighbours`; a node's level-`l` list starts at
    /// `upper_offsets[node] + (l - 1) * m`.
    upper_neighbours: []u32,
    upper_offsets: []u32,

    /// Highest level each node participates in. 0 means level 0 only.
    node_levels: []u8,

    /// Optional stable key per node for the level draw (`assignLevelKey`).
    ///
    /// Null keeps the node id as the key, which is what every graph built
    /// before this existed used. Set, it makes the level assignment a property
    /// of the *point* rather than of the offset it happened to be given, which
    /// is what stops two uploads of one corpus producing two different graphs.
    /// Borrowed, not owned: it has to outlive the build, and nothing reads it
    /// afterwards.
    level_keys: ?[]const u64 = null,
    /// Set while a build fills this graph, so the collection it is for can
    /// stop it: a dropped collection's build is work nothing will read, and
    /// the drop waits for it. Borrowed; null for every graph built elsewhere.
    cancel: ?*const std.atomic.Value(bool) = null,

    /// Optional order to *insert* nodes in, as node ids: `insert_order[k]` is
    /// the node the build links k-th. Null inserts 0, 1, 2, ...
    ///
    /// Which is arrival order, because `ids.IdSpace.reserve` hands out offsets
    /// as points arrive. `decisions.md` measures what that costs: eight
    /// concurrent upload streams link the corpus in a different sequence every
    /// pass, which is the whole of findings 34's per-build recall draw, and it
    /// is the *sequence* rather than the level assignment that does it. An
    /// order derived from something stable (the external id) is what puts
    /// every pass back in one regime.
    ///
    /// Borrowed, not owned, and covers `[from, count)` at its own indices.
    insert_order: ?[]const u32 = null,

    entry_point: u32 = empty_neighbour,
    max_level: u8 = 0,
    count: usize = 0,

    /// Take `other`'s nodes, edges and entry point as this graph's starting
    /// state, so a build can insert only what `other` does not already have.
    ///
    /// Copies exactly the prefix `other` covers and nothing beyond it: the
    /// remaining slots keep the `empty_neighbour` fill from `init`, which is
    /// what the extension writes into. `upper_offsets` carries one extra entry
    /// — the CSR cursor at `other.count` — because that is where the new
    /// nodes' levels start being laid out.
    ///
    /// Both graphs must have been made with the same `params` and `capacity`;
    /// the caller checks, because at this level the arrays would simply be the
    /// wrong length and the copy would be a silent corruption rather than an
    /// error.
    /// Whether the build filling this graph has been asked to stop.
    pub fn cancelled(self: *const Graph) bool {
        return if (self.cancel) |c| c.load(.acquire) else false;
    }

    pub fn copyFrom(self: *Graph, other: *const Graph) void {
        std.debug.assert(self.capacity == other.capacity);
        std.debug.assert(self.params.m0 == other.params.m0);
        const n = other.count;
        @memcpy(self.level0[0 .. n * self.params.m0], other.level0[0 .. n * self.params.m0]);
        @memcpy(self.node_levels[0..n], other.node_levels[0..n]);
        @memcpy(self.upper_offsets[0 .. n + 1], other.upper_offsets[0 .. n + 1]);
        const used = other.upper_offsets[n];
        @memcpy(self.upper_neighbours[0..used], other.upper_neighbours[0..used]);
        self.entry_point = other.entry_point;
        self.max_level = other.max_level;
        self.count = n;
    }

    pub fn init(alloc: std.mem.Allocator, params: Params, capacity: usize) !Graph {
        const level0 = try alloc.alloc(u32, capacity * params.m0);
        errdefer alloc.free(level0);
        @memset(level0, empty_neighbour);

        const node_levels = try alloc.alloc(u8, capacity);
        errdefer alloc.free(node_levels);
        @memset(node_levels, 0);

        const upper_offsets = try alloc.alloc(u32, capacity + 1);
        errdefer alloc.free(upper_offsets);
        @memset(upper_offsets, 0);

        // Upper levels hold ~1/M of level 0's edges in expectation. Allocating
        // for 1/4 of the nodes having one upper level is generous at M=16
        // (expected fraction is 1/16) and avoids a second pass to size it.
        const upper_cap = @max(params.m * 4, (capacity / 4 + 1) * params.m * 2);
        const upper_neighbours = try alloc.alloc(u32, upper_cap);
        @memset(upper_neighbours, empty_neighbour);

        return .{
            .alloc = alloc,
            .params = params,
            .capacity = capacity,
            .level0 = level0,
            .upper_neighbours = upper_neighbours,
            .upper_offsets = upper_offsets,
            .node_levels = node_levels,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.alloc.free(self.level0);
        self.alloc.free(self.upper_neighbours);
        self.alloc.free(self.upper_offsets);
        self.alloc.free(self.node_levels);
    }

    /// §6.5: "Point *n* lives at `base + n × stride`, computed, never looked up."
    pub fn level0Slice(self: *const Graph, node: u32) []u32 {
        std.debug.assert(node < self.capacity);
        const start = @as(usize, node) * self.params.m0;
        std.debug.assert(start + self.params.m0 <= self.level0.len);
        return self.level0[start..][0..self.params.m0];
    }

    pub fn upperSlice(self: *const Graph, node: u32, level: u8) []u32 {
        std.debug.assert(level >= 1);
        const base = self.upper_offsets[node];
        const start = @as(usize, base) + (@as(usize, level) - 1) * self.params.m;
        return self.upper_neighbours[start..][0..self.params.m];
    }

    pub fn neighbours(self: *const Graph, node: u32, level: u8) []u32 {
        return if (level == 0) self.level0Slice(node) else self.upperSlice(node, level);
    }

    /// A checksum of the entire graph structure.
    ///
    /// §8.7 option (b): "accept nondeterminism but checksum the resulting graph
    /// and require the checksum to be stable for a given (seed, thread count,
    /// dataset)". This is that checksum. It covers the neighbour lists, the
    /// level assignment and the entry point, everything a search result can
    /// depend on.
    pub fn checksum(self: *const Graph) u64 {
        var h: u64 = 0xcbf29ce484222325;
        const mix = struct {
            fn f(acc: *u64, v: u64) void {
                acc.* ^= v;
                acc.* *%= 0x100000001b3;
            }
        }.f;

        mix(&h, self.count);
        mix(&h, self.entry_point);
        mix(&h, self.max_level);
        for (0..self.count) |i| {
            const node: u32 = @intCast(i);
            mix(&h, self.node_levels[node]);
            var lvl: u8 = 0;
            while (lvl <= self.node_levels[node]) : (lvl += 1) {
                for (self.neighbours(node, lvl)) |n| mix(&h, n);
            }
        }
        return h;
    }
};

/// Deterministic level assignment.
///
/// Seeded per *node* rather than from a single stream, so the level a node gets
/// does not depend on the order nodes are processed in. A shared RNG stream
/// would make the parallel build's level assignment thread-order-dependent,
/// which would defeat §8.7's checksum stability before the neighbour lists even
/// came into it.
pub fn assignLevel(params: Params, node: u32) u8 {
    return levelFromHash(params, std.hash.Wyhash.hash(params.seed, std.mem.asBytes(&node)));
}

/// The same draw, keyed on something that survives a re-ingest.
///
/// The node id does not. `ids.IdSpace.reserve` hands out internal offsets from
/// a monotone counter as points arrive, so uploading one corpus twice over
/// eight concurrent streams gives the same vector two different node ids and
/// therefore two different levels.
///
/// An instrument, not a fix, and the distinction is measured. A reordered
/// arrival moves both the level assignment and the insertion sequence; this
/// separates them, and `decisions.md` records the answer: over five builds at
/// the same five arrival orders, drawing the level from the point instead of
/// from the slot left the recall spread at 0.00023 against 0.00024 and the
/// mean at 0.99883 against 0.99884. The level draw is not what makes two
/// uploads of one corpus two different graphs; the order the points were
/// linked in is. Nothing in the engine sets `Graph.level_keys`, and keying it
/// on the external id would buy nothing on this evidence.
///
/// Kept because the question recurs: anything that changes the insertion
/// sequence will want this column beside it again.
///
/// **Measured fragile; do not key the engine's levels on it without
/// re-measuring.** In file order at SIFT1M, four seeds of this draw span 0.99726
/// to 0.99958 of recall@10 at `ef` 512, where `assignLevel` over the same four
/// sits within 0.00003 (`graph-diff --seeds`, `decisions.md`, "Closed as a
/// property"). Why is not known.
pub fn assignLevelKey(params: Params, key: u64) u8 {
    return levelFromHash(params, std.hash.Wyhash.hash(params.seed, std.mem.asBytes(&key)));
}

fn levelFromHash(params: Params, h: u64) u8 {
    // Uniform in (0, 1].
    const u = @as(f64, @floatFromInt(h | 1)) / @as(f64, @floatFromInt(std.math.maxInt(u64)));
    const level = -@log(u) * params.levelMultiplier();
    return @intFromFloat(@min(level, @as(f64, level_cap)));
}

/// The highest level `assignLevel` hands out. At mL = 1/ln(16) the
/// probability of exceeding 15 is ~1e-18, and the cap keeps `upper_offsets`
/// arithmetic in a u32. `build.Builder` sizes its per-level selection scratch
/// from it.
pub const level_cap: u8 = 15;

/// The distance callback the index needs. Keeps the graph independent of where
/// vectors live, so the same code serves the in-memory collection and the
/// mmap-backed one of §6.4.
/// How a search scores one candidate.
///
/// No query parameter: the query belongs to `ctx`, which is prepared once per
/// search and holds it in whatever form the comparison needs. It used to be an
/// argument, and both production scorers ignored it (`fn score(ctx, _: []const
/// f32, node)`) because an fp32 query is the wrong thing to compare against a
/// f16 or uint8 collection. A parameter that every real implementation
/// discards is not an abstraction, it is an invitation to use it and score in
/// the wrong precision.
pub const ScoreFn = *const fn (ctx: *const anyopaque, node: u32) f32;

/// A prepared query bound to the function that scores with it.
///
/// The engine has two of these and they are not the same state: `core.Probe`
/// knows what a *stored row* looks like (f32, f16 or u8), `quantized.Query`
/// knows what an *encoding* looks like (SQ8, binary, PQ). Merging them would
/// make a union of two unrelated subsystems. What they genuinely share is this
/// binding, which every call site was spelling out by hand as a `@ptrCast`
/// beside a function name that had to match it. `build.Scorer` is the same
/// idea for the builder; search simply lacked the name.
pub const PrefetchFn = *const fn (ctx: *const anyopaque, node: u32) void;

pub const Scorer = struct {
    ctx: *const anyopaque,
    call: ScoreFn,
    /// Touch the first line of `node`'s stored row so it is on its way
    /// before `call` needs it. Optional: a scorer over a store that has no
    /// row to prefetch (or one too cheap to matter) leaves it null and the
    /// traversal skips the pass.
    prefetch: ?PrefetchFn = null,

    /// Bind a prepared query. `T` must expose
    /// `fn score(ctx: *const anyopaque, node: u32) f32`, and may expose
    /// `fn prefetch(ctx: *const anyopaque, node: u32) void`.
    ///
    /// The pointer and the function come from one place, so they cannot be
    /// paired wrongly: passing one type's context with another's scorer is a
    /// compile error rather than a cast that silently reinterprets memory.
    pub fn of(prepared: anytype) Scorer {
        const T = @typeInfo(@TypeOf(prepared)).pointer.child;
        return .{
            .ctx = @ptrCast(prepared),
            .call = T.score,
            .prefetch = if (@hasDecl(T, "prefetch")) T.prefetch else null,
        };
    }
};

pub const Index = struct {
    graph: *Graph,
    scorer: Scorer,

    /// Per-worker scratch. §6.3: "No allocation on the query path."
    pub const Scratch = struct {
        vis: visited.Selected,
        frontier: []Candidate,
        results: []Candidate,

        /// Sized by the point count (the visited set) and the traversal width
        /// (the heaps). It took an `m0` for a selection buffer the build never
        /// used, and every caller threaded a graph degree through to it.
        pub fn init(alloc: std.mem.Allocator, capacity: usize, max_ef: usize) !Scratch {
            // Built field by field with errdefers rather than as one struct
            // literal. In a literal, a `try` that fails on the third field
            // returns before the struct exists and the first two are already
            // allocated and unreachable: a leak whose only trigger is the
            // allocation failure it is trying to report.
            // `initSelected` rather than a type name: the two implementations
            // are constructed differently and the search path does not need to
            // know which `-Dvisited` gave it.
            var vis = try visited.initSelected(alloc, capacity);
            errdefer vis.deinit(alloc);
            const frontier = try alloc.alloc(Candidate, max_ef * 2);
            errdefer alloc.free(frontier);
            const results = try alloc.alloc(Candidate, max_ef);
            return .{
                .vis = vis,
                .frontier = frontier,
                .results = results,
            };
        }

        pub fn deinit(self: *Scratch, alloc: std.mem.Allocator) void {
            self.vis.deinit(alloc);
            alloc.free(self.frontier);
            alloc.free(self.results);
        }
    };

    /// Greedy descent through the upper levels: from `entry`, repeatedly move
    /// to the best neighbour until no neighbour improves.
    fn descend(self: *const Index, entry: u32, entry_score: f32, level: u8) Candidate {
        var best = Candidate{ .id = entry, .score = entry_score };
        var improved = true;
        while (improved) {
            improved = false;
            // §6.5's prefetch discipline is the scorer's: the arena is behind
            // the `score` callback, so the index cannot issue vector prefetches
            // itself (`Scorer.prefetch`); a named no-op stood here for it.
            const ns = self.graph.neighbours(best.id, level);
            for (ns) |n| {
                if (n == empty_neighbour) break;
                const s = self.scorer.call(self.scorer.ctx, n);
                const c = Candidate{ .id = n, .score = s };
                if (c.better(best)) {
                    best = c;
                    improved = true;
                }
            }
        }
        return best;
    }

    /// The core layer search: best-first traversal bounded by `ef`.
    ///
    /// Returns the `ef` best candidates found, in the scratch result heap.
    ///
    /// `filter` gates *admission to the result heap*, never expansion: a
    /// node it rejects is still scored, still pushed on the frontier and
    /// still expanded, so tombstones keep bridging the graph (§3 keeps them
    /// in it, uncompacted). What changes is that the heap fills with `ef`
    /// candidates the caller will keep, rather than `ef` candidates of which
    /// the caller then drops some. Filtering after the fact left the caller
    /// short: with `ef == limit` and `d` tombstones among the nearest, the
    /// client got `limit - d` results while live points existed.
    fn searchLayer(
        self: *const Index,
        entries: []const Candidate,
        level: u8,
        ef: usize,
        scratch: *Scratch,
        out: *heap.TopK,
        filter: ?Filter,
    ) void {
        // `ef == 0` would make `out.isFull()` true while empty, and
        // `peekWorst` reads an unwritten slot. Not reachable from the API
        // (the handler takes `max(hnsw_ef, limit)` with `limit >= 1`), so
        // this documents the invariant rather than defending against a
        // client.
        std.debug.assert(ef >= 1);
        var frontier = heap.Frontier.init(scratch.frontier);
        out.reset(ef);
        const hop = filter != null and filter.?.two_hop;

        for (entries) |e| {
            if (!scratch.vis.testAndSet(e.id)) continue;
            frontier.push(e);
            if (filter == null or filter.?.admits(e.id)) out.push(e);
        }

        while (frontier.pop()) |current| {
            // §6.5: "Early termination on the standard 'worst candidate is
            // worse than current k-th best' condition."
            if (out.isFull() and current.worse(out.peekWorst())) break;

            const ns = self.graph.neighbours(current.id, level);
            // And the *next* candidate's list, while this one's rows are being
            // fetched. The two-pass prefetch below covers the neighbours of the
            // node in hand; what it cannot cover is the load that starts the
            // next iteration, `neighbours(peek().id)`, which is a random row of
            // a 128 MB array and is the one link of the chain nothing hides.
            // At `-p 1` there is no other query on the core to overlap it with,
            // which is where W3 spends 972k cycles per query against W4's 650k
            // for the same work (profiled: `Probe.prefetch` is 17.1% of samples
            // at `-p 1` against 9.9% saturated).
            if (frontier.peek()) |next| {
                const row = self.graph.neighbours(next.id, level);
                if (row.len > 0) {
                    @prefetch(&row[0], .{ .rw = .read, .locality = 3, .cache = .data });
                }
            }
            // Two passes over the list, as hnswlib does: first ask for every
            // neighbour's visited stamp and the first line of its row, then
            // score. Each neighbour is a random node of the collection, so
            // both are cache misses, and issued one at a time from the
            // scoring loop they serialise: at d=4 on 1M points, where a
            // distance is four multiplies, `searchLayer` was 35% of the
            // server's CPU and the heaps another 20% (W0, perf).
            // The stamp is prefetched for write, since `testAndSet` writes it.
            if (self.scorer.prefetch) |pf| {
                for (ns) |n| {
                    if (n == empty_neighbour) break;
                    scratch.vis.prefetch(n);
                    pf(self.scorer.ctx, n);
                }
            }
            if (hop) {
                self.expandTwoHop(ns, level, scratch, &frontier, out, filter.?);
                continue;
            }
            for (ns) |n| {
                if (n == empty_neighbour) break;
                if (!scratch.vis.testAndSet(n)) continue;
                const s = self.scorer.call(self.scorer.ctx, n);
                const c = Candidate{ .id = n, .score = s };
                if (!out.isFull() or c.better(out.peekWorst())) {
                    frontier.push(c);
                    if (filter == null or filter.?.admits(n)) out.push(c);
                }
            }
        }
    }

    /// One expansion under a selective filter, ACORN-1 (Patel et al., SIGMOD
    /// 2024): an admitted neighbour is scored as usual; a rejected one is
    /// never scored, and its own neighbours stand in for it, admitted ones
    /// only. So the frontier holds only points the caller can keep and no
    /// distance is spent on one it cannot.
    ///
    /// The plain rule scores every neighbour and expands rejected ones too,
    /// which at 10% selectivity is nine distances in ten on points that can
    /// never be returned: W12-sel10's walk at `ef` 32 ran at 284 qps on
    /// dbpedia-openai-1m, and Qdrant, which does not score what its filter
    /// rejects, at 2,226.
    ///
    /// Each expansion scores at most the list's width (`m0` at level 0), as
    /// ACORN-1 truncates its compressed neighbourhood to `M`: the two-hop
    /// fan-out is `m0²`, and uncapped at a loose filter it would score far
    /// more per step than the plain walk does. A rejected neighbour is marked
    /// visited only once its hop is taken, so one skipped for the cap can
    /// still be hopped through from another node.
    fn expandTwoHop(
        self: *const Index,
        ns: []const u32,
        level: u8,
        scratch: *Scratch,
        frontier: *heap.Frontier,
        out: *heap.TopK,
        filter: Filter,
    ) void {
        // First pass: the row of each admitted neighbour, and the neighbour
        // list of each rejected one, which is what the second pass reads.
        if (self.scorer.prefetch) |pf| {
            for (ns) |n| {
                if (n == empty_neighbour) break;
                if (filter.admits(n)) {
                    scratch.vis.prefetch(n);
                    pf(self.scorer.ctx, n);
                } else {
                    const row = self.graph.neighbours(n, level);
                    if (row.len > 0) @prefetch(&row[0], .{ .rw = .read, .locality = 3, .cache = .data });
                }
            }
        }
        const budget = ns.len;
        var scored: usize = 0;
        for (ns) |n| {
            if (n == empty_neighbour) break;
            if (filter.admits(n)) {
                if (!scratch.vis.testAndSet(n)) continue;
                scored += 1;
                self.consider(n, frontier, out);
                continue;
            }
            if (scored >= budget) continue;
            if (!scratch.vis.testAndSet(n)) continue;
            for (self.graph.neighbours(n, level)) |n2| {
                if (n2 == empty_neighbour) break;
                if (scored >= budget) break;
                // Tested before the visited stamp: a rejected two-hop node is
                // not taken here, and stamping it would stop a later hop
                // through it.
                if (!filter.admits(n2)) continue;
                if (!scratch.vis.testAndSet(n2)) continue;
                scored += 1;
                self.consider(n2, frontier, out);
            }
        }
    }

    /// Score an admitted node and keep it if it can still matter.
    fn consider(self: *const Index, n: u32, frontier: *heap.Frontier, out: *heap.TopK) void {
        const c = Candidate{ .id = n, .score = self.scorer.call(self.scorer.ctx, n) };
        if (!out.isFull() or c.better(out.peekWorst())) {
            frontier.push(c);
            out.push(c);
        }
    }

    /// A predicate applied to *results*, never to the traversal.
    ///
    /// §3 keeps deletes as a tombstone bitmap with no compaction, so deleted
    /// points remain in the graph and remain load-bearing: skipping them while
    /// expanding would disconnect the traversal and collapse recall. They must
    /// therefore be dropped on the way out.
    ///
    /// Doing that *after* the caller's heap has already truncated to `k`
    /// returns short pages, the graph path returned 8 of a requested 10 at 30%
    /// tombstones while the exact path returned 10, so the two disagreed on
    /// result *count*, not merely on order. Over-fetching by the expected
    /// deleted fraction does not fix it either: the fraction among a query's
    /// nearest `k` is a small sample and its variance is what shows up as a
    /// short page. Filtering against the `ef`-sized candidate set was
    /// closer, and still short whenever `ef == k`; the predicate is now
    /// applied as candidates are *admitted* to that set (`searchLayer`), so
    /// it holds `ef` live points, and only then truncated to `k`.
    pub const Filter = struct {
        pred: *const fn (ctx: *const anyopaque, node: u32) bool,
        ctx: *const anyopaque,
        /// Traverse ACORN-1 style (`searchLayer`): a rejected node is not
        /// scored but hopped through to its neighbours. For a predicate that
        /// is a bit test; one that reads a payload blob would pay it for every
        /// two-hop neighbour, so it stays off there, and for tombstones.
        two_hop: bool = false,

        pub fn admits(self: Filter, node: u32) bool {
            return self.pred(self.ctx, node);
        }
    };

    pub fn search(self: *const Index, ef: usize, scratch: *Scratch, out: *heap.TopK) void {
        self.searchFiltered(ef, scratch, out, null);
    }

    /// The level-0 search alone, from `entry`, with no descent.
    ///
    /// An instrument, and nothing in the engine calls it. findings 34's seed
    /// loss is flat across `ef`, which says some true neighbours cannot be
    /// reached at any beam width, and there are two places that can happen:
    /// the descent through the upper levels lands somewhere the neighbours are
    /// not reachable from, or level 0 itself does not link them. Started at a
    /// query's true nearest neighbour, the first cannot happen, so the recall
    /// that comes back is the part the upper levels were costing
    /// (`bench/graphdiff`, `--oracle-entry`).
    pub fn searchFrom(self: *const Index, entry: u32, ef: usize, scratch: *Scratch, out: *heap.TopK) void {
        std.debug.assert(ef <= scratch.results.len);
        std.debug.assert(entry < self.graph.count);
        scratch.vis.beginQuery();
        var results = heap.TopK.init(scratch.results, ef);
        const entries = [_]Candidate{.{ .id = entry, .score = self.scorer.call(self.scorer.ctx, entry) }};
        self.searchLayer(&entries, 0, ef, scratch, &results, null);
        for (results.items[0..results.len]) |c| out.push(c);
    }

    pub fn searchFiltered(
        self: *const Index,
        ef: usize,
        scratch: *Scratch,
        out: *heap.TopK,
        filter: ?Filter,
    ) void {
        // §6.3 sizes the frontier and result heaps once at startup. An `ef`
        // past that is a client-controlled out-of-bounds write in ReleaseFast,
        // which is how an unbounded `hnsw_ef` got in once already.
        std.debug.assert(ef <= scratch.results.len);
        const g = self.graph;
        // Nothing to add: the contract is "merge into `out`" (the caller
        // follows this with the pending-tail scan), and this path used to be
        // the one exit that cleared the heap instead.
        if (g.count == 0 or g.entry_point == empty_neighbour) return;

        scratch.vis.beginQuery();

        // Descend the upper levels greedily, one node at a time.
        var entry = g.entry_point;
        var entry_score = self.scorer.call(self.scorer.ctx, entry);
        var level = g.max_level;
        while (level > 0) : (level -= 1) {
            const best = self.descend(entry, entry_score, level);
            entry = best.id;
            entry_score = best.score;
        }

        // Then a full ef-bounded search on level 0, admitting only what the
        // filter accepts.
        var results = heap.TopK.init(scratch.results, ef);
        const entries = [_]Candidate{.{ .id = entry, .score = entry_score }};
        self.searchLayer(&entries, 0, ef, scratch, &results, filter);

        // Copy into the caller's heap, which may want fewer than `ef`.
        for (results.items[0..results.len]) |c| out.push(c);
    }
};

/// §6.5: "Neighbour selection: heuristic pruning (Qdrant/hnswlib
/// `select_neighbors_heuristic`), since the plain top-M variant produces
/// measurably worse graphs on real data."
///
/// The heuristic keeps a candidate only if it is closer to the query node than
/// to every neighbour already selected. That prunes edges into clusters the
/// graph can already reach, which is what preserves long-range connectivity -
/// plain top-M produces a graph where every node's edges point into the same
/// dense region and the traversal cannot escape local minima.
///
/// `candidates` must be sorted best-first. Returns the number selected, written
/// into `out`.
pub fn selectNeighboursHeuristic(
    candidates: []const Candidate,
    m: usize,
    score_between: *const fn (ctx: *const anyopaque, a: u32, b: u32) f32,
    ctx: *const anyopaque,
    out: []u32,
) usize {
    return selectNeighbours(candidates, m, score_between, ctx, out, keep_pruned_default);
}

/// Algorithm 4's `keepPrunedConnections`, on by default.
///
/// The heuristic drops a candidate whenever an already-selected neighbour is
/// closer to it than the query is — good for diversity, and it typically fills
/// only about half of `m`. Those empty slots are not free: the discarded
/// candidates are the ones whose only route into the graph may have been this
/// edge, and dropping them costs *in-edges* rather than out-edges. A node that
/// nothing points at cannot be found at any `ef`.
///
/// Measured on 20,000 random d=128 points, m0=32, before this existed: mean
/// out-degree 17.66 of 32, 352 nodes with in-degree zero, and 327 nodes
/// unreachable from the entry point. At 50,000 points that was 1,759
/// unreachable — 3.5%, growing with n.
pub const keep_pruned_default = true;

pub fn selectNeighbours(
    candidates: []const Candidate,
    m: usize,
    score_between: *const fn (ctx: *const anyopaque, a: u32, b: u32) f32,
    ctx: *const anyopaque,
    out: []u32,
    keep_pruned: bool,
) usize {
    var n: usize = 0;
    for (candidates) |c| {
        if (n >= m) break;
        var keep = true;
        for (out[0..n]) |selected| {
            // Distance from the candidate to an already-selected neighbour.
            const to_selected = score_between(ctx, c.id, selected);
            // If the candidate is closer to a selected neighbour than to the
            // query node, the selected neighbour already covers that direction.
            if (to_selected > c.score) {
                keep = false;
                break;
            }
        }
        if (keep) {
            out[n] = c.id;
            n += 1;
        }
    }

    // Algorithm 4's `keepPrunedConnections`: top the list up from the
    // candidates the heuristic discarded, best first. `candidates` arrives
    // sorted, so a second pass in the same order takes the closest of them.
    if (keep_pruned and n < m) {
        for (candidates) |c| {
            if (n >= m) break;
            var already = false;
            for (out[0..n]) |sel| {
                if (sel == c.id) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                out[n] = c.id;
                n += 1;
            }
        }
    }
    return n;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "keepPrunedConnections fills the spare slots the heuristic left empty" {
    // Algorithm 4's `keepPrunedConnections`. The pruned candidate is not
    // discarded when there is room: it is exactly the node whose only route
    // into the graph may have been this edge, and an empty slot buys nothing.
    const Ctx = struct {
        sim: [3][3]f32,
        fn between(ctx: *const anyopaque, a: u32, b: u32) f32 {
            const self: *const @This() = @ptrCast(@alignCast(ctx));
            return self.sim[a][b];
        }
    };
    var ctx = Ctx{ .sim = .{
        .{ 0.0, 0.95, 0.1 },
        .{ 0.95, 0.0, 0.1 },
        .{ 0.1, 0.1, 0.0 },
    } };
    const cands = [_]Candidate{
        .{ .id = 0, .score = 0.9 },
        .{ .id = 1, .score = 0.85 }, // pruned by the heuristic
        .{ .id = 2, .score = 0.5 },
    };

    var out: [3]u32 = undefined;
    const kept = selectNeighbours(&cands, 3, Ctx.between, @ptrCast(&ctx), &out, true);
    try testing.expectEqual(@as(usize, 3), kept);
    // The heuristic's picks keep their order and precedence; the pruned one
    // lands after them rather than displacing anything.
    try testing.expectEqual(@as(u32, 0), out[0]);
    try testing.expectEqual(@as(u32, 2), out[1]);
    try testing.expectEqual(@as(u32, 1), out[2]);

    // It never exceeds `m`, and never duplicates a selection.
    var two: [2]u32 = undefined;
    const capped = selectNeighbours(&cands, 2, Ctx.between, @ptrCast(&ctx), &two, true);
    try testing.expectEqual(@as(usize, 2), capped);
    try testing.expect(two[0] != two[1]);
}

test "level assignment is deterministic per node and independent of order" {
    const p = Params.fromM(16, 100, 42);
    // Same node, same level, every time.
    for (0..100) |i| {
        const node: u32 = @intCast(i);
        const first = assignLevel(p, node);
        for (0..5) |_| try testing.expectEqual(first, assignLevel(p, node));
    }

    // And it does not depend on how many nodes came before, which is what
    // makes the parallel build's level assignment order-independent.
    const forward = blk: {
        var acc: usize = 0;
        for (0..1000) |i| acc += assignLevel(p, @intCast(i));
        break :blk acc;
    };
    const backward = blk: {
        var acc: usize = 0;
        var i: usize = 1000;
        while (i > 0) {
            i -= 1;
            acc += assignLevel(p, @intCast(i));
        }
        break :blk acc;
    };
    try testing.expectEqual(forward, backward);
}

test "level distribution follows the exponential decay" {
    const p = Params.fromM(16, 100, 7);
    const n = 100_000;
    var counts = [_]usize{0} ** 16;
    for (0..n) |i| counts[assignLevel(p, @intCast(i))] += 1;

    // §6.5: "standard exponential decay, mL = 1/ln(M)". At M=16 the expected
    // fraction at level 0 is 1 - 1/16 = 93.75%.
    const level0_frac = @as(f64, @floatFromInt(counts[0])) / n;
    try testing.expect(level0_frac > 0.92 and level0_frac < 0.955);

    // Each successive level should hold roughly 1/M of the previous.
    try testing.expect(counts[1] > 0);
    try testing.expect(counts[1] < counts[0]);
    try testing.expect(counts[2] < counts[1]);
}

test "a different seed produces a different level assignment" {
    // §8.7 records the seed in meta.json precisely because it changes the
    // graph; if it did not, recording it would be pointless.
    const a = Params.fromM(16, 100, 1);
    const b = Params.fromM(16, 100, 2);
    var differences: usize = 0;
    for (0..1000) |i| {
        if (assignLevel(a, @intCast(i)) != assignLevel(b, @intCast(i))) differences += 1;
    }
    try testing.expect(differences > 0);
}

test "a key-drawn level follows the point, not the slot it landed in" {
    // The whole of findings 34. Two uploads of one corpus give a vector two
    // different internal offsets, and a level drawn from the offset therefore
    // moves with the upload; drawn from a stable key it does not.
    const p = Params.fromM(16, 100, 1);
    const key: u64 = 0xdeadbeef;
    try testing.expectEqual(assignLevelKey(p, key), assignLevelKey(p, key));
    // And it is still the same draw: keyed on a node id's own value, the two
    // functions must agree about the *distribution*, not about each point.
    var counts: [level_cap + 1]usize = @splat(0);
    for (0..100_000) |i| counts[assignLevelKey(p, i)] += 1;
    // At mL = 1/ln(16), level 0 takes 15/16 of the points.
    const at0: f64 = @as(f64, @floatFromInt(counts[0])) / 100_000.0;
    try testing.expect(at0 > 0.92 and at0 < 0.96);
    // A different seed is a different assignment, as for `assignLevel`.
    var differences: usize = 0;
    for (0..1000) |i| {
        if (assignLevelKey(p, i) != assignLevelKey(Params.fromM(16, 100, 2), i)) differences += 1;
    }
    try testing.expect(differences > 0);
}

test "level_keys makes the layout independent of the node order" {
    // Two graphs over the same points in two different orders: with the key
    // following the point, every point keeps its level. This is what the
    // engine would get by keying on the external id.
    const n = 2000;
    const p = Params.fromM(16, 100, 7);
    var forward: [n]u64 = undefined;
    var reversed: [n]u64 = undefined;
    for (0..n) |i| {
        forward[i] = i;
        reversed[i] = n - 1 - i;
    }
    for (0..n) |node| {
        // Node `node` holds point `forward[node]` in one and `reversed[node]`
        // in the other; the point's level has to match in both.
        const point = forward[node];
        const other_node = n - 1 - node;
        try testing.expectEqual(reversed[other_node], point);
        try testing.expectEqual(assignLevelKey(p, forward[node]), assignLevelKey(p, reversed[other_node]));
    }
    // Whereas the node id gives them different levels for at least some.
    var moved: usize = 0;
    for (0..n) |node| {
        if (assignLevel(p, @intCast(node)) != assignLevel(p, @intCast(n - 1 - node))) moved += 1;
    }
    try testing.expect(moved > 0);
}

test "graph layout: level 0 stride is 2M and addresses are computed" {
    var g = try Graph.init(testing.allocator, Params.fromM(16, 100, 1), 100);
    defer g.deinit();

    try testing.expectEqual(@as(usize, 32), g.params.m0);
    // §6.5: "For M=16: 128 B/node = exactly two cache lines".
    try testing.expectEqual(@as(usize, 128), g.params.m0 * @sizeOf(u32));

    const a = g.level0Slice(0);
    const b = g.level0Slice(1);
    try testing.expectEqual(@as(usize, 32), a.len);
    // Contiguous and computed, not looked up.
    try testing.expectEqual(@intFromPtr(a.ptr) + 128, @intFromPtr(b.ptr));
}

test "graph starts fully empty" {
    var g = try Graph.init(testing.allocator, Params.fromM(8, 50, 1), 16);
    defer g.deinit();
    for (0..16) |i| {
        for (g.level0Slice(@intCast(i))) |n| {
            try testing.expectEqual(empty_neighbour, n);
        }
    }
    try testing.expectEqual(empty_neighbour, g.entry_point);
}

test "checksum changes when the graph changes and is stable when it does not" {
    var g = try Graph.init(testing.allocator, Params.fromM(4, 20, 1), 8);
    defer g.deinit();
    g.count = 4;
    g.entry_point = 0;

    const before = g.checksum();
    try testing.expectEqual(before, g.checksum()); // stable

    g.level0Slice(1)[0] = 3;
    const after = g.checksum();
    try testing.expect(before != after);

    // The entry point is part of the graph a search depends on.
    g.entry_point = 2;
    try testing.expect(after != g.checksum());
}

test "heuristic selection prunes a candidate covered by a closer neighbour" {
    // Three candidates on a line. Under the heuristic, a candidate that sits
    // closer to an already-selected neighbour than to the query is dropped,
    // because that direction is already reachable.
    const Ctx = struct {
        // Pairwise similarity (higher = closer), indexed [a][b].
        sim: [3][3]f32,

        fn between(ctx: *const anyopaque, a: u32, b: u32) f32 {
            const self: *const @This() = @ptrCast(@alignCast(ctx));
            return self.sim[a][b];
        }
    };

    // Candidates 0 and 1 are near each other; 2 is far from both.
    var ctx = Ctx{ .sim = .{
        .{ 0.0, 0.95, 0.1 },
        .{ 0.95, 0.0, 0.1 },
        .{ 0.1, 0.1, 0.0 },
    } };

    const cands = [_]Candidate{
        .{ .id = 0, .score = 0.9 }, // best
        .{ .id = 1, .score = 0.85 }, // closer to 0 (0.95) than to query (0.85) -> pruned
        .{ .id = 2, .score = 0.5 }, // far from 0 (0.1) -> kept
    };
    var out: [3]u32 = undefined;
    // `keep_pruned = false` isolates the heuristic itself. With it on — the
    // default, and what the builder uses — candidate 1 is added back into the
    // spare slot rather than left empty; see the test below.
    const n = selectNeighbours(&cands, 3, Ctx.between, @ptrCast(&ctx), &out, false);

    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqual(@as(u32, 0), out[0]);
    try testing.expectEqual(@as(u32, 2), out[1]);
}

test "heuristic selection respects the m bound" {
    const Ctx = struct {
        fn between(_: *const anyopaque, _: u32, _: u32) f32 {
            // Everything mutually distant, so nothing is ever pruned and only
            // the m bound stops the selection.
            return -1000.0;
        }
    };
    var cands: [20]Candidate = undefined;
    for (&cands, 0..) |*c, i| c.* = .{ .id = @intCast(i), .score = 1.0 - @as(f32, @floatFromInt(i)) * 0.01 };

    var out: [20]u32 = undefined;
    var dummy: u8 = 0;
    const n = selectNeighboursHeuristic(&cands, 6, Ctx.between, @ptrCast(&dummy), &out);
    try testing.expectEqual(@as(usize, 6), n);
    // And it kept the best six, in order.
    for (0..6) |i| try testing.expectEqual(@as(u32, @intCast(i)), out[i]);
}

test "searchLayer at ef = 1 returns the single best neighbour of a tiny graph" {
    // The smallest legal `ef`. `ef == 0` is asserted against at the top of
    // `searchLayer` because a zero-capacity result heap reads as full while
    // empty and its `peekWorst` is an unwritten slot; `ef == 1` is the edge
    // that has to keep working.
    const Ctx = struct {
        // Higher is better; node 2 is the best, node 0 the entry.
        pub fn score(_: *const anyopaque, node: u32) f32 {
            return @floatFromInt(node);
        }
    };
    var g = try Graph.init(testing.allocator, Params.fromM(4, 8, 1), 4);
    defer g.deinit();
    g.count = 3;
    g.entry_point = 0;
    g.level0Slice(0)[0] = 1;
    g.level0Slice(1)[0] = 0;
    g.level0Slice(1)[1] = 2;
    g.level0Slice(2)[0] = 1;

    const ctx: u8 = 0;
    const idx = Index{ .graph = &g, .scorer = .{ .ctx = @ptrCast(&ctx), .call = Ctx.score } };
    var scratch = try Index.Scratch.init(testing.allocator, 4, 8);
    defer scratch.deinit(testing.allocator);

    var buf: [1]Candidate = undefined;
    var out = heap.TopK.init(&buf, 1);
    idx.search(1, &scratch, &out);
    const got = out.finish();
    try testing.expectEqual(@as(usize, 1), got.len);
    try testing.expectEqual(@as(u32, 2), got[0].id);
}

test "searchFrom starts where it is told and ignores the entry point" {
    // Two islands on level 0: {0, 1} and {2, 3}, with nothing between them.
    // From the entry point (0) the best node (3) is unreachable; started on
    // the other island it is found. That is the whole difference between
    // what the descent cost and what level 0 cost.
    const Ctx = struct {
        pub fn score(_: *const anyopaque, node: u32) f32 {
            return @floatFromInt(node);
        }
    };
    var g = try Graph.init(testing.allocator, Params.fromM(4, 8, 1), 4);
    defer g.deinit();
    g.count = 4;
    g.entry_point = 0;
    g.level0Slice(0)[0] = 1;
    g.level0Slice(1)[0] = 0;
    g.level0Slice(2)[0] = 3;
    g.level0Slice(3)[0] = 2;

    const ctx: u8 = 0;
    const idx = Index{ .graph = &g, .scorer = .{ .ctx = @ptrCast(&ctx), .call = Ctx.score } };
    var scratch = try Index.Scratch.init(testing.allocator, 4, 8);
    defer scratch.deinit(testing.allocator);
    var buf: [1]Candidate = undefined;

    var out = heap.TopK.init(&buf, 1);
    idx.search(4, &scratch, &out);
    try testing.expectEqual(@as(u32, 1), out.finish()[0].id);

    out = heap.TopK.init(&buf, 1);
    idx.searchFrom(2, 4, &scratch, &out);
    try testing.expectEqual(@as(u32, 3), out.finish()[0].id);
}
