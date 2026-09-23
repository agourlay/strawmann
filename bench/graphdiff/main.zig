//! Why two builds of one collection are two different graphs (findings 34).
//!
//! `buildParallel` inserts concurrently under per-node locks, so which thread
//! reaches a node first decides whose neighbour list fills up and in what order
//! `linkBack` re-prunes. Three builds of one SIFT1M collection at the same
//! parameters spread 0.00288 of recall@10 at `ef` 512, against Qdrant's 0.00009
//! over its own three, which is wider than the gap between the two engines
//! there: a single pass can report either engine ahead at high recall.
//!
//! Unreachable nodes were eliminated as the mechanism (sixteen orphans are
//! worth 0.0000145 of recall, two hundred times too small, and
//! `build.repairUnreachable` now removes them anyway). What is left is *which
//! edges* the pruning race keeps, and nothing measured that. This does.
//!
//! Usage:
//!   graph-diff <vectors.fbin> [--queries <q.fbin>] [flags]
//!
//!   --builds N        graphs to build and compare (default 3)
//!   --threads N       builder threads (default: every core)
//!   --limit N         use only the first N points
//!   --nq N            queries to score recall over (default 1000)
//!   --ef N            search width for the recall scoring (default 512)
//!   --k N             recall@k (default 10)
//!   --m N, --ef-construct N, --seed N
//!   --metric euclid|dot
//!   --histogram       per-node edge-difference histogram
//!   --seeds A,B,...   one build per level seed (up to 8), instead of --builds
//!                     builds at --seed: the seed-to-seed comparison of
//!                     findings 34 in one run
//!   --per-query       which queries each build loses, against the best build,
//!                     and whether the builds lose the same ones
//!   --oracle-entry    also search level 0 from each query's true nearest
//!                     neighbour, skipping the descent (implies --per-query)
//!
//! Structure, not timing, so this needs no quiet host: every number below is
//! exact, and a contaminated run is a *better* input, because foreign load
//! widens the draw (roughly half of the originally reported 0.0062 spread was
//! contamination perturbing the interleaving).
//!
//! Read the output in order. Level membership is a pure function of
//! `(seed, node)` (`hnsw.assignLevel`), so two builds of one collection have
//! identical level assignment by construction, and the only things that can
//! move are the entry point, the upper-level edges and the level-0 edges. The
//! first two are hundreds of edges against level 0's tens of millions, so if
//! the good and the bad draw disagree there, the suspect set is small enough to
//! read by hand and nothing below it matters.
//!
//! The control pair is load-bearing. A racy builder churns edges whether or not
//! the churn costs recall, so "the best and worst build differ by X edges" says
//! nothing on its own: what says something is X against the same number for two
//! builds whose recall agrees.

const std = @import("std");
const Io = std.Io;
const strawmann = @import("strawmann");

const hnsw = strawmann.index.hnsw;
const build_hnsw = strawmann.index.build;
const heap = strawmann.index.heap;
const dist = strawmann.dist;
const Graph = hnsw.Graph;
const Candidate = heap.Candidate;
const empty = hnsw.empty_neighbour;

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// A `.fbin`: `u32 count`, `u32 dim`, then `count * dim` little-endian f32.
///
/// Mapped rather than read, so two runs over SIFT1M share one page cache copy
/// and the 512 MB does not land in this process's heap.
pub const Corpus = struct {
    dim: usize,
    count: usize,
    vectors: []const f32,
    kernel: dist.Kernel,
    map: []align(4096) const u8,
    /// `order[node]` is the vector node `node` holds, or identity when null.
    ///
    /// The harness does not upload SIFT1M in file order: bfb sends batches over
    /// several connections and the server assigns each point the next offset as
    /// it arrives, so which vector becomes node 7 is decided by arrival. That
    /// matters more than it sounds, because `hnsw.assignLevel` is a pure
    /// function of the *node id*: a different arrival order is a different set
    /// of upper-level nodes and a different entry point, not merely a different
    /// interleaving of the same build.
    order: ?[]const u32 = null,

    pub fn open(path: []const u8, kernel: dist.Kernel, limit: usize) !Corpus {
        const fd = std.os.linux.open(
            try nullTerm(path),
            .{ .ACCMODE = .RDONLY },
            0,
        );
        if (std.os.linux.errno(fd) != .SUCCESS) return error.OpenFailed;
        const fdi: i32 = @intCast(fd);
        defer _ = std.os.linux.close(fdi);

        const end = std.os.linux.lseek(fdi, 0, std.os.linux.SEEK.END);
        if (std.os.linux.errno(end) != .SUCCESS) return error.StatFailed;
        const size: usize = end;
        if (size < 8) return error.ShortFile;

        const rc = std.os.linux.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fdi, 0);
        if (std.os.linux.errno(rc) != .SUCCESS) return error.MapFailed;
        const ptr: [*]align(4096) const u8 = @ptrFromInt(rc);
        const map = ptr[0..size];

        const count: usize = std.mem.readInt(u32, map[0..4], .little);
        const dim: usize = std.mem.readInt(u32, map[4..8], .little);
        if (dim == 0 or count == 0) return error.EmptyCorpus;
        if (8 + count * dim * 4 > size) return error.ShortFile;

        const n = if (limit == 0) count else @min(limit, count);
        const floats: [*]const f32 = @ptrCast(@alignCast(map.ptr + 8));
        return .{
            .dim = dim,
            .count = n,
            .vectors = floats[0 .. n * dim],
            .kernel = kernel,
            .map = map,
        };
    }

    pub fn close(self: *Corpus) void {
        _ = std.os.linux.munmap(self.map.ptr, self.map.len);
    }

    /// The vector node `node` holds.
    pub fn vectorOf(self: *const Corpus, node: u32) u32 {
        return if (self.order) |o| o[node] else node;
    }

    pub fn row(self: *const Corpus, node: u32) []const f32 {
        const v = self.vectorOf(node);
        return self.vectors[@as(usize, v) * self.dim ..][0..self.dim];
    }

    /// The row of a *vector*, which is what ground truth and the diff speak in.
    pub fn vectorRow(self: *const Corpus, vec: u32) []const f32 {
        return self.vectors[@as(usize, vec) * self.dim ..][0..self.dim];
    }

    fn between(ctx: *const anyopaque, a: u32, b: u32) f32 {
        const self: *const Corpus = @ptrCast(@alignCast(ctx));
        return dist.similarity(self.kernel, self.row(a), self.row(b));
    }

    pub fn scorer(self: *const Corpus) build_hnsw.Scorer {
        return .{ .ctx = @ptrCast(self), .between = between };
    }

    /// A query bound to the corpus, the shape `hnsw.Index` scores against.
    pub const Probe = struct {
        corpus: *const Corpus,
        query: []const f32,

        pub fn score(ctx: *const anyopaque, node: u32) f32 {
            const self: *const Probe = @ptrCast(@alignCast(ctx));
            return dist.similarity(self.corpus.kernel, self.query, self.corpus.row(node));
        }
    };
};

/// `std.time.nanoTimestamp` moved under the `Io` interface in Zig 0.16 and the
/// builder threads have no `Io` to thread through, so the raw monotonic clock,
/// as `collection.monotonicNs` does.
fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

var path_buf: [4096]u8 = undefined;

fn nullTerm(path: []const u8) ![*:0]const u8 {
    if (path.len + 1 > path_buf.len) return error.PathTooLong;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    return @ptrCast(&path_buf);
}

// ---------------------------------------------------------------------------
// The diff
// ---------------------------------------------------------------------------

/// One node's neighbour list as a set: sorted, empties dropped.
///
/// Slot order is an artifact of the order the builder's threads arrived in, not
/// a property of the graph a search can observe, so comparing rows positionally
/// would report every node as differing for two graphs that are the same graph.
fn rowSet(view: View, row: []const u32, buf: []u32) []u32 {
    var n: usize = 0;
    for (row) |v| {
        if (v == empty) continue;
        buf[n] = view.vectorOf(v);
        n += 1;
    }
    const out = buf[0..n];
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
    return out;
}

/// A graph together with the insertion order it was built under.
///
/// Two builds of one corpus in *file* order share a node space and the
/// identity view is right. Two builds in different arrival orders do not: node
/// 7 is a different point in each, and comparing their rows by node id would
/// report two graphs of the same shape as completely different. Everything the
/// diff reports is therefore in *vector* space, which both builds agree on.
pub const View = struct {
    g: *const Graph,
    /// `order[node]` is the vector that node holds.
    order: ?[]const u32 = null,
    /// `inv[vector]` is the node holding it. Required whenever `order` is set.
    inv: ?[]const u32 = null,

    pub fn of(g: *const Graph) View {
        return .{ .g = g };
    }

    pub fn vectorOf(self: View, node: u32) u32 {
        return if (self.order) |o| o[node] else node;
    }

    pub fn nodeOf(self: View, vector: u32) u32 {
        return if (self.inv) |iv| iv[vector] else vector;
    }

    pub fn levelOf(self: View, vector: u32) u8 {
        return self.g.node_levels[self.nodeOf(vector)];
    }

    /// The entry point as a vector id, so two node spaces are comparable.
    pub fn entry(self: View) u32 {
        if (self.g.entry_point == empty) return empty;
        return self.vectorOf(self.g.entry_point);
    }
};

pub const LevelDiff = struct {
    /// Nodes that participate in this level in either graph.
    nodes: u64 = 0,
    /// Nodes whose neighbour *set* is identical in both.
    same_nodes: u64 = 0,
    edges_a: u64 = 0,
    edges_b: u64 = 0,
    only_a: u64 = 0,
    only_b: u64 = 0,

    pub fn common(self: LevelDiff) u64 {
        return self.edges_a - self.only_a;
    }

    /// `|A n B| / |A u B|` over directed edges. 1.0 is the same graph.
    pub fn jaccard(self: LevelDiff) f64 {
        const union_size = self.common() + self.only_a + self.only_b;
        if (union_size == 0) return 1.0;
        return @as(f64, @floatFromInt(self.common())) / @as(f64, @floatFromInt(union_size));
    }
};

/// Mean edge similarity, higher being closer (euclid's kernel is the negated
/// squared distance).
///
/// The hypothesis findings 34 leaves open is that the race eats the long-range
/// links a traversal depends on. If it does, the edges present in one build and
/// not the other are *further* on average than the ones both builds kept, and
/// these three means say so directly.
pub const EdgeLengths = struct {
    common_sum: f64 = 0,
    common_n: u64 = 0,
    only_a_sum: f64 = 0,
    only_a_n: u64 = 0,
    only_b_sum: f64 = 0,
    only_b_n: u64 = 0,

    fn mean(sum: f64, n: u64) f64 {
        if (n == 0) return 0;
        return sum / @as(f64, @floatFromInt(n));
    }
};

pub const Report = struct {
    count: u64,
    /// Nodes whose `node_levels` entry differs. Expected zero: level
    /// assignment is seeded and pure in the node id, so anything else here is
    /// a bigger finding than the rest of this report.
    level_mismatches: u64 = 0,
    entry_a: u32,
    entry_b: u32,
    max_level_a: u8,
    max_level_b: u8,
    /// Index `l` is level `l`.
    levels: []LevelDiff,
    /// Level-0 nodes with no in-edge, per graph.
    indeg0_a: u64 = 0,
    indeg0_b: u64 = 0,
    /// `hist[k]`: nodes whose level-0 symmetric difference is `k` edges,
    /// saturating at the last slot.
    hist: []u64,
    lengths: ?EdgeLengths = null,
    /// Whether the two builds inserted in the same order. Under one order the
    /// level assignment is identical by construction and a mismatch is a bug;
    /// under two it is the *first* thing that differs and the reason the rest
    /// does.
    same_node_space: bool = true,
    /// Whether the two builds drew levels from the same seed. Set by the
    /// caller, which knows; two seeds are two level assignments by design.
    same_seed: bool = true,

    pub fn deinit(self: *Report, alloc: std.mem.Allocator) void {
        alloc.free(self.levels);
        alloc.free(self.hist);
    }
};

pub const hist_len = 33;

/// Compare two graphs built from the same points.
///
/// `corpus` is optional: with it the report carries edge lengths, without it
/// the diff is pure structure.
pub fn diff(
    alloc: std.mem.Allocator,
    va: View,
    vb: View,
    corpus: ?*const Corpus,
) !Report {
    const a = va.g;
    const b = vb.g;
    if (a.params.m != b.params.m or a.params.m0 != b.params.m0) return error.ShapeMismatch;
    if (a.count != b.count) return error.CountMismatch;

    const n = a.count;
    const top = @max(a.max_level, b.max_level);
    const levels = try alloc.alloc(LevelDiff, @as(usize, top) + 1);
    errdefer alloc.free(levels);
    @memset(levels, .{});
    const hist = try alloc.alloc(u64, hist_len);
    errdefer alloc.free(hist);
    @memset(hist, 0);

    // In-degree over level 0, both graphs. A count rather than a bit, because
    // "held by one edge" and "held by twenty" are different situations for a
    // builder that evicts back-edges; at 1.1M nodes this is 8 MB.
    const indeg_a = try alloc.alloc(u32, n);
    defer alloc.free(indeg_a);
    @memset(indeg_a, 0);
    const indeg_b = try alloc.alloc(u32, n);
    defer alloc.free(indeg_b);
    @memset(indeg_b, 0);

    var report: Report = .{
        .count = n,
        .entry_a = va.entry(),
        .entry_b = vb.entry(),
        .max_level_a = a.max_level,
        .max_level_b = b.max_level,
        .levels = levels,
        .hist = hist,
    };
    var lengths: EdgeLengths = .{};

    const m0 = a.params.m0;
    const set_a = try alloc.alloc(u32, m0);
    defer alloc.free(set_a);
    const set_b = try alloc.alloc(u32, m0);
    defer alloc.free(set_b);

    // Indexed by vector, not by node: under two different arrival orders the
    // same vector wears two different node ids, and it is the vector the two
    // builds agree about.
    for (0..n) |i| {
        const vec: u32 = @intCast(i);
        const node_a = va.nodeOf(vec);
        const node_b = vb.nodeOf(vec);
        const la = a.node_levels[node_a];
        const lb = b.node_levels[node_b];
        if (la != lb) report.level_mismatches += 1;

        var lvl: u8 = 0;
        while (lvl <= @max(la, lb)) : (lvl += 1) {
            const row_a = if (lvl <= la) a.neighbours(node_a, lvl) else &[_]u32{};
            const row_b = if (lvl <= lb) b.neighbours(node_b, lvl) else &[_]u32{};
            const sa = rowSet(va, row_a, set_a);
            const sb = rowSet(vb, row_b, set_b);
            const ld = &levels[lvl];
            ld.nodes += 1;
            ld.edges_a += sa.len;
            ld.edges_b += sb.len;

            // One merge over the two sorted sets, classifying each edge as
            // common, only-A or only-B. The counters and the edge lengths come
            // from the same walk so they cannot disagree about which is which.
            var ia: usize = 0;
            var ib: usize = 0;
            var only_a: u32 = 0;
            var only_b: u32 = 0;
            while (ia < sa.len or ib < sb.len) {
                const take_a = ib == sb.len or (ia < sa.len and sa[ia] < sb[ib]);
                const take_b = ia == sa.len or (ib < sb.len and sb[ib] < sa[ia]);
                if (take_a) {
                    only_a += 1;
                    if (corpus) |c| {
                        lengths.only_a_sum += dist.similarity(c.kernel, c.vectorRow(vec), c.vectorRow(sa[ia]));
                        lengths.only_a_n += 1;
                    }
                    ia += 1;
                } else if (take_b) {
                    only_b += 1;
                    if (corpus) |c| {
                        lengths.only_b_sum += dist.similarity(c.kernel, c.vectorRow(vec), c.vectorRow(sb[ib]));
                        lengths.only_b_n += 1;
                    }
                    ib += 1;
                } else {
                    if (corpus) |c| {
                        lengths.common_sum += dist.similarity(c.kernel, c.vectorRow(vec), c.vectorRow(sa[ia]));
                        lengths.common_n += 1;
                    }
                    ia += 1;
                    ib += 1;
                }
            }
            ld.only_a += only_a;
            ld.only_b += only_b;
            if (only_a == 0 and only_b == 0) ld.same_nodes += 1;

            if (lvl == 0) {
                const sym = @as(usize, only_a) + @as(usize, only_b);
                hist[@min(sym, hist_len - 1)] += 1;
                for (sa) |nb| {
                    if (nb < n) indeg_a[nb] += 1;
                }
                for (sb) |nb| {
                    if (nb < n) indeg_b[nb] += 1;
                }
            }
        }
    }

    for (indeg_a) |d| {
        if (d == 0) report.indeg0_a += 1;
    }
    for (indeg_b) |d| {
        if (d == 0) report.indeg0_b += 1;
    }
    if (corpus != null) report.lengths = lengths;
    report.same_node_space = (va.order == null) == (vb.order == null) and
        (va.order == null or va.order.?.ptr == vb.order.?.ptr);
    return report;
}

// ---------------------------------------------------------------------------
// Recall, so the diff knows which build is the good one
// ---------------------------------------------------------------------------

/// Exact top-`k` for one query, by scanning every point.
fn bruteTop(c: *const Corpus, q: []const f32, k: usize, scratch: []Candidate, out: []u32) void {
    var top = heap.TopK.init(scratch, k);
    for (0..c.count) |i| {
        const id: u32 = @intCast(i);
        _ = top.push(.{ .id = id, .score = dist.similarity(c.kernel, q, c.row(id)) });
    }
    for (0..k) |i| out[i] = if (i < top.len) top.items[i].id else empty;
}

/// Ground truth for the first `nq` queries, computed in parallel.
///
/// The gt files beside the datasets are keyed to the full corpus, and this
/// tool runs at `--limit` as well, so it computes its own: at SIFT1M's
/// 1,000,000 x 128 this is a minute over every core, and it is exact.
pub fn groundTruth(
    alloc: std.mem.Allocator,
    c: *const Corpus,
    queries: *const Corpus,
    nq: usize,
    k: usize,
    threads: usize,
) ![]u32 {
    const gt = try alloc.alloc(u32, nq * k);
    errdefer alloc.free(gt);

    const Job = struct {
        c: *const Corpus,
        queries: *const Corpus,
        gt: []u32,
        nq: usize,
        k: usize,
        next: std.atomic.Value(usize) = .init(0),

        fn work(self: *@This()) void {
            const scratch = std.heap.page_allocator.alloc(Candidate, self.k) catch return;
            defer std.heap.page_allocator.free(scratch);
            while (true) {
                const i = self.next.fetchAdd(1, .monotonic);
                if (i >= self.nq) return;
                bruteTop(self.c, self.queries.row(@intCast(i)), self.k, scratch, self.gt[i * self.k ..][0..self.k]);
            }
        }
    };
    var job: Job = .{ .c = c, .queries = queries, .gt = gt, .nq = nq, .k = k };

    const pool = try alloc.alloc(std.Thread, threads);
    defer alloc.free(pool);
    var spawned: usize = 0;
    for (pool) |*t| {
        t.* = std.Thread.spawn(.{}, Job.work, .{&job}) catch break;
        spawned += 1;
    }
    if (spawned == 0) job.work();
    for (pool[0..spawned]) |t| t.join();
    return gt;
}

/// recall@k of `g` at `ef`, against `gt`.
pub fn recall(
    alloc: std.mem.Allocator,
    g: *Graph,
    c: *const Corpus,
    queries: *const Corpus,
    gt: []const u32,
    nq: usize,
    k: usize,
    ef: usize,
    hits_out: ?[]u8,
) !f64 {
    var scratch = try hnsw.Index.Scratch.init(alloc, c.count, ef);
    defer scratch.deinit(alloc);
    const out = try alloc.alloc(Candidate, k);
    defer alloc.free(out);

    var hits: u64 = 0;
    for (0..nq) |qi| {
        var probe = Corpus.Probe{ .corpus = c, .query = queries.row(@intCast(qi)) };
        const index = hnsw.Index{ .graph = g, .scorer = hnsw.Scorer.of(&probe) };
        var top = heap.TopK.init(out, k);
        index.search(ef, &scratch, &top);
        const h = hitsOf(c, top.items[0..top.len], gt[qi * k ..][0..k]);
        if (hits_out) |ho| ho[qi] = h;
        hits += h;
    }
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(nq * k));
}

/// How many of `want` a result set holds.
fn hitsOf(c: *const Corpus, got: []const Candidate, want: []const u32) u8 {
    var h: u8 = 0;
    for (got) |cand| {
        // The search answers in node ids and the ground truth is in vector
        // ids. Identical until `--shuffle`, and silently wrong after it.
        const v = c.vectorOf(cand.id);
        for (want) |w| {
            if (w == v) {
                h += 1;
                break;
            }
        }
    }
    return h;
}

/// Per-query hits with the descent skipped: level 0 searched from each query's
/// true nearest neighbour (`hnsw.Index.searchFrom`).
///
/// The seed loss of findings 34 is flat across `ef`, so some true neighbours
/// are out of reach at any beam width, and there are two places that can
/// happen. If the descent lands where they cannot be reached from, starting at
/// the answer recovers them; if level 0 does not link them, it does not.
/// `inv` maps a vector to the node holding it, null in file order.
pub fn oracleHits(
    alloc: std.mem.Allocator,
    g: *Graph,
    c: *const Corpus,
    queries: *const Corpus,
    gt: []const u32,
    nq: usize,
    k: usize,
    ef: usize,
    inv: ?[]const u32,
    out_hits: []u8,
) !void {
    var scratch = try hnsw.Index.Scratch.init(alloc, c.count, ef);
    defer scratch.deinit(alloc);
    const out = try alloc.alloc(Candidate, k);
    defer alloc.free(out);
    for (0..nq) |qi| {
        const want = gt[qi * k ..][0..k];
        var probe = Corpus.Probe{ .corpus = c, .query = queries.row(@intCast(qi)) };
        const index = hnsw.Index{ .graph = g, .scorer = hnsw.Scorer.of(&probe) };
        var top = heap.TopK.init(out, k);
        const entry = if (inv) |iv| iv[want[0]] else want[0];
        index.searchFrom(entry, ef, &scratch, &top);
        out_hits[qi] = hitsOf(c, top.items[0..top.len], want);
    }
}

/// One build's per-query hits against the best build's.
pub const Loss = struct {
    /// Queries this build answered worse and better than the best build did.
    worse: usize = 0,
    better: usize = 0,
    /// Neighbours lost and gained across those queries.
    lost: usize = 0,
    gained: usize = 0,
    /// Of the worse queries, how many lost 1, 2, 3 and 4 or more neighbours.
    by_size: [4]usize = @splat(0),
    /// Queries that miss at least one neighbour, whatever the best build did.
    imperfect: usize = 0,

    pub fn of(best: []const u8, this: []const u8, k: usize) Loss {
        var l: Loss = .{};
        for (best, this) |b, t| {
            if (t < k) l.imperfect += 1;
            if (t < b) {
                l.worse += 1;
                l.lost += b - t;
                l.by_size[@min(b - t, 4) - 1] += 1;
            } else if (t > b) {
                l.better += 1;
                l.gained += t - b;
            }
        }
        return l;
    }
};

/// |A & B| / |A | B| over the queries two builds answer worse than `best`.
/// Near 1 says the loss is a fixed set of queries whatever the seed; near 0
/// says each seed loses its own.
pub fn worseOverlap(best: []const u8, a: []const u8, b: []const u8) f64 {
    var both: usize = 0;
    var either: usize = 0;
    for (best, a, b) |x, y, z| {
        const in_a = y < x;
        const in_b = z < x;
        if (in_a and in_b) both += 1;
        if (in_a or in_b) either += 1;
    }
    if (either == 0) return 1;
    return @as(f64, @floatFromInt(both)) / @as(f64, @floatFromInt(either));
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn pct(part: u64, whole: u64) f64 {
    if (whole == 0) return 0;
    return 100.0 * @as(f64, @floatFromInt(part)) / @as(f64, @floatFromInt(whole));
}

pub fn write(w: *Io.Writer, r: Report, histogram: bool) !void {
    if (r.level_mismatches == 0) {
        try w.print("level assignment identical ({d} nodes)\n", .{r.count});
    } else if (!r.same_seed) {
        try w.print("level assignment: {d} of {d} points ({d:.1}%) sit at a different " ++
            "level,\n                  which is what a different seed does\n", .{
            r.level_mismatches, r.count, pct(r.level_mismatches, r.count),
        });
    } else if (r.same_node_space) {
        try w.print("level assignment: {d} nodes DIFFER, expected 0 " ++
            "(`assignLevel` is pure in (seed, node))\n", .{r.level_mismatches});
    } else {
        try w.print("level assignment: {d} of {d} points ({d:.1}%) sit at a different " ++
            "level,\n                  which is what a different arrival order does\n", .{
            r.level_mismatches, r.count, pct(r.level_mismatches, r.count),
        });
    }
    try w.print("entry point      {s} ({d} / {d}), max level {d} / {d}\n\n", .{
        if (r.entry_a == r.entry_b) "same" else "DIFFERS",
        r.entry_a,
        r.entry_b,
        r.max_level_a,
        r.max_level_b,
    });

    try w.print("{s:<6} {s:>11} {s:>13} {s:>13} {s:>12} {s:>9} {s:>10}\n", .{
        "level", "nodes", "edges A", "edges B", "differing", "jaccard", "same rows",
    });
    var lvl: usize = r.levels.len;
    while (lvl > 0) {
        lvl -= 1;
        const d = r.levels[lvl];
        if (d.nodes == 0) continue;
        try w.print("{d:<6} {d:>11} {d:>13} {d:>13} {d:>12} {d:>9.5} {d:>9.2}%\n", .{
            lvl,                 d.nodes,     d.edges_a,                  d.edges_b,
            d.only_a + d.only_b, d.jaccard(), pct(d.same_nodes, d.nodes),
        });
    }

    const l0 = r.levels[0];
    try w.print("\nlevel 0        {d:.3}% of A's edges are not in B\n", .{pct(l0.only_a, l0.edges_a)});
    try w.print("in-degree 0    A {d}, B {d}\n", .{ r.indeg0_a, r.indeg0_b });

    if (r.lengths) |e| {
        try w.print("\nmean edge similarity (higher is closer)\n", .{});
        try w.print("  kept by both   {d:>14.2}  over {d} edges\n", .{
            EdgeLengths.mean(e.common_sum, e.common_n), e.common_n,
        });
        try w.print("  only in A      {d:>14.2}  over {d} edges\n", .{
            EdgeLengths.mean(e.only_a_sum, e.only_a_n), e.only_a_n,
        });
        try w.print("  only in B      {d:>14.2}  over {d} edges\n", .{
            EdgeLengths.mean(e.only_b_sum, e.only_b_n), e.only_b_n,
        });
    }

    if (histogram) {
        try w.print("\nper-node level-0 symmetric difference\n", .{});
        for (r.hist, 0..) |c, k| {
            if (c == 0) continue;
            try w.print("  {s}{d:<3} {d:>12}  {d:>6.2}%\n", .{
                if (k == hist_len - 1) ">=" else "  ", k, c, pct(c, r.count),
            });
        }
    }
}

/// Which queries each build loses against the best, and whether they are the
/// same queries. With `--oracle-entry`, the same counts with the descent
/// skipped: what comes back there is what the upper levels were costing.
fn writePerQuery(w: *Io.Writer, builds: []const Build, best: usize, k: usize, ef: usize) !void {
    const ref = builds[best].hits.?;
    try w.print("\n=== per query at ef {d}, against build {d} (seed 0x{x})\n\n", .{ ef, best, builds[best].seed });
    try w.print("{s:<7} {s:>12} {s:>10} {s:>7} {s:>7} {s:>8} {s:>8}   {s}\n", .{
        "build", "seed", "imperfect", "worse", "better", "lost", "gained", "worse by 1/2/3/4+",
    });
    for (builds, 0..) |b, i| {
        const l = Loss.of(ref, b.hits.?, k);
        try w.print("{d:<7} {s:>2}{x:>10} {d:>10} {d:>7} {d:>7} {d:>8} {d:>8}   {d}/{d}/{d}/{d}\n", .{
            i,            "0x",         b.seed,       l.imperfect,  l.worse, l.better, l.lost, l.gained,
            l.by_size[0], l.by_size[1], l.by_size[2], l.by_size[3],
        });
    }

    try w.print("\nworse-than-best overlap (1 is the same queries, 0 is disjoint)\n", .{});
    for (builds, 0..) |a, i| {
        if (i == best) continue;
        for (builds[i + 1 ..], i + 1..) |b, j| {
            if (j == best) continue;
            try w.print("  {d} & {d}  {d:.3}\n", .{ i, j, worseOverlap(ref, a.hits.?, b.hits.?) });
        }
    }

    if (builds[best].oracle == null) return;
    try w.print("\nlevel 0 from the true nearest neighbour, descent skipped\n", .{});
    try w.print("{s:<7} {s:>12} {s:>18} {s:>18} {s:>12}\n", .{
        "build", "seed", "missed, descent", "missed, oracle", "recovered",
    });
    for (builds, 0..) |b, i| {
        var desc: usize = 0;
        var orc: usize = 0;
        for (b.hits.?, b.oracle.?) |h, o| {
            desc += k - h;
            orc += k - o;
        }
        const rec: f64 = if (desc == 0) 0 else 100.0 * (@as(f64, @floatFromInt(desc)) -
            @as(f64, @floatFromInt(orc))) / @as(f64, @floatFromInt(desc));
        try w.print("{d:<7} {s:>2}{x:>10} {d:>18} {d:>18} {d:>11.1}%\n", .{ i, "0x", b.seed, desc, orc, rec });
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

const Args = struct {
    vectors: []const u8 = "",
    queries: []const u8 = "",
    builds: usize = 3,
    threads: usize = 0,
    limit: usize = 0,
    nq: usize = 1000,
    /// The `ef` ladder to score, comma-separated. findings 34 quotes 128, 256
    /// and 512 over one set of builds, and the builds are what is expensive
    /// here, so one run answers the whole row. Builds are ranked by the *last*
    /// one, which is the operating point the finding is about.
    ef: [4]usize = .{ 512, 0, 0, 0 },
    n_ef: usize = 1,
    k: usize = 10,
    m: usize = 16,
    ef_construct: usize = 100,
    /// `core.Config.seed`'s default, so a build here is the build the engine
    /// makes. Level assignment is a pure function of it, and a different seed
    /// is a different set of upper-level nodes rather than a different draw of
    /// the same experiment.
    seed: u64 = 0x57ea3111,
    kernel: dist.Kernel = .euclid,
    histogram: bool = false,
    /// Insert in a different random order per build, which is what the
    /// harness's concurrent upload does to point ids. Zero is file order;
    /// otherwise the reordering granularity, and 100 is the harness's `-b`.
    shuffle: usize = 0,
    /// Draw each point's level from the *vector* it holds rather than from the
    /// node id it was given (`hnsw.assignLevelKey`).
    ///
    /// With `--shuffle`, this is the experiment that separates the two things
    /// a reordered arrival changes: the level assignment, which decides the
    /// upper levels and the entry point, and the insertion sequence, which
    /// decides what the greedy descent had to work with when each point was
    /// linked. Without it they move together and neither can be blamed.
    stable_levels: bool = false,
    /// Link the points in vector-id order whatever order they arrived in
    /// (`Graph.insert_order`).
    ///
    /// The engine's stand-in for "sorted by external id": the bulk build links
    /// in offset order, which is arrival order, and this is the lever
    /// findings 34 ends at. With `--shuffle`, a spread that survives this is a
    /// spread the insertion sequence does not explain.
    stable_order: bool = false,
    /// One build per seed, in place of `builds` at `seed`.
    seeds: [8]u64 = @splat(0),
    n_seeds: usize = 0,
    per_query: bool = false,
    oracle_entry: bool = false,

    fn seedOf(self: Args, i: usize) u64 {
        return if (self.n_seeds != 0) self.seeds[i] else self.seed;
    }
};

fn parseArgs(argv: []const []const u8) !Args {
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        const val = struct {
            fn next(args: []const []const u8, idx: *usize) ![]const u8 {
                idx.* += 1;
                if (idx.* >= args.len) return error.MissingValue;
                return args[idx.*];
            }
        }.next;
        if (std.mem.eql(u8, arg, "--histogram")) {
            a.histogram = true;
        } else if (std.mem.eql(u8, arg, "--stable-levels")) {
            a.stable_levels = true;
        } else if (std.mem.eql(u8, arg, "--stable-order")) {
            a.stable_order = true;
        } else if (std.mem.eql(u8, arg, "--per-query")) {
            a.per_query = true;
        } else if (std.mem.eql(u8, arg, "--oracle-entry")) {
            a.oracle_entry = true;
            a.per_query = true;
        } else if (std.mem.eql(u8, arg, "--seeds")) {
            var it = std.mem.splitScalar(u8, try val(argv, &i), ',');
            a.n_seeds = 0;
            while (it.next()) |tok| {
                if (a.n_seeds == a.seeds.len) return error.TooManySeeds;
                a.seeds[a.n_seeds] = try std.fmt.parseInt(u64, tok, 0);
                a.n_seeds += 1;
            }
        } else if (std.mem.eql(u8, arg, "--shuffle")) {
            a.shuffle = try std.fmt.parseInt(usize, try val(argv, &i), 10);
            if (a.shuffle == 0) return error.ShuffleBatchZero;
        } else if (std.mem.eql(u8, arg, "--queries")) {
            a.queries = try val(argv, &i);
        } else if (std.mem.eql(u8, arg, "--metric")) {
            const s = try val(argv, &i);
            a.kernel = if (std.mem.eql(u8, s, "dot")) .dot else if (std.mem.eql(u8, s, "euclid")) .euclid else return error.UnknownMetric;
        } else if (std.mem.eql(u8, arg, "--builds")) {
            a.builds = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            a.threads = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--limit")) {
            a.limit = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--nq")) {
            a.nq = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--ef")) {
            var it = std.mem.splitScalar(u8, try val(argv, &i), ',');
            a.n_ef = 0;
            while (it.next()) |tok| {
                if (a.n_ef == a.ef.len) return error.TooManyEf;
                a.ef[a.n_ef] = try std.fmt.parseInt(usize, tok, 10);
                a.n_ef += 1;
            }
            if (a.n_ef == 0) return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--k")) {
            a.k = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--m")) {
            a.m = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--ef-construct")) {
            a.ef_construct = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            a.seed = try std.fmt.parseInt(u64, try val(argv, &i), 0);
        } else if (std.mem.startsWith(u8, arg, "-")) {
            return error.UnknownFlag;
        } else {
            if (a.vectors.len != 0) return error.TooManyPaths;
            a.vectors = arg;
        }
    }
    if (a.vectors.len == 0) return error.NeedVectorFile;
    if (a.n_seeds != 0) a.builds = a.n_seeds;
    if (a.builds < 2) return error.NeedTwoBuilds;
    for (a.ef[0..a.n_ef]) |e| if (a.k > e) return error.KAboveEf;
    return a;
}

const Build = struct {
    graph: *Graph,
    /// The arrival order this build inserted in, and its inverse. Null for the
    /// file-order builds, which share a node space.
    order: ?[]u32 = null,
    inv: ?[]u32 = null,
    seconds: f64,
    repaired: usize,
    checksum: u64,
    seed: u64,
    /// Per-query hits at the last `--ef`, under `--per-query`, and the same
    /// with the descent skipped, under `--oracle-entry`.
    hits: ?[]u8 = null,
    oracle: ?[]u8 = null,
    /// One per `--ef`, in the order given. `rank` is the last, which is the
    /// operating point findings 34 is about.
    recall: [4]f64 = @splat(-1),

    fn rank(self: Build, n_ef: usize) f64 {
        return self.recall[n_ef - 1];
    }

    fn view(self: Build) View {
        return .{ .g = self.graph, .order = self.order, .inv = self.inv };
    }
};

/// A fresh arrival order for build `i`, as a permutation and its inverse.
///
/// `batch` is the granularity the reordering happens at, and it is not a
/// detail. The harness uploads with `-b 100 -t 8 -p 8`: a hundred points go
/// out as one request and eight streams race, so what varies between passes is
/// the order of *batches*, while the points inside one stay contiguous. A
/// per-point shuffle answers a different and easier question ("does arrival
/// order matter at all"), and would overstate what the upload actually does.
///
/// Seeded by the build index rather than by the clock, so a run is
/// reproducible: an unrepeatable answer to this is not worth having.
fn arrivalOrder(
    alloc: std.mem.Allocator,
    n: usize,
    i: usize,
    batch: usize,
) !struct { order: []u32, inv: []u32 } {
    const order = try alloc.alloc(u32, n);
    errdefer alloc.free(order);
    const inv = try alloc.alloc(u32, n);
    errdefer alloc.free(inv);

    const b = @max(batch, 1);
    const n_batches = (n + b - 1) / b;
    const batches = try alloc.alloc(u32, n_batches);
    defer alloc.free(batches);
    for (batches, 0..) |*x, k| x.* = @intCast(k);
    var prng = std.Random.DefaultPrng.init(0x5eed_0000 + i);
    prng.random().shuffle(u32, batches);

    var node: usize = 0;
    for (batches) |bi| {
        const start = @as(usize, bi) * b;
        const end = @min(start + b, n);
        for (start..end) |vec| {
            order[node] = @intCast(vec);
            node += 1;
        }
    }
    std.debug.assert(node == n);
    for (order, 0..) |vec, nd| inv[vec] = @intCast(nd);
    return .{ .order = order, .inv = inv };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const args = parseArgs(argv) catch |err| {
        std.debug.print("graph-diff: {s}\n\nusage: graph-diff <vectors.fbin> " ++
            "[--queries <q.fbin>] [--builds N] [--threads N] [--limit N]\n", .{@errorName(err)});
        return err;
    };

    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const alloc = gpa_state.allocator();

    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(.stdout(), init.io, &buf);
    const w = &fw.interface;

    var corpus = try Corpus.open(args.vectors, args.kernel, args.limit);
    defer corpus.close();
    const threads = if (args.threads != 0) args.threads else std.Thread.getCpuCount() catch 1;

    try w.print("corpus   {s}\n", .{args.vectors});
    try w.print("points   {d} x {d}, {s}\n", .{ corpus.count, corpus.dim, @tagName(args.kernel) });
    if (args.shuffle != 0) try w.print("arrival  reordered in batches of {d} " ++
        "(the harness uploads with -b 100 -t 8 -p 8)\n", .{args.shuffle});
    try w.print("params   m={d} ef_construct={d} seed=0x{x} threads={d}{s}{s}{s}\n", .{
        args.m,                                                                         args.ef_construct,                                              args.seed,                                                 threads,
        if (args.shuffle != 0) ", a fresh arrival order per build" else ", file order", if (args.stable_levels) ", levels keyed on the vector" else "", if (args.stable_order) ", linked in vector order" else "",
    });
    try w.flush();

    var queries: ?Corpus = null;
    defer if (queries) |*q| q.close();
    var gt: []u32 = &.{};
    defer if (gt.len != 0) alloc.free(gt);
    var nq: usize = 0;
    if (args.queries.len != 0) {
        queries = try Corpus.open(args.queries, args.kernel, 0);
        if (queries.?.dim != corpus.dim) return error.QueryDimMismatch;
        nq = @min(args.nq, queries.?.count);
        try w.print("queries  {d} of {d}, recall@{d}\n", .{ nq, queries.?.count, args.k });
        try w.flush();
        const t0 = nowNs();
        gt = try groundTruth(alloc, &corpus, &queries.?, nq, args.k, threads);
        try w.print("         exact ground truth in {d:.1}s\n", .{
            @as(f64, @floatFromInt(nowNs() - t0)) / std.time.ns_per_s,
        });
        try w.flush();
    }

    const builds = try alloc.alloc(Build, args.builds);
    defer alloc.free(builds);
    if (args.per_query and args.queries.len == 0) return error.PerQueryNeedsQueries;
    try w.print("\n{s:<7} {s:>12} {s:>9} {s:>9} {s:>20}", .{ "build", "seed", "seconds", "repaired", "checksum" });
    for (args.ef[0..args.n_ef]) |e| try w.print("  {s:>5}{d:<5}", .{ "ef ", e });
    try w.print("\n", .{});
    for (builds, 0..) |*b, i| {
        var order: ?[]u32 = null;
        var inv: ?[]u32 = null;
        if (args.shuffle != 0) {
            const perm = try arrivalOrder(alloc, corpus.count, i, args.shuffle);
            order = perm.order;
            inv = perm.inv;
        }
        // The builder and the recall pass below both read rows through it; the
        // ground truth above was computed in file order and stays that way.
        corpus.order = order;

        const g = try alloc.create(Graph);
        g.* = try Graph.init(alloc, hnsw.Params.fromM(args.m, args.ef_construct, args.seedOf(i)), corpus.count);
        // The level key is the vector id, which is the stand-in here for the
        // external point id the engine would use: both are stable across
        // ingests, and only stability matters to the draw.
        var keys: ?[]u64 = null;
        defer if (keys) |k| alloc.free(k);
        if (args.stable_levels) {
            const k = try alloc.alloc(u64, corpus.count);
            for (k, 0..) |*x, node| x.* = if (order) |o| o[node] else node;
            g.level_keys = k;
            keys = k;
        }
        // `inv[vector]` is the node holding it, so walking `inv` in vector
        // order links the points in the order the file has them, whatever
        // order they arrived in. With no permutation the two coincide and the
        // graph does not need telling.
        if (args.stable_order) {
            if (inv) |iv| g.insert_order = iv;
        }
        const t0 = nowNs();
        const stats = try build_hnsw.buildParallel(alloc, g, corpus.scorer(), corpus.count, threads);
        const seconds = @as(f64, @floatFromInt(nowNs() - t0)) / std.time.ns_per_s;
        b.* = .{
            .graph = g,
            .order = order,
            .inv = inv,
            .seconds = seconds,
            .repaired = stats.repaired,
            .checksum = g.checksum(),
            .seed = args.seedOf(i),
        };
        if (args.per_query) b.hits = try alloc.alloc(u8, nq);
        try w.print("{d:<7} {s:>2}{x:>10} {d:>9.2} {d:>9} {x:>20}", .{ i, "0x", b.seed, b.seconds, b.repaired, b.checksum });
        for (args.ef[0..args.n_ef], 0..) |e, j| {
            const last = j + 1 == args.n_ef;
            if (nq != 0) b.recall[j] = try recall(alloc, g, &corpus, &queries.?, gt, nq, args.k, e, if (last) b.hits else null);
            try w.print("  {d:>10.5}", .{b.recall[j]});
        }
        if (args.oracle_entry) {
            b.oracle = try alloc.alloc(u8, nq);
            try oracleHits(alloc, g, &corpus, &queries.?, gt, nq, args.k, args.ef[args.n_ef - 1], inv, b.oracle.?);
        }
        try w.print("\n", .{});
        try w.flush();
    }
    corpus.order = null; // everything below is in vector space
    defer for (builds) |b| {
        b.graph.deinit();
        alloc.destroy(b.graph);
        if (b.order) |o| alloc.free(o);
        if (b.inv) |iv| alloc.free(iv);
        if (b.hits) |h| alloc.free(h);
        if (b.oracle) |o| alloc.free(o);
    };

    // Which pair to diff. With recall, the best against the worst is the pair
    // the finding is about, and the two *closest* builds are the control that
    // says whether the churn between best and worst is the churn a racy
    // builder produces anyway. Without recall there is only one pair worth
    // printing.
    var best: usize = 0;
    var worst: usize = 0;
    for (builds, 0..) |b, i| {
        if (b.rank(args.n_ef) > builds[best].rank(args.n_ef)) best = i;
        if (b.rank(args.n_ef) < builds[worst].rank(args.n_ef)) worst = i;
    }
    if (best == worst) {
        best = 0;
        worst = 1;
    }

    if (args.per_query) {
        try writePerQuery(w, builds, best, args.k, args.ef[args.n_ef - 1]);
        try w.flush();
    }

    try w.print("\n=== best (build {d}, {d:.5}) against worst (build {d}, {d:.5}) at ef {d}\n\n", .{
        best, builds[best].rank(args.n_ef), worst, builds[worst].rank(args.n_ef), args.ef[args.n_ef - 1],
    });
    var r = try diff(alloc, builds[best].view(), builds[worst].view(), &corpus);
    defer r.deinit(alloc);
    r.same_seed = builds[best].seed == builds[worst].seed;
    try write(w, r, args.histogram);
    try w.flush();

    if (nq != 0 and args.builds >= 3) {
        // The closest pair by recall, excluding the pair just printed.
        var ca: usize = 0;
        var cb: usize = 1;
        var closest = std.math.inf(f64);
        for (builds, 0..) |x, i| {
            for (builds[i + 1 ..], i + 1..) |y, j| {
                if (i == best and j == worst) continue;
                if (i == worst and j == best) continue;
                const d = @abs(x.rank(args.n_ef) - y.rank(args.n_ef));
                if (d < closest) {
                    closest = d;
                    ca = i;
                    cb = j;
                }
            }
        }
        try w.print("\n=== control: build {d} ({d:.5}) against build {d} ({d:.5}), " ++
            "recall apart by {d:.5}\n\n", .{
            ca, builds[ca].rank(args.n_ef), cb, builds[cb].rank(args.n_ef), closest,
        });
        var cr = try diff(alloc, builds[ca].view(), builds[cb].view(), &corpus);
        defer cr.deinit(alloc);
        cr.same_seed = builds[ca].seed == builds[cb].seed;
        try write(w, cr, args.histogram);
        try w.flush();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// A graph with `n` nodes, level 0 only, where node `i` points at `i+1..i+deg`.
fn chainGraph(alloc: std.mem.Allocator, n: usize, deg: usize) !Graph {
    var g = try Graph.init(alloc, hnsw.Params.fromM(4, 16, 7), n);
    g.count = n;
    g.entry_point = 0;
    for (0..n) |i| {
        const row = g.level0Slice(@intCast(i));
        for (0..deg) |k| row[k] = @intCast((i + 1 + k) % n);
    }
    return g;
}

test "two identical graphs diff to nothing" {
    var a = try chainGraph(testing.allocator, 64, 3);
    defer a.deinit();
    var b = try chainGraph(testing.allocator, 64, 3);
    defer b.deinit();

    var r = try diff(testing.allocator, View.of(&a), View.of(&b), null);
    defer r.deinit(testing.allocator);

    try testing.expectEqual(@as(u64, 0), r.level_mismatches);
    try testing.expectEqual(@as(u64, 0), r.levels[0].only_a);
    try testing.expectEqual(@as(u64, 0), r.levels[0].only_b);
    try testing.expectEqual(@as(f64, 1.0), r.levels[0].jaccard());
    try testing.expectEqual(@as(u64, 64), r.levels[0].same_nodes);
    try testing.expectEqual(@as(u64, 0), r.indeg0_a);
    try testing.expectEqual(@as(?EdgeLengths, null), r.lengths);
}

test "row order is not a difference" {
    // The slot a neighbour lands in is decided by the order the builder's
    // threads arrived in. A diff that read rows positionally would report every
    // node as differing on two graphs that are the same graph, which would make
    // the whole instrument useless while looking like a finding.
    var a = try chainGraph(testing.allocator, 32, 3);
    defer a.deinit();
    var b = try chainGraph(testing.allocator, 32, 3);
    defer b.deinit();
    for (0..32) |i| std.mem.reverse(u32, b.level0Slice(@intCast(i))[0..3]);

    var r = try diff(testing.allocator, View.of(&a), View.of(&b), null);
    defer r.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 0), r.levels[0].only_a + r.levels[0].only_b);
    try testing.expectEqual(@as(u64, 32), r.levels[0].same_nodes);
}

test "one swapped edge is one edge each way" {
    var a = try chainGraph(testing.allocator, 32, 3);
    defer a.deinit();
    var b = try chainGraph(testing.allocator, 32, 3);
    defer b.deinit();
    b.level0Slice(0)[1] = 31;

    var r = try diff(testing.allocator, View.of(&a), View.of(&b), null);
    defer r.deinit(testing.allocator);

    try testing.expectEqual(@as(u64, 1), r.levels[0].only_a);
    try testing.expectEqual(@as(u64, 1), r.levels[0].only_b);
    try testing.expectEqual(@as(u64, 31), r.levels[0].same_nodes);
    try testing.expectEqual(@as(u64, 1), r.hist[2]);
}

test "a shorter row is a difference, not a shape error" {
    // `empty_neighbour` is a suffix and a build that pruned harder leaves a
    // shorter row. That is a real difference in the graph, not a malformed one.
    var a = try chainGraph(testing.allocator, 16, 3);
    defer a.deinit();
    var b = try chainGraph(testing.allocator, 16, 3);
    defer b.deinit();
    b.level0Slice(0)[2] = empty;

    var r = try diff(testing.allocator, View.of(&a), View.of(&b), null);
    defer r.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r.levels[0].only_a);
    try testing.expectEqual(@as(u64, 0), r.levels[0].only_b);
    try testing.expectEqual(@as(u64, 1), r.hist[1]);
    try testing.expectEqual(@as(u64, 15), r.levels[0].same_nodes);
}

test "a node nothing points at is counted" {
    var a = try chainGraph(testing.allocator, 16, 1);
    defer a.deinit();
    var b = try chainGraph(testing.allocator, 16, 1);
    defer b.deinit();
    // Node 0's only in-edge is from node 15 in a 1-degree chain.
    b.level0Slice(15)[0] = 14;

    var r = try diff(testing.allocator, View.of(&a), View.of(&b), null);
    defer r.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 0), r.indeg0_a);
    try testing.expectEqual(@as(u64, 1), r.indeg0_b);
}

test "graphs of different shapes are refused" {
    var a = try chainGraph(testing.allocator, 16, 3);
    defer a.deinit();
    var b = try chainGraph(testing.allocator, 16, 3);
    defer b.deinit();
    b.count = 15;
    try testing.expectError(error.CountMismatch, diff(testing.allocator, View.of(&a), View.of(&b), null));
}

test "the default seed is the engine's" {
    // Level assignment is a pure function of the seed, so a tool building at
    // another one is building a different experiment: different upper-level
    // membership, a different entry point, a different graph to diff. The
    // literal is duplicated from `core.Config` because a bench root should not
    // widen the engine's API; this is what stops it drifting.
    const field = std.meta.fieldInfo(strawmann.core.Config, .seed);
    try testing.expectEqual(field.defaultValue().?, (Args{}).seed);
}

test "the argument parser refuses what it cannot measure" {
    // One build is not a comparison, and `k` above `ef` asks the search for
    // more results than it keeps candidates.
    try testing.expectError(error.NeedTwoBuilds, parseArgs(&.{ "g", "v.fbin", "--builds", "1" }));
    try testing.expectError(error.KAboveEf, parseArgs(&.{ "g", "v.fbin", "--k", "10", "--ef", "4" }));
    // A ladder is scored over one set of builds, and the last rung ranks them.
    try testing.expectError(error.KAboveEf, parseArgs(&.{ "g", "v.fbin", "--ef", "512,4" }));
    const ladder = try parseArgs(&.{ "g", "v.fbin", "--ef", "128,256,512" });
    try testing.expectEqual(@as(usize, 3), ladder.n_ef);
    try testing.expectEqual(@as(usize, 512), ladder.ef[ladder.n_ef - 1]);
    try testing.expectError(error.ShuffleBatchZero, parseArgs(&.{ "g", "v.fbin", "--shuffle", "0" }));
    const stable = try parseArgs(&.{ "g", "v.fbin", "--shuffle", "100", "--stable-levels" });
    try testing.expect(stable.stable_levels and stable.shuffle == 100);
    const ordered = try parseArgs(&.{ "g", "v.fbin", "--shuffle", "100", "--stable-order" });
    try testing.expect(ordered.stable_order and !ordered.stable_levels);
}

test "a reordering keeps every point exactly once" {
    // An `order` that dropped or duplicated a point would build a graph over
    // the wrong corpus and score it against ground truth that no longer
    // describes it, which would look like a finding rather than a bug.
    for ([_]usize{ 1, 7, 100, 4096 }) |batch| {
        const n = 1000;
        const perm = try arrivalOrder(testing.allocator, n, 3, batch);
        defer testing.allocator.free(perm.order);
        defer testing.allocator.free(perm.inv);

        const seen = try testing.allocator.alloc(bool, n);
        defer testing.allocator.free(seen);
        @memset(seen, false);
        for (perm.order) |v| {
            try testing.expect(!seen[v]);
            seen[v] = true;
        }
        for (seen) |x| try testing.expect(x);
        // And the inverse really inverts it.
        for (perm.order, 0..) |vec, node| try testing.expectEqual(node, perm.inv[vec]);
    }
}

test "batches survive the reordering intact" {
    // The harness's batches arrive as units: what varies between passes is
    // their order, not the order within one. A shuffle that broke them up
    // would be measuring a rearrangement the upload never performs.
    const n = 1000;
    const batch = 100;
    const perm = try arrivalOrder(testing.allocator, n, 1, batch);
    defer testing.allocator.free(perm.order);
    defer testing.allocator.free(perm.inv);
    var node: usize = 0;
    while (node < n) : (node += batch) {
        for (1..batch) |k| {
            try testing.expectEqual(perm.order[node] + @as(u32, @intCast(k)), perm.order[node + k]);
        }
    }
    try testing.expectError(error.NeedVectorFile, parseArgs(&.{ "g", "--builds", "3" }));
    try testing.expectError(error.MissingValue, parseArgs(&.{ "g", "v.fbin", "--limit" }));
    const ok = try parseArgs(&.{ "g", "v.fbin", "--builds", "4", "--metric", "dot" });
    try testing.expectEqual(@as(usize, 4), ok.builds);
    try testing.expectEqual(dist.Kernel.dot, ok.kernel);
}

test "--seeds sets one build per seed, and --oracle-entry implies --per-query" {
    const a = try parseArgs(&.{ "g", "v.fbin", "--seeds", "0x57ea3111,1,2,3", "--oracle-entry" });
    try testing.expectEqual(@as(usize, 4), a.builds);
    try testing.expectEqual(@as(u64, 0x57ea3111), a.seedOf(0));
    try testing.expectEqual(@as(u64, 3), a.seedOf(3));
    try testing.expect(a.per_query and a.oracle_entry);
    // Without --seeds every build is at --seed.
    const b = try parseArgs(&.{ "g", "v.fbin", "--seed", "7" });
    try testing.expectEqual(@as(u64, 7), b.seedOf(2));
    try testing.expectError(error.TooManySeeds, parseArgs(&.{ "g", "v.fbin", "--seeds", "1,2,3,4,5,6,7,8,9" }));
}

test "a loss is counted against the best build, by size" {
    const best = [_]u8{ 10, 10, 10, 9, 10 };
    const this = [_]u8{ 10, 9, 6, 10, 10 };
    const l = Loss.of(&best, &this, 10);
    try testing.expectEqual(@as(usize, 2), l.worse);
    try testing.expectEqual(@as(usize, 5), l.lost);
    try testing.expectEqual(@as(usize, 1), l.better);
    try testing.expectEqual(@as(usize, 1), l.gained);
    try testing.expectEqual([4]usize{ 1, 0, 0, 1 }, l.by_size);
    try testing.expectEqual(@as(usize, 2), l.imperfect);
}

test "two builds that lose the same queries overlap fully, disjoint ones not at all" {
    const best = [_]u8{ 10, 10, 10, 10 };
    const a = [_]u8{ 9, 10, 8, 10 };
    try testing.expectEqual(@as(f64, 1), worseOverlap(&best, &a, &a));
    const b = [_]u8{ 10, 9, 10, 9 };
    try testing.expectEqual(@as(f64, 0), worseOverlap(&best, &a, &b));
}
