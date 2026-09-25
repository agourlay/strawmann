//! §3, the data model, and the in-memory collection.
//!
//!   "**Collection**: name → one dense vector space. Named vectors: support the
//!    empty-name default plus N named spaces (bfb's `--vectors-per-point` > 1
//!    needs this), but optimise the single-space case.
//!    **Vector storage**: one contiguous, aligned array per space.
//!    **Deletes**: tombstone bitmap only, no compaction."
//!
//! §6.4 fixes the layout constants this file implements in memory and
//! `storage.zig` persists:
//!
//!   "**Row stride is padded to 64 B** so no vector straddles a cache line
//!    boundary it doesn't need to. For `d=768` fp32 stride is already 3072; for
//!    odd dims the padding is real and the padding bytes are zeroed (so they
//!    contribute nothing to dot/L2)."
//!
//! That last parenthesis is load-bearing and easy to lose: the kernels in
//! `dist/` read whole rows, so non-zero padding would contribute garbage to
//! every score. `addVector` zeroes it explicitly rather than relying on the
//! arena being fresh, because an update-in-place (§2's `--max-id`) writes over
//! a row that already holds another point's data.

const std = @import("std");
const builtin = @import("builtin");
const dist = @import("../dist/dist.zig");
const ids = @import("ids.zig");
const storage = @import("storage.zig");
const heap = @import("../index/heap.zig");
const hnsw = @import("../index/hnsw.zig");
const build_hnsw = @import("../index/build.zig");
const build_options = @import("build_options");
const scroll_mod = @import("scroll.zig");
const quantized = @import("quantized.zig");
const quant_mod = @import("../quant/quant.zig");
const lock = @import("../lock.zig");
const payload_mod = @import("payload.zig");

pub const Metric = dist.Metric;
pub const ExternalId = ids.ExternalId;
pub const Candidate = heap.Candidate;

/// §6.4: row stride padded to a cache line.
pub const row_align = 64;

/// Qdrant's `DEFAULT_FULL_SCAN_THRESHOLD` (`lib/segment/src/types.rs`,
/// VERIFIED at v1.19.0), in KB.
pub const default_full_scan_threshold_kb: usize = 10_000;

pub fn strideFor(dim: usize, dt: dist.Datatype) usize {
    return std.mem.alignForward(usize, dim * dt.elemSize(), row_align);
}

/// The largest dimension a non-fp32 collection accepts.
///
/// A query has to be converted into the collection's storage type once per
/// search, and §6.3 forbids allocating on the query path, so that conversion
/// lands in a fixed stack buffer. This is its size, in elements. fp32
/// collections are unaffected: they convert nothing.
///
/// A cap rather than a heap buffer because the alternative is a per-worker
/// allocation threaded through every search entry point for a case no dataset
/// in §4.2 reaches: the headline tier is d=1536.
pub const max_converted_dim = 16384;

/// §2: "**`optimizer_status` must be present and `Ok`** in `CollectionInfo`,
/// and `points_count` / `indexed_vectors_count` should be truthful."
/// No `toProto` here. It returned bare 1/2/3 while `proto/messages.zig` defined
/// `CollectionStatus` with exactly those names and values, two spellings of one
/// wire enum in two files with nothing tying them together. The mapping now
/// lives at the API boundary (`handlers.collectionStatus`), which is the layer
/// that may name both types: nothing under `core/` depends on `proto/`.
pub const Status = enum {
    green,
    yellow,
    red,
};

/// §2: "We can accept upserts into a flat unindexed buffer, report `Yellow`,
/// run a fully parallel bulk HNSW build, then flip to `Green`."
pub const IndexState = enum(u8) {
    /// No *fresh* graph, and the collection is Yellow. `indexed_vectors_count`
    /// is what the graph searches traverse still covers: 0 when there is none,
    /// its count when there is. `graph` may still be non-null here, when `invalidateIndex`
    /// stepped aside from a published graph for a rebuild, and searches keep
    /// traversing it (`choosePath`): it is still correct for the rows it
    /// covers, only behind on the tail, which `scanPendingTail` serves. Before
    /// this state brute-forced, and W11 spent its whole row in it
    /// (findings 25).
    absent,
    /// A build is running. Yellow; searches traverse the previous graph if
    /// there is one, brute force if there is not.
    building,
    /// Graph published and current to within `rebuild_ratio`. Green.
    ready,
};

pub const Config = struct {
    dim: usize,
    metric: Metric,
    /// What vectors are stored as. §5.4's working-set argument without a
    /// codebook: f16 halves the arena, u8 quarters it. Qdrant's
    /// `VectorParams.datatype`, with Qdrant's semantics (`dist/datatype.zig`).
    datatype: dist.Datatype = .float32,
    capacity: usize,
    /// §6.5 / §12: `hnsw_config { m, ef_construct, full_scan_threshold }`.
    hnsw_m: usize = 16,
    hnsw_ef_construct: usize = 100,
    /// Qdrant's `HnswConfigDiff.full_scan_threshold`, in kilobytes of vector
    /// storage: below it Qdrant 1.19 answers unfiltered queries with a plain
    /// scan instead of the graph. Applied at the API layer
    /// (`handlers.fullScanPreferred`), where the request that could have
    /// asked for `exact` is; the in-process search entry points ignore it so
    /// a unit test of the graph on a small collection tests the graph.
    hnsw_full_scan_threshold_kb: usize = default_full_scan_threshold_kb,
    /// §8.7: "**Seeded RNG** for HNSW level assignment, with the seed recorded
    /// in `meta.json`."
    seed: u64 = 0x57ea3111,
    /// Where the vector arena's bytes live: Qdrant's `VectorParams.memory`,
    /// with Qdrant's three values and Qdrant's resolution order
    /// (`storage.Placement`).
    ///
    /// `pinned` needs no file and is the default. The other two are a mapping
    /// of `<dir>/<name>.vectors.bin`, so `dir` must be set for them; the API
    /// layer refuses the combination rather than silently downgrading, because
    /// a request for `cold` that quietly ran `pinned` would publish a
    /// cold-cache number taken on a warm cache.
    placement: storage.Placement = .pinned,
    /// Where a mapped arena's file goes, as `<dir>/<name>.vectors.bin`.
    /// Ignored when `placement` is `pinned`.
    ///
    /// Borrowed, not owned: `Collection` stores this whole `Config`, so the
    /// slice must outlive the collection. The server passes `argv`, which
    /// outlives everything.
    ///
    /// Note this is *not* `persist`'s layout, which is one directory per
    /// collection containing a bare `vectors.bin`. A live arena is one file per
    /// collection in a shared directory, because the server holds many
    /// collections at once and a snapshot holds one. `persist.load` refuses a
    /// mapped placement rather than pretending the two agree.
    dir: ?[]const u8 = null,
    /// §5.5, back the vector arena with 2 MiB pages.
    ///
    /// Declared rather than ambient: §7.1 requires the environment be stated,
    /// and an env var read deep inside the allocator would be exactly the
    /// hidden variable that makes two result rows incomparable. It is a knob so
    /// the two arms of an A/B can run on one binary against one dataset -
    /// comparing separately-built servers confounds page size with everything
    /// else that differed.
    huge_pages: bool = true,
};

pub const Error = error{
    DimensionMismatch,
    CapacityExceeded,
    MapFull,
    OutOfMemory,
    /// A payload blob past `payload.max_blob`, or malformed map entries.
    PayloadRejected,
    /// `CreateFieldIndex` on a field already indexed as another type.
    IndexKindMismatch,
    /// A non-fp32 datatype at a dimension whose converted query would not fit
    /// the fixed stack buffer the query path uses (§6.3 forbids allocating
    /// there). fp32 collections are never affected.
    DimensionTooLargeForDatatype,
    /// A `cached` or `cold` placement with nowhere to put the file. Refused
    /// rather than downgraded to `pinned`: silently serving a cold-placement
    /// request from pinned memory would publish a page-fault result measured
    /// on memory that never faults.
    PlacementNeedsDirectory,
    /// The arena file could not be created or mapped, so the requested
    /// placement is not available on this host. Same argument as above for why
    /// this is an error and not a fallback.
    PlacementUnavailable,
};

/// One dense vector space.
pub const VectorSpace = struct {
    dim: usize,
    stride: usize,
    /// What each element is stored as. The arena is bytes either way; this is
    /// how to read them.
    datatype: dist.Datatype,
    /// Row-major arena, `capacity * stride` bytes, 64-byte aligned.
    arena: []align(row_align) u8,
    /// Where the arena's bytes live. `pinned` is the historical and default
    /// case: the arena is heap memory this process allocated. The other two are
    /// a mapping of `backing`, and `region` owns it.
    placement: storage.Placement = .pinned,
    /// Set exactly when `placement.isMapped()`. `deinit` unmaps rather than
    /// frees, and a `Region` that outlived its fd would be a segfault on first
    /// touch, so the fd is held for the arena's whole life.
    region: ?storage.Region = null,
    backing: ?std.os.linux.fd_t = null,

    pub fn init(alloc: std.mem.Allocator, dim: usize, dt: dist.Datatype, capacity: usize, huge_pages: bool) !VectorSpace {
        const stride = strideFor(dim, dt);
        const bytes = try alloc.alignedAlloc(u8, .@"64", stride * capacity);
        if (huge_pages) adviseHugePages(bytes);
        // Zero the whole arena once so that padding bytes are correct even for
        // rows never written, which matters for a brute-force scan over a
        // partially-filled collection.
        //
        // This must run *after* the madvise: it is the first touch, so it is
        // where the pages are faulted in, and the kernel decides page size at
        // fault time. Zeroing first gives 4 KiB pages that khugepaged may or may
        // not collapse later.
        @memset(bytes, 0);
        return .{ .dim = dim, .stride = stride, .datatype = dt, .arena = bytes };
    }

    /// An arena backed by a file, under `cached` or `cold`.
    ///
    /// The file is created at full capacity with `fallocate` (§6.4: "The arena
    /// is preallocated with `fallocate` at collection creation"), so no write
    /// during the run triggers block allocation. That preallocation is what
    /// keeps this inside §1: the non-goal is *disk-resident (larger-than-RAM)
    /// operation*, meaning an engine that works when the data does not fit.
    /// Here it always fits; the placement decides where the pages live and who
    /// may evict them, not whether the collection is bounded by RAM.
    ///
    /// Deliberately *not* zeroed. `fallocate` mode 0 already gives zeroed
    /// extents, and touching every page here would prefault the whole arena,
    /// which is precisely what `cold` exists not to do — a `@memset` would turn
    /// the cold arm into the cached arm and nothing would fail.
    pub fn initMapped(
        dim: usize,
        dt: dist.Datatype,
        capacity: usize,
        metric: dist.Metric,
        path: []const u8,
        placement: storage.Placement,
    ) !VectorSpace {
        std.debug.assert(placement.isMapped());
        const stride = strideFor(dim, dt);
        const size = stride * capacity;

        // Unlink first, so the arena is guaranteed to start zeroed.
        //
        // `openSized` opens with O_CREAT and no O_TRUNC, and `fallocate` mode 0
        // allocates without zeroing what is already there. Re-creating a
        // collection of the same name in the same `--data-dir` — which is
        // exactly what bfb's Delete-then-Create setup does on every run — would
        // otherwise map the previous run's bytes. The rows themselves are
        // written before they are read, so that would not show up as obvious
        // garbage; the *padding* between `dim * elemSize` and `stride` would be
        // stale, and the padding is what the SIMD kernels read to the end of
        // their vector width. Wrong distances, from a file, on the second run
        // only. `VectorSpace.init` zeroes for the same reason.
        storage.removeFile(path);

        const fd = try storage.openSized(path, storage.header_size + size, true);
        errdefer storage.closeFd(fd);

        var h = storage.Header{
            .magic = storage.magic,
            .format_version = storage.format_version,
            .kind = @intFromEnum(storage.Header.Kind.vectors),
            .dim = dim,
            .count = 0,
            .capacity = capacity,
            .stride = stride,
            // The metric, because for a `.vectors` file that is what
            // `kind_field` means (`storage.Header.metricField`). Writing 0 here
            // produced a header that passed every structural check and named a
            // metric that does not exist, which is precisely the "plausible
            // garbage" the kind discriminator exists to prevent.
            .kind_field = @intCast(metric.toProto()),
            .flags = if (dt.normalisesAtIngest(metric))
                storage.Header.Flags.normalised
            else
                0,
            .seed = 0,
            .datatype = @intFromEnum(dt),
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);

        const region = try storage.mapRegion(fd, storage.header_size, size, placement);
        // `mapRegion` returns 4 KiB alignment; the arena's contract is 64, and
        // a page boundary satisfies it. The cast is checked here rather than
        // assumed, because every `rowBytes` offset is computed from this base.
        std.debug.assert(@intFromPtr(region.bytes.ptr) % row_align == 0);
        const arena: []align(row_align) u8 = @alignCast(region.bytes[0..size]);

        return .{
            .dim = dim,
            .stride = stride,
            .datatype = dt,
            .arena = arena,
            .placement = placement,
            .region = region,
            .backing = fd,
        };
    }

    /// Ask for 2 MiB pages behind the vector arena (§5.5).
    ///
    /// The arena is the single largest and most randomly-accessed allocation in
    /// the process, which makes it exactly the case transparent huge pages
    /// exist for. At the headline tier it is 6.8 GB, **1.7 million 4 KiB
    /// pages** against an L2 dTLB holding a few thousand entries, so in the
    /// steady state essentially every vector fetch also costs a page walk.
    ///
    /// `storage.zig` already does this for the mmap path; the resident arena
    /// did not, and the resident arena is what every benchmark so far has
    /// actually used. Measured on the running headline collection before the
    /// fix: `AnonHugePages: 0 kB` against 6.9 GB of anonymous memory.
    ///
    /// §5.5's own microbenchmark puts the win at ~16% of access latency for a
    /// ~1000x reduction in dTLB misses. That is a smaller effect than the
    /// page-walk count suggests, because Zen 5 overlaps walks well, which is
    /// why this is advisory and unchecked rather than something to depend on.
    ///
    /// Advisory in the strict sense too: with `/sys/kernel/mm/
    /// transparent_hugepage/enabled` set to `never` this does nothing, and on a
    /// fragmented system the kernel may decline. Nothing here is load-bearing
    /// for correctness, so the return value is deliberately ignored.
    fn adviseHugePages(bytes: []u8) void {
        if (builtin.os.tag != .linux) return;
        const two_mib = 2 * 1024 * 1024;
        if (bytes.len < two_mib) return;

        // Advise only the 2 MiB-aligned interior. The kernel can only back
        // naturally-aligned 2 MiB regions with a huge page, so advising the
        // ragged ends achieves nothing and asking about memory the allocator
        // may share with other objects is impolite.
        const start = std.mem.alignForward(usize, @intFromPtr(bytes.ptr), two_mib);
        const end = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr) + bytes.len, two_mib);
        if (end <= start) return;
        _ = std.os.linux.madvise(@ptrFromInt(start), end - start, std.os.linux.MADV.HUGEPAGE);
    }

    pub fn deinit(self: *VectorSpace, alloc: std.mem.Allocator) void {
        if (self.region) |*r| {
            r.deinit();
            if (self.backing) |fd| storage.closeFd(fd);
            self.region = null;
            self.backing = null;
            self.arena = &.{};
            return;
        }
        alloc.free(self.arena);
    }

    /// The raw bytes of one row, without interpreting them.
    ///
    /// §6.4: "Point *n* lives at `base + n × stride`, computed, never looked
    /// up." An out-of-range offset is a silent read of a neighbouring row in
    /// ReleaseFast: plausible data, wrong answer, no crash.
    pub fn rowBytes(self: *const VectorSpace, offset: u32) []const u8 {
        std.debug.assert(@as(usize, offset) * self.stride + self.stride <= self.arena.len);
        const start = @as(usize, offset) * self.stride;
        return self.arena[start..][0 .. self.dim * self.datatype.elemSize()];
    }

    /// §6.4: "Point *n* lives at `base + n × stride`, computed, never looked up."
    ///
    /// fp32 only, and asserted rather than checked: a caller that reaches for
    /// f32 rows on a f16 collection has a type confusion, not a runtime
    /// condition to handle.
    pub fn row(self: *const VectorSpace, offset: u32) []f32 {
        std.debug.assert(self.datatype == .float32);
        std.debug.assert(@as(usize, offset) * self.stride + self.stride <= self.arena.len);
        const start = @as(usize, offset) * self.stride;
        const bytes = self.arena[start..][0 .. self.dim * @sizeOf(f32)];
        return @alignCast(std.mem.bytesAsSlice(f32, bytes));
    }

    pub fn rowConst(self: *const VectorSpace, offset: u32) []const f32 {
        return self.row(offset);
    }

    pub fn rowF16(self: *const VectorSpace, offset: u32) []const f16 {
        std.debug.assert(self.datatype == .float16);
        return @alignCast(std.mem.bytesAsSlice(f16, self.rowBytes(offset)));
    }

    pub fn rowU8(self: *const VectorSpace, offset: u32) []const u8 {
        std.debug.assert(self.datatype == .uint8);
        return self.rowBytes(offset);
    }

    /// Widen a row into `dst`, whatever it is stored as.
    ///
    /// For the paths that genuinely need fp32 and are not the search path:
    /// persistence, codebook training, and the tests that compare storage
    /// types against each other. It is a conversion, so it is never on the
    /// hot path, and callers that reach for it there are making a mistake this
    /// comment cannot prevent.
    ///
    /// Not covered by `beginInPlace`'s drain: the callers are the build
    /// thread (`quantize`), which holds no `SearchGuard`, so the first
    /// overwrite can land while this is mid-row with the flag unobserved.
    /// That torn row is harmless here, and only here: it went on
    /// `overwrite_log`, and `buildIndex` re-encodes every logged row from
    /// the arena, under the write lock, before the store is served.
    pub fn readIntoGuarded(self: *const VectorSpace, coll: *const Collection, offset: u32, dst: []f32) []const f32 {
        if (!coll.in_place_updates.load(.acquire)) return self.readInto(offset, dst);
        while (true) {
            const v1 = coll.row_versions[offset].load(.acquire);
            if (v1 & 1 != 0) {
                std.atomic.spinLoopHint();
                continue;
            }
            const out = self.readInto(offset, dst);
            if (coll.row_versions[offset].load(.acquire) == v1) return out;
        }
    }

    pub fn readInto(self: *const VectorSpace, offset: u32, dst: []f32) []const f32 {
        std.debug.assert(dst.len >= self.dim);
        const out = dst[0..self.dim];
        switch (self.datatype) {
            .float32 => @memcpy(out, self.rowConst(offset)),
            .float16 => for (self.rowF16(offset), out) |x, *d| {
                d.* = @floatCast(x);
            },
            .uint8 => for (self.rowU8(offset), out) |x, *d| {
                d.* = @floatFromInt(x);
            },
        }
        return out;
    }

    /// Copy a vector into its row, converting to the storage type and zeroing
    /// any stride padding.
    ///
    /// The conversions are Qdrant's, verified against their source: f16 is a
    /// round-to-nearest cast, and u8 is a *saturating truncation* rather than
    /// a validated range, so out-of-range components are silently clamped
    /// exactly as the real server clamps them (`dist/datatype.zig`).
    fn write(self: *VectorSpace, offset: u32, v: []const f32) void {
        std.debug.assert(v.len == self.dim);
        const start = @as(usize, offset) * self.stride;
        const dst = self.arena[start..][0..self.stride];
        const used = self.dim * self.datatype.elemSize();
        switch (self.datatype) {
            .float32 => @memcpy(dst[0..used], std.mem.sliceAsBytes(v)),
            .float16 => dist.datatype.convertToF16(
                @alignCast(std.mem.bytesAsSlice(f16, dst[0..used])),
                v,
            ),
            .uint8 => dist.datatype.convertToU8(dst[0..used], v),
        }
        // §6.4: padding bytes are zeroed so they contribute nothing to dot/L2.
        // Required on every write, not just the first: an update-in-place lands
        // on a row that may already hold a different point's bytes.
        @memset(dst[used..], 0);
    }
};

/// An in-memory collection. §6.4 persistence lives in `storage.zig`.
pub const Collection = struct {
    alloc: std.mem.Allocator,
    name: []u8,
    config: Config,

    space: VectorSpace,
    id_space: ids.IdSpace,

    /// §3: "**Deletes**: tombstone bitmap only, no compaction."
    deleted: std.DynamicBitSet,
    deleted_count: usize = 0,

    /// §3: "**Payload**: opaque, append-only blob per point", and M7's
    /// keyword index. Written under `write_lock` like the arena; read
    /// lock-free like the arena (`payload.zig`).
    payload: payload_mod.Store,

    /// Points covered by the HNSW graph. Below `count()` while a build is
    /// pending, which is exactly what `indexed_vectors_count` must report.
    indexed_count: usize = 0,

    /// Monotonic operation counter, returned as `UpdateResult.operation_id`.
    operation_id: u64 = 0,

    /// §6.5, the HNSW graph, built in bulk once ingest stops.
    graph: ?*hnsw.Graph = null,
    /// `graph.?.count`, published atomically alongside `graph` (0 when there
    /// is no graph: a graph is only ever built over a non-empty collection).
    ///
    /// For the readers that need the number and not the graph: `needsRebuild`
    /// runs on the upsert handler's thread *without* `SearchGuard`, and it
    /// used to dereference `graph` for `count`, which a concurrent rebuild
    /// could retire and `reclaimRetired` free between the load and the
    /// read. A scalar that travels with the pointer needs no guard.
    graph_count: std.atomic.Value(usize) = .init(0),
    /// Guards `graph` publication. A search reads it with acquire ordering, so
    /// a graph is either fully built and visible or not visible at all; there
    /// is no window in which a search can traverse a half-built graph.
    index_state: std.atomic.Value(IndexState) = .init(.absent),
    build_ns: u64 = 0,
    /// Guards every mutation of the collection: the id space, the arena, the
    /// tombstone bitmap and the counters.
    ///
    /// §6.3 forbids allocation on the *query* path and §6's diagram marks
    /// storage "immutable during search", but nothing makes ingest single-
    /// threaded: the worker pool pulls Upsert RPCs from one queue onto N
    /// threads, so two concurrent upserts to the same collection both read
    /// `id_space.next`, both claim the same offset, and one vector is silently
    /// overwritten while an external id resolves to a stranger's row. The
    /// counters race the same way, which can lose the increment that keeps
    /// `IdMap.getOrInsert` from probing forever.
    ///
    /// Search does not take this lock. It reads a graph published with release
    /// ordering and an arena whose rows are only ever appended, so the cost
    /// falls entirely on ingest, which §2's Yellow→Green flow already
    /// separates from querying. The one read path that does take it is
    /// `scrollOrder`, deliberately, to build the id order once per mutation
    /// epoch without racing a publish.
    write_lock: lock.Mutex = .{},

    /// §6.7, quantized codes, when a mode is configured. `null` until the
    /// first build on a quantized collection.
    ///
    /// A pointer, published atomically, rather than the `Store` by value it
    /// used to be. `Store` is a tagged union several words wide, and
    /// `quantize` assigned it whole while `searchQuantized` read it whole from
    /// another thread with no lock between them, so a reader could observe
    /// the new tag with the old payload (or half of each): a `.scalar` tag
    /// over a `.binary` codes slice, dereferenced as scalar codes. A pointer
    /// swap is one word, and the store behind it is immutable once published
    /// and retired-not-freed on replacement, exactly like `graph`.
    quant: std.atomic.Value(?*quantized.Store) = .init(null),
    /// The mode requested at collection creation, applied after the bulk
    /// build. Kept separate from `quant` so that "configured" and "built" are
    /// distinguishable, a collection that asked for binary but has not been
    /// indexed yet must not silently search fp32 while claiming otherwise.
    quant_mode: quant_mod.Mode = .none,
    /// `ScalarQuantization.quantile`, the central fraction of the component
    /// distribution the SQ8 range covers. Only `.scalar` reads it. The API
    /// layer sets it from the request (absent there means Qdrant's 1.0, plain
    /// min/max); in-process callers get bfb's 0.99, which is what every
    /// measured row used.
    quant_quantile: f32 = quant_mod.scalar.default_quantile,

    /// Graphs and quantized stores replaced by a rebuild, kept alive until no
    /// search can still be holding one.
    ///
    /// A rebuild cannot free the artefacts it replaces. A search reads
    /// `index_state` with acquire ordering and *then* dereferences `graph`, so
    /// a rebuild that freed the old graph in between would pull it out from
    /// under a traversal already in progress, a use-after-free reachable by
    /// W11 ("mixed read/write"), which is a workload the spec explicitly runs.
    ///
    /// Retired entries are freed by `reclaimRetired`, from the build thread
    /// after the next publish, once `active_searches` reads zero; whatever is
    /// still here at `deinit` is freed there, when no searches can be running.
    /// The builder live insertion runs through, made on first use and kept.
    ///
    /// One per collection rather than one per worker, because every live insert
    /// happens under `write_lock`: there is exactly one writer, so the builder
    /// runs unsynchronised exactly as the serial build does. Null until a point
    /// is appended to a collection that already has a published graph, which on
    /// most collections is never.
    live_builder: ?build_hnsw.Builder = null,

    retired_graphs: std.ArrayList(*hnsw.Graph) = .empty,
    retired_quant: std.ArrayList(*quantized.Store) = .empty,
    /// Searches currently inside `search` or `searchQuantized`.
    ///
    /// The reclamation rule in one sentence: a search increments this *before*
    /// reading `graph` or `quant`, so once the counter reaches zero after a
    /// replacement has been published, every search that could still be
    /// holding the old pointer has finished and it can be freed.
    ///
    /// This is quiescent-state reclamation with a single counter rather than
    /// epochs or hazard pointers. It is enough here because there is exactly
    /// one writer (the build thread), replacements are rare, and the readers
    /// are short: the longest is W9's exact scan at ~28 ms.
    active_searches: std.atomic.Value(usize) = .init(0),
    /// Requests currently holding a pointer to this collection.
    ///
    /// One level up from `active_searches`, which protects a *graph* from
    /// being freed under a traversal. This protects the *collection* from
    /// being freed under a request: `Delete` runs on a worker thread like
    /// every other RPC, so a search and a drop can overlap, and the drop used
    /// to free the arena the search was reading. Held for the whole handler,
    /// not just the search, because an upsert or a scroll dereferences the
    /// same pointer.
    users: std.atomic.Value(usize) = .init(0),
    /// Set by `Engine.drop` once the collection is unlinked, so a build still
    /// running on it stops rather than finishing for nothing: the drop waits
    /// on `users`, and a bulk build is one for as long as it runs. The W1
    /// read-back started a 240 s build of `bench1` seconds before its drop on
    /// every 0925 pass, and the drop sat through all of it.
    drop_requested: std.atomic.Value(bool) = .init(false),
    /// Per-row write counter, odd while that row is being overwritten.
    ///
    /// Only consulted once `in_place_updates` is set, so an append-only
    /// collection never touches this array and never pays a cache line for it.
    row_versions: []std.atomic.Value(u32) = &.{},
    /// Whether any row has ever been overwritten in place.
    ///
    /// A search reads rows without the write lock, which is safe for as long
    /// as rows are written once and then published: `count()` orders the
    /// contents. An *update* breaks that, because the row is already visible.
    /// This flag is what tells readers to start checking, and it is set (and
    /// drained for) before the first such write rather than after it.
    in_place_updates: std.atomic.Value(bool) = .init(false),

    /// Handle for an in-flight background build.
    ///
    /// The build thread writes into this collection's arenas, so it must be
    /// joined before the collection is freed. A detached thread would keep
    /// running against freed memory, a use-after-free that presents as a
    /// leak report if you are lucky and as corruption if you are not.
    build_thread: ?std.Thread = null,
    /// Serialises `build_thread`'s join-then-store in `ensureIndexBuilding`.
    ///
    /// Winning the `.absent -> .building` compare-exchange elects one
    /// spawner per build, but not one at a time across builds: a fast build
    /// can complete and flip the state back to `.absent` before its spawner
    /// has stored the handle, so a second poll wins the next exchange, reads
    /// the *previous* handle (or none), and the first spawner then overwrites
    /// the second's. One handle is joined twice, the other never, and the
    /// never-joined thread outlives the collection. Holding this across the
    /// join and the store makes the sequence "join what is there, spawn,
    /// store" indivisible, whatever the state machine does meanwhile.
    build_thread_lock: lock.Mutex = .{},

    /// Offsets sorted by external id, built lazily for `Scroll` and dropped on
    /// any mutation.
    ///
    /// Scroll is defined over **point-id order**, and nothing else in the
    /// engine needs that order: §3 stores points in arrival order and every
    /// other read path is either by id (hash) or by similarity (graph). Walking
    /// arrival order instead would be right only when ids happen to be assigned
    /// sequentially, which is exactly what bfb's default upload does, so the
    /// bug would pass every test we would naturally write and then produce
    /// silently wrong pages for anyone using UUIDs or a sparse id space. §1
    /// forbids the silent degradation, so the order is materialised.
    ///
    /// Published atomically and retired-not-freed on drop, for the reason
    /// `retired_graphs` gives: it used to be a plain slice, built by whichever
    /// reader got there first with no lock and freed by `upsert` under the
    /// write lock, so W13 (`-p 8` scroll) racing any write freed the order
    /// under a reader mid-page. Readers hold `SearchGuard` across a page, and
    /// `dropScrollOrder` frees a retired order only when no reader can be
    /// inside one.
    scroll_state: scroll_mod.Scroll = .{},
    /// Points overwritten in place since the published graph was built.
    ///
    /// `needsRebuild` counts the *appended* tail as `total - graph.count`,
    /// which is zero for an overwrite: the row is already in the graph, at the
    /// position its old vector had. Search stays correct in the meantime, the
    /// scorer reads the arena, so an overwritten point is ranked by its new
    /// vector wherever the traversal reaches it, but its edges were chosen for
    /// the old one and stop being useful as it moves. bfb's default upload
    /// into a populated collection (W11: ids from `--offset` 0) is *all*
    /// overwrites, so without this the graph never rebuilt however far every
    /// point had moved. Reset when a build publishes.
    overwritten_since_build: std.atomic.Value(usize) = .init(0),
    /// Rows overwritten while a build is running, one bit per offset.
    ///
    /// The builder reads the arena over tens of seconds and publishes at the
    /// end. A row overwritten after the builder read it but before it
    /// published is in the new graph and the new codes at its *old* position,
    /// and `noteOverwrite` re-encoded only the store that was current at the
    /// time, the one about to be retired. Recording the offset lets the
    /// builder re-encode those rows into the store it is about to publish and
    /// count them as pending for the graph it is about to publish. Written by
    /// `noteOverwrite` and read/cleared by `buildIndex`, both under
    /// `write_lock`; `overwrite_log_active` says whether a build is between
    /// its first arena read and its publish.
    overwrite_log: std.DynamicBitSet,
    overwrite_log_active: bool = false,
    /// Test hook: called by `buildIndex` after the graph and codes are built
    /// and before either is published, so a test can land an overwrite in
    /// exactly the read-to-publish window without racing a thread.
    build_hook: ?*const fn (*Collection) void = null,

    /// Counts of vectors the cosine short-circuit left untouched, for §8.3's
    /// conformance check. A divergence from Qdrant's count localises a
    /// normalisation bug immediately instead of leaving it as a score delta.
    normalized_count: usize = 0,
    short_circuited_count: usize = 0,

    pub fn init(alloc: std.mem.Allocator, name: []const u8, config: Config) !Collection {
        if (config.datatype != .float32 and config.dim > max_converted_dim) {
            return Error.DimensionTooLargeForDatatype;
        }
        const owned = try alloc.dupe(u8, name);
        errdefer alloc.free(owned);

        var space = if (config.placement.isMapped()) blk: {
            const dir = config.dir orelse return Error.PlacementNeedsDirectory;
            var pbuf: [512]u8 = undefined;
            const path = std.fmt.bufPrint(&pbuf, "{s}/{s}.vectors.bin", .{ dir, name }) catch
                return Error.PlacementNeedsDirectory;
            break :blk VectorSpace.initMapped(
                config.dim,
                config.datatype,
                config.capacity,
                config.metric,
                path,
                config.placement,
            ) catch return Error.PlacementUnavailable;
        } else try VectorSpace.init(alloc, config.dim, config.datatype, config.capacity, config.huge_pages);
        errdefer space.deinit(alloc);

        var id_space = try ids.IdSpace.init(alloc, config.capacity);
        errdefer id_space.deinit(alloc);

        const deleted = try std.DynamicBitSet.initEmpty(alloc, config.capacity);
        errdefer {
            var d = deleted;
            d.deinit();
        }

        // 4 bytes per point, against 512 for the vector itself at d=128. Not
        // allocated lazily on the first update: that would put an allocation
        // (and a failure path) inside a write that has already been accepted.
        const versions = try alloc.alloc(std.atomic.Value(u32), config.capacity);
        errdefer alloc.free(versions);
        for (versions) |*v| v.* = .init(0);

        // One bit per row, allocated up front for the same reason as
        // `row_versions`: the write that needs it has already been accepted.
        var overwrite_log = try std.DynamicBitSet.initEmpty(alloc, config.capacity);
        errdefer overwrite_log.deinit();

        const payload = try payload_mod.Store.init(alloc, config.capacity);

        return .{
            .alloc = alloc,
            .name = owned,
            .config = config,
            .space = space,
            .id_space = id_space,
            .deleted = deleted,
            .payload = payload,
            .row_versions = versions,
            .overwrite_log = overwrite_log,
        };
    }

    pub fn deinit(self: *Collection) void {
        // Join before freeing anything the build touches. Callers must have
        // stopped accepting requests first, so no new build can start here.
        if (self.build_thread) |t| {
            t.join();
            self.build_thread = null;
        }
        // Before the graph it points at, though it owns none of it: the
        // builder holds six of its own allocations and nothing else frees them.
        if (self.live_builder) |*b| {
            b.deinit(self.alloc);
            self.live_builder = null;
        }
        if (self.graph) |g| {
            g.deinit();
            self.alloc.destroy(g);
        }
        for (self.retired_graphs.items) |g| {
            g.deinit();
            self.alloc.destroy(g);
        }
        self.retired_graphs.deinit(self.alloc);
        for (self.retired_quant.items) |q| destroyStore(self.alloc, q);
        self.retired_quant.deinit(self.alloc);
        if (self.quant.load(.acquire)) |q| destroyStore(self.alloc, q);
        self.scroll_state.deinit(self.alloc);
        self.alloc.free(self.name);
        self.space.deinit(self.alloc);
        self.alloc.free(self.row_versions);
        self.id_space.deinit(self.alloc);
        self.deleted.deinit();
        self.overwrite_log.deinit();
        self.payload.deinit(self.alloc);
    }

    pub fn count(self: *const Collection) usize {
        return self.id_space.count() - self.deleted_count;
    }

    /// §2: "`wait_index` drives indexing semantics... We can accept upserts
    /// into a flat unindexed buffer, report `Yellow`, run a fully parallel bulk
    /// HNSW build, then flip to `Green`."
    ///
    /// **Derived, never stored.** A stored status is a second source of truth
    /// that drifts: a fresh collection would start Green, accept a million
    /// unindexed points, and still report Green, which is exactly the state §2
    /// warns makes the comparison meaningless ("we must report `Green` only
    /// when actually indexed"). Computing it from the graph state makes that
    /// drift unrepresentable.
    ///
    /// Green with a small unindexed tail. A published graph plus a tail under
    /// `rebuild_ratio` is Green, not Yellow, because the tail is served
    /// exhaustively (`scanPendingTail`), so every result is complete and
    /// §2's condition, "report `Green` only when actually indexed or the
    /// comparison is meaningless", is met in the sense that matters: the
    /// answers are the answers an indexed collection gives. It is also what
    /// Qdrant reports: a collection whose newest segment is below
    /// `indexing_threshold` is Green with `indexed_vectors_count <
    /// points_count`, and bfb's poll loop reads exactly that. Reporting Yellow
    /// here instead wedged the loop, since a tail under the ratio never
    /// triggers a rebuild, so `wait_index` after any post-build write waited
    /// forever for a Green that could not come. `indexed_vectors_count` still
    /// tells the truth about the tail.
    pub fn status(self: *const Collection) Status {
        // An empty collection is trivially fully indexed.
        if (self.id_space.count() == 0) return .green;
        if (self.index_state.load(.acquire) != .ready) return .yellow;
        // Past the ratio the collection is about to step aside for a rebuild
        // (`invalidateIndex`), and until it does the answers are still
        // complete, but the cost is no longer that of an indexed collection.
        return if (needsRebuild(self)) .yellow else .green;
    }

    /// Upsert one point.
    ///
    /// `vec` is consumed by value into the arena; the caller may reuse its
    /// buffer immediately. Returns the internal offset.
    ///
    /// The metric's ingest preprocessing (§8.3: cosine normalisation, with its
    /// short-circuit) happens here and nowhere else, so the stored form is
    /// always the form the search path expects.
    pub fn upsert(self: *Collection, id: ExternalId, vec: []f32) Error!u32 {
        return self.upsertWithPayload(id, vec, &.{}, 0);
    }

    /// `upsert`, with the point's payload: the wire bytes holding the
    /// `map<string, Value>` entries under `map_field` (`PointStruct.payload`
    /// is 3). Empty means no payload, and an upsert *replaces* the payload
    /// whole, as Qdrant's does, so a bare upsert of an existing point clears
    /// what it had. The blob is written before the row is published, so a
    /// reader that can see the point can see its payload.
    pub fn upsertWithPayload(self: *Collection, id: ExternalId, vec: []f32, payload: []const u8, map_field: u32) Error!u32 {
        if (vec.len != self.config.dim) return Error.DimensionMismatch;
        std.debug.assert(self.id_space.count() <= self.config.capacity);

        self.write_lock.lock();
        defer self.write_lock.unlock();
        scroll_mod.dropScrollOrder(self);

        const r = self.id_space.reserve(id) catch |e| return switch (e) {
            error.MapFull => Error.MapFull,
            error.CapacityExceeded => Error.CapacityExceeded,
        };

        // §8.3 normalises cosine at ingest so the search path is a plain dot
        // product, but that rule belongs to the *storage type*, not to the
        // metric alone: a uint8 collection cannot hold normalised components,
        // since everything in [-1, 1] truncates to 0 or 1. Qdrant's byte
        // cosine therefore preprocesses with the identity and pays for the
        // norms per comparison, and this matches it (`dist/datatype.zig`).
        if (self.config.datatype.normalisesAtIngest(self.config.metric)) {
            if (dist.norm.preprocessInPlace(self.config.metric, vec)) {
                self.normalized_count += 1;
            } else {
                self.short_circuited_count += 1;
            }
        }

        if (r.is_new) {
            // Write first, publish second. A reader's bound is `count()`, so
            // until `publish` returns nothing can see this row and the write
            // needs no guard.
            self.space.write(r.offset, vec);
            self.payload.set(self.alloc, r.offset, payload, map_field) catch |e| return switch (e) {
                error.OutOfMemory => Error.OutOfMemory,
                else => Error.PayloadRejected,
            };
            self.id_space.publish(id, r.offset) catch |e| return switch (e) {
                error.MapFull => Error.MapFull,
                error.CapacityExceeded => Error.CapacityExceeded,
            };
            // `-Dlive-insert`: join the published graph now instead of waiting
            // in the pending tail for a rebuild of the whole corpus. Off by
            // default until W11 says what the trade is worth (decisions.md).
            if (build_options.live_insert) liveInsert(self, r.offset);
        } else {
            // An in-place overwrite of a row a search may be reading right
            // now. `beginInPlace` makes that visible to readers and drains the
            // ones already inside; the version bump either side of the write
            // is what lets a reader detect that it read across it.
            beginInPlace(self);
            const ver = &self.row_versions[r.offset];
            const v = ver.load(.monotonic);
            ver.store(v + 1, .release);
            self.space.write(r.offset, vec);
            ver.store(v + 2, .release);
            noteOverwrite(self, r.offset, vec);
            // The slot swap is atomic on its own; a reader sees the old blob
            // or the new one, never a torn one.
            self.payload.set(self.alloc, r.offset, payload, map_field) catch |e| return switch (e) {
                error.OutOfMemory => Error.OutOfMemory,
                else => Error.PayloadRejected,
            };
        }

        // An overwrite of a tombstoned point resurrects it, which is what an
        // upsert means.
        if (self.deleted.isSet(r.offset)) {
            self.deleted.unset(r.offset);
            self.deleted_count -= 1;
        }

        // A write that lands while a build is running needs no mark: the
        // build publishes under this lock and then asks `needsRebuild` with
        // the count this write has already advanced (`buildIndex`).

        return r.offset;
    }

    /// `SetPayload` on one point: merge `payload`'s entries over what it has.
    pub fn setPayload(self: *Collection, id: ExternalId, payload: []const u8, map_field: u32) Error!bool {
        self.write_lock.lock();
        defer self.write_lock.unlock();
        const off = self.id_space.lookup(id) orelse return false;
        if (self.deleted.isSet(off)) return false;
        self.payload.merge(self.alloc, off, payload, map_field) catch |e| return switch (e) {
            error.OutOfMemory => Error.OutOfMemory,
            else => Error.PayloadRejected,
        };
        return true;
    }

    /// `CreateFieldIndex`: index `name` as `kind` over every point so far.
    pub fn createPayloadIndex(self: *Collection, name: []const u8, kind: payload_mod.Kind) Error!void {
        self.write_lock.lock();
        defer self.write_lock.unlock();
        self.payload.createIndex(self.alloc, name, kind, self.id_space.count()) catch |e| return switch (e) {
            error.OutOfMemory => Error.OutOfMemory,
            error.IndexKindMismatch => Error.IndexKindMismatch,
            else => Error.PayloadRejected,
        };
    }

    pub fn delete(self: *Collection, id: ExternalId) bool {
        self.write_lock.lock();
        defer self.write_lock.unlock();
        const off = self.id_space.lookup(id) orelse return false;
        scroll_mod.dropScrollOrder(self);
        if (self.deleted.isSet(off)) return false;
        self.deleted.set(off);
        self.deleted_count += 1;
        return true;
    }

    pub fn isDeleted(self: *const Collection, offset: u32) bool {
        return self.deleted.isSet(offset);
    }

    pub fn nextOperationId(self: *Collection) u64 {
        self.write_lock.lock();
        defer self.write_lock.unlock();
        self.operation_id += 1;
        return self.operation_id;
    }

    /// Preprocess a *query* vector the same way ingest preprocesses a stored
    /// one, into caller-provided scratch.
    ///
    /// Sharing the funnel with `upsert` is what guarantees "did we normalise
    /// the query the same way we normalised the data?" has one answer. A cosine
    /// collection whose queries were not normalised returns plausible but
    /// systematically wrong scores.
    ///
    /// The rule is the *datatype's*, not the metric's, for the same reason it
    /// is in `upsert`: a uint8 cosine collection stores raw bytes and pays for
    /// the norms per comparison (`dist/datatype.zig`). This used to ask the
    /// metric alone, so it normalised the query anyway; `Probe.init` then
    /// converted the unit query to u8 and every component but the largest
    /// truncated to 0, so every score was 0. Same funnel, same answer.
    pub fn prepareQuery(self: *const Collection, scratch: []f32, q: []const f32) Error![]const f32 {
        if (q.len != self.config.dim) return Error.DimensionMismatch;
        if (!self.config.datatype.normalisesAtIngest(self.config.metric)) return q;
        _ = dist.norm.normalizeInto(scratch[0..q.len], q);
        return scratch[0..q.len];
    }
};

/// Record an in-place overwrite of a row the published graph already covers,
/// and keep the quantized codes for that row current.
///
/// Called from `upsert` with the write lock held. A row past `graph_count` is
/// in the pending tail, which is scanned from the arena and re-encoded by the
/// next build, so it needs neither.
///
/// The re-encode is what keeps `searchQuantized` honest between builds. Stage
/// 1 scores on the codes, so a row whose codes still describe its old vector
/// is admitted (or not) to the rescore set on the strength of a position it no
/// longer occupies; the fp32 rescore then ranks it correctly, but only if it
/// was admitted. Encoding one row is a few hundred nanoseconds under a lock
/// the write already holds. A concurrent stage-1 read of the row being
/// re-encoded may see a mixed code and mis-score that one candidate for that
/// one query; it can never see freed memory, since the store is only ever
/// replaced by `quantize`, which retires rather than frees.
///
/// Held under `SearchGuard` for the same reason `search` is: this thread
/// holds the write lock, not the guard, and `reclaimRetired` frees a retired
/// store as soon as `active_searches` is zero. Loading the store here and
/// encoding into it while the build thread swapped and reclaimed it was a
/// write into freed memory. The guard cannot deadlock against
/// `beginInPlace`'s drain: that drain runs on this same thread, earlier in
/// the same `upsert`, and has returned by the time this is reached; no other
/// writer can be inside `beginInPlace` because this thread holds the lock.
fn noteOverwrite(self: *Collection, offset: u32, vec: []const f32) void {
    // Under the write lock, so the builder's read of this bit and its
    // publish are ordered against this write; see `overwrite_log`.
    if (self.overwrite_log_active) self.overwrite_log.set(offset);

    if (offset >= self.graph_count.load(.acquire)) return;
    _ = self.overwritten_since_build.fetchAdd(1, .monotonic);

    const guard = SearchGuard.begin(self);
    defer guard.end();
    if (self.quant.load(.acquire)) |store| {
        store.encodeRow(offset, vec);
        // `quantize` swaps the store under the write lock this thread holds,
        // so the store just written is still the published one: it was not
        // retired under this write, let alone freed.
        std.debug.assert(self.quant.load(.acquire) == store);
    }
}

/// §6.5: "`exact: true` → brute force with a range-partitioned parallel scan,
/// since it's the recall ground truth and the pure-bandwidth benchmark."
///
/// §4: W9 (`--search-exact`) "is the most SIMD-revealing full-stack workload -
/// a linear scan with perfect hardware prefetching, so the memory system stops
/// hiding the kernel, and it doubles as the recall ground truth."
///
/// The scan is deterministic by construction: it visits offsets in ascending
/// order and the heap's total order (§8.7) breaks ties by id, so the result is
/// independent of how the range is partitioned across threads.
pub fn bruteForceRange(
    coll: *const Collection,
    query: []const f32,
    from: u32,
    to: u32,
    out: *heap.TopK,
) void {
    bruteForceRangeFiltered(coll, query, from, to, null, out);
}

/// `bruteForceRange` admitting only what `filter` admits, beside the
/// tombstone check. The predicate runs before the distance, since a payload
/// test is cheaper than a d=1536 dot product and most points fail it.
pub fn bruteForceRangeFiltered(
    coll: *const Collection,
    query: []const f32,
    from: u32,
    to: u32,
    filter: ?hnsw.Index.Filter,
    out: *heap.TopK,
) void {
    std.debug.assert(from <= to);
    std.debug.assert(to <= coll.id_space.count());
    std.debug.assert(query.len == coll.config.dim);
    // Converted once for the whole range. A per-row conversion would be a
    // correctness-preserving way to make the narrow datatypes slower than the
    // wide one, which is the opposite of why they exist.
    var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
    const probe = Probe.init(coll, query, &buf);
    var off = from;
    while (off < to) : (off += 1) {
        if (coll.deleted.isSet(off)) continue;
        if (filter) |f| if (!f.admits(off)) continue;
        out.push(.{ .id = off, .score = probe.scoreNode(off) });
    }
}

/// The most queries one pass of the arena may carry.
///
/// A gathered scan reads each row once for K queries instead of K times, so the
/// arena's bytes cross the bus once rather than K times. K is bounded because
/// the probes are per-query state and §6.3 forbids allocating on the query
/// path: the caller hands over scratch, and at d=1536 sixteen probes are 48 KiB
/// of it.
pub const max_gather = 16;

/// Bytes of scratch one probe needs for this collection, at its alignment.
///
/// `Probe.max_buffer` is the worst case over every dimension this server
/// accepts (16,384 elements at f16, so 32 KiB) and is what a *stack* buffer has
/// to declare. A gathered scan sizes from the collection instead, because K of
/// the worst case is not a stack, and at d=128 this is 256 bytes.
pub fn probeScratchStride(coll: *const Collection) usize {
    const bytes = coll.config.dim * @sizeOf(f16);
    return std.mem.alignForward(usize, bytes, Probe.buffer_align);
}

/// `bruteForceRangeFiltered` for several queries in one pass of the arena.
///
/// findings 45: W9 is bandwidth-bound at 85% of the bus, so nothing about the
/// kernel, the ISA or prefetching moves it. The engine ahead is the one that
/// *reads less*, and Qdrant's plain index scores a whole batch per walk
/// (`BatchFilteredSearcher`), which is why it scales 2.22x from `-p 1` to `-p 8`
/// against strawmANN's 1.51x. `handlers.BatchJob` established that a request's
/// queries can be fanned across workers; this is the inverse, and for an exact
/// search the inverse is the right direction: fanning K exact queries out costs
/// K passes over the same bytes.
///
/// The row is read once and scored K times, so the tombstone and filter tests
/// are paid once too. `outs[i]` receives query `i`'s candidates; the results are
/// identical to calling `bruteForceRangeFiltered` per query, which is what the
/// differential test asserts.
pub fn bruteForceRangeMulti(
    coll: *const Collection,
    queries: []const []const f32,
    from: u32,
    to: u32,
    filter: ?hnsw.Index.Filter,
    scratch: []u8,
    outs: []*heap.TopK,
) void {
    std.debug.assert(from <= to);
    std.debug.assert(to <= coll.id_space.count());
    std.debug.assert(queries.len == outs.len);
    std.debug.assert(queries.len <= max_gather);
    const stride = probeScratchStride(coll);
    std.debug.assert(scratch.len >= queries.len * stride);
    std.debug.assert(@intFromPtr(scratch.ptr) % Probe.buffer_align == 0);

    // Converted once each, for the whole range, exactly as the single-query
    // scan converts once: a per-row conversion would make the narrow datatypes
    // slower than the wide one, which is the opposite of why they exist.
    var probes: [max_gather]Probe = undefined;
    for (queries, 0..) |q, i| {
        std.debug.assert(q.len == coll.config.dim);
        probes[i] = Probe.init(coll, q, scratch[i * stride ..][0..stride]);
    }

    var off = from;
    while (off < to) : (off += 1) {
        if (coll.deleted.isSet(off)) continue;
        if (filter) |f| if (!f.admits(off)) continue;
        // The row's bytes are in cache now, which is the whole point: every
        // query pays the distance and none of them pays the fetch again.
        for (probes[0..queries.len], outs) |*probe, out| {
            out.push(.{ .id = off, .score = probe.scoreNode(off) });
        }
    }
}

/// Score exactly the points whose bit is set in `bits` (`payload.Store.select`
/// wrote them), skipping tombstones. This is the plain path for a selective
/// filter: when the index says the matching set is small, walking it is both
/// cheaper than a traversal and exact, which is the dispatch Qdrant makes on
/// the same threshold (`decisions.md`, W12).
pub fn searchSelected(
    coll: *const Collection,
    query: []const f32,
    bits: []const u64,
    bound: u32,
    out: *heap.TopK,
) void {
    const guard = SearchGuard.begin(coll);
    defer guard.end();
    std.debug.assert(bound <= coll.id_space.count());
    var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
    const probe = Probe.init(coll, query, &buf);
    const words = (@as(usize, bound) + 63) / 64;
    for (bits[0..words], 0..) |word_in, w| {
        var word = word_in;
        while (word != 0) {
            const bit: u6 = @intCast(@ctz(word));
            word &= word - 1;
            const off: u32 = @intCast(w * 64 + bit);
            if (off >= bound) break;
            if (coll.deleted.isSet(off)) continue;
            out.push(.{ .id = off, .score = probe.scoreNode(off) });
        }
    }
}

/// Single-threaded brute force over the whole collection.
pub fn bruteForce(coll: *const Collection, query: []const f32, out: *heap.TopK) void {
    bruteForceRange(coll, query, 0, @intCast(coll.id_space.count()), out);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

pub fn makeCollection(dim: usize, metric: Metric, cap: usize) !Collection {
    return makeCollectionDt(dim, metric, .float32, cap);
}

pub fn makeCollectionDt(dim: usize, metric: Metric, dt: dist.Datatype, cap: usize) !Collection {
    return Collection.init(testing.allocator, "test", .{
        .dim = dim,
        .metric = metric,
        .datatype = dt,
        .capacity = cap,
    });
}

test "§6.4 stride is padded to 64 bytes and padding is zeroed" {
    try testing.expectEqual(@as(usize, 3072), strideFor(768, .float32)); // already aligned
    try testing.expectEqual(@as(usize, 6144), strideFor(1536, .float32));
    try testing.expectEqual(@as(usize, 512), strideFor(128, .float32));
    // d=100 fp32 is 400 bytes -> 448. This is where padding is real.
    try testing.expectEqual(@as(usize, 448), strideFor(100, .float32));

    // A narrower element is a narrower row, which is the entire point of the
    // datatype: the same collection touches half or a quarter of the cache
    // lines per candidate.
    try testing.expectEqual(@as(usize, 1536), strideFor(768, .float16));
    try testing.expectEqual(@as(usize, 768), strideFor(768, .uint8));
    // Padding still applies, and still to 64 bytes rather than to the element.
    try testing.expectEqual(@as(usize, 256), strideFor(100, .float16));
    try testing.expectEqual(@as(usize, 128), strideFor(100, .uint8));

    var c = try makeCollection(100, .dot, 8);
    defer c.deinit();

    var v: [100]f32 = undefined;
    @memset(&v, 1.0);
    _ = try c.upsert(.{ .num = 0 }, &v);

    // Bytes past the vector, within the stride, must be zero, the kernels read
    // whole rows and garbage here would contribute to every score.
    const start = c.space.stride * 0;
    const pad = c.space.arena[start + 400 .. start + 448];
    for (pad) |b| try testing.expectEqual(@as(u8, 0), b);
}

test "padding stays zeroed after an update-in-place overwrites a row" {
    // The case a one-shot @memset at init would miss: row 0 is written twice,
    // and the second vector must not leave the first one's bytes in the pad.
    var c = try makeCollection(100, .dot, 8);
    defer c.deinit();

    var v: [100]f32 = undefined;
    @memset(&v, 3.0);
    _ = try c.upsert(.{ .num = 7 }, &v);
    @memset(&v, 5.0);
    _ = try c.upsert(.{ .num = 7 }, &v);

    const pad = c.space.arena[400..448];
    for (pad) |b| try testing.expectEqual(@as(u8, 0), b);
    try testing.expectEqual(@as(usize, 1), c.count());
}

test "upsert stores vectors retrievably and densely" {
    var c = try makeCollection(4, .dot, 16);
    defer c.deinit();

    for (0..5) |i| {
        var v = [_]f32{ @floatFromInt(i), 1, 2, 3 };
        const off = try c.upsert(.{ .num = i }, &v);
        try testing.expectEqual(@as(u32, @intCast(i)), off);
    }
    try testing.expectEqual(@as(usize, 5), c.count());
    for (0..5) |i| {
        const r = c.space.rowConst(@intCast(i));
        try testing.expectEqual(@as(f32, @floatFromInt(i)), r[0]);
    }
}

test "dimension mismatch is refused" {
    var c = try makeCollection(4, .dot, 16);
    defer c.deinit();
    var wrong = [_]f32{ 1, 2, 3 };
    try testing.expectError(Error.DimensionMismatch, c.upsert(.{ .num = 0 }, &wrong));
}

test "§8.3 cosine: ingest normalises and the short-circuit is counted" {
    var c = try makeCollection(4, .cosine, 16);
    defer c.deinit();

    var unnormalised = [_]f32{ 3, 4, 0, 0 }; // length 5
    _ = try c.upsert(.{ .num = 1 }, &unnormalised);
    try testing.expectEqual(@as(usize, 1), c.normalized_count);
    try testing.expectEqual(@as(usize, 0), c.short_circuited_count);
    const stored = c.space.rowConst(0);
    try testing.expectApproxEqAbs(@as(f32, 0.6), stored[0], 1e-6);

    // Already unit: stored untouched, and counted separately so the
    // conformance harness can compare the population against Qdrant's.
    var already = [_]f32{ 1, 0, 0, 0 };
    _ = try c.upsert(.{ .num = 2 }, &already);
    try testing.expectEqual(@as(usize, 1), c.normalized_count);
    try testing.expectEqual(@as(usize, 1), c.short_circuited_count);
}

test "non-cosine metrics store the vector verbatim" {
    for ([_]Metric{ .dot, .euclid, .manhattan }) |m| {
        var c = try makeCollection(4, m, 8);
        defer c.deinit();
        var v = [_]f32{ 3, 4, 0, 0 };
        _ = try c.upsert(.{ .num = 1 }, &v);
        const stored = c.space.rowConst(0);
        try testing.expectEqual(@as(f32, 3), stored[0]);
        try testing.expectEqual(@as(f32, 4), stored[1]);
        try testing.expectEqual(@as(usize, 0), c.normalized_count);
    }
}

test "prepareQuery preprocesses exactly like ingest" {
    var c = try makeCollection(4, .cosine, 8);
    defer c.deinit();

    var stored = [_]f32{ 3, 4, 0, 0 };
    _ = try c.upsert(.{ .num = 1 }, &stored);

    var scratch: [4]f32 = undefined;
    const q = try c.prepareQuery(&scratch, &[_]f32{ 3, 4, 0, 0 });
    // Query and stored vector went through the same funnel, so the self-score
    // is 1. If only one side were normalised it would be 5.
    const score = dist.similarity(c.config.metric.kernel(), q, c.space.rowConst(0));
    try testing.expectApproxEqAbs(@as(f32, 1.0), score, 1e-6);
}

test "brute force finds the true nearest neighbours" {
    var c = try makeCollection(8, .euclid, 128);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0xb0f);
    const rnd = prng.random();
    const n = 100;
    for (0..n) |i| {
        var v: [8]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    var q: [8]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    var top_storage: [10]Candidate = undefined;
    var top = heap.TopK.init(&top_storage, 10);
    bruteForce(&c, &q, &top);
    const got = top.finish();

    // Cross-check against an independent full scan.
    var all: [n]Candidate = undefined;
    for (0..n) |i| {
        all[i] = .{
            .id = @intCast(i),
            .score = dist.similarity(.euclid, &q, c.space.rowConst(@intCast(i))),
        };
    }
    std.mem.sort(Candidate, &all, {}, struct {
        fn lt(_: void, a: Candidate, b: Candidate) bool {
            return a.better(b);
        }
    }.lt);

    try testing.expectEqual(@as(usize, 10), got.len);
    for (0..10) |i| try testing.expectEqual(all[i].id, got[i].id);
}

test "brute force is invariant to how the range is partitioned" {
    // §6.5 partitions the exact scan across threads. §8.7 requires the result
    // be identical regardless of thread count, which holds because the heap's
    // total order does not depend on arrival order.
    var c = try makeCollection(16, .dot, 256);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0x9a97);
    const rnd = prng.random();
    const n = 200;
    for (0..n) |i| {
        var v: [16]f32 = undefined;
        // Deliberately quantised values so ties are common.
        for (&v) |*x| x.* = @floatFromInt(rnd.uintLessThan(u8, 3));
        _ = try c.upsert(.{ .num = i }, &v);
    }
    var q: [16]f32 = undefined;
    for (&q) |*x| x.* = @floatFromInt(rnd.uintLessThan(u8, 3));

    var s1: [10]Candidate = undefined;
    var whole = heap.TopK.init(&s1, 10);
    bruteForce(&c, &q, &whole);
    const expect = whole.finish();

    // Now the same scan in four ranges, merged.
    for ([_]usize{ 2, 3, 4, 7 }) |parts| {
        var s2: [10]Candidate = undefined;
        var merged = heap.TopK.init(&s2, 10);
        const chunk = (n + parts - 1) / parts;
        var start: u32 = 0;
        while (start < n) : (start += @intCast(chunk)) {
            const end: u32 = @min(@as(u32, n), start + @as(u32, @intCast(chunk)));
            var sub_storage: [10]Candidate = undefined;
            var sub = heap.TopK.init(&sub_storage, 10);
            bruteForceRange(&c, &q, start, end, &sub);
            for (sub.finish()) |cand| merged.push(cand);
        }
        const got = merged.finish();
        for (expect, got) |e, g| {
            try testing.expectEqual(e.id, g.id);
            try testing.expectEqual(e.score, g.score);
        }
    }
}

test "deleted points are skipped by search but keep their offsets" {
    var c = try makeCollection(4, .dot, 16);
    defer c.deinit();
    for (0..5) |i| {
        var v = [_]f32{ @floatFromInt(i + 1), 0, 0, 0 };
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try testing.expect(c.delete(.{ .num = 4 })); // the best match for [1,0,0,0]
    try testing.expect(!c.delete(.{ .num = 4 })); // idempotent
    try testing.expect(!c.delete(.{ .num = 99 })); // unknown
    try testing.expectEqual(@as(usize, 4), c.count());

    var top_storage: [3]Candidate = undefined;
    var top = heap.TopK.init(&top_storage, 3);
    bruteForce(&c, &[_]f32{ 1, 0, 0, 0 }, &top);
    const got = top.finish();
    for (got) |g| try testing.expect(g.id != 4);
    // §3: tombstone only, no compaction, offset 3 is still point 3.
    try testing.expectEqual(@as(u32, 3), got[0].id);
}

test "re-upserting a deleted point resurrects it" {
    var c = try makeCollection(4, .dot, 16);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    _ = try c.upsert(.{ .num = 1 }, &v);
    try testing.expect(c.delete(.{ .num = 1 }));
    try testing.expectEqual(@as(usize, 0), c.count());

    _ = try c.upsert(.{ .num = 1 }, &v);
    try testing.expectEqual(@as(usize, 1), c.count());
    try testing.expect(!c.isDeleted(0));
}

test "capacity is enforced" {
    var c = try makeCollection(4, .dot, 2);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    _ = try c.upsert(.{ .num = 0 }, &v);
    _ = try c.upsert(.{ .num = 1 }, &v);
    try testing.expectError(Error.CapacityExceeded, c.upsert(.{ .num = 2 }, &v));
    // But updating an existing point still works at full capacity (§2 --max-id).
    _ = try c.upsert(.{ .num = 1 }, &v);
}

test "§8.6 metamorphic: score(x,x) is maximal for every metric" {
    for ([_]Metric{ .dot, .cosine, .euclid, .manhattan }) |m| {
        var c = try makeCollection(32, m, 64);
        defer c.deinit();

        var prng = std.Random.DefaultPrng.init(0x5e1f);
        const rnd = prng.random();
        for (0..20) |i| {
            var v: [32]f32 = undefined;
            for (&v) |*x| x.* = rnd.floatNorm(f32);
            _ = try c.upsert(.{ .num = i }, &v);
        }

        // Query with a stored vector; it must come back first.
        const target: u32 = 7;
        var q: [32]f32 = undefined;
        @memcpy(&q, c.space.rowConst(target));

        var top_storage: [5]Candidate = undefined;
        var top = heap.TopK.init(&top_storage, 5);
        // Already in stored form, so no prepareQuery: normalising a normalised
        // vector is a no-op for the short-circuit but this keeps the test exact.
        bruteForce(&c, &q, &top);
        const got = top.finish();
        try testing.expectEqual(target, got[0].id);
    }
}

// =========================================================================
// §6.5, index construction and the Yellow -> Green flow
// =========================================================================

/// Monotonic nanoseconds.
///
/// `std.time.nanoTimestamp` moved under the `Io` interface in Zig 0.16 and the
/// collection has no `Io` to thread through; the raw clock is what a build
/// timer needs anyway (§7.2 wants build time as a first-class W2 measurement).
pub fn monotonicNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Announce that rows are no longer write-once, before the first one is.
///
/// The order matters and the other one has a hole in it. Setting the flag
/// *after* the first in-place write leaves every reader already inside a score
/// unguarded for exactly the write that needs guarding. Setting it first and
/// then draining the searches that are already running closes it: after this
/// returns, every search that starts will check the version, and every search
/// that could have skipped the check has finished.
///
/// The drain costs one wait, once per collection, on the first update. Called
/// with the write lock held, so only one thread can be here.
///
/// The flag and the counter are a store-buffering pair (Dekker): this thread
/// publishes the flag then reads the counter, a search increments the counter
/// (`SearchGuard.begin`) then reads the flag (`Probe.scoreNode`). With a
/// release store here and an acquire load there, x86 may let this thread's
/// store sit in its store buffer past its own load, so both sides read the
/// old value: the drain sees zero while a search that will never check the
/// version is inside. So the publish is a seq_cst read-modify-write and the
/// counter read after it is seq_cst, and the reader's two accesses are
/// seq_cst as well, which is the four-way ordering the store-buffering
/// litmus test needs. Zig 0.16 has no `@fence`; a seq_cst RMW is the
/// portable spelling of the full barrier, and on x86 the reader's cost is
/// nil (a locked RMW already is one, a seq_cst load is a plain load).
fn beginInPlace(self: *Collection) void {
    if (self.in_place_updates.load(.monotonic)) return;
    _ = self.in_place_updates.swap(true, .seq_cst);
    while (self.active_searches.load(.seq_cst) != 0) std.Thread.yield() catch {};
}

/// Marks a search as in flight, so a rebuild knows when its predecessor's
/// graph is unreachable.
///
/// Acquired *before* the collection's `graph` or `quant` pointer is read and
/// released after the last use of either. Both orderings matter: the acquire
/// on entry pairs with the release on exit so the reclaiming thread cannot
/// observe zero while a reader is still inside.
pub const SearchGuard = struct {
    coll: *const Collection,

    pub fn begin(coll: *const Collection) SearchGuard {
        // Cast away const: the counter is the one part of a collection that a
        // *reader* mutates, which is what makes an otherwise immutable-during-
        // search structure reclaimable at all.
        const mutable: *Collection = @constCast(coll);
        // seq_cst, not acquire, and the same is required of what it pairs
        // with. Three reclamation rules have this shape: a reader bumps this
        // counter and *then* loads a pointer the writer may be replacing,
        // while the writer replaces the pointer and *then* reads this counter
        // to decide the old one is unreachable. That is store buffering, and
        // acquire/release does not forbid the outcome that breaks it -- the
        // reader seeing the old pointer while the writer sees a zero count,
        // and freeing it underneath. Only seq_cst on all four accesses does,
        // so `publishedGraph`, `Collection.quant`, `Scroll.published` and the
        // two counter loads (`reclaimRetired`, `Scroll.retire`) are seq_cst
        // as well. On x86 every one of them is the instruction it already
        // was, a locked RMW or a plain mov; the guarantee is for the weaker
        // models the source is written against, not for this host.
        //
        // The fourth instance is `beginInPlace`'s, whose other half is the
        // load of `in_place_updates` in `Probe.scoreNode`.
        _ = mutable.active_searches.fetchAdd(1, .seq_cst);
        return .{ .coll = coll };
    }

    pub fn end(self: SearchGuard) void {
        const mutable: *Collection = @constCast(self.coll);
        _ = mutable.active_searches.fetchSub(1, .release);
    }
};

/// Free everything retired by an earlier rebuild, if no search can hold it.
///
/// Called by the build thread after it has published a replacement. Retained
/// memory is not a leak in the "forgot to free" sense, it is a leak in the
/// sense that matters: measured at 676 KB per graph at capacity 4096, which is
/// ~181 MB per graph at the headline capacity of 1.1M, plus ~141 MB per
/// retired SQ8 store. Five rebuilds under a write-heavy workload held about
/// 1.3 GB that could never be returned, and `rebuild_ratio` bounds the pending
/// tail rather than the number of rebuilds.
///
/// Returns the number of items freed, which is what the test asserts on.
pub fn reclaimRetired(coll: *Collection) usize {
    // Zero readers means every search that could have observed a replaced
    // pointer has returned: a search that starts after this load reads the
    // *current* graph, because publication happened before the caller invoked
    // this. A non-zero count is not a failure, it just defers the work to the
    // next rebuild.
    // seq_cst: the writer half of the store-buffering pair `SearchGuard.begin`
    // describes, against the publication a few lines above the call site.
    if (coll.active_searches.load(.seq_cst) != 0) return 0;

    var freed: usize = 0;
    for (coll.retired_graphs.items) |g| {
        g.deinit();
        coll.alloc.destroy(g);
        freed += 1;
    }
    coll.retired_graphs.clearRetainingCapacity();
    for (coll.retired_quant.items) |q| {
        destroyStore(coll.alloc, q);
        freed += 1;
    }
    coll.retired_quant.clearRetainingCapacity();
    return freed;
}

fn destroyStore(alloc: std.mem.Allocator, store: *quantized.Store) void {
    store.deinit(alloc);
    alloc.destroy(store);
}

/// A query, converted once into the collection's storage type.
///
/// The same shape as `quantized.Query` and for the same reason: a comparison
/// must not convert per candidate. Qdrant converts the query the same way
/// (`metric_query_scorer.rs` calls `slice_from_float_cow` on the preprocessed
/// query), so a f16 collection compares f16 against f16 and a uint8 collection
/// compares bytes against bytes, including the saturating cast. Matching that
/// is what §8.5's T1 tier measures.
///
/// `buf` is caller-owned and lives for the whole search; `Probe` never
/// allocates, per §6.3.
pub const Probe = struct {
    coll: *const Collection,
    kernel: dist.Kernel,
    /// Set for `.uint8` + cosine only: Σq², the query half of the divisor
    /// Qdrant applies at query time. Computed once here rather than per
    /// candidate.
    query_norm2: f32 = 0,
    stored: union(enum) {
        f32: []const f32,
        f16: []const f16,
        u8: []const u8,
    },

    /// The stack buffer every search entry point declares.
    ///
    /// `max_converted_dim` elements at the widest converted element (f16), so
    /// one constant covers every datatype and `Collection.init` refuses the
    /// dimensions it would not cover.
    pub const max_buffer = max_converted_dim * @sizeOf(f16);
    /// What `init`'s `@alignCast` to `[]f16` needs of `buf`. Declared on the
    /// buffers rather than trusted: a `[N]u8` local is only guaranteed
    /// byte-aligned, and the cast asserts in Debug and is UB in ReleaseFast.
    /// 64 rather than `@alignOf(f16)` so the converted query starts on a
    /// cache line, the same reason the arena stride is 64 (§6.4).
    pub const buffer_align = 64;

    pub fn init(coll: *const Collection, query: []const f32, buf: []u8) Probe {
        std.debug.assert(query.len == coll.config.dim);
        std.debug.assert(@intFromPtr(buf.ptr) % @alignOf(f16) == 0);
        const kernel = coll.config.metric.kernel();
        return switch (coll.config.datatype) {
            .float32 => .{ .coll = coll, .kernel = kernel, .stored = .{ .f32 = query } },
            .float16 => blk: {
                const out: []f16 = @alignCast(std.mem.bytesAsSlice(f16, buf[0 .. query.len * 2]));
                dist.datatype.convertToF16(out, query);
                break :blk .{ .coll = coll, .kernel = kernel, .stored = .{ .f16 = out } };
            },
            .uint8 => blk: {
                const out = buf[0..query.len];
                dist.datatype.convertToU8(out, query);
                var n2: u32 = 0;
                if (coll.config.metric == .cosine) {
                    n2 = dist.dot_i8.u8u8_native.call(out, out);
                }
                break :blk .{
                    .coll = coll,
                    .kernel = kernel,
                    .query_norm2 = @floatFromInt(n2),
                    .stored = .{ .u8 = out },
                };
            },
        };
    }

    /// The internal similarity for one candidate. Higher is better, on every
    /// path, which is what lets the result heap stay metric-agnostic.
    ///
    /// Reads the row consistently even while it is being overwritten: see
    /// `Collection.readRow`. On an append-only collection, which is every
    /// workload except a mixed read/write one, that costs a single predictable
    /// load of a flag that is in L1 by the second candidate.
    pub fn scoreNode(self: *const Probe, node: u32) f32 {
        // seq_cst pairs with `SearchGuard.begin`'s increment against
        // `beginInPlace`'s publish-then-drain; the same instruction as an
        // acquire load on x86 and AArch64, so the hot path pays nothing.
        if (!self.coll.in_place_updates.load(.seq_cst)) return self.scoreRow(node);
        // A row being rewritten: take the version either side and retry if it
        // moved. Retrying costs one recomputed distance, which is cheaper than
        // the alternative of holding a lock across every comparison.
        while (true) {
            const v1 = self.coll.row_versions[node].load(.acquire);
            if (v1 & 1 != 0) {
                std.atomic.spinLoopHint();
                continue;
            }
            const s = self.scoreRow(node);
            if (self.coll.row_versions[node].load(.acquire) == v1) return s;
        }
    }

    fn scoreRow(self: *const Probe, node: u32) f32 {
        const space = &self.coll.space;
        return switch (self.stored) {
            .f32 => |q| dist.similarity(self.kernel, q, space.rowConst(node)),
            .f16 => |q| switch (self.kernel) {
                .dot => dist.dot_f16.dot_native.call(q, space.rowF16(node)),
                .euclid => dist.dot_f16.euclid_native.call(q, space.rowF16(node)),
                .manhattan => dist.dot_f16.manhattan_native.call(q, space.rowF16(node)),
            },
            .u8 => |q| self.scoreU8(q, space.rowU8(node)),
        };
    }

    fn scoreU8(self: *const Probe, q: []const u8, row: []const u8) f32 {
        // Cosine is the only metric that does not collapse to a kernel here:
        // with no ingest normalisation the divisor has to be computed, so this
        // is Qdrant's `cosine_similarity_bytes` including its zero-norm case,
        // which returns 0 rather than a NaN.
        if (self.kernel == .dot and self.coll.config.metric == .cosine) {
            const d: f32 = @floatFromInt(dist.dot_i8.u8u8_native.call(q, row));
            // `Σ row²` is a dot product with itself, so it uses the same vector
            // kernel. It was a scalar loop, on the per-candidate path,
            // immediately below a call that was not.
            const row_norm2: f32 = @floatFromInt(dist.dot_i8.u8u8_native.call(row, row));
            if (self.query_norm2 == 0 or row_norm2 == 0) return 0;
            return d / @sqrt(self.query_norm2 * row_norm2);
        }
        return switch (self.kernel) {
            .dot => @floatFromInt(dist.dot_i8.u8u8_native.call(q, row)),
            .euclid => @floatFromInt(dist.dot_i8.euclid_u8_native.call(q, row)),
            .manhattan => @floatFromInt(dist.dot_i8.manhattan_u8_native.call(q, row)),
        };
    }

    /// What `hnsw.Scorer.of` binds to. Named `score` on every prepared query
    /// so one constructor serves them all.
    pub fn score(ctx: *const anyopaque, node: u32) f32 {
        const self: *const Probe = @ptrCast(@alignCast(ctx));
        return self.scoreNode(node);
    }

    /// The first line of `node`'s row, so the traversal can ask for it before
    /// it scores (`hnsw.Scorer.prefetch`). One line, not the row: at d=1536
    /// a row is 96 lines and a 32-neighbour list would be 3,000 prefetches
    /// for 32 that matter first; the hardware prefetcher follows the rest of
    /// a row once the kernel starts streaming it.
    pub fn prefetch(ctx: *const anyopaque, node: u32) void {
        const self: *const Probe = @ptrCast(@alignCast(ctx));
        @prefetch(self.coll.space.rowBytes(node).ptr, .{ .rw = .read, .locality = 3, .cache = .data });
    }
};

/// Compare two stored rows, without converting either.
///
/// The index builder's question, and it never involves an fp32 query: both
/// operands are already in the storage type, so the comparison happens there.
pub fn scoreStored(coll: *const Collection, a: u32, b: u32) f32 {
    if (!coll.in_place_updates.load(.acquire)) return scoreStoredRaw(coll, a, b);
    // The build runs on its own thread while ingest continues, so both rows can
    // be overwritten underneath it. A torn read here is not a wrong score for
    // one query, it is a wrong edge baked into the graph until the next
    // rebuild, which is the more expensive kind of wrong.
    //
    // Builder threads hold no `SearchGuard`, so `beginInPlace`'s drain does
    // not wait for them: the very first overwrite can tear a row under a
    // builder that read the flag as clear. That is tolerated rather than
    // guarded because the row is on `overwrite_log`, so the graph is
    // published counting it as pending (`overwritten_since_build`) and the
    // codes are re-encoded from the arena at publish; the wrong edge is
    // exactly the one a post-publish overwrite would have left, and the
    // rebuild rule already covers that.
    while (true) {
        const va = coll.row_versions[a].load(.acquire);
        const vb = coll.row_versions[b].load(.acquire);
        if ((va | vb) & 1 != 0) {
            std.atomic.spinLoopHint();
            continue;
        }
        const s = scoreStoredRaw(coll, a, b);
        if (coll.row_versions[a].load(.acquire) == va and
            coll.row_versions[b].load(.acquire) == vb) return s;
    }
}

fn scoreStoredRaw(coll: *const Collection, a: u32, b: u32) f32 {
    const space = &coll.space;
    const kernel = coll.config.metric.kernel();
    return switch (coll.config.datatype) {
        .float32 => dist.similarity(kernel, space.rowConst(a), space.rowConst(b)),
        .float16 => switch (kernel) {
            .dot => dist.dot_f16.dot_native.call(space.rowF16(a), space.rowF16(b)),
            .euclid => dist.dot_f16.euclid_native.call(space.rowF16(a), space.rowF16(b)),
            .manhattan => dist.dot_f16.manhattan_native.call(space.rowF16(a), space.rowF16(b)),
        },
        .uint8 => blk: {
            const ra = space.rowU8(a);
            const rb = space.rowU8(b);
            if (kernel == .dot and coll.config.metric == .cosine) {
                // The build's inner loop, so the same rule applies with more
                // force: two vector calls, not two scalar loops.
                const na = dist.dot_i8.u8u8_native.call(ra, ra);
                const nb = dist.dot_i8.u8u8_native.call(rb, rb);
                if (na == 0 or nb == 0) break :blk 0;
                const d: f32 = @floatFromInt(dist.dot_i8.u8u8_native.call(ra, rb));
                break :blk d / @sqrt(@as(f32, @floatFromInt(na)) * @as(f32, @floatFromInt(nb)));
            }
            break :blk switch (kernel) {
                .dot => @as(f32, @floatFromInt(dist.dot_i8.u8u8_native.call(ra, rb))),
                .euclid => @as(f32, @floatFromInt(dist.dot_i8.euclid_u8_native.call(ra, rb))),
                .manhattan => @as(f32, @floatFromInt(dist.dot_i8.manhattan_u8_native.call(ra, rb))),
            };
        },
    };
}

pub fn scorerFor(coll: *const Collection) build_hnsw.Scorer {
    const S = struct {
        fn between(ctx: *const anyopaque, a: u32, b: u32) f32 {
            const c: *const Collection = @ptrCast(@alignCast(ctx));
            return scoreStored(c, a, b);
        }
    };
    return .{ .ctx = @ptrCast(coll), .between = S.between };
}

pub const BuildMode = enum {
    /// §8.7 option (c): deterministic, single-threaded. What conformance uses.
    serial,
    /// §8.7 option (b): all cores, checksum-stable per (seed, thread count).
    parallel,
};

/// Build the HNSW graph over everything currently ingested.
///
/// Synchronous. `buildIndexDetached` runs this on its own thread so the RPC
/// that triggers it can return Yellow immediately, which is what bfb's poll
/// loop expects to see.
pub fn buildIndex(coll: *Collection, mode: BuildMode, threads: usize) !void {
    const n = coll.id_space.count();
    if (n == 0) {
        // Nothing to build over, but the same publish rule as below: under
        // the write lock, then ask whether ingest already outran it. Without
        // the second step a point that landed between the read of `n` and
        // this store left the collection `.ready` over no graph, Yellow, and
        // `ensureIndexBuilding` early-outs on `.ready`, so `wait_index` hung
        // until the next upsert happened to invalidate.
        coll.write_lock.lock();
        defer coll.write_lock.unlock();
        coll.index_state.store(.ready, .release);
        invalidateIndex(coll);
        return;
    }

    // §6.5 quotes 26 s for a 1M bulk build. Measured, that figure
    // holds for no configuration this project runs: a quiet 1M build at d=128
    // is 15-45 s, the same count at d=1536 is 250-348 s, and a build competing
    // with the queries of the row that provoked it is 105 s. Build time scales
    // with the vector width and collapses under contention, so one number
    // cannot describe it. W11 measured 489
    // qps over a 102 s row without ever recovering — which is only explicable
    // if the build ran far longer than that under contention with the queries,
    // or did not finish. Neither could be told from the outside: `server.log`
    // recorded the startup banner and nothing else, so the question stayed a
    // hypothesis through two reports. One line in and one line out is enough
    // to settle it, and a rebuild is bounded by `rebuild_ratio` so this cannot
    // become chatter.
    const build_started_ns = monotonicNs();
    const pending_at_entry = n -| coll.graph_count.load(.acquire);
    if (log_rebuilds) std.debug.print(
        "index: rebuild start points={d} pending={d} overwritten={d} threads={d} mode={s}\n",
        .{ n, pending_at_entry, coll.overwritten_since_build.load(.acquire), threads, @tagName(mode) },
    );

    const g = try coll.alloc.create(hnsw.Graph);
    errdefer coll.alloc.destroy(g);
    g.* = try hnsw.Graph.init(coll.alloc, hnsw.Params.fromM(
        coll.config.hnsw_m,
        coll.config.hnsw_ef_construct,
        coll.config.seed,
    ), coll.config.capacity);
    errdefer g.deinit();
    g.cancel = &coll.drop_requested;

    // Scratch for widening rows the builder has to re-encode at publish
    // (below). Taken before the log opens so that an allocation failure
    // cannot leave a build with overwrites it recorded but cannot apply.
    const widen = try coll.alloc.alloc(f32, coll.config.dim);
    defer coll.alloc.free(widen);

    // From here until publish, every overwrite is logged as well as applied:
    // the builder is about to read rows it will not look at again.
    coll.write_lock.lock();
    coll.overwrite_log_active = true;
    coll.write_lock.unlock();
    // If the build fails partway, close the log so the next one starts clean.
    errdefer {
        coll.write_lock.lock();
        coll.overwrite_log_active = false;
        coll.overwrite_log.unmanaged.unsetAll();
        coll.write_lock.unlock();
    }

    // Extend the published graph rather than rebuild it, where that is sound.
    //
    // A rebuild costs the whole collection every time: W11 appends 200,000
    // points to 1,000,000 and re-inserts all 1,200,000, and every query scans
    // the pending tail until it lands (findings 25, 31 — the row reads 0.41x).
    // A bulk build *is* insertion one node at a time into a graph that already
    // holds its predecessors, so inserting only what arrived is the same
    // operation for six times less work.
    //
    // Three conditions, and all three are about soundness rather than taste:
    //
    //   * the published graph must have the parameters this build was asked
    //     for, or the extension inherits a graph nobody requested;
    //   * nothing may have been overwritten in place since it was built. An
    //     overwritten row sits in the graph at the position its *old* vector
    //     occupied, and only a full rebuild moves it. `overwritten_since_build`
    //     is exactly that count;
    //   * and it must actually be smaller, or there is nothing to extend.
    //
    // The copy is what keeps this safe with no new concurrency: readers keep
    // traversing the old graph, untouched, and see the new points when this one
    // is published. Nothing here mutates a graph a reader can reach — which is
    // what separates it from the live insertion findings 31 asks for, and is
    // why it can be done now.
    const t0 = monotonicNs();
    const prior = publishedGraph(coll);
    const extend_from: usize = blk: {
        if (mode != .parallel) break :blk 0;
        const old_g = prior orelse break :blk 0;
        if (coll.overwritten_since_build.load(.acquire) != 0) break :blk 0;
        if (old_g.count == 0 or old_g.count >= n) break :blk 0;
        if (!std.meta.eql(old_g.params, g.params)) break :blk 0;
        if (old_g.capacity != g.capacity) break :blk 0;
        g.copyFrom(old_g);
        break :blk old_g.count;
    };
    if (log_rebuilds and extend_from > 0) std.debug.print(
        "index: extending from {d} rather than rebuilding {d}\n",
        .{ extend_from, n },
    );
    switch (mode) {
        .serial => {
            var b = try build_hnsw.Builder.init(coll.alloc, g, scorerFor(coll));
            defer b.deinit(coll.alloc);
            try b.buildSerial(n);
        },
        .parallel => {
            const stats = try build_hnsw.extendParallel(coll.alloc, g, scorerFor(coll), extend_from, n, threads);
            // §6.5: "Measure lock contention explicitly; if it shows,
            // partition-then-merge is the fallback." The measurement was
            // taken and then parked on a `build_stats` field that nothing
            // ever read, which is the same as not taking it: the fallback
            // cannot be triggered by a number no operator can see. `repaired`
            // rides along because it is the size of the effect findings 34 is
            // about.
            if (log_rebuilds) std.debug.print(
                "index: parallel build threads={d} nodes={d} repaired={d} contended={d}\n",
                .{ stats.threads, stats.nodes, stats.repaired, stats.contended_acquisitions },
            );
        },
    }
    coll.build_ns = monotonicNs() - t0;

    // Free any previous graph only after the new one is ready, so a concurrent
    // search never observes a freed pointer.
    const old = coll.graph;
    // §6.7's codes are built alongside the graph: both are bulk, post-ingest
    // artefacts of the same Yellow -> Green transition, and publishing the
    // graph before the codes exist would let a query traverse with one and
    // rescore against the other.
    //
    // A failure here fails the build, like every other allocation in it: the
    // `catch {}` that used to sit here published the graph over the codes of
    // the *previous* build, zero for every row appended since, on a
    // collection that told its client it was quantized. `quantize` publishes
    // nothing on error, so the previous store stays served, the errdefers
    // above drop the graph and close the log, and the caller
    // (`ensureIndexBuilding`) returns the collection to `.absent` for the
    // next poll to retry.
    if (coll.quant_mode != .none) {
        try quantizeWith(coll, coll.quant_mode, threads);
    }

    if (coll.build_hook) |hook| hook(coll);

    // Publish, under the write lock.
    //
    // The lock is what closes the read-to-publish window. An overwrite that
    // landed after the builder read its row is in the graph and (until here)
    // in the codes at the row's *old* position, and `noteOverwrite` encoded
    // it into the store that was current at the time, which `quantize` has
    // just retired. Every such row is on `overwrite_log`, so: re-encode it
    // into the store about to be served, and count it as pending for the
    // graph about to be served, exactly as an overwrite after publish would
    // be counted. Nothing can be added to the log while the lock is held, so
    // the log, the counter and the pointers move together.
    coll.write_lock.lock();
    coll.overwrite_log_active = false;
    var moved: usize = 0;
    if (coll.overwrite_log.count() > 0) {
        const store = coll.quant.load(.acquire);
        var it = coll.overwrite_log.iterator(.{});
        while (it.next()) |off| {
            // Past `n` is the tail: neither in this graph nor in these codes,
            // and the next build reads it fresh.
            if (off >= n) continue;
            moved += 1;
            if (store) |st| st.encodeRow(@intCast(off), coll.space.readInto(@intCast(off), widen));
        }
        coll.overwrite_log.unmanaged.unsetAll();
    }
    // An atomic store: since the previous graph is served throughout a
    // rebuild, a search may be loading this pointer at this very moment
    // (`publishedGraph`). It sees the old one or the new one, each whole, and
    // the old one is retired below rather than freed while it is held.
    @atomicStore(?*hnsw.Graph, &coll.graph, g, .seq_cst);
    coll.graph_count.store(n, .release);
    coll.indexed_count = n;
    // Not zero: the rows overwritten after the builder read them are in the
    // graph at the wrong position, and `needsRebuild` must know. Storing (not
    // subtracting) is exact here because every writer of this counter holds
    // the lock this thread holds.
    coll.overwritten_since_build.store(moved, .release);

    // Publish first, *then* ask whether ingest already outran this graph.
    //
    // A write that landed during the build is not lost by publishing
    // `.ready`: an appended row is past `n` and served by `scanPendingTail`,
    // an overwritten row is on the log just replayed and counted above, and a
    // delete is filtered at query time. So the graph goes out, and the same
    // rule that governs a write *after* publish decides whether a rebuild is
    // due (`invalidateIndex` from `.ready`: `needsRebuild`'s ratio). It used
    // to be a flag, set by any write during the build, that sent this to
    // `.absent` unconditionally, so under sustained writes (W11) a correct
    // graph was thrown away at every publish, every query brute-forced, and
    // the collection never returned to `.ready`.
    //
    // Under the lock, so the count `needsRebuild` reads covers every write
    // that observed `.building`; a write that observes `.ready` runs the
    // same check itself from the API layer. Both downgrades are idempotent.
    coll.index_state.store(.ready, .release);
    const due_again = needsRebuild(coll);
    invalidateIndex(coll);
    coll.write_lock.unlock();

    // Published, so the timing is a fact rather than an estimate. `again=true`
    // says ingest outran this build and the next one is already due, which is
    // the shape a row like W11 would show if it never returned to a quiet
    // graph for the whole measurement.
    if (log_rebuilds) std.debug.print(
        "index: rebuild published points={d} re-encoded={d} ms={d} again={}\n",
        .{ n, moved, (monotonicNs() -| build_started_ns) / std.time.ns_per_ms, due_again },
    );
    if (log_rebuilds) logGraphQuality(coll.alloc, g);

    // Retire, never free *here*: a search may still be traversing it. See
    // `retired_graphs`.
    if (old) |o| {
        coll.retired_graphs.append(coll.alloc, o) catch {
            // Nothing safe to do on OOM here except keep the graph alive. It is
            // reachable only from this list, so the leak is bounded by the
            // number of rebuilds and is preferable to a use-after-free.
        };
    }

    // Then reclaim, now that the replacement is published. Without this the
    // retired list only ever grew: every rebuild added a graph (and, on a
    // quantized collection, a store) that lived until the collection was
    // dropped. If a search is in flight the reclaim is skipped and the next
    // rebuild retries, so this is best-effort by design and never blocks the
    // build on a reader.
    _ = reclaimRetired(coll);
}

/// What `quantize` hands the builder: the collection, plus one row of scratch
/// to widen into. Single-threaded by construction, since `quantize` runs
/// inside the build lock.
const SourceCtx = struct {
    coll: *const Collection,
    scratch: []f32,
};

/// Attach a quantized store to the collection, training on what is ingested.
pub fn quantize(coll: *Collection, mode: quant_mod.Mode) !void {
    return quantizeWith(coll, mode, 1);
}

/// `quantize` with `threads` for the parts of the build that parallelise
/// without changing the store (PQ training and encoding, `quantized.
/// BuildOptions.threads`). The index build passes its own thread count.
pub fn quantizeWith(coll: *Collection, mode: quant_mod.Mode, threads: usize) !void {
    // Quantization trains and encodes from fp32. On a narrow collection that
    // means widening each row into scratch: quantizing an already-narrow store
    // is a second lossy step, and §6.7's codebooks are defined over the values,
    // not over their storage width.
    const widen = try coll.alloc.alloc(f32, coll.config.dim);
    defer coll.alloc.free(widen);
    var ctx = SourceCtx{ .coll = coll, .scratch = widen };
    const src = quantized.Source{
        .ctx = @ptrCast(&ctx),
        .row = struct {
            fn f(c: *const anyopaque, offset: u32) []const f32 {
                const sc: *const SourceCtx = @ptrCast(@alignCast(c));
                return sc.coll.space.readIntoGuarded(sc.coll, offset, sc.scratch);
            }
        }.f,
        // The same read into a caller-owned scratch, so the encoding threads
        // do not share `widen`.
        .row_into = struct {
            fn f(c: *const anyopaque, offset: u32, scratch: []f32) []const f32 {
                const sc: *const SourceCtx = @ptrCast(@alignCast(c));
                return sc.coll.space.readIntoGuarded(sc.coll, offset, scratch);
            }
        }.f,
        .count = coll.id_space.count(),
        .dim = coll.config.dim,
    };
    var store = try quantized.buildWith(coll.alloc, mode, src, coll.config.capacity, .{
        .quantile = coll.quant_quantile,
        .threads = threads,
    });
    errdefer store.deinit(coll.alloc);

    // A `.none` store (mode `.none`) is published as *no* store, so
    // `choosePath` never sends a query down the quantized path to score
    // against nothing.
    if (store == .none) {
        coll.write_lock.lock();
        defer coll.write_lock.unlock();
        if (coll.quant.swap(null, .seq_cst)) |old| {
            coll.retired_quant.append(coll.alloc, old) catch {
                // Unreachable in practice, capacity was reserved below on the
                // last publish; kept alive rather than freed under a reader.
            };
        }
        return;
    }

    // Reserve the retire slot *before* publishing, so the publish cannot fail
    // halfway: swapping in the new store and then failing to record the old
    // one would leak the old one, 141 MB for SQ8 at the headline capacity,
    // which is what the previous `catch {}` did. If the reservation fails
    // the collection keeps the encoding it already had and the `errdefer`
    // above frees the new, unpublished store, which nobody can be holding.
    coll.retired_quant.ensureUnusedCapacity(coll.alloc, 1) catch return error.OutOfMemory;
    const boxed = coll.alloc.create(quantized.Store) catch return error.OutOfMemory;
    boxed.* = store;

    // One-word publish; `searchQuantized` loads it under `SearchGuard`, so it
    // sees a whole store or none, and the one it saw stays allocated until no
    // reader can hold it (`reclaimRetired`). seq_cst on both, per
    // `SearchGuard.begin`: this store and that load are the two halves of the
    // same store-buffering pair as the graph pointer's.
    //
    // Under the write lock, so that `noteOverwrite`, which runs under the
    // same lock, never encodes into a store that has just been retired: the
    // one it loads is the one that is still published when it is done.
    coll.write_lock.lock();
    defer coll.write_lock.unlock();
    if (coll.quant.swap(boxed, .seq_cst)) |old| {
        coll.retired_quant.appendAssumeCapacity(old);
    }
}

/// What this particular build produced, beside how long it took.
///
/// §8.7 records that `buildParallel`'s graph depends on the thread
/// interleaving and offers `buildSerial` as the conformance fallback, so an
/// unstable checksum was known and accepted. What was never recorded is what
/// the instability is *worth*: findings 34 measured three builds of SIFT1M at
/// the same parameters on the same host reaching recall@10 of 0.99957, 0.99678
/// and 0.99337 at `ef` 512 — a spread wider than the distance between this
/// engine and Qdrant, against Qdrant's own 0.00009 over three builds.
///
/// A node nothing points at is invisible to search at any `ef`, so the orphan
/// count is where a graph-quality difference should show. Logged rather than
/// asserted: the number is the subject of an open question, and the harness
/// reads it out of `server.log` so a run can put each pass's graph beside that
/// pass's recall instead of inferring one from the other.
///
/// Costs one pass over the edges — about 16M visits on a 1M-point graph at
/// m=16, tens of milliseconds against a build measured in tens of seconds —
/// and is skipped entirely under `zig build test` with the rest of the
/// narration. A failure to allocate is not a failure to build: it prints
/// nothing and returns.
fn logGraphQuality(alloc: std.mem.Allocator, graph: *const hnsw.Graph) void {
    const r = build_hnsw.reachability(alloc, graph) catch return;
    // `seed` is on this line because nothing else in a run records it. §8.7
    // asks for it in `meta.json`; this is what lets a *run* carry it too. It
    // is provenance rather than a correction: under this level draw six builds
    // at four seeds sit within 0.00003 of recall@10 at `ef` 512 (`decisions.md`).
    std.debug.print(
        "index: graph checksum={x} nodes={d} unreachable={d} in_degree_zero={d} " ++
            "unreachable_with_out_edges={d} seed=0x{x}\n",
        .{
            graph.checksum(),             graph.count,
            graph.count -| r.reachable,   r.in_degree_zero,
            r.unreachable_with_out_edges, graph.params.seed,
        },
    );
}

/// Fraction of the collection allowed to sit outside the graph before a
/// rebuild is worth starting.
///
/// The pending tail is scanned exhaustively on every query (see `search`), so
/// its cost is linear and this ratio is the knob that bounds it. Ten percent of
/// a 1M collection is a 100k-vector scan, around 51 MB at d=128 (512-byte
/// padded rows), comparable to one graph traversal's random-access cost, so
/// the tail roughly doubles query time in the worst case before a rebuild
/// reclaims it.
///
/// Lower would rebuild more often. §6.5 puts the bulk build at 26 s for 1M
/// points; measurement puts it at 15-45 s for 1M at d=128, 250-348 s for
/// 990,000 at d=1536, and 105 s when it runs against the searches of the row
/// that triggered it (W11). It runs in the background, but it is
/// not free, it is not a constant, and a write-heavy workload should not spend
/// all its cores rebuilding.
///
/// This ratio is therefore a guess that has never been swept. It is the one
/// knob that decides whether W11 measures a rebuild or a graph, and 0.20 —
/// the ratio `w11_n` holds to the corpus — sits deliberately above it.
pub const rebuild_ratio: f64 = 0.10;

/// Whether `buildIndex` narrates itself.
///
/// Off under `zig build test`: the suite builds hundreds of small indexes and
/// 562 lines of rebuild narration buried the one line a failing test prints.
/// A running server rebuilds at a rate `rebuild_ratio` bounds, so there it is
/// two lines an hour, not chatter.
const log_rebuilds = !builtin.is_test;

/// True when enough points have accumulated outside the graph to justify a
/// rebuild.
///
/// "Outside the graph" is the appended tail plus every point overwritten in
/// place since the build (`overwritten_since_build`): an overwritten point is
/// *in* the graph, but at the position its old vector had, so for the purpose
/// of "is this graph still the graph of this collection" it counts the same
/// as a point the graph has never seen.
///
/// Reads `graph_count`, not `graph`: this runs on request threads with no
/// `SearchGuard`, and the graph itself may be retired and freed underneath.
pub fn needsRebuild(coll: *const Collection) bool {
    const total = coll.id_space.count();
    const covered_n = coll.graph_count.load(.acquire);
    if (covered_n == 0) return total > 0;
    std.debug.assert(covered_n <= total);
    const overwritten = coll.overwritten_since_build.load(.acquire);
    if (total <= covered_n and overwritten == 0) return false;
    const pending: f64 = @floatFromInt(total - covered_n + overwritten);
    const covered: f64 = @floatFromInt(covered_n);
    return pending / covered >= rebuild_ratio;
}

/// Record that points were written.
///
/// **This no longer drops the graph.** It used to: any upsert set the state to
/// `.absent`, so the next query brute-forced the entire collection and kept
/// doing so until a rebuild finished. W11 measured that collapse at 121 qps
/// against 1,645 for an engine that indexes incrementally, the one workload
/// where the strawman lost, and it lost by 15x.
///
/// A graph built over the first N points stays *correct* for those N points
/// when point N+1 arrives; it is merely incomplete. `search` covers the
/// difference by scanning the tail exhaustively, which costs recall nothing and
/// costs time in proportion to the tail. So the graph is kept and a rebuild is
/// scheduled only once the tail is large enough to matter.
pub fn invalidateIndex(coll: *Collection) void {
    switch (coll.index_state.load(.acquire)) {
        .ready => {
            // Keep serving from the graph; `search` scans whatever is past
            // `graph.count`. Only step aside for a rebuild when the tail has
            // grown past `rebuild_ratio`.
            // `indexed_count` is left alone. The graph is still traversed
            // (`choosePath`) and still covers its rows, so they are indexed;
            // zeroing it reported W11's bench2 as 0 of 1,250,000 indexed
            // while 1,000,000 were being answered from the graph, and Qdrant
            // in the same state reported 1,220,500. The rebuild's publish
            // stores the new count.
            if (needsRebuild(coll)) {
                coll.index_state.store(.absent, .release);
            }
        },
        // Points that arrive *during* a build are not in the graph it will
        // publish (`buildIndex` snapshots the count at entry), but nothing
        // needs recording here: the build re-runs this function's `.ready`
        // arm itself, under the write lock, immediately after publishing, so
        // it sees this write's count. Doing more here (a "stale" flag that
        // downgraded the publish to `.absent`) threw away a correct graph on
        // every write-during-build; doing less than the build's own check
        // stranded those points, `status()` Yellow forever with no rebuild
        // able to start.
        .building, .absent => {},
    }
}

/// How a query is answered. One closed set, resolved once, instead of the same
/// two booleans re-tested in each search entry point.
///
/// This was `!exact and index_state == .ready` here and
/// `coll.quant != .none and !params.exact` in the RPC handler, two conditions
/// in two files that between them selected one of four behaviours. Nothing
/// named the set, so nothing noticed when the paths diverged: the pending-tail
/// scan below was added to this function and not to the quantized one, and a
/// quantized collection silently stopped returning points written after its
/// last build.
pub const SearchPath = enum {
    /// §2's `params.exact`. Brute force over fp32, always: it is the recall
    /// ground truth (§6.5) and W9's benchmark, so it must never silently take
    /// the approximate path.
    exact,
    /// No graph published yet, so there is nothing to traverse.
    brute,
    /// Graph traversal over fp32, plus the pending tail.
    graph,
    /// §6.7's two-stage path: traverse on codes, rescore in fp32, plus the
    /// pending tail.
    quantized,
};

/// §2's `params.exact`, as a name rather than a bare `true`.
///
/// It travelled as a `bool` next to `ignore_quant`, another `bool`, so
/// `choosePath(coll, false, true)` compiled with either meaning and
/// `search(coll, q, ef, false, scratch, out)` said nothing at all at four of
/// its call sites. Two distinct enums cannot be transposed.
pub const Exactness = enum {
    /// Traverse the graph, or whatever cheaper representation is published.
    approximate,
    /// Brute force over fp32, always: it is the recall ground truth (§6.5) and
    /// W9's benchmark, so it must never silently take the approximate path.
    exact,
};

/// Whether a published quantized store may serve this query: §6.7's
/// `params.quantization.ignore`.
pub const QuantUse = enum {
    /// Take §6.7's two-stage path if a store is published.
    use,
    /// Answer from fp32 even though a store exists. Also what the fp32 entry
    /// point passes to say "not my path".
    ignore,
};

/// Resolve the path for one query.
///
/// `search` answers `.quantized` exactly as it answers `.graph`, because the
/// quantized store changes which *function* serves the query, not what this
/// one does.
pub fn choosePath(coll: *const Collection, exact: Exactness, quant: QuantUse) SearchPath {
    if (exact == .exact) return .exact;
    // Whatever graph is published is served, in every state. A graph built
    // over the first N rows stays correct for those rows when row N+1
    // arrives, and `scanPendingTail` covers the rest exhaustively; the state
    // says whether a *rebuild* is due or running (`status()`), not whether
    // the traversal is sound. Until 2026-08-19 only `.ready` traversed, so
    // stepping aside for a rebuild meant brute-forcing 1.2M rows per query
    // for the whole build: W11's 175x collapse (findings 25). The one hazard,
    // the rebuild replacing the pointer under a reader, is the same one a
    // `.ready` reader always had with a publish, and is closed the same way:
    // the old graph is retired, never freed, while a `SearchGuard` is held.
    if (publishedGraph(coll) == null) return .brute;
    if (coll.quant.load(.seq_cst) != null and quant == .use) return .quantized;
    return .graph;
}

/// The graph a search may traverse, or null. Loaded once per query, under
/// the caller's `SearchGuard`; a rebuild publishes with a release store.
pub inline fn publishedGraph(coll: *const Collection) ?*hnsw.Graph {
    // seq_cst rather than acquire: see `SearchGuard.begin`. A reader that
    // takes the guard and then reads a stale pointer here, while
    // `reclaimRetired` reads a stale zero from the counter, is exactly the
    // outcome acquire/release permits and this forbids.
    return @atomicLoad(?*hnsw.Graph, &coll.graph, .seq_cst);
}

/// Scan the points appended since the graph was built, merging them into `out`.
///
/// Shared by both graph paths rather than written twice. It is the W11 fix, and
/// leaving it out of one path is not a slower answer, it is a wrong one: the
/// tail is invisible to a traversal, so those points are absent from the result
/// set entirely.
///
/// Exhaustive on the tail means recall is *unaffected*: a pending point is
/// compared against the query directly, which is strictly better than what the
/// graph would have done for it. The cost is linear in the tail, which is why
/// the tail is bounded by `rebuild_ratio`.
pub fn scanPendingTail(coll: *const Collection, query: []const f32, covered: u32, out: *heap.TopK) void {
    scanPendingTailFiltered(coll, query, covered, null, out);
}

/// `scanPendingTail` under a filter. The tail is invisible to a traversal, so
/// a filter applied only there would let unindexed points through
/// unfiltered; this is the fourth place the predicate has to reach.
pub fn scanPendingTailFiltered(coll: *const Collection, query: []const f32, covered: u32, filter: ?hnsw.Index.Filter, out: *heap.TopK) void {
    const total: u32 = @intCast(coll.id_space.count());
    // The graph can never cover points that do not exist. If it does,
    // `bruteForceRange` would scan backwards over an empty range and the
    // pending tail would be silently skipped.
    std.debug.assert(covered <= total);
    if (total > covered) bruteForceRangeFiltered(coll, query, covered, total, filter, out);
}

/// The traversal predicate for one query: tombstones, plus whatever the
/// caller's filter admits. Null when neither applies, so the unfiltered
/// path on a collection with no deletes pays nothing.
///
/// `storage` is the caller's, because the composed predicate has to outlive
/// the call that builds it and a `Filter` is two pointers into it.
pub const Admission = struct {
    coll: *const Collection,
    extra: hnsw.Index.Filter,

    pub fn pred(ctx: *const anyopaque, node: u32) bool {
        const self: *const Admission = @ptrCast(@alignCast(ctx));
        return !self.coll.deleted.isSet(node) and self.extra.admits(node);
    }
};

/// A query's own view of the graph: nodes below `bound`, and whatever the
/// composed `Admission` admits.
///
/// A query answers from two regions, the traversal and `scanPendingTail` over
/// `[covered, total)`, where `covered` is one snapshot taken before the
/// traversal. With a graph that only ever grows by *replacement* the two can
/// never overlap. The moment anything inserts into the live graph they can, in
/// both directions: a node linked after the traversal and counted before the
/// tail scan is in neither, and a node whose edges exist while the snapshot is
/// behind is in *both* — and `heap.TopK.push` does not deduplicate, so the
/// client gets one point twice.
///
/// Bounding the traversal by the reader's own snapshot puts every node in
/// exactly one region. `bounded` is what builds it, and only when the graph
/// actually holds more than the reader covers, so a query on a graph that is
/// not being written pays nothing at all.
pub const Bounded = struct {
    bound: u32,
    inner: ?hnsw.Index.Filter,

    pub fn pred(ctx: *const anyopaque, node: u32) bool {
        const self: *const Bounded = @ptrCast(@alignCast(ctx));
        if (node >= self.bound) return false;
        return if (self.inner) |f| f.admits(node) else true;
    }
};

/// The traversal filter for a reader covering `[0, bound)`, or `inner`
/// unchanged when the graph holds no more than that.
pub fn bounded(inner: ?hnsw.Index.Filter, bound: u32, graph_count: usize, storage_: *Bounded) ?hnsw.Index.Filter {
    if (graph_count <= bound) return inner;
    storage_.* = .{ .bound = bound, .inner = inner };
    return .{ .pred = Bounded.pred, .ctx = storage_, .two_hop = if (inner) |f| f.two_hop else false };
}

pub fn admission(coll: *const Collection, extra: ?hnsw.Index.Filter, storage_: *Admission) ?hnsw.Index.Filter {
    if (extra) |e| {
        storage_.* = .{ .coll = coll, .extra = e };
        return .{ .pred = Admission.pred, .ctx = storage_, .two_hop = e.two_hop };
    }
    if (coll.deleted_count > 0) return .{ .pred = notDeleted, .ctx = coll };
    return null;
}

/// Extend the published graph by the point just appended, or leave it to the
/// tail scan.
///
/// Best effort by design: every reason to decline leaves the point exactly
/// where it would have been without the flag, in `[graph.count, total)`, which
/// `scanPendingTail` covers exhaustively. So a declined insert costs the tail
/// scan it already cost and never correctness.
///
/// Declines when there is no graph to extend, when the graph is already behind
/// (a tail exists, and filling the frontier out of order would strand the
/// nodes in between), and when the builder cannot be made or the CSR is full.
///
/// The caller holds `write_lock`.
fn liveInsert(coll: *Collection, offset: u32) void {
    const g = publishedGraph(coll) orelse return;
    if (@atomicLoad(usize, &g.count, .acquire) != offset) return;
    // Never into a quantized collection. §6.7's stage 1 traverses on *codes*,
    // and the quantizer encoded the points the build covered: a live-inserted
    // point would be reachable in the graph with no code to score it by. The
    // pending tail is what scores those, in fp32, and it only does so while the
    // graph does not claim them. Caught by the quantized path's differential
    // test against the exact scan, which returned a stale point as the nearest.
    if (coll.quant.load(.acquire) != null) return;

    if (coll.live_builder) |*b| {
        // A rebuild publishes a new graph; the builder follows it. The scratch
        // is sized to the collection's capacity, which does not change.
        b.graph = g;
    } else {
        coll.live_builder = build_hnsw.Builder.init(coll.alloc, g, scorerFor(coll)) catch return;
    }
    build_hnsw.insertLive(&coll.live_builder.?, offset) catch return;
    coll.graph_count.store(@as(usize, offset) + 1, .release);
    coll.indexed_count = @as(usize, offset) + 1;
}

/// Search over fp32, using the graph when one is published and brute force
/// otherwise. `searchQuantized` is the §6.7 counterpart.
pub fn search(
    coll: *const Collection,
    query: []const f32,
    ef: usize,
    exact: Exactness,
    scratch: ?*hnsw.Index.Scratch,
    out: *heap.TopK,
) void {
    searchFiltered(coll, query, ef, exact, scratch, null, out);
}

/// `search` admitting only what `filter` admits: on the graph (as results
/// are admitted, see `hnsw.Index.Filter`), on the pending tail, and on the
/// exact scan. The graph is still traversed through rejected nodes, so a
/// selective filter costs a wider walk rather than a disconnected one.
pub fn searchFiltered(
    coll: *const Collection,
    query: []const f32,
    ef: usize,
    exact: Exactness,
    scratch: ?*hnsw.Index.Scratch,
    extra: ?hnsw.Index.Filter,
    out: *heap.TopK,
) void {
    // Held across the whole search, including the pending-tail scan, because
    // everything inside may read a pointer a rebuild is about to replace.
    const guard = SearchGuard.begin(coll);
    defer guard.end();
    var adm: Admission = undefined;
    const filter = admission(coll, extra, &adm);

    switch (choosePath(coll, exact, .ignore)) {
        .exact, .brute => {},
        // `.ignore` above: this path never chooses the quantized store.
        .quantized => unreachable,
        .graph => {
            // One load for the whole query: the traversal and the tail scan
            // must agree on which graph, and therefore which `count`.
            const g = publishedGraph(coll).?;
            // One load of the count, before the traversal, used for both the
            // traversal and the tail scan below.
            //
            // The tail is `[covered, total)`, so the two readings have to come
            // from the same instant or a node can fall between them: linked
            // into the graph after this query traversed, counted before this
            // query scanned the tail, and therefore in neither. Today the
            // count cannot move under a reader — `publishedGraph` is loaded
            // once and a rebuild publishes a whole new graph — so this is a
            // no-op that stops being one the moment anything inserts into a
            // live graph.
            const covered: u32 = @intCast(@atomicLoad(usize, &g.count, .acquire));
            // Nodes past this reader's snapshot belong to its tail scan, not to
            // its traversal, or they land in both and the client sees a point
            // twice (`Bounded`). Costs nothing while nothing inserts into the
            // live graph, because then the graph holds exactly `covered`.
            var bnd: Bounded = undefined;
            const traversal_filter = bounded(filter, covered, @atomicLoad(usize, &g.count, .acquire), &bnd);
            if (scratch) |sc| {
                var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
                const probe = Probe.init(coll, query, &buf);
                const idx = hnsw.Index{ .graph = g, .scorer = .of(&probe) };
                // Tombstones are dropped as results are admitted to `out`, not
                // after it has been truncated to `k`, see `Index.Filter`.
                idx.searchFiltered(ef, sc, out, traversal_filter);

                // Points appended since the graph was built are not in it, so
                // scan them exhaustively and let `out` merge the two sets.
                //
                // This is what keeps a write from destroying read performance.
                // The old behaviour dropped the whole graph on any upsert, so a
                // single new point sent every subsequent query through a brute
                // force of the entire collection until a rebuild finished, a
                // 15x collapse in W11 (121 qps against 1,645 for an engine that
                // indexes incrementally). The graph is still correct for the
                // points it covers; only the tail is missing, and the tail is
                // small by construction because `ensureIndexBuilding` starts a
                // rebuild once it grows past `rebuild_ratio`.
                scanPendingTailFiltered(coll, query, covered, extra, out);
                return;
            }
        },
    }
    bruteForceRangeFiltered(coll, query, 0, @intCast(coll.id_space.count()), extra, out);
}

pub fn notDeleted(ctx: *const anyopaque, node: u32) bool {
    const coll: *const Collection = @ptrCast(@alignCast(ctx));
    return !coll.deleted.isSet(node);
}

test "index build flips the collection to Green and search uses the graph" {
    var c = try makeCollection(16, .euclid, 2048);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0xbeef11);
    const rnd = prng.random();
    const n = 1000;
    for (0..n) |i| {
        var v: [16]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    // Before the build: no graph, count not indexed.
    try testing.expectEqual(IndexState.absent, c.index_state.load(.acquire));
    try testing.expectEqual(@as(usize, 0), c.indexed_count);

    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    try testing.expectEqual(Status.green, c.status());
    try testing.expectEqual(@as(usize, n), c.indexed_count);
    try testing.expect(c.graph != null);

    // And the graph path agrees with brute force on most queries.
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 2048, 256);
    defer scratch.deinit(testing.allocator);

    var hits: usize = 0;
    const queries = 100;
    const k = 10;
    for (0..queries) |_| {
        var q: [16]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);

        var tb: [k]Candidate = undefined;
        var truth = heap.TopK.init(&tb, k);
        bruteForce(&c, &q, &truth);
        const t = truth.finish();

        var gb: [k]Candidate = undefined;
        var got = heap.TopK.init(&gb, k);
        search(&c, &q, 128, .approximate, &scratch, &got);
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
    try testing.expect(recall > 0.95);
}

test "exact:true bypasses the graph even when one exists" {
    // §6.5: the exact path is the recall ground truth, so it must never
    // silently use the approximate index.
    var c = try makeCollection(8, .dot, 512);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x1234);
    const rnd = prng.random();
    for (0..200) |i| {
        var v: [8]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);

    var q: [8]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    var eb: [10]Candidate = undefined;
    var exact_out = heap.TopK.init(&eb, 10);
    search(&c, &q, 64, .exact, null, &exact_out);

    var bb: [10]Candidate = undefined;
    var brute = heap.TopK.init(&bb, 10);
    bruteForce(&c, &q, &brute);

    const e = exact_out.finish();
    const b = brute.finish();
    try testing.expectEqual(b.len, e.len);
    for (e, b) |x, y| try testing.expectEqual(y.id, x.id);
}

test "deleted points are filtered from graph results without disconnecting it" {
    var c = try makeCollection(8, .euclid, 1024);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x777);
    const rnd = prng.random();
    const n = 400;
    for (0..n) |i| {
        var v: [8]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);

    // Delete a quarter of the points.
    for (0..n / 4) |i| _ = c.delete(.{ .num = i * 4 });

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
    defer scratch.deinit(testing.allocator);

    var q: [8]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);
    var gb: [20]Candidate = undefined;
    var got = heap.TopK.init(&gb, 20);
    search(&c, &q, 128, .approximate, &scratch, &got);

    for (got.items[0..got.len]) |cand| {
        try testing.expect(!c.deleted.isSet(cand.id));
    }
    // The graph is still navigable: we still get results back.
    try testing.expect(got.len > 0);
}

test "a write keeps the graph and the pending tail is searched exhaustively" {
    // Pins the flag-off state machine: a write keeps the graph, the tail is
    // scanned, and the graph is dropped once the tail is worth rebuilding for.
    // `-Dlive-insert` deliberately removes the tail, so there is nothing here
    // to observe and the behaviour it would assert is the other arm's.
    if (build_options.live_insert) return error.SkipZigTest;
    // The old contract dropped the graph on any upsert, so one new point sent
    // every subsequent query through a brute force of the whole collection.
    // W11 measured that at 121 qps against 1,645 for an engine that indexes
    // incrementally. The graph is still correct for the points it covers.
    var c = try makeCollection(4, .euclid, 4096);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();

    const indexed = 800;
    for (0..indexed) |i| {
        var v: [4]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));

    // A handful of new points: well under `rebuild_ratio`, so the graph stays.
    var target: [4]f32 = .{ 9, 9, 9, 9 };
    for (0..20) |i| {
        var v: [4]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = indexed + i }, &v);
    }
    // The last point is the exact answer to `target`, and it is in the tail.
    const answer = try c.upsert(.{ .num = 9999 }, &target);

    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    try testing.expect(!needsRebuild(&c));
    // Green, with `indexed_count < count()`: the tail is served exhaustively
    // so every answer is complete, which is what Qdrant reports for a small
    // unindexed segment and what bfb's `wait_index` loop needs to see. It was
    // Yellow, and since a tail this size never triggers a rebuild it stayed
    // Yellow forever, so any write after the first build wedged the poll.
    try testing.expectEqual(Status.green, c.status());
    try testing.expect(c.indexed_count < c.count());

    // And the tail is found: exhaustive on the tail means recall is unaffected
    // by a point being outside the graph.
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 4096, 128);
    defer scratch.deinit(testing.allocator);
    var gb: [8]Candidate = undefined;
    var got = heap.TopK.init(&gb, 5);
    search(&c, &target, 64, .approximate, &scratch, &got);
    // `finish` sorts best-first; `items[0]` before that is the heap root, which
    // for a keep-top-k heap is the *worst* of the k.
    const ranked = got.finish();
    try testing.expect(ranked.len > 0);
    try testing.expectEqual(answer, ranked[0].id);
}

test "a two-hop filtered walk returns only admitted, live points and keeps recall" {
    // ACORN-1 against the exact answer over the same bits, at two
    // selectivities, with tombstones among the admitted points: rejected
    // nodes are hopped through rather than scored, so a result the filter or
    // a tombstone rejects would mean the hop leaked one into the heap.
    const dim = 32;
    const n = 4000;
    var c = try makeCollection(dim, .euclid, n);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0xac0e1);
    const rnd = prng.random();
    var v: [dim]f32 = undefined;
    for (0..n) |i| {
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    for (0..n) |i| if (i % 97 == 0) {
        _ = c.delete(.{ .num = i });
    };

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, n, 256);
    defer scratch.deinit(testing.allocator);
    var bits = [_]u64{0} ** ((n + 63) / 64);
    for ([_]u32{ 10, 50 }) |every| {
        @memset(&bits, 0);
        for (0..n) |i| if (rnd.uintLessThan(u32, every) == 0) {
            bits[i / 64] |= @as(u64, 1) << @intCast(i % 64);
        };
        const bctx = payload_mod.BitsCtx{ .bits = &bits, .bound = n };
        const k = 10;
        const queries = 60;
        var hits: usize = 0;
        var total: usize = 0;
        for (0..queries) |_| {
            for (&v) |*x| x.* = rnd.floatNorm(f32);
            var tb: [k]Candidate = undefined;
            var truth = heap.TopK.init(&tb, k);
            searchSelected(&c, &v, &bits, n, &truth);
            const t = truth.finish();

            var gb: [k]Candidate = undefined;
            var got = heap.TopK.init(&gb, k);
            searchFiltered(&c, &v, 128, .approximate, &scratch, .{
                .pred = payload_mod.BitsCtx.pred,
                .ctx = &bctx,
                .two_hop = true,
            }, &got);
            const g = got.finish();
            try testing.expectEqual(t.len, g.len);
            for (g) |gc| {
                try testing.expect(payload_mod.testBit(&bits, gc.id));
                try testing.expect(!c.deleted.isSet(gc.id));
                for (t) |tc| {
                    if (tc.id == gc.id) {
                        hits += 1;
                        break;
                    }
                }
            }
            total += t.len;
        }
        const recall = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total));
        try testing.expect(recall >= 0.9);
    }
}

test "a build asked to stop publishes nothing, on either path" {
    inline for (.{ BuildMode.serial, BuildMode.parallel }) |mode| {
        var c = try makeCollection(4, .dot, 4096);
        defer c.deinit();
        var v = [_]f32{ 1, 0, 0, 0 };
        for (0..2000) |i| _ = try c.upsert(.{ .num = i }, &v);
        c.drop_requested.store(true, .release);
        try testing.expectError(error.Cancelled, buildIndex(&c, mode, 4));
        try testing.expect(publishedGraph(&c) == null);
    }
}

test "the graph is only dropped once the tail is worth rebuilding for" {
    // Pins the flag-off state machine: a write keeps the graph, the tail is
    // scanned, and the graph is dropped once the tail is worth rebuilding for.
    // `-Dlive-insert` deliberately removes the tail, so there is nothing here
    // to observe and the behaviour it would assert is the other arm's.
    if (build_options.live_insert) return error.SkipZigTest;
    var c = try makeCollection(4, .dot, 4096);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    const indexed = 1000;
    for (0..indexed) |i| _ = try c.upsert(.{ .num = i }, &v);
    try buildIndex(&c, .serial, 1);

    // Just under the ratio: keep serving from the graph.
    // `invalidateIndex` is called by the API layer once per batch rather than
    // by `upsert` per point, a 1000-point batch should evaluate the rebuild
    // decision once, not a thousand times.
    const under = @as(usize, @intFromFloat(@as(f64, indexed) * rebuild_ratio)) - 5;
    for (0..under) |i| _ = try c.upsert(.{ .num = indexed + i }, &v);
    invalidateIndex(&c);
    try testing.expect(!needsRebuild(&c));
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));

    // Cross it: now a rebuild is worth the cores, so step aside for one.
    for (0..20) |i| _ = try c.upsert(.{ .num = indexed + under + i }, &v);
    invalidateIndex(&c);
    try testing.expect(needsRebuild(&c));
    try testing.expectEqual(IndexState.absent, c.index_state.load(.acquire));
    try testing.expectEqual(Status.yellow, c.status());
}

test "concurrent upserts do not lose points or cross-wire ids" {
    // The worker pool (§6.1) runs `handler` on N threads pulling from one
    // shared queue, so nothing serialises two Upsert RPCs to the same
    // collection. Without `write_lock` this test loses points reliably: both
    // threads read the same `id_space.next`, both write the same row, and an
    // external id resolves to a stranger's vector.
    var c = try makeCollection(4, .dot, 4096);
    defer c.deinit();

    const threads = 4;
    const per = 500;
    const Worker = struct {
        fn run(coll: *Collection, base: usize) void {
            for (0..per) |i| {
                const id = base * per + i;
                // Each id gets a vector that encodes it, so a crossed mapping
                // is detectable rather than merely suspected.
                var v = [_]f32{ @floatFromInt(id), 1, 2, 3 };
                _ = coll.upsert(.{ .num = id }, &v) catch unreachable;
            }
        }
    };
    var handles: [threads]std.Thread = undefined;
    for (&handles, 0..) |*h, t| h.* = try std.Thread.spawn(.{}, Worker.run, .{ &c, t });
    for (handles) |h| h.join();

    try testing.expectEqual(@as(usize, threads * per), c.count());

    // Every id resolves, and resolves to *its own* vector.
    for (0..threads * per) |id| {
        const off = c.id_space.lookup(.{ .num = id }) orelse return error.PointLost;
        const row = c.space.rowConst(off);
        try testing.expectEqual(@as(f32, @floatFromInt(id)), row[0]);
    }
}

test "points upserted during a build are not stranded outside the index" {
    // Two bugs, one on each side of this test. `invalidateIndex` used to act
    // only on `.ready`, so a point arriving mid-build left no trace: the
    // build published `.ready` with a stale count, `status()` stayed Yellow
    // forever, and `ensureIndexBuilding`'s `.absent -> .building` CAS could
    // never win again. The fix for that was a flag that sent *every* publish
    // with a write behind it to `.absent`, which threw away a correct graph
    // under sustained writes and brute-forced every query until the next
    // build, which the next write invalidated in turn (W11 never reached
    // `.ready`). The property is: the build publishes, the written point is
    // searchable at once through the tail scan, and the ratio decides.
    const dim = 4;
    const n = 400;
    var c = try makeCollection(dim, .euclid, 1024);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x57a1e);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    // `build_hook` runs on the build thread after the graph is built and
    // before it is published, exactly where an upsert during a build lands.
    const H = struct {
        var target: [dim]f32 = .{ 9, 9, 9, 9 };
        var appended: u32 = 0;
        fn one(coll: *Collection) void {
            appended = coll.upsert(.{ .num = 9999 }, &target) catch unreachable;
        }
    };
    c.build_hook = H.one;
    c.index_state.store(.building, .release);
    try buildIndex(&c, .serial, 1);
    c.build_hook = null;

    // Published, not thrown away: one point is under the ratio.
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    try testing.expectEqual(@as(usize, n), c.indexed_count);
    try testing.expectEqual(@as(usize, n + 1), c.count());
    try testing.expect(!needsRebuild(&c));
    // Green with `indexed_count < count()`, the same answer as for a write
    // after publish (see "a write keeps the graph...").
    try testing.expectEqual(Status.green, c.status());
    try testing.expect(c.indexed_count < c.count());

    // The point written during the build is found, via the tail scan.
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
    defer scratch.deinit(testing.allocator);
    var gb: [8]Candidate = undefined;
    var got = heap.TopK.init(&gb, 5);
    search(&c, &H.target, 64, .approximate, &scratch, &got);
    const ranked = got.finish();
    try testing.expect(ranked.len > 0);
    try testing.expectEqual(H.appended, ranked[0].id);
}

test "the graph path returns a full page even with heavy tombstoning" {
    // Filtering tombstones *after* truncating to k returns short lists, so the
    // graph path and the exact path disagreed on result count, not just on
    // order. A client paging results sees a partial page and stops early.
    var c = try makeCollection(8, .euclid, 2048);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0xd00d);
    const rnd = prng.random();
    const n = 1000;
    for (0..n) |i| {
        var v: [8]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);

    // 30% tombstones: enough that a k=10 query previously came back with ~7.
    for (0..n) |i| {
        if (i % 10 < 3) _ = c.delete(.{ .num = i });
    }

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 4096, 512);
    defer scratch.deinit(testing.allocator);

    const k = 10;
    for (0..25) |_| {
        var q: [8]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);

        var gb: [64]Candidate = undefined;
        var got = heap.TopK.init(&gb, k);
        search(&c, &q, 64, .approximate, &scratch, &got);

        try testing.expectEqual(@as(usize, k), got.len);
        for (got.items[0..got.len]) |cand| try testing.expect(!c.deleted.isSet(cand.id));
    }
}

test "building an empty collection is Green immediately" {
    var c = try makeCollection(4, .dot, 16);
    defer c.deinit();
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(Status.green, c.status());
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
}

test "an empty build leaves a state the first upsert can build from" {
    // The `n == 0` early return published `.ready` and skipped the rest of
    // the publish path, so a stale flag it never cleared left `.ready` over
    // no graph: Yellow, and no poll could start a build until another upsert
    // happened to invalidate. From `.ready`, the first upsert's
    // `invalidateIndex` must step aside.
    var c = try makeCollection(4, .dot, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };

    c.index_state.store(.building, .release);
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));

    // The first upsert after an empty build must trigger a build.
    _ = try c.upsert(.{ .num = 1 }, &v);
    invalidateIndex(&c);
    try testing.expectEqual(IndexState.absent, c.index_state.load(.acquire));
    try testing.expectEqual(Status.yellow, c.status());
    try testing.expectEqual(@as(?IndexState, null), c.index_state.cmpxchgStrong(.absent, .building, .acq_rel, .monotonic));
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    try testing.expectEqual(Status.green, c.status());
    try testing.expectEqual(@as(usize, 1), c.indexed_count);
}

// =========================================================================
// Storage datatypes (§5.4's working-set argument, as an element width)
// =========================================================================

test "uint8 storage keeps qdrant's saturating cast, all the way through ingest" {
    // The conversion is unit-tested in `dist/datatype.zig` against qdrant's own
    // pinned vector; this is the same values arriving through `upsert`, which
    // is where a client actually meets them.
    var c = try makeCollectionDt(6, .dot, .uint8, 4);
    defer c.deinit();

    var v = [_]f32{ -10.0, 1.0, 2.0, 3.0, 255.0, 300.0 };
    const off = try c.upsert(.{ .num = 1 }, &v);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 255, 255 }, c.space.rowU8(off));

    // And a client that reads it back gets the clamped values, not the ones it
    // sent. Silent, and Qdrant's behaviour, which is why it is pinned here.
    var widened: [6]f32 = undefined;
    try testing.expectEqualSlices(
        f32,
        &[_]f32{ 0, 1, 2, 3, 255, 255 },
        c.space.readInto(off, &widened),
    );
}

test "a narrow datatype is a narrower arena, which is the whole point" {
    const dim = 768;
    var wide = try makeCollectionDt(dim, .dot, .float32, 1000);
    defer wide.deinit();
    var half = try makeCollectionDt(dim, .dot, .float16, 1000);
    defer half.deinit();
    var byte = try makeCollectionDt(dim, .dot, .uint8, 1000);
    defer byte.deinit();

    // §5.2's table: 3072 B/vector fp32, 1536 fp16, 768 uint8 at d=768.
    try testing.expectEqual(@as(usize, 3072), wide.space.stride);
    try testing.expectEqual(@as(usize, 1536), half.space.stride);
    try testing.expectEqual(@as(usize, 768), byte.space.stride);
    try testing.expectEqual(wide.space.arena.len / 2, half.space.arena.len);
    try testing.expectEqual(wide.space.arena.len / 4, byte.space.arena.len);
}

test "§8.3's ingest normalisation follows the storage type, not the metric alone" {
    const v = [_]f32{ 3, 4, 0, 0 };

    // fp32 cosine normalises at ingest, so the stored row is a unit vector and
    // the search path is a plain dot product.
    var f = try makeCollectionDt(4, .cosine, .float32, 4);
    defer f.deinit();
    var a = v;
    const off = try f.upsert(.{ .num = 1 }, &a);
    try testing.expectEqual(@as(usize, 1), f.normalized_count);
    try testing.expectApproxEqAbs(@as(f32, 0.6), f.space.rowConst(off)[0], 1e-6);

    // uint8 cosine does not, and cannot: 0.6 would truncate to 0. Qdrant's
    // byte cosine preprocesses with the identity and divides by the norms at
    // query time instead, and this matches it.
    var b = try makeCollectionDt(4, .cosine, .uint8, 4);
    defer b.deinit();
    var bv = v;
    const boff = try b.upsert(.{ .num = 1 }, &bv);
    try testing.expectEqual(@as(usize, 0), b.normalized_count);
    try testing.expectEqualSlices(u8, &[_]u8{ 3, 4, 0, 0 }, b.space.rowU8(boff));
}

test "uint8 cosine scores match qdrant's formula, including its zero-norm case" {
    var c = try makeCollectionDt(4, .cosine, .uint8, 8);
    defer c.deinit();

    var a = [_]f32{ 3, 4, 0, 0 };
    var b = [_]f32{ 6, 8, 0, 0 }; // same direction, twice the length
    var z = [_]f32{ 0, 0, 0, 0 };
    _ = try c.upsert(.{ .num = 1 }, &a);
    const same_dir = try c.upsert(.{ .num = 2 }, &b);
    const zero = try c.upsert(.{ .num = 3 }, &z);

    var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
    const q = [_]f32{ 3, 4, 0, 0 };
    const probe = Probe.init(&c, &q, &buf);

    // Parallel vectors: cosine 1.0 regardless of length, which is the property
    // that would have been destroyed by normalising into bytes.
    try testing.expectApproxEqAbs(@as(f32, 1.0), probe.scoreNode(same_dir), 1e-6);

    // `cosine_similarity_bytes` returns 0.0 for a zero norm rather than a NaN,
    // and a NaN here would poison the result heap's ordering.
    try testing.expectEqual(@as(f32, 0.0), probe.scoreNode(zero));
}

test "uint8 dot, euclid and manhattan are exact integers, not approximations" {
    var prng = std.Random.DefaultPrng.init(0x8b17);
    const rnd = prng.random();
    const dim = 96;

    for ([_]Metric{ .dot, .euclid, .manhattan }) |m| {
        var c = try makeCollectionDt(dim, m, .uint8, 16);
        defer c.deinit();

        var row: [dim]f32 = undefined;
        var qv: [dim]f32 = undefined;
        for (&row, &qv) |*x, *y| {
            x.* = @floatFromInt(rnd.int(u8));
            y.* = @floatFromInt(rnd.int(u8));
        }
        const off = try c.upsert(.{ .num = 1 }, &row);

        var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
        const probe = Probe.init(&c, &qv, &buf);

        var want: i32 = 0;
        for (row, qv) |x, y| {
            const a: i32 = @intFromFloat(x);
            const b: i32 = @intFromFloat(y);
            switch (m) {
                .dot => want += a * b,
                .euclid => want -= (a - b) * (a - b),
                .manhattan => want -= @intCast(@abs(a - b)),
                .cosine => unreachable,
            }
        }
        // Integer arithmetic end to end: exact equality, not a tolerance.
        try testing.expectEqual(@as(f32, @floatFromInt(want)), probe.scoreNode(off));
    }
}

test "float16 search finds the same neighbours as fp32 on data f16 can hold" {
    // f16 has an 11-bit significand, so this uses values it represents
    // exactly-ish and asks for the *ranking* to survive, which is what a
    // storage datatype has to promise.
    const dim = 64;
    const n = 300;
    var prng = std.Random.DefaultPrng.init(0x1f16);
    const rnd = prng.random();

    var wide = try makeCollectionDt(dim, .euclid, .float32, 512);
    defer wide.deinit();
    var half = try makeCollectionDt(dim, .euclid, .float16, 512);
    defer half.deinit();

    var vecs: [n][dim]f32 = undefined;
    for (&vecs, 0..) |*v, i| {
        for (v) |*x| x.* = rnd.floatNorm(f32);
        _ = try wide.upsert(.{ .num = i }, v);
        _ = try half.upsert(.{ .num = i }, v);
    }

    var q: [dim]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    var wb: [8]Candidate = undefined;
    var hb: [8]Candidate = undefined;
    var w_top = heap.TopK.init(&wb, 5);
    var h_top = heap.TopK.init(&hb, 5);
    bruteForce(&wide, &q, &w_top);
    bruteForce(&half, &q, &h_top);

    const wr = w_top.finish();
    const hr = h_top.finish();
    try testing.expectEqual(wr.len, hr.len);
    // The nearest neighbour must survive the storage conversion; the tail of
    // the list may reorder where f16 rounds two candidates together, and
    // pretending otherwise would make this test a lie about precision.
    try testing.expectEqual(wr[0].id, hr[0].id);
    var agree: usize = 0;
    for (hr) |h| {
        for (wr) |w| {
            if (w.id == h.id) agree += 1;
        }
    }
    try testing.expect(agree >= 4);
}

test "the graph builds and searches on every datatype" {
    const dim = 32;
    const n = 500;
    for ([_]dist.Datatype{ .float32, .float16, .uint8 }) |dt| {
        var c = try makeCollectionDt(dim, .euclid, dt, 1024);
        defer c.deinit();

        var prng = std.Random.DefaultPrng.init(0x9e3 + @as(u64, @intFromEnum(dt)));
        const rnd = prng.random();
        var target: [dim]f32 = undefined;
        for (&target) |*x| x.* = 200;

        for (0..n) |i| {
            var v: [dim]f32 = undefined;
            // Values inside u8's range so the same corpus is meaningful for
            // every datatype; a negative one would clamp to zero and the
            // comparison would be between different collections.
            for (&v) |*x| x.* = @floatFromInt(rnd.uintLessThan(u8, 100));
            _ = try c.upsert(.{ .num = i }, &v);
        }
        const answer = try c.upsert(.{ .num = 9999 }, &target);
        try buildIndex(&c, .serial, 1);
        try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));

        var scratch = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
        defer scratch.deinit(testing.allocator);
        var gb: [16]Candidate = undefined;
        var got = heap.TopK.init(&gb, 5);
        search(&c, &target, 64, .approximate, &scratch, &got);

        const ranked = got.finish();
        try testing.expect(ranked.len > 0);
        try testing.expectEqual(answer, ranked[0].id);
    }
}

test "a dimension the query buffer cannot hold is refused at creation" {
    // fp32 converts nothing and is unaffected by the cap.
    var ok = try Collection.init(testing.allocator, "t", .{
        .dim = max_converted_dim + 1,
        .metric = .dot,
        .capacity = 1,
    });
    ok.deinit();

    try testing.expectError(Error.DimensionTooLargeForDatatype, Collection.init(
        testing.allocator,
        "t",
        .{ .dim = max_converted_dim + 1, .metric = .dot, .datatype = .float16, .capacity = 1 },
    ));
}

test "a rebuild after an append extends the graph, and an overwrite forces a full one" {
    // The guard that matters is the overwrite one. An extension keeps every
    // edge the published graph had, and a row overwritten in place sits in
    // that graph at the position its *old* vector occupied — so extending
    // over it would carry a wrong neighbour set forward silently, which is
    // the failure mode this whole project is built to refuse.
    var c = try makeCollection(16, .euclid, 4096);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0xe5a17);
    const rnd = prng.random();
    var vec: [16]f32 = undefined;
    const put = struct {
        fn one(coll: *Collection, id: u64, v: []f32, r: std.Random) !void {
            for (v) |*x| x.* = r.floatNorm(f32);
            _ = try coll.upsert(.{ .num = id }, v);
        }
    }.one;

    for (0..400) |i| try put(&c, i, &vec, rnd);
    try buildIndex(&c, .parallel, 4);
    const first = publishedGraph(&c).?.count;
    try testing.expectEqual(@as(usize, 400), first);

    // Append: the next build must extend, and must not lose the old nodes.
    for (400..600) |i| try put(&c, i, &vec, rnd);
    try buildIndex(&c, .parallel, 4);
    const g2 = publishedGraph(&c).?;
    try testing.expectEqual(@as(usize, 600), g2.count);
    // Every point, old and new, still answers an exact self-query. A dropped
    // or mislinked prefix shows up here and nowhere else.
    var buf: [1]heap.Candidate = undefined;
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 600, 256);
    defer scratch.deinit(testing.allocator);
    for ([_]u32{ 0, 199, 399, 400, 599 }) |id| {
        var out = heap.TopK.init(&buf, 1);
        search(&c, c.space.rowConst(id), 64, .approximate, &scratch, &out);
        const got = out.finish();
        try testing.expect(got.len == 1);
        try testing.expectEqual(id, got[0].id);
    }

    // Overwrite in place: `overwritten_since_build` is nonzero, so the next
    // build must NOT extend.
    try put(&c, 42, &vec, rnd);
    try testing.expect(c.overwritten_since_build.load(.acquire) > 0);
    try buildIndex(&c, .parallel, 4);
    try testing.expectEqual(@as(usize, 600), publishedGraph(&c).?.count);
    try testing.expectEqual(@as(usize, 0), c.overwritten_since_build.load(.acquire));
}

test "the previous graph is served while the collection has stepped aside and while it rebuilds" {
    // Pins the flag-off state machine: a write keeps the graph, the tail is
    // scanned, and the graph is dropped once the tail is worth rebuilding for.
    // `-Dlive-insert` deliberately removes the tail, so there is nothing here
    // to observe and the behaviour it would assert is the other arm's.
    if (build_options.live_insert) return error.SkipZigTest;
    // W11's mechanism (findings 25): once the tail crossed `rebuild_ratio`
    // the state left `.ready` and every query brute-forced the whole
    // collection until the rebuild published. The graph it had stepped
    // aside from was still correct for the rows it covered, and the tail is
    // scanned exhaustively on every path, so nothing was gained by not
    // traversing it. Pinned here: `.absent`-with-graph and `.building`
    // traverse, the tail is found, and a rebuild publishing under a reader
    // retires the old graph rather than freeing it.
    const dim = 16;
    const n = 600;
    // Euclid, so a stored vector's nearest neighbour is itself.
    var c = try makeCollection(dim, .euclid, 4096);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x5e7e);
    const rnd = prng.random();
    var stored = try testing.allocator.alloc(f32, 2 * n * dim);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    for (0..n) |i| _ = try c.upsert(.{ .num = i }, stored[i * dim ..][0..dim]);
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    const g0 = c.graph.?;

    // Append past the ratio: the collection steps aside for a rebuild...
    for (n..2 * n) |i| _ = try c.upsert(.{ .num = i }, stored[i * dim ..][0..dim]);
    invalidateIndex(&c);
    try testing.expectEqual(IndexState.absent, c.index_state.load(.acquire));
    try testing.expectEqual(Status.yellow, c.status());
    // The rows the graph covers are still indexed, and the count says so
    // rather than dropping to zero.
    try testing.expectEqual(@as(usize, n), c.indexed_count);
    // ...and still traverses the graph it had, rather than brute-forcing.
    try testing.expectEqual(SearchPath.graph, choosePath(&c, .approximate, .use));
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 4096, 128);
    defer scratch.deinit(testing.allocator);
    var results: [10]heap.Candidate = undefined;
    const Q = struct {
        fn top1(coll: *Collection, q: []const f32, sc: *hnsw.Index.Scratch, buf: []heap.Candidate) u32 {
            var top = heap.TopK.init(buf, 1);
            search(coll, q, 64, .approximate, sc, &top);
            return top.finish()[0].id;
        }
    };
    // An indexed point and a tail point both lead their own result.
    try testing.expectEqual(c.id_space.lookup(.{ .num = 7 }).?, Q.top1(&c, stored[7 * dim ..][0..dim], &scratch, &results));
    try testing.expectEqual(c.id_space.lookup(.{ .num = n + 5 }).?, Q.top1(&c, stored[(n + 5) * dim ..][0..dim], &scratch, &results));

    // Now a rebuild, held open by the hook while searches run against the
    // old graph from another thread, then released: the publish replaces
    // the pointer under those readers, and the old graph is retired, not
    // freed, for as long as one of them holds it.
    const H = struct {
        var gate = std.atomic.Value(bool).init(false);
        var in_hook = std.atomic.Value(bool).init(false);
        fn hook(_: *Collection) void {
            in_hook.store(true, .release);
            while (!gate.load(.acquire)) std.Thread.yield() catch {};
        }
        fn searcher(coll: *Collection, vecs: []const f32, failures: *std.atomic.Value(u32), stop: *std.atomic.Value(bool)) void {
            var sc = hnsw.Index.Scratch.init(testing.allocator, 4096, 128) catch {
                _ = failures.fetchAdd(1, .monotonic);
                return;
            };
            defer sc.deinit(testing.allocator);
            var buf: [10]heap.Candidate = undefined;
            var i: usize = 0;
            while (!stop.load(.acquire)) : (i += 1) {
                const id = (i * 37) % (2 * n);
                const got = Q.top1(coll, vecs[id * dim ..][0..dim], &sc, &buf);
                if (got != coll.id_space.lookup(.{ .num = id }).?) _ = failures.fetchAdd(1, .monotonic);
            }
        }
    };
    H.gate.store(false, .release);
    H.in_hook.store(false, .release);
    c.build_hook = H.hook;
    var failures = std.atomic.Value(u32).init(0);
    var stop = std.atomic.Value(bool).init(false);
    const t1 = try std.Thread.spawn(.{}, H.searcher, .{ &c, stored, &failures, &stop });
    const t2 = try std.Thread.spawn(.{}, H.searcher, .{ &c, stored, &failures, &stop });
    // The state machine the API layer drives: `.absent -> .building`, build,
    // publish.
    try testing.expect(c.index_state.cmpxchgStrong(.absent, .building, .acq_rel, .monotonic) == null);
    const builder = try std.Thread.spawn(.{}, struct {
        fn run(coll: *Collection) void {
            buildIndex(coll, .serial, 1) catch unreachable;
        }
    }.run, .{&c});
    while (!H.in_hook.load(.acquire)) std.Thread.yield() catch {};
    // Mid-build: still the old graph, still traversed.
    try testing.expectEqual(IndexState.building, c.index_state.load(.acquire));
    try testing.expectEqual(SearchPath.graph, choosePath(&c, .approximate, .use));
    try testing.expectEqual(g0, publishedGraph(&c).?);
    // Let it publish under the two readers.
    H.gate.store(true, .release);
    builder.join();
    c.build_hook = null;
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    try testing.expect(publishedGraph(&c).? != g0);
    // Readers keep going against the new graph for a moment, then stop.
    for (0..2000) |_| std.Thread.yield() catch {};
    stop.store(true, .release);
    t1.join();
    t2.join();
    try testing.expectEqual(@as(u32, 0), failures.load(.monotonic));
    try testing.expectEqual(Status.green, c.status());
    try testing.expectEqual(@as(usize, 2 * n), c.indexed_count);
    // Nothing holds the old graph now; the next rebuild's reclaim frees it.
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(@as(usize, 0), c.retired_graphs.items.len);
}

test "rebuilds do not accumulate retired graphs" {
    // Before reclamation existed this list grew by one graph per rebuild, and
    // at the headline capacity each one is ~181 MB.
    var c = try makeCollection(8, .dot, 4096);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    for (0..500) |i| _ = try c.upsert(.{ .num = i }, &v);

    for (0..5) |_| try buildIndex(&c, .serial, 1);

    // Every rebuild retires its predecessor and then reclaims it, because no
    // search is in flight in this test.
    try testing.expectEqual(@as(usize, 0), c.retired_graphs.items.len);
}

test "a graph in use by a search is not reclaimed" {
    var c = try makeCollection(8, .dot, 4096);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    for (0..200) |i| _ = try c.upsert(.{ .num = i }, &v);
    try buildIndex(&c, .serial, 1);

    // Stand in for a search that is inside `search` right now, holding the
    // pointer a rebuild is about to replace.
    const guard = SearchGuard.begin(&c);

    try buildIndex(&c, .serial, 1);
    // The reclaim ran and correctly declined: the old graph is still reachable
    // from the reader that has not returned.
    try testing.expectEqual(@as(usize, 1), c.retired_graphs.items.len);
    try testing.expectEqual(@as(usize, 0), reclaimRetired(&c));

    guard.end();
    // Once the reader is out, the same call frees it.
    try testing.expectEqual(@as(usize, 1), reclaimRetired(&c));
    try testing.expectEqual(@as(usize, 0), c.retired_graphs.items.len);
}

test "a quantized store is reclaimed on the same rule as the graph" {
    var c = try makeCollection(8, .dot, 1024);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x5107);
    const rnd = prng.random();
    for (0..300) |i| {
        var v: [8]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    c.quant_mode = .scalar;
    try buildIndex(&c, .serial, 1);
    try buildIndex(&c, .serial, 1);

    // `buildIndex` re-quantizes on every build, so without reclamation this
    // grew a store per rebuild as well as a graph per rebuild.
    try testing.expectEqual(@as(usize, 0), c.retired_quant.items.len);
    try testing.expect(c.quant.load(.acquire) != null);
}

test "a search never sees half of one vector and half of another" {
    // W11's actual shape, which is not what its documentation said: bfb
    // assigns ids from `--offset` (0 by default), so uploading into a
    // populated collection *overwrites* rows rather than appending. An
    // overwrite is the one write that lands on a row a search may be reading,
    // and `upsert` takes the write lock while `search` deliberately does not.
    const dim = 64;
    var c = try makeCollection(dim, .dot, 16);
    defer c.deinit();

    var ones: [dim]f32 = @splat(1.0);
    var twos: [dim]f32 = @splat(2.0);
    _ = try c.upsert(.{ .num = 1 }, &ones);

    const Writer = struct {
        fn run(coll: *Collection, a: *[dim]f32, b: *[dim]f32, stop: *std.atomic.Value(bool)) void {
            var i: usize = 0;
            while (!stop.load(.acquire)) : (i += 1) {
                // Same id every time, so every write is an in-place overwrite.
                _ = coll.upsert(.{ .num = 1 }, if (i % 2 == 0) a else b) catch return;
            }
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    const t = try std.Thread.spawn(.{}, Writer.run, .{ &c, &ones, &twos, &stop });

    // A dot product against an all-ones query is `Σ row`, so a consistent read
    // is dim*1 or dim*2 and nothing else. A torn read lands in between, and
    // without the version guard it does: the vector spans eight cache lines at
    // this dimension, so there is plenty of room to catch a write mid-flight.
    var q: [dim]f32 = @splat(1.0);
    var torn: usize = 0;
    var reads: usize = 0;
    while (reads < 20_000) : (reads += 1) {
        var buf: [4]Candidate = undefined;
        var out = heap.TopK.init(&buf, 1);
        bruteForce(&c, &q, &out);
        const got = out.finish();
        if (got.len == 0) continue;
        const s = got[0].score;
        if (s != dim * 1.0 and s != dim * 2.0) torn += 1;
    }
    stop.store(true, .release);
    t.join();

    try testing.expectEqual(@as(usize, 0), torn);
}

test "an appended point is never visible before its vector is" {
    // `count()` is a reader's only bound, so publishing an offset before its
    // row is written let a search score a zeroed row and a build wire a zero
    // vector into the graph. Reserve, write, then publish.
    const dim = 32;
    var c = try makeCollection(dim, .dot, 4096);
    defer c.deinit();

    const Appender = struct {
        fn run(coll: *Collection, stop: *std.atomic.Value(bool)) void {
            var v: [dim]f32 = @splat(3.0);
            var i: usize = 0;
            while (!stop.load(.acquire) and i < 4000) : (i += 1) {
                _ = coll.upsert(.{ .num = i }, &v) catch return;
            }
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    const t = try std.Thread.spawn(.{}, Appender.run, .{ &c, &stop });

    // Every published row holds 3.0 in every component, so `Σ row` against an
    // all-ones query is exactly 3*dim. A zero row would score 0.
    var q: [dim]f32 = @splat(1.0);
    var zeros: usize = 0;
    var reads: usize = 0;
    while (reads < 20_000) : (reads += 1) {
        var buf: [4]Candidate = undefined;
        var out = heap.TopK.init(&buf, 1);
        bruteForce(&c, &q, &out);
        const got = out.finish();
        if (got.len > 0 and got[0].score != 3.0 * dim) zeros += 1;
    }
    stop.store(true, .release);
    t.join();

    try testing.expectEqual(@as(usize, 0), zeros);
}

test "uint8 cosine: prepareQuery leaves the query raw, so scores are Qdrant's" {
    // `prepareQuery` asked the metric alone and normalised, and `Probe.init`
    // then truncated the unit query to bytes: every component but the largest
    // became 0 and every score was 0. The rule is the datatype's.
    var c = try makeCollectionDt(4, .cosine, .uint8, 8);
    defer c.deinit();

    var a = [_]f32{ 3, 4, 0, 0 };
    var b = [_]f32{ 0, 0, 5, 12 };
    const off_a = try c.upsert(.{ .num = 1 }, &a);
    const off_b = try c.upsert(.{ .num = 2 }, &b);

    var scratch: [4]f32 = undefined;
    const q = try c.prepareQuery(&scratch, &[_]f32{ 6, 8, 0, 0 });
    // Untouched: the same slice back, not a normalised copy.
    try testing.expectEqual(@as(f32, 6), q[0]);

    var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
    const probe = Probe.init(&c, q, &buf);
    // Qdrant's `cosine_similarity_bytes`: dot / sqrt(Σq² Σr²), by hand.
    // q=(6,8,0,0), a=(3,4,0,0): 50 / sqrt(100*25) = 1.0.
    try testing.expectApproxEqAbs(@as(f32, 1.0), probe.scoreNode(off_a), 1e-6);
    // b=(0,0,5,12): 0 / ... = 0.
    try testing.expectEqual(@as(f32, 0.0), probe.scoreNode(off_b));

    // And through the search entry point, the parallel vector wins.
    var top_storage: [2]Candidate = undefined;
    var top = heap.TopK.init(&top_storage, 2);
    search(&c, q, 16, .approximate, null, &top);
    const got = top.finish();
    try testing.expectEqual(off_a, got[0].id);
    try testing.expectApproxEqAbs(@as(f32, 1.0), got[0].score, 1e-6);
}

test "an in-place overwrite counts toward the rebuild and is searched at its new position" {
    // bfb's default upload into a populated collection is all overwrites
    // (ids from `--offset` 0), and `needsRebuild` counted only the appended
    // tail, `total - graph.count`, which an overwrite leaves at zero. So a
    // collection could have every point moved and never rebuild.
    const dim = 8;
    var c = try makeCollection(dim, .euclid, 1024);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x0ae2);
    const rnd = prng.random();
    const n = 500;
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(Status.green, c.status());
    try testing.expectEqual(@as(usize, 0), c.overwritten_since_build.load(.acquire));

    // Move one point far away from everything.
    var far: [dim]f32 = @splat(50.0);
    const moved = try c.upsert(.{ .num = 7 }, &far);
    try testing.expectEqual(@as(usize, 1), c.overwritten_since_build.load(.acquire));
    // One point in 500 is under the ratio: still Green, still served from
    // the graph, and correctly, because the scorer reads the arena.
    try testing.expect(!needsRebuild(&c));
    try testing.expectEqual(Status.green, c.status());

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
    defer scratch.deinit(testing.allocator);

    // Near the new position: exact and approximate both put the moved point
    // first. The graph reaches it through its old edges, ranks it by its new
    // vector.
    {
        var eb: [5]Candidate = undefined;
        var exact = heap.TopK.init(&eb, 5);
        search(&c, &far, 64, .exact, null, &exact);
        var gb: [5]Candidate = undefined;
        var approx = heap.TopK.init(&gb, 5);
        search(&c, &far, 64, .approximate, &scratch, &approx);
        try testing.expectEqual(moved, exact.finish()[0].id);
        try testing.expectEqual(moved, approx.finish()[0].id);
    }

    // Now overwrite enough points to cross the ratio: `needsRebuild` flips
    // and `invalidateIndex` steps aside, exactly as for an appended tail.
    const enough = @as(usize, @intFromFloat(@as(f64, n) * rebuild_ratio)) + 1;
    for (0..enough) |i| {
        var v: [dim]f32 = @splat(@floatFromInt(100 + i));
        _ = try c.upsert(.{ .num = 100 + i }, &v);
    }
    try testing.expect(needsRebuild(&c));
    try testing.expectEqual(Status.yellow, c.status());
    invalidateIndex(&c);
    try testing.expectEqual(IndexState.absent, c.index_state.load(.acquire));

    // After the rebuild the counter is reset, the collection is Green again,
    // and exact and approximate agree near the moved point and near where it
    // used to be (nothing is there any more, so the two must agree that the
    // moved point is *not* the answer).
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(@as(usize, 0), c.overwritten_since_build.load(.acquire));
    try testing.expectEqual(Status.green, c.status());
    var scratch2 = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
    defer scratch2.deinit(testing.allocator);
    {
        var eb: [5]Candidate = undefined;
        var exact = heap.TopK.init(&eb, 5);
        search(&c, &far, 64, .exact, null, &exact);
        var gb: [5]Candidate = undefined;
        var approx = heap.TopK.init(&gb, 5);
        search(&c, &far, 64, .approximate, &scratch2, &approx);
        try testing.expectEqual(moved, exact.finish()[0].id);
        try testing.expectEqual(moved, approx.finish()[0].id);
    }
}

test "an overwrite re-encodes the quantized row in place" {
    // Stage 1 of the quantized path scores on codes. A moved point whose
    // codes still described its old vector would be admitted to the rescore
    // set on the strength of a position it no longer holds, so a query at
    // its new position could miss it entirely.
    for ([_]quant_mod.Mode{ .scalar, .binary, .{ .product = .x4 } }) |mode| {
        const dim = 16;
        var c = try makeCollection(dim, .dot, 1024);
        defer c.deinit();
        var prng = std.Random.DefaultPrng.init(0x0ec0de);
        const rnd = prng.random();
        for (0..400) |i| {
            var v: [dim]f32 = undefined;
            for (&v) |*x| x.* = rnd.floatNorm(f32) * 0.1;
            _ = try c.upsert(.{ .num = i }, &v);
        }
        c.quant_mode = mode;
        try buildIndex(&c, .serial, 1);
        const store = c.quant.load(.acquire).?;

        // Move point 3 to a corner and query there: it must come back first
        // through the quantized path, which it cannot unless its codes moved
        // with it. Compare its stage-1 score against a stale copy taken
        // before the move.
        var scratch = try quantized.QueryScratch.init(testing.allocator, dim, 64);
        defer scratch.deinit(testing.allocator);
        var target: [dim]f32 = @splat(1.0);
        const q = try quantized.prepareQuery(store, .dot, &target, &scratch, testing.allocator);
        const before = quantized.Query.score(@ptrCast(&q), 3);
        const moved = try c.upsert(.{ .num = 3 }, &target);
        try testing.expectEqual(@as(u32, 3), moved);
        const after = quantized.Query.score(@ptrCast(&q), 3);
        try testing.expect(after > before);
        try testing.expectEqual(@as(usize, 1), c.overwritten_since_build.load(.acquire));
    }
}

test "the quantized store is published as one pointer and retired, never freed, under a reader" {
    var c = try makeCollection(8, .dot, 512);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x9a1);
    const rnd = prng.random();
    for (0..200) |i| {
        var v: [8]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try testing.expect(c.quant.load(.acquire) == null);
    try quantize(&c, .scalar);
    const first = c.quant.load(.acquire).?;
    try testing.expect(first.* == .scalar);

    // A reader that loaded `first` and is still scoring against it.
    const guard = SearchGuard.begin(&c);
    try quantize(&c, .binary);
    const second = c.quant.load(.acquire).?;
    try testing.expect(second != first);
    try testing.expect(second.* == .binary);
    // The reader's store is intact and reachable, on the retired list.
    try testing.expectEqual(@as(usize, 1), c.retired_quant.items.len);
    try testing.expectEqual(first, c.retired_quant.items[0]);
    try testing.expect(first.* == .scalar);
    try testing.expectEqual(@as(usize, 0), reclaimRetired(&c));
    guard.end();
    try testing.expectEqual(@as(usize, 1), reclaimRetired(&c));
}

test "an overwrite in the builder's read-to-publish window lands in the published codes and counter" {
    // `buildIndex` reads the arena, quantizes, then publishes. A row
    // overwritten between the read and the publish used to be re-encoded
    // into the store about to be *retired*, and the pending counter was then
    // reset to zero, so the published store described the old vector and
    // nothing said the graph was out of date. `build_hook` runs exactly in
    // that window, on the build thread, so the race is deterministic.
    const dim = 16;
    const n = 400;
    const H = struct {
        var far: [dim]f32 = @splat(3.0);
        var moved: usize = 0;
        fn hook(c: *Collection) void {
            // Overwrite `moved` rows the builder has already read.
            for (0..moved) |i| _ = c.upsert(.{ .num = i }, &far) catch unreachable;
        }
    };
    for ([_]usize{ 1, @as(usize, @intFromFloat(@as(f64, n) * rebuild_ratio)) + 1 }) |moved| {
        var c = try makeCollection(dim, .dot, 1024);
        defer c.deinit();
        var prng = std.Random.DefaultPrng.init(0x0b5e);
        const rnd = prng.random();
        for (0..n) |i| {
            var v: [dim]f32 = undefined;
            for (&v) |*x| x.* = rnd.floatNorm(f32) * 0.1;
            _ = try c.upsert(.{ .num = i }, &v);
        }
        c.quant_mode = .scalar;
        try buildIndex(&c, .serial, 1);
        try testing.expectEqual(@as(usize, 0), c.overwritten_since_build.load(.acquire));

        H.moved = moved;
        c.build_hook = H.hook;
        try buildIndex(&c, .serial, 1);
        c.build_hook = null;

        // The published store's codes for the moved rows are the codes of
        // the *new* vector.
        const store = c.quant.load(.acquire).?;
        var want: [dim]u8 = undefined;
        quant_mod.scalar.encode(store.scalar.params, &want, &H.far);
        for (0..moved) |i| try testing.expectEqualSlices(u8, &want, store.scalar.row(@intCast(i)));
        // And the counter says so, so `needsRebuild` and `status` see it.
        try testing.expectEqual(moved, c.overwritten_since_build.load(.acquire));
        try testing.expectEqual(moved > 1, needsRebuild(&c));
        try testing.expectEqual(if (moved > 1) Status.yellow else Status.green, c.status());
        try testing.expectEqual(@as(usize, 0), c.overwrite_log.count());
        try testing.expect(!c.overwrite_log_active);

        // A build with nothing in the window publishes clean.
        try buildIndex(&c, .serial, 1);
        try testing.expectEqual(@as(usize, 0), c.overwritten_since_build.load(.acquire));
        try testing.expectEqual(Status.green, c.status());
    }
}

test "a build that cannot quantize publishes neither the graph nor a store over the old one" {
    // `quantize` inside `buildIndex` was `catch {}`: on OOM the graph went
    // out over the previous build's codes, which are zero for every row
    // appended since, and the collection kept answering as quantized. Sweep
    // the failure point over every allocation of a rebuild: whichever one
    // fails, the collection must still serve exactly the graph and store it
    // served before, and leak nothing (`testing.allocator` underneath).
    const dim = 8;
    var fa = std.testing.FailingAllocator.init(testing.allocator, .{});
    var c = try Collection.init(fa.allocator(), "oom", .{ .dim = dim, .metric = .dot, .capacity = 512 });
    defer {
        fa.fail_index = std.math.maxInt(usize);
        c.deinit();
    }
    c.quant_mode = .scalar;
    var prng = std.Random.DefaultPrng.init(0x00f);
    const rnd = prng.random();
    for (0..100) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    const g0 = c.graph.?;
    const s0 = c.quant.load(.acquire).?;
    // The one allocation a build is allowed to fail without reporting it:
    // retiring the replaced graph keeps it alive rather than freeing it
    // (bounded, by design). Reserve it so the sweep's leak check is exact.
    try c.retired_graphs.ensureUnusedCapacity(c.alloc, 1);

    // Append a tail so the new store's codes would differ from the old.
    for (100..140) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    var k: usize = 0;
    var failed_at_least_once = false;
    while (true) : (k += 1) {
        fa.alloc_index = 0;
        fa.fail_index = k;
        fa.has_induced_failure = false;
        if (buildIndex(&c, .serial, 1)) |_| {
            // Success only once no allocation of the build was denied.
            try testing.expect(!fa.has_induced_failure);
            try testing.expect(c.graph.? != g0);
            try testing.expect(c.quant.load(.acquire).? != s0);
            break;
        } else |e| {
            failed_at_least_once = true;
            try testing.expectEqual(error.OutOfMemory, e);
            try testing.expectEqual(g0, c.graph.?);
            try testing.expectEqual(s0, c.quant.load(.acquire).?);
            try testing.expect(!c.overwrite_log_active);
        }
    }
    try testing.expect(failed_at_least_once);
    try testing.expect(k > 3);
}

test "a build outrun by ingest steps aside on its own, per point and not only per batch" {
    // The API layer's per-batch `invalidateIndex` can run while the build is
    // still `.building`, in which case it records nothing, and no further
    // write need ever arrive. So the build itself must make the rebuild
    // decision after publishing, under the write lock the writes took, or a
    // tail past the ratio sits `.ready`, Yellow, with `ensureIndexBuilding`
    // early-outing on `.ready` and no build ever starting again.
    var c = try makeCollection(4, .dot, 1024);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    const n = 200;
    for (0..n) |i| _ = try c.upsert(.{ .num = i }, &v);

    const H = struct {
        var far: [4]f32 = .{ 0, 1, 0, 0 };
        fn many(coll: *Collection) void {
            // Past `rebuild_ratio`, all during the build.
            const extra = @as(usize, @intFromFloat(@as(f64, n) * rebuild_ratio)) + 1;
            for (0..extra) |i| _ = coll.upsert(.{ .num = n + i }, &far) catch unreachable;
        }
    };
    c.build_hook = H.many;
    c.index_state.store(.building, .release);
    try buildIndex(&c, .serial, 1);
    c.build_hook = null;

    // The build published (the graph is there and counted) and then stepped
    // aside for the rebuild the tail warrants, without an `invalidateIndex`
    // from anyone else.
    try testing.expect(c.graph != null);
    try testing.expectEqual(@as(usize, n), c.graph_count.load(.acquire));
    try testing.expect(needsRebuild(&c));
    try testing.expectEqual(IndexState.absent, c.index_state.load(.acquire));
    try testing.expectEqual(Status.yellow, c.status());

    // So the next poll wins `.absent -> .building` and that build covers
    // every point.
    try testing.expectEqual(@as(?IndexState, null), c.index_state.cmpxchgStrong(.absent, .building, .acq_rel, .monotonic));
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(IndexState.ready, c.index_state.load(.acquire));
    try testing.expectEqual(c.count(), c.indexed_count);
    try testing.expectEqual(Status.green, c.status());
}

test "tombstones among the nearest do not shorten the page when ef equals limit" {
    // Filtering the `ef` results after the traversal returned `ef - d` when
    // `d` of them were deleted, so with `ef == limit` a client asking for 20
    // got 20 minus however many of the true nearest had been deleted, while
    // plenty of live points existed. Tombstones are now skipped as the result
    // heap is filled, so it holds `ef` live points.
    const dim = 8;
    var c = try makeCollection(dim, .euclid, 1024);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0xdead);
    const rnd = prng.random();
    const n = 500;
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
    defer scratch.deinit(testing.allocator);

    const k = 20;
    for (0..10) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);
        // Delete the 10 nearest, found exactly.
        var tb: [10]Candidate = undefined;
        var truth = heap.TopK.init(&tb, 10);
        bruteForce(&c, &q, &truth);
        for (truth.finish()) |t| _ = c.delete(c.id_space.external(t.id));

        var gb: [k]Candidate = undefined;
        var got = heap.TopK.init(&gb, k);
        // `ef == limit`, the case where a post-hoc filter is guaranteed short.
        search(&c, &q, k, .approximate, &scratch, &got);
        try testing.expectEqual(@as(usize, k), got.len);
        for (got.items[0..got.len]) |cand| try testing.expect(!c.deleted.isSet(cand.id));
    }
}

test "a gathered scan returns exactly what the same queries return one at a time" {
    // The property the gather rests on: reading a row once for K queries must
    // not change any of their answers. Randomised rather than crafted, because
    // the failure mode is an indexing slip that shows on one arrival order.
    const dim = 16;
    var c = try makeCollection(dim, .euclid, 2048);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x9A7E);
    const rnd = prng.random();
    const n = 700;
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    // A tombstone among the data, since the gather skips them once per row and
    // the single-query scan skips them per query: the two must still agree.
    _ = c.delete(.{ .num = 3 });
    _ = c.delete(.{ .num = 404 });

    const stride = probeScratchStride(&c);
    const scratch = try testing.allocator.alignedAlloc(u8, .fromByteUnits(Probe.buffer_align), max_gather * stride);
    defer testing.allocator.free(scratch);

    const k = 10;
    for (1..max_gather + 1) |batch| {
        var qs: [max_gather][dim]f32 = undefined;
        var q_slices: [max_gather][]const f32 = undefined;
        for (0..batch) |i| {
            for (&qs[i]) |*x| x.* = rnd.floatNorm(f32);
            q_slices[i] = &qs[i];
        }

        var gathered_store: [max_gather][k]Candidate = undefined;
        var gathered: [max_gather]heap.TopK = undefined;
        var gathered_ptrs: [max_gather]*heap.TopK = undefined;
        for (0..batch) |i| {
            gathered[i] = heap.TopK.init(&gathered_store[i], k);
            gathered_ptrs[i] = &gathered[i];
        }
        bruteForceRangeMulti(&c, q_slices[0..batch], 0, @intCast(c.id_space.count()), null, scratch, gathered_ptrs[0..batch]);

        for (0..batch) |i| {
            var one_store: [k]Candidate = undefined;
            var one = heap.TopK.init(&one_store, k);
            bruteForce(&c, q_slices[i], &one);
            const want = one.finish();
            const got = gathered[i].finish();
            try testing.expectEqual(want.len, got.len);
            for (want, got) |w, g| {
                try testing.expectEqual(w.id, g.id);
                try testing.expectEqual(w.score, g.score);
            }
        }
    }
}

test "a gathered scan honours a filter the same way one query at a time does" {
    const dim = 8;
    var c = try makeCollection(dim, .euclid, 1024);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x1234);
    const rnd = prng.random();
    for (0..300) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    // Admit the even offsets only, which is the shape a payload index gives.
    const Even = struct {
        fn pred(_: *const anyopaque, id: u32) bool {
            return id % 2 == 0;
        }
    };
    const sentinel: u8 = 0;
    const filter: hnsw.Index.Filter = .{ .ctx = &sentinel, .pred = Even.pred };

    const stride = probeScratchStride(&c);
    const scratch = try testing.allocator.alignedAlloc(u8, .fromByteUnits(Probe.buffer_align), 4 * stride);
    defer testing.allocator.free(scratch);

    var qs: [4][dim]f32 = undefined;
    var q_slices: [4][]const f32 = undefined;
    for (0..4) |i| {
        for (&qs[i]) |*x| x.* = rnd.floatNorm(f32);
        q_slices[i] = &qs[i];
    }
    const k = 5;
    var store: [4][k]Candidate = undefined;
    var tops: [4]heap.TopK = undefined;
    var ptrs: [4]*heap.TopK = undefined;
    for (0..4) |i| {
        tops[i] = heap.TopK.init(&store[i], k);
        ptrs[i] = &tops[i];
    }
    bruteForceRangeMulti(&c, &q_slices, 0, @intCast(c.id_space.count()), filter, scratch, &ptrs);

    for (0..4) |i| {
        var one_store: [k]Candidate = undefined;
        var one = heap.TopK.init(&one_store, k);
        bruteForceRangeFiltered(&c, q_slices[i], 0, @intCast(c.id_space.count()), filter, &one);
        for (one.finish(), tops[i].finish()) |w, g| {
            try testing.expectEqual(w.id, g.id);
            try testing.expectEqual(w.score, g.score);
        }
        // And the filter actually bit.
        for (tops[i].finish()) |g| try testing.expect(g.id % 2 == 0);
    }
}

test "a graph covering more than the reader's snapshot returns a point twice" {
    // The hazard P2 item 5 has to close, and the half the search path does not
    // already anticipate. Its comment names the *invisible* case: a node
    // "linked into the graph after this query traversed, counted before this
    // query scanned the tail, and therefore in neither". The mirror case is a
    // node in *both*: the reader's `covered` snapshot is behind, so the tail
    // scan covers `[covered, total)` while the traversal reaches the same node
    // through edges that already exist. `heap.TopK.push` does not deduplicate,
    // so the client gets one point twice.
    //
    // Today this cannot happen, because a rebuild publishes a whole new graph
    // and `count` never moves under a reader. It becomes reachable the moment
    // anything inserts into a live graph, which is what item 5 proposes. The
    // requirement it implies is that the traversal be bounded by the reader's
    // own snapshot, so every node belongs to exactly one of the two regions.
    //
    // This test constructs the window directly rather than racing for it, by
    // searching with a `covered` behind what the graph holds.
    const dim = 8;
    var c = try makeCollection(dim, .euclid, 512);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0xD0B1E);
    const rnd = prng.random();
    const n = 200;
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 512, 64);
    defer scratch.deinit(testing.allocator);

    var q: [dim]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    // The graph covers all 200. A reader whose snapshot says 100 scans
    // [100, 200) exhaustively *and* traverses a graph that holds them.
    const covered: u32 = 100;
    var store: [32]Candidate = undefined;
    var out = heap.TopK.init(&store, 32);
    {
        var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
        const probe = Probe.init(&c, &q, &buf);
        const idx = hnsw.Index{ .graph = c.graph.?, .scorer = .of(&probe) };
        idx.searchFiltered(64, &scratch, &out, null);
        scanPendingTail(&c, &q, covered, &out);
    }

    // At least one id appears twice, which is the defect this pins.
    const got = out.finish();
    var seen = std.AutoHashMap(u32, void).init(testing.allocator);
    defer seen.deinit();
    var dupes: usize = 0;
    for (got) |cand| {
        const e = try seen.getOrPut(cand.id);
        if (e.found_existing) dupes += 1;
    }
    try testing.expect(dupes > 0);

    // And the shape of the fix: bounding the traversal to the reader's own
    // snapshot puts every node in exactly one region, so nothing repeats.
    var one_region_store: [32]Candidate = undefined;
    var one_region = heap.TopK.init(&one_region_store, 32);
    {
        const Bound = struct {
            limit: u32,
            fn pred(ctx: *const anyopaque, node: u32) bool {
                const self: *const @This() = @ptrCast(@alignCast(ctx));
                return node < self.limit;
            }
        };
        const b = Bound{ .limit = covered };
        var buf: [Probe.max_buffer]u8 align(Probe.buffer_align) = undefined;
        const probe = Probe.init(&c, &q, &buf);
        const idx = hnsw.Index{ .graph = c.graph.?, .scorer = .of(&probe) };
        idx.searchFiltered(64, &scratch, &one_region, .{ .ctx = @ptrCast(&b), .pred = Bound.pred });
        scanPendingTail(&c, &q, covered, &one_region);
    }
    var seen2 = std.AutoHashMap(u32, void).init(testing.allocator);
    defer seen2.deinit();
    for (one_region.finish()) |cand| {
        const e = try seen2.getOrPut(cand.id);
        try testing.expect(!e.found_existing);
    }
}

test "live insertion keeps every appended point findable exactly once" {
    // The whole point of the flag, and the two failures it must not have: a
    // point that is in neither region (invisible) or in both (returned twice).
    // Compiled out unless `-Dlive-insert=true`, because with the flag off the
    // graph never grows under a reader and there is nothing to test.
    if (!build_options.live_insert) return error.SkipZigTest;

    const dim = 8;
    var c = try makeCollection(dim, .euclid, 4096);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x11FE);
    const rnd = prng.random();

    const built = 800;
    for (0..built) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(@as(usize, built), c.graph.?.count);

    // Append past the built graph. With the flag on these join it directly.
    const appended = 400;
    for (built..built + appended) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    // The graph grew with the collection: no pending tail is left.
    try testing.expectEqual(@as(usize, built + appended), c.graph.?.count);
    try testing.expectEqual(@as(usize, built + appended), c.count());

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 4096, 256);
    defer scratch.deinit(testing.allocator);

    var dupes: usize = 0;
    var missing: usize = 0;
    for (0..60) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);

        var store: [24]Candidate = undefined;
        var out = heap.TopK.init(&store, 24);
        search(&c, &q, 256, .approximate, &scratch, &out);

        var seen = std.AutoHashMap(u32, void).init(testing.allocator);
        defer seen.deinit();
        for (out.finish()) |cand| {
            const e = try seen.getOrPut(cand.id);
            if (e.found_existing) dupes += 1;
        }

        // Against the exact answer: an appended point must be reachable, not
        // merely present. The graph is approximate, so compare the top-1, which
        // a 256-wide search over 1200 points must not miss.
        var tb: [1]Candidate = undefined;
        var truth = heap.TopK.init(&tb, 1);
        bruteForce(&c, &q, &truth);
        const want = truth.finish()[0].id;
        var found = false;
        for (out.finish()) |cand| {
            if (cand.id == want) found = true;
        }
        if (!found) missing += 1;
    }
    try testing.expectEqual(@as(usize, 0), dupes);
    try testing.expectEqual(@as(usize, 0), missing);
}

test "live insertion leaves the collection searchable after a rebuild" {
    // The builder follows the published graph. A rebuild swaps that pointer,
    // and the next live insert must extend the new graph rather than the
    // retired one.
    if (!build_options.live_insert) return error.SkipZigTest;

    const dim = 8;
    var c = try makeCollection(dim, .euclid, 4096);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0x22FE);
    const rnd = prng.random();
    for (0..400) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);
    for (400..500) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    // Rebuild over everything, then append again.
    try buildIndex(&c, .serial, 1);
    const g_after = c.graph.?;
    for (500..560) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try testing.expectEqual(g_after, c.graph.?);
    try testing.expectEqual(@as(usize, 560), c.graph.?.count);

    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 4096, 256);
    defer scratch.deinit(testing.allocator);
    var q: [dim]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);
    var store: [10]Candidate = undefined;
    var out = heap.TopK.init(&store, 10);
    search(&c, &q, 256, .approximate, &scratch, &out);
    try testing.expect(out.finish().len == 10);
}
