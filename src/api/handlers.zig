//! §2 / §12, the RPC surface, derived from what `bfb` actually issues.
//!
//! The complete set, from §2's table:
//!
//! | gRPC path | Trigger | Required from day one |
//! |---|---|---|
//! | `/qdrant.Qdrant/HealthCheck` | every client construction | yes |
//! | `/qdrant.Collections/CollectionExists` | `--create-if-missing` | yes |
//! | `/qdrant.Collections/Delete` | collection setup | yes |
//! | `/qdrant.Collections/Create` | collection setup | yes |
//! | `/qdrant.Collections/Get` | `wait_index` polling loop | yes |
//! | `/qdrant.Points/Upsert` | upload phase | yes |
//! | `/qdrant.Points/QueryBatch` | `--search` | yes |
//! | `/qdrant.Points/Scroll` | `--scroll`, UUID pre-fetch | phase 2 |
//! | `/qdrant.Points/SetPayload` | `--set-payload` | phase 2 |
//! | `/qdrant.Points/CreateFieldIndex` | payload flags | phase 3 |
//!
//! §12: "Everything else on `qdrant.Points` and `qdrant.Collections` returns
//! `UNIMPLEMENTED` with a message naming the RPC, so an unexpected bfb flag
//! fails loudly and immediately rather than producing a subtly wrong
//! benchmark."

const std = @import("std");
const build_options = @import("build_options");

const proto = @import("../proto/proto.zig");
const net = @import("../net/net.zig");
const core = @import("../core/core.zig");
const collections = @import("collections.zig");
const index = @import("../index/index.zig");
const dist = @import("../dist/dist.zig");

const wire = proto.wire;
const msg = proto.messages;
const grpc = net.grpc;
const server = net.server;

pub const Status = grpc.Status;

/// The shared engine state a handler operates on.
///
/// A single mutex guards the collection registry. §6.3's "no allocation on the
/// query path" is about the *search* path; collection create/delete happen once
/// per benchmark run, so a lock there costs nothing measurable and removes a
/// whole class of race. Search takes a read of the pointer only, the
/// collection itself is immutable during search per §6's diagram ("Storage:
/// ... immutable during search").
pub const Engine = struct {
    alloc: std.mem.Allocator,
    mutex: server.Mutex = .{},
    collections: std.ArrayList(*core.Collection) = .empty,

    /// §2: "several bfb code paths and human eyeballs read them."
    default_capacity: usize = 1_100_000,
    /// §5.5, 2 MiB pages behind the vector arena. Off only for A/B runs.
    huge_pages: bool = true,
    /// Where a `cached` or `cold` collection's arena file goes. `null` means
    /// the server was started without `--data-dir`, and a request for either
    /// placement is refused naming the flag rather than served from pinned
    /// memory.
    data_dir: ?[]const u8 = null,
    /// The placement for a collection that does not ask for one.
    ///
    /// `pinned` because §5.5 is "everything loaded in memory by default" and
    /// every result row on disk was measured that way. Qdrant's default for
    /// dense vectors is `Cached`, so an unconfigured bfb run puts the two
    /// engines in different placements — that is the §2 semantic gap, and
    /// `--default-placement` exists so the gap can be closed for a run rather
    /// than only described.
    default_placement: core.storage.Placement = .pinned,
    /// Threads to use for the bulk build. §6.5: "Bulk, post-ingest, all cores."
    build_threads: usize = 8,
    /// The cpus a bulk build may use, when `--pin` named a set. Null means
    /// "whatever this process already has", which is right for an unpinned
    /// run and wrong for a pinned one: the build is spawned from a worker
    /// pinned to a single core and would otherwise inherit that pin for its
    /// entire pool, running §6.5's "all cores" on exactly one.
    build_cpus: ?[]const usize = null,
    /// §8.7's option, as asked for by `--build-mode`.
    ///
    /// This said "conformance runs use the deterministic serial build", which
    /// they do not and never did: nothing outside a unit test assigned this
    /// field, so every run — conformance included — built in `parallel`.
    /// `spec.md` §8.7 states the true position ("(c) exists as `buildSerial`
    /// (tested bit-reproducible) but is not selectable at runtime, so
    /// conformance runs against the parallel build"), and the flag is what
    /// makes the second half of that sentence stop being true.
    build_mode: core.collection.BuildMode = .parallel,
    /// Largest vector dimension any worker's scratch can hold.
    ///
    /// §6.3 forbids allocating on the query path, so `Workspace.query` is sized
    /// once at startup. A collection created with a larger dimension would make
    /// every upsert and query write past it, a client-triggerable
    /// out-of-bounds from a well-formed `CreateCollection`. The bound is
    /// enforced where the collection is created, not where the overflow would
    /// happen, so the failure names its cause.
    max_dim: usize = 4096,

    pub fn init(alloc: std.mem.Allocator) Engine {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *Engine) void {
        for (self.collections.items) |c| {
            c.deinit();
            self.alloc.destroy(c);
        }
        self.collections.deinit(self.alloc);
    }

    /// Look up by name, without taking a reference.
    ///
    /// For existence checks and for tests, which have no concurrent `Delete`.
    /// A handler that *dereferences* the result must use `acquire` instead:
    /// `Delete` runs on a worker like every other RPC, so a drop can free the
    /// collection while another worker is reading it.
    pub fn find(self: *Engine, name: []const u8) ?*core.Collection {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.findLocked(name);
    }

    /// A collection, held open for as long as the handle lives.
    pub const Handle = struct {
        coll: *core.Collection,

        pub fn release(self: Handle) void {
            _ = self.coll.users.fetchSub(1, .release);
        }
    };

    /// Look up by name and hold it open. The caller must `release`.
    ///
    /// The reference is taken under the registry lock, so it cannot race a
    /// `drop` that has already unlinked the collection: either this finds it
    /// and the drop waits, or the drop unlinked it first and this returns
    /// null.
    pub fn acquire(self: *Engine, name: []const u8) ?Handle {
        self.mutex.lock();
        defer self.mutex.unlock();
        const c = self.findLocked(name) orelse return null;
        _ = c.users.fetchAdd(1, .acquire);
        return .{ .coll = c };
    }

    fn findLocked(self: *Engine, name: []const u8) ?*core.Collection {
        for (self.collections.items) |c| {
            if (std.mem.eql(u8, c.name, name)) return c;
        }
        return null;
    }

    pub fn create(self: *Engine, name: []const u8, config: core.Config) !*core.Collection {
        return (try self.createOrFind(name, config, .{})).coll;
    }

    pub const Created = struct {
        coll: *core.Collection,
        /// False when the name was already taken and `coll` is that
        /// collection, untouched.
        created: bool,
    };

    /// Create under the registry lock, or report the existing one, in one
    /// step. `createCollection` needs the distinction: a check-then-create
    /// with two concurrent creates of the same name let both pass the check,
    /// one create the collection, and the other be handed the same pointer
    /// and write its own quantization settings over it while answering
    /// `result: true`.
    pub fn createOrFind(self: *Engine, name: []const u8, config: core.Config, quant: collections.QuantSpec) !Created {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.findLocked(name)) |existing| return .{ .coll = existing, .created = false };

        const c = try self.alloc.create(core.Collection);
        errdefer self.alloc.destroy(c);
        c.* = try core.Collection.init(self.alloc, name, config);
        errdefer c.deinit();
        // Set before the append publishes the pointer. They used to be written
        // by the caller afterwards, without a handle: a `Delete` of the same
        // name in that window freed the collection under the write, and a
        // poll that reached `ensureIndexBuilding` first built an unquantized
        // store for a collection whose info then reported `scalar`.
        c.quant_mode = quant.mode;
        c.quant_quantile = quant.quantile;
        try self.collections.append(self.alloc, c);
        return .{ .coll = c, .created = true };
    }

    /// Start a background build if the collection needs one and none is running.
    ///
    /// The compare-exchange is what makes this safe to call from every poll:
    /// only the thread that wins the transition spawns, so a client polling
    /// once a second does not spawn a build per second.
    pub fn ensureIndexBuilding(self: *Engine, coll: *core.Collection) void {
        // Cheap early out for the steady state (`.ready` or `.building`), so
        // a poll per second or a query per microsecond does not contend on
        // the lock below.
        if (coll.index_state.load(.acquire) != .absent) return;

        // The lock covers the compare-exchange *and* the handle bookkeeping,
        // so two winners of successive exchanges cannot interleave their
        // join/store (see `Collection.build_thread_lock`). Nothing inside
        // takes another lock: the join is of a thread that has already
        // flipped the state and is exiting, and the spawn does not block.
        coll.build_thread_lock.lock();
        defer coll.build_thread_lock.unlock();
        if (coll.index_state.cmpxchgStrong(.absent, .building, .acq_rel, .monotonic) != null) return;

        const Task = struct {
            fn run(c: *core.Collection, mode: core.collection.BuildMode, threads: usize, cpus: ?[]const usize) void {
                // Widen to the server's whole cpu set before spawning the
                // build pool. This thread was spawned by a worker pinned to
                // one core, and `std.Thread.spawn` children inherit the
                // parent's affinity, so without this every build thread lands
                // on that one core — §6.3 pins a worker to a core, it does not
                // pin the *engine* to one.
                if (cpus) |set| net.server.pinToCpus(set);
                core.collection.buildIndex(c, mode, threads) catch {
                    // A failed build must not leave the collection wedged in
                    // `building` forever; returning it to `absent` lets the
                    // next poll retry. `status()` is derived, so the collection
                    // reports Yellow automatically for as long as it is not
                    // actually indexed.
                    c.index_state.store(.absent, .release);
                };
            }
        };
        // Reap the previous build before starting another. Winning the
        // cmpxchg above means the prior build reached `ready` or `absent`, so
        // its thread has already exited and the join is immediate.
        if (coll.build_thread) |prev| {
            prev.join();
            coll.build_thread = null;
        }
        const t = std.Thread.spawn(.{}, Task.run, .{ coll, self.build_mode, self.build_threads, self.build_cpus }) catch {
            coll.index_state.store(.absent, .release);
            return;
        };
        coll.build_thread = t;
    }

    pub fn drop(self: *Engine, name: []const u8) bool {
        // Unlink first, under the lock, so no new request can acquire it.
        self.mutex.lock();
        var doomed: ?*core.Collection = null;
        for (self.collections.items, 0..) |c, i| {
            if (std.mem.eql(u8, c.name, name)) {
                doomed = c;
                _ = self.collections.orderedRemove(i);
                break;
            }
        }
        self.mutex.unlock();

        const c = doomed orelse return false;
        // Then wait out the requests that acquired it before the unlink. The
        // wait is bounded because no new ones can arrive, and it happens
        // outside the registry lock so an in-flight search does not block
        // every other collection's lookups while it finishes.
        //
        // Spin briefly, then sleep. A search drains in microseconds and the
        // spin catches that without a syscall — but a *bulk build* is also a
        // user, and it holds the collection for as long as the build takes:
        // 250-348 s for 990,000 rows at d=1536. `Thread.yield` in a tight loop
        // is a spin, not a wait, so dropping a collection whose build was
        // still running burned a whole core for minutes and took it from the
        // eight threads doing the build. Measured at 568% CPU on a 200,000-row
        // collection that had not finished indexing.
        var spins: usize = 0;
        while (c.users.load(.acquire) != 0) {
            if (spins < 1000) {
                spins += 1;
                std.Thread.yield() catch {};
            } else {
                // `std.Thread.sleep` moved under the `Io` interface in Zig
                // 0.16 and there is none here; `server.zig` waits the same way.
                var ts = std.os.linux.timespec{ .sec = 0, .nsec = std.time.ns_per_ms };
                _ = std.os.linux.nanosleep(&ts, null);
            }
        }
        c.deinit();
        self.alloc.destroy(c);
        return true;
    }
};

/// Per-worker scratch, so the query path allocates nothing (§6.3).
pub const Workspace = struct {
    /// Preprocessed query vector (cosine normalisation target).
    query: []f32,
    /// Result heap storage, sized to the largest `limit + offset` we accept.
    results: []index.Candidate,
    /// One `Scroll` page. Preallocated per §6.3 like everything else on the
    /// request path, and it bounds the client's `limit`.
    scroll_ids: []core.ExternalId,
    /// §6.3: "Per-worker preallocated: visited set, candidate heap, result
    /// heap, rescore buffer, distance scratch." Owned per worker so no two
    /// concurrent queries share a visited set.
    hnsw: ?*index.hnsw.Index.Scratch = null,
    hnsw_owned: ?index.hnsw.Index.Scratch = null,
    /// §6.7's two-stage search scratch: the encoded query and the candidate
    /// and rescore buffers.
    qscratch: ?core.quantized.QueryScratch = null,
    /// What the scratch above is currently sized for, so a larger collection
    /// grows it instead of overrunning it.
    hnsw_capacity: usize = 0,
    quant_dim: usize = 0,
    /// One bit per point: the set a payload filter admits, built per query
    /// by `payload.Store.select` from the index (§6.3: preallocated, sized
    /// with the HNSW scratch to the largest collection this worker served).
    filter_bits: []u64 = &.{},
    /// `ensureGather`'s buffers, and what they are currently sized for.
    gather_queries: []f32 = &.{},
    gather_results: []index.Candidate = &.{},
    gather_probe: []u8 = &.{},
    gather_dim: usize = 0,
    gather_stride: usize = 0,

    pub const max_limit = 4096;
    pub const max_ef = 4096;

    pub fn init(alloc: std.mem.Allocator, max_dim: usize) !Workspace {
        const query = try alloc.alloc(f32, max_dim);
        errdefer alloc.free(query);
        const results = try alloc.alloc(index.Candidate, max_limit);
        errdefer alloc.free(results);
        const scroll_ids = try alloc.alloc(core.ExternalId, max_limit);
        return .{ .query = query, .results = results, .scroll_ids = scroll_ids };
    }

    /// Allocate the HNSW scratch once the collection capacity is known.
    ///
    /// Deferred because the visited set is sized to the point count, and that
    /// is a per-collection number the server does not have at worker startup.
    /// Sized to the *largest* collection this worker has served, not the first.
    ///
    /// An early version returned as soon as the scratch existed, which latched
    /// it to whichever collection the worker happened to see first. A worker
    /// that served a 64-dim collection and then a 1536-dim one would slice a
    /// visited set and a code buffer sized for the smaller, a bounds panic in
    /// safe builds and a heap overwrite in ReleaseFast.
    pub fn ensureHnsw(self: *Workspace, alloc: std.mem.Allocator, capacity: usize) !void {
        if (self.hnsw_capacity >= capacity and self.hnsw_owned != null) return;
        const want_cap = @max(capacity, self.hnsw_capacity);
        var fresh = try index.hnsw.Index.Scratch.init(alloc, want_cap, max_ef);
        errdefer fresh.deinit(alloc);
        const bits = try alloc.alloc(u64, (want_cap + 63) / 64);
        errdefer alloc.free(bits);
        if (self.hnsw_owned) |*old| old.deinit(alloc);
        if (self.filter_bits.len > 0) alloc.free(self.filter_bits);
        self.filter_bits = bits;
        self.hnsw_owned = fresh;
        self.hnsw = &self.hnsw_owned.?;
        self.hnsw_capacity = want_cap;
    }

    /// Scratch for a gathered exact batch: K query vectors, K probe buffers and
    /// K result heaps, so one pass of the arena can answer K queries.
    ///
    /// Lazily sized like `ensureQuant` and for the same reason: a worker that
    /// never serves an all-exact batch never pays for it, and §6.3's
    /// no-allocation rule is about the *query* path, which this is not on once
    /// it has been sized. Freed and regrown only when a larger collection
    /// arrives, so the steady state is one allocation per worker.
    pub fn ensureGather(self: *Workspace, alloc: std.mem.Allocator, coll: *const core.Collection) !void {
        const stride = core.collection.probeScratchStride(coll);
        const k = core.collection.max_gather;
        if (self.gather_dim >= coll.config.dim and self.gather_stride >= stride) return;

        const queries = try alloc.alloc(f32, k * coll.config.dim);
        errdefer alloc.free(queries);
        const results = try alloc.alloc(index.Candidate, k * max_limit);
        errdefer alloc.free(results);
        const probe = try alloc.alignedAlloc(u8, .fromByteUnits(core.collection.Probe.buffer_align), k * stride);

        if (self.gather_queries.len > 0) alloc.free(self.gather_queries);
        if (self.gather_results.len > 0) alloc.free(self.gather_results);
        if (self.gather_probe.len > 0) alloc.free(self.gather_probe);
        self.gather_queries = queries;
        self.gather_results = results;
        self.gather_probe = probe;
        self.gather_dim = coll.config.dim;
        self.gather_stride = stride;
    }

    pub fn ensureQuant(self: *Workspace, alloc: std.mem.Allocator, dim: usize) !void {
        if (self.quant_dim >= dim and self.qscratch != null) return;
        const want = @max(dim, self.quant_dim);
        var fresh = try core.quantized.QueryScratch.init(alloc, want, max_limit);
        errdefer fresh.deinit(alloc);
        if (self.qscratch) |*old| old.deinit(alloc);
        self.qscratch = fresh;
        self.quant_dim = want;
    }

    /// The scratch bundle `core` asks for. `Workspace` owns these; this is the
    /// view of them the search path takes, and it exists so the fields are not
    /// unpacked into a nine-argument call.
    pub fn searchScratch(self: *Workspace, alloc: std.mem.Allocator) core.quantized_search.Scratch {
        return .{
            .hnsw = self.hnsw.?,
            .quant = &self.qscratch.?,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Workspace, alloc: std.mem.Allocator) void {
        alloc.free(self.query);
        alloc.free(self.results);
        alloc.free(self.scroll_ids);
        if (self.hnsw_owned) |*s| s.deinit(alloc);
        if (self.qscratch) |*s| s.deinit(alloc);
        if (self.filter_bits.len > 0) alloc.free(self.filter_bits);
        if (self.gather_queries.len > 0) alloc.free(self.gather_queries);
        if (self.gather_results.len > 0) alloc.free(self.gather_results);
        if (self.gather_probe.len > 0) alloc.free(self.gather_probe);
    }
};

/// Everything a handler needs. One per worker thread.
pub const Context = struct {
    engine: *Engine,
    workspace: Workspace,
};

/// Dispatch one request. Runs on a worker thread.
pub fn handle(ctx_ptr: *anyopaque, req: *const server.Request, out: *server.ResponseBuf) server.Completion {
    const ctx: *Context = @ptrCast(@alignCast(ctx_ptr));

    // One query of a fanned-out batch, on this worker's scratch. The
    // completion is the job's to make (`BatchJob`); this call returns
    // detached and writes nothing to `out`.
    if (req.job != null) {
        runBatchSub(ctx, req);
        return .{ .conn = req.conn, .stream_idx = req.stream_idx, .stream_id = req.stream_id, .status = .ok, .detached = true };
    }

    const path = grpc.Path.parse(req.path) orelse
        return err(req, .invalid_argument, "malformed :path");

    const body = grpc.decodeMessage(req.body) catch |e| return switch (e) {
        grpc.Error.CompressionUnsupported => err(req, .unimplemented, "gRPC message compression"),
        else => err(req, .invalid_argument, "malformed gRPC message frame"),
    };

    // No `else`. The enum exists so that adding a member is a compile error
    // here rather than a working endpoint that answers UNIMPLEMENTED forever,
    // and an `else` arm gives exactly that back. `disposition` below spells the
    // same groups for the same reason.
    return switch (Rpc.parse(path)) {
        .health_check => healthCheck(req, out),
        .collections_create => collections.createCollection(ctx, req, body, out),
        .collections_delete => collections.deleteCollection(ctx, req, body, out),
        .collections_exists => collections.collectionExists(ctx, req, body, out),
        .collections_get => collections.getCollectionInfo(ctx, req, body, out),
        .points_upsert => upsert(ctx, req, body, out),
        .points_query_batch => queryBatch(ctx, req, body, out),
        .points_scroll => scroll(ctx, req, body, out),
        .points_set_payload => setPayload(ctx, req, body, out),
        .points_create_field_index => createFieldIndex(ctx, req, body, out),

        .collections_list,
        .collections_update,
        .points_delete,
        .points_get,
        .points_search,
        .points_search_batch,
        .points_query,
        .unknown,
        => unimplementedRpc(req, path, out),
    };
}

/// The RPCs §2's compatibility table names, as one closed set.
///
/// This replaces two parallel lists of the same method strings: a dispatch
/// cascade of `std.mem.eql` and a separate `isDeferredToLaterPhase` array used
/// only to phrase the error message. Nothing tied them together, so adding an
/// RPC in one place and forgetting the other produced a working endpoint that
/// still described itself as unimplemented, or worse, a deferred RPC pointing
/// a client at §1's non-goals, which is where a *permanent* refusal lives.
///
/// A closed enum also makes the dispatch switch exhaustive, so a new member
/// cannot be added without the compiler asking what to do with it.
pub const Rpc = enum {
    health_check,
    collections_create,
    collections_delete,
    collections_exists,
    collections_get,
    points_upsert,
    points_query_batch,
    points_scroll,
    /// §2 phase 2, `--set-payload`.
    points_set_payload,
    /// §2 phase 3, "any payload flag without `--skip-field-indices`": the RPC
    /// whose refusal made W12 a row this engine declined.
    points_create_field_index,

    // Listed in §2 against a bfb flag, scheduled for a later phase. These are
    // *deferred*, not refused, and the distinction is what the error message
    // has to convey.
    collections_list,
    collections_update,
    points_delete,
    points_get,

    // §2: "bfb searches via `QueryBatch`, not `Search`", and its table maps
    // `/qdrant.Points/Query` batched onto `QueryBatch`. These three are the
    // legacy single-query spellings of the same thing, not §1 non-goals; a
    // client that reaches one is told where the implemented endpoint is.
    points_search,
    points_search_batch,
    points_query,

    /// Not on §2's table at all.
    unknown,

    pub const Disposition = enum { implemented, deferred, legacy_alias, unlisted };

    /// The wire path each member answers to.
    ///
    /// A `switch` rather than a table beside the enum: a table is a second list
    /// of the same members, and forgetting an entry in it compiles into a
    /// member no request can ever reach. This way the compiler asks.
    pub fn method(self: Rpc) ?struct { service: []const u8, method: []const u8 } {
        return switch (self) {
            .health_check => .{ .service = "qdrant.Qdrant", .method = "HealthCheck" },
            .collections_create => .{ .service = "qdrant.Collections", .method = "Create" },
            .collections_delete => .{ .service = "qdrant.Collections", .method = "Delete" },
            .collections_exists => .{ .service = "qdrant.Collections", .method = "CollectionExists" },
            .collections_get => .{ .service = "qdrant.Collections", .method = "Get" },
            .collections_list => .{ .service = "qdrant.Collections", .method = "List" },
            .collections_update => .{ .service = "qdrant.Collections", .method = "Update" },
            .points_upsert => .{ .service = "qdrant.Points", .method = "Upsert" },
            .points_query_batch => .{ .service = "qdrant.Points", .method = "QueryBatch" },
            .points_scroll => .{ .service = "qdrant.Points", .method = "Scroll" },
            .points_delete => .{ .service = "qdrant.Points", .method = "Delete" },
            .points_get => .{ .service = "qdrant.Points", .method = "Get" },
            .points_set_payload => .{ .service = "qdrant.Points", .method = "SetPayload" },
            .points_create_field_index => .{ .service = "qdrant.Points", .method = "CreateFieldIndex" },
            .points_search => .{ .service = "qdrant.Points", .method = "Search" },
            .points_search_batch => .{ .service = "qdrant.Points", .method = "SearchBatch" },
            .points_query => .{ .service = "qdrant.Points", .method = "Query" },
            // Not a path; it is what `parse` returns when none matched.
            .unknown => null,
        };
    }

    pub fn parse(p: grpc.Path) Rpc {
        inline for (comptime std.enums.values(Rpc)) |rpc| {
            if (comptime rpc.method()) |m| {
                if (p.eql(m.service, m.method)) return rpc;
            }
        }
        return .unknown;
    }

    pub fn disposition(self: Rpc) Disposition {
        return switch (self) {
            .health_check,
            .collections_create,
            .collections_delete,
            .collections_exists,
            .collections_get,
            .points_upsert,
            .points_query_batch,
            .points_scroll,
            .points_set_payload,
            .points_create_field_index,
            => .implemented,

            .collections_list,
            .collections_update,
            .points_delete,
            .points_get,
            => .deferred,

            .points_search,
            .points_search_batch,
            .points_query,
            => .legacy_alias,

            .unknown => .unlisted,
        };
    }
};

pub fn err(req: *const server.Request, status: Status, message: []const u8) server.Completion {
    return .{
        .conn = req.conn,
        .stream_idx = req.stream_idx,
        .stream_id = req.stream_id,
        .status = status,
        .message = message,
    };
}

pub fn ok(req: *const server.Request, body: []const u8) server.Completion {
    std.debug.assert(req.stream_idx < 1024);
    return .{
        .conn = req.conn,
        .stream_idx = req.stream_idx,
        .stream_id = req.stream_id,
        .status = .ok,
        .body = body,
    };
}

/// §12: UNIMPLEMENTED "with a message naming the RPC".
///
/// The message is written into the response buffer so it names the *actual*
/// path rather than a generic string, §2 wants failures legible, and
/// "unimplemented" with no subject sends the reader back to the client's source
/// to work out which call it was.
///
/// It also has to point at the right part of the spec. §2's compatibility table
/// lists RPCs bfb reaches that are deferred to a later phase, `Points/Scroll`,
/// `Points/Delete`, `Collections/List`, and those are *scheduled*, not
/// refused. §1's non-goals are refused outright. Sending a client chasing "§1
/// non-goals" for a phase-2 RPC is a wrong answer that reads like a right one,
/// and W13 (scroll) walks straight into it.
fn unimplementedRpc(
    req: *const server.Request,
    path: grpc.Path,
    out: *server.ResponseBuf,
) server.Completion {
    const where = switch (Rpc.parse(path).disposition()) {
        .deferred => "listed in spec §2 as a later phase",
        .legacy_alias => "a legacy alias of /qdrant.Points/QueryBatch, which bfb uses (spec §2); deferred",
        .unlisted, .implemented => "see spec §1 non-goals",
    };

    const buf = out.available();
    const msg_text = std.fmt.bufPrint(
        buf,
        "/{s}/{s} is not implemented by strawmann ({s})",
        .{ path.service, path.method, where },
    ) catch {
        // A buffer too small for the message is not a reason to lose the
        // status; degrade to the generic text rather than to a 500.
        return err(req, .unimplemented, "RPC not implemented by strawmann");
    };
    return err(req, .unimplemented, msg_text);
}

/// Map a decode failure onto a status, preserving the detail the decoder
/// recorded about *which* unsupported construct was present.
pub fn decodeErr(req: *const server.Request, e: anyerror) server.Completion {
    return switch (e) {
        msg.DecodeError.Unimplemented => err(req, .unimplemented, msg.unimplemented_detail),
        msg.DecodeError.InvalidArgument => blk: {
            // Read once and clear, so a detail set by this request cannot be
            // reported by a later one on the same worker whose decoder had
            // nothing specific to say.
            const detail = msg.takeInvalidDetail();
            break :blk err(req, .invalid_argument, if (detail.len > 0) detail else "missing or invalid field");
        },
        else => err(req, .invalid_argument, "malformed protobuf"),
    };
}

// =========================================================================
// qdrant.Qdrant/HealthCheck
// =========================================================================

/// §2: "**HealthCheck is on the critical path of every connection.**
/// `qdrant-client` builds a client, calls `HealthCheck`, and compares the
/// returned `version` string against its own major/minor. A mismatch is a
/// warning, not an error, but a *failure* to answer is fatal."
fn healthCheck(req: *const server.Request, out: *server.ResponseBuf) server.Completion {
    var w = wire.Writer.init(out.available());
    const reply = msg.HealthCheckReply{
        .title = "strawmann",
        // A build-time knob (§2: "make the version string a config knob so the
        // compatibility check can be silenced for any client version").
        .version = build_options.qdrant_version,
    };
    reply.encode(&w) catch return err(req, .internal, "response buffer overflow");

    var c = ok(req, out.commit(w.pos));
    // `HealthCheckReply` is the one response with no `time` field, its field 2
    // is `version`. Appending a timing here would corrupt the very string
    // `qdrant-client` parses to decide whether it can talk to us at all.
    c.wants_time = false;
    return c;
}

// =========================================================================
// qdrant.Collections
// =========================================================================

/// §6.2: "Upsert decode writes **directly into final storage**, the point's
/// vector lands in `vectors.bin`'s mapped arena, not in an intermediate
/// `Vec<f32>`."
///
/// The one unavoidable copy is the memcpy out of the network frame into the
/// arena row, which `DenseVector.copyInto` performs. There is no intermediate
/// list of points and no per-point allocation.
fn upsert(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    const up = msg.UpsertPoints.decode(&r) catch |e| return decodeErr(req, e);

    const held = ctx.engine.acquire(up.collection_name) orelse
        return err(req, .not_found, "collection not found");
    defer held.release();
    const coll = held.coll;

    var it = up.pointIterator();
    var written: usize = 0;
    // Every exit from the loop below must invalidate, not just the successful
    // one. A batch that fails partway, a dimension mismatch, a bad id, or
    // hitting the preallocated capacity, has already written points, and
    // returning without invalidating leaves the collection `.ready` with
    // `indexed_count < count()`. `status()` then reports Yellow forever *and*
    // `ensureIndexBuilding` can never win its compare-exchange, so no rebuild
    // ever runs and the ingested points are permanently unsearchable.
    defer if (written > 0) core.collection.invalidateIndex(coll);

    while (it.next() catch |e| return decodeErr(req, e)) |pt| {
        const dense = if (pt.vectors.single) |v| v.dense else {
            return err(req, .unimplemented, "named vector spaces (phase 2)");
        };
        if (dense.dim() != coll.config.dim) {
            return err(req, .invalid_argument, "vector dimension does not match collection");
        }

        // Copy out of the frame into the workspace, preprocess, then into the
        // arena. The workspace hop exists because cosine normalisation mutates
        // the vector and the frame buffer is shared, immutable input.
        const scratch = ctx.workspace.query[0..coll.config.dim];
        _ = dense.copyInto(scratch) catch return err(req, .invalid_argument, "vector too large");
        if (!allFinite(scratch)) return err(req, .invalid_argument, "vector contains a NaN or infinite component");

        const external = toExternalId(pt.id) orelse
            return err(req, .invalid_argument, "malformed point id");

        // The payload goes in with the vector (§3: a blob per point), as the
        // wire bytes of its map entries; an upsert without one clears what
        // the point had, which is what Qdrant's does.
        _ = coll.upsertWithPayload(external, scratch, pt.payload_raw orelse &.{}, msg.PointStruct.payload_field) catch |e| return switch (e) {
            core.collection.Error.CapacityExceeded => err(req, .resource_exhausted, "collection at preallocated capacity (§3)"),
            core.collection.Error.DimensionMismatch => err(req, .invalid_argument, "vector dimension does not match collection"),
            core.collection.Error.PayloadRejected => err(req, .invalid_argument, "payload rejected: malformed map entries, or over 16 MiB for one point"),
            else => err(req, .internal, "upsert failed"),
        };
        written += 1;
    }

    // §2: "**`wait` on upsert.** `--wait-on-upsert` sets `wait: true`. Honour
    // it as 'visible to subsequent reads', not as 'fsynced'." Writes land in
    // the arena synchronously here, so they are already visible; the flag
    // changes only which status we report.
    const status: msg.UpdateStatus = if (up.wait) .completed else .acknowledged;

    var w = wire.Writer.init(out.available());
    const resp = msg.PointsOperationResponse{
        .operation_id = coll.nextOperationId(),
        .status = status,
        .time = 0,
    };
    resp.encode(&w) catch return err(req, .internal, "response buffer overflow");
    return ok(req, out.commit(w.pos));
}

// =========================================================================
// qdrant.Points/CreateFieldIndex, qdrant.Points/SetPayload
// =========================================================================

/// §2 phase 3: "`/qdrant.Points/CreateFieldIndex` — any payload flag without
/// `--skip-field-indices`." bfb calls it once per `-k` field, before the
/// upload, with `FieldType::Keyword` and `wait: true`, and `unwrap()`s the
/// answer — which is why W12 carried `--skip-field-indices` for as long as
/// this returned UNIMPLEMENTED, and why Qdrant's W12 was a full scan.
///
/// Keyword and integer indexes; the rest (`float`, `geo`, `text`, `bool`,
/// `datetime`, `uuid`) are refused by name, since a filter over them is not
/// evaluated either.
fn createFieldIndex(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    const cf = msg.CreateFieldIndexCollection.decode(&r) catch |e| return decodeErr(req, e);
    const held = ctx.engine.acquire(cf.collection_name) orelse
        return err(req, .not_found, "collection not found");
    defer held.release();
    const coll = held.coll;

    const kind = core.payload.Kind.fromFieldType(cf.field_type) orelse
        return err(req, .unimplemented, "field index types other than keyword and integer");
    coll.createPayloadIndex(cf.field_name, kind) catch |e| return switch (e) {
        core.collection.Error.IndexKindMismatch => err(req, .invalid_argument, "field is already indexed as another type"),
        core.collection.Error.OutOfMemory => err(req, .resource_exhausted, "out of memory building the payload index"),
        else => err(req, .internal, "payload index build failed"),
    };

    var w = wire.Writer.init(out.available());
    const resp = msg.PointsOperationResponse{
        .operation_id = coll.nextOperationId(),
        // Built synchronously, so it is complete whether or not `wait` was
        // asked; the status says what happened rather than what was asked.
        .status = .completed,
        .time = 0,
    };
    resp.encode(&w) catch return err(req, .internal, "response buffer overflow");
    return ok(req, out.commit(w.pos));
}

/// §2 phase 2: `/qdrant.Points/SetPayload`, `--set-payload`. Merges the
/// given keys over each selected point's payload; other keys survive
/// (Qdrant's `set_payload`, as opposed to `overwrite_payload`).
fn setPayload(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    const sp = msg.SetPayloadPoints.decode(&r) catch |e| return decodeErr(req, e);
    const held = ctx.engine.acquire(sp.collection_name) orelse
        return err(req, .not_found, "collection not found");
    defer held.release();
    const coll = held.coll;

    if (sp.filter_raw != null) return err(req, .unimplemented, "SetPayload selected by filter (only by id list)");
    if (sp.ids_raw == null) return err(req, .invalid_argument, "points_selector is required");

    var it = sp.idIterator();
    while (it.next() catch |e| return decodeErr(req, e)) |pid| {
        const external = toExternalId(pid) orelse
            return err(req, .invalid_argument, "malformed point id");
        const found = coll.setPayload(external, sp.payload_raw orelse &.{}, msg.SetPayloadPoints.payload_field) catch |e| return switch (e) {
            core.collection.Error.PayloadRejected => err(req, .invalid_argument, "payload rejected: malformed map entries, or over 16 MiB for one point"),
            else => err(req, .internal, "set payload failed"),
        };
        // Qdrant: `No point with id ... found`, NOT_FOUND.
        if (!found) return err(req, .not_found, "point not found");
    }

    var w = wire.Writer.init(out.available());
    const resp = msg.PointsOperationResponse{
        .operation_id = coll.nextOperationId(),
        .status = if (sp.wait) .completed else .acknowledged,
        .time = 0,
    };
    resp.encode(&w) catch return err(req, .internal, "response buffer overflow");
    return ok(req, out.commit(w.pos));
}

// =========================================================================
// qdrant.Points/Scroll
// =========================================================================

/// §2: `/qdrant.Points/Scroll`, `--scroll`, and UUID pre-fetch for
/// `--uuid-query`. §12 puts it in the phase-2 proto set; W13 is the workload.
///
/// Paging is by point id, which is a different traversal from anything else in
/// the engine: no vector is touched at all. That makes it the one workload that
/// exercises storage and the id map without the distance kernel in the way.
fn scroll(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    const sp = msg.ScrollPoints.decode(&r) catch |e| return decodeErr(req, e);

    const held = ctx.engine.acquire(sp.collection_name) orelse
        return err(req, .not_found, "collection not found");
    defer held.release();
    const coll = held.coll;

    // Everything we do not store is refused rather than ignored. Returning a
    // page with the vectors field simply absent would read as "these points
    // have no vectors", which is a different and wrong answer.
    if (sp.with_vectors) return err(req, .unimplemented, "vector retrieval in results (phase 2)");
    if (sp.has_order_by) return err(req, .unimplemented, "order_by over payload values");
    if (sp.has_shard_key) return err(req, .unimplemented, "sharding (§1 non-goal)");

    const start: ?core.ExternalId = if (sp.offset) |o| toExternalId(o) orelse
        return err(req, .invalid_argument, "malformed point id in offset") else null;

    // A filtered scroll admits the points the filter does, in id order, the
    // same predicate a filtered query uses (`preparePredicate`).
    const filter_opt = sp.filter() catch |e| return decodeErr(req, e);
    var pred_state: PredicateState = undefined;
    var pred: ?index.hnsw.Index.Filter = null;
    if (filter_opt) |*f| {
        ctx.workspace.ensureHnsw(ctx.engine.alloc, coll.config.capacity) catch
            return err(req, .internal, "workspace allocation failed");
        pred = preparePredicate(ctx, coll, f, &pred_state).filter;
    }

    // The page is bounded by the per-worker scratch, not by the client's
    // `limit`. An unbounded limit would otherwise size an allocation from a
    // wire field, the same shape of bug as the unbounded `hnsw_ef`.
    const cap = @min(sp.limit, ctx.workspace.scroll_ids.len);
    std.debug.assert(cap <= ctx.workspace.scroll_ids.len);
    const page = core.scroll.scrollFiltered(coll, start, pred, ctx.workspace.scroll_ids[0..cap]) catch
        return err(req, .internal, "scroll failed");

    var w = wire.Writer.init(out.available());
    var uuid_buf: [36]u8 = undefined;
    if (page.next) |n| {
        const nested = w.beginNested(1, 2) catch return err(req, .internal, "response buffer overflow");
        toPointId(n, &uuid_buf).encode(&w) catch return err(req, .internal, "response buffer overflow");
        w.endNested(nested) catch return err(req, .internal, "response buffer overflow");
    }
    for (page.ids) |id| {
        const blob: []const u8 = if (sp.with_payload) coll.payload.get(coll.id_space.lookup(id).?) else &.{};
        // Payloads are unbounded where ids are not: refuse the page by name
        // rather than failing mid-encode.
        const need = 64 + msg.payloadEncodedBound(blob);
        if (w.buf.len - w.pos < need) return err(req, .resource_exhausted, "Scroll page with payloads would exceed the response buffer; lower limit, or raise the server's response_buffer");
        const nested = w.beginNested(2, if (blob.len > 8000) 3 else 2) catch return err(req, .internal, "response buffer overflow");
        (msg.RetrievedPoint{ .id = toPointId(id, &uuid_buf), .payload = blob }).encode(&w) catch
            return err(req, .internal, "response buffer overflow");
        w.endNested(nested) catch return err(req, .internal, "response buffer overflow");
    }
    // `time` is field 3 here, not the 2 every other response in §12 uses, and
    // the worker appends it after the handler returns.
    var completion = ok(req, out.commit(w.pos));
    completion.time_field = 3;
    return completion;
}

/// `ExternalId` -> wire `PointId`.
///
/// UUIDs are stored as a `u128` and go back on the wire as their canonical
/// 36-character text, so `scratch` must outlive the encode. Callers pass a
/// buffer that lives to the end of the response.
fn toPointId(id: core.ExternalId, scratch: *[36]u8) msg.PointId {
    return switch (id) {
        .num => |v| .{ .num = v },
        .uuid => {
            id.formatUuid(scratch);
            return .{ .uuid = scratch };
        },
    };
}

fn toExternalId(id: msg.PointId) ?core.ExternalId {
    return switch (id) {
        .num => |v| core.ExternalId{ .num = v },
        .uuid => |s| parseUuidLenient(s),
    };
}

/// Every textual UUID form Qdrant's `PointId` accepts (it goes through Rust's
/// `Uuid::parse_str`): the canonical hyphenated form, the 32-hex "simple"
/// form, and either of those wrapped as `urn:uuid:...` or `{...}`. Only the
/// canonical form is stored or answered; the rest are rewritten to it here so
/// `ids.parseUuid` stays the single strict parser.
fn parseUuidLenient(text: []const u8) ?core.ExternalId {
    var s = text;
    if (std.ascii.startsWithIgnoreCase(s, "urn:uuid:")) {
        s = s["urn:uuid:".len..];
    } else if (s.len >= 2 and s[0] == '{' and s[s.len - 1] == '}') {
        s = s[1 .. s.len - 1];
    }
    if (s.len == 36) return core.ids.ExternalId.parseUuid(s);
    if (s.len != 32) return null;
    var canon: [36]u8 = undefined;
    var i: usize = 0;
    for (s, 0..) |ch, k| {
        if (k == 8 or k == 12 or k == 16 or k == 20) {
            canon[i] = '-';
            i += 1;
        }
        canon[i] = ch;
        i += 1;
    }
    return core.ids.ExternalId.parseUuid(&canon);
}

test "point ids: every UUID spelling Qdrant accepts maps to the same id" {
    const canon = core.ids.ExternalId.parseUuid("550e8400-e29b-41d4-a716-446655440000").?;
    const forms = [_][]const u8{
        "550e8400-e29b-41d4-a716-446655440000",
        "550E8400-E29B-41D4-A716-446655440000",
        "550e8400e29b41d4a716446655440000",
        "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
        "URN:UUID:550e8400e29b41d4a716446655440000",
        "{550e8400-e29b-41d4-a716-446655440000}",
        "{550e8400e29b41d4a716446655440000}",
    };
    for (forms) |f| {
        const got = toExternalId(.{ .uuid = f }) orelse return error.TestUnexpectedResult;
        try std.testing.expect(canon.eql(got));
    }
    // Still strict about everything else: wrong length, wrong hyphen
    // positions, non-hex, a brace with no partner.
    const bad = [_][]const u8{
        "",
        "550e8400-e29b-41d4-a716-44665544000",
        "550e8400e29b41d4a71644665544000",
        "550e8400e29b41d4a716446655440000zz",
        "550e8400-e29b41d4-a716-446655440000",
        "550e8400e29b41d4a71644665544000g",
        "{550e8400-e29b-41d4-a716-446655440000",
        "urn:uuid:",
    };
    for (bad) |b| try std.testing.expect(toExternalId(.{ .uuid = b }) == null);
}

// =========================================================================
// qdrant.Points/QueryBatch
// =========================================================================

/// §2: "**`bfb` searches via `QueryBatch`, not `Search`.**"
///
/// M1 answers every query by brute force. §6.5's `exact: true` path and the
/// default path are therefore the same code here, which is the honest state of
/// the engine at this milestone rather than a shortcut: `indexed_vectors_count`
/// reports 0, so a client that checks can tell.
fn queryBatch(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    const qb = msg.QueryBatchPoints.decode(&r) catch |e| return decodeErr(req, e);

    const held = ctx.engine.acquire(qb.collection_name) orelse
        return err(req, .not_found, "collection not found");
    const coll = held.coll;

    // Sized to the collection's preallocated capacity (§3), so it is allocated
    // once per worker and never on the query path thereafter.
    ctx.workspace.ensureHnsw(ctx.engine.alloc, coll.config.capacity) catch {
        held.release();
        return err(req, .internal, "workspace allocation failed");
    };

    // Gather before fanning out. For an *exact* batch the two go in opposite
    // directions: fanning K exact queries across workers costs K passes over
    // the same arena, and W9's own numbers say the engine ahead is the one that
    // reads less (findings 45). `runGatheredExact` reads each row once and
    // scores all K against it. It answers only the narrow case it is certain
    // about, all-exact and unfiltered, and returns null for anything else, so
    // the fan-out and the sequential path below are untouched.
    if (runGatheredExact(ctx, req, qb, coll, out, held)) |done| return done;

    // Fan out when there is more than one query and room to hold the
    // results; otherwise the plain sequential path below.
    if (BatchJob.plan(req, qb, coll)) |job| {
        return runBatchJob(ctx, req, job, held);
    }
    defer held.release();

    var w = wire.Writer.init(out.available());
    var resp = msg.QueryBatchResponseWriter.init(&w);

    var it = qb.queryIterator();
    while (it.next() catch |e| return decodeErr(req, e)) |q| {
        var fail: ?QueryFailure = null;
        const want = pageWant(q.limit, q.offset) orelse
            return err(req, .invalid_argument, "limit + offset exceeds this server's maximum of " ++ max_limit_text);
        const results = searchOne(ctx, coll, q, want, &fail) orelse {
            const f = fail.?;
            return err(req, f.status, f.message);
        };
        encodeOne(coll, q, results, &w, &resp, &fail) catch {
            const f = fail.?;
            return err(req, f.status, f.message);
        };
    }

    // `time` is filled in by the I/O thread from the measured elapsed time
    // (§2: "measured honestly at the RPC boundary"), so the handler writes
    // nothing here. See `finishTime` in the completion path.
    return ok(req, out.commit(w.pos));
}

/// Why one query of a batch was refused. Qdrant fails a `QueryBatch` as a
/// whole, so the first refusal is the batch's answer whichever query it was.
const QueryFailure = struct {
    status: Status,
    message: []const u8,
};

/// Answer an all-exact, unfiltered `QueryBatch` in one pass of the arena.
///
/// `null` when this batch is not that shape, which leaves every existing path
/// exactly as it was: more queries than `max_gather`, any query that is not
/// exact, any filter, any `offset`/`limit` past the page bound, any of the
/// options `searchOne` refuses by name. The narrowness is the point, since the
/// win is only available where the whole batch scans the same rows.
///
/// Correctness is `collection.bruteForceRangeMulti`'s: the differential test
/// there asserts K gathered queries return the same ids and the same scores as
/// K sequential ones, tombstones and filters included.
fn runGatheredExact(
    ctx: *Context,
    req: *const server.Request,
    qb: msg.QueryBatchPoints,
    coll: *core.Collection,
    out: *server.ResponseBuf,
    held: Engine.Handle,
) ?server.Completion {
    // Only where a scan is what every query would do anyway. `fullScanPreferred`
    // is the other route into the exact path and is a property of the
    // collection, so it applies to the whole batch or to none of it.
    const scan_all = fullScanPreferred(coll);

    var n: usize = 0;
    var it = qb.queryIterator();
    while (it.next() catch return null) |q| {
        if (n == core.collection.max_gather) return null;
        if (!(q.params.exact or scan_all)) return null;
        // Anything with a filter, a refused option or a page past the bound
        // goes the ordinary way rather than being half-handled here.
        if (q.filter() catch return null) |_| return null;
        if (q.params.indexed_only or q.with_vectors) return null;
        if (q.using) |name| if (name.len > 0) return null;
        const want = pageWant(q.limit, q.offset) orelse return null;
        if (want == 0 or want > Workspace.max_limit) return null;
        const query = q.query orelse return null;
        const dense = query.nearest.dense orelse return null;
        if (dense.dim() != coll.config.dim) return null;
        n += 1;
    }
    if (n < 2) return null; // one query is not a gather

    ctx.workspace.ensureGather(ctx.engine.alloc, coll) catch {
        held.release();
        return err(req, .internal, "workspace allocation failed");
    };

    // Decode and preprocess each query into its own slot, by the same funnel
    // `searchOne` uses: per datatype, not per metric.
    const dim = coll.config.dim;
    var queries: [core.collection.max_gather][]const f32 = undefined;
    var tops: [core.collection.max_gather]index.TopK = undefined;
    var top_ptrs: [core.collection.max_gather]*index.TopK = undefined;
    var wants: [core.collection.max_gather]usize = undefined;
    var i: usize = 0;
    it = qb.queryIterator();
    while (it.next() catch return null) |q| {
        const raw = ctx.workspace.gather_queries[i * dim ..][0..dim];
        const dense = q.query.?.nearest.dense.?;
        _ = dense.copyInto(raw) catch {
            held.release();
            return err(req, .invalid_argument, "query vector too large");
        };
        if (!allFinite(raw)) {
            held.release();
            return err(req, .invalid_argument, "query vector contains a NaN or infinite component");
        }
        if (coll.config.datatype.normalisesAtIngest(coll.config.metric)) {
            _ = dist.norm.preprocessInPlace(coll.config.metric, raw);
        }
        queries[i] = raw;
        wants[i] = pageWant(q.limit, q.offset).?;
        tops[i] = index.TopK.init(ctx.workspace.gather_results[i * Workspace.max_limit ..][0..wants[i]], wants[i]);
        top_ptrs[i] = &tops[i];
        i += 1;
    }

    // One pass, under one guard: the arena must not be replaced mid-scan, and
    // taking the guard once is also what makes this cheaper than K searches.
    {
        const guard = core.collection.SearchGuard.begin(coll);
        defer guard.end();
        core.collection.bruteForceRangeMulti(coll, queries[0..n], 0, @intCast(coll.id_space.count()), null, ctx.workspace.gather_probe, top_ptrs[0..n]);
    }
    defer held.release();

    var w = wire.Writer.init(out.available());
    var resp = msg.QueryBatchResponseWriter.init(&w);
    i = 0;
    it = qb.queryIterator();
    while (it.next() catch |e| return decodeErr(req, e)) |q| {
        var fail: ?QueryFailure = null;
        encodeOne(coll, q, tops[i].finish(), &w, &resp, &fail) catch {
            const f = fail.?;
            return err(req, f.status, f.message);
        };
        i += 1;
    }
    return ok(req, out.commit(w.pos));
}

/// Validate one query and search it into this worker's result heap. Returns
/// the ranked candidates, or null with `fail` set. Everything about the query
/// but the response encoding, so a fanned-out batch and the sequential path
/// answer a query identically.
fn searchOne(ctx: *Context, coll: *core.Collection, q: msg.QueryPoints, want: usize, fail: *?QueryFailure) ?[]index.Candidate {
    // An *empty* `Filter{}` is a filter with no conditions, which Qdrant
    // answers as no filter at all (`QueryPoints.filter`); a condition this
    // engine cannot evaluate is refused by name at decode.
    const filter_opt = q.filter() catch |e| return switch (e) {
        msg.DecodeError.Unimplemented => refuse(fail, .unimplemented, msg.unimplemented_detail),
        msg.DecodeError.InvalidArgument => blk: {
            const detail = msg.takeInvalidDetail();
            break :blk refuse(fail, .invalid_argument, if (detail.len > 0) detail else "malformed filter");
        },
        else => refuse(fail, .invalid_argument, "malformed filter"),
    };
    // Honouring it would mean skipping the unindexed tail, which this
    // engine has no way to do; ignoring it would return points the client
    // asked not to see. Same rule as `hnsw_config.on_disk`: refused by
    // name rather than silently degraded.
    if (q.params.indexed_only) return refuse(fail, .unimplemented, "params.indexed_only");
    if (q.with_vectors) return refuse(fail, .unimplemented, "vector retrieval in results (phase 2)");
    if (q.using) |name| {
        if (name.len > 0) return refuse(fail, .unimplemented, "named vector spaces (phase 2)");
    }

    const query = q.query orelse return refuse(fail, .invalid_argument, "query is required");
    const dense = query.nearest.dense orelse {
        if (query.nearest.id != null) return refuse(fail, .unimplemented, "query by point id (phase 2)");
        return refuse(fail, .invalid_argument, "query.nearest.dense is required");
    };
    if (dense.dim() != coll.config.dim) {
        return refuse(fail, .invalid_argument, "query dimension does not match collection");
    }

    // Decode into workspace, then preprocess exactly as ingest did.
    //
    // "Exactly as ingest did" is per *datatype*, not per metric
    // (`dist/datatype.zig`): a uint8 cosine collection stores raw bytes
    // and divides by the norms per comparison, because unit-vector
    // components truncate to 0 or 1 in a byte. Normalising the query
    // here anyway did just that: `Probe.init` converted the unit query to
    // u8, every component but the largest became 0, and every score was 0
    // (or NaN-guarded to 0). `Collection.prepareQuery` is the same funnel
    // for the in-process callers.
    const raw = ctx.workspace.query[0..coll.config.dim];
    _ = dense.copyInto(raw) catch return refuse(fail, .invalid_argument, "query vector too large");
    // A NaN score compares false both ways in the heaps (`heap.zig`
    // `better`), so one non-finite component poisons the whole result
    // set. Qdrant refuses these too.
    if (!allFinite(raw)) return refuse(fail, .invalid_argument, "query vector contains a NaN or infinite component");
    if (coll.config.datatype.normalisesAtIngest(coll.config.metric)) {
        _ = dist.norm.preprocessInPlace(coll.config.metric, raw);
    }

    // A query is as good a reason to start the build as a poll is. Builds
    // used to start only from `Collections/Get`, because that is what
    // bfb's `wait_index` loop calls; with `--skip-wait-index --search`
    // nothing ever called it, so the collection answered every query by
    // brute force for the whole run and reported Yellow throughout. One
    // atomic load per query when the index is already `.ready` or
    // `.building`, which is the steady state; the spawn happens once.
    //
    // Only for a query that would *use* the graph. An `exact` query never
    // does, and W9 (`--search-exact`) is the bandwidth measurement: a
    // background build it did not ask for, running on `build_threads`
    // cores while it scans, is exactly the confound §7.1 forbids.
    // §2: `params.hnsw_ef` and `params.exact` are the two knobs bfb sets.
    // `exact` forces brute force, it is the recall ground truth (§6.5) and
    // W9's benchmark, so the approximate path must never serve it.
    //
    // Not the only reason for a plain scan: Qdrant 1.19 answers an
    // unfiltered query with a plain scan whenever the collection is
    // smaller than `hnsw_config.full_scan_threshold` (`fullScanPreferred`
    // below), and a comparison of Qdrant-exact against strawmann-
    // approximate on a small collection is not a comparison of engines.
    const exact = q.params.exact or fullScanPreferred(coll);
    // The build trigger keys on the *computed* `exact`, not the flag: a
    // collection under the full-scan threshold would never use the graph
    // either, and spawning a build for it is the same confound.
    if (!exact and coll.count() > 0) ctx.engine.ensureIndexBuilding(coll);
    //
    // `hnsw_ef` is client-controlled and must be clamped to the preallocated
    // scratch. §6.3 forbids allocating on the query path, so the frontier
    // and result heaps are sized once at `Workspace.max_ef`; an unclamped
    // `hnsw_ef` of 10^9 would index past them. §4's sweep tops out at
    // ef=512, so the ceiling is far above any legitimate request.
    //
    // Absent means the collection's `ef_construct`, which is Qdrant's
    // rule (`HnswGraphConfig::new`: `ef: ef_construct`; then
    // `search_with_graph`: `params.hnsw_ef.unwrap_or(self.config.ef)`,
    // then `max(ef, top)`, VERIFIED in `lib/segment/src/index/hnsw_index/`).
    // It was 128 here, so a query with no `hnsw_ef` ran at a width
    // unrelated to how the graph was built and different from what
    // Qdrant would use for the same request, a recall difference with no
    // flag behind it.
    const ef = effectiveEf(q.params.hnsw_ef, coll.config.hnsw_ef_construct, want);
    var top = index.TopK.init(ctx.workspace.results, want);

    // The filter, as the traversal's admission predicate, or, when scoring
    // the matching set outright is cheaper, as the whole search
    // (`filteredPlan`). `select` verified every admitted point against its
    // blob, so the plain path is exact over the set.
    var pred_state: PredicateState = undefined;
    var pred: ?index.hnsw.Index.Filter = null;
    if (filter_opt) |*f| {
        const prepared = preparePredicate(ctx, coll, f, &pred_state);
        pred = prepared.filter;
        if (prepared.selected) |n| {
            const m0 = coll.config.hnsw_m * 2;
            const plan: FilteredPlan = if (exact) .scan else filteredPlan(n, prepared.bound, ef, m0);
            switch (plan) {
                .scan => {
                    core.collection.searchSelected(coll, raw, ctx.workspace.filter_bits, @intCast(prepared.bound), &top);
                    return top.finish();
                },
                .two_hop => pred.?.two_hop = true,
                .walk => {},
            }
        }
    }

    if (coll.quant.load(.acquire) != null and !exact) {
        // §6.7's two-stage path. `rescore` absent means the collection
        // default, which is on: silently skipping it would report a recall
        // the configuration cannot deliver (see the measured frontier in
        // core/collection.zig).
        const qp = core.quantized_search.QuantParams{
            .ignore = q.params.quantization.ignore orelse false,
            .rescore = q.params.quantization.rescore orelse true,
            .oversampling = q.params.quantization.oversampling orelse 1.0,
        };
        ctx.workspace.ensureQuant(ctx.engine.alloc, coll.config.dim) catch
            return refuse(fail, .internal, "workspace allocation failed");
        core.quantized_search.searchQuantized(
            coll,
            .{ .query = raw, .ef = ef, .limit = want, .quant = qp, .filter = pred },
            ctx.workspace.searchScratch(ctx.engine.alloc),
            &top,
        ) catch return refuse(fail, .internal, "quantized search failed");
    } else {
        core.collection.searchFiltered(coll, raw, ef, if (exact) .exact else .approximate, ctx.workspace.hnsw, pred, &top);
    }
    return top.finish();
}

/// Where a query's predicate keeps its context: the two shapes outlive the
/// call that builds them, so the caller owns the storage.
const PredicateState = struct {
    bits: core.payload.BitsCtx,
    eval: core.payload.EvalCtx,
};

const Prepared = struct {
    filter: index.hnsw.Index.Filter,
    /// Points the index selected, verified; null when no `must` condition
    /// is indexed and the predicate evaluates blobs directly.
    selected: ?usize,
    /// The point count the selection was built against.
    bound: usize,
};

/// Build the admission predicate for `f`: from the index into this worker's
/// bitset when a `must` condition has one, else by direct evaluation.
/// `ensureHnsw` must have run, so `filter_bits` covers the collection.
fn preparePredicate(ctx: *Context, coll: *core.Collection, f: *const core.payload.Filter, state: *PredicateState) Prepared {
    const bound = coll.id_space.count();
    std.debug.assert(ctx.workspace.filter_bits.len * 64 >= coll.config.capacity);
    if (coll.payload.select(f, ctx.workspace.filter_bits, bound)) |sel| {
        state.bits = .{ .bits = ctx.workspace.filter_bits, .bound = bound };
        return .{
            .filter = .{ .pred = core.payload.BitsCtx.pred, .ctx = &state.bits },
            .selected = sel.count,
            .bound = bound,
        };
    }
    state.eval = .{ .store = &coll.payload, .filter = f };
    return .{
        .filter = .{ .pred = core.payload.EvalCtx.pred, .ctx = &state.eval },
        .selected = null,
        .bound = bound,
    };
}

pub fn refuse(fail: *?QueryFailure, status: Status, message: []const u8) ?[]index.Candidate {
    fail.* = .{ .status = status, .message = message };
    return null;
}

/// Encode one query's ranked candidates as its `BatchResult`, applying the
/// query's `offset` and `score_threshold`. On failure `fail` names it.
fn encodeOne(
    coll: *core.Collection,
    q: msg.QueryPoints,
    results: []const index.Candidate,
    w: *wire.Writer,
    resp: *msg.QueryBatchResponseWriter,
    fail: *?QueryFailure,
) error{Refused}!void {
    const offset: usize = @intCast(q.offset);
    const start = @min(offset, results.len);

    // The response arena is fixed (`Config.response_buffer`, 256 KiB by
    // default) and a batch of large limits can exceed it. Checked before
    // any of this query's points are written, so the answer is the same
    // RESOURCE_EXHAUSTED naming the limit that an oversized request gets,
    // not an INTERNAL "response buffer overflow" that reads as a bug.
    // Payloads are the one part of a point with no fixed bound, so they are
    // summed here rather than priced into `maxEncodedSize`.
    var payload_bytes: usize = 0;
    if (q.with_payload) {
        for (results[start..]) |c| payload_bytes += msg.payloadEncodedBound(coll.payload.get(c.id));
    }
    if (!queryBatchFits(w.buf.len - w.pos, results.len - start, payload_bytes)) {
        fail.* = .{ .status = .resource_exhausted, .message = "QueryBatch response would exceed the response buffer; lower limit or batch size, or raise the server's response_buffer" };
        return error.Refused;
    }

    const overflow = QueryFailure{ .status = .internal, .message = "response buffer overflow" };
    const batch = resp.beginBatch() catch {
        fail.* = overflow;
        return error.Refused;
    };
    // Hoisted out of the loop: `ScoredPoint.encode` copies the text into
    // the response buffer, but the slice must still be live at the moment
    // it is read, so it cannot be scoped to the expression that builds it.
    var uuid_buf: [36]u8 = undefined;
    for (results[start..]) |c| {
        if (q.score_threshold) |th| {
            // Threshold applies to the *returned* score, which for Euclid
            // and Manhattan is order-reversed relative to the internal
            // similarity (§8.3). Comparing the wrong one would silently
            // invert the filter.
            //
            // Strict, as Qdrant's is: `Distance::check_threshold` is
            // `score > threshold` for LargeBetter and `score < threshold`
            // for SmallBetter (VERIFIED, `lib/segment/src/types.rs`). A
            // point scoring exactly the threshold is dropped there, and
            // it used to be kept here, so the two engines disagreed on
            // result count for the same request whenever a score landed
            // on the threshold, which for `score_threshold: 0` with
            // integer-valued uint8 dot products is not rare.
            const returned = coll.config.metric.postprocess(c.score);
            const passes = if (coll.config.metric.postprocessReversesOrder())
                returned < th
            else
                returned > th;
            if (!passes) continue;
        }
        const ext = coll.id_space.external(c.id);
        resp.point(.{
            .id = switch (ext) {
                .num => |v| .{ .num = v },
                .uuid => blk: {
                    ext.formatUuid(&uuid_buf);
                    break :blk .{ .uuid = &uuid_buf };
                },
            },
            .score = coll.config.metric.postprocess(c.score),
            .version = 0,
            .payload = if (q.with_payload) coll.payload.get(c.id) else &.{},
        }) catch {
            fail.* = overflow;
            return error.Refused;
        };
    }
    resp.endBatch(batch) catch {
        fail.* = overflow;
        return error.Refused;
    };
}

/// A `QueryBatch` fanned out across the worker pool.
///
/// One request, one worker was the rule, and for a batch of n queries that
/// is n searches in a row on one core while the other workers idle: the
/// conformance binary's single client sending `QueryBatch` of 32 measured
/// Qdrant faster at every `ef` on a run where bfb, one query per request,
/// measured strawmann at 1.5-1.9x, because Qdrant fans a batch out and this
/// engine did not (`docs/comparison-sift1m.md`). So the worker that
/// dequeues a batch decodes it once, submits every query but the first back
/// onto the same queue as a sub-request (`Request.job`), runs the first
/// itself, and returns a *detached* completion; each sub-request searches
/// on whichever worker dequeued it, into that worker's own scratch, and
/// copies its ranked candidates into this job; the worker that finishes the
/// last query encodes the response in wire order and completes the stream
/// (`server.completeDetached`).
///
/// The job and every query's candidates live in the spare bytes of the
/// stream's request buffer after the body (`server.requestSpare`): free
/// until the stream completes, and the only place §6.3's no-allocation rule
/// leaves for state that outlives the dispatching call. A batch whose
/// candidates would not fit there, or a batch of one, takes the sequential
/// path; nothing is refused for lack of room.
///
/// Semantics are unchanged: results per query are what `searchOne` returns
/// on any worker, the first refusal fails the whole batch as Qdrant's does,
/// `time` still runs from arrival to the last query's end, and the
/// collection handle is held until the finisher releases it.
const BatchJob = struct {
    /// Above this the batch runs sequentially. Qdrant's own clients send
    /// batches of tens to a few hundred; the bound keeps the header a
    /// fixed size rather than a second allocation.
    pub const max_queries = 1024;

    const Slot = struct {
        off: u32,
        want: u32,
        /// Candidates written, valid once the query's worker has finished.
        len: u32 = 0,
    };

    remaining: std.atomic.Value(u32),
    /// The first refusal wins, in two words rather than one.
    ///
    /// `claimed` decides the winner, `failure` is the detail, and `failed` is
    /// what advertises that the detail is there: written in that order, so a
    /// thread that sees `failed` set can read `failure`. One word doing both
    /// jobs cannot have that property, because the winner is only known after
    /// the exchange that sets the flag, so the detail necessarily landed
    /// after it. That was safe only by way of `remaining`: the writer's store
    /// is ordered before its own decrement and `finishBatchJob` reads after
    /// the last one. `runBatchSub` already loads the flag on its own, so the
    /// property the field comment claimed is the one it should have.
    claimed: std.atomic.Value(bool),
    failed: std.atomic.Value(bool),
    failure: QueryFailure,
    n: u32,
    /// The dispatching worker's handle on the collection, held for the
    /// job's lifetime so no sub-request can find it dropped; the finisher
    /// releases it after completing the stream.
    held: Engine.Handle,
    slots: [max_queries]Slot,
    candidates: []index.Candidate,

    /// Lay a job out in the request's spare bytes, or null when the batch
    /// is a single query or the candidates would not fit. `pageWant`
    /// refusals are left to the sequential path so the message is the same.
    fn plan(req: *const server.Request, qb: msg.QueryBatchPoints, coll: *core.Collection) ?*BatchJob {
        _ = coll;
        const spare = server.requestSpare(req);
        if (spare.len < @sizeOf(BatchJob)) return null;
        const job: *BatchJob = @ptrCast(@alignCast(spare.ptr));
        const cand_bytes = spare[@sizeOf(BatchJob)..];
        const cands: []index.Candidate = @alignCast(std.mem.bytesAsSlice(index.Candidate, cand_bytes[0 .. cand_bytes.len - cand_bytes.len % @sizeOf(index.Candidate)]));

        var it = qb.queryIterator();
        var n: u32 = 0;
        var off: usize = 0;
        while (it.next() catch return null) |q| {
            if (n == max_queries) return null;
            const want = pageWant(q.limit, q.offset) orelse return null;
            if (off + want > cands.len) return null;
            job.slots[n] = .{ .off = @intCast(off), .want = @intCast(want) };
            off += want;
            n += 1;
        }
        if (n < 2) return null;
        job.n = n;
        job.remaining = std.atomic.Value(u32).init(n);
        job.claimed = std.atomic.Value(bool).init(false);
        job.failed = std.atomic.Value(bool).init(false);
        job.candidates = cands[0..off];
        return job;
    }

    fn fail(self: *BatchJob, f: QueryFailure) void {
        // First writer wins; later refusals are dropped, as they are on the
        // sequential path where the first one returns.
        if (self.claimed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return;
        self.failure = f;
        // Release, and after the detail: whoever acquires this flag may read
        // `failure`.
        self.failed.store(true, .release);
    }
};

/// Batches that were fanned out (not those that took the sequential path).
/// A test reads it to prove the parallel path ran; nothing else does.
pub var batches_fanned_out: std.atomic.Value(u64) = .init(0);

/// The dispatching worker: submit queries 1..n-1, run query 0, detach.
fn runBatchJob(ctx: *Context, req: *const server.Request, job: *BatchJob, held: Engine.Handle) server.Completion {
    // The handle outlives this call; the finisher releases it. Kept in the
    // job so the sub-requests need nothing but the job pointer.
    job.held = held;
    _ = batches_fanned_out.fetchAdd(1, .monotonic);
    var i: u32 = 1;
    while (i < job.n) : (i += 1) {
        var sub = req.*;
        sub.job = @ptrCast(job);
        sub.sub = i;
        // A full or closed queue does not fail the batch: the piece runs
        // here instead, which is what the sequential path would have done.
        if (!server.submitSub(req, sub)) runBatchSub(ctx, &sub);
    }
    var mine = req.*;
    mine.job = @ptrCast(job);
    mine.sub = 0;
    runBatchSub(ctx, &mine);
    return .{ .conn = req.conn, .stream_idx = req.stream_idx, .stream_id = req.stream_id, .status = .ok, .detached = true };
}

/// One query of a fanned-out batch, on whichever worker dequeued it.
fn runBatchSub(ctx: *Context, req: *const server.Request) void {
    const job: *BatchJob = @ptrCast(@alignCast(req.job.?));
    const i = req.sub;
    const slot = &job.slots[i];

    // Everything before the search that could fail is a decode of bytes
    // already validated by `plan`, so these `catch`es name a state that
    // cannot occur; they fail the batch loudly rather than assert, since a
    // wrong answer on the wire is what §1 forbids.
    blk: {
        if (job.failed.load(.acquire)) break :blk; // someone already refused; skip the work
        const body = grpc.decodeMessage(req.body) catch {
            job.fail(.{ .status = .invalid_argument, .message = "malformed gRPC message frame" });
            break :blk;
        };
        var r = wire.Reader.init(body);
        const qb = msg.QueryBatchPoints.decode(&r) catch {
            job.fail(.{ .status = .invalid_argument, .message = "malformed protobuf" });
            break :blk;
        };
        const coll = job.held.coll;
        ctx.workspace.ensureHnsw(ctx.engine.alloc, coll.config.capacity) catch {
            job.fail(.{ .status = .internal, .message = "workspace allocation failed" });
            break :blk;
        };
        var it = qb.queryIterator();
        it.skip(i) catch {
            job.fail(.{ .status = .invalid_argument, .message = "malformed protobuf" });
            break :blk;
        };
        const q = (it.next() catch null) orelse {
            job.fail(.{ .status = .invalid_argument, .message = "malformed protobuf" });
            break :blk;
        };
        var f: ?QueryFailure = null;
        const results = searchOne(ctx, coll, q, slot.want, &f) orelse {
            job.fail(f.?);
            break :blk;
        };
        std.debug.assert(results.len <= slot.want);
        @memcpy(job.candidates[slot.off..][0..results.len], results);
        slot.len = @intCast(results.len);
    }

    // The last worker out finishes the stream. `acq_rel`: this worker's
    // slot write happens-before the finisher's read, and the finisher sees
    // every other worker's slot too.
    if (job.remaining.fetchSub(1, .acq_rel) == 1) finishBatchJob(req, job);
}

/// Encode every query's candidates in wire order and complete the stream.
fn finishBatchJob(req: *const server.Request, job: *BatchJob) void {
    var parent = req.*;
    parent.job = null;
    parent.sub = 0;
    var rb = server.responseArena(&parent);

    const completion: server.Completion = blk: {
        if (job.failed.load(.acquire)) break :blk err(&parent, job.failure.status, job.failure.message);
        const body = grpc.decodeMessage(parent.body) catch break :blk err(&parent, .invalid_argument, "malformed gRPC message frame");
        var r = wire.Reader.init(body);
        const qb = msg.QueryBatchPoints.decode(&r) catch break :blk err(&parent, .invalid_argument, "malformed protobuf");
        const coll = job.held.coll;

        var w = wire.Writer.init(rb.available());
        var resp = msg.QueryBatchResponseWriter.init(&w);
        var it = qb.queryIterator();
        var i: u32 = 0;
        while (it.next() catch null) |q| : (i += 1) {
            std.debug.assert(i < job.n);
            const slot = job.slots[i];
            const results = job.candidates[slot.off..][0..slot.len];
            var f: ?QueryFailure = null;
            encodeOne(coll, q, results, &w, &resp, &f) catch break :blk err(&parent, f.?.status, f.?.message);
        }
        break :blk ok(&parent, rb.commit(w.pos));
    };
    server.completeDetached(&parent, &rb, completion);
    // The dispatching worker's handle, held for the job's lifetime so the
    // collection could not be dropped under a sub-request. Released after
    // the completion is handed off: nothing reads the job past this point,
    // and the request buffer it lives in is the stream's until the I/O
    // thread drains that completion.
    job.held.release();
}

/// True when every component is a number the kernels and heaps can order.
///
/// One pass, but a cheap one: the same cache lines are about to be read again
/// by the preprocessing or the copy into the arena, so it is a bounds check
/// rather than a second sweep of memory.
fn allFinite(v: []const f32) bool {
    for (v) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

test "allFinite refuses NaN and both infinities, accepts zero and the extremes" {
    try std.testing.expect(allFinite(&.{}));
    try std.testing.expect(allFinite(&.{ 0, -0.0, 1.5, std.math.floatMax(f32), -std.math.floatMax(f32) }));
    try std.testing.expect(!allFinite(&.{ 1, std.math.nan(f32) }));
    try std.testing.expect(!allFinite(&.{ std.math.inf(f32), 1 }));
    try std.testing.expect(!allFinite(&.{ 1, -std.math.inf(f32), 1 }));
}

const max_limit_text = std.fmt.comptimePrint("{d}", .{Workspace.max_limit});

/// `limit + offset` for a QueryBatch page, or null when the preallocated
/// result heap (`Workspace.max_limit`) cannot hold it. Both come off the wire
/// as u64 and are added as u64, so `offset = 2^64-1` is refused rather than
/// wrapped.
fn pageWant(limit: u64, offset: u64) ?usize {
    const sum = std.math.add(u64, limit, offset) catch return null;
    if (sum > Workspace.max_limit) return null;
    return @intCast(sum);
}

test "pageWant refuses limit + offset over max_limit, and cannot overflow" {
    const m = Workspace.max_limit;
    try std.testing.expectEqual(@as(?usize, 10), pageWant(10, 0));
    try std.testing.expectEqual(@as(?usize, m), pageWant(m, 0));
    try std.testing.expectEqual(@as(?usize, m), pageWant(m - 100, 100));
    try std.testing.expectEqual(@as(?usize, null), pageWant(m + 1, 0));
    try std.testing.expectEqual(@as(?usize, null), pageWant(0, m + 1));
    try std.testing.expectEqual(@as(?usize, null), pageWant(m, 1));
    try std.testing.expectEqual(@as(?usize, null), pageWant(1, std.math.maxInt(u64)));
    try std.testing.expectEqual(@as(?usize, null), pageWant(std.math.maxInt(u64), std.math.maxInt(u64)));
}

/// Whether one more `BatchResult` of `points` scored points fits in the
/// `remaining` bytes of the response buffer, at the worst-case encoding.
///
/// Per point: the `BatchResult.result` tag and 2-byte length reservation the
/// writer uses, plus `ScoredPoint.maxEncodedSize` (a UUID id is the widest).
/// Per batch: the `QueryBatchResponse.result` tag and its 4-byte reservation.
/// And the `time` field the I/O thread appends after the handler returns
/// (tag + f64), which the last batch must leave room for.
fn queryBatchFits(remaining: usize, points: usize, payload_bytes: usize) bool {
    // Four, not three: a point carrying a payload over 8 KiB reserves a
    // third length byte (`QueryBatchResponseWriter.point`).
    const per_point = 4 + msg.ScoredPoint.maxEncodedSize();
    const per_batch = 5;
    const time_field = 9;
    const need = std.math.mul(usize, points, per_point) catch return false;
    const with_payload = std.math.add(usize, need, payload_bytes) catch return false;
    return with_payload + per_batch + time_field <= remaining;
}

test "queryBatchFits is the worst-case bound: never says yes to what cannot be encoded" {
    // Encode a batch of `n` UUID points, the widest kind, into a buffer
    // exactly as large as the predicate allows, and check it always fits.
    const per_point = comptime 4 + msg.ScoredPoint.maxEncodedSize();
    var storage_buf: [100 * per_point + 5 + 9]u8 = undefined;
    for ([_]usize{ 0, 1, 7, 100 }) |n| {
        const buf = storage_buf[0 .. n * per_point + 5 + 9];
        var w = wire.Writer.init(buf);
        var resp = msg.QueryBatchResponseWriter.init(&w);
        try std.testing.expect(queryBatchFits(buf.len, n, 0));
        try std.testing.expect(!queryBatchFits(buf.len - 1, n, 0));
        try std.testing.expect(!queryBatchFits(buf.len, n, 1));
        const b = try resp.beginBatch();
        for (0..n) |_| {
            try resp.point(.{
                .id = .{ .uuid = "550e8400-e29b-41d4-a716-446655440000" },
                .score = -1.0e30,
                .version = std.math.maxInt(u64),
            });
        }
        try resp.endBatch(b);
        try resp.finishTime(1.0);
    }
    // 256 KiB, the production default: one page at the largest limit fits,
    // a second one in the same response does not.
    const default_buffer = 256 * 1024;
    try std.testing.expect(queryBatchFits(default_buffer, Workspace.max_limit, 0));
    try std.testing.expect(!queryBatchFits(default_buffer - Workspace.max_limit * per_point, Workspace.max_limit, 0));
    // Overflow of the multiplication is "does not fit", not a panic.
    try std.testing.expect(!queryBatchFits(std.math.maxInt(usize), std.math.maxInt(usize), 0));
    try std.testing.expect(!queryBatchFits(std.math.maxInt(usize), 1, std.math.maxInt(usize)));
}

/// Qdrant 1.19's rule for when a graph is not worth traversing.
///
/// `hnsw/read_view/dispatch.rs`: an unfiltered search is `plain_search` when
/// `exact || is_hnsw_disabled || available_vector_count < full_scan_threshold`,
/// where that last threshold is in *points* and was derived at build time
/// (`hnsw/build.rs`) from the configured one in kilobytes:
/// `full_scan_threshold_kb * 1024 / avg_vector_size`, with
/// `avg_vector_size = size_of_available_vectors_in_bytes / total_vector_count`
/// = `dim * size_of::<T>()` for a dense storage of element `T`; the fallback
/// when that division has nothing to divide is 1. Quantization does not
/// enter: the size is the vector storage's, not the codes'. (VERIFIED at
/// v1.19.0; `DEFAULT_FULL_SCAN_THRESHOLD` is 10_000, `lib/segment/src/types.rs`.)
///
/// So a 1000-point d=128 fp32 collection, 512 KB, is answered exactly by
/// Qdrant at every `ef`, and strawmann accepted `full_scan_threshold` and
/// traversed anyway. Same rule, same integer arithmetic. The one divergence:
/// Qdrant's plain scan on a quantized collection scores the codes and
/// rescores, whereas strawmann's exact path is the fp32 scan (§6.5), which is
/// the answer that scan approximates.
///
/// Live points, as Qdrant's `available_vector_count` is (deleted excluded).
fn fullScanPreferred(coll: *const core.Collection) bool {
    return coll.count() < fullScanThresholdPoints(coll);
}

/// Whether to score a filter's matching set outright instead of traversing
/// the graph under it.
///
/// A filtered traversal admits to the result heap and never prunes the
/// frontier until that heap is *full* (`hnsw.Index.Filter`), so it only stops
/// early once it has met `ef` admitted points — and it meets them at the
/// filter's selectivity. Expected nodes visited is therefore `ef / s` for
/// selectivity `s`, each scoring `m0` neighbours, and below some `s` that
/// exceeds the graph and the walk visits everything.
///
/// Measured, not argued: W12's 200-of-200,000 keyword filter
/// (s = 0.001) ran at 4.5 qps where a *full fp32 scan* of five times the data
/// (W9, 1M points) runs at 192 — six million distance computations per query
/// for a set of 200 points the index had already named. The dispatch had
/// compared the matching set against `full_scan_threshold`, which is a
/// statement about unfiltered scans (10 KB is 20 points at d=128) and says
/// nothing about when a filtered traversal stops working.
///
/// So compare the two costs directly, using the *exact* count the index gives
/// rather than Qdrant's estimate:
///
///     plain      = selected
///     traversal  = (ef / (selected / n)) * m0
///
/// and plain wins exactly when `selected² < ef · m0 · n`. At n = 200,000,
/// ef = 128, m0 = 32 the crossover is a matching set of 28,621: W12's 200
/// scores directly, and a filter admitting half the collection still
/// traverses, which is the regime where the heap fills and the traversal is
/// the cheaper answer.
///
/// A filter no posting list answers has no count to compare (`select`
/// returns null) and is evaluated against the blobs during a traversal; a
/// selective one degenerates there exactly as this fixes for the indexed
/// case, which is the cost `docs/spec.md` records for an unindexed filter.
/// How an indexed filter's query is answered: score the matching set, walk the
/// graph ACORN-1 style (`hnsw.Index.expandTwoHop`), or walk it scoring every
/// neighbour.
const FilteredPlan = enum { scan, two_hop, walk };

/// The plan for a filter the index counted exactly.
///
/// The two-hop walk scores only admitted points, so its cost does not grow as
/// the filter tightens the way the plain walk's does (`plainFilteredSearch`).
/// What it needs is enough admitted points within two hops to fill each
/// expansion's `m0`: `m0²` two-hop neighbours at selectivity `s` hold
/// `s · m0²` of them, so below `s = 1 / m0` the expansions run short and the
/// walk strands. Measured in-process on 200,000 dbpedia-openai-1m vectors
/// (m=16, so m0=32), 300 held-out queries:
///
///                      recall@10 at ef 128    ms/query
///     1%, two-hop            0.8877             0.91   (s < 1/m0: strands)
///     1%, scan               1.0000             0.45
///     10%, two-hop           0.9900             1.57
///     10%, plain walk        0.9973             4.32
///     10%, scan              1.0000             4.87
///
/// Above that floor the two-hop walk scored about `1.5 · ef · m0` rows per
/// query (1.57 ms at ef 128 is ~6,500 rows at the scan's 0.24 µs each; 5.2 ms
/// at ef 512 is ~21,600), so the scan wins while the matching set is smaller
/// than that. Below the floor the choice is the one it always was, between
/// the scan and the plain walk.
fn filteredPlan(selected: usize, bound: usize, ef: usize, m0: usize) FilteredPlan {
    const reach = std.math.mul(usize, selected, @max(m0, 1)) catch std.math.maxInt(usize);
    if (reach >= bound) {
        const walk = blk: {
            const a = std.math.mul(usize, @max(ef, 1), @max(m0, 1)) catch break :blk std.math.maxInt(usize);
            break :blk (std.math.mul(usize, a, 3) catch std.math.maxInt(usize)) / 2;
        };
        return if (selected < walk) .scan else .two_hop;
    }
    return if (plainFilteredSearch(selected, bound, ef, m0)) .scan else .walk;
}

fn plainFilteredSearch(selected: usize, bound: usize, ef: usize, m0: usize) bool {
    // Saturating, because all four come from a request or a collection and
    // the products are large: at the clamps `ef · m0 · n` is ~10^12.
    const lhs = std.math.mul(usize, selected, selected) catch return false;
    const rhs = blk: {
        const a = std.math.mul(usize, @max(ef, 1), @max(m0, 1)) catch break :blk std.math.maxInt(usize);
        break :blk std.math.mul(usize, a, @max(bound, 1)) catch std.math.maxInt(usize);
    };
    return lhs < rhs;
}

/// `hnsw_config.full_scan_threshold` in points, Qdrant's integer arithmetic.
fn fullScanThresholdPoints(coll: *const core.Collection) usize {
    const avg_vector_size = coll.config.dim * coll.config.datatype.elemSize();
    // `saturating_mul`, as Qdrant's is: the value is client-supplied.
    const kb = coll.config.hnsw_full_scan_threshold_kb;
    const bytes: usize = if (kb > std.math.maxInt(usize) / 1024) std.math.maxInt(usize) else kb * 1024;
    return if (avg_vector_size == 0) 1 else bytes / avg_vector_size;
}

/// The traversal width for one query: Qdrant's rule, clamped to the scratch.
///
/// `hnsw_ef` absent means the collection's `ef_construct`, then at least
/// `top` (`max(ef, top)` in `graph_layers.rs`), then never more than
/// `Workspace.max_ef` because the heaps are preallocated at that size (§6.3).
/// Split out so the rule is testable without a socket: it used to be inline
/// with a literal 128 in the else-branch, and nothing exercised that branch.
fn effectiveEf(requested: ?u64, ef_construct: usize, want: usize) usize {
    const base: usize = if (requested) |e|
        @min(@as(usize, @intCast(@min(e, @as(u64, Workspace.max_ef)))), Workspace.max_ef)
    else
        @min(ef_construct, Workspace.max_ef);
    return @min(@max(base, want), Workspace.max_ef);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "a selective filter scores its matching set; a permissive one traverses" {
    // The regime W12 measured: 200 of 200,000 at ef=128, m0=32. It traversed,
    // and the traversal visited the whole graph because the result heap never
    // filled — 4.5 qps against the 192 qps of a full fp32 scan of five times
    // the data.
    try testing.expect(plainFilteredSearch(200, 200_000, 128, 32));
    // The crossover, sqrt(ef*m0*n) = 28,621: either side of it.
    try testing.expect(plainFilteredSearch(28_000, 200_000, 128, 32));
    try testing.expect(!plainFilteredSearch(29_000, 200_000, 128, 32));
    // A permissive filter traverses: the heap fills, and scanning half the
    // collection is the more expensive answer.
    try testing.expect(!plainFilteredSearch(100_000, 200_000, 128, 32));
    // A wider walk is worth more work, so it raises the crossover.
    try testing.expect(plainFilteredSearch(29_000, 200_000, 512, 32));
    // Every filter on a small collection scores directly: at n=600 the
    // crossover (sqrt(600*32*600) = 3,394) is past the collection itself.
    try testing.expect(plainFilteredSearch(590, 600, 600, 32));
    // Degenerate inputs answer rather than divide by zero or overflow.
    try testing.expect(plainFilteredSearch(0, 0, 0, 0));
    try testing.expect(!plainFilteredSearch(std.math.maxInt(usize), 1, 1, 1));
    try testing.expect(plainFilteredSearch(1, std.math.maxInt(usize), std.math.maxInt(usize), 2));
}

test "an indexed filter scans below 1/m0, walks two-hop above it, and scans again at a wide ef" {
    // W12's two grades on bench12 (n = 200,000, m0 = 32).
    try testing.expectEqual(FilteredPlan.scan, filteredPlan(2_000, 200_000, 128, 32));
    try testing.expectEqual(FilteredPlan.two_hop, filteredPlan(20_000, 200_000, 128, 32));
    try testing.expectEqual(FilteredPlan.two_hop, filteredPlan(20_000, 200_000, 256, 32));
    // At ef 512 the walk's ~24,600 rows cost more than scoring the 20,000.
    try testing.expectEqual(FilteredPlan.scan, filteredPlan(20_000, 200_000, 512, 32));
    // A permissive filter walks two-hop too; it used to walk scoring everything.
    try testing.expectEqual(FilteredPlan.two_hop, filteredPlan(100_000, 200_000, 128, 32));
    // The floor is `selected · m0 >= n`: either side of 6,250.
    try testing.expectEqual(FilteredPlan.two_hop, filteredPlan(6_250, 200_000, 32, 32));
    try testing.expect(filteredPlan(6_249, 200_000, 32, 32) != .two_hop);
    // Below the floor, the old rule: a large sparse set on a huge collection
    // at a narrow ef still walks rather than scanning millions.
    try testing.expectEqual(FilteredPlan.walk, filteredPlan(100_000, 10_000_000, 16, 32));
    // A small collection scores every filter directly.
    try testing.expectEqual(FilteredPlan.scan, filteredPlan(590, 600, 600, 32));
    try testing.expectEqual(FilteredPlan.scan, filteredPlan(120, 600, 16, 32));
    // Degenerate inputs answer rather than overflow or divide by zero.
    try testing.expectEqual(FilteredPlan.scan, filteredPlan(0, 0, 0, 0));
    _ = filteredPlan(std.math.maxInt(usize), std.math.maxInt(usize), std.math.maxInt(usize), std.math.maxInt(usize));
}

test "hnsw_ef defaults to the collection's ef_construct, then max(ef, top), then the clamp" {
    // Absent: ef_construct, not a constant. A collection built at
    // ef_construct=64 searched at 64 by default, as Qdrant does; it used to
    // search at 128 whatever the build width was.
    try testing.expectEqual(@as(usize, 64), effectiveEf(null, 64, 10));
    try testing.expectEqual(@as(usize, 100), effectiveEf(null, 100, 10));
    // `max(ef, top)`: a request for more results than ef widens ef.
    try testing.expectEqual(@as(usize, 200), effectiveEf(null, 64, 200));
    try testing.expectEqual(@as(usize, 200), effectiveEf(50, 64, 200));
    // Explicit wins over the default.
    try testing.expectEqual(@as(usize, 300), effectiveEf(300, 64, 10));
    // Both bounded by the preallocated scratch.
    try testing.expectEqual(Workspace.max_ef, effectiveEf(1 << 40, 64, 10));
    try testing.expectEqual(Workspace.max_ef, effectiveEf(null, 1 << 20, 10));
}

test "engine create, find and drop" {
    var e = Engine.init(testing.allocator);
    defer e.deinit();

    try testing.expectEqual(@as(?*core.Collection, null), e.find("bench"));
    const c = try e.create("bench", .{ .dim = 4, .metric = .dot, .capacity = 16 });
    try testing.expectEqual(c, e.find("bench").?);
    try testing.expect(e.drop("bench"));
    try testing.expect(!e.drop("bench"));
    try testing.expectEqual(@as(?*core.Collection, null), e.find("bench"));
}

test "creating an existing collection returns the existing one" {
    var e = Engine.init(testing.allocator);
    defer e.deinit();
    const a = try e.create("x", .{ .dim = 4, .metric = .dot, .capacity = 8 });
    const b = try e.create("x", .{ .dim = 8, .metric = .euclid, .capacity = 8 });
    try testing.expectEqual(a, b);
    try testing.expectEqual(@as(usize, 4), b.config.dim);
}

test "a collection held by a request is not freed under it" {
    // `Delete` runs on a worker thread like every other RPC, so a drop can
    // overlap a search. Before `acquire`, the drop freed the arena the search
    // was reading; the failure mode is a use-after-free, not a leak, so no
    // allocator check would have caught it.
    var e = Engine.init(testing.allocator);
    defer e.deinit();
    _ = try e.create("held", .{ .dim = 4, .metric = .dot, .capacity = 8 });

    const held = e.acquire("held").?;
    try testing.expectEqual(@as(usize, 1), held.coll.users.load(.acquire));

    // While it is held, a lookup by name still finds it, but the collection is
    // pinned: a `drop` would have to wait.
    try testing.expect(e.find("held") != null);

    // The drop unlinks first, so a second acquire cannot appear mid-flight.
    // Release, then drop, which is the ordering a real request produces.
    held.release();
    try testing.expectEqual(@as(usize, 0), held.coll.users.load(.acquire));
    try testing.expect(e.drop("held"));
    try testing.expect(e.find("held") == null);
}

test "drop waits for a request that is already inside" {
    var e = Engine.init(testing.allocator);
    defer e.deinit();
    _ = try e.create("busy", .{ .dim = 4, .metric = .dot, .capacity = 8 });

    const held = e.acquire("busy").?;

    // A drop on another thread must not return until the holder releases.
    const T = struct {
        fn run(engine: *Engine, done: *std.atomic.Value(bool)) void {
            _ = engine.drop("busy");
            done.store(true, .release);
        }
    };
    var done = std.atomic.Value(bool).init(false);
    const t = try std.Thread.spawn(.{}, T.run, .{ &e, &done });

    // The collection is already unlinked, so no new acquire can find it, and
    // the dropper is spinning on the reference this test still holds.
    var spins: usize = 0;
    while (spins < 1000 and e.find("busy") != null) : (spins += 1) std.Thread.yield() catch {};
    try testing.expect(!done.load(.acquire));

    held.release();
    t.join();
    try testing.expect(done.load(.acquire));
}

test "concurrent creates of one name: exactly one creates, and the loser touches nothing" {
    // `createCollection` used to check for the name outside the registry
    // lock and then call `create`, which returned the existing collection to
    // the loser, who then wrote its own quantization settings over it while
    // answering `result: true`. `createOrFind` decides under the lock.
    var e = Engine.init(testing.allocator);
    defer e.deinit();

    const threads = 8;
    const Racer = struct {
        fn run(engine: *Engine, dim: usize, created: *std.atomic.Value(usize), winner_dim: *std.atomic.Value(usize), start: *std.atomic.Value(bool)) void {
            while (!start.load(.acquire)) std.atomic.spinLoopHint();
            const made = engine.createOrFind("race", .{ .dim = dim, .metric = .dot, .capacity = 8 }, .{}) catch unreachable;
            if (made.created) {
                _ = created.fetchAdd(1, .monotonic);
                winner_dim.store(dim, .release);
            }
        }
    };
    var created = std.atomic.Value(usize).init(0);
    var winner_dim = std.atomic.Value(usize).init(0);
    var start = std.atomic.Value(bool).init(false);
    var handles: [threads]std.Thread = undefined;
    // Each racer asks for a distinct dimension, so whose config survived is
    // observable.
    for (&handles, 0..) |*h, t| h.* = try std.Thread.spawn(.{}, Racer.run, .{ &e, 4 + t, &created, &winner_dim, &start });
    start.store(true, .release);
    for (handles) |h| h.join();

    try testing.expectEqual(@as(usize, 1), created.load(.acquire));
    try testing.expectEqual(@as(usize, 1), e.collections.items.len);
    try testing.expectEqual(winner_dim.load(.acquire), e.find("race").?.config.dim);

    // The quantization settings travel with the create, under the same lock:
    // set by the caller afterwards they were written without a handle, in
    // the window a concurrent Delete could free the collection.
    const made = try e.createOrFind("quant", .{ .dim = 4, .metric = .dot, .capacity = 8 }, .{ .mode = .scalar, .quantile = 0.5 });
    try testing.expect(made.created);
    try testing.expectEqual(@as(@TypeOf(made.coll.quant_mode), .scalar), made.coll.quant_mode);
    try testing.expectEqual(@as(f32, 0.5), made.coll.quant_quantile);
}

test "ensureIndexBuilding under a storm of polls joins every build it spawns" {
    // Two winners of successive `.absent -> .building` exchanges could
    // interleave their join/store of `build_thread` when a build was fast
    // enough to complete between them, leaving one thread handle overwritten
    // (never joined) and the other joined twice. `build_thread_lock` makes
    // "join what is there, spawn, store" one step. This forces the state
    // back to `.absent` from several threads at once, which is the storm the
    // lock has to survive; nothing here should crash, hang, or leave a build
    // running when the engine is torn down.
    var e = Engine.init(testing.allocator);
    defer e.deinit();
    e.build_mode = .serial;
    const coll = try e.create("storm", .{ .dim = 4, .metric = .dot, .capacity = 64 });
    var v = [_]f32{ 1, 0, 0, 0 };
    for (0..30) |i| _ = try coll.upsert(.{ .num = i }, &v);

    const Poller = struct {
        fn run(engine: *Engine, c: *core.Collection) void {
            for (0..40) |_| {
                c.index_state.store(.absent, .release);
                engine.ensureIndexBuilding(c);
            }
        }
    };
    var handles: [4]std.Thread = undefined;
    for (&handles) |*h| h.* = try std.Thread.spawn(.{}, Poller.run, .{ &e, coll });
    for (handles) |h| h.join();

    // Whatever is still running is the one build the collection knows about;
    // let it finish and confirm the state machine converged.
    if (coll.build_thread) |t| {
        t.join();
        coll.build_thread = null;
    }
    try testing.expect(coll.index_state.load(.acquire) != .building);
    try testing.expect(coll.graph != null);
}

test "fullScanPreferred is Qdrant's rule: count below threshold_kb * 1024 / (dim * elem)" {
    // d=256 fp32 is exactly 1 KB per vector, so `full_scan_threshold = 2`
    // means "plain scan below 2 points".
    var c = try core.Collection.init(testing.allocator, "t", .{
        .dim = 256,
        .metric = .dot,
        .capacity = 8,
        .hnsw_full_scan_threshold_kb = 2,
    });
    defer c.deinit();
    var v: [256]f32 = @splat(1.0);
    try testing.expect(fullScanPreferred(&c)); // 0 < 2
    _ = try c.upsert(.{ .num = 0 }, &v);
    try testing.expect(fullScanPreferred(&c)); // 1 < 2
    _ = try c.upsert(.{ .num = 1 }, &v);
    try testing.expect(!fullScanPreferred(&c)); // 2 < 2 is false
    // Live points, as `available_vector_count` is.
    try testing.expect(c.delete(.{ .num = 1 }));
    try testing.expect(fullScanPreferred(&c));

    // The default: 10 000 KB, so a 1000-point d=128 fp32 collection (512 KB)
    // is scanned, and threshold 0 never scans.
    var d = try core.Collection.init(testing.allocator, "d", .{ .dim = 128, .metric = .dot, .capacity = 8 });
    defer d.deinit();
    try testing.expectEqual(@as(usize, 10_000), d.config.hnsw_full_scan_threshold_kb);
    try testing.expect(fullScanPreferred(&d));
    d.config.hnsw_full_scan_threshold_kb = 0;
    try testing.expect(!fullScanPreferred(&d));
    // A f16 vector is half the bytes, so the same threshold covers twice the
    // points: at threshold 1 KB, d=256 f16 (512 B) scans below 2 points.
    var h = try core.Collection.init(testing.allocator, "h", .{
        .dim = 256,
        .metric = .dot,
        .datatype = .float16,
        .capacity = 8,
        .hnsw_full_scan_threshold_kb = 1,
    });
    defer h.deinit();
    _ = try h.upsert(.{ .num = 0 }, &v);
    try testing.expect(fullScanPreferred(&h));
    _ = try h.upsert(.{ .num = 1 }, &v);
    try testing.expect(!fullScanPreferred(&h));
    // Saturates rather than overflows on an absurd client value.
    h.config.hnsw_full_scan_threshold_kb = std.math.maxInt(usize);
    try testing.expect(fullScanPreferred(&h));
}
