//! §3's collection lifecycle: create, delete, exists, status, info.
//!
//! Split out of `handlers.zig`. That file held three RPC
//! families and their shared plumbing in 2,300 lines; this is the one that
//! touches no search path at all -- no graph, no scratch, no quantized store
//! at query time -- so it separates without carrying the query machinery with
//! it. `upsert`, `scroll` and `queryBatch` stay where the `Workspace` they
//! spend lives.
//!
//! The quantization *config* parsing lives here rather than beside the
//! quantized search: it is a wire-format concern -- what a client may ask for
//! and how a malformed request is refused -- and its output is a stored
//! collection setting, not a query parameter.

const std = @import("std");
const core = @import("../core/core.zig");
const msg = @import("../proto/messages.zig");
const wire = @import("../proto/wire.zig");
const server = @import("../net/server.zig");
const quant = @import("../quant/quant.zig");
const storage = core.storage;
const dist = @import("../dist/dist.zig");
const Status = handlers.Status; // grpc.Status; the collection one is spelled out where used
const handlers = @import("handlers.zig");

const Context = handlers.Context;
const err = handlers.err;
const ok = handlers.ok;
const decodeErr = handlers.decodeErr;

pub fn createCollection(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);

    var name: []const u8 = &.{};
    var params: ?msg.VectorParams = null;
    var saw_named_vectors = false;
    // The collection-level `hnsw_config` (field 4) and `quantization_config`
    // (field 14). bfb puts *both* here (`from_args.rs`: `.hnsw_config(...)`,
    // `.quantization_config(...)` on the `CreateCollectionBuilder`), and so
    // does the conformance harness; the `VectorParams`-nested copies are what
    // a per-vector override looks like. An earlier version read only the
    // nested ones, so `--quantization scalar` built an fp32 collection and
    // `--hnsw-m 32` built at M=16, and nothing on the wire said so: the create
    // succeeded, the status went Green, and the numbers were for a different
    // configuration than the one on the results row.
    var top_hnsw: ?msg.HnswConfigDiff = null;
    var top_quant_raw: ?[]const u8 = null;

    while (!r.atEnd()) {
        const t = r.tag() catch |e| return decodeErr(req, e);
        switch (t.field) {
            1 => name = r.bytes() catch |e| return decodeErr(req, e),
            // `optional HnswConfigDiff hnsw_config = 4` (VERIFIED,
            // collections.proto).
            4 => {
                var sub = r.nested() catch |e| return decodeErr(req, e);
                top_hnsw = msg.HnswConfigDiff.decode(&sub) catch |e| return decodeErr(req, e);
            },
            // `optional OptimizersConfigDiff optimizers_config = 6`. Accepted
            // and inert, deliberately: every field in it
            // (`default_segment_number`, `indexing_threshold`,
            // `memmap_threshold`, `max_segment_size`, `flush_interval_sec`,
            // ...) tunes Qdrant's segment optimiser, and strawmann has exactly
            // one segment and no optimiser (§2's declared gap: "single segment
            // vs Qdrant's multi-segment"). None of them changes what a query
            // returns, so none of them is a §1 silent degradation; refusing
            // them would refuse every bfb run, which sends the message
            // unconditionally. Nothing is recorded because there is nothing
            // to apply it to.
            6 => r.skip(t.wire_type) catch |e| return decodeErr(req, e),
            // VERIFIED against the generated qdrant-client: `vectors_config` is
            // field **10** on CreateCollection, not 6. With the wrong number
            // the field is skipped as unknown and every create fails with
            // "vectors_config.params is required", which is at least loud.
            10 => {
                var sub = r.nested() catch |e| return decodeErr(req, e);
                // VectorsConfig { oneof { VectorParams params = 1;
                //                         VectorParamsMap params_map = 2; } }
                while (!sub.atEnd()) {
                    const vt = sub.tag() catch |e| return decodeErr(req, e);
                    switch (vt.field) {
                        1 => {
                            var p = sub.nested() catch |e| return decodeErr(req, e);
                            params = msg.VectorParams.decode(&p) catch |e| return decodeErr(req, e);
                        },
                        2 => {
                            // §3 supports named spaces, but the single-space
                            // case is the optimised one and the only one M1
                            // implements. Failing loudly here is §1's rule.
                            saw_named_vectors = true;
                            sub.skip(vt.wire_type) catch |e| return decodeErr(req, e);
                        },
                        else => sub.skip(vt.wire_type) catch |e| return decodeErr(req, e),
                    }
                }
            },
            // §2's deliberate semantic gaps: accepted and ignored, but only at
            // value 1. A sharded or replicated request must not silently run as
            // a single shard. shard_number=7, replication_factor=11.
            7, 11 => {
                const v = r.varint() catch |e| return decodeErr(req, e);
                if (v > 1) return err(req, .unimplemented, "sharding and replication (§1 non-goal); only value 1 is supported");
            },
            // `optional QuantizationConfig quantization_config = 14`
            // (VERIFIED, collections.proto).
            14 => top_quant_raw = r.bytes() catch |e| return decodeErr(req, e),
            else => r.skip(t.wire_type) catch |e| return decodeErr(req, e),
        }
    }

    if (saw_named_vectors) return err(req, .unimplemented, "named vector spaces (phase 2)");
    // Qdrant refuses it (`collection_name` is `#[validate(length(min = 1))]`);
    // here it would create a collection no request can address by name.
    if (name.len == 0) return err(req, .invalid_argument, "collection_name must not be empty");
    var p = params orelse return err(req, .invalid_argument, "vectors_config.params is required");

    // Merge the collection-level config under the per-vector one. Qdrant's
    // precedence (`VectorParams.hnsw_config`: "Configuration of the HNSW
    // index for this vector; overrides the collection config") is per-vector
    // over collection, so a nested value wins and the top level fills in
    // whatever the nested one left unset. The graph placement is checked
    // separately below because its rule differs between the two levels.
    if (top_hnsw) |h| {
        if (p.hnsw_m == null) p.hnsw_m = h.m;
        if (p.hnsw_ef_construct == null) p.hnsw_ef_construct = h.ef_construct;
        if (p.hnsw_full_scan_threshold == null) p.hnsw_full_scan_threshold = h.full_scan_threshold;
        // The top-level graph placement. bfb *always* sends
        // `hnsw_config.on_disk = <--on-disk-index or false>` here
        // (`from_args.rs`: `.on_disk(args.on_disk_index.unwrap_or_default())`),
        // so an explicit `false` is the default run and cannot be refused
        // without refusing every run. Only a request that actually asks for
        // something other than a pinned graph, `on_disk: true` or any
        // `memory`, is refused by name; the nested `VectorParams.hnsw_config`
        // keeps its stricter rule because nothing sends `false` there by
        // default.
        if ((h.on_disk orelse false) or h.memory != 0) {
            return err(req, .unimplemented, "hnsw_config.on_disk / hnsw_config.memory: the HNSW graph is always pinned in RAM (§6.5); only vector storage honours a placement");
        }
    }
    if (p.quantization_raw == null) p.quantization_raw = top_quant_raw;
    if (p.size == 0) return err(req, .invalid_argument, "vector size must be non-zero");
    if (p.size > ctx.engine.max_dim) {
        return err(req, .invalid_argument, "vector size exceeds this server's --max-dim");
    }
    // §5.5's residency, as Qdrant's `VectorParams.memory` (with the deprecated
    // `on_disk` still honoured behind it). This used to be a flat refusal
    // naming §1, which conflated two different things: §1's non-goal is
    // *disk-resident (larger-than-RAM) operation*, an engine that keeps working
    // when the data does not fit. Choosing where bytes that do fit are held is
    // not that, it is §5.5's subject, and refusing it meant a Qdrant run
    // configured the way Qdrant defaults to configure itself had no strawmANN
    // arm to compare against at all.
    //
    // Unstated stays `pinned`, which is *not* Qdrant's default (`Cached`). See
    // `storage.Placement.resolve` for why that divergence is kept rather than
    // fixed, and `README`'s placement section for how it is reported.
    // A `memory` value we do not recognise is refused rather than falling
    // through to the default. Falling through would answer a request for a
    // placement this build has never heard of by quietly serving the one it
    // likes best, and the client would have no way to tell.
    if (p.memory != 0 and storage.Placement.fromProto(p.memory) == null) {
        return err(req, .invalid_argument, "unknown vectors_config.memory value; expected Cold, Cached or Pinned");
    }
    const placement = storage.Placement.resolve(p.memory, p.on_disk) orelse
        ctx.engine.default_placement;
    if (placement.isMapped() and ctx.engine.data_dir == null) {
        return err(req, .failed_precondition, "cached and cold placement need a file to map; start the server with --data-dir <path>");
    }

    // §1's rule, applied to the one placement knob this engine does not
    // implement: the graph is always pinned. Ignoring the field would publish
    // an on-disk-graph measurement taken on an in-memory graph, which is the
    // silent degradation §1 forbids in the same sentence that lists the
    // non-goals.
    if (p.hnsw_placement_requested) {
        return err(req, .unimplemented, "hnsw_config.memory / hnsw_config.on_disk: the HNSW graph is always pinned in RAM (§6.5); only vector storage honours a placement");
    }

    const metric = dist.Metric.fromProto(p.distance) orelse
        return err(req, .invalid_argument, "unknown or unset distance metric");

    // §5.4's working-set argument, as a storage type rather than a codebook.
    // The semantics are Qdrant's and were read from their source rather than
    // inferred; `dist/datatype.zig` carries the table and the two places it
    // surprises, both of which are about uint8.
    const datatype = dist.Datatype.fromProto(p.datatype) orelse
        return err(req, .invalid_argument, "unknown vector datatype");
    if (datatype != .float32 and p.size > core.collection.max_converted_dim) {
        return err(req, .invalid_argument, "float16 and uint8 storage are limited to 16384 dimensions on this server");
    }

    // Creating over an existing name is ALREADY_EXISTS, as it is for Qdrant
    // (`Checker::validate_collection_not_exists` -> `StorageError::
    // AlreadyExists` -> `tonic::Code::AlreadyExists`, VERIFIED in
    // `lib/storage/src/content_manager/collections_ops.rs` and
    // `conversions.rs`). An earlier version silently dropped and recreated,
    // which is a *different* answer to the same request: bfb's setup issues
    // Delete then Create and `--create-if-missing` checks `CollectionExists`
    // first, so neither path ever sees this status, but a client that
    // creates twice would have had its first collection's data thrown away
    // with a `result: true`.
    //
    // Checked early so a request that would fail anyway does not pay for
    // the validation below, but *decided* by `createOrFind` under the
    // registry lock: two concurrent creates both passed this check, and the
    // loser was handed the winner's collection and wrote its own
    // quantization settings over it.
    if (ctx.engine.find(name) != null) {
        return err(req, .already_exists, "collection already exists");
    }

    // §6.7: the quantization mode arrives with the collection. `turbo*` is
    // rejected by name rather than approximated, so nobody can accidentally
    // compare against an encoding we never implemented.
    var qspec: QuantSpec = .{};
    if (p.quantization_raw) |raw| {
        switch (parseQuantizationConfig(raw)) {
            .spec => |sp| qspec = sp,
            .rejected => |rj| return err(req, rj.status, rj.why),
        }
    }

    // §6.5 builds the graph with `M` neighbours per upper level and `2M` on
    // level 0, and sizes the CSR arena from an expected-level distribution that
    // assumes M is comfortably above 1. At M=1 the level multiplier is
    // `1/ln(1) = inf` and every node is assigned the maximum level, blowing the
    // arena; at M=2 the expected upper-level usage exceeds what is allocated.
    // Both are reachable from a client-supplied `hnsw_config`, and the only
    // guard downstream is an assert that vanishes in ReleaseFast.
    if (p.hnsw_m) |m| {
        if (m < 4 or m > 512) {
            return err(req, .invalid_argument, "hnsw_config.m must be between 4 and 512");
        }
    }
    if (p.hnsw_ef_construct) |e| {
        if (e == 0 or e > 65536) {
            return err(req, .invalid_argument, "hnsw_config.ef_construct must be between 1 and 65536");
        }
    }

    const cfg = core.Config{
        .dim = @intCast(p.size),
        .metric = metric,
        .datatype = datatype,
        .capacity = ctx.engine.default_capacity,
        .huge_pages = ctx.engine.huge_pages,
        .placement = placement,
        .dir = ctx.engine.data_dir,
        .hnsw_m = if (p.hnsw_m) |m| @intCast(m) else 16,
        .hnsw_ef_construct = if (p.hnsw_ef_construct) |e| @intCast(e) else 100,
        // Absent means Qdrant's default; a u64 on the wire, saturated rather
        // than trapped on a 32-bit host.
        .hnsw_full_scan_threshold_kb = if (p.hnsw_full_scan_threshold) |t|
            @intCast(@min(t, @as(u64, std.math.maxInt(usize))))
        else
            core.collection.default_full_scan_threshold_kb,
    };
    // The placement failures are separated from a plain allocation failure
    // because they are the client's to fix and the message says how. A mapped
    // placement that fell back to pinned would answer the request and measure
    // something else.
    const made = ctx.engine.createOrFind(name, cfg, qspec) catch |e| switch (e) {
        error.PlacementNeedsDirectory => return err(req, .failed_precondition, "cached and cold placement need a file to map; start the server with --data-dir <path>"),
        error.PlacementUnavailable => return err(req, .internal, "could not create or map the arena file for this placement; check --data-dir is writable and has room for the preallocated arena"),
        else => return err(req, .internal, "collection allocation failed"),
    };
    if (!made.created) return err(req, .already_exists, "collection already exists");
    return writeBoolResult(req, out, true);
}

/// `CollectionOperationResponse { bool result = 1; double time = 2; }`
fn writeBoolResult(req: *const server.Request, out: *server.ResponseBuf, value: bool) server.Completion {
    var w = wire.Writer.init(out.available());
    w.writeBoolField(1, value) catch return err(req, .internal, "response buffer overflow");
    return ok(req, out.commit(w.pos));
}

pub fn deleteCollection(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    var name: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = r.tag() catch |e| return decodeErr(req, e);
        switch (t.field) {
            1 => name = r.bytes() catch |e| return decodeErr(req, e),
            else => r.skip(t.wire_type) catch |e| return decodeErr(req, e),
        }
    }
    // Deleting a nonexistent collection is not an error for Qdrant, but it
    // is not a `true` either: `TableOfContent::delete_collection` returns
    // `Ok(false)` when nothing was loaded under that name (VERIFIED,
    // `lib/storage/src/content_manager/toc/collection_meta_ops.rs`), and
    // `result` carries that straight to the client. bfb prints the outcome
    // and moves on either way, so this changes only what it prints, but a
    // client that keys off `result` was being told it deleted something.
    const existed = ctx.engine.drop(name);
    return writeBoolResult(req, out, existed);
}

/// `CollectionExistsResponse { CollectionExists result = 1; double time = 2; }`
/// `CollectionExists { bool exists = 1; }`
pub fn collectionExists(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    var name: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = r.tag() catch |e| return decodeErr(req, e);
        switch (t.field) {
            1 => name = r.bytes() catch |e| return decodeErr(req, e),
            else => r.skip(t.wire_type) catch |e| return decodeErr(req, e),
        }
    }
    const exists = ctx.engine.find(name) != null;

    var w = wire.Writer.init(out.available());
    const n = w.beginNested(1, 2) catch return err(req, .internal, "response buffer overflow");
    w.writeBoolField(1, exists) catch return err(req, .internal, "response buffer overflow");
    w.endNested(n) catch return err(req, .internal, "response buffer overflow");
    return ok(req, out.commit(w.pos));
}

/// The collection's §2 status as the wire enum.
///
/// Exhaustive on our side and named on qdrant's, so the two cannot drift: a new
/// internal status is a compile error here, and the values come from
/// `msg.CollectionStatus` rather than being written out again as integers.
fn collectionStatus(s: core.collection.Status) msg.CollectionStatus {
    return switch (s) {
        .green => .green,
        .yellow => .yellow,
        .red => .red,
    };
}

/// §2: "**`optimizer_status` must be present and `Ok`** in `CollectionInfo`,
/// and `points_count` / `indexed_vectors_count` should be truthful, several
/// bfb code paths and human eyeballs read them."
///
/// `CollectionInfo { CollectionStatus status = 1; OptimizerStatus optimizer_status = 2;
///                   uint64 vectors_count = 3; uint64 segments_count = 4;
///                   CollectionConfig config = 5; map payload_schema = 6;
///                   uint64 points_count = 7; uint64 indexed_vectors_count = 8; }`
fn writePayloadSchema(w: *wire.Writer, coll: *core.Collection) wire.Writer.Error!void {
    coll.payload.lockFields();
    defer coll.payload.unlockFields();
    for (coll.payload.fields.items) |f| {
        const entry = try w.beginNested(8, 2);
        try w.writeStringField(1, f.name);
        const info = try w.beginNested(2, 2);
        try w.writeVarintField(1, f.kind.schemaType());
        try w.writeVarintFieldAlways(3, f.points); // `optional uint64 points`
        try w.endNested(info);
        try w.endNested(entry);
    }
}

pub fn getCollectionInfo(ctx: *Context, req: *const server.Request, body: []const u8, out: *server.ResponseBuf) server.Completion {
    var r = wire.Reader.init(body);
    var name: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = r.tag() catch |e| return decodeErr(req, e);
        switch (t.field) {
            1 => name = r.bytes() catch |e| return decodeErr(req, e),
            else => r.skip(t.wire_type) catch |e| return decodeErr(req, e),
        }
    }
    const held = ctx.engine.acquire(name) orelse
        return err(req, .not_found, "collection not found");
    defer held.release();
    const coll = held.coll;

    // §2: "`wait_index` drives indexing semantics. After upload, bfb polls
    // `collection_info` once per second and requires `status == Green` three
    // consecutive times. This is a gift: it means we do not need
    // concurrent-with-ingest index construction. We can accept upserts into a
    // flat unindexed buffer, report `Yellow`, run a fully parallel bulk HNSW
    // build, then flip to `Green`."
    //
    // The build runs on its own thread so this RPC returns immediately with
    // Yellow. Reporting Green before the graph exists would make bfb start
    // searching an unindexed collection, and §2 is explicit that if
    // `--skip-wait-index` is used "we must report `Green` only when actually
    // indexed or the comparison is meaningless."
    ctx.engine.ensureIndexBuilding(coll);

    var w = wire.Writer.init(out.available());
    const outer = w.beginNested(1, 3) catch return err(req, .internal, "response buffer overflow");

    w.writeVarintField(1, @intFromEnum(collectionStatus(coll.status()))) catch
        return err(req, .internal, "buf");
    // OptimizerStatus is a message with `bool ok = 1; string error = 2;`.
    // §2 requires it present and Ok. Note `ok = true` is written explicitly
    // rather than elided, a client reading an absent field sees `false`.
    {
        const os = w.beginNested(2, 2) catch return err(req, .internal, "buf");
        w.writeBoolField(1, true) catch return err(req, .internal, "buf");
        w.endNested(os) catch return err(req, .internal, "buf");
    }
    // §2's semantic gap: "single segment vs Qdrant's multi-segment
    // (`--segments 1` on the Qdrant side for the first comparisons)".
    w.writeVarintField(4, 1) catch return err(req, .internal, "buf");
    // VERIFIED: points_count is 9 and indexed_vectors_count is 10 in qdrant
    // 1.19. Writing them at 3 and 7/8 (the older layout) makes bfb's poll loop
    // read zero for both while status says Green, §2 calls truthful counts a
    // requirement precisely because several code paths read them.
    // `CollectionConfig config = 7` (VERIFIED: 7, not the 5 the older layout
    // used). The *effective* configuration, so a client can read back what the
    // create actually built rather than trusting that its request was
    // honoured. This is how the create-path regression tests, and a human
    // with `qdrant-client`, check that a top-level `hnsw_config` or
    // `quantization_config` took effect: without it the only evidence was the
    // recall number, which is exactly the kind of "measured against a
    // configuration other than the one written on the results row" that §7
    // is built to prevent.
    writeCollectionConfig(&w, coll) catch return err(req, .internal, "response buffer overflow");
    // `map<string, PayloadSchemaInfo> payload_schema = 8` (VERIFIED, 1.19):
    // one entry per field index, `PayloadSchemaInfo { PayloadSchemaType
    // data_type = 1; PayloadIndexParams params = 2; uint64 points = 3; }`.
    // This is what the harness reads back (`collection-info`'s
    // `payload_indexes`, docs/workloads.md W12 point 1) to decide whether an
    // engine *says* it built the index a filtered row depends on.
    writePayloadSchema(&w, coll) catch return err(req, .internal, "response buffer overflow");
    // Both `optional uint64` (VERIFIED, collections.proto): Qdrant's prost
    // always emits `Some(0)`, and the zero-eliding writer made an empty or
    // unindexed collection report them *absent*, which a client reads as
    // None, not 0. §2 calls these counts a requirement because bfb's poll
    // loop reads them.
    w.writeVarintFieldAlways(9, coll.count()) catch return err(req, .internal, "buf");
    w.writeVarintFieldAlways(10, coll.indexed_count) catch return err(req, .internal, "buf");

    w.endNested(outer) catch return err(req, .internal, "response buffer overflow");
    return ok(req, out.commit(w.pos));
}

/// `CollectionConfig { CollectionParams params = 1; HnswConfigDiff hnsw_config = 2;
///                     OptimizersConfigDiff optimizer_config = 3; WalConfigDiff wal_config = 4;
///                     optional QuantizationConfig quantization_config = 5; ... }`
/// `CollectionParams { uint32 shard_number = 3; optional VectorsConfig vectors_config = 5;
///                     optional uint32 replication_factor = 6; ... }`
///
/// Field numbers VERIFIED against qdrant-client 1.19's `collections.proto`:
/// `vectors_config` is **5** inside `CollectionParams` (1 and 2 are reserved)
/// and `quantization_config` is **5** inside `CollectionConfig`, neither of
/// which is where a first guess puts them. `optimizer_config` and `wal_config`
/// are omitted: they describe machinery strawmann does not have, and an empty
/// message would read as "configured to all defaults", which is a claim.
fn writeCollectionConfig(w: *wire.Writer, coll: *const core.Collection) wire.Writer.Error!void {
    const cfg = try w.beginNested(7, 2);
    {
        const params = try w.beginNested(1, 2);
        // §2's declared gaps, reported as what they are: one shard, one
        // replica.
        try w.writeVarintField(3, 1);
        {
            const vc = try w.beginNested(5, 2);
            const vp = try w.beginNested(1, 2); // VectorsConfig.params
            try w.writeVarintField(1, coll.config.dim);
            try w.writeVarintField(2, @intCast(coll.config.metric.toProto()));
            try w.writeVarintField(6, @intCast(coll.config.datatype.toProto()));
            // The placement the arena actually has, in the field the client
            // used to ask for it (`Memory memory = 8`; the deprecated
            // `on_disk` is not echoed).
            try w.writeVarintField(8, @intCast(coll.config.placement.toProto()));
            try w.endNested(vp);
            try w.endNested(vc);
        }
        try w.writeVarintField(6, 1);
        try w.endNested(params);
    }
    {
        // What the graph is (or will be) built with. §6.5's `m` and
        // `ef_construct`, which are the two knobs bfb sets and the two that
        // change recall.
        const hc = try w.beginNested(2, 2);
        try w.writeVarintField(1, coll.config.hnsw_m);
        try w.writeVarintField(2, coll.config.hnsw_ef_construct);
        // `full_scan_threshold = 3`: the third knob, and the one that decides
        // whether the graph is consulted at all (`fullScanPreferred`).
        // `optional uint64` (VERIFIED, collections.proto): explicit presence,
        // so a threshold of 0 is written rather than elided.
        try w.writeVarintFieldAlways(3, coll.config.hnsw_full_scan_threshold_kb);
        try w.endNested(hc);
    }
    // The quantization the collection is *configured* for. `quant_mode`
    // rather than `quant`: Qdrant reports config, not build state, and a
    // collection that asked for scalar and has not been indexed yet is still
    // a scalar collection.
    switch (coll.quant_mode) {
        .none => {},
        .scalar => {
            const qc = try w.beginNested(5, 2);
            const sq = try w.beginNested(1, 2);
            try w.writeVarintField(1, 1); // QuantizationType::Int8
            try w.writeFloatField(2, coll.quant_quantile);
            try w.writeBoolField(3, true); // always_ram: §6.7, always
            try w.endNested(sq);
            try w.endNested(qc);
        },
        .product => |cr| {
            const qc = try w.beginNested(5, 2);
            const pq = try w.beginNested(2, 2);
            try w.writeVarintField(1, cr.toProto());
            try w.writeBoolField(2, true);
            try w.endNested(pq);
            try w.endNested(qc);
        },
        .binary => {
            const qc = try w.beginNested(5, 2);
            const bq = try w.beginNested(3, 2);
            try w.writeBoolField(1, true);
            try w.endNested(bq);
            try w.endNested(qc);
        },
    }
    try w.endNested(cfg);
}

/// What a `QuantizationConfig` asked for, in the terms the collection stores.
///
/// `quantile` matters only for `.scalar`; it rides along for the other modes
/// at its default so the struct has one shape.
pub const QuantSpec = struct {
    mode: quant.Mode = .none,
    /// `ScalarQuantization.quantile`. Absent means Qdrant's default, which is
    /// **not** bfb's: Qdrant's `quantile: Option<f32>` is documented "If not
    /// set - use the whole range of values" and `EncodedVectorsU8::encode`
    /// falls through to plain min/max when it is `None` (VERIFIED,
    /// `lib/quantization/src/encoded_vectors_u8.rs`). bfb and the conformance
    /// harness both send 0.99 explicitly, so the default only decides what
    /// a bare `ScalarQuantization { type: Int8 }` builds, and it should build
    /// what Qdrant builds.
    quantile: f32 = 1.0,
};

pub const QuantParse = union(enum) {
    spec: QuantSpec,
    rejected: struct { status: Status, why: []const u8 },
};

/// Parse `QuantizationConfig { oneof { ScalarQuantization scalar = 1;
///                                      ProductQuantization product = 2;
///                                      BinaryQuantization binary = 3; } }`.
///
/// §6.7: "Match bfb's flags, minus the Qdrant-proprietary `turbo*` variants
/// (return a clear error; document the exclusion so nobody accidentally
/// compares against them)."
///
/// Every sub-message field is read rather than skipped, because each one that
/// is skipped is a knob a client can turn without the engine noticing (§1's
/// silent degradation). `always_ram` and `memory` are the two exceptions,
/// accepted and inert by §6.7's own rule ("`always_ram` is always true for
/// us"): the codes are always in RAM, so a request to keep them there is
/// already honoured, and a request to page them is refused nowhere because
/// nothing sends it.
fn parseQuantizationConfig(raw: []const u8) QuantParse {
    var r = wire.Reader.init(raw);
    while (!r.atEnd()) {
        const t = r.tag() catch return malformedQuant("malformed quantization_config");
        switch (t.field) {
            1 => {
                // ScalarQuantization { QuantizationType type = 1;
                //                      optional float quantile = 2;
                //                      optional bool always_ram = 3;
                //                      optional Memory memory = 4; }
                var sub = r.nested() catch return malformedQuant("malformed scalar quantization");
                var spec = QuantSpec{ .mode = .scalar };
                while (!sub.atEnd()) {
                    const st = sub.tag() catch return malformedQuant("malformed scalar quantization");
                    switch (st.field) {
                        // `QuantizationType { UnknownQuantization = 0; Int8 = 1; }`:
                        // Int8 is the only member, so nothing to branch on.
                        1 => _ = sub.varint() catch return malformedQuant("malformed scalar quantization type"),
                        2 => {
                            const q = sub.float() catch return malformedQuant("malformed scalar quantization quantile");
                            // Qdrant's `#[validate(range(min = 0.5, max = 1.0))]`
                            // (VERIFIED, `lib/segment/src/types.rs`), and the
                            // same status it answers with. Below 0.5 the two
                            // tails would overlap; above 1 is not a quantile.
                            if (!(q >= 0.5 and q <= 1.0)) {
                                return .{ .rejected = .{ .status = .invalid_argument, .why = "quantization_config.scalar.quantile must be between 0.5 and 1.0" } };
                            }
                            spec.quantile = q;
                        },
                        // 3 always_ram, 4 memory: accepted, see above.
                        else => sub.skip(st.wire_type) catch return malformedQuant("malformed scalar quantization"),
                    }
                }
                return .{ .spec = spec };
            },
            2 => {
                // ProductQuantization { CompressionRatio compression = 1;
                //                       optional bool always_ram = 2;
                //                       optional Memory memory = 3; }
                var sub = r.nested() catch return malformedQuant("malformed product quantization");
                // qdrant's default when the field is absent.
                var ratio: quant.CompressionRatio = .x16;
                while (!sub.atEnd()) {
                    const st = sub.tag() catch return malformedQuant("malformed product quantization");
                    if (st.field == 1) {
                        const v = sub.varint() catch return malformedQuant("malformed compression ratio");
                        ratio = quant.CompressionRatio.fromProto(v) orelse
                            return .{ .rejected = .{ .status = .invalid_argument, .why = "unknown product quantization compression ratio" } };
                    } else sub.skip(st.wire_type) catch return malformedQuant("malformed product quantization");
                }
                return .{ .spec = .{ .mode = .{ .product = ratio } } };
            },
            3 => {
                // BinaryQuantization { optional bool always_ram = 1;
                //                      optional BinaryQuantizationEncoding encoding = 2;
                //                      optional BinaryQuantizationQueryEncoding query_encoding = 3;
                //                      optional Memory memory = 4; }
                //
                // §6.7's binary is 1 bit/dim, sign-based. Qdrant 1.19 also
                // offers `TwoBits` and `OneAndHalfBits` storage encodings and
                // `Scalar4Bits`/`Scalar8Bits` asymmetric query encodings, and
                // bfb reaches them (`--quantization binary-2bit`,
                // `binary-1.5bit`). Skipping the fields would answer those
                // requests with the 1-bit encoding and publish a comparison
                // against a store that was never built.
                var sub = r.nested() catch return malformedQuant("malformed binary quantization");
                while (!sub.atEnd()) {
                    const st = sub.tag() catch return malformedQuant("malformed binary quantization");
                    switch (st.field) {
                        2 => {
                            // `BinaryQuantizationEncoding { OneBit = 0; TwoBits = 1; OneAndHalfBits = 2; }`
                            const v = sub.varint() catch return malformedQuant("malformed binary quantization encoding");
                            if (v != 0) return .{ .rejected = .{ .status = .unimplemented, .why = "quantization_config.binary.encoding: only OneBit is implemented (§6.7: 1 bit/dim, sign-based); TwoBits and OneAndHalfBits are not" } };
                        },
                        3 => {
                            // `BinaryQuantizationQueryEncoding { oneof variant { Setting setting = 4; } }`
                            // `Setting { Default = 0; Binary = 1; Scalar4Bits = 2; Scalar8Bits = 3; }`
                            // Default and Binary both mean "the query is
                            // encoded like the rows", which is what §6.7's
                            // XOR + popcount does; the scalar settings are an
                            // asymmetric path this engine does not have.
                            var qe = sub.nested() catch return malformedQuant("malformed binary quantization query_encoding");
                            while (!qe.atEnd()) {
                                const qt = qe.tag() catch return malformedQuant("malformed binary quantization query_encoding");
                                if (qt.field == 4) {
                                    const v = qe.varint() catch return malformedQuant("malformed binary quantization query_encoding");
                                    if (v > 1) return .{ .rejected = .{ .status = .unimplemented, .why = "quantization_config.binary.query_encoding: only Default/Binary is implemented; Scalar4Bits and Scalar8Bits asymmetric query encodings are not" } };
                                } else qe.skip(qt.wire_type) catch return malformedQuant("malformed binary quantization query_encoding");
                            }
                        },
                        // 1 always_ram, 4 memory: accepted, see above.
                        else => sub.skip(st.wire_type) catch return malformedQuant("malformed binary quantization"),
                    }
                }
                return .{ .spec = .{ .mode = .binary } };
            },
            // §6.7: "minus the Qdrant-proprietary `turbo*` variants (return a
            // clear error; document the exclusion so nobody accidentally
            // compares against them)". qdrant 1.19 puts TurboQuantization at
            // field 4 of the oneof, so it is rejectable by number rather than
            // only by the `--quantization turbo*` string.
            4 => return .{ .rejected = .{ .status = .unimplemented, .why = "Qdrant-proprietary turbo quantization is deliberately not implemented (spec §6.7)" } },
            else => r.skip(t.wire_type) catch return malformedQuant("malformed quantization_config"),
        }
    }
    return .{ .spec = .{} };
}

fn malformedQuant(why: []const u8) QuantParse {
    return .{ .rejected = .{ .status = .invalid_argument, .why = why } };
}

// =========================================================================
// qdrant.Points/Upsert
// =========================================================================
