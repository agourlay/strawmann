//! §6.4, persistence.
//!
//! Split out of `collection.zig` for the reason given in `quantized_search.zig`:
//! one 2200-line file held four concerns that share only a type. Save/load is
//! the most self-contained of them, it touches `storage`, which nothing else
//! in the collection path does.

const std = @import("std");
const testing = std.testing;

const collection = @import("collection.zig");
const Collection = collection.Collection;
const Status = collection.Status;
const storage = @import("storage.zig");
const ids = @import("ids.zig");
const heap = @import("../index/heap.zig");
const hnsw = @import("../index/hnsw.zig");
const dist = @import("../dist/dist.zig");
const Candidate = heap.Candidate;
const ExternalId = ids.ExternalId;

/// The shared test fixture lives with the type it builds. Duplicating it here
/// would mean two constructors drifting apart, which is exactly the split this
/// refactor was meant to avoid creating.
const makeCollection = collection.makeCollection;
const buildIndex = collection.buildIndex;
const quantize = collection.quantize;
const scroll_mod = @import("scroll.zig");
const payload_mod = @import("payload.zig");
const wire = @import("../proto/wire.zig");
const scroll = scroll_mod.scroll;
const search = collection.search;
const invalidateIndex = collection.invalidateIndex;
const idLess = scroll_mod.idLess;

// =========================================================================
// §6.4, persistence
// =========================================================================

/// §6.4: "Recovery is `open` + `mmap`/`pread`. There is no parse step, no WAL
/// replay, no index rebuild. Startup time for a 3 GB collection should be
/// bounded by sequential read bandwidth."
///
/// So `load` reads the arenas straight into memory and reconstructs only what
/// §6.4 marks rebuildable, the open-addressed id map, which holds no
/// information `ids.bin` does not.
pub const Persistence = struct {
    dir: []const u8,

    /// `error.PathTooLong` rather than `unreachable`, because `dir` is
    /// `--data-dir` and its length is the operator's, not this file's. The
    /// bound is 512 bytes and Linux allows 4096, so a legal path overflows it;
    /// asserting made that illegal behaviour in ReleaseFast, which is the mode
    /// §9 quotes numbers from. `ensureDir` five lines down already returns this
    /// error for the same buffer, and `Collection.init` returns one for the
    /// same `bufPrint` — this was the only one of the three that asserted.
    fn path(buf: []u8, dir: []const u8, name: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, name }) catch
            error.PathTooLong;
    }

    /// Create the directory if absent.
    fn ensureDir(dir: []const u8) !void {
        var z: [512]u8 = undefined;
        if (dir.len + 1 > z.len) return error.PathTooLong;
        @memcpy(z[0..dir.len], dir);
        z[dir.len] = 0;
        const rc = std.os.linux.mkdir(@ptrCast(&z), 0o755);
        switch (std.os.linux.errno(rc)) {
            .SUCCESS, .EXIST => {},
            else => return error.MkdirFailed,
        }
    }
};

/// Bytes per record in `ids.bin`: a tag byte followed by a 16-byte
/// little-endian payload (`num` zero-extended). An explicit encoding rather
/// than the bytes of `ExternalId` itself, whose layout as a Zig tagged union
/// (tag placement, padding) is the compiler's to choose and not part of any
/// format.
const id_record_size = 1 + @sizeOf(u128);

fn encodeId(id: ids.ExternalId, out: *[id_record_size]u8) void {
    switch (id) {
        .num => |v| {
            out[0] = 0;
            std.mem.writeInt(u128, out[1..], v, .little);
        },
        .uuid => |v| {
            out[0] = 1;
            std.mem.writeInt(u128, out[1..], v, .little);
        },
    }
}

fn decodeId(in: *const [id_record_size]u8) !ids.ExternalId {
    const payload = std.mem.readInt(u128, in[1..], .little);
    return switch (in[0]) {
        0 => .{ .num = std.math.cast(u64, payload) orelse return error.BadId },
        1 => .{ .uuid = payload },
        else => error.BadId,
    };
}

/// Write the collection to `dir`.
///
/// Refuses a quantized collection. Nothing here writes the codes or the
/// codebook, and a load that came back fp32 with no error was exactly the
/// silent degradation §1 forbids: a collection that told its client it was
/// quantized would answer every query from a store it no longer had.
/// `persist` is exercised by its own tests and by nothing else, so refusing
/// costs no caller anything and keeps the round-trip honest.
pub fn save(coll: *const Collection, dir: []const u8) !void {
    if (coll.quant_mode != .none or coll.quant.load(.acquire) != null) return error.QuantizedNotPersisted;
    try Persistence.ensureDir(dir);
    const n = coll.id_space.count();
    // Every arena write below is `n * stride` bytes out of a buffer sized for
    // `capacity * stride`, so a count past capacity would read past the arena.
    // The id space and the vector space are grown together and this says so.
    std.debug.assert(n <= coll.config.capacity);
    std.debug.assert(coll.space.stride > 0);
    var pbuf: [512]u8 = undefined;

    // --- vectors.bin ---
    {
        const p = try Persistence.path(&pbuf, dir, "vectors.bin");
        const size = storage.arenaFileSize(coll.config.capacity, coll.space.stride);
        const fd = try storage.openSized(p, size, true);
        defer storage.closeFd(fd);
        var h = storage.Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(storage.Header.Kind.vectors),
            .dim = coll.config.dim,
            .count = n,
            .capacity = coll.config.capacity,
            .stride = coll.space.stride,
            .kind_field = @intCast(coll.config.metric.toProto()),
            // §6.4: "Cosine is normalised at ingest and stored normalised...
            // Record this in the header." A reader must know whether the bytes
            // are pre-normalised or it will normalise them a second time.
            .flags = if (coll.config.datatype.normalisesAtIngest(coll.config.metric))
                storage.Header.Flags.normalised
            else
                0,
            .datatype = @intFromEnum(coll.config.datatype),
            .seed = coll.config.seed,
            .graph_checksum = if (coll.graph) |g| g.checksum() else 0,
            .hnsw_m = @intCast(coll.config.hnsw_m),
            .hnsw_ef_construct = @intCast(coll.config.hnsw_ef_construct),
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);
        try storage.writeData(fd, coll.space.arena[0 .. n * coll.space.stride]);
    }

    // --- ids.bin ---
    {
        const p = try Persistence.path(&pbuf, dir, "ids.bin");
        const bytes = n * id_record_size;
        const fd = try storage.openSized(p, storage.header_size + bytes, true);
        defer storage.closeFd(fd);
        var h = storage.Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(storage.Header.Kind.ids),
            .dim = 0,
            .count = n,
            .capacity = coll.config.capacity,
            .stride = id_record_size,
            .kind_field = 0,
            .flags = 0,
            .seed = 0,
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);
        const buf = try coll.alloc.alloc(u8, bytes);
        defer coll.alloc.free(buf);
        for (coll.id_space.table.ids[0..n], 0..) |id, i| encodeId(id, buf[i * id_record_size ..][0..id_record_size]);
        try storage.writeData(fd, buf);
    }

    // --- deleted.bits ---
    {
        const p = try Persistence.path(&pbuf, dir, "deleted.bits");
        const words = (coll.config.capacity + 63) / 64;
        const bytes = words * @sizeOf(u64);
        const fd = try storage.openSized(p, storage.header_size + bytes, true);
        defer storage.closeFd(fd);
        var h = storage.Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(storage.Header.Kind.deleted),
            .dim = 0,
            .count = coll.deleted_count,
            .capacity = coll.config.capacity,
            .stride = 8,
            .kind_field = 0,
            .flags = 0,
            .seed = 0,
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);
        // Serialise the bitmap word by word rather than casting the DynamicBitSet's
        // internals, whose representation is not part of its API.
        const tmp = try coll.alloc.alloc(u64, words);
        defer coll.alloc.free(tmp);
        @memset(tmp, 0);
        var i: usize = 0;
        while (i < coll.config.capacity) : (i += 1) {
            if (coll.deleted.isSet(i)) tmp[i / 64] |= @as(u64, 1) << @intCast(i % 64);
        }
        try storage.writeData(fd, std.mem.sliceAsBytes(tmp));
    }

    // --- payload.bin ---
    //
    // §6.4 designed `payload.bin` plus a `payload.idx` offset table. Built
    // without the table: each record carries its length, and the count in the
    // header is the point count, so a load walks the records in offset order
    // and needs nothing else. Written whenever any point has a payload.
    if (coll.payload.with_payload > 0) {
        const p = try Persistence.path(&pbuf, dir, "payload.bin");
        var total: usize = 0;
        for (0..n) |i| total += 4 + coll.payload.get(@intCast(i)).len;
        const fd = try storage.openSized(p, storage.header_size + total, true);
        defer storage.closeFd(fd);
        var h = storage.Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(storage.Header.Kind.payload),
            .dim = 0,
            .count = n,
            .capacity = coll.config.capacity,
            .stride = 0,
            .kind_field = 0,
            .flags = 0,
            .seed = 0,
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);
        const buf = try coll.alloc.alloc(u8, total);
        defer coll.alloc.free(buf);
        var w: usize = 0;
        for (0..n) |i| {
            const blob = coll.payload.get(@intCast(i));
            std.mem.writeInt(u32, buf[w..][0..4], @intCast(blob.len), .little);
            w += 4;
            @memcpy(buf[w..][0..blob.len], blob);
            w += blob.len;
        }
        std.debug.assert(w == total);
        try storage.writeData(fd, buf);
    }

    // --- payload.schema ---
    //
    // The postings are rebuilt from the blobs on load (§6.4 calls the id map
    // "rebuildable" for the same reason), so only the schema is written:
    // per field, `[u8 kind][u16 name_len][name]`.
    if (coll.payload.fields.items.len > 0) {
        const p = try Persistence.path(&pbuf, dir, "payload.schema");
        var total: usize = 0;
        for (coll.payload.fields.items) |f| total += 3 + f.name.len;
        const fd = try storage.openSized(p, storage.header_size + total, true);
        defer storage.closeFd(fd);
        var h = storage.Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(storage.Header.Kind.payload_schema),
            .dim = 0,
            .count = coll.payload.fields.items.len,
            .capacity = coll.config.capacity,
            .stride = 0,
            .kind_field = 0,
            .flags = 0,
            .seed = 0,
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);
        const buf = try coll.alloc.alloc(u8, total);
        defer coll.alloc.free(buf);
        var w: usize = 0;
        for (coll.payload.fields.items) |f| {
            buf[w] = @intFromEnum(f.kind);
            std.mem.writeInt(u16, buf[w + 1 ..][0..2], @intCast(f.name.len), .little);
            @memcpy(buf[w + 3 ..][0..f.name.len], f.name);
            w += 3 + f.name.len;
        }
        try storage.writeData(fd, buf);
    }

    // --- graph.bin ---
    if (coll.graph) |g| {
        const p = try Persistence.path(&pbuf, dir, "graph.bin");
        const l0 = std.mem.sliceAsBytes(g.level0);
        const upper = std.mem.sliceAsBytes(g.upper_neighbours);
        const offs = std.mem.sliceAsBytes(g.upper_offsets);
        const levels = g.node_levels;
        const total = l0.len + upper.len + offs.len + levels.len;

        const fd = try storage.openSized(p, storage.header_size + total, true);
        defer storage.closeFd(fd);
        var h = storage.Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(storage.Header.Kind.graph),
            .dim = g.params.m,
            .count = g.count,
            .capacity = g.capacity,
            .stride = g.params.m0,
            .kind_field = @intCast(g.entry_point),
            .flags = g.max_level,
            .seed = g.params.seed,
            .graph_checksum = g.checksum(),
            .header_crc = 0,
        };
        try storage.writeHeader(fd, &h);
        // Concatenated in a fixed order; the header's counts are what let the
        // reader split them again without a manifest.
        // `total` is computed from the four lengths above and the four
        // `@memcpy`s below consume exactly it. Stated because the reader
        // splits this blob back apart using the *header's* counts, not the
        // file's length: a `total` that drifted from the writes would produce
        // a file that loads without error and hands back a graph whose upper
        // levels are somebody else's bytes. §6.4's "no parse step" is what
        // makes the concatenation load-bearing.
        std.debug.assert(total == l0.len + upper.len + offs.len + levels.len);
        const buf = try coll.alloc.alloc(u8, total);
        defer coll.alloc.free(buf);
        var w: usize = 0;
        @memcpy(buf[w..][0..l0.len], l0);
        w += l0.len;
        @memcpy(buf[w..][0..upper.len], upper);
        w += upper.len;
        @memcpy(buf[w..][0..offs.len], offs);
        w += offs.len;
        @memcpy(buf[w..][0..levels.len], levels);
        std.debug.assert(w + levels.len == total);
        try storage.writeData(fd, buf);
    }
}

/// Reopen a collection from `dir`.
///
/// Structural validation of a graph read from `graph.bin`, before anything
/// indexes with it.
///
/// The header CRC covers the header only (`storage.Header.crcRegion`), so the
/// body arrives unvalidated -- and the first thing that touches it is
/// `Graph.checksum`, which walks `neighbours(node, lvl)` for every `lvl` up to
/// `node_levels[node]`. `upperSlice` computes
/// `upper_neighbours[upper_offsets[node] + (lvl - 1) * m ..][0..m]` with no
/// bounds check of its own, so a truncated or corrupted file reached that walk
/// unvalidated and the checksum meant to catch it was itself the out-of-bounds
/// read: a panic in a safe build, a wild read in ReleaseFast. The same shape as
/// the `count`-past-`capacity` header above, one file over.
///
/// What is checked here is the layout invariant the writer maintains
/// (`index/build.zig`: a running cursor of `level * m` per node, laid out in
/// node order) plus the range of every id a traversal can dereference.
/// Everything else about the graph is the checksum's job, which is sound once
/// this has run.
fn validateGraph(g: *const hnsw.Graph) !void {
    if (g.count > g.capacity) return error.GraphCorrupt;
    if (g.max_level > hnsw.level_cap) return error.GraphCorrupt;
    if (g.count == 0) {
        if (g.entry_point != hnsw.empty_neighbour) return error.GraphCorrupt;
    } else {
        // `Index.search` descends from the entry point through every level
        // down to 1, so the entry must own the levels `max_level` claims.
        if (g.entry_point >= g.count) return error.GraphCorrupt;
        if (g.node_levels[g.entry_point] < g.max_level) return error.GraphCorrupt;
    }

    // `usize` rather than the `u32` the file holds: the accumulator is derived
    // from bytes that may be anything, and `level_cap * m * capacity` overflows
    // a u32 long before the comparison that would have caught it.
    var cursor: usize = 0;
    for (g.node_levels[0..g.count], 0..) |lvl, i| {
        if (lvl > hnsw.level_cap) return error.GraphCorrupt;
        if (g.upper_offsets[i] != cursor) return error.GraphCorrupt;
        const start = cursor;
        cursor += @as(usize, lvl) * g.params.m;
        if (cursor > g.upper_neighbours.len) return error.GraphCorrupt;
        for (g.level0[i * g.params.m0 ..][0..g.params.m0]) |nb| {
            if (nb != hnsw.empty_neighbour and nb >= g.count) return error.GraphCorrupt;
        }
        for (g.upper_neighbours[start..cursor]) |nb| {
            if (nb != hnsw.empty_neighbour and nb >= g.count) return error.GraphCorrupt;
        }
    }
    if (g.upper_offsets[g.count] != cursor) return error.GraphCorrupt;
}

/// §6.4: "no index rebuild", the graph is read back rather than reconstructed,
/// and its checksum is verified against the one recorded at save time so a
/// truncated or mismatched graph fails loudly instead of degrading recall.
pub fn load(alloc: std.mem.Allocator, name: []const u8, dir: []const u8, placement: storage.Placement) !Collection {
    var pbuf: [512]u8 = undefined;

    const vp = try Persistence.path(&pbuf, dir, "vectors.bin");
    const vfd = try storage.openSized(vp, 0, false);
    defer storage.closeFd(vfd);
    const vh = try storage.readHeader(vfd, .vectors);

    const metric = dist.Metric.fromProto(@intCast(vh.metricField())) orelse return error.BadMetric;
    // The header decides how to read every byte of the arena, so an
    // unrecognised value is a refusal rather than a fallback to fp32: reading
    // a f16 arena as f32 produces a full result set of plausible nonsense.
    const dt = std.enums.fromInt(dist.Datatype, vh.datatype) orelse return error.BadDatatype;

    // A mapped placement would have `Collection.init` create and map
    // `<dir>/<name>.vectors.bin` — a *different* file from the `vectors.bin`
    // being loaded here, and an empty one. The load would then skip the pread
    // (the bytes are supposedly already mapped) and return a collection
    // reporting `count` rows of zeros: a successful load of nothing.
    //
    // The two layouts differ for a reason — a snapshot is one directory per
    // collection, a live server is one file per collection in a shared
    // directory — so this refuses rather than guessing which one the caller
    // meant. Nothing calls `load` with a placement yet; the server does not
    // persist (see README, "What it does not protect you from").
    if (placement.isMapped()) return error.PlacementNotSupportedOnLoad;

    var coll = try Collection.init(alloc, name, .{
        .dim = @intCast(vh.dim),
        .metric = metric,
        .datatype = dt,
        .capacity = @intCast(vh.capacity),
        .hnsw_m = vh.hnsw_m,
        .hnsw_ef_construct = vh.hnsw_ef_construct,
        .seed = vh.seed,
        .placement = .pinned,
    });
    errdefer coll.deinit();

    // The header's stride is how the file was *written*; `coll.space.stride`
    // is how this build lays rows out. They agree unless `row_align` or a
    // datatype's element size changed between the writer and the reader, and
    // then reading `n * stride` bytes straight into the arena would land every
    // row after the first at the wrong offset, quietly. Refuse instead.
    if (vh.stride != collection.strideFor(@intCast(vh.dim), dt) or vh.stride != coll.space.stride) {
        return error.StrideMismatch;
    }

    const n: usize = @intCast(vh.count);
    // `save` asserts `count <= capacity`; `load` re-derives both from the same
    // header and used to validate neither against the other, so a header with
    // a count past its capacity (and a repaired CRC) sliced past the arena.
    if (n > coll.config.capacity) return error.Mismatch;
    // §6.4's "startup time bounded by sequential read bandwidth" is paid here.
    const want = n * coll.space.stride;
    const got = try storage.readData(vfd, coll.space.arena[0..want]);
    if (got < want) return error.ShortFile;

    // --- ids ---
    {
        const ip = try Persistence.path(&pbuf, dir, "ids.bin");
        const ifd = try storage.openSized(ip, 0, false);
        defer storage.closeFd(ifd);
        const ih = try storage.readHeader(ifd, .ids);
        if (ih.count != vh.count) return error.Mismatch;

        if (ih.stride != id_record_size) return error.StrideMismatch;
        const bytes = n * id_record_size;
        const buf = try alloc.alloc(u8, bytes);
        defer alloc.free(buf);
        const read = try storage.readData(ifd, buf);
        if (read < bytes) return error.ShortFile;
        for (coll.id_space.table.ids[0..n], 0..) |*id, i| {
            id.* = try decodeId(buf[i * id_record_size ..][0..id_record_size]);
        }
        coll.id_space.table.len = n;
        coll.id_space.next.store(n, .release);

        // §6.4 calls idmap.bin "rebuildable"; rebuilding is cheaper than
        // validating a persisted open-addressed table, and it cannot be stale.
        coll.id_space.map.clear();
        for (0..n) |i| {
            _ = try coll.id_space.map.getOrInsert(coll.id_space.table.ids[i], @intCast(i));
        }
    }

    // --- deleted ---
    {
        const dp = try Persistence.path(&pbuf, dir, "deleted.bits");
        if (storage.openSized(dp, 0, false)) |dfd| {
            defer storage.closeFd(dfd);
            const dh = try storage.readHeader(dfd, .deleted);
            const words: usize = ((@as(usize, @intCast(dh.capacity)) + 63) / 64);
            const tmp = try alloc.alloc(u64, words);
            defer alloc.free(tmp);
            // A short bitmap is a refusal, like every other file here: it
            // used to be read into uninitialised words with the byte count
            // ignored, so a truncated file resurrected or tombstoned points
            // at random, with no error.
            const bytes = std.mem.sliceAsBytes(tmp);
            const read = try storage.readData(dfd, bytes);
            if (read < bytes.len) return error.ShortFile;
            var i: usize = 0;
            while (i < coll.config.capacity and i < dh.capacity) : (i += 1) {
                if (tmp[i / 64] & (@as(u64, 1) << @intCast(i % 64)) != 0) {
                    coll.deleted.set(i);
                    coll.deleted_count += 1;
                }
            }
        } else |_| {}
    }

    // --- payload ---
    {
        const pp = try Persistence.path(&pbuf, dir, "payload.bin");
        if (storage.openSized(pp, 0, false)) |pfd| {
            defer storage.closeFd(pfd);
            const ph = try storage.readHeader(pfd, .payload);
            if (ph.count != vh.count) return error.Mismatch;
            const size = try storage.fileSize(pfd);
            if (size < storage.header_size) return error.ShortFile;
            const bytes: usize = @intCast(size - storage.header_size);
            const buf = try alloc.alloc(u8, bytes);
            defer alloc.free(buf);
            const read = try storage.readData(pfd, buf);
            if (read < bytes) return error.ShortFile;
            var r: usize = 0;
            for (0..n) |i| {
                if (r + 4 > bytes) return error.ShortFile;
                const len = std.mem.readInt(u32, buf[r..][0..4], .little);
                r += 4;
                if (r + len > bytes) return error.ShortFile;
                if (len > 0) try coll.payload.setFramed(alloc, @intCast(i), buf[r..][0..len]);
                r += len;
            }
        } else |_| {}
    }

    // --- payload schema ---
    {
        const sp = try Persistence.path(&pbuf, dir, "payload.schema");
        if (storage.openSized(sp, 0, false)) |sfd| {
            defer storage.closeFd(sfd);
            const sh = try storage.readHeader(sfd, .payload_schema);
            const size = try storage.fileSize(sfd);
            if (size < storage.header_size) return error.ShortFile;
            const bytes: usize = @intCast(size - storage.header_size);
            const buf = try alloc.alloc(u8, bytes);
            defer alloc.free(buf);
            const read = try storage.readData(sfd, buf);
            if (read < bytes) return error.ShortFile;
            var r: usize = 0;
            for (0..@intCast(sh.count)) |_| {
                if (r + 3 > bytes) return error.ShortFile;
                const kind = std.enums.fromInt(payload_mod.Kind, buf[r]) orelse return error.BadPayloadIndexKind;
                const name_len = std.mem.readInt(u16, buf[r + 1 ..][0..2], .little);
                r += 3;
                if (r + name_len > bytes) return error.ShortFile;
                try coll.payload.createIndex(alloc, buf[r..][0..name_len], kind, n);
                r += name_len;
            }
        } else |_| {}
    }

    // --- graph ---
    {
        const gp = try Persistence.path(&pbuf, dir, "graph.bin");
        if (storage.openSized(gp, 0, false)) |gfd| {
            defer storage.closeFd(gfd);
            const gh = try storage.readHeader(gfd, .graph);
            // The graph's `m` (its header's `dim`) is the collection's: a
            // graph of another shape is not this collection's graph.
            if (gh.dim != coll.config.hnsw_m) return error.Mismatch;
            // A graph cannot cover more points than exist, or than it has
            // room for; `needsRebuild` asserts the first on the next upsert.
            if (gh.count > gh.capacity or gh.count > vh.count) return error.Mismatch;
            // `gh.capacity` sizes every array below, and the extend-build path
            // (`Graph.copyFrom`) asserts the two graphs' capacities are equal:
            // an assert that vanishes in ReleaseFast and would then memcpy one
            // graph's CSR into the other's shorter arena.
            if (gh.capacity != coll.config.capacity) return error.Mismatch;
            // Level 0's stride, as `vh.stride` is the arena's. A file whose
            // level-0 rows are a different width is read here at *our* width,
            // so every section after it is offset; the checksum would catch
            // that, but as a mismatch rather than as the shape error it is.
            if (gh.stride != 2 * gh.dim) return error.StrideMismatch;

            const g = try alloc.create(hnsw.Graph);
            errdefer alloc.destroy(g);
            g.* = try hnsw.Graph.init(alloc, hnsw.Params.fromM(
                @intCast(gh.dim),
                coll.config.hnsw_ef_construct,
                gh.seed,
            ), @intCast(gh.capacity));
            errdefer g.deinit();

            const l0 = std.mem.sliceAsBytes(g.level0);
            const upper = std.mem.sliceAsBytes(g.upper_neighbours);
            const offs = std.mem.sliceAsBytes(g.upper_offsets);
            const total = l0.len + upper.len + offs.len + g.node_levels.len;
            const buf = try alloc.alloc(u8, total);
            defer alloc.free(buf);
            const read = try storage.readData(gfd, buf);
            if (read < total) return error.ShortFile;

            var r: usize = 0;
            @memcpy(l0, buf[r..][0..l0.len]);
            r += l0.len;
            @memcpy(upper, buf[r..][0..upper.len]);
            r += upper.len;
            @memcpy(offs, buf[r..][0..offs.len]);
            r += offs.len;
            @memcpy(g.node_levels, buf[r..][0..g.node_levels.len]);

            g.count = @intCast(gh.count);
            g.entry_point = @intCast(gh.entryPointField());
            // Range-checked before the narrowing cast, for the same reason
            // `ScrollPoints.limit` is: `@intCast` of a u32 past 255 is a panic
            // in a safe build and a truncation in ReleaseFast, and a truncated
            // level is a plausible one.
            if (gh.flags > hnsw.level_cap) return error.GraphCorrupt;
            g.max_level = @intCast(gh.flags);

            // Before the checksum, because the checksum is what would index
            // with these bytes. See `validateGraph`.
            try validateGraph(g);
            // §8.7 option (b) put to work: the checksum recorded at save time
            // must match what was read back, or the graph is not the one the
            // results were measured against.
            if (g.checksum() != gh.graph_checksum) return error.GraphChecksumMismatch;

            coll.graph = g;
            coll.graph_count.store(@intCast(gh.count), .release);
            coll.indexed_count = @intCast(gh.count);
            coll.index_state.store(.ready, .release);
        } else |_| {}
    }

    return coll;
}

test "a data dir longer than the path buffer is an error, not illegal behaviour" {
    // `dir` is `--data-dir`, so its length is the operator's. The buffer is
    // 512 bytes and Linux allows 4096, so a perfectly legal path overflows it.
    // This used to be `catch unreachable`: a clean error in Debug and illegal
    // behaviour in ReleaseFast, which is the mode §9 quotes numbers from.
    var buf: [512]u8 = undefined;
    const long = "/" ** 600;
    try std.testing.expectError(error.PathTooLong, Persistence.path(&buf, long, "vectors.bin"));

    // And the ordinary case still produces the path it always did.
    const ok = try Persistence.path(&buf, "/tmp/x", "vectors.bin");
    try std.testing.expectEqualStrings("/tmp/x/vectors.bin", ok);
}

test "§6.4: a collection round-trips through save and load" {
    const dim = 32;
    const n = 500;
    var c = try makeCollection(dim, .euclid, 1024);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0x5a7e);
    const rnd = prng.random();
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i * 3 }, &v);
    }
    _ = c.delete(.{ .num = 9 });
    try buildIndex(&c, .serial, 1);

    const dir = "/tmp/strawmann-test-collection";
    try save(&c, dir);

    var reloaded = try load(testing.allocator, "test", dir, .pinned);
    defer reloaded.deinit();

    try testing.expectEqual(c.config.dim, reloaded.config.dim);
    try testing.expectEqual(c.config.metric, reloaded.config.metric);
    try testing.expectEqual(c.count(), reloaded.count());
    try testing.expectEqual(c.deleted_count, reloaded.deleted_count);

    // Vectors identical, bit for bit.
    for (0..n) |i| {
        try testing.expectEqualSlices(f32, c.space.rowConst(@intCast(i)), reloaded.space.rowConst(@intCast(i)));
    }
    // External ids resolve to the same offsets.
    for (0..n) |i| {
        try testing.expectEqual(c.id_space.lookup(.{ .num = i * 3 }), reloaded.id_space.lookup(.{ .num = i * 3 }));
    }
    // §6.4: "no index rebuild", the graph came back, not rebuilt.
    try testing.expect(reloaded.graph != null);
    try testing.expectEqual(c.graph.?.checksum(), reloaded.graph.?.checksum());
    try testing.expectEqual(Status.green, reloaded.status());

    // And it answers queries identically.
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 1024, 128);
    defer scratch.deinit(testing.allocator);
    var q: [dim]f32 = undefined;
    for (&q) |*x| x.* = rnd.floatNorm(f32);

    var ab: [10]Candidate = undefined;
    var a_out = heap.TopK.init(&ab, 10);
    search(&c, &q, 128, .approximate, &scratch, &a_out);
    var bb: [10]Candidate = undefined;
    var b_out = heap.TopK.init(&bb, 10);
    search(&reloaded, &q, 128, .approximate, &scratch, &b_out);

    const ra = a_out.finish();
    const rb = b_out.finish();
    try testing.expectEqual(ra.len, rb.len);
    for (ra, rb) |x, y| {
        try testing.expectEqual(x.id, y.id);
        try testing.expectEqual(x.score, y.score);
    }
}

test "§6.4: payloads and the field indexes survive the round-trip" {
    var c = try makeCollection(4, .euclid, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    // `{ "a": "k<i % 3>" }` as wire map entries at field 3.
    var ebuf: [64]u8 = undefined;
    var nbuf: [8]u8 = undefined;
    for (0..30) |i| {
        var w = wire.Writer.init(&ebuf);
        const entry = try w.beginNested(3, 2);
        try w.writeStringField(1, "a");
        const val = try w.beginNested(2, 2);
        try w.writeStringField(payload_mod.value_string, try std.fmt.bufPrint(&nbuf, "k{d}", .{i % 3}));
        try w.endNested(val);
        try w.endNested(entry);
        _ = try c.upsertWithPayload(.{ .num = i }, &v, w.written(), 3);
    }
    // Point 5 has none.
    _ = try c.upsert(.{ .num = 5 }, &v);
    try c.createPayloadIndex("a", .keyword);

    const dir = "/tmp/strawmann-test-payload";
    try save(&c, dir);
    var reloaded = try load(testing.allocator, "test", dir, .pinned);
    defer reloaded.deinit();

    for (0..30) |i| {
        try testing.expectEqualSlices(u8, c.payload.get(@intCast(i)), reloaded.payload.get(@intCast(i)));
    }
    try testing.expectEqual(@as(usize, 0), reloaded.payload.get(5).len);
    try testing.expectEqual(@as(usize, 29), reloaded.payload.with_payload);
    // The index came back as a schema and was rebuilt into postings.
    try testing.expectEqual(@as(usize, 1), reloaded.payload.fields.items.len);
    try testing.expectEqualStrings("a", reloaded.payload.fields.items[0].name);
    var f = payload_mod.Filter{};
    try f.add(.must, .{ .key = "a", .match = .{ .keyword = "k1" } });
    var bits: [1]u64 = undefined;
    const sel = reloaded.payload.select(&f, &bits, 30).?;
    try testing.expectEqual(@as(usize, 10), sel.count);
    for (0..30) |i| try testing.expectEqual(i % 3 == 1, payload_mod.testBit(&bits, @intCast(i)));
}

test "§6.4: hnsw_m, ef_construct and both id kinds survive the round-trip" {
    // `load` built its `Config` with default `hnsw_m` / `hnsw_ef_construct`,
    // so a collection created at m=24 came back as m=16 wearing a m=24
    // graph, and its next rebuild silently changed shape. And `ids.bin` was
    // the raw bytes of a Zig tagged union, a layout no format owns.
    const dim = 8;
    var c = try Collection.init(testing.allocator, "cfg", .{
        .dim = dim,
        .metric = .dot,
        .capacity = 128,
        .hnsw_m = 24,
        .hnsw_ef_construct = 77,
    });
    defer c.deinit();
    var v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    for (0..30) |i| {
        v[0] = @floatFromInt(i);
        // Alternate the two `PointId` kinds, with a uuid whose high and low
        // halves differ so a truncated payload would be caught.
        const id: ExternalId = if (i % 2 == 0) .{ .num = i * 7 } else .{ .uuid = (@as(u128, i) << 64) | 0xabcd };
        _ = try c.upsert(id, &v);
    }
    try buildIndex(&c, .serial, 1);
    try testing.expectEqual(@as(usize, 24), c.graph.?.params.m);

    const dir = "/tmp/strawmann-test-config";
    try save(&c, dir);
    var reloaded = try load(testing.allocator, "cfg", dir, .pinned);
    defer reloaded.deinit();

    try testing.expectEqual(@as(usize, 24), reloaded.config.hnsw_m);
    try testing.expectEqual(@as(usize, 77), reloaded.config.hnsw_ef_construct);
    try testing.expectEqual(@as(usize, 24), reloaded.graph.?.params.m);
    try testing.expectEqual(@as(usize, 77), reloaded.graph.?.params.ef_construct);
    for (0..30) |i| {
        const id: ExternalId = if (i % 2 == 0) .{ .num = i * 7 } else .{ .uuid = (@as(u128, i) << 64) | 0xabcd };
        try testing.expectEqual(@as(?u32, @intCast(i)), reloaded.id_space.lookup(id));
        try testing.expect(reloaded.id_space.external(@intCast(i)).eql(id));
    }
    // A rebuild after reload keeps the shape it was created with.
    invalidateIndex(&reloaded);
    reloaded.index_state.store(.absent, .release);
    try buildIndex(&reloaded, .serial, 1);
    try testing.expectEqual(@as(usize, 24), reloaded.graph.?.params.m);
}

test "§6.4: a truncated deleted.bits is refused, not read as random tombstones" {
    // The bitmap was read into uninitialised words with the byte count
    // ignored, the one file in `load` that did not return `ShortFile`.
    var c = try makeCollection(4, .dot, 256);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    for (0..10) |i| _ = try c.upsert(.{ .num = i }, &v);
    _ = c.delete(.{ .num = 3 });

    const dir = "/tmp/strawmann-test-short-deleted";
    try save(&c, dir);
    var pbuf: [512]u8 = undefined;
    const p = try Persistence.path(&pbuf, dir, "deleted.bits");
    const fd = try storage.openSized(p, 0, false);
    // Keep the header and one byte of bitmap: a plausible, short file.
    _ = std.os.linux.ftruncate(fd, storage.header_size + 1);
    storage.closeFd(fd);

    try testing.expectError(error.ShortFile, load(testing.allocator, "test", dir, .pinned));
}

test "a quantized collection is refused by save rather than reloaded as fp32" {
    var c = try makeCollection(4, .dot, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    for (0..10) |i| {
        v[1] = @floatFromInt(i);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    // Configured but not yet built is refused too: the collection would
    // reload as one that never asked.
    c.quant_mode = .scalar;
    try testing.expectError(error.QuantizedNotPersisted, save(&c, "/tmp/strawmann-test-quant-refused"));
    try quantize(&c, .scalar);
    try testing.expectError(error.QuantizedNotPersisted, save(&c, "/tmp/strawmann-test-quant-refused"));
}

test "§6.4: a cosine collection records that it stored normalised vectors" {
    const dim = 16;
    var c = try makeCollection(dim, .cosine, 64);
    defer c.deinit();
    var v = [_]f32{ 3, 4 } ++ [_]f32{0} ** 14;
    _ = try c.upsert(.{ .num = 1 }, &v);

    const dir = "/tmp/strawmann-test-cosine";
    try save(&c, dir);

    var pbuf: [512]u8 = undefined;
    const p = try Persistence.path(&pbuf, dir, "vectors.bin");
    const fd = try storage.openSized(p, 0, false);
    defer storage.closeFd(fd);
    const h = try storage.readHeader(fd, .vectors);
    // Without this flag a reader would normalise already-normalised data a
    // second time, which for most vectors is a no-op thanks to §8.3's
    // short-circuit, and for the rest is silent corruption.
    try testing.expect(h.flags & storage.Header.Flags.normalised != 0);

    var reloaded = try load(testing.allocator, "c", dir, .pinned);
    defer reloaded.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.6), reloaded.space.rowConst(0)[0], 1e-6);
}

test "§8.7: a corrupted graph file is refused rather than silently degrading recall" {
    const dim = 8;
    var c = try makeCollection(dim, .dot, 128);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    for (0..50) |i| {
        v[i % dim] = @floatFromInt(i);
        _ = try c.upsert(.{ .num = i }, &v);
    }
    try buildIndex(&c, .serial, 1);

    const dir = "/tmp/strawmann-test-corrupt";
    var pbuf: [512]u8 = undefined;

    const poke = struct {
        fn f(path: []const u8, off: usize, word: u32) !void {
            const fd = try storage.openSized(path, 0, false);
            defer storage.closeFd(fd);
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, word, .little);
            _ = std.os.linux.pwrite(fd, &bytes, 4, @intCast(off));
        }
        fn peek(path: []const u8, off: usize) !u32 {
            const fd = try storage.openSized(path, 0, false);
            defer storage.closeFd(fd);
            var bytes: [4]u8 = undefined;
            _ = std.os.linux.pread(fd, &bytes, 4, @intCast(off));
            return std.mem.readInt(u32, &bytes, .little);
        }
    };

    const g = c.graph.?;
    const w = @sizeOf(u32);
    const upper_off = storage.header_size + g.level0.len * w;
    const offsets_off = upper_off + g.upper_neighbours.len * w;
    const levels_off = offsets_off + g.upper_offsets.len * w;

    // A neighbour id that names no node. Structural, so it is refused by
    // `validateGraph` before the checksum: the checksum walks every neighbour
    // list, so an id it cannot dereference has to be caught ahead of it.
    {
        try save(&c, dir);
        const p = try Persistence.path(&pbuf, dir, "graph.bin");
        try poke.f(p, storage.header_size + 16, 0xefbeadde);
        try testing.expectError(error.GraphCorrupt, load(testing.allocator, "c", dir, .pinned));
    }

    // A CSR base offset that points outside the arena. This is the one that
    // used to crash: `Graph.checksum` walks `neighbours(node, lvl)` up to
    // `node_levels[node]`, and `upperSlice` slices
    // `upper_neighbours[upper_offsets[node] + (lvl - 1) * m ..][0..m]` with no
    // bounds check, so the checksum meant to catch the corruption was itself
    // the out-of-bounds read: a panic here, a wild read in ReleaseFast.
    {
        var node: u32 = 0;
        while (node < g.count and g.node_levels[node] == 0) node += 1;
        // The fixture must contain a node with an upper level, or this case
        // corrupts a row nothing reads and passes for the wrong reason.
        try testing.expect(node < g.count);
        try save(&c, dir);
        const p = try Persistence.path(&pbuf, dir, "graph.bin");
        try poke.f(p, offsets_off + node * w, 0xffff0000);
        try testing.expectError(error.GraphCorrupt, load(testing.allocator, "c", dir, .pinned));
    }

    // A level that does not match the CSR the file also carries: in range, so
    // it is the layout check rather than the range check that refuses it.
    {
        try save(&c, dir);
        const p = try Persistence.path(&pbuf, dir, "graph.bin");
        const orig = try poke.peek(p, levels_off);
        const swapped: u32 = (orig & 0xffffff00) | @as(u32, if (orig & 0xff == 1) 2 else 1);
        try poke.f(p, levels_off, swapped);
        try testing.expectError(error.GraphCorrupt, load(testing.allocator, "c", dir, .pinned));
    }

    // And a corruption that is structurally fine: one valid neighbour id
    // replaced by another. Nothing but the checksum can see this one, which is
    // what §8.7 option (b) is for.
    {
        try save(&c, dir);
        const p = try Persistence.path(&pbuf, dir, "graph.bin");
        const orig = try poke.peek(p, storage.header_size + 16);
        try poke.f(p, storage.header_size + 16, @intCast((@as(u64, orig) + 1) % 50));
        try testing.expectError(error.GraphChecksumMismatch, load(testing.allocator, "c", dir, .pinned));
    }

    // The untouched file still loads, so the cases above are refusals of the
    // corruption rather than of the format.
    {
        try save(&c, dir);
        var reloaded = try load(testing.allocator, "c", dir, .pinned);
        defer reloaded.deinit();
        try testing.expectEqual(g.checksum(), reloaded.graph.?.checksum());
    }
}

test "§6.4: a header whose count exceeds its capacity is refused, not read past the arena" {
    var c = try makeCollection(4, .dot, 16);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    for (0..8) |i| _ = try c.upsert(.{ .num = i }, &v);
    const dir = "/tmp/strawmann-test-overcount";
    try save(&c, dir);

    // Rewrite the vectors header with a count past its capacity and a
    // repaired CRC, which is what `load` used to trust: it sliced
    // `arena[0..count * stride]` past the arena's end.
    var pbuf: [512]u8 = undefined;
    const p = try Persistence.path(&pbuf, dir, "vectors.bin");
    const fd = try storage.openSized(p, 0, false);
    var h = try storage.readHeader(fd, .vectors);
    h.count = h.capacity + 1;
    try storage.writeHeader(fd, &h);
    storage.closeFd(fd);
    try testing.expectError(error.Mismatch, load(testing.allocator, "c", dir, .pinned));
}

test "a rebuild retires the old graph rather than freeing it under a search" {
    // The use-after-free this guards: `search` reads `index_state` with acquire
    // ordering and *then* dereferences `graph`. A rebuild that freed the old
    // graph between those two steps would pull it out from under a traversal
    // already in progress, reachable by W11 ("mixed read/write").
    const dim = 16;
    var c = try makeCollection(dim, .euclid, 512);
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0xf1f1);
    const rnd = prng.random();
    for (0..200) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    try buildIndex(&c, .serial, 1);
    const first = c.graph.?;
    try testing.expectEqual(@as(usize, 0), c.retired_graphs.items.len);

    // A second build must publish a new graph and retire the old one, and
    // must *not* free it while a search could still be traversing it. The
    // guard stands in for that search, which is what the title has always
    // claimed and what this test did not previously establish: without it the
    // rebuild now reclaims the old graph immediately, and the reads below
    // would be a use-after-free rather than an assertion.
    const guard = collection.SearchGuard.begin(&c);
    invalidateIndex(&c);
    try buildIndex(&c, .serial, 1);
    try testing.expect(c.graph.? != first);
    try testing.expectEqual(@as(usize, 1), c.retired_graphs.items.len);
    try testing.expectEqual(first, c.retired_graphs.items[0]);

    // The retired graph is still readable, that is the whole point.
    try testing.expect(first.count > 0);
    try testing.expect(first.entry_point != hnsw.empty_neighbour);
    guard.end();

    // And the live graph answers queries.
    var scratch = try hnsw.Index.Scratch.init(testing.allocator, 512, 128);
    defer scratch.deinit(testing.allocator);
    var buf: [10]Candidate = undefined;
    var out = heap.TopK.init(&buf, 10);
    search(&c, c.space.rowConst(3), 64, .approximate, &scratch, &out);
    try testing.expectEqual(@as(usize, 10), out.finish().len);
}

test "a re-quantize retires the old store" {
    const dim = 16;
    var c = try makeCollection(dim, .dot, 256);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(0xf2f2);
    const rnd = prng.random();
    for (0..100) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = i }, &v);
    }

    try quantize(&c, .scalar);
    try testing.expectEqual(@as(usize, 0), c.retired_quant.items.len);
    try quantize(&c, .binary);
    try testing.expectEqual(@as(usize, 1), c.retired_quant.items.len);
    // The live store is the new one.
    try testing.expect(c.quant.load(.acquire).?.* == .binary);
}

test "§12 scroll pages through id order, not arrival order" {
    // Arrival order is deliberately not id order: walking the arena directly
    // would pass a test that uploads sequentially, which is what bfb does by
    // default, and produce silently wrong pages for anyone else.
    var c = try makeCollection(4, .dot, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    const arrival = [_]u64{ 50, 10, 40, 20, 30 };
    for (arrival) |id| _ = try c.upsert(.{ .num = id }, &v);

    var buf: [8]ExternalId = undefined;
    const page = try scroll(&c, null, buf[0..2]);
    try testing.expectEqual(@as(usize, 2), page.ids.len);
    try testing.expectEqual(@as(u64, 10), page.ids[0].num);
    try testing.expectEqual(@as(u64, 20), page.ids[1].num);
    try testing.expectEqual(@as(u64, 30), page.next.?.num);

    // Resuming from the cursor repeats nothing and skips nothing.
    const page2 = try scroll(&c, page.next, buf[0..2]);
    try testing.expectEqual(@as(u64, 30), page2.ids[0].num);
    try testing.expectEqual(@as(u64, 40), page2.ids[1].num);

    const page3 = try scroll(&c, page2.next, buf[0..2]);
    try testing.expectEqual(@as(usize, 1), page3.ids.len);
    try testing.expectEqual(@as(u64, 50), page3.ids[0].num);
    try testing.expectEqual(@as(?ExternalId, null), page3.next);
}

test "scroll walks the whole collection exactly once" {
    var c = try makeCollection(4, .dot, 512);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    // Sparse, shuffled id space.
    var prng = std.Random.DefaultPrng.init(0xabc);
    const rnd = prng.random();
    var expected: [200]u64 = undefined;
    for (&expected, 0..) |*e, i| e.* = @as(u64, i) * 7 + rnd.uintLessThan(u64, 5);
    var shuffled = expected;
    rnd.shuffle(u64, &shuffled);
    var inserted: usize = 0;
    for (shuffled) |id| {
        _ = c.upsert(.{ .num = id }, &v) catch continue;
        inserted += 1;
    }

    var seen = std.ArrayList(u64).empty;
    defer seen.deinit(testing.allocator);
    var buf: [7]ExternalId = undefined;
    var cursor: ?ExternalId = null;
    var pages: usize = 0;
    while (true) {
        const p = try scroll(&c, cursor, &buf);
        for (p.ids) |id| try seen.append(testing.allocator, id.num);
        pages += 1;
        try testing.expect(pages < 1000); // no infinite paging
        cursor = p.next orelse break;
    }
    try testing.expectEqual(c.count(), seen.items.len);
    // Strictly ascending: no repeats, no regressions.
    for (seen.items[1..], seen.items[0 .. seen.items.len - 1]) |cur, prev| {
        try testing.expect(cur > prev);
    }
}

test "scroll skips tombstones and never hands back a dead cursor" {
    var c = try makeCollection(4, .dot, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    for (0..20) |i| _ = try c.upsert(.{ .num = i }, &v);
    // A long run of tombstones straddling a page boundary: a cursor pointing at
    // a deleted point would yield a page that looks like the end of the
    // collection when it is not.
    for (2..15) |i| _ = c.delete(.{ .num = i });

    var buf: [2]ExternalId = undefined;
    var seen = std.ArrayList(u64).empty;
    defer seen.deinit(testing.allocator);
    var cursor: ?ExternalId = null;
    while (true) {
        const p = try scroll(&c, cursor, &buf);
        for (p.ids) |id| try seen.append(testing.allocator, id.num);
        cursor = p.next orelse break;
    }
    try testing.expectEqual(@as(usize, 7), seen.items.len);
    for (seen.items) |id| try testing.expect(id < 2 or id >= 15);
}

test "upsert invalidates a cached scroll order" {
    var c = try makeCollection(4, .dot, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    for ([_]u64{ 10, 30 }) |id| _ = try c.upsert(.{ .num = id }, &v);

    var buf: [8]ExternalId = undefined;
    _ = try scroll(&c, null, &buf); // caches the order
    _ = try c.upsert(.{ .num = 20 }, &v);

    const p = try scroll(&c, null, &buf);
    try testing.expectEqual(@as(usize, 3), p.ids.len);
    try testing.expectEqual(@as(u64, 20), p.ids[1].num);
}

test "idLess is a total order across both PointId variants" {
    try testing.expect(idLess(.{ .num = 1 }, .{ .num = 2 }));
    try testing.expect(!idLess(.{ .num = 2 }, .{ .num = 1 }));
    try testing.expect(!idLess(.{ .num = 1 }, .{ .num = 1 }));
    // Numeric before uuid, by stated convention.
    try testing.expect(idLess(.{ .num = std.math.maxInt(u64) }, .{ .uuid = 0 }));
    try testing.expect(!idLess(.{ .uuid = 0 }, .{ .num = 0 }));
    try testing.expect(idLess(.{ .uuid = 1 }, .{ .uuid = 2 }));
}

test "§6.4: every storage datatype round-trips, and the header carries which" {
    const dim = 32;
    const n = 200;
    for ([_]dist.Datatype{ .float32, .float16, .uint8 }) |dt| {
        var c = try collection.makeCollectionDt(dim, .euclid, dt, 512);
        defer c.deinit();

        var prng = std.Random.DefaultPrng.init(0x0d7 + @as(u64, @intFromEnum(dt)));
        const rnd = prng.random();
        for (0..n) |i| {
            var v: [dim]f32 = undefined;
            // In u8's range so the same corpus survives every datatype and the
            // comparison below is about persistence, not about clamping.
            for (&v) |*x| x.* = @floatFromInt(rnd.uintLessThan(u8, 200));
            _ = try c.upsert(.{ .num = i }, &v);
        }
        try buildIndex(&c, .serial, 1);

        const dir = "/tmp/strawmann-test-datatype";
        try save(&c, dir);
        var reloaded = try load(testing.allocator, "test", dir, .pinned);
        defer reloaded.deinit();

        // The datatype survives, and with it the only way to read the arena.
        try testing.expectEqual(dt, reloaded.config.datatype);
        try testing.expectEqual(c.space.stride, reloaded.space.stride);
        try testing.expectEqualSlices(
            u8,
            c.space.arena[0 .. n * c.space.stride],
            reloaded.space.arena[0 .. n * reloaded.space.stride],
        );

        // And the reloaded collection answers the same as the original, which
        // is the property a byte comparison alone does not establish.
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = @floatFromInt(rnd.uintLessThan(u8, 200));
        var ab: [8]Candidate = undefined;
        var bb: [8]Candidate = undefined;
        var before = heap.TopK.init(&ab, 5);
        var after = heap.TopK.init(&bb, 5);
        collection.bruteForce(&c, &q, &before);
        collection.bruteForce(&reloaded, &q, &after);
        const x = before.finish();
        const y = after.finish();
        try testing.expectEqual(x.len, y.len);
        for (x, y) |bx, by| {
            try testing.expectEqual(bx.id, by.id);
            try testing.expectEqual(bx.score, by.score);
        }
    }
}

test "a header whose stride does not match the layout is refused" {
    // The stride in the header is the writer's row pitch. Reading a file
    // written with a different `row_align` (or a different element size) as
    // if it matched this build would place every row past the first at the
    // wrong offset and score garbage with no error, so the mismatch has to be
    // a refusal.
    var c = try collection.makeCollectionDt(8, .dot, .float32, 16);
    defer c.deinit();
    var v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = try c.upsert(.{ .num = 1 }, &v);

    const dir = "/tmp/strawmann-test-bad-stride";
    try save(&c, dir);

    var pbuf: [512]u8 = undefined;
    const path = try Persistence.path(&pbuf, dir, "vectors.bin");
    const fd = try storage.openSized(path, 0, false);
    var h = try storage.readHeader(fd, .vectors);
    // As written, the header agrees with the layout, and loads.
    try testing.expectEqual(collection.strideFor(8, .float32), @as(usize, @intCast(h.stride)));
    // A stride that is a plausible pitch (dim * 4, unpadded) but not ours.
    h.stride = 32;
    h.header_crc = h.computeCrc();
    try storage.writeHeader(fd, &h);
    storage.closeFd(fd);

    try testing.expectError(error.StrideMismatch, load(testing.allocator, "test", dir, .pinned));
}

test "a header naming an unknown datatype is refused, not read as fp32" {
    // The failure this guards is not a crash, it is a full result set computed
    // by reading f16 pairs as f32: plausible, wrong, and silent.
    var c = try collection.makeCollectionDt(8, .dot, .float16, 16);
    defer c.deinit();
    var v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = try c.upsert(.{ .num = 1 }, &v);

    const dir = "/tmp/strawmann-test-bad-datatype";
    try save(&c, dir);

    // Corrupt only the datatype byte, then repair the CRC so the file is
    // otherwise valid: without the CRC repair this would fail for the wrong
    // reason and prove nothing about the datatype check.
    var pbuf: [512]u8 = undefined;
    const path = try Persistence.path(&pbuf, dir, "vectors.bin");
    const fd = try storage.openSized(path, 0, false);
    var h = try storage.readHeader(fd, .vectors);
    h.datatype = 99;
    h.header_crc = h.computeCrc();
    try storage.writeHeader(fd, &h);
    storage.closeFd(fd);

    try testing.expectError(error.BadDatatype, load(testing.allocator, "test", dir, .pinned));
}
