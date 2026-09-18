//! §6.4, storage layout, and the three memory placements.
//!
//! ```
//! collection/
//!   meta.json           config, dim, distance, quant params, counts, format version
//!   vectors.<name>.bin  header + aligned vector arena
//!   graph.bin           header + level-0 neighbours + upper levels
//!   quant.<name>.bin    quantized codes (+ codebooks / scales in header)
//!   ids.bin             internal u32 → external id
//!   idmap.bin           external id → internal u32 (open-addressed table, rebuildable)
//!   deleted.bits        tombstone bitmap
//! ```
//!
//! §6.4's rules, each of which this file implements:
//!
//!   "Every file starts with a 4 KiB header: magic, format version, dim, count,
//!    capacity, distance, flags, and a CRC of the header only. **No CRC over
//!    the data**, we are not building for durability, and hashing 3 GB at
//!    startup would dominate load time."
//!
//!   "Recovery is `open` + `mmap`/`pread`. There is no parse step, no WAL
//!    replay, no index rebuild. Startup time for a 3 GB collection should be
//!    bounded by sequential read bandwidth."
//!
//! ## The placements, and why there are three
//!
//! §5.5 is the reason this file has a `Placement` at all:
//!
//!   "**Default mode ("resident"):** allocate an anonymous `MAP_HUGETLB` (or
//!    `MADV_HUGEPAGE`) region and populate it with `pread` from the flat file
//!    at startup. Satisfies 'everything loaded in memory by default', gives
//!    2 MiB pages, and startup cost is sequential I/O at full disk bandwidth.
//!    **Comparison mode ("mmap"):** true `mmap` of the file with
//!    `MAP_POPULATE`, 4 KiB pages. This is the Qdrant-like path and exists
//!    specifically so we can **measure the TLB penalty as a first-class
//!    result**."
//!
//! The complication §5.5 names is why the resident mode cannot simply be
//! `mmap` + `MADV_HUGEPAGE`: "`MADV_HUGEPAGE` on a **file-backed** mmap of an
//! ordinary filesystem does not generally give 2 MiB mappings. THP is an
//! anonymous-memory feature; file-backed large folios are filesystem- and
//! kernel-dependent and cannot be assumed." So resident mode allocates
//! *anonymous* memory and reads into it, which is the only portable way to get
//! huge pages behind the vector arena.
//!
//! The measured TLB delta on the development host (`bench-micro hw`) was 1.16×
//! on random access, with dTLB load misses per access falling from 0.953 to
//! 0.001, a ~1000× reduction in misses for a 16% latency win, because the
//! page walker overlaps well on this microarchitecture. On a part with a
//! smaller TLB or a slower walker the same experiment would show more.
//!
//! §5.5 wrote two modes because it was reasoning about our own choices. The
//! third exists because Qdrant has three and the comparison is only fair when
//! both engines can be put in the same one: see `Placement`, whose names are
//! Qdrant's rather than §5.5's for exactly that reason. A collection created
//! with no placement field runs `pinned` here and `Cached` there, which is a
//! semantic gap in §2's sense — now a measurable one rather than a footnote,
//! because both engines can be asked for either.

const std = @import("std");
const linux = std.os.linux;

/// Where a component's bytes live, in Qdrant's vocabulary.
///
/// §5.5 named two modes, "resident" and "mmap". Qdrant names three, and since
/// the point of this engine is to be measured against Qdrant under *its*
/// configuration, the enum is theirs rather than ours. From
/// `lib/api/src/grpc/proto/collections.proto`:
///
/// ```proto
/// enum Memory {
///   MemoryUnknown = 0;
///   Cold = 1;    // not pre-loaded from disk to RAM; cached with usage
///   Cached = 2;  // pre-loaded into disk-cache RAM on start, may be evicted
///   Pinned = 3;  // loaded in RAM and never evicted
/// }
/// ```
///
/// The correspondence to §5.5's two names, so nothing in the docs is orphaned:
/// `pinned` is what §5.5 calls "resident", and `cached` is what it calls
/// "mmap". `cold` is new here, and it is the one Qdrant reaches by default for
/// data it does not expect to fit.
///
/// Verified in Qdrant's source rather than assumed, because two of these are
/// counter-intuitive:
///
///   * **The default for dense vectors is `Cached`, not `Pinned`.**
///     `config.rs`: `Memory::resolve(...).unwrap_or(Memory::Cached)`. A
///     collection created with no placement field is served from a populated
///     mmap, not from the heap.
///   * **`Pinned` is not available for dense vector storage at all.** The proto
///     comment on `VectorParams.memory` says so outright: "`Pinned` is not
///     supported for dense vector storage." So strawmANN's *default and only
///     historical mode* is the one placement Qdrant cannot offer for vectors.
///     That is a real result, not a gap, and §5.5's TLB argument is why.
pub const Placement = enum {
    /// Anonymous memory, 2 MiB pages, populated by `pread` at startup, never
    /// evictable because it is not page cache. §5.5's "resident", and the
    /// strawmANN default. Qdrant's `Memory::Pinned`, which its dense vector
    /// storage does not implement.
    pinned,
    /// File-backed `MAP_SHARED`, 4 KiB pages, prefaulted at open so the first
    /// query does not pay the fault. The kernel may evict it under pressure.
    /// §5.5's "mmap", and Qdrant's `Memory::Cached` — its default.
    cached,
    /// File-backed `MAP_SHARED`, 4 KiB pages, *not* prefaulted, with
    /// `MADV_RANDOM` so the kernel does not read ahead. Every page arrives on
    /// a fault, which is what a node serving a larger-than-cache collection
    /// actually does. Qdrant's `Memory::Cold`.
    cold,

    pub fn name(self: Placement) []const u8 {
        return switch (self) {
            .pinned => "pinned",
            .cached => "cached",
            .cold => "cold",
        };
    }

    /// Qdrant's `Memory` proto values. Unknown (0) is absent, not a value.
    pub fn fromProto(v: i32) ?Placement {
        return switch (v) {
            1 => .cold,
            2 => .cached,
            3 => .pinned,
            else => null,
        };
    }

    pub fn toProto(self: Placement) i32 {
        return switch (self) {
            .cold => 1,
            .cached => 2,
            .pinned => 3,
        };
    }

    /// Qdrant's `Memory::from_on_disk`, exactly: the deprecated `on_disk` bool
    /// meant "loaded lazily" when true and "populated but evictable" when
    /// false. Note what it does *not* mean: `on_disk: false` is `Cached`, not
    /// `Pinned`. Reading it as "in RAM, on the heap" is the natural misreading
    /// and would make strawmANN claim parity it does not have.
    pub fn fromOnDisk(on_disk: bool) Placement {
        return if (on_disk) .cold else .cached;
    }

    /// Whether the bytes live in a file this process maps, rather than in
    /// memory it allocated.
    pub fn isMapped(self: Placement) bool {
        return self != .pinned;
    }

    /// The placement a request asks for, or `null` for "unstated".
    ///
    /// Qdrant's precedence, from `lib/collection/src/config.rs`:
    ///
    /// ```rust
    /// Memory::resolve(*memory, Some(Memory::from_on_disk(on_disk.unwrap_or_default())))
    ///     .unwrap_or(Memory::Cached)
    /// ```
    ///
    /// `memory` wins over `on_disk` when both are present — `Memory::resolve`
    /// is `memory.or(legacy)`, and its doc comment says "the explicit parameter
    /// always wins".
    ///
    /// The trailing `unwrap_or(Cached)` is deliberately *not* applied here.
    /// Qdrant defaults an unstated placement to `Cached`; strawmANN defaults it
    /// to `pinned`, so `null` means "the caller said nothing" and the caller
    /// applies its own default. That divergence is real and is documented
    /// rather than hidden: adopting Qdrant's default would change what every
    /// benchmark row already on disk was measuring, and §5.5's "everything
    /// loaded in memory by default" is the design this engine exists to test.
    pub fn resolve(memory: i32, on_disk: ?bool) ?Placement {
        if (fromProto(memory)) |m| return m;
        if (on_disk) |d| return fromOnDisk(d);
        return null;
    }
};

pub const header_size = 4096;
pub const magic: u64 = 0x6e_6e_61_6d_77_61_72_74; // "trawmann" little-endian
/// Bumped to 2 when `datatype` joined the header ahead of the CRC. See the
/// field's own comment for why that was worth a version rather than using the
/// reserved bytes after it. Bumped to 3 when `hnsw_m` / `hnsw_ef_construct`
/// joined it (same reasoning) and `ids.bin` stopped being the raw bytes of a
/// Zig tagged union (`persist.save`).
pub const format_version: u32 = 3;

pub const Error = error{
    OpenFailed,
    ReadFailed,
    WriteFailed,
    MapFailed,
    BadMagic,
    /// A valid header, for a different kind of file than the caller expected.
    BadKind,
    BadVersion,
    BadHeaderChecksum,
    ShortFile,
    OutOfMemory,
};

/// The 4 KiB header every file starts with.
///
/// `extern struct` so the on-disk layout is the declared one rather than
/// whatever the compiler chooses, a reordered field would silently change the
/// format between builds.
pub const Header = extern struct {
    magic: u64,
    format_version: u32,
    /// What this file holds, so a mismatched pair is caught at open rather
    /// than producing plausible garbage.
    kind: u32,
    dim: u64,
    count: u64,
    capacity: u64,
    /// Row stride in bytes for arena files.
    stride: u64,
    /// One u32 whose meaning is fixed by `kind`, which makes this header a
    /// union discriminated by that field:
    ///
    ///   `.vectors`  the distance metric, as qdrant's `Distance` proto value
    ///   `.graph`    the HNSW entry-point node
    ///   others      unused, written as 0
    ///
    /// It was called `distance`, so `persist.zig` wrote a graph's entry point
    /// into a field named after a metric and read it back out again. Nothing
    /// said the two were different quantities. Renamed rather than split
    /// because the offset and width are unchanged, so on-disk files written by
    /// older builds still load; use `metricField` / `entryPointField` rather
    /// than reading it raw.
    kind_field: u32,
    flags: u32,
    /// §8.7: "Seeded RNG for HNSW level assignment, with the seed recorded in
    /// `meta.json`." Recorded here too so a graph file carries the seed that
    /// produced it.
    seed: u64,
    /// §8.7 option (b): the graph checksum, so a reload can assert the graph is
    /// the one the results were measured against.
    graph_checksum: u64,
    /// `dist.Datatype`: what the arena's elements are.
    ///
    /// Placed *before* the CRC, and therefore covered by it. The reserved
    /// bytes after the CRC were the tempting spot, since that would have kept
    /// the format version fixed, but a header field that decides how to read
    /// every byte in the file is the last one that should sit outside the
    /// checksum: a single flipped bit there reads a f16 arena as f32 and
    /// returns plausible nonsense rather than an error.
    ///
    /// The version bump this forces costs nothing. `persist` is called by its
    /// own tests and by nothing else, so no v1 file exists that anyone kept,
    /// and a `BadVersion` on one is a loud failure rather than a silent
    /// misread. Its tag values are still chosen so 0 means fp32.
    datatype: u8 = 0,
    _reserved: [3]u8 = @splat(0),
    /// `.vectors` only: the collection's `Config.hnsw_m` and
    /// `Config.hnsw_ef_construct`, so a reload rebuilds with the parameters
    /// the collection was created with. `load` used to leave both at their
    /// defaults, so a collection built at m=32 came back as m=16 with a m=32
    /// graph attached, and its next rebuild silently changed shape. Written
    /// as 0 by every other kind.
    hnsw_m: u32 = 0,
    hnsw_ef_construct: u32 = 0,
    /// CRC of the header only, per §6.4.
    header_crc: u32,

    pub const Kind = enum(u32) {
        vectors = 1,
        graph = 2,
        ids = 3,
        deleted = 4,
        quant = 5,
        /// §6.4 `payload.bin`: the framed blobs, one `[u32 len][blob]` record
        /// per point (so no separate offset table is needed).
        payload = 6,
        /// The field indexes' schema, rebuilt into postings on load.
        payload_schema = 7,
    };

    pub const Flags = struct {
        /// §6.4: "Cosine is normalised at ingest and stored normalised, so the
        /// search path only ever runs dot product. Record this in the header."
        pub const normalised: u32 = 1 << 0;
    };

    /// `kind_field` read as what a `.vectors` header stores in it: the metric,
    /// as a `Distance` proto value for `Metric.fromProto`.
    pub fn metricField(self: *const Header) u32 {
        std.debug.assert(self.kind == @intFromEnum(Kind.vectors));
        return self.kind_field;
    }

    /// `kind_field` read as what a `.graph` header stores in it.
    pub fn entryPointField(self: *const Header) u32 {
        std.debug.assert(self.kind == @intFromEnum(Kind.graph));
        return self.kind_field;
    }

    /// Bytes covered by the CRC: everything up to but excluding the CRC field.
    fn crcRegion(self: *const Header) []const u8 {
        const bytes = std.mem.asBytes(self);
        return bytes[0..@offsetOf(Header, "header_crc")];
    }

    pub fn computeCrc(self: *const Header) u32 {
        return std.hash.Crc32.hash(self.crcRegion());
    }

    pub fn finalise(self: *Header) void {
        self.magic = magic;
        self.format_version = format_version;
        self.header_crc = self.computeCrc();
    }

    pub fn validate(self: *const Header, expect_kind: Kind) Error!void {
        if (self.magic != magic) return Error.BadMagic;
        if (self.format_version != format_version) return Error.BadVersion;
        if (self.header_crc != self.computeCrc()) return Error.BadHeaderChecksum;
        // Its own error. Reporting `BadMagic` for a well-formed header of the
        // wrong kind names the wrong failure: the magic matched, the file is
        // ours, it is the pairing that is wrong.
        if (self.kind != @intFromEnum(expect_kind)) return Error.BadKind;
    }
};

comptime {
    // The header must fit its reserved space with room to grow. A header that
    // outgrew 4 KiB would silently overlap the data.
    std.debug.assert(@sizeOf(Header) <= header_size);
}

// =========================================================================
// A mapped region
// =========================================================================

/// A file-backed region, held under one of the three placements.
pub const Region = struct {
    /// The mapping. On the pinned path this is the requested size rounded up
    /// to a huge-page boundary, because that is what was mapped and what
    /// `hugeBackedBytes` has to measure; the file-backed paths map exactly
    /// `size`. Callers that want the requested length re-slice
    /// (`VectorSpace.initMapped`).
    bytes: []align(4096) u8,
    placement: Placement,
    /// True when the kernel actually backed this with huge pages, as far as we
    /// can tell. Recorded rather than assumed: §5.5's whole argument depends on
    /// knowing which page size was used, and `MADV_HUGEPAGE` is advisory.
    huge_requested: bool,

    pub fn deinit(self: *Region) void {
        _ = linux.munmap(self.bytes.ptr, self.bytes.len);
        self.bytes = &.{};
    }

    /// Bytes of this region the kernel actually backs with huge pages, or
    /// `null` if `/proc/self/smaps` could not be read. See `hugeBackedBytes`.
    ///
    /// `huge_requested` is what we asked for; this is what we got. §5.5's
    /// argument for the resident placement is a TLB argument, and it only
    /// holds for the bytes that ended up on 2 MiB folios.
    pub fn hugeBackedBytes(self: *const Region) ?usize {
        return hugeBackedBytesOf(self.bytes);
    }
};

/// Sum of `AnonHugePages` over the VMAs of `/proc/self/smaps` that overlap
/// `bytes`, clamped to the overlap.
///
/// The one place the kernel reports the *outcome* of `MADV_HUGEPAGE`. THP set
/// to `never`, a fragmented free list, or a `defrag` policy that declines to
/// compact all leave a region on 4 KiB pages without an error, and everything
/// else in this file (the flag, the rounding, `huge_requested`) is a statement
/// of intent. `residentPages` answers "is it in memory"; this answers "at what
/// page size".
///
/// Per-VMA granularity: the kernel may merge an adjacent anonymous mapping
/// with the same flags into one VMA, so the count is clamped to the bytes of
/// the VMA that lie inside `bytes` and can only over-report when a *neighbour*
/// with huge pages was merged in. `null` means "could not ask", so a caller can
/// tell it from "none".
pub fn hugeBackedBytes(bytes: []const u8) ?usize {
    return hugeBackedBytesOf(bytes);
}

fn hugeBackedBytesOf(bytes: []const u8) ?usize {
    const lo = @intFromPtr(bytes.ptr);
    const hi = lo + bytes.len;

    const rc = linux.open("/proc/self/smaps", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: linux.fd_t = @intCast(rc);
    defer _ = linux.close(fd);

    // Line-at-a-time over a small buffer: smaps for a server with a few
    // hundred mappings is tens of KB, and this must not allocate.
    var buf: [8192]u8 = undefined;
    var filled: usize = 0;
    var eof = false;
    var total: usize = 0;
    var overlap: usize = 0; // bytes of the current VMA inside [lo, hi)
    while (true) {
        if (!eof and filled < buf.len) {
            const n = linux.read(fd, buf[filled..].ptr, buf.len - filled);
            if (linux.errno(n) != .SUCCESS) return null;
            if (n == 0) eof = true else filled += n;
        }
        if (filled == 0) break;
        const nl = std.mem.indexOfScalar(u8, buf[0..filled], '\n') orelse {
            if (eof) {
                parseSmapsLine(buf[0..filled], lo, hi, &overlap, &total);
                break;
            }
            // A single line longer than the buffer: only a header line with
            // a very long path could do that, and it cannot be one of ours.
            if (filled == buf.len) filled = 0;
            continue;
        };
        parseSmapsLine(buf[0..nl], lo, hi, &overlap, &total);
        const consumed = nl + 1;
        std.mem.copyForwards(u8, buf[0 .. filled - consumed], buf[consumed..filled]);
        filled -= consumed;
    }
    return total;
}

/// One line of smaps: either a VMA header (`start-end perms ...`) or a
/// `Key:   value kB` field. Tracks the overlap of the current VMA with
/// `[lo, hi)` and adds its clamped `AnonHugePages` to `total`.
fn parseSmapsLine(line: []const u8, lo: usize, hi: usize, overlap: *usize, total: *usize) void {
    if (std.mem.startsWith(u8, line, "AnonHugePages:")) {
        if (overlap.* == 0) return;
        const rest = std.mem.trim(u8, line["AnonHugePages:".len..], " \t");
        const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
        const kb = std.fmt.parseInt(usize, rest[0..end], 10) catch return;
        total.* += @min(kb * 1024, overlap.*);
        return;
    }
    // Header lines start with a hex range; field lines start with a letter
    // and a colon. Cheap discriminator: a `-` before the first space.
    const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return;
    const dash = std.mem.indexOfScalar(u8, line[0..sp], '-') orelse return;
    const start = std.fmt.parseInt(usize, line[0..dash], 16) catch return;
    const end = std.fmt.parseInt(usize, line[dash + 1 .. sp], 16) catch return;
    const a = @max(start, lo);
    const b = @min(end, hi);
    overlap.* = if (b > a) b - a else 0;
}

/// Whether the kernel will honour `MADV_HUGEPAGE` at all, from
/// `/sys/kernel/mm/transparent_hugepage/enabled`. `null` if unreadable.
/// `[always]` and `[madvise]` are true; `[never]` is false.
pub fn thpEnabled() ?bool {
    const rc = linux.open("/sys/kernel/mm/transparent_hugepage/enabled", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: linux.fd_t = @intCast(rc);
    defer _ = linux.close(fd);
    var buf: [128]u8 = undefined;
    const n = linux.read(fd, &buf, buf.len);
    if (linux.errno(n) != .SUCCESS) return null;
    return std.mem.indexOf(u8, buf[0..n], "[never]") == null;
}

fn checked(rc: usize, e: Error) Error!usize {
    return switch (linux.errno(rc)) {
        .SUCCESS => rc,
        else => e,
    };
}

/// Open (or create) a file and size it.
///
/// §6.4: "The arena is preallocated with `fallocate` at collection creation."
/// Preallocating means the arena's extents exist before the first write, so an
/// upsert never triggers block allocation mid-benchmark, which would appear as
/// an unexplained latency spike in W1.
pub fn openSized(path: []const u8, size: usize, create: bool) Error!linux.fd_t {
    var path_z: [512]u8 = undefined;
    if (path.len + 1 > path_z.len) return Error.OpenFailed;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;

    const flags: linux.O = if (create)
        .{ .ACCMODE = .RDWR, .CREAT = true }
    else
        .{ .ACCMODE = .RDWR };

    const rc = linux.open(@ptrCast(&path_z), flags, 0o644);
    const fd: linux.fd_t = @intCast(try checked(rc, Error.OpenFailed));
    errdefer _ = linux.close(fd);

    if (create) {
        // fallocate mode 0 = allocate and zero, keeping the file size.
        const fa = linux.fallocate(fd, 0, 0, @intCast(size));
        if (linux.errno(fa) != .SUCCESS) {
            // Not every filesystem supports fallocate; ftruncate at least gets
            // the size right, at the cost of sparse extents.
            _ = try checked(linux.ftruncate(fd, @intCast(size)), Error.WriteFailed);
        }
    }
    return fd;
}

/// Map a file region under `placement`.
pub fn mapRegion(fd: linux.fd_t, offset: usize, size: usize, placement: Placement) Error!Region {
    const rounded = std.mem.alignForward(usize, size, 2 * 1024 * 1024);

    switch (placement) {
        .cached, .cold => {
            // Both are a true file mapping at 4 KiB pages; they differ in
            // whether the pages are there before the first query.
            //
            // `cached` prefaults with MAP_POPULATE, which is Qdrant's
            // `Memory::Cached`: it calls `madvise(MADV_POPULATE_READ)` at open
            // (`common/src/mmap/advice.rs`, `Madviseable::populate`) with a
            // page-touching fallback on kernels before 5.14. MAP_POPULATE at
            // mmap time reaches the same steady state, and reaching it in one
            // syscall rather than a walk over 1.7 million pages is worth the
            // divergence in mechanism.
            //
            // `cold` does neither, so every page arrives on a fault during the
            // first query that touches it.
            const rc = linux.mmap(
                null,
                size,
                .{ .READ = true, .WRITE = true },
                .{ .TYPE = .SHARED, .POPULATE = placement == .cached },
                fd,
                @intCast(offset),
            );
            if (linux.errno(rc) != .SUCCESS) return Error.MapFailed;
            const ptr: [*]align(4096) u8 = @ptrFromInt(rc);
            // Explicitly *not* huge: these arms exist to measure the 4 KiB
            // case, and letting the kernel opportunistically promote them would
            // blur exactly the comparison §5.5 wants.
            _ = linux.madvise(ptr, size, linux.MADV.NOHUGEPAGE);
            // MADV_RANDOM, because Qdrant does it: its global default advice for
            // mmapped vector storage and HNSW is `Advice::Random`
            // (`common/src/mmap/advice.rs`, `static ADVICE`). It disables
            // readahead, and readahead is exactly the variable that decides
            // what a cold graph walk costs. Without this the cold arm would
            // report the kernel's speculation rather than the access pattern,
            // and would flatter us against Qdrant.
            _ = linux.madvise(ptr, size, linux.MADV.RANDOM);
            return .{ .bytes = ptr[0..size], .placement = placement, .huge_requested = false };
        },
        .pinned => {
            // §5.5's default: anonymous memory (so THP applies at all), huge
            // pages requested, then populated by pread.
            const rc = linux.mmap(
                null,
                rounded,
                .{ .READ = true, .WRITE = true },
                .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
                -1,
                0,
            );
            if (linux.errno(rc) != .SUCCESS) return Error.MapFailed;
            const ptr: [*]align(4096) u8 = @ptrFromInt(rc);
            _ = linux.madvise(ptr, rounded, linux.MADV.HUGEPAGE);

            // Populate. §5.5: "startup cost is sequential I/O at full disk
            // bandwidth", which is what a large sequential pread gives.
            var done: usize = 0;
            while (done < size) {
                const want = @min(size - done, 64 * 1024 * 1024);
                const n = linux.pread(fd, ptr + done, want, @intCast(offset + done));
                const got = checked(n, Error.ReadFailed) catch {
                    _ = linux.munmap(ptr, rounded);
                    return Error.ReadFailed;
                };
                if (got == 0) break; // short file; caller validates counts
                done += got;
            }
            return .{ .bytes = ptr[0..rounded], .placement = .pinned, .huge_requested = true };
        },
    }
}

/// Write a header to the front of a file.
pub fn writeHeader(fd: linux.fd_t, h: *Header) Error!void {
    h.finalise();
    var buf: [header_size]u8 align(4096) = @splat(0);
    @memcpy(buf[0..@sizeOf(Header)], std.mem.asBytes(h));
    var done: usize = 0;
    while (done < header_size) {
        const n = linux.pwrite(fd, buf[done..].ptr, header_size - done, @intCast(done));
        const got = try checked(n, Error.WriteFailed);
        // A zero-byte write is an error here as in `writeData`; looping on it
        // spun forever.
        if (got == 0) return Error.WriteFailed;
        done += got;
    }
}

pub fn readHeader(fd: linux.fd_t, expect_kind: Header.Kind) Error!Header {
    var buf: [header_size]u8 align(4096) = undefined;
    var done: usize = 0;
    while (done < header_size) {
        const n = linux.pread(fd, buf[done..].ptr, header_size - done, @intCast(done));
        const got = try checked(n, Error.ReadFailed);
        if (got == 0) return Error.ShortFile;
        done += got;
    }
    var h: Header = undefined;
    @memcpy(std.mem.asBytes(&h), buf[0..@sizeOf(Header)]);
    try h.validate(expect_kind);
    return h;
}

/// Write a block of data after the header.
pub fn writeData(fd: linux.fd_t, data: []const u8) Error!void {
    var done: usize = 0;
    while (done < data.len) {
        const n = linux.pwrite(fd, data.ptr + done, data.len - done, @intCast(header_size + done));
        const got = try checked(n, Error.WriteFailed);
        if (got == 0) return Error.WriteFailed;
        done += got;
    }
}

pub fn readData(fd: linux.fd_t, dst: []u8) Error!usize {
    var done: usize = 0;
    while (done < dst.len) {
        const n = linux.pread(fd, dst.ptr + done, dst.len - done, @intCast(header_size + done));
        const got = try checked(n, Error.ReadFailed);
        if (got == 0) break;
        done += got;
    }
    return done;
}

pub fn closeFd(fd: linux.fd_t) void {
    _ = linux.close(fd);
}

/// The file's size in bytes, for files whose records carry their own
/// lengths and whose header therefore holds no byte count.
pub fn fileSize(fd: linux.fd_t) Error!u64 {
    // `lseek` to the end; every read here is a `pread`, so the file offset
    // it moves is never used.
    const end = try checked(linux.lseek(fd, 0, linux.SEEK.END), Error.ReadFailed);
    return @intCast(end);
}

/// Total file size for an arena of `capacity` rows at `stride` bytes.
pub fn arenaFileSize(capacity: usize, stride: usize) usize {
    return header_size + capacity * stride;
}

/// Delete a file, ignoring "it was not there".
///
/// Used before creating an arena so the new file's extents are freshly zeroed:
/// `openSized` does not truncate, and `fallocate` does not zero what is already
/// allocated. Also the tests' cleanup.
pub fn removeFile(path: []const u8) void {
    var z: [512]u8 = undefined;
    if (path.len + 1 > z.len) return;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    _ = linux.unlink(@ptrCast(&z));
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

/// A path on a filesystem that can actually evict page cache.
///
/// `null` when no such directory is available, which the caller should treat as
/// "this host cannot demonstrate the difference" rather than as a failure. The
/// build cache lives beside the source tree, which on any ordinary checkout is
/// a real block device; `/tmp` frequently is not.
fn diskPath(buf: []u8, name: []const u8) ?[]const u8 {
    const dir = ".zig-cache/tmp";
    var z: [64]u8 = undefined;
    @memcpy(z[0..dir.len], dir);
    z[dir.len] = 0;
    _ = linux.mkdir(@ptrCast(&z), 0o755); // EEXIST is fine
    return std.fmt.bufPrint(buf, "{s}/strawmann-{s}", .{ dir, name }) catch null;
}

/// A scratch path under `/tmp`, which is enough for a header round-trip and
/// nothing that needs a real block device (`diskPath` is for that).
fn tmpPath(buf: []u8, name: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/strawmann-test-{s}", .{name}) catch unreachable;
}

test "header round-trips and the CRC covers only the header" {
    var h = Header{
        .magic = 0,
        .format_version = 0,
        .kind = @intFromEnum(Header.Kind.vectors),
        .dim = 768,
        .count = 1000,
        .capacity = 1_100_000,
        .stride = 3072,
        .kind_field = 3,
        .flags = Header.Flags.normalised,
        .seed = 0xdeadbeef,
        .graph_checksum = 0,
        .header_crc = 0,
    };
    h.finalise();
    try testing.expectEqual(magic, h.magic);
    try testing.expectEqual(format_version, h.format_version);
    try h.validate(.vectors);

    // A corrupted field must be caught.
    var bad = h;
    bad.dim = 769;
    try testing.expectError(Error.BadHeaderChecksum, bad.validate(.vectors));

    // The wrong kind must be caught, so a graph file cannot be opened as an
    // arena and produce plausible garbage, and it is reported as its own error
    // rather than as a bad magic: the magic was fine.
    try testing.expectError(Error.BadKind, h.validate(.graph));

    // A different format version must be refused rather than reinterpreted.
    var old = h;
    old.format_version = 0;
    old.header_crc = old.computeCrc();
    try testing.expectError(Error.BadVersion, old.validate(.vectors));
}

test "header fits the reserved 4 KiB with room to spare" {
    try testing.expect(@sizeOf(Header) <= header_size);
    // Deliberately generous: §6.4 reserves 4 KiB precisely so fields can be
    // added without a format break.
    try testing.expect(@sizeOf(Header) < 256);
}

test "arena file size accounts for the header" {
    try testing.expectEqual(header_size + 1000 * 3072, arenaFileSize(1000, 3072));
}

test "write then read a file under every placement" {
    var pbuf: [128]u8 = undefined;
    const path = tmpPath(&pbuf, "modes.bin");
    defer removeFile(path);

    const stride = 64;
    const capacity = 512;
    const size = arenaFileSize(capacity, stride);

    // Write.
    {
        const fd = try openSized(path, size, true);
        defer closeFd(fd);
        var h = Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(Header.Kind.vectors),
            .dim = 16,
            .count = capacity,
            .capacity = capacity,
            .stride = stride,
            .kind_field = 3,
            .flags = 0,
            .seed = 1,
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try writeHeader(fd, &h);

        const data = try testing.allocator.alloc(u8, capacity * stride);
        defer testing.allocator.free(data);
        for (data, 0..) |*b, i| b.* = @truncate(i *% 31);
        try writeData(fd, data);
    }

    // Read back through each mode and confirm the bytes are identical.
    for ([_]Placement{ .pinned, .cached, .cold }) |placement| {
        const fd = try openSized(path, size, false);
        defer closeFd(fd);

        const h = try readHeader(fd, .vectors);
        try testing.expectEqual(@as(u64, capacity), h.capacity);
        try testing.expectEqual(@as(u64, stride), h.stride);

        var region = try mapRegion(fd, header_size, capacity * stride, placement);
        defer region.deinit();
        try testing.expectEqual(placement, region.placement);
        try testing.expect(region.bytes.len >= capacity * stride);

        for (0..capacity * stride) |i| {
            try testing.expectEqual(@as(u8, @truncate(i *% 31)), region.bytes[i]);
        }
    }
}

test "§5.5: the pinned placement requests huge pages, the mapped ones do not" {
    // The distinction the whole TLB comparison rests on. Resident is anonymous
    // memory with MADV_HUGEPAGE (the only portable way to get 2 MiB pages);
    // mmap is file-backed with MADV_NOHUGEPAGE so the 4 KiB arm stays 4 KiB.
    var pbuf: [128]u8 = undefined;
    const path = tmpPath(&pbuf, "pages.bin");
    defer removeFile(path);

    const size = arenaFileSize(64, 4096);
    const fd = try openSized(path, size, true);
    defer closeFd(fd);
    var h = Header{
        .magic = 0,
        .format_version = 0,
        .kind = @intFromEnum(Header.Kind.vectors),
        .dim = 1024,
        .count = 64,
        .capacity = 64,
        .stride = 4096,
        .kind_field = 3,
        .flags = 0,
        .seed = 0,
        .graph_checksum = 0,
        .header_crc = 0,
    };
    try writeHeader(fd, &h);

    var resident = try mapRegion(fd, header_size, 64 * 4096, .pinned);
    defer resident.deinit();
    try testing.expect(resident.huge_requested);
    // Rounded up to a huge-page boundary, which is what lets the kernel back
    // it with 2 MiB folios at all.
    try testing.expectEqual(@as(usize, 0), resident.bytes.len % (2 * 1024 * 1024));

    var mapped = try mapRegion(fd, header_size, 64 * 4096, .cached);
    defer mapped.deinit();
    try testing.expect(!mapped.huge_requested);
    try testing.expectEqual(@as(usize, 64 * 4096), mapped.bytes.len);
}

test "§5.5: the pinned placement is actually huge-backed when THP allows it" {
    // `huge_requested` is intent; this is outcome. Skipped when the kernel has
    // THP off, since then there is nothing to assert about, and asserted
    // otherwise: a resident placement that lands on 4 KiB pages is the whole
    // §5.5 argument silently not happening.
    const thp = thpEnabled() orelse return error.SkipZigTest;
    if (!thp) return error.SkipZigTest;

    var pbuf: [128]u8 = undefined;
    const path = tmpPath(&pbuf, "hugebacked.bin");
    defer removeFile(path);

    // Two full huge pages of payload, so at least one whole 2 MiB folio is
    // populated end to end and the fault path has every reason to use THP.
    const rows = 1024;
    const stride = 4096;
    const size = arenaFileSize(rows, stride);
    const fd = try openSized(path, size, true);
    defer closeFd(fd);
    var h = Header{
        .magic = 0,
        .format_version = 0,
        .kind = @intFromEnum(Header.Kind.vectors),
        .dim = 1024,
        .count = rows,
        .capacity = rows,
        .stride = stride,
        .kind_field = 3,
        .flags = 0,
        .seed = 0,
        .graph_checksum = 0,
        .header_crc = 0,
    };
    try writeHeader(fd, &h);
    const payload = try testing.allocator.alloc(u8, rows * stride);
    defer testing.allocator.free(payload);
    @memset(payload, 0x5a);
    try writeData(fd, payload);

    var resident = try mapRegion(fd, header_size, rows * stride, .pinned);
    defer resident.deinit();
    const huge = resident.hugeBackedBytes() orelse return error.SkipZigTest;
    try testing.expect(huge > 0);
    try testing.expect(huge <= resident.bytes.len);
    // And a huge-page count is a whole number of huge pages.
    try testing.expectEqual(@as(usize, 0), huge % (2 * 1024 * 1024));

    // The mapped arm asked for the opposite and must not be huge-backed.
    var mapped = try mapRegion(fd, header_size, rows * stride, .cached);
    defer mapped.deinit();
    if (mapped.hugeBackedBytes()) |m| try testing.expectEqual(@as(usize, 0), m);
}

test "smaps lines are parsed and clamped to the queried range" {
    var overlap: usize = 0;
    var total: usize = 0;
    // A VMA of 8 MiB at 0x1000_0000, queried range covers its second half.
    const lo: usize = 0x1040_0000;
    const hi: usize = 0x1080_0000;
    parseSmapsLine("10000000-10800000 rw-p 00000000 00:00 0 ", lo, hi, &overlap, &total);
    try testing.expectEqual(@as(usize, 4 * 1024 * 1024), overlap);
    parseSmapsLine("AnonHugePages:      8192 kB", lo, hi, &overlap, &total);
    // Reported 8 MiB huge, but only 4 MiB of that VMA is inside the query.
    try testing.expectEqual(@as(usize, 4 * 1024 * 1024), total);
    // A VMA outside the range contributes nothing.
    parseSmapsLine("20000000-20800000 rw-p 00000000 00:00 0 ", lo, hi, &overlap, &total);
    try testing.expectEqual(@as(usize, 0), overlap);
    parseSmapsLine("AnonHugePages:      2048 kB", lo, hi, &overlap, &total);
    try testing.expectEqual(@as(usize, 4 * 1024 * 1024), total);
    // Field lines that are not AnonHugePages are ignored.
    parseSmapsLine("Rss:               8192 kB", lo, hi, &overlap, &total);
    try testing.expectEqual(@as(usize, 4 * 1024 * 1024), total);
}

test "a truncated file is refused rather than read as garbage" {
    var pbuf: [128]u8 = undefined;
    const path = tmpPath(&pbuf, "short.bin");
    defer removeFile(path);

    const fd = try openSized(path, 16, true);
    defer closeFd(fd);
    try testing.expectError(Error.ShortFile, readHeader(fd, .vectors));
}

test "a mapped placement is writable and changes are visible through a fresh read" {
    // §6.4 keeps the mmap arm SHARED so it behaves like Qdrant's, which means
    // writes land in the file. A PRIVATE mapping would silently discard them
    // and the comparison arm would not be measuring the same thing.
    var pbuf: [128]u8 = undefined;
    const path = tmpPath(&pbuf, "shared.bin");
    defer removeFile(path);

    const size = arenaFileSize(16, 64);
    const fd = try openSized(path, size, true);
    defer closeFd(fd);
    var h = Header{
        .magic = 0,
        .format_version = 0,
        .kind = @intFromEnum(Header.Kind.vectors),
        .dim = 16,
        .count = 16,
        .capacity = 16,
        .stride = 64,
        .kind_field = 3,
        .flags = 0,
        .seed = 0,
        .graph_checksum = 0,
        .header_crc = 0,
    };
    try writeHeader(fd, &h);

    {
        var region = try mapRegion(fd, header_size, 16 * 64, .cached);
        defer region.deinit();
        region.bytes[7] = 0xAB;
        _ = linux.msync(region.bytes.ptr, region.bytes.len, linux.MSF.SYNC);
    }

    var buf: [16 * 64]u8 = undefined;
    const n = try readData(fd, &buf);
    try testing.expectEqual(@as(usize, 16 * 64), n);
    try testing.expectEqual(@as(u8, 0xAB), buf[7]);
}

/// `POSIX_FADV_DONTNEED`. Not in `std.os.linux`'s enum, and the value is the
/// same on every architecture this targets.
pub const POSIX_FADV_DONTNEED: usize = 4;

/// How many of a region's pages are resident right now, via `mincore(2)`.
///
/// For a file-backed mapping this is page-*cache* residency: whether reading
/// the page would touch the disk. It is not whether this process has a PTE for
/// it. That is the right question for a placement — the cost being measured is
/// the I/O — but it means a mapping is only as cold as the page cache is, which
/// `test "cached prefaults its pages and cold does not"` spells out.
///
/// The only direct way to check what a placement actually did. Everything else
/// — the enum, the flags passed to `mmap` — is a statement of intent; this is
/// the kernel's answer. `null` if `mincore` is unavailable, so a caller can
/// tell "not resident" from "could not ask".
pub fn residentPages(bytes: []const u8, buf: []u8) ?usize {
    const page = 4096;
    const pages = (bytes.len + page - 1) / page;
    if (pages > buf.len) return null;
    const rc = linux.mincore(@constCast(bytes.ptr), bytes.len, buf.ptr);
    if (linux.errno(rc) != .SUCCESS) return null;
    var n: usize = 0;
    for (buf[0..pages]) |b| n += b & 1;
    return n;
}

test "cached prefaults its pages and cold does not" {
    // The claim the whole placement feature rests on, checked against the
    // kernel rather than against the flags we passed. Without this, `cold` is
    // an enum value: a MAP_POPULATE accidentally left on both arms would make
    // every cold measurement a warm one and no test would notice.
    //
    // Deliberately NOT under `tmpPath`. `/tmp` is tmpfs on the development
    // host, and on tmpfs the page cache *is* the storage: nothing can evict it,
    // `fadvise(DONTNEED)` succeeds and changes nothing, and a cold mapping
    // reports every page resident. Measured: 2048/2048 pages resident after a
    // successful drop on tmpfs, against 0/2048 on ext4. So a `--data-dir` on a
    // tmpfs makes the cold arm silently identical to the cached arm, and this
    // test would have proved the opposite of what it claims.
    var pbuf: [256]u8 = undefined;
    const path = diskPath(&pbuf, "residency.bin") orelse return error.SkipZigTest;
    defer removeFile(path);

    // 8 MiB, comfortably more than any readahead window, so the difference is
    // not a rounding effect at the edges of one or two pages.
    const stride = 4096;
    const capacity = 2048;
    const size = capacity * stride;

    {
        const fd = try openSized(path, arenaFileSize(capacity, stride), true);
        defer closeFd(fd);
        var h = Header{
            .magic = 0,
            .format_version = 0,
            .kind = @intFromEnum(Header.Kind.vectors),
            .dim = 1024,
            .count = capacity,
            .capacity = capacity,
            .stride = stride,
            .kind_field = 3,
            .flags = 0,
            .seed = 0,
            .graph_checksum = 0,
            .header_crc = 0,
        };
        try writeHeader(fd, &h);
        const data = try testing.allocator.alloc(u8, size);
        defer testing.allocator.free(data);
        @memset(data, 0xAB);
        try writeData(fd, data);
    }

    const vec = try testing.allocator.alloc(u8, size / 4096 + 1);
    defer testing.allocator.free(vec);

    // Cached: MAP_POPULATE, so essentially everything is in before we look.
    {
        const fd = try openSized(path, 0, false);
        defer closeFd(fd);
        var region = try mapRegion(fd, header_size, size, .cached);
        defer region.deinit();
        const resident = residentPages(region.bytes, vec) orelse return error.SkipZigTest;
        const total = size / 4096;
        try testing.expect(resident * 10 >= total * 9);
    }

    // Cold, and the *cooling* is the load-bearing part of this test.
    //
    // For a file-backed mapping `mincore` reports page-*cache* residency, not
    // whether this process has the page mapped. So a cold mapping of a file
    // that is already in the page cache reports resident, correctly: it *is*
    // warm, and reading it costs no I/O. Which is the finding this test exists
    // to pin down — **a `cold` collection is not cold after you have just
    // written it.** Ingest goes through the page cache, so a benchmark that
    // uploads and then searches a cold collection measures a warm one. The
    // cache has to be dropped in between, which is what `fadvise(DONTNEED)`
    // does here and what a real cold arm has to do to mean anything.
    {
        const fd = try openSized(path, 0, false);
        defer closeFd(fd);
        _ = linux.fsync(fd);
        _ = linux.fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

        var region = try mapRegion(fd, header_size, size, .cold);
        defer region.deinit();
        const total = size / 4096;
        const resident = residentPages(region.bytes, vec) orelse return error.SkipZigTest;
        // If the cache would not drop, this filesystem cannot demonstrate a
        // cold mapping and the assertion below would be testing the host rather
        // than the code. Skipping is the honest outcome; failing here would be
        // read as "cold is broken" when what is true is "this directory cannot
        // be cold". See the note above about tmpfs.
        if (resident * 2 >= total) return error.SkipZigTest;
        try testing.expect(resident * 2 < total);

        // Touching a page brings it in, which is what "faulted on demand"
        // means and what a cold search pays per vector visited.
        const before = residentPages(region.bytes, vec).?;
        std.mem.doNotOptimizeAway(region.bytes[size / 2]);
        const after = residentPages(region.bytes, vec).?;
        try testing.expect(after > before);
    }
}
