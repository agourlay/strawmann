//! §7.2 layer 1, the hardware baseline.
//!
//!   "STREAM-equivalent and pointer-chase latency microbenchmarks written in
//!    Zig, plus Intel MLC for cross-validation. Produces the per-core and
//!    per-socket bandwidth/latency constants that §5's model consumes. Rerun on
//!    every new machine."
//!
//! Four measurements, each answering a specific line of §5:
//!
//!  1. **Sequential bandwidth**, the denominator of §5.1's roofline and of the
//!     machine-wide ceiling cross-check in §5.3 ("a 300 GB/s socket pair caps
//!     fp32 search at ~39k QPS regardless of core count").
//!
//!  2. **Random-access latency**, the `~85–100 ns of DRAM latency` §5.2
//!     assumes. Measured rather than assumed, because it is the term the whole
//!     per-query cost model multiplies through.
//!
//!  3. **Memory-level parallelism**, the single most important constant in the
//!     document. §5.2: "A core can sustain roughly 10–16 outstanding line fills
//!     (LFB/MSHR-limited)." Everything in §5.2's ns/vector table is derived from
//!     that number, and the binary row's "latency bound unless several vectors
//!     are in flight" is a direct consequence. This benchmark measures it by
//!     chasing K independent pointer chains and finding where throughput stops
//!     improving.
//!
//!  4. **TLB / page size**, §5.5's argument for the resident-mode design.
//!     "3 GB of vectors with 4 KiB pages is 786k pages. A core's dTLB holds
//!     ~1.5–3k entries across levels. Every random vector access is a TLB miss
//!     plus a page walk." The same chase is run over 4 KiB and 2 MiB backed
//!     memory; the delta is the TLB penalty, quantified as a first-class result.

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const strawmann = @import("strawmann");
const storage = strawmann.core.storage;
const perfctr = @import("perfctr.zig");

pub const line_size = 64;

/// Upper bound on the MLP sweep.
///
/// Deliberately far above §5.2's "10–16 outstanding line fills": the knee is
/// the measurement, so the sweep has to be able to overshoot it. A sweep that
/// stops at the expected answer can only ever confirm the expectation.
pub const max_sweep_chains = 256;
pub const huge_page = 2 * 1024 * 1024;

/// How a region is backed, so the TLB comparison of §5.5 is explicit rather
/// than ambient.
pub const Backing = enum {
    /// Ordinary anonymous pages, 4 KiB. The "mmap" comparison mode.
    small_pages,
    /// `MADV_HUGEPAGE` anonymous pages, 2 MiB where the kernel obliges. The
    /// "resident" default mode.
    huge_pages,

    pub fn name(self: Backing) []const u8 {
        return switch (self) {
            .small_pages => "4KiB",
            .huge_pages => "2MiB",
        };
    }
};

pub const Region = struct {
    bytes: []align(4096) u8,
    backing: Backing,

    pub fn alloc(size: usize, backing: Backing) !Region {
        // Round up to a huge-page boundary so the kernel can actually back the
        // whole region with 2 MiB folios; an unaligned tail silently falls back
        // to 4 KiB pages and would contaminate the comparison.
        const rounded = std.mem.alignForward(usize, size, huge_page);
        const mem = try posix.mmap(
            null,
            rounded,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        errdefer posix.munmap(mem);

        switch (backing) {
            .huge_pages => posix.madvise(mem.ptr, rounded, posix.MADV.HUGEPAGE) catch {},
            .small_pages => posix.madvise(mem.ptr, rounded, posix.MADV.NOHUGEPAGE) catch {},
        }

        // Fault every page in now. Without this the first pass measures page
        // faults rather than memory, and §7.1 asks for "an explicit warm-up
        // phase for steady-state runs".
        var i: usize = 0;
        while (i < rounded) : (i += 4096) mem[i] = 0;

        return .{ .bytes = mem[0..rounded], .backing = backing };
    }

    pub fn free(self: Region) void {
        posix.munmap(@alignCast(self.bytes));
    }

    pub fn asF32(self: Region) []f32 {
        return std.mem.bytesAsSlice(f32, self.bytes[0 .. self.bytes.len / 4 * 4]);
    }

    /// Bytes of this region the kernel actually backs with 2 MiB folios,
    /// from `/proc/self/smaps`. `null` when smaps cannot be read.
    ///
    /// `MADV_HUGEPAGE` is a request, and until now the bench reported the
    /// request as if it were the outcome: every "2 MiB pages" row was labelled
    /// from the flag passed to `madvise`, never from what the kernel did. THP
    /// set to `never`, a fragmented free list, or a `defrag` policy that
    /// declines to compact all leave the region on 4 KiB pages with no error,
    /// and then the TLB comparison compares 4 KiB against 4 KiB.
    pub fn hugeBackedBytes(self: Region) ?usize {
        return storage.hugeBackedBytes(self.bytes);
    }

    /// `hugeBackedBytes` as a fraction of the region, or `null`.
    pub fn hugeBackedFraction(self: Region) ?f64 {
        const h = self.hugeBackedBytes() orelse return null;
        if (self.bytes.len == 0) return 0;
        return @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(self.bytes.len));
    }
};

/// The threshold below which a huge-page region is reported with a warning:
/// a region mostly on 4 KiB pages is not measuring what its label says.
pub const huge_backed_warn_fraction = 0.90;

pub const BandwidthResult = struct {
    read_gbps: f64,
    triad_gbps: f64,
    bytes: usize,
};

/// Sequential read and STREAM triad bandwidth.
///
/// Read is a pure load stream; triad (`a[i] = b[i] + k·c[i]`) is the classic
/// STREAM kernel and moves 3 streams, which is closer to what an ingest path
/// does. Both are reported because the ratio between them says whether the
/// machine is limited by load ports or by DRAM.
pub fn bandwidth(region: Region, iters: usize) BandwidthResult {
    const f = region.asF32();
    const n = f.len;

    // --- pure read ---
    // XOR the stream into four independent 512-bit integer accumulators. The
    // earlier version summed into "8 scalar accumulators", which the
    // vectoriser folded into *one* 256-bit register and therefore one FADD
    // dependency chain: 4 cycles per 32 B, ~40 GB/s at 5 GHz, so the loop was
    // measuring the adder, not the bus. Integer XOR has one-cycle latency and
    // no rounding, and four accumulators leave the loop nowhere to serialise;
    // the question is how fast bytes arrive, not how fast we can add them.
    const V = @Vector(8, u64);
    const words = std.mem.bytesAsSlice(u64, region.bytes[0 .. region.bytes.len / 64 * 64]);
    var best_read_ns: u64 = std.math.maxInt(u64);
    for (0..iters) |_| {
        const t0 = perfctr.nowNs();
        var acc: [4]V = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var i: usize = 0;
        while (i + 32 <= words.len) : (i += 32) {
            inline for (0..4) |k| {
                const v: V = words[i + k * 8 ..][0..8].*;
                acc[k] ^= v;
            }
        }
        var s: u64 = 0;
        inline for (0..4) |k| s ^= @reduce(.Xor, acc[k]);
        std.mem.doNotOptimizeAway(s);
        const dt = perfctr.nowNs() - t0;
        best_read_ns = @min(best_read_ns, dt);
    }

    // --- STREAM triad over three thirds of the region ---
    const third = n / 3;
    const a = f[0..third];
    const b = f[third .. 2 * third];
    const c = f[2 * third .. 3 * third];
    var best_triad_ns: u64 = std.math.maxInt(u64);
    for (0..iters) |_| {
        const t0 = perfctr.nowNs();
        for (a, b, c) |*ai, bi, ci| ai.* = bi + 3.0 * ci;
        std.mem.doNotOptimizeAway(a.ptr);
        const dt = perfctr.nowNs() - t0;
        best_triad_ns = @min(best_triad_ns, dt);
    }

    const read_bytes = words.len / 32 * 32 * @sizeOf(u64);
    const triad_bytes = third * @sizeOf(f32) * 3;
    return .{
        .read_gbps = @as(f64, @floatFromInt(read_bytes)) / @as(f64, @floatFromInt(best_read_ns)),
        .triad_gbps = @as(f64, @floatFromInt(triad_bytes)) / @as(f64, @floatFromInt(best_triad_ns)),
        .bytes = read_bytes,
    };
}

/// Build `chains` independent random cycles over the region, one entry per
/// cache line, and return the chain head indices.
///
/// A single random cycle (rather than a shuffled array walked in order) is what
/// makes the chase genuinely dependent: the next address cannot be computed
/// until the current load returns, which defeats every hardware prefetcher and
/// is the access pattern HNSW traversal actually produces (§5.2: "a dependent
/// chain, you cannot know which vector to fetch next until the current
/// expansion has been scored").
fn buildChains(region: Region, chains: usize, rnd: std.Random) []usize {
    const slots = region.bytes.len / line_size;
    const idx = std.mem.bytesAsSlice(usize, region.bytes);

    // Partition the slots among `chains` cycles by interleaving, so every chain
    // spans the whole region rather than a contiguous slice of it.
    const per_chain = slots / chains;

    const perm = std.heap.page_allocator.alloc(usize, per_chain) catch unreachable;
    defer std.heap.page_allocator.free(perm);

    for (0..chains) |c| {
        for (0..per_chain) |i| perm[i] = c + i * chains;
        // Fisher-Yates over the chain's own slots.
        var i: usize = per_chain;
        while (i > 1) {
            i -= 1;
            const j = rnd.uintLessThan(usize, i + 1);
            std.mem.swap(usize, &perm[i], &perm[j]);
        }
        // Link them into a single cycle: perm[k] -> perm[k+1] -> ... -> perm[0].
        for (0..per_chain) |k| {
            const from = perm[k];
            const to = perm[(k + 1) % per_chain];
            idx[from * (line_size / @sizeOf(usize))] = to * (line_size / @sizeOf(usize));
        }
    }

    const heads = std.heap.page_allocator.alloc(usize, chains) catch unreachable;
    for (0..chains) |c| heads[c] = c * (line_size / @sizeOf(usize));
    return heads;
}

pub const LatencyResult = struct {
    ns_per_access: f64,
    accesses: usize,
    sample: perfctr.Sample,
};

/// Single-chain dependent-load latency: the DRAM latency constant of §5.2.
pub fn latency(region: Region, accesses: usize, rnd: std.Random) LatencyResult {
    const heads = buildChains(region, 1, rnd);
    defer std.heap.page_allocator.free(heads);
    const idx = std.mem.bytesAsSlice(usize, region.bytes);

    // Warm up the chain structure itself without warming the data.
    var p = heads[0];
    for (0..1024) |_| p = idx[p];

    const counters = perfctr.Counters.open();
    defer counters.close();

    counters.start();
    const t0 = perfctr.nowNs();
    p = heads[0];
    for (0..accesses) |_| p = idx[p];
    const dt = perfctr.nowNs() - t0;
    const sample = counters.stop();
    std.mem.doNotOptimizeAway(p);

    return .{
        .ns_per_access = @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(accesses)),
        .accesses = accesses,
        .sample = sample,
    };
}

pub const MlpPoint = struct {
    chains: usize,
    ns_per_access: f64,
    /// Aggregate random-access bandwidth implied by this point, in GB/s,
    /// counting one cache line per access. This is the number §5.2 models as
    /// `(12 lines × 64 B) / 90 ns ≈ 8.5 GB/s`.
    gbps: f64,
};

/// Sweep the number of independent chains to find where a core stops issuing
/// more outstanding misses.
///
/// The knee of this curve **is** the LFB/MSHR limit of §5.2, and therefore the
/// number that decides whether a given encoding is latency bound or bandwidth
/// bound. §5.2's whole table hangs off it, so it is measured, not assumed.
pub fn mlpSweep(region: Region, max_chains: usize, accesses_per_chain: usize, rnd: std.Random, out: []MlpPoint) []MlpPoint {
    var n: usize = 0;
    var chains: usize = 1;
    while (chains <= max_chains and n < out.len) : (chains *= 2) {
        const heads = buildChains(region, chains, rnd);
        defer std.heap.page_allocator.free(heads);
        const idx = std.mem.bytesAsSlice(usize, region.bytes);

        // Concurrent chains held in a small stack array so each is an
        // independent dependency chain the out-of-order engine can overlap.
        // The array itself stays L1-resident at every size swept here, so it
        // does not become the thing being measured.
        var p: [max_sweep_chains]usize = undefined;
        for (0..chains) |c| p[c] = heads[c];

        const t0 = perfctr.nowNs();
        for (0..accesses_per_chain) |_| {
            for (0..chains) |c| p[c] = idx[p[c]];
        }
        const dt = perfctr.nowNs() - t0;
        for (0..chains) |c| std.mem.doNotOptimizeAway(p[c]);

        const total = accesses_per_chain * chains;
        const ns_each = @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(total));
        out[n] = .{
            .chains = chains,
            .ns_per_access = ns_each,
            .gbps = @as(f64, line_size) / ns_each,
        };
        n += 1;
    }
    return out[0..n];
}

pub const TlbResult = struct {
    small: LatencyResult,
    huge: LatencyResult,
    /// Fraction of the "2 MiB" arm the kernel actually backed with huge
    /// pages, read from smaps *after* the run. Below
    /// `huge_backed_warn_fraction` the penalty ratio is not a page-size
    /// comparison and the reporter says so.
    huge_backed: ?f64,
    /// Same for the 4 KiB arm, which should be ~0: `MADV_NOHUGEPAGE` is also
    /// only advice.
    small_huge_backed: ?f64,

    /// The §5.5 headline: how much slower random access is on 4 KiB pages.
    pub fn penaltyRatio(self: TlbResult) f64 {
        return self.small.ns_per_access / self.huge.ns_per_access;
    }

    /// dTLB load misses per access on each backing, when the counter is
    /// available. §7.3 names `dtlb_load_misses.walk_active` specifically; the
    /// portable `dTLB-load-misses` event is what a `PERF_TYPE_HW_CACHE` counter
    /// can express and is close enough to show the effect.
    pub fn missesPerAccess(self: TlbResult) struct { small: ?f64, huge: ?f64 } {
        return .{
            .small = if (self.small.sample.dtlb_load_misses) |m|
                @as(f64, @floatFromInt(m)) / @as(f64, @floatFromInt(self.small.accesses))
            else
                null,
            .huge = if (self.huge.sample.dtlb_load_misses) |m|
                @as(f64, @floatFromInt(m)) / @as(f64, @floatFromInt(self.huge.accesses))
            else
                null,
        };
    }
};

/// Run the same dependent chase over 4 KiB and 2 MiB backed memory.
///
/// §5.5 predicts a substantial gap and calls quantifying it "one of the more
/// actionable findings for the Qdrant side", because Qdrant's file-backed mmap
/// cannot generally get 2 MiB mappings: "THP is an anonymous-memory feature;
/// file-backed large folios are filesystem- and kernel-dependent and cannot be
/// assumed."
pub fn tlbComparison(size: usize, accesses: usize, rnd: std.Random) !TlbResult {
    const small = try Region.alloc(size, .small_pages);
    defer small.free();
    const small_r = latency(small, accesses, rnd);
    const small_frac = small.hugeBackedFraction();

    const huge = try Region.alloc(size, .huge_pages);
    defer huge.free();
    const huge_r = latency(huge, accesses, rnd);
    const huge_frac = huge.hugeBackedFraction();

    return .{ .small = small_r, .huge = huge_r, .huge_backed = huge_frac, .small_huge_backed = small_frac };
}

// -------------------------------------------------------------------------
// CPU pinning and cache topology
// -------------------------------------------------------------------------

/// The CPU this thread is running on right now, or `null` if `getcpu` fails.
pub fn currentCpu() ?usize {
    var cpu: usize = 0;
    const rc = linux.getcpu(&cpu, null);
    if (linux.errno(rc) != .SUCCESS) return null;
    return cpu;
}

/// Pin the calling thread to one CPU.
///
/// §7.1 requires it and the bench never did it. On a heterogeneous part the
/// difference is not noise: the host the published tables came from has Zen 5
/// cores sharing a 16 MiB L3 and Zen 5c cores sharing 8 MiB, at different
/// clocks, and an unpinned run migrates between them mid-sweep. Every cell in
/// the matrix is a per-core number and has to say which core.
pub fn pinToCpu(cpu: usize) !void {
    if (cpu >= linux.CPU_SETSIZE * 8) return error.CpuOutOfRange;
    var set: linux.cpu_set_t = @splat(0);
    const bits = @bitSizeOf(usize);
    set[cpu / bits] |= @as(usize, 1) << @intCast(cpu % bits);
    try linux.sched_setaffinity(0, &set);
}

/// Cache sizes for one CPU, read from sysfs.
pub const CpuCaches = struct {
    cpu: usize,
    /// Level-1 data cache, bytes; `null` when sysfs did not say.
    l1d: ?usize,
    l2: ?usize,
    l3: ?usize,
    /// `shared_cpu_list` of the L3, e.g. `0-3,12-15`. Which cores this one
    /// competes with for its LLC, and on a hybrid part, which cluster it is.
    /// A list longer than the buffer is cut and ends in `...` rather than
    /// being silently truncated to a plausible-looking shorter list.
    l3_shared: [l3_shared_capacity]u8,
    l3_shared_len: usize,

    pub const l3_shared_capacity = 256;
    pub const truncation_mark = "...";

    pub fn l3SharedList(self: *const CpuCaches) []const u8 {
        return self.l3_shared[0..self.l3_shared_len];
    }

    /// Store `list` as the L3's `shared_cpu_list`, marking truncation.
    pub fn setL3Shared(self: *CpuCaches, list: []const u8) void {
        if (list.len <= l3_shared_capacity) {
            @memcpy(self.l3_shared[0..list.len], list);
            self.l3_shared_len = list.len;
            return;
        }
        const keep = l3_shared_capacity - truncation_mark.len;
        @memcpy(self.l3_shared[0..keep], list[0..keep]);
        @memcpy(self.l3_shared[keep..], truncation_mark);
        self.l3_shared_len = l3_shared_capacity;
    }

    /// The L3 to size an "L3-resident" working set from, with a fallback for
    /// hosts whose sysfs does not expose one.
    pub fn l3OrFallback(self: *const CpuCaches, fallback: usize) usize {
        return self.l3 orelse fallback;
    }
};

/// Read `/sys/devices/system/cpu/cpuN/cache/indexK/{level,type,size,shared_cpu_list}`.
///
/// Anything unreadable is left `null`; the caller decides what that means.
pub fn cpuCaches(cpu: usize) CpuCaches {
    var out: CpuCaches = .{ .cpu = cpu, .l1d = null, .l2 = null, .l3 = null, .l3_shared = undefined, .l3_shared_len = 0 };
    var k: usize = 0;
    while (k < 8) : (k += 1) {
        var pbuf: [128]u8 = undefined;
        var vbuf: [128]u8 = undefined;
        const level_s = readSysfs(&pbuf, &vbuf, "/sys/devices/system/cpu/cpu{d}/cache/index{d}/level", .{ cpu, k }) orelse break;
        const level = std.fmt.parseInt(usize, level_s, 10) catch continue;
        var tbuf: [128]u8 = undefined;
        const typ = readSysfs(&pbuf, &tbuf, "/sys/devices/system/cpu/cpu{d}/cache/index{d}/type", .{ cpu, k }) orelse continue;
        var sbuf: [128]u8 = undefined;
        const size_s = readSysfs(&pbuf, &sbuf, "/sys/devices/system/cpu/cpu{d}/cache/index{d}/size", .{ cpu, k }) orelse continue;
        const size = parseCacheSize(size_s) orelse continue;
        switch (level) {
            1 => if (std.mem.eql(u8, typ, "Data") or std.mem.eql(u8, typ, "Unified")) {
                out.l1d = size;
            },
            2 => out.l2 = size,
            3 => {
                out.l3 = size;
                // Read into a buffer larger than the field so a long list is
                // *seen* to be long and marked, not cut at the field's edge.
                var lbuf: [4096]u8 = undefined;
                if (readSysfs(&pbuf, &lbuf, "/sys/devices/system/cpu/cpu{d}/cache/index{d}/shared_cpu_list", .{ cpu, k })) |sh| {
                    out.setL3Shared(sh);
                }
            },
            else => {},
        }
    }
    return out;
}

/// Read a small sysfs/procfs file into `buf`, trimmed of the trailing newline.
fn readSysfs(pbuf: []u8, buf: []u8, comptime fmt: []const u8, args: anytype) ?[]const u8 {
    const path = std.fmt.bufPrintZ(pbuf, fmt, args) catch return null;
    const rc = linux.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: linux.fd_t = @intCast(rc);
    defer _ = linux.close(fd);
    const n = linux.read(fd, buf.ptr, buf.len);
    if (linux.errno(n) != .SUCCESS) return null;
    return std.mem.trimEnd(u8, buf[0..n], "\n\r \t");
}

/// `48K` / `1024K` / `16384K` / `2M`, as sysfs writes them.
fn parseCacheSize(s: []const u8) ?usize {
    if (s.len == 0) return null;
    var digits = s;
    var mult: usize = 1;
    switch (s[s.len - 1]) {
        'K', 'k' => {
            mult = 1024;
            digits = s[0 .. s.len - 1];
        },
        'M', 'm' => {
            mult = 1024 * 1024;
            digits = s[0 .. s.len - 1];
        },
        else => {},
    }
    const v = std.fmt.parseInt(usize, digits, 10) catch return null;
    return v * mult;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "region allocates, faults in, and frees" {
    const r = try Region.alloc(4 * 1024 * 1024, .huge_pages);
    defer r.free();
    try std.testing.expect(r.bytes.len >= 4 * 1024 * 1024);
    try std.testing.expectEqual(@as(usize, 0), r.bytes.len % huge_page);
    // Writable across the whole span.
    r.bytes[r.bytes.len - 1] = 42;
    try std.testing.expectEqual(@as(u8, 42), r.bytes[r.bytes.len - 1]);
}

test "cache size strings parse as sysfs writes them" {
    try std.testing.expectEqual(@as(?usize, 48 * 1024), parseCacheSize("48K"));
    try std.testing.expectEqual(@as(?usize, 16384 * 1024), parseCacheSize("16384K"));
    try std.testing.expectEqual(@as(?usize, 2 * 1024 * 1024), parseCacheSize("2M"));
    try std.testing.expectEqual(@as(?usize, 512), parseCacheSize("512"));
    try std.testing.expectEqual(@as(?usize, null), parseCacheSize(""));
    try std.testing.expectEqual(@as(?usize, null), parseCacheSize("big"));
}

test "a long shared_cpu_list is kept whole up to the buffer and marked when cut" {
    var c: CpuCaches = .{ .cpu = 0, .l1d = null, .l2 = null, .l3 = null, .l3_shared = undefined, .l3_shared_len = 0 };
    // Short: stored verbatim.
    c.setL3Shared("0-3,12-15");
    try std.testing.expectEqualStrings("0-3,12-15", c.l3SharedList());
    // Exactly the capacity: still verbatim, no mark.
    const exact = "7," ** (CpuCaches.l3_shared_capacity / 2);
    c.setL3Shared(exact);
    try std.testing.expectEqualStrings(exact, c.l3SharedList());
    // A 512-CPU part listing every other core: "0,2,4,...,1022" is well over
    // the old 64-byte field, which silently kept "0,2,4,...,44,4" and looked
    // like a real list. Now it is cut and says so.
    var long: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&long);
    var cpu: usize = 0;
    while (cpu < 1024) : (cpu += 2) {
        if (cpu > 0) try w.writeByte(',');
        try w.print("{d}", .{cpu});
    }
    const list = w.buffered();
    try std.testing.expect(list.len > CpuCaches.l3_shared_capacity);
    c.setL3Shared(list);
    const got = c.l3SharedList();
    try std.testing.expectEqual(CpuCaches.l3_shared_capacity, got.len);
    try std.testing.expect(std.mem.endsWith(u8, got, CpuCaches.truncation_mark));
    try std.testing.expect(std.mem.startsWith(u8, list, got[0 .. got.len - CpuCaches.truncation_mark.len]));
    // And a real host's list, whatever its length, is never longer than the field.
    const host = cpuCaches(0);
    try std.testing.expect(host.l3_shared_len <= CpuCaches.l3_shared_capacity);
}

test "cpu caches read from sysfs are ordered L1 <= L2 <= L3 when present" {
    const c = cpuCaches(0);
    if (c.l1d) |l1| if (c.l2) |l2| try std.testing.expect(l1 <= l2);
    if (c.l2) |l2| if (c.l3) |l3| try std.testing.expect(l2 <= l3);
    try std.testing.expectEqual(@as(usize, 0), c.cpu);
}

test "pinning to the current cpu keeps us there" {
    const here = currentCpu() orelse return error.SkipZigTest;
    pinToCpu(here) catch return error.SkipZigTest;
    // A few reschedules later we must still be on the same CPU.
    for (0..8) |_| _ = linux.sched_yield();
    try std.testing.expectEqual(here, currentCpu().?);
}

test "a huge-page region reports how much of it is huge-backed" {
    // Whatever THP policy the host has, the *question* must be answerable:
    // smaps is always there for our own process. The fraction is only
    // asserted when the kernel says it will honour MADV_HUGEPAGE.
    const r = try Region.alloc(4 * huge_page, .huge_pages);
    defer r.free();
    const frac = r.hugeBackedFraction() orelse return error.SkipZigTest;
    try std.testing.expect(frac >= 0.0 and frac <= 1.0);
    if (storage.thpEnabled() orelse false) try std.testing.expect(frac > 0.0);
}

test "pointer chains form a single cycle covering every slot" {
    // A chase that does not form one cycle would revisit a small subset and
    // measure L1 latency while claiming to measure DRAM, the classic way this
    // benchmark lies. Verify the cycle length equals the slot count.
    const r = try Region.alloc(4 * 1024 * 1024, .small_pages);
    defer r.free();
    var prng = std.Random.DefaultPrng.init(7);

    const heads = buildChains(r, 1, prng.random());
    defer std.heap.page_allocator.free(heads);
    const idx = std.mem.bytesAsSlice(usize, r.bytes);

    const slots = r.bytes.len / line_size;
    var seen = std.DynamicBitSet.initEmpty(std.testing.allocator, slots) catch unreachable;
    defer seen.deinit();

    var p = heads[0];
    var count: usize = 0;
    while (count < slots) : (count += 1) {
        const slot = p / (line_size / @sizeOf(usize));
        try std.testing.expect(!seen.isSet(slot));
        seen.set(slot);
        p = idx[p];
    }
    // Returned to the start after exactly `slots` hops.
    try std.testing.expectEqual(heads[0], p);
}

test "latency over a small region is fast, over a large region is slower" {
    // Non-vacuous sanity: an L2-resident chase must beat a DRAM-resident one.
    // Deliberately loose, this asserts the benchmark measures *something*
    // real, not a specific host's timings.
    //
    // Pinned and taken best-of-N, which is what the driver does and what this
    // did not. Unpinned, the L2-resident arm swung 9 ns to 47 ns run to run on
    // an idle host: on a hybrid part the thread migrates between clusters
    // mid-measurement, and `pinToCpu`'s own doc says so. The DRAM arm barely
    // moves (125 ns +/- 3), because it is bound by a latency that does not
    // depend on which core is waiting -- so the spread was one-sided and the
    // margin was whatever the small arm happened to draw. Under `zig build
    // test` the three test binaries run as parallel build steps, so this one
    // measures while the engine suite thrashes memory on every other core,
    // and the draw occasionally crossed.
    //
    // Minimum rather than mean: for a latency probe, contention, migration and
    // a frequency dip can only *add* time, so the smallest of a few runs is
    // the least contaminated estimate of the quantity being asserted about.
    // Averaging would fold the contamination in and need a tolerance to
    // survive it, which is the kind of loose threshold that stops the test
    // failing for real reasons too.
    const here = currentCpu() orelse return error.SkipZigTest;
    var prior: linux.cpu_set_t = undefined;
    const had_prior = linux.errno(linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &prior)) == .SUCCESS;
    pinToCpu(here) catch return error.SkipZigTest;
    // Zig runs tests sequentially on one thread, so an affinity set here
    // outlives the test. Put it back rather than pinning everything after it.
    defer if (had_prior) linux.sched_setaffinity(0, &prior) catch {};

    var prng = std.Random.DefaultPrng.init(11);
    const small = try Region.alloc(256 * 1024, .small_pages);
    defer small.free();
    const big = try Region.alloc(256 * 1024 * 1024, .small_pages);
    defer big.free();

    var best_small: f64 = std.math.inf(f64);
    var best_big: f64 = std.math.inf(f64);
    for (0..3) |_| {
        best_small = @min(best_small, latency(small, 100_000, prng.random()).ns_per_access);
        best_big = @min(best_big, latency(big, 100_000, prng.random()).ns_per_access);
    }
    try std.testing.expect(best_small < best_big);
}
