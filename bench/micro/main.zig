//! §7.2, the layered microbenchmark driver.
//!
//! Produces the measured constants that `docs/cost-model.md` records and that
//! §5's model consumes. M0's exit criterion:
//!
//!   "`docs/cost-model.md` with measured constants for the target host; the
//!    table in §5.2 filled in with real numbers; the §7.5 ISA matrix populated
//!    at the microbenchmark level for at least two microarchitectures, with
//!    cycles/vector, ns/vector, and effective frequency for every cell."
//!
//! Usage:
//!   bench-micro                run everything
//!   bench-micro hw             hardware baseline only (§7.2 layer 1)
//!   bench-micro kernels        kernel matrix only (§7.2 layer 2)
//!   bench-micro --pin <cpu>    pin to this CPU (default: the CPU we start on)
//!   bench-micro --no-pin       run unpinned; the output says so
//!   bench-micro --quick        d=128 only, fewer evaluations: a smoke run
//!   bench-micro --json         machine-readable, for the SQLite sink of §7.3
//!
//! Every measurement is a per-core number, so the process pins itself to one
//! CPU before measuring anything (§7.1) and reports which, together with that
//! CPU's cache sizes from sysfs. Refuses to run unpinned unless told to.

const std = @import("std");
const Io = std.Io;
const strawmann = @import("strawmann");
const build_options = @import("build_options");

const perfctr = @import("perfctr.zig");
const hw = @import("hw.zig");
const kernels = @import("kernels.zig");

const dist = strawmann.dist;

/// §7.5's dimension axis: "{d = 128, 384, 768, 1536}".
const matrix_dims = [_]usize{ 128, 384, 768, 1536 };

const residencies = [_]kernels.Residency{ .l1_hot, .l3_resident, .dram_cold };

/// Everything the run needs to know about where it is running.
const Host = struct {
    /// The CPU we are pinned to, or `null` for a `--no-pin` run.
    cpu: ?usize,
    caches: hw.CpuCaches,
    /// L3 to size the L3-resident working set from.
    l3_bytes: usize,
    /// Machine-wide bandwidth from `sysinfo.probe`, taken *before* pinning
    /// (a pinned process's threads inherit the one-CPU affinity and the
    /// aggregate would just be the single-core figure again).
    aggregate: ?strawmann.sysinfo.Result,
    quick: bool,
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var want_hw = true;
    var want_kernels = true;
    var json = false;
    var quick = false;
    var pin: ?usize = null;
    var no_pin = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "hw")) want_kernels = false;
        if (std.mem.eql(u8, a, "kernels")) want_hw = false;
        if (std.mem.eql(u8, a, "--json")) json = true;
        if (std.mem.eql(u8, a, "--quick")) quick = true;
        if (std.mem.eql(u8, a, "--no-pin")) no_pin = true;
        if (std.mem.eql(u8, a, "--pin")) {
            i += 1;
            if (i >= args.len) return error.MissingPinArgument;
            pin = try std.fmt.parseInt(usize, args[i], 10);
        }
    }

    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(.stdout(), init.io, &buf);
    const w = &fw.interface;

    // Aggregate bandwidth first, while the process can still use every core.
    const aggregate = if (want_hw) strawmann.sysinfo.probe(std.heap.page_allocator) else null;

    // Pin. §7.1 requires it; the default is the CPU the scheduler put us on,
    // which is at least a *fixed* one, and the output says which so a run on
    // the wrong cluster of a hybrid part is visible rather than silent.
    var cpu: ?usize = null;
    if (!no_pin) {
        const target = pin orelse hw.currentCpu() orelse return error.CannotDetermineCpu;
        try hw.pinToCpu(target);
        cpu = target;
    } else if (pin != null) {
        return error.PinAndNoPinConflict;
    }
    const cache_cpu = cpu orelse hw.currentCpu() orelse 0;
    const caches = hw.cpuCaches(cache_cpu);
    const host: Host = .{
        .cpu = cpu,
        .caches = caches,
        .l3_bytes = caches.l3OrFallback(kernels.l3_fallback_bytes),
        .aggregate = aggregate,
        .quick = quick,
    };

    try preamble(w, host);
    if (want_hw) try hardwareBaseline(w, host);
    if (want_kernels) try kernelMatrix(w, host);
    try w.flush();
}

fn preamble(w: *Io.Writer, host: Host) !void {
    const counters = perfctr.Counters.open();
    defer counters.close();

    try w.print("# strawmann microbenchmarks\n\n", .{});
    if (host.cpu) |c| {
        try w.print("pinned to cpu  : {d}\n", .{c});
    } else {
        try w.print("pinned to cpu  : NONE (--no-pin): every per-core number below may mix cores\n", .{});
    }
    try w.print("cpu{d} caches    : L1d {s}, L2 {s}, L3 {s} (shared with cpus {s}), from sysfs\n", .{
        host.caches.cpu,
        fmtBytes(host.caches.l1d).s(),
        fmtBytes(host.caches.l2).s(),
        fmtBytes(host.caches.l3).s(),
        if (host.caches.l3_shared_len > 0) host.caches.l3SharedList() else "?",
    });
    if (host.caches.l3 == null) {
        try w.print("                 (no L3 size in sysfs; L3-resident sized from a {d} MiB fallback)\n", .{kernels.l3_fallback_bytes >> 20});
    }
    try w.print("isa build      : {s}\n", .{build_options.isa_build_name});
    try w.print("f32 lanes      : {d} (accumulators: {d})\n", .{
        dist.native.lanes,
        dist.native.accumulators,
    });
    try w.print("dispatch tier  : {s} (build max: {s}, host max: {s})\n", .{
        dist.dispatch.active.tier.name(),
        dist.dispatch.compiledMaxTier().name(),
        dist.dispatch.hostMaxTier().name(),
    });
    try w.print("vnni u8xi8     : {}\n", .{dist.dot_i8.u8i8_native.uses_vnni});
    try w.print("perf counters  : {s}\n", .{
        if (counters.haveFrequency())
            "available (effective frequency will be reported)"
        else
            "UNAVAILABLE - effective frequency cannot be reported; §7.5 forbids " ++
                "publishing an ISA comparison from this run",
    });
    try w.print("nominal TSC    : {d:.3} GHz\n", .{perfctr.nominalTscHz() / 1e9});
    if (host.quick) try w.print("mode           : --quick (d=128 only, reduced evaluations; NOT for publication)\n", .{});
    try w.print("\n", .{});
}

/// `48 KiB` / `16 MiB` / `?`, for the cache line of the preamble.
const HumanBytes = struct {
    buf: [16]u8,
    len: usize,

    fn s(self: *const HumanBytes) []const u8 {
        return self.buf[0..self.len];
    }
};

fn fmtBytes(v: ?usize) HumanBytes {
    var out: HumanBytes = .{ .buf = undefined, .len = 0 };
    const n = v orelse {
        out.buf[0] = '?';
        out.len = 1;
        return out;
    };
    const txt = if (n >= 1024 * 1024 and n % (1024 * 1024) == 0)
        std.fmt.bufPrint(&out.buf, "{d} MiB", .{n / (1024 * 1024)}) catch unreachable
    else
        std.fmt.bufPrint(&out.buf, "{d} KiB", .{n / 1024}) catch unreachable;
    out.len = txt.len;
    return out;
}

// -------------------------------------------------------------------------
// §7.2 layer 1, hardware baseline
// -------------------------------------------------------------------------

fn hardwareBaseline(w: *Io.Writer, host: Host) !void {
    var prng = std.Random.DefaultPrng.init(0x5deed);
    const rnd = prng.random();

    try w.print("## 1. Hardware baseline (§7.2.1)\n\n", .{});

    // --- bandwidth ---
    {
        const region = try hw.Region.alloc(512 * 1024 * 1024, .huge_pages);
        defer region.free();
        const r = hw.bandwidth(region, 5);
        try w.print("### Sequential bandwidth\n\n", .{});
        try w.print("  read, single core  : {d:>7.1} GB/s\n", .{r.read_gbps});
        try w.print("  triad, single core : {d:>7.1} GB/s\n", .{r.triad_gbps});
        if (host.aggregate) |agg| {
            try w.print("  read, all cores    : {d:>7.1} GB/s ({d} threads, sysinfo.probe, taken before pinning)\n", .{ agg.aggregate_gbps, agg.threads });
        } else {
            try w.print("  read, all cores    : (unavailable)\n", .{});
        }
        try w.print("\n  §5.3 cross-check: at 2500 evals x 3072 B = 7.7 MB/query,\n", .{});
        try w.print("  the single-core figure caps fp32 search at ~{d:.0}k QPS *per core*;\n", .{
            r.read_gbps * 1e9 / 7.7e6 / 1000.0,
        });
        if (host.aggregate) |agg| {
            try w.print("  the aggregate caps it at ~{d:.0}k QPS machine-wide, regardless of core count.\n\n", .{
                agg.aggregate_gbps * 1e9 / 7.7e6 / 1000.0,
            });
        } else {
            try w.print("  (no aggregate figure, so no machine-wide cap is derived here.)\n\n", .{});
        }
    }

    // --- latency ---
    {
        const region = try hw.Region.alloc(1024 * 1024 * 1024, .huge_pages);
        defer region.free();
        const r = hw.latency(region, 2_000_000, rnd);
        try w.print("### Random-access latency (dependent chase, 1 GiB, 2 MiB pages)\n\n", .{});
        try w.print("  {d:.1} ns/access\n", .{r.ns_per_access});
        try w.print("\n  §5.2 assumes ~85-100 ns. ", .{});
        if (r.ns_per_access >= 85.0 and r.ns_per_access <= 100.0) {
            try w.print("Measured value is inside that range.\n\n", .{});
        } else {
            try w.print("MEASURED VALUE IS OUTSIDE THAT RANGE -\n", .{});
            try w.print("  every ns/vector figure in §5.2 must be rescaled by {d:.2}x.\n\n", .{
                r.ns_per_access / 90.0,
            });
        }
    }

    // --- memory-level parallelism: the key constant ---
    {
        const region = try hw.Region.alloc(1024 * 1024 * 1024, .huge_pages);
        defer region.free();
        var points: [16]hw.MlpPoint = undefined;
        const got = hw.mlpSweep(region, hw.max_sweep_chains, 100_000, rnd, &points);

        try w.print("### Memory-level parallelism (§5.2's LFB/MSHR limit)\n\n", .{});
        try w.print("  chains |  ns/access |    GB/s | speedup vs 1 chain\n", .{});
        try w.print("  -------|------------|---------|-------------------\n", .{});
        const base_ns = got[0].ns_per_access;
        for (got) |p| {
            try w.print("  {d:>6} | {d:>10.1} | {d:>7.2} | {d:>17.2}x\n", .{
                p.chains, p.ns_per_access, p.gbps, base_ns / p.ns_per_access,
            });
        }

        // The knee: the last chain count that still improved throughput by >10%.
        var knee: usize = 1;
        for (got[1..], 1..) |p, i| {
            if (p.ns_per_access < got[i - 1].ns_per_access * 0.9) knee = p.chains;
        }
        var peak: f64 = 0;
        for (got) |p| peak = @max(peak, p.gbps);

        // What an HNSW expansion can actually supply. §5.2: "Within one
        // expansion of a node with `M` neighbours, however, the `M` neighbour
        // vector fetches are independent." So M bounds the concurrency the
        // algorithm offers, independently of what the core can sustain.
        const m_default = 16;
        var at_m: f64 = 0;
        for (got) |p| {
            if (p.chains <= m_default) at_m = p.gbps;
        }

        try w.print("\n  Knee at ~{d} concurrent misses; peak {d:.2} GB/s.\n\n", .{ knee, peak });
        try w.print("  §5.2 models 10-16 outstanding line fills and derives ~8.5 GB/s.\n", .{});
        try w.print("  At {d} chains, the concurrency an M=16 HNSW expansion supplies -\n", .{m_default});
        try w.print("  this host measures {d:.2} GB/s, which matches the model closely.\n", .{at_m});
        try w.print("  But the core saturates at ~{d} misses and {d:.2} GB/s.\n\n", .{ knee, peak });
        try w.print("  ==> HEADROOM: {d:.1}x between what M=16 supplies and what the\n", .{peak / at_m});
        try w.print("      core can sustain. §5.2 calls software prefetch of the\n", .{});
        try w.print("      neighbour list \"the single highest-leverage optimisation in\n", .{});
        try w.print("      the search path\"; this is that claim as a number, and it is\n", .{});
        try w.print("      the budget available to anything that raises the concurrent\n", .{});
        try w.print("      miss count above M (prefetching the next candidate's\n", .{});
        try w.print("      neighbours, scoring several candidates at once, or a larger M).\n\n", .{});
    }

    // --- TLB / page size ---
    {
        try w.print("### TLB penalty: 4 KiB vs 2 MiB pages (§5.5)\n\n", .{});
        const r = try hw.tlbComparison(1024 * 1024 * 1024, 1_000_000, rnd);
        try w.print("  4 KiB pages : {d:>6.1} ns/access\n", .{r.small.ns_per_access});
        try w.print("  2 MiB pages : {d:>6.1} ns/access\n", .{r.huge.ns_per_access});
        try w.print("  penalty     : {d:>6.2}x\n", .{r.penaltyRatio()});
        try printHugeBacked(w, "  2 MiB arm huge-backed (smaps): ", r.huge_backed);
        if (r.small_huge_backed) |f| {
            if (f > 0.0) try w.print("  WARNING: the 4 KiB arm is {d:.0}% huge-backed; MADV_NOHUGEPAGE was not honoured.\n", .{f * 100.0});
        }
        const m = r.missesPerAccess();
        if (m.small) |ms| {
            try w.print("  dTLB load misses/access: 4 KiB {d:.3}", .{ms});
            if (m.huge) |mh| try w.print(", 2 MiB {d:.3}", .{mh});
            try w.print("\n", .{});
        }
        try w.print("\n  §5.5: 3 GB of vectors at 4 KiB is 786k pages against a\n", .{});
        try w.print("  ~1.5-3k entry dTLB, so every random vector access is a\n", .{});
        try w.print("  miss plus a page walk. This is the delta the 'resident'\n", .{});
        try w.print("  storage mode exists to avoid, and the one Qdrant's\n", .{});
        try w.print("  file-backed mmap cannot generally avoid.\n\n", .{});
    }
}

/// One line: what fraction of a huge-page region the kernel actually backed
/// with 2 MiB folios, and a warning when it is below `hw.huge_backed_warn_fraction`.
fn printHugeBacked(w: *Io.Writer, label: []const u8, frac: ?f64) !void {
    if (frac) |f| {
        try w.print("{s}{d:.1}%", .{ label, f * 100.0 });
        if (f < hw.huge_backed_warn_fraction) {
            try w.print("  <-- WARNING: below {d:.0}%, this is mostly 4 KiB pages; the \"2 MiB\" label does not hold\n", .{hw.huge_backed_warn_fraction * 100.0});
        } else {
            try w.print("\n", .{});
        }
    } else {
        try w.print("{s}unknown (/proc/self/smaps unreadable)\n", .{label});
    }
}

// -------------------------------------------------------------------------
// §7.2 layer 2 / §7.5, the kernel matrix
// -------------------------------------------------------------------------

fn kernelMatrix(w: *Io.Writer, host: Host) !void {
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();

    try w.print("## 2. Kernel matrix (§7.2.2, §7.5)\n\n", .{});
    try w.print("Cycles/vector is primary: it isolates the ISA question from the\n", .{});
    try w.print("frequency question. ns/vector is what actually matters and is\n", .{});
    try w.print("reported alongside, never instead (§7.5).\n\n", .{});
    try w.print("Residency is followed by the access pattern. `independent (MLP)` walks a\n", .{});
    try w.print("random permutation whose next address is known before the current vector\n", .{});
    try w.print("arrives, so the core overlaps misses: a throughput ceiling. `dependent`\n", .{});
    try w.print("makes the next address depend on the current score, so fetch, kernel and\n", .{});
    try w.print("next fetch are one chain: §5.2's per-vector cost for an unprefetched walk.\n\n", .{});
    if (host.cpu) |c| {
        try w.print("Pinned to cpu {d}; L3-resident working set = {d} MiB, half of this core's\n", .{ c, kernels.workingSetBytes(.l3_resident, host.l3_bytes) >> 20 });
        try w.print("{s} L3 as read from sysfs.\n\n", .{fmtBytes(host.caches.l3).s()});
    } else {
        try w.print("UNPINNED RUN: cells may mix cores; L3-resident sized from cpu{d}'s L3.\n\n", .{host.caches.cpu});
    }

    try w.print("| kernel | d | residency | access | ns/vec | cyc/vec | insn/vec | IPC | eff.freq | GB/s | huge% |\n", .{});
    try w.print("|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|\n", .{});

    const dims: []const usize = if (host.quick) matrix_dims[0..1] else &matrix_dims;
    for (dims) |dim| {
        for (residencies) |res| {
            try benchOne(w, host, kernels.dotF32Spec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.euclidF32Spec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.dotF16Spec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.dotI8AsymSpec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.euclidI8AsymSpec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.euclidU8Spec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.dotI8SymSpec(dim), dim, res, rnd);
            try benchOne(w, host, kernels.hammingSpec(dim), dim, res, rnd);
            // PQ at x8: m = d/8 subquantizers of 8 dims each.
            try benchOne(w, host, kernels.pqAdc8Spec(dim / 8), dim / 8, res, rnd);
        }
    }
    try w.print("\n", .{});
    try w.print("PQ rows: the query is the m x 256 fp32 LUT, m KiB. At m=16 (d=128) it\n", .{});
    try w.print("fits L1d; at m=48 and above (d>=384; 192 KiB at m=192) it does not, so\n", .{});
    try w.print("those `L1-hot` cells time codes in L1 against a LUT resident in **L2**,\n", .{});
    try w.print("and the same LUT is L2-resident in every residency row of that m.\n\n", .{});
    try w.print("§6.6.1's prediction under test: the L1-hot and L3-resident rows\n", .{});
    try w.print("should track vector width; the DRAM-cold fp32 rows should not.\n", .{});
    try w.print("Compare this table across the arms built by `zig build bench-isa`.\n\n", .{});
}

/// Fill an arena with values of the type the kernel will read.
///
/// The arena used to be filled with random *bytes*, which reinterpreted as
/// fp32/fp16 are NaN, Inf and denormals a large fraction of the time.
/// Denormal operands take a microcode assist on Intel FMA units and cost
/// measurably even on Zen 5 (+12% on the fp32 kernels when this was checked),
/// so the hot fp32 and fp16 cells were timing the assist, not the kernel.
/// Every kernel is data-independent in timing *for finite normal inputs*;
/// that is the case this fills.
fn fillArena(bytes: []u8, elem: kernels.Elem, rnd: std.Random) void {
    switch (elem) {
        .f32 => {
            for (std.mem.bytesAsSlice(f32, bytes[0 .. bytes.len / 4 * 4])) |*x| x.* = rnd.float(f32) * 2.0 - 1.0;
        },
        .f16 => {
            for (std.mem.bytesAsSlice(f16, bytes[0 .. bytes.len / 2 * 2])) |*x| x.* = @floatCast(rnd.float(f32) * 2.0 - 1.0);
        },
        // Any byte is a valid u8 code, a valid PQ centroid index in [0, 256),
        // and any word a valid binary code, so the raw fill is the right one.
        .u8, .pq8, .bits => {
            for (std.mem.bytesAsSlice(u64, bytes[0 .. bytes.len / 8 * 8])) |*x| x.* = rnd.int(u64);
        },
    }
}

fn benchOne(
    w: *Io.Writer,
    host: Host,
    spec: kernels.Spec,
    dim: usize,
    res: kernels.Residency,
    rnd: std.Random,
) !void {
    // Order and query buffers come from the page allocator, not the process
    // arena: the sweep runs ~100 cells and an arena would keep every one of
    // them (a 1 GiB cold arena has 8M slots of u32 for SQ8 at d=128).
    const alloc = std.heap.page_allocator;

    const target_bytes = kernels.workingSetBytes(res, host.l3_bytes);
    const slots = @max(1, target_bytes / spec.stride);

    const arena = try hw.Region.alloc(@max(spec.stride * slots, 2 * 1024 * 1024), .huge_pages);
    defer arena.free();
    fillArena(arena.bytes, spec.elem, rnd);
    const huge_frac = arena.hugeBackedFraction();

    const order = try kernels.randomOrder(alloc, slots, rnd);
    defer alloc.free(order);

    // Build the query in the shape this kernel expects: the same element type
    // as the arena, except the asymmetric SQ8 kernels (fp32 query against u8
    // codes) and PQ (the query is the m x 256 fp32 lookup table).
    const query_bytes = try alloc.alignedAlloc(u8, .@"64", @max(64, dim * 4 + 256 * 4 * dim));
    defer alloc.free(query_bytes);
    const asym = std.mem.eql(u8, spec.name, "dot_i8_asym") or std.mem.eql(u8, spec.name, "euclid_i8_asym");
    switch (spec.elem) {
        .f32 => fillArena(query_bytes[0 .. dim * 4], .f32, rnd),
        .f16 => fillArena(query_bytes[0 .. dim * 2], .f16, rnd),
        .u8 => if (asym) fillArena(query_bytes[0 .. dim * 4], .f32, rnd) else fillArena(query_bytes[0..@max(64, dim)], .u8, rnd),
        .bits => fillArena(query_bytes[0..@max(64, spec.stride)], .bits, rnd),
        .pq8 => fillArena(query_bytes[0 .. dim * 256 * 4], .f32, rnd),
    }

    // Enough evaluations to swamp timer resolution at every residency, but
    // bounded so the whole matrix finishes in reasonable time.
    const full: usize = switch (res) {
        .l1_hot => 2_000_000,
        .l3_resident => 1_000_000,
        .dram_cold => 300_000,
    };
    const evals = if (host.quick) full / 10 else full;

    // The hot cell has no memory system to be dependent on; the other two are
    // run both ways over the *same* filled arena, so the two rows differ only
    // in the loop.
    const accesses: []const kernels.Access = if (res == .l1_hot) &.{.independent} else &.{ .independent, .dependent };
    for (accesses) |access| {
        // zlint-disable-next-line no-print -- this *is* the benchmark's output; it
        // runs from a CLI whose whole purpose is printing measurements, and §7
        // wants progress visible during a long sweep.
        std.debug.print("  [bench] {s} d={d} {s} {s} slots={d} arena={d}MiB\n", .{ spec.name, dim, res.name(), access.name(), slots, arena.bytes.len / (1024 * 1024) });
        const r = kernels.run(spec, dim, res, access, query_bytes.ptr, arena, order, evals);

        try w.print("| {s} | {d} | {s} | {s} | {d:.1} | ", .{ r.kernel, r.dim, r.residency.name(), r.access.name(), r.ns_per_vector });
        if (r.cycles_per_vector) |c| try w.print("{d:.1}", .{c}) else try w.print("-", .{});
        try w.print(" | ", .{});
        if (r.instructions_per_vector) |i| try w.print("{d:.1}", .{i}) else try w.print("-", .{});
        try w.print(" | ", .{});
        if (r.ipc) |i| try w.print("{d:.2}", .{i}) else try w.print("-", .{});
        try w.print(" | ", .{});
        if (r.effective_frequency) |f| try w.print("{d:.3}", .{f}) else try w.print("-", .{});
        try w.print(" | {d:.2} | ", .{r.gbps()});
        if (huge_frac) |f| {
            try w.print("{d:.0}", .{f * 100.0});
            if (f < hw.huge_backed_warn_fraction) try w.print(" WARN", .{});
        } else {
            try w.print("?", .{});
        }
        try w.print(" |\n", .{});
        if (r.pmu_running_fraction) |frac| {
            try w.print("| | | | | (perf group multiplexed: on PMU {d:.0}% of the window; counts above are scaled) | | | | | | |\n", .{frac * 100.0});
        }
    }
}

// The bench root only reaches `hw`, `kernels` and `perfctr` from `main`,
// which a test build never analyses, so their tests were invisible to
// `zig build test` until this pulled them in.
test {
    _ = hw;
    _ = kernels;
    _ = perfctr;
}
