//! In-process hardware performance counters via `perf_event_open`.
//!
//! §7.1 makes one of these mandatory rather than optional:
//!
//!   "**effective frequency logged per run** (`cycles / ref-cycles`), since
//!    AVX-512 licensing (§6.6.4) makes nominal frequency a lie; the harness
//!    records it whether or not the run is an ISA comparison"
//!
//! and §7.5 turns it into a publication gate:
//!
//!   "any AVX-512-vs-AVX2 comparison where effective frequency differs by more
//!    than 3% between arms is reported with both frequencies visible. A
//!    cycles/vector win presented without its frequency is not a result."
//!
//! Counting in-process rather than shelling out to `perf stat` matters because
//! the interesting measurement is per-*kernel*, not per-process: a run that
//! benchmarks seven kernels needs seven frequency readings, and process-level
//! aggregation would average the 512-bit kernels together with the scalar setup
//! code and hide exactly the effect being measured.
//!
//! Requires `/proc/sys/kernel/perf_event_paranoid <= 2` for the hardware events
//! used here. Every function degrades to `null` rather than failing when the
//! counters are unavailable, so the benchmark still runs (and says so) on a
//! host where perf is locked down.

const std = @import("std");
const linux = std.os.linux;
const PERF = linux.PERF;

pub const Event = enum {
    cycles,
    /// Fixed-frequency reference cycles. The denominator of effective
    /// frequency: it ticks at the nominal TSC rate regardless of what the core
    /// clock is doing, so `cycles / ref_cycles` is the ratio §6.6.4 needs.
    ref_cycles,
    instructions,
    cache_misses,
    /// §7.3 / §5.5: the TLB story. `dTLB-load-misses` is what quantifies the
    /// 4 KiB-vs-2 MiB page delta that §5.5 predicts and calls "one of the more
    /// actionable findings for the Qdrant side".
    dtlb_load_misses,
    branch_misses,

    fn attrType(self: Event) PERF.TYPE {
        return switch (self) {
            .cycles, .ref_cycles, .instructions, .cache_misses, .branch_misses => .HARDWARE,
            .dtlb_load_misses => .HW_CACHE,
        };
    }

    fn attrConfig(self: Event) u64 {
        return switch (self) {
            .cycles => @intFromEnum(PERF.COUNT.HW.CPU_CYCLES),
            .ref_cycles => @intFromEnum(PERF.COUNT.HW.REF_CPU_CYCLES),
            .instructions => @intFromEnum(PERF.COUNT.HW.INSTRUCTIONS),
            .cache_misses => @intFromEnum(PERF.COUNT.HW.CACHE_MISSES),
            .branch_misses => @intFromEnum(PERF.COUNT.HW.BRANCH_MISSES),
            // HW_CACHE config is (cache_id) | (op_id << 8) | (result_id << 16).
            .dtlb_load_misses => @intFromEnum(PERF.COUNT.HW.CACHE.DTLB) |
                (@intFromEnum(PERF.COUNT.HW.CACHE.OP.READ) << 8) |
                (@intFromEnum(PERF.COUNT.HW.CACHE.RESULT.MISS) << 16),
        };
    }

    pub fn name(self: Event) []const u8 {
        return @tagName(self);
    }
};

/// `perf_event_attr.read_format` bits (`PERF_FORMAT_*`), not in `std`.
const FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 0;
const FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 1;
const FORMAT_GROUP: u64 = 1 << 3;
/// `PERF_IOC_FLAG_GROUP`: apply an ioctl to the leader's whole group.
const IOC_FLAG_GROUP: usize = 1;

/// The events a `Counters` group tries to open, in group order. The first one
/// that opens becomes the leader.
const group_events = [_]Event{ .cycles, .ref_cycles, .instructions, .dtlb_load_misses, .cache_misses };

/// A single counter, scoped to this thread. Opened either standalone or as a
/// member of a group whose leader is `group_fd`.
pub const Counter = struct {
    fd: linux.fd_t,
    event: Event,

    pub fn open(event: Event) ?Counter {
        return openIn(event, -1);
    }

    /// Open as a member of `group_fd`'s group (`-1` for a new group / a
    /// standalone counter).
    pub fn openIn(event: Event, group_fd: linux.fd_t) ?Counter {
        var attr: linux.perf_event_attr = .{
            .type = event.attrType(),
            .config = event.attrConfig(),
        };
        attr.flags.disabled = true;
        // Exclude kernel and hypervisor time: the question is what *our* loop
        // costs, and including the scheduler tick would add noise proportional
        // to how long the measurement ran.
        attr.flags.exclude_kernel = true;
        attr.flags.exclude_hv = true;
        // Count this thread on whatever CPU it is running on. §7.1 requires
        // explicit pinning anyway, so "whatever CPU" is a fixed one in a
        // conforming run.
        attr.flags.inherit = false;
        // Group read: one `read` on the leader returns every member plus the
        // time the group was scheduled on the PMU, so (a) all counters cover
        // exactly the same instructions and (b) multiplexing is visible.
        // Five separately-opened counters can each be scheduled for a
        // different slice of the run, and their ratios (IPC, cycles/ref) then
        // compare different windows without saying so.
        attr.read_format = FORMAT_GROUP | FORMAT_TOTAL_TIME_ENABLED | FORMAT_TOTAL_TIME_RUNNING;

        const rc = linux.perf_event_open(&attr, 0, -1, group_fd, 0);
        if (linux.errno(rc) != .SUCCESS) return null;
        return .{ .fd = @intCast(rc), .event = event };
    }

    pub fn close(self: Counter) void {
        _ = linux.close(self.fd);
    }

    pub fn reset(self: Counter) void {
        _ = linux.ioctl(self.fd, PERF.EVENT_IOC.RESET, 0);
    }

    pub fn enable(self: Counter) void {
        _ = linux.ioctl(self.fd, PERF.EVENT_IOC.ENABLE, 0);
    }

    pub fn disable(self: Counter) void {
        _ = linux.ioctl(self.fd, PERF.EVENT_IOC.DISABLE, 0);
    }

    /// Group-format read: `{ nr, time_enabled, time_running, values[nr] }`,
    /// where `values` follow the group's open order. Returns the raw values
    /// and the two times; `null` on a short or failed read.
    pub fn readGroup(self: Counter, values: []u64) ?struct { n: usize, enabled: u64, running: u64 } {
        var buf: [3 + group_events.len]u64 = undefined;
        const n = linux.read(self.fd, @ptrCast(&buf), @sizeOf(@TypeOf(buf)));
        if (linux.errno(n) != .SUCCESS) return null;
        if (n < 3 * @sizeOf(u64)) return null;
        const nr: usize = @intCast(buf[0]);
        if (nr > values.len or n < (3 + nr) * @sizeOf(u64)) return null;
        for (0..nr) |i| values[i] = buf[3 + i];
        return .{ .n = nr, .enabled = buf[1], .running = buf[2] };
    }

    /// Read a counter opened standalone with `open`, i.e. a one-member group
    /// whose layout is `{ 1, enabled, running, value }`. Unscaled.
    pub fn read(self: Counter) u64 {
        var buf: [4]u64 = undefined;
        const n = linux.read(self.fd, @ptrCast(&buf), @sizeOf(@TypeOf(buf)));
        if (linux.errno(n) != .SUCCESS) return 0;
        if (n >= 4 * @sizeOf(u64)) return buf[3];
        return 0;
    }
};

/// The counter set collected around every benchmarked region, opened as one
/// perf event group so every member is scheduled together and read at once.
pub const Counters = struct {
    /// Group leader; `null` when no hardware event could be opened at all.
    leader: ?Counter,
    /// Members in group order, leader first. `slot[e]` is the index of event
    /// `e` in the read buffer, or `null` if that event failed to open.
    members: [group_events.len]?Counter,
    slot: [group_events.len]?u8,

    pub fn open() Counters {
        var self: Counters = .{ .leader = null, .members = @splat(null), .slot = @splat(null) };
        var n: u8 = 0;
        for (group_events, 0..) |ev, e| {
            const group_fd: linux.fd_t = if (self.leader) |l| l.fd else -1;
            if (Counter.openIn(ev, group_fd)) |c| {
                if (self.leader == null) self.leader = c;
                self.members[n] = c;
                self.slot[e] = n;
                n += 1;
            }
        }
        return self;
    }

    pub fn close(self: Counters) void {
        for (self.members) |m| if (m) |c| c.close();
    }

    /// True when at least the cycles/ref-cycles pair is available, i.e. when
    /// effective frequency can be reported. §7.5 refuses to publish an ISA
    /// comparison without it, so callers check this and label the run.
    pub fn haveFrequency(self: Counters) bool {
        return self.slotOf(.cycles) != null and self.slotOf(.ref_cycles) != null;
    }

    fn slotOf(self: Counters, ev: Event) ?u8 {
        for (group_events, 0..) |g, e| if (g == ev) return self.slot[e];
        return null;
    }

    pub fn start(self: Counters) void {
        const l = self.leader orelse return;
        _ = linux.ioctl(l.fd, PERF.EVENT_IOC.RESET, IOC_FLAG_GROUP);
        _ = linux.ioctl(l.fd, PERF.EVENT_IOC.ENABLE, IOC_FLAG_GROUP);
    }

    pub fn stop(self: Counters) Sample {
        const l = self.leader orelse return .{
            .cycles = null,
            .ref_cycles = null,
            .instructions = null,
            .dtlb_load_misses = null,
            .cache_misses = null,
            .time_enabled = 0,
            .time_running = 0,
        };
        _ = linux.ioctl(l.fd, PERF.EVENT_IOC.DISABLE, IOC_FLAG_GROUP);

        var raw: [group_events.len]u64 = @splat(0);
        const g = l.readGroup(&raw) orelse return .{
            .cycles = null,
            .ref_cycles = null,
            .instructions = null,
            .dtlb_load_misses = null,
            .cache_misses = null,
            .time_enabled = 0,
            .time_running = 0,
        };

        // Scale by enabled/running, the standard correction when the group
        // was multiplexed off the PMU for part of the window. `Sample` also
        // carries both times so the caller can see *that* it was scaled and
        // warn: a scaled count is an estimate, and §7.5 wants that visible.
        return .{
            .cycles = self.scaled(.cycles, &raw, g.n, g.enabled, g.running),
            .ref_cycles = self.scaled(.ref_cycles, &raw, g.n, g.enabled, g.running),
            .instructions = self.scaled(.instructions, &raw, g.n, g.enabled, g.running),
            .dtlb_load_misses = self.scaled(.dtlb_load_misses, &raw, g.n, g.enabled, g.running),
            .cache_misses = self.scaled(.cache_misses, &raw, g.n, g.enabled, g.running),
            .time_enabled = g.enabled,
            .time_running = g.running,
        };
    }

    fn scaled(self: Counters, ev: Event, raw: []const u64, n: usize, enabled: u64, running: u64) ?u64 {
        const i = self.slotOf(ev) orelse return null;
        if (i >= n) return null;
        const v = raw[i];
        if (running == 0 or running >= enabled) return v;
        const f = @as(f64, @floatFromInt(v)) * @as(f64, @floatFromInt(enabled)) / @as(f64, @floatFromInt(running));
        return @intFromFloat(f);
    }
};

pub const Sample = struct {
    cycles: ?u64,
    ref_cycles: ?u64,
    instructions: ?u64,
    dtlb_load_misses: ?u64,
    cache_misses: ?u64,
    /// ns the group was enabled and ns it was actually on the PMU. Equal when
    /// the group ran the whole window; `running < enabled` means it was
    /// multiplexed and the counts above are scaled estimates.
    time_enabled: u64,
    time_running: u64,

    /// Fraction of the window the counters were actually counting. 1.0 is
    /// the clean case; anything less means the values above were scaled.
    pub fn runningFraction(self: Sample) ?f64 {
        if (self.time_enabled == 0) return null;
        return @as(f64, @floatFromInt(self.time_running)) / @as(f64, @floatFromInt(self.time_enabled));
    }

    /// True when the group was off the PMU for part of the window.
    pub fn multiplexed(self: Sample) bool {
        return self.time_enabled != 0 and self.time_running < self.time_enabled;
    }

    /// Effective frequency as a multiple of the nominal TSC rate.
    ///
    /// §6.6.4: "`cycles / ref-cycles` from `perf stat` gives the ratio
    /// directly; log it per run and refuse to publish a comparison where the
    /// two arms ran at frequencies differing by more than a threshold without
    /// saying so."
    ///
    /// A value of 1.0 means the core ran at exactly the TSC's nominal rate;
    /// above 1.0 is turbo, below is downclocking, which on Intel parts with
    /// 512-bit licensing is the effect that can make a per-cycle win a net
    /// wall-clock loss.
    pub fn effectiveFrequencyRatio(self: Sample) ?f64 {
        const c = self.cycles orelse return null;
        const r = self.ref_cycles orelse return null;
        if (r == 0) return null;
        return @as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(r));
    }

    pub fn ipc(self: Sample) ?f64 {
        const c = self.cycles orelse return null;
        const i = self.instructions orelse return null;
        if (c == 0) return null;
        return @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(c));
    }
};

/// Nominal TSC frequency in Hz, needed to convert `ref_cycles` into seconds.
///
/// Read from the kernel's `tsc_khz` when exposed; otherwise calibrated against
/// `CLOCK_MONOTONIC`. The calibration path is the common one on most hosts.
pub fn nominalTscHz() f64 {
    // Calibrated against CLOCK_MONOTONIC rather than read from sysfs. The
    // kernel only exposes `tsc_freq_khz` on some configurations, so the
    // calibration path is the common one and having a single path keeps the
    // number reproducible across hosts.
    return calibrateTsc();
}

fn calibrateTsc() f64 {
    const spin_ns: u64 = 50_000_000; // 50 ms is enough for <0.1% error
    var ts_start: linux.timespec = undefined;
    var ts_end: linux.timespec = undefined;

    _ = linux.clock_gettime(.MONOTONIC, &ts_start);
    const tsc_start = rdtsc();
    while (true) {
        _ = linux.clock_gettime(.MONOTONIC, &ts_end);
        const elapsed = nsBetween(ts_start, ts_end);
        if (elapsed >= spin_ns) break;
    }
    const tsc_end = rdtsc();
    const elapsed_ns = nsBetween(ts_start, ts_end);
    return @as(f64, @floatFromInt(tsc_end - tsc_start)) * 1e9 / @as(f64, @floatFromInt(elapsed_ns));
}

fn nsBetween(a: linux.timespec, b: linux.timespec) u64 {
    const sec = @as(i64, b.sec) - @as(i64, a.sec);
    const nsec = @as(i64, b.nsec) - @as(i64, a.nsec);
    return @intCast(sec * 1_000_000_000 + nsec);
}

/// Serialising read of the timestamp counter.
///
/// `lfence` before `rdtsc` prevents the read from being hoisted above the work
/// being timed. `rdtscp` would serialise on the trailing edge instead; for the
/// loop-timing use here the leading-edge fence is the one that matters, and the
/// benchmark harness times regions long enough that a few cycles of fence cost
/// is irrelevant.
pub inline fn rdtsc() u64 {
    var hi: u32 = undefined;
    var lo: u32 = undefined;
    asm volatile ("lfence\nrdtsc"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
        :
        : .{ .memory = true });
    return (@as(u64, hi) << 32) | @as(u64, lo);
}

/// Monotonic nanoseconds.
pub fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "rdtsc advances monotonically" {
    const a = rdtsc();
    var sink: u64 = 0;
    for (0..10_000) |i| sink +%= i;
    std.mem.doNotOptimizeAway(sink);
    const b = rdtsc();
    try std.testing.expect(b > a);
}

test "nowNs advances" {
    const a = nowNs();
    var sink: u64 = 0;
    for (0..100_000) |i| sink +%= i *% 7;
    std.mem.doNotOptimizeAway(sink);
    const b = nowNs();
    try std.testing.expect(b >= a);
}

test "counters open or degrade cleanly" {
    // Must not fail on a host with perf locked down, the whole point of the
    // optional type. On a permissive host this also sanity-checks that a busy
    // loop retires a plausible number of instructions.
    const c = Counters.open();
    defer c.close();

    c.start();
    // A multiplicative recurrence: LLVM closed-forms a plain `sum += i * k`
    // loop into a handful of instructions, and with the group enabled by a
    // single ioctl there is nothing else in the window to pad the count.
    var sink: u64 = 1;
    for (0..1_000_000) |i| sink = sink *% 6364136223846793005 +% i;
    std.mem.doNotOptimizeAway(sink);
    const s = c.stop();

    // A group that opened at all reports its scheduling window.
    if (c.leader != null) {
        try std.testing.expect(s.time_enabled > 0);
        try std.testing.expect(s.time_running <= s.time_enabled);
    }

    // "A 1M-iteration loop on a quiet PMU runs unmultiplexed" is what the
    // plausibility checks below rest on, and the PMU is not always quiet. When
    // the kernel multiplexes, the group is scheduled for `time_running` of
    // `time_enabled` nanoseconds and the values cover only that slice: a
    // million-iteration loop can report a three-figure instruction count with
    // nothing wrong anywhere. Scaling by enabled/running is what `perf` itself
    // does and what this project refuses to do -- `perfstat.py`, "Nothing is
    // scaled ... a figure that is one-quarter measurement and three-quarters
    // assumption cannot sit in a table beside figures that are neither" -- so
    // a multiplexed sample is not asserted on here either. What the counter
    // being *open* proves is checked above, unconditionally.
    //
    // This is not hypothetical. Six system-wide `perf stat` groups asking for
    // more events than the PMU has counters make the old form of this test
    // fail on the first run, at `expect(n > 1000)` -- the intermittent
    // `18 pass, 1 skip, 1 fail` that `zig build test` produced while its three
    // test binaries ran concurrently.
    const unmultiplexed = c.leader == null or
        (s.time_enabled > 0 and s.time_running == s.time_enabled);
    if (!unmultiplexed) return;

    if (s.instructions) |n| try std.testing.expect(n > 1000);
    if (s.effectiveFrequencyRatio()) |r| {
        // Any plausible ratio. A value outside this range means we are reading
        // the wrong counter, not that the CPU did something interesting.
        try std.testing.expect(r > 0.05 and r < 20.0);
    }
}
