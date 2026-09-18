//! Host memory bandwidth, measured at startup and reported in the banner.
//!
//! §5 makes the memory system the thing the engine is actually limited by, and
//! §7.2 requires a hardware baseline before any engine number means anything.
//! A qps figure without the bandwidth of the machine it ran on cannot be
//! checked against the cost model: it is a number with no denominator.
//!
//! ## Aggregate, not single-core
//!
//! `bench/micro/hw.zig` already measures bandwidth, but single-threaded. That
//! is the right figure for a per-core cost model and the wrong one for a
//! ceiling: a single core cannot saturate a wide bus, so it reads a fraction of
//! what the machine can move. Search runs across every worker, so the aggregate
//! is what caps throughput. Both are reported, and the gap between them is the
//! headroom more cores unlock.
//!
//! ## Method
//!
//! A buffer well past L3, read by one thread per core over disjoint chunks,
//! best-of-N rounds. Best rather than mean because the minimum time is the
//! least-interrupted run, which is the closest thing to the true peak that a
//! shared machine offers. Every thread warms its own chunk untimed, then all
//! of them wait on a barrier so the timed window is the copy alone, not the
//! spawns, and every chunk starts with a primed TLB and a ramped clock.
//!
//! The reader XORs the stream into wide integer accumulators rather than
//! summing floats: an FP add chain has a 3-4 cycle latency per accumulator
//! and can hold the loop below the bus; integer XOR cannot.
//!
//! The shape is conventional for a bandwidth probe: fill past the last level of
//! cache, read it back from every core at once, and keep the fastest round.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

/// Past L3 on anything current: a 24 MB L3 would otherwise serve most of it and
/// the number would describe cache, not memory.
pub const probe_bytes: usize = 128 << 20;

pub const Result = struct {
    /// All cores reading at once. The ceiling search throughput divides into.
    aggregate_gbps: f64,
    /// One core. The per-core limit the §5 cost model uses.
    single_gbps: f64,
    threads: usize,

    /// How much of the bus one core can reach.
    pub fn singleCoreFraction(self: Result) f64 {
        if (self.aggregate_gbps <= 0) return 0;
        return self.single_gbps / self.aggregate_gbps;
    }

    /// Full sweeps per second over `n` vectors of `dim` fp32.
    ///
    /// The bound every ANN structure exists to escape: an exhaustive scan must
    /// read `n * dim * 4` bytes, so bandwidth fixes a rate regardless of how
    /// good the kernels are.
    ///
    /// This is a rate of **sweeps**, not of queries, and the difference is not
    /// pedantic. Concurrent queries over one collection walk the same bytes, so
    /// they share fetched cache lines and one sweep can serve many of them. At
    /// concurrency 1 a sweep is a query and the two coincide; at concurrency Q
    /// the query ceiling is up to Q times this, approached as the sweeps
    /// synchronise.
    ///
    /// Getting this wrong is easy and was: labelling it "exhaustive scan qps"
    /// made a measured W9 of 270 look like it beat a 158 ceiling, which reads
    /// as either a broken measurement or a scan that was not exhaustive. It was
    /// neither. W9 runs 8 queries in parallel, and 270 sits exactly where
    /// partial line-sharing puts it, between 158 and 8 x 158.
    pub fn sweepsPerSecond(self: Result, n: usize, dim: usize) f64 {
        const bytes_per_sweep: f64 = @floatFromInt(n * dim * @sizeOf(f32));
        if (bytes_per_sweep <= 0) return 0;
        return self.aggregate_gbps * 1e9 / bytes_per_sweep;
    }

    /// The query ceiling at a given concurrency, assuming perfect line sharing.
    ///
    /// An upper bound, not a prediction: real sweeps drift out of step and
    /// share less than perfectly. A measured qps *above* this is the signal
    /// worth chasing, since it means the scan skipped data.
    pub fn scanQpsCeiling(self: Result, n: usize, dim: usize, concurrency: usize) f64 {
        return self.sweepsPerSecond(n, dim) * @as(f64, @floatFromInt(@max(concurrency, 1)));
    }
};

/// XOR into four independent 512-bit accumulators so the ALU never becomes
/// the bottleneck: the question is how fast bytes arrive, not how fast we can
/// combine them. Same reader as `bench/micro/hw.zig`'s bandwidth loop.
fn readChunk(chunk: []const u64, iters: usize, sink: *u64) void {
    const V = @Vector(8, u64);
    var acc: [4]V = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    for (0..iters) |_| {
        var i: usize = 0;
        while (i + 32 <= chunk.len) : (i += 32) {
            inline for (0..4) |k| {
                const v: V = chunk[i + k * 8 ..][0..8].*;
                acc[k] ^= v;
            }
        }
    }
    var s: u64 = 0;
    inline for (0..4) |k| s ^= @reduce(.Xor, acc[k]);
    sink.* = s;
}

/// Monotonic nanoseconds. Same primitive `bench/micro/perfctr.zig` uses, so a
/// startup figure and a `zig build bench` figure are timed the same way.
fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Start line for the readers: every thread warms its own chunk, reports in,
/// and waits for `go`; the timed window opens when the last one has arrived
/// and closes when the last one is done. Without it the window included the
/// spawns and joins themselves (24 sequential `Thread.spawn` calls at tens of
/// microseconds each, against a copy that takes a few milliseconds), and only
/// chunk 0 had been warmed, so the other 23 threads paid their page walks
/// inside the timed region.
const Barrier = struct {
    ready: std.atomic.Value(u32) = .init(0),
    go: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),

    fn spinUntil(v: *const std.atomic.Value(u32), want: u32) void {
        while (v.load(.acquire) < want) std.atomic.spinLoopHint();
    }
};

fn worker(chunk: []const u64, iters: usize, sink: *u64, b: *Barrier) void {
    // Untimed: ramp the clock and prime the TLB for *this* chunk, so the timed
    // rounds reflect a sustained rate rather than the cost of getting there.
    readChunk(chunk, 1, sink);
    _ = b.ready.fetchAdd(1, .acq_rel);
    Barrier.spinUntil(&b.go, 1);
    readChunk(chunk, iters, sink);
    _ = b.done.fetchAdd(1, .acq_rel);
}

fn measure(gpa: std.mem.Allocator, buf: []u64, want_threads: usize, iters: usize, rounds: usize) f64 {
    std.debug.assert(buf.len > 0);
    std.debug.assert(iters > 0 and rounds > 0);

    const threads = @max(want_threads, 1);
    const chunk = buf.len / threads;
    if (chunk == 0) return 0;

    const sinks = gpa.alloc(u64, threads) catch return 0;
    defer gpa.free(sinks);
    const handles = gpa.alloc(std.Thread, threads) catch return 0;
    defer gpa.free(handles);

    var best_ns: u64 = std.math.maxInt(u64);
    for (0..rounds) |_| {
        var barrier: Barrier = .{};
        var spawned: usize = 0;
        var inline_from: usize = threads;
        for (0..threads) |k| {
            const lo = k * chunk;
            if (std.Thread.spawn(.{}, worker, .{ buf[lo .. lo + chunk], iters, &sinks[k], &barrier })) |h| {
                handles[spawned] = h;
                spawned += 1;
            } else |_| {
                // Read the rest inline instead, so every chunk is always
                // covered and the byte count below stays exact.
                inline_from = k;
                break;
            }
        }
        // Everyone warmed and waiting; the window is the copy alone.
        Barrier.spinUntil(&barrier.ready, @intCast(spawned));
        const t0 = nowNs();
        barrier.go.store(1, .release);
        for (inline_from..threads) |k| {
            const lo = k * chunk;
            readChunk(buf[lo .. lo + chunk], iters, &sinks[k]);
        }
        Barrier.spinUntil(&barrier.done, @intCast(spawned));
        best_ns = @min(best_ns, nowNs() - t0);
        for (handles[0..spawned]) |h| h.join();
    }
    std.mem.doNotOptimizeAway(sinks[0]);
    if (best_ns == 0) return 0;

    const moved: f64 = @floatFromInt(threads * chunk * @sizeOf(u64) * iters);
    return moved / (@as(f64, @floatFromInt(best_ns)) / 1e9) / 1e9;
}

/// Probe the host. Returns null if the buffer cannot be allocated, which is not
/// fatal: the server runs fine without the figure, it just cannot report it.
pub fn probe(gpa: std.mem.Allocator) ?Result {
    const n = probe_bytes / @sizeOf(u64);
    const buf = gpa.alloc(u64, n) catch return null;
    defer gpa.free(buf);

    // Fill rather than leave untouched: fresh anonymous pages are all the same
    // zero page until written, so an unfilled buffer measures the TLB walking
    // one physical page and reports a number many times too high.
    for (buf, 0..) |*x, i| x.* = i *% 2654435761;

    const cores = std.Thread.getCpuCount() catch 1;
    const aggregate = measure(gpa, buf, cores, 4, 3);
    const single = measure(gpa, buf, 1, 2, 3);
    if (aggregate <= 0) return null;
    return .{ .aggregate_gbps = aggregate, .single_gbps = single, .threads = cores };
}

test "probe returns a plausible figure" {
    if (builtin.single_threaded) return error.SkipZigTest;
    const r = probe(std.testing.allocator) orelse return error.SkipZigTest;
    // Nothing shipping reads below 1 GB/s aggregate, and nothing reads above
    // 10 TB/s. A result outside that is a broken measurement, not a fast
    // machine, most likely the unfilled-buffer bug the fill loop above avoids.
    try std.testing.expect(r.aggregate_gbps > 1.0);
    try std.testing.expect(r.aggregate_gbps < 10_000.0);
    try std.testing.expect(r.single_gbps > 0);
    // A single core cannot exceed what every core together can move. Allow a
    // little slack for measurement jitter rather than asserting a strict
    // ordering that a noisy round could violate.
    try std.testing.expect(r.single_gbps <= r.aggregate_gbps * 1.15);
}

test "sweep rate is bandwidth over bytes per sweep" {
    const r: Result = .{ .aggregate_gbps = 100, .single_gbps = 20, .threads = 8 };
    // 1M x 128 fp32 = 512 MB per sweep; 100 GB/s allows ~195 of them.
    const sweeps = r.sweepsPerSecond(1_000_000, 128);
    try std.testing.expect(sweeps > 190 and sweeps < 200);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), r.singleCoreFraction(), 1e-9);
}

test "concurrency raises the query ceiling but not the sweep rate" {
    const r: Result = .{ .aggregate_gbps = 100, .single_gbps = 20, .threads = 8 };
    const one = r.scanQpsCeiling(1_000_000, 128, 1);
    const eight = r.scanQpsCeiling(1_000_000, 128, 8);
    try std.testing.expectApproxEqAbs(one * 8, eight, 1e-6);
    // The measured W9 (270 qps at concurrency 8) must land inside the band,
    // which is the whole point of reporting both ends of it.
    try std.testing.expect(270.0 > one and 270.0 < eight);
}
