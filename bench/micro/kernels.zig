//! §7.2 layer 2, kernel microbenchmarks. "Distance functions, hot and cold,
//! cycles/vector."
//!
//! §6.6.3 states the requirement precisely:
//!
//!   "Every kernel is benchmarked **hot (L1-resident) and cold (random over the
//!    full arena)**. The gap between those two numbers is the memory story of
//!    §5 made concrete; the hot number is the pure SIMD result."
//!
//! That pairing is the whole experiment. §6.6.1 predicts that wider SIMD helps
//! *least* on cold fp32 (bandwidth bound by ~25×) and *most* once quantization
//! has pulled the working set into cache. Running only the hot case would make
//! every kernel look ISA-sensitive; running only the cold case would make none
//! of them look sensitive. The difference between the two columns is the
//! finding.
//!
//! §7.5 fixes what each cell reports: "cycles/vector, ns/vector,
//! instructions/vector, IPC, **effective frequency** (`cycles / ref-cycles`),
//! and uops from the front end. Cycles/vector is the primary number because it
//! isolates the ISA question from the frequency question; ns/vector is what
//! actually matters and must be reported alongside it, never instead of it."

const std = @import("std");
const strawmann = @import("strawmann");
const dist = strawmann.dist;
const perfctr = @import("perfctr.zig");
const hw = @import("hw.zig");

pub const Residency = enum {
    /// One vector pair, reused. Measures the kernel with zero memory pressure -
    /// §6.6.3's "pure SIMD result".
    l1_hot,
    /// A working set sized to sit in L3 but not L2. The regime §5.4 predicts
    /// quantization moves the data into, and where §6.6.1 expects SIMD width to
    /// start paying.
    l3_resident,
    /// Random access over an arena far larger than LLC. The production regime
    /// for fp32 (§5.4: 3.07 GB for 1M×768, "DRAM only").
    dram_cold,

    pub fn name(self: Residency) []const u8 {
        return switch (self) {
            .l1_hot => "L1-hot",
            .l3_resident => "L3-resident",
            .dram_cold => "DRAM-cold",
        };
    }
};

/// How the timed loop walks the arena. The distinction is the whole reason
/// the cold columns exist twice.
///
/// §5.2 models the search path as "a dependent chain, you cannot know which
/// vector to fetch next until the current expansion has been scored", and its
/// ns/vector table is that cost. The original loop walked a random permutation
/// whose *next* address never depended on the *previous* result, so the
/// out-of-order core kept dozens of misses in flight and every "cold" cell was
/// a throughput number: hamming d=128 read 20 ns against a measured 116 ns
/// DRAM latency, which is not a dependent fetch by any reading. Both are real
/// quantities, they just answer different questions, so both are reported and
/// neither is called "cold" on its own any more.
pub const Access = enum {
    /// Next address is known before the current score returns: the OoO core
    /// overlaps misses up to the MLP limit. What a batched / prefetched scan
    /// achieves; the ceiling.
    independent,
    /// Next address depends on the current score's *value*, so no fetch can
    /// issue before the previous one has returned and been scored. What an
    /// unprefetched HNSW walk pays; §5.2's dependent-chain model.
    dependent,

    pub fn name(self: Access) []const u8 {
        return switch (self) {
            .independent => "independent (MLP)",
            .dependent => "dependent",
        };
    }
};

pub const Result = struct {
    kernel: []const u8,
    dim: usize,
    residency: Residency,
    access: Access,
    lanes: usize,
    evaluations: usize,

    ns_per_vector: f64,
    cycles_per_vector: ?f64,
    instructions_per_vector: ?f64,
    ipc: ?f64,
    /// §7.5: "A cycles/vector win presented without its frequency is not a
    /// result." Optional only because the counter may be unavailable; when it
    /// is, the reporter says so rather than omitting the column.
    effective_frequency: ?f64,

    /// Bytes of vector data touched per evaluation, from the encoding, the
    /// numerator of §5.2's model.
    bytes_per_vector: usize,

    /// Set when the perf group was multiplexed off the PMU for part of the
    /// window: the fraction it was on. The counter-derived columns are then
    /// scaled estimates and the reporter says so. `null` for a clean run.
    pmu_running_fraction: ?f64,

    /// Achieved random-access bandwidth, GB/s. Compare against the measured
    /// MLP ceiling to get §5.7's "% of modelled roofline".
    pub fn gbps(self: Result) f64 {
        return @as(f64, @floatFromInt(self.bytes_per_vector)) / self.ns_per_vector;
    }
};

/// A benchmarkable kernel: scores one query against one stored vector at
/// `arena[offset]`, all encodings erased behind a byte arena and a stride.
pub const Spec = struct {
    name: []const u8,
    /// Bytes per stored vector, *as production lays the rows out*: the 64 B
    /// padded stride of §6.4 for the fp32/fp16/u8 arenas, and the packed
    /// `dim` / `m` bytes the quantized code stores use. See `Layout`.
    stride: usize,
    /// Bytes of actual vector payload, for the bandwidth model.
    payload: usize,
    /// What the arena holds, so the driver fills it with values of the right
    /// type. Random *bytes* reinterpreted as f32/f16 are NaN, Inf and
    /// denormals about half the time, and denormal FMA operands take the slow
    /// path on Intel and cost measurably even on Zen 5, so a byte-filled arena
    /// timed the microcode assist rather than the kernel.
    elem: Elem,
    /// Score one vector. Returns f64 so every encoding can share the signature
    /// without the sink being optimised away differently per type.
    score: *const fn (query: *const anyopaque, arena: [*]const u8, offset: usize, dim: usize) f64,
};

/// Element type of a spec's arena (and, unless noted, of its query).
pub const Elem = enum {
    /// Finite normal fp32, uniform in [-1, 1].
    f32,
    /// Finite normal fp16, uniform in [-1, 1].
    f16,
    /// Random bytes: SQ8 / uint8 codes; the query is fp32 (asymmetric) or
    /// i8/u8 codes (symmetric, whole-vector u8).
    u8,
    /// Random bits, whole u64 words.
    bits,
    /// PQ codes, one centroid index in [0, 256) per subquantizer; the query
    /// is the `m x 256` fp32 lookup table.
    pq8,
};

/// Row layouts, mirroring what production stores so the bench walks the same
/// byte pattern the engine does.
///
/// The bench used to pad *every* encoding to 64 B, but the code stores do not:
/// `core/quantized.zig` packs SQ8 codes at `dim` bytes and PQ codes at `m`
/// bytes (`Store.Scalar.row`, `Store.Product.row`), and only the typed arenas
/// go through `collection.strideFor`. Padding a 96 B PQ row to 128 B changes
/// both the bytes touched and the lines crossed per vector, which is exactly
/// what the cold columns measure. Neither packed stride is a named constant in
/// production, so the two here are pinned to the `row` accessors by a test
/// below rather than imported.
pub const Layout = struct {
    /// fp32 / fp16 / uint8 arenas: `collection.strideFor`, padded to 64 B.
    pub fn typed(dim: usize, dt: dist.Datatype) usize {
        return strawmann.core.collection.strideFor(dim, dt);
    }
    /// SQ8 codes: `dim` bytes, packed.
    pub fn sq8(dim: usize) usize {
        return dim;
    }
    /// PQ8 codes: `m` bytes, packed.
    pub fn pq8(m: usize) usize {
        return m;
    }
    /// Binary codes: `paddedWordsFor(dim)` u64 words, as `quant.binary.Codes`.
    pub fn binary(dim: usize) usize {
        return strawmann.quant.binary.paddedWordsFor(dim) * @sizeOf(u64);
    }
};

/// Size of the working set for each residency class, in bytes.
///
/// L3-resident is deliberately sized to a fraction of LLC rather than to LLC
/// exactly: a working set the size of the cache thrashes it and measures
/// something between L3 and DRAM. Half of the L3 the pinned core actually
/// shares keeps the set genuinely resident; the constant this used to be
/// (8 MiB, "half of a conservative 16 MiB") was exactly the *whole* L3 of a
/// Zen 5c core on the host the tables were produced on, so an unpinned run
/// that landed on a compact core measured L3 thrash and called it resident.
/// `l3_bytes` comes from sysfs for the pinned CPU (`hw.cpuCaches`); pass the
/// fallback when sysfs is unavailable.
pub const l3_fallback_bytes: usize = 16 * 1024 * 1024;

pub fn workingSetBytes(r: Residency, l3_bytes: usize) usize {
    return switch (r) {
        .l1_hot => 16 * 1024,
        .l3_resident => @max(l3_bytes, 1024 * 1024) / 2,
        .dram_cold => 1024 * 1024 * 1024,
    };
}

/// Benchmark one kernel at one residency.
///
/// The access pattern for the non-hot cases is a **random** permutation of
/// vector slots, regenerated per run. Sequential access would let the hardware
/// prefetcher hide the entire memory system and turn every row of the table
/// into the L1-hot row, which is precisely the mistake §6.6.1 warns produces
/// "a project that measures SIMD only on cold fp32 [and] will conclude SIMD
/// doesn't matter and will be wrong about every other row."
///
/// `access` picks the loop. `.independent` walks `order` in sequence, so the
/// address of vector `i+1` is known before vector `i` has arrived and the core
/// overlaps the misses. `.dependent` folds the previous score's *value* into
/// the next index (see `dependentStep`), so the address of vector `i+1` is not
/// computable until vector `i` has been loaded *and scored*: the fetch, the
/// kernel, and the next fetch form one chain, which is what §5.2 describes.
pub fn run(
    spec: Spec,
    dim: usize,
    residency: Residency,
    access: Access,
    query: *const anyopaque,
    arena: hw.Region,
    order: []const u32,
    evaluations: usize,
) Result {
    const base = arena.bytes.ptr;
    const start = startSlot(access, order.len);

    // Warm-up: touch the same slots we are about to time, so that page faults
    // and the first-touch NUMA placement are not part of the measurement.
    // For the cold case this *also* leaves the data in whatever cache state
    // steady-state operation would produce, which is the honest baseline.
    // Each access variant warms *its own* prefix (see `startSlot`).
    {
        var sink: f64 = 0;
        var i: usize = 0;
        var wi: usize = start;
        while (i < @min(order.len, warmup_slots)) : (i += 1) {
            sink += spec.score(query, base, @as(usize, order[wi]) * spec.stride, dim);
            wi += 1;
            if (wi == order.len) wi = 0;
        }
        std.mem.doNotOptimizeAway(sink);
    }

    const counters = perfctr.Counters.open();
    defer counters.close();

    var sink: f64 = 0;
    counters.start();
    const t0 = perfctr.nowNs();
    var n: usize = 0;
    var oi: usize = start;
    switch (access) {
        .independent => {
            while (n < evaluations) : (n += 1) {
                sink += spec.score(query, base, @as(usize, order[oi]) * spec.stride, dim);
                oi += 1;
                if (oi == order.len) oi = 0;
            }
        },
        .dependent => {
            // `spec.score` is an indirect call through a function pointer, and
            // that opacity is load-bearing (found by disassembly): with the
            // kernel inlined, LLVM range-analyses e.g. popcount to [0, 64·k],
            // proves `s == floatMax(f64)` false, folds `dependentStep` to 1
            // and the "dependent" loop silently becomes the independent one.
            // Do not make `Spec.score` comptime-known here.
            while (n < evaluations) : (n += 1) {
                const s = spec.score(query, base, @as(usize, order[oi]) * spec.stride, dim);
                sink += s;
                oi += dependentStep(s);
                if (oi >= order.len) oi -= order.len;
            }
        },
    }
    const dt = perfctr.nowNs() - t0;
    const sample = counters.stop();
    std.mem.doNotOptimizeAway(sink);

    const evals_f: f64 = @floatFromInt(evaluations);
    return .{
        .kernel = spec.name,
        .dim = dim,
        .residency = residency,
        .access = access,
        .lanes = dist.native.lanes,
        .evaluations = evaluations,
        .ns_per_vector = @as(f64, @floatFromInt(dt)) / evals_f,
        .cycles_per_vector = if (sample.cycles) |c| @as(f64, @floatFromInt(c)) / evals_f else null,
        .instructions_per_vector = if (sample.instructions) |i| @as(f64, @floatFromInt(i)) / evals_f else null,
        .ipc = sample.ipc(),
        .effective_frequency = sample.effectiveFrequencyRatio(),
        .bytes_per_vector = spec.payload,
        .pmu_running_fraction = if (sample.multiplexed()) sample.runningFraction() else null,
    };
}

/// How many slots the warm-up touches before the timed loop starts.
pub const warmup_slots: usize = 4096;

/// Where in `order` each access variant starts its warm-up and its timed walk.
///
/// The two variants are run back to back over the *same* `order` and the same
/// arena. Both used to start at `order[0]`, so the dependent row's first slots
/// were exactly the ones the independent row and its warm-up had just pulled
/// into cache: at `--quick` evaluation counts that is a large fraction of the
/// walk (measured 86.9 vs 130.5 ns/vector on DRAM-cold hamming, ~5% at the
/// full count). The dependent variant therefore starts half a permutation
/// away, so its first `warmup_slots + evaluations` slots are disjoint from
/// what was just touched for any `evaluations` under `order.len / 2`, and
/// its warm-up (which is what makes the *steady-state* cache state honest)
/// covers its own prefix rather than the other variant's.
pub fn startSlot(access: Access, slots: usize) usize {
    return switch (access) {
        .independent => 0,
        .dependent => slots / 2,
    };
}

/// The step from one slot to the next in the dependent loop.
///
/// Always 1 at run time (no finite score equals `floatMax`), but the compiler
/// cannot know that and the hardware cannot predict it: x86 has no value
/// prediction, so the `setcc` that produces the step waits for the score, the
/// address arithmetic waits for the step, and the next load waits for the
/// address. The chain is a data dependency, not a branch, so the branch
/// predictor cannot speculate through it either. Folding the score in as a
/// *value* rather than chasing a separate index array is what makes the fetch
/// depend on the vector's own bytes: an `order[order[i]]` chase would
/// serialise the index loads and leave the vector loads free to overlap.
inline fn dependentStep(score: f64) usize {
    return 1 + @as(usize, @intFromBool(score == std.math.floatMax(f64)));
}

/// Build a random visit order over `slots` vector positions.
///
/// Returned as u32 because internal offsets are u32 throughout the engine
/// (§3: "Internal offsets are the currency of the entire engine").
pub fn randomOrder(alloc: std.mem.Allocator, slots: usize, rnd: std.Random) ![]u32 {
    const order = try alloc.alloc(u32, slots);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    var i: usize = slots;
    while (i > 1) {
        i -= 1;
        const j = rnd.uintLessThan(usize, i + 1);
        std.mem.swap(u32, &order[i], &order[j]);
    }
    return order;
}

// -------------------------------------------------------------------------
// Kernel specs
// -------------------------------------------------------------------------

pub fn dotF32Spec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const f32 = @ptrCast(@alignCast(query));
            const v: [*]const f32 = @ptrCast(@alignCast(arena + offset));
            return dist.native.dot(q[0..d], v[0..d]);
        }
    };
    return .{
        .name = "dot_f32",
        .stride = Layout.typed(dim, .float32),
        .payload = dim * 4,
        .elem = .f32,
        .score = S.score,
    };
}

pub fn euclidF32Spec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const f32 = @ptrCast(@alignCast(query));
            const v: [*]const f32 = @ptrCast(@alignCast(arena + offset));
            return dist.native.euclid(q[0..d], v[0..d]);
        }
    };
    return .{
        .name = "l2_f32",
        .stride = Layout.typed(dim, .float32),
        .payload = dim * 4,
        .elem = .f32,
        .score = S.score,
    };
}

pub fn dotF16Spec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const f16 = @ptrCast(@alignCast(query));
            const v: [*]const f16 = @ptrCast(@alignCast(arena + offset));
            return dist.dot_f16.dot_native.call(q[0..d], v[0..d]);
        }
    };
    return .{
        .name = "dot_f16",
        .stride = Layout.typed(dim, .float16),
        .payload = dim * 2,
        .elem = .f16,
        .score = S.score,
    };
}

/// SQ8 asymmetric: fp32 query against u8 codes.
pub fn dotI8AsymSpec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const f32 = @ptrCast(@alignCast(query));
            return dist.dot_i8.f32u8_native.call(q[0..d], (arena + offset)[0..d]);
        }
    };
    return .{
        .name = "dot_i8_asym",
        .stride = Layout.sq8(dim),
        .payload = dim,
        .elem = .u8,
        .score = S.score,
    };
}

/// SQ8 asymmetric Euclid: the arm W6 actually runs on a Euclid dataset.
///
/// It sat beside `dot_i8_asym` as a scalar loop while that one was a vector
/// kernel, which is the kind of gap a matrix row makes visible and a code
/// reading does not. Measured at d=128 when it was replaced: 87.3 ns/call
/// scalar against 6.2 ns/call vector.
pub fn euclidI8AsymSpec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const f32 = @ptrCast(@alignCast(query));
            return dist.dot_i8.euclid_f32u8_native.call(q[0..d], (arena + offset)[0..d], .{ .lo = -1.25, .alpha = 0.0098 });
        }
    };
    return .{
        .name = "euclid_i8_asym",
        .stride = Layout.sq8(dim),
        .payload = dim,
        .elem = .u8,
        .score = S.score,
    };
}

/// uint8 storage: the whole-vector Euclid over two stored rows.
pub fn euclidU8Spec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const u8 = @ptrCast(query);
            return @floatFromInt(dist.dot_i8.euclid_u8_native.call(q[0..d], (arena + offset)[0..d]));
        }
    };
    return .{
        .name = "euclid_u8",
        // uint8 *storage*, not SQ8 codes: this one is a typed arena.
        .stride = Layout.typed(dim, .uint8),
        .payload = dim,
        .elem = .u8,
        .score = S.score,
    };
}

/// SQ8 symmetric: i8 query codes against u8 data codes, the `vpdpbusd` shape.
pub fn dotI8SymSpec(dim: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const q: [*]const i8 = @ptrCast(@alignCast(query));
            return @floatFromInt(dist.dot_i8.u8i8_native.call((arena + offset)[0..d], q[0..d]));
        }
    };
    return .{
        .name = "dot_i8_sym",
        .stride = Layout.sq8(dim),
        .payload = dim,
        .elem = .u8,
        .score = S.score,
    };
}

pub fn hammingSpec(dim: usize) Spec {
    // The *padded* word count, matching what the engine stores. Benchmarking
    // the logical count would measure a code path the engine no longer takes:
    // `quant/binary.paddedWordsFor` rounds rows up to a whole vector register
    // precisely so the kernel never falls back to a scalar tail, and measuring
    // the unpadded call would keep reporting the tail-bound numbers that fix
    // was made to remove.
    const words = strawmann.quant.binary.paddedWordsFor(dim);
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, d: usize) f64 {
            const w = strawmann.quant.binary.paddedWordsFor(d);
            const q: [*]const u64 = @ptrCast(@alignCast(query));
            const v: [*]const u64 = @ptrCast(@alignCast(arena + offset));
            return @floatFromInt(dist.hamming.native.call(q[0..w], v[0..w]));
        }
    };
    return .{
        .name = "hamming",
        .stride = Layout.binary(dim),
        .payload = words * 8,
        .elem = .bits,
        .score = S.score,
    };
}

/// PQ8 ADC. The "vector" in the arena is the code array and the query is the
/// precomputed lookup table, so the `dim` parameter carries the subquantizer
/// count `m` rather than the original dimension, for PQ, `m` is what sets both
/// the work per evaluation and the bytes touched, and the original dimension
/// does not appear in the kernel at all.
pub fn pqAdc8Spec(m_count: usize) Spec {
    const S = struct {
        fn score(query: *const anyopaque, arena: [*]const u8, offset: usize, m: usize) f64 {
            const table: [*]const f32 = @ptrCast(@alignCast(query));
            return dist.pq_adc.Pq8Adc.score(table[0 .. m * 256], (arena + offset)[0..m]);
        }
    };
    return .{
        .name = "pq_adc_8",
        .stride = Layout.pq8(m_count),
        .payload = m_count,
        .elem = .pq8,
        .score = S.score,
    };
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "typed layout pads to the 64 B row stride of §6.4" {
    try std.testing.expectEqual(@as(usize, 64), Layout.typed(1, .float32));
    try std.testing.expectEqual(@as(usize, 64), Layout.typed(64, .uint8));
    try std.testing.expectEqual(@as(usize, 128), Layout.typed(65, .uint8));
    // d=768 fp32 is 3072, already a multiple of 64, §6.4 notes this.
    try std.testing.expectEqual(@as(usize, 3072), Layout.typed(768, .float32));
    // d=100 fp32 is 400 -> 448, where the padding is real.
    try std.testing.expectEqual(@as(usize, 448), Layout.typed(100, .float32));
}

test "packed layouts agree with the production code stores" {
    // `Store.Scalar.row` and `Store.Product.row` are the layout; the bench's
    // strides are mirrors of them, and this is what keeps the mirrors honest.
    const quantized = strawmann.core.quantized;
    var codes: [4096]u8 = undefined;
    for ([_]usize{ 96, 128, 384 }) |dim| {
        const sq: quantized.Store.Scalar = .{ .params = undefined, .codes = &codes, .stats = &.{}, .dim = dim };
        const sq_stride = @intFromPtr(sq.row(1).ptr) - @intFromPtr(sq.row(0).ptr);
        try std.testing.expectEqual(sq_stride, Layout.sq8(dim));
        try std.testing.expectEqual(dim, sq.row(0).len);
    }
    for ([_]usize{ 16, 48, 96, 192 }) |m| {
        const pq: quantized.Store.Product = .{
            .codebook = .{ .dim = m * 8, .m = m, .sub_dim = 8, .centroids = 256, .data = &.{} },
            .codes = &codes,
        };
        const pq_stride = @intFromPtr(pq.row(1).ptr) - @intFromPtr(pq.row(0).ptr);
        try std.testing.expectEqual(pq_stride, Layout.pq8(m));
        try std.testing.expectEqual(m, pq.row(0).len);
    }
    // Binary rows are `paddedWordsFor` words, as `quant.binary.Codes` stores.
    var bc = try strawmann.quant.binary.Codes.init(std.testing.allocator, 768, 2);
    defer bc.deinit(std.testing.allocator);
    const b_stride = @intFromPtr(bc.row(1).ptr) - @intFromPtr(bc.row(0).ptr);
    try std.testing.expectEqual(b_stride, Layout.binary(768));
}

test "the dependent step is always one, and cannot be folded" {
    // If this ever returns 2 the dependent loop would skip slots; if the
    // compiler could prove it 1 the loop would lose its dependency. The first
    // is testable, the second is checked by the ns/vector gap it produces.
    try std.testing.expectEqual(@as(usize, 1), dependentStep(0.0));
    try std.testing.expectEqual(@as(usize, 1), dependentStep(-3.5));
    try std.testing.expectEqual(@as(usize, 1), dependentStep(1e300));
    try std.testing.expectEqual(@as(usize, 2), dependentStep(std.math.floatMax(f64)));
}

test "dependent access costs at least as much as independent over a cold arena" {
    // The property the two columns exist to show. Loose on purpose: it asserts
    // the ordering, not a host's numbers, and uses a small arena so it runs in
    // the unit-test budget. In Debug the unoptimised kernel dwarfs the miss
    // cost and the ordering does not hold, so it is a ReleaseFast-only check.
    if (@import("builtin").mode == .Debug) return error.SkipZigTest;
    const dim = 128;
    const spec = hammingSpec(dim);
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    const slots: usize = 64 * 1024 * 1024 / spec.stride;
    const arena = try hw.Region.alloc(spec.stride * slots, .small_pages);
    defer arena.free();
    for (std.mem.bytesAsSlice(u64, arena.bytes)) |*x| x.* = rnd.int(u64);
    const order = try randomOrder(std.testing.allocator, slots, rnd);
    defer std.testing.allocator.free(order);
    var query: [64]u64 = undefined;
    for (&query) |*x| x.* = rnd.int(u64);

    const ind = run(spec, dim, .dram_cold, .independent, &query, arena, order, 100_000);
    const dep = run(spec, dim, .dram_cold, .dependent, &query, arena, order, 100_000);
    try std.testing.expect(dep.ns_per_vector >= ind.ns_per_vector);
}

test "the two access variants start their walks on disjoint slot prefixes" {
    // The property behind `startSlot`: for any order length, the first N
    // slots the independent variant touches (warm-up + timed, from slot 0)
    // and the first N the dependent variant touches share nothing, for every
    // N up to half the order. Checked on the *slots* rather than the
    // offsets, so a change to either walk that made them collide again would
    // show up here rather than as a 5% skew in a table.
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for ([_]usize{ 2, 3, 17, 4096, 8192 + 1, 100_003 }) |slots| {
        const order = try randomOrder(std.testing.allocator, slots, rnd);
        defer std.testing.allocator.free(order);
        const n = slots / 2;
        const ind_start = startSlot(.independent, slots);
        const dep_start = startSlot(.dependent, slots);
        try std.testing.expect(ind_start < slots);
        try std.testing.expect(dep_start < slots);
        var touched = try std.DynamicBitSet.initEmpty(std.testing.allocator, slots);
        defer touched.deinit();
        for (0..n) |k| touched.set(order[(ind_start + k) % slots]);
        for (0..n) |k| {
            const slot = order[(dep_start + k) % slots];
            try std.testing.expect(!touched.isSet(slot));
        }
    }
    // The warm-up prefix specifically, at a realistic order length: the
    // dependent warm-up must not overlap the independent timed walk either.
    const slots: usize = 1 << 20;
    const n = warmup_slots + slots / 4;
    try std.testing.expect(startSlot(.dependent, slots) >= n);
    try std.testing.expect(startSlot(.dependent, slots) + n <= slots);
}

test "the dependent loop scores through an opaque function pointer, not a comptime-known fn" {
    // If `Spec.score` ever became a comptime-known function (or `Spec` a
    // comptime parameter of `run`), LLVM would inline the kernel into the
    // dependent loop, range-analyse its result, fold `dependentStep` to 1 and
    // delete the dependency the whole column exists to measure. Pin the type.
    const info = @typeInfo(@FieldType(Spec, "score"));
    try std.testing.expect(info == .pointer);
    try std.testing.expect(@typeInfo(info.pointer.child) == .@"fn");
    // And `run` takes the spec at runtime, so the pointer's target is unknown
    // to the optimiser at the call site.
    const RunFn = @TypeOf(run);
    const params = @typeInfo(RunFn).@"fn".params;
    try std.testing.expect(!params[0].is_generic);
    try std.testing.expectEqual(Spec, params[0].type.?);
    // A spec built here still routes through the pointer, and scores.
    const spec = hammingSpec(64);
    try std.testing.expect(@TypeOf(spec.score) == *const fn (*const anyopaque, [*]const u8, usize, usize) f64);
    try std.testing.expectEqual(@as(usize, 1), dependentStep(spec.score(&[_]u64{0}, @ptrCast(&[_]u64{0}), 0, 64)));
}

test "randomOrder is a permutation" {
    var prng = std.Random.DefaultPrng.init(3);
    const order = try randomOrder(std.testing.allocator, 1000, prng.random());
    defer std.testing.allocator.free(order);

    var seen = try std.DynamicBitSet.initEmpty(std.testing.allocator, 1000);
    defer seen.deinit();
    for (order) |o| {
        try std.testing.expect(!seen.isSet(o));
        seen.set(o);
    }
    try std.testing.expectEqual(@as(usize, 1000), seen.count());
}

test "a kernel spec scores the same value the direct call does" {
    const dim = 128;
    const spec = dotF32Spec(dim);
    var query: [dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (&query) |*x| x.* = rnd.floatNorm(f32);

    const arena = try hw.Region.alloc(spec.stride * 4, .small_pages);
    defer arena.free();
    const vecs = std.mem.bytesAsSlice(f32, arena.bytes);
    for (vecs) |*x| x.* = rnd.floatNorm(f32);

    const slot: usize = 2;
    const via_spec = spec.score(&query, arena.bytes.ptr, slot * spec.stride, dim);
    const stored = vecs[slot * spec.stride / 4 ..][0..dim];
    const direct = dist.native.dot(&query, stored);
    try std.testing.expectEqual(@as(f64, direct), via_spec);
}
