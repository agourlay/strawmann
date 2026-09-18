//! Shared comptime scaffolding for the distance kernels.
//!
//! §6.6.3: "Each kernel is written once as a comptime-generic Zig function
//! parameterised by lane count and element type, then instantiated per ISA
//! target. Zig's comptime is a genuine advantage here: one source of truth, N
//! machine-code variants, no macro soup and no hand-maintained intrinsic
//! duplicates."
//!
//! Everything in this file is comptime-only glue. It contains no policy.

const std = @import("std");
const builtin = @import("builtin");

/// §6.6.3: "≥8 independent accumulators for f32 (FMA latency 4, two ports → 8
/// in flight to saturate)."
///
/// This is the number of *independent dependency chains* in the inner loop, not
/// a width. It exists to cover FMA latency, so it is a function of the
/// microarchitecture's FMA latency × issue width, not of the vector register
/// size. 8 covers 4-cycle latency on two ports, which is the common case on
/// both recent Intel and Zen.
pub const default_accumulators = 8;

/// Native lane count for `T` on the compiled target.
///
/// This is the hinge of the whole ISA matrix: the *source* never names a width,
/// so a `-Dcpu=x86_64_v4` build and a `-Dcpu=x86_64_v3` build of the same
/// function differ only in what this resolves to. §6.6.5.
pub fn nativeLanes(comptime T: type) usize {
    return std.simd.suggestVectorLength(T) orelse @max(1, 16 / @sizeOf(T));
}

/// Fixed-order pairwise tree reduction over `N` accumulators.
///
/// §8.7: "Fixed reduction order in every kernel." The tree shape is decided at
/// comptime and is identical for every input length, so the kernel is
/// bit-reproducible for a given build. Reassociation *within* this tree is not
/// left to the optimiser's discretion at runtime, the shape is written here.
///
/// `N` must be a power of two so the tree is balanced and the shape is
/// unambiguous. That is not a limitation in practice: the accumulator count is
/// chosen to cover FMA latency, and 4/8/16 are the useful values.
pub fn treeReduce(comptime V: type, comptime N: usize, acc: [N]V) V {
    comptime std.debug.assert(N > 0);
    comptime std.debug.assert(std.math.isPowerOfTwo(N));
    if (N == 1) return acc[0];

    var a = acc;
    comptime var n: usize = N;
    inline while (n > 1) {
        const half = n / 2;
        inline for (0..half) |k| a[k] = a[k] + a[k + half];
        n = half;
    }
    return a[0];
}

/// Whether the compiled target has a hardware fused multiply-add for floats.
///
/// On x86 that is the FMA feature (Haswell / Piledriver and later, part of
/// x86_64_v3). AArch64 has had `fmla` since ARMv8.0, so it is always true
/// there. Anything else is assumed not to.
pub const has_hw_fma = switch (builtin.cpu.arch) {
    .x86, .x86_64 => builtin.cpu.has(.x86, .fma),
    .aarch64, .aarch64_be => true,
    else => false,
};

/// `a * b + c`, fused when the target can do it in hardware and as two
/// separately-rounded ops otherwise.
///
/// Why this exists rather than writing `@mulAdd` everywhere: `@mulAdd` is a
/// *semantic* request for a single rounding, so on a target without FMA (the
/// `baseline` arm of the §6.6.5 matrix, plain x86_64 = SSE2) LLVM must honour
/// it and lowers every vector lane to a `call fmaf` libcall. That turned the
/// baseline dot product into 36 function calls per block, and made the
/// "control arm" of the benchmark matrix measure compiler_rt rather than SSE2
/// (docs/asm/baseline/sm_dot_f32.asm before this helper). Reporting that as
/// "SSE2 is 40x slower than AVX2" would be a lie about the ISA.
///
/// The cost is numerical: the two paths round differently (one rounding vs
/// two), so a kernel's bit pattern differs between an FMA and a non-FMA build
/// of the same source. Within a single build the result is still fixed-order
/// and bit-reproducible (§8.7); §8.4/§8.5 tolerances already allow for this
/// because Qdrant's own SSE and AVX paths differ the same way. §11 lists
/// "fails to contract FMA" as a risk; on FMA hardware this helper *is* an
/// explicit `@mulAdd`, so that risk is still removed there.
///
/// `V` may be a scalar float or a vector of floats.
pub inline fn mulAdd(comptime V: type, a: V, b: V, c: V) V {
    if (has_hw_fma) return @mulAdd(V, a, b, c);
    return a * b + c;
}

/// A **correctly rounded** `x * y + z` in f32, without hardware FMA.
///
/// Why `@mulAdd(f32, ...)` is not enough on a no-FMA build: LLVM lowers it to
/// compiler_rt's `fmaf`, and Zig 0.16's `lib/compiler_rt/fma.zig` computes
/// `@floatCast(@as(f64, x) * y + z)`, which rounds twice (once to f64, once to
/// f32) and is therefore *not* the single rounding an FMA promises.
/// Counterexample: `x = 4.0009613, y = 15.996156, z = 2^30` gives 1073741952
/// on hardware and 1073741824 from compiler_rt. `norm.zig`'s replica of
/// Qdrant's `cosine_preprocess_avx` needs the hardware answer whichever ISA
/// arm *we* were built for, because the value feeds a discontinuous branch.
///
/// The product of two f32s is exact in f64 (48 significant bits into 53), so
/// the only error is in the f64 addition, and Knuth's TwoSum recovers that
/// error exactly. The double-rounding failure is confined to one case: the
/// f64 sum lies *exactly* halfway between two adjacent f32s while the true
/// value does not. musl's `fmaf` detects it by bit pattern and resolves it by
/// re-adding under round-toward-zero; there is no `fesetround` here, so the
/// TwoSum error term supplies the direction instead. Everything else,
/// including f32 subnormal results, is handled by the general halfway test.
pub fn fmafExact(x: f32, y: f32, z: f32) f32 {
    const xy: f64 = @as(f64, x) * @as(f64, y); // exact
    const zz: f64 = z;
    const s = xy + zz;
    // TwoSum: s + err == xy + zz exactly (no overflow possible: |xy| < 2^257).
    const bb = s - xy;
    const err = (xy - (s - bb)) + (zz - bb);

    const r: f32 = @floatCast(s);
    if (!std.math.isFinite(r) or err == 0) return r;
    const rr: f64 = r;
    if (rr == s) return r;
    // s is not an f32. Its other f32 neighbour is one step from r toward s.
    const other: f32 = std.math.nextAfter(f32, r, if (s > rr) std.math.inf(f32) else -std.math.inf(f32));
    const mid: f64 = (rr + @as(f64, other)) * 0.5; // exact: adjacent f32s
    if (s != mid) return r;
    // Exactly halfway, and inexact: the true value is on `err`'s side of s.
    const toward_other = (@as(f64, other) > rr) == (err > 0);
    return if (toward_other) other else r;
}

/// `x * y + z` in f32 with exactly one rounding, on every target: the
/// hardware instruction where there is one, `fmafExact` otherwise.
pub inline fn fmaf(x: f32, y: f32, z: f32) f32 {
    if (has_hw_fma) return @mulAdd(f32, x, y, z);
    return fmafExact(x, y, z);
}

/// Issue a read prefetch for `ptr` into L1.
///
/// §6.5: "on expanding a node, first issue prefetcht0 for the neighbours'
/// vector rows (all of them, since the list is 1–2 cache lines and already
/// resident), then score."
///
/// §5.2 is the reason this exists at all: below ~12 cache lines per vector a
/// single dependent fetch no longer fills the memory pipeline, so the measured
/// cost collapses to raw DRAM latency. Prefetch is what restores memory-level
/// parallelism once quantization has shrunk vectors below the LFB threshold.
pub inline fn prefetchRead(ptr: anytype) void {
    @prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
}

/// Prefetch an entire row of `bytes` bytes starting at `ptr`, one hint per
/// cache line. Used on the neighbour vectors before scoring them.
pub inline fn prefetchRow(ptr: [*]const u8, bytes: usize) void {
    const line = 64;
    var off: usize = 0;
    while (off < bytes) : (off += line) {
        @prefetch(ptr + off, .{ .rw = .read, .locality = 3, .cache = .data });
    }
}

/// Load `L` contiguous elements starting at `slice[off]` as a vector.
///
/// §2: "Beware alignment: the payload is not guaranteed 4-byte aligned in the
/// frame buffer." Our arenas *are* 64 B aligned (§6.4) but query vectors
/// decoded straight out of a frame buffer are not, so every kernel load is
/// expressed as an unaligned load. On every microarchitecture this project
/// targets, an unaligned load that happens to be aligned costs the same as an
/// aligned one, so there is nothing to win by splitting the two cases.
pub inline fn loadVec(comptime V: type, comptime L: usize, comptime T: type, slice: []const T, off: usize) V {
    return @as(V, slice[off..][0..L].*);
}

test "treeReduce is a balanced fixed-shape tree" {
    const V = @Vector(4, f32);
    const acc: [4]V = .{
        @splat(1.0),
        @splat(2.0),
        @splat(3.0),
        @splat(4.0),
    };
    const r = treeReduce(V, 4, acc);
    try std.testing.expectEqual(@as(f32, 10.0), r[0]);
}

test "treeReduce of a single accumulator is the identity" {
    const V = @Vector(8, f32);
    const acc: [1]V = .{@splat(3.5)};
    const r = treeReduce(V, 1, acc);
    try std.testing.expectEqual(@as(f32, 3.5), r[3]);
}

test "nativeLanes is at least 4 for f32 on x86_64" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    try std.testing.expect(nativeLanes(f32) >= 4);
}

test "mulAdd agrees with @mulAdd on FMA hardware and with a*b+c otherwise" {
    const V = @Vector(4, f32);
    const a: V = .{ 1.5, -2.25, 3.0e-3, 7.0 };
    const b: V = .{ 2.0, 4.0, 1.0e3, -0.5 };
    const c: V = .{ 0.25, 1.0, -3.0, 100.0 };
    const got = mulAdd(V, a, b, c);
    const want: V = if (has_hw_fma) @mulAdd(V, a, b, c) else a * b + c;
    inline for (0..4) |i| try std.testing.expectEqual(want[i], got[i]);
    // Scalar instantiation as well; pq_adc's dotSmall uses it.
    try std.testing.expectEqual(
        if (has_hw_fma) @mulAdd(f32, 3.0, 4.0, 5.0) else @as(f32, 3.0 * 4.0 + 5.0),
        mulAdd(f32, 3.0, 4.0, 5.0),
    );
}

test "fmafExact fixes the compiler_rt double-rounding counterexample" {
    // Hardware fmaf(4.0009613, 15.996156, 2^30) = 1073741952; the double-rounded
    // `@floatCast(f64 x*y + z)` gives 1073741824.
    const x: f32 = 4.0009613;
    const y: f32 = 15.996156;
    const z: f32 = 1073741824.0;
    try std.testing.expectEqual(@as(f32, 1073741952.0), fmafExact(x, y, z));
    // And the double-rounded expression really is the wrong one here, so the
    // test cannot pass by accident of the inputs being benign.
    const double_rounded: f32 = @floatCast(@as(f64, x) * @as(f64, y) + @as(f64, z));
    try std.testing.expectEqual(@as(f32, 1073741824.0), double_rounded);
    if (has_hw_fma) try std.testing.expectEqual(@mulAdd(f32, x, y, z), fmafExact(x, y, z));
}

test "fmafExact equals hardware fma bit for bit across random inputs and exponent gaps" {
    if (!has_hw_fma) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0xf3a);
    const rnd = prng.random();
    var halfway_hits: usize = 0;
    var i: usize = 0;
    while (i < 2_000_000) : (i += 1) {
        // Random signs and magnitudes, with the exponent gap between the
        // product and the addend swept over the whole range where the halfway
        // case can arise (gap up to ~50 either way) and beyond.
        const ex: i32 = rnd.intRangeAtMost(i32, -60, 60);
        const ey: i32 = rnd.intRangeAtMost(i32, -60, 60);
        const ez: i32 = rnd.intRangeAtMost(i32, -120, 120);
        const x = std.math.ldexp(rnd.float(f32) * 2.0 - 1.0, ex);
        const y = std.math.ldexp(rnd.float(f32) * 2.0 - 1.0, ey);
        const z = std.math.ldexp(rnd.float(f32) * 2.0 - 1.0, ez);
        const hw = @mulAdd(f32, x, y, z);
        const sw = fmafExact(x, y, z);
        try std.testing.expectEqual(@as(u32, @bitCast(hw)), @as(u32, @bitCast(sw)));
        // Count the cases where the double-rounded answer would have differed,
        // so the loop is known to have exercised the fix-up.
        const dr: f32 = @floatCast(@as(f64, x) * @as(f64, y) + @as(f64, z));
        if (@as(u32, @bitCast(dr)) != @as(u32, @bitCast(hw))) halfway_hits += 1;
    }
    try std.testing.expect(halfway_hits > 0);

    // Constructed halfway cases: z = 2^k with a product whose low bits sit just
    // above or below the f32 half-ulp of z. Also f32 subnormal and huge results.
    const specials = [_][3]f32{
        .{ 4.0009613, 15.996156, 1073741824.0 },
        .{ 1.0 + 0x1p-23, 1.0 + 0x1p-23, -1.0 },
        .{ 0x1p-100, 0x1p-40, 0x1p-149 },
        .{ 0x1p-75, 0x1p-75, 0x1p-149 },
        .{ 3.0e38, 1.5, -1.0e38 },
        .{ 0x1.000002p0, 0x1p-24, 1.0 },
        .{ -0x1.000002p0, 0x1p-25, 1.0 },
        .{ 0.0, 1.0, -0.0 },
    };
    for (specials) |t| {
        const hw = @mulAdd(f32, t[0], t[1], t[2]);
        const sw = fmafExact(t[0], t[1], t[2]);
        try std.testing.expectEqual(@as(u32, @bitCast(hw)), @as(u32, @bitCast(sw)));
    }
}

test "fmaf is the hardware instruction on FMA parts and the exact software path otherwise" {
    const got = fmaf(4.0009613, 15.996156, 1073741824.0);
    try std.testing.expectEqual(@as(f32, 1073741952.0), got);
    try std.testing.expectEqual(fmafExact(4.0009613, 15.996156, 1073741824.0), got);
}
