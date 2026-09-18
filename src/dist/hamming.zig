//! `hamming`, the binary-quantization distance kernel.
//!
//! §6.6.2 singles this out: `vpopcntq` (AVX512_VPOPCNTDQ) does in 1 instruction
//! per 64 B what AVX2 needs ~5–8 instructions per 32 B to do via Muła's
//! nibble-LUT (`vpshufb` + `vpsadbw`) or Harley–Seal. It calls the gap "the
//! single biggest ISA gap in the whole engine".
//!
//! §5.2 explains why this kernel is also where prefetching starts to matter
//! most: at 96 B/vector (d=768 binary) a vector is 2 cache lines, well under
//! the ~12-line threshold at which a single dependent fetch saturates the line
//! fill buffers. Without explicit prefetch the measured cost collapses to raw
//! DRAM latency (~90 ns) rather than the ~15 ns the bandwidth model predicts -
//! a 6× gap that has nothing to do with the ALU.
//!
//! ## One source, both instruction sequences
//!
//! Zig's `@popCount` on a `@Vector(N, u64)` lowers to `vpopcntq` when the
//! target has AVX512_VPOPCNTDQ and to LLVM's `vpshufb` nibble-LUT expansion
//! when it does not. That is exactly the two sequences §6.6.2 wants compared,
//! obtained from one line of source with no intrinsics and no duplication -
//! so the ISA arms of §7.5 differ only in the compiler's lowering, which is the
//! property that makes the comparison meaningful.
//!
//! `docs/asm/` holds the disassembly of both, diffed in CI per §6.6.3, because
//! "a silent change in vectorisation is a regression even if the wall-clock
//! didn't move on the current host."

const std = @import("std");
const common = @import("common.zig");

/// Hamming distance over packed bit vectors, at a fixed lane count.
///
/// `L` is the number of **u64 lanes**, so the register width is `L × 64` bits:
/// L=2 is SSE, L=4 is AVX2, L=8 is AVX-512.
pub fn Hamming(comptime L: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));

    return struct {
        pub const lanes = L;
        pub const bits_per_word = 64;

        const V = @Vector(L, u64);

        /// Population count of `a XOR b`, the number of differing bits.
        ///
        /// Both slices are packed bit vectors of equal word length. The caller
        /// is responsible for zeroing any padding bits in the final word; see
        /// `wordsFor` and the note on `paddingIsZero`. Padding that is not
        /// zeroed in *both* operands contributes spurious differences.
        pub fn call(a: []const u64, b: []const u64) u32 {
            std.debug.assert(a.len == b.len);

            // Four independent accumulator chains. popcount latency is ~3
            // cycles on the parts we target, so fewer chains than the f32 path
            // needs, and u64 accumulators cannot overflow for any realistic
            // vector length.
            const nacc = 4;
            var acc: [nacc]V = @splat(@as(V, @splat(0)));
            const block = L * nacc;
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..nacc) |k| {
                    const off = i + k * L;
                    const av: V = a[off..][0..L].*;
                    const bv: V = b[off..][0..L].*;
                    acc[k] += @popCount(av ^ bv);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: V = a[i..][0..L].*;
                const bv: V = b[i..][0..L].*;
                acc[0] += @popCount(av ^ bv);
            }

            const folded = common.treeReduce(V, nacc, acc);
            var total: u64 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) total += @popCount(a[i] ^ b[i]);
            return @intCast(total);
        }

        /// Internal similarity for binary quantization: `−hamming`.
        ///
        /// Negated so that "higher is better" holds uniformly across every
        /// kernel in `dist/` and the top-k heap never needs to know which
        /// encoding produced its scores.
        pub fn similarity(a: []const u64, b: []const u64) f32 {
            return -@as(f32, @floatFromInt(call(a, b)));
        }
    };
}

pub const native = Hamming(common.nativeLanes(u64));

/// Words needed to pack `dim` bits.
pub fn wordsFor(dim: usize) usize {
    return (dim + 63) / 64;
}

/// Pack the sign bits of `v` into `dst`, the standard binary quantizer.
///
/// §6.7: "`binary` | 1 bit/dim, sign-based". A bit is set when the component is
/// **strictly positive**; zero and negative both clear it. That tie-break has
/// to match Qdrant's or the encodings diverge on any dataset containing exact
/// zeros, which sparse-ish embeddings frequently do.
///
/// Padding bits in the final word are zeroed, which `call` relies on: two
/// vectors with unzeroed padding would report differences in bits that do not
/// exist.
pub fn packSigns(dst: []u64, v: []const f32) void {
    // `>=` rather than `==`: callers may pad the row up to a whole number of
    // vector registers (see `binary.paddedWordsFor`). The memset covers the
    // padding, and zero words XOR to zero and popcount to zero, so padding
    // contributes nothing to any distance.
    std.debug.assert(dst.len >= wordsFor(v.len));
    @memset(dst, 0);
    for (v, 0..) |x, i| {
        if (x > 0.0) dst[i / 64] |= @as(u64, 1) << @intCast(i % 64);
    }
}

/// Recover the `±1` dot product from a Hamming distance.
///
/// For sign vectors, `Σ sign(a)·sign(b) = dim − 2·hamming`. Used when a binary
/// score has to be comparable with an fp32 one, the rescore stage of §6.7.
pub fn signDotFromHamming(dim: usize, h: u32) f32 {
    return @as(f32, @floatFromInt(dim)) - 2.0 * @as(f32, @floatFromInt(h));
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

fn hammingReference(a: []const u64, b: []const u64) u32 {
    var t: u32 = 0;
    for (a, b) |x, y| t += @popCount(x ^ y);
    return t;
}

test "hamming matches the scalar reference across widths and word counts" {
    var prng = std.Random.DefaultPrng.init(0xba1a1);
    const rnd = prng.random();
    // Word counts chosen to exercise the full block, the vector tail and the
    // scalar epilogue for every lane count.
    const counts = [_]usize{ 1, 2, 3, 5, 8, 11, 12, 16, 24, 33, 64 };

    inline for ([_]usize{ 2, 4, 8 }) |L| {
        const K = Hamming(L);
        for (counts) |n| {
            const a = try std.testing.allocator.alloc(u64, n);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(u64, n);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = rnd.int(u64);
                y.* = rnd.int(u64);
            }
            try std.testing.expectEqual(hammingReference(a, b), K.call(a, b));
        }
    }
}

test "hamming of a vector with itself is zero" {
    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    var v: [24]u64 = undefined;
    for (&v) |*x| x.* = rnd.int(u64);
    try std.testing.expectEqual(@as(u32, 0), native.call(&v, &v));
}

test "hamming of a vector with its complement is every bit" {
    var v: [12]u64 = undefined;
    var c: [12]u64 = undefined;
    for (&v, &c, 0..) |*x, *y, i| {
        x.* = @as(u64, i) *% 0x9e3779b97f4a7c15;
        y.* = ~x.*;
    }
    try std.testing.expectEqual(@as(u32, 12 * 64), native.call(&v, &c));
}

test "wordsFor rounds up and packSigns zeroes the padding" {
    try std.testing.expectEqual(@as(usize, 1), wordsFor(1));
    try std.testing.expectEqual(@as(usize, 1), wordsFor(64));
    try std.testing.expectEqual(@as(usize, 2), wordsFor(65));
    try std.testing.expectEqual(@as(usize, 12), wordsFor(768));
    try std.testing.expectEqual(@as(usize, 24), wordsFor(1536));

    // 100 dims: word 1 holds 36 real bits and 28 padding bits, all of which
    // must be zero regardless of the input.
    const d = 100;
    var v: [d]f32 = undefined;
    @memset(&v, 1.0); // every real bit set
    var packed_bits: [2]u64 = undefined;
    packSigns(&packed_bits, &v);
    try std.testing.expectEqual(~@as(u64, 0), packed_bits[0]);
    try std.testing.expectEqual((@as(u64, 1) << 36) - 1, packed_bits[1]);
}

test "packSigns: strictly positive sets the bit, zero and negative clear it" {
    const v = [_]f32{ 1.0, -1.0, 0.0, -0.0, 0.5, -0.5, 1e-30, -1e-30 };
    var packed_bits: [1]u64 = undefined;
    packSigns(&packed_bits, &v);
    // bits 0, 4, 6 set; 1, 2, 3, 5, 7 clear.
    try std.testing.expectEqual(@as(u64, 0b0101_0001), packed_bits[0]);
}

test "signDotFromHamming inverts the sign dot product" {
    const d = 768;
    // Identical sign vectors: hamming 0, dot = d.
    try std.testing.expectEqual(@as(f32, 768.0), signDotFromHamming(d, 0));
    // Complementary: hamming d, dot = -d.
    try std.testing.expectEqual(@as(f32, -768.0), signDotFromHamming(d, d));
    // Orthogonal-ish: half the bits differ, dot = 0.
    try std.testing.expectEqual(@as(f32, 0.0), signDotFromHamming(d, d / 2));
}

test "packSigns then signDotFromHamming reproduces the sign dot product" {
    var prng = std.Random.DefaultPrng.init(0x51a5);
    const rnd = prng.random();
    const d = 768;
    var a: [d]f32 = undefined;
    var b: [d]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }
    var pa: [12]u64 = undefined;
    var pb: [12]u64 = undefined;
    packSigns(&pa, &a);
    packSigns(&pb, &b);

    var expect: f32 = 0;
    for (a, b) |x, y| {
        const sx: f32 = if (x > 0) 1 else -1;
        const sy: f32 = if (y > 0) 1 else -1;
        expect += sx * sy;
    }
    try std.testing.expectEqual(expect, signDotFromHamming(d, native.call(&pa, &pb)));
}

test "similarity is negated so higher is better" {
    var a = [_]u64{0} ** 4;
    var b = [_]u64{0} ** 4;
    b[0] = 0xff; // 8 bits differ
    try std.testing.expectEqual(@as(f32, -8.0), native.similarity(&a, &b));
    try std.testing.expectEqual(@as(f32, 0.0), native.similarity(&a, &a));
    try std.testing.expect(native.similarity(&a, &a) > native.similarity(&a, &b));
}
