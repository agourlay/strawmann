//! `pq_adc`, product-quantization asymmetric distance computation.
//!
//! §6.7: "`product-x{4,8,16,32,64}` | PQ, 8 bit/subquantizer | LUT ADC, scalar
//! gather baseline | k-means codebooks trained on a sample" and "PQ4 /
//! FastScan | 4 bit, `vpshufb`-based LUT | in-register table lookup | stretch
//! goal; the version that actually competes".
//!
//! ADC ("asymmetric distance computation") keeps the query in full precision
//! and precomputes, once per query, a table of partial distances from the query
//! subvector to every centroid of every subquantizer. Scoring a database vector
//! is then `m` table lookups and `m` adds, no arithmetic on the vector itself.
//!
//! §6.6.1 puts PQ in the "partially L3-resident, LUT lookup throughput is the
//! kernel" regime: at 192 MB for 1M×768 at x16, the codes fit in a large L3,
//! so the memory system stops hiding the kernel and lookup throughput becomes
//! the limiter. That is why the 4-bit variant matters so much more than its
//! 2× size advantage suggests, it changes the lookup from a memory access
//! into a register operation.
//!
//! ## The two implementations, and why both exist
//!
//! **PQ8 (`Pq8Adc`)**, 8-bit codes, 256-entry f32 tables. One scalar load per
//! subquantizer. This is the "scalar gather baseline" §6.7 asks for. It is
//! simple, exact, and bounded by L1 load throughput at ~1 lookup/cycle.
//!
//! **PQ4 FastScan (`Pq4FastScan`)**, 4-bit codes, 16-entry tables quantized to
//! u8 so a whole table fits in one `vpshufb` operand. `vpshufb` performs 16
//! lookups per 128-bit lane per instruction and never crosses lanes, so one
//! AVX-512 instruction does 64 lookups: four subquantizers, each in its own
//! lane with its own table, against the same 16 vectors. §6.6.2 rates the gap
//! at "~2×, more with VBMI".
//!
//! FastScan needs codes stored **interleaved by block**: for a block of 32
//! database vectors, subquantizer `m`'s codes for all 32 vectors are contiguous
//! (32 nibbles = 16 bytes). That layout is not a detail, it is what makes the
//! lookup a register operation instead of a gather, and it is why PQ4 storage
//! is a different file format rather than a narrower PQ8.

const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");

/// Is a 3-operand `vpshufb` available at `L` bytes?
///
/// Same reasoning as the `vpdpbusd` gate in `dot_i8.zig`: Zig has no portable
/// builtin for a *runtime*-indexed vector shuffle (`@shuffle` requires a
/// comptime mask), so the in-register table lookup that makes FastScan worth
/// having has to be written as one instruction of inline assembly. Everything
/// else falls back to the scalar loop, which is correct but is not FastScan.
///
/// VEX encoding (AVX) is required for the non-destructive 3-operand form; with
/// only SSSE3 the instruction overwrites its table operand, which would force a
/// reload of the LUT on every subquantizer. The 256-bit form is AVX2 and the
/// 512-bit form is AVX512BW (`vpshufb zmm` is a BW instruction, not an F one).
fn hasVpshufb(comptime L: usize) bool {
    if (builtin.zig_backend != .stage2_llvm) return false;
    return switch (L) {
        16 => builtin.cpu.has(.x86, .avx),
        32 => builtin.cpu.has(.x86, .avx2),
        64 => builtin.cpu.has(.x86, .avx512bw),
        else => false,
    };
}

/// `L / 16` independent 16-entry table lookups, one per 128-bit lane:
/// `out[j] = tbl[(j & ~15) + (idx[j] & 0x0f)]`.
///
/// This is exactly `vpshufb`'s semantics at every width: the instruction never
/// crosses a 128-bit lane, so a 32- or 64-byte shuffle is two or four
/// side-by-side 16-byte shuffles, each with its own table. That is what makes
/// the wide form useful for FastScan, each lane can hold a *different*
/// subquantizer's table and score it against that subquantizer's codes, so one
/// AVX-512 instruction does 64 lookups over four subquantizers.
///
/// Note `vpshufb` zeroes the output byte when the index has bit 7 set. The
/// callers here always mask to 4 bits first, so that behaviour is unreachable -
/// but the scalar fallback masks explicitly so the two paths agree on inputs
/// that would otherwise diverge.
inline fn shuffleBytes(comptime L: usize, tbl: @Vector(L, u8), idx: @Vector(L, u8)) @Vector(L, u8) {
    comptime std.debug.assert(L == 16 or L == 32 or L == 64);
    if (comptime hasVpshufb(L)) {
        // "v" rather than "x": with AVX-512 enabled it admits xmm/ymm/zmm16-31
        // as well, and without it LLVM treats it as "x".
        return asm ("vpshufb %[idx], %[tbl], %[out]"
            : [out] "=v" (-> @Vector(L, u8)),
            : [tbl] "v" (tbl),
              [idx] "v" (idx),
        );
    }
    const t: [L]u8 = tbl;
    const i: [L]u8 = idx;
    var o: [L]u8 = undefined;
    inline for (0..L) |j| o[j] = t[(j & ~@as(usize, 15)) + (i[j] & 0x0f)];
    return o;
}

/// The 16-byte form, kept under its old name for the tail and for readers.
inline fn shuffle16(tbl: @Vector(16, u8), idx: @Vector(16, u8)) @Vector(16, u8) {
    return shuffleBytes(16, tbl, idx);
}

// -------------------------------------------------------------------------
// PQ8, the scalar-LUT baseline.
// -------------------------------------------------------------------------

/// 8-bit product quantization ADC.
///
/// `m` is the number of subquantizers, fixed per collection. The lookup table
/// is `m × 256` f32, built once per query.
pub const Pq8Adc = struct {
    pub const centroids = 256;
    pub const bits = 8;

    /// Score one database vector: `Σ_m table[m][code[m]]`.
    ///
    /// `code` is `m` bytes and `table` is `m × C` row-major, where `C` is
    /// **derived from the two lengths** rather than fixed at 256. PQ8 uses
    /// C=256 and PQ4 uses C=16, but a codebook trained with fewer centroids is
    /// a legitimate configuration (and is what keeps tests fast); hardcoding
    /// 256 would turn that into an assertion failure at the call site rather
    /// than a supported case.
    ///
    /// Deliberately left as a plain loop. The dependency is a load whose
    /// address depends on the code byte, which no amount of vectorisation
    /// removes, `vgatherdps` would express it in one instruction but §6.6.2 is
    /// explicit: "**avoid both.** Gather has historically been slower than
    /// scalar loads plus inserts for our access patterns; benchmark it once,
    /// record the result, move on."
    pub fn score(table: []const f32, code: []const u8) f32 {
        std.debug.assert(code.len > 0);
        std.debug.assert(table.len % code.len == 0);
        const c = table.len / code.len;
        // Four accumulators: the adds are a dependent chain otherwise, and at
        // m=96 (d=768, x8) that chain is the whole kernel.
        var a0: f32 = 0;
        var a1: f32 = 0;
        var a2: f32 = 0;
        var a3: f32 = 0;
        var m: usize = 0;
        while (m + 4 <= code.len) : (m += 4) {
            a0 += table[(m + 0) * c + code[m + 0]];
            a1 += table[(m + 1) * c + code[m + 1]];
            a2 += table[(m + 2) * c + code[m + 2]];
            a3 += table[(m + 3) * c + code[m + 3]];
        }
        var s = (a0 + a1) + (a2 + a3);
        while (m < code.len) : (m += 1) s += table[m * c + code[m]];
        return s;
    }

    /// Build the per-query lookup table for a dot-product metric.
    ///
    /// `table[m][c] = dot(query_subvector_m, centroid[m][c])`, so that summing
    /// over `m` reconstructs the full dot product against the quantized vector.
    /// `codebook` is `m × 256 × sub_dim` f32.
    pub fn buildTableDot(table: []f32, query: []const f32, codebook: []const f32, m_count: usize) void {
        const sub_dim = query.len / m_count;
        std.debug.assert(query.len % m_count == 0);
        std.debug.assert(table.len % m_count == 0);
        const nc = table.len / m_count;
        std.debug.assert(codebook.len == m_count * nc * sub_dim);

        for (0..m_count) |m| {
            const qsub = query[m * sub_dim ..][0..sub_dim];
            for (0..nc) |c| {
                const cent = codebook[(m * nc + c) * sub_dim ..][0..sub_dim];
                table[m * nc + c] = dotSmall(qsub, cent);
            }
        }
    }

    /// Build the per-query table for Euclid: `table[m][c] = −Σ(q−centroid)²`.
    ///
    /// Negated, so summing over `m` yields the internal similarity directly and
    /// the caller never has to know which metric produced the table. §8.3's
    /// "higher is better" convention holds through the PQ path unchanged.
    pub fn buildTableEuclid(table: []f32, query: []const f32, codebook: []const f32, m_count: usize) void {
        const sub_dim = query.len / m_count;
        std.debug.assert(query.len % m_count == 0);
        std.debug.assert(table.len % m_count == 0);
        const nc = table.len / m_count;

        for (0..m_count) |m| {
            const qsub = query[m * sub_dim ..][0..sub_dim];
            for (0..nc) |c| {
                const cent = codebook[(m * nc + c) * sub_dim ..][0..sub_dim];
                var acc: f32 = 0;
                for (qsub, cent) |q, x| {
                    const d = q - x;
                    acc += d * d;
                }
                table[m * nc + c] = -acc;
            }
        }
    }
};

/// A small dot product for subvectors, which are typically 4–16 elements.
/// Too short for the blocked kernel in `dot_f32.zig` to pay for itself.
fn dotSmall(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.optimized);
    var s: f32 = 0;
    for (a, b) |x, y| s = common.mulAdd(f32, x, y, s);
    return s;
}

// -------------------------------------------------------------------------
// PQ4 FastScan, the vpshufb in-register LUT.
// -------------------------------------------------------------------------

/// Number of database vectors scored per FastScan block.
///
/// 32 is the standard choice: 32 four-bit codes per subquantizer is 16 bytes,
/// exactly one `vpshufb` lane, so the interleaved layout lines up with the
/// instruction on every width from SSE to AVX-512.
pub const fastscan_block = 32;

/// Largest subquantizer count `scoreBlock` accepts: `⌊65535 / 255⌋`.
///
/// The block kernel accumulates in u16 and every table entry is at most 255,
/// so `m × 255` has to fit in 16 bits or the sum wraps silently and a far
/// vector scores as a near one. 257 is far beyond any x-factor §6.7 names
/// (x4 at d=768 is m=192), so the bound costs nothing in practice, but it is
/// asserted rather than assumed because a wrap here is undetectable
/// downstream: the score is a plausible small number.
pub const max_subquantizers: usize = std.math.maxInt(u16) / 255;

/// 4-bit product quantization with `vpshufb` table lookup.
///
/// `L` is the byte width of the vector registers: 16 = SSE, 32 = AVX2,
/// 64 = AVX-512. A block is always 32 vectors, so a subquantizer's codes are
/// always 16 bytes; a wider register does not score more vectors per
/// instruction, it scores more **subquantizers** per instruction: `L / 16` of
/// them, each in its own 128-bit lane with its own table, because `vpshufb`
/// never crosses lanes. The per-lane partial sums are folded together once at
/// the end of the block.
pub fn Pq4FastScan(comptime L: usize) type {
    comptime std.debug.assert(L == 16 or L == 32 or L == 64);

    return struct {
        pub const lanes = L;
        pub const centroids = 16;
        pub const bits = 4;
        pub const block = fastscan_block;
        /// Subquantizers scored per instruction.
        pub const subquantizers_per_op = L / 16;

        const Bytes = @Vector(L, u8);
        const Wide = @Vector(L, u16);
        const lanes_128 = subquantizers_per_op;

        /// Score a block of 32 database vectors against a quantized table.
        ///
        /// `table` is `m × 16` u8, the f32 distances quantized to a byte range
        /// by `quantizeTable`. `codes` is the interleaved block: `m × 16` bytes,
        /// where byte `j` of subquantizer `m` packs the codes of vectors `j`
        /// (low nibble) and `j + 16` (high nibble).
        ///
        /// Both `table` and `codes` are laid out subquantizer-major with 16
        /// bytes per subquantizer, so `lanes_128` consecutive subquantizers'
        /// tables are one contiguous `L`-byte load, and so are their codes.
        /// That is the whole trick: no broadcast, no permute, the layout *is*
        /// the register image.
        ///
        /// Accumulates in u16 to avoid the saturation that a u8 accumulator
        /// would hit past ~8 subquantizers. FAISS accumulates in u8 with
        /// periodic widening, which is faster and much fiddlier; this version
        /// trades some throughput for a kernel whose correctness is obvious.
        pub fn scoreBlock(out: *[block]u16, table: []const u8, codes: []const u8) void {
            const m_count = table.len / centroids;
            std.debug.assert(table.len == m_count * centroids);
            std.debug.assert(codes.len == m_count * (block / 2));
            // See `max_subquantizers`: past it the u16 accumulators wrap.
            std.debug.assert(m_count <= max_subquantizers);

            // One u16 lane per (subquantizer slot, vector): slot `g` of the
            // register accumulates subquantizers g, g + lanes_128, ... .
            var acc_lo: Wide = @splat(0);
            var acc_hi: Wide = @splat(0);

            const lo_mask: Bytes = @splat(0x0f);
            const four: Bytes = @splat(4);

            var m: usize = 0;
            const m_wide = m_count - (m_count % lanes_128);
            while (m < m_wide) : (m += lanes_128) {
                // `lanes_128` tables and `lanes_128` code chunks, one straight
                // load each.
                const lut: Bytes = table[m * centroids ..][0..L].*;
                const packed_codes: Bytes = codes[m * (block / 2) ..][0..L].*;

                // Low nibbles are vectors 0..15, high nibbles are 16..31.
                const lo_idx = packed_codes & lo_mask;
                const hi_idx = packed_codes >> four;

                // The lookup itself: one `vpshufb` per nibble half, `L`
                // table lookups each. This is the instruction the whole 4-bit
                // design exists to reach.
                const lo_vals = shuffleBytes(L, lut, lo_idx);
                const hi_vals = shuffleBytes(L, lut, hi_idx);

                acc_lo += @as(Wide, lo_vals);
                acc_hi += @as(Wide, hi_vals);
            }

            // Fold the per-slot partials into slot 0. Fixed order, so the
            // result is the same as the 16-byte kernel's bit for bit (which for
            // u16 sums means: the same, period, but the shape is still fixed
            // so an overflow would reproduce).
            var lo16: @Vector(16, u16) = @shuffle(u16, acc_lo, undefined, std.simd.iota(i32, 16));
            var hi16: @Vector(16, u16) = @shuffle(u16, acc_hi, undefined, std.simd.iota(i32, 16));
            inline for (1..lanes_128) |g| {
                const sel = comptime std.simd.iota(i32, 16) + @as(@Vector(16, i32), @splat(g * 16));
                lo16 += @shuffle(u16, acc_lo, undefined, sel);
                hi16 += @shuffle(u16, acc_hi, undefined, sel);
            }

            // Tail: subquantizers that did not fill a whole register, one at
            // a time on the 16-byte form. Empty when `m_count % lanes_128 == 0`,
            // which it is for every x-factor §6.7 names at d=768.
            while (m < m_count) : (m += 1) {
                const lut: @Vector(16, u8) = table[m * centroids ..][0..centroids].*;
                const packed_codes: @Vector(16, u8) = codes[m * (block / 2) ..][0..16].*;
                const lo_idx = packed_codes & @as(@Vector(16, u8), @splat(0x0f));
                const hi_idx = packed_codes >> @as(@Vector(16, u8), @splat(4));
                lo16 += @as(@Vector(16, u16), shuffle16(lut, lo_idx));
                hi16 += @as(@Vector(16, u16), shuffle16(lut, hi_idx));
            }

            inline for (0..16) |j| {
                out[j] = lo16[j];
                out[j + 16] = hi16[j];
            }
        }

        /// Scalar reference for `scoreBlock`, used to validate it.
        pub fn scoreBlockReference(out: *[block]u16, table: []const u8, codes: []const u8) void {
            const m_count = table.len / centroids;
            std.debug.assert(m_count <= max_subquantizers);
            @memset(out, 0);
            for (0..m_count) |m| {
                for (0..16) |j| {
                    const byte = codes[m * (block / 2) + j];
                    out[j] += table[m * centroids + (byte & 0x0f)];
                    out[j + 16] += table[m * centroids + (byte >> 4)];
                }
            }
        }
    };
}

pub const pq4_native = Pq4FastScan(@max(16, common.nativeLanes(u8)));

/// Quantize an f32 ADC table into the u8 range FastScan needs.
///
/// Returns the affine parameters so a score can be mapped back:
/// `f32_score ≈ offset + scale × u16_score`.
///
/// §8.5 T4 is where this matters: "the distribution of `|quantized_score −
/// fp32_score|`". The table quantization is a *second* lossy stage on top of
/// the code quantization, and conflating the two would misattribute the error.
pub const TableQuant = struct {
    offset: f32,
    scale: f32,

    /// Total quantization error is bounded by `m × scale / 2`, the per-entry
    /// rounding error, summed over subquantizers. Reported so T4 can separate
    /// "the codes lost information" from "the table lost information".
    pub fn maxError(self: TableQuant, m_count: usize) f32 {
        return @as(f32, @floatFromInt(m_count)) * self.scale * 0.5;
    }
};

pub fn quantizeTable(dst: []u8, src: []const f32, m_count: usize) TableQuant {
    std.debug.assert(dst.len == src.len);
    const m_entries = src.len / m_count;

    // Per-subquantizer minimum, so the offset absorbs the per-table bias and
    // the shared scale only has to cover the residual spread. A single global
    // minimum would waste most of the u8 range on tables that happen to sit
    // far from zero.
    var total_min: f32 = 0;
    var max_spread: f32 = 0;
    for (0..m_count) |m| {
        const row = src[m * m_entries ..][0..m_entries];
        var lo = row[0];
        var hi = row[0];
        for (row) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        total_min += lo;
        max_spread = @max(max_spread, hi - lo);
    }

    // 255 rather than 256: the code must be representable in a u8.
    const scale = if (max_spread > 0) max_spread / 255.0 else 1.0;
    for (0..m_count) |m| {
        const row = src[m * m_entries ..][0..m_entries];
        const drow = dst[m * m_entries ..][0..m_entries];
        var lo = row[0];
        for (row) |v| lo = @min(lo, v);
        for (row, drow) |v, *d| {
            const q = @round((v - lo) / scale);
            d.* = @intFromFloat(std.math.clamp(q, 0.0, 255.0));
        }
    }
    return .{ .offset = total_min, .scale = scale };
}

/// Pack 32 vectors' 4-bit codes into the FastScan interleaved layout.
///
/// `codes` is `32 × m` row-major (one row per vector, as produced by the
/// quantizer). `dst` is `m × 16` interleaved.
pub fn packFastScanBlock(dst: []u8, codes: []const u8, m_count: usize) void {
    std.debug.assert(codes.len == fastscan_block * m_count);
    std.debug.assert(dst.len == m_count * (fastscan_block / 2));
    for (0..m_count) |m| {
        for (0..16) |j| {
            const lo = codes[j * m_count + m] & 0x0f;
            const hi = codes[(j + 16) * m_count + m] & 0x0f;
            dst[m * 16 + j] = lo | (hi << 4);
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "PQ8 ADC score matches an explicit sum over subquantizers" {
    var prng = std.Random.DefaultPrng.init(0x9c8);
    const rnd = prng.random();
    const m_count = 96; // d=768 at x8
    const table = try std.testing.allocator.alloc(f32, m_count * 256);
    defer std.testing.allocator.free(table);
    const code = try std.testing.allocator.alloc(u8, m_count);
    defer std.testing.allocator.free(code);
    for (table) |*t| t.* = rnd.floatNorm(f32);
    for (code) |*c| c.* = rnd.int(u8);

    var want: f64 = 0;
    for (0..m_count) |m| want += table[m * 256 + code[m]];

    const got = Pq8Adc.score(table, code);
    try std.testing.expect(@abs(@as(f64, got) - want) < 1e-4);
}

test "PQ8 dot table reconstructs the dot product against the quantized vector" {
    // The defining property of ADC: summing the table entries for a vector's
    // codes must equal the dot product against that vector's reconstruction.
    var prng = std.Random.DefaultPrng.init(0x5150);
    const rnd = prng.random();
    const d = 64;
    const m_count = 8;
    const sub_dim = d / m_count;

    const codebook = try std.testing.allocator.alloc(f32, m_count * 256 * sub_dim);
    defer std.testing.allocator.free(codebook);
    for (codebook) |*x| x.* = rnd.floatNorm(f32);

    var query: [d]f32 = undefined;
    for (&query) |*x| x.* = rnd.floatNorm(f32);

    const table = try std.testing.allocator.alloc(f32, m_count * 256);
    defer std.testing.allocator.free(table);
    Pq8Adc.buildTableDot(table, &query, codebook, m_count);

    var code: [m_count]u8 = undefined;
    for (&code) |*c| c.* = rnd.int(u8);

    // Reconstruct the vector the codes name, then take a plain dot product.
    var recon: [d]f32 = undefined;
    for (0..m_count) |m| {
        const cent = codebook[(m * 256 + code[m]) * sub_dim ..][0..sub_dim];
        @memcpy(recon[m * sub_dim ..][0..sub_dim], cent);
    }
    const direct = @import("reference.zig").dotWide(&query, &recon);
    const via_adc = Pq8Adc.score(table, &code);

    try std.testing.expect(@abs(@as(f64, via_adc) - direct) < 1e-4);
}

test "PQ8 euclid table reconstructs the negated squared distance" {
    var prng = std.Random.DefaultPrng.init(0x5151);
    const rnd = prng.random();
    const d = 64;
    const m_count = 16;
    const sub_dim = d / m_count;

    const codebook = try std.testing.allocator.alloc(f32, m_count * 256 * sub_dim);
    defer std.testing.allocator.free(codebook);
    for (codebook) |*x| x.* = rnd.floatNorm(f32);

    var query: [d]f32 = undefined;
    for (&query) |*x| x.* = rnd.floatNorm(f32);

    const table = try std.testing.allocator.alloc(f32, m_count * 256);
    defer std.testing.allocator.free(table);
    Pq8Adc.buildTableEuclid(table, &query, codebook, m_count);

    var code: [m_count]u8 = undefined;
    for (&code) |*c| c.* = rnd.int(u8);

    var recon: [d]f32 = undefined;
    for (0..m_count) |m| {
        const cent = codebook[(m * 256 + code[m]) * sub_dim ..][0..sub_dim];
        @memcpy(recon[m * sub_dim ..][0..sub_dim], cent);
    }
    const direct = @import("reference.zig").euclidWide(&query, &recon);
    const via_adc = Pq8Adc.score(table, &code);
    try std.testing.expect(@abs(@as(f64, via_adc) - direct) < 1e-4);

    // And the sign convention: higher must still be better.
    try std.testing.expect(via_adc <= 0.0);
}

test "PQ4 FastScan block matches its scalar reference at every register width" {
    var prng = std.Random.DefaultPrng.init(0xfa57);
    const rnd = prng.random();

    // Every residue of m_count mod 4 so the wide kernels' tail loop runs with
    // 0, 1, 2 and 3 leftover subquantizers, plus the §6.7 x-factors at d=768.
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 48, 96, 97, 191, 192 }) |m_count| {
        const table = try std.testing.allocator.alloc(u8, m_count * 16);
        defer std.testing.allocator.free(table);
        const codes = try std.testing.allocator.alloc(u8, m_count * (fastscan_block / 2));
        defer std.testing.allocator.free(codes);
        for (table) |*t| t.* = rnd.int(u8);
        for (codes) |*c| c.* = rnd.int(u8);

        var want: [fastscan_block]u16 = undefined;
        pq4_native.scoreBlockReference(&want, table, codes);

        // The three widths are compiled unconditionally; on a host without
        // the matching `vpshufb` the portable per-lane shuffle runs instead,
        // which is the point: the *result* must not depend on the width.
        inline for (.{ 16, 32, 64 }) |L| {
            var got: [fastscan_block]u16 = undefined;
            Pq4FastScan(L).scoreBlock(&got, table, codes);
            try std.testing.expectEqualSlices(u16, &want, &got);
        }
        var got: [fastscan_block]u16 = undefined;
        pq4_native.scoreBlock(&got, table, codes);
        try std.testing.expectEqualSlices(u16, &want, &got);
    }
}

test "PQ4 FastScan reports the subquantizers each width scores per instruction" {
    try std.testing.expectEqual(@as(usize, 1), Pq4FastScan(16).subquantizers_per_op);
    try std.testing.expectEqual(@as(usize, 2), Pq4FastScan(32).subquantizers_per_op);
    try std.testing.expectEqual(@as(usize, 4), Pq4FastScan(64).subquantizers_per_op);
    try std.testing.expectEqual(pq4_native.lanes / 16, pq4_native.subquantizers_per_op);
}

test "shuffleBytes never crosses a 128-bit lane, at any width" {
    // Table lane g holds bytes g*16 .. g*16+15 = 0x10*g + k; every index in
    // lane g must pick from that lane only, so the result's high nibble names
    // its own lane.
    inline for (.{ 16, 32, 64 }) |L| {
        var t: [L]u8 = undefined;
        var i: [L]u8 = undefined;
        for (0..L) |j| {
            t[j] = @intCast(((j / 16) << 4) | (j % 16));
            i[j] = @intCast((j * 7 + 3) % 16);
        }
        const o: [L]u8 = shuffleBytes(L, t, i);
        for (0..L) |j| {
            try std.testing.expectEqual(@as(u8, @intCast(((j / 16) << 4) | ((j * 7 + 3) % 16))), o[j]);
        }
    }
}

test "PQ4 FastScan u16 accumulator does not overflow at the maximum" {
    // 96 subquantizers × 255 = 24480, well inside u16. The u8 accumulator FAISS
    // uses would have wrapped at m=2, which is why this version widens.
    const m_count = 96;
    const table = try std.testing.allocator.alloc(u8, m_count * 16);
    defer std.testing.allocator.free(table);
    const codes = try std.testing.allocator.alloc(u8, m_count * (fastscan_block / 2));
    defer std.testing.allocator.free(codes);
    @memset(table, 255);
    @memset(codes, 0);

    var got: [fastscan_block]u16 = undefined;
    pq4_native.scoreBlock(&got, table, codes);
    for (got) |v| try std.testing.expectEqual(@as(u16, m_count * 255), v);
}

test "PQ4 FastScan max_subquantizers is the exact u16 bound, and holds at every width" {
    // The constant must be tight: one more subquantizer of all-255 entries
    // would exceed u16, one fewer would leave headroom the doc does not claim.
    try std.testing.expect(max_subquantizers * 255 <= std.math.maxInt(u16));
    try std.testing.expect((max_subquantizers + 1) * 255 > std.math.maxInt(u16));

    // At the bound itself, the worst-case table must not wrap on any register
    // width, including the tail path (257 is not a multiple of 2 or 4).
    const m_count = max_subquantizers;
    const table = try std.testing.allocator.alloc(u8, m_count * 16);
    defer std.testing.allocator.free(table);
    const codes = try std.testing.allocator.alloc(u8, m_count * (fastscan_block / 2));
    defer std.testing.allocator.free(codes);
    @memset(table, 255);
    @memset(codes, 0xff);

    inline for ([_]usize{ 16, 32, 64 }) |L| {
        var got: [fastscan_block]u16 = undefined;
        Pq4FastScan(L).scoreBlock(&got, table, codes);
        for (got) |v| try std.testing.expectEqual(@as(u16, @intCast(m_count * 255)), v);
    }
}

test "packFastScanBlock round-trips through the reference scorer" {
    // Build per-vector codes, pack them, and confirm each vector's score equals
    // the sum of its own table entries. Catches interleaving errors, which are
    // otherwise silent and produce plausible-looking wrong answers.
    var prng = std.Random.DefaultPrng.init(0x9a55);
    const rnd = prng.random();
    const m_count = 8;

    var codes: [fastscan_block * m_count]u8 = undefined;
    for (&codes) |*c| c.* = rnd.uintLessThan(u8, 16);
    var table: [m_count * 16]u8 = undefined;
    for (&table) |*t| t.* = rnd.int(u8);

    var packed_codes: [m_count * (fastscan_block / 2)]u8 = undefined;
    packFastScanBlock(&packed_codes, &codes, m_count);

    var got: [fastscan_block]u16 = undefined;
    pq4_native.scoreBlock(&got, &table, &packed_codes);

    for (0..fastscan_block) |v| {
        var want: u16 = 0;
        for (0..m_count) |m| want += table[m * 16 + codes[v * m_count + m]];
        try std.testing.expectEqual(want, got[v]);
    }
}

test "quantizeTable error bound holds and is reported honestly" {
    var prng = std.Random.DefaultPrng.init(0x7ab1e);
    const rnd = prng.random();
    const m_count = 16;
    const entries = 16;

    var src: [m_count * entries]f32 = undefined;
    for (&src) |*x| x.* = rnd.floatNorm(f32) * 5.0;
    var dst: [m_count * entries]u8 = undefined;
    const tq = quantizeTable(&dst, &src, m_count);

    // Reconstruct and check every entry is within half a quantization step.
    for (0..m_count) |m| {
        const row = src[m * entries ..][0..entries];
        var lo = row[0];
        for (row) |v| lo = @min(lo, v);
        for (0..entries) |c| {
            const recon = lo + tq.scale * @as(f32, @floatFromInt(dst[m * entries + c]));
            try std.testing.expect(@abs(recon - src[m * entries + c]) <= tq.scale * 0.5 + 1e-6);
        }
    }
    try std.testing.expect(tq.maxError(m_count) > 0);
}
