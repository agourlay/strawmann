//! `dot_i8`, the scalar-quantization (SQ8) dot product.
//!
//! §6.6.2, the VNNI row:
//!
//! | capability | AVX2 path | AVX-512 path | expected gap |
//! |---|---|---|---|
//! | int8 dot (SQ) | `vpmaddubsw` → `vpmaddwd` → `vpaddd`, 3 instr + saturation care | `vpdpbusd` (AVX512_VNNI), 1 instr | ~3× instruction count |
//!
//! and the caveat that matters on this project's hardware: "AVX-VNNI gives
//! 256-bit `vpdpbusd` on Alder Lake+ and Zen 4+, which narrows this a lot -
//! **must be a separate row in the matrix, not folded into 'AVX2'**". The
//! `avx2-vnni` arm in `build.zig` exists for exactly that reason, and open
//! question 8 in §11 is what it answers: *is AVX-VNNI at 256 bits enough for
//! the SQ8 kernel, making full AVX-512 unnecessary for that path?*
//!
//! ## Why the primitive is u8 × i8 and not u8 × u8
//!
//! `vpdpbusd` multiplies **unsigned** bytes by **signed** bytes into a signed
//! i32 accumulator. It is not a symmetric instruction, and pretending otherwise
//! is the standard way to get a kernel that is correct on small values and
//! silently wrong above 127.
//!
//! So `dotU8I8`, unsigned data codes against signed query codes, is the
//! primitive that maps to one instruction, and the SQ8 layer in `quant/` is
//! built on top of it. `dotU8` (u8 × u8) is also provided because the
//! *symmetric* variant of §6.7 needs it, but it cannot use `vpdpbusd` directly
//! and its cost should be expected to sit closer to the AVX2 row even on a VNNI
//! part. Both are measured; §6.7 asks for "symmetric (query also quantized) and
//! asymmetric variants, both measured".
//!
//! ## Why this kernel has an explicit `vpdpbusd`, and the others do not
//!
//! Every other kernel in `dist/` is written as portable `@Vector` arithmetic
//! and left to the backend, because the backend does the right thing: `hamming`
//! genuinely lowers to `vpopcntq` on a VPOPCNTDQ target and to Muła's `vpshufb`
//! nibble-LUT otherwise, from one line of source.
//!
//! VNNI does **not** work that way, and the disassembly proves it. Written as
//! widening multiply-accumulate, each byte widened into its own i32 lane -
//! LLVM emits **zero** `vpdpbusd` on every arm including `avx512-vnni`, because
//! that shape is a different, wider computation than VNNI performs. `vpdpbusd`
//! horizontally sums *four* byte products into each dword lane; independent
//! per-lane widening never has that shape to match.
//!
//! So the VNNI path is written as one line of inline assembly, selected at
//! comptime from the target feature set, with the portable version retained as
//! the fallback and as the correctness reference. `docs/asm/` records which arm
//! got which, and a test asserts the two paths agree exactly (they are integer
//! kernels, so "agree" means bit-identical, not within a tolerance).
//!
//! This is the §11 risk "LLVM quietly ... fails to contract" caught by the
//! §6.6.3 disassembly discipline, on the first kernel it applied to. Without
//! `docs/asm/` the SQ8 arm of the matrix would have silently measured the AVX2
//! sequence on every VNNI build and answered §11's open question 8 backwards.

const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");

/// Is `vpdpbusd` available at a register width of `bytes`?
///
/// 512-bit needs AVX512_VNNI. 256- and 128-bit need either AVX-VNNI (the VEX
/// encoding, Alder Lake+ / Zen 4+) or AVX512_VNNI together with AVX512VL (the
/// EVEX encoding at reduced width). §6.6.2 insists the AVX-VNNI case be its own
/// matrix row rather than folded into "AVX2", and this predicate is what makes
/// the `avx2-vnni` arm actually differ from `avx2`.
fn hasVnniAt(comptime bytes: usize) bool {
    // The `v` register constraint is an LLVM constraint. Zig's self-hosted
    // x86_64 backend, the default for Debug builds in 0.16, rejects it, so
    // the asm path is only compiled under the LLVM backend and the portable
    // path covers everything else. `std.simd.suggestVectorLength` guards on the
    // same condition for the same reason.
    //
    // This costs nothing where it matters: §9 says results are only ever quoted
    // from `ReleaseFast`, which is an LLVM build. `build.zig` additionally
    // forces LLVM for the test step so the VNNI path is exercised by the
    // equality test below rather than silently skipped.
    if (builtin.zig_backend != .stage2_llvm) return false;

    const has_512 = builtin.cpu.has(.x86, .avx512vnni);
    const has_vl = builtin.cpu.has(.x86, .avx512vl);
    const has_vex = builtin.cpu.has(.x86, .avxvnni);
    return switch (bytes) {
        64 => has_512,
        32, 16 => has_vex or (has_512 and has_vl),
        else => false,
    };
}

/// `Σ a[i] · b[i]` over unsigned × signed bytes, accumulating in i32.
///
/// This is the `vpdpbusd` shape. `L` is the number of **byte lanes**: 16 = SSE,
/// 32 = AVX2, 64 = AVX-512.
pub fn DotU8I8(comptime L: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));

    return struct {
        pub const lanes = L;

        /// Largest `dim` for which no input can overflow the i32 total: the
        /// widest product is |255 × -128| = 32640, see `max_safe_dim`.
        pub const max_safe_dim: usize = std.math.maxInt(i32) / (255 * 128);

        /// True when this instantiation compiles to `vpdpbusd`. Reported by the
        /// benchmark harness so a result row states which sequence produced it
        /// rather than leaving the reader to infer it from the arm name.
        pub const uses_vnni = hasVnniAt(L);

        // The VNNI accumulator is i32 with L/4 lanes: `vpdpbusd` sums four byte
        // products into each dword lane.
        const Acc = @Vector(L / 4, i32);
        const Bytes = @Vector(L, u8);
        const SBytes = @Vector(L, i8);
        const Wide = @Vector(L, i32);

        const nacc = 4;
        const block = L * nacc;

        /// One `vpdpbusd`: `acc += Σ₄ (u8 × i8)` per dword lane.
        ///
        /// Not marked `volatile`, it is a pure function of its operands, and
        /// letting LLVM schedule and unroll around it is the entire point of
        /// keeping four independent accumulator chains.
        inline fn vpdpbusd(acc: Acc, a: Bytes, b: SBytes) Acc {
            return asm ("vpdpbusd %[b], %[a], %[acc]"
                : [acc] "=v" (-> Acc),
                : [_] "0" (acc),
                  [a] "v" (a),
                  [b] "v" (b),
            );
        }

        pub fn call(a: []const u8, b: []const i8) i32 {
            if (comptime uses_vnni) return callVnni(a, b);
            return callGeneric(a, b);
        }

        /// The `vpdpbusd` path. One instruction per L bytes.
        pub fn callVnni(a: []const u8, b: []const i8) i32 {
            comptime std.debug.assert(uses_vnni);
            std.debug.assert(a.len == b.len);

            var acc: [nacc]Acc = @splat(@as(Acc, @splat(0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..nacc) |k| {
                    const off = i + k * L;
                    const av: Bytes = a[off..][0..L].*;
                    const bv: SBytes = b[off..][0..L].*;
                    acc[k] = vpdpbusd(acc[k], av, bv);
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: Bytes = a[i..][0..L].*;
                const bv: SBytes = b[i..][0..L].*;
                acc[0] = vpdpbusd(acc[0], av, bv);
            }

            const folded = common.treeReduce(Acc, nacc, acc);
            var total: i32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                total += @as(i32, a[i]) * @as(i32, b[i]);
            }
            return total;
        }

        /// The portable path: widen each byte into its own i32 lane and
        /// multiply. Correct on every target and the reference the VNNI path is
        /// checked against, but, per the module doc, it never becomes
        /// `vpdpbusd`, so on an AVX2 target this is what the SQ8 row measures.
        ///
        /// Expressing the widening at i32 rather than going through i16 avoids
        /// the `vpmaddubsw` saturation hazard by construction: products reach
        /// 255×127 = 32385 and a pairwise i16 sum of two of them overflows.
        pub fn callGeneric(a: []const u8, b: []const i8) i32 {
            std.debug.assert(a.len == b.len);

            var acc: [nacc]Wide = @splat(@as(Wide, @splat(0)));
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..nacc) |k| {
                    const off = i + k * L;
                    const av: Bytes = a[off..][0..L].*;
                    const bv: SBytes = b[off..][0..L].*;
                    const aw: Wide = av;
                    const bw: Wide = bv;
                    acc[k] += aw * bw;
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: Bytes = a[i..][0..L].*;
                const bv: SBytes = b[i..][0..L].*;
                const aw: Wide = av;
                const bw: Wide = bv;
                acc[0] += aw * bw;
            }

            const folded = common.treeReduce(Wide, nacc, acc);
            var total: i32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                total += @as(i32, a[i]) * @as(i32, b[i]);
            }
            return total;
        }
    };
}

/// `Σ a[i] · b[i]` over unsigned × unsigned bytes, accumulating in u32.
///
/// The symmetric SQ8 variant of §6.7. Cannot use `vpdpbusd` (see the module
/// doc), so on a VNNI part this should be expected to cost roughly what the
/// AVX2 row costs, and confirming that is the point of measuring it.
pub fn DotU8(comptime L: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));

    return struct {
        pub const lanes = L;

        /// Largest `dim` for which no input can overflow the u32 total: the
        /// widest product is 255 × 255 = 65025, see `max_safe_dim`.
        pub const max_safe_dim: usize = std.math.maxInt(u32) / (255 * 255);

        const Bytes = @Vector(L, u8);
        const Wide = @Vector(L, u32);

        pub fn call(a: []const u8, b: []const u8) u32 {
            std.debug.assert(a.len == b.len);

            const nacc = 4;
            var acc: [nacc]Wide = @splat(@as(Wide, @splat(0)));
            const block = L * nacc;
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..nacc) |k| {
                    const off = i + k * L;
                    const av: Bytes = a[off..][0..L].*;
                    const bv: Bytes = b[off..][0..L].*;
                    const aw: Wide = av;
                    const bw: Wide = bv;
                    acc[k] += aw * bw;
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: Bytes = a[i..][0..L].*;
                const bv: Bytes = b[i..][0..L].*;
                const aw: Wide = av;
                const bw: Wide = bv;
                acc[0] += aw * bw;
            }

            const folded = common.treeReduce(Wide, nacc, acc);
            var total: u32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                total += @as(u32, a[i]) * @as(u32, b[i]);
            }
            return total;
        }
    };
}

/// `-Σ (a[i] - b[i])²` over two u8 vectors, accumulated in i32.
///
/// Integer throughout: the widest difference is ±255 and the widest square
/// 65025, so d=768 tops out around 5·10⁷ and cannot overflow i32 for any
/// dimension this engine accepts: `--max-dim` refuses anything above
/// `max_safe_dim` (33025 for this kernel, the tightest of the four).
/// Doing it in floats instead would cost a conversion per element for no
/// accuracy: every intermediate here is exactly representable.
pub fn EuclidU8(comptime L: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));

    return struct {
        pub const lanes = L;

        /// Largest `dim` for which no input can overflow the i32 total: the
        /// widest square is 255² = 65025, see `max_safe_dim`.
        pub const max_safe_dim: usize = std.math.maxInt(i32) / (255 * 255);

        const Bytes = @Vector(L, u8);
        const Wide = @Vector(L, i32);

        pub fn call(a: []const u8, b: []const u8) i32 {
            std.debug.assert(a.len == b.len);

            const nacc = 4;
            var acc: [nacc]Wide = @splat(@as(Wide, @splat(0)));
            const block = L * nacc;
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..nacc) |k| {
                    const off = i + k * L;
                    const av: Bytes = a[off..][0..L].*;
                    const bv: Bytes = b[off..][0..L].*;
                    // Widen *before* subtracting: u8 - u8 wraps, and the wrap
                    // is invisible in the result because squaring makes it
                    // positive again.
                    const d: Wide = @as(Wide, av) - @as(Wide, bv);
                    acc[k] += d * d;
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: Bytes = a[i..][0..L].*;
                const bv: Bytes = b[i..][0..L].*;
                const d: Wide = @as(Wide, av) - @as(Wide, bv);
                acc[0] += d * d;
            }

            const folded = common.treeReduce(Wide, nacc, acc);
            var total: i32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                const d = @as(i32, a[i]) - @as(i32, b[i]);
                total += d * d;
            }
            return -total;
        }
    };
}

/// `-Σ |a[i] - b[i]|` over two u8 vectors, accumulated in i32.
pub fn ManhattanU8(comptime L: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));

    return struct {
        pub const lanes = L;

        /// Largest `dim` for which no input can overflow the i32 total: the
        /// widest term is |0 - 255| = 255, see `max_safe_dim`.
        pub const max_safe_dim: usize = std.math.maxInt(i32) / 255;

        const Bytes = @Vector(L, u8);
        const Wide = @Vector(L, i32);

        pub fn call(a: []const u8, b: []const u8) i32 {
            std.debug.assert(a.len == b.len);

            const nacc = 4;
            var acc: [nacc]Wide = @splat(@as(Wide, @splat(0)));
            const block = L * nacc;
            var i: usize = 0;

            while (i + block <= a.len) : (i += block) {
                inline for (0..nacc) |k| {
                    const off = i + k * L;
                    const av: Bytes = a[off..][0..L].*;
                    const bv: Bytes = b[off..][0..L].*;
                    // `@abs` on a signed vector yields the unsigned vector of
                    // the same width; the difference of two u8 values fits in
                    // i32 with room to spare, so the cast back cannot wrap.
                    acc[k] += @as(Wide, @intCast(@abs(@as(Wide, av) - @as(Wide, bv))));
                }
            }
            while (i + L <= a.len) : (i += L) {
                const av: Bytes = a[i..][0..L].*;
                const bv: Bytes = b[i..][0..L].*;
                acc[0] += @as(Wide, @intCast(@abs(@as(Wide, av) - @as(Wide, bv))));
            }

            const folded = common.treeReduce(Wide, nacc, acc);
            var total: i32 = @reduce(.Add, folded);
            while (i < a.len) : (i += 1) {
                total += @intCast(@abs(@as(i32, a[i]) - @as(i32, b[i])));
            }
            return -total;
        }
    };
}

/// `Σ a[i] · b[i]` over an fp32 query and u8 data codes.
///
/// The *asymmetric* variant: the query stays in full precision and only the
/// stored vector is quantized. It cannot use any integer dot instruction, but
/// it avoids the query-quantization error entirely, which is often the better
/// accuracy/speed trade at low `ef`. §6.7 wants both measured.
pub fn DotF32U8(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const Bytes = @Vector(L, u8);
        const block = L * NACC;

        pub fn call(q: []const f32, c: []const u8) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(q.len == c.len);

            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= q.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const qv: V = q[off..][0..L].*;
                    const cb: Bytes = c[off..][0..L].*;
                    // u8 -> f32 conversion; on AVX-512 this is a single
                    // vcvtudq2ps after a zero-extend, on AVX2 a shuffle chain.
                    const cv: V = @floatFromInt(cb);
                    acc[k] = common.mulAdd(V, qv, cv, acc[k]);
                }
            }
            while (i + L <= q.len) : (i += L) {
                const qv: V = q[i..][0..L].*;
                const cb: Bytes = c[i..][0..L].*;
                const cv: V = @floatFromInt(cb);
                acc[0] = common.mulAdd(V, qv, cv, acc[0]);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < q.len) : (i += 1) s += q[i] * @as(f32, @floatFromInt(c[i]));
            return s;
        }
    };
}

/// How a u8 code reconstructs to a value: `x ≈ lo + α·c`.
///
/// The two numbers travel together through every asymmetric kernel, and they
/// were passed as two floats because `quant/scalar.zig` owns the trained
/// `Params` and this layer cannot import it (`quant` depends on `dist`, not
/// the other way round). A pair of bare `f32` arguments is not a layering
/// constraint though, it is a data clump: swap them at one call site and the
/// kernel computes a plausible wrong number. Naming the concept here lets
/// `Params.affine()` hand over one value that cannot be mis-ordered.
pub const Affine = struct {
    /// The value code 0 maps to.
    lo: f32,
    /// The quantization step.
    alpha: f32,
};

/// `-Σ (qᵢ - (lo + α·cᵢ))²` over an fp32 query and u8 codes.
///
/// The asymmetric SQ8 Euclid arm. It was a scalar loop while the `dot` arm
/// beside it used a vector kernel, which put a scalar inner loop on the
/// per-candidate path of W6 — and SIFT1M is a Euclid dataset, so that is the
/// row the benchmark actually measures.
///
/// The dequantization is folded into the kernel rather than materialising a
/// reconstructed vector: `lo + α·c` is one FMA against a widened code, and
/// writing the reconstruction to memory first would cost a store and a load
/// per candidate to save nothing.
pub fn EuclidF32U8(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const B = @Vector(L, u8);
        const block = L * NACC;

        pub fn call(q: []const f32, c: []const u8, a: Affine) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(q.len == c.len);

            const lov: V = @splat(a.lo);
            const av: V = @splat(a.alpha);
            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= q.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const qv: V = q[off..][0..L].*;
                    const cb: B = c[off..][0..L].*;
                    const cv: V = @floatFromInt(cb);
                    const d = qv - common.mulAdd(V, av, cv, lov);
                    acc[k] = common.mulAdd(V, d, d, acc[k]);
                }
            }
            while (i + L <= q.len) : (i += L) {
                const qv: V = q[i..][0..L].*;
                const cb: B = c[i..][0..L].*;
                const cv: V = @floatFromInt(cb);
                const d = qv - common.mulAdd(V, av, cv, lov);
                acc[0] = common.mulAdd(V, d, d, acc[0]);
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < q.len) : (i += 1) {
                const d = q[i] - (a.lo + a.alpha * @as(f32, @floatFromInt(c[i])));
                s += d * d;
            }
            return -s;
        }
    };
}

/// `-Σ |qᵢ - (lo + α·cᵢ)|` over an fp32 query and u8 codes.
pub fn ManhattanF32U8(comptime L: usize, comptime NACC: usize) type {
    comptime std.debug.assert(std.math.isPowerOfTwo(L));
    comptime std.debug.assert(std.math.isPowerOfTwo(NACC));

    return struct {
        pub const lanes = L;
        pub const accumulators = NACC;

        const V = @Vector(L, f32);
        const B = @Vector(L, u8);
        const block = L * NACC;

        pub fn call(q: []const f32, c: []const u8, a: Affine) f32 {
            @setFloatMode(.optimized);
            std.debug.assert(q.len == c.len);

            const lov: V = @splat(a.lo);
            const av: V = @splat(a.alpha);
            var acc: [NACC]V = @splat(@as(V, @splat(0.0)));
            var i: usize = 0;

            while (i + block <= q.len) : (i += block) {
                inline for (0..NACC) |k| {
                    const off = i + k * L;
                    const qv: V = q[off..][0..L].*;
                    const cb: B = c[off..][0..L].*;
                    const cv: V = @floatFromInt(cb);
                    acc[k] += @abs(qv - common.mulAdd(V, av, cv, lov));
                }
            }
            while (i + L <= q.len) : (i += L) {
                const qv: V = q[i..][0..L].*;
                const cb: B = c[i..][0..L].*;
                const cv: V = @floatFromInt(cb);
                acc[0] += @abs(qv - common.mulAdd(V, av, cv, lov));
            }

            const folded = common.treeReduce(V, NACC, acc);
            var s: f32 = @reduce(.Add, folded);
            while (i < q.len) : (i += 1) {
                s += @abs(q[i] - (a.lo + a.alpha * @as(f32, @floatFromInt(c[i]))));
            }
            return -s;
        }
    };
}

pub const euclid_f32u8_native = EuclidF32U8(common.nativeLanes(f32), common.default_accumulators);
pub const manhattan_f32u8_native = ManhattanF32U8(common.nativeLanes(f32), common.default_accumulators);

pub const u8i8_native = DotU8I8(common.nativeLanes(u8));

/// The largest `dim` every integer kernel in this file is overflow-free at,
/// for *any* input bytes. Each kernel accumulates in a 32-bit integer with no
/// saturation and no widening, so past its own `max_safe_dim` an adversarial
/// (or merely extreme) pair of vectors wraps silently and the score is
/// garbage with no error. `--max-dim` refuses anything above this constant so
/// the bound is enforced once, at the front door, rather than checked per
/// call on the hot path. The binding constraint is `EuclidU8`: 255² = 65025
/// per element against an i32 total.
pub const max_safe_dim: usize = @min(
    @min(DotU8I8(16).max_safe_dim, DotU8(16).max_safe_dim),
    @min(EuclidU8(16).max_safe_dim, ManhattanU8(16).max_safe_dim),
);
pub const u8u8_native = DotU8(common.nativeLanes(u8));
pub const euclid_u8_native = EuclidU8(common.nativeLanes(u8));
pub const manhattan_u8_native = ManhattanU8(common.nativeLanes(u8));
pub const f32u8_native = DotF32U8(common.nativeLanes(f32), common.default_accumulators);

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

fn refU8I8(a: []const u8, b: []const i8) i32 {
    var t: i32 = 0;
    for (a, b) |x, y| t += @as(i32, x) * @as(i32, y);
    return t;
}

fn refU8(a: []const u8, b: []const u8) u32 {
    var t: u32 = 0;
    for (a, b) |x, y| t += @as(u32, x) * @as(u32, y);
    return t;
}

test "u8xi8 dot matches the reference across widths and dims" {
    var prng = std.Random.DefaultPrng.init(0x18d07);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 7, 16, 17, 32, 33, 63, 64, 65, 128, 384, 768, 1536 };

    inline for ([_]usize{ 16, 32, 64 }) |L| {
        const K = DotU8I8(L);
        for (dims) |d| {
            const a = try std.testing.allocator.alloc(u8, d);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(i8, d);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = rnd.int(u8);
                y.* = rnd.int(i8);
            }
            try std.testing.expectEqual(refU8I8(a, b), K.call(a, b));
        }
    }
}

test "u8xi8 dot is correct at the saturation extremes" {
    // The hazard this pins: a lowering through `vpmaddubsw`, whose i16
    // intermediate saturates at 255×127 = 32385 per product and overflows when
    // two are summed pairwise (64770 > 32767). `callGeneric` deliberately
    // widens to i32 first (on AVX2 that is `vpmovzx`/`vpmovsx` + `vpmaddwd`,
    // see docs/asm/avx2/sm_dot_u8i8.asm), so it must get this case right, and
    // this test is what would catch a future lowering that does not.
    const d = 256;
    var a: [d]u8 = undefined;
    var b: [d]i8 = undefined;
    @memset(&a, 255);
    @memset(&b, 127);
    const want: i32 = @as(i32, d) * 255 * 127;
    inline for ([_]usize{ 16, 32, 64 }) |L| {
        try std.testing.expectEqual(want, DotU8I8(L).call(&a, &b));
    }

    // And the negative extreme.
    @memset(&b, -128);
    const want_neg: i32 = @as(i32, d) * 255 * -128;
    inline for ([_]usize{ 16, 32, 64 }) |L| {
        try std.testing.expectEqual(want_neg, DotU8I8(L).call(&a, &b));
    }
}

test "u8xu8 dot matches the reference across widths and dims" {
    var prng = std.Random.DefaultPrng.init(0x18d08);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 15, 16, 31, 64, 100, 128, 768, 1536 };

    inline for ([_]usize{ 16, 32, 64 }) |L| {
        const K = DotU8(L);
        for (dims) |d| {
            const a = try std.testing.allocator.alloc(u8, d);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(u8, d);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = rnd.int(u8);
                y.* = rnd.int(u8);
            }
            try std.testing.expectEqual(refU8(a, b), K.call(a, b));
        }
    }
}

test "the vpdpbusd path agrees with the portable path bit for bit" {
    // Integer kernels, so "agree" means exactly equal, no tolerance. This is
    // the test that licenses using the asm path at all.
    var prng = std.Random.DefaultPrng.init(0x54321);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 4, 15, 16, 63, 64, 65, 127, 128, 255, 256, 384, 768, 1536 };

    inline for ([_]usize{ 16, 32, 64 }) |L| {
        const K = DotU8I8(L);
        if (comptime !K.uses_vnni) continue;
        for (dims) |d| {
            const a = try std.testing.allocator.alloc(u8, d);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(i8, d);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = rnd.int(u8);
                y.* = rnd.int(i8);
            }
            try std.testing.expectEqual(K.callGeneric(a, b), K.callVnni(a, b));
        }

        // And at the saturation extremes, where a naive i16 lowering breaks.
        const d = 512;
        const a = try std.testing.allocator.alloc(u8, d);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(i8, d);
        defer std.testing.allocator.free(b);
        for ([_]struct { u8, i8 }{ .{ 255, 127 }, .{ 255, -128 }, .{ 0, -128 }, .{ 255, 0 } }) |pair| {
            @memset(a, pair[0]);
            @memset(b, pair[1]);
            try std.testing.expectEqual(K.callGeneric(a, b), K.callVnni(a, b));
        }
    }
}

test "u8xu8 dot is correct at the maximum" {
    const d = 1536;
    var a: [d]u8 = undefined;
    var b: [d]u8 = undefined;
    @memset(&a, 255);
    @memset(&b, 255);
    // 1536 × 65025 = 99_878_400, comfortably inside u32.
    try std.testing.expectEqual(@as(u32, d * 255 * 255), u8u8_native.call(&a, &b));
}

test "f32xu8 asymmetric dot matches the reference" {
    var prng = std.Random.DefaultPrng.init(0x0a5e);
    const rnd = prng.random();
    const dims = [_]usize{ 1, 7, 8, 33, 128, 768, 1536 };

    for (dims) |d| {
        const q = try std.testing.allocator.alloc(f32, d);
        defer std.testing.allocator.free(q);
        const c = try std.testing.allocator.alloc(u8, d);
        defer std.testing.allocator.free(c);
        for (q, c) |*x, *y| {
            x.* = rnd.floatNorm(f32);
            y.* = rnd.int(u8);
        }
        var want: f64 = 0;
        for (q, c) |x, y| want += @as(f64, x) * @as(f64, @floatFromInt(y));

        const got = f32u8_native.call(q, c);
        const scale = @max(1.0, @abs(want));
        try std.testing.expect(@abs(@as(f64, got) - want) / scale < 1e-5);
    }
}

test "all three int kernels agree on non-negative signed input" {
    // Where the ranges overlap (b in [0,127]) the u8×i8 and u8×u8 kernels must
    // produce identical results. A disagreement means one of them is
    // mis-widening.
    var prng = std.Random.DefaultPrng.init(0xabc);
    const rnd = prng.random();
    const d = 768;
    var a: [d]u8 = undefined;
    var bs: [d]i8 = undefined;
    var bu: [d]u8 = undefined;
    for (0..d) |i| {
        a[i] = rnd.int(u8);
        const v = rnd.uintLessThan(u8, 128);
        bs[i] = @intCast(v);
        bu[i] = v;
    }
    const got_si = u8i8_native.call(&a, &bs);
    const got_uu = u8u8_native.call(&a, &bu);
    try std.testing.expectEqual(got_si, @as(i32, @intCast(got_uu)));
}

test "u8 euclid and manhattan match a scalar reference exactly" {
    // Integer kernels, so "close" is not the bar: every intermediate is
    // exactly representable and any difference is a bug in the vector path,
    // not rounding.
    var prng = std.Random.DefaultPrng.init(0xd7a);
    const rnd = prng.random();
    for ([_]usize{ 1, 7, 16, 31, 64, 128, 129, 768 }) |dim| {
        const a = try std.testing.allocator.alloc(u8, dim);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(u8, dim);
        defer std.testing.allocator.free(b);
        for (a, b) |*x, *y| {
            x.* = rnd.int(u8);
            y.* = rnd.int(u8);
        }

        var sq: i32 = 0;
        var abs: i32 = 0;
        for (a, b) |x, y| {
            const d = @as(i32, x) - @as(i32, y);
            sq += d * d;
            abs += @intCast(@abs(d));
        }
        try std.testing.expectEqual(-sq, euclid_u8_native.call(a, b));
        try std.testing.expectEqual(-abs, manhattan_u8_native.call(a, b));
    }
}

test "u8 euclid is zero against itself and never positive" {
    const v = [_]u8{ 0, 255, 17, 3, 200, 1, 99, 4 };
    try std.testing.expectEqual(@as(i32, 0), euclid_u8_native.call(&v, &v));
    try std.testing.expectEqual(@as(i32, 0), manhattan_u8_native.call(&v, &v));
    const w = [_]u8{ 255, 0, 0, 0, 0, 0, 0, 0 };
    try std.testing.expect(euclid_u8_native.call(&v, &w) < 0);
}

test "the asymmetric SQ8 euclid and manhattan kernels match their scalar form" {
    // These replaced scalar loops, so the bar is that they compute the same
    // quantity. Not bit-identical: the vector path uses four accumulators and
    // a tree reduction, exactly as the `dot` arm beside it always has.
    var prng = std.Random.DefaultPrng.init(0xc0de);
    const rnd = prng.random();
    const aff = Affine{ .lo = -1.25, .alpha = 0.0098 };

    for ([_]usize{ 1, 5, 16, 33, 128, 768 }) |dim| {
        const q = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(q);
        const c = try std.testing.allocator.alloc(u8, dim);
        defer std.testing.allocator.free(c);
        for (q, c) |*x, *y| {
            x.* = rnd.floatNorm(f32);
            y.* = rnd.int(u8);
        }

        var want_e: f32 = 0;
        var want_m: f32 = 0;
        for (q, c) |x, y| {
            const d = x - (aff.lo + aff.alpha * @as(f32, @floatFromInt(y)));
            want_e += d * d;
            want_m += @abs(d);
        }
        try std.testing.expectApproxEqRel(-want_e, euclid_f32u8_native.call(q, c, aff), 1e-5);
        try std.testing.expectApproxEqRel(-want_m, manhattan_f32u8_native.call(q, c, aff), 1e-5);
    }
}

test "an exact reconstruction scores zero distance" {
    // A code that dequantizes exactly to the query component must give 0, and
    // the sign convention must stay "higher is better".
    const aff = Affine{ .lo = 0.5, .alpha = 0.25 };
    var q: [16]f32 = undefined;
    var c: [16]u8 = undefined;
    for (&q, &c, 0..) |*x, *y, i| {
        y.* = @intCast(i * 3);
        x.* = aff.lo + aff.alpha * @as(f32, @floatFromInt(i * 3));
    }
    try std.testing.expectEqual(@as(f32, 0), euclid_f32u8_native.call(&q, &c, aff));
    try std.testing.expectEqual(@as(f32, 0), manhattan_f32u8_native.call(&q, &c, aff));
}

test "max_safe_dim per kernel is exactly the arithmetic bound of its accumulator" {
    // Recompute each bound in u128 from first principles: the largest possible
    // magnitude of one element's contribution, against the accumulator's max.
    const Bound = struct {
        fn of(comptime Acc: type, per_elem: u128) usize {
            const limit: u128 = std.math.maxInt(Acc);
            var d: usize = 0;
            // The bound is the largest d with d * per_elem <= limit ...
            d = @intCast(limit / per_elem);
            // ... which the next dimension must exceed.
            std.debug.assert(@as(u128, d) * per_elem <= limit);
            std.debug.assert(@as(u128, d + 1) * per_elem > limit);
            return d;
        }
    };
    // u8 × i8: the widest product in magnitude is 255 × (-128) = -32640.
    try std.testing.expectEqual(Bound.of(i32, 255 * 128), DotU8I8(16).max_safe_dim);
    try std.testing.expectEqual(Bound.of(i32, 255 * 128), u8i8_native.max_safe_dim);
    // u8 × u8: 255 × 255 = 65025 against u32.
    try std.testing.expectEqual(Bound.of(u32, 255 * 255), DotU8(16).max_safe_dim);
    // euclid: (0 - 255)² = 65025 against i32.
    try std.testing.expectEqual(Bound.of(i32, 255 * 255), EuclidU8(16).max_safe_dim);
    try std.testing.expectEqual(@as(usize, 33025), EuclidU8(16).max_safe_dim);
    // manhattan: |0 - 255| = 255 against i32.
    try std.testing.expectEqual(Bound.of(i32, 255), ManhattanU8(16).max_safe_dim);
    // The module-level bound is the tightest of the four, and it is euclid's.
    try std.testing.expectEqual(EuclidU8(16).max_safe_dim, max_safe_dim);
    try std.testing.expect(max_safe_dim <= DotU8I8(16).max_safe_dim);
    try std.testing.expect(max_safe_dim <= DotU8(16).max_safe_dim);
    try std.testing.expect(max_safe_dim <= ManhattanU8(16).max_safe_dim);
    // The bounds do not depend on lane count: the total is what overflows.
    inline for (.{ 16, 32, 64 }) |L| {
        try std.testing.expectEqual(EuclidU8(16).max_safe_dim, EuclidU8(L).max_safe_dim);
        try std.testing.expectEqual(DotU8I8(16).max_safe_dim, DotU8I8(L).max_safe_dim);
    }
}

test "at max_safe_dim the extreme inputs do not overflow, and match a wide reference" {
    // Every kernel at exactly its own bound with the worst-case bytes; the
    // wide (i64) reference is what the score must equal. Past the bound the
    // same inputs would wrap, which is precisely what `--max-dim` prevents.
    const alloc = std.testing.allocator;
    {
        const d = EuclidU8(16).max_safe_dim;
        const a = try alloc.alloc(u8, d);
        defer alloc.free(a);
        const b = try alloc.alloc(u8, d);
        defer alloc.free(b);
        @memset(a, 0);
        @memset(b, 255);
        const want: i64 = -@as(i64, @intCast(d)) * 65025;
        try std.testing.expectEqual(want, @as(i64, euclid_u8_native.call(a, b)));
        try std.testing.expectEqual(-@as(i64, @intCast(d)) * 255, @as(i64, manhattan_u8_native.call(a, b)));
    }
    {
        const d = DotU8I8(16).max_safe_dim;
        const a = try alloc.alloc(u8, d);
        defer alloc.free(a);
        const b = try alloc.alloc(i8, d);
        defer alloc.free(b);
        @memset(a, 255);
        @memset(b, -128);
        const want: i64 = -@as(i64, @intCast(d)) * 32640;
        try std.testing.expectEqual(want, @as(i64, u8i8_native.call(a, b)));
        @memset(b, 127);
        try std.testing.expectEqual(@as(i64, @intCast(d)) * 32385, @as(i64, u8i8_native.call(a, b)));
    }
    {
        const d = DotU8(16).max_safe_dim;
        const a = try alloc.alloc(u8, d);
        defer alloc.free(a);
        @memset(a, 255);
        const want: u64 = @as(u64, @intCast(d)) * 65025;
        try std.testing.expectEqual(want, @as(u64, u8u8_native.call(a, a)));
    }
}
