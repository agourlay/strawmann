//! The two instructions Zig has no builtin for, `vpshufb` and `vpdpbusd`, as
//! inline assembly, and the predicates that say when each one compiles.
//!
//! One copy for the whole engine: the PQ FastScan kernel, the i8 dot kernels
//! and TurboQuant's integer kernel all use these.

const std = @import("std");
const builtin = @import("builtin");

/// Is a 3-operand `vpshufb` available at `L` bytes?
///
/// Zig has no portable builtin for a *runtime*-indexed vector shuffle
/// (`@shuffle` requires a comptime mask), so the in-register table lookup that
/// makes FastScan worth having has to be written as one instruction of inline
/// assembly. Everything else falls back to the scalar loop, which is correct
/// but is not FastScan.
///
/// VEX encoding (AVX) is required for the non-destructive 3-operand form; with
/// only SSSE3 the instruction overwrites its table operand, which would force a
/// reload of the LUT on every subquantizer. The 256-bit form is AVX2 and the
/// 512-bit form is AVX512BW (`vpshufb zmm` is a BW instruction, not an F one).
pub fn hasVpshufb(comptime L: usize) bool {
    if (builtin.zig_backend != .stage2_llvm) return false;
    return switch (L) {
        16 => builtin.target.cpu.has(.x86, .avx),
        32 => builtin.target.cpu.has(.x86, .avx2),
        64 => builtin.target.cpu.has(.x86, .avx512bw),
        else => false,
    };
}

/// Is `vpdpbusd` available at a register width of `bytes`?
///
/// 512-bit needs AVX512_VNNI. 256- and 128-bit need either AVX-VNNI (the VEX
/// encoding, Alder Lake+ / Zen 4+) or AVX512_VNNI together with AVX512VL (the
/// EVEX encoding at reduced width). §6.6.2 insists the AVX-VNNI case be its own
/// matrix row rather than folded into "AVX2", and this predicate is what makes
/// the `avx2-vnni` arm actually differ from `avx2`.
pub fn hasVnniAt(comptime bytes: usize) bool {
    // The `v` register constraint is an LLVM constraint. Zig's self-hosted
    // x86_64 backend, the default for Debug builds in 0.16, rejects it, so
    // the asm path is only compiled under the LLVM backend and the portable
    // path covers everything else. `std.simd.suggestVectorLength` guards on the
    // same condition for the same reason.
    //
    // This costs nothing where it matters: §9 says results are only ever quoted
    // from `ReleaseFast`, which is an LLVM build. `build.zig` additionally
    // forces LLVM for the test step so the VNNI path is exercised by the
    // equality tests rather than silently skipped.
    if (builtin.zig_backend != .stage2_llvm) return false;

    const has_512 = builtin.target.cpu.has(.x86, .avx512vnni);
    const has_vl = builtin.target.cpu.has(.x86, .avx512vl);
    const has_vex = builtin.target.cpu.has(.x86, .avxvnni);
    return switch (bytes) {
        64 => has_512,
        32, 16 => has_vex or (has_512 and has_vl),
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
pub inline fn shuffleBytes(comptime L: usize, tbl: @Vector(L, u8), idx: @Vector(L, u8)) @Vector(L, u8) {
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

/// One `vpdpbusd` over `L` byte lanes: `acc += Σ₄ (u8 × i8)` per dword lane.
/// Only valid where `hasVnniAt(L)`.
///
/// Not marked `volatile`, it is a pure function of its operands, and letting
/// LLVM schedule and unroll around it is the entire point of keeping
/// independent accumulator chains.
pub inline fn vpdpbusd(comptime L: usize, acc: @Vector(L / 4, i32), a: @Vector(L, u8), b: @Vector(L, i8)) @Vector(L / 4, i32) {
    return asm ("vpdpbusd %[b], %[a], %[acc]"
        : [acc] "=v" (-> @Vector(L / 4, i32)),
        : [_] "0" (acc),
          [a] "v" (a),
          [b] "v" (b),
    );
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
