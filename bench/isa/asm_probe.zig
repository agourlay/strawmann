//! §6.6.3, the disassembly probe.
//!
//! "Disassembly of each hot loop is checked into `docs/asm/` and diffed in CI.
//!  A silent change in vectorisation is a regression even if the wall-clock
//!  didn't move on the current host."
//!
//! Every kernel is re-exported here under a stable, greppable symbol name with
//! `export fn`, which forces it to be emitted as a real function rather than
//! inlined into a caller. `dump_asm.py` builds this per ISA arm, disassembles
//! it, and writes one file per (arm, kernel) into `docs/asm/`.
//!
//! This file is the mechanism by which claims about what the compiler emits
//! stop being claims. §11 lists three failure modes it catches directly:
//! LLVM quietly splitting 512-bit ops, failing to contract FMA, and spilling
//! accumulators under deep unroll, all "moderate, and invisible without
//! checking".

const strawmann = @import("strawmann");
const dist = strawmann.dist;

// --- fp32 -----------------------------------------------------------------

export fn sm_dot_f32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    return dist.dot_f32.native.call(a[0..n], b[0..n]);
}

export fn sm_euclid_f32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    return dist.l2_f32.euclid_native.call(a[0..n], b[0..n]);
}

export fn sm_manhattan_f32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    return dist.l2_f32.manhattan_native.call(a[0..n], b[0..n]);
}

// --- int8 / SQ8 -----------------------------------------------------------
//
// `sm_dot_u8i8` is the one that should lower to `vpdpbusd` on any arm with
// VNNI (both the 256-bit AVX-VNNI arm and the 512-bit AVX512_VNNI arm), and to
// the `vpmaddubsw` → `vpmaddwd` → `vpaddd` sequence on plain AVX2. That is the
// §6.6.2 claim; this symbol is where it gets checked.

export fn sm_dot_u8i8(a: [*]const u8, b: [*]const i8, n: usize) i32 {
    return dist.dot_i8.u8i8_native.call(a[0..n], b[0..n]);
}

export fn sm_dot_u8u8(a: [*]const u8, b: [*]const u8, n: usize) u32 {
    return dist.dot_i8.u8u8_native.call(a[0..n], b[0..n]);
}

export fn sm_dot_f32u8(q: [*]const f32, c: [*]const u8, n: usize) f32 {
    return dist.dot_i8.f32u8_native.call(q[0..n], c[0..n]);
}

// --- binary ---------------------------------------------------------------
//
// `sm_hamming` should lower to `vpopcntq` on any arm with AVX512_VPOPCNTDQ and
// to Muła's `vpshufb` nibble-LUT otherwise. §6.6.2 calls this "the single
// biggest ISA gap in the whole engine".

export fn sm_hamming(a: [*]const u64, b: [*]const u64, n: usize) u32 {
    return dist.hamming.native.call(a[0..n], b[0..n]);
}

pub fn main() void {}
