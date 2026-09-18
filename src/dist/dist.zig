//! §6.6, SIMD and the distance kernels.
//!
//! "This is a first-class subsystem, not an implementation detail of §6.5. It
//! gets its own directory, its own microbenchmark suite, and its own ISA
//! matrix."
//!
//! The inventory (§6.6.3):
//!
//! ```
//! dist/
//!   dot_f32.zig        @Vector(L, f32), 8 accumulators, @reduce(.Add)
//!   l2_f32.zig         fused sub+fma, or |a|²+|b|²-2ab on normalised data
//!   dot_f16.zig        F16C convert vs AVX512-FP16 native
//!   dot_i8.zig         VNNI / AVX-VNNI / vpmaddubsw fallback, sym + asym
//!   hamming.zig        vpopcntq / vpshufb-LUT / Harley-Seal
//!   pq_adc.zig         8-bit scalar LUT; 4-bit vpshufb FastScan
//!   norm.zig           ingest-time normalisation
//!   dispatch.zig       cpuid → vtable, selected once at startup
//! ```

const std = @import("std");

pub const common = @import("common.zig");
pub const metric = @import("metric.zig");
pub const datatype = @import("datatype.zig");
pub const reference = @import("reference.zig");
pub const dot_f32 = @import("dot_f32.zig");
pub const l2_f32 = @import("l2_f32.zig");
pub const norm = @import("norm.zig");
pub const hamming = @import("hamming.zig");
pub const dot_i8 = @import("dot_i8.zig");
pub const dot_f16 = @import("dot_f16.zig");
pub const pq_adc = @import("pq_adc.zig");
pub const dispatch = @import("dispatch.zig");

pub const Metric = metric.Metric;
pub const Datatype = datatype.Datatype;
pub const Kernel = metric.Kernel;

/// The native-width instantiations, used by the search path on a
/// `-Dcpu=native` build. §6.6.5: "Primary results come from -Dcpu=native
/// builds, one per target machine. This is a benchmark, not a distributed
/// binary; there is no reason to pay for dispatch on the hot path."
pub const native = struct {
    pub const dot = dot_f32.native.call;
    pub const euclid = l2_f32.euclid_native.call;
    pub const manhattan = l2_f32.manhattan_native.call;

    pub const lanes = dot_f32.native.lanes;
    pub const accumulators = dot_f32.native.accumulators;
};

/// Compute the internal similarity (higher = better) for `k`.
///
/// Note there is no `cosine` arm: §8.3 says Cosine is normalised at ingest and
/// therefore *is* dot on the search path. `Metric.kernel()` performs that
/// collapse, and it is the only place it happens.
pub inline fn similarity(k: Kernel, a: []const f32, b: []const f32) f32 {
    return switch (k) {
        .dot => native.dot(a, b),
        .euclid => native.euclid(a, b),
        .manhattan => native.manhattan(a, b),
    };
}

/// The strict-order scalar equivalent of `similarity`, for §8.2 oracle tier 2.
pub fn similarityReference(k: Kernel, a: []const f32, b: []const f32) f32 {
    return switch (k) {
        .dot => reference.dot(a, b),
        .euclid => reference.euclid(a, b),
        .manhattan => reference.manhattan(a, b),
    };
}

/// The fp64-accumulating equivalent, for tolerance calibration (§8.4).
pub fn similarityWide(k: Kernel, a: []const f32, b: []const f32) f64 {
    return switch (k) {
        .dot => reference.dotWide(a, b),
        .euclid => reference.euclidWide(a, b),
        .manhattan => reference.manhattanWide(a, b),
    };
}

test {
    std.testing.refAllDecls(@This());
    _ = common;
    _ = metric;
    _ = datatype;
    _ = reference;
    _ = dot_f32;
    _ = l2_f32;
    _ = norm;
    _ = hamming;
    _ = dot_i8;
    _ = dot_f16;
    _ = pq_adc;
    _ = dispatch;
}

test "similarity dispatches to the same kernels the metric table names" {
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    var a: [768]f32 = undefined;
    var b: [768]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }

    try std.testing.expectEqual(native.dot(&a, &b), similarity(.dot, &a, &b));
    try std.testing.expectEqual(native.euclid(&a, &b), similarity(.euclid, &a, &b));
    try std.testing.expectEqual(native.manhattan(&a, &b), similarity(.manhattan, &a, &b));

    // Cosine must route to dot, not to a separate cosine kernel.
    try std.testing.expectEqual(Kernel.dot, Metric.cosine.kernel());
}
