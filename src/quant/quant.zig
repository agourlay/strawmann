//! §6.7, quantization.
//!
//! | mode | encoding | distance | notes |
//! |---|---|---|---|
//! | `scalar` | int8, global quantile bounds (bfb uses 0.99) | VNNI `vpdpbusd` where available | symmetric and asymmetric, both measured |
//! | `binary` | 1 bit/dim, sign-based | XOR + `@popCount` | fastest; needs oversampling + rescore |
//! | `product-x{4,8,16,32,64}` | PQ, 8 bit/subquantizer | LUT ADC | k-means codebooks trained on a sample |
//! | PQ4 / FastScan | 4 bit, `vpshufb` LUT | in-register lookup | stretch goal |
//!
//! §6.7 also fixes two behaviours that are not about accuracy at all:
//!
//!   "Match bfb's flags, minus the Qdrant-proprietary `turbo*` variants (return
//!    a clear error; document the exclusion so nobody accidentally compares
//!    against them)."
//!
//!   "`always_ram` is always true for us. `--quantization-in-ram false` is
//!    accepted and ignored, loudly."

const std = @import("std");

pub const scalar = @import("scalar.zig");
pub const binary = @import("binary.zig");
pub const pq = @import("pq.zig");

/// The five PQ compression ratios, as one closed set.
///
/// Named rather than carried as a bare `usize`, because the same set was
/// spelled three ways and only two of them agreed: `0...4 => 4/8/16/32/64` when
/// decoding `ProductQuantization.CompressionRatio` from the wire, `4, 8, 16,
/// 32, 64 =>` when parsing bfb's `--quantization product-xN`, and
/// `@intFromFloat(@round(codebook.compressionRatio()))` when reporting what a
/// built store holds. The third could produce a ratio the first two reject,
/// and `product: usize` accepted it.
///
/// The wire mapping lives here, on the type, exactly as `Metric` keeps
/// `fromProto` in `dist/metric.zig`: nothing under `quant/` may depend on
/// `proto/`.
pub const CompressionRatio = enum(u32) {
    x4 = 0,
    x8 = 1,
    x16 = 2,
    x32 = 3,
    x64 = 4,

    /// qdrant `CompressionRatio`: x4=0, x8=1, x16=2, x32=3, x64=4.
    pub fn fromProto(v: u64) ?CompressionRatio {
        return std.enums.fromInt(CompressionRatio, v);
    }

    pub fn toProto(self: CompressionRatio) u32 {
        return @intFromEnum(self);
    }

    /// The divisor itself: `.x16` compresses 16:1.
    pub fn ratio(self: CompressionRatio) usize {
        return switch (self) {
            .x4 => 4,
            .x8 => 8,
            .x16 => 16,
            .x32 => 32,
            .x64 => 64,
        };
    }

    /// The inverse, for a ratio that arrives as a number: bfb's
    /// `product-x16`, and a trained codebook reporting what it achieved.
    pub fn fromRatio(n: usize) ?CompressionRatio {
        inline for (comptime std.enums.values(CompressionRatio)) |c| {
            if (c.ratio() == n) return c;
        }
        return null;
    }
};

/// Which encoding a store holds, without the request's parameters. A built PQ
/// store does not always achieve the ratio it was asked for, so "what is this"
/// and "what was requested" are different questions (see `Store.encoding`).
pub const Encoding = std.meta.Tag(Mode);

/// The quantization modes bfb can request.
pub const Mode = union(enum) {
    none,
    scalar,
    binary,
    /// `product-xN`; the payload is the compression ratio.
    product: CompressionRatio,

    /// Parse bfb's `--quantization` value.
    ///
    /// §6.7: the `turbo*` variants are Qdrant-proprietary and must produce "a
    /// clear error" rather than falling back to something similar, a silent
    /// fallback would mean publishing a comparison against an encoding we never
    /// implemented, which §11 calls "the classic way benchmark projects become
    /// useless".
    pub fn parse(s: []const u8) ParseResult {
        if (s.len == 0) return .{ .mode = .none };
        // The unparameterised modes are spelled exactly as they are tagged, so
        // they are walked rather than listed: a mode added to the union has to
        // be answered for in the switch below, instead of quietly parsing as
        // "unknown quantization mode".
        inline for (comptime std.enums.values(Encoding)) |e| {
            switch (e) {
                .none, .scalar, .binary => {
                    if (std.mem.eql(u8, s, @tagName(e))) return .{ .mode = @unionInit(Mode, @tagName(e), {}) };
                },
                // Parameterised by a compression ratio; handled below.
                .product => {},
            }
        }
        if (std.mem.startsWith(u8, s, "turbo")) {
            return .{ .rejected = "Qdrant-proprietary turbo quantization is deliberately not implemented (spec §6.7)" };
        }
        if (std.mem.startsWith(u8, s, "product-x")) {
            const n = std.fmt.parseInt(usize, s["product-x".len..], 10) catch {
                return .{ .rejected = "malformed product quantization ratio" };
            };
            const cr = CompressionRatio.fromRatio(n) orelse
                return .{ .rejected = "product quantization supports x4, x8, x16, x32 and x64 only" };
            return .{ .mode = .{ .product = cr } };
        }
        return .{ .rejected = "unknown quantization mode" };
    }

    pub fn name(self: Mode) []const u8 {
        return switch (self) {
            .none => "none",
            .scalar => "scalar",
            .binary => "binary",
            .product => "product",
        };
    }

    /// Bytes per vector under this mode, for the §5.4 working-set table.
    pub fn bytesPerVector(self: Mode, dim: usize) usize {
        return switch (self) {
            .none => dim * 4,
            .scalar => dim,
            // The *stored* size, which is what the memory system actually
            // touches. §5.4's table quotes the logical 96 B at d=768; the
            // padding to a whole vector register makes it 128 B, and reporting
            // the logical number would understate the working set.
            .binary => binary.paddedWordsFor(dim) * 8,
            // The count the store builds, not the one the ratio asks for:
            // they differ whenever `dim` does not divide evenly.
            .product => |cr| pq.effectiveSubquantizerCount(dim, cr.ratio()),
        };
    }
};

pub const ParseResult = union(enum) {
    mode: Mode,
    /// Message for the UNIMPLEMENTED / INVALID_ARGUMENT status.
    rejected: []const u8,
};

/// §6.7: "`always_ram` is always true for us. `--quantization-in-ram false` is
/// accepted and ignored, loudly."
///
/// "Loudly" means the run metadata records that the flag was overridden, so a
/// results table can never claim to have measured an on-disk configuration.
pub const always_ram = true;

pub fn describeInRamOverride(requested_in_ram: bool) ?[]const u8 {
    if (requested_in_ram) return null;
    return "quantization-in-ram=false accepted and IGNORED: strawmann is RAM-only (spec §6.7); results must not be compared against an on-disk Qdrant configuration";
}

test {
    std.testing.refAllDecls(@This());
    _ = scalar;
    _ = binary;
    _ = pq;
}

test "§6.7: turbo variants are rejected by name, not silently substituted" {
    const testing = std.testing;
    for ([_][]const u8{ "turbo", "turbo-x4", "turbo_scalar" }) |s| {
        switch (Mode.parse(s)) {
            .rejected => |msg| try testing.expect(std.mem.indexOf(u8, msg, "turbo") != null),
            .mode => return error.TurboShouldBeRejected,
        }
    }
}

test "§6.7: the modes bfb can emit all parse" {
    const testing = std.testing;
    try testing.expectEqual(Mode.none, Mode.parse("none").mode);
    try testing.expectEqual(Mode.scalar, Mode.parse("scalar").mode);
    try testing.expectEqual(Mode.binary, Mode.parse("binary").mode);
    for ([_]usize{ 4, 8, 16, 32, 64 }) |n| {
        var buf: [32]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "product-x{d}", .{n});
        try testing.expectEqual(n, Mode.parse(s).mode.product.ratio());
    }
    // An unsupported ratio is refused rather than rounded to a supported one.
    switch (Mode.parse("product-x7")) {
        .rejected => {},
        .mode => return error.ShouldReject,
    }
}

test "§5.4 working-set sizes per encoding at d=768" {
    const testing = std.testing;
    const dim = 768;
    // §5.4's table, in bytes per vector: fp32 3072, int8 768, PQ x16 192,
    // binary 96.
    try testing.expectEqual(@as(usize, 3072), Mode.bytesPerVector(.none, dim));
    try testing.expectEqual(@as(usize, 768), Mode.bytesPerVector(.scalar, dim));
    try testing.expectEqual(@as(usize, 192), Mode.bytesPerVector(.{ .product = .x16 }, dim));
    // §5.4 quotes 96 B for binary at d=768; that is the *logical* size. Stored
    // rows are padded to a whole 512-bit register (see
    // `binary.paddedWordsFor`), so what the memory system actually touches is
    // 128 B. Reporting the logical number here would understate the working set
    // that §5.4's cache-residency argument depends on.
    try testing.expectEqual(@as(usize, 96), binary.wordsFor(dim) * 8);
    try testing.expectEqual(@as(usize, 128), Mode.bytesPerVector(.binary, dim));
}

test "§5.4 product bytes/vector reports the subquantizer count the store builds" {
    const testing = std.testing;
    // 100 / (400 / 64 = 6) does not divide; the store decrements to 5.
    try testing.expectEqual(@as(usize, 6), pq.subquantizerCount(100, 64));
    try testing.expectEqual(@as(usize, 5), pq.effectiveSubquantizerCount(100, 64));
    try testing.expectEqual(@as(usize, 5), Mode.bytesPerVector(.{ .product = .x64 }, 100));
    // When it divides, requested and effective agree.
    try testing.expectEqual(@as(usize, 48), Mode.bytesPerVector(.{ .product = .x64 }, 768));
}

test "§6.7: in-ram override is reported loudly" {
    const testing = std.testing;
    try testing.expectEqual(@as(?[]const u8, null), describeInRamOverride(true));
    const msg = describeInRamOverride(false).?;
    try testing.expect(std.mem.indexOf(u8, msg, "IGNORED") != null);
}
