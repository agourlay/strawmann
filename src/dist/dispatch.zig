//! `dispatch`, cpuid → vtable, selected once at startup.
//!
//! §6.6.5 puts this path second, and is explicit about why:
//!
//!   "**Primary results** come from `-Dcpu=native` builds, one per target
//!    machine. This is a benchmark, not a distributed binary; there is no
//!    reason to pay for dispatch on the hot path.
//!    **Runtime dispatch** exists as a secondary path: detect features via
//!    `cpuid` at startup, populate a vtable of kernel function pointers once,
//!    never branch on ISA inside a loop. Used to (a) confirm dispatch overhead
//!    is nil and (b) model what a shipped binary would actually do."
//!
//! ## An honest limitation
//!
//! Zig has no per-function target-feature attribute (no `__attribute__((target
//! ("avx512f")))`), and every module in one `Compile` step shares a target. So
//! a single binary **cannot** contain both an AVX-512 kernel and an AVX2 kernel
//! selected at runtime, the way Qdrant's shipped binary does.
//!
//! What this vtable therefore selects among is *lane widths that the compiled
//! target already permits*. On a `-Dcpu=native` build for the Zen 5 host that
//! is 4/8/16 f32 lanes; on a `-Dcpu=x86_64_v3` build it is 4/8 only, and asking
//! for 16 yields the 8-lane kernel rather than an illegal instruction.
//!
//! That is enough for both jobs §6.6.5 assigns it:
//!
//!  (a) *Confirm dispatch overhead is nil*, an indirect call through a
//!      function pointer costs the same regardless of how the target was
//!      chosen, so measuring this vtable against the direct call answers the
//!      question exactly.
//!
//!  (b) *Model what a shipped binary would do*, the §7.5 forced-ISA matrix is
//!      what actually produces the per-microarchitecture selection table, and
//!      the M9 deliverable is that table, not this vtable. This path models the
//!      dispatch *mechanism*; the matrix supplies the *policy*.
//!
//! The alternative, building each ISA arm as a separate shared object and
//! `dlopen`-ing the right one, would reproduce Qdrant's behaviour exactly, and
//! is noted here as the thing to do if M9 concludes mixed dispatch is the
//! recommendation worth shipping. It is not needed to answer any question the
//! spec poses.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const common = @import("common.zig");
const dot_f32 = @import("dot_f32.zig");
const l2_f32 = @import("l2_f32.zig");
const metric = @import("metric.zig");

pub const Kernel = metric.Kernel;

/// The ISA tiers the vtable can select among.
pub const Tier = enum {
    /// Scalar, strict order. Not a performance tier, this is §8.2's oracle
    /// tier 2 reachable through the same interface, so the conformance harness
    /// can swap it in without a separate code path.
    scalar,
    /// 128-bit: 4 f32 lanes.
    sse,
    /// 256-bit: 8 f32 lanes.
    avx2,
    /// 512-bit: 16 f32 lanes.
    avx512,

    pub fn f32Lanes(self: Tier) usize {
        return switch (self) {
            .scalar => 1,
            .sse => 4,
            .avx2 => 8,
            .avx512 => 16,
        };
    }

    /// The tier's bare name, the spelling `-Dforce-isa=` accepts.
    ///
    /// **This is a lane width, not an instruction set.** On a `-Dcpu=native`
    /// build, `-Dforce-isa=sse2` selects the 4-lane kernel, but that kernel is
    /// still compiled with the native feature set: its multiply-adds are FMA
    /// and its encodings are VEX/EVEX. The label says `sse` because §6.6.5's
    /// knob names the *width* it forces; the ISA of the machine code is the
    /// build's, and only the §7.5 forced-ISA matrix (separate binaries) can
    /// isolate that. Print `Vtable.label()` where the difference matters; it
    /// appends the compiled feature set.
    pub fn name(self: Tier) []const u8 {
        return @tagName(self);
    }
};

/// The features this build's kernels are actually encoded with, beyond the
/// width a `Tier` names, as a suffix for labels: `""` on a plain SSE2 build,
/// `"+fma+vex"` on x86_64_v3, `"+fma+evex"` on an AVX-512 build.
pub const compiled_feature_suffix: []const u8 = blk: {
    if (builtin.cpu.arch != .x86_64) break :blk "";
    var s: []const u8 = "";
    if (builtin.cpu.has(.x86, .fma)) s = s ++ "+fma";
    if (builtin.cpu.has(.x86, .avx512f)) {
        s = s ++ "+evex";
    } else if (builtin.cpu.has(.x86, .avx)) {
        s = s ++ "+vex";
    }
    break :blk s;
};

/// The widest tier this *build* can execute. Not the widest the CPU supports -
/// see the module doc. A build for `x86_64_v3` reports `avx2` even on a Zen 5
/// host, because that is genuinely all its code contains.
pub fn compiledMaxTier() Tier {
    if (builtin.cpu.arch != .x86_64) return .scalar;
    // Mirrors `std.simd.suggestVectorLength`: `prefer_256_bit` caps the
    // effective width even when avx512f is present, which is exactly the trap
    // documented in `build.zig`.
    const lanes = common.nativeLanes(f32);
    return switch (lanes) {
        0, 1 => .scalar,
        2, 3, 4 => .sse,
        5...8 => .avx2,
        else => .avx512,
    };
}

/// What the *CPU* supports, from `cpuid`, independent of what this build
/// contains. Reported in run metadata so a result row can show both, a build
/// running below the host's capability is a fact worth seeing rather than
/// inferring.
pub fn hostMaxTier() Tier {
    if (builtin.cpu.arch != .x86_64) return .scalar;

    // Order matters here, and getting it wrong faults rather than mis-reports.
    //
    // `XGETBV` is `#UD` unless `CR4.OSXSAVE` is set, which CPUID leaf 1 ECX[27]
    // reports. Issuing it unconditionally, as an earlier version did, makes
    // this function SIGILL on a pre-XSAVE part or a VM that masks XSAVE, on a
    // machine that would otherwise have run fine at the SSE tier. Likewise
    // leaf 7 does not exist below `max_basic_leaf = 7` and returns whatever the
    // highest supported leaf returns, which is not zero.
    const leaf0 = cpuidCount(0, 0);
    const max_basic_leaf = leaf0.eax;

    const leaf1 = cpuidCount(1, 0);
    const has_sse2 = (leaf1.edx & (1 << 26)) != 0;
    const has_osxsave = (leaf1.ecx & (1 << 27)) != 0;

    if (!has_osxsave or max_basic_leaf < 7) {
        return if (has_sse2) .sse else .scalar;
    }

    // Feature bits alone are not enough: the OS must also have enabled the
    // wider register state via XCR0, or the instructions fault despite CPUID
    // advertising them. Only now is `XGETBV` safe to issue.
    const xcr0 = xgetbv0();
    const avx_state = (xcr0 & 0x4) != 0; // YMM
    const avx512_state = (xcr0 & 0xe0) == 0xe0; // opmask + ZMM_Hi256 + Hi16_ZMM

    const leaf7 = cpuidCount(7, 0);
    const has_avx2 = (leaf7.ebx & (1 << 5)) != 0;
    const has_avx512f = (leaf7.ebx & (1 << 16)) != 0;

    if (avx_state and avx512_state and has_avx512f) return .avx512;
    if (avx_state and has_avx2) return .avx2;
    // XMM state needs no OSXSAVE, which is what the early return above
    // already relies on; `sse_state` was checked here and not there.
    if (has_sse2) return .sse;
    return .scalar;
}

/// Whether the *host* has AVX and FMA usable, by the same test Rust's
/// `is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")` makes
/// (std_detect: CPUID.1:ECX.AVX[28] and .FMA[12], and only if the OS enabled
/// XMM|YMM state in XCR0, which OSXSAVE[27] gates).
///
/// This is not a tier of *our* vtable, it is the fact Qdrant 1.19.0's
/// `CosineMetric::preprocess` branches on (`simple.rs`, `MIN_DIM_SIZE_AVX`),
/// and `norm.zig` needs the identical answer to reproduce which `Σx²` order
/// Qdrant would have used on this machine. `cpuid` serialises and costs a few
/// hundred cycles, so the answer is probed once and cached; ingest calls this
/// per vector.
pub fn hostHasAvxFma() bool {
    if (builtin.cpu.arch != .x86_64) return false;
    // 0 = not yet probed, 1 = no, 2 = yes. A benign race: two threads probing
    // at once compute the same value.
    const cached = host_avx_fma.load(.acquire);
    if (cached != 0) return cached == 2;
    const answer = probeHostAvxFma();
    host_avx_fma.store(if (answer) 2 else 1, .release);
    return answer;
}

var host_avx_fma = std.atomic.Value(u8).init(0);

fn probeHostAvxFma() bool {
    const leaf1 = cpuidCount(1, 0);
    const has_osxsave = (leaf1.ecx & (1 << 27)) != 0;
    const has_avx = (leaf1.ecx & (1 << 28)) != 0;
    const has_fma = (leaf1.ecx & (1 << 12)) != 0;
    if (!has_osxsave or !has_avx or !has_fma) return false;
    // Same XCR0 gate as `hostMaxTier`: the OS must have enabled YMM state or
    // the instructions fault, and std_detect refuses to report `avx` (and so
    // `fma`) without it.
    const xcr0 = xgetbv0();
    return (xcr0 & 0x6) == 0x6;
}

const CpuidResult = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuidCount(leaf: u32, subleaf: u32) CpuidResult {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [eax] "={eax}" (eax),
          [ebx] "={ebx}" (ebx),
          [ecx] "={ecx}" (ecx),
          [edx] "={edx}" (edx),
        : [leaf] "{eax}" (leaf),
          [subleaf] "{ecx}" (subleaf),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

/// `XGETBV(0)`, the OS-enabled extended state mask.
fn xgetbv0() u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile ("xgetbv"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
        : [idx] "{ecx}" (@as(u32, 0)),
    );
    return (@as(u64, hi) << 32) | @as(u64, lo);
}

/// A distance function over two f32 slices, returning the internal similarity.
pub const SimilarityFn = *const fn ([]const f32, []const f32) f32;

/// The vtable. Populated once at startup; never branched on inside a loop.
pub const Vtable = struct {
    tier: Tier,
    dot: SimilarityFn,
    euclid: SimilarityFn,
    manhattan: SimilarityFn,

    pub fn get(self: Vtable, k: Kernel) SimilarityFn {
        return switch (k) {
            .dot => self.dot,
            .euclid => self.euclid,
            .manhattan => self.manhattan,
        };
    }

    /// The honest label: the tier's width *and* the feature set its kernels
    /// were compiled with, e.g. `sse[4 f32 lanes+fma+evex]` for
    /// `-Dforce-isa=sse2` on a native AVX-512 build. See `Tier.name` for why
    /// the bare name is not enough. The scalar tier is the strict reference,
    /// whose result the feature set cannot change, so it carries no suffix.
    pub fn label(self: Vtable) []const u8 {
        return switch (self.tier) {
            .scalar => "scalar",
            inline else => |t| comptime std.fmt.comptimePrint("{s}[{d} f32 lanes{s}]", .{ @tagName(t), t.f32Lanes(), compiled_feature_suffix }),
        };
    }
};

/// Build a vtable for an explicit tier, clamped to what this build can execute.
///
/// Clamping rather than failing is deliberate: the per-kernel override knob of
/// §6.6.5 ("a run can force, say, AVX2 hamming with AVX-512 fp32 rescore") must
/// not be able to produce an illegal instruction, and a run that asks for more
/// than the build has should still produce a valid measurement, labelled with
/// the tier it actually got.
pub fn vtableFor(requested: Tier) Vtable {
    const max = compiledMaxTier();
    const tier: Tier = if (@intFromEnum(requested) > @intFromEnum(max)) max else requested;

    return switch (tier) {
        .scalar => .{
            .tier = .scalar,
            .dot = &@import("reference.zig").dot,
            .euclid = &@import("reference.zig").euclid,
            .manhattan = &@import("reference.zig").manhattan,
        },
        .sse => makeVtable(.sse, 4),
        .avx2 => makeVtable(.avx2, 8),
        .avx512 => makeVtable(.avx512, 16),
    };
}

fn makeVtable(comptime tier: Tier, comptime L: usize) Vtable {
    const NACC = common.default_accumulators;
    return .{
        .tier = tier,
        .dot = &dot_f32.Dot(L, NACC).call,
        .euclid = &l2_f32.Euclid(L, NACC).call,
        .manhattan = &l2_f32.Manhattan(L, NACC).call,
    };
}

/// Parse the `-Dforce-isa=` build option into a tier.
///
/// "auto" means "whatever this build can execute", which for a `-Dcpu=native`
/// build is the native width and for a forced-ISA arm is that arm's width.
pub fn tierFromOption(s: []const u8) ?Tier {
    if (std.mem.eql(u8, s, "auto")) return compiledMaxTier();
    // Derived from the enum rather than listed again beside it: a tier added to
    // `Tier` used to need a line here too, and forgetting it left an arm the
    // sweep could build but `-Dforce-isa` could not select.
    inline for (comptime std.enums.values(Tier)) |t| {
        if (std.mem.eql(u8, s, @tagName(t))) return t;
    }
    // `sse2` is what `build.zig` calls the arm, after the ISA feature rather
    // than after the register width.
    if (std.mem.eql(u8, s, "sse2")) return .sse;
    return null;
}

/// The process-wide vtable, resolved once from the build option.
///
/// Deliberately a `const` initialised at comptime rather than a mutable global
/// filled in by an `init()` call: there is no window in which it is unset, no
/// atomic load on the hot path, and no way for a worker thread to observe a
/// half-populated table.
pub const active: Vtable = blk: {
    const t = tierFromOption(build_options.force_isa) orelse compiledMaxTier();
    break :blk vtableFor(t);
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "tier ordering matches lane width" {
    try std.testing.expect(@intFromEnum(Tier.scalar) < @intFromEnum(Tier.sse));
    try std.testing.expect(@intFromEnum(Tier.sse) < @intFromEnum(Tier.avx2));
    try std.testing.expect(@intFromEnum(Tier.avx2) < @intFromEnum(Tier.avx512));
    try std.testing.expectEqual(@as(usize, 1), Tier.scalar.f32Lanes());
    try std.testing.expectEqual(@as(usize, 16), Tier.avx512.f32Lanes());
}

test "vtableFor clamps a request beyond what the build can execute" {
    const max = compiledMaxTier();
    const v = vtableFor(.avx512);
    try std.testing.expect(@intFromEnum(v.tier) <= @intFromEnum(max));

    // Scalar is always available, so it is never clamped.
    try std.testing.expectEqual(Tier.scalar, vtableFor(.scalar).tier);
}

test "every tier agrees with the strict reference within tolerance" {
    var prng = std.Random.DefaultPrng.init(0xd15);
    const rnd = prng.random();
    const d = 768;
    var a: [d]f32 = undefined;
    var b: [d]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }

    const wide_dot = @import("reference.zig").dotWide(&a, &b);
    const wide_euc = @import("reference.zig").euclidWide(&a, &b);

    for ([_]Tier{ .scalar, .sse, .avx2, .avx512 }) |t| {
        const v = vtableFor(t);
        const got_dot = v.dot(&a, &b);
        const got_euc = v.euclid(&a, &b);
        try std.testing.expect(@abs(@as(f64, got_dot) - wide_dot) / @max(1.0, @abs(wide_dot)) < 1e-5);
        try std.testing.expect(@abs(@as(f64, got_euc) - wide_euc) / @max(1.0, @abs(wide_euc)) < 1e-5);
    }
}

test "the active vtable resolves and dispatches" {
    var prng = std.Random.DefaultPrng.init(0xac7);
    const rnd = prng.random();
    var a: [384]f32 = undefined;
    var b: [384]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }
    const via_vtable = active.get(.dot)(&a, &b);
    const direct = @import("dist.zig").native.dot(&a, &b);
    // The active vtable on an `auto` build is the native width, so this should
    // be exactly equal, not merely close. A mismatch means the vtable and the
    // direct path disagree about what "native" means.
    if (active.tier == compiledMaxTier() and compiledMaxTier() != .scalar) {
        try std.testing.expectEqual(direct, via_vtable);
    }
}

test "tierFromOption round-trips the documented spellings" {
    try std.testing.expectEqual(Tier.scalar, tierFromOption("scalar").?);
    try std.testing.expectEqual(Tier.sse, tierFromOption("sse2").?);
    try std.testing.expectEqual(Tier.avx2, tierFromOption("avx2").?);
    try std.testing.expectEqual(Tier.avx512, tierFromOption("avx512").?);
    try std.testing.expectEqual(compiledMaxTier(), tierFromOption("auto").?);
    try std.testing.expectEqual(@as(?Tier, null), tierFromOption("avx1024"));
}

test "hostHasAvxFma is stable across calls and matches a fresh probe" {
    const first = hostHasAvxFma();
    for (0..8) |_| try std.testing.expectEqual(first, hostHasAvxFma());
    if (builtin.cpu.arch == .x86_64) {
        try std.testing.expectEqual(probeHostAvxFma(), first);
        // A build that has FMA in its own feature set can only be running on
        // a host that has it, otherwise nothing here would execute at all.
        if (builtin.cpu.has(.x86, .fma)) try std.testing.expect(first);
    } else {
        try std.testing.expect(!first);
    }
}

test "Vtable.label names the width and the compiled feature set" {
    const scalar = vtableFor(.scalar);
    try std.testing.expectEqualStrings("scalar", scalar.label());

    const sse = vtableFor(.sse);
    try std.testing.expect(std.mem.startsWith(u8, sse.label(), "sse[4 f32 lanes"));
    try std.testing.expect(std.mem.endsWith(u8, sse.label(), compiled_feature_suffix ++ "]"));
    // On a build with FMA the bare `sse` would be a lie by omission; the label
    // must say so.
    if (builtin.cpu.arch == .x86_64 and builtin.cpu.has(.x86, .fma)) {
        try std.testing.expect(std.mem.indexOf(u8, sse.label(), "+fma") != null);
    }
    if (builtin.cpu.arch == .x86_64 and !builtin.cpu.has(.x86, .avx)) {
        try std.testing.expectEqualStrings("sse[4 f32 lanes]", sse.label());
    }
}
