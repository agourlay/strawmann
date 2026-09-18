const std = @import("std");

/// §6.6.5, the forced-ISA build matrix.
///
/// "Same source, same everything else, the only variable is the ISA."
///
/// These are *not* runtime-dispatched. Each entry produces a separate binary
/// compiled for exactly that feature set, so a cell of the §7.5 matrix is a
/// whole build rather than a branch. Runtime dispatch exists separately
/// (src/dist/dispatch.zig) and is measured against these to confirm the
/// dispatch overhead is nil.
const IsaTarget = struct {
    name: []const u8,
    /// Baseline CPU model, or null for the generic x86_64 baseline.
    model: ?*const std.Target.Cpu.Model,
    /// Features added on top of the model.
    add: []const std.Target.x86.Feature,
    /// Features subtracted from the model.
    sub: []const std.Target.x86.Feature = &.{},
    doc: []const u8,
};

const isa_matrix = [_]IsaTarget{
    .{
        .name = "baseline",
        .model = &std.Target.x86.cpu.x86_64,
        .add = &.{},
        .doc = "SSE2 only, the scalar/128-bit floor, and the control arm",
    },
    .{
        .name = "avx2",
        .model = &std.Target.x86.cpu.x86_64_v3,
        .add = &.{},
        // v3 is AVX2+FMA+BMI. Explicitly subtract nothing; this is the arm the
        // "AVX2" column of §6.6.2 refers to.
        .doc = "x86_64_v3: AVX2 + FMA + BMI1/2 (no VNNI)",
    },
    .{
        .name = "avx2-vnni",
        .model = &std.Target.x86.cpu.x86_64_v3,
        .add = &.{.avxvnni},
        // §6.6.2: "AVX-VNNI gives 256-bit vpdpbusd on Alder Lake+ and Zen 4+,
        // which narrows this a lot, must be a separate row in the matrix, not
        // folded into AVX2."
        .doc = "x86_64_v3 + AVX-VNNI: 256-bit vpdpbusd, the row that must not be folded into avx2",
    },
    // ---------------------------------------------------------------------
    // A note on `prefer_256_bit`, discovered while wiring this matrix up.
    //
    // LLVM's `x86_64_v4` model sets the `prefer_256_bit` feature. Zig's
    // `std.simd.suggestVectorLength` honours it and returns 8 f32 lanes rather
    // than 16, so a build that looks like "AVX-512" emits 256-bit code with
    // EVEX encodings. Every arm below that intends 512-bit width therefore
    // subtracts it explicitly.
    //
    // This is §11's "LLVM quietly splits 512-bit ops" risk showing up before a
    // single kernel was benchmarked, and it is exactly why §7.3 insists on
    // `fp_arith_inst_retired.512b_packed_single` to confirm the executed width
    // rather than trusting the build flags.
    //
    // The preference is not *wrong*, it is LLVM encoding the §6.6.4 frequency
    // licensing story as a default. That makes it a question to measure, not a
    // setting to override blindly, so `avx512-256` below keeps it and becomes
    // its own row: "AVX-512 instructions at 256-bit width" is a real and often
    // optimal configuration on parts with licensing, and it is precisely the
    // mixed configuration §6.6.5 predicts may win.
    // ---------------------------------------------------------------------
    .{
        .name = "avx512-256",
        .model = &std.Target.x86.cpu.x86_64_v4,
        .add = &.{},
        // Keeps prefer_256_bit: EVEX encodings, 32 registers, mask registers,
        // but 256-bit datapath. What a naive `-Dcpu=x86_64_v4` build gives you.
        .doc = "x86_64_v4 as LLVM defaults it: AVX-512 ISA at 256-bit width (prefer_256_bit kept)",
    },
    .{
        .name = "avx512",
        .model = &std.Target.x86.cpu.x86_64_v4,
        .add = &.{},
        .sub = &.{.prefer_256_bit},
        // v4 is AVX-512 F/BW/DQ/VL/CD. Deliberately *without* VNNI/VPOPCNTDQ so
        // the width effect is separable from the instruction-set effect.
        .doc = "x86_64_v4 at true 512-bit width, width only, no VNNI, no VPOPCNTDQ",
    },
    .{
        .name = "avx512-vnni",
        .model = &std.Target.x86.cpu.x86_64_v4,
        .add = &.{.avx512vnni},
        .sub = &.{.prefer_256_bit},
        .doc = "x86_64_v4 + AVX512_VNNI at 512-bit: vpdpbusd for the SQ8 kernel",
    },
    .{
        .name = "avx512-popcnt",
        .model = &std.Target.x86.cpu.x86_64_v4,
        .add = &.{.avx512vpopcntdq},
        .sub = &.{.prefer_256_bit},
        // §6.6.2 calls vpopcntq "the single biggest ISA gap in the whole engine".
        .doc = "x86_64_v4 + AVX512_VPOPCNTDQ at 512-bit: vpopcntq for the binary kernel",
    },
    .{
        .name = "avx512-full",
        .model = &std.Target.x86.cpu.x86_64_v4,
        .add = &.{ .avx512vnni, .avx512vpopcntdq, .avx512vbmi, .avx512bitalg, .avx512ifma },
        .sub = &.{.prefer_256_bit},
        .doc = "x86_64_v4 + VNNI + VPOPCNTDQ + VBMI + BITALG at 512-bit: everything the engine can use",
    },
};

/// Every value surfaced into the source as `@import("build_options")`. The
/// native build and the eight ISA arms each get their own module, and the
/// single constructor is what keeps the two option sets from drifting apart.
const BuildOptions = struct {
    qdrant_version: []const u8,
    force_isa: []const u8,
    isa_build_name: []const u8,
    optimize: std.builtin.OptimizeMode,
};

fn buildOptions(b: *std.Build, o: BuildOptions) *std.Build.Module {
    const opts = b.addOptions();
    opts.addOption([]const u8, "qdrant_version", o.qdrant_version);
    opts.addOption([]const u8, "force_isa", o.force_isa);
    opts.addOption([]const u8, "isa_build_name", o.isa_build_name);
    // §9: "results are only ever quoted from ReleaseFast." The banner said which
    // ISA a binary carried but never which optimize mode built it, so a Debug
    // binary left at `zig-out/bin/strawmann` served a whole run while every row
    // recorded `isa build: native` and nothing contradicted it — the two were
    // told apart afterwards only by file size. The mode is the single build fact
    // that invalidates a number outright, so it travels with the run.
    opts.addOption([]const u8, "optimize_mode", @tagName(o.optimize));
    return opts.createModule();
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ---------------------------------------------------------------------
    // Build options surfaced into the source as `@import("build_options")`.
    // ---------------------------------------------------------------------
    // §2: "Return { title: strawmann, version: 1.18.0 } and make the version
    // string a config knob so the compatibility check can be silenced for any
    // client version." qdrant-client compares major/minor against its own.
    const qdrant_version = b.option(
        []const u8,
        "qdrant-version",
        "Version string reported by HealthCheck (qdrant-client compares major/minor against its own)",
    ) orelse "1.18.0";

    // §6.6.5: "A per-kernel override knob so a run can force, say, AVX2 hamming
    // with AVX-512 fp32 rescore. Mixed configurations are likely optimal on the
    // parts with frequency licensing."
    const force_isa = b.option(
        []const u8,
        "force-isa",
        "Select the dispatch vtable the micro-benchmarks and --probe use (scalar|sse2|avx2|avx512) instead of cpuid detection; the query path scores through the kernels compiled for -Dcpu and is not affected",
    ) orelse "auto";

    // Name of the ISA build arm, recorded in every result row per §7.1 so a
    // number can never be quoted without knowing which binary produced it.
    const isa_build_name = b.option(
        []const u8,
        "isa-build-name",
        "Name of this ISA build arm, recorded in run metadata",
    ) orelse "native";

    const options_mod = buildOptions(b, .{
        .qdrant_version = qdrant_version,
        .force_isa = force_isa,
        .isa_build_name = isa_build_name,
        .optimize = optimize,
    });

    // ---------------------------------------------------------------------
    // The library module. Everything lives here so that the server, the
    // benchmarks, and the tests all compile the same code.
    // ---------------------------------------------------------------------
    const mod = b.addModule("strawmann", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = options_mod },
        },
    });

    // ---------------------------------------------------------------------
    // The server.
    // ---------------------------------------------------------------------
    const exe = b.addExecutable(.{
        .name = "strawmann",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "strawmann", .module = mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the strawmann server");
    run_step.dependOn(&run_cmd.step);

    // ---------------------------------------------------------------------
    // Tests.
    // ---------------------------------------------------------------------
    // `use_llvm = true` is not a performance choice, Debug builds default to
    // Zig's self-hosted x86_64 backend, which rejects LLVM's `v` register
    // constraint and would therefore compile out the `vpdpbusd` path in
    // `dist/dot_i8.zig` entirely. The test that asserts the VNNI and portable
    // paths agree would then pass vacuously, which is the worst outcome
    // available. Forcing LLVM keeps the asm path live under test.
    // `--test-filter` support, so a single failing test can be isolated
    // without waiting for the whole suite, which matters once the end-to-end
    // tests spin up real sockets and threads.
    const test_filters = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this substring") orelse &.{};

    const mod_tests = b.addTest(.{
        .root_module = mod,
        .use_llvm = true,
        .filters = test_filters,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_mod_tests.step);

    // `src/main.zig` is a separate root, so `mod_tests` never reached it and
    // the CLI was the one module with no tests at all. It owns the flag table,
    // which is exactly the kind of parallel list that rots quietly.
    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "strawmann", .module = mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .use_llvm = true,
        .filters = test_filters,
    });
    test_step.dependOn(&b.addRunArtifact(exe_tests).step);

    // `bench/micro/` is a third root. Its hardware, kernel and perf-counter
    // modules carry tests that were unreachable from `zig build test` until
    // this step existed. Debug so the assertions inside them are live.
    const bench_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/micro/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "strawmann", .module = mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .use_llvm = true,
        .filters = test_filters,
    });
    test_step.dependOn(&b.addRunArtifact(bench_tests).step);

    // ---------------------------------------------------------------------
    // Microbenchmarks (§7.2 layers 1 and 2).
    // ---------------------------------------------------------------------
    const bench_micro = b.addExecutable(.{
        .name = "bench-micro",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/micro/main.zig"),
            .target = target,
            // §9: results are only ever quoted from ReleaseFast.
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "strawmann", .module = mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    b.installArtifact(bench_micro);

    const run_bench_micro = b.addRunArtifact(bench_micro);
    run_bench_micro.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_bench_micro.addArgs(args);
    const bench_step = b.step("bench", "Run the hardware + kernel microbenchmarks");
    bench_step.dependOn(&run_bench_micro.step);

    // ---------------------------------------------------------------------
    // §6.6.5 / §9, `zig build bench-isa` produces the whole matrix from one
    // source tree. Each arm is a separate ReleaseFast binary; the only variable
    // is the ISA.
    // ---------------------------------------------------------------------
    const isa_step = b.step("bench-isa", "Build the forced-ISA benchmark matrix (§7.5)");
    const isa_list_step = b.step("isa-list", "List the ISA build arms and what they isolate");

    for (isa_matrix) |isa| {
        var q: std.Target.Query = .{
            .cpu_arch = .x86_64,
            .os_tag = .linux,
            .abi = .gnu,
        };
        if (isa.model) |m| q.cpu_model = .{ .explicit = m };
        for (isa.add) |f| q.cpu_features_add.addFeature(@intFromEnum(f));
        for (isa.sub) |f| q.cpu_features_sub.addFeature(@intFromEnum(f));

        const isa_target = b.resolveTargetQuery(q);

        // Same helper as the native build, so an option added there is
        // defined here too: the arms once lacked `optimize_mode` and every
        // one of them failed to compile the moment `main.zig` read it.
        const isa_options_mod = buildOptions(b, .{
            .qdrant_version = qdrant_version,
            .force_isa = "auto",
            .isa_build_name = isa.name,
            .optimize = .ReleaseFast,
        });

        const isa_mod = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = isa_target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "build_options", .module = isa_options_mod },
            },
        });

        const isa_bench = b.addExecutable(.{
            .name = b.fmt("bench-micro-{s}", .{isa.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("bench/micro/main.zig"),
                .target = isa_target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "strawmann", .module = isa_mod },
                    .{ .name = "build_options", .module = isa_options_mod },
                },
            }),
        });

        const isa_server = b.addExecutable(.{
            .name = b.fmt("strawmann-{s}", .{isa.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/main.zig"),
                .target = isa_target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "strawmann", .module = isa_mod },
                    .{ .name = "build_options", .module = isa_options_mod },
                },
            }),
        });

        // §6.6.3, the disassembly probe, one per arm. Kernels are re-exported
        // as real symbols so `objdump` can find them; without this they inline
        // into their callers and there is nothing to diff.
        const isa_asm = b.addExecutable(.{
            .name = b.fmt("asm-probe-{s}", .{isa.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("bench/isa/asm_probe.zig"),
                .target = isa_target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "strawmann", .module = isa_mod },
                    .{ .name = "build_options", .module = isa_options_mod },
                },
            }),
        });

        // Install into a per-arm subdirectory so the matrix runner can find
        // them by name and so two arms can never overwrite each other.
        const install_bench = b.addInstallArtifact(isa_bench, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("isa/{s}", .{isa.name}) } },
        });
        const install_server = b.addInstallArtifact(isa_server, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("isa/{s}", .{isa.name}) } },
        });
        const install_asm = b.addInstallArtifact(isa_asm, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("isa/{s}", .{isa.name}) } },
        });
        isa_step.dependOn(&install_bench.step);
        isa_step.dependOn(&install_server.step);
        isa_step.dependOn(&install_asm.step);

        const echo = b.addSystemCommand(&.{ "printf", "  %-16s %s\n", isa.name, isa.doc });
        isa_list_step.dependOn(&echo.step);
    }

    // ---------------------------------------------------------------------
    // §6.6.3 / §11, disassembly of every hot loop is checked into docs/asm/
    // and diffed in CI. "A silent change in vectorisation is a regression even
    // if the wall-clock didn't move on the current host."
    // ---------------------------------------------------------------------
    const asm_step = b.step("asm", "Regenerate docs/asm/ disassembly of the hot loops");
    const asm_cmd = b.addSystemCommand(&.{"bench/isa/dump_asm.py"});
    asm_cmd.step.dependOn(isa_step);
    asm_step.dependOn(&asm_cmd.step);
}
