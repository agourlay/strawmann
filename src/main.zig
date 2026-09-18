//! strawmann server entry point.
//!
//! §6.3: "One I/O thread per core in the I/O set... Query execution: fixed
//! worker pool, one query per worker... Explicit CPU pinning for every thread,
//! configurable, defaulting to a NUMA-local layout."
//!
//! §7.1 requires the environment be *declared* rather than ambient, so every
//! knob that could change a measurement is a flag with its value printed at
//! startup, and the banner is what a result row records.

const std = @import("std");
const Io = std.Io;
const strawmann = @import("strawmann");
const build_options = @import("build_options");

const net = strawmann.net;
const api = strawmann.api;
const core = strawmann.core;
const dist = strawmann.dist;
const sysinfo = strawmann.sysinfo;

/// Every command-line flag, as one closed set.
///
/// This was eleven chained `std.mem.eql` comparisons. A switch over an enum is
/// exhaustive, so adding a member without handling it is a compile error rather
/// than a flag that parses and does nothing.
const Flag = enum {
    help,
    pin,
    no_huge_pages,
    no_bandwidth_probe,
    probe,
    host,
    port,
    io_threads,
    workers,
    capacity,
    max_dim,
    streams,
    connections,
    data_dir,
    default_placement,
    cpus,
    build_threads,
    build_mode,

    /// The text each flag is spelled with.
    ///
    /// A `switch` rather than a table beside the enum, for the same reason the
    /// dispatch below is a switch: a table is a second list of the same
    /// members, and a member missing from it compiles into a flag nobody can
    /// ever pass. The switch makes that a compile error.
    fn text(self: Flag) []const u8 {
        return switch (self) {
            .help => "--help",
            .pin => "--pin",
            .no_huge_pages => "--no-huge-pages",
            .no_bandwidth_probe => "--no-bandwidth-probe",
            .probe => "--probe",
            .host => "--host",
            .port => "--port",
            .io_threads => "--io-threads",
            .workers => "--workers",
            .capacity => "--capacity",
            .max_dim => "--max-dim",
            .streams => "--streams",
            .connections => "--connections",
            .data_dir => "--data-dir",
            .default_placement => "--default-placement",
            .cpus => "--cpus",
            .build_threads => "--build-threads",
            .build_mode => "--build-mode",
        };
    }

    fn parse(arg: []const u8) ?Flag {
        inline for (comptime std.enums.values(Flag)) |f| {
            if (std.mem.eql(u8, arg, comptime f.text())) return f;
        }
        return null;
    }
};

test "every flag is reachable from the command line and documented" {
    // Three lists used to say the same thing: the enum, a `specs` table and
    // the `--help` text. Only the first two are now tied together by the
    // compiler, so the help text is checked here instead.
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try usage(&w);
    const help = w.buffered();

    inline for (comptime std.enums.values(Flag)) |f| {
        try std.testing.expectEqual(f, Flag.parse(f.text()).?);
        if (std.mem.indexOf(u8, help, f.text()) == null) {
            std.debug.print("flag {s} is missing from --help\n", .{f.text()});
            return error.FlagNotDocumented;
        }
    }
}

/// The largest `--max-dim` the server accepts.
///
/// The integer kernels accumulate in 32 bits without saturation, so past
/// `dist.dot_i8.max_safe_dim` (33025, `EuclidU8`'s bound) an extreme pair of
/// u8 vectors wraps and the score is silently wrong. The f32 kernels and the
/// heaps carry no such bound, so this is the only one enforced here.
pub const max_dim_limit: usize = dist.dot_i8.max_safe_dim;

fn validateMaxDim(n: usize) error{MaxDimTooLarge}!void {
    if (n > max_dim_limit) return error.MaxDimTooLarge;
}

test "--max-dim refuses a value the integer kernels could overflow at" {
    try validateMaxDim(1);
    try validateMaxDim(4096);
    try validateMaxDim(max_dim_limit);
    try std.testing.expectError(error.MaxDimTooLarge, validateMaxDim(max_dim_limit + 1));
    try std.testing.expectError(error.MaxDimTooLarge, validateMaxDim(65536));
    // The default is inside the bound, or the server would refuse to start.
    try validateMaxDim((Options{}).max_dim);
    // And the bound is the kernels' own, not a number written down here.
    try std.testing.expectEqual(dist.dot_i8.max_safe_dim, max_dim_limit);
    // The --help text states the same number.
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try usage(&w);
    var num: [16]u8 = undefined;
    const spelled = try std.fmt.bufPrint(&num, "at most {d}", .{max_dim_limit});
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), spelled) != null);
}

/// Parse a required numeric argument, turning a missing value into the same
/// error every flag reports.
fn parse(comptime T: type, value: ?[]const u8) !T {
    return std.fmt.parseInt(T, value orelse return error.MissingValue, 10);
}

/// A `net.Config` field's default. `Config` needs an address, so it cannot be
/// instantiated just to read one off.
fn serverDefault(comptime field: std.meta.FieldEnum(net.Config)) usize {
    return std.meta.fieldInfo(net.Config, field).defaultValue().?;
}

/// An `api.Engine` field's default, for the same reason `serverDefault` exists:
/// the CLI's default must be the engine's, not a second copy of it that can
/// drift (`--connections` said 8 while the server said 16, and the server won
/// every run).
fn engineDefault(comptime field: std.meta.FieldEnum(api.Engine)) std.meta.fieldInfo(api.Engine, field).type {
    return std.meta.fieldInfo(api.Engine, field).defaultValue().?;
}

const Options = struct {
    host: []const u8 = "127.0.0.1",
    /// bfb's default `--uri http://localhost:6334`.
    port: u16 = 6334,
    io_threads: usize = 1,
    workers: usize = 8,
    /// §3: "Capacity is preallocated: the collection is created with a max
    /// point count derived from the first upsert rate or a config knob."
    capacity: usize = 1_100_000,
    max_dim: usize = 4096,
    /// Where a `cached` or `cold` collection's arena file goes. Without it,
    /// those placements are refused rather than silently served from pinned
    /// memory (`handlers.createCollection`).
    data_dir: ?[]const u8 = null,
    /// What a collection that states no placement gets. §5.5's default is
    /// `pinned`; Qdrant's is `cached`, so this is the knob that lets one run
    /// put both engines in the same one.
    default_placement: core.storage.Placement = .pinned,
    /// The CPUs `--pin` may use, as a list like `4-11` or `0,2,4-6`.
    ///
    /// Without it, pinning takes CPUs `0..io_threads+workers`, which assumes
    /// every core is the same core. On a heterogeneous part it is not: the
    /// development host is 4 Zen5 cores at 5.16 GHz sharing 16 MiB of L3 and 8
    /// Zen5c cores at 3.29 GHz sharing 8 MiB, so `--pin --workers 8` put three
    /// workers on the fast CCX and five on the slow one, with different cache
    /// behind each. Same query, 57% difference in clock, depending on which
    /// worker picked it up.
    ///
    /// It is also what makes `isolcpus` usable: isolating cores 4-11 and then
    /// pinning to 0-8 puts most of the server on the cores that were *not*
    /// isolated, which is worse than not pinning at all.
    cpus: ?[]const u8 = null,
    streams_per_conn: usize = serverDefault(.streams_per_conn),
    /// The server's own default, not a number restated here: the CLI said 8
    /// while the server said 16, and the reason 16 was chosen (bfb `-c 8`
    /// plus its handshake connection refused the ninth) was defeated by the
    /// binary every run actually starts.
    connections_per_io: usize = serverDefault(.connections_per_io),
    /// Threads the bulk build runs on, and which of §8.7's options it uses.
    ///
    /// Both were fields on `api.Engine` that nothing outside a unit test ever
    /// assigned, so every real run built on 8 threads in `parallel` and there
    /// was no way to ask for anything else. §8.7 picks (b) "as the default
    /// with (c) available", and (c) — `buildSerial`, tested bit-reproducible —
    /// was not available: `spec.md` says so plainly ("not selectable at
    /// runtime, so conformance runs against the parallel build") while two
    /// comments in the tree said the opposite. These are what make it
    /// selectable, and §7.1's rule that a knob which can change a measurement
    /// is a flag with its value in the banner is why they are flags.
    ///
    /// The defaults are the previous hard-coded values, so a run that names
    /// neither measures what every run so far measured.
    build_threads: usize = engineDefault(.build_threads),
    build_mode: core.collection.BuildMode = engineDefault(.build_mode),
    pin: bool = false,
    /// §5.5. Declared, never ambient, §7.1's rule, so a result row can state
    /// which arm produced it.
    huge_pages: bool = true,
    bandwidth_probe: bool = true,
    probe_only: bool = false,
};

fn usage(w: *Io.Writer) !void {
    const d: Options = .{};
    try w.print(
        \\strawmann, a Qdrant-wire-compatible vector search engine
        \\
        \\  --host <ip>          bind address           (default 127.0.0.1)
        \\  --port <n>           bind port              (default 6334)
        \\  --io-threads <n>     I/O threads            (default 1, at least 1)
        \\  --workers <n>        query workers          (default 8, at least 1)
        \\  --capacity <n>       preallocated points    (default 1100000)
        \\  --max-dim <n>        largest accepted dim   (default 4096, at most 33025)
        \\  --streams <n>        concurrent streams/conn(default {d}, at most {d})
        \\  --connections <n>    connections per io thr (default {d})
        \\  --data-dir <path>    where cached/cold arenas are mapped from
        \\  --default-placement  pinned|cached|cold for collections that ask for
        \\                       none (default pinned; qdrant's default is cached)
        \\  --build-threads <n>  threads the bulk build uses
        \\                       (default {d}, at least 1)
        \\  --build-mode         serial|parallel graph build (§8.7; default {s}.
        \\                       serial is deterministic and slow)
        \\  --pin                pin threads to cores   (§6.3, §7.1)
        \\  --cpus <list>        which cpus --pin may use, e.g. 4-11 or 0,2,4-6
        \\                       (default 0..; name them on a heterogeneous part)
        \\  --no-huge-pages      disable 2 MiB arena pages (§5.5 A/B arm)
        \\  --no-bandwidth-probe skip the startup memory-bandwidth measurement
        \\  --probe              report host capability and exit, binding no port
        \\  --help
        \\
    , .{ d.streams_per_conn, max_streams_limit, d.connections_per_io, d.build_threads, @tagName(d.build_mode) });
}

test "the CLI's connection default is the server's" {
    // Both are read from `net.Config`, so this pins the derivation rather
    // than a number, and the --help text prints the same value.
    try std.testing.expectEqual(serverDefault(.connections_per_io), (Options{}).connections_per_io);
    try std.testing.expectEqual(serverDefault(.streams_per_conn), (Options{}).streams_per_conn);
    // The number the server comment argues for: bfb `-c 8` plus a handshake.
    try std.testing.expect((Options{}).connections_per_io >= 9);
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try usage(&w);
    var num: [64]u8 = undefined;
    const spelled = try std.fmt.bufPrint(&num, "connections per io thr (default {d})", .{serverDefault(.connections_per_io)});
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), spelled) != null);
}

/// The most `--streams` can usefully be: the h2 layer's stream pool is
/// `net.h2.max_streams` wide and any buffers past it are allocated for
/// streams that can never be opened.
pub const max_streams_limit: usize = net.h2.max_streams;

/// The three thread and stream counts a value of zero (or, for streams, an
/// excess) turns into a hang rather than an error: no I/O thread means
/// nothing to `join`, no worker means every request queues forever.
fn validateCounts(io_threads: usize, workers: usize, streams: usize, build_threads: usize) error{ NoIoThreads, NoWorkers, TooManyStreams, NoBuildThreads }!void {
    if (io_threads == 0) return error.NoIoThreads;
    if (workers == 0) return error.NoWorkers;
    if (streams == 0 or streams > max_streams_limit) return error.TooManyStreams;
    // Not because zero hangs — `extendParallel` clamps with `@max(1, threads)`
    // and builds on one. That clamp is the reason to refuse it here: §7.1
    // wants the banner to be what a result row records, and `build_threads=0`
    // printed beside a build that ran on one thread is a declared value that
    // did not happen.
    if (build_threads == 0) return error.NoBuildThreads;
}

test "--io-threads, --workers and --streams refuse the values that hang or overflow" {
    const d: Options = .{};
    try validateCounts(d.io_threads, d.workers, d.streams_per_conn, d.build_threads);
    try validateCounts(1, 1, 1, 1);
    try validateCounts(1, 1, max_streams_limit, 1);
    try std.testing.expectError(error.NoIoThreads, validateCounts(0, 8, 32, 8));
    try std.testing.expectError(error.NoWorkers, validateCounts(1, 0, 32, 8));
    try std.testing.expectError(error.TooManyStreams, validateCounts(1, 8, 0, 8));
    try std.testing.expectError(error.TooManyStreams, validateCounts(1, 8, max_streams_limit + 1, 8));
    try std.testing.expectError(error.TooManyStreams, validateCounts(1, 8, 4096, 8));
    try std.testing.expectError(error.NoBuildThreads, validateCounts(1, 8, 32, 0));
    // The --help text states the same bound.
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try usage(&w);
    var num: [32]u8 = undefined;
    const spelled = try std.fmt.bufPrint(&num, "at most {d})", .{max_streams_limit});
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), spelled) != null);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [8192]u8 = undefined;
    var fw: Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const w = &fw.interface;

    var opts: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const next = struct {
            fn get(argv: []const [:0]const u8, idx: *usize) ?[]const u8 {
                if (idx.* + 1 >= argv.len) return null;
                idx.* += 1;
                return argv[idx.*];
            }
        }.get;

        const flag = Flag.parse(a) orelse {
            try w.print("unknown argument: {s}\n\n", .{a});
            try usage(w);
            try w.flush();
            return error.BadArgument;
        };

        switch (flag) {
            .help => {
                try usage(w);
                try w.flush();
                return;
            },
            .pin => opts.pin = true,
            .no_huge_pages => opts.huge_pages = false,
            .no_bandwidth_probe => opts.bandwidth_probe = false,
            .probe => opts.probe_only = true,
            .host => opts.host = next(args, &i) orelse return error.MissingValue,
            .port => opts.port = try parse(u16, next(args, &i)),
            .io_threads => opts.io_threads = try parse(usize, next(args, &i)),
            .workers => opts.workers = try parse(usize, next(args, &i)),
            .capacity => opts.capacity = try parse(usize, next(args, &i)),
            .max_dim => {
                opts.max_dim = try parse(usize, next(args, &i));
                validateMaxDim(opts.max_dim) catch {
                    try w.print("error: --max-dim {d} exceeds {d}, the largest dimension the integer (SQ8 / uint8) kernels are overflow-free at; above it a score can wrap silently\n", .{ opts.max_dim, max_dim_limit });
                    try w.flush();
                    return error.BadArgument;
                };
            },
            .streams => opts.streams_per_conn = try parse(usize, next(args, &i)),
            .connections => opts.connections_per_io = try parse(usize, next(args, &i)),
            .data_dir => opts.data_dir = next(args, &i) orelse return error.MissingValue,
            .cpus => opts.cpus = next(args, &i) orelse return error.MissingValue,
            .default_placement => {
                const v = next(args, &i) orelse return error.MissingValue;
                opts.default_placement = std.meta.stringToEnum(core.storage.Placement, v) orelse {
                    try w.print("error: --default-placement must be one of pinned, cached, cold\n", .{});
                    try w.flush();
                    return error.BadArgument;
                };
            },
            .build_threads => opts.build_threads = try parse(usize, next(args, &i)),
            .build_mode => {
                const v = next(args, &i) orelse return error.MissingValue;
                opts.build_mode = std.meta.stringToEnum(core.collection.BuildMode, v) orelse {
                    try w.print("error: --build-mode must be one of serial, parallel\n", .{});
                    try w.flush();
                    return error.BadArgument;
                };
            },
        }
    }

    validateCounts(opts.io_threads, opts.workers, opts.streams_per_conn, opts.build_threads) catch |e| {
        switch (e) {
            error.NoIoThreads => try w.print("error: --io-threads must be at least 1\n", .{}),
            error.NoWorkers => try w.print("error: --workers must be at least 1, or no request is ever served\n", .{}),
            error.TooManyStreams => try w.print("error: --streams must be between 1 and {d}, the h2 stream pool size\n", .{max_streams_limit}),
            error.NoBuildThreads => try w.print("error: --build-threads must be at least 1; the build clamps to 1 anyway and the banner would name a thread count that did not run\n", .{}),
        }
        try w.flush();
        return error.BadArgument;
    };

    const gpa = init.gpa;

    var engine = api.Engine.init(gpa);
    defer engine.deinit();
    engine.default_capacity = opts.capacity;
    engine.huge_pages = opts.huge_pages;
    engine.data_dir = opts.data_dir;
    engine.default_placement = opts.default_placement;
    engine.build_threads = opts.build_threads;
    engine.build_mode = opts.build_mode;
    if (opts.default_placement.isMapped() and opts.data_dir == null) {
        try w.print("error: --default-placement {s} needs --data-dir <path>\n", .{opts.default_placement.name()});
        try w.flush();
        return error.BadArgument;
    }
    // Must match the per-worker scratch below, or a collection could be created
    // that no worker can serve.
    engine.max_dim = opts.max_dim;

    // One context per worker, so the per-query workspace is never shared
    // (§6.3: "Per-worker preallocated: visited set, candidate heap, result
    // heap, rescore buffer, distance scratch").
    const contexts = try gpa.alloc(api.Context, opts.workers);
    defer gpa.free(contexts);

    // One cleanup, covering however many were built. An `errdefer` for the
    // partial case *plus* a `defer` for the complete one would both fire on any
    // later error return, the defers unwind in reverse and free the same
    // workspaces twice. `--host` failing to parse, or `bind()` hitting
    // EADDRINUSE, both reach that path.
    var made: usize = 0;
    defer for (contexts[0..made]) |*c| c.workspace.deinit(gpa);
    for (contexts) |*c| {
        c.* = .{ .engine = &engine, .workspace = try api.Workspace.init(gpa, opts.max_dim) };
        made += 1;
    }

    const addr = net.server.Address.parseIp4(opts.host, opts.port) orelse {
        try w.print("error: --host must be a dotted-quad IPv4 address\n", .{});
        try w.flush();
        return error.BadArgument;
    };

    // §6.3 / §7.1: pinning is explicit and declared. I/O threads take the low
    // `--probe` answers "what is this host, and what can it do?" without
    // binding a port or allocating a collection, so `scripts/doctor.py` can ask
    // the real binary rather than reimplementing the capability checks and
    // drifting from them.
    if (opts.probe_only) {
        try w.print("build mode             : {s}\n", .{build_options.optimize_mode});
        try w.print("isa build              : {s}\n", .{build_options.isa_build_name});
        try w.print("f32 lanes / accumulators: {d} / {d}\n", .{ dist.native.lanes, dist.native.accumulators });
        // The query path scores through `dist.native` (the kernels compiled
        // for this build's target, `isa build` above); `dispatch.active` is
        // the comptime-selected vtable the micro-benchmarks and `-Dforce-isa`
        // drive. This line used to be labelled as the tier the kernels
        // dispatch to, which no served query consults. `--probe` is the
        // harness's fallback when no server log survives, so it says the same
        // things the banner does.
        try w.print("dispatch table         : {s} (bench and --probe; queries use the isa build's kernels)\n", .{dist.dispatch.active.label()});
        try w.print("compiled vnni (u8xi8)  : {}\n", .{dist.dot_i8.u8i8_native.uses_vnni});
        try w.print("cores                  : {d}\n", .{std.Thread.getCpuCount() catch 0});
        if (sysinfo.probe(gpa)) |bw| {
            try w.print("bandwidth aggregate    : {d:.1} GB/s ({d} threads)\n", .{ bw.aggregate_gbps, bw.threads });
            try w.print("bandwidth single core  : {d:.1} GB/s ({d:.0}% of bus)\n", .{ bw.single_gbps, bw.singleCoreFraction() * 100 });
            try w.print("sweep 1M x 128 fp32    : {d:.1}/s\n", .{bw.sweepsPerSecond(1_000_000, 128)});
            try w.print("sweep 1M x 1536 fp32   : {d:.2}/s\n", .{bw.sweepsPerSecond(1_000_000, 1536)});
        } else {
            try w.print("bandwidth              : probe failed\n", .{});
        }
        try w.flush();
        return;
    }

    // The CPU pool `--pin` draws from. `--cpus` names it explicitly; without
    // it the pool is 0.. as before, which is only correct when every core is
    // interchangeable.
    // The budget is the *workers*. I/O threads take spare cores when there are
    // any and otherwise share a worker's, which is a change from requiring one
    // core per thread and is worth a paragraph.
    //
    // Requiring `io + workers` cores meant that on an eight-core set the server
    // ran seven workers, and findings 36 measured what that costs: the I/O
    // thread uses about 14% of a core and was holding a whole one, so search
    // ran on seven cores while Qdrant multiplexed its threads across eight. On
    // dbpedia-openai-100K at d=1536, where throughput is very nearly linear in
    // core count, that is the whole of W4's 0.96x.
    //
    // Refusing to oversubscribe was still the right instinct: two *busy*
    // threads on one core is exactly the jitter §6.3's pinning exists to
    // remove. The distinction this draws is between a busy thread and an idle
    // one. Workers still get a core each and are still refused if they cannot;
    // the I/O thread, which is measurably not busy, doubles up.
    const want = opts.io_threads + opts.workers;
    const pool = try gpa.alloc(usize, @max(want, opts.workers));
    defer gpa.free(pool);
    var n_cpus: usize = pool.len;
    if (opts.cpus) |spec| {
        n_cpus = parseCpuList(spec, pool) catch {
            try w.print("error: --cpus must be a list like 4-11 or 0,2,4-6\n", .{});
            try w.flush();
            return error.BadArgument;
        };
        if (n_cpus < opts.workers) {
            try w.print("error: --cpus lists {d} cpu(s) but --workers {d} needs one each\n", .{ n_cpus, opts.workers });
            try w.flush();
            return error.BadArgument;
        }
    } else {
        for (pool, 0..) |*c, k| c.* = k;
    }

    // Workers first, one core each. I/O threads take whatever is left over,
    // and wrap onto the workers' cores when nothing is: with `--workers 8
    // --io-threads 1` on eight cores the I/O thread shares the first worker's,
    // which costs that worker its 14% and buys the other seven a full core.
    const worker_cpus = pool[0..opts.workers];
    const spare = n_cpus - opts.workers;
    const io_cpus = if (spare >= opts.io_threads)
        pool[opts.workers .. opts.workers + opts.io_threads]
    else
        pool[0..opts.io_threads];
    const io_shares = spare < opts.io_threads;

    // §6.5 builds on "all cores", which means the set this server was given,
    // not the single core of whichever worker happened to handle the upsert
    // that triggered the build. `pool` outlives the server: it is freed by the
    // `defer` above, on the way out of `main`, after `srv.run` has returned.
    // `pool[0..n_cpus]`, not `pool`: with `--cpus` naming fewer CPUs than
    // io_threads + workers the tail of `pool` is uninitialised, and a garbage
    // value under 1024 quietly added a CPU to the build's affinity mask.
    if (opts.pin) engine.build_cpus = pool[0..n_cpus];

    var srv = try net.server.Server.init(gpa, .{
        .addr = addr,
        .io_threads = opts.io_threads,
        .worker_threads = opts.workers,
        .streams_per_conn = opts.streams_per_conn,
        .connections_per_io = opts.connections_per_io,
        .pin_io = if (opts.pin) io_cpus else null,
        .pin_workers = if (opts.pin) worker_cpus else null,
    }, dispatchToWorker, @ptrCast(contexts.ptr));
    defer srv.deinit();
    // §7.1: counters an operator can actually read. `connections_refused > 0`
    // is the difference between "the server is broken" and "raise
    // --connections"; without this the counter exists but nothing surfaces it,
    // which is how a whole benchmark run was lost to an opaque transport error.
    defer {
        const st = srv.totalStats();
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(
            &buf,
            "\nstats: accepted={d} connections_refused={d} requests={d} submit_full={d}\n",
            .{ st.accepted, st.connections_refused, st.requests, st.submit_full },
        ) catch "";
        w.print("{s}", .{line}) catch {};
        w.flush() catch {};
    }

    // Workers index into `contexts` by their own index, which `dispatch` cannot
    // see. Stashing the count alongside lets it derive the right slot.
    worker_contexts = contexts;

    const port = try srv.bind();

    try w.print("strawmann listening on {s}:{d}\n", .{ opts.host, port });
    try w.print("  reports qdrant version : {s}\n", .{build_options.qdrant_version});
    try w.print("  build mode             : {s}\n", .{build_options.optimize_mode});
    try w.print("  isa build              : {s}\n", .{build_options.isa_build_name});
    try w.print("  f32 lanes / accumulators: {d} / {d}\n", .{ dist.native.lanes, dist.native.accumulators });
    try w.print("  dispatch table         : {s} (bench and --probe; queries use the isa build's kernels)\n", .{dist.dispatch.active.label()});
    try w.print("  vnni (u8xi8)           : {}\n", .{dist.dot_i8.u8i8_native.uses_vnni});
    try w.print("  io threads / workers   : {d} / {d}\n", .{ opts.io_threads, opts.workers });
    if (opts.pin) {
        // Whether the I/O thread has a core to itself is a property of the
        // measurement, not a detail: findings 36 is entirely about the
        // difference, so §7.1's "every knob printed at startup" covers it.
        try w.print("  pinning                : io {any}{s} workers {any}\n", .{
            io_cpus,
            if (io_shares) " (shared with a worker)" else "",
            worker_cpus,
        });
    } else {
        try w.print("  pinning                : false\n", .{});
    }
    try w.print("  preallocated capacity  : {d} points\n", .{opts.capacity});
    // §6.5: the graph is built in bulk after ingest, on `build_threads`
    // cores; brute force serves only until the first build lands and for
    // `params.exact`. The banner said "brute force (M1)" long after that
    // stopped being true, and a banner that misdescribes the index is exactly
    // the kind of context §7.1 wants stated correctly beside every result.
    // The thread count only where it is used: `buildIndex`'s `.serial` arm
    // calls `buildSerial(n)` and never reads `threads`, so "serial build on 3
    // threads" declares a number that did not act on anything — the same fault
    // `--build-threads 0` is refused for.
    switch (engine.build_mode) {
        .parallel => try w.print("  index                  : HNSW, bulk parallel build on {d} threads (brute force until built and for exact)\n", .{engine.build_threads}),
        .serial => try w.print("  index                  : HNSW, bulk serial build, deterministic, one thread (brute force until built and for exact)\n", .{}),
    }

    // §7.2: the hardware baseline belongs next to the result, not in a separate
    // document nobody opens. A qps figure without the bandwidth of the machine
    // that produced it cannot be checked against the §5 cost model.
    if (opts.bandwidth_probe) {
        if (sysinfo.probe(gpa)) |bw| {
            try w.print("  memory bandwidth       : {d:.1} GB/s aggregate ({d} threads) · {d:.1} GB/s single core ({d:.0}% of bus)\n", .{
                bw.aggregate_gbps, bw.threads, bw.single_gbps, bw.singleCoreFraction() * 100,
            });
            // Sweeps, not queries: concurrent queries over one collection walk
            // the same bytes and share cache lines, so one sweep serves several
            // of them. A measured qps above `sweeps x concurrency` is the
            // signal worth chasing, since it means the scan skipped data.
            try w.print("  exhaustive sweep rate  : {d:.0}/s over 1M x 128 fp32, {d:.1}/s over 1M x 1536\n", .{
                bw.sweepsPerSecond(1_000_000, 128), bw.sweepsPerSecond(1_000_000, 1536),
            });
        } else {
            try w.print("  memory bandwidth       : probe failed (allocation)\n", .{});
        }
    } else {
        try w.print("  memory bandwidth       : not probed (--no-bandwidth-probe)\n", .{});
    }
    try w.flush();

    try srv.start();

    // Run until interrupted. The server owns its threads; joining them here
    // would need a signal handler, which M1 does not need, the benchmark
    // harness stops the process.
    srv.io_threads[0].join();
}

/// Route a request to the worker's own context.
///
/// The server passes the context pointer it was constructed with; workers are
/// distinguished by which OS thread runs them, so the slot is chosen from a
/// thread-local index set on first use.
var worker_contexts: []api.Context = &.{};
threadlocal var my_slot: ?usize = null;
var next_slot: std.atomic.Value(usize) = .init(0);

fn dispatchToWorker(ctx: *anyopaque, req: *const net.server.Request, out: *net.server.ResponseBuf) net.server.Completion {
    _ = ctx;
    const slot = my_slot orelse blk: {
        const s = next_slot.fetchAdd(1, .monotonic) % @max(1, worker_contexts.len);
        my_slot = s;
        break :blk s;
    };
    return api.handle(@ptrCast(&worker_contexts[slot]), req, out);
}

/// Parse a Linux-style CPU list — `4-11`, `0,2,4-6` — into `out`.
///
/// Returns how many CPUs were written. The same spelling `isolcpus` and
/// `taskset -c` use, so the server can be given the very string the kernel
/// cmdline was given rather than a translation of it.
fn parseCpuList(spec: []const u8, out: []usize) !usize {
    var n: usize = 0;
    var parts = std.mem.splitScalar(u8, spec, ',');
    while (parts.next()) |part| {
        const p = std.mem.trim(u8, part, " ");
        if (p.len == 0) continue;
        if (std.mem.indexOfScalar(u8, p, '-')) |dash| {
            const lo = try std.fmt.parseInt(usize, p[0..dash], 10);
            const hi = try std.fmt.parseInt(usize, p[dash + 1 ..], 10);
            if (hi < lo) return error.BadRange;
            var c = lo;
            while (c <= hi) : (c += 1) {
                if (n == out.len) return n; // the pool is full; the rest is spare
                out[n] = c;
                n += 1;
            }
        } else {
            if (n == out.len) return n;
            out[n] = try std.fmt.parseInt(usize, p, 10);
            n += 1;
        }
    }
    return n;
}

test "parseCpuList understands the spelling isolcpus uses" {
    var buf: [16]usize = undefined;

    try std.testing.expectEqual(@as(usize, 8), try parseCpuList("4-11", &buf));
    try std.testing.expectEqual(@as(usize, 4), buf[0]);
    try std.testing.expectEqual(@as(usize, 11), buf[7]);

    try std.testing.expectEqual(@as(usize, 5), try parseCpuList("0,2,4-6", &buf));
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 5, 6 }, buf[0..5]);

    // A pool smaller than the list stops rather than writing past it; the
    // caller checks the count against what it needs.
    var small: [3]usize = undefined;
    try std.testing.expectEqual(@as(usize, 3), try parseCpuList("4-11", &small));

    try std.testing.expectError(error.BadRange, parseCpuList("11-4", &buf));
    try std.testing.expectError(error.InvalidCharacter, parseCpuList("a-b", &buf));
}
