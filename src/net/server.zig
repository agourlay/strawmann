//! The epoll event loop, connection lifecycle, and worker-pool fan-out.
//!
//! §6.1 states the constraint that shapes this whole file:
//!
//!   "**`-c 1` is the bfb default.** One connection means all concurrency is
//!    multiplexed over a single TCP stream handled by one I/O thread. Therefore
//!    request *execution* must fan out to the worker pool, with completions
//!    routed back to the owning I/O thread via an MPSC queue plus `eventfd`. A
//!    design where the I/O thread executes the query serialises the default
//!    benchmark and would be a self-inflicted wound."
//!
//! So the architecture is not "thread per connection" and not "execute inline".
//! It is §6's diagram:
//!
//! ```
//!   N I/O threads, SO_REUSEPORT, own epoll each
//!     h2c framing · HPACK · gRPC framing · protobuf decode
//!                    │ MPSC submit
//!   Query execution pool, one core per query
//!     planner → index scan → rescore → top-k
//!                    │ MPSC complete + eventfd
//!   Storage: immutable during search
//! ```
//!
//! §6.3: "One I/O thread per core in the I/O set, `SO_REUSEPORT` for
//! connection-level distribution." SO_REUSEPORT does nothing for bfb's default
//! single connection, that is precisely why the worker fan-out has to exist -
//! but it is what lets `-c N` scale without a shared accept lock.

const std = @import("std");
const linux = std.os.linux;

const h2 = @import("h2.zig");
const grpc = @import("grpc.zig");

pub const Error = error{
    ListenFailed,
    EpollFailed,
    OutOfConnections,
};

/// A unit of work handed to the worker pool.
pub const Request = struct {
    conn: *Connection,
    stream_idx: usize,
    stream_id: u31,
    path: []const u8,
    body: []const u8,
    /// `nowNs()` when the I/O thread enqueued the request, which is when the
    /// server *received* it. The reported `time` runs from here.
    ///
    /// It used to be stamped at dequeue, so the reported time was execution
    /// only and excluded whatever the request spent in the worker queue. §2
    /// wants `time` "measured honestly at the RPC boundary", and the boundary
    /// is arrival: a saturated worker pool is server-side latency the client
    /// experiences and Qdrant's `time` (a `Instant::now()` taken in the tonic
    /// handler before the request is dispatched) includes its own queueing.
    /// Reporting execution only made a queue-bound run look faster than it
    /// was, on the one number bfb takes on trust.
    started_ns: u64 = 0,
    /// A sub-request of a fanned-out batch (`api.handlers` `BatchJob`): the
    /// handler runs one query of the job and returns a detached completion;
    /// the job's last finisher completes the parent stream. Null for an
    /// ordinary request. Opaque here: the server moves work between workers
    /// and knows nothing about what a batch is.
    job: ?*anyopaque = null,
    sub: u32 = 0,
};

/// A finished request, routed back to the owning I/O thread.
pub const Completion = struct {
    conn: *Connection,
    stream_idx: usize,
    stream_id: u31,
    status: grpc.Status,
    /// Message for a non-ok status. Points into static storage or the
    /// connection's response arena; never heap.
    message: []const u8 = "",
    /// Encoded protobuf response body, in the connection's response arena.
    body: []const u8 = "",
    /// Server-side wall time, seconds. §2: bfb reads `time` from the response
    /// as the server-side timing and feeds it to the "server_timings"
    /// histogram, "so it must be measured honestly at the RPC boundary".
    elapsed_s: f64 = 0,
    /// Whether this response type carries a `double time = 2`.
    ///
    /// Every response in §12 does *except* `HealthCheckReply`, whose field 2 is
    /// `version` (a string). Appending a fixed64 there would produce a message
    /// that decodes to a corrupt version string, so the timing is opt-out per
    /// RPC rather than appended blindly.
    wants_time: bool = true,
    /// The handler has taken ownership of completing this stream itself,
    /// later, from whichever worker finishes last (`completeDetached`): the
    /// worker loop must neither append `time` nor push anything for it. A
    /// batch fanned out across workers returns this from the worker that
    /// dispatched it and from every sub-request.
    detached: bool = false,
    /// Field number the `time` double occupies in *this* response message.
    ///
    /// Two is right for `QueryBatchResponse`, `PointsOperationResponse` and
    /// `CollectionOperationResponse`, which is every response §12 lists for M1
    /// So an early version hardcoded it. `ScrollResponse` breaks the pattern:
    /// `next_page_offset` is 1, `result` is 2 and `time` is **3**. Appending
    /// field 2 there does not merely mislabel the timing, it injects a fixed64
    /// into the middle of the repeated `result` list, and a conformant decoder
    /// reading `result` as a length-delimited field gets a garbage length.
    ///
    /// Verified per message against qdrant-client 1.19's generated `qdrant.rs`.
    time_field: u32 = 2,
};

/// The handler a server is constructed with. Runs on a worker thread.
pub const HandlerFn = *const fn (ctx: *anyopaque, req: *const Request, out: *ResponseBuf) Completion;

/// Scratch space a handler writes its response into.
pub const ResponseBuf = struct {
    buf: []u8,
    len: usize = 0,

    pub fn available(self: *const ResponseBuf) []u8 {
        return self.buf[self.len..];
    }

    pub fn commit(self: *ResponseBuf, n: usize) []const u8 {
        // Committing more than was written hands the caller a slice over
        // whatever the previous response left in the buffer, which goes out on
        // the wire as trailing garbage.
        std.debug.assert(self.len + n <= self.buf.len);
        const s = self.buf[self.len..][0..n];
        self.len += n;
        return s;
    }

    pub fn reset(self: *ResponseBuf) void {
        self.len = 0;
    }
};

const lock = @import("../lock.zig");

/// Re-exported so callers that already have a `server.Mutex` keep working.
pub const Mutex = lock.Mutex;
const Futex = struct {
    const wait = lock.futexWait;
    const wake = lock.futexWake;
};
const wake_all = lock.wake_all;

/// Bounded MPSC queue.
///
/// Fixed capacity, no allocation. §6.3 forbids allocation on the query path,
/// not locking; the queue is touched twice per request against thousands of
/// distance evaluations, so contention here should not be measurable, and
/// `perf c2c` (§7.3) is how that assumption gets checked rather than assumed. A
/// lock-free ring is the thing to write if it ever shows up in a profile.
pub fn Queue(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        items: [capacity]T = undefined,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,
        mutex: Mutex = .{},
        /// Bumped on every push and on close. Consumers park on this value, so
        /// a signal can never be lost between the predicate check and the wait.
        seq: std.atomic.Value(u32) = .init(0),
        closed: std.atomic.Value(bool) = .init(false),

        /// Fails when full or closed. `closed` is checked under the mutex
        /// and set under it in `close`, so a push can never land after the
        /// consumers have seen the queue closed and drained: an item pushed
        /// before `close` is visible to `pop`, one after is refused.
        pub fn push(self: *Self, item: T) bool {
            self.mutex.lock();
            if (self.count == capacity or self.closed.load(.acquire)) {
                self.mutex.unlock();
                return false;
            }
            self.items[self.tail] = item;
            self.tail = (self.tail + 1) % capacity;
            self.count += 1;
            self.mutex.unlock();

            _ = self.seq.fetchAdd(1, .release);
            Futex.wake(&self.seq, 1);
            return true;
        }

        /// Blocking pop. Returns null once closed and drained.
        pub fn pop(self: *Self) ?T {
            while (true) {
                // Read the sequence *before* checking the queue. If a push
                // lands between the check and the wait, the sequence will have
                // moved and the wait returns immediately instead of sleeping
                // through the item that was just added.
                const observed = self.seq.load(.acquire);

                self.mutex.lock();
                if (self.count > 0) {
                    const item = self.items[self.head];
                    self.head = (self.head + 1) % capacity;
                    self.count -= 1;
                    self.mutex.unlock();
                    return item;
                }
                self.mutex.unlock();

                if (self.closed.load(.acquire)) return null;
                Futex.wait(&self.seq, observed);
            }
        }

        /// Non-blocking pop, for draining the completion queue on an eventfd
        /// wakeup without ever parking the I/O thread.
        pub fn tryPop(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.count == 0) return null;
            const item = self.items[self.head];
            self.head = (self.head + 1) % capacity;
            self.count -= 1;
            return item;
        }

        pub fn close(self: *Self) void {
            self.mutex.lock();
            self.closed.store(true, .release);
            self.mutex.unlock();
            _ = self.seq.fetchAdd(1, .release);
            // Wake every parked worker, not one: shutdown must release all of
            // them, and each will observe `closed` and return null.
            Futex.wake(&self.seq, wake_all);
        }

        pub fn isClosed(self: *const Self) bool {
            return self.closed.load(.acquire);
        }
    };
}

pub const max_connections = 64;
pub const submit_capacity = 1024;
pub const complete_capacity = 1024;

/// Per-connection state owned by one I/O thread.
pub const Connection = struct {
    fd: linux.fd_t = -1,
    active: bool = false,

    h2c: h2.Connection = undefined,

    /// Read buffer, compacted as frames are consumed.
    in: []u8 = &.{},
    in_len: usize = 0,

    /// Output frames, flushed once per epoll wakeup (§6.1's writev batching).
    out_backing: []u8 = &.{},
    out: h2.OutBuf = undefined,
    /// Bytes already written from `out`, for partial-write resumption.
    out_sent: usize = 0,

    /// Per-stream request and response scratch. Indexed by stream slot, so two
    /// concurrent requests on one connection never share a buffer.
    scratch: []StreamScratch = &.{},
    /// The request halves of `scratch`, in the shape `h2.Connection.init` wants.
    stream_bufs: [][]u8 = &.{},

    /// Requests dispatched but not yet completed. The connection cannot be
    /// closed while this is non-zero, or a worker would write into freed
    /// buffers.
    inflight: std.atomic.Value(u32) = .{ .raw = 0 },
    /// Set when a close was requested while requests were still in flight.
    /// The actual close is deferred to whoever retires the last completion -
    /// without this the slot and its fd are never reclaimed, because the close
    /// path returns early and nothing revisits the decision.
    closing: bool = false,
    /// Input parsing is suspended because `out` is down to its control-frame
    /// headroom. Every inbound frame may have to be answered (SETTINGS ack,
    /// PING ack, WINDOW_UPDATE top-up), and an answer that could not be
    /// written used to be mapped to a PROTOCOL_ERROR GOAWAY. Instead the
    /// reader stops, EPOLLIN interest is dropped so the kernel holds the
    /// bytes, and `ioMain` resumes parsing once the buffer has been flushed.
    paused: bool = false,

    /// Owning I/O thread's completion queue, so a worker knows where to send
    /// the result back.
    owner: *IoThread = undefined,
};

/// One I/O thread: an epoll set, its connections, and its completion queue.
pub const IoThread = struct {
    index: usize,
    epfd: linux.fd_t = -1,
    listen_fd: linux.fd_t = -1,
    event_fd: linux.fd_t = -1,

    conns: []Connection = &.{},
    completions: Queue(Completion, complete_capacity) = .{},

    submit: *Queue(Request, submit_capacity) = undefined,
    running: *std.atomic.Value(bool) = undefined,

    /// Counters exported via the §5.7 side channel.
    stats: Stats = .{},

    pub const Stats = struct {
        accepted: u64 = 0,
        /// Connections turned away because every slot was busy.
        ///
        /// Counted because the refusal is otherwise invisible: the fd is closed
        /// before the HTTP/2 preface, so the client sees a bare transport error
        /// with no status and no message. W4 failed this way for an entire
        /// benchmark run, `-c 8` against 8 slots, where the client's own
        /// health-check channel needed a ninth, and the only symptom was
        /// `Unknown error transport error MetadataMap {}`.
        connections_refused: u64 = 0,
        requests: u64 = 0,
        bytes_in: u64 = 0,
        bytes_out: u64 = 0,
        /// Requests refused because the submit queue was full. A non-zero value
        /// means the worker pool is the bottleneck, which is a finding rather
        /// than an error.
        submit_full: u64 = 0,
        /// Times a reader was paused on a full write buffer (see
        /// `Connection.paused`). Exposed so a slow-reader test can check the
        /// pause path was actually taken rather than assume it.
        pauses: u64 = 0,
    };
};

/// Notify an I/O thread that a completion is waiting.
fn notify(event_fd: linux.fd_t) void {
    const one: u64 = 1;
    _ = sys.write(event_fd, std.mem.asBytes(&one)) catch {};
}

/// Thin, checked wrappers over the raw syscalls.
///
/// Zig 0.16 moved the socket API out of `std.posix` and into the `Io`
/// abstraction. This server implements its own event loop by design (§6.1,
/// §6.3), so it talks to the kernel directly rather than adopting a runtime it
/// then has to work around. Each wrapper turns the raw `usize` return into a
/// Zig error, which is the only thing the std layer was providing here.
pub const sys = struct {
    pub const SysError = error{SyscallFailed};

    fn check(rc: usize) SysError!usize {
        return switch (linux.errno(rc)) {
            .SUCCESS => rc,
            else => SysError.SyscallFailed,
        };
    }

    /// Errno for a failed call, so callers can distinguish EAGAIN from a real
    /// failure without a second syscall.
    pub fn errnoOf(rc: usize) linux.E {
        return linux.errno(rc);
    }

    pub fn socket(domain: u32, sock_type: u32, protocol: u32) SysError!linux.fd_t {
        return @intCast(try check(linux.socket(domain, sock_type, protocol)));
    }

    pub fn bind(fd: linux.fd_t, addr: *const linux.sockaddr, len: linux.socklen_t) SysError!void {
        _ = try check(linux.bind(fd, addr, len));
    }

    pub fn listen(fd: linux.fd_t, backlog: u32) SysError!void {
        _ = try check(linux.listen(fd, backlog));
    }

    pub fn accept4(fd: linux.fd_t, flags: u32) SysError!?linux.fd_t {
        const rc = linux.accept4(fd, null, null, flags);
        return switch (linux.errno(rc)) {
            .SUCCESS => @as(linux.fd_t, @intCast(rc)),
            // Another I/O thread won the race, or the backlog drained.
            .AGAIN, .CONNABORTED, .INTR => null,
            else => SysError.SyscallFailed,
        };
    }

    pub fn setsockoptInt(fd: linux.fd_t, level: i32, optname: u32, value: c_int) SysError!void {
        var v = value;
        _ = try check(linux.setsockopt(fd, level, optname, @ptrCast(&v), @sizeOf(c_int)));
    }

    pub fn getsockname(fd: linux.fd_t, addr: *linux.sockaddr, len: *linux.socklen_t) SysError!void {
        _ = try check(linux.getsockname(fd, addr, len));
    }

    pub fn close(fd: linux.fd_t) void {
        _ = linux.close(fd);
    }

    pub fn epollCreate() SysError!linux.fd_t {
        return @intCast(try check(linux.epoll_create1(linux.EPOLL.CLOEXEC)));
    }

    pub fn epollAdd(epfd: linux.fd_t, fd: linux.fd_t, events: u32, data: u64) SysError!void {
        var ev: linux.epoll_event = .{ .events = events, .data = .{ .u64 = data } };
        _ = try check(linux.epoll_ctl(epfd, linux.EPOLL.CTL_ADD, fd, &ev));
    }

    pub fn epollMod(epfd: linux.fd_t, fd: linux.fd_t, events: u32, data: u64) SysError!void {
        var ev: linux.epoll_event = .{ .events = events, .data = .{ .u64 = data } };
        _ = try check(linux.epoll_ctl(epfd, linux.EPOLL.CTL_MOD, fd, &ev));
    }

    pub fn epollDel(epfd: linux.fd_t, fd: linux.fd_t) void {
        _ = linux.epoll_ctl(epfd, linux.EPOLL.CTL_DEL, fd, null);
    }

    pub fn epollWait(epfd: linux.fd_t, events: []linux.epoll_event, timeout_ms: i32) usize {
        const rc = linux.epoll_wait(epfd, events.ptr, @intCast(events.len), timeout_ms);
        return switch (linux.errno(rc)) {
            .SUCCESS => rc,
            else => 0,
        };
    }

    pub fn eventfd(initval: u32, flags: u32) SysError!linux.fd_t {
        return @intCast(try check(linux.eventfd(initval, flags)));
    }

    /// Read, returning null on EAGAIN and 0 on clean EOF.
    pub fn read(fd: linux.fd_t, buf: []u8) SysError!?usize {
        const rc = linux.read(fd, buf.ptr, buf.len);
        return switch (linux.errno(rc)) {
            .SUCCESS => rc,
            .AGAIN, .INTR => null,
            else => SysError.SyscallFailed,
        };
    }

    /// Write, returning null on EAGAIN (the socket buffer is full).
    pub fn write(fd: linux.fd_t, buf: []const u8) SysError!?usize {
        const rc = linux.write(fd, buf.ptr, buf.len);
        return switch (linux.errno(rc)) {
            .SUCCESS => rc,
            .AGAIN, .INTR => null,
            else => SysError.SyscallFailed,
        };
    }
};

/// An IPv4 bind address.
///
/// Built here rather than taken from `std.Io.net`, whose address types are
/// shaped around the async runtime this server deliberately does not use. The
/// listener needs a `sockaddr_in` and nothing more.
pub const Address = struct {
    /// Host byte order.
    ip: u32,
    port: u16,

    pub const loopback: u32 = 0x7f000001;
    pub const any: u32 = 0;

    pub fn init(ip: u32, port: u16) Address {
        return .{ .ip = ip, .port = port };
    }

    /// Parse dotted-quad IPv4. Returns null on anything else, the server binds
    /// what an operator configured, and a silently-wrong bind address is worse
    /// than a startup failure.
    pub fn parseIp4(text: []const u8, port: u16) ?Address {
        var octets: [4]u8 = undefined;
        var it = std.mem.splitScalar(u8, text, '.');
        for (&octets) |*o| {
            const part = it.next() orelse return null;
            o.* = std.fmt.parseInt(u8, part, 10) catch return null;
        }
        if (it.next() != null) return null;
        return .{
            .ip = (@as(u32, octets[0]) << 24) | (@as(u32, octets[1]) << 16) |
                (@as(u32, octets[2]) << 8) | octets[3],
            .port = port,
        };
    }

    pub fn sockaddrIn(self: Address) linux.sockaddr.in {
        return .{
            .family = linux.AF.INET,
            .port = std.mem.nativeToBig(u16, self.port),
            .addr = std.mem.nativeToBig(u32, self.ip),
            .zero = @splat(0),
        };
    }
};

/// Bind a listening socket with SO_REUSEPORT.
///
/// §6.3: "One I/O thread per core in the I/O set, `SO_REUSEPORT` for
/// connection-level distribution." Each I/O thread binds its *own* socket to
/// the same port; the kernel hashes incoming connections across them, which
/// avoids the thundering herd and the shared accept lock a single shared
/// listener would need.
pub fn listenReusePort(addr: Address) !linux.fd_t {
    const fd = try sys.socket(
        linux.AF.INET,
        linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC,
        linux.IPPROTO.TCP,
    );
    errdefer sys.close(fd);

    try sys.setsockoptInt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, 1);
    try sys.setsockoptInt(fd, linux.SOL.SOCKET, linux.SO.REUSEPORT, 1);

    const sa = addr.sockaddrIn();
    try sys.bind(fd, @ptrCast(&sa), @sizeOf(linux.sockaddr.in));
    try sys.listen(fd, 512);
    return fd;
}

/// The port a listening socket actually bound, for tests and for logging when
/// port 0 was requested.
pub fn boundPort(fd: linux.fd_t) !u16 {
    var sa: linux.sockaddr.in = undefined;
    var len: linux.socklen_t = @sizeOf(linux.sockaddr.in);
    try sys.getsockname(fd, @ptrCast(&sa), &len);
    return std.mem.bigToNative(u16, sa.port);
}

/// Disable Nagle.
///
/// gRPC responses are small and latency-sensitive, and a request/response
/// pattern with Nagle on interacts badly with delayed ACK: the response sits in
/// the send buffer waiting for more data that will never come until the client
/// sends the next request. That shows up as a bimodal latency distribution with
/// a ~40 ms mode, which would wreck the p99 numbers §7.4 asks for.
pub fn setNoDelay(fd: linux.fd_t) void {
    // TCP_NODELAY == 1.
    sys.setsockoptInt(fd, linux.IPPROTO.TCP, 1, 1) catch {};
}

/// Pin the calling thread to `cpu`.
///
/// §6.3: "Explicit CPU pinning for every thread, configurable, defaulting to a
/// NUMA-local layout." §7.1 additionally requires `isolcpus`/`nohz_full` for
/// server cores, which is a host-level setting `bench/setup.sh` verifies.
pub fn pinToCpu(cpu: usize) void {
    pinToCpus(&.{cpu});
}

/// Pin the calling thread to a *set* of cpus.
///
/// The single-cpu form is right for an I/O thread or a worker, each of which
/// owns one core. It is wrong for anything that then spawns its own pool:
/// `std.Thread.spawn` children inherit the parent's affinity mask, so a bulk
/// index build started from a worker pinned to cpu 10 put all
/// `build_threads` of its workers on cpu 10 as well. Measured:
/// eight build threads, `Cpus_allowed_list: 10` on every one of them, the
/// whole process using 100% of a single core while the banner advertised
/// "bulk parallel build on 8 threads". The build still finished; it took
/// roughly eight times as long, which is how it came to blow through the
/// harness's row timeout and read as a hang.
pub fn pinToCpus(cpus: []const usize) void {
    if (cpus.len == 0) return;
    var set: linux.cpu_set_t = std.mem.zeroes(linux.cpu_set_t);
    const bits_per_elem = @bitSizeOf(@TypeOf(set[0]));
    for (cpus) |cpu| {
        const idx = cpu / bits_per_elem;
        const bit = cpu % bits_per_elem;
        if (idx >= set.len) continue;
        set[idx] |= @as(@TypeOf(set[0]), 1) << @intCast(bit);
    }
    // Affinity may legitimately be denied in a container; a benchmark run
    // that cannot pin is a §7.1 environment failure to report, not a crash.
    linux.sched_setaffinity(0, &set) catch {};
}

/// The nice value an index build runs at: Qdrant's, 10
/// (`common::cpu::linux_low_thread_priority`, applied to every HNSW build
/// thread in `hnsw/build.rs`).
pub const build_nice: i32 = 10;

/// Lower the calling thread to `build_nice`. On Linux a nice value belongs to
/// a thread and threads inherit their creator's, so calling this before the
/// build pool spawns covers the whole pool. Search then preempts the build
/// rather than splitting the cores with it: on 0925's W11 the eight build
/// threads ran at the search workers' weight on the same eight CPUs, and the
/// search spent 1,568 s waiting in the run queue. Soft-fails, as Qdrant's
/// does: a build at normal priority is slower search, not a wrong answer.
pub fn lowerThreadPriority() void {
    const PRIO_PROCESS = 0;
    const tid: usize = @intCast(linux.gettid());
    _ = linux.syscall3(.setpriority, PRIO_PROCESS, tid, @as(usize, @bitCast(@as(isize, build_nice))));
}

/// The calling thread's nice value, from `getpriority`, which returns
/// `20 - nice` so that success is never negative.
pub fn threadNice() i32 {
    const PRIO_PROCESS = 0;
    const tid: usize = @intCast(linux.gettid());
    const r = linux.syscall2(.getpriority, PRIO_PROCESS, tid);
    return 20 - @as(i32, @intCast(r));
}

test "a lowered thread and the threads it spawns run at the build nice value" {
    const Probe = struct {
        child_nice: i32 = 0,
        own_nice: i32 = 0,
        fn child(self: *@This()) void {
            self.child_nice = threadNice();
        }
        fn run(self: *@This()) void {
            lowerThreadPriority();
            self.own_nice = threadNice();
            const t = std.Thread.spawn(.{}, child, .{self}) catch return;
            t.join();
        }
    };
    const before = threadNice();
    var probe: Probe = .{};
    const t = try std.Thread.spawn(.{}, Probe.run, .{&probe});
    t.join();
    try std.testing.expectEqual(build_nice, probe.own_nice);
    try std.testing.expectEqual(build_nice, probe.child_nice);
    // Per thread: the one that spawned the build is untouched.
    try std.testing.expectEqual(before, threadNice());
}

// =========================================================================
// The event loop
// =========================================================================

pub const Config = struct {
    addr: Address,
    /// §6.3: "One I/O thread per core in the I/O set."
    io_threads: usize = 1,
    /// §6.3: "Query execution: fixed worker pool, one query per worker."
    worker_threads: usize = 4,
    /// Concurrent streams per connection that are backed by buffers.
    streams_per_conn: usize = 32,
    /// Request body buffer per stream. Sized for an upsert batch: bfb's
    /// default `-b 100` at d=1536 fp32 is ~614 KB.
    request_buffer: usize = 1024 * 1024,
    /// Response arena per stream. A QueryBatch of 16 x limit 100 encodes to
    /// well under 100 KB; 256 KB leaves room for `--search-limit` far higher.
    response_buffer: usize = 256 * 1024,
    /// Socket read buffer per connection.
    read_buffer: usize = 256 * 1024,
    /// Frame output buffer per connection.
    write_buffer: usize = 1024 * 1024,
    /// SO_SNDBUF for the listeners, inherited by accepted sockets. Null keeps
    /// the kernel's autotuned default. A test sets it small so a client that
    /// stops reading pushes back on the write buffer within kilobytes rather
    /// than after the megabytes the kernel would otherwise absorb.
    sndbuf: ?c_int = null,
    /// Connection slots per I/O thread.
    ///
    /// Each slot preallocates `streams_per_conn` request and response buffers
    /// (§6.3 forbids allocation on the request path), so the memory cost is
    ///
    ///     connections_per_io x streams_per_conn x (request_buffer + response_buffer)
    ///
    /// At the defaults, 16 x 32 x 1.25 MB = 640 MB. That is why this is not
    /// simply set large.
    ///
    /// It was 8, which is below what an ordinary benchmark client asks for:
    /// bfb's `-c 8` opens eight connections and its client needs a further one
    /// for the version handshake, so the ninth was refused and the run died
    /// (see `stats.connections_refused`). Sixteen leaves headroom for that without
    /// quadrupling the footprint.
    connections_per_io: usize = 16,
    /// CPU indices for I/O threads, then workers. Null disables pinning.
    pin_io: ?[]const usize = null,
    pin_workers: ?[]const usize = null,
};

/// Per-stream scratch, owned by the connection so a completion's `body` stays
/// valid until the response has been written.
///
/// A shared per-connection or per-worker arena would be a use-after-free the
/// moment two requests are in flight on one connection, which is the *normal*
/// case under `-c 1 -p 64`.
const StreamScratch = struct {
    request: []u8 = &.{},
    response: []u8 = &.{},
    /// The response being written for this slot, if any. Owned by the I/O
    /// thread from the moment a completion is drained (or an error is decided
    /// on the I/O thread) until the trailers are out.
    resp: PendingResponse = .{},
};

/// A response in progress on the I/O thread.
///
/// A completion is not written in one piece: the DATA is bounded by the
/// peer's max frame size and by both flow-control windows, and every frame is
/// bounded by the room left in the connection's write buffer. Whatever cannot
/// go out now is left here and picked up by `resumePending` after a
/// WINDOW_UPDATE arrives or the buffer is flushed. Holding it in the stream's
/// own scratch keeps the completion queue free of head-of-line blocking: one
/// slow reader parks its own streams and nobody else's.
const PendingResponse = struct {
    stage: enum { none, headers, data, trailers } = .none,
    status: grpc.Status = .ok,
    /// Static storage or this stream's response arena, both stable until the
    /// slot is freed.
    message: []const u8 = "",
    body: []const u8 = "",
    /// Progress through `body` as `grpc.writeData` counts it.
    off: usize = 0,
};

pub const Server = struct {
    alloc: std.mem.Allocator,
    config: Config,
    handler: HandlerFn,
    ctx: *anyopaque,

    submit: Queue(Request, submit_capacity) = .{},
    running: std.atomic.Value(bool) = .init(true),

    io: []IoThread = &.{},
    io_threads: []std.Thread = &.{},
    workers: []std.Thread = &.{},

    /// Backing allocations, freed in `deinit`.
    scratch: [][]StreamScratch = &.{},
    stream_bufs: [][][]u8 = &.{},

    pub fn init(alloc: std.mem.Allocator, config: Config, handler: HandlerFn, ctx: *anyopaque) !*Server {
        const self = try alloc.create(Server);
        self.* = .{ .alloc = alloc, .config = config, .handler = handler, .ctx = ctx };
        // `deinit` walks these arrays, so every slice it will visit is set to
        // empty *before* the allocation that might fail. `Allocator.free`
        // returns immediately on a zero-length slice, which makes a partially
        // built server safe to tear down and lets one errdefer cover the whole
        // nested construction. Previously the only errdefer freed the `Server`
        // struct itself and every buffer allocated before the failure leaked.
        errdefer self.deinit();

        // Responses advance only while the write buffer has `reserve` free
        // (see `pumpResponse`); a buffer that small could never carry one.
        std.debug.assert(config.write_buffer >= 4 * h2.OutBuf.reserve);
        // The stream receive window is never credited for an accepted body
        // (`creditStreamWindow` runs only for a body being discarded), which
        // is correct only while every accepted body fits under the initial
        // window: a larger request buffer would let a client stall forever on
        // one upload with no error. A runtime refusal, since the shipped
        // build compiles asserts out.
        if (config.request_buffer > h2.Settings.default_window) return error.RequestBufferExceedsWindow;

        const total_conns = config.io_threads * config.connections_per_io;

        self.scratch = try alloc.alloc([]StreamScratch, total_conns);
        @memset(self.scratch, &.{});
        self.stream_bufs = try alloc.alloc([][]u8, total_conns);
        @memset(self.stream_bufs, &.{});
        for (self.scratch, self.stream_bufs) |*sc, *sb| {
            sc.* = try alloc.alloc(StreamScratch, config.streams_per_conn);
            @memset(sc.*, .{});
            sb.* = try alloc.alloc([]u8, config.streams_per_conn);
            @memset(sb.*, &.{});
            for (sc.*, sb.*) |*one, *ptr| {
                one.request = try alloc.alloc(u8, config.request_buffer);
                one.response = try alloc.alloc(u8, config.response_buffer);
                ptr.* = one.request;
            }
        }

        self.io = try alloc.alloc(IoThread, config.io_threads);
        // `index` has no default, so the placeholder names it. Every other
        // field defaults to the empty slice `deinit` can walk.
        @memset(self.io, .{ .index = 0 });
        for (self.io, 0..) |*t, i| {
            t.* = .{ .index = i };
            t.submit = &self.submit;
            t.running = &self.running;
            t.conns = try alloc.alloc(Connection, config.connections_per_io);
            @memset(t.conns, .{});
            for (t.conns, 0..) |*c, j| {
                const flat = i * config.connections_per_io + j;
                c.* = .{};
                c.in = try alloc.alloc(u8, config.read_buffer);
                c.out_backing = try alloc.alloc(u8, config.write_buffer);
                c.scratch = self.scratch[flat];
                c.stream_bufs = self.stream_bufs[flat];
                c.owner = t;
            }
        }
        return self;
    }

    pub fn deinit(self: *Server) void {
        for (self.io) |*t| {
            for (t.conns) |*c| {
                self.alloc.free(c.in);
                self.alloc.free(c.out_backing);
            }
            self.alloc.free(t.conns);
        }
        self.alloc.free(self.io);
        for (self.scratch, self.stream_bufs) |sc, sb| {
            for (sc) |one| {
                self.alloc.free(one.request);
                self.alloc.free(one.response);
            }
            self.alloc.free(sc);
            self.alloc.free(sb);
        }
        self.alloc.free(self.scratch);
        self.alloc.free(self.stream_bufs);
        if (self.io_threads.len > 0) self.alloc.free(self.io_threads);
        if (self.workers.len > 0) self.alloc.free(self.workers);
        self.alloc.destroy(self);
    }

    /// Bind every I/O thread's listener before spawning anything.
    ///
    /// Binding up front means a port conflict is a startup error rather than a
    /// thread that dies silently after `start` has already returned success.
    pub fn bind(self: *Server) !u16 {
        var port = self.config.addr.port;
        for (self.io) |*t| {
            t.listen_fd = try listenReusePort(Address.init(self.config.addr.ip, port));
            if (self.config.sndbuf) |n| try sys.setsockoptInt(t.listen_fd, linux.SOL.SOCKET, linux.SO.SNDBUF, n);
            if (port == 0) port = try boundPort(t.listen_fd);
            t.event_fd = try sys.eventfd(0, linux.EFD.NONBLOCK | linux.EFD.CLOEXEC);
            t.epfd = try sys.epollCreate();
            try sys.epollAdd(t.epfd, t.listen_fd, linux.EPOLL.IN, tag_listen);
            try sys.epollAdd(t.epfd, t.event_fd, linux.EPOLL.IN, tag_event);
        }
        return port;
    }

    pub fn start(self: *Server) !void {
        // `stop` (and `deinit`) join exactly `self.workers` and
        // `self.io_threads`, so a spawn that fails midway must leave those
        // slices holding only the threads that exist: joining an undefined
        // handle is UB. The slices are grown one spawned thread at a time and
        // a failure stops what was started, in the order `stop` uses.
        const workers = try self.alloc.alloc(std.Thread, self.config.worker_threads);
        errdefer self.alloc.free(workers);
        self.workers = workers[0..0];
        errdefer {
            self.submit.close();
            for (self.workers) |t| t.join();
            self.workers = &.{};
        }
        for (workers, 0..) |*w, i| {
            w.* = try std.Thread.spawn(.{}, workerMain, .{ self, i });
            self.workers = workers[0 .. i + 1];
        }
        const io_threads = try self.alloc.alloc(std.Thread, self.io.len);
        errdefer self.alloc.free(io_threads);
        self.io_threads = io_threads[0..0];
        errdefer {
            self.running.store(false, .release);
            for (self.io) |*t| notify(t.event_fd);
            for (self.io_threads) |t| t.join();
            self.io_threads = &.{};
        }
        for (io_threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{}, ioMain, .{ self, i });
            self.io_threads = io_threads[0 .. i + 1];
        }
    }

    pub fn stop(self: *Server) void {
        // Workers first, with the I/O threads still serving: the queue is
        // drained, each completion is written, and a request arriving after
        // the close is answered UNAVAILABLE by `dispatch` rather than
        // stranded (`Queue.push` refuses once closed). The wait is bounded by
        // the queue depth times one handler; the response's flush is best
        // effort.
        self.submit.close();
        for (self.workers) |t| t.join();
        self.running.store(false, .release);
        // Wake each I/O thread out of epoll_wait so it observes `running`.
        for (self.io) |*t| notify(t.event_fd);
        for (self.io_threads) |t| t.join();
        for (self.io) |*t| {
            for (t.conns) |*c| {
                if (!c.active) continue;
                // Tell the peer which streams were seen, so a client with
                // requests in flight can retry them elsewhere instead of
                // reporting a bare connection reset. Best effort: the socket
                // is non-blocking and its buffer may be full.
                //
                // Only when nothing is half-written: a GOAWAY spliced into a
                // partially sent frame is a framing error, and the peer would
                // read that instead of the stream id.
                if (c.out_sent == c.out.len) {
                    var gb: [32]u8 = undefined;
                    var go = h2.OutBuf.init(&gb);
                    c.h2c.sendGoaway(&go, .no_error) catch {};
                    _ = sys.write(c.fd, go.written()) catch null;
                }
                sys.close(c.fd);
            }
            sys.close(t.listen_fd);
            sys.close(t.event_fd);
            sys.close(t.epfd);
        }
    }

    pub fn totalStats(self: *const Server) IoThread.Stats {
        var acc: IoThread.Stats = .{};
        for (self.io) |*t| {
            acc.accepted += t.stats.accepted;
            acc.connections_refused += t.stats.connections_refused;
            acc.requests += t.stats.requests;
            acc.bytes_in += t.stats.bytes_in;
            acc.bytes_out += t.stats.bytes_out;
            acc.submit_full += t.stats.submit_full;
            acc.pauses += t.stats.pauses;
        }
        return acc;
    }
};

/// epoll user-data tags. Connections are tagged by index so a wakeup resolves
/// without a lookup.
const tag_listen: u64 = std.math.maxInt(u64);
const tag_event: u64 = std.math.maxInt(u64) - 1;

fn workerMain(server: *Server, index: usize) void {
    std.debug.assert(index < server.config.worker_threads);
    if (server.config.pin_workers) |cpus| {
        if (index < cpus.len) pinToCpu(cpus[index]);
    }
    while (server.submit.pop()) |req| {
        const conn = req.conn;
        const scratch = conn.scratch[req.stream_idx];
        var rb = ResponseBuf{ .buf = scratch.response };

        var r = req;
        // Stamped by `dispatch` at enqueue. A request that arrives without a
        // stamp (a test constructing one by hand) is timed from here rather
        // than from the epoch.
        if (r.started_ns == 0) r.started_ns = nowNs();
        var completion = server.handler(server.ctx, &r, &rb);
        // The handler will complete the stream itself, from the worker that
        // finishes its last piece (`completeDetached`); nothing to append or
        // push here, and the response arena is still theirs.
        if (completion.detached) continue;
        completion.conn = conn;
        completion.stream_idx = req.stream_idx;
        completion.stream_id = req.stream_id;
        finishCompletion(conn, &rb, completion, r.started_ns);
    }
}

/// Complete a stream from a handler that returned `detached`: the same
/// timing, `time` field and hand-off to the I/O thread an ordinary completion
/// gets from the worker loop, invoked by whichever worker finished the last
/// piece of the work. `rb` must be the stream's own response arena
/// (`responseArena`), with the body already committed into it.
pub fn completeDetached(req: *const Request, rb: *ResponseBuf, completion_in: Completion) void {
    var completion = completion_in;
    completion.conn = req.conn;
    completion.stream_idx = req.stream_idx;
    completion.stream_id = req.stream_id;
    completion.detached = false;
    finishCompletion(req.conn, rb, completion, req.started_ns);
}

/// The stream's response arena, for a handler completing a stream later than
/// the call that dispatched it (`completeDetached`). The worker loop builds
/// the same view for the synchronous path.
pub fn responseArena(req: *const Request) ResponseBuf {
    return .{ .buf = req.conn.scratch[req.stream_idx].response };
}

/// The bytes of the stream's request buffer after the body: free until the
/// stream completes, since the body was the last thing written there and the
/// slot is held until its completion is drained (a second `.request` for a
/// dispatched slot is refused, `h2.zig`). A handler may use them as scratch
/// that has to outlive its own call, which is what a batch fanned out across
/// workers needs and what §6.3's no-allocation-on-the-query-path rule leaves
/// no other room for. 64-byte aligned, so any struct fits.
pub fn requestSpare(req: *const Request) []align(64) u8 {
    const buf = req.conn.scratch[req.stream_idx].request;
    const base = @intFromPtr(buf.ptr);
    const body_end = @intFromPtr(req.body.ptr) + req.body.len;
    std.debug.assert(body_end >= base and body_end <= base + buf.len);
    const start = std.mem.alignForward(usize, body_end, 64);
    if (start >= base + buf.len) return &.{};
    const off = start - base;
    return @alignCast(buf[off..]);
}

/// Hand a sub-request to the worker pool: the same queue the I/O threads
/// submit to, so a sub-request queues behind whatever is already waiting and
/// no worker is reserved for it. False when the queue is full or closed; the
/// caller runs the piece itself then, so a batch never depends on the push.
pub fn submitSub(req: *const Request, sub: Request) bool {
    return req.conn.owner.submit.push(sub);
}

/// Time, `time` field, hand-off. Shared by the worker loop and
/// `completeDetached`; see the comments inline for why each part is here.
fn finishCompletion(conn: *Connection, rb: *ResponseBuf, completion_in: Completion, started_ns: u64) void {
    var completion = completion_in;
    // §2: `time` must be "measured honestly at the RPC boundary". From
    // arrival on the I/O thread to the end of the handler: queueing plus
    // execution, which is what the client's server-side latency is. The
    // response write is not included, and neither is it in Qdrant's.
    completion.elapsed_s = elapsedSince(started_ns);
    {
        // Append the `time` field to the response.
        //
        // §2: "`time` (seconds, f64) is read by bfb as the *server-side*
        // timing, it feeds the 'server_timings' histogram, so it must be
        // measured honestly at the RPC boundary." The handler cannot write it
        // because the elapsed time is not known until the handler has
        // returned, so it is appended here.
        //
        // Appending is legal and uniform: protobuf fields may appear in any
        // order, and every response in §12 that carries a timing puts it at
        // field 2 as a `double`, `QueryBatchResponse`, `PointsOperationResponse`
        // and `CollectionOperationResponse` alike. The handler's body is the
        // last thing committed to this stream's buffer, so the appended bytes
        // are contiguous with it.
        if (completion.status == .ok and completion.wants_time) {
            const tail = rb.available();
            // Single-byte tag only: every `time` field number in §12 is below
            // 16, so the tag fits one byte. An assert rather than a varint
            // encoder, because a field number that outgrew it would silently
            // truncate here.
            std.debug.assert(completion.time_field < 16);
            // The arena is sized for the largest response plus this tail
            // (`Config.response_buffer`); a handler that filled it to the
            // last byte would otherwise ship a response with `time` silently
            // missing, and bfb would chart a zero. A runtime check, not an
            // assert: the shipped binary is ReleaseFast, where an assert is
            // `unreachable` and the write below would land nine bytes past
            // the arena. The response becomes a refusal naming the remedy,
            // the same answer `queryBatchFits` gives before writing a byte.
            if (tail.len < 9) {
                completion.status = .resource_exhausted;
                completion.message = "response filled the arena with no room for the time field; raise the server's response_buffer";
                completion.body = "";
                completion.wants_time = false;
            }
        }
        if (completion.status == .ok and completion.wants_time) {
            const tail = rb.available();
            // The extension below is a pointer bump rather than a copy, so it
            // is only correct while the handler's body ends exactly where the
            // arena's free space begins. Every handler commits its body last
            // today; one that grew a sub-slice out of it, or committed
            // anything after it, would ship nine bytes of its neighbour
            // instead of `time`. Asserted here because the wire stays
            // well-formed either way, so nothing downstream would notice.
            std.debug.assert(completion.body.len == 0 or
                @intFromPtr(completion.body.ptr) + completion.body.len == @intFromPtr(tail.ptr));
            tail[0] = @intCast((completion.time_field << 3) | 1); // wire type 1 (fixed64)
            std.mem.writeInt(u64, tail[1..9], @bitCast(completion.elapsed_s), .little);
            const appended = rb.commit(9);
            // An empty body (`CollectionOperationResponse{result=false}`, a
            // delete of a missing collection: proto3 omits a false bool) is
            // the static `""`, not the arena, so it cannot be extended in
            // place; the response *is* the tail. Skipping `time` for it made
            // the client decode `time = 0.0`, and Qdrant always sends it.
            completion.body = if (completion.body.len == 0)
                appended
            else
                completion.body.ptr[0 .. completion.body.len + 9];
        }

        // Push until accepted: the completion queue is sized to the submit
        // queue, so a full one means the I/O thread is briefly behind rather
        // than wedged, and dropping a completion would hang the client.
        while (!conn.owner.completions.push(completion)) {
            std.Thread.yield() catch {};
        }
        notify(conn.owner.event_fd);
    }
}

/// The `SETTINGS_MAX_FRAME_SIZE` a connection advertises: what the read
/// buffer can hold, floored at the RFC's 16384. Split out so the test that
/// pins the invariant exercises this function rather than a copy of its
/// arithmetic, which is what it used to do.
pub fn advertisedMaxFrame(read_buffer: usize) u32 {
    const usable_frame = read_buffer - h2.frame_header_len;
    return @intCast(@max(16384, @min(@as(usize, h2.Settings.default_max_frame), usable_frame)));
}

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Seconds since a `nowNs()` stamp, saturating at zero for a stamp that
/// somehow post-dates now.
fn elapsedSince(started_ns: u64) f64 {
    const now = nowNs();
    return @as(f64, @floatFromInt(now -| started_ns)) / 1e9;
}

fn ioMain(server: *Server, index: usize) void {
    const t = &server.io[index];
    if (server.config.pin_io) |cpus| {
        if (index < cpus.len) pinToCpu(cpus[index]);
    }

    var events: [64]linux.epoll_event = undefined;
    while (server.running.load(.acquire)) {
        const n = sys.epollWait(t.epfd, &events, 50);
        for (events[0..n]) |ev| {
            switch (ev.data.u64) {
                tag_listen => acceptLoop(server, t),
                tag_event => {
                    var drain: [8]u8 = undefined;
                    _ = sys.read(t.event_fd, &drain) catch null;
                    drainCompletions(t);
                },
                else => {
                    const ci: usize = @intCast(ev.data.u64);
                    if (ci >= t.conns.len) continue;
                    const c = &t.conns[ci];
                    if (!c.active) continue;
                    if (ev.events & (linux.EPOLL.HUP | linux.EPOLL.ERR) != 0) {
                        closeConn(t, c, ci);
                        continue;
                    }
                    if (ev.events & linux.EPOLL.OUT != 0) flushOut(t, c, ci);
                    if (ev.events & linux.EPOLL.IN != 0) readConn(server, t, c, ci);
                },
            }
        }
        // Completions can arrive while we were servicing sockets.
        drainCompletions(t);
        for (t.conns, 0..) |*c, ci| {
            if (!c.active) continue;
            if (c.out.len > c.out_sent) flushOut(t, c, ci);
            // A reader paused on a full write buffer resumes here, after the
            // flush that made room, rather than from inside `flushOut`: the
            // parse ends in a flush of its own, and the two calling each
            // other would recurse once per frame of buffered input. The
            // flush comes first so a flush that emptied the buffer this
            // iteration is acted on now, not one epoll timeout later.
            if (c.active and c.paused and c.out.available() >= h2.OutBuf.control_reserve) {
                c.paused = false;
                parseInput(server, t, c, ci);
            }
            // A live connection always has some epoll interest: unpaused
            // means IN, and paused implies a partial flush left OUT armed.
            std.debug.assert(!c.active or c.closing or !c.paused or c.out.len > c.out_sent);
            // The peer said GOAWAY and everything it was owed is on the wire.
            if (c.active and !c.closing and c.h2c.goaway_received and drained(c)) closeConn(t, c, ci);
        }
    }
}

/// Nothing left to deliver: no request in flight, no response parked, and
/// the write buffer fully flushed. Only then may a GOAWAY'd connection close
/// without dropping a response the peer is waiting for.
fn drained(c: *const Connection) bool {
    if (c.inflight.load(.acquire) != 0) return false;
    if (c.out.len > c.out_sent) return false;
    for (c.scratch[0..c.h2c.usable_streams]) |*sc| {
        if (sc.resp.stage != .none) return false;
    }
    return true;
}

fn acceptLoop(server: *Server, t: *IoThread) void {
    while (true) {
        const maybe_fd = sys.accept4(t.listen_fd, linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC) catch return;
        const fd = maybe_fd orelse return;

        var slot: ?usize = null;
        for (t.conns, 0..) |*c, i| {
            if (!c.active) {
                slot = i;
                break;
            }
        }
        const ci = slot orelse {
            // No capacity. Closing immediately is honest; accepting and then
            // stalling would present to the client as a timeout.
            //
            // It is honest but not *legible*, the close lands before the
            // HTTP/2 preface, so there is no stream to carry a status on and
            // the client can only report a transport failure. The counter is
            // the compensation: `refused > 0` in the shutdown stats is the
            // difference between "the server is broken" and "raise
            // --connections".
            t.stats.connections_refused += 1;
            sys.close(fd);
            continue;
        };

        const c = &t.conns[ci];
        setNoDelay(fd);
        c.fd = fd;
        c.active = true;
        c.in_len = 0;
        c.out = h2.OutBuf.init(c.out_backing);
        c.out_sent = 0;
        c.h2c = h2.Connection.init(c.stream_bufs[0..server.config.streams_per_conn]);
        // A frame larger than the read buffer can never be assembled: the read
        // loop fills the buffer, the parser asks for more, and `compact` has
        // nothing to remove, the connection stalls silently and forever.
        // SETTINGS exists to prevent exactly this, so advertise what we can
        // actually hold rather than the protocol maximum.
        //
        // RFC 9113 §6.5.2 floors SETTINGS_MAX_FRAME_SIZE at 16384, so a read
        // buffer smaller than that is a configuration error rather than
        // something to negotiate around.
        c.h2c.local.max_frame_size = advertisedMaxFrame(server.config.read_buffer);
        c.inflight = .{ .raw = 0 };
        c.paused = false;
        for (c.scratch) |*sc| sc.resp = .{};

        sys.epollAdd(t.epfd, fd, linux.EPOLL.IN, @intCast(ci)) catch {
            sys.close(fd);
            c.active = false;
            continue;
        };
        t.stats.accepted += 1;
    }
}

fn closeConn(t: *IoThread, c: *Connection, ci: usize) void {
    _ = ci;
    if (!c.active) return;

    // A worker may still hold pointers into this connection's buffers. Closing
    // underneath it would be a use-after-free, so the socket is dropped from
    // epoll and the slot is marked `closing`; `drainCompletions` performs the
    // real close once the last completion lands.
    //
    // Marking is what makes the deferral safe to *finish*. An earlier version
    // returned early here without recording anything, so a connection closed
    // with work in flight kept its slot and its file descriptor forever, a
    // leak of both, and with `connections_per_io` slots the server eventually
    // stopped accepting.
    sys.epollDel(t.epfd, c.fd);
    if (c.inflight.load(.acquire) != 0) {
        c.closing = true;
        return;
    }
    finishClose(c);
}

/// Release the fd and the slot. Only ever called with `inflight == 0`.
fn finishClose(c: *Connection) void {
    if (c.fd >= 0) sys.close(c.fd);
    c.fd = -1;
    c.active = false;
    c.closing = false;
    c.out.reset();
    c.out_sent = 0;
    c.in_len = 0;
}

fn readConn(server: *Server, t: *IoThread, c: *Connection, ci: usize) void {
    // While paused the kernel keeps the bytes; EPOLLIN interest is off, so
    // this is only reached from a wakeup that also carried EPOLLOUT.
    if (c.paused) return;
    while (true) {
        if (c.in_len == c.in.len) break; // buffer full; drain by parsing first
        const got = sys.read(c.fd, c.in[c.in_len..]) catch {
            closeConn(t, c, ci);
            return;
        } orelse break;
        if (got == 0) {
            closeConn(t, c, ci);
            return;
        }
        c.in_len += got;
        t.stats.bytes_in += got;
    }
    parseInput(server, t, c, ci);
}

/// Parse whatever is buffered in `c.in`, dispatching requests and writing
/// control frames. Stops early, leaving the rest buffered and the connection
/// `paused`, when the write buffer can no longer take a control frame.
fn parseInput(server: *Server, t: *IoThread, c: *Connection, ci: usize) void {
    var pos: usize = 0;
    if (!c.h2c.preface_seen) {
        const consumed = c.h2c.consumePreface(c.in[0..c.in_len]) catch {
            closeConn(t, c, ci);
            return;
        } orelse {
            compact(c, 0);
            return;
        };
        pos += consumed;
        c.h2c.sendInitialFrames(&c.out) catch {};
    }

    while (pos < c.in_len) {
        if (c.out.available() < h2.OutBuf.control_reserve) {
            // Flush and, if that emptied the buffer, keep parsing: leaving it
            // for the end-of-iteration sweep costs an epoll timeout when the
            // sweep has already visited this slot. If the socket is full the
            // flush arms OUT and the sweep resumes us after the write.
            c.paused = true;
            t.stats.pauses += 1;
            compact(c, pos);
            pos = 0;
            resumePending(c);
            flushOut(t, c, ci);
            if (!c.active or c.out.len > c.out_sent) return;
            c.paused = false;
            continue;
        }
        const r = c.h2c.readFrame(c.in[pos..c.in_len], &c.out) catch |e| {
            const code: h2.ErrorCode = switch (e) {
                h2.Error.FlowControlError => .flow_control_error,
                h2.Error.FrameSizeError => .frame_size_error,
                h2.Error.CompressionError => .compression_error,
                else => .protocol_error,
            };
            c.h2c.sendGoaway(&c.out, code) catch {};
            flushOut(t, c, ci);
            closeConn(t, c, ci);
            return;
        };
        if (r.consumed == 0) break; // need more bytes
        pos += r.consumed;

        switch (r.event) {
            .request => |idx| dispatch(server, t, c, idx),
            .request_too_large => |idx| {
                // Stream-level, so the connection and its other streams survive.
                // gRPC's own name for this is RESOURCE_EXHAUSTED, and the
                // message states the limit so a client can pick a batch size
                // rather than bisect one. Reported on the first frame that
                // overflows, while the client may still be uploading; the
                // stream's slot is freed when the error has been written and
                // the rest of the body is drained by the frame reader.
                const sc = &c.scratch[idx];
                const m = std.fmt.bufPrint(
                    sc.response,
                    "request body exceeds the {d} KiB per-stream limit; send smaller batches",
                    .{server.config.request_buffer / 1024},
                ) catch "request body too large";
                queueResponse(c, idx, .resource_exhausted, m, "");
            },
            .reset => |idx| {
                // A response in progress for the slot belongs to this thread
                // (its completion has already drained, or it never had one):
                // drop it and free the slot. Otherwise the slot is either
                // held for a completion still to come, which frees it, or
                // was never dispatched and the frame reader freed it.
                if (c.scratch[idx].resp.stage != .none) {
                    c.scratch[idx].resp = .{};
                    c.h2c.closeStream(idx);
                }
            },
            // `.goaway`: RFC 9113 §6.8, the peer still expects responses on
            // the streams it already opened. Closing here dropped every
            // in-flight completion. The frame reader refuses new streams from
            // now on; the connection is closed by `ioMain` once the in-flight
            // requests have completed and their responses have been written
            // (see `drained`). Parsing continues so the WINDOW_UPDATEs those
            // responses may need still arrive.
            //
            // Nothing to do for any of these, and spelled out rather than left
            // to an `else`: `Event` is a closed union declared in this repo, so
            // a new variant should stop the build here and be decided on, not
            // be silently dropped by a connection loop.
            .goaway, .need_more, .progress => {},
        }
    }
    compact(c, pos);
    // A WINDOW_UPDATE may have opened room for a parked body.
    resumePending(c);
    flushOut(t, c, ci);
}

fn compact(c: *Connection, consumed: usize) void {
    if (consumed == 0) return;
    const rest = c.in_len - consumed;
    if (rest > 0) std.mem.copyForwards(u8, c.in[0..rest], c.in[consumed..c.in_len]);
    c.in_len = rest;
}

fn dispatch(server: *Server, t: *IoThread, c: *Connection, stream_idx: usize) void {
    const s = &c.h2c.streams[stream_idx];
    const req = Request{
        .conn = c,
        .stream_idx = stream_idx,
        .stream_id = s.id,
        .path = s.path(),
        .body = s.bodyBytes(),
        // Arrival, for `time`: see `Request.started_ns`.
        .started_ns = nowNs(),
    };

    _ = c.inflight.fetchAdd(1, .acq_rel);
    if (!server.submit.push(req)) {
        // §6.1 requires the I/O thread never execute the query, so a full
        // queue is shed rather than run inline: RESOURCE_EXHAUSTED is a signal
        // the worker pool is the bottleneck, which is a finding. Running it
        // here would serialise the benchmark and hide that.
        _ = c.inflight.fetchSub(1, .acq_rel);
        if (server.submit.isClosed()) {
            // Shutting down: the workers are gone or going. Answer rather
            // than strand the request, which would otherwise sit unanswered
            // and be counted under `last_stream_id` in the GOAWAY.
            queueResponse(c, stream_idx, .unavailable, "server shutting down", "");
            return;
        }
        t.stats.submit_full += 1;
        queueResponse(c, stream_idx, .resource_exhausted, "worker queue full", "");
        return;
    }
    t.stats.requests += 1;
}

/// Take ownership of a response for stream slot `idx` and write as much of
/// it as fits now. Whatever remains is picked up by `resumePending`.
fn queueResponse(c: *Connection, idx: usize, status: grpc.Status, message: []const u8, body: []const u8) void {
    const pr = &c.scratch[idx].resp;
    std.debug.assert(pr.stage == .none);
    pr.* = .{ .stage = .headers, .status = status, .message = message, .body = body };
    pumpResponse(c, idx);
}

/// Advance a pending response as far as the write buffer and the send
/// windows allow, freeing the stream slot once the trailers are out.
///
/// Each stage starts only with `OutBuf.reserve` free, and each writes less
/// than that, so the buffer never drops below the control-frame headroom the
/// frame reader needs by more than one small frame.
fn pumpResponse(c: *Connection, idx: usize) void {
    const pr = &c.scratch[idx].resp;
    const sid = c.h2c.streams[idx].id;
    while (true) switch (pr.stage) {
        .none => return,
        .headers => {
            if (c.out.available() < h2.OutBuf.reserve) return;
            if (pr.status != .ok) {
                // Trailers-only. Cannot fail: the message is bounded to 256
                // encoded bytes and the headroom was just checked.
                grpc.writeError(&c.out, sid, pr.status, pr.message) catch unreachable;
                pr.stage = .none;
                c.h2c.closeStream(idx);
                return;
            }
            grpc.writeResponseHeaders(&c.out, sid) catch unreachable;
            pr.stage = .data;
        },
        .data => {
            pr.off = grpc.writeData(&c.h2c, &c.out, idx, pr.body, pr.off);
            if (pr.off < grpc.prefix_len + pr.body.len) return; // parked
            pr.stage = .trailers;
        },
        .trailers => {
            if (c.out.available() < h2.OutBuf.reserve) return;
            grpc.writeTrailers(&c.out, sid, .ok, "") catch unreachable;
            pr.stage = .none;
            c.h2c.closeStream(idx);
            return;
        },
    };
}

/// Resume every parked response on the connection. Called after input has
/// been parsed (a WINDOW_UPDATE may have arrived) and after the write buffer
/// has been flushed (there is room again).
fn resumePending(c: *Connection) void {
    for (c.scratch[0..c.h2c.usable_streams], 0..) |*sc, i| {
        if (sc.resp.stage != .none) pumpResponse(c, i);
    }
}

fn drainCompletions(t: *IoThread) void {
    while (t.completions.tryPop()) |cm| {
        const c = cm.conn;
        const remaining = c.inflight.fetchSub(1, .acq_rel) - 1;

        // A close that was deferred because this request was outstanding can
        // now complete. Nothing else revisits that decision, so if this is
        // skipped the slot is never reclaimed.
        if (c.closing) {
            if (remaining == 0) finishClose(c);
            continue;
        }
        if (!c.active) continue;

        // The slot was held for this completion, so it still names this
        // stream, whether or not the peer reset it in the meantime.
        const s = &c.h2c.streams[cm.stream_idx];
        std.debug.assert(s.dispatched and s.id == cm.stream_id);
        if (s.state == .closed) {
            // Reset by the peer while the worker ran: the response is
            // dropped and the slot, kept out of reuse until now, is freed.
            c.h2c.closeStream(cm.stream_idx);
            continue;
        }
        queueResponse(c, cm.stream_idx, cm.status, cm.message, cm.body);
    }
}

/// Flush pending output, tolerating partial writes.
///
/// §6.1: "coalesce all completed responses for a connection into a single
/// `writev` per epoll wakeup." One contiguous buffer written with a single
/// `write` achieves the same syscall count without iovec bookkeeping; the
/// difference would only matter if the response bodies were not already being
/// built in place.
fn flushOut(t: *IoThread, c: *Connection, ci: usize) void {
    while (true) {
        while (c.out_sent < c.out.len) {
            const wrote = sys.write(c.fd, c.out.written()[c.out_sent..]) catch {
                closeConn(t, c, ci);
                return;
            } orelse {
                // Socket buffer full: wait for writability rather than spinning.
                sys.epollMod(t.epfd, c.fd, wantEvents(c, true), @intCast(ci)) catch {};
                return;
            };
            c.out_sent += wrote;
            t.stats.bytes_out += wrote;
        }
        c.out.reset();
        c.out_sent = 0;
        // The buffer is empty again: parked responses can make progress.
        // Their frames go out now rather than on the next flush, which for a
        // paused reader with nothing else pending is one epoll timeout away.
        resumePending(c);
        if (c.out.len == 0) break;
    }
    sys.epollMod(t.epfd, c.fd, wantEvents(c, false), @intCast(ci)) catch {};
}

/// epoll interest for a connection. Reading is suspended while the reader is
/// paused on a full write buffer, otherwise a level-triggered EPOLLIN on the
/// unread bytes would spin the loop.
fn wantEvents(c: *const Connection, want_out: bool) u32 {
    var ev: u32 = 0;
    if (!c.paused) ev |= linux.EPOLL.IN;
    if (want_out) ev |= linux.EPOLL.OUT;
    return ev;
}

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;
const hpack = @import("hpack.zig");

test "queue refuses pushes once closed, and drains what was pushed before" {
    var q: Queue(u32, 4) = .{};
    try testing.expect(q.push(1));
    q.close();
    // Refused, so `dispatch` answers the request instead of stranding it in
    // a queue no worker will drain again.
    try testing.expect(!q.push(2));
    try testing.expect(q.isClosed());
    // What was in flight before the close is still handed out.
    try testing.expectEqual(@as(u32, 1), q.pop().?);
    try testing.expectEqual(@as(?u32, null), q.pop());
}

test "queue is FIFO and reports fullness rather than blocking the producer" {
    var q: Queue(u32, 4) = .{};
    try testing.expect(q.push(1));
    try testing.expect(q.push(2));
    try testing.expect(q.push(3));
    try testing.expect(q.push(4));
    // Full: the producer must learn this rather than block, so the I/O thread
    // can shed load with RESOURCE_EXHAUSTED instead of stalling the event loop.
    try testing.expect(!q.push(5));

    try testing.expectEqual(@as(u32, 1), q.tryPop().?);
    try testing.expectEqual(@as(u32, 2), q.tryPop().?);
    try testing.expect(q.push(5));
    try testing.expectEqual(@as(u32, 3), q.tryPop().?);
    try testing.expectEqual(@as(u32, 4), q.tryPop().?);
    try testing.expectEqual(@as(u32, 5), q.tryPop().?);
    try testing.expectEqual(@as(?u32, null), q.tryPop());
}

test "queue wraps correctly around its ring" {
    var q: Queue(u32, 3) = .{};
    for (0..10) |i| {
        try testing.expect(q.push(@intCast(i)));
        try testing.expectEqual(@as(u32, @intCast(i)), q.tryPop().?);
    }
    try testing.expectEqual(@as(?u32, null), q.tryPop());
}

test "queue pop unblocks on close so workers can exit" {
    var q: Queue(u32, 4) = .{};
    const T = struct {
        fn run(qq: *Queue(u32, 4), got: *?u32) void {
            got.* = qq.pop();
        }
    };
    var got: ?u32 = 0;
    var th = try std.Thread.spawn(.{}, T.run, .{ &q, &got });
    // Closing an empty queue must wake the waiter with null rather than
    // leaving the worker parked forever at shutdown.
    q.close();
    th.join();
    try testing.expectEqual(@as(?u32, null), got);
}

test "queue is safe under concurrent producers" {
    var q: Queue(u32, 256) = .{};
    const producers = 4;
    const per = 500;

    const T = struct {
        fn produce(qq: *Queue(u32, 256), n: usize) void {
            var pushed: usize = 0;
            while (pushed < n) {
                if (qq.push(1)) pushed += 1 else std.Thread.yield() catch {};
            }
        }
    };

    var threads: [producers]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, T.produce, .{ &q, per });

    var consumed: usize = 0;
    while (consumed < producers * per) {
        if (q.tryPop()) |_| consumed += 1 else std.Thread.yield() catch {};
    }
    for (threads) |t| t.join();
    try testing.expectEqual(producers * per, consumed);
    try testing.expectEqual(@as(?u32, null), q.tryPop());
}

test "SO_REUSEPORT lets two sockets bind the same port" {
    // The property §6.3 depends on: without it the second bind fails with
    // EADDRINUSE and the multi-I/O-thread design cannot work.
    const a = listenReusePort(Address.init(Address.loopback, 0)) catch return error.SkipZigTest;
    defer sys.close(a);

    const port = try boundPort(a);
    try testing.expect(port != 0);

    const b = try listenReusePort(Address.init(Address.loopback, port));
    defer sys.close(b);
    try testing.expectEqual(port, try boundPort(b));
}

test "Address parses dotted-quad and rejects everything else" {
    const a = Address.parseIp4("127.0.0.1", 6334).?;
    try testing.expectEqual(@as(u32, 0x7f000001), a.ip);
    try testing.expectEqual(@as(u16, 6334), a.port);

    try testing.expectEqual(@as(u32, 0), Address.parseIp4("0.0.0.0", 1).?.ip);
    try testing.expectEqual(@as(u32, 0xc0a80101), Address.parseIp4("192.168.1.1", 1).?.ip);

    try testing.expectEqual(@as(?Address, null), Address.parseIp4("1.2.3", 1));
    try testing.expectEqual(@as(?Address, null), Address.parseIp4("1.2.3.4.5", 1));
    try testing.expectEqual(@as(?Address, null), Address.parseIp4("1.2.3.256", 1));
    try testing.expectEqual(@as(?Address, null), Address.parseIp4("localhost", 1));
}

test "ResponseBuf hands out non-overlapping regions" {
    var backing: [64]u8 = undefined;
    var rb = ResponseBuf{ .buf = &backing };

    const a = rb.available();
    @memcpy(a[0..3], "abc");
    const first = rb.commit(3);
    try testing.expectEqualStrings("abc", first);

    const b = rb.available();
    @memcpy(b[0..2], "de");
    const second = rb.commit(2);
    try testing.expectEqualStrings("de", second);
    // The first slice must still be intact, a handler holds it until the
    // completion is written.
    try testing.expectEqualStrings("abc", first);
    try testing.expectEqual(@as(usize, 5), rb.len);

    rb.reset();
    try testing.expectEqual(@as(usize, 0), rb.len);
}

test "pinToCpu does not fault on a plausible cpu index" {
    // Behavioural check only: affinity may legitimately be denied in a
    // container, and this must not be fatal there.
    pinToCpu(0);
}

// -------------------------------------------------------------------------
// Socket-level tests: a real server on a loopback port and a raw h2 client
// small enough to live here, so connection-lifecycle behaviour is pinned
// next to the code rather than only in the API-level suite.
// -------------------------------------------------------------------------

/// A blocking h2c client that speaks just enough to open a stream and read
/// frames back, with a receive timeout so a server that stops answering is a
/// test failure rather than a hang.
const RawClient = struct {
    fd: linux.fd_t,
    buf: [64 * 1024]u8 = undefined,
    len: usize = 0,
    /// Bytes of the frame handed out by the last `readFrame`, dropped on
    /// the next.
    consumed: usize = 0,

    fn connect(port: u16) !RawClient {
        const fd = try sys.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, linux.IPPROTO.TCP);
        errdefer sys.close(fd);
        var tv = linux.timeval{ .sec = 5, .usec = 0 };
        _ = linux.setsockopt(fd, linux.SOL.SOCKET, linux.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(linux.timeval));
        setNoDelay(fd);
        const sa = Address.init(Address.loopback, port).sockaddrIn();
        if (linux.errno(linux.connect(fd, @ptrCast(&sa), @sizeOf(linux.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
        var c = RawClient{ .fd = fd };
        try c.writeAll(h2.client_preface);
        var ob: [64]u8 = undefined;
        var out = h2.OutBuf.init(&ob);
        try out.frame(.settings, 0, 0, &.{});
        try c.writeAll(out.written());
        return c;
    }

    fn close(self: *RawClient) void {
        sys.close(self.fd);
    }

    fn writeAll(self: *RawClient, bytes: []const u8) !void {
        var sent: usize = 0;
        while (sent < bytes.len) {
            const n = try sys.write(self.fd, bytes[sent..]) orelse continue;
            if (n == 0) return error.Closed;
            sent += n;
        }
    }

    /// HEADERS with END_STREAM: a bodyless request for `path` on `sid`.
    fn sendRequest(self: *RawClient, sid: u31, path: []const u8) !void {
        var hb: [256]u8 = undefined;
        var hn: usize = 0;
        hn += try hpack.Encoder.writeHeader(hb[hn..], ":method", "POST");
        hn += try hpack.Encoder.writeHeader(hb[hn..], ":path", path);
        var ob: [512]u8 = undefined;
        var out = h2.OutBuf.init(&ob);
        try out.frame(.headers, h2.Flags.end_headers | h2.Flags.end_stream, sid, hb[0..hn]);
        try self.writeAll(out.written());
    }

    fn sendGoaway(self: *RawClient, last: u31) !void {
        var p: [8]u8 = undefined;
        std.mem.writeInt(u32, p[0..4], last, .big);
        std.mem.writeInt(u32, p[4..8], @intFromEnum(h2.ErrorCode.no_error), .big);
        var ob: [64]u8 = undefined;
        var out = h2.OutBuf.init(&ob);
        try out.frame(.goaway, 0, 0, &p);
        try self.writeAll(out.written());
    }

    const Frame = struct { header: h2.FrameHeader, payload: []const u8 };

    /// The next frame, or null on a clean EOF from the server. Errors on the
    /// receive timeout. The returned view into `buf` is valid until the next
    /// call, which is when the frame is compacted away.
    fn readFrame(self: *RawClient) !?Frame {
        if (self.consumed > 0) {
            const rest = self.len - self.consumed;
            std.mem.copyForwards(u8, self.buf[0..rest], self.buf[self.consumed..self.len]);
            self.len = rest;
            self.consumed = 0;
        }
        while (true) {
            if (self.len >= h2.frame_header_len) {
                const hd = h2.FrameHeader.parse(self.buf[0..h2.frame_header_len]);
                const total = h2.frame_header_len + hd.length;
                if (self.len >= total) {
                    self.consumed = total;
                    return .{ .header = hd, .payload = self.buf[h2.frame_header_len..total] };
                }
            }
            const got = sys.read(self.fd, self.buf[self.len..]) catch return error.ReadFailed;
            const n = got orelse return error.Timeout;
            if (n == 0) return null;
            self.len += n;
        }
    }

    /// The next frame on `sid` of the given type, skipping SETTINGS acks,
    /// WINDOW_UPDATEs and anything else on other streams.
    fn expectFrame(self: *RawClient, sid: u31, t: h2.FrameType) !Frame {
        while (true) {
            const f = (try self.readFrame()) orelse return error.UnexpectedEof;
            if (f.header.stream_id == sid and f.header.frame_type == t) return f;
        }
    }
};

/// A handler answering `ok` with an *empty* body, the shape of
/// `CollectionOperationResponse{result=false}` (proto3 omits a false bool).
fn emptyOkDispatch(_: *anyopaque, req: *const Request, _: *ResponseBuf) Completion {
    return .{ .conn = req.conn, .stream_idx = req.stream_idx, .stream_id = req.stream_id, .status = .ok };
}

/// The same after a pause, so a GOAWAY can land while the request is in
/// flight.
fn slowEmptyOkDispatch(ctx: *anyopaque, req: *const Request, out: *ResponseBuf) Completion {
    var ts = linux.timespec{ .sec = 0, .nsec = 50 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    return emptyOkDispatch(ctx, req, out);
}

fn startTestServer(handler: HandlerFn) !struct { srv: *Server, port: u16 } {
    var dummy: u8 = 0;
    const srv = try Server.init(testing.allocator, .{
        .addr = Address.init(Address.loopback, 0),
        .io_threads = 1,
        .worker_threads = 1,
        .streams_per_conn = 4,
        .connections_per_io = 2,
        .request_buffer = 4096,
        .response_buffer = 4096,
        .write_buffer = 64 * 1024,
    }, handler, @ptrCast(&dummy));
    errdefer srv.deinit();
    const port = try srv.bind();
    try srv.start();
    return .{ .srv = srv, .port = port };
}

test "time is appended even when the response body encodes to zero bytes" {
    // `CollectionOperationResponse{result=false}` (a delete of a missing
    // collection) has an empty body, and `time` used to be skipped for it,
    // so the client decoded `time = 0.0`. Qdrant always sends it.
    const s = try startTestServer(emptyOkDispatch);
    defer {
        s.srv.stop();
        s.srv.deinit();
    }
    var cl = try RawClient.connect(s.port);
    defer cl.close();

    try cl.sendRequest(1, "/qdrant.Collections/Delete");
    _ = try cl.expectFrame(1, .headers);
    const data = try cl.expectFrame(1, .data);
    // 5-byte gRPC prefix, then exactly the `time` field: tag (2 << 3 | 1) and
    // a little-endian f64.
    try testing.expectEqual(@as(usize, grpc.prefix_len + 9), data.payload.len);
    try testing.expectEqual(@as(u8, 0x11), data.payload[grpc.prefix_len]);
    const t: f64 = @bitCast(std.mem.readInt(u64, data.payload[grpc.prefix_len + 1 ..][0..8], .little));
    try testing.expect(t >= 0 and t < 5);
    const trailers = try cl.expectFrame(1, .headers);
    try testing.expect(trailers.header.hasFlag(h2.Flags.end_stream));
}

test "a client GOAWAY still gets its in-flight response, and new streams are refused" {
    // RFC 9113 §6.8: the GOAWAY sender expects responses on the streams it
    // already opened. The server used to close the connection on the spot
    // and drop the completion; the client saw EOF and no response.
    const s = try startTestServer(slowEmptyOkDispatch);
    defer {
        s.srv.stop();
        s.srv.deinit();
    }
    var cl = try RawClient.connect(s.port);
    defer cl.close();

    try cl.sendRequest(1, "/a");
    try cl.sendGoaway(1);
    // A stream opened after our own GOAWAY: refused, not served.
    try cl.sendRequest(3, "/b");

    // Read to EOF: stream 3's RST arrives first (sent while parsing), then
    // stream 1's full response once the worker is done, then the server
    // closes.
    var saw_rst3 = false;
    var frames1: usize = 0;
    var ended1 = false;
    while (try cl.readFrame()) |f| {
        if (f.header.stream_id == 3) {
            try testing.expectEqual(h2.FrameType.rst_stream, f.header.frame_type);
            try testing.expectEqual(@intFromEnum(h2.ErrorCode.refused_stream), std.mem.readInt(u32, f.payload[0..4], .big));
            saw_rst3 = true;
        } else if (f.header.stream_id == 1) {
            frames1 += 1;
            switch (frames1) {
                1 => try testing.expectEqual(h2.FrameType.headers, f.header.frame_type),
                2 => {
                    try testing.expectEqual(h2.FrameType.data, f.header.frame_type);
                    try testing.expectEqual(@as(usize, grpc.prefix_len + 9), f.payload.len);
                },
                3 => {
                    try testing.expectEqual(h2.FrameType.headers, f.header.frame_type);
                    try testing.expect(f.header.hasFlag(h2.Flags.end_stream));
                    ended1 = true;
                },
                else => return error.TooManyFrames,
            }
        }
    }
    try testing.expect(saw_rst3);
    try testing.expect(ended1);
}
