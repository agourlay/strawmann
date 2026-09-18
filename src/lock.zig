//! A futex-backed mutex, shared by the transport and the storage layers.
//!
//! Zig 0.16 moved `Mutex`/`Condition` under `std.Io`, where every operation
//! takes an `Io` instance. That is the right shape for code built on the async
//! runtime, but this engine owns its own threads and its own event loop, §6.1
//! and §6.3 specify the threading model down to the CPU pinning, so threading
//! an `Io` through the submit and ingest paths would be plumbing with no
//! payoff. Two futex calls are the whole dependency.
//!
//! It lives at the top of the module rather than inside `net/` because both the
//! I/O layer (the submit queue, the collection registry) and the storage layer
//! (ingest) need it, and `core/` importing `net/` would be a layering
//! inversion.

const std = @import("std");
const linux = std.os.linux;

const wait_op: linux.FUTEX_OP = .{ .cmd = .WAIT, .private = true };
const wake_op: linux.FUTEX_OP = .{ .cmd = .WAKE, .private = true };

/// Sleep until `*addr != expect`. Spurious wakeups are possible and every
/// caller re-checks its predicate in a loop.
pub fn futexWait(addr: *const std.atomic.Value(u32), expect: u32) void {
    _ = linux.futex_4arg(&addr.raw, wait_op, expect, null);
}

pub fn futexWake(addr: *const std.atomic.Value(u32), n: u32) void {
    _ = linux.futex_3arg(&addr.raw, wake_op, n);
}

/// "Wake every waiter" for `FUTEX_WAKE`.
///
/// **`maxInt(u32)` is wrong here and the mistake is invisible in review.** The
/// kernel reads the count field as a signed `int`, so 0xffffffff arrives as
/// `-1` and wakes one waiter (or none) instead of all. `INT_MAX` is the
/// conventional value and the only portable spelling of "all".
///
/// Found by `strace` on a hung shutdown: `FUTEX_WAKE_PRIVATE, 4294967295`
/// returned 1 with two workers parked, so the queue's `close` released one
/// worker and `Thread.join` blocked forever on the other. Every RPC had already
/// completed correctly, which is why no functional test caught it.
pub const wake_all: u32 = std.math.maxInt(i32);

/// Three-state futex mutex: unlocked / locked / locked-with-waiters.
///
/// The third state is what keeps the uncontended path free of syscalls: an
/// unlock only issues `FUTEX_WAKE` when it knows someone is parked.
pub const Mutex = struct {
    state: std.atomic.Value(u32) = .init(unlocked),

    const unlocked: u32 = 0;
    const locked: u32 = 1;
    const contended: u32 = 2;

    pub fn lock(self: *Mutex) void {
        if (self.state.cmpxchgWeak(unlocked, locked, .acquire, .monotonic) == null) return;
        while (self.state.swap(contended, .acquire) != unlocked) {
            futexWait(&self.state, contended);
        }
    }

    pub fn unlock(self: *Mutex) void {
        if (self.state.swap(unlocked, .release) == contended) {
            futexWake(&self.state, 1);
        }
    }
};

const testing = std.testing;

test "mutex serialises concurrent increments" {
    // Non-vacuous: the same loop without the mutex loses updates reliably at
    // this thread count and iteration count.
    const T = struct {
        m: Mutex = .{},
        counter: u64 = 0,

        fn bump(self: *@This(), n: usize) void {
            for (0..n) |_| {
                self.m.lock();
                self.counter += 1;
                self.m.unlock();
            }
        }
    };
    var t = T{};
    const threads = 4;
    const per = 20_000;
    var handles: [threads]std.Thread = undefined;
    for (&handles) |*h| h.* = try std.Thread.spawn(.{}, T.bump, .{ &t, per });
    for (handles) |h| h.join();
    try testing.expectEqual(@as(u64, threads * per), t.counter);
}

test "wake_all is INT_MAX, not maxInt(u32)" {
    // The kernel reads the count as a signed int; 0xffffffff arrives as -1.
    try testing.expectEqual(@as(u32, 0x7fffffff), wake_all);
    try testing.expect(wake_all != std.math.maxInt(u32));
}
