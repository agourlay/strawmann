//! The two thread fan-outs the engine repeats.
//!
//! `runTasks` gives each of a fixed set of tasks a thread, the caller's
//! included, and runs what could not get one on the caller: a spawn failure
//! costs wall clock, never a task. `spawnWorkers` starts identical workers
//! that share their work through the caller's state (an atomic cursor, say),
//! and reports how many started, since what none starting means is the
//! caller's to decide.

const std = @import("std");

/// Run `run` on every task: task 0 on this thread, the others each on a
/// thread of its own. What a failed spawn left without a thread runs here,
/// after task 0. Returns once every task is done.
pub fn runTasks(comptime T: type, alloc: std.mem.Allocator, tasks: []T, comptime run: fn (*T) void) error{OutOfMemory}!void {
    if (tasks.len == 1) {
        run(&tasks[0]);
        return;
    }
    const handles = try alloc.alloc(std.Thread, tasks.len - 1);
    defer alloc.free(handles);
    var spawned: usize = 0;
    for (tasks[1..], 0..) |*task, i| {
        handles[i] = std.Thread.spawn(.{}, run, .{task}) catch break;
        spawned += 1;
    }
    run(&tasks[0]);
    for (tasks[1 + spawned ..]) |*task| run(task);
    for (handles[0..spawned]) |h| h.join();
}

/// Start up to `n` threads running `func` with `args`, stopping at the first
/// that fails to spawn, and join them. Returns how many ran.
pub fn spawnWorkers(alloc: std.mem.Allocator, n: usize, comptime func: anytype, args: anytype) error{OutOfMemory}!usize {
    const handles = try alloc.alloc(std.Thread, n);
    defer alloc.free(handles);
    var spawned: usize = 0;
    for (handles) |*h| {
        h.* = std.Thread.spawn(.{}, func, args) catch break;
        spawned += 1;
    }
    for (handles[0..spawned]) |h| h.join();
    return spawned;
}

const testing = std.testing;

test "runTasks runs every task once, on any number of tasks" {
    const T = struct {
        hits: usize = 0,
        fn run(self: *@This()) void {
            self.hits += 1;
        }
    };
    for ([_]usize{ 1, 2, 7 }) |n| {
        var tasks: [7]T = @splat(.{});
        try runTasks(T, testing.allocator, tasks[0..n], T.run);
        for (tasks[0..n]) |t| try testing.expectEqual(@as(usize, 1), t.hits);
    }
}

test "spawnWorkers shares an atomic cursor and reports the threads it ran" {
    const Job = struct {
        next: std.atomic.Value(usize) = .init(0),
        done: [100]std.atomic.Value(u8) = @splat(.init(0)),
        fn work(self: *@This()) void {
            while (true) {
                const i = self.next.fetchAdd(1, .monotonic);
                if (i >= self.done.len) return;
                _ = self.done[i].fetchAdd(1, .monotonic);
            }
        }
    };
    var job: Job = .{};
    try testing.expectEqual(@as(usize, 4), try spawnWorkers(testing.allocator, 4, Job.work, .{&job}));
    for (&job.done) |*d| try testing.expectEqual(@as(u8, 1), d.load(.monotonic));
}
