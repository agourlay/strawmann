//! Helpers the unit tests share: random fills and recall overlaps that were
//! written out by hand in every test that needed one. Referenced only from
//! test code, so nothing here is compiled into a build that ships.
//!
//! Every helper that draws from a `std.Random` draws the same values in the
//! same order as the loops it replaced, because the recall thresholds those
//! tests assert were tuned against exactly those draws.

const std = @import("std");

/// Upsert `n` points with numeric ids `first .. first + n` into `c` (a
/// `*Collection`), each a `dim`-vector of standard normal draws taken from
/// `rnd` component by component, point by point.
pub fn upsertRandom(c: anytype, rnd: std.Random, comptime dim: usize, first: u64, n: usize) !void {
    for (0..n) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*x| x.* = rnd.floatNorm(f32);
        _ = try c.upsert(.{ .num = first + i }, &v);
    }
}

/// How many elements of `got` have an `id` that also appears in `truth`,
/// each element of `got` counted at most once.
pub fn countShared(got: anytype, truth: anytype) usize {
    var n: usize = 0;
    for (got) |g| {
        for (truth) |t| {
            if (t.id == g.id) {
                n += 1;
                break;
            }
        }
    }
    return n;
}

/// A scored row, for the exact-versus-approximate comparisons.
pub const Pair = struct { id: u32, s: f32 };

/// Sort `pairs` best first: highest score first.
pub fn sortDesc(pairs: []Pair) void {
    std.mem.sort(Pair, pairs, {}, struct {
        fn f(_: void, a: Pair, b: Pair) bool {
            return a.s > b.s;
        }
    }.f);
}

test "countShared counts each element of got once" {
    const P = struct { id: u32 };
    const got = [_]P{ .{ .id = 1 }, .{ .id = 2 }, .{ .id = 5 } };
    const truth = [_]P{ .{ .id = 2 }, .{ .id = 2 }, .{ .id = 1 } };
    try std.testing.expectEqual(@as(usize, 2), countShared(&got, &truth));
}
