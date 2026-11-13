//! The test every kernel in `dist/` has: each instantiation, at each dim,
//! against a reference on random inputs. Nine tests wrote the loop out by hand
//! and differed only in the kernels, the dims, the reference and how close is
//! close enough.

const std = @import("std");

/// How a kernel's result must compare with its reference's.
pub const Tolerance = union(enum) {
    /// Equal: integer kernels, where every intermediate is exact and any
    /// difference is a bug in the vector path, not rounding.
    exact,
    /// `|got - want| / max(1, |want|) < x`, against an fp64 reference: the
    /// scale grows as √d, and below magnitude 1 the error is absolute.
    scaled: f64,
    /// `std.testing.expectApproxEqRel(want, got, x)`, against a reference in
    /// the kernel's own precision.
    rel: f32,
};

/// `K.call(a, b, extra...)` against `reference(a, b, extra...)` for every
/// kernel type in `kernels` and every dim in `dims`.
///
/// Inputs are drawn from one generator seeded with `seed`, kernels outer and
/// dims inner, `a[i]` then `b[i]` element by element: floats from a standard
/// normal (narrowed for f16), integers uniform over their type.
pub fn expectMatches(
    comptime A: type,
    comptime B: type,
    comptime kernels: anytype,
    reference: anytype,
    comptime tol: Tolerance,
    seed: u64,
    dims: []const usize,
    extra: anytype,
) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    inline for (kernels) |K| {
        for (dims) |d| {
            const a = try std.testing.allocator.alloc(A, d);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(B, d);
            defer std.testing.allocator.free(b);
            for (a, b) |*x, *y| {
                x.* = draw(A, rnd);
                y.* = draw(B, rnd);
            }
            const got = @call(.auto, K.call, .{ a, b } ++ extra);
            const want = @call(.auto, reference, .{ a, b } ++ extra);
            switch (tol) {
                .exact => try std.testing.expectEqual(want, got),
                .scaled => |x| {
                    const scale = @max(1.0, @abs(want));
                    try std.testing.expect(@abs(@as(f64, got) - want) / scale < x);
                },
                .rel => |x| try std.testing.expectApproxEqRel(want, got, x),
            }
        }
    }
}

fn draw(comptime T: type, rnd: std.Random) T {
    return switch (@typeInfo(T)) {
        .float => @floatCast(rnd.floatNorm(f32)),
        .int => rnd.int(T),
        else => @compileError("no generator for " ++ @typeName(T)),
    };
}
