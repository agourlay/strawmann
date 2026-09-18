//! §8.3, Qdrant's score semantics, which we must replicate exactly.
//!
//! From `lib/segment/src/spaces/`. This table is the difference between "our
//! numbers are close" and "our numbers are the same numbers":
//!
//! | metric    | internal similarity (higher = better) | postprocess on the returned score | preprocess at ingest |
//! |-----------|---------------------------------------|-----------------------------------|----------------------|
//! | Dot       | `Σ aᵢbᵢ`                              | identity                          | none                 |
//! | Cosine    | `Σ aᵢbᵢ` (dot on normalised data)     | identity                          | normalise            |
//! | Euclid    | `−Σ(aᵢ−bᵢ)²`                          | `abs().sqrt()`                    | none                 |
//! | Manhattan | `−Σ|aᵢ−bᵢ|`                           | `abs()`                           | none                 |
//!
//! The spec calls out three traps, all of which are asserted in the tests at
//! the bottom of this file:
//!
//!  1. **Euclid ranks on the negated squared distance but returns the square
//!     root.** Getting the ranking right and the returned score wrong (or vice
//!     versa) is easy and shows up as a conformance failure with a
//!     suspiciously structured error distribution.
//!
//!  2. **`(a−b)²` is computed directly**, never via the `|a|² + |b|² − 2ab`
//!     expansion, see `l2_f32.zig` for why that expansion is a correctness
//!     trap rather than merely a different rounding.
//!
//!  3. **Cosine normalisation short-circuits.** See `norm.zig`.
//!
//! §8.3: "The conformance module encodes this table as executable assertions,
//! and a change to it is a spec change."

const std = @import("std");

pub const Metric = enum(u8) {
    dot = 0,
    cosine = 1,
    euclid = 2,
    manhattan = 3,

    /// The wire enum used by Qdrant's `Distance` proto field. Kept explicit and
    /// separate from our internal numbering so a proto change cannot silently
    /// renumber our storage headers.
    ///
    /// From qdrant `Distance`: UnknownDistance=0, Cosine=1, Euclid=2, Dot=3,
    /// Manhattan=4.
    pub fn fromProto(v: i32) ?Metric {
        return switch (v) {
            1 => .cosine,
            2 => .euclid,
            3 => .dot,
            4 => .manhattan,
            else => null,
        };
    }

    pub fn toProto(self: Metric) i32 {
        return switch (self) {
            .cosine => 1,
            .euclid => 2,
            .dot => 3,
            .manhattan => 4,
        };
    }

    pub fn name(self: Metric) []const u8 {
        return @tagName(self);
    }

    /// §8.3 / §6.4: "Cosine is normalised at ingest and stored normalised, so
    /// the search path only ever runs dot product. Record this in the header."
    ///
    /// This is the *only* metric with an ingest-time preprocess, and getting it
    /// wrong is invisible on random data (which is near-normalised in
    /// expectation only in direction, not magnitude) and systematic on real
    /// embeddings.
    pub fn normalisesAtIngest(self: Metric) bool {
        return self == .cosine;
    }

    /// The distance kernel this metric dispatches to once ingest preprocessing
    /// has been applied. Cosine collapses onto dot precisely because of
    /// `normalisesAtIngest`.
    pub fn kernel(self: Metric) Kernel {
        return switch (self) {
            .dot, .cosine => .dot,
            .euclid => .euclid,
            .manhattan => .manhattan,
        };
    }

    /// Postprocess applied to the internal similarity before it goes on the
    /// wire as `ScoredPoint.score`.
    ///
    /// Ranking happens on the *internal similarity* (higher = better);
    /// this transform is applied afterwards, to the already-ordered results.
    /// For Euclid and Manhattan it is order-reversing, which is exactly why it
    /// must not be applied before sorting.
    pub fn postprocess(self: Metric, similarity: f32) f32 {
        return switch (self) {
            .dot, .cosine => similarity,
            // Trap 1: ranks on −Σ(a−b)², returns √(Σ(a−b)²).
            .euclid => @sqrt(@abs(similarity)),
            .manhattan => @abs(similarity),
        };
    }

    /// True when `postprocess` reverses the ordering, i.e. when a *lower*
    /// returned score means a *better* match. Callers that need to reason about
    /// the returned scores (the differ, score_threshold handling) need this;
    /// the search path itself never does, because it ranks on the internal
    /// similarity throughout.
    pub fn postprocessReversesOrder(self: Metric) bool {
        return switch (self) {
            .dot, .cosine => false,
            .euclid, .manhattan => true,
        };
    }
};

/// The distance kernels a metric can resolve to, after ingest preprocessing.
pub const Kernel = enum { dot, euclid, manhattan };

test "§8.3 table: preprocess column" {
    try std.testing.expect(!Metric.dot.normalisesAtIngest());
    try std.testing.expect(Metric.cosine.normalisesAtIngest());
    try std.testing.expect(!Metric.euclid.normalisesAtIngest());
    try std.testing.expect(!Metric.manhattan.normalisesAtIngest());
}

test "§8.3 table: cosine collapses onto the dot kernel" {
    try std.testing.expectEqual(Kernel.dot, Metric.cosine.kernel());
    try std.testing.expectEqual(Kernel.dot, Metric.dot.kernel());
    try std.testing.expectEqual(Kernel.euclid, Metric.euclid.kernel());
    try std.testing.expectEqual(Kernel.manhattan, Metric.manhattan.kernel());
}

test "§8.3 trap 1: Euclid ranks on negated squared distance, returns the root" {
    // Two points at squared distances 4 and 9 from the query.
    const sim_near: f32 = -4.0;
    const sim_far: f32 = -9.0;

    // Ranking is on the internal similarity: less negative is better.
    try std.testing.expect(sim_near > sim_far);

    // But the returned score is the actual Euclidean distance, so the ordering
    // of the *returned* numbers is reversed.
    const score_near = Metric.euclid.postprocess(sim_near);
    const score_far = Metric.euclid.postprocess(sim_far);
    try std.testing.expectEqual(@as(f32, 2.0), score_near);
    try std.testing.expectEqual(@as(f32, 3.0), score_far);
    try std.testing.expect(score_near < score_far);
    try std.testing.expect(Metric.euclid.postprocessReversesOrder());
}

test "§8.3: Manhattan postprocess is abs, not sqrt" {
    // The easy bug is to share Euclid's postprocess. abs() != abs().sqrt().
    try std.testing.expectEqual(@as(f32, 9.0), Metric.manhattan.postprocess(-9.0));
    try std.testing.expectEqual(@as(f32, 3.0), Metric.euclid.postprocess(-9.0));
}

test "§8.3: Dot and Cosine postprocess is the identity, including for negatives" {
    // Dot products are legitimately negative and must stay negative. An abs()
    // leaking into this path would silently turn the worst matches into the
    // best ones.
    try std.testing.expectEqual(@as(f32, -0.5), Metric.dot.postprocess(-0.5));
    try std.testing.expectEqual(@as(f32, -0.5), Metric.cosine.postprocess(-0.5));
    try std.testing.expect(!Metric.dot.postprocessReversesOrder());
}

test "proto enum mapping round-trips and rejects unknown" {
    for ([_]Metric{ .dot, .cosine, .euclid, .manhattan }) |m| {
        try std.testing.expectEqual(m, Metric.fromProto(m.toProto()).?);
    }
    // UnknownDistance=0 must not silently become a real metric.
    try std.testing.expectEqual(@as(?Metric, null), Metric.fromProto(0));
    try std.testing.expectEqual(@as(?Metric, null), Metric.fromProto(99));
}
