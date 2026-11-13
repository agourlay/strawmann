//! Test fixtures shared by the `#[cfg(test)]` modules.
//!
//! Each builds the plain case; a test that needs one field different says so
//! with struct update syntax (`GroundTruth { n_base: 1, ..gt(..) }`), so the
//! difference is the thing the reader sees.

use crate::differ::tolerance::{Distribution, Epsilon, calibrate};
use crate::differ::{ConformanceRow, Tier, TierResult};
use crate::oracle::{GroundTruth, Metric};
use crate::relevance::Returned;

/// An unfiltered truth over `n_base` points: `ids` and `scores` row-major at
/// `k` per query, so `n_queries` is `ids.len() / k`. Dimension 4 and zero
/// checksums, which nothing that scores a truth reads.
pub(crate) fn gt(
    metric: Metric,
    n_base: usize,
    k: usize,
    ids: Vec<u32>,
    scores: Vec<f64>,
) -> GroundTruth {
    GroundTruth {
        metric,
        n_base,
        n_queries: ids.len().checked_div(k).unwrap_or(0),
        dim: 4,
        k,
        ids,
        scores,
        base_checksum: 0,
        query_checksum: 0,
        condition: None,
        n_matching: None,
    }
}

/// An absolute tolerance of `v`. Only `value` and `relative` decide a
/// verdict; the metric, dimension and spreads only describe it.
pub(crate) fn abs_eps(v: f64) -> Epsilon {
    let mut e = calibrate(
        Metric::Euclid,
        8,
        Distribution::default(),
        Distribution::default(),
        1.0,
    );
    e.value = v;
    e.relative = false;
    e
}

/// One query's returned list.
pub(crate) fn ret(ids: &[u32], scores: &[f64]) -> Returned {
    Returned {
        ids: ids.to_vec(),
        scores: scores.to_vec(),
    }
}

/// A tier result with no detail and no distribution.
pub(crate) fn tier(tier: Tier, passed: bool) -> TierResult {
    TierResult {
        advisory: false,
        tier,
        passed,
        detail: String::new(),
        distribution: None,
    }
}

/// A SIFT1M row on known builds at ε 1e-6 absolute and limit 10, carrying
/// `tiers`. Its hash is pinned by a test, so these values are part of it.
pub(crate) fn row_with(tiers: &[(Tier, bool)]) -> ConformanceRow {
    ConformanceRow {
        dataset: "sift1m".into(),
        metric: Metric::Euclid,
        dim: 128,
        qdrant_version: "1.19.0".into(),
        strawmann_commit: "abc".into(),
        isa_build: "avx512".into(),
        base_checksum: 1,
        query_checksum: 2,
        epsilon: 1e-6,
        epsilon_relative: false,
        limit: 10,
        tiers: tiers.iter().map(|&(t, passed)| tier(t, passed)).collect(),
    }
}
