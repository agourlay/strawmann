//! §8.2 — the oracle hierarchy.
//!
//! > "Neither engine is ground truth. When they disagree we need to know *who
//! > is wrong*, which requires a third, independent reference:
//! >
//! > 1. **fp64 exhaustive oracle.** A straightforward, deliberately unoptimised
//! >    double-precision exhaustive scan (NumPy is fine; correctness here
//! >    matters far more than speed). Computes exact top-k and exact scores for
//! >    the static datasets of §4.2, cached to disk per §4.3 so it is paid for
//! >    once. This is the arbiter, and it is also what produces the ground truth
//! >    files — the same code serves conformance and relevance.
//! > 2. **strawmann's own strict-order fp32 reference.**
//! > 3. **Qdrant.** A peer under test, not an authority."
//!
//! This module is tier 1. It is intentionally the dullest code in the
//! repository: no SIMD, no blocking, no reassociation, no early exit. Every
//! accumulation is `f64` and every loop is written the obvious way, because the
//! only property that matters is that a reader can convince themselves it is
//! right.
//!
//! ## Why the same code serves conformance and relevance
//!
//! §8.2's last clause is a design constraint, not an observation. If the ground
//! truth files were produced by one implementation and the disagreement arbiter
//! were another, a bug in either would show up as an engine defect. One
//! implementation means a bug in the oracle makes *both* engines look equally
//! wrong, which is a signature that is easy to recognise.

use rayon::prelude::*;

use std::path::Path;

/// §8.3's metrics, in the oracle's own terms.
///
/// The oracle computes the **returned score** — what a client sees — not the
/// internal similarity. That is deliberate: the differ compares what came off
/// the wire, and translating Qdrant's returned score back into an internal
/// similarity to compare it would introduce a transformation that could itself
/// be wrong.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    Dot,
    Cosine,
    Euclid,
    Manhattan,
}

impl Metric {
    /// Inverse of `as_str`, for reading the dataset descriptor.
    ///
    /// Returns `None` rather than defaulting: a metric this does not recognise
    /// means the descriptor names a distance the oracle cannot compute, and
    /// silently picking one would produce a ground truth for the wrong metric.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "dot" => Metric::Dot,
            "cosine" => Metric::Cosine,
            "euclid" | "l2" | "euclidean" => Metric::Euclid,
            "manhattan" | "l1" => Metric::Manhattan,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Dot => "dot",
            Metric::Cosine => "cosine",
            Metric::Euclid => "euclid",
            Metric::Manhattan => "manhattan",
        }
    }

    /// True when a *lower* returned score is a better match.
    ///
    /// §8.3: Euclid and Manhattan return distances, so their returned scores
    /// are order-reversed relative to the internal similarity the engines rank
    /// on. Getting this backwards inverts every result list.
    pub fn lower_is_better(self) -> bool {
        matches!(self, Metric::Euclid | Metric::Manhattan)
    }

    /// Does ingest normalise the stored vector? §8.3's preprocess column.
    pub fn normalises_at_ingest(self) -> bool {
        self == Metric::Cosine
    }
}

/// Reproduce Qdrant's cosine preprocessing exactly, in the oracle.
///
/// §8.3, trap 3: "Qdrant computes `length = Σx²` and, if
/// `length < f32::EPSILON || |length − 1.0| ≤ 1e-6`, **returns the vector
/// untouched**. Note it tests the *squared* length against 1.0 before taking
/// the square root."
///
/// The oracle must reproduce this **in f32**, not in f64, even though every
/// other computation here is f64. The short-circuit is a discontinuous
/// decision driven by an f32 accumulation; computing the length in f64 would
/// put a different population of vectors on each side of the boundary, and the
/// oracle would then disagree with both engines about which vectors were
/// normalised at all.
pub fn cosine_preprocess(v: &[f32]) -> Vec<f32> {
    let mut length: f32 = 0.0;
    for &x in v {
        length += x * x;
    }
    if length < f32::EPSILON || (length - 1.0).abs() <= 1.0e-6 {
        return v.to_vec();
    }
    let length = length.sqrt();
    v.iter().map(|x| x / length).collect()
}

/// Apply a metric's ingest preprocessing.
pub fn preprocess(metric: Metric, v: &[f32]) -> Vec<f32> {
    if metric.normalises_at_ingest() {
        cosine_preprocess(v)
    } else {
        v.to_vec()
    }
}

/// The exact returned score between a query and a stored vector, in f64.
///
/// Both inputs must already have been through `preprocess`.
pub fn score(metric: Metric, q: &[f32], v: &[f32]) -> f64 {
    debug_assert_eq!(q.len(), v.len());
    match metric {
        // §8.3: Dot and Cosine return the similarity unchanged.
        Metric::Dot | Metric::Cosine => {
            let mut acc = 0.0f64;
            for i in 0..q.len() {
                acc += f64::from(q[i]) * f64::from(v[i]);
            }
            acc
        }
        // §8.3: Euclid ranks on −Σ(a−b)² and returns √(Σ(a−b)²). Computed
        // directly, never via |a|²+|b|²−2ab — see §8.3 trap 2.
        Metric::Euclid => {
            let mut acc = 0.0f64;
            for i in 0..q.len() {
                let d = f64::from(q[i]) - f64::from(v[i]);
                acc += d * d;
            }
            acc.sqrt()
        }
        // §8.3: Manhattan returns Σ|a−b|.
        Metric::Manhattan => {
            let mut acc = 0.0f64;
            for i in 0..q.len() {
                acc += (f64::from(q[i]) - f64::from(v[i])).abs();
            }
            acc
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Neighbour {
    /// Row index in the base file. §4.3: "Ground truth indexes into base-file
    /// row order, so point ID must equal row index."
    pub id: u32,
    pub score: f64,
}

/// Exact top-k for one query, by exhaustive scan.
///
/// The tie-break is **lowest id wins**, matching §8.7's total order. Without a
/// deterministic tie-break the ground truth itself would vary between runs on
/// any dataset containing duplicate vectors — which §4.3 notes is "most real
/// ones".
/// The allocating form, kept as the readable entry point and used by the
/// tests; `compute` goes through `top_k_into` so the buffer is reused.
#[allow(dead_code)]
pub fn top_k(
    metric: Metric,
    base: &[f32],
    n: usize,
    dim: usize,
    query: &[f32],
    k: usize,
) -> Vec<Neighbour> {
    let mut scratch = Vec::new();
    top_k_into(metric, base, n, dim, query, k, &mut scratch)
}

/// `top_k`, reusing the caller's scratch buffer.
///
/// The buffer holds one `Neighbour` per base vector — 16 MB at SIFT1M, 190 MB
/// at the headline tier — and `top_k` allocated, filled, and freed one per
/// query on every thread. At 10,000 queries that is 10,000 allocations of 16 MB
/// each and the matching page faults, for a function whose actual work is
/// supposed to be the distance computation. Reusing it across the queries a
/// thread owns costs one allocation per thread instead.
///
/// The output is unchanged: the buffer is truncated and refilled, never read
/// across calls.
pub fn top_k_into(
    metric: Metric,
    base: &[f32],
    n: usize,
    dim: usize,
    query: &[f32],
    k: usize,
    all: &mut Vec<Neighbour>,
) -> Vec<Neighbour> {
    all.clear();
    all.reserve(n);
    for i in 0..n {
        let v = &base[i * dim..(i + 1) * dim];
        all.push(Neighbour {
            id: i as u32,
            score: score(metric, query, v),
        });
    }

    take_k(metric, k, all)
}

/// `top_k_into` over an explicit id set instead of `0..n`.
///
/// §4.3's cached ground truth is *unfiltered*, and the neighbours of `q` among
/// all points are not its neighbours among the points matching a condition
/// (docs/workloads.md, W12 point 3) — so a filtered query scored against the
/// unfiltered list is scored against both the wrong ids and the wrong k-th
/// distance. This is the restricted scan that makes the right one computable.
///
/// Ids at or beyond `n_base` are skipped rather than panicking: the id set
/// comes from what the engine says it holds, and a condition naming a point
/// the corpus does not have should cost that point, not the run.
#[allow(clippy::too_many_arguments)]
pub fn top_k_of_into(
    metric: Metric,
    base: &[f32],
    n_base: usize,
    dim: usize,
    query: &[f32],
    ids: &[u32],
    k: usize,
    all: &mut Vec<Neighbour>,
) -> Vec<Neighbour> {
    all.clear();
    all.reserve(ids.len());
    for &id in ids {
        let i = id as usize;
        if i >= n_base {
            continue;
        }
        let v = &base[i * dim..(i + 1) * dim];
        all.push(Neighbour {
            id,
            score: score(metric, query, v),
        });
    }
    take_k(metric, k, all)
}

/// The k smallest (or largest) of `all` under §8.7's total order.
///
/// One definition, shared by the full and the restricted scan: the order and
/// the partition are what make a ground truth reproducible, and two copies of
/// them are two things that can drift.
fn take_k(metric: Metric, k: usize, all: &mut Vec<Neighbour>) -> Vec<Neighbour> {
    let lower_better = metric.lower_is_better();
    // §8.7's total order: score first, then id. Comparing ids rather than
    // relying on sort stability is what makes the result independent of the
    // sort algorithm — which is exactly what lets the partial sort below be
    // substituted for a full one without changing a single output byte.
    let cmp = |a: &Neighbour, b: &Neighbour| {
        let ord = if lower_better {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        ord.then(a.id.cmp(&b.id))
    };

    // Partition at k rather than sorting all of `n`.
    //
    // The full sort is O(n log n) *per query*, and at SIFT1M's scale that is
    // ~2×10¹¹ comparisons across the query set — hours, and it dominates the
    // distance computation it was supposed to be a footnote to. Partitioning is
    // O(n), leaving only k elements to order.
    //
    // Because `cmp` is a total order this selects the same k elements and puts
    // them in the same sequence as the full sort; `select_nth_unstable_by`
    // being unstable cannot matter when no two elements compare equal.
    if k < all.len() {
        all.select_nth_unstable_by(k, cmp);
        all.truncate(k);
    }
    all.sort_unstable_by(cmp);
    // Only the k survivors are copied out; the scratch keeps its capacity.
    all.clone()
}

/// Ground truth for a whole query set.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroundTruth {
    pub metric: Metric,
    pub n_base: usize,
    pub n_queries: usize,
    pub dim: usize,
    pub k: usize,
    /// `n_queries × k` neighbour ids, row-major.
    pub ids: Vec<u32>,
    /// `n_queries × k` scores, row-major.
    ///
    /// A base set smaller than `k` pads with `NaN` (see `compute`). JSON has
    /// no NaN: serde writes it as `null` and then refuses to read `null` back
    /// into an `f64`, so such a file could be saved and never loaded. The pad
    /// round-trips as `null` explicitly.
    #[serde(with = "nan_as_null")]
    pub scores: Vec<f64>,
    /// §4.3: "Checksums of base, query, and GT files are recorded in every
    /// result row." Recorded here so a result row can carry them.
    pub base_checksum: u64,
    pub query_checksum: u64,
    /// The filter condition this truth is restricted to, `None` for §4.3's
    /// unfiltered cache.
    ///
    /// Recorded so the two cannot be confused by a consumer: a filtered truth
    /// and an unfiltered one have the same shape and describe different
    /// questions, and W12's whole defect was scoring one against the other.
    /// `serde(default)` because every ground truth already on disk predates
    /// the field and must keep loading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// How many points satisfied `condition` — "how many could have" from
    /// docs/workloads.md W12 point 4, and the denominator a filtered recall is
    /// scored against as `min(k, n_matching)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_matching: Option<usize>,
}

impl GroundTruth {
    pub fn neighbours(&self, query: usize) -> &[u32] {
        &self.ids[query * self.k..(query + 1) * self.k]
    }

    pub fn scores_for(&self, query: usize) -> &[f64] {
        &self.scores[query * self.k..(query + 1) * self.k]
    }

    /// §4.3's boundary-tie rule, against the **top-`k` prefix** of the truth
    /// rather than the whole cached list.
    ///
    /// > "Count a returned point as correct if its distance is within ε of the
    /// > k-th ground-truth neighbour's distance, per the ann-benchmarks
    /// > convention. Strict ID matching understates recall on any dataset
    /// > containing duplicate or near-duplicate vectors, which is most real
    /// > ones."
    ///
    /// So a returned id counts if it is in the truth prefix **or** its score
    /// ties the k-th truth score within ε. The second clause is what stops a
    /// duplicate vector from being scored as a miss purely because the oracle
    /// happened to list its twin. `epsilon` is an **absolute** score distance;
    /// callers holding a relative tolerance must scale it first
    /// (`Epsilon::absolute_at`).
    ///
    /// There was a `is_correct` beside this that tested against the whole
    /// cached list. It is gone rather than kept for symmetry: it is the lenient
    /// reading that made `recall@10` accept anything in the true top 100, and
    /// leaving a footgun in the API because it has no callers today is how it
    /// acquires one tomorrow.
    ///
    /// Ground truth is cached at one k — 100 here — and reused for every recall
    /// depth. Testing membership against the full list therefore makes
    /// `recall@10` accept anything in the true top *100*, and ties against the
    /// *100th* score rather than the 10th. Both are far too lenient: a
    /// competent index returns its 11th-nearest inside the top 100 essentially
    /// always, so `recall@10` reads 1.0000 regardless of `ef` and the whole
    /// recall-vs-`ef` curve — the deliverable of W10 and T3 — flattens to a
    /// constant.
    ///
    /// Measured on SIFT1M at 50k: `recall@10` reported 1.0000 at `ef` 16, 64
    /// and 256 while MRDE, which compares distances positionally and so was
    /// unaffected, moved 0.00464 → 0.00076 → 0.00050. Two numbers from the same
    /// results disagreeing about whether `ef` matters is what exposed it.
    /// The verdict on one returned `(id, score)` at depth `k`, with the tie
    /// clause judged on a score the engine did not get to choose.
    ///
    /// The tie clause compared the k-th truth score against the score *the
    /// engine reported* for the returned id. That trusts the thing under
    /// test: an engine that returns a wrong point with a flattering score
    /// gets it counted as correct. Two defences:
    ///
    /// - `oracle_score`, when the caller can supply it, is the fp64 score of
    ///   the returned point recomputed from the base vectors (`Rescorer`);
    ///   the tie test uses that and the reported score is ignored.
    /// - without it, a reported score *better* than the k-th truth by more
    ///   than ε for a point that is not in the truth is impossible — the
    ///   oracle would have listed it — and is judged `ImpossibleScore`: a miss,
    ///   and a violation the caller reports rather than folds into recall.
    ///   With the oracle score the same test applies to it, and then names a
    ///   ground-truth defect rather than an engine one.
    pub fn judge_at(
        &self,
        query: usize,
        k: usize,
        id: u32,
        returned_score: f64,
        oracle_score: Option<f64>,
        epsilon: f64,
    ) -> Judgement {
        let k = k.min(self.k);
        if k == 0 {
            return Judgement::Miss;
        }
        let truth = &self.neighbours(query)[..k];
        if truth.contains(&id) {
            return Judgement::Correct;
        }
        // §8.6's tie clause, against the k-th score *of this prefix*: a vector
        // exactly as far away as the k-th true neighbour is a correct answer,
        // because which of them the engine returns is arbitrary.
        let kth = self.scores_for(query)[k - 1];
        let score = oracle_score.unwrap_or(returned_score);
        if !kth.is_finite() || !score.is_finite() {
            return Judgement::Miss;
        }
        if (score - kth).abs() <= epsilon {
            return Judgement::Correct;
        }
        // Strictly better than the k-th truth, by more than the tie width, for
        // a point the truth does not hold: the exhaustive scan would have
        // ranked it inside the top k. Either the score is not this point's,
        // or the id is not this point's — both are engine defects, never a
        // near miss.
        let better = if self.metric.lower_is_better() {
            score < kth - epsilon
        } else {
            score > kth + epsilon
        };
        if better {
            Judgement::ImpossibleScore
        } else {
            Judgement::Miss
        }
    }
}

/// The verdict on one returned `(id, score)` at depth `k`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Judgement {
    Correct,
    Miss,
    /// A non-member whose score beats the k-th truth: see `judge_at`.
    ImpossibleScore,
}

/// Recomputes, in fp64 from the base vectors, the score of any returned point —
/// so the tie clause in `judge_at` can be judged on the oracle's arithmetic
/// instead of the engine's word.
///
/// Both sides go through `preprocess`, as `compute` does; a cosine collection
/// stores the normalised vector and the score must be of that. The base is
/// borrowed and a row is normalised when it is scored: a rescore touches `k`
/// rows per query, and normalising the whole corpus up front copied it —
/// 5.7 GiB at the headline tier, held beside the raw data for the whole
/// sweep, on the host decisions.md item 7 records being OOM-killed.
pub struct Rescorer<'a> {
    metric: Metric,
    dim: usize,
    n_base: usize,
    base: &'a [f32],
    queries: Vec<f32>,
}

impl<'a> Rescorer<'a> {
    pub fn new(
        metric: Metric,
        base: &'a [f32],
        n_base: usize,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
    ) -> Self {
        let mut q_pre = Vec::with_capacity(queries.len());
        for i in 0..n_queries {
            q_pre.extend_from_slice(&preprocess(metric, &queries[i * dim..(i + 1) * dim]));
        }
        Rescorer {
            metric,
            dim,
            n_base,
            base,
            queries: q_pre,
        }
    }

    /// `None` when the id is not a row of the base file: §4.3 requires ids to
    /// be row indices, so that is itself a finding for the caller.
    pub fn score(&self, query: usize, id: u32) -> Option<f64> {
        let id = id as usize;
        if id >= self.n_base || (query + 1) * self.dim > self.queries.len() {
            return None;
        }
        let q = &self.queries[query * self.dim..(query + 1) * self.dim];
        let raw = &self.base[id * self.dim..(id + 1) * self.dim];
        if self.metric.normalises_at_ingest() {
            let v = preprocess(self.metric, raw);
            Some(score(self.metric, q, &v))
        } else {
            Some(score(self.metric, q, raw))
        }
    }
}

#[cfg(test)]
mod rescorer_tests {
    use super::*;

    #[test]
    fn a_cosine_rescorer_scores_the_normalised_row_without_copying_the_corpus() {
        // Two rows, one query; the score must be of the normalised row, as
        // `compute` scores it, and the base stays a borrow.
        let dim = 4;
        let base = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let q = vec![1.0f32, 1.0, 0.0, 0.0];
        let r = Rescorer::new(Metric::Cosine, &base, 2, &q, 1, dim);
        assert_eq!(r.base.as_ptr(), base.as_ptr());
        let want0 = score(
            Metric::Cosine,
            &preprocess(Metric::Cosine, &q),
            &preprocess(Metric::Cosine, &base[..4]),
        );
        let want1 = score(
            Metric::Cosine,
            &preprocess(Metric::Cosine, &q),
            &preprocess(Metric::Cosine, &base[4..]),
        );
        assert_eq!(r.score(0, 0), Some(want0));
        assert_eq!(r.score(0, 1), Some(want1));
        assert_eq!(r.score(0, 2), None);
        // Against the ground truth the oracle computes for the same data.
        let gt = compute(Metric::Cosine, &base, 2, &q, 1, dim, 2);
        let top = gt.neighbours(0)[0];
        assert_eq!(r.score(0, top), Some(gt.scores_for(0)[0]));
    }
}

/// Compute ground truth for every query.
pub fn compute(
    metric: Metric,
    base: &[f32],
    n_base: usize,
    queries: &[f32],
    n_queries: usize,
    dim: usize,
    k: usize,
) -> GroundTruth {
    compute_over(metric, base, n_base, queries, n_queries, dim, k, None)
}

/// `compute`, restricted to the points satisfying a condition.
///
/// docs/workloads.md, W12 point 3: "the neighbours of `q` among all points are
/// not the neighbours of `q` among the points matching `tag in S`". `matching`
/// is that id set, as the engine reports holding it; `condition` is the label
/// the cache and the row are keyed by.
///
/// `n_base` stays the corpus size, because the ids index into the corpus. What
/// the condition selected is `n_matching`.
#[allow(clippy::too_many_arguments)]
pub fn compute_filtered(
    metric: Metric,
    base: &[f32],
    n_base: usize,
    queries: &[f32],
    n_queries: usize,
    dim: usize,
    k: usize,
    condition: &str,
    matching: &[u32],
) -> GroundTruth {
    let mut gt = compute_over(
        metric,
        base,
        n_base,
        queries,
        n_queries,
        dim,
        k,
        Some(matching),
    );
    gt.condition = Some(condition.to_string());
    gt.n_matching = Some(matching.iter().filter(|&&i| (i as usize) < n_base).count());
    gt
}

#[allow(clippy::too_many_arguments)]
fn compute_over(
    metric: Metric,
    base: &[f32],
    n_base: usize,
    queries: &[f32],
    n_queries: usize,
    dim: usize,
    k: usize,
    matching: Option<&[u32]>,
) -> GroundTruth {
    // §4.3: preprocessing must match what ingest does, or the ground truth is
    // for a different collection than the one under test.
    // Borrowed when no preprocessing is needed, which is three metrics out of
    // four. `base.to_vec()` copied the whole corpus to produce something
    // identical to its input: 0.5 GB at SIFT1M and 6 GB at the headline tier,
    // doubling peak memory for a run whose oracle is already the memory-hungry
    // part.
    let base_pre: std::borrow::Cow<'_, [f32]> = if metric.normalises_at_ingest() {
        let mut out = Vec::with_capacity(base.len());
        for i in 0..n_base {
            out.extend_from_slice(&preprocess(metric, &base[i * dim..(i + 1) * dim]));
        }
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Borrowed(base)
    };

    // Parallel across queries, which is the one axis where it is free: each
    // query reads the shared `base_pre` and writes only its own row, so there
    // is no reduction to order and no accumulation to reassociate. The fp64
    // sums inside `score` stay strictly sequential, which is what §8.2's tier-1
    // oracle requires — parallelising *within* a distance would change the
    // summation order and therefore the answer.
    //
    // `map_init` + `collect` over an indexed parallel iterator preserves order,
    // so the output is identical to the serial loop regardless of thread count.
    // `map_init`'s per-thread state is only the scratch buffer, which is
    // cleared at the top of every call and never read across queries — it
    // cannot carry information from one query into another.
    // `parallel_ground_truth_matches_a_serial_computation` holds this down.
    let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n_queries)
        .into_par_iter()
        .map_init(Vec::new, |scratch: &mut Vec<Neighbour>, qi| {
            let q = preprocess(metric, &queries[qi * dim..(qi + 1) * dim]);
            let top = match matching {
                None => top_k_into(metric, &base_pre, n_base, dim, &q, k, scratch),
                Some(ids) => top_k_of_into(metric, &base_pre, n_base, dim, &q, ids, k, scratch),
            };
            let mut row_ids = Vec::with_capacity(k);
            let mut row_scores = Vec::with_capacity(k);
            for nb in &top {
                row_ids.push(nb.id);
                row_scores.push(nb.score);
            }
            // A base set smaller than k would otherwise produce ragged rows.
            for _ in top.len()..k {
                row_ids.push(u32::MAX);
                row_scores.push(f64::NAN);
            }
            (row_ids, row_scores)
        })
        .collect();

    let mut ids = Vec::with_capacity(n_queries * k);
    let mut scores = Vec::with_capacity(n_queries * k);
    for (row_ids, row_scores) in rows {
        ids.extend_from_slice(&row_ids);
        scores.extend_from_slice(&row_scores);
    }

    GroundTruth {
        metric,
        n_base,
        n_queries,
        dim,
        k,
        ids,
        scores,
        base_checksum: checksum_f32(base),
        query_checksum: checksum_f32(queries),
        condition: None,
        n_matching: None,
    }
}

/// `Vec<f64>` with NaN written as JSON `null` and read back as NaN.
mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        let opt: Vec<Option<f64>> = v
            .iter()
            .map(|&x| if x.is_nan() { None } else { Some(x) })
            .collect();
        opt.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        let opt: Vec<Option<f64>> = Vec::deserialize(d)?;
        Ok(opt.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
    }
}

/// FNV-1a over the raw bytes. §4.3 wants checksums recorded per result row; the
/// specific function does not matter, only that it is stable and cheap.
pub fn checksum_f32(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in data {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// §4.3: "Ground truth is cached per `(dataset, subset size, metric)` and
/// recomputed when any of those change."
pub fn cache_key(dataset: &str, subset: usize, metric: Metric, k: usize) -> String {
    cache_key_for(dataset, subset, metric, k, None)
}

/// `cache_key`, keyed by the condition as well.
///
/// docs/workloads.md, W12 point 3 asks for a cache keyed by "(dataset, metric,
/// k, condition) rather than by (dataset, metric, k)" — without the condition
/// two different filters share a file and the second silently reads the first's
/// answer. The condition is hashed rather than spelled out because it is a
/// filter expression and file names are not: `a=keyword_7` is fine and
/// `a in {keyword_7, ...}` at ten values is not.
pub fn cache_key_for(
    dataset: &str,
    subset: usize,
    metric: Metric,
    k: usize,
    condition: Option<&str>,
) -> String {
    match condition {
        None => format!("{dataset}.n{subset}.{}.k{k}.gt.json", metric.as_str()),
        Some(c) => format!(
            "{dataset}.n{subset}.{}.k{k}.cond-{:016x}.gt.json",
            metric.as_str(),
            condition_hash(c)
        ),
    }
}

/// A stable 64-bit digest of a condition expression, for `cache_key_for`.
///
/// Written out rather than using `DefaultHasher`, whose output std explicitly
/// does not promise across versions — a cache key that changes when the
/// toolchain does silently recomputes every ground truth.
pub fn condition_hash(condition: &str) -> u64 {
    // FNV-1a, 64-bit.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in condition.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn save(gt: &GroundTruth, path: &Path) -> anyhow::Result<()> {
    let f = std::fs::File::create(path)?;
    serde_json::to_writer(f, gt)?;
    Ok(())
}

pub fn load(path: &Path) -> anyhow::Result<GroundTruth> {
    let f = std::fs::File::open(path)?;
    Ok(serde_json::from_reader(f)?)
}

/// §4.3: "**Recompute ground truth in fp64 ourselves and diff it against the
/// published file.** Several published GTs were computed in fp32 and have
/// genuine disagreements near the k-th boundary. Use ours, publish the diff
/// once, and note that a disagreement with the shipped GT is expected rather
/// than alarming."
#[derive(Debug, Default)]
pub struct GtDiff {
    pub queries: usize,
    /// Queries where the two lists are identical.
    pub identical: usize,
    /// Queries whose neighbour **set** first diverges at or after the boundary
    /// band, i.e. only at the k-th cut.
    pub boundary_only: usize,
    /// Queries whose neighbour **set** diverges before the boundary — the
    /// alarming case, and the only one that indicates a real disagreement about
    /// which vectors are nearest.
    pub interior: usize,
    pub total_position_mismatches: usize,
    /// Queries whose two lists hold the **same k ids in a different order**.
    ///
    /// Positional equality conflates two very different things. A ground truth
    /// computed with a different tie-breaking convention — or simply in fp32,
    /// where two genuinely-equal distances can land either way round — permutes
    /// tied runs without disagreeing about which vectors are the nearest k.
    /// SIFT descriptors are quantized integers and SIFT1M is known to contain
    /// duplicates, so exact score ties are expected rather than exotic, and a
    /// diff that cannot separate "reordered" from "different" would report a
    /// convention mismatch as a correctness problem.
    pub same_set_different_order: usize,
    /// Σ|A ∩ B| over all queries, against a maximum of `queries × k`.
    pub id_overlap: usize,
    /// `queries × k`, carried so `overlap_fraction` does not need `k` passed in.
    pub id_overlap_max: usize,
}

impl GtDiff {
    pub fn max_overlap(&self) -> usize {
        self.id_overlap_max
    }

    /// The headline number: what fraction of the published neighbours we agree
    /// are neighbours at all, ignoring order entirely.
    pub fn overlap_fraction(&self) -> f64 {
        if self.id_overlap_max == 0 {
            return f64::NAN;
        }
        self.id_overlap as f64 / self.id_overlap_max as f64
    }

    pub fn describe(&self) -> String {
        format!(
            "{} queries: {} identical, {} same-set-different-order, {} differ only at the k-th boundary, \
{} differ in the interior ({} positions total); id overlap {}/{} = {:.6}",
            self.queries,
            self.identical,
            self.same_set_different_order,
            self.boundary_only,
            self.interior,
            self.total_position_mismatches,
            self.id_overlap,
            self.max_overlap(),
            self.overlap_fraction(),
        )
    }
}

/// Diff our recomputed ground truth against a published one.
pub fn diff_against_published(ours: &GroundTruth, published_ids: &[u32]) -> GtDiff {
    let mut d = GtDiff {
        queries: ours.n_queries,
        id_overlap_max: ours.n_queries * ours.k,
        ..Default::default()
    };
    let mut set_a: std::collections::HashSet<u32> =
        std::collections::HashSet::with_capacity(ours.k);
    let mut published_set: std::collections::HashSet<u32> =
        std::collections::HashSet::with_capacity(ours.k);
    for qi in 0..ours.n_queries {
        let a = ours.neighbours(qi);
        let b = &published_ids[qi * ours.k..(qi + 1) * ours.k];
        let mut first_mismatch: Option<usize> = None;
        let mut mismatches = 0;
        for i in 0..ours.k {
            if a[i] != b[i] {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
            }
        }
        d.total_position_mismatches += mismatches;

        set_a.clear();
        set_a.extend(a.iter().copied());
        published_set.clear();
        published_set.extend(b.iter().copied());
        let overlap = b.iter().filter(|id| set_a.contains(id)).count();
        d.id_overlap += overlap;

        if first_mismatch.is_none() {
            d.identical += 1;
            continue;
        }
        // Same members, different order: a tie-breaking artefact, not a
        // disagreement about which vectors are nearest.
        if overlap == ours.k {
            d.same_set_different_order += 1;
            continue;
        }

        // Classify on where the **set** diverges, not on the first positional
        // mismatch.
        //
        // Those are not the same thing, and using the positional one is
        // actively misleading. A query can have its tied runs permuted at rank
        // 3 — no disagreement at all, just the other tie-breaking convention —
        // while the only genuine set difference is the single id at rank 99 cut
        // off by the k-th boundary. Keyed on first positional mismatch that
        // query reports as an "interior" disagreement, which is the alarming
        // bucket, when nothing interior actually differs.
        //
        // Measured on SIFT1M this was not hypothetical: 63 of 10000 queries
        // landed in `interior`, and every one of the 111 differing ids across
        // the whole set turned out to sit at rank 99 with a score exactly equal
        // to the k-th. The alarm was entirely an artefact of this key.
        // "Boundary" is the k-th rank and nothing wider. This was a band of
        // `k/10 + 1` ranks — 11 of 100 — while `docs/ground-truth.md` reports
        // the SIFT1M result as "differ only at the k-th boundary" and shows
        // every differing id sitting at rank 99. A band that wide would have
        // filed a genuine disagreement at rank 90 under the same reassuring
        // label. Any set difference before the last rank is interior.
        let divergence = (0..ours.k)
            .find(|&i| !published_set.contains(&a[i]))
            .unwrap_or(ours.k - 1);
        if divergence == ours.k - 1 {
            d.boundary_only += 1;
        } else {
            d.interior += 1;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclid_returns_the_distance_not_its_square() {
        // §8.3 trap 1, in the oracle.
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];
        assert_eq!(score(Metric::Euclid, &a, &b), 5.0);
        assert!(Metric::Euclid.lower_is_better());
    }

    #[test]
    fn manhattan_returns_the_l1_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, -4.0, 1.0];
        assert_eq!(score(Metric::Manhattan, &a, &b), 8.0);
    }

    #[test]
    fn dot_keeps_its_sign() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert_eq!(score(Metric::Dot, &a, &b), -1.0);
        assert!(!Metric::Dot.lower_is_better());
    }

    #[test]
    fn cosine_short_circuit_matches_qdrant() {
        // §8.3 trap 3. A vector already within 1e-6 of unit *squared* length is
        // returned untouched.
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(cosine_preprocess(&v), v);

        // 3-4-5 triangle: squared length 25, well outside the window.
        let v = vec![3.0f32, 4.0, 0.0, 0.0];
        let out = cosine_preprocess(&v);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);

        // A zero vector must not divide by zero.
        let v = vec![0.0f32; 8];
        assert_eq!(cosine_preprocess(&v), v);
    }

    #[test]
    fn cosine_length_test_is_on_the_squared_length() {
        // Squared length 1.0000005 is inside the window; its square root is
        // 1.00000025, so a root-based test would accept a wider band. Pin the
        // squared form.
        let mut v = vec![0.0f32; 2];
        v[0] = (1.0000005f32).sqrt();
        let before = v.clone();
        assert_eq!(cosine_preprocess(&v), before);
    }

    #[test]
    fn top_k_breaks_ties_by_lowest_id() {
        // Three identical vectors: the oracle must be deterministic about which
        // it lists, or the ground truth varies run to run.
        let dim = 2;
        let base: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let q = [1.0f32, 0.0];
        let top = top_k(Metric::Dot, &base, 3, dim, &q, 2);
        assert_eq!(top[0].id, 0);
        assert_eq!(top[1].id, 1);
    }

    #[test]
    fn top_k_orders_euclid_by_increasing_distance() {
        let dim = 2;
        let base: Vec<f32> = vec![10.0, 0.0, 1.0, 0.0, 5.0, 0.0];
        let q = [0.0f32, 0.0];
        let top = top_k(Metric::Euclid, &base, 3, dim, &q, 3);
        assert_eq!(top[0].id, 1); // distance 1
        assert_eq!(top[1].id, 2); // distance 5
        assert_eq!(top[2].id, 0); // distance 10
        assert_eq!(top[0].score, 1.0);
    }

    #[test]
    fn ground_truth_round_trips_and_is_deterministic() {
        let dim = 8;
        let n = 50;
        let nq = 5;
        let base: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 37 % 101) as f32) / 101.0)
            .collect();
        let queries: Vec<f32> = (0..nq * dim)
            .map(|i| ((i * 53 % 97) as f32) / 97.0)
            .collect();

        let a = compute(Metric::Euclid, &base, n, &queries, nq, dim, 10);
        let b = compute(Metric::Euclid, &base, n, &queries, nq, dim, 10);
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.base_checksum, b.base_checksum);
        assert_eq!(a.ids.len(), nq * 10);
    }

    #[test]
    fn is_correct_accepts_a_tie_at_the_boundary() {
        // §4.3: "Strict ID matching understates recall on any dataset
        // containing duplicate or near-duplicate vectors."
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![0.5, 0.7, 0.9],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        // An id in the list is correct regardless of score.
        assert_eq!(gt.judge_at(0, 3, 2, 0.7, None, 1e-6), Judgement::Correct);
        // An id not in the list, but tying the k-th score, is also correct.
        assert_eq!(
            gt.judge_at(0, 3, 99, 0.9 + 1e-9, None, 1e-6),
            Judgement::Correct
        );
        // An id not in the list and clearly worse is not.
        assert_eq!(gt.judge_at(0, 3, 99, 2.0, None, 1e-6), Judgement::Miss);
    }

    #[test]
    fn gt_diff_separates_boundary_from_interior_disagreement() {
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 10,
            n_queries: 3,
            dim: 2,
            k: 10,
            ids: (0..30).map(|i| (i % 10) as u32).collect(),
            scores: vec![0.0; 30],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let mut published: Vec<u32> = gt.ids.clone();
        // Query 0 identical; query 1 differs at the last position; query 2
        // differs at position 0.
        published[19] = 99;
        published[20] = 99;

        let d = diff_against_published(&gt, &published);
        assert_eq!(d.queries, 3);
        assert_eq!(d.identical, 1);
        assert_eq!(d.boundary_only, 1);
        assert_eq!(d.interior, 1);
    }

    #[test]
    fn a_ground_truth_padded_past_the_base_size_round_trips_through_json() {
        // n_base=2 < k=4: `compute` pads with NaN, and serde_json wrote it as
        // null and then failed to read the file it had just written.
        let base = vec![0.0f32, 0.0, 1.0, 0.0];
        let q = vec![0.0f32, 0.0];
        let gt = compute(Metric::Euclid, &base, 2, &q, 1, 2, 4);
        assert!(gt.scores[2].is_nan() && gt.scores[3].is_nan());
        assert_eq!(gt.ids[2], u32::MAX);
        let json = serde_json::to_string(&gt).unwrap();
        assert!(json.contains("null"));
        let back: GroundTruth = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ids, gt.ids);
        assert_eq!(back.scores[0], 0.0);
        assert_eq!(back.scores[1], 1.0);
        assert!(back.scores[2].is_nan() && back.scores[3].is_nan());
    }

    #[test]
    fn a_non_member_that_claims_to_beat_the_kth_truth_is_impossible_not_correct() {
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![0.5, 0.7, 0.9],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        // Reported score 0.1 for id 99: closer than the true nearest, and not
        // in the truth. The exhaustive scan would have listed it; the engine
        // is lying about the id or the score.
        assert_eq!(
            gt.judge_at(0, 3, 99, 0.1, None, 1e-6),
            Judgement::ImpossibleScore
        );
        // A boundary tie is still correct, and a plain miss a plain miss.
        assert_eq!(gt.judge_at(0, 3, 99, 0.9, None, 1e-6), Judgement::Correct);
        assert_eq!(gt.judge_at(0, 3, 99, 2.0, None, 1e-6), Judgement::Miss);
        // Members are correct whatever they report.
        assert_eq!(gt.judge_at(0, 3, 2, 0.1, None, 1e-6), Judgement::Correct);

        // With the oracle's own score for the point, the reported one is
        // ignored: an engine claiming 0.9 (a tie) for a point that is really
        // at 2.0 gets a miss.
        assert_eq!(gt.judge_at(0, 3, 99, 0.9, Some(2.0), 1e-6), Judgement::Miss);
        // And a claimed 5.0 for a point really at 0.9 is a tie, hence correct.
        assert_eq!(
            gt.judge_at(0, 3, 99, 5.0, Some(0.9), 1e-6),
            Judgement::Correct
        );

        // Similarity metrics flip "better".
        let sim = GroundTruth {
            metric: Metric::Dot,
            scores: vec![0.9, 0.7, 0.5],
            ..gt.clone()
        };
        assert_eq!(
            sim.judge_at(0, 3, 99, 0.99, None, 1e-6),
            Judgement::ImpossibleScore
        );
        assert_eq!(sim.judge_at(0, 3, 99, 0.1, None, 1e-6), Judgement::Miss);
    }

    #[test]
    fn the_rescorer_reproduces_the_oracle_score_for_any_id() {
        let dim = 2;
        let base = vec![0.0f32, 0.0, 3.0, 4.0, 1.0, 1.0];
        let q = vec![0.0f32, 0.0];
        let r = Rescorer::new(Metric::Euclid, &base, 3, &q, 1, dim);
        assert_eq!(r.score(0, 1), Some(5.0));
        assert_eq!(r.score(0, 0), Some(0.0));
        // Not a base row: §4.3 says ids are row indices, so this is a finding.
        assert_eq!(r.score(0, 7), None);
        // Cosine goes through the same preprocessing as ingest.
        let c = Rescorer::new(Metric::Cosine, &base, 3, &[3.0f32, 4.0], 1, dim);
        assert!((c.score(0, 1).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cache_key_changes_with_every_invalidating_input() {
        // §4.3: "Ground truth is cached per `(dataset, subset size, metric)`
        // and recomputed when any of those change."
        let a = cache_key("sift1m", 1_000_000, Metric::Euclid, 100);
        assert_ne!(a, cache_key("sift1m", 500_000, Metric::Euclid, 100));
        assert_ne!(a, cache_key("sift1m", 1_000_000, Metric::Dot, 100));
        assert_ne!(a, cache_key("gist1m", 1_000_000, Metric::Euclid, 100));
        assert_ne!(a, cache_key("sift1m", 1_000_000, Metric::Euclid, 10));
    }

    #[test]
    fn a_condition_gets_its_own_cache_file() {
        // docs/workloads.md W12 point 3: keyed by "(dataset, metric, k,
        // condition) rather than by (dataset, metric, k)". Without the
        // condition two filters share a file and the second reads the first's
        // answer.
        let plain = cache_key("sift1m", 200_000, Metric::Euclid, 10);
        assert_eq!(
            cache_key_for("sift1m", 200_000, Metric::Euclid, 10, None),
            plain,
            "the unfiltered name is unchanged, so no cached truth is orphaned"
        );
        let a = cache_key_for("sift1m", 200_000, Metric::Euclid, 10, Some("a=keyword_7"));
        let b = cache_key_for("sift1m", 200_000, Metric::Euclid, 10, Some("a=keyword_8"));
        assert_ne!(a, b);
        assert_ne!(a, plain);
        assert_eq!(condition_hash("a=keyword_7"), condition_hash("a=keyword_7"));
        assert_ne!(condition_hash("a=keyword_7"), condition_hash("a=keyword_8"));
    }

    #[test]
    fn restricting_to_every_id_is_the_unfiltered_computation() {
        // The restricted scan has to be the *same* computation over a subset,
        // not a second implementation of it.
        let (dim, n, k) = (4usize, 40usize, 5usize);
        let base: Vec<f32> = (0..n * dim).map(|i| (i % 17) as f32 * 0.5).collect();
        let queries: Vec<f32> = (0..3 * dim).map(|i| (i % 7) as f32).collect();
        let all: Vec<u32> = (0..n as u32).collect();
        let plain = compute(Metric::Euclid, &base, n, &queries, 3, dim, k);
        let filtered = compute_filtered(Metric::Euclid, &base, n, &queries, 3, dim, k, "all", &all);
        assert_eq!(plain.ids, filtered.ids);
        assert_eq!(plain.scores, filtered.scores);
        assert_eq!(filtered.n_matching, Some(n));
        assert_eq!(filtered.condition.as_deref(), Some("all"));
        assert!(plain.condition.is_none(), "the §4.3 cache stays unmarked");
    }

    #[test]
    fn a_filtered_truth_moves_both_the_ids_and_the_kth_distance() {
        // W12 point 3, as a case rather than a sentence: points on a line at
        // x = 0, 1, 2, ..., the query at the origin, and a condition keeping
        // every tenth. The unfiltered answer is 0,1,2 and the filtered one is
        // 9,19,29 -- and the k-th distance, which recall is scored against,
        // is not the same number.
        let (dim, n, k) = (2usize, 50usize, 3usize);
        let base: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0]).collect();
        let queries = [0.0f32, 0.0];
        let matching: Vec<u32> = (0..n as u32).filter(|i| i % 10 == 9).collect();
        let gt = compute_filtered(
            Metric::Euclid,
            &base,
            n,
            &queries,
            1,
            dim,
            k,
            "i%10==9",
            &matching,
        );
        assert_eq!(gt.neighbours(0), &[9, 19, 29]);
        assert_eq!(gt.n_matching, Some(5));
        let plain = compute(Metric::Euclid, &base, n, &queries, 1, dim, k);
        assert_eq!(plain.neighbours(0), &[0, 1, 2]);
        assert_ne!(plain.scores_for(0)[k - 1], gt.scores_for(0)[k - 1]);
    }

    #[test]
    fn a_condition_matching_fewer_than_k_says_how_many_it_had() {
        // W12 point 4: an engine returning all 2 matching points is exactly
        // right, and scores 0.2 at k=10 unless the denominator is
        // `min(k, n_matching)`. The truth carries the number that makes that
        // denominator computable.
        let (dim, n, k) = (2usize, 20usize, 5usize);
        let base: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0]).collect();
        let queries = [0.0f32, 0.0];
        let gt = compute_filtered(
            Metric::Euclid,
            &base,
            n,
            &queries,
            1,
            dim,
            k,
            "two",
            &[3u32, 7],
        );
        assert_eq!(gt.n_matching, Some(2));
        assert_eq!(&gt.neighbours(0)[..2], &[3, 7]);
        assert!(gt.neighbours(0)[2..].iter().all(|&i| i == u32::MAX));
    }

    #[test]
    fn an_id_the_corpus_does_not_hold_costs_that_point_not_the_run() {
        // The id set comes from what the engine says it holds; a point the
        // corpus lacks is skipped, and counted out of `n_matching`.
        let (dim, n, k) = (2usize, 10usize, 3usize);
        let base: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0]).collect();
        let queries = [0.0f32, 0.0];
        let gt = compute_filtered(
            Metric::Euclid,
            &base,
            n,
            &queries,
            1,
            dim,
            k,
            "oob",
            &[1u32, 99, 2],
        );
        assert_eq!(gt.n_matching, Some(2));
        assert_eq!(&gt.neighbours(0)[..2], &[1, 2]);
    }

    #[test]
    fn an_unfiltered_truth_writes_no_condition_and_old_files_still_load() {
        // 24 MiB of sift1m ground truth is already on disk without these
        // fields; a schema change that stopped reading it would recompute
        // every corpus.
        let gt = compute(
            Metric::Euclid,
            &[0.0, 0.0, 1.0, 1.0],
            2,
            &[0.0, 0.0],
            1,
            2,
            1,
        );
        let json = serde_json::to_string(&gt).expect("serialises");
        assert!(
            !json.contains("condition"),
            "no condition on an unfiltered truth"
        );
        assert!(!json.contains("n_matching"));
        let back: GroundTruth = serde_json::from_str(&json).expect("round-trips");
        assert!(back.condition.is_none() && back.n_matching.is_none());
    }
}

#[cfg(test)]
mod perf_equivalence_tests {
    use super::*;

    /// The partial sort must select the same k, in the same order, as the full
    /// sort it replaced. Both optimisations in this module are performance-only
    /// and this is what makes that claim checkable rather than asserted.
    fn top_k_reference(
        metric: Metric,
        base: &[f32],
        n: usize,
        dim: usize,
        query: &[f32],
        k: usize,
    ) -> Vec<Neighbour> {
        let mut all: Vec<Neighbour> = (0..n)
            .map(|i| Neighbour {
                id: i as u32,
                score: score(metric, query, &base[i * dim..(i + 1) * dim]),
            })
            .collect();
        let lower_better = metric.lower_is_better();
        all.sort_by(|a, b| {
            let ord = if lower_better {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            };
            ord.then(a.id.cmp(&b.id))
        });
        all.truncate(k);
        all
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    #[test]
    fn partial_sort_matches_full_sort_bit_for_bit() {
        let dim = 16;
        let n = 2000;
        let mut st = 0x1234_5678u64;
        let base: Vec<f32> = (0..n * dim).map(|_| lcg(&mut st)).collect();

        for metric in [
            Metric::Euclid,
            Metric::Dot,
            Metric::Cosine,
            Metric::Manhattan,
        ] {
            for k in [1usize, 10, 100, 999] {
                let q: Vec<f32> = (0..dim).map(|_| lcg(&mut st)).collect();
                let got = top_k(metric, &base, n, dim, &q, k);
                let want = top_k_reference(metric, &base, n, dim, &q, k);
                assert_eq!(got.len(), want.len(), "{metric:?} k={k}");
                for (g, w) in got.iter().zip(want.iter()) {
                    assert_eq!(g.id, w.id, "{metric:?} k={k}");
                    // Bit-for-bit, not approximately: the same f64 additions in
                    // the same order must produce the same bits.
                    assert_eq!(g.score.to_bits(), w.score.to_bits(), "{metric:?} k={k}");
                }
            }
        }
    }

    /// Heavy ties are where an unstable partial sort would diverge from a
    /// stable full sort if the comparator were not a total order.
    #[test]
    fn partial_sort_is_deterministic_under_massive_ties() {
        let dim = 4;
        let n = 500;
        // Only 5 distinct vectors, so every score has ~100 exact ties.
        let base: Vec<f32> = (0..n)
            .flat_map(|i| {
                let v = (i % 5) as f32;
                vec![v, v + 1.0, v + 2.0, v + 3.0]
            })
            .collect();
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let first = top_k(Metric::Euclid, &base, n, dim, &q, 50);
        let want = top_k_reference(Metric::Euclid, &base, n, dim, &q, 50);
        assert_eq!(
            first.iter().map(|n| n.id).collect::<Vec<_>>(),
            want.iter().map(|n| n.id).collect::<Vec<_>>(),
        );
        // And repeated runs agree with each other.
        for _ in 0..8 {
            let again = top_k(Metric::Euclid, &base, n, dim, &q, 50);
            assert_eq!(
                again.iter().map(|n| n.id).collect::<Vec<_>>(),
                first.iter().map(|n| n.id).collect::<Vec<_>>()
            );
        }
    }

    /// `compute` parallelises across queries; the ground truth must not depend
    /// on how many threads rayon happened to use.
    #[test]
    fn compute_is_independent_of_thread_count() {
        let dim = 8;
        let n = 400;
        let nq = 40;
        let mut st = 0xfeed_beefu64;
        let base: Vec<f32> = (0..n * dim).map(|_| lcg(&mut st)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| lcg(&mut st)).collect();

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| compute(Metric::Cosine, &base, n, &queries, nq, dim, 10))
        };
        let a = run(1);
        let b = run(7);
        assert_eq!(a.ids, b.ids);
        assert_eq!(
            a.scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            b.scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
        );
    }
}

#[cfg(test)]
mod gt_diff_tests {
    use super::*;

    fn gt(k: usize, rows: &[&[u32]]) -> GroundTruth {
        GroundTruth {
            metric: Metric::Euclid,
            n_base: 1000,
            n_queries: rows.len(),
            dim: 4,
            k,
            ids: rows.iter().flat_map(|r| r.iter().copied()).collect(),
            scores: rows.iter().flat_map(|r| r.iter().map(|_| 0.0)).collect(),
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        }
    }

    /// The distinction the whole diff turns on: a permuted tie run is not a
    /// disagreement about which vectors are nearest, and must not be counted as
    /// one. Without this the SIFT1M diff reports a convention mismatch as a
    /// correctness failure.
    #[test]
    fn reordered_ties_are_not_counted_as_interior_disagreements() {
        let ours = gt(4, &[&[10, 11, 12, 13]]);
        // Same four ids, first two swapped — a tie broken the other way.
        let published = [11u32, 10, 12, 13];
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.identical, 0);
        assert_eq!(d.same_set_different_order, 1);
        assert_eq!(
            d.interior, 0,
            "a permutation is not an interior disagreement"
        );
        assert_eq!(d.id_overlap, 4);
        assert_eq!(d.overlap_fraction(), 1.0);
    }

    #[test]
    fn a_genuinely_different_neighbour_is_still_flagged() {
        let ours = gt(4, &[&[10, 11, 12, 13]]);
        // Position 0 holds a vector we do not consider a neighbour at all.
        let published = [99u32, 11, 12, 13];
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.same_set_different_order, 0);
        assert_eq!(d.interior, 1);
        assert_eq!(d.id_overlap, 3);
    }

    #[test]
    fn identical_lists_report_perfect_overlap() {
        let ours = gt(3, &[&[1, 2, 3], &[4, 5, 6]]);
        let published = [1u32, 2, 3, 4, 5, 6];
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.identical, 2);
        assert_eq!(d.total_position_mismatches, 0);
        assert_eq!(d.overlap_fraction(), 1.0);
    }

    #[test]
    fn the_boundary_is_the_kth_rank_only() {
        // docs/ground-truth.md: "differ only at the k-th boundary". At k=100
        // the old band was 11 ranks wide, so a set difference at rank 90 was
        // filed as boundary. It is interior.
        let ids: Vec<u32> = (0..100).collect();
        let ours = gt(100, &[&ids]);
        let mut published = ids.clone();
        published[90] = 999;
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.interior, 1);
        assert_eq!(d.boundary_only, 0);

        let mut published = ids.clone();
        published[99] = 999;
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.boundary_only, 1);
        assert_eq!(d.interior, 0);
    }

    #[test]
    fn boundary_disagreement_is_separated_from_interior() {
        // k=10, so the boundary is the last position.
        let ours = gt(10, &[&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]);
        let published = [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 99];
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.boundary_only, 1);
        assert_eq!(d.interior, 0);
        assert_eq!(d.id_overlap, 9);
    }
}

#[cfg(test)]
mod gt_diff_classification_tests {
    use super::*;

    fn gt_row(k: usize, row: &[u32]) -> GroundTruth {
        GroundTruth {
            metric: Metric::Euclid,
            n_base: 1000,
            n_queries: 1,
            dim: 4,
            k,
            ids: row.to_vec(),
            scores: vec![0.0; k],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        }
    }

    /// The SIFT1M pattern, reduced: tied neighbours permuted early, plus one
    /// genuine set difference at the very last rank because the k-th distance
    /// was tied and one of the two had to be cut.
    ///
    /// Keyed on first *positional* mismatch this reports `interior` — the
    /// alarming bucket — when nothing interior disagrees at all. On the real
    /// dataset that mislabelled 63 of 10000 queries.
    #[test]
    fn early_tie_permutation_plus_boundary_cut_is_a_boundary_disagreement() {
        let k = 10;
        //  ours:      0 1 2 3 4 5 6 7 8 9
        //  published: 1 0 2 3 4 5 6 7 8 99   <- swap at rank 0, real diff at rank 9
        let ours = gt_row(k, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let published = [1u32, 0, 2, 3, 4, 5, 6, 7, 8, 99];

        let d = diff_against_published(&ours, &published);
        assert_eq!(d.interior, 0, "the only set difference is at the last rank");
        assert_eq!(d.boundary_only, 1);
        assert_eq!(d.id_overlap, 9);
    }

    /// The converse must still be caught: a set difference genuinely early is
    /// an interior disagreement even if everything after it lines up.
    #[test]
    fn an_early_set_difference_is_still_interior() {
        let k = 10;
        let ours = gt_row(k, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // Rank 1 holds a vector we do not consider a neighbour at all.
        let published = [0u32, 77, 2, 3, 4, 5, 6, 7, 8, 9];
        let d = diff_against_published(&ours, &published);
        assert_eq!(d.interior, 1);
        assert_eq!(d.boundary_only, 0);
    }

    #[test]
    fn parallel_ground_truth_matches_a_serial_computation() {
        // The oracle is the arbiter, so its answer must not depend on how many
        // threads happened to be available — and it now carries a reused
        // per-thread buffer, which is exactly the kind of state that could make
        // it. Compared against a straightforward serial loop rather than
        // against itself.
        let n = 400usize;
        let dim = 8usize;
        let nq = 32usize;
        let mut base = Vec::with_capacity(n * dim);
        for i in 0..n * dim {
            base.push(((i * 37 % 211) as f32) / 211.0 - 0.5);
        }
        let mut queries = Vec::with_capacity(nq * dim);
        for i in 0..nq * dim {
            queries.push(((i * 53 % 197) as f32) / 197.0 - 0.5);
        }

        for metric in [
            Metric::Euclid,
            Metric::Dot,
            Metric::Cosine,
            Metric::Manhattan,
        ] {
            let gt = compute(metric, &base, n, &queries, nq, dim, 10);

            let base_pre: Vec<f32> = if metric.normalises_at_ingest() {
                let mut out = Vec::with_capacity(base.len());
                for i in 0..n {
                    out.extend_from_slice(&preprocess(metric, &base[i * dim..(i + 1) * dim]));
                }
                out
            } else {
                base.clone()
            };
            for qi in 0..nq {
                let q = preprocess(metric, &queries[qi * dim..(qi + 1) * dim]);
                let want = top_k(metric, &base_pre, n, dim, &q, 10);
                let got_ids = gt.neighbours(qi);
                let got_scores = gt.scores_for(qi);
                for i in 0..10 {
                    assert_eq!(want[i].id, got_ids[i], "{metric:?} query {qi} position {i}");
                    assert_eq!(
                        want[i].score, got_scores[i],
                        "{metric:?} query {qi} position {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_reused_scratch_buffer_gives_the_same_answer_as_a_fresh_one() {
        let n = 200usize;
        let dim = 4usize;
        let base: Vec<f32> = (0..n * dim).map(|i| (i % 91) as f32).collect();
        let q: Vec<f32> = (0..dim).map(|i| (i * 7) as f32).collect();

        let mut scratch = Vec::new();
        // Run several unrelated queries through the same buffer first, so it
        // holds another query's scores when the one under test starts.
        for j in 0..5 {
            let other: Vec<f32> = (0..dim).map(|i| ((i + j) * 13) as f32).collect();
            let _ = top_k_into(Metric::Euclid, &base, n, dim, &other, 7, &mut scratch);
        }
        let reused = top_k_into(Metric::Euclid, &base, n, dim, &q, 7, &mut scratch);
        let fresh = top_k(Metric::Euclid, &base, n, dim, &q, 7);
        assert_eq!(reused, fresh);
    }
}
