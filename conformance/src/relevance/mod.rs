//! §4.4 — relevance metrics, and §8.5 T3's statistical machinery.
//!
//! §4.4 draws a distinction the rest of this file exists to keep:
//!
//! | | geometric relevance | semantic relevance |
//! |---|---|---|
//! | question | did the index find the true nearest neighbours? | did the system retrieve the documents a human considers relevant? |
//! | ground truth | exact k-NN under the metric (fp64 oracle) | qrels / human judgements |
//! | metrics | **recall@k** (k ∈ {1, 10, 100}), mean relative distance error | **nDCG@10**, MRR |
//! | what it measures | index quality — the engine's own responsibility | the whole pipeline, of which the engine is one part |
//!
//! > "Report both; never substitute one for the other."
//!
//! And on why recall alone is not enough:
//!
//! > "Recall@10 is the primary comparison axis and the thing QPS is normalised
//! > against. But it's brittle near ties, so it is always reported alongside
//! > **mean relative distance error** — the average ratio between returned and
//! > ideal distances — which degrades smoothly and distinguishes 'missed the
//! > true neighbour by a hair' from 'returned something unrelated.'"

use crate::differ::tolerance::Epsilon;
use crate::oracle::{GroundTruth, Judgement, Rescorer};

/// One query's returned results, as they came off the wire.
#[derive(Clone, Debug)]
pub struct Returned {
    pub ids: Vec<u32>,
    pub scores: Vec<f64>,
}

/// A 95% confidence interval on a recall.
///
/// Named because `(f64, f64)` does not say which end is which, and the one
/// place it is read decides whether *any* throughput comparison is licensed:
/// `recall_is_matched` destructured two anonymous pairs and tested
/// `a_lo <= b_hi && b_lo <= a_hi`. Transposing either destructuring, or
/// building a pair backwards anywhere, inverts that test silently — it would
/// license the comparisons it exists to refuse, and look normal doing it.
///
/// Not the wire format: the recall JSON carries `recall_at_10_ci95_low` and
/// `..._high` from an explicit struct in `main`, so this is an in-memory type.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Ci95 {
    pub low: f64,
    pub high: f64,
}

impl Ci95 {
    /// No interval: the recall it would bracket was never measured. NaN rather
    /// than `[0, 0.79]`, which is what a Wilson interval on zero trials looks
    /// like and reads as a real bound around a number nobody took.
    pub const UNMEASURED: Ci95 = Ci95 {
        low: f64::NAN,
        high: f64::NAN,
    };

    /// §7.4's "statistically indistinguishable": the precondition for
    /// comparing two engines' throughput at all. One expression, one place.
    pub fn overlaps(&self, other: &Ci95) -> bool {
        self.low <= other.high && other.low <= self.high
    }
}

/// §4.4's geometric metrics for one run.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Geometric {
    pub queries: usize,
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    /// §4.4's "mean relative distance error".
    pub mean_relative_distance_error: f64,
    /// §7.4: "Publish QPS-vs-recall curves; a single QPS number without its
    /// recall is meaningless and this project should never emit one." The
    /// confidence interval is what makes "equal recall" checkable rather than
    /// asserted.
    pub recall_at_10_ci95: Ci95,
    /// Queries for which the engine returned fewer results than were asked
    /// for (and than the collection could supply). Their missing slots are
    /// misses in every recall@k, and they are excluded from MRDE when they
    /// hold nothing to compare — so the count is printed rather than hidden.
    pub short_lists: usize,
    /// Returned points whose reported score beats the k-th truth while the id
    /// is not in the truth: impossible for a correct engine, counted as misses
    /// (`Judgement::ImpossibleScore`).
    pub impossible_scores: usize,
    /// Returned ids that are not rows of the base file, seen only when a
    /// `Rescorer` was supplied. §4.3 makes ids row indices; these are misses.
    pub unknown_ids: usize,
    /// docs/workloads.md W12 point 4: "The row reports how many points came
    /// back, and how many could have."
    ///
    /// `returned` is summed over the queries; `asked` is what was requested of
    /// them, already capped by what the corpus — or, under a condition, the
    /// matching set — could supply. A filtered page shorter than `k` is
    /// ambiguous between an index that searched too narrowly and a filter that
    /// simply matches fewer than `k` points, and recall alone cannot separate
    /// them; this pair is what does.
    pub returned: usize,
    pub asked: usize,
    /// How many points satisfied the condition, from the truth. `None` for an
    /// unfiltered sweep, where the answer is the whole corpus.
    pub n_matching: Option<usize>,
}

impl Geometric {
    pub fn describe(&self) -> String {
        // NaN prints as "n/a": the request could not produce that recall, and
        // a number would misrepresent a limit as a measurement.
        let fmt = |v: f64| {
            if v.is_nan() {
                "n/a".to_string()
            } else {
                format!("{v:.4}")
            }
        };
        let mut s = format!(
            "n={} recall@1={} recall@10={} [{},{}] recall@100={} MRDE={:.4e}",
            self.queries,
            fmt(self.recall_at_1),
            fmt(self.recall_at_10),
            fmt(self.recall_at_10_ci95.low),
            fmt(self.recall_at_10_ci95.high),
            fmt(self.recall_at_100),
            self.mean_relative_distance_error
        );
        if self.short_lists > 0 {
            s.push_str(&format!(" short_lists={}", self.short_lists));
        }
        if self.impossible_scores > 0 {
            s.push_str(&format!(" IMPOSSIBLE_SCORES={}", self.impossible_scores));
        }
        if self.unknown_ids > 0 {
            s.push_str(&format!(" UNKNOWN_IDS={}", self.unknown_ids));
        }
        s
    }
}

/// What one query contributed, beyond its recall fractions.
#[derive(Default)]
struct Tally {
    impossible_scores: usize,
    unknown_ids: usize,
}

/// Recall@k for one query, using §4.3's boundary-tie rule.
///
/// Returns `NaN` when the request could not have produced a meaningful value —
/// specifically when fewer than `k` results were **asked for** (`limit < k`),
/// or when ground truth holds fewer than `k` neighbours. A run with
/// `limit = 10` mechanically caps recall@100 at 0.1, and printing "0.1000"
/// invites reading a request parameter as a quality result. §7.4's insistence
/// that "a single QPS number without its recall is meaningless" cuts both
/// ways: a recall number that measures the limit rather than the engine is
/// worse than no number.
///
/// It is **not** `NaN` when `limit >= k` and the engine *returned* fewer than
/// `k`: that is the engine's doing, and every slot it left empty is a miss.
/// The earlier rule keyed on `returned.ids.len() < k` and so dropped exactly
/// the queries an engine answered worst from the mean, flattering it.
///
/// A returned id counts once. An engine repeating its best hit `k` times has
/// found one neighbour, not `k`.
#[allow(clippy::too_many_arguments)]
fn recall_one(
    gt: &GroundTruth,
    qi: usize,
    returned: &Returned,
    k: usize,
    eps: &Epsilon,
    limit: usize,
    rescorer: Option<&Rescorer>,
    tally: &mut Tally,
) -> f64 {
    if k == 0 || k > gt.k || limit < k {
        return f64::NAN;
    }
    // §4.3's boundary-tie clause compares against the k-th truth score, so that
    // score is the magnitude the tolerance is read at. Passing `eps.value`
    // straight through treated a *relative* ε as an absolute score distance:
    // at SIFT1M euclid that is 4.172e-7 against distances of ~300, so no tie
    // was ever within tolerance and the clause never fired. §4.3 exists because
    // "strict ID matching understates recall on any dataset containing
    // duplicate or near-duplicate vectors, which is most real ones" — and SIFT
    // descriptors are quantized integers with known duplicates, so this
    // understated recall on precisely the data it was written for.
    // docs/workloads.md W12 point 4: "recall is scored against
    // `min(k, |matching set|)`". A condition matching 6 points cannot produce
    // 10 neighbours, and an engine returning all 6 is exactly right -- scored
    // against `k` it reads 0.6, and the §7.4 frontier shows a regression that
    // is the filter's doing rather than the engine's. `None` is §4.3's
    // unfiltered truth, whose matching set is the whole corpus.
    let available = gt.n_matching.map_or(k, |m| k.min(m));
    if available == 0 {
        return f64::NAN;
    }
    // Read at the *available* k-th rather than the requested one: a filtered
    // truth shorter than `k` pads its tail with NaN, and a NaN tolerance judges
    // every tie a miss.
    let kth = gt.scores_for(qi)[available.min(gt.k) - 1];
    let tol = eps.absolute_at(kth);
    let mut hits = 0usize;
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::with_capacity(k);
    for i in 0..k.min(returned.ids.len()) {
        let id = returned.ids[i];
        if !seen.insert(id) {
            continue;
        }
        // The oracle's own score for this point when the base vectors are to
        // hand, so the tie clause does not take the engine's word for it.
        let oracle_score = match rescorer {
            Some(r) => match r.score(qi, id) {
                Some(s) => Some(s),
                None => {
                    tally.unknown_ids += 1;
                    continue;
                }
            },
            None => None,
        };
        // `available`, not `k`: the truth prefix a filtered condition actually
        // has. Identical for the unfiltered case, where they are the same
        // number.
        match gt.judge_at(qi, available, id, returned.scores[i], oracle_score, tol) {
            Judgement::Correct => hits += 1,
            Judgement::Miss => {}
            Judgement::ImpossibleScore => tally.impossible_scores += 1,
        }
    }
    hits as f64 / available as f64
}

/// Mean relative distance error for one query.
///
/// §4.4: "the average ratio between returned and ideal distances". Defined so
/// that a perfect result is 0 and a result whose distances are twice the ideal
/// is 1.0, for either ordering convention.
///
/// `None` when nothing is comparable — an empty list, or one whose scores are
/// all non-finite. This returned `0.0`, the *best possible* value, for an
/// engine that returned nothing at all; the caller now excludes such queries
/// from the mean and reports how many there were (`Geometric::short_lists`).
fn mrde_one(gt: &GroundTruth, qi: usize, returned: &Returned, k: usize) -> Option<f64> {
    let k = k.min(gt.k).min(returned.ids.len());
    if k == 0 {
        return None;
    }
    let ideal = gt.scores_for(qi);
    let lower_better = gt.metric.lower_is_better();

    let mut acc = 0.0f64;
    let mut n = 0usize;
    for (&got, &want) in returned.scores.iter().zip(ideal.iter()).take(k) {
        if !want.is_finite() || !got.is_finite() {
            continue;
        }
        let e = if lower_better {
            // Distances: how much further than ideal, as a ratio.
            if want.abs() < 1e-30 {
                // Ideal distance is zero (an exact duplicate). Any non-zero
                // returned distance is infinitely worse in ratio terms, so fall
                // back to the absolute difference rather than emitting inf.
                got.abs()
            } else {
                (got - want).abs() / want.abs()
            }
        } else {
            // Similarities: how much less similar than ideal.
            let scale = want.abs().max(1e-30);
            (want - got).abs() / scale
        };
        acc += e;
        n += 1;
    }
    if n == 0 { None } else { Some(acc / n as f64) }
}

/// Wilson score interval for a proportion.
///
/// §8.5 T3 asks for "recall and mean relative distance error for both engines
/// over an `ef` sweep, thousands of queries, **with confidence intervals**",
/// and the publishable statement is "backed by overlapping CIs at matched
/// recall".
///
/// Wilson rather than the normal approximation because recall is often near 1,
/// where the normal interval produces upper bounds above 1 and is badly wrong
/// exactly where the interesting comparisons happen.
pub fn wilson_interval(successes: f64, trials: usize) -> Ci95 {
    if trials == 0 {
        return Ci95 {
            low: 0.0,
            high: 0.0,
        };
    }
    let n = trials as f64;
    let p = successes / n;
    let z = 1.959_963_984_540_054f64; // 95%
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
    Ci95 {
        low: (centre - margin).max(0.0),
        high: (centre + margin).min(1.0),
    }
}

/// Evaluate a full run against ground truth.
///
/// `limit` is what every query asked for; it decides which recall depths are
/// measurable at all (see `recall_one`). `rescorer`, when the caller holds the
/// base vectors, lets the tie clause use the oracle's fp64 score for each
/// returned point rather than the engine's reported one.
pub fn evaluate(
    gt: &GroundTruth,
    results: &[Returned],
    eps: &Epsilon,
    limit: usize,
    rescorer: Option<&Rescorer>,
) -> Geometric {
    let n = results.len().min(gt.n_queries);
    if n == 0 {
        // Nothing measured is not recall 0 with a CI of (0, 0): that value
        // *overlaps itself*, and `recall_is_matched` then licensed a QPS
        // comparison on zero queries. Every figure is NaN — "not measured" —
        // which the match test refuses.
        return Geometric {
            recall_at_1: f64::NAN,
            recall_at_10: f64::NAN,
            recall_at_100: f64::NAN,
            mean_relative_distance_error: f64::NAN,
            recall_at_10_ci95: Ci95::UNMEASURED,
            ..Geometric::default()
        };
    }

    // Each k is averaged over the queries where it was *measurable*, and stays
    // NaN when none were.
    let mut acc = [0.0f64; 3];
    let mut cnt = [0usize; 3];
    let mut mrde = 0.0;
    let mut mrde_n = 0usize;
    let mut r10_successes = 0.0;
    let mut tally = Tally::default();
    // Capped by the matching set where there is one: a condition matching 6
    // points cannot fill a 10-slot page, and counting every such query as a
    // "short list" reports the filter's selectivity as though it were the
    // engine answering badly.
    let expected_len = limit.min(gt.n_matching.unwrap_or(gt.n_base));
    let mut returned_total = 0usize;
    let mut short_lists = 0usize;

    for (qi, r) in results.iter().enumerate().take(n) {
        returned_total += r.ids.len().min(limit);
        if r.ids.len() < expected_len {
            short_lists += 1;
        }
        for (slot, k) in [(0usize, 1usize), (1, 10), (2, 100)] {
            let v = recall_one(gt, qi, r, k, eps, limit, rescorer, &mut tally);
            if !v.is_nan() {
                acc[slot] += v;
                cnt[slot] += 1;
                if slot == 1 {
                    r10_successes += v;
                }
            }
        }
        if let Some(m) = mrde_one(gt, qi, r, 10) {
            mrde += m;
            mrde_n += 1;
        }
    }

    let mean = |slot: usize| -> f64 {
        if cnt[slot] == 0 {
            f64::NAN
        } else {
            acc[slot] / cnt[slot] as f64
        }
    };

    Geometric {
        queries: n,
        recall_at_1: mean(0),
        recall_at_10: mean(1),
        recall_at_100: mean(2),
        // Over the queries with something to compare; `NaN` if none had.
        mean_relative_distance_error: if mrde_n == 0 {
            f64::NAN
        } else {
            mrde / mrde_n as f64
        },
        // The interval is over the queries where recall@10 was measurable.
        //
        // `r10_successes` is a sum of per-query recall *fractions*, not a count
        // of successes, so this is Wilson applied to a quantity it was not
        // derived for. The direction is safe: a variable bounded in [0, 1] has
        // variance at most p(1-p), which is exactly what Wilson assumes, so the
        // interval is at least as wide as the truth. It therefore errs toward
        // refusing a QPS comparison rather than licensing one, which is the
        // right way for §7.4's gate to be wrong. Stated because T3's whole job
        // is statistical rigour and an undocumented approximation there is
        // indistinguishable from an error.
        // NaN, like the recall it brackets, when no query could measure it:
        // `wilson_interval(0, 1)` is [0, 0.79], a real-looking interval
        // around a number that was never taken.
        recall_at_10_ci95: if cnt[1] == 0 {
            Ci95::UNMEASURED
        } else {
            wilson_interval(r10_successes, cnt[1])
        },
        short_lists,
        returned: returned_total,
        asked: n * expected_len,
        n_matching: gt.n_matching,
        impossible_scores: tally.impossible_scores,
        unknown_ids: tally.unknown_ids,
    }
}

/// §7.4 / §8.5 T3: "The publishable statement is *'equal recall at lower
/// latency'*, backed by overlapping CIs at matched recall — never a raw QPS
/// ratio."
///
/// Returns true when the two recalls are statistically indistinguishable, which
/// is the precondition for comparing their throughput at all.
pub fn recall_is_matched(a: &Geometric, b: &Geometric) -> bool {
    // An unmeasurable recall cannot be "matched". Saying it is would license a
    // QPS comparison on the strength of a number that was never taken.
    if a.recall_at_10.is_nan() || b.recall_at_10.is_nan() {
        return false;
    }
    a.recall_at_10_ci95.overlaps(&b.recall_at_10_ci95)
}

// =========================================================================
// §4.4 — semantic relevance
// =========================================================================

/// A graded relevance judgement, from a qrels file.
#[derive(Clone, Debug)]
pub struct Qrel {
    pub query: usize,
    pub doc: u32,
    pub grade: u8,
}

/// nDCG@k over graded judgements.
///
/// §4.4: "Semantic relevance answers the question Qdrant users actually have,
/// and almost nobody publishes it: *does aggressive quantization cost anything
/// that matters?* A drop from recall@10 = 0.98 to 0.93 may move nDCG@10 by a
/// rounding error, or may not. Measuring both on the BEIR slice turns 'binary
/// quantization is 8× faster at 93% recall' into a statement someone can make
/// a decision from."
/// Reachable only through `evaluate_semantic`; see its note on qrels.
#[allow(dead_code)]
pub fn ndcg_at_k(qrels: &[Qrel], query: usize, returned: &[u32], k: usize) -> f64 {
    let grade_of = |doc: u32| -> f64 {
        qrels
            .iter()
            .find(|q| q.query == query && q.doc == doc)
            .map(|q| f64::from(q.grade))
            .unwrap_or(0.0)
    };

    let mut dcg = 0.0f64;
    for (i, &doc) in returned.iter().take(k).enumerate() {
        let g = grade_of(doc);
        if g > 0.0 {
            // Standard graded DCG: (2^g - 1) / log2(i + 2).
            dcg += (2f64.powf(g) - 1.0) / ((i + 2) as f64).log2();
        }
    }

    // Ideal DCG: the same judgements, best-first.
    let mut ideal: Vec<f64> = qrels
        .iter()
        .filter(|q| q.query == query)
        .map(|q| f64::from(q.grade))
        .collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut idcg = 0.0f64;
    for (i, g) in ideal.iter().take(k).enumerate() {
        if *g > 0.0 {
            idcg += (2f64.powf(*g) - 1.0) / ((i + 2) as f64).log2();
        }
    }

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Mean reciprocal rank of the first relevant result.
/// Reachable only through `evaluate_semantic`; see its note on qrels.
#[allow(dead_code)]
pub fn mrr(qrels: &[Qrel], query: usize, returned: &[u32], k: usize) -> f64 {
    for (i, &doc) in returned.iter().take(k).enumerate() {
        let relevant = qrels
            .iter()
            .any(|q| q.query == query && q.doc == doc && q.grade > 0);
        if relevant {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Semantic {
    pub queries: usize,
    pub ndcg_at_10: f64,
    pub mrr_at_10: f64,
}

/// **Not reachable from any command yet**, and the reason is the data rather
/// than the code: semantic relevance needs qrels — human judgements — and none
/// of §4.2's dataset tiers ship them. §4.4's rule is "report both; never
/// substitute one for the other", so this stays implemented and tested against
/// the day a judged dataset is added, rather than being deleted and rebuilt
/// wrongly later. Until then, no run emits nDCG or MRR and nothing should
/// advertise that it does.
#[allow(dead_code)]
pub fn evaluate_semantic(qrels: &[Qrel], results: &[Returned], n_queries: usize) -> Semantic {
    if n_queries == 0 {
        return Semantic::default();
    }
    let mut ndcg = 0.0;
    let mut rr = 0.0;
    for (qi, r) in results.iter().enumerate().take(n_queries) {
        ndcg += ndcg_at_k(qrels, qi, &r.ids, 10);
        rr += mrr(qrels, qi, &r.ids, 10);
    }
    let n = n_queries.min(results.len()) as f64;
    Semantic {
        queries: n as usize,
        ndcg_at_10: ndcg / n,
        mrr_at_10: rr / n,
    }
}

/// §4.1's guard, enforced by the harness rather than by convention.
///
/// > "harness rejects relevance runs with `--offset != 0`, `--uuids`, or
/// > `--max-id` set"
///
/// and §4.3:
///
/// > "**ID alignment is a hard requirement.** Ground truth indexes into
/// > base-file row order, so point ID must equal row index... Any run violating
/// > this is rejected by the harness rather than silently producing meaningless
/// > recall."
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub offset: u64,
    pub uuids: bool,
    pub max_id: Option<u64>,
}

pub fn validate_relevance_run(cfg: &RunConfig) -> Result<(), String> {
    if cfg.offset != 0 {
        return Err(format!(
            "relevance runs require --offset 0 (got {}): ground truth indexes into base-file row order (§4.3)",
            cfg.offset
        ));
    }
    if cfg.uuids {
        return Err(
            "relevance runs cannot use --uuids: point ID must equal base-file row index (§4.3)"
                .into(),
        );
    }
    if cfg.max_id.is_some() {
        return Err(
            "relevance runs cannot use --max-id: overwriting IDs breaks the row-index correspondence (§4.3)"
                .into(),
        );
    }
    Ok(())
}

/// W12 point 3's guard: the sweep's condition and its ground truth's must be
/// the same one.
///
/// Both directions are wrong in the same way and for opposite reasons. An
/// unfiltered search judged against a restricted truth misses every neighbour
/// the filter removed; a filtered search judged against §4.3's unfiltered cache
/// misses every neighbour the filter kept out of the whole-collection top k.
/// Either produces a number shaped exactly like a recall, which is why this
/// refuses rather than warns -- nothing downstream can tell the difference.
///
/// A pure function beside `validate_relevance_run` so the refusals are tested
/// without an engine, a corpus or a tokio runtime.
pub fn validate_condition(
    asked: Option<&str>,
    truth: Option<&str>,
    quantized: bool,
) -> Result<(), String> {
    match (asked, truth) {
        (Some(a), Some(t)) if a != t => {
            return Err(format!(
                "this sweep filters on `{a}` and the ground truth is restricted to \
                 `{t}` — the two are different questions (W12 point 3)"
            ));
        }
        (Some(a), None) => {
            return Err(format!(
                "--filter-values asks for `{a}` but the ground truth is §4.3's \
                 *unfiltered* cache: the neighbours of a query among all points are \
                 not its neighbours among the points the filter keeps. Build one \
                 with `conformance filtered-truth`."
            ));
        }
        (None, Some(t)) => {
            return Err(format!(
                "the ground truth is restricted to `{t}` but this sweep sends no \
                 filter, so it would search the whole collection and be scored \
                 against a fraction of it. Pass --filter-values, or use an \
                 unfiltered ground truth."
            ));
        }
        _ => {}
    }
    if asked.is_some() && quantized {
        return Err("a filtered sweep of a quantized collection is not implemented: \
                    the restricted truth is fp32, and what the rescore pool contains \
                    under a filter is a second question. Sweep the fp32 collection, \
                    or drop --filter-values."
            .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sweep_and_its_truth_must_share_one_condition() {
        // W12 point 3. The pairs that are fine:
        assert!(validate_condition(None, None, false).is_ok());
        assert!(validate_condition(None, None, true).is_ok(), "quantized, unfiltered");
        assert!(validate_condition(Some("a in {k0}"), Some("a in {k0}"), false).is_ok());

        // An unfiltered search against a restricted truth: every neighbour the
        // filter removed reads as a miss.
        let e = validate_condition(None, Some("a in {k0}"), false).unwrap_err();
        assert!(e.contains("sends no filter"), "{e}");

        // A filtered search against §4.3's cache: every neighbour the filter
        // kept out of the whole-collection top k reads as a miss.
        let e = validate_condition(Some("a in {k0}"), None, false).unwrap_err();
        assert!(e.contains("unfiltered"), "{e}");

        // Two different conditions is the same error as either half missing.
        let e = validate_condition(Some("a in {k0}"), Some("a in {k1}"), false).unwrap_err();
        assert!(e.contains("different questions"), "{e}");

        // And the combination that is merely unbuilt, refused rather than
        // guessed at.
        let e = validate_condition(Some("a in {k0}"), Some("a in {k0}"), true).unwrap_err();
        assert!(e.contains("quantized"), "{e}");
    }

    /// An absolute tolerance of `v`, which is what these tests mean when they
    /// pass a bare number: they use synthetic scores around 1.0, where absolute
    /// and relative coincide closely enough not to be the subject.
    pub(super) fn abs_eps(v: f64) -> Epsilon {
        let mut e = crate::differ::tolerance::calibrate(
            crate::oracle::Metric::Euclid,
            8,
            Default::default(),
            Default::default(),
            1.0,
        );
        e.value = v;
        e.relative = false;
        e
    }
    use crate::oracle::Metric;

    fn gt(metric: Metric, k: usize, ids: Vec<u32>, scores: Vec<f64>, nq: usize) -> GroundTruth {
        GroundTruth {
            metric,
            n_base: 1000,
            n_queries: nq,
            dim: 8,
            k,
            ids,
            scores,
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        }
    }

    #[test]
    fn perfect_results_score_perfect_recall_and_zero_mrde() {
        let g = gt(Metric::Euclid, 3, vec![1, 2, 3], vec![0.1, 0.2, 0.3], 1);
        let r = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![0.1, 0.2, 0.3],
        }];
        let out = evaluate(&g, &r, &abs_eps(1e-9), 3, None);
        assert_eq!(out.recall_at_1, 1.0);
        assert!(out.mean_relative_distance_error.abs() < 1e-12);
    }

    #[test]
    fn missing_every_neighbour_scores_zero_recall() {
        let g = gt(Metric::Euclid, 3, vec![1, 2, 3], vec![0.1, 0.2, 0.3], 1);
        let r = vec![Returned {
            ids: vec![90, 91, 92],
            scores: vec![9.0, 9.1, 9.2],
        }];
        let out = evaluate(&g, &r, &abs_eps(1e-9), 3, None);
        assert_eq!(out.recall_at_1, 0.0);
        // And MRDE is large, which is the signal §4.4 wants: "returned
        // something unrelated" rather than "missed by a hair".
        assert!(out.mean_relative_distance_error > 10.0);
    }

    #[test]
    fn mrde_distinguishes_a_near_miss_from_an_unrelated_result() {
        // §4.4's whole reason for reporting MRDE alongside recall: both of
        // these score recall@1 = 0, but they are not the same failure.
        let g = gt(Metric::Euclid, 1, vec![1], vec![1.0], 1);

        let near = vec![Returned {
            ids: vec![99],
            scores: vec![1.001],
        }];
        let far = vec![Returned {
            ids: vec![99],
            scores: vec![50.0],
        }];

        let a = evaluate(&g, &near, &abs_eps(1e-9), 1, None);
        let b = evaluate(&g, &far, &abs_eps(1e-9), 1, None);
        assert_eq!(a.recall_at_1, 0.0);
        assert_eq!(b.recall_at_1, 0.0);
        assert!(a.mean_relative_distance_error < 0.01);
        assert!(b.mean_relative_distance_error > 10.0);
    }

    #[test]
    fn boundary_ties_count_as_correct() {
        // §4.3's duplicate-vector rule, through the recall path.
        let g = gt(Metric::Euclid, 2, vec![1, 2], vec![0.5, 0.5], 1);
        // Point 7 is a duplicate of point 2: same distance, different id.
        let r = vec![Returned {
            ids: vec![1, 7],
            scores: vec![0.5, 0.5],
        }];
        let out = evaluate(&g, &r, &abs_eps(1e-6), 2, None);
        assert_eq!(out.recall_at_1, 1.0);
        // Strict id matching would have scored the second position a miss.
        let strict = recall_one(
            &g,
            0,
            &r[0],
            2,
            &abs_eps(0.0),
            2,
            None,
            &mut Tally::default(),
        );
        assert!(strict >= 0.5);
    }

    #[test]
    fn wilson_interval_stays_inside_zero_one_near_the_edges() {
        // The reason for Wilson over the normal approximation: at recall near 1
        // the normal interval exceeds 1, exactly where the comparisons matter.
        let Ci95 { low: lo, high: hi } = wilson_interval(1000.0, 1000);
        assert!(lo > 0.99 && hi <= 1.0);
        let Ci95 { low: lo, high: hi } = wilson_interval(0.0, 1000);
        assert!(lo >= 0.0 && hi < 0.01);
        let Ci95 { low: lo, high: hi } = wilson_interval(500.0, 1000);
        assert!(lo < 0.5 && hi > 0.5);
    }

    #[test]
    fn ci_overlap_is_symmetric_and_ordered() {
        // The rule `recall_is_matched` rests on, tested as itself rather than
        // through two engines. Overlap is symmetric; a gap is a gap either way.
        let a = Ci95 {
            low: 0.90,
            high: 0.94,
        };
        let b = Ci95 {
            low: 0.93,
            high: 0.97,
        };
        let far = Ci95 {
            low: 0.60,
            high: 0.64,
        };
        assert!(a.overlaps(&b) && b.overlaps(&a));
        assert!(!a.overlaps(&far) && !far.overlaps(&a));
        // Touching at a single point counts: the bounds are inclusive.
        assert!(a.overlaps(&Ci95 {
            low: 0.94,
            high: 0.99
        }));
        // An unmeasured interval overlaps nothing, NaN comparing false
        // throughout. `recall_is_matched` also refuses on the NaN recall, so
        // this is the second of two guards rather than the only one.
        assert!(!Ci95::UNMEASURED.overlaps(&a));
        assert!(!a.overlaps(&Ci95::UNMEASURED));
    }

    #[test]
    fn wilson_interval_narrows_with_more_queries() {
        let Ci95 {
            low: lo_small,
            high: hi_small,
        } = wilson_interval(90.0, 100);
        let Ci95 {
            low: lo_big,
            high: hi_big,
        } = wilson_interval(9000.0, 10000);
        assert!((hi_big - lo_big) < (hi_small - lo_small));
    }

    #[test]
    fn matched_recall_requires_overlapping_intervals() {
        // §7.4: QPS is only comparable at equal recall.
        let a = Geometric {
            recall_at_10_ci95: Ci95 {
                low: 0.90,
                high: 0.94,
            },
            ..Default::default()
        };
        let b = Geometric {
            recall_at_10_ci95: Ci95 {
                low: 0.93,
                high: 0.97,
            },
            ..Default::default()
        };
        let c = Geometric {
            recall_at_10_ci95: Ci95 {
                low: 0.60,
                high: 0.64,
            },
            ..Default::default()
        };
        assert!(recall_is_matched(&a, &b));
        assert!(!recall_is_matched(&a, &c));
    }

    #[test]
    fn ndcg_rewards_putting_relevant_documents_first() {
        let qrels = vec![
            Qrel {
                query: 0,
                doc: 5,
                grade: 3,
            },
            Qrel {
                query: 0,
                doc: 6,
                grade: 1,
            },
        ];
        let best = ndcg_at_k(&qrels, 0, &[5, 6, 7], 10);
        let worse = ndcg_at_k(&qrels, 0, &[7, 6, 5], 10);
        assert!((best - 1.0).abs() < 1e-12);
        assert!(worse < best);
        assert!(worse > 0.0);
    }

    #[test]
    fn ndcg_is_zero_when_nothing_relevant_is_returned() {
        let qrels = vec![Qrel {
            query: 0,
            doc: 5,
            grade: 3,
        }];
        assert_eq!(ndcg_at_k(&qrels, 0, &[1, 2, 3], 10), 0.0);
    }

    #[test]
    fn ndcg_is_zero_when_there_are_no_judgements() {
        // A query with no qrels must not divide by zero.
        assert_eq!(ndcg_at_k(&[], 0, &[1, 2, 3], 10), 0.0);
    }

    #[test]
    fn mrr_is_the_reciprocal_of_the_first_relevant_rank() {
        let qrels = vec![Qrel {
            query: 0,
            doc: 9,
            grade: 1,
        }];
        assert_eq!(mrr(&qrels, 0, &[1, 2, 9], 10), 1.0 / 3.0);
        assert_eq!(mrr(&qrels, 0, &[9], 10), 1.0);
        assert_eq!(mrr(&qrels, 0, &[1, 2, 3], 10), 0.0);
    }

    #[test]
    fn relevance_runs_reject_the_configurations_that_break_id_alignment() {
        // §4.1 / §4.3: the harness rejects rather than silently producing
        // meaningless recall.
        assert!(
            validate_relevance_run(&RunConfig {
                offset: 0,
                uuids: false,
                max_id: None
            })
            .is_ok()
        );

        let e = validate_relevance_run(&RunConfig {
            offset: 100,
            uuids: false,
            max_id: None,
        })
        .unwrap_err();
        assert!(e.contains("offset"));

        let e = validate_relevance_run(&RunConfig {
            offset: 0,
            uuids: true,
            max_id: None,
        })
        .unwrap_err();
        assert!(e.contains("uuids"));

        let e = validate_relevance_run(&RunConfig {
            offset: 0,
            uuids: false,
            max_id: Some(50),
        })
        .unwrap_err();
        assert!(e.contains("max-id"));
    }
}

#[cfg(test)]
mod recall_depth_tests {
    use super::tests::abs_eps;
    use super::*;
    use crate::oracle::{GroundTruth, Metric};

    /// Ground truth cached at k=100, queried at k=10.
    fn gt100() -> GroundTruth {
        GroundTruth {
            metric: Metric::Euclid,
            n_base: 1000,
            n_queries: 1,
            dim: 4,
            k: 100,
            // Truth is ids 0..100, in order, at increasing distance.
            ids: (0..100u32).collect(),
            scores: (0..100).map(f64::from).collect(),
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        }
    }

    /// A truth restricted to 2 points, for W12 point 4's denominator.
    fn gt_matching(n_matching: usize) -> GroundTruth {
        let mut gt = gt100();
        // The restricted scan pads past the matching set, exactly as
        // `oracle::compute_filtered` does.
        gt.ids = (0..n_matching as u32)
            .chain(std::iter::repeat_n(u32::MAX, 100 - n_matching))
            .collect();
        gt.scores = (0..n_matching)
            .map(|i| i as f64)
            .chain(std::iter::repeat_n(f64::NAN, 100 - n_matching))
            .collect();
        gt.condition = Some("a in {keyword_0}".to_string());
        gt.n_matching = Some(n_matching);
        gt
    }

    #[test]
    fn a_filtered_truth_is_scored_against_min_k_and_the_matching_set() {
        // docs/workloads.md W12 point 4: an engine that returns all 2 points a
        // condition matched is exactly right. Against `k` that reads 0.2 and
        // the §7.4 frontier shows a regression the filter caused, not the
        // engine.
        let gt = gt_matching(2);
        let all_of_them = Returned {
            ids: vec![0, 1],
            scores: vec![0.0, 1.0],
        };
        let out = evaluate(&gt, &[all_of_them], &abs_eps(1e-9), 10, None);
        assert_eq!(
            out.recall_at_10, 1.0,
            "2 of the 2 points that could match is recall 1, not 0.2"
        );

        // And a miss inside the matching set is still a miss: the denominator
        // moved, the scoring did not become lenient.
        let one_wrong = Returned {
            ids: vec![0, 900],
            scores: vec![0.0, 900.0],
        };
        let out = evaluate(&gt, &[one_wrong], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 0.5);
    }

    #[test]
    fn a_full_page_under_a_narrow_filter_is_not_a_short_list() {
        // W12 point 4's ambiguity, in the counter that used to create it: with
        // `expected_len` capped only by the corpus, a condition matching 2
        // points made every query a "short list" -- reporting the filter's
        // selectivity as though the engine had answered badly.
        let gt = gt_matching(2);
        let full = Returned {
            ids: vec![0, 1],
            scores: vec![0.0, 1.0],
        };
        let out = evaluate(&gt, &[full], &abs_eps(1e-9), 10, None);
        assert_eq!(out.short_lists, 0, "2 of the 2 that exist is a full page");
        assert_eq!(out.returned, 2);
        assert_eq!(out.asked, 2, "asked is what could have come back, not `k`");
        assert_eq!(out.n_matching, Some(2));

        // Genuinely short under the same condition: one of the two missing.
        let one = Returned {
            ids: vec![0],
            scores: vec![0.0],
        };
        let out = evaluate(&gt, &[one], &abs_eps(1e-9), 10, None);
        assert_eq!(out.short_lists, 1);
        assert_eq!(out.returned, 1);
        assert_eq!(out.asked, 2);
    }

    #[test]
    fn an_unfiltered_sweep_reports_no_matching_set() {
        // `None` rather than the corpus size: the field means "the condition
        // selected this many", and an unfiltered sweep had no condition.
        let ten = Returned {
            ids: (0..10u32).collect(),
            scores: (0..10).map(f64::from).collect(),
        };
        let out = evaluate(&gt100(), &[ten], &abs_eps(1e-9), 10, None);
        assert!(out.n_matching.is_none());
        assert_eq!(out.returned, 10);
        assert_eq!(out.asked, 10);
        assert_eq!(out.short_lists, 0);
    }

    #[test]
    fn an_unfiltered_truth_keeps_scoring_against_k() {
        // The same returned list against §4.3's unfiltered truth: 2 of 10
        // asked-for slots, which is the behaviour every published sift1m
        // recall was measured under and must not move.
        let two = Returned {
            ids: vec![0, 1],
            scores: vec![0.0, 1.0],
        };
        let out = evaluate(&gt100(), &[two], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 0.2);
    }

    /// Zero queries used to evaluate to recall 0 with a CI of (0, 0), which
    /// overlaps itself: two empty runs were "matched" and T3 passed on
    /// nothing. Nothing measured is NaN throughout, and never matched.
    #[test]
    fn evaluating_no_queries_is_not_measured_and_never_matched() {
        let out = evaluate(&gt100(), &[], &abs_eps(1e-9), 10, None);
        assert_eq!(out.queries, 0);
        assert!(out.recall_at_10.is_nan());
        assert!(out.recall_at_10_ci95.low.is_nan() && out.recall_at_10_ci95.high.is_nan());
        assert!(!recall_is_matched(&out, &out));
    }

    /// An unmeasurable recall@10 has no interval either: `[0.0000,0.7935]`
    /// beside `recall@10=n/a` was Wilson on one imaginary trial.
    #[test]
    fn an_unmeasurable_recall_prints_no_interval() {
        let short = Returned {
            ids: vec![0, 1, 2],
            scores: vec![0.0, 1.0, 2.0],
        };
        let out = evaluate(&gt100(), &[short], &abs_eps(1e-9), 3, None);
        assert!(out.recall_at_10.is_nan());
        assert!(out.recall_at_10_ci95.low.is_nan() && out.recall_at_10_ci95.high.is_nan());
        assert!(
            out.describe().contains("recall@10=n/a [n/a,n/a]"),
            "{}",
            out.describe()
        );
    }

    /// The bug that made every recall-vs-`ef` curve flat.
    ///
    /// Returning the true 91st–100th neighbours as your top 10 is a recall@10
    /// of 0, not 1. Testing membership against the whole cached truth list
    /// scores it 1.0 because all ten are somewhere in the top 100.
    #[test]
    fn recall_at_10_uses_the_top_10_not_the_cached_k() {
        let gt = gt100();
        let returned = Returned {
            ids: (90..100u32).collect(),
            scores: (90..100).map(f64::from).collect(),
        };
        let out = evaluate(&gt, &[returned], &abs_eps(1e-9), 10, None);
        assert_eq!(
            out.recall_at_10, 0.0,
            "these are the 91st-100th, not the top 10"
        );
        assert_eq!(out.recall_at_1, 0.0);
    }

    #[test]
    fn a_genuinely_perfect_top_10_still_scores_one() {
        let gt = gt100();
        let returned = Returned {
            ids: (0..10u32).collect(),
            scores: (0..10).map(f64::from).collect(),
        };
        let out = evaluate(&gt, &[returned], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 1.0);
        assert_eq!(out.recall_at_1, 1.0);
    }

    #[test]
    fn half_right_scores_half() {
        let gt = gt100();
        let mut ids: Vec<u32> = (0..5).collect();
        ids.extend(80..85);
        let scores: Vec<f64> = ids.iter().map(|&i| f64::from(i)).collect();
        let out = evaluate(&gt, &[Returned { ids, scores }], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 0.5);
    }

    /// §8.6's tie clause must key off the k-th score of the *prefix*, so a
    /// vector exactly as far as the 10th true neighbour counts at k=10 — but a
    /// vector as far as the 100th does not.
    #[test]
    fn the_tie_clause_uses_the_kth_score_of_the_prefix() {
        let gt = gt100();
        // id 500 is not in the truth at all, but sits exactly at the 10th
        // neighbour's distance (9.0).
        assert_eq!(gt.judge_at(0, 10, 500, 9.0, None, 1e-9), Judgement::Correct);
        // At the 100th neighbour's distance (99.0) it is not a top-10 answer.
        assert_eq!(gt.judge_at(0, 10, 500, 99.0, None, 1e-9), Judgement::Miss);
        // But it is a legitimate top-100 answer.
        assert_eq!(
            gt.judge_at(0, 100, 500, 99.0, None, 1e-9),
            Judgement::Correct
        );
    }

    /// The engine, not the request, came up short: every empty slot is a miss.
    #[test]
    fn an_engine_returning_fewer_than_k_is_scored_not_excused() {
        let gt = gt100();
        // Asked for 10, returned 5 — all correct.
        let returned = Returned {
            ids: (0..5u32).collect(),
            scores: (0..5).map(f64::from).collect(),
        };
        let out = evaluate(&gt, &[returned], &abs_eps(1e-9), 10, None);
        assert_eq!(
            out.recall_at_10, 0.5,
            "5 of 10 slots filled correctly is 0.5, not n/a"
        );
        assert_eq!(out.short_lists, 1);
        // recall@100 stays n/a: the *request* was for 10.
        assert!(out.recall_at_100.is_nan());

        // Returned nothing at all: recall 0 and MRDE has nothing to average,
        // which is reported as such rather than as a perfect 0.0.
        let empty = Returned {
            ids: vec![],
            scores: vec![],
        };
        let out = evaluate(&gt, &[empty], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 0.0);
        assert_eq!(out.recall_at_1, 0.0);
        assert!(
            out.mean_relative_distance_error.is_nan(),
            "got {}",
            out.mean_relative_distance_error
        );
        assert_eq!(out.short_lists, 1);
        assert!(out.describe().contains("short_lists=1"));
    }

    #[test]
    fn a_duplicated_id_counts_once() {
        let gt = gt100();
        // The true nearest, repeated ten times.
        let returned = Returned {
            ids: vec![0; 10],
            scores: vec![0.0; 10],
        };
        let out = evaluate(&gt, &[returned], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 0.1, "one neighbour found, not ten");
        assert_eq!(out.recall_at_1, 1.0);
    }

    #[test]
    fn a_flattering_score_on_a_wrong_id_is_a_miss_and_a_reported_violation() {
        let gt = gt100();
        // id 500 is not a neighbour. First the engine reports it exactly at
        // the 10th true distance (9.0): a claimed boundary tie.
        let mut ids: Vec<u32> = (0..9).collect();
        ids.push(500);
        let mut scores: Vec<f64> = (0..9).map(f64::from).collect();
        scores.push(9.0);
        let claimed_tie = Returned {
            ids: ids.clone(),
            scores,
        };
        // Without base vectors the tie clause must take the engine's word for
        // a claimed tie: correct.
        let out = evaluate(
            &gt,
            std::slice::from_ref(&claimed_tie),
            &abs_eps(1e-9),
            10,
            None,
        );
        assert_eq!(out.recall_at_10, 1.0);
        assert_eq!(out.impossible_scores, 0);

        // But a claim that *beats* the truth is impossible and is a miss.
        let mut scores: Vec<f64> = (0..9).map(f64::from).collect();
        scores.push(0.5);
        let claimed_better = Returned { ids, scores };
        let out = evaluate(&gt, &[claimed_better], &abs_eps(1e-9), 10, None);
        assert_eq!(out.recall_at_10, 0.9);
        assert_eq!(out.impossible_scores, 1);
        assert!(out.describe().contains("IMPOSSIBLE_SCORES=1"));

        // With the base vectors, the claimed tie is checked against the
        // oracle's own score for id 500. Build a base where rows 0..100 sit at
        // distance i and row 500 sits at 50: the tie claim is false.
        let dim = 1;
        let mut base = vec![0.0f32; 501];
        for (i, v) in base.iter_mut().enumerate().take(100) {
            *v = i as f32;
        }
        base[500] = 50.0;
        let q = [0.0f32];
        let r = Rescorer::new(Metric::Euclid, &base, 501, &q, 1, dim);
        let out = evaluate(&gt, &[claimed_tie], &abs_eps(1e-9), 10, Some(&r));
        assert_eq!(
            out.recall_at_10, 0.9,
            "the oracle says id 500 is at 50, not 9: no tie"
        );
    }
}
