//! §8.5 — the differ, T0 through T4.
//!
//! > "`conformance/` is a Rust binary using the same `qdrant-client` crate that
//! > bfb uses. This is deliberate: one client, one encoder, one decoder, both
//! > engines. Any difference observed is then necessarily server-side, not an
//! > artefact of how the harness talks to each engine."
//!
//! The five tiers, and what each one licenses:
//!
//! - **T0 Wire conformance** — structural comparison. "This catches 'right
//!   numbers, wrong shape,' which no amount of score comparison will find."
//! - **T1 Exact-search value equality** — "**This is the tier that licenses the
//!   performance claim.**"
//! - **T2 Rank agreement under ties** — multisets grouped into
//!   score-equivalence classes at width ε, never ordered lists.
//! - **T3 ANN statistical equivalence** — against §4.3's shared ground truth,
//!   *not* against either engine's own exact search.
//! - **T4 Quantization fidelity** — "the tier that catches the most seductive
//!   way to fake a win: a cheaper quantizer that is faster because it is less
//!   accurate."
//!
//! And the rule that binds them to the performance numbers (§8):
//!
//! > "**no performance number is publishable unless the same build, on the same
//! > data, has a green conformance row.** This is enforced by the harness, not
//! > by convention — the results sink rejects a perf row whose conformance hash
//! > doesn't match a passing run."

pub mod tolerance;

use crate::oracle::{GroundTruth, Metric};
use crate::relevance::{self, Geometric, Returned};
use tolerance::{Distribution, Epsilon};

/// §8.1's three claims, of decreasing strength.
///
/// > "Three distinct claims, of decreasing strength. Conflating them is the
/// > standard way this kind of comparison goes wrong."
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Claim {
    /// §8.1: "**No.** ... This is not a defect in either engine."
    BitExact,
    /// §8.1: "**Yes, and this is the load-bearing claim.**"
    ValueWithinEpsilon,
    /// §8.1: "**Yes, as distributions.**"
    StatisticalEquivalence,
}

impl Claim {
    /// Is this claim achievable at all? §8.1 answers for each.
    pub fn achievable(self, scope: Scope) -> bool {
        match (self, scope) {
            // "Anything promising ID-level equality for HNSW results is
            // promising something false. The spec does not."
            (Claim::BitExact, _) => false,
            (Claim::ValueWithinEpsilon, Scope::ExactSearch) => true,
            (Claim::ValueWithinEpsilon, Scope::AnnSearch) => false,
            (Claim::StatisticalEquivalence, _) => true,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    ExactSearch,
    AnnSearch,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Tier {
    T0Wire,
    T1ExactValue,
    /// §8.6's oracle-free invariants, recorded as a tier so they enter the
    /// row and its hash.
    ///
    /// They were run and printed and then dropped: `tiers` never saw them, so
    /// a strawmann that violated offset consistency or self-retrieval still
    /// produced a green row with the same hash as one that did not. §8.9 runs
    /// "metamorphic + T0 + T1" per commit as one gate. Ordered after T1 so
    /// the arithmetic verdict stands on its own, and before T2 so a violated
    /// invariant stops tier progression and with it any comparative claim.
    /// `passed` is strawmann's side only: a property Qdrant fails is a finding
    /// (§8.6), reported in the detail, not a defect in the row under test.
    Metamorphic,
    T2RankTies,
    T3AnnStatistical,
    T4QuantFidelity,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::T0Wire => "T0 wire conformance",
            Tier::T1ExactValue => "T1 exact-search value equality",
            Tier::Metamorphic => "§8.6 metamorphic properties",
            Tier::T2RankTies => "T2 rank agreement under ties",
            Tier::T3AnnStatistical => "T3 ANN statistical equivalence",
            Tier::T4QuantFidelity => "T4 quantization fidelity",
        }
    }
}

/// §8.1's claims, stated before any tier runs.
///
/// The enum and its `achievable()` existed as executable documentation that
/// nothing executed. Printing it is what makes it documentation an outside
/// reader sees: the differ's output is read by someone deciding whether to
/// believe a comparison, and "bit-exact identity is not claimed" belongs at the
/// top of that output rather than in a spec they have not read.
pub fn claims_banner() -> String {
    let mut s = String::from("§8.1 claims under test:\n");
    for (claim, scope, tier) in [
        (Claim::BitExact, Scope::ExactSearch, "not tested"),
        (Claim::ValueWithinEpsilon, Scope::ExactSearch, "T1"),
        (Claim::ValueWithinEpsilon, Scope::AnnSearch, "not tested"),
        (Claim::StatisticalEquivalence, Scope::AnnSearch, "T3"),
    ] {
        s.push_str(&format!(
            "  {:<22} over {:<12} {:<13} {}\n",
            format!("{claim:?}"),
            format!("{scope:?}"),
            if claim.achievable(scope) {
                "achievable"
            } else {
                "NOT achievable"
            },
            tier,
        ));
    }
    s.push_str("  (bit-exact ID equality for HNSW is not achievable and is not claimed;\n");
    s.push_str("   §8.1: \"anything promising ID-level equality is promising something false\")");
    s
}

/// A tier's outcome.
///
/// §8.4's reporting rule: "conformance reports the **distribution** of
/// `|Δscore|` — max, p99.9, p99, mean — not a pass/fail bit." So `passed` never
/// travels alone.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TierResult {
    pub tier: Tier,
    pub passed: bool,
    pub detail: String,
    pub distribution: Option<Distribution>,
    /// A finding about the *other* engine rather than a tier of the row under
    /// test: Qdrant's T4 fidelity, pushed beside strawmann's. Recorded in the
    /// row and printed, but `tier_reached` does not stop on it — the rule
    /// `Tier::Metamorphic` already states for its properties, applied to the
    /// one other place a Qdrant verdict enters the row. Absent in rows written
    /// before the field existed, which is "not advisory".
    #[serde(default)]
    pub advisory: bool,
}

// =========================================================================
// T0 — wire conformance
// =========================================================================

/// A response's *shape*, independent of its values.
///
/// §8.5 T0: "Structural comparison of response messages: field presence, ID
/// variant (`num` vs `uuid`), payload round-trip fidelity, `version`/
/// `operation_id` semantics, presence of `time`, and — importantly — **error
/// behaviour**: unsupported operations must produce the same gRPC status code
/// as Qdrant, not a hang or a wrong-shaped success."
///
/// `version` is not here. The tier's own verdict text says versions are on
/// the ignore-list, and they are: strawmann answers `version = 0` on every
/// point (a documented gap, `handlers.encodeOne`) where Qdrant counts
/// updates, so comparing presence would fail every run on a value §8.5 lists
/// beside timings as ignorable. The struct used to carry a `has_version` the
/// comparison never read, which described a check that was not made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireShape {
    pub result_count: usize,
    pub id_variant: IdVariant,
    pub has_time: bool,
    pub has_payload: bool,
    pub has_vectors: bool,
    pub status_code: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IdVariant {
    Num,
    Uuid,
    Absent,
    Mixed,
}

/// §8.5 T0: "Compare as proto trees with an explicit ignore-list (timings,
/// versions, commit hashes)."
///
/// The ignore-list is the whole reason this is a struct comparison rather than
/// a byte comparison: `time` differs by construction on every call, and
/// comparing it would make T0 fail always and therefore mean nothing.
pub fn compare_shape(ours: &WireShape, theirs: &WireShape) -> TierResult {
    let mut problems: Vec<String> = Vec::new();

    if ours.result_count != theirs.result_count {
        problems.push(format!(
            "result count {} vs {}",
            ours.result_count, theirs.result_count
        ));
    }
    if ours.id_variant != theirs.id_variant {
        problems.push(format!(
            "id variant {:?} vs {:?}",
            ours.id_variant, theirs.id_variant
        ));
    }
    // `time` must be *present*; its value is on the ignore-list. §2: bfb reads
    // it as the server-side timing, so an absent field silently zeroes the
    // "server_timings" histogram.
    if ours.has_time != theirs.has_time {
        problems.push(format!(
            "time presence {} vs {}",
            ours.has_time, theirs.has_time
        ));
    }
    if ours.has_payload != theirs.has_payload {
        problems.push(format!(
            "payload presence {} vs {}",
            ours.has_payload, theirs.has_payload
        ));
    }
    if ours.has_vectors != theirs.has_vectors {
        problems.push(format!(
            "vectors presence {} vs {}",
            ours.has_vectors, theirs.has_vectors
        ));
    }
    if ours.status_code != theirs.status_code {
        problems.push(format!(
            "gRPC status {} vs {}",
            ours.status_code, theirs.status_code
        ));
    }

    TierResult {
        advisory: false,
        tier: Tier::T0Wire,
        passed: problems.is_empty(),
        detail: if problems.is_empty() {
            "shapes identical (timings and versions on the ignore-list)".into()
        } else {
            problems.join("; ")
        },
        distribution: None,
    }
}

// =========================================================================
// T1 — exact-search value equality
// =========================================================================

/// Do two result lists for one query hold the same ids, up to permutation
/// within ε-ties?
///
/// §8.3: "A divergence in the *ids* or the *number* of results is never
/// excused ... that is a compatibility defect, and no amount of numerical
/// accuracy repairs it." T1 compared `scores[i]` positionally and never looked
/// at `ids[i]`, so an engine returning the right scores against the wrong
/// points — an id-mapping bug — passed the tier that licenses the performance
/// numbers.
///
/// The rule is deliberately not positional equality, because §8.5 T2 is right
/// that the order *within* an ε-tie is arbitrary in both engines. Two checks:
///
/// 1. the id multisets are equal — same points, whatever the order;
/// 2. an id at position `i` in one list and `j` in the other moved only
///    within a tie: positions `i` and `j` are ε-tied in at least one of the
///    two lists. An id that crossed a real score gap fails this even though
///    (1) holds; an id that swapped places with its ε-twin passes.
///
/// (2) is judged on each list's *own* scores, never across the two lists.
/// Comparing an id's score in ours against its score in theirs would fold
/// value drift into the id verdict, and §8.3 excuses value drift when the
/// oracle sides with us; the id verdict must not depend on that.
///
/// (2) is also what "aligned tie classes hold the same ids" reduces to
/// without the fragility of drawing class boundaries: two lists that agree
/// everywhere to within ε can still cut their classes one position apart, and
/// comparing class-by-class would then report a divergence where there is
/// none. T2 keeps the class view, cut only where both lists cut.
pub fn ids_agree_up_to_ties(a: &Returned, b: &Returned, eps: &Epsilon) -> Result<(), String> {
    if a.ids.len() != b.ids.len() {
        return Err(format!("{} vs {} results", a.ids.len(), b.ids.len()));
    }
    // Membership below is by set, so a list that returns one point twice in
    // place of another used to pass whenever the missing id sat on the cut.
    // An engine returning the same point twice is a defect on its own
    // (`relevance::evaluate` counts it once), and it is caught here first.
    for (side, r) in [("ours", a), ("theirs", b)] {
        let mut seen = std::collections::HashSet::with_capacity(r.ids.len());
        if let Some(dup) = r.ids.iter().copied().find(|id| !seen.insert(*id)) {
            return Err(format!("{side} returned id {dup} more than once"));
        }
    }
    // An id one list holds and the other does not is a divergence unless it
    // sits on the cut (`excused_at_cut`): the k-th and (k+1)-th scores tie,
    // and which of the tied points makes the list is arbitrary in both
    // engines. Everything else — a missing interior id, an id at the cut whose
    // score the other list's cut does not tie — remains a violation.
    let unexcused = |x: &Returned, y: &Returned| -> Vec<u32> {
        (0..x.ids.len())
            .filter(|&i| !y.ids.contains(&x.ids[i]) && !excused_at_cut(x, i, y, eps))
            .map(|i| x.ids[i])
            .take(5)
            .collect()
    };
    let (only_a, only_b) = (unexcused(a, b), unexcused(b, a));
    if !only_a.is_empty() || !only_b.is_empty() {
        return Err(format!(
            "id sets differ (only in ours: {only_a:?}, only in theirs: {only_b:?})"
        ));
    }
    let tied_in = |r: &Returned, i: usize, j: usize| -> bool {
        i == j || eps.holds((r.scores[i] - r.scores[j]).abs(), r.scores[i])
    };
    for i in 0..a.ids.len() {
        let id = a.ids[i];
        if !b.ids.contains(&id) {
            continue; // excused at the cut above
        }
        let moved_within_tie = b
            .ids
            .iter()
            .enumerate()
            .filter(|&(_, &oid)| oid == id)
            .any(|(j, _)| tied_in(a, i, j) || tied_in(b, i, j));
        if !moved_within_tie {
            let j = b.ids.iter().position(|&oid| oid == id).unwrap_or(i);
            return Err(format!(
                "id {id} is at pos {i} in ours (score {:.9}) and pos {j} in theirs (score {:.9}), \
across a gap wider than ε in both lists",
                a.scores[i], b.scores[j]
            ));
        }
    }
    Ok(())
}

/// Is the id at position `i` of `r` a member of the boundary tie class the
/// other list could legitimately have cut?
///
/// Both lists are cut at k. When the k-th and (k+1)-th scores tie, which of
/// the tied points makes the list is decided by tie-breaking — internal
/// offset in strawmann, heap and segment-merge order in Qdrant — and the
/// shipped SIFT ground truth has 137 of 10 000 queries with an exact tie
/// across the limit=10 cut. Excused iff the id's score is within ε of its own
/// list's last score (it is in that list's boundary tie class) **and** within
/// ε of the other list's last score (the other side's cut ties it, so cutting
/// it was legitimate). The second clause is what makes a last id whose score
/// is 2ε past the other's cut a violation rather than a coin toss: the last
/// id of any list trivially ties its own cut.
///
/// Used by T1 (`ids_agree_up_to_ties`), T2 (`compare_ranks`) and
/// `metamorphic::permutation_invariance`, so all three excuse the same ids.
pub fn excused_at_cut(r: &Returned, i: usize, other: &Returned, eps: &Epsilon) -> bool {
    let (Some(&own_cut), Some(&other_cut)) = (r.scores.last(), other.scores.last()) else {
        return false;
    };
    let s = r.scores[i];
    eps.holds((s - own_cut).abs(), own_cut) && eps.holds((s - other_cut).abs(), other_cut)
}

/// §8.5 T1: "`params.exact = true`, fp32, every metric, every dimension in the
/// matrix. Element-wise comparison of returned `(id, score)` pairs against
/// Qdrant and against the fp64 oracle. **This is the tier that licenses the
/// performance claim.**"
///
/// `limit` is what every query asked for. Each list must hold exactly
/// `min(limit, n_base)` entries: a run that compared zero queries, or lists
/// the engines had silently truncated, used to pass with an empty
/// distribution — nothing was over tolerance because nothing was measured.
pub fn compare_exact(
    ours: &[Returned],
    theirs: &[Returned],
    oracle_gt: &GroundTruth,
    eps: &Epsilon,
    limit: usize,
) -> TierResult {
    let mut deltas: Vec<f64> = Vec::new();
    let mut magnitudes: Vec<f64> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    // Structural divergence — differing result counts, differing ids — is
    // tracked separately from value divergence, because only the latter can be
    // excused by being closer to the oracle. Returning a different *number* of
    // results, or different *points*, than Qdrant is a compatibility defect no
    // amount of numerical accuracy fixes, and folding the two together let a
    // count mismatch pass.
    let mut structural: Vec<String> = Vec::new();
    // Every violation is counted even though only the first 8 are quoted.
    // Without this the report cannot distinguish nine bad scores from nine
    // million: both print eight examples and a distribution.
    let mut violations = 0usize;
    let mut id_violations = 0usize;

    let n = ours.len().min(theirs.len());
    // Nothing compared is not a pass. An empty query set, or an engine that
    // answered nothing, produced an empty distribution and `problems.is_empty()`
    // — a green T1 licensing a performance claim on zero evidence.
    if n == 0 {
        structural.push(format!(
            "no queries compared (ours answered {}, theirs {})",
            ours.len(),
            theirs.len()
        ));
        problems.push("no queries compared".into());
    }
    // A differing number of *queries* answered is structural, and was silently
    // truncated: `min` compared the overlap and dropped the rest with nothing
    // said. An engine that answered 50 of 100 queries would be compared on its
    // 50 and pass.
    if ours.len() != theirs.len() {
        structural.push(format!(
            "answered {} vs {} queries; compared the first {n}",
            ours.len(),
            theirs.len()
        ));
        problems.push(format!(
            "answered {} vs {} queries",
            ours.len(),
            theirs.len()
        ));
    }
    // And both must have answered the whole ground-truth query set: the row
    // claims conformance over that set, not over whatever subset came back.
    if ours.len() != oracle_gt.n_queries || theirs.len() != oracle_gt.n_queries {
        structural.push(format!(
            "ground truth covers {} queries; ours answered {}, theirs {}",
            oracle_gt.n_queries,
            ours.len(),
            theirs.len()
        ));
        problems.push(format!(
            "ground truth covers {} queries; ours answered {}, theirs {}",
            oracle_gt.n_queries,
            ours.len(),
            theirs.len()
        ));
    }
    // Every list must be exactly as long as was asked for, or as long as the
    // collection allows. Shorter lists compared equal position-by-position
    // and passed; two engines each returning nothing agree perfectly.
    let expected_len = limit.min(oracle_gt.n_base);
    let mut short_lists = 0usize;
    for qi in 0..n {
        let a = &ours[qi];
        let b = &theirs[qi];
        if a.ids.len() != expected_len || b.ids.len() != expected_len {
            short_lists += 1;
            if short_lists <= 3 {
                let msg = format!(
                    "query {qi}: expected {expected_len} results (limit {limit}, n_base {}), got ours {} theirs {}",
                    oracle_gt.n_base,
                    a.ids.len(),
                    b.ids.len()
                );
                structural.push(msg.clone());
                problems.push(msg);
            }
        }
        if a.ids.len() != b.ids.len() {
            structural.push(format!(
                "query {qi}: {} vs {} results",
                a.ids.len(),
                b.ids.len()
            ));
            problems.push(format!(
                "query {qi}: {} vs {} results",
                a.ids.len(),
                b.ids.len()
            ));
            continue;
        }
        // The *points* first: §8.3's "never excused" clause.
        if let Err(why) = ids_agree_up_to_ties(a, b, eps) {
            id_violations += 1;
            if id_violations <= 3 {
                structural.push(format!("query {qi}: {why}"));
                problems.push(format!("query {qi}: {why}"));
            }
        }
        for i in 0..a.ids.len() {
            let d = (a.scores[i] - b.scores[i]).abs();
            deltas.push(d);
            magnitudes.push(b.scores[i].abs().max(a.scores[i].abs()));
            if !eps.holds(d, b.scores[i]) {
                violations += 1;
                if problems.len() < 8 {
                    // The *allowed* value at this magnitude, and the overshoot
                    // as a multiple. Printing the bare `eps.value` beside an
                    // absolute delta compared a distance against a ratio and
                    // read as a 7000x violation where the real one was 24x.
                    problems.push(format!(
                        "query {qi} pos {i}: |Δ|={d:.3e} exceeds the {:.3e} allowed at |score|={:.3e} ({:.1}x over) \
(ours {} @ {:.9}, theirs {} @ {:.9})",
                        eps.absolute_at(b.scores[i]),
                        b.scores[i].abs(),
                        eps.overshoot(d, b.scores[i]),
                        a.ids[i], a.scores[i], b.ids[i], b.scores[i]
                    ));
                }
            }
        }
    }
    if short_lists > 3 {
        structural.push(format!(
            "{short_lists} queries returned other than {expected_len} results"
        ));
    }
    if id_violations > 3 {
        structural.push(format!(
            "{id_violations} queries with id disagreements (first 3 shown)"
        ));
    }

    // §8.2: "if strawmann and Qdrant disagree beyond ε, compare both to the
    // fp64 oracle." Doing it unconditionally means the arbiter's verdict is
    // always in the report, not only when something went wrong.
    let (ours_oracle, theirs_oracle) = (
        delta_vs_oracle(ours, oracle_gt, eps),
        delta_vs_oracle(theirs, oracle_gt, eps),
    );
    let (ours_vs_oracle, theirs_vs_oracle) = (ours_oracle.max_abs, theirs_oracle.max_abs);
    // Our ids disagreeing with the *oracle* is a defect in us, whatever Qdrant
    // did. Qdrant's ids disagreeing with the oracle is a finding, reported but
    // not ours to fail on.
    if ours_oracle.id_violations > 0 {
        let msg = format!(
            "{} of our results hold ids the fp64 oracle does not (first: {})",
            ours_oracle.id_violations,
            ours_oracle.first_id_violation.as_deref().unwrap_or("?")
        );
        structural.push(msg.clone());
        problems.push(msg);
    }
    let theirs_ids = if theirs_oracle.id_violations > 0 {
        format!(
            " (Qdrant: {} results hold ids the oracle does not — a finding, first: {})",
            theirs_oracle.id_violations,
            theirs_oracle.first_id_violation.as_deref().unwrap_or("?")
        )
    } else {
        String::new()
    };

    let dist = Distribution::from_samples(deltas, &magnitudes);
    let counted = if violations > 8 {
        format!(" ({violations} scores over tolerance, first 8 shown)")
    } else {
        String::new()
    };
    let verdict = if problems.is_empty() {
        format!(
            "{} | ids agree up to ε-ties | vs fp64 oracle: ours max {:.3e}, Qdrant max {:.3e}{theirs_ids}",
            dist.describe(),
            ours_vs_oracle,
            theirs_vs_oracle
        )
    } else {
        format!(
            "{} | {}{counted} | vs fp64 oracle: ours max {:.3e}, Qdrant max {:.3e}{theirs_ids} — {}",
            dist.describe(),
            problems.join("; "),
            ours_vs_oracle,
            theirs_vs_oracle,
            if ours_vs_oracle > theirs_vs_oracle {
                "the oracle says WE are further off"
            } else {
                "the oracle says QDRANT is further off"
            }
        )
    };

    // The tier is graded against **truth**, not against Qdrant.
    //
    // T1 originally failed whenever the two engines disagreed by more than ε,
    // which silently makes Qdrant the definition of correct. At d=1536 cosine
    // that inverted the verdict: the engines differ by 1.967e-6 against a
    // calibrated ε of 9.537e-7, and the fp64 oracle shows strawmann is 1.501e-7
    // from the true value while Qdrant is 1.358e-6 — nine times further. Under
    // the old rule strawmann *failed a conformance tier for being more
    // accurate*, and the only way to pass would have been to reproduce Qdrant's
    // rounding error deliberately.
    //
    // §8.1 already concedes bit-exactness is unachievable, so equality with
    // Qdrant was never the real requirement; being right was. The rule is
    // therefore:
    //
    //   * agree within ε  -> pass, nothing to arbitrate;
    //   * disagree, but we are within ε of the fp64 oracle -> pass, and record
    //     that Qdrant is the one that drifted;
    //   * disagree and we are further from the oracle than Qdrant -> fail. That
    //     is a real defect and the tier still catches it.
    //
    // A divergence in *ids* is never excused this way: `structural` holds the
    // count mismatches and the id disagreements, and any entry there fails
    // the tier outright.
    // Judged with `OracleDelta::accurate`, which scales by each score's own
    // magnitude. This was `eps.holds(ours_vs_oracle, 1.0)`, which fed a
    // magnitude of 1.0 to a *relative* ε and so demanded an absolute deviation
    // below a dimensionless ratio. For Euclid, Dot and Manhattan — three of the
    // four metrics, including SIFT1M's — that is a bound below one fp32 ulp,
    // so the branch below could never be taken and this whole adjudication was
    // dead code on the metrics that matter. The comment above described a fix
    // that only ever applied to cosine.
    let we_are_accurate = ours_oracle.accurate();
    let we_are_better = ours_vs_oracle <= theirs_vs_oracle;
    let value_ok =
        structural.is_empty() && (problems.is_empty() || (we_are_accurate && we_are_better));

    let detail = if problems.is_empty() {
        verdict
    } else if value_ok && structural.is_empty() {
        format!(
            "{verdict} | ACCEPTED: we are within ε of the fp64 oracle ({:.3e} <= ε {:.3e}); \
Qdrant is the engine that drifted, and §8.3 does not require reproducing its error",
            ours_vs_oracle, eps.value
        )
    } else {
        verdict
    };

    TierResult {
        advisory: false,
        tier: Tier::T1ExactValue,
        passed: value_ok,
        detail,
        distribution: Some(dist),
    }
}

/// How far a result set is from the fp64 oracle, both ways it can be read.
///
/// `max_abs` is the largest absolute deviation, which is what a reader wants to
/// see. `max_overshoot` is the largest deviation *as a multiple of what ε
/// permits at that score's own magnitude*, which is what a judgement must use:
/// a relative ε says nothing until it is applied to a magnitude.
///
/// `id_violations` counts positions whose *point* the oracle disagrees with,
/// under the same up-to-ε-ties rule `ids_agree_up_to_ties` applies between the
/// engines. The oracle's list is cut at `k`, so a returned id the oracle does
/// not list is excused only if its score ties the cut (§4.3's boundary rule).
#[derive(Clone, Debug, Default)]
struct OracleDelta {
    max_abs: f64,
    max_overshoot: f64,
    id_violations: usize,
    first_id_violation: Option<String>,
}

impl OracleDelta {
    /// Within tolerance everywhere.
    fn accurate(&self) -> bool {
        self.max_overshoot <= 1.0
    }

    fn id_violation(&mut self, msg: String) {
        self.id_violations += 1;
        if self.first_id_violation.is_none() {
            self.first_id_violation = Some(msg);
        }
    }
}

fn delta_vs_oracle(results: &[Returned], gt: &GroundTruth, eps: &Epsilon) -> OracleDelta {
    let mut out = OracleDelta::default();
    for (qi, r) in results.iter().enumerate().take(gt.n_queries) {
        let truth = gt.scores_for(qi);
        let truth_ids = gt.neighbours(qi);
        for (i, &s) in r.scores.iter().enumerate() {
            if i >= truth.len() || !truth[i].is_finite() {
                continue;
            }
            let d = (s - truth[i]).abs();
            out.max_abs = out.max_abs.max(d);
            // Scaled by the *truth*, not by the returned score: the oracle is
            // the reference, so it sets the magnitude the tolerance is read at.
            out.max_overshoot = out.max_overshoot.max(eps.overshoot(d, truth[i]));
        }

        // The ids. Compared over the depth both lists cover; a base set
        // smaller than k pads the truth with NaN (see `compute`), and those
        // positions carry no verdict.
        let depth = r.ids.len().min(gt.k);
        let depth = (0..depth).take_while(|&i| truth[i].is_finite()).count();
        if depth == 0 {
            continue;
        }
        // Same rule as `ids_agree_up_to_ties`, judged on each list's own
        // scores: an id at rank `i` here and rank `j` in the truth moved only
        // within a tie. The truth is cut at `depth`, so a returned id the
        // oracle does not list is legitimate only if it sits on *both* cuts,
        // exactly as `excused_at_cut` demands between the engines: within ε
        // of the returned list's own last score **and** within ε of the
        // truth's boundary (§4.3's boundary rule: which of two equidistant
        // points makes the list is arbitrary). The last returned id trivially
        // ties its own cut, so the one-sided form excused any id at rank k
        // whatever its score — and both engines returning the same wrong last
        // point then passed T1, because the score deviation is only consulted
        // when the engines disagree.
        let boundary = truth[depth - 1];
        let own_cut = r.scores[depth - 1];
        let tied_truth =
            |i: usize, j: usize| i == j || eps.holds((truth[i] - truth[j]).abs(), truth[i]);
        let tied_ret = |i: usize, j: usize| {
            i == j || eps.holds((r.scores[i] - r.scores[j]).abs(), r.scores[i])
        };
        for i in 0..depth {
            let (id, s) = (r.ids[i], r.scores[i]);
            let ok = match truth_ids.iter().position(|&tid| tid == id) {
                Some(j) if truth[j].is_finite() => {
                    tied_truth(i, j) || (j < r.scores.len() && tied_ret(i, j))
                }
                _ => eps.holds((s - boundary).abs(), boundary) && tied_ret(i, depth - 1),
            };
            if !ok {
                out.id_violation(format!(
                    "query {qi} pos {i}: id {id} @ {s:.9} is not an oracle neighbour at that rank \
(oracle has {} @ {:.9} there)",
                    truth_ids[i], truth[i]
                ));
            }
        }
        // And every oracle neighbour inside the cut must have been returned,
        // unless it sits on the cut and lost the coin toss: it ties the
        // truth's own boundary and the returned list's cut ties it too. The
        // k-th truth neighbour trivially ties its own boundary, so without the
        // second clause the last oracle neighbour could go missing unnoticed.
        for j in 0..depth {
            let tid = truth_ids[j];
            let excused =
                tied_truth(j, depth - 1) && eps.holds((truth[j] - own_cut).abs(), own_cut);
            if !r.ids.contains(&tid) && !excused {
                out.id_violation(format!(
                    "query {qi}: oracle neighbour {tid} @ {:.9} (rank {j}) was not returned",
                    truth[j]
                ));
            }
        }
    }
    out
}

// =========================================================================
// T2 — rank agreement under ties
// =========================================================================

/// §8.5 T2: "Equal scores occur constantly (duplicate vectors, quantized
/// encodings, small dims) and their ordering is arbitrary in both engines.
/// Compare results as multisets grouped into score-equivalence classes at width
/// ε, never as ordered lists. Where a soft measure is needed, report Kendall's τ
/// and rank-biased overlap rather than exact-match rate."
///
/// The multiset comparison is done at the class boundaries **both** engines
/// draw. Class-by-class `zip` was fragile: two lists that agree everywhere to
/// within ε can still cut a class one position apart, and then every class
/// after the cut compares unequal. Cutting only where both lists cut, and
/// comparing the ids accumulated since the previous common cut, is the same
/// test without the artefact — and the end of the list is always a common cut,
/// so a differing id multiset overall is always caught.
///
/// `passed` is that comparison, not a constant. It was hard-wired `true` on
/// the grounds that T2 is "soft"; τ and RBO are the soft part, and §8.5 lists
/// them as what to *report*. The multiset test is a yes/no question with an
/// answer, and a tier that cannot fail cannot gate anything.
pub fn compare_ranks(ours: &[Returned], theirs: &[Returned], eps: &Epsilon) -> TierResult {
    let mut mismatched = 0usize;
    let mut segments = 0usize;
    let mut structural = 0usize;
    let mut tau_sum = 0.0f64;
    let mut tau_n = 0usize;
    let mut tau_excluded = 0usize;
    let mut rbo_sum = 0.0f64;
    let mut first: Option<String> = None;
    // Queries both engines answered with nothing. Two empty lists have no
    // segment to compare, τ of None and an RBO of 1.0, so a run in which
    // every query came back empty used to pass this tier; T1 and T3 already
    // refuse to pass on nothing, and this is the same rule.
    let mut empty = 0usize;
    let n = ours.len().min(theirs.len());
    if ours.len() != theirs.len() {
        structural += 1;
        first = Some(format!(
            "answered {} vs {} queries",
            ours.len(),
            theirs.len()
        ));
    }

    for qi in 0..n {
        let a = &ours[qi];
        let b = &theirs[qi];
        if a.ids.is_empty() && b.ids.is_empty() {
            empty += 1;
            continue;
        }
        if a.ids.len() != b.ids.len() {
            structural += 1;
            first.get_or_insert_with(|| {
                format!("query {qi}: {} vs {} results", a.ids.len(), b.ids.len())
            });
            continue;
        }
        let cuts_a = class_cuts(a, eps);
        let cuts_b = class_cuts(b, eps);
        // The cuts are anchor-based per side, so an id excused at the cut on
        // one side (`excused_at_cut`: within ε of *both* lists' last score)
        // can sit in a non-final segment on that side only, and its
        // counterpart in the final one on the other. From the first excused
        // position on either side, everything is one final segment.
        let len = a.ids.len();
        let first_excused =
            (0..len).find(|&p| excused_at_cut(a, p, b, eps) || excused_at_cut(b, p, a, eps));
        let mut start = 0usize;
        for i in 1..=len {
            let is_cut = i == len || first_excused.is_none_or(|p| i <= p);
            if is_cut && cuts_a.contains(&i) && cuts_b.contains(&i) {
                segments += 1;
                // Minus the boundary-tie members the other side legitimately
                // cut (`excused_at_cut`), same rule as T1.
                let keep = |x: &Returned, y: &Returned| -> Vec<u32> {
                    (start..i)
                        .filter(|&p| y.ids.contains(&x.ids[p]) || !excused_at_cut(x, p, y, eps))
                        .map(|p| x.ids[p])
                        .collect()
                };
                let mut sa = keep(a, b);
                let mut sb = keep(b, a);
                sa.sort_unstable();
                sb.sort_unstable();
                if sa != sb {
                    mismatched += 1;
                    first.get_or_insert_with(|| {
                        format!("query {qi} positions {start}..{i}: {sa:?} vs {sb:?}")
                    });
                }
                start = i;
            }
        }
        // τ needs two common ids to say anything; a query without them is
        // excluded from the average and counted, not averaged in as 0.
        match kendall_tau(&a.ids, &b.ids) {
            Some(t) => {
                tau_sum += t;
                tau_n += 1;
            }
            None => tau_excluded += 1,
        }
        rbo_sum += rank_biased_overlap(&a.ids, &b.ids, 0.9);
    }

    let nf = n.max(1) as f64;
    // Nothing compared is not agreement: at least one query with results.
    let passed = n > empty && structural == 0 && mismatched == 0;
    TierResult {
        advisory: false,
        tier: Tier::T2RankTies,
        passed,
        detail: format!(
            "{n} queries, {mismatched}/{segments} ε-tie segments differ{}{}; Kendall τ={} over {tau_n} queries ({tau_excluded} with <2 common ids excluded), RBO={:.4}{}",
            if structural > 0 {
                format!(", {structural} structural mismatches")
            } else {
                String::new()
            },
            if empty > 0 {
                format!(", {empty} queries answered empty by both")
            } else {
                String::new()
            },
            if tau_n > 0 {
                format!("{:.4}", tau_sum / tau_n as f64)
            } else {
                "n/a".to_string()
            },
            rbo_sum / nf,
            first.map(|f| format!(" — first: {f}")).unwrap_or_default()
        ),
        distribution: None,
    }
}

/// The positions at which a result list's ε-tie classes end (always includes
/// `len`, so an empty list has the single cut 0).
fn class_cuts(r: &Returned, eps: &Epsilon) -> std::collections::HashSet<usize> {
    let mut cuts = std::collections::HashSet::new();
    let mut anchor: Option<f64> = None;
    for i in 0..r.ids.len() {
        let s = r.scores[i];
        match anchor {
            Some(a) if eps.holds((s - a).abs(), a) => {}
            _ => {
                if i > 0 {
                    cuts.insert(i);
                }
                anchor = Some(s);
            }
        }
    }
    cuts.insert(r.ids.len());
    cuts
}

/// Kendall's τ over the ids common to both lists; `None` with fewer than two
/// common ids, which is no evidence either way.
pub fn kendall_tau(a: &[u32], b: &[u32]) -> Option<f64> {
    let pos_b: std::collections::HashMap<u32, usize> =
        b.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let common: Vec<(usize, usize)> = a
        .iter()
        .enumerate()
        .filter_map(|(i, x)| pos_b.get(x).map(|&j| (i, j)))
        .collect();
    // Fewer than two common ids is no evidence of agreement, not perfect
    // agreement: this returned 1.0, so a quantizer returning entirely wrong
    // points passed T4's τ floor. Nor is it disagreement: as 0.0 it made T4
    // fail unconditionally at `--limit 1` and dragged T2's average. The
    // callers exclude such queries from their averages and report how many.
    if common.len() < 2 {
        return None;
    }
    let (mut conc, mut disc) = (0i64, 0i64);
    for i in 0..common.len() {
        for j in (i + 1)..common.len() {
            let s = (common[i].0 as i64 - common[j].0 as i64)
                * (common[i].1 as i64 - common[j].1 as i64);
            if s > 0 {
                conc += 1;
            } else if s < 0 {
                disc += 1;
            }
        }
    }
    let total = conc + disc;
    Some(if total == 0 {
        1.0
    } else {
        (conc - disc) as f64 / total as f64
    })
}

/// Rank-biased overlap, top-weighted list similarity, normalised to [0, 1].
///
/// The raw (un-extrapolated) RBO of two *identical* lists of length `k` is
/// `1 - p^k`, not 1 — at `p = 0.9` and `k = 10` that is 0.651. Reporting the
/// raw value alongside a Kendall τ of 1.0 reads as disagreement where there is
/// none, so it is divided by `1 - p^k`: identical prefixes score exactly 1.0
/// and disjoint ones exactly 0.0, whatever the depth.
///
/// `p` is the top-weighting: 0.9 means the first ~10 positions carry most of
/// the weight, which matches §4.4's focus on recall@10.
pub fn rank_biased_overlap(a: &[u32], b: &[u32], p: f64) -> f64 {
    let depth = a.len().min(b.len());
    if depth == 0 {
        return 1.0;
    }
    let mut sa = std::collections::HashSet::new();
    let mut sb = std::collections::HashSet::new();
    let mut acc = 0.0f64;
    for d in 0..depth {
        sa.insert(a[d]);
        sb.insert(b[d]);
        let overlap = sa.intersection(&sb).count() as f64;
        acc += (overlap / (d + 1) as f64) * p.powi(d as i32);
    }
    // Raw RBO over a finite prefix, then normalised by its own maximum so the
    // scale is interpretable without knowing `depth` and `p`.
    let raw = (1.0 - p) * acc;
    let max = 1.0 - p.powi(depth as i32);
    if max <= 0.0 {
        1.0
    } else {
        (raw / max).clamp(0.0, 1.0)
    }
}

// =========================================================================
// T3 — ANN statistical equivalence
// =========================================================================

/// §8.5 T3: "Held-out query set against the static ground truth of §4.3 — *not*
/// against either engine's own exact search, which is what bfb's
/// `--search-quality` does and why it can't be used here."
pub fn compare_ann(
    ours: &[Returned],
    theirs: &[Returned],
    gt: &GroundTruth,
    eps: &Epsilon,
    limit: usize,
    rescorer: Option<&crate::oracle::Rescorer>,
) -> (TierResult, Geometric, Geometric) {
    let a = relevance::evaluate(gt, ours, eps, limit, rescorer);
    let b = relevance::evaluate(gt, theirs, eps, limit, rescorer);
    let matched = relevance::recall_is_matched(&a, &b);

    (
        TierResult {
            advisory: false,
            tier: Tier::T3AnnStatistical,
            passed: matched,
            detail: format!(
                "strawmann {} | qdrant {} | recall@10 CIs {}",
                a.describe(),
                b.describe(),
                if matched {
                    "overlap — QPS is comparable at this point"
                } else {
                    "DO NOT overlap — a QPS comparison here would be at unequal recall (§7.4)"
                }
            ),
            distribution: None,
        },
        a,
        b,
    )
}

// =========================================================================
// T4 — quantization fidelity
// =========================================================================

/// §8.5 T4: "Per encoding: the distribution of `|quantized_score − fp32_score|`,
/// rank correlation against the fp64 oracle, and recall at matched
/// oversampling. This is the tier that catches the most seductive way to fake a
/// win: a cheaper quantizer that is faster because it is less accurate.
/// Comparing at equal recall (§7.4) already guards against it; T4 makes the
/// mechanism visible instead of merely controlling for it."
///
/// `dominance` is §8.6's quantization-dominance property measured on the same
/// collection — recall non-decreasing in oversampling with `rescore = true` —
/// which is the "recall at matched oversampling" clause of the tier. `passed`
/// is that property together with the rank-correlation floor below; it was
/// hard-wired `true` on the grounds that T4 "characterises rather than gates",
/// which left the tier that exists to catch a lossy quantizer unable to
/// report one.
pub fn quantization_fidelity(
    quantized: &[Returned],
    fp32: &[Returned],
    gt: &GroundTruth,
    encoding: &str,
    dominance: &crate::metamorphic::PropertyResult,
) -> TierResult {
    let mut deltas = Vec::new();
    let mut mags = Vec::new();
    // Matched by id, not by position: the two lists are two ANN candidate
    // sets, and position `i` of one need not hold the point at position `i`
    // of the other. Ids only one list holds carry no score delta; they are
    // counted and reported beside the distribution instead.
    let mut uncommon = 0usize;
    let n = quantized.len().min(fp32.len());
    for qi in 0..n {
        let a = &quantized[qi];
        let b = &fp32[qi];
        for (i, id) in a.ids.iter().enumerate() {
            match b.ids.iter().position(|x| x == id) {
                Some(j) => {
                    deltas.push((a.scores[i] - b.scores[j]).abs());
                    mags.push(b.scores[j].abs());
                }
                None => uncommon += 1,
            }
        }
        uncommon += b.ids.iter().filter(|id| !a.ids.contains(id)).count();
    }
    let dist = Distribution::from_samples(deltas, &mags);

    // Bounded by the ground truth as well as by the two result sets:
    // `gt.neighbours` panics past `n_queries`, and a run with more queries than
    // the cached truth covers is a configuration mistake, not a reason to
    // abort. `max_delta_vs_oracle` already took `.take(gt.n_queries)`; this did
    // not.
    let scored = n.min(gt.n_queries);
    // A query with fewer than two ids in common with the oracle carries no
    // rank evidence (at `--limit 1`, none can): excluded from the average
    // and counted. No query with evidence at all is a failure, not a pass.
    let mut tau_sum = 0.0f64;
    let mut tau_n = 0usize;
    for (qi, q) in quantized.iter().enumerate().take(scored) {
        if let Some(t) = kendall_tau(&q.ids, gt.neighbours(qi)) {
            tau_sum += t;
            tau_n += 1;
        }
    }
    let tau_excluded = scored - tau_n;
    let tau = if tau_n > 0 {
        tau_sum / tau_n as f64
    } else {
        f64::NAN
    };

    let mut problems: Vec<String> = Vec::new();
    if scored == 0 {
        problems.push("no queries scored".into());
    }
    if tau_n == 0 {
        problems.push(format!(
            "no evidence of rank agreement: none of the {scored} queries has two or more ids in common with the fp64 oracle"
        ));
    } else if tau.is_nan() || tau < T4_MIN_KENDALL_TAU {
        // The NaN arm is spelled out rather than left to `!(tau >= floor)`: a
        // NaN τ must fail the gate, and a bare `<` would pass it.
        problems.push(format!(
            "Kendall τ vs fp64 oracle {tau:.4} is below the {T4_MIN_KENDALL_TAU} floor: the encoding is reordering neighbours, not merely perturbing scores"
        ));
    }
    if !dominance.held {
        problems.push(format!(
            "quantization_dominance violated: {}",
            dominance.detail
        ));
    }

    TierResult {
        advisory: false,
        tier: Tier::T4QuantFidelity,
        passed: problems.is_empty(),
        detail: format!(
            "{encoding}: |Δscore| over common ids {} ({uncommon} ids not common to both lists) | Kendall τ vs fp64 oracle = {tau:.4} over {tau_n} queries ({tau_excluded} with <2 common ids excluded) | quantization_dominance {}{}",
            dist.describe(),
            if dominance.held { "holds" } else { "VIOLATED" },
            if problems.is_empty() {
                String::new()
            } else {
                format!(" | {}", problems.join("; "))
            }
        ),
        distribution: Some(dist),
    }
}

/// The rank-correlation floor for T4.
///
/// A quantizer's job is to preserve *ordering* while spending fewer bits on
/// the score; τ against the fp64 oracle measures exactly that. The floor is
/// deliberately loose — SQ8 at bfb's 0.99 quantile lands far above it, and the
/// number to *read* is the τ in the detail, per §8.4's reporting rule — but an
/// encoding whose returned order agrees with the truth no better than this is
/// not lossy, it is wrong.
pub const T4_MIN_KENDALL_TAU: f64 = 0.5;

// =========================================================================
// The conformance row
// =========================================================================

/// §8: "the results sink rejects a perf row whose conformance hash doesn't
/// match a passing run."
/// What `main` writes into an identity field when the environment did not say.
///
/// Three fields, two spellings of `None`, because they were written as bare
/// literals at the point of use and the words differ per field. They are
/// constants so the rule below can be one expression rather than a list of
/// comparisons that has to be extended by hand — which is how `isa_build` came
/// to be hashed into the row's identity, named in `main`'s warning, and left
/// out of the check.
pub const COMMIT_UNSET: &str = "unknown";
pub const QDRANT_UNSET: &str = "unpinned";
pub const ISA_UNSET: &str = "unknown";

/// Whether a row can say which builds it judged.
///
/// Free-standing so `main` can ask before it has a row to ask about, and so
/// the warning it prints and the licence bit it sets cannot disagree.
///
/// `isa_build` counts. §8 binds a perf row to "the same build", and T1 — the
/// tier `licenses_performance_claim` gates — is exactly the single-engine
/// case: a kernel microbenchmark, a cost-model row, an ISA-matrix cell. A cell
/// of the ISA matrix that cannot say which ISA it was built for is the one
/// number where the omission matters most.
pub fn identity_known(strawmann_commit: &str, qdrant_version: &str, isa_build: &str) -> bool {
    strawmann_commit != COMMIT_UNSET && qdrant_version != QDRANT_UNSET && isa_build != ISA_UNSET
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConformanceRow {
    /// The dataset's **name**, not the path it was read from. See `hash`.
    pub dataset: String,
    pub metric: Metric,
    pub dim: usize,
    pub qdrant_version: String,
    pub strawmann_commit: String,
    pub isa_build: String,
    pub tiers: Vec<TierResult>,
    pub base_checksum: u64,
    pub query_checksum: u64,
    /// The tolerance the tiers were judged under, and whether it is relative.
    ///
    /// Part of the row because it is the parameter that decides the verdicts.
    /// §8.4's whole argument is that ε must be derived rather than chosen, and
    /// a gate that binds the dataset, the builds and the outcomes but not the
    /// tolerance would accept a green row produced with `--epsilon 1.0`.
    pub epsilon: f64,
    pub epsilon_relative: bool,
    /// The `limit` every query was asked with. T1 at `limit=1` and T1 at
    /// `limit=100` are different amounts of evidence, so they are different
    /// rows.
    pub limit: usize,
}

impl ConformanceRow {
    /// The highest tier reached with everything below it green.
    ///
    /// §8.9's report has a "conformance tier reached" column; a tier that
    /// passed while a lower one failed does not count, because the lower tier
    /// is what licensed it.
    pub fn tier_reached(&self) -> Option<Tier> {
        let mut sorted: Vec<&TierResult> = self.tiers.iter().filter(|t| !t.advisory).collect();
        sorted.sort_by_key(|t| t.tier);
        let mut reached = None;
        for t in sorted {
            if !t.passed {
                break;
            }
            reached = Some(t.tier);
        }
        reached
    }

    /// Does the row name the builds it was taken on?
    ///
    /// `main` fills these with `COMMIT_UNSET`, `QDRANT_UNSET` and `ISA_UNSET`
    /// when the environment did not say. §8 binds a perf row
    /// to "the same build, on the same data": a row that cannot say which
    /// build it judged licenses nothing, whatever its tiers say. The hash is
    /// still computed and printed — it is an identity, not a verdict — but
    /// the licence bits are false. A `-dirty` commit is a known build with
    /// uncommitted changes; it stays a warning, and the suffix is in the
    /// commit string the hash binds.
    pub fn build_identity_known(&self) -> bool {
        identity_known(
            &self.strawmann_commit,
            &self.qdrant_version,
            &self.isa_build,
        )
    }

    /// §8: no performance number publishes without this being true.
    ///
    /// T1 is the right bar for a **single-engine** number — a kernel
    /// microbenchmark, a cost-model row, an ISA-matrix cell. It establishes
    /// that our arithmetic matches Qdrant's, which is what makes our cycles
    /// comparable to theirs *per unit of work*.
    pub fn licenses_performance_claim(&self) -> bool {
        self.build_identity_known()
            && matches!(self.tier_reached(), Some(t) if t >= Tier::T1ExactValue)
    }

    /// A **comparative** throughput or latency claim needs more, and T1 is not
    /// enough.
    ///
    /// §7.4: "Publish QPS-vs-recall curves; a single QPS number without its
    /// recall is meaningless and this project should never emit one." Two
    /// engines at different recall are doing different amounts of work, so
    /// their QPS is not comparable in either direction — the faster one may
    /// simply be searching less.
    ///
    /// T3 is the tier that establishes equal recall with overlapping
    /// confidence intervals, so T3 is the bar here. This split exists because
    /// the differ once printed "this run LICENSES a performance claim" on the
    /// line after "recall@10 CIs DO NOT overlap — a QPS comparison here would
    /// be at unequal recall", which is precisely the wrong number waiting to be
    /// published.
    pub fn licenses_comparative_claim(&self) -> bool {
        self.build_identity_known()
            && matches!(self.tier_reached(), Some(t) if t >= Tier::T3AnnStatistical)
    }

    /// The hash a perf row must carry to be accepted.
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        // The dataset *name*. This mixed `self.dataset` when that field held a
        // full path, which made the hash machine-dependent: the same corpus
        // under a different home directory produced a different identity, so a
        // perf row could never be matched against a conformance row taken
        // elsewhere. The JSON writer had already been fixed to strip the path
        // and the hash had not, so the two disagreed about what identifies a
        // run.
        mix(self.dataset.as_bytes());
        mix(self.metric.as_str().as_bytes());
        // Widths fixed at u64: `usize::to_le_bytes` is 4 bytes on a 32-bit
        // host, and the hash is an identity that has to be reproducible on
        // whichever machine recomputes it.
        mix(&(self.dim as u64).to_le_bytes());
        mix(self.qdrant_version.as_bytes());
        mix(self.strawmann_commit.as_bytes());
        mix(self.isa_build.as_bytes());
        mix(&self.base_checksum.to_le_bytes());
        mix(&self.query_checksum.to_le_bytes());
        mix(&self.epsilon.to_le_bytes());
        mix(&[u8::from(self.epsilon_relative)]);
        mix(&(self.limit as u64).to_le_bytes());
        // Which tiers ran, how many, and how each came out. Mixing only the
        // pass bits made a row that ran T0+T1 (two passes) collide with one
        // that ran T0+T2 with T1 absent, and a row that never ran T1 hash the
        // same as one that ran it — the gate then accepted evidence that was
        // never gathered.
        mix(&(self.tiers.len() as u64).to_le_bytes());
        for t in &self.tiers {
            mix(t.tier.as_str().as_bytes());
            mix(&[u8::from(t.passed)]);
        }
        h
    }
}

/// The gate §8 describes, as a function.
///
/// This is the *specification* of the gate and is unit-tested as one;
/// `bench/harness/results.py:insert_perf` is the copy that actually runs,
/// because the perf rows live in SQLite on the Python side. Two implementations
/// of one rule is a risk, and the mitigation is that this one is executable and
/// tested rather than prose.
#[allow(dead_code)]
pub fn accept_performance_row(
    conformance: &ConformanceRow,
    claimed_hash: u64,
) -> Result<(), String> {
    if conformance.hash() != claimed_hash {
        return Err(
            "performance row's conformance hash does not match any passing run (§8): rejected"
                .into(),
        );
    }
    if !conformance.licenses_performance_claim() {
        return Err(format!(
            "conformance reached {:?}, which does not license a performance claim: §8.5 T1 is the tier that does",
            conformance.tier_reached()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn identity_needs_all_three_builds() {
        // `isa_build` is hashed into the row's identity and named in `main`'s
        // warning, and was left out of the check: a run without
        // STRAWMANN_ISA_BUILD licensed performance claims while the same run
        // without STRAWMANN_COMMIT was refused. T1, the tier this gates, is
        // the single-engine case — an ISA-matrix cell that cannot say which
        // ISA it was built for is where the omission matters most.
        assert!(identity_known("abc", "1.19.0", "avx512"));
        assert!(!identity_known(COMMIT_UNSET, "1.19.0", "avx512"));
        assert!(!identity_known("abc", QDRANT_UNSET, "avx512"));
        assert!(!identity_known("abc", "1.19.0", ISA_UNSET));
    }
    use super::*;
    use tolerance::calibrate;

    fn eps(metric: Metric, value: f64) -> Epsilon {
        let mut e = calibrate(
            metric,
            8,
            Distribution::default(),
            Distribution::default(),
            1.0,
        );
        e.value = value;
        e.relative = false;
        e
    }

    fn gt3() -> GroundTruth {
        GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
            base_checksum: 7,
            query_checksum: 9,
            condition: None,
            n_matching: None,
        }
    }

    #[test]
    fn t0_ignores_timings_but_catches_shape_differences() {
        let a = WireShape {
            result_count: 10,
            id_variant: IdVariant::Num,
            has_time: true,
            has_payload: false,
            has_vectors: false,
            status_code: 0,
        };
        assert!(compare_shape(&a, &a).passed);

        let mut b = a.clone();
        b.id_variant = IdVariant::Uuid;
        let r = compare_shape(&a, &b);
        assert!(!r.passed);
        assert!(r.detail.contains("id variant"));

        // §8.5 T0: "unsupported operations must produce the same gRPC status
        // code as Qdrant".
        let mut c = a.clone();
        c.status_code = 12;
        assert!(!compare_shape(&a, &c).passed);
    }

    #[test]
    fn t0_catches_a_missing_time_field() {
        // §2: bfb reads `time` as the server-side timing. Absent means the
        // histogram silently records zero.
        let a = WireShape {
            result_count: 1,
            id_variant: IdVariant::Num,
            has_time: true,
            has_payload: false,
            has_vectors: false,
            status_code: 0,
        };
        let mut b = a.clone();
        b.has_time = false;
        let r = compare_shape(&a, &b);
        assert!(!r.passed);
        assert!(r.detail.contains("time"));
    }

    #[test]
    fn t1_passes_within_epsilon_and_reports_the_distribution() {
        let ours = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let theirs = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0 + 1e-7, 2.0, 3.0],
        }];
        let r = compare_exact(&ours, &theirs, &gt3(), &eps(Metric::Euclid, 1e-6), 3);
        assert!(r.passed);
        // §8.4: never a bare pass/fail.
        let d = r.distribution.unwrap();
        assert_eq!(d.count, 3);
        assert!(d.max > 0.0 && d.max < 1e-6);
    }

    #[test]
    fn t1_fails_beyond_epsilon_and_names_the_oracle_verdict() {
        let ours = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.5, 2.0, 3.0],
        }];
        let theirs = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let r = compare_exact(&ours, &theirs, &gt3(), &eps(Metric::Euclid, 1e-6), 3);
        assert!(!r.passed);
        // §8.2: the arbiter says who is wrong, not just that someone is.
        assert!(r.detail.contains("oracle"));
        assert!(r.detail.contains("WE are further off"));
    }

    #[test]
    fn t2_treats_tied_scores_as_an_unordered_class() {
        // §8.5 T2: "their ordering is arbitrary in both engines".
        let ours = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![5.0, 5.0, 9.0],
        }];
        let theirs = vec![Returned {
            ids: vec![2, 1, 3],
            scores: vec![5.0, 5.0, 9.0],
        }];
        let r = compare_ranks(&ours, &theirs, &eps(Metric::Dot, 1e-6));
        assert!(r.passed);
        assert!(
            r.detail.contains("0/"),
            "no class should differ: {}",
            r.detail
        );
    }

    #[test]
    fn kendall_tau_is_one_for_identical_and_negative_for_reversed() {
        assert!((kendall_tau(&[1, 2, 3, 4], &[1, 2, 3, 4]).unwrap() - 1.0).abs() < 1e-12);
        assert!((kendall_tau(&[1, 2, 3, 4], &[4, 3, 2, 1]).unwrap() + 1.0).abs() < 1e-12);
        // Disjoint lists have no common elements to compare: no evidence.
        assert_eq!(kendall_tau(&[1, 2], &[8, 9]), None);
    }

    #[test]
    fn rbo_is_one_for_identical_lists_and_zero_for_disjoint_ones() {
        // The normalisation exists so this reads as agreement. Un-normalised,
        // identical length-10 lists at p=0.9 score 0.651, which alongside a
        // Kendall tau of 1.0 looks like a contradiction.
        let same = rank_biased_overlap(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            0.9,
        );
        assert!((same - 1.0).abs() < 1e-12, "identical lists scored {same}");

        let disjoint = rank_biased_overlap(&[1, 2, 3], &[7, 8, 9], 0.9);
        assert!(disjoint.abs() < 1e-12, "disjoint lists scored {disjoint}");
    }

    #[test]
    fn rbo_is_higher_for_lists_agreeing_at_the_top() {
        let top_agrees = rank_biased_overlap(&[1, 2, 3, 4], &[1, 2, 9, 8], 0.9);
        let top_differs = rank_biased_overlap(&[1, 2, 3, 4], &[9, 8, 3, 4], 0.9);
        assert!(top_agrees > top_differs, "{top_agrees} vs {top_differs}");
    }

    #[test]
    fn t3_flags_unequal_recall_rather_than_comparing_qps_anyway() {
        // §7.4: "Compare at equal recall."
        //
        // k must be at least 10 and each result list at least 10 long, or
        // recall@10 is not measurable and the harness correctly refuses to
        // call it matched.
        let k = 10;
        let nq = 4;
        let ids: Vec<u32> = (0..nq)
            .flat_map(|_| (1..=k as u32).collect::<Vec<_>>())
            .collect();
        let scores: Vec<f64> = (0..nq)
            .flat_map(|_| (1..=k).map(|i| i as f64).collect::<Vec<_>>())
            .collect();
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: nq,
            dim: 4,
            k,
            ids,
            scores,
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };

        let good: Vec<Returned> = (0..nq)
            .map(|_| Returned {
                ids: (1..=k as u32).collect(),
                scores: (1..=k).map(|i| i as f64).collect(),
            })
            .collect();
        let bad: Vec<Returned> = (0..nq)
            .map(|_| Returned {
                ids: (900..900 + k as u32).collect(),
                scores: (0..k).map(|_| 999.0).collect(),
            })
            .collect();

        let (r, a, b) = compare_ann(&good, &bad, &gt, &eps(Metric::Euclid, 1e-9), k, None);
        assert!(!r.passed);
        assert!(r.detail.contains("DO NOT overlap"));
        assert!(
            (a.recall_at_10 - 1.0).abs() < 1e-12,
            "got {}",
            a.recall_at_10
        );
        assert!(b.recall_at_10.abs() < 1e-12, "got {}", b.recall_at_10);
    }

    #[test]
    fn recall_is_not_reported_when_the_request_could_not_produce_it() {
        // A run with limit=3 mechanically caps recall@10 at 0.3. Reporting
        // that number would present a request parameter as a quality result.
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 100,
            ids: (0..100).collect(),
            scores: (0..100).map(f64::from).collect(),
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let short = vec![Returned {
            ids: vec![0, 1, 2],
            scores: vec![0.0, 1.0, 2.0],
        }];
        let g = relevance::evaluate(&gt, &short, &eps(Metric::Euclid, 1e-9), 3, None);

        assert!(!g.recall_at_1.is_nan());
        assert!(
            g.recall_at_10.is_nan(),
            "recall@10 should be unmeasurable at limit 3"
        );
        assert!(g.recall_at_100.is_nan());
        // And an unmeasurable recall can never be "matched".
        assert!(!relevance::recall_is_matched(&g, &g));
    }

    /// README: none of the tiers passes on nothing. T3 used to: two empty
    /// runs both evaluated to recall 0 with a (0, 0) interval, which overlaps.
    #[test]
    fn t3_does_not_pass_on_nothing() {
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 0,
            dim: 4,
            k: 10,
            ids: vec![],
            scores: vec![],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let (r, _, _) = compare_ann(&[], &[], &gt, &eps(Metric::Euclid, 1e-9), 10, None);
        assert!(!r.passed, "{}", r.detail);
        assert!(r.detail.contains("n=0"), "{}", r.detail);
    }

    #[test]
    fn t4_reports_the_distribution_and_gates_on_dominance_and_rank_correlation() {
        let q = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.1, 2.1, 3.1],
        }];
        let f = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let held = crate::metamorphic::quantization_dominance(&[(1.0, 0.9), (2.0, 0.95)]);
        let r = quantization_fidelity(&q, &f, &gt3(), "binary", &held);
        assert!(r.passed, "{}", r.detail);
        assert!(r.detail.contains("binary"));
        assert!(r.detail.contains("Kendall"));
        assert!(r.distribution.unwrap().max > 0.0);

        // §8.6: recall falling with oversampling under rescore is the failure
        // T4 exists to surface, and it must fail the tier, not decorate it.
        let fell = crate::metamorphic::quantization_dominance(&[(1.0, 0.9), (2.0, 0.5)]);
        let r = quantization_fidelity(&q, &f, &gt3(), "binary", &fell);
        assert!(!r.passed);
        assert!(r.detail.contains("VIOLATED"), "{}", r.detail);

        // An encoding that returns the oracle's neighbours in reverse order
        // has broken the ranking, and τ says so.
        let reversed = vec![Returned {
            ids: vec![3, 2, 1],
            scores: vec![3.1, 2.1, 1.1],
        }];
        let r = quantization_fidelity(&reversed, &f, &gt3(), "binary", &held);
        assert!(!r.passed);
        assert!(r.detail.contains("floor"), "{}", r.detail);

        // And nothing scored is not fidelity.
        let r = quantization_fidelity(&[], &[], &gt3(), "binary", &held);
        assert!(!r.passed);
    }

    #[test]
    fn t1_fails_when_ids_disagree_even_though_every_score_matches() {
        // §8.3: "A divergence in the *ids* ... is never excused." Same scores
        // at every position; one point swapped for another outside any tie.
        // Before ids were compared this passed T1, the tier that licenses the
        // performance claim, with an id-mapping bug.
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let theirs = vec![Returned {
            ids: vec![1, 9, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let e = eps(Metric::Euclid, 1e-6);
        let r = compare_exact(&ours, &theirs, &gt, &e, 3);
        assert!(!r.passed, "{}", r.detail);
        assert!(r.detail.contains("id sets differ"), "{}", r.detail);

        // Same multiset, but an id moved across a non-tie: still a defect.
        let swapped = vec![Returned {
            ids: vec![2, 1, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let r = compare_exact(&ours, &swapped, &gt, &e, 3);
        assert!(!r.passed, "{}", r.detail);
        assert!(
            r.detail.contains("across a gap wider than ε"),
            "{}",
            r.detail
        );

        // Our own ids disagreeing with the oracle fails regardless of Qdrant.
        let both_wrong = vec![Returned {
            ids: vec![1, 9, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let r = compare_exact(&both_wrong, &both_wrong, &gt, &e, 3);
        assert!(!r.passed, "{}", r.detail);
        assert!(r.detail.contains("oracle does not"), "{}", r.detail);
    }

    #[test]
    fn t1_accepts_ids_permuted_within_an_exact_tie() {
        // §8.5 T2: "their ordering is arbitrary in both engines" — a tie
        // broken the other way is not an id divergence.
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![1.0, 1.0, 3.0],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 1.0, 3.0],
        }];
        let theirs = vec![Returned {
            ids: vec![2, 1, 3],
            scores: vec![1.0, 1.0, 3.0],
        }];
        let r = compare_exact(&ours, &theirs, &gt, &eps(Metric::Euclid, 1e-6), 3);
        assert!(r.passed, "{}", r.detail);

        // And a tie at the oracle's cut may be resolved either way: id 4 is
        // not in the truth's top 3 but sits exactly at the 3rd score.
        let cut = vec![Returned {
            ids: vec![1, 2, 4],
            scores: vec![1.0, 1.0, 3.0],
        }];
        let r = compare_exact(&cut, &cut, &gt, &eps(Metric::Euclid, 1e-6), 3);
        assert!(r.passed, "{}", r.detail);
    }

    #[test]
    fn t2_does_not_pass_on_nothing() {
        let e = eps(Metric::Euclid, 1e-6);
        assert!(!compare_ranks(&[], &[], &e).passed);
        // Every query answered empty by both engines: no segment, τ None,
        // RBO 1.0 -- and it used to pass.
        let empty = vec![
            Returned {
                ids: vec![],
                scores: vec![]
            };
            3
        ];
        let r = compare_ranks(&empty, &empty, &e);
        assert!(!r.passed, "{}", r.detail);
        assert!(r.detail.contains("answered empty by both"), "{}", r.detail);
        // One real query among empties is enough to judge.
        let mut mixed = empty.clone();
        mixed[0] = Returned {
            ids: vec![1, 2],
            scores: vec![2.0, 1.0],
        };
        assert!(compare_ranks(&mixed, &mixed, &e).passed);
    }

    #[test]
    fn a_duplicated_id_is_a_divergence_not_a_tie() {
        let e = eps(Metric::Euclid, 1e-6);
        let a = Returned {
            ids: vec![1, 2, 3],
            scores: vec![5.0, 5.0, 5.0],
        };
        let b = Returned {
            ids: vec![1, 1, 3],
            scores: vec![5.0, 5.0, 5.0],
        };
        let err = ids_agree_up_to_ties(&a, &b, &e).unwrap_err();
        assert!(err.contains("more than once"), "{err}");
    }

    #[test]
    fn t1_does_not_pass_on_nothing() {
        // An empty comparison has an empty distribution and no violations,
        // and used to be a green T1 licensing a performance claim.
        let r = compare_exact(&[], &[], &gt3(), &eps(Metric::Euclid, 1e-6), 3);
        assert!(!r.passed);
        assert!(r.detail.contains("no queries compared"), "{}", r.detail);

        // Lists shorter than the limit, when the collection could fill it,
        // are a structural failure even when the two engines agree.
        let short = vec![Returned {
            ids: vec![1],
            scores: vec![1.0],
        }];
        let r = compare_exact(&short, &short, &gt3(), &eps(Metric::Euclid, 1e-6), 3);
        assert!(!r.passed, "{}", r.detail);
        assert!(r.detail.contains("expected 3 results"), "{}", r.detail);

        // But a collection smaller than the limit legitimately fills less.
        let mut small = gt3();
        small.n_base = 1;
        small.k = 1;
        small.ids = vec![1];
        small.scores = vec![1.0];
        let r = compare_exact(&short, &short, &small, &eps(Metric::Euclid, 1e-6), 3);
        assert!(r.passed, "{}", r.detail);
    }

    #[test]
    fn t2_fails_when_the_tie_multisets_differ_and_passes_across_a_shifted_class_cut() {
        let e = eps(Metric::Dot, 1e-6);
        // Different points inside the first tie class.
        let ours = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![5.0, 5.0, 9.0],
        }];
        let theirs = vec![Returned {
            ids: vec![1, 7, 3],
            scores: vec![5.0, 5.0, 9.0],
        }];
        let r = compare_ranks(&ours, &theirs, &e);
        assert!(!r.passed, "{}", r.detail);
        assert!(r.detail.contains("1/"), "{}", r.detail);

        // The lists agree everywhere to within ε but draw a class boundary one
        // apart (0.9ε vs 1.1ε gaps). Class-by-class zip called that a
        // mismatch; cutting only where both cut does not.
        let a = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 1.0 + 0.9e-6, 2.0],
        }];
        let b = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 1.0 + 1.1e-6, 2.0],
        }];
        let r = compare_ranks(&a, &b, &e);
        assert!(r.passed, "{}", r.detail);

        // Nothing compared is not agreement.
        assert!(!compare_ranks(&[], &[], &e).passed);
    }

    #[test]
    fn the_hash_binds_the_limit_and_the_tier_identities() {
        let base = ConformanceRow {
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
            tiers: vec![
                TierResult {
                    advisory: false,
                    tier: Tier::T0Wire,
                    passed: true,
                    detail: String::new(),
                    distribution: None,
                },
                TierResult {
                    advisory: false,
                    tier: Tier::T1ExactValue,
                    passed: true,
                    detail: String::new(),
                    distribution: None,
                },
            ],
        };
        let h = base.hash();

        let mut l = base.clone();
        l.limit = 100;
        assert_ne!(
            h,
            l.hash(),
            "T1 at limit 100 is different evidence from T1 at limit 10"
        );

        // Two passes are not the same two passes: T0+T2 must not hash like T0+T1.
        let mut t = base.clone();
        t.tiers[1].tier = Tier::T2RankTies;
        assert_ne!(h, t.hash());

        // And a row that never ran T1 is not a row that ran it.
        let mut fewer = base.clone();
        fewer.tiers.pop();
        assert_ne!(h, fewer.hash());

        let mut m = base.clone();
        m.metric = Metric::Dot;
        assert_ne!(h, m.hash());
    }

    #[test]
    fn an_advisory_result_does_not_decide_the_tier_reached() {
        // Qdrant's T4 is pushed beside strawmann's. A stable sort kept
        // strawmann's first, so its pass was the tier reached whatever
        // Qdrant's said -- and the reverse order would have filed Qdrant's
        // failure as the row's. Advisory results are reported, not counted.
        let ok = |tier: Tier| TierResult {
            advisory: false,
            tier,
            passed: true,
            detail: String::new(),
            distribution: None,
        };
        let mut row = ConformanceRow {
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
            tiers: vec![
                ok(Tier::T0Wire),
                ok(Tier::T1ExactValue),
                ok(Tier::Metamorphic),
                ok(Tier::T2RankTies),
                ok(Tier::T3AnnStatistical),
                ok(Tier::T4QuantFidelity),
                TierResult {
                    advisory: true,
                    tier: Tier::T4QuantFidelity,
                    passed: false,
                    detail: "qdrant".into(),
                    distribution: None,
                },
            ],
        };
        assert_eq!(row.tier_reached(), Some(Tier::T4QuantFidelity));
        // And strawmann's own failing T4 still stops it.
        row.tiers[5].passed = false;
        assert_eq!(row.tier_reached(), Some(Tier::T3AnnStatistical));
    }

    #[test]
    fn tier_reached_stops_at_the_first_failure() {
        let row = ConformanceRow {
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
            tiers: vec![
                TierResult {
                    advisory: false,
                    tier: Tier::T0Wire,
                    passed: true,
                    detail: String::new(),
                    distribution: None,
                },
                TierResult {
                    advisory: false,
                    tier: Tier::T1ExactValue,
                    passed: false,
                    detail: String::new(),
                    distribution: None,
                },
                // A later tier passing must not raise the reached level past
                // the failure below it.
                TierResult {
                    advisory: false,
                    tier: Tier::T3AnnStatistical,
                    passed: true,
                    detail: String::new(),
                    distribution: None,
                },
            ],
        };
        assert_eq!(row.tier_reached(), Some(Tier::T0Wire));
        assert!(!row.licenses_performance_claim());
    }

    #[test]
    fn t1_is_the_tier_that_licenses_a_performance_claim() {
        let mut row = ConformanceRow {
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
            tiers: vec![
                TierResult {
                    advisory: false,
                    tier: Tier::T0Wire,
                    passed: true,
                    detail: String::new(),
                    distribution: None,
                },
                TierResult {
                    advisory: false,
                    tier: Tier::T1ExactValue,
                    passed: true,
                    detail: String::new(),
                    distribution: None,
                },
            ],
        };
        assert!(row.licenses_performance_claim());
        assert!(accept_performance_row(&row, row.hash()).is_ok());

        // §8: a perf row carrying the wrong hash is rejected, not flagged.
        assert!(accept_performance_row(&row, row.hash() ^ 1).is_err());

        // And T0 alone does not license anything.
        row.tiers[1].passed = false;
        assert!(accept_performance_row(&row, row.hash()).is_err());
    }

    #[test]
    fn the_conformance_hash_changes_with_every_input_that_matters() {
        let base = ConformanceRow {
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
            tiers: vec![TierResult {
                advisory: false,
                tier: Tier::T1ExactValue,
                passed: true,
                detail: String::new(),
                distribution: None,
            }],
        };
        let h = base.hash();

        // §8.9: "The Qdrant version is pinned and recorded per row."
        let mut v = base.clone();
        v.qdrant_version = "1.20.0".into();
        assert_ne!(h, v.hash());

        // §7.1: results from different ISA builds never share a row.
        let mut i = base.clone();
        i.isa_build = "avx2".into();
        assert_ne!(h, i.hash());

        // §4.3: dataset checksums are part of the identity.
        let mut c = base.clone();
        c.base_checksum = 99;
        assert_ne!(h, c.hash());
    }

    #[test]
    fn section_8_1_claims_are_encoded_honestly() {
        // "Anything promising ID-level equality for HNSW results is promising
        // something false. The spec does not."
        assert!(!Claim::BitExact.achievable(Scope::ExactSearch));
        assert!(!Claim::BitExact.achievable(Scope::AnnSearch));
        assert!(Claim::ValueWithinEpsilon.achievable(Scope::ExactSearch));
        assert!(!Claim::ValueWithinEpsilon.achievable(Scope::AnnSearch));
        assert!(Claim::StatisticalEquivalence.achievable(Scope::AnnSearch));
    }
}

#[cfg(test)]
mod licensing_tests {
    use super::*;

    #[test]
    fn the_hash_binds_the_tolerance_the_tiers_were_judged_under() {
        // §8.4 makes ε the parameter that decides every verdict, so a gate that
        // did not bind it would accept a green row produced with --epsilon 1.0
        // as interchangeable with a calibrated one.
        let a = row_with(vec![(Tier::T0Wire, true), (Tier::T1ExactValue, true)]);
        let mut b = row_with(vec![(Tier::T0Wire, true), (Tier::T1ExactValue, true)]);
        assert_eq!(a.hash(), b.hash(), "identical rows must agree");

        b.epsilon = 1.0;
        assert_ne!(
            a.hash(),
            b.hash(),
            "a different ε is a different conformance run"
        );

        let mut c = row_with(vec![(Tier::T0Wire, true), (Tier::T1ExactValue, true)]);
        c.epsilon_relative = true;
        assert_ne!(
            a.hash(),
            c.hash(),
            "absolute and relative ε are not the same tolerance"
        );
    }

    #[test]
    fn the_hash_binds_the_builds_under_test() {
        let a = row_with(vec![(Tier::T1ExactValue, true)]);
        let mut b = row_with(vec![(Tier::T1ExactValue, true)]);
        b.strawmann_commit = "another".into();
        assert_ne!(
            a.hash(),
            b.hash(),
            "§8 binds a perf row to 'the same build'"
        );
    }

    fn row_with(tiers: Vec<(Tier, bool)>) -> ConformanceRow {
        ConformanceRow {
            dataset: "t".into(),
            metric: Metric::Euclid,
            dim: 4,
            qdrant_version: "1.19.0".into(),
            strawmann_commit: "test".into(),
            isa_build: "native".into(),
            base_checksum: 0,
            query_checksum: 0,
            epsilon: 1e-6,
            epsilon_relative: false,
            limit: 10,
            tiers: tiers
                .into_iter()
                .map(|(tier, passed)| TierResult {
                    advisory: false,
                    tier,
                    passed,
                    detail: String::new(),
                    distribution: None,
                })
                .collect(),
        }
    }

    /// The exact shape of the SIFT1M run: wire and value equality hold, ranks
    /// agree, but the two engines are at materially different recall. A
    /// single-engine cycle count is still meaningful; a QPS *comparison* is not.
    #[test]
    fn t3_failure_blocks_a_comparative_claim_but_not_a_single_engine_one() {
        let r = row_with(vec![
            (Tier::T0Wire, true),
            (Tier::T1ExactValue, true),
            (Tier::T2RankTies, true),
            (Tier::T3AnnStatistical, false),
        ]);
        assert!(
            r.licenses_performance_claim(),
            "T1 passed: kernel numbers are valid"
        );
        assert!(
            !r.licenses_comparative_claim(),
            "T3 failed: the engines are at unequal recall, so QPS is not comparable"
        );
    }

    /// A row that does not know which build it judged licenses nothing: the
    /// hash would bind a perf row to "the same build" without naming one.
    #[test]
    fn an_unknown_build_identity_licenses_nothing() {
        let green = vec![
            (Tier::T0Wire, true),
            (Tier::T1ExactValue, true),
            (Tier::T2RankTies, true),
            (Tier::T3AnnStatistical, true),
        ];
        let mut r = row_with(green.clone());
        r.strawmann_commit = "unknown".into();
        assert!(!r.licenses_performance_claim());
        assert!(!r.licenses_comparative_claim());
        assert!(accept_performance_row(&r, r.hash()).is_err());
        let mut r = row_with(green);
        r.qdrant_version = "unpinned".into();
        assert!(!r.licenses_performance_claim());
        assert!(!r.licenses_comparative_claim());
        // Uncommitted changes are a known build, flagged in the string.
        r.qdrant_version = "1.19.0".into();
        r.strawmann_commit = "abc-dirty".into();
        assert!(r.licenses_performance_claim());
    }

    #[test]
    fn t3_pass_licenses_both() {
        let r = row_with(vec![
            (Tier::T0Wire, true),
            (Tier::T1ExactValue, true),
            (Tier::T2RankTies, true),
            (Tier::T3AnnStatistical, true),
        ]);
        assert!(r.licenses_performance_claim());
        assert!(r.licenses_comparative_claim());
    }

    #[test]
    fn a_t1_failure_blocks_everything() {
        let r = row_with(vec![(Tier::T0Wire, true), (Tier::T1ExactValue, false)]);
        assert!(!r.licenses_performance_claim());
        assert!(!r.licenses_comparative_claim());
    }
}

#[cfg(test)]
mod t1_adjudication_tests {
    use super::*;

    fn gt_one(scores: &[f64]) -> GroundTruth {
        GroundTruth {
            metric: Metric::Cosine,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: scores.len(),
            ids: (0..scores.len() as u32).collect(),
            scores: scores.to_vec(),
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        }
    }

    fn ret(ids: &[u32], scores: &[f64]) -> Vec<Returned> {
        vec![Returned {
            ids: ids.to_vec(),
            scores: scores.to_vec(),
        }]
    }

    fn eps(v: f64) -> Epsilon {
        Epsilon {
            metric: Metric::Cosine,
            dim: 1536,
            value: v,
            relative: false,
            qdrant_spread: Distribution::default(),
            strawmann_spread: Distribution::default(),
            multiple: 4.0,
        }
    }

    /// The headline case, reduced: the two engines disagree by more than ε, and
    /// the oracle says **we** are the accurate one. Failing here would mean
    /// failing a conformance tier for being more correct, and the only remedy
    /// would be to reproduce Qdrant's rounding error on purpose.
    #[test]
    fn we_pass_when_qdrant_is_the_one_that_drifted() {
        let truth = gt_one(&[0.9059384]);
        let ours = ret(&[0], &[0.9059384]); // exact
        let theirs = ret(&[0], &[0.9059391]); // 7e-7 off
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 1);
        assert!(r.passed, "detail: {}", r.detail);
        assert!(r.detail.contains("ACCEPTED"), "{}", r.detail);
        assert!(r.detail.contains("QDRANT is further off"), "{}", r.detail);
    }

    /// The tier must still catch a genuine defect: if we are the ones further
    /// from truth, no amount of "Qdrant also differs" excuses it.
    #[test]
    fn we_fail_when_we_are_the_one_that_drifted() {
        let truth = gt_one(&[0.9059384]);
        let ours = ret(&[0], &[0.9059500]); // way off
        let theirs = ret(&[0], &[0.9059385]); // nearly exact
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 1);
        assert!(!r.passed, "detail: {}", r.detail);
        assert!(r.detail.contains("WE are further off"), "{}", r.detail);
    }

    /// Both drifting badly is a failure even if we happen to be marginally
    /// closer — accuracy is graded against ε, not against a race to the bottom.
    #[test]
    fn both_inaccurate_still_fails() {
        let truth = gt_one(&[0.9059384]);
        let ours = ret(&[0], &[0.9070000]);
        let theirs = ret(&[0], &[0.9080000]);
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 1);
        assert!(
            !r.passed,
            "we are closer but nowhere near the truth: {}",
            r.detail
        );
    }

    /// Agreement within ε short-circuits the whole question.
    #[test]
    fn agreement_within_epsilon_passes_without_arbitration() {
        let truth = gt_one(&[0.5]);
        let ours = ret(&[0], &[0.5]);
        let theirs = ret(&[0], &[0.5]);
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 1);
        assert!(r.passed);
        assert!(
            !r.detail.contains("ACCEPTED"),
            "nothing to arbitrate: {}",
            r.detail
        );
    }

    /// Both engines returning the *same* wrong last point used to pass: the
    /// last returned id trivially tied its own cut, so the oracle id check
    /// excused it whatever its score, and the score deviation is only
    /// consulted when the engines disagree. Agreement between the engines is
    /// not agreement with the truth.
    #[test]
    fn t1_fails_when_both_engines_agree_on_a_wrong_last_id() {
        let truth = GroundTruth {
            metric: Metric::Cosine,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours = ret(&[1, 2, 99], &[1.0, 2.0, 50.0]);
        let theirs = ret(&[1, 2, 99], &[1.0, 2.0, 50.0]);
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 3);
        assert!(
            !r.passed,
            "id 99 is not an oracle neighbour and 3 is missing: {}",
            r.detail
        );
        assert!(
            r.detail.contains("ids the fp64 oracle does not"),
            "{}",
            r.detail
        );
    }

    /// The two-sided rule still excuses a genuine boundary tie: the returned
    /// cut and the truth's cut carry the same score, and which of the tied
    /// points made the list is arbitrary.
    #[test]
    fn t1_still_excuses_a_genuinely_tied_oracle_cut() {
        let truth = GroundTruth {
            metric: Metric::Cosine,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: 3,
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours = ret(&[1, 2, 99], &[1.0, 2.0, 3.0]);
        let theirs = ret(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 3);
        assert!(r.passed, "99 ties 3 at the cut on both sides: {}", r.detail);
    }

    /// A structural divergence is never excused by accuracy.
    #[test]
    fn a_result_count_mismatch_still_fails() {
        let truth = gt_one(&[0.5, 0.4]);
        let ours = ret(&[0, 1], &[0.5, 0.4]);
        let theirs = ret(&[0], &[0.5]);
        let r = compare_exact(&ours, &theirs, &truth, &eps(1e-7), 2);
        assert!(!r.passed, "{}", r.detail);
    }

    // =====================================================================
    // The relative/absolute ε confusion, pinned so it cannot return
    // =====================================================================

    #[test]
    fn t1_accepts_us_when_the_oracle_says_qdrant_drifted_on_a_relative_metric() {
        // SIFT1M's metric and dimension, with the ε the differ derives when no
        // calibration run is supplied.
        let e = tolerance::calibrate(
            Metric::Euclid,
            128,
            Distribution::default(),
            Distribution::default(),
            tolerance::DEFAULT_MULTIPLE,
        );
        assert!(e.relative, "euclid's tolerance is relative (§8.4)");

        // A euclid distance of 300, typical for SIFT descriptors. We are one
        // fp32 ulp off the fp64 truth; Qdrant is a hundred times further.
        let truth = 300.0f64;
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 1,
            n_queries: 1,
            dim: 128,
            k: 1,
            ids: vec![7],
            scores: vec![truth],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours = vec![Returned {
            ids: vec![7],
            scores: vec![truth + 3.0e-5],
        }];
        let theirs = vec![Returned {
            ids: vec![7],
            scores: vec![truth + 3.0e-3],
        }];

        let r = compare_exact(&ours, &theirs, &gt, &e, 1);
        assert!(
            r.passed,
            "we are 100x closer to the fp64 oracle than Qdrant and T1 failed: {}",
            r.detail
        );
        assert!(
            r.detail.contains("ACCEPTED"),
            "the verdict must say why: {}",
            r.detail
        );
    }

    #[test]
    fn t1_still_fails_when_we_are_the_engine_that_drifted() {
        // The other direction, so the fix above is not simply "always pass".
        let e = tolerance::calibrate(
            Metric::Euclid,
            128,
            Distribution::default(),
            Distribution::default(),
            tolerance::DEFAULT_MULTIPLE,
        );
        let truth = 300.0f64;
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 1,
            n_queries: 1,
            dim: 128,
            k: 1,
            ids: vec![7],
            scores: vec![truth],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours = vec![Returned {
            ids: vec![7],
            scores: vec![truth + 3.0e-3],
        }];
        let theirs = vec![Returned {
            ids: vec![7],
            scores: vec![truth + 3.0e-5],
        }];

        let r = compare_exact(&ours, &theirs, &gt, &e, 1);
        assert!(
            !r.passed,
            "we drifted further than Qdrant; that is a real defect"
        );
    }

    #[test]
    fn a_differing_query_count_is_structural_and_never_silently_truncated() {
        let e = eps(1e-6);
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 4,
            n_queries: 2,
            dim: 2,
            k: 1,
            ids: vec![0, 1],
            scores: vec![1.0, 1.0],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        // Identical where they overlap, but one engine answered half the queries.
        let ours = vec![
            Returned {
                ids: vec![0],
                scores: vec![1.0],
            },
            Returned {
                ids: vec![1],
                scores: vec![1.0],
            },
        ];
        let theirs = vec![Returned {
            ids: vec![0],
            scores: vec![1.0],
        }];

        let r = compare_exact(&ours, &theirs, &gt, &e, 1);
        assert!(
            !r.passed,
            "answering fewer queries is a compatibility defect"
        );
        assert!(
            r.detail.contains("2 vs 1 queries"),
            "and it must say so: {}",
            r.detail
        );
    }
}

/// The boundary-tie exemption: an id at the k-th/(k+1)-th tie is a coin toss
/// in both engines, and must not be a T1 or T2 red. Everything else stays a
/// violation.
#[cfg(test)]
mod boundary_tie_tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    const EPS: f64 = 1e-6;

    fn e() -> Epsilon {
        Epsilon {
            metric: Metric::Euclid,
            dim: 8,
            value: EPS,
            relative: false,
            qdrant_spread: Distribution::default(),
            strawmann_spread: Distribution::default(),
            multiple: 4.0,
        }
    }

    fn r(ids: &[u32], scores: &[f64]) -> Returned {
        Returned {
            ids: ids.to_vec(),
            scores: scores.to_vec(),
        }
    }

    // (a) identical except the last id, both last scores equal.
    #[test]
    fn a_boundary_tie_broken_differently_is_excused() {
        let a = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let b = r(&[1, 2, 4], &[1.0, 2.0, 3.0]);
        assert!(ids_agree_up_to_ties(&a, &b, &e()).is_ok());
        assert!(ids_agree_up_to_ties(&b, &a, &e()).is_ok());
    }

    // (b) the two last scores differ by ε/2: still a tie.
    #[test]
    fn a_boundary_tie_within_half_epsilon_is_excused() {
        let a = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let b = r(&[1, 2, 4], &[1.0, 2.0, 3.0 + EPS / 2.0]);
        assert!(ids_agree_up_to_ties(&a, &b, &e()).is_ok());
        assert!(ids_agree_up_to_ties(&b, &a, &e()).is_ok());
        let c = r(&[1, 2, 4], &[1.0, 2.0, 3.0 - EPS / 2.0]);
        assert!(ids_agree_up_to_ties(&a, &c, &e()).is_ok());
    }

    // (c) the two last scores differ by 2ε: not a tie, a different point.
    #[test]
    fn a_last_id_two_epsilon_past_the_other_cut_is_a_violation() {
        let a = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let b = r(&[1, 2, 4], &[1.0, 2.0, 3.0 + 2.0 * EPS]);
        let err = ids_agree_up_to_ties(&a, &b, &e()).unwrap_err();
        assert!(err.contains("id sets differ"), "{err}");
        assert!(err.contains("[3]") && err.contains("[4]"), "{err}");
        assert!(ids_agree_up_to_ties(&b, &a, &e()).is_err());
        let c = r(&[1, 2, 4], &[1.0, 2.0, 3.0 - 2.0 * EPS]);
        assert!(ids_agree_up_to_ties(&a, &c, &e()).is_err());
    }

    // (d) an interior swap outside a tie.
    #[test]
    fn an_interior_swap_across_a_gap_is_a_violation_even_with_a_boundary_tie() {
        // Same multiset, ids 1 and 2 swapped across a real gap; the cut ties.
        let a = r(&[1, 2, 3, 4], &[1.0, 2.0, 3.0, 3.0]);
        let b = r(&[2, 1, 4, 3], &[1.0, 2.0, 3.0, 3.0]);
        let err = ids_agree_up_to_ties(&a, &b, &e()).unwrap_err();
        assert!(err.contains("across a gap wider than ε"), "{err}");
        // The boundary swap alone is fine.
        let c = r(&[1, 2, 4, 3], &[1.0, 2.0, 3.0, 3.0]);
        assert!(ids_agree_up_to_ties(&a, &c, &e()).is_ok());
    }

    // (e) an id missing from the interior even though the cut ties.
    #[test]
    fn a_missing_interior_id_is_a_violation_even_when_the_cut_ties() {
        let a = r(&[1, 2, 3, 4], &[1.0, 2.0, 3.0, 3.0]);
        let b = r(&[1, 9, 3, 5], &[1.0, 2.0, 3.0, 3.0]);
        let err = ids_agree_up_to_ties(&a, &b, &e()).unwrap_err();
        assert!(err.contains("id sets differ"), "{err}");
        assert!(err.contains("[2]"), "the interior id is named: {err}");
        assert!(err.contains("[9]"), "{err}");
        // With the interior restored, only the boundary differs: excused.
        let c = r(&[1, 2, 3, 5], &[1.0, 2.0, 3.0, 3.0]);
        assert!(ids_agree_up_to_ties(&a, &c, &e()).is_ok());
        // An interior id at the *score* of the cut but not present in the
        // other list at all, where the other list holds an id the first does
        // not at an interior score: still a violation.
        let d = r(&[1, 8, 3, 4], &[1.0, 3.0, 3.0, 3.0]);
        assert!(ids_agree_up_to_ties(&a, &d, &e()).is_err());
    }

    // (f) a 3-wide boundary class, each engine returns a different 2 of 3.
    #[test]
    fn a_three_wide_boundary_class_cut_two_ways_is_excused() {
        let a = r(&[1, 2, 10, 11], &[1.0, 2.0, 5.0, 5.0]);
        let b = r(&[1, 2, 11, 12], &[1.0, 2.0, 5.0, 5.0]);
        let c = r(&[1, 2, 12, 10], &[1.0, 2.0, 5.0, 5.0]);
        for (x, y) in [(&a, &b), (&b, &a), (&a, &c), (&c, &a), (&b, &c), (&c, &b)] {
            assert!(
                ids_agree_up_to_ties(x, y, &e()).is_ok(),
                "{:?} vs {:?}",
                x.ids,
                y.ids
            );
            assert!(compare_ranks(std::slice::from_ref(x), std::slice::from_ref(y), &e()).passed);
        }
        // But a member of the class that has *left* the tie is not excused.
        let d = r(&[1, 2, 11, 12], &[1.0, 2.0, 5.0, 5.0 + 3.0 * EPS]);
        assert!(ids_agree_up_to_ties(&a, &d, &e()).is_err());
    }

    // (f') an id excused at the cut on one side sits in a non-final ε-class
    // on that side only: the anchor-based cuts put a's id 2 (5+0.9ε) in the
    // first class and id 3 (5+1.1ε) in the second, while b holds 3 in its
    // first class. T1 passes (2 and 7 are boundary-tie members the other side
    // cut); T2 must not report segment 0..2 as [1] vs [1,3].
    #[test]
    fn t2_merges_segments_from_the_first_excused_id_so_t1_pass_implies_t2_pass() {
        let a = r(&[1, 2, 3], &[5.0, 5.0 + 0.9 * EPS, 5.0 + 1.1 * EPS]);
        let b = r(&[1, 3, 7], &[5.0, 5.0 + 0.9 * EPS, 5.0 + 1.1 * EPS]);
        assert!(ids_agree_up_to_ties(&a, &b, &e()).is_ok());
        assert!(ids_agree_up_to_ties(&b, &a, &e()).is_ok());
        let t2 = compare_ranks(std::slice::from_ref(&a), std::slice::from_ref(&b), &e());
        assert!(t2.passed, "T1 passed, T2 did not: {}", t2.detail);
        let t2 = compare_ranks(std::slice::from_ref(&b), std::slice::from_ref(&a), &e());
        assert!(t2.passed, "T1 passed, T2 did not (reversed): {}", t2.detail);
        // The merge does not excuse an interior difference before it: id 9
        // for id 1, at the same score, is still a mismatch.
        let c = r(&[9, 3, 7], &[5.0, 5.0 + 0.9 * EPS, 5.0 + 1.1 * EPS]);
        assert!(!compare_ranks(std::slice::from_ref(&a), std::slice::from_ref(&c), &e()).passed);
        // Nor one well before the excused class.
        let d = r(&[1, 2, 5, 6], &[1.0, 2.0, 5.0 + 0.9 * EPS, 5.0 + 1.1 * EPS]);
        let f = r(&[1, 8, 6, 7], &[1.0, 2.0, 5.0 + 0.9 * EPS, 5.0 + 1.1 * EPS]);
        assert!(!compare_ranks(std::slice::from_ref(&d), std::slice::from_ref(&f), &e()).passed);
    }

    #[test]
    fn a_length_mismatch_is_never_excused_by_the_cut() {
        let a = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let b = r(&[1, 2], &[1.0, 2.0]);
        assert!(ids_agree_up_to_ties(&a, &b, &e()).is_err());
        assert!(ids_agree_up_to_ties(&b, &a, &e()).is_err());
        // Even where the extra id ties the shorter list's cut.
        let c = r(&[1, 2, 3], &[1.0, 2.0, 2.0]);
        assert!(ids_agree_up_to_ties(&c, &b, &e()).is_err());
    }

    #[test]
    fn excused_at_cut_needs_both_cuts_to_tie() {
        let a = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let same = r(&[1, 2, 4], &[1.0, 2.0, 3.0]);
        let far = r(&[1, 2, 4], &[1.0, 2.0, 3.0 + 2.0 * EPS]);
        assert!(excused_at_cut(&a, 2, &same, &e()));
        assert!(!excused_at_cut(&a, 2, &far, &e()));
        // An interior position never is, however the cuts stand.
        assert!(!excused_at_cut(&a, 1, &same, &e()));
        // Empty lists have no cut.
        let empty = r(&[], &[]);
        assert!(!excused_at_cut(&a, 2, &empty, &e()));
    }

    /// A random pair of lists that differ only by (i) order within ε-tie
    /// classes and (ii) which members of the boundary class made the cut.
    /// `interior` is the number of positions before the boundary class.
    struct Pair {
        a: Returned,
        b: Returned,
        interior: usize,
        next_id: u32,
    }

    fn gen_pair(seed: u64) -> Pair {
        gen_pair_gap(seed, 3.0)
    }

    /// `gap` is the least distance, in ε, between one class's base score and
    /// the next's.
    fn gen_pair_gap(seed: u64, gap: f64) -> Pair {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let n_classes = rng.random_range(2..=5);
        let mut next_id: u32 = 100;
        let mut base = 1.0f64;
        let mut a_ids = Vec::new();
        let mut a_scores = Vec::new();
        let mut b_ids = Vec::new();
        let mut b_scores = Vec::new();
        let mut interior = 0;
        for c in 0..n_classes {
            let size = rng.random_range(1..=3usize);
            let boundary = c + 1 == n_classes;
            // The boundary class draws from a pool up to two wider than the
            // number of slots left; the two lists cut it independently.
            let pool_size = if boundary {
                size + rng.random_range(0..=2usize)
            } else {
                size
            };
            let pool: Vec<(u32, f64)> = (0..pool_size)
                .map(|_| {
                    let id = next_id;
                    next_id += 1;
                    (id, base + rng.random::<f64>() * EPS / 2.0)
                })
                .collect();
            let pick = |rng: &mut rand_chacha::ChaCha8Rng| -> Vec<(u32, f64)> {
                let mut p = pool.clone();
                // Fisher-Yates, then take `size`: a random subset in random order.
                for i in (1..p.len()).rev() {
                    let j = rng.random_range(0..=i);
                    p.swap(i, j);
                }
                p.truncate(size);
                p
            };
            for (id, s) in pick(&mut rng) {
                a_ids.push(id);
                a_scores.push(s);
            }
            for (id, s) in pick(&mut rng) {
                b_ids.push(id);
                // Each engine's own arithmetic: perturbed by < ε/8.
                b_scores.push(s + (rng.random::<f64>() - 0.5) * EPS / 4.0);
            }
            if !boundary {
                interior += size;
            }
            base += gap * EPS + rng.random::<f64>();
        }
        Pair {
            a: Returned {
                ids: a_ids,
                scores: a_scores,
            },
            b: Returned {
                ids: b_ids,
                scores: b_scores,
            },
            interior,
            next_id,
        }
    }

    // (g) the property, ~1000 seeds each way.
    #[test]
    fn property_tie_permutations_and_boundary_cuts_always_pass() {
        let e = e();
        for seed in 0..1000u64 {
            let p = gen_pair(seed);
            assert!(
                ids_agree_up_to_ties(&p.a, &p.b, &e).is_ok(),
                "seed {seed}: {:?}/{:?} vs {:?}/{:?}: {:?}",
                p.a.ids,
                p.a.scores,
                p.b.ids,
                p.b.scores,
                ids_agree_up_to_ties(&p.a, &p.b, &e)
            );
            assert!(
                ids_agree_up_to_ties(&p.b, &p.a, &e).is_ok(),
                "seed {seed} (reversed)"
            );
            let t2 = compare_ranks(std::slice::from_ref(&p.a), std::slice::from_ref(&p.b), &e);
            assert!(t2.passed, "seed {seed}: T2 {}", t2.detail);
            assert!(
                crate::metamorphic::permutation_invariance(
                    std::slice::from_ref(&p.a),
                    std::slice::from_ref(&p.b),
                    &e
                )
                .held,
                "seed {seed}: permutation invariance"
            );
        }
    }

    // (g') the same property with the classes closer together, down to 1.5ε
    // between class bases (members spread over ε/2, perturbed by ε/8): the
    // anchor-based cuts can then differ between the sides. Wherever T1
    // passes, T2 must.
    #[test]
    fn property_t1_pass_implies_t2_pass_with_class_gaps_down_to_1_5_eps() {
        let e = e();
        for gap in [1.5f64, 2.0, 2.5] {
            let mut t1_passed = 0usize;
            for seed in 0..1000u64 {
                let p = gen_pair_gap(seed, gap);
                let ab = ids_agree_up_to_ties(&p.a, &p.b, &e).is_ok();
                let ba = ids_agree_up_to_ties(&p.b, &p.a, &e).is_ok();
                if !(ab && ba) {
                    continue;
                }
                t1_passed += 1;
                let t2 = compare_ranks(std::slice::from_ref(&p.a), std::slice::from_ref(&p.b), &e);
                assert!(
                    t2.passed,
                    "gap {gap}ε seed {seed}: T1 passed, T2 {}: {:?}/{:?} vs {:?}/{:?}",
                    t2.detail, p.a.ids, p.a.scores, p.b.ids, p.b.scores
                );
                let t2 = compare_ranks(std::slice::from_ref(&p.b), std::slice::from_ref(&p.a), &e);
                assert!(
                    t2.passed,
                    "gap {gap}ε seed {seed} (reversed): T2 {}",
                    t2.detail
                );
            }
            assert!(
                t1_passed > 500,
                "gap {gap}ε: only {t1_passed}/1000 pairs passed T1"
            );
        }
    }

    #[test]
    fn property_one_perturbed_interior_id_always_fails() {
        let e = e();
        for seed in 0..1000u64 {
            let mut p = gen_pair(seed);
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0xdead_beef);
            let victim = rng.random_range(0..p.interior);
            // Replace one non-tied id in `b` by a point neither list holds,
            // at the very same score: only the identity changes.
            p.b.ids[victim] = p.next_id;
            assert!(
                ids_agree_up_to_ties(&p.a, &p.b, &e).is_err(),
                "seed {seed}: pos {victim} of {:?} replaced and T1 still passed",
                p.a.ids
            );
            assert!(
                ids_agree_up_to_ties(&p.b, &p.a, &e).is_err(),
                "seed {seed} (reversed)"
            );
            let t2 = compare_ranks(&[p.a.clone()], &[p.b.clone()], &e);
            assert!(!t2.passed, "seed {seed}: T2 {}", t2.detail);
            assert!(
                !crate::metamorphic::permutation_invariance(&[p.a.clone()], &[p.b.clone()], &e)
                    .held
            );
        }
    }

    #[test]
    fn property_one_interior_id_moved_across_a_gap_always_fails() {
        let e = e();
        for seed in 0..1000u64 {
            let mut p = gen_pair(seed);
            // Move the first id (lowest score, its own class or the head of
            // it) to the cut, shifting the rest up: the multiset is intact
            // but that id crossed at least 3ε.
            let first = p.b.ids.remove(0);
            p.b.ids.push(first);
            assert!(
                ids_agree_up_to_ties(&p.a, &p.b, &e).is_err(),
                "seed {seed}: {:?} rotated to {:?} passed",
                p.a.ids,
                p.b.ids
            );
            assert!(
                !compare_ranks(&[p.a.clone()], &[p.b.clone()], &e).passed,
                "seed {seed}"
            );
        }
    }

    // (h) T2's last class compares minus the boundary members the other side cut.
    #[test]
    fn t2_excuses_the_boundary_tie_and_nothing_else() {
        let e = e();
        let a = vec![r(&[1, 2, 3], &[5.0, 7.0, 9.0])];
        let b = vec![r(&[1, 2, 4], &[5.0, 7.0, 9.0])];
        let t = compare_ranks(&a, &b, &e);
        assert!(t.passed, "{}", t.detail);
        assert!(t.detail.contains("0/"), "{}", t.detail);

        // 2ε apart at the cut: different points.
        let far = vec![r(&[1, 2, 4], &[5.0, 7.0, 9.0 + 2.0 * EPS])];
        let t = compare_ranks(&a, &far, &e);
        assert!(!t.passed, "{}", t.detail);
        assert!(t.detail.contains("1/"), "{}", t.detail);

        // A wider boundary class, cut differently, with the class shuffled.
        let a = vec![r(&[1, 2, 3, 4], &[5.0, 7.0, 7.0, 7.0])];
        let b = vec![r(&[1, 4, 5, 2], &[5.0, 7.0, 7.0, 7.0])];
        assert!(compare_ranks(&a, &b, &e).passed);

        // Two of the three boundary members replaced: the pool behind the
        // cut is five wide, each engine returned three of it. Still excused.
        let c = vec![r(&[1, 9, 3, 5], &[5.0, 7.0, 7.0, 7.0])];
        assert!(compare_ranks(&a, &c, &e).passed);
        // An id *outside* the boundary class swapped for another is a T2
        // mismatch even when the cut is also broken differently.
        let d = vec![r(&[9, 2, 3, 5], &[5.0, 7.0, 7.0, 7.0])];
        let t = compare_ranks(&a, &d, &e);
        assert!(!t.passed, "{}", t.detail);
        assert!(t.detail.contains("1/"), "{}", t.detail);
    }

    // (j) the recorded situation: bit-identical scores, a few queries with the
    // boundary broken differently.
    #[test]
    fn t1_and_t2_pass_on_the_recorded_sift_boundary_ties() {
        let e = e();
        let n = 50usize;
        let k = 10usize;
        // Every query: neighbours 1..=10 at distances 1..=10; a hidden id 11
        // also at distance 10 ties the cut. The oracle lists 1..=10.
        let gt = GroundTruth {
            metric: Metric::Euclid,
            n_base: 1000,
            n_queries: n,
            dim: 128,
            k,
            ids: (0..n).flat_map(|_| 1..=k as u32).collect(),
            scores: (0..n).flat_map(|_| (1..=k).map(|i| i as f64)).collect(),
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let ours: Vec<Returned> = (0..n)
            .map(|_| Returned {
                ids: (1..=k as u32).collect(),
                scores: (1..=k).map(|i| i as f64).collect(),
            })
            .collect();
        let mut theirs = ours.clone();
        for qi in [3usize, 17, 29, 44] {
            theirs[qi].ids[k - 1] = 11; // same score, the other coin toss
        }
        let t1 = compare_exact(&ours, &theirs, &gt, &e, k);
        assert!(t1.passed, "T1: {}", t1.detail);
        assert!(!t1.detail.contains("id sets differ"), "{}", t1.detail);
        let t2 = compare_ranks(&ours, &theirs, &e);
        assert!(t2.passed, "T2: {}", t2.detail);
        assert!(
            t2.detail.contains("0/"),
            "no equivalence class differs: {}",
            t2.detail
        );
        // Kendall τ is unaffected by the boundary swap (τ is over common ids)
        // and RBO barely.
        assert!(t2.detail.contains("τ=1.0000"), "{}", t2.detail);

        // The same four queries with the *ninth* id swapped are a real
        // divergence, and both tiers say so.
        let mut wrong = ours.clone();
        for qi in [3usize, 17, 29, 44] {
            wrong[qi].ids[k - 2] = 11;
        }
        let t1 = compare_exact(&ours, &wrong, &gt, &e, k);
        assert!(!t1.passed, "T1: {}", t1.detail);
        assert!(t1.detail.contains("id sets differ"), "{}", t1.detail);
        assert!(
            t1.detail
                .contains("only in ours: [9], only in theirs: [11]"),
            "{}",
            t1.detail
        );
        assert!(!compare_ranks(&ours, &wrong, &e).passed);
    }
}

#[cfg(test)]
mod kendall_and_t4_tests {
    use super::*;

    fn gt(ids: &[u32], scores: &[f64]) -> GroundTruth {
        GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 1,
            dim: 4,
            k: ids.len(),
            ids: ids.to_vec(),
            scores: scores.to_vec(),
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        }
    }

    fn r(ids: &[u32], scores: &[f64]) -> Vec<Returned> {
        vec![Returned {
            ids: ids.to_vec(),
            scores: scores.to_vec(),
        }]
    }

    fn held() -> crate::metamorphic::PropertyResult {
        crate::metamorphic::quantization_dominance(&[(1.0, 0.9), (2.0, 0.95)])
    }

    // (a) τ with 0 and 1 common ids is no evidence: neither agreement nor
    // disagreement, and excluded from the averages rather than averaged in.
    #[test]
    fn kendall_tau_with_fewer_than_two_common_ids_is_no_evidence() {
        assert_eq!(kendall_tau(&[1, 2, 3], &[7, 8, 9]), None);
        assert_eq!(kendall_tau(&[1, 2, 3], &[7, 2, 9]), None);
        assert_eq!(kendall_tau(&[], &[]), None);
        assert_eq!(kendall_tau(&[1], &[1]), None);
        // T2 at `--limit 1`: every query is excluded, the average is n/a,
        // and the report says how many were excluded.
        let e = Epsilon {
            metric: Metric::Euclid,
            dim: 8,
            value: 1e-6,
            relative: false,
            qdrant_spread: Distribution::default(),
            strawmann_spread: Distribution::default(),
            multiple: 4.0,
        };
        let one = |id: u32| Returned {
            ids: vec![id],
            scores: vec![1.0],
        };
        let t2 = compare_ranks(&[one(1), one(2)], &[one(1), one(2)], &e);
        assert!(t2.passed, "{}", t2.detail);
        assert!(
            t2.detail
                .contains("τ=n/a over 0 queries (2 with <2 common ids excluded)"),
            "{}",
            t2.detail
        );
        // Mixed: the query with evidence sets the average, the other is counted.
        let two = Returned {
            ids: vec![1, 2],
            scores: vec![1.0, 2.0],
        };
        let t2 = compare_ranks(&[two.clone(), one(3)], &[two.clone(), one(3)], &e);
        assert!(
            t2.detail
                .contains("τ=1.0000 over 1 queries (1 with <2 common ids excluded)"),
            "{}",
            t2.detail
        );
    }

    // (b) identical, reversed, and a hand-computed partial overlap.
    #[test]
    fn kendall_tau_hand_computed_values() {
        assert!((kendall_tau(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]).unwrap() - 1.0).abs() < 1e-12);
        assert!((kendall_tau(&[1, 2, 3, 4, 5], &[5, 4, 3, 2, 1]).unwrap() + 1.0).abs() < 1e-12);
        // Common ids 1, 2, 3 at positions (0,1), (1,2), (2,0):
        //   (1,2): concordant; (1,3): discordant; (2,3): discordant → -1/3.
        let t = kendall_tau(&[1, 2, 3, 4, 5], &[3, 1, 2, 9, 8]).unwrap();
        assert!((t + 1.0 / 3.0).abs() < 1e-12, "got {t}");
        // Two common ids in the same order: exactly one concordant pair.
        assert!((kendall_tau(&[1, 2, 3], &[9, 1, 2]).unwrap() - 1.0).abs() < 1e-12);
        // And in the opposite order.
        assert!((kendall_tau(&[1, 2, 3], &[2, 9, 1]).unwrap() + 1.0).abs() < 1e-12);
        // Symmetric.
        assert_eq!(
            kendall_tau(&[1, 2, 3, 4, 5], &[3, 1, 2, 9, 8]),
            kendall_tau(&[3, 1, 2, 9, 8], &[1, 2, 3, 4, 5])
        );
    }

    // (c) a quantizer returning disjoint points fails T4; a faithful one passes.
    #[test]
    fn t4_fails_a_quantizer_returning_entirely_wrong_points() {
        let truth = gt(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let f = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let disjoint = r(&[7, 8, 9], &[1.0, 2.0, 3.0]);
        let t = quantization_fidelity(&disjoint, &f, &truth, "pq", &held());
        assert!(!t.passed, "{}", t.detail);
        assert!(
            t.detail.contains("no evidence of rank agreement"),
            "{}",
            t.detail
        );
        assert!(
            t.detail.contains("(1 with <2 common ids excluded)"),
            "{}",
            t.detail
        );

        // One right point among wrong ones is no better.
        let one = r(&[7, 2, 9], &[1.0, 2.0, 3.0]);
        assert!(!quantization_fidelity(&one, &f, &truth, "pq", &held()).passed);
        // Two wrong-order points among wrong ones: evidence, and it is bad.
        let rev = r(&[3, 8, 1], &[1.0, 2.0, 3.0]);
        let t = quantization_fidelity(&rev, &f, &truth, "pq", &held());
        assert!(!t.passed, "{}", t.detail);
        assert!(t.detail.contains("floor"), "{}", t.detail);

        // Faithful: same points, near scores.
        let q = r(&[1, 2, 3], &[1.05, 2.05, 3.05]);
        let t = quantization_fidelity(&q, &f, &truth, "pq", &held());
        assert!(t.passed, "{}", t.detail);
        // Faithful in a different in-tie order still passes: τ over the ids
        // is 1/3 for one swap of three... so use four with one adjacent swap:
        // 5 concordant of 6 → τ = 2/3 ≥ 0.5.
        let truth4 = gt(&[1, 2, 3, 4], &[1.0, 2.0, 3.0, 4.0]);
        let f4 = r(&[1, 2, 3, 4], &[1.0, 2.0, 3.0, 4.0]);
        let q4 = r(&[1, 3, 2, 4], &[1.0, 3.0, 2.0, 4.0]);
        assert!(quantization_fidelity(&q4, &f4, &truth4, "pq", &held()).passed);
    }

    #[test]
    fn t4_with_no_query_carrying_rank_evidence_fails_and_says_so() {
        // `--limit 1`: every list holds one id, so no query has two ids in
        // common with the oracle and τ is undefined everywhere. That is not
        // a τ of 0 (which failed T4 unconditionally at limit 1) and not a
        // pass by default: the verdict is failure with the "no evidence"
        // reason, and the excluded count is reported.
        let truth = gt(&[1], &[1.0]);
        let f = r(&[1], &[1.0]);
        let q = r(&[1], &[1.01]);
        let t = quantization_fidelity(&q, &f, &truth, "pq", &held());
        assert!(!t.passed, "{}", t.detail);
        assert!(
            t.detail.contains("no evidence of rank agreement"),
            "{}",
            t.detail
        );
        assert!(
            t.detail.contains("(1 with <2 common ids excluded)"),
            "{}",
            t.detail
        );
        assert!(
            !t.detail.contains("floor"),
            "not the floor reason: {}",
            t.detail
        );
        // With evidence in one query and none in another, the one with
        // evidence decides and the other is counted, not averaged in as 0.
        let truth2 = GroundTruth {
            metric: Metric::Euclid,
            n_base: 100,
            n_queries: 2,
            dim: 4,
            k: 2,
            ids: vec![1, 2, 3, 4],
            scores: vec![1.0, 2.0, 1.0, 2.0],
            base_checksum: 0,
            query_checksum: 0,
            condition: None,
            n_matching: None,
        };
        let f2 = vec![
            Returned {
                ids: vec![1, 2],
                scores: vec![1.0, 2.0],
            },
            Returned {
                ids: vec![3, 4],
                scores: vec![1.0, 2.0],
            },
        ];
        let q2 = vec![
            Returned {
                ids: vec![1, 2],
                scores: vec![1.0, 2.0],
            },
            Returned {
                ids: vec![9, 4],
                scores: vec![1.0, 2.0],
            },
        ];
        let t = quantization_fidelity(&q2, &f2, &truth2, "pq", &held());
        assert!(t.passed, "{}", t.detail);
        assert!(
            t.detail
                .contains("= 1.0000 over 1 queries (1 with <2 common ids excluded)"),
            "{}",
            t.detail
        );
    }

    // (d) the fidelity delta over disjoint lists is empty and says how many
    // ids were not common, rather than a spurious distribution.
    #[test]
    fn t4_fidelity_delta_is_over_common_ids_only() {
        let truth = gt(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        let f = r(&[1, 2, 3], &[1.0, 2.0, 3.0]);

        let disjoint = r(&[7, 8, 9], &[1.0, 2.0, 3.0]);
        let t = quantization_fidelity(&disjoint, &f, &truth, "pq", &held());
        let d = t.distribution.clone().unwrap();
        assert_eq!(d.count, 0, "no common id, no delta: {}", t.detail);
        assert_eq!(d.max, 0.0);
        assert!(t.detail.contains("6 ids not common"), "{}", t.detail);

        // Positional comparison would have reported |Δ| = 0 everywhere here
        // (same scores at every position); matched by id there is one common
        // point with a real delta, and four ids not common.
        let partial = r(&[9, 1, 8], &[1.0, 2.5, 3.0]);
        let t = quantization_fidelity(&partial, &f, &truth, "pq", &held());
        let d = t.distribution.clone().unwrap();
        assert_eq!(d.count, 1, "{}", t.detail);
        assert!((d.max - 1.5).abs() < 1e-12, "{}", t.detail);
        assert!(t.detail.contains("4 ids not common"), "{}", t.detail);

        // Same points in a different order: every delta matched by id.
        let reordered = r(&[3, 1, 2], &[3.1, 1.1, 2.1]);
        let t = quantization_fidelity(&reordered, &f, &truth, "pq", &held());
        let d = t.distribution.clone().unwrap();
        assert_eq!(d.count, 3);
        assert!((d.max - 0.1).abs() < 1e-9, "{}", t.detail);
        assert!(t.detail.contains("0 ids not common"), "{}", t.detail);
    }

    #[test]
    fn identical_rows_hash_identically_and_limit_or_epsilon_change_it() {
        let build = || ConformanceRow {
            dataset: "sift1m".into(),
            metric: Metric::Euclid,
            dim: 128,
            qdrant_version: "1.19.0".into(),
            strawmann_commit: "abc".into(),
            isa_build: "avx512".into(),
            base_checksum: 1,
            query_checksum: 2,
            epsilon: 1e-6,
            epsilon_relative: true,
            limit: 10,
            tiers: vec![
                TierResult {
                    advisory: false,
                    tier: Tier::T0Wire,
                    passed: true,
                    detail: "x".into(),
                    distribution: None,
                },
                TierResult {
                    advisory: false,
                    tier: Tier::T1ExactValue,
                    passed: true,
                    detail: "y".into(),
                    distribution: None,
                },
            ],
        };
        let a = build();
        let b = build();
        assert_eq!(a.hash(), b.hash());
        // The detail text is not part of the identity.
        let mut c = build();
        c.tiers[0].detail = "something else".into();
        assert_eq!(a.hash(), c.hash());

        let mut l = build();
        l.limit = 11;
        assert_ne!(a.hash(), l.hash());
        let mut e = build();
        e.epsilon = 2e-6;
        assert_ne!(a.hash(), e.hash());
        let mut rel = build();
        rel.epsilon_relative = false;
        assert_ne!(a.hash(), rel.hash());
        assert_ne!(l.hash(), e.hash());
    }
}
