//! §8.6 — metamorphic properties.
//!
//! > "Invariants that need no oracle at all, cheap enough to run on every
//! > commit, and historically very good at finding real bugs:
//! >
//! > - `score(x, x)` is maximal, and exact top-1 for an indexed vector as query
//! >   is that vector itself (under ANN this becomes a recall canary rather than
//! >   an assertion).
//! > - **Permutation invariance:** insertion order must not change exact-search
//! >   results.
//! > - **Prefix property:** `top_k` is a prefix of `top_(k+1)`.
//! > - **Offset consistency:** `query(limit=L, offset=O)` equals
//! >   `query(limit=L+O)[O..]`.
//! > - **Idempotent upsert:** re-upserting identical points changes nothing
//! >   observable.
//! > - **Filter subsetting:** filtered results ⊆ unfiltered results ∩ points
//! >   matching the filter (phase 3).
//! > - **Quantization dominance:** with `rescore = true` and oversampling `n`,
//! >   recall is monotonically non-decreasing in `n`.
//! >
//! > Each is asserted against strawmann alone, and the interesting ones against
//! > Qdrant too — a property that holds for one engine and not the other is a
//! > finding."
//!
//! The checks here take *results*, not a client, so the same code runs against
//! either engine and the "holds for one but not the other" comparison is a
//! matter of calling them twice.

use crate::differ::tolerance::Epsilon;
use crate::relevance::Returned;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyResult {
    pub name: &'static str,
    pub held: bool,
    pub detail: String,
}

fn ok(name: &'static str) -> PropertyResult {
    PropertyResult {
        name,
        held: true,
        detail: "holds".into(),
    }
}

fn fail(name: &'static str, detail: String) -> PropertyResult {
    PropertyResult {
        name,
        held: false,
        detail,
    }
}

/// A property with nothing to check has not held. Every check here used to
/// return `ok` on empty input, so an engine that answered no queries — or a
/// harness that forgot to ask any — reported six invariants "holding".
fn nothing_to_check(name: &'static str) -> PropertyResult {
    fail(name, "nothing to check: no results were compared".into())
}

/// §8.6: "`score(x, x)` is maximal, and exact top-1 for an indexed vector as
/// query is that vector itself (under ANN this becomes a recall canary rather
/// than an assertion)."
///
/// `exact` selects which reading applies: an assertion for exact search, a rate
/// for ANN. Conflating them would either produce spurious failures on ANN or
/// let a genuine exact-search bug through.
///
/// `same_vector(a, b)` says whether points `a` and `b` hold identical vectors.
/// A duplicate of the query is at the same distance as the query itself, and
/// §8.7's tie-break then returns the lower id — legitimately, and possibly not
/// the expected one. Such a top-1 is a hit; anything else at the top is not.
pub fn self_retrieval(
    results: &[Returned],
    expected_ids: &[u32],
    exact: bool,
    same_vector: &dyn Fn(u32, u32) -> bool,
) -> PropertyResult {
    let n = results.len().min(expected_ids.len());
    if n == 0 {
        return nothing_to_check(if exact {
            "self_retrieval(exact)"
        } else {
            "self_retrieval(ann canary)"
        });
    }
    let mut hits = 0usize;
    let mut first_miss: Option<(usize, u32, Option<u32>)> = None;
    for i in 0..n {
        let got = results[i].ids.first().copied();
        if got == Some(expected_ids[i]) || got.is_some_and(|g| same_vector(g, expected_ids[i])) {
            hits += 1;
        } else if first_miss.is_none() {
            first_miss = Some((i, expected_ids[i], got));
        }
    }
    let rate = hits as f64 / n as f64;

    if exact {
        if hits == n {
            ok("self_retrieval(exact)")
        } else {
            let (i, want, got) = first_miss.unwrap();
            fail(
                "self_retrieval(exact)",
                format!("query {i}: expected top-1 {want}, got {got:?} ({hits}/{n})"),
            )
        }
    } else {
        // A canary, per §8.6. 0.99 is a threshold on a stochastic structure,
        // not a correctness bound.
        PropertyResult {
            name: "self_retrieval(ann canary)",
            held: rate >= 0.99,
            detail: format!("{hits}/{n} = {rate:.4}"),
        }
    }
}

/// §8.6: "**Permutation invariance:** insertion order must not change
/// exact-search results."
///
/// `a` and `b` are the same exact queries against two collections holding the
/// same points inserted in different orders. This is the property that would
/// catch a search whose output depends on arena order rather than on the data
/// — and it is why §8.7 requires a total order on results in the first place.
///
/// Compared up to ε-ties, not positionally: §8.7's total order breaks ties on
/// *internal* id, which is insertion order, so two equidistant points may
/// legitimately swap. The points and their scores must not change; the order
/// within a tie may.
pub fn permutation_invariance(a: &[Returned], b: &[Returned], eps: &Epsilon) -> PropertyResult {
    let n = a.len().min(b.len());
    if n == 0 {
        return nothing_to_check("permutation_invariance");
    }
    for qi in 0..n {
        if a[qi].ids.len() != b[qi].ids.len() {
            return fail(
                "permutation_invariance",
                format!(
                    "query {qi}: insertion order changed the result count ({} vs {})",
                    a[qi].ids.len(),
                    b[qi].ids.len()
                ),
            );
        }
        if let Err(why) = crate::differ::ids_agree_up_to_ties(&a[qi], &b[qi], eps) {
            return fail(
                "permutation_invariance",
                format!("query {qi}: insertion order changed the result list: {why}"),
            );
        }
        for i in 0..a[qi].ids.len() {
            let (x, y) = (a[qi].scores[i], b[qi].scores[i]);
            if !eps.holds((x - y).abs(), x) {
                return fail(
                    "permutation_invariance",
                    format!(
                        "query {qi} pos {i}: insertion order changed the score ({x:.9} vs {y:.9})"
                    ),
                );
            }
        }
    }
    ok("permutation_invariance")
}

/// The order queries are *asked* in must not change exact-search results.
///
/// This is not §8.6's permutation invariance, which is about insertion order
/// and is `permutation_invariance` above; it was labelled as such for a while,
/// which overstated what had been checked. It is still worth asserting — a
/// batch handler that leaked state between queries would fail it — and it is
/// free, so it stays under its own name.
///
/// `a` and `b` are the same queries answered in two different batch orders,
/// already put back into the same order by the caller.
///
/// Compared up to ε-ties, like every other positional property here: on a
/// dataset with duplicate vectors Qdrant's order within a tie is not
/// deterministic between two requests, and an exact `ids ==` recorded that
/// as a finding against a property it does not violate.
pub fn query_order_invariance(a: &[Returned], b: &[Returned], eps: &Epsilon) -> PropertyResult {
    let n = a.len().min(b.len());
    if n == 0 {
        return nothing_to_check("query_order_invariance");
    }
    for qi in 0..n {
        if let Err(why) = crate::differ::ids_agree_up_to_ties(&a[qi], &b[qi], eps) {
            return fail(
                "query_order_invariance",
                format!("query {qi}: batch order changed the result list: {why}"),
            );
        }
    }
    ok("query_order_invariance")
}

/// §8.6: "**Prefix property:** `top_k` is a prefix of `top_(k+1)`."
///
/// A prefix up to ε-ties: the longer list cut at `k` is compared with the
/// shorter one under the same rule T1 applies between the engines, so two
/// equidistant points swapped between the requests, or exchanged across the
/// `k` cut, are not a violation. On SIFT's duplicate vectors Qdrant orders
/// ties differently from one request to the next, and the exact comparison
/// recorded that as a Qdrant finding.
pub fn prefix_property(small: &[Returned], large: &[Returned], eps: &Epsilon) -> PropertyResult {
    let n = small.len().min(large.len());
    if n == 0 {
        return nothing_to_check("prefix_property");
    }
    for qi in 0..n {
        let k = small[qi].ids.len();
        if k > large[qi].ids.len() {
            return fail(
                "prefix_property",
                format!(
                    "query {qi}: smaller limit returned more results ({k} > {})",
                    large[qi].ids.len()
                ),
            );
        }
        let head = Returned {
            ids: large[qi].ids[..k].to_vec(),
            scores: large[qi].scores[..k].to_vec(),
        };
        if let Err(why) = crate::differ::ids_agree_up_to_ties(&small[qi], &head, eps) {
            return fail(
                "prefix_property",
                format!(
                    "query {qi}: top_{k} is not a prefix of the longer list ({:?} vs {:?}): {why}",
                    small[qi].ids,
                    &large[qi].ids[..k]
                ),
            );
        }
    }
    ok("prefix_property")
}

/// §8.6: "**Offset consistency:** `query(limit=L, offset=O)` equals
/// `query(limit=L+O)[O..]`."
///
/// `with_offset` is the answer to `(limit=L, offset=O)`; `without` to
/// `(limit=L+O, offset=0)`. The slice must be equal in *length* as well as
/// content: comparing only the common prefix let an engine that ignored
/// `offset` and returned nothing — or returned the first `L` again, when `L`
/// happened to be a prefix of the tail — pass. It was also called with `O=0`
/// on two lists that were both offset-free, which checked nothing at all.
///
/// The tail and the offset list are compared up to ε-ties (T1's rule), so a
/// tie reordered between the two requests is not a violation. A tie that
/// straddles the *offset* itself — rank `O-1` and rank `O` equidistant, cut
/// differently by the two requests — is not excused: the helper excuses the
/// far cut only. It has not been observed; it would show as an id present in
/// one list only, at position 0.
pub fn offset_consistency(
    with_offset: &[Returned],
    without: &[Returned],
    offset: usize,
    eps: &Epsilon,
) -> PropertyResult {
    let n = with_offset.len().min(without.len());
    if n == 0 {
        return nothing_to_check("offset_consistency");
    }
    for qi in 0..n {
        let a = &with_offset[qi].ids;
        let full = &without[qi].ids;
        if offset >= full.len() {
            if !a.is_empty() {
                return fail(
                    "offset_consistency",
                    format!(
                        "query {qi}: offset {offset} past the end but {} results returned",
                        a.len()
                    ),
                );
            }
            continue;
        }
        let expect = &full[offset..];
        if a.len() != expect.len() {
            return fail(
                "offset_consistency",
                format!(
                    "query {qi}: offset {offset} returned {} results where the unoffset tail has {}",
                    a.len(),
                    expect.len()
                ),
            );
        }
        let tail = Returned {
            ids: expect.to_vec(),
            scores: without[qi].scores[offset..].to_vec(),
        };
        if let Err(why) = crate::differ::ids_agree_up_to_ties(&with_offset[qi], &tail, eps) {
            return fail(
                "offset_consistency",
                format!("query {qi}: offset slice differs ({a:?} vs {expect:?}): {why}"),
            );
        }
    }
    ok("offset_consistency")
}

/// §8.6: "**Idempotent upsert:** re-upserting identical points changes nothing
/// observable."
///
/// "Nothing observable" is judged up to ε-ties (T1's rule): the same points at
/// the same scores, with the order inside a tie free to differ between the two
/// requests, as it does on Qdrant with duplicate vectors.
pub fn idempotent_upsert(
    before: &[Returned],
    after: &[Returned],
    count_before: u64,
    count_after: u64,
    eps: &Epsilon,
) -> PropertyResult {
    if count_before != count_after {
        return fail(
            "idempotent_upsert",
            format!("points_count changed from {count_before} to {count_after}"),
        );
    }
    let n = before.len().min(after.len());
    if n == 0 {
        return nothing_to_check("idempotent_upsert");
    }
    for qi in 0..n {
        if let Err(why) = crate::differ::ids_agree_up_to_ties(&before[qi], &after[qi], eps) {
            return fail(
                "idempotent_upsert",
                format!("query {qi}: results changed after re-upserting identical points: {why}"),
            );
        }
    }
    ok("idempotent_upsert")
}

/// §8.6: "**Quantization dominance:** with `rescore = true` and oversampling
/// `n`, recall is monotonically non-decreasing in `n`."
///
/// Takes `(oversampling, recall)` pairs in ascending oversampling order.
pub fn quantization_dominance(points: &[(f64, f64)]) -> PropertyResult {
    if points.len() < 2 {
        return nothing_to_check("quantization_dominance");
    }
    // An unmeasurable recall (NaN: `relevance::evaluate` cannot produce
    // recall@10 below limit 10) compares false against everything, so a
    // curve of NaN used to "hold". A property that could not be measured
    // has not been shown to hold.
    if let Some((ov, _)) = points.iter().find(|(_, r)| r.is_nan()) {
        return fail(
            "quantization_dominance",
            format!("not measurable: recall is n/a at {ov}x (limit below the recall depth?)"),
        );
    }
    for w in points.windows(2) {
        let (ov_a, r_a) = w[0];
        let (ov_b, r_b) = w[1];
        // A small tolerance because recall is a sample statistic; a genuine
        // violation shows up as a large drop, not a noisy one.
        if r_b + 0.01 < r_a {
            return fail(
                "quantization_dominance",
                format!("recall fell from {r_a:.4} at {ov_a}x to {r_b:.4} at {ov_b}x"),
            );
        }
    }
    ok("quantization_dominance")
}

/// §8.6: "**Filter subsetting:** filtered results ⊆ unfiltered results ∩ points
/// matching the filter (phase 3)."
/// **Not run**: this client never sends a `Filter` (`Engine::query_full`),
/// so neither engine is asked for a filtered query in any current run. Not
/// because filtering is a §1 non-goal -- it is not on that list, and
/// strawmann has served it since 2026-09-03 -- but because the oracle has no
/// per-condition ground truth yet (docs/workloads.md, W12 point 3). The
/// other six §8.6 properties all execute (`main::run_metamorphic` and the
/// T4 block).
#[allow(dead_code)]
pub fn filter_subsetting(
    filtered: &[Returned],
    unfiltered: &[Returned],
    matching: &[u32],
) -> PropertyResult {
    let matching: std::collections::HashSet<u32> = matching.iter().copied().collect();
    let n = filtered.len().min(unfiltered.len());
    if n == 0 {
        return nothing_to_check("filter_subsetting");
    }
    for qi in 0..n {
        let unf: std::collections::HashSet<u32> = unfiltered[qi].ids.iter().copied().collect();
        for &id in &filtered[qi].ids {
            if !matching.contains(&id) {
                return fail(
                    "filter_subsetting",
                    format!("query {qi}: returned {id}, which does not match the filter"),
                );
            }
            // Note: filtered results need not be a subset of an *unfiltered
            // top-k*, since filtering can surface points the unfiltered top-k
            // did not reach. The containment only holds when the unfiltered
            // list is the full ranking, which is what the caller must pass.
            if !unf.is_empty() && !unf.contains(&id) {
                return fail(
                    "filter_subsetting",
                    format!("query {qi}: returned {id}, absent from the unfiltered ranking"),
                );
            }
        }
    }
    ok("filter_subsetting")
}

/// Run every property that applies to a pair of engines and report which held
/// where.
///
/// §8.6: "a property that holds for one engine and not the other is a finding."
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Comparison {
    pub property: &'static str,
    pub strawmann: bool,
    pub qdrant: bool,
}

impl Comparison {
    pub fn is_finding(&self) -> bool {
        self.strawmann != self.qdrant
    }

    pub fn describe(&self) -> String {
        if self.is_finding() {
            format!(
                "FINDING: {} holds for {} but not for {}",
                self.property,
                if self.strawmann {
                    "strawmann"
                } else {
                    "qdrant"
                },
                if self.strawmann {
                    "qdrant"
                } else {
                    "strawmann"
                }
            )
        } else {
            format!(
                "{}: both {}",
                self.property,
                if self.strawmann { "hold" } else { "FAIL" }
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(ids: &[u32]) -> Returned {
        Returned {
            ids: ids.to_vec(),
            scores: ids.iter().map(|&i| f64::from(i)).collect(),
        }
    }

    #[test]
    fn self_retrieval_is_an_assertion_for_exact_and_a_canary_for_ann() {
        let perfect = vec![r(&[0, 5]), r(&[1, 6])];
        assert!(self_retrieval(&perfect, &[0, 1], true, &|_, _| false).held);

        let one_miss = vec![r(&[0, 5]), r(&[9, 6])];
        // Exact: a single miss is a failure.
        assert!(!self_retrieval(&one_miss, &[0, 1], true, &|_, _| false).held);
        // ANN: 50% is below the canary threshold, so still a failure here...
        assert!(!self_retrieval(&one_miss, &[0, 1], false, &|_, _| false).held);

        // ...but at scale a single miss is fine under ANN.
        let mut many: Vec<Returned> = (0..200).map(|i| r(&[i])).collect();
        let ids: Vec<u32> = (0..200).collect();
        many[7] = r(&[999]);
        assert!(!self_retrieval(&many, &ids, true, &|_, _| false).held);
        assert!(self_retrieval(&many, &ids, false, &|_, _| false).held);
    }

    fn eps() -> Epsilon {
        let mut e = crate::differ::tolerance::calibrate(
            crate::oracle::Metric::Euclid,
            8,
            Default::default(),
            Default::default(),
            1.0,
        );
        e.value = 1e-6;
        e.relative = false;
        e
    }

    #[test]
    fn permutation_invariance_catches_order_dependence() {
        let a = vec![r(&[1, 2, 3])];
        assert!(permutation_invariance(&a, &a, &eps()).held);
        let b = vec![r(&[1, 3, 2])];
        let res = permutation_invariance(&a, &b, &eps());
        assert!(!res.held);
        assert!(res.detail.contains("insertion order"));

        // §8.7 breaks ties on internal id, which *is* insertion order, so two
        // equidistant points may swap: not a violation.
        let tied_a = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![5.0, 5.0, 9.0],
        }];
        let tied_b = vec![Returned {
            ids: vec![2, 1, 3],
            scores: vec![5.0, 5.0, 9.0],
        }];
        assert!(permutation_invariance(&tied_a, &tied_b, &eps()).held);
        // But a changed score is.
        let drifted = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![5.0, 5.0, 9.1],
        }];
        assert!(!permutation_invariance(&tied_a, &drifted, &eps()).held);

        // The batch-order check is a different, weaker property under its own
        // name.
        assert!(query_order_invariance(&a, &a, &eps()).held);
        assert!(!query_order_invariance(&a, &b, &eps()).held);
    }

    #[test]
    fn permutation_invariance_excuses_a_boundary_tie_but_not_an_interior_difference() {
        // §8.7 breaks ties on internal id, which is insertion order: the two
        // collections hold the same points inserted in different orders, so
        // whichever of two points tied at the k-th score makes the cut may
        // change. That is not an order dependence.
        let e = eps();
        let a = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let b = vec![Returned {
            ids: vec![1, 2, 4],
            scores: vec![1.0, 2.0, 3.0],
        }];
        assert!(permutation_invariance(&a, &b, &e).held);
        assert!(permutation_invariance(&b, &a, &e).held);
        // Within ε/2 at the cut, still.
        let near = vec![Returned {
            ids: vec![1, 2, 4],
            scores: vec![1.0, 2.0, 3.0 + 0.5e-6],
        }];
        assert!(permutation_invariance(&a, &near, &e).held);
        // A three-wide boundary class, cut two ways and shuffled.
        let ta = vec![Returned {
            ids: vec![1, 10, 11],
            scores: vec![1.0, 5.0, 5.0],
        }];
        let tb = vec![Returned {
            ids: vec![1, 12, 10],
            scores: vec![1.0, 5.0, 5.0],
        }];
        assert!(permutation_invariance(&ta, &tb, &e).held);

        // But a *different* point at the cut, 2ε away, is a real change...
        let far = vec![Returned {
            ids: vec![1, 2, 4],
            scores: vec![1.0, 2.0, 3.0 + 2e-6],
        }];
        let res = permutation_invariance(&a, &far, &e);
        assert!(!res.held);
        assert!(
            res.detail
                .contains("insertion order changed the result list"),
            "{}",
            res.detail
        );
        // ...as is an interior id that changed while the cut ties.
        let interior = vec![Returned {
            ids: vec![1, 9, 4],
            scores: vec![1.0, 2.0, 3.0],
        }];
        let res = permutation_invariance(&a, &interior, &e);
        assert!(!res.held);
        assert!(res.detail.contains("id sets differ"), "{}", res.detail);
        // ...as is an id moved across a gap with the multiset intact.
        let moved = vec![Returned {
            ids: vec![2, 1, 3],
            scores: vec![1.0, 2.0, 3.0],
        }];
        assert!(!permutation_invariance(&a, &moved, &e).held);
        // The batch-order property is judged under the same rule.
        assert!(query_order_invariance(&a, &b, &e).held);
        assert!(!query_order_invariance(&a, &interior, &e).held);
    }

    #[test]
    fn a_property_with_nothing_to_check_has_not_held() {
        // Every one of these returned `ok` on empty input.
        let e = eps();
        assert!(!self_retrieval(&[], &[], true, &|_, _| false).held);
        assert!(!self_retrieval(&[], &[], false, &|_, _| false).held);
        assert!(!permutation_invariance(&[], &[], &e).held);
        assert!(!query_order_invariance(&[], &[], &e).held);
        assert!(!prefix_property(&[], &[], &e).held);
        assert!(!offset_consistency(&[], &[], 3, &e).held);
        assert!(!idempotent_upsert(&[], &[], 1, 1, &e).held);
        assert!(!quantization_dominance(&[]).held);
        assert!(!quantization_dominance(&[(1.0, 0.5)]).held);
    }

    #[test]
    fn prefix_property_holds_and_is_violated_detectably() {
        let e = eps();
        let small = vec![r(&[1, 2])];
        let large = vec![r(&[1, 2, 3])];
        assert!(prefix_property(&small, &large, &e).held);

        let broken = vec![r(&[9, 2, 3])];
        assert!(!prefix_property(&small, &broken, &e).held);
    }

    /// The four positional properties are judged up to ε-ties, like T1: a
    /// tied pair reordered between two requests — Qdrant does this on SIFT's
    /// duplicate vectors — is not a violation, while a real change still is.
    #[test]
    fn positional_properties_excuse_a_reordered_tie_but_not_a_real_change() {
        let e = eps();
        let a = vec![Returned {
            ids: vec![1, 2, 3],
            scores: vec![1.0, 2.0, 2.0],
        }];
        let tied = vec![Returned {
            ids: vec![1, 3, 2],
            scores: vec![1.0, 2.0, 2.0],
        }];
        let changed = vec![Returned {
            ids: vec![9, 2, 3],
            scores: vec![1.0, 2.0, 2.0],
        }];

        assert!(query_order_invariance(&a, &tied, &e).held);
        assert!(!query_order_invariance(&a, &changed, &e).held);
        assert!(idempotent_upsert(&a, &tied, 10, 10, &e).held);
        assert!(!idempotent_upsert(&a, &changed, 10, 10, &e).held);

        // top_3 against a top_5 whose tie is ordered the other way.
        let large = vec![Returned {
            ids: vec![1, 3, 2, 4, 5],
            scores: vec![1.0, 2.0, 2.0, 4.0, 5.0],
        }];
        assert!(prefix_property(&a, &large, &e).held);
        assert!(!prefix_property(&changed, &large, &e).held);
        // And a tie exchanged across the cut: top_2 = [1, 3] of that top_5.
        let two = vec![Returned {
            ids: vec![1, 2],
            scores: vec![1.0, 2.0],
        }];
        assert!(prefix_property(&two, &large, &e).held);

        // offset 1 of the top_5, with the tie reordered.
        let off = vec![Returned {
            ids: vec![2, 3, 4, 5],
            scores: vec![2.0, 2.0, 4.0, 5.0],
        }];
        assert!(offset_consistency(&off, &large, 1, &e).held);
        let off_changed = vec![Returned {
            ids: vec![2, 3, 9, 5],
            scores: vec![2.0, 2.0, 4.0, 5.0],
        }];
        assert!(!offset_consistency(&off_changed, &large, 1, &e).held);
    }

    /// §8.7 breaks ties on the lower id, so a duplicate of the query that
    /// carries a lower id is the legitimate exact top-1. Only a duplicate,
    /// though.
    #[test]
    fn self_retrieval_accepts_a_duplicate_of_the_query_at_the_top() {
        let dup = |a: u32, b: u32| (a, b) == (3, 7) || (a, b) == (7, 3);
        let results = vec![r(&[3]), r(&[8])];
        assert!(self_retrieval(&results, &[7, 8], true, &dup).held);
        assert!(!self_retrieval(&results, &[7, 9], true, &dup).held);
        assert!(!self_retrieval(&results, &[7, 8], true, &|_, _| false).held);
    }

    #[test]
    fn offset_consistency_compares_the_right_slice() {
        let e = eps();
        let full = vec![r(&[1, 2, 3, 4, 5])];
        let offset2 = vec![r(&[3, 4, 5])];
        assert!(offset_consistency(&offset2, &full, 2, &e).held);

        let wrong = vec![r(&[4, 5])];
        assert!(!offset_consistency(&wrong, &full, 2, &e).held);

        // An engine that ignores the offset and returns nothing, or a prefix
        // of the tail, is not consistent — the tail is 3 long.
        let nothing = vec![r(&[])];
        assert!(!offset_consistency(&nothing, &full, 2, &e).held);
        let truncated = vec![r(&[3, 4])];
        assert!(!offset_consistency(&truncated, &full, 2, &e).held);

        // An offset past the end must return nothing.
        let empty = vec![r(&[])];
        assert!(offset_consistency(&empty, &full, 99, &e).held);
        assert!(!offset_consistency(&offset2, &full, 99, &e).held);
    }

    #[test]
    fn idempotent_upsert_watches_both_results_and_the_count() {
        let e = eps();
        let a = vec![r(&[1, 2])];
        assert!(idempotent_upsert(&a, &a, 100, 100, &e).held);
        // §2's `--max-id` case: re-upserting must update in place, not append.
        let res = idempotent_upsert(&a, &a, 100, 200, &e);
        assert!(!res.held);
        assert!(res.detail.contains("points_count"));
    }

    /// `NaN + 0.01 < NaN` is false, so a curve of unmeasurable recalls —
    /// which `relevance::evaluate` yields for recall@10 below limit 10 —
    /// used to "hold".
    #[test]
    fn quantization_dominance_is_not_measurable_on_nan_recall() {
        let res = quantization_dominance(&[(1.0, f64::NAN), (4.0, f64::NAN)]);
        assert!(!res.held);
        assert!(res.detail.contains("not measurable"), "{}", res.detail);
        assert!(!quantization_dominance(&[(1.0, 0.5), (4.0, f64::NAN)]).held);
    }

    #[test]
    fn quantization_dominance_accepts_noise_but_not_a_real_drop() {
        assert!(quantization_dominance(&[(1.0, 0.50), (4.0, 0.70), (16.0, 0.90)]).held);
        // Sampling noise.
        assert!(quantization_dominance(&[(1.0, 0.500), (4.0, 0.495)]).held);
        // A real violation.
        let res = quantization_dominance(&[(1.0, 0.90), (4.0, 0.50)]);
        assert!(!res.held);
        assert!(res.detail.contains("fell"));
    }

    #[test]
    fn filter_subsetting_rejects_a_non_matching_result() {
        let filtered = vec![r(&[1, 3])];
        let unfiltered = vec![r(&[1, 2, 3, 4])];
        assert!(filter_subsetting(&filtered, &unfiltered, &[1, 3, 5]).held);

        let bad = vec![r(&[1, 2])];
        let res = filter_subsetting(&bad, &unfiltered, &[1, 3, 5]);
        assert!(!res.held);
        assert!(res.detail.contains("does not match the filter"));
    }

    #[test]
    fn a_property_holding_for_one_engine_only_is_reported_as_a_finding() {
        // §8.6's closing sentence, made mechanical.
        let same = Comparison {
            property: "prefix_property",
            strawmann: true,
            qdrant: true,
        };
        assert!(!same.is_finding());

        let differ = Comparison {
            property: "prefix_property",
            strawmann: true,
            qdrant: false,
        };
        assert!(differ.is_finding());
        assert!(differ.describe().contains("FINDING"));
        assert!(differ.describe().contains("strawmann"));
    }
}
