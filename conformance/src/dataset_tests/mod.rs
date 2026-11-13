//! Findings 55: a vector-db-benchmark `tests.jsonl` read as a filtered
//! workload, with the neighbours it ships as that workload's truth.
//!
//! `npy_convert` leaves `conditions` and `closest_ids` alone on purpose: the
//! filtered entries' ids are neighbours *under that query's condition*, not
//! §4.3's unfiltered truth. This module is the reader for exactly those
//! entries. It turns each record's `conditions` into the Qdrant filter the
//! query is sent with, turns its `closest_ids` / `closest_scores` into a
//! `GroundTruth` that `relevance::evaluate` scores with the §4.3 tie rule, and
//! checks the shipped ids against our own filtered fp64 oracle
//! (`dataset-truth-check`), as `ground-truth.md` §6 did for the unconditional
//! sets.
//!
//! The grammar is vector-db-benchmark's (`engine/base_client/parser.py`):
//! `{"and": [...], "or": [...]}`, each entry `{field: {type: criteria}}`,
//! `and` read as `must` and `or` as `should`. What this reader cannot send the
//! way the dataset means it is refused, naming the query, rather than
//! approximated: a recall scored against a condition the engine was never
//! asked is the W12 defect over again.

use anyhow::Context;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{Condition, Filter, Range};
use rayon::prelude::*;
use serde_json::Value;
use std::io::BufRead;
use std::path::Path;

use crate::differ::tolerance::Epsilon;
use crate::oracle::{self, GroundTruth, Metric, Neighbour};

/// The `condition` a dataset-tests truth and sweep carry. Each query has its
/// own condition, so no single one names the sweep; this names where they
/// came from, and `RecallSweep::dataset_tests` names the file.
pub const DATASET_CONDITION: &str = "dataset-tests";

/// One `match` value, typed the way Qdrant's `MatchValue` is.
#[derive(Clone, Debug, PartialEq)]
pub enum Exact {
    Keyword(String),
    Integer(i64),
    Boolean(bool),
}

/// A `match_any` list: Qdrant has keyword and integer lists, nothing mixed.
#[derive(Clone, Debug, PartialEq)]
pub enum AnyOf {
    Keywords(Vec<String>),
    Integers(Vec<i64>),
}

/// One field condition of the grammar.
#[derive(Clone, Debug, PartialEq)]
pub enum Cond {
    Match {
        key: String,
        value: Exact,
    },
    MatchAny {
        key: String,
        any: AnyOf,
    },
    Range {
        key: String,
        lt: Option<f64>,
        gt: Option<f64>,
        lte: Option<f64>,
        gte: Option<f64>,
    },
}

/// A record's `conditions`: `and` as `must`, `or` as `should`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Conditions {
    pub must: Vec<Cond>,
    pub should: Vec<Cond>,
}

/// Parse a record's `conditions`. `Ok(None)` is an unconditioned query: the
/// field absent, `null`, `{}`, or clauses that hold no entries.
pub fn parse_conditions(v: Option<&Value>) -> Result<Option<Conditions>, String> {
    let obj = match v {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(o)) => o,
        Some(other) => return Err(format!("conditions is not an object: {other}")),
    };
    let mut c = Conditions::default();
    for (clause, entries) in obj {
        let out = match clause.as_str() {
            "and" => &mut c.must,
            "or" => &mut c.should,
            other => {
                return Err(format!(
                    "clause `{other}` is not part of the grammar (only `and` and `or`)"
                ));
            }
        };
        let entries = entries
            .as_array()
            .ok_or_else(|| format!("`{clause}` is not an array"))?;
        for entry in entries {
            let entry = entry
                .as_object()
                .ok_or_else(|| format!("an `{clause}` entry is not an object: {entry}"))?;
            for (field, spec) in entry {
                let spec = spec
                    .as_object()
                    .ok_or_else(|| format!("`{field}`: condition is not an object: {spec}"))?;
                for (kind, criteria) in spec {
                    out.push(parse_one(field, kind, criteria)?);
                }
            }
        }
    }
    if c.must.is_empty() && c.should.is_empty() {
        return Ok(None);
    }
    Ok(Some(c))
}

fn parse_one(field: &str, kind: &str, criteria: &Value) -> Result<Cond, String> {
    let key = field.to_string();
    match kind {
        "match" => {
            let v = criteria
                .get("value")
                .ok_or_else(|| format!("`{field}`: match has no `value`: {criteria}"))?;
            let value = match v {
                Value::String(s) => Exact::Keyword(s.clone()),
                Value::Bool(b) => Exact::Boolean(*b),
                Value::Number(n) if n.is_i64() || n.is_u64() => Exact::Integer(
                    n.as_i64()
                        .ok_or_else(|| format!("`{field}`: match value {n} does not fit an i64"))?,
                ),
                other @ (Value::Null | Value::Number(_) | Value::Array(_) | Value::Object(_)) => {
                    return Err(format!(
                        "`{field}`: match on {other} is refused: Qdrant matches keywords, \
                         integers and booleans, and has no exact float match"
                    ));
                }
            };
            Ok(Cond::Match { key, value })
        }
        "match_any" => {
            let any = criteria
                .get("any")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("`{field}`: match_any has no `any` array: {criteria}"))?;
            let any = if any.iter().all(Value::is_string) {
                AnyOf::Keywords(
                    any.iter()
                        .map(|v| v.as_str().expect("checked").to_string())
                        .collect(),
                )
            } else if any.iter().all(Value::is_i64) {
                AnyOf::Integers(any.iter().map(|v| v.as_i64().expect("checked")).collect())
            } else {
                return Err(format!(
                    "`{field}`: match_any over {criteria} is refused: Qdrant takes a list \
                     of keywords or a list of integers, not a mix"
                ));
            };
            Ok(Cond::MatchAny { key, any })
        }
        "range" => {
            let bounds = criteria
                .as_object()
                .ok_or_else(|| format!("`{field}`: range is not an object: {criteria}"))?;
            let (mut lt, mut gt, mut lte, mut gte) = (None, None, None, None);
            for (b, v) in bounds {
                let slot = match b.as_str() {
                    "lt" => &mut lt,
                    "gt" => &mut gt,
                    "lte" => &mut lte,
                    "gte" => &mut gte,
                    other => return Err(format!("`{field}`: range bound `{other}` is unknown")),
                };
                *slot = match v {
                    Value::Null => None,
                    Value::Number(n) => n.as_f64(),
                    // parser.py reads a string bound as a datetime, which is a
                    // `datetime_range` in Qdrant, not a numeric `range`.
                    Value::String(s) => {
                        return Err(format!(
                            "`{field}`: range bound `{b}` is the string {s:?}, a datetime \
                             range, which this reader refuses"
                        ));
                    }
                    other @ (Value::Bool(_) | Value::Array(_) | Value::Object(_)) => {
                        return Err(format!("`{field}`: range bound `{b}` is {other}"));
                    }
                };
            }
            Ok(Cond::Range {
                key,
                lt,
                gt,
                lte,
                gte,
            })
        }
        "match_text" => Err(format!(
            "`{field}`: match_text is refused: a full-text match is a tokenizer \
             question, and neither the oracle nor this sweep models one"
        )),
        "geo" => Err(format!(
            "`{field}`: geo conditions are refused: no benchmark payload carries one \
             and the oracle has no geo semantics"
        )),
        other => Err(format!("`{field}`: condition type `{other}` is unknown")),
    }
}

impl Cond {
    /// The Qdrant condition. Keywords go out as `Keyword` explicitly:
    /// qdrant-client's `From<String> for MatchValue` turns any value holding
    /// whitespace into a full-text match, the defect findings 55 records in
    /// bfb ("Light Orange", 4,255 of h-and-m's 10,000 queries).
    fn to_condition(&self) -> Condition {
        match self {
            Cond::Match { key, value } => Condition::matches(
                key.clone(),
                match value {
                    Exact::Keyword(s) => MatchValue::Keyword(s.clone()),
                    Exact::Integer(i) => MatchValue::Integer(*i),
                    Exact::Boolean(b) => MatchValue::Boolean(*b),
                },
            ),
            Cond::MatchAny { key, any } => Condition::matches(
                key.clone(),
                match any {
                    AnyOf::Keywords(v) => MatchValue::from(v.clone()),
                    AnyOf::Integers(v) => MatchValue::from(v.clone()),
                },
            ),
            Cond::Range {
                key,
                lt,
                gt,
                lte,
                gte,
            } => Condition::range(
                key.clone(),
                Range {
                    lt: *lt,
                    gt: *gt,
                    lte: *lte,
                    gte: *gte,
                },
            ),
        }
    }

    /// Whether a payload satisfies this condition, as Qdrant decides it: an
    /// array value matches when any element does, a missing key never does.
    fn holds(&self, payload: &Value) -> bool {
        let key = match self {
            Cond::Match { key, .. } | Cond::MatchAny { key, .. } | Cond::Range { key, .. } => key,
        };
        let Some(v) = payload.get(key) else {
            return false;
        };
        match v {
            Value::Array(items) => items.iter().any(|x| self.holds_scalar(x)),
            Value::Null
            | Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Object(_) => self.holds_scalar(v),
        }
    }

    fn holds_scalar(&self, x: &Value) -> bool {
        match self {
            Cond::Match { value, .. } => match value {
                Exact::Keyword(s) => x.as_str() == Some(s.as_str()),
                Exact::Integer(i) => x.as_i64() == Some(*i),
                Exact::Boolean(b) => x.as_bool() == Some(*b),
            },
            Cond::MatchAny { any, .. } => match any {
                AnyOf::Keywords(v) => x.as_str().is_some_and(|s| v.iter().any(|k| k == s)),
                AnyOf::Integers(v) => x.as_i64().is_some_and(|i| v.contains(&i)),
            },
            Cond::Range {
                lt, gt, lte, gte, ..
            } => {
                let Some(f) = x.as_f64() else {
                    return false;
                };
                lt.is_none_or(|b| f < b)
                    && gt.is_none_or(|b| f > b)
                    && lte.is_none_or(|b| f <= b)
                    && gte.is_none_or(|b| f >= b)
            }
        }
    }
}

impl Conditions {
    pub fn to_filter(&self) -> Filter {
        Filter {
            must: self.must.iter().map(Cond::to_condition).collect(),
            should: self.should.iter().map(Cond::to_condition).collect(),
            ..Default::default()
        }
    }

    /// Qdrant's reading: every `must`, and at least one `should` when there
    /// are any.
    pub fn matches(&self, payload: &Value) -> bool {
        self.must.iter().all(|c| c.holds(payload))
            && (self.should.is_empty() || self.should.iter().any(|c| c.holds(payload)))
    }
}

/// One line of `tests.jsonl`.
#[derive(Clone, Debug)]
pub struct TestRecord {
    /// 1-based, for messages that name the query.
    pub line: usize,
    pub query: Vec<f32>,
    pub conditions: Option<Conditions>,
    pub closest_ids: Vec<u32>,
    pub closest_scores: Vec<f64>,
}

/// Read the first `limit` records (all when `None`).
///
/// Every refusal names the line and the query index, so a dataset that grows
/// a condition this reader cannot send stops the run at the query that
/// carries it.
pub fn read_tests(path: &Path, limit: Option<usize>) -> anyhow::Result<Vec<TestRecord>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("{}:{} (query {})", path.display(), i + 1, out.len());
        let v: Value =
            serde_json::from_str(&line).with_context(|| format!("{}: not JSON", at()))?;
        let query: Vec<f32> = v
            .get("query")
            .and_then(Value::as_array)
            .with_context(|| format!("{}: no `query` array", at()))?
            .iter()
            .map(|x| x.as_f64().map(|f| f as f32))
            .collect::<Option<_>>()
            .with_context(|| format!("{}: `query` holds a non-number", at()))?;
        let conditions =
            parse_conditions(v.get("conditions")).map_err(|e| anyhow::anyhow!("{}: {e}", at()))?;
        let closest_ids: Vec<u32> = v
            .get("closest_ids")
            .and_then(Value::as_array)
            .with_context(|| format!("{}: no `closest_ids` array", at()))?
            .iter()
            .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()))
            .collect::<Option<_>>()
            .with_context(|| format!("{}: `closest_ids` holds a non-id", at()))?;
        let closest_scores: Vec<f64> = v
            .get("closest_scores")
            .and_then(Value::as_array)
            .with_context(|| format!("{}: no `closest_scores` array", at()))?
            .iter()
            .map(Value::as_f64)
            .collect::<Option<_>>()
            .with_context(|| format!("{}: `closest_scores` holds a non-number", at()))?;
        anyhow::ensure!(
            closest_ids.len() == closest_scores.len(),
            "{}: {} closest_ids and {} closest_scores",
            at(),
            closest_ids.len(),
            closest_scores.len()
        );
        out.push(TestRecord {
            line: i + 1,
            query,
            conditions,
            closest_ids,
            closest_scores,
        });
    }
    anyhow::ensure!(!out.is_empty(), "{}: no test records", path.display());
    let dim = out[0].query.len();
    if let Some(r) = out.iter().find(|r| r.query.len() != dim) {
        anyhow::bail!(
            "{}:{}: query is {}-dimensional and the first is {dim}",
            path.display(),
            r.line,
            r.query.len()
        );
    }
    Ok(out)
}

/// The shipped neighbours as a `GroundTruth` at depth `k`.
///
/// `k` is the sweep's `--limit`. A record with a condition may hold fewer
/// than `k` ids: its condition matches fewer points (h-and-m: 676 of 10,000
/// tests ship 1 to 9). Its row is padded with `NaN` scores and an id no base
/// row has, and scoring reads it at its own depth (`GroundTruth::depth`).
/// An unconditioned record shorter than `k` cannot say what its k-th
/// neighbour is and is refused. An id past the corpus is refused too: §4.3
/// makes ids row indices.
pub fn truth_from_tests(
    records: &[TestRecord],
    k: usize,
    metric: Metric,
    n_base: usize,
    base_checksum: u64,
) -> Result<GroundTruth, String> {
    if k == 0 {
        return Err("--limit 0 has no k-th neighbour to score against".into());
    }
    let dim = records.first().map_or(0, |r| r.query.len());
    let mut ids = Vec::with_capacity(records.len() * k);
    let mut scores = Vec::with_capacity(records.len() * k);
    let mut queries = Vec::with_capacity(records.len() * dim);
    for (qi, r) in records.iter().enumerate() {
        if r.closest_ids.len() < k && r.conditions.is_none() {
            return Err(format!(
                "query {qi} (line {}) ships {} closest_ids and --limit is {k}: \
                 the truth cannot be deeper than the dataset's",
                r.line,
                r.closest_ids.len()
            ));
        }
        let have = r.closest_ids.len().min(k);
        if let Some(&bad) = r.closest_ids[..have]
            .iter()
            .find(|&&id| id as usize >= n_base)
        {
            return Err(format!(
                "query {qi} (line {}) names neighbour {bad}, past the corpus of {n_base} (§4.3)",
                r.line
            ));
        }
        ids.extend_from_slice(&r.closest_ids[..have]);
        scores.extend_from_slice(&r.closest_scores[..have]);
        ids.extend(std::iter::repeat_n(u32::MAX, k - have));
        scores.extend(std::iter::repeat_n(f64::NAN, k - have));
        queries.extend_from_slice(&r.query);
    }
    Ok(GroundTruth {
        metric,
        n_base,
        n_queries: records.len(),
        dim,
        k,
        ids,
        scores,
        base_checksum,
        query_checksum: oracle::checksum_f32(&queries),
        condition: Some(DATASET_CONDITION.to_string()),
        // Per query, and the file does not say: each condition matches its
        // own set. A row shorter than `k` carries its depth in its NaN pad.
        n_matching: None,
    })
}

/// Refuse shipped scores that are not this metric's scores.
///
/// The tie clause compares the oracle's fp64 score of a returned point with
/// the shipped k-th score, so the two must be the same quantity. A file of
/// distances judged as similarities would make every tie a miss, or worse,
/// every near miss a tie. The bound is three orders above the fp32 kernel
/// noise `docs/tolerance.md` measures (2.4e-7 at cosine 512 and 2048) and
/// far below any change of convention, such as `1 - cos`.
pub fn check_shipped_scores(
    records: &[TestRecord],
    k: usize,
    rescore: impl Fn(usize, u32) -> Option<f64>,
) -> Result<(), String> {
    for (qi, r) in records.iter().enumerate() {
        for (&id, &shipped) in r.closest_ids.iter().zip(&r.closest_scores).take(k) {
            let Some(ours) = rescore(qi, id) else {
                return Err(format!(
                    "query {qi} (line {}): id {id} is not a base row",
                    r.line
                ));
            };
            if (ours - shipped).abs() > 1e-3 * shipped.abs().max(1.0) {
                return Err(format!(
                    "query {qi} (line {}): id {id} ships score {shipped} and the oracle \
                     scores it {ours}; the file's scores are not this metric's",
                    r.line
                ));
            }
        }
    }
    Ok(())
}

/// One query's verdict in `dataset-truth-check`.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// The shipped id set is ours.
    Exact,
    /// Different ids, every difference a tie with our k-th within ε (§4.3).
    EqualUnderTies,
    Mismatch,
}

#[derive(Clone, Debug)]
pub struct QueryCheck {
    pub query: usize,
    pub line: usize,
    pub verdict: Verdict,
    /// Our k-th score, what the tie clause is read at.
    pub kth: f64,
    /// Shipped ids we do not hold and that do not tie: `(id, our score)`.
    pub extra: Vec<(u32, f64)>,
    /// Shipped ids that fail the query's own condition.
    pub violating: Vec<u32>,
    /// Ours that the shipped set lacks: `(id, our score)`.
    pub missing: Vec<(u32, f64)>,
    /// Largest `|shipped score - our fp64 score|` over the shipped ids.
    pub score_delta: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TruthCheck {
    pub checked: usize,
    pub conditioned: usize,
    pub exact: usize,
    pub equal_under_ties: usize,
    pub mismatched: usize,
    pub max_score_delta: f64,
    /// Every non-matching query, in query order.
    pub mismatches: Vec<QueryCheck>,
}

/// Check each record's shipped ids against our filtered fp64 top-k.
///
/// `base_pre` is the corpus after `oracle::preprocess`; `payloads[i]` is row
/// `i`'s payload. The depth is each record's own `closest_ids` length.
pub fn check_truth(
    metric: Metric,
    base_pre: &[f32],
    n_base: usize,
    dim: usize,
    payloads: &[Value],
    records: &[TestRecord],
    eps: &Epsilon,
) -> TruthCheck {
    let per: Vec<QueryCheck> = records
        .par_iter()
        .enumerate()
        .map_init(Vec::new, |scratch: &mut Vec<Neighbour>, (qi, r)| {
            let q = oracle::preprocess(metric, &r.query);
            let k = r.closest_ids.len();
            let ours = match &r.conditions {
                None => oracle::top_k_into(metric, base_pre, n_base, dim, &q, k, scratch),
                Some(c) => {
                    let ids: Vec<u32> = (0..n_base as u32)
                        .filter(|&i| payloads.get(i as usize).is_some_and(|p| c.matches(p)))
                        .collect();
                    oracle::top_k_of_into(metric, base_pre, n_base, dim, &q, &ids, k, scratch)
                }
            };
            judge(
                metric, base_pre, n_base, dim, payloads, r, qi, &q, &ours, eps,
            )
        })
        .collect();
    let mut out = TruthCheck {
        checked: records.len(),
        conditioned: records.iter().filter(|r| r.conditions.is_some()).count(),
        ..Default::default()
    };
    for c in per {
        out.max_score_delta = out.max_score_delta.max(c.score_delta);
        match c.verdict {
            Verdict::Exact => out.exact += 1,
            Verdict::EqualUnderTies => out.equal_under_ties += 1,
            Verdict::Mismatch => {
                out.mismatched += 1;
                out.mismatches.push(c);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn judge(
    metric: Metric,
    base_pre: &[f32],
    n_base: usize,
    dim: usize,
    payloads: &[Value],
    r: &TestRecord,
    qi: usize,
    q: &[f32],
    ours: &[Neighbour],
    eps: &Epsilon,
) -> QueryCheck {
    let score_of = |id: u32| -> Option<f64> {
        let i = id as usize;
        (i < n_base).then(|| oracle::score(metric, q, &base_pre[i * dim..(i + 1) * dim]))
    };
    let kth = ours.last().map_or(f64::NAN, |n| n.score);
    let tol = eps.absolute_at(kth);
    let our_ids: std::collections::HashSet<u32> = ours.iter().map(|n| n.id).collect();
    let shipped: std::collections::HashSet<u32> = r.closest_ids.iter().copied().collect();
    let mut extra = Vec::new();
    let mut violating = Vec::new();
    let mut score_delta = 0.0f64;
    let mut ties = 0usize;
    for (j, &id) in r.closest_ids.iter().enumerate() {
        let Some(s) = score_of(id) else {
            extra.push((id, f64::NAN));
            continue;
        };
        score_delta = score_delta.max((s - r.closest_scores[j]).abs());
        if let Some(c) = &r.conditions
            && !payloads.get(id as usize).is_some_and(|p| c.matches(p))
        {
            violating.push(id);
            continue;
        }
        if our_ids.contains(&id) {
            continue;
        }
        if kth.is_finite() && (s - kth).abs() <= tol {
            ties += 1;
        } else {
            extra.push((id, s));
        }
    }
    let missing: Vec<(u32, f64)> = ours
        .iter()
        .filter(|n| !shipped.contains(&n.id))
        .map(|n| (n.id, n.score))
        .collect();
    let verdict = if !extra.is_empty() || !violating.is_empty() || shipped.len() != our_ids.len() {
        Verdict::Mismatch
    } else if ties == 0 {
        Verdict::Exact
    } else {
        Verdict::EqualUnderTies
    };
    QueryCheck {
        query: qi,
        line: r.line,
        verdict,
        kth,
        extra,
        violating,
        missing,
        score_delta,
    }
}

/// `payloads.jsonl`, one object per row in row order (§4.3: id = row index).
pub fn read_payloads(path: &Path) -> anyhow::Result<Vec<Value>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .with_context(|| format!("{}:{}: not JSON", path.display(), i + 1))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::Judgement;
    use crate::relevance::{Returned, evaluate};
    use serde_json::json;

    fn parse(v: Value) -> Result<Option<Conditions>, String> {
        parse_conditions(Some(&v))
    }

    fn eps() -> Epsilon {
        crate::differ::tolerance::default_epsilon(Metric::Cosine, 2).0
    }

    #[test]
    fn an_absent_or_empty_condition_is_an_unfiltered_query() {
        assert_eq!(parse_conditions(None), Ok(None));
        assert_eq!(parse(Value::Null), Ok(None));
        assert_eq!(parse(json!({})), Ok(None));
        assert_eq!(parse(json!({"and": []})), Ok(None));
    }

    #[test]
    fn match_values_keep_their_type_and_whitespace_stays_a_keyword() {
        let c = parse(json!({"and": [
            {"colour": {"match": {"value": "Light Orange"}}},
            {"n": {"match": {"value": 87}}},
            {"b": {"match": {"value": true}}},
        ]}))
        .unwrap()
        .unwrap();
        assert_eq!(
            c.to_filter(),
            Filter {
                must: vec![
                    Condition::matches("colour", MatchValue::Keyword("Light Orange".into())),
                    Condition::matches("n", MatchValue::Integer(87)),
                    Condition::matches("b", MatchValue::Boolean(true)),
                ],
                ..Default::default()
            }
        );
        // The trap the explicit `Keyword` avoids: the client's own conversion
        // makes the same string a full-text match.
        assert_eq!(
            MatchValue::from("Light Orange".to_string()),
            MatchValue::Text("Light Orange".into())
        );
    }

    #[test]
    fn match_any_and_range_map_and_or_becomes_should() {
        let c = parse(json!({
            "and": [{"t": {"match_any": {"any": ["x", "y z"]}}}],
            "or": [
                {"i": {"match_any": {"any": [1, 2]}}},
                {"s": {"range": {"gt": 0.25, "lte": 1, "lt": null}}},
            ],
        }))
        .unwrap()
        .unwrap();
        assert_eq!(
            c.to_filter(),
            Filter {
                must: vec![Condition::matches(
                    "t",
                    vec!["x".to_string(), "y z".to_string()]
                )],
                should: vec![
                    Condition::matches("i", vec![1i64, 2]),
                    Condition::range(
                        "s",
                        Range {
                            gt: Some(0.25),
                            lte: Some(1.0),
                            ..Default::default()
                        }
                    ),
                ],
                ..Default::default()
            }
        );
    }

    #[test]
    fn what_cannot_be_sent_as_the_dataset_means_it_is_refused() {
        for (v, says) in [
            (
                json!({"and": [{"d": {"match_text": {"text": "a b"}}}]}),
                "match_text",
            ),
            (
                json!({"and": [{"t": {"range": {"gt": "2020-01-01"}}}]}),
                "datetime",
            ),
            (
                json!({"and": [{"g": {"geo": {"lat": 1, "lon": 2, "radius": 3}}}]}),
                "geo",
            ),
            (json!({"and": [{"f": {"match": {"value": 0.5}}}]}), "float"),
            (
                json!({"and": [{"m": {"match_any": {"any": ["a", 1]}}}]}),
                "mix",
            ),
            (json!({"and": [{"x": {"wat": {}}}]}), "unknown"),
            (
                json!({"and": [{"r": {"range": {"between": 1}}}]}),
                "unknown",
            ),
            (json!({"not": []}), "not part of the grammar"),
            (json!({"and": {}}), "not an array"),
        ] {
            let e = parse(v.clone()).expect_err(&v.to_string());
            assert!(e.contains(says), "{v}: {e}");
        }
    }

    #[test]
    fn the_oracle_reads_conditions_as_qdrant_does() {
        let c = parse(json!({"and": [
            {"tag": {"match": {"value": "a"}}},
            {"s": {"range": {"gt": 0.3}}},
        ]}))
        .unwrap()
        .unwrap();
        assert!(c.matches(&json!({"tag": "a", "s": 0.31})));
        // Arrays match when any element does.
        assert!(c.matches(&json!({"tag": ["b", "a"], "s": [0.1, 0.5]})));
        assert!(!c.matches(&json!({"tag": "a", "s": 0.3})), "gt is strict");
        assert!(
            !c.matches(&json!({"tag": "a"})),
            "a missing key never matches"
        );
        assert!(
            !c.matches(&json!({"tag": "A", "s": 1})),
            "keywords are exact"
        );
        let i = parse(json!({"and": [{"n": {"match": {"value": 3}}}]}))
            .unwrap()
            .unwrap();
        assert!(i.matches(&json!({"n": 3})));
        assert!(!i.matches(&json!({"n": "3"})));
        assert!(!i.matches(&json!({"n": 3.5})));
        let or = parse(json!({"or": [
            {"a": {"match": {"value": 1}}},
            {"b": {"match_any": {"any": ["x"]}}},
        ]}))
        .unwrap()
        .unwrap();
        assert!(or.matches(&json!({"b": "x"})));
        assert!(!or.matches(&json!({"a": 2, "b": "y"})));
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("strawmann-dstests-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_tests_file_becomes_a_truth_and_refuses_a_shallow_one() {
        let d = scratch("truth");
        let p = d.join("tests.jsonl");
        let lines = [
            r#"{"query":[1,0],"conditions":{"and":[{"s":{"range":{"gt":0.5}}}]},"closest_ids":[2,0,1],"closest_scores":[0.9,0.8,0.7]}"#,
            "",
            r#"{"query":[0,1],"conditions":{},"closest_ids":[1,2],"closest_scores":[0.6,0.5]}"#,
            r#"{"query":[0,1],"conditions":{"and":[{"x":{"geo":{}}}]},"closest_ids":[],"closest_scores":[]}"#,
        ];
        std::fs::write(&p, lines.join("\n")).unwrap();
        // The limit stops before the geo record, so it is never read.
        let recs = read_tests(&p, Some(2)).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[1].line, 3);
        assert!(recs[0].conditions.is_some() && recs[1].conditions.is_none());
        let gt = truth_from_tests(&recs, 2, Metric::Cosine, 3, 7).unwrap();
        assert_eq!(gt.ids, vec![2, 0, 1, 2]);
        assert_eq!(gt.scores, vec![0.9, 0.8, 0.6, 0.5]);
        assert_eq!(gt.condition.as_deref(), Some(DATASET_CONDITION));
        assert_eq!((gt.dim, gt.k, gt.n_queries, gt.base_checksum), (2, 2, 2, 7));
        let e = truth_from_tests(&recs, 3, Metric::Cosine, 3, 7).unwrap_err();
        assert!(e.contains("query 1 (line 3) ships 2"), "{e}");
        let e = truth_from_tests(&recs, 2, Metric::Cosine, 2, 7).unwrap_err();
        assert!(e.contains("past the corpus"), "{e}");
        // Read in full, the refusal names the query that carries it.
        let e = read_tests(&p, None).unwrap_err().to_string();
        assert!(e.contains(":4 (query 2)") && e.contains("geo"), "{e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_condition_matching_fewer_than_k_points_is_scored_at_its_own_depth() {
        // h-and-m: 676 of 10,000 tests ship fewer than 10 neighbours because
        // their condition matches fewer points. The sweep refused the file.
        let cond = Some(Conditions {
            must: vec![],
            should: vec![],
        });
        let rec = |ids: Vec<u32>, conditions| TestRecord {
            line: 1,
            query: vec![1.0, 0.0],
            conditions,
            closest_scores: ids.iter().map(|&i| 0.9 - f64::from(i) * 0.01).collect(),
            closest_ids: ids,
        };
        let recs = [
            rec(vec![4, 5, 6], cond.clone()),
            rec((0..10).collect(), None),
        ];
        let gt = truth_from_tests(&recs, 10, Metric::Cosine, 12, 0).unwrap();
        assert_eq!((gt.depth(0), gt.depth(1)), (3, 10));
        assert!(gt.scores_for(0)[3].is_nan());
        // All three returned is recall 1.0 and not a short list; two is 2/3.
        let all = Returned {
            ids: vec![4, 5, 6],
            scores: vec![0.86, 0.85, 0.84],
        };
        let two = Returned {
            ids: vec![4, 5],
            scores: vec![0.86, 0.85],
        };
        let full = Returned {
            ids: (0..10).collect(),
            scores: (0..10).map(|i| 0.9 - f64::from(i) * 0.01).collect(),
        };
        let e = eps();
        let g = crate::relevance::evaluate(&gt, &[all, full.clone()], &e, 10, None);
        assert_eq!(
            (g.recall_at_10, g.short_lists, g.asked, g.returned),
            (1.0, 0, 13, 13)
        );
        let g = crate::relevance::evaluate(&gt, &[two, full], &e, 10, None);
        assert!((g.recall_at_10 - (2.0 / 3.0 + 1.0) / 2.0).abs() < 1e-12);
        assert_eq!(g.short_lists, 1);
        // Unconditioned and short is still a malformed record.
        let e = truth_from_tests(&[rec(vec![1, 2], None)], 10, Metric::Cosine, 12, 0).unwrap_err();
        assert!(e.contains("ships 2"), "{e}");
    }

    #[test]
    fn shipped_scores_in_another_convention_are_refused() {
        let r = TestRecord {
            line: 1,
            query: vec![1.0, 0.0],
            conditions: None,
            closest_ids: vec![0],
            closest_scores: vec![0.9],
        };
        let ok = check_shipped_scores(std::slice::from_ref(&r), 1, |_, _| Some(0.9 + 1e-6));
        assert!(ok.is_ok());
        // A distance `1 - cos` where a similarity was expected.
        let e = check_shipped_scores(&[r], 1, |_, _| Some(0.1)).unwrap_err();
        assert!(e.contains("not this metric's"), "{e}");
    }

    #[test]
    fn a_tie_at_the_kth_boundary_counts_and_a_near_miss_does_not() {
        // Shipped truth at k=10: ids 0..10, the 10th scored 0.5. Id 10 is not
        // in it; returned at the 10th score plus half of ε it is a tie, at
        // 1e-3 below it a miss.
        let e = eps();
        let mut scores = vec![0.9];
        scores.extend([0.8; 8]);
        scores.push(0.5);
        let rec = TestRecord {
            line: 1,
            query: vec![1.0, 0.0],
            conditions: None,
            closest_ids: (0..10).collect(),
            closest_scores: scores.clone(),
        };
        let gt = truth_from_tests(&[rec], 10, Metric::Cosine, 12, 0).unwrap();
        let mut ids: Vec<u32> = (0..9).collect();
        ids.push(10);
        let at = |last: f64| {
            let mut s = scores.clone();
            s[9] = last;
            Returned {
                ids: ids.clone(),
                scores: s,
            }
        };
        let tie = 0.5 + e.value / 2.0;
        let g = evaluate(&gt, &[at(tie)], &e, 10, None);
        assert_eq!(g.recall_at_10, 1.0);
        assert_eq!(evaluate(&gt, &[at(0.499)], &e, 10, None).recall_at_10, 0.9);
        // Judged on the oracle's score when one is supplied, not the engine's.
        let tol = e.absolute_at(0.5);
        assert_eq!(
            gt.judge_at(0, 10, 10, 0.0, Some(tie), tol),
            Judgement::Correct
        );
        assert_eq!(
            gt.judge_at(0, 10, 10, tie, Some(0.499), tol),
            Judgement::Miss
        );
    }

    #[test]
    fn the_truth_check_finds_the_one_deliberate_mismatch() {
        // Six 2-d rows, cosine, query (1, 0). `s > 0.5` keeps every row but 4.
        let base: Vec<f32> = vec![
            1.0, 0.0, // 0: the nearest
            0.9, 0.1, // 1
            0.0, 1.0, // 2
            0.9, 0.1, // 3: the twin of 1, a tie at k=2
            1.0, 0.01, // 4: second nearest, filtered out
            -1.0, 0.0, // 5
        ];
        let base_pre: Vec<f32> = base
            .chunks(2)
            .flat_map(|r| oracle::preprocess(Metric::Cosine, r))
            .collect();
        let payloads: Vec<Value> = [0.9, 0.8, 0.7, 0.6, 0.1, 0.9]
            .iter()
            .map(|s| json!({"s": s}))
            .collect();
        let cond = parse(json!({"and": [{"s": {"range": {"gt": 0.5}}}]})).unwrap();
        let q = oracle::preprocess(Metric::Cosine, &[1.0, 0.0]);
        let rec = |ids: Vec<u32>, conditions: Option<Conditions>| TestRecord {
            line: 1,
            query: vec![1.0, 0.0],
            conditions,
            closest_scores: ids
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    oracle::score(Metric::Cosine, &q, &base_pre[i * 2..i * 2 + 2])
                })
                .collect(),
            closest_ids: ids,
        };
        let recs = vec![
            rec(vec![0, 1], cond.clone()),
            rec(vec![0, 3], cond.clone()),
            // The deliberate mismatch: the unfiltered neighbour shipped as a
            // filtered one.
            rec(vec![0, 4], cond.clone()),
            rec(vec![0, 4], None),
        ];
        let c = check_truth(Metric::Cosine, &base_pre, 6, 2, &payloads, &recs, &eps());
        assert_eq!(
            (
                c.checked,
                c.conditioned,
                c.exact,
                c.equal_under_ties,
                c.mismatched
            ),
            (4, 3, 2, 1, 1)
        );
        let m = &c.mismatches[0];
        assert_eq!((m.query, &m.verdict), (2, &Verdict::Mismatch));
        assert_eq!(m.violating, vec![4]);
        assert_eq!(m.missing.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1]);
        assert_eq!(c.max_score_delta, 0.0);
    }
}
