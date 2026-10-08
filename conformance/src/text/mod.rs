//! The BM25 oracle: §8.2's tier 1 for the `text` query (decisions.md, 2026-10-07).
//!
//! Qdrant `dev` at 850859ec9 ranks points by BM25 over a payload field's text
//! index (`lib/segment/.../full_text_index/inverted_index/bm25`). This is the
//! same ranking written the obvious way, in `f64`, as `oracle` is for vectors:
//! tokenize every document, count, score every candidate, sort. No pruning, no
//! segments, no `f32`.
//!
//! What it reproduces, and from where:
//!
//! - **Tokens.** Qdrant's `word` tokenizer splits on `!char::is_alphanumeric`
//!   and runs each piece through `TokensProcessor::process_token_cow`: lowercase,
//!   then the stopword check (on the lowercased token, before stemming), then
//!   the Snowball stemmer. The stemmer is the same crate and version Qdrant
//!   links, and the stopword list is Qdrant's, copied verbatim. ASCII folding,
//!   token length bounds and every other tokenizer are outside decisions.md's
//!   scope and have no constructor here.
//! - **Documents.** A point is a document when its field yields at least one
//!   token; the tokens of an array's values are concatenated (Qdrant counts
//!   points, not array elements, since 0a0fd8790). Qdrant counted an array of
//!   two or more values as a document even with no tokens, through its phrase
//!   boundary token, until 8cec8ad (qdrant/qdrant#11016, fixing #11010); the
//!   oracle mirrored that under revision 2 (decisions.md, 2026-10-07).
//! - **Statistics.** `N` documents, `df` per term, `avgdl` = total tokens / `N`,
//!   over the whole collection. Qdrant gathers them per shard; the comparison
//!   runs it on one, where the two are the same.
//! - **Score.** Lucene's BM25 summed over the query's *distinct* terms:
//!   `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl))`, with
//!   `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` clamped at 0, as Qdrant clamps
//!   it. Without deletions `df <= N`, and the clamp never fires.
//! - **Candidates.** A point that holds any query term (OR), restricted by an
//!   optional filter that does not change the statistics.
//!
//! Ties: the oracle orders equal scores by point id. Qdrant's order among equal
//! scores is undefined, which is what the differ's tie-aware tiers are for.

mod stopwords_english;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Context;
use rayon::prelude::*;

/// Which rules a truth was computed under. A truth from another revision is
/// refused rather than compared: the ids would look right and the statistics
/// would not be. 2: an array of two or more values was a document (Qdrant's
/// boundary token). 3: only a point with a token is, as in Qdrant since
/// 8cec8ad (qdrant/qdrant#11016).
pub const ORACLE_REVISION: u32 = 3;

/// Qdrant's BM25 defaults (`Bm25Params`, `bm25/mod.rs`).
pub const DEFAULT_K1: f64 = 1.2;
pub const DEFAULT_B: f64 = 0.75;

/// The tokenizer options decisions.md puts in scope: the `word` tokenizer, with
/// lowercasing, English stopwords and the English Snowball stemmer, each on or
/// off. Qdrant's defaults are lowercase on and the other two off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextParams {
    pub lowercase: bool,
    pub english_stopwords: bool,
    pub english_stemmer: bool,
}

impl Default for TextParams {
    fn default() -> Self {
        Self {
            lowercase: true,
            english_stopwords: false,
            english_stemmer: false,
        }
    }
}

pub struct Tokenizer {
    lowercase: bool,
    stopwords: HashSet<String>,
    stemmer: Option<rust_stemmers::Stemmer>,
}

impl Tokenizer {
    pub fn new(params: TextParams) -> Self {
        // Qdrant lowercases the list when the index lowercases tokens
        // (`StopwordsFilter::add_stopword`), so a capitalised entry still matches.
        let stopwords = if params.english_stopwords {
            stopwords_english::ENGLISH_STOPWORDS
                .iter()
                .map(|w| {
                    if params.lowercase {
                        w.to_lowercase()
                    } else {
                        (*w).to_string()
                    }
                })
                .collect()
        } else {
            HashSet::new()
        };
        Self {
            lowercase: params.lowercase,
            stopwords,
            stemmer: params
                .english_stemmer
                .then(|| rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English)),
        }
    }

    /// Every token of `text`, in order, repeats kept: `tf` and document length
    /// count them.
    pub fn tokens(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.each_token(text, |t| out.push(t.to_string()));
        out
    }

    /// `tokens`, handed to `f` one at a time instead of collected, so a large
    /// corpus is not a `String` per token at once.
    pub fn each_token(&self, text: &str, mut f: impl FnMut(&str)) {
        for piece in text.split(|c: char| !c.is_alphanumeric()) {
            if piece.is_empty() {
                continue;
            }
            let token = if self.lowercase {
                piece.to_lowercase()
            } else {
                piece.to_string()
            };
            if self.stopwords.contains(&token) {
                continue;
            }
            match &self.stemmer {
                Some(stemmer) => f(&stemmer.stem(&token)),
                None => f(&token),
            }
        }
    }

    /// A query's distinct terms. Qdrant scores each once, with no query-term
    /// frequency (`bm25/mod.rs`).
    pub fn query_terms(&self, text: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        self.tokens(text)
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect()
    }
}

/// Corpus statistics and per-document term counts over one collection.
///
/// Terms are interned to `u32` ids and a document holds its `(term, tf)`
/// pairs sorted by term, with a posting list per term so a search visits the
/// points that hold a query term rather than every point: the 1M-document
/// synthetic corpus needs about a gigabyte this way where one map per document
/// needed eighteen. The arithmetic is unchanged.
pub struct Bm25Index {
    tokenizer: Tokenizer,
    term_ids: HashMap<String, u32>,
    df: Vec<u32>,
    /// Points holding each term, ascending.
    postings: Vec<Vec<u32>>,
    /// Indexed by point id; empty for a point with no tokens.
    tf: Vec<Box<[(u32, u32)]>>,
    doc_len: Vec<u32>,
    documents: u32,
    total_tokens: u64,
}

impl Bm25Index {
    /// `docs[i]` holds point `i`'s values of the field: one string, or an
    /// array's elements, or none.
    pub fn build(params: TextParams, docs: &[Vec<String>]) -> Self {
        let tokenizer = Tokenizer::new(params);
        let mut term_ids: HashMap<String, u32> = HashMap::new();
        let mut df: Vec<u32> = Vec::new();
        let mut postings: Vec<Vec<u32>> = Vec::new();
        let mut tf = Vec::with_capacity(docs.len());
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut documents = 0u32;
        let mut total_tokens = 0u64;
        for (point, values) in docs.iter().enumerate() {
            let mut counts: HashMap<u32, u32> = HashMap::new();
            let mut len = 0u32;
            for value in values {
                tokenizer.each_token(value, |token| {
                    let id = match term_ids.get(token) {
                        Some(&id) => id,
                        None => {
                            let id = u32::try_from(term_ids.len()).expect("fewer than 2^32 terms");
                            term_ids.insert(token.to_string(), id);
                            df.push(0);
                            postings.push(Vec::new());
                            id
                        }
                    };
                    *counts.entry(id).or_default() += 1;
                    len += 1;
                });
            }
            let mut pairs: Vec<(u32, u32)> = counts.into_iter().collect();
            pairs.sort_unstable();
            if len > 0 {
                documents += 1;
                total_tokens += u64::from(len);
                let p = u32::try_from(point).expect("point ids are u32 row indices (§4.3)");
                for &(term, _) in &pairs {
                    df[term as usize] += 1;
                    postings[term as usize].push(p);
                }
            }
            tf.push(pairs.into_boxed_slice());
            doc_len.push(len);
        }
        Self {
            tokenizer,
            term_ids,
            df,
            postings,
            tf,
            doc_len,
            documents,
            total_tokens,
        }
    }

    pub fn documents(&self) -> u32 {
        self.documents
    }

    pub fn avgdl(&self) -> Option<f64> {
        (self.documents > 0).then(|| self.total_tokens as f64 / f64::from(self.documents))
    }

    pub fn idf(&self, term: &str) -> f64 {
        let n = f64::from(self.documents);
        let df = f64::from(
            self.term_ids
                .get(term)
                .map_or(0, |&id| self.df[id as usize]),
        );
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
    }

    /// `term`'s frequency in `point`, if it holds it.
    fn tf_of(&self, point: usize, term: &str) -> Option<u32> {
        let id = *self.term_ids.get(term)?;
        let pairs = self.tf.get(point)?;
        pairs
            .binary_search_by_key(&id, |&(t, _)| t)
            .ok()
            .map(|i| pairs[i].1)
    }

    /// Whether `point` holds any of `terms`.
    fn holds_any(&self, point: usize, terms: &[String]) -> bool {
        terms.iter().any(|t| self.tf_of(point, t).is_some())
    }

    /// BM25 of `point` for already-deduplicated `terms`.
    pub fn score(&self, point: usize, terms: &[String], k1: f64, b: f64) -> f64 {
        let Some(avgdl) = self.avgdl() else {
            return 0.0;
        };
        let len = f64::from(self.doc_len[point]);
        let norm = if b > 0.0 {
            k1 * (1.0 - b + b * len / avgdl)
        } else {
            k1
        };
        terms
            .iter()
            .map(|term| match self.tf_of(point, term) {
                Some(tf) => {
                    let tf = f64::from(tf);
                    self.idf(term) * tf * (k1 + 1.0) / (tf + norm)
                }
                None => 0.0,
            })
            .sum()
    }

    /// The exact top `limit` for `query`: every point holding a query term,
    /// optionally restricted by `allowed`, by score descending and point id
    /// ascending among equal scores.
    pub fn search(
        &self,
        query: &str,
        k1: f64,
        b: f64,
        limit: usize,
        allowed: Option<&dyn Fn(usize) -> bool>,
    ) -> Vec<(u32, f64)> {
        let terms = self.tokenizer.query_terms(query);
        let mut candidates: Vec<u32> = terms
            .iter()
            .filter_map(|t| self.term_ids.get(t))
            .flat_map(|&id| self.postings[id as usize].iter().copied())
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        let mut hits: Vec<(u32, f64)> = candidates
            .into_iter()
            .filter(|&p| allowed.is_none_or(|f| f(p as usize)))
            .map(|p| (p, self.score(p as usize, &terms, k1, b)))
            .collect();
        hits.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        hits.truncate(limit);
        // A common term holds most of the corpus, and `truncate` keeps the
        // capacity: a thousand truths kept 16 GB of it on the 1M corpus.
        hits.shrink_to_fit();
        hits
    }
}

/// The `parity` keyword point `point` carries: `keyword_0` on even rows and
/// `keyword_1` on odd, the values bfb's keyword filter draws at cardinality 2.
/// The text collections the differ loads and the benchmark's `bfb/` layouts
/// (`datasets.py`'s `write_bfb_text`) hold the same rule.
pub fn parity_of(point: usize) -> &'static str {
    if point.is_multiple_of(2) {
        "keyword_0"
    } else {
        "keyword_1"
    }
}

/// FNV-1a 64 over a file's bytes, the hash `oracle::checksum_f32` uses: a
/// truth names the exact corpus and query files it was computed from (§4.3).
pub fn checksum_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// One line of `corpus.jsonl`: a point's field values. The line number is the
/// point id (§4.3: ids are row indices); the line's source `id` is for people.
#[derive(serde::Deserialize)]
struct CorpusLine {
    values: Vec<String>,
}

/// `corpus.jsonl`, in row order, and its checksum.
pub fn read_corpus(path: &Path) -> anyhow::Result<(Vec<Vec<String>>, u64)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text =
        std::str::from_utf8(&bytes).with_context(|| format!("{} is not UTF-8", path.display()))?;
    let docs = text
        .lines()
        .enumerate()
        .map(|(row, line)| {
            serde_json::from_str::<CorpusLine>(line)
                .map(|l| l.values)
                .with_context(|| format!("{}: line {}", path.display(), row + 1))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((docs, checksum_bytes(&bytes)))
}

/// `queries.txt`, one query per line, every line kept (a blank one is a query
/// that matches nothing), and its checksum.
pub fn read_queries(path: &Path) -> anyhow::Result<(Vec<String>, u64)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text =
        std::str::from_utf8(&bytes).with_context(|| format!("{} is not UTF-8", path.display()))?;
    Ok((
        text.lines().map(str::to_string).collect(),
        checksum_bytes(&bytes),
    ))
}

/// `qrels.tsv` (`query row, point row, grade` per line, `convert-beir`'s
/// layout), and its checksum.
pub fn read_qrels(path: &Path) -> anyhow::Result<(Vec<crate::relevance::Qrel>, u64)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text =
        std::str::from_utf8(&bytes).with_context(|| format!("{} is not UTF-8", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        anyhow::ensure!(
            cols.len() == 3,
            "{}: line {} is not query\tpoint\tgrade",
            path.display(),
            i + 1
        );
        let parse = || -> anyhow::Result<crate::relevance::Qrel> {
            Ok(crate::relevance::Qrel {
                query: cols[0].parse()?,
                doc: cols[1].parse()?,
                grade: cols[2].parse()?,
            })
        };
        out.push(parse().with_context(|| format!("{}: line {}", path.display(), i + 1))?);
    }
    Ok((out, checksum_bytes(&bytes)))
}

/// What a BM25 truth was computed under: the tokenizer, `k1`, `b` and depth.
#[derive(Clone, Copy, Debug)]
pub struct Bm25Settings {
    pub params: TextParams,
    pub k1: f64,
    pub b: f64,
    pub limit: usize,
}

/// Exact BM25 top-k for every query, and everything it was computed from.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Bm25Truth {
    /// `ORACLE_REVISION` when computed; absent in a file from before it existed.
    #[serde(default)]
    pub revision: u32,
    pub corpus_checksum: String,
    pub query_checksum: String,
    pub lowercase: bool,
    pub english_stopwords: bool,
    pub english_stemmer: bool,
    pub k1: f64,
    pub b: f64,
    pub limit: usize,
    pub documents: u32,
    pub avgdl: Option<f64>,
    /// Per query: `(point id, score)`, best first.
    pub hits: Vec<Vec<(u32, f64)>>,
}

impl Bm25Truth {
    pub fn compute(
        settings: Bm25Settings,
        docs: &[Vec<String>],
        corpus_checksum: u64,
        queries: &[String],
        query_checksum: u64,
    ) -> Self {
        let index = Bm25Index::build(settings.params, docs);
        Self::from_index(
            &index,
            settings,
            corpus_checksum,
            queries,
            query_checksum,
            None,
        )
    }

    /// The truth over an index already built, restricted to the points
    /// `allowed` keeps: a filter narrows the candidates and leaves the
    /// statistics whole, as the engines' filters do.
    pub fn from_index(
        index: &Bm25Index,
        settings: Bm25Settings,
        corpus_checksum: u64,
        queries: &[String],
        query_checksum: u64,
        allowed: Option<&(dyn Fn(usize) -> bool + Sync)>,
    ) -> Self {
        let Bm25Settings {
            params,
            k1,
            b,
            limit,
        } = settings;
        let hits = queries
            .par_iter()
            .map(|q| {
                index.search(
                    q,
                    k1,
                    b,
                    limit,
                    allowed.map(|f| f as &dyn Fn(usize) -> bool),
                )
            })
            .collect();
        Self {
            revision: ORACLE_REVISION,
            corpus_checksum: format!("{corpus_checksum:x}"),
            query_checksum: format!("{query_checksum:x}"),
            lowercase: params.lowercase,
            english_stopwords: params.english_stopwords,
            english_stemmer: params.english_stemmer,
            k1,
            b,
            limit,
            documents: index.documents(),
            avgdl: index.avgdl(),
            hits,
        }
    }
}

/// One tier's outcome in a text conformance run, as `text-differ` reports it.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TierOutcome {
    pub tier: &'static str,
    pub engine: String,
    /// `passed` in the JSON, as the vector differ's tiers spell it, which
    /// `compare.py` and the report read.
    #[serde(rename = "passed")]
    pub pass: bool,
    /// Not run; neither passes nor licenses anything.
    pub skipped: bool,
    pub detail: String,
}

/// The tiers that license strawmANN's text rows, in order: T1 is the one §8.5
/// names, exact value equality against the oracle, and T0 comes before it.
pub const LICENSING_TIERS: [&str; 3] =
    ["T0 refusal statuses", "T1 score value", "T2 tie-aware rank"];

/// What a text conformance run was, for `results.py`'s conformance table:
/// the vector differ's `ConformanceRow` for the `text` query.
pub struct TextConformanceRow<'a> {
    pub dataset: &'a str,
    pub params: TextParams,
    pub k1: f64,
    pub b: f64,
    pub limit: usize,
    pub value_epsilon: f64,
    pub tie_epsilon: f64,
    pub qdrant_version: &'a str,
    pub strawmann_commit: &'a str,
    pub isa_build: &'a str,
    pub corpus_checksum: u64,
    pub query_checksum: u64,
    pub tiers: &'a [TierOutcome],
}

impl TextConformanceRow<'_> {
    /// The last of `LICENSING_TIERS` strawmANN passed with every one before it
    /// passed too. A tier run for both engines at once counts for strawmANN.
    pub fn tier_reached(&self) -> Option<&'static str> {
        let mut reached = None;
        for name in LICENSING_TIERS {
            let mut ran = self
                .tiers
                .iter()
                .filter(|t| {
                    t.tier == name && !t.skipped && (t.engine == "strawmann" || t.engine == "both")
                })
                .peekable();
            if ran.peek().is_none() || !ran.all(|t| t.pass) {
                break;
            }
            reached = Some(name);
        }
        reached
    }

    /// Whether the row licenses a performance claim: T1 reached, and builds
    /// that are named (`differ::identity_known`).
    pub fn licenses_perf(&self) -> bool {
        let t1 = LICENSING_TIERS
            .iter()
            .position(|t| Some(*t) == self.tier_reached());
        t1.is_some_and(|i| i >= 1)
            && crate::differ::identity_known(
                self.strawmann_commit,
                self.qdrant_version,
                self.isa_build,
            )
    }

    /// FNV-1a 64 over everything the row is, as `ConformanceRow::hash` is for
    /// vectors: the corpus, the tokenizer and query settings, both builds, the
    /// checksums, and every tier's name, engine and outcome.
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |bytes: &[u8]| {
            for &x in bytes {
                h ^= u64::from(x);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        mix(b"text");
        mix(self.dataset.as_bytes());
        mix(&ORACLE_REVISION.to_le_bytes());
        mix(&[
            u8::from(self.params.lowercase),
            u8::from(self.params.english_stopwords),
            u8::from(self.params.english_stemmer),
        ]);
        for x in [self.k1, self.b, self.value_epsilon, self.tie_epsilon] {
            mix(&x.to_le_bytes());
        }
        mix(&(self.limit as u64).to_le_bytes());
        mix(self.qdrant_version.as_bytes());
        mix(self.strawmann_commit.as_bytes());
        mix(self.isa_build.as_bytes());
        mix(&self.corpus_checksum.to_le_bytes());
        mix(&self.query_checksum.to_le_bytes());
        mix(&(self.tiers.len() as u64).to_le_bytes());
        for t in self.tiers {
            mix(t.tier.as_bytes());
            mix(t.engine.as_bytes());
            mix(&[u8::from(t.pass), u8::from(t.skipped)]);
        }
        h
    }
}

/// How one engine's `text` results agree with the oracle, over a query set.
#[derive(Debug, Default, serde::Serialize)]
pub struct TextAgreement {
    pub queries: usize,
    pub limit: usize,
    /// Mean recall@`limit` against the oracle's top `limit`, tie-aware: a
    /// returned point outside the truth counts when the oracle scores it within
    /// `tie_epsilon` (relative) of the truth's last score.
    pub recall: f64,
    pub tie_epsilon: f64,
    /// Queries whose ids came back in exactly the oracle's order.
    pub exact_order: usize,
    /// Queries answered with fewer points than the truth holds.
    pub short_lists: usize,
    /// Returned points holding none of the query's terms: impossible for a
    /// correct engine, whatever its statistics.
    pub non_matching: usize,
    /// `|engine score - oracle score|` over every returned point.
    pub max_abs_delta: f64,
    pub p99_abs_delta: f64,
    pub max_rel_delta: f64,
    pub p99_rel_delta: f64,
    pub scored_points: usize,
    /// nDCG@10 and MRR@10 of the engine's results against the qrels, and of
    /// the oracle's own ranking: BM25's semantic relevance on this corpus,
    /// which the engine should reproduce. Absent without qrels.
    pub semantic: Option<crate::relevance::Semantic>,
    pub oracle_semantic: Option<crate::relevance::Semantic>,
}

/// Compare `returned[q]` with the truth for each query. `index` recomputes the
/// oracle's score of any point an engine returns, inside the truth or not.
pub fn agreement(
    index: &Bm25Index,
    truth: &Bm25Truth,
    queries: &[String],
    returned: &[crate::relevance::Returned],
    limit: usize,
    tie_epsilon: f64,
) -> TextAgreement {
    let mut out = TextAgreement {
        queries: queries.len(),
        limit,
        tie_epsilon,
        ..TextAgreement::default()
    };
    let mut abs = Vec::new();
    let mut rel = Vec::new();
    let mut recall_sum = 0.0;
    let mut counted = 0usize;
    for (q, query) in queries.iter().enumerate() {
        let terms = index.tokenizer.query_terms(query);
        let want: &[(u32, f64)] = &truth.hits[q][..truth.hits[q].len().min(limit)];
        let got = &returned[q];
        if got.ids.len() < want.len() {
            out.short_lists += 1;
        }
        if got.ids.iter().copied().eq(want.iter().map(|h| h.0)) {
            out.exact_order += 1;
        }
        let floor = want.last().map(|h| h.1 * (1.0 - tie_epsilon));
        let in_truth: HashSet<u32> = want.iter().map(|h| h.0).collect();
        let mut hits = 0usize;
        for (&id, &score) in got.ids.iter().zip(&got.scores) {
            let point = id as usize;
            if point >= index.tf.len() || !index.holds_any(point, &terms) {
                out.non_matching += 1;
                continue;
            }
            let oracle = index.score(point, &terms, truth.k1, truth.b);
            let d = (score - oracle).abs();
            abs.push(d);
            rel.push(if oracle > 0.0 { d / oracle } else { d });
            if in_truth.contains(&id) || floor.is_some_and(|f| want.len() == limit && oracle >= f) {
                hits += 1;
            }
        }
        if !want.is_empty() {
            recall_sum += hits.min(want.len()) as f64 / want.len() as f64;
            counted += 1;
        }
    }
    out.recall = if counted > 0 {
        recall_sum / counted as f64
    } else {
        1.0
    };
    out.scored_points = abs.len();
    let p99 = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(((v.len() as f64 * 0.99).ceil() as usize).saturating_sub(1))
            .copied()
            .unwrap_or(0.0)
    };
    out.max_abs_delta = abs.iter().copied().fold(0.0, f64::max);
    out.max_rel_delta = rel.iter().copied().fold(0.0, f64::max);
    out.p99_abs_delta = p99(&mut abs);
    out.p99_rel_delta = p99(&mut rel);
    out
}

/// One engine's ranking for one query against the oracle's.
#[derive(Debug, Default, Clone, Copy)]
pub struct RankCheck {
    /// Largest `|engine score - oracle score| / oracle score` over the points
    /// it returned (T1).
    pub max_rel_delta: f64,
    /// The returned points, read through the oracle's scores, are not the
    /// truth's score sequence within `tie_epsilon`: a point missing, an extra
    /// one, or an order that is not a tie reshuffled (T2).
    pub rank_mismatch: bool,
    /// A returned point holding none of the query's terms.
    pub non_matching: bool,
}

impl Bm25Index {
    /// Check `got` against `want`, the oracle's top list for `query` under the
    /// same `k1`, `b` and candidate set. Equal scores may come back in any
    /// order and either of two tied points may take the last place: what must
    /// hold is that the oracle's score of the i-th returned point is the i-th
    /// truth score.
    pub fn check_ranking(
        &self,
        query: &str,
        want: &[(u32, f64)],
        got: &crate::relevance::Returned,
        k1: f64,
        b: f64,
        tie_epsilon: f64,
    ) -> RankCheck {
        let terms = self.tokenizer.query_terms(query);
        let mut out = RankCheck::default();
        if got.ids.len() != want.len() {
            out.rank_mismatch = true;
        }
        for (i, (&id, &score)) in got.ids.iter().zip(&got.scores).enumerate() {
            let point = id as usize;
            if point >= self.tf.len() || !self.holds_any(point, &terms) {
                out.non_matching = true;
                out.rank_mismatch = true;
                continue;
            }
            let oracle = self.score(point, &terms, k1, b);
            let rel = if oracle > 0.0 {
                (score - oracle).abs() / oracle
            } else {
                (score - oracle).abs()
            };
            out.max_rel_delta = out.max_rel_delta.max(rel);
            match want.get(i) {
                Some(&(_, w))
                    if (oracle - w).abs() <= tie_epsilon * w.abs().max(f64::MIN_POSITIVE) => {}
                _ => out.rank_mismatch = true,
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(tier: &'static str, engine: &str, pass: bool) -> TierOutcome {
        TierOutcome {
            tier,
            engine: engine.into(),
            pass,
            skipped: false,
            detail: String::new(),
        }
    }

    fn row(tiers: &[TierOutcome]) -> TextConformanceRow<'_> {
        TextConformanceRow {
            dataset: "scifact",
            params: all_on(),
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            limit: 10,
            value_epsilon: 1e-5,
            tie_epsilon: 1e-5,
            qdrant_version: "1.19.3-dev",
            strawmann_commit: "99e284a",
            isa_build: "native",
            corpus_checksum: 1,
            query_checksum: 2,
            tiers,
        }
    }

    #[test]
    fn a_text_row_licenses_perf_from_strawmanns_t1_and_named_builds() {
        let full = [
            outcome("T0 refusal statuses", "both", true),
            outcome("T1 score value", "strawmann", true),
            outcome("T2 tie-aware rank", "strawmann", true),
            outcome("T1 score value", "qdrant", false),
        ];
        let r = row(&full);
        assert_eq!(r.tier_reached(), Some("T2 tie-aware rank"));
        assert!(
            r.licenses_perf(),
            "Qdrant's own tiers license nothing about strawmANN"
        );

        let t2_failed = [
            outcome("T0 refusal statuses", "both", true),
            outcome("T1 score value", "strawmann", true),
            outcome("T2 tie-aware rank", "strawmann", false),
        ];
        assert_eq!(row(&t2_failed).tier_reached(), Some("T1 score value"));
        assert!(row(&t2_failed).licenses_perf());

        let t1_failed = [
            outcome("T0 refusal statuses", "both", true),
            outcome("T1 score value", "strawmann", false),
            outcome("T2 tie-aware rank", "strawmann", true),
        ];
        assert_eq!(row(&t1_failed).tier_reached(), Some("T0 refusal statuses"));
        assert!(!row(&t1_failed).licenses_perf());

        let mut anonymous = row(&full);
        anonymous.strawmann_commit = crate::differ::COMMIT_UNSET;
        assert!(!anonymous.licenses_perf());
    }

    #[test]
    fn a_text_rows_hash_moves_with_every_outcome_and_build() {
        let pass = [outcome("T1 score value", "strawmann", true)];
        let fail = [outcome("T1 score value", "strawmann", false)];
        let base = row(&pass).hash();
        assert_eq!(base, row(&pass).hash());
        assert_ne!(base, row(&fail).hash());
        let mut other = row(&pass);
        other.qdrant_version = "1.19.4";
        assert_ne!(base, other.hash());
        let mut other = row(&pass);
        other.b = 0.0;
        assert_ne!(base, other.hash());
    }

    #[test]
    fn parity_alternates_from_keyword_0() {
        assert_eq!(
            (parity_of(0), parity_of(1), parity_of(2)),
            ("keyword_0", "keyword_1", "keyword_0")
        );
    }

    #[test]
    fn a_filtered_truth_keeps_the_whole_corpus_statistics() {
        let docs: Vec<Vec<String>> = ["apple pie", "apple", "apple tart", "banana"]
            .iter()
            .map(|s| vec![(*s).to_string()])
            .collect();
        let settings = Bm25Settings {
            params: TextParams::default(),
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            limit: 10,
        };
        let index = Bm25Index::build(settings.params, &docs);
        let queries = vec!["apple".to_string()];
        let even = |p: usize| parity_of(p) == "keyword_0";
        let t = Bm25Truth::from_index(&index, settings, 0, &queries, 0, Some(&even));
        let ids: Vec<u32> = t.hits[0].iter().map(|h| h.0).collect();
        assert_eq!(ids, [0, 2]);
        let whole = Bm25Truth::from_index(&index, settings, 0, &queries, 0, None);
        let score_of = |id| whole.hits[0].iter().find(|h| h.0 == id).unwrap().1;
        assert_eq!(t.hits[0][0].1, score_of(0));
    }

    fn all_on() -> TextParams {
        TextParams {
            lowercase: true,
            english_stopwords: true,
            english_stemmer: true,
        }
    }

    #[test]
    fn the_word_tokenizer_splits_on_anything_not_alphanumeric() {
        let t = Tokenizer::new(TextParams::default());
        assert_eq!(
            t.tokens("Hello, World! foo_bar x42 -- naïve café"),
            ["hello", "world", "foo", "bar", "x42", "naïve", "café"]
        );
        // Unicode alphanumerics stay whole, as `char::is_alphanumeric` keeps them.
        assert_eq!(t.tokens("日本語 Ⅻ"), ["日本語", "ⅻ"]);
        let cased = Tokenizer::new(TextParams {
            lowercase: false,
            ..TextParams::default()
        });
        assert_eq!(cased.tokens("Hello hello"), ["Hello", "hello"]);
    }

    #[test]
    fn stopwords_are_dropped_before_stemming() {
        let t = Tokenizer::new(all_on());
        assert_eq!(
            t.tokens("The runners were running to THE dogs"),
            ["runner", "run", "dog"]
        );
        // "being" is a stopword and is dropped; "beings" is not, so it is stemmed,
        // and its stem may be one: the check came first, as in Qdrant.
        let beings = t.tokens("beings being");
        assert_eq!(beings.len(), 1, "{beings:?}");
        assert_eq!(
            beings[0],
            rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English).stem("beings")
        );
    }

    #[test]
    fn a_query_counts_each_term_once() {
        let t = Tokenizer::new(all_on());
        assert_eq!(t.query_terms("dog dogs Dog cat"), ["dog", "cat"]);
    }

    fn corpus() -> Bm25Index {
        let docs = vec![
            vec!["alpha beta".to_string()],
            vec!["alpha alpha gamma delta".to_string()],
            vec![],
            vec!["beta".to_string(), "gamma gamma".to_string()],
            vec!["the and of".to_string()],
        ];
        Bm25Index::build(all_on(), &docs)
    }

    #[test]
    fn an_array_of_values_without_tokens_is_not_a_document() {
        let docs = vec![
            vec!["alpha".to_string()],
            vec![String::new(), "the".to_string()],
            vec!["the".to_string()],
            vec![String::new()],
        ];
        let idx = Bm25Index::build(all_on(), &docs);
        // Points 1 to 3 tokenize to nothing, an array of two values (point 1)
        // included, as in Qdrant since 8cec8ad (qdrant/qdrant#11016).
        assert_eq!(idx.documents(), 1);
        assert_eq!(idx.avgdl(), Some(1.0));
    }

    #[test]
    fn statistics_count_documents_with_tokens_and_concatenate_arrays() {
        let idx = corpus();
        // Point 2 has no values, and point 4 only stopwords: neither is a document.
        assert_eq!(idx.documents(), 3);
        // 2 + 4 + 3 tokens over 3 documents; point 3's array is one document of 3.
        assert_eq!(idx.avgdl(), Some(3.0));
        let n = 3.0_f64;
        assert!((idx.idf("alpha") - ((n - 2.0 + 0.5) / 2.5 + 1.0).ln()).abs() < 1e-15);
        assert!((idx.idf("delta") - ((n - 1.0 + 0.5) / 1.5 + 1.0).ln()).abs() < 1e-15);
    }

    #[test]
    fn scores_are_lucene_bm25_by_hand() {
        let idx = corpus();
        let (k1, b) = (DEFAULT_K1, DEFAULT_B);
        let terms = idx.tokenizer.query_terms("alpha gamma");
        // Point 1: tf(alpha) = 2, tf(gamma) = 1, len 4, avgdl 3.
        let norm = k1 * (1.0 - b + b * 4.0 / 3.0);
        let want = idx.idf("alpha") * 2.0 * (k1 + 1.0) / (2.0 + norm)
            + idx.idf("gamma") * 1.0 * (k1 + 1.0) / (1.0 + norm);
        assert!((idx.score(1, &terms, k1, b) - want).abs() < 1e-12);
        // b = 0 ignores length, and k1 = 0 leaves the idf alone.
        let flat = idx.idf("alpha") * 2.0 * (k1 + 1.0) / (2.0 + k1)
            + idx.idf("gamma") * (k1 + 1.0) / (1.0 + k1);
        assert!((idx.score(1, &terms, k1, 0.0) - flat).abs() < 1e-12);
        assert!(
            (idx.score(1, &terms, 0.0, b) - (idx.idf("alpha") + idx.idf("gamma"))).abs() < 1e-12
        );
    }

    #[test]
    fn search_is_or_exact_and_filtered_without_moving_statistics() {
        let idx = corpus();
        let got = idx.search("gamma delta", DEFAULT_K1, DEFAULT_B, 10, None);
        let ids: Vec<u32> = got.iter().map(|h| h.0).collect();
        // Points holding either term, best first; point 0 holds neither.
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1) && ids.contains(&3));
        assert!(got[0].1 >= got[1].1);
        // A filter narrows the candidates; the survivors' scores do not move.
        let only_three = |p: usize| p == 3;
        let filtered = idx.search("gamma delta", DEFAULT_K1, DEFAULT_B, 10, Some(&only_three));
        let three = got.iter().find(|h| h.0 == 3).unwrap();
        assert_eq!(filtered, vec![*three]);
        // A duplicated query term scores as one.
        assert_eq!(
            idx.search("gamma gamma delta", DEFAULT_K1, DEFAULT_B, 10, None),
            got
        );
        // An all-stopword query matches nothing.
        assert!(
            idx.search("the of", DEFAULT_K1, DEFAULT_B, 10, None)
                .is_empty()
        );
    }

    #[test]
    fn agreement_scores_every_returned_point_against_the_oracle() {
        use crate::relevance::Returned;
        let docs: Vec<Vec<String>> = ["a b", "a a c", "b c", "c", "d"]
            .iter()
            .map(|s| vec![(*s).to_string()])
            .collect();
        let settings = Bm25Settings {
            params: TextParams::default(),
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            limit: 2,
        };
        let qs = vec!["a c".to_string()];
        let truth = Bm25Truth::compute(settings, &docs, 0, &qs, 0);
        let index = Bm25Index::build(TextParams::default(), &docs);
        let want = &truth.hits[0];
        // An engine that returns the truth, in f32: exact order, deltas at f32.
        let exact = Returned {
            ids: want.iter().map(|h| h.0).collect(),
            scores: want.iter().map(|h| f64::from(h.1 as f32)).collect(),
        };
        let a = agreement(&index, &truth, &qs, &[exact], 2, 1e-6);
        assert_eq!((a.exact_order, a.short_lists, a.non_matching), (1, 0, 0));
        assert!((a.recall - 1.0).abs() < 1e-12);
        assert!(a.max_rel_delta < 1e-6 && a.max_rel_delta > 0.0);
        // One wrong point, one that holds no query term, and a short list.
        let wrong = Returned {
            ids: vec![want[0].0, 4],
            scores: vec![want[0].1, 1.0],
        };
        let a = agreement(&index, &truth, &qs, &[wrong], 2, 1e-6);
        assert_eq!((a.exact_order, a.non_matching), (0, 1));
        assert!((a.recall - 0.5).abs() < 1e-12);
        let short = Returned {
            ids: vec![want[0].0],
            scores: vec![want[0].1],
        };
        assert_eq!(
            agreement(&index, &truth, &qs, &[short], 2, 1e-6).short_lists,
            1
        );
    }

    #[test]
    fn a_ranking_check_accepts_ties_reordered_and_nothing_else() {
        use crate::relevance::Returned;
        // Points 0 and 1 tie, point 2 scores lower.
        let docs: Vec<Vec<String>> = ["a b", "a c", "a d d d"]
            .iter()
            .map(|s| vec![(*s).to_string()])
            .collect();
        let idx = Bm25Index::build(TextParams::default(), &docs);
        let want = idx.search("a", DEFAULT_K1, DEFAULT_B, 3, None);
        let as_ret = |ids: &[u32]| Returned {
            ids: ids.to_vec(),
            scores: ids
                .iter()
                .map(|&i| idx.score(i as usize, &["a".to_string()], DEFAULT_K1, DEFAULT_B))
                .collect(),
        };
        let check =
            |ids: &[u32]| idx.check_ranking("a", &want, &as_ret(ids), DEFAULT_K1, DEFAULT_B, 1e-9);
        assert!(!check(&[0, 1, 2]).rank_mismatch);
        assert!(
            !check(&[1, 0, 2]).rank_mismatch,
            "a tie may come back either way"
        );
        assert!(check(&[2, 0, 1]).rank_mismatch, "not a tie");
        assert!(check(&[0, 1]).rank_mismatch, "a point missing");
        let off = Returned {
            ids: vec![0, 1, 2],
            scores: as_ret(&[0, 1, 2]).scores.iter().map(|s| s * 1.01).collect(),
        };
        let c = idx.check_ranking("a", &want, &off, DEFAULT_K1, DEFAULT_B, 1e-9);
        assert!(!c.rank_mismatch && (c.max_rel_delta - 0.01).abs() < 1e-9);
    }

    #[test]
    fn qrels_read_in_row_space_and_a_bad_line_is_named() {
        let dir = std::env::temp_dir().join(format!("bm25-qrels-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qrels.tsv");
        std::fs::write(&path, "0\t7\t1\n1\t2\t2\n").unwrap();
        let (q, _) = read_qrels(&path).unwrap();
        assert_eq!(
            q.iter()
                .map(|x| (x.query, x.doc, x.grade))
                .collect::<Vec<_>>(),
            [(0, 7, 1), (1, 2, 2)]
        );
        std::fs::write(&path, "0\t7\t1\n0\tseven\t1\n").unwrap();
        let err = read_qrels(&path).err().unwrap();
        assert!(format!("{err:#}").contains("line 2"), "{err:#}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_truth_reads_its_files_in_row_order_and_names_them() {
        let dir = std::env::temp_dir().join(format!("bm25-truth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus.jsonl");
        std::fs::write(
            &corpus,
            "{\"id\": \"d7\", \"values\": [\"Alpha\", \"beta beta\"]}\n\
             {\"id\": 3, \"values\": []}\n\
             {\"id\": \"d9\", \"values\": [\"beta gamma\"]}\n",
        )
        .unwrap();
        let queries = dir.join("queries.txt");
        std::fs::write(&queries, "beta\n\nalpha gamma\n").unwrap();
        let (docs, cc) = read_corpus(&corpus).unwrap();
        let (qs, qc) = read_queries(&queries).unwrap();
        assert_eq!(docs.len(), 3);
        assert_eq!(qs, ["beta", "", "alpha gamma"]);
        let settings = Bm25Settings {
            params: TextParams::default(),
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            limit: 10,
        };
        let truth = Bm25Truth::compute(settings, &docs, cc, &qs, qc);
        assert_eq!(truth.documents, 2);
        assert_eq!(
            truth.hits[0].iter().map(|h| h.0).collect::<Vec<_>>(),
            [0, 2]
        );
        assert!(truth.hits[1].is_empty());
        assert_eq!(truth.hits[2].len(), 2);
        let bytes = std::fs::read(&corpus).unwrap();
        assert_eq!(
            truth.corpus_checksum,
            format!("{:x}", checksum_bytes(&bytes))
        );
        // A malformed line is named, not skipped.
        std::fs::write(&corpus, "{\"id\": 1, \"values\": [\"a\"]}\nnot json\n").unwrap();
        let err = read_corpus(&corpus).err().unwrap();
        assert!(format!("{err:#}").contains("line 2"), "{err:#}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
