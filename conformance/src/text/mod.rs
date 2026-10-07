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
//!   points, not array elements, since 0a0fd8790).
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
            let token = match &self.stemmer {
                Some(stemmer) => stemmer.stem(&token).into_owned(),
                None => token,
            };
            out.push(token);
        }
        out
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
pub struct Bm25Index {
    tokenizer: Tokenizer,
    /// Indexed by point id; empty for a point with no tokens.
    tf: Vec<HashMap<String, u32>>,
    doc_len: Vec<u32>,
    df: HashMap<String, u32>,
    documents: u32,
    total_tokens: u64,
}

impl Bm25Index {
    /// `docs[i]` holds point `i`'s values of the field: one string, or an
    /// array's elements, or none.
    pub fn build(params: TextParams, docs: &[Vec<String>]) -> Self {
        let tokenizer = Tokenizer::new(params);
        let mut tf = Vec::with_capacity(docs.len());
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut documents = 0u32;
        let mut total_tokens = 0u64;
        for values in docs {
            let mut counts: HashMap<String, u32> = HashMap::new();
            let mut len = 0u32;
            for value in values {
                for token in tokenizer.tokens(value) {
                    *counts.entry(token).or_default() += 1;
                    len += 1;
                }
            }
            if len > 0 {
                documents += 1;
                total_tokens += u64::from(len);
                for term in counts.keys() {
                    *df.entry(term.clone()).or_default() += 1;
                }
            }
            tf.push(counts);
            doc_len.push(len);
        }
        Self {
            tokenizer,
            tf,
            doc_len,
            df,
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
        let df = f64::from(self.df.get(term).copied().unwrap_or(0));
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
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
            .map(|term| match self.tf[point].get(term) {
                Some(&tf) => {
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
        let mut hits: Vec<(u32, f64)> = (0..self.tf.len())
            .filter(|&p| terms.iter().any(|t| self.tf[p].contains_key(t)))
            .filter(|&p| allowed.is_none_or(|f| f(p)))
            .map(|p| {
                let id = u32::try_from(p).expect("point ids are u32 row indices (§4.3)");
                (id, self.score(p, &terms, k1, b))
            })
            .collect();
        hits.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        hits.truncate(limit);
        hits
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
        let Bm25Settings {
            params,
            k1,
            b,
            limit,
        } = settings;
        let index = Bm25Index::build(params, docs);
        let hits = queries
            .par_iter()
            .map(|q| index.search(q, k1, b, limit, None))
            .collect();
        Self {
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

#[cfg(test)]
mod tests {
    use super::*;

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
