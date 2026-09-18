//! §8.5 — the conformance driver.
//!
//! One client, both engines. Every subcommand corresponds to something the spec
//! asks for by name:
//!
//! - `oracle`     — §8.2 tier 1: compute and cache fp64 ground truth (M-1)
//! - `gt-diff`    — §4.3: diff our recomputed GT against a published one
//! - `differ`     — §8.5: run T0–T4 against both engines
//! - `calibrate`  — §8.4: derive ε from measured cross-ISA spread
//! - `relevance`  — §4.4: recall@k, MRDE, nDCG@10 from held-out queries
//! - `dump-scores`, `scroll-check`, `collection-info`, `drop-collections`,
//!   `convert-*`, `datasets` — the tooling around them
//!
//! §8.6's metamorphic properties are not a subcommand: `differ` runs them.
//!
//! §8.9's CI split: "metamorphic + T0 + T1 on a 100k slice per commit; full
//! T0–T4 against the pinned Qdrant container nightly on the full dataset; the
//! fuzz corpus continuously."

mod datasets;
mod differ;
mod engine;
mod metamorphic;
mod oracle;
mod relevance;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "conformance",
    about = "strawmann conformance and relevance harness (spec §8)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// `oracle`'s arguments.
///
/// A `clap::Args` struct rather than fields inline on the variant, so the arm
/// is a delegation and the body lives in a named function. Same flags, same
/// help: clap derives both from these fields either way.
#[derive(clap::Args)]
struct OracleRun {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long, default_value = "euclid")]
    metric: String,
    #[arg(long, default_value_t = 100)]
    k: usize,
    /// §4.3: GT is cached per (dataset, subset size, metric).
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "unnamed")]
    dataset: String,
}

/// `differ`'s arguments.
///
/// A `clap::Args` struct rather than eleven fields on the variant feeding
/// eleven positional parameters. The arm destructured the variant and rebuilt
/// the same values in the same order as `&strawmann, &qdrant, &base,
/// &queries, &ground_truth, ...` — five `&Path`s and three `&str`s among them,
/// any adjacent pair of which could be swapped without the compiler noticing.
#[derive(clap::Args)]
struct DifferRun {
    #[arg(long, default_value = "http://localhost:6334")]
    strawmann: String,
    /// Qdrant's **gRPC** port, not its REST one.
    ///
    /// Defaulted to 6333 — the REST port — which cannot answer a
    /// `qdrant-client` call: the run reaches "[qdrant] recreating
    /// collection" and dies with `h2 protocol error`, which reads like a
    /// transport fault rather than a port number. Qdrant serves REST on
    /// 6333 and gRPC on 6334, and this binary only ever speaks gRPC.
    #[arg(long, default_value = "http://localhost:6334")]
    qdrant: String,
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    ground_truth: PathBuf,
    #[arg(long, default_value = "euclid")]
    metric: String,
    /// §4.3: the corpus this row is about, by name.
    ///
    /// Defaulted empty and inferred from the base file's stem, which is
    /// what this always did — and which is right only when the fbin
    /// happens to be named after its dataset. `sift1m.fbin` is; `base.fbin`
    /// is not, and `dbpedia-openai-1m` and `dbpedia-openai-100K-1536-angular`
    /// both convert to that name, so both rows called themselves "base".
    #[arg(long, default_value = "")]
    dataset: String,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// §8.5's tiers are cumulative; this is the highest to attempt.
    #[arg(long, default_value = "T4")]
    max_tier: String,
    /// ε from a `calibrate` run (§8.4). Without it the differ falls back to
    /// the derived floor, which is tighter than the physical noise at high
    /// dimension and reports a divergence the hardware guarantees. The value
    /// keeps the metric's tolerance kind: relative for dot, euclid and
    /// manhattan, absolute for cosine (`tolerance_kind`).
    #[arg(long)]
    epsilon: Option<f64>,
    /// Write the conformance row as JSON, for `results.py` to record.
    ///
    /// §8's gate is "no performance number is publishable unless the same
    /// build, on the same data, has a green conformance row", and the sink
    /// enforces it by refusing rows whose hash it has never seen. The row
    /// was built here and printed, so the hash existed only on a terminal
    /// and the gate could never actually be satisfied.
    #[arg(long)]
    json: Option<PathBuf>,
}

/// `gt-diff`'s arguments.
#[derive(clap::Args)]
struct GtDiffRun {
    #[arg(long)]
    ours: PathBuf,
    #[arg(long)]
    published: PathBuf,
}

/// `calibrate`'s arguments.
#[derive(clap::Args)]
struct CalibrateRun {
    /// Scores from each ISA arm, one JSON file per arm.
    #[arg(long, num_args = 1..)]
    strawmann_arms: Vec<PathBuf>,
    #[arg(long, num_args = 1..)]
    qdrant_arms: Vec<PathBuf>,
    #[arg(long, default_value = "cosine")]
    metric: String,
    #[arg(long, default_value_t = 768)]
    dim: usize,
    #[arg(long, default_value_t = differ::tolerance::DEFAULT_MULTIPLE)]
    multiple: f64,
    #[arg(long)]
    out: PathBuf,
}

/// `filtered-truth`'s arguments.
///
/// docs/workloads.md W12 point 3: "Ground truth is recomputed per condition."
/// The condition's *members* are not derivable here -- bfb assigns keyword
/// payloads from an unseeded RNG, so which point carries which keyword is only
/// knowable by asking the engine that holds them. This scrolls for that, then
/// computes the restricted fp64 truth and caches it under a condition-keyed
/// name.
#[derive(clap::Args)]
struct FilteredTruthRun {
    #[arg(long, default_value = "http://localhost:6334")]
    engine: String,
    /// The loaded collection whose payloads decide who matches.
    #[arg(long)]
    collection: String,
    /// The payload field the condition is on. bfb's first keyword field is `a`.
    #[arg(long, default_value = "a")]
    field: String,
    /// The values the condition accepts, comma-separated. With bfb's `-k V`
    /// each value is ~1/V of the collection, so one value and ten values are
    /// the two selectivity grades W12 point 2 asks for.
    #[arg(long, num_args = 1.., value_delimiter = ',')]
    values: Vec<String>,
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long, default_value = "euclid")]
    metric: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Cap base vectors, to match a collection holding a prefix of the corpus.
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, default_value_t = 4096)]
    page: u32,
    #[arg(long)]
    out: PathBuf,
}

// There is deliberately no `--limit-queries` here. `relevance` compares a
// ground truth's `query_checksum` against the *whole* query file and applies
// its own cap afterwards, so a truth computed over a truncated query set is
// refused by §4.3 no matter how right it is otherwise — measured, as
// `checksum e25ac9eaf10f1b80 vs 1eb61b7e8c2a5b46`. `run_oracle` caps nothing
// either; a truth is over the query file, and how many of it a sweep reads is
// the sweep's business.

/// `convert-vecs`'s arguments.
#[derive(clap::Args)]
struct ConvertVecsRun {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    limit: Option<usize>,
}

/// `convert-npy`'s arguments.
#[derive(clap::Args)]
struct ConvertNpyRun {
    /// The extracted bundle directory, holding `vectors.npy` and `tests.jsonl`.
    #[arg(long)]
    in_dir: PathBuf,
    #[arg(long)]
    out_base: PathBuf,
    #[arg(long)]
    out_queries: PathBuf,
}

/// `convert-parquet`'s arguments.
#[derive(clap::Args)]
struct ConvertParquetRun {
    /// Directory of `.parquet` shards.
    #[arg(long)]
    in_dir: PathBuf,
    #[arg(long, default_value = "openai")]
    column: String,
    #[arg(long)]
    out_base: PathBuf,
    #[arg(long)]
    out_queries: PathBuf,
    /// Rows taken from the tail as the held-out query set (§4.2).
    #[arg(long, default_value_t = 10_000)]
    n_queries: usize,
    /// Cap total rows read, for a quick subset.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Subcommand)]
enum Command {
    /// §8.2 tier 1 / M-1: compute fp64 ground truth and cache it.
    Oracle(OracleRun),

    /// §4.3: diff recomputed ground truth against a published `.ivecs`.
    GtDiff(GtDiffRun),

    /// §8.5: run the differ tiers against both engines.
    Differ(DifferRun),

    /// §8.4's input: dump exact-search scores from one engine build to JSON.
    ///
    /// `calibrate` derives ε from how much an engine already disagrees with
    /// *itself* across ISA arms. That needs one score file per arm, and nothing
    /// produced them — which is why every differ run so far fell back to the ε
    /// floor and printed a note saying so.
    ///
    /// Exact search only: the point is to isolate fp32 summation-order
    /// differences in the kernel, and an approximate traversal would mix graph
    /// non-determinism into the measurement.
    DumpScores(DumpRun),

    /// §8.4: derive ε from measured cross-ISA spread and write docs/tolerance.md.
    Calibrate(CalibrateRun),

    /// §8.4 / W12 point 3: fp64 ground truth restricted to the points matching
    /// a payload condition, cached per condition.
    ///
    /// §4.3's cached truth is unfiltered and does not score a filtered query:
    /// the neighbours of `q` among all points are not its neighbours among the
    /// points the filter keeps, and the k-th distance recall is measured
    /// against moves with the condition.
    FilteredTruth(FilteredTruthRun),

    /// §8.6 / W10: measure recall against our fp64 ground truth on **one**
    /// engine, over an `ef` sweep.
    ///
    /// The differ needs both engines; this needs one, which is what W10's
    /// harness side is (§4.1: bfb reports latency, this reports recall, joined
    /// on `ef`). It is also the only way to test the engine end-to-end at
    /// dataset scale without a Qdrant instance to compare against.
    Relevance(RelevanceRun),

    /// §9: convert `fvecs`/`ivecs`-era files to `fbin`.
    ///
    /// §4.3 makes `fbin` canonical and bfb's dataset path only reads `fbin`
    /// (§4.1) — but SIFT1M ships as `fvecs`, so W1 cannot load the same corpus
    /// the fp64 oracle read without this step. Using a different corpus path
    /// for load than for ground truth is exactly the discrepancy §4.1 warns
    /// about.
    ConvertVecs(ConvertVecsRun),

    /// §9: convert a vector-db-benchmark bundle (`vectors.npy` +
    /// `tests.jsonl`) to `fbin`.
    ///
    /// bfb reads that layout natively, so this is not needed to *load* one of
    /// §4.2's `tar` entries — but the oracle, the differ and `relevance` all
    /// read `fbin`, so without it such a dataset can be searched and never
    /// scored, and §7.4 forbids a qps number without its recall.
    ///
    /// Unlike `convert-parquet` there is no query split to hold out: the
    /// queries arrive in their own file and were never part of `vectors.npy`.
    ConvertNpy(ConvertNpyRun),

    /// §9: convert parquet shards to `fbin`, holding out a query split.
    ConvertParquet(ConvertParquetRun),

    /// §12 phase 2 / W13: page a collection with `Scroll` through the real
    /// client, checking that **prost** decodes what we emit.
    ///
    /// The e2e tests drive scroll over a real socket but with our own decoder.
    /// This is the compatibility question that actually matters: a wrong field
    /// number produces bytes our decoder tolerates and qdrant-client rejects.
    ScrollCheck {
        #[arg(long, default_value = "http://localhost:6334")]
        engine: String,
        #[arg(long)]
        collection: String,
        #[arg(long, default_value_t = 1000)]
        page: u32,
        /// Stop after this many points; 0 walks the whole collection.
        #[arg(long, default_value_t = 0)]
        max_points: usize,
    },

    /// Capture what a collection **is**, from the engine, rather than what it
    /// was asked to be.
    ///
    /// §4 stamps the settings the harness sent and §8.9 pins the image; neither
    /// reads anything back. `--segments 1` sets Qdrant's
    /// `default_segment_number`, a target its optimizer may miss, and the
    /// segment count is the largest confound in the `ef` comparison
    /// (`docs/comparison.md` §2). One client, both engines, per §8.5.
    CollectionInfo {
        #[arg(long, default_value = "http://localhost:6334")]
        engine: String,
        #[arg(long, default_value = "engine")]
        label: String,
        /// Comma-separated. Missing collections are reported, not fatal: a run
        /// that skipped a workload has no collection for it.
        #[arg(long, default_value = "bench0,bench2,bench6,bench7,bench8")]
        collections: String,
        #[arg(long)]
        json: PathBuf,
    },

    /// Delete collections a run has finished with, while the engine is up.
    ///
    /// strawmANN holds every collection resident: one arena of
    /// `capacity x dim x 4` bytes per collection, first-touched at create, so
    /// a collection costs its full capacity from the moment it exists until
    /// the process exits. Nothing dropped them, because until a tier arrived
    /// where they did not all fit, nothing had to. dbpedia-openai-1m did:
    /// five 1536-dimension collections at 7.08 GiB each is 35.4 GiB before a
    /// quantized store or a graph, and the arm was OOM-killed at 45.6 GiB.
    ///
    /// `bench1` is the clearest case — W1 measures ingest into it and nothing
    /// reads it again, so it holds 7.08 GiB from the first row to the last for
    /// a measurement that has already finished.
    DropCollections {
        #[arg(long, default_value = "http://localhost:6334")]
        engine: String,
        /// Comma-separated. A collection that is already absent is not an
        /// error: this runs after phases that may have been skipped.
        #[arg(long)]
        collections: String,
    },

    /// List the §4.2 dataset tiers.
    Datasets,
}

/// One spelling table: `Metric::parse` is what the dataset descriptor goes
/// through, and a second copy here had drifted (`euclidean` was legal in a
/// descriptor and refused on the command line).
fn parse_metric(s: &str) -> anyhow::Result<oracle::Metric> {
    oracle::Metric::parse(s)
        .ok_or_else(|| anyhow::anyhow!("unknown metric {s}; expected dot|cosine|euclid|manhattan"))
}

/// `--max-tier` as a rank: T0..T4, and nothing else. It was compared as a
/// string at each gate, so an unknown value ran everything but T4 without a
/// word, and `T2` did not stop before T3.
fn parse_max_tier(s: &str) -> anyhow::Result<u8> {
    Ok(match s {
        "T0" => 0,
        "T1" => 1,
        "T2" => 2,
        "T3" => 3,
        "T4" => 4,
        other => anyhow::bail!("unknown --max-tier {other}; expected T0|T1|T2|T3|T4"),
    })
}

fn read_vectors(path: &Path, limit: Option<usize>) -> anyhow::Result<datasets::Vectors> {
    // §4.3 makes fbin canonical, but fvecs inputs are common enough that
    // picking by extension avoids a conversion step for a one-off run.
    match path.extension().and_then(|e| e.to_str()) {
        Some("fvecs") => datasets::read_fvecs(path, limit),
        _ => datasets::read_fbin(path, limit),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Relevance(a) => run_relevance(a),

        Command::ConvertVecs(a) => run_convert_vecs(a),

        Command::ConvertNpy(a) => run_convert_npy(a),

        Command::ConvertParquet(a) => run_convert_parquet(a),

        Command::ScrollCheck {
            engine,
            collection,
            page,
            max_points,
        } => run_scroll_check(&engine, &collection, page, max_points),

        Command::CollectionInfo {
            engine,
            label,
            collections,
            json,
        } => run_collection_info(&engine, &label, &collections, &json),

        Command::DropCollections {
            engine,
            collections,
        } => run_drop_collections(&engine, &collections),

        Command::Datasets => {
            run_datasets();
            Ok(())
        }

        Command::Oracle(a) => run_oracle(a),

        Command::GtDiff(a) => run_gt_diff(a),

        Command::DumpScores(a) => run_dump_scores(a),

        Command::Calibrate(a) => run_calibrate(a),

        Command::FilteredTruth(a) => run_filtered_truth(a),

        Command::Differ(a) => run_differ(a),
    }
}

/// §4.2: parquet shards to fbin, base and held-out queries.
fn run_convert_parquet(a: ConvertParquetRun) -> anyhow::Result<()> {
    let ConvertParquetRun {
        in_dir,
        column,
        out_base,
        out_queries,
        n_queries,
        limit,
    } = a;

    let c = datasets::parquet_convert::convert(
        &in_dir,
        &column,
        &out_base,
        &out_queries,
        n_queries,
        limit,
    )?;
    println!("{}", c.describe());
    println!("  base    {}", c.base_path.display());
    println!("  queries {}", c.queries_path.display());
    println!();
    println!("§4.2 marks this tier's ground truth 'recompute'; the query split is");
    println!("held out of the base set, so no query is its own nearest neighbour.");
    Ok(())
}

/// §4.2: a .npy corpus to fbin, base and queries.
fn run_convert_npy(a: ConvertNpyRun) -> anyhow::Result<()> {
    let ConvertNpyRun {
        in_dir,
        out_base,
        out_queries,
    } = a;

    let c = datasets::npy_convert::convert(&in_dir, &out_base, &out_queries)?;
    println!("{}", c.describe());
    println!("  base    {}", c.base_path.display());
    println!("  queries {}", c.queries_path.display());
    println!();
    println!("§4.2 marks these entries' shipped neighbours filtered or k<100, so the");
    println!("ground truth is the fp64 oracle's: run `oracle` against these two files.");
    Ok(())
}

/// §4.3: fvecs or fbin to fbin, with its checksum.
fn run_convert_vecs(a: ConvertVecsRun) -> anyhow::Result<()> {
    let ConvertVecsRun { input, out, limit } = a;

    let v = read_vectors(&input, limit)?;
    datasets::write_fbin(&out, &v)?;
    println!(
        "wrote {} x {} to {} (checksum {:x})",
        v.n,
        v.dim,
        out.display(),
        oracle::checksum_f32(&v.data)
    );
    Ok(())
}

/// §8.4's ε, derived from measured cross-arm spread.
fn run_calibrate(a: CalibrateRun) -> anyhow::Result<()> {
    let CalibrateRun {
        strawmann_arms,
        qdrant_arms,
        metric,
        dim,
        multiple,
        out,
    } = a;

    let m = parse_metric(&metric)?;
    let s = spread_across_arms(&strawmann_arms)?;
    let q = spread_across_arms(&qdrant_arms)?;

    eprintln!("strawmann cross-ISA spread: {}", s.describe());
    eprintln!("qdrant    cross-ISA spread: {}", q.describe());

    let eps = differ::tolerance::calibrate(m, dim, q, s, multiple);
    println!("{}", eps.describe());

    // One cell per run, merged into the document rather than
    // replacing it: writing the table from this single entry wiped
    // every other (metric, dim) row.
    let table = differ::tolerance::Table {
        entries: vec![eps.clone()],
    };
    let md = match std::fs::read_to_string(&out) {
        Ok(existing) => table.merge_markdown(&existing),
        Err(_) => table.to_markdown(),
    };
    std::fs::write(&out, md)?;
    println!("wrote {}", out.display());
    // The mirror in `CALIBRATED` is hand-maintained and has drifted
    // from the document before; the exact entry is printed so it is
    // pasted rather than retyped.
    println!("add to `CALIBRATED` in conformance/src/differ/tolerance.rs:");
    println!("{}", eps.rust_literal());
    Ok(())
}

/// §4.3: recomputed ground truth against a published `.ivecs`.
fn run_gt_diff(a: GtDiffRun) -> anyhow::Result<()> {
    let GtDiffRun { ours, published } = a;

    let gt = oracle::load(&ours)?;
    let (n, k, ids) = datasets::read_ivecs(&published)?;
    anyhow::ensure!(
        n >= gt.n_queries && k >= gt.k,
        "published GT is {n}x{k}, ours is {}x{} — cannot compare",
        gt.n_queries,
        gt.k
    );
    // Reshape the published rows to our k.
    let mut trimmed = Vec::with_capacity(gt.n_queries * gt.k);
    for qi in 0..gt.n_queries {
        trimmed.extend_from_slice(&ids[qi * k..qi * k + gt.k]);
    }
    let d = oracle::diff_against_published(&gt, &trimmed);
    println!("{}", d.describe());
    println!();
    println!("§4.3: \"Several published GTs were computed in fp32 and have genuine");
    println!("disagreements near the k-th boundary. Use ours, publish the diff once,");
    println!("and note that a disagreement with the shipped GT is expected rather");
    println!("than alarming.\"");
    if d.interior > 0 {
        println!();
        println!(
            "NOTE: {} queries differ in the interior, not just at the boundary. That is",
            d.interior
        );
        println!("not the expected fp32-rounding pattern and is worth investigating.");
    }
    Ok(())
}

/// §4.2's tiers, and the sets pinned alongside them.
fn run_datasets() {
    println!(
        "conformance/datasets/datasets.json: §4.2's tiers, and the sets \
                 pinned alongside them:\n"
    );
    // Widened to the longest name rather than pinned at 20: the
    // filtered-benchmark entries run past it, and a row whose first
    // column overflows shifts every column after it on that row alone.
    let w = datasets::DATASETS
        .iter()
        .map(|d| d.name.len())
        .max()
        .unwrap_or(20)
        .max(20);
    println!(
        "{:<w$} {:<10} {:>11} {:>6} {:>9} {:>10} {:>9}  role",
        "name", "status", "n", "dim", "queries", "metric", "gt"
    );
    for d in datasets::DATASETS.iter() {
        println!(
            "{:<w$} {:<10} {:>11} {:>6} {:>9} {:>10} {:>9}  {}",
            d.name,
            d.status,
            d.n,
            d.dim,
            d.n_queries,
            d.metric.as_str(),
            if d.gt_shipped { "shipped" } else { "recompute" },
            d.role
        );
    }
    let declared = datasets::DATASETS.len() - datasets::available().count();
    if declared > 0 {
        println!(
            "\n{declared} dataset(s) are `declared`: §4.2 names them but no files \
                     are pinned yet."
        );
        println!("Pin them with `conformance/datasets/datasets.py add`.");
    }
}

/// §8.2 tier 1 / M-1: fp64 ground truth for one corpus, cached.
fn run_oracle(a: OracleRun) -> anyhow::Result<()> {
    let OracleRun {
        base,
        queries,
        metric,
        k,
        limit,
        out,
        dataset,
    } = a;

    let m = parse_metric(&metric)?;
    let b = read_vectors(&base, limit)?;
    let q = read_vectors(&queries, None)?;
    anyhow::ensure!(
        b.dim == q.dim,
        "base dim {} != query dim {} — these are not the same dataset",
        b.dim,
        q.dim
    );

    eprintln!(
        "computing fp64 ground truth: {} base x {} queries, d={}, {}, k={} ({} comparisons)",
        b.n,
        q.n,
        b.dim,
        m.as_str(),
        k,
        b.n * q.n
    );
    let gt = oracle::compute(m, &b.data, b.n, &q.data, q.n, b.dim, k);
    oracle::save(&gt, &out)?;
    println!(
        "wrote {} ({}), base checksum {:016x}, query checksum {:016x}",
        out.display(),
        oracle::cache_key(&dataset, b.n, m, k),
        gt.base_checksum,
        gt.query_checksum
    );
    Ok(())
}

/// Whether one point's payload satisfies the condition.
///
/// Both shapes bfb can write are accepted: a bare string when `--max-keywords`
/// is 1, and a list when it is more. Anything else -- a missing field, a number
/// -- is a point the condition does not keep, which is not an error: a
/// collection may legitimately hold points the filter excludes.
fn payload_matches(
    payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
    field: &str,
    wanted: &std::collections::HashSet<String>,
) -> bool {
    use qdrant_client::qdrant::value::Kind;
    let Some(v) = payload.get(field) else {
        return false;
    };
    match v.kind.as_ref() {
        Some(Kind::StringValue(s)) => wanted.contains(s),
        Some(Kind::ListValue(l)) => l.values.iter().any(|x| match x.kind.as_ref() {
            Some(Kind::StringValue(s)) => wanted.contains(s),
            _ => false,
        }),
        _ => false,
    }
}

/// W12 point 3, as a command: which ids the condition keeps, then the fp64
/// truth over exactly those.
#[tokio::main(flavor = "multi_thread")]
async fn run_filtered_truth(a: FilteredTruthRun) -> anyhow::Result<()> {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    use qdrant_client::qdrant::{PointId, ScrollPointsBuilder};

    anyhow::ensure!(!a.values.is_empty(), "--values needs at least one value");
    let m = parse_metric(&a.metric)?;
    let wanted: std::collections::HashSet<String> = a.values.iter().cloned().collect();
    let condition = format!("{} in {{{}}}", a.field, {
        let mut v = a.values.clone();
        v.sort();
        v.join(",")
    });

    let eng = engine::Engine::connect("strawmann", &a.engine)?;
    let mut cursor: Option<PointId> = None;
    let mut matching: Vec<u32> = Vec::new();
    let mut scanned: usize = 0;
    let t = std::time::Instant::now();
    loop {
        let mut b = ScrollPointsBuilder::new(&a.collection)
            .limit(a.page)
            .with_payload(true)
            // The vectors are the corpus and are already on disk here; asking
            // for 200k of them back would dominate the request for no use.
            .with_vectors(false);
        if let Some(c) = cursor.clone() {
            b = b.offset(c);
        }
        let resp = eng.client.scroll(b).await?;
        for pt in &resp.result {
            let id = match pt.id.as_ref().and_then(|i| i.point_id_options.as_ref()) {
                Some(PointIdOptions::Num(n)) => *n,
                _ => anyhow::bail!("non-numeric point id"),
            };
            scanned += 1;
            if payload_matches(&pt.payload, &a.field, &wanted) {
                matching.push(u32::try_from(id)?);
            }
        }
        match resp.next_page_offset {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    // Ascending, so the restricted scan reads the corpus in order and the
    // condition's id set is comparable between runs.
    matching.sort_unstable();
    anyhow::ensure!(
        !matching.is_empty(),
        "no point in `{}` matches {condition}: the condition would score every \
         query against an empty truth, which is not a recall of 0 but an absence \
         of a measurement",
        a.collection
    );
    eprintln!(
        "condition {condition}: {} of {scanned} points ({:.2}%) in {:.1}s",
        matching.len(),
        100.0 * matching.len() as f64 / scanned.max(1) as f64,
        t.elapsed().as_secs_f64()
    );

    let b = read_vectors(&a.base, a.limit)?;
    let q = read_vectors(&a.queries, None)?;
    anyhow::ensure!(
        b.dim == q.dim,
        "base dim {} != query dim {} — these are not the same dataset",
        b.dim,
        q.dim
    );
    eprintln!(
        "computing fp64 ground truth over the condition: {} of {} base x {} queries, \
         d={}, {}, k={}",
        matching.len(),
        b.n,
        q.n,
        b.dim,
        m.as_str(),
        a.k
    );
    let gt = oracle::compute_filtered(
        m, &b.data, b.n, &q.data, q.n, b.dim, a.k, &condition, &matching,
    );
    oracle::save(&gt, &a.out)?;
    println!(
        "wrote {} ({}), {} matching, base checksum {:016x}, query checksum {:016x}",
        a.out.display(),
        oracle::cache_key_for(&a.collection, b.n, m, a.k, Some(&condition)),
        gt.n_matching.unwrap_or(0),
        gt.base_checksum,
        gt.query_checksum
    );
    Ok(())
}

/// §8.9's argument, applied to the collection: a pin nobody reads back is not
/// a pin.
///
/// The harness stamps what it *sent* — `--segments 1`, `indexing_threshold`,
/// the quantization it asked for — and nothing has ever checked what the engine
/// built. For Qdrant `--segments` sets `default_segment_number`, which its
/// optimizer treats as a target; `docs/comparison.md` §2 records a run where it
/// produced eight, and eight segments searched at the requested `ef` is roughly
/// eight times the traversal of one. That is the difference between a
/// comparable knob and an incomparable one, and it was invisible.
// `multi_thread`, as every other subcommand here: the client keeps background
// tasks alive and a current-thread runtime simply never returns from the first
// call. Measured, not guessed — the smoke test timed out at 120 s.
#[tokio::main(flavor = "multi_thread")]
async fn run_collection_info(
    url: &str,
    label: &str,
    collections: &str,
    out: &Path,
) -> anyhow::Result<()> {
    let eng = engine::Engine::connect(label, url)?;
    let mut got = Vec::new();
    for name in collections
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        match eng.collection_config(name).await {
            Ok(v) => {
                println!("  {name}: {v}");
                got.push(v);
            }
            // A collection a run never created is absent, not a failure: the
            // row that would have built it may simply not have run.
            Err(e) => println!("  {name}: absent ({e})"),
        }
    }
    let doc = serde_json::json!({ "label": label, "engine": url, "collections": got });
    std::fs::write(out, serde_json::to_string_pretty(&doc)? + "\n")?;
    println!("wrote {}", out.display());
    Ok(())
}

/// Delete each named collection, reporting what went and what was already gone.
#[tokio::main(flavor = "multi_thread")]
async fn run_drop_collections(url: &str, collections: &str) -> anyhow::Result<()> {
    let eng = engine::Engine::connect("drop", url)?;
    for name in collections
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let before = eng.count(name).await.ok();
        eng.delete_collection(name).await;
        match before {
            Some(n) => println!("  dropped {name} ({n} points)"),
            None => println!("  {name}: absent"),
        }
    }
    Ok(())
}

#[tokio::main(flavor = "multi_thread")]
async fn run_scroll_check(
    url: &str,
    collection: &str,
    page: u32,
    max_points: usize,
) -> anyhow::Result<()> {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    use qdrant_client::qdrant::{PointId, ScrollPointsBuilder};

    let eng = engine::Engine::connect("strawmann", url)?;
    let mut cursor: Option<PointId> = None;
    let mut seen: usize = 0;
    let mut pages: usize = 0;
    let mut last: Option<u64> = None;
    let t = std::time::Instant::now();

    loop {
        let mut b = ScrollPointsBuilder::new(collection).limit(page);
        if let Some(c) = cursor.clone() {
            b = b.offset(c);
        }
        // prost decodes the response here. A `time` double written at field 2
        // instead of 3 lands inside the repeated `result` list and this call
        // fails rather than returning a short page.
        let resp = eng.client.scroll(b).await?;
        pages += 1;
        for p in &resp.result {
            let id = match p.id.as_ref().and_then(|i| i.point_id_options.as_ref()) {
                Some(PointIdOptions::Num(n)) => *n,
                _ => anyhow::bail!("non-numeric point id"),
            };
            // Ids must come back strictly ascending, or pagination is repeating
            // or skipping points.
            if let Some(prev) = last {
                anyhow::ensure!(id > prev, "scroll went backwards: {prev} then {id}");
            }
            last = Some(id);
            seen += 1;
        }
        match resp.next_page_offset {
            Some(next) => cursor = Some(next),
            None => break,
        }
        if max_points > 0 && seen >= max_points {
            println!("stopping early at {seen} points");
            break;
        }
    }

    let secs = t.elapsed().as_secs_f64();
    println!(
        "scrolled {seen} points in {pages} pages ({page}/page) in {secs:.2}s = {:.0} points/s",
        seen as f64 / secs
    );
    println!("ids strictly ascending, prost decoded every page");
    Ok(())
}

#[derive(clap::Args)]
struct DumpRun {
    #[arg(long, default_value = "http://localhost:6334")]
    engine: String,
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long, default_value = "cosine")]
    metric: String,
    #[arg(long, default_value_t = 10)]
    limit: u64,
    #[arg(long)]
    limit_queries: Option<usize>,
    #[arg(long, default_value = "calib")]
    collection: String,
    /// Skip create+upload when the collection is already loaded.
    #[arg(long, default_value_t = false)]
    skip_upload: bool,
    #[arg(long)]
    out: PathBuf,
}

#[tokio::main(flavor = "multi_thread")]
async fn run_dump_scores(d: DumpRun) -> anyhow::Result<()> {
    let m = parse_metric(&d.metric)?;
    let base = read_vectors(&d.base, None)?;
    let q = read_vectors(&d.queries, d.limit_queries)?;
    anyhow::ensure!(
        base.dim == q.dim,
        "dim mismatch: base {} queries {}",
        base.dim,
        q.dim
    );

    let eng = engine::Engine::connect("dump", &d.engine)?;
    if !d.skip_upload {
        eng.recreate_collection(&d.collection, base.dim as u64, m, base.n as u64)
            .await?;
        eng.upsert(&d.collection, &base.data, base.n, base.dim)
            .await?;
        eng.wait_green(&d.collection, 3600).await?;
    }

    // `exact: true` — see the doc on `DumpScores`.
    let res = eng
        .query(&d.collection, &q.data, q.n, q.dim, d.limit, true, None)
        .await?;

    let scores: Vec<f64> = res.iter().flat_map(|r| r.scores.iter().copied()).collect();
    std::fs::write(&d.out, serde_json::to_string(&scores)?)?;
    println!(
        "wrote {} scores from {} queries to {}",
        scores.len(),
        q.n,
        d.out.display()
    );
    Ok(())
}

/// §8.6 / W10's harness side — recall on one engine over an `ef` sweep.
#[derive(clap::Args)]
struct RelevanceRun {
    #[arg(long, default_value = "http://localhost:6334")]
    engine: String,
    #[arg(long, default_value = "strawmann")]
    label: String,
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    ground_truth: PathBuf,
    #[arg(long, default_value = "euclid")]
    metric: String,
    #[arg(long, default_value_t = 10)]
    limit: u64,
    /// `ef` values to sweep. `0` means "let the server choose".
    #[arg(long, num_args = 1.., default_values_t = [32u64, 64, 128, 256, 512])]
    ef: Vec<u64>,
    /// Cap base vectors, for a smaller run.
    #[arg(long)]
    limit_base: Option<usize>,
    /// Cap query count.
    #[arg(long)]
    limit_queries: Option<usize>,
    /// Skip create+upload; the collection is already loaded.
    #[arg(long, default_value_t = false)]
    skip_upload: bool,
    #[arg(long, default_value = "relevance")]
    collection: String,
    /// ε for §8.6's tie-aware recall.
    ///
    /// Defaults to the calibrated value for this `(metric, dim)` from
    /// `docs/tolerance.md` when one exists, else the §8.4 floor — the same
    /// rule `differ` uses. It defaulted to `1e-5`, ~24× the calibrated
    /// cosine/1536 ε, so the tie clause accepted returned points far
    /// outside any measured kernel noise as "correct".
    #[arg(long)]
    epsilon: Option<f64>,
    /// Control run: `exact: true`, bypassing the graph. Recall below 1.0
    /// here is a harness or tie problem, never an index one.
    #[arg(long, default_value_t = false)]
    exact: bool,
    /// §6.5 `m`, sent at collection creation.
    #[arg(long)]
    hnsw_m: Option<u64>,
    /// §6.5 `ef_construct`, sent at collection creation.
    #[arg(long)]
    hnsw_ef_construct: Option<u64>,
    /// `quantization.oversampling` on every query, as bfb's
    /// `--quantization-oversampling` sends it. W7 searches at 4 with
    /// rescore; a sweep of its collection that sent neither measured a
    /// different search, and the harness used to join it anyway.
    /// Requires `--quantization-rescore`, so the request states both.
    #[arg(long, requires = "quantization_rescore")]
    quantization_oversampling: Option<f64>,
    /// `quantization.rescore` on every query (bfb's `--quantization-rescore`).
    #[arg(long)]
    quantization_rescore: Option<bool>,
    /// Write the sweep as JSON, for the harness to join onto W10's
    /// latency rows.
    ///
    /// §4.1 splits the two measurements deliberately: bfb reports
    /// latency, this reports recall, and they meet on `ef`. Printing a
    /// table and nothing else made that join a human transcription step,
    /// which is how a qps number ends up published without the recall it
    /// was measured at, the thing §7.4 forbids.
    #[arg(long)]
    json: Option<PathBuf>,
    /// The payload field a filtered sweep conditions on (bfb's first keyword
    /// field is `a`). Only read when `--filter-values` is given.
    #[arg(long, default_value = "a")]
    filter_field: String,
    /// W12: search under `field in {values}`, the condition the ground truth
    /// must have been restricted to.
    ///
    /// A sweep speaks for a row only if it searched the way the row did, and
    /// for a filtered row that includes the filter: an unfiltered search
    /// scored against a restricted truth is not an approximation of the right
    /// answer, it is a different question. Both halves are refused separately
    /// below -- values without a restricted truth, and a restricted truth
    /// without values.
    #[arg(long, num_args = 1.., value_delimiter = ',')]
    filter_values: Vec<String>,
}

impl RelevanceRun {
    /// The `(ignore, rescore)` pair `Engine::query_quant_oversampled` takes,
    /// or `None` when the sweep sends no quantization params at all.
    fn quant(&self) -> Option<(bool, bool)> {
        self.quantization_rescore.map(|rescore| (false, rescore))
    }

    /// The condition this sweep searches under, spelled exactly the way
    /// `oracle::compute_filtered` spells it so the two can be compared.
    fn condition(&self) -> Option<String> {
        if self.filter_values.is_empty() {
            return None;
        }
        let mut v = self.filter_values.clone();
        v.sort();
        Some(format!("{} in {{{}}}", self.filter_field, v.join(",")))
    }

    /// The filter the search carries, or `None` for an unfiltered sweep.
    fn filter(&self) -> Option<qdrant_client::qdrant::Filter> {
        if self.filter_values.is_empty() {
            return None;
        }
        Some(qdrant_client::qdrant::Filter::must([
            qdrant_client::qdrant::Condition::matches(
                self.filter_field.clone(),
                self.filter_values.clone(),
            ),
        ]))
    }
}

/// One `ef` of the sweep, as the harness consumes it.
#[derive(serde::Serialize)]
struct RecallPoint {
    ef: Option<u64>,
    exact: bool,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    recall_at_10_ci95_low: f64,
    recall_at_10_ci95_high: f64,
    mean_relative_distance_error: f64,
    queries: usize,
    /// Queries the engine answered with fewer than `limit` results; their
    /// empty slots are misses.
    short_lists: usize,
    /// Returned points whose reported score beats the k-th truth without
    /// being in it: impossible for a correct engine, counted as misses.
    impossible_scores: usize,
    /// Returned ids that are not rows of the base file at all (§4.3). Printed
    /// since the counter existed and written here since 2026-09-01: a sweep
    /// run non-interactively is read from this file, and a counter that only
    /// reaches stdout is a counter no reader of the join can act on.
    unknown_ids: usize,
    /// A smoke-test rate from a batching client, kept so that a reader who
    /// sees it in the file knows it is not the benchmark's qps (§7.4).
    smoke_qps: f64,
    /// W12 point 4: how many points came back, and how many could have. A
    /// filtered page shorter than `k` is ambiguous between an index that
    /// searched too narrowly and a filter matching fewer than `k` points, and
    /// recall alone cannot separate them.
    returned: usize,
    asked: usize,
    /// How many points the condition matched. `null` on an unfiltered sweep,
    /// where it is the whole collection.
    n_matching: Option<usize>,
}

#[derive(serde::Serialize)]
struct RecallSweep {
    label: String,
    collection: String,
    metric: String,
    limit: u64,
    epsilon: f64,
    epsilon_relative: bool,
    /// "flag", "calibrated" or "floor" — where ε came from.
    epsilon_source: String,
    /// The filter condition every query carried, `null` for an unfiltered
    /// sweep. Part of the join key: a filtered recall describes a different
    /// question from an unfiltered one on the same collection and `ef`, so a
    /// reader that joined on (dataset, collection, ef) alone would mix them.
    condition: Option<String>,
    queries: usize,
    ground_truth: String,
    base_checksum: String,
    /// The quantization search parameters every query sent, part of the
    /// join key beside collection and dataset: `null` means none sent.
    quantization_oversampling: Option<f64>,
    quantization_rescore: Option<bool>,
    points: Vec<RecallPoint>,
}

#[tokio::main(flavor = "multi_thread")]
async fn run_relevance(r: RelevanceRun) -> anyhow::Result<()> {
    let m = parse_metric(&r.metric)?;
    let gt = oracle::load(&r.ground_truth)?;
    let b = read_vectors(&r.base, r.limit_base)?;
    // The whole query file, so its checksum can be compared against
    // the ground truth's; the cap is applied afterwards.
    let q_full = read_vectors(&r.queries, None)?;

    // §4.3: ground truth is keyed to the exact r.base and query files it
    // was computed from. A mismatched pair produces recall numbers that
    // are wrong in a way nothing downstream can detect, so this is a
    // hard failure rather than a warning.
    anyhow::ensure!(
        gt.base_checksum == oracle::checksum_f32(&b.data),
        "ground truth was computed for a different r.base set (checksum {:x} vs {:x})",
        gt.base_checksum,
        oracle::checksum_f32(&b.data),
    );
    // The query file too. `run_differ` checked both; this checked one,
    // so a ground truth for another query set was accepted here and
    // every recall number that followed was against the wrong answers.
    anyhow::ensure!(
        gt.query_checksum == oracle::checksum_f32(&q_full.data),
        "ground truth was computed for a different query set (checksum {:x} vs {:x}) (§4.3)",
        gt.query_checksum,
        oracle::checksum_f32(&q_full.data),
    );
    // And for this metric. A euclid ground truth judging a cosine
    // collection is not approximately right, it is a different question.
    anyhow::ensure!(
        gt.metric == m,
        "ground truth was computed for {} but --metric is {} (§4.3)",
        gt.metric.as_str(),
        m.as_str()
    );
    // Under the same condition, in both directions. Either mismatch produces a
    // number that looks like an engine result and is not one: an unfiltered
    // search judged against a restricted truth misses every neighbour the
    // filter removed, and a filtered search judged against §4.3's cache misses
    // every neighbour the filter kept out of the top k.
    // docs/workloads.md W12 point 3.
    let condition = r.condition();
    relevance::validate_condition(
        condition.as_deref(),
        gt.condition.as_deref(),
        r.quant().is_some(),
    )
    .map_err(|e| anyhow::anyhow!("{e} (ground truth: {})", r.ground_truth.display()))?;
    let filter = r.filter();
    let q = match r.limit_queries {
        Some(cap) if cap < q_full.n => datasets::Vectors {
            n: cap,
            dim: q_full.dim,
            data: q_full.data[..cap * q_full.dim].to_vec(),
        },
        _ => q_full,
    };
    anyhow::ensure!(
        gt.dim == b.dim && b.dim == q.dim,
        "dimension mismatch: gt={} r.base={} r.queries={}",
        gt.dim,
        b.dim,
        q.dim
    );

    // §4.3's hard requirement, checked rather than trusted.
    //
    // "Ground truth indexes into base-file row order, so point ID must
    // equal row index... Any run violating this is rejected by the
    // harness rather than silently producing meaningless recall." This
    // guard existed with that sentence as its doc comment and was
    // called from nowhere, so nothing was rejected and the meaningless
    // recall it describes would have been produced and published.
    //
    // The conformance harness controls its own ingest, so it always
    // uploads at offset 0 with numeric ids; the value of asserting it
    // is that `--skip-upload` hands the collection over to whatever
    // filled it, and `recall.py` uses exactly that path.
    relevance::validate_relevance_run(&relevance::RunConfig {
        offset: 0,
        uuids: false,
        max_id: None,
    })
    .map_err(anyhow::Error::msg)?;

    let eng = engine::Engine::connect(&r.label, &r.engine)?;
    let n_q = q.n.min(gt.n_queries);

    // §4.3: the ground truth indexes into base-file row order, so it
    // describes a corpus of exactly `gt.n_base` points. A collection
    // holding a different number is a different corpus, and recall
    // against it is meaningless rather than merely approximate.
    //
    // This is not hypothetical. A smoke run uploaded the first 50,000
    // SIFT vectors and swept against the 1M ground truth: almost every
    // query's true neighbour was not in the collection at all, and the
    // harness reported recall@10 = 0.1132, flat across every `ef`, with
    // MRDE 0.996. Plausible-looking numbers for a collection that
    // simply did not contain the answers. §4.3 is explicit that such a
    // run is "rejected by the harness rather than silently producing
    // meaningless recall".
    if r.skip_upload {
        let have = eng.count(&r.collection).await?;
        if have != gt.n_base as u64 {
            anyhow::bail!(
                "collection {} holds {} points but the ground truth describes {} \
(§4.3: ground truth indexes into base-file row order). Recall against a different \
corpus is meaningless, not approximate. Rebuild the ground truth for this size with \
`oracle --limit {}`, or point --collection at the full corpus.",
                r.collection,
                have,
                gt.n_base,
                have
            );
        }
    }

    if !r.skip_upload {
        eprintln!(
            "creating {} ({} x {}, {})",
            r.collection, b.n, b.dim, r.metric
        );
        eng.recreate_collection_hnsw(
            &r.collection,
            b.dim as u64,
            m,
            r.hnsw_m,
            r.hnsw_ef_construct,
            b.n as u64,
        )
        .await?;
        let t = std::time::Instant::now();
        eng.upsert(&r.collection, &b.data, b.n, b.dim).await?;
        eprintln!(
            "uploaded {} points in {:.1}s",
            b.n,
            t.elapsed().as_secs_f64()
        );
        let t = std::time::Instant::now();
        eng.wait_green(&r.collection, 3600).await?;
        eprintln!("green in {:.1}s", t.elapsed().as_secs_f64());
    }

    // §4.3's tie clause needs to know whether `--epsilon` is a ratio or
    // a distance, which is a property of the metric (§8.4): cosine
    // scores live in [-1, 1] so an absolute bound is meaningful, while
    // dot, euclid and manhattan scale with vector magnitude. Passing the
    // bare f64 through made it absolute for all four, so on euclid the
    // tie clause never fired.
    let (eps, eps_source) = {
        let (mut e, src) = differ::tolerance::default_epsilon(m, b.dim);
        match r.epsilon {
            Some(v) => {
                e.value = v;
                (e, differ::tolerance::EpsilonSource::Flag)
            }
            None => (e, src),
        }
    };
    eprintln!(
        "tie tolerance: {} [{}]",
        eps.describe(),
        match eps_source {
            differ::tolerance::EpsilonSource::Flag => "from --epsilon",
            differ::tolerance::EpsilonSource::Calibrated => "calibrated, docs/tolerance.md",
            differ::tolerance::EpsilonSource::Floor =>
                "§8.4 floor: no calibrated cell for this (metric, dim); run `calibrate`",
        }
    );
    // §4.3's tie clause judged on the oracle's own fp64 score of each
    // returned point, not on the score the engine reported for it.
    let rescorer = oracle::Rescorer::new(m, &b.data, b.n, &q.data, n_q, b.dim);

    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  recall@10 CI95",
        "ef", "recall@1", "recall@10", "recall@100", "mrde"
    );
    let mut points: Vec<RecallPoint> = Vec::new();
    for &e in &r.ef {
        let hnsw_ef = if e == 0 { None } else { Some(e) };
        let t = std::time::Instant::now();
        let res = match r.quant() {
            None if filter.is_some() => {
                eng.query_filtered(
                    &r.collection,
                    &q.data,
                    n_q,
                    q.dim,
                    r.limit,
                    r.exact,
                    hnsw_ef,
                    filter.as_ref().expect("this arm is guarded on it"),
                )
                .await?
            }
            None => {
                eng.query(
                    &r.collection,
                    &q.data,
                    n_q,
                    q.dim,
                    r.limit,
                    r.exact,
                    hnsw_ef,
                )
                .await?
            }
            Some(quant) => {
                eng.query_quant_oversampled(
                    &r.collection,
                    &q.data,
                    n_q,
                    q.dim,
                    r.limit,
                    r.exact,
                    hnsw_ef,
                    Some(quant),
                    r.quantization_oversampling,
                )
                .await?
            }
        };
        let elapsed = t.elapsed().as_secs_f64();
        let g = relevance::evaluate(&gt, &res, &eps, r.limit as usize, Some(&rescorer));
        println!(
            "{:>6}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.5}  [{:.4}, {:.4}]   {:.1} q/s{}{}{}",
            if r.exact {
                "exact".to_string()
            } else {
                e.to_string()
            },
            g.recall_at_1,
            g.recall_at_10,
            g.recall_at_100,
            g.mean_relative_distance_error,
            g.recall_at_10_ci95.low,
            g.recall_at_10_ci95.high,
            n_q as f64 / elapsed,
            if g.short_lists > 0 {
                format!(
                    "  short_lists={} (fewer than {} results; empty slots are misses)",
                    g.short_lists, r.limit
                )
            } else {
                String::new()
            },
            if g.impossible_scores > 0 {
                format!(
                    "  IMPOSSIBLE_SCORES={} (non-neighbours reported better than the k-th truth)",
                    g.impossible_scores
                )
            } else {
                String::new()
            },
            if g.unknown_ids > 0 {
                format!(
                    "  UNKNOWN_IDS={} (ids that are not base rows, §4.3)",
                    g.unknown_ids
                )
            } else {
                String::new()
            },
        );
        points.push(RecallPoint {
            ef: if r.exact || e == 0 { None } else { Some(e) },
            exact: r.exact,
            recall_at_1: g.recall_at_1,
            recall_at_10: g.recall_at_10,
            recall_at_100: g.recall_at_100,
            recall_at_10_ci95_low: g.recall_at_10_ci95.low,
            recall_at_10_ci95_high: g.recall_at_10_ci95.high,
            mean_relative_distance_error: g.mean_relative_distance_error,
            // The count the recalls were averaged over, not the count
            // asked for: an engine answering 50 of 100 queries had its
            // recall averaged over 50 while the file said 100.
            queries: g.queries,
            short_lists: g.short_lists,
            impossible_scores: g.impossible_scores,
            unknown_ids: g.unknown_ids,
            smoke_qps: g.queries as f64 / elapsed,
            returned: g.returned,
            asked: g.asked,
            n_matching: g.n_matching,
        });
    }

    if let Some(path) = &r.json {
        let sweep = RecallSweep {
            label: r.label.clone(),
            collection: r.collection.clone(),
            metric: r.metric.clone(),
            limit: r.limit,
            epsilon: eps.value,
            epsilon_relative: eps.relative,
            epsilon_source: format!("{eps_source:?}").to_lowercase(),
            condition: condition.clone(),
            queries: n_q,
            ground_truth: r.ground_truth.display().to_string(),
            // §4.3: the corpus this recall was measured against, so a
            // joined table cannot pair recall from one base file with
            // latency from another.
            base_checksum: format!("{:x}", gt.base_checksum),
            quantization_oversampling: r.quantization_oversampling,
            quantization_rescore: r.quantization_rescore,
            points,
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&sweep)?)?;
        eprintln!("wrote {}", path.display());
    }
    println!();
    println!("§7.4: these q/s figures are a smoke-test rate from a batching client,");
    println!("not a W3/W4 latency measurement. Use bfb for load; this reports recall.");
    Ok(())
}

/// The corpus name for a conformance row: what the caller said, or the base
/// file's stem when it said nothing.
///
/// The stem alone was the rule until a dataset arrived whose fbin is not named
/// after it. §4.3 binds a conformance row to a corpus, and `base.fbin` names
/// two different ones — `dbpedia-openai-1m` and `dbpedia-openai-100K-1536-angular`
/// both convert to it — so a row identified by stem could not tell them apart.
/// The checksums still could, which is why this was wrong without being unsafe.
///
/// The fallback stays, because `differ` is runnable by hand and a name inferred
/// from the file is better than an empty one.
fn dataset_name(explicit: &str, base_path: &Path) -> String {
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    base_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| base_path.display().to_string())
}

#[cfg(test)]
mod dataset_name_tests {
    use super::*;

    #[test]
    fn an_explicit_name_wins_over_the_filename() {
        let p = Path::new("/d/dbpedia-openai-100K-1536-angular/base.fbin");
        assert_eq!(
            dataset_name("dbpedia-openai-100K-1536-angular", p),
            "dbpedia-openai-100K-1536-angular"
        );
    }

    #[test]
    fn the_stem_is_the_fallback_and_is_why_this_exists() {
        let p = Path::new("/d/dbpedia-openai-100K-1536-angular/base.fbin");
        assert_eq!(
            dataset_name("", p),
            "base",
            "the old behaviour, kept for hand runs"
        );
        // SIFT1M is the case where the two agree, which is why the stem rule
        // survived as long as it did.
        assert_eq!(
            dataset_name("", Path::new("/d/sift1m/sift1m.fbin")),
            "sift1m"
        );
    }
}

/// §8.5 — drive both engines through identical operation sequences.
#[tokio::main(flavor = "multi_thread")]
async fn run_differ(a: DifferRun) -> anyhow::Result<()> {
    // Destructured with the names the body already uses, so the body is
    // untouched: the point of the struct is that the *call* cannot transpose
    // two of eleven arguments, not that this function reads differently.
    let DifferRun {
        strawmann: strawmann_url,
        qdrant: qdrant_url,
        base: base_path,
        queries: query_path,
        ground_truth: gt_path,
        metric: metric_name,
        dataset,
        limit,
        max_tier,
        epsilon: epsilon_override,
        json: json_out,
    } = a;
    let (strawmann_url, qdrant_url) = (strawmann_url.as_str(), qdrant_url.as_str());
    let (base_path, query_path, gt_path) =
        (base_path.as_path(), query_path.as_path(), gt_path.as_path());
    let (metric_name, max_tier) = (metric_name.as_str(), max_tier.as_str());
    let dataset = dataset.as_str();
    let json_out = json_out.as_deref();
    let m = parse_metric(metric_name)?;
    let max_tier = parse_max_tier(max_tier)?;
    let base = read_vectors(base_path, None)?;
    let queries = read_vectors(query_path, None)?;
    let gt = oracle::load(gt_path)?;

    // §4.3: "Checksums of base, query, and GT files are recorded in every
    // result row." Checking them here means a stale ground truth is caught
    // before it can quietly produce meaningless recall.
    anyhow::ensure!(
        gt.base_checksum == oracle::checksum_f32(&base.data),
        "ground truth was computed for a different base file (§4.3)"
    );
    anyhow::ensure!(
        gt.query_checksum == oracle::checksum_f32(&queries.data),
        "ground truth was computed for a different query file (§4.3)"
    );
    anyhow::ensure!(
        gt.metric == m,
        "ground truth was computed for {} but --metric is {} (§4.3)",
        gt.metric.as_str(),
        m.as_str()
    );

    let engines = [
        engine::Engine::connect("strawmann", strawmann_url)?,
        engine::Engine::connect("qdrant", qdrant_url)?,
    ];

    let collection = "conformance";
    for e in &engines {
        eprintln!("[{}] recreating collection", e.label);
        e.recreate_collection(collection, base.dim as u64, m, base.n as u64)
            .await?;
        eprintln!("[{}] upserting {} points", e.label, base.n);
        e.upsert(collection, &base.data, base.n, base.dim).await?;
        e.wait_green(collection, 600).await?;
        // The pin, read back. T3 compares recall at a nominal `ef`, and an
        // engine searching several graphs at that `ef` reports a higher one
        // without being more accurate -- which is how the
        // dbpedia-openai-1m run failed T3 and lost its comparative licence.
        // Said, not asserted: an empty appendable segment counts here too, so
        // this is an upper bound on the graphs rather than a verdict.
        match e.segments_count(collection).await {
            Ok(1) => {}
            Ok(n) => eprintln!(
                "[{}] WARNING: {} segments after load, not 1; T3 compares recall \
                 at a nominal ef and more graphs raise it without more accuracy. \
                 One may be an empty appendable, so this is an upper bound.",
                e.label, n
            ),
            Err(err) => eprintln!("[{}] could not read segment count back: {err}", e.label),
        }
    }

    let mut tiers: Vec<differ::TierResult> = Vec::new();

    // --- T0: wire conformance ---
    {
        let a = engines[0]
            .wire_shape(collection, &queries.data[..queries.dim], limit as u64)
            .await?;
        let b = engines[1]
            .wire_shape(collection, &queries.data[..queries.dim], limit as u64)
            .await?;
        let r = differ::compare_shape(&a, &b);
        println!(
            "{}: {} - {}",
            r.tier.as_str(),
            if r.passed { "PASS" } else { "FAIL" },
            r.detail
        );
        tiers.push(r);
    }

    // --- T1: exact-search value equality ---
    // §8.5: "**This is the tier that licenses the performance claim.**"
    // Only when T1 is asked for: `--max-tier T0` used to run T1 anyway and
    // emit a row whose tier set (T0+T1, no metamorphic) was a combination
    // nothing else produces, bound into the hash as a distinct identity.
    let (ours_exact, theirs_exact) = if max_tier >= 1 {
        (
            engines[0]
                .query(
                    collection,
                    &queries.data,
                    queries.n,
                    queries.dim,
                    limit as u64,
                    true,
                    None,
                )
                .await?,
            engines[1]
                .query(
                    collection,
                    &queries.data,
                    queries.n,
                    queries.dim,
                    limit as u64,
                    true,
                    None,
                )
                .await?,
        )
    } else {
        (Vec::new(), Vec::new())
    };

    println!("{}", differ::claims_banner());
    println!();

    // §8.4: ε is derived, not chosen. The calibrated cell from
    // `docs/tolerance.md` when there is one for this (metric, dim); else the
    // floor from the expected summation error, and the run says so rather
    // than a round number being invented.
    let (mut eps, eps_source) = differ::tolerance::default_epsilon(m, base.dim);
    if let Some(v) = epsilon_override {
        eps.value = v;
        eprintln!("using epsilon from --epsilon: {v:.3e} (§8.4)");
    } else if eps_source == differ::tolerance::EpsilonSource::Calibrated {
        eprintln!(
            "using the calibrated epsilon from docs/tolerance.md: {} (§8.4)",
            eps.describe()
        );
    } else {
        eprintln!(
            "NOTE: using the §8.4 floor for epsilon ({}); run `calibrate` with per-ISA score dumps to derive it properly",
            eps.describe()
        );
    }
    // §4.3's tie clause judged on the oracle's own fp64 score for each
    // returned point (T3 and T4 go through `relevance::evaluate`).
    let rescorer = oracle::Rescorer::new(m, &base.data, base.n, &queries.data, queries.n, base.dim);

    if max_tier >= 1 {
        let r = differ::compare_exact(&ours_exact, &theirs_exact, &gt, &eps, limit);
        println!(
            "{}: {} - {}",
            r.tier.as_str(),
            if r.passed { "PASS" } else { "FAIL" },
            r.detail
        );
        tiers.push(r);
    }

    if max_tier >= 2 {
        // --- T2: rank agreement under ties ---
        let r = differ::compare_ranks(&ours_exact, &theirs_exact, &eps);
        println!(
            "{}: {} - {}",
            r.tier.as_str(),
            if r.passed { "PASS" } else { "FAIL" },
            r.detail
        );
        tiers.push(r);
    }

    if max_tier >= 3 {
        // --- T3: ANN statistical equivalence ---
        let ours_ann = engines[0]
            .query(
                collection,
                &queries.data,
                queries.n,
                queries.dim,
                limit as u64,
                false,
                Some(128),
            )
            .await?;
        let theirs_ann = engines[1]
            .query(
                collection,
                &queries.data,
                queries.n,
                queries.dim,
                limit as u64,
                false,
                Some(128),
            )
            .await?;
        let (r, _a, _b) =
            differ::compare_ann(&ours_ann, &theirs_ann, &gt, &eps, limit, Some(&rescorer));
        println!(
            "{}: {} - {}",
            r.tier.as_str(),
            if r.passed { "PASS" } else { "FAIL" },
            r.detail
        );
        tiers.push(r);
    }

    // §8.9: "metamorphic + T0 + T1 on a 100k slice per commit" — so the
    // properties run whenever T1 does, and their verdict enters the row.
    // After T3 rather than before it: the idempotent-upsert check re-sends
    // points, and re-optimising the collection under T3's ANN queries would
    // measure a different index than the one that went green.
    if max_tier >= 1 {
        let r = run_metamorphic(&engines, collection, &base, &queries, limit, m, &eps).await?;
        println!(
            "{}: {} - {}",
            r.tier.as_str(),
            if r.passed { "PASS" } else { "FAIL" },
            r.detail
        );
        tiers.push(r);
    }

    // --- T4: quantization fidelity ---
    //
    // §8.5 calls this "the tier that catches the most seductive class of bug".
    // It was implemented and unit-tested but never invoked from a run, so
    // `--max-tier T4` silently topped out at T3 and the whole §6.7 quantization
    // axis went unchecked against Qdrant.
    //
    // It needs its own collection: the same points with SQ8 configured, so the
    // *only* difference between the two score sets is the encoding.
    if max_tier >= 4 {
        let qcoll = format!("{collection}_sq8");
        for e in &engines {
            e.recreate_collection_quantized(&qcoll, base.dim as u64, m, base.n as u64)
                .await?;
            e.upsert(&qcoll, &base.data, base.n, base.dim).await?;
            e.wait_green(&qcoll, 3600).await?;
        }
        for (i, e) in engines.iter().enumerate() {
            // Quantized scores, then the same queries forced onto the fp32 path
            // via `ignore`. Rescore stays off: with it on we would measure the
            // rescorer, not the encoding.
            let q = e
                .query_quant(
                    &qcoll,
                    &queries.data,
                    queries.n,
                    queries.dim,
                    limit as u64,
                    false,
                    Some(128),
                    Some((false, false)),
                )
                .await?;
            let f = e
                .query_quant(
                    &qcoll,
                    &queries.data,
                    queries.n,
                    queries.dim,
                    limit as u64,
                    false,
                    Some(128),
                    Some((true, false)),
                )
                .await?;
            let label = if i == 0 { "strawmann" } else { "qdrant" };

            // §8.6: "Quantization dominance: with `rescore = true` and
            // oversampling `n`, recall is monotonically non-decreasing in `n`."
            //
            // This is the property that catches a quantizer sold as a speedup
            // because it searches less, and it needs a *sweep* rather than a
            // single point, which is why it sits here beside T4 rather than in
            // `run_metamorphic`. README finding 6 records the case it is
            // watching for: oversampling without rescore is inert, and recall@10
            // stayed flat from 1x to 16x.
            let mut curve: Vec<(f64, f64)> = Vec::new();
            for ov in [1.0f64, 2.0, 4.0, 8.0] {
                let r = e
                    .query_quant_oversampled(
                        &qcoll,
                        &queries.data,
                        queries.n,
                        queries.dim,
                        limit as u64,
                        false,
                        Some(128),
                        Some((false, true)),
                        Some(ov),
                    )
                    .await?;
                curve.push((
                    ov,
                    relevance::evaluate(&gt, &r, &eps, limit, Some(&rescorer)).recall_at_10,
                ));
            }
            let p = metamorphic::quantization_dominance(&curve);
            println!(
                "  §8.6 quantization_dominance/{label}: {} — recall@10 {}",
                if p.held { "holds" } else { "VIOLATED" },
                curve
                    .iter()
                    .map(|(o, r)| format!("{o:.0}x={r:.4}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            if !p.held {
                println!("    {}", p.detail);
            }

            // The tier's verdict is the fidelity numbers *and* the dominance
            // property together; see `quantization_fidelity`.
            let mut r = differ::quantization_fidelity(&q, &f, &gt, &format!("sq8/{label}"), &p);
            // Qdrant's fidelity is a finding about Qdrant, reported in the
            // row; strawmann's is the tier of the row under test
            // (`TierResult::advisory`).
            r.advisory = i != 0;
            println!(
                "{}: {} - {}",
                r.tier.as_str(),
                if r.passed { "PASS" } else { "FAIL" },
                r.detail
            );
            tiers.push(r);
        }
    }

    // The scratch collections are this run's and stay resident otherwise:
    // two full-capacity collections per engine at the headline tier, on the
    // host decisions.md item 7 records being OOM-killed.
    for e in &engines {
        e.delete_collection(collection).await;
        if max_tier >= 4 {
            e.delete_collection(&format!("{collection}_sq8")).await;
        }
    }

    // The build identity §8 binds the row to. These default to "unknown", and
    // for a long time nothing set them — so the gate bound the dataset, the
    // checksums and the outcomes, but not the build under test, which is the
    // one thing "the same build, on the same data" names first. `fullrun.py`
    // sets all three; a run that does not is warned about rather than left to
    // produce a confidently anonymous row.
    let qdrant_version =
        std::env::var("QDRANT_VERSION").unwrap_or_else(|_| differ::QDRANT_UNSET.into());
    let strawmann_commit =
        std::env::var("STRAWMANN_COMMIT").unwrap_or_else(|_| differ::COMMIT_UNSET.into());
    let isa_build =
        std::env::var("STRAWMANN_ISA_BUILD").unwrap_or_else(|_| differ::ISA_UNSET.into());
    // The same rule the licence bit uses, so the warning and the bit cannot
    // disagree. It used to test two of the three variables it asks for.
    if !differ::identity_known(&strawmann_commit, &qdrant_version, &isa_build) {
        eprintln!(
            "WARNING: this row identifies its builds as strawmann={strawmann_commit} \
qdrant={qdrant_version} isa={isa_build}. §8 binds a perf row to 'the same build'; set STRAWMANN_COMMIT, \
STRAWMANN_ISA_BUILD and QDRANT_VERSION (fullrun.py does). The hash is still printed, but \
the row licenses no claim (`ConformanceRow::build_identity_known`)."
        );
    }

    let row = differ::ConformanceRow {
        // The name, not the path: the hash is an identity that has to survive
        // being computed on another machine.
        dataset: dataset_name(dataset, base_path),
        metric: m,
        dim: base.dim,
        qdrant_version,
        strawmann_commit,
        isa_build,
        tiers,
        base_checksum: gt.base_checksum,
        query_checksum: gt.query_checksum,
        epsilon: eps.value,
        epsilon_relative: eps.relative,
        limit,
    };

    if let Some(path) = json_out {
        // The shape `results.py` records. Written from the row that was just
        // built rather than restated, so the hash in the file is the hash the
        // gate will check against.
        let t1 = row
            .tiers
            .iter()
            .find(|t| matches!(t.tier, differ::Tier::T1ExactValue));
        let out = serde_json::json!({
            "hash": format!("{:016x}", row.hash()),
            // The *name*, not the path it happened to be read from: a row that
            // records `/home/someone/.cache/...` cannot be matched against a
            // perf row measured on another machine, and publishes a home
            // directory while failing to identify the corpus.
            "dataset": row.dataset,
            "dataset_path": base_path.display().to_string(),
            "metric": format!("{:?}", row.metric),
            "dim": row.dim,
            "qdrant_version": row.qdrant_version,
            "strawmann_commit": row.strawmann_commit,
            "isa_build": row.isa_build,
            "tier_reached": row.tier_reached().map(|t| t.as_str().to_string()),
            "licenses_perf": row.licenses_performance_claim(),
            "licenses_comparative": row.licenses_comparative_claim(),
            "epsilon": row.epsilon,
            "epsilon_relative": row.epsilon_relative,
            "epsilon_source": if epsilon_override.is_some() { "flag".to_string() } else { format!("{eps_source:?}").to_lowercase() },
            "limit": row.limit,
            "base_checksum": format!("{:016x}", row.base_checksum),
            "query_checksum": format!("{:016x}", row.query_checksum),
            "detail": t1.map(|t| t.detail.clone()).unwrap_or_default(),
            // §8.9 names `max|Δscore|` and `p99|Δscore|` as row fields, and
            // §8.4 says the point of measuring the distribution rather than a
            // pass/fail bit is that "a run that passes with a max delta 10x
            // worse than yesterday's is a regression". Both were computed,
            // rendered into T1's `detail` sentence by `Distribution::describe`,
            // and then dropped: `results.py` reads these two keys straight into
            // the columns that hold them and found nothing, so the sink's
            // `max_delta` and `p99_delta` were null for every row ever
            // ingested. A number that only exists inside a formatted string is
            // a number nothing downstream can compare against yesterday's.
            "max_delta": t1.and_then(|t| t.distribution.as_ref()).map(|d| d.max),
            "p99_delta": t1.and_then(|t| t.distribution.as_ref()).map(|d| d.p99),
            "tiers": row.tiers.iter().map(|t| serde_json::json!({
                "tier": t.tier.as_str(),
                "passed": t.passed,
                "detail": t.detail,
                // The whole distribution, so the relative pair §8.4 wants for
                // the magnitude-scaling metrics is there beside the absolute
                // one rather than only in the prose.
                "distribution": t.distribution,
            })).collect::<Vec<_>>(),
        });
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&out)?)?;
        eprintln!("wrote {}", path.display());
    }

    println!();
    println!("conformance tier reached: {:?}", row.tier_reached());
    println!("conformance hash: {:016x}", row.hash());
    if row.licenses_performance_claim() {
        println!("§8: LICENSES single-engine performance rows (kernels, cost model, ISA");
        println!("     matrix). They must carry the hash above.");
    } else if !row.build_identity_known() {
        println!("§8: does NOT license any performance number - the build under test is");
        println!(
            "     not identified (strawmann={}, qdrant={}).",
            row.strawmann_commit, row.qdrant_version
        );
    } else {
        println!("§8: does NOT license any performance number - T1 did not pass.");
    }
    if row.licenses_comparative_claim() {
        println!("§7.4: LICENSES a strawmann-vs-Qdrant throughput or latency comparison:");
        println!("      T3 passed, so the two engines are at equal recall.");
    } else {
        println!("§7.4: does NOT license a strawmann-vs-Qdrant throughput or latency");
        println!("      comparison - T3 did not pass, so the engines are at unequal recall");
        println!("      and the faster one may simply be searching less.");
    }
    Ok(())
}

/// §8.6: "Each is asserted against strawmann alone, and the interesting ones
/// against Qdrant too - a property that holds for one engine and not the other
/// is a finding."
/// §8.6's properties, run against both engines.
///
/// Six of the seven §8.6 lists; `filter_subsetting` is phase 3 and needs a
/// payload index neither engine is being asked for here.
///
/// Five of these were implemented, unit-tested, and invoked by nothing. Only
/// `prefix_property` and `offset_consistency` ever ran, while §8.9 described CI
/// as running "metamorphic + T0 + T1 on a 100k slice per commit". That includes
/// `quantization_dominance`, which is the guard against the exact failure T4
/// exists for — a cheaper quantizer that is faster because it is less accurate.
/// This is the third time in this repository that a check was written and never
/// called; the other two are recorded in `differ::quantization_fidelity` and in
/// `results.py`'s sink.
///
/// Returns the verdict as a tier so it enters the row and the hash. `passed`
/// is strawmann's side: a property Qdrant fails is a finding, in the detail.
async fn run_metamorphic(
    engines: &[engine::Engine; 2],
    collection: &str,
    base: &datasets::Vectors,
    queries: &datasets::Vectors,
    limit: usize,
    m: oracle::Metric,
    eps: &differ::tolerance::Epsilon,
) -> anyhow::Result<differ::TierResult> {
    println!();
    println!("§8.6 metamorphic properties:");

    let nq = queries.n.min(64);
    let mut results: Vec<Vec<metamorphic::PropertyResult>> = Vec::new();

    // §8.6: "Permutation invariance: insertion order must not change
    // exact-search results." A slice of the base, inserted forwards into one
    // collection and backwards into another, then queried exactly. Two small
    // collections rather than a re-insert of the whole corpus: the property
    // is about order, and 10k points exercise it as well as 1M would.
    let n_perm = base.n.min(10_000);
    let perm_fwd = format!("{collection}_perm_fwd");
    let perm_rev = format!("{collection}_perm_rev");
    let fwd: Vec<u32> = (0..n_perm as u32).collect();
    let rev: Vec<u32> = (0..n_perm as u32).rev().collect();

    for e in engines {
        let mut props = Vec::new();
        let small = e
            .query(
                collection,
                &queries.data,
                nq,
                queries.dim,
                limit as u64,
                true,
                None,
            )
            .await?;
        // limit + 5 at offset 0, and 5 at offset `limit`: §8.6's
        // `query(limit=L, offset=O) == query(limit=L+O)[O..]` with L=5, O=limit.
        let large = e
            .query(
                collection,
                &queries.data,
                nq,
                queries.dim,
                (limit + 5) as u64,
                true,
                None,
            )
            .await?;
        // §8.6: "Prefix property: top_k is a prefix of top_(k+1)."
        props.push(metamorphic::prefix_property(&small, &large, eps));

        // §8.6: "Offset consistency". A request that carries the offset,
        // compared against the tail of the longer offset-free list. This was
        // `offset_consistency(&small, &large, 0)`: two offset-free lists at
        // offset zero, i.e. `x == x`.
        let offset = e
            .query_offset(
                collection,
                &queries.data,
                nq,
                queries.dim,
                5,
                limit as u64,
                true,
            )
            .await?;
        props.push(metamorphic::offset_consistency(&offset, &large, limit, eps));

        // §8.6: "`score(x, x)` is maximal, and exact top-1 for an indexed
        // vector as query is that vector itself." Queried with the *base*
        // vectors, whose row index is their id (§4.3), so the expected answer
        // is known without consulting either engine.
        let nb = base.n.min(64);
        let self_q = e
            .query(collection, &base.data, nb, base.dim, 1, true, None)
            .await?;
        let expected: Vec<u32> = (0..nb as u32).collect();
        // A duplicate row is at distance zero from the query too, and §8.7's
        // tie-break returns the lower id: that is a hit, not a miss.
        let row = |id: u32| -> Option<&[f32]> {
            let (id, d) = (id as usize, base.dim);
            base.data.get(id * d..(id + 1) * d)
        };
        let same_vector = |a: u32, b: u32| row(a).is_some() && row(a) == row(b);
        props.push(metamorphic::self_retrieval(
            &self_q,
            &expected,
            true,
            &same_vector,
        ));

        // §8.6's permutation invariance proper.
        e.recreate_collection(&perm_fwd, base.dim as u64, m, base.n as u64)
            .await?;
        e.upsert_in_order(&perm_fwd, &base.data, base.dim, &fwd)
            .await?;
        e.wait_green(&perm_fwd, 600).await?;
        e.recreate_collection(&perm_rev, base.dim as u64, m, base.n as u64)
            .await?;
        e.upsert_in_order(&perm_rev, &base.data, base.dim, &rev)
            .await?;
        e.wait_green(&perm_rev, 600).await?;
        let from_fwd = e
            .query(
                &perm_fwd,
                &queries.data,
                nq,
                queries.dim,
                limit as u64,
                true,
                None,
            )
            .await?;
        let from_rev = e
            .query(
                &perm_rev,
                &queries.data,
                nq,
                queries.dim,
                limit as u64,
                true,
                None,
            )
            .await?;
        props.push(metamorphic::permutation_invariance(
            &from_fwd, &from_rev, eps,
        ));
        e.delete_collection(&perm_fwd).await;
        e.delete_collection(&perm_rev).await;

        // The weaker cousin, under its own name: the same queries in reverse
        // batch order, compared position-for-position after reversing back.
        // Exact search over a fixed collection cannot depend on the order the
        // questions were asked in. This used to carry the name
        // "permutation_invariance".
        let mut reversed: Vec<f32> = Vec::with_capacity(nq * queries.dim);
        for qi in (0..nq).rev() {
            reversed.extend_from_slice(&queries.data[qi * queries.dim..(qi + 1) * queries.dim]);
        }
        let mut rev_res = e
            .query(
                collection,
                &reversed,
                nq,
                queries.dim,
                limit as u64,
                true,
                None,
            )
            .await?;
        rev_res.reverse();
        props.push(metamorphic::query_order_invariance(&small, &rev_res, eps));

        // §8.6: "Idempotent upsert: re-upserting identical points changes
        // nothing observable." Re-sends the first slice of the base data,
        // which is already in the collection at the same ids.
        let n_re = base.n.min(1000);
        let before = e.count(collection).await?;
        e.upsert(collection, &base.data[..n_re * base.dim], n_re, base.dim)
            .await?;
        let after = e.count(collection).await?;
        let requeried = e
            .query(
                collection,
                &queries.data,
                nq,
                queries.dim,
                limit as u64,
                true,
                None,
            )
            .await?;
        props.push(metamorphic::idempotent_upsert(
            &small, &requeried, before, after, eps,
        ));

        results.push(props);
    }

    let mut lines: Vec<String> = Vec::new();
    let mut strawmann_failed: Vec<String> = Vec::new();
    for (i, p) in results[0].iter().enumerate() {
        let q = &results[1][i];
        let c = metamorphic::Comparison {
            property: p.name,
            strawmann: p.held,
            qdrant: q.held,
        };
        println!("  {}", c.describe());
        if !p.held {
            println!("    strawmann: {}", p.detail);
            strawmann_failed.push(format!("{}: {}", p.name, p.detail));
        }
        if !q.held {
            println!("    qdrant:    {}", q.detail);
        }
        lines.push(format!(
            "{}={}/{}",
            p.name,
            if p.held { "ok" } else { "FAIL" },
            if q.held { "ok" } else { "FAIL" }
        ));
    }
    Ok(differ::TierResult {
        advisory: false,
        tier: differ::Tier::Metamorphic,
        passed: strawmann_failed.is_empty(),
        detail: format!(
            "{} properties (strawmann/qdrant): {}{}",
            results[0].len(),
            lines.join(", "),
            if strawmann_failed.is_empty() {
                String::new()
            } else {
                format!(" — strawmann violated: {}", strawmann_failed.join("; "))
            }
        ),
        distribution: None,
    })
}

/// §8.4 steps 1 and 2: measure an engine's disagreement with *itself* across
/// ISA arms.
///
/// Each file holds one arm's scores for the same query set. The spread is the
/// max pairwise delta per position, which is what §8.4 means by "the
/// score-delta distribution".
fn spread_across_arms(paths: &[PathBuf]) -> anyhow::Result<differ::tolerance::Distribution> {
    if paths.len() < 2 {
        // One arm cannot disagree with itself; §8.4's whole method needs at
        // least two. Returning an empty distribution would silently produce the
        // ε floor rather than a derived value, which is exactly the
        // "unprincipled" outcome §8.4 exists to prevent.
        anyhow::bail!(
            "need at least two ISA arms to measure spread (got {}); §8.4 derives epsilon from how \
             much an engine already disagrees with itself",
            paths.len()
        );
    }
    let mut arms: Vec<Vec<f64>> = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(p)?;
        let scores: Vec<f64> = serde_json::from_str(&text)?;
        arms.push(scores);
    }
    let len = arms.iter().map(|a| a.len()).min().unwrap_or(0);
    let mut deltas = Vec::with_capacity(len);
    let mut mags = Vec::with_capacity(len);
    for i in 0..len {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for a in &arms {
            lo = lo.min(a[i]);
            hi = hi.max(a[i]);
        }
        deltas.push(hi - lo);
        mags.push(hi.abs().max(lo.abs()));
    }
    Ok(differ::tolerance::Distribution::from_samples(deltas, &mags))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The harness reads a sweep from `--json`, never from stdout, so every
    /// violation counter the console prints has to survive serialisation.
    /// `unknown_ids` did not: it was printed beside `short_lists` and
    /// `impossible_scores` and then dropped on the way into `RecallPoint`, so
    /// a sweep whose engine returned ids that are not rows of the base file
    /// reached `recall.py` looking clean.
    #[test]
    fn the_sweep_json_carries_every_violation_counter_the_console_prints() {
        let pt = RecallPoint {
            ef: Some(128),
            exact: false,
            recall_at_1: 1.0,
            recall_at_10: 1.0,
            recall_at_100: 1.0,
            recall_at_10_ci95_low: 1.0,
            recall_at_10_ci95_high: 1.0,
            mean_relative_distance_error: 0.0,
            queries: 1,
            short_lists: 1,
            impossible_scores: 2,
            unknown_ids: 3,
            smoke_qps: 1.0,
            returned: 7,
            asked: 10,
            n_matching: Some(6),
        };
        let v = serde_json::to_value(&pt).unwrap();
        assert_eq!(v["short_lists"], 1);
        assert_eq!(v["impossible_scores"], 2);
        assert_eq!(v["unknown_ids"], 3);
        // W12 point 4, by the same argument the rest of this test makes: the
        // harness reads the file, so a figure that only reaches stdout is a
        // figure no reader of the join can act on.
        assert_eq!(v["returned"], 7);
        assert_eq!(v["asked"], 10);
        assert_eq!(v["n_matching"], 6);
    }

    #[test]
    fn metric_parsing_accepts_the_spelling_variants() {
        assert_eq!(parse_metric("euclid").unwrap(), oracle::Metric::Euclid);
        assert_eq!(parse_metric("l2").unwrap(), oracle::Metric::Euclid);
        assert_eq!(parse_metric("l1").unwrap(), oracle::Metric::Manhattan);
        assert_eq!(parse_metric("cosine").unwrap(), oracle::Metric::Cosine);
        assert!(parse_metric("hamming").is_err());
    }

    #[test]
    fn max_tier_is_validated_and_ordered() {
        assert_eq!(parse_max_tier("T0").unwrap(), 0);
        assert_eq!(parse_max_tier("T2").unwrap(), 2);
        assert_eq!(parse_max_tier("T4").unwrap(), 4);
        assert!(parse_max_tier("T2").unwrap() < parse_max_tier("T3").unwrap());
        // Unknown names used to run every tier but T4, silently.
        assert!(parse_max_tier("T9").is_err());
        assert!(parse_max_tier("t1").is_err());
        assert!(parse_max_tier("").is_err());
    }

    #[test]
    fn relevance_passes_quantization_params_through() {
        // W7's join: the sweep has to send what the row sent, and say so.
        let cli = Cli::try_parse_from([
            "conformance",
            "relevance",
            "--base",
            "b",
            "--queries",
            "q",
            "--ground-truth",
            "g",
            "--quantization-oversampling",
            "4",
            "--quantization-rescore",
            "true",
        ])
        .unwrap();
        let Command::Relevance(r) = cli.command else {
            panic!("parsed as another command")
        };
        let (quantization_oversampling, quantization_rescore) =
            (r.quantization_oversampling, r.quantization_rescore);
        assert_eq!(quantization_oversampling, Some(4.0));
        assert_eq!(quantization_rescore, Some(true));
        // Neither given: no quantization params are sent, and none recorded.
        let cli = Cli::try_parse_from([
            "conformance",
            "relevance",
            "--base",
            "b",
            "--queries",
            "q",
            "--ground-truth",
            "g",
        ])
        .unwrap();
        let Command::Relevance(r) = cli.command else {
            panic!("parsed as another command")
        };
        let (quantization_oversampling, quantization_rescore) =
            (r.quantization_oversampling, r.quantization_rescore);
        assert_eq!(quantization_oversampling, None);
        assert_eq!(quantization_rescore, None);
        // Oversampling without rescore is refused rather than defaulted:
        // the request must state both, or the sweep's record names a search
        // the engine's default may or may not have run.
        assert!(
            Cli::try_parse_from([
                "conformance",
                "relevance",
                "--base",
                "b",
                "--queries",
                "q",
                "--ground-truth",
                "g",
                "--quantization-oversampling",
                "4",
            ])
            .is_err()
        );
    }

    #[test]
    fn calibration_requires_at_least_two_arms() {
        // §8.4's method is "measure the spread"; one arm has none.
        let err = spread_across_arms(&[PathBuf::from("a.json")]).unwrap_err();
        assert!(err.to_string().contains("at least two"));
    }
}
