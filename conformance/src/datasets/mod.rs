//! §4.2 / §4.3 — datasets, formats, and the invariants the harness enforces.
//!
//! §4.3 fixes the format:
//!
//! > "**Canonical format:** `fbin`/`ibin` (big-ann-benchmarks convention),
//! > since bfb already reads `fbin`. `fvecs`/`ivecs` inputs are converted once
//! > at ingest, not at benchmark time. Ground truth is `(nq, k)` int32 neighbour
//! > IDs plus `(nq, k)` float32 distances, `k = 100`."
//!
//! ## fbin vs fvecs
//!
//! They are different formats and confusing them produces a file that parses
//! and is wrong:
//!
//! - **fbin**: a `(u32 n, u32 dim)` header, then `n × dim` little-endian f32.
//!   One header for the whole file.
//! - **fvecs**: no header. Each vector is `(u32 dim, dim × f32)`, so the
//!   dimension is repeated per row.
//!
//! Reading an fvecs file as fbin yields `n = dim` and `dim = bitcast(first
//! float)`, which is usually an absurd number — but for some dimensions and
//! some data it is merely wrong rather than absurd, and the run proceeds.

pub mod npy_convert;
pub mod parquet_convert;

use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct Vectors {
    pub n: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Vectors {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// §4.3: "Subsetting invalidates GT."
    ///
    /// Taking a prefix is the only safe subset: ground truth indexes into
    /// base-file row order (§4.3), so a sampled subset would renumber every
    /// point and silently invalidate every neighbour id.
    #[allow(dead_code)]
    pub fn prefix(&self, n: usize) -> Vectors {
        let n = n.min(self.n);
        Vectors {
            n,
            dim: self.dim,
            data: self.data[..n * self.dim].to_vec(),
        }
    }
}

#[cfg(test)]
mod fbin_header_tests {
    use super::*;
    use std::io::Write;

    /// A header claiming more rows than the file holds must be rejected before
    /// the allocation, not after. Without the size check this test allocates
    /// 2 GB and either OOMs or succeeds by luck.
    #[test]
    fn corrupt_row_count_is_rejected_before_allocating() {
        let dir = std::env::temp_dir().join("strawmann_fbin_hdr_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.fbin");
        let mut f = std::fs::File::create(&path).unwrap();
        // n = 4,000,000 at dim = 128 would need ~2 GB; the file holds one row.
        f.write_all(&4_000_000u32.to_le_bytes()).unwrap();
        f.write_all(&128u32.to_le_bytes()).unwrap();
        f.write_all(&vec![0u8; 128 * 4]).unwrap();
        drop(f);

        let err = read_fbin(&path, None).unwrap_err().to_string();
        assert!(
            err.contains("Truncated download"),
            "unexpected error: {err}"
        );
        assert!(
            err.contains("4000000"),
            "the error should name the declared count: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_well_formed_file_still_reads() {
        let dir = std::env::temp_dir().join("strawmann_fbin_hdr_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("good.fbin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        for x in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
        drop(f);

        let v = read_fbin(&path, None).unwrap();
        assert_eq!((v.n, v.dim), (2, 3));
        assert_eq!(v.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        std::fs::remove_file(&path).ok();
    }
}

/// Read a `.fbin` file: `(u32 n, u32 dim)` then `n × dim` f32.
pub fn read_fbin(path: &Path, limit: Option<usize>) -> anyhow::Result<Vectors> {
    let mut f = std::fs::File::open(path)?;
    let mut header = [0u8; 8];
    f.read_exact(&mut header)?;
    let n_file = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    // A misidentified fvecs file lands here with an implausible dimension.
    // Failing loudly is the point of the check; §4 is emphatic that a silently
    // wrong dataset poisons everything downstream.
    if dim == 0 || dim > 65536 {
        anyhow::bail!(
            "{}: dim={dim} is implausible — is this an fvecs file being read as fbin? (§4.3)",
            path.display()
        );
    }

    // The header declares the row count, so it also predicts the file size.
    // Checking that before allocating turns a corrupt header into a clear error
    // instead of a multi-terabyte allocation: `dim` is validated above, but a
    // garbage `n` of 4e9 at dim=128 asks for 2 TB and dies as an OOM naming
    // nothing. A truncated download lands here too, one file short of its
    // digest check.
    let on_disk = f.metadata()?.len();
    let expected = 8u64 + (n_file as u64) * (dim as u64) * 4;
    if on_disk != expected {
        anyhow::bail!(
            "{}: header declares n={n_file} dim={dim}, which needs {expected} bytes, \
             but the file is {on_disk}. Truncated download, or not an fbin? (§4.2)",
            path.display()
        );
    }

    let n = limit.map(|l| l.min(n_file)).unwrap_or(n_file);
    let mut data = vec![0f32; n * dim];
    let mut buf = vec![0u8; dim * 4];
    // Buffered: a million rows at d=128 was a million 512-byte `read`
    // syscalls on the path every oracle, differ and sweep takes.
    let mut f = std::io::BufReader::with_capacity(1 << 20, f);
    for i in 0..n {
        f.read_exact(&mut buf)?;
        for j in 0..dim {
            data[i * dim + j] = f32::from_le_bytes(buf[j * 4..j * 4 + 4].try_into().unwrap());
        }
    }
    Ok(Vectors { n, dim, data })
}

/// Read a `.fvecs` file: each row is `(u32 dim, dim × f32)`, no file header.
pub fn read_fvecs(path: &Path, limit: Option<usize>) -> anyhow::Result<Vectors> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 4 {
        anyhow::bail!("{}: too short to be fvecs", path.display());
    }
    let dim = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if dim == 0 || dim > 65536 {
        anyhow::bail!("{}: dim={dim} is implausible for fvecs", path.display());
    }
    let row_bytes = 4 + dim * 4;
    if bytes.len() % row_bytes != 0 {
        anyhow::bail!(
            "{}: length {} is not a multiple of the {row_bytes}-byte row — ragged or wrong format",
            path.display(),
            bytes.len()
        );
    }
    let n_file = bytes.len() / row_bytes;
    let n = limit.map(|l| l.min(n_file)).unwrap_or(n_file);

    let mut data = vec![0f32; n * dim];
    for i in 0..n {
        let base = i * row_bytes;
        // Every row repeats the dimension; a mismatch means the file is not
        // what it claims.
        let d = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap()) as usize;
        if d != dim {
            anyhow::bail!(
                "{}: row {i} declares dim {d}, expected {dim}",
                path.display()
            );
        }
        for j in 0..dim {
            let o = base + 4 + j * 4;
            data[i * dim + j] = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        }
    }
    Ok(Vectors { n, dim, data })
}

/// Read `.ivecs` ground truth: each row is `(u32 k, k × i32)`.
pub fn read_ivecs(path: &Path) -> anyhow::Result<(usize, usize, Vec<u32>)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 4 {
        anyhow::bail!("{}: too short to be ivecs", path.display());
    }
    let k = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let row_bytes = 4 + k * 4;
    if bytes.len() % row_bytes != 0 {
        anyhow::bail!("{}: not a multiple of the row size", path.display());
    }
    let n = bytes.len() / row_bytes;
    let mut out = vec![0u32; n * k];
    for i in 0..n {
        let base = i * row_bytes;
        for j in 0..k {
            let o = base + 4 + j * 4;
            out[i * k + j] = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        }
    }
    Ok((n, k, out))
}

/// §4.3: "`fvecs`/`ivecs` inputs are converted once at ingest, not at benchmark
/// time."
pub fn write_fbin(path: &Path, v: &Vectors) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(v.n as u32).to_le_bytes())?;
    f.write_all(&(v.dim as u32).to_le_bytes())?;
    let mut buf = Vec::with_capacity(v.dim * 4);
    for i in 0..v.n {
        buf.clear();
        for &x in v.row(i) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    Ok(())
}

/// §4.2's dataset table, read from `conformance/datasets/datasets.json`.
///
/// This used to be a hand-maintained `const` array. It duplicated the fetcher's
/// manifest, and the two had already drifted: §4.2 lists six datasets, the
/// manifest pinned files for two, and nothing reported the gap. The descriptor
/// is now the single source of truth and this parses it, so adding a dataset is
/// one JSON edit rather than an edit here plus an edit there plus remembering
/// that both exist.
///
/// `include_str!` rather than a runtime read: the path would otherwise depend
/// on the working directory, and a conformance binary that cannot find its own
/// dataset table depending on where it was invoked from is a worse failure than
/// a compile error.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct DatasetSpec {
    pub name: String,
    #[serde(default)]
    pub n: usize,
    #[serde(rename = "vector_size", default)]
    pub dim: usize,
    #[serde(rename = "distance", deserialize_with = "de_metric")]
    pub metric: crate::oracle::Metric,
    #[serde(default)]
    pub n_queries: usize,
    /// §4.2: whether ground truth ships with the dataset or must be recomputed.
    #[serde(default)]
    pub gt_shipped: bool,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub status: Status,
}

/// §4.2's two states for a dataset.
///
/// An enum rather than a `String` because the value was compared against a
/// literal in seven places across two languages, and nothing validated it. A
/// typo in `datasets.json` did not fail to parse — it made the dataset
/// invisible to every `status == "available"` filter, in Rust and in
/// `datasets.py` alike, while `datasets-descriptor` still passed because
/// listing a descriptor does not check what its statuses say. Unknown values
/// are now a parse error on both sides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Every file is pinned and fetchable.
    Available,
    /// §4.2 names it, but nobody has pinned its files yet. The default,
    /// matching `datasets.py`: a dataset says nothing until someone pins it.
    #[default]
    Declared,
}

impl std::fmt::Display for Status {
    /// The word the descriptor uses, so `datasets list` prints what
    /// `datasets.json` says and the two cannot drift apart.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Status::Available => "available",
            Status::Declared => "declared",
        })
    }
}

fn de_metric<'de, D>(d: D) -> Result<crate::oracle::Metric, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    use serde::de::Error;
    let s = String::deserialize(d)?;
    crate::oracle::Metric::parse(&s)
        .ok_or_else(|| D::Error::custom(format!("unknown metric {s:?}")))
}

const DESCRIPTOR: &str = include_str!("../../datasets/datasets.json");

pub static DATASETS: std::sync::LazyLock<Vec<DatasetSpec>> = std::sync::LazyLock::new(|| {
    serde_json::from_str(DESCRIPTOR).expect("conformance/datasets/datasets.json is malformed")
});

/// Lookup by name, for callers that have a name rather than a tier.
#[allow(dead_code)]
pub fn find(name: &str) -> Option<&'static DatasetSpec> {
    DATASETS.iter().find(|d| d.name == name)
}

/// Datasets whose files are pinned and therefore fetchable.
pub fn available() -> impl Iterator<Item = &'static DatasetSpec> {
    DATASETS.iter().filter(|d| d.status == Status::Available)
}

/// §4.1's rule about which workloads may use random vectors.
///
/// > "Random vectors remain acceptable for W0–W2 (plumbing and ingest) and
/// > nothing else. Uniform high-dimensional noise has no cluster structure, so
/// > graph shape, recall curves, and cache behaviour are all unrepresentative."
///
/// §4.2's rule as a predicate. Enforced in practice by
/// `bench/harness/workloads.py`, which routes every search row through the
/// dataset's own query file (`query_collection`) and leaves only W0–W2 on
/// generated vectors; this is the statement of the rule that the Rust side can
/// assert against when it grows a workload runner of its own.
#[allow(dead_code)]
pub fn random_vectors_acceptable(workload: &str) -> bool {
    matches!(workload, "W0" | "W1" | "W2")
}

/// §4.1's other rule, enforced rather than documented.
///
/// > "**`bfb` always generates random query vectors.** `get_dense_queries`
/// > calls `random_dense_vector` unconditionally — `--fbin` supplies the *base*
/// > set only... bfb is the *load generator* — throughput, latency, tail
/// > behaviour, concurrency. The `conformance/` harness ... is the *relevance
/// > generator* ... Never let one report the other's numbers."
///
/// §4.1's rule as a guard. Nothing in this binary can violate it — the
/// conformance harness is the relevance generator by construction — so it has
/// no call site here and is kept as the executable form of the rule.
#[allow(dead_code)]
pub fn reject_relevance_from_bfb(source: &str) -> Result<(), String> {
    if source == "bfb" {
        return Err(
            "bfb cannot produce relevance numbers: its queries are random and `--search-quality` is \
             self-referential (§4.1). Use the conformance harness."
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("strawmann-ds-test-{name}"));
        p
    }

    #[test]
    fn fbin_round_trips() {
        let p = tmp("rt.fbin");
        let v = Vectors {
            n: 3,
            dim: 4,
            data: (0..12).map(|i| i as f32).collect(),
        };
        write_fbin(&p, &v).unwrap();
        let back = read_fbin(&p, None).unwrap();
        assert_eq!(back.n, 3);
        assert_eq!(back.dim, 4);
        assert_eq!(back.data, v.data);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn fbin_honours_a_limit_without_reading_the_rest() {
        let p = tmp("limit.fbin");
        let v = Vectors {
            n: 100,
            dim: 2,
            data: (0..200).map(|i| i as f32).collect(),
        };
        write_fbin(&p, &v).unwrap();
        let back = read_fbin(&p, Some(10)).unwrap();
        assert_eq!(back.n, 10);
        assert_eq!(back.data.len(), 20);
        assert_eq!(back.row(9), &[18.0, 19.0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn fvecs_round_trips_and_rejects_a_ragged_file() {
        let p = tmp("rt.fvecs");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            for i in 0..3u32 {
                f.write_all(&4u32.to_le_bytes()).unwrap();
                for j in 0..4 {
                    f.write_all(&((i * 4 + j) as f32).to_le_bytes()).unwrap();
                }
            }
        }
        let v = read_fvecs(&p, None).unwrap();
        assert_eq!(v.n, 3);
        assert_eq!(v.dim, 4);
        assert_eq!(v.row(2), &[8.0, 9.0, 10.0, 11.0]);

        // Truncate mid-row: must be refused, not silently short.
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.truncate(bytes.len() - 3);
        std::fs::write(&p, &bytes).unwrap();
        assert!(read_fvecs(&p, None).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn reading_fvecs_as_fbin_is_caught() {
        // The confusion this check exists for. An fvecs file's first 8 bytes
        // are (dim, first_float), which as an fbin header gives an absurd dim.
        let p = tmp("confused.fvecs");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(&128u32.to_le_bytes()).unwrap();
            for _ in 0..128 {
                f.write_all(&1.5f32.to_le_bytes()).unwrap();
            }
        }
        let err = read_fbin(&p, None).unwrap_err().to_string();
        assert!(err.contains("fvecs"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn ivecs_reads_ground_truth() {
        let p = tmp("gt.ivecs");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            for q in 0..2u32 {
                f.write_all(&3u32.to_le_bytes()).unwrap();
                for j in 0..3u32 {
                    f.write_all(&(q * 10 + j).to_le_bytes()).unwrap();
                }
            }
        }
        let (n, k, ids) = read_ivecs(&p).unwrap();
        assert_eq!(n, 2);
        assert_eq!(k, 3);
        assert_eq!(&ids[..3], &[0, 1, 2]);
        assert_eq!(&ids[3..], &[10, 11, 12]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn prefix_is_the_only_subset_that_preserves_row_indices() {
        // §4.3: ground truth indexes into base-file row order.
        let v = Vectors {
            n: 5,
            dim: 2,
            data: (0..10).map(|i| i as f32).collect(),
        };
        let p = v.prefix(3);
        assert_eq!(p.n, 3);
        for i in 0..3 {
            assert_eq!(p.row(i), v.row(i));
        }
    }

    #[test]
    fn dataset_table_matches_section_4_2() {
        let sift = find("sift1m").unwrap();
        assert_eq!(sift.dim, 128);
        assert_eq!(sift.metric, crate::oracle::Metric::Euclid);
        assert!(sift.gt_shipped);

        let headline = find("dbpedia-openai-1m").unwrap();
        assert_eq!(headline.dim, 1536);
        assert_eq!(headline.metric, crate::oracle::Metric::Cosine);
        // §4.2 marks the headline tier's GT as "recompute".
        assert!(!headline.gt_shipped);

        // §4.2's cross-modal entry, where query and base distributions differ.
        let ood = find("text2image-10m").unwrap();
        assert!(ood.role.contains("OOD") || ood.role.contains("cross-modal"));
    }

    #[test]
    fn random_vectors_are_only_acceptable_for_plumbing_and_ingest() {
        // §4.1: "Random vectors remain acceptable for W0–W2 ... and nothing else."
        assert!(random_vectors_acceptable("W0"));
        assert!(random_vectors_acceptable("W2"));
        assert!(!random_vectors_acceptable("W3"));
        assert!(!random_vectors_acceptable("W9"));
        assert!(!random_vectors_acceptable("W10"));
    }

    #[test]
    fn bfb_may_not_report_relevance() {
        // §4.1: "Never let one report the other's numbers."
        assert!(reject_relevance_from_bfb("conformance").is_ok());
        let e = reject_relevance_from_bfb("bfb").unwrap_err();
        assert!(e.contains("self-referential"));
    }
}
