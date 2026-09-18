//! §9: `conformance/datasets/  fetch, convert, checksum, subset, GT invalidation`
//! — this is `convert`, for the headline tier.
//!
//! `dbpedia-openai-1m` ships as 26 Snappy-compressed parquet shards whose
//! `openai` column is a `list<double>` of 1536 components. Nothing downstream
//! reads parquet: the oracle, the differ and the engine all speak `fbin`, so
//! this is the one place the format is crossed.
//!
//! ## Why the split happens here
//!
//! §4.2 gives this dataset a **held-out split** and marks its ground truth
//! "recompute". Held-out has to mean held out: if the query set is drawn from
//! the base set, every query's nearest neighbour is itself at distance 0, every
//! recall number is inflated, and the inflation is invisible because the run
//! still looks internally consistent. §4.1 is already emphatic that a query
//! distribution which does not match the intended one invalidates relevance
//! work, and querying with vectors that are literally in the index is the
//! degenerate case of that.
//!
//! So the tail of the row order becomes the query set and the head becomes the
//! base set, and they do not overlap. The consequence is that the base set is
//! **990,012 vectors, not 1,000,000** — §4.2's "1M" describes the raw dataset,
//! and holding queries out necessarily costs the base set those rows.
//!
//! ## Why f64 → f32 here rather than later
//!
//! The embeddings are stored as `double`; §3 stores fp32. Casting at conversion
//! time — rather than letting each consumer cast — is what makes the ground
//! truth and the engine agree: the oracle reads the same `fbin` the engine
//! ingests, so both see bit-identical inputs and any difference between them is
//! attributable to the engine rather than to the parquet decode. Computing GT
//! from the f64 parquet while the engine indexes an f32 copy would inject a
//! difference at the very bottom of the stack that no differ tier could
//! localise.
//!
//! The cast is lossy — f64's 53-bit mantissa to f32's 24 — but it is the same
//! loss the engine takes at ingest, which is the point.

use anyhow::Context;
use arrow_array::{Array, cast::AsArray, types::Float64Type};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// What a conversion produced, for the run record.
#[derive(Debug, Clone)]
pub struct Converted {
    pub shards: usize,
    pub rows_total: usize,
    pub n_base: usize,
    pub n_queries: usize,
    pub dim: usize,
    pub base_path: PathBuf,
    pub queries_path: PathBuf,
}

impl Converted {
    pub fn describe(&self) -> String {
        format!(
            "{} shards, {} rows, d={} -> base {} + held-out queries {} (no overlap)",
            self.shards, self.rows_total, self.dim, self.n_base, self.n_queries
        )
    }
}

/// Shards in a deterministic order.
///
/// Row order defines point ids, and point ids are what ground truth indexes
/// into (§4.3). Directory iteration order is filesystem-dependent, so relying
/// on it would make the ids — and therefore every cached GT — depend on how the
/// files happened to land on disk. The `train-000NN-of-00026` prefix sorts
/// correctly as a string, so a plain sort is enough.
pub fn shard_paths(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .collect();
    v.sort();
    anyhow::ensure!(!v.is_empty(), "no .parquet files in {}", dir.display());
    Ok(v)
}

/// The leaf column index for `<column>.list.item`.
fn leaf_index(path: &Path, column: &str) -> anyhow::Result<usize> {
    let r = SerializedFileReader::new(File::open(path)?)?;
    let d = r.metadata().file_metadata().schema_descr();
    let want = format!("{column}.list.item");
    for i in 0..d.num_columns() {
        let p = d.column(i).path().string();
        if p == want || p == column {
            return Ok(i);
        }
    }
    let have: Vec<String> = (0..d.num_columns())
        .map(|i| d.column(i).path().string())
        .collect();
    anyhow::bail!(
        "column {column:?} not found in {}; have {:?}",
        path.display(),
        have
    )
}

/// Row count per shard, read from the footer only — no column data is decoded.
///
/// This is also where a half-downloaded shard is caught. `datasets.py` writes to
/// the final path and resumes in place, so an in-progress transfer is a
/// well-named `.parquet` file with a truncated footer — indistinguishable from
/// a complete one by directory listing, and parquet's own error for it is a
/// bare "Corrupt footer" that names nothing. Since a shard is only skipped or
/// misread *silently* if this passes, the error is worth spelling out.
fn row_counts(paths: &[PathBuf]) -> anyhow::Result<Vec<usize>> {
    paths
        .iter()
        .map(|p| {
            let r = SerializedFileReader::new(File::open(p)?).with_context(|| {
                format!(
                    "reading the parquet footer of {} ({} bytes on disk) — if a download is \
still running or was interrupted, this shard is incomplete; \
`conformance/datasets/datasets.py verify` checks every shard against its \
upstream digest",
                    p.display(),
                    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
                )
            })?;
            Ok(r.metadata().file_metadata().num_rows() as usize)
        })
        .collect()
}

/// Streams an fbin file: header first, then rows, verifying the count at close.
struct FbinWriter {
    w: BufWriter<File>,
    dim: usize,
    declared: usize,
    written: usize,
    path: PathBuf,
}

impl FbinWriter {
    fn create(path: &Path, n: usize, dim: usize) -> anyhow::Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(&(n as u32).to_le_bytes())?;
        w.write_all(&(dim as u32).to_le_bytes())?;
        Ok(Self {
            w,
            dim,
            declared: n,
            written: 0,
            path: path.to_path_buf(),
        })
    }

    fn push(&mut self, row: &[f32]) -> anyhow::Result<()> {
        debug_assert_eq!(row.len(), self.dim);
        // `to_le_bytes` per component rather than a transmute of the slice:
        // `read_fbin` is explicitly little-endian, and a big-endian host writing
        // native bytes would produce a file that reads back as garbage on the
        // machine that wrote it.
        let mut buf = Vec::with_capacity(self.dim * 4);
        for x in row {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        self.w.write_all(&buf)?;
        self.written += 1;
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<()> {
        self.w.flush()?;
        // The header was written from the footer-derived count. If the decoded
        // rows disagree, the file claims a length it does not have and every
        // downstream reader silently reads past the end of the real data.
        anyhow::ensure!(
            self.written == self.declared,
            "{}: header declares {} rows but {} were written",
            self.path.display(),
            self.declared,
            self.written
        );
        Ok(())
    }
}

/// Convert a directory of parquet shards into `base.fbin` + `queries.fbin`.
///
/// `n_queries` rows are taken from the **tail** of the row order and excluded
/// from the base set.
pub fn convert(
    dir: &Path,
    column: &str,
    out_base: &Path,
    out_queries: &Path,
    n_queries: usize,
    limit: Option<usize>,
) -> anyhow::Result<Converted> {
    let paths = shard_paths(dir)?;
    let counts = row_counts(&paths)?;
    let rows_available: usize = counts.iter().sum();
    let rows_total = limit
        .map(|l| l.min(rows_available))
        .unwrap_or(rows_available);

    anyhow::ensure!(
        rows_total > n_queries,
        "{rows_total} rows is not enough to hold out {n_queries} queries"
    );
    let n_base = rows_total - n_queries;

    let leaf = leaf_index(&paths[0], column)?;

    // The dimension is discovered from the first row and then enforced on every
    // subsequent one. A ragged embedding column would otherwise produce an fbin
    // whose rows silently desynchronise from its header.
    let mut dim = 0usize;
    let mut base_w: Option<FbinWriter> = None;
    let mut query_w: Option<FbinWriter> = None;

    let mut row_idx = 0usize;
    let mut row = Vec::<f32>::new();

    'outer: for path in &paths {
        if row_idx >= rows_total {
            break;
        }
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("opening {}", path.display()))?;
        // Project to the embedding column alone. The shards also carry `text`,
        // which is the bulk of the bytes and is never used; decoding it would
        // roughly triple the conversion time for nothing.
        let mask = ProjectionMask::leaves(builder.parquet_schema(), [leaf]);
        let reader = builder
            .with_projection(mask)
            .with_batch_size(1024)
            .build()?;

        for batch in reader {
            let batch = batch?;
            let col = batch.column(0);
            let list = col.as_list_opt::<i32>().ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: column {column:?} is {:?}, expected a list",
                    path.display(),
                    col.data_type()
                )
            })?;

            for i in 0..list.len() {
                if row_idx >= rows_total {
                    break 'outer;
                }
                let vals = list.value(i);
                let f = vals
                    .as_primitive_opt::<Float64Type>()
                    .ok_or_else(|| anyhow::anyhow!("{}: list items are not f64", path.display()))?;

                if dim == 0 {
                    dim = f.len();
                    anyhow::ensure!(dim > 0 && dim <= 65536, "implausible dim {dim}");
                    row = vec![0.0; dim];
                    base_w = Some(FbinWriter::create(out_base, n_base, dim)?);
                    query_w = Some(FbinWriter::create(out_queries, n_queries, dim)?);
                }
                anyhow::ensure!(
                    f.len() == dim,
                    "{}: row {row_idx} has {} components, expected {dim}",
                    path.display(),
                    f.len()
                );

                for (j, slot) in row.iter_mut().enumerate() {
                    *slot = f.value(j) as f32;
                }

                if row_idx < n_base {
                    base_w.as_mut().unwrap().push(&row)?;
                } else {
                    query_w.as_mut().unwrap().push(&row)?;
                }
                row_idx += 1;
            }
        }
    }

    let base_w = base_w.ok_or_else(|| anyhow::anyhow!("no rows decoded from {}", dir.display()))?;
    let query_w = query_w.unwrap();
    base_w.finish()?;
    query_w.finish()?;

    Ok(Converted {
        shards: paths.len(),
        rows_total,
        n_base,
        n_queries,
        dim,
        base_path: out_base.to_path_buf(),
        queries_path: out_queries.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_array::builder::{Float64Builder, ListBuilder};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    /// Write a shard whose row `i` is the constant vector `[i, i, ..., i]`, so
    /// a decoded row identifies itself and any reordering or off-by-one is
    /// visible rather than merely suspected.
    fn write_shard(path: &Path, first_row: usize, rows: usize, dim: usize) {
        let mut b = ListBuilder::new(Float64Builder::new());
        for r in 0..rows {
            for _ in 0..dim {
                b.values().append_value((first_row + r) as f64);
            }
            b.append(true);
        }
        let arr = b.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "openai",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
        let f = File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn read_fbin_rows(path: &Path) -> (usize, usize, Vec<f32>) {
        let v = super::super::read_fbin(path, None).unwrap();
        (v.n, v.dim, v.data)
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("strawmann-parquet-test-{tag}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn converts_and_holds_the_query_split_out_of_base() {
        let d = tmpdir("split");
        let dim = 4;
        write_shard(&d.join("train-00000-of-00002.parquet"), 0, 10, dim);
        write_shard(&d.join("train-00001-of-00002.parquet"), 10, 10, dim);

        let base = d.join("base.fbin");
        let queries = d.join("q.fbin");
        let c = convert(&d, "openai", &base, &queries, 5, None).unwrap();

        assert_eq!(c.rows_total, 20);
        assert_eq!(c.n_base, 15);
        assert_eq!(c.n_queries, 5);
        assert_eq!(c.dim, dim);

        let (nb, db, bdata) = read_fbin_rows(&base);
        let (nq, dq, qdata) = read_fbin_rows(&queries);
        assert_eq!((nb, db), (15, dim));
        assert_eq!((nq, dq), (5, dim));

        // Base holds rows 0..15 in order; queries hold the *tail* 15..20.
        for r in 0..nb {
            assert_eq!(bdata[r * dim], r as f32, "base row {r}");
        }
        for r in 0..nq {
            assert_eq!(qdata[r * dim], (15 + r) as f32, "query row {r}");
        }

        // The invariant the whole split exists for: no query vector is in the
        // base set. If it were, every query's nearest neighbour would be itself
        // at distance 0 and every recall number would be silently inflated.
        for r in 0..nq {
            let q = &qdata[r * dim..(r + 1) * dim];
            for b in 0..nb {
                assert_ne!(q, &bdata[b * dim..(b + 1) * dim], "query {r} == base {b}");
            }
        }
    }

    /// Point ids are row indices and ground truth indexes into them (§4.3), so
    /// shard order must not depend on how the filesystem enumerates a directory.
    #[test]
    fn shard_order_is_sorted_not_directory_order() {
        let d = tmpdir("order");
        // Created out of order on purpose.
        for name in [
            "train-00002-of-00003.parquet",
            "train-00000-of-00003.parquet",
            "train-00001-of-00003.parquet",
        ] {
            write_shard(&d.join(name), 0, 1, 2);
        }
        let got: Vec<String> = shard_paths(&d)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            got,
            vec![
                "train-00000-of-00003.parquet",
                "train-00001-of-00003.parquet",
                "train-00002-of-00003.parquet"
            ]
        );
    }

    #[test]
    fn limit_takes_a_prefix_and_still_holds_out_the_tail() {
        let d = tmpdir("limit");
        write_shard(&d.join("train-00000-of-00001.parquet"), 0, 100, 3);
        let base = d.join("base.fbin");
        let queries = d.join("q.fbin");
        let c = convert(&d, "openai", &base, &queries, 10, Some(50)).unwrap();
        assert_eq!(c.rows_total, 50);
        assert_eq!(c.n_base, 40);
        let (_, dim, qdata) = read_fbin_rows(&queries);
        // The held-out rows are the tail of the *limited* range, 40..50.
        assert_eq!(qdata[0], 40.0);
        assert_eq!(qdata[9 * dim], 49.0);
    }

    #[test]
    fn refuses_a_query_split_larger_than_the_dataset() {
        let d = tmpdir("toosmall");
        write_shard(&d.join("train-00000-of-00001.parquet"), 0, 5, 2);
        let e = convert(&d, "openai", &d.join("b.fbin"), &d.join("q.fbin"), 10, None)
            .expect_err("should refuse");
        assert!(e.to_string().contains("not enough"), "{e}");
    }

    #[test]
    fn names_a_missing_column_and_lists_what_is_there() {
        let d = tmpdir("badcol");
        write_shard(&d.join("train-00000-of-00001.parquet"), 0, 2, 2);
        let e = convert(
            &d,
            "embedding",
            &d.join("b.fbin"),
            &d.join("q.fbin"),
            1,
            None,
        )
        .expect_err("should refuse");
        let s = e.to_string();
        assert!(s.contains("embedding"), "{s}");
        assert!(
            s.contains("openai"),
            "should list the columns that do exist: {s}"
        );
    }

    /// A half-downloaded shard is a well-named `.parquet` with a truncated
    /// footer. Parquet's own error for that names nothing, which sent one
    /// debugging session looking at the converter instead of the download.
    #[test]
    fn an_incomplete_shard_is_named_in_the_error() {
        let d = tmpdir("truncated");
        let p = d.join("train-00000-of-00001.parquet");
        write_shard(&p, 0, 10, 4);
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() / 2]).unwrap();

        let e = convert(&d, "openai", &d.join("b.fbin"), &d.join("q.fbin"), 1, None)
            .expect_err("should refuse a truncated shard");
        let s = format!("{e:#}");
        assert!(s.contains("train-00000-of-00001.parquet"), "{s}");
        assert!(s.contains("datasets.py"), "should point at the fix: {s}");
    }
}
