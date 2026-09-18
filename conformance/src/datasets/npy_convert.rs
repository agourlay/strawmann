//! §4.2 — the vector-db-benchmark bundles: `vectors.npy` + `tests.jsonl`.
//!
//! Three of §4.2's entries ship this way (`laion-small-clip`,
//! `h-and-m-2048-angular-filters`, `dbpedia-openai-100K-1536-angular`). bfb
//! reads the layout natively — its `tar` source opens the directory — but the
//! oracle, the differ and `relevance` all read `fbin` (§4.3 makes it
//! canonical), so a dataset that only exists as `.npy` can be loaded and
//! searched and can never be *scored*. §7.4 forbids a qps number without its
//! recall, which makes the conversion the thing standing between these
//! datasets and a publishable row.
//!
//! ## The query split is upstream's, not ours
//!
//! `convert-parquet` holds out the tail of the base rows, because dbpedia's
//! shards are one undivided corpus. Here the queries arrive in their own file
//! and were never part of `vectors.npy`, so there is nothing to hold out and no
//! way for a query to be its own nearest neighbour. Taking a second split here
//! would silently shrink the corpus the shipped neighbour ids index into.
//!
//! ## What is deliberately not read
//!
//! `tests.jsonl` also carries `conditions` and `closest_ids`. For entries whose
//! `conditions` are empty those ids are unfiltered ground truth and could be
//! transcribed; for the filtered entries they are neighbours *under that
//! query's condition* and are not §4.3's ground truth at all. Rather than emit
//! something that is ground truth for some datasets and a trap for others, this
//! converts vectors only and leaves the ground truth to the fp64 oracle, which
//! computes the same thing for every dataset.

use anyhow::Context;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

use super::Vectors;

/// What a conversion produced, for the run record.
#[derive(Debug, Clone)]
pub struct Converted {
    pub n_base: usize,
    pub n_queries: usize,
    pub dim: usize,
    pub dtype: String,
    pub base_path: PathBuf,
    pub queries_path: PathBuf,
}

impl Converted {
    pub fn describe(&self) -> String {
        format!(
            "{} base + {} queries, d={}, {} -> f32 (upstream's split; no overlap by construction)",
            self.n_base, self.n_queries, self.dim, self.dtype
        )
    }
}

/// One IEEE-754 half decoded to `f32`.
///
/// `laion-small-clip` is stored as `<f2` and is the only entry here that is not
/// `f32`; the descriptor's note is blunt about the failure mode, which is that
/// a reader assuming `f32` gets half the rows at twice the dimension and no
/// error at all.
///
/// Written as arithmetic rather than bit-shuffling because the correctness
/// argument is then one line: every half — subnormals included — has an exact
/// `f32` representation, since f32 carries both the range and 13 more mantissa
/// bits. So this is a widening with no rounding step to get wrong.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0f32 } else { 1.0f32 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    match exp {
        // Subnormal (and zero): 2^-14 x frac/1024, i.e. frac x 2^-24.
        0 => sign * frac * f32::from_bits(0x3380_0000),
        // Inf and NaN. NaN keeps its payload rather than being canonicalised,
        // so a corrupt input stays distinguishable from a legitimate one.
        0x1f => {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + frac / 1024.0) * 2.0f32.powi(exp - 15),
    }
}

/// The dtype and shape declared by a `.npy` header.
#[derive(Debug, PartialEq, Eq)]
struct NpyHeader {
    descr: String,
    rows: usize,
    dim: usize,
    /// Bytes from the start of the file to the first element.
    data_offset: usize,
}

/// Parse the `.npy` preamble: magic, version, header length, header dict.
///
/// The header is a Python dict *literal*, not JSON — `False` rather than
/// `false`, single quotes, a trailing comma, and a tuple for `shape`. It is
/// small and rigidly generated, so the fields are pulled out by hand rather
/// than by parsing Python.
fn parse_header(bytes: &[u8], path: &Path) -> anyhow::Result<NpyHeader> {
    anyhow::ensure!(
        bytes.len() > 10 && &bytes[0..6] == b"\x93NUMPY",
        "{}: not a .npy file (bad magic)",
        path.display()
    );
    let (major, minor) = (bytes[6], bytes[7]);
    // v1 stores the header length as u16, v2 and v3 as u32. Reading a v2 file
    // with the v1 rule lands the data offset inside the header text.
    let (len_bytes, header_start) = match major {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => anyhow::bail!(
            "{}: unsupported .npy version {major}.{minor}",
            path.display()
        ),
    };
    let end = header_start + len_bytes;
    anyhow::ensure!(
        bytes.len() >= end,
        "{}: header declares {len_bytes} bytes and the file holds {}",
        path.display(),
        bytes.len()
    );
    let header = String::from_utf8_lossy(&bytes[header_start..end]).to_string();

    let field = |key: &str| -> Option<String> {
        let at = header.find(&format!("'{key}'"))?;
        let rest = &header[at + key.len() + 2..];
        let colon = rest.find(':')?;
        let tail = rest[colon + 1..].trim_start();
        let end = tail.find(',').unwrap_or(tail.len());
        Some(tail[..end].trim().trim_matches('\'').to_string())
    };

    let descr =
        field("descr").with_context(|| format!("{}: header has no 'descr'", path.display()))?;

    // Fortran order transposes the element layout, so reading it as C order
    // yields a matrix of the right size holding entirely wrong rows — the
    // failure that produces a plausible corpus and nonsense neighbours.
    anyhow::ensure!(
        header.contains("'fortran_order': False"),
        "{}: fortran_order is not False; this reader assumes C order",
        path.display()
    );

    let open = header
        .find("'shape'")
        .and_then(|at| header[at..].find('(').map(|p| at + p + 1))
        .with_context(|| format!("{}: header has no 'shape'", path.display()))?;
    let close = header[open..]
        .find(')')
        .map(|p| open + p)
        .with_context(|| format!("{}: malformed 'shape'", path.display()))?;
    let dims: Vec<usize> = header[open..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .with_context(|| format!("shape component {s:?}"))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(
        dims.len() == 2,
        "{}: shape is {dims:?}; this reader wants a 2-D array of dense vectors",
        path.display()
    );

    Ok(NpyHeader {
        descr,
        rows: dims[0],
        dim: dims[1],
        data_offset: end,
    })
}

/// Read a 2-D `.npy` of dense vectors as `f32`.
pub fn read_npy(path: &Path) -> anyhow::Result<(Vectors, String)> {
    // Read whole. The largest of §4.2's bundles is 861 MB
    // (`h-and-m-2048-angular-filters`), against a host the §7.1 gate already
    // requires enough memory to hold a 1M x 1536 collection — so streaming
    // would buy nothing that the oracle reading the same corpus does not
    // already need.
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .read_to_end(&mut bytes)?;
    let h = parse_header(&bytes, path)?;

    let width = match h.descr.as_str() {
        "<f2" => 2,
        "<f4" => 4,
        other => anyhow::bail!(
            "{}: dtype {other:?} is not supported; this reader handles '<f2' and '<f4'",
            path.display()
        ),
    };
    let want = h.rows * h.dim * width;
    let have = bytes.len() - h.data_offset;
    // A truncated download is the common way this goes wrong, and it is silent:
    // the header still declares the full shape.
    anyhow::ensure!(
        have >= want,
        "{}: header declares {} x {} of {} ({want} bytes) and the file holds {have}",
        path.display(),
        h.rows,
        h.dim,
        h.descr
    );

    let raw = &bytes[h.data_offset..h.data_offset + want];
    // `as_chunks` rather than `chunks_exact`: the width is a constant here, so
    // the array size is known at compile time and the element is `&[u8; N]`
    // instead of a slice that has to be re-indexed. `.1` is the remainder,
    // which is empty by the length check above.
    let data: Vec<f32> = match width {
        2 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        _ => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    };
    Ok((
        Vectors {
            n: h.rows,
            dim: h.dim,
            data,
        },
        h.descr,
    ))
}

/// Read the `query` vector from each line of a `tests.jsonl`.
///
/// The other fields are left alone; see the module note on why the shipped
/// neighbour ids are not treated as ground truth here.
pub fn read_tests_jsonl(path: &Path, dim: usize) -> anyhow::Result<Vectors> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut data: Vec<f32> = Vec::new();
    let mut n = 0usize;
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("{}:{}: not JSON", path.display(), i + 1))?;
        let q = v
            .get("query")
            .and_then(|q| q.as_array())
            .with_context(|| format!("{}:{}: no 'query' array", path.display(), i + 1))?;
        // A query at a different width than the corpus cannot be compared to
        // it, and every downstream reader would take the fbin header's word
        // for the dimension rather than noticing.
        anyhow::ensure!(
            q.len() == dim,
            "{}:{}: query is {}-dimensional and the corpus is {dim}",
            path.display(),
            i + 1,
            q.len()
        );
        for x in q {
            let x = x.as_f64().with_context(|| {
                format!("{}:{}: query holds a non-number", path.display(), i + 1)
            })?;
            data.push(x as f32);
        }
        n += 1;
    }
    anyhow::ensure!(n > 0, "{}: no queries", path.display());
    Ok(Vectors { n, dim, data })
}

/// Convert a vector-db-benchmark bundle directory into `base.fbin` + `queries.fbin`.
pub fn convert(in_dir: &Path, out_base: &Path, out_queries: &Path) -> anyhow::Result<Converted> {
    let vectors = in_dir.join("vectors.npy");
    let tests = in_dir.join("tests.jsonl");
    anyhow::ensure!(
        vectors.exists(),
        "{}: no vectors.npy — is this the extracted bundle directory? (`datasets.py extract`)",
        in_dir.display()
    );
    anyhow::ensure!(
        tests.exists(),
        "{}: no tests.jsonl — the queries live there, and without them the corpus cannot be scored",
        tests.display()
    );

    let (base, dtype) = read_npy(&vectors)?;
    let queries = read_tests_jsonl(&tests, base.dim)?;
    for (p, v) in [(out_base, &base), (out_queries, &queries)] {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        super::write_fbin(p, v)?;
    }
    Ok(Converted {
        n_base: base.n,
        n_queries: queries.n,
        dim: base.dim,
        dtype,
        base_path: out_base.to_path_buf(),
        queries_path: out_queries.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, rows: usize, dim: usize, payload: &[u8]) -> Vec<u8> {
        npy_ordered(descr, rows, dim, "False", payload)
    }

    fn npy_ordered(descr: &str, rows: usize, dim: usize, fortran: &str, payload: &[u8]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': ({rows}, {dim}), }}"
        );
        // Built as bytes throughout: the magic starts with 0x93, which is not
        // valid UTF-8, so a test that round-tripped this through a String would
        // corrupt the very file it was trying to describe.
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn a_header_is_parsed_and_locates_the_data() {
        let bytes = npy("<f4", 2, 3, &[0u8; 24]);
        let h = parse_header(&bytes, Path::new("t.npy")).unwrap();
        assert_eq!(h.descr, "<f4");
        assert_eq!((h.rows, h.dim), (2, 3));
        assert_eq!(
            bytes.len() - h.data_offset,
            24,
            "the data begins after the header"
        );
    }

    /// Reading a Fortran-order file as C order yields a matrix of exactly the
    /// right shape holding entirely wrong rows, which is the failure §4 keeps
    /// warning about: plausible, and wrong.
    #[test]
    fn fortran_order_is_refused_rather_than_transposed() {
        let bytes = npy_ordered("<f4", 2, 3, "True", &[0u8; 24]);
        let err = parse_header(&bytes, Path::new("t.npy"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("fortran_order"), "unexpected error: {err}");
    }

    #[test]
    fn a_truncated_file_is_refused_before_it_is_read_as_short_rows() {
        let dir = std::env::temp_dir().join("strawmann_npy_test_trunc");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("vectors.npy");
        // The header still claims 2 x 3; only half the payload is present.
        std::fs::write(&p, npy("<f4", 2, 3, &[0u8; 12])).unwrap();
        let err = read_npy(&p).unwrap_err().to_string();
        assert!(err.contains("the file holds"), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unsupported_dtype_is_named_rather_than_misread() {
        let dir = std::env::temp_dir().join("strawmann_npy_test_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("vectors.npy");
        std::fs::write(&p, npy("<f8", 1, 2, &[0u8; 16])).unwrap();
        let err = read_npy(&p).unwrap_err().to_string();
        assert!(
            err.contains("<f8"),
            "the error should name the dtype: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every half has an exact f32 representation, so these are equalities and
    /// not tolerances. If this widening ever rounds, it is a bug.
    #[test]
    fn halves_widen_exactly() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95); // nearest half to 1/3
        // Largest normal, and the smallest subnormal: the two ends the
        // arithmetic form has to get right without a special case.
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), (1023.0 / 1024.0) * 2.0f32.powi(-14));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn half_rows_survive_the_round_trip_at_the_right_width() {
        let dir = std::env::temp_dir().join("strawmann_npy_test_f16");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("vectors.npy");
        // Two rows of two: [1.0, 2.0] and [-1.0, 0.5].
        let halves: [u16; 4] = [0x3c00, 0x4000, 0xbc00, 0x3800];
        let payload: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        std::fs::write(&p, npy("<f2", 2, 2, &payload)).unwrap();
        let (v, descr) = read_npy(&p).unwrap();
        assert_eq!(descr, "<f2");
        assert_eq!(
            (v.n, v.dim),
            (2, 2),
            "f16 must not read as half the rows at twice the width"
        );
        assert_eq!(v.data, vec![1.0, 2.0, -1.0, 0.5]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_query_at_the_wrong_width_is_refused() {
        let dir = std::env::temp_dir().join("strawmann_npy_test_q");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tests.jsonl");
        std::fs::write(&p, "{\"query\": [1.0, 2.0, 3.0], \"conditions\": {}}\n").unwrap();
        let v = read_tests_jsonl(&p, 3).unwrap();
        assert_eq!((v.n, v.dim), (1, 3));
        assert_eq!(v.data, vec![1.0, 2.0, 3.0]);
        let err = read_tests_jsonl(&p, 4).unwrap_err().to_string();
        assert!(err.contains("3-dimensional"), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
