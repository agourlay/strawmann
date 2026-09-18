//! §8.4 — calibrating ε instead of guessing it.
//!
//! > "Picking `1e-6` because it looks reasonable is unprincipled. Instead,
//! > derive the tolerance from measured variance:
//! >
//! > 1. **Measure Qdrant's own internal spread.** Qdrant dispatches to
//! >    AVX/SSE/NEON/scalar kernels by runtime feature detection, and these have
//! >    different accumulator counts and FMA usage — so *Qdrant does not agree
//! >    with itself across ISAs*. Run the same query set against Qdrant
//! >    builds/hosts forcing each path, and record the score-delta distribution.
//! > 2. **Measure strawmann's spread** across its own ISA matrix (§7.5) the same
//! >    way.
//! > 3. **ε := a small multiple of the larger of the two**, per metric and per
//! >    dimension, justified in `docs/tolerance.md` with the underlying
//! >    distributions."
//!
//! And the reporting rule that follows:
//!
//! > "conformance reports the **distribution** of `|Δscore|` — max, p99.9, p99,
//! > mean — not a pass/fail bit. A run that passes with a max delta 10× worse
//! > than yesterday's is a regression, and a pass/fail gate would hide it."
//!
//! So nothing in this file returns a bool. `Distribution` is the unit of
//! output, and a `Verdict` carries the distribution alongside its judgement.

use crate::oracle::Metric;

/// The measured distribution of `|Δscore|`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Distribution {
    pub count: usize,
    pub max: f64,
    pub p999: f64,
    pub p99: f64,
    pub p50: f64,
    pub mean: f64,
    /// Relative deltas, for the metrics §8.4 says need them.
    pub max_relative: f64,
    pub p99_relative: f64,
}

impl Distribution {
    /// Build from raw absolute deltas and the magnitudes they were measured
    /// against.
    ///
    /// §8.4: "Unnormalised dot and Euclid scale with vector magnitude and want
    /// a relative tolerance instead." Both are computed so the caller picks the
    /// right one per metric rather than the harness assuming.
    pub fn from_samples(mut abs: Vec<f64>, magnitudes: &[f64]) -> Distribution {
        if abs.is_empty() {
            return Distribution::default();
        }
        let mut rel: Vec<f64> = abs
            .iter()
            .zip(magnitudes.iter())
            .map(|(d, m)| if *m > 0.0 { d / m } else { *d })
            .collect();

        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        rel.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let pick = |v: &[f64], q: f64| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            // Nearest-rank percentile. With a handful of samples an
            // interpolating percentile invents values that were never
            // measured, which is the wrong thing for a tolerance.
            let idx = ((q * v.len() as f64).ceil() as usize).saturating_sub(1);
            v[idx.min(v.len() - 1)]
        };

        let sum: f64 = abs.iter().sum();
        Distribution {
            count: abs.len(),
            max: *abs.last().unwrap(),
            p999: pick(&abs, 0.999),
            p99: pick(&abs, 0.99),
            p50: pick(&abs, 0.50),
            mean: sum / abs.len() as f64,
            max_relative: *rel.last().unwrap(),
            p99_relative: pick(&rel, 0.99),
        }
    }

    /// §8.4's reporting rule made into a line of output.
    pub fn describe(&self) -> String {
        format!(
            "n={} max={:.3e} p99.9={:.3e} p99={:.3e} p50={:.3e} mean={:.3e} (rel max={:.3e} p99={:.3e})",
            self.count,
            self.max,
            self.p999,
            self.p99,
            self.p50,
            self.mean,
            self.max_relative,
            self.p99_relative
        )
    }
}

/// Whether a metric's tolerance should be absolute or relative.
///
/// §8.4: "for normalised fp32 dot at d=768, blocked summation error is roughly
/// `O(log d · 2⁻²⁴)`, so an absolute delta around `1e-6` on scores in `[-1, 1]`.
/// Unnormalised dot and Euclid scale with vector magnitude and want a relative
/// tolerance instead."
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ToleranceKind {
    Absolute,
    Relative,
}

pub fn tolerance_kind(metric: Metric) -> ToleranceKind {
    match metric {
        // Cosine scores live in [-1, 1] by construction, so an absolute bound
        // is meaningful and does not vary with the data.
        Metric::Cosine => ToleranceKind::Absolute,
        // Dot, Euclid and Manhattan all scale with vector magnitude.
        Metric::Dot | Metric::Euclid | Metric::Manhattan => ToleranceKind::Relative,
    }
}

/// The expected magnitude of blocked-summation error, from §8.4.
///
/// `O(log d · 2⁻²⁴)`. This is a *sanity check*, not the tolerance:
///
/// > "Expected magnitudes, as a sanity check rather than a target."
///
/// If the measured spread is far below this, the measurement is probably not
/// exercising the kernels it thinks it is; far above, something other than
/// summation order is in play.
pub fn expected_summation_error(dim: usize) -> f64 {
    let log_d = (dim as f64).log2().max(1.0);
    log_d * 2f64.powi(-24)
}

/// A calibrated tolerance for one (metric, dim) cell.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Epsilon {
    pub metric: Metric,
    pub dim: usize,
    /// The value to compare against.
    pub value: f64,
    pub relative: bool,
    /// What it was derived from, so `docs/tolerance.md` can be generated rather
    /// than written by hand.
    pub qdrant_spread: Distribution,
    pub strawmann_spread: Distribution,
    pub multiple: f64,
}

impl Epsilon {
    /// The **absolute** tolerance this ε permits at a given score magnitude.
    ///
    /// The one place the relative/absolute distinction is resolved. `value` on
    /// its own is not a tolerance for three of the four metrics — it is a
    /// *ratio* — and every caller that treated it as one was comparing a delta
    /// in the units of the data against a dimensionless number.
    ///
    /// That was not theoretical. At SIFT1M euclid, distances are ~300 and the
    /// derived ε is 4.172e-7 relative, so the true tolerance is 1.25e-4 — about
    /// 4 fp32 ulp. Read as absolute it becomes 4.172e-7, which is **0.014 of a
    /// single ulp** and unsatisfiable by any fp32 engine. Two call sites did
    /// exactly that; see the tests below.
    pub fn absolute_at(&self, magnitude: f64) -> f64 {
        if self.relative {
            // `max(1.0)`, not `max(1e-30)`: dot scores straddle zero, and a
            // relative tolerance at |score| ≈ 0 permitted ~ε·1e-30, which no
            // engine can meet -- and `delta_vs_oracle` scales by the *truth*,
            // so one near-zero neighbour disabled the whole oracle
            // arbitration for the run. Below magnitude 1 the tolerance is the
            // absolute ε, the usual |Δ| ≤ ε·max(1, |x|) form.
            self.value * magnitude.abs().max(1.0)
        } else {
            self.value
        }
    }

    pub fn holds(&self, delta: f64, magnitude: f64) -> bool {
        delta <= self.absolute_at(magnitude)
    }

    /// How far a delta is past what is permitted, as a multiple.
    ///
    /// `<= 1.0` is within tolerance. Reported instead of a bare absolute delta
    /// beside a relative ε, which reads as a 7000x violation when the real one
    /// is 24x.
    pub fn overshoot(&self, delta: f64, magnitude: f64) -> f64 {
        let allowed = self.absolute_at(magnitude);
        if allowed <= 0.0 {
            if delta > 0.0 { f64::INFINITY } else { 0.0 }
        } else {
            delta / allowed
        }
    }

    /// The two measured spreads `calibrate` chose between.
    fn spreads(&self) -> (f64, f64) {
        if self.relative {
            (
                self.qdrant_spread.max_relative,
                self.strawmann_spread.max_relative,
            )
        } else {
            (self.qdrant_spread.max, self.strawmann_spread.max)
        }
    }

    /// Names the branch of `calibrate` that produced the value. It used to
    /// print "= 4x max(qdrant …, strawmann …)" unconditionally, which for a
    /// value clamped to the summation-error floor was an equation that did not
    /// hold — and the floor is what a run with no measured spread gets.
    pub fn describe(&self) -> String {
        let (q, s) = self.spreads();
        let derived = self.multiple * q.max(s);
        let how = if derived >= self.value {
            format!(
                "= {}x max(qdrant {q:.3e}, strawmann {s:.3e})",
                self.multiple
            )
        } else {
            format!(
                "= §8.4 floor expected_summation_error(d={}) (the {}x max(qdrant {q:.3e}, strawmann {s:.3e}) = {derived:.3e} is below it)",
                self.dim, self.multiple
            )
        };
        format!(
            "eps({}, d={}) = {:.3e} {} {how}",
            self.metric.as_str(),
            self.dim,
            self.value,
            if self.relative {
                "relative"
            } else {
                "absolute"
            },
        )
    }

    /// The `CALIBRATED` entry for this cell, ready to paste. Printed by
    /// `calibrate` so the constant and `docs/tolerance.md` stop drifting apart
    /// through hand-editing.
    pub fn rust_literal(&self) -> String {
        format!(
            "    // docs/tolerance.md: {} d={}, {}.\n    (Metric::{:?}, {}, {:.3e}, {}),",
            self.metric.as_str(),
            self.dim,
            self.describe()
                .split_once(" = ")
                .map(|(_, r)| r)
                .unwrap_or(""),
            self.metric,
            self.dim,
            self.value,
            self.relative
        )
    }

    /// One row of the `docs/tolerance.md` table.
    fn markdown_row(&self) -> String {
        format!(
            "| {} | {} | {:.3e} | {} | {:.0} | {} | {} |",
            self.metric.as_str(),
            self.dim,
            self.value,
            if self.relative {
                "relative"
            } else {
                "absolute"
            },
            self.multiple,
            self.qdrant_spread.describe(),
            self.strawmann_spread.describe(),
        )
    }
}

/// §8.4 step 3: "ε := a small multiple of the larger of the two, per metric and
/// per dimension".
///
/// The multiple is 4 by default. It is a judgement call and is recorded in the
/// output rather than hidden: the point of §8.4 is that the number is
/// *derived and justified*, not that any particular multiple is correct.
pub const DEFAULT_MULTIPLE: f64 = 4.0;

pub fn calibrate(
    metric: Metric,
    dim: usize,
    qdrant_spread: Distribution,
    strawmann_spread: Distribution,
    multiple: f64,
) -> Epsilon {
    let relative = tolerance_kind(metric) == ToleranceKind::Relative;
    let q = if relative {
        qdrant_spread.max_relative
    } else {
        qdrant_spread.max
    };
    let s = if relative {
        strawmann_spread.max_relative
    } else {
        strawmann_spread.max
    };

    // §8.4 says a multiple of the larger of the two. The floor exists for the
    // degenerate case where both spreads measured exactly zero — which happens
    // when every ISA arm produced bit-identical results, and an ε of exactly 0
    // would then fail on the first legitimate rounding difference.
    let floor = expected_summation_error(dim);
    let value = (multiple * q.max(s)).max(floor);

    Epsilon {
        metric,
        dim,
        value,
        relative,
        qdrant_spread,
        strawmann_spread,
        multiple,
    }
}

/// The calibrated cells recorded in `docs/tolerance.md`, as data.
///
/// §8.4 step 3 puts the derived ε in `docs/tolerance.md`; a run that needs a
/// tolerance and is not handed one on the command line should reach for that
/// value, not for a round number. Until the differ reads the markdown table
/// back, the measured cells are mirrored here so both `differ` and
/// `relevance` default to the same calibrated ε for the same
/// `(metric, dim)`. `relevance --epsilon` used to default to `1e-5`, ~24× the
/// calibrated cosine/1536 value: the tie clause then accepted as "correct" a
/// returned point 24× further from the boundary than any measured kernel
/// noise, which flatters recall.
///
/// A new `calibrate` run must add its cell here as well as to the document.
pub const CALIBRATED: &[(Metric, usize, f64, bool)] = &[
    // docs/tolerance.md: cosine d=1536, 4 × max(qdrant 0, strawmann 2.384e-7).
    (Metric::Cosine, 1536, 9.537e-7, false),
    // docs/tolerance.md: euclid d=128, measured 2026-09-08 over 8 forced-ISA
    // arms and two Qdrant arms, n=30,000 scores. Both spreads came back
    // *exactly* zero, so `calibrate`'s floor clause applies and the value is
    // §8.4's `expected_summation_error(128)` — but as a measurement rather than
    // an absence, which is the difference between this cell existing and not.
    //
    // Zero is not luck and not a broken dump: SIFT descriptors are uint8, so a
    // squared euclid over d=128 is an integer at most 128·255² = 8,323,200
    // (observed max 126,546), every partial sum is exactly representable in
    // fp32 below 2²⁴ = 16,777,216, and a sum of exact integers is the same in
    // any order. §8.1's "different summation order gives different last bits"
    // is true of the general case and false of this corpus, so no arrangement
    // of accumulators — 4 or 8, SSE2 through AVX-512 — can disagree here.
    (Metric::Euclid, 128, 4.172e-7, true),
];

/// Where a default ε came from, so the run can print it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EpsilonSource {
    /// A measured cell from `docs/tolerance.md` (`CALIBRATED`).
    Calibrated,
    /// The §8.4 sanity floor: no measurement exists for this cell.
    Floor,
    /// `--epsilon` on the command line.
    Flag,
}

/// The ε a run should use when none is passed: the calibrated cell if
/// `docs/tolerance.md` has one for this `(metric, dim)`, else the §8.4 floor.
///
/// Returns the source alongside so the caller prints which one it got; a
/// floor silently standing in for a calibration is what §8.4 calls
/// "unprincipled".
pub fn default_epsilon(metric: Metric, dim: usize) -> (Epsilon, EpsilonSource) {
    let mut e = calibrate(
        metric,
        dim,
        Distribution::default(),
        Distribution::default(),
        DEFAULT_MULTIPLE,
    );
    for &(m, d, value, relative) in CALIBRATED {
        if m == metric && d == dim {
            e.value = value;
            e.relative = relative;
            return (e, EpsilonSource::Calibrated);
        }
    }
    (e, EpsilonSource::Floor)
}

/// A tolerance table, one entry per (metric, dim).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub entries: Vec<Epsilon>,
}

impl Table {
    /// Not yet called: `calibrate` currently writes `docs/tolerance.md` and the
    /// differ takes its ε from `--epsilon`. This is how a differ run will look
    /// the value up per (metric, dim) once the table is passed rather than a
    /// single number.
    #[allow(dead_code)]
    pub fn get(&self, metric: Metric, dim: usize) -> Option<&Epsilon> {
        self.entries
            .iter()
            .find(|e| e.metric == metric && e.dim == dim)
    }

    /// Render `docs/tolerance.md`.
    ///
    /// §8.4 requires the tolerance be "justified in `docs/tolerance.md` with
    /// the underlying distributions". Generating the document from the measured
    /// data is the only way that stays true as the data changes.
    pub fn to_markdown(&self) -> String {
        Self::render_markdown(
            &self
                .entries
                .iter()
                .map(Epsilon::markdown_row)
                .collect::<Vec<_>>(),
        )
    }

    /// `to_markdown`, keeping the rows an existing `docs/tolerance.md` holds
    /// for *other* `(metric, dim)` cells.
    ///
    /// `calibrate` measures one cell at a time and used to write the document
    /// from that one entry, so every calibration overwrote every other row.
    /// The existing rows are carried as text — the distributions they quote
    /// are not parsed back — and a row for a cell this table holds is
    /// replaced in place; new cells are appended.
    pub fn merge_markdown(&self, existing: &str) -> String {
        let cell_of = |row: &str| -> Option<(String, String)> {
            let mut cells = row.trim().trim_matches('|').split('|').map(str::trim);
            Some((cells.next()?.to_string(), cells.next()?.to_string()))
        };
        let mut rows: Vec<(Option<(String, String)>, String)> = existing
            .lines()
            .filter(|l| l.starts_with('|'))
            // The header and its separator, and any header-like row.
            .filter(|l| !l.starts_with("| metric") && !l.starts_with("|---"))
            .map(|l| (cell_of(l), l.to_string()))
            .collect();
        for e in &self.entries {
            let key = Some((e.metric.as_str().to_string(), e.dim.to_string()));
            let row = e.markdown_row();
            match rows.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = row,
                None => rows.push((key, row)),
            }
        }
        Self::render_markdown(&rows.into_iter().map(|(_, r)| r).collect::<Vec<_>>())
    }

    fn render_markdown(rows: &[String]) -> String {
        let mut s = String::new();
        s.push_str("# Calibrated tolerances (§8.4)\n\n");
        s.push_str("Derived, not chosen. Each ε is a multiple of the larger of two measured\n");
        s.push_str("spreads: Qdrant's disagreement with itself across its runtime-dispatched\n");
        s.push_str("kernels, and strawmann's across the §7.5 forced-ISA matrix.\n\n");
        s.push_str("| metric | dim | ε | kind | ×  | Qdrant spread | strawmann spread |\n");
        s.push_str("|---|--:|--:|---|--:|---|---|\n");
        for row in rows {
            s.push_str(row);
            s.push('\n');
        }
        s.push_str(
            "\nThese cells are mirrored in `conformance/src/differ/tolerance.rs` (`CALIBRATED`),\n",
        );
        s.push_str("which is what `differ` and `relevance` default to when `--epsilon` is not\n");
        s.push_str(
            "given; a new calibration must update both, and each run prints which ε it used.\n",
        );
        // Which cells are *absent* is the part a reader needs and the table
        // cannot show -- and, since 2026-09-08, so is the part where a cell is
        // present and equals the floor anyway. Both live here, in the generated
        // prose, because a hand-written note outside the table does not survive
        // the next `calibrate`.
        s.push_str("\nOnly the cells above are calibrated. Every other (metric, dim) falls back\n");
        s.push_str("to §8.4's computed floor, `expected_summation_error(dim)`. A floor is not a\n");
        s.push_str("calibration: it is what summation error *should* be at that width, not what\n");
        s.push_str("these two engines were measured to disagree by. `calibrate` is what turns\n");
        s.push_str("the one into the other, and each run says which of the two it used.\n");
        s.push_str("\n### euclid at d=128: measured, and measured at zero\n\n");
        s.push_str("The euclid/128 cell is SIFT1M, the corpus the published comparison runs\n");
        s.push_str("on, and its ε *equals* the floor — so the row looks like a floor and is\n");
        s.push_str("not one. Across eight forced-ISA arms (SSE2 through AVX-512, VNNI and\n");
        s.push_str("VPOPCNTDQ) and two Qdrant arms, over 30,000 scores, both cross-arm spreads\n");
        s.push_str("came back **exactly** zero, so `calibrate`'s floor clause set the value.\n\n");
        s.push_str("That zero is arithmetic rather than luck, and it is worth knowing before\n");
        s.push_str("anyone re-runs the calibration expecting a number. SIFT descriptors are\n");
        s.push_str("uint8, so a squared euclid over d=128 is an integer of at most\n");
        s.push_str("128·255² = 8,323,200 — the observed maximum is 126,546 — and every partial\n");
        s.push_str("sum is exactly representable in fp32 below 2²⁴ = 16,777,216. A sum of\n");
        s.push_str("exact integers is the same in any order, so no arrangement of accumulators\n");
        s.push_str("can disagree, and the final square root is a correctly-rounded operation on\n");
        s.push_str("an identical input. The distinction the cell buys is therefore the label\n");
        s.push_str("and not the number: `calibrated` says a measurement found the disagreement\n");
        s.push_str("to be nil, where `floor` says nobody looked.\n");
        s.push_str("\n## Why these are not `1e-6`\n\n");
        s.push_str("§8.1 is explicit that bit-exact equality with Qdrant is **not achievable**:\n");
        s.push_str(
            "its AVX path uses 4 accumulators over 32 floats with FMA, ours uses 8 with a\n",
        );
        s.push_str("different reduction tree, and different summation order gives different\n");
        s.push_str("last bits. That is not a defect in either engine, and it is a statement\n");
        s.push_str("about the general case: on a corpus whose distances are exact integers it\n");
        s.push_str("does not hold, which is what the euclid/128 row above measures. What *is*\n");
        s.push_str("achievable is value equality within a tolerance derived from how much each\n");
        s.push_str("engine already disagrees with itself — which is what this table records.\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_percentiles_are_nearest_rank() {
        let abs: Vec<f64> = (1..=100).map(f64::from).collect();
        let mags = vec![1.0; 100];
        let d = Distribution::from_samples(abs, &mags);
        assert_eq!(d.count, 100);
        assert_eq!(d.max, 100.0);
        assert_eq!(d.p99, 99.0);
        assert_eq!(d.p50, 50.0);
        // Nearest-rank never invents a value that was not measured.
        assert_eq!(d.p999, 100.0);
    }

    #[test]
    fn distribution_of_nothing_is_zero_not_a_panic() {
        let d = Distribution::from_samples(vec![], &[]);
        assert_eq!(d.count, 0);
        assert_eq!(d.max, 0.0);
    }

    #[test]
    fn relative_deltas_are_scaled_by_magnitude() {
        let abs = vec![1.0, 2.0];
        let mags = vec![100.0, 100.0];
        let d = Distribution::from_samples(abs, &mags);
        assert!((d.max_relative - 0.02).abs() < 1e-12);
        assert_eq!(d.max, 2.0);
    }

    #[test]
    fn cosine_uses_an_absolute_tolerance_and_dot_a_relative_one() {
        // §8.4: cosine scores live in [-1,1]; dot and euclid scale with
        // magnitude.
        assert_eq!(tolerance_kind(Metric::Cosine), ToleranceKind::Absolute);
        assert_eq!(tolerance_kind(Metric::Dot), ToleranceKind::Relative);
        assert_eq!(tolerance_kind(Metric::Euclid), ToleranceKind::Relative);
        assert_eq!(tolerance_kind(Metric::Manhattan), ToleranceKind::Relative);
    }

    #[test]
    fn expected_summation_error_matches_the_spec_magnitude() {
        // §8.4: "for normalised fp32 dot at d=768 ... an absolute delta around
        // 1e-6". log2(768) ≈ 9.58, times 2^-24 ≈ 5.96e-8, so ~5.7e-7 — the
        // same order of magnitude the spec quotes.
        let e = expected_summation_error(768);
        assert!(e > 1e-7 && e < 1e-5, "got {e}");
    }

    #[test]
    fn epsilon_is_a_multiple_of_the_larger_spread() {
        let q = Distribution::from_samples(vec![1e-6], &[1.0]);
        let s = Distribution::from_samples(vec![4e-6], &[1.0]);
        let e = calibrate(Metric::Cosine, 768, q, s, 4.0);
        // Larger spread is strawmann's 4e-6, times 4.
        assert!((e.value - 1.6e-5).abs() < 1e-12, "got {}", e.value);
        assert!(!e.relative);
        assert!(e.holds(1.5e-5, 1.0));
        assert!(!e.holds(2e-5, 1.0));
    }

    #[test]
    fn epsilon_never_collapses_to_zero() {
        // Both engines bit-identical across ISAs is a legitimate outcome; an ε
        // of exactly zero would then fail on the first rounding difference.
        let zero = Distribution::from_samples(vec![0.0, 0.0], &[1.0, 1.0]);
        let e = calibrate(Metric::Cosine, 768, zero.clone(), zero, 4.0);
        assert!(e.value > 0.0);
        assert_eq!(e.value, expected_summation_error(768));
    }

    #[test]
    fn relative_epsilon_scales_with_the_score_magnitude() {
        let q = Distribution::from_samples(vec![1.0], &[1000.0]); // rel 1e-3
        let s = Distribution::from_samples(vec![0.0], &[1000.0]);
        let e = calibrate(Metric::Dot, 128, q, s, 2.0);
        assert!(e.relative);
        // 2 x 1e-3 = 2e-3 relative.
        assert!(e.holds(1.0, 1000.0)); // 1e-3 <= 2e-3
        assert!(!e.holds(10.0, 1000.0)); // 1e-2 > 2e-3
    }

    #[test]
    fn table_lookup_is_per_metric_and_per_dim() {
        let d = Distribution::default();
        let t = Table {
            entries: vec![
                calibrate(Metric::Cosine, 768, d.clone(), d.clone(), 4.0),
                calibrate(Metric::Cosine, 1536, d.clone(), d.clone(), 4.0),
            ],
        };
        assert!(t.get(Metric::Cosine, 768).is_some());
        assert!(t.get(Metric::Cosine, 1536).is_some());
        assert!(t.get(Metric::Cosine, 128).is_none());
        assert!(t.get(Metric::Dot, 768).is_none());
        // Different dims get different ε, since summation error grows with d.
        assert_ne!(
            t.get(Metric::Cosine, 768).unwrap().value,
            t.get(Metric::Cosine, 1536).unwrap().value
        );
    }

    #[test]
    fn markdown_includes_the_distributions_that_justify_the_number() {
        let q = Distribution::from_samples(vec![1e-7, 2e-7], &[1.0, 1.0]);
        let s = Distribution::from_samples(vec![3e-7], &[1.0]);
        let t = Table {
            entries: vec![calibrate(Metric::Cosine, 768, q, s, 4.0)],
        };
        let md = t.to_markdown();
        assert!(md.contains("cosine"));
        assert!(md.contains("768"));
        // The justification, not just the number.
        assert!(md.contains("max="));
        assert!(md.contains("not achievable") || md.contains("bit-exact"));
    }

    /// The description is an equation, and it has to be the one that held:
    /// a floor-clamped ε is not "4x max(spreads)".
    #[test]
    fn describe_names_the_branch_that_produced_the_value() {
        let q = Distribution::from_samples(vec![1e-6], &[1.0]);
        let s = Distribution::from_samples(vec![4e-6], &[1.0]);
        let measured = calibrate(Metric::Cosine, 768, q, s, 4.0);
        assert!(
            measured
                .describe()
                .contains("= 4x max(qdrant 1.000e-6, strawmann 4.000e-6)"),
            "{}",
            measured.describe()
        );
        assert!(!measured.describe().contains("floor"));

        let zero = Distribution::from_samples(vec![0.0], &[1.0]);
        let floored = calibrate(Metric::Cosine, 768, zero.clone(), zero, 4.0);
        assert!(
            floored.describe().contains("floor"),
            "{}",
            floored.describe()
        );
        assert!(
            !floored.describe().contains("= 4x max"),
            "{}",
            floored.describe()
        );
        // And the literal for `CALIBRATED` is what a human would have typed.
        assert!(
            floored.rust_literal().contains("(Metric::Cosine, 768, "),
            "{}",
            floored.rust_literal()
        );
        assert!(
            floored.rust_literal().contains("false),"),
            "{}",
            floored.rust_literal()
        );
    }

    /// One calibration writes one cell; the others in the document survive.
    #[test]
    fn merging_into_the_document_replaces_only_the_matching_cell() {
        let d = Distribution::default();
        let existing = Table {
            entries: vec![
                calibrate(Metric::Cosine, 1536, d.clone(), d.clone(), 4.0),
                calibrate(Metric::Euclid, 128, d.clone(), d.clone(), 4.0),
            ],
        }
        .to_markdown();
        let q = Distribution::from_samples(vec![1e-6], &[1.0]);
        let fresh = Table {
            entries: vec![calibrate(Metric::Cosine, 1536, q, d.clone(), 4.0)],
        };
        let md = fresh.merge_markdown(&existing);
        let rows: Vec<&str> = md
            .lines()
            .filter(|l| l.starts_with("| ") && !l.starts_with("| metric"))
            .collect();
        assert_eq!(rows.len(), 2, "{md}");
        assert!(
            rows[0].starts_with("| cosine | 1536 | 4.000e-6 |"),
            "{}",
            rows[0]
        );
        assert!(rows[1].starts_with("| euclid | 128 |"), "{}", rows[1]);
        // A new cell is appended; nothing else moves.
        let more = Table {
            entries: vec![calibrate(Metric::Dot, 768, d.clone(), d, 4.0)],
        };
        let md = more.merge_markdown(&md);
        let rows: Vec<&str> = md
            .lines()
            .filter(|l| l.starts_with("| ") && !l.starts_with("| metric"))
            .collect();
        assert_eq!(rows.len(), 3, "{md}");
        assert!(rows[2].starts_with("| dot | 768 |"), "{}", rows[2]);
        // From nothing, it is `to_markdown`.
        assert_eq!(more.merge_markdown(""), more.to_markdown());
    }

    #[test]
    fn default_epsilon_prefers_the_documented_calibration_over_the_floor() {
        // docs/tolerance.md carries a measured cell for cosine/1536; a run
        // that is not handed --epsilon must use it, and must say so.
        let (e, src) = default_epsilon(Metric::Cosine, 1536);
        assert_eq!(src, EpsilonSource::Calibrated);
        assert!((e.value - 9.537e-7).abs() < 1e-12, "got {}", e.value);
        assert!(!e.relative);

        // euclid/128 is measured too, since 2026-09-08. Its value *equals*
        // the floor because both cross-arm spreads came back exactly zero --
        // which is the one case where the label matters more than the number:
        // the ε a run uses is identical either way, and only `Calibrated` says
        // a measurement found the disagreement to be nil rather than that
        // nobody looked.
        let (e, src) = default_epsilon(Metric::Euclid, 128);
        assert_eq!(src, EpsilonSource::Calibrated);
        // The cell records the floor to the four significant figures
        // `Epsilon::rust_literal` prints, which is the convention the cosine
        // cell above is already stored at -- so this is "the floor" to within
        // a display rounding rather than bit-for-bit.
        assert!(
            (e.value - expected_summation_error(128)).abs() < 1e-10,
            "got {}, want ~{}",
            e.value,
            expected_summation_error(128)
        );
        assert!(e.relative);

        // A cell nobody has measured is still the floor, labelled as such --
        // the behaviour this test was written for, on a pair that has none.
        let (e, src) = default_epsilon(Metric::Euclid, 960);
        assert_eq!(src, EpsilonSource::Floor);
        assert_eq!(e.value, expected_summation_error(960));
        assert!(e.relative);
    }
}
