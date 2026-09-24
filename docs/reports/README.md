# Archived run reports

The page `bench/harness/report.py` renders, kept for the runs whose numbers this
repository publishes. `bench/results/` is gitignored and every label directory
in it is disposable; these are not, because each is the only rendering of a
comparison the docs quote.

A page is self-contained, no network, and states its own licence in the
header: the §7.1 gate verdict per arm, the conformance tier and hash, the
environment hash, the pass count, and a banner on every row the comparison
refuses. Read that before quoting a number off it. At ~5 MB the GitHub blob view
shows source rather than rendering, so read them at
<https://agourlay.github.io/strawmann/reports/>, which serves this directory
over Pages with Jekyll disabled.

Naming is `report-<dataset>-<a>-vs-<b>-<day>-<hhmm>`, from `report.default_out`:
a day holds more than one run of one pair.

| page | what it is |
|---|---|
| `sift1m-...-perf-0923-...-2026-09-23-0230` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4, gate `pass` both arms, 3 passes, **equal-work**, `--perf`, and the same Qdrant binary as the 09-21 page (sha256 `9508155702721ac9`); its `run.json` names checkout `63c6a797d229-dirty`, which that binary predates, so the sha256 is the identity that holds. One row needed a contamination retry (Qdrant W4-sat90, pass 2) and came back clean, so no folded row carries foreign load, and 36 of 43 rows per engine carry a measured spread, the seven that do not being the ingest and index-build rows. The drift that left W11, W11-steady and W12-sel10-ef32 unbanded on 09-21 did not recur. It is the first page to record `oversampling_policy` (`defaults`), so `compare.py` refuses ratios between it and the 09-21 page as STALE, although both measured Qdrant's defaults. |
| `sift1m-...-rel-0921-...-2026-09-21-1916` | The pair the front page quoted until 2026-09-23. T4, gate `pass` both arms, 3 passes, **equal-work**, `--perf`, Qdrant's release binary. No row in any pass came back contaminated, and 36 of 43 rows per engine carry a measured spread; the three that do not are W11, W11-steady and W12-sel10-ef32, whose qps moved monotonically across the passes, which `aggregate.py` reads as a trend and refuses to band. Qdrant was rebuilt since the 09-08 page (1.19.2-dev, different sha256), and the saturating ratio came back at the same 2.20x, so the two pages are a reproducibility check across that rebuild. |
| `sift1m-...-rel-0908-...-2026-09-08-1805` | The pair the front page quoted until 2026-09-21. T4, gate `pass` both arms, 3 passes, **equal-work**, `--perf`, Qdrant's release binary. The first page to carry W12's two selectivity grades with recall measured under their conditions, block-layer `disk read/write ops` for both engines rather than `unknown`, and a *calibrated* euclid/128 ε rather than §8.4's floor. Its W11 rows measure a concurrent write rather than the rebuild an unthrottled append provoked, so they are not comparable with the rows above them. |
| `sift1m-...-rel-0907-...-2026-09-07-0730` | The pair the front page quoted until 2026-09-08. T4, gate `pass` both arms, 3 passes, **as-deployed**, `--perf`, Qdrant's release binary, so it and the page above are the two Qdrant experiments of §7.4 rather than two measurements of one, and no ratio crosses them. |
| `sift1m-...-rel-0903-...-2026-09-03-1652` | The pair the front page quoted until 2026-09-07, and `regression.py`'s baseline. Same two binaries as above, so the two pages are a reproducibility check rather than two measurements. |
| `dbpedia-openai-1m-...-perf-0924-...-2026-09-23-1816` | The current dbpedia-openai-1m comparison, the headline tier at d=1536 and 1M, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing, the first licensed comparison at that width and scale. Gate `pass` both arms, 3 passes, **equal-work**, **matched** oversampling, `--perf`, Qdrant's release binary at `878843e6e` (sha256 `dbeb0f73dea2d371`), bfb at the pin. No row in any pass came back contaminated, and 36 of 43 rows per engine carry a measured spread. Qdrant's SQ8 recall controls at `ef` 32 and 64 paged the collection back in during passes 1 and 2 and are refused as drift; the cause is in `findings.md`. |
| `dbpedia-openai-1m-...-rel-0903-...-2026-09-03-1952` | The pair the headline tier had until 2026-09-24. Its differ reached only T2, and the reason is now measured: the differ built its own Qdrant collection at Qdrant's default segment count, four populated graphs against strawmANN's one, while the rows ran `equal-work`. The rows' recall sweeps match the 09-24 page within 0.003 at every `ef` (`decisions.md`, 2026-09-24). No per-dataset document was written from it. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. Between `rel-0907` and `rel-0908` the bfb pin moved
(`0c1aafee` to `fc6632e5`), W12 became two rows on a differently-uploaded
`bench12`, its query count went from 5,000 to 50,000, W11's append became
throttled, and Qdrant was rebuilt twice, so `compare.py` refuses ratios across
them as STALE, and it is right to. Between `rel-0921` and `perf-0923` only the
strawmANN build and the recorded oversampling policy changed, and the refusal
holds there too. Read each page against itself.
