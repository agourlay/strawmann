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
| `sift1m-...-rel-0908-...-2026-09-08-1805` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4, gate `pass` both arms, 3 passes, **equal-work**, `--perf`, Qdrant's release binary. The first page to carry W12's two selectivity grades with recall measured under their conditions, block-layer `disk read/write ops` for both engines rather than `unknown`, and a *calibrated* euclid/128 ε rather than §8.4's floor. Its W11 rows measure a concurrent write rather than the rebuild an unthrottled append provoked, so they are not comparable with the rows above them. |
| `sift1m-...-rel-0907-...-2026-09-07-0730` | The pair the front page quoted until 2026-09-08. T4, gate `pass` both arms, 3 passes, **as-deployed**, `--perf`, Qdrant's release binary, so it and the page above are the two Qdrant experiments of §7.4 rather than two measurements of one, and no ratio crosses them. |
| `sift1m-...-rel-0903-...-2026-09-03-1652` | The pair the front page quoted until 2026-09-07, and `regression.py`'s baseline. Same two binaries as above, so the two pages are a reproducibility check rather than two measurements. |
| `dbpedia-openai-1m-...-rel-0903-...-2026-09-03-1952` | The headline tier, d=1536 at 1M. Its differ reached only T2, so §8 licenses no ratio and there is no `comparison-dbpedia-openai-1m.md`; the per-engine numbers are here. |
| `dbpedia-openai-1m-strawmann-vs-qdrant-2026-08-27-2309` | The only rendering of the run at labels `strawmann`/`qdrant`, whose rows are still on disk. It never got a per-dataset document, because it carries no `conformance.json` and §8 licenses no comparison without one; `readme-table-current` reports that rather than failing. A licensed dbpedia-openai-1m run retires this page. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. Between `rel-0907` and `rel-0908` the bfb pin moved
(`0c1aafee` to `fc6632e5`), W12 became two rows on a differently-uploaded
`bench12`, its query count went from 5,000 to 50,000, W11's append became
throttled, and Qdrant was rebuilt twice, so `compare.py` refuses ratios across
them as STALE, and it is right to. Read each page against itself.
