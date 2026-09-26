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
| `sift1m-...-perf-0926-...-2026-09-25-2021` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4 with T3 passing, gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling (every quantized row sends `ef / limit`, so both engines rescore `ef` candidates), `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in **production mode**. It is the first pair measured that way: every earlier page ran Qdrant's development profile (findings 12), whose `max_search_threads: 4` capped its saturating rows, so W4 reads 1.65x where 09-23 read 2.24x. W8 (PQ) is licensed for the first time, at 0.73x. No row in any pass came back contaminated. The 09-23 page it replaces is in git history. |
| `dbpedia-openai-1m-...-perf-0925-...-2026-09-24-2152` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing (0.9665 against 0.9684), gate `pass` both arms, 3 passes, **equal-work**, **matched** oversampling, `--perf`, and the same Qdrant binary as the 09-24 page (sha256 `dbeb0f73dea2d371`); its checkout had moved to `2874d0f1d`, which that binary predates, so the sha256 is the identity that holds. No row in any pass came back contaminated and none folded as drift. It is the first page with ACORN-1 filtered search and trusted postings (W12-sel1 1,873 q/s, W12-sel10 811), bench1 dropped before W2 (Qdrant's time-to-green 303 s against 658 s), and the SQ8 sweep straight after W6 (no major faults). Its W11 rows append 10x slower than 09-24's (200 and 500 points/s), so they are a different measurement; the settle discipline changed as well, so `compare.py` refuses ratios between the two pages as STALE. |
| `dbpedia-openai-1m-...-perf-0924-...-2026-09-23-1816` | The pair `comparison-dbpedia-openai-1m.md` was written from until 2026-09-25, and the first licensed comparison at d=1536 and 1M. T4 with T3 passing, the first licensed comparison at that width and scale. Gate `pass` both arms, 3 passes, **equal-work**, **matched** oversampling, `--perf`, Qdrant's release binary at `878843e6e` (sha256 `dbeb0f73dea2d371`), bfb at the pin. No row in any pass came back contaminated, and 36 of 43 rows per engine carry a measured spread. Qdrant's SQ8 recall controls at `ef` 32 and 64 paged the collection back in during passes 1 and 2 and are refused as drift; the cause is in `findings.md`. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. Between `perf-0924` and `perf-0925` the settle discipline in the harness stamp changed, and W11's append rate with it, so `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`).
