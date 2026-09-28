# Archived run reports

The page `bench/harness/report.py` renders, kept for the runs whose numbers this
repository publishes. `bench/results/` is gitignored and every label directory
in it is disposable; these are not, because each is the only rendering of a
comparison the docs quote.

A page is self-contained, no network, and states its own licence in the
header: the §7.1 gate verdict per arm, the conformance tier and hash, the
environment hash, the pass count, and a banner on every row the comparison
refuses. Read that before quoting a number off it. At ~5 MB the GitHub blob view
shows source rather than rendering, so read them from
<https://agourlay.github.io/strawmann/>, the landing page in `docs/index.html`;
Pages serves `docs/` with Jekyll disabled.

Naming is `report-<dataset>-<a>-vs-<b>-<day>-<hhmm>`, from `report.default_out`:
a day holds more than one run of one pair.

The landing page, `docs/index.html`, has one results panel per dataset, written
by `compare.py --write-readme` as the README's table is. A panel links its
pair's page here once the page is published: copy it in, then run
`--write-readme` for that pair again.

| page | what it is |
|---|---|
| `sift1m-...-perf-0926-...-2026-09-25-2021` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4 with T3 passing, gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling (every quantized row sends `ef / limit`, so both engines rescore `ef` candidates), `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in **production mode**. It is the first pair measured that way: every earlier page ran Qdrant's development profile (findings 12), whose `max_search_threads: 4` capped its saturating rows, so W4 reads 1.65x where 09-23 read 2.24x. W8 (PQ) is licensed for the first time, at 0.73x. No row in any pass came back contaminated. The 09-23 page it replaces is in git history. |
| `laion-small-clip-...-perf-0927-...-2026-09-27-1720` | The first laion-small-clip comparison, and the source of `comparison-laion-small-clip.md`: 100k x 512 cosine CLIP embeddings, float16 on disk, with our own fp64 ground truth. T4 with T3 passing, gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode. Each engine's open-loop rows are fractions of its own saturation (`--rps-reference none`), so the latency percentiles are not compared across engines. At matched recall strawmANN serves 1.59x down to 0.38x: its recall stops at 0.9875 (Qdrant 0.9978, findings 54). W3, W4 and W9 read parity, W5 2.95x, W8 0.90x, W12-sel1 1.83x, W12-sel10 0.74x. Licensed by a differ re-run at the rows' commit, `2b24c9c`, after the night's own was refused by a port check reading a pid as a port (5454c5f). |
| `dbpedia-openai-1m-...-perf-0927-...-2026-09-26-2207` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing (0.9665 against 0.9679), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in **production mode**, the first dbpedia pair measured that way. At matched recall strawmANN serves 1.01x to 1.32x Qdrant's throughput, where the development-profile 0925 page read 1.26x to 1.44x; W4 reads parity (3,549 against 3,487), W3 1.75x. SQ8's bounds hold at this width for the first time (T4 `|Δscore|` p50 9.7e-4, from 4.9e-1), and W7 (binary, parity) and W8 (PQ, 0.96x) are licensed under `pool`. No row in any pass came back contaminated. Qdrant's SQ8 control at `ef` 128 drifted -12% across passes and is refused; both W11 rows are refused by §8, as mixed rows always are. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. When the harness configuration changes between runs, `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`). The dbpedia-openai-1m pair `perf-0924` was removed on 2026-09-26 (last present at `69831f0`). Its successor `perf-0925`, measured on Qdrant's development profile, was removed on 2026-09-27 (last present at `24a2ad8`).
