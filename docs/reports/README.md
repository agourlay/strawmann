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
| `sift1m-...-perf-0929-...-2026-09-28-1619` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4 with T3 passing (0.9888 against 0.9873), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode. The first pair with both 2026-09-28 engine fixes, the duplicate-vector rule (c237ba5) and the quantized traversal's prefetch (f7c9ecb): PQ (W8) reads 1.08x where 0926 read 0.73x, binary (W7) 0.98x from 0.76x, SQ8 (W6) 0.94x from 0.90x. W4 1.65x, W5 4.91x, W9 1.63x, W12-sel1 1.50x. At matched recall 1.29x to 2.09x: strawmANN's recall at `ef` 512 drew 0.9980 this pass, under Qdrant's 0.9993, so 0926's top anchor (1.78x at 0.9993) cannot form and a new one does at 0.9980 (findings 34's per-upload draw, not a code change). No row in any pass came back contaminated. |
| `h-and-m-2048-angular-filters-...-perf-0929-...-2026-09-29-0505` | The first h-and-m-2048-angular-filters comparison, and the source of `comparison-h-and-m-2048-angular-filters.md`: 105,100 x 2048 cosine product embeddings, the widest here, with our own fp64 ground truth (the oracle's cosine length in Qdrant's order, a9d8cf8). T4 with T3 passing (0.9988 against 0.9972), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, each engine's open-loop rows at its own saturation (`--rps-reference none`). At matched recall strawmANN serves 1.30x to 1.66x Qdrant's throughput; at equal `ef` it is slower above 64 (0.88x at 512) and more accurate at every `ef`. W4 reads 0.82x (findings 5), W3 1.30x, W5 2.15x, W6 1.33x, W8 1.30x; W7, W9 and both W12 rows parity. Licensed by a differ re-run at the rows' commit, `e149208`: the night's own T3 missed by 0.0001 on Qdrant's build draw (0.9969), which `decisions.md` keeps two-sided. |
| `laion-small-clip-...-perf-0928-...-2026-09-28-0714` | The current laion-small-clip comparison, and the source of `comparison-laion-small-clip.md`: 100k x 512 cosine CLIP embeddings, float16 on disk, with our own fp64 ground truth. T4 with T3 passing, gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, open-loop rows at Qdrant's 0927 saturation (10,846 q/s, `auto`). The first pair after strawmANN's duplicate-vector fix (c237ba5): at matched recall it serves 1.63x to 1.85x Qdrant's throughput, and recall@10 at `ef` 512 reads 0.9988 against Qdrant's 0.9979, with no unreachable node in any build. W3 1.09x, W4 and W9 parity, W5 2.90x, W8 0.95x, W12-sel1 1.98x; W12-sel10 is refused for unequal recall (0.9972 against 0.9847). The 0927 page it replaces was measured with the sink the fix removes, and read 1.59x falling to 0.38x. |
| `dbpedia-openai-1m-...-perf-0929-...-2026-09-28-2051` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing (0.9665 against 0.9687), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode. The first pair with both 2026-09-28 engine fixes, the duplicate-vector rule (c237ba5) and the quantized traversal's prefetch (f7c9ecb): at matched recall 1.11x to 1.44x (0927: 1.01x to 1.32x); SQ8 (W6) 1.47x from 1.31x, PQ (W8) 1.18x from 0.96x, binary (W7) 1.06x. W3 1.71x, W4 and W9 parity, W5 2.71x. No row in any pass came back contaminated. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. When the harness configuration changes between runs, `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`). The dbpedia-openai-1m pair `perf-0924` was removed on 2026-09-26 (last present at `69831f0`). Its successor `perf-0925`, measured on Qdrant's development profile, was removed on 2026-09-27 (last present at `24a2ad8`). The laion-small-clip pair `perf-0927`, measured before strawmANN's duplicate-vector fix, was removed on 2026-09-28 (last present at `c237ba5`). The sift1m pair `perf-0926` and the dbpedia-openai-1m pair `perf-0927`, both measured before the two engine fixes, were removed on 2026-09-29 (last present at `e149208`).
