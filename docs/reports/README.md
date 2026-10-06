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
| `sift1m-...-perf-1006-...-2026-10-05-1702` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4 with T3 passing (0.9888 against 0.9874), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode on its compiled-in config, within 2% of 0930 on every search row. At matched recall 1.40x to 2.30x over 8 anchors (0930: 1.81x to 2.16x), the top anchor lower on strawmANN's `ef` 512 build draw (0.9980 against 0.9995). strawmANN's fp32 rows are 15 to 21% faster than 0930 at the same recall (W4 1.96x, W5 5.68x, W3 parity), the first sift pair with the 64-byte-aligned kernels (`decisions.md`, 2026-10-03). SQ8 (W6) 1.17x, PQ (W8) 1.05x, binary (W7) 0.96x and its 2-bit and 1.5-bit encodings 1.07x, TurboQuant (W14) 0.94x to 1.10x, W9 1.56x, W12-sel1 1.59x. W11 covered 52% of its append on strawmANN, which ran 2.25x faster than the sizing assumed (findings 3). |
| `h-and-m-2048-angular-filters-...-perf-0929-...-2026-09-29-0505` | The first h-and-m-2048-angular-filters comparison, and the source of `comparison-h-and-m-2048-angular-filters.md`: 105,100 x 2048 cosine product embeddings, the widest here, with our own fp64 ground truth (the oracle's cosine length in Qdrant's order, a9d8cf8). T4 with T3 passing (0.9988 against 0.9972), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, each engine's open-loop rows at its own saturation (`--rps-reference none`). At matched recall strawmANN serves 1.30x to 1.66x Qdrant's throughput; at equal `ef` it is slower above 64 (0.88x at 512) and more accurate at every `ef`. W4 reads 0.82x (findings 5), W3 1.30x, W5 2.15x, W6 1.33x, W8 1.30x; W7, W9 and both W12 rows parity. Licensed by a differ re-run at the rows' commit, `e149208`: the night's own T3 missed by 0.0001 on Qdrant's build draw (0.9969), which `decisions.md` keeps two-sided. |
| `laion-small-clip-...-perf-1003-...-2026-10-03-0654` | The current laion-small-clip comparison, and the source of `comparison-laion-small-clip.md`: 100k x 512 cosine CLIP embeddings with our own fp64 ground truth. T4 with T3 passing (0.9939 against 0.9904), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, within 1.8% of 0930 on every search row. At matched recall 1.53x to 1.81x (0930: 1.62x to 1.92x): strawmANN's fp32 graph rows lost 5 to 6% to code placement, so W4 reads parity from 1.04x (`decisions.md`, 2026-10-03; fixed after this pair). SQ8 (W6) 1.24x from parity (f557a37, e8be580), binary (W7) 1.25x, its 2-bit and 1.5-bit encodings 1.18x and 1.22x, PQ (W8) 1.19x, TurboQuant (W14) parity at 1 bit and 1.07x to 1.19x above, W5 2.83x, W3 and W9 parity. The first laion pair whose W11 search covers the whole append on both engines (findings 3); refused by design. W6-ef512, W12-sel1 and W12-sel1-ef256 are refused for one pass apart from two that agree. |
| `dbpedia-openai-1m-...-perf-0930-...-2026-09-29-2134` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing (0.9665 against 0.9685), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, Qdrant within about 2% of 0929 on every search row. At matched recall 1.09x to 1.40x (0929: 1.11x to 1.44x). Exact search (W9) 1.33x from parity with the 4 GB scan cap (99d2776), though its p99 rose to 1.5 s; PQ (W8) 1.34x from 1.18x, binary (W7) 1.16x, SQ8 (W6) 1.50x, W3 1.70x, W4 1.02x, W5 2.71x. The first pair with the tail drainer (1fb2e43): W11 reads 251 q/s against Qdrant's 99, refused by design and measured over the append. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. When the harness configuration changes between runs, `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`). The dbpedia-openai-1m pair `perf-0924` was removed on 2026-09-26 (last present at `69831f0`). Its successor `perf-0925`, measured on Qdrant's development profile, was removed on 2026-09-27 (last present at `24a2ad8`). The laion-small-clip pair `perf-0927`, measured before strawmANN's duplicate-vector fix, was removed on 2026-09-28 (last present at `c237ba5`). The sift1m pair `perf-0926` and the dbpedia-openai-1m pair `perf-0927`, both measured before the two engine fixes, were removed on 2026-09-29 (last present at `e149208`). The sift1m pair `perf-0929`, the laion-small-clip pair `perf-0928` and the dbpedia-openai-1m pair `perf-0929` were removed on 2026-09-30 (last present at `1fb2e43`). The laion-small-clip pair `perf-0930` was removed on 2026-10-03 (last present at `f346aea`). The sift1m pair `perf-0930` was removed on 2026-10-06 (last present at `ebb1c5a`).
