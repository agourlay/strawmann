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
| `dbpedia-openai-1m-...-perf-1110-...-2026-11-10-2305` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. **equal-work**, **pool** oversampling, `--perf`, Qdrant `dev` b4ebd966f (sha256 `8550a49c`, release, 1.20.1-dev), strawmANN 31fe897 on Zig 0.17.1, gate `pass` both arms on the first ask, 3 passes, T4 with T3 passing (0.9666 against 0.9692 at ef 128). STALE against 1008 by design (Qdrant pin and Zig moved). At matched recall 1.05x to 1.34x (1008: 1.06x to 1.36x). W3 1.74x, W4 and W5 parity, W7-2bit 1.64x, W8 1.21x to 1.24x, W9 1.17x, W12-sel1 2.81x, sel10 2.97x, sel10-ef512 2.53x (1008: 1.31x, now walked). SQ8 lost its lead: W6 parity (1008: 1.13x), ef256 and ef512 0.86x to 0.88x, Qdrant 13% to 19% faster on the new pin. W7 carries no ratio: one Qdrant pass ran 15% low. |
| `sift1m-...-perf-1109-...-2026-11-09-1913` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. **equal-work**, **pool** oversampling, `--perf`, Qdrant `dev` b4ebd966f (sha256 `8550a49c`, release, 1.20.1-dev), bfb d2a6b6c, strawmANN 515eeb4 on Zig 0.17.1, gate `pass` both arms on the first ask, 3 passes, T4 with T3 passing. STALE against 1102 by design (Qdrant pin and Zig moved). No matched-recall ratio is stated: one Qdrant W10-ef64 pass ran 25% low. W3 1.09x, W4 1.94x, W5 1.64x, SQ8 (W6) 1.78x, PQ (W8) 1.39x, W9 1.53x (1102: parity; strawmANN 192 q/s against 142), W12-sel10 2.43x; W12-sel1 withheld (Qdrant's passes spread 10.8%). |
| `laion-small-clip-...-perf-1109-...-2026-11-09-2246` | The current laion-small-clip comparison, and the source of `comparison-laion-small-clip.md`. **equal-work**, **pool** oversampling, `--perf`, Qdrant `dev` b4ebd966f (sha256 `8550a49c`, release, 1.20.1-dev), bfb d2a6b6c, strawmANN 515eeb4 on Zig 0.17.1, gate `pass` both arms on the first ask, 3 passes, T4 with T3 passing. STALE against 1102 by design. At matched recall 1.55x to 1.93x (1102: 1.43x to 1.86x). W12-sel10 1.68x (1102: 0.63x), the planner refit fe782b5. Qdrant leads W5 (0.86x); W3 1.05x, W4 1.03x, W6 1.69x, W7 1.74x, W8 1.36x, W14-2bit 1.58x, W9 parity, W12-sel1 1.18x. |
| `h-and-m-2048-angular-filters-...-perf-1109-...-2026-11-10-0125` | The current h-and-m-2048-angular-filters comparison, and the source of `comparison-h-and-m-2048-angular-filters.md`. **equal-work**, **pool** oversampling, `--perf`, Qdrant `dev` b4ebd966f (sha256 `8550a49c`, release, 1.20.1-dev), bfb d2a6b6c, strawmANN 515eeb4 on Zig 0.17.1, gate `pass` both arms on the first ask, 3 passes, T4 with T3 passing at matched recall (Qdrant at ef 256). STALE against 1102 by design. At matched recall 1.70x to 2.22x over 6 anchors. W12-sel10-ef512 1.18x (1102: 0.52x), the planner refit fe782b5. Qdrant leads W4 (0.83x) and W5 (0.79x); W3 1.32x, W6 parity, W6-ef64 1.13x, W7 1.53x, W8 1.32x, W9 parity, W12-sel10 1.25x; W12-sel10-ef64 withheld. |
| `scifact-...-perf-1111-...-2026-11-12-0108` | The current scifact BM25 comparison (W15), and the source of `comparison-scifact.md`. Qdrant `dev` b4ebd966f (sha256 `8550a49c`, release, 1.20.1-dev), bfb d2a6b6c, strawmANN 7874a23 on Zig 0.17.1, gate `pass` both arms on the first ask, 3 passes, T2 conformance, nothing refused. STALE against 1109 by design (strawmANN's BM25 late filter, f33ee1f). W15 2.25x, W15-sat 2.37x, filtered 2.16x, k0.9-b0.4 2.40x, b0 2.32x: flat, as a corpus this small predicts. |
| `fiqa-...-perf-1111-...-2026-11-12-0129` | The current FiQA BM25 comparison, the source of `comparison-fiqa.md`, same pins and conditions. W15 2.96x, W15-sat 5.70x (1109: 4.03x), **filtered 6.47x** (1109: 1.87x), k0.9-b0.4 5.23x, b0 4.67x. |
| `arguana-...-perf-1111-...-2026-11-12-0152` | The current ArguAna BM25 comparison, the source of `comparison-arguana.md`, same pins and conditions. W15 1.39x, W15-sat 2.17x, **filtered 1.97x** (1109: Qdrant led, 0.67x), k0.9-b0.4 1.91x, b0 1.98x. |
| `quora-...-perf-1111-...-2026-11-12-0218` | The current Quora BM25 comparison, the source of `comparison-quora.md`, same pins and conditions. W15 1.52x, W15-sat 2.11x, **filtered 3.89x** (1109: 2.51x), k0.9-b0.4 1.86x, b0 1.90x. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. When the harness configuration changes between runs, `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`). The dbpedia-openai-1m pair `perf-0924` was removed on 2026-09-26 (last present at `69831f0`). Its successor `perf-0925`, measured on Qdrant's development profile, was removed on 2026-09-27 (last present at `24a2ad8`). The laion-small-clip pair `perf-0927`, measured before strawmANN's duplicate-vector fix, was removed on 2026-09-28 (last present at `c237ba5`). The sift1m pair `perf-0926` and the dbpedia-openai-1m pair `perf-0927`, both measured before the two engine fixes, were removed on 2026-09-29 (last present at `e149208`). The sift1m pair `perf-0929`, the laion-small-clip pair `perf-0928` and the dbpedia-openai-1m pair `perf-0929` were removed on 2026-09-30 (last present at `1fb2e43`). The laion-small-clip pair `perf-0930` was removed on 2026-10-03 (last present at `f346aea`). The sift1m pair `perf-0930` was removed on 2026-10-06 (last present at `ebb1c5a`). The dbpedia-openai-1m pair `perf-0930` was removed on 2026-10-08 (last present at `7be7ce5`). The vector pairs `perf-1102` (sift1m, laion-small-clip, h-and-m-2048-angular-filters) and the BM25 pairs `perf-1031` (scifact, fiqa, arguana, quora) were removed on 2026-11-10, superseded by `perf-1109` on Qdrant b4ebd966f (last present at `7033047`). The dbpedia-openai-1m pair `perf-1008` was removed on 2026-11-11, superseded by `perf-1110` on Qdrant b4ebd966f (last present at `31fe897`). The BM25 pairs `perf-1109` (scifact, fiqa, arguana, quora) were removed on 2026-11-12, superseded by `perf-1111` with the late filter (last present at `7874a23`).
