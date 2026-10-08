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
| `sift1m-...-perf-1007-...-2026-10-06-2330` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. T4 with T3 passing (0.9888 against 0.9875), gate `pass` both arms on the first ask, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode on its compiled-in config, the same binary as 1006. At matched recall 1.59x to 2.41x over 8 anchors (1006: 1.40x to 2.30x), the top anchor higher on strawmANN's `ef` 512 build draw (0.9987 against 0.9980). The first pair with W4's client on the batched, quantized and filtered rows (`decisions.md`, 2026-10-06), so those rows are STALE against 1006 by design: W5 1.58x (from 5.68x, findings 70), SQ8 (W6) 1.77x, PQ (W8) 1.37x, TurboQuant 4-bit (W14) 1.69x. W12 runs Qdrant's 10,000 KB threshold and ACORN, under which Qdrant scans both tiers: sel1 1.95x, sel10 2.25x (findings 71). W11 covers 100% of its append on both engines, its search now ending on the clock (findings 3). W4 1.93x, W3 1.05x, W9 1.55x. |
| `h-and-m-2048-angular-filters-...-perf-0929-...-2026-09-29-0505` | The first h-and-m-2048-angular-filters comparison, and the source of `comparison-h-and-m-2048-angular-filters.md`: 105,100 x 2048 cosine product embeddings, the widest here, with our own fp64 ground truth (the oracle's cosine length in Qdrant's order, a9d8cf8). T4 with T3 passing (0.9988 against 0.9972), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, each engine's open-loop rows at its own saturation (`--rps-reference none`). At matched recall strawmANN serves 1.30x to 1.66x Qdrant's throughput; at equal `ef` it is slower above 64 (0.88x at 512) and more accurate at every `ef`. W4 reads 0.82x (findings 5), W3 1.30x, W5 2.15x, W6 1.33x, W8 1.30x; W7, W9 and both W12 rows parity. Licensed by a differ re-run at the rows' commit, `e149208`: the night's own T3 missed by 0.0001 on Qdrant's build draw (0.9969), which `decisions.md` keeps two-sided. |
| `laion-small-clip-...-perf-1007-...-2026-10-07-0300` | The current laion-small-clip comparison, and the source of `comparison-laion-small-clip.md`: 100k x 512 cosine CLIP embeddings with our own fp64 ground truth. T4 with T3 passing (0.9940 against 0.9911), gate `pass` both arms on the first ask, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode on its compiled-in config. At matched recall 1.57x to 1.92x over 8 anchors (1003: 1.53x to 1.81x). The first laion pair with W4's client on the batched, quantized and filtered rows, so those rows are STALE against 1003 by design: W5 0.88x (from 2.83x; Qdrant is the cheaper engine per batched query, findings 70), SQ8 (W6) 1.74x, binary (W7) 1.71x and its 2-bit and 1.5-bit encodings 1.67x and 1.68x, PQ (W8) 1.30x, TurboQuant (W14) 1.38x to 1.60x. W4 1.05x, W0 1.18x, W3 and W9 parity. **Every W12 row is refused (recall missing):** the rows searched with ACORN and the night's sweep without it, so the recall it measured does not describe them (findings 71, d226d92), and bench12 was gone before the fixed sweep existed. W11 covers 100% of its append on both engines, refused by design. W12-sel1-ef512 is also refused for one pass apart from two that agree. The page was re-rendered after d226d92, so it carries no W12 recall. |
| `dbpedia-openai-1m-...-perf-1008-...-2026-10-07-2107` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing (0.9667 against 0.9682), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, the same binary as 1006. At matched recall 1.06x to 1.36x (0930: 1.09x to 1.40x). The first dbpedia pair with W4's client on W5, the quantized rows and W12, and with W12 on Qdrant's shipped threshold and ACORN, so those rows are STALE against 0930 by design: W5 parity from 2.71x (findings 70), W7 1.72x, W6 1.13x, W8 1.21x (findings 72), W12-sel1 2.84x and sel10 3.04x (findings 71). W3 1.69x, W4 parity, W9 1.27x. W11 covered 100% of its append on both rows and engines (findings 3). |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. When the harness configuration changes between runs, `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`). The dbpedia-openai-1m pair `perf-0924` was removed on 2026-09-26 (last present at `69831f0`). Its successor `perf-0925`, measured on Qdrant's development profile, was removed on 2026-09-27 (last present at `24a2ad8`). The laion-small-clip pair `perf-0927`, measured before strawmANN's duplicate-vector fix, was removed on 2026-09-28 (last present at `c237ba5`). The sift1m pair `perf-0926` and the dbpedia-openai-1m pair `perf-0927`, both measured before the two engine fixes, were removed on 2026-09-29 (last present at `e149208`). The sift1m pair `perf-0929`, the laion-small-clip pair `perf-0928` and the dbpedia-openai-1m pair `perf-0929` were removed on 2026-09-30 (last present at `1fb2e43`). The laion-small-clip pair `perf-0930` was removed on 2026-10-03 (last present at `f346aea`). The sift1m pair `perf-0930` was removed on 2026-10-06 (last present at `ebb1c5a`). The dbpedia-openai-1m pair `perf-0930` was removed on 2026-10-08 (last present at `7be7ce5`).
