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
| `dbpedia-openai-1m-...-perf-1008-...-2026-10-07-2107` | The current dbpedia-openai-1m comparison, and the source of `comparison-dbpedia-openai-1m.md`. T4 with T3 passing (0.9667 against 0.9682), gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371` in production mode, the same binary as 1006. At matched recall 1.06x to 1.36x (0930: 1.09x to 1.40x). The first dbpedia pair with W4's client on W5, the quantized rows and W12, and with W12 on Qdrant's shipped threshold and ACORN, so those rows are STALE against 0930 by design: W5 parity from 2.71x (findings 70), W7 1.72x, W6 1.13x, W8 1.21x (findings 72), W12-sel1 2.84x and sel10 3.04x (findings 71). W3 1.69x, W4 parity, W9 1.27x. W11 covered 100% of its append on both rows and engines (findings 3). |
| `sift1m-...-perf-1012-...-2026-10-12-1858` | The current sift1m comparison, and the source of the tables in `README.md` and `comparison-sift1m.md`. Gate `pass` both arms on the first ask, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371`, the same binary as 1007; bfb moves to 96fd5f0 (dependency update), so the pair is STALE against 1007 by design, which shows on client-bound rows (W6-ef32 +28% and W13 +10% for strawmANN). At matched recall 1.34x to 2.40x over 8 anchors. W3 1.06x, W9 1.56x, W5 1.64x, SQ8 (W6) 1.77x, PQ (W8) 1.41x. **W4's ratio is refused:** one strawmANN pass read -14% against two that agree (1007: 1.93x). |
| `laion-small-clip-...-perf-1012-...-2026-10-12-2245` | The current laion-small-clip comparison, and the source of `comparison-laion-small-clip.md`. Gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371`, bfb 96fd5f0. The first laion pair whose W12 rows license (with ACORN and the fixed sweep, d226d92): Qdrant leads W12-sel10 (0.64x) and its ef 128 rung (0.62x), where strawmANN's planner scans; sel1 parity to 1.05x. W5 0.89x, W6 1.73x, W7 1.72x, W8 1.39x, W14 1.38x to 1.58x, W4 parity, W3 and W9 parity. |
| `h-and-m-2048-angular-filters-...-perf-1012-...-2026-10-13-0124` | The current h-and-m-2048-angular-filters comparison, and the source of `comparison-h-and-m-2048-angular-filters.md`. Gate `pass` both arms, 3 passes, **equal-work**, **pool** oversampling, `--perf`, Qdrant sha256 `dbeb0f73dea2d371`, bfb 96fd5f0; the first h-and-m run to exit 0. STALE against 0929 by design (W4's client on W5 to W14, the W12 threshold and ACORN, the bfb pin). W4 0.82x and W5 0.78x (Qdrant leads), W3 1.29x, W6 1.18x, W7 1.53x, W8 1.34x, W9 parity, W12-sel10 1.25x, sel1 parity. One outlying W10 point refuses the matched-recall frontier. |
| `scifact-...-perf-1014-...-2026-10-14-0001` | The current scifact BM25 comparison (W15), and the source of `comparison-scifact.md`. Qdrant `dev` d6d3bd35b (sha256 `0e0fbc18`, release, with #11113) serving one optimized segment, bfb d2a6b6c, strawmANN 1bb4b85 with b02d08c and eb5c411, T2 conformance, gate `pass` both arms on the first ask, 3 passes, nothing refused. STALE against 1012 by design. W15 1.53x, W15-sat 2.48x, filtered 2.36x, k0.9-b0.4 2.20x, b0 2.37x; the saturating rows are client-bound for strawmANN. |
| `fiqa-...-perf-1014-...-2026-10-14-0024` | The current FiQA BM25 comparison, the source of `comparison-fiqa.md`, same pins and conditions. W15 1.64x, W15-sat 1.96x, filtered 1.61x (strawmANN +27% from eb5c411), k0.9-b0.4 1.63x, b0 1.52x. |
| `arguana-...-perf-1014-...-2026-10-14-0047` | The current ArguAna BM25 comparison, the source of `comparison-arguana.md`, same pins and conditions. Long queries: W15 5.50x, W15-sat 6.27x, filtered 3.68x, k0.9-b0.4 6.00x, b0 6.30x. |
| `quora-...-perf-1014-...-2026-10-14-0124` | The current Quora BM25 comparison, the source of `comparison-quora.md`, same pins and conditions. strawmANN leads again (1012: Qdrant led every saturating row): W15-sat 1.44x from b02d08c's posting lengths (19,927 against 10,994 q/s for strawmANN), filtered 2.37x with eb5c411's dense bitsets (16,011 against 4,996), k0.9-b0.4 1.26x, b0 1.23x, W15 1.21x. |

Only gated, licensed runs belong here. A `--lax` run or a smoke test says on its
own page that it may not be quoted, so archiving one would add a page that
exists to be disregarded.

The pages are not a series. When the harness configuration changes between runs, `compare.py` refuses ratios across them as STALE, and it is right to. The sift1m and dbpedia-openai-1m pages are different corpora at different widths. Read each page against itself.

Superseded pages are retired rather than kept: the sift1m pairs `rel-0903`, `rel-0907`, `rel-0908` and `rel-0921` and the dbpedia-openai-1m pair `rel-0903` were removed on 2026-09-25 and are in git history (last present at `feb03d9`). The dbpedia-openai-1m pair `perf-0924` was removed on 2026-09-26 (last present at `69831f0`). Its successor `perf-0925`, measured on Qdrant's development profile, was removed on 2026-09-27 (last present at `24a2ad8`). The laion-small-clip pair `perf-0927`, measured before strawmANN's duplicate-vector fix, was removed on 2026-09-28 (last present at `c237ba5`). The sift1m pair `perf-0926` and the dbpedia-openai-1m pair `perf-0927`, both measured before the two engine fixes, were removed on 2026-09-29 (last present at `e149208`). The sift1m pair `perf-0929`, the laion-small-clip pair `perf-0928` and the dbpedia-openai-1m pair `perf-0929` were removed on 2026-09-30 (last present at `1fb2e43`). The laion-small-clip pair `perf-0930` was removed on 2026-10-03 (last present at `f346aea`). The sift1m pair `perf-0930` was removed on 2026-10-06 (last present at `ebb1c5a`). The dbpedia-openai-1m pair `perf-0930` was removed on 2026-10-08 (last present at `7be7ce5`).
