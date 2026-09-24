# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a ratio on the headline page

The d=1536 comparison is licensed since 2026-09-24 (`sm/qd-dbp1m-perf-0924`,
T4, 1.29x to 1.40x at matched recall; the account is in `decisions.md`). What
that page still refuses, and could not:

**1. The SQ8 recall controls page the collection back in on the Qdrant arm.**
`W6-ef32` folded at a 41% spread and `W6-ef64` is refused as drift (+19%,
monotone across passes). The rows carry the cause and the fold does not read
it: Qdrant's passes 1 and 2 took 152,697 and 254,138 major faults on `W6-ef32`
(9.7k and 17.8k on `ef64`, 5.4k and 8.9k on `ef128`) and pass 3 took one.
`bench6` is evicted by `W7-upload`, `W8-upload` and `W9`'s 6 GB scan before the
`W6-ef` sweep reaches it, so the first rows of the sweep measure a page-in on
a row that claims `cached` residency, and `aggregate._rep_drift` reads the
monotone slide as a trend. strawmANN's arm pins its arena and did not page.
Two ratios refused on the headline page. Either mark and drop a pass whose
`major_faults` on a cached search row exceed a floor, or order the `W6-ef`
sweep before the scan that evicts it; the second changes the row set and its
stamp.

**2. W11's append spans are sift1m's.** `W11_STEADY_SPAN_S` 25 s and
`W11_SPAN_S` 60 s were sized for d=128. At d=1536 the search covered 12 to 23%
of the append on every pass of both engines, so both mixed rows measured the
rebuild the append provoked rather than a concurrent write, as their own note
says, and both are refused. Derive the span from the corpus: the previous
pair's W3 latency times the query count, which `resolve_rps_reference`
already reads.

**3. `matched` oversampling matches SQ8 and nothing else.** `W6` is licensed
at 1.29x under the policy; `W7` (0.9474 against 0.8887) and `W8` (0.9656
against 0.9428) stay refused because Qdrant's binary-quantized recall is flat
in `ef` (0.838 at 32, 0.906 at 512) where strawmANN's rises to 0.986, so
oversampling 4 on both is not a matched pool at d=1536. Either a per-encoding
oversampling for Qdrant, or `ef` sweeps on `bench7` and `bench8` so the
matched-recall interpolation can cover them.

### P2. Cost of the run, and one engine question

**4. Filtered search is the licensed loss.** `W12-sel10` is flat at 166 qps
from `ef` 64 with 45 MB of DRAM and 29.5M instructions per query and recall
1.0000 at every `ef`: strawmANN scores the 20,054 matches directly. Qdrant's
curve moves with `ef` (2,226 to 288 qps), so it walks the graph, and at
matched recall strawmANN reads 0.34x to 0.59x. `docs/workloads.md` quotes
~14,800 q/s for the `sel1` scan and the row measures 1,221. Read the dispatch
in `handlers.searchOne` against the 10 KB `--full-scan-threshold` bfb passes,
and correct the doc or the engine.

**5. Qdrant's `W1` settle hits its 180 s timeout on every pass.** `W1`
uploads with `--skip-wait-index`; Qdrant indexes in the background and
`settle_engine` waits for it, nine minutes per Qdrant arm on a row that does
not measure the index. Skip the settle for `--skip-wait-index` rows, or accept
it and say so in the row.

**6. The pre-run estimate was three hours short.** `estimated_minutes` prices
the passes from the previous run's rows, and the table grew from 32 to 43 rows
between `rel-0903` and `perf-0924` (the `W12` selectivity grades and their `ef`
controls). Scale the basis by the rows present, and print the settle count the
run will actually make (10 here, against `12 x <= 300s`).
