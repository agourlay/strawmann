# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The d=1536 comparison is licensed since 2026-09-24 (`sm/qd-dbp1m-perf-0924`,
T4, 1.29x to 1.40x at matched recall; the account is in `decisions.md`). What
that page publishes wrongly, or still refuses and could not:

**1. strawmANN's SQ8 path returns a score that is not the fp32 score.** T4 on
2026-09-24: `sq8/strawmann |Δscore| p50=4.866e-1` against the fp64 oracle with
`rescore=true`, Kendall τ 0.6628; Qdrant's SQ8 reads `1.943e-3` and τ 0.7935 on
the same queries. Identical on 2026-09-03, so it is the engine and not the
run. Recall@10 is equal to Qdrant's at every `ef` and `quantization_dominance`
holds, so the candidate set is right and the tier passes; the score field a
client receives is off by half a cosine unit. Either the rescore is not what
lands in the response, or fewer candidates are rescored than the pool claims.
A wrong published value on every SQ8 query costs more than any refused ratio.

**2. The SQ8 recall controls page the collection back in on the Qdrant arm.**
`W6-ef32` folded at a 41% spread and `W6-ef64` is refused as drift (+19%,
monotone across passes). The rows carry the cause and the fold does not read
it: Qdrant's passes 1 and 2 took 152,697 and 254,138 major faults on `W6-ef32`
(9.7k and 17.8k on `ef64`, 5.4k and 8.9k on `ef128`) and pass 3 took one.
`bench6` is evicted by `W7-upload`, `W8-upload` and `W9`'s 6 GB scan before the
`W6-ef` sweep reaches it, so the first rows of the sweep measure a page-in on
a row that claims `cached` residency, and `aggregate._rep_drift` reads the
monotone slide as a trend. strawmANN's arm pins its arena and took nine.
Two ratios refused on the headline page. Either mark and drop a pass whose
`major_faults` on a cached search row exceed a floor, or order the `W6-ef`
sweep before the scan that evicts it; the second changes the row set and its
stamp.

**3. W11's append spans are sift1m's.** `W11_STEADY_SPAN_S` 25 s and
`W11_SPAN_S` 60 s were sized for d=128. At d=1536 the search covered 12 to 23%
of the append on every pass of both engines, so both mixed rows measured the
rebuild the append provoked rather than a concurrent write, as their own note
says, and both are refused. Derive the span from the corpus: the previous
pair's W3 latency times the query count, which `resolve_rps_reference`
already reads. Item 8 is what the row will show once it measures what it
claims to.

**4. `matched` oversampling matches SQ8 and nothing else.** `W6` is licensed
at 1.29x under the policy; `W7` (0.9474 against 0.8887) and `W8` (0.9656
against 0.9428) stay refused. The sweeps carry the signature findings 42 named:
Qdrant's recall@1 climbs to 0.978 (binary) and 0.989 (PQ) at `ef` 512 while
its recall@10 flattens at 0.906 and 0.963, which is a rescore pool of `limit`
times oversampling, 40 and 20 candidates whatever `ef` is. strawmANN's pool is
`ef`-sized and its recall@10 rises to 0.986 and 0.989. The policy sends
oversampling 2 only to rows that do not already name one, and `W7` names 4, so
it never touched binary. Matching the pools means an oversampling of
`ef / limit` per row on Qdrant's side, or `ef` sweeps on `bench7` and `bench8`
so the matched-recall interpolation can cover them without touching the pool.

### P2. What the licensed numbers are made of, and what the run costs

**5. The saturating win at d=1536 is two unexplained halves.** W4 reads 1.38x,
and the decomposition puts it at 0.85x less work per query times 1.61x cores
busy. Each factor is a question. strawmANN's cycles per query rise from 1.66M
at W3 to 4.01M at W4 while its DRAM per query holds at 4.4 MB and its IPC
halves from 0.71 to 0.29; aggregate traffic is 16 GB/s against a 73 GB/s bus,
so the seven workers are waiting on their own misses, not on the bus. The
next-candidate prefetch (715343f) was measured on sift1m, where W4's IPC is
1.08; it has not been measured at d=1536, where the latency it exists to hide
is the whole cost. Qdrant, on the same row and the same client parallelism of
64, fills 4.47 of its 8 pinned cores and holds IPC 0.60, with 0.2 s of runqueue
wait over the row. Why its search pool leaves 3.5 cores idle under a saturating
closed loop is not known from the files, and the 1.38x is mostly that.

**6. Qdrant's single-query cost doubles at d=1536.** W3 flips between the
tiers: 0.90x on sift1m, 1.73x here. Both engines run one core; strawmANN
spends 1.66M cycles per query and Qdrant 3.13M at equal IPC (0.71 against
0.68) and similar DRAM (5.9 against 4.7 MB), so Qdrant executes about 1.8x the
instructions for the same walk at the same recall, where on d=128 the two are
within 5%. Something Qdrant does per visited node scales with the dimension.
The 1.73x is licensed and correct as measured; what it measures is not known,
and if it is a setting rather than a path the ratio is being read wrongly.

**7. Filtered search is the licensed loss, and the loss is a dispatch.**
`W12-sel10` is flat at 166 qps from `ef` 64 with 45 MB of DRAM and 23M cycles
per query and recall 1.0000 at every `ef`: strawmANN scores the 20,054 matches
directly. Qdrant's curve moves with `ef` (2,226 to 288 qps, recall 0.76 to
0.999), so it walks the graph under the filter, and at matched recall strawmANN
reads 0.34x to 0.59x. At 1% the curves cross at `ef` 256. The scan itself is
slow for what it is: 1,974 vectors, 12 MB, in 1.63 ms is about 7 GB/s on two
cores, a fraction of the W9 kernel's per-core rate. `docs/workloads.md` quotes
~14,800 q/s for that scan and the row measures 1,221. Read the dispatch in
`handlers.searchOne` against the 10 KB `--full-scan-threshold` bfb passes, and
the scan loop it lands in, and correct the doc or the engine.

**8. The exhaustive pending tail costs 15x at d=1536.** `W11-steady` is refused
(item 3), but its counters are readable: during a 5% append strawmANN's search
fell from 3,570 to 244 qps at 57.6M cycles and 59 MB of DRAM per query, which
is every query scanning 49,500 pending vectors, 304 MB at this width, before it
returns. On sift1m the same row ran at 78% of W4 because that tail is 25 MB.
The cost is tail size times dimension, and findings 31 asked for incremental
insertion on exactly this ground. Live insertion exists behind `-Dlive-insert`
(1f03ba2) and was characterised on sift1m only (7cc278d). The measurement is
`W11-steady` at d=1536 with it on, once item 3 makes the row measure a
concurrent write.

**9. Qdrant's `W1` settle hits its 180 s timeout on every pass.** `W1`
uploads with `--skip-wait-index`; Qdrant indexes in the background and
`settle_engine` waits for it, nine minutes per Qdrant arm on a row that does
not measure the index. Skip the settle for `--skip-wait-index` rows, or accept
it and say so in the row.

**10. The pre-run estimate was three hours short.** `estimated_minutes` prices
the passes from the previous run's rows, and the table grew from 32 to 43 rows
between `rel-0903` and `perf-0924` (the `W12` selectivity grades and their `ef`
controls). Scale the basis by the rows present, and print the settle count the
run will actually make (10 here, against `12 x <= 300s`).
