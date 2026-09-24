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

**1. strawmANN's SQ8 bounds clip dbpedia's dominant dimension, and the
score carries the loss.** T4 on both d=1536 pages: `sq8/strawmann |Δscore|
p50=4.866e-1` against the fp64 oracle, Qdrant's SQ8 `1.943e-3`, and 38% of
strawmANN's stage-1 top-10 ids are not the fp32 top-10 against Qdrant's 21%.
The differ measures with `rescore=false`, so this is the quantized score
itself, and the cause is measured (`decisions.md`, 2026-09-24): `scalar.train`
clips the 0.5% and 99.5% *values* of the pooled sample, and dbpedia's
component 194 sits at -0.64 in every vector (std 0.013), with 954 at +0.20 and
1120 at -0.16. Three dimensions carry 0.48 of a typical 0.81 top-10 dot, each
is 0.065% of the values, so all three are clipped to `lo=-0.050`, and the
reconstruction loses the 0.4876 they carried. Qdrant reads the same
`quantile: 0.99` as "cut `⌊vectors·(1-q)/2⌋` values per end", 25 of 7.7
million, keeps the dimension, and lands at `lo=-0.67`. Simulated on 50,000
vectors, Qdrant's rule takes strawmANN's SQ8 to `|Δ|` 0.0010 and the stage-1
top-10 overlap from 0.810 to 0.947; on sift1m the same change moves overlap
from 0.972 to 0.985. One knob, two semantics, so the "same configuration" was
never the same. With `rescore` on (the default) the returned score is fp32 and
recall is unaffected, which is why W6 matched; what the clipping costs is the
`ef`-sized rescore pool that compensates for it (3.3 MB of DRAM per SQ8 query
against Qdrant's 0.5). The decision is whether `quantile` means what Qdrant's
does; the measurement after it is the SQ8 sweep and W6 on both corpora.

**2. The SQ8 recall controls page the collection back in on the Qdrant arm.**
`W6-ef32` folded at a 41% spread and `W6-ef64` is refused as drift (+19%,
monotone across passes). The rows carry the cause and the fold does not read
it: Qdrant's passes 1 and 2 took 152,697 and 254,138 major faults on `W6-ef32`
(9.7k and 17.8k on `ef64`, 5.4k and 8.9k on `ef128`) and pass 3 took one.
`bench6` is evicted by `W7-upload`, `W8-upload` and `W9`'s 6 GB scan before the
`W6-ef` sweep reaches it, so the first rows of the sweep measure a page-in on
a row that claims `cached` residency, and `aggregate._rep_drift` reads the
monotone slide as a trend. strawmANN's arm pins its arena and took nine.
Two ratios refused on the headline page. The sweep now runs straight after
`W6`, before anything can evict `bench6` (2026-09-24); the 0925 pair's
`W6-ef` major faults and spreads say whether that was all of it. If a pass
still pages in, the other fix remains: drop a pass whose `major_faults` on a
cached search row exceed a floor.

**3. W11's append spans are sift1m's.** `W11_STEADY_SPAN_S` 25 s and
`W11_SPAN_S` 60 s were sized for d=128. At d=1536 the search covered 12 to 23%
of the append on every pass of both engines, so both mixed rows measured the
rebuild the append provoked rather than a concurrent write, as their own note
says, and both are refused. Since 2026-09-24 `fullrun.resolve_w11_spans`
reads each row's span off the previous pair, the slower engine's search times
1.25 (256 s and 431 s for the 0925 pair), and gives both arms the same value.
The 0925 pair's `write overlap` says whether that holds; item 8 is what the
row shows once it measures what it claims to.

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

**7. Filtered search is the licensed loss; at 10% it is the dispatch.**
`W12-sel10` is flat at 166 qps from `ef` 64 with recall 1.0000 at every `ef`:
strawmANN scores the 20,054 matches directly. Qdrant's curve moves with `ef`
(2,226 to 288 qps, recall 0.76 to 0.999), so it walks the graph under the
filter, and at matched recall strawmANN reads 0.34x to 0.59x. The scan was
not the slow part. Profiled on 2026-09-24, 43% of W12-sel1's samples were
`payload.Store.select` re-reading every matching blob, which the append-only
bench12 never needed; trusting exact postings for a single-condition filter
took sel1 from 1,194 to 1,854 qps and sel10 from 167 to 285 (one pass each,
not a gated run), and sel1 is now 81% distance kernel. Software prefetch in
`searchSelected` was measured in the same session and lost (0.75x to 0.80x).
What is left at 10% is 123 MB scanned per query against Qdrant's filtered
walk. strawmANN's walk scores every neighbour the filter rejects
(`hnsw.zig`: the filter gates admission, not expansion), so the graph cannot
take over from the scan until it stops doing that (ACORN-style: skip scoring
rejected neighbours, hop through them), and `plainFilteredSearch`'s cost model
changes with it. The next night run's W12 rows are the measurement of record.
`docs/workloads.md` still quotes ~14,800 q/s for the 1% scan; correct it from
that run.

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

**9. Qdrant's published `W2` was measured beside `bench1`'s index build.**
The settle after `W1` timed out at 180 s on every pass with Qdrant still on
7.0 cores, building the index of a collection nothing reads again, and `W2`
started under it: Qdrant's 657.7 s Time-to-Green and the 2.63x upload-and-index
ratio include that. `fullrun` now drops `bench1` between `W1` and `W2` instead
of settling (`workloads.rows_writing_dead_collections`), and the settle stamp
records it, so the next pair is STALE against 0924 by design. That pair's `W2`
is the clean figure; until then read 657.7 s as an upper bound on Qdrant's
build and the 2.63x as an upper bound on strawmANN's lead.
