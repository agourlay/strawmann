# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted, and its number is not reused, since commits
and code cite items by number. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The d=1536 comparison is licensed since 2026-09-24, and the current page is
`sm/qd-dbp1m-perf-0927`, the first with Qdrant in production mode (T4 with T3
passing, 1.01x to 1.32x at matched recall; the account of the licence is in
`decisions.md`). What that page publishes wrongly, or still refuses and could
not:

**3. W11's write rate is fixed, and its search is sized to fit.** The 25 s
and 60 s spans were sized for d=128, where a 50,000-query search ends inside
them; at d=1536 the writer covered 12 to 23% of the search and both mixed rows
measured the rebuild the append provoked. Since 2026-09-25 the rate is fixed
(1,900 and 3,300 points/s at 1M) and hashed into the stamp, and the search was
sized to end inside the append from the previous pair's qps. That never held:
0927 covered 10 to 12% of W11-steady's append, and on 0929 W11's search saw
51% (strawmANN) and 84% (Qdrant) of it, with `W11-steady` drifting +24%
monotonically over three passes. The qps it sized from was set by the search's
own length, since the tail past the writer ran against a different collection.
Since 2026-09-29 the row is measured over the append alone: bfb stamps every
search, `workloads.write_window_qps` counts those that completed while the
writer ran, and `resolve_w11_queries` sizes the search to *outlast* the append
on the faster engine (`decisions.md`). What is open is the next dbpedia pair
confirming it: both rows' `write overlap` (now the share of the append the qps
saw) at 90% or more, and `W11-steady` flat across passes. Item 8 is what the
row shows once it measures what it claims to.

### P2. What the licensed numbers are made of, and what the run costs

**7. At 1% selectivity strawmANN offers only the exact answer.** On 0925,
with trusted postings (202841c) and the ACORN-1 walk (15bddf7), `W12-sel10`
reads 811 q/s and leads at matched recall up to 0.988 (1.44x to 1.78x; not
established above 0.994), and `W12-sel1` reads 1,873. What is left is sel1's
sweep: strawmANN scans the ~1,900 matches at every `ef`, flat at recall
1.0000, because below `1/m0` the walk strands (0.888 recall at twice the
cost, in-process), while Qdrant's walk trades recall for speed and at `ef` 64
serves 2,383 q/s at 0.9963. That point reads 0.79x and is a trade strawmANN
has no setting for; at equal recall it leads (1.61x at 0.9999). The query is
latency-bound at the client's two in flight, 0.96 ms on one core reading 11.7
MB of scattered rows at about 12 GB/s. Two ways to offer the trade were looked
at on 2026-09-25 and not taken: an SQ8 pre-score needs codes `bench12` does not
have (it is unquantized), and a first stage whose SQ8 bounds were only fixed
at this width on the 0927 pair (T4 `|Δscore|` p50 9.7e-4, from 4.9e-1);
splitting one query's scan across the idle workers would cut the latency but
is an occupancy choice, not a cheaper scan, and is the user's call. Software
prefetch in `searchSelected` was measured on 2026-09-24 and lost (0.75x to
0.80x).

**8. The exhaustive pending tail costs 15x at d=1536.** During a 5% append
strawmANN's search fell from 3,570 to 244 qps on 0927, every query scanning
49,500 pending vectors (304 MB at this width), and below `rebuild_ratio` the
tail never left: nothing indexes it until a rebuild the ratio does not call.
Since 2026-09-29 a background drainer links the tail into the live graph
(`collection.drainTail`, `decisions.md`). Measured on dbpedia-1m in-process
(random queries, W11-steady's 49,500 points at 1,900 points/s): it links
about 1,020 points/s against the search, so during the append it is behind
and search reads within 8 to 17% of `--no-drain` (412 and 445 against 381
q/s); 22 s after the append the tail is gone and search runs 900 to 1,800
q/s, where `--no-drain` stays at 134 for as long as the tail stands. What is
open: the published row (W11-steady on the next dbpedia pair, with item 3's
window), and the in-append half, which is the drainer's CPU against the
search's: at 2 ms of core per insertion at d=1536, keeping up with 1,900
points/s is about four of the eight cores.

**10. Exact search at d=1536 is lost to scan contention, and a shared pass
wins it back.** `W9` read 0.77x on the development-profile dbpedia page and
parity on 0927 (1.50x on sift1m). Measured on
2026-09-26 (`w9_ab.py`, pinned, three reps each, no foreign load):

| arm | q/s | cores | IPC | implied GB/s |
|---|--:|--:|--:|--:|
| 7 workers, one query a request (W9 as published) | 9.74 | 7.00 | 0.26 | 59 |
| 4 workers | 13.04 | 4.00 | 0.60 | 79 |
| 7 workers, 8 queries a request | 51.06 | 6.94 | 1.33 | 311 |

Seven concurrent 6 GB scans get less from the bus than four: 1.34x from
dropping three workers, about the 12.8 Qdrant read on the development
profile's 4 threads. In production mode it runs 7.97 cores on the 0927 pair's
W9 and reads 10.0 q/s, parity with strawmANN's 9.9: the same contention on its
side.
And one pass shared by 8 queries is 5.2x: the 2026-09-22 decision against a
cross-request gather rested on sift1m, where concurrent scans shared lines
through the cache, and tested a gather onto one worker. At 6 GB there is no
sharing, and a gather split across the workers is the lever, which reopens
that decision for d=1536. The smaller lever landed in 99d2776: scans of 4 GB
and more wait for one of four slots, 10.98 q/s against ~10 (`decisions.md`,
2026-09-29). The shared pass is what is left.

**11. strawmANN's index build ran at the search workers' priority during
`W11`.** On 0925 strawmANN's `W11` served 102 q/s against Qdrant's 216, with
1,568 s of run-queue wait over the row: the append's extending builds ran
eight threads at normal priority on the search workers' eight CPUs. Since
2026-09-25 builds run at nice 10, as Qdrant's HNSW builds do (`decisions.md`).
On 0927 the run-queue wait fell to 867 s, and the search got slower: 50 q/s
against Qdrant's 99, at 313M cycles per query. The deprioritised build never
published within the row (`rebuild start points=1089100 pending=99100` and no
`published`, in every pass), so every query scanned the whole pending tail
(item 8). Nice 10 bought the search workers their CPUs back and spent them on
the exhaustive tail. The drainer (item 8) does not reach this row: W11's fifth of the
corpus arrives at 3,300 points/s, faster than it links, so the tail crosses
`rebuild_ratio` and the rebuild takes over as before.

### P3. What the datasets offer that no row measures yet

**55. h-and-m and laion ship real filtered queries, and W12 uses synthetic
ones.** `h-and-m-2048-angular-filters` carries a 24-field product payload per
vector (`payloads.jsonl`: product type and group, colour, department, section,
garment group, a free-text description), the value sets its conditions draw
from (`filters.json`), and 10,000 queries that each hold a condition such as
`{"and": [{"product_group_name": {"match": {"value": "Shoes"}}}]}` with their
25 nearest neighbours *under that condition* (`tests.jsonl`). `laion-small-clip`
ships the same shape with a float range condition on every query, and k=10.
The harness uploads the vectors alone, from the fbin, and W12 filters bfb's
synthetic keywords over `bench12`, so neither engine is measured on a
realistic filter, a multi-field payload or a range condition, and the shipped
filtered truth goes unread. What it needs: the payload uploaded with the
collection, a W12-style row driven by each query's own condition, and recall
scored against the shipped `closest_ids` with the ε-tie rule (checked once
against our own filtered oracle, as `ground-truth.md` §6 did for the
unconditional sets). This is also the workload vector-db-benchmark runs on
these two datasets.
