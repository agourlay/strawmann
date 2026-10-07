# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted, and its number is not reused, since commits
and code cite items by number. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The current pages are the 1007 pair for sift1m, the 0930 pair for
dbpedia-openai-1m, the 1003 pair for laion-small-clip and the 0929 pair for h-and-m, all T4 with T3
passing and production-mode Qdrant. What they publish wrongly, or still refuse and could
not:

**3. W11's search still ends inside its append on three corpora of four.**
The write rate is fixed (1,900 and 3,300 points/s at 1M, hashed into the
stamp), and since 10a274a the row is measured over the append alone
(`workloads.write_window_qps`), with the search sized to outlast the append on
the faster engine (`decisions.md`, 2026-09-29). The 0930 pairs show the sizing
does not hold:

- **sift1m and laion:** the sizing asks for 305,000 to 633,000 queries and
  `min(QUERIES, n)` caps it at 50,000, so the search covered 7 to 37% of the
  append while fullrun printed "running 25% past the append". README's
  `mixed read/write` cell (8,033 against 2,387) is therefore the append's first
  10%, published under its note.
- **dbpedia:** W11-steady covered 100% on both engines. W11 covered 53% on
  strawmANN, because it was sized from 0929's faster engine (Qdrant, 106 q/s)
  and the drainer made strawmANN the faster one (251 q/s), and Qdrant's writer
  ran 2,344 to 2,729 points/s against the 3,300 asked, so its append lasted 72
  to 84 s, not 60.

Since 2026-09-30 `resolve_w11_sizing` has no cap at `QUERIES`, sizes each
label over its own measured writer span (`background_s`), bounds the slower
engine's search at `W11_MAX_SEARCH_S` (240 s), and prints `!!` when that
bound bites. It closes when a pair shows both rows' `write overlap` at 90% or
more on every corpus and `W11-steady` flat across passes. laion 1003 is the
first pair measured with it: `write overlap` 100% on both rows, both engines,
every pass, with the search running 38% (strawmANN) and 75% (Qdrant) past the
append and no `!!` bound line. strawmANN's W11-steady is flat (10,108 / 10,094
/ 10,102); Qdrant's is not (6,996 / 7,651 / 7,170, rsd 4.7%), while its
optimizer rebuilt on the same eight CPUs (run-queue wait 196 to 249 s a pass).
sift1m 1006 did not hold: Qdrant's W11 and both
engines' W11-steady covered 100%, but strawmANN's W11 covered 52.3%, its search
sized from 0930's 8,033 q/s and running at 18,066 (W11_QUERIES=572,856). A
count sized from the previous pair cannot follow an engine that doubles
between pairs; a search bounded by the writer's end is the sizing that can.
dbpedia 1006 measured but is unlicensed (the differ hung, item 60). Since
2026-10-06 there is no count: the search runs until its append ends, then a
quarter of the append's span more, at most 240 s (`workloads.w11_stop_at`),
and the harness stops bfb with SIGINT (`decisions.md`). sift1m 1007 is the
first pair measured with it: `write overlap` 100% on both rows, both engines,
every pass, the search ending at the append's span times 1.25 (W11: 60.6 s
append, 75.8 s search). strawmANN's W11-steady is flat (20,727 / 21,082 /
21,074); Qdrant's is not (4,416 / 4,431 / 3,554, rsd 12.1%), its third pass
low as laion 1003's first was. laion 1007 also read 100% everywhere but is
unpublished (item 71). What is open is dbpedia, and whether Qdrant's spread is
its own. Item 8 is what the row shows once it measures what it claims to.

**70. The published W5 ratios are mostly parallelism.** Every current
page's batched row ran at two requests in flight, where strawmANN used 7 cores
and Qdrant 2 (`decisions.md`, 2026-10-06): 3.6x of sift1m's 5.68x, laion's
2.83x and dbpedia's 2.69x, and per batched query Qdrant is the cheaper engine
on laion and dbpedia. W5 now runs W4's client. It closes when each corpus has
a pair measured with it. sift1m 1007 is: W5 reads 1.58x (27,641 against
17,448), the 3.6 factor gone, with Qdrant on 7.89 cores and strawmANN 7.02.
laion 1007 read 0.88x, Qdrant ahead as predicted, but is unpublished (item 71).

**71. The published W12 ratios are partly harness configuration.** Every
current page ran Qdrant's filtered search with a 10 KB full-scan threshold (5
points at d=512, so every filter walked the graph) and without ACORN, which
strawmANN's plan takes on its own (`decisions.md`, 2026-10-06). sel1's
2.3x to 4.0x at high `ef` and sel10's recall gap at low `ef` (0.94 to 0.99
against 0.77 to 0.87 at `ef` 32) both carry it. W12 now runs Qdrant's shipped
threshold and ACORN. It closes when each corpus has a pair measured with them.

sift1m 1007 is: 10,000 KB at d=128 is 20,000 points, above both tiers'
matches, so Qdrant scans both (recall 1.0000 at every `ef`, cycles per query
at sel10 4.3x 1006's) and never takes ACORN. sel1 reads 1.95x, sel10 2.25x
(1006: 0.61x). laion 1007 is not publishable, because its W12 recall is not
the recall of the search its rows ran. At d=512 the threshold is about 5,000
points, so sel10 (about 10,000 matches) walks the graph. The rows walked it
with ACORN (bfb `--acorn`), but the sweep they join (`recall.py`, the
conformance `relevance` binary) had no way to send ACORN, so it scored a walk
without it: Qdrant 0.0901 at `ef` 32, 0.2652 at 128 and 0.4126 at 512 (1003,
at the 10 KB threshold: 0.8164, 0.9854, 0.9993). A probe on a rebuilt laion
bench12 at the default threshold (`bench/results/acorn-probe-laion/`, bfb
`--search-quality` against Qdrant's exact search under the same filter, 1,000
queries) read 0.090 / 0.202 / 0.332 without ACORN and 0.962 / 0.992 / 0.997
with it. ACORN is not what costs the recall: it is what restores it on a graph
the default threshold builds without payload-block links. `relevance` now
takes `--acorn`, `recall.py` sends it on the filtered grades when the rows do,
the sweep records it, and the join refuses a sweep whose ACORN differs from the
label's `w12_acorn` stamp. Swept that way on a rebuilt bench12, Qdrant's sel10
reads 0.9649 / 0.9930 / 0.9982 against fp64 truth. bench12 is dropped after a
run, so laion 1007's W12 rows cannot be re-swept, and laion's page stays at
1003 until a pair measured with the fix. sift1m's W12 recall is unaffected:
Qdrant scans both tiers there. dbpedia (d=1536, about 1,700 points) walks
sel10 too, so its next pair needs the fix.

**58. Qdrant read 8 to 12% faster on laion 0930 than on 0928, same binary.**
W10-ef128 +10.4%, W10-ef256 +11.5%, W12-sel1 +10.2%, W11 +10%, with cycles
per query down 10% at the same recall, a byte-identical binary (sha256
`dbeb0f73dea2d371`), the same stamp and the same `env_hash`. On sift1m and
dbpedia the same binary moved under 2.3%. Nothing in the files explains it, so
part of every laion ratio move between the two pairs is Qdrant's, and the
laion W4/W10 moves cannot be read as strawmANN's. laion 1003 reproduced the 0930
level within 1.5% (W10-ef128 8,312 to 8,268, W12-sel1 3,331 to 3,368), on a
newer kernel and with Qdrant on its compiled-in config rather than the
checkout's, so 0928 is the odd pair. What it needs: a Qdrant-only same-binary
A/B on laion at two times of day (`qdrant_ab.py` with one build on both
sides).

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
MB of scattered rows at about 12 GB/s. Since 2026-10-06 the W12 rows run W4's
client (64 in flight), so the next pairs read sel1 under load, where the
exact scan's cost per query decides it rather than its latency. Two ways to offer the trade were looked
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
q/s, where `--no-drain` stays at 134 for as long as the tail stands. On the dbpedia
0930 pair W11-steady read 580 q/s over the append (Qdrant 504), and about 850
over the 17 to 21 s the search outlived it; the first night cannot say how
much of that is the drainer, since W11 changed definition with it and there
is no `--no-drain` arm. At d=128 and d=512 the drainer kept up with the writer
while a search ran (`drain linked=58000 ... pending=0` on sift1m).

The drainer started only from a search or an info call, so once W11's search
ended the rest of its tail waited for the read-back's rebuild (`rebuild start
points=1250000 pending=179100` on sift1m); since 2026-09-30 an upsert to a
`.ready` collection starts it too. At the sift1m and laion rates W11 then
never crosses `rebuild_ratio`, so it measures the incremental path there and
the rebuild only where the writer outruns the drainer
(`workloads.W11_STEADY_RATIO`). What is open is the in-append half at d=1536:
at 2 ms of core per insertion, keeping up with 1,900 points/s is about four of
the eight cores.

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
2026-09-29). The shared pass is what is left. Below the cap
the same contention is a draw: sift1m's W9 read 202 q/s on 0929 and 186 on
0930, and in-process two runs of one build read 190 and 182 at 423k and 590k
demand fills per query, as far apart as any two commits (e149208 to 1fb2e43,
177 to 189). Instructions are identical; what moves is how much the seven
0.5 GB scans share lines, which is scheduling and not code.

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

**60. Exact search past 4 GB admits waiting scans in no order, and W9's tail
doubled.** dbpedia 0930 W9 reads 13.06 q/s (1.33x, from parity) with the cap,
and p99 1,505 ms, p99.9 2,481 ms, max 3,292 ms against p50 576 ms (0929 p99
1,109 ms). `acquireScanSlot` is a compare-exchange plus a 50 us sleep, so a
waiter can lose every race. Since 2026-09-30 the slots are a ticket gate
(`collection.ScanGate`): admission in arrival order, at most `exact_scan_cap`
in flight, waiters asleep on a futex, and a test that holds both. What is
open is the dbpedia W9 row read against 13.06 q/s and 1,505 ms. dbpedia
1006 read it, unlicensed: 12.28 q/s, p99 828 ms and max 842 ms, the tail
gone. The same night the differ's closed 32-query exact batch hung for 600 s
and left the pair without a licence: `release` woke one sleeper
(`maxInt(u32)`, which `lock.zig` documents as waking one), and when that was
not the next ticket the next ticket slept on with nothing left to wake it.
W9's continuous arrivals kept rotating the queue, which is why its rows and
the gate's test passed. Fixed with `lock.wake_all`, and a test that queues
the admissible ticket second (it hung before). What is open is a licensed
dbpedia pair with the fix.

**62. strawmANN's third pass is 6 to 7% slower on every W10 point on
dbpedia, two nights running.** W10-ef32 10,625 / 10,752 / 9,934 on 0930 and
10,525 / 10,546 / 9,888 on 0929, while W3 and W4 on the same `bench2` are
flat. Unexplained. Since 2026-09-30 every row records its engine's
`AnonHugePages` and `FilePmdMapped` after the row (`anon_huge_bytes`,
`file_pmd_bytes`, read outside the bracket since the walk takes
`mmap_lock`). What is open is reading them on the next dbpedia pair: whether
pass 3's arenas are mapped by smaller pages.

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
