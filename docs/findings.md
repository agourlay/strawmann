# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted, and its number is not reused, since commits
and code cite items by number. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The current pages are the 1109 pairs for sift1m, laion-small-clip,
h-and-m and the four BM25 corpora and the 1110 pair for dbpedia-openai-1m, all
on Qdrant `dev` b4ebd966f in release mode, the vector pairs T4 with T3 passing
and the text pairs T2. What they publish wrongly, or still refuse and could
not:

**3. dbpedia's W11-steady is not flat across passes.** W11's coverage half
is closed: since the clock (`workloads.w11_stop_at`, 2026-10-06) every corpus
reads `write overlap` 100% on both rows, both engines, every pass, h-and-m
last (1109). What is left is the spread on dbpedia. strawmANN's W11-steady read
618 / 541 / 534 on 1008 and 600 / 574 / 566 on 1110, pass 1 high both times
(1008's pass 1 was also the one with 90,509 major faults, yet the fastest, so
paging does not explain it); Qdrant read 504 / 494 / 460 on 1008 and is flat
on 1110 (506 / 499 / 499). h-and-m 1109 is within 4% on both engines. Whether
the first pass's head start is the drainer (item 8) meeting a colder tail, or
the engines' own, is open.

**74. strawmANN lost the SQ8 lead on dbpedia.** On 1110 W6 reads parity
(0.94x; 1008: 1.13x) and W6-ef256 and ef512 0.88x and 0.86x, matched-recall SQ8
0.80x to 0.90x from recall 0.967 up. Qdrant's W6 ran 13 to 19% faster on the
new pin with a third fewer instructions per query (plausibly qdrant#10975's
VNNI kernels). strawmANN still runs fewer instructions (0.96M against 1.79M)
at equal cycles, but pulls 2.3x the demand DRAM per query (2,444 against
1,056 KiB) at IPC 0.53 against 0.99, on 7.1 cores busy against 7.9. Two
suspects: stage 2 rescores up to `ef` fp32 rows (6 KiB each at d=1536) with
no prefetch (`binary.Rescorer.run`), and strawmANN's `1 io + 7 workers` on
eight CPUs. What it needs: an in-process A/B of a rescore prefetch and of
`--workers 8`, W6 ladder with W4 as the control.

**75. laion's per-query cost.** laion 1109 W4 reads 1.03x where sift1m reads
1.94x, so W5 reads 0.86x: Qdrant gains 18% from batching (W4 10,933 to W5
12,939) and strawmANN nothing (11,247 to 11,095). The fan-out of a batch over
the workers is not the cause (ab-1110: serial 11,160 against fan-out 11,156).
Suspects: occupancy (item 74's second) and Zig 0.17's kernel inlining.

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

**62. One strawmANN pass sits apart on every W10 point on dbpedia.** The
third pass read 6 to 7% slow two nights running, W10-ef32 10,625 / 10,752 /
9,934 on 0930 and 10,525 / 10,546 / 9,888 on 0929, while W3 and W4 on the same
`bench2` are flat. Since 2026-09-30 every row records its engine's
`AnonHugePages` and `FilePmdMapped` after the row (`anon_huge_bytes`,
`file_pmd_bytes`, read outside the bracket since the walk takes
`mmap_lock`). 1110 read them: `anon_huge_bytes` and `file_pmd_bytes` are 0 on
every row of every pass, so the arenas are not mapped by larger pages on any
pass, and page size is not the difference. On 1110 the odd pass moved: pass 1
is 5 to 7% *faster* (W10-ef32 10,617 against 10,029 and 10,087; ef128 3,854
against 3,611 and 3,620), still with W3 flat (1,079 / 1,071 / 1,077), which
widens strawmANN's W10 bands to ±9 to ±13%. Still unexplained.

**73. The filtered planner's constants at saturation.** The walk's fixed
term moved to 350,000 (4c08d9d) after bfb's with-replacement keyword draw put
63% of sift1m's sel10 queries at about 20,000 matches, not the fitted mean
19,124; ab-1110 measured sift1m sel10-ef128 +5.6% from it (now a scan,
recall 1.0) and laion flat. Two things the model still gets wrong: laion's
9-keyword queries (about 9,000 matches) scan at `ef` 256 under either
constant, and the scan's cost per value is not flat in `d`. What it needs:
a refit with `d` as a term, on the 1109 sweeps and an in-process ladder.

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
