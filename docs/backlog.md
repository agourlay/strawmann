# Backlog

Two sections. **Findings** is what is still unmeasured or unexplained, in
priority order, and nothing else: an item that closes is deleted, and its
number is not reused, since commits and code cite items by number ("findings
62"). **Next** is what is ready to run, waiting on a decision, or being built,
and points at the findings it answers. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did
it and in the test that holds it.

## Findings

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The current pages are the 1109 pairs for sift1m, laion-small-clip,
h-and-m and the four BM25 corpora and the 1110 pair for dbpedia-openai-1m, all
on Qdrant `dev` b4ebd966f in release mode, the vector pairs T4 with T3 passing
and the text pairs T2. What they publish wrongly, or still refuse and could
not:

**74. strawmANN lost the SQ8 lead on dbpedia.** On 1110 W6 reads parity
(0.94x; 1008: 1.13x) and W6-ef256 and ef512 0.88x and 0.86x, matched-recall SQ8
0.80x to 0.90x from recall 0.967 up. Qdrant's W6 ran 13 to 19% faster on the
new pin with a third fewer instructions per query (plausibly qdrant#10975's
VNNI kernels). strawmANN still runs fewer instructions (0.96M against 1.79M)
at equal cycles, but pulls 2.3x the demand DRAM per query (2,444 against
1,056 KiB) at IPC 0.53 against 0.99, on 7.1 cores busy against 7.9. Two
suspects: stage 2 rescores up to `ef` fp32 rows (6 KiB each at d=1536) with
no prefetch (`binary.Rescorer.run`), and strawmANN's `1 io + 7 workers` on
eight CPUs. Since 2026-11-11 the engine takes `--rescore-prefetch off|first|whole`
(off by default) and `ab_chain_1112.py` runs it and `--workers 8` on
dbpedia-100K's W6 ladder. `--workers 8` did nothing for h-and-m's W4 on
2026-09-29 (decisions.md), which bounds what to expect from it.

ab-1112 (2026-11-11, dbpedia-100K, three passes): `--rescore-prefetch whole`
W6 +1.7%, ef256 +2.0%, ef512 +2.1%, demand DRAM -16 to -18%, now the
default; `--workers 8` +1.4 to +3.1% at more cycles per query and -7 to
-15% on W10-ef128, not taken. Neither closes the 6 to 14% gap: what is left
is SQ8's per-query DRAM against Qdrant's.

**75. laion's per-query cost.** laion 1109 W4 reads 1.03x where sift1m reads
1.94x, so W5 reads 0.86x: Qdrant gains 18% from batching (W4 10,933 to W5
12,939) and strawmANN nothing (11,247 to 11,095). The fan-out of a batch over
the workers is not the cause (ab-1110: serial 11,160 against fan-out 11,156).
Occupancy is not it: on W4 both engines hold about 7 cores (0.97x), so
1.03x is the work ratio (1.06x). strawmANN runs 40% fewer instructions per
query (0.66M against 1.10M) at IPC 0.51 against 0.79, on the same demand
DRAM (1,766 against 1,708 KiB). On sift1m it pulls 64 KiB against 1,089 at
IPC 1.48: there `collection.Probe.prefetch` asks for the whole 512-byte row
(b3ca8d2, demand fills 12,400 to 3,000 a query), while laion's 2 KiB rows
get their first line, and about 880 rows a query arrive as demand misses.
The W10 ladder's 1.36x at ef 128 runs `-p 8`, where Qdrant holds 5.3 cores,
so it is not W4's like for like. Since 2026-11-11 `--probe-prefetch-bytes`
sets the width, and `ab_chain_1112.py` runs 512 and 2048 on laion and 512 on
dbpedia-100K beside the rule. The whole 8 KiB row lost on h-and-m
(decisions.md, 2026-09-29); d=512 was never swept.

ab-1112 refuted the prefetch width: 512 bytes -2.1% and the whole 2 KiB row
-10.6% on W4, with demand DRAM down 56%, so laion's stall is not the
misses a wider hint can buy back; the knob is gone. `--workers 8` reads W4
+2.1%, W5 +2.7%, W3 +11% but W10-ef128 -15%. Open: what bounds laion's IPC
at 0.52 against Qdrant's 0.80 at the same DRAM per query.

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

2026-11-11: in-process at d=1536 a drained point cost 1,564 us, 1,065 of
them in its ef_construct search and 477 in re-pruning full neighbour lists.
`--prune-memo` cut the re-prune to 149 us with the same graph but read flat
through the server (ab-1112: W11-steady 518 to 524, W11 154 to 153) and was
reverted; the search is the lever left.

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

**62. strawmANN's first pass of a dbpedia night runs 5 to 9% faster.** The
same work at fewer cycles: on 1110 pass 1 read W10-ef32 10,617 against 10,029
and 10,087 (cycles per query 1.26M against 1.38M, instructions flat at 467k),
W10-ef128 +7%, W4 +2.5% and W11-steady 600 against 574 and 566 (the item 3
spread, closed into this one: the drainer and the tail start every pass in the
same state, and the engine's total instructions over the row are equal, so
pass 1's lower instructions per query are the drainer's fixed cost spread
over more queries). 1008 reads the same. 0929 and 0930 read the mirror image,
pass 3 slow, and their per-pass directories are gone. Huge pages are not it:
`anon_huge_bytes` and `file_pmd_bytes` are 0 on every row of every pass.

What sets pass 1 apart is memory. It is the night's first pass, the only one
that starts after strawmANN's arena wipe of 0.64 GB rather than a Qdrant pass
and a 7.6 GB one, and it runs under pressure: about 10 GB less file RSS at W10
(19.0 against 29.4 GB), 1.2 GB less anonymous, W1 with 30,576 major faults
against 0, and a recall sweep that paged 2.77M faults against 0.75M and
0.84M. The pass whose pages were reclaimed and read back runs faster than the
ones whose pages stayed resident since the engine wrote them; h-and-m 1109's
slow pass 3 is its most resident (4.3 against 3.6 and 3.4 GB file RSS).
Qdrant's W10 is flat at 31.2 GB file RSS every pass. Since 2026-11-11 each row
records the host's `MemAvailable` and used swap and the engine's `VmSwap`
(`procstat.memory_state`). What it needs: `reclaim_ab.py` (in `ab_chain_1112.py`),
on one built bench2: W10 with the collection as written, then the engine's
cgroup reclaimed through `memory.reclaim` (a running engine's mapped pages are
out of `posix_fadvise`'s reach, and the server does not reload collections on
a restart), one warming round, and W10 again on the pages read back. If that
buys the 5 to 7%, every pass should start from the same page-cache state.

Confirmed on purpose by ab-1112's reclaim-dbpedia (one upload, three rounds
each): W10-ef128 3,636 resident against 3,851 after the engine's cgroup was
reclaimed and read back (+5.9%), W10-ef32 10,150 against 10,821 (+6.6%), at
equal instructions per query. Open: why re-read pages run faster than the
ones the engine wrote, and whether every pass should start from the same
state (an operator decision; it changes the stamp).

**73. The filtered planner's constants at saturation.** The walk's fixed
term moved to 350,000 (4c08d9d) after bfb's with-replacement keyword draw put
63% of sift1m's sel10 queries at about 20,000 matches, not the fitted mean
19,124; ab-1110 measured sift1m sel10-ef128 +5.6% from it (now a scan,
recall 1.0) and laion flat. Two things the model still gets wrong: laion's
9-keyword queries (about 9,000 matches) scan at `ef` 256 under either
constant, and the scan's cost per value is not flat in `d`. What it needs:
a refit with `d` as a term, on the 1109 sweeps and an in-process ladder.
The 1103 forced-plan ladders say a single per-value cost cannot do it: per
matched row the scan costs about 137, 769, 1,440 and 2,270 cycles at
d = 128, 512, 1,536 and 2,048, about 17 cycles per cache line
everywhere but laion, at 24. laion's rows are dear to read in the scan as in
the traversal (item 75), so the refit waits on item 75's A/B: a prefetch
that moves laion's line cost moves this break-even with it. Refit 11-12 in
cycles from the 1112 and 1103 ladders: the scan as a per-row table over `d`
(145, 800, 1,530, 2,250), the walk as 1,000,000 + 580 per `ef · m0`. It picks
right at all forty points (twenty, each at 10 and 9 keywords), laion's
9,000 at ef 256 now walking. Open: a W12-sel10 A/B against the parent on
laion (ef 256 should rise) and sift1m (unchanged picks, flat).

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

Built on 2026-11-11; what is left is measuring it.

- **Rows:** W16-upload loads the dataset's own bundle (vectors, the whole
  payload, the fields the tests filter on indexed) into `bench16` through bfb's
  dataset path, and W16-ef32 to ef512 search it with each test's own
  conditions on W4's client (`workloads.tests_table`, `paths.FILTERED_TESTS`).
  recall@10, from a sweep joined on (dataset, collection, ef) like W12's
  grades.
- **Recall:** `conformance relevance --dataset-tests <tests.jsonl>` sends each
  query under its own condition and scores it against the shipped
  `closest_ids` with the ε-tie rule; `recall.py --collections bench16` runs it.
  `dataset-truth-check` checks those ids against our filtered fp64 top-k: on
  the first 200 tests of each corpus, all 200 shipped sets are our exact sets
  (laion 86 conditioned, max score delta 2.6e-7; h-and-m 200 conditioned,
  4.4e-7).
- **Engine:** laion's 2,530 conditioned tests are `range {gt}` on
  `similarity`; the engine decodes `range` and builds a float index
  (`Kind.float = 3`) since 2026-11-11.
- **h-and-m's W16 is on.** qdrant/bfb#193 merged and the pin moved to
  1498125 on 2026-11-13, so a value with whitespace ("Light Orange", 4,255 of
  the 10,000 tests) goes out as a keyword match, not a full-text one.

What it needs: a laion pair with W16 (a quiet machine), then h-and-m's once
the pin moves.

## Next

### Done 2026-11-11 night (unit `strawmann-night-1112`)

The ab-1112 chain (decisions.md, "The 1112 A/B chain's flags"), the BM25 x4
republish (published a01b629), a laion pair whose W16 rows failed on
`--acorn` and which carried two regressions (not published), and the
planner ladder on sift1m and laion (`bench/results/planner-ab-1112-*`,
within 1% of 1103's costs).

### Before the next night

- The PQ regression (laion W8 -6.4% at equal instructions against 1109),
  suspected in the `Query` union (ee3dbcc): in-process, being chased.
- Short result lists under laion's range conditions at low ef (W16 sweep:
  `short_lists=32` at ef 32 and 64): engine, being fixed.
- W16's ACORN: bfb refuses `--acorn` on a dataset source. Operator call:
  W16 without ACORN on both engines, or a bfb PR first.
- A pair with FAILED rows still rewrites its page (fullrun publish policy).
- fp32 +3% instructions per query since 1109: first suspect removed
  (d37d931); confirm in the next pair.

### Next nights

- **Night 2:** sift1m, laion (with W16 once it measures) and h-and-m, three
  passes each, on the promoted defaults (`--exact-gather`, rescore prefetch
  `whole`), about 11 h.
- **Night 3:** dbpedia-openai-1m, about 10.8 h.
- **An h-and-m pair with W16** (bfb pin 1498125, a new stamp for h-and-m).

### Waiting on a decision

- **Occupancy at 1% selectivity (findings 7):** splitting one query's scan
  across idle workers is an occupancy choice, not a cheaper scan.

### To build

- **The real-filter rows (findings 55):** an upload row and a search row on
  bfb's dataset path, a recall the row carries (bfb's precision) that
  `compare.py` and the report read, and the ε-tie check against our filtered
  oracle. After the bfb decision, since the rows run on it.

### Dropped

- dbpedia W9's 8% since the Zig 0.16 build (1008 to 1110, equal instructions):
  no 0.16 toolchain is left and today's source needs 0.17, so a toolchain arm
  cannot isolate it.
