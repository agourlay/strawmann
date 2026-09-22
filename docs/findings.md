# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. The comparison's largest open questions

**1. Re-run dbpedia-openai-1m with the segment ceiling set (findings 37, 39).**
The README calls d=1536 the headline tier and it has no licensed comparison:
T3 failed at 1M x 1536 with strawmANN 0.0135 behind, and the Qdrant arm was four
populated graphs against strawmANN's one, a confound that covaries perfectly
with the result. `--max-segment-size` is in the harness and the sift1m tier is
clean; the tier the finding is about was never re-run. Until it is, the largest
claim this project wants to make is unmeasured.

**2. The parallel build is a per-build draw (findings 34).** strawmANN's graphs
disagree with each other by 0.00288 at `ef` 512 against Qdrant's 0.00009, which
is wider than the gap between the engines at that point, so a single pass can
report either engine ahead at high recall. Unreachable nodes were eliminated as
the mechanism (two hundred times too small); what is left is which edges the
pruning race keeps, and nothing measures that yet. A graph-diff between two
builds is the instrument. The named candidates are a deterministic insertion
order, a post-build reachability repair, or refusing to prune a node's last
in-edge in `linkBack`.

### P2. Engine work with a named lever

**3. The quantized row's traffic is the visited set, not the rescore pool.**
The premise this item started with was wrong and the counters on disk say so.
Model the two rows at one `ef`: `fp32 = N * (512 + V)` and
`sq8 = N * (128 + V) + 512 * pool`, where N is nodes actually scored, V is the
non-vector bytes each one costs, and the pool is `max(asked, ef) = ef`
(decisions.md §5). Two measured numbers per rung, two unknowns, and V is shared
across rungs, so five rungs overdetermine it. From `rel-0921`:

| ef | fp32 KiB/q | sq8 KiB/q | nodes scored | V B/node |
|--:|--:|--:|--:|--:|
| 32 | 249.9 | 162.1 | 277 | 413 |
| 64 | 440.0 | 297.6 | 465 | 457 |
| 128 | 794.9 | 538.0 | 856 | 439 |
| 256 | 1,442.0 | 959.0 | 1,629 | 394 |
| 512 | 2,582.5 | 1,693.4 | 3,054 | 354 |

V holds at 354 to 457 B across a sixteen-fold change in `ef`, which is the
check: a wrong model would not keep still. At `ef` 128, where the row is
published, the 538 KiB splits **64 KiB rescore pool (12%), 107 KiB of quantized
codes (20%), 367 KiB of walk overhead (68%)**. The same overhead is 47% of the
fp32 row. Rescoring `ef` vectors instead of `limit` is not what costs the row.

**The named lever is already written and not wired in.** `V` is roughly six
cache lines per scored node, and §6.5's visited set is a generation-stamped
`u32` array, 4 MB per worker at 1M points, one random line per neighbour probed
(findings 28 measured that as "one cache miss per neighbour").
`src/index/visited.zig` also implements `Bitmap`, 128 KB at 1M, tested, behind
the same interface, and `hnsw.zig` constructs `visited.Generation`. §11's open
question 3 asks where the two cross; this says the answer is worth 68% of a
quantized row's memory traffic and 47% of an fp32 one's.

Next step is a profile, not a patch: confirm where V goes on a quiet machine
(`perf record` on W6-ef128, or a counter on visited probes) before swapping the
structure, because the bitmap trades footprint for a per-query clear and this
model does not price the clear.

**4. Gather concurrent exact queries into one scan (findings 45).** W9 is
bandwidth-bound at 85% of the bus, so no kernel, ISA or prefetch work touches
it; the prefetch attempt halved demand misses and bought one percent. Qdrant
scales 2.22x from `-p 1` to `-p 8` against strawmANN's 1.51x because it reads the
corpus less than once per query. `handlers.BatchJob` established that a request's
queries can be fanned across workers; the inverse does not exist.

**5. Incremental insertion for W11 (findings 31).** Serving the old graph
through a rebuild was worth 5x on the slowest queries and moved the median the
wrong way, because every query pays the tail scan for as long as the rebuild
takes. The row's ceiling is that tail scan, and the fix is not having a rebuild
window: inserting appended points into a graph sized to capacity as they arrive.

**6. The unsaturated path (findings 50).** strawmANN spends 972k cycles per
query at `-p 1` against 650k at saturation. W3 is the one search row it loses
(0.85x), and 322k cycles of per-query overhead that saturation amortises away is
a wake or spin cost.
