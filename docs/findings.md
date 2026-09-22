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

**3. Measure the visited set, then pick one (findings 28).** The traffic
decomposition stands: at `ef` 128 the quantized row's 538 KiB per query is 12%
rescore pool, 20% quantized codes and **68% walk overhead**, and the same
overhead is 47% of the fp32 row's. V holds at 354 to 457 B per scored node
across a sixteen-fold change in `ef`, which is roughly six cache lines, and
§6.5's visited set is a 4 MB generation-stamped array taking one random line per
neighbour probed.

Both implementations now build: `-Dvisited=generation|bitmap` selects at
comptime, the arm is in the banner and on every row, and a randomised
differential test pins that they are indistinguishable through the interface
search uses. What is left is the measurement, and it is scripted:
`bench/harness/visited_ab.sh` builds both, alternates them A/B/A/B over W4,
W10-ef128 and W6-ef128, and folds. **Needs a quiet host.** The bitmap trades 4 MB
of footprint for a reset proportional to what the query touched, and nothing has
priced the reset.

**4. Gather concurrent exact queries into one scan (findings 45).** W9 is
bandwidth-bound at 85% of the bus, so no kernel, ISA or prefetch work touches
it: the prefetch attempt halved demand misses and bought one percent. Qdrant
scales 2.22x from `-p 1` to `-p 8` against strawmANN's 1.51x because it reads
the corpus less than once per query.

Two things reading the code settles about the shape of the fix. **The batch path
is not it.** `handlers.BatchJob` fans a `QueryBatch` *out* across workers, which
for exact queries means N workers each scanning the whole arena; gathering there
would fix that pathology and would not move W9, which sends single-query
requests at `-p 8`. Moving W9 needs gathering *across concurrent requests*,
which is a scheduler change. **And the probes cannot live on the stack.**
`Probe.max_buffer` is `max_converted_dim * @sizeOf(f16)` = 32 KiB, so K
converted queries need the per-worker workspace, not a local, if §6.3's
no-allocation-on-the-query-path rule is to hold. Budget it as a workspace
change plus a scheduler change, not an afternoon.

**5. Incremental insertion for W11 (findings 31).** Serving the old graph
through a rebuild was worth 5x on the slowest queries and moved the median the
wrong way, because every query pays the tail scan for as long as the rebuild
takes. The row's ceiling is that tail scan, and the fix is not having a rebuild
window: inserting appended points into a graph sized to capacity as they arrive.

**6. The unsaturated path (findings 50).** strawmANN spends 972k cycles per
query at `-p 1` against 650k at saturation. W3 is the one search row it loses
(0.85x), and 322k cycles of per-query overhead that saturation amortises away is
a wake or spin cost. Scripted as `bench/harness/unsaturated_profile.sh`, which
samples three states rather than one, W3 at `-p 1`, W4 saturated, and the engine
idle, because the answer is the difference between them and a flat profile of
any one cannot show it. **Needs a quiet host.**
