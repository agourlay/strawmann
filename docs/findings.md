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

**4. Gather exact queries across concurrent requests (findings 45).** W9 is
bandwidth-bound at 85% of the bus, so no kernel, ISA or prefetch work touches
it; Qdrant scales 2.22x from `-p 1` to `-p 8` against strawmANN's 1.51x because
it reads the corpus less than once per query.

**The batch half is done** (2026-09-22): `collection.bruteForceRangeMulti` reads
each row once and scores K queries against it, and `handlers.runGatheredExact`
routes an all-exact unfiltered `QueryBatch` of 2 to 16 queries through it, so
such a batch costs one pass rather than K. Differential tests at every batch
size, an e2e test comparing one batch of four against four batches of one, and
the path instrumented once to confirm the tests reach it.

**What is left is the half that moves W9**, which sends single-query requests at
`-p 8`: gathering across *concurrent requests* rather than within one. That is a
scheduler change, collecting pending exact queries and scoring them in one pass,
and `bruteForceRangeMulti` is the scan it would gather into. Worth doing only
with a measurement beside it, since the win is a slope (1.51x against Qdrant's
2.22x) and not a row.

**5. Incremental insertion for W11 (findings 31), and the invariant it rests on
is weaker than that entry says.** Serving the old graph through a rebuild was
worth 5x on the slowest queries and moved the median the wrong way: every query
pays the tail scan for as long as the rebuild takes, and a 200k tail at d=128 is
~3 ms of memory traffic per query however it is served. The fix is not having a
rebuild window, by inserting appended points into a graph sized to capacity as
they arrive.

Findings 31 justifies that as safe because "the stale-but-valid, never-empty
list invariant already holds". Reading `build.zig` (2026-09-22), that is not
what the writers guarantee, and the file says so at
`assertEmptiesAreASuffix`: `linkBack` publishes a pruned row as a prefix write
followed by `@memset(list[kept..], empty)`, so mid-rewrite an unlocked reader
genuinely observes `[new0, new1, empty, old3]`, breaks at the first empty and
expands a *shorter* row. What holds is narrower, that slot 0 always carries a
live id, so a reader never sees an empty list and never an invalid id.

That changes the question rather than closing it. Inserting under live readers
is **safe** (no torn ids, no use-after-free, unlike the `noteOverwrite` class of
bug) and costs **recall**, transiently, in proportion to the write rate, which
is the same connectivity loss the parallel build already measures at 8 threads.
So the design question is no longer "is it safe" but "is a small transient
recall loss during writes better than the tail scan it replaces", and both sides
of that are measurable: W11's search rate against the recall of queries served
during the append window.

Preconditions the code already sets: the CSR must have slots past
`graph.count`, so the published graph has to be sized to `--capacity` rather
than to the built count; `graph_count` must become the thing readers observe
monotonically, published after a node is fully linked; and `insertOne` and
`linkBack` already take per-node locks and promote the entry point under
`entry_lock` with `EntrySnapshot`, so writer-writer races are handled. What is
new is reader-writer, and the test for it is the one the concurrency work in
this project has always needed: N searchers against a live inserter, asserting
no invalid id is ever returned and recall recovers, run in a loop for flakiness.

**6. W3's deficit is memory-level parallelism, not a wake cost, so the lever is
intra-query parallelism (findings 50).** strawmANN spends 972k cycles per query
at `-p 1` against 650k at saturation, and W3 is the one search row it loses
(0.85x). This entry used to guess that the 322k of overhead was "a wake or spin
cost". **Profiled 2026-09-22 and it is not.** Three states, same binary, same
collection:

| | kernel | user | `Probe.prefetch` share |
|---|--:|--:|--:|
| idle (no queries) | 86.8% | 13.2% | - |
| W3, `-p 1` | **6.5%** | 90.4% | **17.1%** |
| W4, saturated | 5.3% | 91.4% | 9.9% |

A wake or spin cost would put W3's kernel share far above W4's. It is 1.2 points
above, and both rows are ~91% userspace in the same four symbols. What actually
moves is the stall absorber: `Probe.prefetch` takes 17.1% of samples at `-p 1`
against 9.9% saturated, which against each row's own cycles is **166k cycles per
query against 65k**, a third of the whole gap concentrated in the instruction
that waits for memory.

So the 322k is unhidden memory latency. One query is a serial dependent chain of
cache misses and there is nothing else on the core to overlap them with; at
`-p 64` sixty-four queries' misses interleave. That also explains the W3
measurement from `rel-0921`: Qdrant spends 1.05 cores per query there against
strawmANN's 0.86, which is Qdrant spreading one query across cores and hiding
its own misses. §6.3's one-thread-per-core discipline is what buys strawmANN
every other row and what costs it this one (findings 36 argues the same trade
from the I/O thread's side).

The open work is therefore intra-query parallelism for the single-query case,
and it is a design change rather than a tuning knob. What would settle whether
it is worth it: the same profile against a build that splits one traversal's
neighbour expansion across two workers, which is the smallest version of it.

*Caveats.* Profiled with ~0.4 cores of foreign load, which perturbs the absolute
rates and not the relative shares this reads. `kptr_restrict` left the kernel
symbols unresolved, so the idle row's 86.8% is unattributed; it does not affect
the W3-against-W4 comparison, which is entirely userspace.
