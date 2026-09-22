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

**3. Gather exact queries across concurrent requests (findings 45).** W9 is
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

**4. Incremental insertion for W11 (findings 31). The safety question is
answered; what is left is the trade.** Serving the old graph through a rebuild
was worth 5x on the slowest queries and moved the median the wrong way, because
every query pays the tail scan for as long as the rebuild takes. The fix is not
having a rebuild window, by inserting appended points into the live graph.

Findings 31 justified that as safe with an invariant that does not hold as
stated: `linkBack` publishes a pruned row as a prefix write followed by
`@memset(list[kept..], empty)`, so mid-rewrite a reader observes
`[new0, new1, empty, old3]`, breaks at the first empty and expands a shorter
row. What holds is narrower, that slot 0 always carries a live id.

**Measured rather than argued (2026-09-22).** "searches against a graph being
mutated see only valid ids" drives three readers against a four-thread builder
inserting into the same graph, three rounds, asserting every returned id is in
range and every score finite, with a query counter so the assertion cannot pass
vacuously. Zero bad results, and green four times in a row. So a concurrent
reader gets a *worse* answer and never a wrong one, and the hazard is recall,
not safety. That is the test that gates the change, written before the change.

**The hard part is visibility, not safety or recall.** A query answers from two
regions, the traversal over the graph and `scanPendingTail` over
`[covered, total)`, and `covered` is one snapshot taken before the traversal.
The search path already anticipates one failure: a node "linked into the graph
after this query traversed, counted before this query scanned the tail, and
therefore in neither", which its comment calls "a no-op that stops being one the
moment anything inserts into a live graph". The mirror case is not recorded
anywhere and is worse, because it is silent: a node whose edges exist while the
reader's snapshot is behind is reached by the traversal *and* scored by the tail,
and `heap.TopK.push` does not deduplicate, so the client gets one point twice.

Pinned by "a graph covering more than the reader's snapshot returns a point
twice", which constructs the window directly rather than racing for it, asserts
the duplicate, and then shows the fix's shape: bounding the traversal to the
reader's own snapshot puts every node in exactly one region and nothing repeats.
So live insertion needs that bound, and the bound costs a predicate on the
traversal's hot path unless it is hoisted behind `graph.count <= covered`, which
is true on every query today.

The other preconditions are better than findings 31 assumed, and all three are
checked rather than assumed: `buildIndex` calls
`hnsw.Graph.init(..., coll.config.capacity)` (`collection.zig:1839`), so the
published graph has slots up to `--capacity` and not merely to the built count;
`upper_offsets` already carries a CSR cursor for appending; and
`quantized_search` already reads `g.count` with an acquire load, as does the
search path itself. What is left is the bounded traversal, publishing `count` after a
node is fully linked, and the measurement that decides it: whether a transient
recall dip during writes beats the tail scan it replaces, on W11's search rate
and on the recall of queries served during the append window.

**5. W3's deficit is memory-level parallelism, not a wake cost, so the lever is
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
