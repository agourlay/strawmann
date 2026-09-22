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

**4. Measure what live insertion trades, then decide its default (findings 31).**
W11 spends its row scanning the pending tail while a rebuild re-does the whole
corpus. **Implemented 2026-09-22 behind `-Dlive-insert`, off by default**, the
same shape `-Dvisited` used: an appended point joins the published graph instead
of waiting, `build.insertLive` publishing `count` with a release store only
after the node is fully linked, and `collection.Bounded` bounding a reader's
traversal by its own snapshot so every node is in exactly one of the two regions
a query answers from. The bound is composed only when the graph holds more than
the reader covers, so with the flag off nothing is built and no query pays.

`liveInsert` declines rather than half-working: no published graph, a graph
already behind its frontier, a quantized collection, a full CSR, or a builder
that will not allocate all leave the point in the pending tail exactly as
before. A declined insert costs what it cost yesterday and never correctness.

Three defects the suite caught, none of them predicted: the builder leaked six
allocations because `Collection.deinit` never freed it; the quantized path was
unbounded and so hit the duplicate-and-invisible failure that had only been
fixed on the fp32 side; and **live insertion silently breaks a quantized
collection**, because a live-inserted point is reachable in the graph while the
quantizer encoded only what the build covered, so stage 1 has no code to score
it by. The differential test against the exact scan returned a stale point as
the nearest, which is exactly the class of defect this project exists to catch.
Quantized collections therefore opt out entirely.

**What is left is the measurement that sets the default.** W11's search rate
with the flag on against off, and the recall of queries served during the append
window, because the trade is a transient recall dip while a row is rewritten
under readers against a tail scan that costs ~3 ms per query for as long as a
rebuild takes. Both arms build and pass the full suite today; neither has been
run against W11.

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
