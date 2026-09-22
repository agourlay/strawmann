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

**3. W3's deficit is memory-level parallelism, not a wake cost, so the lever is
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
