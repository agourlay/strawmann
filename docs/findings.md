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

**3. Intra-query parallelism for W3, and what a prefetch already recovered
(findings 50).** W3 spends 972k cycles per query at `-p 1` against 650k
saturated, and the profile says the difference is unhidden memory latency, not a
wake cost: 6.5% kernel against 5.3%, with `Probe.prefetch` at 17.1% of samples
against 9.9%. One query is a serial dependent chain of cache misses with nothing
on the core to overlap it.

**The cheap half of that is done** (2026-09-22). The traversal prefetched each
popped node's neighbours but not the load that *starts the next iteration*,
`neighbours(peek().id)`, a random row of a 128 MB array and the one link nothing
covered. Prefetching it is worth **1.049x on W3** (server p50 488 to 459 µs) and
1.026x on W4 and W10-ef128, all clearing a 2.0 to 2.2% band over three
alternated passes. That is about 15% of W3's unsaturated overhead, and it helps
every search row rather than only the one.

What is left is the expensive half: expanding one query's frontier across
workers, which means giving up §6.3's one-thread-per-core discipline for the
single-query case. That discipline is what buys every other row its tail latency
(findings 36, 41: zero migrations against Qdrant's tens of thousands), so the
measurement that decides it is not W3's rate alone but W4-sat90's p99 beside it.
The ceiling is known from the counters: closing the whole 322k would take W3's
p50 to roughly 338 µs against Qdrant's 411, so the row would flip from 0.85x to
about 1.2x.
