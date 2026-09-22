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
