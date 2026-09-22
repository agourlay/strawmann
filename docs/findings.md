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

**2. Why one level assignment beats another by 0.0022 (findings 34).**
The per-pass draw is the upload's arrival order, demonstrated in the real path:
a serial upload returns the same recall three times over and a concurrent one
spreads 0.00120 at `ef` 512 ([`decisions.md`](decisions.md)). What is not
settled is *why*, and neither half of a reordered arrival is separately
fixable. Stabilising the level draw changes nothing, stabilising the insertion
sequence multiplies the spread tenfold, and stabilising both builds a graph
0.0029 below what file order builds.

Underneath it is something larger than the finding it came from. Four level
seeds, one corpus, one insertion order, two builds each: 0.99955, 0.99955,
0.99739, 0.99794 at `ef` 512, each reproducible to 0.00001. **The seed is worth
0.00216**, more than the pass-to-pass draw and more than the gap between the
engines at that point, and §8.7 records it as provenance without recording what
it costs. Until that is understood, a recall figure is quoted without 0.002 of
its own uncertainty, and the instruments for it are in place: `graph-diff
--seed`, `--stable-levels`, `--stable-order`.
