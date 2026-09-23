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

**2. Why a keyed level draw builds a worse graph (findings 34).**
The per-pass draw is the upload's arrival order, demonstrated in the real path:
a serial upload returns the same recall three times over and a concurrent one
spreads 0.00120 at `ef` 512 ([`decisions.md`](decisions.md)). What is not
settled is *why*. It survives the whole harness path, and widens there: with
`bench1`'s index still building alongside, as W1 leaves it, three passes spread
0.00227, which is the published magnitude. Neither half of a reordered arrival
is separately
fixable: stabilising the level draw changes nothing, stabilising the insertion
sequence multiplies the spread tenfold, and stabilising both builds a graph
0.0029 below what file order builds.

Underneath it is a narrower question than it first looked. Under the engine's
own level draw the seed does not matter: six builds at four seeds sit within
0.00003 at `ef` 512 (`graph-diff --seeds`, [`decisions.md`](decisions.md)), and
an earlier table showing a 0.00216 seed effect did not reproduce. Under
`assignLevelKey`, the draw keyed on something that survives a re-ingest, the
same four seeds span 0.00232; the loss sits in regions of level 0, differs by
seed, and starting the search at the answer recovers only about a quarter of it.
Why that assignment is fragile and the node one is not is open, and it has to be
answered before anything keys levels on external ids. The instruments are in
place: `graph-diff --seeds`, `--per-query`, `--oracle-entry`, `--stable-levels`,
`--stable-order`.
