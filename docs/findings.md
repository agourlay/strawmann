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

**2. Bulk-build in external-id order, not in arrival order (findings 34).**
The mechanism is settled and it was not the builder: the per-pass recall draw
is the *upload*. `ids.IdSpace.reserve` hands out internal offsets as points
arrive, the bulk build then inserts in offset order, and W2 uploads over eight
streams, so every pass links the corpus in a different sequence. Uploading
serially collapses the `ef` 512 spread from 0.00120 to 0.00000 and buys 0.0007
of recall ([`decisions.md`](decisions.md), `bench/harness/upload_order_ab.py`).
The level assignment, which the same reordering also moves, was measured and is
not the lever.

So the lever is the sequence: `build.extendParallel` hands its workers node ids
0 to n, and handing them a permutation sorted by external id would put every
pass in the file-order regime whatever the upload did. What is unmeasured: what
the sort costs W2 (1M u64, once per build), whether the gain survives the whole
harness path rather than an isolated `bench2`, what it means for a corpus whose
ids are not the base-file row index, and whether the entry point needs a
deterministic tie-break too (`promoteEntry` resolves equal max-level nodes by
arrival).
