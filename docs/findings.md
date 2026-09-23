# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. The comparison's largest open questions

**1. Re-run dbpedia-openai-1m, and get a matched-recall ratio out of it
(findings 37, 39, 40).** The README calls d=1536 the headline tier and it has
no licensed comparison. The first run (2026-08-26) failed T3 with strawmANN
0.0135 behind and the Qdrant arm four populated graphs against strawmANN's
one, a confound that covaries perfectly with the result. The re-run with the
ceiling set happened on 2026-09-03 (`*-dbp1m-perf-rel-0903`, three passes,
`equal-work`, one populated graph per Qdrant collection): T3 failed again at
0.9667 against 0.9831, so the deficit is the engine's and not the segments'.
What that run still does not give is the comparison findings 40 argues should
survive a T3 failure: the page refuses the matched-recall ratio because
strawmANN's W10-ef128 and ef256 drifted 25% and 27% across passes, with a
pass spread of 11 to 16% against Qdrant's 0.3%, on the engine that holds
0.5% on sift1m. The per-pass directories are pruned, so whether that was
contamination or a d=1536 instability cannot be read from disk. The next
run answers both, and is the first at this tier with the level seed
recorded, `--oversampling-policy matched` available, and the differ's own
collections held to the segment policy. Until it exists, the largest claim
this project wants to make is unmeasured.
