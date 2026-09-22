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

**3. Decompose W4's 2.20x on the page (findings 50).** It factors into 1.57x
less work per query, 1.35x core occupancy and 1.03x frequency, and the occupancy
reproduces across four runs and both segment policies. **The client has been
ruled out** (2026-09-22): W4 re-run against Qdrant alone, through `run_one` so
the queries and the arithmetic are the harness's, at four concurrencies from the
standard `-p 64 -t 16 -c 2` to `-p 512 -t 32 -c 16`:

| offered | qps | cores busy |
|---|--:|--:|
| `-p 64 -t 16 -c 2` | 9,122 | 5.30 |
| `-p 128 -t 16 -c 4` | 9,004 | 5.30 |
| `-p 256 -t 32 -c 8` | 8,910 | 5.29 |
| `-p 512 -t 32 -c 16` | 8,875 | 5.38 |

Eight times the offered concurrency moves occupancy 1.02x and throughput 0.97x,
so Qdrant does not fill more cores when offered more work and the 1.35x term is
its own. Development grade, and the level does not reproduce the run's: 9,122
qps at 5.30 cores against 10,351 at 5.55, a single pass without `--perf`, run
after W2 alone rather than after W0-upload, W1, W2 and W3. The *slope* is what
the probe was for and it is flat in all of it.

What is left is the reporting change: the page presents 2.20x whole, a third of
it is a scaling property a differently-tuned Qdrant might not have, and the
matched-recall 2.04x to 2.12x is the licensed comparison that should lead.

### P2. Engine work with a named lever

**4. Where strawmANN's SQ8 rescore spends its cycles.** Per query at `ef` 128,
three passes with `--perf` on `rel-0921`: 783k cycles against its own fp32 path's
640k, IPC 0.78 against 1.07, and 538 KiB of DRAM traffic where Qdrant's SQ8
reads 60 KiB for the same encoding. A 13x traffic gap for one encoding says the
rescore pool is being read from fp32 far more widely here. Measure the pool size
before touching code. Findings 29 already took stage 1 from 46% to 26% of server
CPU, so what is left is the walk and the rescore rather than the distance kernel.

**5. Gather concurrent exact queries into one scan (findings 45).** W9 is
bandwidth-bound at 85% of the bus, so no kernel, ISA or prefetch work touches
it; the prefetch attempt halved demand misses and bought one percent. Qdrant
scales 2.22x from `-p 1` to `-p 8` against strawmANN's 1.51x because it reads the
corpus less than once per query. `handlers.BatchJob` established that a request's
queries can be fanned across workers; the inverse does not exist.

**6. Incremental insertion for W11 (findings 31).** Serving the old graph
through a rebuild was worth 5x on the slowest queries and moved the median the
wrong way, because every query pays the tail scan for as long as the rebuild
takes. The row's ceiling is that tail scan, and the fix is not having a rebuild
window: inserting appended points into a graph sized to capacity as they arrive.

**7. The unsaturated path (findings 50).** strawmANN spends 972k cycles per
query at `-p 1` against 650k at saturation. W3 is the one search row it loses
(0.85x), and 322k cycles of per-query overhead that saturation amortises away is
a wake or spin cost.

### P3. Decisions and hygiene

**8. Whether T3 should gate the matched-recall table (findings 40).** Matched
recall does not assume equal recall, it constructs it, so gating it on T3
withholds the one comparison that survives exactly when it is needed. Left
deliberately unchanged, because loosening a licence gate because a banner is
inconvenient is the pressure this project exists to resist. It needs a spec
answer, not a patch.

**9. The matched-oversampling experiment (findings 42, `validation.md` item 6).**
Qdrant's SQ8 plateau is entirely recoverable at `oversampling 2`, at which point
it matches strawmANN. Whether equalising with a knob one engine did not need is
the same experiment is the open question.

