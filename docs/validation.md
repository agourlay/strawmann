# Validation status

What has been checked, what has not, and what would close the gap.

This document exists because the rest of the repository reads as more settled
than it is. There is a host gate, a conformance harness, an fp64 oracle, a
contamination tracker and a generated report, and the combined effect is to
suggest the numbers have been vetted. They have been vetted **by this project,
against itself**. That is a different claim, and the difference matters most to
the people the tool is aimed at.

## Internally validated

These are checks the project runs on itself and passes.

| what | how |
|---|---|
| Correctness against truth | T1 graded against an fp64 exhaustive oracle, not against Qdrant. Exact search is bit-identical to the oracle on both dataset tiers. |
| Ground truth itself | Recomputed in fp64 and diffed against the published SIFT1M GT: 999,889 of 1,000,000 neighbours agree, and all 111 differences are boundary ties. |
| Tolerance | ε calibrated from measured cross-ISA spread (§8.4), never chosen. Where no calibrated cell exists for a (metric, dim) the differ uses §8.4's analytic summation-error floor and records `epsilon_source: floor`; sift1m (euclid, d=128) has a calibrated cell whose value equals the floor, because the measured cross-ISA spread was exactly zero on integer-valued SIFT distances (`docs/tolerance.md`); the runs from `rel-0908` on record `epsilon_source: calibrated`, the two before it (`rel-0903`, `rel-0907`) `floor`, and each report says which it used. |
| Wire compatibility | The same pinned `qdrant-client` drives both engines, so any difference is server-side by construction (§8.5). |
| Measurement discipline | §7.1 gate stamps every result row; per-row ambient load and foreign processes recorded (foreign load is the per-pid CPU delta across the row, `workloads.foreign_between`) and a contaminated row is *refused* a ratio, not annotated; contamination cost calibrated (18 cores = +41%, 3 cores = +11%, 1 core = noise). |
| Row provenance | Every row carries `gate`, `profile`, `engine_build`, `engine_binary`, `isa_build`, `optimize`, `harness_hash`, `n_requested` and `warmup_s`; `compare.py` refuses a label whose rows carry more than one build identity as STALE, and a row where bfb exited 0 but left neither JSON nor a `Median qps` line is filed `no-output` (a failure) rather than `ok` with no qps. `noise.json` records the `harness_hash` and `n_dirs` it was measured from, and `regression.py` warns when a floor was taken under a different harness stamp than the rows it is applied to. |
| The ISA sweep | Predictions written before the run, then matched: W9 (purest fp32 kernel) 5.80x, W3 (diluted by traversal) 2.57x, W13 (no distance arithmetic, the control) 0.99x. |
| Self-inspection | The bandwidth probe caught a modelling error in its own first output; the gate caught a hardcoded path; reading the rendered report caught rows describing themselves as the opposite of what they measure. |

## Measured

**The noise floor is measured, on a gated host.** Six identical passes over one
unchanged server, corpus built once, load generator on the client cores and the
engine pinned to the server cores. `W11` is excluded on purpose: it appends
200k to `bench2`, so a second pass is not a repetition of the first. All six
repetitions passed §7.1 and none was discarded.

| row | what it measures | rsd | one engine | two engines |
|---|---|--:|--:|--:|
| W0 | transport plumbing floor | 0.33% | 2.0% | 2.0% |
| W3 | search, fp32, single query | 0.46% | 2.0% | 2.0% |
| W4 | search, saturating (closed loop) | 0.32% | 2.0% | 2.0% |
| W5 | search batched, 16 queries/request | 3.09% | 9.3% | 13.1% |
| W6 | quantized: scalar | 1.53% | 4.6% | 6.5% |
| W7 | quantized: binary + oversampling | 0.54% | 2.0% | 2.3% |
| W8 | quantized: PQ | 0.19% | 2.0% | 2.0% |
| W9 | exact / brute force | 1.39% | 4.2% | 5.9% |
| W10 ef=32 | recall/latency frontier | 0.64% | 2.0% | 2.7% |
| W10 ef=64 | | 0.67% | 2.0% | 2.8% |
| W10 ef=128 | | 0.30% | 2.0% | 2.0% |
| W10 ef=256 | | 0.13% | 2.0% | 2.0% |
| W10 ef=512 | | 0.15% | 2.0% | 2.0% |
| W13 | scroll / pagination | 1.25% | 3.8% | 5.3% |

The threshold has one definition, `regression.noise_band(rsd, arms)`:
`max(3·rsd, 2%)` for one engine against itself (`regression.py`, the "one
engine" column), and `max(3·√2·rsd, 2%)` for a ratio between two engines
measured with the same spread (`report.py`, the "two engines" column, whose
spreads add in quadrature).

**Nine of the fourteen rows sit at the 2% floor**, which is the interesting
result: the limit on what the comparison may call is no longer this machine, it
is the project's own policy on the smallest difference worth naming. Only W5
(13.1%) and W6 (6.5%) need a gap wide enough to be worth quoting as a caveat.

The floor before this one was taken on an **ungated** host (`powersave`, boost
enabled, and, as it turned out, `signal-desktop` at 20%) and it covered six
rows at 1.10-18.63%. It is superseded rather than merged. The difference is
large enough to change conclusions: W13 needed a **56%** gap to resolve anything
under the old floor and needs **5.3%** under this one, and W4 went from 13.2% to
2.0%. A contaminated floor is not a conservative floor; it hides real
differences behind a threshold that was never about noise.

Reproduce it with `bench/harness/noisefloor.sh`, which is where the sequence
lives now. Two things in that script are load-bearing and were each learned by
getting them wrong. It **settles before every pass**, because `workloads.py`
runs the gate at the *start* of an invocation and `load1` is a one-minute
average, so a pass launched straight after a 1M-point upload reads that upload's
load and stamps `FAIL`, the measurement's own load, with no foreign process to
blame. And it **aborts the moment a repetition fails the gate**, rather than
completing all six and discovering it at the end.

`bench/harness/regression.py` consumes this and renders a verdict per row. It
has been checked in both directions:

- **Specificity.** Two runs of an *unchanged* binary: 6 of 6 inconclusive, no
  false positives. Two rows moved 5.2% and -6.3% between those runs, and
  without the floor the second reads as a regression.
- **Sensitivity.** An AVX-512 build against an SSE2 build, a real and
  independently understood slowdown: W3 and W9 flagged as regressions (61% and
  83%), and W13, which does no distance arithmetic, correctly inconclusive.

## Checked against Qdrant's source

Read from a `dev` checkout, which settled four assumptions that had been
inferred from behaviour. Two held, and two were wrong in my favour:

| claim | verdict |
|---|---|
| `ef` is per segment | **holds**, and the retraction that rests on it stands |
| "8 to 13 segments" | **wrong**, capped at 8; the 13 was invented and then cited in six files |
| cosine is dot after ingest normalisation | **holds** |
| the accuracy gap is fp32 accumulation depth | **holds**: 48 FMA steps per chain against 12 |

It also surfaced something not previously known here: **Qdrant's distance path
has no AVX-512**, topping out at four 256-bit accumulators, so part of the
measured throughput gap is 512-bit kernels against 256-bit ones rather than
anything architectural.

This is source verification, not external validation. It removes guesses from
the comparison; it does not establish that the harness would catch a regression
someone at Qdrant already knows about.

## Not validated

**No Qdrant developer has used this tool.** Nobody outside the project has run
it, checked a result against something they already knew, or disagreed with a
finding. Every number here has one author and one reviewer, and they are the
same, and that person works at Qdrant, which is what made the source reading
above possible and what stops any of it from counting as review.

**It has never been pointed at a real Qdrant regression.** The ISA sweep shows
the harness can detect a known performance difference and correctly report *no*
difference on a control row. That is internal consistency. It is not evidence
that the tool would flag a real regression in Qdrant's tree, or that it would
stay quiet when Qdrant is merely refactored.

**Figures written into these documents by hand predate the gate.** Before
2026-08-17 the development machine ran a `powersave` governor with boost
enabled, which §7.1 makes disqualifying, and re-measuring is the only fix. The
qualitative findings are stable across repetitions; those figures are not
publishable. The generated tables are a different matter: they carry their own
provenance and are replaced wholesale by the run that writes them.

**No latency comparison has ever been published, and until now none could be.**
§7.4 reserves `--rps` for latency claims and forbids quoting a closed-loop p99
as one. Every open-loop row this project has measured was bfb's own drain
backlog rather than the engine (findings 32), so the second half of "equal
recall at lower latency", the statement §8 is aiming at, was unmeasured.
Findings 33 fixes the generator and shows the instrument working against
strawmANN alone: client p50 within 0.12 to 0.37 ms of server p50 across a factor
of eight in offered load, rising with load rather than falling. That is an
instrument that works, not a comparison. No published run has used it yet, and
the report decides for itself whether to print the refusal. It reads the
inversion out of the rows rather than asserting it from this document.

**The new instruments have not produced a published run.** The stall counters
(`delayacct_blkio_ticks`, cgroup PSI) and the per-row hardware counters
(`perf stat` attached to the engine) are checked against synthetic input in the
gate and have been exercised end to end against a live engine on SIFT1M, but no
table in this repository was measured with them: `--perf` is off by default and
every published row predates the columns. The pressure columns additionally
need the engine in a cgroup of its own. A container has one, a binary started
from a shell shares the operator's session, and its figures are then refused
rather than reported.

**The sidecar is visible in the measurement, which is why `--perf` is opt-in.**
`perfstat.py --overhead` compares a single-threaded spin with and without
counters attached and puts the cost below that arm's own 7.6% run-to-run
spread; that is a floor, not a bound, because a spin loop does none of the
switching, faulting and forking whose cost the sidecar actually tracks.
Measured instead on W3 against a live engine, four passes each way, SIFT1M,
1M points:

| | qps | |
|---|--:|---|
| without the sidecar | 2,707.5 | median of 4, spread 2.72% |
| with the sidecar | 2,592.5 | median of 2 |
| **cost** | **4.25%** | against W3's 2.00% noise band |

Four percent is outside the band, so a row measured with counters attached is
**not** comparable against one measured without, and the report banners a pair
of arms that disagree about it.

That figure is provisional and reads as an upper bound. The passes ran
`off`-then-`on` in every pair rather than alternating, so any drift within a
pair is charged to the sidecar; two of the eight rows were discarded for
foreign load, and the host was compiling throughout. `bench/harness/perf-overhead.sh`
is the same measurement done properly, ABBA ordering, a settle before every
pass, contaminated rows discarded rather than averaged, and a refusal to state
a figure at all when fewer than two clean pairs survive. It wants a quiet host,
which is what §7.1 defines.

**Known deviations from §7.** §7.1's warm-up phase is implemented: every
measured search row is preceded by a discarded pass of `WARMUP_N` (2000)
queries through the same bfb invocation, its wall time recorded as `warmup_s`
(`--no-warmup` skips it; upload rows have none). §7.2's interleaved A/B/A/B differential
is implemented **and now used**: `fullrun.py --reps N` alternates the arms per
repetition and `aggregate.py` folds the median, and every published pair since
2026-09-03 has run `--reps 3`, sift1m and dbpedia-openai-1m both. §7.4's
"minimum three interleaved repetitions per cell, median of medians" therefore
has the runs as well as the machinery, and each label carries its own
`noise.json` folded from its own three passes, so the spread under a ratio is
that corpus's on those two engines rather than another corpus's from one
engine. What three passes do not buy is a good *estimate* of the spread: an rsd
from three samples is itself noisy, and one unlucky pass widens a band far more
than it moves a median. `--reps 1` remains what `smoke-test.sh` runs, and the
report banners it.

Absent `isolcpus`/`nohz_full` is no longer part of that verdict. It now selects
the `as-deployed` **profile** rather than failing the gate (see
[decisions.md](decisions.md)), so a run on an ordinary machine is publishable
as a statement about a deployment. Just not as a statement about the engine in
isolation, and not against an `isolated` run.

## What the dbpedia-openai-1m report refuses, and what clears each refusal

The 2026-08-26 dbpedia-openai-1m report carried six banners. None was a
reporting bug, every one was correct about the run it described, and none
could be removed by editing anything, because each was a property of how that
run was measured. The causes were tracked down 2026-08-26, the fixes went into
the harness, and a `--reps 3` run on 2026-09-03
(`sm/qd-dbp1m-perf-rel-0903`, its page retired from [`reports/`](reports/) to git history) is the one
that tested them. **Four came off. The two that matter did not**, and a
third run on 2026-09-24 (`sm/qd-dbp1m-perf-0924`, archived) took those off
too.

| banner | cause | what clears it | 2026-09-03 |
|---|---|---|---|
| Not interleaved (§7.2(5)) | `--reps 1` measured A-then-B | `--reps 3` | cleared |
| No noise floor for this corpus | `--reps 1` folds no per-label floor, so the global sift1m one applies | `--reps 3` | cleared: the floor records `dbpedia-openai-1m`, 25 rows |
| No noise floor for this pair | same, and the global floor is strawmANN's alone | `--reps 3` | cleared: folded per label, both arms |
| Open-loop rows not compared | `--rps-reference` was unset, so each arm used its own saturation | now defaults to `auto` | cleared: both arms offered 1,742 qps, 50% of the slower engine's 3,484 |
| Not a licensed comparative claim (×2) | T3: Qdrant searched four populated graphs to strawmANN's one | `--max-segment-size` for the rows, and `default_segment_number` pinned at the server for the differ's own collections | still refused on 2026-09-03; **cleared 2026-09-24**: T3 passed, 0.9665 against 0.9691 |

The prediction below was that removing the segment confound might not be
enough, and on 2026-09-03 it was not: with `--max-segment-size` passed the
differ reached **T2**, because strawmANN measured recall@10 0.9667 [0.9630,
0.9700] against Qdrant's 0.9831 [0.9804, 0.9854]. This section then read that
as "at d=1536 and 1M points strawmANN simply retrieves less well". **It was the
setup after all**, one level down: `--max-segment-size` governed the rows'
collections and not the differ's, which `conformance/` builds itself at
Qdrant's default segment count, four populated graphs at this size. With
`default_segment_number` pinned at the server for the differ's Qdrant
(`fullrun.qdrant_segment_env`), the 2026-09-24 run measured 0.9665 [0.9628,
0.9699] against **0.9691** [0.9655, 0.9723], the intervals overlap, T3 passed,
and `docs/comparison-dbpedia-openai-1m.md` exists. strawmANN did not move;
Qdrant's differ-collection recall fell to what its row collection had measured
in both runs (0.9692 and 0.9683 at `ef` 128). The account is in
[`decisions.md`](decisions.md), 2026-09-24.

Three of the four cleared are one flag. `--reps N` alternates the arms *and* makes
`aggregate.py` fold each label its own `noise.json`, which `regression.floor_for`
prefers over the global file, so a repeated run measures this corpus's spread
on both engines instead of borrowing another corpus's from one engine. Verified
on synthetic folds: the floor then records `dbpedia-openai-1m` and combines both
arms as `sqrt(mean(rsd^2))` rather than using one for both.

The two licensing banners were the ones worth watching. `--max-segment-size`
removes the *confound* from the rows (measured, five segments merge to one
990,000-vector graph), and findings 39 put the segments at a minority of the
0.0135 gap. That estimate was wrong: the segments were the whole gap, in the
one collection the flag did not reach. On 2026-09-03 the tier refused at
0.0164 and was read as a clean statement about the engine; on 2026-09-24, with
the differ's collection held to one graph too, the gap is 0.0026 and inside the
intervals. The question "why does this engine lose recall at 1536 dimensions"
is closed as asked of the wrong collection; the comparison at d=1536 is
licensed and reads 1.29x to 1.40x at matched recall.

Not on that list any more: W12 read `n/a` on the strawmANN arm until
2026-09-03, when payload storage, `CreateFieldIndex` and filtered search were
built (`src/core/payload.zig`). The row now asks both engines for the keyword
index; what it still lacks is a recall measured *under the filter*, so §7.4
licenses it no ratio, and the four things docs/workloads.md lists for W12
remain open.

Its *Qdrant* arm was a different matter, and was refused rather than reported.
The row passed `--skip-field-indices` (because bfb panicked on strawmANN's
refusal), so Qdrant answered every filtered query by checking all 200,000
points, and its rate went into the table under the words "filtered search" with
nothing beside it to say so. The flag is gone; `compare.py` still reads the index
state back off the engine (`collection-info`'s `payload_indexes`) and refuses the
row when there is none, or when nobody looked. What W12 has to become before it
is a filtered-search measurement is spelled out in
[workloads.md](workloads.md#w12-filtered-search-phase-3).

**The re-run happened on 2026-08-26** (`--reps 3 --perf`, native Qdrant, six
arms and 192 rows over nine and a half hours) and cleared four of the six:

| banner | after |
|---|---|
| Not interleaved (§7.2(5)) | gone |
| No noise floor for this corpus | gone; both engines' spread measured on dbpedia-openai-1m for the first time |
| No noise floor for this pair | gone |
| Open-loop rows not compared | gone |
| Not a licensed comparative claim (×2) | **untested** |

The two licensing banners are untested rather than cleared: the run was stopped
in the morning one phase short of the §8 differ, so the report carries
UNLICENSED, no conformance row at all, instead of the T3 verdict this was
meant to obtain. The rows are complete and the licence is recoverable on its
own in about half an hour; the report's own banner now prints the command.

The §9 blocker this paragraph used to name is gone: the pin needed a bfb built
at `6f216634`, a one-commit fork, and the checkout kept drifting off it. The
patch merged upstream as qdrant/bfb#172 on 2026-08-27, and `BFB_PIN` has been
plain upstream `dev` ever since (`0c1aafee` then, `fc6632e5` since
2026-09-08) which any clone can check out.

## Milestone status

§10 states an exit criterion per milestone and no verdict against it, so this is
the verdict. It was on the front page until it grew into a table a new reader
had to scroll past to reach the comparison.

| milestone | status |
|---|---|
| **M-1** datasets, fp64 oracle, GT diff | met for both tiers, [`docs/ground-truth.md`](ground-truth.md) |
| **M0** hardware model + SIMD kernels | full `dist/` inventory, the ISA matrix measured across 8 build arms (`baseline` SSE2 through `avx512-full`), `docs/asm/` checked in (`bench/isa/dump_asm.py --check`, manual, not a gate step). **Exit not met:** §10 asks for two microarchitectures, and eight arms are eight ISAs on *one*, so [`isa-matrix.md`](isa-matrix.md)'s selection table is §7.5's Zen 5 row alone. Zen 4 should read differently, since it double-pumps 512-bit ops and the fp32 result above is an MLP effect |
| **M1** transport + differ | h2c/HPACK/gRPC hand-written, T0-T4 live |
| **M2** storage | flat files and headers implemented and tested; **not reachable from the server**, and `load` refuses mapped placements |
| **M3** HNSW | bulk build serial + parallel, flat level-0, Yellow→Green |
| **M4/M5** quantization | SQ8, binary, PQ8 ADC wired to a collection; PQ4 FastScan is a kernel with tests (`dist/pq_adc.zig`) that no `Store` reaches |
| **M6** native transport | from the start; no nghttp2 on the build host |
| **M7** payloads and filtering | payload storage, `SetPayload`, `CreateFieldIndex` (keyword, integer), filtered `QueryBatch` and `Scroll`, `with_payload` (`src/core/payload.zig`); the matching set is scored directly when `selected² < ef·m0·n` and traversed under the predicate otherwise. W12 runs with the index on both engines; its cost model is not yet documented (M7's second exit criterion) |
| **M8** NUMA | not implemented; **§10** marks it optional, and it is untestable on the build host: one NUMA node, so its exit criterion (a two-socket scaling curve) needs hardware that is not here |

Every "not met" above needs something this repository cannot supply by editing:
a second microarchitecture, a two-socket host, or a documented cost model.

## What would close it

In rough order of how much each would buy:

1. **One regression a Qdrant developer already understands.** Two commits from
   their history where the performance delta is known, run through this harness
   blind. If the verdict matches what they know, the tool has demonstrated the
   thing it claims. If it does not, that is the more valuable outcome.

   *Offered and deliberately deferred (2026-08-16).* The source check above was
   judged sufficient grounding for now. `bench/harness/qdrant_ab.py` exists so
   that this stays a single command whenever a pair is to hand:
   `qdrant_ab.py <img-a> <img-b> --reps 3`. Deferred is not the same as
   unnecessary, which is why it stays on this list rather than being struck
   from it.

2. **A second pair of eyes on the workload table.** §4's rows encode assumptions
   about what is worth measuring. W12 has already proven unusable as specified,
   and W11 was silently corrupting W13 until an audit caught the ordering. There
   are likely more.

3. **A corpus that separates dimension from metric** (findings 34). The
   per-build graph draw is ~7x narrower at d=1536 than at SIFT1M, uniformly
   across fp32, SQ8, binary and PQ. Corpus size is now ruled out directly,
   dbpedia-openai-100K at the same d=1536 spreads 0.0002 to 0.0006 at `ef` 512,
   the same order as 1M (2026-08-29). Dimensionality and metric still change
   together. A d=1536 euclid corpus, or SIFT1M scored under cosine, would
   separate them.

4. **An SMT-off pass over the settled sift1m labels** (findings 46). Across
   the SMT change strawmANN moved 1.5% and Qdrant 30%, which is what findings
   36 and 41 predict from one-thread-per-core pinning against 43 multiplexed
   threads, but it is one environment against a recollection of another, and
   that is the error findings 46 is about. One pass at SMT-off, same labels,
   settles whether the asymmetry is real. Until it exists the pairing is a
   hypothesis and must not be quoted.

5. **Gathering concurrent exact queries into one scan** (findings 45). The
   only lever on a bandwidth-bound row is reading fewer bytes, and W9 has eight
   queries in flight over one 614 MB arena. Concurrency scaling is 1.51x
   against Qdrant's 2.22x. `handlers.BatchJob` already fans one request's
   queries across workers; the inverse, gathering independent in-flight
   queries into a single pass, does not exist. It trades latency for
   throughput and needs a deadline, so it is a design question rather than a
   patch, and it is the only named change with a quantified ceiling at
   d=1536.

6. **A decision on whether to keep an h5 copy per tier.** bfb's accuracy
   path needs h5, tar or sparse (see
   [ground-truth.md](ground-truth.md) §1.1), so scoring recall with bfb
   would mean a second copy of every dataset, scored against a third
   party's recomputed ground truth. Spec §4.1 declines it and keeps
   relevance in `conformance/`; it is only worth revisiting if numbers
   directly comparable to published vector-db-benchmark runs become a
   goal.

## How to read a number from this repository

Read the row, not this page. Every row carries its own `gate`, `profile`,
`harness_hash` and environment hash, and `compare.gate_of` reports `mixed`
when an arm's rows disagree, which they can, because the gate is re-checked
per `workloads.py` invocation and `fullrun` calls it more than once per arm.
Anything not stamped `pass` is development-grade: useful for spotting effects
of 2x and larger, unusable for publication. The report says so in a banner
rather than a footnote, and `compare.py` repeats it under every table.

A list of blessed dates used to live here and was wrong within two runs of
being written. The stamp is on the data.

A ratio is only a finding if it clears its row's detection threshold. Those are
in the table above and enforced by `bench/harness/regression.py`. A ratio near
1.0 is inconclusive on every row, and the generated tables mark each one, so
the thresholds do not have to be applied by hand or quoted here.
