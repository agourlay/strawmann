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
| Tolerance | ε calibrated from measured cross-ISA spread (§8.4), never chosen. Where no calibrated cell exists for a (metric, dim) the differ uses §8.4's analytic summation-error floor and records `epsilon_source: floor`; the sift1m runs to date (euclid, d=128) are on the floor, and the report says so. |
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
(`sm/qd-dbp1m-perf-rel-0903`, archived in [`reports/`](reports/)) is the one
that tested them. **Four came off. The two that matter did not.**

| banner | cause | what clears it | 2026-09-03 |
|---|---|---|---|
| Not interleaved (§7.2(5)) | `--reps 1` measured A-then-B | `--reps 3` | cleared |
| No noise floor for this corpus | `--reps 1` folds no per-label floor, so the global sift1m one applies | `--reps 3` | cleared: the floor records `dbpedia-openai-1m`, 25 rows |
| No noise floor for this pair | same, and the global floor is strawmANN's alone | `--reps 3` | cleared: folded per label, both arms |
| Open-loop rows not compared | `--rps-reference` was unset, so each arm used its own saturation | now defaults to `auto` | cleared: both arms offered 1,742 qps, 50% of the slower engine's 3,484 |
| Not a licensed comparative claim (×2) | T3: Qdrant searched four populated graphs to strawmANN's one | `--max-segment-size`, now passed | **still refused**, and now for the engine rather than the setup |

The prediction below was that removing the segment confound might not be
enough, and it was not: with `--max-segment-size` passed the differ reached
**T2**, because strawmANN measured recall@10 0.9667 [0.9630, 0.9700] against
Qdrant's 0.9831 [0.9804, 0.9854] and the intervals do not overlap. At d=1536
and 1M points strawmANN simply retrieves less well, so §8 refuses the ratio and
there is no `docs/comparison-dbpedia-openai-1m.md` for it to go in. That is the
refusal working: it is now a statement about the engine, which is what the last
paragraph of this section asked for. Findings 40 argues the *matched-recall*
comparison should survive a T3 failure, since matched recall constructs equal
recall rather than assuming it; that has not been acted on.

Three of the four cleared are one flag. `--reps N` alternates the arms *and* makes
`aggregate.py` fold each label its own `noise.json`, which `regression.floor_for`
prefers over the global file, so a repeated run measures this corpus's spread
on both engines instead of borrowing another corpus's from one engine. Verified
on synthetic folds: the floor then records `dbpedia-openai-1m` and combines both
arms as `sqrt(mean(rsd^2))` rather than using one for both.

The two licensing banners were the ones worth watching. `--max-segment-size`
removes the *confound* (measured, five segments merge to one 990,000-vector
graph) but T3 also needs the two engines' recall confidence intervals to
overlap once it is gone, and findings 39 put the segments at a minority of the
0.0135 gap. The prediction was that the tier might refuse again; on 2026-09-03
it did, at a gap of 0.0164, and the refusal is now a clean statement about the
engine rather than about the setup. That was the point, and it means the open
question at d=1536 is no longer "is the comparison set up fairly" but "why does
this engine lose recall at 1536 dimensions". Which
[findings 39](findings.md) starts on and nothing has closed.

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

1. ~~A gated host.~~ **Done for the `as-deployed` profile** (2026-08-17: fixed
   governor, boost off, quiescent; not isolated cores, which selects the
   profile rather than failing the gate). An `isolated` run does not exist.
2. ~~A measured noise floor.~~ **Done, and redone on a gated host** (see
   above). The suspicion recorded here was correct: W13's 18.63% was something
   specific rather than general jitter, and on a quiet gated box the same row
   measures 1.25%.
3. **One regression a Qdrant developer already understands.** Two commits from
   their history where the performance delta is known, run through this harness
   blind. If the verdict matches what they know, the tool has demonstrated the
   thing it claims. If it does not, that is the more valuable outcome.

   *Offered and deliberately deferred (2026-08-16).* The source check above was
   judged sufficient grounding for now. `bench/harness/qdrant_ab.py` exists so
   that this stays a single command whenever a pair is to hand:
   `qdrant_ab.py <img-a> <img-b> --reps 3`. Deferred is not the same as
   unnecessary, which is why it stays on this list rather than being struck
   from it.
4. **A second pair of eyes on the workload table.** §4's rows encode assumptions
   about what is worth measuring. W12 has already proven unusable as specified,
   and W11 was silently corrupting W13 until an audit caught the ordering. There
   are likely more.

5. **A licence for the 2026-08-28 dbpedia-openai-1m rows.** That run was
   stopped after five of its six arms, so it has two complete interleaved
   passes per engine and no conformance row: the differ runs after the passes.
   The rows are measured and the report renders, banner and all; only §8's
   licence is missing. It needs no re-measurement and is immune to load,
   because it grades correctness rather than speed:

   ```
   fullrun.py --dataset dbpedia-openai-1m --skip build \
              --skip strawmann --skip qdrant \
              --strawmann-label strawmann --qdrant-label qdrant
   ```

   About thirty minutes, and it would also produce this project's first T0-T4
   row at 1M x 1536. The build must still be the one those rows were measured
   at (2026-08-27 22:25, "The header names the engine the run will actually
   measure") against Qdrant 1.19.0.

6. ~~Whether Qdrant's SQ8 plateau is the rescore pool.~~ **Done, and it is**
   (2026-08-28, findings 42). `bench6` swept at oversampling none/2/4/8:
   recall@10 rises 0.8929 -> 0.9888 at `ef` 512 while recall@1 stays invariant
   to four decimals across the whole eightfold range at `ef` >= 128. The
   plateau is the pool and nothing else, `oversampling 2` recovers ~90% of it,
   and at that setting Qdrant reaches 0.9888 against strawmANN's 0.9891 -- so
   §7.4's refusal of the quantized rows is measuring a default rather than an
   engine that recalls worse.

   **What it did not settle** is whether the comparison should therefore be
   re-run at matched oversampling. §7.4 compares at equal recall, and reaching
   equality by giving one engine a knob the other did not need is a different
   experiment from the one the table holds. That question is now the open one,
   and it is a spec question rather than a measurement.

7. ~~A `sched_coverage` that survives a thread exiting.~~ **Done, and the
   answer reverses the reading** (2026-08-28, findings 41).
   `procstat.SchedSampler` banks per-thread counters during the row, so a
   thread that exits is still counted; it costs a measured `threads x 45 us`
   per poll, which is 0.2% of a core for strawmANN and 1.4% for Qdrant, and is
   charged to the client cpuset rather than the engine's. On by default since,
   with `--no-sched-sampler` to opt out, and every row carries `sched_sampled`
   beside a `sched_coverage` that still means what it always did. The share of
   the row's CPU the surviving threads account for, which is the engine's
   thread churn and the signal this whole entry rests on.

   Measured, W3 and W4 do not show the effect the W10 sweep does: Qdrant waits
   0.02% of its on-cpu time on W3 (*less* than strawmANN's 0.75%) and 0.61% on
   W4, against 143x-264x on the W10 rows. The entry's refusal to extend the
   claim to those rows was right.

   **What is now open** is why W10 differs. It runs at about twice W4's
   throughput; and W3/W4 follow an upload whose indexing threads exit during
   them, so they may be measuring an engine still finishing background work.
   Neither is established, and the second would be a different problem from
   scheduling.

   **Both candidates resolved, 2026-08-29.** The second was right: findings 44
   traced W3 to Qdrant's optimizer still running, and `settle_engine` removes
   it. On the settled pair Qdrant's W3 wait is 0.003 s against strawmANN's
   0.027 s. The first is right too, and it inverts findings 41's title. Across
   the settled W10 sweep the wait *falls* as `ef` rises (2.81 / 2.45 / 2.09 /
   1.55 s at `ef` 32 / 64 / 128 / 512, three passes agreeing to two decimals),
   because a wider search serves fewer queries per second and so makes fewer
   scheduling decisions per second. Queueing tracks the arrival rate rather
   than the depth of each search, which is why the high-throughput row is the
   one that shows it. The ratio to strawmANN stays in the hundreds to
   thousands and that half of findings 41 stands; "it scales with `ef`" does
   not.

8. ~~A per-engine noise floor, from one completed `--reps 3`.~~ **Done**
   (2026-08-28, `sm-sift-perf` / `qd-sift-perf`). Three interleaved passes on
   sift1m with `--perf`, all 192 rows `foreign=0` and `gate=pass`, T4 licensed.
   Qdrant's W3 floor is **28.16%** against strawmANN's 0.51%, so findings 38 is
   confirmed and understated, and `compare.py` now bands W3 at ±84.5% and
   prints `parity`. Findings 44 has the cause.

   **Quantified 2026-08-29** (findings 44): with `settle_engine` on, Qdrant's
   W3 reads 1,951 / 1,965 / 1,977 across three passes, RSD 0.66%, at 541 µs
   per query on every pass, and the headline `search, p=1` inverts to 0.86x.
   The shared `--rps-reference` also landed, so the open-loop rows ran both
   engines at one offered load. What is still not on disk is a *complete*
   clean pair: the settled run's strawmANN rep1 carries five `foreign` rows
   from builds on the host. One more three-pass sift1m run on a box with
   nothing else on it (this session closed too, since the CLI idles at 8 to 10%
   of a core and was the margin on at least one refused row) is what the
   front page needs before it can quote either engine's W3.

   **The clean pair landed 2026-08-29 16:03 to 18:02.** `foreign=0` and
   `gate=pass` on all 8 label directories x 32 rows, `perf=32` and
   `sampled=32` on each, T4 `licenses_comparative=true`, hash
   `8983d7b1b8da68b4`. Both arms fold W3 at **0.56%** (strawmANN 1,671,
   Qdrant 1,971) so the ratio is banded at ±2.38% instead of ±84.5% and
   `search, p=1` reads **0.85x** rather than `parity`. strawmANN's 3.24%
   in the note above was `rep1`'s contamination, not the engine. The front
   page may now quote either engine's W3.

9. **A corpus that separates dimension from metric** (findings 34). The
   per-build graph draw is ~7x narrower at d=1536 than at SIFT1M, uniformly
   across fp32, SQ8, binary and PQ. Corpus size is now ruled out directly,
   dbpedia-openai-100K at the same d=1536 spreads 0.0002 to 0.0006 at `ef` 512,
   the same order as 1M (2026-08-29). Dimensionality and metric still change
   together. A d=1536 euclid corpus, or SIFT1M scored under cosine, would
   separate them.

10. ~~Software prefetch in the scan and the search loop.~~ **Refuted**
   (findings 45). At
   d=1536 strawmANN's brute force demand-misses 24% of the corpus per query
   against Qdrant's 2.5%, at IPC 0.26, and W4's workers run at 0.31 against
   0.68 on the same DRAM traffic. Prefetching row `off + k` in
   `bruteForceRange` and the neighbour rows in the HNSW loop, then W9 and W4
   on db100k with counters on, is the experiment; findings 2 already measured
   the window it would be filling.

   ~~**Done, and the answer is no** (2026-08-30).~~ Both prefetch shapes were
   built and measured. The whole-row version halves W9's demand DRAM fills
   (4.56 G to 2.14 G) and changes the time by 1%: the scan runs at 61.8 GB/s
   against a 73.1 GB/s bus, so it is bandwidth-bound and there is no latency
   left to hide. Findings 37 had already measured 61 GB/s on a corpus ten times
   larger and called it bandwidth-bound; findings 45 contradicted it and was
   wrong. The code is reverted and this item is closed as refuted rather than
   done.


11. ~~A refusal when an aggregate's environment hash is not the run's.~~
   **Done** (findings 46). `bench/results/noise.json` and the twelve-run table in
   findings 38 are scoped to one environment hash, and the harness has no
   opinion when a figure from a different one is read against them. The
   SMT-off/SMT-on comparison that cost this session two diagnostic runs was
   made in prose, by a reader who had the hash on screen. The cross-corpus
   and cross-engine refusals already exist and are the same shape;
   `regression.floor_for` already knows which run it is judging. Cheap, and
   it forecloses a mistake that reads as a 30% engine regression.

   ~~**Done** (2026-08-30).~~ `bench/setup.py` already printed `env_hash=` and
   nothing read it. `workloads.py` now parses it from the gate's own output.
   Beside `profile`, and for the reason that comment gives, so `setup.py` stays
   the one definition, and stamps it into `run.json`. `aggregate.py` copies it
   into each folded `noise.json` next to `dataset`, `regression.floor_for`
   carries it through a fold and resolves a disagreement to `None`, and
   `compare.parity_band` refuses a floor whose environment is not the run's.
   Permissive where it cannot know, like the dataset rule: an unstamped floor
   predates this and stays usable, a run with no hash cannot contradict
   anything, and only a disagreement refuses. One arm disagreeing is enough,
   because a ratio's band is both arms' spread.

12. **An SMT-off pass over the settled sift1m labels** (findings 46). Across
   the SMT change strawmANN moved 1.5% and Qdrant 30%, which is what findings
   36 and 41 predict from one-thread-per-core pinning against 43 multiplexed
   threads, but it is one environment against a recollection of another, and
   that is the error findings 46 is about. One pass at SMT-off, same labels,
   settles whether the asymmetry is real. Until it exists the pairing is a
   hypothesis and must not be quoted.
13. ~~What Qdrant's W9 is actually doing.~~ **Done** (findings 45). It
   serves 128.1 qps
   where a full fp32 scan of the 614.4 MB corpus would need 78.7 GB/s, above
   the 73.1 GB/s this host delivers, with a quarter of strawmANN's demand
   fills. Something is reading less than the corpus. Until it is known whether
   that is cross-query caching, a narrower representation, or a non-exhaustive
   path, W9 is not an exact-search comparison and its ratio must not be quoted
  . The same defect findings 21 found in the first recall run.

   **Both sides read, 2026-08-30.** strawmANN's exact path is
   `search(..., .exact, ...)` falling straight through to
   `bruteForce(coll, query, out)`: one full scan of the collection, per query,
   on the worker that dequeued it. There is no batching interface anywhere on
   that path. Qdrant's plain (exact) index is built around the opposite shape,
   `search(query_vectors: &[&QueryVector], ...) -> Vec<Vec<ScoredPointOffset>>`,
   feeding a `BatchFilteredSearcher` whose `peek_top_visible` walks the storage
   once and scores *every query in the batch* per harvested block of points
   (`point_scorer.rs`, `VECTOR_READ_BATCH_SIZE`). One engine re-reads the
   corpus per query; the other is written to read it once per batch.

   What is not yet established is whether W9 fills that batch. bfb sends one
   query per request at `-p 8`, so the slice may well arrive length-1 and the
   amortisation come from somewhere else. Eight concurrent scans of one
   614 MB array can convoy in LLC, the follower riding lines the leader pulled.

   **The measurement that separates them, and a number that makes it
   falsifiable.** At `-p 1` no sharing of any kind is available, so a full
   fp32 scan is bounded by the bus: 73.1 GB/s / 614.4 MB = **119 qps, for
   either engine**. Run W9 at `-p 1` and `-p 8` on both. If Qdrant exceeds 119
   at `-p 1`, it is not scanning the corpus and this is a correctness question
   about the row rather than an efficiency one. If it obeys the bound at `-p 1`
   and beats it at `-p 8`, the amortisation is real and the only question left
   is which mechanism.

   ~~**Answered 2026-08-30: the amortisation is real.**~~ Qdrant obeys the
   bound alone (57.7 qps, and *slower* than strawmANN's 66.7, so nothing is
   being skipped) and beats it concurrently (128.1 qps, an implied 78.7 GB/s
   against a 73.1 GB/s bus). W9 is a fair row and its ratio stands. strawmANN
   runs at 90% of single-core bandwidth at `-p 1` and 85% of aggregate at
   `-p 8`, so it is at the ceiling of a design that re-reads the corpus per
   query. Closed; what it opens is item 14.

14. **Gathering concurrent exact queries into one scan** (findings 45). The
   only lever on a bandwidth-bound row is reading fewer bytes, and W9 has eight
   queries in flight over one 614 MB arena. Concurrency scaling is 1.51x
   against Qdrant's 2.22x. `handlers.BatchJob` already fans one request's
   queries across workers; the inverse, gathering independent in-flight
   queries into a single pass, does not exist. It trades latency for
   throughput and needs a deadline, so it is a design question rather than a
   patch, and it is the only named change with a quantified ceiling at
   d=1536.

15. **A decision on whether to keep an h5 copy per tier.** bfb's accuracy
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
