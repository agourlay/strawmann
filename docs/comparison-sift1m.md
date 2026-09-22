# strawmann vs Qdrant 1.19 — SIFT1M

**The generated tables below are from a gated run; the prose after them is
not.** Which run, on what host, under which gate and conformance hash, is
recorded *in* the generated block and in `bench/results/<label>/run.json`,
because a sentence up here saying which run the tables carry was wrong within
two runs of being written.

Everything hand-written below, §1's conformance detail and §4's caveats,
describes **earlier** measurements, taken on a `powersave` governor with boost
on, and in some cases with the parallel index build confined to a single core by
the affinity bug fixed on 2026-08-17, "The parallel index build runs on all
the server's cores, not one worker's". Read it for the reasoning, not for the
figures, until each section is re-derived.

**§2, §3 and §4f's "What it says" were re-derived against the 2026-08-21 run**
and carried current verdicts then; the generated tables have moved on twice
since (2026-09-03 and 2026-09-07), so treat those verdicts as the last ones
checked rather than as this run's. That distinction matters: a stale *figure* is
covered by the paragraph above, but a stale *conclusion* is not, and those three
sections had each reversed: T3 failed and now passes, the frontier put Qdrant
ahead above recall 0.96 and now does not, and "faster on every search row" was
never true at d=1536.

The generated tables are the median of three interleaved passes per engine
(§7.4), which is not the same as a repeated *measurement*: it is one run, on
one host, on one day. This host is a developer laptop, so `as-deployed` here
means "a developer machine", not a cloud deployment.

## Pinned target (§8.9)

The image digest, both client versions, the dataset checksums, the host and the
conformance hash are recorded per arm in `bench/results/<label>/run.json` and
summarised under the generated table. They were transcribed here once and went
stale immediately: the digest named a different image than the one being run,
and the host line described a 24-core machine this comparison has never used.
§8.9 asks for the target to be pinned, not for it to be pinned twice.

Both collections are created with identical HNSW parameters, `m=16` and
`ef_construct=100`, from `workloads.collection_settings()`.

---

## 1. Conformance: T0, T1, T2 pass (the run now reaches T4)

**T0 wire conformance: PASS.** Response shapes identical, with only timings and
version strings on the ignore-list. The hand-written HTTP/2 + HPACK + protobuf
stack is wire-compatible with the real server, checked through the same client
rather than against our own decoder.

**T1 exact-search value equality: PASS, and exactly.**

```
n=100000  max=0.000e0  p99.9=0.000e0  p99=0.000e0  p50=0.000e0  mean=0.000e0
vs fp64 oracle: ours max 1.526e-5, Qdrant max 1.526e-5
```

Not ε-bounded, **bit-for-bit identical** across 100,000 score comparisons, and
both engines diverge from the fp64 oracle by precisely the same amount, meaning
they make identical fp32 rounding decisions on the same summation order. This is
what §8.3's semantics table bought: ranking on the squared distance, returning
the square root, summing in strict f32 order.

**T2 rank agreement under ties: 4 of 98,676 equivalence classes differ**,
Kendall τ = 0.9981, RBO = 0.9977. That was recorded as a pass at the time;
under the current `compare_ranks` a non-zero count of differing ε-tie segments
is a T2 **FAIL** (`mismatched == 0` is required). The licensed 2026-08-17 run
reports `0/98676 ε-tie segments differ; Kendall τ=0.9970, RBO=0.9974` and
passes T2 on that rule.

## 2. T3 passes now; it used to fail, and the reason it failed is architectural

**Current status, 2026-08-21 (`conformance.json`, hash `21475c99fb41115d`): T3
passes.**

```
strawmann  recall@1 0.9954  recall@10 0.9888 [0.9866, 0.9907]  MRDE 2.55e-4
qdrant     recall@1 0.9930  recall@10 0.9879 [0.9855, 0.9898]  MRDE 3.29e-4
```

The confidence intervals overlap, so the differ licenses a comparative claim at
this operating point, and the run reaches T4 with every tier passing.

The rest of this section is the earlier reading, kept because its explanation is
still the right one for why `ef` is not a comparable knob between these engines.
It is history, not status:

```
strawmann  recall@1 0.9918  recall@10 0.9842 [0.9816, 0.9865]  MRDE 4.33e-4
qdrant     recall@1 0.9919  recall@10 0.9989 [0.9980, 0.9994]  MRDE 5.30e-5
```

Those intervals did not overlap, and the differ refused to license a comparative
claim.

The cause was not a defect. Qdrant built the collection as **8 segments**;
strawmann holds **one graph**. Qdrant searches all 8 at the requested `ef` and
merges, so a nominal `ef=64` is roughly 8× the work against graphs covering
~125k points each. Smaller graphs also have better recall per unit of search
effort. Identical `m` and `ef_construct` therefore do *not* make `ef` a
comparable knob between these two engines.

**Recall at matched `ef` is not a meaningful comparison here, in either
direction.** The frontier is.

## 3. The recall/throughput frontier

**Superseded, in both its data and its conclusion.** §3a below already inverted
it once, and the 2026-08-21 run settles it: at matched `ef` the two engines now
reach the *same* recall, so the sweep reads directly off the W10 rows in §4f
with no interpolation and no crossover.

| ef | recall@10 (sm / qd) | strawmANN q/s | Qdrant q/s | ratio |
|--:|---|--:|--:|--:|
| 128 | 0.9860 / 0.9852 | 21,057 | 10,692 | **1.97x** |
| 256 | 0.9954 / 0.9958 | 11,665 | 6,842 | **1.71x** |
| 512 | 0.9984 / 0.9991 | 6,513 | 4,082 | **1.60x** |

The claim below, that the crossover sits at recall 0.96 to 0.97 and that Qdrant is
ahead above it by 1.63x and then 1.92x, does not hold on this run. strawmANN
leads by 1.60x to 1.97x across exactly the high-recall band that claim was about.
The mechanism it blamed, Qdrant's 8-segment build reaching high recall at low
`ef`, is also absent: recall per `ef` now matches to within 0.001 at ef=128 and
above, which is what let T3 pass (§2).

That said, this is a d=128 reading. On dbpedia-openai-100K the same three rows
read 1.05x, 1.00x and 1.00x, parity, not a lead.

The original measurement follows, kept as history: 2000 held-out queries,
`limit=10`, same harness, same ground truth.

| ef | strawmann recall@10 | strawmann q/s | qdrant recall@10 | qdrant q/s |
|--:|--:|--:|--:|--:|
| 16 | 0.7691 | 27546 | 0.9261 | 8844 |
| 32 | 0.8821 | 17822 | 0.9737 | 6968 |
| 64 | 0.9498 | 10128 | 0.9935 | 4709 |
| 128 | 0.9817 | 5480 | 0.9991 | 3082 |
| 256 | 0.9939 | 2895 | 0.9998 | 2011 |
| 512 | 0.9977 | 1606 | 0.9999 | 1193 |

Read at **matched recall**, which is the only honest way to read it:

| recall@10 | strawmann | qdrant | |
|--:|--:|--:|---|
| ≈0.95 | 10128 q/s (ef 64) | ~8000 q/s (interp.) | strawmann ~1.27× |
| ≈0.9937 | 2895 q/s (ef 256) | 4709 q/s (ef 64) | **qdrant 1.63×** |
| ≈0.998 | 1606 q/s (ef 512) | 3082 q/s (ef 128) | **qdrant 1.92×** |

At `ef` 256 strawmann reaches recall@10 = 0.9939 [0.9894, 0.9965] and Qdrant at
`ef` 64 reaches 0.9935 [0.9889, 0.9962], overlapping intervals, so this is a
T3-equivalent matched-recall pair. **Qdrant serves it at 1.63× the rate.**

**The crossover is around recall 0.96 to 0.97.** Below it strawmann is ahead;
above it, which is where production systems actually operate, Qdrant is
ahead, and the gap widens with recall.

That contradicts the project's implicit premise. A strawman with no
compatibility debt, no distribution, and no product concerns is *losing* at the
recall levels that matter, against an engine carrying all three. §11's rule
applies in reverse here: this is recorded rather than explained away.

The likely mechanism is the same segmentation that broke the `ef` comparison.
Eight graphs of 125k points reach high recall with far less traversal than one
graph of 1M, because HNSW's search cost grows with graph size while its recall
per unit effort does not. Segmentation is a genuine engineering advantage, not
an artefact, and strawmann does not have it.

### 3a. That conclusion did not survive a better load generator

Everything above is measured with the conformance harness's own client: one
connection, batched queries, sequential. §4 below flags that as the weaker of
the two problems with this axis and predicts what it might cost: *"At low
concurrency that is a real advantage for Qdrant **in this measurement** which
would not survive a saturating load... The ranking above could invert under
W4."*

It inverts. Joining the same recall sweep to **bfb's** W10 throughput at the
same `ef` values, rather than to the conformance client's rate:

| recall@10 | measured at | strawmANN q/s | Qdrant q/s | ratio |
|--:|---|--:|--:|--:|
| 0.9742 | Qdrant ef=32, strawmANN interpolated | 29,254 | 8,024 | **3.65x** |
| 0.9815 | strawmANN ef=128, Qdrant interpolated | 26,470 | 7,069 | **3.74x** |
| 0.9934 | Qdrant ef=64, strawmANN interpolated | 15,313 | 5,758 | **2.66x** |
| 0.9938 | strawmANN ef=256, Qdrant interpolated | 15,069 | 5,578 | **2.70x** |
| 0.9975 | strawmANN ef=512, Qdrant interpolated | 8,221 | 3,969 | **2.07x** |

strawmANN is ahead at every matched recall, and the gap *narrows* with recall
rather than reversing. There is no crossover in the measured range.

Both readings are real measurements; they differ in what generated the load,
and that is the finding. A single batching client leaves an engine that can
parallelise one query across eight segments with idle cores to use, and leaves
an engine that cannot with nothing to do. Under bfb's concurrency every core is
already busy serving other queries, and intra-query parallelism stops being
free. **Which means the earlier conclusion was a property of the client, not of
the engines, and it was published as if it were the second.**

What this pairing assumes, stated because it is not nothing: throughput comes
from the bfb run and recall from a later conformance sweep, on the same dataset
with the same `m` and `ef_construct` but a separately built graph, and the
interpolated column is linear in log(q/s) between bracketing measurements with
no extrapolation past them. The host is still ungated. The generated version of
this table, with its provenance attached, is in the HTML report.

## 4. Why the throughput axis is not yet trustworthy

Larger than the §7.1 gate failure:

- **The harness is closed-loop and low-concurrency.** It sends `QueryBatch` of
  32 sequentially. That is a latency-ish measurement, not a saturation
  throughput, and it is exactly the coordinated-omission shape flagged when
  re-deriving W4.
- **Qdrant can use intra-query parallelism across its 8 segments; strawmann has
  nothing to parallelise within one query.** At low concurrency that is a real
  advantage for Qdrant *in this measurement* which would not survive a
  saturating load, where every core is already busy with other queries. The
  ranking above could invert under W4.
- **Neither engine was pinned, and the host is unquiesced.**

So the correct reading is: **the recall numbers are solid, the throughput
numbers are indicative and unsafe to quote.** Settling it needs bfb (not built
here), the fixed-rate `--rps` arm of W4, and a host that passes §7.1.

## 4a. The two engines do not mean the same thing by "time to Green"

Watching the headline load, Qdrant indexes **incrementally during ingest**, at
137,592 points uploaded it already reported 127,386 indexed, status Yellow -
while strawmann ingests everything and then bulk-builds one graph before
publishing Green.

That is not a detail. §4's W2 is defined as "W1 + time-to-Green", and the two
architectures make that measure different quantities:

- for **strawmann** it is a clean bulk-build time, with ingest and index
  strictly separated;
- for **Qdrant** it overlaps ingest and indexing, so the wall-clock includes
  work strawmann has already finished, and the *ingest* number is depressed by
  indexing running concurrently.

So W1 and W2 cannot be compared row-for-row between them; only W1+W2 together is
meaningful, and even then Qdrant is doing the extra work of keeping the
collection queryable throughout. Bulk-loading into a quiescent index is the
easier problem, and strawmann solves the easier problem.

## 4a-ii. Qdrant's control plane starves under bulk ingest at d=1536

From the container log during the headline load:

```
"GET /collections/conformance HTTP/1.1" 200 847 ... 55.770117
```

**A collection-info request took 55.8 seconds.** Not a failure, the server was
healthy and indexing throughout, but the control plane is effectively starved
while the optimizer runs at the headline dimension.

Two consequences:

- §2 has bfb polling `collection_info` **once per second** during upload and
  requiring Green three consecutive times. Those polls queue behind indexing
  work, so "time to Green" as bfb measures it includes control-plane queueing
  that has nothing to do with index construction.
- It is the fuller explanation for the client timeout that killed the previous
  attempt. "Upserts are slow" was only half of it; the whole gRPC surface is
  contended.

Both are properties of the comparison target under load, and both argue for
timing ingest and indexing from the *client's* completed-work count rather than
from a status endpoint.

## 4b. A setup error worth not repeating: Qdrant on tmpfs

The first headline attempt failed with

```
status: red
optimizer: Not enough space available for optimization,
           needed: 1004.30 MiB, available: 532.25 MiB
```

Qdrant's storage volume had been placed under the session scratchpad, which is
on **`/tmp`, a tmpfs, RAM-backed**. Two things were wrong with that, and the
second is worse than the failure:

1. Its 5.8 GB of segment data was consuming the same RAM the two engines were
   competing for, so tmpfs free space collapsed as memory filled and the
   optimizer's space check failed. The container reported 21 GB free on the
   mount while the optimizer saw 532 MiB, which is what tmpfs accounting looks
   like under memory pressure.
2. **Measuring a disk-backed engine on a RAM disk is not a measurement of that
   engine.** Qdrant's segment layout, mmap behaviour and optimizer are all
   designed around real storage. Had the run *succeeded* on tmpfs it would have
   produced numbers that looked fine and meant nothing, the more dangerous
   outcome.

Moved to the home partition (377 GB free, real block device). Worth stating in
the environment record alongside the §7.1 gate: **where the comparison target's
storage lives is part of the comparison.**

## 4c. T1 fails at d=1536, and the oracle says it is **Qdrant** that drifts

The headline run failed T1:

```
max |Δ| 1.967e-6   rel max 2.293e-6   vs ε 6.309e-7 (the §8.4 *floor*)
```

with **identical ids in identical order** (`ours 851798 @ 0.905938387,
theirs 851798 @ 0.905939102`) and τ = 1.0000 / RBO = 1.0000 across 10,000
queries. The obvious reading, and the one taken at first, is that the ε floor
is simply too tight for a 1536-term accumulation, and calibration would clear it.

**That reading was wrong.** Calibrating ε properly (§8.4), from strawmann's own
spread across the baseline / avx2 / avx512-full arms:

```
strawmann cross-ISA spread  max 2.384e-7  (rel max 2.987e-7)
qdrant run-to-run spread    max 0.000e0   (deterministic, its ISA cannot be forced,
                                           so this measures determinism, not dispatch)
eps(cosine, d=1536) = 9.537e-7 = 4 x 2.384e-7
```

The observed delta of 1.967e-6 is still **2× above** the calibrated ε. The
divergence is real, not a tolerance artefact.

Measuring both engines against the fp64 oracle localises it:

| | max error vs fp64 | mean |
|---|--:|--:|
| **strawmann** | **1.501e-7** | 3.80e-8 |
| qdrant | 1.358e-6 | 1.63e-7 |

**Qdrant is roughly 9× further from the true value than strawmann at d=1536
cosine.** The differ reaches the same conclusion independently and prints it:
*"the oracle says QDRANT is further off"*.

**This explanation is incomplete, and the missing piece is measured in
[`findings.md`](findings.md) §14.** Cosine normalises at ingest, and the two
engines do not normalise identically: on any AVX host Qdrant sums the length
with four accumulators and multiplies by `1.0/√length`, while this engine sums
strictly and divides. At d=1536 that difference alone produces up to
**1.490e-6** of score divergence, against the 1.967e-6 measured here and the
9.537e-7 ε. So some of what is attributed to accumulation depth in the distance
kernel belongs to the vectors having been different in storage before any
distance was computed.

So T1's failure is not strawmann deviating from a reference, it is the two
engines disagreeing, with strawmann the more accurate of the two. At d=128
Euclid the same comparison was bit-identical (§1), because 128 terms accumulate
identically under both summation structures; 1536 terms do not.

### The decision, now made

§8.3 has been amended: **semantics are replicated exactly; arithmetic error is
not.** Which similarity is ranked on, what postprocessing is applied, what
happens at ingest, those must match, because a difference there is systematic
and no tolerance absorbs it. Floating-point accuracy is a separate axis, and on
that axis strawmann does not chase Qdrant.

T1 is therefore graded against the **fp64 oracle** rather than against Qdrant:

- agree within ε → pass;
- disagree, but strawmann is within ε of the oracle → **pass**, recording that
  Qdrant is the engine that drifted;
- disagree and strawmann is further from the oracle → fail; that is a real
  defect and the tier still catches it.

The old rule made Qdrant the definition of correct, so strawmann failed a
conformance tier *for being more accurate*, and the only remedy would have been
to reproduce Qdrant's rounding error on purpose. §8.1 already conceded
bit-exactness was unachievable, which means equality with Qdrant was never the
requirement; being right was.

One guard came out of writing the adversarial test for this: the first version
excused *any* discrepancy when we were closer to the oracle, **including a
mismatched result count**. That is a compatibility defect no amount of numerical
accuracy repairs. Structural divergence is now tracked separately and is never
excused; `a_result_count_mismatch_still_fails` pins it.

Practically the difference is negligible for ranking: τ = 1.0000 means no query
in 10,000 had its result order changed. It matters for a tier that asserts value
equality, and it means the headline conformance hash carries `tier reached: T0`
for a reason that is not a defect in this engine.

## 4d. The headline tier, final

Re-run with the calibrated ε (9.537e-7) and T1 graded against the oracle.

```
T0 wire conformance    PASS
T1 exact-search value  PASS   ours 1.982e-7 vs oracle | Qdrant 1.976e-6 | eps 9.537e-7
T2 rank agreement      32/99978 tie classes differ; Kendall tau=1.0000, RBO=1.0000
T3 ANN statistical     FAIL   strawmann recall@10 0.9557 [0.9515, 0.9596]
                              qdrant    recall@10 0.9864 [0.9840, 0.9885]
metamorphic            prefix_property, offset_consistency: both hold
tier reached           T2RankTies
conformance hash       955f69f1c82af3b8
```

(As graded then. `compare_ranks` now fails T2 on any differing tie segment, so
this transcript would read `tier reached T1` today; the licensed run above the
fold has 0 differing segments.)

T1's verdict line, in full:

```
vs fp64 oracle: ours max 1.982e-7, Qdrant max 1.976e-6, the oracle says QDRANT is further off
ACCEPTED: we are within ε of the fp64 oracle (1.982e-7 <= ε 9.537e-7); Qdrant is the
engine that drifted, and §8.3 does not require reproducing its error
```

**strawmann is 10× closer to the true value than Qdrant at d=1536 cosine.** The
tier now records that as a pass instead of punishing it, and τ = 1.0000 confirms
no query in 10,000 had its ordering changed by the difference.

**T3's numbers reproduce.** Against the earlier run: 0.9557 vs 0.9561 for
strawmann, 0.9864 vs 0.9874 for Qdrant. That stability matters, the recall gap
is an architectural property, not run-to-run noise, and nothing changed in this
round touched it.

Both gates behaved as intended:

```
§8:   LICENSES single-engine performance rows (kernels, cost model, ISA matrix).
§7.4: does NOT license a strawmann-vs-Qdrant throughput or latency comparison
      - T3 did not pass, so the engines are at unequal recall and the faster one
        may simply be searching less.
```

### The scoreboard

| axis | verdict |
|---|---|
| wire compatibility | strawmann matches Qdrant exactly, both datasets |
| numerical accuracy | **strawmann better**, bit-exact at d=128, 10× closer to truth at d=1536 |
| ranking agreement | τ = 1.0000 |
| recall at matched `ef` | **Qdrant better**, 0.9864 vs 0.9557 |
| throughput | **not licensed**, T3 must pass first |

### The "recall deficit" was a units error, mine

The paragraph that stood here called the T3 gap "the one substantive open
problem" and recommended considering segmentation. **That was wrong**, and two
experiments show why.

**A build-parameter sweep on the headline tier moves nothing:**

| m | ef_construct | ef 64 | ef 128 | ef 256 | ef 512 | build |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 100 | 0.9227 | 0.9549 | 0.9710 | 0.9821 | 161.7s |
| 16 | 400 | 0.9219 | 0.9542 | 0.9716 | 0.9828 | 189.3s |
| 32 | 200 | 0.9215 | 0.9552 | 0.9718 | 0.9832 | 186.5s |
| 48 | 400 | 0.9225 | 0.9541 | 0.9710 | 0.9821 | 327.4s |

Tripling `m` and quadrupling `ef_construct` changes recall by **less than 0.1
point**, while search `ef` moves it six points across the same rows. The
parameters were applied, build time doubled at m=48, so the graph genuinely
got denser and recall did not care. A graph-*quality* problem would respond to
graph-quality knobs.

**Raising `ef` alone closes the gap:**

| strawmann ef | recall@10 | | qdrant |
|--:|--:|---|--:|
| 128 | 0.9537 | | ef 128 → **0.9864** |
| 512 | 0.9831 | | (8 segments) |
| **1024** | **0.9882** | ← matches/beats Qdrant | |
| 2048 | 0.9919 | | |

The ratio is exactly what the architecture predicts. Qdrant searches `ef=128` in
*each* segment and merges, and the default segment count is capped at 8, so
its real budget is up to 1024 node
visits; strawmann needs ~1024 to match. **`ef` is not the same unit in the two
engines**, and comparing recall at equal `ef` compares two different work
budgets, which is what §2 of this document said about the *frontier* and what
this section failed to apply to the recall number itself.

So the open question is not "why is our recall worse". It is: **at matched
recall, is strawmann's single-graph traversal cheaper in total CPU, and does
Qdrant's advantage survive a saturating load?** At ~0.988 recall strawmann
serves 320 q/s to Qdrant's ~3000 in this harness, but that harness is
closed-loop at low concurrency, where Qdrant can fan one query across up to 8
segments and strawmann cannot parallelise within a query at all. W4's fixed-rate
arm is what settles it.

## 4e. Every number on this host was taken under unmeasured load

§7.1's gate checks governor, boost, SMT, isolation, THP, NUMA, perf paranoia and
toolchain, eight *configuration* properties, every one of which can be correct
on a machine that is busy. It had no notion of whether anything else was
running.

Adding that check found a third process immediately:

```
a local LLM inference server  1008% CPU  16:54 elapsed
```

Ten cores' worth of unrelated work, overlapping the W0-W13 table. Nobody was
tracking it. The two identical W0 runs that differed by 53% (5,290 vs 8,084 rps)
now have an obvious candidate cause, and so does the earlier claim, made here -
that the strawmann table ran on a "near-idle box" while only the Qdrant table was
contaminated. Both were contaminated.

`check_quiescent` now fails the gate above 10% load per core and names the busy
processes (because "load is high" is not actionable). Load was briefly fed into
the environment hash and then taken out again (`setup.py` `observe`): a float
in the hash made every run a different environment. The hash answers "same
machine, same configuration"; ambient load is recorded per row (`load_start`,
`load_end`, `foreign` as a per-pid CPU delta across the row) and a row with
`foreign` set is refused a ratio, which is the stronger rule.

This is the difference between a benchmark that is correct when operated
carefully and one that can be trusted by someone who is not watching. A redline
has to be the second kind.

## 4f. The full W0-W13 comparison, both engines measured alone

Generated by `bench/harness/compare.py strawmann qdrant`. Each engine is
measured with the other stopped, and every row records its own start/end load,
any foreign process, and the §7.1 verdict in force when it ran. The host and
that verdict are printed under the table by the same command, so they describe
the run the table came from; this paragraph used to describe a 24-core box the
table had not been measured on for some time.

**Units are queries per second.** bfb computes `rps = per_sec() /
search_batch_size`, so on a batched row `rps` counts batch *requests*. Reading
`rps` on W5 (batch 16) made it look like a 7x regression against the unbatched
W3; it is 2.2x faster. Two independent runs agreed on the wrong number, because
repetition does not correct a units error, the harness now reports qps and
shows both only where they differ.

This table is spliced in by `compare.py --write-readme` and checked by the gate,
so it is the current run rather than a transcription of one. Row names come from
`workloads.py`, which is §4's only definition of the table.

**Methodology of the `qps` and `ratio` columns.** For rows measured by the
current harness, `qps` is wall-clock: the number of queries bfb issued divided
by the `duration_secs` in its own JSON. bfb's printed `Median qps` is the
median of a per-request rate series and understates rows that finish in a
second or two (strawmANN's W4: 27,201 printed against 31,056 by the clock); it
is kept per row as `qps_bfb_median` and rows shorter than 2 s carry a `short`
note. A `ratio` is printed only where §7.4 permits one: both engines' recall@10
at the row's `ef`, joined from the sweep of *the collection the row searched*
(so W6/W7/W8 join their quantized sweeps, not `bench2`'s fp32 one), known and
within 0.01. Otherwise the cell is `-` and the note says which half is missing
or how far apart they are. If the two result sets were not produced by the same
harness configuration (`run.json` `harness` stamp: metric, query source, sizes,
collection settings) the whole table is **STALE** and carries no ratio at all;
if the differ's row does not set `licenses_comparative`, the banner says the
ratios are **not a licensed comparative claim**. The published tables below are
currently in both states, and say so.

<!-- BEGIN compare-full (generated by bench/harness/compare.py --write-readme) -->
| id | workload | sm-sift-perf-rel-0921 qps | qd-sift-perf-rel-0921 qps | ratio | sm-sift-perf-rel-0921 p50/p99 | qd-sift-perf-rel-0921 p50/p99 | recall@10 | notes |
|---|---|--:|--:|--:|--:|--:|--:|---|
| W0 | d=4 floor: graph traversal with the distance taken out | 4,052 | 3,368 | 1.20x | 245 µs / 302 µs | 296 µs / 370 µs | - | 1.20x is 1.62x less work per query x 0.67x cores busy during the row (0.71 against 1.07) x 1.11x clock: the middle term is occupancy, not search speed |
| W3 | search, fp32, single query | 1,673 | 1,977 | 0.85x | 613 µs / 733 µs | 511 µs / 625 µs | 0.9873 / 0.9877 |  |
| W4 | search, saturating (closed loop) | 22,784 | 10,351 | 2.20x | 2.81 ms / 3.12 ms | 6.19 ms / 7.38 ms | 0.9873 / 0.9877 | 2.20x is 1.57x less work per query x 1.35x cores busy during the row (7.50 against 5.55) x 1.03x clock: the middle term is occupancy, not search speed |
| W4-sat50 | search, fixed rate at 50% of saturation (open loop) | 5,223 | 5,223 | offered | 495 µs / 1.09 ms | 777 µs / 1.72 ms | 0.9873 / 0.9877 |  |
| W4-sat70 | search, fixed rate at 70% of saturation (open loop) | 7,312 | 7,312 | offered | 499 µs / 992 µs | 919 µs / 1.86 ms | 0.9873 / 0.9877 |  |
| W4-sat90 | search, fixed rate at 90% of saturation (open loop) | 9,400 | 9,400 | offered | 511 µs / 1.10 ms | 1.11 ms / 2.33 ms | 0.9873 / 0.9877 |  |
| W5 | search batched (16 distinct dataset queries per request) | 23,140 | 4,871 | 4.75x | 1.38 ms / 1.54 ms | 6.55 ms / 7.14 ms | 0.9873 / 0.9877 | per-batch latency (16 queries/request); queries: dataset, random-sample 4.75x is 1.32x less work per query x 3.55x cores busy during the row (7.05 against 1.98) x 1.01x clock: the middle term is occupancy, not search speed |
| W6 | quantized: scalar | 4,222 | 4,877 | - | 463 µs / 661 µs | 410 µs / 488 µs | 0.9887 / 0.9638 | recall unequal: 0.9887 vs 0.9638; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W7 | quantized: binary + oversampling | 3,528 | 4,791 | - | 563 µs / 761 µs | 415 µs / 502 µs | 0.0580 / 0.0422 | recall below the usable floor: 0.0580 and 0.0422, both under 0.50 — this row characterises the encoding, not the engines |
| W8 | quantized: PQ | 3,018 | 4,497 | - | 681 µs / 813 µs | 444 µs / 537 µs | 0.9854 / 0.6978 | recall unequal: 0.9854 vs 0.6978; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W9 | exact / brute force | 191 | 144 | 1.33x | 37.23 ms / 69.42 ms | 54.59 ms / 75.08 ms | - | 1.33x is 0.78x less work per query x 1.70x cores busy during the row (6.95 against 4.08) x 1.00x clock: the middle term is occupancy, not search speed |
| W10-ef32 | recall control, ef=32 (latency only) | 40,444 | 20,663 | 1.96x | 189 µs / 255 µs | 365 µs / 691 µs | 0.9075 / 0.8990 | sm-sift-perf-rel-0921: -n 100000 (2x the table's 50000) |
| W10-ef64 | recall control, ef=64 (latency only) | 31,281 | 15,392 | 2.03x | 252 µs / 316 µs | 499 µs / 879 µs | 0.9626 / 0.9602 | sm-sift-perf-rel-0921: -n 100000 (2x the table's 50000) |
| W10-ef128 | recall control, ef=128 (latency only) | 20,405 | 9,607 | 2.12x | 394 µs / 489 µs | 816 µs / 1.34 ms | 0.9873 / 0.9877 | 2.12x is 1.59x less work per query x 1.30x cores busy during the row (6.74 against 5.17) x 1.03x clock: the middle term is occupancy, not search speed |
| W10-ef256 | recall control, ef=256 (latency only) | 11,925 | 5,573 | 2.14x | 677 µs / 900 µs | 1.43 ms / 2.19 ms | 0.9959 / 0.9970 | 2.14x is 1.44x less work per query x 1.46x cores busy during the row (7.02 against 4.81) x 1.02x clock: the middle term is occupancy, not search speed |
| W10-ef512 | recall control, ef=512 (latency only) | 6,465 | 3,134 | 2.06x | 1.24 ms / 1.75 ms | 2.56 ms / 3.75 ms | 0.9980 / 0.9993 | 2.06x is 1.30x less work per query x 1.57x cores busy during the row (7.08 against 4.51) x 1.02x clock: the middle term is occupancy, not search speed |
| W6-ef32 | SQ8 recall control, ef=32 (latency only) | 8,924 | 7,997 | - | 223 µs / 282 µs | 244 µs / 312 µs | 0.9079 / 0.8864 | recall unequal: 0.9079 vs 0.8864; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W6-ef64 | SQ8 recall control, ef=64 (latency only) | 5,958 | 6,521 | - | 339 µs / 439 µs | 302 µs / 379 µs | 0.9643 / 0.9415 | recall unequal: 0.9643 vs 0.9415; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W6-ef128 | SQ8 recall control, ef=128 (latency only) | 4,163 | 4,876 | - | 469 µs / 664 µs | 410 µs / 489 µs | 0.9887 / 0.9638 | recall unequal: 0.9887 vs 0.9638; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W6-ef256 | SQ8 recall control, ef=256 (latency only) | 2,835 | 3,238 | - | 720 µs / 842 µs | 624 µs / 729 µs | 0.9973 / 0.9716 | recall unequal: 0.9973 vs 0.9716; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W6-ef512 | SQ8 recall control, ef=512 (latency only) | 1,650 | 1,985 | - | 1.24 ms / 1.46 ms | 1.03 ms / 1.19 ms | 0.9995 / 0.9735 | recall unequal: 0.9995 vs 0.9735; §7.4 compares at equal recall; rescore pools differ (decisions §5) |
| W12-sel1 | filtered search, one keyword (~1% of bench12) | 2,560 | 3,812 | 0.67x | 787 µs / 854 µs | 520 µs / 606 µs | 1.0000 |  |
| W12-sel10 | filtered search, any of 10 keywords (~10% of bench12) | 301 | 2,223 | - | 6.76 ms / 7.32 ms | 902 µs / 1.09 ms | 1.0000 / 0.9895 | recall unequal: 1.0000 vs 0.9895; §7.4 compares at equal recall |
| W12-sel1-ef32 | filtered recall control, one keyword, ef=32 (latency only) | 2,562 | 6,465 | 0.40x | 787 µs / 857 µs | 304 µs / 380 µs | 1.0000 / 0.9959 |  |
| W12-sel1-ef64 | filtered recall control, one keyword, ef=64 (latency only) | 2,564 | 5,136 | 0.50x | 787 µs / 850 µs | 385 µs / 466 µs | 1.0000 / 0.9996 |  |
| W12-sel1-ef128 | filtered recall control, one keyword, ef=128 (latency only) | 2,564 | 3,832 | 0.67x | 787 µs / 853 µs | 517 µs / 603 µs | 1.0000 |  |
| W12-sel1-ef256 | filtered recall control, one keyword, ef=256 (latency only) | 2,562 | 2,617 | parity | 788 µs / 853 µs | 758 µs / 883 µs | 1.0000 | within the ±4.1% band this dataset's noise floor puts on W12-sel1-ef256: no measured difference, not a small one |
| W12-sel1-ef512 | filtered recall control, one keyword, ef=512 (latency only) | 2,567 | 1,714 | 1.50x | 786 µs / 859 µs | 1.16 ms / 1.32 ms | 1.0000 |  |
| W12-sel10-ef32 | filtered recall control, any of 10 keywords, ef=32 (latency only) | 382 | 4,614 | - | 5.26 ms / 6.06 ms | 428 µs / 549 µs | 0.9876 / 0.7962 | sm-sift-perf-rel-0921 +39% across its passes, monotonically: the spread on this row is drift rather than noise, so the median is a trend's midpoint and not a repeatable measurement recall unequal: 0.9876 vs 0.7962; §7.4 compares at equal recall |
| W12-sel10-ef64 | filtered recall control, any of 10 keywords, ef=64 (latency only) | 302 | 3,354 | - | 6.74 ms / 7.32 ms | 595 µs / 727 µs | 1.0000 / 0.9334 | recall unequal: 1.0000 vs 0.9334; §7.4 compares at equal recall |
| W12-sel10-ef128 | filtered recall control, any of 10 keywords, ef=128 (latency only) | 299 | 2,209 | - | 6.80 ms / 7.44 ms | 908 µs / 1.10 ms | 1.0000 / 0.9895 | recall unequal: 1.0000 vs 0.9895; §7.4 compares at equal recall |
| W12-sel10-ef256 | filtered recall control, any of 10 keywords, ef=256 (latency only) | 301 | 1,391 | 0.22x | 6.76 ms / 7.36 ms | 1.44 ms / 1.75 ms | 1.0000 / 0.9991 |  |
| W12-sel10-ef512 | filtered recall control, any of 10 keywords, ef=512 (latency only) | 299 | 836 | 0.36x | 6.80 ms / 7.46 ms | 2.40 ms / 2.91 ms | 1.0000 / 0.9999 |  |
| W13 | scroll / pagination | 51,030 | 42,965 | 1.19x | 137 µs / 233 µs | 168 µs / 361 µs | - | sm-sift-perf-rel-0921: -n 200000 (4x the table's 50000) qd-sift-perf-rel-0921: -n 100000 (2x the table's 50000) sm-sift-perf-rel-0921: 0.13 ms of the client's 0.14 ms p50 is not server time (12x), so 92% of this row is the load generator and the socket 1.19x is 4.07x less work per query x 0.30x cores busy during the row (1.39 against 4.69) x 0.98x clock: the middle term is occupancy, not search speed |
| W11-steady | mixed read/write below the rebuild threshold: search bench2 while 50,000 synthetic points append | 17,859 | 4,444 | - | 446 µs / 577 µs | 1.69 ms / 4.78 ms | - | append 2,000 points/s sm-sift-perf-rel-0921 +24% across its passes, monotonically: the spread on this row is drift rather than noise, so the median is a trend's midpoint and not a repeatable measurement search-during-write; no recall join |
| W11 | mixed read/write: search bench2 while 200,000 synthetic points append (runs last) | 1,794 | 1,773 | - | 3.42 ms / 13.51 ms | 3.76 ms / 14.48 ms | - | append 3,300 points/s sm-sift-perf-rel-0921 +25% across its passes, monotonically: the spread on this row is drift rather than noise, so the median is a trend's midpoint and not a repeatable measurement search-during-write; no recall join |

**Storage and I/O.** Queries per second is half the comparison: the two engines hold and write very different amounts for the same collection, and only this table shows what that buys and costs.

| | sm-sift-perf-rel-0921 | qd-sift-perf-rel-0921 |
|---|--:|--:|
| storage on disk | 3.7 GiB | 3.5 GiB |
| peak RSS | 6.1 GiB | 18.6 GiB |
| disk read ops | 9 | 49,287 |
| disk write ops | 42,702 | 405,205 |
| disk read bytes | 16.0 KiB | 1.9 GiB |
| disk write bytes | 2.7 GiB | 22.8 GiB |
| *syscall reads (all fds)* | *10,292,666* | *46,431* |
| *syscall writes (all fds)* | *4,888,923* | *10,363,579* |
| measured via | proc+cgroup | proc+cgroup |

Storage and RSS are end states; the rest are totals over every row. The syscall rows count every descriptor, sockets included, so on a search row they measure the network rather than the disk, and no ratio between them and the disk rows means anything. Storage on disk is the level before W11-steady, W11, the rows with a concurrent writer: an engine that rewrites segments is caught mid-rewrite there, and the same row has read 3.47 and 10.11 GiB on two runs of one binary. Peak RSS does include those rows, being a peak.

qps is wall-clock: queries / bfb's `duration_secs`, not bfb's `Median qps` (kept per row as `qps_bfb_median`), which is a median of a rate series and understates short rows. Latency is client-side round trip. §7.4: a closed-loop p99 is not a latency result, because a stalled server stops receiving requests and the tail it did not serve never appears; the `W4-sat` rows are the open-loop ones, and read `saturated` where a client p50 past one second says the offered rate was not served. A ratio is printed only where both engines' recall@10 at the row's `ef` is known and within 0.01; `-` with `recall unequal` or `recall missing` says which. A `-` under recall means the sweep does not speak for that row's configuration rather than that recall was poor.

**§8 conformance.** The tiers that license the table above.

| tier | result | detail |
|---|---|---|
| T0 wire conformance | pass | shapes identical (timings and versions on the ignore-list) |
| T1 exact-search value equality | pass | n=100000 max=0.000e0 p99.9=0.000e0 p99=0.000e0 p50=0.000e0 mean=0.000e0 (rel max=0.000e0 p99=0.000e0) \| ids agree up to ε-ties \| vs fp64 oracle: ours max 1.526e-5, Qdrant max 1.526e-5 |
| T2 rank agreement under ties | pass | 10000 queries, 0/98676 ε-tie segments differ; Kendall τ=0.9966 over 10000 queries (0 with <2 common ids excluded), RBO=0.9969 |
| T3 ANN statistical equivalence | pass | strawmann n=10000 recall@1=0.9954 recall@10=0.9888 [0.9866,0.9907] recall@100=n/a MRDE=2.5479e-4 \| qdrant n=10000 recall@1=0.9929 recall@10=0.9880 [0.9856,0.9899] recall@100=n/a MRDE=3.2889e-4 \| recall@10 CIs overlap — QPS is comparable at this point |
| §8.6 metamorphic properties | pass | 6 properties (strawmann/qdrant): prefix_property=ok/ok, offset_consistency=ok/ok, self_retrieval(exact)=ok/ok, permutation_invariance=ok/ok, query_order_invariance=ok/ok, idempotent_upsert=ok/ok |
| T4 quantization fidelity | pass | sq8/strawmann: \|Δscore\| over common ids n=97414 max=3.244e1 p99.9=1.251e1 p99=5.751e0 p50=1.806e-1 mean=4.564e-1 (rel max=1.988e-1 p99=3.835e-2) (5172 ids not common to both lists) \| Kendall τ vs fp64 oracle = 0.9467 over 10000 queries (0 with <2 common ids excluded) \| quantization_dominance holds |
| T4 quantization fidelity | pass | sq8/qdrant: \|Δscore\| over common ids n=97299 max=1.578e1 p99.9=2.020e0 p99=1.452e0 p50=3.786e-1 mean=4.496e-1 (rel max=7.840e-2 p99=8.895e-3) (5402 ids not common to both lists) \| Kendall τ vs fp64 oracle = 0.9413 over 10000 queries (0 with <2 common ids excluded) \| quantization_dominance holds |

Reached **T4 quantization fidelity**; licenses performance and comparative claims. Conformance hash `570a610ea49c9e9b`.

**sm-sift-perf-rel-0921**: §7.1 gate pass, peak system load 23% of one core (all processes, this engine included), measured 2026-09-21. **qd-sift-perf-rel-0921**: §7.1 gate pass, peak system load 25% of one core (all processes, this engine included), measured 2026-09-21.
<!-- END compare-full -->

The `ratio` column reads `offered` on the open-loop W4 arms rather than a
number, because their figure is the *offered* rate: both engines served 500,
1000 and 2000 qps on target, so those rows say "neither saturated", not "they
are equally fast".

W6's ratio is recall-matched but not quantizer-matched: strawmANN's SQ8 uses
256 levels and order-statistic quantiles, Qdrant 1.19's uses 127 levels and
effectively min/max bounds (`src/quant/scalar.zig`), so the row compares two
SQ8 encodings at equal recall, not one encoding on two engines. The same
applies to T4's `sq8` rows. (The 2026-08-18 run also kept the query in fp32
on strawmANN's side; the query has been quantised since, as Qdrant's is,
findings 29.)

The narrative below quotes an earlier run than the table above. Both the
figures and the thresholds that comparison rested on have been superseded.
It was written against the ungated noise floor, the one `validation.md` records
as replaced because it hid real differences behind a threshold that was never
about noise. Read the sections below for the reasoning; the table above is the
measurement.

### What it says

**strawmANN is faster on every rated search row at this dimension, by 1.17x to
4.78x**, and the qualifier "at this dimension" is load-bearing. On
dbpedia-openai-100K (d=1536) the same rows flatten: W4 goes to Qdrant at 0.94x,
W9 reads 1.01x, and W10-ef256 and W10-ef512 both read 1.00x. At d=1536 a single
distance is 1,536 multiply-adds, the query becomes memory-bandwidth bound, and
§6's savings (a flat level-0 graph, zero-allocation request handling,
comptime-specialised kernels) are per-request overheads that stop mattering
when each request does twelve times the arithmetic. See
[docs/comparison-dbpedia-openai-100K-1536-angular.md](comparison-dbpedia-openai-100K-1536-angular.md).

The two rows that keep their margin at d=1536 say the same thing from the other
side: **W5 keeps 2.47x** because batching amortises exactly the per-request
overhead §6 attacks, and **W13 keeps 1.14x** because scroll does no distance
work at all. Transport is where the strawman is genuinely faster.

That is still the expected shape (§1 removed distribution, payload storage,
filtering, replication and multi-tenancy) and a strawman that was *not* faster
on the rows it implements would mean the strawman was badly built. It is simply
a narrower claim than "faster on every search row" was.

**Four rows are refused a ratio, and three of them share one cause.** W6, W7 and
W8 all show strawmANN at materially higher recall than Qdrant, past §7.4's 0.01
tolerance, because `quantized_search.zig` sizes its stage-1 pool as
`if (rescore) @max(asked, ef)`: with W8's parameters (no
`--quantization-oversampling`, so 1.0, `rescore=true`, `limit=10`, `ef=128`)
strawmANN rescores the whole 128-node walk where Qdrant rescores a limit-sized
pool. The engines no longer answer the same question at `rescore=true`, and no
conformance tier asserts that they should, T4 checks quantization dominance and
Kendall τ against the fp64 oracle, not pool policy. W10-ef32 is refused
separately, on recall 0.8969 against 0.8853.

**W0's 1.17x is the least meaningful number here.** At d=4 the vector is
irrelevant, so it measures per-request overhead, a hand-written zero-allocation
path against a general-purpose server. Real, but a transport measurement rather
than an engine one.

**W11 is a loss: 489 against 1,045, or 0.47x.** Earlier revisions of this
document reported 7,260 and then 7,588 for this row and called it a 4.4x win.
Those numbers were real, and they measured almost nothing. Two corrections
landed since:

- Until the review passes of 08-17 ("Second review pass: the comparison was not
  one, and the engine dropped bfb's config") and 08-18 ("A review pass over the
  whole tree, and the regression tests it left behind"), a point overwritten *in
  place* did not count toward `needsRebuild`, whose tail term is
  `total - graph.count` and is zero for an overwrite.
- Until 08-20 16:45 ("No row may ask bfb for more points than the corpus
  holds"), W11 wrote with bfb's default `--offset 0`, so its "appends"
  overwrote ids 0..199,999 of a collection W2 had already filled.

Together those meant the row triggered no rebuild and grew no tail: **7,588 qps
was an uncontended traversal of a graph that had stopped describing its own
collection**, every point in it having moved. A previous "Correction" paragraph
here concluded from the same evidence that W11 "has no pending tail, and never
did". That was right about the binary and workload of the day and is wrong now:
`--offset upload_n()` makes every write a new point, so the tail is back.

With both corrections in place the row trips the rebuild trigger:

```
pending / covered = 200,000 / 1,000,000 = 0.20  >=  rebuild_ratio 0.10
```

so the collection steps aside to `.absent`, a from-scratch bulk build of 1.2M
points starts, and every query additionally scans the 200,000-point tail through
`scanPendingTail` while that runs. The tail alone accounts for roughly half the
latency directly, a fifth of W9's 1M scan is 36.88 ms ÷ 5 ≈ 7.4 ms against the
16.00 ms p50 observed with 8 requests in flight, and contention with the
rebuild over the same 8 pinned cores accounts for the rest.

**dbpedia-100K reads 0.51x on the same row**, which is the confirmation rather
than a coincidence: `W11_N` is scaled to hold its *ratio* to the corpus (200k
into 1M, 20k into 100k, both 0.20), so W11 trips `rebuild_ratio` **by
construction on every dataset**. It therefore only ever measures the rebuild
path, never steady-state mixed load.

Search itself does not collapse. `choosePath` traverses whatever graph is
published in *every* state, so the pre-2026-08-19 behaviour (any upsert setting
the index to `.absent` and every subsequent query brute-forcing the whole
collection, which is where the original 121 qps came from) is not back. Keeping
the graph is still the fix, and recall is unaffected by construction: a pending
point is compared against the query directly, which is strictly better than what
an approximate traversal would have done for it.

What is **not** yet established is why the row's average never recovers. §6.5
quotes 26 s for a 1M bulk build, so a rebuild should publish well inside the
102 s row and leave most of it fast. Either the build runs far longer under
contention or it does not complete; `server.log` carries no rebuild events, so
the question cannot be settled without adding them.

What remains true is the shape of the original finding. Qdrant degrades
gracefully under writes because it indexes incrementally into segments;
strawmANN rebuilds from scratch and pays for it in exactly the window W11
measures. Both are answers to the same question, and this row is where the
strawman's answer is the weaker one.

### Contamination, recorded rather than hidden

Two rows had to be re-measured after foreign processes appeared mid-run, and the
magnitudes calibrate how much that matters:

| row | contaminant | before | after | delta |
|---|---|--:|--:|--:|
| W9 (qdrant) | `mold(1870%)`, 18 cores | 92 | 130 | **+41%** |
| W11 (strawmANN) | `pinch-points(291%)`, 3 cores | 109 | 121 | +11% |
| W7 (qdrant) | `rustc(100%)`, 1 core | 3,762 | 3,580 | −5% (noise) |

The magnitudes are ordered the way you would hope: eighteen foreign cores cost
41%, three cost 11%, one is indistinguishable from noise on a 24-core box. W7's
re-run is still flagged because `rustc` reappears periodically, an editor probe
in this environment, and the flag is left standing rather than suppressed,
since the correct response to persistent low-level noise is to state it, not to
raise the threshold until it disappears.

Without per-row tracking, W9 would have entered the table 29% low and nothing
would have said so.

## 5. A gate defect this run exposed

The differ printed, one line apart:

```
T3 ... FAIL ... recall@10 CIs DO NOT overlap, a QPS comparison here would be
at unequal recall (§7.4)
§8: this run LICENSES a performance claim.
```

`licenses_performance_claim` required only T1. T1 establishes that the two
engines compute the same *values*, which is the right bar for a single-engine
number, a kernel microbenchmark, a cost-model row, an ISA-matrix cell. It says
nothing about whether the engines are doing the same amount of *work*.

A comparative throughput claim needs T3, because T3 is the tier that establishes
equal recall. Split into `licenses_comparative_claim`, and the differ now prints
both verdicts separately. Without it, this very run would have licensed the
frontier table above as a publishable Qdrant comparison.

Since then the licence has been tightened further, each by a case that had
passed: T3 on **zero queries** returned recall 0 with a CI of (0, 0), which
overlaps itself, and licensed a comparison of nothing (now every figure is NaN
and the match test refuses); `quantization_dominance` on a NaN recall reads
"not measurable" and fails rather than holding vacuously; the positional
metamorphic properties (`prefix_property`, `offset_consistency`) compare up to
ε-ties rather than position, because Qdrant's order inside a tie is not
deterministic and a duplicate-vector dataset was failing them for nothing; a
row whose build identity is `unknown`/`unpinned` (`STRAWMANN_COMMIT`,
`QDRANT_VERSION`, `STRAWMANN_ISA_BUILD` unset) licenses no claim, since "the
same build" is what §8 binds a perf row to; `--max-tier` is validated (T0-T4)
and `T2` stops before T3 instead of running everything but T4; and the hash
mixes fixed-width `u64` fields so it is the same number on a 32-bit host.

## 6. Not yet run

- **T4** (quantization fidelity), **it was never wired into a run.** The tier
  is implemented and unit-tested, and `--max-tier` defaults to `T4`, but the
  execution path stopped after T3 and silently reported `tier reached: T2`
  without anyone noticing T4 had not been attempted. §8.5 calls T4 "the tier
  that catches the most seductive class of bug", so the entire §6.7
  quantization axis was going unchecked against Qdrant while the CLI implied
  otherwise. Now wired: it builds an SQ8 collection on both engines and scores
  the same queries through the quantized and fp32 paths (`ignore: true` forces
  the latter), so the only difference between the two score sets is the
  encoding. Rescore stays off deliberately, with it on the measurement is of
  the rescorer, not of the encoding.
- **The headline tier.** dbpedia-openai-1m at d=1536 cosine is loaded in
  strawmann but has not been loaded into Qdrant, so no comparison exists at the
  dimension §4.2 calls the headline.
- **`calibrate`.** The differ used §8.4's ε *floor* rather than a measured
  cross-ISA spread. It did not matter here, the measured max delta was exactly
  zero, but it will on a metric where the engines genuinely diverge.


## Appendix: earlier readings, before both engines were measured alone

Kept because the corrections are the useful part.


Qdrant 1.19.0 pinned by digest, same client, same fp64 ground truth, SIFT1M.
The full write-up is the rest of this document.

**Correctness holds, on both tiers.**

| | SIFT1M (d=128, euclid) | dbpedia-openai-1m (d=1536, cosine) |
|---|---|---|
| T0 wire | PASS | PASS |
| T1 value | PASS, **bit-for-bit**, max \|Δ\| = 0.000e0 | PASS, ours 1.982e-7 vs oracle, **Qdrant 1.976e-6** |
| T2 rank | τ=0.9981 | **τ=1.0000** |
| T3 recall | FAIL 0.9842 vs 0.9989 | FAIL 0.9557 vs 0.9864 |

At d=128 the two engines agree bit-for-bit. At d=1536 they diverge, and the
fp64 oracle says **strawmann is 10× closer to the true value**. T1 is graded
against the oracle, not against Qdrant (§8.3), so being more accurate is a pass
rather than a failure.

**Performance does not.** Read at matched recall, the only honest reading,
since Qdrant runs 8 segments to strawmann's single graph and `ef` is therefore
not a comparable knob:

| recall@10 | strawmann | qdrant | |
|--:|--:|--:|---|
| ≈0.95 | 10128 q/s | ~8000 q/s | strawmann 1.27× |
| ≈0.9937 | 2895 q/s | 4709 q/s | **qdrant 1.63×** |
| ≈0.998 | 1606 q/s | 3082 q/s | **qdrant 1.92×** |

The crossover is around recall 0.96 to 0.97; above it, where production systems
run, Qdrant wins and the gap widens. A strawman with no compatibility debt is
losing to an engine carrying all of it, and the likely reason is segmentation:
eight graphs of 125k points reach high recall with far less traversal than one
graph of 1M.

**The throughput axis is not trustworthy yet**, the harness is closed-loop at
low concurrency, which flatters Qdrant's intra-query segment parallelism in a
way that would not survive a saturating load. Settling it needs bfb and a host
that passes §7.1.
