# Headline findings

The reason the project exists: the spec's job was to predict these, and where it
did not, that is the result.

**Development-grade, and for a reason that has since changed.** These were
measured before this host passed §7.1, `powersave` with boost on, so none of
the figures may be quoted in a chart. The host passes the gate now, which makes
them re-measurable rather than retroactively publishable; what a given run may
be quoted for is recorded per row and summarised in
[`validation.md`](validation.md). The qualitative findings hold across
repetitions; the numbers need re-measuring.

### 1. §6.6.1's central prediction is inverted

The spec predicts wide SIMD matters *least* on cold fp32 and *most* when cached.
Measured on Zen 5, AVX-512 gives fp32 **1.25 to 1.54× cold** and only **1.02 to 1.07×
L1-hot**.

§6.6.1 anticipated the possibility and named the cause: "512-bit loads are
improving memory-level parallelism by retiring fewer uops per byte". Halving the
instruction count lets the out-of-order window hold roughly twice as many
vectors' worth of loads in flight, and the cold case is short of exactly that.
Details in [`docs/isa-matrix.md`](isa-matrix.md).

### 2. §5.2's outstanding-miss constant is 4× low, and the gap is the prefetch budget

§5.2 assumes 10 to 16 outstanding line fills and derives 8.5 GB/s per-core random
access. Measured: saturation at **~64 misses and 25.3 GB/s**.

The spec's 8.5 GB/s is nonetheless right *for the concurrency an `M=16` HNSW
expansion supplies*, 7.84 GB/s measured at 16 chains. The 3.2× between them is
the budget available to §5.2's "single highest-leverage optimisation", now
quantified rather than asserted.

### 3. §11's open question 8 is answered: AVX-VNNI at 256 bits is enough

| | avx2 | avx2-vnni | avx512-full |
|---|--:|--:|--:|
| SQ8 symmetric dot, d=768, L1-hot | 141.6 cyc | **30.7** | 28.3 |

AVX-VNNI captures **92%** of the available 5× win. Full AVX-512 buys the
remaining 8%. A binary shipped to a fleet with Alder Lake and Zen 4 parts does
not need AVX-512 for the SQ8 path.

### 4. AVX-512 makes the fp16 kernel 1.5× *slower*

The one kernel where the wide build loses. This is the concrete instance of
§6.6.5's prediction that "mixed configurations are likely optimal", and the
`-Dforce-isa=` knob exists for it.

### 5. Qdrant's cosine short-circuit is dimension-dependent

§8.3 warns that Qdrant leaves an already-normalised vector untouched. Measured,
the window effectively closes as dimension grows: of 2000 mathematically-unit
vectors,

| d | pass the `\|Σx²−1\| ≤ 1e-6` test |
|--:|--:|
| 128 | 100% |
| 384 | 99.0% |
| 768 | 92.8% |
| 1536 | **79.1%** |

Confirmed on real data: of 2000 `dbpedia-openai-1m` ada-002 embeddings at
d=1536, **86.2%** pass with a strict f32 sum against **100%** with an f64 sum.
The vectors are unit vectors; whether Qdrant agrees depends on the summation
precision alone.

At the headline dimension a fifth of already-normalised vectors get divided a
second time. This makes computing the sum of squares in *strict f32 order* -
matching Qdrant bit for bit, mandatory rather than merely prudent: a
more-accurate sum puts a different population on each side of a discontinuous
branch. See `src/dist/norm.zig`.

### 6. Oversampling without rescore does nothing

Measured on binary quantization, recall@10 at d=64:

| oversampling | no rescore | with fp32 rescore |
|--:|--:|--:|
| 1× | 0.173 | 0.173 |
| 4× | 0.173 | 0.410 |
| 16× | 0.173 | 0.768 |
| 64× | 0.178 | **1.000** |

The no-rescore column is flat: widening the candidate set changes nothing if the
final ranking is still by binary score. §6.7's insistence on "the **combined**
curve ... rather than tuning stages independently" is not a preference, the two
knobs are not independent, and one is inert without the other.

### 7. At d=1536 cosine, strawmann is ~9× closer to the true answer than Qdrant

The headline differ failed T1, scores differing by 1.967e-6 against a
calibrated ε of 9.537e-7, with identical ids in identical order and Kendall
τ = 1.0000. Measured against the fp64 oracle, the divergence localises cleanly:

| | max error vs fp64 | mean |
|---|--:|--:|
| **strawmann** | **1.501e-7** | 3.80e-8 |
| qdrant | 1.358e-6 | 1.63e-7 |

At d=128 Euclid the two engines agree *bit-for-bit*; at 1536 terms the
summation structures diverge and Qdrant carries the larger error. ε here is
derived per §8.4 from strawmann's own spread across the baseline/avx2/avx512
arms (2.384e-7), not chosen, see [`docs/tolerance.md`](tolerance.md).

This leaves a design question open rather than closed: §8.3's premise is
*matching* Qdrant so score deltas are attributable to the index. Matching it at
d=1536 would mean reproducing its error on purpose.

### 8. An `ef` tuned on SIFT1M is badly wrong at d=1536

Same engine, same harness, same ground-truth methodology, recall@10:

| ef | SIFT1M | dbpedia-openai-1m |
|--:|--:|--:|
| 16 | 0.7711 | 0.7882 |
| 128 | 0.9816 | 0.9535 |
| 512 | 0.9973 | **0.9822** |

The headline tier starts *ahead* and ends 1.5 points behind, still climbing
where SIFT has flattened. SIFT hits 0.99 recall@10 around `ef` 150; dbpedia has
not reached it by 512. Tune `ef` on SIFT and ship it for 1536-dimensional
embeddings and you are at ~0.95 while believing you are at 0.99, which is the
argument for §4.2 having a headline tier at all.

A second effect only the headline tier shows: **recall@1 falls below recall@10**
past `ef` 64. The graph finds the right neighbourhood but misses the single
closest point ~2% of the time. Recorded as unexplained rather than rationalised,
per §11.

> **Re-measured 2026-08-28 at 1M, and it is not ours.** The aborted
> dbpedia-openai-1m run rebuilt `bench2` three times per engine, so the sign of
> this effect can be checked against five independent graphs rather than one.
> `recall@1 - recall@10`, fp32, 10,000 held-out queries:
>
> | ef | sm build 1 | sm build 2 | sm build 3 | qd build 1 | qd build 2 |
> |--:|--:|--:|--:|--:|--:|
> | 32 | +0.0022 | +0.0038 | +0.0026 | +0.0046 | +0.0054 |
> | 64 | **-0.0021** | **-0.0019** | **-0.0022** | **-0.0019** | **-0.0031** |
> | 128 | -0.0000 | -0.0009 | -0.0005 | -0.0018 | -0.0029 |
> | 256 | +0.0004 | -0.0010 | -0.0008 | -0.0013 | -0.0021 |
> | 512 | +0.0002 | -0.0003 | -0.0007 | -0.0004 | -0.0002 |
>
> Three things the single measurement could not say. The effect **reproduces**:
> at `ef` 64 all five graphs are negative, across two engines that share no
> code, which is not a property of `buildParallel`. It is **an order of
> magnitude smaller at 1M** than the ~2 points reported above -- 0.002, or 0.2
> points -- so whatever the earlier figure was measuring, most of it was not
> this. And it is **a crossover, not a plateau**: positive at `ef` 32 in all
> five, negative at 64 in all five, decaying back toward zero by 512 as both
> recalls approach one.
>
> **Two explanations checked and eliminated, same day.** The first draft of
> this block guessed an ε-tie artefact. The code says otherwise:
> `recall_one` computes `tol = eps.absolute_at(kth)` and `Oracle::judge_at`
> applies it at every depth, so a rank-1 within ε of the true nearest is
> *already* credited at k=1. Ties within tolerance cannot be the gap.
>
> What is real is that the two statistics are not the same shape.
> `judge_at(query, 1, ...)` takes `truth = &neighbours[..1]`, so an engine
> whose rank-1 is the true **#2** scores a complete miss at k=1 and a hit at
> k=10, which credits any of the true top ten. The gap therefore measures
> ordering error *inside* a neighbourhood the engine did find, by
> construction, not by defect. That also fits the crossover: at `ef` 32 whole
> neighbourhoods are still missing, so filling ten slots is the harder job and
> the gap is positive; once the neighbourhood is reliably found, only the
> ordering is left to get wrong.
>
> The obvious follow-on -- that distances concentrate at d=1536, so ordering is
> intrinsically harder there -- was measured against both ground truths and
> **does not hold**. Where the second neighbour sits within the `d_1..d_10`
> span, over 10,000 queries: median 0.289 on dbpedia against 0.264 on SIFT1M,
> with `(d_2 - d_1)` under 1% of the span for 2.04% of dbpedia queries against
> **3.46%** of SIFT1M's. SIFT1M's top ten is the flatter of the two, so
> concentration does not explain why only the headline tier inverts.
>
> So it stays unexplained per §11, with two candidates struck off and the
> arithmetic understood. It is no longer unexplained *and* strawmANN's, which
> is the part that was costing the engine a defect it does not have.

### 9. The recall metric was measuring the wrong thing, and MRDE caught it

The first SIFT1M run reported `recall@10 = 1.0000` at *every* `ef` from 16 up -
on an approximate index, while MRDE moved 9× across the same rows. Two numbers
from the same returned results disagreeing about whether `ef` mattered at all.

Ground truth is cached once at k=100 and reused for every recall depth, and the
membership test used the *whole* cached list: `recall@10` accepted any id in the
true top **100**. A competent index puts its 11th-nearest inside the top 100
essentially always, so the curve was pinned at 1.0.

This would have flattened W10, T3's recall-vs-`ef` comparison, and §6.7's entire
`(ef, oversampling, rescore)` frontier into a constant, **for both engines** -
while looking entirely reasonable. Details in
[`docs/dataset-runs.md`](dataset-runs.md).

### 10. Half of SIFT1M's "ground-truth disagreement" is tie-breaking

Recomputing SIFT1M's ground truth in fp64 and diffing against the shipped
`ivecs`: **999,889 of 1,000,000 neighbours agree**. All 111 disagreements sit at
rank 99 with a score exactly equal to the k-th, i.e. they are boundary ties.

But 5443 of 10000 queries, 54%, return the *same* 100 neighbours in a
different order, which a positional diff counts as 19,519 mismatched positions.
§4.3 attributes published-GT disagreement to fp32 rounding; on SIFT1M that is
structurally impossible, since integer components cap the squared distance at
8.3M against fp32's exact-integer limit of 16.8M. The disagreement is purely
tie-breaking convention. Details in [`docs/ground-truth.md`](ground-truth.md).

### 11. PQ's cost is the lookup table, not the codes

PQ8 at `m=96` measured **3.7× slower than modelled** cold. The ADC table is
`96 × 256 × 4 B = 98 KB` and does not fit in a 48 KB L1. §5.4's working-set
table counts the codes and should also count the table. This is the quantitative
argument for §6.7's PQ4/FastScan "stretch goal": a 4-bit table is 1.5 KB.

---

### 12. The recall/throughput frontier inverts when the load generator changes

Measured with the conformance harness's own client, one connection issuing
batched queries sequentially, Qdrant leads above recall ≈0.96 and the gap widens
with recall: 1.63× at recall 0.9937, 1.92× at 0.998. `comparison-sift1m.md` §3 recorded
that as the project's most uncomfortable result, since a strawman with none of
Qdrant's obligations was losing where production systems operate.

Joining the *same* recall sweep to **bfb's** W10 throughput at the same `ef`
values instead, strawmANN leads at every matched recall, and the gap narrows
rather than reverses:

| recall@10 | strawmANN q/s | Qdrant q/s | ratio |
|--:|--:|--:|--:|
| 0.9742 | 29,254 | 8,024 | 3.65× |
| 0.9815 | 26,470 | 7,069 | 3.74× |
| 0.9934 | 15,313 | 5,758 | 2.66× |
| 0.9938 | 15,069 | 5,578 | 2.70× |
| 0.9975 | 8,221 | 3,969 | 2.07× |

Both are real measurements of the same two engines at the same recall. The
difference is entirely in what generated the load, and §4 of `comparison-sift1m.md` had
already named the mechanism before the second measurement existed: Qdrant
parallelises one query across up to eight segments, which is free when a single
client leaves cores idle and worthless when bfb's concurrency has already filled
them. The earlier number measured that idle capacity.

The finding is not "strawmANN wins after all". It is that **a frontier is a
property of the client as much as of the engines**, and this project published
one without saying which client drew it. §4.1 exists precisely to keep load
generation and relevance measurement in separate tools; the failure here was
using the relevance tool's incidental rate as a throughput axis because it was
the one that happened to know the recall. The join now goes the other way, and
the report states which side each number came from.

The pairing is across runs: throughput from the bfb run, recall from a later
conformance sweep on the same dataset with the same `m` and `ef_construct` but a
separately built graph. The host is ungated, and the differ reached T2, which
§7.4 says does *not* license a comparative claim at all. So this is a
development-grade observation about a measurement, which is exactly what it is
being reported as.

> **The inversion was a defect, not a property of the client. Findings 27.**
> The single-client ordering above was one strawmANN core against eight of
> Qdrant's: §6.3's "one query per worker" was implemented as one *request* per
> worker, so a `QueryBatch` of 32 was searched serially on whichever core
> dequeued it. With `handlers.BatchJob` fanning the batch across workers the
> ordering flips and agrees with bfb's, 3.19x to 3.75x in strawmANN's favour
> over the same `ef` range.
>
> The entry's lesson survives its own example, which is why it is kept. "A
> frontier is a property of the client as much as of the engines" is still
> true, and this project still published one without saying which client drew
> it. What is no longer true is that these two measurements were both
> faithful: one of them was measuring a scheduling bug.

---

### 13. Qdrant's `uint8` storage does not normalise cosine, and clamps rather than refusing

Two things fell out of reading `lib/segment/src/data_types/primitive.rs` and
`spaces/metric_uint/` at `0e3397469` while implementing `VectorParams.datatype`.

**Cosine is not normalised at ingest for `uint8`.** Every other combination
preprocesses the vector so the search path is a plain dot product, which is
§8.3's rule and this engine's architecture. `impl Metric<VectorElementTypeByte>
for CosineMetric` has `fn preprocess(vector) -> vector`, the identity, and
`cosine_similarity_bytes` divides by both norms per comparison instead. It could
hardly do otherwise: normalised components lie in [−1, 1] and every one of them
would truncate to 0 or 1.

So §8.3's rule is a property of the *storage type*, not of the metric. That is
now `Datatype.normalisesAtIngest`, and a `uint8` cosine collection here pays for
the norms per candidate exactly as Qdrant does.

**The conversion saturates rather than validating.** Qdrant's cast is Rust's
`x as u8`, and their own test pins what that means:

```text
[-10.0, 1.0, 2.0, 3.0, 255.0, 300.0] -> [0, 1, 2, 3, 255, 255]
```

A client that sends fp32 data into a `uint8` collection gets a silently mangled
vector, not an error. The tempting move is to be stricter and reject the
out-of-range component, and it is the wrong one: §8.5's T1 tier compares values
against the real server, and "we refuse what they accept" is a difference that
shows up as a conformance failure rather than as the improvement it feels like.
The clamp is matched, and both the unit test and the end-to-end test assert it.

The pattern is the same one `docs/bugs.md` records for "8 to 13 segments": the
behaviour is not what an outsider would guess from the documentation, and the
only reliable way to know it is to read the source. Both cells are now in a
table in `src/dist/datatype.zig` with the file and function they came from.

---

### 14. The two engines do not store the same normalised vector, and it is most of the T1 gap at d=1536

§4c of `comparison-sift1m.md` records T1 failing on the headline tier: max |Δ| **1.967e-6**
against a calibrated ε of **9.537e-7**, with the fp64 oracle saying Qdrant is the
one that drifts. It attributes the divergence to fp32 accumulation depth in the
distance kernel, 48 FMA chains against 12.

That is not the only contributor, and it may not be the largest. Cosine is
normalised at ingest (§8.3), so before any distance is computed the two engines
have already written *different bytes* for the same input vector, for two
reasons found by reading `simple_avx.rs`:

1. **The length is summed differently.** `cosine_preprocess_avx` uses four
   8-lane accumulators reduced as `(a+b) + (c+d)`; `norm.squaredLength` is a
   strict scalar sum, deliberately, to match §8.3's ordering rule.
2. **The division is not a division.** Their AVX path computes
   `1.0 / length.sqrt()` and *multiplies*, carrying a reciprocal's rounding on
   top of the square root's. Ours divides, which is what `norm.zig` argues for
   and what their own **scalar** fallback does.

So the engines agree when Qdrant runs without AVX and disagree when it has it,
which is every host in the comparison.

Measured at d=1536 over 64 random vectors, reproducing their AVX path exactly
(`norm.zig`, `qdrantAvxNormalize`):

```
max |Δcomponent|  6.706e-8
max |Δscore|      1.490e-6
```

**1.490e-6 against an observed T1 divergence of 1.967e-6 and an ε of 9.537e-7.**
The stored-vector difference alone is above the tolerance and is the same order
as the total. §4c's explanation is not wrong, but it is incomplete, and the
missing part was never isolated because nothing had compared the two
normalisations directly.

Two consequences worth stating. Attributing the whole gap to accumulation depth
overstates a claim about the *distance kernel* using evidence that partly
belongs to *ingest*. And "make our normalisation faster" is not a free
optimisation here: vectorising `squaredLength` would change what this engine
stores, so the question is which of the two orders to match, not which is
quicker. That is a conformance decision and it has not been made.

**Resolved, with one correction to the above.** The decision was made in favour
of matching Qdrant's path selection: `norm.zig` now chooses at runtime, from
the same cpuid facts as `is_x86_feature_detected!`, among order-preserving
transcriptions of `cosine_preprocess_avx` (AVX+FMA and d ≥ 32),
`cosine_preprocess_sse` (d ≥ 16) and the scalar `cosine_preprocess`, and the
tests check each against a scalar-indexed by-hand transcription bit for bit
(`docs/bugs.md`, "Cosine normalisation followed Qdrant's scalar path"). The
correction: point 2 is true of Qdrant `dev` after PR #8650 (2026-08), **not of
1.19.0**, the version `comparison-sift1m.md` pins; in 1.19.0 every path divides by
`length.sqrt()`, so the reciprocal is not reproduced. Point 1 was also slightly
off: the earlier replica reduced with `@reduce(.Add)`, whereas
`hsum256_ps_avx` is `((x0+x4)+(x1+x5)) + ((x2+x6)+(x3+x7))`, and it is that
order that is now used. What survives unchanged is the finding itself: on
1.19.0 the sums alone move the branch for 37 of 400 vectors swept across
±4e-6 of unit length at d=1536, and the uniform unit vector at d=384/768/1536
is stored untouched by both SIMD sums and divided through by the scalar one.

---

### 15. The open-loop rows measure bfb's pacing, not the engine

§7.4 draws a hard line: `--parallel` is a closed loop whose p99 understates the
tail, so "use `--rps` for latency claims". The report was built to honour that,
labelling every row with the loop it ran under. Then it showed both the client's
latency and the server's, and the open-loop rows stopped making sense.

| row | offered | server p50 | client p50 | ratio |
|---|--:|--:|--:|--:|
| strawmANN W4-rps500 | 500/s | 0.458 ms | 12.59 ms | 27x |
| strawmANN W4-rps1000 | 1000/s | 0.377 ms | 87.45 ms | 232x |
| strawmANN W4-rps2000 | 2000/s | 0.273 ms | 1023.94 ms | **3,758x** |
| Qdrant W4-rps500 | 500/s | 2.379 ms | 108.27 ms | 46x |
| Qdrant W4-rps2000 | 2000/s | 35.235 ms | 1819.23 ms | 52x |

strawmANN saturates above **27,000 q/s** in W4. At an offered 2,000/s it is at
7% utilisation, and a queue cannot form at 7% utilisation. The server agrees: it
reports 273 µs. The client reports a full second.

So the gap is on the generator's side of the wire. Whether that is bfb's rate
limiter accumulating a backlog, or its timer starting at the *intended* send
time while the sender itself falls behind, the consequence is the same: the
number measures the sender. The closed-loop rows are not affected in the same
way, and W4's 8x client/server gap is the queue `-p 64` explicitly asks for.

**Neither loop currently yields a publishable latency**, which is a stronger
statement than §7.4's and was not visible until both timings were on the page
next to each other. The report flags the affected rows and says so rather than
presenting them as the trustworthy ones.

**The experiment, run.** Same collection, same queries, three offered rates:

| offered | server p50 | client p50 | ratio |
|--:|--:|--:|--:|
| 100/s | 2.1 ms | 3.0 ms | **1.4x** |
| 500/s | 15.9 ms | 636.5 ms | 40.1x |
| 2000/s | 3.3 ms | 93.7 ms | 28.8x |

At 100/s the client tracks the server and the open-loop arm is exactly what
§7.4 wants. By 500/s it is not, and the ratio does not grow smoothly with the
rate: 500/s is worse than 2000/s here, and the *server* p50 at 500/s is five
times the server p50 at 2000/s, which no queueing model explains. That shape
says the generator is emitting bursts rather than pacing: the server sees a
clump, and the client's own backlog does the rest.

So the rule for the gated run is narrow: an open-loop row is faithful only at a
rate where the client's pacing holds, and this harness cannot currently
demonstrate that above ~100/s. Until it can, the *server-side* percentiles are
the meaningful half of those rows, and the harness records both.

> **"Until it can" arrived. Findings 32 and 33.** The generator was not
> pacing badly; `process_with_rps` matched `Some(Err(err))` in a `select!`, so
> every *successful* completion disabled the reaping branch and the backlog
> drained one request per rate-limiter tick. That is why the reported latency
> was worst at the *lowest* offered rate, which no queueing model explains and
> which this entry could only describe. Binding the result instead of
> pattern-matching it is the whole fix; it is upstream as qdrant/bfb#172.
>
> So "neither loop currently yields a publishable latency" is spent. The
> open-loop client now sits 0.12 to 0.37 ms above the server across an eightfold
> change in load, against the closed-loop control's 80 µs, and the client p50
> rises with offered load the way a queue does. The server-side halves quoted
> above were also affected, not merely the client's: a loop starved of reaping
> delivers in bursts, so those rows were *loading* the engine differently from
> what the arm claimed to offer.

---

## 16. Time-to-Green has a three-second floor, and W2 was sitting on it

W2's headline is the index build time, and the harness reported it as
`int(wall_clock)` around the whole `bfb` invocation: one-second resolution on a
row whose entire result is a duration.

Underneath that, the number itself has a floor. `bfb`'s `wait_index`
(`collection/mod.rs`) sleeps one second *before* its first poll, polls once a
second, and breaks only after **three consecutive** Green replies. Nothing it
reports can be below three seconds, and a build that finished instantly is
indistinguishable from one that took 2.9 s.

Measured on a 20k build against strawmANN:

| | |
|---|--:|
| harness wall clock (what W2 used to report) | 5.18 s |
| bfb upload phase | 0.038 s |
| bfb Time-to-Green | **3.008 s** |

Three of those five seconds are the polling loop, and the build itself is below
this instrument's resolution. The bias is a *constant added to both engines*,
so the difference between them survives it and the ratio does not, and it is
the faster engine whose ratio is understated. At a plausible 4 s against 40 s,
the reported figures would be 7 s against 43 s and the ratio would read 6.1x
instead of 10x.

The rows now carry `wall_s`, `upload_s` and `index_wait_s` separately, and a
`time_to_green_floored` flag; the report marks a floored cell and says what the
mark means. Removing the floor needs a change in the load generator, not here.

---

## 17. W13 published a qps with no latency, from reading the wrong key

The harness pulled percentiles from `results.search`, and a scroll row writes
them under `results.scroll`. So W13 arrived with a five-figure qps and an empty
latency object, which reads as an instrument that failed rather than a lookup
in the wrong place. The timings were in the file the whole time.

Fixed, and the recovered numbers say something about the row: client p50 74 µs
against a server p50 of 0.5 µs. An id-ordered scroll of ten points really is
nearly free, so W13 is measuring the transport almost entirely, in the same way
W0 does deliberately. Worth reading as a plumbing row rather than as a
pagination result.

## 18. Qdrant cannot pin dense vectors, which is what strawmANN has always done

Reading Qdrant's placement code to implement `VectorParams.memory` turned up
two things that change how the comparison should be read.

**`Pinned` is not available for dense vector storage.** The proto says so
outright: "`Pinned` is not supported for dense vector storage"
(`collections.proto`, `VectorParams.memory`). Qdrant's dense vectors are either
`Cold` or `Cached`: an mmap of a file, evictable, at 4 KiB pages. strawmANN's
only historical mode (anonymous memory, 2 MiB pages, never evicted) is the
one placement Qdrant does not offer. That is not a gap on either side, it is
the §5.5 TLB argument showing up as a configuration difference, and it means
any headline ratio taken at default settings includes it.

**Qdrant's default for dense vectors is `Cached`, not in-RAM.**
`Memory::resolve(...).unwrap_or(Memory::Cached)` in `lib/collection/src/config.rs`.
A collection created with no placement field is served from a populated mmap.
strawmANN defaults to `Pinned`, so an unconfigured `bfb` run has been comparing
two different residencies all along. The default is kept, changing it would
re-measure every row already recorded, and `--default-placement` now exists so
a run can put both engines in the same one.

Also worth recording, because it is the natural misreading: the deprecated
`on_disk: false` means `Cached`, not `Pinned` (`Memory::from_on_disk`). An
engine that read it as "in RAM" would claim parity with a Qdrant configuration
that does not exist.

---

## 19. Two ways a "cold" measurement is silently warm

Both found with `mincore(2)` while testing the placement work, and both would
have produced a cold-cache number taken on a warm cache.

**Ingest warms the collection it just wrote.** Uploads go through the page
cache, so a `Cold` collection is fully resident by the time the search phase
starts. Measured on an 8 MiB arena: 2048/2048 pages resident under `Cold`
immediately after writing, and 0/2048 after `fsync` + `fadvise(DONTNEED)`. Any
workload that uploads and then searches without dropping the cache in between
is measuring `Cached` and calling it `Cold`.

**tmpfs cannot be cold at all.** On tmpfs the page cache *is* the storage:
`fadvise(DONTNEED)` returns success, evicts nothing, and every page stays
resident. Measured: 2048/2048 pages still resident after a successful drop on
tmpfs against 0/2048 on ext4 for the identical code. Since `/tmp` is tmpfs on
this host, a `--data-dir` there makes the cold arm bit-for-bit the cached arm
with nothing to indicate it.

The residency test in `storage.zig` asserts against the kernel rather than
against the flags passed to `mmap`, and skips rather than fails when the
filesystem cannot demonstrate eviction. A failure there would read as "cold is
broken" when what is true is "this directory cannot be cold".

---

## 20. The development host is two different CPUs, and `--pin` straddled both

Preparing the isolated-core boot for the gated run, the topology turned out to
be heterogeneous:

| cores | µarch | max clock | L3 | shared by |
|---|---|--:|--:|---|
| 0 to 3 | Zen 5 | 5.16 GHz | 16 MiB | 0 to 3 |
| 4 to 11 | Zen 5c | 3.29 GHz | 8 MiB | 4 to 11 |

Two clock domains 57% apart and two L3s of different size, in one package
(AMD Ryzen AI 9 HX PRO 370, 12 cores + SMT).

`--pin` assigned CPUs `0..io_threads+workers`, so `--pin --io-threads 1
--workers 8` put the I/O thread and workers on 0 to 8: **three workers on the fast
CCX with 16 MiB of L3, five on the slow one with 8 MiB**. Which worker picked
up a query decided its clock and its cache, so the latency distribution had a
shape the engine did not put there. Runs without `--pin` are worse, not better.
Then the scheduler chooses, and it may choose differently between the two
arms of a comparison.

It also breaks the thing it was needed for: isolating cores 4 to 11 on the kernel
cmdline and then pinning to 0 to 8 puts most of the server on cores that were not
isolated.

`--cpus <list>` now names the pool, in the same spelling `isolcpus` and
`taskset -c` use, and the startup line prints the assignment it made
(`pinning: io { 4 } workers { 5, 6, ... }`) rather than the bare `true` it used
to print. Verified against `/proc/<pid>/task/*/status`.

This also puts a caveat on §5.4 and §5.5's cache arguments: "the LLC boundary"
is not one number on this host. It is 16 MiB on four cores and 8 MiB on eight,
so the discontinuity a quantized working set is supposed to show as it crosses
L3 sits in a different place depending on where the thread ran. The
kernel-matrix tables' "L3-resident" rows inherit the same ambiguity.

---

## 21. Why strawmANN's recall looked worse than Qdrant's. Three reasons, one of them ours

The first end-to-end differ run reported strawmANN recall@10 = 0.9941 against
Qdrant 1.0000, confidence intervals disjoint, so T3 refused to license a
comparative claim. Qdrant scoring *exactly* 1.0000 on 10,000 queries is not
what HNSW does, and pulling that thread found three separate things.

**Qdrant was not using HNSW.** `indexed_vectors_count: 0`. Qdrant split 50,000
points across 8 segments (3,125 KB each) and both of its thresholds default
to 10,000 KB, so no segment was large enough to index. Every query was a
brute-force scan. The differ was comparing approximate search against exact
search and calling the difference a recall gap.

**There are two thresholds, and clearing one is not enough.** Lowering
`optimizers_config.indexing_threshold` built the index (`indexed_vectors_count:
50000`) and changed nothing: recall stayed 1.0000 at every `ef` and throughput
*rose* with `ef`, which is backwards. `hnsw_config.full_scan_threshold` decides
whether the index is **used**, separately from whether it is **built**. Only
after lowering that did Qdrant produce an HNSW curve at all.

**Segment count inflates recall at fixed `ef`.** Each segment searches its own
graph at the full `ef` and the results are merged, so N segments explore more
than one. Same data, same `ef=128`: 8 segments gave 1.0000, 2 gave 0.9978. §2
already lists this as a semantic gap requiring `--segments 1`; this is what it
is worth.

### The residual gap is ours, and it is a plateau

With both engines indexed and Qdrant down to 2 segments, on the same 50k slice,
`m=16`, `ef_construct=100`, 2,000 held-out queries:

| ef | strawmANN recall@10 | Qdrant recall@10 | strawmANN recall@1 |
|--:|--:|--:|--:|
| 32 | 0.9531 | 0.9614 | 0.9715 |
| 64 | 0.9866 | 0.9906 | 0.9930 |
| 128 | 0.9941 | 0.9978 | **0.9950** |
| 256 | 0.9954 | 0.9996 | **0.9950** |
| 512 | 0.9957 | 0.9997 | **0.9950** |

strawmANN's `recall@1` is *identical to four decimal places* from `ef=128`
upward while Qdrant keeps climbing to 0.9997. Ten queries in 2,000 never find
their true nearest neighbour no matter how wide the beam.

That shape is diagnostic. More `ef` means more of the graph explored, so a
deficit that does not close with `ef` is not a search-effort problem. And it is
not the documented `ef` clamp: `Workspace.max_ef` is 4096, so 512 is passed
through unmodified.

**Exact search on the same collection returns recall@1 = 1.0000 and
recall@10 = 1.0000.** The vectors are all present, stored correctly, and scored
correctly, brute force finds every one of them. So whatever the graph is
failing to return, it is not a storage or arithmetic problem.

I read that as a reachability defect. **It was not**, on this corpus: building
the same 50k SIFT slice and walking the graph gives 50,000/50,000 nodes
reachable and zero with in-degree zero. Finding 22 has what it actually was,
and the three defects the search for it turned up. One of which *is* a
reachability defect, on a corpus that is not SIFT.

---

## 22. Localising the recall plateau: three defects, and the one that mattered

Finding 21 left strawmANN's recall@1 pinned at exactly 0.9950 for `ef` = 128,
256 and 512 while Qdrant climbed to 0.9997. A deficit that does not close as the
beam widens is not a search-effort problem, so the graph was the suspect.

### Building the diagnostic changed the question

The existing connectivity test asks whether every node has a level-0 out-edge.
That is the wrong question: a node with 32 out-edges and no *in*-edges is
invisible to search at any `ef`. Walking the graph the way search does, a
directed BFS at level 0 from every node the descent can land on, found, on
random d=128 vectors:

| n | unreachable | in-degree 0 |
|--:|--:|--:|
| 5,000 | 18 | 20 |
| 20,000 | 327 | 352 |
| 50,000 | **1,759 (3.5%)** | 1,848 |

Growing with n, and every unreachable node had out-edges: they had been pruned
out of everyone else's list. Two causes, both real:

**The heuristic discarded candidates into empty slots.** Algorithm 4's
`keepPrunedConnections` was not implemented. Mean out-degree was 17.66 of a
possible 32, the diversity rule fired, and the freed slots stayed empty rather
than being topped up from the pruned candidates. Those candidates are precisely
the nodes whose only route into the graph may have been that edge.

**Level 0 selected `m0` neighbours where the reference selects `m`.** hnswlib's
`mutuallyConnectNewElement` calls the heuristic with `M_` for the new element
and uses `Mmax0 = 2M` only as the eviction cap when pruning a *neighbour's*
list. Using `m0` for both doubled the back-links every insertion forces, and
every back-link into a full list evicts somebody.

Together: 1,759 → 217 unreachable at 50k, an eightfold reduction.

### And then SIFT said none of that was the answer

Built on the actual 50k SIFT slice, before *and* after those fixes, the graph is
**100% reachable with zero in-degree-zero nodes** (measured with the same BFS,
run as a one-off against the dataset cache rather than kept in the suite). Random normal vectors in 128
dimensions are nearly equidistant, which is the worst case for a diversity
heuristic; clustered real data is not. The orphaning is real and worth fixing.
It is what a poorly-conditioned corpus would hit, but it was never what the
SIFT plateau was made of.

The clue that pointed somewhere else: `ef_construct` of 100, 200 and 400 gave
*identical* recall, to four decimals, at both `ef`=128 and `ef`=512. Build
effort was buying nothing.

### The frontier was throwing away good candidates

`heap.Frontier.push` dropped the incoming candidate whenever the heap was full:

> Dropping is safe: the frontier is sized to `ef` and a dropped candidate is by
> construction worse than `ef` others, so it could not have entered the result
> set.

That is false of a heap that is merely *full*. Arrival order says nothing about
rank, and `searchLayer` only pushes candidates it has already checked are better
than the current worst result, so every dropped push was a promising candidate
discarded for arriving late.

Instrumented, building the 50k SIFT graph at `ef_construct = 100` dropped
**178,027 of 11,021,631 pushes, 1.6%**. And because the frontier and the result
heap both scale with `ef_construct`, the drop *rate* is invariant to it. Which
is exactly why recall was flat across 100, 200 and 400. The graph could not
improve, because the same fraction of good candidates was discarded at every
setting.

Fixed by evicting the heap's *worst* element instead, and only when the newcomer
beats it. The scan covers the leaf half, where a max-heap's minimum must be, so
there is still no allocation on the query path (§6.3).

### What it was worth

50k SIFT, `m=16`, `ef_construct=100`, 2,000 held-out queries:

| ef | recall@10 before | after | recall@1 before | after |
|--:|--:|--:|--:|--:|
| 32 | 0.9531 | **0.9682** | 0.9715 | 0.9790 |
| 64 | 0.9866 | **0.9899** | 0.9930 | 0.9925 |
| 128 | 0.9941 | **0.9952** | 0.9950 | 0.9950 |
| 256 | 0.9954 | **0.9960** | 0.9950 | 0.9955 |
| 512 | 0.9957 | **0.9968** | 0.9950 | 0.9960 |

The plateau is gone: `recall@1` now rises with `ef` (0.9950 → 0.9955 → 0.9960)
instead of repeating 0.9950 three times. The largest gain is at low `ef`, where
throughput is highest, `ef=32` gained 1.5 points of recall@10 at ~17k q/s.

A gap to Qdrant remains at high `ef` (0.9968 against 0.9997), and part of it is
still the segment count: Qdrant was measured at 2 segments, each searching its
own graph at the full `ef`, which explores more than one graph does. Closing
that comparison properly needs a genuine single-segment Qdrant, which its
optimizer would not produce on a collection this small.

## 23. At 1M the recall plateau is back, and it is not a search-width limit

§22 localised the plateau at 50k and closed most of it: `recall@1` started
rising with `ef` again instead of repeating 0.9950. On the full SIFT1M
collection, measured 2026-08-18 on the gated host, it returns, and this time
the shape rules out the obvious explanation.

| ef | strawmANN recall@10 | Qdrant recall@10 |
|--:|--:|--:|
| 512 | 0.9949 | 0.9988 |
| 1024 | 0.9954 | not measured |
| 2048 | **0.9955** | not measured |
| 4096 | **0.9955** | not measured |

> **Still holding.** The latest gated sift1m run measures recall@10 **0.9995**
> at `ef` 512, against the 0.9994 the fix recorded below, so the repair has
> held across every run since. `ef` 2048 and 4096 have not been re-measured,
> so the *shape* of the table below is untested rather than refuted.

Eight times the search width buys 0.0006, and the last doubling buys *exactly
nothing*: 2048 and 4096 return recall equal to sixteen decimal places, which is
what it looks like when a wider frontier finds no candidate it had not already
found. The ceiling is not the clamp, `Workspace.max_ef` is 4096 and every value
above was inside it, and it is not oversampling, because this is the fp32
collection with no quantization in the path.

So roughly **0.45% of true nearest neighbours are unreachable through this
graph at any width**, which makes it a construction property rather than a
search parameter: an entry point and a set of edges that cannot get there from
here. §22's three defects were all in the search; this one is not.

What it costs is more than a ratio. Qdrant reaches 0.9988 at `ef` 512 and
strawmANN cannot reach it at 4096, so at the high-recall operating points
production systems actually use, the comparison is not "slower". It is "cannot
serve that point at all". Every matched-recall claim this project makes is
therefore bounded above by 0.9955, and the `1.64x at ef=512` in the generated
table is measured at *unequal* recall (0.9940 against 0.9988) and is not a win.

**Resolved, later the same day.** The construction property was the parallel
build orphaning nodes: `insertOne` memset its own list under lock, discarding
back-links a concurrent `linkBack` had already placed there, and 43 of 20,000
nodes had no in-edge on level 0 (`docs/bugs.md`; the reachability test that
now guards it). With every level's list written before the first back-link,
the 2026-08-18 gated sweep reaches **0.9994** at `ef` 512 against Qdrant's
0.9987, the two recall@10 CIs overlap along the whole sweep, and the
matched-recall comparison is no longer bounded at 0.9955. The table above is
kept as the record of what the ceiling looked like from the search side.

## 24. Binary quantization is not a usable configuration on SIFT1M, for either engine

W7 reports recall@10 of 0.026 for strawmANN and 0.027 for Qdrant, which reads
as a defect and is not one. §6 above ("Oversampling without rescore does
nothing") measured the same curve at d=64 and found 0.410 at 4x oversampling and
1.000 at 64x. At d=128 on 1M points, with rescore on throughout:

| oversampling | recall@10 |
|--:|--:|
| 4x | 0.0254 |
| 16x | 0.0510 |
| 64x | 0.1025 |

Recall doubles for every 4x of oversampling, from a base so low that the
doubling does not help: reaching 0.8 from here needs roughly 4096x, which is
not a search, it is a scan with extra steps. One bit per dimension leaves 128
bits per SIFT vector, and the Hamming ordering they induce is close to no signal
at all for this distribution. Both engines agree to within 0.001, which is the
evidence that this is the encoding's property and not either implementation's.

W7 therefore measures how fast two engines can return the wrong answer. It is
kept as a characterisation row and must not be read as a search-quality
comparison on this dataset; binary quantization belongs on the d=1536 tier
(§4.2's dbpedia-openai-1m), where the bit budget is an order of magnitude
larger and the technique is designed to work.

## 25. W11's loss is one line of state machine, and it costs 175x for ~200 seconds

W11 reads 205 q/s against Qdrant's 1,042, the one row where the strawman loses
badly, and the note on it says 99% of the row searched a quiet collection. So
the loss is not write contention. Measured directly on 2026-08-18, appending
W11's 200k at `--offset 1000000` onto a built 1M `bench2` and then sampling the
search rate:

| phase | rate |
|---|--:|
| quiet 1M collection, graph published | **13,126 q/s** |
| after the append, first 8k queries | 75 q/s |
| next 8k queries | 82 q/s |
| from ~200 s onward | **15,043 q/s** |

A **175x collapse**, lasting about 200 seconds, then full recovery, faster than
the baseline, because the graph now covers 1.2M. W11 sends 50,000 queries and
takes 244 s at 205 q/s, so the row lies entirely inside the collapse. It is not
measuring search-during-write. It is measuring the rebuild window.

The mechanism is `invalidateIndex`: on `.ready`, if `needsRebuild` is true it
stores `.absent`, and `choosePath` maps anything that is not `.ready` to
`.brute`. So the graph is discarded for query purposes the moment the pending
tail crosses `rebuild_ratio` (0.10), and W11 appends 200k onto 1M = 0.20. The
comment directly above `invalidateIndex` argues the opposite case correctly:
"a graph built over the first N points stays *correct* for those N points when
point N+1 arrives; it is merely incomplete", and that reasoning is what the
`.ready` path already implements with its tail scan. It simply stops applying at
exactly the point it would matter most.

`choosePath` carried a `TODO(rebuild)` naming this and deferring it as a
state-machine change (`.absent`-with-graph versus without, and retiring a
graph that is still being served; the separate `build_stale` flag that once
downgraded a finished build to `.absent` under writes is gone, see
`bugs.md`). That judgement is not disputed here. The retire/reclaim hazard is
the same class as the one a three-thread stress test caught 3/3 in an earlier
pass, and it should not ride along in a measurement commit. What this finding
adds is the price: serving the old graph plus a 200k tail scan instead of
brute-forcing 1.2M is worth roughly 6x on this row, and the difference between
"the strawman loses one workload by 5x" and "the strawman is briefly a linear
scanner" is the difference between an architectural result and a scheduling
artefact. Finding 31 is where the deferral was paid off: the state-machine
change turned out to be a small one.

## 26. Publishing a container port taxes one arm of the comparison

Qdrant ran under `docker run -p 6334:6334`, which is the ordinary way to reach a
container and the way this harness had always started it. It also puts Docker's
**userland proxy** in the request path. `docker-proxy` is a host process that
relays every byte between the load generator and the engine in userspace, and
strawmANN, a pinned native process that `bfb` connects to directly, pays
nothing equivalent. The comparison was charging one engine for its transport and
the other for transport plus a relay.

Two things make it worse than a constant offset:

| | |
|---|---|
| **It is not pinned.** | All four `docker-proxy` processes report an affinity of `0-11`: neither the server set `4-11` nor the client set `0-3`. The busy one was observed on core 4: *inside* the eight cores Qdrant had been given by `--cpuset-cpus`, competing with the engine it was serving. |
| **Its CPU is charged to neither engine.** | It is outside the container, so the cgroup accounting `procstat.py` reads does not see it, and it is not the harness, so the client-side accounting does not either. |

Measured rather than assumed: `5.13 s` of CPU in twelve minutes, and it is not
spread evenly. Sampled over 20 s during W2's index build, with the container at
`680%`, the proxy read `0.0%`. The cost lands **entirely in the rows that carry
traffic**, which are exactly the rows whose latency percentiles the report
compares, and it lands per request, where a userspace hop and its context switch
sit directly in the p99 path.

The fix is `--network host`, so the container binds Qdrant's own ports and both
engines have the same path to `bfb`. `start_qdrant` now does that, and refuses
rather than times out if asked for ports the image does not bind.

The reason this is worth a finding and not a footnote is the scale it has to be
judged against. The gated noise floor puts nine of fourteen rows at the 2%
policy floor, so an artefact worth ~2% of the server's compute is no longer
below the resolution of the instrument. It is right at the threshold where the
report starts calling differences real. An overhead that was invisible against a
13-79% floor is a confound against a 2% one. **Tightening the measurement
changes which artefacts matter**, and every result taken with a published port
is a result whose Qdrant arm paid a tax that was never named.

## 27. A batched request was the one place Qdrant was faster, and it was one core doing the work

The 2026-08-18 gated run's headline paragraph reports two orderings from the
same recall sweep. Read against bfb, one query per request, strawmANN serves
1.77x-2.13x Qdrant's rate at matched recall. Read against the conformance
binary's own smoke rate, one client sending `QueryBatch` of 32 and waiting
for each, the ordering flips: Qdrant faster at every `ef`, 1.07x at 32 up to
1.49x at 512. W5 (bfb `--search-batch-size 16`) showed the same pressure from
the other side, 1.33x against W4's 1.52x.

The mechanism was the worker model. §6.3's "one query per worker" was
implemented as one *request* per worker: a `QueryBatch` was decoded and its
queries searched one after another on the core that dequeued it, while the
other six workers idled unless another connection was busy. Qdrant fans a
batch out across its search runtime. So a single batching client measured
one strawmANN core against eight Qdrant cores.

`handlers.BatchJob` (2026-08-19): the dispatching worker decodes the batch
once, submits queries 1..n-1 back onto the same worker queue as
sub-requests, runs the first itself and returns a detached completion; each
sub-request searches on whichever worker picks it up, into that worker's
own scratch, and copies its ranked candidates into the job; the worker
that finishes the last query encodes the response in wire order and
completes the stream. The job and the candidates live in the spare bytes
of the stream's request buffer after the body, which is the only room §6.3's
no-allocation rule leaves for state that outlives the dispatching call; a
batch that would not fit there runs sequentially as before. Answers are
byte-identical to the sequential path (the e2e test compares 64 single
queries with the same 64 as one batch), the first refusal fails the whole
batch as Qdrant's does, and `time` still runs from arrival to the last
query's end.

Measured on the same host state, server on cpus 4-11, client on 0-3, the
same 2,000 queries at each `ef` (single client, `QueryBatch` of 32; not a
bfb row, so not a table figure):

| ef | before (q/s) | after (q/s) | after / before | Qdrant (q/s) | after / Qdrant | recall@10 |
|--:|--:|--:|--:|--:|--:|--:|
| 32 | 9,991 | 33,960 | 3.40x | 10,651 | 3.19x | 0.8974 |
| 64 | 6,046 | 27,054 | 4.47x | 7,213 | 3.75x | 0.9596 |
| 128 | 3,337 | 15,483 | 4.64x | 4,368 | 3.54x | 0.9869 |
| 256 | 1,847 | 9,205 | 4.98x | 2,607 | 3.53x | 0.9964 |
| 512 | 1,017 | 5,104 | 5.02x | 1,516 | 3.37x | 0.9994 |

Seven workers, so the ceiling is 7x; 5x at the widths where a query is long
enough to amortise the queue round trip, 3.4x at `ef` 32 where it is not.
The single-client ordering now agrees with bfb's, and W5's ratio should
close on W4's at the next gated run, which is what the row exists to show.
What did not change: W3 and W4 are one query per request and take no part
of this path.

## 28. W0 is not a transport floor: at d=4 the server is still three quarters HNSW

W0 (`-d 4`, 1M random points, `-p 1`) was named the "transport plumbing
floor" on the reasoning that four dimensions leave nothing but RPC overhead.
Profiled during the row (server pinned to cpus 4-11, client on 0-3, `perf
record -F 2000 -g` on the server, 2026-08-19): 76% of the server's samples
were userspace, and of those `searchFiltered` self time was 35%,
`Frontier.pop` 11%, `TopK.push` 7%, the d=4 Euclid kernel 10%,
`Probe.scoreRow` 3%; the kernel (epoll, read, write, futex) was 24%; HPACK,
protobuf and the handler together under 3%. bfb's own numbers agree:
`Median server time` 172 µs of a 263 µs `Median request time`. Four
dimensions take the distance out and leave the traversal - a random walk over
128 MB of neighbour lists and 4 MB of visited stamps, one cache miss per
neighbour - which is the part every other search row also pays.

Two consequences. The row is still useful, as the floor of *graph traversal*
rather than of transport, and both engines walk the same graph, which is why
it sits at 0.94x rather than at either engine's RPC cost. And it is where a
traversal optimisation shows first. The neighbour loop now issues a prefetch
pass before the scoring pass - the visited stamp for write and the first line
of each neighbour's row - as hnswlib's does. Same host state, same server,
minutes apart, three repetitions each: server-side median at d=4 fell from
112 µs to 95 µs (bfb `Median server time`, three of three); at d=128 on
SIFT1M W3 (453 vs 452 µs) and W4 (15.3k vs 15.5k q/s) were within the run
to run spread, so the distance kernel already hides the misses there. Not a
table figure: those measurements were taken with a desktop session live on
the box, which is why they are quoted against each other and not against the
gated run's numbers (whose W4 was 21.3k under the same binary).

## 29. SQ8 stage 1 was dequantising every code, and the row barely noticed

W6 (SIFT1M, SQ8, ef 128, rescore on) read 0.81x in the 2026-08-18 run: 3,871
q/s to Qdrant's 4,804, with a server-side p50 of 434 µs against a fp32 W3 of
476 µs. Scalar quantization bought strawmANN 9% of latency; it bought Qdrant
34% (328 µs against 499 µs). Profiled with the run's queries (`bfb search
--file`, the SIFT query file; random queries give a different row, see
below): 46% of the server's CPU was `quantized.Query.score`, and its inner
loop was the asymmetric fp32×u8 Euclid kernel, `vpmovzxbd, vcvtdq2ps,
vfmadd (lo + α·c), vsubps, vfmadd` per 16 components. Qdrant quantises the
query too and scores `u8 × u8` with precomputed per-vector offsets.

`SymmetricQuery` (2026-08-19): the query is quantised once per request with
the same bounds, the store keeps `Σcode` and `Σcode²` beside every row, and
stage 1 is `vpdpbusd` for dot and Euclid (by expansion, `α²·(Σa² − 2Σab +
Σb²)`) and `ManhattanU8` for L1: one instruction per 64 components instead
of five per 16. The asymmetric variant stays measured, as §6.7 asks; it is
no longer served.

Measured, same host, real queries, back to back:

| | asymmetric | symmetric |
|---|--:|--:|
| W6 server-side p50 | 357 µs | 342 µs |
| W6 q/s (bfb default `-p`) | 4,281 / 4,341 | 4,425 / 4,495 |
| `Query.score` + kernel, share of server CPU | 46% | 26% |
| recall@10 at ef 32 / 128 / 512 | 0.8842 / 0.9665 / 0.9780 | 0.8820 / 0.9658 / 0.9769 |
| single-client rate at ef 32 / 128 / 512 | 38.0k / 16.7k / 5.4k | 43.0k / 17.1k / 5.5k |

The kernel's share halved and the row moved 4%. What is left is the graph
walk: `searchFiltered` 35%, the two heaps 22%, and both engines pay it, which
is why Qdrant's SQ8 row (328 µs) and this one (342 µs) now sit close. The
recall curve did not move, so the fp32 query was not what the finer quantizer
was buying; the 256 levels and the order-statistic bounds are (still not
Qdrant's, `src/quant/scalar.zig`).

Two things learnt on the way. The row is very sensitive to what the queries
are: bfb's default random queries give a SQ8 server-side p50 of ~100 µs and a
fp32 one of ~430 µs on the same graphs. Uniform noise in SIFT's space lands
far from every point, and the quantised traversal evidently terminates much
earlier on it than the exact one does; why the two paths diverge on noise is
not investigated here. The harness's `--file` config is what makes the row
mean something, and an ad-hoc measurement without it does not. And the desktop session on this laptop moves W4 by 30% between
minutes; only back-to-back alternation is worth reading while it is live.

## 30. PQ training ran one subspace at a time, and the W8 build paid 60 s for it

W8-upload (SIFT1M, `product-x16`) took 107 s to Green on 2026-08-18 against
Qdrant's 88 s, on an engine whose fp32 build (W2) takes 38 s to Qdrant's
90 s. The difference was the codebook: `pq.train` ran k-means over the
`m = 8` subspaces one after another on the build thread, 65,536 samples ×
256 centroids × 25 iterations each, and then encoded the 1M rows one after
another on the same thread, while the eight build threads that had just
finished the graph sat idle.

Both are embarrassingly parallel and both are deterministic per unit -
each subspace's k-means is its own sequential computation with its own
scratch, each row's codes are a function of that row - so running them on
`build_threads` (`pq.train`'s `Training.threads`, `quantized.encodeAllProduct`, one
subspace per task and rows in contiguous ranges) changes the wall clock and
nothing else, which a test pins: serial and five-thread builds of the same
source give bit-identical codebooks and codes (§8.7).

Measured on the same host, back to back, same source, `--quantization
product-x16`:

| | before | after | Qdrant 1.19 (2026-08-18 run) |
|---|--:|--:|--:|
| W8-upload, time to Green | 107.3 s | 48.1 s | 88.2 s |
| W8 server-side p50 | 556 µs | 557 µs | 554 µs |
| recall@10 at ef 128 | 0.6982 | 0.6964 | 0.6858 |

The 0.002 of recall is the parallel HNSW build's own run-to-run spread
(0.6949 and 0.6954 on two earlier builds of the same collection); the PQ
codes did not change. The row the search path measures did not move, which
is the point: this was build time only.

## 31. Serving the old graph through a rebuild is better per query, and W11 still wants incremental insertion

Finding 25 named the mechanism: once an appended tail crossed
`rebuild_ratio`, `invalidateIndex` left `.ready`, `choosePath` mapped every
other state to brute force, and W11 spent its row scanning 1.2M rows per
query while the rebuild ran. The `TODO(rebuild)` deferred serving the old
graph as a state-machine change. It turned out to be a small one: the graph
pointer is published with a release store and loaded once per query under
the `SearchGuard` (`publishedGraph`), `choosePath` traverses whatever graph
is published in any state, and the rebuild's publish retires the old graph
rather than freeing it - the same retire/reclaim a `.ready` reader always
relied on. A test holds a rebuild open with `build_hook` while two threads
search, asserts `.absent`-with-graph and `.building` both answer `.graph`
and find tail points, lets the publish land under the readers, and checks
the old graph was reclaimed afterwards (2026-08-19).

Measured as the harness runs W11 (1M SIFT, 200k appended at `-p 8` while
50,000 queries run at `-p 8`, server on cpus 4-11), old binary against new,
back to back:

| | brute force through the rebuild | old graph + tail scan through it |
|---|--:|--:|
| row wall, 50,000 queries | 178 s | 110 s |
| request p99 | 191 ms | 34 ms |
| request median | 5.5 ms | 17 ms |
| append rate | 60k points/s | 256k points/s |

Better, and not good. A query during the rebuild now streams the 200k-row
tail (100 MB) instead of the whole arena (614 MB), so the slowest queries
are 5x cheaper, and the append is faster because the workers are no longer
saturating the memory bus against it. But the median went *up*: the eight
build threads share the eight cores with seven busy workers, the rebuild
does not finish inside the row, and every query pays the tail scan; the
old binary's 80% of queries at 4-8 ms were the ones served *after* its
rebuild had published, bought with 20% at 80-130 ms. Read together, the
row's ceiling is the tail scan for as long as a rebuild takes under load,
and a 200k tail at d=128 is ~3 ms of memory traffic per query however it
is served.

What W11 measures is therefore not fixable by how the rebuild window is
served. It is fixable by not having one: inserting appended points into a
graph sized to the collection's capacity as they arrive - what Qdrant's
"indexing during ingest" is, and what the parallel builder's locked
`insertOne` already does for a graph that is not yet published. Making the
live graph accept concurrent inserts under readers (the stale-but-valid,
never-empty list invariant already holds; entry-point promotion is already
under a lock) is a design change to the index, and it is the next one.


## 32. The open-loop rows measure bfb's rate limiter, at every rate, and no rate fixes it

`--rps` is the instrument §7.4 reserves for a latency claim: "use `--rps` for
latency claims and closed-loop for saturation throughput. Never quote a
closed-loop p99 as a latency result." So the open-loop arms are the only rows
on the page that may sit next to a latency number, and the report has always
said they were not usable, first because they offered a constant 500/s
against an engine serving 22,000 (finding: a queue cannot form at 2% of
capacity), and after that was fixed to §4's fractions of measured saturation,
because 692 ms of a 700 ms client p50 was not server time.

The obvious next move was to pick a better rate: measure where the generator
stops keeping up and set the arms below it. That is what this sweep was for,
and it says the premise is wrong.

strawmANN, 1M SIFT, ef=128, server on 4-11, bfb on 0-3, one 50,000-query row
per arm:

| arm | offered | delivered | % | client p50 | server p50 | gap |
|---|--:|--:|--:|--:|--:|--:|
| closed `-p 64` | - | 22,685 | - | 2.81 ms | 2.72 ms | **0.08 ms** |
| `--rps 2000` | 2,000 | 1,971 | 99% | 1,411 ms | 7.58 ms | **1,404 ms** |
| `--rps 4000` | 4,000 | 3,865 | 97% | 612 ms | 8.09 ms | 604 ms |
| `--rps 6000` | 6,000 | 5,595 | 93% | 702 ms | 8.15 ms | 694 ms |
| `--rps 8000` | 8,000 | 7,469 | 93% | 651 ms | 8.03 ms | 643 ms |
| `--rps 12000` | 12,000 | 10,056 | 84% | 400 ms | 8.31 ms | 391 ms |
| `--rps 16000` | 16,000 | 12,730 | 80% | 382 ms | 8.22 ms | 374 ms |
| `--rps 20000` | 20,000 | 15,382 | 77% | 348 ms | 8.24 ms | **340 ms** |

Two readings kill the "choose a better rate" theory.

**The gap is worst at the lowest rate.** At 2,000 q/s, nine percent of what
this engine serves in a closed loop on the same cores, the client reports
1.4 *seconds* against the server's 7.6 ms, on a row that delivered 99% of its
offer. There is nothing for a queue to form out of.

**And the gap shrinks as load rises**, monotonically, 1,404 ms down to 340 ms
across a tenfold increase in offered rate. Queueing does the opposite. A
number that falls when you add load is not measuring the server, and it is
not measuring contention; it is measuring how long each request waited
between the slot the rate limiter planned for it and the moment it was
actually sent. Delivery degrading in the same direction (99% to 77%) is the
same fact from the other side: the generator is behind its own plan, and it
charges the wait to the request.

The closed-loop row is the control and it behaves: client 2.81 ms against
server 2.72 ms, a gap of 80 µs, at eleven times the throughput of the worst
open-loop arm.

Two consequences, and they are different in kind.

**Delivery has an honest ceiling and the arms should respect it.** Above
~4,000/s this generator does not send what it was asked to, so an arm at
7,490 reports a rate nobody offered. §4's fractions must be capped by what
the generator can actually deliver, not only by what the engine can serve.

**Latency does not have an honest rate at all.** Even at 99% delivery the
client p50 is off by two orders of magnitude, so no cap rescues it. §7.4's
"use `--rps` for latency claims" cannot be satisfied with bfb at the pinned
commit, and the project has no usable latency comparison until that changes:
either a load generator that timestamps at send, or a `--jsonl-searches`
capture with `--absolute-time` from which the send schedule can be
reconstructed and the planned-slot lag subtracted. `docs/validation.md`'s
"not validated" list gains a line, because "equal recall at lower latency" is
the publishable statement §8 is aiming at and the second half of it is
currently unmeasured.

What the report must not do meanwhile is print the open-loop percentiles
beside the closed-loop ones as though the two were the same kind of number.
The per-row "N ms of the client's M ms p50 is not server time" flag already
says something is wrong; what this sweep adds is that the flag is not a
caveat on a measurement, it is the measurement.

## 33. The open-loop rows were measuring one line of `select!`, and the fix is one line

Finding 32 established that the `--rps` arms measure bfb's own scheduling
rather than the engine, at every rate, and that no choice of rate rescues
them. It stopped one step short of the mechanism, and the mechanism turns out
to be a single expression.

`process_with_rps` drives its in-flight requests from a `tokio::select!`:

```rust
tokio::select! {
    _ = interval.tick(), if !all_sent => { /* push one request */ }
    Some(Err(err)) = in_flight.next(), if !in_flight.is_empty() => { /* ... */ }
}
```

A request that succeeds returns `Some(Ok(()))`. That does not match
`Some(Err(err))`, and `select!` **disables a branch whose pattern fails** for
the remainder of that invocation, so every successful completion sent the loop
back to waiting on the interval tick. `FuturesUnordered::poll_next` compounds
it: it returns at the *first* ready child and leaves the rest of the ready
queue unpolled. Between them, the loop reaped roughly one completion per
rate-limiter tick.

A request's elapsed time is taken inside its own future. A child woken by its
response but not yet polled is therefore accumulating the wait for its turn in
the drain queue, and charging it to the request. That is the whole of the
"unaccounted" gap, and it explains the shape finding 32 could only describe:
the backlog drains at the tick rate, so *fewer ticks per second means a slower
drain*, and the reported latency is **worst at the lowest offered rate**. It
also explains the delivery decay, which is the same fact seen from the other
side. A loop that cannot keep up with its own plan skips ticks.

Binding the result instead of pattern-matching it keeps the branch enabled:

```rust
    res = in_flight.next(), if !in_flight.is_empty() => {
        if let Some(Err(err)) = res { /* ... */ }
    }
```

### What it was worth

strawmANN, 100,000 × 128 random, ef=128, server on 4-11, bfb on 0-3,
`-t 16 -c 2`, 100,000 queries per arm. Closed-loop saturation of this server:
**17,038 rps**.

| offered | before: delivered | client p50 | after: delivered | client p50 | server p50 |
|--:|--:|--:|--:|--:|--:|
| 2,000 | 1,977 (99%) | 2,722.61 ms | 1,998 (100%) | **0.52 ms** | 0.39 ms |
| 4,000 | 3,779 (94%) | 2,435.83 ms | 3,975 (99%) | **0.63 ms** | 0.41 ms |
| 8,000 | 6,763 (85%) | 1,133.73 ms | 7,963 (100%) | **0.80 ms** | 0.46 ms |
| 12,000 | 9,615 (80%) | 993.25 ms | 11,963 (100%) | **1.03 ms** | 0.66 ms |
| 16,000 | 12,115 (76%) | 825.02 ms | 15,950 (100%) | **1.10 ms** | 0.73 ms |

Three things to read off it.

**The inversion is gone.** Client p50 now *rises* with offered load, 0.52 ms to
1.10 ms, which is what a queue does. Before, it fell from 2.7 s to 0.8 s, which
is what a drain backlog does.

**The gap is 0.12 to 0.37 ms.** Client minus server, across a factor of eight in
load. Finding 32's control was the closed-loop row at 80 µs; the open-loop rows
now sit in the same range instead of three orders of magnitude away.

**The server-side figure moved too**, 10 ms to 0.4 to 0.7 ms. Starved of reaping,
the loop delivered its requests in bursts, and the server was honestly
reporting the queue those bursts made. So the old rows were not merely
mismeasuring the engine. They were *loading* it differently from what the arm
said it was offering.

### Two constants were the bug measuring itself

`GENERATOR_CEILING_QPS` was 4,000, derived from finding 32's delivery decay.
The fixed generator delivers 99 to 100% at every rate tested up to 30,000/s on the
same four client cores, 29,734 of 30,000 against a server saturating at
32,900, above which the two ceilings cannot be told apart on this host. Raised
to 30,000, which puts every one of §4's fractions back on the engine's own
saturation rather than on the generator's.

And `BFB_PIN` was a string in `workloads.py` while the binary was whatever had
last been built next door. §9 pins the toolchain because "a benchmark that
silently changes its load generator is not a benchmark", and nothing enforced
it for the one tool that generates every number. `check_bfb_pin` now refuses a
run whose checkout is not the pinned commit, since a mismatch is not a caveat:
the stamp in `run.json` would name a commit that did not produce the row.

### What this licenses, and what it does not

§7.4 reserves `--rps` for latency claims. It is now an instrument that can
satisfy that, so "equal recall at lower latency", the publishable statement §8
is aiming at, has both halves measurable for the first time. It is not
measured yet: every number above is strawmANN against itself on a synthetic
corpus, run to establish that the generator works, not to compare engines. The
next full run is the first that can carry an open-loop latency comparison, and
until it exists the report's own detector decides whether the banner prints:
`open_loop_is_instrument_limited` reads the inversion out of the rows rather
than asserting it, so it will fall silent on its own if the fix holds and keep
firing if it does not.

The fix lived on a fork (`dev @ 6f216634`, one commit ahead of upstream
8b6dec0a) and belonged upstream: it is a correctness bug in `bfb --rps` for
anyone using it, not a strawmANN-specific patch.

**Merged as qdrant/bfb#172 on 2026-08-27**, and `BFB_PIN` moved to upstream
`dev @ 0c1aafee`. `src/stats.rs` at the two commits differs only in the wording
of the comment above the fix, diffed at both, not inferred from the subject,
so every open-loop row measured under `6f216634` was measured with the same
instrument the harness pins now, and the two are comparable. `bench/patches/`
kept the patch for a while, against the worry that reports already published
name the fork commit and deleting the only way to rebuild it would strand those
rows. It was deleted on 2026-09-07, because the worry was misplaced: the fix is
upstream, `6f216634` is reachable in `qdrant/bfb` for anyone who wants that
exact commit, and every pin since is a plain upstream commit any clone can
check out.

### The fix held, and the detector could not say so

The dbpedia-openai-1m run of 2026-08-26 was that next full run, driven by
`dev @ 6f216634`, the same code the pin now names upstream. The unaccounted
gap it measured:

| arm | 619/s | 866/s | 1,114/s |
|---|---|---|---|
| sm-dbp1m | 0.106 ms | 0.105 ms | 0.104 ms |
| qd-dbp1m | 0.141 ms | 0.141 ms | 0.139 ms |

Flat, and about a tenth of a millisecond against a 0.9-1.6 ms server p50.
That is the fix working: no inversion, no collapse, a constant client-side
cost of roughly 100 µs that distorts nothing.

It is not the only run where it worked, and the detector's problem is sharper
than "it fired once". Every arm measured since the pin shows the same clean
shape:

| arm | gaps across the ladder | client p50 |
|---|---|---|
| sm-sift-perf | 0.119 → 0.121 ms | 0.66 ms |
| qd-sift-perf | 0.157 → 0.158 → 0.196 ms | 0.77 ms |
| strawmann | 0.109 → 0.123 → 0.115 ms | 0.48 ms |
| qdrant | 0.160 → 0.192 → 0.353 ms | 0.71 ms |
| sm-dbp100k | 0.098 → 0.099 → 0.107 ms | 0.93 ms |
| sm-dbp1m | 0.106 → 0.105 → 0.104 ms | 0.90 ms |

Only the last one *descends*, by two microseconds, and only it got the
banner. Which report was told its rows were worthless came down to which
direction a tenth of a millisecond happened to wobble.

The report said the opposite. `open_loop_is_instrument_limited` gated on
`arms[0] > arms[-1]`, *any* fall, and a two-microsecond drift across the
ladder satisfied it. So the banner printed, at `.0f` precision:

> These are not latency measurements. The time the client saw beyond what the
> server reported gets *smaller* as the offered rate gets larger (sm-dbp1m
> **0 ms** unaccounted at 619/s against **0 ms** at 1,114/s), and no queue
> behaves that way.

A sentence whose own evidence rounds to zero on both sides, telling the reader
to discard the rows the fix had just made good. The claim above, that the
detector "will fall silent on its own if the fix holds", was wrong as
written: it could not fall silent, because measurement noise cannot be
distinguished from a signature by a strict inequality.

It now requires both halves of what was actually observed in the broken case:
the gap must fall to at most half of itself across the ladder (the measured
break was 1,404 ms to 340 ms, a factor of four), **and** it must be at least a
quarter of the client-side p50, since a lag far under the server's own latency
cannot be what the client is seeing. Both thresholds are named constants next
to the function, and the figures no longer round their own evidence away.

The general lesson is worth more than the constant: **a detector that exists
to fall silent when something is fixed needs a threshold, not a comparison.**
The one direction it will never report is "fixed".

## 34. strawmANN's graph quality is a per-build draw, and the draw is wider than the engine gap

Every published table this project has produced was one pass. `fullrun.py
--reps 3` existed and had never been used, so nothing had ever asked what two
builds of the same collection, with the same parameters, on the same host,
agree about.

Measured 2026-08-22, SIFT1M, `m=16 ef_construct=100`, three alternated passes
per engine, each pass rebuilding `bench2` from scratch, recall scored over all
10,000 held-out queries against the fp64 oracle:

| ef | strawmANN's three builds | spread | Qdrant's three builds | spread |
|--:|---|--:|---|--:|
| 128 | 0.98881 / 0.98624 / 0.98271 | 0.00610 | 0.98749 / 0.98766 / 0.98735 | 0.00031 |
| 256 | 0.99743 / 0.99467 / 0.99131 | 0.00612 | 0.99700 / 0.99670 / 0.99703 | 0.00033 |
| 512 | 0.99957 / 0.99678 / 0.99337 | 0.00620 | 0.99945 / 0.99938 / 0.99947 | **0.00009** |

Sixty-nine times the build-to-build spread at `ef` 512. Qdrant's three graphs
are indistinguishable from each other; strawmANN's are not.

> **Re-measured 2026-08-23, on a quiet host, and the effect is real but smaller.**
> The run above had eighteen contamination events; this one had none.
> strawmANN's three builds spread **0.00288** at `ef` 512 (0.99781 / 0.99948 /
> 0.99660) against Qdrant's **0.00009**. A factor of **32**, not 69. The
> difference is not noise in the measurement of the noise: `buildParallel`'s
> insertion order depends on thread scheduling, so foreign load perturbs the
> interleaving and produces worse graphs, and roughly half the originally
> reported spread was the contamination rather than the builder.
>
> The conclusion stands and the number moves. At the top of the frontier
> strawmANN's graphs still disagree with each other by 32 times what Qdrant's
> do, which is wider than the distance between the two engines there.
>
> **Re-measured 2026-08-28 on dbpedia-openai-1m, and the draw nearly vanishes.**
> Three builds per collection, same engine defaults for `m` and `ef_construct`
> as above, recall@10 over 10,000 held-out queries:
>
> | collection | | ef=128 spread | ef=512 spread |
> |---|---|--:|--:|
> | bench2 | fp32 | 0.00016 | 0.00045 |
> | bench6 | SQ8 | 0.00091 | 0.00037 |
> | bench7 | binary | 0.00061 | 0.00037 |
> | bench8 | PQ | 0.00101 | 0.00040 |
>
> Against **0.00288** at `ef` 512 on SIFT1M, that is roughly seven times
> narrower, and it is uniform across all four collections. The uniformity is
> the informative part: fp32, SQ8, binary and PQ are four different search
> paths over one graph built by one `buildParallel`, so the narrowing belongs
> to the build rather than to any quantisation path.
>
> It also inverts this entry's title for this corpus. At `ef` 128 the fp32
> build spread is 0.00016 while the gap between the engines is 0.0016
> (0.9674 against 0.9689). The engine gap is **ten times the draw**, where on
> SIFT1M the draw was the wider of the two.
>
> **What varies is not the scale.** SIFT1M was 1,100,600 points and this is
> 990,000, so corpus size is not the variable; dimensionality (128 against
> 1536) and metric (euclid against cosine) are. A plausible reading is that
> insertion order matters less where neighbour selection is less ambiguous,
> and distances concentrate differently at d=1536, but this run cannot
> separate dimension from metric, and it compares different points on the
> curve besides (SIFT1M reaches 0.997 at `ef` 512, this corpus 0.989). A
> d=1536 euclid corpus, or SIFT1M under cosine, would separate them.
>
> Qdrant's side is two builds here rather than three, which by `aggregate._rsd`'s
> own rule is one difference and not a spread, so no factor against Qdrant is
> quoted for this corpus.
>>
> **Corpus size ruled out directly, 2026-08-29.** dbpedia-openai-100K at the
> same d=1536 and cosine, three builds, `sm-dbp100k-perf`:
>
> | collection | 100K ef=128 | 100K ef=512 | 1M ef=128 | 1M ef=512 |
> |---|--:|--:|--:|--:|
> | bench2 fp32 | 0.00178 | 0.00062 | 0.00016 | 0.00045 |
> | bench6 SQ8 | 0.00130 | 0.00022 | 0.00091 | 0.00037 |
> | bench7 binary | not measured | 0.00018 | 0.00061 | 0.00037 |
> | bench8 PQ | 0.00160 | 0.00040 | 0.00101 | 0.00040 |
>
> At `ef` 512 the 100K and 1M spreads are the same order, 0.0002 to 0.0006, and
> both sit five to fifteen times below SIFT1M's 0.00288. Ten times the points
> at the same dimensionality changes nothing; what separates SIFT1M from these
> is d=128 against d=1536, or euclid against cosine, and those two are still
> not separated from each other.

> **The mechanism is still open, and it is not unreachable nodes.** The
> per-build instrumentation reports 0 to 16 of them across those same three
> builds, and that number varies with the recall, which is what suggested a
> cause and is not one. With 1,100,600 points and 10,000 queries at limit 10
> there are 100,000 true-neighbour slots, so an average point sits in 0.09 of
> them and sixteen orphans account for 1.45 missed slots: a recall@10 deficit
> of **0.0000145** against a measured spread of **0.00288**. Two hundred times
> too small. Both quantities move with the build because both are downstream of
> the thread interleaving; neither causes the other.
>
> What is left is the ordinary edge set (*which* nodes end up neighbouring
> which, not whether a node can be reached at all) and nothing here measures
> that yet. A graph-diff between two builds of one collection would: how many
> level-0 edges differ, and whether the differing ones are the long-range links
> a traversal depends on.

`recall@1` moves with it, 0.9999 / 0.9973 / 0.9933, with `short_lists` and
`impossible_scores` both zero, so this is not truncation and not a scoring bug.
At `ef` 512 the search visits 512 candidates; a query whose true nearest
neighbour is missed at that width is a query whose answer the graph cannot
reach. Between the best and worst build that is 1 query in 10,000 against 67.
Mean relative distance error tracks it over two orders of magnitude, 8.61e-06
to 7.10e-04, which is the same fact said about how far wrong the misses are.

**It is not the build conditions.** The three builds cost 41.39, 41.43 and
41.42 seconds of wall time, 286.9, 284.7 and 285.4 seconds of CPU, with
identical minor-fault counts and zero runqueue wait. Nothing about the machine
distinguishes the good draw from the bad one.

**It is the parallel builder's insertion order.** `buildParallel` inserts
concurrently under per-node locks, so which thread reaches a node first decides
whose neighbour list fills up and in what order `linkBack` re-prunes. `linkBack`
re-runs the selection heuristic over the existing list plus the new edge and
keeps `m` of them, so an existing back-edge can be evicted; a node whose last
in-edge is evicted is reachable only from wherever else it happens to be
pointed at. §22 fixed the case where `insertOne` memset a list a concurrent
`linkBack` had already written, and §23 recorded the plateau closing. What is
left is not a lost write, it is a legal pruning decision whose *outcome* depends
on a race.

### What it costs the comparison

At the top of the frontier (the operating point production systems use)
strawmANN's median build reads 0.9965 at `ef` 512 against Qdrant's 0.9994, and
the matched-recall row there is **0.90x**. On its best build it reads 0.99957
against Qdrant's 0.99945 and leads. A single-pass run can therefore report
either engine ahead at high recall, and both would be honest measurements of
different graphs.

That is what n=1 was hiding, and it is why the 2026-08-21 report showed
strawmANN's recall "plateauing below Qdrant's" while §23 had recorded 0.9994
four days earlier. Neither was wrong. They were different draws.

### What the harness now does about it

`aggregate.py` medians the recall sweeps as well as the throughput (it used to
copy pass 1's sweep verbatim beside a median-of-three qps, pairing one build's
recall with three builds' throughput) takes each interval as the union of the
passes', and records `rep_spread` per point. The recall table prints it, and
says under the table which engine's graphs disagree with each other by more
than the two engines' medians differ.

### What would fix it

Not a search parameter: `ef` 512 already visits four times what `ef` 128 does
and buys 0.0007. A build that cannot be reached from the entry point at that
width will not be reached at 4096 either (§23 measured exactly that). The
candidates are a deterministic parallel build (partition the insertion order so
the interleaving cannot vary), or a post-build reachability repair pass (cheap
to check, since the test that guards §22's fix already walks in-edges) or
refusing to prune a node's last in-edge in `linkBack`. Which of those is right
is not settled here; that this is a defect worth fixing is.

## 35. What the residency confound was worth: 1.3%

§7.4 now compares at one residency (`decisions.md`), so the question the old
reports raised and never answered, "residency can move throughput and latency
too", is answerable by measuring the same engine both ways.

strawmANN, SIFT1M, three interleaved passes each, same binary, same gated host,
2026-08-23. `pinned` is anonymous RAM, its own default; `cached` maps a file and
is the residency both engines can serve:

| row | pinned | cached | cached/pinned |
|---|--:|--:|--:|
| W0 | 4,290 | 4,268 | 0.995x |
| W3 | 1,830 | 1,724 | **0.942x** |
| W4 | 22,750 | 22,534 | 0.990x |
| W5 | 23,067 | 22,761 | 0.987x |
| W9 | 191 | 195 | **1.021x** |
| W10-ef32 | 41,420 | 40,514 | 0.978x |
| W10-ef512 | 6,503 | 6,338 | 0.975x |
| W13 | 51,555 | 51,369 | 0.996x |
| W11 | 416 | 400 | 0.960x |

**Median 0.987x across sixteen rows, range 0.942x to 1.021x.** Mapping a file
instead of holding anonymous RAM costs strawmANN about 1.3%, worst case 6% on
W3, and W9 is 2% *faster* that way, which is within its own noise floor and is
the reminder that a 1-3% row is not a result.

Two things follow.

**The comparison never rested on it.** The matched-recall ratios on the same run
are 1.10x to 1.94x. A confound worth 1.3% cannot produce them, so the earlier
mismatched-residency reports were not wrong about who was faster. They were
unable to say how much of the margin was theirs, which is a different and
smaller defect than it looked.

**And it is no longer a caveat.** "Residency can move throughput" was true,
unbounded, and printed above a table of ratios. It is now a number, measured on
the same host in the same session, and the run that produces the ratios holds
the residency fixed anyway.

The build side is not covered here: W2's time-to-Green is 38.1 s pinned against
40.1 s cached, one pass each, and no noise floor covers the build rows.

## 36. W4 at d=1536 is the idle I/O core, not the search

The one search row where Qdrant is ahead on a clean, one-residency run is W4 on
dbpedia-openai-100K: 4,292 q/s against 4,482, **0.96x**, and reproducible,
4,269 / 4,295 / 4,292 against 4,501 / 4,476 / 4,482 across three interleaved
passes. Tight enough that it is not noise, and worth explaining rather than
noting.

### It is not the I/O path

The first hypothesis was that one I/O thread cannot keep up at d=1536, where a
query carries 6 KB of vector against 512 B at d=128. Measured, 100,000 random
d=1536 points, `-p 64`, same eight cores:

| | rps |
|---|--:|
| `--io-threads 1 --workers 7` | 1,951 |
| `--io-threads 2 --workers 6` | 1,933 |
| `--io-threads 3 --workers 5` | 1,910 |

Adding I/O threads makes it slightly *worse*, tracking the workers given up for
them. The I/O path is not the bottleneck, and the hypothesis was wrong.

### It is the core the I/O thread occupies and does not use

The scheduler columns say it plainly. On that row strawmANN burns 83.4 s of CPU
at **714%** of wall; Qdrant burns 88.7 s at **792%**. Seven saturated workers is
700%, so the I/O thread accounts for about 14% of one core. While holding a
whole one, because the engine pins one thread per core and refuses to
oversubscribe:

    error: BadArgument
     8 cpu(s) but --io-threads 1 + --workers 8 needs 9

So on eight cores strawmANN searches with **seven** and Qdrant searches with
eight, its 43 threads multiplexed across the whole cpuset by the scheduler.
strawmANN is giving up an eighth of the machine to a thread that uses a seventh
of a core.

That is enough to account for the row, and for the shape of the whole
comparison:

| | actual | at 8/7 cores |
|---|--:|--:|
| sift1m, d=128 | 1.53x | 1.74x |
| dbpedia, d=1536 | **0.96x** | **1.09x** |

At d=128 a query is cheap enough that seven saturated cores still beat eight
multiplexed ones by half again. At d=1536 a distance is 1,536 multiply-adds,
throughput goes very nearly linear in core count, and losing an eighth of the
cores is larger than the margin.

### What to do about it is a design question, not a bug

The strict one-thread-per-core pinning is deliberate (§6.3) and is why every
other row is quiet: 212 involuntary context switches on this row against
Qdrant's 41,187, and 4.0 ms of runqueue wait against 57.5 s. The engine buys its
tail latency with that discipline, W4-sat90 on sift1m is p99 1.47 ms against
4.02 ms, and the cost is one core's worth of throughput on the widest vectors.

Three ways out, none free, none taken here:

  * let the I/O thread share a core with a worker when it is measurably idle,
    which trades a little jitter for an eighth of the throughput;
  * scale I/O threads with the *cpuset* rather than fixing them at one, so the
    ratio is right at 8 cores and at 64;
  * accept it and say so, which is what this finding does.

What it must not do is stay unexplained. A reader who sees 0.96x is owed the
reason, and the reason is not that the search is slower.

> **Revised by findings 45, and the row itself has moved.** Two corrections,
> and they run in opposite directions.
>
> The explanation above is too small. With hardware counters on the same row,
> strawmANN's *workers* run at IPC 0.31 against Qdrant's 0.68 on essentially
> identical DRAM traffic per query (53,937 demand fills against 49,732). That
> is a 2.2x per-core efficiency gap, where this entry attributes the row to
> losing one core of eight, a 12.5% one. The idle I/O core is real, measured,
> and not what decides this row; at d=1536 a 6 KB row per neighbour visit
> outruns the out-of-order window's ability to overlap the misses, which is
> findings 2's prefetch budget showing up in the search path.
>
> And the premise has expired. "The one search row where Qdrant is ahead" was
> 0.96x on 2026-08-26; the same row read **1.25x** on 2026-08-29. Four things
> differ between those runs (container against native, 1.19.0 against
> 1.19.1-dev, `--perf` on, and the earlier segment count is not on disk), so
> findings 45 records the move without attributing it. The mechanism this entry
> describes is worth keeping either way (one thread per core is still one core
> given up, and the three ways out are still the three ways out) but it is no
> longer the explanation of a Qdrant win, because there is no longer a Qdrant
> win on that row to explain.

## 37. The recall deficit at d=1536 needs the scale too, and it costs the comparison

`fullrun.py --dataset dbpedia-openai-1m` had never been run. The first one
(2026-08-26, 990,000 × 1536, cosine, one pass per engine, `m=16
ef_construct=100`) produced the fastest strawmANN numbers this project has
measured and no comparison at all, because §8's differ stopped at T2.

```
T3 ANN statistical equivalence: FAIL
  strawmann recall@10 = 0.9666  [0.9629, 0.9699]
  qdrant    recall@10 = 0.9801  [0.9772, 0.9827]
  CIs DO NOT overlap. A QPS comparison here would be at unequal recall (§7.4)
```

strawmANN saturates at 3,595 qps against Qdrant's 1,237, and none of that is
publishable as a ratio: §7.4 compares at equal recall and the recall is not
equal. The gate is not being pedantic. 2.9x throughput bought with 1.35 points
of recall is a different engine, not a faster one.

### It is not a bad draw

Findings 34 measured strawmANN's build-to-build recall spread at 0.0061 on
SIFT1M, wide enough that a single pass can flatter or damn a build. So the
first question the CIs raise is whether this build was unlucky. The sweep
answers it:

| ef | strawmANN | Qdrant | delta |
|--:|--:|--:|--:|
| 32 | 0.8887 | 0.9302 | −0.0415 |
| 64 | 0.9399 | 0.9633 | −0.0234 |
| 128 | 0.9675 | 0.9808 | −0.0133 |
| 256 | 0.9813 | 0.9898 | −0.0085 |
| 512 | 0.9895 | 0.9944 | −0.0049 |

Below at every `ef`, monotonically worse as `ef` falls, reaching seven times
findings 34's spread at `ef` 32. A draw moves one point on a curve; this moves
the curve.

### It needs both the dimension and the scale

The same differ, the same night, on the two tiers that hold one variable each:

| corpus | n × d | strawmANN | Qdrant |
|---|---|--:|--:|
| sift1m | 1M × 128 | 0.9888 | 0.9879 |
| dbpedia-openai-100K | 100K × 1536 | 0.9803 | 0.9791 |
| dbpedia-openai-1m | 1M × 1536 | 0.9666 | 0.9801 |

Ahead at 1M × 128. Level at 100K × 1536. Behind at 1M × 1536. Neither the
dimension nor the point count produces the deficit alone, which is what makes
this a graph-construction question rather than a kernel one: the search path is
the same code at all three tiers, and T1 passes here with `max=3.576e-7`, so
the distances are right and the graph the search walks is not.

### The quantized path has the same shape

T4 passes for both engines, `quantization_dominance` holds, but the fidelity
gap is not small:

| | median abs score delta | Kendall τ vs fp64 oracle | ids not common |
|---|--:|--:|--:|
| strawmANN sq8 | 4.866e-1 | 0.6628 | 38,160 |
| Qdrant sq8 | 1.897e-3 | 0.7641 | 23,830 |

Two orders of magnitude on the score delta, over a corpus where the fp32 graph
is already behind.

### What this licenses

The single-engine rows are licensed (conformance `0764e5339cc87e09`, T2, which
§8 accepts for kernels and cost model), and they are worth having: time to
green 248.8 s for 990k × 1536 against SIFT1M's 23.8 s, and exact search at
10 qps × 6.08 GB per query ≈ 61 GB/s, 81% of this host's measured 75.4 GB/s
aggregate. The scan kernel is bandwidth-bound rather than leaving throughput
in the ISA.

What is not licensed is any sentence comparing the two engines at this tier,
and that is the tier the README calls the headline.

## 38. The noise floor is one engine's, and the front page bands the other with it

`bench/results/noise.json` gives W3 an RSD of 0.46%, and `compare.py` uses it
as the ± band for the *ratio*, both engines. Every one of its six repetitions
is `engine_comm=strawmann` (`bench/results/nf/rep1..6`, 2026-08-18), and the
file records a `dataset` and a `harness_hash` but no engine, so nothing on
either side says the number describes one arm only.

Over the twelve runs this project has recorded at the SMT-off environment hash:

| | n | mean qps | RSD | min | max |
|---|--:|--:|--:|--:|--:|
| strawmANN W3 | 12 | 1,830.2 | 4.54% | 1,724.2 | 2,013.2 |
| Qdrant W3 | 12 | 1,470.2 | **13.16%** | 934.1 | 1,805.9 |
| strawmANN W4 | 12 | 22,854.3 | 0.72% | 22,533.8 | 23,107.0 |
| Qdrant W4 | 12 | 14,716.6 | 1.27% | 14,250.3 | 14,978.5 |

Qdrant's run-to-run spread on the single-query row is 28× the floor that bands
it. The saturating row is fine on both. This is one row, and it is the one the
README leads with.

**The scoping in that table's first line is load-bearing.** "At the SMT-off
environment hash" is not decoration: this host has since run with SMT on, and
findings 46 records a W4 figure being read against these twelve runs as though
the two were commensurable. They are not, and the hash is what says so.

The cost is already on the front page. `search, p=1` read 1.11x on 2026-08-23
and 1.26x on 2026-08-25, and the movement is not a movement: 1,553 and 1,384
are both unremarkable draws from a distribution spanning 934 to 1,806. Two
consecutive headline numbers differ by 14% because the denominator wandered.

The harness already refuses to carry a floor *across corpora*. Every
comparison document that is not sift1m's prints `NO NOISE FLOOR ... run-to-run
spread does not carry between corpora`. Spread does not carry between engines
either, and nothing says so.

> **Confirmed 2026-08-28, larger than recorded, and the cause is found.** The
> first three-pass sift1m run folds Qdrant its own floor on W3: **28.16%**
> against strawmANN's 0.51%, from 1,180 / 2,013 / 1,402 qps. W4 is 0.76%
> against 0.73%, exactly as this entry says. With the real floor in hand
> `compare.py` now bands W3 at ±84.5% and prints `parity` ("no measured
> difference, not a small one") which is the right answer for the headline
> row.
>
> The spread is not jitter. Findings 44 traces it to Qdrant's optimizer still
> running when W3 starts, so this entry's number is real and its subject was
> never Qdrant's search.
>
> A doubt raised earlier the same day is withdrawn: two Qdrant passes on
> dbpedia-openai-1m spread 0.94% on W3, and that was read here as evidence one
> of the two numbers was measuring something else. It was neither the corpus
> this entry is about nor, at two passes, a spread at all by
> `aggregate._rsd`'s own rule.

> **Both engines measured clean, 2026-08-29 16:03 to 18:02, and W3 finally has a
> two-engine floor.** The complete clean pair (`foreign=0`, `gate=pass` on all
> 8 label directories x 32 rows, T4, `8983d7b1b8da68b4`) folds W3 at **0.56%
> on both arms**, strawmANN 1,675 / 1,657 / 1,671, Qdrant 1,972 / 1,953 /
> 1,971. `floor_for` combines them to 0.56% and `noise_band` gives the ratio
> **±2.38%**.
>
> This entry's claim survives its own correction. The floor *was* one engine's,
> and banding the ratio with strawmANN's 0.46% was wrong, but the number that
> made the case, Qdrant's 13.16% over twelve runs and 28.16% on 2026-08-28, was
> not Qdrant's run-to-run spread either. It was findings 44's residue. With the
> residue removed both engines sit at 0.56%, and the honest statement is
> narrower than the one above: a floor must be measured per engine because
> nothing guarantees they match, not because these two differ by 28x.
>
> What the real band buys is the opposite of what the wide one did. At ±84.5%
> W3 read `parity`, "no measured difference", and that verdict is now
> withdrawn: at ±2.38% the ratio is **0.85x** (1,671 against 1,971), a 15%
> deficit six times the band. The row the README leads with does have a winner,
> it is Qdrant, and the floor that hid it was the polluted one.

`regression.floor_for` already prefers a `--reps N` run's own folded spread
over the global file, so §7.4's three passes fix this where they are used. What
is missing is the refusal in between: a single-pass ratio banded by another
engine's floor should say so the way the cross-corpus case does.

**Since built.** `compare.parity_band` returns the half-width only when a floor
exists for the row, and `compare.floor_covers` answers "was this floor measured
on both engines whose ratio it would band?" with `True`, `False`, or `None` when
the floor does not say. A floor that cannot answer yes no longer bands anything.
This paragraph is kept as the statement of the gap rather than as an open one.

## 39. `--segments 1` is a request, and the answer depends on the corpus

Chasing the banners on `report-dbpedia-openai-1m-sm-dbp1m-vs-qd-dbp1m.html`.
The report refuses a comparative claim because §8's differ stopped at T2, and
findings 37 reads that refusal as a graph-construction deficit at 1M × 1536.
The refusal is real. The reading rests on a table with a third variable in it.

### What `--segments 1` actually produces

Measured directly, 2026-08-26, on `qdrant/qdrant:v1.19.0` with the harness's own
collection settings (`--segments 1 --indexing-threshold 1
--full-scan-threshold 10`). The segment *count* is the wrong thing to read, and
reading it alone is what made the first version of this finding wrong.

**200,000 × 128.** Two segments, green, all points indexed, and the second one
is empty:

| segment | index | vectors |
|---|---|--:|
| `6f52a3d4` | `hnsw` | 200,000 |
| `7fd0a9c2` | `plain` | 0 |

`matrix.dat` in the HNSW segment is 102,400,004 bytes, which is exactly
200,000 × 128 × 4. Every point is in one graph. The `plain` segment is
Qdrant's write target: it carries no vector index, so it *would* be scanned
exhaustively, but there is nothing in it to scan. At this tier "two segments"
is one graph and an empty box, and it inflates nothing. Polling for two
minutes never merges them, because there is nothing to merge.

**990,000 × 1536.** The tier the report is about. Five segments, and here they
are not empty:

| segment | index | vectors |
|---|---|--:|
| `b9921cf1` | `hnsw` | 342,100 |
| `97e303e4` | `hnsw` | 251,100 |
| `819b9b81` | `hnsw` | 208,600 |
| `104ab40a` | `hnsw` | 188,200 |
| `67d1ca8b` | `plain` | 0 |

Four populated graphs summing to exactly 990,000. Qdrant searches each at the
full `ef` and merges, so at a nominal `ef` of 128 it explores four graphs where
strawmANN explores one, and each of its graphs is a quarter the size, which
makes each one more accurate per candidate visited as well. That is the
mechanism findings 33 measured at 8-versus-2 segments, here at four-versus-one.

### The count is not constant across the tiers findings 37 compares

`collections.json` for `bench2`, from the runs that produced the table in
findings 37:

| corpus | n × d | strawmANN | Qdrant | recall@10 delta |
|---|---|--:|--:|--:|
| sift1m | 1M × 128 | 1 | 2 | **+0.0014** |
| dbpedia-openai-100K | 100K × 1536 | 1 | 2 | **+0.0012** |
| dbpedia-openai-1m | 1M × 1536 | 1 | **4** | **−0.0135** |

(Deltas read from each label's own `conformance.json` T3 line, not from the
table in findings 37, which gives Qdrant 0.9879 on sift1m where the file on
disk says 0.9874.)

Qdrant held two segments on both tiers where strawmANN is level or ahead and
four on the one where it is behind, and on the evidence above, "two" at those
tiers is one graph plus an empty box while "four" is four graphs. So the tiers
where strawmANN wins were single-graph-against-single-graph, and the tier where
it loses was one graph against four. The confound covaries perfectly with the
result the table is used to explain.

### The fix is a collection setting, not an engine change

`default_segment_number` is a target the optimizer cannot reach while the
merged segment would exceed `max_segment_size`, and that ceiling, not the
target, is what produced four graphs out of a 6.08 GB corpus. Raising it above
the whole corpus lets the optimizer do what was asked. Patched onto that same
990,000 × 1536 collection:

```
max_segment_size: 20000000     five segments -> two, green, ~3 minutes
  51711a46  hnsw   990,000 vectors
  00c2243a  plain        0
```

One graph over every point, plus the empty appendable. That is structurally
what strawmANN serves: `collection.zig` keeps one graph and scans the pending
tail past `graph.count` exhaustively, and after a settled load that tail is
empty. The two engines are then the same shape, and `ef` means the same thing
on both sides of the comparison.

`workloads.py` now passes `--max-segment-size`, sized from the corpus rather
than fixed. A constant that does not scale with the collection it governs is
the bug `FULL_SCAN_THRESHOLD_KB` above already learned once.

### What this does not settle

How much of the 0.0135 deficit was the segments. Findings 33's own arithmetic
(8 segments 1.0000 against 2 segments 0.9978 on SIFT1M) puts four-against-one
at a few thousandths, so most of the gap is likely still strawmANN's, but the
ef sweep in findings 37 cannot narrow it, because every Qdrant point on that
curve came from the same four-graph collection. A re-run with the ceiling set
is what separates them, and it is the same re-run that retires the interleaving
and noise-floor banners.

> **The ceiling is in the harness, and sift1m is clean, 2026-08-29.** With
> `--max-segment-size` sized from the corpus, every Qdrant collection on the
> settled sift1m pair reports `segments_count: 2`, bench0, 1, 2, 6, 7, 8 and
> 12, each with all of its points indexed. By this entry's own dissection at
> this scale that is one populated `hnsw` graph plus the empty `plain`
> appendable, so the tier is one graph against strawmANN's one and `ef` means
> the same thing on both sides. That removes the confound from every sift1m
> comparison this project publishes, findings 38's and 44's W3 included.
>
> What is still owed is the tier this entry is actually about. The four
> populated graphs were on dbpedia-openai-1m, and that re-run has not happened,
> so how much of findings 37's 0.0135 deficit was the segments remains exactly
> as open as it was.
>
> **And on sift1m the policy is a no-op**, measured 2026-08-30. A pass at
> `--segment-policy as-deployed` (`default_segment_number: 0`, no ceiling,
> Qdrant left entirely to itself) produces `segments_count: 2` on all seven
> collections, which is what `equal-work` forces. Every row lands within 2% of
> the forced pass. At 1M x 128 the corpus fits under Qdrant's own
> `max_segment_size`, so its optimizer merges to one populated graph without
> being asked, and the two policies are the same experiment. The distinction
> `SegmentPolicy` draws is real only where the corpus exceeds that ceiling,
> which is the d=1536 tier; a reader of the enum would reasonably expect a
> choice that this corpus does not offer.

## 40. The matched-recall comparison is refused by the tier it exists to answer

Not a measurement. A question about a gate, raised while tracking the banners on
the dbpedia-openai-1m report, and left as a question because the answer changes
what §8 licenses.

`matched_recall_refusal` withholds the equal-recall comparison when
`licenses_comparative` is false. That flag is T3, and T3 fails when the two
engines' recall confidence intervals do not overlap **at a fixed `ef`**. Its
own message says so: "the engines are at unequal recall, so QPS is not
comparable".

But matched recall does not assume equal recall. It *constructs* it: at each
recall one engine actually reached, it interpolates what the other served.
That is §7.4's instruction in full ("compare at equal recall; a single QPS
number without its recall is meaningless") and it is the remedy for exactly
the condition T3 reports. Gating it on T3 means the report withholds the one
comparison that survives, precisely when it is needed.

The data is on disk. Both arms carry five bfb W10 points with recall and qps,
and the comparison computes without re-measuring anything:

| recall@10 | strawmANN qps | Qdrant qps | ratio |
|--:|--:|--:|--:|
| 0.9302 | 6,852 | 3,199 | 2.14x |
| 0.9633 | 3,983 | 1,962 | 2.03x |
| 0.9808 | 2,129 | 1,185 | 1.80x |

Against the fixed-`ef` ratios the same report prints and refuses, this is the
more defensible number, and for a second reason: it is largely **robust to the
segment confound of findings 39**. Whatever configuration an engine used to
reach a given recall, one graph or four, the cost of getting there is in its
qps. A fixed-`ef` ratio cannot say that; this can. Not perfectly robust, since
four smaller graphs may search faster in wall clock than one large one at the
same recall, but far more so than the comparison it is being withheld in favour
of.

**Deliberately not changed.** Loosening a §8 licence gate because a banner is
inconvenient is the exact pressure this project exists to resist, and "the tier
does not mean what the gate uses it for" is a spec argument (§8.5, §8.9, §7.4)
rather than a bug fix. If §8 accepts it, the gate becomes `licenses_perf`, the
numbers are publishable per engine, plus the existing STALE, contamination and
missing-W10 refusals, which are about whether the two curves are measurable
against each other at all. T1 and T2 both pass here with `max=3.576e-7`, so the
engines demonstrably compute the same scores; only their ANN recall at one `ef`
differs.


## 41. Qdrant's tail is runqueue wait, and it scales with `ef`

Findings 36 measured this once, on one row: 4.0 ms of runqueue wait against
Qdrant's 57.5 s, and 212 involuntary context switches against 41,187, on W4 at
d=1536. It read there as a property of that row, the idle I/O core. The
dbpedia-openai-1m run of 2026-08-28 has it on the whole W10 sweep, and the
sweep shows it is not a constant tax but one that **grows with the search
effort**.

Both engines are pinned to the same eight cores (`--server-cpus 4-11`; the
container is started with `CpusetCpus=4-11`, so this is not a container
artefact).

| row | strawmANN wait | Qdrant wait | ratio | strawmANN migrations | Qdrant migrations |
|---|--:|--:|--:|--:|--:|
| W10-ef32 | 0.101 s | 14.53 s | 143x | 0 | 97,874 |
| W10-ef128 | 0.146 s | 32.75 s | 225x | 0 | 109,682 |
| W10-ef512 | 0.300 s | 79.07 s | 264x | 0 | 119,954 |
| W13 | 0.059 s | 0.99 s | 17x | 0 | 210,841 |
| W9 | 2.860 s | 12.91 s | 5x | 0 | 5,963 |

143x → 225x → 264x as `ef` goes 32 → 128 → 512. More work per query means more
threads runnable at once, and past the point where runnable threads exceed the
cpuset the excess is queueing rather than searching. strawmANN's **zero**
migrations across every row is the other half: §6.3's one-thread-per-core
pinning means a worker never leaves the core it started on, so there is no
queue to wait in.

**This is the mechanism behind the tail, and not behind the throughput.** W4
p99 is 20.07 ms against 30.66 ms and W4-sat90 is 2.89 ms against 7.24 ms, while
W4 throughput is 1.03x. Inside the noise floor (this run's three strawmANN
passes measure a median 0.87% relative spread over 24 rows; see the caveat of
findings 38, which is that a floor is always one engine's). At saturation both
engines are bandwidth-bound and the scheduling tax does not change the total;
it changes who waits, and how long the unlucky query waits. A latency SLO would
see this where a throughput benchmark does not.

**What this row cannot say.** `sched_coverage` for Qdrant is **0.106 on W3 and
0.310 on W4**, the two headline rows, so the harness withholds their
runqueue figures and this finding must not be extended to them. Coverage is
~1.0 on both sides across the W10 sweep, which is where the claim is made.

The reason is not row length, which is what this entry first said.
`procstat.py` computes `sched_coverage` as `oncpu_s / (cpu_user_s +
cpu_system_s)`, and the two halves come from different places: the denominator
is the *thread-group* line of `/proc/<pid>/stat`, which keeps the CPU of
threads that have since exited, while the numerator is summed over
`task/<tid>/schedstat` for the threads still alive when the row ends. W3's
0.106 therefore says that **89% of Qdrant's CPU on that row was burned by
threads that no longer existed**, it retires pool threads between rows, where
strawmANN's seven pinned workers never exit, which is why its coverage is ~1.0
everywhere and why the asymmetry is structural rather than incidental.

So the fix is not a faster sampler. It is to bank per-thread deltas *during*
the row (poll `task/*/schedstat` on an interval, accumulate per tid, and
retire a tid's last value when it disappears) at whatever interval keeps the
polling cost below what it is measuring.

> **Built and measured 2026-08-28, and W3 and W4 do not show the effect.**
> `procstat.SchedSampler` banks the counters as the row runs, at a measured
> `threads x 45 us` per poll. W3 and W4 re-measured on dbpedia-openai-1m with
> it on, both engines native, all four rows `gate=pass` and unstamped after the
> harness re-measured through four contamination events:
>
> | row | engine | runqueue | on-cpu | runq/on-cpu | migrations |
> |---|---|--:|--:|--:|--:|
> | W3 | strawmANN | 0.324 s | 43.05 s | 0.75% | 0 |
> | W3 | Qdrant | **0.018 s** | 81.18 s | **0.02%** | 3,162 |
> | W4 | strawmANN | 0.046 s | 99.95 s | 0.05% | 0 |
> | W4 | Qdrant | **0.537 s** | 87.70 s | **0.61%** | 43,921 |
>
> These four rows record `sched_coverage` 1.000, and that figure is an
> artefact rather than a measurement: the first cut of the sampler overwrote
> coverage with 1.0 whenever it ran, which discarded the very signal that
> makes this entry, how much of a row's CPU belongs to threads that exited.
> Fixed the same day, so coverage is computed from the survivors either way
> and `sched_sampled` says whether the counters were banked. The runqueue and
> migration figures above are the banked sums and are unaffected.
>
> On W3 Qdrant waits *less* than strawmANN, in absolute seconds and as a
> fraction of its own on-cpu time. On W4 it waits 11x more, not the 143x-264x
> the W10 sweep shows. This entry says the claim "must not be extended to"
> those rows; measured, the extension would have been wrong, so that caution
> was load-bearing rather than pedantic.
>
> What survives every row is migrations: 3,162 and 43,921 against strawmANN's
> zero. §6.3's pinning holds everywhere; the queueing cost it avoids does not.
>
> **Why W10 differs is open.** It runs at roughly twice W4's throughput, which
> is one candidate. Another is that W3 and W4 follow an upload, and the
> indexing threads exiting during them are what drove coverage to 0.106 in the
> first place, so those two rows may be measuring a Qdrant still finishing
> background work, which is a different problem from scheduling.
>
> **The second one was right** (2026-08-28, findings 44). On sift1m W3 the
> runqueue wait is 54.6 s, 32.7 s and 0.004 s across three passes, and it
> tracks how much of Qdrant's index build was still running: coverage 0.140,
> 0.051 and 0.897, CPU per query 5,024 µs, 3,252 µs and 528 µs. So W3's
> runqueue wait is a search thread queueing behind its own engine's background
> work, not thread-pool oversubscription. The W10 sweep, which follows no
> upload, still shows 991x to 2,783x at d=128 and remains this entry's
> subject.
>
> Single pass per row and no conformance row, so this is development-grade like
> the rest of this page, and it is a smaller claim than the one it replaces
> rather than a larger one.

Measured from the two complete interleaved passes of an aborted three-pass run.
No conformance row was written (the differ runs after the passes and the run
was stopped before it), so under §8 none of this licenses a comparative claim.

> **Re-measured 2026-08-29 on the settled sift1m pair, and the `ef` scaling
> does not reproduce.** Three interleaved passes, `foreign=0` and `gate=pass`
> throughout, `sched_sampled` on every row, so this is the clean-run test this
> entry never had. Qdrant's W10 runqueue wait, all three passes:
>
> | row | pass 1 | pass 2 | pass 3 | median | strawmANN | ratio |
> |---|--:|--:|--:|--:|--:|--:|
> | W10-ef32 | 2.81 s | 2.79 s | 2.81 s | 2.81 s | 0.0020 s | 1,416x |
> | W10-ef64 | 2.45 s | 2.37 s | 2.45 s | 2.45 s | 0.0019 s | 1,317x |
> | W10-ef128 | 2.10 s | 2.06 s | 2.09 s | 2.09 s | 0.0013 s | 1,657x |
> | W10-ef512 | 1.66 s | 1.54 s | 1.55 s | 1.55 s | 0.0037 s | 416x |
>
> **Half of this entry's title is wrong.** "Qdrant's tail is runqueue wait"
> holds and holds hugely. Three orders of magnitude, against strawmANN's zero
> migrations on every row, which is §6.3's pinning doing exactly what the entry
> says. "And it scales with `ef`" does not: the wait *falls* monotonically as
> `ef` rises, 2.81 s to 1.55 s, and the passes agree to two decimals, so this
> is not a draw. The original 143x -> 225x -> 264x was one pass on one corpus
> (dbpedia-openai-1m, d=1536); the claim was then extended to d=128 in the note
> above, and at d=128 measured cleanly it runs backwards.
>
> The direction has a reading the entry never considered. A wider `ef` is more
> work *per query*, so the row serves fewer queries per second and makes fewer
> scheduling decisions per second: W10-ef512 runs at a fraction of ef32's
> rate. Queueing tracks the arrival rate, not the depth of each search, which
> is the opposite of "more work per query means more threads runnable at once".
> What scales with `ef` is the *number of chances to queue*, and it scales down.
>
> W3 inverts too, as findings 44 leads one to expect once the residue is gone:
> Qdrant waits **0.003 s** against strawmANN's 0.027 s. The entry's caution
> that the claim "must not be extended to" W3 and W4 was right twice over.

## 42. Qdrant's SQ8 recall@10 plateaus, and the signature is the recall@1 gap

`decisions.md` §5 records the cause already: strawmANN's quantized rescore pool
is `max(asked, ef)` and Qdrant's is `limit`-sized, so at `limit = 10` Qdrant
rescores ten nodes of a walk however wide the walk was. What that section
predicts but never measured is what the pool does *as `ef` rises*, and the
answer is: nothing. The pool is `limit`, `limit` does not depend on `ef`, so
search effort buys top-1 and cannot buy top-10.

The signature is the gap between the two recalls, which isolates it from every
other difference between the engines:

| | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
|---|--:|--:|--:|--:|--:|
| **fp32** strawmANN | +0.0022 | −0.0021 | −0.0000 | +0.0004 | +0.0002 |
| **fp32** Qdrant | +0.0046 | −0.0019 | −0.0018 | −0.0013 | −0.0004 |
| **SQ8** strawmANN | +0.0062 | −0.0001 | +0.0008 | +0.0004 | +0.0003 |
| **SQ8** Qdrant | **+0.0702** | **+0.0804** | **+0.0867** | **+0.0922** | **+0.0967** |

`recall@1 − recall@10`. Under fp32 every cell is zero to within ±0.005. Both
engines' graph search and the harness's recall join are sound. Under SQ8
strawmANN stays at zero and Qdrant's gap **widens monotonically**, because
recall@1 climbs the way the graph lets it (0.897 → 0.990, essentially
strawmANN's 0.887 → 0.989) while recall@10 saturates (0.827 → 0.893).

The practical consequence is the part the fixed-`ef` rows do not say: **for
Qdrant under SQ8, tuning `ef` is close to useless for top-10 quality.** ef=32 →
512 is a 16x cost for +0.066 recall@10; strawmANN spends the same and gains
+0.109, ending at 0.989. This is why §7.4 refuses those rows a ratio, and it is
a stronger reason than "the pools differ in size": they differ in whether they
respond to the one knob the row is sweeping.

Reproduced across both passes to three decimals (ef=32: 0.827/0.827; ef=512:
0.893/0.894), so it is deterministic configuration behaviour rather than noise.
Qdrant's oversampling would be the knob that moves it; this run left it at the
default, which is what a user gets.

> **Demonstrated 2026-08-28, and the knob moves it exactly as predicted.**
> `bench6` swept at oversampling none/2/4/8, `rescore true` throughout,
> 10,000 queries, native Qdrant 1.19.1-dev. The baseline reproduces the
> v1.19.0 rows above to within 0.003, so the two builds are comparable.
>
> | ef | none | 2 | 4 | 8 | gain |
> |--:|--:|--:|--:|--:|--:|
> | 32 | 0.8268 | 0.8935 | 0.9133 | 0.9528 | +0.1260 |
> | 64 | 0.8612 | 0.9422 | 0.9431 | 0.9528 | +0.0916 |
> | 128 | 0.8800 | 0.9677 | 0.9690 | 0.9690 | +0.0890 |
> | 256 | 0.8884 | 0.9808 | 0.9822 | 0.9823 | +0.0939 |
> | 512 | 0.8929 | 0.9888 | 0.9904 | 0.9904 | +0.0975 |
>
> recall@10. The control is recall@1 over the same sweeps, which at `ef` >= 128
> is **invariant to four decimals across an eightfold change in pool size**:
> 0.9640 at every setting for `ef` 128, 0.9786 for 256, 0.9894 for 512. The
> top-1 was never pool-limited and the top-10 always was, which is the whole
> claim.
>
> Three things the inference could not say. The plateau is **entirely
> recoverable and cheap**: `oversampling 2` takes ~90% of the total gain and 4
> and 8 add almost nothing above `ef` 64, so the pool stops binding at roughly
> twice `limit`. Once it does not bind, **Qdrant matches strawmANN** -- 0.9888
> against 0.9891 at `ef` 512 -- so the three quantized rows §7.4 refuses are
> measuring a *default*, not an engine that recalls worse. And `ef` 32 is the
> exception: there recall@1 does move (0.8970 -> 0.9471), because at the
> narrowest search a wider stage-1 set changes which point comes back first.
> The invariance claim is made at `ef` >= 64 and not below it.
>
> What this does not license is a re-run of the comparison at matched
> oversampling. §7.4 compares at equal recall, and equalising it by giving one
> engine a knob the other did not need is a different experiment than the table
> is written to hold; `validation.md` item 6 records it as the question rather
> than the answer.

## 43. The disk-backed engine takes ~1.6 million major faults per upload

Nothing in this document had measured major faults. They are the demand-paging
side of the same architecture the storage table shows from the write side, and
at d=1536 they are the clearest single number for what disk-backed costs.

| row | strawmANN | Qdrant | strawmANN s | Qdrant s |
|---|--:|--:|--:|--:|
| W1 upload, no index wait | 0 | 1,581,923 | 10 | 47 |
| W2 upload and index | 1 | 1,575,151 | 263 | 521 |
| W6-upload SQ8 | 2 | 1,657,748 | 290 | 262 |
| W7-upload binary | 2 | 1,825,976 | 271 | 332 |
| W8-upload PQ | 8 | 1,724,710 | 396 | 1,065 |
| W12-upload payloads | 70 | 430,486 | 6 | 54 |

Single digits against roughly 1.6 million, on every corpus build. Each one is a
synchronous page-in, and W1 + W2 together are **257.5 s against 516.5 s,
2.01x**. The build-time half of the trade whose serving half is findings 36's
tail. PQ is the worst case at 2.7x, where training reads the corpus again.

The same run's totals: 29.8 GiB written against 536.8 GiB, an **18x write
amplification**, and 42.6 GiB on disk against 11.0 GiB with peak RSS 37.3 GiB
against 48.2 GiB. Qdrant is a quarter of the disk footprint and more than
eighteen times the write traffic; strawmANN buys its numbers with residency.
Neither is free, and the storage table is the only place the comparison says so.

**A caveat that applies to the whole run, and to W12 in particular.** bfb is run
with `--skip-field-indices` because it panics on strawmANN's `UNIMPLEMENTED`
for `CreateFieldIndex`, so a supporting engine full-scans every filtered query.
Qdrant's W12 at 14 qps is therefore the cost of the *other* engine's missing
feature, not a measurement of Qdrant's filtered search, and must not be read as
one.

## 44. Green is not idle, and W3 has been measuring Qdrant's index build

The first fully clean run of the sift1m `--perf` labels (three interleaved
passes, `foreign=0` and `gate=pass` on all 192 rows, T4 licensed
(`6635c00ecb43a339`)) put a per-engine noise floor on W3 for the first time
and found Qdrant's at **28.16%** against strawmANN's 0.51%. Findings 38
predicted that and is confirmed. What the same run also shows is *why*, and it
is not a property of Qdrant's search.

Qdrant's W3, pass by pass:

| pass | qps | secs | CPU | CPU/query | `sched_coverage` | runqueue |
|---|--:|--:|--:|--:|--:|--:|
| rep1 | 1,180 | 42 | 251.2 s | 5,024 µs | 0.140 | 54.58 s |
| rep2 | **2,013** | 24 | 26.4 s | **528 µs** | 0.897 | **0.004 s** |
| rep3 | 1,402 | 35 | 162.6 s | 3,252 µs | 0.051 | 32.70 s |

strawmANN over the same three passes: 1,687 / 1,676 / 1,693 qps at **520 / 524
/ 518 µs** per query. Flat to three digits.

W3 is a single-query row and should cost about one core. rep2 does, 528 µs
per query, within 2% of strawmANN's. rep1 burns **nine times** that and rep3
six times, for the same 50,000 queries. That is not search, and the ordering is
inverse across every column at once: most CPU, lowest qps, lowest coverage,
largest runqueue wait, longest row.

**`W2`'s own `index_wait_s` says which pass got lucky**: 133 s for rep2 against
96 s and 88 s for rep1 and rep3. bfb's `--wait-index` returns at *green*, and
green is the indexing threshold rather than an idle engine: Qdrant's optimizer
keeps working past it. The pass whose upload happened to wait longest carried
the least residue into the row after it.

Three things follow.

**Findings 38's spread has a mechanism, and findings 41's runqueue wait on this
row is the same one.** Neither describes Qdrant's search. The 1,567x runqueue
ratio on W3 is a search thread queueing behind its own engine's background
work, and the 28% qps spread is how much of that work happened to be left.

**When the row is clean, Qdrant wins it.** rep2 is 2,013 qps against
strawmANN's 1,687. Every published `search, p=1` figure favouring strawmANN was
drawn from a distribution polluted by this, which is a correction in the other
engine's favour and the reason to make it loudly.

**The counters said so too.** Folded over three passes, Qdrant's W3 reads
106,454 DRAM fills and 19,508 dTLB walks per query against strawmANN's 12,613
and 3,333, eight-fold and six-fold. On the rows where nothing is left running,
the two engines are indistinguishable: W4 IPC 1.03 against 1.02, W10-ef128 1.05
against 1.04, both at ~1.9 GHz.

**Fixed by waiting for the engine rather than for green.** `settle_engine`
watches the engine's own CPU after every upload row and holds the next row
until it is under 0.05 of a core for three consecutive seconds, recording
`engine_settle_s` on the row so a reader can see whether it was needed. It
watches the process rather than asking Qdrant about its optimizer, so strawmANN
returns on the first window and an engine added later needs no new code. The
signal that detects the problem was already on every row: `sched_coverage`,
0.897 on the clean pass and 0.140 and 0.051 on the dirty ones.

> **Re-measured 2026-08-29 with `settle_engine` on, same labels, three
> interleaved passes.** Qdrant's W3:
>
> | | rep1 | rep2 | rep3 | folded | RSD |
> |---|--:|--:|--:|--:|--:|
> | before (2026-08-28) | 1,180 | 2,013 | 1,402 | 1,402 | 28.16% |
> | after | 1,951 | 1,965 | 1,977 | **1,965** | **0.66%** |
> | CPU per query, before | 5,024 µs | 528 µs | 3,252 µs | | |
> | CPU per query, after | 544 µs | 541 µs | 538 µs | | |
>
> Every pass now costs what the one clean pass cost, the spread is gone, and
> the headline row inverts: `search, p=1` reads **0.86x** (Qdrant ahead of
> strawmANN's 1,683) where every table this project has published had it
> the other way. The unsettled run is archived under
> `bench/results/archive-2026-08-28-sift-perf-unsettled/`.
>
> **The mechanism is narrower than this entry first said.** `engine_settle_s`
> is 3.0, 3.0 and 4.0 s on the three passes, with the engine at 0.00 cores
> when it returned: Qdrant was *already quiet* three seconds after green. A
> long-running optimizer would have held the settle for tens of seconds and
> did not. So what the pause changes is not "wait for the work to finish" but
> something about starting the next row a few seconds later, plausibly a
> deferred segment optimization that Qdrant schedules on a short timer after
> the last upsert and that W3's immediate load used to land on top of. What
> exactly happens in those seconds is not established here; that the pause
> removes the residue is.
>
> The other arm is not clean this time: `sm-sift-perf-rep1` carries five
> `foreign` rows (W7-upload, W8-upload, W8, W12-upload, W12) from builds on
> the host between 00:09 and 00:25, so strawmANN's label is refused
> publication and its W3 spread reads 3.24% (rep1 1,778 against 1,680 and
> 1,683). Neither run is a complete clean pair: the archived one has the
> contaminated Qdrant W3, this one the contaminated strawmANN rep1.
>
> One more thing this run fixed in passing: the same labels gave
> `--rps-reference auto` a W4 to read, so W4-sat50/70/90 ran both engines at
> one offered load (5,146 / 7,204 / 9,262 qps on both) and the report no
> longer refuses the cross-engine latency read there. At 50% of Qdrant's
> saturation strawmANN serves p50 487 µs against 759 µs.

> **The complete clean pair, 2026-08-29 16:03 to 18:02, and the paragraph above
> is wrong about the optimizer.** Both arms clean this time: `foreign=0` and
> `gate=pass` on all 8 label directories x 32 rows, `perf=32` and `sampled=32`
> on every one, T4 with `licenses_comparative=true`, hash `8983d7b1b8da68b4`.
> This is the pair `docs/validation.md` item 8 asked for.
>
> | W3 | rep1 | rep2 | rep3 | folded | RSD |
> |---|--:|--:|--:|--:|--:|
> | strawmANN | 1,675 | 1,657 | 1,671 | **1,671** | 0.56% |
> | Qdrant | 1,972 | 1,953 | 1,971 | **1,971** | 0.56% |
>
> strawmANN's 3.24% in the note above was the contamination in `rep1`, not the
> engine: with no foreign rows it folds to 0.56%, the same figure as Qdrant's.
> The headline holds where the previous note put it (`search, p=1` is
> **0.85x**, Qdrant ahead) and now with both floors measured it clears a
> ±2.38% band rather than the ±84.5% one (findings 38). `search, saturating`
> is 2.18x and `exact` 1.30x, both strawmANN's.
>
> **"A long-running optimizer would have held the settle for tens of seconds
> and did not" is withdrawn. It did, and I read the wrong row.** The 3.0 /
> 3.0 / 4.0 s quoted above is the settle after **W2**. The settle after
> **W1**, ingest throughput, is **77.0 / 77.0 / 68.0 s** on this run and
> **73.0 / 78.0 / 66.0 s** on the archived one: Qdrant's optimizer runs for
> more than a minute after the ingest row, on every pass, in both runs. It was
> in the data the whole time and the note generalised from the row nearest to
> W3.
>
> So this entry's original mechanism was right and the hedge on it was not.
> Nothing mysterious happens in a few quiet seconds. `settle_engine` absorbs
> ~70 s of optimizer at W1; W3 runs two rows later and starts clean, which is
> why *its* settle is only 3 s. Unsettled, nothing waited, and how much residue
> W3 inherited depended on how long the intervening upload happened to take,
> exactly what `W2`'s `index_wait_s` showed (133 s for the clean pass against
> 96 s and 88 s). strawmANN returns in 3.0 s at 0.00 cores on every row of both
> arms, so the wait is Qdrant's alone.

## 45. At d=1536 strawmANN's scan is demand-miss bound, and it is not the I/O core

The first `--perf` runs with a spread (three passes on sift1m and on
dbpedia-openai-100K-1536-angular, 2026-08-29, both T4) put hardware counters
beside every row for the first time. Per query, folded over the passes:

| corpus | row | engine | qps | cycles | instructions | IPC | demand DRAM fills |
|---|---|---|--:|--:|--:|--:|--:|
| sift1m | W9 | strawmANN | 184 | 76.0 M | 147.4 M | 1.94 | 315,046 |
| sift1m | W9 | Qdrant | 146 | 55.9 M | 167.5 M | 3.00 | 53,833 |
| db100k | W9 | strawmANN | 100 | 141.5 M | **37.2 M** | **0.26** | **2,279,504** |
| db100k | W9 | Qdrant | 126 | 64.9 M | 59.1 M | 0.91 | 245,985 |
| sift1m | W4 | strawmANN | 22,498 | 0.66 M | 0.67 M | 1.02 | 12,676 |
| sift1m | W4 | Qdrant | 10,292 | 1.01 M | 1.03 M | 1.02 | 17,456 |
| db100k | W4 | strawmANN | 4,291 | 3.35 M | 1.03 M | **0.31** | 53,937 |
| db100k | W4 | Qdrant | 3,445 | 2.62 M | 1.79 M | 0.68 | 49,732 |

Read W9 at d=1536 first, because brute force has no graph to hide behind. The
corpus is 100,000 rows of 6,144 bytes. strawmANN executes **37% fewer
instructions** per query than Qdrant, §6.6.1's wide kernels doing what they
were built for, and takes **2.2 times the cycles** to do it, at an IPC of
0.26 against 0.91. The reason is in the last column: 2.28 million *demand*
DRAM fills per query, 9.3 times Qdrant's. That is 146 MB of the 614 MB corpus
arriving at the core because the core asked for it, line by line, rather than
because the prefetcher had it ready. Qdrant demand-misses 2.5% of the corpus
per query; strawmANN 24%.

At d=128 the same comparison reads IPC 1.94 against 3.00 with strawmANN
*ahead*, 184 against 146, the ratio findings 3 and 6 built the ISA story on.
The kernel is not slower at d=1536; the memory system stops covering for it.
`bruteForceRange` is a plain sequential walk that calls `probe.scoreNode` on
each row, and nothing on that path asks for a line before it needs it. (The
parenthetical this entry first carried -- "`@prefetch` appears once in the tree,
in `visited.zig`" -- was already stale when written: `dist/common.zig` has
`prefetchRead` and `prefetchRow`, and `Probe.prefetch` asks for a neighbour's
first line, which is findings 28's work. The graph path prefetches; the scan
did not.) The scan relies on
out-of-order memory-level parallelism to overlap misses, which is exactly what
findings 2 measured §5.2's outstanding-miss constant against and found 4x
short: "the gap is the prefetch budget". A 512-byte row fits inside that
window; a 6,144-byte row does not, and the counters are the shape of it.

**This revises findings 36.** That entry attributed W4's parity at d=1536 to
the pinned I/O thread idling on one of eight cores, a cost of one eighth. The
counters say strawmANN's *workers* run at IPC 0.31 there against Qdrant's
0.68, on the same DRAM traffic per query (53,937 against 49,732 fills), a
2.2x per-core efficiency gap, not a 12.5% one. The idle core is real and is
not the explanation. HNSW at d=1536 reads a 6 KB row per neighbour visit and
pays the same demand-miss cost W9 pays, per row.

**The counters are not free, and the tables above do not say so.** A pass with
`--perf` dropped and nothing else changed puts Qdrant's W13 at 43,317 qps
against 38,906 with it on, W10-ef32 at 23,299 against 21,278, and W10-ef128 at
10,254 against 9,732, 11%, 9% and 5%. W4 and W9 are unaffected at 1.00x and
1.01x, so the tax lands on the fastest rows rather than uniformly, which is
what a fixed per-request instrumentation cost looks like. One pass against a
three-pass fold, so the figures are indicative; the direction is not in doubt.
No ratio in this entry is drawn across that boundary, both arms of every row
above carry `--perf`, but a qps from a `--perf` run and one from a run without
it are not the same measurement, and nothing on the page currently says so.

> **Attempt one, measured 2026-08-30: no effect, and the counters say why.**
> `bruteForceRange` was given the *head line* of the row two ahead, following
> the reasoning `Probe.prefetch` documents for the graph path -- one line, and
> the hardware streams the rest. db100k re-measured on the same host at the
> same SMT state, three interleaved passes, `--perf` on:
>
> | W9 | before | after | |
> |---|--:|--:|--:|
> | qps | 100 | 99 | 0.99x |
> | demand DRAM fills | 4,559,007,978 | 4,801,192,314 | 1.05x |
> | instructions | 74.40 G | 75.83 G | 1.02x |
> | IPC | 0.263 | 0.266 | 1.01x |
>
> W4 identical to three digits. The only thing that moved is the instruction
> count, which is the prefetches themselves.
>
> **The arithmetic was against it from the start.** W9 scans 100,000 rows for
> each of 2,000 queries, so 4.80 G fills is **24 demand misses per row**, and a
> 6,144-byte row is **96 cache lines**. A quarter of every row misses, spread
> through it. One prefetch can reach one of those 24 -- about 4%, under the
> run-to-run spread. The change could not have shown up.
>
> **What that rules out is worth more than the attempt.** "The hardware
> prefetcher follows the rest of a row once the kernel starts streaming it" is
> true enough on the graph path to be worth writing down, and it is *false on
> the scan*: streaming sequentially through 6,144-byte rows, the hardware
> covers about three quarters of a row and no more. The stride crosses a page
> every 1.5 rows, which is where the stream is lost. So the miss is not at the
> row's head, and a head-line prefetch is the wrong instrument -- the row wants
> asking for whole.

> **Attempt two settles it, 2026-08-30, and this entry's title is wrong.**
> `bruteForceRange` was given the *whole* row one ahead, all 96 lines, via
> `dist.common.prefetchRow`. It worked, and it bought nothing:
>
> | W9 | baseline | head-line | whole-row | vs baseline |
> |---|--:|--:|--:|--:|
> | demand DRAM fills | 4,559,007,978 | 4,801,192,314 | **2,136,969,192** | **0.47x** |
> | cycles | 283.1 G | 285.3 G | 280.2 G | 0.99x |
> | qps | 100 | 99 | 101 | 1.01x |
> | instructions | 74.40 G | 75.83 G | 154.81 G | 2.08x |
>
> The demand misses **halved**, the prefetch does exactly what it was written
> to do, and the row did not get faster by one percent. (The IPC "improvement"
> to 0.552 is arithmetic, not speed: same cycles, twice the instructions,
> because the prefetches are themselves instructions.)
>
> **The scan is bandwidth-bound, not demand-miss bound.** W9 reads
> 100,000 x 6,144 B = 614.4 MB per query and serves 100.6 of them a second,
> which is **61.8 GB/s against this host's measured 73.1 GB/s aggregate: 85%
> of the bus.** There is no latency left to hide: the bytes have to cross, and
> no prefetch creates bandwidth. Halving the demand misses moved the *fetches*
> from demand to prefetch and left the traffic exactly where it was.
>
> **Findings 37 already had this right, and this entry contradicted it.** That
> entry measured the same kernel on dbpedia-openai-1m: "exact search at 10 qps
> x 6.08 GB per query = 61 GB/s, 81% of this host's measured 75.4 GB/s
> aggregate -- the scan kernel is bandwidth-bound rather than leaving
> throughput in the ISA". Both corpora put the scan at **61 GB/s**, ten times
> apart in size, which is what a bus limit looks like. The demand-miss count
> this entry leads with is a *symptom* of streaming 614 MB per query, not the
> cause of the time, and "demand-miss bound" in the title is withdrawn.
>
> **W4 never moved and could not have.** It is 1.00x on every counter across
> all three builds, because `bruteForceRange` is not W4's path, W4 is HNSW
> search. This entry's extension of the demand-miss story to W4's IPC 0.31 is
> therefore untested by any of this, and rests on the same reasoning that just
> failed for W9.
>
> The code is reverted. It halves a counter, costs 80 G instructions a row-set,
> and buys no time; findings 2's ~64-fill window was never the binding
> constraint, so the "ask for the row in halves" fallback is moot.
>
> **What it exposed instead: Qdrant's W9 cannot be doing what strawmANN's is.**
> On the same run Qdrant serves 128.1 qps, which at 614.4 MB per query implies
> **78.7 GB/s, above the 73.1 GB/s this host can deliver**, with 474 M demand
> fills against strawmANN's 2,137 M. A full fp32 scan of the corpus per query
> is not physically available at that rate, so Qdrant is reading less: caching
> across concurrent queries, answering from something narrower, or not scanning
> exhaustively at all. Until that is known, W9 is not an exact-search
> comparison and the 100-against-128 in the table above should not be read as
> one. This is findings 21's shape exactly. A row that looked like a search
> result and was a configuration difference.

> **Answered 2026-08-30: the amortisation is real, and W9 is a fair row.**
> W9 re-run at `-p 1`, where no engine can share a pass between queries, against
> the bus bound for an exhaustive fp32 re-read (73.1 GB/s / 614.4 MB = 119 qps):
>
> | | `-p 1` | `-p 8` | scaling | implied stream at `-p 8` |
> |---|--:|--:|--:|--:|
> | strawmANN | 66.7 | 100.6 | 1.51x | 61.8 GB/s, 85% of the bus |
> | Qdrant | **57.7** | **128.1** | **2.22x** | **78.7 GB/s, above the bus** |
>
> Both arms `gate=pass`. **Qdrant obeys the bound alone and beats it
> concurrently**, which is only possible if it reads the corpus less than once
> per query, so the batched shape of its plain index (`BatchFilteredSearcher`,
> one walk scoring every query in the batch) is reaching this row. The
> suspicion that Qdrant might not be scanning at all is dismissed: at `-p 1` it
> is not merely under the bound, it is **slower than strawmANN**, 57.7 against
> 66.7. Nothing is being skipped. W9 is a fair comparison and its ratio stands.
>
> **strawmANN is at the ceiling of what its design allows.** At `-p 1` it
> streams 41.0 GB/s against this host's 45.7 GB/s single-core figure, **90% of
> one core's bandwidth**. At `-p 8`, 61.8 GB/s against 73.1 aggregate, 85% of
> the machine. It is not leaving cycles on the floor at either end; it is doing
> the one thing its exact path can do, which is re-read the corpus per query,
> as fast as the memory system permits.
>
> **So the gap is architectural and the lever is named.** The engine ahead is
> the one that reads less, not the one that reads faster, and 1.51x against
> 2.22x concurrency scaling is that difference stated as a slope. No kernel
> work, ISA work or prefetch touches it. The prefetch attempt above is the
> proof, halving demand misses for nothing. What would touch it is scoring
> several concurrent exact queries per pass of the arena, and `handlers.BatchJob`
> (findings 27) already established that a request's queries can be fanned
> across workers; what does not exist is the inverse, gathering queries *into*
> one scan.
>
> One caveat on the pairing: the `-p 1` figures are one pass and the `-p 8`
> figures come from the three-pass run above, so this is a cross-run
> comparison. strawmANN's `-p 8` W9 is 100 / 99 / 101 across three independent
> builds, and the effect here is a slope difference of 47%, so the pairing is
> not carrying the result, but an interleaved run at both concurrencies is
> what would license quoting these as table figures.

One thing the same run does *not* explain: db100k's W4 read 0.96x on
2026-08-26 and 1.25x here. Qdrant 1.19.0 in a container then, 1.19.1-dev
native now, `--perf` on now, and the earlier run's segment count is not on
disk to compare with this one's two (findings 39). Four candidates and no way
to pick, so it is recorded and not attributed.


## 46. The 30% W4 "regression" was two environments, and the hash already said so

The settled sift1m pair put Qdrant's saturating row at **10,313 qps**. Findings
38's table, twelve runs deep, puts it at **14,716 ± 1.27%**, range 14,250 to
14,978. strawmANN over the same change reads 22,510 against a recorded 22,854,
1.5%, inside its own band. One engine apparently lost 30% of the headline
throughput row and the other lost nothing, reproducibly: three independent
sift1m `--perf` runs give 10,292 / 10,292 / 10,313.

Two candidates were measured, one variable at a time, against the clean pair.

| row | clean pair | `as-deployed` | no `--perf` |
|---|--:|--:|--:|
| W3 | 1,971 | 1,956 (0.99x) | 2,025 (1.03x) |
| **W4** | **10,313** | **10,404 (1.01x)** | **10,335 (1.00x)** |
| W10-ef128 | 9,732 | 9,793 (1.01x) | 10,254 (1.05x) |
| W9 | 146 | 149 (1.02x) | 148 (1.01x) |

Neither is the 30%. And **the premise was the defect.**

### The two numbers are from two hosts

Findings 38's aggregate is scoped in its own first line: "over the twelve runs
this project has recorded at the SMT-off environment hash". This host now runs
**SMT on**: `run.json` records `cores: 24`, and `/sys/devices/system/cpu/smt/control`
reads `on`. `bench/setup.py` keeps `smt=` in the environment hash body for
exactly this reason, and `check_smt` is deliberately "declared, not required
off" so a host that ships with SMT on can still measure. The hash separates the
two states; nothing pools them.

So 10,313 against 14,716 is not a number against another number. It is one
environment against a different one, and this project's own discipline refuses
that comparison the way it refuses a noise floor across corpora (findings 38)
or across engines. There is no established regression here to explain, and the
remaining candidate: Qdrant 1.19.1-dev native against 1.19.0 in a container.
Could not have isolated a version effect anyway, because it is confounded with
the environment change. It was not run.

The core count is not what moved. `--server-cpus 4-11` still resolves to eight
*distinct physical* cores, `core_id` 8 through 15. What changed is that each of
them now has a live sibling in 16-23 that under SMT-off did not exist, which is
the ambiguity `setup.py`'s `smt_topology` comment names: "with SMT on, whether
`4-11` is eight cores or four cores twice over" is a question the CPU numbers
alone do not answer. Here it is eight.

### What survives, and what it licenses

The clean pair is untouched by any of this. Both engines were measured on one
host in one session, interleaved per §7.2(5), so `search, saturating` 2.18x and
`search, p=1` 0.85x are sound *within this environment*, which is all a ratio
ever claims. The worry that sent this investigation, that 2.18x might be
inflated by a Qdrant regression, dissolves with the premise that raised it.

**An asymmetry worth keeping as a question.** Across the SMT change strawmANN
moved 1.5% and Qdrant 30%. That is the direction findings 36 and 41 predict:
strict one-thread-per-core pinning has no sibling to contend with, and 43
multiplexed threads do. But this is one environment measured against a
recollection of another, which is the error this entry is about, so it is a
hypothesis and not a result. An SMT-off pass over the same labels would settle
it, and it is the only measurement that would.

### The refusal that was missing

The harness refuses a floor across corpora and, since findings 38, across
engines. It has no opinion about a ratio quoted against an aggregate from
another environment hash, because that comparison was made in prose rather than
by a tool: findings 38's table was read as a baseline by a reader who had the
hash in front of him and did not check it. `regression.floor_for` already knows
which run it is reading; what it does not do is refuse a stored aggregate whose
environment hash differs from the run being judged. That is the guard this
entry argues for, and it is the same shape as the two that already exist.
## 47. Qdrant reads back 8.55% of what it writes, and not one byte of it in a search

The `disk read ops` and `disk write ops` columns read `unknown` on every report
this project has produced. The cause was not the counters but the gate: §7.1
checked the `io` controller at `user.slice` and reported block-layer operations
"measurable", while the engines run four levels deeper in a `systemd-run --user
--scope` under `user@UID.service/app.slice`, and `app.slice` passed down only
`memory pids`. A cgroup has `io.stat` where its *parent* enables `io`, so the
file did not exist where it was read. `sm-sift-perf-rel-0908` /
`qd-sift-perf-rel-0908` is the first run with the chain complete, and the first
number out of it is a 1,000x gap:

| | strawmANN | Qdrant |
|---|--:|--:|
| disk read ops | 48 | 49,092 |
| disk read | 0.1 MiB | 1,961.8 MiB |
| disk write | 2,723.0 MiB | 22,942.0 MiB |
| **read back, as a share of bytes written** | **0.00%** | **8.55%** |

Both arms report `io_source: proc+cgroup`, so both figures are block-layer
counts from the engine's own scope. Same instrument, same units.

**It is not the query path.** Every search row reads zero on *both* engines:
W3, W4, W9, W10-ef128 and both W12 grades are `0 ops, 0 MiB` on Qdrant as well.
Searches are served from cache on both sides, and any reading of this gap as a
serving cost is wrong. (W12 is a real filtered search here, on an index both
engines built, and is two graded rows since 2026-09-08. Findings 43's caveat
about `--skip-field-indices` describes the row as it was before 2026-09-03 and
does not apply to these numbers.)

**All of it is the write path, at a constant 41 KiB per operation.**

| row | Qdrant read | Qdrant write | KiB/op | strawmANN read |
|---|--:|--:|--:|--:|
| W0-upload | 3,378 ops / 131 MiB | 409 MiB | 39.7 | 3 ops |
| W2 upload and index | 4,443 ops / 177 MiB | 2,481 MiB | 40.8 | 9 ops |
| W6-upload SQ8 | 4,014 ops / 160 MiB | 2,720 MiB | 40.9 | 12 ops |
| W7-upload binary | 4,791 ops / 190 MiB | 2,711 MiB | 40.5 | 12 ops |
| W8-upload PQ | 3,588 ops / 145 MiB | 2,253 MiB | 41.4 | 0 ops |
| W12-upload payloads | 1,440 ops / 52 MiB | 561 MiB | 37.0 | 9 ops |
| W11-steady | 8,886 ops / 362 MiB | 3,522 MiB | 41.7 | 0 ops |
| W11 | 18,285 ops / 738 MiB | 6,752 MiB | 41.3 | 0 ops |

That uniformity is what rules out the obvious alternative. Cache pressure from
writing eight times as much would produce sporadic misses at whatever size the
fault happened to touch; six unrelated rows all landing between 37 and 42 KiB
is a code path with a fixed read size, not eviction. The signature is a segment
optimizer: Qdrant merges as it ingests, and a merge reads back what it wrote.
strawmANN builds its arenas and never reads them through the block layer again:
48 operations and 0.1 MiB across an entire three-pass run is nothing at all.

This compounds rather than complicates the trade findings 43 records from the
write side. strawmANN writes 8x less *and* reads back essentially nothing;
Qdrant's ingest is read-modify-write and its write amplification already showed
that from one direction. What is new is that the read side is now measurable, so
the storage table states both halves instead of one and a blank.

### What this does not establish

**It is a property of `equal-work`, not of Qdrant.** This run holds Qdrant to
one populated segment. Its optimizer's merge schedule is exactly what
`default_segment_number` governs, so `as-deployed`, where Qdrant resolves it to
the CPU count, would very likely read back a different share, and this figure
must not be quoted as Qdrant's. The two policies differ by 109% on the
saturating row on dbpedia; there is no reason to expect the ingest path to be
more stable across them than the search path.

**One run, three passes, one corpus.** The share is not banded: the noise floor
is folded over qps, not over I/O counters, so 8.55% is a measurement rather than
an interval. And SIFT1M at d=128 is the small end. Findings 43's 18x write
amplification was measured at d=1536, where the same architecture costs more.

**The 41 KiB is unexplained.** It is consistent enough to be a configured read
size, and nothing here reads Qdrant's source to say which. `read_view` and the
optimizer are where it would be, and naming it would turn this from an
observation into an explanation.
## 48. W12-sel10's dash was a missing ladder, and the frontier it hid runs the other way

`rel-0908` published `W12-sel10`, filtered search at 10% selectivity, with a
dash where its ratio belongs, and the reason looked like bad luck. strawmANN
measured recall@10 1.0000 against Qdrant's 0.9900, a difference of 0.0100, and
§7.4 refuses a ratio outside a 0.01 band. One ten-thousandth further apart and
the widest gap on the corpus would have been published as 7.5x.

It was not luck, and it was not the band. §7.4 compares at *equal* recall, and
`report_data.frontier_points` gets there by joining `<prefix>-ef<ef>`
throughput rows to the sweep's recall at the same `ef`. W12-sel10 measured
throughput at one `ef`, so `matched_ratios` had nothing to interpolate. The
comparison was not refused for being close; it was never attempted. W10 has
been the unfiltered version of that ladder since it existed, and the filtered
row had no equivalent.

`W12-sel10-ef{32,64,128,256,512}` is the ladder, W12-sel10's own search at
five widths, in `sm-w12-ladder` / `qd-w12-ladder`, one pass each.

| | ef 32 | ef 64 | ef 128 | ef 256 | ef 512 |
|---|--:|--:|--:|--:|--:|
| strawmANN recall@10 | 0.98790 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| strawmANN qps | 486 | 383 | 371 | 363 | 367 |
| Qdrant recall@10 | 0.79689 | 0.93328 | 0.99022 | 0.99922 | 0.99998 |
| Qdrant qps | 4,701 | 3,376 | 2,230 | 1,228 | 584 |

**At equal recall, strawmANN serves 0.21x to 0.66x of Qdrant's rate**: Qdrant
1.52x to 4.80x faster, four bracketed anchors where there was a dash:

| matched recall | anchor | ratio |
|--:|---|--:|
| 0.98790 | strawmANN ef32 | 0.21x |
| 0.99022 | Qdrant ef128 | 0.21x |
| 0.99922 | Qdrant ef256 | 0.32x |
| 0.99998 | Qdrant ef512 | 0.66x |

**Re-measured 2026-09-09, and the top of the range moved.** The labels above
were re-run to add the narrow grade's ladder, and the same four anchors came
back 0.21x to **0.66x** rather than 0.21x to 0.44x. The move is one row:
Qdrant's `ef=512` came in at 584 qps against 827, ~30%, on the slowest and
longest rung of a single pass. Nothing here is banded, see below, and a
spread whose top end moves by a third between two passes of one row is the
reason. The shape is what reproduces; the top anchor's value is not.

### The two frontiers have opposite shapes, and that is the result

strawmANN's filtered throughput is **flat**: 486 to 367 qps across a sixteen-fold
range of `ef`, recall 1.0 from `ef=64` and nothing bought above it. Qdrant's is
**steep**: 4,701 to 584, an 8x throughput cost to climb from 0.797 to 0.99998.

So the gap narrows monotonically as the recall requirement tightens (0.21x at
~0.99, 0.44x at ~0.99996) and `ef=128`, the single point `rel-0908` published,
is close to the worst point on the curve for strawmANN. The dash was hiding a
comparison that is bad for strawmANN at every recall *and* getting better as
the bar rises, which is more informative than either the 7.5x the raw rates
suggested or the refusal that replaced it.

Extrapolating past `ef=512` is not measured and is not claimed. Qdrant is at
584 qps and still short of 1.0 at the top rung, already *below* strawmANN's flat
367, so on this pass the curves have crossed inside the measured range, and the
0.66x anchor is that crossing rather than an extrapolation toward it. Whether it
reproduces is the question the 30% move on that rung raises.

### What this does not establish

**The rung is the check, and it caught the first attempt.** The `ef=128` rung
must reproduce `W12-sel10` (same search, same width) and in the first
measurement it came in 3.49x (strawmANN) and 2.31x (Qdrant) above it, because
the ladder had been given `-p 8` copied from W10 while the graded rows send no
`-p`. That run measured a concurrency its own row does not and was discarded;
its ratios, 0.47x to 0.74x, are wrong and are recorded here only so the number
is not met twice. The ladder now sends the row's whole argv with the width
substituted, the rungs reproduce at 1.012x and 1.000x, and the test asserts the
argv rather than field by field. Checking `-n` and `ef` individually is what
let `-p` diverge.

**One pass, own labels, no noise floor.** These are not banded: a spread across
the frontier, from a single pass per engine, with `--perf` off. The bands in the
table are `recall_band`'s propagation of the recall CIs through the
interpolation, not run-to-run variance, and the anchors are correlated because
adjacent ones share bracketing segments.

**`equal-work`, and the two ladders are not comparable to each other.** Qdrant
is held to one populated segment throughout. And W12's rows send no `-p` while
W10's ladder sends `-p 8`, inherited from the original W12, so the filtered
frontier and the unfiltered one are measured at different concurrencies and a
ratio between *them* would be meaningless. Only the two engines' filtered
frontiers are compared here.
## 49. The 0.01 recall band licensed a ratio that points the wrong way

`rel-0908` publishes `W12-sel1`, filtered search at 1% selectivity, at
**0.67x**, a loss, and the ratio is licensed: strawmANN's recall@10 is 1.0000
and Qdrant's 0.99998, two hundred-thousandths apart and far inside §7.4's 0.01
band. Nothing refuses it and nothing marks it.

The matched-recall read over the same grade's ladder says strawmANN is
**1.92x to 3.32x faster**, over seven anchors. Not a different magnitude, the
other direction.

| | ef 32 | ef 64 | ef 128 | ef 256 | ef 512 |
|---|--:|--:|--:|--:|--:|
| strawmANN recall@10 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| strawmANN qps | 3,491 | 3,387 | 3,442 | 3,493 | 3,418 |
| Qdrant recall@10 | 0.99582 | 0.99976 | 0.99996 | 1.00000 | 1.00000 |
| Qdrant qps | 5,459 | 4,073 | 2,723 | 1,764 | 1,052 |

The whole disagreement is in that last fraction of recall. strawmANN reaches
1.0 at `ef=32` and holds ~3,400-3,500 qps at every width above it: the narrow
filter leaves 2,000 points, it walks them, and `ef` cannot steer a scan.
Qdrant is at 0.99996 by `ef=128` and does not reach 1.0 until `ef=256`, by
which point it serves 1,764, and 1,052 at 512. So the last **0.00004** of
recall costs Qdrant a factor of 2.6, and reaching parity of recall at all costs
it the comparison.

§7.4's band asks "are these two recalls within 0.01" and correctly answers yes.
The frontier answers a different question, "what does each engine serve *at* a
recall both reach", and near the top of the curve those are not the same
question, because 0.01 of recall there is worth more than the ratio being
measured. `_matched_table` has printed that caveat for as long as it has
existed: "near the top of the sweep 0.003 of recall spans a factor of 1.8".
What is new is a *licensed* row where it inverts the sign, and a page that
published the inverted one as a headline loss.

So the two W12 grades disagree about which engine wins, and both disagreements
are real: at 1% strawmANN is 1.92-3.32x faster at matched recall, at 10% it is
0.21-0.66x (findings 48). The dispatch boundary in the middle is the whole
subject of the two grades, and a single `ef` per grade reported one of them
backwards and the other as a dash.

### What this does not establish

**The two figures come from different runs and are not each other's
correction.** `rel-0908`'s 0.67x is three passes with `--perf` attached; the
ladder is one pass without it, and its `ef=128` strawmANN rung is 3,442 qps
against the published row's 2,571. What is compared here is the *direction* two
methods report over one grade, not one number against another. A ladder
measured inside a `--perf` publication run is what would settle the magnitude,
and the rows are in the table now, so the next one does it.

**Single pass, and the neighbouring grade's spread moved 30% on re-measurement**
(findings 48). These anchors are not banded either. The inversion is not a
30%-sized effect, 0.67x against 1.92x is a factor of three and a sign, but no
individual figure above should be quoted as a rate.

**It is an argument about reading, not a defect in the band.** A 0.01 band is
the right refusal for a row-by-row table, which cannot know the shape of either
curve; the fix is not a narrower band but the matched-recall table beside it,
which `report_charts.filtered_matched_recall_table` now renders for both
grades. What the band cannot do is notice that it is standing on a cliff.

## 50. The 2.20x is three factors multiplied, and only one of them is per-query efficiency

`rel-0921` publishes W4 at **2.20x**, the same ratio `rel-0908` published, and
the page reads it as throughput. The hardware counters say what it is made of.
Throughput is `cores_busy x frequency / cycles_per_query`, and for this row that
identity closes to within 5 qps on both engines:

| | cores busy | eff. GHz | cycles/query | predicted | measured | 8-core ceiling |
|---|--:|--:|--:|--:|--:|--:|
| strawmANN | 7.50 | 1.97 | 649,557 | 22,789 | 22,784 | 24,311 |
| Qdrant | 5.55 | 1.91 | 1,022,894 | 10,352 | 10,351 | 14,922 |

So the ratio factors as **1.57x less work per query, 1.35x better core
occupancy, 1.03x frequency** (1.57 x 1.35 x 1.03 = 2.18 against the measured
2.20). Roughly a third of the gap is Qdrant not filling the cores it was given,
not strawmANN being cheaper per query.

The occupancy is not an artefact of this run or of the segment policy. Four
runs, three weeks, both Qdrant experiments:

| label | policy | W4 qps | cores busy | cycles/query |
|---|---|--:|--:|--:|
| `qd-sift-perf-rel-0921` | equal-work | 10,351 | 5.55 | 1,022,894 |
| `qd-sift-perf-rel-0908` | equal-work | 10,446 | 5.48 | 998,093 |
| `qd-sift-perf-rel-0907` | **as-deployed** | 10,492 | 5.48 | 995,699 |
| `qd-sift-perf-rel-0903` | equal-work | 10,364 | 5.47 | 1,006,195 |

`as-deployed` gives Qdrant `default_segment_number` = CPU count and changes the
occupancy by 0.00 cores, so findings 39's segment question does not explain it.
W4 offers `-p 64 -t 16` and strawmANN reaches 7.50 cores on the identical
client, so it is not the load generator either. What remains is Qdrant's own
search runtime, and on this row it is a 31% shortfall against its own
per-query cost.

Two per-query counters worth keeping beside that. DRAM traffic is 795 KiB
against 1,086 KiB, so 27% less memory moved per query, which tracks the work
ratio. **Branch misses do not:** 3,645 per query against 3,577, a 1.9%
difference, while instruction counts differ by 34%. strawmANN's 5.4 MPKI
against Qdrant's 3.5 is entirely the denominator. The misses are the graph
walk's, both engines pay the same number of them, and at roughly 17 cycles each
they are about 10% of strawmANN's per-query cycles. Any plan that reads MPKI as
a strawmANN defect is reading the instruction count.

### What this does not establish

The equal-`ef` ratio is not the comparison this page exists to make. The
matched-recall table on the same run reads **2.04x to 2.12x** (CI 1.95x to
2.29x), and that is the number to quote. The decomposition above explains the
2.20x; it does not license it.

## 51. The W12 ladders, measured inside a publication run: the directions hold, the magnitudes shrink

Findings 49 closed by naming what would settle it: "a ladder measured inside a
`--perf` publication run is what would settle the magnitude, and the rows are in
the table now, so the next one does it." `rel-0921` is that run: both grades'
ladders, three passes, `--perf`, one licensed environment. Full-precision recall,
not the page's four decimals:

| ef | sm recall | sm qps | qd recall | qd qps |
|--:|--:|--:|--:|--:|
| **sel1** | | | | |
| 32 | 1.000000 | 2,562 | 0.995950 | 6,465 |
| 64 | 1.000000 | 2,564 | 0.999650 | 5,136 |
| 128 | 1.000000 | 2,564 | 0.999970 | 3,832 |
| 256 | 1.000000 | 2,562 | 1.000000 | 2,617 |
| 512 | 1.000000 | 2,567 | 1.000000 | 1,714 |
| **sel10** | | | | |
| 32 | 0.987640 | 382 | 0.796170 | 4,614 |
| 64 | 1.000000 | 302 | 0.933450 | 3,354 |
| 128 | 1.000000 | 299 | 0.989470 | 2,209 |
| 256 | 1.000000 | 301 | 0.999140 | 1,391 |
| 512 | 1.000000 | 299 | 0.999940 | 836 |

At the first width where Qdrant actually reaches 1.000000 on the narrow grade
(`ef=256`, 2,617 qps) strawmANN serves 2,562, so **0.98x**, rising to **1.50x**
at `ef=512`. Findings 49's single-pass ladder said 1.92x to 3.32x. The sign of
its claim survives (the published `W12-sel1` row of 0.67x still understates
strawmANN, which ties rather than loses at matched recall) and the magnitude
does not.

On the wide grade the anchors come back **0.17x to 0.36x** against findings 48's
0.21x to 0.66x. The shape reproduces, as 48 predicted it would, and the top
anchor moved by a third for the second time running, as 48 warned it does.

The reason to trust these over the earlier ladders is not that they are newer:
it is that both grades, both engines and the recall they are joined against came
out of one gated run at one environment hash, where the 2026-09-09 ladders were
single-pass, `--perf`-less, and separately labelled. `compare.py` would refuse a
ratio across the two sets as STALE, and it is right to; what is compared here is
one method against itself.

### What this does not establish

Nothing here is banded. The ladders carry a spread now, but the anchors are
interpolations between rungs and the interpolation is not error-propagated. And
the whole sel1 result turns on the last 0.00003 of recall: at `ef=128` Qdrant
is at 0.999970 and serves 3,832, which rounds to the same 1.0000 the page prints
for strawmANN. A reader working from the four-decimal table would conclude
Qdrant is 1.49x faster at matched recall. That is findings 49's cliff, still
standing, now inside the published page's own rounding.

## What is open, in priority order

Only open work belongs here. Finding numbers are identifiers, cited from
`src/*.zig` and `bench/harness/*.py`, so they are never renumbered and a closed
entry leaves a gap rather than a shuffle. Ranked by what a wrong or missing
number costs. Anything not listed is settled; what was wrong and what fixed it
is in [`bugs.md`](bugs.md), and why a choice was made is in
[`decisions.md`](decisions.md).

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

**10. Fold the W12 result into the entries it settles (findings 51).** Findings
48 and 49 each name an open item that `rel-0921` answers, and the file holds
three sets of anchors with no statement of which run is authoritative.

**11. Annotate the two rows that measure the harness (findings 50).** W13's
server-side p50 is 11 µs against a 137 µs client p50, so 92% of it is the load
generator and the socket. W11 climbs 1,475 to 1,794 to 1,847 qps monotonically
across three passes, so its median is a warm-up average rather than a steady
state.
