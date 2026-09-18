# Dataset runs

The first end-to-end runs of the engine against a real dataset at full scale.
Everything here is **development-grade**: `bench/setup.py check` was failing on
this host when these were taken (§7.1), and none of it is a comparison against
Qdrant, Qdrant was not started. The host has passed the gate since; that makes
these numbers re-measurable, not retroactively publishable. What these runs establish is that the engine is correct and complete
enough to produce a recall curve at 1M scale, and what that curve looks like.

Reproduce:

(`$STRAWMANN_DATA` is the dataset root; see [datasets.md](datasets.md#choosing-the-dataset-directory).)

```sh
./zig-out/bin/strawmann --port 6334 --capacity 1100000 &
cd conformance
cargo run --release -- relevance \
    --base    $STRAWMANN_DATA/sift1m/sift/sift_base.fvecs \
    --queries $STRAWMANN_DATA/sift1m/sift/sift_query.fvecs \
    --ground-truth $STRAWMANN_DATA/sift1m/gt/sift1m.euclid.k100.gt.json \
    --metric euclid --limit 10 --limit-queries 2000 --collection sift1m \
    --ef 16 32 64 128 256 512
```

---

## 1. SIFT1M, 1,000,000 × 128, Euclid

```
uploaded 1000000 points in 1.0s
green in 23.8s
```

| ef | recall@1 | recall@10 | MRDE | recall@10 CI95 |
|--:|--:|--:|--:|---|
| 16 | 0.8420 | 0.7711 | 0.00980 | [0.7522, 0.7890] |
| 32 | 0.9245 | 0.8812 | 0.00410 | [0.8662, 0.8946] |
| 64 | 0.9710 | 0.9499 | 0.00153 | [0.9394, 0.9586] |
| 128 | 0.9900 | 0.9816 | 0.00052 | [0.9747, 0.9866] |
| 256 | 0.9970 | 0.9938 | 0.00019 | [0.9892, 0.9964] |
| 512 | 0.9975 | 0.9973 | 0.00013 | [0.9939, 0.9988] |

2000 held-out queries, recall measured against our own fp64 ground truth with
§8.6's ε-aware tie handling. `recall@100` is `n/a` throughout because the
requests asked for `limit=10`; reporting a number there would misrepresent a
limit as a measurement.

The curve is unremarkable, which is the point, it sits where published HNSW
results on SIFT1M sit, so the index is doing what an HNSW index does rather than
something subtly wrong that still produces plausible aggregates.

**Ingest was 1.0 s for 512 MB of vectors**, ~1M points/s through gRPC, HTTP/2,
protobuf decode and into the arena. §6.2's "upsert decode writes directly into
final storage" is load-bearing here. **Time to Green was 23.8 s** for the
parallel HNSW build.

## 2. The control run is what makes the curve trustworthy

```
exact   recall@1 1.0000   recall@10 1.0000   MRDE 0.00000
```

`--exact` bypasses the graph and brute-forces. It returns **exactly** the fp64
oracle's answer: perfect recall at both depths and a mean relative distance
error of zero to five decimal places.

That single line carries most of the confidence in the table above, because it
tests the whole chain *except* the graph: the fvecs reader, the protobuf
encoding, the wire format, the distance kernel, the ε-aware recall computation,
and the ground truth itself. If any of those were wrong the exact row would not
be 1.0000. So the approximate rows can be read as measuring the graph, which is
the only thing left.

It also settles a question the 50k run raised. There, `recall@1` plateaued at
0.9910 and would not improve past `ef` 128, nine queries in a thousand never
finding their true nearest neighbour no matter how much search effort was spent.
That is the signature of a graph connectivity problem *or* of a tie/ε artefact,
and the exact control distinguishes them: exact search finds all of them, so the
harness and the ties are fine and the plateau is a property of the graph. At 1M
the same measurement reaches 0.9975 and is still climbing at `ef` 512, so
whatever it is does not worsen with scale.

## 3. A harness bug this run found

The first 50k run reported:

```
ef 16    recall@1 1.0000   recall@10 1.0000   MRDE 0.00464
ef 64    recall@1 1.0000   recall@10 1.0000   MRDE 0.00076
ef 256   recall@1 1.0000   recall@10 1.0000   MRDE 0.00050
```

Perfect recall at every `ef`, from `ef` 16 upward, on an approximate index, and
MRDE moving by 9× across the same rows. Two numbers computed from the *same*
returned results disagreeing about whether `ef` matters at all.

`recall_one` called `GroundTruth::is_correct`, which tested membership against
`neighbours(query)`, the **whole cached ground-truth list**, k=100, and broke
ties against `truth.last()`, the *100th* score. Ground truth is cached once at
k=100 and reused for every recall depth (§4.3), so `recall@10` was accepting any
id in the true top **100**. A competent index returns its 11th-nearest somewhere
inside the top 100 essentially always, so `recall@10` read 1.0000 regardless of
effort.

MRDE was unaffected because it compares distances positionally, which is why the
two disagreed and why the bug was visible at all.

The fix is `is_correct_at(query, k, ...)` (since folded into
`GroundTruth::judge_at`), which tests against the top-`k`
prefix and ties against the k-th score *of that prefix*.
`recall_at_10_uses_the_top_10_not_the_cached_k` pins it with a case that returns
the true 91st to 100th neighbours: recall@10 of that is 0, and the old code scored
it 1.0.

**This would have flattened every deliverable that depends on a recall curve** -
W10, T3's recall-vs-`ef` comparison, and §6.7's whole `(ef, oversampling,
rescore)` frontier, into a constant 1.0, for both engines, while looking
entirely reasonable.

## 4. Scroll (W13) over the same collection

```
scrolled 1000000 points in 1000 pages (1000/page) in 0.18s = 5572580 points/s
ids strictly ascending, prost decoded every page
```

Driven through `qdrant-client` rather than our own decoder, because that is the
compatibility question that matters: a wrong field number produces bytes our
decoder tolerates and prost rejects. It found one, see `docs/workloads.md` W13
for the `time`-at-field-2 bug, which corrupted the repeated `result` list.

## 5. The headline dimension needed two fixes before it would run at all

The cosine path at d=1536 had never been exercised, SIFT is Euclid at d=128 -
so before committing to a ~30-minute ground-truth computation the whole chain
was run on a 19,500-vector subset from the shards already fetched. It failed
immediately, twice.

**An oversized request killed the connection.** The harness uploaded 512 points
per request; at d=1536 that is 3.1 MB, past the 1 MiB per-stream request buffer.
`handleData` returned `Error.RequestTooLarge`, which the I/O loop treated as a
*connection*-level fault: GOAWAY, every other in-flight stream destroyed, and
the client left holding `transport error MetadataMap { headers: {} }`.

The 1 MiB limit is legitimate, it is sized for bfb's `-b 100` at d=1536, about
614 KB, but exceeding it is an ordinary client mistake and deserves an ordinary
`RESOURCE_EXHAUSTED`, which is what gRPC returns for a message over
`max_receive_message_length`. It is now stream-level: the oversized body is
drained so the connection stays in sync, that one stream fails with a message
stating the limit, and everything else continues.

This was invisible at d=128, where the same 512-point batch is 262 KB. Only the
headline dimension crosses the threshold, so the entire test suite passed while
the headline tier could not upload a single batch.

**The harness chunked by point count, not bytes.** Fixed at 512 points, a request
is 262 KB at d=128 and 3.1 MB at d=1536. Now sized to a byte target, which also
keeps the request shape constant across tiers, otherwise ingest throughput
between SIFT1M and the headline tier would differ partly because the requests
were differently shaped.

### 5.1 Cosine at d=1536, 19,500 vectors

| ef | recall@1 | recall@10 | MRDE |
|--:|--:|--:|--:|
| 16 | 0.9460 | 0.9284 | 0.00179 |
| 64 | 0.9800 | 0.9766 | 0.00078 |
| 256 | 0.9940 | 0.9880 | 0.00032 |
| **exact** | **1.0000** | **1.0000** | **0.00000** |

The exact row is the one that mattered here: at the headline dimension, with
cosine, ingest-time normalisation and the short-circuit of §8.3 in the path, the
brute-force result still reproduces the fp64 oracle bit for bit. That is the
piece most worth confirming before spending half an hour computing ground truth
against it.

## 6. dbpedia-openai-1m, the headline tier

990,000 × 1536, cosine, real OpenAI `text-embedding-ada-002` embeddings. §4.2's
headline dataset: "the dimensionality Qdrant users actually run". Full pipeline
in `bench/harness/headline.py`.

```
fetch verified   8.9 GB, 0 files outstanding (upstream Git-LFS digests)
convert          26 parquet shards -> base.fbin 6.08 GB + queries.fbin 61 MB
ground truth     990000 x 10000, d=1536, cosine, k=100 -> 31 min, fp64
uploaded         990000 points in 9.5s   (640 MB/s)
green in         269.4s
```

| ef | recall@1 | recall@10 | MRDE | recall@10 CI95 |
|--:|--:|--:|--:|---|
| 16 | 0.8070 | 0.7882 | 0.00409 | [0.7697, 0.8055] |
| 32 | 0.8740 | 0.8692 | 0.00224 | [0.8537, 0.8832] |
| 64 | 0.9135 | 0.9232 | 0.00116 | [0.9107, 0.9341] |
| 128 | 0.9460 | 0.9535 | 0.00066 | [0.9433, 0.9618] |
| 256 | 0.9665 | 0.9709 | 0.00041 | [0.9626, 0.9774] |
| 512 | 0.9810 | 0.9822 | 0.00024 | [0.9754, 0.9871] |
| **exact** | **1.0000** | **1.0000** | **0.00000** |, |

Scroll over the same collection: **990,000 points in 990 pages in 0.17 s**
(5.97M points/s), ids strictly ascending, every page decoded by prost.

### 6.1 The exact control, again

`exact` reproduces the fp64 oracle bit for bit at 990,000 × 1536. That is the
whole chain minus the graph, conversion, the held-out split, the wire, ingest
normalisation, §8.3's short-circuit at the dimension where it is most fragile,
the d=1536 kernel, and the ε-aware recall computation. Everything in the table
above is therefore attributable to the graph and nothing else.

Its 7.8 q/s is the memory-bandwidth floor rather than a slow kernel: 200 queries
× 6.08 GB is 1.2 TB, and at this host's measured 47 GB/s sequential read that is
25.6 s. Brute force is saturating memory, which is what §5.1 predicts it should.

### 6.2 The headline tier is harder than SIFT1M, and the gap widens with effort

recall@10, same engine, same harness, same ground-truth methodology:

| ef | SIFT1M (d=128, euclid) | dbpedia (d=1536, cosine) |
|--:|--:|--:|
| 16 | 0.7711 | 0.7882 |
| 64 | 0.9499 | 0.9232 |
| 128 | 0.9816 | 0.9535 |
| 256 | 0.9938 | 0.9709 |
| 512 | 0.9973 | **0.9822** |

At `ef` 16 the headline tier is *slightly ahead*; by `ef` 512 it is 1.5 points
behind and still climbing where SIFT has flattened. SIFT reaches 0.99 recall@10
around `ef` 150; dbpedia has not reached it by 512.

This is the practical shape of the curse of dimensionality on a graph index, and
it is the argument for measuring on the headline tier rather than on SIFT: an
`ef` chosen from a SIFT curve is badly wrong at d=1536. Anyone tuning `ef` on
SIFT1M and shipping it for 1536-dimensional embeddings is running at roughly
0.95 recall while believing they are at 0.99.

### 6.3 recall@1 and recall@10 cross over

On SIFT1M `recall@1 ≥ recall@10` at every `ef`. On dbpedia they swap between
`ef` 32 and 64:

| ef | recall@1 | recall@10 |
|--:|--:|--:|
| 32 | 0.8740 | 0.8692 |
| 64 | 0.9135 | **0.9232** |
| 512 | 0.9810 | **0.9822** |

Past the crossover the graph reliably finds the right *neighbourhood* while
still missing the single closest point about 2% of the time. That is consistent
with high-dimensional embedding geometry, the top-1 is barely separated from
the top-10, so an `ef`-bounded traversal that lands in the region has little to
distinguish the winner, but it is stated here as an observation, not an
explanation. §11's discipline applies: a result the model does not predict is
recorded as unexplained rather than rationalised.

It matters practically because recall@1 is what a "find the single best match"
workload experiences, and it is the *lower* of the two here. Reporting recall@10
alone would overstate that case.

### 6.4 Build time scales with dimension, not worse

SIFT1M built in 23.8 s at d=128; dbpedia in 269.4 s at d=1536, **11.3× for a
12× dimension increase**. Build cost is dominated by distance evaluations during
neighbour selection, and the parallel builder is not hitting a synchronisation
ceiling as vectors get wider. Ingest is likewise flat in bytes: 640 MB/s here
against SIFT's ~512 MB/s.

## 7. Status

| dataset | fetched | GT | engine run |
|---|---|---|---|
| **sift1m** | yes, digest-verified | fp64, diffed vs published | **yes, §1** |
| **dbpedia-openai-1m** | yes, digest-verified | fp64, recompute-only per §4.2 | **yes, §6** |

Both §4.2 tiers are complete: fetched, checksummed, ground truth computed, and
run end to end against the engine with an exact control.

What is still missing is the comparison. Qdrant has never been started, so every
number here describes one engine in isolation, and `bench/setup.py check` fails
on this host, so none of it is publishable. The remaining §4.2 tiers, GIST1M,
GloVe-100, the BEIR slice, are unfetched.
