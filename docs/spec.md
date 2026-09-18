# strawmann, a Qdrant-wire-compatible vector search engine in Zig

> **strawm*ANN***, a deliberately unfair strawman for approximate nearest neighbour search: no compatibility debt, no distribution, no product concerns. Just the hardware, the graph, and the SIMD units, measured honestly.

**Status:** draft spec, v0.1
**Purpose:** a no-compromise performance reference implementation, driven end-to-end by `qdrant/bfb`.

---

## 1. Why this project exists

This is not a product. It is a **performance oracle**.

The goal is to answer, with numbers rather than intuition: *what does the hardware actually allow for HNSW-based vector search, and how much of that is Qdrant currently leaving on the table?*

To be a useful oracle it must satisfy two constraints simultaneously:

1. **Identical workload.** The same `bfb` invocation must run against Qdrant and against strawmann, with matching semantics, so any delta is attributable to implementation rather than to benchmark methodology.
2. **No architectural debt.** Every layer is free to be rewritten to whatever the measurement says is optimal. No compatibility with existing storage formats, no multi-tenancy, no distribution, no gradual-migration constraints.

The output of the project is not the binary. The output is **a set of validated cost models** ("a `d=768` fp32 distance evaluation over a cold random-access working set costs N ns/vector, here is why, here is the roofline") and a demonstration of how close a real server can get to them.

### Success criteria

- `bfb` runs against strawmann unmodified, for the workload matrix in §4, and produces valid results.
- For each workload, we can state: measured throughput, the modelled hardware ceiling, and the ratio between them.
- Every gap between measured and modelled is either closed or explained by an identified, quantified cause.
- Recall is measured and matched when comparing against Qdrant, QPS is compared **at equal recall**, never at equal `ef`.
- Exact search returns values equal to Qdrant's within a calibrated tolerance (§8.4), for every metric and dimension in the matrix, with the delta distribution published alongside every performance number. The headline claim the project is trying to earn is *identical outputs, N× faster*, not *faster*.

### Non-goals (explicit)

Distribution, sharding, replication, consensus, snapshots, aliases, REST API, TLS, authentication, multi-tenancy, disk-resident (larger-than-RAM) operation, crash-durability guarantees, deletes with space reclamation, sparse vectors (phase 2 at best), multivectors, ColBERT, inference/`Document`/`Image` query variants, RRF/fusion/MMR/formula queries, geo/text/full-text indices.

Anything in this list that `bfb` can be told to emit is handled by returning a clean `UNIMPLEMENTED`, never by silently degrading.

---

## 2. The compatibility surface (derived from bfb, not from the full Qdrant API)

`bfb` links `qdrant-client` from the rust-client `dev` branch (`Cargo.lock` at the pinned bfb: `1.16.1-dev`, `workloads.BFB_CLIENT`); the conformance binary pins `qdrant-client =1.19.0` from crates.io (`conformance/Cargo.toml`); the wire messages are hand-written against the Qdrant 1.19 protos (§12). These are three different version strings and they do not have to agree: the two clients speak the same gRPC surface for the RPCs in §2, and the server-side check that matters is §12's proto pin. It speaks gRPC over **h2c with prior knowledge** (the default `--uri http://localhost:6334` implies no TLS, no ALPN). Compression is not enabled. This collapses the transport problem enormously, see §6.1.

The complete set of RPCs `bfb` can issue, from reading its source:

| gRPC path | Trigger | Required from day one |
|---|---|---|
| `/qdrant.Qdrant/HealthCheck` | **every client construction** (`check_compatibility: true` is the client default) | yes |
| `/qdrant.Collections/CollectionExists` | `--create-if-missing` | yes |
| `/qdrant.Collections/Delete` | collection setup | yes |
| `/qdrant.Collections/Create` | collection setup | yes |
| `/qdrant.Collections/Get` (`collection_info`) | `wait_index` polling loop | yes |
| `/qdrant.Points/Upsert` | upload phase | yes |
| `/qdrant.Points/Query` **batched** → `/qdrant.Points/QueryBatch` | `--search` | yes |
| `/qdrant.Points/Scroll` | `--scroll`, and UUID pre-fetch for `--uuid-query` | phase 2 |
| `/qdrant.Points/SetPayload` | `--set-payload` | phase 2 |
| `/qdrant.Points/CreateFieldIndex` | any payload flag without `--skip-field-indices` | phase 3 |

### Behavioural details that will bite

These are the non-obvious bits that make the difference between "bfb connects" and "bfb completes a run".

- **HealthCheck is on the critical path of every connection.** `qdrant-client` builds a client, calls `HealthCheck`, and compares the returned `version` string against its own major/minor. A mismatch is a warning, not an error, but a *failure* to answer is fatal. Return `{ title: "strawmann", version: "1.18.0" }` and make the version string a config knob so the compatibility check can be silenced for any client version. *As built:* `build.zig -Dqdrant-version=` defaults to `"1.18.0"`, so a stock server answers `1.18.0` to a bfb linking `1.16.1-dev` and to a conformance binary linking `1.19.0`; both mismatches are the client-side warning this paragraph describes, not errors, and the comparison target itself is `qdrant/qdrant:v1.19.0`.
- **`bfb` searches via `QueryBatch`, not `Search`.** This is the modern universal query endpoint. We need `QueryPoints` with `query = Query{ nearest: VectorInput{ dense } }`, plus `filter`, `params` (`hnsw_ef`, `exact`, `indexed_only`, `quantization.{rescore,oversampling}`), `limit`, `with_payload`, `with_vectors`. Everything else in the `Query` oneof → `UNIMPLEMENTED`. Note the response is `QueryBatchResponse { repeated BatchResult result }`, and `time` (seconds, f64) is read by bfb as the *server-side* timing, it feeds the "server_timings" histogram, so it must be measured honestly at the RPC boundary.
- **`wait_index` drives indexing semantics.** After upload, bfb polls `collection_info` once per second and requires `status == Green` **three consecutive times**. This is a gift: it means we do not need concurrent-with-ingest index construction. We can accept upserts into a flat unindexed buffer, report `Yellow`, run a fully parallel bulk HNSW build, then flip to `Green`. Bulk build is both simpler and considerably faster than incremental insert. `--skip-wait-index` must therefore not be used in comparison runs; if it is, we must report `Green` only when actually indexed or the comparison is meaningless.
- **`optimizer_status` must be present and `Ok`** in `CollectionInfo`, and `points_count` / `indexed_vectors_count` should be truthful, several bfb code paths and human eyeballs read them.
- **Dense vectors on the wire.** `repeated float data = 1` inside `DenseVector` is packed in proto3 → arrives as one length-delimited little-endian `f32` run. On x86-64 this is byte-identical to our in-memory layout, so decoding a vector should be a bounds check plus a copy into aligned storage, never a per-element loop. Beware alignment: the payload is not guaranteed 4-byte aligned in the frame buffer.
- **Point IDs** may be `num` (u64) or `uuid` (string), `--uuids` switches. Both must map to a dense internal `u32` offset.
- **`--max-id`** means upserts overwrite existing IDs. The ID map must handle update-in-place.
- **`wait` on upsert.** `--wait-on-upsert` sets `wait: true`. Honour it as "visible to subsequent reads", not as "fsynced".
- **Errors.** bfb has `--retry` and `--ignore-errors`; do not rely on them. Return proper gRPC status codes in trailers, with `grpc-message` set, so failures are legible rather than showing up as timeouts.

*As built* (`src/api/handlers.zig`, `src/proto/messages.zig`; each has an e2e test):

- `QueryBatch` with `limit + offset > 4096` (`Workspace.max_limit`, the preallocated result heap) is `INVALID_ARGUMENT`, not a silent clamp; a response that would not fit the per-worker response buffer is `RESOURCE_EXHAUSTED` naming the remedy.
- `Scroll` `limit` of 0 or above `u32` is `INVALID_ARGUMENT`; absent is 10 (Qdrant's default); the page is capped at 4096.
- An empty `Filter{}` is a filter with no conditions and is accepted as no filter (Qdrant's reading). `must`/`must_not`/`should` conditions of `keyword`, `integer`, `boolean`, `keywords` and `integers` matches are evaluated (since 2026-09-03, `core/payload.zig`); range, geo, full-text, `has_id`, `is_empty`/`is_null`, nested and `min_should` conditions are `UNIMPLEMENTED` naming the construct.
- A NaN or infinite vector component is `INVALID_ARGUMENT` on upsert and on query; a wrong dimension likewise.
- An empty collection name is `INVALID_ARGUMENT` at create.
- UUID point ids are accepted in the four forms `Uuid::parse_str` takes (canonical hyphenated, 32-hex simple, and either wrapped as `urn:uuid:` or `{}`), stored as `u128`, and always returned canonical.
- `SearchParams.indexed_only = true` is `UNIMPLEMENTED` by name: honouring it would mean skipping the unindexed tail, which this engine cannot do.
- `/qdrant.Points/Search`, `/Query` and `/SearchBatch` are answered as legacy aliases of `QueryBatch` (`UNIMPLEMENTED`, message naming the implemented endpoint), not as §1 non-goals.

### Deliberate semantic gaps to document

Any of these silently diverging would poison the comparison, so each must be a loud, documented flag in the results table:

- ~~payload storage/filtering absent in phase 1 (run with `--skip-field-indices` and no `-k`)~~. Built 2026-09-03 (`core/payload.zig`): the payload is stored as the wire bytes of its map entries, `CreateFieldIndex` builds keyword and integer posting lists, and a `must`/`must_not`/`should` filter of `keyword`/`integer`/`boolean`/`keywords`/`integers` matches is evaluated on every search path. Range, geo, full-text, `has_id`, `is_empty`/`is_null` and nested conditions are refused by name. W12 asks for its index on both engines,
- single segment vs Qdrant's multi-segment (`--segments 1` on the Qdrant side for the first comparisons),
- no on-disk **payload** (`--on-disk-payload false`); vector placement is no longer a gap: `VectorParams.memory` is honoured for the vector arena in all three of Qdrant's values (`Pinned`/`Cached`/`Cold`, see `core/storage.zig`), while the HNSW graph and quantized codes stay pinned and refuse the field by name. Note the *defaults* still differ: unstated is `Pinned` here and `Cached` on Qdrant, so a run that does not set it is comparing two residencies,
- `replication_factor` / `shards` accepted and ignored, only value `1` supported.

---

## 3. Data model

Deliberately minimal.

- **Collection**: name → one dense vector space. Named vectors: support the empty-name default plus N named spaces (bfb's `--vectors-per-point` > 1 needs this), but optimise the single-space case.
- **Point**: external ID (u64 or u128 UUID) → internal `u32` offset, assigned densely and monotonically. Internal offsets are the currency of the entire engine; nothing below the API layer sees external IDs.
- **Vector storage**: one contiguous, aligned array per space.
- **Payload**: opaque, append-only blob per point, phase 2. *As built:* the map entries as they arrived on the wire, framed, in a chunk arena that never moves; a `u64` slot per point; overwrites append and repoint, so "no compaction" applies to payloads too. The keyword/integer index is a posting list per value whose entries are hints re-checked against the blob, never truth (`core/payload.zig`).
- **Deletes**: tombstone bitmap only, no compaction.

Capacity is preallocated: the collection is created with a max point count derived from the first upsert rate or a config knob, `fallocate`d, and never grown mid-run. Growth policy is a benchmark artefact we choose not to pay for.

---

## 4. Workloads, datasets, and ground truth

Every entry runs against both engines on the same host, same pinning, in the same session. Invocations are re-derived against bfb `dev` at the pin recorded in §4.1; `docs/workloads.md` carries the full derivation and the per-flag reasoning.

**Flags common to every row**, and not repeated below: `--json results/WNN.json` (the unified results document of #150, parse this, never stdout), `--jsonl-searches` for per-request timings, `--retry 0` so transport errors surface instead of being retried into the average, `--timeout 30`, `--p9 3`. Every row also records `--datatype`, the `--memory-*` placements, and `--hnsw-inline-storage`, because each changes what Qdrant is doing and therefore what the comparison means.

| ID | Purpose | bfb invocation (abridged) |
|---|---|---|
| W0 | d=4 floor | `-n 1M -d 4 --skip-wait-index --search -p 1`, tiny vectors take the distance out; what is left is graph traversal, not RPC overhead (findings 28) |
| W1 | ingest throughput | `--fbin base.fbin -n 1M -d 768 -b 100 -t 8 -p 8` |
| W2 | index build time | W1 without `--skip-wait-index`; time-to-Green is the measurement |
| W3 | search, fp32, single query | `--skip-setup --search --search-limit 10 --search-hnsw-ef 128 -p 1` |
| W4 | search, fp32, saturating | W3 with `-p 64 -t 16 -c 8`; **plus a fixed-rate arm** `--rps R` at R ∈ {0.5, 0.7, 0.9} × measured saturation |
| W5 | search batched | W3 with `--search-batch-size 16` |
| W6 | quantized: scalar | W3 with `--quantization scalar --quantization-rescore true` |
| W7 | quantized: binary + oversampling | W3 with `--quantization binary --quantization-oversampling 4 --quantization-rescore true` |
| W8 | quantized: PQ | W3 with `--quantization product-x16` |
| W9 | exact / brute force | W3 with `--search-exact` |
| W10 | recall control | `--search-hnsw-ef` ∈ {32,64,128,256,512}, **latency only**; recall at the same `ef` comes from `conformance/relevance` (§4.1) |
| W11 | mixed read/write | upload and search concurrently: W1 and W3 as two processes against one collection |
| W12 | filtered search (phase 3) | `-k 100` at upload; `W12-sel1` and `W12-sel10` search it |
| W13 | scroll / pagination | `bfb scroll --file scroll.yaml -n 50k -p 8 --search-limit 10`, at `mode` ∈ {`scroll`, `sequential`, `sample`}; unfiltered arm is a two-engine comparison, filtered arm is Qdrant-only |

**Four changes from the pre-`dev` table**, each explained in `docs/workloads.md`:

- **W10 no longer uses `--search-quality`.** That flag scores an engine against its own exact search, which §4.1 rules out. bfb sweeps `ef` for latency; the harness supplies recall at matched `ef`. The two are joined on `ef`, never reported from one source.
- **W4 gains an open-loop arm.** `--parallel N` is closed loop: a stalled server stops receiving requests, so the latencies it did not serve never appear and the tail is understated. `--rps` sends at a fixed rate regardless, which is what makes a p99 comparable between two engines of different throughput. The saturating arm still answers "what is peak QPS"; the fixed-rate arms answer "what is the latency at a load both engines can serve", and only the second belongs next to a latency claim.
- **`--skip-upload` becomes `--skip-setup`** for the search-only rows, which also skips collection creation and the index wait rather than leaving them to be re-derived per row.
- **W13 is new.** `dev` promoted scroll to a sub-command with its own config and three traversal modes (#151), and `/qdrant.Points/Scroll` was already in §2's table as phase 2. It is the only workload that exercises storage and the id map with no vector comparison at all, which makes it the one place the §5 memory model is testable without the distance kernel in the way.

**ISA axis:** W3, W6, W7, W8 and W9 are each run against every forced-ISA build in §7.5. W9 (`--search-exact`) is the most SIMD-revealing full-stack workload, a linear scan with perfect hardware prefetching, so the memory system stops hiding the kernel, and it doubles as the recall ground truth.

### 4.1 bfb can measure relevance, and we still don't let it

**The reference branch is `dev`, not `master`**, pinned per §8.9 alongside the Qdrant version. The pin is `workloads.BFB_PIN`, and `check_bfb_pin` refuses a run whose checkout is not on it: currently `qdrant/bfb` `dev` at `fc6632e5` (#176), which replaced `0c1aafee` (#172, the `--rps` reaping fix of findings 32) on 2026-09-08 (additively, in that the three intervening commits add a serverless mode in new files and leave the single-collection path unchanged) which had itself replaced `8b6dec0a` ("Fix jsonl read bottleneck", 2026-08-13) on 2026-08-27. Earlier drafts of this section were written against `master` and asserted that bfb could not measure relevance at all. That is no longer true, and the reasoning below replaces it.

What `dev` provides, from `feat: measure search accuracy against reference dataset query sets` (#147, 2026-07-08):

1. **Query vectors can come from a dataset's held-out query set.** A YAML search request with `source: { type: dataset }` draws from the dataset's query set rather than generating uniform noise. The old behaviour, `random_dense_vector` unconditionally, with `--fbin` supplying only the base set, is still what you get *without* a dataset source, so it remains the default and remains a trap.
2. **Recall is measured against the dataset's ground truth.** `recall_against_ground_truth` computes `|returned ∩ expected[:k]| / k`, matching vector-db-benchmark, and reports it under `--- Precision ---`. This is a shared reference, so two engines scored this way are comparable to each other, unlike `--search-quality`, which still exists and still compares each engine against *its own* exact search.

Three properties of that implementation decide how we use it:

- **Recall is a strict id-set intersection with no tie handling.** §8.6 requires ε-aware recall: a returned id counts if it is in the truth set **or** its score ties the k-th truth score within ε. bfb does not do this, so an engine whose tie-breaking differs from whoever computed the shipped ground truth loses recall it has not lost. This is measured, not hypothetical, on SIFT1M, 5443 of 10000 queries return the same 100 neighbours in a different order, and every one of the 111 id disagreements against the published ground truth sits at rank 99 with a score exactly equal to the k-th. The effect scales inversely with k and with the corpus's duplicate rate.
- **Ground truth exists only for `h5`, `tar` and `sparse` sources.** `npy`, `parquet` and partitioned sources return `dataset has no ground truth`. The headline tier (§4.2) is parquet, so bfb can ingest its corpus but cannot score it.
- **`fbin` is not a dataset source.** `--fbin` is a separate legacy upload path with no associated query set or ground truth. Our converted artefacts are therefore invisible to the accuracy path; using it would mean maintaining a second copy of every dataset in `h5`, whose ground truth is in turn somebody else's recomputation rather than ours (§4.3).

Consequences for this project:

- **Split responsibilities cleanly, as before, but for these reasons rather than the old ones:** bfb is the *load generator*, throughput, latency, tail behaviour, concurrency. The `conformance/` harness (§8.5), which already speaks `qdrant-client` to both engines, is the *relevance generator*, held-out queries, our fp64 ground truth, ε-aware recall and nDCG. Never let one report the other's numbers.
- **There is nothing left to contribute upstream.** Earlier drafts proposed adding `--query-fbin`; #147 supplies the capability in a different shape and the item is closed.
- **Maintaining `h5` copies is optional and currently declined.** The only thing it buys is bfb recall numbers directly comparable to published vector-db-benchmark runs. It costs a second copy of every dataset and scores against a third party's ground truth. Revisit only if a published-baseline comparison becomes a goal.
- **Ingest bfb's results through the unified JSON document** (`feat: Unified --json results document`, #150), not by scraping stdout.
- Random vectors remain acceptable for W0-W2 (plumbing and ingest) and nothing else. Uniform high-dimensional noise has no cluster structure, so graph shape, recall curves, and cache behaviour are all unrepresentative. Note this is still bfb's default when no dataset source is configured.

W0-W13 have been re-derived against this branch and the derivation is `docs/workloads.md`, which also tracks the other `dev` changes touching the compatibility surface. Notably `--shard-key` on search (#159), which is sharding and a §1 non-goal, and the `scroll` sub-command (#151), which reaches `/qdrant.Points/Scroll`. Scroll **is** in §2's table, deferred to phase 2, and now has a workload of its own (W13).

### 4.2 The dataset set

Static, versioned, checksummed. Two tiers: a small one for CI iteration and a realistic one for headline numbers, plus a deliberately awkward one.

| dataset | N | dim | metric | queries | GT | role |
|---|---|---|---|---|---|---|
| **SIFT1M** | 1M | 128 | L2 | 10k | shipped (`ivecs`) | CI tier, fast, universally comparable, but SIFT descriptors are not embeddings |
| GIST1M | 1M | 960 | L2 | 1k | shipped | high-dim stress |
| GloVe-100 | 1.18M | 100 | angular | 10k | ann-benchmarks | angular/cosine path, well-known baselines |
| **dbpedia-openai-1M** | 1M | 1536 | cosine | held-out split | recompute | **headline tier**, the dimensionality Qdrant users actually run |
| MS MARCO / BEIR slice, fixed embedding model | ~1M | 768 | cosine | shipped + **qrels** | recompute | the only source of *semantic* relevance judgements (§4.4) |
| Yandex Text2Image-10M | 10M | 200 | IP | 100k | shipped | **cross-modal: query distribution ≠ base distribution.** OOD queries are a known hard case for HNSW and a place where implementations diverge |

The embedding model used to produce any generated dataset is pinned by version and checked into the bench config. A dataset regenerated with a different model is a different dataset.

Three more are pinned in the descriptor alongside them, and are deliberately *not* tiers of this table: `dbpedia-openai-100K-1536-angular` (100k × 1536, the headline dimensionality at a sixteenth of the download, for iterating), and the two filtered sets `laion-small-clip` (100k × 512, a float range condition per query) and `h-and-m-2048-angular-filters` (105k × 2048, a 24-field product payload). W12 filters bfb-generated keywords over SIFT; those two are what a real payload distribution looks like. Their shipped neighbours are computed *under each query's condition*, so they are filtered ground truth and cannot stand in for §4.3's unfiltered k=100, and until a `.npy` reader exists they drive load through bfb only, not the relevance path. `docs/datasets.md` holds the details.

### 4.3 Ground truth: provenance, format, alignment

- **Canonical format:** `fbin`/`ibin` (big-ann-benchmarks convention), since bfb already reads `fbin`. `fvecs`/`ivecs` inputs are converted once at ingest, not at benchmark time. Ground truth is `(nq, k)` int32 neighbour IDs plus `(nq, k)` float32 distances, `k = 100`.
- **Recompute ground truth in fp64 ourselves and diff it against the published file.** Several published GTs were computed in fp32 and have genuine disagreements near the k-th boundary. Use ours, publish the diff once, and note that a disagreement with the shipped GT is expected rather than alarming.
- **ID alignment is a hard requirement.** Ground truth indexes into base-file row order, so point ID must equal row index. bfb assigns sequential IDs from `--offset`, so relevance runs use `--offset 0`, no `--max-id`, no `--uuids`. Any run violating this is rejected by the harness rather than silently producing meaningless recall.
- **Queries are held out.** Never sampled from the base set, using base vectors as queries makes top-1 trivially self-matching and inflates recall across the board.
- **Boundary ties.** Count a returned point as correct if its distance is within ε of the k-th ground-truth neighbour's distance, per the ann-benchmarks convention. Strict ID matching understates recall on any dataset containing duplicate or near-duplicate vectors, which is most real ones.
- **Subsetting invalidates GT.** Ground truth is cached per `(dataset, subset size, metric)` and recomputed when any of those change. Checksums of base, query, and GT files are recorded in every result row.

### 4.4 Relevance: two different questions, both worth answering

| | geometric relevance | semantic relevance |
|---|---|---|
| question | did the index find the true nearest neighbours? | did the system retrieve the documents a human considers relevant? |
| ground truth | exact k-NN under the metric (fp64 oracle) | qrels / human judgements |
| metrics | **recall@k** (k ∈ {1, 10, 100}), mean relative distance error | **nDCG@10**, MRR |
| what it measures | index quality, the engine's own responsibility | the whole pipeline, of which the engine is one part |

Recall@10 is the primary comparison axis and the thing QPS is normalised against. But it's brittle near ties, so it is always reported alongside **mean relative distance error**, the average ratio between returned and ideal distances, which degrades smoothly and distinguishes "missed the true neighbour by a hair" from "returned something unrelated."

Semantic relevance answers the question Qdrant users actually have, and almost nobody publishes it: *does aggressive quantization cost anything that matters?* A drop from recall@10 = 0.98 to 0.93 may move nDCG@10 by a rounding error, or may not. Measuring both on the BEIR slice turns "binary quantization is 8× faster at 93% recall" into a statement someone can make a decision from. Report both; never substitute one for the other.

---

## 5. First-principles performance model

This section is the spine of the project. Write it before writing the server; validate each number with a microbenchmark; then treat the ratio *measured / modelled* as the primary engineering KPI.

### 5.1 Arithmetic intensity

A dot product between a query held in registers and a document vector streamed from memory:

```
d = 768, fp32
bytes moved  = 768 × 4 = 3072 B
flops        = 2 × 768 = 1536
intensity    = 0.5 flop/byte
```

A modern core does ~100 to 200 GFLOP/s of FMA and sustains ~10 to 15 GB/s of its own memory traffic. The break-even intensity is roughly 10 flop/byte. At **0.5**, we are memory bound by a factor of ~20 to 30. 

**Consequence: the entire optimisation programme is about bytes moved and memory-level parallelism, not about FLOPs.** SIMD work on the distance kernel matters only insofar as it stops the ALU from becoming a secondary bottleneck. Anyone optimising the dot product loop before the memory path is optimising the wrong thing, and this project should be able to prove that with a counter.

### 5.2 The real limiter: outstanding misses per core

HNSW graph traversal is a dependent chain, you cannot know which vector to fetch next until the current expansion has been scored. Within one expansion of a node with `M` neighbours, however, the `M` neighbour vector fetches are independent.

A core can sustain roughly 10 to 16 outstanding line fills (LFB/MSHR-limited). With ~85 to 100 ns of DRAM latency:

```
per-core random-access throughput ≈ (12 lines × 64 B) / 90 ns ≈ 8.5 GB/s
```

Modelled cost per distance evaluation, cold, `d=768`:

| encoding | bytes/vector | cache lines | modelled ns/vector | notes |
|---|---|---|---|---|
| fp32 | 3072 | 48 | ~360 | 4 LFB-fulls; bandwidth bound |
| fp16 | 1536 | 24 | ~180 | |
| int8 (SQ) | 768 | 12 | ~90 | exactly saturates the fill buffers |
| PQ x16 | 192 | 3 | ~35 | needs ≥4 concurrent fetches to stay bandwidth-bound |
| binary | 96 | 2 | ~15 to 90 | **latency bound** unless several vectors are in flight |

The last row is the important one. Below ~12 cache lines per vector, a single dependent fetch no longer fills the memory pipeline, and measured cost collapses to raw DRAM latency unless we explicitly issue neighbour prefetches. **Software prefetch of the neighbour list's vectors, issued before scoring the current candidate, is the single highest-leverage optimisation in the search path**, and it only becomes visible once quantization has shrunk vectors below the LFB threshold.

### 5.3 Per-query cost

Let `Nd` = distance evaluations per query (to be measured; expect ~1500 to 4000 for 1M points at `ef=128, M=16`).

```
query_latency ≈ Nd × ns_per_vector + graph_overhead + heap/visited overhead
```

At `Nd = 2500`:

| encoding | modelled latency | modelled QPS/core |
|---|---|---|
| fp32 | ~900 µs | ~1.1k |
| int8 | ~225 µs | ~4.4k |
| binary + rescore top-100 fp32 | ~37 µs + ~36 µs | ~13k |

Note how, in the binary row, **rescoring becomes the dominant term**. That immediately makes oversampling factor a first-class tuning parameter and suggests rescoring should itself run against a cheaper-than-fp32 representation.

Machine-wide bandwidth ceiling as a cross-check: at 2500 × 3072 B = 7.7 MB/query, a 300 GB/s socket pair caps fp32 search at ~39k QPS regardless of core count. If measured aggregate QPS approaches that, the bottleneck is settled and only quantization can move it.

### 5.4 The working-set / hierarchy argument

For 1M × 768:

| encoding | dataset size | fits in |
|---|---|---|
| fp32 | 3.07 GB | DRAM only |
| int8 | 768 MB | DRAM |
| PQ x16 | 192 MB | large L3 (partially) |
| binary | 96 MB | L3 on many server parts |

Quantization's real win is not fewer ALU ops and not even fewer bytes, it is **moving the working set up a level of the memory hierarchy**. A binary index resident in L3 has ~15 ns access latency instead of ~90 ns. The model must predict a discontinuity, not a smooth curve, as the quantized set crosses the LLC boundary. Measuring that discontinuity precisely is one of the more valuable outputs of this project.

### 5.5 TLB, and why "everything in memory via mmap" needs care

3 GB of vectors with 4 KiB pages is 786k pages. A core's dTLB holds ~1.5 to 3k entries across levels. **Every random vector access is a TLB miss plus a page walk**, adding tens of nanoseconds and consuming memory bandwidth for the walk itself. With 2 MiB pages the same data is 1536 entries and stays resident.

The complication: `MADV_HUGEPAGE` on a **file-backed** mmap of an ordinary filesystem does not generally give 2 MiB mappings. THP is an anonymous-memory feature; file-backed large folios are filesystem- and kernel-dependent and cannot be assumed.

Design decision:

- **Default mode ("resident"):** allocate an anonymous `MAP_HUGETLB` (or `MADV_HUGEPAGE`) region and populate it with `pread` from the flat file at startup. Satisfies "everything loaded in memory by default", gives 2 MiB pages, and startup cost is sequential I/O at full disk bandwidth.
- **Comparison mode ("mmap"):** true `mmap` of the file with `MAP_POPULATE`, 4 KiB pages. This is the Qdrant-like path and exists specifically so we can **measure the TLB penalty as a first-class result**.

Expect this delta to be one of the more actionable findings for the Qdrant side. Quantify it with `dtlb_load_misses.walk_active` and `perf stat -e dTLB-load-misses`.

### 5.6 What the model says about SIMD

Falling out of §5.1 to §5.4: **vector width should matter least exactly where people spend the most effort on it** (cold fp32 distance) and most where the working set has been shrunk into cache (binary, PQ, and the rescore stage). That is a falsifiable prediction, and §6.6 and §7.5 exist to test it rather than assume it. If it holds, it reorders the optimisation priorities for any HNSW implementation; if it fails, the failure locates the model's error.

### 5.7 Metrics the engine must expose

Not optional; these *are* the deliverable. Per query, aggregated per run:

- distance evaluations, split by encoding stage (quantized / rescore)
- bytes touched (derived) and `% of modelled roofline`
- nodes visited, hops, per-level breakdown
- visited-set operations, heap push/pop counts
- cycles in: protobuf decode, HTTP/2 framing, index traversal, distance kernel, response encode
- recall@k when ground truth is loaded

Exposed via a side channel (a plain HTTP endpoint on a second port, or a shared-memory ring read by the harness) so that collecting them never perturbs the gRPC path.

**Not implemented.** There is no `src/obs/`, no second port and no shared-memory ring. Distance evaluations, hops and per-stage cycle splits are not exported by the engine.

What exists is measured from outside it, which covers the cost side of the list above and none of the algorithmic side:

- `bfb --json` per row, and the server's connection/request counters at shutdown (`main.zig`, `stats:` line).
- `procstat.py` samples the server through `/proc` and cgroup v2: CPU, peak RSS and its anon/file split, minor and major faults, block I/O, run-queue wait, voluntary and involuntary context switches, migrations, time blocked in the block layer (`delayacct_blkio_ticks`), and PSI stall time for the engine's own cgroup.
- `perfstat.py` attaches `perf stat` to the engine for the length of a row (§7.3), giving IPC, effective frequency, branch misses, demand fills from DRAM and dTLB walks. Per query, which is the only form in which two engines' hardware counters compare.
- `bench/micro/perfctr.zig` reads the same kinds of counter in-process for the distance kernels.

---

## 6. Architecture

```
      ┌──────────────────────────────────────────────┐
      │ N I/O threads, SO_REUSEPORT, own epoll each   │
      │   h2c framing · HPACK · gRPC framing          │
      │   protobuf decode (zero-copy where possible)  │
      └───────────────┬──────────────────────────────┘
                      │ MPSC submit
      ┌───────────────▼──────────────────────────────┐
      │ Query execution pool, one core per query      │
      │   planner → index scan → rescore → top-k      │
      └───────────────┬──────────────────────────────┘
                      │ MPSC complete + eventfd
      ┌───────────────▼──────────────────────────────┐
      │ Storage: vectors · graph · quantized · ids    │
      │ mmap / hugetlb, immutable during search       │
      └──────────────────────────────────────────────┘
```

### 6.1 Transport

**h2c with prior knowledge, no TLS, no compression, server-side only.** No ALPN, no upgrade dance, no push, no priorities. This is a small enough subset that a hand-written HTTP/2 implementation is realistic, roughly: connection preface, SETTINGS, HEADERS/CONTINUATION, DATA, WINDOW_UPDATE, RST_STREAM, GOAWAY, PING, plus HPACK.

Staging:

- **M1: nghttp2 via Zig's C interop.** Correct on day one, callback-driven, some allocation. Unblocks everything downstream.
- **M6: hand-rolled `h2`,** if and only if profiling shows framing/HPACK on the critical path. Design it for zero per-request allocation: arena per stream reset on completion, fixed-size header table, HPACK encoding using static-table indices plus literal-without-indexing (always legal, avoids maintaining an encoder dynamic table).

Points to get right regardless of implementation:

- **Flow control.** A 100-point batch of 768-dim fp32 is ~307 KB. The default 64 KiB initial window will stall it into five round trips. Advertise a large `SETTINGS_INITIAL_WINDOW_SIZE` (e.g. 8 MiB) and keep the connection-level window topped up aggressively.
- **gRPC framing** is a 5-byte prefix (compressed-flag + big-endian u32 length) per message, and the status lives in **trailers** (`grpc-status`, `grpc-message`) in a second HEADERS frame with END_STREAM. Tonic sends `te: trailers` and expects trailers even on success.
- **`-c 1` is the bfb default.** One connection means all concurrency is multiplexed over a single TCP stream handled by one I/O thread. Therefore request *execution* must fan out to the worker pool, with completions routed back to the owning I/O thread via an MPSC queue plus `eventfd`. A design where the I/O thread executes the query serialises the default benchmark and would be a self-inflicted wound.
- Writev batching: coalesce all completed responses for a connection into a single `writev` per epoll wakeup.

### 6.2 Protobuf

Hand-written codecs for the ~30 messages actually needed. No generic reflection, no code generator dependency, no `unknown_fields` retention. Zig's comptime makes this pleasant: describe each message as a comptime field table and generate encode/decode functions from it, so the fast path is a switch over field tags with no indirection.

Fast-path requirements:

- `DenseVector.data` decode: bounds-check, then one memcpy into the destination slot in the vector arena. Never element-wise. Handle unaligned source with unaligned loads.
- Upsert decode writes **directly into final storage**, the point's vector lands in `vectors.bin`'s mapped arena, not in an intermediate `Vec<f32>`.
- Response encode for `QueryBatchResponse` writes directly into the output frame buffer; pre-size it from `limit × batch_size`.
- Strict-but-cheap validation: unknown field → skip by wire type; unsupported oneof variant → `UNIMPLEMENTED`.

### 6.3 Threading and placement

- One I/O thread per core in the I/O set, `SO_REUSEPORT` for connection-level distribution.
- Query execution: fixed worker pool, one query per worker (no intra-query parallelism except for exact search, which splits the scan by range). A `QueryBatch` is one *request* but n queries: it is fanned out across the pool as sub-requests on the same queue and completed by whichever worker finishes last (`handlers.BatchJob`, findings 27); a batch of one, or one too large for the room its stream has, runs on the dequeuing worker.
- Explicit CPU pinning for every thread, configurable, defaulting to a NUMA-local layout.
- NUMA: phase 1 assumes single socket and *says so*. Phase 7 investigates per-node replication of the vector arena, worth measuring, since read-only replication trades memory for the elimination of cross-socket traffic, and vector search is the ideal candidate for it.
- No allocation on the query path. Per-worker preallocated: visited set, candidate heap, result heap, rescore buffer, distance scratch.

### 6.4 Storage layout

One directory per collection, flat files, no directory of blocks, no B-tree, no LSM.

As designed:

```
collection/
  meta.json           config, dim, distance, quant params, counts, format version
  vectors.<name>.bin  header + aligned vector arena
  graph.bin           header + level-0 neighbours + upper levels
  quant.<name>.bin    quantized codes (+ codebooks / scales in header)
  ids.bin             internal u32 → external id
  idmap.bin           external id → internal u32 (open-addressed table, rebuildable)
  payload.bin         append-only blobs      (phase 2)
  payload.idx         internal u32 → (offset,len)  (phase 2)
  deleted.bits        tombstone bitmap
```

As built (`src/core/persist.zig`, `storage.zig`, header `format_version = 3`), four files, one directory per collection:

```
collection/
  vectors.bin         header (dim, count, capacity, stride, metric, normalised flag,
                      datatype, seed, graph checksum, hnsw_m, hnsw_ef_construct) + arena
  ids.bin             internal u32 → external id, 17-byte records (tag + u128 LE)
  deleted.bits        tombstone bitmap
  graph.bin           header (m in `dim`, indexed count) + level-0 + upper levels
```

Plus, since 2026-09-03, `payload.bin` (one `[u32 len][framed blob]` record per point, so no `payload.idx` offset table is needed) and `payload.schema` (the field indexes' names and kinds; the postings are rebuilt from the blobs on load, as the id map is from `ids.bin`).

No `meta.json` (the vectors header carries the config), no `idmap.bin` (rebuilt from `ids.bin` on load), no `quant.*.bin`: `save` refuses a quantized collection (`error.QuantizedNotPersisted`), and `load` refuses a mapped placement (`error.PlacementNotSupportedOnLoad`). Nothing in the server calls either; a live `cached`/`cold` arena is a different file, `<data-dir>/<name>.vectors.bin`, with the same header.

Rules:

- Every file starts with a 4 KiB header: magic, format version, dim, count, capacity, distance, flags, and a CRC of the header only. No CRC over the data, we are not building for durability, and hashing 3 GB at startup would dominate load time.
- **Row stride is padded to 64 B** so no vector straddles a cache line boundary it doesn't need to. For `d=768` fp32 stride is already 3072; for odd dims the padding is real and the padding bytes are zeroed (so they contribute nothing to dot/L2).
- The arena is preallocated with `fallocate` at collection creation. Point *n* lives at `base + n × stride`, computed, never looked up.
- Recovery is `open` + `mmap`/`pread`. There is no parse step, no WAL replay, no index rebuild. Startup time for a 3 GB collection should be bounded by sequential read bandwidth.
- Cosine is normalised at ingest and stored normalised, so the search path only ever runs dot product. Record this in the header.

### 6.5 HNSW

**Build.** Bulk, post-ingest, all cores. Triggered when a collection transitions out of the ingest phase (status `Yellow` → build → `Green`, which is exactly what bfb's poll loop wants).

- Level assignment with the standard exponential decay, `mL = 1/ln(M)`.
- Parallel insert with fine-grained per-node locks (or CAS on neighbour arrays). Measure lock contention explicitly; if it shows, partition-then-merge is the fallback.
- Neighbour selection: heuristic pruning (Qdrant/hnswlib `select_neighbors_heuristic`), since the plain top-M variant produces measurably worse graphs on real data.
- Build is a legitimate benchmark target in its own right (W2), Qdrant's build times are a real user pain point and a place where a first-principles implementation may show a large gap.

**Layout.** This is where a from-scratch implementation can differ most from an incrementally-grown one.

- **Level 0**: a single flat array, stride `M0 = 2×M` `u32`s, indexed by internal offset. For `M=16`: 128 B/node = exactly two cache lines, 128 MB for 1M points. No per-node allocation, no pointer chasing, no indirection.
- **Upper levels**: CSR (offsets + neighbours), tiny (~1/M of level 0), and hot, pin them and consider keeping them in a hugepage region separate from level 0.
- **Optional co-location experiment**: interleave each node's neighbour list with its (quantized) vector so one fetch brings both. This trades a larger stride for halving the number of dependent misses; whether it wins is exactly the kind of question this project exists to answer. Build it behind a flag and measure.

**Search.**

- Visited set: **generation-stamped `u32` array**, sized to point count, with a monotonically incrementing per-query epoch. No clearing between queries, no allocation, no hashing. 4 MB per worker at 1M points, measure against a bitmap-plus-dirty-list alternative, which is smaller but requires clearing.
- Candidate and result heaps: flat arrays, `ef`-bounded, branch-light sift. At `ef ≤ 512` the heaps live comfortably in L1.
- **Prefetch discipline** (per §5.2): on expanding a node, first issue `prefetcht0` for the neighbours' vector rows (all of them, since the list is 1 to 2 cache lines and already resident), then score. Tune prefetch distance empirically per encoding.
- Early termination on the standard "worst candidate is worse than current k-th best" condition.
- `exact: true` → brute force with a range-partitioned parallel scan, since it's the recall ground truth and the pure-bandwidth benchmark.

### 6.6 SIMD and the distance kernels

This is a first-class subsystem, not an implementation detail of §6.5. It gets its own directory, its own microbenchmark suite, and its own ISA matrix.

#### 6.6.1 Where SIMD actually matters

§5 argues that cold fp32 search is memory bound by ~25×, which would seem to make vector width irrelevant. That conclusion is only true for **one cell** of the matrix. The picture across encodings:

| encoding | working set (1M×768) | regime | does wider SIMD help? |
|---|---|---|---|
| fp32, cold | 3.07 GB | bandwidth bound | little or nothing, and that's a *prediction to test* |
| fp32, brute force / `--search-exact` | streamed, perfectly prefetched | closer to compute bound | yes |
| int8 SQ | 768 MB | mixed | yes, substantially (VNNI) |
| PQ x16 | 192 MB | partially L3-resident | yes (LUT lookup throughput is the kernel) |
| binary | 96 MB, often L3-resident | latency/compute bound | yes, strongly (`vpopcntq`) |
| rescore stage | the walk's candidates, hot | compute bound | yes |

> **Deviation, 2026-08-21.** §6.7 words the rescore input as `limit ×
> oversampling`, and `core/quantized_search.zig` now reads that as a floor:
> with rescore on it rescores `max(limit × oversampling, ef)`, because the walk
> has already scored `ef` candidates and choosing `limit` of them *by the
> quantized score* discards exactly what rescoring exists to correct. Measured
> on dbpedia-openai-100K, that was 0.8062 → 0.9757 recall@10 at `ef` 128. The
> spec sentence has not been changed; this is the implementation departing from
> it deliberately, and the cost row above follows the implementation.

So the SIMD payoff grows exactly as quantization shrinks the working set, the two optimisations are coupled, not independent. A project that measures SIMD only on cold fp32 will conclude SIMD doesn't matter and will be wrong about every other row.

**The AVX-512 delta is itself a diagnostic.** If the cold fp32 kernel shows ~0% improvement from 512-bit while the L3-resident binary kernel shows +40%, the §5 cost model is validated. If cold fp32 *does* improve meaningfully, the model is wrong somewhere (likely we were never bandwidth bound, or 512-bit loads are improving memory-level parallelism by retiring fewer uops per byte), and finding that out is worth more than the speedup.

#### 6.6.2 AVX2 vs AVX-512: the instructions that actually differ

Not a width story. The width doubling is the least interesting part; the ISA additions are where the real gaps are.

| capability | AVX2 path | AVX-512 path | expected gap |
|---|---|---|---|
| **int8 dot** (SQ) | `vpmaddubsw` → `vpmaddwd` → `vpaddd`, 3 instr + saturation care | `vpdpbusd` (AVX512_VNNI), 1 instr | ~3× instruction count; note AVX-VNNI gives 256-bit `vpdpbusd` on Alder Lake+ and Zen 4+, which narrows this a lot, **must be a separate row in the matrix, not folded into "AVX2"** |
| **popcount** (binary) | Muła's nibble-LUT via `vpshufb` + `vpsadbw`, or Harley-Seal for long runs, ~5 to 8 instr/32 B | `vpopcntq` (AVX512_VPOPCNTDQ), 1 instr/64 B | large; the single biggest ISA gap in the whole engine |
| **PQ4 LUT lookup** | `vpshufb`, 32 lookups/instr | `vpshufb` at 512 bits, 64 lookups/instr; `vpermi2b` (VBMI) for 128-entry tables | ~2×, more with VBMI |
| **fp32 FMA** | `vfmadd231ps` ymm | same, zmm | ~2× in theory, subject to frequency licensing (§6.6.4) |
| **tail handling** (dims not a multiple of the lane count) | scalar epilogue or masked load via `vmaskmovps` | native mask registers `k0-k7`, zero-cost predication | matters for 384/768/1536 (all fine) and for odd real-world dims |
| **register pressure** | 16 ymm | 32 zmm | 8 accumulators to cover 4-cycle FMA latency × 2 ports fits comfortably in AVX-512, spills under AVX2 at deep unroll |
| **fp16** | `vcvtph2ps` (F16C) convert-then-compute | AVX512-FP16 native, or `vdpbf16ps` for bf16 | relevant to bfb's `--datatype Float16` |
| **gather** | `vgatherdps` | `vgatherdps`/`vpgatherdd` | **avoid both.** Gather has historically been slower than scalar loads plus inserts for our access patterns; benchmark it once, record the result, move on |
| **ternary logic** | AND+XOR sequences | `vpternlogd`, arbitrary 3-input boolean in 1 instr | small but free win in binary/masked paths |

#### 6.6.3 Kernel inventory

Each kernel is written once as a comptime-generic Zig function parameterised by lane count and element type, then instantiated per ISA target. Zig's comptime is a genuine advantage here: one source of truth, N machine-code variants, no macro soup and no hand-maintained intrinsic duplicates.

```
dist/
  dot_f32.zig        @Vector(L, f32), 8 accumulators, @reduce(.Add)
  l2_f32.zig         fused sub+fma, or |a|²+|b|²-2ab on normalised data
  dot_f16.zig        F16C convert vs AVX512-FP16 native
  dot_i8.zig         VNNI / AVX-VNNI / vpmaddubsw fallback, sym + asym
  hamming.zig        vpopcntq / vpshufb-LUT / Harley-Seal
  pq_adc.zig         8-bit scalar LUT; 4-bit vpshufb FastScan
  norm.zig           ingest-time normalisation
  dispatch.zig       cpuid → vtable, selected once at startup
```

Common requirements:

- `@setFloatMode(.optimized)` **inside the kernels only**, so FMA contraction and reassociation are permitted where they're safe and nowhere else. Reassociation changes results; the correctness harness compares against a strict-order reference with an explicit tolerance rather than exact equality.
- ≥8 independent accumulators for f32 (FMA latency 4, two ports → 8 in flight to saturate). Verify the compiler actually kept them in registers by reading the disassembly, this is exactly the kind of thing that silently regresses on a toolchain bump.
- Explicit `@prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data })` on neighbour vectors per §6.5, with the prefetch distance tuned per encoding.
- Every kernel is benchmarked **hot (L1-resident) and cold (random over the full arena)**. The gap between those two numbers is the memory story of §5 made concrete; the hot number is the pure SIMD result.
- Disassembly of each hot loop is checked into `docs/asm/` and diffed by `bench/isa/dump_asm.py --check` (a manual step; no CI or `scripts/check.py` step runs it). A silent change in vectorisation is a regression even if the wall-clock didn't move on the current host.

#### 6.6.4 Frequency licensing, the trap

On Skylake-SP and Cascade Lake, sustained 512-bit FMA work drops core and all-core turbo frequency substantially (the L1/L2 "license" levels). A kernel that is 1.8× faster per cycle can be net *slower* once the whole socket downclocks, and worse, it can slow down unrelated threads on neighbouring cores. Ice Lake-SP reduced the penalty considerably; Sapphire Rapids and Granite Rapids largely removed it for typical mixes. On AMD, Zen 4 executes most 512-bit ops as double-pumped 256-bit (no frequency penalty, roughly 256-bit throughput, but still a real win on instruction count and front-end pressure), while Zen 5 has a full 512-bit datapath and behaves very differently.

Two consequences:

1. **"Is AVX-512 faster?" is not a question with one answer.** It is a per-microarchitecture answer, and the matrix in §7.5 must be run per target class. Expect the honest conclusion to be something like *"+45% on Zen 5, +30% on Granite Rapids, −5% on Cascade Lake"*.
2. **Effective frequency must be measured on every run**, or none of the comparisons mean anything. `cycles / ref-cycles` from `perf stat` gives the ratio directly; log it per run and refuse to publish a comparison where the two arms ran at frequencies differing by more than a threshold without saying so.

This has a direct read-across to Qdrant, which ships one binary to a heterogeneous fleet: the right dispatch policy may not be "use the widest ISA available."

#### 6.6.5 Dispatch and build strategy

- **Primary results** come from `-Dcpu=native` builds, one per target machine. This is a benchmark, not a distributed binary; there is no reason to pay for dispatch on the hot path.
- **Forced-ISA builds** for the matrix: `-Dcpu=x86_64_v3` (AVX2+FMA), `-Dcpu=x86_64_v4` (AVX-512F/BW/DQ/VL), plus explicit feature sets for `+avxvnni`, `+avx512vnni`, `+avx512vpopcntdq`, `+avx512vbmi`, `+avx512fp16`. Same source, same everything else, the only variable is the ISA.
- **Runtime dispatch** exists as a secondary path: detect features via `cpuid` at startup, populate a vtable of kernel function pointers once, never branch on ISA inside a loop. Used to (a) confirm dispatch overhead is nil and (b) model what a shipped binary would actually do.
- A per-kernel override knob so a run can force, say, AVX2 hamming with AVX-512 fp32 rescore. Mixed configurations are likely optimal on the parts with frequency licensing, and finding that out is a project deliverable.

### 6.7 Quantization

Match bfb's flags, minus the Qdrant-proprietary `turbo*` variants (return a clear error; document the exclusion so nobody accidentally compares against them).

| mode | encoding | distance | notes |
|---|---|---|---|
| `scalar` | int8, global quantile bounds (bfb uses 0.99) | VNNI `vpdpbusd` where available, else `vpmaddubsw`+`vpmaddwd` | symmetric (query also quantized, the served path) and asymmetric variants, both measured. Not Qdrant 1.19's quantizer: 256 levels vs 127, order-statistic quantile vs effectively min/max (`src/quant/scalar.zig`); a recall-matched SQ8 row is not like-for-like |
| `binary` | 1 bit/dim, sign-based | XOR + `@popCount` over `@Vector(u64)` | fastest; needs oversampling + rescore. With `rescore = false` the score on the wire is Qdrant's sign-dot scale, `dim − 2·hamming` (before metric postprocess), not `−hamming` |
| `product-x{4,8,16,32,64}` | PQ, 8 bit/subquantizer | LUT ADC, scalar gather baseline | k-means codebooks trained on a sample |
| PQ4 / FastScan | 4 bit, `vpshufb`-based LUT | in-register table lookup | stretch goal; the version that actually competes |

Rescoring: quantized search produces `limit × oversampling` candidates, rescored with the full-precision (or next-tier) representation. Given §5.3, the rescore stage is often the dominant cost at low `ef`, so measure the **combined** curve of (recall, latency) across `(ef, oversampling, rescore-encoding)` rather than tuning stages independently.

`always_ram` is always true for us. `--quantization-in-ram false` is accepted and ignored, loudly.

---

## 7. Measurement methodology

Being fast is easy to claim and hard to demonstrate. The methodology is part of the spec, not an afterthought.

### 7.1 Host discipline

Codify in a `bench/setup.py` that refuses to run if any check fails:

- fixed CPU frequency governor (`performance`), turbo disabled or pinned, C-states restricted
- **effective frequency logged per run** (`cycles / ref-cycles`), since AVX-512 licensing (§6.6.4) makes nominal frequency a lie; the harness records it whether or not the run is an ISA comparison
- microarchitecture recorded by name, not by core count, `Cascade Lake` and `Zen 5` are different experiments, not different samples of one
- SMT explicitly on or off, declared per run, never left ambient
- **measurement profile**, declared per run and carried in the environment hash: `isolated` means `isolcpus` / `nohz_full` on the server cores, and a difference is attributable to the engine; `as-deployed` means the scheduler is left as it ships, and a difference is what a user would see. Neither is a rank and neither is publishable *as the other*. The hash keeps them from sharing a chart. bfb is pinned to a disjoint set in both (ideally a separate host, with the NIC's contribution then measured separately)
- THP setting recorded; hugepage pool reserved for the resident mode
- ASLR, NUMA balancing, and kernel mitigations state recorded in the run metadata
- `/proc/sys/vm/drop_caches` between cold-start runs, and an explicit warm-up phase for steady-state runs

Every result row carries a hash of the environment description. Results from different environments never share a chart.

### 7.2 Layered benchmarks

1. **Hardware baseline.** STREAM-equivalent and pointer-chase latency microbenchmarks written in Zig, plus Intel MLC for cross-validation. Produces the per-core and per-socket bandwidth/latency constants that §5's model consumes. Rerun on every new machine.
2. **Kernel microbenchmarks.** Distance functions, hot and cold, cycles/vector.
3. **Index microbenchmarks.** In-process, no network: `Nd`, latency, recall vs `ef`, for each encoding.
4. **Full-stack.** bfb, the matrix in §4.
5. **Differential.** The same bfb invocation against Qdrant, same host, interleaved (A/B/A/B, not A-then-B, to control for drift).

The gap between (3) and (4) is the transport-and-serialisation tax. Tracking it as an explicit number is one of the more interesting outputs, it is the part of the stack most easily overlooked and, at W0, the part that dominates.

### 7.3 Tooling

- `perf stat` with a fixed event set: `cycles, ref-cycles, instructions, cache-misses, LLC-load-misses, dTLB-load-misses, dtlb_load_misses.walk_active, l1d_pend_miss.pending`, and per-encoding `mem_load_retired.l3_miss`.
- SIMD-specific events: `fp_arith_inst_retired.512b_packed_single` / `.256b_packed_single` to confirm the kernel is executing at the width we think it is (a compiler that quietly split 512-bit ops is a common and invisible failure), plus `core_power.lvl1_turbo_license` / `lvl2_turbo_license` on Intel parts to see licensing directly.
- `llvm-mca` and/or `uica` on the extracted hot loops for a static throughput prediction, compared against measured cycles/vector. A large divergence usually means a spill, a missing FMA contraction, or an unexpected port conflict.
- Top-down analysis (`toplev`) to classify each configuration as front-end / back-end / memory bound, and check the classification against the model's prediction. A configuration the model calls memory-bound that top-down calls back-end-core-bound is a bug in the model or the code, and either way is worth finding.
- `perf c2c` for false sharing once the worker pool exists.
- Flamegraphs from `perf record -g`, with frame pointers kept (`-fno-omit-frame-pointer` equivalent) in bench builds.
- `bfb --json` output plus server-side counters merged into a single row per run, stored in SQLite, with a small script producing the comparison tables.

**Status.** The fixed event set is implemented for full-stack rows in `bench/harness/perfstat.py`, attached to the running engine with `perf stat -p` for the same window as the `/proc` counters, and switched on per run with `workloads.py run ... --perf`. Three departures from the list above, each deliberate:

- **The list is spelled in Intel event names and the reference host is a Zen 5 part.** `dtlb_load_misses.walk_active`, `l1d_pend_miss.pending` and `mem_load_retired.l3_miss` do not exist there, and the part has no L3/DF uncore PMU at all, so LLC traffic has to come from core-side fill events (`ls_dmnd_fills_from_sys.dram_io_all`). Each *role* therefore carries a list of candidate spellings, the first supported one is used, and the row records which. A DRAM-fill count from `LLC-load-misses` and one from a demand-fill event are both "cache misses" and are not the same measurement.
- **Six hardware events, not nine.** The part has six core counters; a seventh event takes the whole group to ~75% enabled, and `perf` then prints values extrapolated from the fraction of the row each counter was on. Nothing here is scaled: below 95% enabled the counter is withheld and the percentage recorded in its place. `cache-misses` and `l1d_pend_miss.pending` lost their slots to `ref-cycles` (§7.5's frequency guard rail) and the dTLB walk count (§5.5's claim, never checked on a running engine).
- **`perf c2c` is not implemented.** It is a sampling workflow with a multi-gigabyte capture, it cannot share a run with published numbers, and on AMD it goes through IBS with coarser classification than the Intel HITM path the bullet assumes. The cheap proxy is implemented instead, as an opt-in second event set (`--perf sharing`): demand fills served from *another core's cache*, which scales with sharing and says whether it is worth going looking. It is a proxy and not a proof. It cannot separate true sharing from false.

### 7.4 Statistics

- Report p50/p95/p99/p99.9 **and** throughput; bfb's `--p9` controls the tail digits.
- Watch for coordinated omission: `--rps` (fixed rate) and `--parallel` (closed loop) measure different things. Use `--rps` for latency claims and closed-loop for saturation throughput. Never quote a closed-loop p99 as a latency result.
- Minimum three interleaved repetitions per cell; report median of medians and the spread. A run whose spread exceeds a threshold is discarded and the environment investigated.
- **Compare at equal recall.** Publish QPS-vs-recall curves; a single QPS number without its recall is meaningless and this project should never emit one.
### 7.5 The ISA matrix

A dedicated benchmark axis, run at both the microbenchmark and full-stack levels.

**Axes:** `{AVX2, AVX2+AVX-VNNI, AVX-512 (F/BW/DQ/VL), AVX-512 + VNNI/VPOPCNTDQ/VBMI}` × `{dot_f32, l2_f32, dot_i8 sym, dot_i8 asym, hamming, pq_adc_8, pq_adc_4}` × `{L1-hot, L3-resident, DRAM-cold}` × `{d = 128, 384, 768, 1536}`.

**Reported per cell:** cycles/vector, ns/vector, instructions/vector, IPC, **effective frequency** (`cycles / ref-cycles`), and uops from the front end. Cycles/vector is the primary number because it isolates the ISA question from the frequency question; ns/vector is what actually matters and must be reported alongside it, never instead of it.

**Then the same matrix end-to-end:** identical source, forced-ISA builds, W3/W6/W7/W8 from §4, QPS at fixed recall. The microbenchmark tells you the kernel got faster; only the full-stack run tells you whether that survived contact with the memory system.

**Cross-microarchitecture:** the matrix is meaningless on one machine. Minimum target set: one pre-Ice-Lake Intel server part (where 512-bit licensing bites), one Sapphire Rapids or later, one Zen 4 (double-pumped), one Zen 5 (full-width). The expected deliverable is a table of *"which ISA to select, per microarchitecture, per kernel"*, which is directly actionable for anyone shipping one binary to a mixed fleet.

**Guard rails:** any AVX-512-vs-AVX2 comparison where effective frequency differs by more than 3% between arms is reported with both frequencies visible. A cycles/vector win presented without its frequency is not a result.

---

## 8. Correctness and differential testing

The performance claim is worthless without this chapter. "strawmann is 3× faster than Qdrant" is a statement about two programs that might be computing different things. The job of `conformance/` is to remove that objection entirely, so the only remaining variable is implementation quality.

The framing: **no performance number is publishable unless the same build, on the same data, has a green conformance row.** This is enforced by the harness, not by convention, the results sink rejects a perf row whose conformance hash doesn't match a passing run.

### 8.1 What can and cannot be identical

Three distinct claims, of decreasing strength. Conflating them is the standard way this kind of comparison goes wrong.

| claim | scope | achievable? |
|---|---|---|
| **Bit-exact equality** | any fp32 kernel | **No.** Qdrant's AVX path uses 4 accumulators over 32 floats/iteration with FMA; ours uses 8 with a different reduction tree. Different summation order → different last-bit results. This is not a defect in either engine. |
| **Value equality within ε** | exact search (`params.exact = true`), all metrics, fp32 | **Yes, and this is the load-bearing claim.** Same inputs, same outputs to within a calibrated tolerance, N× faster. |
| **Statistical equivalence** | ANN search (HNSW), all quantized modes | **Yes, as distributions.** Different graphs (RNG seed, insertion order, neighbour-selection tie-breaks) mean per-query results legitimately differ. The comparable quantity is recall@k against shared ground truth, not the result list. |

Anything promising ID-level equality for HNSW results is promising something false. The spec does not.

### 8.2 The oracle hierarchy

Neither engine is ground truth. When they disagree we need to know *who is wrong*, which requires a third, independent reference:

1. **fp64 exhaustive oracle.** A straightforward, deliberately unoptimised double-precision exhaustive scan (NumPy is fine; correctness here matters far more than speed). Computes exact top-k and exact scores for the static datasets of §4.2, cached to disk per §4.3 so it is paid for once. This is the arbiter, and it is also what produces the ground truth files, the same code serves conformance and relevance.
2. **strawmann's own strict-order fp32 reference.** Scalar, single accumulator, no FMA, no reassociation. Isolates "our SIMD kernel is wrong" from "our algorithm is wrong."
3. **Qdrant.** A peer under test, not an authority.

Concretely: if strawmann and Qdrant disagree beyond ε, compare both to the fp64 oracle. Historically this kind of harness finds real bugs on both sides, and finding one in Qdrant would be a legitimate output of the project rather than an embarrassment.

### 8.3 Qdrant's score semantics, which we must replicate exactly, but not its rounding

From `lib/segment/src/spaces/`. These are the details that produce silent, systematic disagreement if we guess instead of read:

| metric | internal similarity (higher = better) | postprocess applied to the returned score | preprocess at ingest |
|---|---|---|---|
| Dot | `Σ aᵢbᵢ` | identity | none |
| Cosine | `Σ aᵢbᵢ` (i.e. dot on normalised data) | identity | normalise |
| Euclid | **`−Σ(aᵢ−bᵢ)²`** | `abs().sqrt()` | none |
| Manhattan | `−Σ|aᵢ−bᵢ|` | `abs()` | none |

Three traps in there:

- **Euclid ranks on the negated squared distance but returns the square root.** Getting the ranking right and the returned score wrong (or vice versa) is easy and would show up as a conformance failure with a suspiciously structured error distribution.
- **`(a−b)²` is computed directly, not via the `|a|² + |b|² − 2ab` expansion.** The expansion is tempting because it turns L2 into a dot product with precomputed norms, but it suffers catastrophic cancellation exactly where it matters, since the error scales with `|a|²` rather than with the (small) distance between near neighbours. For close neighbours the relative error can be several orders of magnitude worse. If strawmann wants the expansion for speed, it must be a separate, separately-validated mode with its own tolerance, never silently substituted into the conformance path.
- **Cosine normalisation short-circuits.** Qdrant computes `length = Σx²` and, if `length < f32::EPSILON || |length − 1.0| ≤ 1e-6`, **returns the vector untouched**. Note it tests the *squared* length against 1.0 before taking the square root. A vector that is already near-normalised is therefore stored exactly as sent, while one just outside that window is divided through. Replicating the divide but not the short-circuit produces a small population of systematically-off points, the worst kind of bug, because it passes on random data and fails on real normalised embeddings. Also: it is a **division by `length`**, not a multiply by a reciprocal, and definitely not `rsqrtps` (whose ~12-bit approximation would blow ε to ~1e-3).

The conformance module encodes this table as executable assertions, and a change to it is a spec change.

**Semantics are replicated; arithmetic error is not.** The table above is about *what quantity* each metric returns, which similarity is ranked on, what postprocessing is applied, what happens at ingest. Those must match exactly, because a difference there is a systematic disagreement that no tolerance can absorb.

Floating-point *accuracy* is a separate axis, and on that axis strawmann does not chase Qdrant. Measured at d=1536 cosine on the headline tier: strawmann is 1.501e-7 from the fp64 oracle where Qdrant is 1.358e-6, nine times further, and the two therefore disagree by more than the ε calibrated in §8.4. Matching Qdrant there would mean *reproducing its rounding error on purpose*, which would be a strange thing for a project whose stated output is a validated cost model.

So **T1 is graded against the fp64 oracle, not against Qdrant**:

- agree within ε → pass;
- disagree, but strawmann is within ε of the oracle → **pass**, and record that Qdrant is the engine that drifted;
- disagree and strawmann is further from the oracle than Qdrant → fail. That is a real defect and the tier still catches it.

A divergence in the *ids* or the *number* of results is never excused this way: that is a compatibility defect, and no amount of numerical accuracy repairs it.

The earlier rule, fail on any disagreement beyond ε, made Qdrant the definition of correct, so strawmann failed a conformance tier *for being more accurate*. §8.1 already concedes bit-exactness is unachievable, which means equality with Qdrant was never the real requirement; being right was.

### 8.4 Calibrating ε instead of guessing it

Picking `1e-6` because it looks reasonable is unprincipled. Instead, derive the tolerance from measured variance:

1. **Measure Qdrant's own internal spread.** Qdrant dispatches to AVX/SSE/NEON/scalar kernels by runtime feature detection, and these have different accumulator counts and FMA usage, so *Qdrant does not agree with itself across ISAs*. Run the same query set against Qdrant builds/hosts forcing each path, and record the score-delta distribution.
2. **Measure strawmann's spread** across its own ISA matrix (§7.5) the same way.
3. **ε := a small multiple of the larger of the two**, per metric and per dimension, justified in `docs/tolerance.md` with the underlying distributions.

Expected magnitudes, as a sanity check rather than a target: for normalised fp32 dot at d=768, blocked summation error is roughly `O(log d · 2⁻²⁴)`, so an absolute delta around `1e-6` on scores in `[-1, 1]`. Unnormalised dot and Euclid scale with vector magnitude and want a relative tolerance instead.

Reporting rule: conformance reports the **distribution** of `|Δscore|`, max, p99.9, p99, mean, not a pass/fail bit. A run that passes with a max delta 10× worse than yesterday's is a regression, and a pass/fail gate would hide it.

### 8.5 The differ

`conformance/` is a Rust binary using the same `qdrant-client` crate that bfb uses. This is deliberate: one client, one encoder, one decoder, both engines. Any difference observed is then necessarily server-side, not an artefact of how the harness talks to each engine. It reuses bfb's fbin reader so the datasets are literally the same bytes.

It drives both engines through identical operation sequences and compares at five tiers:

**T0, Wire conformance.** Structural comparison of response messages: field presence, ID variant (`num` vs `uuid`), payload round-trip fidelity, `version`/`operation_id` semantics, presence of `time`, and, importantly, **error behaviour**: unsupported operations must produce the same gRPC status code as Qdrant, not a hang or a wrong-shaped success. Compare as proto trees with an explicit ignore-list (timings, versions, commit hashes). This catches "right numbers, wrong shape," which no amount of score comparison will find.

**T1, Exact-search value equality.** `params.exact = true`, fp32, every metric, every dimension in the matrix. Element-wise comparison of returned `(id, score)` pairs against Qdrant and against the fp64 oracle. **This is the tier that licenses the performance claim.**

**T2, Rank agreement under ties.** Equal scores occur constantly (duplicate vectors, quantized encodings, small dims) and their ordering is arbitrary in both engines. Compare results as multisets grouped into score-equivalence classes at width ε, never as ordered lists. Where a soft measure is needed, report Kendall's τ and rank-biased overlap rather than exact-match rate.

**T3, ANN statistical equivalence.** Held-out query set against the static ground truth of §4.3, *not* against either engine's own exact search, which is what bfb's `--search-quality` does and why it can't be used here. bfb `dev` also offers recall against a *dataset's shipped* ground truth (§4.1), which is a shared reference but a strict id-set intersection with no ε-tie handling; T3 needs the ε-aware form of §8.6 and our own fp64 truth, so it is computed here regardless. Recall@k and mean relative distance error for both engines over an `ef` sweep, thousands of queries, with confidence intervals. Where qrels exist, nDCG@10 alongside. The publishable statement is *"equal recall at lower latency"*, backed by overlapping CIs at matched recall, never a raw QPS ratio.

**T4, Quantization fidelity.** Per encoding: the distribution of `|quantized_score − fp32_score|`, rank correlation against the fp64 oracle, and recall at matched oversampling. This is the tier that catches the most seductive way to fake a win: a cheaper quantizer that is faster because it is less accurate. Comparing at equal recall (§7.4) already guards against it; T4 makes the mechanism visible instead of merely controlling for it.

### 8.6 Metamorphic properties

Invariants that need no oracle at all, cheap enough to run on every commit, and historically very good at finding real bugs:

- `score(x, x)` is maximal, and exact top-1 for an indexed vector as query is that vector itself (under ANN this becomes a recall canary rather than an assertion).
- **Permutation invariance:** insertion order must not change exact-search results.
- **Prefix property:** `top_k` is a prefix of `top_(k+1)`.
- **Offset consistency:** `query(limit=L, offset=O)` equals `query(limit=L+O)[O..]`.
- **Idempotent upsert:** re-upserting identical points changes nothing observable.
- **Filter subsetting:** filtered results ⊆ unfiltered results ∩ points matching the filter (phase 3).
- **Quantization dominance:** with `rescore = true` and oversampling `n`, recall is monotonically non-decreasing in `n`.

Each is asserted against strawmann alone, and the interesting ones against Qdrant too, a property that holds for one engine and not the other is a finding.

### 8.7 Determinism is a design constraint, not a nice-to-have

The differ cannot distinguish "strawmann disagrees with Qdrant" from "strawmann disagrees with itself" unless strawmann is deterministic. This flows backwards into the engine design:

- **Fixed reduction order** in every kernel. No float accumulation into atomics, no work-stealing partial sums merged in completion order. Parallel brute force partitions by fixed ranges and merges deterministically.
- **Total order on results:** score descending, then internal ID ascending. Our own output is then stable even under ties. We do *not* require Qdrant to match that order, hence T2.
- **Seeded RNG** for HNSW level assignment, with the seed recorded in `meta.json`.
- **Reproducible graph builds.** This is the hard one: a lock-based parallel build with neighbour lists mutated in arrival order produces a thread-count-dependent graph. Options, in preference order: (a) deterministic build order with parallelism only inside each insertion's candidate search, (b) accept nondeterminism but checksum the resulting graph and require the checksum to be stable for a given (seed, thread count, dataset), (c) a slow deterministic build mode used only for conformance. Pick (b) as the default with (c) available; *as built*: (b) is the default and does not in fact reproduce across thread counts, and (c) exists as `buildSerial` (tested bit-reproducible) but is not selectable at runtime, so conformance runs against the parallel build; note that this constraint is a real cost on §6.5's build design and should be priced in, not discovered later.

Same input + same build + same thread count → bit-identical output. That property is what makes every other tier meaningful.

### 8.8 Fuzzing

- **Protocol fuzzing:** malformed HTTP/2 frames, truncated HPACK, oversized length prefixes, wrong wire types, deeply nested messages, dimension mismatches. Target: no crash, no hang, no UB, a clean gRPC error. Run against a `Debug` build so Zig's safety checks are live.
- **Differential fuzzing:** randomly generated collections (dim, count, metric, ID type, batch shape) and operation sequences, run against both engines, asserting the T1-T2 tiers. Shrink failing cases to a minimal reproducer and check it into the regression corpus.

### 8.9 The report

One artifact, one row per (workload, ISA build, encoding, engine pair):

```
workload · dataset name + base/query/GT checksums · qdrant version ·
strawmann commit · ISA build · conformance tier reached ·
max|Δscore| · p99|Δscore| · recall@1/10/100 · mean rel. distance error ·
nDCG@10 (where qrels exist) · QPS · p50/p99 latency ·
effective frequency · %-of-roofline
```

The Qdrant version is pinned and recorded per row: comparing against a moving target across months of work produces numbers that cannot be reasoned about. Bump it deliberately, re-baseline everything, keep both.

CI: metamorphic + T0 + T1 on a 100k slice per commit; full T0-T4 against the pinned Qdrant container nightly on the full dataset; the fuzz corpus continuously.

---

## 9. Repository layout and build

```
strawmann/
  build.zig, build.zig.zon
  src/
    main.zig, root.zig, lock.zig, sysinfo.zig
    net/      epoll loop, native h2 + hpack, grpc framing, fuzzers (fuzz.zig)
    proto/    hand-written codecs (wire.zig, messages.zig)
    api/      RPC handlers, request validation, status mapping, e2e tests
    core/     collection, ids, storage arenas (placements), persist, quantized store
    index/    hnsw build, hnsw search, brute force, visited, heaps
    quant/    sq8, binary, pq, codebook training
    dist/     simd kernels, comptime-generic per lane width, + cpuid dispatch
  bench/
    micro/    bandwidth, latency, kernels, perf counters (hw.zig, kernels.zig, perfctr.zig)
    isa/      forced-ISA asm probe + dump_asm.py (docs/asm/ regeneration and --check)
    harness/  workloads.py, compare.py, recall.py, fullrun.py, isa_sweep.py, ...
              report.py + report_data.py + report_charts.py, layered in that order
    setup.py  §7.1 environment gate
  conformance/          Rust, uses qdrant-client, one client, both engines
    src/differ/         T0-T4 tiers, tolerance calibration
    src/relevance/      held-out query driver, recall@k, MRDE, nDCG (needs qrels)
    src/oracle/         fp64 exhaustive reference + ground-truth cache
    src/metamorphic/    oracle-free invariants
    src/datasets/       fbin/fvecs/parquet readers
    datasets/           datasets.json + datasets.py (fetch, verify, convert)
  scripts/    check.py (the gate), doctor.py
  docs/asm/   checked-in disassembly of every hot loop, diffed by dump_asm.py --check (manual)
  docs/       this spec, the cost model, results
```

Not present, against the original layout: `src/obs/` (§5.7 is not implemented), `conformance/fuzz/` (the fuzzers live in `src/net/fuzz.zig`), and a vendored `proto/` directory (the codecs are hand-written against the 1.19 protos, §12; nothing is compiled from `.proto`).

- Pin an exact Zig toolchain version in `build.zig.zon` and CI. The language still moves; a benchmark that silently changes compiler is not a benchmark.
- Build modes: `Debug` (safety on, used for correctness tests), `ReleaseSafe` (default for correctness runs), `ReleaseFast -Dcpu=native` (the only mode results are ever quoted from). Every published number states its mode **and its ISA build target**.
- `build.zig` exposes the forced-ISA targets of §6.6.5 as first-class build steps, so `zig build bench-isa` produces the whole matrix from one source tree. Toolchain version is pinned: LLVM's autovectoriser and its AVX-512 cost model both move between releases, and an unpinned compiler turns the ISA matrix into noise.
- Correctness harness: see §8. The rule enforced by the build and the results sink is that **every index or kernel change is validated before any performance number is recorded**, and a perf row without a matching green conformance hash is rejected rather than merely flagged.

---

## 10. Milestones

Each milestone has an exit criterion that is a *measurement*, not a feature.

**M-1, Datasets and ground truth.** Fetch, convert, checksum, subset; fp64 oracle producing GT for every dataset in §4.2; diff against published GT. (Earlier drafts also listed `--query-fbin` contributed upstream to bfb; per §4.1 that capability now exists on `dev` in another shape and the item is closed.) Nothing else depends on it, and everything else depends on it.
*Exit:* GT cached and checksummed for SIFT1M and the headline dataset; published-vs-recomputed GT diff documented.

**M0, Hardware model + SIMD kernels.** Microbenchmarks for bandwidth, latency, MLP, TLB effects. The full `dist/` kernel set (§6.6.3) written comptime-generic and correct, with the forced-ISA build matrix wired up. No server, no index, this is the milestone that de-risks everything downstream, and it is where the AVX2/AVX-512 question gets its first real answer.
*Exit:* `docs/cost-model.md` with measured constants for the target host; the table in §5.2 filled in with real numbers; the §7.5 ISA matrix populated at the microbenchmark level for at least two microarchitectures, with cycles/vector, ns/vector, and effective frequency for every cell; disassembly of every hot loop checked into `docs/asm/`.

**M1, bfb says hello, and the differ says yes.** nghttp2-backed gRPC; `HealthCheck`, collection create/delete/exists/info, `Upsert`, `QueryBatch` over brute-force search in RAM. No persistence, no index. **The `conformance/` differ (§8.5) is built in this milestone, not later**, it is how we discover that our Euclid postprocessing or Cosine short-circuit is wrong, and discovering that after the index exists is much more expensive.
*Exit:* `bfb -n 100k -d 768 --search` completes end-to-end; **T0 and T1 green against pinned Qdrant for all four metrics**, with ε calibrated per §8.4 and `docs/tolerance.md` written; W0 gives us the RPC-overhead floor in µs/request.

**M2, Storage.** Flat files, both resident (hugetlb + pread) and mmap modes, aligned arenas, id maps, restart without rebuild.
*Exit:* restart of a 1M×768 collection under sequential-read time; brute-force QPS within X% of the M0 bandwidth model in both modes, and the TLB delta between modes quantified.

**M3, HNSW.** Parallel bulk build, flat level-0 layout, generation-stamped visited set, prefetching, Yellow→Green status flow, reproducible builds per §8.7.
*Exit:* **T3 green**, recall-vs-`ef` curves for both engines against the shared fp64 ground truth, with confidence intervals; graph-checksum stability verified for a fixed (seed, thread count); metamorphic suite passing; `Nd` measured against the model; W3/W4 with a stated %-of-roofline.

**M4, Scalar and binary quantization.** SQ8 + binary + rescore + oversampling, on top of M0's kernels.
*Exit:* **T4 green**, quantization fidelity distributions published, so the recall/latency frontier cannot be read as a cheaper-quantizer artefact; the (recall, latency) frontier for all three encodings on one chart; the L3-crossing discontinuity of §5.4 observed or explained away; **the §7.5 ISA matrix re-run end-to-end**, confirming or refuting M0's microbenchmark verdict once the memory system is in the loop. The predicted result, negligible AVX-512 gain on cold fp32, large gain on binary, is either demonstrated or the cost model is revised.

**M5, Product quantization.** PQ8 baseline, then PQ4/FastScan.
*Exit:* W8 on the frontier chart; a statement of where PQ beats SQ8 and where it doesn't.

**M6, Transport optimisation.** Native HTTP/2 if justified by M1-M5 profiles; zero-allocation request path; writev batching.
*Exit:* W0 floor improved by a measured margin, or a written decision that nghttp2 is not the bottleneck.

**M7, Payloads and filtering** *(optional, scope-dependent).* Payload storage, keyword index, filtered HNSW search.
*Exit:* W12 runs; filtered-search cost model documented.

**M8, NUMA and multi-socket** *(optional).* Per-node arena replication, socket-local worker pools.
*Exit:* scaling curve to 2 sockets with cross-socket traffic quantified.

**M9, Cross-microarchitecture ISA report** *(runs alongside M4-M6, not after).* The §7.5 matrix on the full target set: pre-Ice-Lake Intel, Sapphire Rapids or later, Zen 4, Zen 5.
*Exit:* a per-microarchitecture, per-kernel ISA selection table, and a recommendation for what a single shipped binary should dispatch to. This is the most directly transferable output of the project.

---

## 11. Risks and open questions

| Risk | Assessment | Mitigation |
|---|---|---|
| HTTP/2 + HPACK is a bigger rabbit hole than expected | moderate; it is a known-shape problem but full of small conformance details | nghttp2 first; native only if profiles justify |
| Zig ecosystem gaps (no mature gRPC, no protobuf codegen worth using, async story in flux) | high certainty, low severity, the subset needed is small | hand-write it; the narrow scope is what makes this tractable |
| Bulk build path diverges from Qdrant's incremental one, making build-time comparison unfair | real | report build time separately and describe both models honestly; do not claim an apples-to-apples build comparison |
| Uniform-random bfb data produces misleading recall and cache behaviour | high | `--fbin` mandatory for anything past W2 (§4) |
| Comparing against Qdrant with mismatched semantics (segments, payload, on-disk) | high, this is the classic way benchmark projects become useless | the semantic-gap checklist in §2 is part of every results table |
| Measuring on a noisy or thermally throttled host | high | `bench/setup.py` gates every run; interleaved A/B; discard high-spread runs |
| Zig version churn breaking builds mid-project | moderate | pin the toolchain; upgrade deliberately with a full re-baseline |
| AVX-512 frequency licensing makes single-host ISA conclusions wrong | high on pre-Ice-Lake Intel, low on Zen 5 / Granite Rapids | never conclude from one machine (§7.5); log effective frequency on every run; report per-microarchitecture |
| LLVM quietly splits 512-bit ops, or fails to contract FMA, or spills accumulators | moderate, and invisible without checking | disassembly checked into `docs/asm/` and diffed by `dump_asm.py --check` (manual, not in CI); `fp_arith_inst_retired.*_packed_single` counters confirm executed width; `llvm-mca` cross-check |
| Reassociation under `.optimized` float mode changes scores enough to move recall | low but real | correctness harness compares against a strict-order reference with an explicit tolerance; recall measured per ISA build, not assumed constant across them |
| Determinism requirement conflicts with the fastest parallel HNSW build | real, and discovered late is expensive | decided up front in §8.7: checksum-stable builds per (seed, thread count) as default, slow deterministic mode for conformance; priced into the M3 design. Outcome: the parallel build is not checksum-stable and the serial mode is not wired to a flag (decisions.md) |
| Chasing bit-exactness with Qdrant and burning weeks on it | moderate, it is a seductive goal | §8.1 states plainly that it is impossible; ε is calibrated from measured cross-ISA variance (§8.4), including Qdrant's own |
| Qdrant version drift invalidating months of comparison data | certain over the project's lifetime | pin the container tag, record it per result row, re-baseline deliberately and keep both series |
| Relevance measured with bfb's random queries (still the default without a dataset source), its self-referential `--search-quality`, or its tie-blind dataset recall (§4.1) | high, it is the path of least resistance and silently produces meaningless or subtly deflated recall | §4.1: bfb generates load, `conformance/relevance` generates recall; harness rejects relevance runs with `--offset != 0`, `--uuids`, or `--max-id` set |
| Ground truth drift, subsetting, re-embedding, or a new model silently invalidating GT | moderate | GT cached per `(dataset, subset, metric)`; base/query/GT checksums in every result row |
| A quantization win that is really an accuracy loss | high, and the classic way vector benchmarks mislead | equal-recall comparison (§7.4) plus T4 fidelity distributions (§8.5) |
| Optimising for the benchmark rather than for reality | the deepest risk in a project like this | keep the cost model primary, an optimisation that improves QPS without a model explaining *why* is treated as suspicious, not as a win |

**Open questions to resolve early:**

1. What is `Nd` really, on real data, as a function of `ef`/`M`/`N`? Everything in §5.3 hangs off it.
2. Does neighbour-list/vector co-location (§6.5) win, and at which encoding?
3. Generation-stamped array vs bitmap-plus-dirty-list for the visited set, at what point count does the 4 MB/worker footprint start hurting?
4. How much does the fixed 5 µs-ish gRPC floor matter at binary-quantized latencies of ~40 µs? (At that point transport is 10%+ of the response, which reframes the whole project.)
5. Is there a defensible way to compare build times at all, given the bulk/incremental asymmetry?
6. Does AVX-512 pay for itself on cold fp32 search? The model says no. If it does, what is the model missing, MLP from fewer, wider loads, or were we never bandwidth bound?
7. What is the real gap between `vpopcntq` and the `vpshufb` nibble-LUT for binary distance, once the working set is L3-resident rather than register-resident?
8. Is AVX-VNNI at 256 bits enough for the SQ8 kernel, making full AVX-512 unnecessary for that path on Alder Lake+ / Zen 4+?
9. Does the optimal configuration turn out to be *mixed*, wide for some kernels, narrow for others, on parts with frequency licensing?
10. What is Qdrant's own cross-ISA score spread, and is it larger or smaller than ours? If larger, our ε is set by *its* variance, which is itself a publishable observation.
11. Can the HNSW build be made checksum-stable without giving up meaningful build parallelism, or does conformance cost us build throughput?
12. How much does recall@10 loss from quantization actually cost in nDCG@10 on a real retrieval task? If the answer is "almost nothing down to 1 bit", that reframes the entire quantization trade-off for users.
13. Do the two engines diverge more on the cross-modal / OOD dataset (Text2Image) than on in-distribution ones? OOD queries are where HNSW implementations are most likely to differ in ways recall@10 on SIFT would never reveal.

---

## 12. Appendix, minimal proto subset

Messages required for M1 (from Qdrant 1.19 protos, `package qdrant`):

- `HealthCheckRequest` / `HealthCheckReply{title, version, commit}`
- `CreateCollection` (read: `collection_name`, `vectors_config`, `hnsw_config{m, ef_construct, full_scan_threshold}`, `quantization_config`, `optimizers_config`; validate and reject the rest)
- `DeleteCollection`, `CollectionExistsRequest/Response`
- `GetCollectionInfoRequest` → `CollectionInfo{status, optimizer_status, points_count, indexed_vectors_count, segments_count, config, payload_schema}`
- `UpsertPoints{collection_name, wait, points[], ordering}`, `PointStruct{id, vectors, payload}`, `PointId{num|uuid}`, `Vectors`, `Vector{dense|sparse|multi_dense}`, `DenseVector{data[]}`
- `PointsOperationResponse{result{operation_id, status}, time}`
- `QueryBatchPoints{collection_name, query_points[], timeout}`, `QueryPoints{query, using, filter, params, limit, offset, with_payload, with_vectors, score_threshold}`, `Query{nearest}`, `VectorInput{dense}`, `SearchParams{hnsw_ef, exact, indexed_only, quantization{ignore, rescore, oversampling}}`
- `QueryBatchResponse{result[], time}`, `BatchResult{result[]}`, `ScoredPoint{id, payload, score, version, vectors}`
- Phase 2+: `ScrollPoints`/`ScrollResponse`, `SetPayloadPoints`, `CreateFieldIndexCollection`, `Filter`/`Condition`/`FieldCondition`/`Match`

Everything else on `qdrant.Points` and `qdrant.Collections` returns `UNIMPLEMENTED` with a message naming the RPC, so an unexpected bfb flag fails loudly and immediately rather than producing a subtly wrong benchmark.
