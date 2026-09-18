# `conformance/`. The half that decides whether a number may be published

This is not a test suite for strawmANN. It is the **correctness gate that
licenses the performance claim**, and it is a separate binary in a different
language for a reason.

§8's rule, which everything here exists to enforce:

> **No performance number is publishable unless the same build, on the same
> data, has a green conformance row.** This is enforced by the harness, not by
> convention. The results sink rejects a perf row whose conformance hash
> doesn't match a passing run.

You can watch it fire. A benchmark run with no conformance row ends like this,
and this is the machinery working rather than failing:

```
=== results sink (§8) ===
REJECTED, and this is the gate working: perf row carries no conformance hash.
0 perf row(s) recorded, 16 refused by §8
```

## Why Rust, and why a separate binary

It links the **same `qdrant-client` crate that `bfb` links** (this crate pins
`=1.19.0`; bfb's pinned commit builds the crate's `dev` branch, `1.16.1-dev`,
against the same 1.19 protos), and talks to both engines through it. §8.5:

> "`conformance/` is a Rust binary using the same `qdrant-client` crate that
> bfb uses. This is deliberate: one client, one encoder, one decoder, both
> engines. Any difference observed is then necessarily server-side, not an
> artefact of how the harness talks to each engine."

If strawmANN's own client spoke to strawmANN and something else spoke to
Qdrant, every divergence would have a second possible explanation, and the one
thing this project sells is that a divergence has only one. `src/engine.rs` is
deliberately a thin wrapper for the same reason: the less code between the
comparison and the wire, the fewer places a difference can be manufactured.

## The three claims, and which one is load-bearing

§8.1 separates three claims of decreasing strength, because conflating them is
the standard way this kind of comparison goes wrong. They are an enum in
`src/differ/mod.rs`, with `achievable()` answering for each:

| claim | achievable? |
|---|---|
| **Bit-exact identity** with Qdrant | **No**, and not a defect in either engine. "Anything promising ID-level equality for HNSW results is promising something false. The spec does not." |
| **Value equality within ε** for *exact* search | **Yes, and this is the load-bearing claim.** Not for ANN search. |
| **Statistical equivalence** of ANN result distributions | **Yes, as distributions.** |

The headline the project is trying to earn is *identical outputs, N× faster*,
so the middle row is the one that has to hold, and T1 is the tier that tests
it.

## The tiers, T0 through T4

Cumulative: each licenses something the next depends on.

| tier | what it checks | what it catches |
|---|---|---|
| **T0** wire conformance | structural comparison of responses | "right numbers, wrong shape", which no amount of score comparison will find |
| **T1** exact-search value equality | `(id, score)` pairs: scores within ε, **ids equal up to permutation within an ε-tie**, against Qdrant and against the fp64 oracle; every query answered, every list exactly `limit` long | **the tier that licenses the performance claim**, an id-mapping bug with correct scores fails it |
| **§8.6** metamorphic properties | the oracle-free invariants below, on strawmann (Qdrant's side is a finding) | recorded as a tier so a violated invariant enters the row and its hash |
| **T2** rank agreement under ties | multisets grouped into score-equivalence classes at width ε, never ordered lists; passes iff the tie-class multisets agree at every cut both engines draw | a tie broken differently being read as a wrong answer |
| **T3** ANN statistical equivalence | against §4.3's shared ground truth, *not* against either engine's own exact search | one engine's bug becoming the other's target |
| **T4** quantization fidelity | encoded-vs-original behaviour; passes iff §8.6's quantization dominance holds and rank correlation with the oracle clears a floor | "the most seductive way to fake a win: a cheaper quantizer that is faster because it is less accurate" |

None of the tiers passes on nothing: an empty comparison is a structural
failure, not a green row.

## Commands

Each maps to a spec section by name.

| command | § | what it does |
|---|---|---|
| `oracle` | 8.2 | fp64 exhaustive ground truth, cached per (dataset, subset, metric) |
| `gt-diff` | 4.3 | diff our recomputed ground truth against a published `.ivecs` |
| `differ` | 8.5 | run T0-T4 against both engines; `--json` writes the row the sink needs |
| `dump-scores` | 8.4 | dump one build's exact-search scores, the input to calibration |
| `calibrate` | 8.4 | derive ε from measured cross-ISA spread, write `docs/tolerance.md` |
| `relevance` | 4.4 | recall@k and MRDE over an `ef` sweep, against **one** engine |
| `convert-vecs`, `convert-parquet`, `convert-npy` | 9 | dataset conversion; §4.3 makes `fbin` canonical |
| `drop-collections` | 8.5 | delete the scratch collections a differ run leaves on both engines |
| `scroll-check` | 12 | page a collection with `Scroll` through the real client, checking `prost` decodes what we emit |
| `collection-info` | 4, 8.9 | read back what a collection **is** (segments, config) from the engine, as JSON, rather than what it was asked to be |
| `datasets` | 4.2 | list the dataset tiers |

§8.6's metamorphic properties are **not** a subcommand: `differ` runs them
itself (five whenever T1 runs, `quantization_dominance` beside T4 (where it
is part of T4's verdict)) because they are assertions about a running engine
rather than a separate exercise of one. Six of the seven run; `filter_subsetting`
waits on per-condition ground truth (docs/workloads.md, W12 point 3) and on this
client sending a `Filter`, not on the engine, which serves filtered search since
2026-09-03. Their result is recorded as a tier in
the conformance row, so it is in the hash the sink checks.

### `oracle`. Because neither engine is ground truth

> "When they disagree we need to know *who is wrong*, which requires a third,
> independent reference."

A deliberately unoptimised fp64 exhaustive scan. Correctness matters far more
than speed here, but it is also the one part of the harness where wall-clock
bites: SIFT1M is 10¹⁰ distance evaluations and the headline tier is 12× that.
So `rayon` parallelises **across queries only, never inside a distance**. The
fp64 summation order, and therefore the ground truth, does not depend on thread
count. A ground truth that changed with `--threads` would not be one.

### `calibrate` (ε is derived, not chosen

> "Picking `1e-6` because it looks reasonable is unprincipled."

Qdrant dispatches to AVX/SSE/NEON/scalar kernels by runtime feature detection,
and those differ in accumulator count and FMA usage) so **Qdrant does not
agree with itself** across ISAs. ε is derived from that measured spread, which
means a divergence the hardware guarantees is not reported as a bug. Both
`differ` and `relevance` default to the calibrated cell for their
`(metric, dim)` from `docs/tolerance.md` (mirrored in `tolerance.rs`), print
which ε they used and where it came from, and only fall back to a derived floor,
tighter than the physical noise at high dimension, when no cell exists.
`calibrate` merges its one cell into the existing table rather than replacing
the document, and prints the exact `CALIBRATED` entry to paste into
`tolerance.rs`, so the two stop drifting through hand-editing.

### Metamorphic properties (invariants needing no oracle

Cheap enough to run on every commit, and historically very good at finding real
bugs:

- `score(x, x)` is maximal; exact top-1 for an indexed vector as query is that
  vector itself (under ANN this is a recall canary, not an assertion)
- **permutation invariance**) insertion order must not change exact results.
  Run for real: a slice of the base is inserted forwards into one scratch
  collection and backwards into another, and the exact results compared up to
  ε-ties (§8.7 breaks ties on internal id, which *is* insertion order). The
  weaker "query batch order does not matter" check that used to carry this
  name still runs, as `query_order_invariance`.
- **prefix property**, `top_k` is a prefix of `top_(k+1)`
- **offset consistency**, `query(limit=L, offset=O)` equals `query(limit=L+O)[O..]`,
  with a request that actually carries `offset = limit`
- **idempotent upsert**, re-upserting identical points changes nothing observable
- **filter subsetting**, filtered ⊆ unfiltered ∩ matching (phase 3)

The positional ones (prefix, offset, batch order, idempotent upsert) are
judged up to ε-ties with the same rule T1 uses, and self-retrieval accepts a
duplicate of the query at the top: on a dataset with duplicate vectors Qdrant
orders a tie differently from one request to the next, and an exact `ids ==`
recorded that as a Qdrant finding against properties it does not violate.
- **quantization dominance**. With `rescore = true`, recall is monotonically
  non-decreasing in oversampling `n`

### What is *not* measured: semantic relevance

§4.4 splits relevance in two and says "report both; never substitute one for
the other":

| | geometric | semantic |
|---|---|---|
| question | did the index find the true nearest neighbours? | did the system retrieve what a human considers relevant? |
| ground truth | exact k-NN under the metric (fp64 oracle) | qrels / human judgements |
| metrics | recall@k, MRDE | nDCG@10, MRR |

Only the geometric half runs. `evaluate_semantic`, `ndcg_at_k` and `mrr` are
implemented and unit-tested but **no command reaches them**, because none of
§4.2's dataset tiers ship qrels. So no run emits nDCG or MRR today, and nothing
should claim otherwise, the functions carry `#[allow(dead_code)]` with that
reason attached rather than sitting silently unreferenced.

### `relevance`. The other half of every qps

§7.4: a throughput number without its recall is not publishable. `bfb` generates
load and knows nothing about relevance; this measures relevance and its request
rate is a smoke test rather than a benchmark. They meet on `ef`, and
`bench/harness/recall.py` drives this per collection so the join is against the
collection actually measured.

The tie clause (§4.3) is judged on the oracle's own fp64 score of each returned
point, recomputed from the base vectors, not on the score the engine reported;
a returned non-neighbour whose reported score beats the k-th truth is counted
as a miss and reported as an impossible score. An engine that returns fewer
than `limit` results has its empty slots scored as misses (recall is `n/a` only
when the *request* could not produce it), and the count of such short lists is
printed.

It needs only **one** engine, which is also the only way to test end-to-end at
dataset scale with no Qdrant instance to compare against.

## Running it

```bash
cargo build --release
cargo test --release          # 167 tests: oracle, differ tiers, tolerance, metrics

datasets/datasets.py list     # §4.2's datasets
datasets/datasets.py fetch    # fetch and verify checksums
```

Ground truth first, then the differ with both engines up:

(`$STRAWMANN_DATA` is the dataset root; see [docs/datasets.md](../docs/datasets.md#choosing-the-dataset-directory).)

```bash
cargo run --release -- oracle \
  --base $STRAWMANN_DATA/sift1m/sift1m.fbin \
  --queries $STRAWMANN_DATA/sift1m/sift1m_query.fbin \
  --metric euclid --k 100 --dataset sift1m \
  --out $STRAWMANN_DATA/sift1m/gt/sift1m.euclid.k100.gt.json

cargo run --release -- differ \
  --strawmann http://localhost:6344 --qdrant http://localhost:6333 \
  --base .../sift1m.fbin --queries .../sift1m_query.fbin \
  --ground-truth .../sift1m.euclid.k100.gt.json \
  --json ../bench/results/strawmann/conformance.json
```

`bench/harness/fullrun.py` does all of this in order, and is the intended way
to produce a full run. Including the detail that the differ is the **only**
phase where both engines run at once. That would be wrong for a throughput
measurement and is fine for a correctness one: it compares answers, not speeds.

## What CI runs

§8.9's split:

> "metamorphic + T0 + T1 on a 100k slice per commit; full T0-T4 against the
> pinned Qdrant container nightly on the full dataset; the fuzz corpus
> continuously."

## Layout

```
src/
  main.rs            the subcommands above
  engine.rs          one client, both engines (deliberately thin
  oracle/            fp64 exhaustive reference + ground-truth cache (§8.2)
  differ/            T0-T4 tiers (§8.5)
  differ/tolerance   ε calibration from cross-ISA spread (§8.4)
  metamorphic/       oracle-free invariants (§8.6)
  relevance/         recall@k, MRDE, nDCG@10 (§4.4)
  datasets/          fbin conversion, parquet shards (§9)
datasets/
  datasets.json      the one definition of every dataset, with checksums
  datasets.py        fetch and verify
```

## See also

- [`../docs/spec.md`](../docs/spec.md) §8) the conformance gate in full
- [`../docs/ground-truth.md`](../docs/ground-truth.md), how the oracle is validated
- [`../docs/tolerance.md`](../docs/tolerance.md), the calibrated ε, and how it was measured
- [`../docs/validation.md`](../docs/validation.md). What is currently licensed, and what is not
