# Ground truth: published vs recomputed

**M-1's exit criterion**, per §10:

> *Exit:* GT cached and checksummed for SIFT1M and the headline dataset;
> published-vs-recomputed GT diff documented.

This document is the diff, and M-1 is met for both tiers. SIFT1M carries the
diff itself, since it is the tier that ships ground truth to disagree with.
The headline tier (`dbpedia-openai-1m`) is fetched and verified (26 of 26
shards, cosine k=100 recomputed and cached (§4 below)) and has *no* published
ground truth, so §4.2 marks it recompute-only and there is nothing to diff:
M-1's diff obligation is discharged by SIFT1M alone. Its conversion path is
§5, and §6.1 gives the one external control this chain has.

Reproduce with:

(`$STRAWMANN_DATA` is the dataset root; see [datasets.md](datasets.md#choosing-the-dataset-directory).)

```sh
conformance/datasets/datasets.py fetch sift1m
cd conformance
cargo run --release -- oracle \
    --base   $STRAWMANN_DATA/sift1m/sift/sift_base.fvecs \
    --queries $STRAWMANN_DATA/sift1m/sift/sift_query.fvecs \
    --metric euclid --k 100 --dataset sift1m \
    --out $STRAWMANN_DATA/sift1m/gt/sift1m.euclid.k100.gt.json
cargo run --release -- gt-diff \
    --ours      $STRAWMANN_DATA/sift1m/gt/sift1m.euclid.k100.gt.json \
    --published $STRAWMANN_DATA/sift1m/sift/sift_groundtruth.ivecs
```

---

## 1. Provenance

| | |
|---|---|
| source | `fzliu/sift1m` on Hugging Face, the original TEXMEX `sift.tar.gz`, byte-for-byte |
| why not `ftp.irisa.fr` | the canonical host's passive FTP data channel is unreachable from this network; HTTPS is not |
| why not `ann-benchmarks.com` | reachable, but its ground truth is *recomputed by ann-benchmarks*. Diffing our oracle against someone else's recomputation answers a different question than the one M-1 asks. It is also the copy bfb's accuracy path would need, see *Which dataset kinds carry ground truth* below |
| tarball sha256 | `92f1270c5e3a0cb4…` (upstream Git-LFS oid, pinned in `conformance/datasets/datasets.json`) |
| files | `sift_base.fvecs` 516,000,000 B · `sift_query.fvecs` 5,160,000 B · `sift_groundtruth.ivecs` 4,040,000 B · `sift_learn.fvecs` 51,600,000 B |
| base checksum | `7bf7ead7809a45cd` (§4.3's per-row checksum) |
| query checksum | `1eb61b7e8c2a5b46` |
| GT sha256 | `8abd756e7a6874f1…` |

Checksums in the manifest are **upstream LFS digests**, not digests we computed
after downloading. A self-generated checksum only proves the file has not rotted
on our disk; the upstream one also proves we fetched what its publisher
published, which is what §4.2's "static, versioned, checksummed" is for.

### 1.1 Which dataset kinds carry ground truth

bfb's accuracy path needs a query set *and* a ground truth, and only three of
its source kinds carry both (`src/dataset/reader.rs`: `query_ground_truth`
bails with `dataset has no ground truth` for the rest):

| kind | corpus | query set + GT |
|---|---|---|
| h5 (ann-benchmarks: `test` + `neighbors`) | yes | **yes** |
| tar (ann-filtering-benchmark: `tests.jsonl`) | yes | **yes** |
| sparse (`queries.csr` + `results.gt`) | yes | **yes** |
| npy | yes | no |
| parquet | yes | no |
| partitioned | yes | no |
| **fbin** | `--fbin`, legacy upload path only | **not a dataset source at all** |

This cuts against both §4.2 tiers, in opposite directions. **SIFT1M**: bfb's
accuracy path would need the ann-benchmarks `sift-128-euclidean.hdf5`, whose
`neighbors` is ann-benchmarks' own recomputation, the artefact this document
deliberately avoids for M-1, so bfb's recall would be scored against a
different ground truth than our fp64 oracle's. **dbpedia-openai-1m**: bfb can
ingest the parquet shards as a corpus but cannot measure accuracy on them,
because parquet carries none. Our own converted `fbin` artefacts are invisible
to the accuracy path entirely. This is why spec §4.1 keeps relevance in
`conformance/` and uses bfb purely for load.

## 2. The oracle run

fp64, exhaustive, strict summation order, §8.2's tier 1.

```
1,000,000 base × 10,000 queries, d=128, euclid, k=100
10,000,000,000 distance evaluations
2 m 24 s wall, 1.4 GB peak RSS, 24 threads
```

Two changes were needed to make this tractable, both performance-only and both
covered by equivalence tests that assert bit-identical output:

- **Partial sort instead of full sort.** `top_k` sorted all `n` candidates per
  query, O(n log n) per query, ~2×10¹¹ comparisons across the query set, and it
  dominated the distance computation it was supposed to be a footnote to.
  `select_nth_unstable_by` is O(n). This is only safe because the comparator is
  a *total* order on `(score, id)`: no two elements compare equal, so an
  unstable partition cannot reorder anything relative to the stable full sort.
  `partial_sort_matches_full_sort_bit_for_bit` checks all four metrics at four
  values of k, comparing score *bit patterns* rather than approximately.
- **Parallel across queries.** Each query reads shared immutable base data and
  writes only its own row, so there is no reduction to order. Parallelism is
  applied **only** across queries and never inside a distance, reassociating an
  fp64 sum would change the answer, which for a tier-1 oracle is the one thing
  that must not happen. `compute_is_independent_of_thread_count` asserts
  bit-identical ids and scores at 1 and 7 threads.

## 3. The diff

```
10000 queries: 4446 identical,
               5443 same-set-different-order,
                111 differ only at the k-th boundary,
                  0 differ in the interior
               (19519 positions total)
id overlap 999889/1000000 = 0.999889
```

**We agree with the published ground truth about 999,889 of 1,000,000
neighbours.** All 111 disagreements are at the k-th boundary; none is interior.

### 3.1 Over half the "disagreement" is tie-breaking, not disagreement

5443 of 10000 queries, **54%**, return exactly the same 100 neighbours in a
different order. A positional diff counts each of those as a mismatch, and the
first version of this tool did: it reported 19,519 mismatched positions and
flagged 63 queries as interior disagreements.

That framing is wrong, and wrong in the alarming direction. Separating *set*
overlap from *positional* equality is what turns "19,519 positions differ" into
"111 ids differ", and the two numbers support completely different conclusions.

SIFT descriptors are quantized integers, so exact distance ties are common
rather than exotic; the two ground truths simply break them differently. Ours
breaks ties by lowest id (§8.7's total order), deterministically.

### 3.2 §4.3's stated expectation does not apply to this dataset

§4.3 predicts the mechanism:

> "Several published GTs were computed in fp32 and have genuine disagreements
> near the k-th boundary."

For SIFT1M that prediction is **structurally impossible**, and the data agrees.
SIFT components are integers in `[0, 255]` (`[0, 183]` across a 20k-vector
sample), so a squared L2 distance is at most `128 × 255² = 8,323,200`, well
below fp32's exact-integer limit of `2²⁴ = 16,777,216`. **Every squared distance
in this dataset is exactly representable in fp32.** There is no rounding for
fp32 and fp64 to disagree about. Ranking by `√·` cannot change that, since the
square root is monotonic.

So on SIFT1M the residual disagreement is *entirely* tie-breaking convention,
and observing 0 interior disagreements is the confirmation. The fp32-rounding
mechanism §4.3 describes is real, but it will show up on a dataset of genuine
embeddings, the headline tier, not on this one.

### 3.3 A classifier bug this dataset exposed

The first run reported **63 interior disagreements**, with the tool's own note
that this "is not the expected fp32-rounding pattern and is worth
investigating." Investigating it found the fault in the tool, not the data.

Every one of the 111 differing ids sits at **rank 99**, the last position, and
every one has a score *exactly* equal to the 100th score. They are boundary ties
without exception: two vectors equidistant from the query, one of which has to
be cut at `k=100`.

They were being classified as interior because the classifier keyed on the first
*positional* mismatch. A query whose tied runs are permuted at rank 3 has its
first positional mismatch at rank 3, even when the sole set difference is the
single id at rank 99. The fix keys on where the neighbour **set** diverges
instead; `early_tie_permutation_plus_boundary_cut_is_a_boundary_disagreement`
pins the reduced case.

This is worth recording because it is the failure mode §8 exists to catch,
pointed at the harness rather than the engine: a metric that produced a
plausible, specific, alarming number, "63 queries disagree in the interior" -
that was an artefact of how it was computed.

### 3.4 Which GT to use

Ours. §4.3:

> "Use ours, publish the diff once, and note that a disagreement with the
> shipped GT is expected rather than alarming."

The published `ivecs` carries ids only, no scores, and §8.6's ε-aware recall
needs the k-th score to decide whether a returned id ties the boundary. The
recomputed GT carries fp64 scores for exactly that reason.

---

## 4. Status

| dataset | fetched | GT computed | diffed |
|---|---|---|---|
| **sift1m** | yes, verified | yes, 2 m 24 s | yes, 0 interior disagreements |
| **dbpedia-openai-1m** | yes, 26 of 26 shards verified | yes, cosine k=100 | n/a, §4.2 marks GT as recompute-only, so there is nothing to diff against |
| **dbpedia-openai-100K-1536-angular** | yes, verified | yes, 1 m 44 s, cosine k=100 | yes, against the shipped k=10: see §6.1 |
| **laion-small-clip** | yes, verified | no | its shipped neighbours are filtered (§6) |
| **h-and-m-2048-angular-filters** | yes, verified | no | as above |

## 5. The headline tier's conversion path

Built and tested; waiting only on the download.

```sh
cargo run --release -- convert-parquet \
    --in-dir $STRAWMANN_DATA/dbpedia-openai-1m \
    --out-base   $STRAWMANN_DATA/dbpedia-openai-1m/base.fbin \
    --out-queries $STRAWMANN_DATA/dbpedia-openai-1m/queries.fbin \
    --n-queries 10000
```

Implemented in Rust (`conformance/src/datasets/parquet_convert.rs`) rather than
with `pyarrow`, because §9 puts `convert` in `conformance/datasets/` and the
pinned toolchain is the whole reason results are reproducible. Measured at 0.8 s
per 20k rows on the shards already fetched, so ~40 s for the full set.

Three decisions worth stating, because each is a place a plausible shortcut
produces silently wrong relevance numbers:

**The query split is held out, so the base set is 990,012, not 1,000,000.**
§4.2 marks this tier "held-out split". If queries are drawn from the base set
then every query's nearest neighbour is itself at distance 0, recall is inflated
towards 1, and nothing about the run looks wrong. The tail 10,000 rows become
the query set and are excluded from the base. §4.2's "1M" describes the raw
dataset; holding queries out necessarily costs the base those rows.
`converts_and_holds_the_query_split_out_of_base` asserts no query vector appears
in the base set at all.

**Shards are processed in sorted order, not directory order.** Row order defines
point ids and ground truth indexes into them (§4.3), so taking `read_dir` order
would make every cached GT depend on how files happened to land on disk.

**f64 → f32 happens at conversion, not per consumer.** The shards store
`double`; §3 stores fp32. Converting once means the oracle reads the same
`fbin` the engine ingests, so both see bit-identical input and any difference
between them is attributable to the engine. Computing GT from the f64 parquet
while the engine indexes an f32 copy would inject a discrepancy at the very
bottom of the stack that no differ tier could localise.

### 5.1 A first look at the data confirms finding #5 on real embeddings

ada-002 embeddings are unit-normalised, and at d=1536 this is exactly the regime
where §8.3's cosine short-circuit is most fragile. Over 2000 real base vectors:

| summation | pass `\|Σx²−1\| ≤ 1e-6` |
|---|--:|
| f64 | 2000/2000 = **100.0%** |
| strict f32, as Qdrant computes it | 1725/2000 = **86.2%** |

The vectors *are* unit vectors. Whether Qdrant thinks so depends entirely on the
summation precision, and in strict f32 order ~14% of them drift outside the
window and get divided a second time. The synthetic measurement in
`docs/isa-matrix.md` put this at 79.1% at d=1536; real embeddings are somewhat
better behaved but nowhere near 100%.

This is the concrete argument for computing the sum of squares in strict f32
order rather than more accurately (`src/dist/norm.zig`): a better sum puts a
different population on each side of a discontinuous branch, and the population
is large.

### 5.2 Remaining work for this tier

Its GT run is ~12× SIFT1M's work at d=1536, so roughly half an hour on this
host. There is nothing to diff it against, §4.2 marks its ground truth
recompute-only, so M-1's diff obligation is discharged by SIFT1M alone.

**These are dataset-preparation numbers and carry no §7.1 gate implications** -
the wall-clock figures above describe how long the oracle took, not engine
performance, and nothing here is a comparison against Qdrant.

---

## 6. The bundle tiers' conversion path (`vectors.npy` + `tests.jsonl`)

Three of §4.2's entries ship as a vector-db-benchmark bundle. bfb reads that
layout natively, its `tar` source opens the directory, so nothing was needed
to *load* them. But the oracle, the differ and `relevance` all read `fbin`
(§4.3 makes it canonical), so until now such a dataset could be searched and
never scored, and §7.4 forbids a qps number without its recall.

```sh
cargo run --release -- convert-npy \
    --in-dir      ~/Documents/datasets/dbpedia-openai-100K-1536-angular/dbpedia_openai_100K \
    --out-base    ~/Documents/datasets/dbpedia-openai-100K-1536-angular/base.fbin \
    --out-queries ~/Documents/datasets/dbpedia-openai-100K-1536-angular/queries.fbin
```

`conformance/src/datasets/npy_convert.rs`. Three things it does differently
from the parquet path, each because the alternative is silently wrong:

**There is no query split to hold out.** `convert-parquet` takes the tail of the
base rows because dbpedia's shards are one undivided corpus. Here the queries
arrive in `tests.jsonl` and were never part of `vectors.npy`, so no query can be
its own nearest neighbour and a second split would only shrink the corpus that
the shipped neighbour ids index into.

**`float16` is widened, and refusing the wrong dtype is the point.**
`laion-small-clip` is stored as `<f2`. A reader assuming `f32` gets half the
rows at twice the dimension and no error at all, so the dtype is read from the
header and an unsupported one is named rather than guessed. The widening is
exact (every half, subnormals included, is representable in f32) so it is
written as arithmetic and asserted with equalities rather than tolerances.

**Fortran order is refused, not transposed.** A Fortran-ordered array read as C
order yields a matrix of exactly the right shape holding entirely wrong rows:
plausible corpus, nonsense neighbours.

**The shipped `closest_ids` are not treated as ground truth.** For entries whose
`conditions` are empty they *are* unfiltered neighbours; for `laion-small-clip`
and `h-and-m-2048-angular-filters` every query carries a range or match
condition and the shipped ids are neighbours *under that condition*. Emitting
something that is ground truth for one dataset and a trap for another is worse
than emitting nothing, so the converter writes vectors only and the ground truth
comes from the fp64 oracle, which computes the same thing for every dataset.

That is also why those two are not yet `--dataset` choices: they convert
cleanly, and they have no unfiltered ground truth to be scored against. Filtered
recall is its own piece of work.

### 6.1 The conversion has an external control, and it passes

`dbpedia-openai-100K-1536-angular` is the one bundle whose `conditions` are
empty, so upstream's `closest_ids` are genuine unfiltered neighbours and the
recomputed ground truth can be checked against them. Over all 5,000 queries:

| check | result |
|---|--:|
| top-10 set identical | 5000 / 5000 = **100.0%** |
| top-10 identical including order | 4999 / 5000 = **99.98%** |

The single ordering difference is query #1694, positions 8 and 9, whose scores
differ by **1.96e-07**, a tie that fp64 and upstream's f32 resolve differently.
The sets agree; §8.3's eps-aware recall exists for exactly this.

This exercises the whole chain at once (the npy header, C-order row layout, the
`tests.jsonl` query parse, the f32 write, and the cosine oracle) against
neighbours computed by someone else. A row-order or normalisation error anywhere
in it would not survive a 100% set match.

**These are dataset-preparation numbers and carry no §7.1 gate implications.**
