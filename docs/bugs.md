# Bugs worth naming

Each produced a plausible wrong number rather than a failure, which is the
failure mode this project exists to catch. Every entry here is fixed, and most
carry a regression test that fails without the fix; the four marked *(no test)*
are covered by construction and a comment.

The first three were found by the measurement discipline the spec mandates. The
rest came from review passes over the finished code.

**LLVM silently capped every AVX-512 arm at 256 bits.** `x86_64_v4` sets
`prefer_256_bit`, which `std.simd.suggestVectorLength` honours. Every arm built
as "AVX-512" emitted 256-bit code until the feature was explicitly subtracted.
This is §11's "LLVM quietly splits 512-bit ops" risk landing before a single
kernel had been benchmarked, and `docs/asm/SUMMARY.md` is what caught it.

**A tail-handling bug masqueraded as an ISA result.** `hamming` measured 0.99×
for `vpopcntq` against `vpshufb`, no benefit at all, against §6.6.2's
assessment that it is "the single biggest ISA gap in the whole engine". The
kernel's scalar tail grows as the register widens: at d=384 the 512-bit build
did *no* vector work. Fixed by padding stored rows to a whole register, which
took the cold ratio from 1.57× to **2.72×** at d=768 and binary's cold cost from
29.8 to 20.3 ns. The instruction-count column is what made it visible; a
cycles-only table would have recorded it as a finding about the ISA.

**`FUTEX_WAKE` with `maxInt(u32)` wakes one waiter, not all.** The kernel reads
the count as a signed `int`, so `0xffffffff` arrives as `-1`. Shutdown released
one worker and `Thread.join` blocked forever on the other, every RPC having
already completed correctly, which is why no functional test caught it. Found by
`strace` on a hung test; `INT_MAX` is the portable spelling of "all".

**Found by the review pass.** Each is reachable from a well-formed client
request or a supported build. Most carry a regression test that fails without
the fix; four are covered by construction and a comment only, and are marked
*(no test)* below: the failed-partway batch, the `linkBack` aliasing, the
`XGETBV`/`OSXSAVE` order, and the `defer`/`errdefer` pair in `main`.

*Concurrency and lifecycle, the group that actually loses data:*

- **`Collection.upsert` took no lock while running on N worker threads.** Two
  concurrent Upsert RPCs both read `id_space.next`, both claimed the same row,
  and one vector was silently overwritten while an external id resolved to a
  stranger's vector. The lost `IdMap.count` increment could also defeat the
  guard that stops `getOrInsert` probing forever. Ingest is now serialised per
  collection; search still takes no lock.
- **Points upserted *during* a build were stranded permanently.**
  `invalidateIndex` only acted on `.ready`, so a point arriving mid-build left
  no trace: the build published `.ready` with a stale count, `status()` stayed
  Yellow forever, and the `.absent → .building` CAS could never win again, so no
  rebuild ever ran. bfb's poll loop overlapping the tail of an upload hits this
  every time.
- **A batch that failed partway skipped invalidation entirely**, wedging the
  collection the same way, a single bad point in a 1000-point batch was enough.
  *(no test; the build now re-evaluates `needsRebuild` itself after publishing,
  below, so the per-batch invalidation is no longer load-bearing.)*
- **A rebuild freed the graph a concurrent search was traversing.** Reachable by
  W11 (mixed read/write). Replaced artefacts are now retired, not freed.
- **Connection slots and file descriptors leaked** when a connection closed with
  requests still in flight.
- **The parallel HNSW build promoted the entry point without synchronisation**,
  making `entry_lock` dead code and letting `max_level` and `entry_point` go
  inconsistent; a second buffer-aliasing bug had `insertOne` iterating a
  neighbour list that `linkBack` was concurrently rewriting *(the aliasing has
  no test of its own; the entry-word race has "a builder reading a shared entry
  word never pairs a level with the wrong node")*.
- **The parallel build published a node before all of its lists were written.**
  Back-links were created level by level, so a node was reachable at level
  l+1 while its level-l list was still the `memset` zero: a thread descending
  through it searched level l from a node with no neighbours, linked itself to
  a one-element result, and the write of the real list then dropped that
  back-link. 43 of 20,000 nodes unreachable at 8 threads (0 serially), each
  with structurally zero recall and no error anywhere. Every level's list is
  now written first and back-links created afterwards; test "parallel build:
  every node is reachable from the entry point on level 0".
- **A build outrun by sustained writes downgraded the index to `.absent`.**
  `build_stale` marked a build whose count was passed by ingest and the
  publish step dropped it, so under W11-style load the graph could vanish for
  queries and never come back while the writer kept the tail moving. The
  build now publishes `.ready` and re-evaluates `needsRebuild` under the write
  lock, so a tail past the ratio starts the next build rather than deleting
  the current one; tests "points upserted during a build are not stranded
  outside the index" and "a build outrun by ingest steps aside on its own,
  per point and not only per batch".

*Unvalidated client input, the group that crashes the server:*

- **Unbounded `hnsw_ef`** indexed a heap sized once at startup. An `hnsw_ef` of
  10⁹ was an assertion failure in Debug and out-of-bounds writes in ReleaseFast.
- **`oversampling` is an unvalidated `double`.** `NaN <= 1.0` is *false*, so NaN
  passed the guard into `@intFromFloat`; `1e30` passed it and overflowed. Both
  are one field in a well-formed `QueryPoints`.
- **`CreateCollection` did not bound `size`** against the per-worker scratch, so
  a large dimension wrote past it on the first upsert.
- **`SETTINGS_MAX_FRAME_SIZE` could exceed the read buffer**, so a legal DATA
  frame between 256 KiB and 1 MiB could never be assembled, a silent permanent
  stall.
- **An oversized request killed the whole connection.** A body past the 1 MiB
  per-stream buffer returned a connection-level error, so the server sent GOAWAY
  and tore down every other in-flight stream, leaving the client with
  `transport error` and nothing else. It is now a stream-level
  `RESOURCE_EXHAUSTED` naming the limit. Invisible at d=128, the same 512-point
  batch is 262 KB there and 3.1 MB at the headline tier's d=1536, so only the
  headline dimension crosses it.

*Wrong answers, returned confidently:*

- **The GT differ reported 63 phantom "interior" disagreements** on SIFT1M by
  classifying on first *positional* mismatch rather than on where the neighbour
  set diverges, so a query with tied neighbours permuted at rank 3 looked like
  an interior disagreement when its only real difference was one id at rank 99.
  A plausible, specific, alarming number that was an artefact of its own
  measurement.

- **The graph path returned short pages under tombstoning.** Filtering deletes
  after truncating to `k` returned 8 of a requested 10 at 30% tombstones, while
  the exact path returned 10, the two paths disagreed on result *count*, not
  merely on order. Over-fetching by the expected deleted fraction does not fix
  it; the fraction among a query's nearest `k` is a small sample. Deletes are
  now filtered as results are admitted, against the full `ef`-sized set.
- **Manhattan on the SQ8 path returned a squared Euclidean distance.** The two
  metrics shared a kernel arm. With `rescore: false` that value goes out on the
  wire as the score.
- **k-means read its assignment array before writing it**, so how many
  iterations ran depended on what the allocator handed back, and the dependence
  was on reuse *across subspaces*, so it did not reproduce in isolation. That
  contradicts the determinism the codebook trainer promises.

- **Quantized search never scanned the pending tail**, so a quantized
  collection silently stopped returning points written after its last build.
  The W11 fix keeps the graph across writes and leaves the collection `.ready`
  until the tail passes `rebuild_ratio`, with `search` covering the difference
  by scanning past `graph.count`. That scan was added to the fp32 path only.
  Any non-exact query against a quantized collection therefore omitted up to
  10% of the most recently written points, and returned a full page of results
  while doing it, so nothing looked wrong.

  The cause is worth more than the fix. The strategy, exact against graph
  against quantized, was two booleans tested in two files, `!exact and
  index_state == .ready` in `collection.zig` and `quant != .none and
  !params.exact` in `handlers.zig`. Nothing named the set, so nothing noticed
  that a rule had been added to one member of it. It is now a `SearchPath` enum
  resolved once, and the tail scan is a shared function both paths call.

  §4 could not have caught it: W11 is the only workload that writes after
  indexing and it runs against the unquantized collection, while W6/W7/W8 never
  write. Found by reading for missing enums, not by measuring.

*Portability:*

- **`XGETBV` was issued before `OSXSAVE` was checked**, which is `#UD` on a
  pre-XSAVE part or a VM that masks XSAVE, a SIGILL at startup on a machine
  that would otherwise have run fine at the SSE tier. CPUID leaf 7 was read
  without checking the max basic leaf. *(no test: the host has OSXSAVE; the
  order is asserted in `dispatch.zig`'s comment.)*
- **A `defer` and an `errdefer` covered the same slice in `main`**, double-freeing
  every worker workspace on a `--host` parse failure or an `EADDRINUSE` bind.
  *(no test.)*
- **`zig build bench-isa` did not compile.** The forced-ISA arms built their
  `build_options` module by hand, and it lacked the `optimize_mode` option the
  native build had gained, so every arm failed the moment `main.zig` read it;
  the step §7.5 hangs off was dead until someone ran it. `build.zig` now
  derives the native build's and every arm's options through one helper.
- **An empty response body carried no `time`.** `time` is appended after the
  encoded body and the append was skipped when the body encoded to zero bytes
  (`CollectionOperationResponse{result=false}`, a delete of a missing
  collection), so the client decoded `time = 0.0` where Qdrant always sends
  it; test "time is appended even when the response body encodes to zero
  bytes" (`server.zig`).

*Documentation, which is where a wrong number is most likely to be read:*

- **Two tables in the README quoted different runs.** The generated block above
  the fold said 4,095 qps for W3; the hand-written table below it, and the copy
  of that table in `docs/comparison-sift1m.md`, said 3,985. Both were measured, neither
  was wrong, and nothing on the page said they came from different runs, so a
  reader comparing them would conclude one of them was a mistake. Fixed the way
  the header block was fixed once already: `compare.py` now generates both, the
  gate checks both, and there is no hand-maintained copy left to drift.
  `docs/comparison-table.txt` was a third copy, from a run older than either and
  linked from nowhere; it was deleted rather than regenerated.

  Worth noting what the disagreement actually was, since the noise floor now
  makes it answerable: every row with a measured floor moved *within* it. The
  two tables were not in conflict, they were one run apart, which is the
  distinction `regression.py` exists to draw.

---

**Found by the ISA review.** Three of these are in the distance layer, where a
wrong number reads as a finding about hardware.

- **The `baseline` arm was measuring compiler_rt, not SSE2.** `@mulAdd` is a
  request for a single rounding, so on a target without FMA (plain `x86_64`,
  the control arm of the §7.5 matrix) LLVM honours it by calling `fmaf` once
  per lane: `docs/asm/baseline/sm_dot_f32.asm` had 36 `call
  compiler_rt.fma.fmaf` in the hot loop, and `sm_euclid_f32`/`sm_dot_f32u8`
  the same. Any "AVX2 is N× faster than SSE2" read off that arm was mostly the
  cost of a function call per element. `common.mulAdd` now emits `@mulAdd` only
  when the build has hardware FMA and `a * b + c` otherwise; the baseline
  `sm_dot_f32` went from 476 instructions and 36 calls to 132 and none. The
  fp32 result on that arm now rounds twice per step where the FMA arms round
  once, which is the honest description of what SSE2 does. The f16 kernels on
  that arm still call `__extendhfsf2` per element, because F16C is genuinely
  not in the x86_64 baseline; that one is real.
- **Cosine normalisation followed Qdrant's *scalar* path; Qdrant on every
  comparison host takes the AVX one.** `CosineMetric::preprocess` (Qdrant
  1.19.0) branches at runtime: AVX+FMA and `dim >= 32` → four 8-lane fused
  accumulators reduced in `hsum256_ps_avx`'s fixed order; SSE and `dim >= 16`
  → four 4-lane mul+add accumulators reduced per register; else strict scalar.
  The `Σx²` feeds the discontinuous `|Σx² − 1| ≤ 1e-6` short-circuit, so a
  different summation order is a different *branch* for near-unit vectors, and
  the population of "systematically-off points" §8.3 warns about was being
  created here on real normalised embeddings. Measured at d=1536: the scalar
  and AVX sums disagree on the branch for 37 of 400 vectors swept across
  ±4e-6 of unit length, and the uniform unit vector at d=384/768/1536 is
  divided through by the scalar sum and left untouched by both SIMD sums.
  `norm.zig` now makes the same three-way choice from the same cpuid facts and
  transcribes each order intrinsic by intrinsic (order-preserving portable
  vectors, so bit-exact whichever ISA *we* were built for), checked against a
  scalar-indexed by-hand transcription with every tail residue.
- **`Pq4FastScan(L)` ignored `L`.** The doc said one AVX-512 `vpshufb` does 64
  lookups and `pq4_native` reported `lanes = 64`, but the kernel only ever
  issued the 16-byte form; `lanes_128` was computed and unused. `vpshufb` never
  crosses a 128-bit lane, so the wide form scores `L / 16` *subquantizers* per
  instruction, each lane against its own table, which the subquantizer-major
  layout gives for free as one contiguous load. Now implemented for 32 and 64
  bytes with a 16-byte tail, and every width is tested against the scalar
  reference at every `m_count mod 4`; the AVX-512 arm emits `vpshufb zmm`.
- **`-Dforce-isa=sse2` on a native build labelled its kernels `sse`** though
  they were compiled with the native feature set (FMA, EVEX encodings). The
  knob forces a *width*, not an instruction set; only the separate-binary
  matrix can isolate the ISA. `Tier.name` now says so at the definition and
  `Vtable.label()` appends the compiled feature set (`sse[4 f32 lanes+fma+evex]`).

---


**Found by the second review pass.** The theme is the same as the
first: each produced a plausible number rather than a failure. Grouped by what
the wrong number was.

*The published comparison:*

- **`CreateCollection` dropped bfb's top-level `hnsw_config`,
  `quantization_config` and `optimizers_config`.** Only the `VectorParams`-nested
  copies were read, and bfb (and the conformance harness) send them at the top
  level, so `--quantization scalar|binary|product-x16` and `--hnsw-m` built an
  fp32, m=16 collection that returned OK. strawmANN's W6/W7/W8 rows were fp32
  HNSW rows and conformance T4 measured an fp32 collection; the p50s already
  said so (246/291/257 µs against fp32 W3's 236 µs) and nobody read them that
  way. Both levels are parsed now with Qdrant's precedence, and
  `CollectionInfo.config` is written back so a client can see what it got.
- **`report.py` printed a ratio `compare.py` had refused.** The per-workload
  section divided the two qps cells itself, so a row shown as `-` in the
  throughput table (STALE, unequal recall, W11's policy) read `2.00x` a page
  further down, from the same run. `ratio_value` is now the one parse of
  `compare.Row.ratio`, and a refusal there is a refusal everywhere.
- **`compare.py` printed a ratio at unequal recall.** The README said it
  refused to; the ±0.01 guard lived in `results.py compare`, which the README
  pipeline never called, and `fullrun.py` invoked a `results.py record`
  subcommand that did not exist and wrote the README regardless. Ratios now
  need equal recall at the row's `ef`, a stamped `run.json` that agrees between
  the two arms, and a `licenses_comparative` differ row; the tables of the time
  rendered as STALE and unlicensed because they predated all three. Whether a
  current table carries them is checked by `compare.py --check-readme` rather
  than asserted here.
- **W0 compared HNSW against a plain scan.** 1M × d=4 is 16 MB across Qdrant's
  default ≤8 segments, under `indexing_threshold_kb`; Qdrant's `wait_secs` of
  3.02 s was bfb's polling floor. Every collection-creating row now passes
  `--segments 1` and both thresholds, and records them.

  **The first fix was sized for one tier and the bug came back underneath it.**
  1000 KB cleared the 1M tier; at 100k the same collection is 1.6 MB with ~0.8
  MB segments, so Qdrant left 27,500 of bench0's 100,000 points
  unindexed and W0 read 6.05x. The thresholds are now 1 KB and Qdrant's own
  minimum of 10 KB ("index everything, scan nothing", which is what strawmann
  does unconditionally) and an unindexed remainder is a reason for `fullrun`
  to refuse to publish, since `collections.json` had recorded
  `indexed_vectors_count` all along and only the report ever read it.
- **W11 could not run at full scale.** `--fbin sift1m.fbin --offset 1000000`
  makes bfb slice past a 1M-vector file; the smoke test passed because
  50k + 200k < 1M. It was also upload-*then*-search in one bfb process.
- **W5 sent sixteen copies of one query per batch** once it read dataset
  queries with `from-start`; `random-sample` draws per element.
- **Recall joins keyed on `ef` alone.** `headline.py` wrote the dbpedia cosine
  sweep to the same `recall.json` SIFT used, and the sink joined the quantized
  rows to the fp32 sweep. Files are keyed by dataset and collection, and the
  key is checked inside the file, not read off its name.
- **bfb's `Median qps` is a median of a 15 s EWMA** and understates rows that
  finish in under two seconds by 10-20% (only the faster engine's). Rows now
  carry `n / duration_secs`, keep bfb's figure beside it, and are flagged or
  re-run when short.

*The engine:*

- **uint8 + cosine returned 0 for every point.** Ingest correctly stored raw
  bytes (Qdrant does not normalise bytes) but the query path normalised by
  metric alone, and a unit vector truncated to u8 is all zeros. Results were
  the lowest ids, no error. The query path asks the datatype now, and an e2e
  pins Qdrant's `cosine_similarity_bytes` values.
- **A small append after Green left the collection Yellow forever.** Below
  `rebuild_ratio` no rebuild starts, `status()` compared counts, and the only
  build trigger was `Collections/Get`, so bfb's `wait_index` hung and
  `--skip-wait-index --search` brute-forced every query. Green now means "the
  graph is published and the tail it serves exhaustively is small", which is
  what Qdrant's Green with an unindexed small segment means, and a query
  starts the build.
- **Overwriting an id never re-indexed it.** Edges and quantized codes kept
  the old vector; `needsRebuild` counted only appended points. Overwrites are
  counted, the quantized row is re-encoded in place, and the rebuild triggers.
- Smaller drifts from Qdrant, each now tested: default `hnsw_ef` was 128, not
  `ef_construct`; `score_threshold` was inclusive; `limit=0` became 10 instead
  of INVALID_ARGUMENT; binary `encoding` and scalar `quantile` were ignored;
  payloads were accepted and discarded; `time` started at dequeue rather than
  arrival; delete-missing returned true and create-existing recreated.
- Three races: the scroll order was freed under a reader, the quantized store
  was published as a multi-word write, and the parallel builder read
  `entry_point` and `max_level` separately.

*The conformance gate:*

- **T1 never compared ids.** Scores were compared positionally and ids appeared
  only in the diagnostic string, so an id-mapping bug with correct scores would
  have licensed every performance number. And when it did compare them, the
  last returned id trivially tied its own cut, so the oracle check excused it
  whatever its score and the score deviation was only consulted when the
  engines disagreed: both engines returning the same wrong last id passed.
  Agreement between the engines is not agreement with the truth
  (`t1_fails_when_both_engines_agree_on_a_wrong_last_id`). T2 and T4 were hard-wired
  `passed: true`, an empty comparison passed, and the metamorphic checks were
  printed but entered neither the tiers nor the hash. `recall_one` excused an
  engine that returned fewer than `k` results and trusted the engine's own
  score for the tie clause; the relevance ε defaulted to 24× the calibrated
  value.

*The cost model:*

- **"DRAM-cold" was a throughput number.** The timed loop's next address did
  not depend on the previous result, so the core overlapped misses: hamming
  d=128 read 20 ns against a measured 116 ns DRAM latency, and §5.2's
  dependent-chain model was graded against it. A dependent column exists now
  and the published tables say which one they are.
- **Nothing was pinned on a heterogeneous host** (Zen 5 with 16 MiB L3 beside
  Zen 5c with 8 MiB), so an "L3-resident" set was the whole L3 of one core
  type, and one published cell ran at 1.17 effective frequency, 3.5× its
  neighbours. The arena was random bytes reinterpreted as f32, i.e. NaNs and
  denormals; the bandwidth probe was one FADD chain; huge pages were requested
  and never verified; perf counters were not a group.

## Found in Qdrant 1.19.0, by §8.6's metamorphic properties

**Neither reproduces any more, and nothing here has run 1.19.0 since.** Both
were found against the pinned v1.19.0 image and held for strawmANN. Every
conformance record now on disk (16 runs across 1.19.1-dev and 1.19.2-dev,
sift1m and both dbpedia tiers) reports `prefix_property=ok/ok` and
`idempotent_upsert=ok/ok`. So either upstream fixed them or the conditions
changed, and the two are very different conclusions: the mechanism guessed at
below is segmentation, and the runs since hold one populated segment where the
original held several, which would leave no cross-segment merge to disagree with
itself. Settling it needs one differ run against a 1.19.0 container.

They are kept because they were real, and because the shape is the point: both
were found by properties rather than by comparison, so nothing about them
required a second engine to be *right*, only a definition of what a correct
answer looks like. Neither is a performance claim.

**`prefix_property`: the top 10 is not a prefix of the top 20.** Asking for more
results reorders the ones already returned. Query 5 of SIFT1M:

```
top_10   [187470, 67875, 220473, 460733, 896005, ...]
top_20   [187470, 220473, 67875, 460733, 896005, ...]
```

Positions 2 and 3 swap. §8.6 lists this as a property an ANN engine should hold
because a client paginating by widening `limit` sees results move between pages
that nothing has written to. The likely mechanism is the same segmentation that
makes `ef` incomparable between the two engines (`docs/comparison-sift1m.md` §2): a
merge across segments at one `limit` need not agree with the merge at another
when scores tie or nearly tie.

**`idempotent_upsert`: re-upserting identical points changes the results.**
Upserting the same ids with the same vectors is by §8.6's reading a no-op, and
the query answer moves afterwards. This is the property whose violation is
hardest to reason about downstream, because it makes a collection's answers
depend on ingest history rather than on its contents.

Not reported upstream, and no longer reportable from a run: the differ prints
each as a `FINDING` line with the offending query and lists, which made a bug
report a copy-paste, but no current build produces one. A report would send a
Qdrant developer after behaviour their own version does not have.
