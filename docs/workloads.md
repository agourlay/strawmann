# W0-W13, re-derived against bfb `dev`

§4's table is abridged. This is the derivation: what each row runs, what changed
from the pre-`dev` version, and why.

The executable form is **`bench/harness/workloads.py`**, where the table is a
list of `Workload` objects rather than shell. It was two shell scripts -
`run-w0-w13.sh` and `run-workload.sh`, each encoding §4's matrix independently,
which is exactly the drift risk this document exists to prevent. Shell is now
reserved for setup: `bench/setup.py` pokes `/sys` and needs root;
`conformance/datasets/datasets.py` wraps curl for the transfer and does the
manifest and digest work in Python.

Pinned reference, per §8.9 and §4.1:

```
qdrant/bfb  branch dev  HEAD fc6632e5  "paginated serverless ListCollections
                                        and KeywordIndex builder (#176)"  2026-09-08
```

Advanced from `0c1aafee` (#172, "--rps reaped at most one completion per
rate-limiter tick", 2026-08-27), whose fix this still carries. The three
commits between them add a serverless multi-collection mode in new files, and
the shared code they touch keeps the single-collection path these rows measure:
`processor.request_count`/`request_size` default to the arithmetic `stats.rs`
used to inline, and `retry_with_clients` only became generic over the client
type. `workloads.BFB_PIN` records the reading; #176's "KeywordIndex builder" is
that module's own, not the field index W12 uses.

The pin lives in `workloads.BFB_PIN` and `check_bfb_pin` refuses a run whose
checkout is not on it: `fullrun.py` asks in its preflight, so a stale checkout
costs a second rather than a whole run. W0-W13 below were derived against
`8b6dec0a`; #172 changed how `--rps` reaps completions, which is what made the
open-loop arms measurable at all (findings 32), and did not change the rows.

---

## 1. What `dev` changed about the CLI

**Subcommands now exist**, `upload`, `search`, `scroll`, `schema`, each driven
by a YAML config. **Omitting the subcommand keeps the legacy flag-driven
benchmark**, which is what §4's table uses and what W0-W13 continue to use. The
YAML path is required only for dataset-sourced query sets (§4.1) and for payload
shapes more elaborate than the `-k/--int-payloads/...` flags express.

Other `dev` commits that touch the surface this harness depends on:

| commit | why it matters here |
|---|---|
| `0a88bcdb` Unified `--json` results document (#150) | `bench/harness/results.py` parses bfb output; the JSON document is a better ingest path than scraping stdout |
| `e2317d59` track qdrant-client dev, expose 1.19 collection & search options (#160) | consistent with our pinned `qdrant-client =1.19.0` |
| `47ac3de2` `--shard-key` in search/query/scroll (#159) | sharding is a §1 non-goal; we must keep returning `UNIMPLEMENTED` |
| `9e5a350c` scroll sub-command (#151) | reaches `/qdrant.Points/Scroll`, which §2 lists as phase 2 and which is implemented; W13 runs the legacy `--scroll` flag path, see §3 |
| `7a0153d3` split large modules, dissolve `common.rs` (#148) | **`src/search.rs` no longer exists**, which spec §4.1 used to cite by name |
| `a4f10962` fetch http(s) `type: file` vector sources (#149) | dataset fetch has its own download path |

**The short flags in the old table all still work.** They are derived by clap
from the field names (`#[clap(short, long)]`), so `-n -d -b -t -p -c` remain
`--num-vectors --dim --batch-size --threads --parallel --connections`. Most are
`global = true` and therefore also valid with a subcommand; **`-d/--dim` is
not**, it is top-level only, which matters if a row is ever ported to
`bfb search --file`.

**Our fbin artefacts load.** `src/fbin_reader.rs` reads a little-endian
`i32` pair then f32 payload; `conformance`'s writer emits `u32`. Those are
byte-identical below 2³¹ and every dimension and count we produce is far below
it, so `--fbin base.fbin` works on files we generate. A base set of ≥ 2³¹
vectors would read back negative and be rejected, not a limit we can reach.

New flags that matter to §7's methodology:

| flag | why it matters |
|---|---|
| `--rps <f64>` | fixed-rate (open-loop) load; supersedes `--parallel` when set |
| `--json <path>` | the unified results document (#150), config plus every phase |
| `--jsonl-searches` / `--jsonl-updates` / `--jsonl-rps` | per-request timings, viewable with `qdrant/mri` |
| `--p9 <n>` | how many nines to report |
| `--memory-vectors` / `--memory-index` / `--memory-quantization` / `--memory-payload` | explicit residency placement, superseding the `--on-disk-*` booleans |
| `--retry` / `--retry-interval` / `--ignore-errors` | error handling, which silently changes what a mean latency means |
| `--prevent-unoptimized` | keeps unoptimized segments out of search |
| `--indexed-only` | skips un-indexed segments |
| `--hnsw-inline-storage` | stores HNSW links inline with vectors, a different layout to compare against |
| `--connections` | client-side connection count, distinct from `--parallel` |

## 2. Flags common to every row

Not repeated per row in §4's table:

```
--json results/WNN.json      # parse this, never stdout
--retry 0                    # errors surface; they do not average away
--timeout 60                 # per request; `workloads.BFB_TIMEOUT_S`
--p9 3                       # p99.9
```

(`workloads.py`'s `common` list, in that order, plus `--uri`. The
`--jsonl-*` per-request logs are not requested: nothing in the harness reads
them.)

Every collection-creating row also passes, explicitly, `--distance Euclid
--segments 1 --indexing-threshold 1 --full-scan-threshold 10
--on-disk-payload false` (`workloads.CREATE`; kilobytes, as Qdrant's API takes
them; `false` is what spec §2 lists under the documented semantic gaps, and the
harness passed `true` against it until this was checked). Qdrant's own defaults are `default_segment_number: 0` (auto, up to the
CPU count) and 10,000 KB for both thresholds, under which W0's 1M x d=4 = 16 MB
collection, split over eight segments, never reached the indexing threshold:
Qdrant answered W0 from a plain scan while strawmANN answered from its graph,
and the row's ratio measured that. The thresholds are 1 KB and Qdrant's minimum
of 10 KB ("index everything, scan nothing") rather than a size chosen to
clear one tier's collections: 1000 KB cleared the 1M tier and the same bug
returned underneath it at 100k, where the d=4 collection is 1.6 MB and its
segments fall under 1000 KB again. An absolute size cannot express the intent
when the collection it governs scales with the dataset. strawmANN accepts the
same flags (inert for the thresholds: it always builds the graph).

**`--retry 0` is deliberate.** The default is already 0, but it is pinned
explicitly because a retried request is a request whose latency was measured
twice and whose failure was not recorded. §7 wants the failure. Likewise
`--ignore-errors` stays off: a run with errors is a void run, not a run with an
asterisk.

**Recorded per run even when defaulted:** `--datatype`, all four `--memory-*`
placements, `--hnsw-inline-storage`, the segment count and both thresholds
above, the metric, each row's query source (dataset `from-start`, dataset
`random-sample`, or random) and `ef`, `UPLOAD_N`/`W11_N`, and the bfb pin, in
`run.json` under `harness` (`workloads.harness_stamp`). Each changes what
Qdrant is actually doing, and a result row that does not carry them cannot be
compared to another one. §7.1 already requires an environment hash; these
belong to the same category, and `compare.py` refuses to ratio two labels
whose stamps disagree or that lack one.

**`--datatype Turbo4` is never used.** §6.7 excludes the Qdrant-proprietary
`turbo*` variants and strawmann returns a clear error for them; benchmarking
against one would compare against something we deliberately do not implement.

## 3. The rows

### W0, the d=4 floor

```
W0-upload:  -n UPLOAD_N -d 4                       (create, load, wait green)
W0:         --skip-setup -n QUERIES --search -p 1
```

Two rows, not one, and no `--skip-wait-index`: §5b below records that carrying
the flag made W0 a brute-force scan of a million vectors at 34.9 rps, and the
upload is split off so the search half runs against an indexed collection with
`QUERIES` (50,000) requests rather than `-n`'s upload count. `-p 1` is the
explicit part. Four dimensions makes the *distance*
irrelevant; `-p 1` makes the number a per-request latency rather than a
throughput. It was called the "transport plumbing floor" on the assumption
that what remained was RPC overhead. Measured (findings 28), what remains
is the HNSW traversal itself over 1M nodes: 76% of the server's CPU in
`searchLayer`, the heaps and the visited set, 24% in the kernel, and the
server-side `time` was 172 µs of a 263 µs round trip. So this row is the
floor of *graph traversal*, which every other search row also pays, and
the RPC path is the small remainder. Both engines do the same traversal, which
is why the row sits near 1.0x.

### W1, ingest throughput

```
--fbin base.fbin -n UPLOAD_N -d DIM -b 100 -t 8 -p 8
```

`-d DIM` is the dataset's width (128 for sift1m, 1536 for dbpedia), not a
constant. For the headline tier there is now an alternative, `bfb upload
--file <yaml>` with a parquet dataset source, since `dev` reads parquet
directly, which would skip our fbin conversion entirely. We keep `--fbin`
because §4.1's split means our converted artefact is the one the fp64 oracle
also reads, and using a different corpus path for load than for ground truth
reintroduces exactly the discrepancy `conformance/datasets/parquet_convert.rs`
exists to avoid.

### W2, index build time

W1 **without** `--skip-wait-index`. The measurement is time-to-Green, which is
the §2 status flow, not a bfb-reported number.

### W3, search, fp32, single query

```
--skip-setup --search --search-limit 10 --search-hnsw-ef 128 -p 1
```

`--skip-setup` replaces the old `--skip-upload`: it implies `--skip-create
--skip-upload --skip-wait-index`, which is what a search-only phase actually
wants and removes three ways to get the phase boundary wrong.

### W4, search, fp32, saturating, **now two arms**

```
closed loop:  W3 with -p 64 -t 16 -c W4_CONNS          (W4)
open loop:    W3 with --rps R -t 16 -c W4_CONNS, R ∈ {0.5, 0.7, 0.9} × saturation
                                                       (W4-sat50/70/90)
```

`W4_CONNS` defaults to 2 (env override), not 8: `-t 16 -c 8` wants ~128
sockets and lost a whole run to a server with fewer slots (§5b).

**The offered rates were constants (500, 1000, 2000) and are now §4's
fractions.** The constants were chosen so the open-loop rows could be compared
across engines and runs without first agreeing on what "saturation" was, which
is a real property and is why the deviation was recorded rather than hidden.
What it cost was the arms themselves: strawmANN saturates above 22,000 qps, so
`--rps 500` offered **2% of capacity**. No queue forms at 2% of capacity, so
what the row measured was the load generator's own scheduling. The report had
to print *"15.9 ms of the client's 16.7 ms p50 is not server time"* beside
every one of them and say in its lede that neither loop was a usable latency
result. §7.4 wants `--rps` to be *the* instrument for a latency claim, and at a
fixed 500/s it could never be one. A constant that makes the rows comparable
and meaningless is worse than a fraction that makes them mean something.

The comparability the constants bought is kept as a flag rather than thrown
away. `workloads.py run <uri> <label> --rps-reference <qps>` pins the
saturation both engines' arms are a fraction of; pass the **slower** engine's,
and both are offered the same absolute rate, which is a fraction of a measured
number *and* a load both engines can serve. Without it each engine uses its own
measured W4, the two arms sit at different offered loads, and the report says so
above the latency table rather than letting the percentiles be read across
(`report.open_loop_mismatch`). The row records `rps_target`, `rps_fraction`,
`saturation_qps` and whether the reference was `pinned` or `own`, so which of
the two happened is a property of the artifact and not of anyone's memory.

The rows are named for the fraction, not the rate: a row called `W4-rps500`
that offers 11,420/s would describe itself as the opposite of what it measures,
which is the failure mode `W0`'s rename already fixed once. Archived runs keep
their `W4-rps*` ids and `report.VARIANTS` still names them, so a report over an
older result set reads correctly; those rows really were a constant 500/s.

There is no default rate. A `W4-sat*` row with no measured W4 in the run, no
W4 in the label's existing `rows.json`, and no `--rps-reference` is filed
`FAILED` naming what is missing, rather than run at whatever bfb would pick.

This is the most substantive change `dev` allows, and it is a correctness fix to
the methodology rather than an extra data point.

`--parallel N` is a **closed loop**: N requests are in flight, and a new one is
sent only when one completes. When the server slows, the client sends more
slowly, so the requests that *would* have arrived during the stall are never
issued and their latency is never recorded. The measured p99 is therefore an
underestimate, and it is underestimated *more* for the slower engine, which is
the direction that flatters whichever engine we are advocating for. This is
coordinated omission, and it is the standard way a latency comparison becomes
wrong while looking careful.

`--rps R` sends at a fixed rate regardless of what the server is doing, so the
queue builds and the tail is real.

The two arms answer different questions and both are needed:

- the closed-loop arm answers **"what is peak QPS"**, a throughput claim;
- the fixed-rate arms answer **"what is the latency at a load both engines can
  serve"**, and only this belongs next to a p99.

§8's rule that the publishable statement is *"equal recall at lower latency"*
requires the second: comparing p99s taken at each engine's own saturation point
compares two different offered loads.

### W5, search batched

W3 with `--search-batch-size 16`, and `strategy: random-sample` in its query
config rather than W3's `from-start`. bfb's config path builds every element of
a batch from the same `req_id` (`search/from_config.rs`), and `from-start` maps
that to one row of the query file, so with `from-start` a 16-query batch was
one query sixteen times: a warm-cache measurement of batching overhead, not of
batched search. `random-sample` draws each element independently from the
held-out set, which keeps the dataset's queries and gives distinct ones per
batch, at the cost of point-for-point comparability between engines (the two
runs see the same distribution, not the same sequence). The flag path would
also give distinct queries, but uniform-random ones far from the data. The row's
note says which was used, and its p50/p99 are **per request of 16 queries**,
which the row's `per-batch latency` note states.

### W6, quantized: scalar

W3 with `--quantization scalar --quantization-rescore true`, split into
`W6-upload` and `W6` (§5d). `--memory-quantization` is *not* passed: the codes'
residency is left at each engine's default and `run.json` says so
(`memory_placement: engine default`). §5.4's working-set argument is about
exactly this residency, so a placement-controlled re-run would have to add it.

**SQ8 is not the same quantizer on both sides.** strawmANN's scalar quantizer
uses 256 levels and order-statistic quantiles; Qdrant 1.19 uses 127 levels
and its quantile is effectively min/max (`src/quant/scalar.zig` module doc).
Both quantise the query (strawmANN's did not in the 2026-08-18 run; findings
29). A recall-matched W6 row therefore compares two different SQ8 encodings
at equal recall, not one encoding on two engines.

### W7, quantized: binary + oversampling

```
W3 with --quantization binary --quantization-oversampling 4 --quantization-rescore true
```

**`--quantization-rescore true` added deliberately.** Our own measurement is that
oversampling without rescore does nothing at all, recall@10 is flat at 0.173
from 1× to 16× oversampling, because widening the candidate set changes nothing
if the final ranking is still by binary score. An oversampling sweep with
rescore off measures a knob that is inert, and would report that as a finding
about oversampling. §6.7's insistence on the *combined* curve is the same point.

### W8, quantized: PQ

W3 with `--quantization product-x16`. Unchanged.

### W9, exact / brute force

```
--skip-setup -n EXACT_QUERIES --search --search-exact --search-limit 10 -p W9_PARALLEL
```

Not W3 with a flag: 2,000 queries rather than 50,000 (a brute-force scan of
the corpus per query), at `-p 8` rather than W3's `-p 1`, and no `ef` because
there is no graph in the path. Still the most SIMD-revealing full-stack row per
§4's ISA axis, measured as an eight-way concurrent scan and read as such.

### W10, recall control, **changed**

```
--search-hnsw-ef ∈ {32,64,128,256,512}, -p 8, latency only
```

Concurrent, not W3's single-client basis: the sweep measures throughput at
each `ef`, and its recall comes from `conformance relevance` at the same `ef`.
The five `W6-ef*` rows are the same sweep over the SQ8 collection, added so
the quantized frontier has a throughput half (`workloads.py` explains why the
top of that frontier is a recall only one engine serves).

Was `--search-quality` at the same `ef` values. §4.1 rules that out: it scores
each engine against *its own* exact search, so two engines' numbers are not
comparable to each other, and it cannot validate the exact path itself.

The dataset-recall path `dev` added is also not used here, for the three reasons
in §4.1, it is a strict id-set intersection with no ε-tie handling, it has no
ground truth for the parquet tier, and it would score against a third party's
ground truth rather than our fp64 one.

So W10 splits: **bfb sweeps `ef` and reports latency; `conformance/relevance`
reports recall at the same `ef` values.** The two series are joined on `ef` to
produce the recall-vs-latency frontier. Neither tool reports the other's axis -
which is §4.1's rule, made operational.

### W11, mixed read/write

Two concurrent bfb processes against `bench2`: one appends `W11_N` points with
`--offset UPLOAD_N + W11_STEADY_N --skip-wait-index` (W11-steady, the row
before it, appends its own 5% of the corpus at `--offset UPLOAD_N`, below the
rebuild threshold, so W11's offsets follow), the other searches at `ef=128, -p 8`
with the dataset's queries while that happens. The row's qps is the search's;
the append rate and how long the two overlapped are recorded beside it
(`background_pps`, `overlap_s`, and a note when one finished well before the
other). Until this was two processes it was one bfb invocation that uploaded,
skipped the wait, and *then* searched, i.e. search after an unwaited append,
described as mixed.

The appended vectors are **synthetic** (bfb's random generator, `-d 128`).
They cannot come from the corpus: `sift1m.fbin` holds exactly `UPLOAD_N` rows
and bfb reads row `offset + i` from it, so `--fbin` with `--offset UPLOAD_N`
slices past the end of the file and panics on the first batch at full scale.
Recall is not measured on W11 in any case: the collection is being mutated
under the search, so no sweep describes it and the row carries no recall cell
and no ratio; the row says so as its `ratio_policy` ("search-during-write; no
recall join") rather than reading as a sweep someone forgot. Two more things
the row records: `write_overlap_pct`, the share of the search phase (bfb's
`duration_secs`) the append was in flight for, below 90% of which the row is mostly a search of a
quiet collection and `compare.py` refuses its ratio with the note; and it is
excluded from `--min-duration` re-runs, because a second pass would re-append
the same id range (an overwrite) into a collection the first pass already
changed.

This is the row that found the use-after-free where a rebuild freed a graph a
search was traversing, so it earns its place.

### W12, filtered search, keyword index

`-k 100` at upload, then two filtered search rows. Both engines create the keyword index before the upload
(bfb's `create_field_indices`, `CreateFieldIndex` on field `a`) and answer the
filtered queries from it. Payload filtering is phase 3 of §2's surface and
milestone M7; strawmann implements it since 2026-09-03 (`src/core/payload.zig`):
the payload is stored as the wire bytes of its map entries, the index is a
posting list per value, and a filter is applied on every search path. The
graph traversal (as an admission predicate, the way tombstones are), the
exact scan, the pending tail and the quantized stage 1.

**Both engines choose the path by selectivity, by different rules.** Qdrant
picks it by the filter's estimated cardinality against `full_scan_threshold`
(`read_view/dispatch.rs`, verified in source; `decisions.md`): below it,
`search_vectors_plain` scores the matching points; above it, the graph is
traversed under the filter. strawmANN compares the two costs directly from the
exact count its index gives (`handlers.filteredPlan`): it scans the matching
set when that is cheaper than the walk, and above a selectivity of `1/m0`
walks ACORN-1 style, hopping through rejected neighbours without scoring them
(15bddf7). Either way which path a query takes is a function of its
selectivity, which is the reason for two grades rather than one. At `-k 100`
over 200k points a keyword covers about 2,000 matches: `W12-sel1` scores that
set directly on both engines' cheaper side, so strawmANN's throughput and
recall are flat across every `ef` (5,325 q/s on sift1m's `perf-0926`, 1,873 on
dbpedia-openai-1m's `perf-0925`, recall 1.0000) because a scan cannot be
steered by `ef`. `W12-sel10`, at ten keywords and about 20,000 matches, walks
the graph under the filter, so its cost and recall move with `ef`: 811 q/s at
`ef` 128 on dbpedia-openai-1m `perf-0925` and 1,269 on sift1m `perf-0926`.

An earlier draft of this paragraph said "both score the matching set directly"
at `-k 1000`, which was true of the single 0.1% grade the row then had and is
true of neither pair member's neighbour. The row until
2026-09-03 carried `--skip-field-indices` and measured a Qdrant full scan
against a strawmann refusal; `compare.py` still reads the index back before it
licenses the row, point 1 below, because the flags say what was asked, not
what was built.

#### What W12 had to become before it was a filtered-search row

**All four hold as of 2026-09-08.** M7's exit criterion was literally "W12
runs", which the row satisfied while measuring a scan. That criterion was too
weak, and the four things below replaced it. Three are lifted from [Percona's
`vector-bench` methodology][percona], which is a benchmark of SQL engines and
shares almost nothing with this one except the traps.

[percona]: https://www.percona.com/blog/benchmarking-vector-indexes/

**1. The index is a precondition of the row, and the harness checks it.**
The first of the four to be built, and for a while the only one.
`Workload.needs_payload_index` marks the row; `compare.py` asks two questions in order.
Did the run itself deny the engine an index (`--skip-field-indices`, which no
read-back can undo), and does the engine *say* it built one, read back into
`collections.json` by `collection-info`'s `payload_indexes` after the row ran.
Intent settles the first and cannot settle the second: asking for no index
guarantees there is none, asking for one guarantees nothing. An unknown answer
is refused like an empty one, because an index state nobody read back is not
evidence that an index existed.

**2. Two selectivity grades, not one.** Built as `W12-sel1` and `W12-sel10`,
two rows on one `bench12`.

The condition is a *keyword* rather than the integer `tag` this paragraph
originally specified, because the load generator cannot express the latter: its
integer filter is `tag >= random(0..range)`, whose selectivity is uniform in
[0,1] per query rather than fixed, so a row built on it measures a different
regime on every request. With `-k 100` at upload each keyword covers ~1% of the
points and any ten of them ~10%, which is the pair this asked for. The
substance (two fixed grades, separately reported) is unchanged; the mechanism
is not the one written here. One filtered number hides which
regime it came from, and the regimes are not the same measurement: at 1% the
matching set is small enough that walking it beats traversing the graph, at 10%
it is not, and an engine that picks well at one selectivity and badly at the
other reports a single figure that is true of neither. Qdrant's dispatch already
turns on exactly this threshold, so a row that does not vary selectivity is
measuring one arm of a branch and calling it the function.

**3. Ground truth is recomputed per condition.** §4.3's cached fp64 k=100 is
*unfiltered* and does not score a filtered query: the neighbours of `q` among
all points are not the neighbours of `q` among the points matching `tag ∈ S`.
The oracle computes the same thing it always computes (fp64, ε-aware ties,
`MRDE` alongside the id sets) over the restricted id set, and caches keyed by
(dataset, metric, k, condition) rather than by (dataset, metric, k). Note that
the shipped neighbours in `ann-filtering-benchmark`'s `tests.jsonl` are filtered
ground truth for *their* conditions at k=10 (25 for h-and-m), not ours
(`datasets.md`), so they are a cross-check on a matching condition and never a
substitute.

As built, with one deviation: `oracle::cache_key_for` does key by (dataset,
metric, k, condition), but the file is written under the *run's label* and is
not reused across runs. The condition's membership is a property of one upload.
Bfb redraws the keywords each time, so a cross-run cache would answer for a
collection it never saw. The key still earns its place: it stops the two grades
of a single run sharing a file.

**4. The row reports how many points came back, and how many could have.**
A filtered page shorter than `k` is ambiguous between two unrelated causes (the
index searched too narrowly and needs a wider `ef` or an iteration, or the filter
simply matches fewer than `k` points) and recall alone cannot separate them: an
engine that returns all 6 matching points scores 0.6 at k=10 while being exactly
right. So the row records `returned / asked` and `|matching set|`, and
**recall is scored against `min(k, |matching set|)`**. Without that denominator a
restrictive filter caps recall below 1 for a reason that has nothing to do with
the engine, and the frontier in §7.4 reads as a regression.

As built, with one deviation: this paragraph asked for the pair *per query* and
the sweep records it **per sweep**, `returned` and `asked` summed over the
queries, beside the one `n_matching` the condition selected. One condition per
sweep makes `|matching set|` a single number rather than a series, and the
per-query signal that would have been lost is kept as `short_lists`, the count
of queries answered with fewer results than the matching set could supply.
Recording 1,000 identical pairs would say no more than their total and the
count of exceptions.

#### What the first measurement said

Both grades, strawmANN alone, `bench12` at 200,000 points, 1,000 of the 10,000
held-out queries, `ef` swept 32-512 against ground truth restricted to each
condition. A development reading from before the published harness (one
engine, no gate, an earlier build); its q/s column is not comparable with the
published rows above, and the recall columns are what it established:

| grade | matching set | recall@10 at ef=32 | at ef>=64 | q/s |
|---|--:|--:|--:|--:|
| `sel1`, one keyword | 2,082 (1.04%) | 1.0000 | 1.0000 | ~14,800 |
| `sel10`, any of ten | 19,984 (9.99%) | 0.9970 | 1.0000 | ~850 |

The grades are two measurements rather than one, which is what point 2 was for,
but not in the way it predicted. Recall separates them barely and only at
`ef=32`; what separates them is the **dispatch path**, visible as 17x in
throughput. At 1% the engine walks the 2,082 matching points, so `ef` cannot
matter and the row is cheap; at 10% it traverses the graph under the filter, so
`ef` does matter and the row costs seventeen times as much per query. A single
row would have reported one of those regimes and been true of neither.

Recall reaching 1.0 by the row's own `ef=128` at both grades is worth stating
plainly: on SIFT1M these grades differentiate cost and code path, not accuracy.
A corpus where filtered recall is actually hard would need either a harder one
or a narrower `ef`, and this table is the evidence for saying so rather than an
assumption either way.

Two properties of the harness that this run happened to demonstrate, both
load-bearing:

* The matching set is **redrawn on every upload** (2,082 here against 2,010 on
  the previous run of the same condition) because bfb assigns keyword payloads
  from an unseeded RNG. The restricted ground truth therefore lives under the
  run's label and is not cached across runs; a cache keyed by (dataset, metric,
  k, condition) would be read back against a collection whose keywords had
  changed, and the failure would look like a recall regression.
* `relevance` refuses a filtered search scored against §4.3's unfiltered cache,
  and an unfiltered search scored against a restricted truth. Both were
  refused in practice while this was being built, the second by the
  `query_checksum` guard catching a truth computed over a truncated query set.

### W13, scroll / pagination, **new**

```
bfb --scroll --collection-name bench2 --skip-setup -n 50k -p 8 --search-limit 10
```

`dev` promoted scroll from a bare `--scroll` flag to a sub-command with its own
YAML config and three traversal modes (#151). The legacy flag still works and is
unchanged, and **it is what W13 runs today** (`workloads.py`: `--scroll` on
the flag path, `scroll` mode only, no `--file`). The sub-command's three modes
are the interesting axis and are described below because they are the row's
intended shape; they are not yet in the table.

**Why it earns a row.** W13 is the only workload that touches storage and the id
map with **no vector comparison at all**. Every other search row measures the
distance kernel and the memory system together, and §6.6.3's whole point is how
hard those are to separate. Scroll removes the kernel from the picture: what is
left is id → offset resolution, the tombstone check, and page assembly. §5's
memory model makes predictions about that path which no other row can isolate.

**The three modes measure different things and all three are wanted (only
`scroll` is run):**

| mode | what it does | what it exposes |
|---|---|---|
| `scroll` | fetch the first page matching the filter; every request restarts at the top | pure page-one cost, fully cached after the first request, the floor |
| `sequential` | resume from the previous page's cursor, opening at a random point | a real traversal, so the id map is walked rather than hit repeatedly at one spot |
| `sample` | vector-less `query` with `sample: random` | random access across the whole id space, the anti-locality case |

`sequential` is the one to watch. Its walks deliberately open at random offsets
so concurrent walkers cover different stretches, a fix that landed on `dev` as
`fix: scroll sequential walks were racing and re-reading page one (#153)`. Before
that fix every concurrent walker re-read page one, which measured cache
residency rather than traversal. **A W13 run against a bfb older than #153 is
measuring something else**, which is a concrete reason the pin in §4.1 matters.

**Two arms, like W12:**

- **unfiltered** (`filters: []`), a genuine two-engine comparison. Scroll
  without a filter needs no payload storage: it is id-ordered pagination, which
  strawmann can implement within §1's non-goals.
- **filtered** (keyword/integer conditions), Qdrant-only, and recorded as such.
  Payload filtering is phase 3 (§2) and M7 (§10, optional), not a §1 non-goal;
  strawmann answers
  `UNIMPLEMENTED`. `sample` mode likewise is a Qdrant sampling feature we do not
  implement.

**Status against strawmann: implemented.** `/qdrant.Points/Scroll` decodes
`ScrollPoints`, pages in point-id order and returns `ScrollResponse` with a
resume cursor. Measured on the SIFT1M collection: **1,000,000 points in 1,000
pages of 1,000, in 0.18 s (5.6M points/s)**, ids strictly ascending, every page
decoded by `qdrant-client`'s prost.

Paging is by *point id*, which nothing else in the engine needs, §3 stores
points in arrival order and every other read path is by hash or by graph. Naively
walking arrival order would be correct only when ids happen to be assigned
sequentially, which is exactly what bfb's default upload does, so the bug would
pass every test we would naturally write and then hand back silently wrong pages
to anyone using UUIDs or a sparse id space. The id order is therefore
materialised (sorted offsets, cached per collection, dropped on any mutation).

The filtered and `sample` arms remain Qdrant-only: payload filtering is phase 3
and `order_by` needs payload storage, so both return `UNIMPLEMENTED` naming the
construct rather than quietly returning an unfiltered page.

Implementing it turned up two bugs, both of the kind that survive because the
output still looks like a valid response.

**The `time` field is not always field 2.** The worker appends the honest
server-side timing after the handler returns (§2 requires it be measured at the
RPC boundary), and it wrote field 2 unconditionally, correct for
`QueryBatchResponse`, `PointsOperationResponse` and `CollectionOperationResponse`,
which is every response §12 lists for M1. `ScrollResponse` is
`next_page_offset=1, result=2, time=3`. Appending field 2 there does not mislabel
the timing: it injects a fixed64 **into the middle of the repeated `result`
list**, so a conformant decoder reads a garbage length for the next element.
Our own decoder happened to tolerate it because the corruption landed after the
last element read, which is why the e2e assertion now checks the field number on
the wire rather than only that the page parses. `Completion.time_field` is now
per-response.

That one is worth dwelling on: the bug was introduced by a *correct*
generalisation ("every §12 response puts time at field 2" was true when written)
that a later message invalidated. The comment asserting the uniformity was
accurate and became false without anything touching it.

The second: `unimplementedRpc` carried a doc
comment claiming the message "names the *actual* path rather than a generic
string" while the body did `_ = path;` and returned a fixed string citing "§1
non-goals". Both halves were wrong: the client could not tell which of the
several unimplemented paths it had hit, and a deferred phase-2 RPC is not a §1
refusal, so the message sent the reader to the wrong section to learn that a
scheduled feature was permanently absent. It now formats the real path and
distinguishes deferred RPCs from non-goals.

That in turn exposed a second one: `grpc-message` is percent-encoded per the
gRPC spec, and `§` is U+00A7, two non-ASCII bytes. **Every** UNIMPLEMENTED
message strawmann sends therefore travels as `%C2%A7…`. That is correct on the
wire and real clients decode it, but the e2e client did not, so any test
asserting on message text was asserting on the encoded form. `grpc.percentDecode`
now exists and the test client uses it.

## 5. Not added, but noted

**`--shard-key`** on search/query/scroll (#159) is sharding, a §1 non-goal. It
must keep returning `UNIMPLEMENTED`; `src/api/handlers.zig` should be checked
against the 1.19 field number before any run that might set it.

## 5b. What running them actually taught

The invocations in §3 were derived from `dev`'s argument surface by reading it.
Executing them found four errors in that derivation, all mine, none in either
engine, and they are worth recording because each produced a plausible number
or a plausible failure rather than an obvious one.

**`-n` sizes both phases.** `--num-vectors` drives the upload *and* the search
loop, and bfb has no separate search count. §4's abridged `-n 1M` reads as an
upload size; at `-p 1` it also means a million sequential round trips. W0 ran 30
minutes without finishing. Rows that can use `--skip-setup` now upload once and
then search with `-n` bounding only the query loop.

**W0 needs an index.** It carried `--skip-wait-index` straight from §4's table,
so it measured a brute-force scan over a million vectors and reported 34.9 rps.
That is not a transport floor; it is the opposite of one. The row is only
meaningful once the index exists, because the point is that d=4 makes the
*vector* irrelevant, not that the search is.

**W4's socket count is `threads x connections`, not `connections`.** `-t 16 -c
8` opens ~128 sockets. A server with fewer slots closes the excess *before* the
HTTP/2 preface, so there is no stream to carry a status and the client reports
only `Unknown error transport error MetadataMap {}`. This killed W4 for an
entire run, and the flakiness across `-t` values, 4 working while 2, 8 and 16
failed, was slot pressure crossing the threshold, not a race. Fixed in the
engine (`connections_per_io` raised, a `connections_refused` counter added, and
stats printed at shutdown so the counter is readable), and the row now passes a
connection count that fits a default server.

**W12 asks for its field index, since 2026-09-03.** Before that strawmann
answered `UNIMPLEMENTED` for `/qdrant.Points/CreateFieldIndex` (correctly, it
was §2 phase 3 and §10's optional M7) and bfb `unwrap()`s the response and
panics, so the row carried `--skip-field-indices` and the cost was Qdrant's
payload index: a full scan per query filed under "filtered search".
`compare.py` refused the row for that and still checks the read-back (§3's
W12 row), which is now what licenses it rather than what refuses it.

**W11 measures index invalidation, and that is the finding.** It cannot use
`--skip-setup` because it must upload, so `-n` drives both phases. Worse, its
searches run against a collection whose index strawmann's own upsert
invalidates, so every query becomes a brute-force scan of 1.2M vectors, 200k of
them ran 25 minutes without finishing. Qdrant indexes incrementally and does not
degrade this way. §4 designed W11 to expose exactly this, and it did; the query
count simply has to be survivable for the row to report it.

## 5b-ii. What a row has to carry, and what it used to

§8.9 fixes the row shape: *workload · dataset and checksums · qdrant version ·
strawmann commit · ISA build · conformance tier · recall@1/10/100 · MRDE ·
**QPS · p50/p99 latency** · %-of-roofline*. `rows.json` carried the id, the
status, a duration, the ambient load and `qps`. Everything else was either
measured and discarded or never collected, which is how a table of ratios ends
up being the only artifact.

What it carries now, and why each was missing:

| field | why it matters |
|---|---|
| `when` | rows are merged in place across runs, so without an absolute timestamp one file can hold rows from different days against different binaries and nothing distinguishes them |
| `load_mode` | §7.4's rule that a closed-loop p99 is not a latency result cannot be applied unless the row says which loop it ran under |
| `latency` | bfb reports p50/p95/max and writes every request time; p99 and p99.9 are recovered from the raw array. A saturating row exists to expose the tail, and the tail was the part not being reported |
| `ef` | what the recall sweep joins on |
| `run.json` | engine commit *and* binary hash, Qdrant image digest and version, dataset with pinned checksums, host, gate. A reader who did not run it cannot check a number without these |
| `recall.json` | §7.4: "a single QPS number without its recall is meaningless." Written by `conformance relevance --json` |
| `conformance.json` | §8's licence: which tier the two engines reached together, written by `differ --json`, and the row the results sink checks against |
| `gate`, `profile`, `engine_build`, `engine_binary`, `isa_build`, `optimize` | the build identity and gate verdict *per row* (`workloads.build_identity`). `run.json` is rewritten by every invocation, so a later single-row run on a rebuilt binary re-vouched for every row in the file; `compare.py` refuses a label whose rows carry more than one identity as STALE |
| `warmup_s` | wall seconds of the discarded warm-up pass (§7.1) that ran the same bfb invocation at `WARMUP_N` = 2000 queries before the measured row; `None` under `--no-warmup` and on upload rows |
| `foreign` | the processes other than the engine and the harness that consumed CPU *during* the row, as a per-pid `/proc` delta (`foreign_between`); a row with it set is refused a ratio, not annotated |
| status `no-output` | bfb exited 0 and left neither its JSON nor a `Median qps` line: a failure, filed as one rather than as `ok` with no qps (which `compare` then dropped as an upload row) |

A run taken before these fields existed is not lost: `workloads.py backfill
<label>` recomputes latency from the per-row bfb JSON still on disk, and writes
a partial `run.json` recording the dataset (which §4's table names) while
stating that the engine build was never recorded and cannot be recovered. What
is unrecoverable is said to be unrecoverable rather than filled in from
whatever binary happens to be in `zig-out` today.

Recall is joined onto a latency row **only** where the sweep speaks for that
row's configuration: an fp32 search at a stated `ef`. A quantized row searches a
different representation, the exact row is 1.0 by construction, and scroll
returns no ranking at all, so those show `-` rather than a borrowed number.

## 5c. What every row also records: storage and I/O

Throughput alone flatters exactly one of the two engines here. strawmann holds
its collection in RAM, and the server never calls `persist` at all, so it does
no disk I/O to serve a query and stores nothing between runs; Qdrant is
disk-backed and pays for that on every row. A table of queries per second
charges Qdrant for durability strawmann does not provide, and reports the
difference as speed.

So each row now carries, via `bench/harness/procstat.py`:

| field | meaning |
|---|---|
| `storage_bytes` | bytes on disk under the engine's storage directory, after the row |
| `rss_peak_bytes` | `VmHWM`, peak resident set since the process started |
| `disk_read_ops` / `disk_write_ops` | block-layer requests during the row |
| `disk_read_bytes` / `disk_write_bytes` | block-layer bytes during the row |
| `syscall_reads` / `syscall_writes` | read and write syscalls, **all descriptors** |

**The last pair is not a disk figure and is never presented as one.** `syscr`
and `syscw` count syscalls on every descriptor, sockets included: 5,000 queries
against strawmann measured 24,995 `syscr` and zero bytes at the block layer.
That column is the network. Printing it beside a disk-backed engine's
block-layer count under one heading would compare one engine's sockets with the
other's disk, which is the shape of the `rps`/`qps` error §4.1 already caught.

Two interfaces are read, because neither is complete. `/proc/<pid>/io` gives
syscall counts and block-layer bytes but no block-layer operation count, and is
readable only by the owning user, so a container running as root is opaque to
it. cgroup v2's `io.stat` gives operations and bytes and is world-readable, but
exists only where the io controller is enabled for that cgroup, which a plain
user session scope usually lacks. A field neither source supplies reads
`unknown`, never `0`: "nothing happened" and "we could not look" are different
claims, and only one of them is a result.

The storage directory is discovered for a Qdrant container from its bind mount,
and otherwise comes from `--storage <path>` or `$STORAGE_DIR`. Unknown is again
reported as unknown; strawmann's zero disk *bytes* over a whole run is the
measured evidence that it stores nothing, and is worth more than an assumed 0.

### What every row also records: whether the engine was running

The same argument one level down. A row's queries per second says how fast it
went; it does not say whether the engine was *on a core* while it went, and the
three ways of not being on one have three different causes and three different
fixes. Also via `procstat.py`, differenced across the same bracket:

| field | meaning |
|---|---|
| `cpu_user_s` / `cpu_system_s` | CPU the thread group consumed |
| `minor_faults` / `major_faults` | pages touched, and pages fetched from a device |
| `runqueue_wait_s` | runnable and **not scheduled** (starved of a core, not slow |
| `blkio_delay_s` | **not runnable at all**, blocked in the block layer |
| `ctx_switches_voluntary` / `_involuntary` | chose to sleep / was preempted |
| `migrations` | moved between cores) the pinning claim, checked |
| `psi_*_some_s` / `psi_*_full_s` | cgroup stall time for cpu, io and memory |
| `rss_anon_bytes` / `rss_file_bytes` | resident bytes the engine allocated / mapped |
| `threads`, `sched_coverage` | how much of the row the surviving threads account for |

**Voluntary switches are the free lock-contention signal.** An involuntary
switch is the scheduler taking the core away; a voluntary one is the engine
choosing to sleep, on a futex, a socket or a disk. Per unit of work, a row that
sleeps more often is a row spending more of itself waiting for another thread,
not a proof of contention (an idle connection thread sleeps too), but free, and
already on disk for every row ever measured.

**Two of these have blanks that are not zeros, and they are not the same
blank.** `blkio_delay_s` needs `kernel.task_delayacct=1`, which most hosts ship
off; off means the kernel reports a permanent zero, which is exactly what "this
engine never waited for a disk" looks like. The strongest claim in §5c, and it
would be manufactured by a sysctl. The PSI columns need the engine in a cgroup
of its own: a container has one, a binary started from a shell shares the
operator's session, and differencing *that* scope across a row measures the
desktop. Both are refused rather than reported, and the report says which kind
of blank a blank is.

### What a row records when it is asked: hardware counters

`workloads.py run ... --perf` attaches `perf stat` to the engine for the length
of each row (`bench/harness/perfstat.py`), which is what §7.3's fixed event set
means at the full-stack level. It is opt-in because it is an instrument that
touches its subject.

| field | meaning |
|---|---|
| `perf_cycles`, `perf_ref_cycles` | core cycles, and cycles at the constant reference rate |
| `perf_instructions` | retired instructions, with cycles, IPC |
| `perf_branch_misses` | mispredicted branches; reported per thousand instructions |
| `perf_dram_fills` | demand loads served from DRAM; × the cache line, DRAM traffic |
| `perf_dtlb_walks` | data TLB misses that reached a page walk (§5.5) |
| `perf_page_faults`, `perf_ctx_switches`, ... | the same quantities `/proc` reports, counted a second way |
| `perf_events`, `perf_enabled_pct`, `perf_note` | what was counted, how much of the row, and why a blank is blank |

Everything is reported **per query**, which is the only form in which two
engines' hardware counters compare: an absolute count is a function of how long
the row ran and how many threads ran it, and the two engines hold neither
equal.

Nothing is scaled. Where `perf` has to multiplex a group it prints values
extrapolated from the fraction of the row each counter was on, and those are
withheld rather than shown, the same rule the thread-summed counters follow
when a thread exits. An event this part does not implement is likewise blank.
The event *spelling* is recorded per row because the roles above are not the
measurement: a DRAM-fill count from `LLC-load-misses` and one from
`ls_dmnd_fills_from_sys.dram_io_all` are both "cache misses" and they are not
the same number.

## 5d. Four corrections from reviewing what each row measures

**Queries come from the dataset now, not from a random generator.** bfb's
flag-driven search builds every query with `random_dense_vector(rng, dim)`
(`src/search/from_args.rs`), and `--fbin` supplies only the *upload* corpus. So
every search row was querying uniform-random points in a space where SIFT
descriptors are anything but uniform, out-of-distribution queries, in a region
with no data. Worse for the report: recall came from the conformance sweep,
which uses the real query file, so every joined recall/throughput pair
described two different query sets.

The rows marked `query_collection` now run through `bfb search --file`, whose
`source: {type: file, strategy: from-start}` walks the dataset's own queries in
order. Measured on a 100k collection, the change is worth **13%**: 3,786 q/s on
random queries against 4,294 q/s on real ones. A random query far from every
cluster walks further down the graph before it converges, which is the opposite
of what one might guess and the reason to measure rather than assume.

**W11 appends.** It used to overwrite, because bfb assigns ids from `--offset`
(0 by default) and W11 uploads into a collection W2 already filled. There was
no pending tail and no rebuild, see `comparison-sift1m.md` §4f's correction. It now
passes `--offset UPLOAD_N + W11_STEADY_N` (W11-steady appends first, at
`UPLOAD_N`), so the collection reaches `UPLOAD_N + W11_STEADY_N + W11_N` points
and the server needs `--capacity` for all three; `required_capacity()` takes
that against W12's own collection and the runner prints the number before the
first row. Leaving W11-steady out of the sum once left `bench2` 50,000 short.

**The quantized rows state their `ef` and have a recall sweep.** W6, W7 and W8
searched at whatever each engine defaults to, which is not the same number on
both sides, and no sweep covered their collections at all, so they carried a
qps with no recall. The number §7.4 calls meaningless and the §8 sink refuses.
All three now pin `ef=128`, so the encoding is the only variable between them,
and `bench/harness/recall.py` sweeps `bench2`, `bench6`, `bench7` and `bench8`
with `relevance --skip-upload`, which scores the collections the benchmark
actually measured rather than fresh ones. Each sweep sends the quantization
search parameters the row it speaks for sends (`--quantization-oversampling
4 --quantization-rescore true` for W7, rescore alone for W6/W8; `recall.py
--oversampling/--rescore` overrides), records them in the JSON as
`oversampling`/`rescore`, and every reader joins a row only to a sweep whose
recorded parameters equal the row's own (`quantization_oversampling`,
`quantization_rescore` in `rows.json`). A `bench7` sweep taken at oversampling
1 does not speak for W7 at 4.

**Rows are stamped individually.** `run.json` carries the harness stamp of
the *last* invocation and `rows.json` merges rows in place, so a subset re-run
(`workloads.py run <uri> <label> W3`) leaves rows measured under an older
harness beside one measured today, all vouched for by one stamp. Each row now
carries `harness_hash` (`workloads.stamp_hash`: the stamp's comparable fields)
and `n_requested` (the row's effective `-n`, which `--n-factor`/`--min-duration`
change; noted on the row when it differs from the table's, but not part of the
hash: qps is a rate and `--min-duration` scales `-n` per engine); `compare.py`
refuses a ratio for a row whose hash differs from its label's current stamp or
from its counterpart's, and says `re-run under a different harness`.

**The open-loop rows are faithful only at low rates.** Measured at three
offered rates on one collection: 1.4x client/server p50 at 100/s, 40.1x at
500/s, 28.8x at 2000/s. `findings.md` §15 has the table and what it implies.
Both timings are recorded per row so the server-side half stays usable.

## 6. Status

W0-W13 have been run against both engines under `fullrun.py`; the joined
table is `comparison-sift1m.md`. What this document fixed originally is that the
*previous* invocations were derived from a branch we are not comparing
against. `Scroll` is implemented and W13 has both arms (unfiltered, `--scroll`
flag path).
