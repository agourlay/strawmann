# Engineering decision log

A running record of non-obvious decisions, with the *why*, so future work does
not have to reverse-engineer the reasoning, and so decisions that were made by
accident get made deliberately instead.

Entries include the ones that were **wrong**. A decision log that only records
the good calls is a marketing document; the reversals are where most of the
information is.

Newest entries at the bottom of each section.

---

## Toolchain

- **Zig 0.16.0**, pinned in `build.zig.zon` via `minimum_zig_version`. §9: "a
  benchmark that silently changes compiler is not a benchmark." LLVM's
  autovectoriser and its AVX-512 cost model both move between releases, so an
  unpinned compiler turns the §7.5 ISA matrix into noise.

- **Rust for the conformance harness, and this is not a preference.** §8.5
  requires the harness use *the same `qdrant-client` crate bfb uses*: "one
  client, one encoder, one decoder, both engines. Any difference observed is
  then necessarily server-side." That crate is Rust-only.

- **Python everywhere; no shell scripts remain** (2026-08-16). This started as
  "Python for experiments, shell for setup only", which held until the two
  survivors were looked at properly. Every real bug in the old shell runner came
  from something shell makes awkward: `-n` sizing two phases, `rps` vs `qps`
  being a grep rather than a field, per-row load tracking in `awk`, and a
  running bash script being read by byte offset so editing one mid-flight
  corrupts the run.

  The last two went the same way. `dump-asm.sh` was 123 lines of bash wrapped
  around an embedded `awk` program that split objdump output per symbol, with
  the splitting rule and the counting rules in two different languages, neither
  unit-testable; the port produces **byte-identical** `.asm` for all 56 kernels,
  which is what made it safe to swap. `fetch.sh` parsed a TSV with `IFS`
  splitting, where a filename containing a space would have silently become two
  fields. `datasets.py` still shells out to `curl` for the transfer itself, because
  resumable HTTP with redirect-following and retry-on-anything is the one part
  curl genuinely does better, and replacing a battle-tested downloader to win a
  language argument would be the wrong trade.

  Both are stdlib-only: they run on a fresh clone before `uv sync` exists.

- **The analysis tooling is a uv project, and has dependencies** (2026-08-16).
  `bench/pyproject.toml`, locked by `bench/uv.lock`: jinja2, plotly, pandas.
  The `dependencies: none` badge is a claim about `src/`, and a vector engine
  that pulls in a package manager is a different project. The report path is not
  the engine. The first version of `report.py` hand-rolled SVG path geometry and
  tick-rounding rather than admit a chart library, which is the worse trade: it
  reimplemented a plotting library, badly, in a file nobody would maintain.
  Locked for the same reason the Zig toolchain, the bfb commit and the Qdrant
  image digest are locked.

- **The spec lives at `docs/spec.md`** (2026-08-16). It was at the repo root as
  `strawmann-spec.md`, which contradicted §9's own layout block, that block has
  always read `docs/  this spec, the cost model, results`. The move makes the
  tree match what the spec describes rather than the other way round.

- **`zlint.json` enumerates all seventeen rules** (2026-08-16). zlint's `rules`
  block is an allowlist: naming any rule disables every rule not named. A config
  listing three rules as `off` silently disables everything and the tree reports
  `0 warnings` vacuously. That mistake has been made and caught elsewhere,
  where it went unnoticed for weeks. `scripts/check.py` therefore has a canary step that appends an unused
  declaration and fails if zlint still passes.

## Architecture

- **The HTTP/2 stack is hand-written from M1, not nghttp2.** §6.1 stages nghttp2
  first with a native implementation at M6. The build host has no nghttp2, so
  the native path was the only option, and M6's constraints (zero per-request
  allocation, fixed header table, no encoder dynamic table) applied from the
  start rather than being retrofitted.

- **Runtime ISA dispatch selects among widths the compiled target permits.** Zig
  has no per-function target attribute, so one binary cannot hold both an
  AVX-512 and an AVX2 kernel the way Qdrant's shipped binary does. §6.6.5 makes
  the forced-ISA matrix primary and dispatch secondary; the matrix is complete.
  Building each arm as a shared object and `dlopen`-ing is what would fix it, and
  is worth doing only if M9 concludes mixed dispatch is the shipping
  recommendation.

- **The parallel HNSW build is not checksum-stable.** §8.7 offers this as option
  (b) with the serial build as (c) "for conformance", and measurement says the
  lock-based build does not in fact reproduce. `buildSerial` exists and is
  tested bit-reproducible ("§8.7: the serial build is bit-reproducible for a
  given seed"), but it is not selectable at runtime: `handlers.zig` sets
  `build_mode = .parallel` unconditionally, no CLI flag changes it, and only a
  unit test sets `.serial`. Conformance therefore runs against the parallel
  build, and the graph checksum in `vectors.bin` is a per-build value, not a
  cross-run invariant.

- **One graph, not segments.** Qdrant runs up to 8 segments; strawmANN holds a
  single graph. This is a real difference with real consequences, it is why
  `ef` is not a comparable knob between the two engines (see *Corrections*) -
  and it is not currently planned to change. Segmentation would buy intra-query
  parallelism and better recall per unit of traversal; it would cost the flat
  level-0 layout §6.5 is built around.

- **Writes keep the graph** (2026-08-16). `invalidateIndex` used to set the index
  state to `.absent` on any upsert, discarding a graph that was still perfectly
  correct for the points it covered, so every subsequent query brute-forced the
  whole collection. W11 measured that cliff at 121 qps against Qdrant's 1,645.
  A graph over the first N points is *incomplete*, not *wrong*, when point N+1
  arrives: `search` now scans the tail past `graph.count` exhaustively and
  `rebuild_ratio` (10%) bounds how large that tail may grow. 121 → 7,260 qps,
  and recall is unaffected by construction because a pending point is compared
  against the query directly.

- **`proto/messages.zig` stays one file** (2026-08-16; 1372 lines then, ~1500
  now). A split by
  service was attempted and reverted. The section banners do not partition the
  types they name: `QueryPoints` and `SearchParams` are defined below the
  "Collections" banner, so any line-based split scatters related codecs across
  files. The file is *coherent* even though it is long, it is the hand-written
  codec for one wire API, and a bad split is worse than a long cohesive file.
  Splitting it properly means moving declarations individually, which is worth
  doing only if it grows further.

  `collection.zig` split cleanly (2202 to 1310, with `quantized_search.zig` and
  `persist.zig`) because its sections were genuinely separable: §6.5, §6.7 and
  §6.4 shared only the `Collection` type. It has since grown back past 3300
  lines, mostly the in-place-update seqlock, the overwrite log and their tests;
  the split rule still holds, and the next candidate is the tests.

- **One dataset descriptor, read by everything** (2026-08-16).
  `conformance/datasets/datasets.json` replaced `manifest.tsv` plus the
  `DATASETS` const in Rust. The two had already drifted: §4.2's table listed six
  datasets, the manifest pinned files for two, and nothing compared them because
  nothing could. Rust now reads the same file via `include_str!`, so the oracle
  cannot disagree with the fetcher about a dimension or a metric.

  Entries carry `status`: `available` (files pinned and fetchable) or `declared`
  (§4.2 names it, files not pinned). The gap between spec and reality is now a
  column in `datasets.py list` rather than a discrepancy between two files.

  The schema deliberately tracks bfb's `src/dataset/config.rs`, itself
  compatible with vector-db-benchmark, so a dataset defined here can drive bfb.
  It is a superset, not a copy: bfb carries no checksums and §4.2 requires them,
  and bfb's config is `deny_unknown_fields`, so `bfb-config` emits a projection.

  Sharded datasets keep an explicit ordered file list rather than bfb's `{i}`
  template, because Hugging Face appends a per-shard content hash that no
  template reproduces and each shard needs its own digest. The order is
  load-bearing: row order defines point ids, and point ids are what ground truth
  indexes into. `bfb-config --link` builds templatable symlinks so bfb can still
  be driven from the same definition; without it the command warns instead of
  emitting a path that looks right and matches nothing.

## Measurement methodology

- **ε is calibrated, never chosen.** §8.4 derives it from how much each engine
  already disagrees with *itself* across ISA arms. The differ fell back to a
  floor for a long time and said so on every run; `dump-scores` + `calibrate`
  now produce the real value (`ε(cosine, d=1536) = 9.537e-7` from a measured
  spread of 2.384e-7).

- **T1 is graded against the fp64 oracle, not against Qdrant** (2026-08-16).
  The old rule failed on any disagreement beyond ε, which silently makes Qdrant
  the definition of correct, so strawmANN *failed a conformance tier for being
  more accurate*, and the only remedy would have been to reproduce Qdrant's
  rounding error deliberately. §8.1 already conceded bit-exactness was
  unachievable, so equality with Qdrant was never the requirement; being right
  was. Semantics (§8.3's table) are still replicated exactly; arithmetic error
  is not. A divergence in *ids* or result *count* is never excused this way.

- **Two licensing gates, not one** (2026-08-16). `licenses_performance_claim`
  requires T1, right for a single-engine kernel number, since T1 establishes the
  arithmetic matches. `licenses_comparative_claim` requires T3, because T3 is the
  tier that establishes *equal recall*, and two engines at different recall are
  doing different amounts of work. The differ once printed "T3 FAIL, a QPS
  comparison here would be at unequal recall" and "this run LICENSES a
  performance claim" one line apart.

- **The §7.1 gate checks whether the machine is busy** (2026-08-16). It
  previously checked eight *configuration* properties, governor, boost, SMT,
  isolation, THP, NUMA, perf paranoia, toolchain, every one of which can be
  correct on a host running something else. Adding `check_quiescent` immediately
  found a Qwen3-Coder inference server at 1008% CPU that had run through an
  entire W0-W13 table unnoticed. Load is *reported* by the gate but not
  hashed (`setup.py` `observe`, as against `note`): a float like `load1=1.53`
  in the hash made every run a different environment and the warning fired on
  every comparison. The hash answers "same machine, configured the same way?";
  whether something else ran during a row is a property of the row, recorded
  per row as `load_start`, `load_end` and `foreign` (the per-pid CPU delta
  across the row, `foreign_between`), and a row with `foreign` set is refused a
  ratio by `compare.py` rather than annotated.

- **CPU isolation names the experiment instead of gating it** (2026-08-17).
  §7.1 required `isolcpus` and `nohz_full` on the server cores and refused to
  run without them, which made every measurement on an ordinary machine
  development-grade by construction. That is the right rule for attributing a
  difference to the engine and the wrong one for saying whether anybody would
  notice it: nobody deploys with the scheduler amputated, so a hermetic run
  answers a question users are not asking.

  The isolation state is now a **profile** carried in the environment hash:
  `isolated` for attribution, `as-deployed` for external validity. Both pass
  the gate; neither is publishable as the other, and because `isolcpus=` and
  `nohz_full=` were already hashed, the mechanism that keeps two machines from
  sharing a chart already keeps the two profiles apart. `bench/setup.py check`
  prints the profile, `run.json` records it, and the report names it and
  refuses to let the two be read as one comparison.

  What stayed a hard gate: quiescence and the per-row foreign-load flags. An
  as-deployed run tolerates the load a user's machine has. It does not tolerate
  this run's two arms getting *different* amounts of it, because that lands in
  the ratio and nothing downstream can separate it out, which is exactly what
  happened on 2026-08-17, when `rustc` contaminated four strawmANN rows and one
  Qdrant row of the same table.

  The honest caveat, recorded so it is not rediscovered as a surprise: a laptop
  running a browser is not a model of a cloud VM either. `as-deployed` on this
  host means "a developer machine", and a run that wants to describe a cloud
  deployment should be measured on one.

- **The parallel index build runs on the server's cores, not on one worker's**
  (2026-08-17). `--pin --cpus 4-11` pins each worker to a single core, and
  `ensureIndexBuilding` spawns the build from the worker that handled the
  upsert. `std.Thread.spawn` children inherit the parent's affinity, so every
  thread of the "bulk parallel build on 8 threads" the banner advertises
  inherited that one worker's pin. Measured mid-build: `Cpus_allowed_list: 10`
  on all eight, and the whole process using 100% of a single core. §6.5 says
  the bulk build uses "all cores"; it had used one, in every pinned run this
  project had ever made, while the banner said otherwise.

  `pinToCpus` widens the build task to the server's pool before it spawns.
  Build threads went from one core to eight, process CPU from 100% to 800%, and
  a W7 index build from never finishing inside 300 s to 41.2 s.

  This corrupted a published metric rather than only a failed row: W2 is
  time-to-Green, an index-build measurement, and every pinned run of it had
  timed a build confined to a single core.

- **A row that blows a timeout is not a hang, and the difference is the whole
  diagnosis** (2026-08-17). W7-upload and W8-upload each looked like a crash:
  the row failed, and every row after it failed against a server that would not
  answer. Three hypotheses were wrong before the right one. A crash (the
  process was alive), foreign load (it reproduced on a quiet, gated host), and
  the quantizer (the same rows passed unpinned). What the engine was actually
  doing was answering queries at 64 q/s while an index build ground through one
  core, so the collection never reached Green and the client waited forever.

  The lesson worth keeping is procedural: the evidence that settled it was
  per-thread `Cpus_allowed_list` from `/proc`, and the reason it took three
  wrong turns is that "the engine is unreachable" and "the engine is busy" look
  identical from the client. Ask the server what it is doing before theorising
  about why it stopped.

- **The gate must be given a quiet machine, not asked on a busy one**
  (2026-08-18). `workloads.py` re-runs §7.1 on every invocation and overwrites
  the arm's `env.txt`, and `measure` invokes it twice per arm, the second time
  straight after the recall sweeps. So the stamp that survived was the one
  taken at the heaviest moment of the run: an arm whose own gate passed at load
  1.09 was stamped `FAIL ... load 4.28`. Worse, the arms run back to back, so
  the second inherited the first's decaying one-minute average and failed on
  quiescence through no fault of its own, costing a 53-minute run its
  comparison.

  `settle` now waits before every gate, including `fullrun`'s own, to a target
  *below* the gate's threshold. Releasing at the threshold left no headroom and
  the gate read 13% seconds after settle released at 10%. A load that has
  stopped falling counts as settled, because `load1`'s floor is whatever the
  machine idles at and insisting on a lower number would burn the timeout and
  then measure anyway.

- **Every run records the build mode and the ISA that served it**
  (2026-08-17). `zig-out/bin/strawmann` was a Debug, avx2 binary while the
  harness recorded `isa build: native` and nothing contradicted it; the two
  were told apart afterwards by file size. §9 licenses numbers from ReleaseFast
  only. The optimize mode is now a build option printed in the banner and in
  `--probe`, alongside the dispatch tier, which `dispatch.active` fixes at
  comptime, making it the one fact that separates two binaries at one path. The
  report prints `ReleaseFast, native build, dispatch avx512, vnni on`, and
  refuses anything else with a pill rather than showing it as a neutral fact.

- **An uncontrolled placement refuses the storage comparison** (2026-08-18).
  §5.5's placement decides whether vectors live in anonymous memory or in a file
  mapping, and no run had ever set it: strawmANN defaults to `pinned`, Qdrant's
  default for dense vectors is `Cached`, and `Pinned` is not available to Qdrant
  for dense vectors at all. So the Storage and I/O section read as a
  RAM-resident engine against a disk-backed one when it was substantially
  reading a configuration nobody chose: Qdrant wrote 41.8 GB because it was
  asked to map a file.

  Both engines already speak `VectorParams.memory`; nothing read it back.
  `collection-info` captures it, and where the placements differ (or are
  unknown, as for every run measured before the capture existed) the storage
  and memory rows are shown per engine with no comparison. Unknown and equal
  are deliberately different answers: a run with no capture does not get the
  benefit of the doubt. `cached` is the only placement both engines can serve,
  so a like-for-like I/O comparison needs `--data-dir` and
  `--default-placement cached`.

- **Every workload row records its own load and any foreign process.** The gate
  catches a busy machine at run start; a process appearing mid-run is invisible
  to it. Re-measuring the flagged rows calibrated what contamination costs:
  18 foreign cores = +41%, 3 cores = +11%, 1 core = noise.

  **The threshold is that calibration, summed, and in cores.** It used to be
  20% of *one* core applied to each process separately, which was wrong twice
  over. Too strict: it refused runs over a fifth of a core, five times below
  the smallest load measured to have any effect, and on a twelve-core host
  that is 1.7% of the machine. It cost three runs in one night, to an editor
  and to the agent writing the report. Too lax: judging each process alone,
  ten processes at 15% of a core apiece are 1.5 cores of real cost that no
  single one of them declares. The verdict is now the *total* foreign CPU
  against `FOREIGN_BUDGET_CORES`, default 1.0. The largest load this project
  has measured as sitting inside run-to-run noise, which makes it the last
  value defensible as harmless rather than the first that is obviously
  harmful. Processes are still named well below it, because a list of what to
  close is useful even when the verdict is pass.

- **Both engines are measured at one residency, or the run refuses.** §5.5's
  placement decides whether vectors sit in anonymous memory or in a file
  mapping, and a ratio taken across two residencies measures the residency.
  The run used to leave each engine at its own default (strawmANN `pinned`,
  Qdrant a mapping) refuse the storage and memory rows, note that residency
  "can move throughput too", and print the throughput ratios anyway. The
  confound was named on the page and bounded nowhere.

  `--placement` is asked of *both* arms and defaults to `cached`, the only
  residency both can serve: Qdrant v1.19.0 answers `pinned` with "`pinned`
  memory placement is not supported for dense vector storage", and strawmANN
  answers `cached` without a store with "cached and cold placement need a file
  to map", so it is given a `--data-dir`. `preflight_placement` refuses an
  impossible request in a second rather than an hour in, and
  `placement_mismatch` checks what the engines *read back* after both arms.
  A request is a target an optimizer may miss, which is the lesson
  `--segments 1` taught.

  The cost is explicit and is the right trade: strawmANN is no longer measured
  at its own default, so a gated run no longer says what a strawmANN user gets
  out of the box. It says what the two engines do at one residency, which is
  the only thing a ratio between them can mean. Its default is a separate
  measurement (`--placement pinned --skip qdrant`), and the difference between
  the two is what finally bounds the confound.

- **Each engine is measured alone.** The other engine is stopped for each run,
  because a co-resident engine doing background work perturbs the measurement.

  The original justification here was wrong and is worth correcting rather than
  quietly editing: it claimed Qdrant "idles at ~50% CPU doing background
  optimisation". Measured, a freshly started Qdrant with no data sits at
  **0.44-0.88%**. The optimiser is condition-driven, not a polling loop
  (`is_optimization_required` in `lib/shard/src/optimizers/indexing_optimizer.rs`
  checks indexing and mmap thresholds), so it does work when there is work and
  stops when there is not.

  What was actually observed was CPU *during* indexing after a bulk ingest,
  which is real work rather than idle spin, and which is exactly what a
  benchmark run triggers. The decision stands; the number attached to it did
  not, and "idles at 50%" would have misled anyone reasoning about Qdrant's
  steady state.

- **Qdrant's storage lives on a real block device, never tmpfs.** The first
  headline attempt put it under the session scratchpad, which is tmpfs: its
  segment data consumed the RAM both engines were competing for, and the
  optimiser failed with "not enough space" while `df` reported 21 GB free.
  Measuring a disk-backed engine on a RAM disk is not a measurement of that
  engine, and had it *succeeded* it would have produced clean-looking numbers
  that meant nothing.

- **CI does not run the workload table** (2026-08-16). W0-W13 need bfb, multi-GB
  datasets, a live engine and a §7.1-gated host. A GitHub runner is none of
  those. Benchmarks run on a gated host; CI proves the code that produces them
  is correct.

- **CI runs per push, and `ci-covers-gate` refuses a workflow that cannot**.
  Auto-runs were off from 2026-08-16 because this was a private repo with no
  Actions credits to spend per push, and the workflow was kept as the definition
  of what a run checks. That cost more than it saved: `ci-covers-gate` matched
  `# gate: <label>` as a substring of the file and never read `on:`, so it
  reported all 21 gate steps covered for a workflow that had never executed. A
  comment also outlives the step it describes, so a marker was not evidence of
  anything. The check now reads the `--only` patterns out of live `run:` lines
  and fails when no trigger fires without being asked; the markers stayed, as
  documentation. There is no `paths-ignore`: `readme-table-current` guards the
  generated blocks in `README.md` and `docs/comparison-<dataset>.md`, so a
  docs-only commit is the one that most needs checking.

- **The Rust toolchain is pinned, in one place**. §9 pins Zig because "a
  benchmark that silently changes compiler is not a benchmark". CI installed
  Zig 0.16.0 and then `dtolnay/rust-toolchain@stable`, a moving branch, with
  no `rust-toolchain.toml` and no `rust-version` in `conformance/Cargo.toml`.
  The differ is what §8 makes the precondition for publishing any number, and
  `rust-clippy` denies warnings, so an unpinned stable reddens a green tree on
  a day nobody touched it. The pin is `conformance/rust-toolchain.toml`, scoped
  to the only crate; the workflow reads the channel out of that file rather
  than naming a version of its own, because two spellings of one version drift
  and the drift shows up as a lint failure nobody can reproduce.

- **The README comparison table is generated, not transcribed** (2026-08-16).
  It was hand-written once and went stale immediately: the header carried 4,055
  for search p=1 while the full table said 3,985, a figure from before the W11
  fix. `compare.py --write-readme` splices it between markers, and the gate step
  `readme-table-current` fails if the two disagree. `bench/results/` is
  gitignored, so a fresh clone has nothing to check against and the step passes
  with a note rather than failing; it only bites where results exist.

- **Deleting result directories cost the raw data behind published numbers**
  (2026-08-16). Clearing `bench/results/` before a re-run that was then blocked
  left the reported figures with no file behind them: strawmANN's surviving copy
  predates the W11 fix, and Qdrant's is a single row from a run killed as
  contaminated. Both were removed rather than kept, on the same principle
  applied to the contaminated runs, no data beats wrong data. The tables stand
  as reported and will regenerate from the next clean sweep.

- **SMT is declared, not required off** (2026-08-25). `check_smt` failed any
  host whose `smt/control` was not `off`, so measuring on an ordinary machine
  began by offlining half its threads and ended by putting them back. §7.1
  asks for "SMT explicitly on or off, declared per run, never left ambient".
  A declaration, and the gate was reading it as a setting.

  `smt=` stays in the hash body and stops being a verdict, exactly as the
  isolation state did above: an SMT-on row and an SMT-off row have different
  environment hashes and never share a chart. `apply` no longer writes
  `smt/control` either; offlining the operator's threads was a change to their
  machine that nothing downstream now asks for.

  Recorded because the reversal is not free, and the first draft of this entry
  claimed it was: the hash keeps SMT-on rows away from SMT-off ones, and does
  nothing for the two arms *within* an SMT-on run. Both engines are pinned to
  the same `--server-cpus` set, so placement is symmetric, but each sizes its
  own thread pool, and an engine running more threads than that set has CPUs
  meets sibling contention the other never sees. That lands in the ratio, and
  it is the argument the old gate had. What replaces the gate is disclosure,
  not a measurement: `check_smt` now prints which CPUs share a core, because
  `--server-cpus` is raw CPU numbers and whether `4-11` names eight cores or
  four cores twice over depends on the kernel's numbering (here siblings are
  N and N+12, so the usual `4-11`/`0-3` split stays sibling-free. On an
  interleaved enumeration it would not). An `isolated`-profile comparison that
  needs the old guarantee should still run with SMT off; the hash records
  which was done.

## Verified against Qdrant's source

Read from a `dev` checkout at `0e3397469`, not inferred from behaviour. Each of
these was load-bearing for a published claim.

- **`ef` is per segment.** `lib/segment/src/index/hnsw_index/hnsw/read_view/search.rs`
  takes `params.hnsw_ef` verbatim inside `search_with_graph`, which
  `segments_searcher.rs` spawns once per segment. The probabilistic sampling in
  that file reduces `top`/`limit` per segment, **not** `ef`. This is what makes
  the ISA-unit retraction below correct rather than a convenient excuse.

- **The segment count is capped at 8, and "8 to 13" was invented.**
  `lib/shard/src/optimizers/config.rs:232` is `get_num_cpus() / 2` clamped to
  `2..8`. On this 24-core host that is 12 clamped to 8, matching the
  `segments_count: 8` the API reports. The upper bound of 13 appeared in six
  files and had no source: it was a guess written once and then cited. So
  `ef=128` is *up to 1024* node visits, a definite number, not a range.

- **Qdrant has no AVX-512 in the distance path.** No `_mm512` or `avx512`
  anywhere in `lib/segment`. `dot_similarity_avx` uses four 256-bit
  accumulators, selected by `is_x86_feature_detected!("avx")`. strawmANN runs
  eight 512-bit accumulators on the same host. Part of the measured 2-7x is
  therefore 512-bit kernels against 256-bit ones rather than anything
  architectural, and the comparison should say so.

- **The cosine accuracy gap has a mechanism.** Cosine is normalised at ingest
  and scored as plain dot (§8.3 assumption confirmed). At d=1536, Qdrant's 32
  fp32 lanes give 48 sequential FMA steps per accumulator chain against
  strawmANN's 12 over 128 lanes: a 4x deeper summation tree, which is the right
  shape for the measured 1.36e-6 against 1.5e-7 from fp64 truth. A depth
  difference, not a correctness difference.

- **W12's cost is the filter lookup, not the vector scan** (verified in source).
  `read_view/dispatch.rs:120` picks the search path by filter *cardinality*, not
  by whether a payload index exists: a low-cardinality filter takes
  `search_vectors_plain`, which calls `iter_filtered_points`. Without a payload
  index, finding the ~200 matching points of 200k means checking all 200k, per
  query. At 200k queries that is ~4e10 filter evaluations, so the 44 minutes
  without finishing is the expected cost of `--skip-field-indices` rather than a
  defect. Confirms that stopping the row and refusing to publish it was right.

## Corrections, decisions that were wrong

Four of these are the same mistake: **comparing two numbers in different units
and reporting the difference as a finding.** Repetition does not correct a units
error, which is what makes it more dangerous than noise, two independent runs
agreeing on the wrong number reads as confirmation.

- **"strawmANN has a 3-point recall deficit at d=1536."** Wrong. `ef` is not the
  same unit in the two engines: Qdrant searches `ef` candidates in *each*
  segment, capped at 8 by default, so its nominal `ef=128` is up to 1024 node
  visits against
  strawmANN's 128. Sweeping `m` and `ef_construct` moved recall by <0.1 point
  while `ef` moved it 6, a graph-quality problem would have responded to
  graph-quality knobs. strawmANN at `ef=1024` reaches 0.9882, matching Qdrant's
  0.9864 at its "128".

- **"W5 shows a 7× regression in the batch path."** Wrong. bfb computes
  `rps = per_sec() / search_batch_size`, so on a batched row `rps` counts batch
  *requests*. W5 is 8,989 **qps** against W3's 4,055, 2.2× faster. Two
  independent runs agreed on the wrong number. The harness now reports qps and
  shows both only where they differ.

- **"Huge pages improved throughput."** Claimed on the strength of
  `AnonHugePages: 0 → 6.17 GB`, a change in the *mechanism*, not the outcome -
  while measured throughput had gone *down*. The before/after was also
  uncontrolled: different collections, different server instances, cycle counts
  differing 27× for nominally identical work. A controlled A/B (one binary, one
  dataset, a declared `--no-huge-pages` flag) later showed ~1.24× median, which
  is suggestive and still not established at three reps.

- **"The strawmANN table ran on a near-idle box; only Qdrant's was
  contaminated."** Both were. The quiescence gate found the 10-core inference
  server afterwards. The 53% spread between two identical W0 runs had an obvious
  cause all along.

- **Editing a running shell script.** `run-w0-w13.sh` was edited mid-run to strip
  a scratchpad path before a push. Bash reads scripts by byte offset, so the
  interpreter resumed mid-file and re-entered the W4 loop. The individual bfb
  invocations were still valid, but the *sequence* was untrustworthy, so the run
  was discarded. Runs now execute from a frozen copy, and, since 2026-08-16,
  from Python, which reads the file once.

- **W12 nearly repeated the `-n` bug** (2026-08-16). Qdrant spent 44 minutes on
  W12 without finishing, because `--skip-field-indices` denies it the payload
  index it would normally build, so all 200,000 filtered queries became full
  scans. The flag exists because strawmANN answers `UNIMPLEMENTED` for
  `CreateFieldIndex` and bfb unwraps that into a panic.

  The row was stopped rather than published: strawmANN declines it in 8 seconds
  as a documented non-goal, so there is no ratio, and a Qdrant filtered-search
  figure taken with its index deliberately disabled is unfair by construction.

  The first attempted fix was to cut `-n`. That would have repeated the exact
  bug already recorded above: bfb's `-n` sizes *both* phases, upload and search
  (`stats.rs` derives the request count from it), so the corpus would silently
  have shrunk from 200k to 5k while the row still read as a 200k result. W12 is
  now split into `W12-upload` and `W12`, the pattern W6/W7/W8 already used.

## Deliberate non-goals, restated

§1 lists these; they are repeated here because each has been *reached* by a
workload and answered with `UNIMPLEMENTED` naming the construct, which is the
specified behaviour rather than a defect:

- sharding and replication (`--shard-key` on bfb `dev`)
- sparse vectors, multivectors, and an on-disk *graph* (`hnsw_config.on_disk`
  / `hnsw_config.memory` are refused; the vector arena's `on_disk: true` and
  `VectorParams.memory` are honoured as `Cold`/`Cached`/`Pinned` placements:
  README, "Where the vectors live", which is not larger-than-RAM operation)
- the Qdrant-proprietary `turbo*` quantization variants (§6.7 excludes them
  explicitly; benchmarking against one would compare against something we
  deliberately do not implement)


## Storage datatypes, and where they made the architecture bend

`VectorParams.datatype` (fp32, fp16, uint8) is §5.4's working-set argument
without a codebook: the same collection at half or a quarter of the bytes per
candidate. Three decisions were not obvious.

**The query is converted, not the rows.** A comparison happens in the storage
type, which means an fp32 query is converted once per search and never per
candidate. That is also what Qdrant does (`metric_query_scorer.rs` converts the
preprocessed query through `slice_from_float_cow`), so it is both the fast
choice and the conformant one. `Probe` is where it happens, and it is the same
shape as `quantized.Query` for the same reason.

**§6.3 forbids allocating on the query path, so the converted query lands in a
fixed stack buffer**, and a non-fp32 collection above `max_converted_dim`
(16384) is refused at creation rather than allocating per search. The cap is
generous against §4.2's tiers, whose headline is d=1536, and fp32 collections
are unaffected because they convert nothing.

**The builder lost two of its three scoring hooks.** `Scorer` had `between(a,
b)`, `to_query(vector, node)` and `vector(node)`, and the builder used the last
two as `to_query(vector(node), other)`, which is `between` spelled with an
intermediate. That intermediate assumed a stored vector *is* an fp32 slice, and
the assumption was invisible until a f16 build panicked in a test. One hook now,
and the builder never sees a query at all.

The header records the datatype **before** the CRC rather than in the reserved
bytes after it. Using the reserved bytes would have kept the format version
fixed, but a field that decides how to read every byte in the file is the last
one that should sit outside the checksum: one flipped bit reads a f16 arena as
f32 and returns plausible nonsense. The version bump costs nothing, because
`persist` is called by its own tests and by nothing else, so no file exists that
anyone kept.

## Two Qdrant configurations, and which question each answers

Measured 2026-08-27 on dbpedia-openai-1m, three interleaved passes per engine
with hardware counters. `--segments 1` had never actually produced one segment
(findings 39); `--max-segment-size` above the corpus does. Running it that way
for the first time showed the choice is not a detail:

| row | Qdrant, 4 segments | Qdrant, 1 segment | |
|---|--:|--:|--:|
| W3, one query in flight | 584 | 353 | **−39%** |
| W4, saturating | 1,237 | 2,584 | **+109%** |

Segments are Qdrant's intra-query parallelism. Four of them fan one query across
four graphs, which helps when there is one query and nothing else to do with the
cores, and hurts under saturation because each query becomes four units of work
competing for them. Forcing one segment moves Qdrant hard in *both* directions.

That matters for how the number is quoted. The published dbpedia headline was
strawmANN 3,595 against Qdrant 1,237 on the saturating row, 2.91x. At one
segment the same row reads 3,566 against 2,593, or **1.38x**, and it is Qdrant
that moved: strawmANN is within 0.8% of itself. Most of that headline was a
segment count.

### So they are two experiments, not one with a knob

**Equal work.** Qdrant pinned to one segment, which is what `--segments 1` has
always claimed and now delivers. `ef` then means the same thing on both sides:
one graph searched to a width, against one graph searched to a width. This is
the configuration §8's licensing is built around. It is what let T3 pass on
SIFT, because recall at a fixed `ef` is only comparable when the two engines
explored comparable candidate sets (findings 33: 8 segments gave 1.0000 where 2
gave 0.9978). Answers *whose code is faster on the same index structure*.

**As deployed.** Qdrant's own default, `default_segment_number: 0`, which it
resolves to the CPU count. Nobody runs a Qdrant at one segment; the equal-work
configuration is a control, and §2 says as much in passing, "`--segments 1` on
the Qdrant side **for the first comparisons**". Answers *what a user would see*.
Its recall at fixed `ef` is legitimately not strawmANN's, so a ratio at equal
`ef` is not available there and the honest comparison is throughput at matched
recall.

The two are not rankable against each other and must never share a table. The
mechanism that keeps them apart already exists: `segments` and
`max_segment_size_kb` are in `collection_settings()`, which is hashed into every
row's `harness_hash`, so `compare.py` refuses a ratio across them as STALE. What
is missing is that the choice is currently spelled as two numbers rather than
named, so a report says `segments: 1` where it should say which experiment it
is, the same distinction §7.1's `isolated` / `as-deployed` profile already
draws for the scheduler, and for the same reason.

### Implemented as `--segment-policy`

`fullrun.py --segment-policy equal-work|as-deployed`, defaulting to
`equal-work`. It binds `workloads.SEGMENT_POLICY` in this process and exports
`$SEGMENT_POLICY` for the per-arm subprocesses in one call. The same two-part
move `use_dataset` makes, and for the same reason: an arm that read a
different value would not be caught as a wrong answer, only as STALE.

`as-deployed` omits `--segments` and `--max-segment-size` altogether rather
than passing Qdrant's documented default back to it. "Qdrant's own default"
means the one Qdrant picks; a harness that names it pins it to whatever it was
on the day someone read it, and passing a ceiling at all is the mechanism that
*defeats* the default count, so the two policies would have differed by less
than their names claim. The indexing and full-scan thresholds stay in both:
they decide whether an HNSW graph is built at all, which is a different axis
and one W0 needs settled the same way either way.

The name is stamped into `collection_settings()` beside the numbers it
produced, so a run that overrides `SEGMENTS` for a one-off cannot silently
claim to be the experiment it is no longer running. Two policies hash
differently, which is the refusal. The report names the experiment in the same
sentence as §7.1's profile ("Measured `as-deployed`: the scheduler was left as
it ships … Qdrant ran `as-deployed` too, in the other sense") which is how the
one word on two axes stays unambiguous. A run made before the flag reads
`unnamed` and the banner prints the segment count it asked for, rather than the
report deciding what an old run intended.

**The differ is not governed by it, and that is a hole.** `conformance/`'s
`recreate_collection_hnsw` sets no `optimizers_config`, so §8's differ builds
its collection at Qdrant's own segment count whichever policy the rows ran
under, and the differ is what issues the T3 claim that the two engines are at
equal recall. On SIFT it does not bite: the differ runs n=100,000 at d=128,
and at 200,000 x 128 that collection measures as one populated `hnsw` plus one
empty appendable, so T3 passed under one graph by accident of size rather than
by configuration. At 990,000 x 1536 the same settings produced four populated
graphs. So the licence and the rows can be built at different segment counts,
silently, and the failure mode grows with the corpus. Fixing it means an
`optimizers_config` on the differ's builder and a flag to carry the policy
across. Not done here, and named so it is not rediscovered.

## What the 2026-08-26 run cost, and what the harness now says

A `--reps 3 --dataset dbpedia-openai-1m --perf` run held the machine from
21:45 to 07:12, nine hours and twenty-seven minutes between the first row and
the last, and was stopped in the morning by its operator, one phase short of
the end. It produced 192 good rows across six arms, the first per-engine noise
floors ever measured on that corpus, and hardware counters on both labels.
None of that was wasted. What was wasted was the operator's night, because
nobody had said it would be one.

Three separate failures, and only the first is about benchmarking.

**Nothing stated the cost before the cost was paid.** The flags did not, the
header did not, and the first pass finished at 00:40. By which point three
more hours were already sunk. `fullrun.py` now estimates its own wall clock
from the previous run's rows on the same corpus and prints it *before* the
§7.1 gate settles, so the minute the gate spends measuring quiescence is also
a minute the operator can spend pressing Ctrl-C. The estimate is measured, not
assumed: the previous run's row time, its own measured overhead ratio, and the
passes asked for. It says nothing at all for a corpus this machine has not run
before, which is the honest answer and better than a constant that would be
wrong per dataset.

**It estimates the measurement passes and says so.** A first version padded
them by 35% to stand in for the conformance differ and the render, and the
total then agreed with the night's observed 9 h 27 m to within 2%. Which
looked like validation and was two errors cancelling. That night was three
passes with *no* conformance phase, and ninety minutes of it was the machine
sitting idle inside `qd-dbp1m-rep3` after the run was stopped. Nothing on disk
times the differ (`conformance.json` carries a verdict, not a duration), so
the tail is now named in the header instead of priced with an invented
constant. The passes alone estimate at 7.93 h; the same six arms with the
interruption removed measure 8.03 h.

Getting there took four corrections, none of which would have been visible
from the number alone. A pass is one arm of *each* engine, not two of the
slower: Qdrant's dbpedia arm is 79 minutes of rows against strawmANN's 52.
Arms group by the engine `run.json` names, not by label text, because
`qd-sift-perf` and `qdrant` are both Qdrant. Each engine is priced at its own
overhead ratio, since rows are 86% of a Qdrant arm's wall clock and 79% of a
strawmANN one, and one pooled ratio ran 6% high. And that ratio is the
*smallest* of the engine's arms rather than the median: contamination is
one-sided. A span can only be inflated relative to its rows, never shrunk,
so the least-inflated arm measures the intrinsic overhead and every other is
that plus something. `qd-dbp1m-rep3` reads 2.43 against its siblings' 1.17;
a median survives that only while three arms exist, and splitting the pools
per engine made two arms the common case.

The same absence made a worse decision look reasonable earlier: asked to "run
all benchmarks", the queue was three tiers deep. Priced by the new estimator
that queue is **14.4 hours**, and it was described to the operator as twelve.
An estimate that exists cannot be replaced by one that was guessed.

**§8's differ runs last, so stopping early forfeits the licence for rows that
are complete.** The conformance phase takes about half an hour; it was the
only thing missing from nine hours of otherwise-publishable measurement, and
the report says UNLICENSED across all of it. It was always recoverable (the
phase starts its own pair of engines and ignores `--skip strawmann` /
`--skip qdrant`, so it runs on its own against labels already on disk) but
nothing said so, and an unlicensed report reads like data that must be
re-measured. The UNLICENSED banner now prints the exact command, labels and
dataset filled in. Half an hour against nine hours is a ratio worth naming on
the page rather than in a doc nobody reads at 07:26.

**The run is too long, and shortening it is a §7.4 judgement rather than a
tuning knob.** Of 6.45 h of row time across the six arms, **44.3% is corpus
construction**, every `upload_only` row, led by W8-upload at 18.6% and W2 at
10.6%. The W4 saturation ladder is another 19.0%, and W12 is 5.2% spent on a
row strawmANN answers `n/a`.

Building each corpus once per label instead of once per pass removes two
thirds of that 44.3%, or **29.6% of row time**, the single largest cut
available, and the one that changes the measurement most. It changes what
`--reps 3` means: the spread it folds would stop including build-to-build
variation, which is real, which the current floors include, and which is
precisely the variation finding 38 shows Qdrant has a lot of. It also collides
with W11/W11-steady (13.8%), which mutate the collection and therefore need it
rebuilt or restored. A floor measured over three searches of one graph is a
floor of the search, not of the engine; that may well be the more useful
number, but it is a different number and the report would have to say which
one it is printing. Deliberately not taken here.

## Deviations from the spec

Each is deliberate and stated where it occurs in the source.

1. **The HTTP/2 stack is hand-written from M1, not nghttp2.** §6.1 stages
   nghttp2 first with a native implementation at M6. The build host has no
   nghttp2, so the native path was the only option. The M6 design constraints
   (zero per-request allocation, fixed header table, no encoder dynamic table)
   therefore applied from the start.

2. **Runtime ISA dispatch selects among widths the compiled target permits.**
   Zig has no per-function target attribute, so one binary cannot hold both an
   AVX-512 and an AVX2 kernel. §6.6.5 makes the forced-ISA matrix primary and
   dispatch secondary, and the matrix is complete; `src/dist/dispatch.zig`
   documents what this costs and what would fix it.

3. **The parallel HNSW build is not checksum-stable.** §8.7 offers this as
   option (b) with the serial build as (c) "for conformance", and measurement
   says the lock-based build does not in fact reproduce. `buildSerial` is
   tested bit-reproducible and selectable with `--build-mode serial`
   (`main.zig`, printed in the startup banner); the default is `parallel`,
   and conformance runs against that.

4. **Sparse vectors and NUMA replication are absent.** §1 lists them as
   non-goals or optional; each returns `UNIMPLEMENTED` naming itself. Payload
   storage and filtering *were* on this list until 2026-09-03; see 6.

5. **strawmANN's quantized rescore pool is `max(asked, ef)`; Qdrant's is
   `limit`-sized. The divergence is kept, and the comparison changes instead.**

   `quantized_search.zig` sizes stage 1 as
   `if (rescore) @max(asked, ef) else asked`, so with W8's parameters (no
   `--quantization-oversampling`, therefore 1.0, `rescore = true`,
   `limit = 10`, `ef = 128`) strawmANN rescores the whole 128-node walk where
   Qdrant rescores ten nodes of it. One line, and it is why all three quantized
   rows are refused a ratio at once: W6 0.9868 against 0.9627, W8 0.9774
   against 0.6938, W7 0.0485 against 0.0267, every one of them past §7.4's 0.01
   tolerance and every one of them with strawmANN ahead.

   Restoring parity would mean deliberately returning worse answers so that a
   table can print a number. The walk has already visited those nodes; not
   rescoring them discards work already done. Measured on 2026-08-21, keeping
   the larger pool cost nothing anyway: W6 rose 0.9624 → 0.9868 in recall
   *and* 4,081 → 4,698 in throughput.

   So the divergence stands and §7.4's equal-recall rule does its job: the
   quantized rows do not yield a ratio at nominal parameters, and comparing
   them means the recall/QPS frontier, the way W10 already does. What this
   costs is a d=128 SQ8 ratio; what it buys is that neither engine is asked to
   be worse than it is.

   No conformance tier asserts pool policy (T4 checks quantization dominance
   and Kendall τ against the fp64 oracle, both of which a higher-recall
   implementation passes comfortably) so nothing catches this except the
   recall-equality rule, which is why the rule exists.

6. **The payload index is a hint, and `payload.idx` was not built.** §3 asks
   for "an opaque, append-only blob per point" and §6.4 for `payload.bin` plus
   a `payload.idx` offset table. Built 2026-09-03 (`src/core/payload.zig`):
   the blob is the point's map entries exactly as they arrived on the wire,
   framed, so `with_payload` re-tags bytes rather than decoding a `Value`;
   `payload.bin` carries a length per record and needs no table. The keyword
   and integer indexes are posting lists per value, and an overwrite leaves
   the old value's posting in place rather than searching it out. Every
   offset a posting yields is re-checked against the point's blob before it
   is admitted, so the index over-approximates and never misses, and a filter
   on a field nobody indexed is answered by evaluating blobs directly. The
   answer therefore never depends on `CreateFieldIndex`, only its cost does,
   which is Qdrant's contract too. The dispatch is Qdrant's as well: a
   filter whose verified matching set is under `full_scan_threshold` is
   scored directly (exact over the set), one above it traverses the graph
   with the filter gating admission the way tombstones already did, so a
   selective filter costs a wider walk rather than a disconnected one. The
   filtered-search cost model M7's exit criterion asks for is not written;
   the first W12 run with the index on both engines is what would write it.

   **Measured 2026-08-28, and the asymmetry is smaller than this section
   implies.** Sweeping `bench6` on dbpedia-openai-1m at oversampling
   none/2/4/8 lifts Qdrant's recall@10 from 0.8929 to 0.9888 at `ef` 512 while
   its recall@1 does not move, and 0.9888 is strawmANN's 0.9891. So Qdrant's
   pool is not a ceiling on what its SQ8 *can* recall. It is the default it
   ships, and one flag closes the gap. "Restoring parity would mean
   deliberately returning worse answers" remains true of strawmANN's side and
   is still why the divergence stands; what changes is that the other side is
   a setting rather than a limit, which is the more accurate thing to say
   about another project's engine. findings 42 carries the sweep.
6. **W11's rebuild trigger and core budget are undecided, pending the
   instrumented run.**

   W11 reads 0.47x on sift1m and 0.51x on dbpedia-100K, and the mechanism is
   known: `W11_N` holds its ratio to the corpus at 0.20 while `rebuild_ratio`
   is 0.10, so the row trips a from-scratch bulk rebuild **by construction on
   every dataset**, then searches against a 20%-sized pending tail while that
   rebuild competes for the same pinned cores.

   What is *not* known is why the row never recovers. §6.5 quotes 26 s for a
   1M build, which should publish well inside a 102 s row and leave most of it
   fast. `buildIndex` now logs `rebuild start` and `rebuild published` with
   points, re-encoded rows, elapsed ms and whether another rebuild is already
   due, so the next W11 run answers it directly. Three candidate changes,
   a builder core budget, a different `rebuild_ratio`, incremental
   maintenance. Are all premature until that log exists, and choosing among
   them from the current data would be guessing.

7. **The 1M tier at d=1536 does not fit this host, and the reason is that every
   collection preallocates the same capacity.**

   Measured 2026-08-22 01:00: the strawmANN arm completed its rows with
   `gate: pass` and was then OOM-killed at 45.6 GB RSS before the recall sweeps
   could run.

       Out of memory: Killed process (strawmann)
       total-vm:51989252kB  anon-rss:45612788kB

   `required_capacity()` is a max over collections and the server preallocates
   it for *every* collection, so at `upload_n = 990,000` each fp32 arena is
   1,237,500 x 1536 x 4 B = 7.08 GiB. The workload set holds five at once.
   Bench1, bench2, bench6, bench7, bench8, which is 35.4 GiB before a single
   quantized store or graph. Only `bench2` ever receives W11's appends; the
   other four are over-allocated by 1.4 GiB each.

   The recall sweeps are what force the co-residency: §7.2 runs them after all
   rows, and they need bench2, bench6, bench7 and bench8 alive together. So
   there are two candidate fixes, and they are not equivalent:

   (a) Size each collection to its own contents. Saves ~5.6 GiB and is a
       narrow change, but it does not alter the shape, five ~1M x 1536
       collections still co-reside, and a larger corpus fails again.
   (b) Interleave sweep-and-drop per collection instead of sweeping at the end.
       Peak residency becomes two collections rather than five. This changes
       §7.2's ordering, which exists so that the sweeps see the collections the
       rows measured, so it needs care rather than a quick edit.

   Undecided pending that care. What is *not* in question is that the tier
   needs one of them or a larger machine; it is not a tuning problem.

   One measurement worth keeping from the attempt: at d=1536 a quiet 990,000
   point bulk build took 250-348 s, against 15-45 s for the same count at
   d=128. §6.5's "26 s for 1M" describes neither, and the figure should be
   re-derived per dimension rather than quoted as a constant. Corrected at both
   sites in `collection.zig`.

8. **W11-steady runs before W11, not after, and that is the whole design.**

   Added on 2026-08-21 ("Twelve items: what the numbers claim, and what the
   harness lets them claim") to measure writes that do *not* provoke a
   rebuild, it was ordered after W11 and measured the opposite. W11 leaves ~9%
   of `bench2` pending, the points that land during its own rebuild, 99,400
   of 1,100,600 on sift1m, just under the 10% that would re-trigger.
   W11-steady's 5% crossed that with 11,100 of its own writes, provoked a
   second 1.21M rebuild that never published inside the row, and read 0.39x:
   worse than W11's 0.47x. A row named "below the rebuild threshold" measured
   a rebuild.

   Ordered first, `bench2` is exactly what W2 built: `graph.count` covers every
   point, nothing is pending, and 5% of the corpus stays under `rebuild_ratio`
   on its own. W11 then appends its 20% on top and crosses as it always did,
   sooner by this row's 5%, which is if anything more like a collection that
   has been written to before. The offsets follow the order, so the two id
   ranges abut and the preallocation ends exactly where the last append does.

   This costs nothing: no new collection, no second upload, no change to
   `required_capacity`, and the harness stamp is unchanged, so the runs already
   measured stay comparable.

---

## §6.5's visited set stays the generation-stamped array, measured

§6.5 specifies "a generation-stamped `u32` array, sized to point count ... 4 MB
per worker at 1M points" and asks for it to be measured against "a bitmap plus
dirty list, which is smaller but requires clearing". §11's open question 3 puts
it as "at what point count does the 4 MB/worker footprint start hurting?".
Both were implemented behind one interface and only one was ever reachable.

`-Dvisited=generation|bitmap` selects at comptime, so the traversal's inner loop
carries no branch it did not have before, and the arm is recorded in the banner,
in `--probe`, on every row and in `BUILD_KEYS`: two binaries that answer
identically and move different amounts of memory are exactly the case that
record exists for.

**Measured 2026-09-22**, sift1m at 1M points, three alternated passes per arm on
a gated host, every pass carrying its arm:

| row | generation | bitmap | ratio |
|---|--:|--:|--:|
| W4 | 22,114 | 22,464 | 1.016x |
| W10-ef128 | 20,142 | 20,458 | 1.016x |

The band those must clear is 5.1% to 8.9% (`regression.noise_band` at the
measured spreads, two arms), so this is **no measured difference**, and the
default does not move. W6-ef128 drifted monotonically in both arms, -13% and
-23%, so `aggregate.py` refuses to band it and it is not read here.

**Why it does not move is the more useful half**, from a counter pass on each
arm: the bitmap is 21x smaller and cuts DRAM traffic by **0.9%**, 787 KiB per
query against 795. It does cut dTLB walks 18% (2,350 against 2,853) and lift IPC
from ~1.05 to ~1.12, which is worth about 1% of cycles and disappears into the
noise.

So the visited set is not where the walk's memory traffic goes. A traffic
decomposition of `rel-0921` put 367 KiB per query in "stamps, lists and heap"
and the stamps are ~7 KiB of it; the rest is the neighbour lists and the rows of
candidates that are scored and discarded. Anyone returning to this should start
there and not here.

The flag stays. It is cheap, it is tested, and it is the instrument that will
answer the same question at a scale where the footprint might matter.

---

## Live insertion is built, measured, and off by default

findings 31 argued that W11's ceiling is the pending tail scan, which costs every
query ~3 ms for as long as a rebuild takes, and that the fix is to insert
appended points into the live graph. `-Dlive-insert` does that (`build.insertLive`,
`collection.Bounded`). Measured 2026-09-22 on a quiet gated host, the trade is
sharper and stranger than that entry expected.

**Search during writes gets ten times faster.** W11's server-side p50 falls from
**3,307 µs to 329 µs**. The tail scan disappears exactly as predicted.

**Ingest collapses.** Each point pays a single-threaded HNSW insertion under the
collection's write lock, where the bulk path builds the whole graph in parallel
afterwards. W11's throttled append fell 3,300 to 2,426 points/s, and an
*unthrottled* 200,000-point upload **timed out after 40,700 points** against a
bulk rate of ~950,000 points/s. That is the number that decides the default.

**W11's headline qps reads 0.33x and that is an artefact.** The row's wall clock
is set by the append, not by the queries: 50,000 searches finish long before
200,000 appends do, and the engine sits at 1.19 cores through it. A row whose
duration is the writer's cannot report the reader's rate.

**The graph is not worse.** Two collections of exactly sift1m's 1M points, one
built in a single pass and one with its last 50,000 inserted live, scored
against the shipped ground truth at `ef` 128: recall@10 **0.9870 against
0.9866**, CIs [0.9810, 0.9911] and [0.9805, 0.9908], MRDE 3.04e-4 against
3.00e-4, and recall@1 marginally *higher* incrementally. Whatever incremental
insertion costs, it is not graph quality.

So the default is **off**, because bulk load is the common path and it would
stall there. The flag stays because the split is a workload property rather than
a defect: a collection taking a trickle of writes would get ten times better
search latency during them at no cost in recall, and a collection being
bulk-loaded must not have it. Turning it on by default would need the insertion
moved off the write path, which is a different piece of work.

Two things it must never do, and does not: a quantized collection opts out
entirely, because a live-inserted point is reachable in the graph while the
quantizer encoded only what the build covered, so stage 1 would have no code to
score it by (the differential test against the exact scan returned a stale point
as the nearest). And every reader bounds its traversal by its own snapshot, or a
node lands in both the traversal and the tail scan and the client sees it twice.

---

## Gathering exact queries is not the fix for W9, measured

findings 45 reads W9's concurrency scaling (strawmANN 1.51x from `-p 1` to
`-p 8` against Qdrant's 2.22x) as Qdrant "reading the corpus less than once per
query", and proposes gathering concurrent exact queries into one pass of the
arena. `collection.bruteForceRangeMulti` and `handlers.runGatheredExact` do that
*within a request*; the open item was to do it *across* requests, which needs a
scheduler change.

Measured 2026-09-22 before building it, using the batch path as the proxy: a
batched exact request already takes one arena pass for the whole batch, so
batches of 8 are what a perfect cross-request gather would achieve.

| | queries in flight | passes per 8 queries | qps |
|---|--:|--:|--:|
| `-p 8`, one query per request | 8 | 8, on 8 workers | 191.8 |
| `-p 1`, batches of 8 | 8 | 1, on 1 worker | **66.4** |
| `-p 8`, batches of 8 | 64 | 8, on 8 workers | **467.6** |

**The middle row is the experiment.** A cross-request gather at W9's own
concurrency collects the 8 queries in flight into one pass on one worker, and
that is 66.4 qps against the 191.8 the same 8 queries reach today: three times
worse. Eight workers each streaming the same 512 MB at the same time already
share the lines, so the arena is not being read eight times in any sense the
memory system cares about. `191.8 x 614.4 MB` is 118 GB/s of implied traffic
against a 73 GB/s bus, which is the amortisation findings 45 attributed to
Qdrant alone, happening here too.

Gathering wins only where in-flight queries greatly exceed workers, which is the
bottom row: 64 in flight, gathered eight ways, 2.44x the baseline. That is a
real result for a client that batches, and the intra-request gather already
serves it. It is not a reason to change the scheduler for W9, whose `-p 8` is
the case the change would make worse.

So the cross-request gather is dropped rather than deferred. What remains true
from findings 45 is narrower than it looked: the exact path is bandwidth-bound,
and the lever is reading less per query, which batching achieves when the client
offers batches and a scheduler cannot conjure.

---

## Intra-query parallelism is not worth §6.3's discipline, measured

findings 50 profiled W3's unsaturated cost (972k cycles per query at `-p 1`
against 650k saturated) and found unhidden memory latency rather than a wake
cost: 6.5% kernel against 5.3%, `Probe.prefetch` 17.1% of samples against 9.9%.
The lever it named was intra-query parallelism, expanding one query's frontier
across workers so its misses overlap. That means giving up §6.3's
one-thread-per-core pinning for the single-query case, which is the discipline
that holds every other row's tail (findings 36, 41: zero migrations against
Qdrant's tens of thousands, and W4-sat90 p99 1.47 ms against 4.02 ms).

Measured 2026-09-22 before building it. The same row at rising concurrency, on
the theory that concurrency buys for free the overlap intra-query parallelism
would buy with threads:

| | qps | cycles/query (whole engine) | server p50 |
|---|--:|--:|--:|
| `-p 1` | 1,323 | 1,310,640 | 640 µs |
| `-p 2` | 3,392 | 1,012,563 | 480 µs |
| `-p 4` | 8,093 | 862,530 | **398 µs** |
| `-p 8` | 15,023 | 913,887 | 438 µs |
| `-p 16` | 16,208 | 905,231 | 893 µs |

**A single query's own latency falls 38% as others arrive**, bottoming at 398 µs
before queueing takes over. That floor is what the traversal costs when the
memory system is well used, and it is the ceiling any amount of overlap can
reach. Qdrant's W3 is 411 µs.

So the arithmetic against the change is: the next-candidate prefetch already
took W3's server p50 from 488 to 459 µs, the floor is ~398 µs, and the remaining
headroom is therefore about 60 µs, 13% of one row. Against that, the discipline
being given up is worth 2.7x on W4-sat90's p99 across every row. The trade is
not close, and intra-query parallelism is dropped rather than deferred.

*Caveat on the table.* `cycles/query` here is the whole engine process over the
row's queries, so idle workers are in it and the absolute figures are not the
harness's per-row counters; the shape and the server-side p50s are what this
reads. And part of the fall from `-p 1` to `-p 4` is inter-query cache sharing
rather than intra-query overlap, which a threaded single query would not get.
Both caveats push the same way: the real headroom is at most the 60 µs above.

## findings 34's per-build draw is the upload, not the builder

Every published run measured strawmANN's graphs disagreeing with each other by
more than the two engines differ. rel-0921's three passes read recall@10 at
`ef` 512 of 0.99957 / 0.99789 / 0.99796, a spread of **0.00168** against
Qdrant's **0.00004**; rel-0908's read 0.00320 against 0.00006. findings 34
named `buildParallel`'s thread interleaving, because `linkBack` re-prunes in
arrival order and the graph is legitimately a draw. It named the wrong thing.

Measured 2026-09-22 with `zig build graph-diff`, which builds one corpus N
times in one process and scores each build against its own exact ground truth.
SIFT1M, `m=16 ef_construct=100`, recall@10 over all 10,000 held-out queries,
five builds per row, spread across the five:

| how the corpus was inserted | threads | ef 128 | ef 256 | ef 512 |
|---|--:|--:|--:|--:|
| file order | 24 | 0.00011 | 0.00003 | 0.00002 |
| file order | 8 | 0.00004 | 0.00002 | **0.00000** |
| arrival reordered, per point | 24 | 0.00062 | 0.00033 | 0.00021 |
| arrival reordered, batches of 100 | 8 | 0.00026 | 0.00048 | 0.00024 |

**The pruning race is worth 0.00002.** Five builds in file order at 8 threads
returned 0.99955 five times. That is below Qdrant's own 0.00004 and eighty
times below what the published runs show, at the thread count the server uses
and at the one that races hardest. Reordering *arrival* is what moves it, and
it also costs 0.0007 of recall outright.

### Why arrival order is not a detail

`ids.IdSpace.reserve` hands out internal offsets from a monotone counter as
points arrive. `hnsw.assignLevel` is a pure function of the offset. W2 uploads
with `-b 100 -t 8 -p 8`. So which vector becomes node 7 is decided by eight
racing streams, and with it that point's level, the membership of every upper
layer, and the entry point. Two uploads of one corpus are two different graphs
before the builder has pruned anything.

`graph-diff` says how different. Two builds in file order share 99.7% of their
level-0 edges, agree about the entry point, and differ by a handful of edges
above level 1 (four of 4,048 at level 3, none at all in some pairs). Two builds
under reordered arrival share **35.8%** of level-0 edges, agree about *no*
upper-level row, and pick different entry points, with 11.7% of points sitting
at a different level.

### The measurement that settles it

`bench/harness/upload_order_ab.py`, the real engine and the real upload, three
passes per arm, changing one flag:

| arm | ef 128 | ef 256 | ef 512 |
|---|--:|--:|--:|
| `-t 8 -p 8`, as every published run uploads | 0.00111 | 0.00114 | **0.00120** |
| `-t 1 -p 1`, one stream | 0.00003 | 0.00001 | **0.00000** |

The serial arm returned 0.99955 at `ef` 512 in all three passes, which is the
same figure the in-process file-order builds returned, and is more reproducible
than Qdrant. The concurrent arm reproduced the published behaviour, including
its shape: two passes at 0.99958 and 0.99955 and one bad draw at 0.99838. That
bimodality is what rel-0921 (0.99957, 0.99789, 0.99796) and rel-0908 (0.99951,
0.99951, 0.99631) both show, and it is why a single pass could report either
engine ahead at high recall.

The spread is also flat across `ef` in both the published runs and the
concurrent arm, which is the tell that was there all along: a graph that is
merely *worse built* loses less as the search widens, and these do not.

### Which half of "arrival order" does the damage

A reordered arrival changes two things at once: the level each point is drawn
into, because `assignLevel` is keyed on the offset, and the sequence the points
are linked in. `Graph.level_keys` separates them, drawing the level from the
point instead. Five builds, the same five arrival orders in both columns, so
this is paired:

| build | level from the arrival slot | level from the point |
|--:|--:|--:|
| 0 | 0.99883 | 0.99883 |
| 1 | 0.99888 | 0.99883 |
| 2 | 0.99895 | 0.99893 |
| 3 | 0.99881 | 0.99884 |
| 4 | 0.99871 | 0.99870 |
| spread | 0.00024 | 0.00023 |

**Nothing.** Stabilising the level assignment leaves both the spread and the
level of recall exactly where they were, and both stay 0.0007 below what file
order returns, despite the level draw being the part of this that looks most
like a bug.

The obvious next candidate is the other half, the sequence, and
`Graph.insert_order` tests it the same way: link the points in vector order
whatever order they arrived in. **It is worse.** Five
builds, batch-100 arrival, `ef` 512:

| | recall | spread |
|---|--:|--:|
| arrival order, arrival levels | 0.99871 to 0.99895 | 0.00024 |
| arrival order, point levels | 0.99870 to 0.99893 | 0.00023 |
| vector order, arrival levels | 0.99634 to 0.99880 | **0.00246** |
| vector order, point levels | 0.99646 to 0.99665 | 0.00019 |
| file order (both, by construction) | 0.99955 | 0.00000 |

Fixing the sequence alone multiplies the spread tenfold. Fixing both together is
tight again, and 0.0029 *below* the file-order build it was meant to reproduce,
where the only remaining difference is which points sit at which level.

### The level draw is worth more than the whole published spread

The cleanest arm says it alone. File order, one insertion sequence, three
builds, changing nothing but the hash the level is drawn from (the node id as
four bytes, against the same number as eight):

| level drawn from | ef 128 | ef 512 |
|---|--:|--:|
| `assignLevel(node)` | 0.98870 / 0.98873 / 0.98874 | 0.99955 x3 |
| `assignLevelKey(node)` | 0.98665 / 0.98667 / 0.98665 | 0.99727 / 0.99726 / 0.99727 |

Each is reproducible to 0.00002 and they are **0.0023 apart** at `ef` 512, which
is larger than the 0.00168 this entry is about. Two equally valid pseudo-random
level assignments over one corpus, in one insertion order, build graphs that
differ by more than the published pass-to-pass draw.

And it is not an artifact of that one hash. The same build at four level seeds,
file order, two builds each, recall@10:

| seed | ef 128 | ef 512 |
|---|--:|--:|
| `0x57ea3111` (the engine's default) | 0.98871 / 0.98872 | 0.99955 / 0.99955 |
| `0x1` | 0.98883 / 0.98884 | 0.99955 / 0.99955 |
| `0x2` | 0.98665 / 0.98663 | 0.99739 / 0.99738 |
| `0x3` | 0.98739 / 0.98740 | 0.99794 / 0.99794 |

Each seed reproduces itself to 0.00001 and the four span **0.00216**. §8.7
records the seed in `meta.json` as provenance; what it does not record is that
the choice is worth more than the pass-to-pass draw this entry set out to
explain, and that two of these four sit at what looks like a ceiling while two
do not. Nothing in the comparison is wrong because of it, since both engines are
measured as configured, but a recall figure quoted without its seed is quoted
without 0.002 of its own uncertainty.

That is where this stops, because it changes the question. The upload's arrival
order is demonstrated as *what varies between passes*, the serial arm being
exactly reproducible and the concurrent one not; but neither of its halves is
separately fixable. Stabilising the levels changes nothing, stabilising the
sequence makes it worse, and stabilising both lands on a different and worse
graph than file order builds. `assignLevelKey`, `Graph.level_keys` and
`Graph.insert_order` stay as the instruments that measured this; nothing in the
engine sets any of them.

### What this does not say

Nothing here exonerates the pruning race as a source of *some* variation, only
as the source of this one. It is measurable, at 0.00002, and it is eighty times
too small to be what the reports carried. Unreachable nodes were already
eliminated at two hundred times too small (findings 34, `repairUnreachable`),
and both eliminations have the same shape: a real effect of the right kind and
the wrong magnitude.

The remaining candidates from findings 34's list are retired with it. A
deterministic insertion order and refusing to prune a last in-edge were both
answers to a question whose premise was wrong.
