#!/usr/bin/env python3
"""§4's workload table, W0-W13, as data rather than as shell.

This replaces `run-w0-w13.sh` and `run-workload.sh`, which both encoded §4's
table independently, two definitions of the same matrix, free to drift, and
they had. The table is now a list of `Workload` objects: one definition, read by
the runner, the comparison and anything else that needs it.

Bash was the wrong tool for this. Every real bug in the runner came from
something bash makes awkward:

  * `-n` sizes both the upload and the query loop, so search rows need a
    separate invocation, expressed here as two workloads with a shared
    collection rather than as duplicated command lines;
  * bfb reports `rps` (batch requests) and `qps` (queries), and reading the
    wrong one turned a 2.2x speedup into an apparent 7x regression. The unit is
    a field here, not a grep;
  * per-row load tracking, foreign-process detection and JSON parsing are all
    things `awk` can do and none it does clearly.

`bench/setup.py` is the §7.1 host gate: it reads `/sys` and `/proc` and stamps
its verdict into every result directory.

Every row also records what the engine cost in storage and I/O, via
`procstat`: bytes on disk, peak RSS, and block-layer operations and bytes per
row. `--storage <path>` (or `$STORAGE_DIR`) says where the engine's data lives
when that cannot be discovered; a qdrant container's bind mount is found
automatically.

Usage:
    workloads.py list
    workloads.py run <uri> <label> [W3 W4 ...]
    workloads.py run <uri> <label> --report     also write the HTML report
    workloads.py run <uri> <label> --storage "$STRAWMANN_CACHE/qdrant-storage"
    workloads.py run <uri> <label> --sink       also record rows in the §8 sink
    workloads.py run <uri> <label> --min-duration   re-run rows shorter than 2 s with more queries
    workloads.py run <uri> <label> --n-factor 4     multiply every search row's -n
    workloads.py run <uri> <label> --no-warmup      skip the discarded warm-up pass (§7.1)
    workloads.py backfill <label>               recover latency from a finished run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# `bench/`, for `setup.py`: the gate's own constants and process lists, so the
# two samplers cannot disagree about what "ours" or "over budget" means.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perfstat

import paths
import procstat
import provenance
import setup

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))


def _find_bfb() -> Path:
    """Locate the pinned bfb binary.

    A single hardcoded default sent the ISA sweep looking under a path that has
    not existed on this machine for weeks, and every arm failed identically with
    a message naming that path. Searching the obvious places first makes the
    common case work and leaves $BFB as the override.
    """
    if os.environ.get("BFB"):
        return Path(os.environ["BFB"])
    for cand in (ROOT.parent / "bfb/target/release/bfb",
                 Path.home() / "Workspace/bfb/target/release/bfb",
                 Path.home() / "src/bfb/target/release/bfb"):
        if cand.exists():
            return cand
    return ROOT.parent / "bfb/target/release/bfb"


BFB = _find_bfb()

#: The dataset §4's table is written for, from the environment because
#: `fullrun.py --dataset` exports it into this subprocess. Every dataset-decided
#: parameter below (metric, width, corpus paths) derives from this name, so no
#: row can run against one dataset's corpus at another's dimension.
DATASET = os.environ.get("STRAWMANN_DATASET", "sift1m")


def use_dataset(name: str) -> str:
    """Choose the corpus for this process *and* every process it spawns.

    The mirror of `paths.use_data_dir`. `DATASET` is bound at import, before
    `fullrun.py` has parsed `--dataset`, so a process that imports and only then
    chooses holds the default — which `fullrun.py` did: every arm it spawned
    measured the right corpus while its own `required_capacity()` still answered
    for SIFT1M. Survivable only because SIFT1M needs the most of the three.

    Rebinding the globals covers this process and exporting `$STRAWMANN_DATASET`
    covers its children; doing both in one call is what stops them disagreeing.
    `METRIC` and `CREATE` too, or the new corpus's collections are created under
    the old corpus's distance.
    """
    global DATASET, METRIC, DIM, CREATE
    DATASET = name
    METRIC = os.environ.get("METRIC", paths.metric(name).capitalize())
    DIM = paths.dim(name)
    CREATE = ["--distance", METRIC, *collection_flags()]
    os.environ["STRAWMANN_DATASET"] = name
    return name


def use_segment_policy(name: str) -> str:
    """Bind the segment policy for this process and everything it spawns.

    The same two-part move `use_dataset` makes and for the same reason: the
    globals cover this process, `$SEGMENT_POLICY` covers the `workloads.py`
    subprocesses `fullrun.py` starts per arm, and doing both in one call is
    what stops them from disagreeing. A run whose two arms disagreed here
    would not be caught by `compare.py` as a wrong answer — it would be caught
    as STALE, since the policy is hashed into every row, which is a correct
    refusal of a run that should never have happened.

    `CREATE` is rebuilt because it is `collection_flags()` frozen at import,
    and the policy is exactly which flags those are.
    """
    global SEGMENT_POLICY, CREATE
    SEGMENT_POLICY = SegmentPolicy(name)
    os.environ["SEGMENT_POLICY"] = str(SEGMENT_POLICY)
    CREATE = ["--distance", METRIC, *collection_flags()]
    return str(SEGMENT_POLICY)


def corpus() -> Path:
    """The converted base, resolved on call so `--data-dir` reaches it.

    A module constant would be bound at import, which is before the flag is
    parsed, so every row would keep naming the default root.
    """
    return paths.dataset(DATASET)[0]


def corpus_queries() -> Path:
    """§4.2's held-out queries, in the same `fbin` the fp64 oracle read.

    The ground truth is keyed to this file's checksum, so a search row that
    uses it and a recall sweep that uses it are talking about the same queries.
    """
    return paths.dataset(DATASET)[1]


#: Queries a search row issues. Far past where the latency distribution
#: stabilises, and small enough to keep a row to seconds.
QUERIES = int(os.environ.get("QUERIES", 50_000))
#: Exact search scans the whole collection per query, so it gets its own count.
EXACT_QUERIES = int(os.environ.get("EXACT_QUERIES", 2_000))
#: The open-loop arms only. Their length is `-n / offered rate`, so they shorten
#: as the engine speeds up — at `QUERIES` the sift1m 90% arm lasted 2.5 s, barely
#: over `MIN_ROW_S`, and a tail percentile wants tens of seconds. 200,000 puts
#: the arms at ~10-35 s here. Closed-loop rows keep `QUERIES`: they saturate by
#: construction, so no chosen rate sets their length.
OPEN_LOOP_QUERIES = int(os.environ.get("OPEN_LOOP_QUERIES", 200_000))

#: W12 only, and no longer cut. The 5,000 this held was there because an engine
#: with its payload index suppressed scans every point per query, so the full
#: set did not terminate in reasonable time. Nothing suppresses it since
#: 2026-09-03: `--skip-field-indices` is gone from both halves of the row and
#: both engines score the matching set from the index, so the reason for the cut
#: went with it while the cut stayed.
#:
#: What it cost was a row too short to be a measurement. At 5,000 the filtered
#: row came in under `MIN_ROW_S`, so `--min-duration` scaled it -- and the
#: factor it settled on *straddled* the threshold: 4x on strawmANN for 2.10 s
#: against 2x on Qdrant for 1.96 s, which is still short and was published
#: saying so. Measured at 9,512 and 5,105 qps, the full count puts the row at
#: 5.3 s and 9.8 s and needs no scaling at all; a row that never scales cannot
#: straddle, which is a stronger guarantee than `--n-pin` freezing whichever
#: side of 2 s the first pass happened to land on.
FILTERED_QUERIES = int(os.environ.get("FILTERED_QUERIES", QUERIES))
#: Points §4's table asks each loading row to upload. A *request*: the corpus
#: is the ceiling, and `upload_n()` applies it.
UPLOAD_N = int(os.environ.get("UPLOAD_N", 1_000_000))


def corpus_rows() -> int | None:
    """Rows the converted corpus actually holds, from its own fbin header.

    Not the descriptor's `n`, which is the size of the dataset upstream
    publishes: `dbpedia-openai-1m` declares 1,000,000 and its `base.fbin` holds
    990,000, because §4.2's query split is held out of the base. The difference
    is not cosmetic — bfb is *told* how many points to upload and slices the
    mmap unchecked (`fbin_reader.rs`), so asking for more than the file holds
    is a read past the end and a panic on the batch that crosses it.

    Resolved on call, so `--data-dir` reaches it. `None` when the corpus is not
    there: `workloads.py list` and every test that builds the table without a
    dataset on disk must still work, and a missing file is the preflight's
    error to report, not this function's.
    """
    try:
        with open(corpus(), "rb") as f:
            head = f.read(8)
        return int.from_bytes(head[0:4], "little") if len(head) == 8 else None
    except OSError:
        return None


def upload_n() -> int:
    """`UPLOAD_N`, capped at what the corpus holds.

    SIFT1M holds exactly 1,000,000 rows, so this was invisible for as long as
    there was one dataset — the constant and the corpus agreed by coincidence,
    and the comment on W11 below states the coincidence as a fact ("`sift1m.fbin`
    holds exactly UPLOAD_N rows"). Every other corpus is smaller, so `--dataset`
    turned that coincidence into a panic on the first loading row.

    Capping rather than refusing: a row that uploads the whole corpus is the
    row §4 describes, and 100,000 points is what "load the corpus" means for a
    100,000-point dataset. `run.json` records what was actually uploaded.
    """
    rows = corpus_rows()
    return min(UPLOAD_N, rows) if rows else UPLOAD_N


def w12_n() -> int:
    """`W12_N`, capped the same way: W12 loads its 200k from the corpus too."""
    rows = corpus_rows()
    return min(W12_N, rows) if rows else W12_N
#: W11 only. The write volume is a *fraction* of the collection — a fifth
#: arriving during the search is the experiment — so a smoke run at reduced
#: `UPLOAD_N` must scale it. 200k writes into a 50k collection is a far heavier
#: workload, not a smaller W11: the pending tail dwarfs the graph.
W11_N = int(os.environ.get("W11_N", 200_000))


def w11_n() -> int:
    """`W11_N` held at its stated *ratio* to what was actually uploaded.

    The ratio is the experiment, and `--dataset` shrinks the collection without
    anyone touching a knob — so the expectation had to become arithmetic. It was
    already wrong: on dbpedia-openai-100K the unscaled row appended 200,000
    points to a 100,000-point collection and reported `write overlap 4%`, its
    writer absent for almost all of a row calling itself mixed read/write.

    Scaled by `upload_n() / UPLOAD_N`, so a reduced `UPLOAD_N` and a smaller
    corpus take the same rule and SIFT1M is untouched.
    """
    if UPLOAD_N <= 0:
        return W11_N
    return max(1, round(W11_N * upload_n() / UPLOAD_N))


#: W11 and W11-steady: the appender's batch size. Named rather than repeated as
#: a literal because the throttle below is denominated in *batches* per second,
#: so the two cannot be allowed to drift apart.
W11_BATCH = 100

#: How long each mixed row spreads its append over, in seconds.
#:
#: Unthrottled, bfb lands the whole append in the first few percent of the
#: search: measured 0.09 s of writing against a 14.5 s search on strawmANN (1%
#: overlap) and 1.0 s against 19.0 s on Qdrant (5%). Both rows then reported
#: "the writer covered under 90% of either search", and what they measured was
#: the rebuild the append provoked rather than a concurrent write -- which is
#: not what a row called "search bench2 while N synthetic points append" says
#: it is. The volumes are the experiment (W11 above `rebuild_ratio`, W11-steady
#: deliberately below it) so the throttle changes *when* the writes land, never
#: how many.
#:
#: One span per row, taken from the slower engine's measured search on sift1m
#: plus margin: W11-steady ran 19.0 s and W11 51.3 s. The appender is started
#: before the search and the row waits for it, so a span that overshoots costs
#: wall clock and keeps the coverage at 100%, while one that undershoots puts
#: the caveat straight back. Raise them for a corpus whose mixed rows run
#: longer than sift1m's; the row says so whenever a span is too short.
W11_STEADY_SPAN_S = float(os.environ.get("W11_STEADY_SPAN_S", 25.0))
W11_SPAN_S = float(os.environ.get("W11_SPAN_S", 60.0))


def w11_throttle(points: int, span_s: float) -> int:
    """bfb's `-T`, in batches per second, so `points` take `span_s` to append."""
    return max(1, round(points / W11_BATCH / span_s))


#: W11-steady's write volume, as a fraction of the corpus, and below
#: `collection.rebuild_ratio` (0.10) on purpose. W11 sits at 0.20 on every
#: corpus, so it trips a from-scratch rebuild by construction and can only
#: measure the rebuild path (0.47x sift1m, 0.51x dbpedia-100K). This row is the
#: ordinary case: writes that do not provoke one.
W11_STEADY_RATIO = 0.05


def w11_steady_n() -> int:
    return max(1, round(upload_n() * W11_STEADY_RATIO))


#: W12's keyword cardinality, and with it the selectivity of one value.
#:
#: docs/workloads.md W12 point 2 asks for two grades, 10% and 1% of the
#: collection, as separate rows -- "an engine that picks well at one
#: selectivity and badly at the other reports a single figure that is true of
#: neither", and Qdrant's dispatch turns on exactly that threshold.
#:
#: The spec writes the condition as an integer `tag` in 0-99. The load
#: generator cannot express that: its integer filter is
#: `tag >= random(0..range)`, whose selectivity is uniform in [0,1] *per query*
#: rather than fixed, so a row built on it measures a different regime on every
#: request. Its keyword filter can -- with `-k V` each of V values covers about
#: `1/V` of the points -- so the grades are keyword conditions and the spec's
#: substance (two fixed grades, ground truth per condition, a `min(k, |M|)`
#: denominator) is met by a different mechanism.
#:
#: It was 1000, which is a 0.1% grade, and only one of them.
W12_KEYWORDS = int(os.environ.get("W12_KEYWORDS", 100))

#: How many of `W12_KEYWORDS` the wide grade accepts.
#:
#: The row and its sweep arrive at slightly different selectivities here, and
#: both figures are right about different things. bfb draws the values *with
#: replacement* per query, so the row's ten draws over a hundred values cover
#: `100 * (1 - (1 - 1/100)^10)` = 9.56% on average. `recall.py`'s sweep fixes a
#: list of ten *distinct* keywords, which is 10% — measured at 20,023 of
#: 200,000, against 2,010 for the narrow grade.
#:
#: Half a percentage point of selectivity apart, which is inside the
#: distribution-level caveat the rows already carry and does not reach the
#: recall: `min(k, |M|)` is `k` on both sides at 10 against 20,023. What the
#: page reports is the measured `n_matching`, never this arithmetic.
W12_MATCH_ANY = int(os.environ.get("W12_MATCH_ANY", 10))

#: W12's collection, sized independently of `UPLOAD_N` because the filtered
#: row's cost is cardinality against corpus size. Here rather than a literal in
#: `table()` so `required_capacity` can see it — as a literal it did not, and a
#: smoke run advertised 70,000 and hit the engine's own capacity refusal.
W12_N = int(os.environ.get("W12_N", 200_000))


def collections_read_after_rows() -> set[str]:
    """Collections still read once the stable-row phase is over.

    Two things read after it: the recall sweeps (`recall.COLLECTIONS`) and the
    mutating rows, which `fullrun` defers past the sweeps. Everything else a
    row created is finished with, and on a resident engine "finished with" and
    "still costing its full capacity" are the same state until something drops
    it.

    Imported here rather than duplicated: the sweep set is `recall.py`'s to
    define, and a list of collection names kept in step by hand is the bug
    `upload_collections` already exists to prevent.
    """
    import recall
    live = set(recall.COLLECTIONS)
    for w in table():
        if w.background and w.query_collection:
            live.add(w.query_collection)
    return live


def collections_droppable_after_rows() -> list[str]:
    """Collections nothing reads again once the stable rows are done.

    `bench1` is the one that matters: W1 measures ingest into it and no row,
    sweep or later phase touches it again, so it holds a full arena — 7.08 GiB
    at the 1M tier — from the first row to process exit for a measurement that
    finished minutes earlier.
    """
    live = collections_read_after_rows()
    return [c for c in upload_collections() if c not in live]


def collections_droppable_after_sweeps() -> list[str]:
    """Swept collections nothing reads once their sweep is done.

    The mutating rows run after the sweeps and query `bench2` alone, so the
    quantized collections can go before W11 starts — which is exactly when a
    rebuild of the whole of `bench2` wants the memory.
    """
    keep = {w.query_collection for w in table() if w.background and w.query_collection}
    return [c for c in collections_read_after_rows() if c not in keep]


def upload_collections() -> list[str]:
    """Every collection §4's table creates, in table order.

    `fullrun.py` reads back each engine's collection config after its arm, and
    the list it asked for was a literal in the conformance CLI's default. It
    named five of the seven, so `bench1` and `bench12` were missing from the
    report — the second of which is the collection W12's numbers are entirely
    about. Derived here so a row added to the table cannot be left out of the
    read-back by omission.
    """
    seen = []
    for w in table():
        a = [str(x) for x in w.args]
        if "--collection-name" not in a:
            continue
        name = a[a.index("--collection-name") + 1]
        if name not in seen:
            seen.append(name)
    return seen


def required_capacity() -> int:
    """Points the largest single collection will hold.

    The server preallocates one capacity for every collection (§3), so this is
    a max over collections, not a sum over rows. Two of them compete for it:
    `bench2`, which W2 fills and which W11 *and then* W11-steady append to, and
    `bench12`, which W12 fills on its own.

    Both W11 rows count. They append to the same collection in sequence, so
    leaving W11-steady out here left `bench2` 50,000 points short of what the
    last row needs, and the server refuses the row rather than growing.
    """
    return max(upload_n() + w11_n() + w11_steady_n(), w12_n())

#: bfb opens roughly `threads x connections` sockets. `-t 16 -c 8` wants ~128;
#: a server with fewer closes the excess *before* the HTTP/2 preface, so the
#: client reports only `transport error` with no status. W4 lost an entire run
#: to that.
W4_CONNS = int(os.environ.get("W4_CONNS", 2))

#: W9's client concurrency, eight by default. The override exists because at
#: `-p 1` the row is falsifiable: no engine can share a scan between queries, so
#: one above `aggregate_bandwidth / corpus_bytes` (119 qps on dbpedia-100K here)
#: is not scanning the corpus (validation 13).
W9_PARALLEL = int(os.environ.get("W9_PARALLEL", 8))

#: bfb's `wait_index` sleeps 1 s then needs three consecutive Green replies, so
#: no Time-to-Green below three seconds is real (a 0.064 s build reported
#: 3.008 s). A constant bias: the difference between engines survives it, the
#: ratio does not, and it understates the faster one.
WAIT_INDEX_FLOOR_S = 3.0

#: Pinned per §8.9 alongside the Qdrant version, recorded in every run, and
#: enforced: `check_bfb_pin` refuses a run whose checkout is not this commit.
#:
#: The patch that first set this pin is findings 32 — `--rps` reaped at most one
#: completion per rate-limiter tick, so every open-loop row measured the
#: generator's drain backlog. It was the one-commit fork `6f216634` until that
#: merged upstream as qdrant/bfb#172 (`0c1aafee`), and the fix is still in
#: everything since.
#:
#: Moved forward to `fc6632e5` on 2026-09-08, three commits on: #173, #174 and
#: #176, which add a serverless multi-collection benchmark mode. Advanced
#: rather than reverted because the whole of it is *additive* — `src/serverless/`
#: is new files and a new subcommand — and because the six shared files it
#: touches do not change the single-collection path §4's rows measure:
#:
#:   * `processor.rs` gains `request_count` / `request_size`, whose default
#:     implementations are the `total_items.div_ceil(get_batch_size())` and
#:     `get_batch_size()` that `stats.rs` used to inline. Same arithmetic, so
#:     the default processor's batch count and progress accounting are
#:     unchanged.
#:   * `client.rs` makes `retry_with_clients` generic over the client type so
#:     the serverless client can share one retry policy. Same policy.
#:   * `args/mod.rs` adds the `serverless` subcommand and widens
#:     `parse_number`'s visibility.
#:
#: Verified by reading the diff rather than assumed from the titles: "KeywordIndex
#: builder" in #176 is the serverless collections module, not the field index
#: W12 depends on, which bfb still builds the same way.
#:
#: A pin move is still a change of instrument. Rows measured under `0c1aafee`
#: and rows measured under this one carry different `bfb_pin` stamps, and
#: `compare.py` reads that stamp — which is the mechanism working, not a
#: problem to route around.
BFB_PIN = ("dev @ fc6632e5 (qdrant/bfb#176; carries #172, findings 32's "
           "--rps reaping fix)")

#: The short hash inside `BFB_PIN`, which is what the checkout is checked
#: against. Written once, parsed once, rather than repeated.
BFB_COMMIT = "fc6632e"
BFB_CLIENT = "qdrant-client 1.16.1-dev (git dev branch)"
HARNESS_CLIENT = "qdrant-client =1.19.0"

#: bfb's per-request `--timeout`, in seconds. docs/workloads.md §2 quotes it.
BFB_TIMEOUT_S = int(os.environ.get("BFB_TIMEOUT_S", 60))


def stale_bfb_binary(repo: Path) -> str | None:
    """Why this binary cannot have been built from the pinned commit, or None.

    The checkout being right is not the binary being right, and moving the pin is
    when the two come apart: `git checkout` was instant and
    `cargo build` was not, so for several minutes `check_bfb_pin` passed against a
    binary five days older than the commit it vouched for.

    One-way and therefore cheap: older than the commit definitely did not come
    from it, newer only might have. Nothing here rebuilds anything. Silent when
    either timestamp is unavailable — the absence of a check is not a verdict.
    """
    if not BFB.exists():
        return None
    r = subprocess.run(["git", "-C", str(repo), "show", "-s", "--format=%ct",
                        BFB_COMMIT], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip().isdigit():
        return None
    committed = int(r.stdout.strip())
    built = BFB.stat().st_mtime
    if built >= committed:
        return None
    why = (f"bfb at {BFB} was built {time.strftime('%Y-%m-%d %H:%M', time.localtime(built))}, "
           f"before {BFB_COMMIT} was committed "
           f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(committed))}.\n"
           f"  The checkout is pinned but the binary is not built from it, so "
           f"every row would be stamped with code that did not produce it.\n"
           f"  `cargo build --release --manifest-path {repo}/Cargo.toml`\n"
           f"  (or BFB_PIN_LAX=1 to measure with this one anyway)")
    if os.environ.get("BFB_PIN_LAX"):
        print(f"!! {why}", file=sys.stderr)
        return None
    return why


def check_bfb_pin() -> str | None:
    """Why this bfb checkout may not produce a published row, or None.

    §9 pins the toolchain because "a benchmark that silently changes its load
    generator is not a benchmark", and nothing enforced that for the one tool
    that generates every number: `BFB_PIN` was a string in this file and the
    binary was whatever had last been built next door. The two disagreeing is
    not a caveat — the stamp in `run.json` would name a commit that did not
    produce the row.

    Skipped when the checkout cannot be found or is not a git tree (a released
    tarball, CI): silence there is the absence of a check, and the run stamps
    what it was told. `BFB_PIN_LAX=1` downgrades a mismatch to a warning, for
    bisecting the generator itself.
    """
    repo = BFB.parent.parent.parent if BFB.parent.name == "release" else BFB.parent
    if not (repo / ".git").exists():
        return None
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    head = r.stdout.strip()
    if head.startswith(BFB_COMMIT):
        return stale_bfb_binary(repo)
    why = (f"bfb at {repo} is {head[:8]}, and this harness is pinned to "
           f"{BFB_PIN}.\n  Every row would be stamped with a commit that did not "
           f"produce it.\n  `git -C {repo} fetch origin dev && git -C {repo} "
           f"checkout {BFB_COMMIT} && cargo build --release` — it is plain "
           f"upstream now.\n  (or BFB_PIN_LAX=1 to measure with this one anyway)")
    if os.environ.get("BFB_PIN_LAX"):
        print(f"!! {why}", file=sys.stderr)
        return None
    return why


def arm_started(results: Path, now: str | None, session: str | None) -> str | None:
    """When this arm started, not when its last phase did.

    `provenance.collect` stamps `started` with the time it runs, and `run.json`
    is rewritten wholesale by every `workloads.py` invocation. `fullrun` calls
    it more than once per arm — the mutating rows are deferred past the recall
    sweeps and run as a second invocation — so `started` recorded the *last*
    call. That made a sift1m Qdrant arm look like it ran in seven
    minutes: the field said 21:29:30Z and the arm's first row was measured at
    21:11:46Z.

    So the earliest stamp within one `--session` wins, and a new session starts
    over. Keying on the session rather than on the file's existence is what
    stops a re-run of the same label inheriting the previous run's start —
    tonight's db100k re-run wrote the directory its failed attempt had left.
    """
    prev = results / "run.json"
    if not session or not prev.exists():
        return now
    try:
        old = json.loads(prev.read_text())
    except (OSError, json.JSONDecodeError):
        return now
    if old.get("session") != session:
        return now
    return min((s for s in (old.get("started"), now) if s), default=now)


def foreign_dataset(results: Path, dataset: str) -> str | None:
    """Why `results` must not be written by a run of `dataset`, or None.

    `rows.json` merges in place and `run.json` is rewritten wholesale, so a label
    reused for a second corpus ends up holding whichever rows the two runs did
    not both measure, stamped with the newer dataset — and nothing downstream can
    separate them again. `compare.stale_reasons` compares two *labels*, and both
    halves of this one agree with themselves.

    The same rule `recall_path` and `report.default_out` apply one and two
    directories down. Refused rather than bannered, and before anything is
    written: there is no correct reading of a merged directory. An unreadable or
    dataset-less `run.json` is not evidence of a conflict and does not refuse.
    """
    meta = results / "run.json"
    if not meta.exists():
        return None
    try:
        prev = (json.loads(meta.read_text()).get("dataset") or {}).get("name")
    except (OSError, json.JSONDecodeError):
        return None
    if prev is None or prev == dataset:
        return None
    return (f"it holds {prev} rows and this run measures {dataset}; rows.json "
            f"merges in place, so the two corpora would be indistinguishable "
            f"afterwards. Use another label (--strawmann-label/--qdrant-label, "
            f"e.g. {results.name}-{dataset}) or delete the directory.")


#: `--session`, for the rows measured by this invocation. Module-level because
#: it is a property of the process, not of any one row, and threading it
#: through `run_one` to `Result` would put it in eight signatures that have no
#: other use for it.
SESSION: str = ""


def harness_stamp() -> dict:
    """What the harness itself was when it measured, written into `run.json`.

    Two result sets are only comparable if the same table produced them, and
    the table changes: the metric default moved from cosine to Euclid, the
    query source moved from random vectors to the dataset's held-out set, and
    the published tables were built from runs taken before both changes with
    nothing in them that said so. `compare.py` refuses to ratio two labels
    whose stamps disagree on any of the fields that change what a row means,
    and treats a missing stamp as a disagreement.
    """
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                               text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() or None if r.returncode == 0 else None

    return {
        "commit": git("log", "-1", "--format=%H", "--", "bench/harness/workloads.py"),
        "head": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "bench/harness")),
        "dataset": DATASET,
        "metric": METRIC,
        "query_source": {w.id: ("dataset:" + w.query_strategy if w.query_collection
                                else "random") for w in table() if not w.upload_only},
        "upload_n": upload_n(), "w11_n": w11_n(), "w12_n": w12_n(),
        "queries": QUERIES, "exact_queries": EXACT_QUERIES,
        # The settle discipline, as the numbers that define it rather than a
        # bare flag: changing the threshold or the window changes how much of
        # an engine's background work lands in the next row.
        "engine_settle": {"cores": ENGINE_IDLE_CORES,
                          "window_s": ENGINE_IDLE_WINDOW_S,
                          "stable": ENGINE_IDLE_STABLE,
                          "timeout_s": ENGINE_IDLE_TIMEOUT_S},
        "filtered_queries": FILTERED_QUERIES,
        "bfb_pin": BFB_PIN, "bfb_timeout_s": BFB_TIMEOUT_S,
        "collection": collection_settings(),
        "ef": {w.id: ef_of(w) for w in table() if not w.upload_only},
        "qps_definition": "n_queries / duration_secs from bfb's JSON (qps_bfb_median kept beside it)",
    }


#: The `run.json` fields two labels must agree on before their rows can be
#: ratioed. Each changes what the row *is*: a cosine collection is not a Euclid
#: one, a random query is not a dataset query, a Qdrant collection in eight
#: segments below its indexing threshold is a plain scan.
#: `compare.stale_reasons` checks them per label; `stamp_hash` folds them into
#: every row.
#:
#: **A key here has to earn the invalidation of every published row**, since
#: adding one changes `stamp_hash` for all of them. `perf` and `dataset` do not
#: earn it: both are already guarded elsewhere (`perf_set` on the row,
#: `dataset.name` in `stale_reasons`), so adding them would only make unchanged
#: rows report themselves as re-run. `engine_settle` does: it moves the row
#: after an upload by up to 1.7x — Qdrant's W3 went 1,180/2,013/1,402 to
#: 1,951/1,965/1,977 when it landed (findings 44), inverting the headline — and
#: the affected row carries no local signal, so without it `regression.py`
#: serves a 28.16% floor over rows whose spread is 0.66%.
STAMP_KEYS = ["metric", "query_source", "upload_n", "w11_n", "queries",
              "exact_queries", "collection", "ef", "bfb_pin", "engine_settle"]


def stamp_hash(stamp: dict) -> str:
    """One row's identity under the harness: `STAMP_KEYS`, hashed.

    `run.json` carries the stamp of the *last* invocation and `rows.json`
    merges rows in place, so after `workloads.py run <uri> <label> W3` the file
    holds one row measured under today's harness beside twenty measured under
    whatever it was before, and the label-level stamp vouches for all of them.
    Each row therefore carries a hash of the stamp it was measured under, so a
    comparison can refuse the rows the stamp does not describe. The `-n` a row
    actually ran with is *not* part of it: qps is a rate, and `--min-duration`
    doubles `-n` for the faster engine only, by design. It is recorded on the
    row (`n_requested`) and noted when it differs from the table's.
    """
    import hashlib
    key = {k: stamp.get(k) for k in STAMP_KEYS}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Gate(StrEnum):
    """§7.1's verdict for a row, or for the arm a set of rows makes up.

    The same argument as `Status` below, for the field beside it: the values
    were spelled out in six files and `Result` typed two of its four state
    fields. `mixed` is the one that made it worth doing — it is not a value any
    single row carries, only what an *arm* reads when its rows disagree, and a
    reader written as `gate == "FAIL"` silently treats a half-gated arm as
    passing. `compare.gate_of_rows` is the only thing that produces it.

    A `str` subclass, so `rows.json` is byte-identical and every file already
    on disk still reads.
    """

    #: Every check passed when this row was measured.
    passed = "pass"
    #: At least one did not. §7.1: the row is development-grade, and the
    #: environment hash it carries says so.
    failed = "FAIL"
    #: Only an arm, never a row: its rows do not agree, because
    #: `workloads.py run` re-checks per invocation and `fullrun` calls it more
    #: than once per arm. A Qdrant arm was 25 `FAIL` and one
    #: `pass`, and was published as passing.
    mixed = "mixed"


class Status(StrEnum):
    """What a row's measurement means.

    A `StrEnum` rather than three string literals: the values are compared in
    four files (`compare.py`, `report.py`, `results.py` and `check.py`'s
    fixture) and were spelled out at every site. Being a `str` subclass keeps
    `rows.json` byte-identical, so this is a type for the code and nothing for
    the artifact.
    """

    #: Measured, and the number means what the row says it means.
    ok = "ok"
    #: The engine declined the workload, naming the construct (§12). Not a
    #: failure: that would make a documented boundary look like a bug.
    #:
    #: Two boundaries reach this status and they read differently. §1's
    #: non-goals are permanent (sharding, replication, sparse, disk-resident);
    #: §2's phases and §10's optional milestones are merely not built yet.
    #: Calling the second a §1 non-goal publishes "never" where the spec says
    #: "not yet", and a throughput ratio reads differently under each.
    not_applicable = "n/a"
    failed = "FAILED"
    #: bfb exited 0 and left neither its JSON nor a `Median qps` line. A
    #: failure, not a measurement: the row used to be filed `ok` with
    #: `qps=None`, and `compare.joined` then skipped it as an upload row, so
    #: it vanished from every table rather than reading as broken.
    no_output = "no-output"


class Placement(StrEnum):
    """Where a collection's vectors live, as `--memory-vectors` asks for it.

    Provenance rather than configuration: the engines do not offer the same
    set, and `cached` is the only one both can serve, so the report shows the
    storage and memory rows per engine instead of comparing them.

    Note `pinned` is one of two unrelated vocabularies using that word: this
    one, and `RpsSource`, where it means the offered rate came from
    `--rps-reference` rather than the engine's own measured saturation.
    """

    cold = "cold"
    cached = "cached"
    pinned = "pinned"


class SegmentPolicy(StrEnum):
    """Which of the two Qdrant experiments a run is.

    Segments are Qdrant's intra-query parallelism, and the count is worth most
    of a headline: on dbpedia-openai-1m, forcing one segment moved Qdrant's
    saturating row 1,237 -> 2,584 qps and its single-query row 584 -> 353, while
    strawmANN stayed within 0.8% of itself (decisions.md).

    `equal-work`   one populated graph, so `ef` means the same on both sides,
                   which is what §8's T3 licensing needs. Needs
                   `max_segment_size` above the corpus, not just `--segments 1`
                   (findings 39). Answers: whose code is faster on the same
                   index structure.
    `as-deployed`  Qdrant's `default_segment_number: 0`, resolved to the CPU
                   count. What a user runs, and legitimately not strawmANN's
                   recall at a fixed `ef` — so compare at matched recall there.

    Two experiments, never one table. Stamped into `collection_settings()` and
    hashed into every row, so `compare.py` refuses a ratio across them as STALE.
    Unrelated to `setup.Profile.as_deployed`, which is the scheduler.
    """

    equal_work = "equal-work"
    as_deployed = "as-deployed"


class RpsSource(StrEnum):
    """Where an open-loop row's offered rate came from.

    §4's arms are a fraction of "measured saturation", and taken per engine
    that puts the two arms at two different offered loads — the error the
    fixed rates existed to avoid. `pinned` says one reference was used for
    both; `own` says this engine's own W4.
    """

    pinned = "pinned"
    own = "own"


class Direction(StrEnum):
    """Whether two instruments agree on which engine is ahead."""

    #: Both put the same engine ahead across every pair.
    same = "same"
    #: Both put the *other* engine ahead across every pair.
    opposite = "opposite"
    #: The pairs disagree among themselves — not a verdict about either
    #: engine, and unrelated to `Gate.mixed` despite the shared word.
    mixed = "mixed"


class LoadMode(StrEnum):
    """Which loop generated the load, which §7.4 makes load-bearing.

    "Use `--rps` for latency claims and closed-loop for saturation throughput.
    Never quote a closed-loop p99 as a latency result." A reader can only apply
    that if the row says which it was, and the code can only check it if the
    value is a closed set.
    """

    open_loop = "open-loop"
    closed_loop = "closed-loop"
    upload = "upload"
    #: Not recorded. Rows written before the field existed carry it, and
    #: `Result` defaulted to a bare `""` — which is why the annotation had to
    #: be widened to `LoadMode | str` and could not say what the field is.
    unset = ""


@dataclass(frozen=True)
class Workload:
    """One row of §4's table."""

    id: str
    purpose: str
    args: list[str]
    #: Rows that only load data. They produce no qps and are excluded from the
    #: comparison, but their timing is the W1/W2 measurement.
    upload_only: bool = False
    #: The collection whose queries come from the dataset; `None` keeps bfb's
    #: legacy flag path, where every query is `random_dense_vector(rng, dim)`.
    #:
    #: Not a tuning knob. Uniform random vectors sit where no SIFT data lives,
    #: so their traversal is not a real query's and the recall measured on the
    #: real query set does not describe it — which made every joined
    #: recall/throughput pair unsound. `bfb search --file <yaml>` fixes it.
    query_collection: str | None = None
    #: How the config path walks the query file. `from-start` is `req_id % n`
    #: for *every element of a batch* (bfb `generators/queries.rs`
    #: `gen_dense_vector`), so a `--search-batch-size 16` row sends sixteen
    #: copies of one query per request. `random-sample` draws each element
    #: independently, which is the only way the config path yields distinct
    #: dataset queries inside one batch. W5 uses it; see `table()`.
    query_strategy: str = "from-start"
    #: A second bfb invocation run *concurrently* with this row's own. W11 is
    #: the only user: it appends while `args` searches, which is what "mixed
    #: read/write" means and what one sequential bfb process (upload, then
    #: search) never measured.
    background: list[str] | None = None
    #: W12: the payload condition every query carries, as
    #: `(field, cardinality, match_any)`. `match_any` of `None` matches one
    #: value, so the selectivity is `1/cardinality`; an integer matches any of
    #: that many drawn values.
    #:
    #: Only the config path can send a dataset query *and* a filter, which is
    #: what makes a filtered row's recall joinable at all: the flag path's
    #: queries are `random_dense_vector`, and a recall measured on the real
    #: query set does not describe uniform random vectors (see
    #: `query_collection`).
    keyword_filter: tuple[str, int, int | None] | None = None
    #: Whether a recall sweep of `query_collection` describes this row. False
    #: for W11, whose collection is being mutated while it searches: the sweep
    #: measured a different corpus and joining it would borrow a number.
    recall_joinable: bool = True
    #: Why this row gets no ratio, when that is by design rather than by a
    #: missing measurement. `compare.py` prints it in place of "recall
    #: missing", which is what W11 said for as long as it existed and which
    #: reads as a sweep someone forgot to run.
    ratio_policy: str = ""
    #: A payload index is a precondition of a filtered row, not a tuning
    #: choice: Qdrant picks its path by filter cardinality, not by whether an
    #: index exists, so without one a low-cardinality filter checks every point
    #: — a full scan wearing a filtered search's name. `compare.py` reads the
    #: index state back out of `collections.json` and refuses the row unless the
    #: engine says it built one, because the failure is an engine quietly not
    #: doing what it was asked.
    needs_payload_index: bool = False
    #: §4's open-loop arms offer "R x measured saturation", R in {0.5,0.7,0.9}.
    #: Saturation is a property of the engine, so the row carries the fraction
    #: and the runner resolves it against W4's qps in the same session. A row
    #: with a fraction and no `--rps` is unrunnable on purpose: no default rate
    #: means anything.
    rps_fraction: float | None = None


def flags(*parts: object) -> list[str]:
    return [str(p) for p in parts]


def payload_index_suppressed(w: Workload) -> bool:
    """The row told bfb not to create field indices, so the engine has none.

    The one direction in which the harness's *intent* is evidence: asking for
    no index guarantees there is none, while asking for one guarantees nothing
    — which is why `Workload.needs_payload_index` reads the engine back instead
    of trusting the flags. W12 carries `--skip-field-indices` because strawmANN
    answers `UNIMPLEMENTED` for `CreateFieldIndex` and bfb `unwrap()`s that into
    a panic; the flag keeps the row runnable and costs Qdrant its index, and
    both halves of that belong in the published row rather than in a comment.
    """
    return "--skip-field-indices" in w.args


#: Collection settings passed explicitly to every collection-creating row, so
#: both engines are asked for the same thing and the request is on record. bfb
#: forwards them into `optimizers_config`/`hnsw_config` in kilobytes; strawmann
#: accepts the same fields (inert for the thresholds — it always builds the
#: graph). Recorded in `run.json` under `collection`.
#:
#: The thresholds mean "index everything, scan nothing", and they are 10 KB
#: rather than a size chosen per tier because an absolute size stops applying at
#: the next tier up. Qdrant's own 10,000 KB default left W0 — 1M x d=4, 2 MB per
#: segment — with no HNSW graph at all, so the row that exists to isolate
#: traversal cost compared a graph against a scan; 1000 KB fixed the 1M tier and
#: the same bug returned at 100k, unindexing 27,500 of 100,000 points and
#: reading 6.05x. 10 is Qdrant's floor ("value 1 invalid, must be 10 or larger",
#: a create-time error that would fail every row on the arm) and still far under
#: any bench segment at any tier.
QDRANT_MIN_FULL_SCAN_KB = 10

#: Which experiment this run is. `equal-work` is the default because it is
#: what §8's comparative licensing needs; `as-deployed` is the one a reader
#: asks about and the one no ratio at equal `ef` is available for.
SEGMENT_POLICY = SegmentPolicy(os.environ.get("SEGMENT_POLICY",
                                              SegmentPolicy.equal_work))

#: The segment target asked of Qdrant, or None to ask for nothing and take its
#: own default. `SEGMENTS` remains an escape hatch for a one-off, and setting
#: it does not rename the experiment -- a run that overrides the number is
#: still whichever policy it declared, and the number is stamped beside the
#: name so the two cannot silently disagree.
def segments() -> int | None:
    env = os.environ.get("SEGMENTS")
    if env:
        return int(env)
    return 1 if SEGMENT_POLICY is SegmentPolicy.equal_work else None
INDEXING_THRESHOLD_KB = int(os.environ.get("INDEXING_THRESHOLD_KB", 1))
FULL_SCAN_THRESHOLD_KB = max(QDRANT_MIN_FULL_SCAN_KB,
                             int(os.environ.get("FULL_SCAN_THRESHOLD_KB",
                                                QDRANT_MIN_FULL_SCAN_KB)))
#: bfb's own default for `--on-disk-payload` is `true` (`args/mod.rs`); it is
#: passed rather than defaulted so the row carries it, per docs/workloads.md §2.
#: `false`, because spec §2 lists "no on-disk payload (`--on-disk-payload
#: false`)" as one of the documented semantic gaps: the harness passed `true`
#: for weeks against a spec that said the opposite, and the stamp carried it.
ON_DISK_PAYLOAD = os.environ.get("ON_DISK_PAYLOAD", "false")

#: The ceiling above which Qdrant's optimizer splits a segment, in kilobytes.
#:
#: `--segments 1` alone does not produce one segment. Measured on v1.19.0,
#: 990,000 x 1536 cosine: the collection settles at five, four populated
#: (188,200/208,600/251,100/342,100 vectors) plus an empty appendable. Qdrant
#: searches each at the full `ef` and merges, so a nominal `ef` of 128 explores
#: four graphs against strawmANN's one — which T3 reports as unequal recall, and
#: which voided the dbpedia-openai-1m comparative claim (findings 39).
#:
#: `default_segment_number` is a target the optimizer cannot reach while a merged
#: segment would exceed this. Above the whole corpus it can: the same collection
#: merged to one 990,000-vector graph in about three minutes.
#:
#: Sized from the corpus, not a constant, for the reason
#: `FULL_SCAN_THRESHOLD_KB` above learned the hard way. Four bytes per
#: component, times the largest collection any row builds, times four for
#: overhead and headroom.
def max_segment_size_kb() -> int | None:
    env = os.environ.get("MAX_SEGMENT_SIZE_KB")
    if env:
        return int(env)
    if SEGMENT_POLICY is not SegmentPolicy.equal_work:
        # `as-deployed` is Qdrant's own configuration, and a ceiling is not
        # part of it. Raising one is the mechanism that *defeats* the default
        # segment count, so passing it here would make the two policies differ
        # by less than their names claim.
        return None
    corpus_kb = required_capacity() * DIM * 4 // 1024
    return max(1, corpus_kb * 4)


def policy_means() -> str:
    """What the policy did to the collection, in one clause.

    Shared by `pin.txt` and the report banner so the two cannot drift into
    describing the same run differently.
    """
    if SEGMENT_POLICY is SegmentPolicy.equal_work:
        return "Qdrant held to one populated graph, as strawmANN serves"
    return "Qdrant's own default_segment_number, resolved to the CPU count"


def collection_flags() -> list[str]:
    """What every collection-creating row asks for.

    The segment flags are omitted entirely under `as-deployed` rather than
    passed with a default-looking value: "Qdrant's own default" means the one
    Qdrant chooses, and a harness that names the default is a harness that
    pins it to whatever the default was on the day someone wrote the number
    down. The indexing and full-scan thresholds stay in both policies -- they
    decide whether an HNSW graph is built at all, which is not the axis here
    and which W0 needs settled the same way either way.
    """
    seg = segments()
    ceiling = max_segment_size_kb()
    return flags(*(("--segments", seg) if seg is not None else ()),
                 *(("--max-segment-size", ceiling) if ceiling is not None else ()),
                 "--indexing-threshold", INDEXING_THRESHOLD_KB,
                 "--full-scan-threshold", FULL_SCAN_THRESHOLD_KB,
                 "--on-disk-payload", ON_DISK_PAYLOAD)


def collection_settings() -> dict:
    """The collection configuration every creating row asks for, for `run.json`.

    docs/workloads.md §2 says these are "recorded per row even when
    defaulted"; until this existed they were recorded nowhere. Placement and
    HNSW parameters are left at each engine's default and *say so*, which is
    different from being silent about them.
    """
    return {
        # The name first, because the numbers under it are what the name means
        # and a report that prints `segments: 1` has said which knob moved
        # without saying which question is being asked.
        "segment_policy": str(SEGMENT_POLICY),
        "segments": segments() if segments() is not None else "engine default",
        "max_segment_size_kb": max_segment_size_kb() or "engine default",
        "indexing_threshold_kb": INDEXING_THRESHOLD_KB,
        "full_scan_threshold_kb": FULL_SCAN_THRESHOLD_KB,
        "on_disk_payload": ON_DISK_PAYLOAD,
        # Not the *requested* placement: that is per engine, and this dict is
        # hashed into every row's identity, so an arm-specific value would make
        # `compare.py` refuse every ratio as STALE. The request is in `run.json`
        # (`memory_vectors_requested`) and the outcome in `collections.json`.
        "memory_placement": "per engine: see run.json memory_vectors_requested "
                            "and collections.json placement",
        "hnsw_m": "engine default", "hnsw_ef_construct": "engine default",
        "datatype": "float32 (bfb default)",
        "hnsw_inline_storage": False,
    }


def search_config(collection: str, strategy: str = "from-start",
                  keyword_filter: tuple[str, int, int | None] | None = None) -> str:
    """`bfb search --file` config sourcing queries from the dataset.

    `strategy: from-start` walks the query file in order rather than sampling
    it, so two engines measured with the same `-n` see the same queries in the
    same sequence. A random sample would make the two runs comparable only in
    distribution, and §4.1 wants them comparable point for point.

    The exception is a batched row. bfb builds every element of a batch from
    the same `req_id` (`search/from_config.rs`: `(0..batch).map(|_|
    make_query_for(template_idx, req_id))`), and `from-start` maps `req_id` to
    one file index, so a 16-query batch is one query sixteen times: the server
    answers from a hot cache and the row measures batching overhead rather than
    batched search. `random-sample` is the config path's only per-element
    draw, so W5 takes distribution-level comparability (still the dataset's
    queries, still the same distribution on both engines) over point-for-point.
    """
    assert strategy in ("from-start", "random-sample"), strategy
    out = (
        "collection:\n"
        f"  name: {collection}\n"
        "requests:\n"
        "  - kind: dense\n"
        "    source:\n"
        "      type: file\n"
        f"      path: {corpus_queries()}\n"
        f"      strategy: {strategy}\n"
    )
    if keyword_filter is not None:
        field, cardinality, match_any = keyword_filter
        # `cardinality` must be the count the *upload* generated, or the filter
        # draws values no point carries and every query matches nothing --
        # which reads as an engine returning nothing rather than as a config
        # that asked the wrong question.
        out += (
            "    filters:\n"
            f"      - name: {field}\n"
            "        type: keyword\n"
            "        source:\n"
            "          type: random\n"
            f"          cardinality: {cardinality}\n"
        )
        if match_any is not None:
            out += f"        match_any: {match_any}\n"
    return out


#: The metric every collection is created with, derived from `DATASET` because a
#: literal is right for exactly one corpus.
#:
#: bfb defaults to `Cosine` and nothing overrode it, so every collection was
#: cosine while the ground truth, `recall.py` and the differ were all euclid.
#: Recall survived at ~0.99 (on SIFT the neighbour sets largely coincide) so
#: nothing looked wrong; MRDE read 0.996 against a true 0.001, comparing a
#: cosine similarity in [0,1] to a euclid distance around 300.
METRIC = os.environ.get("METRIC", paths.metric(DATASET).capitalize())

#: The vector width every corpus-loading row sends as `-d`. From the descriptor
#: for the same reason: bfb is told the dimension rather than reading it, so a
#: stale one uploads a corpus at the wrong width and still indexes it.
DIM = paths.dim(DATASET)

#: What every collection-creating row starts with: the metric and the
#: collection settings above. One spelling, so no row can drop a flag.
CREATE = ["--distance", METRIC, *collection_flags()]

C = "bench"


def table() -> list[Workload]:
    """§4's table. `docs/workloads.md` carries the per-flag reasoning."""
    rows: list[Workload] = [
        # W0, transport floor. d=4 makes the vector irrelevant; -p 1 makes the
        # number a per-request latency. The index must exist, or this measures
        # a brute-force scan of a million vectors instead: it read 34.9 qps that
        # way, against 8,230 with an index.
        Workload("W0-upload", "transport floor: load",
                 flags(*CREATE, "--collection-name", f"{C}0", "-n", upload_n(), "-d", 4), upload_only=True),
        # `upload_n()` even though these points are synthetic: W0 is the floor
        # for the rows below it and traversal cost scales with graph size, so a
        # 1M-point W0 is not the floor of a 100k-point W3. It must also fit
        # `required_capacity()`, which budgets the capped rows.
        Workload("W0", "d=4 floor: graph traversal with the distance taken out",
                 flags("--collection-name", f"{C}0", "--skip-setup", "-n", QUERIES, "-d", 4,
                       "--search", "-p", 1)),

        # W1, ingest. fbin so the corpus is the one the fp64 oracle read
        # (§4.1); its checksum matches the ground truth's base checksum.
        Workload("W1", "ingest throughput",
                 flags(*CREATE, "--collection-name", f"{C}1", "--fbin", corpus(), "-n", upload_n(), "-d", DIM,
                       "-b", 100, "-t", 8, "-p", 8, "--skip-wait-index"), upload_only=True),

        # W2, index build. Time-to-Green is the measurement, and the two
        # engines do not mean the same thing by it: Qdrant indexes during
        # ingest, strawmANN bulk-builds after.
        Workload("W2", "index build time",
                 flags(*CREATE, "--collection-name", f"{C}2", "--fbin", corpus(), "-n", upload_n(), "-d", DIM,
                       "-b", 100, "-t", 8, "-p", 8), upload_only=True),

        Workload("W3", "search, fp32, single query",
                 flags("--collection-name", f"{C}2", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128, "-p", 1), query_collection="bench2"),

        # W4, saturating, plus the open-loop arms. `--parallel` is a closed
        # loop: a stalled server stops receiving requests, so latencies it did
        # not serve never appear and the tail is understated, more so for the
        # slower engine.
        Workload("W4", "search, saturating (closed loop)",
                 flags("--collection-name", f"{C}2", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128,
                       "-p", 64, "-t", 16, "-c", W4_CONNS), query_collection="bench2"),
    ]

    # §4 asks for "R x measured saturation", R in {0.5, 0.7, 0.9}. These were
    # the constants 500/1000/2000, which at strawmANN's 22,000 qps offered 2% of
    # capacity — no queue forms there, so the rows measured the generator's own
    # scheduling and the report had to disown both loops as latency results.
    # Named for the fraction, not the rate: a row called `W4-rps500` that offers
    # 11,420/s describes itself as the opposite of what it measures.
    for pct in (50, 70, 90):
        rows.append(Workload(
            f"W4-sat{pct}", f"search, fixed rate at {pct}% of saturation (open loop)",
            flags("--collection-name", f"{C}2", "--skip-setup", "-n", OPEN_LOOP_QUERIES,
                  "--search",
                  "--search-limit", 10, "--search-hnsw-ef", 128,
                  "-t", 16, "-c", W4_CONNS),
            query_collection=f"{C}2", rps_fraction=pct / 100))

    rows += [
        # W5's batches must hold sixteen *different* queries. bfb's config path
        # builds every element of a batch from one `req_id`, and `from-start`
        # maps that to one file row, so the row used to send one query sixteen
        # times per request. `random-sample` is the config path's only
        # per-element draw (see `search_config`); the flag path would give
        # distinct queries too, but uniform-random ones far from the data.
        Workload("W5", "search batched (16 distinct dataset queries per request)",
                 flags("--collection-name", f"{C}2", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128, "--search-batch-size", 16),
                 query_collection="bench2", query_strategy="random-sample"),

        # W7 carries --quantization-rescore because oversampling without it is
        # inert: recall@10 is flat from 1x to 16x (README finding 6).
        Workload("W6-upload", "scalar quantization: load",
                 flags(*CREATE, "--collection-name", f"{C}6", "--fbin", corpus(), "-n", upload_n(), "-d", DIM,
                       "--quantization", "scalar"), upload_only=True),
        # §6.7's three encodings, all at the same `ef` so the encoding is the
        # only variable between them, and so a recall sweep has a stated key to
        # join on. They previously searched at whatever each engine defaults
        # to, which is not the same number on both sides.
        Workload("W6", "quantized: scalar",
                 flags("--collection-name", f"{C}6", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128,
                       "--quantization-rescore", "true"), query_collection="bench6"),
        Workload("W7-upload", "binary quantization: load",
                 flags(*CREATE, "--collection-name", f"{C}7", "--fbin", corpus(), "-n", upload_n(), "-d", DIM,
                       "--quantization", "binary"), upload_only=True),
        # Findings 24: binary quantization returns recall@10 ~0.026 on sift1m
        # for *both* engines, and 0.10 at 64x oversampling — one bit per
        # dimension carries almost no signal at d=128. The recall-equality rule
        # cannot catch it (0.0260 vs 0.0270 is inside the 0.01 tolerance), so a
        # ratio would compare how fast two engines return the wrong answer.
        # `compare.RECALL_FLOOR` refuses it at join time, where the recall is
        # known: as a `ratio_policy` sentence naming d=128 it was stamped onto
        # every dataset and refused a legal d=1536 ratio measuring 0.9076.

        Workload("W7", "quantized: binary + oversampling",
                 flags("--collection-name", f"{C}7", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128,
                       "--quantization-oversampling", 4,
                       "--quantization-rescore", "true"), query_collection="bench7"),
        Workload("W8-upload", "PQ: load",
                 flags(*CREATE, "--collection-name", f"{C}8", "--fbin", corpus(), "-n", upload_n(), "-d", DIM,
                       "--quantization", "product-x16"), upload_only=True),
        Workload("W8", "quantized: PQ",
                 flags("--collection-name", f"{C}8", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128,
                       "--quantization-rescore", "true"), query_collection="bench8"),

        Workload("W9", "exact / brute force",
                 flags("--collection-name", f"{C}2", "--skip-setup", "-n", EXACT_QUERIES,
                       "--search", "--search-exact", "--search-limit", 10, "-p", W9_PARALLEL),
                 query_collection="bench2"),
    ]

    # W10, recall control: latency only. Recall at matched ef comes from
    # `conformance relevance`, never from bfb (§4.1); the two join on ef.
    for ef in (32, 64, 128, 256, 512):
        rows.append(Workload(
            f"W10-ef{ef}", f"recall control, ef={ef} (latency only)",
            flags("--collection-name", f"{C}2", "--skip-setup", "-n", QUERIES, "--search",
                  "--search-limit", 10, "--search-hnsw-ef", ef, "-p", 8), query_collection=f"{C}2"))

    # W6's frontier, for the same reason W10 is bench2's. W6/W7/W8 are refused a
    # ratio because the engines size their rescore pools differently
    # (decisions.md §5), which left SQ8 — the one encoding usable on both
    # corpora — with no comparison at all. Sweeping `ef` and reading the frontier
    # vertically at a recall both engines reach is the same answer W10 gives
    # fp32; only the throughput half was missing.
    #
    # On sift1m Qdrant's SQ8 recall saturates near 0.974 against strawmANN's
    # 0.997, so the top of this frontier is a recall only one engine serves.
    for ef in (32, 64, 128, 256, 512):
        rows.append(Workload(
            f"W6-ef{ef}", f"SQ8 recall control, ef={ef} (latency only)",
            flags("--collection-name", f"{C}6", "--skip-setup", "-n", QUERIES, "--search",
                  "--search-limit", 10, "--search-hnsw-ef", ef,
                  "--quantization-rescore", "true"), query_collection=f"{C}6"))

    rows += [
        # W12, filtered. bfb builds the keyword index before the upload on both
        # engines — strawmANN answers `CreateFieldIndex` since 2026-09-03
        # (`core/payload.zig`, M7), so the `--skip-field-indices` this row
        # carried, and the full scan it forced on Qdrant, are gone.
        #
        # Both sides now make the same decision on the same threshold: search
        # path by estimated filter cardinality against `full_scan_threshold`
        # (Qdrant's `read_view/dispatch.rs`, our `handlers.searchOne`). `-k 1000`
        # over 200k points is ~200 matches per keyword, so both score the
        # matching set from the index. `compare.py` still reads the index back
        # from `collections.json` before licensing the row: flags say what was
        # asked, not what was built.
        #
        # Split into upload and search because bfb's `-n` sizes both phases, so
        # one row cannot hold the collection at 200k while cutting the query
        # count. `FILTERED_QUERIES` stays cut from the days Qdrant scanned;
        # raising it is a measurement decision.
        Workload("W12-upload", f"filtered search: load {w12_n():,} with payloads",
                 flags(*CREATE, "--collection-name", f"{C}12", "--fbin", corpus(), "-n", w12_n(),
                       "-d", DIM, "-k", W12_KEYWORDS), upload_only=True),
        # `-d` because this row has no query file: it is the only searching row
        # whose queries bfb *generates*, and generation is sized by `-d`, which
        # bfb defaults to 128. That default matched SIFT1M exactly, so the flag
        # was never missed until a corpus at another width created `bench12` at
        # `DIM` and this row asked it for 128-dimensional neighbours —
        # "expected dim: 1536, got 128", from the engine rather than from here.
        # Two grades, and on the *config* path rather than the flag path. The
        # flag path's queries are `random_dense_vector`, which sit where no
        # SIFT data lives, so no recall measured on the real query set
        # describes them -- and a filtered row whose recall cannot be joined is
        # the row §8 has been refusing all along. Only the config path sends a
        # dataset query and a filter in the same request.
        #
        # `-k` is gone from the search side: the cardinality lives in the
        # generated YAML, and the flag path's keyword count has no meaning on
        # the subcommand. It stays on `W12-upload`, which is what generates the
        # payloads the conditions select from.
        #
        # A caveat the join has to state rather than hide: bfb draws the
        # keyword value *per query*, so the row filters on a different value
        # each request while the sweep fixes one. The two therefore agree on
        # the selectivity grade and not on the condition, which is the same
        # distribution-level comparability W5 takes with `random-sample`. At
        # these grades the recall denominator is unaffected -- `min(k, |M|)` is
        # `k` for both, since 1% of 200,000 is 2,000 and k is 10 -- so what the
        # caveat costs is point-for-point identity, not the number.
        Workload("W12-sel1", f"filtered search, one keyword "
                              f"(~{100 / W12_KEYWORDS:.0f}% of bench12)",
                 flags("--collection-name", f"{C}12", "--skip-setup", "-n", FILTERED_QUERIES,
                       "--search", "--search-limit", 10, "--search-hnsw-ef", 128),
                 query_collection=f"{C}12", needs_payload_index=True,
                 keyword_filter=("a", W12_KEYWORDS, None)),
        Workload("W12-sel10", f"filtered search, any of {W12_MATCH_ANY} keywords "
                               f"(~{100 * (1 - (1 - 1 / W12_KEYWORDS) ** W12_MATCH_ANY):.0f}% "
                               f"of bench12)",
                 flags("--collection-name", f"{C}12", "--skip-setup", "-n", FILTERED_QUERIES,
                       "--search", "--search-limit", 10, "--search-hnsw-ef", 128),
                 query_collection=f"{C}12", needs_payload_index=True,
                 keyword_filter=("a", W12_KEYWORDS, W12_MATCH_ANY)),

        # A ladder per grade, narrow first, matching the order the grades are
        # in everywhere else (`FILTER_GRADES`, the row pair, the report's
        # labels). Rows run in *table* order, so this order is part of what a
        # run measures -- `W11-steady` before `W11` is load-bearing for exactly
        # that reason and is asserted below. This one is not load-bearing, and
        # saying so beside one that is costs nothing.
        #
        # The narrow grade's ratio is licensed at `ef=128` — both
        # engines sit at ~1.0 recall there — so this is not fixing a refusal.
        # It is the same question W12-sel10 answered badly with one point: a
        # single-point ratio says nothing about the shape of either curve, and
        # on the wide grade the shapes turned out to be opposite and the
        # published point near strawmANN's worst (findings 48). 1% selectivity
        # is the cheap grade, so knowing rather than assuming costs little.
        *[Workload(
            f"W12-sel1-ef{ef}",
            f"filtered recall control, one keyword, ef={ef} (latency only)",
            flags("--collection-name", f"{C}12", "--skip-setup", "-n", FILTERED_QUERIES,
                  "--search", "--search-limit", 10, "--search-hnsw-ef", ef),
            query_collection=f"{C}12", needs_payload_index=True,
            keyword_filter=("a", W12_KEYWORDS, None))
          for ef in (32, 64, 128, 256, 512)],

        *[Workload(
            f"W12-sel10-ef{ef}",
            f"filtered recall control, any of {W12_MATCH_ANY} keywords, "
            f"ef={ef} (latency only)",
            # Every flag `W12-sel10` sends, with only the width changed. It
            # carried `-p 8` for one measurement, copied from W10's ladder,
            # and the ef=128 rung then came in 3.49x (strawmANN) and 2.31x
            # (Qdrant) above the row it is supposed to reproduce -- a ladder
            # measuring a concurrency its own row does not. The rung *is* the
            # reproduction check, so it has to be the same search.
            flags("--collection-name", f"{C}12", "--skip-setup", "-n", FILTERED_QUERIES,
                  "--search", "--search-limit", 10, "--search-hnsw-ef", ef),
            query_collection=f"{C}12", needs_payload_index=True,
            keyword_filter=("a", W12_KEYWORDS, W12_MATCH_ANY))
          for ef in (32, 64, 128, 256, 512)],

        Workload("W13", "scroll / pagination",
                 flags("--collection-name", f"{C}2", "--skip-setup", "--scroll",
                       "-n", QUERIES, "-p", 8, "--search-limit", 10)),

        # Both mixed rows: two concurrent bfb processes, `background` appending
        # with `--skip-wait-index` while `args` searches `bench2` at ef=128. The
        # row's qps is the search's; the append rate and overlap are recorded
        # beside it (`background_*`, `overlap_s`). No recall — the collection is
        # mutated while searched, so no sweep describes it.
        #
        # Appends are synthetic and must be: the corpus holds exactly
        # `upload_n()` rows and bfb reads row `offset + i` from the mmap
        # unchecked, so `--fbin` past the corpus panics. `--offset` is also what
        # makes it an append rather than an overwrite of ids W2 filled, and the
        # collection ends at `upload_n() + w11_n()`, which `required_capacity`
        # sizes for.
        #
        # W11-steady runs FIRST and the order is the measurement: run after W11
        # it inherited ~9% pending, its own 5% crossed `rebuild_ratio`, and the
        # row named "below the rebuild threshold" provoked a rebuild and read
        # 0.39x against W11's 0.47x.
        Workload("W11-steady", f"mixed read/write below the rebuild threshold: search "
                               f"bench2 while {w11_steady_n():,} synthetic points append",
                 flags("--collection-name", f"{C}2", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128, "-p", 8),
                 query_collection=f"{C}2", recall_joinable=False,
                 ratio_policy="search-during-write; no recall join",
                 background=flags("--collection-name", f"{C}2", "-n", w11_steady_n(),
                                  "-d", DIM,
                                  "--offset", upload_n(), "-b", W11_BATCH, "-t", 8,
                                  "-T", w11_throttle(w11_steady_n(), W11_STEADY_SPAN_S),
                                  "-p", 8, "--skip-create", "--skip-wait-index")),

        # W11 runs LAST: it is the only row that mutates a collection others
        # read, appending to the `bench2` that W3, W4, W5, W9, W10 and W13
        # search. Ordered earlier it made W13 scroll 1.2M points instead of 1M,
        # so measuring W13 alone differed from measuring the table. Appending to
        # an already-indexed collection is the point, so the fix is ordering
        # rather than a private collection.
        Workload("W11", f"mixed read/write: search bench2 while {w11_n():,} synthetic "
                        f"points append (runs last)",
                 flags("--collection-name", f"{C}2", "--skip-setup", "-n", QUERIES, "--search",
                       "--search-limit", 10, "--search-hnsw-ef", 128, "-p", 8),
                 query_collection=f"{C}2", recall_joinable=False,
                 ratio_policy="search-during-write; no recall join",
                 background=flags("--collection-name", f"{C}2", "-n", w11_n(), "-d", DIM,
                                  "--offset", upload_n() + w11_steady_n(),
                                  "-b", W11_BATCH, "-t", 8, "-p", 8,
                                  "-T", w11_throttle(w11_n(), W11_SPAN_S),
                                  "--skip-create", "--skip-wait-index")),
    ]

    # The ordering is load-bearing, so assert it rather than trusting the next
    # reader to notice the comment.
    assert rows[-2].id == "W11-steady", \
        "W11-steady needs a bench2 with nothing pending, so it runs before W11"
    assert rows[-1].id == "W11", "W11 mutates bench2 hardest and runs last"
    return rows


# --------------------------------------------------------------------------
# Host state, §7.1 covers configuration, none of which notices a busy machine
# --------------------------------------------------------------------------

def load_pct() -> int:
    """1-minute load as a percentage of one core."""
    with open("/proc/loadavg") as fh:
        one = float(fh.read().split()[0])
    return int(one / os.cpu_count() * 100)


#: Processes that are *supposed* to be running during a measurement, matched by
#: prefix because Linux truncates `comm` to 15 characters: the ISA arms appear as
#: `strawmann-avx5` and `strawmann-base`, so exact matching flagged every ISA
#: sweep row as contaminated by its own engine.
_OURS = setup._OURS

#: An interpreter is ours only when it is running one of the harness's own
#: scripts. `python3` and `uv` used to be exempt by name, so a foreign
#: `python3 train.py` at 800% CPU was invisible to the very check that exists
#: to see it. Matched against the full command line (`ps -o args`).
_OUR_SCRIPTS = ("bench/harness/", "bench/setup.py", "scripts/check.py", "scripts/doctor.py",
                # The reference-rate calibration `perfstat` runs, which is a
                # bare `python3 -c` and therefore foreign by the rule above.
                perfstat.CALIBRATION_TAG)

#: ...and by basename, for a script run from its own directory (`./workloads.py`,
#: `python3 setup.py check` from `bench/`), where no path prefix is visible.
_OUR_SCRIPT_NAMES = ("workloads.py", "setup.py", "compare.py", "recall.py", "fullrun.py",
                     "report.py", "regression.py", "isa_sweep.py", "qdrant_ab.py",
                     "headline.py", "check.py", "doctor.py")


def _is_ours(cmdline: str) -> bool:
    argv = cmdline.split()
    if not argv:
        return False
    name = os.path.basename(argv[0])
    if any(name.startswith(p) for p in _OURS) or name in _OUR_SCRIPT_NAMES:
        return True
    # The sidecar this harness starts for the row it is measuring. Matched on
    # `perf stat -x,` — this harness's invocation and nobody else's — rather
    # than by adding `perf` to `_OURS`, because an operator's own `perf record`
    # during a row *is* foreign load. Without the exemption the harness condemns
    # its own measurement and re-runs it, naming a process it started itself.
    if name.startswith("perf") and " stat " in cmdline and "-x," in cmdline:
        return True
    if name.startswith(("python", "uv")):
        return (any(p in cmdline for p in _OUR_SCRIPTS)
                or any(os.path.basename(a) in _OUR_SCRIPT_NAMES for a in argv[1:]))
    return False


def foreign_from_ps(out: str, threshold: float = 20.0) -> str:
    """The busy foreign processes in `ps -eo pcpu=,args=` output."""
    busy = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pct = float(parts[0])
        except ValueError:
            continue
        cmd = parts[1].strip()
        if pct > threshold and not _is_ours(cmd):
            busy.append(f"{os.path.basename(cmd.split()[0])}({pct:.0f}%)")
    return " ".join(busy[:8])



def foreign_load(threshold: float = 20.0) -> str:
    """Busy processes that are neither the benchmark nor an engine, right now.

    A row measured while something else runs is not comparable to one measured
    idle, and the difference is invisible in the number. A Qwen inference server
    at 1008% CPU went unnoticed through an entire table before this existed.

    `ps`'s `pcpu` is a *lifetime* average, so this is a moment's view: the
    rows use `cpu_sample` / `foreign_between`, which difference `/proc` across
    the row and see only what ran during it.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pcpu=,args="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return ""
    return foreign_from_ps(out, threshold)


@dataclass
class CpuSample:
    """Every process's CPU seconds so far, and the machine's, at one instant."""
    t: float
    #: pid -> (user+system seconds, command line). Kernel threads carry their
    #: `comm` in place of a command line.
    procs: dict[int, tuple[float, str]]
    #: Non-idle seconds summed over all cores from `/proc/stat`, or None.
    busy_s: float | None
    #: pids with no command line: kernel threads, whose CPU is the kernel's
    #: (writeback, softirq on the engine's behalf) and not a foreign process's.
    kernel: frozenset[int] = frozenset()
    #: CPU seconds of this harness's *reaped* children so far
    #: (`getrusage(RUSAGE_CHILDREN)`). bfb starts and exits inside every row,
    #: so at `after` its pid is gone and its time is on the system-wide line
    #: only; without this it was reported as `exited-processes(124%)` on the
    #: very first gated row after the per-pid rewrite, and contamination
    #: refuses the ratio.
    children_s: float = 0.0


def _system_busy_s() -> float | None:
    text = procstat._read("/proc/stat")
    if not text:
        return None
    fields = text.split("\n", 1)[0].split()
    if len(fields) < 5 or fields[0] != "cpu":
        return None
    try:
        vals = [int(x) for x in fields[1:]]
    except ValueError:
        return None
    # user nice system idle iowait irq softirq steal ...: idle and iowait are
    # the two that are not work.
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return (sum(vals) - idle) / procstat._CLK_TCK


def cpu_sample() -> CpuSample:
    procs: dict[int, tuple[float, str]] = {}
    kernel = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        st = procstat._read(f"/proc/{pid}/stat")
        parsed = procstat.parse_proc_stat(st) if st else None
        if not parsed:
            continue
        cmd = provenance.cmdline(pid)
        if not cmd:
            kernel.add(pid)
            cmd = st[st.find("(") + 1:st.rfind(")")]
        procs[pid] = (parsed["cpu_user_s"] + parsed["cpu_system_s"], cmd)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return CpuSample(time.monotonic(), procs, _system_busy_s(), frozenset(kernel),
                     ru.ru_utime + ru.ru_stime)


#: Total foreign CPU, in cores, a row may run alongside. `bench/setup.py`'s
#: budget, imported rather than repeated: a process the gate is willing to
#: start a run alongside must not be one that condemns every row it touches.
#: (The comment said "imported" over a second copy of the literal for a
#: month; now it is.)
FOREIGN_BUDGET_CORES = setup.FOREIGN_BUDGET_CORES

#: Percent of one core above which a foreign process is worth naming in the
#: row's note. Naming, not condemning — the budget above is the verdict.
FOREIGN_NAME_PCT = 5.0


def foreign_between(before: CpuSample, after: CpuSample,
                    budget_cores: float | None = None,
                    threshold: float = FOREIGN_NAME_PCT) -> str:
    """The foreign processes that ran *during* the interval, as `ps` spells them.

    Percent of one core over the interval, from the difference of each
    process's `/proc/<pid>/stat` CPU time; a process that appears only in
    `after` started inside the interval and all of its time counts. Two `ps`
    samples around a row saw a lifetime average instead: a long-idle process
    with a busy history was flagged on every row and a compiler that started
    and finished inside one was never seen. The second case is what the
    system-wide line is for: non-idle time nobody visible accounts for is
    reported as `exited-processes(N%)`.
    """
    wall = after.t - before.t
    if wall <= 0:
        return ""
    if budget_cores is None:
        budget_cores = FOREIGN_BUDGET_CORES
    busy: list[tuple[float, str]] = []
    accounted = seen_foreign = 0.0
    for pid, (cpu1, cmd) in after.procs.items():
        prev = before.procs.get(pid)
        # A pid whose command line changed was reused; its earlier time was
        # someone else's.
        cpu0 = prev[0] if prev is not None and prev[1] == cmd else 0.0
        d = max(0.0, cpu1 - cpu0)
        if pid in after.kernel or _is_ours(cmd):
            accounted += d
            continue
        seen_foreign += d
        pct = 100 * d / wall
        if pct > threshold:
            busy.append((pct, f"{os.path.basename(cmd.split()[0])}({pct:.0f}%)"))
    busy.sort(key=lambda x: -x[0])
    out = [s for _, s in busy[:8]]
    exited = 0.0
    if before.busy_s is not None and after.busy_s is not None:
        # Children this harness has reaped since `before` (bfb, taskset,
        # docker exec) are ours: their pids are gone but their time is not.
        accounted += max(0.0, after.children_s - before.children_s)
        exited = max(0.0, after.busy_s - before.busy_s - accounted - seen_foreign)
        pct = 100 * exited / wall
        if pct > threshold:
            out.append(f"exited-processes({pct:.0f}%)")
    # The verdict is the *total*, in cores, against the calibrated budget —
    # not each process against a share of one core. Load that arrives as ten
    # small processes costs what one large one costs, and the per-process test
    # saw neither: it condemned a row for 0.2 of a core, which
    # `docs/decisions.md` measures as inside run-to-run noise, and passed 1.5
    # cores spread thinly, which it measures as costing real throughput.
    cores = (seen_foreign + exited) / wall
    if cores <= budget_cores:
        return ""
    return f"{cores:.2f} cores" + (f": {' '.join(out)}" if out else "")


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

@dataclass
class Result:
    """One row of §4's table, as measured.

    Deliberately flat, and this is the one place in the harness where that is
    the right call. `rows.json` is an artifact: `compare.py`, `report.py`,
    `results.py` and every run already on disk read these names. Nesting the
    I/O counters or the storage levels into sub-objects would be tidier code
    and a silent mismatch against every file measured before the change, on a
    project whose whole subject is results that cannot be compared across
    runs. The typing lives in `Status`, `LoadMode` and `procstat.Snapshot`
    instead, where it costs the artifact nothing.
    """

    id: str
    status: Status
    seconds: int
    load_start: int
    load_end: int
    foreign: str
    qps: float | None = None
    rps: float | None = None
    detail: str = ""
    #: §5.5's other axis. A row's I/O is a difference between two snapshots of
    #: the engine process; its storage size and peak RSS are levels, read after
    #: the row ran. `io_source` says whether the counts are syscalls (`proc`) or
    #: block-layer requests (`cgroup`), which are not interchangeable.
    #: `syscall_*` counts every descriptor, sockets included, and is not a
    #: disk figure; `disk_*` is the block layer. See `procstat`.
    syscall_reads: int | None = None
    syscall_writes: int | None = None
    disk_read_ops: int | None = None
    disk_write_ops: int | None = None
    disk_read_bytes: int | None = None
    disk_write_bytes: int | None = None
    io_source: str = ""
    rss_peak_bytes: int | None = None
    storage_bytes: int | None = None
    #: What the scheduler and the fault handler did during the row, from /proc.
    #: `cpu_*_s` and the fault counts are thread-group totals, so their
    #: differences are exact; the rest are summed over live threads and are
    #: `None` when a thread exited mid-row (`procstat.SCHED_FIELDS`).
    #: `runqueue_wait_s` answers what no qps number can: slow, or starved.
    cpu_user_s: float | None = None
    cpu_system_s: float | None = None
    minor_faults: float | None = None
    major_faults: float | None = None
    oncpu_s: float | None = None
    runqueue_wait_s: float | None = None
    timeslices: float | None = None
    ctx_switches_voluntary: float | None = None
    ctx_switches_involuntary: float | None = None
    migrations: float | None = None
    threads: float | None = None
    #: Fraction of the row's CPU time the surviving threads still account for.
    #: Below `procstat._COVERAGE_MIN` the thread-summed counters above are
    #: `None`: an index build spends its time in threads that exit before the
    #: row ends, and their counters leave with them.
    sched_coverage: float | None = None
    #: 1.0 when `procstat.SchedSampler` banked per-thread counters through the
    #: row, 0.0 when they were summed once at the end. `sched_coverage` means the
    #: same thing either way, so this is what says whether a low coverage came
    #: with its counters withheld or banked anyway — which a reader comparing
    #: `runqueue_wait_s` across rows needs to know.
    sched_sampled: float | None = None
    #: Time the engine was blocked in the block layer, from delay accounting.
    #: `runqueue_wait_s` is runnable-and-not-running; this is not runnable at
    #: all, waiting on a device. `None` where the host has `task_delayacct=0`,
    #: which is the default on most distributions and reports a permanent zero
    #: — see `procstat.delayacct_on`.
    blkio_delay_s: float | None = None
    #: Pressure-stall seconds for the engine's own cgroup: `some` is "at least
    #: one task stalled", `full` is "every runnable task stalled". The only
    #: stall figures here that work identically for a containerised engine and
    #: a native one.
    psi_cpu_some_s: float | None = None
    psi_cpu_full_s: float | None = None
    psi_io_some_s: float | None = None
    psi_io_full_s: float | None = None
    psi_mem_some_s: float | None = None
    psi_mem_full_s: float | None = None
    #: Whose stall time the six above are: `engine` when the cgroup holds this
    #: process and nothing else, `shared` when it holds the harness and the
    #: desktop too (and the figures are therefore withheld), empty when the
    #: kernel has no PSI. A blank column with a reason, rather than a blank.
    psi_scope: str = ""
    #: Resident bytes by what backs them, as levels. Two engines holding the
    #: same total differ in this split: one allocated it, the other mapped a
    #: file and let the kernel decide what stays.
    rss_anon_bytes: int | None = None
    rss_file_bytes: int | None = None
    rss_shmem_bytes: int | None = None
    #: What the *hardware* did during the row (`perfstat.py`). `None` unless the
    #: run was given `--perf`, which is most rows and deliberate: the sidecar
    #: touches its subject. `perf_events` records the spellings counted, since a
    #: DRAM-fill count from `LLC-load-misses` is not one from
    #: `ls_dmnd_fills_from_sys`. `perf_note` says why a blank is blank.
    perf_set: str = ""
    perf_events: str = ""
    perf_note: str = ""
    perf_enabled_pct: float | None = None
    perf_window_s: float | None = None
    #: The rate this host's `ref-cycles` counted at, calibrated when the row
    #: was measured. Without it the row has a frequency *ratio* and no
    #: frequency; see `perfstat.derive`.
    perf_ref_hz: float | None = None
    perf_cycles: float | None = None
    perf_ref_cycles: float | None = None
    perf_instructions: float | None = None
    perf_branch_misses: float | None = None
    perf_dram_fills: float | None = None
    perf_dtlb_walks: float | None = None
    perf_page_faults: float | None = None
    perf_minor_faults: float | None = None
    perf_major_faults: float | None = None
    perf_ctx_switches: float | None = None
    perf_migrations: float | None = None
    perf_task_clock_s: float | None = None
    perf_fills_local_ccx: float | None = None
    perf_fills_near_cache: float | None = None
    perf_fills_far_cache: float | None = None
    perf_fills_all: float | None = None
    #: Absolute, UTC. `rows.json` merges a re-measured row in place, so without
    #: this a file can hold rows from different days against different binaries
    #: with nothing to tell them apart.
    when: str = ""
    #: `fullrun.py`'s id for the invocation that measured this row. `when`
    #: separates days; this separates *runs*, which a timestamp cannot — a
    #: resumed arm minutes later is one run, a re-run half a day later is not.
    #: `run.json` is rewritten by an arm's last invocation, so without this a
    #: dead arm leaves the new run's gate verdict over the old run's rows.
    session: str = ""
    #: §7.4: `--rps` and `--parallel` measure different things, and a
    #: closed-loop p99 is not a latency result. Which one produced this row is
    #: a property of the row, not something a reader should infer from its id.
    load_mode: LoadMode = LoadMode.unset
    #: Client-side and server-side percentiles, in microseconds.
    latency: dict = field(default_factory=dict)
    #: The `ef` this row searched at, when it stated one. It is what the recall
    #: sweep joins on, and deriving it downstream means re-reading §4's table in
    #: two places.
    ef: int | None = None
    #: Exhaustive search (`--search-exact`), whose recall@10 is 1.0 by
    #: construction. See `exact_of`.
    exact: bool = False
    #: Wall clock around the bfb invocation, unrounded. `seconds` is its
    #: truncated integer and stays for the artifacts already on disk; a row
    #: whose headline result is a duration cannot be reported at one-second
    #: resolution.
    wall_s: float | None = None
    #: bfb's own phase timings and the polling-floor flag. See `phases_of`.
    upload_s: float | None = None
    index_wait_s: float | None = None
    #: How long the harness waited after this row for the engine's own CPU to
    #: go quiet, and what it was still using when it stopped waiting. Only
    #: upload rows settle; a search row reads `None`. bfb's `index_wait_s`
    #: above ends at *green*, and green is not idle — see `settle_engine`.
    engine_settle_s: float | None = None
    engine_settle_cores: float | None = None
    time_to_green_floored: bool | None = None
    #: The collection this row searched, and whether a recall sweep of it
    #: describes the row. Both are the join key for recall (with `ef`), written
    #: into the artifact so no reader has to re-derive §4's table to find them.
    collection: str | None = None
    recall_joinable: bool = True
    #: bfb's `Median qps` is the median of a per-request EWMA-style series, and
    #: on a row that finishes in a second or two it understates: strawmann's W4
    #: printed 27,201 while 50,000 queries in 1.61 s is 31,056. `qps` is the
    #: wall-clock figure `n_queries / duration_s` from bfb's own JSON, which is
    #: what "queries per second" means; the median is kept as `qps_bfb_median`.
    qps_bfb_median: float | None = None
    duration_s: float | None = None
    n_queries: int | None = None
    #: The second bfb process a concurrent row (W11) ran, and how the two
    #: overlapped in time. `overlap_s` is the interval during which both were
    #: in flight; a search that outlives the append is partly measuring a quiet
    #: collection and the note says by how much.
    background_pps: float | None = None
    background_s: float | None = None
    overlap_s: float | None = None
    #: Free-form caveats the tables print beside the number: `short (<2 s)`,
    #: `per-batch latency`, the W5 query strategy, W11's overlap.
    notes: str = ""
    #: Wall seconds of the discarded warm-up pass that preceded the measured
    #: one (§7.1), or None when there was none (`--no-warmup`, upload rows).
    warmup_s: float | None = None
    #: The build that served this row and the gate it ran under, per row:
    #: `env.txt` and `run.json` are rewritten by every invocation, so a later
    #: single-row run on a rebuilt binary re-vouched for the whole file.
    #: `compare` refuses a label whose rows disagree. `engine_build` is the
    #: commit or Qdrant version, `engine_binary` the sha256 or image digest.
    gate: Gate | str | None = None
    profile: str | None = None
    engine_build: str | None = None
    engine_binary: str | None = None
    isa_build: str | None = None
    optimize: str | None = None
    #: The `-n` this row was invoked with (after `--n-factor` /
    #: `--min-duration` scaling) and `stamp_hash` of the harness stamp it ran
    #: under. `rows.json` merges rows in place, so this is what says which rows
    #: the label's `run.json` stamp actually describes.
    n_requested: int | None = None
    harness_hash: str | None = None
    #: What an open-loop row offered, the fraction of saturation §4 asked for,
    #: and the measured qps that fraction was taken of. All three, because the
    #: rate is derived rather than declared: a row saying only "1,142/s" leaves
    #: a reader unable to tell whether the engine was being asked for half its
    #: capacity or a fiftieth of it, which is the difference between a latency
    #: measurement and a measurement of the load generator.
    rps_target: int | None = None
    rps_fraction: float | None = None
    saturation_qps: float | None = None
    #: `pinned` when `--rps-reference` supplied the saturation (so both engines
    #: were offered the same absolute rate and their percentiles may be read
    #: side by side), `own` when each engine used its own measured W4 (so they
    #: were not, and may not).
    rps_reference_source: str | None = None
    #: The quantization search parameters the row sent, when it sent any. A
    #: recall sweep speaks for a row only if it was taken under the same ones:
    #: W7 searches at oversampling 4 with rescore, and a sweep of `bench7` that
    #: sent neither measured a different search.
    quantization_oversampling: float | None = None
    quantization_rescore: bool | None = None
    #: `Workload.ratio_policy`, written into the row.
    ratio_policy: str = ""
    #: `Workload.needs_payload_index`, written into the row.
    needs_payload_index: bool = False
    #: `payload_index_suppressed(w)`, written into the row: the run itself
    #: denied the engine the index, so no read-back is needed to know the
    #: filtered queries scanned.
    payload_index_suppressed: bool = False
    #: W11: how much of the search phase (bfb's `duration_secs`) the append
    #: was in flight for. Below `W11_MIN_OVERLAP` the row is mostly a search of a quiet
    #: collection and is not the row §4 describes; `compare` refuses it.
    write_overlap_pct: float | None = None


#: The share of a concurrent row's search phase the write must cover for the
#: row to count as "search during write" (W11).
W11_MIN_OVERLAP = 0.9


#: The `Result` fields that together name the build a row was measured on.
#: `compare.stale_reasons` refuses a label whose rows carry more than one
#: value of this tuple; the gate is flagged separately (`compare.gate_of`).
BUILD_KEYS = ("engine_build", "engine_binary", "isa_build", "optimize", "profile")


def build_identity(run_meta: dict) -> dict:
    """What `run.json` says about the build and the gate, in row form.

    Stamped onto every `Result` of the invocation, so the identity travels
    with the measurement rather than with the file it is merged into.
    """
    sm = run_meta.get("strawmann") or {}
    qd = run_meta.get("qdrant") or {}
    if sm:
        build = (sm.get("commit") + ("-dirty" if sm.get("dirty") else "")
                 if sm.get("commit") else None)
        binary = sm.get("binary_sha256")
    else:
        # The image digest where there is one, the binary's sha256 where the
        # run used `--qdrant-binary`. Same role — which image actually ran —
        # and §8's sink refuses a row that carries neither, so a native Qdrant
        # would have been unpublishable for want of a fallback rather than for
        # want of an identity.
        build = qd.get("version")
        binary = qd.get("digest") or qd.get("binary_sha256")
    return {"gate": run_meta.get("gate"), "profile": run_meta.get("profile"),
            "engine_build": build, "engine_binary": binary,
            "isa_build": run_meta.get("isa_build"),
            "optimize": sm.get("optimize") if sm else None}


#: A row that finished faster than this measured bfb's startup and the ramp as
#: much as the engine. It is flagged, and `--min-duration` scales `-n` up.
MIN_ROW_S = 2.0


#: The percentiles §8.9's row shape requires, plus the two tails §7.4 argues
#: about. bfb's own JSON reports min/avg/p50/p95/max and no p99, but it also
#: writes every request time, so the tail is recoverable rather than lost.
PERCENTILES = (50, 95, 99, 99.9)


def percentiles_us(times: list[float]) -> dict[str, float]:
    """Percentiles in microseconds, from bfb's raw per-request times.

    Nearest-rank on the sorted sample, not interpolated: with 50,000 requests
    the difference is below the noise this measures, and a rank is a request
    that actually happened. The rank is `ceil(p/100 * n)`, the textbook
    nearest-rank; `round()` (banker's, to even) put p50 of five samples at the
    second value rather than the third, and p50 of 3,125 (W5's batches) at
    rank 1,562 rather than 1,563: the wrong element wherever `p/100 * n`
    ends in .5.
    """
    if not times:
        return {}
    xs = sorted(times)
    out = {}
    for p in PERCENTILES:
        k = max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))
        out[f"p{p:g}".replace(".", "")] = xs[k] * 1e6
    out["max"] = xs[-1] * 1e6
    out["mean"] = sum(xs) / len(xs) * 1e6
    return out


def exact_of(w: Workload) -> bool:
    """Whether the row searched exhaustively (`--search-exact`).

    An exact row's recall@10 is 1.0 by construction, not by measurement, and
    §8.5's T1 is what certifies the construction: it checks exact search
    against the fp64 oracle. Recording the flag is what lets the §8 sink say
    so, instead of refusing the row for a recall it can never join — the sweep
    joins on `ef`, and an exhaustive search has none. W9 was rejected on every
    run this project has made for exactly that reason.
    """
    return "--search-exact" in w.args


def ef_of(w: Workload) -> int | None:
    """The `ef` a row searches at, if it names one.

    Rows that do not name one search at the engine's default, which is not the
    same claim, so they report None rather than 128.
    """
    if "--search-hnsw-ef" not in w.args:
        return None
    i = w.args.index("--search-hnsw-ef")
    try:
        return int(w.args[i + 1])
    except (IndexError, ValueError):
        return None


def n_of(w: Workload) -> int | None:
    """The `-n` a row was invoked with, or None if it names none."""
    if "-n" not in w.args:
        return None
    try:
        return int(w.args[w.args.index("-n") + 1])
    except (IndexError, ValueError):
        return None


def quant_of(w: Workload) -> dict:
    """`--quantization-oversampling` / `--quantization-rescore`, as the row sent them."""
    out: dict = {"quantization_oversampling": None, "quantization_rescore": None}
    if "--quantization-oversampling" in w.args:
        try:
            out["quantization_oversampling"] = float(
                w.args[w.args.index("--quantization-oversampling") + 1])
        except (IndexError, ValueError):
            pass
    if "--quantization-rescore" in w.args:
        try:
            v = w.args[w.args.index("--quantization-rescore") + 1].lower()
            out["quantization_rescore"] = {"true": True, "false": False}[v]
        except (IndexError, KeyError):
            pass
    return out


def load_mode_of(w: Workload) -> LoadMode:
    """Closed loop, open loop, or neither.

    §7.4: "`--rps` (fixed rate) and `--parallel` (closed loop) measure
    different things. Use `--rps` for latency claims and closed-loop for
    saturation throughput. Never quote a closed-loop p99 as a latency result."
    A reader cannot apply that rule unless the row says which it is, and
    inferring it from the workload id means encoding §4's table a second time.

    `--scroll -p 8` is a closed loop too: W13 was labelled `upload` because
    the test asked only about `--search`, and a scroll row with a five-figure
    qps then carried the load mode of an ingest row.
    """
    # `rps_fraction` as well as the flag: §4's open-loop arms carry the
    # fraction in the table and gain `--rps` only once the runner has resolved
    # it against measured saturation, so a row read straight from `table()`
    # has the intent and not yet the flag. Keying on the flag alone filed
    # every one of them as a closed loop — which is the exact §7.4 confusion
    # this function exists to prevent.
    if "--rps" in w.args or w.rps_fraction is not None:
        return LoadMode.open_loop
    if "--search" in w.args or "--scroll" in w.args:
        return LoadMode.closed_loop
    return LoadMode.upload


def collection_of(w: Workload) -> str | None:
    if "--collection-name" in w.args:
        i = w.args.index("--collection-name")
        if i + 1 < len(w.args):
            return w.args[i + 1]
    return w.query_collection


def batch_size_of(w: Workload) -> int:
    if "--search-batch-size" in w.args:
        i = w.args.index("--search-batch-size")
        try:
            return int(w.args[i + 1])
        except (IndexError, ValueError):
            pass
    return 1


def static_notes(w: Workload) -> list[str]:
    """What a reader has to know about a row that the number does not say."""
    out = []
    if batch_size_of(w) > 1:
        # bfb times the batch request, so p50/p99 are per 16 queries, not per
        # query; the table would otherwise read them as a 16x slower engine.
        out.append(f"per-batch latency ({batch_size_of(w)} queries/request)")
    if w.query_collection and w.query_strategy != "from-start":
        out.append(f"queries: dataset, {w.query_strategy}")
    if w.background:
        out.append("concurrent append of synthetic vectors; recall not measured")
    return out


def phases_of(results: Path, wid: str) -> dict:
    """bfb's own upload and index-wait times, in seconds, unrounded.

    The harness's `seconds` is `int(wall_clock)` around the whole invocation:
    one-second resolution on a row whose headline *is* a duration, and it
    includes process startup, collection creation and teardown. bfb reports the
    two phases separately and as floats, so W1's ingest and W2's Time-to-Green
    can be the number they claim to be.

    It also exposes the bias. `wait_index` sleeps one second *before* its first
    poll and requires three consecutive Green observations, so its floor is
    three seconds no matter how fast the build is — a 50k build measured here
    reported 3.008 s against an upload of 0.064 s. `time_to_green_floored`
    marks a row sitting on that floor, where the number is an upper bound on
    the build and a measurement of the polling loop.
    """
    p = results / f"{wid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    res = d.get("results", {})
    out: dict = {}
    if isinstance(res.get("upload"), dict):
        v = res["upload"].get("duration_secs")
        if isinstance(v, (int, float)):
            out["upload_s"] = round(float(v), 4)
    if isinstance(res.get("index"), dict):
        v = res["index"].get("wait_secs")
        if isinstance(v, (int, float)):
            out["index_wait_s"] = round(float(v), 3)
            out["time_to_green_floored"] = float(v) < WAIT_INDEX_FLOOR_S + 0.5
    return out


def wall_qps_of(results: Path, wid: str, batch: int = 1) -> dict:
    """`n_queries / duration_secs` from bfb's JSON, the figure the tables print.

    bfb's `Median qps` (`stats.rs`) is the median of a rate series sampled per
    request from a moving window, and on a short row that median sits below
    the mean by construction, more so the shorter the row: it read 27,201 on a
    W4 whose 50,000 queries took 1.61 s (31,056). The wall figure has no such
    dependence, and it is what a reader means by queries per second. The
    request count comes from `full_timings` rather than `-n`, so an
    interrupted row reports what it did rather than what it was asked.
    """
    p = results / f"{wid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    res = d.get("results", {})
    phase = res.get("search") or res.get("scroll") or {}
    dur = phase.get("duration_secs")
    reqs = phase.get("full_timings")
    if not isinstance(dur, (int, float)) or dur <= 0 or not isinstance(reqs, list) or not reqs:
        return {}
    n = len(reqs) * max(1, batch)
    return {"qps_wall": round(n / float(dur), 3), "duration_s": round(float(dur), 3),
            "n_queries": n}


def background_of(results: Path, wid: str) -> dict:
    """The upload phase of a row's concurrent bfb, from its own JSON."""
    p = results / f"{wid}-write.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    up = (d.get("results") or {}).get("upload") or {}
    out = {}
    if isinstance(up.get("points_per_sec"), (int, float)):
        out["background_pps"] = round(float(up["points_per_sec"]), 1)
    if isinstance(up.get("duration_secs"), (int, float)):
        out["background_s"] = round(float(up["duration_secs"]), 3)
    return out


def latency_of(results: Path, wid: str) -> dict:
    """Client-side and server-side latency for one row.

    Both, because their difference is the transport and the queue: §7.4's rule
    that a closed-loop p99 is not a latency result is exactly about the gap
    between them. bfb calls them `full_timings` (round trip) and
    `server_timings` (what the server reported).

    Read from `search` *or* `scroll`, because bfb records the same two series
    under whichever phase ran. Reading only `search` cost W13 its percentiles
    entirely: it published a five-figure qps with no latency next to it and no
    stated reason, which reads as a measurement failure rather than as a lookup
    in the wrong key.
    """
    p = results / f"{wid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    res = d.get("results", {})
    search = res.get("search") or res.get("scroll") or {}
    out = {}
    for key, prefix in (("full_timings", "client"), ("server_timings", "server")):
        v = search.get(key)
        if isinstance(v, list) and v:
            for name, val in percentiles_us(v).items():
                out[f"{prefix}_{name}_us"] = round(val, 3)
    if out:
        out["latency_samples"] = len(search.get("full_timings") or [])
    return out


#: bfb prints both. `rps = per_sec() / search_batch_size`, so on a batched row
#: `rps` counts batch *requests* while `qps` counts the queries inside them.
#: Reading `rps` on W5 made a 2.2x speedup look like a 7x regression, and two
#: independent runs agreed on the wrong number, repetition does not correct a
#: units error.
_QPS = re.compile(r"Median qps: ([0-9.]+)")
_RPS = re.compile(r"Median rps: ([0-9.]+)")

#: §12 requires an unsupported construct answer UNIMPLEMENTED naming itself,
#: whether it is a permanent §1 non-goal (sharding, sparse vectors) or an
#: unbuilt phase of §2's surface (payload, filtering — M7, optional per §10).
#: An engine doing that is behaving as specified; reporting it as FAILED makes
#: a boundary indistinguishable from a bug.
_UNIMPL = "Operation is not implemented or not supported"


def command_for(w: Workload, uri: str, results: Path, common: list[str]) -> list[str]:
    """The bfb invocation for one row.

    A row with `query_collection` goes through `bfb search --file`, whose
    queries come from the dataset; everything else keeps the flag-driven path.
    The generated config is written into the result directory rather than a
    temporary file, so what a run measured is recoverable from what it left
    behind.

    The global flags still apply on the subcommand path: `--search-limit`,
    `--search-hnsw-ef`, `--quantization-*`, `-n`, `-p`, `-t`, `-c`, `--rps` and
    `--json` are all read from `Args` by the config processor.
    """
    base = [str(BFB), *common, "--uri", uri, "--json", str(results / f"{w.id}.json")]
    if w.query_collection is None:
        return base + w.args
    cfg = results / f"{w.id}.search.yaml"
    cfg.write_text(search_config(w.query_collection, w.query_strategy,
                                 w.keyword_filter))
    # `--collection-name`, `--search` and `--skip-setup` are the flag path's
    # spelling; the subcommand takes the collection from the config and does
    # not create anything.
    passthrough = []
    skip_next = False
    for i, a in enumerate(w.args):
        if skip_next:
            skip_next = False
            continue
        if a == "--collection-name":
            skip_next = True
            continue
        if a in ("--search", "--skip-setup"):
            continue
        passthrough.append(a)
    return base + ["search", "--file", str(cfg), *passthrough]


#: §7.1: a warm-up pass before every measured search row, its output
#: discarded. Connections, page cache and the engine's own caches are cold on
#: the first requests, and `qps = n / duration` charged that ramp to the row.
WARMUP_N = int(os.environ.get("WARMUP_N", 2_000))


def warmup_command(cmd: list[str], results: Path, wid: str) -> list[str] | None:
    """The same bfb invocation at `WARMUP_N` queries, writing beside the row.

    None where there is nothing to warm up: a row without `-n` (an upload) is
    not searched twice, since creating its collection twice is not a warm-up.
    The JSON goes to `<id>.warmup.json`, never to the row's own path: a
    measured pass that then failed to write would otherwise leave the warm-up's
    file to be read as the measurement.
    """
    if "-n" not in cmd:
        return None
    out = list(cmd)
    i = out.index("-n")
    try:
        out[i + 1] = str(min(int(out[i + 1]), WARMUP_N))
    except (IndexError, ValueError):
        return None
    if "--json" in out:
        out[out.index("--json") + 1] = str(results / f"{wid}.warmup.json")
    return out


def scale_n(args: list[str], factor: int) -> list[str]:
    """`-n` multiplied, for `--min-duration`: a row that finishes in under
    `MIN_ROW_S` is measuring bfb's ramp, and the fix is more queries."""
    if factor <= 1 or "-n" not in args:
        return list(args)
    out = list(args)
    i = out.index("-n")
    try:
        out[i + 1] = str(int(out[i + 1]) * factor)
    except (IndexError, ValueError):
        pass
    return out


def min_duration_action(w: Workload, r: Result, tries: int) -> str | None:
    """What `--min-duration` does with a row that came in under `MIN_ROW_S`.

    `"rerun"`, a reason it is *not* re-run, or None when the row is not short
    (or has already been re-run twice: a row still short at 4x is reported
    short). A row with a concurrent writer is never re-run: a second pass of
    W11 appends the same id range again, which overwrites rather than appends,
    and searches a collection the first pass already changed.
    """
    if (r.status != Status.ok or r.duration_s is None or r.duration_s >= MIN_ROW_S
            or w.upload_only):
        return None
    if w.background:
        return (f"short row ({r.duration_s:.2f} s < {MIN_ROW_S:.0f} s) but not re-run: "
                f"{w.id} mutates its collection, so a second pass would re-append the "
                f"same ids and search a different corpus")
    if tries >= 2:
        return None
    return "rerun"


#: Retries before a contaminated row is published contaminated. The load that
#: reaches a row is a burst rather than a standing process (the gate refuses
#: those up front), and a burst can outlast one row — one retry was not enough
#: on. The asymmetry is stark: a retry costs that row's wall clock,
#: and not retrying costs the run, since one contaminated row refuses the whole
#: document.
CONTAMINATED_RETRIES = int(os.environ.get("CONTAMINATED_RETRIES", 3))


#: How long to wait for the box to go quiet before re-measuring a contaminated
#: row, and how often to look. Short: the load these rows catch is a burst, and
#: a re-measurement that waits five minutes for a two-second interruption has
#: cost more than the row it is saving.
RETRY_SETTLE_TIMEOUT_S = 180.0
RETRY_SETTLE_SAMPLE_S = 2.0


#: How many times a §7.1 gate that fails *only* on quiescence is taken again.
#: The static checks — governor, boost, SMT, THP, NUMA — describe the machine
#: and do not change while a run waits; quiescence describes the moment, and a
#: moment is exactly the thing worth sampling twice.
GATE_RETRIES = int(os.environ.get("GATE_RETRIES", 5))


def env_hash_of_gate(out: str) -> str | None:
    """§7.1's environment hash, read out of the gate's own output.

    Read rather than re-derived, for the reason `profile` is: `bench/setup.py`
    decides what goes into the hash — `smt=` among them — and a second
    derivation here would be a second definition that could drift from it.

    The pattern is the one `check` prints (`setup.py`, "environment hash: ..."),
    which is *not* the `env_hash=` line in the describe-mode body. The first cut
    of this matched the latter, so it never fired and every run.json carried a
    null hash while looking like it carried a real one -- a stamp that silently
    stamps nothing is worse than no stamp, because the refusal that reads it
    (`compare.env_matches`) then permits everything.
    """
    m = re.search(r"environment hash: ([0-9a-f]+)", out)
    return m.group(1) if m else None


def run_gate_with_retry() -> subprocess.CompletedProcess:
    """`bench/setup.py check`, taken again while its only failure is a transient.

    `fullrun.settle` waits for the box to go quiet, then the gate samples /proc
    for a second; a burst landing between the two condemns the whole arm: a
    23%-of-a-core process arrived in that window and all 25 rows of
    pass 1 were stamped `gate=FAIL`.

    A failing governor will fail again in ten seconds; a process that happened to
    be running will not. So a gate failing on anything *other* than quiescence
    returns immediately, and one failing only on quiescence is waited out. The
    last attempt's output lands in `env.txt` either way, and after `GATE_RETRIES`
    the failing verdict stands and stamps every row.
    """
    for attempt in range(GATE_RETRIES + 1):
        gate = subprocess.run([sys.executable, str(ROOT / "bench/setup.py"), "check"],
                              capture_output=True, text=True)
        if gate.returncode == 0:
            return gate
        out = gate.stdout + gate.stderr
        fails = [ln for ln in out.splitlines() if "FAIL" in ln]
        transient = fails and all("NOT quiescent" in ln for ln in fails)
        if not transient or attempt == GATE_RETRIES:
            return gate
        who = next((ln.split("NOT quiescent:", 1)[1].strip()
                    for ln in fails if "NOT quiescent:" in ln), "foreign load")
        print(f"  §7.1 gate failed on {who} alone; waiting and taking it again "
              f"({attempt + 1} of {GATE_RETRIES})", flush=True)
        # The gate has two quiescence verdicts and they need two waits: a
        # named process is watched by the sampler that condemned it, but
        # "load X over N cores" with nobody named is the one-minute average
        # still carrying the previous phase, which no process sample can see
        # go away. Waiting on the sampler there returned in one tick, so five
        # retries were spent in fifteen seconds against a sixty-second decay.
        if who.startswith("load "):
            settle_load_for_retry()
        else:
            settle_for_retry()
    return gate  # unreachable; the loop returns


def load_per_core_pct() -> float:
    """The one-minute load average as a percentage of one core, the gate's reading."""
    try:
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0
    return load1 / (os.cpu_count() or 1) * 100


def settle_load_for_retry() -> None:
    """Wait for the load average to fall under the gate's own threshold.

    The same bound as `settle_for_retry`, polled at the same interval, and
    silent about giving up for the same reason.
    """
    deadline = time.monotonic() + RETRY_SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        if load_per_core_pct() <= setup.LOAD_PER_CORE_MAX_PCT:
            return
        print("      waiting for the load average to decay", flush=True)
        time.sleep(RETRY_SETTLE_SAMPLE_S)


def settle_for_retry() -> None:
    """Wait for the foreign process to go away before measuring the row again.

    Uses `foreign_between`, the same sampler that condemned the row, so waiting
    and judging are one instrument — the mistake `fullrun.settle` made by
    watching the load average while the gate watched processes.

    Bounded, and silent about giving up: if the box will not go quiet the row
    is measured anyway and comes back contaminated again, which
    `contaminated_action` then reports as contaminated on every attempt. That
    is the honest outcome and it needs no second message here.
    """
    deadline = time.monotonic() + RETRY_SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        before = cpu_sample()
        time.sleep(RETRY_SETTLE_SAMPLE_S)
        if not foreign_between(before, cpu_sample()):
            return
        print("      waiting for the box to go quiet", flush=True)


def contaminated_action(w: Workload, r: Result, tries: int) -> str | None:
    """What to do with a row that had another process on the box while it ran.

    `"rerun"`, a reason it is not, or None when the row is clean.

    A contaminated row is not a slightly worse measurement: the calibration in
    `docs/validation.md` puts 18 foreign cores at +41% and 3 at +11%, and the
    report refuses the row a ratio. One such row also refuses the whole
    document — `render` will not splice a table into README.md when either arm
    has one — so a run of 27 rows can be lost to a burst of load during one of
    them. Measuring it again costs that row's wall clock and usually recovers
    it, which is a much better trade than the run.

    Never a row with a concurrent writer, for the same reason `--min-duration`
    will not re-run one: a second pass of W11 re-appends the same id range,
    which overwrites rather than appends, and searches a collection the first
    pass already changed. That guard is the reason this is a sibling of
    `min_duration_action` rather than a flag on it.
    """
    if r.status != Status.ok or not r.foreign or w.upload_only:
        return None
    if w.background:
        return (f"contaminated ({r.foreign}) but not re-run: {w.id} mutates its "
                f"collection, so a second pass would re-append the same ids and "
                f"search a different corpus")
    if tries >= CONTAMINATED_RETRIES:
        return (f"contaminated ({r.foreign}) on every one of "
                f"{CONTAMINATED_RETRIES + 1} attempts")
    return "rerun"


#: The row whose measured throughput the open-loop arms are a fraction of.
SATURATION_ROW = "W4"

#: The highest rate bfb, at the pinned commit on four client cores, will
#: actually send. Measured rather than extrapolated: 29,734 delivered of 30,000
#: offered, against a server saturating at 32,900; above that the generator's
#: ceiling and the server's could not be told apart here.
#:
#: Was 4,000 until findings 33, which was the reaping bug measuring itself
#: (delivery decayed with load: 99% at 2,000/s, 77% at 20,000). §4's fractions
#: are of what the *engine* can serve, so capping here keeps the offered rate a
#: rate that was offered.
GENERATOR_CEILING_QPS = 30000.0


def with_placement(w: Workload, placement: str | None) -> Workload:
    """`w` with `--memory-vectors <placement>` where the row creates a collection.

    The two engines' defaults differ — strawmANN `pinned`, Qdrant `cached` — so an
    unconfigured run compares two residencies. Qdrant does not report a placement
    unless asked, so the refusal read "not captured for qdrant", which looks like
    a harness oversight rather than the real state. Asking it for `cached` changes
    nothing it does and makes it say so.

    Provenance, not configuration, and so deliberately absent from the harness
    stamp. Only the collection-creating rows: `--memory-vectors` is creation-time
    and a search row runs with `--skip-setup`.
    """
    if not placement or "--skip-setup" in w.args:
        return w
    return dataclasses.replace(w, args=[*w.args, "--memory-vectors", placement])


def saturation_qps(done: list, label_dir: Path) -> float | None:
    """The saturating row's measured qps, from this session or the last one.

    This session first: a `--only W4,W4-sat50` run measures W4 and the arm
    against *that* number, which is the point of "measured saturation". A
    previous `rows.json` second, so `--only W4-sat50` on its own resolves
    rather than refusing — but only from a row that actually succeeded, since
    a fraction of a failed row's qps is a fraction of nothing.

    None when neither has it. The caller refuses the row; it does not invent a
    rate, because a rate nobody measured is exactly the constant this change
    removed.
    """
    for r in done:
        if getattr(r, "id", None) == SATURATION_ROW and r.status == Status.ok and r.qps:
            return float(r.qps)
    path = label_dir / "rows.json"
    if path.exists():
        try:
            for r in json.loads(path.read_text()):
                if r.get("id") == SATURATION_ROW and r.get("status") == Status.ok and r.get("qps"):
                    return float(r["qps"])
        except (OSError, json.JSONDecodeError):
            pass
    return None


def resolve_rps(w: Workload, sat: float | None) -> tuple[Workload, int | None]:
    """`w` with its `--rps` filled in from `sat`, and the rate chosen.

    A row with no fraction passes through untouched. A row with one and no
    saturation to scale comes back with `None` for the rate, and the caller
    files it as a failure naming the missing row rather than running it at
    whatever bfb defaults to.
    """
    if w.rps_fraction is None:
        return w, None
    if not sat:
        return w, None
    # The *reference* is capped, not the resulting rate. Capping the rate
    # collapsed all three arms onto the ceiling — 50%, 70% and 90% of 22,995
    # all became 4,000 — which is one arm wearing three names. Capping the
    # reference keeps three distinct rates inside the range the generator has
    # been measured to deliver, which is what the fractions are for: a load
    # ladder, not a single point.
    rate = max(1, round(min(sat, GENERATOR_CEILING_QPS) * w.rps_fraction))
    return dataclasses.replace(w, args=[*w.args, "--rps", str(rate)]), rate


#: How quiet the engine has to get after an upload before the next row starts,
#: as a fraction of one core, and for how long.
ENGINE_IDLE_CORES = 0.05
ENGINE_IDLE_WINDOW_S = 1.0
ENGINE_IDLE_STABLE = 3
ENGINE_IDLE_TIMEOUT_S = 180.0


def settle_engine(pid: int | None, what: str) -> dict[str, float | None]:
    """Wait for the engine's own background work to finish, after an upload.

    bfb's `--wait-index` returns at *green*, and green is not idle: Qdrant's
    optimizer keeps running past it, so the next row measures search plus the
    residue. Measured, W3 straight after W2, three passes: 1,180 /
    2,013 / 1,402 qps at 5,024 / 528 / 3,252 us of CPU per query, inverse across
    every column at once, while strawmANN held 520/524/518. W2's own
    `index_wait_s` says which pass got lucky — 133 s against 96 and 88.

    That is findings 38's 28% spread on W3 and findings 41's runqueue wait, and
    neither is a property of Qdrant's search.

    Engine-agnostic on purpose: it watches the process's own CPU rather than
    asking Qdrant about its optimizer, so strawmANN returns on the first window.
    A timeout, not a loop forever, and what it waited is recorded on the row.
    """
    if pid is None:
        return {"engine_settle_s": None, "engine_settle_cores": None}
    t0 = time.monotonic()
    stable = 0
    last = procstat.cpu_seconds(pid)
    cores = None
    while time.monotonic() - t0 < ENGINE_IDLE_TIMEOUT_S:
        time.sleep(ENGINE_IDLE_WINDOW_S)
        now = procstat.cpu_seconds(pid)
        if last is None or now is None:
            # Engine gone; the row after this will say so. The last rate we
            # saw is not this engine's final rate and publishing it describes
            # a process that is not there, so the row carries no rate at all.
            cores = None
            break
        cores = (now - last) / ENGINE_IDLE_WINDOW_S
        last = now
        if cores <= ENGINE_IDLE_CORES:
            stable += 1
            if stable >= ENGINE_IDLE_STABLE:
                break
        else:
            stable = 0
    waited = round(time.monotonic() - t0, 2)
    # `cores` is still None when the very first window ended with the engine
    # gone, and a slow window can push `waited` past the threshold on that same
    # path — a loaded box where one `sleep` overruns is enough. Formatting None
    # with `{:.2f}` raised out of here instead of returning the row's nulls,
    # on every upload row, unguarded from `run_one`.
    if cores is None:
        if waited >= ENGINE_IDLE_WINDOW_S:
            print(f"    engine vanished while settling after {what} "
                  f"({waited:.1f}s); the row after this will say so", flush=True)
    elif cores > ENGINE_IDLE_CORES:
        # Only worth saying when it gave up *still busy*. Reporting the last
        # window on a normal exit prints a quiet number under a "still busy"
        # heading, which reads as the opposite of what happened.
        print(f"    engine still busy after {what}: waited {waited:.1f}s and it was "
              f"using {cores:.2f} cores when the wait timed out", flush=True)
    return {"engine_settle_s": waited,
            "engine_settle_cores": None if cores is None else round(cores, 3)}


def run_one(w: Workload, uri: str, results: Path, common: list[str],
            storage: str | None = None, n_factor: int = 1,
            stamp: dict | None = None, build: dict | None = None,
            warmup: bool = True, perf: str | None = None,
            sched_sampler: bool = False) -> Result:
    stdout_path = results / f"{w.id}.stdout"
    n_table = n_of(w)
    if n_factor > 1 and not w.upload_only:
        w = dataclasses.replace(w, args=scale_n(w.args, n_factor))
    cmd = command_for(w, uri, results, common)

    # The warm-up runs before anything is sampled or bracketed, so nothing it
    # does lands on the row: not its CPU, not its I/O, not its foreign load.
    warmup_s = None
    wcmd = warmup_command(cmd, results, w.id) if warmup and not w.upload_only else None
    if wcmd is not None:
        tw = time.monotonic()
        subprocess.run(wcmd, capture_output=True, text=True)
        warmup_s = round(time.monotonic() - tw, 3)

    l0, cpu0 = load_pct(), cpu_sample()
    when = provenance.now_iso()
    # Bracket the row: I/O is the difference, storage and RSS are the level
    # afterwards. Taken around `bfb` rather than around the whole run so an
    # upload row's writes are attributed to it and not smeared over the table.
    io0 = procstat.snapshot(storage)

    # `perf stat` on the same bracket as the /proc snapshots either side, so a
    # disagreement between the two instruments is about the counters and not
    # about what they covered. Off unless asked for.
    # Per-thread scheduler counters banked while the row runs, so a thread that
    # exits mid-row is still counted. Also off by default — it costs `open`s per
    # thread per tick. Without it the end-of-row sum is survivors-only and
    # `sched_coverage` says how much was missed (0.106 on Qdrant's W3,
    # findings 41).
    sampler = None
    if sched_sampler and io0 is not None:
        sampler = procstat.SchedSampler(io0.pid).start()

    car = None
    if perf and io0 is not None:
        car = perfstat.Sidecar(io0.pid, perf)
        if not car.start():
            # Said out loud. A silent failure here produces a full table of
            # blank hardware columns and nothing that says the instrument
            # never attached.
            print(f"    !! perf sidecar did not attach: {car.note}")

    # A concurrent row starts its second bfb first, so the writes are in
    # flight when the first query goes out. Its output goes to its own files
    # (`<id>-write.json`, `<id>-write.stdout`) so neither process's JSON
    # overwrites the other's.
    bg = None
    bg_t0 = bg_t1 = None
    bg_done: dict = {}
    if w.background:
        bg_cmd = [str(BFB), *common, "--uri", uri,
                  "--json", str(results / f"{w.id}-write.json"), *w.background]
        bg_t0 = time.monotonic()
        bg = subprocess.Popen(bg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)

        # Reaped on its own thread so its finish time is when it finished, not
        # when the search did — `communicate()` after the search returned
        # timestamped every append at the search's end, so `overlap_s` was always
        # the whole row. An exception here is recorded, not lost: `t1` is always
        # set, so the row is still built and says what happened to its writer.
        def _reap() -> None:
            try:
                bg_done["text"] = bg.communicate()[0]
            except Exception as e:
                bg_done["error"] = f"{type(e).__name__}: {e}"
            finally:
                bg_done["t1"] = time.monotonic()

        reaper = threading.Thread(target=_reap, daemon=True)
        reaper.start()

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.monotonic()
    wall = t1 - t0
    secs = int(wall)

    bg_text, bg_rc, bg_err = "", 0, ""
    if bg is not None:
        reaper.join()
        bg_text, bg_t1, bg_err = bg_done.get("text") or "", bg_done.get("t1"), bg_done.get("error", "")
        if bg.poll() is None:  # the reaper died before it reaped
            bg.kill()
            bg.wait()
        bg_rc = 1 if bg_err else bg.returncode
        (results / f"{w.id}-write.stdout").write_text(bg_text or "")

    # Stopped before the closing snapshot, so the hardware window ends where
    # the /proc window does rather than also covering the thread walk.
    perf_row = car.stop() if car is not None else perfstat.blank(perf or "")
    banked = sampler.stop() if sampler is not None else None
    io = procstat.delta(io0, procstat.snapshot(storage), banked)
    stdout_path.write_text(proc.stdout + proc.stderr)
    # Every number bracketing the row is closed here, ambient ones included,
    # because `foreign_between` divides foreign CPU by `after.t - before.t`: a
    # settle inside the bracket lengthens the denominator and dilutes the §7.1
    # verdict on exactly the rows that needed it (1.50 cores over a 90 s upload
    # fails; 0.61 over a wall stretched by a 130 s settle passes). The settle
    # runs after this, not before.
    l1, cpu1 = load_pct(), cpu_sample()
    # What ran beside the row, from CPU time consumed *during* it.
    foreign = foreign_between(cpu0, cpu1)
    # Now that nothing is left to measure, wait for the engine's own
    # background work. This protects the *next* row and cannot touch this one.
    settled = (settle_engine(io0.pid if io0 is not None else None, w.id)
               if w.upload_only else
               {"engine_settle_s": None, "engine_settle_cores": None})

    text = proc.stdout + proc.stderr
    if proc.returncode == 0 and bg_rc == 0:
        status, detail = Status.ok, ""
    elif _UNIMPL in text:
        status = Status.not_applicable
        m = re.search(r"not supported ([^M]*)", text)
        detail = (m.group(1).strip() if m else "documented non-goal")
    elif proc.returncode != 0:
        status, detail = Status.failed, "\n".join(text.strip().splitlines()[-4:])
    else:
        status = Status.failed
        detail = "concurrent append failed: " + (
            bg_err or "\n".join((bg_text or "").strip().splitlines()[-4:]))

    qps_med = rps = None
    wallq: dict = {}
    notes = static_notes(w)
    if n_of(w) != n_table:
        notes.append(f"-n {n_of(w)} ({n_factor}x the table's {n_table})")
    if bg_err:
        notes.append(f"append reaper failed: {bg_err}")
    if status == Status.ok:
        mq, mr = _QPS.findall(text), _RPS.findall(text)
        qps_med = float(mq[-1]) if mq else None
        rps = float(mr[-1]) if mr else None
        wallq = wall_qps_of(results, w.id, batch_size_of(w))
        if wallq and wallq["duration_s"] < MIN_ROW_S:
            notes.append(f"short (<{MIN_ROW_S:.0f} s): {wallq['duration_s']:.2f} s of "
                         f"queries; the ramp is a visible share of the row")
    # The tables print the wall figure; bfb's median stays beside it. A row
    # whose JSON was not written (a failed row) keeps the median so nothing
    # that was measured is lost.
    qps = wallq.get("qps_wall", qps_med) if status == Status.ok else None
    if status == Status.ok and qps is None and not w.upload_only:
        # Exit 0 with nothing to read is not a measurement.
        status = Status.no_output
        detail = (f"bfb exited 0 but wrote no {w.id}.json and printed no `Median qps`; "
                  f"see {stdout_path.name}")

    bgd = background_of(results, w.id) if w.background else {}
    overlap = None
    overlap_pct = None
    if bg is not None and bg_t0 is not None and bg_t1 is not None:
        overlap = round(max(0.0, min(t1, bg_t1) - max(t0, bg_t0)), 3)
        bg_wall = bg_t1 - bg_t0
        # Against the reader's search phase (bfb's own `duration_secs`), not
        # its process wall: the ramp and connect are not searching.
        search_s = wallq.get("duration_s") or wall
        overlap_pct = round(min(100.0, 100 * overlap / max(search_s, 1e-9)), 1)
        if status == Status.ok and overlap_pct < 100 * W11_MIN_OVERLAP:
            # The reader's qps spans its whole duration whether or not the
            # writer was still there; below the floor the row is not measuring
            # concurrent *writes*. It is still measuring concurrent engine
            # work, which is why the note below no longer calls it quiet.
            notes.append(f"write overlap {overlap_pct:.0f}%")
        if status == Status.ok:
            if bg_t1 < t1:
                # Not "a quiet collection": the writer exiting is not the
                # engine going quiet. W11's fifth of the corpus trips a
                # from-scratch rebuild that far outlives the append — on
                # dbpedia-openai-1m a 15 s append left the search running 282 s
                # at 175 qps against W10-ef128's 3,001 at the same `ef`. Reading
                # the remainder as quiet inverts the row's meaning.
                notes.append(f"append finished {t1 - bg_t1:.1f} s before the search did")
            elif t1 < bg_t1:
                notes.append(f"search finished {bg_t1 - t1:.1f} s before the append did "
                             f"({100 * overlap / max(bg_wall, 1e-9):.0f}% of the append "
                             f"was covered)")
            if bgd.get("background_pps") is not None:
                notes.append(f"append {bgd['background_pps']:,.0f} points/s")

    return Result(
        w.id, status, secs, l0, l1, foreign, qps, rps, detail,
        when=when, session=SESSION,
        load_mode=load_mode_of(w), latency=latency_of(results, w.id),
        ef=ef_of(w),
        exact=exact_of(w),
        wall_s=round(wall, 3),
        **phases_of(results, w.id),
        io_source=io.get("io_source", ""),
        rss_peak_bytes=io.get("rss_peak_bytes"),
        storage_bytes=io.get("storage_bytes"),
        **{k: io.get(k) for k in procstat.IO_FIELDS},
        **{k: io.get(k) for k in procstat.SCHED_FIELDS},
        **{k: io.get(k) for k in procstat.PSI_FIELDS},
        **{k: io.get(k) for k in procstat.RSS_FIELDS},
        psi_scope=io.get("psi_scope", ""),
        **perf_row,
        collection=collection_of(w), recall_joinable=w.recall_joinable,
        qps_bfb_median=qps_med,
        duration_s=wallq.get("duration_s"), n_queries=wallq.get("n_queries"),
        overlap_s=overlap, notes="; ".join(notes),
        n_requested=n_of(w),
        harness_hash=stamp_hash(stamp) if stamp is not None else None,
        ratio_policy=w.ratio_policy, write_overlap_pct=overlap_pct,
        needs_payload_index=w.needs_payload_index,
        payload_index_suppressed=payload_index_suppressed(w),
        warmup_s=warmup_s,
        **quant_of(w),
        **bgd,
        **settled,
        **(build or {}),
    )


def write_report(label: str) -> None:
    """Render the HTML report for this run, if the uv environment exists.

    Invoked through `uv run` rather than imported: this module is stdlib-only
    by design so a run works on a fresh clone, and importing pandas at the top
    would make an unrelated `workloads.py list` fail on a machine that has not
    synced. A missing environment is reported, not fatal, since the numbers are
    already written by the time this runs and losing them to a reporting
    failure would be the wrong order of priorities.
    """
    report = ROOT / "bench/harness/report.py"
    if not report.exists():
        return
    cmd = ["uv", "run", "--project", str(ROOT / "bench"), str(report), label]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print("\n" + r.stdout.strip())
    else:
        print("\nreport generation failed (the results themselves are intact):",
              file=sys.stderr)
        print("  " + "\n  ".join((r.stderr or r.stdout).strip().splitlines()[-4:]),
              file=sys.stderr)
        print(f"  retry with: uv run --project bench bench/harness/report.py {label}",
              file=sys.stderr)


def backfill(label: str) -> int:
    """Recompute what a finished run left recoverable, without re-measuring.

    Latency was always measured: bfb writes every request time into
    `<row>.json`, and the harness simply never extracted it. So a run taken
    before that extraction existed is not missing the data, only the summary,
    and re-running the workloads to obtain a number already on disk would be
    both wasteful and dishonest about which run produced it.

    What cannot be recovered is stated as unknown rather than invented: the
    kernel's I/O counters are gone with the process, and no timestamp was
    recorded, so `when` stays empty and the report says "not recorded".
    """
    results = Path(os.environ.get("RESULTS_DIR", ROOT / "bench/results" / label))
    path = results / "rows.json"
    if not path.exists():
        print(f"no rows.json in {results}", file=sys.stderr)
        return 1
    rows = json.loads(path.read_text())
    by_id = {w.id: w for w in table()}

    changed = 0
    for r in rows:
        w = by_id.get(r["id"])
        lat = latency_of(results, r["id"])
        if lat and not r.get("latency"):
            r["latency"] = lat
            changed += 1
        if w is not None:
            r.setdefault("load_mode", "") or r.update(load_mode=load_mode_of(w))
        # `ef` is NOT backfilled from today's table. A row that ran without a
        # stated `ef` searched at the engine's default, and stamping the value
        # the table carries *now* onto it would join a recall sweep at ef=128
        # to a measurement that was not taken there. Unknown stays None.
        r.setdefault("ef", None)
        # The collection is recoverable, and from the artifact rather than the
        # table: bfb writes `config.collection_name` into its own JSON.
        if r.get("collection") is None:
            bj = results / f"{r['id']}.json"
            try:
                r["collection"] = json.loads(bj.read_text()).get("config", {}).get("collection_name")
            except (OSError, json.JSONDecodeError):
                r["collection"] = None
        r.setdefault("recall_joinable", w.recall_joinable if w is not None else True)
        # The wall-clock qps is in the JSON already; the median that was
        # published stays under its own name.
        if r.get("qps") is not None and r.get("qps_bfb_median") is None:
            wallq = wall_qps_of(results, r["id"], batch_size_of(w) if w is not None else 1)
            if wallq:
                r["qps_bfb_median"] = r["qps"]
                r["qps"] = wallq["qps_wall"]
                r["duration_s"] = wallq["duration_s"]
                r["n_queries"] = wallq["n_queries"]
                changed += 1
        for k in ("when", "notes"):
            r.setdefault(k, "")
    path.write_text(json.dumps(rows, indent=1) + "\n")

    # A minimal run.json for a run that predates it. The dataset is knowable
    # from §4's table, which names it; the engine build is not knowable at all
    # after the fact, so it is written as unrecorded rather than guessed from
    # whatever binary happens to be in zig-out today. That distinction is the
    # whole point of recording provenance.
    meta_path = results / "run.json"
    if not meta_path.exists():
        meta = {
            "label": label,
            "backfilled": provenance.now_iso(),
            "host": provenance.host(),
            "dataset": provenance.dataset_identity(),
            "note": ("reconstructed from a finished run: the dataset is the one §4's "
                     "table names, the engine build was never recorded and cannot be "
                     "recovered, and the I/O counters died with the process"),
        }
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{label}: wrote a partial {meta_path.name} "
              f"(dataset and host; engine build unrecoverable)")
    # The memory bandwidth is a property of the host, and the host is still
    # here, so a run that predates the record can have it filled in. It is
    # measured *now* rather than when the rows were taken, and the record says
    # so; the report shows the source beside the number.
    meta = json.loads(meta_path.read_text())
    host_meta = meta.setdefault("host", provenance.host())
    if not host_meta.get("memory_bandwidth"):
        bw = provenance.memory_bandwidth(server_log=results / "server.log")
        if bw:
            if bw["source"] != "strawmann startup banner":
                bw["source"] = "strawmann --probe, backfilled after the run"
            host_meta["memory_bandwidth"] = bw
            meta_path.write_text(json.dumps(meta, indent=1) + "\n")
            print(f"{label}: memory bandwidth backfilled into {meta_path.name} "
                  f"({bw['aggregate_gbps']:.1f} GB/s aggregate, {bw['source']})")

    print(f"{label}: latency recovered for {changed} row(s) from the bfb JSON "
          f"already in {results}")
    return 0


#: The boolean flags `run` accepts. Anything else is an error, not a no-op.


def main(argv: list[str]) -> int:
    # Python block-buffers stdout when it is not a tty, so a run redirected to a
    # file shows nothing until it exits. A sweep is 50 minutes; watching it is
    # how a stall gets noticed before it wastes the whole run.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    # argparse, like every other driver here. Was ~95 lines of hand-rolled argv
    # scanning plus a separate flag set and usage string — three copies of one
    # list, which is how `--min-durations` came to run the table without
    # re-running short rows and exit 0. The accepted syntax is unchanged,
    # interspersed flags included.
    #: `--data-dir` is accepted before the subcommand *and* on `run`, which is
    #: what the hand-rolled parser did by scanning argv before it dispatched.
    #: A parent parser is how argparse spells that.
    root = argparse.ArgumentParser(add_help=False)
    #: SUPPRESS, because a subparser sharing a parent writes the parent's
    #: default over whatever the top level already parsed: `--data-dir X list`
    #: set the root and then `list` reset it to None.
    root.add_argument("--data-dir", metavar="PATH", default=argparse.SUPPRESS)

    ap = argparse.ArgumentParser(
        prog="workloads.py", description=__doc__, parents=[root],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", parents=[root], help="print §4's table")
    bf = sub.add_parser("backfill", parents=[root], help="recompute derived fields on an existing label")
    bf.add_argument("label")

    run = sub.add_parser("run", parents=[root], help="measure §4's rows against a running engine")
    run.add_argument("uri")
    run.add_argument("label")
    run.add_argument("rows", nargs="*", help="row ids; default is every row")
    #: The dataset root, for a direct invocation. Under `fullrun.py` it arrives
    #: as $STRAWMANN_DATA instead, which is why this sets the root rather than
    #: threading a value through `table()`.
    run.add_argument("--storage", metavar="PATH", default=os.environ.get("STORAGE_DIR"))
    #: Pins the saturation the open-loop arms are a fraction of, instead of each
    #: engine using its own measured W4. Taken per engine, §4's fractions put the
    #: two arms at two different offered loads, which is the comparison
    #: `docs/workloads.md` warns against. One reference — the *slower* engine's
    #: saturation — gives both properties: a fraction of a measured rate, and the
    #: same offered load on both sides.
    run.add_argument("--rps-reference", type=float, metavar="QPS")
    #: Asked of the collection-creating rows. Provenance rather than
    #: configuration: see `with_placement`.
    run.add_argument("--memory-vectors", choices=tuple(Placement))
    #: Which of the two Qdrant experiments this arm is. Both arms of a
    #: comparison must agree; `fullrun.py` passes it through `$SEGMENT_POLICY`
    #: so they cannot be given different ones by accident, and this flag is
    #: for driving `workloads.py` on its own.
    run.add_argument("--segment-policy", choices=tuple(SegmentPolicy),
                     default=None)
    #: `fullrun.py`'s id for one invocation of it, stamped onto `run.json` and
    #: every row this call measures. See `fullrun.SESSION`.
    run.add_argument("--session", metavar="ID", default=None)
    #: Multiplies every search row's `-n`; `--min-duration` re-runs a row that
    #: finished under MIN_ROW_S with a larger `-n`.
    run.add_argument("--n-factor", type=int, default=1, metavar="K")
    #: Where `--min-duration`'s per-row verdict is remembered, so an arm's
    #: passes measure one configuration. The threshold can be straddled: on
    #: W12 settled at 2x in one pass and 4x in another, and the fold
    #: published a median of two configurations. The first pass decides and
    #: writes the factor; later passes read it. Per arm, not per run — the slower
    #: engine's rows are already long enough.
    run.add_argument("--n-pin", metavar="PATH", default=None)
    run.add_argument("--min-duration", action="store_true")
    run.add_argument("--sink", action="store_true")
    run.add_argument("--report", action="store_true")
    run.add_argument("--no-warmup", action="store_true")
    #: Attach `perf stat` to the engine for each row. Opt-in, because it is an
    #: instrument that touches its subject: `perfstat.py --overhead` measures
    #: what it costs, and §7.1's discipline applies to the instrument as much
    #: as to the host. The set name is recorded on every row it measures.
    run.add_argument("--perf", nargs="?", const="default", default=None,
                     choices=tuple(perfstat.EVENT_SETS), metavar="SET",
                     help="hardware counters per row (default set: %(const)s)")
    #: On by default since 2026-08-28. Measured at roughly `threads x 45 us`
    #: per poll, which at the 250 ms interval is 0.2% of one core for
    #: strawmANN's nine threads and 1.4% for Qdrant's seventy — and charged to
    #: the client cpuset, not the engine's. Without it the thread-summed
    #: counters are simply missing on any engine that retires threads mid-row,
    #: which is the whole of findings 41's limitation.
    run.add_argument("--no-sched-sampler", dest="sched_sampler",
                     action="store_false", default=True,
                     help="do not bank per-thread scheduler counters during "
                          "each row. Without the sampler a thread that exits "
                          "mid-row takes its counters with it, "
                          "`sched_coverage` reports how much of the row that "
                          "missed, and every thread-summed counter is withheld "
                          "below 0.95")

    # argparse raises on a bad argument; `main` returns a code. Callers and
    # tests use the latter, and a driver that raises where it used to return
    # turns an argument typo into a traceback in the middle of a sweep.
    try:
        args = ap.parse_args(argv[1:])
    except SystemExit as e:
        return int(e.code or 0)
    if args.cmd is None:
        args = ap.parse_args(["list"])

    data_dir = getattr(args, "data_dir", None)
    if data_dir:
        paths.use_data_dir(data_dir)

    session = getattr(args, "session", None)
    if session:
        global SESSION
        SESSION = session
    if args.cmd == "run":
        if args.segment_policy:
            use_segment_policy(args.segment_policy)
        storage = args.storage
        rps_reference = args.rps_reference
        memory_vectors = args.memory_vectors
        n_factor = max(1, args.n_factor)
        flags = {f"--{name.replace('_', '-')}"
                 for name in ("min_duration", "sink", "report", "no_warmup")
                 if getattr(args, name)}
        perf_set = args.perf
        sched_sampler = args.sched_sampler

    if args.cmd == "list":
        for w in table():
            kind = "load" if w.upload_only else "search"
            print(f"  {w.id:<12} {kind:<7} {w.purpose}")
        return 0

    if args.cmd == "backfill":
        return backfill(args.label)

    uri, label = args.uri, args.label
    wanted = set(args.rows)
    unknown_ids = sorted(wanted - {w.id for w in table()})
    if unknown_ids:
        # `W3x` used to select nothing, run nothing and exit 0.
        print(f"unknown workload id(s): {' '.join(unknown_ids)}; "
              f"`workloads.py list` prints the table", file=sys.stderr)
        return 2
    results = Path(os.environ.get("RESULTS_DIR", ROOT / "bench/results" / label))
    if why := foreign_dataset(results, DATASET):
        print(f"refusing to write into {results}\n  {why}", file=sys.stderr)
        return 2
    results.mkdir(parents=True, exist_ok=True)

    if not BFB.exists():
        print(f"bfb not built at {BFB}", file=sys.stderr)
        print("  git clone -b dev https://github.com/qdrant/bfb && cd bfb", file=sys.stderr)
        print(f"  git checkout {BFB_COMMIT}", file=sys.stderr)
        print("  cargo build --release", file=sys.stderr)
        return 1
    if why := check_bfb_pin():
        print(why, file=sys.stderr)
        return 1

    # §7.1's gate. A failing host does not stop the run, it stamps the result
    # directory so no number from it can be quoted by accident.
    env_txt = results / "env.txt"
    gate = run_gate_with_retry()
    # Appended, not overwritten. `fullrun` runs this twice per arm, and the
    # second invocation used to erase the first's verdict — which is how a run
    # whose search rows were measured under a failed gate came to carry an
    # `env.txt` saying every check passed, and why the evidence of what was busy
    # was gone by the time anyone looked.
    with env_txt.open("a") as f:
        f.write(gate.stdout + gate.stderr)
    gate_ok = gate.returncode == 0
    # Which experiment this is, not how well it went: `isolated` attributes a
    # difference to the engine, `as-deployed` says whether a user would see it.
    # Read from the gate's own output rather than re-deriving it here, so there
    # is one definition of the word and `bench/setup.py` owns it.
    m = re.search(r"measurement profile: (\S+)", gate.stdout + gate.stderr)
    profile = m.group(1) if m else "unknown"
    # The §7.1 hash of the machine, read from the gate's output rather than
    # re-derived here: `bench/setup.py` owns what goes into it.
    #
    # Recorded because a stored spread is only a floor for the environment it
    # was measured in. A Qdrant W4 of 10,313 was read against a
    # 14,716 aggregate and taken for a 30% regression; the aggregate was
    # SMT-off and the host had since booted SMT-on, and nothing carried the
    # hash in a form a tool could check (findings 46).
    env_hash = env_hash_of_gate(gate.stdout + gate.stderr)
    print(f"§7.1 environment gate: {'pass' if gate_ok else 'FAIL'}  "
          f"[{profile}]  (see {env_txt})")
    if not gate_ok:
        print("  -> every number below is development-grade and unpublishable")

    # What can be measured about the engine besides its throughput, reported
    # before the run rather than discovered as nulls in the table afterwards.
    probe = procstat.snapshot(storage)
    if not probe:
        print("engine: no strawmann or qdrant process found; storage and I/O "
              "will be blank in this run")
    else:
        print(f"engine: {probe.comm} pid {probe.pid}, "
              f"I/O via {probe.io_source or 'UNAVAILABLE'}, "
              f"storage {probe.storage_path or 'UNKNOWN'}")
        if probe.engines_running > 1:
            print("  !! more than one engine process is running. §7.1 measures "
                  "one at a time; they share the page cache and the memory bus")
        if not probe.io_source:
            print("  -> /proc/<pid>/io is owner-only and this process has no "
                  "cgroup io.stat; I/O counts will be blank")
        if probe.storage_path is None:
            if probe.storage_bytes == 0:
                # `procstat.store_bytes_for`: no `--data-dir` on the engine's
                # command line means there is no store, and its size is 0 by
                # construction. Saying "unknown" here while the row recorded 0
                # made the log contradict the table.
                print("  -> no storage directory: the engine runs without "
                      "--data-dir, so storage is 0 by construction, not measured")
            else:
                print("  -> storage directory unknown, reported as such rather "
                      "than as zero. Pass --storage <path> or set STORAGE_DIR")
        # The socket demand, stated before the run rather than discovered as a
        # bare "transport error" on W4. bfb opens roughly `threads x
        # connections`; a server with fewer closes the excess before the HTTP/2
        # preface, and the client cannot tell that from an engine fault.
        want_sockets = 16 * W4_CONNS
        print(f"  -> W4 will open ~{want_sockets} sockets (-t 16 -c {W4_CONNS}). "
              f"strawmann must be started with io_threads x connections >= "
              f"{want_sockets}, e.g. --connections {want_sockets}")
        # Name the collection that actually binds, not one of the two. The
        # message used to explain the requirement with W11's arithmetic
        # regardless, so at a smoke scale it printed a correct number beside a
        # reason that did not produce it.
        need = required_capacity()
        why = (f"W11 appends {w11_n():,} on top of W2's {upload_n():,} in bench2"
               if upload_n() + w11_n() >= w12_n()
               else f"W12 loads {w12_n():,} into bench12")
        print(f"  -> {why}, so the largest single collection reaches {need:,}. "
              f"strawmann must be started with --capacity {need} or that row "
              f"fails with RESOURCE_EXHAUSTED")

    if perf_set:
        # Before the rows, not discovered as a table of blanks after them.
        pairs = perfstat.resolve(perfstat.EVENT_SETS[perf_set])
        got = {e.field for e, _ in pairs}
        print(f"perf: sidecar on, event set {perf_set!r}")
        print("  -> " + ", ".join(name for _, name in pairs))
        # Warmed here, before the first row, because it spins a core for 200 ms
        # and every row counts what ran during it against the quiet-host
        # budget. Called lazily from the sidecar it landed inside the first
        # row's own foreign-load window. See `perfstat.ref_cycles_hz`.
        hz = perfstat.ref_cycles_hz()
        print(f"  -> ref-cycles calibrated at {hz / 1e9:.4f} GHz; effective "
              f"frequency is that times each row's cycles/ref-cycles"
              if hz else
              "  !! could not calibrate the reference rate: rows will carry a "
              "frequency *ratio* and no frequency")
        absent = [e.field for e in perfstat.EVENT_SETS[perf_set] if e.field not in got]
        if absent:
            print("  !! not supported on this host, and reported as unmeasured "
                  "rather than zero: " + " ".join(absent))
        if not probe:
            print("  !! no engine process to attach to; every hardware column "
                  "will be blank")
        elif (blocked := perfstat.attach_blocked(probe.pid)):
            # Said before the rows, because it is not fixable from inside the
            # run and the whole arm's hardware columns depend on it. A
            # containerised Qdrant is this case on every host where the harness
            # is not root.
            print(f"  !! CANNOT ATTACH to {probe.comm} (pid {probe.pid}): {blocked}")
            print("     every hardware column on this arm will be blank, and the "
                  "report will say the engine could not be counted rather than "
                  "that it counted zero")

    # One machine-readable record of what this run was, beside the human one.
    # The HTML report is written for people who did not run it, and a number
    # whose engine, dataset and date are unstated is not checkable by them.
    run_meta = provenance.collect(label, uri,
                                  probe.pid if probe else None,
                                  probe.comm if probe else None,
                                  dataset=DATASET, results=results)
    run_meta.update(
        gate=Gate.passed if gate_ok else Gate.failed,
        profile=profile,
        env_hash=env_hash,
        bfb_pin=BFB_PIN, bfb_client=BFB_CLIENT, harness_client=HARNESS_CLIENT,
        queries=QUERIES, exact_queries=EXACT_QUERIES, upload_n=upload_n(),
        io_source=probe.io_source if probe else "",
        storage_path=probe.storage_path if probe else None,
        engines_running=probe.engines_running if probe else 0,
        # Outside `harness_stamp()` on purpose: this is per engine and the
        # stamp is hashed into every row's identity, so a value here would make
        # the two arms STALE against each other and refuse every ratio. See
        # `with_placement`.
        memory_vectors_requested=memory_vectors,
        # Which instrument was attached, at the run level, so a report can see
        # the mismatch between two arms without reading every row. Outside
        # `harness_stamp()` on purpose: see `STAMP_KEYS`.
        perf_set=perf_set or "",
        harness=harness_stamp(),
    )
    # Which ISA build served, from the server's own banner (or its `--probe`),
    # so an ISA-sweep arm does not ingest into the §8 sink as `native`.
    if "strawmann" in run_meta:
        # Asked of the image that is serving (`/proc/<pid>/exe`), not of the
        # path it was started from: `binary_sha256` is taken through /proc, and
        # probing the on-disk path answered for whatever had been built there
        # since. See `provenance.probe_path_of`.
        binary = provenance.probe_path_of(run_meta["strawmann"])
        server_log = results / "server.log"
        isa = provenance.isa_build(server_log=server_log, binary=binary)
        if isa:
            run_meta["isa_build"] = isa
        # Optimize mode and the tier the kernels actually reached. §9 licenses
        # numbers from ReleaseFast only, and until this was recorded a Debug
        # binary at the default path served a full run indistinguishably: same
        # commit, same `isa build: native`, nothing to contradict it.
        run_meta["strawmann"].update(
            provenance.build_flags(server_log=server_log, binary=binary))
    if session:
        run_meta["session"] = session
    run_meta["started"] = arm_started(results, run_meta.get("started"), session)
    (results / "run.json").write_text(json.dumps(run_meta, indent=1) + "\n")

    stamp = run_meta["harness"]
    build = build_identity(run_meta)
    warmup = "--no-warmup" not in flags
    (results / "pin.txt").write_text(
        f"harness        {stamp.get('commit') or 'unknown'}"
        f"{' (dirty)' if stamp.get('dirty') else ''}\n"
        f"dataset        {DATASET} (d={DIM})\n"
        f"metric         {METRIC}\n"
        f"segments       {SEGMENT_POLICY} ({policy_means()})\n"
        f"collection     segments={segments() or 'engine default'} "
        f"indexing_threshold_kb={INDEXING_THRESHOLD_KB} "
        f"full_scan_threshold_kb={FULL_SCAN_THRESHOLD_KB} on_disk_payload={ON_DISK_PAYLOAD}\n"
        f"bfb            {BFB_PIN}\n"
        f"bfb client     {BFB_CLIENT}\n"
        f"harness client {HARNESS_CLIENT}\n"
        f"engine         {label} @ {uri}\n"
        f"gate           {'pass' if gate_ok else 'FAIL'}\n"
        f"profile        {profile}\n"
        f"env_hash       {env_hash or 'unknown'}\n"
        f"queries        {QUERIES} ({EXACT_QUERIES} for exact)\n"
        f"upload         {upload_n()} of {UPLOAD_N} requested "
        f"({corpus_rows() or 'corpus not read'} in the corpus)\n"
        f"io source      {(probe.io_source if probe else '') or 'unavailable'}"
        f" (proc = syscalls + block bytes, cgroup = block ops + bytes)\n"
        f"storage path   {(probe.storage_path if probe else None) or 'unknown'}\n")

    common = ["--retry", "0", "--timeout", str(BFB_TIMEOUT_S), "--p9", "3"]

    rows = [w for w in table() if not wanted or w.id in wanted]
    # `--min-duration`'s verdict from this arm's first pass, if there was one.
    n_pin_path = Path(n_pin) if (n_pin := getattr(args, "n_pin", None)) else None
    n_pins: dict[str, int] = {}
    if n_pin_path is not None and n_pin_path.exists():
        try:
            n_pins = {k: int(v) for k, v in json.loads(n_pin_path.read_text()).items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            n_pins = {}
    # Only `--min-duration` produces the verdict this file records, and only
    # the default `--n-factor` may be overridden by it. Without the first guard
    # a hand run that omitted `--min-duration` would write 1 for every row, and
    # the pass that read it would skip the escalation entirely and publish
    # short rows as the ramp measurements the flag exists to prevent. Without
    # the second an explicit `--n-factor 4` would be silently discarded by a
    # stale pin, and a flag quietly ignored is worse than one refused.
    pinning = "--min-duration" in flags and n_factor == 1
    if n_pins and not pinning:
        why = ("--n-factor was given explicitly" if n_factor != 1
               else "this run has no --min-duration verdict to pin")
        print(f"  ignoring the -n pin in {n_pin_path}: {why}")
    elif n_pins:
        print(f"  -n factors pinned from this arm's first pass: "
              f"{', '.join(f'{k} x{v}' for k, v in sorted(n_pins.items()) if v > 1) or 'all 1x'}")
    out: list[Result] = []
    for w in rows:
        # §4's open-loop arms are a fraction of *measured* saturation, so the
        # rate is only known once W4 has run. Resolved here rather than in
        # `table()`, which cannot see a measurement.
        w = with_placement(w, memory_vectors)
        rate, sat = None, None
        if w.rps_fraction is not None:
            # The pinned reference wins where one was given: it is what makes
            # the two engines' arms the *same* offered load, which is the
            # property the constants had and the per-engine reading loses.
            sat = rps_reference or saturation_qps(out, results)
            # A pinned reference above what this engine just measured turns the
            # high arms into queues: the p50 that comes back is the queue's. On
            # W4-sat90 offered 3,136/s against a 2,564/s saturation
            # and returned 7.5 s. Said when the arm resolves and the run could
            # still be stopped, not only in the row's note afterwards.
            if rps_reference is not None:
                own = saturation_qps(out, results)
                offered_rate = w.rps_fraction * rps_reference
                if own and rps_reference > own and offered_rate > own:
                    print(f"    !! offering {offered_rate:,.0f}/s ({w.rps_fraction:.0%} of the "
                          f"pinned {rps_reference:,.0f}) to an engine that measured "
                          f"{own:,.0f}/s at {SATURATION_ROW}: this arm will queue, and "
                          f"its percentiles will be the queue's rather than the "
                          f"engine's", flush=True)
            w, rate = resolve_rps(w, sat)
            if rate is None:
                print(f"\n=== {w.id} ===\n    skipped: no measured "
                      f"{SATURATION_ROW} to take {w.rps_fraction:.0%} of")
                out.append(Result(
                    id=w.id, status=Status.failed, seconds=0,
                    load_start=0, load_end=0, foreign="",
                    notes=(f"no measured {SATURATION_ROW} in this run or in "
                           f"rows.json and no --rps-reference, so the offered rate "
                           f"is unknown; §4 sets it at {w.rps_fraction:.0%} of "
                           f"saturation and this harness will not substitute a "
                           f"constant"),
                    when=provenance.now_iso()))
                continue
        print(f"\n=== {w.id} ===\n    {' '.join(w.args)}")
        if rate is not None:
            src = ("--rps-reference" if rps_reference
                   else f"this label's measured {SATURATION_ROW}")
            print(f"    offering {rate:,}/s = {w.rps_fraction:.0%} of "
                  f"{sat:,.0f} qps ({src})")
        if w.background:
            print(f"    concurrently: {' '.join(w.background)}")
        def offered(row: Result, rate=rate, w=w, sat=sat) -> Result:
            # What was offered, what it was a fraction of, and whether the
            # reference was pinned or this engine's own — the report can only
            # refuse a cross-arm latency read if the row says which it used.
            # Applied to every measurement, not the first: the retry loops below
            # rebind `r`, and a re-measured open-loop row used to publish with no
            # target at all.
            if rate is None:
                return row
            return dataclasses.replace(
                row, rps_target=rate, rps_fraction=w.rps_fraction,
                saturation_qps=sat,
                rps_reference_source=RpsSource.pinned if rps_reference else RpsSource.own)

        # A pinned row measures what the first pass settled on, so the passes
        # fold one configuration. Both guards matter: without the first, a hand
        # run without `--min-duration` writes 1 for every row and later passes
        # skip the escalation entirely; without the second, an explicit
        # `--n-factor 4` is silently discarded by a stale pin.
        pinned = n_pins.get(w.id) if pinning else None
        base_factor = pinned if pinned is not None else n_factor
        r = offered(run_one(w, uri, results, common, storage, base_factor, stamp, build,
                            warmup, perf_set, sched_sampler))
        # `--min-duration`: a search row that came in under MIN_ROW_S is run
        # again with more queries rather than published as a ramp measurement.
        # Twice at most; a row that is still short at 4x is reported short.
        # Never a row with a concurrent writer: re-running W11 appends the
        # same id range again, which overwrites, and the second search is
        # against a collection the first one already changed.
        tries = 0
        while pinned is None and "--min-duration" in flags:
            action = min_duration_action(w, r, tries)
            if action is None:
                break
            if action != "rerun":
                print(f"    {action}")
                r = dataclasses.replace(r, notes="; ".join(n for n in (r.notes, action) if n))
                break
            tries += 1
            factor = base_factor * (2 ** tries)
            print(f"    short row ({r.duration_s:.2f} s < {MIN_ROW_S:.0f} s); "
                  f"re-running with -n x{factor}")
            r = offered(run_one(w, uri, results, common, storage, factor, stamp, build,
                                warmup, perf_set, sched_sampler))

        # What this row ended up measuring, for the passes that follow. Written
        # per row rather than at the end, so a run that dies mid-table still
        # pins what it finished.
        final_factor = base_factor * (2 ** tries)
        if n_pin_path is not None and pinning and not w.upload_only:
            n_pins[w.id] = final_factor
            try:
                n_pin_path.parent.mkdir(parents=True, exist_ok=True)
                n_pin_path.write_text(json.dumps(n_pins, indent=1, sort_keys=True))
            except OSError as exc:
                print(f"    !! could not write -n pin to {n_pin_path}: {exc}")

        # A row measured under foreign load is refused a ratio, and one of them
        # refuses the whole document. Measuring it again is cheap next to
        # losing the run; `contaminated_action` carries the guards.
        dirty_tries = 0
        while True:
            action = contaminated_action(w, r, dirty_tries)
            if action is None:
                break
            if action != "rerun":
                print(f"    {action}")
                r = dataclasses.replace(r, notes="; ".join(n for n in (r.notes, action) if n))
                break
            dirty_tries += 1
            print(f"    contaminated ({r.foreign}); re-measuring "
                  f"({dirty_tries} of {CONTAMINATED_RETRIES})")
            settle_for_retry()
            r = offered(run_one(w, uri, results, common, storage, final_factor,
                                stamp, build, warmup, perf_set, sched_sampler))
        out.append(r)
        if r.status == Status.ok:
            unit = ""
            if r.qps is not None:
                unit = f"  {r.qps:.0f} qps"
                # Only worth showing both where batching makes them differ.
                if r.rps is not None and abs(r.rps - r.qps) > 0.5:
                    unit += f" ({r.rps:.0f} req/s)"
            wall = f"{r.wall_s:.1f}" if r.wall_s is not None else str(r.seconds)
            print(f"    ok in {wall}s  load {r.load_start}%->{r.load_end}%{unit}")
            if r.upload_s is not None or r.index_wait_s is not None:
                parts = []
                if r.upload_s is not None:
                    parts.append(f"upload {r.upload_s:.3f}s")
                if r.index_wait_s is not None:
                    parts.append(f"time-to-green {r.index_wait_s:.3f}s")
                print(f"    phase {', '.join(parts)}  [bfb's own timings]")
            if r.time_to_green_floored:
                print(f"    !! at bfb's {WAIT_INDEX_FLOOR_S:.0f}s polling floor: "
                      f"wait_index sleeps 1s before its first poll and wants three")
                print("       consecutive Greens, so this is an upper bound on the "
                      "build, not a measurement of it")
            if r.io_source:
                print(f"    disk  {procstat.human_count(r.disk_read_ops)} reads / "
                      f"{procstat.human_count(r.disk_write_ops)} writes, "
                      f"{procstat.human_bytes(r.disk_read_bytes)} in / "
                      f"{procstat.human_bytes(r.disk_write_bytes)} out"
                      f"  (syscalls {procstat.human_count(r.syscall_reads)}/"
                      f"{procstat.human_count(r.syscall_writes)}, all fds)")
                print(f"    store {procstat.human_bytes(r.storage_bytes)} on disk, "
                      f"peak rss {procstat.human_bytes(r.rss_peak_bytes)}"
                      f"  [via {r.io_source}]")
        elif r.status == Status.no_output:
            print(f"    NO OUTPUT after {r.seconds}s: {r.detail}")
        elif r.status == Status.not_applicable:
            print("    n/a, engine declines this workload, as documented:")
            print(f"      {r.detail}")
            print("      (§12 requires UNIMPLEMENTED naming the construct; see §1 for a "
                  "permanent non-goal, §2/§10 M7 for an unbuilt phase)")
        else:
            print(f"    FAILED after {r.seconds}s:")
            for line in r.detail.splitlines():
                print(f"      {line}")
        if r.notes:
            print(f"    note  {r.notes}")
        if r.foreign:
            print(f"    !! FOREIGN LOAD during this row: {r.foreign}")
            print("       this row is not comparable to one taken on an idle host")

    # One machine-readable record per run, merged with anything already there so
    # a re-measured row updates in place rather than truncating the history.
    path = results / "rows.json"
    prev = {}
    if path.exists():
        try:
            prev = {r["id"]: r for r in json.loads(path.read_text())}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # The merge exists so a re-measured row updates in place instead of
            # truncating the history. Silently starting from {} on a parse error
            # truncates exactly that history, which is the thing this code is
            # for. Keep the damaged file and say so.
            broken = path.with_suffix(".json.broken")
            path.rename(broken)
            print(f"  !! {path.name} is unreadable ({exc.__class__.__name__}); "
                  f"moved to {broken.name}. Rows already measured in earlier runs "
                  f"are NOT in this file.", file=sys.stderr)
            prev = {}
    for r in out:
        prev[r.id] = r.__dict__
    order = [w.id for w in table() if w.id in prev]
    path.write_text(json.dumps([prev[i] for i in order], indent=1) + "\n")

    # §8's gate, run rather than described. `results.py` refuses a perf row
    # without a green conformance row, and until this call existed nothing ever
    # asked it to, so the rule held only in the documentation.
    if "--sink" in flags:
        print("\n=== results sink (§8) ===")
        subprocess.run([sys.executable, str(ROOT / "bench/harness/results.py"),
                        "ingest", label])

    if "--report" in flags:
        write_report(label)

    print(f"\nresults in {results}  (gate: {Gate.passed if gate_ok else Gate.failed})")

    # Non-zero when a row did not measure. This returned 0 unconditionally
    # while three callers tested it, so a run against an engine that was never
    # up produced 26 FAILED rows and exited 0 — the sift1m run
    # recorded `qdrant 0/26 ok` for a whole arm. `n/a` is not a failure: it is
    # the engine declining a construct §2 has not built.
    unmeasured = [r.id for r in out if r.status in (Status.failed, Status.no_output)]
    if unmeasured:
        print(f"  {len(unmeasured)} row(s) did not measure: "
              f"{', '.join(unmeasured[:8])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
