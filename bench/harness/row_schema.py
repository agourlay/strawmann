"""The schema of one row of `rows.json`: `Result` and the enums it carries.

Moved out of `workloads.py`, which re-exports every name here, so
`workloads.Result`, `workloads.Status` and `workloads.Gate` still resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


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


@dataclass
class RowCore:
    """What was measured, whether it held, and the headline rate."""
    id: str
    status: Status
    seconds: int
    load_start: int
    load_end: int
    foreign: str
    qps: float | None = None
    rps: float | None = None
    detail: str = ""
    #: Free-form caveats the tables print beside the number: `short (<2 s)`,
    #: `per-batch latency`, the W5 query strategy, W11's overlap.
    notes: str = ""


@dataclass(kw_only=True)
class LoadFields:
    """How the load was offered and what latency it saw."""
    #: §7.4: `--rps` and `--parallel` measure different things, and a
    #: closed-loop p99 is not a latency result. Which one produced this row is
    #: a property of the row, not something a reader should infer from its id.
    load_mode: LoadMode = LoadMode.unset
    #: What the load generator offered: bfb's `-p`, `-t` and `-c` for this row,
    #: with `client_defaults` naming the ones it did not pin. The qps column is
    #: not readable across rows without them (`client_concurrency`).
    client_parallel: int | None = None
    client_threads: int | None = None
    client_connections: int | None = None
    client_defaults: str = ""
    #: Client-side and server-side percentiles, in microseconds.
    latency: dict = field(default_factory=dict)
    #: bfb's `Median qps` is the median of a per-request EWMA-style series, and
    #: on a row that finishes in a second or two it understates: strawmann's W4
    #: printed 27,201 while 50,000 queries in 1.61 s is 31,056. `qps` is the
    #: wall-clock figure `n_queries / duration_s` from bfb's own JSON, which is
    #: what "queries per second" means; the median is kept as `qps_bfb_median`.
    qps_bfb_median: float | None = None
    duration_s: float | None = None
    n_queries: int | None = None
    #: The `-n` this row was invoked with (after `--n-factor` /
    #: `--min-duration` scaling) and `stamp_hash` of the harness stamp it ran
    #: under. `rows.json` merges rows in place, so this is what says which rows
    #: the label's `run.json` stamp actually describes.
    n_requested: int | None = None
    #: Wall seconds of the discarded warm-up pass that preceded the measured
    #: one (§7.1), or None when there was none (`--no-warmup`, upload rows).
    warmup_s: float | None = None
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


@dataclass(kw_only=True)
class SearchFields:
    """The search the row sent, and what may be joined to it."""
    #: The `ef` this row searched at, when it stated one. It is what the recall
    #: sweep joins on, and deriving it downstream means re-reading §4's table in
    #: two places.
    ef: int | None = None
    #: Exhaustive search (`--search-exact`), whose recall@10 is 1.0 by
    #: construction. See `exact_of`.
    exact: bool = False
    #: The collection this row searched, and whether a recall sweep of it
    #: describes the row. Both are the join key for recall (with `ef`), written
    #: into the artifact so no reader has to re-derive §4's table to find them.
    collection: str | None = None
    recall_joinable: bool = True
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


@dataclass(kw_only=True)
class IngestFields:
    """An upload row's phases."""
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


@dataclass(kw_only=True)
class BackgroundWriteFields:
    """A mixed row's concurrent writer."""
    #: The second bfb process a concurrent row (W11) ran, and how the two
    #: overlapped in time. `overlap_s` is the interval during which both were
    #: in flight; a search that outlives the append is partly measuring a quiet
    #: collection and the note says by how much.
    background_pps: float | None = None
    background_s: float | None = None
    overlap_s: float | None = None
    #: W11: how much of the search phase (bfb's `duration_secs`) the append
    #: was in flight for. Below `W11_MIN_OVERLAP` the rest of the search ran
    #: against the rebuild the append provoked, not a concurrent write, and is
    #: not the row §4 describes; `compare` refuses it.
    write_overlap_pct: float | None = None


@dataclass(kw_only=True)
class IoFields:
    """`procstat.IO_FIELDS` and the storage level."""
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
    storage_bytes: int | None = None


@dataclass(kw_only=True)
class MemoryFields:
    """Resident memory, `procstat.RSS_FIELDS` and the peak."""
    rss_peak_bytes: int | None = None
    #: Resident bytes by what backs them, as levels. Two engines holding the
    #: same total differ in this split: one allocated it, the other mapped a
    #: file and let the kernel decide what stays.
    rss_anon_bytes: int | None = None
    rss_file_bytes: int | None = None
    rss_shmem_bytes: int | None = None


@dataclass(kw_only=True)
class SchedFields:
    """`procstat.SCHED_FIELDS`: what the scheduler did to the engine."""
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


@dataclass(kw_only=True)
class PressureFields:
    """`procstat.PSI_FIELDS` and whose cgroup they describe."""
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


@dataclass(kw_only=True)
class PerfFields:
    """The `perf stat` sidecar's counters (`perfstat`)."""
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


@dataclass(kw_only=True)
class ProvenanceFields:
    """When, under which gate and harness, on which build."""
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
    #: Which visited set this row's engine was built with (`-Dvisited`), from
    #: the banner. `None` for Qdrant, which has no such knob, and for a
    #: strawmANN built before the flag existed.
    visited_set: str | None = None
    harness_hash: str | None = None


@dataclass
class Result(ProvenanceFields, PerfFields, PressureFields, SchedFields, MemoryFields,
             IoFields, BackgroundWriteFields, IngestFields, SearchFields, LoadFields, RowCore):
    """One row of §4's table, as measured.

    Deliberately flat, and this is the one place in the harness where that is
    the right call. `rows.json` is an artifact: `compare.py`, `report.py`,
    `results.py` and every run already on disk read these names. Nesting the
    I/O counters or the storage levels into sub-objects would be tidier code
    and a silent mismatch against every file measured before the change, on a
    project whose whole subject is results that cannot be compared across
    runs. The typing lives in `Status`, `LoadMode` and `procstat.Snapshot`
    instead, where it costs the artifact nothing.

    Composed of the groups above rather than written as one list of 109
    fields. The row stays flat, which is the property this docstring defends:
    every group is a base class, so its fields are `Result`'s own fields,
    constructed, read and serialised by the same names as before. Only
    `RowCore`'s fields are positional; every other group is keyword-only,
    which is also what lets a group with defaults precede them.
    """
