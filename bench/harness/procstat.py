#!/usr/bin/env python3
"""What an engine costs in storage and I/O, read from /proc and cgroup v2.

§4's table measures throughput and §7.1 measures the environment; neither says
anything about what the engine *stores* or how much I/O it does to serve a row.
For a comparison whose whole subject is a memory-resident engine against a
disk-backed one, that is a strange omission: strawmann's speed is partly bought
by never touching a block device, and a table of queries per second cannot show
the price.

What is measured, and the axes it is deliberately split across:

  * **storage on disk**, bytes under the engine's storage directory.
  * **peak RSS**, because a RAM-resident engine and a disk-backed one trade
    these two against each other, and reporting either alone favours one of
    them by construction.
  * **disk operations and disk bytes** per workload row, at the block layer.
  * **syscall counts**, kept separate from the above and never presented as
    "I/O operations".

## Three counters that look alike and are not

The last point is not pedantry, it is the trap this module exists to avoid.
`/proc/<pid>/io`'s `syscr`/`syscw` count read and write syscalls on *every*
descriptor, sockets included. Measured here, 5,000 queries against strawmann
produced 24,995 `syscr` and **zero** bytes at the block layer: that column is
the network, and putting it next to a disk-backed engine's block-layer count
under a heading like "I/O ops" would compare the two engines' *sockets against
one engine's disk*. §4.1 already caught this shape once, reading bfb's `rps`
where `qps` was meant.

So:

  `syscall_reads` / `syscall_writes`   all descriptors, mostly network
  `disk_read_ops` / `disk_write_ops`   block-layer requests
  `disk_read_bytes` / `disk_write_bytes`   block-layer bytes

Only the last two rows are available from both interfaces, which is why they
are the ones a comparison should lead with.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class IoSource(StrEnum):
    """Which interface a row's I/O counters came from.

    Not decoration: `proc` counts syscalls and block-layer *bytes*, `cgroup`
    counts block-layer *operations* and bytes, and a comparison that mixes them
    is a units error. The value is carried into every result row, so it is a
    closed set rather than a string spelled out in three files.
    """

    #: `/proc/<pid>/io`. Owner-only.
    proc = "proc"
    #: cgroup v2 `io.stat`. World-readable where the io controller is enabled.
    cgroup = "cgroup"
    #: Both, each filling what the other cannot supply.
    both = "proc+cgroup"
    #: Neither could be read.
    none = ""


#: The engine processes a run may legitimately find. Prefixes, because Linux
#: truncates `comm` to 15 characters and the ISA arms are `strawmann-avx5`.
ENGINE_PREFIXES = ("strawmann", "qdrant")


def engine_processes() -> list[tuple[int, str]]:
    """Every running engine process, as `(pid, comm)`.

    More than one means two engines are up at once, which §7.1 forbids for a
    measurement: they share the page cache and the memory bus. The caller
    reports that rather than picking one.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue  # exited between listing and reading
        if any(comm.startswith(p) for p in ENGINE_PREFIXES):
            found.append((int(entry.name), comm))
    return sorted(found)


def _proc_io(pid: int) -> dict[str, int] | None:
    """`/proc/<pid>/io`. Owner-only, so this fails for a container as root."""
    try:
        text = Path(f"/proc/{pid}/io").read_text()
    except OSError:
        return None
    f = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        try:
            f[k.strip()] = int(v)
        except ValueError:
            continue
    if "syscr" not in f:
        return None
    return {
        # `syscr`/`syscw` count read and write syscalls on *every* descriptor,
        # sockets included. On a search row they are almost entirely network:
        # 5,000 queries against strawmann measured 24,995 reads and zero bytes
        # at the block layer. Reporting them as "I/O operations" next to a
        # disk-backed engine's block-layer count would be a units error of the
        # same family as reading `rps` for `qps`.
        "syscall_reads": f.get("syscr", 0),
        "syscall_writes": f.get("syscw", 0),
        # These two are the block layer, and they are the ones that compare.
        "disk_read_bytes": f.get("read_bytes", 0),
        "disk_write_bytes": f.get("write_bytes", 0),
    }


def parse_io_stat(text: str) -> dict[str, int] | None:
    """cgroup v2 `io.stat`, summed over devices.

    One line per device, `major:minor key=value ...`. Summing over devices is
    right here: a container's storage volume and its image layers can sit on
    different block devices, and the question is what the engine did in total.

    An *empty* file is zero, not unmeasured, and the difference cost a published
    figure: a scope the harness has just created lists no device until its first
    block I/O completes, so W0-upload's first pass returned `None` for
    `disk_read_ops` and `disk_write_ops` while its byte counters were present,
    and `aggregate` then refused the median with "measured in 2 of 3 passes".
    The very next row in that pass read 0 and 987. A file with content that
    parses to nothing is still unmeasured: that is a format this does not
    understand, and guessing zero there would turn a parser failure into a
    measurement.
    """
    total = {"disk_read_ops": 0, "disk_write_ops": 0,
             "disk_read_bytes": 0, "disk_write_bytes": 0}
    key = {"rios": "disk_read_ops", "wios": "disk_write_ops",
           "rbytes": "disk_read_bytes", "wbytes": "disk_write_bytes"}
    seen = False
    for line in text.splitlines():
        for field in line.split()[1:]:
            k, _, v = field.partition("=")
            if k in key:
                try:
                    total[key[k]] += int(v)
                    seen = True
                except ValueError:
                    pass
    if seen:
        return total
    return total if not text.strip() else None


def _cgroup_io(pid: int) -> dict[str, int] | None:
    """The process's own cgroup `io.stat`.

    The fallback for a process whose `/proc/<pid>/io` we may not read. Counts
    block-layer requests, not syscalls, which is why the caller keeps the
    source alongside the numbers.

    Deliberately does *not* walk up to a parent cgroup when the leaf has no
    `io.stat`, which is the common case for a user session scope: the parent's
    counters include everything else on the machine, so reporting them as the
    engine's would turn "we could not measure this" into a large, specific,
    wrong number. Unmeasured is reported as unmeasured.
    """
    d = cgroup_dir(pid)
    if d is None:
        return None
    try:
        text = (d / "io.stat").read_text()
    except OSError:
        return None
    return parse_io_stat(text)


def cgroup_dir(pid: int) -> Path | None:
    """The unified (v2) cgroup directory `pid` lives in, or None.

    Only the `0::` line, which is v2's. A hybrid host also lists v1 controllers
    with their own paths, and joining a v1 path onto `/sys/fs/cgroup` produces a
    directory that either does not exist or belongs to a different hierarchy.
    """
    try:
        rel = ""
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            parts = line.split(":")
            if len(parts) == 3 and parts[0] == "0":
                rel = parts[2].strip()
    except OSError:
        return None
    if not rel:
        return None
    d = Path("/sys/fs/cgroup") / rel.lstrip("/")
    return d if d.is_dir() else None


def cgroup_holds_only(pid: int, d: Path,
                      ancestors_of: Callable[[int], set[int]] | None = None) -> bool:
    """Whether `d` contains the engine's own process tree and nothing else.

    The gate on every pressure figure. A container gets a scope of its own, so
    its `io.pressure` is the engine's; strawmann started from a shell lands in
    the user session's scope, alongside the harness, bfb and the editor, and
    differencing that measures the desktop.

    Ancestors count as the engine — a container whose entrypoint is a script
    leaves that script alive as the engine's parent, and rejecting it cost the
    containerised arm its pressure columns. A sibling is still foreign, which
    is what keeps the user-session case refused.

    Takes `ancestors_of` so the shapes can be tested without building a fake
    `/proc`.
    """
    up = ancestors_of or ancestors
    try:
        pids = [int(x) for x in (d / "cgroup.procs").read_text().split()]
    except (OSError, ValueError):
        return False
    if not pids:
        return False
    mine = up(pid)                 # the engine and everything above it
    return all(other in mine or pid in up(other) for other in pids)


def parse_pressure(text: str) -> dict[str, float] | None:
    """`some` and `full` totals out of one `*.pressure` file, in seconds.

    `avg10`/`avg60`/`avg300` are dropped rather than recorded. They are
    decaying averages over the last ten seconds to five minutes, and a row is a
    window with two ends: only `total=` (microseconds, monotonic) can be
    differenced across one. A ten-second average sampled at the end of a
    forty-second row describes its last quarter and reads like a property of
    the whole.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        for field in parts[1:]:
            k, _, v = field.partition("=")
            if k != "total":
                continue
            try:
                out[parts[0]] = int(v) / 1e6
            except ValueError:
                pass
    return out or None


def pressure_counters(pid: int) -> tuple[dict[str, float | None], str]:
    """Cumulative PSI stall time for the engine's cgroup, and whose it is.

    The second element is the scope: `engine` when the cgroup holds this
    process and nothing else, `shared` when it holds other processes too, and
    empty when the host has no PSI (a kernel without `CONFIG_PSI`, or one
    booted `psi=0`). Only `engine` carries numbers; `shared` is recorded so the
    report can say *why* the column is blank rather than leaving a reader to
    guess that the kernel lacks the feature.
    """
    out: dict[str, float | None] = dict.fromkeys(PSI_FIELDS)
    d = cgroup_dir(pid)
    if d is None:
        return out, ""
    if not cgroup_holds_only(pid, d):
        return out, "shared"
    found = False
    for res, prefix in (("cpu", "psi_cpu"), ("io", "psi_io"), ("memory", "psi_mem")):
        text = _read(str(d / f"{res}.pressure"))
        if not text:
            continue
        parsed = parse_pressure(text)
        if not parsed:
            continue
        for kind in ("some", "full"):
            if kind in parsed:
                out[f"{prefix}_{kind}_s"] = parsed[kind]
                found = True
    return out, ("engine" if found else "")


#: Every field a row can carry, so a missing source leaves an explicit None
#: rather than an absent key that later reads as zero.
IO_FIELDS = ("syscall_reads", "syscall_writes",
             "disk_read_ops", "disk_write_ops",
             "disk_read_bytes", "disk_write_bytes")

#: Fields a row carries only when a concurrent writer ran during it (W11,
#: W11-steady). Their presence is what marks a row as mutating, which no single
#: field does reliably: `background_pps` is absent when bfb's upload stats are.
MUTATING_KEYS = ("background_pps", "background_s", "overlap_s")


def is_mutating(row) -> bool:
    """Did a concurrent writer run during this row?"""
    return any(row.get(k) is not None for k in MUTATING_KEYS)


def settled_rows(rows: list) -> list:
    """The rows an end-state level may be read from: those with no writer.

    `storage on disk` is a level, and a level is the last row that saw one. The
    last row of a run is W11, which appends 200,000 points, and Qdrant's
    optimiser rewrites segments while it does: its storage moves 4.03 GiB to
    10.11 GiB inside that row and the sample lands wherever the rewrite happens
    to be. The same row read 3.47 GiB on a single-pass run of the same binary
    three hours earlier, and the published `rel-0908` page says 10.7 GiB, a 3x
    spread in a cell a reader takes for a property of the format (findings 53).

    Excluding the mutating rows gives the footprint of the settled corpus,
    which is what the cell is read as, and it reproduces: 3.47 GiB on both of
    those runs. A peak is different and is not filtered here, because a peak
    during a rewrite is still a peak the engine reached.

    Falls back to every row when a run is *all* mutating rows, so a caller
    never gets an empty list and a partial run still reports something.
    """
    kept = [r for r in rows if not is_mutating(r)]
    return kept or list(rows)


#: What the scheduler and the fault handler did to the engine during a row.
#:
#: Split by how trustworthy the difference is, which is not cosmetic:
#:
#:   * `cpu_user_s`, `cpu_system_s`, `minor_faults`, `major_faults` come from
#:     `/proc/<pid>/stat`, which the kernel reports for the whole thread group
#:     and which keeps an exited thread's totals. Monotonic, so the difference
#:     is exact.
#:   * `oncpu_s`, `runqueue_wait_s`, `timeslices`, the two context-switch
#:     counts and `migrations` have no thread-group form and are summed over
#:     `/proc/<pid>/task/*`. A thread that exits between two snapshots takes
#:     its counters with it, so the sum can fall and the difference is then
#:     unusable rather than merely small — the bulk index build starts and
#:     joins threads inside a single row, so this is the normal case, not an
#:     edge one. `threads` is recorded beside them so a reader can see it.
#:
#: `runqueue_wait_s` is the one worth leading with: it is time the engine was
#: runnable and not running, which is the difference between "slow" and
#: "not given a core", and no throughput number distinguishes those.
#: `blkio_delay_s` joins the exact group: it comes from the same thread-group
#: line and survives a thread exit for the same reason the CPU times do.
SCHED_FIELDS = ("cpu_user_s", "cpu_system_s", "minor_faults", "major_faults",
                "blkio_delay_s",
                "oncpu_s", "runqueue_wait_s", "timeslices",
                "ctx_switches_voluntary", "ctx_switches_involuntary",
                "migrations", "threads", "sched_coverage", "sched_sampled")

#: Pressure-stall time for the engine's own cgroup, in seconds, split the way
#: the kernel splits it: `some` is "at least one task was stalled on this
#: resource", `full` is "every runnable task was". `full` is the one that costs
#: throughput; `some` is the one that appears first.
#:
#: This is the only interface here that measures *memory* and *I/O* stalls
#: directly rather than by proxy, and the only one that works identically for a
#: containerised engine and a native one — both have a cgroup, and neither has
#: to be readable by us for it to be counted.
#:
#: `full` means different things at different scopes, and the difference is
#: worth stating because it inverts the obvious reading. At the system root
#: (`/proc/pressure/cpu`) the kernel defines CPU `full` as zero by construction
#: — there is always something runnable somewhere — so a root reading of zero
#: says nothing. Inside a cgroup it is a real measurement: every task *in that
#: cgroup* stalled, which for an engine with the machine to itself is exactly
#: the quantity wanted. These are read from the engine's own cgroup, so `full`
#: is the meaningful one; measured on this host, a single-threaded moment gives
#: `some` and `full` the same total, which is correct rather than a parse bug.
PSI_FIELDS = ("psi_cpu_some_s", "psi_cpu_full_s",
              "psi_io_some_s", "psi_io_full_s",
              "psi_mem_some_s", "psi_mem_full_s")

#: Resident bytes split by what backs them, as levels rather than differences.
#: Peak RSS says how much the engine held; this says whether it was its own
#: allocation (`anon`) or the page cache under an mmap (`file`), which is the
#: whole difference between a RAM-resident engine and a disk-backed one holding
#: the same total.
RSS_FIELDS = ("rss_anon_bytes", "rss_file_bytes", "rss_shmem_bytes")

#: Below this, the thread-summed counters are suppressed rather than reported.
#: A dropped thread does not make its counter negative, it makes it *small*,
#: which is the dangerous shape: an index-build row measured 12.98 s of CPU and
#: 0.076 s of on-cpu time, because the bulk-build threads did the work and then
#: exited, and the surviving thread count was identical either side. Nothing in
#: the counters themselves says so.
_COVERAGE_MIN = 0.95
#: Below this much CPU in a row, `sched_coverage` is not computed at all: CPU
#: time is quantised to 10 ms ticks, so an idle row divides by noise.
_COVERAGE_FLOOR_S = 0.1

#: The subset summed over live threads, and therefore droppable. `threads` is a
#: level rather than a counter and is excluded.
_THREAD_SUMMED = ("oncpu_s", "runqueue_wait_s", "timeslices",
                  "ctx_switches_voluntary", "ctx_switches_involuntary",
                  "migrations")

_CLK_TCK = os.sysconf("SC_CLK_TCK")


def parse_proc_stat(text: str) -> dict[str, float] | None:
    """CPU time and fault counts out of `/proc/<pid>/stat`.

    `comm` is an arbitrary string in parentheses and may itself contain spaces
    or brackets, so the fields are taken relative to the *last* `)` rather than
    by splitting the line — the usual way this parser breaks.
    """
    close = text.rfind(")")
    if close < 0:
        return None
    rest = text[close + 2:].split()
    # `rest[0]` is `state`, proc(5)'s field 3, so field N is `rest[N - 3]`:
    # minflt 10 -> 7, majflt 12 -> 9, utime 14 -> 11, stime 15 -> 12. Counting
    # from the wrong end of that arithmetic reads `session` and `tpgid`
    # instead, which is silent for minflt and gives majflt a value of -1.
    if len(rest) < 13:
        return None
    try:
        out = {"minor_faults": float(rest[7]), "major_faults": float(rest[9]),
               "cpu_user_s": int(rest[11]) / _CLK_TCK,
               "cpu_system_s": int(rest[12]) / _CLK_TCK}
    except ValueError:
        return None
    # `delayacct_blkio_ticks`, proc(5)'s field 42, is the one number in this
    # file that answers "was the engine waiting for a *disk*". `runqueue_wait_s`
    # is time runnable and not running; this is time not runnable at all,
    # blocked in the block layer, and a disk-backed engine against a
    # RAM-resident one is the comparison where the difference is the subject.
    #
    # Parsed unconditionally so this stays a pure function of its text; whether
    # the kernel is *collecting* it is a property of the host and is applied by
    # `sched_counters`. A kernel with `task_delayacct=0` reports the field as a
    # permanent zero, which is the shape this module exists to refuse.
    if len(rest) >= 40:
        try:
            out["blkio_delay_s"] = int(rest[39]) / _CLK_TCK
        except ValueError:
            pass
    return out


def parse_schedstat(text: str) -> dict[str, float] | None:
    """`/proc/<pid>/task/<tid>/schedstat`: on-cpu ns, runqueue-wait ns, slices."""
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return {"oncpu_s": int(parts[0]) / 1e9,
                "runqueue_wait_s": int(parts[1]) / 1e9,
                "timeslices": float(parts[2])}
    except ValueError:
        return None


#: Whether the kernel is collecting delay accounting, read once. `None` until
#: the first ask; the sysctl does not change under a run.
_DELAYACCT: bool | None = None


def delayacct_on() -> bool:
    """Whether `delayacct_blkio_ticks` is being maintained at all.

    `kernel.task_delayacct` defaults to 0 on most distributions, and a task
    started while it is off reports the field as a permanent zero. Zero is
    exactly what "this engine never waited for the disk" looks like, so the
    distinction has to be drawn here or the strongest claim in the storage
    section — that the RAM-resident engine never blocks on a device — would be
    manufactured by a sysctl rather than measured.

    `bench/setup.py apply` turns it on; `check_delayacct` reports it.
    """
    global _DELAYACCT
    if _DELAYACCT is None:
        v = _read("/proc/sys/kernel/task_delayacct")
        # Absent on a kernel built without CONFIG_TASK_DELAY_ACCT, where the
        # field is present in /proc/<pid>/stat and always zero.
        _DELAYACCT = (v or "0").strip() == "1"
    return _DELAYACCT


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def thread_counters(pid: int) -> dict[str, dict[str, float]]:
    """Per-thread scheduler counters, keyed by tid, at this instant.

    The per-tid split `sched_counters` immediately sums away. Kept separate
    here because a sum taken once at the end cannot see a thread that has
    already exited, and that is the whole difficulty — see `SchedSampler`.
    """
    out: dict[str, dict[str, float]] = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        t = f"/proc/{pid}/task/{tid}"
        got: dict[str, float] = {}
        if (sc := _read(f"{t}/schedstat")) and (p := parse_schedstat(sc)):
            got.update(p)
        if sd := _read(f"{t}/sched"):
            for row in sd.splitlines():
                if row.startswith("se.nr_migrations"):
                    try:
                        got["migrations"] = float(int(row.split(":")[1]))
                    except (IndexError, ValueError):
                        pass
                    break
        # The two context-switch counters live in `status`, not `schedstat`.
        # The sampler's whole purpose is to bank the thread-summed counters
        # before a thread exits, and `_THREAD_SUMMED` names these two; a
        # sampler that did not read them left them to the end-of-row
        # subtraction, published unsuppressed on every sampled row.
        if st := _read(f"{t}/status"):
            for line, key in (("voluntary_ctxt_switches:", "ctx_switches_voluntary"),
                              ("nonvoluntary_ctxt_switches:", "ctx_switches_involuntary")):
                for row in st.splitlines():
                    if row.startswith(line):
                        try:
                            got[key] = float(int(row.split()[1]))
                        except (IndexError, ValueError):
                            pass
                        break
        if got:
            out[tid] = got
    return out


class SchedSampler:
    """Per-thread scheduler counters banked *while* the row runs.

    `sched_counters` sums over the threads alive when it is called, while
    `/proc/<pid>/stat` keeps the time of threads that have exited — so the two
    halves of `sched_coverage` disagree exactly when an engine retires threads
    mid-row. Qdrant's W3 on dbpedia-openai-1m measured 0.106, meaning 89% of its
    CPU belonged to threads gone by the end; strawmANN, whose workers live for the
    process, reads ~1.0 (findings 41).

    This banks each tid's first and last observation as it goes. A tid seen in the
    first poll contributes `last - first`; one appearing later contributes `last`,
    since its counters started at zero. The undercount is bounded by what a new
    thread did before its first poll.

    The cost is `open`s — measured, not argued: ~`threads x 45 us` per poll, so at
    the 250 ms default 0.2% of a core for strawmANN's 9 threads and 1.4% for
    Qdrant's 37-70. Cheap enough to default on, against the perf sidecar's 4.25%.
    Charged to the client side too: `workloads.py` runs under `taskset`, so the
    sampler competes with bfb, not the engine — which puts the open-loop rows
    (findings 15) at 1.4% of one of four client cores. `overhead_s` and `polls`
    are kept per sampler so the figure can be re-taken rather than trusted.
    """

    def __init__(self, pid: int, interval_s: float = 0.25) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self._first: dict[str, dict[str, float]] = {}
        self._last: dict[str, dict[str, float]] = {}
        self._new: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # `stop` polls once more after joining, and the join has a timeout, so
        # the two can in principle be inside `_poll` at once. The window is
        # tiny -- a poll is `threads x 45 us` against a join budget of four
        # intervals -- and the cost of losing the race is a double-counted or
        # dropped delta on one thread, not a crash. Cheap to close, so closed.
        self._lock = threading.Lock()
        self.polls = 0
        self.overhead_s = 0.0
        self._started = False

    def _poll(self) -> None:
        t0 = time.monotonic()
        counters = thread_counters(self.pid)
        with self._lock:
            self._bank(counters)
        self.polls += 1
        self.overhead_s += time.monotonic() - t0

    def _bank(self, counters: dict[str, dict[str, float]]) -> None:
        for tid, got in counters.items():
            if tid not in self._first:
                self._first[tid] = got
                if self._started:
                    # Created after the row began: its counters started at zero,
                    # so the whole of `last` is this row's.
                    self._new.add(tid)
            self._last[tid] = got

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._poll()

    def start(self) -> SchedSampler:
        self._poll()          # the baseline, before anything is marked new
        self._started = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sched-sampler")
        self._thread.start()
        return self

    def stop(self) -> dict[str, float]:
        """Summed per-thread deltas over every tid ever seen."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4)
        self._poll()          # catch whatever happened since the last tick
        sums: dict[str, float] = {}
        with self._lock:
            first, last_seen = dict(self._first), dict(self._last)
        for tid, last in last_seen.items():
            base = {} if tid in self._new else first.get(tid, {})
            for k, v in last.items():
                sums[k] = sums.get(k, 0.0) + (v - base.get(k, 0.0))
        return sums


def cpu_seconds(pid: int) -> float | None:
    """The engine's total CPU so far, threads included, or None if it is gone.

    The thread-group line, so it keeps the time of threads that have already
    exited — which is the point: background work that finishes is exactly what
    this is used to wait for.
    """
    stat = _read(f"/proc/{pid}/stat")
    if stat is None:
        return None
    got = parse_proc_stat(stat)
    if not got:
        return None
    return (got.get("cpu_user_s") or 0.0) + (got.get("cpu_system_s") or 0.0)


def sched_counters(pid: int) -> dict[str, float | None] | None:
    """Scheduler and fault counters for the whole engine, threads included.

    Free in the sense that matters here: no root, no eBPF, no probe on a hot
    path — just files the kernel already keeps. Reading them costs a few dozen
    `open`s per snapshot, twice per row, which is not measurable against a row
    that runs for seconds.
    """
    stat = _read(f"/proc/{pid}/stat")
    if stat is None:
        return None
    out: dict[str, float | None] = dict.fromkeys(SCHED_FIELDS)
    base = parse_proc_stat(stat)
    if base:
        out.update(base)
    # Off means "not collected", not "no delay". See `delayacct_on`.
    if not delayacct_on():
        out["blkio_delay_s"] = None

    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    out["threads"] = float(len(tids))

    sums: dict[str, float] = {}
    seen: set[str] = set()
    for tid in tids:
        t = f"/proc/{pid}/task/{tid}"
        if (s := _read(f"{t}/schedstat")) and (p := parse_schedstat(s)):
            for k, v in p.items():
                sums[k] = sums.get(k, 0.0) + v
                seen.add(k)
        if st := _read(f"{t}/status"):
            for line, key in (("voluntary_ctxt_switches:", "ctx_switches_voluntary"),
                              ("nonvoluntary_ctxt_switches:", "ctx_switches_involuntary")):
                for row in st.splitlines():
                    # `nonvoluntary_` ends with the same text as `voluntary_`,
                    # so the prefix has to be anchored or every nonvoluntary
                    # count lands in both fields.
                    if row.startswith(line):
                        try:
                            sums[key] = sums.get(key, 0.0) + int(row.split()[1])
                            seen.add(key)
                        except (IndexError, ValueError):
                            pass
                        break
        if sc := _read(f"{t}/sched"):
            for row in sc.splitlines():
                if row.startswith("se.nr_migrations"):
                    try:
                        sums["migrations"] = sums.get("migrations", 0.0) + int(row.split(":")[1])
                        seen.add("migrations")
                    except (IndexError, ValueError):
                        pass
                    break
    # A counter no thread reported is absent (the kernel lacks CONFIG_SCHEDSTATS
    # or CONFIG_SCHED_DEBUG), which is not the same as measuring zero.
    for k in seen:
        out[k] = sums[k]
    return out


def io_counters(pid: int) -> tuple[dict[str, int | None], IoSource]:
    """Cumulative I/O for `pid`, taking each field from whichever source has it.

    Both interfaces are consulted because neither is complete: `/proc/<pid>/io`
    gives syscall counts and block-layer *bytes* but no operation count, and is
    readable only by the owner, so a root container is opaque to it; cgroup v2
    `io.stat` gives operations and bytes and is world-readable, but exists only
    where the io controller is enabled, which a plain user session usually lacks.

    So one run can end up with disk *ops* for the containerised engine and only
    disk *bytes* for the local one. That asymmetry is reported, not smoothed over:
    the alternative quotes syscall counts for one engine beside block-layer counts
    for the other and looks like one measurement.
    """
    out: dict[str, int | None] = dict.fromkeys(IO_FIELDS)
    sources = []

    p = _proc_io(pid)
    if p is not None:
        out.update(p)
        sources.append("proc")

    # The same gate PSI applies: a leaf scope that also holds the harness, bfb
    # and the operator's editor reports their block I/O as the engine's, and
    # `io.stat` used to be read from it without asking.
    d = cgroup_dir(pid)
    c = _cgroup_io(pid) if d is not None and cgroup_holds_only(pid, d) else None
    if c is not None:
        out["disk_read_ops"] = c["disk_read_ops"]
        out["disk_write_ops"] = c["disk_write_ops"]
        # cgroup byte counts only fill in what /proc could not supply, so a run
        # that has both does not report one engine's bytes from two interfaces.
        if out["disk_read_bytes"] is None:
            out["disk_read_bytes"] = c["disk_read_bytes"]
            out["disk_write_bytes"] = c["disk_write_bytes"]
        sources.append("cgroup")

    if not sources:
        return dict.fromkeys(IO_FIELDS), IoSource.none
    return out, IoSource("+".join(sources))


def rss_peak_bytes(pid: int) -> int | None:
    """`VmHWM`, the peak resident set since the process started.

    Peak rather than current, because the interesting number for a memory
    resident engine is the high-water mark: a table quoting RSS sampled between
    two rows would report whatever the allocator happened to be holding.
    """
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def rss_split(pid: int) -> dict[str, int | None]:
    """Resident bytes by what backs them, from `/proc/<pid>/status`.

    Anonymous memory is the engine's own allocation; file-backed is page cache
    it has mapped. The split is the difference between two engines that hold
    the same total: one that read its index into memory it owns, and one that
    mapped a file and let the kernel decide what stays.

    From `status` rather than `smaps_rollup`, which also reports
    `AnonHugePages` and would answer §5.5's hugepage question directly.
    `smaps_rollup` walks the process's page tables under `mmap_lock` to produce
    that answer, and this is read twice per row against an engine holding tens
    of gigabytes. A measurement that perturbs the engine to describe it is the
    wrong trade here; the THP question belongs to a diagnostic run.
    """
    out: dict[str, int | None] = dict.fromkeys(RSS_FIELDS)
    text = _read(f"/proc/{pid}/status")
    if text is None:
        return out
    keys = {"RssAnon:": "rss_anon_bytes", "RssFile:": "rss_file_bytes",
            "RssShmem:": "rss_shmem_bytes"}
    for line in text.splitlines():
        head = line.split(":")[0] + ":"
        if head in keys:
            try:
                out[keys[head]] = int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                pass
    return out


def dir_bytes(path: str | os.PathLike) -> int | None:
    """Bytes on disk under `path`, following no symlinks out of it.

    `st_blocks`, not `st_size`: qdrant's segments are ordinary files but a
    sparse or preallocated one would make apparent size a fiction, and the
    question here is what the device actually holds.
    """
    root = Path(path)
    if not root.exists():
        return None
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                continue  # compaction can delete a segment mid-walk
    return total


def ancestors(pid: int) -> set[int]:
    """`pid` and every process above it, up to init.

    Docker reports a container's `State.Pid`, which is PID 1 *inside* the
    container. The process that shows up in the host's `/proc` as `qdrant` is
    that process's child, so matching the two directly finds nothing: measured
    on this host, `State.Pid` 349653 against a visible engine at 349761. Every
    storage figure for the containerised engine read "unknown" because of it,
    which is honest and useless.
    """
    out: set[int] = set()
    cur = pid
    for _ in range(64):  # a cycle is impossible, a bound is still cheap
        if cur <= 0 or cur in out:
            break
        out.add(cur)
        try:
            text = Path(f"/proc/{cur}/status").read_text()
        except OSError:
            break
        ppid = None
        for line in text.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                break
        if ppid is None:
            break
        cur = ppid
    return out


def docker_storage_for(pid: int) -> str | None:
    """The host path a container's storage volume is bind-mounted from.

    Qdrant's data lives inside the container, which we cannot walk without
    root; the bind mount's host side we can. `docs/comparison-sift1m.md` §4b records
    why this matters: an earlier run put that volume on tmpfs and measured a
    disk-backed engine on a RAM disk.
    """
    try:
        ids = subprocess.run(["docker", "ps", "-q"], capture_output=True,
                             text=True, timeout=10)
        if ids.returncode != 0:
            return None
        for cid in ids.stdout.split():
            out = subprocess.run(
                ["docker", "inspect", "-f",
                 "{{.State.Pid}}|{{range .Mounts}}{{.Source}}>{{.Destination}} {{end}}",
                 cid],
                capture_output=True, text=True, timeout=10)
            if out.returncode != 0 or "|" not in out.stdout:
                continue
            cpid, _, mounts = out.stdout.strip().partition("|")
            if not cpid.strip().isdigit() or int(cpid) not in ancestors(pid):
                continue
            for m in mounts.split():
                src, _, dst = m.partition(">")
                if "storage" in dst or "qdrant" in dst:
                    return src
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def data_dir_of(pid: int) -> str | None:
    """The `--data-dir` the engine was actually started with, from its cmdline.

    strawmANN in `cached` or `cold` maps its arenas from a directory given on the
    command line, and nothing told the harness where — `--storage` and
    `$STORAGE_DIR` exist for it and `fullrun.py` sets neither for that arm, so the
    storage section read `unknown` on one side and `8.0 GiB` on the other. The
    engine knows: it is on its own cmdline, which this module already opens.

    Both spellings, because both work: `--data-dir /path` and `--data-dir=/path`.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().decode("utf-8", "replace").split("\0")
    except OSError:
        return None
    for i, a in enumerate(argv):
        if a == "--data-dir" and i + 1 < len(argv) and argv[i + 1]:
            return argv[i + 1]
        if a.startswith("--data-dir="):
            return a.split("=", 1)[1] or None
    return None


def storage_path_for(pid: int, comm: str, override: str | None) -> str | None:
    """Where this engine's bytes live, or None if that is not known.

    None is reported as "unknown" rather than as 0. An engine that stores
    nothing and an engine whose directory we failed to find both write zero
    into a table otherwise, and the difference between them is the entire
    subject of the comparison.
    """
    if override:
        return override
    env = os.environ.get("STORAGE_DIR")
    if env:
        return env
    if comm.startswith("qdrant"):
        return docker_storage_for(pid)
    # strawmANN's own `--data-dir`, which is the answer whenever it has one.
    return data_dir_of(pid)


def store_bytes_for(pid: int, comm: str, path: str | None) -> int | None:
    """Bytes on disk, distinguishing "nothing to measure" from "not measured".

    `storage_bytes` read `unknown` for strawmANN on every run, in the one row whose
    whole point is a RAM-resident engine against a disk-backed one. The gap was
    wrong: started with no `--data-dir` there is no store, and its size is 0.

    Proven from the engine's cmdline rather than assumed — an engine that *was*
    given a `--data-dir` the harness cannot locate is genuinely unmeasured and
    still reports None, because "could not look" and "nothing there" are different
    claims and only one is a result.
    """
    if path is not None:
        return dir_bytes(path)
    if comm.startswith("qdrant"):
        return None                      # containerised; the bind mount is the store
    # No path resolved. `data_dir_of` is the same parse `storage_path_for` just
    # made, so reaching here with a data dir on the command line means the flag
    # is there and its value is not usable — unmeasured, not zero. And a
    # command line that could not be read at all is "could not look", which is
    # not "nothing there" either.
    if _read(f"/proc/{pid}/cmdline") is None:
        return None
    return None if data_dir_of(pid) is not None else 0


@dataclass(frozen=True)
class Snapshot:
    """What is true of the engine at one instant.

    A dataclass rather than a dict because three consumers read these fields by
    name, and a mistyped key in a dict is a silently missing figure rather than
    an error. The JSON that reaches `rows.json` stays flat: it is an artifact
    with four readers and runs already on disk, so its shape is a contract even
    where the code's is not.
    """

    pid: int
    comm: str
    engines_running: int
    io: dict[str, int | None] | None
    io_source: IoSource
    rss_peak_bytes: int | None
    storage_path: str | None
    storage_bytes: int | None
    #: `sched_counters`, or None where /proc could not be read. Optional with a
    #: default so a `Snapshot(...)` built positionally in a test keeps working.
    sched: dict[str, float | None] | None = None
    #: `pressure_counters`, and the scope its numbers describe. Defaulted for
    #: the same reason as `sched`.
    psi: dict[str, float | None] | None = None
    psi_scope: str = ""
    #: `rss_split`: levels, like `rss_peak_bytes` beside them.
    rss: dict[str, int | None] | None = None

    @property
    def measured(self) -> bool:
        return self.io is not None


def snapshot(override: str | None = None) -> Snapshot | None:
    """Everything measurable about the engine right now.

    `None` when no engine is running, which is not an error: `workloads.py run`
    probes before the rows and says so ("engine: no strawmann or qdrant process
    found"), and a row measured against a dead server fails on its own with a
    better message than this could give.

    Said twice wrongly before: it claimed to return `{}`, which no caller would
    have survived — `delta` reads `.io` off it — and it cited a `--dry-run`
    flag that no driver in this repository implements.
    """
    procs = engine_processes()
    if not procs:
        return None
    pid, comm = procs[0]
    io, source = io_counters(pid)
    path = storage_path_for(pid, comm, override)
    psi, psi_scope = pressure_counters(pid)
    return Snapshot(
        pid=pid,
        comm=comm,
        engines_running=len(procs),
        io=io,
        io_source=source,
        rss_peak_bytes=rss_peak_bytes(pid),
        storage_path=path,
        storage_bytes=store_bytes_for(pid, comm, path),
        sched=sched_counters(pid),
        psi=psi,
        psi_scope=psi_scope,
        rss=rss_split(pid),
    )


def _sched_delta(before: Snapshot, after: Snapshot,
                 banked: dict[str, float] | None = None) -> dict:
    """Scheduler and fault counters consumed by the row between two snapshots.

    `threads` is the count at the end, not a difference: it is a level, and it
    is here so that a dropped thread-summed counter has a visible cause.
    """
    out: dict = dict.fromkeys(SCHED_FIELDS)
    a, b = before.sched, after.sched
    if not a or not b or before.pid != after.pid:
        return out
    out["threads"] = b.get("threads")
    # A thread-summed counter that went *backwards* is the coverage gate's
    # worst case -- so many threads exited that the survivors' sum fell -- and
    # it used to be the one case the gate did not fire on: the dropped delta
    # left `oncpu_s` None, coverage was never computed, and every other
    # thread-summed counter was published without a badge.
    fell = False
    for k in SCHED_FIELDS:
        if k in ("threads", "sched_coverage"):
            continue
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        d = y - x
        # Same rule the I/O fields use, and it fires here for a second reason:
        # a thread that exited during the row took its `/proc/<tid>` counters
        # with it, so a fallen sum is missing data rather than negative work.
        if d < 0:
            if k in _THREAD_SUMMED:
                fell = True
            continue
        out[k] = round(d, 6) if isinstance(d, float) else d

    # How much of the row's CPU the surviving threads can still account for.
    # `cpu_*_s` is the thread group's and survives an exit; `oncpu_s` is summed
    # over the threads still alive at the end. When they disagree, the
    # thread-summed counters describe a fraction of the row and there is no way
    # to tell which fraction, so they are withheld rather than printed small.
    # A sampler that ran through the row has already counted the threads that
    # exited during it, so its sums are the row's and the end-of-row
    # subtraction below would only lose them again. Coverage is then 1.0 by
    # construction and nothing is suppressed: the reason for suppressing was
    # that the survivors' sum described an unknown fraction of the row, and
    # this does not.
    # Coverage is computed the same way whether or not a sampler ran, because
    # it describes the *engine* rather than the instrument: what fraction of
    # the row's CPU the threads alive at the end still account for, and so how
    # much thread churn there was. Overwriting it with 1.0 on a sampled row —
    # which this did when the sampler landed — throws away the one signal that
    # says an engine retires threads mid-row, which is what made findings 41
    # visible in the first place.
    out["sched_sampled"] = 1.0 if banked else 0.0
    cpu_total = sum(v for k in ("cpu_user_s", "cpu_system_s")
                    if (v := out.get(k)) is not None)
    oncpu = out.get("oncpu_s")
    if cpu_total >= _COVERAGE_FLOOR_S and oncpu is not None:
        cov = oncpu / cpu_total
        out["sched_coverage"] = round(min(cov, 1.0), 4)

    if fell:
        out["sched_coverage"] = 0.0

    if banked:
        # The banked sums counted the threads that exited, so low coverage no
        # longer means the counters describe an unknown fraction of the row —
        # which was the only reason to withhold them. Only the counters the
        # sampler actually banked, though: one it did not read is still the
        # survivors' subtraction and is judged like an unsampled row.
        for k in _THREAD_SUMMED:
            if k in banked:
                out[k] = banked[k]
        low = fell or (cpu_total >= _COVERAGE_FLOOR_S and oncpu is not None
                       and oncpu / cpu_total < _COVERAGE_MIN)
        if low:
            for k in _THREAD_SUMMED:
                if k not in banked:
                    out[k] = None
        return out

    if fell or (cpu_total >= _COVERAGE_FLOOR_S and oncpu is not None
                and oncpu / cpu_total < _COVERAGE_MIN):
        for k in _THREAD_SUMMED:
            out[k] = None
    return out


def _psi_delta(before: Snapshot, after: Snapshot) -> dict:
    """Stall time the engine's cgroup accumulated between two snapshots.

    Withheld wholesale when the scope is not the engine's alone, and when the
    process changed underneath us: a restarted engine gets a new cgroup whose
    totals start from zero, and the difference against the old one is a large
    negative or a meaningless positive depending on which scope it landed in.
    """
    out: dict = dict.fromkeys(PSI_FIELDS)
    a, b = before.psi, after.psi
    scope = after.psi_scope or before.psi_scope
    if not a or not b or before.pid != after.pid or scope != "engine":
        return out
    for k in PSI_FIELDS:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        d = y - x
        if d >= 0:
            out[k] = round(d, 6)
    return out


def delta(before: Snapshot | None, after: Snapshot | None,
          banked: dict[str, float] | None = None) -> dict:
    """I/O performed between two snapshots, plus the end state of the rest.

    A row's I/O is a difference; its storage size and RSS are levels, and the
    level that matters is the one after the row ran.
    """
    # Returns the flat mapping a result row carries, which is the artifact's
    # shape rather than this module's.
    if before is None or after is None:
        return {"io_source": IoSource.none, "rss_peak_bytes": None,
                "storage_bytes": None, "storage_path": None, "psi_scope": "",
                **dict.fromkeys(SCHED_FIELDS), **dict.fromkeys(PSI_FIELDS),
                **dict.fromkeys(RSS_FIELDS)}
    out: dict = {
        "io_source": after.io_source or before.io_source,
        "rss_peak_bytes": after.rss_peak_bytes,
        "storage_bytes": after.storage_bytes,
        "storage_path": after.storage_path,
        # A level, taken after the row like the two above it.
        **(after.rss or dict.fromkeys(RSS_FIELDS)),
        "psi_scope": after.psi_scope or before.psi_scope,
        **_sched_delta(before, after, banked),
        **_psi_delta(before, after),
    }
    a, b = before.io, after.io
    if not a or not b:
        return out
    # A restarted engine resets its counters; a negative delta means the "after"
    # process is not the "before" one, so the row has no meaningful I/O figure.
    if before.pid != after.pid:
        return out
    for k in IO_FIELDS:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            out[k] = None
            continue
        d = y - x
        out[k] = d if d >= 0 else None
    return out


def human_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n == 0:
        return "0"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n} B"


def human_count(n: int | None) -> str:
    return "unknown" if n is None else f"{n:,}"


def _self_test() -> int:
    """Check the parsers and the delta rules without needing an engine running.

    The interesting cases are all failure cases: a counter that went backwards
    because the engine restarted, a snapshot with no I/O at all, and an
    `io.stat` spread over two devices. Each of them, handled wrongly, produces
    a plausible number rather than an error.
    """
    two_devices = ("252:0 rbytes=1000 wbytes=2000 rios=10 wios=20 dbytes=0 dios=0\n"
                   "259:0 rbytes=500 wbytes=250 rios=5 wios=3 dbytes=0 dios=0\n")
    got = parse_io_stat(two_devices)
    assert got == {"disk_read_ops": 15, "disk_write_ops": 23,
                   "disk_read_bytes": 1500, "disk_write_bytes": 2250}, got
    # Empty is zero: a freshly created scope lists no device until its first
    # block I/O completes, and reading that as "unmeasured" lost W0-upload's
    # ops column to `aggregate`'s "measured in 2 of 3 passes" refusal.
    assert parse_io_stat("") == {"disk_read_ops": 0, "disk_write_ops": 0,
                                 "disk_read_bytes": 0, "disk_write_bytes": 0}
    assert parse_io_stat("   \n\n") == {"disk_read_ops": 0, "disk_write_ops": 0,
                                        "disk_read_bytes": 0, "disk_write_bytes": 0}
    # Content this does not understand stays unmeasured: guessing zero for a
    # format change would turn a parser failure into a measurement.
    assert parse_io_stat("garbage\n") is None
    assert parse_io_stat("252:0 nothing=1\n") is None

    # A real one, when the host has it, so a kernel that changes the format
    # fails here rather than silently in a results table.
    root = Path("/sys/fs/cgroup/io.stat")
    if root.exists():
        live = parse_io_stat(root.read_text())
        assert live is not None and live["disk_read_bytes"] >= 0, live

    def io(**kw):
        base = dict.fromkeys(IO_FIELDS)
        base.update(kw)
        return base

    def snap(pid, fields, rss=None, storage=None):
        return Snapshot(pid=pid, comm="engine", engines_running=1, io=fields,
                        io_source=IoSource.both, rss_peak_bytes=rss,
                        storage_path=None, storage_bytes=storage)

    before = snap(1, io(syscall_reads=5, disk_read_bytes=100, disk_read_ops=1))
    after = snap(1, io(syscall_reads=9, disk_read_bytes=300, disk_read_ops=4),
                 rss=42, storage=7)
    d = delta(before, after)
    assert d["syscall_reads"] == 4 and d["disk_read_bytes"] == 200, d
    assert d["disk_read_ops"] == 3 and d["rss_peak_bytes"] == 42, d
    # A field neither snapshot could measure stays None rather than becoming 0.
    assert d["disk_write_ops"] is None, d

    # A restarted engine: same fields, different process, so the difference
    # between the two is meaningless and must not be reported as zero.
    restarted = delta(before, snap(2, after.io, rss=42, storage=7))
    assert "syscall_reads" not in restarted, restarted

    # A counter that went backwards without the pid changing (wraparound, or a
    # cgroup that was reset) is not a negative amount of work.
    backwards = delta(before, snap(1, io(syscall_reads=1, disk_read_bytes=1,
                                         disk_read_ops=0)))
    assert backwards["syscall_reads"] is None, backwards

    nothing = delta(None, None)
    assert nothing["io_source"] == IoSource.none, nothing
    assert all(nothing[k] is None for k in
               ("rss_peak_bytes", "storage_bytes", "storage_path")), nothing
    # The scheduler fields are present and null rather than absent: a row that
    # could not measure them must still carry the columns, or `rows.json` grows
    # a second shape and every reader has to handle both.
    assert set(SCHED_FIELDS) <= nothing.keys(), nothing
    assert all(nothing[k] is None for k in SCHED_FIELDS), nothing
    # Same contract for the two families added beside them.
    assert set(PSI_FIELDS) <= nothing.keys(), nothing
    assert set(RSS_FIELDS) <= nothing.keys(), nothing
    assert all(nothing[k] is None for k in (*PSI_FIELDS, *RSS_FIELDS)), nothing
    assert nothing["psi_scope"] == "", nothing

    # `/proc/<pid>/stat` field offsets. `comm` is arbitrary and may hold spaces
    # and brackets, and the fields are counted from the last `)`; getting that
    # arithmetic wrong reads `session` as minflt and `tpgid` as majflt, which
    # is silent for one and yields -1 for the other.
    line = ("1234 (straw mann) R 1 1234 999 0 -1 4194560 "  # ... tpgid = -1
            "8888 7 77 9 " + f"{3 * _CLK_TCK} {2 * _CLK_TCK} " + "0 0 20 0 6 0 0")
    got = parse_proc_stat(line)
    assert got == {"minor_faults": 8888.0, "major_faults": 77.0,
                   "cpu_user_s": 3.0, "cpu_system_s": 2.0}, got
    assert parse_proc_stat("no parenthesis here") is None
    assert parse_schedstat("1331764 55 3") == {"oncpu_s": 1331764 / 1e9,
                                               "runqueue_wait_s": 55 / 1e9,
                                               "timeslices": 3.0}
    assert parse_schedstat("") is None

    # `SchedSampler` against a fake /proc: the case the end-of-row sum cannot
    # see. Two threads run the whole row; a third does 4 s of work and exits
    # before the end. A sum over the survivors misses that 4 s entirely, which
    # is how W3 came to read `sched_coverage` 0.106 on an engine that retires
    # pool threads (findings 41).
    live = {"1": 0.0, "2": 0.0}
    frames = [
        {"1": 1.0, "2": 1.0, "3": 1.0},   # baseline: three threads alive
        {"1": 2.0, "2": 2.0, "3": 5.0},   # thread 3 does its work
        {"1": 3.0, "2": 3.0},             # ...and is gone
    ]
    seq = iter(frames)
    current = {"frame": next(seq)}

    def fake_counters(_pid):
        return {tid: {"oncpu_s": v, "migrations": 0.0}
                for tid, v in current["frame"].items()}

    real_counters = thread_counters
    globals()["thread_counters"] = fake_counters
    try:
        smp = SchedSampler(1, interval_s=0.0)
        smp._poll()                      # baseline frame
        smp._started = True
        current["frame"] = next(seq)
        smp._poll()
        current["frame"] = next(seq)
        smp._poll()
        banked = smp.stop()
    finally:
        globals()["thread_counters"] = real_counters

    # 1 and 2 each moved 1.0 -> 3.0; thread 3 moved 1.0 -> 5.0 and exited. The
    # banked total keeps all three: 2 + 2 + 4.
    assert abs(banked["oncpu_s"] - 8.0) < 1e-9, banked
    # What the old end-of-row sum would have said, for contrast: the survivors
    # only, and thread 3's work simply absent.
    assert abs((3.0 - 1.0) * 2 - 4.0) < 1e-9

    # A thread created after the row began contributes all of its counter,
    # because the kernel started it at zero rather than at the baseline.
    seq2 = iter([{"1": 10.0}, {"1": 11.0, "9": 3.0}])
    current2 = {"frame": next(seq2)}
    globals()["thread_counters"] = lambda _pid: {
        tid: {"oncpu_s": v} for tid, v in current2["frame"].items()}
    try:
        smp2 = SchedSampler(1, interval_s=0.0)
        smp2._poll()
        smp2._started = True
        current2["frame"] = next(seq2)
        smp2._poll()
        banked2 = smp2.stop()
    finally:
        globals()["thread_counters"] = real_counters
    # tid 1: 11 - 10 = 1. tid 9: all 3.0, not 3.0 - 3.0 = 0.
    assert abs(banked2["oncpu_s"] - 4.0) < 1e-9, banked2

    # The sampler reports what it cost, so the interval can be argued about
    # with a number rather than asserted to be free.
    assert smp.polls >= 3 and smp.overhead_s >= 0.0

    # Threads that exit mid-row take their per-thread counters with them, so a
    # sum over the survivors can be a small fraction of the row's real work
    # while the thread *count* is unchanged either side. Suppressed, not
    # printed small: an index build measured 13 s of CPU and 0.07 s on-cpu.
    def sched(cpu, oncpu, threads=6.0):
        return {"cpu_user_s": cpu, "cpu_system_s": 0.0, "minor_faults": 0.0,
                "major_faults": 0.0, "oncpu_s": oncpu, "runqueue_wait_s": 0.0,
                "timeslices": 10.0, "ctx_switches_voluntary": 10.0,
                "ctx_switches_involuntary": 1.0, "migrations": 2.0,
                "threads": threads}

    def snap_sched(before_cpu, after_cpu, before_on, after_on):
        b = dataclasses.replace(snap(1, io()), sched=sched(before_cpu, before_on))
        a = dataclasses.replace(snap(1, io()), sched=sched(after_cpu, after_on))
        return delta(b, a)

    lost = snap_sched(0.0, 13.0, 0.0, 0.07)
    assert lost["cpu_user_s"] == 13.0, lost           # thread-group total survives
    assert lost["minor_faults"] == 0.0, lost
    assert lost["sched_coverage"] == 0.0054, lost
    assert all(lost[k] is None for k in _THREAD_SUMMED), lost

    kept = snap_sched(0.0, 4.0, 0.0, 4.0)
    assert kept["sched_coverage"] == 1.0, kept
    assert kept["oncpu_s"] == 4.0 and kept["migrations"] == 0.0, kept

    # Below the CPU floor the ratio is noise (CPU time is quantised to ticks),
    # so no coverage is claimed and nothing is suppressed on its say-so.
    idle = snap_sched(0.0, 0.0, 0.0, 0.0009)
    assert idle["sched_coverage"] is None, idle
    assert idle["oncpu_s"] == 0.0009, idle

    # And the banked sums reach the row: with a sampler, the thread-summed
    # fields are the sampler's and coverage is 1.0 by construction, because the
    # reason for suppressing them -- a survivors-only sum covering an unknown
    # fraction of the row -- no longer applies.
    lost_pair = snap_sched(0.0, 13.0, 0.0, 0.07)
    assert lost_pair["sched_coverage"] == 0.0054, lost_pair
    assert lost_pair["runqueue_wait_s"] is None
    b = dataclasses.replace(snap(1, io()), sched=sched(0.0, 0.0))
    a = dataclasses.replace(snap(1, io()), sched=sched(13.0, 0.07))
    banked_row = delta(b, a, {"oncpu_s": 12.9, "runqueue_wait_s": 4.2,
                              "migrations": 900.0})
    # Coverage still reports the engine's thread churn -- 0.0054 here, the same
    # number the unsampled row gives -- because it describes the engine and not
    # the instrument. Overwriting it with 1.0 would discard the only signal
    # that says threads are exiting mid-row, which is what findings 41 is about.
    assert banked_row["sched_coverage"] == 0.0054, banked_row
    assert banked_row["runqueue_wait_s"] == 4.2, banked_row
    assert banked_row["migrations"] == 900.0, banked_row
    # The thread-group totals are untouched by the sampler: they were never
    # the broken half.
    assert banked_row["cpu_user_s"] == 13.0, banked_row
    # Which of the two ways coverage reached 1.0. A row whose threads all
    # survived also reads 1.0, and the two are different claims: one says
    # nothing exited, the other says what exited was counted anyway.
    assert banked_row["sched_sampled"] == 1.0, banked_row
    assert kept["sched_sampled"] == 0.0, kept
    assert kept["sched_coverage"] == 1.0, kept
    # The pair that matters to a reader: same low coverage, and the counters
    # present in one and withheld in the other. `sched_sampled` is what says
    # which, and without it the two rows look like one engine behaving
    # differently rather than one instrument being present.
    assert lost_pair["sched_coverage"] == banked_row["sched_coverage"]
    assert lost_pair["runqueue_wait_s"] is None
    assert banked_row["runqueue_wait_s"] == 4.2

    # `delayacct_blkio_ticks` is proc(5)'s field 42, twenty-nine fields past the
    # last one this parser used to want. Counted from the wrong end it reads
    # `rt_priority` or `policy`, both of which are small integers that look
    # exactly like a plausible number of ticks blocked on a disk.
    full = ("1234 (straw mann) R 1 1234 999 0 -1 4194560 "
            "8888 7 77 9 " + f"{3 * _CLK_TCK} {2 * _CLK_TCK} " +
            "0 0 20 0 6 0 0 " +                     # fields 16-22
            " ".join(["0"] * 18) + " " +            # 23-40
            f"0 {5 * _CLK_TCK} " +                  # 41 (policy), 42 (blkio)
            "0 0 0")
    got = parse_proc_stat(full)
    assert got["blkio_delay_s"] == 5.0, got
    assert got["cpu_user_s"] == 3.0 and got["major_faults"] == 77.0, got
    # A short line (a kernel that reports fewer fields) leaves the key absent
    # rather than guessing from whatever the last field happens to be.
    assert "blkio_delay_s" not in parse_proc_stat(line), parse_proc_stat(line)

    # PSI: only `total=` is differenceable. The decaying averages beside it
    # describe the last ten seconds of a forty-second row.
    psi_text = ("some avg10=0.02 avg60=0.10 avg300=0.08 total=887452708\n"
                "full avg10=0.00 avg60=0.05 avg300=0.06 total=850102709\n")
    assert parse_pressure(psi_text) == {"some": 887.452708, "full": 850.102709}
    # `cpu.pressure` on an older kernel has no `full` line at all.
    assert parse_pressure("some avg10=0.00 total=17\n") == {"some": 1.7e-05}
    assert parse_pressure("") is None
    assert parse_pressure("garbage\n") is None

    def psi_snap(pid, io_secs, scope="engine"):
        return dataclasses.replace(
            snap(pid, io()),
            psi={**dict.fromkeys(PSI_FIELDS), "psi_io_some_s": io_secs},
            psi_scope=scope,
            rss={"rss_anon_bytes": 1024, "rss_file_bytes": 512,
                 "rss_shmem_bytes": 0})

    stalled = delta(psi_snap(1, 10.0), psi_snap(1, 12.5))
    assert stalled["psi_io_some_s"] == 2.5, stalled
    assert stalled["psi_scope"] == "engine", stalled
    # A level, taken after the row, not a difference.
    assert stalled["rss_anon_bytes"] == 1024, stalled
    assert stalled["psi_cpu_some_s"] is None, stalled

    # The cgroup holds the harness and the editor as well as the engine, so
    # differencing its stall time measures the desktop. Refused, and the row
    # says which kind of blank it is.
    shared = delta(psi_snap(1, 10.0, "shared"), psi_snap(1, 12.5, "shared"))
    assert shared["psi_io_some_s"] is None, shared
    assert shared["psi_scope"] == "shared", shared

    # A restarted engine lands in a new cgroup whose totals start at zero.
    moved = delta(psi_snap(1, 10.0), psi_snap(2, 0.5))
    assert moved["psi_io_some_s"] is None, moved

    # `--data-dir` off the engine's own command line, both spellings. The
    # storage section's whole subject is a RAM-resident engine against a
    # disk-backed one, and it read `unknown` on the strawmANN side of every
    # dbpedia-openai-1m row because the harness was never told a path the
    # engine had been started with.
    def parsed(argv):
        import io as _io
        import unittest.mock as _mock
        blob = "\0".join(argv).encode()
        with _mock.patch("builtins.open", lambda *a, **k: _io.BytesIO(blob)):
            return data_dir_of(1)

    assert parsed(["strawmann", "--data-dir", "/srv/x", "--port", "1"]) == "/srv/x"
    assert parsed(["strawmann", "--data-dir=/srv/y"]) == "/srv/y"
    # A flag with nothing after it is not a path, and must not become one.
    assert parsed(["strawmann", "--data-dir"]) is None
    assert parsed(["strawmann", "--data-dir="]) is None
    assert parsed(["strawmann", "--port", "6334"]) is None

    # Whose cgroup it is. Four shapes, and the one that matters is the third:
    # a container whose entrypoint is a script leaves that script alive as the
    # engine's parent, and rejecting it refused pressure for the containerised
    # engine — the arm this measurement exists to describe.
    tree = {1: {1}, 2: {2, 1}, 3: {3, 2, 1}, 9: {9, 1}}   # 1 -> 2 -> 3, and 1 -> 9

    class FakeDir:
        def __init__(self, pids): self.pids = pids
        def __truediv__(self, _name): return self
        def read_text(self): return " ".join(str(p) for p in self.pids)

    def holds(engine, pids):
        return cgroup_holds_only(engine, FakeDir(pids), ancestors_of=tree.get)

    assert holds(2, [2]) is True                  # the engine alone
    assert holds(2, [2, 3]) is True               # a worker it forked
    assert holds(2, [1, 2]) is True               # the wrapper that started it
    assert holds(2, [1, 2, 9]) is False           # a sibling: foreign
    assert holds(2, []) is False                  # an empty file says nothing

    assert human_bytes(None) == "unknown"
    assert human_bytes(0) == "0"
    assert human_bytes(2048) == "2.0 KiB"
    assert human_count(None) == "unknown"
    assert human_count(1234) == "1,234"

    print("ok: procstat parsers and delta rules")
    return 0


if __name__ == "__main__":
    import json
    import sys

    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    snap = snapshot()
    print(json.dumps(dataclasses.asdict(snap) if snap else None, indent=2))
