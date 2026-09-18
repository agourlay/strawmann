#!/usr/bin/env python3
"""§7.1 host discipline: verify the machine is fit to produce published numbers.

    bench/setup.py check      verify and describe (exit 1 on any failure)
    bench/setup.py describe   print the environment block only
    bench/setup.py apply      set what is settable (needs root)
    bench/setup.py check --lax   downgrade failures to warnings

§7.1 requires every result row carry a hash of the environment description, and
that rows from different environments never share a chart. That hash is computed
here from the same text this prints, so the description and the hash cannot
disagree.

Ported from `setup.sh`. Reading `/sys` and `/proc` is what shell is good at, but
the file had grown to 351 lines of accumulating a description string, comparing
values, and deriving a hash, which shell does grudgingly.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path

RED, GREEN, YELLOW, OFF = "\033[31m", "\033[32m", "\033[33m", "\033[0m"

#: The two experiments this harness can run, decided by `check_isolation` and
#: carried into every result row. They are not ranks: `isolated` attributes a
#: difference to the engine, `as-deployed` says whether a user would see it,
#: and a number from one is not evidence about the other.
class Profile(StrEnum):
    """Which experiment this host is set up to run.

    Not a verdict: both pass, and §7.1's environment hash keeps them apart, so
    an `isolated` row and an `as-deployed` one never share a chart. A `StrEnum`
    for the same reason `workloads.Status` and `Gate` are — the two words are
    compared and printed in several places, and `Result.profile` carries one
    of them as a bare string because this file, which decides it, had them as
    two loose constants.

    This module is deliberately stdlib-only so it runs on a fresh clone, so the
    harness parses this word out of the output below rather than importing it.
    """

    isolated = "isolated"
    as_deployed = "as-deployed"


class Env:
    """Accumulates the description §7.1 hashes, and the pass/fail tally."""

    def __init__(self, lax: bool) -> None:
        self.lax = lax
        self.lines: list[str] = []
        self.observed: list[str] = []
        self.failures = 0
        #: Which experiment this host is set up to run, from `check_isolation`.
        #: Not a verdict: both profiles pass, and the hash keeps them apart.
        self.profile = Profile.as_deployed

    def note(self, text: str) -> None:
        """Record a line that is part of the environment's identity."""
        self.lines.append(text)

    def observe(self, text: str) -> None:
        """Record a line that is reported but NOT hashed.

        For anything transient. Ambient load used to go through `note`, which
        put a float like `load1=1.53` into the hash, so every run produced a
        different environment hash and §7.1's "rows with different hashes never
        share a chart" became impossible to satisfy: the warning fired on every
        comparison and meant nothing.

        The hash answers "is this the same machine, configured the same way?",
        which is what decides comparability. Whether something else was running
        during a particular row is a property of the row, not of the
        environment, and it is already recorded per row as `load_start`,
        `load_end` and `foreign`.
        """
        self.observed.append(text)

    def ok(self, msg: str, *hint: str) -> None:
        print(f"  {GREEN}ok{OFF}   {msg}", file=sys.stderr)
        for h in hint:
            print(f"       {h}", file=sys.stderr)

    def fail(self, msg: str, *hint: str) -> None:
        if self.lax:
            print(f"  {YELLOW}warn{OFF} {msg}", file=sys.stderr)
        else:
            print(f"  {RED}FAIL{OFF} {msg}", file=sys.stderr)
            self.failures += 1
        for h in hint:
            print(f"       {h}", file=sys.stderr)

    def hash(self) -> str:
        body = "\n".join(self.lines) + "\n"
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def glob_read(pattern: str) -> list[str]:
    out = []
    for p in sorted(Path("/").glob(pattern.lstrip("/"))):
        v = read(str(p))
        if v is not None:
            out.append(v)
    return out


def cpu_list(v: str) -> list[int]:
    """A kernel CPU list — `0,12`, `4-7`, `0-3,8` — as numbers.

    Both spellings occur in the same tree, and for the same file: `/sys` uses
    the range form once a group is contiguous, so a sibling pair reads `0,12`
    on one machine and `0-1` on another.
    """
    out: list[int] = []
    for part in v.split(","):
        lo, sep, hi = part.strip().partition("-")
        if sep and lo.isdigit() and hi.isdigit():
            out.extend(range(int(lo), int(hi) + 1))
        elif lo.isdigit():
            out.append(int(lo))
    return out


def smt_topology(lists: list[str]) -> str:
    """Which CPUs share a core, in one line, from `thread_siblings_list`.

    Printed and deliberately NOT hashed. It is a function of the CPU model and
    of whether SMT is on, both already in the hash body, so adding it would
    give every host a new environment hash for no new information — and §7.1's
    "rows from different environments never share a chart" would then fire on
    two runs of the same machine.

    The operator needs it because `fullrun.py` takes `--server-cpus` as raw
    CPU numbers. With SMT off the offline siblings make any set unambiguous;
    with SMT on, whether `4-11` is eight cores or four cores twice over is a
    property of the kernel's numbering, which differs between machines.
    """
    groups: list[list[int]] = []
    for v in lists:
        cpus = cpu_list(v)
        if len(cpus) > 1 and cpus not in groups:
            groups.append(cpus)
    groups.sort()
    if not groups:
        return ""
    strides = {g[1] - g[0] for g in groups}
    if all(len(g) == 2 for g in groups) and len(strides) == 1:
        return (f"cpu N and cpu N+{strides.pop()} share a core "
                f"({len(groups)} pairs)")
    shown = " ".join(",".join(str(c) for c in g) for g in groups[:8])
    more = f" (+{len(groups) - 8} more)" if len(groups) > 8 else ""
    return f"cpus sharing a core: {shown}{more}"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_uarch(e: Env) -> None:
    """§7.1: microarchitecture by name, not by core count.

    'Cascade Lake' and 'Zen 4' behave differently enough that a core count tells
    a reader nothing about whether two rows are comparable.
    """
    model = "unknown"
    for line in (read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    cores = os.cpu_count() or 0
    e.note(f"cpu_model={model}")
    e.note(f"cpu_cores={cores}")
    e.ok(f"{model} ({cores} logical)")


def check_governor(e: Env) -> None:
    govs = set(glob_read("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"))
    val = ",".join(sorted(govs)) or "unknown"
    e.note(f"governor={val}")
    if govs == {"performance"}:
        e.ok(f"governor={val}")
    else:
        e.fail(f"governor={val}, expected performance",
               "a scaling governor makes throughput depend on how warm the run is")


def check_turbo(e: Env) -> None:
    """Boost has to be off, or the clock depends on how many cores are busy.

    That turns a comparison between two engines into a comparison of how well
    each happens to spread its work.
    """
    state = "unknown"
    no_turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    boost = read("/sys/devices/system/cpu/cpufreq/boost")
    if no_turbo is not None:
        state = "disabled" if no_turbo == "1" else "enabled"
    elif boost is not None:
        state = "disabled" if boost == "0" else "enabled"
    e.note(f"boost={state}")
    if state == "disabled":
        e.ok("boost disabled")
    else:
        e.fail(f"boost={state}, expected disabled")


def check_smt(e: Env) -> None:
    """SMT declared, not required off.

    §7.1 asks for "SMT explicitly on or off, declared per run, never left
    ambient", which is a declaration and not a setting. This checked
    `smt=off` and failed anything else, stricter than the spec it cites, and
    the cost was paid on every run: a host that ships with SMT on had to have
    half its threads offlined before the gate would let it measure, and put
    back afterwards to be a laptop again.

    So `smt=` stays in the hash body and stops being a verdict, the same move
    `check_isolation` made: an SMT-on row and an SMT-off row have different
    environment hashes and never share a chart.

    What that does *not* buy is symmetry between two arms of one comparison.
    Two hyperthreads share a core's execution ports and its L1/L2, and while
    both engines are pinned to the same `--server-cpus` set, they size their
    own thread pools: an engine running more threads than that set has CPUs
    meets sibling contention a leaner one never sees, and that lands in the
    ratio rather than in both arms equally. SMT on is a declared and accepted
    risk to a comparison, not one this gate measures away, which is why the
    hint names which CPUs share a core: keep both members of a pair out of
    `--server-cpus`, and the engine runs on the cores the flag appears to
    give it.
    """
    smt = read("/sys/devices/system/cpu/smt/control") or "unknown"
    e.note(f"smt={smt}")
    if smt in {"off", "notsupported"}:
        e.ok(f"smt={smt}")
        return
    hints = ["declared rather than required (§7.1 asks for SMT on or off, not",
             "for off), and hashed, so these rows never share a chart with",
             "SMT-off ones"]
    topo = smt_topology(glob_read(
        "/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"))
    if topo:
        hints += [f"{topo};",
                  "a --server-cpus or --client-cpus set holding both cpus of a",
                  "pair measures the engine on half the cores it names"]
    e.ok(f"smt={smt}", *hints)


def check_isolation(e: Env) -> None:
    """Which measurement profile this host is in, rather than pass or fail.

    §7.1 originally required `isolcpus` and `nohz_full` on the benchmark cores,
    and refused to run without them. That is the right rule for attributing a
    change to the engine, and the wrong one for describing what a user gets:
    nobody deploys with the scheduler amputated, so a hermetic run answers
    "is this engine faster" and not "will anyone notice".

    Both are legitimate, and they are different experiments, so the isolation
    state now *names* the experiment instead of gating it:

      `isolated`     benchmark cores taken from the scheduler and the tick.
                     Attribution: a difference is the engine's.
      `as-deployed`  the scheduler running as it ships. External validity: a
                     difference is what a user would see, and the ambient load
                     is part of the measurement rather than an error in it.

    Neither is publishable *as the other*, which is what the environment hash
    already enforces: `isolcpus=` and `nohz_full=` are hashed, so the two
    profiles cannot share a chart, by the same mechanism that keeps two
    machines apart. The profile is named in the hash body as well, so a reader
    of `describe` sees the word rather than having to infer it.

    What is emphatically still a gate: quiescence, and the per-row foreign-load
    flags. An as-deployed run tolerates the load a user's machine has; it does
    not tolerate *this* run's arms getting different amounts of it, because
    that lands in the ratio and nothing downstream can separate it out.
    """
    cmdline = read("/proc/cmdline") or ""
    iso = re.search(r"isolcpus=(\S+)", cmdline)
    nohz = re.search(r"nohz_full=(\S+)", cmdline)
    iso_v = iso.group(1) if iso else "none"
    nohz_v = nohz.group(1) if nohz else "none"
    e.note(f"isolcpus={iso_v}")
    e.note(f"nohz_full={nohz_v}")
    e.profile = Profile.isolated if (iso and nohz) else Profile.as_deployed
    e.note(f"profile={e.profile}")
    if e.profile == Profile.isolated:
        e.ok(f"profile=isolated (isolcpus={iso_v} nohz_full={nohz_v})")
    else:
        e.ok(f"profile=as-deployed (isolcpus={iso_v} nohz_full={nohz_v})",
             "the scheduler is running as it ships, so these numbers describe a",
             "deployment rather than the engine in isolation; they may not be",
             "compared against an `isolated` run, and the environment hash",
             "enforces that")


def check_memory(e: Env) -> None:
    thp = read("/sys/kernel/mm/transparent_hugepage/enabled") or ""
    m = re.search(r"\[(\w+)\]", thp)
    thp_v = m.group(1) if m else "unknown"
    nodes = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
    balancing = read("/proc/sys/kernel/numa_balancing") or "unknown"
    e.note(f"thp={thp_v}")
    e.note(f"numa_nodes={nodes}")
    e.note(f"numa_balancing={balancing}")
    e.ok(f"thp={thp_v} numa_nodes={nodes} numa_balancing={balancing}")
    if balancing == "1":
        e.fail("numa_balancing=1", "page migration during a run moves the working set")


def check_perf(e: Env) -> None:
    """§6.6.4 reads cycles and ref-cycles via perf_event_open."""
    paranoid = read("/proc/sys/kernel/perf_event_paranoid") or "unknown"
    e.note(f"perf_event_paranoid={paranoid}")
    try:
        ok = int(paranoid) <= 0
    except ValueError:
        ok = False
    (e.ok if ok else e.fail)(f"perf_event_paranoid={paranoid}"
                             + ("" if ok else ", need <= 0 for cycle counts"))


#: cgroup v2's mount point, the same one `procstat` reads `io.stat` from.
CGROUP_ROOT = "/sys/fs/cgroup"


def check_instruments(e: Env) -> None:
    """What the harness will be able to *measure* about a row, beyond its rate.

    Observed rather than noted, which is the deliberate part. These two
    settings decide what `bench/harness/perfstat.py` and `procstat.py` can
    read; they do not change how fast the engine runs, so they are reported
    without entering the environment hash. A hashed line here would make every
    future run incomparable with every past one — §7.1's "rows with different
    hashes never share a chart" — in exchange for recording the state of an
    instrument, and the instrument's state is already recorded on the rows it
    produced (`perf_set`, `perf_note`, and a blank `blkio_delay_s`).

    None of them is a failure. A host without them measures fewer things and
    measures them honestly; a host with them measures more.
    """
    delayacct = read("/proc/sys/kernel/task_delayacct")
    e.observe(f"task_delayacct={delayacct or 'absent'}")
    if delayacct == "1":
        e.ok("task_delayacct=1, so time blocked on a device is measurable")
    else:
        e.ok(f"task_delayacct={delayacct or 'absent'}",
             "delayacct_blkio_ticks will read a permanent zero, which is",
             "indistinguishable from an engine that never waited for a disk;",
             "the harness reports it as unmeasured. `setup.py apply` sets it.")

    perf = shutil.which("perf")
    e.observe(f"perf={'present' if perf else 'absent'}")
    if perf:
        e.ok(f"perf at {perf}, so `workloads.py run --perf` can attach")
    else:
        e.ok("perf not installed",
             "hardware counters per row are unavailable; everything else runs")

    # Block-layer *operations* come only from cgroup v2 `io.stat`;
    # `/proc/<pid>/io` has bytes and syscalls but no request counts
    # (`procstat.IoSource`). The engines run under `systemd-run --user
    # --scope`, so their cgroup is under `user.slice`, and the `io` controller
    # is not delegated there by default — which is why every report so far has
    # printed `disk read ops: unknown` beside real byte counts.
    available = (read(f"{CGROUP_ROOT}/cgroup.controllers") or "").split()
    missing = io_missing() if "io" in available else []
    e.observe(f"cgroup_io_delegated={'io' in available and not missing}")
    if "io" not in available:
        e.ok("no cgroup io controller on this kernel",
             "block-layer operation counts are unavailable; bytes still are")
    elif not missing:
        e.ok("cgroup io controller reaches the engines' own scope, so "
             "block-layer read/write operations are measurable")
    else:
        # Named per level, because the level that was missing is the whole
        # bug: this check used to read `user.slice` alone and say "measurable"
        # while `app.slice` -- the transient scope's actual parent -- passed
        # down only `memory pids`, so every report printed `disk read ops:
        # unknown` under a gate that had called them measurable.
        e.ok("cgroup io controller present but not enabled at "
             + ", ".join(d.name for d in missing),
             "the storage table reports `disk read ops` / `disk write ops` as",
             "unknown: /proc/<pid>/io has bytes and syscalls but no request",
             "counts, and a cgroup has `io.stat` only where its parent enabled",
             "`io`. `setup.py apply` enables it at every level.")


def io_levels(root: str | None = None) -> list[Path]:
    """Every cgroup level whose `subtree_control` must carry `io` before the
    engines' own scope has an `io.stat`.

    A cgroup gets `io.stat` when its *parent* enables `io`, and `systemd-run
    --user --scope` puts the engine at
    `user.slice/user-UID.slice/user@UID.service/app.slice/<unit>.scope`. So the
    chain runs one level deeper than `user@UID.service`, which is where both the
    check and `delegate_cgroup_io` used to stop -- and stopping there is why the
    gate could report the controller "delegated to user.slice, so block-layer
    read/write operations are measurable" on a host where `app.slice` passed
    down `memory pids` and every report still printed `disk read ops: unknown`.

    Takes `root` so the shape can be tested against a fake tree, the way
    `procstat.cgroup_holds_only` takes `ancestors_of`.
    """
    base = Path(root or f"{CGROUP_ROOT}/user.slice")
    levels = [base]
    for uid_dir in sorted(base.glob("user-*.slice")):
        levels.append(uid_dir)
        for mgr in sorted(uid_dir.glob("user@*.service")):
            levels.append(mgr)
            # Whichever slice the user manager puts transient units in:
            # `app.slice` for `systemd-run --user --scope`, and the others
            # because a scope started from a login session can land beside it.
            levels += sorted(mgr.glob("*.slice"))
    return [d for d in levels if (d / "cgroup.subtree_control").exists()]


def io_missing(root: str | None = None) -> list[Path]:
    """The levels of `io_levels` that do not pass `io` down to their children."""
    return [d for d in io_levels(root)
            if "io" not in (read(str(d / "cgroup.subtree_control")) or "").split()]


def check_toolchain(e: Env) -> None:
    """§9 pins the compiler; an unpinned one turns the ISA matrix into noise."""
    try:
        ver = subprocess.run(["zig", "version"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception:
        ver = "missing"
    e.note(f"zig={ver}")
    (e.ok if ver.startswith("0.16") else e.fail)(f"zig={ver}"
                                                 + ("" if ver.startswith("0.16") else ", expected 0.16.x"))


#: Processes that are supposed to be running during a measurement.
#: Our own processes, matched by prefix. Prefix and not equality because Linux
#: truncates `comm` to 15 characters, so the ISA arm binaries appear as
#: `strawmann-avx5` and `strawmann-base`, none of which equal "strawmann". With
#: exact matching every ISA sweep row was flagged as contaminated by its own
#: engine, which does not merely add noise: a warning that fires on every row is
#: a warning nobody reads, so it destroys the signal it exists to carry.
#: No `ps`: the sampler reads `/proc` and never shells out, and the prefix
#: match made `psql`, `pserve` and `psi-notify` invisible to the gate.
_OURS = ("bfb", "strawmann", "qdrant")

#: An interpreter is ours only when it is running one of the harness's own
#: scripts. `python3` and `uv` used to be exempt by name, so a foreign
#: `python3 train.py` at 800% CPU was invisible to the very check that exists
#: to see it. Matched against the full command line (`ps -o args`).
_OUR_SCRIPTS = ("bench/harness/", "bench/setup.py", "scripts/check.py", "scripts/doctor.py")

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
    if name.startswith(("python", "uv")):
        return (any(p in cmdline for p in _OUR_SCRIPTS)
                or any(os.path.basename(a) in _OUR_SCRIPT_NAMES for a in argv[1:]))
    return False



#: How long the quiescence check watches the machine for.
QUIESCENT_SAMPLE_S = 1.0

#: Percent of one core above which a foreign process is worth *naming*. Not a
#: verdict on its own — see `FOREIGN_BUDGET_CORES`. Low, because the point of
#: the list is to tell the operator what to close.
FOREIGN_PCT = 5.0

#: Total foreign CPU, in whole cores, above which the machine is not quiescent.
#:
#: The verdict was `FOREIGN_PCT` applied per process: any single foreign
#: process over 20% of *one* core failed the gate. Two things were wrong with
#: that, and they point in opposite directions.
#:
#: It is far too strict. `docs/decisions.md` records what contamination
#: actually costs, measured by re-running flagged rows: 18 foreign cores =
#: +41%, 3 cores = +11%, **1 core = noise**. The gate fired at 0.2 cores, five
#: times below the smallest load this project has measured as having any
#: effect, and on a twelve-core host that is 1.7% of the machine. A threshold
#: that refuses runs over an amount of load the project has measured as
#: indistinguishable from noise is not protecting the measurement, it is
#: costing runs — three of them in one night.
#:
#: And it is too lax in the other direction, because it judges each process
#: alone. Ten processes at 15% of a core apiece are 1.5 cores of foreign load,
#: past the point where the calibration says the cost is real, and not one of
#: them crosses a per-process threshold. `check_quiescent`'s own comment
#: admitted the gap, leaving `load1` to catch "load made of many small
#: processes that no single entry catches" — an instrument it also says cannot
#: tell "quiet now" from "busy a moment ago".
#:
#: So: sum the foreign time and compare it to a budget denominated in cores.
#: One core is the largest load measured to sit inside run-to-run noise, which
#: makes it the last value that can be defended as harmless rather than the
#: first that is obviously harmful.
FOREIGN_BUDGET_CORES = float(os.environ.get("FOREIGN_BUDGET_CORES", 1.0))

#: One-minute load average, as a percentage of one core, above which the gate
#: refuses a host that names no busy process: load made of many small ones.
LOAD_PER_CORE_MAX_PCT = 10.0


def _cpu_seconds() -> dict[int, tuple[float, str]]:
    """user+system CPU seconds and command line per pid, from `/proc`.

    Kernel threads (no command line) are left out: their time is the kernel's
    on someone's behalf, not a foreign process's.
    """
    tck = os.sysconf("SC_CLK_TCK")
    out: dict[int, tuple[float, str]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            st = Path(f"/proc/{name}/stat").read_text()
            cmd = (Path(f"/proc/{name}/cmdline").read_bytes().replace(b"\0", b" ")
                   .decode("utf-8", "replace").strip())
        except OSError:
            continue
        if not cmd:
            continue
        rest = st[st.rfind(")") + 2:].split()   # field N of proc(5) is rest[N - 3]
        if len(rest) < 13:
            continue
        try:
            out[int(name)] = ((int(rest[11]) + int(rest[12])) / tck, cmd)
        except ValueError:
            continue
    return out


def foreign_cores(before: dict[int, tuple[float, str]], after: dict[int, tuple[float, str]],
                  wall_s: float) -> float:
    """Total foreign CPU between the two samples, in cores.

    The quantity the calibration in `docs/decisions.md` is expressed in, and
    the one `FOREIGN_BUDGET_CORES` is compared against. Summed over every
    foreign process, because load that arrives as ten small processes costs the
    same as load that arrives as one large one.
    """
    if wall_s <= 0:
        return 0.0
    total = 0.0
    for pid, (cpu1, cmd) in after.items():
        prev = before.get(pid)
        cpu0 = prev[0] if prev is not None and prev[1] == cmd else 0.0
        if not _is_ours(cmd):
            total += max(0.0, cpu1 - cpu0)
    return total / wall_s


def busy_processes(before: dict[int, tuple[float, str]], after: dict[int, tuple[float, str]],
                   wall_s: float, threshold: float = FOREIGN_PCT) -> list[str]:
    """Foreign processes worth naming, as `name(N%)`, busiest first, at most eight.

    A list for the operator, not a verdict: `foreign_cores` decides. The
    threshold here is low so that load made of several small processes is
    *named* even though no single one of them would fail anything.
    """
    if wall_s <= 0:
        return []
    found = []
    for pid, (cpu1, cmd) in after.items():
        prev = before.get(pid)
        cpu0 = prev[0] if prev is not None and prev[1] == cmd else 0.0
        pct = 100 * max(0.0, cpu1 - cpu0) / wall_s
        if pct > threshold and not _is_ours(cmd):
            found.append((pct, f"{os.path.basename(cmd.split()[0])}({pct:.0f}%)"))
    found.sort(key=lambda x: -x[0])
    return [s for _, s in found[:8]]


def check_quiescent(e: Env) -> None:
    """Is the machine actually idle?

    Every other check here is a configuration property, and all of them can be
    correct on a machine that is busy. This one exists because that happened: a
    benchmark ran at load average 30 against an arm measured on an idle box, and
    nothing in the tooling said so. Adding the check immediately found a
    10-core LLM inference server that had run through an entire workload table
    unnoticed.
    """
    # `cores` is the CPU count for the whole function: it used to be rebound
    # to the foreign-core sum below, so `env.txt` archived "quiescent (load
    # 0.4 over 0.03 cores)".
    cores = os.cpu_count() or 1
    load1 = float((read("/proc/loadavg") or "0").split()[0])
    per_core = load1 / cores * 100
    e.observe(f"load1={load1} load_per_core_pct={per_core:.0f}")

    try:
        # Two `/proc` samples a second apart, not `ps`: its `pcpu` is a lifetime
        # average, which flagged a long-idle process with a busy history and
        # said nothing about one that had just started.
        before = _cpu_seconds()
        time.sleep(QUIESCENT_SAMPLE_S)
        after = _cpu_seconds()
        busy = busy_processes(before, after, QUIESCENT_SAMPLE_S)
        foreign = foreign_cores(before, after, QUIESCENT_SAMPLE_S)
    except Exception as exc:
        # Not `pass`. This check exists to catch foreign load; swallowing the
        # error makes it report a clean machine when it could not look, which
        # is the one outcome worse than not having the check.
        e.fail(f"could not enumerate processes ({exc.__class__.__name__}), so "
               f"foreign load is UNKNOWN for this run")
        return

    # `observe`, not `note`: which processes happen to be busy right now is
    # transient, and hashing it makes the gate's hash disagree with the one
    # stamped on the rows. That is the same mistake `observe` was introduced
    # for, one line further down. A run passed the gate at hash
    # 1cdcf393fa18f478 and recorded every row under 473591cd7728f349.
    if busy:
        e.observe(f"busy_processes={' '.join(busy[:8])}")
        print(f"       busy: {' '.join(busy[:8])}", file=sys.stderr)

    # The verdict uses the sample above, not the load average alone. load1 is a
    # one-minute *decaying* average: it cannot tell "quiet now" from "busy until
    # a moment ago", and it read 1.15 — inside the threshold — sixty seconds
    # after a `rustc` at 102% had been seen by this very sampler. That run
    # measured ten of one engine's rows against a pinned core and published
    # them. `busy` is the measurement; load1 stays as the check for load made of
    # many small processes that no single entry catches.
    if foreign > FOREIGN_BUDGET_CORES:
        e.fail(f"machine is NOT quiescent: {foreign:.2f} cores of foreign load"
               + (f" ({' '.join(busy[:8])})" if busy else ""),
               f"more than {FOREIGN_BUDGET_CORES:.2f} cores summed over "
               f"{QUIESCENT_SAMPLE_S:.0f}s;",
               "measured cost of contamination (decisions.md): 1 core is within",
               "run-to-run noise, 3 cores add 11%, 18 cores add 41%.")
    elif busy:
        # Named but not fatal: under the budget the calibration calls noise.
        # Said anyway, because the operator may still want to close it, and
        # because a row that later drifts over the budget should not be the
        # first time this appears in the log.
        e.observe(f"foreign_load_cores={foreign:.2f} (budget "
                  f"{FOREIGN_BUDGET_CORES:.2f}, within noise)")
    elif per_core <= LOAD_PER_CORE_MAX_PCT:
        e.ok(f"machine is quiescent (load {load1} over {cores} cores)")
    else:
        e.fail(f"machine is NOT quiescent: load {load1} over {cores} cores "
               f"({per_core:.0f}% per core)",
               "numbers taken now are not comparable to numbers taken idle,",
               "and the difference is invisible in the results.")


CHECKS = [check_uarch, check_governor, check_turbo, check_smt, check_isolation,
          check_memory, check_perf, check_instruments, check_toolchain,
          check_quiescent]


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def cpu_is_online(gov_path: Path) -> bool:
    """Whether the CPU a `cpufreq` path belongs to is currently online.

    An offline CPU keeps its `cpufreq` directory, so `Path.exists()` is true
    and the kernel then refuses the write with EBUSY. That is how `apply`
    came to print twelve "could not set ... Device or resource busy" lines on
    a host it had configured correctly: back when `apply` turned SMT off, the
    twelve sibling threads it had just offlined still had governor files. An
    operator reading that output has no way to tell it from a real failure to
    set the governor. `apply` no longer offlines anything, but a host with SMT
    off for its own reasons produces exactly the same situation.

    `check` never had the problem because an offline CPU's `scaling_governor`
    is unreadable, so `glob_read` drops it. `exists()` is the weaker test and
    this is the difference between them.

    `cpu0` has no `online` file at all, because it cannot be offlined; a
    missing file therefore means online.
    """
    v = read(str(gov_path.parent.parent / "online"))
    return v is None or v.strip() == "1"


def apply() -> int:
    if os.geteuid() != 0:
        print("apply needs root", file=sys.stderr)
        return 1
    print("applying §7.1 settings...", file=sys.stderr)
    writes: list[tuple[str, str]] = []
    skipped = 0
    for g in Path("/").glob("sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        if not cpu_is_online(g):
            skipped += 1
            continue
        writes.append((str(g), "performance"))
    if skipped:
        # Said, not silent: "I set fewer than there are CPUs" is information,
        # and an operator who sees nothing about the other twelve is entitled
        # to wonder whether they were missed.
        print(f"  {skipped} offline cpu(s) skipped for the governor "
              f"(SMT siblings; they take the setting when brought online)",
              file=sys.stderr)
    writes += [
        ("/sys/devices/system/cpu/intel_pstate/no_turbo", "1"),
        ("/sys/devices/system/cpu/cpufreq/boost", "0"),
        ("/proc/sys/kernel/numa_balancing", "0"),
        ("/proc/sys/kernel/perf_event_paranoid", "-1"),
        # Off by default on most distributions, and off means the kernel keeps
        # no per-task delay accounting at all: `delayacct_blkio_ticks` reads
        # zero forever, which looks exactly like an engine that never waited
        # for a disk. See `procstat.delayacct_on`.
        ("/proc/sys/kernel/task_delayacct", "1"),
        # Not `smt/control`. `apply` did turn SMT off, back when `check_smt`
        # failed without it; now that the gate declares SMT instead of
        # requiring it, offlining half the machine's threads is a change to
        # the operator's box that nothing downstream asked for.
    ]
    for path, value in writes:
        p = Path(path)
        if not p.exists():
            continue
        try:
            p.write_text(value)
        except OSError as exc:
            print(f"  could not set {path}: {exc}", file=sys.stderr)
    delegate_cgroup_io()
    # isolcpus and nohz_full are kernel cmdline and need a reboot, so they are
    # reported by `check` and never silently half-applied here.
    print("done. Everything settable at runtime is set; isolcpus/nohz_full are\n"
          "kernel cmdline and need a reboot. Re-run `check` to confirm.",
          file=sys.stderr)
    return 0


def delegate_cgroup_io() -> None:
    """Enable the cgroup io controller down to the engines' own cgroup.

    A cgroup has `io.stat` when its *parent* enables `io` in
    `cgroup.subtree_control`, so reaching a `systemd-run --user --scope` means
    enabling it at every level from `user.slice` down to the slice the user
    manager puts transient units in (`app.slice`) -- not merely down to
    `user@UID.service`, which is where this stopped, one level short of the
    scope's own parent and therefore short of any effect.
    The root already has it; `user.slice` does not, which is why `procstat`
    falls back to `/proc/<pid>/io` — bytes and syscalls, no request counts —
    and every report so far printed `disk read ops: unknown`.

    Transient by design, and matched to how `apply` is used: systemd rewrites
    a delegated subtree on the next login, exactly as the governor reverts, and
    `apply` is re-run before a publication run anyway. An operator who wants it
    permanent wants `Delegate=cpu cpuset io memory pids` in a drop-in for
    `user@.service`, which is a change to their login session rather than to a
    knob, so this prints it rather than writing it.

    Best effort throughout: a kernel without the controller, or a systemd that
    takes the level back, costs the operation counts and nothing else.
    """
    if "io" not in (read(f"{CGROUP_ROOT}/cgroup.controllers") or "").split():
        return
    # One list, shared with the check above, so the two cannot disagree about
    # which levels matter -- they did, and the check was the optimistic one.
    levels = io_levels()
    done, failed = [], []
    for d in levels:
        ctl = d / "cgroup.subtree_control"
        if not ctl.exists():
            continue
        if "io" in (read(str(ctl)) or "").split():
            continue
        try:
            ctl.write_text("+io")
            done.append(str(d.name))
        except OSError as exc:
            failed.append(f"{d.name} ({exc.strerror or exc})")
    if done:
        print(f"  cgroup io delegated to: {', '.join(done)} — block-layer "
              f"read/write operation counts are now measurable", file=sys.stderr)
    if failed:
        print(f"  could not delegate cgroup io to: {', '.join(failed)};",
              file=sys.stderr)
        print("  `disk read ops` / `disk write ops` stay unknown. For a "
              "permanent fix add", file=sys.stderr)
        print("  `[Service]\\nDelegate=cpu cpuset io memory pids` to "
              "/etc/systemd/system/user@.service.d/delegate.conf,", file=sys.stderr)
        print("  then `systemctl daemon-reload` and log in again.", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("mode", nargs="?", default="check",
                    choices=["check", "describe", "apply"])
    ap.add_argument("--lax", action="store_true",
                    help="downgrade failures to warnings; a --lax run MUST NOT "
                         "produce published numbers")
    args = ap.parse_args()

    if args.mode == "apply":
        return apply()

    e = Env(lax=args.lax)
    if args.mode == "check":
        print("§7.1 host discipline:", file=sys.stderr)
    for c in CHECKS:
        c(e)

    h = e.hash()
    e.note(f"env_hash={h}")

    if args.mode == "describe":
        print("\n".join(e.lines))
        return 0

    print(file=sys.stderr)
    print(f"measurement profile: {e.profile}", file=sys.stderr)
    print(f"environment hash: {h}", file=sys.stderr)
    print("(every result row carries this; rows with different hashes never "
          "share a chart, §7.1)", file=sys.stderr)

    if e.failures:
        print(file=sys.stderr)
        print(f"{e.failures} check(s) failed. §7.1: this script refuses to run "
              f"if any check fails.", file=sys.stderr)
        print(f"Run '{sys.argv[0]} apply' as root for the settable ones, or "
              f"'--lax' to proceed anyway", file=sys.stderr)
        print("for development. A run made with --lax MUST NOT produce "
              "published numbers.", file=sys.stderr)
        return 1

    print(f"all checks passed; this host is fit to produce published numbers "
          f"for the {e.profile} profile", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
