"""What else the host is doing: load averages, CPU samples, and foreign processes.

Moved out of `workloads.py`, which re-exports every name here.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# `setup` lives in bench/, beside this directory, as it does for workloads.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import perfstat

import procstat
import provenance
import setup

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
