#!/usr/bin/env python3
"""Why Qdrant keeps 4.5 of its 8 cores busy on W4 (findings 5).

On dbpedia-openai-1m's W4 (64 requests in flight) Qdrant served 2,580 q/s on
4.46 of 8 pinned cores, with 0.2 s of run-queue wait over the row, zero major
faults and 5.85 voluntary context switches per query against strawmANN's 1.8.
Nothing in its source caps the searches below 8: `max_search_threads: 0`
builds a 32-thread `search-io` pool, and the adaptive handle moves to the
8-thread `search-cpu` pool only above 7.2 busy cores. So its threads block,
and what on is the question: the hand-off through the search runtime's single
async worker, or lock waits on the segment.

One collection, uploaded once, Qdrant restarted per arm without wiping it:

    default   the repository's config in production mode: `max_search_threads: 0`
    mst8      `QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8`, which sizes
              both search pools to 8 and removes the adaptive switch

alternated over `--reps`, W4 with the `perf stat` sidecar each time, and each W4
sampled for thread states: per pool, how often its threads were running
against sleeping. Then one more default W4 under `perf record` of its
context switches with call stacks, for where the sleeping threads sleep.

Development-grade: same hour, no strawmANN, no publication gate. Usage:

    taskset -c 0-3 qdrant_w4_probe.py --server-cpus 4-11 --qdrant-binary PATH

(the `taskset` places the load generator; Qdrant is placed by its own).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import fullrun
import perfstat
import w9_ab

import procstat
import workloads

MST_ENV = "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS"

#: name, value for `MST_ENV` (None: the config's own `0`).
ARMS: tuple[tuple[str, str | None], ...] = (("default", None), ("mst8", "8"))

#: Thread-state samples per second during a row.
SAMPLE_HZ = 20


def plan(reps: int) -> list[tuple[str, int]]:
    """`(arm, rep)` in run order: A/B/A/B, so a drift lands on both."""
    return [(arm, rep) for rep in range(1, reps + 1) for arm, _ in ARMS]


def pool_of(comm: str) -> str:
    """A thread's pool, from its name with the index taken off:
    `search-io-12` and `search-io-3` are one pool."""
    return re.sub(r"[-_ ]?\d+$", "", comm) or comm


def summarise_states(samples: list[list[tuple[str, str]]]) -> dict[str, dict[str, float]]:
    """Per pool, the mean number of its threads in each scheduler state.

    `samples` is one list of `(comm, state)` per sampling instant. `R` is
    running or runnable, `S` interruptible sleep (a futex, a socket, a condvar),
    `D` uninterruptible (I/O). A search pool whose threads sit in `S` while 64
    requests are in flight is being starved of work, not of CPU.
    """
    acc: dict[str, dict[str, int]] = {}
    for snap in samples:
        for comm, state in snap:
            by = acc.setdefault(pool_of(comm), {})
            by[state] = by.get(state, 0) + 1
    n = max(1, len(samples))
    return {pool: {st: c / n for st, c in sorted(by.items())}
            for pool, by in sorted(acc.items())}


def thread_states(pid: int) -> list[tuple[str, str]]:
    """`(comm, state)` for every thread of `pid`, from `/proc`."""
    out = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            stat = (task / "stat").read_text()
        except OSError:
            continue
        # comm is in parentheses and may contain spaces; the state follows it.
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        out.append((comm, stat[stat.rindex(")") + 2]))
    return out


class Sampler:
    """Samples a process's thread states on a thread of its own."""

    def __init__(self, pid: int):
        self.pid, self.samples, self._stop = pid, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(thread_states(self.pid))
            except OSError:
                return
            time.sleep(1 / SAMPLE_HZ)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


def qdrant_pid() -> int | None:
    found = [pid for pid, comm in procstat.engine_processes() if comm.startswith("qdrant")]
    return found[0] if len(found) == 1 else None


STARTS: list[str] = []


def keep_log(out: Path) -> None:
    """`start_qdrant_binary` truncates one `qdrant-server.log` per start; copy
    it aside before the next start, so each arm's configuration is on file."""
    src = fullrun.RESULTS / "qdrant-server.log"
    if STARTS and src.exists():
        (out / f"qdrant-server.{len(STARTS)}-{STARTS[-1]}.log").write_bytes(src.read_bytes())


def start(arm_value: str | None, server_cpus: str, wipe: bool, out: Path) -> bool:
    keep_log(out)
    if arm_value is None:
        os.environ.pop(MST_ENV, None)
    else:
        os.environ[MST_ENV] = arm_value
    STARTS.append("mst" + arm_value if arm_value else "default")
    return fullrun.start_qdrant_binary(server_cpus, fullrun.QDRANT_GRPC,
                                       fullrun.QDRANT_REST, wipe=wipe) is not None


def wait_loaded(n: int, timeout_s: int = 900) -> bool:
    """Until bench2 answers green with every point, after a restart."""
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://localhost:{fullrun.QDRANT_REST}/collections/{workloads.C}2",
                    timeout=10) as r:
                res = json.loads(r.read()).get("result") or {}
                if res.get("status") == "green" and res.get("points_count") == n:
                    return True
        except (OSError, ValueError):
            pass
        time.sleep(5)
    return False


COMMON = ["--retry", "0", "--timeout", str(workloads.BFB_TIMEOUT_S), "--p9", "3"]


def measure(out: Path, arm: str, rep: int) -> dict:
    w4 = {w.id: w for w in workloads.table()}["W4"]
    d = out / f"{arm}-r{rep}"
    d.mkdir(parents=True, exist_ok=True)
    w9_ab.warn_other_engines()
    pid = qdrant_pid()
    smp = Sampler(pid) if pid else None
    with smp or contextlib.nullcontext():
        res = workloads.run_one(w4, f"http://localhost:{fullrun.QDRANT_GRPC}", d, COMMON,
                                perf="default")
    row = dataclasses.asdict(res)
    row.update(perfstat.derive(row))
    row.update({"arm": arm, "rep": rep,
                "states": summarise_states(smp.samples) if smp else {}})
    (d / "row.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    cores = (row["perf_task_clock_s"] / row["duration_s"]
             if row.get("perf_task_clock_s") and row.get("duration_s") else None)
    print(f"  {arm} rep {rep}: {row.get('qps')} q/s, {cores and round(cores, 2)} cores, "
          f"foreign {row.get('foreign') or '-'}", flush=True)
    return row


def profile(out: Path) -> None:
    """One more default W4 under `perf record` of the scheduler's switches."""
    pid = qdrant_pid()
    if pid is None:
        print("  !! no single qdrant process to profile; skipped", flush=True)
        return
    d = out / "profile"
    d.mkdir(parents=True, exist_ok=True)
    data = d / "perf.data"
    # The software event, not `sched:sched_switch`: tracefs is root-only on
    # this host. Every tenth switch with a 4 KB user stack keeps a 20 s row at
    # ~120 MB instead of ~2 GB at Qdrant's ~15,000 switches a second.
    rec = subprocess.Popen(["perf", "record", "-e", "context-switches", "-c", "10",
                            "-p", str(pid), "--call-graph", "dwarf,4096",
                            "-o", str(data), "--quiet"])
    try:
        w4 = {w.id: w for w in workloads.table()}["W4"]
        workloads.run_one(w4, f"http://localhost:{fullrun.QDRANT_GRPC}", d, COMMON)
    finally:
        w9_ab.stop_perf(rec, 300)
    rep = subprocess.run(["perf", "report", "-i", str(data), "--stdio", "--no-children",
                          "--sort", "comm,sym", "--percent-limit", "1"],
                         capture_output=True, text=True)
    (d / "offcpu-report.txt").write_text(rep.stdout[:2_000_000] + rep.stderr)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--qdrant-binary", required=True)
    ap.add_argument("--dataset", default="dbpedia-openai-1m")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default=f"qdw4-{time.strftime('%m%d')}")
    args = ap.parse_args(argv[1:])

    workloads.use_dataset(args.dataset)
    fullrun.QDRANT_BINARY = Path(args.qdrant_binary)
    if busy := procstat.builders_alive():
        print(f"!! build processes alive ({', '.join(busy)}): foreign load every row "
              f"will record", flush=True)
    out = fullrun.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    try:
        return run_arms(args, out, rows)
    finally:
        fullrun.stop_qdrant()
        keep_log(out)
        os.environ.pop(MST_ENV, None)
        write_summary(out, rows)


def run_arms(args, out: Path, rows: list[dict]) -> int:
    if not start(None, args.server_cpus, wipe=True, out=out):
        return 1
    w2 = {w.id: w for w in workloads.table()}["W2"]
    (out / "upload").mkdir(parents=True, exist_ok=True)
    up = workloads.run_one(w2, f"http://localhost:{fullrun.QDRANT_GRPC}",
                           out / "upload", COMMON)
    if up.status != workloads.Status.ok:
        print(f"upload failed: {up.status}", file=sys.stderr)
        return 1
    current = "default"
    for arm, rep in plan(args.reps):
        value = dict(ARMS)[arm]
        if arm != current:
            if not start(value, args.server_cpus, wipe=False, out=out) or \
                    not wait_loaded(workloads.upload_n()):
                print(f"{arm}: qdrant did not come back with the collection",
                      file=sys.stderr)
                return 1
            current = arm
        fullrun.settle(f"{arm} rep {rep}")
        rows.append(measure(out, arm, rep))
    if current != "default" and (not start(None, args.server_cpus, wipe=False, out=out)
                                 or not wait_loaded(workloads.upload_n())):
        return 1
    fullrun.settle("the profile")
    profile(out)
    return 0


def write_summary(out: Path, rows: list[dict]) -> None:
    summary = {"arms": {}, "rows": rows}
    for arm, _ in ARMS:
        got = [r for r in rows if r["arm"] == arm and r.get("qps")]
        summary["arms"][arm] = {
            "qps": [r["qps"] for r in got],
            "cores": [round(r["perf_task_clock_s"] / r["duration_s"], 2)
                      for r in got if r.get("perf_task_clock_s") and r.get("duration_s")],
            "voluntary_switches_per_q": [round(r["ctx_switches_voluntary"] / r["n_queries"], 2)
                                         for r in got if r.get("ctx_switches_voluntary")
                                         and r.get("n_queries")],
            "states": [r["states"] for r in got],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    lines = []
    for arm, s in summary["arms"].items():
        lines.append(f"{arm:<8} q/s {s['qps']}  cores {s['cores']}  "
                     f"voluntary switches/q {s['voluntary_switches_per_q']}")
        if s["states"]:
            for pool, st in s["states"][0].items():
                lines.append(f"    {pool:<28} " + "  ".join(f"{k} {v:.1f}" for k, v in st.items()))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
