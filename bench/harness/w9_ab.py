#!/usr/bin/env python3
"""Why W9 loses at d=1536: three measurements of strawmANN's exact scan.

W9 is 1.31x on sift1m and 0.77x on dbpedia-openai-1m (0925). At d=1536
strawmANN's IPC falls from 2.06 to 0.26 and 9.8 q/s x 6.08 GB is ~60 GB/s,
about the bus: each query streams the whole arena from DRAM. The 2026-09-22
decision dropped a cross-request gather because on sift1m the eight scans
already shared lines through the cache (118 GB/s implied against a 73 GB/s
bus); at 6 GB that sharing is gone. Qdrant implies ~78 GB/s from 4 cores.

Same engine, same collection, same query count, one thing varied per arm:

    stream7   7 workers, one query per request    W9 as published
    stream4   4 workers, one query per request    fewer concurrent streams
    batch8    7 workers, 8 queries per request    one arena pass per 8 queries,
                                                  the ceiling of a gather that
                                                  splits the arena across workers

plus a `perf record` of one extra stream7 pass, for where the scan's cycles go
(the demand misses are 1,633 MB/query against Qdrant's 179).

Development-grade: same hour, no Qdrant, no §7.1 publication gate. Each row
still records its own foreign load. Usage:

    w9_ab.py --server-cpus 4-11 --client-cpus 0-3 [--reps 3] [--queries 1000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import fullrun
import perfstat

import procstat
import workloads

#: name, query workers, queries per request.
ARMS: tuple[tuple[str, int, int], ...] = (
    ("stream7", 7, 1),
    ("stream4", 4, 1),
    ("batch8", 7, 8),
)

PORT = 6334

#: bfb's flags every row passes, as `workloads.main` builds them.
COMMON = ["--retry", "0", "--timeout", str(workloads.BFB_TIMEOUT_S), "--p9", "3"]


def cpu_set(spec: str) -> set[int]:
    """`0-3,8` as `{0, 1, 2, 3, 8}`."""
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def plan(reps: int) -> list[tuple[int, list[tuple[str, int]]]]:
    """Engine sessions in order: `(workers, [(arm, rep), ...])`.

    The worker count is a start-up flag, so arms that share one share a
    session, and within it they alternate A/B/A/B rather than run in blocks,
    so a drift over the session lands on both.
    """
    sessions: dict[int, list[tuple[str, int]]] = {}
    for rep in range(1, reps + 1):
        for arm, workers, _ in ARMS:
            sessions.setdefault(workers, []).append((arm, rep))
    return list(sessions.items())


def set_n(args: list[str], n: int) -> list[str]:
    out = list(args)
    out[out.index("-n") + 1] = str(n)
    return out


def w9_row(arm: str, batch: int, queries: int) -> workloads.Workload:
    """The table's W9, with its query count and batch size as given."""
    w9 = {w.id: w for w in workloads.table()}["W9"]
    args = set_n(list(w9.args), queries)
    if batch > 1:
        args += ["--search-batch-size", str(batch)]
    return dataclasses.replace(w9, id=f"W9-{arm}", args=args)


def strawmann_pid() -> int | None:
    """The engine itself: `start_strawmann` may return the `systemd-run`
    that scopes it, whose pid is not the one to profile."""
    found = [pid for pid, comm in procstat.engine_processes() if comm.startswith("strawmann")]
    return found[0] if len(found) == 1 else None


def perf_record_argv(pid: int, out: Path) -> list[str]:
    """A flat profile of the engine: where the cycles are, not who called."""
    return ["perf", "record", "-F", "499", "-p", str(pid), "-o", str(out), "--quiet"]


def summarise(rows: list[dict], corpus_bytes: int) -> list[dict]:
    """Per arm: medians over its reps of the figures that tell the causes apart.

    `implied_gbs` is qps x the arena: the traffic if every query streamed it
    once. Above the measured bus, the scans are sharing lines; at it, they
    are not.
    """
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    out = []
    for arm, _, batch in ARMS:
        got = [r for r in by_arm.get(arm, []) if r.get("qps")]
        if not got:
            continue
        out.append({
            "arm": arm, "batch": batch, "reps": len(got),
            "qps": _median(got, lambda r: r["qps"]),
            "qps_spread": max(r["qps"] for r in got) - min(r["qps"] for r in got),
            "cores": _median(got, lambda r: _ratio(r, "perf_task_clock_s", "duration_s")),
            "cycles_per_q": _median(got, lambda r: _ratio(r, "perf_cycles", "n_queries")),
            "ipc": _median(got, lambda r: r.get("ipc")),
            "dram_mb_per_q": _median(got, lambda r: (v / 1e6 if (v := _ratio(
                r, "dram_bytes", "n_queries")) is not None else None)),
            "implied_gbs": _median(got, lambda r: r["qps"] * corpus_bytes / 1e9),
            "foreign": sorted({r["foreign"] for r in got if r.get("foreign")}),
        })
    return out


def _ratio(r: dict, num: str, den: str) -> float | None:
    return r[num] / r[den] if r.get(num) and r.get(den) else None


def _median(rows: list[dict], f) -> float | None:
    vals = [v for r in rows if (v := f(r)) is not None]
    return statistics.median(vals) if vals else None


def table_text(summary: list[dict], bus: str) -> str:
    lines = [f"bus (strawmann --probe): {bus}",
             f"{'arm':<8} {'reps':>4} {'q/s':>8} {'spread':>7} {'cores':>6} "
             f"{'Mcyc/q':>8} {'IPC':>5} {'DRAM MB/q':>10} {'implied GB/s':>13}  foreign"]
    for s in summary:
        def f(v, fmt):
            return format(v, fmt) if v is not None else "-"
        lines.append(
            f"{s['arm']:<8} {s['reps']:>4} {f(s['qps'], '8.2f')} {f(s['qps_spread'], '7.2f')} "
            f"{f(s['cores'], '6.2f')} {f(s['cycles_per_q'] and s['cycles_per_q'] / 1e6, '8.0f')} "
            f"{f(s['ipc'], '5.2f')} {f(s['dram_mb_per_q'], '10.0f')} "
            f"{f(s['implied_gbs'], '13.1f')}  {', '.join(s['foreign']) or '-'}")
    return "\n".join(lines) + "\n"


def measure(w: workloads.Workload, uri: str, label: str, arm: str, rep: int) -> dict:
    results = fullrun.RESULTS / label
    results.mkdir(parents=True, exist_ok=True)
    res = workloads.run_one(w, uri, results, COMMON, perf="default")
    row = dataclasses.asdict(res)
    row.update(perfstat.derive(row))
    row.update({"arm": arm, "rep": rep})
    (results / "row.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(f"  {arm} rep {rep}: {row.get('qps')} q/s, status {row.get('status')}, "
          f"foreign {row.get('foreign') or '-'}", flush=True)
    return row


def profile(w: workloads.Workload, uri: str, label: str) -> None:
    """One more stream7 pass under `perf record`, and its report beside it."""
    results = fullrun.RESULTS / label
    results.mkdir(parents=True, exist_ok=True)
    data = results / "perf.data"
    pid = strawmann_pid()
    if pid is None:
        print("  !! no single strawmann process to profile; skipped", flush=True)
        return
    rec = subprocess.Popen(perf_record_argv(pid, data))
    try:
        workloads.run_one(w, uri, results, COMMON)
    finally:
        rec.send_signal(signal.SIGINT)
        rec.wait(timeout=120)
    rep = subprocess.run(["perf", "report", "-i", str(data), "--stdio", "--no-children",
                          "--percent-limit", "0.5", "--sort", "symbol"],
                         capture_output=True, text=True)
    (results / "perf-report.txt").write_text(rep.stdout + rep.stderr)
    ann = subprocess.run(["perf", "annotate", "-i", str(data), "--stdio"],
                         capture_output=True, text=True)
    (results / "perf-annotate.txt").write_text(ann.stdout[:400_000] + ann.stderr)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--client-cpus", required=True)
    ap.add_argument("--dataset", default="dbpedia-openai-1m")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--tag", default=f"w9ab-{time.strftime('%m%d')}")
    args = ap.parse_args(argv[1:])

    workloads.use_dataset(args.dataset)
    # The load generator, its warm-ups and the perf sidecars are this process's
    # children, so its affinity is theirs; the engine pins itself with
    # `--pin --cpus`.
    os.sched_setaffinity(0, cpu_set(args.client_cpus))
    # Warned, not refused: the rows record their own foreign load, and the
    # sessions are the user's.
    if busy := procstat.builders_alive():
        print(f"!! build processes alive ({', '.join(busy)}): foreign load every row "
              f"will record, and memory pressure the engine can be oom-killed under",
              flush=True)
    claude = [c for _, c in procstat.processes() if c == "claude"]
    if claude:
        print(f"!! {len(claude)} claude process(es) alive; their idle CPU is foreign load",
              flush=True)
    if fullrun.cpu_count(args.server_cpus) < 1 + max(w for _, w, _ in ARMS):
        print("--server-cpus must hold one I/O thread plus the widest arm's workers",
              file=sys.stderr)
        return 2
    if not fullrun.build_strawmann():
        return 1
    out = fullrun.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([str(fullrun.ROOT / "zig-out/bin/strawmann"), "--probe"],
                           capture_output=True, text=True)
    (out / "probe.txt").write_text(probe.stdout + probe.stderr)
    bus = next((ln.strip() for ln in probe.stdout.splitlines() if "GB/s" in ln), "unknown")
    corpus_bytes = workloads.upload_n() * workloads.DIM * 4

    rows: list[dict] = []
    batch_of = {arm: batch for arm, _, batch in ARMS}
    uri = f"http://localhost:{PORT}"
    for workers, runs in plan(args.reps):
        print(f"\n=== {workers} workers: {', '.join(f'{a} rep {r}' for a, r in runs)}", flush=True)
        fullrun.wipe_strawmann_storage()
        log = out / f"server-{workers}w.log"
        engine = fullrun.start_strawmann(args.server_cpus, PORT, log, workers,
                                         workloads.required_capacity())
        if engine is None:
            return 1
        try:
            w2 = {w.id: w for w in workloads.table()}["W2"]
            upload_dir = out / f"upload-{workers}w"
            upload_dir.mkdir(parents=True, exist_ok=True)
            up = workloads.run_one(w2, uri, upload_dir, COMMON)
            if up.status != workloads.Status.ok:
                print(f"upload failed: {up.status}", file=sys.stderr)
                return 1
            for arm, rep in runs:
                fullrun.settle(f"{arm} rep {rep}")
                rows.append(measure(w9_row(arm, batch_of[arm], args.queries), uri,
                                    f"{args.tag}-{arm}-r{rep}", arm, rep))
            if workers == 7:
                fullrun.settle("the profile")
                profile(w9_row("profile", 1, args.queries), uri, f"{args.tag}-stream7-profile")
        finally:
            fullrun.stop_strawmann(engine)

    summary = summarise(rows, corpus_bytes)
    (out / "summary.json").write_text(json.dumps(
        {"bus": bus, "corpus_bytes": corpus_bytes, "queries": args.queries,
         "arms": summary, "rows": rows}, indent=2, default=str) + "\n")
    text = table_text(summary, bus)
    (out / "summary.txt").write_text(text)
    print("\n" + text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
