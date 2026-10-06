#!/usr/bin/env python3
"""The same rows on several Qdrant binaries, for a change proposed to Qdrant.

`commit_ab.py` does this for strawmANN's commits; this is its Qdrant side,
for the two changes `docs/decisions.md` (2026-10-06) traced the largest
gaps to: no software prefetch on dense fp32 vectors (sift1m W4, 16x the
demand DRAM fills) and AVX2-only f32 kernels (dbpedia W3, 75% more
instructions). Each arm is a binary built by the caller; each rep starts it
on wiped storage, uploads W2, and measures the rows with perf counters,
rotating the arm order every rep.

Development-grade: no strawmANN, no §7.1 publication gate. Each row still
records its own foreign load. Usage:

    qdrant_build_ab.py --server-cpus 4-11 --client-cpus 0-3 \\
        --arm base=/path/to/qdrant --arm prefetch=/path/to/qdrant \\
        [--dataset sift1m] [--rows W3 W4 W10-ef128] [--reps 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import commit_ab
import fullrun
import w9_ab

import procstat
import workloads

ROWS = ("W3", "W4", "W10-ef128")


def parse_arms(specs: list[str]) -> dict[str, Path]:
    """`name=path` pairs, in the order given; a name may appear once."""
    arms: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--arm wants name=path, got {spec!r}")
        if name in arms:
            raise ValueError(f"--arm {name} given twice")
        arms[name] = Path(path).expanduser()
    return arms


def run_sessions(args, out: Path, arms: dict[str, Path], rows: list[dict]) -> int:
    table = {w.id: w for w in workloads.table()}
    uri = f"http://localhost:{fullrun.QDRANT_GRPC}"
    for rep, arm in commit_ab.plan(list(arms), args.reps):
        fullrun.say(f"{arm} rep {rep}")
        fullrun.QDRANT_BINARY = arms[arm]
        if fullrun.start_qdrant_binary(args.server_cpus, fullrun.QDRANT_GRPC,
                                       fullrun.QDRANT_REST) is None:
            return 1
        results = out / f"{arm}-r{rep}"
        results.mkdir(parents=True, exist_ok=True)
        try:
            up = workloads.run_one(table["W2"], uri, results, w9_ab.COMMON)
            if up.status != workloads.Status.ok:
                print(f"upload failed: {up.status}", file=sys.stderr)
                return 1
            for wid in args.rows:
                fullrun.settle(f"{arm} rep {rep} {wid}")
                rows.append(commit_ab.measure(table[wid], uri, out, arm, rep))
        finally:
            fullrun.stop_qdrant()
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--client-cpus", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default=f"qdrant-ab-{time.strftime('%m%d')}")
    args = ap.parse_args(argv[1:])

    arms = parse_arms(args.arm)
    if missing := [str(p) for p in arms.values() if not os.access(p, os.X_OK)]:
        print(f"no executable at {', '.join(missing)}", file=sys.stderr)
        return 2
    workloads.use_dataset(args.dataset)
    os.sched_setaffinity(0, w9_ab.cpu_set(args.client_cpus))
    if busy := procstat.builders_alive():
        print(f"!! build processes alive ({', '.join(busy)}): foreign load every row "
              f"will record", flush=True)
    claude = [c for _, c in procstat.processes() if c == "claude"]
    if claude:
        print(f"!! {len(claude)} claude process(es) alive; their idle CPU is foreign load",
              flush=True)
    out = fullrun.RESULTS / f"{args.tag}-{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    try:
        return run_sessions(args, out, arms, rows)
    finally:
        commit_ab.write_summary(out, rows, list(arms))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
