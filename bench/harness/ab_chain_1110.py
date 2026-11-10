#!/usr/bin/env python3
"""The 1109 night's three engine fixes, each A/B'd against its parent.

1. BM25 late filter (f33ee1f against 14bec9e): W15-filtered, which ArguAna
   ran at 0.67x from a mispredict per posting, and W15-sat beside it so a
   slower unfiltered row would show, on arguana, fiqa and quora.
2. The planner's walk term at 350,000 (4c08d9d against f33ee1f): the
   W12-sel10 ladder on sift1m, where ef 128 should now scan, and on laion,
   where ef 256 should still walk.
3. Batched search (73936bd, both arms): W4 and W5 with and without
   `--no-batch-fanout`, on laion, where W5 ran 0.86x, and on sift1m, where it
   ran 1.64x and must not regress.

Development-grade, like `phase1_chain.py`, whose runner this reuses: no
Qdrant, no publication. Scheduled, not run from a session:

    systemd-run --user --on-calendar='2026-11-10 22:00:00' --unit=strawmann-ab-1110 \\
        -p WorkingDirectory=$PWD/bench/harness \\
        $PWD/bench/.venv/bin/python ab_chain_1110.py
"""

from __future__ import annotations

import argparse
import sys
import time

import nightrun
import phase1_chain
import planner_ab
from phase1_chain import BFB, CPUS, RESULTS, Step

REPORT_FIX = "14bec9e"   # the parent of the late filter
LATE_FILTER = "f33ee1f"  # text: filter a BM25 search late
WALK_TERM = "4c08d9d"    # engine: the walk's fixed part at 350,000
FANOUT_FLAG = "73936bd"  # engine: --no-batch-fanout

BM25_CORPORA = ("arguana", "fiqa", "quora")
PLANNER_CORPORA = ("sift1m", "laion-small-clip")
BATCH_CORPORA = ("laion-small-clip", "sift1m")


def steps(tag: str) -> list[Step]:
    out = []
    for ds in BM25_CORPORA:
        out.append(Step(f"bm25-{ds}", [
            "commit_ab.py", *CPUS, "--dataset", ds, "--reps", "3", "--tag", f"{tag}-bm25-{ds}",
            "--upload", "W15-upload", "--rows", "W15-filtered", "W15-sat",
            "--commits", f"base={REPORT_FIX}", f"late={LATE_FILTER}"]))
    for ds in PLANNER_CORPORA:
        out.append(Step(f"planner-{ds}", [
            "commit_ab.py", *CPUS, "--dataset", ds, "--reps", "3", "--tag", f"{tag}-planner-{ds}",
            "--upload", "W12-upload", "--rows", *planner_ab.rows(),
            "--commits", f"walk200k={LATE_FILTER}", f"walk350k={WALK_TERM}"]))
    for ds in BATCH_CORPORA:
        out.append(Step(f"batch-{ds}", [
            "commit_ab.py", *CPUS, "--dataset", ds, "--reps", "3", "--tag", f"{tag}-batch-{ds}",
            "--rows", "W4", "W5",
            "--commits", f"fanout={FANOUT_FLAG}", f"serial={FANOUT_FLAG}",
            "--flags", "serial=--no-batch-fanout"]))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=f"ab-{time.strftime('%m%d')}")
    ap.add_argument("--only", nargs="+", help="run these steps by name")
    args = ap.parse_args(argv[1:])
    if not BFB.is_file():
        print(f"no bfb at {BFB}", file=sys.stderr)
        return 2
    chain = [s for s in steps(args.tag) if not args.only or s.name in args.only]
    out = RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    env = {**nightrun.child_env(), "BFB": str(BFB)}
    done = phase1_chain.run(chain, out, env)
    return 0 if all(d["rc"] == 0 for d in done) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
