#!/usr/bin/env python3
"""Whether a reclaimed and re-read collection searches faster than a resident one.

findings 62: strawmANN's first pass of a dbpedia night runs W10 5 to 9% faster
at the same instructions per query, and it is the one pass whose arenas were
reclaimed under memory pressure and read back (about 10 GB less file RSS, the
only major faults); the passes whose pages stayed resident since the engine
wrote them are the slow ones. This tests that on one engine and one upload:
W10 rows with the collection as written, then the engine's cgroup reclaimed
through `memory.reclaim` (its scope is user-delegated, so no privilege), one
warming round, and the same rows again on the pages read back.

Development-grade, like `commit_ab.py`, whose measuring it reuses: no Qdrant,
no publication. Needs a quiet machine. Usage:

    reclaim_ab.py --server-cpus 4-11 --client-cpus 0-3 \\
                  [--dataset dbpedia-openai-1m] [--rows W10-ef32 W10-ef128] [--rounds 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ab_common
import commit_ab
import fullrun

import procstat
import workloads

ROWS = ("W10-ef32", "W10-ef128")
PORT = commit_ab.PORT


def phases(rounds: int) -> list[tuple[str, int]]:
    """`(arm, round)` in run order: resident rounds, a reclaim, a warm round,
    re-read rounds. `reclaim` is the step between, not a measurement."""
    return ([("resident", r) for r in range(1, rounds + 1)] + [("reclaim", 0), ("warm", 1)]
            + [("reread", r) for r in range(1, rounds + 1)])


def memory_current(cg: Path) -> int | None:
    try:
        return int((cg / "memory.current").read_text())
    except (OSError, ValueError):
        return None


def reclaim(cg: Path, nbytes: int) -> str:
    """Ask the kernel to reclaim `nbytes` from cgroup `cg`, and say what it did.

    `memory.reclaim` raises EAGAIN when it reclaims less than asked, which is
    the usual answer for a request sized to everything: what it did reclaim
    stays reclaimed, so that is reported from `memory.current`, not refused.
    """
    before = memory_current(cg)
    err = ""
    try:
        (cg / "memory.reclaim").write_text(str(nbytes))
    except OSError as exc:
        err = f" ({exc.strerror or exc})"
    after = memory_current(cg)
    if before is None or after is None:
        return f"reclaim {cg.name}: memory.current unreadable{err}"
    return (f"reclaim {cg.name}: {procstat.human_bytes(before)} -> "
            f"{procstat.human_bytes(after)}{err}")


def engine_cgroup() -> Path | None:
    """The engine's own cgroup, refused when anything else shares it."""
    procs = procstat.engine_processes()
    if len(procs) != 1:
        return None
    pid = procs[0][0]
    cg = procstat.cgroup_dir(pid)
    return cg if cg is not None and procstat.cgroup_holds_only(pid, cg) else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--client-cpus", required=True)
    ap.add_argument("--dataset", default="dbpedia-openai-1m")
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--binary", type=Path, default=fullrun.ROOT / "zig-out/bin/strawmann")
    ap.add_argument("--tag", default=f"reclaim-ab-{time.strftime('%m%d')}")
    args = ap.parse_args(argv[1:])

    ab_common.prepare(args.dataset, args.client_cpus)
    table = workloads.by_id()
    out = fullrun.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    log = out / "server.log"
    workers = max(1, fullrun.cpu_count(args.server_cpus) - 1)
    uri = f"http://localhost:{PORT}"
    rows: list[dict] = []

    def start():
        return fullrun.start_strawmann(args.server_cpus, PORT, log, workers,
                                       workloads.required_capacity(), binary=args.binary)

    def body():
        if (cg := engine_cgroup()) is None:
            print("the engine has no cgroup of its own to reclaim; nothing measured",
                  file=sys.stderr)
            return 1
        for arm, rnd in phases(args.rounds):
            if arm == "reclaim":
                print(f"  {reclaim(cg, 1 << 50)}", flush=True)
                continue
            for wid in args.rows:
                fullrun.settle(f"{arm} {rnd} {wid}")
                row = commit_ab.measure(table[wid], uri, out, arm, rnd)
                if arm != "warm":
                    rows.append(row)
        return 0

    def stop(engine):
        fullrun.stop_strawmann(engine)
        commit_ab.write_summary(out, rows, ["resident", "reread"])

    return ab_common.session(start, stop, table["W2"], uri, out, body)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
