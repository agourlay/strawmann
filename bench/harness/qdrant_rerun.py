#!/usr/bin/env python3
"""One native Qdrant binary's rows, measured as they are at this hour.

findings 58: Qdrant read laion 8 to 12% faster on 0930 than on 0928 with a
byte-identical binary, the same stamp and the same `env_hash`, and 1003
reproduced 0930, so 0928 is the odd night, unexplained. This measures the
pinned binary's W4 and W10-ef128 on a fresh laion upload, `--rounds` times;
`ab_chain_1112.py` runs it first and last, so one binary is read twice a
couple of hours apart on one night, and the two summaries say whether the
same binary moves on this host between hours.

Development-grade, like `commit_ab.py`, whose measuring it reuses: no
strawmANN, no publication. Usage:

    qdrant_rerun.py --server-cpus 4-11 --client-cpus 0-3 --qdrant-binary PATH \\
                    [--dataset laion-small-clip] [--rows W4 W10-ef128] [--rounds 3]
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

import workloads

ROWS = ("W4", "W10-ef128")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--client-cpus", required=True)
    ap.add_argument("--qdrant-binary", type=Path,
                    default=Path(os.environ["QDRANT_BINARY"]) if os.environ.get("QDRANT_BINARY")
                    else None)
    ap.add_argument("--dataset", default="laion-small-clip")
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default=f"qdrant-rerun-{time.strftime('%m%d-%H%M')}")
    args = ap.parse_args(argv[1:])
    if args.qdrant_binary is None or not args.qdrant_binary.is_file():
        print(f"no qdrant binary at {args.qdrant_binary}", file=sys.stderr)
        return 2

    workloads.use_dataset(args.dataset)
    os.sched_setaffinity(0, w9_ab.cpu_set(args.client_cpus))
    fullrun.QDRANT_BINARY = args.qdrant_binary
    table = {w.id: w for w in workloads.table()}
    out = fullrun.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    uri = f"http://localhost:{fullrun.QDRANT_GRPC}"
    rows: list[dict] = []
    if fullrun.start_qdrant_binary(args.server_cpus, fullrun.QDRANT_GRPC,
                                   fullrun.QDRANT_REST) is None:
        return 1
    try:
        up = workloads.run_one(table["W2"], uri, out, w9_ab.COMMON)
        if up.status != workloads.Status.ok:
            print(f"upload failed: {up.status}", file=sys.stderr)
            return 1
        for rnd in range(1, args.rounds + 1):
            for wid in args.rows:
                fullrun.settle(f"qdrant round {rnd} {wid}")
                rows.append(commit_ab.measure(table[wid], uri, out, "qdrant", rnd))
    finally:
        fullrun.stop_qdrant()
        fullrun.wipe_qdrant_storage()
        commit_ab.write_summary(out, rows, ["qdrant"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
