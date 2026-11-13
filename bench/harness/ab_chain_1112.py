#!/usr/bin/env python3
"""The memory-stall arms for findings 74 and 75, on one build, in one quiet stretch.

1. SQ8 on dbpedia (findings 74): stage 2 rescores up to `ef` fp32 rows with
   no prefetch. `--rescore-prefetch first` and `whole` against `off`, and
   `--workers 8` (the I/O thread sharing a worker's core) against 7, on the
   W6 ladder. dbpedia-100K rather than 1M: the same d=1536 rows at a tenth
   of the upload.
2. fp32 traversal on laion (findings 75): W4 pulls Qdrant's DRAM per query at
   IPC 0.51 against 0.79, and rows past 512 bytes are prefetched by their
   first line. `--probe-prefetch-bytes 512` and `2048` (laion's whole row)
   against the rule, and `--workers 8`, on W3, W4, W5 and W10-ef128.
3. The same traversal knob on dbpedia-100K's 6 KiB rows (IPC 0.30 on 1M W4),
   at 512 bytes only: the whole row lost on h-and-m (decisions.md,
   2026-09-29), and `--workers 8` beside it.
4. findings 62 (`reclaim_ab.py`): dbpedia-1m's W10 with the collection as
   written, then reclaimed through the engine's cgroup and read back.
5. findings 10: dbpedia-1m's W9 with and without `--exact-gather`, separate
   requests' exact queries sharing one pass split over their workers. It
   applies from 4 GiB, so the 1M corpus: about 8 minutes a session, 50 in all.
6. The quantized `Query` union (ee3dbcc, 16accd6) against its parent
   (154628e), on dbpedia-100K's W6 ladder: a refactor of the hot path's
   score dispatch, which must read flat.
7. findings 8: dbpedia-1m's W11-steady and W11 with and without
   `--prune-memo` (a back-link re-prune from the list's recorded selection).
   Each session's W2 upload also times the build, which the flag covers too.
8. findings 58 (`qdrant_rerun.py`): the pinned Qdrant's laion W4 and
   W10-ef128 first and last, so one binary is read hours apart on one night.
   Needs `QDRANT_BINARY` in the environment, as the night runs have it.

Development-grade, like `ab_chain_1110.py`, whose runner this reuses: no
Qdrant, no publication, every arm one commit. Scheduled, not run from a
session:

    systemd-run --user --on-calendar='YYYY-MM-DD HH:MM:SS' --unit=strawmann-ab-1112 \\
        -p WorkingDirectory=$PWD/bench/harness \\
        $PWD/bench/.venv/bin/python ab_chain_1112.py --commit <sha>
"""

from __future__ import annotations

import subprocess
import sys
import time

import commit_ab
import phase1_chain
from phase1_chain import CPUS, Step

DBPEDIA = "dbpedia-openai-100K-1536-angular"
QUERY_UNION_PARENT = "154628e"  # before the quantized Query union
QUERY_UNION = "16accd6"         # the union and QuantParams.use
LAION = "laion-small-clip"


def _step(name: str, tag: str, dataset: str, commit: str, upload: str, rows: list[str],
          arms: dict[str, str]) -> Step:
    """One `commit_ab.py` session: every arm `commit`, each with its flags."""
    flags = [x for arm, f in arms.items() if f for x in ("--flags", f"{arm}={f}")]
    return Step(name, [
        "commit_ab.py", *CPUS, "--dataset", dataset, "--reps", "3", "--tag", f"{tag}-{name}",
        "--upload", upload, "--rows", *rows,
        "--commits", *(f"{arm}={commit}" for arm in arms), *flags])


def steps(tag: str, commit: str) -> list[Step]:
    qdrant = ["qdrant_rerun.py", *CPUS]
    return [
        Step("qdrant-early", [*qdrant, "--tag", f"{tag}-qdrant-early"]),
        _step("sq8-dbpedia", tag, DBPEDIA, commit, "W6-upload", ["W6", "W6-ef256", "W6-ef512"],
              {"off": "", "first": "--rescore-prefetch first",
               "whole": "--rescore-prefetch whole", "w8": "--workers 8"}),
        _step("probe-laion", tag, LAION, commit, "W2", ["W3", "W4", "W5", "W10-ef128"],
              {"rule": "", "pf512": "--probe-prefetch-bytes 512",
               "pf2048": "--probe-prefetch-bytes 2048", "w8": "--workers 8"}),
        _step("probe-dbpedia", tag, DBPEDIA, commit, "W2", ["W4", "W10-ef128"],
              {"rule": "", "pf512": "--probe-prefetch-bytes 512", "w8": "--workers 8"}),
        # The build the steps above made, not the checkout's own.
        Step("reclaim-dbpedia", ["reclaim_ab.py", *CPUS, "--tag", f"{tag}-reclaim-dbpedia",
                                 "--binary", str(commit_ab.build_path(commit))]),
        _step("gather-dbpedia", tag, "dbpedia-openai-1m", commit, "W2", ["W9"],
              {"each": "", "gather": "--exact-gather"}),
        Step("query-union-dbpedia", [
            "commit_ab.py", *CPUS, "--dataset", DBPEDIA, "--reps", "3",
            "--tag", f"{tag}-query-union-dbpedia", "--upload", "W6-upload",
            "--rows", "W6", "W6-ef256", "W6-ef512",
            "--commits", f"before={QUERY_UNION_PARENT}", f"union={QUERY_UNION}"]),
        _step("prune-memo-dbpedia", tag, "dbpedia-openai-1m", commit, "W2",
              ["W11-steady", "W11"], {"full": "", "memo": "--prune-memo"}),
        Step("qdrant-late", [*qdrant, "--tag", f"{tag}-qdrant-late"]),
    ]


def main(argv: list[str]) -> int:
    def add_args(ap):
        ap.add_argument("--commit", help="the build every arm runs (default: HEAD)")

    def chain(args):
        commit = args.commit or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
        return steps(args.tag, commit)

    return phase1_chain.main_for(argv, __doc__, f"ab-{time.strftime('%m%d')}", chain,
                                 phase1_chain.no_bfb, add_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
