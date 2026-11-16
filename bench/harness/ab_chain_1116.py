#!/usr/bin/env python3
"""Night A's A/Bs (docs/backlog.md, Next nights), after the h-and-m pair.

1. The filtered walk (5ae0616: the two-hop walk bridges through rejected
   points, a tight filter is scanned) against its parent 4a13a33, on
   W12-sel10 on all four vector corpora and on W16 where the corpus ships
   filtered tests (laion, h-and-m). Expect laion W12-sel10 about 5% lower.
   `commit_ab` records throughput only: W16 recall on the new walk comes
   from the h-and-m pair and the 11-13 laion measurements.
2. The shared layer search (a98d0b1, `hnsw.Walk`, results proven identical)
   against its parent 790da01, on sift1m. It lands only if flat or better.
3. `--build-prefetch` repeated on dbpedia-openai-1m's W11-steady and W11,
   off against on, one build (1113: W11-steady +5.9%, W11 -2.5% under
   desktop load).

Development-grade, like `ab_chain_1112.py`: no Qdrant, no publication.
Scheduled, not run from a session:

    systemd-run --user --on-calendar='YYYY-MM-DD HH:MM:SS' --unit=strawmann-ab-1116 \\
        -p WorkingDirectory=$PWD/bench/harness \\
        $PWD/bench/.venv/bin/python ab_chain_1116.py --commit <sha>
"""

from __future__ import annotations

import subprocess
import sys
import time

import phase1_chain
from phase1_chain import CPUS, Step

WALK_PARENT = "4a13a33"   # before the bridge and the selective scan
WALK = "5ae0616"          # the bridge and the selective scan
LAYER_PARENT = "790da01"  # before the shared layer search
LAYER = "a98d0b1"         # hnsw.Walk, parked in a worktree branch

#: The vector corpora, and whether each ships filtered tests (W16).
CORPORA = (("sift1m", "sift1m", False),
           ("laion", "laion-small-clip", True),
           ("hnm", "h-and-m-2048-angular-filters", True),
           ("dbpedia", "dbpedia-openai-1m", False))
W12_ROWS = ["W12-sel10", "W12-sel10-ef256"]
W16_ROWS = ["W16-ef64", "W16-ef128", "W16-ef256"]


def _ab(name: str, tag: str, dataset: str, upload: str, rows: list[str],
        commits: list[str], flags: dict[str, str] | None = None) -> Step:
    """One `commit_ab.py` session, three reps."""
    extra = [x for arm, f in (flags or {}).items() for x in ("--flags", f"{arm}={f}")]
    return Step(name, [
        "commit_ab.py", *CPUS, "--dataset", dataset, "--reps", "3", "--tag", f"{tag}-{name}",
        "--upload", upload, "--rows", *rows, "--commits", *commits, *extra])


def steps(tag: str, commit: str) -> list[Step]:
    walk = [f"before={WALK_PARENT}", f"after={WALK}"]
    out = []
    for short, dataset, w16 in CORPORA:
        out.append(_ab(f"walk-w12-{short}", tag, dataset, "W12-upload", W12_ROWS, walk))
        if w16:
            out.append(_ab(f"walk-w16-{short}", tag, dataset, "W16-upload", W16_ROWS, walk))
    out.append(_ab("layer-sift1m", tag, "sift1m", "W2",
                   ["W3", "W4", "W10-ef128", "W11-steady"],
                   [f"base={LAYER_PARENT}", f"walk={LAYER}"]))
    out.append(_ab("build-prefetch-dbpedia", tag, "dbpedia-openai-1m", "W2",
                   ["W11-steady", "W11"], [f"off={commit}", f"on={commit}"],
                   {"on": "--build-prefetch"}))
    return out


def main(argv: list[str]) -> int:
    def add_args(ap):
        ap.add_argument("--commit", help="the build the --build-prefetch arms run "
                                         "(default: HEAD)")

    def chain(args):
        commit = args.commit or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
        return steps(args.tag, commit)

    return phase1_chain.main_for(argv, __doc__, f"ab-{time.strftime('%m%d')}", chain,
                                 phase1_chain.no_bfb, add_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
