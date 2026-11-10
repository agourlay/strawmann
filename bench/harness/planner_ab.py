#!/usr/bin/env python3
"""Refit strawmANN's filtered planner: each plan forced, on the same W12 ladder.

`handlers.filteredPlan` picks scanning a counted filter's matches or the
two-hop walk from a cost model fit to the 0929 pairs, whose W12 ran 2 requests
in flight. W12 now runs W4's client, and the 1102 pairs show the model
mispicking under it: laion W12-sel10 0.63x and h-and-m W12-sel10-ef512 0.52x
against Qdrant. This runs the W12-sel10 ef ladder once per plan, through
`commit_ab.py` with `--filtered-plan` forced, on each corpus; `--fit` reads
the summaries back and sets the measured break-even width beside the model's.

Throughput only: a dev A/B records no recall. The scan is exact, so a two-hop
width that wins on q/s still has to clear the recall the conformance pass
measures before the model may pick it.

    planner_ab.py --server-cpus 4-11 --client-cpus 0-3 [--datasets sift1m ...]
    planner_ab.py --fit [--tag planner-ab-1103]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import commit_ab
import fullrun
import planner_calibration

import workloads

DATASETS = {"sift1m": 128, "laion-small-clip": 512, "h-and-m-2048-angular-filters": 2048,
            "dbpedia-openai-100K-1536-angular": 1536}
PLANS = ("scan", "two_hop")
EFS = (32, 64, 128, 256, 512)
#: `hnsw_m * 2` at the engine's default m of 16, which W12 does not override.
M0 = 32


def rows() -> list[str]:
    return [f"W12-sel10-ef{ef}" for ef in EFS]


def argv(dataset: str, commit: str, tag: str, server_cpus: str, client_cpus: str,
         reps: int) -> list[str]:
    """One corpus's `commit_ab.py` run: both plans on one build."""
    return ["commit_ab.py", "--server-cpus", server_cpus, "--client-cpus", client_cpus,
            "--dataset", dataset, "--reps", str(reps), "--tag", f"{tag}-{dataset}",
            "--upload", "W12-upload", "--rows", *rows(),
            "--commits", *(f"{p}={commit}" for p in PLANS),
            *(x for p in PLANS for x in ("--flags", f"{p}=--filtered-plan {p}"))]


def model_break_even(selected: int, dim: int, m0: int = M0) -> float:
    """The ef at which `filteredPlan` flips: `selected d = 560 ef m0 + 350,000`.

    The same constants as handlers.zig, which this cannot import: change both.
    """
    return (selected * dim - 350_000) / (560 * m0)


def selected_rows() -> int:
    """W12-sel10's expected match count over the corpus in use."""
    return round(workloads.w12_n() * (1 - (1 - 1 / workloads.W12_KEYWORDS)
                                      ** workloads.W12_MATCH_ANY))


def ladder(summary: list[dict]) -> dict[str, dict[int, float]]:
    """`{plan: {ef: q/s}}` from a `commit_ab` summary."""
    out: dict[str, dict[int, float]] = {}
    for s in summary:
        if s["commit"] in PLANS and s["row"].startswith("W12-sel10-ef") and s.get("qps"):
            out.setdefault(s["commit"], {})[int(s["row"].rsplit("ef", 1)[1])] = s["qps"]
    return out


def fit_line(dataset: str, dim: int, got: dict[str, dict[int, float]], selected: int) -> str:
    scan, walk = got.get("scan", {}), got.get("two_hop", {})
    if not scan or not walk:
        return f"{dataset}: measured {', '.join(sorted(got)) or 'no plan'}, not both"
    sq = sum(scan.values()) / len(scan)  # the scan does not depend on ef
    ratios = " ".join(f"ef{ef}:{q / sq:.2f}" for ef, q in sorted(walk.items()))
    be = planner_calibration.break_even(walk, sq)
    model = model_break_even(selected, dim)
    measured = f"{be:.0f}" if be else "none in range"
    picks = " ".join(f"ef{ef}:{'two_hop' if ef < model else 'scan'}"
                     f"{'' if (ef < model) == (walk[ef] > sq) else '(wrong)'}"
                     for ef in sorted(walk))
    return (f"{dataset} d={dim} selected~{selected:,}: scan {sq:,.0f} q/s, two_hop/scan "
            f"{ratios}; break-even ef measured {measured}, model {model:.0f}; model picks "
            f"{picks}")


def report(tag: str) -> int:
    for ds, dim in DATASETS.items():
        f = fullrun.RESULTS / f"{tag}-{ds}" / "summary.json"
        if not f.exists():
            print(f"{ds}: no {f}")
            continue
        workloads.use_dataset(ds)
        print(fit_line(ds, dim, ladder(json.loads(f.read_text())["summary"]),
                       selected_rows()))
    print("throughput only: check a two_hop pick against the conformance recall")
    return 0


def head_commit() -> str:
    """The checkout's commit, by hash: `commit_ab` keeps a worktree per name, so
    one called `HEAD` would build whatever HEAD was when it was made."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=fullrun.ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def main(args_in: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", default="4-11")
    ap.add_argument("--client-cpus", default="0-3")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--commit", help="the build to measure (default: this checkout's HEAD)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tag", default=f"planner-ab-{time.strftime('%m%d')}")
    ap.add_argument("--fit", action="store_true", help="read the results back and fit")
    args = ap.parse_args(args_in[1:])
    if args.fit:
        return report(args.tag)
    commit = args.commit or head_commit()
    rc = 0
    for ds in args.datasets:
        fullrun.say(f"planner A/B on {ds} at {commit}")
        code = commit_ab.main(argv(ds, commit, args.tag, args.server_cpus,
                                   args.client_cpus, args.reps))
        print(f"{ds}: rc={code}", flush=True)
        rc |= code
    report(args.tag)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
