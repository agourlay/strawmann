#!/usr/bin/env python3
"""Calibrate a cost-aware scan-or-walk choice for Qdrant's filtered search.

Qdrant picks scoring a filter's matches (scan) over walking the graph when the
estimated match count is under `full_scan_threshold`, a byte-based constant
that ignores `ef`. Measured 2026-10-06, walk cost grows about as ef^0.5 and the
break-even is far from proportional to vector bytes. This runs the W12 ladders
on one Qdrant binary with the choice forced each way, over widths and
selectivities, through `qdrant_build_ab.py`; `--fit` reads the results back.

Usage:
    planner_calibration.py --binary ~/.cache/strawmann/qdrant-ab/bin/qdrant-base
    planner_calibration.py --fit
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DATASETS = {"sift1m": 128, "laion-small-clip": 512, "dbpedia-openai-100K-1536-angular": 1536}
#: (W12_KEYWORDS, W12_MATCH_ANY): sel1 is 1/K of the points, sel10 1-(1-1/K)^ANY.
FILTERS = ((200, 6), (100, 10))
ARMS = {"walk": 10, "scan": 100_000_000}  # W12_FULL_SCAN_THRESHOLD_KB
EFS = (32, 128, 512)


def selectivities(keywords: int, match_any: int) -> dict[str, float]:
    return {"sel1": 1 / keywords, "sel10": 1 - (1 - 1 / keywords) ** match_any}


def plan() -> list[dict]:
    """Every run, in order: dataset, filter configuration, arm, and its env."""
    runs = []
    for ds in DATASETS:
        for keywords, match_any in FILTERS:
            for arm, threshold in ARMS.items():
                runs.append({
                    "dataset": ds, "arm": arm, "keywords": keywords, "match_any": match_any,
                    "tag": f"cal-k{keywords}a{match_any}-{arm}",
                    "env": {"W12_FULL_SCAN_THRESHOLD_KB": str(threshold), "W12_ACORN": "0",
                            "W12_KEYWORDS": str(keywords), "W12_MATCH_ANY": str(match_any)},
                })
    return runs


def rows() -> list[str]:
    return ["W12-upload", *(f"W12-{g}-ef{ef}" for g in ("sel1", "sel10") for ef in EFS)]


def run(binary: Path, server_cpus: str, client_cpus: str) -> int:
    rc = 0
    for r in plan():
        log = RESULTS / f"{r['tag']}-{r['dataset']}.log"
        cmd = [sys.executable, str(HERE / "qdrant_build_ab.py"), "--server-cpus", server_cpus,
               "--client-cpus", client_cpus, "--arm", f"{r['arm']}={binary}",
               "--dataset", r["dataset"], "--rows", *rows(), "--reps", "1",
               "--tag", r["tag"], "--settle-timeout", "20"]
        with log.open("w") as fh:
            code = subprocess.run(cmd, env={**os.environ, **r["env"]}, stdout=fh,
                                  stderr=subprocess.STDOUT, cwd=HERE).returncode
        print(f"{r['dataset']} k={r['keywords']} any={r['match_any']} {r['arm']}: rc={code}",
              flush=True)
        rc |= code
    return rc


def fit(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of log(y) on log(x): the exponent of y ~ x^a."""
    pts = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den if den else None


def break_even(walk_qps: dict[int, float], scan_qps: float) -> float | None:
    """The ef at which walking costs what scanning does, interpolated in log ef;
    None when one side wins at every ef measured."""
    efs = sorted(walk_qps)
    for lo, hi in itertools.pairwise(efs):
        a, b = walk_qps[lo] / scan_qps, walk_qps[hi] / scan_qps
        if (a - 1) * (b - 1) <= 0 and a != b:
            t = math.log(a) / (math.log(a) - math.log(b))
            return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
    return None


def summary_of(tag: str, dataset: str) -> dict[str, dict]:
    f = RESULTS / f"{tag}-{dataset}" / "summary.json"
    return {s["row"]: s for s in json.loads(f.read_text())["summary"]} if f.exists() else {}


def report() -> None:
    for ds, dim in DATASETS.items():
        for keywords, match_any in FILTERS:
            walk = summary_of(f"cal-k{keywords}a{match_any}-walk", ds)
            scan = summary_of(f"cal-k{keywords}a{match_any}-scan", ds)
            for grade, sel in selectivities(keywords, match_any).items():
                w = {ef: walk.get(f"W12-{grade}-ef{ef}", {}).get("qps") for ef in EFS}
                w = {ef: q for ef, q in w.items() if q}
                s = [scan.get(f"W12-{grade}-ef{ef}", {}).get("qps") for ef in EFS]
                s = [q for q in s if q]
                if not w or not s:
                    continue
                sq = sum(s) / len(s)  # the scan does not depend on ef
                alpha = fit([(ef, 1 / q) for ef, q in w.items()])
                be = break_even(w, sq)
                ratios = " ".join(f"ef{ef}:{q / sq:.2f}" for ef, q in sorted(w.items()))
                print(f"d={dim:<5} sel={sel:6.2%}  scan {sq:8,.0f} q/s  walk/scan {ratios}  "
                      f"walk ~ ef^{alpha:.2f}  break-even ef {be:.0f}" if alpha and be else
                      f"d={dim:<5} sel={sel:6.2%}  scan {sq:8,.0f} q/s  walk/scan {ratios}  "
                      f"walk ~ ef^{alpha if alpha is None else round(alpha, 2)}  no crossover in range")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--server-cpus", default="4-11")
    ap.add_argument("--client-cpus", default="0-3")
    ap.add_argument("--fit", action="store_true", help="read the results back and fit")
    args = ap.parse_args(argv[1:])
    if args.fit:
        report()
        return 0
    if args.binary is None:
        ap.error("--binary is required to run")
    return run(args.binary.expanduser(), args.server_cpus, args.client_cpus)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
