#!/usr/bin/env python3
"""The headline tier end to end: verify -> convert -> ground truth -> recall.

Replaces `bench/headline-run.sh`.

§4.2 makes `dbpedia-openai-1m` the headline dataset, "the dimensionality Qdrant
users actually run", and marks its ground truth recompute-only, so every step
below has to happen before a single recall number exists.

The steps are cached: each checks for its output and skips if present, so a
re-run after a failure costs seconds rather than the half hour the fp64 oracle
takes at d=1536.

Usage:
    headline.py [--engine http://localhost:6334] [--queries 10000]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
import recall

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
CONF = ROOT / "conformance"


#: §4.2's headline tier. Named once here and resolved through `paths` below,
#: so the metric this script computes ground truth under is the metric every
#: reader of that ground truth will score against.
DATASET = "dbpedia-openai-1m"


def tier() -> Path:
    """The headline tier's directory, resolved on call so `--data-dir` reaches it."""
    return paths.DATA / DATASET


def step(n: int, title: str) -> None:
    print(f"\n=== {n}. {title} ===", flush=True)


def cargo(*args: object, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["cargo", "run", "--release", "--quiet", "--", *[str(a) for a in args]]
    return subprocess.run(cmd, cwd=CONF, text=True,
                          capture_output=capture, check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://localhost:6334")
    ap.add_argument("--queries", type=int, default=10_000,
                    help="held-out query count (§4.2's split)")
    ap.add_argument("--eval-queries", type=int, default=2_000)
    ap.add_argument("--collection", default="dbpedia")
    ap.add_argument("--label", default="strawmann",
                    help="result directory the recall sweep belongs to")
    paths.add_data_argument(ap)
    args = ap.parse_args()
    paths.use_data_dir(args.data_dir)

    # From `paths`, not spelled out again: it is the same trio `fullrun.py`
    # and `recall.py` resolve, and this file writing its own copy is how the
    # layout ends up defined in two places that can disagree. This script is
    # what *creates* those files, so a divergence here produces a corpus the
    # rest of the harness cannot find.
    base, queries, gt = paths.dataset(DATASET)

    step(1, "verify the fetch against upstream digests (§4.2)")
    r = subprocess.run([str(ROOT / "conformance/datasets/datasets.py"),
                        "--verify", DATASET])
    if r.returncode != 0:
        print(f"fetch incomplete; run conformance/datasets/datasets.py fetch {DATASET}",
              file=sys.stderr)
        return 1

    step(2, f"parquet -> fbin, holding out {args.queries} queries (§9)")
    if base.exists() and queries.exists():
        print("  present, skipping")
    else:
        # The held-out split is what makes recall meaningful: a query drawn from
        # the base set is its own nearest neighbour at distance 0.
        cargo("convert-parquet", "--in-dir", tier(),
              "--out-base", base, "--out-queries", queries,
              "--n-queries", args.queries)

    step(3, "fp64 exhaustive ground truth, cosine, k=100 (§8.2 tier 1)")
    if gt.exists():
        print("  present, skipping")
    else:
        # ~12x SIFT1M's oracle work at d=1536; roughly half an hour.
        gt.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        cargo("oracle", "--base", base, "--queries", queries,
              "--metric", paths.metric(DATASET), "--k", 100,
              "--dataset", DATASET, "--out", gt)
        print(f"  computed in {time.monotonic() - t0:.0f}s")

    step(4, "exact control, must be 1.0000 / 1.0000 / MRDE 0")
    # Validates the whole chain *except* the graph: conversion, the split, the
    # wire, ingest normalisation, §8.3's short-circuit at the dimension where it
    # is most fragile, the d=1536 kernel, and the eps-aware recall computation.
    # If this row is not perfect, the sweep below is measuring the harness.
    cargo("relevance", "--engine", args.engine, "--base", base, "--queries", queries,
          "--ground-truth", gt, "--metric", paths.metric(DATASET), "--limit", 10,
          "--limit-queries", 200, "--collection", args.collection, "--exact", "--ef", 0)

    step(5, "recall vs ef (W10's harness side)")
    # Written as JSON as well as printed: §7.4 forbids a qps number without its
    # recall, and the harness can only honour that if the recall is somewhere it
    # can read. `--label` names the result directory it belongs to.
    # Keyed by dataset and collection (`recall.py`): this used to be
    # `recall.json`, the same name SIFT's `bench2` sweep used, so a label that
    # had run both held whichever finished last and the SIFT rows joined a
    # dbpedia cosine curve on `ef`.
    recall_json = recall.recall_path(args.label, DATASET, args.collection)
    recall_json.parent.mkdir(parents=True, exist_ok=True)
    cargo("relevance", "--engine", args.engine, "--base", base, "--queries", queries,
          "--ground-truth", gt, "--metric", paths.metric(DATASET), "--limit", 10,
          "--limit-queries", args.eval_queries, "--collection", args.collection,
          "--skip-upload", "--json", recall_json, "--ef", 16, 32, 64, 128, 256, 512)
    recall.stamp(recall_json, DATASET, args.collection, paths.metric(DATASET))
    print(f"\nrecall written to {recall_json}; readers join it on "
          f"(dataset, collection, ef), so it cannot be mistaken for a SIFT sweep.")

    step(6, "scroll (W13)")
    cargo("scroll-check", "--engine", args.engine,
          "--collection", args.collection, "--page", 1000)

    print("\nAll numbers above are development-grade: bench/setup.py check fails on")
    print("this host (§7.1), and Qdrant was not run, so nothing here is a comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
