"""Reading a result directory's JSON files: one reader, one missing-file rule.

And `run_into`, the older drivers' one way of filling a result directory.

`run.json`, `noise.json`, `conformance.json` and the like are optional to
their readers: a label measured before a field existed, or a file a crashed
run never wrote, reads as `{}`. Row files are not optional to most callers,
so `rows_of` says None and the caller decides whether that is an error.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def read_json(path: Path) -> dict:
    """The JSON object at `path`, or `{}` when it is missing, unreadable, not
    JSON, or not an object."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def rows_of(path: Path) -> list[dict] | None:
    """A `rows.json` list, or None when the file is missing."""
    return json.loads(path.read_text()) if path.exists() else None


def run_into(argv: list[str], out: Path, root: Path, what: str, timeout: int,
             no_rows: str, extra_env: dict[str, str] | None = None) -> None:
    """Run `workloads.py run` (`argv`) with its results in `out`, or exit saying why.

    For `isa_sweep` and `qdrant_ab`: `RESULTS_DIR` and `STRAWMANN_ROOT` point
    the child at `out` and `root`, its output goes to `out/run.log`, a failure
    raises with the log's last six lines, and a run that wrote no `rows.json`
    raises `no_rows`. A failed arm used to leave an empty result directory
    while the driver carried on, so the only symptom was an empty table two
    steps later.
    """
    env = dict(os.environ, RESULTS_DIR=str(out), STRAWMANN_ROOT=str(root), **(extra_env or {}))
    r = subprocess.run(argv, env=env, stdout=(out / "run.log").open("w"),
                       stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0:
        tail = (out / "run.log").read_text().strip().splitlines()[-6:]
        raise SystemExit(f"{what} failed (exit {r.returncode}):\n  " + "\n  ".join(tail))
    if not (out / "rows.json").exists():
        raise SystemExit(no_rows)
