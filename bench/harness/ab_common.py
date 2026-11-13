"""What the development A/B drivers share.

`commit_ab`, `w9_ab`, `qdrant_w4_probe`, `qdrant_build_ab`, `qdrant_rerun`
and `reclaim_ab` each start an engine, upload, settle and measure rows with
the `perf stat` sidecar, then summarise. The steps they do alike live here;
what each varies (its arms, its row tags, its summary's columns) stays in
the driver.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import perfstat

import procstat
import provenance
import workloads

#: bfb's flags every row passes, as `workloads.main` builds them.
COMMON = workloads.bfb_common()

#: `0-3,8` as `{0, 1, 2, 3, 8}`.
cpu_set = provenance.cpus_in


def prepare(dataset: str, client_cpus: str | None) -> None:
    """The dataset, and this process's CPUs.

    The load generator, its warm-ups and the perf sidecars are this process's
    children, so its affinity is theirs; the engine pins itself with
    `--pin --cpus`.
    """
    workloads.use_dataset(dataset)
    if client_cpus is not None:
        os.sched_setaffinity(0, cpu_set(client_cpus))


def warn_foreign(oom: bool = False, claude: bool = True) -> None:
    """Warned, not refused: the rows record their own foreign load, and the
    sessions are the user's."""
    if busy := procstat.builders_alive():
        tail = (", and memory pressure the engine can be oom-killed under" if oom else "")
        print(f"!! build processes alive ({', '.join(busy)}): foreign load every row "
              f"will record{tail}", flush=True)
    if not claude:
        return
    if found := [c for _, c in procstat.processes() if c == "claude"]:
        print(f"!! {len(found)} claude process(es) alive; their idle CPU is foreign load",
              flush=True)


def warn_other_engines() -> None:
    """`run_one` attaches its counters to the lowest-pid engine process, so
    a second one alive at measurement time would be the row's engine."""
    procs = procstat.engine_processes()
    if len(procs) > 1:
        print(f"  !! {len(procs)} engine processes alive "
              f"({', '.join(f'{c}:{p}' for p, c in procs)}); the row's counters "
              f"describe the lowest pid", flush=True)


def stop_perf(rec: subprocess.Popen, timeout_s: int) -> None:
    """SIGINT so perf writes its file, and a kill if it will not stop."""
    rec.send_signal(signal.SIGINT)
    try:
        rec.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        rec.kill()
        rec.wait(timeout=30)


def upload_ok(w: workloads.Workload, uri: str, results: Path) -> bool:
    """The session's upload row, said when it fails."""
    up = workloads.run_one(w, uri, results, COMMON)
    if up.status != workloads.Status.ok:
        print(f"upload failed: {up.status}", file=sys.stderr)
        return False
    return True


def session(start: Callable[[], object | None], stop: Callable[[object], None],
            upload: workloads.Workload, uri: str, upload_dir: Path,
            body: Callable[[], int | None]) -> int:
    """Start an engine, upload, run `body`, and stop the engine whatever happens.

    1 when the engine does not start or the upload fails, else what `body`
    returns (None for 0). `stop` is not called when the engine never started.
    """
    engine = start()
    if engine is None:
        return 1
    try:
        if not upload_ok(upload, uri, upload_dir):
            return 1
        return body() or 0
    finally:
        stop(engine)


def run_row(w: workloads.Workload, uri: str, results: Path, warmup: bool = True,
            during: contextlib.AbstractContextManager | None = None) -> dict:
    """One row with the `perf stat` sidecar, as a dict with the derived counters.

    `during` is entered around the row itself, for a sampler of its own.
    """
    results.mkdir(parents=True, exist_ok=True)
    warn_other_engines()
    with during or contextlib.nullcontext():
        res = workloads.run_one(w, uri, results, COMMON, perf="default", warmup=warmup)
    row = dataclasses.asdict(res)
    row.update(perfstat.derive(row))
    return row


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")


def write_summary(out: Path, doc: dict, text: str) -> None:
    """`summary.json`, `summary.txt`, and the text on stdout."""
    write_json(out / "summary.json", doc)
    (out / "summary.txt").write_text(text)
    print("\n" + text, flush=True)


def ratio(r: dict, num: str, den: str) -> float | None:
    return r[num] / r[den] if r.get(num) and r.get(den) else None


def per_q(r: dict, key: str) -> float | None:
    return ratio(r, key, "n_queries")


def median_of(rows: list[dict], f) -> float | None:
    """The median of `f` over the rows it gives a figure for."""
    vals = [v for r in rows if (v := f(r)) is not None]
    return statistics.median(vals) if vals else None
