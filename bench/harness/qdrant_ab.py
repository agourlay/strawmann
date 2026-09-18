#!/usr/bin/env python3
"""Run the workload table against two Qdrant builds and render a verdict.

    qdrant_ab.py qdrant/qdrant:v1.19.0 qdrant/qdrant:dev
    qdrant_ab.py <img-a> <img-b> --rows W3 W4 W6 --reps 3
    qdrant_ab.py <img-a> <img-b> --labels v1.19 dev

This is the handoff `docs/validation.md` asks for — "one regression a Qdrant
developer already understands" — so the remaining work is supplying the two
builds rather than working out how to drive anything.

For each image, alternating between them across repetitions: stop and remove
any previous container and **wipe the storage directory**, start the image on a
fresh volume, wait for it to answer and record the version it reports, run the
requested rows, stop it. Then `regression.py` renders a per-row verdict against
the measured noise floor.

The wipe matters because two Qdrant versions may write incompatible segment
formats, and a version inheriting another's storage spends its first minutes
migrating — measuring that is measuring the migration. Each arm therefore pays
a full ingest.

A verdict says whether a difference exceeds this host's measured run-to-run
spread. It does not say the difference is attributable to any particular
change: two builds differ in many ways at once, which is why the comparison is
most useful when someone already knows what changed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
STORAGE = paths.QDRANT_STORAGE
CONTAINER = "strawmann-qdrant-ab"
REST_PORT, GRPC_PORT = 6336, 6335

#: Rows worth comparing between two engine builds. Upload rows come along
#: because each arm starts from empty storage and has to ingest.
DEFAULT_ROWS = ["W3", "W4", "W6", "W9", "W13"]
SETUP_ROWS = ["W0-upload", "W2", "W6-upload"]


def sh(argv: list[str], timeout: int = 300) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def stop_container() -> None:
    sh(["docker", "rm", "-f", CONTAINER], 120)


def wipe_storage() -> None:
    """Empty the storage directory.

    Done from inside a container rather than with `shutil.rmtree`: the Qdrant
    image runs as root, so everything it writes is root-owned on the host and a
    user-level delete fails with EACCES partway through, leaving a half-wiped
    directory that the next arm would inherit. Deleting the *contents* rather
    than the directory keeps the bind-mount path stable.
    """
    STORAGE.mkdir(parents=True, exist_ok=True)
    code, out = sh(["docker", "run", "--rm", "-v", f"{STORAGE}:/s",
                    "alpine:3", "sh", "-c", "rm -rf /s/..?* /s/.[!.]* /s/* 2>/dev/null; true"], 180)
    if code != 0:
        raise SystemExit(f"could not wipe {STORAGE} via a container:\n{out.strip()}\n"
                         f"The Qdrant image runs as root, so its files are root-owned; "
                         f"remove the directory manually and re-run.")
    leftover = list(STORAGE.iterdir())
    if leftover:
        raise SystemExit(f"{STORAGE} still holds {len(leftover)} entr(y|ies) after the wipe: "
                         f"{[p.name for p in leftover[:5]]}. Refusing to run, because the "
                         f"next arm would inherit another version's segments.")


def start(image: str) -> str:
    """Start the image on a clean volume and return the version it reports."""
    stop_container()
    wipe_storage()
    # `--network host` and the ports set inside the container, rather than
    # `-p`: published ports do not reach the container on every host this runs
    # on, and the failure is a 120s timeout waiting for a server that started
    # fine. `fullrun.py` has always used host networking; this did not, so this
    # script could time out on a box where the full run works.
    #
    # DEFAULT_SEGMENT_NUMBER matches `fullrun.py` for the reason `workloads.py`
    # records beside the threshold constants: it set the variable and this did
    # not, so the two drivers measured different Qdrant configurations under the
    # same row ids.
    code, out = sh(["docker", "run", "-d", "--name", CONTAINER,
                    "--network", "host",
                    "-e", f"QDRANT__SERVICE__HTTP_PORT={REST_PORT}",
                    "-e", f"QDRANT__SERVICE__GRPC_PORT={GRPC_PORT}",
                    "-e", "QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER=1",
                    "-v", f"{STORAGE}:/qdrant/storage",
                    image], 300)
    if code != 0:
        raise SystemExit(f"could not start {image}:\n{out.strip()}")

    deadline = time.time() + 120
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{REST_PORT}/", timeout=5) as r:
                got = json.loads(r.read())
                # Record what the server says it is, not what the tag claims. A
                # tag is mutable; the reported version is what actually ran, and
                # it is what belongs beside the numbers.
                return str(got.get("version", "unknown"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            last = str(e)
            time.sleep(2)
    stop_container()
    raise SystemExit(f"{image} did not answer on :{REST_PORT} within 120s ({last})")


def wait_quiet(limit_pct: int = 10, cap_s: int = 300) -> int:
    cores = os.cpu_count() or 1
    waited = 0
    while waited < cap_s:
        if float(Path("/proc/loadavg").read_text().split()[0]) / cores * 100 <= limit_pct:
            return waited
        time.sleep(15)
        waited += 15
    return waited


def run_rows(label: str, rows: list[str], out_dir: Path,
             extra_env: dict[str, str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RESULTS_DIR=str(out_dir), STRAWMANN_ROOT=str(ROOT),
               **(extra_env or {}))
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "bench/harness/workloads.py"),
         "run", f"http://localhost:{GRPC_PORT}", label, *SETUP_ROWS, *rows],
        env=env, stdout=(out_dir / "run.log").open("w"),
        stderr=subprocess.STDOUT, timeout=10800)
    if r.returncode != 0:
        tail = (out_dir / "run.log").read_text().strip().splitlines()[-6:]
        raise SystemExit(f"{label} failed (exit {r.returncode}):\n  " + "\n  ".join(tail))
    if not (out_dir / "rows.json").exists():
        raise SystemExit(f"{label} wrote no rows.json")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="A/B two Qdrant builds.")
    ap.add_argument("image_a")
    ap.add_argument("image_b")
    ap.add_argument("--labels", nargs=2, metavar=("A", "B"),
                    help="result-set names (default derived from the tags)")
    ap.add_argument("--rows", nargs="*", help=f"default: {' '.join(DEFAULT_ROWS)}")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="leave the last container running")
    # A/B a *configuration*, not only a build. The collection settings come from
    # `workloads.py`'s environment, so the two arms could only ever differ by
    # image — and "the two runs differed in the thresholds and also in the hour
    # they ran" is not an isolated comparison. These apply per arm, to the
    # workload subprocess, so the alternation below controls the rest.
    ap.add_argument("--env-a", nargs="*", default=[], metavar="K=V",
                    help="environment for arm A's rows, e.g. INDEXING_THRESHOLD_KB=1000")
    ap.add_argument("--env-b", nargs="*", default=[], metavar="K=V",
                    help="environment for arm B's rows")
    paths.add_data_argument(ap)
    args = ap.parse_args(argv[1:])
    # Exported, so the `workloads.py` this spawns per image reads the same root.
    paths.use_data_dir(args.data_dir)

    if not shutil.which("docker"):
        raise SystemExit("docker not found")
    rows = args.rows or DEFAULT_ROWS
    labels = args.labels or [img.rsplit(":", 1)[-1].replace("/", "-")
                             for img in (args.image_a, args.image_b)]
    if labels[0] == labels[1]:
        raise SystemExit(f"both images resolve to the label {labels[0]!r}; pass --labels")

    def as_env(pairs: list[str]) -> dict[str, str]:
        out = {}
        for kv in pairs:
            if "=" not in kv:
                raise SystemExit(f"--env-a/--env-b take K=V, got {kv!r}")
            k, v = kv.split("=", 1)
            out[k] = v
        return out

    envs = [as_env(args.env_a), as_env(args.env_b)]
    arms = list(zip(labels, [args.image_a, args.image_b], envs))
    print(f"{len(arms)} image(s) x {args.reps} rep(s) x {len(rows)} row(s)")
    print(f"rows: {' '.join(rows)}")
    print(f"storage: {STORAGE} (wiped between arms)\n")

    versions: dict[str, str] = {}
    try:
        for rep in range(1, args.reps + 1):
            # Alternate rather than running each image to completion: a machine
            # that warms up over an hour would otherwise charge the drift to
            # whichever image happened to run last.
            for label, image, extra in arms:
                waited = wait_quiet()
                ver = start(image)
                versions.setdefault(label, ver)
                cfg = ("  " + " ".join(f"{k}={v}" for k, v in extra.items())) if extra else ""
                print(f"  [rep {rep}] {label:<12} {image}  reports {ver} "
                      f"(quiet after {waited}s){cfg}", flush=True)
                out = ROOT / "bench/results" / (label if args.reps == 1
                                                else f"{label}-rep{rep}")
                run_rows(label, rows, out, extra)
    finally:
        if not args.keep:
            stop_container()

    print("\nversions reported by the running servers:")
    for label, ver in versions.items():
        print(f"  {label:<12} {ver}")

    if args.reps > 1:
        # One label's reps only: a floor is one engine measured against
        # itself, and a glob over both labels would fold two builds' spread
        # into one number.
        print(f"\n{args.reps} repetitions written as <label>-rep<n>. Build a noise "
              f"floor from them with:\n"
              f"  bench/harness/regression.py --measure-noise "
              f"'bench/results/{labels[0]}-rep*'")
        return 0

    print("\nverdict:\n")
    r = subprocess.run([str(ROOT / "bench/harness/regression.py"), *labels])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
