#!/usr/bin/env python3
"""An unattended publication run, scheduled for a quiet hour.

    python3 bench/harness/nightrun.py 2026-09-10 sift1m
    systemd-run --user --on-calendar='2026-09-10 04:00:00' \\
        --unit=strawmann-night-20260910 -p WorkingDirectory=$PWD \\
        /usr/bin/python3 bench/harness/nightrun.py 2026-09-10 sift1m

Why a systemd --user timer and not a Claude session's cron: the §7.1 gate
counts the CLI's idle CPU as foreign load (findings 44), so the run must not
need a session open. Why a waiter: the gate refuses a busy host with "refusing
to run", and asking again every five minutes is what gets a run admitted on a
laptop that also builds things. Any other failure is not retried: a run that
died mid-corpus must be read, not repeated.

Fresh labels are stamped with the date so a night never merges into a
published set. `--rps-reference` is resolved from the previous pair of the
same family, since a fresh label gives `auto` nothing to read and the fallback
would refuse the cross-engine latency read.

Afterwards a headless `claude -p`, restricted to read-only tools, writes
`analysis.md` beside the log: every warning, refusal and note, with its cause.
The prompt is `nightrun-analysis.md` next to this file.

This was `nightrun.sh`, 224 lines of bash whose logic was tested only through
two read-only flags. Its bugs were all in that logic: a resolver's refusal
sentence captured as the reference number (the helper ran `python3 -c` and
kept the last line of stdout), two launches sharing one log, and a re-render
overwriting the run's report. Here the resolver is called in-process and each
step is a function a test can reach.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: Exit codes a caller can tell apart. 96 and 97 refuse before anything runs;
#: 98 gave up waiting for the engine ports.
EXIT_LOG_EXISTS = 96
EXIT_NO_QDRANT = 97
EXIT_PORTS_BUSY = 98

#: The engine ports a leftover engine would hold (Qdrant REST and gRPC, and
#: the strawmANN conformance port).
ENGINE_PORTS = (6333, 6334, 6344)

#: Seconds between asks, and how many asks: two hours of asking the gate.
WAIT_S = 300
MAX_TRIES = int(os.environ.get("MAX_TRIES", 24))

#: What the analysis may run. Read-only by construction: no editor, no write.
ANALYSIS_TOOLS = ",".join([
    "Read", "Grep", "Glob", "Bash(cat:*)", "Bash(grep:*)", "Bash(head:*)",
    "Bash(tail:*)", "Bash(ls:*)", "Bash(wc:*)", "Bash(sed -n:*)", "Bash(python3:*)",
    "Bash(uv run:*)", "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)",
    "Bash(jq:*)", "Bash(ps:*)", "Bash(pgrep:*)"])

#: A dataset's label family. Anything else is `<dataset>-perf`.
FAMILIES = {"sift1m": "sift-perf",
            "dbpedia-openai-100K-1536-angular": "dbp100k-perf",
            "dbpedia-openai-1m": "dbp1m-perf"}


def child_env() -> dict[str, str]:
    """The environment every child runs in.

    systemd --user starts us with a minimal PATH, so uv, cargo, claude and the
    pyenv interpreter `fullrun.py` has always run under are put back in front.
    """
    home = os.environ["HOME"]
    path = ":".join([f"{home}/.local/bin", f"{home}/.pyenv/shims", f"{home}/.cargo/bin",
                     "/usr/local/bin", "/usr/bin", "/bin"])
    return {**os.environ, "PATH": path}


def family(dataset: str) -> str:
    return FAMILIES.get(dataset, f"{dataset}-perf")


def labels(date: str, dataset: str) -> tuple[str, str]:
    """This night's two labels, `sm-<family>-MMDD` and `qd-<family>-MMDD`."""
    tag = date[5:7] + date[8:10]
    fam = family(dataset)
    return f"sm-{fam}-{tag}", f"qd-{fam}-{tag}"


def prev_pair(results: Path, dataset: str, own: str) -> str | None:
    """The newest previous strawmANN label of this family, or None.

    Two bugs lived here, both silent. The glob required the date to follow the
    family directly, and every published label carries `rel-` before it
    (`sm-dbp1m-perf-rel-0903`), so nothing matched and the open-loop arms each
    used their own engine's saturation, which makes the report refuse the
    cross-engine latency read. And the sort was lexicographic, so `rel-0903`
    beat `0910` on `r` > `0`. Ordered by the trailing MMDD; a `-repN` pass
    directory does not end in one and is not a pair.
    """
    fam = family(dataset)
    found = []
    for p in results.glob(f"sm-{fam}-*"):
        m = re.search(r"-(\d{4})$", p.name)
        if p.is_dir() and m and p.name != own:
            found.append((int(m.group(1)), p.name))
    return max(found)[1] if found else None


def resolve_ref(prev: str, resolver: Callable | None = None,
                perf_set: str = "default") -> int | None:
    """The previous pair's reference rate, or None.

    `fullrun.resolve_rps_reference` compares a candidate against its *own*
    module global `PERF_SET`, which is None until `fullrun` parses its
    arguments, so it is set to what this run passes; otherwise every
    perf-measured reference reads as measured with the wrong instrument. A
    refusal is printed and returned as None, or raised as ValueError; either
    way there is no number, and nothing printed can be taken for one.
    """
    if resolver is None:
        import fullrun
        fullrun.PERF_SET = perf_set
        resolver = fullrun.resolve_rps_reference
    try:
        v = resolver("auto", [prev, "qd-" + prev.removeprefix("sm-")])
    except ValueError:
        return None
    return round(v) if v else None


class Log:
    """`night.log`, one timestamped line per event, appended."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        with self.path.open("a") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def git(args: list[str], cwd: Path) -> str | None:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def qdrant_provenance(binary: Path) -> str:
    """The log line for the Qdrant binary: path, sha256 and the checkout's commit.

    The checkout's HEAD is not evidence about the binary. On 2026-09-23 this
    line read `commit=63c6a797d` for a binary built at an earlier commit, and
    `run.json` carried the same claim. When the binary is older than the
    commit, the line says so.
    """
    d = binary.parent
    commit = git(["rev-parse", "--short", "HEAD"], d) or "?"
    predates = ""
    if commit != "?":
        head_ts = int(git(["log", "-1", "--format=%ct"], d) or 0)
        if binary.stat().st_mtime < head_ts:
            predates = " (BINARY PREDATES THIS COMMIT; sha256 is the identity that holds)"
    return f"qdrant binary {binary} sha256={sha256_prefix(binary)} commit={commit}{predates}"


def port_bound(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def is_gate_refusal(output: str) -> bool:
    """The §7.1 gate refused before measuring anything: worth asking again.

    "refusing to run" is the gate's phrase, and a refusal after pass 1 began
    is a different failure that happens to use it.
    """
    return "refusing to run" in output and not re.search(r"^=== pass 1", output, re.M)


def run_until_admitted(cmd: list[str], night: Path, log: Log, *,
                       max_tries: int = MAX_TRIES, wait_s: float = WAIT_S,
                       ports_busy: Callable[[], bool] | None = None,
                       run: Callable[[list[str], Path], int] | None = None,
                       sleep: Callable[[float], None] = time.sleep) -> int:
    """Run `fullrun.py`, asking again while the gate refuses or a port is held.

    Every other failure is returned at once: a run that died mid-corpus must
    be read, not repeated.
    """
    ports_busy = ports_busy or (lambda: any(port_bound(p) for p in ENGINE_PORTS))
    run = run or _run_to_file
    for attempt in range(1, max_tries + 1):
        if ports_busy():
            log(f"attempt {attempt}: an engine port "
                f"({'/'.join(map(str, ENGINE_PORTS))}) is already bound; waiting 5 min")
            if attempt == max_tries:
                log(f"EXIT={EXIT_PORTS_BUSY} (ports stayed busy)")
                return EXIT_PORTS_BUSY
            sleep(wait_s)
            continue
        log(f"=== attempt {attempt} ===")
        out = night / f"fullrun_attempt{attempt}.out"
        rc = run(cmd, out)
        log(f"attempt {attempt}: fullrun.py exited {rc} (output in {out})")
        if rc == 0:
            log("EXIT=0")
            return 0
        if not is_gate_refusal(out.read_text(errors="replace")):
            log(f"EXIT={rc} (not a gate refusal; not retried)")
            return rc
        log(f"gate refused (attempt {attempt}); retrying in 5 min")
        if attempt == max_tries:
            log(f"EXIT={rc} (gave up after {attempt} refusals)")
            return rc
        sleep(wait_s)
    return 1


def _run_to_file(cmd: list[str], out: Path, cwd: Path = ROOT) -> int:
    with out.open("w") as f:
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                              env=child_env()).returncode


def newest_report(results: Path, dataset: str, sm: str, qd: str) -> Path | None:
    pages = sorted(results.glob(f"report-{dataset}-{sm}-vs-{qd}-*.html"),
                   key=lambda p: p.stat().st_mtime)
    return pages[-1] if pages else None


def keep_report(root: Path, night: Path, dataset: str, sm: str, qd: str,
                log: Log) -> Path | None:
    """Find the run's report, render it if the run did not, and keep a copy.

    A copy, not just the path. The stamped name comes from the newest arm's
    `started` (`report.run_stamp`), so any later re-render of the same pair
    writes the same file: on 2026-09-23 a session re-rendered at 07:04 and the
    07:00 artifact was gone, with only `report.path` pointing at what was now
    a different file. The copy beside the log is what the run produced.
    """
    results = root / "bench/results"
    report = newest_report(results, dataset, sm, qd)
    if report is None and all((results / x / "rows.json").is_file() for x in (sm, qd)):
        log("no report from render; running report.py by hand")
        with (night / "report_by_hand.out").open("w") as f:
            rc = subprocess.run(["uv", "run", "--project", "bench", "bench/harness/report.py",
                                 sm, qd], cwd=root, stdout=f, stderr=subprocess.STDOUT,
                                env=child_env()).returncode
        log(f"report.py exited {rc}")
        report = newest_report(results, dataset, sm, qd)
    log(f"report: {report or 'none'}")
    (night / "report.path").write_text(f"{report or 'none'}\n")
    if report is not None:
        shutil.copy2(report, night / report.name)
        log(f"report copied into {night}")
    return report


def analyse(root: Path, night: Path, dataset: str, sm: str, qd: str, prev: str | None,
            log: Log) -> None:
    """Headless `claude -p` with read-only tools writes `analysis.md`."""
    claude = shutil.which("claude", path=child_env()["PATH"])
    if claude is None:
        log("no claude on PATH; analysis skipped")
        return
    log("analysis starting")
    prompt = (HERE / "nightrun-analysis.md").read_text()
    for key, value in (("@SM@", sm), ("@QD@", qd), ("@DATASET@", dataset),
                       ("@DIR@", str(night.relative_to(root))), ("@PREV@", prev or "")):
        prompt = prompt.replace(key, value)
    out, err = night / "analysis.md", night / "analysis.err"
    with out.open("w") as fo, err.open("w") as fe:
        rc = subprocess.run([claude, "-p", prompt, "--output-format", "text",
                             "--allowedTools", ANALYSIS_TOOLS],
                            cwd=root, stdout=fo, stderr=fe, env=child_env()).returncode
    lines = out.read_text(errors="replace").count("\n") if out.exists() else "?"
    log(f"analysis exited {rc} ({lines} lines in {out})")


def main(argv: list[str], root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description="An unattended publication run.")
    ask = ap.add_mutually_exclusive_group()
    # Both answer a question about the run before it is scheduled, and write
    # nothing. A run that silently found no previous pair loses its
    # cross-engine latency read and says so only in the report, hours later.
    ask.add_argument("--print-prev", action="store_true",
                     help="print the pair this run would take its reference from, and exit")
    ask.add_argument("--print-ref", action="store_true",
                     help="print the reference it would take from that pair, and exit")
    ap.add_argument("date", help="YYYY-MM-DD, which names the labels and the night directory")
    ap.add_argument("dataset", nargs="?", default="sift1m")
    args = ap.parse_args(argv)

    results = root / "bench/results"
    sm, qd = labels(args.date, args.dataset)
    prev = prev_pair(results, args.dataset, sm)
    if args.print_prev:
        if prev:
            print(results / prev)
        return 0
    if args.print_ref:
        # The resolver explains itself on stdout; the answer is the only
        # thing this flag prints there.
        with contextlib.redirect_stdout(sys.stderr):
            ref = resolve_ref(prev) if prev else None
        if ref is not None:
            print(ref)
        return 0

    # One run per night directory. On 2026-09-23 a failed 03:51 launch had its
    # analysis still running when the directory was removed and a second
    # launch recreated it, so a stray "analysis exited 0" from the dead run
    # landed in the middle of the live one's log. The directory comes from the
    # date alone, so a second run of one night needs the first moved aside.
    night = results / f"night-{args.date.replace('-', '')}"
    if (night / "night.log").exists():
        print(f"nightrun: {night / 'night.log'} exists; another run of {args.date} has "
              f"used this directory.\n  move it aside, or pass a different date, rather "
              f"than sharing a log.", file=sys.stderr)
        return EXIT_LOG_EXISTS
    night.mkdir(parents=True, exist_ok=True)
    log = Log(night / "night.log")

    reps = os.environ.get("REPS", "3")
    log(f"night run starting: {args.dataset}, labels {sm} / {qd}, reps {reps}")
    dirty = len((git(["status", "--short"], root) or "").splitlines())
    log(f"strawmann {git(['rev-parse', '--short', 'HEAD'], root)} dirty={dirty}")
    binary = Path(os.environ.get("QDRANT_BINARY",
                                 Path.home() / "Workspace/qdrant/target/release/qdrant"))
    if not os.access(binary, os.X_OK):
        log(f"EXIT={EXIT_NO_QDRANT} (no Qdrant binary at {binary}; set QDRANT_BINARY)")
        return EXIT_NO_QDRANT
    log(qdrant_provenance(binary))

    ref = resolve_ref(prev) if prev else None
    if prev is None:
        log(f"no previous {family(args.dataset)} pair: each engine uses its own saturation "
            f"(report refuses the cross-engine latency read)")
    elif ref is None:
        log(f"no usable rps reference from {prev} / qd-{prev.removeprefix('sm-')} (see the "
            f"resolver's refusal); each engine uses its own saturation")
    else:
        log(f"rps reference {ref} from {prev} / qd-{prev.removeprefix('sm-')}")

    # Sessions of the CLI are ambient load the gate sees (findings 44). Named,
    # not killed: they are the user's.
    alive = subprocess.run(["pgrep", "-x", "claude"], capture_output=True, text=True)
    n = len(alive.stdout.split())
    if n:
        log(f"warning: {n} claude process(es) alive; the gate counts their idle CPU "
            f"as foreign load")

    python = shutil.which("python3", path=child_env()["PATH"]) or sys.executable
    cmd = [python, "bench/harness/fullrun.py",
           "--server-cpus", os.environ.get("SERVER_CPUS", "4-11"),
           "--client-cpus", os.environ.get("CLIENT_CPUS", "0-3"),
           "--dataset", args.dataset, "--reps", reps, "--segment-policy", "equal-work",
           "--perf", "--qdrant-binary", str(binary),
           *(["--rps-reference", str(ref)] if ref is not None else []),
           "--strawmann-label", sm, "--qdrant-label", qd]
    start = time.monotonic()
    rc = run_until_admitted(cmd, night, log,
                            run=lambda c, out: _run_to_file(c, out, cwd=root))
    log(f"measurement finished, rc={rc}, {int((time.monotonic() - start) // 60)} min")

    keep_report(root, night, args.dataset, sm, qd, log)
    analyse(root, night, args.dataset, sm, qd, prev, log)
    log("ALL_DONE")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
