#!/usr/bin/env python3
"""An unattended publication run, scheduled for a quiet hour.

    python3 bench/harness/nightrun.py 2026-09-10 sift1m
    OVERSAMPLING_POLICY=matched python3 bench/harness/nightrun.py 2026-09-24 dbpedia-openai-1m
    systemd-run --user --on-calendar='2026-09-10 04:00:00' \\
        --unit=strawmann-night-20260910 -p WorkingDirectory=$PWD \\
        /usr/bin/python3 bench/harness/nightrun.py 2026-09-10 sift1m

Why a systemd --user timer and not a Claude session's cron: the §7.1 gate
counts the CLI's idle CPU as foreign load (findings 44), so the run must not
need a session open.

What this adds around `fullrun.py`, which does the measuring:

- labels stamped with the date, so a night never merges into a published set,
  and one run per night directory;
- `night.log`, with the provenance a reader reaches for first;
- `--wait-for-gate`, so a busy host is asked again rather than refused;
- a copy of the report beside the log;
- a headless `claude -p`, restricted to read-only tools, writing
  `analysis.md`: every warning, refusal and note, with its cause. The prompt is
  `nightrun-analysis.md` next to this file.

Waiting for the gate and finding the latency reference used to live here too,
as a loop that relaunched `fullrun.py` per ask and grepped its output, and a
helper that parsed a resolver's stdout for a number. Both are `fullrun.py`'s
now (`wait_until_admitted`, and `auto` reading the previous pair of the
family), where a hand run gets them as well.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: Exit codes a caller can tell apart, both before anything runs.
EXIT_LOG_EXISTS = 96
EXIT_NO_QDRANT = 97

#: How long `fullrun.py` asks a busy host again before giving up: two hours.
WAIT_FOR_GATE_MIN = float(os.environ.get("WAIT_FOR_GATE_MIN", 120))

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


class Log:
    """`night.log`, one timestamped line per event, appended."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        with self.path.open("a") as f:
            f.write(f"{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S} {msg}\n")


def git(args: list[str], cwd: Path) -> str | None:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def qdrant_provenance(binary: Path) -> str:
    """The log line for the Qdrant binary: path, sha256 and the checkout's commit.

    The checkout's HEAD is not evidence about the binary. On 2026-09-23 this
    line read `commit=63c6a797d` for a binary built at an earlier commit, and
    `run.json` carried the same claim, so the line says when the binary is
    older than the commit. The verdict is `provenance`'s, the one `fullrun.py`
    records.
    """
    import provenance
    commit = git(["rev-parse", "--short", "HEAD"], binary.parent) or "?"
    predates = (" (BINARY PREDATES THIS COMMIT; sha256 is the identity that holds)"
                if provenance.qdrant_binary_predates_head(binary) else "")
    digest = (provenance.file_digest(binary) or "?")[:16]
    return f"qdrant binary {binary} sha256={digest} commit={commit}{predates}"


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

    import fullrun
    sm, qd = labels(args.date, args.dataset)
    pair = fullrun.previous_pair([sm, qd])
    if args.print_prev:
        if pair:
            print(" ".join(pair))
        return 0
    if args.print_ref:
        # Asked exactly as the run will ask it: `auto`, with `--perf`'s
        # instrument. The resolver explains itself on stdout; the answer is
        # the only thing this flag prints there.
        fullrun.PERF_SET = "default"
        with contextlib.redirect_stdout(sys.stderr):
            try:
                ref = fullrun.resolve_rps_reference("auto", [sm, qd])
            except ValueError as e:
                print(e)
                ref = None
        if ref:
            print(round(ref))
        return 0

    # One run per night directory. On 2026-09-23 a failed 03:51 launch had its
    # analysis still running when the directory was removed and a second
    # launch recreated it, so a stray "analysis exited 0" from the dead run
    # landed in the middle of the live one's log. The directory comes from the
    # date alone, so a second run of one night needs the first moved aside.
    night = root / "bench/results" / f"night-{args.date.replace('-', '')}"
    if (night / "night.log").exists():
        print(f"nightrun: {night / 'night.log'} exists; another run of {args.date} has "
              f"used this directory.\n  move it aside, or pass a different date, rather "
              f"than sharing a log.", file=sys.stderr)
        return EXIT_LOG_EXISTS
    night.mkdir(parents=True, exist_ok=True)
    log = Log(night / "night.log")

    reps = os.environ.get("REPS", "3")
    # `defaults` is `fullrun.py`'s own default. The two quantized experiments
    # hash apart, so a night at `matched` cannot merge into a `defaults` page.
    oversampling = os.environ.get("OVERSAMPLING_POLICY", "defaults")
    log(f"night run starting: {args.dataset}, labels {sm} / {qd}, reps {reps}, "
        f"oversampling {oversampling}")
    dirty = len((git(["status", "--short"], root) or "").splitlines())
    log(f"strawmann {git(['rev-parse', '--short', 'HEAD'], root)} dirty={dirty}")
    binary = Path(os.environ.get("QDRANT_BINARY",
                                 Path.home() / "Workspace/qdrant/target/release/qdrant"))
    if not os.access(binary, os.X_OK):
        log(f"EXIT={EXIT_NO_QDRANT} (no Qdrant binary at {binary}; set QDRANT_BINARY)")
        return EXIT_NO_QDRANT
    log(qdrant_provenance(binary))
    # An explicit reference, when the previous pair's is not this pair's
    # regime: 0925's W4 was Qdrant's development profile, 4 search threads,
    # and "90% of saturation" off it is ~72% of a production Qdrant.
    rps_override = os.environ.get("RPS_REFERENCE")
    if rps_override:
        log(f"rps reference: {rps_override} q/s, from $RPS_REFERENCE")
    else:
        log(f"rps reference: `auto`, from {' / '.join(pair)}" if pair else
            f"rps reference: no previous {family(args.dataset)} pair, so each engine uses "
            f"its own saturation (the report refuses the cross-engine latency read)")

    # Sessions of the CLI are ambient load the gate sees (findings 44). Named,
    # not killed: they are the user's.
    alive = subprocess.run(["pgrep", "-x", "claude"], capture_output=True, text=True)
    if n := len(alive.stdout.split()):
        log(f"warning: {n} claude process(es) alive; the gate counts their idle CPU "
            f"as foreign load")
    import procstat
    if busy := procstat.builders_alive():
        log(f"warning: build processes alive ({', '.join(busy)}); a build in another "
            f"session is foreign load the gate counts, and memory pressure the engine "
            f"can be oom-killed under")

    python = shutil.which("python3", path=child_env()["PATH"]) or sys.executable
    cmd = [python, "bench/harness/fullrun.py",
           "--server-cpus", os.environ.get("SERVER_CPUS", "4-11"),
           "--client-cpus", os.environ.get("CLIENT_CPUS", "0-3"),
           "--dataset", args.dataset, "--reps", reps, "--segment-policy", "equal-work",
           "--oversampling-policy", oversampling,
           "--perf", "--qdrant-binary", str(binary),
           "--wait-for-gate", f"{WAIT_FOR_GATE_MIN:g}",
           "--strawmann-label", sm, "--qdrant-label", qd,
           *(["--rps-reference", rps_override] if rps_override else [])]
    # The one line an analysis otherwise has to reconstruct from this file.
    log("command: " + shlex.join(cmd))
    out = night / "fullrun.out"
    start = time.monotonic()
    with out.open("w") as f:
        rc = subprocess.run(cmd, cwd=root, stdout=f, stderr=subprocess.STDOUT,
                            env=child_env()).returncode
    log(f"fullrun.py exited {rc} (output in {out})")
    log(f"measurement finished, rc={rc}, {int((time.monotonic() - start) // 60)} min")

    keep_report(root, night, args.dataset, sm, qd, log)
    analyse(root, night, args.dataset, sm, qd, pair[0] if pair else None, log)
    log("ALL_DONE")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
