#!/usr/bin/env python3
"""The verification gate: one command that checks everything.

    scripts/check.py                 run every step
    scripts/check.py --list          show the steps
    scripts/check.py --only 'zlint'  run steps whose label matches a regex

The shape follows a lesson learned the hard way elsewhere: a hand-maintained
YAML mirror of a local gate *drifts*. Eight of eighteen checks ended up with no
CI counterpart and nothing reported it, because a mirror has no failure mode --
it just covers less.

So CI invokes this file with `--only`, rather than reimplementing the checks in
YAML. One definition, run twice. `check_ci_covers_gate` fails if a step here has
no `# gate: <label>` marker in the workflow.

This is a gate, not an experiment, so it is Python rather than shell for the
same reason the workload runner is: subprocess handling, regex and structured
output are all things shell does grudgingly.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"


@dataclass
class Step:
    label: str
    why: str
    argv: list[str]
    cwd: Path = ROOT
    #: A step CI cannot run (needs a dataset, a live engine, a quiet host).
    local_only: bool = False


def steps() -> list[Step]:
    return [
        Step("format", "zig fmt is not advisory; an unformatted tree makes every diff noisy",
             ["zig", "fmt", "--check", "src", "bench", "build.zig"]),

        # zlint's config is an allowlist: naming any rule disables the rest. The
        # canary below is what stops this step from passing vacuously.
        Step("zlint", "lint with the explicit allowlist in zlint.json",
             ["zlint", "--deny-warnings"]),

        Step("zlint-not-vacuous", "prove zlint would still fail: a config that names any "
                                  "rule disables every rule it does not name",
             [sys.executable, str(Path(__file__).resolve()), "--canary"]),

        # The Python half of the toolchain. `src/` has zlint and `conformance/`
        # has clippy; the harness that produces every number in the reports had
        # nothing, and the rule set is an explicit `select` in ruff.toml for the
        # same reason zlint's is an allowlist. Run through `uv` so it is the
        # version bench/uv.lock pins, not whatever is on the machine.
        Step("ruff", "lint the harness against the explicit select in ruff.toml",
             ["uv", "run", "--project", str(ROOT / "bench"), "ruff", "check", "."]),

        Step("zig-test", "the Zig suite, unit and end-to-end over a real socket (2 skip off-host)",
             ["zig", "build", "test"]),

        Step("zig-build", "the release binary must build, not merely typecheck",
             ["zig", "build", "-Doptimize=ReleaseFast"]),

        Step("rust-test", "conformance harness: oracle, differ tiers, relevance metrics",
             ["cargo", "test", "--release", "--quiet"], cwd=ROOT / "conformance"),

        # `-D warnings` rather than a bare run, for the reason the zlint canary
        # exists: `cargo clippy` exits 0 while printing warnings, so a step
        # without it passes just as loudly on a clean tree as on one with
        # twenty-nine lints. The lint set lives in `conformance/Cargo.toml`
        # under `[workspace.lints.clippy]`, and several of its members guard
        # things this project cares about specifically -- `cast_lossless`
        # covers the `as f64` widenings inside the fp64 oracle's own distance
        # kernels, which is the code every recall figure is graded against.
        # `--all-targets` so the tests are linted too; most of the casts were
        # in them.
        Step("rust-clippy", "the conformance lint set in Cargo.toml, enforced rather than "
                            "printed: clippy exits 0 while warning, so this denies",
             ["cargo", "clippy", "--all-targets", "--quiet", "--", "-D", "warnings"],
             cwd=ROOT / "conformance"),

        # Checked rather than trusted: two files sat unformatted on main, so
        # every `cargo fmt` of a change reformatted them too and the diff
        # carried someone else's whitespace.
        Step("rust-fmt", "the conformance crate is rustfmt-clean, so a formatted change "
                         "diffs as its own lines",
             ["cargo", "fmt", "--check"], cwd=ROOT / "conformance"),

        Step("python-syntax", "the experiment drivers must at least parse",
             [sys.executable, str(Path(__file__).resolve()), "--check-syntax"]),

        Step("no-scratchpad-paths", "a session-specific /tmp path in a tracked file is "
                                    "meaningless to anyone else and leaks the machine",
             [sys.executable, str(Path(__file__).resolve()), "--check-paths"]),

        # `bench/results/` is gitignored, so a fresh clone has nothing to check
        # against and this passes with a note. It only bites on a machine that
        # has measured results and a README that disagrees with them. When the
        # measured results themselves are stale (no harness stamp, or two
        # labels from different harness configurations) or unlicensed
        # (`licenses_comparative` false), compare.py prints a `!! WARNING`
        # naming the reason and the tables carry STALE / UNLICENSED banners;
        # the step still passes, so a host holding old results can develop.
        Step("readme-table-current", "the generated tables in README.md and "
                                     "docs/comparison-<dataset>.md must match the measured "
                                     "results, not a transcription of them",
             [str(ROOT / "bench/harness/compare.py"), "strawmann", "qdrant",
              "--check-readme"]),

        # Unlike the step above, this one bites in CI: its source is the
        # committed report pages, not the gitignored results. A page copied
        # into docs/reports/ or retired from it without regenerating the
        # cards leaves the landing page describing an archive that is gone.
        Step("landing-cards-current", "the report cards on docs/index.html must "
                                      "match the pages in docs/reports/",
             [sys.executable, str(ROOT / "bench/harness/landing_cards.py"), "--check"]),

        # The stdlib half of the harness has unit tests that need no engine,
        # dataset or measured results: the recall-equality rule on ratios, the
        # staleness refusal, the licence banner, wall-clock qps, W11's command
        # shape, the recall join key and the backfill that must not invent an
        # `ef`.
        # Under `uv run` when uv is there (it is what CI installs), because the
        # report tests need the uv project's jinja2/pandas/plotly and skip
        # themselves without it: 45 of 50 tests passing under the bare
        # interpreter looked like the whole suite passing.
        Step("harness-unit", "the harness's refusals and joins behave as documented",
             [sys.executable, str(Path(__file__).resolve()), "--check-harness-unit"]),

        # The report path has real dependencies (jinja2, plotly, pandas) pinned
        # by bench/uv.lock. Checking it imports is what catches a lock that no
        # longer resolves, which otherwise surfaces only at the end of a
        # 50-minute sweep when the report is generated.
        # datasets.json is now read by three consumers: datasets.py, the Rust
        # conformance binary via include_str!, and anything driving bfb through
        # `bfb-config`. A malformed or self-inconsistent descriptor should fail
        # here, not at the start of a 9 GB download.
        # The doctor is the entry point someone runs first on a new machine. If
        # it crashes, the failure lands on whoever is least equipped to debug
        # it, so it has to at least run end to end.
        # regression.py is the redline function: it decides whether a
        # difference between two runs is real. Its null test must keep passing,
        # since a detector that reports regressions on an unchanged binary is
        # worse than none.
        Step("regression-null", "the detector must find nothing between two identical runs",
             [sys.executable, str(Path(__file__).resolve()), "--check-regression"]),

        Step("doctor", "the environment report must run without crashing",
             [str(ROOT / "scripts/doctor.py"), "--quick", "--json", "--exit-zero"]),

        # procstat turns kernel counters into result rows, and every way it can
        # be wrong produces a plausible number: a restarted engine differencing
        # to nonsense, a cgroup format change parsing to zero, a syscall count
        # presented as a disk figure. Its parsers and delta rules are checked
        # against synthetic input, and against the host's real io.stat when
        # there is one.
        Step("procstat", "the storage and I/O counters must parse and difference correctly",
             [str(ROOT / "bench/harness/procstat.py"), "--self-test"]),

        # perfstat turns `perf stat` output into result rows, and its failure
        # modes are quieter than procstat's: an event the part does not
        # implement, a group perf had to multiplex and therefore extrapolated,
        # and a duration that arrives in milliseconds among a dozen raw counts.
        # Each of them, handled wrongly, is a plausible number rather than an
        # error, and two of them are indistinguishable from a measured zero.
        Step("perfstat", "the hardware counters must parse, and refuse what they could not measure",
             [str(ROOT / "bench/harness/perfstat.py"), "--self-test"]),

        # §8's gate: no performance number without a green conformance row. It
        # has to refuse the five ways it claims to *and* accept a licensed row,
        # or the pipeline it guards is blocked rather than gated.
        Step("results-sink", "the §8 gate must accept a licensed row and refuse the rest",
             [str(ROOT / "bench/harness/results.py"), "--self-test"]),

        Step("datasets-descriptor", "the §4.2 dataset descriptor parses and is consistent",
             [str(ROOT / "conformance/datasets/datasets.py"), "list", "--fast"]),

        Step("datasets-self-test", "the fetcher's checksum scaffolding reads the final hop only",
             [str(ROOT / "conformance/datasets/datasets.py"), "self-test"]),

        Step("report-tooling", "the uv-managed report path must import and render",
             [sys.executable, str(Path(__file__).resolve()), "--check-report"]),

        Step("ci-covers-gate", "every step here must be run by a live CI command, and CI "
                               "must have a trigger that fires without being asked",
             [sys.executable, str(Path(__file__).resolve()), "--check-ci"]),
    ]


# --------------------------------------------------------------------------
# Self-checks, invoked as subprocesses so they appear as ordinary steps
# --------------------------------------------------------------------------

def canary() -> int:
    """Add an unused declaration and confirm zlint fails on it.

    A `zlint.json` listing three rules as `off` and nothing else silently
    disables every other rule; a tree in that state reports `0 errors, 0
    warnings` vacuously, and can stay that way for weeks. A clean lint run is only evidence if a deliberate
    violation would break it.
    """
    target = ROOT / "src/dist/common.zig"
    original = target.read_text()
    try:
        target.write_text(original + "\nconst zlint_gate_canary_unused = 42;\n")
        r = subprocess.run(["zlint", "--deny-warnings"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            print("zlint PASSED with a deliberately unused decl present.", file=sys.stderr)
            print("The config is vacuous: check that zlint.json lists every rule.",
                  file=sys.stderr)
            return 1
        print("ok: zlint still catches an unused decl")
        return 0
    finally:
        target.write_text(original)


def check_syntax() -> int:
    """Every driver parses. Skips dot-directories: `bench/.venv` is 3,700 files
    of someone else's Python and was being parsed on every run."""
    n = 0
    for top in ("bench", "scripts"):
        for p in sorted((ROOT / top).rglob("*.py")):
            if any(part.startswith(".") for part in p.relative_to(ROOT).parts):
                continue
            ast.parse(p.read_text(), filename=str(p))
            n += 1
    print(f"ok: {n} files parse")
    return 0


def check_harness_unit() -> int:
    """The harness unit tests, with the report tests actually run.

    `ReportHostTests` skip without the uv project's dependencies. A skipped
    test is not a passed one, so a run that skips fails here and says why,
    rather than the gate reporting green on the stdlib half alone.
    """
    # Discovery, not a file path: the suite is `test_compare`, `test_workloads`,
    # `test_report`, `test_results` and `test_paths`, and naming one of them
    # here is how a sixth would be written and never run.
    discover = ["-m", "unittest", "discover", "-s", str(ROOT / "bench/harness"),
                "-t", str(ROOT / "bench/harness"), "-p", "test_*.py", "-q"]
    if shutil.which("uv"):
        argv = ["uv", "run", "--project", str(ROOT / "bench"), "python", *discover]
    else:
        argv = [sys.executable, *discover]
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    tail = out.strip().splitlines()[-3:]
    for line in tail:
        print(line)
    if r.returncode != 0:
        print(out[-3000:], file=sys.stderr)
        return 1
    if "skipped" in (tail[-1] if tail else ""):
        print("harness tests were SKIPPED, not passed: the report tests need the uv "
              "project (install uv, then `uv sync --project bench`)", file=sys.stderr)
        return 1
    return 0


def check_paths() -> int:
    """No session-specific scratchpad paths in tracked files."""
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    bad: list[str] = []
    # Assembled from fragments so this checker does not match itself. The
    # previous pattern anchored on `\d+` after the prefix and so missed a path
    # written with literal dots in a comment — which is exactly the shape a
    # human types when quoting a path they are half-anonymising, and is how one
    # survived in `regression.py` through every run of this gate.
    session_tmp = "/tmp/" + "claude-"
    home_ws = "/home/" + "[a-z]+/Workspace"
    pattern = re.compile(f"{re.escape(session_tmp)}|{home_ws}")
    for rel in tracked:
        p = ROOT / rel
        if not p.is_file() or p.suffix in {".png", ".bin"}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                bad.append(f"{rel}:{i}: {line.strip()[:90]}")
    if bad:
        print("machine-specific paths in tracked files:", file=sys.stderr)
        for b in bad[:10]:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"ok: {len(tracked)} tracked files, no scratchpad paths")
    return 0


def check_report() -> int:
    """The report tooling imports and its template renders.

    `uv run` syncs from the lock first, so this also proves the lock resolves on
    this machine. A missing `uv` is reported rather than failed: the engine and
    every measurement work without it, and only the HTML report does not.
    """
    if not shutil.which("uv"):
        print("uv not installed; report tooling not checked "
              "(install uv, then `uv sync --project bench`)")
        return 0
    r = subprocess.run(
        ["uv", "run", "--project", str(ROOT / "bench"), "python", "-c",
         ("import sys; sys.path.insert(0, 'bench/harness'); import report; "
         "from jinja2 import Environment, FileSystemLoader; "
         "e = Environment(loader=FileSystemLoader('bench/harness/templates')); "
         "e.get_template('report.html.j2'); print('ok')")],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("report tooling failed to load:", file=sys.stderr)
        print((r.stderr or r.stdout).strip()[-700:], file=sys.stderr)
        return 1
    print("ok: report.py imports and report.html.j2 parses")
    return 0


def check_regression() -> int:
    """Synthetic null test: identical rows in, no verdict out.

    Uses fabricated rows rather than measured ones so it runs on any machine,
    including CI, where `bench/results/` is empty. It checks the decision rule,
    not the engine: given a difference inside the noise floor, the answer must
    be "inconclusive", and given one far outside it, "regression".
    """
    import json
    import tempfile

    reg = ROOT / "bench/harness/regression.py"
    if not reg.exists():
        print("regression.py missing", file=sys.stderr)
        return 1

    def rows(vals):
        return json.dumps([{"id": k, "status": "ok", "seconds": 1, "load_start": 0,
                            "load_end": 0, "foreign": "", "qps": v, "rps": v,
                            "detail": ""} for k, v in vals.items()])

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        noise = base / "noise.json"
        noise.write_text(json.dumps({
            "rsd": {"A": 0.05, "B": 0.05}, "reps": {"A": 5, "B": 5},
            "source": "synthetic", "discarded": []}))
        results = base / "bench/results"
        for label, vals in (("old", {"A": 1000.0, "B": 1000.0}),
                            # A is +4%, inside 3 sigma (15%). B is -50%, far outside.
                            ("new", {"A": 1040.0, "B": 500.0})):
            d = results / label
            d.mkdir(parents=True)
            (d / "rows.json").write_text(rows(vals))

        env = dict(os.environ, STRAWMANN_ROOT=str(base), NO_COLOR="1")
        r = subprocess.run([str(reg), "old", "new", "--noise", str(noise)],
                           capture_output=True, text=True, env=env)
        out = r.stdout
        problems = []
        for line in out.splitlines():
            if line.strip().startswith("A ") and "inconclusive" not in line:
                problems.append(f"A moved 4% inside a 15% threshold but was not "
                                f"inconclusive: {line.strip()}")
            if line.strip().startswith("B ") and "REGRESSION" not in line:
                problems.append(f"B halved but was not called a regression: {line.strip()}")
        if "1 regression(s)" not in out:
            problems.append("expected exactly 1 regression in the summary")
        if problems:
            print("regression detector misbehaved:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            print(out, file=sys.stderr)
            return 1
    print("ok: inside-noise reads inconclusive, outside-noise reads regression")
    return 0


def ci_triggers(ci_text: str) -> set[str]:
    """The event names under the workflow's `on:` key.

    Comment lines are skipped, which is the whole point: the events that
    re-enabled per-push CI sat in a comment block for weeks, and a coverage
    check reading the file as text cannot tell those from the live ones.
    """
    lines = ci_text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "on:")
    except StopIteration:
        return set()
    events = set()
    for ln in lines[start + 1:]:
        bare = ln.strip()
        if not bare or bare.startswith("#"):
            continue
        if not ln[:1].isspace():          # back to column 0: `on:` is over
            break
        if ln[:2] == "  " and ln[2:3] not in (" ", "-"):
            events.add(bare.split(":", 1)[0].strip())
    return events


def ci_only_patterns(ci_text: str) -> list[str]:
    """Every `--only <regex>` a live `run:` line in the workflow passes.

    Read from the commands rather than from the `# gate:` comments beside them.
    A comment outlives the step it describes, so a marker is documentation and
    this is the check: delete a step and its coverage goes with it.
    """
    pats = []
    for ln in ci_text.splitlines():
        bare = ln.strip()
        if not bare or bare.startswith("#"):
            continue
        for m in re.finditer(r"--only\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", bare):
            pats.append(next(g for g in m.groups() if g is not None))
    return pats


def check_ci() -> int:
    """Every gate step must be run by a CI step, and CI must actually fire.

    Two failures this used to miss, both of them live at the time it was
    rewritten. It matched `# gate: <label>` as a substring of the file, so a
    marker in a comment block counted as coverage; and it never looked at
    `on:`, so it reported "all 21 gate steps are covered by CI" for a workflow
    triggered only by `workflow_dispatch`, which had never run once.
    """
    if not CI.exists():
        print(f"no CI workflow at {CI}", file=sys.stderr)
        return 1
    ci_text = CI.read_text()

    auto = ci_triggers(ci_text) - {"workflow_dispatch"}
    if not auto:
        print("the CI workflow has no automatic trigger: every step below can be "
              "covered on paper and never run.", file=sys.stderr)
        print("Add `push:` (and `pull_request:`) under `on:`.", file=sys.stderr)
        return 1

    pats = ci_only_patterns(ci_text)
    uncovered = [s.label for s in steps()
                 if not s.local_only
                 and not any(re.search(p, s.label) for p in pats)]
    if uncovered:
        print("gate steps no CI command runs:", file=sys.stderr)
        for m in uncovered:
            print(f"  {m}", file=sys.stderr)
        print("\nAdd a CI step running `scripts/check.py --only '<label>'`, with a",
              file=sys.stderr)
        print("`# gate: <label>` comment beside it for the reader. Do not reimplement",
              file=sys.stderr)
        print("the check in YAML: a hand-maintained mirror has no failure mode, it",
              file=sys.stderr)
        print("just covers less.", file=sys.stderr)
        return 1

    unmarked = [s.label for s in steps()
                if not s.local_only and f"# gate: {s.label}" not in ci_text]
    if unmarked:
        print("gate steps run by CI but not marked for the reader:", file=sys.stderr)
        for m in unmarked:
            print(f"  {m}  (add `# gate: {m}` beside the step that runs it)",
                  file=sys.stderr)
        return 1

    stray = [p for p in pats if not any(re.search(p, s.label) for s in steps())]
    if stray:
        print("CI runs `--only` patterns matching no gate step:", file=sys.stderr)
        for p in stray:
            print(f"  {p}", file=sys.stderr)
        print("\nA renamed step leaves CI running nothing under the old name.",
              file=sys.stderr)
        return 1

    print(f"ok: all {len(steps())} gate steps are run by CI, on {sorted(auto)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", metavar="REGEX")
    ap.add_argument("--canary", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-paths", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-ci", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-report", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-regression", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-syntax", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-harness-unit", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.canary:
        return canary()
    if args.check_syntax:
        return check_syntax()
    if args.check_harness_unit:
        return check_harness_unit()
    if args.check_paths:
        return check_paths()
    if args.check_ci:
        return check_ci()
    if args.check_report:
        return check_report()
    if args.check_regression:
        return check_regression()

    todo = steps()
    if args.only:
        rx = re.compile(args.only)
        todo = [s for s in todo if rx.search(s.label)]
        if not todo:
            print(f"no step matches {args.only!r}", file=sys.stderr)
            return 2

    if args.list:
        for s in steps():
            print(f"  {s.label:<20} {s.why}")
        return 0

    failed: list[str] = []
    for s in todo:
        print(f"\n=== {s.label} ===\n    {s.why}", flush=True)
        t0 = time.monotonic()
        r = subprocess.run(s.argv, cwd=s.cwd, capture_output=True, text=True)
        secs = time.monotonic() - t0
        if r.returncode == 0:
            print(f"    ok in {secs:.1f}s")
            # A passing step may still have something to say (`!!` lines):
            # readme-table-current passes on stale or unlicensed results and
            # says why. Swallowing that would make the pass look clean.
            for line in (r.stdout + r.stderr).splitlines():
                if line.startswith("!!"):
                    print(f"      {line}")
        else:
            failed.append(s.label)
            print(f"    FAILED in {secs:.1f}s")
            for line in (r.stdout + r.stderr).strip().splitlines()[-15:]:
                print(f"      {line}")

    print()
    if failed:
        print(f"{len(failed)} of {len(todo)} steps failed: {', '.join(failed)}")
        return 1
    print(f"all {len(todo)} steps passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
