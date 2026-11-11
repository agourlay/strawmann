#!/usr/bin/env python3
"""Which commit moved a row: the same rows on several builds of the engine.

findings 69: strawmANN's fp32 graph search on laion lost 5 to 6% between the
0930 and 1003 pairs (W4 11,442 to 10,800 q/s, cycles per query +6.1% at
+1.6% instructions), with four candidates in the range and a kernel update
beside them. This builds each commit in a detached worktree and measures
each build's W4 and W10-ef128 over a fresh W2 upload, rotating the build
order every rep so a drift over the session lands on every build alike.

Development-grade: no Qdrant, no §7.1 publication gate. Each row still
records its own foreign load. Usage:

    commit_ab.py --server-cpus 4-11 --client-cpus 0-3 [--reps 3]
                 [--dataset laion-small-clip] [--commits A B ...]
                 [--flags NAME='--filtered-plan scan' ...] [--upload W12-upload]

An arm is a commit (`b3ca8d2`), a named commit (`scan=HEAD`, so one commit can
run as several arms), or a named engine already built (`zig016=/path/to/
strawmann`, for a build this checkout's `zig` cannot make). `--flags` adds
server flags to one arm by name.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import statistics
import sys
import time
from pathlib import Path

import fullrun
import perfstat
import w9_ab

import procstat
import workloads

#: findings 69's range, oldest first: the 0930 build, the drainer-on-upsert
#: commit, either side of the wide-row prefetch, and the 1003 build.
COMMITS = ("1fb2e43", "0509410", "b3ca8d2^", "b3ca8d2", "f346aea")
ROWS = ("W4", "W10-ef128")
PORT = 6334
WORKTREES = Path.home() / ".cache/strawmann/ab"

_DRAIN = re.compile(r"index: drain .*")


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    #: A commit to build, or the path of an engine already built.
    source: str
    flags: tuple[str, ...] = ()

    @property
    def prebuilt(self) -> bool:
        return "/" in self.source


def parse_arms(specs: list[str], flags: list[str] = ()) -> list[Arm]:
    """`commit`, `name=commit` or `name=/path/strawmann`, in the order given.

    A prebuilt engine must be called `strawmann`: the foreign-load census
    knows the engine by its process name, so a binary called anything else is
    counted as load on itself.
    """
    arms: dict[str, Arm] = {}
    for spec in specs:
        name, eq, source = spec.partition("=")
        if not eq:
            source = name
        if not name or not source or name in arms:
            raise ValueError(f"arm {spec!r}: want COMMIT, NAME=COMMIT or NAME=PATH, "
                             f"each name once")
        if "/" in source:
            source = str(Path(source).expanduser())
            if Path(source).name != "strawmann":
                raise ValueError(f"arm {spec!r}: name the binary `strawmann`, or the "
                                 f"census counts it as foreign load")
        arms[name] = Arm(name, source)
    for f in flags:
        name, eq, argv = f.partition("=")
        if not eq or name not in arms or not argv.strip():
            raise ValueError(f"--flags {f!r}: want NAME='--flag value' for an arm given")
        arms[name] = dataclasses.replace(arms[name], flags=(*arms[name].flags,
                                                            *shlex.split(argv)))
    return list(arms.values())


def plan(commits: list[str], reps: int) -> list[tuple[int, str]]:
    """`(rep, commit)` in run order, the order rotated by one each rep."""
    out = []
    for rep in range(reps):
        k = rep % len(commits)
        out += [(rep + 1, c) for c in commits[k:] + commits[:k]]
    return out


def worktree_dir(commit: str) -> Path:
    """One directory per commit, `^` spelled out so the path is plain."""
    return WORKTREES / commit.replace("^", "-parent")


def build(commit: str) -> Path | None:
    """The commit's ReleaseFast engine, built in a worktree of its own."""
    wt = worktree_dir(commit)
    if not wt.exists():
        code, out = fullrun.sh(["git", "worktree", "add", "--detach", str(wt), commit],
                               120, cwd=fullrun.ROOT)
        if code != 0:
            print(out.rstrip(), file=sys.stderr)
            return None
    fullrun.say(f"building {commit} (ReleaseFast)")
    code, out = fullrun.sh(["zig", "build", "-Doptimize=ReleaseFast"], 900, cwd=wt)
    if code != 0:
        print(out.rstrip(), file=sys.stderr)
        return None
    return build_path(commit)


def build_path(commit: str) -> Path:
    """Where `build` leaves a commit's engine."""
    return worktree_dir(commit) / "zig-out/bin/strawmann"


def drain_lines(log: Path) -> list[str]:
    """The drainer's progress lines, which say whether it ran during a row."""
    if not log.exists():
        return []
    return [m.group(0) for m in _DRAIN.finditer(log.read_text(errors="replace"))]


def summarise(rows: list[dict]) -> list[dict]:
    """Per row and commit, medians over the reps."""
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("qps"):
            by.setdefault((r["id"], r["commit"]), []).append(r)
    out = []
    for (wid, commit), got in by.items():
        def med(f, got=got):
            vals = [v for r in got if (v := f(r)) is not None]
            return statistics.median(vals) if vals else None
        out.append({
            "row": wid, "commit": commit, "reps": len(got),
            "qps": med(lambda r: r["qps"]),
            "qps_all": [r["qps"] for r in got],
            "cycles_per_q": med(lambda r: _per_q(r, "perf_cycles")),
            "instructions_per_q": med(lambda r: _per_q(r, "perf_instructions")),
            "dram_kib_per_q": med(lambda r: (v / 1024 if (v := _per_q(r, "dram_bytes"))
                                             is not None else None)),
            "foreign": sorted({r["foreign"] for r in got if r.get("foreign")}),
        })
    return out


def _per_q(r: dict, key: str) -> float | None:
    return r[key] / r["n_queries"] if r.get(key) and r.get("n_queries") else None


def table_text(summary: list[dict], commits: list[str]) -> str:
    def f(v, fmt):
        return format(v, fmt) if v is not None else "-"
    lines = [(f"{'row':<10} {'commit':<9} {'reps':>4} {'q/s':>9} {'kcyc/q':>8} "
              f"{'kins/q':>8} {'DRAM KiB/q':>11}  passes")]
    order = {c: i for i, c in enumerate(commits)}
    for s in sorted(summary, key=lambda s: (s["row"], order.get(s["commit"], 99))):
        lines.append(
            f"{s['row']:<10} {s['commit']:<9} {s['reps']:>4} {f(s['qps'], '9,.0f')} "
            f"{f(s['cycles_per_q'] and s['cycles_per_q'] / 1e3, '8.0f')} "
            f"{f(s['instructions_per_q'] and s['instructions_per_q'] / 1e3, '8.0f')} "
            f"{f(s['dram_kib_per_q'], '11,.0f')}  "
            f"{' / '.join(f'{q:,.0f}' for q in s['qps_all'])}"
            f"{'  foreign: ' + ', '.join(s['foreign']) if s['foreign'] else ''}")
    return "\n".join(lines) + "\n"


def measure(w: workloads.Workload, uri: str, out: Path, commit: str, rep: int) -> dict:
    results = out / f"{worktree_dir(commit).name}-r{rep}"
    results.mkdir(parents=True, exist_ok=True)
    w9_ab.warn_other_engines()
    res = workloads.run_one(w, uri, results, w9_ab.COMMON, perf="default")
    row = dataclasses.asdict(res)
    row.update(perfstat.derive(row))
    row.update({"commit": commit, "rep": rep})
    (results / f"{w.id}.row.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(f"  {commit} rep {rep} {w.id}: {row.get('qps')} q/s, status "
          f"{row.get('status')}, foreign {row.get('foreign') or '-'}", flush=True)
    return row


def binaries_for(arms: list[Arm]) -> dict[str, Path] | None:
    """Each arm's engine, a commit built once however many arms run it."""
    built: dict[str, Path] = {}
    out = {}
    for a in arms:
        if a.prebuilt:
            if not Path(a.source).is_file():
                print(f"arm {a.name}: no engine at {a.source}", file=sys.stderr)
                return None
            out[a.name] = Path(a.source)
            continue
        if a.source not in built:
            if (b := build(a.source)) is None:
                return None
            built[a.source] = b
        out[a.name] = built[a.source]
    return out


def run_sessions(args, out: Path, binaries: dict[str, Path], rows: list[dict]) -> int:
    table = {w.id: w for w in workloads.table()}
    flags = {a.name: list(a.flags) for a in args.arms}
    uri = f"http://localhost:{PORT}"
    workers = max(1, fullrun.cpu_count(args.server_cpus) - 1)
    for rep, commit in plan([a.name for a in args.arms], args.reps):
        fullrun.say(f"{commit} rep {rep}")
        log = out / f"{worktree_dir(commit).name}-r{rep}" / "server.log"
        engine = fullrun.start_strawmann(args.server_cpus, PORT, log, workers,
                                         workloads.required_capacity(),
                                         binary=binaries[commit], extra=flags[commit])
        if engine is None:
            return 1
        try:
            up = workloads.run_one(table[args.upload], uri, log.parent, w9_ab.COMMON)
            if up.status != workloads.Status.ok:
                print(f"upload failed: {up.status}", file=sys.stderr)
                return 1
            for wid in args.rows:
                fullrun.settle(f"{commit} rep {rep} {wid}")
                row = measure(table[wid], uri, out, commit, rep)
                row["drain"] = drain_lines(log)[-3:]
                rows.append(row)
        finally:
            fullrun.stop_strawmann(engine)
    return 0


def write_summary(out: Path, rows: list[dict], commits: list[str]) -> None:
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(
        {"commits": commits, "summary": summary, "rows": rows},
        indent=2, default=str) + "\n")
    text = table_text(summary, commits)
    (out / "summary.txt").write_text(text)
    print("\n" + text, flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--client-cpus", required=True)
    ap.add_argument("--dataset", default="laion-small-clip")
    ap.add_argument("--commits", nargs="+", default=list(COMMITS),
                    help="arms: COMMIT, NAME=COMMIT or NAME=/path/to/strawmann")
    ap.add_argument("--flags", action="append", default=[], metavar="NAME=FLAGS",
                    help="server flags for one arm, e.g. scan='--filtered-plan scan'")
    ap.add_argument("--upload", default="W2",
                    help="the upload row each session starts with (W12-upload for W12)")
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default=f"commit-ab-{time.strftime('%m%d')}")
    args = ap.parse_args(argv[1:])

    workloads.use_dataset(args.dataset)
    os.sched_setaffinity(0, w9_ab.cpu_set(args.client_cpus))
    if busy := procstat.builders_alive():
        print(f"!! build processes alive ({', '.join(busy)}): foreign load every row "
              f"will record", flush=True)
    claude = [c for _, c in procstat.processes() if c == "claude"]
    if claude:
        print(f"!! {len(claude)} claude process(es) alive; their idle CPU is foreign load",
              flush=True)
    try:
        args.arms = parse_arms(args.commits, args.flags)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if (binaries := binaries_for(args.arms)) is None:
        return 1
    out = fullrun.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    names = [a.name for a in args.arms]
    try:
        return run_sessions(args, out, binaries, rows)
    finally:
        write_summary(out, rows, names)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
