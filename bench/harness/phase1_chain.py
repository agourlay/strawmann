#!/usr/bin/env python3
"""The 1102 plan's phase 1 measurements, chained for one quiet stretch.

1. Kernel placement (findings 69): laion W4 and W10-ef128 on the Zig 0.16
   build, the last commit with the fp32 kernels inlined, and the first with
   them `noinline`.
2. sift1m W9 (-26% from 0.16 to 0.17): the same three engines, W9 with
   `perf stat`, to say whether it is the same cause.
3. The filtered planner: `planner_ab.py`, both plans forced on the W12-sel10
   ladder, four corpora.

Each step is its own driver in a subprocess, so one that fails leaves the next
to run; `chain.json` records each one's exit code, duration and log.
Scheduled, not run from a session (the CLI's idle CPU is foreign load):

    systemd-run --user --on-calendar='2026-11-03 18:00:00' --unit=strawmann-phase1-1103 \\
        -p WorkingDirectory=$PWD/bench/harness \\
        $PWD/bench/.venv/bin/python phase1_chain.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import nightrun

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
#: b54b283 built with Zig 0.16, the last toolchain that kept the kernels out of line.
ZIG016 = Path.home() / ".cache/strawmann/wt-zig016/zig-out/bin/strawmann"
INLINED = "74cc8e6"   # Zig 0.17, kernels inlined into Probe.scoreRow
NOINLINE = "ea84cbc"  # ef271d1's noinline, plus --filtered-plan for step 3
BFB = Path.home() / ".cache/strawmann/bfb-pinned/target/release/bfb"
CPUS = ("--server-cpus", "4-11", "--client-cpus", "0-3")


@dataclasses.dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]


def steps(tag: str, zig016: Path = ZIG016) -> list[Step]:
    arms = ["--commits", f"zig016={zig016}", f"inlined={INLINED}", f"noinline={NOINLINE}"]
    return [
        Step("kernel-laion", ["commit_ab.py", *CPUS, "--dataset", "laion-small-clip",
                              "--reps", "3", "--tag", f"{tag}-kernel-laion",
                              "--rows", "W4", "W10-ef128", *arms]),
        Step("w9-sift1m", ["commit_ab.py", *CPUS, "--dataset", "sift1m", "--reps", "3",
                           "--tag", f"{tag}-w9-sift1m", "--rows", "W9", *arms]),
        Step("planner", ["planner_ab.py", *CPUS, "--commit", NOINLINE,
                         "--tag", f"{tag}-planner"]),
    ]


def missing(zig016: Path = ZIG016, bfb: Path = BFB) -> list[str]:
    """What would make a step fail an hour in, checked before the first."""
    return [f"no {what} at {p}" for what, p in (("Zig 0.16 engine", zig016), ("bfb", bfb))
            if not p.is_file()]


def run(chain: list[Step], out: Path, env: dict[str, str], runner=subprocess.run) -> list[dict]:
    done = []
    for s in chain:
        log = out / f"{s.name}.log"
        print(f"{time.strftime('%H:%M:%S')} {s.name}: {' '.join(s.argv)}", flush=True)
        t0 = time.monotonic()
        with log.open("w") as fh:
            rc = runner([sys.executable, str(HERE / s.argv[0]), *s.argv[1:]], cwd=HERE,
                        env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        done.append({"step": s.name, "rc": rc, "minutes": round((time.monotonic() - t0) / 60, 1),
                     "log": str(log), "argv": s.argv})
        print(f"{time.strftime('%H:%M:%S')} {s.name}: rc={rc}", flush=True)
        (out / "chain.json").write_text(json.dumps(done, indent=2) + "\n")
    return done


def main_for(argv: list[str], doc: str, default_tag: str,
             steps: Callable[[argparse.Namespace], list[Step]],
             missing: Callable[[], list[str]],
             add_args: Callable[[argparse.ArgumentParser], None] | None = None) -> int:
    """A dated chain's `main`: its `--tag`, `--only`, what it checks before the
    first step, and its own step list, built from the parsed arguments."""
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=default_tag)
    if add_args is not None:
        add_args(ap)
    ap.add_argument("--only", nargs="+", help="run these steps by name")
    args = ap.parse_args(argv[1:])
    if gone := missing():
        print("\n".join(gone), file=sys.stderr)
        return 2
    chain = [s for s in steps(args) if not args.only or s.name in args.only]
    out = RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    env = {**nightrun.child_env(), "BFB": str(BFB)}
    done = run(chain, out, env)
    return 0 if all(d["rc"] == 0 for d in done) else 1


def no_bfb() -> list[str]:
    """The check the later chains make: they need no Zig 0.16 engine."""
    return [] if BFB.is_file() else [f"no bfb at {BFB}"]


def main(argv: list[str]) -> int:
    return main_for(argv, __doc__, f"phase1-{time.strftime('%m%d')}",
                    lambda args: steps(args.tag), missing)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
