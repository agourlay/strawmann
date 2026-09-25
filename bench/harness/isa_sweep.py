#!/usr/bin/env python3
"""§7.5: run the workload table against every forced-ISA build arm.

    isa_sweep.py list                       the arms and what each isolates
    isa_sweep.py run                        every arm, the default row set
    isa_sweep.py run --arms baseline avx512-full
    isa_sweep.py run --rows W3 W9 --reps 3
    isa_sweep.py table                      the matrix from existing results

`zig build bench-isa` must have run first; it produces one ReleaseFast server
per arm under `zig-out/isa/<arm>/strawmann-<arm>`.

## Why this is a first-class benchmark

§6.6.5 makes the forced-ISA matrix the primary evidence for every SIMD claim,
because runtime dispatch cannot be: Zig has no per-function target attribute,
so one binary cannot hold both an AVX-512 and an AVX2 kernel and choose between
them the way Qdrant's shipped binary does.

It also answers what no cross-engine ratio can. "2.3x faster than Qdrant" mixes
the index, the transport, the allocator and the kernels; holding all of those
fixed and moving only the instruction set isolates the kernels, which is the
one component §5 actually predicts. A model predicting 1.3x that measures 1.05x
is wrong in a way a cross-engine ratio never reveals.

## Reading the result

Each arm differs from the one below by a named capability, so a difference
between adjacent arms attributes to that capability — which is why `avx2` and
`avx2-vnni` are separate rows: §6.6.2 warns that folding them hides 256-bit
`vpdpbusd`, most of what scalar quantization gains on Zen 4+.

The comparison is against `baseline` (SSE2 only) unless `--vs` says otherwise.
Rows that do no distance arithmetic, scroll in particular, should show **no**
difference: they are the control, and a difference there means the measurement
is picking up something other than the instruction set.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
ISA_DIR = ROOT / "zig-out/isa"
RESULTS = ROOT / "bench/results/isa"

#: Mirrors `isa_matrix` in build.zig. Ordered weakest to strongest so adjacent
#: pairs isolate one capability.
ARMS = [
    ("baseline", "SSE2 only: the 128-bit floor, and the control arm"),
    ("avx2", "x86_64_v3: AVX2 + FMA + BMI, no VNNI"),
    ("avx2-vnni", "AVX2 + AVX-VNNI: 256-bit vpdpbusd"),
    ("avx512", "AVX-512F/BW/DQ/VL: 512-bit fp32"),
    ("avx512-256", "AVX-512 capped to 256-bit (prefer_256_bit)"),
    ("avx512-vnni", "AVX-512 + VNNI: 512-bit vpdpbusd"),
    ("avx512-popcnt", "AVX-512 + VPOPCNTDQ: vpopcntq for binary quantization"),
    ("avx512-full", "everything the host offers"),
]

#: The rows worth sweeping, and what each is expected to show. Kept small on
#: purpose: eight arms times the full table is hours, and most rows measure the
#: transport rather than the kernels.
DEFAULT_ROWS = ["W3", "W4", "W6", "W7", "W8", "W9", "W13"]

#: Rows that do no distance arithmetic. These are controls: an arm-to-arm
#: difference here is measurement error, not an ISA effect, and the table calls
#: it out rather than letting it read as a result.
CONTROL_ROWS = {"W13"}

#: Rows whose kernel the arm is expected to change, and the capability that
#: should drive it. Stated up front so the sweep can be checked against a
#: prediction rather than explained after the fact.
EXPECTED = {
    "W3": "fp32 dot/euclid: wider vectors",
    "W4": "fp32 under saturation: wider vectors",
    "W6": "SQ8 int8: vpdpbusd (VNNI)",
    "W7": "binary: vpopcntq",
    "W8": "PQ ADC: gather/shuffle width",
    "W9": "exhaustive fp32: the purest kernel row",
    "W13": "scroll: no distance arithmetic, expect no change",
}


@dataclass
class Arm:
    name: str
    doc: str

    @property
    def binary(self) -> Path:
        return ISA_DIR / self.name / f"strawmann-{self.name}"

    @property
    def built(self) -> bool:
        return self.binary.exists() and os.access(self.binary, os.X_OK)


def arms(names: list[str] | None) -> list[Arm]:
    out = [Arm(n, d) for n, d in ARMS]
    if names:
        known = {a.name for a in out}
        for n in names:
            if n not in known:
                raise SystemExit(f"unknown arm {n!r}; known: {', '.join(sorted(known))}")
        out = [a for a in out if a.name in names]
    return out


def cmd_list(args) -> int:
    print(f"{'arm':<16} {'built':<7} isolates")
    print(f"{'-' * 16} {'-' * 7} {'-' * 52}")
    for a in arms(None):
        print(f"{a.name:<16} {'yes' if a.built else 'NO':<7} {a.doc}")
    missing = [a.name for a in arms(None) if not a.built]
    if missing:
        print(f"\n{len(missing)} arm(s) not built. Run: zig build bench-isa")
    print(f"\ndefault rows: {' '.join(DEFAULT_ROWS)}")
    print(f"controls (expect no ISA effect): {', '.join(sorted(CONTROL_ROWS))}")
    return 0


def wait_quiet(limit_pct: int = 10, cap_s: int = 300) -> int:
    """§7.1: do not start a timed row on a busy machine."""
    cores = os.cpu_count() or 1
    waited = 0
    while waited < cap_s:
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
        if load1 / cores * 100 <= limit_pct:
            return waited
        time.sleep(15)
        waited += 15
    return waited


def kill_stale_servers() -> list[str]:
    """Kill any strawmann still running, whatever arm it is.

    Matching on `comm == "strawmann"` is not enough: Linux truncates `comm` to
    15 characters, so `strawmann-avx512-full` appears as `strawmann-avx5` and an
    exact match silently spares it. A survivor holds the port, the arm under
    test fails to bind, and bfb happily measures the *previous* binary while the
    results are filed under the new arm's name. That happened, and nothing in
    the harness noticed.
    """
    killed = []
    for comm in pathlib.Path("/proc").glob("*/comm"):
        try:
            name = comm.read_text().strip()
        except OSError:
            continue
        if not name.startswith("strawmann"):
            continue
        pid = comm.parent.name
        try:
            os.kill(int(pid), signal.SIGTERM)
            killed.append(f"{name}({pid})")
        except (OSError, ValueError):
            pass
    if killed:
        time.sleep(2)
    return killed


def port_is_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def confirm_arm(out: Path, arm: Arm) -> None:
    """The running server must be the arm we think it is.

    The banner prints `isa build : <name>`, so the claim is checkable rather
    than assumed. Without this the sweep cannot tell a real ISA effect from
    having measured the same binary twice under two labels, which is the one
    failure that would invalidate every number in the matrix at once.
    """
    banner = (out / "server.log").read_text()
    want = f"isa build              : {arm.name}"
    if want not in banner:
        got = next((ln.strip() for ln in banner.splitlines() if "isa build" in ln),
                   "(no isa build line)")
        raise SystemExit(f"arm {arm.name}: the server on this port reports "
                         f"{got!r}, not {arm.name!r}. Refusing to attribute its "
                         f"numbers to this arm.")


def server_argv(arm: Arm, port: int, server_cpus: str | None = None) -> list[str]:
    """The arm's server, pinned to `server_cpus` when given.

    Unpinned, as every table in `docs/isa-matrix.md` was measured, an arm on
    this hybrid host could land on the Zen 5 or the Zen 5c cluster, whose
    clocks and L3 differ: a cycles difference that has nothing to do with the
    ISA. Pinned as `fullrun` pins the published engine: one I/O thread plus
    the workers inside the set.
    """
    argv = [str(arm.binary), "--port", str(port), "--capacity", "1100000",
            "--max-dim", "2048", "--connections", "64", "--streams", "16",
            # The probe would run once per arm start and add nothing: it
            # measures the host, which does not vary across arms.
            "--no-bandwidth-probe"]
    if server_cpus:
        import fullrun
        workers = max(1, fullrun.cpu_count(server_cpus) - 1)
        argv += ["--workers", str(workers), "--io-threads", "1", "--pin", "--cpus", server_cpus]
    return argv


def client_argv(port: int, name: str, rows: list[str], client_cpus: str | None = None,
                perf: str | None = None) -> list[str]:
    """`workloads.py run` for the arm's rows, on `client_cpus` when given,
    with the `perf stat` sidecar when asked: an ISA question is a cycles and
    instructions question, and qps alone cannot say which moved."""
    argv = [sys.executable, "-u", str(ROOT / "bench/harness/workloads.py"),
            "run", f"http://localhost:{port}", name,
            *(["--perf", perf] if perf else []), *rows]
    return (["taskset", "-c", client_cpus] if client_cpus else []) + argv


def run_arm(arm: Arm, rows: list[str], rep: int, port: int,
            server_cpus: str | None = None, client_cpus: str | None = None,
            perf: str | None = None) -> Path:
    out = RESULTS / f"{arm.name}-rep{rep}"
    out.mkdir(parents=True, exist_ok=True)
    stale = kill_stale_servers()
    if stale:
        print(f"      killed stale server(s): {', '.join(stale)}")
    if not port_is_free(port):
        raise SystemExit(f"port {port} is still in use; refusing to run, because "
                         f"bfb would measure whatever is listening there and the "
                         f"numbers would be filed under {arm.name!r}")
    srv = subprocess.Popen(server_argv(arm, port, server_cpus),
                           stdout=(out / "server.log").open("w"), stderr=subprocess.STDOUT)
    try:
        time.sleep(3)
        if srv.poll() is not None:
            log = (out / "server.log").read_text().strip().splitlines()[-6:]
            raise SystemExit(f"arm {arm.name}: server exited immediately "
                             f"(exit {srv.returncode}). Stale binary? Rebuild with "
                             f"`zig build bench-isa`.\n  " + "\n  ".join(log))
        confirm_arm(out, arm)
        # Upload rows first: each arm gets a fresh server, so the corpus has to
        # be rebuilt per arm. That is the cost of holding everything but the
        # instruction set fixed.
        setup = ["W0-upload", "W2"]
        if any(r in rows for r in ("W6", "W7", "W8")):
            setup += [f"W{n}-upload" for n in (6, 7, 8) if f"W{n}" in rows]
        env = dict(os.environ, RESULTS_DIR=str(out), STRAWMANN_ROOT=str(ROOT))
        r = subprocess.run(client_argv(port, arm.name, [*setup, *rows], client_cpus, perf),
                           env=env, stdout=(out / "run.log").open("w"),
                           stderr=subprocess.STDOUT, timeout=7200)
        # A failed arm used to leave an empty result directory while the sweep
        # printed "sweep complete", so the only symptom was an empty table two
        # steps later. Fail where the failure happens.
        if r.returncode != 0:
            tail = (out / "run.log").read_text().strip().splitlines()[-6:]
            raise SystemExit(f"arm {arm.name} failed (exit {r.returncode}):\n  "
                             + "\n  ".join(tail))
        if not (out / "rows.json").exists():
            raise SystemExit(f"arm {arm.name} wrote no rows.json to {out}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=15)
        except subprocess.TimeoutExpired:
            srv.kill()
    return out


def cmd_run(args) -> int:
    todo = arms(args.arms)
    missing = [a.name for a in todo if not a.built]
    if missing:
        raise SystemExit(f"not built: {', '.join(missing)}\nRun: zig build bench-isa")
    rows = args.rows or DEFAULT_ROWS
    print(f"{len(todo)} arm(s) x {args.reps} rep(s) x {len(rows)} row(s)")
    print(f"rows: {' '.join(rows)}\n")
    for rep in range(1, args.reps + 1):
        # Alternate arms within a rep rather than running each arm to
        # completion. A machine that warms up over an hour would otherwise
        # charge the whole drift to whichever arm ran last.
        for a in todo:
            waited = wait_quiet()
            print(f"  [rep {rep}] {a.name:<16} (quiet after {waited}s)", flush=True)
            run_arm(a, rows, rep, args.port, args.server_cpus, args.client_cpus, args.perf)
    print("\nsweep complete")
    return cmd_table(args)


def collect(rows_wanted: list[str]) -> dict[str, dict[str, list[float]]]:
    """arm -> row -> [qps across reps]"""
    acc: dict[str, dict[str, list[float]]] = {}
    for d in sorted(RESULTS.glob("*-rep*")):
        arm = d.name.rsplit("-rep", 1)[0]
        f = d / "rows.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text()):
            if r.get("qps") and (not rows_wanted or r["id"] in rows_wanted):
                acc.setdefault(arm, {}).setdefault(r["id"], []).append(r["qps"])
    return acc


def cmd_table(args) -> int:
    rows_wanted = args.rows or DEFAULT_ROWS
    acc = collect(rows_wanted)
    if not acc:
        print(f"no results under {RESULTS}; run `isa_sweep.py run` first")
        return 1
    base = args.vs
    order = [a.name for a in arms(None) if a.name in acc]
    rows = [r for r in rows_wanted if any(r in acc[a] for a in order)]

    print(f"§7.5 forced-ISA matrix, queries/second (median of reps), vs {base}\n")
    head = f"{'row':<8} {'expectation':<34}" + "".join(f"{a:>16}" for a in order)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r:<8} {EXPECTED.get(r, ''):<34}"
        b = acc.get(base, {}).get(r)
        bmed = statistics.median(b) if b else None
        for a in order:
            v = acc[a].get(r)
            if not v:
                line += f"{'-':>16}"
                continue
            med = statistics.median(v)
            if bmed and a != base:
                line += f"{med:>10,.0f} {med / bmed:>4.2f}x"
            else:
                line += f"{med:>16,.0f}"
        print(line)

    print()
    ctrl = [r for r in rows if r in CONTROL_ROWS]
    if ctrl and base in acc:
        print("controls (no distance arithmetic; a ratio far from 1.00x here means the")
        print("measurement is picking up something other than the instruction set):")
        for r in ctrl:
            b = acc[base].get(r)
            if not b:
                continue
            bmed = statistics.median(b)
            # Guarded the way the table above guards the same variable. A
            # control row that measured zero is a failed run rather than a
            # ratio of zero, and dividing by it crashes the sweep at the point
            # where it would otherwise report the failure.
            if not bmed:
                print(f"  {r}: {base} measured 0, so no ratio is formed")
                continue
            spread = [f"{a} {statistics.median(acc[a][r]) / bmed:.2f}x"
                      for a in order if acc[a].get(r) and a != base]
            print(f"  {r}: {'  '.join(spread)}")

    reps = {len(v) for arm in acc.values() for v in arm.values()}
    if reps == {1}:
        print("\nSingle repetition per arm: no variance estimate, so a ratio near 1.00x")
        print("cannot be distinguished from noise. Re-run with --reps 3 to bound it.")
    else:
        print("\nper-row spread across reps (max-min as % of median):")
        for r in rows:
            # `default=`: a row every arm measured once has no spread to
            # report, and an empty `max()` here used to end the sweep's own
            # summary with a ValueError after the measurements were in.
            worst = max((((max(v) - min(v)) / statistics.median(v) * 100)
                         for a in order if (v := acc[a].get(r)) and len(v) > 1),
                        default=None)
            print(f"  {r:<8} {worst:>5.1f}%" if worst is not None
                  else f"  {r:<8}   n/a  (one rep per arm)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="§7.5 forced-ISA workload sweep.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the arms and what each isolates")
    for name, helptext in (("run", "sweep the arms"), ("table", "matrix from existing results")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--arms", nargs="*", help="limit to these arms")
        p.add_argument("--rows", nargs="*", help="limit to these workload rows")
        p.add_argument("--vs", default="baseline", help="arm to compare against")
        p.add_argument("--results", default=None,
                       help="where the arms' results go (default bench/results/isa)")
        if name == "run":
            p.add_argument("--reps", type=int, default=1)
            p.add_argument("--port", type=int, default=6334)
            p.add_argument("--server-cpus", default=None,
                           help="pin each arm's server to this set, e.g. 4-11")
            p.add_argument("--client-cpus", default=None,
                           help="run the load generator on this set, e.g. 0-3")
            p.add_argument("--perf", nargs="?", const="default", default=None,
                           help="the perf stat sidecar's event set, per row")
    paths.add_data_argument(ap)
    args = ap.parse_args(argv[1:])
    # Exported, so the `workloads.py` this spawns per arm reads the same root.
    paths.use_data_dir(args.data_dir)
    if args.cmd in ("run", "table") and args.results:
        global RESULTS
        RESULTS = Path(args.results).resolve()
    if not args.cmd:
        return cmd_list(args)
    return {"list": cmd_list, "run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
