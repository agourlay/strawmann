#!/usr/bin/env python3
"""Is the per-build recall draw the *upload's* arrival order? (findings 34)

findings 34 measured three builds of one SIFT1M collection spreading 0.00288 of
recall@10 at `ef` 512 against Qdrant's 0.00009, and attributed it to
`buildParallel`'s thread interleaving. `zig build graph-diff` then built the
same corpus five times in one process, in file order, and the spread was
**0.00002** at `ef` 512 and 0.00011 at `ef` 128, which is Qdrant's level. At 8
builder threads, matching the server's pool, it was the same. So the pruning
race is not what the harness sees.

What the harness has and that experiment does not is a *concurrent upload*.
`ids.getOrInsert` hands out internal offsets as points arrive, W2 uploads with
`-b 100 -t 8 -p 8`, and `hnsw.assignLevel` is a pure function of the offset. So
which vector becomes node 7 is a per-pass draw, and with it the whole
upper-level membership and the entry point. Reordering arrival in the diff tool
widened the draw tenfold, which is the right direction and still short of what
the harness shows.

This runs the real path and changes one thing:

    concurrent   -t 8 -p 8   what every published run uploaded with
    serial       -t 1 -p 1   one stream, so arrival order is the file's

Same engine, same corpus, same recall sweep, three passes each. If the serial
arm's passes agree with each other and the concurrent arm's do not, the draw is
the upload's, and it is fixable without touching the builder: assign the offset
from the point's external id rather than from its arrival.

Not a timing measurement, so no host gate. Usage:

    upload_order_ab.py --server-cpus 4-12 --client-cpus 0-3 [--passes 3]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import fullrun

import recall
import workloads

#: name, bfb `-t`, bfb `-p`, and whether a second 1M collection's index build
#: is left running while this one uploads.
#:
#: The third arm is W1 followed by W2. W1 ingests `bench1` with
#: `--skip-wait-index`, so the server is still building that graph on its eight
#: workers when W2 creates `bench2`, uploads it and waits. Whether the two
#: builds overlap, and by how much, is a per-pass accident of timing, and it is
#: the one thing a published pass has that an isolated `bench2` does not.
ARMS: tuple[tuple[str, int, int, bool], ...] = (
    ("concurrent", 8, 8, False),
    ("serial", 1, 1, False),
    ("overlapped", 8, 8, True),
)

#: The sweep's widths. findings 34 quotes `ef` 512, and the lower rungs are
#: where a graph difference shows up largest.
EF = [128, 256, 512]

PORT = 6334


def upload(uri: str, label: str, threads: int, parallel: int, client_cpus: str,
           collection: str = "bench2", skip_wait: bool = False) -> int:
    """W2's bfb invocation with its upload concurrency as the variable.

    Spelled out rather than taken from `workloads.table()` because the row's
    `-t 8 -p 8` is exactly what this varies, and a table row that could be
    overridden by an environment variable is a published row that could be
    measured at a concurrency its stamp does not name.
    """
    results = fullrun.RESULTS / label
    results.mkdir(parents=True, exist_ok=True)
    argv = [
        "taskset", "-c", client_cpus,
        str(workloads.BFB),
        "--retry", "0", "--timeout", str(workloads.BFB_TIMEOUT_S), "--p9", "3",
        "--uri", uri,
        "--json", str(results / f"upload-{collection}.json"),
        *workloads.CREATE,
        "--collection-name", collection,
        "--fbin", str(workloads.corpus()),
        "-n", str(workloads.upload_n()),
        "-d", str(workloads.DIM),
        "-b", "100",
        "-t", str(threads),
        "-p", str(parallel),
        *(["--skip-wait-index"] if skip_wait else []),
    ]
    (results / f"upload-{collection}.cmd").write_text(" ".join(argv) + "\n")
    p = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    (results / f"upload-{collection}.stdout").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        print(p.stdout[-2000:], file=sys.stderr)
    return p.returncode


def one_pass(label: str, threads: int, parallel: int, overlap: bool,
             args) -> dict[int, float] | None:
    """Upload, build, sweep. Returns recall@10 per `ef`, or None if it failed."""
    fullrun.wipe_strawmann_storage()
    log = fullrun.RESULTS / label / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    n_server = fullrun.cpu_count(args.server_cpus)
    engine = fullrun.start_strawmann(
        args.server_cpus, PORT, log, max(1, n_server - 1), workloads.required_capacity())
    if engine is None:
        return None
    uri = f"http://localhost:{PORT}"
    try:
        # W1's ingest, whose index build this deliberately does not wait for.
        if overlap and upload(uri, label, 8, 8, args.client_cpus,
                              collection="bench1", skip_wait=True) != 0:
            return None
        if upload(uri, label, threads, parallel, args.client_cpus) != 0:
            return None
        if recall.sweep(uri, label, "bench2", queries=args.queries, ef=list(EF)) != 0:
            return None
    finally:
        fullrun.stop_strawmann(engine)
        # The graph the sweep scored, in the record beside it: `graph_builds`
        # carries the checksum, so two passes that agree about recall can still
        # be shown to have built different graphs.
        try:
            fullrun.record_graph_quality(label)
        except Exception:
            pass

    doc = json.loads((fullrun.RESULTS / label /
                      f"recall.{workloads.DATASET}.bench2.json").read_text())
    return {p["ef"]: p["recall_at_10"] for p in doc["points"]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True)
    ap.add_argument("--client-cpus", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--tag", default="uporder")
    ap.add_argument("--arms", default="", help="comma-separated subset of ARMS")
    args = ap.parse_args(argv[1:])

    got: dict[str, list[dict[int, float]]] = {}
    for arm, threads, parallel, overlap in ARMS:
        if args.arms and arm not in args.arms.split(","):
            continue
        got[arm] = []
        for i in range(1, args.passes + 1):
            label = f"{args.tag}-{arm}-p{i}"
            print(f"\n=== {arm} (-t {threads} -p {parallel}"
                  f"{', bench1 building alongside' if overlap else ''}), "
                  f"pass {i} of {args.passes}", flush=True)
            r = one_pass(label, threads, parallel, overlap, args)
            if r is None:
                print(f"  {label} failed; arm abandoned", file=sys.stderr)
                break
            got[arm].append(r)
            print("  " + "  ".join(f"ef {e} {r[e]:.5f}" for e in sorted(r)), flush=True)

    print(f"\n{'arm':<12} {'ef':>5}  " + "  ".join(f"{'pass ' + str(i):>9}"
                                                   for i in range(1, args.passes + 1)) +
          f"  {'spread':>9}")
    for arm, _, _, _ in ARMS:
        for e in EF:
            vals = [r[e] for r in got.get(arm, []) if e in r]
            if not vals:
                continue
            cells = "  ".join(f"{v:>9.5f}" for v in vals)
            print(f"{arm:<12} {e:>5}  {cells}  {max(vals) - min(vals):>9.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
