#!/usr/bin/env python3
"""§7.4's repetitions, folded into one publishable label.

    aggregate.py <label> <rep-label>...

§7.4 asks for "minimum three interleaved repetitions per cell; report median
of medians and the spread", and §7.2(5) for A/B/A/B rather than A-then-B. The
harness ran one pass per engine, so every published ratio was n=1 against n=1
and the only spread available was `noise.json`, measured on another binary in
another session over an already-built graph.

`fullrun.py --reps N` alternates the arms and writes each pass to its own
label; this folds them into the label the report reads. What it produces is
not an average of everything it was given:

  * the **median** qps per row, not the mean: one contaminated or thermally
    unlucky pass should move the answer by nothing, and with three passes a
    median cannot be moved by one outlier at all.
  * the **spread** beside it, as the relative standard deviation over the
    passes, which is the noise floor for that row measured *in this run* on
    *this binary* rather than borrowed. `regression.noise_band` consumes the
    same shape.
  * a row is aggregated only where all `n` passes report the *same* status --
    all `ok`, or all `n/a` (`W12` is `n/a` on strawmANN in every pass and must
    still fold, or the report loses its "declined, not implemented" note). A
    row that
    succeeded twice and failed once is not two thirds of a measurement, and
    silently taking the median of the survivors is how a flaky row becomes a
    published number.

Latency percentiles are medianed per percentile. That is not the same as the
p99 of the pooled requests — pooling would need every pass's raw timings and
would let a long pass dominate — and it is what §7.4's "median of medians"
names.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

import perfstat

import procstat
from regression import MIXED_STAMP, stamp_hash_of
from workloads import Gate

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))

#: Fields medianed across passes. Everything else is taken from the first pass
#: that carries it, because it describes the configuration rather than the
#: outcome: `ef`, `collection`, the harness stamp, the build identity.
#: Everything a fold takes the median of. Anything else is copied from pass 1,
#: which is right for configuration and wrong for a measurement.
#:
#: Built from the modules that produce the fields rather than typed out here,
#: because a hand-kept list of measured columns is a list that drifts. It had:
#: `ctx_switches_voluntary`, every PSI figure, the anon/file resident split and
#: every hardware counter were absent, so a `--reps 3` row would have carried
#: pass one's IPC, DRAM traffic and stall time beside a median qps, with
#: `reps: 3` on the row saying all of it was folded. Nothing would have looked
#: wrong.
#:
#: The string fields those modules also define — `io_source`, `psi_scope`,
#: `perf_set`, `perf_events`, `perf_note` — are excluded by `_median` itself,
#: which keeps only numbers; they describe the capture and belong to pass 1.
_BASE_NUMERIC = ("qps", "rps", "wall_s", "duration_s", "upload_s", "index_wait_s",
                 "storage_bytes",
                 "disk_read_bytes", "disk_write_bytes", "disk_read_ops",
                 "disk_write_ops", "syscall_reads", "syscall_writes")

NUMERIC = tuple(dict.fromkeys(
    _BASE_NUMERIC
    + tuple(procstat.SCHED_FIELDS)      # cpu, faults, runqueue, switches, ...
    + tuple(procstat.PSI_FIELDS)        # cgroup stall time
    + tuple(procstat.RSS_FIELDS)        # anon / file / shmem resident split
    + ("rss_peak_bytes",)
    + tuple(perfstat.PERF_FIELDS)))     # the hardware counters, `_median` drops
                                        # the three string ones

#: Numeric fields the fold takes from pass 1 instead of medianing, because they
#: describe the *configuration* the passes share rather than what any one pass
#: measured. Everything else numeric on a row is folded, derived from the rows
#: themselves rather than named anywhere.
#:
#: `NUMERIC` above named what to fold and is now only a floor under this list.
#: It was built from `procstat` and `perfstat` so their columns could not drift
#: out of it, and the half it could not derive -- `workloads.py`'s own -- drifted
#: exactly as its comment feared. Found by diffing a folded row
#: against its three passes: 87 values were pass one's, among them `seconds`,
#: `warmup_s`, `load_start`, `load_end`, `qps_bfb_median` and all four of W11's
#: write-overlap figures. The published W11 note read "write overlap 18.8%,
#: append 21,117 points/s" -- pass one's -- beside a throughput folded from
#: three, and `reps: 3` on the row said the whole of it was folded.
#:
#: Inverting the list inverts the failure mode with it: a column is folded by
#: having been measured, and a configuration column has to be named here to
#: escape. Naming one wrongly is visible, because `notes` reports any field on
#: this list that disagrees between passes -- a configuration that changed
#: mid-run is not a configuration, and that is worth a line either way.
CONFIG_NUMERIC = frozenset((
    "ef", "n_requested", "n_queries",
    # What the client offered. A constant of the row, and `notes` reports it if
    # two passes ever disagree, which would mean the row changed mid-run.
    "client_parallel", "client_threads", "client_connections",
    "rps_target", "rps_fraction", "saturation_qps",
    "quantization_oversampling",
    # Written by the fold itself, further down.
    "reps", "rep_rsd", "rep_drift",
))


def _median(vals: list) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


#: How far the passes must have moved, one way, before the move is called
#: drift rather than the ordering luck of three samples. See `_rep_drift`.
DRIFT_MIN = float(os.environ.get("DRIFT_MIN", 0.10))


def _rep_drift(vals: list) -> float | None:
    """Relative first-to-last change, when the passes moved one way only.

    `rep_rsd` describes a spread as if the passes were draws from one
    distribution. They are not always: on dbpedia-openai-1m,
    strawmANN's W10-ef128 read 3,662 then 3,001 then 2,764 qps — a 25% slide in
    one direction, at a constant clock, a constant instruction count per query,
    no major faults and no I/O wait. Folded as noise that became `rsd 14.8%`,
    which `regression.noise_band` turns into a +/-63% parity band, and the
    report then called a 1.18x ratio "no measured difference". A monotone slide
    is a property of the run rather than of the measurement, and the two want
    opposite treatment: noise is averaged out, drift invalidates the average.

    `None` when the passes are not monotone or the move is small, which is the
    ordinary case; two passes cannot show a trend and are `None` too. Sign is
    kept: which way it went is the first thing a reader asks.

    Monotonicity alone is weak evidence and the threshold carries the claim.
    Three exchangeable draws come out sorted one time in three, so at
    `MIN_EFFECT` this marked `W4 -2%` and `W6-ef64 +2%` on rows whose spread
    was plainly noise — and a mark refuses a ratio, so a threshold that low
    would refuse a third of a clean run. `DRIFT_MIN` is set where the move is
    too large to be the ordering luck of three samples that agree to within a
    couple of percent: on the run that prompted this it keeps W10-ef128 (-25%),
    W10-ef256 (-27%), W9 (-16%), W11 (-15%) and Qdrant's W6-ef32 (+65%), all of
    which carry an `rsd` of 8-25%, and drops the +/-2% rows.
    """
    got = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(got) < 3 or len(got) != len(vals) or got[0] == 0:
        return None
    rising = all(b > a for a, b in itertools.pairwise(got))
    falling = all(b < a for a, b in itertools.pairwise(got))
    if not (rising or falling):
        return None
    move = (got[-1] - got[0]) / abs(got[0])
    return move if abs(move) >= DRIFT_MIN else None


def _rsd(vals: list) -> float | None:
    """Relative standard deviation, or None below three passes.

    Two passes give a spread that is really one difference, and calling that a
    standard deviation invites a threshold to be built on it. `regression.py`
    already refuses a floor from fewer than three.
    """
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < 3:
        return None
    m = statistics.mean(vals)
    return (statistics.stdev(vals) / m) if m else None


def load_rows(label: str) -> list[dict]:
    p = ROOT / "bench/results" / label / "rows.json"
    return json.loads(p.read_text()) if p.exists() else []


def pass_spans(reps: list[str], passes: list[list[dict]]) -> list[dict]:
    """`[{label, started, last_row}]`, one per pass, from what each recorded."""
    out = []
    for rep, rows in zip(reps, passes, strict=True):
        p = ROOT / "bench/results" / rep / "run.json"
        try:
            started = json.loads(p.read_text()).get("started")
        except (OSError, json.JSONDecodeError):
            started = None
        stamps = sorted(r.get("when") for r in rows if isinstance(r.get("when"), str))
        out.append({"label": rep, "started": started,
                    "last_row": stamps[-1] if stamps else None})
    return out


def fold(passes: list[list[dict]]) -> tuple[list[dict], dict, list[str]]:
    """`(rows, rsd_by_row, notes)` from one list of rows per pass."""
    by_id: dict[str, list[dict]] = {}
    order: list[str] = []
    for rows in passes:
        for r in rows:
            if r["id"] not in by_id:
                by_id[r["id"]] = []
                order.append(r["id"])
            by_id[r["id"]].append(r)

    out, rsd, notes = [], {}, []
    n = len(passes)
    for wid in order:
        got = by_id[wid]
        if len(got) != n:
            notes.append(f"{wid}: measured in {len(got)} of {n} passes, not aggregated")
            continue
        # Agreement, which is what the docstring has always claimed and what
        # `oks and len(oks) != n` did not check: that form fires only when at
        # least one pass is `ok`, so three passes reading `n/a`, `FAILED`,
        # `FAILED` folded into one row wearing pass 1's `n/a` with `reps: 3`
        # and no note. `compare.py` and `report.py` key on the status value
        # alone, so that row published as a clean "declined, not implemented"
        # and `report_data` counted it into the headline declined tally.
        statuses = {r.get("status") for r in got}
        if len(statuses) != 1:
            seen = ", ".join(sorted(str(x) for x in statuses))
            notes.append(f"{wid}: statuses disagree across the {n} passes "
                         f"({seen}), not aggregated")
            continue
        dirty = [r for r in got if r.get("foreign")]
        merged = dict(got[0])
        # Every numeric column these passes carry, from the passes rather than
        # from a list. `bool` is an `int` subclass and a median of three flags
        # is not a flag, so they stay with the rest of pass 1's description.
        numeric = {k for r in got for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for key in sorted(numeric - CONFIG_NUMERIC):
            vals = [r.get(key) for r in got]
            have = [v for v in vals if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
            # The column-level form of the row-level refusal above. `_median`
            # drops non-numbers, so a counter the sampler produced in one pass
            # of three was medianed over that single value and published on a
            # row stamped `reps: 3` with nothing to distinguish it from a
            # three-pass median — `qd-dbp100k-perf`'s W8-upload
            # `ctx_switches_voluntary` is a shipped example.
            if len(have) != n:
                merged[key] = None
                notes.append(f"{wid}: {key} measured in {len(have)} of {n} "
                             f"passes, so no median is published for it")
                continue
            merged[key] = _median(vals)
        for key in sorted(numeric & CONFIG_NUMERIC):
            seen = {v for v in (r.get(key) for r in got) if v is not None}
            if len(seen) > 1:
                notes.append(f"{wid}: {key} differs between passes "
                             f"({sorted(seen)}); pass 1's kept, but a "
                             f"configuration the passes did not share is not one")
        lat = [r.get("latency") or {} for r in got]
        keys = {k for d in lat for k in d}
        if keys:
            # Same rule as the numeric columns above, and for the same reason:
            # `_median` drops non-numbers, so a percentile one pass of three
            # carried was medianed over that single value and published on a
            # row stamped `reps: 3` with nothing to say so. The percentiles
            # feed the comparison tables, so an unmarked pass-1 value there is
            # the same silently-wrong number the loop above refuses.
            folded = {}
            for k in sorted(keys):
                have = [d.get(k) for d in lat
                        if isinstance(d.get(k), (int, float))
                        and not isinstance(d.get(k), bool)]
                if len(have) != n:
                    folded[k] = None
                    notes.append(f"{wid}: latency {k} measured in {len(have)} "
                                 f"of {n} passes, so no median is published "
                                 f"for it")
                    continue
                folded[k] = _median(have)
            merged["latency"] = folded
        merged["reps"] = n
        # Said on the row, because a reader looking at one row should not have
        # to find the run-level record to learn it is a median of three.
        if dirty:
            merged["foreign"] = "; ".join(
                f"pass {i + 1}: {r['foreign']}" for i, r in enumerate(got) if r.get("foreign"))
        # The gate is the *worst* of the passes, not the first one's.
        #
        # Everything else here comes from `got[0]` because it describes the
        # configuration, which the passes share. The gate does not: it is a
        # verdict on the machine at the moment each pass ran. Taking pass 1's
        # propagated its failure to a folded row two thirds of which was
        # measured on a quiet box — and, in the other direction, would have
        # hidden a failure in passes 2 and 3 behind a clean pass 1, which is
        # the dangerous half. `gate_failures` says how many, so a reader is not
        # left reading one bad pass as three.
        failed = [i + 1 for i, r in enumerate(got)
                  if r.get("gate") not in (None, Gate.passed, "pass")]
        if failed:
            merged["gate"] = Gate.failed
            merged["gate_failures"] = failed
        elif any(r.get("gate") for r in got):
            merged["gate"] = Gate.passed
        s = _rsd([r.get("qps") for r in got])
        if s is not None:
            rsd[wid] = s
            merged["rep_rsd"] = s
        if (drift := _rep_drift([r.get("qps") for r in got])) is not None:
            merged["rep_drift"] = drift
            notes.append(f"{wid}: qps moved {drift:+.0%} monotonically across the "
                         f"{n} passes; that spread is a trend, not noise, and a "
                         f"band built from it would call a real change parity")
        out.append(merged)
    return out, rsd, notes


#: Recall fields medianed across passes. Point estimates only: the interval
#: beside them is handled separately, because a median of three Wilson
#: intervals is not an interval on the median.
RECALL_NUMERIC = ("recall_at_1", "recall_at_10", "recall_at_100",
                  "mean_relative_distance_error", "smoke_qps",
                  "short_lists", "impossible_scores")


def fold_recall(dest: Path, reps: list[str], notes: list[str]) -> None:
    """Median the recall sweeps too, and widen their intervals to the spread.

    The fold used to copy pass 1's `recall.*.json` verbatim while `rows.json`
    carried the median qps of three. The matched-recall table joins the two on
    `ef`, so it was pairing one build's recall with three builds' throughput —
    and on strawmANN those are not interchangeable. Two passes of the same
    collection, same parameters, same host, measured: recall@10 at
    `ef` 512 of 0.99957 and 0.99678, and mean relative distance error of
    8.61e-06 and 3.83e-04, a factor of 44. The parallel builder's insertion
    order is not deterministic, and at the top of the frontier the spread
    between two of its graphs is wider than the distance between the two
    engines.

    So: median the point estimates, per `ef`, per sweep file — §7.4's "median
    of medians" applied to the half of the measurement it had not reached. And
    take the interval as the *union* of the passes' own 95% intervals, which is
    conservative and, unlike a median of three intervals, cannot report a band
    narrower than the values it was computed from. `rep_spread` records the
    observed range so a reader can see the build variance rather than infer it.

    A sweep file present in some passes and not others is medianed over the
    passes that have it, and said so in `notes`: a missing sweep is a sweep
    that did not run, not a zero.
    """
    names = sorted({f.name for r in reps
                    for f in (ROOT / "bench/results" / r).glob("recall.*.json")})
    for name in names:
        docs, from_reps = [], []
        for r in reps:
            f = ROOT / "bench/results" / r / name
            if not f.exists():
                continue
            try:
                docs.append(json.loads(f.read_text()))
                from_reps.append(r)
            except (OSError, json.JSONDecodeError):
                continue
        if not docs:
            continue
        if len(docs) < len(reps):
            notes.append(f"{name}: present in {len(docs)} of {len(reps)} passes, "
                         f"medianed over those ({', '.join(from_reps)})")
        merged = dict(docs[0])
        by_ef: dict[object, list[dict]] = {}
        for d in docs:
            for pt in d.get("points") or []:
                by_ef.setdefault(pt.get("ef"), []).append(pt)
        points = []
        for ef in dict.fromkeys(pt.get("ef") for d in docs
                                for pt in d.get("points") or []):
            group = by_ef[ef]
            pt = dict(group[0])
            for key in RECALL_NUMERIC:
                vals = [g.get(key) for g in group]
                if any(isinstance(v, (int, float)) for v in vals):
                    pt[key] = _median(vals)
            los = [g.get("recall_at_10_ci95_low") for g in group]
            his = [g.get("recall_at_10_ci95_high") for g in group]
            los = [v for v in los if isinstance(v, (int, float))]
            his = [v for v in his if isinstance(v, (int, float))]
            if los and his:
                pt["recall_at_10_ci95_low"] = min(los)
                pt["recall_at_10_ci95_high"] = max(his)
            r10 = [g.get("recall_at_10") for g in group
                   if isinstance(g.get("recall_at_10"), (int, float))]
            pt["reps"] = len(group)
            if len(r10) > 1:
                #: The build variance itself, not folded away. This is the
                #: number that says whether a matched-recall ratio at the top
                #: of the frontier is a property of the engine or of the graph
                #: that run happened to build.
                pt["rep_spread"] = max(r10) - min(r10)
                pt["rep_values"] = r10
            points.append(pt)
        merged["points"] = points
        merged["reps"] = len(docs)
        merged["rep_labels"] = from_reps
        (dest / name).write_text(json.dumps(merged, indent=2) + "\n")


def main(argv: list[str]) -> int:
    # argparse, like the rest of the harness. `aggregate.py` and `compare.py`
    # were the last two reading `argv` by index.
    ap = argparse.ArgumentParser(
        prog="aggregate.py", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="the folded label the report reads")
    ap.add_argument("reps", nargs="+", metavar="REP",
                    help="the per-pass labels `fullrun.py --reps N` wrote")
    try:
        args = ap.parse_args(argv[1:])
    except SystemExit as e:
        return int(e.code or 0)
    label, reps = args.label, args.reps
    passes = [load_rows(r) for r in reps]
    missing = [r for r, rows in zip(reps, passes, strict=True) if not rows]
    if missing:
        print(f"no rows.json for {', '.join(missing)}", file=sys.stderr)
        return 1

    rows, rsd, notes = fold(passes)
    dest = ROOT / "bench/results" / label
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    # The run-level record comes from the first pass, plus what makes this
    # label a fold rather than a pass. Copied rather than regenerated: it
    # describes engines and a host that are no longer running.
    src = ROOT / "bench/results" / reps[0]
    for name in ("run.json", "env.txt", "conformance.json", "collections.json"):
        f = src / name
        if not f.exists():
            continue
        if name == "run.json":
            meta = json.loads(f.read_text())
            meta["reps"] = len(reps)
            meta["rep_labels"] = reps
            # The copy kept pass 1's `label` and `started`, so the folded
            # record introduced itself as `-rep1` and the report's "measured"
            # window covered a third of the run. One entry per pass: when it
            # started and when its last row did.
            meta["label"] = label
            meta["passes"] = pass_spans(reps, passes)
            (dest / name).write_text(json.dumps(meta, indent=2) + "\n")
        else:
            (dest / name).write_text(f.read_text())
    fold_recall(dest, reps, notes)

    # The spread measured here, in the shape `regression.py` and `report.py`
    # already read, so a run that repeats itself stops borrowing a floor from
    # another binary in another session.
    if rsd:
        # Stamped, like `regression.py --measure-noise` writes it. Without
        # `harness_hash` a floor cannot say which configuration's spread it is,
        # and `report.noise_provenance` falls to its "carries no harness stamp"
        # branch — the generic caveat instead of the specific one, which is the
        # gap the floor had until it was re-derived.
        stamps = {stamp_hash_of(ROOT / "bench/results" / r) for r in reps}
        stamped = stamps - {None}
        # `load_rows` here returns a list, not the id-keyed dict
        # `regression.load_rows` returns. Two functions, one name, two shapes.
        when = [w for r in reps for w in
                (x.get("when", "")[:10] for x in load_rows(r)) if w]
        (dest / "noise.json").write_text(json.dumps({
            "rsd": rsd,
            "reps": {k: len(reps) for k in rsd},
            "source": f"{label}: median of {len(reps)} interleaved passes "
                      f"({', '.join(reps)})",
            "discarded": [],
            "harness_hash": (stamped.pop() if len(stamped) == 1 and not (stamps - stamped)
                             else MIXED_STAMP if len(stamps) > 1 else None),
            "n_dirs": len(reps),
            "measured_on": max(when) if when else None,
            # Which dataset's spread this is. A floor is measured on one corpus
            # at one dimension, and run-to-run spread is not portable across
            # either: the sift1m floor (d=128) was the only one that
            # existed, so dbpedia-openai-100K rows at 0.94x, 1.00x and 1.01x
            # were reported against a band measured on a different dataset, or
            # silently against none. Unstamped floors stay usable and are
            # labelled; a floor stamped with *another* dataset is refused.
            "dataset": (json.loads((src / "run.json").read_text())
                        .get("dataset", {}).get("name")
                        if (src / "run.json").exists() else None),
            # Which machine's spread this is, on the same argument as
            # `dataset` above and for the same kind of mistake. SMT on or off
            # is in this hash, and it moved Qdrant's saturating row by 30%
            # while leaving strawmANN's alone, so a floor measured under one
            # is not a floor under the other (findings 46).
            "env_hash": (json.loads((src / "run.json").read_text())
                         .get("env_hash")
                         if (src / "run.json").exists() else None),
        }, indent=2) + "\n")

    print(f"{label}: {len(rows)} row(s) from {len(reps)} passes; "
          f"{len(rsd)} carry a measured spread")
    for n in notes:
        print(f"  !! {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
