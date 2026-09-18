#!/usr/bin/env python3
"""Regression or noise? Compare two runs against a measured noise floor.

    regression.py <baseline> <candidate>
    regression.py <baseline> <candidate> --noise bench/results/noise.json
    regression.py --measure-noise <label-glob>    build the noise floor

This is the redline function. Everything else in `bench/` produces numbers; this
is the part that says whether a difference between two of them means anything.

## Why a noise floor is not optional

Without one, a verdict is eyeballed. "7,588 against 7,260, that looks like an
improvement" is not a finding if the same row measured twice on an unchanged
binary spans 8%. The question a redline exists to answer is precisely the one
that cannot be answered from a single pair of observations.

So a comparison here refuses to render a verdict for any row that has no noise
estimate, rather than falling back on a fixed threshold. A made-up threshold is
worse than no verdict: it looks like a measurement.

## The test

For each row, the noise floor gives a relative standard deviation `s` estimated
from repeated passes over an unchanged binary. A change of `d` (relative) is
called:

    regression / improvement   |d| > max(3s, min_effect)
    inconclusive               otherwise

Three sigma rather than two because the sweep tests many rows at once, and at
two sigma a table of twenty rows produces roughly one false call per run by
construction. `min_effect` (default 2%) is a floor for rows whose measured `s`
is implausibly small, which happens when few repetitions happen to agree.

This is deliberately blunt. The distribution of a throughput measurement is not
normal, and a proper treatment would use the per-rep samples rather than a
summary. That is worth doing when the noise floor has enough repetitions to
support it; with six, a t-test would give false precision.

## What it does not do

It does not know *why* something changed. A confirmed regression on W6 and not
on W3 points at the quantized path, and that is a hint rather than a diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "bench/results"
DEFAULT_NOISE = RESULTS / "noise.json"

#: Sigma multiplier. See the module docstring: two sigma over a twenty-row table
#: is about one false call per run.
SIGMA = 3.0
#: Floor on what counts as an effect, whatever the measured spread says.
MIN_EFFECT = 0.02


#: What `harness_hash` says when the repetitions disagree.
#:
#: A sentinel inside a hash-valued field, which is a smell worth naming even
#: where it is not worth changing: `noise.json` is on disk and readers already
#: parse it. Unrelated to `Gate.mixed` and `Direction.mixed` beyond the word.
MIXED_STAMP = "mixed"


def noise_band(rsd: float, arms: int = 1, min_effect: float = MIN_EFFECT) -> float:
    """The relative change that clears the noise floor, for `rsd` per arm.

    One definition, used by both consumers. `regression.py` compares one
    engine against itself (`arms=1`): `max(3*rsd, 2%)`. `report.py` ratios two
    engines measured with the same spread (`arms=2`), so the spreads add in
    quadrature: `3*sqrt(2)*rsd`, with the same floor. The two files each had
    their own arithmetic and `docs/validation.md` described a third (`3*rsd`,
    no floor); this is now the only place the threshold is written down.
    """
    return max(SIGMA * (arms ** 0.5) * rsd, min_effect)

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = DIM = BOLD = OFF = ""


@dataclass
class Noise:
    """Relative standard deviation per row, and how it was obtained."""
    rsd: dict[str, float]
    reps: dict[str, int]
    source: str
    discarded: list[str]
    #: `workloads.stamp_hash` of the harness stamp the repetitions ran under
    #: (their `run.json`), `"mixed"` if they disagree, None when unstamped. A
    #: floor measured by one harness configuration describes that
    #: configuration's spread; a comparison of rows stamped differently says
    #: so beside the verdicts rather than refusing them.
    harness_hash: str | None = None
    #: How many repetition directories were read, contaminated ones included.
    n_dirs: int = 0
    #: The date the repetitions were measured, from their own rows, so that
    #: recomputing the floor from the same directories does not restamp it with
    #: the date of the recomputation.
    measured_on: str | None = None
    #: Which engine the repetitions ran against, from their `run.json`
    #: `engine_comm`. A floor is measured over one label's repetition
    #: directories, so it is always *one* engine's spread — and the file did
    #: not say which. `bench/results/noise.json` on this host is six passes of
    #: strawmANN, and `compare.py` was banding a strawmANN-vs-
    #: Qdrant *ratio* with it. Findings 38: Qdrant's run-to-run spread on W3 is
    #: 13.16% against the 0.46% floor that banded it.
    arms: list[str] | None = None
    #: The corpus and the environment hash the repetitions ran on, the two
    #: keys `aggregate.py` writes and `compare.parity_band` refuses on. This
    #: type had neither, so `--measure-noise` wrote floors nothing could
    #: refuse and `compare` judged rows against them.
    dataset: str | None = None
    env_hash: str | None = None

    def of(self, row: str) -> float | None:
        return self.rsd.get(row)

    @classmethod
    def from_dict(cls, raw: dict) -> Noise:
        return cls(raw.get("rsd", {}), raw.get("reps", {}), raw.get("source", "?"),
                   raw.get("discarded", []), raw.get("harness_hash"),
                   raw.get("n_dirs", 0), raw.get("measured_on"), raw.get("arms"),
                   raw.get("dataset"), raw.get("env_hash"))


def engine_of(d: Path) -> str | None:
    """Which engine a result directory measured, from its `run.json`.

    `engine_comm` rather than the label, because the label is chosen per run
    (`sm-dbp1m`, `qdrant`, `nf/rep3`) and the engine is what a spread belongs
    to.
    """
    p = d / "run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("engine_comm")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def dataset_of(d: Path) -> str | None:
    p = d / "run.json"
    if not p.exists():
        return None
    try:
        return (json.loads(p.read_text()).get("dataset") or {}).get("name")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def env_hash_of(d: Path) -> str | None:
    p = d / "run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("env_hash")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def floor_refusals(noise: Noise, labels: list[str]) -> list[str]:
    """Why this floor may not band these labels' rows: the dataset and
    environment rules `compare.parity_band` and `report.load_noise` apply,
    which the verdict tool did not. Empty means the floor applies."""
    out = []
    if noise.dataset:
        rows_ds = {dataset_of(RESULTS / lbl) for lbl in labels} - {None}
        if rows_ds and rows_ds != {noise.dataset}:
            out.append(f"noise floor was measured on {noise.dataset} and these rows are "
                       f"{', '.join(sorted(rows_ds))}; spread does not carry between corpora")
    if noise.env_hash:
        rows_env = {env_hash_of(RESULTS / lbl) for lbl in labels} - {None}
        if rows_env and rows_env != {noise.env_hash}:
            out.append("noise floor was measured under another environment hash than "
                       "these rows (SMT, governor, isolation: findings 46)")
    return out


def stamp_hash_of(d: Path) -> str | None:
    """The harness stamp hash a result directory's `run.json` carries, or None."""
    p = d / "run.json"
    if not p.exists():
        return None
    try:
        stamp = json.loads(p.read_text()).get("harness")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not stamp:
        return None
    from workloads import stamp_hash
    return stamp_hash(stamp)


def floor_for(labels: list[str]) -> dict:
    """The run-to-run spread that describes *these* labels, as a `noise.json` dict.

    `aggregate.py --reps N` already writes each folded label its own
    `noise.json`: the spread of that engine, on that corpus, on that binary, in
    that session. Nothing read it. Both consumers went straight to the global
    `bench/results/noise.json`, which on this host is a sift1m fold measured
    on strawmANN alone — so a repeated dbpedia run measured its own
    floor and was still judged against another corpus's, or against none.

    A row is kept only where *every* arm that carries a floor measured it: a
    ratio's spread is both arms' spread, and one arm's is not it. The two are
    combined as `sqrt(mean(rsd^2))`, which is the value `noise_band(·, arms=2)`
    turns into `3*sqrt(rsd_a^2 + rsd_b^2)` — the band for two arms with
    *different* measured spreads, where the borrowed floor could only assume
    one shared one. That assumption is the "using it for both sides assumes
    Qdrant is no noisier" caveat, and a run that measures both retires it.

    Falls through to the global file when no label carries a fold, which is
    every single-pass run.
    """
    found = []
    for label in labels:
        p = RESULTS / label / "noise.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("rsd"):
            found.append((label, d))
    if not found:
        return _read_floor(DEFAULT_NOISE)
    rows = set.intersection(*(set(d["rsd"]) for _, d in found))
    rsd = {r: (sum(d["rsd"][r] ** 2 for _, d in found) / len(found)) ** 0.5
           for r in rows}
    first = found[0][1]
    stamps = {d.get("harness_hash") for _, d in found}
    datasets = {d.get("dataset") for _, d in found}
    envs = {d.get("env_hash") for _, d in found}
    return {
        "rsd": rsd,
        "reps": {r: min(d.get("reps", {}).get(r, 0) for _, d in found) for r in rows},
        "source": "; ".join(d.get("source", lab) for lab, d in found),
        "discarded": [x for _, d in found for x in (d.get("discarded") or [])],
        "harness_hash": (stamps.pop() if len(stamps) == 1 else MIXED_STAMP),
        "n_dirs": sum(d.get("n_dirs", 0) for _, d in found),
        "measured_on": max((d.get("measured_on") or "" for _, d in found), default=None)
                       or first.get("measured_on"),
        "dataset": (datasets.pop() if len(datasets) == 1 else None),
        #: One machine's spread or none. Arms folded from different environment
        #: hashes cannot share a floor any more than arms from different
        #: corpora can, so a disagreement resolves to `None` and the floor
        #: declines to name an environment rather than naming the wrong one.
        "env_hash": (envs.pop() if len(envs) == 1 else None),
        #: Which arms the spread was measured on, as engines rather than
        #: labels: a label is per run (`sm-dbp1m`, `qdrant`) and a spread
        #: belongs to the engine. `measure_noise` records the same thing for
        #: the global file, so one key means one thing in both.
        "arms": sorted({engine_of(RESULTS / lab) or lab for lab, _ in found}),
    }


def _read_floor(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return d if isinstance(d, dict) else {}


def load_rows(label: str) -> dict[str, dict]:
    p = RESULTS / label / "rows.json"
    if not p.exists():
        raise SystemExit(f"no results for {label!r} at {p}")
    return {r["id"]: r for r in json.loads(p.read_text())}


def measure_noise(dirs: list[Path], source: str) -> Noise:
    """Relative standard deviation per row, from repeated identical passes.

    Contaminated repetitions are **discarded, not averaged in**. A rep that ran
    alongside a compiler measures interference; folding it in inflates the
    noise floor, which then hides real regressions behind a threshold that was
    never about noise. An earlier attempt at this reported 5-30% and was
    measuring eight concurrent rustc processes.
    """
    per_row: dict[str, list[float]] = {}
    discarded: list[str] = []
    hashes: set[str | None] = set()
    when: list[str] = []
    arms: set[str] = set()
    datasets: set[str | None] = set()
    envs: set[str | None] = set()
    n_dirs = 0
    for d in sorted(dirs):
        f = d / "rows.json"
        if not f.exists():
            continue
        n_dirs += 1
        hashes.add(stamp_hash_of(d))
        arms.add(engine_of(d) or d.name)
        datasets.add(dataset_of(d))
        envs.add(env_hash_of(d))
        rows = json.loads(f.read_text())
        when += [r["when"][:10] for r in rows if r.get("when")]
        dirty = [r["id"] for r in rows if r.get("foreign")]
        if dirty:
            discarded.append(f"{d.name}: {', '.join(dirty)}")
            continue
        for r in rows:
            if r.get("qps"):
                per_row.setdefault(r["id"], []).append(r["qps"])

    rsd, reps = {}, {}
    for row, vals in per_row.items():
        reps[row] = len(vals)
        if len(vals) >= 3:
            rsd[row] = statistics.stdev(vals) / statistics.mean(vals)
    stamped = hashes - {None}
    harness_hash = (stamped.pop() if len(stamped) == 1 and not (hashes - stamped)
                    else MIXED_STAMP if len(hashes) > 1 else None)
    return Noise(rsd, reps, source, discarded, harness_hash, n_dirs,
                 max(when) if when else None, sorted(arms),
                 datasets.pop() if len(datasets) == 1 else None,
                 envs.pop() if len(envs) == 1 else None)


def rep_dirs(base: Path) -> list[Path]:
    """The repetition directories a `--measure-noise` argument names.

    A glob is expanded as given. A directory is searched for `rep*` (the
    noise-floor layout `noisefloor.sh` writes) and for `<label>-rep<n>`, which
    is what `qdrant_ab.py --reps` and `isa_sweep.py` write; the second form
    used to be invisible here, so the invocation `qdrant_ab.py` printed at
    the end of its own run exited with "no repetition directories".
    """
    if any(c in base.name for c in "*?["):
        return sorted(p for p in base.parent.glob(base.name) if p.is_dir())
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir()
                  if p.is_dir() and (p.name.startswith("rep") or "-rep" in p.name))


def cmd_measure(args) -> int:
    base = Path(args.measure_noise)
    dirs = rep_dirs(base)
    if not dirs:
        raise SystemExit(f"no repetition directories under {base}")
    n = measure_noise(dirs, str(base))
    # The source path is provenance, and provenance that points at a session
    # scratchpad is provenance with a deletion date: `noise.json` recorded a
    # path under `/tmp` that no longer existed, so the floor quoted in
    # `docs/validation.md` could not be re-derived once that directory went
    # away. Record what survives it: how many repetitions, over which rows, on
    # what date.
    # The date the rows were *measured*, read from the rows, not the date this
    # ran: recomputing the floor from the same repetition directories used to
    # restamp it with today, which is how a measurement came to
    # describe itself as.
    n.source = (f"{base} ({len(dirs)} repetitions, "
                f"{n.measured_on or time.strftime('%Y-%m-%d', time.gmtime())})")
    DEFAULT_NOISE.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else DEFAULT_NOISE
    out.write_text(json.dumps(asdict(n), indent=2) + "\n")

    print(f"noise floor from {len(dirs)} repetition(s) -> {out}\n")
    if n.discarded:
        print(f"{YELLOW}discarded {len(n.discarded)} contaminated rep(s){OFF}, because a rep")
        print("that ran alongside something else measures interference, not jitter:")
        for d in n.discarded:
            print(f"    {d}")
        print()
    if not n.rsd:
        print(f"{RED}No row has 3+ clean repetitions.{OFF} No noise floor was produced;")
        print("a comparison against this file would refuse every verdict, correctly.")
        return 1
    print(f"  {'row':<12} {'reps':>4} {'rsd':>7} {'detectable at 3 sigma':>24}")
    for row in sorted(n.rsd):
        s = n.rsd[row]
        thr = noise_band(s)
        print(f"  {row:<12} {n.reps[row]:>4} {s * 100:>6.2f}% {thr * 100:>22.1f}%")
    thin = [r for r, c in n.reps.items() if c < 3]
    if thin:
        print(f"\n  {DIM}no estimate (fewer than 3 clean reps): {', '.join(sorted(thin))}{OFF}")
    return 0


def floor_stamp_notes(noise: Noise, labels: list[str]) -> list[str]:
    """Where the floor's harness stamp and the compared rows' disagree, as notes.

    A note, not a refusal: the floor is a spread, and a spread measured under
    a slightly different harness configuration is still the best estimate
    there is; what a reader must not do is take it as measured on these rows.
    """
    if not noise.rsd:
        return []
    out = []
    if noise.harness_hash is None:
        out.append("noise floor carries no harness stamp (measured before floors "
                   "were stamped); its spread may not describe these rows")
        return out
    if noise.harness_hash == MIXED_STAMP:
        out.append("noise floor was measured over repetitions with differing harness "
                   "stamps; its spread is not one configuration's")
        return out
    for lbl in labels:
        h = stamp_hash_of(RESULTS / lbl)
        if h is not None and h != noise.harness_hash:
            out.append(f"noise floor was measured under a different harness stamp "
                       f"than {lbl}'s rows; its spread may not describe them")
    return out


def cmd_compare(args) -> int:
    a, b = load_rows(args.baseline), load_rows(args.candidate)
    # The floor that describes *these* labels (`floor_for`: a folded label's
    # own `noise.json` first, the global file otherwise), read the way the
    # two table renderers read it rather than straight from the global file.
    if args.noise:
        raw = json.loads(Path(args.noise).read_text()) if Path(args.noise).exists() else {}
    else:
        raw = floor_for([args.baseline, args.candidate])
    noise = Noise.from_dict(raw) if raw.get("rsd") else Noise({}, {}, "(none)", [])
    refused = floor_refusals(noise, [args.baseline, args.candidate])
    if refused:
        # A floor from another corpus or another environment bands nothing:
        # this tool printed REGRESSION (30%) from exactly such a floor.
        noise = Noise({}, {}, f"{noise.source} (refused)", noise.discarded)

    print(f"{BOLD}{args.baseline} -> {args.candidate}{OFF}")
    print(f"noise floor: {noise.source if noise.rsd else 'NONE'}"
          f"{'' if noise.rsd else '  (no verdicts will be rendered)'}")
    for why in refused:
        print(f"{YELLOW}floor refused:{OFF} {why}")
    for note in floor_stamp_notes(noise, [args.baseline, args.candidate]):
        print(f"  {YELLOW}{note}{OFF}")
    print()

    print(f"  {'row':<12} {'baseline':>10} {'candidate':>10} {'change':>9} "
          f"{'noise':>7}  verdict")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 9} {'-' * 7}  {'-' * 30}")

    regressions = improvements = inconclusive = unknown = 0
    contaminated: list[str] = []
    reoffered: list[str] = []
    for row in [r for r in a if r in b]:
        qa, qb = a[row].get("qps"), b[row].get("qps")
        if qa is None or qb is None or qa <= 0:
            continue
        d = (qb - qa) / qa
        s = noise.of(row)
        dirty = a[row].get("foreign") or b[row].get("foreign")
        # An open-loop row's qps is the rate it was asked for, not a capacity it
        # reached, so two runs at different `--rps-reference` differ by exactly
        # the ratio of the references. That read as
        # `improvement (4.3%)` on three rows of *both* engines over a 0.00%
        # floor — zero because the offered rate is pinned, which is what made
        # the arithmetic exact and the finding pure instrument.
        offered = (a[row].get("rps_target"), b[row].get("rps_target"))
        moved = (a[row].get("load_mode") == "open-loop"
                 and all(isinstance(o, (int, float)) for o in offered)
                 and offered[0] != offered[1])

        if moved:
            verdict, colour = (f"not attributable: offered rate changed "
                               f"({offered[0]:,.0f} -> {offered[1]:,.0f} rps)"), YELLOW
            reoffered.append(row)
        elif dirty:
            # A contaminated row's difference is not attributable to the change
            # under test, whatever the arithmetic says, so it gets no verdict
            # and no place in the tally: `decisions.md` makes the per-row
            # foreign-load flag a hard gate, and a "REGRESSION [contaminated]"
            # that still counted towards the exit code was a soft one.
            verdict, colour = f"not attributable: contaminated ({dirty})", YELLOW
            contaminated.append(row)
        elif s is None:
            verdict, colour = "no noise estimate", DIM
            unknown += 1
        else:
            thr = noise_band(s)
            if abs(d) <= thr:
                verdict, colour = f"inconclusive (<{thr * 100:.1f}%)", DIM
                inconclusive += 1
            elif d < 0:
                verdict, colour = f"REGRESSION ({abs(d) * 100:.1f}%)", RED
                regressions += 1
            else:
                verdict, colour = f"improvement ({d * 100:.1f}%)", GREEN
                improvements += 1
        print(f"  {row:<12} {qa:>10,.0f} {qb:>10,.0f} {d * 100:>8.1f}% "
              f"{(s * 100 if s else 0):>6.2f}%  {colour}{verdict}{OFF}")

    print(f"\n  {regressions} regression(s), {improvements} improvement(s), "
          f"{inconclusive} inconclusive, {unknown} with no noise estimate")
    if contaminated:
        print(f"  {YELLOW}{len(contaminated)} row(s) not attributable: measured under "
              f"foreign load ({', '.join(contaminated)}); excluded from the tally and "
              f"the exit code.{OFF}")
    if reoffered:
        print(f"  {YELLOW}{len(reoffered)} open-loop row(s) not attributable: the two runs "
              f"were given different `--rps-reference`, so their qps is the offered rate "
              f"and not a result ({', '.join(reoffered)}); excluded from the tally and "
              f"the exit code.{OFF}")
    if unknown:
        print(f"  {DIM}Rows without a noise estimate get no verdict. A fixed threshold")
        print(f"  would look like a measurement and would not be one.{OFF}")
    if not noise.rsd:
        print(f"\n  {YELLOW}No noise floor exists.{OFF} Build one with:")
        print("    regression.py --measure-noise <dir-of-rep*-directories>")
    return 1 if regressions else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Regression or noise?")
    ap.add_argument("baseline", nargs="?")
    ap.add_argument("candidate", nargs="?")
    ap.add_argument("--noise", help=f"noise floor json (default {DEFAULT_NOISE})")
    ap.add_argument("--measure-noise", metavar="DIR",
                    help="build a noise floor from repetition directories")
    ap.add_argument("--out", help="where to write the noise floor")
    args = ap.parse_args(argv[1:])

    if args.measure_noise:
        return cmd_measure(args)
    if not (args.baseline and args.candidate):
        ap.print_help()
        return 2
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
