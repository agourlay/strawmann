#!/usr/bin/env python3
"""What a result label *is*, before anything renders it.

Split out of `report.py`, which had grown to 3,728 lines across four concerns
that share only the `Run` type: this, the tables, the per-workload sections and
the charts. This is the one seam among them that points in a single direction —
it needs nothing from the other three, and they need it — so it is the split
that can be made without inventing an import cycle.

Keeping it separate is also where the gate verdict belongs. `report.py` used to
derive §7.1's result from `env.txt` here, a file written once per `workloads.py`
invocation, and reported an arm as `pass` whose search rows were stamped `FAIL`.
The rule has one home now, in `compare.gate_of_rows`, and this asks it.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import compare
import recall as recall_mod
from workloads import Gate, Status

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STRAWMANN_ROOT", HERE.parents[1]))
ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Zig's amber for strawmANN, Qdrant's own crimson for Qdrant. Two hues rather
#: than two shades, so the series stay distinct without a legend lookup and for
#: readers who cannot separate them by colour at all.
#: Series colours, per theme, chosen against three measured constraints rather
#: than by eye:
#:
#:   1. each colour vs its background >= 3:1, the WCAG minimum for graphical
#:      objects. Zig's amber #F7A41D is the project's identity colour and is
#:      right in the page chrome, but as a chart colour on white it is 2.04:1
#:      and the bars wash out.
#:   2. the two series vs *each other* >= 2:1. This is the constraint the first
#:      fix missed: darkening the amber to #B86E00 fixed its background contrast
#:      and left the two series 1.20:1 apart, so the lines were legible and
#:      indistinguishable at the same time.
#:   3. a lightness gap, not only a hue gap, so the pair survives being printed
#:      greyscale or read by someone with a red-green deficiency.
SERIES = {"strawmann": "#C97A00", "qdrant": "#8E0A2E"}
SERIES_DARK = {"strawmann": "#FFC65C", "qdrant": "#E8446A"}
FALLBACK = ("#4C9AFF", "#2EA44F", "#A970FF")


#: Upload and index rows: a duration, not a query rate.
UPLOAD_ROWS = ("W0-upload", "W1", "W2", "W6-upload", "W7-upload", "W8-upload",
               "W12-upload")

def colour(label: str, i: int = 0) -> str:
    return SERIES.get(label.lower(), FALLBACK[i % len(FALLBACK)])


def engine_of(run) -> str | None:
    """Which engine a run measured, from its own `run.json`.

    The label does not say: `--strawmann-label` takes any string, and the
    label-per-corpus rule means a `--dataset` run has to use one. `run.json`
    carries a block named for the engine, which is the engine's own account of
    itself rather than a naming convention.
    """
    return next((n for n in ("strawmann", "qdrant") if run.meta.get(n)), None)


def run_colour(run, i: int = 0) -> str:
    """A run's series colour, keyed on the engine rather than on its position.

    `colour(label, i)` matched the label against `SERIES` and otherwise fell
    back to `FALLBACK[i]` — the run's *index*. So the hue followed rank, not
    identity: a chart iterating a subset painted the same engine a different
    colour than the chart above it, and any label that is not literally
    "strawmann" or "qdrant" — every label a `--dataset` run can use, since a
    result label belongs to one corpus — lost the engine palette entirely.
    """
    return (SERIES.get(run.label.lower())
            or SERIES.get(engine_of(run) or "")
            or FALLBACK[i % len(FALLBACK)])


#: What bfb's quantization names mean to a reader who did not send the flag.
_QUANT_NAMES = {"scalar": "SQ8 scalar", "binary": "binary (1 bit/dim)",
                "product": "PQ (product)", None: "none (fp32)"}


#: What an engine that names nothing for a field is actually using. Only
#: fields whose default is defined by the wire protocol rather than by a
#: build: a dense vector with no datatype is `float32` for both engines.
_DEFAULTS = {"datatype": "float32 (default)"}


def glance_of(runs: list[Run]) -> dict | None:
    """The four facts a reader needs before any number means anything.

    Dataset, size, storage datatype and quantization were all on the page
    already — spread across the Dataset card, the collection tables and the
    per-row configs — which is a different thing from being *stated*. A reader
    who has to assemble the subject of the measurement from four places has to
    already know what they are looking for.

    Read back from the engines (`collections.json`) rather than from what the
    harness asked for, on the same principle as the collection tables: the
    request is not the configuration. Where the engines disagree, both are
    shown; where one does not report a field at all, that is said rather than
    left blank, because a blank reads as fp32 to anyone who assumes a default.
    """
    ds = next((r.meta.get("dataset") for r in runs if r.meta.get("dataset")), None)
    if not ds:
        return None
    cols = {r.label: ((r.collections or {}).get("collections") or []) for r in runs}

    def field(key: str) -> list[str]:
        """One value per engine, or "not reported" — never a blank.

        `bench0` is excluded throughout: it is the d=4 synthetic floor, and its
        datatype says nothing about the corpus.
        """
        vals = []
        for cs in cols.values():
            seen = {c.get(key) for c in cs if c.get("collection") != "bench0"}
            seen.discard(None)
            if seen:
                vals.append(", ".join(sorted(str(v) for v in seen)))
            elif cs:
                # Captured, and the engine named nothing: it is on its default,
                # which is a fact about the engine and not a gap in the record.
                # "not reported" beside another engine's `float32` reads as a
                # missing capability; Qdrant simply does not echo a datatype it
                # was not given, and its dense default is float32.
                vals.append(_DEFAULTS.get(key, "default"))
            else:
                vals.append("not reported")
        return vals

    dtypes = field("datatype")
    # One line when the engines agree, which is the ordinary case; both when
    # they do not, because that difference is the reader's to weigh.
    dtype = dtypes[0] if len(set(dtypes)) == 1 else " · ".join(
        f"{lbl} {v}" for lbl, v in zip(cols, dtypes, strict=True))
    # `bench2` is what the headline rows search — W3, W4, W10 and the recall
    # frontier all query it — so its encoding is *the* answer to "which
    # quantization". The others are separate collections measured by their own
    # rows, and listing all four as one value read as though a single run had
    # used them simultaneously.
    def quant_of(name: str):
        for cs in cols.values():
            for c in cs:
                if c.get("collection") == name:
                    return c.get("quantization")
        return None

    headline = _QUANT_NAMES.get(quant_of("bench2"), str(quant_of("bench2")))
    others = sorted({c.get("quantization") for cs in cols.values() for c in cs
                     if c.get("collection") not in ("bench0", "bench2")
                     and c.get("quantization")})
    def num(v) -> str:
        # A result set that predates a field renders as "unknown", not as a
        # crash and not as a plausible zero.
        return f"{v:,}" if isinstance(v, (int, float)) else "unknown"

    n, dim = ds.get("n"), ds.get("dim")
    vectors = f"{num(n)} × {num(dim)}" if dim is not None else num(n)
    return {
        "dataset": ds.get("name") or "unknown",
        "vectors": vectors,
        "queries": f"{num(ds['n_queries'])} held-out queries"
                   if ds.get("n_queries") is not None else "",
        "metric": ds.get("metric") or "unknown",
        "datatype": dtype,
        "quantization": headline,
        "quantization_more": ", ".join(_QUANT_NAMES.get(q, str(q)) for q in others),
    }


def series_meta(label: str) -> dict:
    """The tag `THEME_JS` re-colours a trace by.

    It used to match on the first word of the trace name, which is the engine
    label on a series chart and *also* on the ambient-load chart, where the two
    traces per engine are `clean` (green) and `contaminated` (red) and the
    colour is the entire reading. Both were restyled to the engine's series
    colour on every load, so Qdrant's clean bars rendered crimson under a
    caption saying red means another process was on the box. Only a trace that
    opts in by carrying this tag is a series trace.
    """
    return {"series": label.lower()}


def run_meta_tag(run) -> dict:
    """`series_meta` keyed on the engine, for the same reason as `run_colour`.

    `THEME_JS` looks the tag up in a dict keyed `strawmann`/`qdrant`; a tag
    carrying an arbitrary label misses it and the trace is simply never
    restyled, so dark mode left those series in their light-mode hue.
    """
    return {"series": engine_of(run) or run.label.lower()}


class CheckKind(StrEnum):
    """One line of `setup.py check`'s output, by what it is.

    The closed set was a comment — `kind: str  # pass | fail | cont` — beside a
    field that four sites compare against literals. Distinct from
    `workloads.Gate`: this is one *check* within an arm's gate, `fail` is
    lowercase where `Gate.failed` is not, and `cont` has no counterpart at all.
    """

    #: A check that passed: `setup.py` prints it as `ok   <text>`.
    passed = "pass"
    #: A check that did not.
    failed = "fail"
    #: A continuation line belonging to the check above it — the explanation
    #: `setup.py` prints under a failure, which is not a check of its own.
    cont = "cont"


@dataclass
class Check:
    kind: CheckKind
    text: str


def _is_quiescence(text: str) -> bool:
    """`setup.py`'s quiescence line, whichever way it went: "machine is
    quiescent (...)" on a pass and "machine is NOT quiescent: ..." on a fail.
    Keyed on the pass spelling alone, a busy machine's FAIL line counted as a
    static property of the host and made two runs on one box "different
    machines"."""
    return text.startswith("machine is") and ("quiescent" in text or "busy" in text)


@dataclass
class Run:
    label: str
    rows: list[dict]
    env: str
    gate_pass: bool
    env_hash: str
    detail: dict[str, dict] = field(default_factory=dict)
    #: `run.json`: which engine, which dataset, which host, when. The report is
    #: read by people who did not run it, and a number whose engine build and
    #: dataset are unstated is not something they can check.
    meta: dict = field(default_factory=dict)
    #: `recall.json`: the conformance side of W10, joined on `ef`.
    recall: dict = field(default_factory=dict)
    #: The same sweep at limit 100, for §8.9's `recall@100`. Its own file and
    #: its own field because it is a different measurement: at limit 100 the
    #: engines search at `max(ef, limit)`, so it carries only the ef values at
    #: or above 100 and cannot stand in for the sweep above. Empty on a run
    #: measured before it was taken, which the recall table reports as such.
    recall_k100: dict = field(default_factory=dict)
    #: `collections.json`: what the engine said its collections are, as
    #: distinct from what the harness asked for. Empty for a run measured
    #: before the capture existed, which the section reports as such.
    collections: dict = field(default_factory=dict)
    #: `conformance.json`: which §8.5 tier the two engines reached together,
    #: and therefore what these numbers are allowed to be used for.
    conformance: dict = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if r["status"] == Status.ok)

    @property
    def n_declined(self) -> int:
        """Rows the engine declined, naming the construct (§12).

        Split out of the header count because `25/27 ok` reads as two failures,
        and on a run where nothing failed it sat next to a *green* pill saying
        the gate passed — the number and the colour disagreeing about what
        happened. Both of strawmANN's are W12: filtered search is §2 phase 3
        and §10's optional M7, "not built yet" rather than "broken".
        """
        return sum(1 for r in self.rows if r["status"] == Status.not_applicable)

    @property
    def n_measured(self) -> int:
        """Rows that were attempted, i.e. everything a decline did not remove."""
        return len(self.rows) - self.n_declined

    @property
    def checks(self) -> list[Check]:
        out = []
        for ln in self.env.splitlines():
            s = ln.strip()
            # Everything after the hash line is `setup.py`'s advice to the
            # operator ("run apply as root", "--lax MUST NOT ..."), not a
            # check; it used to be swept up as continuation lines and printed
            # in the gate card as if the machine had said it.
            if s.startswith("environment hash"):
                break
            if s.startswith("ok"):
                out.append(Check(CheckKind.passed, s[2:].strip()))
            elif s.startswith("FAIL"):
                out.append(Check(CheckKind.failed, s[4:].strip()))
            elif s and not s.startswith(("§", "(every", "Run ")) \
                    and "check(s) failed" not in s:
                out.append(Check(CheckKind.cont, s))
        return out

    @property
    def static_checks(self) -> list[Check]:
        """The §7.1 checks that describe the machine, not the moment.

        Governor, boost, SMT, isolation, THP, NUMA, the CPU model: the same
        answer for every run on this box until someone reboots it. The
        quiescence line ("machine is quiescent (load ...)" and its `busy:`
        continuation) is taken at the start of *this* run and is the only
        check that legitimately differs between two runs on one host.
        """
        out, skip_cont = [], False
        for c in self.checks:
            if c.kind == CheckKind.cont:
                if skip_cont:
                    continue
                out.append(c)
                continue
            skip_cont = _is_quiescence(c.text)
            if not skip_cont:
                out.append(c)
        return out

    @property
    def moment_checks(self) -> list[Check]:
        """The complement of `static_checks`: what the box was doing at run start."""
        out, keep_cont = [], False
        for c in self.checks:
            if c.kind == CheckKind.cont:
                if keep_cont:
                    out.append(c)
                continue
            keep_cont = _is_quiescence(c.text)
            if keep_cont:
                out.append(c)
        return out

    def by_id(self) -> dict[str, dict]:
        return {r["id"]: r for r in self.rows}

    def timings(self, wid: str, kind: str = "full_timings") -> list[float]:
        """Raw per-request times, in seconds.

        bfb names them `full_timings` (client round trip) and `server_timings`
        (what the server reported). Guessing `request_timings` returned nothing
        and the latency charts silently vanished from the report rather than
        erroring, which is the failure mode worth naming: an absent chart reads
        as "not applicable", not as "the key was wrong".
        """
        assert kind in ("full_timings", "server_timings"), kind
        d = self.detail.get(wid, {}).get("results", {}).get("search", {})
        v = d.get(kind)
        return v if isinstance(v, list) else []


def gate_pass_of(rows: list[dict], env: str) -> bool:
    """Whether §7.1 passed for these rows, from `compare.gate_of_rows`.

    This had its own copy of that rule, reading `"checks passed" in env.txt` —
    a file `workloads.py` writes per invocation and `fullrun` triggers more than
    once per arm. An arm whose 25 search rows were stamped `FAIL`
    was reported as `pass`, because the two-row invocation that ran last had
    passed. `mixed` is not a pass.
    """
    return compare.gate_of_rows(rows, env) == Gate.passed


def load_run(label: str) -> Run:
    d = ROOT / "bench/results" / label
    rows_p = d / "rows.json"
    if not rows_p.exists():
        raise SystemExit(f"no results for {label!r} at {rows_p}\n"
                         f"run: bench/harness/workloads.py run <uri> {label}")
    env_p = d / "env.txt"
    env = ANSI.sub("", env_p.read_text()) if env_p.exists() else "(no env.txt recorded)"
    m = re.search(r"environment hash: ([0-9a-f]+)", env)

    rows = json.loads(rows_p.read_text())
    detail = {}
    for r in rows:
        p = d / f"{r['id']}.json"
        if p.exists():
            with contextlib.suppress(json.JSONDecodeError):
                detail[r["id"]] = json.loads(p.read_text())
    def read_json(name: str) -> dict:
        p = d / name
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}

    meta = read_json("run.json")
    # `bench2`'s sweep, by (dataset, collection): `recall.json` used to be
    # read by name, and that name was also what `headline.py` wrote its
    # dbpedia cosine sweep to. `recall_mod.load_recall_json` checks the fields
    # inside the file against what is asked for.
    dataset = (meta.get("dataset") or {}).get("name") or "sift1m"
    return Run(label, rows, env, gate_pass_of(rows, env),
               m.group(1) if m else "unknown", detail,
               meta=meta, recall=recall_mod.load_recall_json(label, dataset, "bench2"),
               recall_k100=read_json(
                   recall_mod.recall_path(label, dataset, "bench2", 100).name),
               conformance=read_json("conformance.json"),
               collections=read_json("collections.json"))


# --------------------------------------------------------------------------
# The recall/throughput frontier
#
# Analysis, not rendering, and it lived in `report.py`'s chart section — which
# is what made that file impossible to split: `sections` read these two, the
# charts read `matched_recall_refusal` and `segment_note` back out of
# `sections`, and the pair could not be separated without inventing an import
# cycle. Down here they are what everything else stands on.
# --------------------------------------------------------------------------

def frontier_points(run: Run, bfb_only: bool = False, prefix: str = "W10",
                    collection: str = "bench2",
                    recall_doc: dict | None = None) -> list[dict]:
    """`(recall, qps)` per `ef`, preferring bfb's throughput to the smoke rate.

    `bfb_only` drops the smoke-rate fallback: a chart may draw whichever rate
    it has and say which, a *ratio* against the other engine may not mix the
    two (`matched_recall_refusal`).

    Recall comes from the conformance sweep and throughput from W10, which is
    §4.1's separation working as intended: bfb generates load and knows nothing
    about relevance, the conformance binary measures relevance with a
    single batching client whose rate is not a benchmark. Joining them on `ef`
    is the only way to get a frontier whose y-axis is a real throughput
    measurement.

    The join is across runs, not within one: the recall sweep builds its own
    collection. Same dataset, same `m` and `ef_construct`, different graph
    build, and that is stated wherever the result is shown rather than being
    left for someone to discover.
    """
    by_id = run.by_id()
    out = []
    points = (recall_doc or run.recall).get("points", [])
    for p in points:
        if p.get("exact") or p.get("ef") is None:
            continue
        ef = int(p["ef"])
        row = by_id.get(f"{prefix}-ef{ef}") or {}
        if row.get("collection") not in (None, collection) \
                or not row.get("recall_joinable", True):
            row = {}
        qps, source = row.get("qps"), f"bfb {prefix}"
        if qps is None and not bfb_only:
            qps, source = p.get("smoke_qps"), "conformance smoke rate"
        if qps is None:
            continue
        out.append({"ef": ef, "recall": p["recall_at_10"], "qps": qps,
                    "source": source,
                    "ci": Ci95(p.get("recall_at_10_ci95_low"),
                           p.get("recall_at_10_ci95_high"))})
    out.sort(key=lambda p: p["recall"])
    return out


class Segment(NamedTuple):
    """An interpolated rate at a target recall, and the local log-slope.

    `q, slope = seg` at the call site, two floats in a fixed order, and the
    slope is what `recall_band` multiplies the CI half-width by — so a
    transposition would widen the published band by the rate."""

    qps: float
    slope: float


def _interp_segment(points: list[dict], recall: float) -> Segment | None:
    """`(qps, d ln(qps)/d recall)` at `recall`, or None outside the measured range.

    Linear in log(qps) against recall, between the two bracketing measurements.
    Outside that range it returns None rather than extrapolating: the curve
    steepens sharply near recall 1.0, and an extrapolated point there is an
    invention that would land in a comparison table looking like a measurement.

    The slope comes back with the value because the same steepness that makes
    extrapolation an invention makes *interpolation* leveraged: in the top
    segment measured here, 0.003 of recall spans 1.8x of throughput, so a
    slope of ~190 per unit recall turns the recall CI into a factor of 1.4 on
    the interpolated rate. `recall_band` is what does that arithmetic; without
    the slope it could not.
    """
    import math
    for (a, b) in itertools.pairwise(points):
        if a["recall"] <= recall <= b["recall"]:
            if b["recall"] == a["recall"]:
                return Segment(qps=a["qps"], slope=0.0)
            t = (recall - a["recall"]) / (b["recall"] - a["recall"])
            slope = (math.log(b["qps"]) - math.log(a["qps"])) / (b["recall"] - a["recall"])
            return Segment(
                qps=math.exp(math.log(a["qps"]) * (1 - t) + math.log(b["qps"]) * t),
                slope=slope)
    return None


class Ci95(NamedTuple):
    """A 95% Wilson interval on a recall, as the sweep reports it.

    The same pair `conformance`'s `Ci95` carries, and named here for the same
    reason: it arrived as a bare `tuple` with no annotation, was read as
    `ci[0]` / `ci[1]`, and the one place it is used decides how wide a ratio's
    stated uncertainty is.
    """

    low: float | None = None
    high: float | None = None


class Band(NamedTuple):
    """The range a ratio moves through when its anchor recall moves by its CI.

    Deliberately *not* a `Ci95`, because `recall_band` is emphatic that this is
    a sensitivity rather than a confidence interval: it moves the anchor's
    recall only, and leaves the throughput measurements and the other engine's
    recall alone. Two names because they are two things, and a single `Interval`
    would invite the page to call this an uncertainty.
    """

    low: float
    high: float


def recall_band(ratio: float, slope: float, ci: Ci95 | tuple | None) -> Band | None:
    """`ratio` moved by the anchor's own recall CI, or None when there is no CI.

    The matched-recall ratio pairs a *measured* rate at a measured recall
    against the other engine's rate interpolated to that recall. The recall is
    itself an estimate — the table one section up prints its 95% Wilson
    interval, +/-0.013 at ef=32 and +/-0.0019 at ef=512 — and the interpolation
    reads it as exact. Shifting the target recall by the CI half-width moves
    the interpolated rate by `exp(|slope| * halfwidth)`, and the ratio with it.

    This is a sensitivity, not a confidence interval, and it is the narrow
    side of one: it moves the anchor's recall only, and leaves the throughput
    measurements (one pass each) and the other engine's recall CI alone. It is
    reported because the alternative is the page quoting a ratio to three
    significant figures on a quantity whose own stated uncertainty spans a
    factor of two.
    """
    import math
    lo, hi = Ci95(*ci) if ci else Ci95()
    if lo is None or hi is None or not (hi > lo):
        return None
    f = math.exp(abs(slope) * (hi - lo) / 2)
    return Band(low=ratio / f, high=ratio * f)


def matched_ratios(pa: list[dict], pb: list[dict]) -> list[dict]:
    """Every recall one engine measured that the other's sweep brackets.

    One definition, because `summary` and `matched_recall_table` each grew
    their own copy of this loop and the headline was therefore free to drift
    from the table it claims to summarise.

    `swapped` says which engine the anchor is: the ratio is always a/b, so an
    anchor on b's side inverts. The rows are correlated rather than
    independent — adjacent anchors share bracketing segments, and eight rows
    come from ten measurements — so min/max over them is a spread across the
    frontier and not a sampling range. The table says so.
    """
    out = []
    for src, other, swapped in ((pa, pb, False), (pb, pa, True)):
        for p in src:
            seg = _interp_segment(other, p["recall"])
            if seg is None:
                continue
            q, slope = seg
            ratio = (q / p["qps"]) if swapped else (p["qps"] / q)
            out.append({"recall": p["recall"], "ef": p["ef"], "swapped": swapped,
                        "measured": p["qps"], "interpolated": q, "ratio": ratio,
                        "band": recall_band(ratio, slope, p.get("ci"))})
    out.sort(key=lambda r: r["recall"])
    return out


# --------------------------------------------------------------------------
# What may not be said about a pair of runs
#
# Refusals and annotations over runs rather than rendering of them, and the
# last two names the charts needed from `report.py`. With these down here the
# chart module is a leaf: it reads this and nothing above it.
# --------------------------------------------------------------------------

def segments_of(runs: list[Run], collection: str = "bench2") -> dict[str, int]:
    """How many segments each engine actually held, read back from the engine.

    §2's gap and `docs/comparison-sift1m.md` §2 both turn on this number, and the page
    quoted the *cap* — "up to 8 segments", "nominal 128 is up to 1024 node
    visits" — in three places while `collections.json` sat in the same report
    saying 5. `--segments 1` sets Qdrant's `default_segment_number`, which its
    optimizer treats as a target and misses; the whole reason
    `collection_table` exists is that §4 recorded the request and nothing
    recorded the answer. Quoting the cap when the answer is known repeats that
    mistake one section above the table that disproves it.

    Empty for a run with no capture, and the callers then fall back to the cap
    and say that is what they are quoting.
    """
    out: dict[str, int] = {}
    for r in runs:
        for c in (r.collections or {}).get("collections") or []:
            if c.get("collection") == collection and c.get("segments_count"):
                out[r.label] = int(c["segments_count"])
    return out


def populated_of(runs: list[Run], collection: str = "bench2") -> dict[str, int]:
    """How many of those segments held any points, where it was read back.

    Recorded from Qdrant's per-segment telemetry (`fullrun.qdrant_segments`),
    so only its arm has it, and only on runs captured since. An empty
    appendable is a segment and not a graph: `held 2 segments` on sift1m was
    one graph over every point plus that, and the note called the count an
    upper bound because nothing on disk said which.
    """
    out: dict[str, int] = {}
    for r in runs:
        for c in (r.collections or {}).get("collections") or []:
            if (c.get("collection") == collection
                    and c.get("populated_segments_count") is not None):
                out[r.label] = int(c["populated_segments_count"])
    return out

def _arms(runs: list[Run]) -> tuple[Run | None, Run | None]:
    """The strawmANN run and the Qdrant run, by engine rather than by label.

    `--strawmann-label`/`--qdrant-label` take any string, and the
    label-per-corpus rule means a `--dataset` run *has* to use one: the dbpedia
    pair is `sm-dbp100k` and `qd-dbp100k`. Splitting on `label != "strawmann"`
    therefore picked the strawmANN run as "the other engine" and read its
    segment count as Qdrant's — which published "ef is the same unit here" on a
    run where Qdrant held two segments, the exact confound the sentence exists
    to warn about. `engine_of` reads `run.json`, which is the engine's own
    account of itself; the label match stays as the fallback for a run measured
    before that block existed.
    """
    sm = next((r for r in runs if engine_of(r) == "strawmann"), None)
    qd = next((r for r in runs if engine_of(r) == "qdrant"), None)
    if sm is None and qd is not None:
        sm = next((r for r in runs if r is not qd), None)
    elif qd is None and sm is not None:
        qd = next((r for r in runs if r is not sm), None)
    elif sm is None and qd is None:
        sm = next((r for r in runs if r.label.lower() == "strawmann"), None)
        qd = next((r for r in runs if r is not sm), None) if sm else None
    return sm, qd


def segment_note(runs: list[Run]) -> str:
    """One sentence on what `ef` bought each engine, from the measured counts."""
    seg = segments_of(runs)
    pop = populated_of(runs)
    sm_run, qd_run = _arms(runs)
    qd = qd_run.label if qd_run else None
    sm = sm_run.label if sm_run else None
    # More segments than graphs, measured: the rest are empty appendables,
    # which a search visits and finds nothing in. One populated segment on
    # each side is equal work, whatever the raw count says.
    if qd and sm and seg.get(sm) == 1 and pop.get(qd) == 1 and seg.get(qd, 0) > 1:
        n = seg[qd]
        return (f"ef is the same unit here: {qd} held {n} segments for this run, one "
                f"populated and {n - 1} empty (read back per segment from the engine), "
                f"and strawmANN held one graph, so equal ef is equal traversal width "
                f"and the curves may be read at equal x. That is a property of this "
                f"run, not of the engines.")
    # `seg.get(sm, 1)` defaulted a *missing* capture to one segment, which
    # asserts a measurement that was never taken. Both counts have to be on
    # disk before this branch may say the confound does not apply.
    if qd and sm and seg.get(qd) == 1 and seg.get(sm) == 1:
        # Measured, and one apiece: the caveat the rest of the page repeats
        # simply does not apply to this run. Falling through to the cap here
        # would quote a confound that was measured not to exist, and claim no
        # capture on a run that has one.
        return (f"ef is the same unit here: {qd} held one segment for this run "
                f"(read back from the engine) and so did strawmANN, so equal ef "
                f"is equal traversal width and the curves may be read at equal x. "
                f"That is a property of this run, not of the engines.")
    if qd and seg.get(qd, 0) > 1:
        n = seg[qd]
        mine = seg.get(sm) if sm else None
        mine_txt = (f"strawmANN held {mine} and searches it once"
                    if mine else "strawmANN searches one graph once")
        # Where the populated count is known it is the multiplier: an empty
        # segment adds no node visits.
        graphs = pop.get(qd)
        held = (f"{n} segments, {graphs} of them populated," if graphs is not None
                else f"{n} segments")
        # Plain text: the template renders a chart note through Jinja's
        # autoescape, so markup here would arrive as visible angle brackets.
        return (f"ef is not the same unit in both engines. {qd} held {held} "
                f"for this run — read back from the engine, since `--segments 1` "
                f"sets a target its optimizer missed — and searches ef candidates "
                f"in each, so its nominal 128 is up to {128 * (graphs or n):,} node "
                f"visits while {mine_txt}. Reading the curves at equal x overstates "
                f"strawmANN.")
    return ("ef is not the same unit in both engines. Qdrant searches ef candidates "
            "in each segment and the default segment count is capped at 8, so its "
            "nominal 128 is up to 1,024 node visits against strawmANN's 128 — the "
            "cap, not a measurement, because this run captured no segment count. "
            "Reading the curves at equal x overstates strawmANN.")

def matched_recall_refusal(runs: list[Run], row_prefix: str = "W10-ef") -> str:
    """Why no matched-recall ratio may be stated between these two runs, or "".

    The matched-recall figure is the headline, and it is built from the W10
    rows and the recall sweep rather than from a `compare.Row`, so the
    refusals `compare` applies per row did not reach it: a STALE pair, an
    unlicensed one, a contaminated or unstamped W10 row all still produced
    "N.NNx to N.NNx". Every one of those refuses here too. So does a run with
    no W10 rows at all: `frontier_points` falls back to the conformance
    binary's own single-client smoke rate for the chart, and a ratio between
    that and bfb's throughput on the other side is a ratio between two
    instruments, which is what "50x to 81x" turned out to be.
    """
    if len(runs) != 2:
        return ""
    a, b = runs
    stale = compare.stale_reasons(a.label, b.label)
    if stale:
        return "STALE: " + "; ".join(stale)
    lic = compare.licence(a.label, b.label)
    # A T3 failure alone no longer withholds the table. T3 says the two engines
    # differ in ANN recall at a fixed `ef`, which is the condition this table
    # exists to correct: it does not assume equal recall, it constructs it at
    # each recall one engine actually reached. Withholding it there meant the
    # dbpedia-openai-1m tier published no comparison at all while 2.14x, 2.03x
    # and 1.80x sat computable on disk. What still withholds is a pair whose
    # scores do not agree (T1 or T2 failed, or no conformance row at all),
    # because then no interpolation of either curve means anything.
    # `matched_recall_caveat` supplies the banner the table is shown under.
    if not lic.get("comparative") and not lic.get("scores_agree"):
        return lic.get("banner") or "licenses_comparative is false"
    for jr in compare.joined(a.label, b.label, a.by_id(), b.by_id(), stale=False):
        if jr.id.startswith(row_prefix) and jr.refusal:
            return f"{jr.id}: {jr.refusal}"
    for run in runs:
        if not run.recall.get("points"):
            return f"{run.label} has no recall sweep (recall.py was not run for it)"
        if not any(p["source"] == "bfb W10" for p in frontier_points(run)):
            return (f"{run.label} has no W10 rows, so its frontier would be the conformance "
                    f"binary's single-client smoke rate; no ratio is formed across "
                    f"different instruments")
    return ""


def matched_recall_caveat(runs: list[Run]) -> str:
    """The banner a shown-but-unlicensed matched-recall table carries, or "".

    Empty when the pair is licensed, which is the ordinary case and prints
    nothing. Non-empty only in the state `matched_recall_refusal` now lets
    through: T1 and T2 passed, T3 did not, so the table is computable and
    honest and §8 has not licensed a comparative claim from it.
    """
    if len(runs) != 2:
        return ""
    lic = compare.licence(runs[0].label, runs[1].label)
    if lic.get("comparative") or not lic.get("scores_agree"):
        return ""
    return lic.get("caveat") or ""


# --------------------------------------------------------------------------
# What each row is called, and what a ratio string means
#
# The vocabulary the tables, the sections and the charts all name rows with.
# It sat above the tables in `report.py` and was the last thing keeping the
# chart module from being a leaf.
# --------------------------------------------------------------------------

_PURPOSES = None

#: Variant rows carry their own description where inheriting the base row's
#: would be wrong. `W4-rps500` inherited "search, saturating (closed loop)" from
#: W4 and so described itself as the exact opposite of what it measures.
#:
#: The `W4-rps*` ids are the pre-§4 arms, at a constant 500/1000/2000 per
#: second. They are kept so a report over an archived run still names its rows;
#: `W4-sat*` are the arms §4 actually specifies, at a fraction of the engine's
#: own measured saturation, and the fraction is in the name because the rate is
#: not the same number on the two engines.
VARIANTS = {
    "W4-sat50": "search, fixed rate at 50% of saturation (open loop)",
    "W4-sat70": "search, fixed rate at 70% of saturation (open loop)",
    "W4-sat90": "search, fixed rate at 90% of saturation (open loop)",
    "W4-rps500": "search, fixed 500/s (open loop, pre-§4 constant rate)",
    "W4-rps1000": "search, fixed 1000/s (open loop, pre-§4 constant rate)",
    "W4-rps2000": "search, fixed 2000/s (open loop, pre-§4 constant rate)",
    "W10-ef32": "recall/latency frontier, ef=32",
    "W10-ef64": "recall/latency frontier, ef=64",
    "W10-ef128": "recall/latency frontier, ef=128",
    "W10-ef256": "recall/latency frontier, ef=256",
    "W10-ef512": "recall/latency frontier, ef=512",
}

#: Fallback descriptions, used only for a row §4's table does not name. The
#: table itself is `workloads.py`, and reading it is what stops this file from
#: calling W0 "smoke, d=4" while every other artifact calls it "d=4 floor".
DESCRIPTIONS = {
    "W0": "d=4 floor: graph traversal with the distance taken out",
    "W1": "ingest throughput (no index wait)",
    "W2": "index build, time to Green",
    "W3": "search, fp32, single query",
    "W4": "search, saturating (closed loop)",
    "W5": "search, batched 16",
    "W6": "search, SQ8 scalar quantization",
    "W7": "search, binary quantization",
    "W8": "search, product quantization",
    "W9": "exact search (brute force)",
    "W10": "recall/latency frontier",
    "W11": "mixed read/write",
    "W11-steady": "mixed read/write, below the rebuild threshold",
    "W12-sel1": "filtered search (payload), 1% selectivity",
    "W12-sel10": "filtered search (payload), 10% selectivity",
    "W13": "scroll, id-ordered pagination",
}

def purposes() -> dict[str, str]:
    """§4's own name for each row, from the one definition of the table."""
    try:
        import workloads
    except ImportError:  # pragma: no cover - only outside bench/harness
        return {}
    return {w.id: w.purpose for w in workloads.table()}

def describe(wid: str) -> str:
    global _PURPOSES
    if _PURPOSES is None:
        _PURPOSES = purposes()
    return _PURPOSES.get(wid) or VARIANTS.get(wid) or DESCRIPTIONS.get(wid.split("-")[0], "")

def ratio_value(rs: str | None) -> float | None:
    """`compare.Row.ratio` as a number, or None when it is a refusal or a word
    (`-`, `offered`, `saturated`). The one parse of that string, so no section
    forms a ratio the throughput table refused."""
    if not rs or not rs.endswith("x"):
        return None
    try:
        return float(rs[:-1])
    except ValueError:
        return None


#: The rows that put data into a collection, in table order. `W12-upload`
#: records no phase split — it is declined by one engine — so it appears only
#: when it has something to show.
BUILD_ROWS = ["W0-upload", "W1", "W2", "W6-upload", "W7-upload", "W8-upload",
              "W12-upload"]

#: The encodings §4 measures, the collection each lives in, and a hue.
#:
#: Colour is the *encoding* on this one chart, not the engine — the subject is
#: what quantization costs, and the engines are the secondary reading. Stated
#: in the chart's note, because a page whose every other chart colours by
#: engine has taught the reader otherwise by the time they reach it.
#:
#: Four hues validated against both surfaces: every check passes in light and
#: in dark, worst adjacent CVD ΔE 25.3 deutan. That is also why these traces
#: carry no theme tag — they do not need re-colouring, and a tag would have
#: `THEME_JS` repaint them all in one engine's colour.
ENCODINGS = [("bench2", "fp32 (no quantization)", "#C97A00"),
             ("bench7", "binary, 4x oversampling", "#2F6FD0"),
             ("bench8", "PQ (product)", "#2EA44F"),
             ("bench6", "SQ8 scalar", "#A970FF")]
