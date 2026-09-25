#!/usr/bin/env python3
"""Build a self-contained HTML report from one or two workload runs.

    uv run --project bench bench/harness/report.py strawmann
    uv run --project bench bench/harness/report.py strawmann qdrant
    uv run --project bench bench/harness/report.py strawmann qdrant --open

Dependencies live in `bench/pyproject.toml` and are pinned by `bench/uv.lock`.
The engine still has none; this is the analysis path, and the first version of
this script hand-rolled SVG path geometry rather than admit a chart library,
which is a worse trade than the dependency.

The output is one file that opens with no network. Plotly's JS is embedded
rather than pulled from a CDN, so a report mailed to someone renders for them
in six months when the CDN URL has moved.

What it carries beyond the numbers, and why:

  * the §7.1 host gate verbatim, per check. A number without its environment is
    not a measurement, and the environment hash decides which results may share
    a chart at all.
  * per-row ambient load and any foreign process. The gate runs once, at the
    start; contamination arriving mid-table is invisible to it.
  * both `qps` and `rps` wherever they differ, because reading one as the other
    has already produced a wrong published finding once.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import perfstat
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

import compare
import procstat
import provenance
import regression
import workloads
from regression import SIGMA, noise_band
from report_charts import (  # noqa: F401  (re-exported: `report.X` is the tests' entry point)
    bfb_commit,
    chart_branch_mpki,
    chart_build,
    chart_dram_per_query,
    chart_ef_sweep,
    chart_frontier,
    chart_io_pressure,
    chart_ipc,
    chart_latency_distribution,
    chart_latency_percentiles,
    chart_load,
    chart_memory,
    chart_memory_pressure,
    chart_quantization_recall,
    chart_query_cost,
    chart_runqueue,
    chart_server_vs_client,
    chart_throughput,
    chart_tlb_per_query,
    datasets_of,
    matched_recall_table,
    refused_ids,
)
from report_data import (  # noqa: F401
    ANSI,
    BUILD_ROWS,
    DESCRIPTIONS,
    ENCODINGS,
    FALLBACK,
    HERE,
    ROOT,
    SERIES,
    SERIES_DARK,
    VARIANTS,
    Check,
    Run,
    _interp_segment,
    colour,
    describe,
    engine_of,
    frontier_points,
    gate_pass_of,
    glance_of,
    load_run,
    matched_ratios,
    matched_recall_refusal,
    populated_of,
    purposes,
    ratio_value,
    recall_band,
    run_colour,
    run_meta_tag,
    segment_note,
    segments_of,
    series_meta,
)
from workloads import Direction, Gate, Status

#: Upload and index rows: a duration, not a query rate.
UPLOAD_ROWS = ("W0-upload", "W1", "W2", "W6-upload", "W7-upload", "W8-upload",
               "W12-upload")




#: Rows that produce a number which should not be read as a like-for-like
#: comparison, and why. Printing the ratio without this reads as a result.
CAVEATS = {
    "W0": "d=4 floor: this measures graph traversal and heaps with the distance taken out, "
          "not request plumbing (findings 28: 76% of the server's CPU was HNSW search, "
          "24% kernel; the RPC path is a small remainder)",
    "W5": "batched 16, so qps and rps differ by 16x; the ratio uses qps",
    "W9": "brute force over the whole collection, no index involved",
    # "stopped, not measured" described one run and was printed on every one.
    # Qdrant's W12 ran 44 minutes without finishing and was
    # stopped; it completed in 288 s at 17 qps, and the sentence
    # then contradicted the number beside it. What is true of every run is the
    # cause and what the figure therefore means, so that is what it says; the
    # row reports its own outcome.
    # One note for both grades: what differs between them is the selectivity,
    # which each row's own purpose states. The clause about recall going
    # unmeasured is gone as of 2026-09-08 — it is measured per condition, over
    # a ground truth restricted to the points the filter keeps, and scored
    # against `min(k, |matching set|)`.
    "W12-sel1": "filtered search over a keyword payload index both engines were "
                "asked to build; whether each did is read back from the engine "
                "and refused beside the row when it did not. Recall is measured "
                "under the filter, against ground truth restricted to the "
                "matching set",
    "W12-sel10": "filtered search over a keyword payload index both engines were "
                 "asked to build; whether each did is read back from the engine "
                 "and refused beside the row when it did not. Recall is measured "
                 "under the filter, against ground truth restricted to the "
                 "matching set",
}




# --------------------------------------------------------------------------
# Tables, via pandas so the join across result sets is a join
# --------------------------------------------------------------------------

def frame(runs: list[Run]) -> pd.DataFrame:
    """Every field a row carries, plus the engine it came from.

    Deliberately not a whitelist. It was one, and the four duration fields
    added for the ingest table (`wall_s`, `upload_s`, `index_wait_s`,
    `time_to_green_floored`) were simply absent from it — so `_duration_cell`
    always fell through to its last resort and printed `5 s wall` for a row
    whose `rows.json` said `index_wait_s: 3.01`. Nothing failed; the report
    just quietly showed the worse number.

    Carrying the row through whole means a field added to `workloads.Result`
    reaches the report by existing, which is the only version of this that
    stays true. The defaults below are the columns other sections index
    unconditionally.
    """
    recs = []
    for run in runs:
        for r in run.rows:
            recs.append({
                "foreign": "", "load_start": 0, "load_end": 0,
                "qps": None, "rps": None, "seconds": None,
                **r,
                "engine": run.label,
            })
    return pd.DataFrame.from_records(recs)










#: What a reader calls each engine. The run label (`sm-dbp1m-perf-0924`) is an
#: identifier for the result directory, and as a column header it made every
#: table read "sm-dbp1m-perf-0924 recall@100". The label stays where it
#: identifies something: the provenance table and the regenerate command.
ENGINE_NAMES = {"strawmann": "strawmANN", "qdrant": "Qdrant"}


def display_names(runs: list[Run]) -> dict[str, str]:
    """`{label: name}`: the engine's name where it tells the arms apart.

    Two arms of one engine (an A/B of two commits) keep their labels, since
    the engine name would then say the same thing twice.
    """
    engines = [engine_of(r) for r in runs]
    if len(runs) == 2 and all(engines) and len(set(engines)) == 2:
        return {r.label: ENGINE_NAMES.get(e, e) for r, e in zip(runs, engines)}
    return {r.label: r.label for r in runs}


#: A region `renamed` leaves alone: where the label *is* the information.
KEEP_OPEN, KEEP_CLOSE = "<!--keep-labels-->", "<!--/keep-labels-->"


def renamed(text: str, names: dict[str, str]) -> str:
    """`text` with each run label replaced by its display name, outside
    `KEEP_OPEN`/`KEEP_CLOSE` regions.

    Done once over the rendered page rather than at each of the hundred sites
    that print a label, because the labels also reach the page inside
    `compare.py`'s notes and `report_data`'s sentences, which are shared with
    renderers that have no display name to use. A label is matched only as a
    whole token: never inside a path (`bench/results/<label>`) or a longer id.
    """
    subs = [(re.compile(rf"(?<![\w/.-]){re.escape(lab)}(?![\w-])"), name)
            for lab, name in names.items() if lab != name]
    if not subs:
        return text
    # `<code>` and backticked prose too: a label there is part of a command or
    # a path a reader copies (`--strawmann-label sm-dbp1m`), and renaming it
    # breaks the command.
    parts = re.split(f"({re.escape(KEEP_OPEN)}.*?{re.escape(KEEP_CLOSE)}"
                     r"|<code>.*?</code>|`[^`<>]*`)", text, flags=re.S)
    out = []
    for p in parts:
        if not p.startswith((KEEP_OPEN, "<code>", "`")):
            for rx, name in subs:
                p = rx.sub(name, p)
        out.append(p)
    return "".join(out)


#: Below this qps/rps ratio the two are the same number counted twice (bfb's
#: rps is its own clock, a few requests off); above it the row is batched and
#: the difference is the batch size, which is worth showing.
RPS_SHOWN_ABOVE = 1.5


def _num_cell(r: pd.Series | None) -> str:
    if r is None or r.empty:
        return '<td class="num muted">-</td>'
    if pd.isna(r["qps"]):
        klass = "na" if r["status"] == Status.not_applicable else "bad"
        return f'<td class="num {klass}">{r["status"]}</td>'
    sub = ""
    # `4,562` above `4,542 rps` read as one broken number, `4,5624,542`, and
    # said nothing: the two differ only on a batched row.
    if pd.notna(r["rps"]) and r["rps"] > 0 and r["qps"] / r["rps"] >= RPS_SHOWN_ABOVE:
        sub = f'<span class="sub">{r["rps"]:,.0f} requests/s</span>'
    return f'<td class="num">{r["qps"]:,.0f}{sub}</td>'


def ratio_cell(wid: str, rs: str, noise: dict[str, float], reps: int | None) -> str:
    """One ratio, coloured by its verdict, with the verdict on hover.

    The verdict ("clears the ±6% measured noise floor") was printed under every
    ratio, so the column said the same sentence thirty times and the number was
    the smaller text. A ratio the floor cannot tell from 1 is grey and marked
    `≈`; the band is in the tooltip for a reader who wants it.
    """
    v = ratio_value(rs)
    if v is None:
        return f'<td class="num muted">{html.escape(rs or "-")}</td>'
    verdict = ratio_verdict(wid, v, noise, reps)
    tip = f' title="{html.escape(verdict)}"' if verdict else ""
    if verdict.startswith(("inconclusive", "too noisy")):
        return f'<td class="num muted"{tip}>≈{rs}</td>'
    return f'<td class="num {"up" if v >= 1 else "down"}"{tip}>{rs}</td>'


#: `compare.Row` notes a summary table keeps, condensed. Everything else a row
#: carries (the work/occupancy decomposition, §4's caveats, the W11 overlap
#: arithmetic) is in the full table and the row's own section.
_RECALL_UNEQUAL = re.compile(r"recall unequal: ([\d.]+) vs ([\d.]+)")
_NOT_SERVER = re.compile(r"^(.+?): .*?so (\d+)% of this row is the load generator")


def short_notes(jr: compare.Row | None) -> tuple[list[str], bool]:
    """The notes a summary row needs, as markup, and whether any row was
    refused for the rescore-pool reason (said once, under the table)."""
    if jr is None:
        return [], False
    out, rescore = [], False
    for n in jr.notes:
        t = n.strip("[]")
        if "contaminated" in t:
            out.append(f'<span class="warn">{html.escape(t)}</span>')
        elif "FAILED" in t or "not measured" in t:
            out.append(f'<span class="bad">{html.escape(t)}</span>')
        elif "declined" in t:
            out.append(f'<span class="na">{html.escape(t)}</span>')
        elif m := _RECALL_UNEQUAL.match(t):
            mark = ""
            if "rescore pools differ" in t:
                rescore, mark = True, "<sup>1</sup>"
            out.append(f'<span class="caveat">recall differs: {m[1]} vs {m[2]}{mark}</span>')
        elif t.startswith(("recall below", "recall missing", "saturated",
                           "offered rate not served")):
            out.append(f'<span class="caveat">{html.escape(t.split(". ")[0])}</span>')
        elif m := _NOT_SERVER.match(t):
            out.append(f'<span class="caveat">{html.escape(m[1])}: {m[2]}% is client '
                       f'and socket, not server</span>')
        elif "across its passes, monotonically" in t:
            out.append(f'<span class="caveat">drifted: {html.escape(t.split(" across")[0])}'
                       f' across passes</span>')
        elif "payload index" in t or "harness" in t:
            out.append(f'<span class="caveat">{html.escape(t.split(", so")[0])}</span>')
    if jr.refusal and not out:
        out.append(f'<span class="caveat">{html.escape(jr.refusal.split("; the writer")[0])}</span>')
    return out, rescore


#: One footnote for every rescoring row, in place of the sentence on each.
RESCORE_FOOTNOTE = ("<sup>1</sup> Expected, not a defect: strawmANN rescores "
                    "<code>max(asked, ef)</code> candidates and Qdrant rescores "
                    "<code>limit</code>, so at equal settings the two are not answering "
                    "the same question.")

#: `W10-ef128` is one point of the W10 sweep; the family is what the summary
#: table shows as one row.
SWEEP_ID = re.compile(r"^(?P<fam>.+)-ef(?P<ef>\d+)$")

#: Rows the summary table leaves to the full one. W0 is a diagnostic floor
#: (`SUMMARY_EXCLUDES`), and the open-loop rows are latency measurements at an
#: offered rate: their throughput column is the offer.
COMPACT_SKIP = ("W0",)


def _is_open_loop_id(wid: str) -> bool:
    return wid.startswith(("W4-sat", "W4-rps"))


def _qps_range(rows: list) -> str:
    vals = [r["qps"] for r in rows if r is not None and pd.notna(r["qps"])]
    if not vals:
        return '<td class="num muted">-</td>'
    lo, hi = min(vals), max(vals)
    return f'<td class="num">{lo:,.0f} to {hi:,.0f}</td>'


def _sweep_row(fam: str, members: list[str], a: str, b: str | None, df: pd.DataFrame,
               joined_rows: dict, noise: dict, reps: int | None) -> tuple[str, bool]:
    """A whole `ef` sweep as one row: the qps range per engine and the range of
    the ratios that survived, with each point's ratio on hover."""
    efs = [int(SWEEP_ID.match(w)["ef"]) for w in members]
    desc = re.sub(r",? ef=\d+", "", describe(members[0]))

    def rows_of(label: str) -> list:
        return [next((r for _, r in df[(df["id"] == w) & (df["engine"] == label)].iterrows()),
                     None) for w in members]

    cells = [f'<td class="wid">{fam} <span class="muted">ef {min(efs)} to {max(efs)}</span></td>',
             f'<td class="desc">{desc}</td>', _qps_range(rows_of(a))]
    rescore, notes = False, []
    if b:
        cells.append(_qps_range(rows_of(b)))
        got, tips, refused, reasons = [], [], 0, []
        for w, ef in zip(members, efs):
            jr = joined_rows.get(w)
            v = ratio_value(jr.ratio) if jr is not None else None
            if v is None:
                refused += 1
                tips.append(f"ef={ef} not compared")
                sn, rs_ = short_notes(jr)
                rescore |= rs_
                # One of each kind: three points refused for unequal recall
                # are one reason, and each point's figures are on hover.
                sn = [re.sub(r": [\d.]+ vs [\d.]+", "", x) for x in sn]
                reasons += [x for x in sn if x not in reasons]
                continue
            verdict = ratio_verdict(w, v, noise, reps)
            got.append((v, verdict))
            tips.append(f"ef={ef} {v:.2f}x" + (f" ({verdict})" if verdict else ""))
        if got:
            lo, hi = min(v for v, _ in got), max(v for v, _ in got)
            text = f"{lo:.2f}x" if lo == hi else f"{lo:.2f} to {hi:.2f}x"
            clear = all(vd.startswith("clears") for _, vd in got)
            klass = ("up" if lo >= 1 else "down" if hi < 1 else "muted") if clear else "muted"
            cells.append(f'<td class="num {klass}" title="{html.escape("; ".join(tips))}">'
                         f'{text}</td>')
        else:
            cells.append('<td class="num muted">-</td>')
        if refused:
            notes.append(f'<span class="caveat">{refused} of {len(members)} points not '
                         f'compared</span>')
            notes += reasons
    cells.append(f'<td class="notes">{" ".join(notes)}</td>')
    return "<tr>" + "".join(cells) + "</tr>", rescore


def compact_throughput_table(runs: list[Run], df: pd.DataFrame) -> str:
    """The throughput table a reader needs: one row per question.

    The full table (`throughput_table`) has a row per measurement, 36 of them,
    of which 20 are points of four `ef` sweeps and three are open-loop latency
    rows whose throughput is the rate they were offered. Here a sweep is one
    row carrying its range, the open-loop rows and the W0 floor are left to the
    full table, and the notes column keeps only what explains a missing ratio.
    Same `compare.Row` behind every cell, so it cannot state a ratio the full
    table refuses.
    """
    noise = load_noise(runs)
    reps = reps_of(runs)
    a = runs[0].label
    b = runs[1].label if len(runs) > 1 else None
    joined_rows = ({r.id: r for r in compare.joined(a, b, runs[0].by_id(), runs[1].by_id())}
                   if b else {})
    order = [w for w in dict.fromkeys(df["id"])
             if w not in UPLOAD_ROWS and w not in COMPACT_SKIP and not _is_open_loop_id(w)]
    families: dict[str, list[str]] = {}
    for w in order:
        if m := SWEEP_ID.match(w):
            families.setdefault(m["fam"], []).append(w)
    body, any_rescore, done = [], False, set()
    for wid in order:
        m = SWEEP_ID.match(wid)
        if m and len(families[m["fam"]]) > 1:
            fam = m["fam"]
            if fam in done:
                continue
            done.add(fam)
            row, rs_ = _sweep_row(fam, families[fam], a, b, df, joined_rows, noise, reps)
            body.append(row)
            any_rescore |= rs_
            continue
        sub = df[df["id"] == wid]
        ra = sub[sub["engine"] == a]
        rb = sub[sub["engine"] == b] if b else pd.DataFrame()
        x = ra.iloc[0] if len(ra) else None
        y = rb.iloc[0] if len(rb) else None
        if x is None and y is None:
            continue
        cells = [f'<td class="wid">{wid}</td>', f'<td class="desc">{describe(wid)}</td>',
                 _num_cell(x)]
        jr = joined_rows.get(wid)
        notes, rs_ = short_notes(jr)
        any_rescore |= rs_
        if b:
            cells.append(_num_cell(y))
            cells.append(ratio_cell(wid, jr.ratio if jr is not None else "-", noise, reps))
        cells.append(f'<td class="notes">{" ".join(notes)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        return ""
    head = f"<th>workload</th><th></th><th class=\"num\">{a}</th>"
    if b:
        head += (f'<th class="num">{b}</th>'
                 f'<th class="num" title="{a} throughput divided by {b} throughput">ratio</th>')
    head += "<th>notes</th>"
    banners = compare.banners(a, b) if b else []
    top = "".join(f'<div class="banner"><b>{html.escape(bnr)}</b></div>' for bnr in banners)
    foot = (f'<p class="note">{RESCORE_FOOTNOTE}</p>' if any_rescore else "")
    return (f'{top}<div class="tablewrap"><table class="throughput"><thead><tr>{head}</tr>'
            f'</thead><tbody>{"".join(body)}</tbody></table></div>{foot}')


def throughput_table(runs: list[Run], df: pd.DataFrame) -> str:
    """The joined table, with `compare.py`'s arithmetic and refusals.

    The ratio, and whether one is printed at all, come from `compare.Row`:
    this table used to divide the two qps columns itself and so printed
    ratios `compare.py` refused (STALE labels, unequal or missing recall,
    unstamped rows), and the HTML and the README disagreed about the same
    run. The banners `compare.banners` puts above every rendering go above
    this one too.
    """
    noise = load_noise(runs)
    a = runs[0].label
    b = runs[1].label if len(runs) > 1 else None
    order = list(dict.fromkeys(df["id"]))
    head = f"<th>workload</th><th></th><th>{a}</th>"
    if b:
        head += f"<th>{b}</th><th>ratio</th>"
    head += "<th>notes</th>"

    joined_rows: dict[str, compare.Row] = {}
    banners: list[str] = []
    if b:
        joined_rows = {r.id: r for r in compare.joined(a, b, runs[0].by_id(), runs[1].by_id())}
        banners = compare.banners(a, b)

    body = []
    for wid in order:
        if wid in UPLOAD_ROWS:
            continue
        sub = df[df["id"] == wid]
        ra = sub[sub["engine"] == a]
        rb = sub[sub["engine"] == b] if b else pd.DataFrame()
        x = ra.iloc[0] if len(ra) else None
        y = rb.iloc[0] if len(rb) else None
        if x is None and y is None:
            continue

        desc = describe(wid)
        cells = [f'<td class="wid">{wid}</td>', f'<td class="desc">{desc}</td>',
                 _num_cell(x)]

        notes = []
        jr = joined_rows.get(wid)
        if jr is not None:
            for n in jr.notes:
                text = n.strip("[]")
                klass = ("warn" if "contaminated" in text else "bad" if "FAILED" in text
                         else "na" if "declined" in text else "caveat")
                notes.append(f'<span class="{klass}">{html.escape(text)}</span>')
        else:
            for lab, r in ((a, x), (b, y)):
                if r is None:
                    continue
                if r["foreign"]:
                    notes.append(f'<span class="warn">{lab} contaminated: {r["foreign"]}</span>')
                if r["status"] == Status.not_applicable:
                    notes.append(f'<span class="na">{lab}: declined, not implemented</span>')
                elif r["status"] == Status.failed:
                    notes.append(f'<span class="bad">{lab}: FAILED</span>')
        if wid in CAVEATS:
            notes.append(f'<span class="caveat">{CAVEATS[wid]}</span>')

        if b:
            cells.append(_num_cell(y))
            # `compare.Row.ratio` is "N.NNx", or a word saying why not.
            # A ratio inside the measured floor is not a small win, it is no
            # measured difference, and the cell says which (`ratio_cell`).
            cells.append(ratio_cell(wid, jr.ratio if jr is not None else "-",
                                    noise, reps_of(runs)))
        cells.append(f'<td class="notes">{" ".join(notes)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    top = "".join(f'<div class="banner"><b>{html.escape(bnr)}</b></div>' for bnr in banners)
    return (f'{top}<div class="tablewrap"><table class="throughput"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# --------------------------------------------------------------------------
# Per-workload sections: one table each, metric down and engine across
# --------------------------------------------------------------------------

def _dur(v) -> str:
    """Microseconds as ms or µs. Named apart from the latency table's `_us`,
    which returns markup for a missing value: two functions of the same name at
    module scope means the later one wins, and the earlier one's callers get
    behaviour they were never written against. That is what put five empty
    latency rows in every ingest section."""
    if v is None:
        return "-"
    return f"{v / 1000:,.2f} ms" if v >= 1000 else f"{v:,.0f} \u00b5s"


def _lat(r: dict, key: str):
    return (r.get("latency") or {}).get(key)


#: What each per-workload table shows, in the order a reader needs it: the
#: headline, then what it cost in latency, then whether the two engines were
#: answering the same question (recall), then where the time went.
#:
#: One row per metric and one column per engine, which is the whole point of
#: the change: the joined table put twenty workloads down the page and every
#: metric across it, so comparing two engines on one workload meant reading
#: along a wide row and comparing two numbers separated by four columns.
#: `(label, format, raw, direction)`. The comparison column reads `raw`, never
#: the formatted cell: `422 µs` against `6.23 ms` is a real 14.8x that string
#: parsing refused, because the two cells had chosen different units. Comparing
#: what is displayed rather than what was measured is a bug waiting for a unit
#: change. `flat` means no direction — `queries sent` is an input, and a ratio
#: of inputs describes the harness.
def _cpu(r: dict):
    v = (r.get("cpu_user_s") or 0) + (r.get("cpu_system_s") or 0)
    return v or None


#: Rows whose value is decided by §5.5's placement rather than by the engine:
#: an engine mapping a file writes bytes and holds page cache for that reason.
#: Their comparison is suppressed unless both engines are known to have served
#: from the same placement.
PLACEMENT_SENSITIVE = {"peak RSS", "disk written", "disk read"}

WORKLOAD_METRICS: list[tuple[str, object, object, str]] = [
    ("client p50", lambda r: _dur(_lat(r, "client_p50_us")),
     lambda r: _lat(r, "client_p50_us"), "down"),
    ("client p99", lambda r: _dur(_lat(r, "client_p99_us")),
     lambda r: _lat(r, "client_p99_us"), "down"),
    ("client p99.9", lambda r: _dur(_lat(r, "client_p999_us")),
     lambda r: _lat(r, "client_p999_us"), "down"),
    ("server p50", lambda r: _dur(_lat(r, "server_p50_us")),
     lambda r: _lat(r, "server_p50_us"), "down"),
    ("server p99", lambda r: _dur(_lat(r, "server_p99_us")),
     lambda r: _lat(r, "server_p99_us"), "down"),
    ("wall clock", lambda r: f"{r['wall_s']:,.1f} s" if r.get("wall_s") else "-",
     lambda r: r.get("wall_s"), "down"),
    # What an ingest row is actually about, and what it showed none of: the
    # split between pushing the points and waiting for the index. `wall clock`
    # is their sum plus the harness's own overhead, so a row that is slow to
    # Green and fast to upload read exactly like its opposite.
    ("  of which upload", lambda r: f"{r['upload_s']:,.1f} s" if r.get("upload_s") else "-",
     lambda r: r.get("upload_s"), "down"),
    ("  of which index wait", lambda r: f"{r['index_wait_s']:,.1f} s" if r.get("index_wait_s") else "-",
     lambda r: r.get("index_wait_s"), "down"),
    # `time_to_green_floored` is the *flag* (bfb's 3 s polling floor was hit);
    # the duration is `index_wait_s`. Formatting the flag with `:,.1f` printed
    # `1.0 s` for every floored row and nothing for the rest.
    ("time to Green", lambda r: (f"{r['index_wait_s']:,.1f} s"
                                 + (" (at bfb's polling floor)" if r.get("time_to_green_floored") else "")
                                 if r.get("index_wait_s") else "-"),
     lambda r: r.get("index_wait_s"), "down"),
    ("queries sent", lambda r: f"{r['n_queries']:,.0f}" if r.get("n_queries") else "-",
     lambda r: None, "flat"),
    ("cpu", lambda r: f"{_cpu(r):,.1f} s" if _cpu(r) else "-", _cpu, "down"),
    ("cpu, % of wall", lambda r: _pct_of_wall(r), lambda r: None, "flat"),
    ("waiting for a core", lambda r: (f"{r['runqueue_wait_s'] * 1000:,.1f} ms"
                                      if r.get("runqueue_wait_s") is not None else "-"),
     lambda r: r.get("runqueue_wait_s"), "down"),
    ("migrations", lambda r: (f"{r['migrations']:,.0f}"
                              if r.get("migrations") is not None else "-"),
     lambda r: r.get("migrations"), "down"),
    # No comparison column. The cell is two numbers and the ratio was computed
    # from one of them: W6 showed `105,601 / 0` against `426,480 / 295,895`
    # labelled 4.04x, which is the minor-fault ratio sitting under a cell whose
    # larger figure is the major-fault count. The pair is the reading (see
    # `_faults_cell`), and there is no single ratio of a pair. Major faults are
    # also decided by §5.5's placement — an engine mapping a file faults for
    # that reason — so the comparison this row wants is the one
    # `PLACEMENT_SENSITIVE` suppresses anyway.
    ("faults min/maj", lambda r: _faults_cell(r)[0],
     lambda r: None, "flat"),
    ("peak RSS", lambda r: procstat.human_bytes(r.get("rss_peak_bytes")),
     lambda r: r.get("rss_peak_bytes"), "down"),
    ("disk written", lambda r: procstat.human_bytes(r.get("disk_write_bytes")),
     lambda r: r.get("disk_write_bytes"), "flat"),
    ("disk read", lambda r: procstat.human_bytes(r.get("disk_read_bytes")),
     lambda r: r.get("disk_read_bytes"), "flat"),
]


def _pct_of_wall(r: dict) -> str:
    cpu = (r.get("cpu_user_s") or 0) + (r.get("cpu_system_s") or 0)
    wall = r.get("wall_s") or r.get("duration_s")
    return f"{cpu / wall * 100:,.0f}%" if cpu and wall else "-"


#: Numbers hiding inside a formatted cell, so a comparison column can be
#: computed from what the reader is actually looking at rather than from a
#: second, differently-rounded path through the same fields.
def _compare(x, y, direction: str) -> str:
    """`x` against `y`, from the measured values.

    A `flat` metric gets nothing by construction, and so does a comparison
    against zero: strawmANN writes 0 bytes to disk, and "infinitely better" is
    not a number a table should print.
    """
    if direction == "flat" or x is None or y is None:
        return ""
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return ""
    if x == 0 or y == 0:
        return ""
    ratio = x / y if direction == "up" else y / x
    if 0.98 <= ratio <= 1.02:
        return '<span class="muted">~equal</span>'
    klass = "up" if ratio > 1 else "down"
    return f'<span class="{klass}">{ratio:,.2f}x</span>'


def _row_config(rows: list) -> str:
    """What this row actually ran, from the row itself.

    A section that says `search, saturating` without saying `ef=128, -p 8` is
    describing a purpose rather than a measurement, and `ef` is the knob the
    whole comparison turns on.
    """
    r = next((x for x in rows if x), None)
    if not r:
        return ""
    bits = []
    if r.get("ef") is not None:
        bits.append(f"ef={r['ef']}")
    if r.get("exact"):
        bits.append("exact")
    if r.get("load_mode"):
        bits.append(str(r["load_mode"]))
    # An open-loop row's offered rate, and what it is a fraction of. §4 sets
    # the arms at a fraction of *measured* saturation, so the absolute rate
    # differs between the two engines by design and the row has to say both
    # numbers for either to mean anything.
    if r.get("rps_target"):
        bit = f"offered={r['rps_target']:,}/s"
        if r.get("rps_fraction"):
            bit += f" ({r['rps_fraction']:.0%} of saturation"
            if r.get("saturation_qps"):
                # `--rps-reference` pins a previous run's saturation for both
                # arms; the row said "measured" for that too.
                how = "pinned reference" if r.get("rps_reference_source") == "pinned" \
                    else "measured"
                bit += f", {how} {r['saturation_qps']:,.0f} qps"
            bit += ")"
        bits.append(bit)
    if r.get("collection"):
        bits.append(f"collection={r['collection']}")
    # What the client offered. The rows do not agree about it (W10's ladder is
    # `-p 8`, W6's and W12's are bfb's default of 2) and their throughput
    # shares one column, so a qps read across rows without this is a
    # comparison of two different loads (findings 52). A defaulted value is
    # marked, because "pinned at 8" and "bfb's default" are different facts.
    if r.get("client_parallel") is not None or r.get("client_threads") is not None:
        got = [(f"-p {r['client_parallel']}" if r.get("client_parallel") is not None
                else "-p n/a under --rps")]
        for flag, key in (("-t", "client_threads"), ("-c", "client_connections")):
            if r.get(key) is not None:
                got.append(f"{flag} {r[key]}")
        deflt = (r.get("client_defaults") or "").split()
        bits.append("client " + " ".join(got)
                    + (f" ({', '.join(deflt)} bfb default)" if deflt else ""))
    if r.get("n_requested"):
        bits.append(f"n={r['n_requested']:,}")
    ov, rs = r.get("quantization_oversampling"), r.get("quantization_rescore")
    if ov is not None:
        bits.append(f"oversampling={ov:g}")
    if rs is not None:
        bits.append(f"rescore={str(rs).lower()}")
    return " · ".join(bits)


def workload_sections(runs: list[Run], df: pd.DataFrame) -> str:
    """One section per workload, with the two engines side by side.

    The joined table above answers "which rows differ"; this answers "what
    happened on this row", which is the question a reader has once they have
    picked one. Rows a workload did not measure are dropped rather than shown
    as dashes, so a search row does not carry eleven empty ingest cells.
    """
    noise = load_noise(runs)
    # One question asked once, not per row: did both engines serve from the
    # same residency? If not, the memory and disk rows are about the placement
    # rather than about the engine (§5.5).
    ps = placements_of(runs)
    same_placement = len(runs) == 2 and len(set(ps)) == 1 and ps[0] is not None
    a = runs[0].label
    b = runs[1].label if len(runs) > 1 else None
    joined: dict[str, compare.Row] = {}
    if b:
        joined = {r.id: r for r in compare.joined(a, b, runs[0].by_id(), runs[1].by_id())}
    by = [r.by_id() for r in runs]

    out: list[str] = []
    index: list[tuple[str, str, str]] = []
    for wid in list(dict.fromkeys(df["id"])):
        rows = [d.get(wid) for d in by]
        if not any(rows):
            continue
        jr = joined.get(wid)

        # The headline: qps and the verdict on the ratio, in the heading, so
        # the section is scannable without opening the table.
        verdict = ""
        if jr is not None:
            rs = jr.ratio
            if ratio_value(rs) is not None:
                v = ratio_verdict(wid, ratio_value(rs), noise, reps_of(runs))
                klass = "muted" if v.startswith(("inconclusive", "too noisy")) else \
                    ("up" if ratio_value(rs) >= 1 else "down")
                tip = f' title="{html.escape(v)}"' if v else ""
                verdict = f'<span class="wl-ratio {klass}"{tip}>{rs}</span>'
                index.append((wid, rs, klass))
            elif rs and rs != "-":
                verdict = f'<span class="wl-ratio muted">{rs}</span>'
                index.append((wid, rs, "muted"))
        if not any(w == wid for w, _, _ in index):
            index.append((wid, "\u2014", "muted"))

        two = len(runs) == 2
        # A row one engine declined is not a row it won. strawmANN answers
        # W12 with `n/a` — M7 is not built — and its "2.2 s" is the
        # time to refuse, not to ingest 200,000 points; the table printed that
        # as ingesting 6.93x faster. No status of `ok` on both sides, no ratios.
        comparable = two and all(r and r.get("status") == "ok" for r in rows)
        # The open-loop arms offer a fixed rate. Both engines served it, so the lead
        # figure is the *offer*, and "~equal" describes the harness.
        offered = any(r and r.get("load_mode") == "open-loop" for r in rows)
        head = ("<th>metric</th>" + "".join(f'<th class="num">{r.label}</th>' for r in runs)
                + ("<th class=\"num\"></th>" if two else ""))
        body = []
        # qps first and always, even where it is a status word rather than a
        # number: "n/a" on a declined construct is the answer, not a gap.
        # An ingest row has no queries per second, and printing its status word
        # under that heading made `ok` the headline of every upload section.
        # Lead with the figure the row is actually about.
        if any(r and r.get("qps") is not None for r in rows):
            texts = [f'{r["qps"]:,.0f}' if r and r.get("qps") is not None
                     else (r.get("status", "-") if r else "-") for r in rows]
            lead_raw = [r.get("qps") if r else None for r in rows]
            lead, lead_dir = "queries/second", "up"
        else:
            texts = [(f"{r['wall_s']:,.1f} s" if r and r.get("wall_s") else
                      (r.get("status", "-") if r else "-")) for r in rows]
            lead_raw = [r.get("wall_s") if r else None for r in rows]
            lead, lead_dir = "wall clock", "down"
        cells = "".join(f'<td class="num"><b>{t}</b></td>' for t in texts)
        # A refusal in `compare.Row` (STALE, contaminated, unstamped, unequal
        # recall, W11's policy) is a refusal here too. This cell used to divide
        # the two qps figures itself and printed `2.00x` under a `-` in the
        # throughput table, for the same row of the same run.
        refused = jr is not None and jr.ratio == "-"
        if refused:
            comparable = False
            lead_cmp = '<span class="muted">-</span>'
        elif jr is not None:
            v = ratio_value(jr.ratio)
            lead_cmp = (_compare(v, 1.0, "up") if v is not None
                        else f'<span class="muted">{html.escape(jr.ratio)}</span>')
        else:
            lead_cmp = "" if offered else _compare(lead_raw[0], lead_raw[1], lead_dir)
        cmp_ = (f'<td class="num">{lead_cmp}</td>' if comparable or refused
                else ("<td></td>" if two else ""))
        body.append(f'<tr><td class="desc"><b>{lead}</b></td>{cells}{cmp_}</tr>')

        if jr is not None and (jr.rec_a is not None or jr.rec_b is not None):
            rec = [jr.rec_a, jr.rec_b][:len(runs)]
            texts = [f"{x:.4f}" if x is not None else "-" for x in rec]
            cells = "".join(f'<td class="num">{t}</td>' for t in texts)
            cmp_ = (f'<td class="num">{_compare(rec[0], rec[1], "up")}</td>' if comparable
                    else ("<td></td>" if two else ""))
            body.append(f'<tr><td class="desc">recall@10</td>{cells}{cmp_}</tr>')

        for name, fn_, raw_, direction in WORKLOAD_METRICS:
            vals = [fn_(r) if r else "-" for r in rows]
            raws = [raw_(r) if r else None for r in rows]
            if all(v == "-" for v in vals):
                continue
            if name == lead:          # already the headline
                continue
            cells = "".join(f'<td class="num">{v}</td>' for v in vals)
            ok = comparable and (same_placement or name not in PLACEMENT_SENSITIVE)
            cmp_ = (f'<td class="num">{_compare(raws[0], raws[1], direction)}</td>' if ok
                    else ("<td></td>" if two else ""))
            body.append(f'<tr><td class="desc">{name}</td>{cells}{cmp_}</tr>')

        cfg = _row_config(rows)
        cfg_html = f'<div class="wl-cfg">{html.escape(cfg)}</div>' if cfg else ""

        notes, seen_notes = [], set()
        if jr is not None:
            for n in jr.notes:
                t = n.strip("[]")
                # A caveat that applies to both engines was printed once per
                # engine, prefixed with each label: W5 said "per-batch latency
                # (16 queries/request)" twice. Same sentence, said once.
                body_only = t.split(": ", 1)[-1]
                if body_only in seen_notes:
                    continue
                seen_notes.add(body_only)
                klass = ("warn" if "contaminated" in t else "bad" if "FAILED" in t
                         else "na" if "declined" in t else "caveat")
                notes.append(f'<span class="{klass}">{html.escape(t)}</span>')
        if wid in CAVEATS:
            notes.append(f'<span class="caveat">{CAVEATS[wid]}</span>')
        note_html = f'<div class="notes">{" ".join(notes)}</div>' if notes else ""

        out.append(
            f'<section class="wl" id="wl-{wid}">'
            f'<h3><span class="wl-id">{wid}</span> '
            f'<span class="wl-desc">{describe(wid)}</span> {verdict}</h3>'
            f'{cfg_html}'
            f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{note_html}</section>')
    if not out:
        return ""
    # Twenty-six sections is a scroll, not a document. The index carries each
    # row's headline verdict so it is also the summary: a reader who wants only
    # "which rows differ, and by how much" never has to enter the sections.
    links = "".join(
        f'<a href="#wl-{w}">{w}<span class="wl-ix-r {k}">{r}</span></a>'
        for w, r, k in index)
    return f'<nav class="wl-index">{links}</nav>' + "".join(out)


#: What a collection's config table shows, and the label a reader needs. The
#: requested column comes from the harness stamp; the rest is what the engine
#: said when asked, which is the whole point — §4 recorded the request and
#: nothing ever recorded the answer.
COLLECTION_FIELDS = [
    ("segments", "segments_count"),
    # Qdrant only, and only on runs captured since `fullrun.qdrant_segments`:
    # how many of those segments hold points, which is how many graphs a
    # search visits. An empty appendable is the usual difference.
    ("populated segments", "populated_segments_count"),
    ("requested segments", "default_segment_number"),
    ("points", "points_count"),
    ("indexed vectors", "indexed_vectors_count"),
    ("vector size", "vector_size"),
    ("hnsw m", "hnsw_m"),
    ("hnsw ef_construct", "hnsw_ef_construct"),
    ("quantization", "quantization"),
    ("shards", "shard_number"),
    ("vectors on disk", "vector_on_disk"),
    # Residency belongs here more than anywhere: it is the field the storage
    # and memory rows are refused over, and it earned a banner of its own,
    # while the table that lists what each engine says about its collections
    # did not mention it. A reader who went looking for the evidence behind
    # that banner could not find it.
    ("vector residency", "placement"),
]






class GateSplit(NamedTuple):
    """How many of a run's rows were measured under a passing gate.

    `passed` and `total` are adjacent ints and the page prints them as a
    fraction, so a transposition reads as "26 of 24" — or, worse, as a
    plausible fraction the other way round."""

    passed: int
    total: int
    failing: list[str]


def gate_split(run) -> GateSplit:
    """How many of a run's rows were measured under a passing §7.1 gate.

    `env.txt` and the label-level verdict describe *one* invocation of
    `workloads.py`, and an arm is more than one: `fullrun.py` runs the table,
    then the recall sweeps, then W11 on its own, because W11 mutates the
    collection the sweeps read. Each invocation re-runs the gate and rewrites
    `env.txt`, so the file the report shows is the last one's — and a label
    whose 25 table rows passed on a quiet box reads as "NOT quiescent" because
    the machine had not settled again by the time W11 started.

    The rows know better: each carries the gate it was measured under
    (`Row.gate`). Rows measured before the field existed count as neither.
    """
    stamped = [r for r in run.rows if r.get("gate")]
    bad = [r["id"] for r in stamped if r["gate"] != Gate.passed]
    return GateSplit(passed=len(stamped) - len(bad), total=len(stamped), failing=bad)


#: What a run shows when the engine served from a residency it does not name.
#: Qdrant at its dense default reports no `memory` field; saying so is the
#: honest rendering, and is not the same as having captured nothing.
UNREPORTED_PLACEMENT = "unreported (its default)"


def placements_of(runs: list[Run]) -> list[str | None]:
    """The residency each engine actually served from, per run.

    §5.5's placement decides whether vectors live in anonymous memory or a file
    mapping, and therefore whether an I/O or RSS figure is about the engine at all.
    strawmANN defaults to `pinned`; Qdrant reports no placement unless asked and
    cannot serve `pinned` at all ("not supported for dense vector storage").
    So an unconfigured run compares heap against an unnamed
    default, and the disk column mostly reports which engine was asked to use a
    disk.

    The harness used to ask Qdrant for `cached` so it would say *something*, on the
    belief that this was its default. It is not — a collection created without the
    field reads back without it — so that was moving Qdrant off its default to fill
    in a sentence, and it no longer happens.

    `None` where the run has no capture. Unknown and equal are different answers.
    """
    out = []
    for r in runs:
        cs = (r.collections or {}).get("collections") or []
        ps = {c.get("placement") for c in cs if c.get("placement")}
        if len(ps) == 1:
            out.append(ps.pop())
        elif ps:
            out.append(Direction.mixed)
        else:
            # Captured, and the engine named no placement: that is a real
            # answer about an engine at its default, and distinct from having
            # no capture at all. Dropping it left the banner naming one engine
            # and silently omitting the other.
            out.append(UNREPORTED_PLACEMENT if cs else None)
    return out


def placement_refusal(runs: list[Run]) -> str:
    """Why the storage and memory rows may not be compared, when they may not."""
    ps = placements_of(runs)
    if len(runs) < 2:
        return ""
    if any(p is None for p in ps):
        return ("Placement was not captured for {}, so the storage and memory rows "
                "below are shown per engine and not compared: §5.5's residency decides "
                "whether vectors sit in anonymous memory or in a file mapping, and an "
                "engine asked to map a file writes bytes for that reason alone."
                ).format(" and ".join(r.label for r, p in zip(runs, ps) if p is None))
    if len(set(ps)) > 1:
        pairs = ", ".join(f"{r.label} {p}" for r, p in zip(runs, ps))
        return (f"These engines ran in different placements ({pairs}), so the storage "
                f"and memory rows are not comparable: the disk figures largely report "
                f"which engine was asked to use a disk. <code>cached</code> is the only "
                f"placement both can serve — Qdrant does not implement "
                f"<code>pinned</code> for dense vectors — and equalising there is a "
                f"different experiment rather than a flag: strawmANN refuses "
                f"<code>cached</code> without a file to map, so it would have to run "
                f"with <code>--data-dir</code> and would then be measured mapping a "
                f"file instead of holding anonymous RAM, which is not its default. "
                f"<b>This is not confined to the rows below.</b> Residency decides "
                f"whether a search reads resident memory or faults against the page "
                f"cache, so it can move throughput and latency too — the throughput "
                f"comparison on this page is between two engines at their own "
                f"defaults, which is the honest <code>as-deployed</code> question, and "
                f"not a controlled comparison at one residency. No such comparison is "
                f"available: there is no residency that is both engines' default, and "
                f"Qdrant cannot serve the one strawmANN defaults to.")
    return ""


def collection_table(runs: list[Run]) -> str:
    """What each engine says its collections *are*.

    §4's harness stamp records the settings the run sent and §8.9 pins the
    image; neither reads anything back, so `--segments 1` sat in the record as
    a request. For Qdrant it sets `default_segment_number`, a target the
    optimizer may miss, and the segment count is the largest confound in the
    `ef` comparison (`docs/comparison-sift1m.md` §2: a segment searched at the
    requested `ef` multiplies the traversal by the segment count). A run with no
    capture says so rather than showing the request as though it were checked.

    The survivors are read back twice: once before the mutating rows and once
    after. The table shows the first, which is the state every throughput,
    latency and recall row on the page was measured in; what W11's appends left
    behind is `after_mutating_rows` on the same record, and is W11's own subject
    rather than the search rows'. A run measured before the pre-mutation capture
    existed has only the late one, and `_capture_lag` says so.

    One table, a collection per row and a field per column. It was one table
    per collection, seven on sift1m, each eleven rows of mostly the same two
    numbers, so the one field that differs (Qdrant's segment count) had to be
    found seven times over. A cell holds one value where the engines agree and
    `a / b` where they do not, and a disagreement is marked (`differs`).
    """
    caps = [(r, (r.collections or {}).get("collections") or []) for r in runs]
    if not any(c for _, c in caps):
        return ""
    names = list(dict.fromkeys(c.get("collection") for _, cs in caps for c in cs))

    def fmt(v) -> str:
        return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)

    grid = {name: [next((x for x in cs if x.get("collection") == name), None) or {}
                   for _, cs in caps] for name in names}
    early = _read_before_drop(runs)
    fields = [(label, key) for label, key in COLLECTION_FIELDS
              if any(c.get(key) is not None for cs in grid.values() for c in cs)]
    if not fields:
        return ""
    body = []
    for name in names:
        cells = []
        for _, key in fields:
            vals = [c.get(key) for c in grid[name]]
            present = [v for v in vals if v is not None]
            if not present:
                cells.append('<td class="num muted">-</td>')
            elif len(vals) == len(present) and len(set(map(str, present))) == 1:
                cells.append(f'<td class="num">{fmt(present[0])}</td>')
            else:
                text = " / ".join("-" if v is None else fmt(v) for v in vals)
                differs = len(set(map(str, present))) > 1 and name not in early
                cells.append(f'<td class="num{" differs" if differs else ""}">{text}</td>')
        body.append(f'<tr><td class="wid">{html.escape(str(name))}</td>{"".join(cells)}</tr>')
    head = "<th>collection</th>" + "".join(f'<th class="num">{label}</th>'
                                            for label, _ in fields)
    key = ""
    if len(runs) > 1:
        key = (f'<p class="note">Where the engines disagree a cell reads '
               f'{" / ".join(html.escape(r.label) for r in runs)}, and is marked.</p>')
    notes = "".join(_capture_lag(runs, caps, name) + _after_mutation_note(caps, name)
                    for name in names)
    if seen := [n for n in names if n in early]:
        notes += (f'<p class="note"><b>{", ".join(html.escape(str(n)) for n in seen)}</b> '
                  f'was read back straight after its row and dropped, with no settle '
                  f'between, so an engine still building it reports a snapshot taken '
                  f'mid-ingest. Its cells are not marked as a disagreement.</p>')
    return (f'{key}<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{notes}')


def _read_before_drop(runs: list[Run]) -> set[str]:
    """Collections read back right after their row and then dropped unsettled.

    From each run's own stamp (`harness.engine_settle.dropped_not_settled`),
    not from today's table: a run measured before 9c7b68d settled on bench1
    and its read-back is an end state. 0925's Qdrant bench1 read 988,196
    points and 5 segments seconds before the drop reported 990,000.
    """
    by_id = {w.id: w for w in workloads.table()}
    out: set[str] = set()
    for r in runs:
        settle = ((r.meta or {}).get("harness") or {}).get("engine_settle") or {}
        for wid in settle.get("dropped_not_settled") or []:
            if wid in by_id and (c := workloads.collection_of(by_id[wid])):
                out.add(c)
    return out


def _after_mutation_note(caps: list, name: str) -> str:
    """What the mutating rows left this collection in, beside what it was.

    Not a banner: the table above it is now the state the search rows were
    measured in, so this is a second fact about the same collection rather than
    a warning about the first. It exists because the post-append state is W11's
    subject — findings 25, the rebuild still catching up — and dropping the late
    capture to fix the early one would have thrown it away.
    """
    cells = []
    for run, cs in caps:
        c = next((x for x in cs if x.get("collection") == name), None)
        after = (c or {}).get("after_mutating_rows") or {}
        pts, idx = after.get("points_count"), after.get("indexed_vectors_count")
        if not pts:
            continue
        was = (c or {}).get("points_count")
        grew = f" (+{pts - was:,})" if was and pts > was else ""
        tail = f", {idx:,} of them indexed" if idx is not None else ""
        cells.append(f"{run.label} {pts:,}{grew}{tail}")
    if not cells:
        return ""
    return (f'<p class="note"><b>After the mutating rows</b> {html.escape(str(name))} held '
            f'{"; ".join(cells)}. The table above is the state the throughput, '
            f'latency and recall rows searched; this is what W11 left, and is '
            f'that row\'s subject rather than theirs.</p>')


def _capture_lag(runs: list[Run], caps: list, name: str) -> str:
    """Whether this capture postdates the rows that searched the collection.

    Data-driven rather than keyed on W11 by name: the run stamp records how
    many points the uploads put in (`upload_n`), and a capture holding more
    than that was taken after something appended. `w11_n` names the culprit
    when the surplus matches it, and the sentence degrades to "more than the
    uploads placed" when it does not.
    """
    def stamped(run: Run, key: str):
        """`run.json` spreads some stamp fields at the top level and keeps the
        rest under `harness`. `upload_n` is in both; `w11_n` only in the
        second, so reading the top level alone found nothing and the sentence
        silently dropped the clause naming what appended."""
        meta = run.meta or {}
        if meta.get(key) is not None:
            return meta[key]
        return (meta.get("harness") or {}).get(key)

    over = []
    for run, cs in caps:
        c = next((x for x in cs if x.get("collection") == name), None)
        got = (c or {}).get("points_count")
        want = stamped(run, "upload_n")
        if got and want and got > want:
            over.append((run.label, got, want, got - want))
    if not over:
        return ""
    w11 = next((v for v in (stamped(r, "w11_n") for r in runs) if v), None)
    who = (f" — W11's {w11:,} appended points"
           if w11 and all(d == w11 for _, _, _, d in over) else "")
    per = "; ".join(f"{lab} {got:,} against {want:,} uploaded" for lab, got, want, _ in over)
    return (f'<div class="banner soft"><b>Captured after the mutating rows.</b> '
            f'{html.escape(str(name))}: {per}{who}. The read-back happens at the end of the run, so this '
            f'describes a state no search row was measured in: the throughput, '
            f'latency and recall figures for this collection were taken at the '
            f'uploaded size.</div>')


def _fmt(v, unknown: str = "unknown") -> str:
    return unknown if v in (None, "") else str(v)


def _tilde(v, unknown: str = "unknown") -> str:
    """`_fmt` with the operator's home collapsed to `~`.

    An archived report is a tracked file, and `check.py --only
    no-scratchpad-paths` refuses `/home/<user>/Workspace` in one. The path is
    worth keeping (which checkout, which `target/`, which flags); the username
    is not.
    """
    if v in (None, ""):
        return unknown
    try:
        home = str(Path.home())
    except (OSError, RuntimeError):
        return str(v)
    return str(v).replace(home + "/", "~/") if home and home != "/" else str(v)


def noise_dataset_mismatch(runs: list) -> str | None:
    """Whether the floor on disk describes these rows' dataset.

    Run-to-run spread is a property of a corpus at a dimension, not of the
    harness alone. The only floor that existed was measured on sift1m (d=128)
    on, and it was applied to whatever ran next: dbpedia-openai-100K
    rows reading 0.94x, 1.00x and 1.01x — exactly the band where the verdict
    decides whether a number is a result at all — were judged against a spread
    measured on another corpus at a twelfth the dimension.

    An *unstamped* floor stays usable and is labelled by `noise_provenance`,
    because refusing it would silently drop banding for every row measured
    before floors carried a dataset. A floor stamped with a *different* dataset
    is refused: that is not a weaker answer, it is an answer to another question.
    """
    floor_ds = noise_meta(runs).get("dataset")
    if not floor_ds:
        return None
    row_ds = next(((r.meta.get("dataset") or {}).get("name") for r in runs
                   if (r.meta.get("dataset") or {}).get("name")), None)
    if row_ds and row_ds != floor_ds:
        return (f"The noise floor was measured on {floor_ds} and these rows are "
                f"{row_ds}; run-to-run spread does not carry between corpora, so "
                f"no row here is banded.")
    return None


def load_noise(runs: list | None = None) -> dict[str, float]:
    """The measured run-to-run spread, per row, from `bench/results/noise.json`.

    Empty when the floor names a different dataset than `runs`, so every row
    falls to `ratio_verdict`'s "no noise floor measured for this row" instead of
    borrowing another corpus's spread.

    `regression.py --measure-noise` writes it from repeated identical passes
    over one unchanged server, contaminated repetitions discarded rather than
    averaged in; how many of each is in the file (`noise_meta`), not here.
    """
    if runs is not None and (noise_dataset_mismatch(runs) or noise_engine_mismatch(runs)
                             or noise_env_mismatch(runs)):
        return {}
    return noise_meta(runs).get("rsd", {}) or {}


def noise_env_mismatch(runs: list) -> str | None:
    """Whether the floor was measured in the environment it would band.

    `compare.env_matches`'s rule, which this page did not apply: an SMT-off
    floor against an SMT-on run printed a bare ratio in the table and "clears
    the measured noise floor" under it, from the floor `compare` had refused.
    """
    if len(runs) < 2:
        return None
    meta = noise_meta(runs)
    if not meta.get("rsd"):
        return None
    if compare.env_matches(meta, runs[0].label, runs[1].label):
        return None
    return ("The noise floor was measured under another environment hash than "
            "these rows (SMT, governor, isolation: findings 46), so no row here "
            "is banded.")


def noise_engine_mismatch(runs: list) -> str | None:
    """Whether the floor on disk describes both engines whose ratio it bands.

    The dataset rule above keeps an *unstamped* floor usable, because it might
    have been measured on this corpus. This one does not, because it cannot
    have been measured on both engines: `measure_noise` reads one label's
    repetition directories, so a floor is one engine's spread and a file that
    omits `arms` is one whose engine was simply never written down.

    Findings 38: the floor here is six passes of strawmANN and gave W3 an RSD
    of 0.46%, while Qdrant's own spread on that row is 13.16% over twelve runs.
    Every ratio banded with it assumed Qdrant was no noisier than strawmANN.
    """
    if len(runs) < 2:
        return None
    meta = noise_meta(runs)
    if not meta.get("rsd"):
        return None
    arms = meta.get("arms")
    engines = {(r.meta.get("engine_comm") or r.label) for r in runs}
    if arms and engines.issubset(set(arms)):
        return None
    whose = f"was measured on {', '.join(arms)}" if arms else "does not say which engine it was measured on"
    return (f"The noise floor {whose} and these rows are "
            f"{', '.join(sorted(engines))}; run-to-run spread does not carry between "
            f"engines, so no row here is banded.")


def noise_meta(runs: list | None = None) -> dict:
    """The floor that describes these runs, whole, or `{}`.

    `regression.floor_for` prefers the folded per-label `noise.json` a
    `--reps N` run writes for itself over the global one, which on this host is
    a sift1m fold measured on strawmANN alone in another session. Called
    without `runs` it is the global file, as before: the callers that have the
    runs pass them, and the ones that do not are asking about the file.
    """
    return regression.floor_for([r.label for r in runs] if runs else [])


def noise_provenance(runs: list[Run]) -> dict:
    """How the floor was measured, said from the file rather than from memory.

    The page used to say "six identical passes, one discarded as contaminated"
    in prose, while `noise.json` on the same host said `discarded: []`. And a
    floor measured under another harness stamp than these rows' is noted, not
    refused: it is still a spread of this host, and not one of these rows.
    """
    m = noise_meta(runs)
    if not m:
        return {}
    reps = m.get("reps") or {}
    n = m.get("n_dirs") or (max(reps.values()) + len(m.get("discarded") or [])
                            if reps else 0)
    out = {"passes": n, "discarded": len(m.get("discarded") or []),
           "kept": max(reps.values()) if reps else 0, "source": m.get("source", ""),
           # Which arms the spread was measured on. Empty for the borrowed
           # global floor, which is strawmANN's alone; a `--reps N` run folds
           # one per engine and the sentence about assuming Qdrant is no
           # noisier stops being true of it.
           "arms": m.get("arms") or [], "stamp_note": ""}
    # Which of *these* rows the floor speaks for. Silence on a row with no
    # floor already reads as "judged and fine" in `ratio_verdict`; saying it
    # once, in aggregate, is what stops a reader assuming the whole table was
    # judged. The ingest rows are the ones this matters most for: the floor is
    # built from search passes over an already-built graph, so no build time
    # on the page has a spread behind it, and the section that reports build
    # time carries no verdicts at all.
    rsd = m.get("rsd") or {}
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    out["covered"] = sum(1 for i in ids if i in rsd)
    out["rows"] = len(ids)
    out["uncovered_ingest"] = [i for i in ids if i not in rsd and i in UPLOAD_ROWS]
    ds_note = noise_dataset_mismatch(runs) or noise_engine_mismatch(runs)
    if ds_note:
        out["stamp_note"] = ds_note
        return out
    fh = m.get("harness_hash")
    stamps = {r.label: (r.meta.get("harness") or None) for r in runs}
    if fh is None:
        out["stamp_note"] = ("The floor carries no harness stamp (measured before floors "
                             "were stamped), so it may not describe these rows' "
                             "configuration.")
    elif fh == "mixed":
        out["stamp_note"] = ("The floor was measured over repetitions with differing "
                             "harness stamps; its spread is not one configuration's.")
    else:
        from workloads import stamp_hash
        off = [l for l, st in stamps.items() if st and stamp_hash(st) != fh]
        if off:
            out["stamp_note"] = (f"The floor was measured under a different harness "
                                 f"stamp than {' and '.join(off)}, so its spread may "
                                 f"not describe these rows.")
    return out


#: Half-width past which a noise band stops being a verdict and starts being
#: an admission that the row did not discriminate. 25% is chosen, not measured:
#: it is comfortably wider than any real effect this table reports (the largest
#: is W5 at 4.8x, the smallest called difference is 2%) and comfortably narrower
#: than the ±27% and ±64% bands a single unlucky pass produced.
UNDISCRIMINATING_BAND = 0.25


def reps_of(runs) -> int | None:
    """How many passes each arm folded, when they agree. `None` otherwise."""
    ns = {(r.meta or {}).get("reps") for r in runs}
    return ns.pop() if len(ns) == 1 and isinstance(next(iter(ns), None), int) else None


def ratio_verdict(wid: str, ratio: float | None, noise: dict[str, float],
                  reps: int | None = None) -> str:
    """Whether a ratio clears the noise floor, or is indistinguishable from 1.

    Two engines, so the spreads add: a 3σ threshold on the ratio is
    `3 · √2 · rsd` either side of 1.0 (`regression.noise_band(rsd, arms=2)`,
    the one definition of the threshold). The floor was measured on strawmann
    alone, on this host, so using it for both sides assumes Qdrant is no
    noisier, which is an assumption and is labelled as one.

    A ratio inside that band is not a small difference, it is *no measured
    difference*, and the entire point of having a floor is that the table says
    so instead of leaving a reader to treat 1.05x as a result.
    """
    if ratio is None:
        return ""
    if wid not in noise:
        # Silence here reads as "judged and fine": W4 carried "clears the ±19%
        # noise floor" while W10-ef32 carried a bare 2.11x, and nothing told a
        # reader that the second had never been measured for spread at all.
        # `noise.json` does not cover every row, and the count moves as
        # rows are added; `noise_provenance` reports the coverage of
        # *these* rows rather than a number frozen in a comment.
        return "no noise floor measured for this row"
    band = noise_band(noise[wid], arms=2)
    # `noise_band` returns `max(3*sqrt(2)*rsd, MIN_EFFECT)`, so on a quiet row
    # the number printed is the policy minimum and not this host's measured
    # spread. Calling both of them "the noise floor" told a reader that 2% had
    # been measured here; it is the smallest effect this project will report,
    # and the measured spread is usually well under it.
    measured = SIGMA * (2 ** 0.5) * noise[wid]
    what = ("minimum reportable effect" if band > measured + 1e-12
            else "measured noise floor")
    if abs(ratio - 1.0) > band:
        return f"clears the ±{band * 100:.0f}% {what}"
    # A band this wide is not a verdict, it is an admission.
    #
    # "within the ±64% band: no measured difference, not a small one" is the
    # right sentence for ±5% and a false comfort at ±64%: nothing this row
    # could plausibly have measured would have cleared it, so the row did not
    # discriminate between "identical" and "one engine twice the other". The
    # sift1m run printed exactly that, from a three-sample rsd where
    # Qdrant's W3 read 1,368, 2,063 and 1,580 — one unlucky pass, and an rsd
    # estimated from three points is itself a very noisy number.
    #
    # Past this width the honest report is that the row needs more passes.
    if band >= UNDISCRIMINATING_BAND:
        return (f"too noisy to judge: ±{band * 100:.0f}% over {reps or '?'} passes, "
                f"wider than any difference this row could show — needs more passes")
    return f"inconclusive: within the ±{band * 100:.0f}% {what}"


#: Percentiles the latency table shows, and the field they are stored under.
LAT_COLS = [("p50", "p50_us"), ("p95", "p95_us"), ("p99", "p99_us"),
            ("p99.9", "p999_us"), ("max", "max_us")]


def _us(v: float | None) -> str:
    if v is None:
        return '<span class="muted">-</span>'
    return f"{v / 1000:.2f} ms" if v >= 1000 else f"{v:.0f} µs"


#: The gap must fall to at most this fraction of itself across the ladder.
#: findings 32 measured a fall to 24%; a generator that is merely noisy drifts
#: by a percent or two, which is what fired this banner falsely.
OMISSION_FALL = 0.5
#: ...and must be at least this share of the client-side p50 to be capable of
#: distorting it. 0.1 ms of lag under a 0.9 ms latency is not the story.
OMISSION_SHARE = 0.25


def _ms(v: float) -> str:
    """A millisecond figure that never rounds its own evidence to zero.

    The banner below quotes two of these as proof that one is much smaller
    than the other, so a fixed `.0f` turned "0.106 against 0.104" into "0
    against 0" — a sentence arguing with itself.
    """
    if v >= 10:
        return f"{v:,.0f} ms"
    if v >= 1:
        return f"{v:.1f} ms"
    return f"{v:.3f} ms"


def open_loop_is_instrument_limited(runs: list[Run]) -> str:
    """Evidence, from these rows, that the open-loop latency is not the engine's.

    findings 32: bfb at the pinned commit charges each request the wait between
    the slot its rate limiter planned and the moment it was sent, so an open-loop
    row's client latency is a property of the generator. The signature is an
    unaccounted gap that *falls* as the offered rate rises — 1,404 ms at 2,000/s
    down to 340 ms at 20,000/s — which no queueing model produces.

    Reported from the rows rather than asserted, so it stops printing once the
    generator is fixed. `""` with fewer than two arms, or when the gap grows with
    load like a real queue.

    The gap must collapse, not drift: a fall from 0.106 ms to 0.104 ms across the
    ladder is a flat tenth of a millisecond against a 0.9 ms server latency and
    distorts nothing, but printed as "0 ms against 0 ms" it told the reader good
    rows were worthless. Hence both an absolute floor and a ratio.
    """
    pts = []
    for run in runs:
        for r in run.rows:
            lat = r.get("latency") or {}
            c, srv = lat.get("client_p50_us"), lat.get("server_p50_us")
            if r.get("rps_target") and c is not None and srv is not None:
                pts.append((run.label, r["id"], r["rps_target"], (c - srv) / 1000,
                            c / 1000))
    if len(pts) < 2:
        return ""
    worst = []
    # Sorted, not a bare set: set iteration over strings depends on the hash
    # seed, so two renders of one result set could order the labels in this
    # sentence differently. A report that is not reproducible byte for byte
    # cannot be diffed against the last one.
    for label in sorted({p[0] for p in pts}):
        arms = sorted((p for p in pts if p[0] == label), key=lambda p: p[2])
        if len(arms) < 2:
            continue
        lo, hi = arms[0], arms[-1]
        collapsed = lo[3] > 0 and hi[3] <= lo[3] * OMISSION_FALL
        material = lo[4] > 0 and lo[3] >= lo[4] * OMISSION_SHARE
        if collapsed and material:
            worst.append((label, lo, hi))
    if not worst:
        return ""
    detail = "; ".join(
        f"{label} {_ms(lo[3])} unaccounted at {lo[2]:,}/s against "
        f"{_ms(hi[3])} at {hi[2]:,}/s"
        for label, lo, hi in worst)
    return (f"<b>These are not latency measurements.</b> The time the client saw "
            f"beyond what the server reported gets <em>smaller</em> as the offered "
            f"rate gets larger ({detail}), and no queue behaves that way. It is the "
            f"lag between the slot bfb's rate limiter planned for a request and the "
            f"moment it sent one, charged to the request (findings 32). §7.4 reserves "
            f"<code>--rps</code> for latency claims, so until the generator timestamps "
            f"at send there is no usable latency comparison here — the closed-loop "
            f"rows understate the tail by construction and these do not measure the "
            f"engine at all.")


def open_loop_mismatch(runs: list[Run]) -> str:
    """Why two open-loop rows may not be read side by side, or "".

    §4 sets the fixed-rate arms at a fraction of *measured* saturation. Taken
    per engine that is two different offered loads, and `docs/workloads.md` is
    explicit that "comparing p99s taken at each engine's own saturation point
    compares two different offered loads" — which is the error the old
    constant rates existed to avoid, and which reintroducing the fractions
    naively would bring back. `workloads.py --rps-reference <qps>` pins one
    saturation for both arms; a run that used it says `pinned` and the rows are
    comparable. A run that did not says `own`, and this sentence goes above the
    table.
    """
    if len(runs) != 2:
        return ""
    srcs, rates = set(), set()
    for run in runs:
        for r in run.rows:
            if r.get("rps_target"):
                srcs.add(r.get("rps_reference_source") or "own")
                rates.add((r["id"], r["rps_target"]))
    if not srcs or srcs == {"pinned"}:
        return ""
    by_id: dict[str, set] = {}
    for wid, rate in rates:
        by_id.setdefault(wid, set()).add(rate)
    differing = sorted(w for w, v in by_id.items() if len(v) > 1)
    if not differing:
        return ""
    return ("The open-loop rows were offered different rates on the two engines "
            f"({', '.join(differing)}): each is a fraction of its own engine's measured "
            "saturation, so the two columns are the same fraction of capacity and not "
            "the same load. Read the fraction across, never the percentiles. Pin one "
            "reference with <code>workloads.py run … --rps-reference &lt;qps&gt;</code> "
            "— the slower engine's saturation — to make them comparable.")


def latency_table(runs: list[Run], which: str = "client") -> str:
    """Per-row latency percentiles, which the report previously showed for W4 only.

    Client-side round trip by default; the server's own timing is one column
    away in the same JSON and the difference between them is the transport and
    the queue.

    The `load mode` column is here rather than implied by the row id because
    §7.4's rule depends on it: a closed-loop p99 is not a latency result, since
    a stalled server stops receiving requests and the tail it never served does
    not appear in the sample.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    body = []
    for wid in ids:
        if wid in UPLOAD_ROWS:
            continue
        cells, any_value = [], False
        mode = ""
        for run in runs:
            row = run.by_id().get(wid, {})
            lat = row.get("latency") or {}
            mode = mode or row.get("load_mode", "")
            for _, key in LAT_COLS:
                v = lat.get(f"{which}_{key}")
                any_value = any_value or v is not None
                cells.append(f'<td class="num">{_us(v)}</td>')
        if not any_value:
            continue
        # Neutral, not `bad`. §7.4's point is that a closed-loop tail is
        # understated and an open-loop one may be quoted, so `open loop` earns
        # the positive colour. But `closed loop` is a standing property of
        # seventeen of nineteen rows, and painting the majority of a table in
        # the error colour spends it on a label rather than on a problem: the
        # sift1m report carried twenty red pills, three of which meant
        # something. The lede above the table says what a closed loop costs.
        # `workloads.LoadMode`, not the strings it spells. The enum has existed
        # since the field did and this was the one consumer still comparing
        # literals, so a rename there would have left these silently matching
        # nothing and every pill blank.
        pill = ('<span class="pill">closed loop</span>'
                if mode == workloads.LoadMode.closed_loop
                else '<span class="pill ok">open loop</span>'
                if mode == workloads.LoadMode.open_loop
                else "")
        # A client p50 far above the server's means the row is measuring the
        # load generator. Only flagged on the *open-loop* rows: a closed loop
        # is supposed to queue, that is what `-p 64` asks for, and W4's 8x gap
        # is the queue working as designed. An open-loop row at a few percent
        # of saturation has no such excuse.
        gap = ""
        if mode == workloads.LoadMode.open_loop:
            for run in runs:
                lat = (run.by_id().get(wid, {}).get("latency") or {})
                c, srv = lat.get("client_p50_us"), lat.get("server_p50_us")
                # The absolute gap, not the ratio: see `compare.py`. A ratio
                # whose denominator is the engine's own speed rewards a slow
                # engine with a calmer-looking flag.
                if c and srv and srv > 0 and c / srv >= 5:
                    gap = (f'<span class="pill bad">{(c - srv) / 1000:,.1f} ms '
                           f'not server time</span>')
                    break
        body.append(f'<tr><td class="wid">{wid}</td>'
                    f'<td class="desc">{describe(wid)} {pill} {gap}</td>'
                    f'{"".join(cells)}</tr>')
    if not body:
        return ""

    head = "<th>workload</th><th></th>"
    for run in runs:
        head += "".join(f'<th class="num">{run.label} {name}</th>' for name, _ in LAT_COLS)
    mismatch = open_loop_mismatch(runs)
    warn = (f'<div class="banner soft"><b>Not compared.</b> {mismatch}</div>'
            if mismatch else "")
    instrument = open_loop_is_instrument_limited(runs)
    if instrument:
        warn = f'<div class="banner">{instrument}</div>' + warn
    return (warn + f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


#: The summary latency table's percentiles. p95 sits between two columns a
#: reader already has, and `max` is one request.
COMPACT_LAT_COLS = [("p50", "p50_us"), ("p99", "p99_us"), ("p99.9", "p999_us")]

#: Closed-loop rows the summary latency table keeps beside the open-loop ones:
#: one client at a time (W3) has no queue to understate, and W5 is the batched
#: request. The rest of the closed-loop rows are in the full table.
COMPACT_LAT_CLOSED = ("W3", "W5")


def compact_latency_table(runs: list[Run]) -> str:
    """Latency where it can be quoted: the open-loop rows, plus W3 and W5.

    A closed-loop p99 understates the tail (a stalled server stops receiving
    requests), so of the full table's thirty-odd rows only the fixed-rate ones
    carry a latency claim. Three percentiles, each engine's once over its
    columns, and the same banners the full table raises.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    mode = {r["id"]: r.get("load_mode") for run in runs for r in run.rows}
    keep = ([w for w in ids if w in COMPACT_LAT_CLOSED[:1]]
            + [w for w in ids if mode.get(w) == workloads.LoadMode.open_loop]
            + [w for w in ids if w in COMPACT_LAT_CLOSED[1:]])
    body = []
    for wid in keep:
        cells, any_value = [], False
        for run in runs:
            lat = run.by_id().get(wid, {}).get("latency") or {}
            for i, (_, key) in enumerate(COMPACT_LAT_COLS):
                v = lat.get(f"client_{key}")
                any_value = any_value or v is not None
                cells.append(f'<td class="num{" grp" if i == 0 else ""}">{_us(v)}</td>')
        if any_value:
            body.append(f'<tr><td class="wid">{wid} <span class="desc">{describe(wid)}'
                        f'</span></td>{"".join(cells)}</tr>')
    if not body:
        return ""
    warn = ""
    if mismatch := open_loop_mismatch(runs):
        warn = f'<div class="banner soft"><b>Not compared.</b> {mismatch}</div>'
    if instrument := open_loop_is_instrument_limited(runs):
        warn = f'<div class="banner">{instrument}</div>' + warn
    return warn + _engine_grouped(runs, [n for n, _ in COMPACT_LAT_COLS], body)


def recall_table(runs: list[Run]) -> str:
    """The conformance side of W10: recall against the fp64 oracle, by `ef`.

    §7.4: "a single QPS number without its recall is meaningless and this
    project should never emit one." The throughput table above is that number;
    this is the rest of it. §4.1 keeps the two in separate tools on purpose,
    bfb measures load and the conformance binary measures relevance, and they
    meet here on `ef`.
    """
    have = [r for r in runs if r.recall.get("points")]
    if not have:
        return ""
    efs: list = []
    for r in have:
        for p in r.recall["points"]:
            key = "exact" if p.get("exact") else p.get("ef")
            if key not in efs:
                efs.append(key)
    body = []
    for ef in efs:
        cells = []
        for r in have:
            p = next((x for x in r.recall["points"]
                      if ("exact" if x.get("exact") else x.get("ef")) == ef), None)
            if p is None:
                cells.append('<td class="num muted">-</td>' * 4)
                continue
            # Every field with `.get`, and a dash where a sweep does not carry
            # one. These were subscripts: a point without the CI pair — an
            # older conformance binary, or a row whose recall was not
            # measurable, where the binary writes null — raised a KeyError
            # inside `build`, so one such point took down the whole report
            # rather than one cell of one table.
            lo, hi = p.get("recall_at_10_ci95_low"), p.get("recall_at_10_ci95_high")
            # On hover rather than as two more lines under every figure: the
            # table is read for the numbers, and the interval is the method.
            ci = (f"95% CI [{lo:.4f}, {hi:.4f}]"
                  if lo is not None and hi is not None else "")
            # The spread across the run's own repeated builds, where it has
            # them. On strawmANN it is the load-bearing number at the top of
            # this table: two builds of one collection, same parameters, same
            # host, differed by 0.0028 at ef=512 while the two *engines*
            # differed by 0.0001. Printing the folded recall alone would state
            # a draw as a property.
            spread = p.get("rep_spread")
            if isinstance(spread, (int, float)) and spread > 0:
                ci += (("; " if ci else "") + f"±{spread / 2:.4f} over "
                       f'{p.get("reps", 0)} builds')
            num = lambda v, f: format(v, f) if isinstance(v, (int, float)) else "-"
            # §8.9's recall@100, from the limit-100 sweep, matched on the same
            # `ef`. Absent below ef=100 by construction, not by omission: that
            # sweep drops the ef values under its own limit because the engines
            # would have searched at `max(ef, limit)` and recorded a width they
            # did not run.
            k = next((x for x in (r.recall_k100.get("points") or [])
                      if not x.get("exact") and x.get("ef") == ef), None)
            r100 = num((k or {}).get("recall_at_100"), ".4f")
            tip = f' title="{ci}"' if ci else ""
            cells.append(f'<td class="num grp">{num(p.get("recall_at_1"), ".4f")}</td>'
                         f'<td class="num"{tip}>'
                         f'{num(p.get("recall_at_10"), ".4f")}</td>'
                         f'<td class="num">{r100}</td>'
                         f'<td class="num">'
                         f'{num(p.get("mean_relative_distance_error"), ".2e")}</td>')
        body.append(f'<tr><td class="wid">{"exact" if ef == "exact" else f"ef={ef}"}</td>'
                    f'{"".join(cells)}</tr>')
    note = ""
    first = have[0].recall
    if first.get("queries"):
        note = (f'<p class="note">{first["queries"]:,} held-out queries, limit '
                f'{first.get("limit", 10)}, ε={first.get("epsilon")}, against the fp64 '
                f'oracle rather than against the other engine. Base checksum '
                f'<code>{first.get("base_checksum", "?")}</code>: the same corpus the '
                f'latency rows were measured on.'
                + _build_spread_note(have) + '</p>')
    # The CI and the build spread are two more lines under every recall@10,
    # and the sentence under the table is the method; both behind one
    # disclosure, so the table reads as five rows of numbers.
    more = (f'<details class="more"><summary>How recall was measured</summary>{note}'
            f'</details>' if note else "")
    return (_engine_grouped(have, ["recall@1", "recall@10", "recall@100", "MRDE"], body,
                            first="ef") + more)


def _build_spread_note(have: list[Run]) -> str:
    """What the repeated builds disagreed by, said once under the table.

    A folded run rebuilds the collection every pass, so `rep_spread` is the
    range of recall@10 across independent graphs of the same corpus at the same
    parameters. It belongs beside the interval because a reader comparing two
    engines at the top of this table is otherwise comparing two draws: the two
    were 0.0028 apart on one engine and 0.0001 on the other in the same run,
    and the engines themselves were 0.0001 apart.

    Silent on a single-pass run, which has nothing to say here.
    """
    # At one `ef`, not at each engine's own widest.
    #
    # Comparing maxima independently is meaningless when they peak in different
    # places. On the sift1m run that reported strawmANN 0.0031 at
    # ef=64 against Qdrant 0.0014 at ef=64 — a factor of two, and the wrong
    # story. Held at ef=512, where the matched-recall ratios are most leveraged,
    # the same builds spread 0.00288 and 0.00009: a factor of thirty-two. The
    # note under the matched-recall table is why the top of the sweep is the
    # place to look — 0.003 of recall spans a factor of 1.8 in throughput there.
    per_ef: dict[object, dict[str, float]] = {}
    reps = 0
    for r in have:
        for pt in r.recall.get("points") or []:
            sp = pt.get("rep_spread")
            if isinstance(sp, (int, float)):
                per_ef.setdefault(pt.get("ef"), {})[r.label] = sp
                reps = max(reps, pt.get("reps") or 0)
    shared = {ef: d for ef, d in per_ef.items()
              if len(d) == len(have) and isinstance(ef, int)}
    if not shared:
        return ""
    top = max(shared)
    at_top = shared[top]
    detail = "; ".join(f"{lab} {sp:.5f}" for lab, sp in at_top.items())
    tail = ""
    if len(at_top) == 2:
        (la, a), (lb, b) = at_top.items()
        (hl, hv), (ll, lv) = ((la, a), (lb, b)) if a > b else ((lb, b), (la, a))
        if lv > 0 and hv / lv >= 4:
            tail = (f" {hl}'s graphs disagree with each other {hv / lv:.0f}x as much "
                    f"as {ll}'s do, so a ratio read at the top of this table reports "
                    f"which graph the build produced as much as which engine built it.")
    return (f" Each pass rebuilds the collection. Across {reps} independent builds of "
            f"it, recall@10 at ef={top} spread by: {detail}.{tail}"
            + _graph_quality_note(have))


def _graph_quality_note(have: list[Run]) -> str:
    """What the engine said about the graphs it built, where it says anything.

    The recall spread above is the *effect*; this is the graph. §8.7 accepts
    that a parallel build depends on its thread interleaving, so the checksums
    differing is expected and not the point — the point is whether the orphan
    count moves with the recall, since a node nothing points at is invisible at
    any `ef`. Qdrant prints no such line, so this speaks for whichever engines
    do and says nothing for the rest.
    """
    out = []
    for r in have:
        builds = (r.meta or {}).get("graph_builds") or []
        vals = [b.get("unreachable") for b in builds
                if isinstance(b.get("unreachable"), int)]
        nodes = [b.get("nodes") for b in builds if isinstance(b.get("nodes"), int)]
        if not vals or not nodes:
            continue
        span = (f"{min(vals):,}" if min(vals) == max(vals)
                else f"{min(vals):,}–{max(vals):,}")
        out.append(f"{r.label} {span} of {max(nodes):,}")
    if not out:
        return ""
    return (f" Over the same builds the engine reported unreachable nodes: "
            f"{'; '.join(out)} — a node nothing points at is invisible at any "
            f"ef, so that is where a graph-quality difference shows rather than "
            f"being inferred from the recall beside it." + _seed_note(have))


def _seed_note(have: list[Run]) -> str:
    """Which level seed the graphs above were drawn at.

    Provenance, §8.7, and no more than that. The page used to add that four
    seeds span 0.00216 of recall@10 at `ef` 512, from a `decisions.md` table
    that did not reproduce: re-measured with `graph-diff --seeds`, six builds
    at four seeds sit within 0.00003 under the engine's own level draw. The
    0.0023 spread is real only under `assignLevelKey`, which nothing in the
    engine uses.

    Qdrant draws every point's level from a thread-local RNG seeded by the OS
    (`segment_builder.rs`, `rand::rng()`), so it has no seed to name and this
    says nothing for it.
    """
    out = []
    for r in have:
        seeds = sorted({b.get("seed") for b in ((r.meta or {}).get("graph_builds") or [])
                        if b.get("seed")})
        if len(seeds) == 1:
            out.append(f"{r.label} 0x{seeds[0]}")
        elif seeds:
            # Two collections at two seeds is not a thing any path does, and
            # their recall curves would not be comparable if it were.
            out.append(f"{r.label} {len(seeds)} DIFFERENT SEEDS: "
                       + ", ".join(f"0x{x}" for x in seeds))
    if not out:
        return ""
    return (f" Level seed: {'; '.join(out)}. Recorded as provenance: under this "
            f"engine's level draw, six builds at four seeds sit within 0.00003 of "
            f"recall@10 at ef 512, well inside the spread above.")


def storage_rows_table(runs: list[Run]) -> str:
    """Disk work per workload, rather than one total for the run.

    The total answers "what did it cost"; this answers "where", and the answer
    is the interesting part: ingest writes, search should not, and a search row
    doing block-layer reads is an engine going to disk to answer a query.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    body = []
    for wid in ids:
        cells, any_value = [], False
        for run in runs:
            row = run.by_id().get(wid, {})
            for key, fmt in (("disk_read_bytes", procstat.human_bytes),
                             ("disk_write_bytes", procstat.human_bytes),
                             ("disk_read_ops", procstat.human_count),
                             ("disk_write_ops", procstat.human_count)):
                v = row.get(key)
                any_value = any_value or v is not None
                cells.append(f'<td class="num">{fmt(v) if v is not None else "-"}</td>')
        if any_value:
            body.append(f'<tr><td class="wid" title="{describe(wid)}">{wid}</td>'
                        f'{"".join(cells)}</tr>')
    if not body:
        return ""
    head = "<th>workload</th>" + "".join(
        f'<th class="num">{r.label} read</th><th class="num">{r.label} written</th>'
        f'<th class="num">{r.label} read ops</th><th class="num">{r.label} write ops</th>'
        for r in runs)
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def profile_of(run: Run) -> str:
    """Which experiment a run is: `isolated`, `as-deployed`, or unknown.

    `run.json` first, because that is where the harness records it; `env.txt`
    second, so a run measured by an older harness that still captured the gate
    output is readable; `unknown` last, which is what every run made before the
    profile existed will say. Never guessed from `isolcpus` here — one
    definition of the word, and `bench/setup.py` owns it.
    """
    p = (run.meta or {}).get("profile")
    if p:
        return str(p)
    m = re.search(r"measurement profile: (\S+)", run.env or "")
    return m.group(1) if m else "unknown"


def profiles_of(runs: list[Run]) -> list[str]:
    return sorted({profile_of(r) for r in runs})


def segment_policy_of(run: Run) -> str:
    """Which Qdrant experiment a run is, or `unnamed`.

    The second axis called `as-deployed` in this project, and deliberately
    reported beside the first: `profile_of` is the *scheduler* left as it
    ships, this is Qdrant's *segment count* left as it ships. A report that
    prints both words in one sentence is the reason neither has to be guessed
    from context.

    Read from the harness stamp, never inferred. A run made before
    `--segment-policy` existed asked for `segments: 1` and a ceiling above the
    corpus, which is exactly what `equal-work` now means — but calling it that
    here would be the report deciding what an old run intended, and the whole
    point of naming the axis is that a segment count is not a name. Those runs
    read `unnamed` and the banner prints the numbers they did ask for.
    """
    return str(((run.meta or {}).get("harness") or {})
               .get("collection", {}).get("segment_policy") or "unnamed")


def segment_policies_of(runs: list[Run]) -> list[str]:
    return sorted({segment_policy_of(r) for r in runs})


def segments_requested(runs: list[Run]) -> str:
    """What an `unnamed` run asked for, so the banner can say it in numbers."""
    asked = sorted({str(((r.meta or {}).get("harness") or {})
                        .get("collection", {}).get("segments", "?"))
                    for r in runs})
    return ", ".join(asked)


def _cell(row: dict, key: str, scale: float = 1.0, unit: str = "",
          digits: int = 0) -> tuple[str, bool]:
    """One scheduler cell, and whether the row had the figure at all.

    The flag is what keeps a workload out of the table entirely when no run
    measured it, so a report over older results has no half-empty section.
    """
    v = row.get(key)
    if v is None:
        return "-", False
    return f"{v * scale:,.{digits}f}{unit}", True


def _faults_cell(row: dict) -> tuple[str, bool]:
    """Minor and major faults in one column, because the pair is the reading.

    Minor faults alone say the engine touched new pages; major faults are the
    ones that went to disk, and a major count that is not zero is the memory
    narrative — an engine that is supposed to be RAM-resident paging in. Shown
    together so the zero is visible rather than merely unmentioned.
    """
    minor, major = row.get("minor_faults"), row.get("major_faults")
    if minor is None and major is None:
        return "-", False
    fmt = lambda v: "?" if v is None else f"{v:,.0f}"
    return f"{fmt(minor)} / {fmt(major)}", True


def _switches_cell(row: dict) -> tuple[str, bool]:
    """Voluntary and involuntary context switches in one column.

    Both, because they mean opposite things and only the pair reads. An
    **involuntary** switch is the scheduler taking the core away, and it is the
    pinning claim: an engine pinned to its own cores that is being preempted is
    not pinned. A **voluntary** switch is the engine *choosing* to sleep —
    blocking on a futex, or on a socket, or on a disk — and per unit of work it
    is the cheapest signal there is of lock contention: the same row, doing the
    same queries, sleeping more often is a row spending more of itself waiting
    for another thread to let go.

    It is not a *proof* of contention. A voluntary switch is also what an idle
    connection thread does when it waits for the next request, and a row that
    blocks more may simply be doing more I/O — `blkio_delay_s` in the stalls
    table is what separates those. What it is, is free, already on disk for
    every row ever measured, and the first place to look.
    """
    vol, invol = row.get("ctx_switches_voluntary"), row.get("ctx_switches_involuntary")
    if vol is None and invol is None:
        return "-", False
    fmt = lambda v: "?" if v is None else f"{v:,.0f}"
    return f"{fmt(vol)} / {fmt(invol)}", True


def _threads_cell(row: dict) -> tuple[str, bool]:
    """Live threads at the end of the row, and how much of it they account for.

    The cause, printed beside the effect. When `sched_coverage` is below the
    floor every thread-summed column in this table is a dash, and until this
    was shown the reason lived only in `procstat`'s source: an index build does
    its work in threads that exit before the row does and take their
    `/proc/<tid>` counters with them.
    """
    threads, cov = row.get("threads"), row.get("sched_coverage")
    if threads is None and cov is None:
        return "-", False
    text = "?" if threads is None else f"{threads:,.0f}"
    if cov is not None and cov < 0.995:
        # Only where it is not ~1: a coverage of 100% on every healthy row is
        # a column of noise that hides the one row where it matters.
        text += f' <span class="warn">{cov:.0%}</span>'
    return text, True


def scheduler_rows_table(runs: list[Run]) -> str:
    """What the scheduler and the fault handler did, per workload.

    Throughput says how fast a row went; this says whether the engine was
    *running* while it did.

      * **cpu** against wall clock and core count: how much of the machine the row
        used. Well under 100% was not throughput-bound, whatever it is quoted for.
      * **waiting** is `runqueue_wait_s` — runnable, not given a core. Separates
        "slow" from "starved", i.e. an engine result from a §7.1 failure.
      * **switches** voluntary over involuntary, and **migrations**: the pinning
        claim checked rather than asserted, since an engine pinned to 4-11 that
        migrates is not pinned. Voluntary switches are the free proxy for lock
        contention (`_switches_cell`).
      * **faults**: a preallocating engine takes most of its minor faults at
        startup, so a row that faults is touching something new, and any major
        fault is a RAM-resident engine going to disk.

    The counters span a slightly wider window than `wall` — snapshots are taken
    either side of the timed region, never inside it — so the CPU total includes a
    few milliseconds of setup, which is the safe direction.

    Rows whose `sched_coverage` fell below the floor show cpu and faults and a dash
    for the rest: an index build spends its time in threads that exit before the
    row does. A dash is "not measurable here", not a zero.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    body = []
    for wid in ids:
        cells, any_value = [], False
        for run in runs:
            row = run.by_id().get(wid, {})
            cpu = (row.get("cpu_user_s") or 0) + (row.get("cpu_system_s") or 0)
            wall = row.get("wall_s") or row.get("duration_s")
            busy = (f"{cpu / wall * 100:,.0f}%" if cpu and wall else "-")
            for i, (text, present) in enumerate((
                    (f"{cpu:,.1f} s" if cpu else "-", bool(cpu)),
                    (busy, busy != "-"),
                    _cell(row, "runqueue_wait_s", 1000, " ms", 1),
                    _switches_cell(row),
                    _cell(row, "migrations"),
                    _faults_cell(row),
                    _threads_cell(row))):
                any_value = any_value or present
                cells.append(f'<td class="num{" grp" if i == 0 else ""}">{text}</td>')
        if any_value:
            body.append(f'<tr><td class="wid" title="{describe(wid)}">{wid}</td>'
                        f'{"".join(cells)}</tr>')
    if not body:
        return ""
    return _engine_grouped(runs, ["cpu", "of wall", "waiting", "switches vol/invol",
                                  "migrations", "faults min/maj", "threads"], body)


def _psi_pair_cell(row: dict, prefix: str) -> tuple[str, bool]:
    """One resource's `some` and `full` stall seconds, together.

    The pair is the reading. `some` means at least one task was stalled on the
    resource, which a busy engine does constantly and which costs nothing on
    its own; `full` means *every* task in the engine's cgroup was stalled,
    which is time the engine could not have used a core even if it had one. A
    large `some` beside a small `full` is an engine overlapping its waits,
    which is what it is supposed to do.

    Both are read from the engine's own cgroup, where `full` is a measurement.
    The same word at the system root means something else: `/proc/pressure/cpu`
    defines CPU `full` as zero by construction, and a reader who has seen that
    file could reasonably assume this column is the same structural zero.
    """
    some, full = row.get(f"{prefix}_some_s"), row.get(f"{prefix}_full_s")
    if some is None and full is None:
        return "-", False
    fmt = lambda v: "?" if v is None else f"{v * 1000:,.0f}"
    return f"{fmt(some)} / {fmt(full)} ms", True


def psi_scopes(runs: list[Run]) -> dict[str, str]:
    """Per run, whose stall time its pressure columns describe.

    `engine` is the engine's cgroup alone. `shared` means the cgroup also held
    the harness, `bfb` and whatever else the operator had running, so the
    figures were refused: differencing that scope across a row measures the
    desktop. Empty means the kernel has no PSI at all.

    Surfaced per run rather than per row because it is a property of how the
    engine was *started* — a container gets its own scope, a binary launched
    from a shell inherits the session's — and the fix is to start it in a scope
    of its own (`systemd-run --scope`), which is a thing the reader can do.

    `unrecorded` is the fourth answer and it is not the same as the other
    three: the row was measured before this harness recorded pressure at all.
    A report over older results that said "this kernel has no PSI" would be
    making a claim about the host from the absence of a field, which is how a
    missing measurement turns into a false statement about a machine.
    """
    out = {}
    for run in runs:
        scopes = {(r["psi_scope"] or "") if "psi_scope" in r else "unrecorded"
                  for r in run.rows}
        out[run.label] = (scopes.pop() if len(scopes) == 1 else "mixed")
    return out


def _engine_grouped(runs: list[Run], cols: list[str], body: list[str],
                    first: str = "workload") -> str:
    """A per-row table whose columns repeat once per engine, headed that way:
    the scheduler, stalls, hardware and cache-line sharing tables.

    The label was in every header, so the stalls table read "sm-sift-perf-0923
    waiting, sm-sift-perf-0923 blocked on disk, ..." five times per engine and
    the hardware table six: the metric was the tail of a long string, and with
    the table scrolled sideways nothing said where one engine's columns ended.
    The label now spans its group once, the metrics sit under it, and each
    group opens with a rule (`grp`) that the body rows repeat.
    """
    top = (f'<tr><th rowspan="2">{first}</th>'
           + "".join(f'<th class="eng" colspan="{len(cols)}">{html.escape(r.label)}</th>'
                     for r in runs) + "</tr>")
    sub = "<tr>" + "".join(f'<th class="num{" grp" if i == 0 else ""}">{c}</th>'
                           for _ in runs for i, c in enumerate(cols)) + "</tr>"
    return (f'<div class="tablewrap"><table class="grouped"><thead>{top}{sub}</thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def stalls_rows_table(runs: list[Run]) -> str:
    """Where a row's wall clock went when it was not running.

    Three ways of not running, which no throughput number distinguishes and which
    have three different fixes.

      * **waiting**: runnable and not scheduled — oversubscribed, or the pinning is
        wrong.
      * **blocked on disk** (`delayacct_blkio_ticks`): not runnable at all, waiting
        on a device. For a RAM-resident engine against a disk-backed one this is
        the column the section exists for, and a dash is not a zero: most hosts
        ship `task_delayacct=0` and report a permanent zero, indistinguishable from
        "never touched a disk" — which would manufacture the strongest claim in the
        document.
      * **pressure**: PSI, the kernel's own stall accounting for the engine's
        cgroup, split `some`/`full`.

    Every figure is a difference across the same bracket as the row's other
    counters, and each is withheld rather than guessed when its source is missing
    or its cgroup is shared.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    body = []
    for wid in ids:
        cells, any_value = [], False
        for run in runs:
            row = run.by_id().get(wid, {})
            for i, (text, present) in enumerate((
                    _cell(row, "runqueue_wait_s", 1000, " ms", 1),
                    _cell(row, "blkio_delay_s", 1000, " ms", 0),
                    _psi_pair_cell(row, "psi_cpu"),
                    _psi_pair_cell(row, "psi_io"),
                    _psi_pair_cell(row, "psi_mem"))):
                any_value = any_value or present
                cells.append(f'<td class="num{" grp" if i == 0 else ""}">{text}</td>')
        if any_value:
            body.append(f'<tr><td class="wid" title="{describe(wid)}">{wid}</td>'
                        f'{"".join(cells)}</tr>')
    if not body:
        return ""
    return _engine_grouped(runs, ["waiting", "blocked on disk", "cpu some/full",
                                  "io some/full", "memory some/full"], body)


def perf_arms(runs: list[Run]) -> dict[str, str]:
    """Which hardware-counter set each run was actually measured with.

    Measured, not asked for, and the difference is a whole section. `perf_set`
    records the *request*, and a request that failed leaves it set: attach
    `perf stat` to a Qdrant container and it fails outright — the engine runs
    as root, the harness as the operator, and `perf_event_open` on another
    user's task needs ptrace permission that `perf_event_paranoid` does not
    grant at any value. Both arms then say `default`, the mismatch check sees
    no mismatch, and the page renders strawmANN's IPC and cycles-per-query
    beside a column of dashes with nothing saying the other engine could not be
    counted at all.

    So an arm counts as measured only where a row carries a number. The reason
    it did not is on the rows too (`perf_note`), and `perf_refusals` reads it.
    """
    out = {}
    counters = [f for f in perfstat.PERF_FIELDS
                if f not in ("perf_set", "perf_events", "perf_note",
                             "perf_enabled_pct", "perf_window_s", "perf_ref_hz")]
    for run in runs:
        sets = {(r.get("perf_set") or "") for r in run.rows
                if any(r.get(f) is not None for f in counters)}
        sets.discard("")
        out[run.label] = ", ".join(sorted(sets))
    return out


def perf_refusals(runs: list[Run]) -> list[str]:
    """Arms that were asked for counters and produced none, with the reason.

    One line per arm rather than per row: the cause is a property of how the
    engine was started, so every row carries the same sentence.
    """
    out = []
    measured = perf_arms(runs)
    for run in runs:
        asked = {(r.get("perf_set") or "") for r in run.rows}
        asked.discard("")
        if not asked or measured.get(run.label):
            continue
        why = next((r.get("perf_note") for r in run.rows if r.get("perf_note")), "")
        out.append(f"{run.label} was asked for {', '.join(sorted(asked))} counters and "
                   f"produced none" + (f": {why}" if why else "."))
    return out


def _perf_cell(row: dict, key: str, digits: int = 2, unit: str = "") -> tuple[str, bool]:
    """One derived hardware figure, from `perfstat.derive`."""
    v = perfstat.derive(row).get(key)
    if v is None:
        return "-", False
    return f"{v:,.{digits}f}{unit}", True


def _per_query_cell(row: dict, key: str, digits: int = 0,
                    scale: float = 1.0, unit: str = "") -> tuple[str, bool]:
    """A raw counter divided by the queries that caused it.

    The only form in which two engines' counters can be compared. An absolute
    count is a function of how long the row ran and how many threads it ran on,
    and the two engines hold neither equal; per query is what it cost to answer
    the same question twice. A row with no query count — a load row — gets a
    dash rather than an absolute number in a per-query column.
    """
    v = perfstat.per_query(row.get(key), row.get("n_queries"))
    if v is None:
        return "-", False
    return f"{v * scale:,.{digits}f}{unit}", True


def _freq_cell(row: dict) -> tuple[str, bool]:
    """Effective frequency, and the ratio to nominal that §7.5 is defined on.

    Both in one column because the guard rail is the ratio — "arms differing by
    more than 3%" — and the reader wants the gigahertz. A row that recorded no
    reference rate shows the ratio alone rather than a frequency derived some
    other way: the obvious other way, cycles over `task-clock`, averages halted
    time into a frequency and reads 14% low on any row where the engine sleeps
    between queries. See `perfstat.derive`.
    """
    d = perfstat.derive(row)
    ratio, ghz = d.get("freq_ratio"), d.get("ghz")
    if ratio is None:
        return "-", False
    if ghz is None:
        return f'{ratio:.3f}<span class="sub">x nominal</span>', True
    return f'{ghz:,.2f}<span class="sub">{ratio:.3f}x nominal</span>', True


def hardware_rows_table(runs: list[Run]) -> str:
    """What the core did per query, from `perf stat` attached to the engine.

    §5 is a set of claims about arithmetic intensity, outstanding misses and TLB
    reach; §7.3 specifies the events that check them. Until this table they were
    checked only against `bench/micro`'s kernels, never against the engine serving
    the row the headline quotes.

      * **IPC** and **GHz**: how well the core was fed, and at what clock. §7.5
        makes the frequency a guard rail, not a detail. Derived from
        `cycles / ref-cycles` against a calibrated reference, not `task-clock`
        (`_freq_cell`).
      * **branch MPKI**: a graph traversal is a chain of data-dependent branches,
        and this is where an engine pays for one.
      * **cycles/query**: the cost model's own unit, and the most directly
        comparable number between the two engines.
      * **demand DRAM/query**: demand loads served from DRAM times the line size.
        Demand *only* — prefetcher fills are a separate counter — so a streaming
        row moved more than this says, while on a graph walk it is most of the
        traffic and is §5.2's outstanding-miss quantity.
      * **TLB walks/query**: what not taking §5.5's hugepage care costs.

    Blank on every row measured without `--perf`, which is most of them: the
    sidecar touches its subject, so it is opt-in and its cost is measured
    (`perfstat.py --overhead`) rather than assumed.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    body = []
    for wid in ids:
        cells, any_value = [], False
        for run in runs:
            row = run.by_id().get(wid, {})
            for i, (text, present) in enumerate((
                    _perf_cell(row, "ipc", 2),
                    _freq_cell(row),
                    _perf_cell(row, "branch_mpki", 2),
                    _per_query_cell(row, "perf_cycles", 0),
                    # Fills times the line size, in KiB: the same conversion
                    # `perfstat.derive` uses, read from the host rather than
                    # assumed to be 64.
                    _per_query_cell(row, "perf_dram_fills", 1,
                                    perfstat.line_bytes() / 1024, " KiB"),
                    _per_query_cell(row, "perf_dtlb_walks", 1))):
                any_value = any_value or present
                cells.append(f'<td class="num{" grp" if i == 0 else ""}">{text}</td>')
        if any_value:
            body.append(f'<tr><td class="wid" title="{describe(wid)}">{wid}</td>'
                        f'{"".join(cells)}</tr>')
    if not body:
        return ""
    return _engine_grouped(runs, ["IPC", "GHz", "branch MPKI", "cycles/query",
                                  "demand DRAM/query", "TLB walks/query"], body)


def _c2c_fills(row: dict) -> float | None:
    """Demand fills served from another core's cache, however far away.

    The three distances summed, because for this question they are one
    quantity: a line that had to be fetched from a cache rather than from DRAM
    or from this core's own levels was somewhere else, and something else had
    it. Which distance it came from is a NUMA question.

    Summable only because `perfstat.SHARING_EVENTS` picks three *disjoint*
    umasks. The natural-looking third choice, `remote_cache`, overlaps
    `near_cache` and would count the near fills twice — see the note there.
    """
    parts = [row.get(k) for k in ("perf_fills_local_ccx", "perf_fills_near_cache",
                                  "perf_fills_far_cache")]
    got = [v for v in parts if v is not None]
    return sum(got) if got else None


def sharing_rows_table(runs: list[Run]) -> str:
    """How often a load was served from another core's cache, per query.

    The cheap half of §7.3's `perf c2c` bullet. It **cannot** distinguish true
    sharing — two threads reading one datum, often intended — from false sharing,
    where two threads write different data on one 64-byte line and pass it back and
    forth for nothing. Both look identical from the load-store unit.

    What it can do is *scale*: hold the queries constant, raise the thread count,
    and a per-query cache-to-cache fill count that climbs with it is a line being
    passed around. That is the signal worth spending a `perf c2c` capture on.

    Filled only by `--perf sharing`, a second pass: the events do not fit in the
    default group, and multiplexing them would extrapolate both rather than
    measure either.
    """
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    body = []
    for wid in ids:
        cells, any_value = [], False
        for run in runs:
            row = run.by_id().get(wid, {})
            c2c, allf = _c2c_fills(row), row.get("perf_fills_all")
            per_q = perfstat.per_query(c2c, row.get("n_queries"))
            share = (c2c / allf * 100) if (c2c is not None and allf) else None
            for i, (text, present) in enumerate((
                    ("-" if per_q is None else f"{per_q:,.1f}", per_q is not None),
                    ("-" if share is None else f"{share:.1f}%", share is not None))):
                any_value = any_value or present
                cells.append(f'<td class="num{" grp" if i == 0 else ""}">{text}</td>')
        if any_value:
            body.append(f'<tr><td class="wid" title="{describe(wid)}">{wid}</td>'
                        f'{"".join(cells)}</tr>')
    if not body:
        return ""
    return _engine_grouped(runs, ["cache-to-cache fills/query", "of all fills"], body)


def perf_crosscheck(runs: list[Run]) -> list[str]:
    """Rows where `perf` and `/proc` disagree about the same quantity.

    The software events cost no PMU counter, so the sidecar carries page faults
    and context switches as well — quantities `procstat` already reads from
    `/proc`. Carrying both is only worth it if the disagreement is *reported*:
    two instruments over the same window that do not agree mean one of them did
    not cover the window, and the usual cause is the one `sched_coverage`
    describes, a thread that exited and took its `/proc` counters with it.

    A disagreement is a finding about the instruments, not about the engines,
    so it is a note under the table rather than a column in it.
    """
    out = []
    for run in runs:
        for row in run.rows:
            for a, b, what in (("perf_minor_faults", "minor_faults", "minor faults"),
                               ("perf_ctx_switches", "ctx_switches_voluntary",
                                "context switches")):
                x, y = row.get(a), row.get(b)
                if x is None or y is None:
                    continue
                if b == "ctx_switches_voluntary":
                    # /proc splits them and perf does not, so the comparable
                    # quantity is the sum.
                    inv = row.get("ctx_switches_involuntary")
                    if inv is None:
                        continue
                    y += inv
                if max(x, y) < 1000:
                    continue  # small counts differ by rounding, not by coverage
                if abs(x - y) / max(x, y) > 0.05:
                    out.append((run.label, row["id"], what,
                                abs(x - y) / max(x, y)))
    # One line per (label, quantity), naming the rows and the worst gap, rather
    # than a sentence per row: the same three rows disagreeing on minor faults
    # is one finding about the instruments, and spelling out both counts for
    # each of them was 120 words saying it.
    grouped: dict[tuple[str, str], list] = {}
    for label, wid, what, gap in out:
        grouped.setdefault((label, what), []).append((wid, gap))
    lines = []
    for (label, what), hits in grouped.items():
        ids = ", ".join(w for w, _ in hits)
        lines.append(f"{label}: perf and /proc disagree on {what} by up to "
                     f"{max(g for _, g in hits):.0%} on {ids}")
    return lines


def licence_of(runs: list[Run]) -> dict:
    """What §8 and §7.4 permit these numbers to be used for.

    The differ already decides this and prints it; a report that omits it hands
    a reader a table of ratios with no statement of whether the two engines
    were ever shown to compute the same answers. Absent conformance is reported
    as absent, which is itself the finding on this host today.
    """
    conf = next((r.conformance for r in runs if r.conformance), {})
    # `compare.licence` refuses a pair whose arms carry different conformance
    # rows; this card used to take the first arm's and say "licensed" under a
    # top banner saying UNLICENSED.
    hashes = {r.conformance.get("hash") for r in runs if r.conformance}
    if len(hashes) > 1:
        return {
            "known": True,
            "delta": None,
            "delta_missing": False,
            "tier": "mixed",
            "perf": False,
            "comparative": False,
            "hash": None,
            "headline": "The two labels carry different conformance rows.",
            "detail": ("§8 licenses a comparison from one differ run over both "
                       "engines; these arms were licensed by different runs "
                       f"({', '.join(sorted(h or 'none' for h in hashes))}), so "
                       "nothing here is a comparative claim. Run the differ once "
                       "against both labels."),
        }
    if not conf:
        # The command named here used to be `cargo run --release -- differ
        # --json bench/results/<label>/conformance.json`, which is eight flags
        # short of the invocation `fullrun.run_conformance` builds and starts
        # neither engine -- the differ speaks gRPC to both at once. A reader
        # who followed it got an error, on a page already telling them their
        # numbers were unpublishable. The phase ignores `--skip strawmann` and
        # `--skip qdrant`, so `fullrun.py` can run it alone against labels
        # already on disk, and that is what to say.
        labels = " ".join(f"--{'strawmann' if i == 0 else 'qdrant'}-label {r.label}"
                          for i, r in enumerate(runs[:2]))
        names = datasets_of(runs)
        ds = names[0] if len(names) == 1 else ""
        # `--qdrant-binary` defaults to the pinned image. Omitting it here
        # would license rows measured against a source-built Qdrant with a
        # conformance row taken against a different one, and nothing compares
        # the two versions afterwards. See `compare.conformance_recovery`.
        binary = next((b for r in runs
                       if (b := (r.meta.get("qdrant") or {}).get("binary"))), "")
        # Named, not arranged: the differ records the commit the *checkout* is
        # on, not the one that built the binary in `zig-out`, so neither the
        # flag above nor `--skip build` can guarantee §8's "same build".
        named = [f"strawmANN {c}" for r in runs
                 if (c := (r.meta.get("strawmann") or {}).get("commit"))]
        named += [f"Qdrant {v}" for r in runs
                  if (v := (r.meta.get("qdrant") or {}).get("version"))]
        builds = (f" These rows were measured at {' and '.join(named)}; §8 wants "
                  f"the row from those builds, so check the checkout before "
                  f"trusting the one it writes." if named else "")
        return {
            "known": False,
            "delta": None,
            "delta_missing": False,
            "headline": "No conformance run is recorded beside these numbers.",
            "detail": ("§8: a performance number is publishable only if the same "
                       "build, on the same data, has a green conformance row. "
                       "Nothing here has one, so every figure below is a "
                       "development observation rather than a result about the "
                       "engines. The rows themselves are fine; run the differ "
                       "against them with `python3 bench/harness/fullrun.py"
                       + (f" --dataset {ds}" if ds else "")
                       + (f" --qdrant-binary {_tilde(binary)}" if binary else "")
                       + f" --skip build --skip strawmann --skip qdrant {labels}`."
                       + builds),
        }
    tier = conf.get("tier_reached") or "none"
    # §8.9 lists `max|Δscore|` and `p99|Δscore|` as row fields and §8.1 calls
    # value equality within ε "the load-bearing claim". The differ measured
    # both and `Distribution::describe` rendered them into T1's prose, so the
    # page carried the verdict and not the number behind it — and §8.4's
    # reason for measuring a distribution rather than a bit ("a run that
    # passes with a max delta 10x worse than yesterday's is a regression")
    # needs the number. A run whose differ predates the structured fields says
    # so rather than showing a blank.
    delta = None
    if conf.get("max_delta") is not None:
        # `epsilon_relative` is a *bool* (`tolerance.rs`: `pub relative: bool`)
        # saying which kind of tolerance ε is, not a second ε. Formatted as a
        # number it rendered `True` as `1.000e+00`, so the page claimed a
        # tolerance of 100% beside a max delta of zero — a plausible-looking
        # figure that was a boolean in disguise.
        delta = {"max": conf["max_delta"], "p99": conf.get("p99_delta"),
                 "eps": conf.get("epsilon"),
                 "kind": "relative" if conf.get("epsilon_relative") else "absolute",
                 # `calibrate` writes a measured cell; without one the differ
                 # falls back to §8.4's analytic floor and records which. The
                 # page said "calibrated" for both.
                 "source": conf.get("epsilon_source")}
    return {
        "known": True,
        "delta": delta,
        "delta_missing": delta is None,
        "tier": tier,
        "perf": bool(conf.get("licenses_perf")),
        "comparative": bool(conf.get("licenses_comparative")),
        "hash": conf.get("hash"),
        "headline": f"Conformance tier reached: {tier}.",
        "detail": (
            ("T1 passed, so single-engine performance rows are licensed (§8). "
             if conf.get("licenses_perf") else
             "T1 did not pass, so §8 licenses no performance number here. ")
            + ("T3 passed, so a strawmANN-vs-Qdrant throughput comparison is "
               "licensed: the engines are at equal recall (§7.4)."
               if conf.get("licenses_comparative") else
               "T3 did not pass, so §7.4 does not license a throughput or latency "
               "comparison between the engines: they are at unequal recall, and "
               "the faster one may simply be searching less.")),
    }


#: Rows the summary card leaves out of its ranges, and why. Both are in the
#: table with their caveats; neither belongs in the first sentence a skimming
#: reader sees. W0's 34.52x was once the top of the headline range and measures
#: traversal overhead at d=4, which the table itself says is not a search over
#: real vectors. Applied to the losses list for the same reason it was applied
#: to the range: a summary that excludes a row from the wins and admits it to
#: the losses is not a summary, it is an argument.
SUMMARY_EXCLUDES = {
    "W0": "the d=4 floor, which measures graph traversal overhead rather than search",
}


def summary(runs: list[Run], df: pd.DataFrame) -> dict:
    """The conclusion, computed rather than written.

    A reader who opens this page should be able to state what it shows and what
    it does not before scrolling. Every figure here is derived from the tables
    below, so it cannot drift from them the way a hand-written abstract does.
    """
    have = [r for r in runs if r.recall.get("points")]
    out: dict = {"matched": None, "losses": [], "licence": None, "matched_refusal": ""}

    if len(runs) == 2:
        out["matched_refusal"] = matched_recall_refusal(runs)
        # A summary that states only the wins is an advertisement. These are
        # the rows the same arithmetic, with the same refusals, calls losses.
        out["losses"] = losses(runs)

    if len(have) == 2 and not out["matched_refusal"]:
        a, b = have
        pa, pb = frontier_points(a, bfb_only=True), frontier_points(b, bfb_only=True)
        rows = matched_ratios(pa, pb)
        if rows:
            bands = [r["band"] for r in rows if r["band"]]
            out["matched"] = {"lo": min(r["ratio"] for r in rows),
                              "hi": max(r["ratio"] for r in rows),
                              "lo_recall": min(r["recall"] for r in rows),
                              "hi_recall": max(r["recall"] for r in rows),
                              # The same range once the recall CIs the Recall
                              # section prints are carried through the
                              # interpolation. Stated beside the headline
                              # because it is wider than the headline by more
                              # than the headline's own spread.
                              "ci_lo": min(b_[0] for b_ in bands) if bands else None,
                              "ci_hi": max(b_[1] for b_ in bands) if bands else None,
                              "a": a.label, "b": b.label,
                              "smoke": smoke_ordering(a, b)}
    return out


def losses(runs: list[Run]) -> list[dict]:
    """Rows where the first engine is the slower one, past the noise floor.

    Read from `compare.Row` like every other ratio on the page, so a row the
    throughput table refused cannot appear here either, and `SUMMARY_EXCLUDES`
    (the d=4 floor, the declined row) is honoured for the same reason
    it is honoured in the headline range.
    """
    if len(runs) != 2:
        return []
    noise = load_noise(runs)
    out = []
    for jr in compare.joined(runs[0].label, runs[1].label,
                             runs[0].by_id(), runs[1].by_id()):
        if jr.id in SUMMARY_EXCLUDES:
            continue
        v = ratio_value(jr.ratio)
        if v is None or v >= 1.0:
            continue
        # A bullet in the summary is a claim, so it needs a floor to clear.
        # Without this a row with no measured spread was listed as a loss on
        # the strength of never having been judged — the same trap
        # `ratio_verdict` avoids by refusing to stay silent, inverted.
        if not ratio_verdict(jr.id, v, noise).startswith("clears"):
            continue
        out.append({"id": jr.id, "desc": describe(jr.id), "ratio": jr.ratio})
    return out


def grouped_losses(items: list[dict]) -> list[dict]:
    """`losses`, one entry per question rather than per sweep point.

    The summary listed six rows, four of them points of two filtered `ef`
    sweeps, each under its full description: a paragraph where the finding is
    "filtered search and exact search". A sweep point joins its base row
    (`W12-sel1-ef64` joins `W12-sel1`) and the entry carries the ratio range.
    """
    out: dict[str, dict] = {}
    for l in items:
        m = SWEEP_ID.match(l["id"])
        fam = m["fam"] if m else l["id"]
        g = out.setdefault(fam, {"id": fam, "ids": [], "ratios": [],
                                 "desc": describe(fam) or re.sub(r",? ef=\d+", "", l["desc"])})
        g["ids"].append(l["id"])
        g["ratios"].append(ratio_value(l["ratio"]))
    for g in out.values():
        lo, hi = min(g["ratios"]), max(g["ratios"])
        g["ratio"] = f"{lo:.2f}x" if lo == hi else f"{lo:.2f} to {hi:.2f}x"
        g["points"] = len(g["ids"])
        # A lone sweep point is named as itself: under its family's id it read
        # as the base row, which the throughput table shows winning.
        if g["points"] == 1:
            g["id"] = g["ids"][0]
    return list(out.values())


def _kpi(label: str, values: list[str], sub: str, raw: list[float]) -> dict:
    """A two-engine tile; `better` is the index of the lower figure, if any."""
    better = None if min(raw) == max(raw) else raw.index(min(raw))
    return {"label": label, "cells": values, "sub": sub, "better": better}


def kpis(runs: list[Run], df: pd.DataFrame, summ: dict) -> list[dict]:
    """The summary's headline figures, each from the section that states it.

    Throughput at matched recall (the page's conclusion), the open-loop tail at
    the highest offered fraction, upload plus build, and the memory and disk
    the run cost. A tile the data cannot support is left out rather than
    drawn empty: no matched ratio, a latency the generator is known to have
    distorted, or a residency split that refuses the storage rows.
    """
    if len(runs) != 2:
        return []
    a, b = runs
    out = []
    m = summ.get("matched")
    if m:
        rng = f'{m["lo"]:.2f}x' if m["lo"] == m["hi"] else f'{m["lo"]:.2f} to {m["hi"]:.2f}x'
        out.append({"label": "throughput at equal recall", "headline": rng,
                    "sub": f'{a.label} over {b.label}, recall@10 '
                           f'{m["lo_recall"]:.3f} to {m["hi_recall"]:.3f}',
                    "cells": [], "better": None})
    if not open_loop_is_instrument_limited(runs):
        opens = [r for r in a.rows if r.get("load_mode") == workloads.LoadMode.open_loop
                 and r.get("rps_fraction")]
        top = max(opens, key=lambda r: r["rps_fraction"], default=None)
        if top is not None:
            ps = [((run.by_id().get(top["id"]) or {}).get("latency") or {}).get("client_p99_us")
                  for run in runs]
            if all(p is not None for p in ps):
                whose = "each engine's own" if open_loop_mismatch(runs) else "measured"
                out.append(_kpi(f'p99 latency at {top["rps_fraction"]:.0%} load',
                                [_us(p) for p in ps],
                                f'{top["id"]}, open loop, {whose} saturation', ps))
    totals = []
    for run in runs:
        by = run.by_id()
        parts = [_seconds_of(by[w]) for w in INGEST_SUM if w in by]
        totals.append(sum(parts) if parts and all(p is not None for p in parts) else None)
    if all(t is not None for t in totals):
        out.append(_kpi("upload and index build", [f"{t:,.0f} s" for t in totals],
                        " + ".join(INGEST_SUM), totals))
    if not placement_refusal(runs):
        for label, key, agg in (("peak memory (RSS)", "rss_peak_bytes", max),
                                ("written to disk", "disk_write_bytes", sum)):
            vals = []
            for run in runs:
                got = [r.get(key) for r in run.rows if r.get(key) is not None]
                vals.append(agg(got) if got else None)
            if all(v is not None for v in vals):
                out.append(_kpi(label, [procstat.human_bytes(v) for v in vals],
                                "largest across the run" if agg is max else "whole run",
                                vals))
    for k in out:
        k.setdefault("headline", "")
    return out


def smoke_ordering(a: Run, b: Run) -> dict | None:
    """How the conformance binary's own single-client rate orders the engines.

    The headline said "the same recall sweep read against the conformance
    harness's own single-client rate gives the opposite ordering" as a fixed
    sentence in the template. It happened to be true of the run it was written
    on and would have been printed unchanged on one where it was not. So it is
    computed: per `ef` where both sweeps carry a smoke rate, the ratio a/b, and
    a verdict of `opposite`, `same` or `mixed` against the bfb matched-recall
    ratio (which is > 1 when `a` is faster). `None` when either side lacks the
    smoke rate, and the template then says nothing about it.

    The smoke rate is a `QueryBatch` of 32 from one client, sequentially: a
    different instrument from bfb, which is why no ratio between the two is
    formed and why its ordering is reported as an observation, not a result.
    """
    sa = {int(p["ef"]): p.get("smoke_qps") for p in a.recall.get("points", [])
          if p.get("ef") is not None and not p.get("exact")}
    sb = {int(p["ef"]): p.get("smoke_qps") for p in b.recall.get("points", [])
          if p.get("ef") is not None and not p.get("exact")}
    pairs = [(ef, sa[ef] / sb[ef]) for ef in sorted(sa) if ef in sb and sa[ef] and sb[ef]]
    if not pairs:
        return None
    ratios = [r for _, r in pairs]
    if all(r < 1 for r in ratios):
        verdict = Direction.opposite
    elif all(r >= 1 for r in ratios):
        verdict = Direction.same   # equal rates order nobody, and are not "opposite"
    else:
        verdict = Direction.mixed
    return {"lo": min(ratios), "hi": max(ratios), "n": len(pairs), "verdict": verdict}




def interleaving(runs: list[Run]) -> dict:
    """Whether the two arms were interleaved, or run one after the other.

    §7.2(5) is explicit about the differential comparison: "the same bfb
    invocation against Qdrant, same host, **interleaved (A/B/A/B, not
    A-then-B, to control for drift)**". A-then-B confounds every difference
    between the engines with everything that changed on the box between the
    two blocks — thermals, page cache, whatever else drifted over the half
    hour — and the confound points the same way for every row, so it does not
    average out and no per-row noise floor can see it.

    Computed from the row timestamps rather than declared: merge every row's
    `when` across both runs, sort, and count the maximal same-engine stretches.
    Two blocks is A-then-B. More than two is some interleaving, and the count
    is how much. `{}` whenever the stamps cannot answer it: fewer than two
    engines carry them, they are all one instant, or an instant appears on
    both sides —
    unknown and sequential are different answers.
    """
    if len(runs) != 2:
        return {}
    # A folded run was alternated by construction: `fullrun --reps N` runs
    # pass 1 on A then B, pass 2 on A then B, and so on. Its rows carry the
    # first pass's timestamps, so the stamp test below would read the fold as
    # A-then-B and print a banner about a confound the run was built to
    # remove. The record says what happened; the stamps only survive it.
    reps = [(r.meta or {}).get("reps") for r in runs]
    if all(isinstance(n, int) and n > 1 for n in reps):
        return {"blocks": 2 * min(reps), "sequential": False,
                "first": runs[0].label,
                "spans": [f"{r.label} folded from {n} alternated passes"
                          for r, n in zip(runs, reps)]}
    stamped = [(r["when"], run.label) for run in runs for r in run.rows if r.get("when")]
    if len({lab for _, lab in stamped}) < 2 or len({w for w, _ in stamped}) < 2:
        return {}
    per_engine: dict[str, set] = {}
    for w, lab in stamped:
        per_engine.setdefault(lab, set()).add(w)
    # An instant that appears on both sides makes the merge order the sort's
    # rather than the run's, and a banner saying "not interleaved" on the
    # strength of a tie is worse than no banner.
    if set.intersection(*per_engine.values()):
        return {}
    stamped.sort()
    blocks = 1 + sum(1 for (_, a), (_, b) in zip(stamped, stamped[1:]) if a != b)
    spans = []
    for run in runs:
        w = sorted(r["when"] for r in run.rows if r.get("when"))
        if w:
            spans.append(f"{run.label} {w[0]} to {w[-1]}")
    return {"blocks": blocks, "sequential": blocks <= 2,
            "first": stamped[0][1], "spans": spans}


def same_host(runs: list[Run]) -> bool:
    """One machine, or not.

    §7.1: results from different environments never share a chart. The test
    is the *static* half of the gate, the CPU model, governor, boost, SMT,
    isolation, THP and NUMA lines, plus the recorded hostname when there is
    one. The quiescence line and the environment hash change from run to run
    on the same box (the hash covers the busy-process list), and reading a
    hash difference as "two environments" put two gate cards for one laptop
    in the report and a banner saying they must not share a chart.
    """
    if len(runs) < 2:
        return True
    hosts = {(r.meta.get("host") or {}).get("hostname") for r in runs}
    hosts.discard(None)
    if len(hosts) > 1:
        return False
    first = [(c.kind, c.text) for c in runs[0].static_checks]
    return all([(c.kind, c.text) for c in r.static_checks] == first for r in runs[1:])


def bandwidth_of(runs: list[Run]) -> dict:
    """The memory bandwidth of the machine, and where the number came from.

    Recorded per run in `run.json` (`host.memory_bandwidth`) by
    `provenance.memory_bandwidth`: the strawmann server's own startup banner
    when `fullrun.py` kept its log, else `strawmann --probe` at run start, else
    backfilled after the fact. It is a host property, so on one machine the
    report says it once; it prefers the startup measurement, and if the two
    runs' figures disagree by more than 10% it shows both, because that gap
    is itself a fact about the runs (a busy box, a different power state).
    """
    recs = [(r.label, (r.meta.get("host") or {}).get("memory_bandwidth"))
            for r in runs]
    recs = [(l, b) for l, b in recs if b and b.get("aggregate_gbps")]
    if not recs:
        return {"text": '<span class="muted">not recorded</span>',
                "source": "no run.json carries host.memory_bandwidth; "
                          "`workloads.py backfill <label>` probes it",
                "aggregate_gbps": None, "records": []}

    def fmt(b: dict) -> str:
        t = f'<b>{b["aggregate_gbps"]:.1f} GB/s</b> aggregate ({b.get("threads", "?")} threads)'
        if b.get("single_core_gbps") is not None:
            t += (f' · {b["single_core_gbps"]:.1f} GB/s single core '
                  f'({b.get("single_core_pct_of_bus", "?")}% of bus)')
        return t

    recs.sort(key=lambda lb: 0 if "startup" in (lb[1].get("source") or "") else 1)
    label, best = recs[0]
    aggs = [b["aggregate_gbps"] for _, b in recs]
    if max(aggs) > 1.1 * min(aggs):
        text = " / ".join(f"{l}: {fmt(b)}" for l, b in recs)
        source = ("the two runs measured different bandwidths, which is a fact "
                  "about the runs, not the machine: " + "; ".join(
                      f"{l}: {b.get('source', 'unrecorded')} ({b.get('measured_at', '?')})"
                      for l, b in recs))
    else:
        text = fmt(best)
        source = f"{best.get('source', 'unrecorded')}, {best.get('measured_at', '?')}"
        if len(recs) > 1:
            source += f"; the {recs[1][0]} run agrees within 10%"
    return {"text": text, "source": source, "aggregate_gbps": best["aggregate_gbps"],
            "records": recs}


def _network_path(mode: str | None) -> str:
    """How the load generator reached the engine, which is not free.

    Publishing a port puts Docker's userland proxy in the request path: a host
    process relaying every gRPC byte, unpinned (affinity `0-11`, so neither the
    server set nor the client set, and observed running inside the cpuset the
    container was given), and charged to neither engine. strawmANN is a pinned
    native process the load generator connects to directly, so a published port
    taxes one arm of the comparison and not the other -- per request, in the
    percentiles this page compares.

    It belongs beside the digest rather than in a footnote because it is the
    same kind of fact: what actually ran. A run from before this was recorded
    says so instead of being presented as clean, since every such run published
    a port -- unknown and known-bad differ here only in that one of them can be
    checked.
    """
    # Classified once, then every case handled. The `mode == "host"` /
    # `if mode:` shape this replaces had no exhaustive form, which is how
    # `native` came to be reported as a published port for a fortnight.
    match provenance.classify_network(mode):
        case provenance.NetworkPath.native:
            return ('network <code>native</code> '
                    '<span class="pill ok">no container in the path, '
                    'like strawmann</span>')
        case provenance.NetworkPath.host:
            return ('network <code>host</code> '
                    '<span class="pill ok">same path as strawmann</span>')
        case provenance.NetworkPath.published:
            return (f'network <code>{_fmt(mode)}</code> '
                    f'<span class="pill bad">published port: a userland proxy '
                    f'relays every request (finding 26)</span>')
        case provenance.NetworkPath.unknown:
            return ('<span class="pill bad">network path not recorded</span> '
                    '<span class="muted">runs before 2026-08-18 published a '
                    'port, which put a userland proxy in the request path '
                    '(finding 26)</span>')


def _cpus_in(mask: str) -> set[int]:
    """The CPUs a list like `4-11,14` names."""
    out: set[int] = set()
    for part in mask.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _as_ranges(cpus: set[int]) -> str:
    """`{4,5,6,7,8,9,10,11}` back to `4-11`."""
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _mask_groups(text: str) -> list[tuple[str, int, int]]:
    """`(mask, threads, cores)` per group, with one-core groups coalesced.

    strawmANN pins each worker to a *single* core, so the raw capture on this
    host is nine groups — `0-11 x1, 4 x1, 5 x1, ... 11 x1` — which is true and
    unreadable, and buries the fact a reader wants: those eight workers cover
    exactly the `4-11` that was requested. Groups of one core each are merged
    into the range they span, carrying the thread count, so the cell shows
    "4-11 (8 cores) on 8 threads" beside the main thread's own line.
    """
    groups: list[tuple[str, int, int]] = []
    singles: dict[int, int] = {}
    for group in text.split(", "):
        mask, _, n = group.rpartition(" x")
        threads = int(n) if n.isdigit() else 1
        cpus = _cpus_in(mask)
        if len(cpus) == 1:
            singles[next(iter(cpus))] = singles.get(next(iter(cpus)), 0) + threads
        else:
            groups.append((mask, threads, len(cpus)))
    if singles:
        cpus = set(singles)
        groups.append((_as_ranges(cpus), sum(singles.values()), len(cpus)))
    groups.sort(key=lambda g: -g[1])
    return groups


def _affinity_cell(run: Run) -> str:
    """One engine's allowed cores, per thread group, or why there is none.

    `provenance.affinity` reports `<mask> x<threads>` per distinct mask,
    busiest first, because a per-thread-pinned engine has no single answer:
    strawmANN's `--pin --cpus` binds its workers and leaves the main thread on
    every core, while a container cpuset binds all of them. The first form of
    this cell read `/proc/<pid>/status` alone and so printed the main thread's
    mask — `0-11` for an engine whose workers were on `4-11`, against Qdrant's
    genuinely-restricted `4-11`, which stated the comparison's central fairness
    fact backwards and in the direction that flatters this project.

    A value with no `x<n>` is from that first form, and says so rather than
    being shown as though it described every thread.
    """
    from fullrun import cpu_count  # the one definition of "how many cores is `4-11`"
    meta = run.meta or {}
    lines = []

    # What the harness asked for. This is the fairness statement, and it is the
    # line to read across the two columns: `fullrun.py` gives both arms one
    # set, and where both say the same thing the comparison was set up even.
    want = meta.get("server_cpus_requested")
    if want:
        lines.append(f'<code>{html.escape(str(want))}</code> '
                     f'({cpu_count(str(want))} cores) '
                     f'<span class="muted">requested by the harness</span>')

    v = meta.get("engine_affinity")
    if not v:
        lines.append('<span class="pill bad">observed: not recorded</span> '
                     '<span class="muted">this run predates the capture</span>')
    else:
        text = str(v)
        if " x" not in text:
            # The single-mask form is `/proc/<pid>/status`, i.e. the main
            # thread. For an engine that pins its workers individually that is
            # not where the work ran, and reported plainly it invented a core
            # advantage: `0-11` for strawmANN against Qdrant's genuinely
            # cpuset-bound `4-11`, on a run where both were given `4-11`.
            lines.append(f'<code>{html.escape(text)}</code> ({cpu_count(text)} cores) '
                         f'<span class="pill bad">main thread only</span> '
                         f'<span class="muted">captured before per-thread masks were '
                         f'read, so it does not describe where the workers ran</span>')
        else:
            for mask, n, cores in _mask_groups(text):
                lines.append(f'<code>{html.escape(mask)}</code> ({cores} core'
                             f'{"" if cores == 1 else "s"}) '
                             f'<span class="muted">observed on {n} '
                             f'thread{"" if n == 1 else "s"}</span>')
    if not want:
        lines.append('<span class="muted">the requested set was not recorded, so '
                     'whether both engines were given the same cores cannot be '
                     'checked from this run</span>')
    return "<br>".join(lines)


def provenance_table(runs: list[Run]) -> str:
    """What was measured, on what, by whom, and when.

    First section on the page because it is the first question an outside
    reader has, and because every number below is uninterpretable without it.
    A commit alone does not identify what ran, so the binary's own hash is
    beside it, and `dirty` is what turns a commit from an identity into an
    approximation.
    """
    def engine_ident(r: Run) -> str:
        m = r.meta or {}
        if "qdrant" in m:
            q = m["qdrant"]
            if q.get("binary"):
                # A native binary. `run.json` identifies it more precisely than
                # an image tag would -- commit, binary hash, dirty flag -- and
                # until 2026-08-28 none of that was shown; the page printed
                # "image unknown, digest unknown" instead, which reads as "we
                # do not know what ran" for the one kind of run where it is
                # known best. §8.9 pins the published comparison by digest and
                # a path cannot carry one, so the pill says so rather than
                # pretending the two are equivalent.
                commit = _fmt(q.get("commit"))
                if q.get("dirty"):
                    commit += "-dirty"
                code = f"<code>{commit}</code>"
                commit = f"commit {code}"
                pill = ('<span class="pill warn">native binary: pinned by commit, '
                        'not by image digest (§8.9)</span>')
                # A binary older than its checkout was not built from it, so
                # the commit is only where the tree stood when the run began.
                if q.get("binary_predates_commit"):
                    commit = f"commit unknown (the binary predates checkout {code})"
                    pill = ('<span class="pill warn">binary predates its checkout: '
                            'sha256 is the identity (§8.9)</span>')
                bits = [f"version {_fmt(q.get('version'))}",
                        f"binary <code>{_tilde(q.get('binary'))}</code>"
                        + (f" (cargo profile <code>{q['cargo_profile']}</code>)"
                           if q.get("cargo_profile") else ""),
                        f"sha256 <code>{_fmt(q.get('binary_sha256'))}</code> {commit}",
                        pill,
                        _network_path(q.get("network"))]
                return "<br>".join(bits)
            bits = [f"version {_fmt(q.get('version'))}",
                    f"image <code>{_fmt(q.get('image'))}</code>",
                    f"digest <code>{_fmt(q.get('digest'))}</code>",
                    _network_path(q.get("network"))]
            return "<br>".join(bits)
        sm = m.get("strawmann") or {}
        if not sm:
            # A run whose provenance was reconstructed says so, in its own
            # words, rather than being reported as merely absent: "we know the
            # dataset and not the binary" is a different and more useful state
            # than "we know nothing".
            note = m.get("note")
            return (f'<span class="muted">{note}</span>' if note else
                    '<span class="muted">unknown, this run predates run.json</span>')
        dirty = sm.get("dirty")
        # §9 quotes numbers from ReleaseFast only, so anything else is called
        # out rather than printed as a neutral fact. A run that predates the
        # recording says "not recorded" and gets the same warning: unknown and
        # wrong are both reasons not to trust the row.
        mode = sm.get("optimize")
        if mode == "ReleaseFast":
            build = f"<code>{mode}</code>"
        elif mode:
            build = f'<code>{mode}</code> <span class="pill bad">not ReleaseFast (§9)</span>'
        else:
            build = '<span class="pill bad">build mode not recorded</span>'
        isa = m.get("isa_build")
        tier = sm.get("dispatch_tier")
        isa_bits = f"{_fmt(isa)} build" + (f", dispatch <code>{tier}</code>" if tier else "")
        if sm.get("vnni") is not None:
            isa_bits += f", vnni {'on' if sm['vnni'] else 'off'}"
        bits = [f"commit <code>{_fmt(sm.get('commit'))}</code>"
                + ("" if dirty is False else
                   ' <span class="pill bad">tree dirty</span>' if dirty else ""),
                f"{build}, {isa_bits}",
                f"binary sha256 <code>{_fmt(sm.get('binary_sha256'))}</code>",
                f"built {_fmt(sm.get('binary_mtime'))} with zig {_fmt(sm.get('zig'))}"]
        # A run long enough to outlast a commit to the tree writes rows under
        # two commit labels while one binary serves all of them. The §8 sink
        # used to refuse that as "a subset re-run on a rebuilt binary", which
        # is a false diagnosis when the sha256 agrees; the true fact is worth
        # a line here rather than nothing at all.
        if drift := compare.commit_drift(r.label, r.by_id()):
            bits.append(f'<span class="muted">{drift}</span>')
        return "<br>".join(bits)

    def when(r: Run) -> str:
        # A folded label's rows carry pass 1's stamps; `aggregate.py` records
        # each pass's span, and the window is the whole run.
        spans = [p for p in ((r.meta or {}).get("passes") or [])
                 if isinstance(p, dict) and p.get("started")]
        if spans:
            first = min(p["started"] for p in spans)
            last = max((p.get("last_row") or p["started"]) for p in spans)
            return f"{first} to {last} ({len(spans)} passes)"
        stamps = sorted(x for x in (row.get("when") for row in r.rows) if x)
        if not stamps:
            return '<span class="muted">not recorded</span>'
        return stamps[0] if stamps[0] == stamps[-1] else f"{stamps[0]} to {stamps[-1]}"

    ds = next((r.meta.get("dataset") for r in runs if r.meta.get("dataset")), {}) or {}
    host = next((r.meta.get("host") for r in runs if r.meta.get("host")), {}) or {}

    shared_host = same_host(runs)
    rows = [("engine", [engine_ident(r) for r in runs]),
            ("measured", [when(r) for r in runs]),
            # Stated per run rather than once, because two runs having the same
            # profile is exactly the thing a reader must be able to check.
            ("profile", [f'<code>{profile_of(r)}</code>' for r in runs])]
    if not shared_host:
        # Two machines, or two states of one: the gate and the hash belong to
        # each run. On one machine they are said once, in the Host card.
        rows += [("§7.1 gate", [('<span class="pill ok">pass</span>' if r.gate_pass
                                 else '<span class="pill bad">FAIL</span>') for r in runs]),
                 ("environment hash", [f"<code>{r.env_hash}</code>" for r in runs])]
    rows += [
            ("load generator", [f"bfb {_fmt((r.meta or {}).get('bfb_pin'))}" for r in runs]),
            ("client", [_fmt((r.meta or {}).get('bfb_client')) for r in runs]),
            # How the engine was launched. strawmann's `--connections` decides
            # whether W4 can run at all, and a run that omits it fails with a
            # bare transport error that reads as an engine fault.
            ("launched as", [f"<code>{_tilde((r.meta or {}).get('engine_cmdline'), 'not recorded')}</code>"
                             for r in runs]),
            # The fairness fact, in the same units for both engines. It cannot
            # be read off `launched as`: strawmANN carries `--pin --cpus 4-11`
            # in its argv and the container carries `--cpuset-cpus` on the
            # `docker run` that is not in the engine's own command line, so the
            # table showed one engine pinned and the other apparently free on
            # the whole box. This is `Cpus_allowed_list`, read back from the
            # running process on both sides.
            ("cores the engine could use", [_affinity_cell(r) for r in runs])]

    def _engine_row(name: str, cells: list[str]) -> str:
        """One cell spanning both engines when the value is the same for both.

        §8.5 puts one client, one encoder and one decoder in front of both
        engines on purpose — "any difference observed is then necessarily
        server-side". Printing the load generator and the client twice, in two
        columns, made that shared-by-design fact look like a coincidence of two
        independent choices, and cost a column of width to say nothing. Spanned
        and labelled, the row states the design instead of repeating it.
        """
        if len(cells) > 1 and len(set(cells)) == 1:
            return (f'<tr><td class="desc">{name}</td>'
                    f'<td colspan="{len(cells)}">{cells[0]}'
                    f'<span class="muted"> · both engines</span></td></tr>')
        return (f'<tr><td class="desc">{name}</td>'
                + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    body = "".join(_engine_row(name, cells) for name, cells in rows)
    # The one place the run label is the information: it names the result
    # directory every number here was read from. `renamed` leaves it alone.
    names = display_names(runs)
    head = "<th></th>" + "".join(
        f"<th>{names[r.label]}"
        + (f'<br><code>{r.label}</code>' if names[r.label] != r.label else "")
        + "</th>" for r in runs)
    engines = (f'{KEEP_OPEN}<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
               f"<tbody>{body}</tbody></table></div>{KEEP_CLOSE}")

    shared = []
    if ds:
        # The count, not the hashes: four truncated sha256s and "+22 more"
        # identified nothing a reader could check, and `datasets.json` pins
        # every part in full.
        n_parts = max(ds.get("checksum_count") or 0, len(ds.get("checksums") or {}))
        pins = ""
        extra = (f'<div class="muted">{n_parts} file{"" if n_parts == 1 else "s"}, '
                 f"each pinned by sha256 in <code>datasets.json</code></div>"
                 if n_parts else "")
        shared.append(
            f'<div class="factcard"><h3>Dataset</h3>'
            f'<div><b>{_fmt(ds.get("name"))}</b>, {_fmt(ds.get("n"))} × '
            f'{_fmt(ds.get("dim"))}, {_fmt(ds.get("metric"))}, '
            f'{_fmt(ds.get("n_queries"))} held-out queries</div>'
            f'<div class="muted">ground truth: '
            f'{"recomputed by our fp64 oracle" if not ds.get("gt_shipped") else "shipped, and diffed against our fp64 recompute"}'
            f"</div>{pins}{extra}</div>")
    if host:
        bw = bandwidth_of(runs)
        gate = ""
        per_run = ""
        if shared_host:
            r0 = runs[0]
            gate = (' <span class="pill ok">§7.1 gate pass</span>' if r0.gate_pass
                    else ' <span class="pill bad">§7.1 gate FAIL</span>')
            def _scope(r) -> str:
                """Which rows the environment above actually describes.

                Silence here is the bug: the last invocation's gate was being
                read as the label's, so one late row could stamp an entire arm
                development-grade — or, worse in the other direction, a quiet
                final invocation could vouch for rows measured on a busy box.
                """
                ok, total, bad = gate_split(r)
                if not total or not bad:
                    return ""
                shown = ", ".join(bad[:4]) + ("…" if len(bad) > 4 else "")
                return (f' · <b>{ok} of {total} rows</b> were measured under a '
                        f'passing gate; {shown} {"was" if len(bad) == 1 else "were"} '
                        f'not. This line is the <em>last</em> invocation of '
                        f'<code>workloads.py</code> for this label, and an arm is '
                        f'several: the per-row stamp is what each figure carries.')

            per_run = "".join(
                f'<div class="muted"><b>{r.label}</b> run start: '
                f'environment hash <code>{r.env_hash}</code>'
                + "".join(f" · {c.text}" for c in r.moment_checks)
                + _scope(r) + "</div>"
                for r in runs)
        shared.append(
            f'<div class="factcard"><h3>Host{gate}</h3>'
            f'<div>{_fmt(host.get("cpu"))}</div>'
            f'<div class="muted">{_fmt(host.get("cores"))} logical cores · '
            f'{(host.get("memory_bytes") or 0) / 1e9:.1f} GB · '
            f'kernel {_fmt(host.get("kernel"))}</div>'
            f'<div>memory bandwidth: {bw["text"]}</div>'
            f'<div class="muted">{bw["source"]}</div>'
            f'{per_run}</div>')
    cards = f'<div class="factwrap">{"".join(shared)}</div>' if shared else ""
    return engines + cards


#: The storage and I/O rows, in the order they are shown, as
#: `(label, key, formatter, kind)`. `kind` separates end states from totals and
#: keeps the syscall counts visibly apart from the block-layer ones: they count
#: every descriptor, so on a search row they are the network.
#: Comparable rows first. `procstat`'s docstring: block-layer *bytes* are the
#: only fields both interfaces supply, so they are what a comparison reads
#: before anything that exists on one side only.
STORAGE_ROWS = [
    ("storage on disk", "storage_bytes", "bytes", "level"),
    ("peak RSS", "rss_peak_bytes", "bytes", "level"),
    # The split under it, because two engines holding the same total are not
    # holding the same thing: anonymous is memory the engine allocated, file is
    # page cache it mapped and the kernel may reclaim. For a RAM-resident
    # engine against a disk-backed one that distinction is the section's whole
    # subject, and peak RSS alone hides it.
    ("  of which anonymous", "rss_anon_bytes", "bytes", "level"),
    ("  of which file-backed", "rss_file_bytes", "bytes", "level"),
    ("disk read bytes", "disk_read_bytes", "bytes", "total"),
    ("disk write bytes", "disk_write_bytes", "bytes", "total"),
    ("disk read ops", "disk_read_ops", "count", "total"),
    ("disk write ops", "disk_write_ops", "count", "total"),
    ("syscall reads (all fds)", "syscall_reads", "count", "syscall"),
    ("syscall writes (all fds)", "syscall_writes", "count", "syscall"),
]

#: Which fields an interface simply does not carry. A cell missing for this
#: reason is not a failed measurement, and printing both as "unknown" made the
#: section read as though the two engines shared no numbers at all.
NOT_IN_SOURCE = {
    "proc": ("disk_read_ops", "disk_write_ops"),
    "cgroup": ("syscall_reads", "syscall_writes"),
}


#: The rows the summary's memory-and-disk card keeps, and what it calls them.
#: The anonymous/file split, the op counts and the syscall counts are in the
#: full table.
COMPACT_STORAGE = {"peak RSS": "peak memory (RSS)", "storage on disk": "storage on disk",
                   "disk read bytes": "read from disk", "disk write bytes": "written to disk"}


def storage_table(runs: list[Run], compact: bool = False) -> str:
    """What each engine stored, and what it did to the disk to serve the run.

    `compact` is the four rows a reader compares, with one sentence under
    them; the full table and its method note are the appendix's.

    Empty, and therefore absent from the report, for runs measured before these
    fields existed. An absent section reads as "not measured"; a section full
    of zeros would read as "measured, and nothing happened", which is a
    different and much stronger claim.
    """
    def cell(run: Run, key: str, fmt: str, kind: str) -> str:
        # A level is read from the settled rows only: the run's last row is
        # W11, and an end state sampled while Qdrant's optimiser rewrites
        # segments is that rewrite rather than the corpus (findings 53,
        # `procstat.settled_rows`). A peak keeps every row, a peak being a peak.
        src = run.rows if (kind != "level" or key.startswith("rss_")) \
            else procstat.settled_rows(run.rows)
        vals = [r.get(key) for r in src if r.get(key) is not None]
        if not vals:
            # Runs measured before `procstat.store_bytes_for` recorded this can
            # still be answered from provenance they did keep: an engine whose
            # recorded command line has no `--data-dir` had nowhere to write, so
            # its store is 0 rather than unmeasured. Derived, and only from what
            # the run wrote down.
            if key == "storage_bytes":
                argv = (run.meta or {}).get("engine_cmdline") or ""
                if argv and "--data-dir" not in argv and "qdrant" not in argv:
                    return '<td class="num">0<span class="sub">no --data-dir</span></td>'
            src = next((r.get("io_source") for r in run.rows if r.get("io_source")), "")
            for name, missing in NOT_IN_SOURCE.items():
                if name in (src or "") and key in missing:
                    return f'<td class="num muted">n/a via {name}</td>'
            return '<td class="num muted">unknown</td>'
        # A level is the end state; a total is summed over the rows.
        # The resident levels take the largest across the rows, like the peak
        # they sit under; every other level is the end state.
        v = (max(vals) if key.startswith("rss_") else vals[-1]) if kind == "level" else sum(vals)
        shown = procstat.human_bytes(v) if fmt == "bytes" else procstat.human_count(v)
        # Named rather than silently dropped: the reader is owed the row the
        # level came from when it is not the run's last one.
        if (kind == "level" and not key.startswith("rss_") and len(src) != len(run.rows)
                and not compact):  # the compact card's note says it once
            skipped = [r.get("id") for r in run.rows if procstat.is_mutating(r)]
            return (f'<td class="num">{shown}'
                    f'<span class="sub">before {", ".join(str(x) for x in skipped)}</span></td>')
        return f'<td class="num">{shown}</td>'

    body = []
    rows = ([(COMPACT_STORAGE[lab], k, f, kd) for lab, k, f, kd in STORAGE_ROWS
             if lab in COMPACT_STORAGE] if compact else STORAGE_ROWS)
    for label, key, fmt, kind in rows:
        cells = "".join(cell(run, key, fmt, kind) for run in runs)
        if 'class="num"' not in cells:
            continue  # nothing measured this row on any engine
        klass = ' class="syscall"' if kind == "syscall" else ""
        body.append(f'<tr{klass}><td class="desc">{label}</td>{cells}</tr>')
    if not body:
        return ('<p class="note">Not measured for these runs. The kernel\'s I/O '
                'counters live with the process and cannot be recovered '
                'afterwards, so this is blank rather than zero: a run taken '
                'before the harness read them has no storage figures, which is '
                'not the same as an engine that did no I/O.</p>')

    refusal = placement_refusal(runs)
    warn = (f'<div class="banner soft"><b>Not compared.</b> {refusal}</div>'
            if refusal else "")
    sources = " / ".join(
        f"{run.label}: " + (next((r.get("io_source") for r in run.rows if r.get("io_source")),
                                 "unmeasured"))
        for run in runs)
    head = "<th></th>" + "".join(f'<th class="num">{r.label}</th>' for r in runs)
    table = (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table></div>')
    if compact:
        return (warn + table + '<p class="note">Peak memory is the largest across the '
                'run; disk figures are totals over every row. Storage on disk is read '
                'before the concurrent-write rows (W11), when an engine rewriting '
                'segments would be caught mid-rewrite.</p>')
    return (warn + table +
            f'<p class="note">measured via {sources}. <code>proc</code> supplies '
            f'syscall counts and block-layer bytes, <code>cgroup</code> supplies '
            f'block-layer operations and bytes, so a row one interface does not carry '
            f'reads <em>n/a via …</em>. <em>unknown</em> means it was not measured, '
            f'and <b>0</b> means it was: an engine started with no <code>--data-dir</code> '
            f'has no store, which is the row this section exists for. '
            f'<b>storage on disk</b> is the level before the rows with a concurrent '
            f'writer: during those an engine that rewrites segments is caught '
            f'mid-rewrite, and the same row has read 3.47 and 10.11 GiB on two runs '
            f'of one binary. <b>peak RSS</b> does include them, being a peak.</p>')


#: bfb's `wait_index` polling floor, restated here because the report is read
#: without the harness. See `workloads.WAIT_INDEX_FLOOR_S`.
WAIT_INDEX_FLOOR_S = 3.0


def _duration_cell(row) -> tuple[str, bool]:
    """The best duration a row can offer, and whether it sits on the floor.

    Three sources in descending order of what they actually measure: bfb's
    Time-to-Green, bfb's upload phase, and the harness's wall clock around the
    whole invocation. The last includes process startup and collection
    creation, so a row reported from it is labelled `wall` rather than passed
    off as a phase timing.
    """
    def num(k):
        v = row.get(k) if hasattr(row, "get") else None
        return float(v) if isinstance(v, (int, float)) and pd.notna(v) else None

    green, upload = num("index_wait_s"), num("upload_s")
    floored = bool(row.get("time_to_green_floored")) if hasattr(row, "get") else False
    if green is not None:
        return f"{green:,.2f} s{' †' if floored else ''}", floored
    if upload is not None:
        return f"{upload:,.2f} s", False
    wall = num("wall_s")
    if wall is not None:
        return f'{wall:,.1f} s <span class="muted">wall</span>', False
    return f'{row["seconds"]:,.0f} s <span class="muted">wall</span>', False


#: The pair the section's own advice is about: push the points, then wait for
#: the index. `W1` is the upload with no index wait and `W2` is the wait, and
#: they are the two rows a reader is told to add up.
INGEST_SUM = ("W1", "W2")


def _seconds_of(row) -> float | None:
    """The duration a row contributes to the ingest total, in seconds."""
    for k in ("index_wait_s", "upload_s", "wall_s", "seconds"):
        v = row.get(k) if hasattr(row, "get") else None
        if isinstance(v, (int, float)) and pd.notna(v):
            return float(v)
    return None


def _ingest_total_row(runs: list[Run], df: pd.DataFrame) -> str:
    """W1 + W2, because the section tells the reader to compare the sum.

    "Compare the sum, not the upload line" was the lede's advice and the table
    printed neither the sum nor a ratio, so the one comparison the section
    exists to support was left as arithmetic for the reader — over two rows
    that mean different things on the two engines, which is exactly why the
    advice is there.

    No verdict beside it, deliberately: the noise floor is built from search
    passes over an already-built graph and has no spread for any ingest row, so
    there is nothing to judge this difference against. The note says that
    rather than letting a bare ratio imply it was checked.
    """
    totals = []
    for run in runs:
        by = run.by_id()
        parts = [_seconds_of(by[w]) for w in INGEST_SUM if w in by]
        totals.append(sum(parts) if parts and all(p is not None for p in parts) else None)
    if len(totals) < 2 or any(t is None for t in totals):
        return ""
    cells = "".join(f'<td class="num"><b>{t:,.1f} s</b></td>' for t in totals)
    ratio = totals[1] / totals[0] if totals[0] else None
    tail = (f'<td class="num {"up" if ratio >= 1 else "down"}" '
            f'title="no noise floor for these rows, so no verdict">{ratio:,.2f}x</td>'
            if ratio else "<td></td>")
    return (f'<tr><td class="wid">{" + ".join(INGEST_SUM)}</td>'
            f'<td class="desc"><b>upload and index, together</b></td>{cells}{tail}</tr>')


def ingest_table(runs: list[Run], df: pd.DataFrame) -> str:
    rows, any_floored = [], False
    for wid in UPLOAD_ROWS:
        sub = df[df["id"] == wid]
        if sub.empty:
            continue
        cells = [f'<td class="wid">{wid}</td>',
                 f'<td class="desc">{DESCRIPTIONS.get(wid.split("-")[0], "upload")}</td>']
        for run in runs:
            r = sub[sub["engine"] == run.label]
            if len(r):
                text, floored = _duration_cell(r.iloc[0])
                any_floored |= floored
                cells.append(f'<td class="num">{text}</td>')
            else:
                cells.append('<td class="num muted">-</td>')
        if len(runs) > 1:
            cells.append("<td></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    if not rows:
        return ""
    total = _ingest_total_row(runs, df) if len(runs) > 1 else ""
    if total:
        rows.append(total)
    head = ("<th>workload</th><th></th>" + "".join(f"<th>{r.label}</th>" for r in runs)
            + ("<th></th>" if len(runs) > 1 else ""))
    note = ""
    if any_floored:
        note = (
            f'<p class="note">† at the load generator\'s polling floor. bfb decides a '
            f"collection is Green by polling once a second and requiring three "
            f"consecutive Green replies, having slept a second before the first poll, "
            f"so no Time-to-Green it can report is below {WAIT_INDEX_FLOOR_S:.0f} s "
            f"regardless of how fast the build was. A marked cell is an upper bound on "
            f"the build and a measurement of the polling loop. The bias is a constant "
            f"added to both engines, so the difference between them survives it and "
            f"the ratio does not \u2014 and it is the faster engine the ratio "
            f"understates.</p>")
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{note}')


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------









































def build_figures(runs: list[Run], df: pd.DataFrame) -> list[dict]:
    out = [chart_throughput(runs, df), chart_ef_sweep(runs, df),
           chart_frontier(runs),
           chart_latency_percentiles(runs), chart_server_vs_client(runs),
           chart_latency_distribution(runs), chart_build(runs, df),
           chart_runqueue(runs, df), chart_quantization_recall(runs),
           chart_memory(runs, df), chart_load(runs, df),
           chart_io_pressure(runs, df), chart_memory_pressure(runs, df),
           chart_query_cost(runs, df), chart_dram_per_query(runs, df),
           chart_tlb_per_query(runs, df), chart_ipc(runs, df),
           chart_branch_mpki(runs, df)]
    return [c for c in out if c]


CSS = """
:root{--bg:#fbfbfa;--card:#fff;--ink:#1a1a19;--muted:#6b6b66;--line:#e6e4df;
--accent:#F7A41D;--ok:#2EA44F;--bad:#DC244B;--code:#f4f2ee;--link:#A85A00}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#131312;--card:#1b1b1a;--ink:#e9e7e2;--muted:#9a978f;--line:#2c2b28;--code:#232220;
--link:#FFB547}}
:root[data-theme=dark]{--bg:#131312;--card:#1b1b1a;--ink:#e9e7e2;--muted:#9a978f;
--line:#2c2b28;--code:#232220;--link:#FFB547}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
/* The browser's own #00e link blue, on a page that otherwise has no blue, and
   about 2:1 on the dark card. */
a{color:var(--link);text-underline-offset:2px}
/* The page is mostly tables and charts, and 960px left half a wide screen
   empty while twelve-column latency tables scrolled inside it. Prose is the
   only thing that suffers from a long line, so the container is wide and the
   *measure* is constrained instead — text stays readable, data gets the room. */
.wrap{max-width:min(1600px,95vw);margin:0 auto;padding:44px 26px 90px}
.wrap p{max-width:86ch}
/* The subject of the measurement, before any number. Grid rather than flex so
   the labels line up in a column on a narrow screen instead of wrapping into
   an unreadable ribbon. */
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
gap:1px;background:var(--line);border:1px solid var(--line);border-radius:9px;
overflow:hidden;margin:18px 0 6px}
.glance>div{background:var(--card);padding:11px 14px}
.glance dt{font:600 10.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0}
.glance dd{margin:2px 0 0;font-size:14px}
.glance dd .sub{color:var(--muted);font-size:12.5px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h1 em{font-style:normal;color:var(--accent)}
h2{font-size:19px;margin:46px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:14px;margin:0 0 10px;font-weight:600}
.lede{color:var(--muted);margin:0 0 20px}
.bandwidth{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 18px;margin:0 0 16px;font-size:1.05em}
.bandwidth .muted{font-size:.85em;margin-top:4px}
.gatecard .moment{margin-top:10px;padding-top:8px;border-top:1px dashed var(--line)}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 8px}
.pill{font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace;padding:4px 9px;
border-radius:99px;border:1px solid var(--line);background:var(--card);color:var(--muted);
/* "closed loop" broke across two lines in the latency table, where twelve
   columns squeeze the description cell. A pill is one label. */
white-space:nowrap;display:inline-block}
.pill.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,transparent)}
.pill.bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 40%,transparent)}
.pill.warn{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 40%,transparent)}
.banner{background:color-mix(in srgb,var(--bad) 9%,var(--card));
border:1px solid color-mix(in srgb,var(--bad) 30%,transparent);
border-radius:9px;padding:13px 15px;margin:18px 0;font-size:13.5px}
.banner.soft{background:color-mix(in srgb,var(--accent) 10%,var(--card));
border-color:color-mix(in srgb,var(--accent) 38%,transparent)}
.banner b{color:var(--bad)}.banner.soft b{color:var(--accent)}
/* A passing verdict was drawn in the amber the page uses for a caveat, so the
   good news and the warnings below it looked alike. */
.banner.ok{background:color-mix(in srgb,var(--ok) 8%,var(--card));
border-color:color-mix(in srgb,var(--ok) 35%,transparent)}
.banner.ok b{color:var(--ok)}
.wl{margin:0 0 22px;padding:0 0 4px;border-bottom:1px solid var(--line)}.wl:last-child{border-bottom:0}.wl-index{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 18px}.wl-index a{display:inline-flex;gap:5px;align-items:baseline;text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:3px 8px;font-size:11.5px;font-family:ui-monospace,monospace;color:var(--ink)}.wl-index a:hover{border-color:var(--accent)}.wl-ix-r{font-weight:700}.wl-cfg{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted);margin:-4px 0 8px}.wl h3{margin:0 0 8px;font-size:14px;display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}.wl-id{font-family:ui-monospace,monospace;font-weight:700}.wl-desc{color:var(--muted);font-weight:400}.wl-ratio{font-family:ui-monospace,monospace;font-weight:700;margin-left:auto}.wl table td.desc{color:var(--muted);white-space:nowrap}.wl table{font-size:13px}.wl .notes{margin-top:6px}.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px;background:var(--card)}
td.desc{white-space:nowrap}
/* In the throughput table descriptions and verdicts wrap. With both held to
   one line its widest row (W11-steady's description, "clears the ±17%
   measured noise floor") outgrew the page: the ratio column was clipped at
   the card's edge and the notes column shrank to one word wide, every row
   hundreds of pixels tall. */
table.throughput td.desc{white-space:normal;min-width:22ch}
table.throughput td.notes{min-width:34ch}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{text-align:left;font-weight:600;color:var(--muted);font-size:11.5px;
text-transform:uppercase;letter-spacing:.05em;padding:11px 13px;
border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
/* The hardware and stall tables scroll sideways, and past the first screen of
   columns nothing said which row a number was on. The row id stays put. */
.tablewrap thead tr:first-child th:first-child,.tablewrap td:first-child{position:sticky;
left:0;z-index:1;background:var(--card)}
/* Engine-grouped tables: the label once over its columns, and a rule where
   one engine's group ends and the next begins. */
table.grouped th.eng{text-align:center;color:var(--ink);border-left:1px solid var(--line);
border-bottom:1px solid var(--line)}
table.grouped th[rowspan]{vertical-align:bottom}
table.grouped .grp{border-left:1px solid var(--line)}
/* A collection field on which the engines disagree. */
td.differs{color:var(--ink);font-weight:600;
background:color-mix(in srgb,var(--accent) 14%,var(--card))}
tbody tr:hover td{background:color-mix(in srgb,var(--accent) 7%,var(--card))}
.wid{font:600 12.5px ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap}
.desc{color:var(--muted)}
.num{text-align:right;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap}
.num .sub{display:block;font-size:10.5px;color:var(--muted);margin-top:2px}
table.throughput .num .sub{white-space:normal;min-width:16ch;max-width:24ch;margin-left:auto}

.up{color:var(--ok);font-weight:600}.down{color:var(--bad);font-weight:600}
.muted,.na{color:var(--muted)}.bad{color:var(--bad)}
.notes{font-size:11.5px}
.caveat{display:inline-block;color:var(--muted);
background:color-mix(in srgb,var(--muted) 12%,transparent);
border-radius:4px;padding:1px 6px;margin:1px 2px 1px 0}
.warn{display:inline-block;color:var(--bad);
background:color-mix(in srgb,var(--bad) 10%,transparent);
border-radius:4px;padding:1px 6px;margin:1px 2px 1px 0}
.gatewrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.gatecard{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:15px}
.gatecard h3{display:flex;align-items:center;gap:9px;justify-content:space-between}
.hash{font:11px ui-monospace,monospace;color:var(--muted);margin-bottom:9px}
.chk{font:11.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;padding:2.5px 0;
display:flex;gap:8px}
.chk span{font-weight:700;flex:none;width:32px}
.chk.pass span{color:var(--ok)}.chk.fail span{color:var(--bad)}
.chk.cont{color:var(--muted);padding-left:40px;display:block}
/* `.card` is the same frame. The ingest, scheduler, stall and hardware charts
   used it and nothing defined it, so they sat unframed on the page while the
   throughput, recall and latency charts had a card. */
figure.chart,.card{margin:0 0 26px;background:var(--card);border:1px solid var(--line);
border-radius:9px;padding:16px 17px 13px}
.note{font-size:11.5px;color:var(--muted);margin-top:9px;line-height:1.5}
/* A long page with no way through it is a page nobody reads past the first
   screen. The sections were already anchored; only the links were missing. */
.summary{border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:9px;background:var(--card);padding:16px 20px;margin:24px 0 4px}
.summary h2{margin:0 0 10px;border:0;padding:0;font-size:13px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted)}
.summary p{margin:0 0 9px;font-size:14px}
/* Sticky: sixteen sections below it and the links were only on the first
   screen. Wrapped where there is room for two lines of it; one line scrolling
   sideways on a phone, where two would be five. */
.toc{display:flex;flex-wrap:wrap;gap:6px;margin:26px -26px 4px;padding:8px 26px;
position:sticky;top:0;z-index:5;background:var(--bg);
border-bottom:1px solid var(--line)}
.toc a{flex:none}
h2,h3,#bandwidth{scroll-margin-top:96px}
@media(max-width:900px){.toc{flex-wrap:nowrap;overflow-x:auto}
h2,h3,#bandwidth{scroll-margin-top:60px}}
.toc a{font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace;padding:5px 10px;
border-radius:7px;border:1px solid var(--line);background:var(--card);
color:var(--muted);text-decoration:none}
.toc a:hover{color:var(--ink);border-color:var(--muted)}
.factwrap{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.factcard{flex:1 1 300px;border:1px solid var(--line);border-radius:9px;
background:var(--card);padding:13px 15px;font-size:12.5px;line-height:1.7}
.factcard h3{margin:0 0 6px;font-size:11px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted)}
.factcard code{font-size:11px}
th code{text-transform:none;letter-spacing:0}
.muted{color:var(--muted)}
/* Syscall counts are a different axis from the block-layer rows above them:
   they include sockets. The rule is the visual form of that separation. */
tr.syscall td{border-top:2px solid var(--line)}
tr.syscall td.desc{color:var(--muted)}
.prose{font-size:13.5px;color:var(--ink)}
.prose p{margin:0 0 12px}
/* The glossary. A two-column grid so a term and its sentence sit on one line
   at reading width, and stack rather than crush on a phone. */
.terms{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;
margin:0 0 18px;padding:14px 16px;border:1px solid var(--line);border-radius:9px;
background:var(--card);font-size:12.5px}
.terms dt{font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:12px;white-space:nowrap}
.terms dd{margin:0;color:var(--muted)}
@media(max-width:560px){.terms{grid-template-columns:1fr;gap:2px}
.terms dd{margin:0 0 8px}}
code{background:var(--code);border-radius:4px;padding:1px 5px;
font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);
color:var(--muted);font-size:12px}
.js-plotly-plot{width:100%!important}
/* The summary's figures. One tile per question, the lower (better) figure of
   a pair in the engine-neutral "up" colour. */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;
margin:16px 0 8px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px 15px}
.kpi .k{font:600 10.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.06em;
text-transform:uppercase;color:var(--muted)}
.kpi .h{font:700 24px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;margin:4px 0 2px}
.kpi .v{display:flex;justify-content:space-between;gap:8px;font:13.5px/1.7 ui-monospace,
SFMono-Regular,Menlo,monospace}
.kpi .v span:first-child{color:var(--muted);font-family:ui-sans-serif,system-ui,sans-serif}
.kpi .v.better span:last-child{color:var(--ok);font-weight:700}
.kpi .s{font-size:11.5px;color:var(--muted);margin-top:4px}
/* Everything past the summary's level, one disclosure per section. */
details.sec{border:1px solid var(--line);border-radius:9px;background:var(--card);
margin:0 0 10px}
details.sec>summary{cursor:pointer;padding:11px 15px;font-weight:600;font-size:14px}
details.sec>summary .muted{font-weight:400;font-size:12.5px}
details.sec[open]>summary{border-bottom:1px solid var(--line)}
details.sec>.body{padding:14px 15px 4px}
details.sec>.body>h2:first-child{margin-top:0}
details.more{margin:8px 0 0}
details.more>summary{cursor:pointer;font-size:11.5px;color:var(--muted)}
details.glossary{margin:10px 0 0}
details.glossary>summary{cursor:pointer;font-size:12.5px;color:var(--muted)}
.slower{margin:0;padding-left:20px;font-size:14px}
.wid .desc{font:400 13px ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
margin-left:4px}
[title]{text-decoration-style:dotted}
th[title],td[title]{cursor:help}
"""



#: Re-theme every chart to match the page.
#:
#: Plotly cannot read CSS variables, so a chart cannot inherit the theme the way
#: the rest of the page does. Without this the axis text, tick labels and legend
#: keep their light-theme colour on a dark card, which measures about 1.3:1 and
#: is simply unreadable. Runs on load, and again whenever the OS theme flips or
#: something sets data-theme, so a chart never disagrees with the page around it.
THEME_JS = """
(function () {
  var LIGHT = {ink:'#1a1a19', muted:'#6b6b66', grid:'#e6e4df', line:'#d5d2cb',
               paper:'#ffffff', series:{strawmann:'#C97A00', qdrant:'#8E0A2E'}};
  var DARK  = {ink:'#e9e7e2', muted:'#9a978f', grid:'#2c2b28', line:'#3a3835',
               paper:'#1b1b1a', series:{strawmann:'#FFC65C', qdrant:'#E8446A'}};

  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function apply() {
    if (!window.Plotly) return;
    var c = isDark() ? DARK : LIGHT;
    var axis = {gridcolor: c.grid, linecolor: c.line, zerolinecolor: c.line,
                tickfont: {color: c.muted}, title: {font: {color: c.muted}}};
    document.querySelectorAll('.plotly-graph-div').forEach(function (d) {
      Plotly.relayout(d, {
        'font.color': c.ink,
        'legend.font.color': c.ink,
        'xaxis.gridcolor': axis.gridcolor, 'xaxis.linecolor': axis.linecolor,
        'xaxis.zerolinecolor': axis.zerolinecolor,
        'xaxis.tickfont.color': c.muted, 'xaxis.title.font.color': c.muted,
        'yaxis.gridcolor': axis.gridcolor, 'yaxis.linecolor': axis.linecolor,
        'yaxis.zerolinecolor': axis.zerolinecolor,
        'yaxis.tickfont.color': c.muted, 'yaxis.title.font.color': c.muted,
        'hoverlabel.bgcolor': c.paper, 'hoverlabel.bordercolor': c.line,
        'hoverlabel.font.color': c.ink
      });
      // Series colours are per-trace, so they need restyle rather than relayout.
      var data = d.data || [];
      for (var i = 0; i < data.length; i++) {
        // Only traces tagged by `series_meta`. Matching on the trace name
        // also matched the ambient-load chart, whose colours mean clean and
        // contaminated rather than which engine.
        var name = (data[i].meta || {}).series;
        if (!name || !c.series[name]) continue;
        Plotly.restyle(d, {'marker.color': c.series[name], 'line.color': c.series[name]}, [i]);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply);
  } else {
    apply();
  }
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(apply);
  }
  new MutationObserver(apply).observe(document.documentElement,
      {attributes: true, attributeFilter: ['data-theme']});
  // A chart drawn inside a closed <details> is laid out at zero width, so it
  // is resized when its section opens. `toggle` does not bubble; capture does.
  document.addEventListener('toggle', function (e) {
    if (!window.Plotly || !e.target.open) return;
    e.target.querySelectorAll('.plotly-graph-div').forEach(function (d) {
      Plotly.Plots.resize(d);
    });
  }, true);
  // A link into the appendix lands inside a closed <details>, which would
  // scroll to nothing visible; open every disclosure around the target.
  function reveal() {
    var el = location.hash && document.getElementById(location.hash.slice(1));
    for (; el; el = el.parentElement) { if (el.tagName === 'DETAILS') el.open = true; }
  }
  window.addEventListener('hashchange', reveal);
  reveal();
})();
"""




@dataclass(frozen=True)
class Tables:
    """The rendered tables, as one value.

    They were eight of the twenty-three keyword arguments `build` passed to the
    template: a view model that existed only as an argument list, where adding
    a section meant touching the call, the signature and the template with
    nothing tying the three together.
    """

    throughput: str
    ingest: str
    storage: str
    storage_rows: str
    scheduler: str
    stalls: str
    hardware: str
    sharing: str
    workloads: str
    collections: str
    provenance: str
    latency: str
    recall: str
    matched_recall: str
    #: The summary-level versions, shown by default; the full tables above
    #: are the appendix's.
    throughput_compact: str
    latency_compact: str
    resources: str


TEMPLATE = "report.html.j2"


def template_env() -> Environment:
    """The Jinja environment the page is rendered through.

    A function so the escaping rule below is checkable without rendering a
    whole report: it was wrong for as long as the template has existed, and
    nothing on the page said so.
    """
    # `select_autoescape` matches on the *suffix* of the template name, and the
    # template is `report.html.j2` — which ends in `.j2`, not `.html`. So
    # `["html"]` turned autoescaping off for the only template there is, and
    # every value the page interpolated went in raw.
    #
    # Not theoretical. §8's unlicensed banner names the file to produce as
    # `bench/results/<label>/conformance.json`; the browser parsed `<label>` as
    # an element, rendered nothing for it, and wrapped the rest of the document
    # in an unclosed `<label>`. The page told the reader to write a path that
    # does not contain the placeholder it is supposed to substitute. Row ids,
    # engine labels and the `ps` output behind `foreign` reach the page the
    # same way.
    #
    # Everything the report builds *as* HTML — the tables, the figures, their
    # notes — is marked `|safe` at its interpolation, so this escapes prose and
    # leaves markup alone. Verified by rendering four result pairs both ways
    # and diffing.
    return Environment(loader=FileSystemLoader(HERE / "templates"),
                       autoescape=select_autoescape(["html", "j2"]),
                       trim_blocks=True, lstrip_blocks=True)


def build(runs: list[Run], title: str) -> str:
    df = frame(runs)
    env = template_env()
    # Table and chart HTML are pre-built; the template marks them `|safe`.
    tpl = env.get_template(TEMPLATE)
    hashes = sorted({r.env_hash for r in runs})
    cpu = next((c.text for r in runs for c in r.checks if "logical)" in c.text),
               "unknown CPU")
    shared_host = same_host(runs)
    bandwidth = bandwidth_of(runs)
    summ = summary(runs, df)

    body = tpl.render(
        title=title,
        glance=glance_of(runs),
        # Surfaced at the top as well as in Storage. Residency can move
        # throughput, and the throughput ratio is the first thing on the page:
        # a caveat a reader meets after forming their opinion is a footnote,
        # not a caveat.
        placement_split=[f"{r.label} {p}" for r, p in zip(runs, placements_of(runs))
                         if p] if len(set(placements_of(runs))) > 1 else None,
        css=CSS,
        runs=runs,
        cpu=cpu,
        stamp=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        bfb_commit=bfb_commit(runs),
        ungated=[r.label for r in runs if not r.gate_pass],
        hashes=hashes,
        # One machine whose busy-process list differed at the two run starts
        # is one environment with two moments, not two environments; the
        # per-run lines in the Host card say what differed.
        hash_mismatch=len(hashes) > 1 and not shared_host,
        hash_moment_only=len(hashes) > 1 and shared_host,
        # Which experiment these numbers are, and the one comparison that is
        # never legitimate: an isolated run against an as-deployed one measures
        # the scheduler, not the engines. Named explicitly because "the hashes
        # differ" is true but does not tell a reader *what* differed.
        profiles=profiles_of(runs),
        profile_mismatch=len(profiles_of(runs)) > 1,
        segment_policies=segment_policies_of(runs),
        segment_policy_mismatch=len(segment_policies_of(runs)) > 1,
        segments_requested=segments_requested(runs),
        # The one refusal that repeats: nine rows, one cause. Said in the
        # section lede when any row carries the marker, not on each row. Read
        # off `compare.Row`, because the marker is compare's and not the
        # harness row's own note.
        rescore_cause=(compare.RESCORE_CAUSE if len(runs) > 1 and any(
            "rescore pools differ" in n
            for r in compare.joined(runs[0].label, runs[1].label,
                                    runs[0].by_id(), runs[1].by_id())
            for n in r.notes) else None),
        # The measured segment counts, so the prose can quote what these
        # engines held rather than the cap they might have held.
        segments=segments_of(runs),
        ef_note=segment_note(runs),
        # §7.2(5): A/B/A/B, not A-then-B. Checked from the row stamps rather
        # than assumed, and stated where a reader forms their opinion.
        interleave=interleaving(runs),
        # How many passes each arm folded, when every arm folded the same
        # number. `None` for a single-pass run and for a mismatched pair, which
        # the page then describes as unrepeated rather than guessing a count.
        reps=(lambda ns: ns.pop() if len(ns) == 1 else None)(
            {(r.meta or {}).get("reps") for r in runs}
            if all(isinstance((r.meta or {}).get("reps"), int) for r in runs) else {None}),
        shared_host=shared_host,
        bandwidth=bandwidth,
        n_dirty=int((df["foreign"] != "").sum()),
        tables=Tables(
            throughput=throughput_table(runs, df),
            ingest=ingest_table(runs, df),
            storage=storage_table(runs),
            storage_rows=storage_rows_table(runs),
            scheduler=scheduler_rows_table(runs),
            stalls=stalls_rows_table(runs),
            hardware=hardware_rows_table(runs),
            sharing=sharing_rows_table(runs),
            workloads=workload_sections(runs, df),
            collections=collection_table(runs),
            provenance=provenance_table(runs),
            latency=latency_table(runs),
            recall=recall_table(runs),
            matched_recall=matched_recall_table(runs),
            throughput_compact=compact_throughput_table(runs, df),
            latency_compact=compact_latency_table(runs),
            resources=storage_table(runs, compact=True),
        ),
        licence=licence_of(runs),
        summary=summ,
        kpis=kpis(runs, df, summ),
        slower=grouped_losses(summ["losses"]),
        names=display_names(runs),
        noise=load_noise(),
        noise_prov=noise_provenance(runs),
        figures=build_figures(runs, df),
        # Whose stall time the pressure columns are, and which arms carried a
        # hardware sidecar at all. Both are properties of how the run was
        # taken, and both decide whether the columns beside them may be read
        # across the two engines.
        psi_scopes=psi_scopes(runs),
        perf_arms=perf_arms(runs),
        perf_mismatch=len(set(perf_arms(runs).values())) > 1,
        perf_refusals=perf_refusals(runs),
        perf_crosscheck=perf_crosscheck(runs),
    )
    # Engine names in place of run labels, everywhere but the regions that
    # keep them (`renamed`). Before the scripts go in: plotly's source is not
    # the page's text.
    body = renamed(body, display_names(runs))
    # Plotly's JS is embedded rather than fetched: a report mailed to someone
    # has to render in six months, when a CDN URL has moved.
    return body.replace(
        "</head>",
        f"<script>{get_plotlyjs()}</script></head>", 1
    ).replace("</body>", f"<script>{THEME_JS}</script></body>", 1)




def run_stamp(runs: list[Run]) -> str:
    """When these runs were measured, `YYYY-MM-DD-HHMM`, for the file name.

    The newest arm's `started` wins. For a single-pass pair that is the second
    arm's start, so a comparison straddling midnight is filed under the later
    day. For a folded label `started` is its first pass, so a three-pass night
    is filed under the start of the last engine's first pass (0925's page is
    `-2026-09-24-2152`, the Qdrant arm's pass 1, in UTC), not the day it
    finished. Kept that way: the archived pages are named by it.

    The stamp is the *run's*, never today's: keying on render time would file a
    re-render of August's numbers under the moment it was re-rendered, and spray a
    fresh copy on every render. A run predating `started` falls back to
    `rows.json`'s mtime, read as UTC so an unstamped arm and a stamped one are the
    same clock.

    The minute is here because the day was not enough: a settled SIFT
    pair was measured over the same labels as that morning's unsettled pair, and
    the second render would have replaced the first silently.
    """
    stamps = []
    for r in runs:
        started = (r.meta or {}).get("started")
        if isinstance(started, str) and len(started) >= 16:
            stamps.append(f"{started[:10]}-{started[11:13]}{started[14:16]}")
        else:
            mtime = (ROOT / "bench/results" / r.label / "rows.json").stat().st_mtime
            stamps.append(time.strftime("%Y-%m-%d-%H%M", time.gmtime(mtime)))
    return max(stamps)


def default_out(runs: list[Run]) -> Path:
    """Where the page goes when the caller does not name a path.

    Keyed by the dataset, the labels *and* the day the runs were measured.
    `report.html` was keyed by none of them, so every combination of engine,
    arm and corpus resolved to one path and the last render won. That is the
    mistake `recall.json` made one directory down, where a dbpedia sweep
    replaced a SIFT one under a shared name and every reader joined whichever
    had finished last; it cost a real afternoon, and `recall_path` is keyed by
    (label, dataset, collection, limit) because of it.

    The date is the last key added, and for the same reason one directory up:
    the labels are *reused*. `fullrun.py` defaults to `strawmann`/`qdrant`
    every night, so a week of comparisons of the same corpus were a week of
    renders to one path, each silently replacing the last — and a page open in
    a tab, or linked from `docs/findings.md`, said nothing about which night
    it came from. The stamp goes last so that every render of one comparison
    still sorts together, dated copies adjacent, and it carries the minute
    because one day holds more than one run of one pair -- see `run_stamp`.
    """
    labels = "-vs-".join(r.label for r in runs)
    stem = f"report-{datasets_of(runs)[0]}-{labels}-{run_stamp(runs)}"
    return ROOT / "bench/results" / f"{stem}.html"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build an HTML benchmark report.")
    ap.add_argument("labels", nargs="+", help="result set names under bench/results/")
    ap.add_argument("-o", "--out",
                    help="output path (default bench/results/report-<dataset>-<labels>-<date>-<hhmm>.html)")
    ap.add_argument("--open", action="store_true", help="open it when done")
    args = ap.parse_args(argv[1:])

    runs = [load_run(x) for x in args.labels]
    names = datasets_of(runs)
    if len(names) > 1:
        # Not a comparison. Two engines on two corpora share a page and a set
        # of ratio columns, and every one of those ratios would be a statement
        # about the datasets rather than about the engines (§4.1). A stamp that
        # differs is bannered and still worth reading; this is not.
        pairs = ", ".join(f"{r.label} measured {datasets_of([r])[0]}" for r in runs)
        print(f"refusing to put these runs on one page: {pairs}. A ratio across two "
              f"datasets compares the corpora, not the engines (§4.1). Render them "
              f"separately, one call per dataset.", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else default_out(runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(runs, f"strawmANN benchmark on {names[0]}: "
                               + " vs ".join(args.labels)))

    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    for r in runs:
        dirty = [x["id"] for x in r.rows if x.get("foreign")]
        # The verdict word, not a boolean: `mixed` is what an arm reads when
        # its rows disagree, and collapsing that to FAIL loses the one detail
        # that says where to look.
        # Against `n_measured`, not `len(rows)`. `Run.n_declined` was split out
        # because "25/27 ok" reads as two failures when it is two rows the
        # engine declined by name, and the page was fixed while this line was
        # not: the dbpedia-openai-1m render printed "sm-dbp1m 30/32 ok" for a
        # run in which nothing failed, W12 being filtered search that strawmANN
        # does not implement.
        print(f"  {r.label:<12} {r.n_ok}/{r.n_measured} ok"
              + (f", {r.n_declined} declined" if r.n_declined else "")
              + f"  gate={compare.gate_of_rows(r.rows, r.env)}"
              + (f"  contaminated: {', '.join(dirty)}" if dirty else ""))
    if args.open:
        subprocess.run(["xdg-open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
