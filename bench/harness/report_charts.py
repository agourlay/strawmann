#!/usr/bin/env python3
"""The report's figures: one function per chart, each returning a Plotly dict.

A leaf. It reads `report_data` and nothing above it, which is a recent state
of affairs: the frontier analysis used to live here while the per-workload
sections read it, and these charts read the sections' refusals back — so
neither half could move without inventing an import cycle. The analysis and
the refusals are in `report_data` now, under both halves.

The four functions that assemble a page — `build`, `build_figures`,
`default_out`, `main` — stay in `report.py`, interleaved among these until
this split; they orchestrate, and orchestration is what `report.py` is.
"""

from __future__ import annotations

import html
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import perfstat
import plotly.graph_objects as go
import plotly.io as pio

import compare
import recall as recall_mod
from report_data import (
    BUILD_ROWS,
    ENCODINGS,
    ROOT,
    Run,
    _interp_segment,
    describe,
    frontier_points,
    matched_ratios,
    matched_recall_caveat,
    matched_recall_refusal,
    ratio_value,
    run_colour,
    run_meta_tag,
    segment_note,
)

#: Tick labels for an axis whose values run into the thousands: `41901` reads
#: as `40k`. d3's SI format, which is what plotly's `tickformat` speaks.
#:
#: Per axis and not in `layout()` below, because it is wrong for most of them.
#: The latency charts are milliseconds under one, where `~s` renders 0.5 as
#: `500m` — a milli prefix on a figure already labelled "milliseconds" — and
#: the recall sweep's x axis names ef values (32, 64, ...) that want their
#: digits. Only three axes carry thousands, and they ask for this by name.
#:
#: It is set at all because nothing was: every axis took plotly's defaults,
#: which abbreviate on a linear axis and not on a log one, so the same figure
#: read `20k` in one chart and `20000` in another.
SI_TICKS = "~s"


def layout(**over) -> dict:
    """Shared chart layout, with per-chart overrides merged in.

    A module-level dict would collide: `update_layout(**LAYOUT, height=...)`
    passes `height` twice and raises.
    """
    base = dict(
        template="none",
        margin=dict(l=64, r=24, t=10, b=52),
        height=380,
        font=dict(family="ui-sans-serif, system-ui, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="closest",
        # Light-theme values. `plotly_white` bakes in #2a3f5f text and #EBF0F8
        # grid lines, which land at about 1.3:1 on the dark card: the axis
        # labels and legend disappear entirely. Plotly cannot read CSS
        # variables, so these are set explicitly here and swapped at runtime by
        # THEME_JS below.
        font_color="#1a1a19",
        xaxis=dict(gridcolor="#e6e4df", linecolor="#d5d2cb", zerolinecolor="#d5d2cb",
                   tickfont=dict(color="#6b6b66")),
        yaxis=dict(gridcolor="#e6e4df", linecolor="#d5d2cb", zerolinecolor="#d5d2cb",
                   tickfont=dict(color="#6b6b66")),
        legend_font_color="#1a1a19",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d5d2cb",
                        font=dict(color="#1a1a19")),
    )
    base.update(over)
    return base

def fig_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})

#: Longest y-axis label a chart will draw, in characters.
#:
#: The margin these charts reserve is `min(30 + 7 * widest, _LABEL_PX)` px, and the
#: cap is load-bearing — without it one long row pushes the plot area off the
#: card. But the label was never trimmed to match, so anything past the cap was
#: simply cut off at the left edge: `W4-sat50` rendered as `N4-sat50` and
#: `W11-steady  search bench2 while 50,000 synthetic points append` as
#: `eshold: search bench2 while 50,000 synthetic points append`. A row the
#: reader cannot identify, and an id that reads as a different id.
#:
#: The *id* is what identifies the row, so the description is what gets
#: shortened — and the cap is set from §4's own labels rather than from a round
#: number. They run 17 to 108 characters; all but two fit inside 62, and the two
#: that do not (`W11` at 86, `W11-steady` at 108) are the concurrent rows whose
#: purpose reads as a sentence. Cutting at 62 keeps every other row's wording
#: whole — including the phrasing `test_chart_labels_use_the_tables_purpose`
#: asserts comes from §4 rather than from a hand-written string — and bounds the
#: margin at 464 px, which is what stops one long row from pushing the plot area
#: off the card.
_LABEL_CHARS = 62
_LABEL_PX = 30 + 7 * _LABEL_CHARS


def row_label(wid: str, extra: str = "") -> str:
    """`W3  search, fp32, single query`, short enough to be drawn in full."""
    desc = describe(wid)
    head = f"{wid}  "
    room = _LABEL_CHARS - len(head) - len(extra)
    if room > 3 and len(desc) > room:
        desc = desc[:room - 1].rstrip() + "\u2026"
    return head + desc + extra


def refused_ids(runs: list[Run]) -> dict[str, str]:
    """The rows `compare.py` declines to compare, and why, keyed by row id.

    The tables have honoured these refusals for a while; the charts never did.
    A bar chart drawing W7 (recall@10 0.03 on both sides, so the rate measures
    nothing) and W11 (2% write overlap against 19%, so the two rows are not the
    same experiment) beside the rows that *are* comparable hands a skimmer
    exactly the comparison every table on the page refuses to state.
    """
    if len(runs) != 2:
        return {}
    out = {}
    for jr in compare.joined(runs[0].label, runs[1].label,
                             runs[0].by_id(), runs[1].by_id()):
        if ratio_value(jr.ratio) is None:
            reason = jr.refusal or "; ".join(n.strip("[]") for n in jr.notes) or "not comparable"
            out[jr.id] = reason
    return out

def chart_throughput(runs: list[Run], df: pd.DataFrame) -> dict:
    wanted = ["W0", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W11", "W13"]
    present = [w for w in wanted if w in set(df["id"])]
    refused = refused_ids(runs)
    # Named in the label rather than dropped: the bar is still that engine's
    # own measured rate, which is licensed on its own (§8 T1). What is not
    # licensed is reading the pair as a comparison, and that is what the label
    # and the caption say.
    label = {w: row_label(w, "  \u2014 not compared" if w in refused else "")
             for w in present}
    fig = go.Figure()
    for i, run in enumerate(runs):
        sub = df[(df["engine"] == run.label) & (df["id"].isin(present))]
        m = {r["id"]: r["qps"] for _, r in sub.iterrows()}
        fig.add_bar(
            name=run.label, orientation="h",
            y=[label[w] for w in present],
            x=[m.get(w) for w in present],
            marker_color=run_colour(run, i), meta=run_meta_tag(run),
            marker_pattern_shape=["/" if w in refused else "" for w in present],
            hovertemplate="%{y}<br>%{x:,.0f} qps<extra>" + run.label + "</extra>")
    # The left margin has to fit the category labels, which are the workload id
    # plus its description: "W13  scroll, id-ordered pagination" is 34
    # characters and the shared 64 px margin truncated every one of them.
    # Sized from the longest label rather than by a constant that goes stale
    # the moment a description is reworded.
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(barmode="group", height=90 + 42 * len(present),
                              margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24, t=10, b=52),
                              xaxis_title="queries per second"))
    fig.update_yaxes(autorange="reversed")
    note = "Higher is better."
    shown = [w for w in present if w in refused]
    if shown:
        # The reasons are the table's notes column; repeating them here put the
        # same three sentences on the page twice, a screen apart.
        why = html.escape("; ".join(f"{w}: {refused[w]}" for w in shown))
        note += (f' Hatched bars are rows the table does not compare (<span title="{why}">'
                 + ", ".join(shown) + "</span>): each bar is that engine's own rate, and "
                 "the pair is not a result.")
    return dict(section="throughput", title="Throughput by workload", html=fig_html(fig),
                note=note)

def chart_ef_sweep(runs: list[Run], df: pd.DataFrame) -> dict | None:
    fig = go.Figure()
    any_pts = False
    for i, run in enumerate(runs):
        sub = df[(df["engine"] == run.label) & (df["id"].str.startswith("W10-ef"))]
        pts = sorted((int(r["id"].removeprefix("W10-ef")), r["qps"])
                     for _, r in sub.iterrows() if pd.notna(r["qps"]))
        if not pts:
            continue
        any_pts = True
        fig.add_scatter(name=run.label, x=[p[0] for p in pts], y=[p[1] for p in pts],
                        mode="lines+markers", line=dict(width=2.4, color=run_colour(run, i)),
                        marker=dict(size=8), meta=run_meta_tag(run),
                        hovertemplate="ef=%{x}<br>%{y:,.0f} qps<extra>" + run.label + "</extra>")
    if not any_pts:
        return None
    fig.update_layout(**layout(xaxis_title="ef (candidate list size)",
                              yaxis_title="queries per second"))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", tickformat=SI_TICKS)
    return dict(
        section="throughput-more", title="W10: throughput against ef",
        html=fig_html(fig),
        note=segment_note(runs))

def chart_latency_percentiles(runs: list[Run]) -> dict | None:
    order = ["p50", "p95", "p99", "p99.9", "max"]
    qs = [0.50, 0.95, 0.99, 0.999, 1.0]
    fig = go.Figure()
    any_pts = False
    for i, run in enumerate(runs):
        v = run.timings("W4")
        if not v:
            continue
        any_pts = True
        s = pd.Series(v) * 1000
        fig.add_scatter(name=run.label, x=order, y=[s.quantile(q) for q in qs],
                        mode="lines+markers", line=dict(width=2.4, color=run_colour(run, i)),
                        marker=dict(size=8), meta=run_meta_tag(run),
                        hovertemplate="%{x}: %{y:.3f} ms<extra>" + run.label + "</extra>")
    if not any_pts:
        return None
    fig.update_layout(**layout(yaxis_title="milliseconds (log)"))
    fig.update_yaxes(type="log")
    return dict(
        section="latency", title="W4: request latency percentiles under saturation",
        html=fig_html(fig),
        note="Client-observed round trip, queueing included. Log axis because p99.9 sits "
             "an order of magnitude above p50 and a linear axis flattens everything below "
             "it. W4 is a closed loop, so its tail is understated for whichever engine is "
             "slower; the open-loop W4-sat rows are the honest tail.")

def chart_latency_distribution(runs: list[Run]) -> dict | None:
    """Two latency distributions on one axis, as step outlines.

    They were overlaid filled histograms at 55% opacity. Where the two cross,
    translucent fills blend into a colour that belongs to neither series, so a
    reader sees three populations where there are two — and the blended region
    is exactly the region worth reading, because it is where the engines are
    alike. An outline crosses another outline without mixing.

    The bins are shared. Each histogram chose its own from its own range
    before, so the two curves were drawn on different grids: a difference in
    bin width was indistinguishable from a difference in shape, and the
    comparison the chart exists for was the thing the binning confounded.
    """
    series = []
    for i, run in enumerate(runs):
        v = run.timings("W4")
        if not v:
            continue
        s = pd.Series(v) * 1000
        # Clip at p99.5: a few multi-millisecond outliers otherwise squeeze the
        # body of the distribution into one bin.
        # Both: the clipped series is what is drawn, the full one is what the
        # note quotes. Quoting the clipped series' max prints the clip
        # boundary — 16.1 ms for a row whose slowest request took 36.2 ms —
        # which is the clip describing itself as the data.
        series.append((run, i, s[s <= s.quantile(0.995)], s))
    if not series:
        return None

    lo = min(float(s.min()) for _, _, s, _ in series)
    hi = max(float(s.max()) for _, _, s, _ in series)
    edges = np.linspace(lo, hi, 71)
    fig = go.Figure()
    for run, i, s, _full in series:
        counts, _ = np.histogram(s, bins=edges)
        # Each bin's height held flat across its own width, so the line is the
        # histogram's outline rather than a curve through its midpoints — a
        # smoothed curve would invent shape between bins.
        fig.add_scatter(x=np.repeat(edges, 2)[1:-1], y=np.repeat(counts, 2),
                        name=run.label, mode="lines",
                        line=dict(color=run_colour(run, i), width=2),
                        meta=run_meta_tag(run),
                        hovertemplate="%{x:.3f} ms<br>%{y} requests<extra>"
                                      + run.label + "</extra>")
    # The clip is what makes the body readable, and it removes the tail — which
    # on this row is where the engines actually differ. Marking each engine's
    # p99 puts the boundary on the chart instead of leaving the reader to infer
    # that the visible shape is the whole distribution.
    for run, i, _s, full in series:
        fig.add_vline(x=float(full.quantile(0.99)), line_width=1.4, line_dash="dot",
                      line_color=run_colour(run, i))
    fig.update_layout(**layout(hovermode="x unified",
                              xaxis_title="request latency (ms), body clipped at p99.5",
                              yaxis_title="requests"))
    # How much of each engine's distribution is past the clip, not only how far
    # it reaches. "max 31.3 ms against 26.2 ms" reads as the worse tail and on
    # the sift1m run it was 56 requests of 50,000 against 40, while
    # the same row put 72 requests over 10 ms against the other engine's 942.
    # A max is one request; the count is the tail's weight, and the two answer
    # different questions about the same shape.
    mark = max(float(full.quantile(0.99)) for _, _, _, full in series)
    weights = "; ".join(
        f"{r.label} {int((full > mark).sum()):,}"
        for r, _i, _s, full in series)
    return dict(
        section="latency", title="W4: request latency distribution",
        html=fig_html(fig),
        note=("Every request bfb timed, not a reconstruction from percentiles. "
              "Bimodality is visible here and invisible in a percentile table. "
              "Dotted lines are each engine's p99. <b>The clip hides the tail, and the "
              "tail is where these engines differ</b> — read the percentile table above "
              "as well as this shape: "
              + "; ".join(f"{r.label} p99 {full.quantile(0.99):,.1f} ms, "
                          f"max {full.max():,.1f} ms" for r, _i, _s, full in series)
              + ". A distribution that looks better in the body can be the worse one "
                f"past the clip — and a max is one request. Past {mark:,.1f} ms, the "
                f"higher of the two p99s, each engine put: {weights} requests of "
                f"{len(series[0][3]):,}."))

def chart_server_vs_client(runs: list[Run]) -> dict | None:
    """Server-side time against client round trip, which separates queueing."""
    fig = go.Figure()
    any_pts = False
    for i, run in enumerate(runs):
        req, srv = run.timings("W4"), run.timings("W4", "server_timings")
        if not req or not srv:
            continue
        any_pts = True
        c = run_colour(run, i)
        for name, v, dash in ((f"{run.label} client", req, "solid"),
                              (f"{run.label} server", srv, "dot")):
            s = pd.Series(v) * 1000
            fig.add_scatter(name=name, x=["p50", "p95", "p99", "p99.9"],
                            y=[s.quantile(q) for q in (0.5, 0.95, 0.99, 0.999)],
                            mode="lines+markers", line=dict(width=2.2, color=c, dash=dash),
                            marker=dict(size=7), meta=run_meta_tag(run),
                            hovertemplate="%{x}: %{y:.3f} ms<extra>" + name + "</extra>")
    if not any_pts:
        return None
    fig.update_layout(**layout(yaxis_title="milliseconds (log)"))
    fig.update_yaxes(type="log")
    return dict(
        section="latency", title="W4: server time against client round trip",
        html=fig_html(fig),
        note="The gap between the two lines is queueing and transport, not search. A "
             "server-side p99 that is flat while the client p99 climbs means the engine "
             "is keeping up and the load generator is the queue.")

def chart_quantization_recall(runs: list[Run]) -> dict | None:
    """recall@10 against ef, one line per encoding, per engine.

    The frontier chart reads `bench2` only, so three of the four sweeps the
    run produces were on disk and never drawn: what SQ8, binary and PQ cost in
    recall was recoverable only from the per-row tables, one ef at a time.

    Worth drawing here for a reason beyond completeness. README finding 24
    records binary quantization as unusable — recall@10 near 0.03 at d=128,
    "characterisation only" — and on a 1536-dimensional corpus it is the best
    of the three lossy encodings by a wide margin. A finding scoped to one
    dataset reads as a property of the method until a second dataset is drawn
    beside it.
    """
    dash = ["solid", "dot", "dash"]
    fig = go.Figure()
    drawn = 0
    for coll, name, hue in ENCODINGS:
        name = encoding_name(coll, name)
        for j, run in enumerate(runs):
            ds = (run.meta.get("dataset") or {}).get("name") or "sift1m"
            pts = (recall_mod.load_recall_points(run.label, ds, coll)
                   or {}).get("points") or []
            pts = [p for p in pts if p.get("recall_at_10") is not None]
            if not pts:
                continue
            drawn += 1
            fig.add_scatter(
                x=[p["ef"] for p in pts], y=[p["recall_at_10"] for p in pts],
                mode="lines+markers", name=f"{name} · {run.label}",
                legendgroup=name,
                line=dict(color=hue, width=2, dash=dash[j % len(dash)]),
                marker=dict(color=hue, size=8,
                            line=dict(color="#fbfbfa", width=1.2)),
                hovertemplate="ef=%{x}<br>recall@10 %{y:.4f}<extra>"
                              + f"{name} · {run.label}" + "</extra>")
    if not drawn:
        return None
    fig.update_layout(**layout(xaxis_title="ef (search breadth)",
                              yaxis_title="recall@10"))
    fig.update_xaxes(type="log", tickvals=[32, 64, 128, 256, 512])
    return dict(
        section="recall", title="What each quantization costs in recall",
        html=fig_html(fig),
        note="<b>Colour is the encoding here, not the engine</b> — the engines are the "
             "line style. Each sweep is joined on the quantization parameters it "
             "actually sent, so a curve speaks only for the search its row ran. "
             "Higher is better, and the encodings are not free: read this against the "
             "throughput their rows bought.")

def chart_build(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Upload against index wait, per loading row, per engine.

    This section's own lede says the two engines do not mean the same thing by
    Time to Green — Qdrant indexes during ingest, strawmANN uploads raw and
    bulk-builds after — and then asks the reader to "compare the sum, not the
    upload line" from a table of nine rows and two columns of seconds. The
    split is the point, so the chart shows it: one bar per engine per row, the
    solid part the upload, the hatched part the wait for Green.

    Stacked *within* an engine and grouped *across* them, which is what
    `offsetgroup` buys: the sum the lede asks for is the bar's length, and the
    difference in where the time goes is visible without arithmetic.
    """
    present = [w for w in BUILD_ROWS if w in set(df["id"])]
    if not present:
        return None
    label = {w: row_label(w) for w in present}
    fig = go.Figure()
    for i, run in enumerate(runs):
        sub = df[df["engine"] == run.label]
        by = {r["id"]: r for _, r in sub.iterrows()}
        col = run_colour(run, i)
        for phase, fld, hatch in (("upload", "upload_s", ""),
                                  ("wait for Green", "index_wait_s", "/")):
            xs = [(by.get(w, {}).get(fld) if by.get(w) is not None else None)
                  for w in present]
            if not any(pd.notna(x) and x for x in xs):
                continue
            fig.add_bar(
                name=f"{run.label} · {phase}", orientation="h",
                y=[label[w] for w in present], x=xs,
                offsetgroup=run.label, legendgroup=run.label,
                marker=dict(color=col,
                            # The hatch in the engine's own colour. A page-
                            # coloured one drew near-white lines on a
                            # transparent fill: the wait bars, and their
                            # legend swatches, did not show.
                            pattern=dict(shape=hatch, solidity=0.35)
                            if hatch else None,
                            # A 2px surface gap so the two phases read as two
                            # segments rather than one bar with a texture.
                            line=dict(color="#fbfbfa", width=2)),
                meta=run_meta_tag(run),
                hovertemplate="%{y}<br>" + phase + " %{x:,.2f} s<extra>"
                              + run.label + "</extra>")
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(barmode="stack", height=90 + 46 * len(present),
                              margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24,
                                          t=10, b=52),
                              xaxis_title="seconds"))
    fig.update_yaxes(autorange="reversed")
    return dict(
        section="ingest", title="Where the load time goes",
        html=fig_html(fig),
        note="Solid is upload, hatched is the wait for Green. The bar's whole length "
             "is the sum this section asks you to compare; the split is why the upload "
             "line alone is not comparable between these engines.")

def chart_runqueue(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Time each row spent waiting for a core, per engine.

    The Scheduler and memory table is 26 rows by twelve columns, and the
    largest number on the page is inside it: W4-sat70 waited 7.8 ms on one
    engine and 272,790 ms on the other. Three hundred cells is not where a
    four-and-a-half-minute runqueue wait should be discovered.

    A dumbbell rather than bars, and a log axis. The values span five orders of
    magnitude, and a bar encodes magnitude as length from zero — on a log axis
    that mapping is broken, and a bar three quarters as long can be a
    thousandth of the value. A dot encodes position, which a log axis leaves
    honest; the connector between the pair is the gap, which is the reading.

    Rows where neither engine recorded a wait are absent, and so are zeros: a
    log axis has no place to put one. Both are named in the note rather than
    silently dropped — an absent row otherwise reads as "no wait".
    """
    if "runqueue_wait_s" not in df.columns:
        return None
    ms = {}
    for i, run in enumerate(runs):
        for _, r in df[df["engine"] == run.label].iterrows():
            v = r.get("runqueue_wait_s")
            if pd.notna(v) and v > 0:
                ms.setdefault(r["id"], {})[run.label] = float(v) * 1000
    order = [w for w in dict.fromkeys(df["id"]) if w in ms]
    if not order:
        return None
    label = {w: row_label(w) for w in order}

    fig = go.Figure()
    # Connectors first, so the markers sit on top of them.
    xs, ys = [], []
    for w in order:
        vals = list(ms[w].values())
        if len(vals) > 1:
            xs += [min(vals), max(vals), None]
            ys += [label[w], label[w], None]
    if xs:
        fig.add_scatter(x=xs, y=ys, mode="lines", showlegend=False,
                        line=dict(color="#8a8781", width=1.5),
                        hoverinfo="skip")
    for i, run in enumerate(runs):
        pts = [(label[w], ms[w][run.label]) for w in order if run.label in ms[w]]
        if not pts:
            continue
        fig.add_scatter(
            x=[v for _, v in pts], y=[y for y, _ in pts], mode="markers",
            name=run.label, marker=dict(color=run_colour(run, i), size=11,
                                        line=dict(color="#fbfbfa", width=1.5)),
            meta=run_meta_tag(run),
            hovertemplate="%{y}<br>%{x:,.1f} ms waiting<extra>"
                          + run.label + "</extra>")
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(height=90 + 26 * len(order),
                              margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24,
                                          t=10, b=52),
                              xaxis_title="time waiting for a core (ms, log scale)"))
    fig.update_xaxes(type="log", tickformat=SI_TICKS)
    fig.update_yaxes(autorange="reversed")
    missing = [w for w in dict.fromkeys(df["id"]) if w not in ms]
    note = "Lower is better; the bar between a pair is the gap."
    if missing:
        note += _absent_note(missing, "waited for a core")
    return dict(section="scheduler", title="Time spent waiting for a core",
                html=fig_html(fig), note=note)

def chart_memory(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Peak resident set per row, per engine.

    `peak rss` is one column of the twelve in Scheduler and memory, and it is
    the one a reader sizing a machine actually wants: what did holding this
    collection and serving this row cost in memory. Read across a table that
    wide, the shape of it — flat through the search rows, stepping up at each
    load — is invisible.

    Not refused the way the storage rows are. Residency changes where the
    bytes live and therefore what the disk counters say; the resident set is
    what the process actually held either way, which is the number a reader
    budgets against.
    """
    if "rss_peak_bytes" not in df.columns:
        return None
    order = [w for w in dict.fromkeys(df["id"])
             if df[(df["id"] == w)]["rss_peak_bytes"].notna().any()]
    if not order:
        return None
    label = {w: row_label(w) for w in order}
    fig = go.Figure()
    for i, run in enumerate(runs):
        by = {r["id"]: r.get("rss_peak_bytes") for _, r in
              df[df["engine"] == run.label].iterrows()}
        xs = [(by.get(w) or 0) / 1e9 for w in order]
        if not any(xs):
            continue
        fig.add_bar(name=run.label, orientation="h",
                    y=[label[w] for w in order], x=xs,
                    marker_color=run_colour(run, i), meta=run_meta_tag(run),
                    hovertemplate="%{y}<br>%{x:,.2f} GB peak RSS<extra>"
                                  + run.label + "</extra>")
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(barmode="group", height=90 + 30 * len(order),
                              margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24,
                                          t=10, b=52),
                              xaxis_title="peak resident set (GB)"))
    fig.update_yaxes(autorange="reversed")
    return dict(section="scheduler", title="Peak memory per row",
                html=fig_html(fig),
                note="Peak RSS while the row ran, from the engine's own process. "
                     "Unlike the disk counters this is not refused across the two "
                     "engines: residency decides where bytes live, and this is what "
                     "the process held either way.")

def _worth_plotting(values: dict[str, dict[str, float]]) -> bool:
    """Whether a dumbbell has anything a table does not already say.

    One dot is not a chart. A single engine with a single row that stalled
    renders as one marker on a log axis, showing neither a comparison nor a
    distribution — and the number is in the table below it, stated exactly
    rather than positioned approximately. Two points is the floor: either two
    rows to compare, or two engines to compare on one row.
    """
    return sum(len(v) for v in values.values()) >= 2


#: Said whenever an axis ends up logarithmic, and not when it does not. A log
#: axis makes the *distance* between two marks meaningless while leaving their
#: order and position honest, and a reader who does not know which axis they
#: are looking at will read a gap as a difference.
#: Now said once, in the appendix's lede, and on the axis title of every
#: chart it applies to: repeated under each chart it was the same sentence
#: eight times.
LOG_NOTE = ""


def _absent_note(missing: list[str], what: str) -> str:
    """The rows a chart leaves out because they had nothing to draw.

    Counted, with the ids on hover: listing forty ids to say "these were zero"
    was a paragraph of row names under a chart of six rows.
    """
    ids = html.escape(", ".join(missing))
    return (f' <span title="{ids}">{len(missing)} row{"" if len(missing) == 1 else "s"} '
            f'never {what} and {"is" if len(missing) == 1 else "are"} not drawn.</span>')

#: Below this ratio between the largest and smallest value, a log axis is the
#: wrong choice: plotly labels its minor decades, and over a narrow range those
#: labels collide into each other and into the axis title — `700m 800m 900m 1`
#: on top of one another, which is what the first render of the stall charts
#: looked like. Twenty is a little over one decade.
_LOG_SPREAD = 20.0


def _dumbbell(runs: list[Run], values: dict[str, dict[str, float]],
              order: list[str], axis_title: str, hover: str,
              digits: str = ":,.1f", si: bool = False) -> tuple[go.Figure, bool]:
    """One dot per engine per row, joined by the gap between them.

    `chart_runqueue`'s shape, factored out once three more charts wanted it. A
    dot rather than a bar because these quantities can span orders of magnitude
    across §4's table — a d=4 transport floor against an exhaustive scan over a
    million vectors — and a bar encodes magnitude as length from zero, which a
    log axis breaks. A dot encodes position, which a log axis leaves honest,
    and the connector is the comparison.

    The log axis is the *conditional* part, and it was not until a screenshot
    showed why. `chart_runqueue` earns one by spanning five orders of
    magnitude; a stall chart with two rows on it spans three-fold, and a log
    axis over that range labels every minor decade and collides them. So the
    axis follows the data: log where the spread justifies it, linear where it
    does not. Returns which it chose, because the note has to say.

    `si` is asked for by name, for the reason `SI_TICKS` gives: d3's SI format
    abbreviates *downwards* as well as up, so on an axis of sub-millisecond
    values already labelled `(ms)` it renders 0.02 as `20m` — twenty
    milli-milliseconds. Right for a cycle count that runs to six figures,
    wrong for every stall chart, and it took a screenshot to see it.
    """
    label = {w: row_label(w) for w in order}
    fig = go.Figure()
    xs, ys = [], []
    for w in order:
        vals = list(values[w].values())
        if len(vals) > 1:
            xs += [min(vals), max(vals), None]
            ys += [label[w], label[w], None]
    if xs:
        # Connectors first, so the markers sit on top of them.
        fig.add_scatter(x=xs, y=ys, mode="lines", showlegend=False,
                        line=dict(color="#8a8781", width=1.5), hoverinfo="skip")
    for i, run in enumerate(runs):
        pts = [(label[w], values[w][run.label]) for w in order if run.label in values[w]]
        if not pts:
            continue
        fig.add_scatter(
            x=[v for _, v in pts], y=[y for y, _ in pts], mode="markers",
            name=run.label, marker=dict(color=run_colour(run, i), size=11,
                                        line=dict(color="#fbfbfa", width=1.5)),
            meta=run_meta_tag(run),
            hovertemplate="%{y}<br>%{x" + digits + "} " + hover
                          + "<extra>" + run.label + "</extra>")
    flat = [v for row in values.values() for v in row.values()]
    log = bool(flat) and min(flat) > 0 and max(flat) / min(flat) >= _LOG_SPREAD
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(height=90 + 26 * len(order),
                               margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24,
                                           t=10, b=52),
                               xaxis_title=axis_title + (" (log scale)" if log else "")))
    ticks = dict(tickformat=SI_TICKS) if si else {}
    if log:
        fig.update_xaxes(type="log", **ticks)
    else:
        # Linear from zero, so a dot's position is its magnitude and the
        # spacing between two dots is the difference between them.
        fig.update_xaxes(rangemode="tozero", **ticks)
    fig.update_yaxes(autorange="reversed")
    return fig, log


def _per_query(df: pd.DataFrame, runs: list[Run], key: str,
               scale: float = 1.0) -> dict[str, dict[str, float]]:
    """A hardware counter divided by the queries that caused it, per row.

    The only form in which two engines' counters compare: an absolute count is
    a function of how long a row ran and how many threads ran it, and the two
    engines hold neither equal. A row with no query count — a load row — is
    absent rather than plotted with an absolute number under a per-query axis.
    """
    if key not in df.columns or "n_queries" not in df.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        for _, r in df[df["engine"] == run.label].iterrows():
            v, n = r.get(key), r.get("n_queries")
            if pd.notna(v) and pd.notna(n) and n and v > 0:
                out.setdefault(r["id"], {})[run.label] = float(v) * scale / float(n)
    return out


def chart_query_cost(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Core cycles spent answering one query.

    The cost model's own unit, and the number two engines are most directly
    comparable on: throughput folds in thread count and offered load, and a
    ratio of queries per second says which engine finished first without
    saying what either spent. This says what one query cost.
    """
    vals = _per_query(df, runs, "perf_cycles")
    order = [w for w in dict.fromkeys(df["id"]) if w in vals]
    if not order or not _worth_plotting(vals):
        return None
    # The one axis here that runs into six figures, and therefore the one that
    # asks for SI ticks: `600k` rather than `600000`.
    fig, log = _dumbbell(runs, vals, order, "cycles per query", "cycles",
                         digits=":,.0f", si=True)
    return dict(section="hardware", title="What one query cost, in cycles",
                html=fig_html(fig),
                note="Lower is better; the bar between a pair is the gap. "
                     + (LOG_NOTE if log else "")
                     + "Load rows are absent: they have no queries to divide by, "
                       "and an absolute count under a per-query axis would be a "
                       "different quantity wearing this one's label.")


def chart_dram_per_query(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Demand loads served from DRAM per query, times the cache line.

    **Demand** is the load-bearing word and the title says so, because the
    event is `ls_dmnd_fills_from_sys.dram_io_all` and the hardware prefetcher's
    fills are a *different* counter (`ls_hw_pf_dc_fills`). On a pointer-chasing
    graph walk the prefetcher has little to predict and demand fills are most
    of the traffic; on a sequential scan it supplies nearly all of it, and this
    number is then a small fraction of what the engine actually moved.

    Measured on the dbpedia-openai-1m run, which is why this says `demand`
    rather than `DRAM traffic`: W9 scans a 6.08 GB collection per query and
    this reads 1.5 GB. The counter is right and the old title was not.

    §5 argues the engine is limited by outstanding misses and the traffic they
    imply, and demand fills are exactly the outstanding-miss half of that —
    which makes them the right number for the graph rows and the wrong one to
    compare against a bandwidth roofline on the scan rows.
    """
    vals = _per_query(df, runs, "perf_dram_fills", scale=perfstat.line_bytes() / 1024)
    order = [w for w in dict.fromkeys(df["id"]) if w in vals]
    if not order or not _worth_plotting(vals):
        return None
    fig, log = _dumbbell(runs, vals, order, "KiB of demand fills from DRAM per query",
                         "KiB")
    return dict(section="hardware", title="Demand loads from DRAM per query",
                html=fig_html(fig),
                note="Demand loads served from DRAM, times the cache line this host "
                     "reports. Lower is better. " + (LOG_NOTE if log else "")
                     + "<b>Demand only</b>: the hardware prefetcher's fills are a "
                       "separate counter and are not in this number, so a row that "
                       "streams — an exact scan above all — moved far more than this "
                       "says. On the graph rows, where there is little for a "
                       "prefetcher to predict, it is most of the traffic and is the "
                       "outstanding-miss quantity §5.2 argues the engine is limited "
                       "by.")


def chart_ipc(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Instructions retired per cycle, per row.

    Bars and a linear axis, unlike the two above: IPC is bounded by the width
    of the machine, every row lands between roughly 0.5 and 4, and length from
    zero is exactly the right encoding for it.

    It is not a score. A low IPC on a memory-bound row is the *expected*
    reading — the core is waiting for lines, which is what §5.2 says the
    limiter is — and a high IPC on a row that does more work per query is not
    a win. It is here to be read beside the two charts above: cheap cycles and
    low DRAM traffic is a kernel that fits; expensive cycles and low IPC is one
    that is waiting.
    """
    if "perf_cycles" not in df.columns or "perf_instructions" not in df.columns:
        return None
    ipc: dict[str, dict[str, float]] = {}
    for run in runs:
        for _, r in df[df["engine"] == run.label].iterrows():
            c, i_ = r.get("perf_cycles"), r.get("perf_instructions")
            if pd.notna(c) and pd.notna(i_) and c:
                ipc.setdefault(r["id"], {})[run.label] = float(i_) / float(c)
    order = [w for w in dict.fromkeys(df["id"]) if w in ipc]
    if not order:
        return None
    label = {w: row_label(w) for w in order}
    fig = go.Figure()
    for i, run in enumerate(runs):
        xs = [ipc[w].get(run.label) for w in order]
        if not any(x is not None for x in xs):
            continue
        fig.add_bar(name=run.label, orientation="h",
                    y=[label[w] for w in order], x=xs,
                    marker_color=run_colour(run, i), meta=run_meta_tag(run),
                    hovertemplate="%{y}<br>%{x:,.2f} instructions per cycle<extra>"
                                  + run.label + "</extra>")
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(barmode="group", height=90 + 30 * len(order),
                               margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24,
                                           t=10, b=52),
                               xaxis_title="instructions per cycle"))
    fig.update_yaxes(autorange="reversed")
    return dict(section="hardware", title="Instructions per cycle",
                html=fig_html(fig),
                note="How well the core was fed while it ran. Not a score: a low "
                     "IPC on a memory-bound row is what §5.2 predicts, and a high "
                     "one on a row that does more work per query is not a win. Read "
                     "it beside the two charts above.")


def chart_tlb_per_query(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Page walks per query — §5.5's argument, measured.

    §5.5 is an argument about TLB reach: a memory-resident index touched
    randomly walks the page tables, and hugepages are the answer. Until the
    sidecar existed the argument had never been checked against a running
    engine, and it is the one hardware column with a *design* behind it rather
    than a diagnosis.
    """
    vals = _per_query(df, runs, "perf_dtlb_walks")
    order = [w for w in dict.fromkeys(df["id"]) if w in vals]
    if not order or not _worth_plotting(vals):
        return None
    fig, log = _dumbbell(runs, vals, order, "data TLB walks per query", "walks")
    return dict(section="hardware", title="TLB walks per query",
                html=fig_html(fig),
                note="Data TLB misses that reached a page walk, per query. Lower is "
                     "better. " + (LOG_NOTE if log else "")
                     + "§5.5 argues a memory-resident index needs hugepage care; this "
                       "is what not taking it costs, and `--no-huge-pages` is the A/B "
                       "arm that prices it.")


def chart_branch_mpki(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Mispredicted branches per thousand instructions.

    Bars and a linear axis, like IPC and for the same reason: MPKI on this kind
    of code lands in single digits, and length from zero is the right encoding.

    A graph traversal is a chain of data-dependent branches — which neighbour
    next, is this one visited, does it enter the heap — and none of them is
    predictable from the last query. This is what that costs, and it is the
    column that separates "the kernel is slow" from "the walk is unpredictable".
    """
    if "perf_branch_misses" not in df.columns or "perf_instructions" not in df.columns:
        return None
    mpki: dict[str, dict[str, float]] = {}
    for run in runs:
        for _, r in df[df["engine"] == run.label].iterrows():
            m, i_ = r.get("perf_branch_misses"), r.get("perf_instructions")
            if pd.notna(m) and pd.notna(i_) and i_:
                mpki.setdefault(r["id"], {})[run.label] = float(m) / (float(i_) / 1000)
    order = [w for w in dict.fromkeys(df["id"]) if w in mpki]
    if not order or not _worth_plotting(mpki):
        return None
    label = {w: row_label(w) for w in order}
    fig = go.Figure()
    for i, run in enumerate(runs):
        xs = [mpki[w].get(run.label) for w in order]
        if not any(x is not None for x in xs):
            continue
        fig.add_bar(name=run.label, orientation="h",
                    y=[label[w] for w in order], x=xs,
                    marker_color=run_colour(run, i), meta=run_meta_tag(run),
                    hovertemplate="%{y}<br>%{x:,.2f} misses per 1k instructions"
                                  "<extra>" + run.label + "</extra>")
    widest = max((len(v) for v in label.values()), default=0)
    fig.update_layout(**layout(barmode="group", height=90 + 30 * len(order),
                               margin=dict(l=min(30 + 7 * widest, _LABEL_PX), r=24,
                                           t=10, b=52),
                               xaxis_title="mispredicted branches per 1k instructions"))
    fig.update_yaxes(autorange="reversed")
    return dict(section="hardware", title="Branch mispredictions per 1k instructions",
                html=fig_html(fig),
                note="A graph traversal is a chain of data-dependent branches and none "
                     "of them is predictable from the last query, so this is the column "
                     "that separates a slow kernel from an unpredictable walk. Per "
                     "thousand instructions rather than per branch: `branches` costs a "
                     "PMU counter that `ref-cycles` needs more, and MPKI is the "
                     "comparable form anyway.")


def _pressure_chart(runs: list[Run], df: pd.DataFrame, resource: str,
                    title: str, what: str) -> dict | None:
    """One PSI resource, per row, per engine.

    `full` rather than `some`: `some` means at least one task was stalled,
    which a busy engine does constantly and which costs nothing on its own;
    `full` means every task in the engine's cgroup was stalled, and that is
    time the engine could not have used a core if it had one.

    Absent where nothing stalled. That is the common and correct reading for a
    memory-resident engine, and a chart of zeros on a log axis would say it
    less clearly than no chart plus the table's zeros.
    """
    key = f"psi_{resource}_full_s"
    if key not in df.columns:
        return None
    ms: dict[str, dict[str, float]] = {}
    for run in runs:
        for _, r in df[df["engine"] == run.label].iterrows():
            v = r.get(key)
            if pd.notna(v) and v > 0:
                ms.setdefault(r["id"], {})[run.label] = float(v) * 1000
    order = [w for w in dict.fromkeys(df["id"]) if w in ms]
    if not order or not _worth_plotting(ms):
        return None
    fig, log = _dumbbell(runs, ms, order, f"time fully stalled on {what} (ms)", "ms")
    missing = [w for w in dict.fromkeys(df["id"]) if w not in ms]
    note = (f"Time every task in the engine's cgroup was stalled on {what}. Lower "
            f"is better. " + (LOG_NOTE if log else ""))
    if missing:
        note += _absent_note(missing, f"stalled on {what}")
    return dict(section="stalls", title=title, html=fig_html(fig), note=note)


def chart_io_pressure(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """The disk-backed engine's tax, if it paid one."""
    return _pressure_chart(runs, df, "io", "Time stalled on I/O", "I/O")


def chart_memory_pressure(runs: list[Run], df: pd.DataFrame) -> dict | None:
    """Reclaim, refault and allocation stalls — the price of holding the index."""
    return _pressure_chart(runs, df, "mem", "Time stalled on memory", "memory")


def chart_load(runs: list[Run], df: pd.DataFrame) -> dict:
    """Peak ambient load per row, coloured by engine and hatched when dirty.

    Colour used to mean clean-or-contaminated: green for one, red for the
    other, for both engines. So the two engines' clean rows — nearly every row
    on a good run — were the same green, and the chart could not answer "whose
    row is this?", which is the question every other chart on the page answers
    with the same two colours.

    Colour is the engine here as everywhere else, and contamination is a hatch
    with a red edge. That also makes it not colour-alone, which is what a
    status encoding needs: a reader who cannot separate the two hues still sees
    which bars are hatched.
    """
    fig = go.Figure()
    for i, run in enumerate(runs):
        sub = df[df["engine"] == run.label]
        peak = sub[["load_start", "load_end"]].max(axis=1)
        clean = sub["foreign"] == ""
        col = run_colour(run, i)
        for mask, name, dirty in ((clean, f"{run.label} clean", False),
                                  (~clean, f"{run.label} contaminated", True)):
            if not mask.any():
                continue
            fig.add_bar(name=name, orientation="h", y=sub[mask]["id"], x=peak[mask],
                        marker=dict(
                            color=col,
                            pattern=dict(shape="/", solidity=0.42,
                                         fgcolor="#DC244B") if dirty else None,
                            line=dict(color="#DC244B", width=1.4) if dirty
                            else dict(width=0)),
                        opacity=0.85,
                        # Tagged, so the theme switch recolours it. It was
                        # deliberately untagged while colour meant the reading
                        # rather than the engine; now that it means the engine,
                        # the tag is correct and dark mode restyles it.
                        meta=run_meta_tag(run),
                        customdata=sub[mask]["foreign"],
                        hovertemplate="%{y}: %{x}% of one core"
                                      "<br>%{customdata}<extra></extra>")
    fig.update_layout(**layout(barmode="group",
                              height=110 + 22 * len(df["id"].unique()),
                              xaxis_title="peak ambient load, percent of one core"))
    fig.update_yaxes(autorange="reversed")
    return dict(
        section="host", title="Ambient load per row",
        html=fig_html(fig),
        note="The §7.1 gate checks the machine once, at the start. Colour is the engine, "
             "as everywhere else; a hatched bar with a red edge is a row that had another "
             "process on the box while it ran, which the gate cannot see.")

def recall_for(run: Run, collection: str) -> dict:
    """The sweep for one collection of one run.

    `run.recall` is `bench2`'s, loaded eagerly because the frontier needs it;
    the quantized collections are read on demand and are simply absent on a run
    that predates `recall.py`.
    """
    if collection == "bench2":
        return run.recall
    dataset = (run.meta.get("dataset") or {}).get("name") or "sift1m"
    return recall_mod.load_recall_json(run.label, dataset, collection)

def _interp_qps(points: list[dict], recall: float) -> float | None:
    """Throughput at a recall the engine was not measured at, or None."""
    seg = _interp_segment(points, recall)
    return None if seg is None else seg[0]

def matched_recall_table(runs: list[Run]) -> str:
    """The comparison at equal recall, which is the only fair one.

    `ef` is not comparable between these engines (Qdrant searches it per
    segment, capped at 8; strawmANN searches one graph once), so a row-by-row
    table at equal `ef` cannot settle anything. This can: at each recall one
    engine actually reached, what did the other serve?
    """
    have = [r for r in runs if r.recall.get("points")]
    if len(have) != 2:
        return ""
    refusal = matched_recall_refusal(runs) if len(runs) == 2 else ""
    if refusal:
        return (f'<div class="banner"><b>No matched-recall ratio is available.</b> '
                f'{html.escape(refusal)}</div>')
    a, b = have
    pa, pb = frontier_points(a, bfb_only=True), frontier_points(b, bfb_only=True)
    if not pa or not pb:
        return ""
    # Shown under a banner rather than withheld when only T3 failed; empty on a
    # licensed pair, which is most of them.
    caveat = matched_recall_caveat(runs)
    head = (f'<div class="banner soft">{html.escape(caveat)}</div>' if caveat else "")
    return (head + _matched_table(a, b, pa, pb)
            + sq8_matched_recall_table(runs)
            + filtered_matched_recall_table(runs))


def filtered_matched_recall_table(runs: list[Run]) -> str:
    """The same comparison under a 10% payload filter, which had a dash.

    `W12-sel10` measures throughput at one `ef`, so `matched_ratios` had
    nothing to interpolate and §7.4's refusal read as a near-miss on the recall
    band: strawmANN 1.0000 against Qdrant 0.9900, 0.0100 apart. The comparison
    was not refused for being close, it was never attempted, and the frontier
    it hid runs the other way — strawmANN's filtered throughput is flat across
    `ef` while Qdrant's falls by a factor of five, so the gap narrows as the
    recall bar rises and the published `ef=128` point is near strawmANN's worst
    (findings 48).

    `W12-sel10-ef*` supplies the throughput half, and this is the same table
    over it. Empty until a run carries those rows, which is every run before
    2026-09-09.
    """
    have = [r for r in runs if r.recall.get("points")]
    # The ladder's own refusals, not W10's — the same argument
    # `sq8_matched_recall_table` makes: a contaminated `W12-sel10-ef256` must
    # not reach a published filtered ratio because W10's rows were clean.
    if len(have) != 2 or matched_recall_refusal(runs, "W12-sel10-ef"):
        return ""
    a, b = have
    grade = "sel10"
    pts = []
    for run in (a, b):
        ds = (run.meta.get("dataset") or {}).get("name") or "sift1m"
        # The graded sweep, by grade: an unfiltered sweep of `bench12` is not
        # written and would answer a different question if it were.
        doc = recall_mod.load_recall_json(run.label, ds, "bench12",
                                          grade=grade) or {}
        # No graded sweep, no table: an empty document made `frontier_points`
        # fall back to the run's unfiltered bench2 recall and pair it with the
        # filtered throughput, with nothing on the page to say so.
        pts.append(frontier_points(run, bfb_only=True, prefix=f"W12-{grade}",
                                   collection="bench12", recall_doc=doc) if doc else [])
    if not pts[0] or not pts[1]:
        return ""
    inner = _matched_table(a, b, pts[0], pts[1], sweep="W12-sel10",
                           caveat=False)
    if not inner:
        return ""
    # Each engine's own count: bfb draws the keyword payloads unseeded at
    # every upload, so the two matching sets differ, and one number was one
    # engine's first pass.
    def n_of(run: Run, p: list):
        return ((p[0] or {}).get("n_matching")
                or (recall_mod.load_recall_json(run.label,
                    (run.meta.get("dataset") or {}).get("name") or "sift1m",
                    "bench12", grade=grade).get("points") or [{}])[0].get("n_matching"))
    na, nb = n_of(a, pts[0]), n_of(b, pts[1])
    if na and nb and na != nb:
        matched = (f" over the points the condition matched, {na:,} in {a.label} "
                   f"and {nb:,} in {b.label} (bfb draws the keyword payloads "
                   f"unseeded at each upload)")
    else:
        matched = f" over the {na or nb:,} points the condition matched" if na or nb else ""
    return ('<h3 style="margin-top:28px">At matched recall, filtered to 10%</h3>'
            '<p class="note">The same reading under a keyword filter'
            + html.escape(matched) +
            '. The per-row table refuses <code>W12-sel10</code> a ratio because '
            'the two engines land just outside the recall band at the one '
            '<code>ef</code> it measures; held at equal recall instead, the '
            'comparison exists at every recall both engines reach. Note the '
            'shape rather than any single number: one engine\'s curve is flat '
            'in <code>ef</code> and the other\'s is steep, so where you match '
            'decides the ratio.</p>' + inner)


def sq8_matched_recall_table(runs: list[Run]) -> str:
    """The same comparison on SQ8, which had none at all.

    W6, W7 and W8 are refused a ratio on every dataset because the engines size
    their rescore pool differently (`decisions.md` §5) and so are not doing the
    same work at one `ef`. That refusal is right and its consequence was that
    the only lossy encoding usable on both corpora could not be compared at all.

    W10 already solves this shape for fp32 — sweep `ef` for throughput, join the
    conformance sweep for recall, read it at a recall both engines reach — and
    the recall half has always existed for `bench6`. `W6-ef*` supplies the
    other half, and this is the same table over it.

    Empty until a run carries those rows, which is every run before 2026-08-24.
    """
    have = [r for r in runs if r.recall.get("points")]
    # The SQ8 table is built from the `W6-ef*` rows, so it is *their* refusals
    # that apply; checking W10's let a contaminated W6-ef256 into a published
    # SQ8 ratio while the throughput table printed a dash for it.
    if len(have) != 2 or matched_recall_refusal(runs, "W6-ef"):
        return ""
    a, b = have
    pts = []
    for run in (a, b):
        ds = (run.meta.get("dataset") or {}).get("name") or "sift1m"
        doc = recall_mod.load_recall_points(run.label, ds, "bench6") or {}
        # As for the filtered table: no SQ8 sweep is no SQ8 table, not fp32's.
        pts.append(frontier_points(run, bfb_only=True, prefix="W6",
                                   collection="bench6", recall_doc=doc) if doc else [])
    if not pts[0] or not pts[1]:
        return ""
    inner = _matched_table(a, b, pts[0], pts[1], sweep="W6", caveat=False)
    if not inner:
        return ""
    import workloads
    why = ("Under the `pool` policy both engines rescore `ef` candidates, so W6 "
           "is compared row by row as well; held at equal recall, the reading "
           "does not depend on the rows landing in one recall band"
           if workloads.OVERSAMPLING_POLICY is workloads.OversamplingPolicy.pool else
           "W6 itself is refused a ratio because the engines rescore differently "
           "sized pools (decisions §5); held at equal recall instead, that "
           "divergence becomes an operating point rather than a refusal")
    return ('<h3 style="margin-top:28px">At matched recall, SQ8</h3>'
            '<p class="note">The same reading over the scalar-quantized '
            f'collection. {why}. '
            'Note where each engine\'s curve stops, '
            'because a recall only one of them reaches is the more useful '
            'fact about an encoding than any ratio.</p>' + inner)


def _matched_table(a: Run, b: Run, pa: list, pb: list, sweep: str = "W10",
                   caveat: bool = True) -> str:
    """One matched-recall table, shared by the fp32 and SQ8 readings.

    `caveat` is the paragraph on what the last column means. It is the same
    paragraph for both readings, so the second table points at the first
    instead of repeating it.
    """
    # Whether the throughput behind the interpolation is one pass or a fold.
    # The note asserted "single passes with no repetition behind them", which
    # is false of every `--reps 3` run and is the one claim in it a reader
    # cannot check.
    reps = min((r.meta or {}).get("reps") or 1 for r in (a, b))
    behind = "medians of the run's passes" if reps > 1 else "single passes"
    body = []
    for r in matched_ratios(pa, pb):
        src_run, other_run = (b, a) if r["swapped"] else (a, b)
        meas = f'<td class="num">{r["measured"]:,.0f}</td>'
        interp = f'<td class="num muted">{r["interpolated"]:,.0f}</td>'
        cells = (interp + meas) if r["swapped"] else (meas + interp)
        band = (f'{r["band"][0]:.2f} &ndash; {r["band"][1]:.2f}x'
                if r["band"] else '<span class="muted">-</span>')
        body.append(
            f'<tr><td class="num">{r["recall"]:.4f}</td>'
            f'<td class="desc">{src_run.label} at ef={r["ef"]}, '
            f'{other_run.label} interpolated</td>{cells}'
            f'<td class="num {"up" if r["ratio"] >= 1 else "down"}">'
            f'{r["ratio"]:.2f}x</td>'
            f'<td class="num muted">{band}</td></tr>')
    if not body:
        return ""
    rows = "".join(body)
    head = (f"<th>recall@10</th><th>measured at</th>"
            f'<th class="num">{a.label} q/s</th><th class="num">{b.label} q/s</th>'
            f'<th class="num">{a.label} / {b.label}</th>'
            f'<th class="num" title="the ratio with the anchor recall moved across its '
            f'95% interval">range over recall CI</th>')
    # The method under a disclosure: the table is four columns a reader needs
    # and two paragraphs a reviewer does.
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<details class="more"><summary>How this is computed</summary>'
            f'<p class="note">Interpolated linear in log(q/s) between the two '
            f'bracketing measurements, nothing extrapolated past the measured range. '
            f'Throughput is bfb\'s {sweep} sweep, recall '
            f'the conformance sweep, joined on <code>ef</code> — same <code>m</code> and '
            f'<code>ef_construct</code>, but separate builds, so the pairing assumes two '
            f'builds with identical parameters are equivalent.</p>'
            + (f'<p class="note"><b>The last column is why the ratio is not a result on its '
            f'own.</b> The ratio treats the anchor\'s recall as exact; it is an estimate, '
            f'and moving it across its 95% interval moves the interpolated rate with it — '
            f'near the top of the sweep 0.003 of recall spans a factor of 1.8, so two '
            f'decimals there quote the interpolation. Even that is the narrow reading: it '
            f'moves one recall and not the other, adjacent anchors share bracketing '
            f'segments, and the throughputs behind it are '
            f'{behind}.</p>' if caveat else
               '<p class="note">The last column reads as it does in the fp32 table '
               'above.</p>') + '</details>')

def encoding_name(collection: str, default: str) -> str:
    """A curve's legend, with the oversampling its rows sent under the run's
    policy: under `pool` binary is not "4x", it is `ef / limit` per row."""
    import workloads
    if workloads.OVERSAMPLING_POLICY is workloads.OversamplingPolicy.pool and \
            collection in ("bench6", "bench7", "bench8"):
        return default.split(",")[0] + ", rescore pool matched to ef"
    return default


def chart_frontier(runs: list[Run]) -> dict | None:
    """Throughput against recall: the only comparison `ef` does not distort.

    §7.4's rule and `docs/comparison-sift1m.md` §2 both land here. Qdrant searches
    `ef` candidates in *each* segment and caps the count at 8; strawmANN holds
    one graph and searches `ef` once, so equal `ef` is not equal work and the
    W10 curve read at equal x flatters strawmANN by roughly an order of
    magnitude. Recall is the axis both engines mean the same thing by.

    Read vertically and nowhere else. Which engine leads, and by how much, is
    whatever the current data says and is deliberately not asserted here: this
    docstring once described a crossover at ~0.96 recall that a later run did
    not have, and a stale claim in the code that draws the chart is worse than
    no claim. The matched-recall table computes the ordering from the points.
    """
    have = [r for r in runs if r.recall.get("points")]
    if len(have) < 1:
        return None
    fig = go.Figure()
    plotted, sources = 0, set()
    for i, run in enumerate(have):
        pts = frontier_points(run)
        if not pts:
            continue
        sources.update(p["source"] for p in pts)
        fig.add_trace(go.Scatter(
            x=[p["recall"] for p in pts],
            y=[p["qps"] for p in pts],
            mode="lines+markers", name=run.label,
            line=dict(color=run_colour(run, i), width=2),
            marker=dict(size=8), meta=run_meta_tag(run),
            text=[f"ef={p['ef']}" for p in pts],
            hovertemplate="%{text}<br>recall@10 %{x:.4f}<br>%{y:,.0f} q/s<extra></extra>"))
        plotted += 1
    if not plotted:
        return None
    fig.update_layout(**layout(xaxis_title="recall@10 (vs the fp64 oracle)",
                               yaxis_title="queries per second (log)"))
    fig.update_yaxes(type="log", tickformat=SI_TICKS)
    return dict(
        section="summary",
        title="Throughput against recall",
        html=fig_html(fig),
        note="Read it vertically: at any recall both engines reach, the higher curve "
             "is faster. Throughput from " + " and ".join(sorted(sources)) +
             ", recall from the conformance sweep, joined on ef. " + segment_note(runs))

def bfb_commit(runs: list[Run] | None = None) -> str:
    """Which load generator produced these rows, from the rows.

    §9 pins the toolchain because "a benchmark that silently changes its load
    generator is not a benchmark", and this read the *live checkout's* HEAD —
    so the page attributed its rows to whatever happened to be checked out when
    the report was rendered. Caught on the dbpedia-openai-1m report: it claimed
    `dev @ 14d793d` for rows `run.json` says were measured under
    `dev @ 6f216634`, because the checkout had moved in between. Not a cosmetic
    difference either — that pair is the same fork patch rewritten onto eight
    newer upstream commits, which is a different generator, which is the exact
    thing §9's pin exists to notice.

    So: the pin each run recorded, which travels with the measurement. Two runs
    that disagree are named rather than reconciled — a comparison whose arms
    were driven by different generators is not one. The live checkout is the
    last resort, for a run taken before the pin was recorded, and says that it
    is a guess about the past.
    """
    pins = {(r.meta or {}).get("harness", {}).get("bfb_pin") for r in (runs or [])}
    pins.discard(None)
    if len(pins) == 1:
        return str(pins.pop())
    if len(pins) > 1:
        return "MISMATCH: " + " vs ".join(sorted(str(x) for x in pins))
    candidates = [Path(os.environ["BFB_REPO"])] if os.environ.get("BFB_REPO") else []
    candidates += [ROOT.parent / "bfb", Path.home() / "Workspace/bfb"]
    for p in candidates:
        if (p / ".git").exists():
            r = subprocess.run(["git", "-C", str(p), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return f"dev @ {r.stdout.strip()} (checkout now; this run recorded none)"
    return "dev (commit unknown)"


def datasets_of(runs: list[Run]) -> list[str]:
    """Every dataset these runs measured, sorted. One name is the healthy case.

    `sift1m` for a run whose `run.json` predates the field, which is what every
    run of §4's table measured before the dataset was recorded at all.
    """
    return sorted({(r.meta.get("dataset") or {}).get("name") or "sift1m" for r in runs})
