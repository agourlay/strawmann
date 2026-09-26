"""The engine-counter tables: storage, scheduler, stalls, hardware counters and cache sharing.

Moved out of `report.py`, which re-exports every name here.
"""

from __future__ import annotations

import html

import perfstat

import procstat
from report_data import Run, describe


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
