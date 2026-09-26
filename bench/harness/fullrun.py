#!/usr/bin/env python3
"""The gated run, end to end: both engines, both recall sweeps, one report.

Every piece of this existed and nothing ran them in order, so a "full run" was
a sequence of commands held in someone's head — which is exactly how the two
arms of the published comparison ended up carrying different environment
hashes (§7.1: "results from different environments never share a chart"). This
script is that sequence, written down.

    §7.1 gate            refuse to start unless the host is disciplined
    strawmann perf       all of §4's rows, then the recall sweeps
    qdrant perf          the same rows, on the same cores, same sweeps
    conformance          both engines up, the differ writes the §8 row
    sink + report        results.py records, compare.py and report.py render

## The two things this gets right that a hand-run sequence did not

**Both engines get the same cores.** `--cpuset-cpus` for the container and
`--pin --cpus` for strawmANN, from one `--server-cpus` argument, and the load
generator is pinned to the complement with `taskset`. Without this the
comparison silently includes whatever the scheduler decided, and on a
heterogeneous part (findings 20: 5.16 GHz cores with 16 MiB of L3 beside
3.29 GHz cores with 8 MiB) that decision is worth more than most of what is
being measured.

**One environment for both arms.** The gate runs at the start of the sequence
and again inside each arm, via `workloads.py`, which is what stamps the arm's
own `run.json`. Two arms measured hours apart on a machine whose governor moved
are not a comparison, and §7.1's hash is what catches it — after the fact, when
the numbers are already published.

The per-arm gate has to be *given* a quiet machine rather than assumed one:
running the arms back to back means the second inherits the first's decaying
one-minute load average, which failed Qdrant's gate at load 1.79 against
strawmANN's 1.24 and cost the whole run its comparison. `settle`
waits for the load to fall back under the gate's own threshold before each arm.

Usage:
    fullrun.py --server-cpus 4-11 --client-cpus 0-3
    fullrun.py --server-cpus 4-11 --client-cpus 0-3 --skip qdrant
    fullrun.py --lax ...        # development only; stamps every row unpublishable
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
# `bench/`, for `setup.py`: `settle` uses the gate's own quiescence sampler
# rather than a second opinion about the same machine.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
import provenance
import setup
import workloads

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
CONF = ROOT / "conformance"
RESULTS = ROOT / "bench/results"

#: §8.9 pins the comparison target. The tag is resolved to a digest by
#: `provenance.qdrant_build` from the running container, so this is the request
#: and that is the record.
QDRANT_IMAGE = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.19.0")
QDRANT_CONTAINER = "strawmann-fullrun-qdrant"

#: A Qdrant binary to run natively instead of the container, from
#: `--qdrant-binary`. None means the image, which is the right default (§8.9
#: pins by digest, a path cannot) but cannot be *counted*: `perf stat -p` on a
#: containerised engine needs ptrace permission on root's task, which
#: `perf_event_paranoid` does not grant at any value.
QDRANT_BINARY: Path | None = None
QDRANT_STORAGE = paths.QDRANT_STORAGE

STRAWMANN_PORT = 6334
QDRANT_GRPC = 6334
QDRANT_REST = 6333
#: The conformance differ needs both engines at once, so strawmANN moves aside.
STRAWMANN_CONF_PORT = 6344


def sh(argv: list[str], timeout: int = 600, **kw) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


def say(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def qdrant_segment_env() -> dict[str, str]:
    """The server-level segment default this policy asks Qdrant for.

    `equal-work` pins `default_segment_number` to 1 so the differ's own
    collections -- created without bfb's `--segments` -- are held to one graph
    like the measured ones. `as-deployed` asks for nothing: the whole point of
    that policy is Qdrant's own default, resolved to the CPU count, and until
    both start paths exported the pin unconditionally, so the
    as-deployed run measured equal-work under the other name and no stamp
    could tell, because the stamp records the policy rather than the value.
    """
    if workloads.SEGMENT_POLICY is workloads.SegmentPolicy.equal_work:
        return {"QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER": "1"}
    return {}


def cpu_count(spec: str) -> int:
    """How many CPUs a list like `4-11` names."""
    n = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            n += int(hi) - int(lo) + 1
        else:
            n += 1
    return n


# -------------------------------------------------------------------------
# The gate
# -------------------------------------------------------------------------

#: What the gate decided: whether to proceed, and whether it had to be carried
#: past a failure to do so. Returned rather than left in a module global that
#: one function set and another read three hundred lines away.
class Gate(NamedTuple):
    proceed: bool
    failed: bool

    def __bool__(self) -> bool:
        return self.proceed


def gate(lax: bool) -> Gate:
    """§7.1, once, before anything is measured.

    Refusing here rather than stamping afterwards is the point: a four-hour
    sequence that produces unpublishable numbers has wasted four hours.

    Settles first because this gate can refuse outright, and a run started shortly
    after a previous one refused at the front door on `load 3.43` — the last run's
    own conformance phase decaying out of the one-minute average.
    """
    settle("the §7.1 gate")
    say("§7.1 host gate")
    code, out = sh([sys.executable, str(ROOT / "bench/setup.py"), "check"], 120)
    print(out.rstrip(), flush=True)
    # The return code, not a string. `"check(s) failed" not in out` passed
    # when setup.py was missing, crashed, or timed out, because a traceback
    # does not contain that phrase either.
    ok = code == 0
    if ok:
        return Gate(proceed=True, failed=False)
    if code != 0 and "check(s) failed" not in out:
        print(f"\n!! bench/setup.py check did not run to a verdict (exit {code}); "
              f"treating that as a failed gate", file=sys.stderr)
    if lax:
        print("\n!! --lax: proceeding on a host that failed the gate.\n"
              "!! Every row this produces is development-grade. Do not publish it,\n"
              "!! and do not put it beside a row measured on a disciplined host.",
              flush=True)
        return Gate(proceed=True, failed=True)
    print("\nrefusing to run. Fix what `bench/setup.py apply` can fix (needs root),\n"
          "add isolcpus/nohz_full to the kernel cmdline and reboot for the rest,\n"
          "or pass --lax to take development-grade numbers deliberately.",
          file=sys.stderr)
    return Gate(proceed=False, failed=True)


# -------------------------------------------------------------------------
# strawmANN
# -------------------------------------------------------------------------

def resolve_rps_reference(arg: str | None, labels: list[str]) -> float | None:
    """The saturation both open-loop arms are a fraction of, or None.

    §4 sets the fixed-rate arms at a fraction of *measured* saturation, and taken
    per engine that is two different offered loads — 20,375/s against 13,362/s for
    the row both call "90% of saturation" — so the report refuses to read their
    percentiles across. One number for both makes that read legal, and it has to
    be the *slower* engine's: the faster one can serve the slower one's rate, not
    the reverse.

    `auto` reads it off this machine's previous run of these two labels. It is a
    reference, not a measurement; what matters is that both arms are offered the
    same absolute rate. Raises `ValueError` when `auto` has nothing to read,
    because falling back to per-engine references silently is how the refusal
    turns up in a report nobody expected it in.

    A run measured under a different instrument is not a reference: `--perf` costs
    throughput, so a saturation read off a perf-less run is high for a run that
    attaches it. Measured — a 3,484 reference from a perf-less
    container against a real 2,564 under perf, so W4-sat90 offered 3,136 to an
    engine serving 2,542 and returned a 7.5 s p50.
    """
    if arg is None:
        return None
    if str(arg).lower() != "auto":
        try:
            return float(arg)
        except (TypeError, ValueError):
            raise ValueError(f"{arg!r} is not a number or `auto`") from None
    #: `perf_set` is "" for a run that attached nothing and the global is None
    #: for the same state, so they are spelled alike here. Returns whether the
    #: setting is *known* alongside it: a missing `run.json` is no evidence
    #: rather than evidence perf was off. Both disqualify the reference; only
    #: the sentence differs.
    def perf_of(label: str) -> tuple[bool, str | None]:
        meta = ROOT / "bench/results" / label / "run.json"
        if not meta.exists():
            return False, None
        try:
            return True, (json.loads(meta.read_text()).get("perf_set") or "") or None
        except (OSError, json.JSONDecodeError):
            return False, None

    # A fresh pair (a night's date-stamped labels) has no W4 of its own to
    # read, so `auto` reads the previous pair of the same family instead.
    if not any((ROOT / "bench/results" / x / "rows.json").is_file() for x in labels):
        prev = previous_pair(labels)
        if prev is not None:
            print(f"--rps-reference auto: {', '.join(labels)} have no rows yet; "
                  f"reading the previous pair, {', '.join(prev)}", flush=True)
            labels = prev
    seen, wrong_instrument = {}, {}
    for label in labels:
        rows = ROOT / "bench/results" / label / "rows.json"
        if not rows.exists():
            continue
        try:
            got = {r["id"]: r for r in json.loads(rows.read_text())}
        except (OSError, json.JSONDecodeError):
            continue
        qps = (got.get(workloads.SATURATION_ROW) or {}).get("qps")
        if not (isinstance(qps, (int, float)) and qps > 0):
            continue
        known, was = perf_of(label)
        if not known or was != PERF_SET:
            wrong_instrument[label] = (known, was, float(qps))
            continue
        seen[label] = float(qps)
    if wrong_instrument:
        for label, (known, was, qps) in wrong_instrument.items():
            how = (f"it was measured with perf {was or 'off'}" if known
                   else "there is no run.json saying which instrument measured it")
            print(f"!! ignoring {label}'s {workloads.SATURATION_ROW} of {qps:,.0f} qps "
                  f"for --rps-reference: {how}, and this run attaches perf "
                  f"{PERF_SET or 'off'}", flush=True)
    if len(seen) < len(labels):
        missing = [x for x in labels if x not in seen]
        why = ""
        if wrong_instrument:
            why = (f" ({', '.join(sorted(wrong_instrument))} was not measured under "
                   f"this run's perf setting, or does not say what measured it, so it "
                   f"is not a reference for this run)")
        raise ValueError(
            f"`auto` needs a previous {workloads.SATURATION_ROW} for both labels on "
            f"this machine, and {', '.join(missing)} has none{why}. Run once without it "
            f"(each engine uses its own saturation, and the report refuses the "
            f"cross-engine latency read), then re-run with `auto`.")
    slower = min(seen, key=lambda k: seen[k])
    print(f"--rps-reference auto: {seen[slower]:,.0f} qps, {slower}'s previous "
          f"{workloads.SATURATION_ROW} — the slower of "
          + ", ".join(f"{k} {v:,.0f}" for k, v in seen.items()), flush=True)
    return seen[slower]


#: Each mixed row's search length, as the variable `workloads` reads.
W11_QUERIES_ENV = {"W11-steady": "W11_STEADY_QUERIES", "W11": "W11_QUERIES"}

#: How far inside the append the slower engine's search should end. The row
#: waits for the appender either way; a search that outlives it puts the
#: "writer covered under 90%" refusal straight back, so the margin is on the
#: search.
W11_SPAN_MARGIN = 1.25

#: The spans every run used before 0a3de76 recorded one.
W11_SPANS_BEFORE_RECORDING = {"W11-steady": 25.0, "W11": 60.0}


def w11_append_rates_of(label: str) -> dict[str, int] | None:
    """The mixed rows' write rates a label was measured at, from its `run.json`.

    Recorded since the rate was hashed; before that, rebuilt from the spans
    and volumes the stamp does carry, at today's rounding of `-T`, so 0924
    reads as 1,900 and 3,300 points/s and 0925 as 100 and 400. The rate
    only has to say whether a pair's search speed is today's.
    """
    try:
        h = json.loads((ROOT / "bench/results" / label / "run.json").read_text()).get("harness") or {}
    except (OSError, json.JSONDecodeError):
        return None
    if h.get("w11_append_rate"):
        return h["w11_append_rate"]
    up, w11 = h.get("upload_n"), h.get("w11_n")
    if not up or not w11:
        return None
    spans = h.get("w11_spans_s") or W11_SPANS_BEFORE_RECORDING
    steady = max(1, round(up * workloads.W11_STEADY_RATIO))
    return {"W11-steady": workloads.w11_append_rate(steady, spans["W11-steady"]),
            "W11": workloads.w11_append_rate(w11, spans["W11"])}


def resolve_w11_queries(labels: list[str]) -> dict[str, int]:
    """`{env var: queries}` so each mixed row's search ends inside its append.

    The write rate is fixed (`workloads.W11_STEADY_SPAN_S`); what varies by
    corpus is how long `QUERIES` takes to search. At d=1536 it took 117 to 345
    s against a 25 s and 60 s append, the writer covered 12 to 23% of it, and
    both rows were refused (findings 3). So the search is shortened to what
    the slower engine searched inside the append on that row (its rate times
    the append, or the covered share of its queries where the writer finished
    first), over `W11_SPAN_MARGIN`, and never lengthened past `QUERIES` or cut
    below `MIN_ROW_S` on the faster engine. A search that still overruns
    shrinks the next one by the margin again, so it converges on coverage.

    The rate comes from the newest pair of this family measured at *this*
    write rate (these labels' own rows first): a search is faster against a
    slower writer, so 0925's 2,206 q/s at 200 points/s would size a search
    nine times too long for 1,900. Empty when no pair was, and `QUERIES` stands.
    """
    want = {"W11-steady": workloads.w11_append_rate(workloads.w11_steady_n(),
                                                    workloads.W11_STEADY_SPAN_S),
            "W11": workloads.w11_append_rate(workloads.w11_n(), workloads.W11_SPAN_S)}
    append_s = {"W11-steady": workloads.w11_steady_n() / want["W11-steady"],
                "W11": workloads.w11_n() / want["W11"]}
    pair = (labels if any((ROOT / "bench/results" / x / "rows.json").is_file() for x in labels)
            else previous_pair(labels))
    out: dict[str, int] = {}
    seen: set[str] = set()
    while pair and pair[0] not in seen and len(out) < len(W11_QUERIES_ENV):
        seen.add(pair[0])
        for wid, env in W11_QUERIES_ENV.items():
            if env in out:
                continue
            qps, fits, sent = [], [], []
            for label in pair:
                if (w11_append_rates_of(label) or {}).get(wid) != want[wid]:
                    break
                try:
                    got = {r["id"]: r for r in json.loads(
                        (ROOT / "bench/results" / label / "rows.json").read_text())}
                except (OSError, json.JSONDecodeError):
                    break
                r = got.get(wid) or {}
                # The rate came from `run.json`, which any later `workloads.py
                # run` of the label rewrites; it describes this row only if
                # the row was measured under that stamp.
                if not stamped_by_run_json(label, r):
                    break
                q, n, cover = r.get("qps"), r.get("n_queries"), r.get("write_overlap_pct")
                if not (isinstance(q, (int, float)) and q > 0):
                    break
                qps.append(q)
                # Only a search that was itself sized is a step to damp from:
                # 0924 ran `QUERIES` and says nothing about where the last
                # estimate landed.
                if n and label_sized_w11(label):
                    sent.append(n)
                # A search that outlived its writer ran its tail against a
                # quiet collection, faster, so its qps overstates the search
                # under the write and would size the next one too long again.
                # The queries that did fit are at most the covered share of it.
                fit = q * append_s[wid]
                if isinstance(cover, (int, float)) and cover < 100 and n:
                    fit = min(fit, n * cover / 100)
                fits.append(fit)
            if len(qps) == len(pair):
                n = math.floor(min(fits) / W11_SPAN_MARGIN)
                # Halfway, in log, from what that pair ran to what its rate
                # says. The rate is not uniform over the append: strawmANN's
                # queries slow as the unindexed tail grows, so 0924's average
                # sized a search that would end in the append's first 8%, and
                # the undamped step from each night's short search to the
                # next's long one oscillated (4,840, 29,178, 6,918, 29,178).
                # The geometric mean converges on any monotone response.
                if sent:
                    n = math.floor(math.sqrt(n * min(sent)))
                n = max(n, math.ceil(max(qps) * workloads.MIN_ROW_S))
                out[env] = min(workloads.QUERIES, n)
        pair = previous_pair(pair)
    return out


def label_sized_w11(label: str) -> bool:
    """Whether the label's stamp records a sized mixed-row search."""
    try:
        h = json.loads((ROOT / "bench/results" / label / "run.json").read_text()).get("harness")
    except (OSError, json.JSONDecodeError):
        return False
    return bool((h or {}).get("w11_queries"))


def stamped_by_run_json(label: str, row: dict) -> bool:
    """Whether `row` was measured under the harness stamp `label`'s
    `run.json` holds now. A row without a hash predates row hashing and is
    taken as the stamp's, as `compare` takes it."""
    got = row.get("harness_hash")
    if not got:
        return True
    try:
        h = json.loads((ROOT / "bench/results" / label / "run.json").read_text()).get("harness")
    except (OSError, json.JSONDecodeError):
        return False
    return bool(h) and workloads.stamp_hash(h) == got


def previous_pair(labels: list[str]) -> list[str] | None:
    """The newest earlier `sm-`/`qd-` pair of the family these labels belong to.

    A family is a label with its date taken off: `sm-sift-perf-0924` and
    `sm-sift-perf-rel-0921` are both `sm-sift-perf`, and `-rel-` is how every
    published pair is spelled, so it is part of the date suffix and not of the
    family. Ordered by the trailing MMDD, not lexicographically: `rel-0903`
    sorted after `0910` on `r` > `0`, and the older pair won. A `-repN` pass
    directory does not end in a date and is not a pair. None for labels that
    do not follow the convention, or a family with no earlier pair.
    """
    dated = [re.fullmatch(r"(sm|qd)-(.+?)(?:-rel)?-\d{4}", x) for x in labels]
    if len(labels) != 2 or not all(dated):
        return None
    if {m.group(1) for m in dated} != {"sm", "qd"} or len({m.group(2) for m in dated}) != 1:
        return None
    stem = "sm-" + dated[0].group(2)
    # Strictly earlier than these labels, as the name says. "Newest other
    # than these" let a walk over pairs (`resolve_w11_queries`) step from the
    # newest pair to the second and back to the newest, and stop there.
    own = min(int(x[-4:]) for x in labels)
    found = []
    for d in (ROOT / "bench/results").glob(f"{stem}-*"):
        m = re.fullmatch(re.escape(stem) + r"(-rel)?-(\d{4})", d.name)
        if m and d.is_dir() and d.name not in labels and int(m.group(2)) < own:
            found.append((int(m.group(2)), d.name))
    if not found:
        return None
    prev = max(found)[1]
    return [prev, "qd-" + prev.removeprefix("sm-")]


#: Between asks of a busy host, under `--wait-for-gate`.
GATE_RETRY_S = 300


def wait_until_admitted(lax: bool, wait_min: float,
                        sleep=time.sleep, clock=time.monotonic) -> Gate:
    """Ask the ports and the §7.1 gate, and under `--wait-for-gate` ask again.

    An unattended run on a laptop that also builds things meets a busy host
    more often than not, and asking every five minutes is what gets it
    admitted. This was `nightrun`'s loop, which relaunched this whole process
    per ask and grepped its output for "refusing to run" to tell a gate
    refusal from any other failure. Asked here, before anything is built,
    there is nothing to tell apart. Every other failure still stops the run at
    once: a run that died mid-corpus must be read, not repeated.

    Returns the gate's own verdict, since `failed` outlives admission: under
    `--lax` a run proceeds on a failed gate and its render must say so.
    """
    deadline = clock() + wait_min * 60
    ask = 0
    while True:
        ask += 1
        if busy := ports_in_use():
            print(f"port(s) already in use: {', '.join(busy)}\n"
                  f"  the engine would bind, fail and be reported as "
                  f"`exited immediately`, which names neither the port nor the "
                  f"process. Free them and re-run.", file=sys.stderr)
        elif verdict := gate(lax):
            return verdict
        if clock() + GATE_RETRY_S > deadline:
            if wait_min:
                print(f"!! --wait-for-gate {wait_min:g}: gave up after {ask} ask(s)",
                      file=sys.stderr)
            return Gate(proceed=False, failed=True)
        print(f"--wait-for-gate: asking again in {GATE_RETRY_S // 60} min "
              f"(ask {ask})", flush=True)
        sleep(GATE_RETRY_S)


def iso(stamp: str) -> dt.datetime:
    """A `when` stamp as a datetime. They are written `...Z`, which
    `fromisoformat` only learned in 3.11 and which reads clearer named."""
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def planned_rows() -> list[str] | None:
    """The rows a pass will make, which `estimated_minutes` prices. None prices
    each basis arm as it ran."""
    return [w.id for w in workloads.table()]


def estimated_minutes(dataset: str, reps: int) -> tuple[float, str] | None:
    """Minutes of measurement passes, from the arms already run on this corpus.

    Printed before the run, not discovered during it:'s
    dbpedia-openai-1m run held the machine all night and nothing said so first.

    **Passes only.** The §8 differ and the render come after and are not in the
    figure, because nothing on disk records how long either took. The caller
    states that tail in words rather than pricing it with a constant nobody
    measured.

    Measured per engine, not pooled: the heaviest arm's row time (a pass is one
    arm of each, and Qdrant's dbpedia arm is 79 minutes of rows against
    strawmANN's 52), times that engine's smallest observed overhead ratio. Arms
    group by the engine `run.json` names, so `qd-sift-perf` and `qdrant` are both
    Qdrant.

    Folded labels count, and on a pruned machine they are all there is. Their
    row times are the per-row medians, which is one pass's worth; their wall
    clocks come from `run.json`'s `passes` rather than from the rows' own `when`
    stamps, for the reason given below.
    """
    per_engine: dict[str, tuple[float, str, list[float]]] = {}
    table_ids = planned_rows()
    for d in sorted(RESULTS.glob("*/rows.json")):
        try:
            meta = json.loads((d.parent / "run.json").read_text())
            if (meta.get("dataset") or {}).get("name") != dataset:
                continue
            rows = json.loads(d.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        engine = next((e for e in ("strawmann", "qdrant") if e in meta), None)
        if engine is None:
            continue

        def secs(r):
            return r.get("wall_s") or r.get("seconds") or 0

        row_s = sum(secs(r) for r in rows)
        stamped = sorted((r for r in rows if r.get("when")), key=lambda r: r["when"])
        if not row_s or len(stamped) < 2:
            continue
        # `when` is stamped when a row *starts*, so the span between the first
        # and last stamp is short by the last row -- which on this harness is a
        # sat-ladder rung, not a cheap one.
        last = secs(stamped[-1])
        if meta.get("rep_labels"):
            # A folded label is a basis too. `aggregate.py` folds the median per
            # row, so `row_s` is still one pass's row time -- but the `when`
            # stamps beside it are not one pass's, because each came from
            # whichever pass won that row's median, and the span across them is
            # therefore nobody's wall clock. `passes` holds the real ones, one
            # per pass, which is what it is written for.
            #
            # Skipping folded labels cost every estimate on this machine, for
            # every corpus: `prunable` removes the unfolded rep labels once they
            # are folded, so after the first prune the folded label is the only
            # record left that has measured the corpus at all, and this function
            # answered `unknown` in front of a nine-hour run.
            spans = [(iso(p["last_row"]) - iso(p["started"])).total_seconds() + last
                     for p in meta.get("passes") or []
                     if p.get("started") and p.get("last_row")]
            if not spans:
                # Folded before `passes` was recorded, or by a pass whose own
                # `started` was missing: no span, so no ratio to contribute.
                continue
        else:
            spans = [(iso(stamped[-1]["when"]) - iso(stamped[0]["when"])).total_seconds()
                     + last]
        # A span shorter than the rows it contains is not describing this pass --
        # rows copied between labels, or a hand-edited stamp. Dropped per span
        # rather than per label, so one bad pass does not cost the other two.
        spans = [s for s in spans if s >= row_s]
        if not spans:
            continue
        heaviest, label, ratios = per_engine.get(engine, (0.0, "", []))
        ratios.extend(s / row_s for s in spans)
        # The rows *this* run will make, priced from that arm. The table grew
        # from 32 to 43 rows between rel-0903 and perf-0924 and the estimate,
        # priced on the old arm's rows alone, came out three hours short
        # (findings 10). A row the basis never ran costs its mean row; a row it
        # ran that the table has since dropped costs nothing.
        by_id = {r["id"]: secs(r) for r in rows if r.get("id")}
        mean = row_s / max(len(by_id), 1)
        priced = row_s if table_ids is None else sum(by_id.get(i, mean) for i in table_ids)
        if priced > heaviest:
            heaviest, label = priced, d.parent.name
        per_engine[engine] = (heaviest, label, ratios)
    if not per_engine:
        return None

    arms = [(rows_s * min(ratios), label)
            for rows_s, label, ratios in per_engine.values()]
    # Only one engine has ever run here: the best guess for the other arm is
    # that it costs the same.
    per_pass = sum(a for a, _ in arms) * (1 if len(arms) == 2 else 2)
    basis = ", ".join(label for _, label in sorted(arms, reverse=True))
    return reps * per_pass / 60, basis


def ports_in_use() -> list[str]:
    """Which of the run's ports something already holds, and what holds them.

    strawmANN binds and exits on failure, which the harness reports as `strawmann
    exited immediately` over a log saying `error: SyscallFailed`. Neither says
    "another process owns 6334" — a container left by an interrupted
    run cost two full runs their first arm before anyone read a log.

    Checked before the gate, so a run refuses in a second rather than after
    building the engine and uploading a corpus. Our own leftover container is
    named and removed rather than reported; anything else is somebody's.
    """
    ours = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{QDRANT_CONTAINER}$"],
                          capture_output=True, text=True)
    if ours.returncode == 0 and ours.stdout.strip():
        print(f"  removing a leftover {QDRANT_CONTAINER} from an interrupted run",
              flush=True)
        sh(["docker", "rm", "-f", QDRANT_CONTAINER], 120)
        time.sleep(2)

    busy = []
    for port in (STRAWMANN_PORT, STRAWMANN_CONF_PORT, QDRANT_GRPC, QDRANT_REST):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind(("0.0.0.0", port))
            except OSError:
                who = subprocess.run(["ss", "-ltnp", f"sport = :{port}"],
                                     capture_output=True, text=True)
                detail = " ".join(who.stdout.split()[-1:]) if who.returncode == 0 else ""
                busy.append(f"{port}{' ' + detail if detail else ''}")
    return busy


def build_strawmann() -> bool:
    say("building strawmann (ReleaseFast)")
    code, out = sh(["zig", "build", "-Doptimize=ReleaseFast"], 900, cwd=ROOT)
    if code != 0:
        print(out.rstrip(), file=sys.stderr)
        return False
    print("  ok", flush=True)
    return True


def wipe_strawmann_storage() -> None:
    """Empty strawmANN's arena directory, for the reason Qdrant's is emptied.

    Qdrant's storage is wiped at the start of its arm; strawmANN's never was, and
    nothing noticed until the column turned from `unknown` into a number. It was
    then wrong in the section that matters: 7.50 GB against Qdrant's 1.41 GB, of
    which 3.07 GB was conformance arenas written hours earlier by another run on
    another corpus.

    Only the arena files this harness's collections produce, matched by suffix —
    the directory is a cache the operator may keep other things in.
    """
    d = paths.STRAWMANN_STORAGE
    if not d.exists():
        return
    freed = 0
    for f in d.glob("*.vectors.bin"):
        try:
            freed += f.stat().st_size
            f.unlink()
        except OSError as exc:
            print(f"  could not remove {f.name}: {exc}", flush=True)
    if freed:
        print(f"  wiped {freed / 1e9:.2f} GB of previous arenas from {d}", flush=True)


def start_strawmann(server_cpus: str, port: int, log: Path,
                    workers: int, capacity: int) -> subprocess.Popen | None:
    """Start the engine pinned to `server_cpus`.

    One I/O thread plus `workers` must fit the set, because `--cpus` refuses
    rather than wrapping around — a pool short by one used to mean a worker
    landed on a core the run had isolated *away* from the server.
    """
    argv = [
        str(ROOT / "zig-out/bin/strawmann"),
        "--port", str(port),
        "--capacity", str(capacity),
        # bfb opens ~threads x connections sockets; a server with fewer closes
        # the excess before the HTTP/2 preface and W4 reports a bare transport
        # error with no status.
        "--connections", "64",
        "--workers", str(workers),
        "--io-threads", "1",
        "--pin", "--cpus", server_cpus,
        # §7.4's one-residency rule. `cached` and `cold` are mappings and need
        # a file; `pinned` is anonymous RAM and must not be given one, or the
        # engine is measured mapping a store it would not have had.
        *(["--data-dir", str(paths.STRAWMANN_STORAGE),
           "--default-placement", str(PLACEMENT)]
          if workloads.Placement.pinned != PLACEMENT else []),
        # No `--no-bandwidth-probe`: the startup banner is the harness's
        # preferred record of the machine's bandwidth (`provenance.memory_
        # bandwidth` reads it from server.log and only falls back to running
        # `strawmann --probe` itself, from under the client's taskset). The
        # bus is saturated once either way, before any row runs; this way the
        # server measures it with the whole machine and the record says so.
    ]
    # In a scope of its own, as the Qdrant side already is: a process started
    # from a shell inherits the shell's cgroup, which here holds the editor and
    # the harness, so `cgroup_holds_only` correctly refuses its pressure
    # figures — Qdrant reported `psi_scope=engine` and strawmANN `shared` in the
    # same report, in a section comparing the two. Wiped before the engine maps
    # anything, so the storage figure is this run's arenas.
    if workloads.Placement.pinned != PLACEMENT:
        wipe_strawmann_storage()
    # Here, not at each call site, because opening the log is here: the per-pass
    # caller made the directory and the conformance caller did not, and nothing
    # creates the merged label's until a row has run.
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    argv = maybe_scope(argv, STRAWMANN_SCOPE)
    p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    for _ in range(60):
        time.sleep(0.5)
        if p.poll() is not None:
            print(f"strawmann exited immediately; see {log}", file=sys.stderr)
            print(log.read_text()[-2000:], file=sys.stderr)
            return None
        if "listening" in log.read_text() or "bind" in log.read_text():
            break
    time.sleep(1)
    print(f"  strawmann pid {p.pid} on cpus {server_cpus}, port {port}", flush=True)
    return p


#: `collection.logGraphQuality`'s line, which the engine prints after every
#: build publishes. Parsed rather than re-derived: the alternative is a second
#: implementation of reachability in Python over a graph the harness cannot see.
#: `seed` is optional so a log written before the engine printed it still
#: parses: an older arm reads as "not recorded" rather than failing to load.
_GRAPH_LINE = re.compile(
    r"index: graph checksum=(?P<checksum>[0-9a-f]+) nodes=(?P<nodes>\d+) "
    r"unreachable=(?P<unreachable>\d+) in_degree_zero=(?P<in_degree_zero>\d+) "
    r"unreachable_with_out_edges=(?P<unreachable_with_out_edges>\d+)"
    r"(?: seed=0x(?P<seed>[0-9a-f]+))?")


def _graph_field(key: str, value: str) -> object:
    """One field of a graph line, typed.

    `checksum` and `seed` stay hex strings: both are identities rather than
    quantities, and an int would print the seed in a base nobody configured it
    in. Everything else is a count.
    """
    return value if key in ("checksum", "seed") else int(value)


def record_graph_quality(label: str) -> None:
    """Copy the engine's per-build graph statistics into this arm's `run.json`.

    §8.7 accepts that `buildParallel`'s graph depends on thread interleaving, and
    findings 34 measured what that is worth: three builds of one collection
    spreading 0.0062 of recall@10 at `ef` 512, wider than the gap between the two
    engines. It could only infer the mechanism, because nothing recorded the graph.

    Now the engine prints one line per published build and this lifts them into
    the record in build order, so a repeated run can put each pass's graph beside
    that pass's recall. Never fails the arm: this is provenance, and a missing
    line reads as "not recorded" rather than as zero.
    """
    log = RESULTS / label / "server.log"
    meta = RESULTS / label / "run.json"
    if not log.exists() or not meta.exists():
        return
    try:
        builds = [
            {k: _graph_field(k, v) for k, v in m.groupdict().items() if v is not None}
            for m in _GRAPH_LINE.finditer(log.read_text(errors="replace"))
        ]
        if not builds:
            return
        doc = json.loads(meta.read_text())
        doc["graph_builds"] = builds
        meta.write_text(json.dumps(doc, indent=2) + "\n")
    except (OSError, ValueError):
        return
    worst = max(builds, key=lambda b: b["unreachable"])
    seeds = {b["seed"] for b in builds if b.get("seed")}
    # Named in the line because §8.7 asks for the seed as provenance (six
    # builds at four seeds sit within 0.00003 of recall@10 under the engine's
    # draw, `decisions.md`). More than one means two collections were
    # built at different seeds, which no current path does and which would
    # make their recall curves incomparable.
    seed = f", seed 0x{seeds.pop()}" if len(seeds) == 1 else (
        f", {len(seeds)} DIFFERENT SEEDS" if seeds else ", seed not recorded")
    print(f"  graph: {len(builds)} build(s) published, worst "
          f"{worst['unreachable']:,} of {worst['nodes']:,} unreachable{seed}", flush=True)


def finish_strawmann_arm(p: subprocess.Popen | None, label: str) -> None:
    """Stop the engine, read its graph builds off the finished log, and free
    its arenas now rather than at the next strawmANN arm's start.

    Nothing reads them once the arm is over, and kept they shared the disk with
    the Qdrant arm that follows: on dbpedia-openai-1m 45.7 GB of arenas beside
    Qdrant's 65.5 GiB W11-steady peak is about 110 GB, on a host with 86 GB free.
    """
    stop_strawmann(p)
    # After the engine exits, so the log is complete.
    record_graph_quality(label)
    if workloads.Placement.pinned != PLACEMENT:
        wipe_strawmann_storage()


def finish_qdrant_arm() -> None:
    """Stop Qdrant and free its storage, for the same reason as strawmANN's."""
    stop_qdrant()
    wipe_qdrant_storage()


def stop_strawmann(p: subprocess.Popen | None) -> None:
    if p is None:
        return
    # The scope first: with one, `p` is `systemd-run` and terminating it is not
    # reliably the same as stopping the engine it placed in a cgroup.
    stop_scope(STRAWMANN_SCOPE)
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


# -------------------------------------------------------------------------
# Qdrant
# -------------------------------------------------------------------------

#: The natively-started Qdrant, when `--qdrant-binary` is in use.
QDRANT_PROC: subprocess.Popen | None = None

#: The transient systemd scopes the engines run in, so each one's cgroup holds
#: that engine and nothing else — which is what makes its PSI columns its own.
#: Named, so a leftover from an interrupted run is findable and stoppable
#: rather than anonymous.
QDRANT_SCOPE = "strawmann-fullrun-qdrant"
STRAWMANN_SCOPE = "strawmann-fullrun-engine"


def maybe_scope(argv: list[str], unit: str) -> list[str]:
    """`argv` wrapped in a transient systemd scope, where that is available.

    Falls through unchanged without a user manager — a container, a CI runner,
    a machine without systemd. The run then measures everything except
    pressure, and the report says which kind of blank that is rather than
    reporting a zero.
    """
    if shutil.which("systemd-run") and os.environ.get("XDG_RUNTIME_DIR"):
        stop_scope(unit)
        return ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", *argv]
    return argv


def stop_scope(unit: str) -> None:
    """Stop a transient scope by name, including one an interrupted run left."""
    if shutil.which("systemctl"):
        sh(["systemctl", "--user", "stop", f"{unit}.scope"], 60)


def stop_qdrant() -> None:
    global QDRANT_PROC
    if QDRANT_PROC is not None:
        p, QDRANT_PROC = QDRANT_PROC, None
        # The scope first, where there is one: `p` is then `systemd-run`, and
        # terminating it is not reliably the same as stopping the engine it
        # placed in a cgroup. `systemctl stop` is.
        stop_scope(QDRANT_SCOPE)
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=30)
        return
    sh(["docker", "rm", "-f", QDRANT_CONTAINER], 120)


def wipe_qdrant_storage() -> bool:
    """Empty the storage directory from inside a container.

    The image runs as root, so everything it writes is root-owned and a
    user-level delete fails partway through with EACCES, leaving a half-wiped
    directory the next arm would inherit. `qdrant_ab.py` learned this the hard
    way and the reasoning is copied rather than re-derived.
    """
    QDRANT_STORAGE.mkdir(parents=True, exist_ok=True)
    # A native Qdrant writes as this user, so a plain delete works without
    # needing Docker. But the directory is shared with the container path and
    # anything a previous containerised run left is root-owned, so this is the
    # fast path and not the only one — the first native SIFT run died on
    # `Permission denied: .../qdrant-storage/aliases`.
    plain_ok = True
    if QDRANT_BINARY is not None:
        for entry in QDRANT_STORAGE.iterdir():
            try:
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            except OSError:
                plain_ok = False
                break
    if QDRANT_BINARY is None or not plain_ok:
        if not shutil.which("docker"):
            print(f"{QDRANT_STORAGE} holds files this user cannot delete and "
                  f"docker is not available to do it as root; wipe it by hand",
                  file=sys.stderr)
            return False
        code, out = sh(["docker", "run", "--rm", "-v", f"{QDRANT_STORAGE}:/s", "alpine:3",
                        "sh", "-c", "rm -rf /s/..?* /s/.[!.]* /s/* 2>/dev/null; true"], 300)
        if code != 0:
            print(f"could not wipe {QDRANT_STORAGE}:\n{out}", file=sys.stderr)
            return False
    leftover = list(QDRANT_STORAGE.iterdir())
    if leftover:
        print(f"{QDRANT_STORAGE} still holds {[p.name for p in leftover[:5]]}; "
              f"refusing, the next arm would inherit another run's segments",
              file=sys.stderr)
        return False
    return True


def start_qdrant_binary(server_cpus: str, grpc: int, rest: int,
                        wipe: bool = True) -> str | None:
    """Start a Qdrant *binary* on the same cores, instead of the container.

    Same configuration expressed natively: `taskset` for `--cpuset-cpus`,
    `QDRANT__` variables for what the image took through `-e`, and no port to
    publish because there is no Docker proxy in the path.

    It exists because a container cannot be counted — `perf stat -p` on it fails
    (it runs as root, we do not, and `perf_event_paranoid` grants ptrace at no
    value), so `--perf` needs a binary to get two columns instead of one.

    The trade is provenance: §8.9 pins the target by digest and a path is not one,
    so `provenance.qdrant_build` records the sha256 and the version the server
    reports. A run that does not need counters should keep the image.

    Run from the root of its own checkout, because Qdrant reads
    `config/config.yaml` relative to the working directory — without that it starts
    on built-in defaults and a storage path the harness never wipes.

    `wipe=False` keeps the storage for a restart of the same binary within one
    experiment, which reloads its segments instead of paying a full ingest:
    `qdrant_w4_probe.py` restarts it per arm to change one setting.
    """
    global QDRANT_PROC
    if QDRANT_BINARY is None or not QDRANT_BINARY.exists():
        print(f"no qdrant binary at {QDRANT_BINARY}", file=sys.stderr)
        return None
    stop_qdrant()
    # A leftover from an interrupted run, which `stop_qdrant` cannot reach
    # because its Popen handle died with the process that held it. The
    # container path has `docker rm -f <name>` for exactly this; the native
    # path needs the scope stopped by name, or `await_qdrant` polls the REST
    # port, gets an answer from the *old* engine, and reports a successful
    # start of a process that never started. Found by doing it: a second dry
    # run "succeeded" against the first one's server.
    stop_scope(QDRANT_SCOPE)
    stale = [p for p in ports_in_use() if str(rest) in p or str(grpc) in p]
    if stale:
        print(f"something still holds qdrant's ports: {'; '.join(stale)}",
              file=sys.stderr)
        print("  refusing rather than measuring whatever answers there",
              file=sys.stderr)
        return None
    if wipe and not wipe_qdrant_storage():
        return None

    cwd = Path(os.environ.get("QDRANT_CWD") or QDRANT_BINARY.parent.parent.parent)
    if not (cwd / "config" / "config.yaml").exists():
        print(f"  !! no config/config.yaml under {cwd}; qdrant will use its "
              f"built-in defaults. Set $QDRANT_CWD to its checkout root if the "
              f"storage path or ports come out wrong.", flush=True)
    env = {
        **os.environ,
        # Qdrant's `settings.rs` defaults RUN_MODE to `development` and layers
        # `config/development.yaml` over the base config when run from a
        # checkout: `max_search_threads: 4`, audit logging of every request,
        # DEBUG logs, every feature flag. The Docker image sets `production`.
        # Unset, every native pair to 2026-09-25 measured that profile, and its
        # 4-thread search cap was Qdrant's 4.46 busy cores on W4 (findings 5).
        "RUN_MODE": "production",
        "QDRANT__STORAGE__STORAGE_PATH": str(QDRANT_STORAGE),
        **qdrant_segment_env(),
        "QDRANT__SERVICE__HTTP_PORT": str(rest),
        "QDRANT__SERVICE__GRPC_PORT": str(grpc),
        "QDRANT__TELEMETRY_DISABLED": "true",
    }
    log = RESULTS / "qdrant-server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("wb")
    # In a cgroup of its own, so its PSI columns are the engine's and not the
    # operator's session. Docker gives every container a scope for free; a
    # binary from a shell inherits the shell's, so without this the native path
    # would trade PSI for perf rather than simply gaining perf. Falls back to a
    # bare `taskset` without systemd's user manager, and the report says which
    # kind of blank that is.
    argv = maybe_scope(["taskset", "-c", server_cpus, str(QDRANT_BINARY)], QDRANT_SCOPE)
    print(f"  {' '.join(argv)}", flush=True)
    QDRANT_PROC = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                   cwd=str(cwd), env=env)
    return await_qdrant(server_cpus, grpc, rest, f"binary {QDRANT_BINARY}")


def await_qdrant(server_cpus: str, grpc: int, rest: int, what: str) -> str | None:
    """Poll the REST root until it answers, and report what version did.

    Shared by both start paths: the readiness condition is the same question
    whichever way the process was started, and the version is read from the
    server rather than from the tag or the path, because both of those are
    claims and this is the answer.
    """
    import urllib.error
    import urllib.request
    deadline = time.time() + 180
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{rest}/", timeout=5) as r:
                got = json.loads(r.read())
                v = str(got.get("version", "unknown"))
                print(f"  qdrant {v} on cpus {server_cpus}, grpc {grpc} ({what})",
                      flush=True)
                return v
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            last = str(e)
            time.sleep(2)
    stop_qdrant()
    print(f"{what} did not answer within 180s ({last})", file=sys.stderr)
    return None


def start_qdrant(server_cpus: str, grpc: int, rest: int) -> str | None:
    """Start Qdrant on the same cores strawmANN was given.

    `--cpuset-cpus` is why this exists rather than `qdrant_ab.start`: unpinned, the
    container may get this host's 5.16 GHz cores while strawmANN had the 3.29 GHz
    ones, and the ratio measures the scheduler.

    `--network host` rather than `-p`, because publishing a port puts Docker's
    userland proxy in the request path. It is not part of Qdrant, not covered by
    `--cpuset-cpus`, and its CPU is charged to neither engine — 5.13 s inside
    twelve minutes, all in the traffic rows, against a 2% noise band — while
    strawmANN is a native process bfb connects to directly. Host networking also
    means the container binds Qdrant's own defaults, so `rest` and `grpc` describe
    where it will be rather than where it is mapped to.

    `--segments 1` is §2's documented gap, set through the env var the image reads
    so it lands in the container's own config.
    """
    if QDRANT_BINARY is not None:
        return start_qdrant_binary(server_cpus, grpc, rest)
    # Host networking ignores port mapping, so a caller asking for anything but
    # the image's own ports would get a container listening somewhere else and
    # a readiness probe that never connects. Say so here rather than time out.
    if (rest, grpc) != (6333, 6334):
        print(f"--network host binds qdrant's defaults 6333/6334, not "
              f"{rest}/{grpc}", file=sys.stderr)
        return None

    stop_qdrant()
    if not wipe_qdrant_storage():
        return None
    argv = [
        "docker", "run", "-d", "--name", QDRANT_CONTAINER,
        "--cpuset-cpus", server_cpus,
        "--network", "host",
        "-v", f"{QDRANT_STORAGE}:/qdrant/storage",
        *(x for k, v in qdrant_segment_env().items() for x in ("-e", f"{k}={v}")),
        QDRANT_IMAGE,
    ]
    code, out = sh(argv, 300)
    if code != 0:
        print(f"could not start {QDRANT_IMAGE}:\n{out}", file=sys.stderr)
        return None

    # What the server says it is, not what the tag claims: a tag is mutable and
    # the reported version is what actually ran.
    return await_qdrant(server_cpus, grpc, rest, QDRANT_IMAGE)


# -------------------------------------------------------------------------
# The measured phases
# -------------------------------------------------------------------------

#: §4's open-loop arms are a fraction of measured saturation, and taken per
#: The hardware-counter set every row is measured with, or None. Set from
#: `--perf`; threaded into each arm's `workloads.py run` so both arms carry it.
PERF_SET: str | None = None

#: Which of the two Qdrant experiments this run is. Set from
#: `--segment-policy` via `workloads.use_segment_policy`, which also exports it
#: so every arm's subprocess creates its collections the same way.
SEGMENT_POLICY: str = str(workloads.SegmentPolicy.equal_work)

#: engine that is two different offered loads. Set from `--rps-reference`, it
#: pins one saturation for both arms so they are offered the same absolute
#: rate — which is what "a load both engines can serve" means, and what lets
#: the report compare their percentiles instead of refusing to. Module-level
#: because it is one setting for the whole run, read where the argv is built,
#: rather than a parameter threaded through `measure` that every caller would
#: have to remember to pass.
RPS_REFERENCE: float | None = None

#: The dataset every phase of this run reads, set from `--dataset` in `main`
#: and exported as `$STRAWMANN_DATASET` so `workloads.py` and `recall.py` —
#: which this spawns — agree without being passed the flag again. Both arms and
#: the conformance row get the same one by construction: a run comparing two
#: engines on two corpora is not a comparison, and there is no flag here that
#: can produce one.
DATASET = "sift1m"


#: One id for this invocation, written into both arms' `run.json` and onto
#: every row they measure. Both arms must be one session for their ratio to
#: mean anything, and nothing recorded that: `run.json` is rewritten by an
#: arm's last invocation while `rows.json` merges in place, so
#: `qd-dbp100k` held rows measured 05:49Z under a `run.json` claiming 17:46Z,
#: against an arm measured at 17:30Z. Same corpus, so `foreign_dataset` cannot
#: see it — which is exactly when the merge is silent.
#:
#: A resumed arm inside one invocation keeps this id, which is why it belongs
#: here and not to `workloads.py`: resume is a feature, re-running an arm half
#: a day later is not.
SESSION = f"{int(time.time())}-{os.getpid()}"


def run_workloads(uri: str, label: str, client_cpus: str, storage: str | None,
                  only: list[str] | None = None, placement: str | None = None) -> int:
    """§4's rows, with the load generator kept off the server's cores.

    `taskset` on the harness rather than on bfb itself, because the harness
    spawns bfb and affinity is inherited. A load generator sharing cores with
    the engine measures the two competing, and §7.4's open-loop rows are
    already sensitive enough to the client's own capacity (findings 15).
    """
    # Every invocation re-runs the §7.1 gate and rewrites `env.txt`, so every
    # invocation needs the machine to have gone quiet first — not just the
    # first one of an arm. See `settle`.
    settle(f"{label} rows")
    argv = ["taskset", "-c", client_cpus,
            str(ROOT / "bench/harness/workloads.py"), "run", uri, label,
            # A row under `MIN_ROW_S` measures bfb's ramp as much as the
            # engine. `workloads.py` has had the re-run since it had
            # `MIN_ROW_S`, but the sequence producing every published table
            # never asked for it, so W10-ef32/ef64 and W13 shipped short on
            # with a caveat instead. Not a knob here: one command
            # means no mode that publishes a ramp measurement.
            "--min-duration"]
    # `--min-duration`'s verdict, shared by every pass of this arm. Without it
    # a row whose duration straddles `MIN_ROW_S` settles at 2x in one pass and
    # 4x in another, and `aggregate.fold` then publishes a median of two
    # configurations under one `reps: 3` stamp — measured on W12 of
    # qd-dbp1m-perf-rel-0903,. Keyed on the arm's base label, so the
    # two engines pin independently: the slower one's rows are already long
    # enough and should not inherit the faster one's escalation.
    argv += ["--n-pin", str(RESULTS / re.sub(r"-rep\d+$", "", label) / "n_factors.json")]
    if storage:
        argv += ["--storage", storage]
    if RPS_REFERENCE:
        argv += ["--rps-reference", str(RPS_REFERENCE)]
    if placement:
        argv += ["--memory-vectors", placement]
    # §7.3's hardware counters, asked of both arms or of neither: a row measured
    # with `perf stat` attached and one measured without are not the same
    # experiment, and the report banners a pair that disagrees.
    if PERF_SET:
        argv += ["--perf", PERF_SET]
    argv += ["--session", SESSION]
    if only:
        argv += only
    print(f"  {' '.join(argv)}", flush=True)
    r = subprocess.run(argv, cwd=ROOT)
    return r.returncode


#: §5.5's residency, asked of *both* arms, because a ratio across two
#: residencies measures the residency. The run refuses if they do not end up
#: there.
#:
#: `cached` is the only one both can serve: v1.19.0 answers `pinned` with
#: "not supported for dense vector storage", and strawmANN needs a `--data-dir`
#: to map. The cost is that strawmANN is not measured at its own default, so
#: this says nothing about what a user gets out of the box —
#: `--placement pinned --skip qdrant` is that measurement, and the difference
#: between the two bounds the confound.
PLACEMENT = workloads.Placement.cached

#: Residencies each engine can actually serve, measured rather than assumed.
#: `preflight_placement` refuses in a second rather than after an hour.
_SERVES = {
    "strawmann": {workloads.Placement.pinned, workloads.Placement.cached,
                  workloads.Placement.cold},
    "qdrant": {workloads.Placement.cached, workloads.Placement.cold},
}


def preflight_placement(placement: str, engines: list[str]) -> str | None:
    """Why `placement` cannot be the run's shared residency, or None.

    Checked before the gate, because the alternative is discovering it when the
    second arm's first collection is refused, an hour in.
    """
    bad = [e for e in engines if placement not in _SERVES.get(e, set())]
    if not bad:
        return None
    return (f"{', '.join(bad)} cannot serve `{placement}` for dense vectors, and "
            f"§7.4 compares at one residency or not at all.\n"
            f"  servable by both: "
            f"{', '.join(sorted(set.intersection(*(_SERVES[e] for e in engines))))}")


def placement_mismatch(labels: list[str]) -> str | None:
    """Why the arms did not end up at the same residency, or None.

    The request is not the fact — the same lesson `--segments 1` taught, where
    Qdrant treated a request as a target and missed it. This reads
    `collections.json` back and compares what the engines *said* they served
    from, per collection, after both arms have run.
    """
    seen: dict[str, set] = {}
    for label in labels:
        f = RESULTS / label / "collections.json"
        if not f.exists():
            continue
        try:
            doc = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        seen[label] = {c.get("placement") for c in doc.get("collections") or []
                       if c.get("collection")}
    if len(seen) < 2:
        return None
    flat = {p for ps in seen.values() for p in ps}
    if len(flat) == 1 and None not in flat:
        return None
    detail = "; ".join(f"{lab}: {', '.join(str(p) for p in sorted(ps, key=str))}"
                       for lab, ps in seen.items())
    return (f"the arms did not serve from one residency ({detail}). A ratio "
            f"between two residencies measures the residency.")


#: The collection §8.9's recall@100 is measured on: the fp32 one every
#: unquantized search row queries. The quantized collections get the default
#: sweep only — their §8.9 row is about quantization fidelity (T4), and a
#: hundred-deep recall on a lossy encoding answers a question nobody asked.
RECALL_100_COLLECTION = "bench2"


def run_recall(uri: str, label: str, client_cpus: str) -> int:
    """§7.4's other half: a qps without its recall is not publishable.

    Two passes. The first is the sweep the W10 rows join on `ef`, at limit 10. The
    second is §8.9's `recall@100`, which was a row field, a sink column, and null
    in all 64 rows ever ingested because nothing asked for more than ten results.

    Two passes rather than one wider one because the engines resolve width as
    `max(ef, limit)`: at limit 100 the nominal ef=32 and ef=64 both search at 100.
    `recall.py` drops those two and writes its own `.k100` file.
    """
    rc = 0
    # `bench12` holds a prefix of the corpus, so a filtered grade's restricted
    # truth has to be built over that prefix. Passed from the table rather than
    # re-derived in `recall.py`: `workloads` owns `W12_N` and its corpus cap,
    # and a second derivation is a second definition that can drift.
    prefix = ["--filtered-base-n", str(workloads.w12_n())]
    for extra in ([*prefix],
                  ["--collections", RECALL_100_COLLECTION, "--limit", "100"]):
        argv = ["taskset", "-c", client_cpus,
                str(ROOT / "bench/harness/recall.py"), label, "--engine", uri, *extra]
        print(f"  {' '.join(argv)}", flush=True)
        rc |= subprocess.run(argv, cwd=ROOT).returncode
    return rc


def qdrant_target() -> str:
    """What the header line says this run is about to measure.

    `QDRANT_IMAGE` was printed unconditionally, so every `--qdrant-binary` run —
    which is every `--perf` run — announced `qdrant/qdrant:v1.19.0` and then
    measured a native build reporting 1.19.1-dev. `run.json` and the report had it
    right all along; the wrong one was the single line a reader watches while the
    hours pass.

    The version cannot be printed here — it comes from the server, which has not
    started — so this names the path that was pinned and says where the version
    will come from.
    """
    if QDRANT_BINARY is not None:
        return f"{QDRANT_BINARY} (native; version read from the server at startup)"
    return f"{QDRANT_IMAGE} (container, pinned by digest)"


def build_identity(qdrant_version: str | None) -> dict:
    """What §8 binds a perf row to: "the same build, on the same data".

    The differ reads these from the environment and defaults them to
    `"unknown"`/`"unpinned"`. Nothing set them, so every conformance hash
    identified its dataset and its outcomes but not the builds that produced
    them — which is the first half of §8's sentence.
    """
    env = dict(os.environ)
    commit = _run_out(["git", "rev-parse", "--short=12", "HEAD"]) or "unknown"
    if provenance.tree_dirty():
        # A dirty tree is a different build and must hash differently, or two
        # runs from one commit with different working trees collide.
        # `provenance.tree_dirty` rather than a second `git status`: this asked
        # over the whole tree while the perf rows asked over the build inputs,
        # so §8 once refused a whole dataset over a disagreement about
        # README.md.
        commit = f"{commit}-dirty"
    env["STRAWMANN_COMMIT"] = commit
    env["STRAWMANN_ISA_BUILD"] = os.environ.get("STRAWMANN_ISA_BUILD", "native")
    if qdrant_version:
        env["QDRANT_VERSION"] = qdrant_version
    return env


def _run_out(argv: list[str]) -> str | None:
    code, out = sh(argv, 30)
    return out.strip() if code == 0 else None


#: Rows that mutate a collection the recall sweeps read, so the sweeps run
#: first. W11 appends to `bench2`, which then no longer matches a ground truth
#: describing exactly `UPLOAD_N`, and `recall.py` refuses that outright (§4.3).
#: Before the guard existed a `bench2` of 70,000 against a ground truth of
#: 1,000,000 produced a confident recall@10 of 0.1132.
#:
#: Derived, not a literal list: as `["W11"]` it missed `W11-steady`, which
#: appends to `bench2` for the same reason. The property is not the id, it is
#: that the row runs a concurrent writer, and `Workload.background` says so.
def mutating_rows() -> list[str]:
    """Row ids that write to a collection while other phases read it."""
    return [w.id for w in workloads.table() if w.background]


#: The §7.1 gate calls the machine quiescent at or under 10% per core. Settling
#: aims *below* that: releasing at the threshold itself leaves no headroom, and
#: the gate then runs a few seconds later on the wrong side of it. Measured:
#: settle released at exactly 10% and `workloads.py` gated at 13%.
SETTLE_TARGET_PCT = 6
#: `load1` is a one-minute decaying average, so straight after a heavy phase it
#: describes work that has already finished. Where it has stopped falling,
#: waiting longer buys nothing — that is the machine's baseline, and this box
#: idles at 5-9% with a desktop on it. Three consecutive readings within this
#: relative distance of each other count as a plateau.
SETTLE_PLATEAU_REL = 0.05
SETTLE_PLATEAU_POLLS = 3
SETTLE_TIMEOUT_S = 300
SETTLE_POLL_S = 10


def foreign_load() -> list[str]:
    """Foreign load the gate would actually refuse, named, or nothing.

    The gate's instrument, not the load average: two `/proc` samples a second
    apart, everything the harness owns excluded. `settle` uses it so waiting and
    being judged use the same measurement.

    They did not, one layer down. This returned `busy_processes`, the operator's
    *list*, thresholded at `FOREIGN_PCT` (5% of one core, deliberately low so load
    made of several small processes gets named); the gate's verdict is
    `foreign_cores` against a whole core summed. So settle waited on things the
    gate passes without comment — `Xorg(6%)` and an editor — burning the full
    timeout before every gate re-run, twice per arm, on a machine about to be
    called quiescent. Verdict first, list second.

    Empty when it cannot look: `settle` only decides how long to wait, and the
    gate fails loudly on the same condition.
    """
    try:
        before = setup._cpu_seconds()
        time.sleep(setup.QUIESCENT_SAMPLE_S)
        after = setup._cpu_seconds()
        if setup.foreign_cores(before, after,
                               setup.QUIESCENT_SAMPLE_S) <= setup.FOREIGN_BUDGET_CORES:
            return []
        return setup.busy_processes(before, after, setup.QUIESCENT_SAMPLE_S)
    except Exception:
        return []


def settle(what: str) -> None:
    """Wait for the load average to describe now, before the gate is asked.

    `workloads.py` re-runs the gate on every invocation and overwrites the arm's
    `env.txt`, and `measure` invokes it twice per arm — the second straight after
    the recall sweeps, the heaviest phase. So the stamp that survives is the one
    taken at the worst moment: that put `FAIL ... load 4.28` into the
    stamp of an arm whose own gate had passed at 1.09.

    Hence both constants above: settle to a target *below* the gate's threshold so
    the gate has headroom, and treat a load that has stopped falling as settled,
    because `load1` decays over a minute and its floor is whatever the machine
    idles at.

    Called from `run_workloads` so every gate re-run gets it. A timeout rather
    than a loop forever: if the box will not go quiet that is the gate's business
    to report.
    """
    cores = os.cpu_count() or 1
    t0 = time.monotonic()
    deadline = t0 + SETTLE_TIMEOUT_S
    recent: list[float] = []
    announced = False
    while True:
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
        pct = int(load1 / cores * 100)
        if pct <= SETTLE_TARGET_PCT:
            # Low average is not quiet: the gate fails on any single foreign
            # process over 20% of a core, which moves a twelve-core average by
            # 2.4% — under this threshold, so the fast path returned and the
            # gate then refused the run on a process settle never looked for
            # (`NOT quiescent: claude(29%)` at load 0.35). Both conditions, and
            # the same instrument the gate uses for the one that matters.
            if not (busy := foreign_load()):
                if announced:
                    print(f"  settled at load {load1} ({pct}% per core) after "
                          f"{time.monotonic() - t0:.0f} s", flush=True)
                return
            if not announced:
                print(f"  waiting for {' '.join(busy)} before {what}: the load "
                      f"average is quiet ({pct}% per core) and the §7.1 gate "
                      f"would still fail", flush=True)
                announced = True

        recent.append(load1)
        recent = recent[-SETTLE_PLATEAU_POLLS:]
        if len(recent) == SETTLE_PLATEAU_POLLS:
            lo, hi = min(recent), max(recent)
            if hi <= 0 or (hi - lo) / hi <= SETTLE_PLATEAU_REL:
                # A plateau is the machine's floor only if nothing foreign is
                # holding it up: one process steadily burning a fifth of a core
                # plateaus the average exactly like an idle desktop, and the
                # gate samples per process. Settle once released three arms on a
                # "floor" that was an editor at 20%, and the failing verdict
                # went into six rows' stamps. So the plateau test asks the
                # gate's own instrument before believing itself.
                if not (busy := foreign_load()):
                    print(f"  load has stopped falling at {load1} ({pct}% per core) "
                          f"after {time.monotonic() - t0:.0f} s; that is this "
                          f"machine's floor, measuring {what}", flush=True)
                    return
                if not announced:
                    print(f"  waiting for {' '.join(busy)} before {what}: a plateau "
                          f"at {load1} ({pct}% per core) that the §7.1 gate would "
                          f"fail on", flush=True)
                    announced = True

        if time.monotonic() >= deadline:
            why = f"load {load1} ({pct}% per core)"
            if busy := foreign_load():
                why += f", {' '.join(busy)}"
            print(f"!! still {why} after {SETTLE_TIMEOUT_S}s; measuring {what} "
                  f"anyway — the gate will record it", flush=True)
            return
        if not announced:
            print(f"  waiting for the machine to go quiet before {what}: "
                  f"load {load1} ({pct}% per core, want <= {SETTLE_TARGET_PCT}%)",
                  flush=True)
            announced = True
        time.sleep(SETTLE_POLL_S)


#: Keys under which a collection record carries a *second* read-back of itself
#: rather than another collection. Preserved across merges, and skipped by
#: anything walking a record's fields.
NESTED_CAPTURES = ("after_mutating_rows",)


def parse_segment_telemetry(doc: dict) -> dict[str, list[dict]]:
    """Per collection, what each of Qdrant's segments holds, from `/telemetry`.

    gRPC's `CollectionInfo` gives `segments_count` and nothing per segment, so
    "held 2 segments" could not say whether the second was a graph or the empty
    appendable Qdrant keeps for writes: `indexed_vectors_count ==
    points_count` made it likely, the files did not prove it, and the
    equal-work note had to call the count an upper bound. REST telemetry lists
    every segment with its points, which is the fact the note needs.
    """
    out: dict[str, list[dict]] = {}
    colls = ((doc.get("result") or {}).get("collections") or {}).get("collections") or []
    for c in colls:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        segs = []
        for shard in c.get("shards") or []:
            for seg in ((shard or {}).get("local") or {}).get("segments") or []:
                info = (seg or {}).get("info") or {}
                segs.append({"points": info.get("num_points"),
                             "indexed": info.get("num_indexed_vectors"),
                             "type": info.get("segment_type"),
                             "appendable": info.get("is_appendable")})
        out[c["id"]] = segs
    return out


def qdrant_segments(uri: str) -> dict[str, list[dict]]:
    """`parse_segment_telemetry` for the engine at `uri`, or {} when it is not
    Qdrant or the REST port does not answer. The arm is addressed on its gRPC
    port; telemetry is on the REST one beside it."""
    import urllib.parse
    import urllib.request
    u = urllib.parse.urlparse(uri)
    if u.port != QDRANT_GRPC:
        return {}
    try:
        with urllib.request.urlopen(
                f"http://{u.hostname}:{QDRANT_REST}/telemetry?details_level=10",
                timeout=30) as r:
            return parse_segment_telemetry(json.loads(r.read()))
    except (OSError, ValueError):
        return {}


def capture_collections(uri: str, label: str, client_cpus: str,
                        names: list[str] | None = None,
                        into: str | None = None) -> int:
    """Read back what the engine actually built, while it is still up.

    §4 stamps the settings the harness *sent* and §8.9 pins the image by digest;
    neither reads anything back, so `--segments 1` was a request in the record
    rather than a fact — and for Qdrant it sets `default_segment_number`, which
    the optimizer treats as a target, making the segment count the largest single
    confound in the `ef` comparison.

    `into` nests the read-back under a key instead of replacing it, for a second
    look at a collection whose first look is the one the report should show. Must
    run before the engine stops, so it sits at the end of the arm.
    """
    out = RESULTS / label / "collections.json"
    names = names or workloads.upload_collections()
    tmp = out.with_suffix(".part.json")
    say(f"{label}: reading back the collection config for {', '.join(names)}")
    code, text = sh(["taskset", "-c", client_cpus,
                     "cargo", "run", "--release", "--quiet", "--",
                     "collection-info", "--engine", uri, "--label", label,
                     # Every collection any row creates, from the table rather
                     # than from a list kept in step by hand. The default was
                     # `bench0,bench2,bench6,bench7,bench8`, so `bench1` and
                     # `bench12` were simply absent from the report — and
                     # `bench12` is the one W12's numbers depend on.
                     "--collections", ",".join(names),
                     "--json", str(tmp)], 300, cwd=CONF)
    print(text.rstrip(), flush=True)
    if code != 0:
        print(f"!! could not read {label}'s collection config; the report will "
              f"say so rather than assume the requested settings held", flush=True)
        tmp.unlink(missing_ok=True)
        return 0
    # Qdrant's segments, one by one, beside the count gRPC gives. Absent for
    # strawmANN, which holds one graph per collection by construction.
    segs = qdrant_segments(uri)
    if segs:
        try:
            doc = json.loads(tmp.read_text())
            for c in doc.get("collections", []):
                got = segs.get((c or {}).get("collection"))
                if got is not None:
                    c["segments"] = got
                    c["populated_segments_count"] = sum(1 for g in got if g.get("points"))
            tmp.write_text(json.dumps(doc, indent=2) + "\n")
        except (OSError, json.JSONDecodeError):
            pass
    # Merged by collection name, because this now runs more than once: a
    # collection is read back immediately before it is dropped, so the record
    # describes it while it existed rather than reporting it absent. Without
    # the merge the last call would overwrite the earlier ones and
    # `unindexed_remainders` — which reads `indexed_vectors_count` from here —
    # would go blind on everything dropped early.
    merged = {}
    for path in (out, tmp):
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        nest = into if path is tmp else None
        for c in doc.get("collections", []):
            if not isinstance(c, dict) or not c.get("collection"):
                continue
            name = c["collection"]
            was = merged.get(name, {})
            if nest:
                merged[name] = {**(was or c), nest: c}
            else:
                # A later plain read-back replaces the record but keeps the
                # nested second looks already attached to it.
                merged[name] = {**c, **{k: was[k] for k in NESTED_CAPTURES
                                        if k in was}}
    order = {c: i for i, c in enumerate(workloads.upload_collections())}
    out.write_text(json.dumps({
        "label": label, "engine": uri,
        "collections": sorted(merged.values(),
                              key=lambda c: order.get(c["collection"], 99)),
    }, indent=2) + "\n")
    tmp.unlink(missing_ok=True)
    return 0  # never fails the run: this is provenance, not a measurement


def drop_collections(uri: str, client_cpus: str, names: list[str]) -> None:
    """Delete collections nothing reads again, while the engine is still up.

    strawmANN is resident by design: one arena of `capacity x dim x 4` bytes per
    collection, first-touched at create, so a collection costs its full capacity
    from creation to process exit whether or not anything still reads it. Nothing
    dropped them until a tier had more collections than fit — five 1536-dimension
    arenas is 35.4 GiB before a graph, and that arm was OOM-killed at 45.6 GiB.

    Never fails the run: a drop that does not happen costs memory, and one that
    fails the arm costs the measurement.
    """
    if not names:
        return
    say(f"dropping collections nothing reads again: {', '.join(names)}")
    code, text = sh(["taskset", "-c", client_cpus,
                     "cargo", "run", "--release", "--quiet", "--",
                     "drop-collections", "--engine", uri,
                     "--collections", ",".join(names)], 300, cwd=CONF)
    print(text.rstrip(), flush=True)
    if code != 0:
        print("!! could not drop them; the run continues and simply holds the "
              "memory it would have released", flush=True)


def split_after_dead_writers(stable: list[str]) -> tuple[list[str], list[str], list[str]]:
    """`stable` cut after the last row whose collection nothing reads again.

    Returns the rows up to and including it, the rows after, and the
    collections to drop in between. With no such row, everything is in the
    first part and nothing is dropped early. A dead writer the run does not
    include (`--only` without W1) drops nothing either.
    """
    writers = [r for r in workloads.rows_writing_dead_collections() if r in stable]
    if not writers:
        return list(stable), [], []
    cut = max(stable.index(r) for r in writers) + 1
    by_id = {w.id: w for w in workloads.table()}
    early = [c for r in writers if (c := workloads.collection_of(by_id[r]))]
    return stable[:cut], stable[cut:], early


def invocations_per_arm() -> int:
    """How many times `measure` runs `workloads.py` for one arm, each behind
    its own `settle`: the stable rows (in two parts around a dead writer), then
    the mutating rows."""
    rows = [w.id for w in workloads.table()]
    mutators = mutating_rows()
    _, rest, _ = split_after_dead_writers([r for r in rows if r not in mutators])
    return 1 + bool(rest) + any(r in mutators for r in rows)


def measure(uri: str, label: str, client_cpus: str, storage: str | None,
            placement: str | None = None) -> int:
    """Every row, the recall sweeps, then the rows that would invalidate them."""
    rows = [w.id for w in workloads.table()]
    mutators = mutating_rows()
    stable = [r for r in rows if r not in mutators]
    mutating = [r for r in rows if r in mutators]

    say(f"{label}: §4's rows, except {', '.join(mutating)}")
    # In two parts when a row writes a collection nothing reads again: that
    # collection goes the moment its row returns, before the next row shares
    # the engine with its background index build (`split_after_dead_writers`).
    first, rest, early = split_after_dead_writers(stable)
    rc = run_workloads(uri, label, client_cpus, storage, only=first, placement=placement)
    if early:
        capture_collections(uri, label, client_cpus, names=early)
        drop_collections(uri, client_cpus, early)
    if rest:
        rc |= run_workloads(uri, label, client_cpus, storage, only=rest, placement=placement)

    # Read back, *then* drop: `collections.json` is provenance and
    # `unindexed_remainders` reads `indexed_vectors_count` out of it, so a
    # collection has to be recorded while it still exists.
    dead = [c for c in workloads.collections_droppable_after_rows() if c not in early]
    capture_collections(uri, label, client_cpus, names=dead)
    drop_collections(uri, client_cpus, dead)

    say(f"{label}: recall sweeps, while the collections still match the ground truth")
    rc |= run_recall(uri, label, client_cpus)

    # Before the mutating rows, which query `bench2` alone — and which are
    # exactly where a rebuild of the whole of `bench2` wants the memory.
    swept = workloads.collections_droppable_after_sweeps()
    capture_collections(uri, label, client_cpus, names=swept)
    drop_collections(uri, client_cpus, swept)

    # Only the survivors: the rest were read back before they were dropped,
    # and the merge above keeps all of it in one file.
    survivors = [c for c in workloads.upload_collections()
                 if c not in set(early) | set(dead) | set(swept)]

    # Before the mutating rows, not only after them. The survivor read-back
    # used to happen once, at the end of the arm, and the mutating rows append
    # into `bench2` — so the record said 1,250,000 points for a collection
    # every throughput, latency and recall row on the page searched at
    # 1,000,000. The report banners that honestly ("captured after the mutating
    # rows ... describes a state no search row was measured in"), which is the
    # right thing to do with the wrong capture, and the wrong capture was ours.
    capture_collections(uri, label, client_cpus, names=survivors)

    if mutating:
        say(f"{label}: {', '.join(mutating)} — mutates bench2, so it runs after the sweeps")
        rc |= run_workloads(uri, label, client_cpus, storage, only=mutating,
                            placement=placement)
        # And after them, nested rather than over the top: what the appends
        # left behind is W11's own subject (findings 25, the rebuild still
        # catching up), so both states are kept and the report says which is
        # which.
        capture_collections(uri, label, client_cpus, names=survivors,
                            into="after_mutating_rows")
    return rc


def run_conformance(strawmann_uri: str, qdrant_uri: str, labels: list[str],
                    client_cpus: str, env: dict) -> int:
    """§8's green row, without which the sink refuses every perf row.

    Both engines run at once here. That would be wrong for a throughput
    measurement and is fine for a correctness one: the differ compares answers,
    not speeds, and it is the only phase where the two are up together.
    """
    out = RESULTS / labels[0] / "conformance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Read here rather than at import, so `--data-dir` reaches the differ too.
    base, queries, gt = paths.dataset(DATASET)
    argv = [
        "taskset", "-c", client_cpus,
        "cargo", "run", "--release", "--quiet", "--",
        "differ",
        "--strawmann", strawmann_uri,
        "--qdrant", qdrant_uri,
        "--base", str(base),
        "--queries", str(queries),
        "--ground-truth", str(gt),
        "--metric", paths.metric(DATASET),
        # Named, not inferred: the differ falls back to the base file's stem,
        # and `base.fbin` is the stem of two different corpora.
        "--dataset", DATASET,
        "--json", str(out),
    ]
    print(f"  {' '.join(argv)}", flush=True)
    print(f"  build identity: strawmann={env.get('STRAWMANN_COMMIT')} "
          f"isa={env.get('STRAWMANN_ISA_BUILD')} qdrant={env.get('QDRANT_VERSION')}",
          flush=True)
    r = subprocess.run(argv, cwd=CONF, env=env)
    if r.returncode == 0:
        # The same row licenses both arms: it is one comparison of one pair of
        # builds, and copying it is how the sink finds it under either label.
        for other in labels[1:]:
            dest = RESULTS / other / "conformance.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(out.read_text())
    return r.returncode


class Refusal(str):
    """One reason a label's rows may not be spliced into the front page.

    `by_design` splits the two kinds, because `render` turns this list into an exit
    code. A single pass is the refusal the caller *asked for* — `smoke-test.sh`
    fixes `--reps 1` precisely so a fast run cannot reach README.md — so it fires
    on every smoke run and has not failed anything. Every other refusal here names
    a defect in the measurement and still fails the run.

    A `str` subclass rather than a pair, so callers that only read the message are
    unchanged.
    """

    by_design = False

    def __new__(cls, text: str, by_design: bool = False) -> Refusal:
        r = super().__new__(cls, text)
        r.by_design = by_design
        return r


def is_defect(reason: str) -> bool:
    """Whether a refusal should fail the run, for reasons from any source.

    `render` mixes `publish_refusals`' output with its own plain strings, and
    those -- a refused sink, `--lax`, an unlicensed differ -- are all defects.
    """
    return not getattr(reason, "by_design", False)


def publish_refusals(label: str) -> list[Refusal]:
    """Why this label's rows may not be spliced into the front page.

    Each of these was found by a run that published anyway and reported
    success: foreign load, a half-gated arm and an unindexed
    remainder. The rows carry the evidence in every case; nothing
    read it at the point where the decision to publish is made.
    """
    out: list[str] = []
    rows_path = RESULTS / label / "rows.json"
    if not rows_path.exists():
        return out
    try:
        rows = json.loads(rows_path.read_text())
    except json.JSONDecodeError:
        return out
    rows = rows["rows"] if isinstance(rows, dict) else rows

    # Foreign load that arrived *during* the run. The gate refuses to start
    # beside a busy process, so this is the case it cannot cover: a compile that
    # begins after the first row. `compare.py` already refuses the affected
    # rows' ratios, but the run went on to splice the survivors into the front
    # page and exit 0, which is how ten rows measured against a pinned `rustc`
    # came to be published with nothing but one summary line
    # saying so.
    dirty = [r["id"] for r in rows if r.get("foreign")]
    if dirty:
        out.append(f"{label} measured {len(dirty)} row(s) under foreign load "
                   f"({', '.join(dirty[:6])}{', …' if len(dirty) > 6 else ''}): "
                   f"§7.1 does not tolerate one arm getting load the other did not")

    # §7.1's verdict travels on each row, and `workloads.py` re-checks per
    # invocation, so an arm can be half-gated: the sift1m run stamped
    # 25 of Qdrant's 26 rows FAIL, published anyway, and reported `gate=pass`
    # from an `env.txt` the two-row invocation had overwritten.
    bad_gate = [r["id"] for r in rows if r.get("gate") and r["gate"] != workloads.Gate.passed]
    if bad_gate:
        out.append(f"{label} measured {len(bad_gate)} row(s) under a failed §7.1 gate "
                   f"({', '.join(bad_gate[:6])}{', …' if len(bad_gate) > 6 else ''}): "
                   f"those rows are development-grade and cannot be quoted")

    # A single pass gives neither §7.2(5)'s alternation nor §7.4's three
    # repetitions, and folds no noise floor, so no ratio is banded. The report
    # banners that; README.md is keyed by corpus, not by label, so a clean
    # smoke run would otherwise have replaced the headline table with unbanded
    # numbers and exited 0. `reps` is stamped by `aggregate.fold`, so its
    # absence is one pass. `by_design`: withholds the README write without
    # failing the run, since it is what was asked for.
    if rows and not any((r.get("reps") or 1) >= 2 for r in rows):
        out.append(Refusal(f"{label} is a single pass: §7.2(5) wants the arms alternated "
                           f"and §7.4 three repetitions, and no per-corpus noise floor is "
                           f"folded below three, so no ratio here is banded (--reps 3)",
                           by_design=True))

    out += unindexed_remainders(label, rows)
    return out


def unindexed_remainders(label: str, rows: list | None = None) -> list[str]:
    """Collections the engine left partly outside its index, from the read-back.

    A row searching one is measuring a graph search *plus* a scan of whatever did
    not make it in, and nothing in the row says so — the qps is simply lower. Both
    engines are asked to index everything, so a remainder is a defect in the run:
    27,500 of bench0's 100,000 points is most of what W0's 6.05x
    was.

    Only collections a *query* row read. `bench1` is loaded with
    `--skip-wait-index` because W1 measures ingest and nothing searches it, so its
    whole corpus is outside the index by design — and once the engine reported an
    honest `0` there instead of `null`, every full run refused its own README over
    it. W1's row does carry `collection: bench1`, so `n_queries` is what separates
    a row that searched from one that uploaded.

    `rows` absent flags everything: a refusal is the fail-safe direction.
    """
    path = RESULTS / label / "collections.json"
    if not path.exists():
        return []
    try:
        info = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    out = []
    # From this run's own `run.json`, not from `workloads.upload_n()`: that
    # global follows `$STRAWMANN_DATASET`, so the check read the sift1m count
    # against dbpedia rows whenever the variable was not set in the calling
    # process, and flagged W11's own append as a remainder.
    meta = RESULTS / label / "run.json"
    uploaded = 0
    if meta.exists():
        try:
            uploaded = int(json.loads(meta.read_text()).get("upload_n") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            uploaded = 0
    if not uploaded:
        uploaded = workloads.upload_n()
    searched: set[str] | None = None
    if rows:
        named = {r.get("collection") for r in rows if r.get("n_queries")}
        named.discard(None)
        searched = named or None
    for c in info.get("collections", []):
        pts, idx = c.get("points_count"), c.get("indexed_vectors_count")
        if pts is None or idx is None or idx >= pts:
            continue
        if searched is not None and c.get("collection") not in searched:
            continue
        # More points than the run uploaded means a mutating row grew it.
        # Survivors are now read back *before* those run, so this branch is for
        # results predating that capture, where the only read-back is the late
        # one and W11's still-rebuilding remainder is that row's subject
        # (findings 25) rather than a defect in the rows before it. The nested
        # `after_mutating_rows` record is not walked here for the same reason.
        if pts > uploaded:
            continue
        out.append(f"{label}: {c['collection']} has {pts - idx:,} of {pts:,} points "
                   f"outside the index, so rows searching it measure a graph search "
                   f"plus a scan of the remainder")
    return out


def render(labels: list[str], lax: bool = False) -> int:
    """Record, then render, then decide whether the README may change.

    Three refusals, each printed with what would have been done:

      * `results.py ingest` (there is no `record` subcommand; the call to one
        exited 2 on every full run and nothing read the code) refused the rows,
        so §8's sink holds nothing that licenses them;
      * `--lax` was used, so the §7.1 gate failed and every row is
        development-grade;
      * the differ's row has `licenses_comparative` false, so a table of ratios
        between the engines is not a claim §8 lets this project make.

    ...plus whatever `publish_refusals` found in the rows themselves.

    The HTML report is rendered regardless, from both labels in one page, since
    it carries its own licence banner; what is refused is splicing numbers into
    README.md and the per-dataset comparison document.

    Only a refusal that names a *defect* fails the run (`is_defect`). The three
    above all do; a single pass does not, because `smoke-test.sh` fixes
    `--reps 1` and a run doing what it was told is not a run that went wrong.
    """
    say("recording and rendering")
    rc = 0
    sink_ok = True
    for label in labels:
        code, out = sh([sys.executable, str(ROOT / "bench/harness/results.py"),
                        "ingest", label], 300, cwd=ROOT)
        print(out.rstrip(), flush=True)
        if code != 0:
            sink_ok = False
            rc |= 1
            print(f"  !! results.py ingest {label} exited {code}", file=sys.stderr)

    conf = {}
    cpath = RESULTS / labels[0] / "conformance.json"
    if cpath.exists():
        try:
            conf = json.loads(cpath.read_text())
        except json.JSONDecodeError:
            conf = {}
    comparative = bool(conf.get("licenses_comparative"))

    reasons = []
    if not sink_ok:
        reasons.append("the §8 sink refused the rows (see above)")
    for label in labels:
        reasons += publish_refusals(label)
    if lax:
        reasons.append("--lax: the §7.1 gate failed, every row is development-grade")
    if not conf:
        reasons.append("no conformance.json: nothing licenses these numbers")
    elif not comparative:
        reasons.append(f"licenses_comparative is false (differ reached "
                       f"{conf.get('tier_reached')!r}): a ratio between the engines "
                       f"is not a claim §8 allows")

    cmp_argv = [sys.executable, str(ROOT / "bench/harness/compare.py"), labels[0], labels[1]]
    if reasons:
        print("\n!! NOT writing README.md / docs/comparison-<dataset>.md, "
              "because:", flush=True)
        for r in reasons:
            print(f"!!   - {r}", flush=True)
        print(f"!! would have run: {' '.join(cmp_argv + ['--write-readme'])}", flush=True)
        print("!! the block it would have written follows, for the record:\n", flush=True)
        code, out = sh(cmp_argv + ["--readme"], 300, cwd=ROOT)
        print(out.rstrip(), flush=True)
        # Refusing to publish is not itself a failure, and conflating the two
        # cost the exit code its meaning: the single-pass refusal fires on every
        # smoke run, so `rc |= 1` made the script incapable of exiting 0 and
        # `done` cried wolf on a clean sift1m run that had just passed T4. Only
        # a refusal naming a defect fails the run; a by-design one still
        # withholds README.md.
        if any(is_defect(r) for r in reasons):
            rc |= 1
        else:
            print("!! (every reason above is a property of the run that was "
                  "asked for, not a fault in it: this run has not failed)",
                  flush=True)
    else:
        code, out = sh(cmp_argv + ["--write-readme"], 300, cwd=ROOT)
        print(out.rstrip(), flush=True)
        rc |= code

    # One report, both labels, side by side — called once per label it produced
    # two single-engine pages and printed a path neither was at. No `-o`:
    # `report.default_out` keys the name by dataset and labels, and a second
    # spelling here is the copy that cannot see what the runs measured.
    code, out = sh(["uv", "run", "--project", str(ROOT / "bench"),
                    str(ROOT / "bench/harness/report.py"), *labels], 600, cwd=ROOT)
    print(out.rstrip(), flush=True)
    if code != 0:
        # Not "no report was written": `report.py` renders for ungated and
        # refused runs on purpose, since the page carries its own banner. A
        # non-zero exit here means it could not render, and it says why above.
        print(f"  !! report.py exited {code}; see its output above", file=sys.stderr)
        rc |= 1
    return rc


#: How long a label directory in `bench/results/` survives without a citation.
#:
#: Nothing pruned that directory: it is gitignored, and a run only ever clears
#: the label it is about to write. It reached 40 directories, 12 MB of which no
#: file in the tree referenced -- experiments, diagnostics, smoke output and
#: superseded pairs, indistinguishable by name from the runs the documents quote.
#:
#: But a bare age rule is what deleted `sm-dbp100k`/`qd-dbp100k`, and
#: `docs/comparison-dbpedia-openai-100K-1536-angular.md` can no longer be
#: regenerated because of it: the block is marked generated and there is nothing
#: left to generate it from. So age is the last of four tests, not the only one.
RETAIN_DAYS = 14


def cited_labels(results: Path | None = None) -> set[str]:
    """The label directories README.md or a document under `docs/` names.

    Documents only. A label in a harness docstring is an example of a command;
    a label in a document is a citation a reader can follow, and deleting what
    it points at is what orphans a generated block. `docs/reports/*.html` is
    skipped: 5 MB a page, and a page already names its labels in its filename.
    """
    results = results or RESULTS
    if not results.is_dir():
        return set()
    names = {p.name for p in results.iterdir() if p.is_dir()}
    if not names:
        return set()
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    blob = "\n".join(p.read_text(errors="replace") for p in docs if p.is_file())
    cited = {n for n in names
             if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", blob)}
    # A committed page cites its labels in its own filename
    # (`report-<dataset>-<a>-vs-<b>-<day>-<hhmm>`, `report.default_out`), and
    # nowhere else: `docs/reports/README.md` abbreviates the middle. Without
    # this the 0903 pair -- the second measurement of the same two binaries,
    # and so the only reproducibility check there is -- reads as disposable.
    pages = " ".join(q.name for q in (ROOT / "docs/reports").glob("report-*.html"))
    cited |= {n for n in names if n in pages}
    # An engine pair is one measurement. `compare.py` needs both sides, so
    # citing one arm protects the other: the dbp100k document was orphaned
    # because one label was cited by name and its opposite number was not.
    stems = {n[3:] for n in cited if n[:3] in ("sm-", "qd-")}
    return cited | {n for n in names
                    if n[:3] in ("sm-", "qd-") and n[3:] in stems}


def prunable(keep: set[str] | None = None, retain_days: int = RETAIN_DAYS,
             now: float | None = None, results: Path | None = None) -> list[Path]:
    """Label directories that may be deleted, oldest first.

    A directory has to fail all four tests to be returned:

      * cited by a document, or the other arm of a pair that is
        (`cited_labels`);
      * holds anything hand-written (`*.md`) or the only rendering of a run
        (`report-*.html`) -- an analysis nobody can re-derive from rows;
      * touched within `retain_days`;
      * named by the run doing the pruning.

    Files at the top of `bench/results/` are never touched: `noise.json` is the
    measured floor every ratio is banded with, and it is not a label.
    """
    results = results or RESULTS
    if not results.is_dir():
        return []
    keep = (keep or set()) | cited_labels(results)
    now = time.time() if now is None else now
    cutoff = now - retain_days * 86400
    out = []
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        if d.name in keep:
            continue
        if any(d.rglob("*.md")) or any(d.rglob("report-*.html")):
            continue
        newest = max((p.stat().st_mtime for p in d.rglob("*")), default=d.stat().st_mtime)
        if newest > cutoff:
            continue
        out.append(d)
    return out


def prune(keep: set[str] | None = None, retain_days: int = RETAIN_DAYS) -> None:
    """Apply `RETAIN_DAYS`, naming every directory it removes.

    Printed rather than silent. A run that deletes measurements without saying
    which is the same failure as one that publishes numbers without saying
    where they came from.
    """
    dead = prunable(keep, retain_days)
    if not dead:
        return
    print(f"  retention: removing {len(dead)} label(s) no document cites and "
          f"nothing has touched in {retain_days} days", flush=True)
    for d in dead:
        print(f"          {d.name}", flush=True)
        shutil.rmtree(d)


# -------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cpus", required=True,
                    help="cpus the engine runs on, e.g. 4-11 (both engines get these)")
    ap.add_argument("--client-cpus", required=True,
                    help="cpus bfb and the harness run on, e.g. 0-3")
    #: Off by default, and the default is the measurement rather than caution:
    #: the sidecar costs W3 4.25% against a 2.00% noise band, so it is visible,
    #: and a nightly whose headline is a ratio should not pay it silently. Both
    #: arms carry it when on, so the ratio stays honest; what shifts is the
    #: absolute qps and its comparability with runs taken without it.
    #: Which Qdrant to measure. The image stays the default (§8.9 pins by
    #: digest, a path cannot); the binary is what `--perf` needs. Naming the
    #: image explicitly is allowed so a script can say which it meant.
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--qdrant-binary", metavar="PATH", default=None,
                     help="run this Qdrant binary natively instead of the "
                          "container, so `--perf` can attach to it. Provenance "
                          "becomes a sha256 rather than an image digest")
    src.add_argument("--qdrant-docker", action="store_true",
                     help=f"measure {QDRANT_IMAGE} in a container (the default). "
                          f"Pinned by digest, and cannot be counted by --perf")
    ap.add_argument("--perf", nargs="?", const="default", default=None,
                    metavar="SET",
                    help="attach perf stat to both engines for every row, adding "
                         "the Hardware counters section to the report (IPC, "
                         "effective frequency, branch MPKI, DRAM and TLB per "
                         "query). Costs ~4%% of throughput on this host, so the "
                         "absolute numbers are not comparable with runs taken "
                         "without it")
    ap.add_argument("--reps", type=int, default=1,
                    help="passes per engine, alternated A/B/A/B (§7.2(5)) and folded "
                         "by aggregate.py into the published label as the median with "
                         "its spread (§7.4). 1 is the historical single pass; 3 is what "
                         "§7.4 asks for, and roughly triples the wall clock")
    #: Defaults to `auto`, which is a change of policy and not just of value.
    #: Unset, every arm resolves §4's fractions against its *own* saturation, so
    #: "90% of saturation" is two different offered loads and the report refuses
    #: to read the percentiles across — which is what the dbpedia-openai-1m run
    #: did, because the default did it.
    #:
    #: `auto` needs a previous W4 for both labels; with none, a first run
    #: proceeds on per-engine references and says so at the top, since it has
    #: nothing else it could do. An explicit `--rps-reference auto` still fails
    #: loudly — asking by name and silently not getting it is the case.
    ap.add_argument("--rps-reference", default=None, metavar="QPS|auto|none",
                    help="saturation qps the open-loop arms are a fraction of, for "
                         "BOTH engines. Pass the slower engine's, so the arms are a "
                         "load both can serve and their latency percentiles are "
                         "comparable. Default `auto`: the slower of the two labels' "
                         "previous W4 on this machine. `none` gives each engine its "
                         "own measured W4, and the report then refuses the "
                         "cross-engine latency read")
    ap.add_argument("--placement", default=None, choices=tuple(workloads.Placement),
                    help="§5.5 residency, asked of BOTH engines: a ratio across two "
                         "residencies measures the residency. Default `cached`, the "
                         "only one both can serve — Qdrant refuses `pinned` for dense "
                         "vectors. `--placement pinned --skip qdrant` measures "
                         "strawmANN at its own default instead")
    ap.add_argument("--strawmann-label", default="strawmann")
    ap.add_argument("--qdrant-label", default="qdrant")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["build", "strawmann", "qdrant", "conformance", "render"],
                    help="phases to leave alone (repeatable)")
    ap.add_argument("--oversampling-policy", default="defaults",
                    choices=tuple(workloads.OversamplingPolicy),
                    help="which of the three quantized experiments this run is. "
                         "`defaults` (default) leaves each engine on its own "
                         "rescore pool, which is `limit`-sized for Qdrant and "
                         "`max(asked, ef)` for strawmANN, so §7.4 refuses the "
                         "quantized rows a ratio: the engines are not at equal "
                         "recall. `matched` sends "
                         "`--quantization-oversampling 2` to BOTH engines on the "
                         "quantized search rows that do not already name one, "
                         "which moves only Qdrant (strawmANN's pool is already "
                         "ef-sized) and takes its SQ8 recall@10 from 0.8929 to "
                         "0.9888 against strawmANN's 0.9891 (findings 42). "
                         "`pool` sends ef / limit on every quantized search row, "
                         "so Qdrant rescores ef candidates as strawmANN does, "
                         "binary and PQ included. They are different experiments "
                         "and never one table, so "
                         "the choice is hashed into every row and `compare.py` "
                         "refuses a ratio across them as STALE")
    ap.add_argument("--segment-policy", default="equal-work",
                    choices=tuple(workloads.SegmentPolicy),
                    help="which of the two Qdrant experiments this run is. "
                         "`equal-work` (default) holds Qdrant to one populated "
                         "graph, so `ef` means the same thing on both sides and "
                         "§8 can license a comparative claim. `as-deployed` "
                         "takes Qdrant's own default_segment_number, which it "
                         "resolves to the CPU count: what a user would see, and "
                         "legitimately not at strawmANN's recall for a given "
                         "`ef`, so the honest comparison there is throughput at "
                         "matched recall. Measured on dbpedia-openai-1m the two "
                         "differ by 109%% on the saturating row and -39%% on the "
                         "single-query one, in opposite directions, so they are "
                         "two experiments and never one table. The choice is "
                         "hashed into every row, so `compare.py` refuses a ratio "
                         "across them as STALE")
    ap.add_argument("--dataset", default="sift1m", choices=paths.runnable(),
                    help="the corpus every row, both recall sweeps and the "
                         "conformance differ read (default sift1m). Its metric "
                         "and vector width come from `datasets.json`, so the "
                         "collection a row creates and the ground truth it is "
                         "scored against cannot disagree. Entries fetched but "
                         "not converted to fbin are absent from this list")
    ap.add_argument("--wait-for-gate", type=float, default=0, metavar="MINUTES",
                    help="when the engine ports are held or the §7.1 gate refuses, ask "
                         "again every 5 min for up to this long instead of exiting; "
                         "for unattended runs (default 0: refuse at once)")
    ap.add_argument("--lax", action="store_true",
                    help="run on a host that failed §7.1; unpublishable")
    ap.add_argument("--storage", default=os.environ.get("STORAGE_DIR"),
                    help="strawmann's storage dir, for the on-disk size column")
    paths.add_data_argument(ap)
    args = ap.parse_args(argv[1:])
    # Exported as $STRAWMANN_DATA, so `workloads.py` and `recall.py` — which
    # this spawns — read the same root without being passed the flag again.
    paths.use_data_dir(args.data_dir)
    global RPS_REFERENCE, DATASET, PERF_SET, QDRANT_BINARY, SEGMENT_POLICY
    global OVERSAMPLING_POLICY
    PERF_SET = args.perf
    if args.qdrant_binary:
        QDRANT_BINARY = Path(args.qdrant_binary).expanduser().resolve()
        if not QDRANT_BINARY.exists():
            print(f"--qdrant-binary: no such file: {QDRANT_BINARY}", file=sys.stderr)
            return 2
        if not os.access(QDRANT_BINARY, os.X_OK):
            print(f"--qdrant-binary: not executable: {QDRANT_BINARY}", file=sys.stderr)
            return 2
        # Said, not refused: measuring an unrebuilt binary is legitimate, and
        # knowing it is what makes it so. `run.json` recording a commit dated
        # after the binary it names is not.
        if provenance.qdrant_binary_predates_head(QDRANT_BINARY):
            print(f"!! --qdrant-binary: {QDRANT_BINARY} was built before the commit "
                  f"its checkout is now on, so `commit` in run.json is not what "
                  f"produced it.", file=sys.stderr)
            print("   `binary_sha256` is the identity that holds; rebuild if you "
                  "meant to measure the checkout.", file=sys.stderr)
    # `--perf` on the container measures one arm and blanks the other, which is
    # worse than not measuring either: the page would carry a Hardware counters
    # section with strawmANN's numbers and Qdrant's dashes. Refused here, with
    # the flag that fixes it, rather than discovered per row.
    if PERF_SET and QDRANT_BINARY is None:
        print("--perf needs --qdrant-binary: perf_event_open cannot attach to the "
              "containerised engine.", file=sys.stderr)
        print("  The image runs as root and this harness does not, and "
              "perf_event_paranoid does not grant ptrace permission at any value.",
              file=sys.stderr)
        print("  Either pass --qdrant-binary <path> so both engines run as this "
              "user, or drop --perf.", file=sys.stderr)
        return 2

    engines = [e for e in ("strawmann", "qdrant") if e not in args.skip]

    # `workloads.py` refuses a row whose bfb is off the pin, but per invocation,
    # and only the accumulated return code is read at the end:
    # that cost twelve refused invocations, three empty passes and a traceback
    # 500 lines from the cause. Checked before `wait_until_admitted`, which asks
    # a busy host again for up to two hours: a pin never clears, so waiting on it
    # would turn a one-second failure into a two-hour one. Not worded "refusing
    # to run" either, which is the gate's phrase and reads as a busy host.
    if engines and (why := workloads.check_bfb_pin()):
        print(f"the load generator is not the pinned one:\n  {why}", file=sys.stderr)
        return 1

    labels = [args.strawmann_label, args.qdrant_label]
    # From the parser, not `sys.argv`: `--rps-reference=auto` was invisible to
    # the substring test, so the loud failure the flag promises became the
    # silent fallback below.
    asked = args.rps_reference is not None
    reference = args.rps_reference if asked else "auto"
    try:
        RPS_REFERENCE = resolve_rps_reference(
            None if str(reference).lower() == "none" else reference,
            labels)
    except ValueError as e:
        if asked:
            # Named on the command line and unsatisfiable: that is an error.
            print(f"--rps-reference: {e}", file=sys.stderr)
            return 2
        # The default, on a pair that has never been run. Proceed, and say what
        # it costs, before the hours rather than in the report afterwards.
        RPS_REFERENCE = None
        print(f"!! --rps-reference auto (the default) has nothing to read: {e}",
              file=sys.stderr)
        print("   this run's open-loop arms will each use their own engine's "
              "saturation, so the report will refuse to read their latency "
              "percentiles across the two columns. Re-running these labels "
              "afterwards will pick up a reference automatically.", file=sys.stderr)
    # Same mechanism as `--data-dir` above, and for the same reason: the phases
    # below are subprocesses, and a choice made here has to survive the spawn.
    # `use_dataset` also rebinds `workloads`' own globals, which matters because
    # this process asks it questions — `required_capacity()` sizes the server —
    # and exporting the variable alone left those answers describing SIFT1M.
    DATASET = workloads.use_dataset(args.dataset)
    # Before anything creates a collection, and by the same mechanism: this
    # rebinds `workloads`' globals here and exports `$SEGMENT_POLICY` for the
    # per-arm subprocesses, so both arms cannot be handed different ones.
    SEGMENT_POLICY = workloads.use_segment_policy(args.segment_policy)
    # Same mechanism, same reason: bound here for this process and exported for
    # the per-arm subprocesses, so the two arms cannot be handed different ones.
    OVERSAMPLING_POLICY = workloads.use_oversampling_policy(args.oversampling_policy)
    if str(workloads.OversamplingPolicy.matched) == OVERSAMPLING_POLICY:
        print(f"oversampling {OVERSAMPLING_POLICY}: the quantized search rows carry "
              f"--quantization-oversampling {workloads.MATCHED_OVERSAMPLING} on both "
              f"engines, which moves Qdrant's rescore pool and not strawmANN's. "
              f"These rows may not be ratioed against a `defaults` run (§8, STALE).")
    elif str(workloads.OversamplingPolicy.pool) == OVERSAMPLING_POLICY:
        print(f"oversampling {OVERSAMPLING_POLICY}: every quantized search row carries "
              f"--quantization-oversampling ef / limit on both engines, so Qdrant "
              f"rescores ef candidates as strawmANN does. These rows may not be "
              f"ratioed against a `defaults` or `matched` run (§8, STALE).")
    # Both arms are given this set — `--pin --cpus` for strawmANN and
    # `--cpuset-cpus` for the container — and until now it survived only in
    # this script's own output. The engine's observed affinity cannot stand in
    # for it: strawmANN pins its workers and leaves the main thread free, so
    # the observed mask and the requested set legitimately differ.
    os.environ["BENCH_SERVER_CPUS"] = args.server_cpus
    # The mixed rows' search lengths, by the same mechanism as the dataset:
    # bound here and exported, so both arms' subprocesses build the same `-n`.
    for env, n in resolve_w11_queries(labels).items():
        os.environ[env] = str(n)
        setattr(workloads, env, n)
        print(f"{env}={n:,}: the slower engine's previous search at this write rate, "
              f"ending {W11_SPAN_MARGIN - 1:.0%} inside the append")
    if RPS_REFERENCE:
        print(f"open-loop arms pinned to {RPS_REFERENCE:,.0f} qps for both engines "
              f"(§4's fractions of one reference, so the two arms are the same "
              f"offered load and not merely the same fraction of capacity)")

    for f in paths.dataset(DATASET):
        if not f.exists():
            # Named separately from the fetch: for every dataset but SIFT1M the
            # missing file is the converted corpus or the fp64 ground truth,
            # neither of which `fetch` produces. Sending the reader there costs
            # a re-download and still leaves the file missing.
            print(f"missing {f}\n"
                  f"the archives come from: conformance/datasets/datasets.py fetch {DATASET}\n"
                  f"the fbin and the ground truth are made by the dataset's own "
                  f"conversion (for dbpedia-openai-1m: bench/harness/headline.py)",
                  file=sys.stderr)
            return 1

    # Before the gate and the build, not after: this costs a millisecond and
    # the two phases it precedes cost minutes, so a mistyped label should not
    # be discovered on the far side of them. `workloads.py` re-checks it as
    # the arms start, which is what makes the rule unbypassable — this is only
    # the early, cheap half.
    for label in (args.strawmann_label, args.qdrant_label):
        if why := workloads.foreign_dataset(RESULTS / label, DATASET):
            print(f"refusing to write into {RESULTS / label}\n  {why}", file=sys.stderr)
            return 1

    global PLACEMENT
    if args.placement:
        PLACEMENT = workloads.Placement(args.placement)
    if engines and (why := preflight_placement(PLACEMENT, engines)):
        print(f"--placement {PLACEMENT}: {why}", file=sys.stderr)
        return 1
    print(f"placement   {PLACEMENT} (both engines; §7.4 compares at one residency)",
          flush=True)

    # Said before the gate rather than after it: the gate settles for a minute
    # and then starts the run, and an operator who learns the cost afterwards
    # has already committed. dbpedia-openai-1m at --reps 3 is nine and a half
    # hours, and nothing said so until it was morning.
    est = estimated_minutes(DATASET, max(1, args.reps))
    if est:
        mins, src = est
        done = time.strftime("%H:%M", time.localtime(time.time() + mins * 60))
        tail = "" if "conformance" in args.skip else \
            ", plus the §8 differ and the render after them (not timed here)"
        # One settle per `workloads.py` invocation, each up to
        # SETTLE_TIMEOUT_S: nine took 1-2 min apiece and were the whole gap
        # between the 05:41 estimate and the 05:56 finish. Counted from what
        # `measure` actually invokes rather than written down, since splitting
        # the stable rows around W1 made it three per arm.
        settles = 2 * invocations_per_arm() * max(1, args.reps)
        print(f"estimate    ~{mins / 60:.1f} h of measurement passes for "
              f"{DATASET} x{args.reps}{tail}"
              f"\n            passes alone would end about {done}; measured from "
              f"{src}'s own rows, not a guess"
              f"\n            plus up to {settles * SETTLE_TIMEOUT_S // 60} min of settle "
              f"waits between invocations ({settles} x <= {SETTLE_TIMEOUT_S}s), "
              f"not in the figure above", flush=True)
        if mins > 240:
            print(f"            that is {mins / 60:.1f}+ hours of this machine, "
                  f"starting now. Ctrl-C during the gate below if that is not "
                  f"the intention.", flush=True)
    else:
        print(f"estimate    unknown -- no previous {DATASET} run on this machine "
              f"to measure against", flush=True)

    verdict = wait_until_admitted(args.lax, args.wait_for_gate)
    if not verdict:
        return 1

    n_server = cpu_count(args.server_cpus)
    # One I/O thread, the rest workers. Sized from the isolated set rather than
    # left at the default 8: a pool one short makes `--cpus` refuse to start,
    # and silently running 8 workers on 8 cores plus an I/O thread would put
    # two threads on one core.
    workers = max(1, n_server - 1)
    capacity = workloads.required_capacity()
    print(f"\ndataset     {DATASET} (d={paths.dim(DATASET)}, {paths.metric(DATASET)})"
          f"\nserver cpus {args.server_cpus} ({n_server}) -> 1 io + {workers} workers"
          f"\nclient cpus {args.client_cpus} ({cpu_count(args.client_cpus)})"
          f"\ncapacity    {capacity:,}"
          f"\nqdrant      {qdrant_target()}"
          f"\nsegments    {SEGMENT_POLICY} ({workloads.policy_means()})"
          f"\nreps        {args.reps}", flush=True)

    # What a single pass gives up, said before the hours rather than found in
    # the report after them. One root cause: `--reps N` alternates the arms
    # A/B/A/B (§7.2(5)) *and* makes `aggregate.py` fold each label its own
    # `noise.json`, which `regression.floor_for` prefers over the global file —
    # so a repeated run measures this corpus's spread on both engines instead of
    # borrowing sift1m's, which the report refuses.
    if args.reps < 2:
        print("\n!! --reps 1: this run will not satisfy §7.2(5) or §7.4, and its "
              "report will say so twice.", flush=True)
        print("   - the arms are measured A-then-B, so anything that drifts on this "
              "machine between them is confounded with the engine, in the same "
              "direction for every row.", flush=True)
        print("   - no per-label noise floor is folded, so ratios fall back to the "
              "global floor and are refused when it names another corpus.", flush=True)
        print("   `--reps 3` is what §7.4 asks for; it removes both banners and "
              "roughly triples the wall clock.", flush=True)

    labels = [args.strawmann_label, args.qdrant_label]
    rc = 0

    if "build" not in args.skip and not build_strawmann():
        return 1

    # A/B/A/B at the granularity this host allows. Interleaving *within* a row
    # would need both engines up at once, and they are pinned to the same cores
    # by design, so they would contend for exactly the resource the pinning
    # isolates. The finest available is therefore the table — which is also
    # §7.4's "three interleaved repetitions per cell", so one mechanism serves
    # both. Each pass writes its own label; `aggregate.py` folds them.
    rep_labels: dict[str, list[str]] = {args.strawmann_label: [], args.qdrant_label: []}
    for rep in range(1, max(1, args.reps) + 1):
        multi = args.reps > 1
        sm_label = f"{args.strawmann_label}-rep{rep}" if multi else args.strawmann_label
        qd_label = f"{args.qdrant_label}-rep{rep}" if multi else args.qdrant_label
        rep_labels[args.strawmann_label].append(sm_label)
        rep_labels[args.qdrant_label].append(qd_label)
        if multi:
            say(f"pass {rep} of {args.reps}")

        if "strawmann" not in args.skip:
            say(f"strawmann: §4's rows -> {sm_label}")
            log = RESULTS / sm_label / "server.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            p = start_strawmann(args.server_cpus, STRAWMANN_PORT, log, workers, capacity)
            if p is None:
                return 1
            try:
                rc |= measure(f"http://localhost:{STRAWMANN_PORT}", sm_label,
                              args.client_cpus, args.storage,
                              placement=PLACEMENT)
            finally:
                finish_strawmann_arm(p, sm_label)

        # --- Qdrant, this pass ---
        if "qdrant" not in args.skip:
            say(f"qdrant: §4's rows -> {qd_label}")
            version = start_qdrant(args.server_cpus, QDRANT_GRPC, QDRANT_REST)
            if version is None:
                return 1
            try:
                rc |= measure(f"http://localhost:{QDRANT_GRPC}", qd_label,
                              args.client_cpus, str(QDRANT_STORAGE),
                              placement=PLACEMENT)
            finally:
                finish_qdrant_arm()

    if args.reps > 1:
        say(f"folding {args.reps} passes per engine (§7.4: median and spread)")
        for label, reps in rep_labels.items():
            if (label == args.strawmann_label and "strawmann" in args.skip) or \
               (label == args.qdrant_label and "qdrant" in args.skip):
                continue
            argv = [sys.executable, str(ROOT / "bench/harness/aggregate.py"), label, *reps]
            print(f"  {' '.join(argv)}", flush=True)
            rc |= subprocess.run(argv, cwd=ROOT).returncode

    # --- conformance, both up ---
    if "conformance" not in args.skip:
        say("§8 conformance: the row that licenses every number above")
        log = RESULTS / args.strawmann_label / "server-conformance.log"
        p = start_strawmann(args.server_cpus, STRAWMANN_CONF_PORT, log, workers, capacity)
        if p is None:
            return 1
        version = start_qdrant(args.server_cpus, QDRANT_GRPC, QDRANT_REST)
        try:
            if version is None:
                rc |= 1
            else:
                # The differ speaks gRPC to both engines, so this is
                # QDRANT_GRPC and not QDRANT_REST. Pointing it at the REST port
                # fails with `h2 protocol error` after the collection is
                # created, which reads as a transport fault rather than a port.
                rc |= run_conformance(f"http://localhost:{STRAWMANN_CONF_PORT}",
                                      f"http://localhost:{QDRANT_GRPC}",
                                      labels, args.client_cpus,
                                      build_identity(version))
        finally:
            stop_strawmann(p)
            stop_qdrant()

    # The invariant, checked against what the engines said rather than what
    # they were asked. A run that drifted off one residency has not produced a
    # comparison, and saying so here is louder than a banner nobody reads.
    both_ran = len(labels) == 2 and not (set(args.skip) & {"strawmann", "qdrant"})
    if both_ran and (why := placement_mismatch(labels)):
        print(f"!! REFUSING the comparison: {why}", file=sys.stderr)
        rc |= 1

    if "render" not in args.skip:
        rc |= render(labels, lax=verdict.failed)

    say("done")
    print(f"  rows:   {RESULTS / args.strawmann_label / 'rows.json'}")
    print(f"          {RESULTS / args.qdrant_label / 'rows.json'}")
    prune(keep={*labels, *rep_labels, *(r for reps in rep_labels.values() for r in reps)})
    if rc != 0:
        print("\n  some phase reported a non-zero status; read the log above before "
              "quoting anything from this run", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
