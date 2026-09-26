"""Reading a row's figures out of bfb's own JSON output.

Moved out of `workloads.py`, which re-exports every name here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

#: bfb's `wait_index` sleeps 1 s then needs three consecutive Green replies, so
#: no Time-to-Green below three seconds is real (a 0.064 s build reported
#: 3.008 s). A constant bias: the difference between engines survives it, the
#: ratio does not, and it understates the faster one.
WAIT_INDEX_FLOOR_S = 3.0


#: The percentiles §8.9's row shape requires, plus the two tails §7.4 argues
#: about. bfb's own JSON reports min/avg/p50/p95/max and no p99, but it also
#: writes every request time, so the tail is recoverable rather than lost.
PERCENTILES = (50, 95, 99, 99.9)



def percentiles_us(times: list[float]) -> dict[str, float]:
    """Percentiles in microseconds, from bfb's raw per-request times.

    Nearest-rank on the sorted sample, not interpolated: with 50,000 requests
    the difference is below the noise this measures, and a rank is a request
    that actually happened. The rank is `ceil(p/100 * n)`, the textbook
    nearest-rank; `round()` (banker's, to even) put p50 of five samples at the
    second value rather than the third, and p50 of 3,125 (W5's batches) at
    rank 1,562 rather than 1,563: the wrong element wherever `p/100 * n`
    ends in .5.
    """
    if not times:
        return {}
    xs = sorted(times)
    out = {}
    for p in PERCENTILES:
        k = max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))
        out[f"p{p:g}".replace(".", "")] = xs[k] * 1e6
    out["max"] = xs[-1] * 1e6
    out["mean"] = sum(xs) / len(xs) * 1e6
    return out



def phases_of(results: Path, wid: str) -> dict:
    """bfb's own upload and index-wait times, in seconds, unrounded.

    The harness's `seconds` is `int(wall_clock)` around the whole invocation:
    one-second resolution on a row whose headline *is* a duration, and it
    includes process startup, collection creation and teardown. bfb reports the
    two phases separately and as floats, so W1's ingest and W2's Time-to-Green
    can be the number they claim to be.

    It also exposes the bias. `wait_index` sleeps one second *before* its first
    poll and requires three consecutive Green observations, so its floor is
    three seconds no matter how fast the build is — a 50k build measured here
    reported 3.008 s against an upload of 0.064 s. `time_to_green_floored`
    marks a row sitting on that floor, where the number is an upper bound on
    the build and a measurement of the polling loop.
    """
    p = results / f"{wid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    res = d.get("results", {})
    out: dict = {}
    if isinstance(res.get("upload"), dict):
        v = res["upload"].get("duration_secs")
        if isinstance(v, (int, float)):
            out["upload_s"] = round(float(v), 4)
    if isinstance(res.get("index"), dict):
        v = res["index"].get("wait_secs")
        if isinstance(v, (int, float)):
            out["index_wait_s"] = round(float(v), 3)
            out["time_to_green_floored"] = float(v) < WAIT_INDEX_FLOOR_S + 0.5
    return out



def wall_qps_of(results: Path, wid: str, batch: int = 1) -> dict:
    """`n_queries / duration_secs` from bfb's JSON, the figure the tables print.

    bfb's `Median qps` (`stats.rs`) is the median of a rate series sampled per
    request from a moving window, and on a short row that median sits below
    the mean by construction, more so the shorter the row: it read 27,201 on a
    W4 whose 50,000 queries took 1.61 s (31,056). The wall figure has no such
    dependence, and it is what a reader means by queries per second. The
    request count comes from `full_timings` rather than `-n`, so an
    interrupted row reports what it did rather than what it was asked.
    """
    p = results / f"{wid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    res = d.get("results", {})
    phase = res.get("search") or res.get("scroll") or {}
    dur = phase.get("duration_secs")
    reqs = phase.get("full_timings")
    if not isinstance(dur, (int, float)) or dur <= 0 or not isinstance(reqs, list) or not reqs:
        return {}
    n = len(reqs) * max(1, batch)
    return {"qps_wall": round(n / float(dur), 3), "duration_s": round(float(dur), 3),
            "n_queries": n}



def background_of(results: Path, wid: str) -> dict:
    """The upload phase of a row's concurrent bfb, from its own JSON."""
    p = results / f"{wid}-write.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    up = (d.get("results") or {}).get("upload") or {}
    out = {}
    if isinstance(up.get("points_per_sec"), (int, float)):
        out["background_pps"] = round(float(up["points_per_sec"]), 1)
    if isinstance(up.get("duration_secs"), (int, float)):
        out["background_s"] = round(float(up["duration_secs"]), 3)
    return out



def latency_of(results: Path, wid: str) -> dict:
    """Client-side and server-side latency for one row.

    Both, because their difference is the transport and the queue: §7.4's rule
    that a closed-loop p99 is not a latency result is exactly about the gap
    between them. bfb calls them `full_timings` (round trip) and
    `server_timings` (what the server reported).

    Read from `search` *or* `scroll`, because bfb records the same two series
    under whichever phase ran. Reading only `search` cost W13 its percentiles
    entirely: it published a five-figure qps with no latency next to it and no
    stated reason, which reads as a measurement failure rather than as a lookup
    in the wrong key.
    """
    p = results / f"{wid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    res = d.get("results", {})
    search = res.get("search") or res.get("scroll") or {}
    out = {}
    for key, prefix in (("full_timings", "client"), ("server_timings", "server")):
        v = search.get(key)
        if isinstance(v, list) and v:
            for name, val in percentiles_us(v).items():
                out[f"{prefix}_{name}_us"] = round(val, 3)
    if out:
        out["latency_samples"] = len(search.get("full_timings") or [])
    return out
