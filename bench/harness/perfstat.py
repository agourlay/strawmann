#!/usr/bin/env python3
"""What the *hardware* did during a row, from `perf stat` attached to the engine.

`procstat.py` reads what the kernel counted; this reads what the core counted.
A row that spent 40 s of CPU tells you the engine was running, not whether
those cycles retired instructions, waited on DRAM or walked page tables — and
§5's cost model is a set of claims about exactly that. §7.3 specifies the event
set and §7.5 makes effective frequency a guard rail, and both were implemented
only in `bench/micro/perfctr.zig`, which measures a kernel in-process rather
than the engine serving the row the README quotes.

## Why a sidecar and not `perf_event_open` from here

The engine is already running, is sometimes root in a container, and must not
be restarted to be measured. `perf stat -p <pid>` attaches to a live process,
is inherited by threads created after the attach, and — unlike `procstat`'s
per-thread sums — keeps the counts of threads that exit mid-row, since the
kernel rolls an exiting task's counters into its parent's.

## What is deliberately not done

**Nothing is scaled.** `perf` multiplexes when more events are asked for than
the PMU has counters, and the value it prints is already extrapolated from the
fraction of the row the counter was on. That extrapolation is a model of the
row, and cannot sit in a table beside figures that are not. Below
`ENABLED_MIN` the counter is withheld and the percentage recorded in its place.
The default set is sized to fit: six events at `enabled=100.00%` on this host,
where a seventh takes the whole group to ~75%.

**No event is assumed to exist.** §7.3 is spelled in Intel event names and this
host is Zen 5, where half do not exist and the L3 has no uncore PMU. Each role
has candidate spellings, the first supported one is used, and the row records
which — `LLC-load-misses` and `ls_dmnd_fills_from_sys.dram_io_all` are both
"cache misses" and not the same measurement.

**It is off by default.** `--perf` opts in, the event set is part of the row's
harness stamp, and `perfstat.py --overhead` measures what the sidecar costs so
the answer is a number rather than an assurance.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass

#: Below this `enabled%`, a counter was multiplexed and its printed value is an
#: extrapolation. Withheld rather than reported. The same floor, and the same
#: argument, as `procstat._COVERAGE_MIN`.
ENABLED_MIN = 95.0

#: How long to wait for `perf` to open its events before assuming it attached.
#: It prints nothing on success, so there is no banner to wait for; what this
#: catches is the failure, which is immediate and loud (no such process,
#: `perf_event_paranoid` too high, unknown event).
_ATTACH_S = 0.25

#: How long to wait for `perf` to write its counts after SIGINT.
_DRAIN_S = 15.0


@dataclass(frozen=True)
class Event:
    """One counter: the row field it fills, and how to ask for it.

    `names` is a candidate list in preference order, most specific first. The
    generic kernel aliases (`LLC-load-misses`, `dTLB-load-misses`) come last
    because they are portable and vague: the kernel maps them to whatever the
    part has, which is not necessarily the demand-load event the cost model is
    about. Whichever is used is recorded on the row.
    """

    field: str
    names: tuple[str, ...]
    #: What it is, in one phrase, for the report's column help.
    means: str


#: The six hardware events plus the free software ones. Software events do not
#: consume a PMU counter, so they cost nothing to carry and are worth carrying:
#: they are the same quantities `procstat` reads from `/proc`, counted a second
#: way, and a disagreement between the two is a finding rather than a nuisance.
DEFAULT_EVENTS = (
    Event("perf_cycles", ("cycles",), "core cycles"),
    # §7.5's guard rail. `cycles / ref-cycles` is the frequency the core
    # actually ran at, and an AVX-512 comparison whose arms differ by more than
    # 3% is reported with both frequencies visible or not reported at all.
    Event("perf_ref_cycles", ("ref-cycles",), "constant-rate reference cycles"),
    Event("perf_instructions", ("instructions",), "retired instructions"),
    Event("perf_branch_misses", ("branch-misses",), "mispredicted branches"),
    # §5.2: the limiter is outstanding misses per core, and the traffic that
    # implies. A demand fill from DRAM is a line the caches did not have;
    # times 64 bytes it is the DRAM traffic §5.1's roofline is a claim about.
    Event("perf_dram_fills",
          ("ls_dmnd_fills_from_sys.dram_io_all",   # AMD Zen 4/5
           "mem_load_retired.l3_miss",             # Intel, Skylake and later
           "LLC-load-misses"),                     # generic, vague, portable
          "demand loads served from DRAM (prefetcher fills not included)"),
    # §5.5 is an argument about TLB reach that has never been checked against
    # a running engine. A page walk is what "everything in memory via mmap
    # needs care" costs when the care was not taken.
    Event("perf_dtlb_walks",
          ("ls_l1_d_tlb_miss.all_l2_miss",         # AMD Zen 4/5
           "dtlb_load_misses.walk_completed",      # Intel
           "dTLB-load-misses"),                    # generic
          "data TLB misses that reached a page walk"),
    Event("perf_page_faults", ("page-faults",), "page faults"),
    Event("perf_minor_faults", ("minor-faults",), "minor faults"),
    Event("perf_major_faults", ("major-faults",), "major faults"),
    Event("perf_ctx_switches", ("context-switches",), "context switches"),
    Event("perf_migrations", ("cpu-migrations",), "cpu migrations"),
    Event("perf_task_clock_s", ("task-clock",), "cpu time perf saw"),
)

#: The opt-in second pass. Where a line was filled *from another core's cache*
#: rather than from DRAM, which is what sharing looks like from the load-store
#: unit: two threads touching one line pull it back and forth, and the fills
#: are counted here rather than against DRAM.
#:
#: A proxy and not a proof. It cannot distinguish true sharing (two threads
#: reading one datum, which is fine and often intended) from false sharing (two
#: threads writing different data that happen to share a line, which is pure
#: waste). What it can do is *scale*: hold the work constant, raise the thread
#: count, and a per-query fill-from-cache count that climbs is a line being
#: passed around. `perf c2c` says which line; this says whether to go looking.
#: The three distances are chosen to be **disjoint**, which the obvious
#: selection is not. `ls_dmnd_fills_from_sys` is one event select with a umask
#: per source, and `remote_cache`'s umask (0x14) is a *superset* of
#: `near_cache`'s (0x04) — it is `far_cache | near_cache`. Measured on this
#: single-socket part, where `far_cache` is structurally zero, the two read
#: exactly the same number (51,991 against 51,991 in a probe), so adding them
#: to a "fills from another core's cache" total counts the near ones twice and
#: the size of the error depends on the machine's topology. `far_cache` (0x10)
#: is the disjoint spelling and is used instead; on one socket it reads zero,
#: which is correct rather than missing.
SHARING_EVENTS = (
    Event("perf_fills_local_ccx",
          ("ls_dmnd_fills_from_sys.local_ccx",),
          "demand loads filled from another core in this CCX"),
    Event("perf_fills_near_cache",
          ("ls_dmnd_fills_from_sys.near_cache",),
          "demand loads filled from a cache in another CCX on this socket"),
    Event("perf_fills_far_cache",
          ("ls_dmnd_fills_from_sys.far_cache",),
          "demand loads filled from a cache on another socket"),
    Event("perf_fills_all",
          ("ls_dmnd_fills_from_sys.all",),
          "demand fills from anywhere"),
    Event("perf_cycles", ("cycles",), "core cycles"),
    Event("perf_task_clock_s", ("task-clock",), "cpu time perf saw"),
)

EVENT_SETS: dict[str, tuple[Event, ...]] = {
    "default": DEFAULT_EVENTS,
    "sharing": SHARING_EVENTS,
}

#: Every field a row can carry from here, so a run without the sidecar writes
#: explicit nulls rather than growing a second row shape. Same contract as
#: `procstat.IO_FIELDS`.
PERF_FIELDS = tuple(dict.fromkeys(
    [e.field for evs in EVENT_SETS.values() for e in evs]
    + ["perf_set", "perf_events", "perf_enabled_pct", "perf_window_s", "perf_note",
       "perf_ref_hz"]))

_line_bytes_cache: list[int] = []


def line_bytes() -> int:
    """Bytes per cache line, for turning a fill count into DRAM traffic.

    64 on every x86-64 part this project targets, and read from sysfs anyway so
    the number in the table is the machine's rather than the assumption's.
    Cached: the report asks once per cell, and this is a file open.
    """
    if not _line_bytes_cache:
        try:
            with open("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size") as f:
                _line_bytes_cache.append(int(f.read().strip()))
        except (OSError, ValueError):
            _line_bytes_cache.append(64)
    return _line_bytes_cache[0]


_supported_cache: dict[str, bool] = {}

#: Passed as a spare argv element to the calibration's spin, so the harness's
#: contamination check can recognise it. `workloads._OUR_SCRIPTS` matches it.
#:
#: An interpreter is only "ours" when it is running one of the harness's own
#: scripts, which is deliberate — a foreign `python3 train.py` at 800% CPU used
#: to be invisible to the very check that exists to see it. A bare `python3 -c`
#: is therefore foreign, correctly, and this calibration *is* a bare
#: `python3 -c`. Without a marker it is 0.28 s of a core that the row's note
#: blames on the operator's machine.
CALIBRATION_TAG = "strawmann-perfstat-calibration"

#: The rate `ref-cycles` ticks at, calibrated once per process. A cache rather
#: than a value because `None` is a legitimate answer — no perf, or no such
#: event — and must not be retried on every row.
_ref_hz_cache: dict[str, float | None] = {}


def ref_cycles_hz() -> float | None:
    """How fast `ref-cycles` counts, in Hz, measured rather than assumed.

    Needed to turn `cycles / ref-cycles` — which is a *ratio* to the nominal
    rate — into a frequency. The kernel exposes the TSC rate only on some
    configurations (`tsc_freq_khz` is absent on this host), so it is calibrated,
    which is the same choice and the same reasoning as
    `bench/micro/perfctr.zig`'s `nominalTscHz`: one path, reproducible across
    hosts.

    A busy spin of known wall time is the whole method. `ref-cycles` counts
    *unhalted* reference cycles, and a core spinning is never halted, so over
    that window its count divided by the elapsed seconds is the rate. Measured
    on this host at 1.996 GHz idle and 1.995 GHz with every core loaded — the
    constancy is the property being relied on, and it is worth 200 ms to check
    it on a host rather than to assume it.

    **Call this before a measurement, never inside one.** It spins a core for
    200 ms, and a row brackets its own foreign-load sample around everything
    that happens between its two CPU samples: called lazily from `stop()`, the
    first row measured with `--perf` paid for the calibration *and* had it
    counted against the machine's quiet-host budget. On a 1.5 s row that is 18%
    of a core, over the 5% threshold at which a row names what else was
    running — so the harness would report the operator's box as contaminated by
    an instrument the harness itself had just started. `workloads.py` warms it
    in the `--perf` preamble, outside every row; `CALIBRATION_TAG` is the
    backstop for anyone who does not.
    """
    if "hz" in _ref_hz_cache:
        return _ref_hz_cache["hz"]
    _ref_hz_cache["hz"] = None
    perf = perf_path()
    if perf is not None and supported("ref-cycles"):
        spin = ("import time\n"
                "t = time.monotonic()\n"
                "while time.monotonic() - t < 0.2: pass\n")
        try:
            r = subprocess.run([perf, "stat", "-x,", "-e", "ref-cycles,task-clock",
                                "--", "python3", "-c", spin, CALIBRATION_TAG],
                               capture_output=True, text=True, timeout=60)
            got = parse(r.stderr or "", {"ref-cycles": "ref", "task-clock": "clock"})
            ref, clock = got.get("ref"), got.get("clock")
            if ref and clock:
                _ref_hz_cache["hz"] = ref / clock
        except (OSError, subprocess.SubprocessError):
            pass
    return _ref_hz_cache["hz"]


def perf_path() -> str | None:
    return shutil.which("perf")


def attach_blocked(pid: int) -> str | None:
    """Why this process cannot be counted, before trying, or None.

    One case, and it is the one that matters here: Qdrant runs in a container
    as **root**, the harness runs as the operator, and `perf_event_open` on
    another user's task needs ptrace permission — which `perf_event_paranoid`
    does not grant at any setting. Measured on this host with paranoid already
    at `-1`, attaching to the containerised engine fails, and perf's own
    diagnosis is actively misleading:

        Access to performance monitoring and observability operations is
        limited. Consider adjusting /proc/sys/kernel/perf_event_paranoid

    It is already `-1`. Following that advice changes nothing, and a reader who
    follows it and sees no improvement learns nothing about why. So the
    ownership case is named here instead, with the two things that actually
    work.
    """
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None
    me = os.geteuid()
    if uid == me or me == 0:
        return None
    return (f"the engine runs as uid {uid} and this harness as uid {me}; "
            f"perf_event_open on another user's process needs ptrace permission, "
            f"which perf_event_paranoid does not grant at any value (it is "
            f"{(_read_paranoid() or '?')} here). A containerised engine is this "
            f"case. Either run the harness as root, or grant perf itself the "
            f"capability once: setcap cap_perfmon,cap_sys_ptrace+ep $(which perf)")


def _read_paranoid() -> str | None:
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return f.read().strip()
    except OSError:
        return None


def supported(name: str) -> bool:
    """Whether this host can count `name`, asked of `perf` rather than guessed.

    Two different failures, both of which have to be caught here: an event the
    tool has never heard of makes `perf stat` exit non-zero with a syntax
    error, and an event it knows but the part does not implement is accepted
    and prints `<not supported>`. Guessing from `perf list` would catch the
    first and miss the second.

    Cached per process. Each probe runs `true` under `perf`, which costs a few
    milliseconds, and the answer cannot change under a run.
    """
    if name in _supported_cache:
        return _supported_cache[name]
    perf = perf_path()
    if perf is None:
        _supported_cache[name] = False
        return False
    try:
        r = subprocess.run([perf, "stat", "-x,", "-e", name, "--", "true"],
                           capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0 and "<not supported>" not in r.stderr
    except (OSError, subprocess.SubprocessError):
        ok = False
    _supported_cache[name] = ok
    return ok


def resolve(events: tuple[Event, ...]) -> list[tuple[Event, str]]:
    """Pair each event with the first spelling this host supports.

    An event with no supported spelling is dropped from the group rather than
    failing the set: a host without the DRAM-fill event should still get IPC
    and branch misses, and the row will say what it counted.
    """
    out = []
    for e in events:
        for name in e.names:
            if supported(name):
                out.append((e, name))
                break
    return out


def parse(text: str, chosen: dict[str, str]) -> dict:
    """`perf stat -x,` output into row fields.

    The CSV is `value,unit,event,run_ns,enabled_pct,...`, with `value` replaced
    by `<not supported>` or `<not counted>` where there is no number. `event`
    comes back as the spelling that was asked for, which is why the caller's
    map from event name to row field is passed in rather than re-derived: two
    roles can resolve to the same generic alias on a part that has neither
    specific one, and the row must not silently take one event's count twice.
    """
    out: dict = {}
    enabled: list[float] = []
    #: Fields dropped because the counter was multiplexed, and fields the part
    #: never produced. Two different blanks, and the row says which it is.
    multiplexed: list[str] = []
    missing: list[str] = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw, unit, name = parts[0].strip(), parts[1].strip(), parts[2].strip()
        field = chosen.get(name)
        if field is None:
            continue
        try:
            # `parts[3]` is the time each counter was enabled, summed over the
            # threads it was attached to — not a window. On a ten-thread engine
            # it reads five times the row's wall clock, and named as a window it
            # would say a 22-second row took 127 seconds. For an unmultiplexed
            # group it equals `task-clock`, which is where that quantity is
            # already reported, so it is dropped here and `perf_window_s` is
            # measured by the sidecar instead.
            pct = float(parts[4]) if len(parts) > 4 and parts[4] else None
        except ValueError:
            pct = None
        try:
            value = float(raw)
        except ValueError:
            # `<not supported>` / `<not counted>`: no number was produced, and
            # the difference between that and a zero is the whole point.
            out[field] = None
            missing.append(field)
            continue
        # Only events that produced a number vote on the group's `enabled%`.
        # An unsupported event reports 0.00% beside its `<not supported>`, and
        # counting that as the minimum labels a group that was never
        # multiplexed at all as "enabled 0% of the row".
        if pct is not None:
            enabled.append(pct)
        # `task-clock` is printed in milliseconds with its unit beside it;
        # everything else is a raw count.
        if unit == "msec":
            value /= 1000.0
        # Multiplexed: what is printed has been extrapolated from the fraction
        # of the row the counter was on. Not a measurement, so not reported.
        if pct is not None and pct < ENABLED_MIN:
            out[field] = None
            multiplexed.append(field)
        else:
            out[field] = value
    if enabled:
        out["perf_enabled_pct"] = round(min(enabled), 2)
    # Private, and popped by the caller: they describe the capture, not the
    # row, and `rows.json`'s shape is a contract.
    out["_multiplexed"] = multiplexed
    out["_missing"] = missing
    return out


def blank(set_name: str = "") -> dict:
    """Every field this module can produce, unmeasured.

    A row measured without the sidecar carries the same keys as one measured
    with it, holding explicit nulls. `rows.json` has four readers and runs
    already on disk; a second row shape is how one of them comes to treat
    "not measured" as zero.
    """
    out: dict = dict.fromkeys(PERF_FIELDS)
    out["perf_set"] = set_name
    out["perf_events"] = ""
    out["perf_note"] = ""
    return out


#: Sidecars that have been started and not yet stopped.
#:
#: `perf stat -p` outlives the process that started it. A row that raises
#: between `start` and `stop` — an I/O error, a `bfb` that vanished, anything
#: the harness does not catch — therefore leaves a `perf` attached to the
#: engine *permanently*: orphaned, still counting, and still holding counters
#: open across every row measured afterwards. Reproduced before this existed,
#: and it is the worst shape a leak can take here, because the damage is to
#: the measurements that follow rather than to the one that failed.
_live: set[Sidecar] = set()


def _stop_live() -> None:
    for car in list(_live):
        car.stop()


atexit.register(_stop_live)


class Sidecar:
    """`perf stat` attached to the engine for the length of one row.

    Started before the row and stopped after it, in the same bracket as the
    `/proc` snapshots, so the counters span a slightly *wider* window than the
    row's `wall_s` — the engine is idle at both ends of it, which adds a little
    to the fault and switch counts and nothing to the cycle counts. The window
    is recorded (`perf_window_s`) rather than assumed away.
    """

    def __init__(self, pid: int, set_name: str = "default") -> None:
        self.pid = pid
        self.set_name = set_name
        self.proc: subprocess.Popen | None = None
        self.chosen: dict[str, str] = {}
        self.note = ""
        #: When counting started, so the row can say how much wider than itself
        #: the counters' window was.
        self.t0: float | None = None

    def start(self) -> bool:
        perf = perf_path()
        if perf is None:
            self.note = "perf not installed"
            return False
        events = EVENT_SETS.get(self.set_name)
        if events is None:
            self.note = f"unknown event set {self.set_name!r}"
            return False
        pairs = resolve(events)
        if not pairs:
            self.note = "no event in the set is supported on this host"
            return False
        self.chosen = {name: e.field for e, name in pairs}
        # Checked here, after the set is known to be a real one: an unknown set
        # name is a programming error and is reported as one whatever the host
        # permits. Checked *before* spawning, so the row carries the real reason
        # rather than perf's advice to change a setting that is already correct.
        blocked = attach_blocked(self.pid)
        if blocked:
            self.note = blocked
            return False
        cmd = [perf, "stat", "-x,", "-e", ",".join(name for _, name in pairs),
               "-p", str(self.pid)]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            self.note = f"could not start perf: {exc}"
            return False
        # Counting begins when perf opens its events, which is as soon as it is
        # up — before the wait below, not after it.
        self.t0 = time.monotonic()
        _live.add(self)
        # It prints nothing on success, so the only thing to wait for is the
        # failure: a bad pid or a paranoid setting that forbids the attach
        # makes it exit at once, and starting a row believing it is measured
        # when it is not is the outcome worth spending 250 ms to avoid.
        time.sleep(_ATTACH_S)
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() if self.proc.stderr else "") or ""
            self.note = "perf exited immediately: " + " ".join(err.split())[:200]
            self.proc = None
            return False
        return True

    def stop(self) -> dict:
        """Counts for the window, or explicit nulls and a reason.

        `perf` writes its table on SIGINT, so the counts are collected by
        interrupting it rather than by killing it: `SIGKILL` gets a clean
        process exit and no numbers at all.
        """
        unmeasured = blank(self.set_name)
        unmeasured["perf_note"] = self.note
        _live.discard(self)
        if self.proc is None:
            return unmeasured
        try:
            self.proc.send_signal(signal.SIGINT)
            _, err = self.proc.communicate(timeout=_DRAIN_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.communicate()
            unmeasured["perf_note"] = ("perf did not write its counts within "
                                       f"{_DRAIN_S:.0f}s of SIGINT")
            return unmeasured
        except OSError as exc:
            unmeasured["perf_note"] = f"perf could not be stopped: {exc}"
            return unmeasured
        finally:
            self.proc = None
        out = dict(unmeasured)
        out.update(parse(err or "", self.chosen))
        # Wall clock from attach to interrupt, measured here rather than taken
        # from perf. The counters bracket the row from outside, so this is
        # always a little wider than the row's own `wall_s`; the excess is an
        # idle engine at both ends, which adds to the fault and switch counts
        # and nothing to the cycle counts.
        if self.t0 is not None:
            out["perf_window_s"] = round(time.monotonic() - self.t0, 3)
        # Carried on the row rather than looked up when the report is built:
        # it is a property of the host at the moment of measurement, and a
        # report rendered on another machine must not silently substitute that
        # machine's rate into this row's frequency.
        out["perf_ref_hz"] = ref_cycles_hz()
        multiplexed = out.pop("_multiplexed", [])
        missing = out.pop("_missing", [])
        # What was actually counted, in the row, because the role names above
        # are not the measurement — the spellings are.
        out["perf_events"] = " ".join(sorted(self.chosen))
        notes = []
        if multiplexed:
            notes.append(f"multiplexed below {ENABLED_MIN:.0f}% and withheld: "
                         + " ".join(sorted(multiplexed)))
        if missing:
            notes.append("no count produced for: " + " ".join(sorted(missing)))
        if len(missing) == len(self.chosen):
            notes = ["perf produced no counts for this row"]
        out["perf_note"] = "; ".join(notes)
        return out


def derive(row: dict) -> dict:
    """The ratios the raw counts exist to produce.

    Kept here rather than in the report so there is one definition of each, and
    computed from the row rather than stored on it so a row measured before a
    ratio existed still gets it.

      * **ipc** — retired instructions per core cycle.
      * **branch_mpki** — mispredicted branches per thousand instructions.
        Per instruction rather than per branch: `branches` costs a PMU counter
        that `ref-cycles` needs more, and MPKI is the comparable form anyway.
      * **freq_ratio** — `cycles / ref-cycles`, which §6.6.4 names directly:
        the effective frequency as a multiple of the nominal rate. 1.0 is the
        nominal rate, above is turbo, below is downclocking — the effect that
        can make a per-cycle win a wall-clock loss. §7.5 makes it a guard rail
        rather than a detail.
      * **effective GHz** — that ratio times the calibrated rate.

        **Not** cycles over `task-clock`, which is the obvious formula and is
        wrong for exactly the rows that matter. `task-clock` is time the task
        was *on a core*; `ref-cycles` counts only while the core is *unhalted*,
        and the two diverge whenever the engine sleeps. Measured on a real
        `-p 1` search row, where the engine blocks between queries: cycles over
        task-clock said 1.740 GHz, the ratio said 2.011 GHz, and the row's own
        `cycles/ref-cycles` of 1.0075 against a calibrated 1.996 GHz says the
        second is right. The first averages halted time into a frequency, which
        is not a frequency. On a CPU-bound row the two agree, which is why the
        error is invisible until it matters.
      * **dram_bytes** — fills times the coherency line size, which is the
        traffic §5.1's roofline is a claim about.
    """
    def num(key):
        v = row.get(key)
        return v if isinstance(v, (int, float)) else None

    cycles, instr = num("perf_cycles"), num("perf_instructions")
    ref, ref_hz = num("perf_ref_cycles"), num("perf_ref_hz")
    misses, fills = num("perf_branch_misses"), num("perf_dram_fills")
    out: dict = {}
    if cycles and instr:
        out["ipc"] = instr / cycles
    if instr and misses is not None:
        out["branch_mpki"] = misses / (instr / 1000)
    if cycles and ref:
        out["freq_ratio"] = cycles / ref
        # Only where the row recorded the rate its own reference counted at.
        # A row measured before that was recorded, or on a host where the
        # calibration failed, gets the ratio and no frequency — there is no
        # second formula to fall back to, because the plausible-looking one is
        # wrong on any row that sleeps.
        if ref_hz:
            out["ghz"] = cycles / ref * ref_hz / 1e9
    if fills is not None:
        out["dram_bytes"] = fills * line_bytes()
    return out


def per_query(value: float | None, n: float | None) -> float | None:
    """A count divided by the queries that caused it, or None.

    The only form in which two engines' hardware counters can be compared at
    all. Absolute counts are a function of how long the row ran and how many
    threads it ran on, neither of which the two engines hold equal; per query
    is the cost of answering the same question twice.
    """
    if value is None or not n:
        return None
    return value / n


def _fixture() -> str:
    """One real `perf stat -x,` capture, including both ways a count goes missing."""
    return (
        "1964145726,,cycles,976535914,100.00,,\n"
        "1901000000,,ref-cycles,976535914,100.00,,\n"
        "11682238319,,instructions,976546554,100.00,,\n"
        "116348,,branch-misses,976557104,100.00,,\n"
        "<not supported>,,ls_dmnd_fills_from_sys.dram_io_all,0,0.00,,\n"
        "150396,,ls_l1_d_tlb_miss.all_l2_miss,976557104,74.00,,\n"
        "19899,,page-faults,976561402,100.00,,\n"
        "1145,,context-switches,976563877,100.00,,\n"
        "976.56,msec,task-clock,976563877,100.00,,\n")


def _self_test() -> int:
    """The parser and the ratios, without needing an engine or a PMU.

    Every interesting case is a failure case: an event the part does not
    implement, an event that was multiplexed and therefore extrapolated, and a
    duration that arrives in different units from the counts beside it.
    """
    chosen = {"cycles": "perf_cycles", "ref-cycles": "perf_ref_cycles",
              "instructions": "perf_instructions",
              "branch-misses": "perf_branch_misses",
              "ls_dmnd_fills_from_sys.dram_io_all": "perf_dram_fills",
              "ls_l1_d_tlb_miss.all_l2_miss": "perf_dtlb_walks",
              "page-faults": "perf_page_faults",
              "context-switches": "perf_ctx_switches",
              "task-clock": "perf_task_clock_s"}
    got = parse(_fixture(), chosen)
    assert got["perf_cycles"] == 1964145726.0, got
    assert got["perf_instructions"] == 11682238319.0, got
    # `<not supported>` is not zero: the part does not implement the event.
    assert got["perf_dram_fills"] is None, got
    # Multiplexed at 74%: perf printed a number, and it is an extrapolation.
    assert got["perf_dtlb_walks"] is None, got
    # The unsupported event reports 0.00% beside its `<not supported>` and does
    # not get a vote: the group's floor is the 74% the dtlb counter ran at.
    assert got["perf_enabled_pct"] == 74.0, got
    assert got["_multiplexed"] == ["perf_dtlb_walks"], got
    assert got["_missing"] == ["perf_dram_fills"], got
    # msec, not a count. Read as a raw value this is a 976-second row.
    assert abs(got["perf_task_clock_s"] - 0.97656) < 1e-5, got
    # The window is wall clock measured by the sidecar, not perf's per-event
    # enabled time: that field is summed over the threads the counter was
    # attached to, and on a ten-thread engine it reads five times the row.
    assert "perf_window_s" not in got, got

    # An event nobody asked for (a stray line, a group leader perf added) is
    # ignored rather than landing in whichever field sorts first.
    assert "perf_cycles" not in parse("7,,made-up-event,1,100.00,,\n", {})

    clean = {k: v for k, v in got.items() if not k.startswith("_")}
    d = derive(clean)
    assert abs(d["ipc"] - 5.9477) < 1e-3, d
    assert abs(d["branch_mpki"] - 0.00996) < 1e-4, d
    assert abs(d["freq_ratio"] - 1.0332) < 1e-3, d
    # No calibrated reference rate on this row, so there is a ratio and no
    # frequency. Emphatically not `cycles / task_clock`, which this fixture
    # would make 2.011 GHz: that formula averages halted time into a frequency
    # and is wrong by 14% on a real sleeping row.
    assert "ghz" not in d, d
    with_rate = derive({**clean, "perf_ref_hz": 1.9958e9})
    assert abs(with_rate["ghz"] - 2.0621) < 1e-3, with_rate
    # A row whose ref-cycles went missing has neither, rather than a frequency
    # computed from whatever else is lying around.
    no_ref = derive({**clean, "perf_ref_cycles": None, "perf_ref_hz": 1.9958e9})
    assert "ghz" not in no_ref and "freq_ratio" not in no_ref, no_ref
    # No fills were measured, so there is no DRAM traffic figure. Not zero
    # bytes: an engine that reads nothing from DRAM and one whose counter is
    # missing must not print the same thing.
    assert "dram_bytes" not in d, d

    full = dict(got, perf_dram_fills=1000.0)
    assert derive(full)["dram_bytes"] == 1000 * line_bytes()

    assert per_query(6400.0, 3200) == 2.0
    assert per_query(6400.0, 0) is None
    assert per_query(None, 10) is None

    # The row shape is a contract: a run without the sidecar still writes every
    # key, or `rows.json` grows a second shape.
    row = Sidecar(1, "default").stop()
    assert set(row) == set(PERF_FIELDS), set(PERF_FIELDS) ^ set(row)
    assert row["perf_set"] == "default" and row["perf_cycles"] is None, row
    assert set(blank()) == set(PERF_FIELDS), set(blank())

    # An unknown set is refused with a reason rather than measuring nothing
    # quietly.
    s = Sidecar(1, "nope")
    assert s.start() is False and "unknown event set" in s.note, s.note

    # A target owned by another user cannot be counted, whatever
    # `perf_event_paranoid` says, and the note has to say that rather than
    # repeat perf's advice to change a setting that is already correct. pid 1
    # is root's; this assertion is skipped when the tests run as root.
    if os.geteuid() != 0:
        why = attach_blocked(1)
        assert why and "ptrace" in why and "setcap" in why, why
    assert attach_blocked(os.getpid()) is None, "our own process is attachable"

    # A sidecar that started is tracked until it is stopped, and `_stop_live`
    # is what closes the leak: `perf stat -p` outlives the process that started
    # it, so a row that raises between start and stop leaves one attached to
    # the engine for every row measured afterwards. Reproduced against a live
    # engine before this existed.
    class FakeProc:
        returncode = 0
        stderr = None

        def __init__(self): self.signalled = False
        def send_signal(self, _sig): self.signalled = True
        def communicate(self, timeout=None): return ("", "")

    car = Sidecar(1)
    car.proc, car.t0 = FakeProc(), time.monotonic()
    _live.add(car)
    assert car in _live
    _stop_live()
    assert car not in _live, "a stopped sidecar must leave the live set"
    assert not _live, _live
    # ...and stopping twice is not an error: `atexit` runs after a row that
    # already stopped its own.
    _stop_live()

    print("ok: perfstat parser, ratios and row shape")
    return 0


def _overhead(reps: int = 5) -> int:
    """What the sidecar costs the thing it measures, as a number.

    §7.1's discipline applied to the instrument itself: a counter that changes
    the row is not measuring the row. A fixed CPU-bound child is run `reps`
    times with the sidecar attached and `reps` times without, alternating, and
    the difference in its runtime is printed against the spread of the arm it
    is compared to. Alternating rather than one arm then the other, for the
    same reason §7.2(5) interleaves the engines.

    This is a floor, not a ceiling: the child here is one thread doing
    arithmetic, and the sidecar's cost is a function of how often the engine
    switches, faults and forks. Run it, then check a real row.
    """
    child = ("import time\n"
             "t = time.monotonic()\n"
             "x = 0\n"
             "while time.monotonic() - t < 2.0: x += 1\n"
             "print(x)\n")

    def once(with_perf: bool) -> float | None:
        p = subprocess.Popen(["python3", "-c", child],
                             stdout=subprocess.PIPE, text=True)
        car = Sidecar(p.pid)
        started = car.start() if with_perf else False
        if with_perf and not started:
            print(f"  cannot attach: {car.note}")
            p.kill(), p.communicate()
            return None
        out = p.communicate()[0]
        if with_perf:
            car.stop()
        try:
            return float(out.strip())
        except ValueError:
            return None

    arms: dict[bool, list[float]] = {True: [], False: []}
    for _ in range(reps):
        for with_perf in (False, True):
            v = once(with_perf)
            if v is None:
                return 1
            arms[with_perf].append(v)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    bare, watched = med(arms[False]), med(arms[True])
    spread = (max(arms[False]) - min(arms[False])) / bare * 100
    delta = (watched - bare) / bare * 100
    print(f"  without perf: {bare:,.0f} iterations (median of {reps}, "
          f"spread {spread:.2f}%)")
    print(f"  with perf:    {watched:,.0f} iterations (median of {reps})")
    print(f"  sidecar cost: {-delta:+.2f}% of the work done, against a "
          f"{spread:.2f}% spread on the unwatched arm")
    if abs(delta) <= spread:
        print("  -> below this arm's own run-to-run spread")
    else:
        print("  -> LARGER than this arm's spread: the sidecar is visible in "
              "the measurement, and rows measured with it are not comparable "
              "against rows measured without it")
    return 0


if __name__ == "__main__":
    import json
    import sys

    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    if "--overhead" in sys.argv:
        raise SystemExit(_overhead())
    if "--events" in sys.argv:
        at = sys.argv.index("--events") + 1
        name = sys.argv[at] if len(sys.argv) > at else "default"
        for e, spelling in resolve(EVENT_SETS[name]):
            print(f"  {e.field:<24} {spelling:<40} {e.means}")
        missing = [e.field for e in EVENT_SETS[name]
                   if e.field not in {x.field for x, _ in resolve(EVENT_SETS[name])}]
        if missing:
            print("  unsupported on this host: " + " ".join(missing))
        raise SystemExit(0)

    # No argument: attach to whatever engine is running for a few seconds, so
    # the thing can be pointed at a live server without a whole run.
    import procstat
    procs = procstat.engine_processes()
    if not procs:
        print("no engine process found", file=sys.stderr)
        raise SystemExit(1)
    car = Sidecar(procs[0][0])
    if not car.start():
        print(car.note, file=sys.stderr)
        raise SystemExit(1)
    time.sleep(float(os.environ.get("PERF_SECONDS", "3")))
    row = car.stop()
    print(json.dumps({**row, **derive(row)}, indent=2))
