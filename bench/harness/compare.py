#!/usr/bin/env python3
"""Join two workload result sets into one table.

Replaces `compare.sh`. The arithmetic is trivial; what is not trivial is
refusing to present numbers that should not be compared, and that is the part
shell made awkward:

  * a row measured under foreign load is not comparable to one measured idle,
    and the difference is invisible in the number itself;
  * a construct the engine declines is not a failure and must not read as one,
    whether it is a permanent §1 non-goal (sharding, sparse vectors) or an
    unbuilt phase of §2's surface (payload, filtering: M7, optional per §10);
  * the open-loop `--rps` rows report the *offered* rate, so "1.00x" there means
    "neither engine saturated", not "they are equally fast", printing a ratio
    at all is misleading;
  * `qps` and `rps` differ wherever `--search-batch-size > 1`;
  * §7.4: QPS is compared at equal recall, never at equal `ef`. A search row
    gets a ratio only when both engines' recall@10 at that row's `ef` is known
    and within 0.01; otherwise the cell says why not;
  * two result sets are only comparable if the same harness produced them.
    `run.json` carries a `harness` stamp (metric, query source, sizes,
    collection settings); labels whose stamps disagree, or lack one, are STALE
    and get no ratios at all;
  * a ratio is a comparative claim, and §8 licenses one only through the
    conformance differ's `licenses_comparative`. Without that the table says
    UNLICENSED above the numbers, in every rendering.

Usage:
    compare.py <label-a> <label-b>              full table
    compare.py <label-a> <label-b> --readme     the compact block README embeds
    compare.py <label-a> <label-b> --write-readme   splice it into README.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import procstat
import recall as recall_mod
import regression
from workloads import BUILD_KEYS, STAMP_KEYS, W11_MIN_OVERLAP, Gate, Status, stamp_hash

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))


def load(label: str, required: bool = True) -> dict[str, dict]:
    path = ROOT / "bench/results" / label / "rows.json"
    if not path.exists():
        if not required:
            return {}
        raise SystemExit(f"no results for {label!r} at {path}\n"
                         f"run: bench/harness/workloads.py run <uri> {label}")
    return {r["id"]: r for r in json.loads(path.read_text())}


def read_json(label: str, name: str) -> dict:
    p = ROOT / "bench/results" / label / name
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def gate_of_rows(rows, env: str = "") -> str:
    """The §7.1 verdict a set of rows was measured under. One definition.

    From the rows first: each carries the gate at its own measurement time
    (`Result.gate`), and `env.txt` is written per `workloads.py` invocation
    while `fullrun` calls it more than once per arm, so the file describes the
    last few rows rather than the table. Rows that disagree read `mixed`.

    Both halves of that were learned the hard way. A later single-row run on a
    gated host used to re-vouch for a table measured under `--lax`; and an arm
    whose search rows were stamped `FAIL` was reported as
    `pass`, because `report.py` had its own copy of this rule that read the
    file. It no longer has one. `rows` is a sequence of row dicts or the
    mapping `load()` returns.
    """
    values = rows.values() if hasattr(rows, "values") else rows
    gates = {r.get("gate") for r in values if r.get("gate")}
    if len(gates) > 1:
        return Gate.mixed
    if gates:
        # Coerced, because rows come back from JSON as plain strings and a
        # caller should not have to care which side of the round trip it is on.
        # A value this enum does not know is returned as it is rather than
        # raising: an old row is a thing to report, not to crash the report.
        got = gates.pop()
        try:
            return Gate(got)
        except ValueError:
            return got
    # Rows measured before the field existed: the file is all there is, and
    # since it is appended per invocation, the *last* verdict in it is the one
    # that describes the most recent rows. "checks passed" anywhere used to do,
    # which reads a pass out of a file whose first block failed.
    ok, bad = env.rfind("checks passed"), env.rfind("check(s) failed")
    if ok < 0 and bad < 0:
        return Gate.failed
    return Gate.passed if ok > bad else Gate.failed


def gate_of(label: str) -> str:
    """`gate_of_rows` for a result label on disk."""
    env_p = ROOT / "bench/results" / label / "env.txt"
    env = env_p.read_text() if env_p.exists() else ""
    return gate_of_rows(load(label, required=False), env)


def build_identities(rows: dict[str, dict]) -> dict[tuple, list[str]]:
    """The distinct build identities the rows carry, each with its row ids.

    Rows without one (measured before `BUILD_KEYS` were stamped) are not an
    identity: unknown is already refused per row by the harness stamp, and
    counting it as a second build would refuse every label with one legacy
    row for a reason that is not the real one.
    """
    out: dict[tuple, list[str]] = {}
    for wid, r in rows.items():
        ident = tuple(r.get(k) for k in BUILD_KEYS)
        if any(v is not None for v in ident):
            out.setdefault(ident, []).append(wid)
    return out


def mixed_build_reasons(label: str, rows: dict[str, dict] | None = None) -> list[str]:
    """Why a label's rows are not one measurement, or `[]`.

    `rows.json` merges in place and `run.json` is rewritten whole, so a
    single-row re-run on a rebuilt binary leaves a file whose label-level
    record describes the new build and whose rows were mostly served by the
    old one. Rows carrying more than one build identity is that file, and no
    ratio from it is a claim about either build.
    """
    if rows is None:
        rows = load(label, required=False)
    idents = build_identities(rows)
    if len(idents) <= 1:
        return []
    parts = []
    for ident, wids in idents.items():
        named = ", ".join(f"{k}={v}" for k, v in zip(BUILD_KEYS, ident) if v is not None)
        parts.append(f"{', '.join(wids[:4])}{'...' if len(wids) > 4 else ''}: {named}")
    if commit_label_only(idents):
        # Identical binary, identical ISA/optimize/profile: one build served
        # every row, and what differs is the repository commit the tree was at
        # when each was written. A commit touching only the harness or the docs
        # produces a byte-identical engine, and a run long enough to span one
        # is not a mixed build. Refusing it printed "a subset re-run on a
        # rebuilt binary" over rows whose `engine_binary` agreed to the sha256,
        # which is a false diagnosis rather than a strict one.
        #
        # Not silent: `commit_drift` reports it, and the report prints it in
        # the provenance table. The binary is the identity; the commit is a
        # label on it, and both belong in the record.
        return []
    return [f"{label}: rows were measured on {len(idents)} different builds "
            f"(a subset re-run on a rebuilt binary) -- " + "; ".join(parts)]


def commit_label_only(idents: dict) -> bool:
    """Whether these build identities differ *only* in the commit label.

    True when every group names the same non-null `engine_binary` and agrees on
    every other key. A build identity with no binary hash cannot answer this and
    is never treated as equal: unknown is not a match.
    """
    if len(idents) <= 1:
        return True
    keys = list(BUILD_KEYS)
    try:
        i_build, i_binary = keys.index("engine_build"), keys.index("engine_binary")
    except ValueError:                                  # pragma: no cover
        return False
    binaries = {ident[i_binary] for ident in idents}
    if len(binaries) != 1 or None in binaries:
        return False
    rest = {tuple(v for k, v in enumerate(ident) if k not in (i_build, i_binary))
            for ident in idents}
    return len(rest) == 1


def commit_drift(label: str, rows: dict[str, dict] | None = None) -> str:
    """The repository commits a label's rows were written across, or "".

    One binary, more than one commit label: the run outlasted a commit to the
    tree. Worth printing beside the provenance and not worth refusing, because
    the engine that served every row is the same one — see `mixed_build_reasons`.
    """
    if rows is None:
        rows = load(label, required=False)
    idents = build_identities(rows)
    if len(idents) <= 1 or not commit_label_only(idents):
        return ""
    i_build = list(BUILD_KEYS).index("engine_build")
    seen = list(dict.fromkeys(ident[i_build] for ident in idents))
    return (f"measured across {len(seen)} repository commits "
            f"({', '.join(str(c) for c in seen)}); one binary served every row")


#: `STAMP_KEYS` (the `run.json` fields two labels have to agree on) lives in
#: `workloads`, beside the stamp and the per-row hash of it.


def stale_reasons(a_label: str, b_label: str) -> list[str]:
    """Why the two labels cannot be ratioed, or `[]`.

    A missing stamp is a reason, not a pass: the published result sets predate
    the stamp, the Euclid default and the dataset queries, and their tables
    carried ratios for weeks with nothing on the page saying so. Refusing on
    "unknown" is the only rule that catches that.
    """
    out = []
    ma, mb = read_json(a_label, "run.json"), read_json(b_label, "run.json")
    for lbl, m in ((a_label, ma), (b_label, mb)):
        if not m:
            out.append(f"{lbl}: no run.json")
        elif not m.get("harness"):
            out.append(f"{lbl}: run.json has no harness stamp (measured before the "
                       f"harness recorded metric, query source and collection settings)")
    if out:
        return out
    ha, hb = ma["harness"], mb["harness"]
    for k in STAMP_KEYS:
        if ha.get(k) != hb.get(k):
            va, vb = ha.get(k), hb.get(k)
            if isinstance(va, dict) and isinstance(vb, dict):
                diff = sorted(set(va) ^ set(vb) | {x for x in va if x in vb and va[x] != vb[x]})
                out.append(f"harness.{k} differs on {', '.join(diff[:6])}")
            else:
                out.append(f"harness.{k}: {a_label}={va!r} {b_label}={vb!r}")
    da = (ma.get("dataset") or {}).get("name")
    db = (mb.get("dataset") or {}).get("name")
    if da != db:
        out.append(f"dataset: {a_label}={da!r} {b_label}={db!r}")
    for lbl in (a_label, b_label):
        out += mixed_build_reasons(lbl)
        out += orphaned_row_reasons(lbl)
    return out


def orphaned_row_reasons(label: str) -> list[str]:
    """Rows this label's `run.json` cannot be describing, because they predate it.

    `run.json` is rewritten wholesale at the start of an arm and `rows.json` merges
    in place, so an arm that starts and dies leaves the *metadata* of the new run —
    gate verdict, host, stamp, timestamps — over the *rows* of the old:
    `qd-dbp100k` once held 26 rows measured 05:00-06:00Z under a `run.json`
    saying the run started at 17:43Z and passed its gate.

    `foreign_dataset` does not catch it: both runs measured the same corpus, which
    is exactly when the merge is silent. That rule protects against two corpora
    under one label; this protects against two runs. Refused rather than bannered,
    for the same reason: there is no reading of a row whose gate verdict belongs to
    a different measurement.
    """
    meta = read_json(label, "run.json")
    session = meta.get("session")
    if not session:
        # Result sets written before `--session` existed carry no id, and every
        # row in them is equally unvouched-for. Refusing on "unknown" here
        # would refuse every published table retroactively, and unlike the
        # harness stamp there is a cheap forward fix: the next run stamps.
        return []
    # Not `read_json`: it coerces to dict and `rows.json` is a list, so it
    # would return {} and this check would silently never fire.
    path = ROOT / "bench/results" / label / "rows.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = rows if isinstance(rows, list) else list((rows or {}).values())
    stale = sorted({r["id"] for r in rows
                    if isinstance(r, dict) and r.get("id")
                    and r.get("session") != session})
    if not stale:
        return []
    shown = ", ".join(stale[:6]) + ("…" if len(stale) > 6 else "")
    return [f"{label}: {len(stale)} row(s) were not measured by the run their "
            f"run.json describes ({shown}); the arm was re-run and these rows "
            f"are an earlier run's, carrying this run's gate verdict and host. "
            f"Delete bench/results/{label} and re-run the arm whole."]


def conformance_recovery(a_label: str, b_label: str) -> str:
    """How to license rows that are already measured.

    §8's differ runs last, so a run stopped early loses the licence for data that
    is otherwise complete — which is what happened. The rows
    survived; the row that permits quoting them did not.

    It is recoverable: the conformance phase starts its own engines and ignores
    `--skip strawmann`/`--skip qdrant`, so it can run on its own against labels
    already on disk. Half an hour against nine hours of rows, and the only reason
    it was not done is that nothing said it could be.

    **The Qdrant source has to be carried over.** `--qdrant-binary` defaults to the
    pinned image, and the first version of this command omitted it — so rows from a
    source-built Qdrant would have been licensed by a conformance row taken against
    `qdrant/qdrant:v1.19.0`. §8 asks for the same build, and nothing downstream
    compares the two version strings, so the substitution would have been silent
    and this banner would have caused it. The flag is emitted from `run.json`.
    """
    ds = binary = ""
    builds = []
    for label in (a_label, b_label):
        meta = read_json(label, "run.json") or {}
        name = (meta.get("dataset") or {}).get("name")
        if name and not ds:
            ds = f" --dataset {name}"
        path = (meta.get("qdrant") or {}).get("binary")
        if path and not binary:
            binary = f" --qdrant-binary {path}"
        if commit := (meta.get("strawmann") or {}).get("commit"):
            builds.append(f"strawmANN {commit}")
        if version := (meta.get("qdrant") or {}).get("version"):
            builds.append(f"Qdrant {version}")
    out = (f" The rows are measured; only the licence is missing, and the differ "
           f"can be run on its own against them: `python3 bench/harness/fullrun.py"
           f"{ds}{binary} --skip build --skip strawmann --skip qdrant "
           f"--strawmann-label {a_label} --qdrant-label {b_label}`.")
    if binary:
        out += (" Without `--qdrant-binary` that command runs the pinned image "
                "instead, and would license these rows with a conformance row "
                "from a different Qdrant.")
    if builds:
        # Neither the flag nor `--skip build` can guarantee this: the differ
        # records the commit the *checkout* is on, not the one that built the
        # binary in `zig-out`. §8 asks for a green row from the same build, so
        # the operator is told which one that is rather than left to assume
        # the harness arranged it.
        out += (f" These rows were measured at {' and '.join(builds)}; §8 wants "
                f"the row from those builds, so check the checkout before "
                f"trusting the one it writes.")
    return out


def licence(a_label: str, b_label: str) -> dict:
    """What §8 says these numbers may be used for.

    Read from the differ's `conformance.json`, which `fullrun.py` copies to
    both labels. `licenses_perf` (T1) says a number may be published;
    `licenses_comparative` (T3) says the two engines are at equal recall and a
    ratio between them means something. The tables printed ratios on
    `licenses_perf` alone, which is the first half of §8's sentence.
    """
    ca, cb = read_json(a_label, "conformance.json"), read_json(b_label, "conformance.json")
    conf = ca or cb
    out = {"present": bool(conf), "perf": bool(conf.get("licenses_perf")),
           "comparative": bool(conf.get("licenses_comparative")),
           "tier": conf.get("tier_reached"), "hash": conf.get("hash"),
           "banner": ""}
    if not conf:
        out["banner"] = ("UNLICENSED: no conformance row (bench/results/<label>/"
                         "conformance.json). §8: no performance number is publishable "
                         "without one, and none of these is a comparative claim."
                         + conformance_recovery(a_label, b_label))
    elif ca and cb and ca.get("hash") != cb.get("hash"):
        out["banner"] = (f"UNLICENSED: the two labels carry different conformance rows "
                         f"({ca.get('hash')} vs {cb.get('hash')}); one comparison needs one row.")
        out["comparative"] = False
    elif not out["comparative"]:
        out["banner"] = (f"NOT A LICENSED COMPARATIVE CLAIM: the differ reached "
                         f"{out['tier'] or 'an unknown tier'} and set licenses_comparative="
                         f"false, so the engines are not at verified equal recall (§8.5 T3). "
                         f"The numbers may be quoted per engine"
                         + ("" if out["perf"] else " -- and not even that: licenses_perf is false")
                         + "; the ratios between them are not a result."
                         + segment_confound(a_label, b_label))
    return out


def segment_confound(a_label: str, b_label: str) -> str:
    """The measured segment counts, when T3 was refused and they differ.

    "Not at verified equal recall" is a true refusal and a useless one: it names
    the tier that failed, not the reason. On the dbpedia-openai-1m pair the reason
    was in `collections.json` in the same report — strawmANN held the one segment
    the run asked for and Qdrant held four.

    Qdrant searches every *populated* segment at the full `ef` and merges, so N
    segments explore N graphs where strawmANN explores one, and its recall at a
    fixed `ef` rises accordingly. The count alone overstates it, though: at
    200,000 x 128 the two segments are one `hnsw` holding every point plus an
    empty appendable, so there is no second graph. At 990,000 x 1536 the five are
    four populated graphs (188,200/208,600/251,100/342,100) and one empty — a real
    four-to-one.

    Said only when the counts differ, so a run that does reach equal segments gets
    the plain refusal.
    """
    counts = {}
    for label in (a_label, b_label):
        doc = read_json(label, "collections.json")
        for c in doc.get("collections") or []:
            if c.get("collection") == "bench2" and c.get("segments_count"):
                counts[label] = int(c["segments_count"])
    if len(counts) < 2 or len(set(counts.values())) < 2:
        return ""
    said = ", ".join(f"{k} held {v} segment{'' if v == 1 else 's'}"
                     for k, v in counts.items())
    return (f" The two engines did not search the same number of graphs: {said}. "
            f"Qdrant searches every populated segment at the full ef and merges, so "
            f"more segments explore more graphs and its recall at a fixed ef rises; "
            f"one of its segments is an empty appendable that contributes nothing, so "
            f"the count is an upper bound on the graphs. Set --max-segment-size above "
            f"the corpus and its optimizer merges to one, which is what the harness "
            f"now asks for.")


def env_mismatch(a_label: str, b_label: str) -> list[str]:
    """Where the two runs' environments disagree, as notes.

    Not the harness stamp (that is `stale_reasons` and refuses); this is the
    rest of `run.json` and `env.txt`: host, gate, sizes. A mismatch here is
    printed beside the table rather than blocking it, because §7.1 already
    stamps every row development-grade on a failing gate.
    """
    ma, mb = read_json(a_label, "run.json"), read_json(b_label, "run.json")
    out = []
    for key, get in (("host", lambda m: (m.get("host") or {}).get("hostname")),
                     ("cpu", lambda m: (m.get("host") or {}).get("cpu")),
                     ("dataset", lambda m: (m.get("dataset") or {}).get("name")),
                     ("upload_n", lambda m: m.get("upload_n")),
                     ("queries", lambda m: m.get("queries")),
                     ("gate", lambda m: m.get("gate"))):
        va, vb = get(ma), get(mb)
        if va != vb:
            out.append(f"{key}: {a_label}={va!r} {b_label}={vb!r}")
    ga, gb = gate_of(a_label), gate_of(b_label)
    if ga != gb:
        out.append(f"env.txt gate: {a_label}={ga} {b_label}={gb}")
    return out


#: Recall@10 within this of each other counts as equal (`results.py compare`
#: uses the same rule).
RECALL_TOL = 0.01

#: Said once per page instead of once per row. Every rescoring row (W6, W7, W8
#: and their `ef` sweeps) is refused for the same reason, and stamping the
#: sentence onto each of nine rows read as nine problems.
RESCORE_CAUSE = ("Rows marked <em>rescore pools differ</em> are refused for one "
                 "shared reason: strawmANN rescores <code>max(asked, ef)</code> "
                 "candidates and Qdrant rescores <code>limit</code> of them, so at "
                 "equal nominal parameters they are not answering the same question "
                 "(decisions §5). Expected, not a defect.")

#: Recall@10 below which a row is not answering the question well enough for its
#: speed to mean anything, whichever engine is better at it.
#:
#: `RECALL_TOL` alone cannot catch this: binary quantization at d=128 measured
#: 0.0273 against 0.0265, which is *equal* to well inside the tolerance, so the
#: rule licensed a ratio comparing how fast two engines return the wrong answer.
#: That was patched with a sentence hard-coded onto W7 saying binary
#: quantization is unusable "at d=128" — which the dbpedia run then printed
#: verbatim under a row whose own recall was 0.9076, and which suppressed a
#: ratio that qualified. A floor is that sentence stated as a rule, so it fires
#: on the geometry that earns it and stays quiet on the one that does not.
RECALL_FLOOR = 0.50

#: Open-loop rows whose client p50 is past this are not serving the offered
#: rate, whatever the qps column says.
SATURATED_P50_US = 1_000_000.0

#: How much of an open-loop row's offered rate has to arrive before the row
#: counts as having served it. 5% short is the load generator or the engine
#: failing to keep up, and either way the row is not "both engines kept up".
SERVED_FRACTION = 0.95


class Row:
    """One joined row, with the arithmetic and the refusals in one place."""

    def __init__(self, wid: str, ra: dict, rb: dict, a_label: str, b_label: str,
                 rec_a: float | None, rec_b: float | None, recall_required: bool,
                 stale: bool, stamps: tuple[dict | None, dict | None] = (None, None)):
        self.id = wid
        self.ra, self.rb = ra, rb
        self.a_label, self.b_label = a_label, b_label
        self.rec_a, self.rec_b = rec_a, rec_b
        self.recall_required = recall_required
        self.stale = stale
        #: Each label's current `run.json` harness stamp, for the per-row check.
        self.stamps = stamps
        self.notes: list[str] = []
        #: Why the ratio was refused for a reason that is not about recall
        #: (STALE, contamination, an unstamped or re-run row, a `ratio_policy`),
        #: or "". `report.py` reads it wherever it forms a figure from these
        #: rows other than the per-row ratio: a matched-recall table built on
        #: a contaminated W10 row is the same refused claim in another shape.
        self.refusal = ""
        self.qa, self.qb = ra.get("qps"), rb.get("qps")
        self.sa = f"{self.qa:,.0f}" if self.qa is not None else ra["status"]
        self.sb = f"{self.qb:,.0f}" if self.qb is not None else rb["status"]
        self.ratio = self._ratio()

    def _refuse(self, why: str) -> str:
        if not self.refusal:
            self.refusal = why
        return "-"

    def _ratio(self) -> str:
        ra, rb, wid = self.ra, self.rb, self.id
        dirty = []
        for lbl, r in ((self.a_label, ra), (self.b_label, rb)):
            if r.get("foreign"):
                self.notes.append(f"[{lbl} contaminated: {r['foreign']}]")
                dirty.append(lbl)
            if r["status"] == Status.not_applicable:
                # "declined" rather than "non-goal": §1's exclusions are
                # permanent and W12's filtering is not one of them, so the word
                # that covered both said the stronger thing about both.
                self.notes.append(f"[{lbl}: declined, not implemented]")
            elif r["status"] == Status.failed:
                self.notes.append(f"[{lbl}: FAILED]")
            elif r["status"] == Status.no_output:
                self.notes.append(f"[{lbl}: bfb exited 0 with no output; not measured]")

        # Both engines ran one workload, so their row notes are usually the
        # same sentence twice: W5's "per-batch latency (16 queries/request)"
        # was printed under each label in the front-page block. Identical
        # notes are said once, without a label.
        na, nb = trim_row_note(ra.get("notes") or ""), trim_row_note(rb.get("notes") or "")
        if na and na == nb:
            self.notes.append(f"[{na}]")
        else:
            for lbl, note in ((self.a_label, na), (self.b_label, nb)):
                if note:
                    self.notes.append(f"[{lbl}: {note}]")

        # Asked before the open-loop branch, because these two are gates on the
        # *row* and not on the ratio: a contaminated or STALE W4-sat row used
        # to answer "offered" and leave `refusal` empty, and `report.py` reads
        # that flag wherever it forms another figure from these rows. The gate
        # held for the ratio column and nowhere else.
        if self.stale:
            return self._refuse("STALE: the two labels were not measured by the same "
                                "harness configuration")
        # A row measured under foreign load is not comparable to one measured
        # idle, and the difference is invisible in the number: `decisions.md`
        # makes the per-row flag a hard gate, and a `2.00x` with a
        # contamination note beside it was a soft one.
        if dirty:
            return self._refuse(f"contaminated: {', '.join(dirty)}")
        # A filtered-search row whose engine had no payload index measured a
        # full scan. Asked here, with the other gates on the *row*: it is not
        # about the ratio, and W12 has no ratio to lose — its cells are the
        # problem, and until this fired they read as filtered-search figures.
        no_index = self._unindexed_filter()
        if no_index:
            return self._refuse(no_index)
        if ra.get("load_mode") == "open-loop" or wid.startswith("W4-sat"):
            # The number is the offered rate; a ratio would read as a speed
            # comparison when it means "both kept up". Unless one did not:
            # a client p50 past a second is a queue, not a served rate.
            sat = []
            for lbl, r in ((self.a_label, ra), (self.b_label, rb)):
                lat = r.get("latency") or {}
                c, srv = lat.get("client_p50_us"), lat.get("server_p50_us")
                if c is not None and c > SATURATED_P50_US:
                    sat.append(lbl)
                # The *gap* leads, not the ratio. `c / srv` divides by a number
                # that is small precisely because the engine is fast, so the
                # better an engine gets the more alarming its own flag looks:
                # W13 read "23x" for strawmANN against "5x" for Qdrant while
                # strawmANN's absolute client latency was lower (0.13 ms vs
                # 0.21 ms), because its server p50 was 6 µs. What a reader needs
                # is how much time is unaccounted for by the server.
                if c and srv and srv > 0 and c / srv >= 5:
                    self.notes.append(
                        f"[{lbl}: {(c - srv) / 1000:,.1f} ms of the client's "
                        f"{c / 1000:,.1f} ms p50 is not server time "
                        f"({c / srv:,.0f}x)]")
            # Whether the offered rate was actually served, measured rather
            # than inferred. `client p50 > 1 s` catches a queue that has run
            # away; it does not catch a row that quietly delivered 93% of what
            # it was asked for, and the word "offered" then reads as "both
            # engines kept up" for a row where neither did. The rows carry
            # `rps_target` since §4's fractions replaced the constant rates, so
            # this is a subtraction rather than a guess.
            short = []
            for lbl, r in ((self.a_label, ra), (self.b_label, rb)):
                target, got = r.get("rps_target"), r.get("qps")
                if target and got and got < target * SERVED_FRACTION:
                    short.append(f"{lbl} {got:,.0f} of {target:,.0f} "
                                 f"({got / target:.0%})")
            if short:
                self.notes.append(
                    f"[offered rate not served: {'; '.join(short)}. The figure is "
                    f"what was delivered, not what was asked for]")
            if sat:
                self.notes.append(f"[saturated: {', '.join(sat)} p50 > 1 s]")
                return "saturated"
            return "short of offer" if short else "offered"
        if self.qa is None or self.qb is None or self.qb <= 0:
            return "-"
        # A row whose author says it gets no ratio (W11: the collection is
        # being written while it is searched) says why, in place of a note
        # that reads as a sweep someone forgot to run.
        policy = ra.get("ratio_policy") or rb.get("ratio_policy") or ratio_policy_of().get(wid)
        # W11: below the overlap floor the writer was gone for most of the row.
        # That is not the same as the collection being quiet -- W11's append
        # trips a rebuild that outlives it, and the search spends the remainder
        # competing with that -- so the note says what is actually known.
        # Which engines, not how much: the per-engine bracket already carries
        # each one's `write overlap N%`. Naming them matters — one arm can be
        # below the floor while the other is not, and a note that says "each
        # row" then describes a row it does not apply to.
        low = [lbl for lbl, r in ((self.a_label, ra), (self.b_label, rb))
               if r.get("write_overlap_pct") is not None
               and r["write_overlap_pct"] < 100 * W11_MIN_OVERLAP]
        floor = f"{100 * W11_MIN_OVERLAP:.0f}%"
        if len(low) == 2:
            low_note = (f"the writer covered under {floor} of either search, so most of "
                        f"both rows measured the rebuild the append provoked rather "
                        f"than a concurrent write")
        elif low:
            low_note = (f"the writer covered under {floor} of {low[0]}'s search, so most "
                        f"of that row measured the rebuild the append provoked rather "
                        f"than a concurrent write")
        else:
            low_note = ""
        if policy:
            self.notes.append(f"[{policy}{'; ' + low_note if low_note else ''}]")
            return self._refuse(policy)
        if not self._rows_stamped():
            return self._refuse("row not measured under the harness its run.json describes")
        if low_note:
            self.notes.append(f"[{low_note}]")
            return self._refuse(low_note)
        if self.recall_required:
            if self.rec_a is None or self.rec_b is None:
                which = [l for l, v in ((self.a_label, self.rec_a), (self.b_label, self.rec_b))
                         if v is None]
                self.notes.append(f"[recall missing: {', '.join(which)}]")
                return "-"
            # The floor is asked first, and it used to be asked second.
            #
            # W7 on sift1m returns 0.0485 against 0.0267: both useless, and
            # further apart than `RECALL_TOL`, so the tolerance arm answered
            # first and the row read "recall unequal — §7.4 compares at equal
            # recall". That describes a comparison that could be repaired by
            # matching the parameters. Nothing can be repaired here: one bit
            # per dimension is 128 bits per SIFT vector and the Hamming
            # ordering it induces carries almost no signal at this dimension
            # (findings 24), so the row characterises an encoding and never
            # yields a ratio at any tolerance. Saying which of the two it is
            # matters, because the first sends a reader looking for a bug.
            if max(self.rec_a, self.rec_b) < RECALL_FLOOR:
                self.notes.append(
                    f"[recall below the usable floor: {self.rec_a:.4f} and "
                    f"{self.rec_b:.4f}, both under {RECALL_FLOOR:.2f} — this row "
                    f"characterises the encoding, not the engines]")
                return self._refuse(f"recall below {RECALL_FLOOR:.2f} on both engines")
            if abs(self.rec_a - self.rec_b) > RECALL_TOL:
                self.notes.append(f"[recall unequal: {self.rec_a:.4f} vs {self.rec_b:.4f}; "
                                  f"§7.4 compares at equal recall{self._rescore_cause()}]")
                return "-"
        # A band is only a noise band if the spread it came from was noise.
        # `aggregate._rep_drift` marks an arm whose passes moved one way only:
        # strawmANN's W10-ef128 on dbpedia-openai-1m slid 3,662 -> 3,001 ->
        # 2,764 qps within one run, which folded to `rsd 14.8%` and a +/-63%
        # band, and the ratio then published as "parity: no measured
        # difference". The difference was real; the arm was drifting. Saying so
        # beats calling it either parity or a clean ratio.
        drifted = [f"{lbl} {r['rep_drift']:+.0%}"
                   for lbl, r in ((self.a_label, ra), (self.b_label, rb))
                   if r.get("rep_drift") is not None]
        # This string is also the matched-recall frontier's explanation when a
        # W10-ef row carries it (`report_data.matched_recall_refusal`), so it
        # names the arm and the size of the move rather than saying only that
        # something drifted. A frontier with a drifted point on one axis and a
        # stable one on the other is a smear, not a frontier, which is why the
        # refusal reaches that chart at all.
        band = parity_band(self.id, self.a_label, self.b_label)
        if drifted:
            self.notes.append(
                f"[{', '.join(drifted)} across its passes, monotonically: the "
                f"spread on this row is drift rather than noise, so it is not a "
                f"noise band and the ratio is not a repeatable measurement]")
            return self._refuse(f"drifted across passes ({', '.join(drifted)})")
        if band is not None and abs(self.qa / self.qb - 1.0) <= band:
            self.notes.append(
                f"[within the ±{100 * band:.1f}% band this dataset's noise floor "
                f"puts on {self.id}: no measured difference, not a small one]")
            return "parity"
        return f"{self.qa / self.qb:.2f}x"

    @staticmethod
    def _row_flag(r: dict, key: str, table_default: bool) -> bool:
        """A boolean the row records, falling back to §4's table when the row
        predates the field. `or`-ing the two, as `ratio_policy` does, would let
        the table override a row that recorded `False` on purpose — which is
        exactly what a row measured *after* filtering is implemented will say.
        """
        v = r.get(key)
        return table_default if v is None else bool(v)

    def _unindexed_filter(self) -> str:
        """Why this filtered-search row is not a filtered-search measurement.

        An engine that quietly abandons its index still answers, still returns the
        right points, and still reports a number: the failure has no symptom but the
        number being wrong about what it measured. Percona's `vector-bench` validates
        the query plan every run for this reason; the equivalent here is two facts, in
        this order:

          1. did the *run* deny the engine an index (`--skip-field-indices`), which no
             read-back can undo, and
          2. does the engine say it built one, read back into `collections.json`.

        Intent settles the first and cannot settle the second: asking for no index
        guarantees there is none, asking for one guarantees nothing.

        Paid for once already. W12 ran 44 minutes without finishing
        because the flag denied Qdrant its index and all 200,000 filtered queries
        became scans; the same row completed in 288 s and its 17 q/s
        went into the published table beside the words "filtered search". The hang was
        caught; the completed run was not.

        Unknown is refused as well as empty: an index state nobody read back is not
        evidence one existed.
        """
        needs = needs_payload_index_of().get(self.id, False)
        if not (self._row_flag(self.ra, "needs_payload_index", needs)
                or self._row_flag(self.rb, "needs_payload_index", needs)):
            return ""
        suppressed = payload_index_suppressed_of().get(self.id, False)
        why = ""
        for lbl, r in ((self.a_label, self.ra), (self.b_label, self.rb)):
            # A row the engine declined has no figure to disqualify: strawmANN
            # answers UNIMPLEMENTED for `CreateFieldIndex` (§2 phase 3, M7), and
            # saying "no payload index" about it would report the boundary
            # twice, the second time as a defect.
            if r.get("status") != Status.ok:
                continue
            coll = r.get("collection") or collection_of().get(self.id) or "?"
            if self._row_flag(r, "payload_index_suppressed", suppressed):
                self.notes.append(
                    f"[{lbl}: the run denied this engine a payload index "
                    f"(--skip-field-indices), so every filtered query scanned {coll} end "
                    f"to end; this row measures a full scan and must not be read as a "
                    f"filtered-search result]")
                why = "payload index suppressed: the row measured a full scan"
                continue
            idx = payload_indexes(lbl, coll)
            if idx is None:
                self.notes.append(
                    f"[{lbl}: whether {coll} carried a payload index was never read back, "
                    f"so this row cannot be shown to have searched an index rather than "
                    f"scanning {coll} once per query]")
                why = why or "payload index state not read back"
            elif not idx:
                self.notes.append(
                    f"[{lbl}: no payload index on {coll}, so every filtered query scanned "
                    f"the whole collection; this row measures a full scan and must not be "
                    f"read as a filtered-search result]")
                why = "no payload index: the row measured a full scan"
        return why

    def _rescore_cause(self) -> str:
        """Marks a rescoring row, whose unequal recall has one known cause.

        W6, W7 and W8 are all refused for unequal recall and all three have the
        same cause: strawmann rescores `max(asked, ef)` candidates and Qdrant
        rescores `limit` of them, so at equal nominal parameters they are not
        answering the same question (decisions §5). Three rows carrying a
        sentence each read as three problems, and the first response was to go
        looking for three bugs — so the row is marked and the sentence is said
        once, by the reader's own document: `RESCORE_CAUSE`.
        """
        if not (self.ra.get("quantization_rescore")
                or self.rb.get("quantization_rescore")):
            return ""
        return "; rescore pools differ (decisions §5)"

    def _rows_stamped(self) -> bool:
        """Both rows carry the harness stamp their label's `run.json` carries,
        and the same one as each other. The `-n` they ran with is not part of
        it: qps is a rate, and `--min-duration` scales `-n` per engine.

        `rows.json` merges in place, so a subset re-run rewrites `run.json`
        with today's stamp while the rows it did not re-run keep yesterday's
        measurement. The label-level check cannot see that; this can.
        """
        ok = True
        hashes = []
        for lbl, r, stamp in ((self.a_label, self.ra, self.stamps[0]),
                              (self.b_label, self.rb, self.stamps[1])):
            h = r.get("harness_hash")
            if not h:
                self.notes.append(f"[{lbl}: row has no harness stamp (measured before "
                                  f"rows were stamped)]")
                ok = False
                continue
            if stamp is not None and h != stamp_hash(stamp):
                self.notes.append(f"[{lbl}: re-run under a different harness: this row "
                                  f"was not measured by the run its run.json describes]")
                ok = False
            hashes.append(h)
        if ok and len(hashes) == 2 and hashes[0] != hashes[1]:
            self.notes.append("[re-run under a different harness: harness stamps differ]")
            ok = False
        return ok

    @property
    def note_text(self) -> str:
        return " ".join(self.notes)


def recall_required_of() -> dict[str, bool]:
    """Which rows §7.4's rule applies to, from §4's table.

    A ratio needs equal recall wherever recall is defined: an approximate
    search of the dataset's queries. Exempt, and saying why: W0 (d=4 random
    plumbing floor, no relevance), exact rows (recall 1.0 by construction on
    both sides), scroll (no ranking). A row not in the table is not exempt.
    """
    try:
        from workloads import table
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for w in table():
        if w.upload_only:
            continue
        exempt = (w.id == "W0" or "--search-exact" in w.args or "--scroll" in w.args
                  or "--search" not in w.args)
        out[w.id] = not exempt
    return out


def recall_key_of(row: dict, wid: str) -> tuple[str | None, int | None, bool]:
    """(collection, ef, joinable) for a row: from the row itself when it says,
    from §4's table otherwise. `ef` is the row's own and never the table's:
    the table's `ef` today is not what a row measured before it was stated."""
    coll = row.get("collection") or collection_of().get(wid)
    ef = row.get("ef")
    joinable = row.get("recall_joinable", True) and recall_joinable_of().get(wid, True)
    return coll, ef, joinable


def quant_key_of(row: dict) -> recall_mod.QuantParams:
    """The quantization search parameters the row sent; the sweep must match.

    From the row only, never from today's table: a row that predates the
    field sent whatever it sent, and stamping W7's oversampling onto it would
    join a sweep to a measurement it does not describe.

    `recall.QuantParams`, not a second unnamed pair of the same two values:
    §6.7's join key is one thing, and it was spelled `(float | None, bool |
    None)` in this module and named in that one."""
    return recall_mod.QuantParams(row.get("quantization_oversampling"),
                                  row.get("quantization_rescore"))


def joined(a_label: str, b_label: str,
           a: dict[str, dict], b: dict[str, dict],
           stale: bool | None = None) -> list[Row]:
    """The join, once, for every rendering of it.

    Three renderings of this table existed: the text one below, the compact
    block in README.md and a markdown one per dataset. Two were
    hand-maintained, and they drifted exactly as you would expect: the README
    header said 4,095 for W3 while the full tables in README.md and
    the markdown table both said 3,985, a figure from an earlier run. Nothing was
    wrong with either number; they described different runs, and no reader
    could tell.

    So the arithmetic and the row filtering live here and the renderers only
    format. Returns `Row`s in measurement order: W11 runs last because it
    leaves a pending tail that would otherwise be measured as part of W13.
    """
    if stale is None:
        stale = bool(stale_reasons(a_label, b_label))
    required = recall_required_of()
    dataset = dataset_of(a_label) or dataset_of(b_label) or "sift1m"
    stamps = (read_json(a_label, "run.json").get("harness") or None,
              read_json(b_label, "run.json").get("harness") or None)
    out = []
    for wid in list(a) + [k for k in b if k not in a]:
        ra, rb = a.get(wid), b.get(wid)
        if not ra or not rb:
            continue
        # Upload rows have no qps; their timing is the W1/W2 measurement and is
        # reported by the runner, not here.
        if ra.get("qps") is None and rb.get("qps") is None and ra["status"] == Status.ok:
            continue
        rec_a = recall_at(a_label, dataset, ra, wid)
        rec_b = recall_at(b_label, dataset, rb, wid)
        out.append(Row(wid, ra, rb, a_label, b_label, rec_a, rec_b,
                       required.get(wid, True), stale, stamps))
    return out


def dataset_of(label: str) -> str | None:
    return ((read_json(label, "run.json").get("dataset") or {}).get("name"))


def dataset_spec_of(a_label: str, b_label: str) -> dict:
    """The measured dataset's name, dimension and metric, from either arm."""
    for label in (a_label, b_label):
        spec = read_json(label, "run.json").get("dataset") or {}
        if spec.get("name"):
            return spec
    return {}


def parity_band(wid: str, a_label: str, b_label: str) -> float | None:
    """Half-width of the band in which a ratio is no measured difference.

    `None` when this dataset has no measured floor for this row, which is the
    common case and is why the ratio still prints: a missing floor is not a
    verdict. `report.py` says so per row; here the table simply keeps the
    number.

    The floor is refused when it names another dataset. dbpedia-openai-100K's
    document printed `1.00x` on W10-ef256 and `1.01x` on W9 against a spread
    measured on sift1m at a twelfth the dimension, and `1.00x` reads as "these
    engines are equally fast" rather than "this run cannot tell them apart".
    """
    meta = read_noise_meta(a_label, b_label)
    rsd = (meta.get("rsd") or {}).get(wid)
    if rsd is None:
        return None
    floor_ds = meta.get("dataset")
    if floor_ds:
        row_ds = dataset_spec_of(a_label, b_label).get("name")
        if row_ds and row_ds != floor_ds:
            return None
    if not env_matches(meta, a_label, b_label):
        return None
    if floor_covers(meta, a_label, b_label) is not True:
        return None
    return regression.noise_band(rsd, arms=2)


def env_hash_of(label: str) -> str | None:
    """The §7.1 environment hash a label was measured under, from `run.json`."""
    return read_json(label, "run.json").get("env_hash")


def env_matches(meta: dict, a_label: str, b_label: str) -> bool:
    """Was this floor measured in the environment it would be banding?

    The same rule as the dataset one above, for the same reason and against a
    mistake that has already been made. `bench/setup.py`'s hash covers SMT,
    the governor, isolation and the rest of §7.1's body, and those move
    throughput: Qdrant's saturating row read 10,313 against a
    twelve-run aggregate of 14,716 and was taken for a 30% regression, when
    the aggregate had been measured with SMT off and the run with it on.
    strawmANN's own figure moved 1.5% across the same change, so the two
    engines are not even affected alike and no correction factor exists.

    Permissive where it cannot know, like the dataset rule: a floor with no
    `env_hash` predates this stamp and stays usable, and a run whose own hash
    is missing cannot contradict anything. Only a *disagreement* refuses, which
    is the case that produced a wrong number.
    """
    floor_env = meta.get("env_hash")
    if not floor_env:
        return True
    return not any(h and h != floor_env
                   for h in (env_hash_of(a_label), env_hash_of(b_label)))


def engine_of(label: str) -> str | None:
    """The engine a label measured, from its `run.json`."""
    return read_json(label, "run.json").get("engine_comm")


def floor_covers(meta: dict, a_label: str, b_label: str) -> bool | None:
    """Was this floor measured on both engines whose ratio it would band?

    `True` yes, `False` no, `None` the floor does not say.

    Unlike the dataset rule above, an unattributed floor is refused rather than
    used-and-labelled. The cases are not alike. A floor with no `dataset` might
    have been measured on this corpus; a floor with no `arms` cannot have been
    measured on both engines, because `measure_noise` reads the repetition
    directories of *one* label. "Which engine" is unknown; "one engine" is not.

    Findings 38: `bench/results/noise.json` here is six passes of strawmANN
    giving W3 an RSD of 0.46%, and it banded the strawmANN-vs-
    Qdrant ratio on a row where Qdrant's own run-to-run spread is 13.16% over
    twelve runs — 934 to 1,806 qps. The single-query row read 1.11x and
    then 1.26x with nothing having changed but the draw.
    """
    arms = meta.get("arms")
    if not arms:
        return None
    want = {engine_of(a_label) or a_label, engine_of(b_label) or b_label}
    return want.issubset(set(arms))


def read_noise_meta(*labels: str) -> dict:
    """The floor that describes these labels, whole, or `{}`.

    `regression.floor_for` prefers a `--reps N` run's own folded spread over
    the global `bench/results/noise.json`, which is one corpus's and one
    engine's. With no labels it is the global file.
    """
    return regression.floor_for([l for l in labels if l])


def scope_line(a_label: str, b_label: str) -> str:
    """What the headline ratios are ratios *of*.

    The README's four rows carried no dimension. Measured on sift1m
    (d=128) put strawmann 1.46x-4.78x ahead on every search row, while
    dbpedia-openai-100K (d=1536) put Qdrant ahead on W4 and flattened the top of
    the recall sweep to 1.00x: at d=1536 a distance is 1,536 multiply-adds, the
    query goes memory-bandwidth bound, and §6's per-request savings stop
    mattering. A front page that states the first set of numbers with no
    dimension on them reads as a general claim and is not one.

    Generated rather than captioned by hand for the reason the table itself is:
    the hand-written caption under this block already named SIFT1M and still
    left the dimension out.
    """
    spec = dataset_spec_of(a_label, b_label)
    name = spec.get("name") or "unknown"
    dim, metric = spec.get("dim"), (spec.get("metric") or "").lower()
    what = f"  {name}"
    if dim:
        what += f", d={dim}"
    if metric:
        what += f", {metric}"
    others = sorted(q.name[len("comparison-"):-len(".md")]
                    for q in (ROOT / "docs").glob("comparison-*.md")
                    if q.name not in (f"comparison-{name}.md", "comparison.md"))
    if others:
        what += (f" — these ratios are this dimension's. "
                 f"Also measured: {', '.join(others)}")
    else:
        what += " — these ratios are this dimension's"
    return what


_RECALL_CACHE: dict[tuple, dict[int, dict]] = {}


def recall_sweep(label: str, dataset: str, collection: str,
                 oversampling: float | None = None,
                 rescore: bool | None = None,
                 grade: str | None = None) -> dict[int, dict]:
    key = (label, dataset, collection, oversampling, rescore, grade)
    if key not in _RECALL_CACHE:
        _RECALL_CACHE[key] = recall_mod.load_recall(label, dataset, collection,
                                                    oversampling, rescore, grade)
    return _RECALL_CACHE[key]


def recall_at(label: str, dataset: str, row: dict, wid: str) -> float | None:
    coll, ef, joinable = recall_key_of(row, wid)
    if not coll or ef is None or not joinable:
        return None
    # The grade comes off the row id: a filtered row is joined only to the
    # sweep of its own condition (W12 point 3), and an unfiltered row only to
    # an unfiltered sweep.
    pt = recall_sweep(label, dataset, coll, *quant_key_of(row),
                      grade=recall_mod.grade_of(wid)).get(int(ef))
    return None if pt is None else pt.get("recall_at_10")


def latency_cell(row: dict, which: str = "client") -> str:
    """p50 / p99 for a row, in the unit that fits the number.

    Both percentiles, never one: p50 alone hides the tail a saturating row
    exists to expose, and p99 alone hides that the median was fine.
    """
    p50, p99 = row.get(f"{which}_p50_us"), row.get(f"{which}_p99_us")
    if p50 is None and p99 is None:
        return "-"

    def fmt(v):
        if v is None:
            return "?"
        return f"{v / 1000:.2f} ms" if v >= 1000 else f"{v:.0f} µs"

    return f"{fmt(p50)} / {fmt(p99)}"


def recall_cell(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def detail_rows(a_label: str, b_label: str, a: dict[str, dict], b: dict[str, dict],
                stale: bool | None = None):
    """The comparison with its latency and recall, one row at a time.

    §8.9 fixes the row shape as QPS *and* p50/p99 *and* recall. The table used
    to print the first of those three, which is the shape of number this
    project exists to argue against.
    """
    for r in joined(a_label, b_label, a, b, stale):
        la = a[r.id].get("latency", {}) or {}
        lb = b[r.id].get("latency", {}) or {}
        yield {
            "id": r.id,
            "a_qps": r.sa, "b_qps": r.sb, "ratio": r.ratio, "notes": r.note_text,
            "a_lat": latency_cell(la), "b_lat": latency_cell(lb),
            "a_recall": recall_cell(r.rec_a), "b_recall": recall_cell(r.rec_b),
            "ef": a[r.id].get("ef"),
            "load_mode": a[r.id].get("load_mode") or b[r.id].get("load_mode") or "",
        }


#: Rows §4's table no longer carries, and what they were. A result set is an
#: archive: `W4-rps500` was a real row measured at a constant 500/s before the
#: arms became §4's fractions of measured saturation, and a table rendered over
#: that archive must still name it. Without this the regenerated block printed
#: those three rows with an empty description, which reads as a row nobody
#: bothered to describe rather than a row the table has moved on from.
RETIRED_PURPOSES = {
    "W4-rps500": "search, fixed 500/s (open loop, pre-§4 constant rate)",
    "W4-rps1000": "search, fixed 1000/s (open loop, pre-§4 constant rate)",
    "W4-rps2000": "search, fixed 2000/s (open loop, pre-§4 constant rate)",
}


def purpose_of() -> dict[str, str]:
    """§4's own name for each row, read from the one definition of the table,
    plus the retired ids an archived result set may still contain."""
    try:
        from workloads import table
    except ImportError:  # pragma: no cover - only if run from outside bench/harness
        return dict(RETIRED_PURPOSES)
    return {**RETIRED_PURPOSES, **{w.id: w.purpose for w in table()}}


def collection_of() -> dict[str, str]:
    """Which collection each row searches, from §4's one definition."""
    try:
        from workloads import collection_of as _coll
        from workloads import table
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for w in table():
        c = _coll(w)
        if c:
            out[w.id] = c
    return out


def recall_joinable_of() -> dict[str, bool]:
    try:
        from workloads import table
    except ImportError:  # pragma: no cover
        return {}
    return {w.id: w.recall_joinable for w in table()}


def payload_indexes(label: str, collection: str) -> list[str] | None:
    """Payload fields the engine said it had indexed on `collection`, as read
    back by `collection-info`, or `None` when this run has no answer for it.

    `None` covers three cases that a caller must not tell apart by guessing:
    the run predates the field, the read-back failed (`fullrun.capture_collections`
    treats that as provenance and does not fail the arm), or the collection was
    never created. All three mean the same thing to a filtered-search row —
    nobody checked — and `Row._unindexed_filter` refuses on all three.
    """
    doc = read_json(label, "collections.json")
    for c in doc.get("collections") or []:
        if isinstance(c, dict) and c.get("collection") == collection:
            idx = c.get("payload_indexes")
            return list(idx) if isinstance(idx, list) else None
    return None


def needs_payload_index_of() -> dict[str, bool]:
    """`Workload.needs_payload_index` per row, for rows measured before it was
    recorded — the same fallback `ratio_policy_of` exists for, and for the same
    reason: the check must cover the results already on disk."""
    try:
        from workloads import table
    except ImportError:  # pragma: no cover
        return {}
    return {w.id: w.needs_payload_index for w in table() if w.needs_payload_index}


def payload_index_suppressed_of() -> dict[str, bool]:
    """`payload_index_suppressed` per row, the same fallback for rows on disk."""
    try:
        from workloads import payload_index_suppressed, table
    except ImportError:  # pragma: no cover
        return {}
    return {w.id: True for w in table() if payload_index_suppressed(w)}


def ratio_policy_of() -> dict[str, str]:
    """`Workload.ratio_policy` per row, for rows measured before it was recorded."""
    try:
        from workloads import table
    except ImportError:  # pragma: no cover
        return {}
    return {w.id: w.ratio_policy for w in table() if w.ratio_policy}


#: How each engine is written in prose. The tables used the raw directory
#: label, so a document that says strawmANN throughout had a column headed
#: `strawmann`.
DISPLAY = {"strawmann": "strawmANN", "qdrant": "Qdrant"}


def display_name(label: str) -> str:
    return DISPLAY.get(label.lower(), label)


def environment_note(a_label: str, b_label: str,
                     a: dict[str, dict], b: dict[str, dict]) -> str:
    """The environment line as prose rather than as padded columns.

    It was built with fixed-width padding and then rendered inside `<sub>`,
    where the padding collapses: monospace formatting in a proportional
    context. Markdown gets a sentence; the text table keeps the columns.
    """
    parts = []
    for lbl, rows in ((a_label, a), (b_label, b)):
        peak = max((r["load_start"] for r in rows.values()), default=0)
        dirty = [r["id"] for r in rows.values() if r.get("foreign")]
        when = sorted(r.get("when", "") for r in rows.values() if r.get("when"))
        span = ""
        if when:
            span = (f", measured {when[0][:10]}" if when[0][:10] == when[-1][:10]
                    else f", measured {when[0][:10]} to {when[-1][:10]}")
        # Not "ambient": `load_start` is `/proc/loadavg` over `os.cpu_count()`,
        # the whole machine's one-minute load average, which counts the engine
        # under test and the client driving it. Reading it as foreign load is a
        # mistake this line invited and one that got made: Qdrant's arm shows
        # 126-133% against strawmANN's 62-66% on the same host, because Qdrant
        # runs 78 threads to strawmANN's 9 — its own threads, not someone
        # else's. Foreign load is measured separately and per row, by the
        # `/proc` sampler that excludes `_OURS`, and lands in `foreign`.
        s = (f"**{display_name(lbl)}**: §7.1 gate {gate_of(lbl)}, peak system "
             f"load {peak}% of one core (all processes, this engine included)"
             f"{span}")
        if dirty:
            s += f", contaminated rows {', '.join(dirty)}"
        parts.append(s + ".")
    return " ".join(parts)


def environment_line(label: str, rows: dict[str, dict]) -> str:
    """The environment a set of rows was measured in, printed with them.

    Takes the rows it was given rather than re-reading them, so that generating
    a block on a fresh clone (no `bench/results/`) is a no-op rather than an
    error: the gate step has to survive a machine that has never measured
    anything.
    """
    peak = max((r["load_start"] for r in rows.values()), default=0)
    dirty = [r["id"] for r in rows.values() if r.get("foreign")]
    return (f"{label:<10} peak system load {peak}% of one core (engine included); "
            f"§7.1 gate: {gate_of(label)}"
            + (f"; contaminated rows: {', '.join(dirty)}" if dirty else ""))


def storage_and_io(a_label: str, b_label: str,
                   a: dict[str, dict], b: dict[str, dict],
                   markdown: bool = False) -> list[str]:
    """What each engine stored and how much I/O it did, as lines.

    Separate from the throughput table rather than four more columns on it,
    because it answers a different question and one of the two engines is
    expected to score zero on half of it: strawmann holds everything in RAM and
    the server never calls `persist`, so a table showing only queries per
    second charges Qdrant for durability strawmann does not provide and reports
    the difference as speed.

    Counts are not ratioed across engines when they come from different
    interfaces. `/proc/<pid>/io` counts syscalls and cgroup `io.stat` counts
    block-layer requests, and a `pwrite` of 4 MiB is one of the first and many
    of the second.
    """
    def totals(rows: dict[str, dict]) -> dict:
        out: dict = dict.fromkeys(procstat.IO_FIELDS)
        out.update(source="", storage=None, rss=None, storage_excludes=[])
        # Totals and the peak see every row; the storage *level* sees only the
        # settled ones, because the last row appends 200,000 points and catches
        # a segment-rewriting engine mid-rewrite (findings 53).
        settled = {id(r) for r in procstat.settled_rows(list(rows.values()))}
        out["storage_excludes"] = [r["id"] for r in rows.values()
                                   if procstat.is_mutating(r)]
        for r in rows.values():
            if r.get("io_source"):
                out["source"] = r["io_source"]
            for k in procstat.IO_FIELDS:
                v = r.get(k)
                if v is not None:
                    # Summed over rows: a field no row could measure stays
                    # None, so "nothing happened" and "we could not look" are
                    # different cells rather than the same zero.
                    out[k] = (out[k] or 0) + v
            # Levels, not sums: the last row that saw one is the end state.
            if r.get("storage_bytes") is not None and id(r) in settled:
                out["storage"] = r["storage_bytes"]
            if r.get("rss_peak_bytes") is not None:
                out["rss"] = max(out["rss"] or 0, r["rss_peak_bytes"])
        return out

    ta, tb = totals(a), totals(b)
    if not ta["source"] and not tb["source"] and ta["storage"] is None and tb["storage"] is None:
        return []

    #: `(label, key, formatter)`, in two groups: the block layer, then the
    #: syscall counts that are not a disk figure and must not read as one.
    disk = [("storage on disk", "storage", procstat.human_bytes),
            ("peak RSS", "rss", procstat.human_bytes),
            ("disk read ops", "disk_read_ops", procstat.human_count),
            ("disk write ops", "disk_write_ops", procstat.human_count),
            ("disk read bytes", "disk_read_bytes", procstat.human_bytes),
            ("disk write bytes", "disk_write_bytes", procstat.human_bytes)]
    syscalls = [("syscall reads (all fds)", "syscall_reads", procstat.human_count),
                ("syscall writes (all fds)", "syscall_writes", procstat.human_count)]

    if markdown:
        out = [f"| | {display_name(a_label)} | {display_name(b_label)} |",
               "|---|--:|--:|"]
        for name, key, fmt in disk:
            out.append(f"| {name} | {fmt(ta[key])} | {fmt(tb[key])} |")
        for name, key, fmt in syscalls:
            out.append(f"| *{name}* | *{fmt(ta[key])}* | *{fmt(tb[key])}* |")
        out.append(f"| measured via | {ta['source'] or 'n/a'} | {tb['source'] or 'n/a'} |")
        out += ["", note_text(ta, tb)]
        return out

    width = 24
    lines = [f"{'':<{width}} {a_label:>16} {b_label:>16}"]
    for name, key, fmt in disk:
        lines.append(f"{name:<{width}} {fmt(ta[key]):>16} {fmt(tb[key]):>16}")
    lines.append("")
    for name, key, fmt in syscalls:
        lines.append(f"{name:<{width}} {fmt(ta[key]):>16} {fmt(tb[key]):>16}")
    lines.append(f"{'measured via':<{width}} {(ta['source'] or 'n/a'):>16} "
                 f"{(tb['source'] or 'n/a'):>16}")
    lines += [""] + wrap(note_text(ta, tb))
    return lines


#: Clauses a stored row note no longer needs to carry, because the row-level
#: policy note says each of them once for both engines. The note lives in
#: `rows.json`, so rows measured before `workloads.py` stopped writing them
#: still have them and the front page is generated from those rows.
_REDUNDANT_CLAUSES = (
    # The complement of `write overlap N%`, plus the rebuild explanation the
    # policy note now carries.
    re.compile(r";?\s*append finished [\d.]+ s before the search did"
               r"(\s*\([^)]*\))?"),
    re.compile(r";?\s*search finished [\d.]+ s before the append did"
               r"(\s*\([^)]*\))?"),
    # "no recall join" says this in the policy note.
    re.compile(r";?\s*recall not measured"),
    # The policy note names the workload; the per-engine bracket carries numbers.
    re.compile(r"concurrent append of synthetic vectors;?\s*"),
)


def trim_row_note(note: str) -> str:
    """A stored row note with the clauses the policy note repeats removed."""
    for pat in _REDUNDANT_CLAUSES:
        note = pat.sub("", note)
    return note.strip().strip(";").strip()


def note_text(ta: dict, tb: dict) -> str:
    # Run order, not sorted: "W11, W11-steady" is alphabetical and reverses the
    # order they ran in, which is what the sentence is describing.
    skipped: list[str] = []
    for x in (ta.get("storage_excludes") or []) + (tb.get("storage_excludes") or []):
        if x not in skipped:
            skipped.append(x)
    s = ("Storage and RSS are end states; the rest are totals over every row. "
         "The syscall rows count every descriptor, sockets included, so on a "
         "search row they measure the network rather than the disk, and no "
         "ratio between them and the disk rows means anything.")
    if skipped:
        s += (f" Storage on disk is the level before {', '.join(skipped)}, the rows "
              f"with a concurrent writer: an engine that rewrites segments is caught "
              f"mid-rewrite there, and the same row has read 3.47 and 10.11 GiB on two "
              f"runs of one binary. Peak RSS does include those rows, being a peak.")
    if ta["source"] and tb["source"] and ta["source"] != tb["source"]:
        s += (f" The two engines were measured through different interfaces "
              f"({ta['source']} and {tb['source']}): `proc` supplies syscalls and "
              f"block-layer bytes, `cgroup` supplies block-layer operations and "
              f"bytes, so a row unavailable on one side reads `unknown` rather "
              f"than 0.")
    return s


def wrap(text: str, width: int = 76) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


#: The rows the README header shows, and the name it shows them under. Four is
#: as many as fits above the fold; the full table is one link away.
HEADER_ROWS = [
    # Not "p=1": `p` is bfb's `--parallel`, which is the load generator's
    # vocabulary and not the reader's. The row below it is the same search with
    # the client pushing as hard as it can, and the pair only reads as a
    # contrast if both ends are named in words.
    ("W3", "search, one at a time"),
    ("W4", "saturating"),
    ("W11", "mixed read/write"),
    ("W9", "exact"),
]

#: Each generated block, and the file it belongs to. One command writes them
#: all and one gate step checks them all, so a run that updates one cannot
#: leave the other behind.
#:
#: `compare-full` is per dataset. It used to be one `docs/comparison.md`, which
#: meant a run of the second dataset overwrote the first dataset's table with a
#: table of different numbers under the same heading — the same overwrite the
#: README headline has, except the README is one deliberate tier's front page
#: and this is the full record. Two datasets are not ratioable (`dataset` is in
#: the harness stamp), so they do not belong in one document either. There was
#: also an index listing those documents; a generated list of two filenames in
#: a directory that already lists them is not worth a third target.
def blocks_targets(dataset: str) -> list[tuple[str, str]]:
    out = []
    if dataset == HEADLINE_DATASET:
        out.append(("README.md", "compare-table"))
    out.append((f"docs/comparison-{dataset}.md", "compare-full"))
    return out


#: The dataset whose four rows are the README's headline table. Every other
#: dataset writes its own document and leaves the front page alone.
#:
#: Without this, the last run to render owned the front page. Running sift1m and
#: then dbpedia-openai-100K left the README quoting d=1536 numbers
#: under a caption naming SIFT1M, and the restore was a hand-run `compare.py`
#: afterwards — a step nobody would know to take.
HEADLINE_DATASET = "sift1m"


def new_comparison_doc(dataset: str) -> str:
    """A per-dataset document for a dataset measured here for the first time.

    Created rather than demanded: the alternative is a run that measures a new
    dataset for forty minutes and then refuses to write its table because a file
    nobody could have known to add is missing.
    """
    begin, end = markers("compare-full")
    return (f"# strawmann vs Qdrant 1.19.0 — {dataset}\n\n"
            f"The table below is generated from the last full run that measured "
            f"this dataset. Anything written under it by hand is not, and is not "
            f"re-derived when the table is.\n\n"
            f"{begin}\n{end}\n")




class Markers(NamedTuple):
    """The comment pair a generated block sits between.

    Named because splicing reads `begin` and `end` positionally at four sites
    and a transposition is silent: `split(end)[0]` against a file whose blocks
    are intact still returns *something*, and what it returns is the file up to
    the wrong marker."""

    begin: str
    end: str


def markers(kind: str) -> Markers:
    return Markers(
        begin=f"<!-- BEGIN {kind} (generated by bench/harness/compare.py --write-readme) -->",
        end=f"<!-- END {kind} -->")


def banners(a_label: str, b_label: str) -> list[str]:
    """What has to be said above the numbers, in every rendering.

    Ordered by severity: STALE first (no ratio in the table means anything),
    then the licence (a ratio is not a comparative claim), then contamination.
    """
    out = []
    stale = stale_reasons(a_label, b_label)
    if stale:
        out.append("STALE: the two result sets were not produced by the same harness "
                   "configuration, so no ratio is printed. " + "; ".join(stale) + ".")
    lic = licence(a_label, b_label)
    if lic["banner"]:
        out.append(lic["banner"])
    out += unbanded_banner(a_label, b_label)
    return out


def unbanded_banner(a_label: str, b_label: str) -> list[str]:
    """Say when no measured spread applies to this dataset's ratios.

    `parity_band` refuses a floor measured on another corpus, which is right —
    run-to-run spread does not carry across a twelvefold change in dimension.
    But refusing it silently leaves `1.00x` and `0.99x` printed bare, and a
    bare ratio reads as a result. On dbpedia-openai-100K three rows sit within
    1% of parity (W9 1.01x, W10-ef256 1.00x, W10-ef512 0.99x) and nothing has
    judged any of them.

    One line for the table, not a verdict per row: the absence is a property of
    the dataset, and `report.py` already says it per row where a reader is
    looking at one row.
    """
    meta = read_noise_meta(a_label, b_label)
    if not meta.get("rsd"):
        return []
    out = []
    ds = dataset_spec_of(a_label, b_label).get("name")
    floor_ds = meta.get("dataset")
    if ds and floor_ds and ds != floor_ds:
        out.append(f"NO NOISE FLOOR for {ds}: the measured spread on disk is "
                   f"{floor_ds}'s, and run-to-run spread does not carry between corpora, "
                   f"so no ratio below is banded. A ratio near 1.0 here has not been "
                   f"shown to be a difference, nor shown not to be.")
    # The same sentence about the other axis. Said separately because the fixes
    # are different: another corpus needs a floor for this corpus, another
    # engine needs `--reps N` (which folds each arm's own spread) or a floor
    # measured on the engine that is missing.
    covers = floor_covers(meta, a_label, b_label)
    if covers is not True:
        arms = meta.get("arms")
        whose = (f"the measured spread on disk is {', '.join(arms)}'s alone" if arms
                 else "the measured spread on disk does not record which engine it "
                      "was measured on, and a floor is always one engine's — "
                      "`measure_noise` reads one label's repetitions")
        out.append(f"NO NOISE FLOOR for this pair: {whose}. Run-to-run spread does "
                   f"not carry between engines, so no ratio below is banded. "
                   f"Findings 38: on W3 this floor reads 0.46% while Qdrant's own "
                   f"spread over twelve runs is 13.16%, 934 to 1,806 qps.")
    return out


def column_title(label: str) -> str:
    """The label with the engine version its `run.json` recorded, if any.

    The header used to hardcode `qdrant 1.19` beside whatever label was
    passed, so a run against another image would have printed the wrong
    version under a column of its numbers. Only Qdrant reports a version
    string; strawmann is identified by commit in the full table's provenance.
    Major.minor: the block is the README's above-the-fold summary, and the
    full version and image digest are one link away in the provenance table.
    """
    meta = read_json(label, "run.json")
    # The engine, not the label. A column of this block identifies which
    # *engine* the numbers came from, and labels are per run and reused: the
    # front page read `strawmann` / `qdrant` only because those happened to be
    # the label names, and pointing it at the same pair measured under
    # `sm-sift-perf` / `qd-sift-perf` would have put run bookkeeping above the
    # fold. The full table underneath still names the labels, and the
    # provenance section names commits and digests.
    name = meta.get("engine_comm") or label
    v = (meta.get("qdrant") or {}).get("version")
    return f"{name} {'.'.join(str(v).split('.')[:2])}" if v else name


def header_block(a_label: str, b_label: str,
                 a: dict[str, dict], b: dict[str, dict]) -> str:
    """The compact block README.md embeds.

    Generated rather than hand-written because it was hand-written once and went
    stale immediately: the header carried 4,055 for search p=1 while the full
    table said 3,985, a figure from before the W11 fix. Two copies of a number
    drift, and the copy a reader sees first is the one that drifts unnoticed.

    It carries the same refusals as the full table. It used to keep only
    `(a, b, ratio)` and skip any row without a ratio, so a contaminated row
    printed clean and a row refused for unequal recall simply vanished, which
    is the one rendering where a refusal most needs to be visible.
    """
    rows = {r.id: r for r in joined(a_label, b_label, a, b)}
    width = max(len(name) for _, name in HEADER_ROWS)
    # This block names engines rather than labels (`column_title`), so its prose
    # has to as well: a header reading `strawmann` above a note reading
    # `sm-sift-perf` makes the reader work out that they are the same run. The
    # full table below keeps labels, because there the columns *are* labels.
    engines = {lab: (read_json(lab, "run.json").get("engine_comm") or lab)
               for lab in (a_label, b_label)}

    def as_engines(text: str) -> str:
        for lab, eng in engines.items():
            if lab != eng:
                text = text.replace(lab, eng)
        return text

    lines = [f"  {'':<{width}}  {column_title(a_label):>12} "
             f"{column_title(b_label):>16} {'ratio':>10}"]
    notes = []
    for wid, name in HEADER_ROWS:
        r = rows.get(wid)
        if not r:
            continue
        lines.append(f"  {name:<{width}}  {r.sa:>12} {r.sb:>16} {r.ratio:>10}")
        if r.notes:
            notes.append(f"  {name}: {as_engines(r.note_text)}")
    lines.append(f"  {'':<{width}}  {'':>12} {'':>16} {'queries/second':>10}")
    lines += ["", as_engines(scope_line(a_label, b_label))]
    if notes:
        lines += [""] + notes
    for bnr in banners(a_label, b_label):
        lines += [""] + ["  " + ln for ln in wrap(as_engines(bnr), 74)]
    return "```\n" + "\n".join(lines) + "\n```"


def full_block(a_label: str, b_label: str,
               a: dict[str, dict], b: dict[str, dict]) -> str:
    """The full W0-W13 markdown table `docs/comparison-<dataset>.md` embeds.

    The prose around it discusses runs that are now history (the W11 fix was
    measured at 7,260 qps, and that number stays in the narrative because it is
    what the fix measured). This block is the current run, always, and says so.
    """
    purpose = purpose_of()
    a_name, b_name = display_name(a_label), display_name(b_label)
    rows = []
    for bnr in banners(a_label, b_label):
        rows += [f"> **{bnr}**", ""]
    # The id and its purpose are separate columns, as they are in the HTML
    # report: one cell holding "W10-ef32 recall control, ef=32 (latency only)"
    # is a label, not a table.
    rows += [f"| id | workload | {a_name} qps | {b_name} qps | ratio "
             f"| {a_name} p50/p99 | {b_name} p50/p99 | recall@10 | notes |",
             "|---|---|--:|--:|--:|--:|--:|--:|---|"]
    for r in detail_rows(a_label, b_label, a, b):
        wid = r["id"]
        name = purpose.get(wid, "")
        note = r["notes"].replace("[", "").replace("]", "")
        # Recall is a property of the configuration, not of the engine, when
        # both engines searched the same ef; two cells that always agree are
        # one cell.
        recall = r["a_recall"] if r["a_recall"] == r["b_recall"] else \
            f"{r['a_recall']} / {r['b_recall']}"
        rows.append(f"| {wid} | {name} | {r['a_qps']} | {r['b_qps']} | {r['ratio']} "
                    f"| {r['a_lat']} | {r['b_lat']} | {recall} | {note} |")

    io = storage_and_io(a_label, b_label, a, b, markdown=True)
    if io:
        rows += ["", "**Storage and I/O.** Queries per second is half the "
                 "comparison: the two engines hold and write very different "
                 "amounts for the same collection, and only this table shows "
                 "what that buys and costs.", "", *io]
    wall = any(r.get("qps_bfb_median") is not None for r in [*a.values(), *b.values()])
    rows += ["", ("qps is wall-clock: queries / bfb's `duration_secs`, not bfb's "
                  "`Median qps` (kept per row as `qps_bfb_median`), which is a median of "
                  "a rate series and understates short rows. " if wall else
                  "qps here is bfb's `Median qps`: these rows predate the harness's "
                  "wall-clock figure (queries / `duration_secs`), which understates "
                  "short rows by up to ~14% (W4). ") +
             "Latency is client-side "
             "round trip. §7.4: a closed-loop p99 is not a latency result, because a "
             "stalled server stops receiving requests and the tail it did not serve "
             "never appears; the `W4-sat` rows are the open-loop ones, and read "
             "`saturated` where a client p50 past one second says the offered rate "
             "was not served. A ratio is printed only where both engines' recall@10 "
             "at the row's `ef` is known and within 0.01; `-` with `recall unequal` "
             "or `recall missing` says which. A `-` under recall means the sweep "
             "does not speak for that row's configuration rather than that recall "
             "was poor."]
    mism = env_mismatch(a_label, b_label)
    if mism:
        rows += ["", "**Environment mismatch between the two runs:** " + "; ".join(mism) + "."]
    conf = conformance_block(a_label, b_label)
    if conf:
        rows += ["", *conf]
    rows += ["", environment_note(a_label, b_label, a, b)]
    return "\n".join(rows)


def conformance_block(a_label: str, b_label: str) -> list[str]:
    """§8's tiers, generated into the same block as the table.

    Every dataset's document carried a throughput table and no conformance
    result, and the tiers are what license the table: two engines have to be
    shown to answer the same question before their speeds mean anything.
    sift1m's document had them only because §1 was written by hand, so
    dbpedia-openai-100K — measured later, never hand-annotated — asserted
    nothing at all about d=1536 conformance while printing ratios all the same.

    Read from whichever arm ran the differ; it writes one `conformance.json`
    describing the pair, not one per engine.
    """
    for label in (a_label, b_label):
        conf = read_json(label, "conformance.json")
        if conf.get("tiers"):
            break
    else:
        return []
    out = ["**§8 conformance.** The tiers that license the table above.", "",
           "| tier | result | detail |", "|---|---|---|"]
    for t in conf["tiers"]:
        verdict = "pass" if t.get("passed") else "**FAIL**"
        # The differ's details carry `|` as a field separator ("... max=1.5e-5 |
        # ids agree up to ε-ties | ..."), which closes the cell early and shears
        # the rest of the row into columns that do not exist.
        detail = " ".join(str(t.get("detail") or "").split()).replace("|", "\\|")
        out.append(f"| {t.get('tier','?')} | {verdict} | {detail} |")
    reached = conf.get("tier_reached")
    licenses = []
    if conf.get("licenses_perf"):
        licenses.append("performance")
    if conf.get("licenses_comparative"):
        licenses.append("comparative")
    tail = f"Reached **{reached}**" if reached else "No tier reached"
    tail += (f"; licenses {' and '.join(licenses)} claims."
             if licenses else "; licenses no claims.")
    if conf.get("hash"):
        tail += f" Conformance hash `{conf['hash']}`."
    out += ["", tail]
    return out


def blocks_for(a_label: str, b_label: str,
               a: dict[str, dict], b: dict[str, dict]) -> dict[str, str]:
    return {
        "compare-table": header_block(a_label, b_label, a, b),
        "compare-full": full_block(a_label, b_label, a, b),
    }


def splice_readme(blocks: dict[str, str], dataset: str) -> int:
    rc = 0
    for name, kind in blocks_targets(dataset):
        path = ROOT / name
        begin, end = markers(kind)
        if kind == "compare-full" and not path.exists():
            path.write_text(new_comparison_doc(dataset))
            print(f"{name}: created for a dataset measured here for the first time")
        s = path.read_text()
        if begin not in s or end not in s:
            print(f"{name} has no {begin} / {end} markers", file=sys.stderr)
            rc = 1
            continue
        head, rest = s.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = f"{head}{begin}\n{blocks[kind]}\n{end}{tail}"
        if new == s:
            print(f"{name}: {kind} already current")
            continue
        path.write_text(new)
        print(f"{name}: {kind} updated")
    return rc


def readme_is_current(blocks: dict[str, str], have_results: bool,
                      a_label: str = "", b_label: str = "",
                      dataset: str = "sift1m") -> int:
    """Gate step: fail if a generated block is not what compare.py would emit.

    `bench/results/` is gitignored, so a fresh clone and CI have nothing to check
    against. Reported rather than failed: "no data" is not "wrong data".

    Neither is "data that may not be published". `fullrun.render` calls
    `--write-readme` only when nothing refuses the results, so a STALE or
    UNLICENSED set can never have been rendered into these blocks, and demanding
    freshness of it asks for a file the publish gate forbids writing. Those
    mismatches are printed, because they are real, but do not fail the step —
    which is what its contract in `scripts/check.py` says.

    Found by the dbpedia-openai-1m run: stopped before its differ, it
    left measured rows, no conformance row, and a red gate whose printed remedy was
    the one thing §8 forbids for unlicensed rows.
    """
    if not have_results:
        print("no measured results on this host; generated tables not checked")
        return 0
    rc = 0
    problems = 0
    stale = stale_reasons(a_label, b_label) if a_label and b_label else []
    lic = licence(a_label, b_label) if a_label and b_label else {"comparative": True}
    publishable = not stale and bool(lic.get("comparative"))
    for name, kind in blocks_targets(dataset):
        path = ROOT / name
        if kind == "compare-full" and not path.exists():
            print(f"{name} does not exist; this dataset has not been rendered here",
                  file=sys.stderr)
            problems += 1
            continue
        s = path.read_text()
        begin, end = markers(kind)
        if begin not in s or end not in s:
            print(f"{name} is missing the {kind} markers", file=sys.stderr)
            problems += 1
            continue
        current = s.split(begin, 1)[1].split(end, 1)[0].strip()
        if current != blocks[kind].strip():
            print(f"{name}'s {kind} table is stale.", file=sys.stderr)
            print("Run: bench/harness/compare.py strawmann qdrant --write-readme",
                  file=sys.stderr)
            problems += 1
            continue
        print(f"ok: {name}'s {kind} table matches the measured results")
    # The measured results themselves may be what is stale. That is reported
    # loudly and does not fail the gate: development on this host would
    # otherwise be blocked by result sets that only a full re-run can replace,
    # and the tables already say STALE / UNLICENSED where a reader will see it.
    if stale or not lic.get("comparative"):
        print("\n!! WARNING: the measured results on this host do not support a "
              "comparative claim, and the generated tables say so:", file=sys.stderr)
        for reason in stale:
            print(f"!!   STALE: {reason}", file=sys.stderr)
        if not lic.get("comparative"):
            print(f"!!   {lic.get('banner') or 'licenses_comparative is false'}",
                  file=sys.stderr)
        print("!! Re-run bench/harness/fullrun.py to replace them.", file=sys.stderr)
    if problems:
        if publishable:
            rc = 1
        else:
            print(f"!! {problems} generated block(s) above do not match, and this "
                  f"step is not failing for them: a result set that may not be "
                  f"published cannot be written into those blocks at all.",
                  file=sys.stderr)
    return rc




def main(argv: list[str]) -> int:
    # argparse, like the rest of the harness. Flags used to be collected into a
    # set and anything unrecognised dropped, so a misspelt `--write-readmee`
    # printed the table and exited 0 while the README kept the old numbers.
    ap = argparse.ArgumentParser(
        prog="compare.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a_label")
    ap.add_argument("b_label")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--readme", action="store_true",
                     help="print the generated blocks instead of the full table")
    out.add_argument("--write-readme", action="store_true",
                     help="splice them into README.md and the per-dataset document")
    out.add_argument("--check-readme", action="store_true",
                     help="gate step: fail if a generated block is stale")
    try:
        args = ap.parse_args(argv[1:])
    except SystemExit as e:
        return int(e.code or 0)

    a_label, b_label = args.a_label, args.b_label
    lenient = args.check_readme
    a, b = load(a_label, not lenient), load(b_label, not lenient)

    if args.readme or args.write_readme or args.check_readme:
        blocks = blocks_for(a_label, b_label, a, b)
        dataset = dataset_of(a_label) or dataset_of(b_label) or "sift1m"
        if args.write_readme:
            return splice_readme(blocks, dataset)
        if args.check_readme:
            return readme_is_current(blocks, bool(a) and bool(b), a_label, b_label, dataset)
        print(blocks["compare-table"])
        print()
        print(blocks["compare-full"])
        return 0

    print(f"# {a_label} vs {b_label}, queries per second (higher is better)\n")
    for bnr in banners(a_label, b_label):
        for ln in wrap(bnr):
            print(ln)
        print()
    hdr = (f"{'workload':<14} {a_label + ' qps':>13} {b_label + ' qps':>13} {'ratio':>8}"
           f" {'p50/p99 ' + a_label:>19} {'p50/p99 ' + b_label:>19} {'recall@10':>10}  notes")
    print(hdr)
    print("-" * len(hdr))

    for r in detail_rows(a_label, b_label, a, b):
        recall = r["a_recall"] if r["a_recall"] == r["b_recall"] else \
            f"{r['a_recall']}/{r['b_recall']}"
        print(f"{r['id']:<14} {r['a_qps']:>13} {r['b_qps']:>13} {r['ratio']:>8}"
              f" {r['a_lat']:>19} {r['b_lat']:>19} {recall:>10}  {r['notes']}")

    print()
    print("units: qps = queries/second. bfb's `rps` counts batch requests and differs")
    print("       from qps wherever --search-batch-size > 1 (W5). Reading rps there")
    print("       turned a 2.2x speedup into an apparent 7x regression.")
    print("W4-sat rows are open loop: the figure is the *offered* rate, so serving it")
    print("       means the engine kept up, not that it was faster.")
    print("latency is the client-side round trip. §7.4: a closed-loop p99 is not a")
    print("       latency result; the W4-sat rows are the open-loop ones.")
    print("recall@10 is joined from the conformance sweep on (dataset, collection, ef),")
    print("       and only for rows it speaks for. `-` means not measured for that")
    print("       configuration. A ratio needs both recalls known and within 0.01.")
    print("qps is wall-clock queries / duration from bfb's JSON; bfb's `Median qps` is")
    print("       kept per row as `qps_bfb_median`.")
    print()
    for m in env_mismatch(a_label, b_label):
        print(f"environment mismatch: {m}")

    io = storage_and_io(a_label, b_label, a, b)
    if io:
        print("storage and I/O over the whole run:")
        print()
        for line in io:
            print("  " + line if line else "")
        print()

    for label, rows in ((a_label, a), (b_label, b)):
        print(environment_line(label, rows))

    print()
    print("A failing §7.1 gate makes every number here development-grade. §7.1: results")
    print("from different environments never share a chart, and an ungated host is a")
    print("different environment from a gated one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
