#!/usr/bin/env python3
"""§7.3 / §8.9, the results sink.

§7.3: "`bfb --json` output plus server-side counters merged into a single row
per run, stored in SQLite, with a small script producing the comparison tables."

§8 states the rule this file exists to enforce:

    "no performance number is publishable unless the same build, on the same
     data, has a green conformance row. This is enforced by the harness, not by
     convention, the results sink rejects a perf row whose conformance hash
     doesn't match a passing run."

So `insert_perf` is not a writer with a validation helper beside it. It refuses.
A perf row without a matching green conformance row does not enter the database,
because a number that exists in the results table is a number someone will
eventually put in a chart.

§8.9 fixes the row shape:

    workload · dataset name + base/query/GT checksums · qdrant version ·
    strawmann commit · ISA build · conformance tier reached ·
    max|Δscore| · p99|Δscore| · recall@1/10/100 · mean rel. distance error ·
    nDCG@10 (where qrels exist) · QPS · p50/p99 latency ·
    effective frequency · %-of-roofline
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conformance (
    hash              TEXT PRIMARY KEY,
    dataset           TEXT NOT NULL,
    metric            TEXT NOT NULL,
    dim               INTEGER NOT NULL,
    qdrant_version    TEXT NOT NULL,
    strawmann_commit  TEXT NOT NULL,
    isa_build         TEXT NOT NULL,
    tier_reached      TEXT NOT NULL,
    licenses_perf     INTEGER NOT NULL,
    -- §8.5 T3: the engines are at equal recall, so a ratio between them is a
    -- claim. `compare` prints one only when this is set.
    licenses_comparative INTEGER NOT NULL DEFAULT 0,
    base_checksum     TEXT NOT NULL,
    query_checksum    TEXT NOT NULL,
    max_delta         REAL,
    p99_delta         REAL,
    detail            TEXT,
    created_at        TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS perf (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    workload          TEXT NOT NULL,
    engine            TEXT NOT NULL,
    dataset           TEXT NOT NULL,
    isa_build         TEXT NOT NULL,
    -- §7.1: "Every result row carries a hash of the environment description.
    -- Results from different environments never share a chart."
    env_hash          TEXT NOT NULL,
    -- §8: the gate. A row cannot exist without a green conformance row.
    conformance_hash  TEXT NOT NULL REFERENCES conformance(hash),

    qps               REAL,
    p50_us            REAL,
    p95_us            REAL,
    p99_us            REAL,
    p999_us           REAL,
    -- §7.4: "a single QPS number without its recall is meaningless and this
    -- project should never emit one."
    recall_at_1       REAL,
    recall_at_10      REAL,
    recall_at_100     REAL,
    mrde              REAL,
    ndcg_at_10        REAL,

    -- §7.5: "A cycles/vector win presented without its frequency is not a result."
    effective_freq    REAL,
    pct_of_roofline   REAL,
    -- §7.4: "--rps (fixed rate) and --parallel (closed loop) measure different
    -- things. Use --rps for latency claims and closed-loop for saturation
    -- throughput. Never quote a closed-loop p99 as a latency result."
    load_mode         TEXT NOT NULL,
    notes             TEXT,
    -- The measurement's own timestamp (`rows.json` `when`), so a re-ingest of
    -- the same run is the same row and not a second one. `created_at` is when
    -- the sink saw it, which is a different fact.
    measured_at       TEXT NOT NULL DEFAULT '',
    created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (workload, engine, isa_build, env_hash, conformance_hash, measured_at)
);

CREATE INDEX IF NOT EXISTS perf_by_workload ON perf(workload, engine, isa_build);
"""


def _migrate(db):
    """Older databases predate two columns and the UNIQUE constraint.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so a sink
    created before this schema would keep taking duplicate rows and would have
    no `licenses_comparative` to gate on. Columns are added in place; the
    UNIQUE constraint cannot be added to a live SQLite table, so the perf table
    is rebuilt with duplicates collapsed to their latest row.
    """
    cols = {r[1] for r in db.execute("PRAGMA table_info(conformance)")}
    if "licenses_comparative" not in cols:
        db.execute("ALTER TABLE conformance ADD COLUMN licenses_comparative "
                   "INTEGER NOT NULL DEFAULT 0")
    pcols = {r[1] for r in db.execute("PRAGMA table_info(perf)")}
    if "measured_at" not in pcols:
        db.execute("ALTER TABLE perf ADD COLUMN measured_at TEXT NOT NULL DEFAULT ''")
    has_unique = any("UNIQUE" in (r[0] or "").upper() for r in db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='perf'"))
    if not has_unique:
        db.execute("ALTER TABLE perf RENAME TO perf_old")
        db.executescript(SCHEMA)
        db.execute("""INSERT OR REPLACE INTO perf
                      (workload, engine, dataset, isa_build, env_hash, conformance_hash,
                       qps, p50_us, p95_us, p99_us, p999_us, recall_at_1, recall_at_10,
                       recall_at_100, mrde, ndcg_at_10, effective_freq, pct_of_roofline,
                       load_mode, notes, measured_at, created_at)
                      SELECT workload, engine, dataset, isa_build, env_hash, conformance_hash,
                             qps, p50_us, p95_us, p99_us, p999_us, recall_at_1, recall_at_10,
                             recall_at_100, mrde, ndcg_at_10, effective_freq, pct_of_roofline,
                             load_mode, notes, measured_at, created_at
                      FROM perf_old ORDER BY id""")
        db.execute("DROP TABLE perf_old")
        # The index followed the rename onto `perf_old` and was dropped with
        # it, and `CREATE INDEX IF NOT EXISTS` above had seen it there and
        # done nothing; the rebuilt table was left without one.
        db.executescript(SCHEMA)
    db.commit()


class RejectedRow(Exception):
    """§8's gate, as an exception rather than a return code.

    A caller that ignores a return code writes the row anyway; a caller that
    ignores an exception does not exist.
    """


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(SCHEMA)
    _migrate(db)
    return db


def insert_conformance(db, row):
    db.execute(
        """INSERT OR REPLACE INTO conformance
           (hash, dataset, metric, dim, qdrant_version, strawmann_commit, isa_build,
            tier_reached, licenses_perf, licenses_comparative, base_checksum, query_checksum,
            max_delta, p99_delta, detail)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row["hash"], row["dataset"], row["metric"], row["dim"],
            row["qdrant_version"], row["strawmann_commit"], row["isa_build"],
            row["tier_reached"], int(row["licenses_perf"]),
            int(bool(row.get("licenses_comparative", False))),
            row["base_checksum"], row["query_checksum"],
            row.get("max_delta"), row.get("p99_delta"), row.get("detail", ""),
        ),
    )
    db.commit()


def insert_perf(db, row):
    """§8: reject rather than flag."""
    h = row.get("conformance_hash")
    if not h:
        raise RejectedRow(
            "perf row carries no conformance hash. §8: no performance number is "
            "publishable unless the same build, on the same data, has a green "
            "conformance row."
        )

    got = db.execute(
        "SELECT tier_reached, licenses_perf, isa_build, dataset, strawmann_commit, "
        "qdrant_version FROM conformance WHERE hash = ?",
        (h,),
    ).fetchone()
    if got is None:
        raise RejectedRow(
            f"conformance hash {h} matches no recorded run. §8: the results sink "
            "rejects a perf row whose conformance hash doesn't match a passing run."
        )

    tier, licenses, c_isa, c_dataset, c_commit, c_qdrant = got
    if not licenses:
        raise RejectedRow(
            f"conformance run {h} reached {tier}, which does not license a "
            "performance claim. §8.5: T1 (exact-search value equality) is the tier "
            "that does."
        )

    # §7.1: results from different environments never share a chart. The same
    # argument applies within a row: a perf number measured on one ISA build
    # cannot be licensed by a conformance run from another.
    if row["isa_build"] != c_isa:
        raise RejectedRow(
            f"perf row is from ISA build '{row['isa_build']}' but its conformance "
            f"run is from '{c_isa}'. §7.1: results from different environments "
            "never share a chart."
        )
    if row["dataset"] != c_dataset:
        raise RejectedRow(
            f"perf row is on dataset '{row['dataset']}' but its conformance run "
            f"is on '{c_dataset}'. §8 requires the same build *on the same data*."
        )
    # "The same build": the ISA build above is a third of it. The strawmann
    # commit and the Qdrant version are the other two, and a row that names
    # neither cannot be shown to be from the build the conformance run tested.
    sm_commit, qd_version = row.get("strawmann_commit"), row.get("qdrant_version")
    if sm_commit is None and qd_version is None:
        raise RejectedRow(
            "perf row carries no build identity (strawmann_commit / qdrant_version). "
            "§8: no performance number is publishable unless *the same build* has "
            "a green conformance row, and an unnamed build cannot be the same one."
        )
    if sm_commit is not None and sm_commit != c_commit:
        raise RejectedRow(
            f"perf row is from strawmann commit '{sm_commit}' but its conformance "
            f"run is from '{c_commit}'. §8 requires the same build."
        )
    if qd_version is not None and qd_version != c_qdrant:
        raise RejectedRow(
            f"perf row is from Qdrant '{qd_version}' but its conformance run "
            f"is from '{c_qdrant}'. §8 requires the same build."
        )

    # §7.4: a QPS number without its recall is not publishable. Exempt by name
    # rather than by omission: W0-W2 are plumbing and ingest, and W13 is
    # `--scroll`, which walks ids in order and returns no ranking — there is no
    # recall to have, as opposed to one that went unmeasured. W12 is deliberately
    # not here either, and now for the opposite reason: it is a filtered
    # *search*, its recall exists, and since 2026-09-08 it is measured — per
    # condition, by `recall.py`'s two grades against a restricted ground truth.
    # A W12 row reaching this refusal now means the sweep did not run or its
    # condition did not match the row's, which is a fault to fix rather than a
    # state to exempt.
    plumbing = row["workload"] in ("W0", "W1", "W2", "W13")
    if not plumbing and row.get("qps") is not None and row.get("recall_at_10") is None:
        raise RejectedRow(
            f"{row['workload']}: QPS given without recall@10. §7.4: 'a single QPS "
            "number without its recall is meaningless and this project should never "
            "emit one.'"
        )

    # An upsert on the measurement's identity: ingesting the same run twice
    # used to add a second identical row, and `compare` then read whichever
    # came first. The UNIQUE constraint makes the re-ingest a replace.
    db.execute(
        """INSERT OR REPLACE INTO perf
           (workload, engine, dataset, isa_build, env_hash, conformance_hash,
            qps, p50_us, p95_us, p99_us, p999_us,
            recall_at_1, recall_at_10, recall_at_100, mrde, ndcg_at_10,
            effective_freq, pct_of_roofline, load_mode, notes, measured_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row["workload"], row["engine"], row["dataset"], row["isa_build"],
            row["env_hash"], h,
            row.get("qps"), row.get("p50_us"), row.get("p95_us"),
            row.get("p99_us"), row.get("p999_us"),
            row.get("recall_at_1"), row.get("recall_at_10"), row.get("recall_at_100"),
            row.get("mrde"), row.get("ndcg_at_10"),
            row.get("effective_freq"), row.get("pct_of_roofline"),
            row.get("load_mode", "closed-loop"), row.get("notes", ""),
            row.get("measured_at") or "",
        ),
    )
    db.commit()


def compare(db, workload):
    """§7.4: "Compare at equal recall. Publish QPS-vs-recall curves."

    The table deliberately puts recall next to QPS in every row, and refuses to
    compute a ratio when the recalls differ, §7.4 again: "QPS is compared at
    equal recall, never at equal `ef`."
    """
    rows = db.execute(
        """SELECT p.engine, p.isa_build, p.qps, p.recall_at_10, p.p50_us, p.p99_us,
                  p.effective_freq, p.pct_of_roofline, p.load_mode,
                  p.id, p.measured_at, p.conformance_hash, c.licenses_comparative,
                  c.tier_reached
           FROM perf p JOIN conformance c ON c.hash = p.conformance_hash
           WHERE p.workload = ? ORDER BY p.engine, p.isa_build, p.measured_at, p.id""",
        (workload,),
    ).fetchall()
    if not rows:
        print(f"no rows for {workload}")
        return

    print(f"# {workload}\n")
    print(f"{'engine':<12} {'isa':<14} {'QPS':>10} {'recall@10':>10} "
          f"{'p50 us':>9} {'p99 us':>9} {'eff.freq':>9} {'%roofline':>10}  load")
    for r in rows:
        eng, isa, qps, rec, p50, p99, ef, roof, load, *_ = r
        fmt = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
        print(f"{eng:<12} {isa:<14} {fmt(qps,0):>10} {fmt(rec,4):>10} "
              f"{fmt(p50):>9} {fmt(p99):>9} {fmt(ef,3):>9} {fmt(roof,1):>10}  {load}")

    # The ratio, only where it is meaningful: the *latest* matched pair, i.e.
    # the newest strawmann row and the newest qdrant row that share a
    # conformance hash (one comparison is one differ run), both with recall.
    # `byeng[eng][0]` used to take the oldest row of each, which after a
    # re-measure compared a fresh strawmann against a stale qdrant.
    latest = {}
    for eng, _isa, qps, rec, *_rest in rows:
        rowid, when, chash, comparative, tier = _rest[5:10]
        if qps is None or rec is None:
            continue
        # Newest by measurement time, then rowid. The rows arrive ordered by
        # `isa_build` first, so "last seen" was the alphabetically last ISA
        # build's row rather than the latest measurement.
        cur = latest.setdefault(chash, {}).get(eng)
        if cur is None or (when or "", rowid) > (cur[2], cur[3]):
            latest[chash][eng] = (rec, qps, when or "", rowid, comparative, tier)
    pairs = [(h, d) for h, d in latest.items() if "strawmann" in d and "qdrant" in d]
    if pairs:
        # Newest pair by the later of the two measurements, then by rowid.
        h, d = max(pairs, key=lambda hd: (max(hd[1]["strawmann"][2], hd[1]["qdrant"][2]),
                                          max(hd[1]["strawmann"][3], hd[1]["qdrant"][3])))
        s_rec, s_qps, _, _, comparative, tier = d["strawmann"]
        q_rec, q_qps, *_ = d["qdrant"]
        print()
        if not comparative:
            print(f"NOT COMPARABLE: conformance run {h} reached {tier} and does not "
                  f"license a comparative claim (§8.5 T3, licenses_comparative=false).")
            print("Per-engine numbers above may be quoted; no ratio between them may.")
        elif abs(s_rec - q_rec) > 0.01:
            print(f"NOT COMPARABLE: recall@10 differs ({s_rec:.4f} vs {q_rec:.4f}).")
            print("§7.4: QPS is compared at equal recall, never at equal ef.")
        elif not (s_qps > 0 and q_qps > 0):
            # A row that served nothing is not a measurement, and the pair is
            # still the latest one: substituting an older row here would
            # compare a fresh arm against a stale one, which is the bug the
            # `latest` selection above exists to prevent. Refused by name
            # rather than divided by zero.
            print(f"NOT COMPARABLE: a zero QPS row is not a measurement "
                  f"(strawmann {s_qps:,.0f}, qdrant {q_qps:,.0f}, conformance {h}).")
        else:
            print(f"at matched recall@10 ~= {s_rec:.4f}: strawmann {s_qps/q_qps:.2f}x qdrant"
                  f"  (conformance {h}, latest matched pair)")

    # By name. The same rows are unpacked by name above, and this used to be
    # `r[8]`: adding a column to the SELECT would have silently changed which
    # field this reads.
    # `load_mode` is the ninth SELECT column; `*_, load` bound the *last* one
    # (`tier_reached`), so this note never printed.
    if any(load == "closed-loop" for (_e, _i, _q, _r, _p50, _p99, _f, _pr, load, *_rest) in rows):
        print()
        print("NOTE: closed-loop rows present. §7.4: 'Never quote a closed-loop p99")
        print("as a latency result.' Use --rps rows for latency claims.")


def ingest_run(db, label: str, root: Path) -> int:
    """Record a completed run, from the files the harness already writes.

    This is the step that was missing. §8's rule, "the results sink rejects a
    perf row whose conformance hash doesn't match a passing run", was
    implemented here and enforced by nothing, because nothing ever called it:
    the perf rows lived in `rows.json`, the conformance hash was printed to a
    terminal by the Rust differ, and no code path joined them. A gate that is
    never invoked is a comment.

    So this reads what a run leaves on disk and does the join:

      `run.json`          which engine, which dataset, which host
      `conformance.json`  the differ's row, including the hash §8 checks
      `rows.json`         the measured workloads
      `recall.*.json`     §7.4's other half, joined on (dataset, collection, ef)

    A missing conformance file is reported as the refusal it is, not as a
    warning: without it there is no green row to license anything, which is the
    project's actual state today and worth saying out loud on every run.
    """
    d = root / "bench/results" / label

    def load(name: str) -> dict | list | None:
        p = d / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  !! {p} is unreadable; skipping", file=sys.stderr)
            return None

    rows = load("rows.json") or []
    meta = load("run.json") or {}
    conf = load("conformance.json")

    if conf:
        insert_conformance(db, conf)
        print(f"recorded conformance {conf['hash']} "
              f"(tier {conf.get('tier_reached')}, "
              f"{'licenses' if conf.get('licenses_perf') else 'does NOT license'} perf)")

    env = (d / "env.txt")
    env_hash = "unknown"
    if env.exists():
        m = re.search(r"environment hash: ([0-9a-f]+)", env.read_text())
        if m:
            env_hash = m.group(1)

    sm = meta.get("strawmann") or {}
    qd = meta.get("qdrant") or {}
    # No default. `native` was the exact failure `provenance.isa_build`'s
    # docstring names: an ISA-sweep arm whose run never said which build it
    # was ingested as the native one. Unknown stays None and the gate's
    # "never share a chart" refusal applies. A Qdrant arm has no ISA build of
    # its own; the differ's row is one comparison of one pair of builds, and
    # copying it to the Qdrant label is what licenses that arm, so its rows
    # file under the build the row names.
    isa = meta.get("isa_build") or (((conf or {}).get("isa_build")) if qd and not sm else None)
    dataset = (meta.get("dataset") or {}).get("name", "unknown")

    # A file whose rows were served by more than one build is not a run:
    # `rows.json` merges in place, and a subset re-run on a rebuilt binary
    # would otherwise ingest under whichever identity `run.json` now carries.
    import compare
    mixed = compare.mixed_build_reasons(label, {r["id"]: r for r in rows if "id" in r})
    if mixed:
        print(f"REJECTED, and this is the gate working: {mixed[0]}", file=sys.stderr)
        return 1

    # The recall the sweep measured at each `ef` *of the collection the row
    # searched* (§7.4). It used to read `recall.json`, bench2's fp32 sweep,
    # and join it onto every row by `ef` alone, so W6/W7/W8's quantized
    # searches carried fp32's recall. Now the key is (dataset, collection, ef),
    # read through `recall.load_recall`, which also checks the fields inside
    # the file; a row whose collection has no sweep gets NULL, and the sink's
    # qps-without-recall rule then applies to it.
    import recall as recall_mod
    sweeps: dict = {}

    def recall_for(r: dict) -> dict | None:
        coll, ef = r.get("collection"), r.get("ef")
        # An exhaustive search returns the true neighbours, so its recall@10 is
        # 1.0 by construction — and the construction is checked, not assumed:
        # this row is only ingested at all if its conformance run passed T1,
        # exact-search value equality against the fp64 oracle. Without this an
        # exact row can never carry a recall (the sweep joins on `ef`, which it
        # has none of) and §7.4's qps-without-recall rule refuses it forever.
        if r.get("exact"):
            return {"recall_at_1": 1.0, "recall_at_10": 1.0}
        if not coll or ef is None or not r.get("recall_joinable", True):
            return None
        # W7 searches at oversampling 4; a sweep recorded at 1 is a different
        # search, so the join key carries the row's quantization params — and
        # the selectivity grade, for the same reason: `bench12` has one sweep
        # per condition and a row filtered to 1% is not described by the 10%
        # one. Without the grade a W12 row asked for an *unfiltered* sweep of
        # `bench12`, got nothing because none is written, and was refused for
        # having no recall — the whole feature defeated at the last join.
        #
        # This is the second of two independent lookups; `compare.recall_at` is
        # the other, and it keys on the grade too. They stay separate because
        # one feeds the §8 sink and one the report, but a row must not be
        # joined differently by them.
        key = (coll, r.get("quantization_oversampling"), r.get("quantization_rescore"),
               recall_mod.grade_of(r["id"]))
        if key not in sweeps:
            sweeps[key] = recall_mod.load_recall(label, dataset, coll, key[1], key[2],
                                                 key[3])
        return sweeps[key].get(int(ef))

    accepted = rejected = 0
    #: Every refused row, not the first: the sink refused 2 + 3
    #: rows and named one per label, so the other three had to be found by
    #: elimination.
    refused: list[tuple[str, str, str]] = []
    for r in rows:
        if r.get("qps") is None:
            continue
        lat = r.get("latency") or {}
        rec = recall_for(r)
        row = {
            "workload": r["id"], "engine": label, "dataset": dataset,
            # The row's own build stamp where it has one; the label-level
            # value only for rows measured before rows were stamped.
            "isa_build": r.get("isa_build") or isa, "env_hash": env_hash,
            "conformance_hash": (conf or {}).get("hash"),
            "qps": r.get("qps"),
            "p50_us": lat.get("client_p50_us"), "p95_us": lat.get("client_p95_us"),
            "p99_us": lat.get("client_p99_us"), "p999_us": lat.get("client_p999_us"),
            "recall_at_1": (rec or {}).get("recall_at_1"),
            "recall_at_10": (rec or {}).get("recall_at_10"),
            "recall_at_100": (rec or {}).get("recall_at_100"),
            "mrde": (rec or {}).get("mean_relative_distance_error"),
            "load_mode": r.get("load_mode") or "closed-loop",
            "notes": "; ".join(x for x in (r.get("foreign", ""), r.get("notes", "")) if x),
            "measured_at": r.get("when", ""),
            "version": sm.get("commit") or qd.get("version"),
            # As the differ spells them (`fullrun.build_identity`): the short
            # commit with `-dirty` when the tree was, and Qdrant's own version.
            "strawmann_commit": (sm["commit"] + ("-dirty" if sm.get("dirty") else "")
                                 if sm.get("commit") else None),
            "qdrant_version": qd.get("version"),
        }
        try:
            insert_perf(db, row)
            accepted += 1
        except RejectedRow as e:
            rejected += 1
            refused.append((r["id"], r.get("collection") or "", str(e)))
            if rejected == 1:
                print(f"REJECTED, and this is the gate working: {e}", file=sys.stderr)

    print(f"{accepted} perf row(s) recorded, {rejected} refused by §8")
    if rejected:
        for rid, _coll, why in refused:
            print(f"  refused {rid}: {why.split('. ')[0]}", file=sys.stderr)
        if not conf:
            print(f"  run the differ with --json bench/results/{label}/conformance.json to "
                  "produce the green row these need", file=sys.stderr)
        without = [(rid, coll) for rid, coll, why in refused if "without recall" in why]
        if without:
            # The hint used to send the operator to sweep a collection that
            # W11 had just mutated, or W12's that nothing swept. `bench12` is
            # swept per grade since 2026-09-08, so W12 is no longer in the
            # by-design half of this sentence.
            print("  a refused search row needs a recall sweep of the collection it "
                  f"searched (bench/harness/recall.py {label}, "
                  "recall.<dataset>.<collection>.json); none is on disk for "
                  + ", ".join(f"{rid} on {coll or '?'}" for rid, coll in without)
                  + ". A row that mutates its collection (W11) cannot have one "
                  "and stays refused by design; a filtered row needs the sweep "
                  "of its own grade (recall.<dataset>.bench12.<grade>.json), "
                  "which `--filtered-base-n` is required for.", file=sys.stderr)
    return 0 if accepted or not rows else 1


def _self_test() -> int:
    """§8's gate, exercised in both directions on a throwaway database.

    A gate is only worth having if it refuses, and a refusal is only worth
    having if it also *accepts* the good case: a check that rejects everything
    passes a test suite that only ever feeds it bad rows, and then silently
    blocks the pipeline it was meant to guard.
    """
    db = connect(":memory:")
    conf = {
        "hash": "abc123", "dataset": "sift1m", "metric": "Euclid", "dim": 128,
        "qdrant_version": "1.19.0", "strawmann_commit": "deadbeef",
        "isa_build": "native", "tier_reached": "T1", "licenses_perf": True,
        "base_checksum": "7bf7", "query_checksum": "1eb6", "detail": "",
    }
    insert_conformance(db, conf)

    good = {
        "workload": "W3", "engine": "strawmann", "dataset": "sift1m",
        "isa_build": "native", "env_hash": "e0", "conformance_hash": "abc123",
        "strawmann_commit": "deadbeef",
        "qps": 4095.0, "p50_us": 95.0, "p99_us": 154.0,
        "recall_at_10": 0.9817, "load_mode": "closed-loop",
    }
    insert_perf(db, good)
    n = db.execute("SELECT count(*) FROM perf").fetchone()[0]
    assert n == 1, n

    def refuses(row, because: str):
        try:
            insert_perf(db, row)
        except RejectedRow as e:
            assert because in str(e), f"wrong reason for {because}: {e}"
            return
        raise AssertionError(f"accepted a row it should refuse: {because}")

    refuses({**good, "conformance_hash": None}, "no conformance hash")
    refuses({**good, "conformance_hash": "nosuch"}, "matches no recorded run")
    refuses({**good, "isa_build": "avx2"}, "never share a chart")
    refuses({**good, "dataset": "gist1m"}, "same build *on the same data*")
    # The other two thirds of "the same build".
    refuses({**good, "strawmann_commit": "cafebabe"}, "strawmann commit")
    refuses({**good, "strawmann_commit": None}, "no build identity")
    refuses({**good, "strawmann_commit": None, "qdrant_version": "1.18.0"}, "Qdrant '1.18.0'")
    insert_perf(db, {**good, "strawmann_commit": None, "qdrant_version": "1.19.0",
                     "engine": "qdrant-ok", "measured_at": "x"})
    # §7.4, the rule this project argues for most often.
    refuses({k: v for k, v in good.items() if k != "recall_at_10"},
            "QPS given without recall")
    # W0-W2 are plumbing: recall is not defined for them, so their exemption is
    # by name and has to keep working.
    insert_perf(db, {k: v for k, v in {**good, "workload": "W0"}.items()
                     if k != "recall_at_10"})

    insert_conformance(db, {**conf, "hash": "t0only", "tier_reached": "T0",
                            "licenses_perf": False})
    refuses({**good, "conformance_hash": "t0only"}, "does not license")

    assert db.execute("SELECT count(*) FROM perf").fetchone()[0] == 3

    # Re-ingesting the same measurement is a replace, not a second row.
    insert_perf(db, {**good, "measured_at": "2026-01-01T00:00:00Z", "qps": 4000.0})
    insert_perf(db, {**good, "measured_at": "2026-01-01T00:00:00Z", "qps": 4001.0})
    n = db.execute("SELECT count(*) FROM perf WHERE workload='W3' AND engine='strawmann' "
                   "AND measured_at='2026-01-01T00:00:00Z'").fetchone()[0]
    assert n == 1, f"duplicate ingest produced {n} rows"
    assert db.execute("SELECT qps FROM perf WHERE measured_at='2026-01-01T00:00:00Z'"
                      ).fetchone()[0] == 4001.0

    # `compare` picks the latest matched pair and gates on T3.
    import contextlib
    import io as _io

    def compare_text(wl: str) -> str:
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            compare(db, wl)
        return buf.getvalue()

    insert_perf(db, {**good, "engine": "qdrant", "qps": 2000.0, "recall_at_10": 0.9820,
                     "measured_at": "2026-01-01T00:00:00Z"})
    out = compare_text("W3")
    assert "NOT COMPARABLE" in out and "T3" in out, out
    insert_conformance(db, {**conf, "hash": "t3ok", "tier_reached": "T3",
                            "licenses_comparative": True})
    insert_perf(db, {**good, "conformance_hash": "t3ok", "qps": 3000.0,
                     "measured_at": "2026-02-01T00:00:00Z"})
    insert_perf(db, {**good, "conformance_hash": "t3ok", "engine": "qdrant", "qps": 1000.0,
                     "recall_at_10": 0.9820, "measured_at": "2026-02-01T00:00:00Z"})
    out = compare_text("W3")
    assert "3.00x" in out and "t3ok" in out, out
    # Every row above is closed-loop, so §7.4's note must be there; it bound
    # the last SELECT column instead of `load_mode` and never printed.
    assert "closed-loop rows present" in out, out
    insert_perf(db, {**good, "conformance_hash": "t3ok", "engine": "qdrant", "qps": 1000.0,
                     "recall_at_10": 0.90, "measured_at": "2026-03-01T00:00:00Z"})
    out = compare_text("W3")
    assert "recall@10 differs" in out, out

    print("ok: §8 gate accepts a licensed row, refuses nine ways, dedupes re-ingest, "
          "and compare gates on T3 with the latest pair")
    return 0


def db_path(arg: str, root: Path) -> str:
    """`--db` against the repository root, so `results.py ingest x` from
    `bench/` and from the root write the same file. `:memory:` and absolute
    paths pass through."""
    if arg == ":memory:" or os.path.isabs(arg):
        return arg
    return str(root / arg)


def main():
    ap = argparse.ArgumentParser(description="strawmann results sink (§7.3, §8)")
    if "--self-test" in sys.argv:
        return _self_test()
    ap.add_argument("--db", default="bench/results.sqlite",
                    help="relative paths resolve against the repository root, not the cwd")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("conformance", help="record a conformance run")
    p.add_argument("--json", required=True, help="JSON file, or - for stdin")

    p = sub.add_parser("perf", help="record a performance run (gated by §8)")
    p.add_argument("--json", required=True)

    p = sub.add_parser("compare", help="print the comparison table for a workload")
    p.add_argument("workload")

    p = sub.add_parser("ingest", help="record a completed run from bench/results/<label>")
    p.add_argument("label")
    p.add_argument("--root", default=None)

    args = ap.parse_args()
    root = Path(args.root) if getattr(args, "root", None) else \
        Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
    db = connect(db_path(args.db, root))

    if args.cmd == "compare":
        compare(db, args.workload)
        return 0

    if args.cmd == "ingest":
        return ingest_run(db, args.label, root)

    text = sys.stdin.read() if args.json == "-" else Path(args.json).read_text()
    row = json.loads(text)

    if args.cmd == "conformance":
        insert_conformance(db, row)
        print(f"recorded conformance {row['hash']} (tier {row['tier_reached']})")
        return 0

    try:
        insert_perf(db, row)
    except RejectedRow as e:
        print(f"REJECTED: {e}", file=sys.stderr)
        return 1
    print(f"recorded perf row for {row['workload']} / {row['engine']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
