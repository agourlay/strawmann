#!/usr/bin/env python3
"""Unit tests: the sweeps, the noise floor, the §8 sink and what they record."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from harness_fixtures import _reload, _row, _sweep, good_stamp

from workloads import stamp_hash


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_stamp_adds_dataset_and_refuses_mismatch(self):
        rc = self.m["recall"]
        p = rc.recall_path("lbl", "sift1m", "bench2")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"collection": "bench2", "points": []}))
        self.assertTrue(rc.stamp(p, "sift1m", "bench2", "euclid"))
        d = json.loads(p.read_text())
        self.assertEqual((d["dataset"], d["metric"]), ("sift1m", "euclid"))
        p.write_text(json.dumps({"collection": "relevance", "points": []}))
        self.assertFalse(rc.stamp(p, "sift1m", "bench2", "euclid"))
        self.assertFalse(p.exists())
        self.assertTrue(p.with_suffix(".mismatch.json").exists())

    def test_stamp_refuses_a_sweep_that_scored_a_different_corpus(self):
        """The names lining up does not mean the contents do.

        `relevance --skip-upload` scores whatever is in the collection against
        whatever the ground-truth file holds, and nothing outside the binary
        checked that those were the same corpus. `impossible_scores` counts
        returned points the fp64 rescorer puts *better* than the k-th truth
        while the truth does not hold them, which an exhaustive scan cannot
        produce: stale ground truth, or a collection rebuilt from other data.
        It was printed to the console and then joined onto throughput rows
        anyway.
        """
        rc = self.m["recall"]
        p = rc.recall_path("lbl", "sift1m", "bench2")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"collection": "bench2", "points": [
            {"ef": 128, "recall_at_10": 0.99, "impossible_scores": 0},
            {"ef": 256, "recall_at_10": 0.99, "impossible_scores": 4},
        ]}))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertFalse(rc.stamp(p, "sift1m", "bench2", "euclid"))
        self.assertFalse(p.exists())
        self.assertTrue(p.with_suffix(".mismatch.json").exists())
        self.assertIn("impossible_scores=4", buf.getvalue())

    def test_ids_that_are_not_base_rows_are_not_joined(self):
        """The same refusal on the counter the JSON did not used to carry.

        `unknown_ids` reaches the file only from 2026-09-01, so the reader has
        to enforce this too: `stamp` never saw the legacy names, and a file
        written before the check existed is still on disk.
        """
        rc = self.m["recall"]
        p = rc.recall_path("lbl", "sift1m", "bench2")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"collection": "bench2", "dataset": "sift1m",
                                 "oversampling": None, "rescore": None,
                                 "points": [{"ef": 128, "recall_at_10": 0.99,
                                             "unknown_ids": 1}]}))
        self.assertEqual(rc.load_recall_json("lbl", "sift1m", "bench2"), {})
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench2"), {})

    def test_a_clean_sweep_still_joins(self):
        """Zero counters, and counters a older binary never wrote, are clean."""
        rc = self.m["recall"]
        self.assertIsNone(rc.corpus_disagreement(
            {"points": [{"ef": 128, "impossible_scores": 0, "unknown_ids": 0}]}))
        self.assertIsNone(rc.corpus_disagreement({"points": [{"ef": 128}]}))
        p = rc.recall_path("lbl", "sift1m", "bench2")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"collection": "bench2", "points": [
            {"ef": 128, "recall_at_10": 0.99, "impossible_scores": 0,
             "unknown_ids": 0}]}))
        self.assertTrue(rc.stamp(p, "sift1m", "bench2", "euclid"))
        self.assertEqual(
            rc.load_recall("lbl", "sift1m", "bench2")[128]["recall_at_10"], 0.99)

    def test_two_grades_of_one_collection_do_not_collide(self):
        """docs/workloads.md W12 point 2 puts two selectivity grades on one
        collection, so `bench12` has two sweeps and a row must join its own.

        The writer's key is the reader's key, the same argument the
        oversampling case makes: any key left out of the name lets the second
        sweep overwrite the first and then be joined onto the rows of both.
        """
        rc = self.m["recall"]
        self.assertNotEqual(rc.recall_path("l", "sift1m", "bench12", grade="sel1"),
                            rc.recall_path("l", "sift1m", "bench12", grade="sel10"))
        for grade, recall in (("sel1", 0.91), ("sel10", 0.82)):
            q = rc.recall_path("lbl", "sift1m", "bench12", grade=grade)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(json.dumps({"collection": "bench12", "points": [
                {"ef": 128, "recall_at_10": recall, "n_matching": 2000}]}))
            self.assertTrue(rc.stamp(q, "sift1m", "bench12", "euclid", grade=grade))
        self.assertEqual(
            rc.load_recall("lbl", "sift1m", "bench12", grade="sel1")[128]["recall_at_10"],
            0.91)
        self.assertEqual(
            rc.load_recall("lbl", "sift1m", "bench12", grade="sel10")[128]["recall_at_10"],
            0.82)
        # And neither answers for an unfiltered row: an unfiltered search of
        # `bench12` is a different question from either grade.
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench12"), {})

    def test_the_grade_a_row_joins_comes_off_its_id(self):
        rc = self.m["recall"]
        self.assertEqual(rc.grade_of("W12-sel1"), "sel1")
        self.assertEqual(rc.grade_of("W12-sel10"), "sel10")
        # Every other row is unfiltered, including ones with a suffix that is
        # not a grade.
        for wid in ("W3", "W10-ef128", "W11-steady", "W6-ef32", "W12-upload"):
            self.assertIsNone(rc.grade_of(wid), wid)

    def test_headline_and_sift_paths_differ(self):
        rc = self.m["recall"]
        self.assertNotEqual(rc.recall_path("l", "sift1m", "bench2"),
                            rc.recall_path("l", "dbpedia-openai-1m", "dbpedia"))


    def test_two_oversampling_sweeps_of_one_collection_do_not_collide(self):
        """The writer's key must be the reader's key.

        `load_recall_json` has always skipped a sweep whose quantization
        parameters differ from the row's, so it keys on five things; the path
        keyed on three, and a second sweep of one collection at another
        oversampling silently replaced the first. Every row then failed its
        recall join against whichever file survived.

        Found sweeping `bench6` at oversampling none/2/4/8 to
        settle findings 42: each run overwrote the last and the results had to
        be copied aside by hand between sweeps. A standard `fullrun` never hits
        it -- §4 gives each collection one parameter set -- so it bites exactly
        the repeated-sweep experiment findings 42 and `validation.md` item 6
        both call for.
        """
        rc = self.m["recall"]
        base = rc.recall_path("lbl", "dbpedia-openai-1m", "bench6")
        ov2 = rc.recall_path("lbl", "dbpedia-openai-1m", "bench6",
                             oversampling=2.0, rescore=True)
        ov4 = rc.recall_path("lbl", "dbpedia-openai-1m", "bench6",
                             oversampling=4.0, rescore=True)
        self.assertNotEqual(base, ov2)
        self.assertNotEqual(ov2, ov4)
        # `%g`, so 4.0 and 4 name one file rather than two.
        self.assertEqual(ov4, rc.recall_path("lbl", "dbpedia-openai-1m", "bench6",
                                             oversampling=4, rescore=True))

        base.parent.mkdir(parents=True)
        for path, ov, r10 in ((base, None, 0.88), (ov2, 2.0, 0.97), (ov4, 4.0, 0.99)):
            path.write_text(json.dumps({"collection": "bench6", "points": [
                {"ef": 128, "exact": False, "recall_at_10": r10}]}))
            self.assertTrue(rc.stamp(path, "dbpedia-openai-1m", "bench6", "cosine",
                                     ov, None if ov is None else True))

        # All three survive, and each answers only for its own parameters.
        self.assertEqual(rc.load_recall("lbl", "dbpedia-openai-1m", "bench6")[128]
                         ["recall_at_10"], 0.88)
        self.assertEqual(rc.load_recall("lbl", "dbpedia-openai-1m", "bench6", 2.0, True)[128]
                         ["recall_at_10"], 0.97)
        self.assertEqual(rc.load_recall("lbl", "dbpedia-openai-1m", "bench6", 4.0, True)[128]
                         ["recall_at_10"], 0.99)

    def test_a_sweep_written_before_the_params_were_in_the_name_is_still_read(self):
        """Files on disk from before this change keep working.

        The unsuffixed name stays a candidate, and every candidate is still
        checked against its own recorded fields -- so a legacy file answers for
        the parameters it recorded and for no others.
        """
        rc = self.m["recall"]
        legacy = rc.recall_path("lbl", "sift1m", "bench7")
        legacy.parent.mkdir(parents=True)
        legacy.write_text(json.dumps({"collection": "bench7", "points": [
            {"ef": 128, "exact": False, "recall_at_10": 0.91}]}))
        self.assertTrue(rc.stamp(legacy, "sift1m", "bench7", "euclid", 4.0, True))
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 4.0, True)[128]
                         ["recall_at_10"], 0.91)
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 2.0, True), {})

    def test_stamp_records_quantization_params_and_reader_keys_on_them(self):
        rc = self.m["recall"]
        p = rc.recall_path("lbl", "sift1m", "bench7")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"collection": "bench7", "points": [
            {"ef": 128, "exact": False, "recall_at_10": 0.9}]}))
        self.assertTrue(rc.stamp(p, "sift1m", "bench7", "euclid", 4.0, True))
        d = json.loads(p.read_text())
        self.assertEqual((d["oversampling"], d["rescore"]), (4.0, True))
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 4.0, True)[128]["recall_at_10"], 0.9)
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7"), {})
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 1.0, True), {})
        # A sweep that sent nothing (or predates the field) serves rows that sent nothing.
        p.write_text(json.dumps({"collection": "bench7", "dataset": "sift1m", "points": [
            {"ef": 128, "exact": False, "recall_at_10": 0.5}]}))
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7")[128]["recall_at_10"], 0.5)
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 4.0, True), {})

    def test_reader_prefers_the_binarys_own_quantization_fields(self):
        rc = self.m["recall"]
        p = rc.recall_path("lbl", "sift1m", "bench7")
        p.parent.mkdir(parents=True)
        pts = [{"ef": 128, "exact": False, "recall_at_10": 0.9}]
        # The binary's own record, no harness stamp: joins at what it sent.
        p.write_text(json.dumps({"collection": "bench7", "dataset": "sift1m", "points": pts,
                                 "quantization_oversampling": 4.0,
                                 "quantization_rescore": True}))
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 4.0, True)[128]["recall_at_10"], 0.9)
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7"), {})
        # Stamp and binary disagree: the file speaks for no row, under either key.
        p.write_text(json.dumps({"collection": "bench7", "dataset": "sift1m", "points": pts,
                                 "quantization_oversampling": 4.0, "quantization_rescore": True,
                                 "oversampling": 1.0, "rescore": True}))
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 4.0, True), {})
        self.assertEqual(rc.load_recall("lbl", "sift1m", "bench7", 1.0, True), {})
        self.assertIsNone(rc.sweep_quant_params(json.loads(p.read_text())))
        # Stamp only (a file from before the binary recorded them): the stamp.
        self.assertEqual(rc.sweep_quant_params({"oversampling": 2.0, "rescore": False}),
                         (2.0, False))

    def test_sweep_sends_each_collections_own_params(self):
        rc = self.m["recall"]
        self.assertEqual(rc.quant_params_of("bench7"), (4.0, True))
        self.assertEqual(rc.quant_params_of("bench6"), (None, True))
        self.assertEqual(rc.quant_params_of("bench2"), (None, None))
        # The relevance command line carries them, spelled as bfb spells them.
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            out = Path(cmd[cmd.index("--json") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"collection": cmd[cmd.index("--collection") + 1],
                                       "points": []}))
            return type("R", (), {"returncode": 0})()

        with mock.patch("subprocess.run", fake_run):
            self.assertEqual(rc.sweep("http://e", "lbl", "bench7", 10, 4.0, True), 0)
            cmd = seen["cmd"]
            self.assertEqual(cmd[cmd.index("--quantization-oversampling") + 1], "4.0")
            self.assertEqual(cmd[cmd.index("--quantization-rescore") + 1], "true")
            # The parameterised name: the sweep sent 4.0/true, so that is where
            # it belongs and where a reader asking for 4.0/true will look.
            d = json.loads(rc.recall_path("lbl", "sift1m", "bench7",
                                          oversampling=4.0, rescore=True).read_text())
            self.assertEqual((d["oversampling"], d["rescore"]), (4.0, True))
            self.assertEqual(rc.sweep("http://e", "lbl", "bench2", 10), 0)
            self.assertNotIn("--quantization-oversampling", seen["cmd"])
            self.assertNotIn("--quantization-rescore", seen["cmd"])
            # `main` with only `--rescore` keeps bench7's default oversampling.
            with mock.patch.object(rc, "sweep", wraps=rc.sweep) as sw, \
                    mock.patch.object(rc.Path, "exists", return_value=True):
                self.assertEqual(rc.main(["recall.py", "lbl", "--collections", "bench7",
                                          "--rescore", "false"]), 0)
            self.assertEqual(sw.call_args.args[4:6], (4.0, False))
            cmd = seen["cmd"]
            self.assertEqual(cmd[cmd.index("--quantization-oversampling") + 1], "4.0")
            self.assertEqual(cmd[cmd.index("--quantization-rescore") + 1], "false")


class NoiseTests(unittest.TestCase):
    def test_one_definition(self):
        reg = _reload(Path(tempfile.gettempdir()))["regression"]
        self.assertAlmostEqual(reg.noise_band(0.011), 0.033)
        self.assertAlmostEqual(reg.noise_band(0.001), 0.02)   # floored
        self.assertAlmostEqual(reg.noise_band(0.011, arms=2), 3 * 2 ** 0.5 * 0.011)

    def _rep(self, root: Path, name: str, rows: list[dict], stamp: dict | None = "default"):
        d = root / "bench/results" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "rows.json").write_text(json.dumps(rows))
        if stamp is not None:
            (d / "run.json").write_text(json.dumps(
                {"harness": good_stamp() if stamp == "default" else stamp}))
        return d

    def test_a_folded_floor_carries_the_stamp_its_readers_look_for(self):
        """`aggregate.py` writes the one floor measured on *this* binary.

        It wrote `rsd` and `reps` and nothing else, so a floor folded from a
        run's own repetitions could not say which harness configuration's
        spread it was, and `report.noise_provenance` fell to its "carries no
        harness stamp" branch — the generic caveat instead of the specific one.
        That is the gap the floor had until it was re-derived, in
        the one path that would have put it straight back.
        """
        import importlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _reload(root)
            agg = importlib.reload(importlib.import_module("aggregate"))
            rows = lambda q: [_row("W3", q), _row("W4", q * 2)]
            for i, q in enumerate((100.0, 110.0, 105.0), start=1):
                self._rep(root, f"lbl-rep{i}", rows(q))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(agg.main(["aggregate.py", "lbl",
                                           "lbl-rep1", "lbl-rep2", "lbl-rep3"]), 0)
            floor = json.loads((root / "bench/results/lbl/noise.json").read_text())
            self.assertEqual(floor["harness_hash"], stamp_hash(good_stamp()))
            self.assertEqual(floor["n_dirs"], 3)
            self.assertEqual(floor["measured_on"], "2026-01-01")
            self.assertIn("W3", floor["rsd"])

    def test_a_monotone_slide_is_folded_as_drift_and_not_as_spread(self):
        """`rep_rsd` describes passes as draws from one distribution; a slide
        in one direction is not that.

        strawmANN's W10-ef128 on dbpedia-openai-1m read 3,662, 3,001, 2,764
        qps in a single run at a constant clock and a constant instruction
        count per query. Folded as `rsd 14.8%` it produced a ±63% parity band
        and the report called a real 1.18x difference "no measured
        difference".
        """
        import importlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _reload(root)
            agg = importlib.reload(importlib.import_module("aggregate"))
            # W3 slides one way; W4 wobbles by more in total but not monotonically.
            for i, (slide, wobble) in enumerate(
                    ((3662.0, 3555.0), (3001.0, 3502.0), (2764.0, 3512.0)), start=1):
                self._rep(root, f"lbl-rep{i}", [_row("W3", slide), _row("W4", wobble)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(agg.main(["aggregate.py", "lbl",
                                           "lbl-rep1", "lbl-rep2", "lbl-rep3"]), 0)
            rows = {r["id"]: r for r in
                    json.loads((root / "bench/results/lbl/rows.json").read_text())}
            self.assertAlmostEqual(rows["W3"]["rep_drift"], -0.2452, places=3)
            # Not every spread is a trend: the wobbling row carries none.
            self.assertIsNone(rows["W4"].get("rep_drift"))
            # Nor is every trend a drift. Three exchangeable draws sort
            # themselves one time in three, so a couple of percent in one
            # direction is ordering luck and must not refuse a ratio.
            agg2 = importlib.reload(importlib.import_module("aggregate"))
            self.assertIsNone(agg2._rep_drift([1000.0, 1010.0, 1020.0]))
            self.assertIsNotNone(agg2._rep_drift([1000.0, 900.0, 800.0]))
            # And the median is still published either way.
            self.assertEqual(rows["W3"]["qps"], 3001.0)

    def test_a_folded_floor_says_mixed_when_the_passes_disagree(self):
        import importlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _reload(root)
            agg = importlib.reload(importlib.import_module("aggregate"))
            for i, q in enumerate((100.0, 110.0, 105.0), start=1):
                self._rep(root, f"lbl-rep{i}", [_row("W3", q)],
                          stamp=good_stamp(metric="Cosine") if i == 3 else "default")
            with contextlib.redirect_stdout(io.StringIO()):
                agg.main(["aggregate.py", "lbl", "lbl-rep1", "lbl-rep2", "lbl-rep3"])
            floor = json.loads((root / "bench/results/lbl/noise.json").read_text())
            self.assertEqual(floor["harness_hash"], "mixed")

    def test_measure_noise_accepts_label_rep_dirs_and_stamps_the_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = _reload(root)["regression"]
            for i in range(1, 4):
                self._rep(root, f"qd-rep{i}", [_row("W3", 1000 + i)])
            self._rep(root, "other-label", [_row("W3", 5)])
            # `<label>-rep<n>` under a directory, and as the glob qdrant_ab prints.
            found = reg.rep_dirs(root / "bench/results")
            self.assertEqual([d.name for d in found], ["qd-rep1", "qd-rep2", "qd-rep3"])
            found = reg.rep_dirs(root / "bench/results/qd-rep*")
            self.assertEqual(len(found), 3)
            n = reg.measure_noise(found, "x")
            self.assertEqual(n.reps["W3"], 3)
            self.assertEqual(n.harness_hash, stamp_hash(good_stamp()))
            self.assertEqual(n.n_dirs, 3)
            # Reps under two harness stamps: the floor says so instead of picking one.
            self._rep(root, "qd-rep3", [_row("W3", 1003)], good_stamp(metric="Cosine"))
            self.assertEqual(reg.measure_noise(found, "x").harness_hash, "mixed")
            self.assertEqual(reg.floor_stamp_notes(reg.Noise({"W3": 0.01}, {}, "s", [], "mixed"),
                                                   ["qd-rep1"])[0][:34],
                             "noise floor was measured over repe")
            # And a floor stamped differently from the compared rows is noted.
            notes = reg.floor_stamp_notes(
                reg.Noise({"W3": 0.01}, {}, "s", [], stamp_hash(good_stamp(metric="Cosine"))),
                ["qd-rep1"])
            self.assertTrue(any("different harness stamp than qd-rep1" in x for x in notes), notes)
            self.assertEqual(reg.floor_stamp_notes(
                reg.Noise({"W3": 0.01}, {}, "s", [], stamp_hash(good_stamp())), ["qd-rep1"]), [])
            # `cmd_measure` writes the stamp into noise.json.
            import contextlib
            import io
            out = root / "noise.json"
            args = type("A", (), {"measure_noise": str(root / "bench/results/qd-rep*"),
                                  "out": str(out)})()
            with contextlib.redirect_stdout(io.StringIO()):
                reg.cmd_measure(args)
            self.assertIn("harness_hash", json.loads(out.read_text()))

    def test_contaminated_row_is_not_attributable_and_not_tallied(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = _reload(root)["regression"]
            self._rep(root, "old", [_row("A", 1000.0), _row("B", 1000.0)])
            # B halved under a compiler: not a regression of the candidate.
            self._rep(root, "new", [_row("A", 1040.0), _row("B", 500.0, foreign="rustc(100%)")])
            noise = root / "noise.json"
            noise.write_text(json.dumps({"rsd": {"A": 0.05, "B": 0.05}, "reps": {"A": 5, "B": 5},
                                         "source": "synthetic", "discarded": []}))
            args = type("A", (), {"baseline": "old", "candidate": "new", "noise": str(noise)})()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = reg.cmd_compare(args)
            out = buf.getvalue()
            self.assertEqual(rc, 0, out)
            self.assertIn("0 regression(s)", out)
            self.assertIn("1 row(s) not attributable", out)
            b_line = next(l for l in out.splitlines() if l.strip().startswith("B "))
            self.assertIn("not attributable: contaminated", b_line)
            self.assertNotIn("REGRESSION", b_line)
            # An unstamped floor against stamped rows is noted, not refused.
            self.assertIn("carries no harness stamp", out)
            self.assertIn("inconclusive", next(l for l in out.splitlines()
                                               if l.strip().startswith("A ")))

    def test_an_open_loop_row_re_offered_at_another_rate_is_not_attributable(self):
        """`--rps-reference` sets an open-loop row's offered rate, and its qps is
        that rate rather than a capacity it reached, so two runs given different
        references differ by exactly the ratio of the references.

        Measured on a sift1m pair re-run against a reference of
        10,364 instead of 9,939 reported W4-sat50, W4-sat70 and W4-sat90 as
        `improvement (4.3%)` for *both* engines. The noise floor on those rows is
        0.00% — bfb delivers the pinned rate to three decimals — which is what
        makes the arithmetic exact and the verdict entirely an artefact of the
        instrument. Six green findings, none of them about either engine.
        """
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = _reload(root)["regression"]
            sat = {"load_mode": "open-loop"}
            self._rep(root, "old", [_row("W4-sat50", 4970.0, rps_target=4970, **sat),
                                    _row("W3", 1000.0)])
            self._rep(root, "new", [_row("W4-sat50", 5182.0, rps_target=5182, **sat),
                                    _row("W3", 1040.0)])
            noise = root / "noise.json"
            noise.write_text(json.dumps({"rsd": {"W4-sat50": 0.0, "W3": 0.05},
                                         "reps": {"W4-sat50": 3, "W3": 3},
                                         "source": "synthetic", "discarded": []}))
            args = type("A", (), {"baseline": "old", "candidate": "new",
                                  "noise": str(noise)})()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = reg.cmd_compare(args)
            out = buf.getvalue()
            self.assertEqual(rc, 0, out)
            self.assertIn("0 improvement(s)", out)
            self.assertIn("1 open-loop row(s) not attributable", out)
            line = next(x for x in out.splitlines() if x.strip().startswith("W4-sat50 "))
            self.assertIn("offered rate changed", line)
            self.assertNotIn("improvement", line)

    def test_an_open_loop_row_at_the_same_rate_still_gets_a_verdict(self):
        """The exemption is the reference *moving*, not the row being open-loop:
        two runs at one offered rate are comparable, and a row that cannot ever
        be called a regression would be worse than the false improvement."""
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = _reload(root)["regression"]
            sat = {"load_mode": "open-loop", "rps_target": 4970}
            self._rep(root, "old", [_row("W4-sat50", 4970.0, **sat)])
            # The rate was offered and not served: bfb asked for 4,970 and the
            # engine returned 3,000, which is the failure this row exists for.
            self._rep(root, "new", [_row("W4-sat50", 3000.0, **sat)])
            noise = root / "noise.json"
            noise.write_text(json.dumps({"rsd": {"W4-sat50": 0.01},
                                         "reps": {"W4-sat50": 3},
                                         "source": "synthetic", "discarded": []}))
            args = type("A", (), {"baseline": "old", "candidate": "new",
                                  "noise": str(noise)})()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = reg.cmd_compare(args)
            out = buf.getvalue()
            self.assertEqual(rc, 1, out)
            self.assertIn("1 regression(s)", out)
            self.assertNotIn("not attributable", out)


class BandwidthTests(unittest.TestCase):
    """The machine's bandwidth is recorded from the engine and shown once."""

    def test_banner_and_probe_parse(self):
        prov = _reload(Path(tempfile.gettempdir()))["provenance"]
        b = prov.parse_bandwidth_banner(
            "  memory bandwidth       : 47.3 GB/s aggregate (24 threads) · "
            "25.1 GB/s single core (53% of bus)\n")
        self.assertEqual(b, {"aggregate_gbps": 47.3, "threads": 24,
                             "single_core_gbps": 25.1, "single_core_pct_of_bus": 53})
        p = prov.parse_bandwidth_probe(
            "cores                  : 24\nbandwidth aggregate    : 70.9 GB/s (24 threads)\n"
            "bandwidth single core  : 39.3 GB/s (55% of bus)\n")
        self.assertEqual(p["aggregate_gbps"], 70.9)
        self.assertEqual(p["single_core_pct_of_bus"], 55)
        self.assertIsNone(prov.parse_bandwidth_banner("no such line"))

    def test_startup_banner_preferred_over_probe(self):
        prov = _reload(Path(tempfile.gettempdir()))["provenance"]
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "server.log"
            log.write_text("strawmann 0.1\n  memory bandwidth       : 47.3 GB/s aggregate "
                           "(24 threads) · 25.1 GB/s single core (53% of bus)\n")
            b = prov.memory_bandwidth(server_log=log, binary=Path(d) / "absent")
        self.assertEqual(b["source"], "strawmann startup banner")
        self.assertEqual(b["aggregate_gbps"], 47.3)
        # No log and no binary: honestly nothing, not a guess.
        self.assertIsNone(prov.memory_bandwidth(server_log=None,
                                                binary=Path(d) / "absent"))


    def test_probe_runs_with_the_whole_machine_and_says_so(self):
        # Under `fullrun` the harness sits under `taskset -c <client cpus>`;
        # the probe must not inherit that, or the "machine" bandwidth is the
        # client cpuset's.
        prov = _reload(Path(tempfile.gettempdir()))["provenance"]
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"], seen["kw"] = cmd, kw
            return type("R", (), {"stdout": "cores                  : 24\n"
                                            "bandwidth aggregate    : 70.9 GB/s (24 threads)\n"
                                            "bandwidth single core  : 39.3 GB/s (55% of bus)\n",
                                  "stderr": ""})()

        # The affinity calls are faked: the test process is never re-pinned,
        # and the outcome does not depend on the machine (or container) it
        # runs in. `aff` is the pretend kernel state.
        ncpu = os.cpu_count()
        aff = {"cpus": {2, 3}, "fail": False, "sets": []}

        def fake_set(pid, cpus):
            aff["sets"].append(set(cpus))
            if aff["fail"]:
                raise OSError(22, "Invalid argument")
            aff["cpus"] = set(cpus)

        with tempfile.TemporaryDirectory() as d, mock.patch("subprocess.run", fake_run), \
                mock.patch("os.sched_getaffinity", lambda pid: set(aff["cpus"])), \
                mock.patch("os.sched_setaffinity", fake_set):
            binary = Path(d) / "strawmann"
            binary.write_text("")
            bw = prov.memory_bandwidth(server_log=None, binary=binary)
            self.assertEqual(seen["cmd"], [str(binary), "--probe"])
            # Widened to every CPU for the probe, restored to {2,3} after.
            self.assertEqual(aff["sets"], [set(range(ncpu)), {2, 3}])
            self.assertEqual(aff["cpus"], {2, 3})
            self.assertEqual(bw["cpuset"], f"0-{ncpu - 1}")
            self.assertNotIn("affinity_error", bw)
            self.assertEqual(bw["threads"], 24)
            self.assertEqual(bw["source"], "strawmann --probe at run start")
            # A cpuset-limited container refuses the widening: the record
            # holds the affinity actually in effect and says why.
            aff["fail"] = True
            bw = prov.memory_bandwidth(server_log=None, binary=binary)
            self.assertEqual(bw["cpuset"], "2-3")
            self.assertIn("Invalid argument", bw["affinity_error"])
        # And the spelling of a real affinity, without touching it.
        self.assertRegex(prov.probe_cpuset(), r"^\d+(-\d+)?(,\d+(-\d+)?)*$")

    def test_isa_build_from_banner_then_probe(self):
        prov = _reload(Path(tempfile.gettempdir()))["provenance"]
        self.assertEqual(prov.parse_isa_build("  isa build              : avx512\n"), "avx512")
        self.assertIsNone(prov.parse_isa_build("no such line"))
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "server.log"
            log.write_text("strawmann 0.1\n  isa build              : avx2\n")
            binary = Path(d) / "strawmann"
            binary.write_text("")
            calls = []

            def fake_run(cmd, **kw):
                calls.append(cmd)
                return type("R", (), {"stdout": "isa build              : baseline\n", "stderr": ""})()

            with mock.patch("subprocess.run", fake_run):
                # The banner wins and the probe is not run.
                self.assertEqual(prov.isa_build(server_log=log, binary=binary), "avx2")
                self.assertEqual(calls, [])
                # No banner: ask the binary that served.
                self.assertEqual(prov.isa_build(server_log=Path(d) / "absent", binary=binary),
                                 "baseline")
                self.assertEqual(calls, [[str(binary), "--probe"]])
                # Nothing to ask: unknown, not "native".
                self.assertIsNone(prov.isa_build(server_log=None, binary=Path(d) / "absent"))
                # A binary `memory_bandwidth` already probed is not probed twice.
                prov._PROBE_TEXT[str(binary)] = "isa build              : avx512\n"
                self.assertEqual(prov.isa_build(server_log=None, binary=binary), "avx512")
                self.assertEqual(len(calls), 1)


class ProvenanceTests(unittest.TestCase):
    def test_probe_asks_the_serving_image_not_the_path(self):
        prov = _reload(Path(tempfile.gettempdir()))["provenance"]
        # With a pid the binary was hashed through /proc; the probe goes there
        # too, so a rebuilt zig-out/bin/strawmann cannot answer for the server.
        build = {"binary": "/x/zig-out/bin/strawmann", "binary_source": "/proc/4242/exe",
                 "binary_sha256": "abcd"}
        self.assertEqual(prov.probe_path_of(build), Path("/proc/4242/exe"))
        # No pid: the default path is all there is, and the record says so.
        build = {"binary": "/x/zig-out/bin/strawmann",
                 "binary_source": "zig-out/bin/strawmann (default build path; no engine pid)"}
        self.assertEqual(prov.probe_path_of(build), Path("/x/zig-out/bin/strawmann"))
        self.assertIsNone(prov.probe_path_of({"binary": None, "binary_source": "x"}))

    def test_collect_reads_the_server_log_from_the_results_dir(self):
        prov = _reload(Path(tempfile.gettempdir()))["provenance"]
        seen = {}

        def fake_bw(server_log=None, binary=None):
            seen["log"] = server_log
            return None

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(prov, "memory_bandwidth", fake_bw), \
                mock.patch.object(prov, "qdrant_build", lambda pid: {}), \
                mock.patch.object(prov, "strawmann_build", lambda pid: {}):
            prov.collect("arm-rep1", "http://l", None, "strawmann", results=Path(d) / "out")
            self.assertEqual(seen["log"], Path(d) / "out/server.log")
            # RESULTS_DIR is how isa_sweep / qdrant_ab redirect a run.
            with mock.patch.dict(os.environ, {"RESULTS_DIR": str(Path(d) / "env")}):
                prov.collect("arm-rep1", "http://l", None, "strawmann")
            self.assertEqual(seen["log"], Path(d) / "env/server.log")


class IsaSweepTests(unittest.TestCase):
    def test_table_survives_a_single_rep_row(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _reload(root)
            isa = importlib.reload(importlib.import_module("isa_sweep"))
            for name, rows in (("baseline-rep1", {"W3": 100.0, "W13": 50.0}),
                               ("baseline-rep2", {"W13": 52.0}),
                               ("avx2-rep1", {"W3": 110.0, "W13": 51.0})):
                d = isa.RESULTS / name
                d.mkdir(parents=True)
                (d / "rows.json").write_text(json.dumps(
                    [{"id": k, "qps": v, "status": "ok"} for k, v in rows.items()]))
            args = type("A", (), {"rows": ["W3", "W13"], "vs": "baseline"})()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = isa.cmd_table(args)   # W3 has one rep per arm: no spread to report
            self.assertEqual(rc, 0)
            self.assertIn("W3         n/a", buf.getvalue())
            self.assertIn("W13", buf.getvalue())


class ResultsSinkTests(unittest.TestCase):
    """§8's gate on build identity: commit and Qdrant version, not only ISA."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _reload(Path(self.tmp.name))
        self.res = importlib.reload(importlib.import_module("results"))

    def tearDown(self):
        self.tmp.cleanup()

    #: A fixture read by several tests and never mutated; `ClassVar` says so,
    #: since a mutable class attribute one test edits is a test that passes
    #: alone and fails in a suite.
    CONF: ClassVar[dict] = {"hash": "c1", "dataset": "sift1m", "metric": "Euclid", "dim": 128,
            "qdrant_version": "1.19.0", "strawmann_commit": "abcdef123456-dirty",
            "isa_build": "avx512", "tier_reached": "T1", "licenses_perf": True,
            "base_checksum": "7bf7", "query_checksum": "1eb6", "detail": ""}

    def test_insert_perf_refuses_mismatched_commit_and_version(self):
        res = self.res
        db = res.connect(":memory:")
        res.insert_conformance(db, self.CONF)
        good = {"workload": "W0", "engine": "strawmann", "dataset": "sift1m",
                "isa_build": "avx512", "env_hash": "e", "conformance_hash": "c1", "qps": 1.0,
                "strawmann_commit": "abcdef123456-dirty"}
        res.insert_perf(db, good)
        with self.assertRaisesRegex(res.RejectedRow, "strawmann commit 'abcdef123456'"):
            res.insert_perf(db, {**good, "strawmann_commit": "abcdef123456"})
        with self.assertRaisesRegex(res.RejectedRow, "no build identity"):
            res.insert_perf(db, {**good, "strawmann_commit": None})
        with self.assertRaisesRegex(res.RejectedRow, "Qdrant '1.18.0'"):
            res.insert_perf(db, {**good, "strawmann_commit": None, "qdrant_version": "1.18.0"})
        res.insert_perf(db, {**good, "strawmann_commit": None, "qdrant_version": "1.19.0",
                             "engine": "qdrant"})
        self.assertEqual(db.execute("SELECT count(*) FROM perf").fetchone()[0], 2)

    def _label(self, name: str, meta: dict, conf: dict) -> Path:
        d = Path(self.tmp.name) / "bench/results" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "rows.json").write_text(json.dumps([_row("W0", 100.0, ef=None, collection="bench0")]))
        (d / "run.json").write_text(json.dumps(meta))
        (d / "conformance.json").write_text(json.dumps(conf))
        return d

    def test_a_filtered_row_is_ingested_against_the_sweep_of_its_own_grade(self):
        """The sink has its own recall lookup, separate from
        `compare.recall_at`, and it keyed on (collection, quantization params)
        alone.

        So a filtered row asked for an *unfiltered* sweep of `bench12`, which
        is never written — only graded ones are — got nothing, and was refused
        for having no recall. Measured on sm-sift-perf-rel-0908: 22 rows
        recorded and 4 refused, of which both W12 grades were the feature
        working everywhere except the last join. The report was unaffected,
        because `compare.py` had the grade; two consumers, one of them wrong,
        is exactly the shape a test has to pin.
        """
        import contextlib
        import io
        res = self.res
        root = Path(self.tmp.name)
        meta = {"dataset": {"name": "sift1m"}, "isa_build": "avx512",
                "strawmann": {"commit": "abcdef123456", "dirty": True}}
        d = self._label("arm", meta, self.CONF)
        (d / "rows.json").write_text(json.dumps([
            _row("W12-sel1", 2571.0, ef=128, collection="bench12"),
            _row("W12-sel10", 296.0, ef=128, collection="bench12")]))
        # One sweep per grade, as `recall.py` writes them, and nothing under
        # the unfiltered name.
        for grade, recall in (("sel1", 1.0), ("sel10", 0.99)):
            (d / f"recall.sift1m.bench12.{grade}.json").write_text(json.dumps({
                "collection": "bench12", "dataset": "sift1m", "grade": grade,
                "oversampling": None, "rescore": None,
                "points": [{"ef": 128, "recall_at_10": recall, "recall_at_1": 1.0,
                            "n_matching": 2000, "returned": 100, "asked": 100}]}))
        db = res.connect(":memory:")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = res.ingest_run(db, "arm", root)
        self.assertEqual(rc, 0, f"both grades should be licensed: {err.getvalue()}")
        got = dict(db.execute(
            "SELECT workload, recall_at_10 FROM perf WHERE workload LIKE 'W12%'"
        ).fetchall())
        self.assertEqual(got, {"W12-sel1": 1.0, "W12-sel10": 0.99},
                         "each grade joins its own sweep, not the other's")

    def test_ingest_names_every_refused_row_and_why(self):
        """The sink printed the first refusal per label and a hint to sweep the
        collection -- which for W11 is the collection the row had just mutated.
        It refused 2 + 3 rows and named two."""
        import contextlib
        import io
        res = self.res
        root = Path(self.tmp.name)
        meta = {"dataset": {"name": "sift1m"}, "isa_build": "avx512",
                "strawmann": {"commit": "abcdef123456", "dirty": True}}
        d = self._label("arm", meta, self.CONF)
        (d / "rows.json").write_text(json.dumps([
            _row("W11-steady", 3000.0, collection="bench2"),
            _row("W11", 1200.0, collection="bench2"),
            _row("W12", 26.0, collection="bench12")]))
        db = res.connect(":memory:")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(res.ingest_run(db, "arm", root), 1)
        out = err.getvalue()
        for wid in ("W11-steady", "W11", "W12"):
            self.assertIn(f"refused {wid}: {wid}: QPS given without recall@10", out)
        self.assertIn("W11 on bench2", out)
        self.assertIn("W12 on bench12", out)
        self.assertIn("refused by design", out)

    def test_ingest_binds_the_row_to_the_conformance_build(self):
        import contextlib
        import io
        res = self.res
        root = Path(self.tmp.name)
        meta = {"dataset": {"name": "sift1m"}, "isa_build": "avx512",
                "strawmann": {"commit": "abcdef123456", "dirty": True}}
        self._label("arm", meta, self.CONF)
        db = res.connect(":memory:")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(res.ingest_run(db, "arm", root), 0)
        self.assertEqual(db.execute("SELECT isa_build FROM perf").fetchall(), [("avx512",)])
        # Same rows under a conformance row from another commit: refused.
        self._label("arm2", meta, {**self.CONF, "hash": "c2", "strawmann_commit": "0000"})
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(res.ingest_run(db, "arm2", root), 1)
        self.assertIn("strawmann commit", err.getvalue())
        # A run.json without any build identity: refused, not filed as native.
        self._label("arm3", {"dataset": {"name": "sift1m"}, "isa_build": "avx512"}, self.CONF)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(res.ingest_run(db, "arm3", root), 1)
        self.assertIn("no build identity", err.getvalue())

    def test_migrate_keeps_the_index_and_db_path_is_root_relative(self):
        res = self.res
        # A pre-UNIQUE database: the rebuild renames, recreates and used to
        # leave the new table without `perf_by_workload`.
        db = res.sqlite3.connect(":memory:")
        old = res.SCHEMA.split("CREATE TABLE IF NOT EXISTS perf")[0]
        old += """CREATE TABLE IF NOT EXISTS perf (
            id INTEGER PRIMARY KEY AUTOINCREMENT, workload TEXT NOT NULL, engine TEXT NOT NULL,
            dataset TEXT NOT NULL, isa_build TEXT NOT NULL, env_hash TEXT NOT NULL,
            conformance_hash TEXT NOT NULL REFERENCES conformance(hash), qps REAL, p50_us REAL,
            p95_us REAL, p99_us REAL, p999_us REAL, recall_at_1 REAL, recall_at_10 REAL,
            recall_at_100 REAL, mrde REAL, ndcg_at_10 REAL, effective_freq REAL,
            pct_of_roofline REAL, load_mode TEXT NOT NULL, notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS perf_by_workload ON perf(workload, engine, isa_build);"""
        db.executescript(old)
        # Sanity: the old table has the index and no UNIQUE.
        self.assertIn("perf_by_workload", {r[1] for r in db.execute("PRAGMA index_list(perf)")})
        res._migrate(db)
        names = {r[1] for r in db.execute("PRAGMA index_list(perf)")}
        self.assertIn("perf_by_workload", names, names)
        self.assertIn("measured_at", {r[1] for r in db.execute("PRAGMA table_info(perf)")})
        root = Path("/repo")
        self.assertEqual(res.db_path("bench/results.sqlite", root), "/repo/bench/results.sqlite")
        self.assertEqual(res.db_path("/abs/x.sqlite", root), "/abs/x.sqlite")
        self.assertEqual(res.db_path(":memory:", root), ":memory:")

    def test_ingest_does_not_default_the_isa_build_and_refuses_mixed_builds(self):
        import contextlib
        import io
        res = self.res
        root = Path(self.tmp.name)
        # No isa_build anywhere for a strawmann arm: unknown, and refused,
        # rather than filed as `native` (`native` is what the conformance row says).
        meta = {"dataset": {"name": "sift1m"}, "strawmann": {"commit": "abcdef123456", "dirty": True}}
        self._label("noisa", meta, {**self.CONF, "isa_build": "native"})
        db = res.connect(":memory:")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(res.ingest_run(db, "noisa", root), 1)
        self.assertIn("ISA build 'None'", err.getvalue())
        # A Qdrant arm has no ISA build of its own; it files under the row that
        # licenses the pair, and is accepted.
        qmeta = {"dataset": {"name": "sift1m"}, "qdrant": {"version": "1.19.0"}}
        self._label("qd", qmeta, {**self.CONF, "isa_build": "native"})
        db = res.connect(":memory:")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(res.ingest_run(db, "qd", root), 0)
        self.assertEqual(db.execute("SELECT isa_build FROM perf").fetchall(), [("native",)])
        # The row's own stamp wins over the label's.
        meta = {"dataset": {"name": "sift1m"}, "isa_build": "avx512",
                "strawmann": {"commit": "abcdef123456", "dirty": True}}
        d = self._label("stamped", meta, self.CONF)
        (d / "rows.json").write_text(json.dumps([_row("W0", 100.0, ef=None, collection="bench0",
                                                     isa_build="avx2")]))
        db = res.connect(":memory:")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(res.ingest_run(db, "stamped", root), 1)
        self.assertIn("ISA build 'avx2'", err.getvalue())
        # Rows from two builds in one file: refused whole, like STALE.
        d = self._label("mixed", meta, self.CONF)
        (d / "rows.json").write_text(json.dumps([
            _row("W0", 100.0, ef=None, collection="bench0", engine_binary="1111", isa_build="avx512"),
            _row("W3", 100.0, engine_binary="2222", isa_build="avx512")]))
        db = res.connect(":memory:")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(res.ingest_run(db, "mixed", root), 1)
        self.assertIn("2 different builds", err.getvalue())
        self.assertEqual(db.execute("SELECT count(*) FROM perf").fetchone()[0], 0)

    def test_compare_picks_the_latest_pair_by_time_not_by_isa_name(self):
        import contextlib
        import io
        res = self.res
        db = res.connect(":memory:")
        res.insert_conformance(db, {**self.CONF, "tier_reached": "T3", "licenses_comparative": True})
        base = {"workload": "W3", "dataset": "sift1m", "env_hash": "e", "conformance_hash": "c1",
                "strawmann_commit": "abcdef123456-dirty", "recall_at_10": 0.98}
        # The pair is chosen by measurement time, explicitly, not by which row
        # the SELECT's `ORDER BY engine, isa_build, ...` happens to yield last.
        # The rows are put in with the ordering that used to matter scrambled:
        # a later `measured_at` on a smaller rowid, and the SELECT's ordering
        # keys equal otherwise.
        res.insert_perf(db, {**base, "engine": "strawmann", "isa_build": "avx512", "qps": 3000.0,
                             "measured_at": "2026-02-01T00:00:00Z"})
        res.insert_perf(db, {**base, "engine": "qdrant", "isa_build": "avx512", "qps": 1000.0,
                             "strawmann_commit": None, "qdrant_version": "1.19.0",
                             "measured_at": "2026-02-01T00:00:00Z"})
        res.insert_perf(db, {**base, "engine": "strawmann", "isa_build": "avx512", "qps": 6000.0,
                             "measured_at": "2026-01-01T00:00:00Z"})
        # Bypass the sink's ISA check to plant the alphabetically-last ISA
        # build's row: an old measurement filed under `zzz`.
        db.execute("UPDATE perf SET isa_build='zzz' WHERE qps=6000.0")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res.compare(db, "W3")
        # 3000/1000, the February pair; the January 6000 under `zzz` is older
        # whatever order the rows arrive in.
        self.assertIn("3.00x", buf.getvalue())

    def test_a_zero_qps_arm_is_refused_rather_than_divided_by(self):
        """A row that served nothing is not a measurement, and `qps` is only
        ever filtered for `None` on the way in."""
        import contextlib
        import io
        res = self.res
        db = res.connect(":memory:")
        res.insert_conformance(db, {**self.CONF, "tier_reached": "T3", "licenses_comparative": True})
        base = {"workload": "W3", "dataset": "sift1m", "env_hash": "e", "conformance_hash": "c1",
                "strawmann_commit": "abcdef123456-dirty", "recall_at_10": 0.98,
                "measured_at": "2026-02-01T00:00:00Z"}
        res.insert_perf(db, {**base, "engine": "strawmann", "isa_build": "avx512", "qps": 3000.0})
        res.insert_perf(db, {**base, "engine": "qdrant", "isa_build": "avx512", "qps": 0.0,
                             "strawmann_commit": None, "qdrant_version": "1.19.0"})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res.compare(db, "W3")
        self.assertIn("NOT COMPARABLE", buf.getvalue())
        self.assertNotIn("x qdrant", buf.getvalue())

    def test_ingest_joins_w7_to_its_own_oversampling_sweep(self):
        """The sink joins a quantized row to a sweep at the row's params, not bench2's."""
        import contextlib
        import io
        res = self.res
        root = Path(self.tmp.name)
        meta = {"dataset": {"name": "sift1m"}, "isa_build": "avx512",
                "strawmann": {"commit": "abcdef123456", "dirty": True}}
        d = self._label("w7", meta, self.CONF)
        rows = [_row("W7", 500.0, ef=128, collection="bench7",
                     quantization_oversampling=4.0, quantization_rescore=True),
                _row("W6", 600.0, ef=128, collection="bench6",
                     quantization_oversampling=1.0, quantization_rescore=True)]
        (d / "rows.json").write_text(json.dumps(rows))
        # bench7 sweep recorded at oversampling 1 → must NOT join W7 (recall NULL);
        # bench6 sweep at 1 → joins W6.
        for coll, ovs, rec in (("bench7", 1.0, 0.5), ("bench6", 1.0, 0.9)):
            sw = _sweep(coll, "sift1m", rec)
            sw.update(oversampling=ovs, rescore=True)
            (d / f"recall.sift1m.{coll}.json").write_text(json.dumps(sw))
        db = res.connect(":memory:")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res.ingest_run(db, "w7", root)
        got = dict(db.execute("SELECT workload, recall_at_10 FROM perf").fetchall())
        self.assertEqual(got.get("W6"), 0.9)
        self.assertIsNone(got.get("W7"))
        # A bench7 sweep at oversampling 4 does join.
        sw = _sweep("bench7", "sift1m", 0.7)
        sw.update(oversampling=4.0, rescore=True)
        (d / "recall.sift1m.bench7.json").write_text(json.dumps(sw))
        db = res.connect(":memory:")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res.ingest_run(db, "w7", root)
        got = dict(db.execute("SELECT workload, recall_at_10 FROM perf").fetchall())
        self.assertEqual(got.get("W7"), 0.7)


if __name__ == "__main__":
    unittest.main()


class GraphSeedProvenanceTests(unittest.TestCase):
    """The level seed reaches `run.json`, and an older log still parses.

    §8.7 asks for the seed as provenance, and until then no part of a run
    recorded which one was used.
    """

    GRAPH_LINE = ("index: graph checksum=abc123 nodes=1000000 unreachable=0 "
                  "in_degree_zero=0 unreachable_with_out_edges=0 seed=0x57ea3111\n")
    #: What the engine printed before the seed was on the line.
    OLD_LINE = ("index: graph checksum=def456 nodes=1000 unreachable=2 "
                "in_degree_zero=1 unreachable_with_out_edges=0\n")

    def _record(self, log_text):
        import fullrun
        with tempfile.TemporaryDirectory() as d:
            label = "seed-arm"
            arm = Path(d) / label
            arm.mkdir()
            (arm / "server.log").write_text(log_text)
            (arm / "run.json").write_text(json.dumps({"label": label}))
            with mock.patch.object(fullrun, "RESULTS", Path(d)), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                fullrun.record_graph_quality(label)
            return json.loads((arm / "run.json").read_text()), out.getvalue()

    def test_seed_is_carried_into_the_run_record(self):
        doc, printed = self._record(self.GRAPH_LINE)
        self.assertEqual(doc["graph_builds"][0]["seed"], "57ea3111")
        # Hex, not an int: the seed is an identity, and 1475248401 is not the
        # number anyone configured.
        self.assertIsInstance(doc["graph_builds"][0]["seed"], str)
        self.assertIn("seed 0x57ea3111", printed)

    def test_a_log_without_a_seed_still_loads(self):
        # The field is optional so an arm measured before this existed reads as
        # "not recorded" rather than failing to parse and losing the rest.
        doc, printed = self._record(self.OLD_LINE)
        self.assertEqual(doc["graph_builds"][0]["nodes"], 1000)
        self.assertNotIn("seed", doc["graph_builds"][0])
        self.assertIn("seed not recorded", printed)

    def test_two_seeds_in_one_arm_are_called_out(self):
        # No current path builds two collections at different seeds, and if one
        # ever does their recall curves are not comparable.
        doc, printed = self._record(self.GRAPH_LINE + self.GRAPH_LINE.replace(
            "seed=0x57ea3111", "seed=0x1"))
        self.assertEqual(len(doc["graph_builds"]), 2)
        self.assertIn("2 DIFFERENT SEEDS", printed)
