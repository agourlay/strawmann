#!/usr/bin/env python3
"""Unit tests: compare.py's joins and its refusals."""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import io
import json
import os
import re
import tempfile
import typing
import unittest
from pathlib import Path
from unittest import mock

from harness_fixtures import (
    CONF_T1_FAILED,
    CONF_T2,
    CONF_T3,
    CONF_T3_FAILED,
    Fixture,
    N,
    _reload,
    _row,
    _sweep,
    good_stamp,
)

from workloads import stamp_hash


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_share_of_the_append_the_search_covered_survives_the_trim(self):
        """Stripped whole as the complement of `write overlap`, which it is not:
        the front page read "[append 2,000 points/s]" for a search that was
        there for 11% of the write."""
        trim = self.m["compare"].trim_row_note
        note = ("concurrent append of synthetic vectors; recall not measured; search "
                "finished 22.2 s before the append did (11% of the append was covered); "
                "append 2,000 points/s")
        self.assertEqual(trim(note), "search covered 11% of the append; append 2,000 points/s")
        # A note from before the percentage was recorded loses the seconds as before.
        self.assertEqual(trim("search finished 13.9 s before the append did; "
                              "append 2,000 points/s"), "append 2,000 points/s")

    def test_segment_confound_counts_graphs_where_it_can(self):
        """"Held 2 segments" against 1 was a caveat about nothing when the
        second was the empty appendable; read back per segment, it says so."""
        cmp = self.m["compare"]

        def pair(qd):
            self.fx.label("a", [], good_stamp(),
                          collections=[{"collection": "bench2", "segments_count": 1}])
            self.fx.label("b", [], good_stamp(),
                          collections=[{"collection": "bench2", **qd}])
            return cmp.segment_confound("a", "b")

        self.assertEqual(pair({"segments_count": 2, "populated_segments_count": 1}), "")
        four = pair({"segments_count": 5, "populated_segments_count": 4})
        self.assertIn("b held 5 segments (4 populated)", four)
        self.assertIn("read back per segment", four)
        # Captured before the per-segment read-back: the old upper-bound caveat.
        old = pair({"segments_count": 2})
        self.assertIn("b held 2 segments", old)
        self.assertIn("upper bound", old)

    def _joined(self, a_rows, b_rows, stamp_a, stamp_b, conf_a=CONF_T3, conf_b=CONF_T3,
                sweeps_a=(), sweeps_b=(), colls_a=None, colls_b=None):
        self.fx.label("a", a_rows, stamp_a, conf_a, sweeps_a, collections=colls_a)
        self.fx.label("b", b_rows, stamp_b, conf_b, sweeps_b, collections=colls_b)
        cmp = self.m["compare"]
        cmp._RECALL_CACHE.clear()
        a, b = cmp.load("a"), cmp.load("b")
        return {r.id: r for r in cmp.joined("a", "b", a, b)}, cmp

    def test_ratio_only_at_equal_recall(self):
        sw_a = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9815))]
        sw_b = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9820))]
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw_a, sweeps_b=sw_b)
        self.assertEqual(rows["W3"].ratio, "2.00x", rows["W3"].note_text)

    def test_no_ratio_when_recall_unequal(self):
        sw_a = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9815))]
        sw_b = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9988))]
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw_a, sweeps_b=sw_b)
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertIn("recall unequal", rows["W3"].note_text)

    def test_no_ratio_when_recall_missing(self):
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp())
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertIn("recall missing", rows["W3"].note_text)

    def test_no_ratio_when_both_recalls_are_below_the_floor(self):
        # Binary quantization at d=128: 0.0273 against 0.0265 is *equal* to well
        # inside RECALL_TOL, so the equality rule licenses a ratio comparing how
        # fast two engines return the wrong answer. The floor refuses it.
        w7 = dict(collection="bench7", quantization_oversampling=4.0, quantization_rescore=True)
        q = dict(oversampling=4.0, rescore=True)
        sw_a = [("recall.sift1m.bench7.json", _sweep("bench7", "sift1m", 0.0273, **q))]
        sw_b = [("recall.sift1m.bench7.json", _sweep("bench7", "sift1m", 0.0265, **q))]
        rows, _ = self._joined([_row("W7", 3564, **w7)], [_row("W7", 5050, **w7)],
                               good_stamp(), good_stamp(), sweeps_a=sw_a, sweeps_b=sw_b)
        self.assertEqual(rows["W7"].ratio, "-")
        self.assertIn("recall below the usable floor", rows["W7"].note_text)

    def test_ratio_survives_the_floor_where_the_encoding_works(self):
        # The same row and the same encoding at d=1536, where recall is 0.90 for
        # both. The rule that refuses the row above must not refuse this one:
        # the sentence it replaced was hard-coded to d=128 and fired here too.
        w7 = dict(collection="bench7", quantization_oversampling=4.0, quantization_rescore=True)
        q = dict(oversampling=4.0, rescore=True)
        sw_a = [("recall.sift1m.bench7.json", _sweep("bench7", "sift1m", 0.9076, **q))]
        sw_b = [("recall.sift1m.bench7.json", _sweep("bench7", "sift1m", 0.9086, **q))]
        rows, _ = self._joined([_row("W7", 4423, **w7)], [_row("W7", 4667, **w7)],
                               good_stamp(), good_stamp(), sweeps_a=sw_a, sweeps_b=sw_b)
        self.assertEqual(rows["W7"].ratio, "0.95x", rows["W7"].note_text)

    def test_compare_full_is_per_dataset_and_only_sift1m_owns_the_readme(self):
        cmp = self.m["compare"]
        head = dict(cmp.blocks_targets("sift1m"))
        self.assertEqual(head["README.md"], "compare-table")
        self.assertIn("docs/comparison-sift1m.md", head)
        # Another dataset writes its own document and leaves the front page be.
        other = dict(cmp.blocks_targets("dbpedia-openai-100K-1536-angular"))
        self.assertNotIn("README.md", other)
        self.assertIn("docs/comparison-dbpedia-openai-100K-1536-angular.md", other)
        # No index target: a generated list of the files in `docs/` was a third
        # block to keep current for no reader it served.
        self.assertNotIn("docs/comparison.md", other)

    def test_recall_join_rejects_wrong_collection_and_dataset(self):
        # A `recall.json` from the differ's own `relevance` collection, and one
        # keyed to another dataset, must not speak for bench2 on SIFT.
        sw = [("recall.json", _sweep("relevance", "sift1m", 0.98)),
              ("recall.sift1m.bench2.json", _sweep("bench2", "dbpedia-openai-1m", 0.98))]
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertIsNone(rows["W3"].rec_a)
        self.assertEqual(rows["W3"].ratio, "-")

    def test_quantized_row_joins_its_own_collection(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98)),
              ("recall.sift1m.bench6.json", _sweep("bench6", "sift1m", 0.95))]
        rows, _ = self._joined([_row("W6", 4000, collection="bench6")],
                               [_row("W6", 2000, collection="bench6")],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertAlmostEqual(rows["W6"].rec_a, 0.95)
        self.assertEqual(rows["W6"].ratio, "2.00x")

    def test_stale_when_the_two_labels_ran_different_datasets(self):
        """`--dataset` makes this reachable, so the refusal has to be real.

        Two corpora at the same dimension and metric — dbpedia-openai-1m and
        dbpedia-openai-100K — would otherwise agree on every stamp key there
        is, and their ratio would be one engine's throughput over another's
        divided by a factor of ten in collection size.
        """
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        self.fx.label("a", [_row("W3", 4000)], good_stamp(), CONF_T3, sw,
                      dataset="dbpedia-openai-1m")
        self.fx.label("b", [_row("W3", 2000)], good_stamp(), CONF_T3, sw,
                      dataset="sift1m")
        cmp = self.m["compare"]
        cmp._RECALL_CACHE.clear()
        reasons = cmp.stale_reasons("a", "b")
        self.assertTrue(any(r.startswith("dataset:") for r in reasons), reasons)
        rows = {r.id: r for r in cmp.joined("a", "b", cmp.load("a"), cmp.load("b"))}
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertTrue(any(b.startswith("STALE") for b in cmp.banners("a", "b")))

    def test_stale_when_stamp_missing_or_differs(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)], None, good_stamp(),
                                 sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertTrue(any("no harness stamp" in r for r in cmp.stale_reasons("a", "b")))
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                                 good_stamp(metric="Cosine"), good_stamp(),
                                 sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertTrue(any("metric" in r for r in cmp.stale_reasons("a", "b")))
        self.assertTrue(any(b.startswith("STALE") for b in cmp.banners("a", "b")))

    def test_licence_banner_when_not_comparative(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                                 good_stamp(), good_stamp(), CONF_T2, CONF_T2, sw, sw)
        bl = cmp.blocks_for("a", "b", cmp.load("a"), cmp.load("b"))
        self.assertIn("NOT A LICENSED COMPARATIVE CLAIM", bl["compare-table"])
        self.assertIn("NOT A LICENSED COMPARATIVE CLAIM", bl["compare-full"])
        # No conformance row at all -> UNLICENSED.
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                                 good_stamp(), good_stamp(), None, None, sw, sw)
        banner = cmp.licence("a", "b")["banner"]
        self.assertIn("UNLICENSED", banner)
        # And it says how to get the licence without re-measuring. §8's differ
        # runs last, so a run stopped early loses the licence for rows that are
        # otherwise complete -- the dbpedia-openai-1m run lost
        # nine hours of good rows to a phase that takes half an hour on its
        # own. The banner names that half hour.
        self.assertIn("--skip strawmann", banner)
        self.assertIn("--skip qdrant", banner)
        self.assertIn("--strawmann-label a", banner)
        self.assertIn("--qdrant-label b", banner)

    def test_the_recovery_command_carries_the_qdrant_the_rows_ran(self):
        """`--qdrant-binary` defaults to the pinned image.

        Omitting it licenses rows measured against a source-built Qdrant with
        a conformance row taken against `qdrant/qdrant:v1.19.0`, and nothing
        downstream compares `conformance.json`'s `qdrant_version` with the one
        in `run.json`. The dbpedia-openai-1m night ran 1.19.1-dev natively, so
        the first version of this banner would have caused exactly that.
        """
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                                 good_stamp(), good_stamp(), None, None, sw, sw)
        native = {"dataset": {"name": "sift1m"},
                  "qdrant": {"binary": "/x/target/release/qdrant",
                             "version": "1.19.1-dev"}}
        (cmp.ROOT / "bench/results" / "b" / "run.json").write_text(json.dumps(native))
        banner = cmp.licence("a", "b")["banner"]
        self.assertIn("--qdrant-binary /x/target/release/qdrant", banner)
        self.assertIn("different Qdrant", banner)

        # A run that used the image needs no flag, and gets no warning.
        (cmp.ROOT / "bench/results" / "b" / "run.json").write_text(json.dumps(
            {"dataset": {"name": "sift1m"},
             "qdrant": {"image": "qdrant/qdrant:v1.19.0", "binary": None}}))
        image = cmp.licence("a", "b")["banner"]
        self.assertNotIn("--qdrant-binary", image)
        self.assertNotIn("different Qdrant", image)

    def test_header_block_keeps_contamination_and_refusals(self):
        # A contaminated arm refuses the ratio (`decisions.md`: the per-row
        # foreign-load flag is a hard gate), and the header block says so.
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000, foreign="rustc(100%)")],
                                 [_row("W3", 2000)], good_stamp(), good_stamp(),
                                 sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertIn("contaminated", rows["W3"].refusal)
        hb = cmp.header_block("a", "b", cmp.load("a"), cmp.load("b"))
        self.assertIn("contaminated: rustc(100%)", hb)
        self.assertNotIn("2.00x", hb)
        # The clean pair still ratios, and carries no refusal.
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "2.00x")
        self.assertEqual(rows["W3"].refusal, "")

    def test_header_block_names_the_recorded_qdrant_version(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                                 good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        # No version recorded: the label alone, not a hardcoded `1.19`.
        self.assertEqual(cmp.column_title("b"), "b")
        self.assertNotIn("1.19", cmp.header_block("a", "b", cmp.load("a"), cmp.load("b")))
        d = Path(self.tmp.name) / "bench/results/b/run.json"
        meta = json.loads(d.read_text())
        meta["qdrant"] = {"version": "1.20.3"}
        d.write_text(json.dumps(meta))
        self.assertEqual(cmp.column_title("b"), "b 1.20")
        self.assertIn("b 1.20", cmp.header_block("a", "b", cmp.load("a"), cmp.load("b")))

    # --- per-row build identity (env.txt / run.json are rewritten per invocation) ---

    def test_rows_from_two_builds_are_stale(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        b1 = dict(engine_build="abc", engine_binary="1111", isa_build="avx512",
                  optimize="ReleaseFast", profile="isolated", gate="pass")
        b2 = dict(b1, engine_binary="2222")   # W3 re-run on a rebuilt binary
        rows, cmp = self._joined([_row("W3", 4000, **b2), _row("W4", 8000, **b1)],
                                 [_row("W3", 2000, **b1), _row("W4", 4000, **b1)],
                                 good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual({k: r.ratio for k, r in rows.items()}, {"W3": "-", "W4": "-"})
        reasons = cmp.stale_reasons("a", "b")
        self.assertTrue(any("2 different builds" in r and r.startswith("a:") for r in reasons),
                        reasons)
        self.assertTrue(any(b.startswith("STALE") for b in cmp.banners("a", "b")))
        # One build per label: not stale, and rows without the fields (measured
        # before they existed) are unknown rather than a second build.
        rows, cmp = self._joined([_row("W3", 4000, **b1), _row("W4", 8000)],
                                 [_row("W3", 2000, **b1), _row("W4", 4000, **b1)],
                                 good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(cmp.stale_reasons("a", "b"), [])
        self.assertEqual(rows["W3"].ratio, "2.00x", rows["W3"].note_text)

    def test_gate_of_reads_the_rows_before_env_txt(self):
        cmp = self.m["compare"]
        d = self.fx.label("g", [_row("W3", 1, gate="FAIL"), _row("W4", 1, gate="FAIL")],
                          good_stamp())
        (d / "env.txt").write_text("checks passed\n")   # rewritten by a later gated run
        self.assertEqual(cmp.gate_of("g"), "FAIL")
        self.fx.label("g", [_row("W3", 1, gate="pass"), _row("W4", 1, gate="FAIL")], good_stamp())
        self.assertEqual(cmp.gate_of("g"), "mixed")
        self.fx.label("g", [_row("W3", 1), _row("W4", 1)], good_stamp())
        self.assertEqual(cmp.gate_of("g"), "pass")   # legacy rows: env.txt

    def test_no_output_row_is_not_ok(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, _ = self._joined([_row("W3", None, status="no-output")], [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertIn("W3", rows, "a no-output row must not vanish from the join")
        self.assertEqual(rows["W3"].sa, "no-output")
        self.assertIn("no output", rows["W3"].note_text)

    def test_open_loop_saturated_and_queueing(self):
        lat_sat = {"client_p50_us": 1_100_000.0, "server_p50_us": 300.0}
        lat_ok = {"client_p50_us": 500.0, "server_p50_us": 400.0}
        rows, _ = self._joined([_row("W4-rps2000", 1998, load_mode="open-loop", latency=lat_sat)],
                               [_row("W4-rps2000", 1990, load_mode="open-loop", latency=lat_ok)],
                               good_stamp(), good_stamp())
        self.assertEqual(rows["W4-rps2000"].ratio, "saturated")
        # The note leads with the absolute time the server does not account
        # for, because a client/server *ratio* shrinks as the engine slows: it
        # once read 23x for the engine with the lower absolute latency of the
        # two, purely because its server p50 was 6 µs.
        note = rows["W4-rps2000"].note_text
        self.assertIn("not server time", note)
        # Two decimals, because the note now fires on closed-loop rows too and
        # W13's are sub-millisecond: at one decimal it read "0.1 ms of the
        # client's 0.1 ms p50", which says nothing.
        self.assertIn("1,099.70 ms", note)
        rows, _ = self._joined([_row("W4-rps500", 500, load_mode="open-loop", latency=lat_ok)],
                               [_row("W4-rps500", 500, load_mode="open-loop", latency=lat_ok)],
                               good_stamp(), good_stamp())
        self.assertEqual(rows["W4-rps500"].ratio, "offered")

    def test_open_loop_row_still_honours_the_contamination_gate(self):
        # The open-loop branch answers "offered" in place of a ratio, and it
        # used to answer before the contamination gate was asked: the row read
        # as a served rate and left `refusal` empty, which is the flag
        # `report.py` reads for every *other* figure built from these rows. A
        # hard gate that holds for one column only is a soft one.
        lat_ok = {"client_p50_us": 500.0, "server_p50_us": 400.0}
        rows, _ = self._joined(
            [_row("W4-rps500", 500, load_mode="open-loop", latency=lat_ok,
                  foreign="rustc(100%)")],
            [_row("W4-rps500", 500, load_mode="open-loop", latency=lat_ok)],
            good_stamp(), good_stamp())
        self.assertEqual(rows["W4-rps500"].ratio, "-")
        self.assertIn("contaminated", rows["W4-rps500"].refusal)

    def test_env_mismatch_reported(self):
        self.fx.label("a", [_row("W3", 1)], good_stamp())
        d = self.fx.label("b", [_row("W3", 1)], good_stamp())
        meta = json.loads((d / "run.json").read_text())
        meta["upload_n"] = 50_000
        (d / "run.json").write_text(json.dumps(meta))
        notes = self.m["compare"].env_mismatch("a", "b")
        self.assertTrue(any("upload_n" in n for n in notes), notes)

    def test_readme_check_warns_not_fails_on_stale_results(self):
        # Blocks written by --write-readme match, so the check passes even
        # though the results are stale; the warning is printed to stderr.
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)], None, None,
                                 CONF_T2, CONF_T2, sw, sw)
        a, b = cmp.load("a"), cmp.load("b")
        blocks = cmp.blocks_for("a", "b", a, b)
        self.assertEqual(cmp.splice_readme(blocks, "sift1m"), 0)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cmp.readme_is_current(blocks, True, "a", "b")
        self.assertEqual(rc, 0)
        self.assertIn("STALE", err.getvalue())


    def _stale_blocks(self, conf, stamp):
        """Splice the blocks, then hand back ones that no longer match."""
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, cmp = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                                 stamp, stamp, conf, conf, sw, sw)
        a, b = cmp.load("a"), cmp.load("b")
        blocks = cmp.blocks_for("a", "b", a, b)
        self.assertEqual(cmp.splice_readme(blocks, "sift1m"), 0)
        drifted = {k: v + "\n\nhand-edited" for k, v in blocks.items()}
        return cmp, drifted

    def test_unpublishable_results_do_not_fail_the_readme_gate(self):
        """A block that may not be written may not be demanded either.

        `fullrun.render` calls `--write-readme` only when nothing refuses the
        results, so a STALE or UNLICENSED set can never have been rendered into
        these blocks. Failing the step for that asks for a file the publish gate
        forbids writing, and prints `compare.py --write-readme` as the remedy --
        which is the one thing §8 forbids for unlicensed rows.

        Found by the dbpedia-openai-1m run: stopped before its
        differ, it left measured rows, no conformance row, no per-dataset
        document, and a permanently red gate. `scripts/check.py` already
        promised this step "still passes, so a host holding old results can
        develop"; only the licence branch honoured it.
        """
        cmp, drifted = self._stale_blocks(CONF_T2, None)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cmp.readme_is_current(drifted, True, "a", "b")
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("is stale", out)              # still reported
        self.assertIn("not failing for them", out)  # and says why it did not fail

    def test_a_publishable_stale_block_still_fails_the_readme_gate(self):
        """The gate this keeps: when the results *could* be written, drift is a
        failure, so the fix above cannot be widened into never failing."""
        cmp, drifted = self._stale_blocks(CONF_T3, good_stamp())
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cmp.readme_is_current(drifted, True, "a", "b")
        self.assertEqual(rc, 1, err.getvalue())
        self.assertNotIn("not failing for them", err.getvalue())

    # --- per-row harness stamps (rows.json merges in place; run.json does not) ---

    def test_subset_rerun_leaves_unstamped_rows_without_ratio(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        old = good_stamp(queries=10_000)   # the harness before a subset re-run
        # Label b re-ran W3 only: run.json carries today's stamp, W3 was
        # measured under it, W4 still holds the row measured under `old`.
        rows, _ = self._joined([_row("W3", 4000), _row("W4", 8000)],
                               [_row("W3", 2000), _row("W4", 4000, stamp=old)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "2.00x", rows["W3"].note_text)
        self.assertEqual(rows["W4"].ratio, "-")
        self.assertIn("re-run under a different harness", rows["W4"].note_text)
        self.assertNotIn("recall", rows["W4"].note_text)

    def test_full_run_ratios_every_row(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, _ = self._joined([_row("W3", 4000), _row("W4", 8000)],
                               [_row("W3", 2000), _row("W4", 4000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual({k: r.ratio for k, r in rows.items()}, {"W3": "2.00x", "W4": "2.00x"})

    def test_duration_rerun_with_larger_n_keeps_the_ratio_and_notes_the_n(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        # `--min-duration` re-ran a's W3 at 4x -n (the faster engine came in
        # short); b ran at the table's -n. qps is a rate: same stamp, ratio.
        rows, _ = self._joined([_row("W3", 4000, n=4 * N, notes=f"-n {4 * N} (4x the table's {N})")],
                               [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "2.00x", rows["W3"].note_text)
        self.assertNotIn("different harness", rows["W3"].note_text)
        self.assertEqual(rows["W3"].ra["n_requested"], 4 * N)
        self.assertIn(f"-n {4 * N}", rows["W3"].ra["notes"])
        # A stamp difference is still refused (a's row measured by an older harness).
        rows, _ = self._joined([_row("W3", 4000, stamp=good_stamp(queries=10_000))],
                               [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertIn("re-run under a different harness", rows["W3"].note_text)

    def test_row_without_stamp_is_refused_even_under_a_stamped_label(self):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows, _ = self._joined([_row("W3", 4000, stamp=None)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(), sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "-")
        self.assertIn("a: row has no harness stamp", rows["W3"].note_text)

    def test_stamp_hash_keys(self):
        self.assertEqual(stamp_hash(good_stamp()), stamp_hash(good_stamp(commit="zzz")),
                         "the commit is not a comparable field; STAMP_KEYS are")
        self.assertNotEqual(stamp_hash(good_stamp()), stamp_hash(good_stamp(metric="Cosine")))

    # --- W7: the sweep has to have searched the way the row did ---

    def test_w7_joins_only_a_sweep_at_its_own_quantization_params(self):
        w7 = dict(collection="bench7", quantization_oversampling=4.0, quantization_rescore=True)
        # A bench7 sweep that sent no quantization params (oversampling 1, the
        # engine's default): does not speak for W7 at 4x with rescore.
        sw1 = [("recall.sift1m.bench7.json", _sweep("bench7", "sift1m", 0.90))]
        rows, _ = self._joined([_row("W7", 4000, **w7)], [_row("W7", 2000, **w7)],
                               good_stamp(), good_stamp(), sweeps_a=sw1, sweeps_b=sw1)
        self.assertIsNone(rows["W7"].rec_a)
        self.assertEqual(rows["W7"].ratio, "-")
        # The same sweep taken at oversampling 4 with rescore: joins.
        sw4 = [("recall.sift1m.bench7.json",
                _sweep("bench7", "sift1m", 0.90, oversampling=4.0, rescore=True))]
        rows, _ = self._joined([_row("W7", 4000, **w7)], [_row("W7", 2000, **w7)],
                               good_stamp(), good_stamp(), sweeps_a=sw4, sweeps_b=sw4)
        # The join key is what this test is about. At 0.90 the recall clears
        # `RECALL_FLOOR`, so the row also gets its ratio: the refusal that used
        # to stand here was a hard-coded sentence about d=128, and this sweep
        # says 0.90.
        self.assertAlmostEqual(rows["W7"].rec_a, 0.90)
        self.assertEqual(rows["W7"].ratio, "2.00x")
        # And a row that sent nothing does not borrow the oversampled sweep.
        rows, _ = self._joined([_row("W7", 4000, collection="bench7")],
                               [_row("W7", 2000, collection="bench7")],
                               good_stamp(), good_stamp(), sweeps_a=sw4, sweeps_b=sw4)
        self.assertIsNone(rows["W7"].rec_a)

    # --- W11 ---

    def test_gate_verdict_has_one_definition(self):
        # `report.gate_pass_of` used to carry its own copy of this rule, which
        # is how an arm with 25 FAIL rows was published as `pass`.
        cmp = self.m["compare"]
        rep = importlib.import_module("report")
        rows = [_row("W3", 1, gate="FAIL"), _row("W11", 1, gate="pass")]
        self.assertEqual(cmp.gate_of_rows(rows), "mixed")
        self.assertFalse(rep.gate_pass_of(rows, "all checks passed"))
        # And it takes the mapping `load()` returns as readily as a list.
        self.assertEqual(cmp.gate_of_rows({r["id"]: r for r in rows}), "mixed")

    def test_env_txt_fallback_reads_the_last_verdict_in_the_file(self):
        # env.txt is appended per `workloads.py` invocation, so a file whose
        # first block failed and whose second passed describes rows measured
        # under both. Only rows with no `gate` field reach this path.
        cmp = self.m["compare"]
        legacy = [{"id": "W3", "qps": 1.0}]
        fail_then_pass = "1 check(s) failed\n...\nall checks passed\n"
        pass_then_fail = "all checks passed\n...\n1 check(s) failed\n"
        self.assertEqual(cmp.gate_of_rows(legacy, fail_then_pass), "pass")
        self.assertEqual(cmp.gate_of_rows(legacy, pass_then_fail), "FAIL")
        self.assertEqual(cmp.gate_of_rows(legacy, ""), "FAIL")

    def test_gate_verdict_comes_from_the_rows_not_from_env_txt(self):
        # `workloads.py run` rewrites env.txt and `fullrun` calls it twice per
        # arm, so a passing second invocation used to re-vouch for rows the
        # first had stamped FAIL. 25 of Qdrant's 26 sift1m rows.
        rep = importlib.import_module("report")
        passing_env = "all checks passed; this host is fit to produce published numbers"
        mixed = [_row("W3", 1, gate="FAIL"), _row("W11", 1, gate="pass")]
        self.assertFalse(rep.gate_pass_of(mixed, passing_env))
        self.assertTrue(rep.gate_pass_of([_row("W3", 1, gate="pass")], passing_env))
        self.assertFalse(rep.gate_pass_of([_row("W3", 1, gate="FAIL")], passing_env))
        # Rows from before the field existed still fall back to the file.
        legacy = [{"id": "W3", "qps": 1.0}]
        self.assertTrue(rep.gate_pass_of(legacy, passing_env))
        self.assertFalse(rep.gate_pass_of(legacy, "1 check(s) failed"))

    def test_a_failed_gate_on_any_row_is_a_reason_not_to_publish(self):
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/gated"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "rows.json").write_text(json.dumps(
            [_row("W3", 1, gate="FAIL"), _row("W4", 1, gate="pass")]))
        why = list(fr.publish_refusals("gated"))
        self.assertTrue(any("failed §7.1 gate" in r for r in why), why)
        # `reps=3`, to isolate the gate: a single pass is its own refusal below.
        (d / "rows.json").write_text(json.dumps([_row("W3", 1, gate="pass", reps=3)]))
        self.assertEqual(list(fr.publish_refusals("gated")), [])

    def test_a_single_pass_may_not_be_spliced_into_the_front_page(self):
        """Separate labels do not protect README.md, because it is keyed by
        corpus rather than by label.

        `smoke-test.sh` writes to `smoke-sm`/`smoke-qd` so a one-pass run
        cannot touch published results, and that covers `bench/results/` and
        the report HTML but not the generated markdown: `blocks_targets`
        emits README.md for any run whose dataset is sift1m, which is the
        corpus the smoke test fixes. A clean smoke run on a quiet host would
        have replaced the headline table with unbanded, A-then-B numbers and
        exited 0.

        §7.2(5) and §7.4 are the reason it must not: one pass alternates
        nothing and folds no noise floor, so the page cannot say whether any
        ratio is a difference -- which the report already banners twice while
        `render` spliced the numbers anyway.
        """
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/smoke-sm"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "rows.json").write_text(json.dumps(
            [_row("W3", 1, gate="pass"), _row("W4", 2, gate="pass")]))
        why = list(fr.publish_refusals("smoke-sm"))
        self.assertTrue(any("single pass" in r for r in why), why)
        self.assertTrue(any("--reps 3" in r for r in why), why)
        # Folded by `aggregate.py`, which stamps `reps`: publishable again.
        (d / "rows.json").write_text(json.dumps(
            [_row("W3", 1, gate="pass", reps=3), _row("W4", 2, gate="pass", reps=3)]))
        self.assertEqual(list(fr.publish_refusals("smoke-sm")), [])
        # A row folded from two passes is still not three, and `_rsd` gives no
        # floor below three -- but interleaving did happen, so this is the
        # boundary the message names rather than a second refusal.
        (d / "rows.json").write_text(json.dumps([_row("W3", 1, gate="pass", reps=2)]))
        self.assertEqual(list(fr.publish_refusals("smoke-sm")), [])

    def test_a_single_pass_is_refused_but_is_not_a_failure(self):
        """The two kinds of refusal, and why the exit code must tell them apart.

        `smoke-test.sh` fixes `--reps 1`, so the single-pass refusal fires on
        every smoke run. While `render` failed the run for any refusal at all,
        that made the script incapable of exiting 0: a clean sift1m smoke run
        that passed T4 and wrote its report still ended `some phase reported a
        non-zero status`, so the exit code carried no signal and the phase lines
        were the only way to tell a real fault from the configuration.
        """
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/smoke-sm"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"

        (d / "rows.json").write_text(json.dumps([_row("W3", 1, gate="pass")]))
        why = list(fr.publish_refusals("smoke-sm"))
        self.assertTrue(any("single pass" in r for r in why), why)
        # Refused, and yet not a defect: this run did what it was asked.
        self.assertEqual([r for r in why if fr.is_defect(r)], [])

        # A defect alongside it fails the run, and the single-pass reason is
        # still printed rather than swallowed by the louder one.
        (d / "rows.json").write_text(json.dumps(
            [_row("W3", 1, gate="pass"), _row("W4", 1, gate="FAIL")]))
        why = list(fr.publish_refusals("smoke-sm"))
        self.assertTrue(any("single pass" in r for r in why), why)
        self.assertTrue(any("failed §7.1 gate" in r and fr.is_defect(r) for r in why), why)

        # Reasons `render` raises itself are plain strings and must not read as
        # by-design just because they carry no flag.
        self.assertTrue(fr.is_defect("--lax: the §7.1 gate failed"))

    def _render(self, fr, labels, rows, conf=CONF_T3):
        """Run `render` with every subprocess stubbed out, returning (rc, out)."""
        fr.RESULTS = self.fx.root / "bench/results"
        for label in labels:
            d = fr.RESULTS / label
            d.mkdir(parents=True, exist_ok=True)
            (d / "rows.json").write_text(json.dumps(rows))
        (fr.RESULTS / labels[0] / "conformance.json").write_text(json.dumps(conf))
        real = fr.sh
        fr.sh = lambda *a, **k: (0, "")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = fr.render(list(labels))
            return rc, buf.getvalue()
        finally:
            fr.sh = real

    def test_render_exits_zero_when_the_only_refusal_is_the_asked_for_one(self):
        """A smoke run must be able to exit 0.

        Everything else about the run passed -- the sink took the rows, the
        differ licensed the comparison, the report rendered -- and the one
        refusal is `--reps 1`, which is what was requested. README.md is still
        withheld, which is the part that protects the front page.
        """
        fr = importlib.import_module("fullrun")
        rc, out = self._render(fr, ["smoke-sm", "smoke-qd"], [_row("W3", 1, gate="pass")])
        self.assertEqual(rc, 0, out)
        self.assertIn("NOT writing README.md", out)
        self.assertIn("not a fault in it", out)

    def test_render_still_fails_the_run_on_a_refusal_that_names_a_defect(self):
        """The gate this keeps: a defect fails even when a by-design reason is
        printed beside it, so the fix above cannot be widened into silence."""
        fr = importlib.import_module("fullrun")
        rc, out = self._render(fr, ["smoke-sm", "smoke-qd"],
                               [_row("W3", 1, gate="pass", foreign="cargo")])
        self.assertEqual(rc, 1, out)
        self.assertIn("foreign load", out)
        self.assertIn("single pass", out)
        self.assertNotIn("not a fault in it", out)

        # An unlicensed differ is `render`'s own reason rather than a row's,
        # and it fails the run on rows that are otherwise publishable.
        rc, out = self._render(fr, ["gated-sm", "gated-qd"],
                               [_row("W3", 1, gate="pass", reps=3)], conf=CONF_T2)
        self.assertEqual(rc, 1, out)
        self.assertIn("licenses_comparative is false", out)

    def test_the_header_names_the_engine_the_run_will_measure(self):
        """The header line was `QDRANT_IMAGE` whatever the run was doing.

        `--perf` requires `--qdrant-binary`, so every perf run -- every
        `smoke-test.sh` run that finds a binary -- printed
        `qdrant/qdrant:v1.19.0` and then measured a native build reporting
        1.19.1-dev. `run.json` carried the truth (`image: None, digest: None,
        version: 1.19.1-dev`) and so did the report; the console header, which
        is what anyone watches during the run, did not.
        """
        fr = importlib.import_module("fullrun")
        prev = fr.QDRANT_BINARY
        try:
            fr.QDRANT_BINARY = None
            self.assertIn(fr.QDRANT_IMAGE, fr.qdrant_target())
            self.assertIn("container", fr.qdrant_target())

            fr.QDRANT_BINARY = Path("/opt/qdrant/target/release/qdrant")
            target = fr.qdrant_target()
            self.assertIn("/opt/qdrant/target/release/qdrant", target)
            self.assertIn("native", target)
            # The image must not appear once a binary is pinned: naming both is
            # how the reader is left to guess which one was measured.
            self.assertNotIn(fr.QDRANT_IMAGE, target)
        finally:
            fr.QDRANT_BINARY = prev

    def test_unindexed_remainder_is_a_reason_not_to_publish(self):
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/lbl"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "collections.json").write_text(json.dumps({"collections": [
            {"collection": "bench0", "points_count": 100_000,
             "indexed_vectors_count": 72_500},
            {"collection": "bench2", "points_count": 120_000,
             "indexed_vectors_count": 120_000},
        ]}))
        why = fr.unindexed_remainders("lbl")
        self.assertEqual(len(why), 1, why)
        self.assertIn("27,500 of 100,000", why[0])
        self.assertIn("bench0", why[0])

    def test_w11s_own_append_is_not_an_unindexed_remainder(self):
        # W11 appends 200k to bench2 and the read-back happens after it, so the
        # rebuild is still catching up. That is the row's subject, not a defect,
        # and every row that searched bench2 ran before the append.
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/lbl2"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "collections.json").write_text(json.dumps({"collections": [
            {"collection": "bench2", "points_count": 1_200_000,
             "indexed_vectors_count": 1_100_255},
        ]}))
        self.assertEqual(fr.unindexed_remainders("lbl2"), [])

    def test_ingest_only_collection_is_not_an_unindexed_remainder(self):
        # bench1 is loaded with `--skip-wait-index` so W1 can measure ingest,
        # and no row, sweep or later phase searches it. Its entire corpus is
        # outside the index by design. This refused every full run's README on
        # 2026-09-03, once strawmANN began reporting an honest 0 there instead
        # of null.
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/lbl3"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "collections.json").write_text(json.dumps({"collections": [
            {"collection": "bench1", "points_count": 1_000_000,
             "indexed_vectors_count": 0},
            {"collection": "bench2", "points_count": 1_000_000,
             "indexed_vectors_count": 1_000_000},
        ]}))
        # W1 names bench1 too, which is the near-miss: scoping by "collections
        # some row names" would still flag it. Only W3 searched.
        rows = [
            {"id": "W1", "collection": "bench1", "n_queries": None},
            {"id": "W3", "collection": "bench2", "n_queries": 50_000},
        ]
        self.assertEqual(fr.unindexed_remainders("lbl3", rows), [])

    def test_a_searched_collection_is_still_a_remainder(self):
        # The scoping must not swallow the defect it was written around: the
        # same shortfall on a collection a query row *did* read still refuses.
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/lbl4"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "collections.json").write_text(json.dumps({"collections": [
            {"collection": "bench0", "points_count": 100_000,
             "indexed_vectors_count": 72_500},
        ]}))
        rows = [{"id": "W0", "collection": "bench0", "n_queries": 50_000}]
        why = fr.unindexed_remainders("lbl4", rows)
        self.assertEqual(len(why), 1, why)
        self.assertIn("27,500 of 100,000", why[0])

    def test_remainder_without_row_evidence_still_refuses(self):
        # No rows to scope by: keep the old behaviour rather than pass a run
        # nothing has vouched for. A refusal is the fail-safe direction.
        fr = importlib.import_module("fullrun")
        d = self.fx.root / "bench/results/lbl5"
        d.mkdir(parents=True, exist_ok=True)
        fr.RESULTS = self.fx.root / "bench/results"
        (d / "collections.json").write_text(json.dumps({"collections": [
            {"collection": "bench0", "points_count": 100_000,
             "indexed_vectors_count": 72_500},
        ]}))
        self.assertEqual(len(fr.unindexed_remainders("lbl5", [])), 1)
        self.assertEqual(len(fr.unindexed_remainders("lbl5", None)), 1)

    def test_qdrant_ab_carries_a_per_arm_environment(self):
        # A/B of a *configuration* needs the two arms to differ in the
        # environment the workload subprocess reads, not only in the image.
        ab = importlib.import_module("qdrant_ab")
        seen = {}

        def fake_run(argv, **kw):
            seen.update(kw.get("env") or {})
            class R:
                returncode = 0
            (kw["stdout"]).close()
            return R()

        import subprocess as sp
        real = sp.run
        out = self.fx.root / "ab"
        try:
            sp.run = fake_run
            with self.assertRaises(SystemExit):   # no rows.json from the fake
                ab.run_rows("arm", ["W3"], out, {"INDEXING_THRESHOLD_KB": "1000"})
        finally:
            sp.run = real
        self.assertEqual(seen.get("INDEXING_THRESHOLD_KB"), "1000")
        self.assertIn("RESULTS_DIR", seen)

    def test_full_scan_threshold_cannot_go_under_qdrants_minimum(self):
        # 1 KB was rejected at create time by every collection-creating row
        # ("value 1 invalid, must be 10 or larger"), which reads as a 0/26 arm
        # rather than as a bad constant.
        w = self.m["workloads"]
        self.assertGreaterEqual(w.FULL_SCAN_THRESHOLD_KB, w.QDRANT_MIN_FULL_SCAN_KB)

    def _w12(self, qps, **extra):
        """A W12 row measured with filtering implemented: the row says the index
        was *not* suppressed, so the check has to read the engine back."""
        extra.setdefault("payload_index_suppressed", False)
        return _row("W12-sel1", qps, collection="bench12", needs_payload_index=True,
                    **extra)

    @staticmethod
    def _coll(name="bench12", **extra):
        return [{"collection": name, "points_count": 200_000, **extra}]

    def test_a_suppressed_payload_index_needs_no_read_back(self):
        """W12 as measured until 2026-09-03: `--skip-field-indices` denied
        Qdrant its index, so the filtered queries scanned whatever
        `collections.json` goes on to say. The row records the flag itself;
        the live table no longer carries it."""
        rows, _ = self._joined([_row("W12-sel1", None, status="n/a",
                                     collection="bench12")],
                               [_row("W12-sel1", 17, collection="bench12",
                                     payload_index_suppressed=True)],
                               good_stamp(), good_stamp())
        self.assertEqual(rows["W12-sel1"].ratio, "-")
        self.assertIn("--skip-field-indices", rows["W12-sel1"].note_text)
        self.assertEqual(rows["W12-sel1"].refusal,
                         "payload index suppressed: the row measured a full scan")

    def test_filtered_row_without_a_payload_index_measured_a_full_scan(self):
        """The number is real and the row it is filed under is not.

        Qdrant answers a filtered search with no payload index by checking
        every point per query. It returns the right points and reports a rate,
        so the only symptom is the table calling a full scan "filtered search".
        """
        rows, _ = self._joined([self._w12(None, status="n/a")], [self._w12(17)],
                               good_stamp(), good_stamp(),
                               colls_a=self._coll(payload_indexes=[]),
                               colls_b=self._coll(payload_indexes=[]))
        self.assertEqual(rows["W12-sel1"].ratio, "-")
        self.assertIn("no payload index on bench12", rows["W12-sel1"].note_text)
        self.assertIn("must not be read as a filtered-search result",
                      rows["W12-sel1"].note_text)
        self.assertEqual(rows["W12-sel1"].refusal,
                         "no payload index: the row measured a full scan")

    def test_filtered_row_with_a_payload_index_is_not_refused_for_one(self):
        rows, _ = self._joined([self._w12(300)], [self._w12(17)],
                               good_stamp(), good_stamp(),
                               colls_a=self._coll(payload_indexes=["tag"]),
                               colls_b=self._coll(payload_indexes=["tag"]))
        self.assertNotIn("payload index", rows["W12-sel1"].note_text)
        self.assertNotIn("payload index", rows["W12-sel1"].refusal)

    def test_an_unread_payload_index_state_is_refused_like_a_missing_one(self):
        """No `collections.json` is not evidence that an index existed.

        The check exists because the row's flags cannot vouch for the engine;
        accepting silence would put the meaning of the number back on the
        flags, which is the thing being refused.
        """
        rows, _ = self._joined([self._w12(300)], [self._w12(17)],
                               good_stamp(), good_stamp())
        self.assertEqual(rows["W12-sel1"].ratio, "-")
        self.assertIn("never read back", rows["W12-sel1"].note_text)
        self.assertEqual(rows["W12-sel1"].refusal, "payload index state not read back")
        # Read back, but for another collection: the row's own is still unseen.
        rows, _ = self._joined([self._w12(300)], [self._w12(17)],
                               good_stamp(), good_stamp(),
                               colls_a=self._coll("bench2", payload_indexes=["tag"]),
                               colls_b=self._coll("bench2", payload_indexes=["tag"]))
        self.assertIn("never read back", rows["W12-sel1"].note_text)

    def test_a_declined_filtered_row_is_not_also_called_unindexed(self):
        """strawmANN answers UNIMPLEMENTED for `CreateFieldIndex`: a documented
        phase-3 boundary, and it has no figure for the check to disqualify.
        Naming it here reports the boundary twice, the second time as a defect."""
        rows, _ = self._joined([self._w12(None, status="n/a")], [self._w12(17)],
                               good_stamp(), good_stamp(),
                               colls_a=self._coll(payload_indexes=[]),
                               colls_b=self._coll(payload_indexes=[]))
        self.assertNotIn("a: no payload index", rows["W12-sel1"].note_text)
        self.assertIn("b: no payload index", rows["W12-sel1"].note_text)

    def test_the_check_covers_rows_measured_before_the_field_existed(self):
        """`needs_payload_index` is read off the table when the row lacks it, so
        results already on disk are checked rather than grandfathered — and a
        row that records `False` keeps it, which `or` would not have allowed."""
        old = _row("W12-sel1", 17, collection="bench12")
        self.assertNotIn("needs_payload_index", old)
        rows, _ = self._joined([old], [old], good_stamp(), good_stamp(),
                               colls_a=self._coll(payload_indexes=[]),
                               colls_b=self._coll(payload_indexes=[]))
        # The table stopped suppressing the index, so a row
        # that records nothing is judged by the read-back: none is a scan.
        self.assertNotIn("--skip-field-indices", rows["W12-sel1"].note_text)
        self.assertIn("no payload index on bench12", rows["W12-sel1"].note_text)
        self.assertEqual(rows["W12-sel1"].ratio, "-")
        # And one whose engines both read an index back is not refused for it.
        good, _ = self._joined([self._w12(4000)], [self._w12(2000)],
                               good_stamp(), good_stamp(),
                               colls_a=self._coll(payload_indexes=["a"]),
                               colls_b=self._coll(payload_indexes=["a"]))
        self.assertNotIn("payload index", good["W12-sel1"].note_text)

    def test_only_filtered_rows_are_checked_for_an_index(self):
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp())
        self.assertNotIn("payload index", rows["W3"].note_text)
        self.assertNotIn("payload index", rows["W3"].refusal)

    def test_w11_ratio_policy_replaces_recall_missing(self):
        w11 = dict(recall_joinable=False, ratio_policy="search-during-write; no recall join")
        rows, cmp = self._joined([_row("W11", 4000, **w11)], [_row("W11", 2000, **w11)],
                                 good_stamp(), good_stamp())
        self.assertEqual(rows["W11"].ratio, "-")
        self.assertIn("[search-during-write; no recall join]", rows["W11"].note_text)
        self.assertNotIn("recall missing", rows["W11"].note_text)
        # A W11 row measured before the field existed gets the table's policy.
        rows, cmp = self._joined([_row("W11", 4000, recall_joinable=False)],
                                 [_row("W11", 2000, recall_joinable=False)],
                                 good_stamp(), good_stamp())
        self.assertIn("search-during-write", rows["W11"].note_text)
        self.assertNotIn("recall missing", rows["W11"].note_text)
        hb = cmp.header_block("a", "b", cmp.load("a"), cmp.load("b"))
        self.assertIn("search-during-write; no recall join", hb)
        self.assertNotIn("recall missing", hb)

    def test_a_monotone_slide_across_passes_is_not_a_noise_band(self):
        """Drift and noise want opposite treatment, and `rep_rsd` cannot tell
        them apart.

        strawmANN's W10-ef128 on dbpedia-openai-1m read 3,662, 3,001 then
        2,764 qps in one run — one direction, at a constant clock, a constant
        instruction count per query, no major faults and no I/O wait. Folded as
        spread that is `rsd 14.8%`, which becomes a ±63% parity band, and the
        1.18x ratio published as "no measured difference". The difference was
        real; the arm was sliding.
        """
        # Equal recall, so the row reaches the ratio rather than being refused
        # earlier: drift is the only thing under test here.
        sw_a = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9815))]
        sw_b = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9820))]
        rows, _ = self._joined([_row("W3", 2764, rep_drift=-0.245)],
                               [_row("W3", 2537)],
                               good_stamp(), good_stamp(), sweeps_a=sw_a, sweeps_b=sw_b)
        note = rows["W3"].note_text
        self.assertNotEqual(rows["W3"].ratio, "parity")
        self.assertIn("monotonically", note)
        self.assertIn("drift rather than noise", note)
        self.assertIn("-24%", note)
        # The refusal string reaches the matched-recall frontier as its
        # explanation, so it must name the arm and the move, not just say that
        # something drifted.
        self.assertIn("-24%", rows["W3"].refusal)
        self.assertIn("a", rows["W3"].refusal)

        # Without the mark the row behaves exactly as before: this guard must
        # not start refusing every ratio.
        rows, _ = self._joined([_row("W3", 2764)], [_row("W3", 2537)],
                               good_stamp(), good_stamp(), sweeps_a=sw_a, sweeps_b=sw_b)
        self.assertEqual(rows["W3"].ratio, "1.09x")
        self.assertNotIn("monotonically", rows["W3"].note_text)

    def test_w11_policy_note_carries_a_low_write_overlap(self):
        # The live path: W11 carries its policy, and below the overlap floor
        # the policy note says how much of the row ran after the writer exited
        # -- not that the collection was quiet, which it is not: W11's append
        # trips a rebuild that outlives it.
        w11 = dict(recall_joinable=False, ratio_policy="search-during-write; no recall join")
        rows, _ = self._joined([_row("W11", 4000, write_overlap_pct=40.0, **w11)],
                               [_row("W11", 2000, write_overlap_pct=95.0, **w11)],
                               good_stamp(), good_stamp())
        self.assertEqual(rows["W11"].ratio, "-")
        # Only the arm below the floor is named: the other one's writer covered
        # 95% of its search, and a note saying "both rows" would describe a row
        # it does not apply to.
        self.assertIn("under 90% of a's search", rows["W11"].note_text)
        self.assertNotIn("either search", rows["W11"].note_text)
        self.assertNotIn(" b's search", rows["W11"].note_text)
        # The engine is at its busiest in the remainder, rebuilding what the
        # append forced. Calling that a quiet collection inverts the row.
        self.assertNotIn("quiet collection", rows["W11"].note_text)
        self.assertIn("rebuild the append provoked", rows["W11"].note_text)
        rows, _ = self._joined([_row("W11", 4000, write_overlap_pct=95.0, **w11)],
                               [_row("W11", 2000, write_overlap_pct=95.0, **w11)],
                               good_stamp(), good_stamp())
        self.assertIn("[search-during-write; no recall join]", rows["W11"].note_text)
        self.assertNotIn("write overlap", rows["W11"].note_text)

    def test_w11_note_says_most_only_when_it_was_most(self):
        """0925's strawmANN W11 ran at 82% overlap and the note said "most of
        that row measured the rebuild": 18% of it did."""
        w11 = dict(recall_joinable=False, ratio_policy="search-during-write; no recall join")
        note = lambda pa, pb: self._joined(
            [_row("W11", 4000, write_overlap_pct=pa, **w11)],
            [_row("W11", 2000, write_overlap_pct=pb, **w11)],
            good_stamp(), good_stamp())[0]["W11"].note_text
        text = note(82.0, 100.0)
        self.assertIn("under 90% of a's search, so the last 18% of that row", text)
        self.assertNotIn("most", text)
        self.assertIn("most of that row", note(40.0, 100.0))
        self.assertIn("most of both rows", note(12.0, 22.0))
        text = note(82.0, 60.0)
        self.assertIn("the last 18% and 40% of the two rows", text)
        self.assertNotIn("most", text)

    def test_system_load_is_not_read_as_foreign_load(self):
        """`load_start` counts the engine under test; `foreign` does not.

        A banner was added on `load_start` and fired on both arms of two clean
        runs — Qdrant at 126-133% against strawmANN's 62-66%, which is Qdrant's
        78 threads against strawmANN's 9 and not contamination at all. Foreign
        load has its own per-row measurement, and it refuses the row.
        """
        cmp = self.m["compare"]
        busy = _row("W3", 1000)
        busy["load_start"] = 900          # nine cores' worth of load average
        rows, _ = self._joined([busy], [_row("W3", 1000)],
                               good_stamp(), good_stamp())
        self.assertEqual([b for b in cmp.banners("a", "b")
                          if "LOAD" in b.upper()], [],
                         "a busy load average is not, by itself, a refusal")
        self.assertNotIn("contaminated", rows["W3"].note_text)

        # A row the /proc sampler actually flagged is a different matter.
        dirty = _row("W3", 1000)
        dirty["foreign"] = "rustc(700%)"
        rows, _ = self._joined([dirty], [_row("W3", 1000)],
                               good_stamp(), good_stamp())
        self.assertIn("contaminated", rows["W3"].note_text)
        self.assertTrue(rows["W3"].refusal)

    def test_a_dead_encoding_is_refused_as_dead_not_as_unequal(self):
        """W7: 0.0485 against 0.0267 is not a tolerance problem.

        Both are under `RECALL_FLOOR`, and they are also further apart than
        `RECALL_TOL`. The tolerance arm used to answer first, so the row read
        "§7.4 compares at equal recall" — which reads as repairable.
        """
        self.m["compare"]
        sw_a = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.0485))]
        sw_b = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.0267))]
        rows, _ = self._joined([_row("W3", 4000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(),
                               sweeps_a=sw_a, sweeps_b=sw_b)
        note = rows["W3"].note_text
        self.assertIn("characterises the encoding", note)
        self.assertNotIn("compares at equal recall", note)
        self.assertTrue(rows["W3"].refusal)

    def test_rows_from_an_earlier_run_refuse_the_label(self):
        """`qd-dbp100k`,: an arm re-run and killed left the new
        run's `run.json` over the old run's rows, same corpus, so
        `foreign_dataset` saw nothing."""
        cmp = self.m["compare"]
        self.fx.label("a", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        d = cmp.ROOT / "bench/results/a"

        # No session anywhere: legacy result set, nothing to say.
        self.assertEqual(cmp.orphaned_row_reasons("a"), [])

        meta = json.loads((d / "run.json").read_text())
        meta["session"] = "1000-1"
        (d / "run.json").write_text(json.dumps(meta))

        # run.json is stamped, the rows are not: they predate this run.
        self.assertIn("were not measured by the run their run.json describes",
                      cmp.orphaned_row_reasons("a")[0])

        rows = json.loads((d / "rows.json").read_text())
        for r in rows:
            r["session"] = "1000-1"
        (d / "rows.json").write_text(json.dumps(rows))
        self.assertEqual(cmp.orphaned_row_reasons("a"), [])

        # One row left behind by an earlier run is enough to refuse.
        rows[0]["session"] = "999-1"
        (d / "rows.json").write_text(json.dumps(rows))
        reason = cmp.orphaned_row_reasons("a")[0]
        self.assertIn(rows[0]["id"], reason)
        self.assertIn("re-run the arm whole", reason)

    def test_the_column_names_the_engine_not_the_label(self):
        """The block is the README's above-the-fold summary.

        It read `strawmann` / `qdrant` only because those were the label
        names; the same pair measured under `sm-sift-perf` / `qd-sift-perf`
        would have put run bookkeeping above the fold.
        """
        cmp = self.m["compare"]
        self.fx.label("sm-sift-perf", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        f = cmp.ROOT / "bench/results/sm-sift-perf/run.json"
        d = json.loads(f.read_text())
        d["engine_comm"] = "strawmann"
        f.write_text(json.dumps(d))
        self.assertEqual(cmp.column_title("sm-sift-perf"), "strawmann")

        # Qdrant keeps its version beside the engine name.
        self.fx.label("qd-sift-perf", [_row("W3", 900)], good_stamp(), CONF_T3, ())
        f = cmp.ROOT / "bench/results/qd-sift-perf/run.json"
        d = json.loads(f.read_text())
        d["engine_comm"] = "qdrant"
        d["qdrant"] = {"version": "1.19.1-dev"}
        f.write_text(json.dumps(d))
        self.assertEqual(cmp.column_title("qd-sift-perf"), "qdrant 1.19")

        # A run predating `engine_comm` still names itself.
        self.fx.label("ancient", [_row("W3", 900)], good_stamp(), CONF_T3, ())
        f = cmp.ROOT / "bench/results/ancient/run.json"
        d = json.loads(f.read_text())
        d.pop("engine_comm", None)
        f.write_text(json.dumps(d))
        self.assertEqual(cmp.column_title("ancient"), "ancient")

    def test_the_block_prose_names_engines_like_its_header(self):
        """A header saying `strawmann` over a note saying `sm-sift-perf`
        makes the reader work out that they are the same run."""
        cmp = self.m["compare"]
        for lab, eng in (("sm-sift-perf", "strawmann"), ("qd-sift-perf", "qdrant")):
            self.fx.label(lab, [_row("W3", 1000)], good_stamp(), CONF_T3, ())
            f = cmp.ROOT / "bench/results" / lab / "run.json"
            d = json.loads(f.read_text())
            d["engine_comm"] = eng
            f.write_text(json.dumps(d))
        block = cmp.header_block("sm-sift-perf", "qd-sift-perf",
                                 cmp.load("sm-sift-perf"), cmp.load("qd-sift-perf"))
        self.assertIn("strawmann", block)
        self.assertNotIn("sm-sift-perf", block)
        self.assertNotIn("qd-sift-perf", block)

    def test_parity_band_refuses_a_floor_from_another_dataset(self):
        """A floor stamped with another dataset does not band these rows.

        dbpedia-openai-100K printed `1.00x` on W10-ef256 against a spread
        measured on sift1m at a twelfth the dimension.
        """
        cmp = self.m["compare"]
        noise = cmp.ROOT / "bench/results/noise.json"
        noise.parent.mkdir(parents=True, exist_ok=True)

        # `arms` throughout: this test is about the dataset axis, and an
        # unattributed floor is refused on the engine axis regardless (see
        # `test_parity_band_refuses_a_floor_from_one_engine`).
        both = ["a", "b"]

        # Dataset unstamped: usable, so the band comes back.
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6},
                                     "arms": both}))
        self.fx.label("a", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        self.fx.label("b", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        self.assertIsNotNone(cmp.parity_band("W3", "a", "b"))

        # Stamped with this run's dataset: still usable.
        ds = cmp.dataset_spec_of("a", "b").get("name")
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6},
                                     "dataset": ds, "arms": both}))
        self.assertIsNotNone(cmp.parity_band("W3", "a", "b"))

        # Stamped with another dataset: refused.
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6},
                                     "dataset": "some-other-corpus", "arms": both}))
        self.assertIsNone(cmp.parity_band("W3", "a", "b"))

        # A row the floor never covered is never banded.
        self.assertIsNone(cmp.parity_band("W99", "a", "b"))

    def test_parity_band_refuses_a_floor_from_another_environment(self):
        """A floor measured on a different machine does not band these rows.

        Qdrant's saturating row read 10,313 against a twelve-run
        aggregate of 14,716 and was taken for a 30% regression. The aggregate
        was measured with SMT off and the run with it on; strawmANN moved 1.5%
        across the same change and Qdrant 30%, so there is not even a
        correction factor. Nothing refused it because nothing carried the hash
        where a tool could see it (findings 46).
        """
        cmp = self.m["compare"]
        noise = cmp.ROOT / "bench/results/noise.json"
        noise.parent.mkdir(parents=True, exist_ok=True)
        both = ["a", "b"]
        base = {"rsd": {"W3": 0.01}, "reps": {"W3": 6}, "arms": both}

        self.fx.label("a", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        self.fx.label("b", [_row("W3", 1000)], good_stamp(), CONF_T3, ())

        def stamp_env(label, value):
            f = cmp.ROOT / "bench/results" / label / "run.json"
            d = json.loads(f.read_text())
            d["env_hash"] = value
            f.write_text(json.dumps(d))

        # Floor unstamped: predates the stamp, stays usable.
        noise.write_text(json.dumps(base))
        stamp_env("a", "smt-on-hash")
        stamp_env("b", "smt-on-hash")
        self.assertIsNotNone(cmp.parity_band("W3", "a", "b"))

        # Floor from this environment: usable.
        noise.write_text(json.dumps({**base, "env_hash": "smt-on-hash"}))
        self.assertIsNotNone(cmp.parity_band("W3", "a", "b"))

        # Floor from another environment: refused.
        noise.write_text(json.dumps({**base, "env_hash": "smt-off-hash"}))
        self.assertIsNone(cmp.parity_band("W3", "a", "b"))

        # A run that does not know its own environment cannot contradict the
        # floor, so the floor is still used -- permissive where it cannot know,
        # like the dataset rule, and refusing only on a disagreement.
        stamp_env("a", None)
        stamp_env("b", None)
        self.assertIsNotNone(cmp.parity_band("W3", "a", "b"))

        # One arm disagreeing is enough: a ratio is both arms' spread.
        stamp_env("a", "smt-on-hash")
        self.assertIsNone(cmp.parity_band("W3", "a", "b"))

    def test_parity_band_refuses_a_floor_from_one_engine(self):
        """A ratio's band is both arms' spread, and one arm's is not it.

        Findings 38: `noise.json` here is six passes of strawmANN and gave W3
        an RSD of 0.46%, while Qdrant's own spread on that row is 13.16% over
        twelve runs. The single-query row read 1.11x and then 1.26x
        with nothing changed but the draw, and the band said neither
        was noise.
        """
        cmp = self.m["compare"]
        noise = cmp.ROOT / "bench/results/noise.json"
        noise.parent.mkdir(parents=True, exist_ok=True)
        self.fx.label("a", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        self.fx.label("b", [_row("W3", 1000)], good_stamp(), CONF_T3, ())

        # Both arms measured: banded.
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6},
                                     "arms": ["a", "b"]}))
        self.assertIsNotNone(cmp.parity_band("W3", "a", "b"))
        self.assertEqual(cmp.unbanded_banner("a", "b"), [])

        # One arm measured: refused, and the page says whose it was.
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6},
                                     "arms": ["a"]}))
        self.assertIsNone(cmp.parity_band("W3", "a", "b"))
        self.assertIn("spread on disk is a's alone", " ".join(cmp.unbanded_banner("a", "b")))

        # Unattributed: also refused. Unlike a missing `dataset`, a missing
        # `arms` cannot mean "maybe both" — a floor is one label's repetitions.
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6}}))
        self.assertIsNone(cmp.parity_band("W3", "a", "b"))
        self.assertIn("does not record which engine",
                      " ".join(cmp.unbanded_banner("a", "b")))

    def test_a_ratio_inside_the_band_reads_parity_not_a_number(self):
        cmp = self.m["compare"]
        noise = cmp.ROOT / "bench/results/noise.json"
        noise.parent.mkdir(parents=True, exist_ok=True)
        noise.write_text(json.dumps({"rsd": {"W3": 0.01}, "reps": {"W3": 6},
                                     "arms": ["a", "b"]}))
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9815))]
        rows, _ = self._joined([_row("W3", 2000)], [_row("W3", 2000)],
                               good_stamp(), good_stamp(),
                               sweeps_a=sw, sweeps_b=sw)
        self.assertEqual(rows["W3"].ratio, "parity")
        self.assertIn("no measured difference, not a small one",
                      rows["W3"].note_text)

    def test_conformance_block_escapes_the_pipes_in_a_tier_detail(self):
        """The differ writes `|` as a field separator inside `detail`."""
        cmp = self.m["compare"]
        self.fx.label("a", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        self.fx.label("b", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        conf = cmp.ROOT / "bench/results/a/conformance.json"
        conf.write_text(json.dumps({
            "tiers": [{"tier": "T1", "passed": True,
                       "detail": "max=0 | ids agree | vs oracle 1.5e-5"}],
            "tier_reached": "T1", "licenses_perf": True, "hash": "abc"}))
        block = cmp.conformance_block("a", "b")
        row = next(l for l in block if l.startswith("| T1 "))
        # Split on delimiters the escape did not neutralise: leading empty,
        # three cells, trailing empty.
        self.assertEqual(len(re.split(r"(?<!\\)\|", row)), 5)
        self.assertIn(r"\|", row)
        self.assertIn("licenses performance claims", block[-1])

    def test_scope_line_names_the_dimension_the_ratios_belong_to(self):
        cmp = self.m["compare"]
        self.fx.label("a", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        self.fx.label("b", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        line = cmp.scope_line("a", "b")
        self.assertIn(cmp.dataset_spec_of("a", "b")["name"], line)
        self.assertIn("these ratios are this dimension's", line)
        # And the dimension itself, whenever the run recorded one.
        self.fx.label("c", [_row("W3", 1000)], good_stamp(), CONF_T3, ())
        meta = json.loads((cmp.ROOT / "bench/results/c/run.json").read_text())
        meta["dataset"]["dim"] = 1536
        (cmp.ROOT / "bench/results/c/run.json").write_text(json.dumps(meta))
        self.assertIn("d=1536", cmp.scope_line("c", "c"))



class MixedBuildTests(unittest.TestCase):
    """`rows.json` merges in place, so a subset re-run on a rebuilt binary
    leaves a file describing two builds. The check for that keyed on the
    repository commit, and a run long enough to outlast a commit to the tree —
    to the harness, or to the docs — has two commit labels and one binary."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.c = self.m["compare"]

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, *specs):
        return {f"W{i}": {"id": f"W{i}", "engine_build": b, "engine_binary": s,
                          "isa_build": "native", "optimize": "ReleaseFast",
                          "profile": "as-deployed"}
                for i, (b, s) in enumerate(specs)}

    def test_a_commit_label_is_not_a_rebuilt_binary(self):
        rows = self._rows(("aaaa", "sha1"), ("bbbb", "sha1"))
        self.assertEqual(self.c.mixed_build_reasons("l", rows), [])
        drift = self.c.commit_drift("l", rows)
        self.assertIn("2 repository commits", drift)
        self.assertIn("one binary served every row", drift)

    def test_a_different_binary_is_still_refused(self):
        rows = self._rows(("aaaa", "sha1"), ("bbbb", "sha2"))
        why = self.c.mixed_build_reasons("l", rows)
        self.assertTrue(why)
        self.assertIn("2 different builds", why[0])
        self.assertEqual(self.c.commit_drift("l", rows), "")

    def test_an_unknown_binary_is_not_a_match(self):
        """Two rows with no sha256 cannot show they came from one build, and
        unknown must not read as equal."""
        rows = self._rows(("aaaa", None), ("bbbb", None))
        self.assertTrue(self.c.mixed_build_reasons("l", rows))

    def test_a_differing_isa_is_refused_even_on_one_binary(self):
        rows = self._rows(("aaaa", "sha1"), ("bbbb", "sha1"))
        rows["W1"]["isa_build"] = "avx2"
        self.assertTrue(self.c.mixed_build_reasons("l", rows))
        self.assertEqual(self.c.commit_drift("l", rows), "")

    def test_one_build_says_nothing(self):
        rows = self._rows(("aaaa", "sha1"), ("aaaa", "sha1"))
        self.assertEqual(self.c.mixed_build_reasons("l", rows), [])
        self.assertEqual(self.c.commit_drift("l", rows), "")

class RetentionTests(unittest.TestCase):
    """`fullrun.prunable`: what a run is allowed to delete from bench/results.

    The rule exists because nothing pruned that directory. It has four
    protections because a bare age rule already cost this project a document:
    `sm-dbp100k`/`qd-dbp100k` were deleted and
    `docs/comparison-dbpedia-openai-100K-1536-angular.md` can no longer be
    generated from anything.
    """

    OLD = 1_600_000_000.0  # any fixed epoch; `now` is passed explicitly

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "docs/reports").mkdir(parents=True)
        (self.root / "README.md").write_text("front page\n")
        self.results = self.root / "bench/results"
        self.results.mkdir(parents=True)
        self.fr = importlib.import_module("fullrun")
        self._saved = (self.fr.ROOT, self.fr.RESULTS, self.fr.planned_rows)
        self.fr.ROOT, self.fr.RESULTS = self.root, self.results
        # Spans and folding, not the row set: each basis priced as it ran.
        self.fr.planned_rows = lambda: None

    def tearDown(self):
        self.fr.ROOT, self.fr.RESULTS, self.fr.planned_rows = self._saved
        self.tmp.cleanup()

    def _label(self, name, *, age_days=400, files=("rows.json",)):
        d = self.results / name
        d.mkdir(parents=True, exist_ok=True)
        when = self.OLD - age_days * 86400
        for f in files:
            (d / f).write_text("{}")
            os.utime(d / f, (when, when))
        os.utime(d, (when, when))
        return d

    def _prunable(self, **kw):
        return [p.name for p in self.fr.prunable(now=self.OLD, **kw)]

    def test_an_old_uncited_label_is_disposable(self):
        self._label("exp-w9-cached")
        self.assertEqual(self._prunable(), ["exp-w9-cached"])

    def test_a_label_a_document_names_is_kept(self):
        self._label("sm-sift-perf")
        (self.root / "docs/findings.md").write_text(
            "the run at `sm-sift-perf` is where this was measured\n")
        self.assertEqual(self._prunable(), [])

    def test_citing_one_arm_keeps_the_other(self):
        """The dbp100k lesson: compare.py needs both sides or the block it
        backs can never be regenerated."""
        self._label("sm-dbp100k")
        self._label("qd-dbp100k")
        (self.root / "docs/findings.md").write_text("see `sm-dbp100k`\n")
        self.assertEqual(self._prunable(), [])

    def test_a_committed_report_page_cites_its_labels_by_filename(self):
        self._label("sm-sift-perf-rel-0903")
        self._label("qd-sift-perf-rel-0903")
        (self.root / "docs/reports"
         / "report-sift1m-sm-sift-perf-rel-0903-vs-qd-sift-perf-rel-0903-2026-09-03-1652.html"
         ).write_text("<html>")
        self.assertEqual(self._prunable(), [])

    def test_a_hand_written_note_is_not_regenerable_so_the_label_stays(self):
        self._label("night-20260903", files=("rows.json", "analysis.md"))
        self.assertEqual(self._prunable(), [])

    def test_a_lone_rendering_of_a_run_stays(self):
        self._label("archive-old", files=("report-sift1m-a-vs-b-2026-08-29.html",))
        self.assertEqual(self._prunable(), [])

    def test_a_recent_label_stays_however_uncited(self):
        self._label("run-yesterday", age_days=1)
        self.assertEqual(self._prunable(), [])

    def test_the_running_run_keeps_its_own_labels(self):
        self._label("sm-fresh")
        self.assertEqual(self._prunable(keep={"sm-fresh"}), [])
        self.assertEqual(self._prunable(), ["sm-fresh"])

    def test_the_noise_floor_is_not_a_label(self):
        (self.results / "noise.json").write_text("{}")
        (self.results / "qdrant-server.log").write_text("x")
        os.utime(self.results / "noise.json", (self.OLD - 400 * 86400,) * 2)
        self.assertEqual(self._prunable(), [])
        self.assertTrue((self.results / "noise.json").is_file())

    def test_prune_removes_exactly_what_prunable_named(self):
        """Protected by citation rather than by age, so the assertion does not
        depend on how long the test takes to run."""
        self._label("exp-a")
        self._label("exp-b")
        self._label("sm-cited")
        (self.root / "docs/findings.md").write_text("measured at `sm-cited`\n")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.fr.prune(retain_days=0)
        self.assertIn("exp-a", out.getvalue())
        left = sorted(q.name for q in self.results.iterdir())
        self.assertEqual(left, ["sm-cited"])


class EstimateTests(unittest.TestCase):
    """`fullrun.estimated_minutes`: the figure printed before the gate.

    It exists so the cost is on screen before the hours rather than discovered
    in the morning. The bug regression-tested here is the quiet half of that:
    folded labels were skipped outright, and `RetentionTests`' pruning removes
    the unfolded rep labels once they have been folded -- so on the measuring
    host every label of every corpus was folded, and this function answered
    `unknown` for all three, including in front of a nine-hour dbpedia run.
    """

    T0 = dt.datetime(2026, 9, 7, 7, 0, tzinfo=dt.UTC)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results = self.root / "bench/results"
        self.results.mkdir(parents=True)
        self.fr = importlib.import_module("fullrun")
        self._saved = (self.fr.ROOT, self.fr.RESULTS, self.fr.planned_rows)
        self.fr.ROOT, self.fr.RESULTS = self.root, self.results
        # Spans and folding, not the row set: each basis priced as it ran.
        self.fr.planned_rows = lambda: None

    def tearDown(self):
        self.fr.ROOT, self.fr.RESULTS, self.fr.planned_rows = self._saved
        self.tmp.cleanup()

    def _at(self, offset_s: float) -> str:
        return (self.T0 + dt.timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _label(self, name, engine, *, each=10.0, n_rows=4, spans=None, row_gap=None,
               dataset="sift1m", passes_key=True):
        """One result directory, `n_rows` rows of `each` seconds.

        `spans` makes it a folded label of that many passes, each entry being
        seconds from the pass starting to its last row *starting* -- which is
        what `aggregate.pass_spans` records, so the estimator adds the last
        row's own duration back on. `None` leaves the label unfolded.
        """
        d = self.results / name
        d.mkdir(parents=True, exist_ok=True)
        gap = each if row_gap is None else row_gap
        rows = [{"id": f"W{i}", "wall_s": each, "when": self._at(i * gap)}
                for i in range(n_rows)]
        (d / "rows.json").write_text(json.dumps(rows))
        meta = {"label": name, "dataset": {"name": dataset}, engine: {"commit": "abc"},
                "started": self._at(0)}
        if spans is not None:
            meta["reps"] = len(spans)
            meta["rep_labels"] = [f"{name}-rep{i + 1}" for i in range(len(spans))]
            if passes_key:
                meta["passes"] = [{"label": f"{name}-rep{i + 1}", "started": self._at(0),
                                   "last_row": self._at(s)} for i, s in enumerate(spans)]
        (d / "run.json").write_text(json.dumps(meta))
        return d

    def test_a_folded_label_is_a_basis(self):
        """The regression. strawmANN: 40 s of rows per pass, fastest pass 50 s
        to its last row +10 s for that row = 60 s, so 1.5x. Qdrant: 80 s of
        rows, fastest 120+20 = 140 s, so 1.75x. A pass is one arm of each."""
        self._label("sm-x", "strawmann", spans=[50, 70, 90])
        self._label("qd-x", "qdrant", each=20.0, spans=[120, 160])
        got = self.fr.estimated_minutes("sift1m", 1)
        self.assertIsNotNone(got, "a folded label is the only record left")
        mins, basis = got
        self.assertAlmostEqual(mins, (60 + 140) / 60)
        self.assertEqual(basis, "qd-x, sm-x", "heaviest arm first")
        self.assertAlmostEqual(self.fr.estimated_minutes("sift1m", 3)[0], 3 * 200 / 60)

    def test_a_folded_label_ignores_its_own_row_stamps(self):
        """Why `passes` is written at all. A folded row's `when` came from
        whichever pass won that row's median, so the span across them is
        nobody's wall clock -- here every row shares one stamp, making it
        shorter than the rows it supposedly contains."""
        self._label("sm-x", "strawmann", spans=[50, 70, 90], row_gap=0)
        self._label("qd-x", "qdrant", spans=[50, 70, 90], row_gap=0)
        mins, _ = self.fr.estimated_minutes("sift1m", 1)
        self.assertAlmostEqual(mins, (60 + 60) / 60)

    def test_a_fold_that_recorded_no_passes_is_not_a_basis(self):
        """Folded by a version that did not write `passes`: the rows are a
        median with no wall clock beside them, so there is nothing to read."""
        self._label("sm-x", "strawmann", spans=[50, 70], passes_key=False)
        self.assertIsNone(self.fr.estimated_minutes("sift1m", 1))

    def test_one_impossible_pass_does_not_cost_the_others(self):
        """40 s of rows cannot fit a 10 s span, so that pass is not describing
        this run -- but dropping the whole label for it would throw away two
        good passes. One engine, so the other arm is assumed to match."""
        self._label("sm-x", "strawmann", spans=[0, 50, 70])
        mins, _ = self.fr.estimated_minutes("sift1m", 1)
        self.assertAlmostEqual(mins, 2 * 60 / 60)

    def test_an_unfolded_label_still_reads_its_row_stamps(self):
        self._label("sm-x", "strawmann")
        mins, basis = self.fr.estimated_minutes("sift1m", 1)
        self.assertAlmostEqual(mins, 2 * 40 / 60)
        self.assertEqual(basis, "sm-x")

    def test_another_corpus_is_not_a_basis(self):
        self._label("sm-x", "strawmann", spans=[50, 70], dataset="dbpedia-openai-1m")
        self.assertIsNone(self.fr.estimated_minutes("sift1m", 1))


if __name__ == "__main__":
    unittest.main()


class StorageLevelTests(unittest.TestCase):
    """`storage on disk` is a level, and the last row is a rewrite in progress.

    Findings 53: the cell is sampled after W11's 200,000-point append, during
    which Qdrant's storage moves 4.03 GiB to 10.11 GiB while its optimiser
    rewrites segments. The same row read 3.47 GiB on a single-pass run of the
    same binary three hours earlier, and the published `rel-0908` page says
    10.7 GiB: a 3x spread in a cell read as a property of the format.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    #: A run in measurement order: two settled rows, then the mutating pair.
    def _rows(self, settled_bytes: int, mutating_bytes: int) -> dict[str, dict]:
        rows = [
            _row("W3", 1000.0, storage_bytes=settled_bytes - 10, rss_peak_bytes=100),
            _row("W13", 2000.0, storage_bytes=settled_bytes, rss_peak_bytes=110),
            _row("W11-steady", 300.0, storage_bytes=settled_bytes + 400,
                 rss_peak_bytes=180, background_pps=2000.0, background_s=10.0),
            _row("W11", 200.0, storage_bytes=mutating_bytes,
                 rss_peak_bytes=900, background_pps=3300.0, background_s=20.0),
        ]
        return {r["id"]: r for r in rows}

    def test_level_comes_from_before_the_mutating_rows(self):
        cmp = self.m["compare"]
        a = self._rows(3_900_000_000, 3_900_000_000)     # does not rewrite
        b = self._rows(3_700_000_000, 10_900_000_000)    # rewrites under append
        out = cmp.storage_and_io("a", "b", a, b, markdown=True)
        line = next(l for l in out if l.startswith("| storage on disk"))
        self.assertIn(self.m["procstat"].human_bytes(3_900_000_000), line)
        self.assertIn(self.m["procstat"].human_bytes(3_700_000_000), line)
        self.assertNotIn(self.m["procstat"].human_bytes(10_900_000_000), line)

    def test_peak_rss_is_read_before_the_writers(self):
        """It was every row, "a peak being a peak", until 0925's Qdrant read
        68.1 GiB on a 54.6 GiB host while W11 rewrote its segments: RSS counts
        a file mapped twice during a rewrite twice, so that peak was never
        memory the machine held."""
        cmp = self.m["compare"]
        a = self._rows(1_000, 1_000)
        out = cmp.storage_and_io("a", "b", a, self._rows(1_000, 1_000), markdown=True)
        line = next(l for l in out if l.startswith("| peak RSS"))
        self.assertIn(self.m["procstat"].human_bytes(110), line)
        self.assertNotIn(self.m["procstat"].human_bytes(900), line)
        self.assertIn("peak RSS are read before W11-steady, W11", out[-1])

    def test_the_note_names_the_excluded_rows_in_run_order(self):
        cmp = self.m["compare"]
        a = self._rows(1_000, 9_000)
        out = cmp.storage_and_io("a", "b", a, a, markdown=True)
        note = out[-1]
        self.assertIn("W11-steady, W11", note)
        self.assertNotIn("W11, W11-steady", note)

    def test_a_run_of_only_mutating_rows_still_reports(self):
        """Fallback: a partial run reports something rather than nothing."""
        cmp = self.m["compare"]
        only = {r["id"]: r for r in [
            _row("W11", 200.0, storage_bytes=7_000, rss_peak_bytes=10,
                 background_pps=3300.0)]}
        out = cmp.storage_and_io("a", "b", only, only, markdown=True)
        line = next(l for l in out if l.startswith("| storage on disk"))
        self.assertIn(self.m["procstat"].human_bytes(7_000), line)

    def test_totals_are_unaffected_and_still_sum_every_row(self):
        """Only the level changed: a total over the run is still the run's."""
        cmp = self.m["compare"]
        rows = {r["id"]: r for r in [
            _row("W13", 2000.0, disk_write_ops=5, storage_bytes=1),
            _row("W11", 200.0, disk_write_ops=7, storage_bytes=2,
                 background_pps=3300.0)]}
        out = cmp.storage_and_io("a", "b", rows, rows, markdown=True)
        line = next(l for l in out if l.startswith("| disk write ops"))
        self.assertIn("12", line)


class EnvironmentFloorTests(unittest.TestCase):
    """A floor measured in another environment bands nothing (findings 46).

    Qdrant's saturating row read 10,313 against a twelve-run aggregate of
    14,716 and was taken for a 30% regression; the aggregate was SMT-off and
    the run SMT-on. strawmANN moved 1.5% across the same change, so the two
    engines are not even affected alike and no correction factor exists.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _label(self, name: str, env: str | None):
        self.fx.label(name, [_row("W4", 1000.0)], good_stamp())
        p = Path(self.tmp.name) / "bench/results" / name / "run.json"
        meta = json.loads(p.read_text())
        if env is not None:
            meta["env_hash"] = env
        meta["engine_comm"] = name
        p.write_text(json.dumps(meta))

    def test_a_floor_from_another_environment_is_refused(self):
        self._label("a", "smt-on")
        self._label("b", "smt-on")
        self.assertFalse(self.m["compare"].env_matches({"env_hash": "smt-off"}, "a", "b"))

    def test_the_same_environment_is_accepted(self):
        self._label("a", "smt-on")
        self._label("b", "smt-on")
        self.assertTrue(self.m["compare"].env_matches({"env_hash": "smt-on"}, "a", "b"))

    def test_one_arm_disagreeing_is_enough_to_refuse(self):
        self._label("a", "smt-on")
        self._label("b", "smt-off")
        self.assertFalse(self.m["compare"].env_matches({"env_hash": "smt-on"}, "a", "b"))

    def test_permissive_where_it_cannot_know(self):
        """A floor predating the stamp, or a run without one, refuses nothing."""
        self._label("a", None)
        self._label("b", None)
        self.assertTrue(self.m["compare"].env_matches({}, "a", "b"))
        self.assertTrue(self.m["compare"].env_matches({"env_hash": "smt-off"}, "a", "b"))

    def test_parity_band_declines_across_environments(self):
        """The guard has to reach the band, not merely exist beside it."""
        self._label("a", "smt-on")
        self._label("b", "smt-on")
        cmp = self.m["compare"]
        floor = {"rsd": {"W4": 0.01}, "arms": ["a", "b"], "env_hash": "smt-off"}
        with mock.patch.object(cmp, "read_noise_meta", return_value=floor):
            self.assertIsNone(cmp.parity_band("W4", "a", "b"))
        floor["env_hash"] = "smt-on"
        with mock.patch.object(cmp, "read_noise_meta", return_value=floor):
            self.assertIsNotNone(cmp.parity_band("W4", "a", "b"))

    def test_the_verdict_tool_refuses_it_too(self):
        """`regression.floor_refusals` applies the same rule to `cmd_compare`."""
        self._label("a", "smt-on")
        self._label("b", "smt-on")
        reg = self.m["regression"]
        noise = reg.Noise({"W4": 0.01}, {"W4": 3}, "src", [], env_hash="smt-off")
        why = reg.floor_refusals(noise, ["a", "b"])
        self.assertTrue(any("environment" in w for w in why), why)
        clean = reg.Noise({"W4": 0.01}, {"W4": 3}, "src", [], env_hash="smt-on")
        self.assertEqual(reg.floor_refusals(clean, ["a", "b"]), [])


class HarnessBoundRowTests(unittest.TestCase):
    """The two rows whose numbers are mostly not the engine.

    Both notes existed and neither reached the row that needed it: the
    unaccounted-time note fired only inside the open-loop branch, and the drift
    note sat below the early returns, so a row with no ratio got no warning.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _joined(self, a_rows, b_rows):
        self.fx.label("a", a_rows, good_stamp(), CONF_T3)
        self.fx.label("b", b_rows, good_stamp(), CONF_T3)
        cmp = self.m["compare"]
        cmp._RECALL_CACHE.clear()
        return {r.id: r for r in cmp.joined("a", "b", cmp.load("a"), cmp.load("b"))}

    #: W13 as `rel-0921` measured it: 137 µs client against an 11 µs server p50.
    W13_LAT: typing.ClassVar[dict] = {"client_p50_us": 136.5, "server_p50_us": 10.9,
                                      "client_p99_us": 230.0}

    def test_a_closed_loop_row_says_how_little_of_it_is_the_engine(self):
        rows = self._joined(
            [_row("W13", 51106.0, ef=None, latency=self.W13_LAT)],
            [_row("W13", 43241.0, ef=None, latency={"client_p50_us": 168.2,
                                                    "server_p50_us": 33.8})])
        note = rows["W13"].note_text
        self.assertIn("not server time", note)
        self.assertIn("92% of this row is the load generator", note)

    def test_a_row_whose_latencies_are_close_says_nothing(self):
        """The note is for rows the harness dominates, not every row."""
        rows = self._joined(
            [_row("W3", 1673.0, latency={"client_p50_us": 613.0, "server_p50_us": 506.0})],
            [_row("W3", 1977.0, latency={"client_p50_us": 511.0, "server_p50_us": 411.0})])
        self.assertNotIn("not server time", rows["W3"].note_text)

    def test_a_row_with_no_ratio_still_reports_drift(self):
        """W11 cannot have a recall join, and drifted +25% across three passes."""
        rows = self._joined(
            [_row("W11", 1794.0, ef=None, rep_drift=0.252, recall_joinable=False)],
            [_row("W11", 1773.0, ef=None, recall_joinable=False)])
        note = rows["W11"].note_text
        self.assertIn("+25%", note)
        self.assertIn("monotonically", note)
        self.assertIn("median is a trend's midpoint", note)

    def test_drift_still_refuses_a_ratio_it_would_have_banded(self):
        """The refusal survives the note moving above the early returns."""
        rows = self._joined([_row("W4", 1000.0, rep_drift=0.30)], [_row("W4", 900.0)])
        self.assertIn("monotonically", rows["W4"].note_text)
        self.assertIsNone(re.fullmatch(r"[\d.]+x", rows["W4"].ratio or ""),
                          f"drifted row published a ratio: {rows['W4'].ratio!r}")

    def test_a_steady_row_is_not_called_drifted(self):
        rows = self._joined([_row("W4", 1000.0)], [_row("W4", 900.0)])
        self.assertNotIn("monotonically", rows["W4"].note_text)


class FoldedOverlapTests(unittest.TestCase):
    def test_the_note_carries_the_folded_figure(self):
        compare = importlib.import_module("compare")
        row = {"write_overlap_pct": 81.0,
               "notes": "concurrent append; write overlap 82%; append 500 points/s"}
        self.assertIn("write overlap 81%", compare.with_folded_overlap(row["notes"], row))
        self.assertEqual(compare.with_folded_overlap("x", {"write_overlap_pct": None}), "x")


class RescoreCauseTests(unittest.TestCase):
    """What a quantized row's unequal recall is blamed on."""

    def _cause(self, ov_a, ov_b, ef=128):
        compare = importlib.import_module("compare")
        row = object.__new__(compare.Row)
        row.ra = {"quantization_rescore": True, "quantization_oversampling": ov_a, "ef": ef}
        row.rb = {"quantization_rescore": True, "quantization_oversampling": ov_b, "ef": ef}
        return row._rescore_cause()

    def test_matched_pools_leave_the_encoders_to_blame(self):
        """Under `pool` both engines rescore `ef` candidates; saying the pools
        differ would send the reader after the one cause that was removed."""
        self.assertEqual(self._cause(12.8, 12.8), "; the rescore pools match, so the encoders differ")
        self.assertEqual(self._cause(3.2, 3.2, ef=32), "; the rescore pools match, so the encoders differ")
        self.assertEqual(self._cause(4.0, 4.0), "; rescore pools differ (decisions §5)")
        self.assertEqual(self._cause(2.0, None), "; rescore pools differ (decisions §5)")


class ComparisonDocStubTests(unittest.TestCase):
    def test_a_new_comparison_doc_names_no_qdrant_version(self):
        """The stub said "Qdrant 1.19.0" above 0925's 1.19.2-dev rows."""
        compare = importlib.import_module("compare")
        doc = compare.new_comparison_doc("dbpedia-openai-1m")
        self.assertEqual(doc.splitlines()[0], "# strawmann vs Qdrant: dbpedia-openai-1m")
        self.assertNotRegex(doc.splitlines()[0], r"\d+\.\d+")
        begin, end = compare.markers("compare-full")
        self.assertIn(f"{begin}\n{end}\n", doc)


class DecompositionTests(unittest.TestCase):
    """A ratio said as the three things that produce it.

    Throughput is `cores_busy * frequency / cycles_per_query`, so a ratio of two
    throughputs factors exactly. W4's 2.20x is 1.57x work per query, 1.35x core
    occupancy and 1.03x clock, and a reader who takes the whole of it for
    per-query efficiency has misread a third.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _perf(cores, ghz, cyc_per_q, n=N, dur=2.0):
        """A row whose counters say exactly these three things."""
        return {"duration_s": dur, "n_queries": n,
                "perf_task_clock_s": cores * dur,
                "perf_cycles": cyc_per_q * n,
                "perf_ref_hz": 2e9}

    def _row_note(self, a_extra, b_extra, qa=22784.0, qb=10351.0):
        # W4 needs recall on both sides or it refuses before reaching any of
        # this; equal recall, so the only thing under test is the arithmetic.
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.9873))]
        self.fx.label("a", [_row("W4", qa, **a_extra)], good_stamp(), CONF_T3, sw)
        self.fx.label("b", [_row("W4", qb, **b_extra)], good_stamp(), CONF_T3, sw)
        cmp = self.m["compare"]
        cmp._RECALL_CACHE.clear()
        rows = {r.id: r for r in cmp.joined("a", "b", cmp.load("a"), cmp.load("b"))}
        return rows["W4"].note_text

    def test_the_three_factors_multiply_to_the_ratio(self):
        """`rel-0921`'s W4, to the digits the page prints."""
        a = self._perf(cores=7.50, ghz=1.97e9, cyc_per_q=649_557)
        b = self._perf(cores=5.55, ghz=1.91e9, cyc_per_q=1_022_894)
        # The cycles the counters imply have to agree with the frequency and the
        # occupancy, or the fixture is not a measurement of anything.
        a["perf_cycles"] = 1.97e9 * a["perf_task_clock_s"]
        b["perf_cycles"] = 1.91e9 * b["perf_task_clock_s"]
        a["n_queries"] = a["perf_cycles"] / 649_557
        b["n_queries"] = b["perf_cycles"] / 1_022_894
        note = self._row_note(a, b)
        self.assertIn("1.57x less work per query", note)
        self.assertIn("1.35x cores busy", note)
        self.assertIn("1.03x clock", note)
        self.assertIn("(7.50 against 5.55)", note)

    def test_the_clock_is_the_reference_counters_not_cycles_over_task_clock(self):
        """0925's W0 read "x 1.08x clock" beside a GHz column of 2.01 on both
        engines: `cycles / task_clock` was 1.735 against 1.604 GHz because the
        cycle counter saw a different share of on-CPU time on each."""
        def row(cores, counted_ghz, cyc_per_q):
            r = self._perf(cores, counted_ghz, cyc_per_q)
            r["perf_cycles"] = counted_ghz * r["perf_task_clock_s"]
            r["n_queries"] = r["perf_cycles"] / cyc_per_q
            r["perf_ref_hz"] = 2.0e9
            # Both at 2.006 GHz by the reference counter.
            r["perf_ref_cycles"] = r["perf_cycles"] * 2.0e9 / 2.006e9
            return r
        a, b = row(0.67, 1.735e9, 400_000), row(1.05, 1.604e9, 784_000)
        note = self._row_note(a, b, qa=4572.0, qb=3380.0)
        self.assertIn("1.00x clock", note)
        self.assertIn("1.08x counted on-CPU share", note)
        # The factors still multiply to the throughput ratio the counters imply.
        qps = lambda r: r["n_queries"] / r["duration_s"]
        self.assertIn(f"[{qps(a) / qps(b):.2f}x is", note)
        # Where the two shares agree the fourth term is not printed.
        a, b = row(7.15, 2.0e9, 4_030_000), row(4.46, 2.0e9, 3_410_000)
        note = self._row_note(a, b, qa=3552.0, qb=2580.0)
        self.assertIn("1.00x clock", note)
        self.assertNotIn("counted", note)

    def test_silent_when_the_engines_are_occupancy_matched(self):
        """The ordinary case: the ratio is per-query efficiency and says so by
        carrying no note at all."""
        note = self._row_note(self._perf(6.0, 2e9, 500_000),
                              self._perf(6.1, 2e9, 1_000_000))
        self.assertNotIn("cores busy", note)

    def test_fires_once_the_gap_passes_a_fifth(self):
        note = self._row_note(self._perf(7.5, 2e9, 500_000),
                              self._perf(5.5, 2e9, 1_000_000))
        self.assertIn("cores busy", note)

    def test_silent_without_counters(self):
        """A run measured without `--perf` states nothing about occupancy."""
        self.assertNotIn("cores busy", self._row_note({}, {}))

    def test_silent_when_the_row_has_no_ratio(self):
        """A decomposition of a number the page refuses is furniture."""
        a = self._perf(7.5, 2e9, 500_000)
        b = self._perf(5.5, 2e9, 1_000_000)
        a["foreign"] = "rustc(120%)"
        note = self._row_note(a, b)
        self.assertNotIn("cores busy", note)
        self.assertIn("contaminated", note)


class MatchedLineTests(unittest.TestCase):
    """The front page carries the comparison §7.4 actually asks for.

    Every ratio in the header block is at equal `ef`, which is not equal work.
    The frontier lived only on the report page, so the front page led with 2.20x
    and nothing told a reader the licensed number reads lower and is not flat.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_states_the_best_and_the_top_of_the_curve(self):
        cmp = self.m["compare"]
        # A licensed pair on disk: `matched_line` checks the licence before it
        # asks `report_data` for anything, so a mock alone cannot exercise it.
        fx = Fixture(Path(self.tmp.name))
        for lab in ("a", "b"):
            fx.label(lab, [_row("W10-ef128", 1.0)], good_stamp(), CONF_T3)
        anchors = [{"recall": 0.9075, "ratio": 2.04}, {"recall": 0.9626, "ratio": 2.12},
                   {"recall": 0.9980, "ratio": 1.48}]

        class FakeRD:
            @staticmethod
            def load_run(label): return label
            @staticmethod
            def frontier_points(run, bfb_only=False): return run
            @staticmethod
            def matched_ratios(pa, pb): return anchors

        with mock.patch.dict("sys.modules", {"report_data": FakeRD}):
            line = cmp.matched_line("a", "b")
        self.assertIn("2.12x at recall 0.9626", line)
        self.assertIn("falling to 1.48x at 0.9980", line)
        self.assertIn("3 anchors", line)

    def test_a_pair_with_no_frontier_drops_the_line(self):
        """A T3 refusal or a missing sweep loses the line, never the block."""
        cmp = self.m["compare"]

        class Empty:
            @staticmethod
            def load_run(label): return label
            @staticmethod
            def frontier_points(run, bfb_only=False): return []
            @staticmethod
            def matched_ratios(pa, pb): return []

        with mock.patch.dict("sys.modules", {"report_data": Empty}):
            self.assertIsNone(cmp.matched_line("a", "b"))

    def test_a_label_with_no_rows_is_not_an_error(self):
        """CI's case, and it failed the build once.

        `report_data.load_run` raises `SystemExit` for a label with no
        `rows.json`, which is a CLI saying "run the benchmark first" and is not
        an `Exception` subclass, so `except Exception` let it out. CI has no
        results at all, `--check-readme` runs there against the default labels,
        and an optional line took the build down with it.
        """
        cmp = self.m["compare"]
        # The real loader against the real empty temp root, not a mock: the
        # mock is what would have missed this.
        self.assertIsNone(cmp.matched_line("strawmann", "qdrant"))

    def test_an_import_failure_is_not_an_error(self):
        """`report_data` imports compare, so this import is deferred; a stdlib
        caller without pandas still gets its block."""
        cmp = self.m["compare"]

        class Boom:
            @staticmethod
            def load_run(label): raise RuntimeError("no pandas")

        with mock.patch.dict("sys.modules", {"report_data": Boom}):
            self.assertIsNone(cmp.matched_line("a", "b"))


class MatchedRecallGateTests(unittest.TestCase):
    """What a T3 failure does to the matched-recall table.

    T3 says the two engines' ANN recall differs at a fixed `ef`. That is the
    condition the matched-recall table corrects for: it does not assume equal
    recall, it constructs it at each recall one engine actually reached. Gating
    the table on T3 withheld it exactly when it was the only comparison left,
    and dbpedia-openai-1m published nothing while 2.14x, 2.03x and 1.80x sat
    computable on disk.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))
        self.m = _reload(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _runs(self, conf):
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        rows = [_row("W10-ef128", 20000.0), _row("W4", 22000.0)]
        for lab in ("a", "b"):
            self.fx.label(lab, rows, good_stamp(), conf, sw)
        rd = self.m["report_data"]
        return [rd.load_run("a"), rd.load_run("b")]

    def test_a_licensed_pair_is_shown_with_no_banner(self):
        rd = self.m["report_data"]
        runs = self._runs(CONF_T3)
        self.assertEqual(rd.matched_recall_refusal(runs), "")
        self.assertEqual(rd.matched_recall_caveat(runs), "")

    def test_a_t3_failure_alone_is_shown_under_a_banner(self):
        rd = self.m["report_data"]
        runs = self._runs(CONF_T3_FAILED)
        self.assertEqual(rd.matched_recall_refusal(runs), "")
        caveat = rd.matched_recall_caveat(runs)
        self.assertIn("NOT A LICENSED COMPARATIVE CLAIM", caveat)
        self.assertIn("constructs it", caveat)

    def test_scores_disagreeing_still_withholds(self):
        """T1 failed: no interpolation of either curve means anything."""
        rd = self.m["report_data"]
        runs = self._runs(CONF_T1_FAILED)
        self.assertNotEqual(rd.matched_recall_refusal(runs), "")
        self.assertEqual(rd.matched_recall_caveat(runs), "")

    def test_a_conformance_row_without_tiers_keeps_the_old_refusal(self):
        """Rows measured before the differ recorded per-tier verdicts cannot
        claim T1 and T2 passed, so they keep the conservative behaviour."""
        rd = self.m["report_data"]
        runs = self._runs(CONF_T2)
        self.assertNotEqual(rd.matched_recall_refusal(runs), "")

    def test_the_front_page_line_never_carries_an_unlicensed_frontier(self):
        """The page shows it under a banner; README and the docs do not show it."""
        self._runs(CONF_T3_FAILED)
        self.assertIsNone(self.m["compare"].matched_line("a", "b"))
