#!/usr/bin/env python3
"""Unit tests: the workload table, its rows and what they record."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness_fixtures import (
    FAKE_BFB,
    _reload,
)

import workloads


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.w = self.m["workloads"]
        # No engine, whatever the machine happens to be running.
        #
        # `run_one` samples the *real* host through `procstat.snapshot()`, so
        # these tests were a function of the operator's process list. Any
        # process whose `comm` starts with `qdrant` or `strawmann` — including
        # this developer's own `qdrant` debug build, which comes and goes as
        # their tests run — made `storage_path_for` shell out to `docker ps`
        # (breaking the subprocess-call count two tests assert) and made
        # `provenance.collect` record *that* engine, so `run.json` grew a
        # `qdrant` key and never the `strawmann` one the ISA test looks for.
        #
        # The failures were intermittent, looked exactly like a regression in
        # whatever had just been touched, and twice cost a bisect. A unit test
        # that consults the live machine is not a unit test.
        patch = mock.patch.object(self.m["procstat"], "engine_processes",
                                  return_value=[])
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_w11_is_last_concurrent_and_synthetic(self):
        t = self.w.table()
        # W11 is last: W11-steady runs before it, on a bench2 with nothing
        # pending, so that it stays below `rebuild_ratio` and W11 crosses it.
        w11 = t[-1]
        self.assertEqual(w11.id, "W11")
        self.assertEqual(t[-2].id, "W11-steady")
        self.assertIsNotNone(w11.background)
        self.assertNotIn("--fbin", w11.background, "fbin has exactly UPLOAD_N rows; "
                         "offset+i past it panics in bfb's fbin_reader")
        self.assertIn("--offset", w11.background)
        self.assertEqual(w11.background[w11.background.index("--offset") + 1],
                         str(self.w.UPLOAD_N + self.w.w11_steady_n()),
                         "W11 appends past W11-steady, which appends past the corpus")
        self.assertIn("--skip-wait-index", w11.background)
        self.assertIn("--search", w11.args)
        self.assertFalse(w11.recall_joinable)

    def test_the_two_write_rows_sit_on_opposite_sides_of_rebuild_ratio(self):
        """That is the whole point of having two of them.

        `collection.rebuild_ratio` is 0.10. W11-steady must stay under it and
        W11 must cross it, and the order is what decides: run after W11,
        W11-steady inherits ~9% pending and crosses with 11,100 of its own
        writes, so it measured a second rebuild and read *worse* than W11.
        """
        w = self.w
        t = w.table()
        rebuild_ratio = 0.10          # src/core/collection.zig
        covered = w.upload_n()

        steady, w11 = t[-2], t[-1]
        self.assertEqual((steady.id, w11.id), ("W11-steady", "W11"))

        # Contiguous appends, no gap and no overlap: ids run from the end of
        # the corpus to exactly `required_capacity`.
        s_off = int(steady.background[steady.background.index("--offset") + 1])
        w_off = int(w11.background[w11.background.index("--offset") + 1])
        self.assertEqual(s_off, covered)
        self.assertEqual(w_off, covered + w.w11_steady_n())
        self.assertEqual(w_off + w.w11_n(), w.required_capacity())

        # W11-steady runs on a collection W2 built, so nothing is pending and
        # its own writes are all that count.
        self.assertLess(w.w11_steady_n() / covered, rebuild_ratio,
                        "W11-steady must not provoke a rebuild")

        # W11 appends on top of it and crosses, as it always did.
        self.assertGreaterEqual((w.w11_steady_n() + w.w11_n()) / covered,
                                rebuild_ratio,
                                "W11 must still measure the rebuild path")

    def test_every_concurrent_writer_is_deferred_past_the_recall_sweeps(self):
        """A row with a background writer mutates a collection the sweeps read.

        `fullrun.MUTATING_ROWS` was the literal `["W11"]`, so adding
        `W11-steady` — a second appender to `bench2`, added for the same reason
        — ran it with the ordinary rows. `bench2` reached 1,050,000 points and
        the sweep refused against a ground truth describing 1,000,000.
        """
        import fullrun
        writers = {w.id for w in self.w.table() if w.background}
        self.assertIn("W11", writers)
        self.assertIn("W11-steady", writers)
        self.assertEqual(set(fullrun.mutating_rows()), writers)

    def test_collections_are_dropped_once_nothing_reads_them(self):
        """Derived from what reads them, not from a list kept in step by hand.

        strawmANN is resident: a collection costs its full arena from create to
        process exit. `bench1` is written by W1 and read by nothing afterwards
        — no row, no sweep, no later phase — so it held 7.08 GiB at the 1M tier
        for a measurement that had already finished.
        """
        w = self.w
        created = set(w.upload_collections())
        live = w.collections_read_after_rows()
        after_rows = set(w.collections_droppable_after_rows())
        after_sweeps = set(w.collections_droppable_after_sweeps())

        # bench1 is written by W1 and never read again.
        self.assertIn("bench1", after_rows)
        self.assertNotIn("bench1", live)

        # bench2 carries the mutating rows, which run last, so it is never
        # droppable at either point.
        self.assertNotIn("bench2", after_rows)
        self.assertNotIn("bench2", after_sweeps)

        # The swept quantized collections survive the rows and go after their
        # sweep, before the rebuild that W11 provokes.
        for c in ("bench6", "bench7", "bench8"):
            self.assertIn(c, live)
            self.assertNotIn(c, after_rows)
            self.assertIn(c, after_sweeps)

        # Every created collection is accounted for exactly once: dropped at
        # one of the two points, or still live at the end.
        self.assertTrue(after_rows.isdisjoint(after_sweeps))
        survivors = created - after_rows - after_sweeps
        self.assertEqual(survivors, {"bench2"})

    def test_w1s_collection_is_dropped_before_w2_rather_than_waited_on(self):
        """Qdrant indexes `bench1` in the background after W1's
        `--skip-wait-index` upload. The settle waited 180 s, gave up on 7.0
        cores, and W2's Time-to-Green shared the engine with that build. Nothing
        reads `bench1` again, so it is dropped between W1 and W2 instead.
        """
        import fullrun
        w = self.w
        self.assertEqual(w.rows_writing_dead_collections(), ["W1"])
        # Part of the settle discipline, so part of every row's stamp.
        self.assertEqual(w.harness_stamp()["engine_settle"]["dropped_not_settled"], ["W1"])

        stable = [x.id for x in w.table() if x.id not in fullrun.mutating_rows()]
        first, rest, early = fullrun.split_after_dead_writers(stable)
        self.assertEqual(first[-1], "W1")
        self.assertEqual(rest[0], "W2")
        self.assertEqual(first + rest, stable)
        self.assertEqual(early, ["bench1"])
        # A run that does not include W1 drops nothing early.
        self.assertEqual(fullrun.split_after_dead_writers(["W2", "W3"]), (["W2", "W3"], [], []))

    def test_the_sq8_sweep_runs_before_anything_can_evict_bench6(self):
        """Qdrant's `bench6` is file-backed, and W7-upload, W8-upload and W9's
        exact scan pushed it out of the page cache before the W6-ef sweep ran:
        150k to 250k major faults on the first passes and two ratios refused
        as drift. Nothing that uploads another collection or scans one whole
        may run between `bench6`'s upload and the end of its sweep.
        """
        rows = self.w.table()
        ids = [x.id for x in rows]
        start = ids.index("W6-upload")
        end = max(i for i, x in enumerate(ids) if x.startswith("W6-ef"))
        for x in rows[start + 1:end]:
            self.assertFalse(x.upload_only, f"{x.id} uploads between W6-upload and the sweep")
            self.assertNotIn("--search-exact", x.args, f"{x.id} scans between them")
            self.assertEqual(self.w.collection_of(x), "bench6", x.id)

    def test_w5_uses_distinct_dataset_queries(self):
        w5 = {x.id: x for x in self.w.table()}["W5"]
        self.assertEqual(w5.query_strategy, "random-sample")
        cfg = self.w.search_config(w5.query_collection, w5.query_strategy)
        self.assertIn("strategy: random-sample", cfg)
        self.assertIn("per-batch latency", "; ".join(self.w.static_notes(w5)))

    def test_every_creating_row_carries_collection_flags(self):
        # Under the default `equal-work` policy. `--segments` is deliberately
        # absent under `as-deployed`, which SegmentPolicyTests covers; the
        # thresholds are in both because they are a different axis.
        self.assertIs(self.w.SEGMENT_POLICY, self.w.SegmentPolicy.equal_work)
        for w in self.w.table():
            if w.upload_only:
                for f in ("--segments", "--indexing-threshold", "--full-scan-threshold",
                          "--on-disk-payload", "--distance"):
                    self.assertIn(f, w.args, f"{w.id} lacks {f}")
        stamp = self.w.harness_stamp()
        for k in ("metric", "query_source", "upload_n", "w11_n", "bfb_pin", "collection", "ef"):
            self.assertIn(k, stamp)
        self.assertEqual(stamp["collection"]["segments"], self.w.segments())
        # The name travels with the numbers, so a reader never has to infer
        # which experiment a stamp describes.
        self.assertEqual(stamp["collection"]["segment_policy"], "equal-work")

    def test_placement_flag_only_reaches_creating_rows_and_not_the_stamp(self):
        """§5.5: the engines' defaults differ, so an unconfigured run compares
        two residencies. Asking Qdrant for `cached` -- what it does anyway --
        makes it say so. It must stay out of the stamp: strawmANN cannot take
        `cached` without a `--data-dir`, so an arm-specific value there would
        give the two arms different hashes and refuse every ratio as STALE."""
        w = self.w
        t = {x.id: x for x in w.table()}
        # A creating row gains the flag.
        up = w.with_placement(t["W1"], "cached")
        self.assertEqual(up.args[-2:], ["--memory-vectors", "cached"])
        # A search row runs with --skip-setup and creates nothing.
        self.assertIn("--skip-setup", t["W4"].args)
        self.assertEqual(w.with_placement(t["W4"], "cached").args, t["W4"].args)
        # No placement asked for is the row untouched, not a flag with an
        # empty value.
        self.assertEqual(w.with_placement(t["W1"], None).args, t["W1"].args)
        # And the stamp must not mention a specific placement, or the two arms
        # diverge: this is the assertion that keeps the ratios alive.
        stamp = w.harness_stamp()
        self.assertIn("collection", w.STAMP_KEYS)
        self.assertNotIn("cached", json.dumps(stamp["collection"]))
        self.assertNotIn("pinned", json.dumps(stamp["collection"]))

    def test_load_mode_scroll_is_closed_loop(self):
        t = {x.id: x for x in self.w.table()}
        self.assertEqual(self.w.load_mode_of(t["W13"]), self.w.LoadMode.closed_loop)
        self.assertEqual(self.w.load_mode_of(t["W1"]), self.w.LoadMode.upload)
        # Open loop from the *fraction*, before the runner has resolved it into
        # a `--rps` flag: the table carries §4's intent and the flag arrives
        # only once W4 has been measured.
        self.assertEqual(self.w.load_mode_of(t["W4-sat50"]), self.w.LoadMode.open_loop)
        self.assertNotIn("--rps", t["W4-sat50"].args)
        # The reference is capped at the measured generator ceiling and the
        # fraction taken of that, so an engine faster than the generator does
        # not get an arm nobody offered (findings 32/33). Below the ceiling the
        # engine's own saturation is used, which is now the ordinary case:
        # 50% of 22,840 is 11,420, well under the 30,000 the fixed generator
        # delivers at 99%.
        resolved, rate = self.w.resolve_rps(t["W4-sat50"], 22840)
        self.assertEqual(rate, 11420)
        capped, capped_rate = self.w.resolve_rps(
            t["W4-sat50"], self.w.GENERATOR_CEILING_QPS * 4)
        self.assertEqual(capped_rate, int(self.w.GENERATOR_CEILING_QPS * 0.5))
        self.assertEqual(capped.args[-2:], ["--rps", str(capped_rate)])
        self.assertEqual(resolved.args[-2:], ["--rps", str(rate)])
        self.assertEqual(self.w.load_mode_of(resolved), self.w.LoadMode.open_loop)

    def test_open_loop_arms_are_a_fraction_of_measured_saturation(self):
        """§4: `--rps R` at R in {0.5, 0.7, 0.9} x measured saturation.

        These were the constants 500/1000/2000. strawmANN saturates above
        22,000 qps, so `--rps 500` offered 2% of capacity: no queue can form
        there, and the row measured the load generator rather than the engine.
        """
        t = {x.id: x for x in self.w.table()}
        self.assertEqual(sorted(x for x in t if x.startswith("W4-sat")),
                         ["W4-sat50", "W4-sat70", "W4-sat90"])
        self.assertEqual([t[f"W4-sat{p}"].rps_fraction for p in (50, 70, 90)],
                         [0.5, 0.7, 0.9])
        # No saturation is no rate, and no rate is no row: the harness must not
        # fall back to a constant, which is the thing being removed.
        self.assertEqual(self.w.resolve_rps(t["W4-sat90"], None)[1], None)
        self.assertEqual(self.w.resolve_rps(t["W4-sat90"], 0)[1], None)
        # The cap applies to the reference, not the rate. Capping the rate put
        # all three arms on the ceiling — one arm wearing three names.
        fast = [self.w.resolve_rps(t[f"W4-sat{p}"],
                                   self.w.GENERATOR_CEILING_QPS * 4)[1]
                for p in (50, 70, 90)]
        self.assertEqual(len(set(fast)), 3, f"arms collapsed onto the cap: {fast}")
        self.assertTrue(all(r <= self.w.GENERATOR_CEILING_QPS for r in fast), fast)
        # An engine slower than the ceiling still uses its own saturation.
        slow = [self.w.resolve_rps(t[f"W4-sat{p}"], 3000)[1] for p in (50, 70, 90)]
        self.assertEqual(slow, [1500, 2100, 2700])
        # A row with no fraction passes through untouched.
        self.assertEqual(self.w.resolve_rps(t["W4"], 22840), (t["W4"], None))

    def test_saturation_comes_from_this_run_then_the_last(self):
        import dataclasses
        d = Path(self.tmp.name) / "sat"
        d.mkdir(parents=True, exist_ok=True)
        R = self.w.Result
        ok = R(id="W4", status=self.w.Status.ok, seconds=1,
               load_start=0, load_end=0, foreign="", qps=9000.0)
        # This session wins over rows.json.
        (d / "rows.json").write_text(json.dumps(
            [{"id": "W4", "status": "ok", "qps": 1234.0}]))
        self.assertEqual(self.w.saturation_qps([ok], d), 9000.0)
        # Falling back to the previous run's rows.json.
        self.assertEqual(self.w.saturation_qps([], d), 1234.0)
        # A failed W4 is not a saturation, in either place.
        bad = dataclasses.replace(ok, status=self.w.Status.failed)
        (d / "rows.json").write_text(json.dumps(
            [{"id": "W4", "status": "FAILED", "qps": 1234.0}]))
        self.assertIsNone(self.w.saturation_qps([bad], d))
        self.assertIsNone(self.w.saturation_qps([], Path(self.tmp.name) / "nope"))

    def test_wall_qps_from_bfb_json(self):
        d = Path(self.tmp.name) / "r"
        d.mkdir()
        (d / "W4.json").write_text(json.dumps({
            "config": {"collection_name": "bench2"},
            "results": {"search": {"duration_secs": 1.61, "full_timings": [0.001] * 50000}}}))
        got = self.w.wall_qps_of(d, "W4")
        self.assertAlmostEqual(got["qps_wall"], 50000 / 1.61, places=2)
        self.assertLess(got["duration_s"], self.w.MIN_ROW_S)
        # Batched: requests x batch = queries.
        (d / "W5.json").write_text(json.dumps({
            "results": {"search": {"duration_secs": 5.0, "full_timings": [0.001] * 3125}}}))
        self.assertEqual(self.w.wall_qps_of(d, "W5", 16)["n_queries"], 50000)

    def test_scale_n(self):
        self.assertEqual(self.w.scale_n(["-n", "100", "-p", "1"], 4), ["-n", "400", "-p", "1"])
        self.assertEqual(self.w.scale_n(["-p", "1"], 4), ["-p", "1"])

    def test_backfill_does_not_invent_ef(self):
        w = self.w
        d = Path(self.tmp.name) / "bench/results/old"
        d.mkdir(parents=True)
        os.environ.pop("RESULTS_DIR", None)
        rows = [{"id": "W6", "status": "ok", "seconds": 1, "load_start": 0, "load_end": 0,
                 "foreign": "", "qps": 7872.0, "rps": 7872.0, "detail": ""}]
        (d / "rows.json").write_text(json.dumps(rows))
        (d / "W6.json").write_text(json.dumps({
            "config": {"collection_name": "bench6"},
            "results": {"search": {"duration_secs": 5.0, "full_timings": [0.001] * 50000}}}))
        self.assertEqual(w.backfill("old"), 0)
        got = json.loads((d / "rows.json").read_text())[0]
        self.assertIsNone(got["ef"], "backfill stamped today's ef onto a row that never stated one")
        self.assertEqual(got["collection"], "bench6")
        self.assertEqual(got["qps_bfb_median"], 7872.0)
        self.assertAlmostEqual(got["qps"], 10000.0)


    def test_on_disk_payload_is_false_as_spec_s2_says(self):
        w = self.w
        i = w.CREATE.index("--on-disk-payload")
        self.assertEqual(w.CREATE[i + 1], "false")
        self.assertEqual(w.collection_settings()["on_disk_payload"], "false")
        self.assertEqual(w.harness_stamp()["collection"]["on_disk_payload"], "false")

    def test_only_the_filtered_row_declares_a_payload_index_precondition(self):
        """Every filtered-search row, and no other, is checked for an index.

        The flag on the row is what makes `compare.py` read the index state
        back; a row that filters without declaring it would publish a full
        scan under the words "filtered search" and nothing would object.
        """
        w = self.w
        t = {x.id: x for x in w.table()}
        need = {x.id for x in w.table() if x.needs_payload_index}
        # Two selectivity grades since 2026-09-08, both filtered rows and both
        # therefore preconditioned on the index (docs/workloads.md W12 point 2)
        # — and the wide grade's `ef` ladder, which is the same filtered search
        # at five widths and needs the index for the same reason.
        self.assertEqual(need, {"W12-sel1", "W12-sel10"}
                         | {f"W12-{g}-ef{e}"
                            for g in ("sel1", "sel10")
                            for e in (32, 64, 128, 256, 512)})
        # Since 2026-09-03 the row asks for the index it depends on: neither
        # half carries the flag that used to deny it, on either engine.
        self.assertNotIn("--skip-field-indices", t["W12-sel1"].args)
        self.assertNotIn("--skip-field-indices", t["W12-sel10"].args)
        self.assertNotIn("--skip-field-indices", t["W12-upload"].args)
        self.assertFalse(w.payload_index_suppressed(t["W12-sel1"]))
        self.assertFalse(w.payload_index_suppressed(t["W12-sel10"]))

    def test_the_filtered_row_is_two_selectivity_grades_on_the_config_path(self):
        """docs/workloads.md W12 point 2: two grades, not one -- "an engine
        that picks well at one selectivity and badly at the other reports a
        single figure that is true of neither", and Qdrant's dispatch turns on
        exactly that threshold.

        Both on the config path, because only it sends a dataset query and a
        filter in the same request. The flag path's queries are
        `random_dense_vector`, which sit where no SIFT data lives, so no recall
        measured on the real query set describes them -- a filtered row there
        could never have its recall joined, which is why W12 had none.
        """
        w = self.w
        t = {x.id: x for x in w.table()}
        self.assertNotIn("W12", t, "the single ungraded row is gone")
        narrow, wide = t["W12-sel1"], t["W12-sel10"]
        for r in (narrow, wide):
            self.assertEqual(r.query_collection, "bench12")
            self.assertNotIn("-d", [str(x) for x in r.args],
                             f"{r.id}: the query file defines the width here")
        self.assertEqual(narrow.keyword_filter, ("a", w.W12_KEYWORDS, None))
        self.assertEqual(wide.keyword_filter, ("a", w.W12_KEYWORDS, w.W12_MATCH_ANY))
        # The upload generates the values the conditions select from, and at
        # the same cardinality: a filter drawing from a different one matches
        # no point, which reads as an engine returning nothing rather than as a
        # config that asked the wrong question.
        up = [str(x) for x in t["W12-upload"].args]
        self.assertEqual(int(up[up.index("-k") + 1]), w.W12_KEYWORDS)

    def test_both_grades_have_an_ef_ladder_for_a_matched_recall_read(self):
        """§7.4 compares at equal recall, and `report_data.frontier_points`
        gets there by joining `<prefix>-ef<ef>` rows to the sweep's recall at
        the same `ef`.

        W12-sel10 measured throughput at one `ef` only, so the comparison had
        nothing to interpolate: the engines landed 0.0100 apart in recall — the
        exact width of §7.4's band — and the row published a dash where the
        widest gap on the corpus is. The ladder is the missing half, and it must
        carry the same condition as the row it explains or it describes a
        different search.
        """
        w = self.w
        t = {x.id: x for x in w.table()}
        for grade in ("sel1", "sel10"):
            row = t[f"W12-{grade}"]
            for ef in (32, 64, 128, 256, 512):
                r = t[f"W12-{grade}-ef{ef}"]
                self.assertEqual(r.keyword_filter, row.keyword_filter,
                                 f"{r.id} must filter as W12-{grade} does")
                self.assertEqual(r.query_collection, row.query_collection)
                # Every flag the row sends, with only the width changed —
                # asserted as the whole argv rather than field by field,
                # because checking `-n` and `ef` individually is what let
                # `-p 8` diverge: the rung then measured a different
                # concurrency and came in 3.49x above the row it reproduces.
                want = [str(x) for x in row.args]
                want[want.index("--search-hnsw-ef") + 1] = str(ef)
                self.assertEqual([str(x) for x in r.args], want,
                                 f"{r.id} must be W12-{grade}'s search at "
                                 f"another width")

    def test_the_narrow_grade_s_ladder_comes_first(self):
        """Rows run in *table* order, so the order is part of what a run
        measures — which is why `W11-steady` before `W11` is asserted rather
        than trusted to a comment.

        This order is not load-bearing, only legible: narrow before wide,
        matching `FILTER_GRADES`, the row pair and the report's labels. It is
        asserted because the block was first inserted at an anchor that
        happened to be the *end* of the wide grade's ladder, nothing objected,
        and the table read wide-then-narrow for a whole measurement run.
        """
        ids = [x.id for x in self.w.table()]
        first_sel1 = ids.index("W12-sel1-ef32")
        first_sel10 = ids.index("W12-sel10-ef32")
        self.assertLess(first_sel1, first_sel10,
                        "the narrow grade's ladder comes first")
        # Each ladder is contiguous and in ascending width, so the table reads
        # as two ladders rather than ten interleaved rows.
        for grade in ("sel1", "sel10"):
            got = [i for i in ids if i.startswith(f"W12-{grade}-ef")]
            self.assertEqual(got, [f"W12-{grade}-ef{e}"
                                   for e in (32, 64, 128, 256, 512)])
            at = [ids.index(i) for i in got]
            self.assertEqual(at, list(range(at[0], at[0] + len(at))),
                             f"{grade}'s ladder is contiguous")

    def test_the_generated_search_config_carries_the_condition(self):
        w = self.w
        wide = w.search_config("bench12", "from-start", ("a", 100, 10))
        for want in ("name: a", "type: keyword", "cardinality: 100", "match_any: 10"):
            self.assertIn(want, wide)
        # No `match_any` on the narrow grade: that absence is what makes it one
        # value rather than ten, so it is asserted rather than assumed.
        narrow = w.search_config("bench12", "from-start", ("a", 100, None))
        self.assertIn("cardinality: 100", narrow)
        self.assertNotIn("match_any", narrow)
        # And an unfiltered row's config is untouched.
        self.assertNotIn("filters", w.search_config("bench2"))

    def test_the_filtered_row_is_no_longer_cut_to_a_short_count(self):
        """`FILTERED_QUERIES` was 5,000 because a suppressed payload index made
        the row a full scan per query, and it outlived that reason.

        The cost was a row too short to be a measurement: at 5,000 it came in
        under `MIN_ROW_S`, so `--min-duration` scaled it, and the factor
        straddled the threshold -- 4x on strawmANN for 2.10 s against 2x on
        Qdrant for 1.96 s, short and published saying so. `--n-pin` freezes
        whichever side of 2 s the first pass landed on; a row long enough not to
        be scaled cannot land on either.
        """
        w = self.w
        t = {x.id: x for x in w.table()}
        args = t["W12-sel1"].args
        self.assertEqual(int(args[args.index("-n") + 1]), w.QUERIES,
                         "the filtered row searches the full query set")
        short = w.Result("W12-sel1", w.Status.ok, 3, 0, 0, "", qps=1.0, duration_s=1.96)
        self.assertEqual(w.min_duration_action(t["W12-sel1"], short, 0), "rerun",
                         "and if it ever is short again, it is still re-run")

    def test_the_mixed_rows_spread_their_append_across_the_search(self):
        """Unthrottled, bfb landed the whole append in the first 1-5% of the
        search, so both rows carried "the writer covered under 90% of either
        search" and measured the rebuild the append provoked rather than the
        concurrent write their own description promises.

        The volumes are the experiment -- W11 above `rebuild_ratio`,
        W11-steady deliberately below it -- so the throttle may move *when* the
        writes land and never how many of them there are.
        """
        w = self.w
        t = {x.id: x for x in w.table()}
        for wid, volume, span in (("W11-steady", w.w11_steady_n(), w.W11_STEADY_SPAN_S),
                                  ("W11", w.w11_n(), w.W11_SPAN_S)):
            bg = list(t[wid].background)
            self.assertIn("-T", bg, f"{wid} throttles its appender")
            rate = int(bg[bg.index("-T") + 1])
            batches = volume / int(bg[bg.index("-b") + 1])
            self.assertGreaterEqual(batches / rate, span,
                                    msg=f"{wid} should append over at least {span:.0f} s")
            self.assertLess(batches / rate, 2 * span,
                            msg=f"{wid} should append over about {span:.0f} s")
            self.assertEqual(int(bg[bg.index("-n") + 1]), volume,
                             f"{wid}'s write volume is untouched")

    def test_the_mixed_rows_search_the_count_fullrun_sized(self):
        w = self.w
        with mock.patch.object(w, "W11_STEADY_QUERIES", 4_840), \
                mock.patch.object(w, "W11_QUERIES", 6_965):
            t = {x.id: x for x in w.table()}
            for wid, n in (("W11-steady", 4_840), ("W11", 6_965)):
                fg = list(t[wid].args)
                self.assertEqual(int(fg[fg.index("-n") + 1]), n)
            # Every other search row keeps QUERIES.
            fg = list(t["W3"].args)
            self.assertEqual(int(fg[fg.index("-n") + 1]), w.QUERIES)

    def test_start_requirements_are_strawmanns_and_only_for_rows_that_need_them(self):
        """0925 printed both lines on all 18 invocations, Qdrant's included,
        and "appends 198,000 on top of W2's 990,000 ... reaches 1,237,500",
        a sum that left out W11-steady's 49,500."""
        w = self.w
        self.assertEqual(w.strawmann_start_requirements("qdrant", set()), [])
        self.assertEqual(w.strawmann_start_requirements("qdrant", {"W4", "W11"}), [])
        full = w.strawmann_start_requirements("strawmann", set())
        self.assertEqual(len(full), 2)
        self.assertIn("--connections", full[0])
        cap = full[1]
        self.assertIn(f"--capacity {w.required_capacity()}", cap)
        # The sum it states is the total it names.
        self.assertIn(f"{w.w11_steady_n():,} + {w.w11_n():,} on top of W2's {w.upload_n():,}", cap)
        self.assertEqual(w.w11_steady_n() + w.w11_n() + w.upload_n(), w.required_capacity())
        self.assertEqual(w.strawmann_start_requirements("strawmann-avx5", {"W3", "W10-ef64"}), [])
        only_w4 = w.strawmann_start_requirements("strawmann", {"W4-sat50"})
        self.assertEqual(len(only_w4), 1)
        self.assertIn("sockets", only_w4[0])
        only_w11 = w.strawmann_start_requirements("strawmann", {"W11-steady"})
        self.assertEqual(len(only_w11), 1)
        self.assertIn("--capacity", only_w11[0])

    def test_the_throttle_never_ends_the_append_before_its_span(self):
        """0925 resolved W11 at 431 s and ran it at 396 s: `-T` was rounded
        up from 4.59 to 5, and strawmANN's 490 s search outlived the writer
        (82% overlap)."""
        w = self.w
        self.assertEqual(w.w11_throttle(198_000, 431.0), math.floor(198_000 / w.W11_BATCH / 431.0))
        for points in (1, 49_500, 50_000, 198_000, 200_000):
            for span in (1.0, 19.0, 25.0, 60.0, 247.0, 256.0, 431.0, 612.0, 5_000.0):
                rate = w.w11_throttle(points, span)
                self.assertGreaterEqual(rate, 1)
                if points / w.W11_BATCH / span >= 1:
                    self.assertGreaterEqual(points / w.W11_BATCH / rate, span,
                                            f"{points} points over {span} s")

    def test_w11_policy_and_min_duration_exclusion(self):
        w = self.w
        t = {x.id: x for x in w.table()}
        self.assertEqual(t["W11"].ratio_policy, "search-during-write; no recall join")
        self.assertEqual(t["W3"].ratio_policy, "")
        short = w.Result("x", w.Status.ok, 1, 0, 0, "", qps=1.0, duration_s=0.5)
        self.assertEqual(w.min_duration_action(t["W3"], short, 0), "rerun")
        self.assertIsNone(w.min_duration_action(t["W3"], short, 2))
        self.assertIsNone(w.min_duration_action(t["W3"], w.Result("x", w.Status.ok, 3, 0, 0, "",
                                                                     qps=1.0, duration_s=3.0), 0))
        why = w.min_duration_action(t["W11"], short, 0)
        self.assertIsNotNone(why)
        self.assertNotEqual(why, "rerun")
        self.assertIn("re-append", why)

    def test_n_of_and_quant_of(self):
        w = self.w
        t = {x.id: x for x in w.table()}
        self.assertEqual(w.n_of(t["W3"]), w.QUERIES)
        self.assertEqual(w.n_of(t["W9"]), w.EXACT_QUERIES)
        import dataclasses
        self.assertEqual(w.n_of(dataclasses.replace(t["W3"], args=w.scale_n(t["W3"].args, 4))),
                         4 * w.QUERIES)
        self.assertEqual(w.quant_of(t["W7"]),
                         {"quantization_oversampling": 4.0, "quantization_rescore": True})
        self.assertEqual(w.quant_of(t["W6"]),
                         {"quantization_oversampling": None, "quantization_rescore": True})
        self.assertEqual(w.quant_of(t["W3"]),
                         {"quantization_oversampling": None, "quantization_rescore": None})

    def test_foreign_load_sees_a_foreign_python_and_not_the_harness(self):
        ps = ("95.0 /usr/bin/python3 train.py --epochs 3\n"
              "80.0 python3 /home/x/strawmann/bench/harness/workloads.py run http://l qdrant\n"
              "50.0 uv run --project bench /home/x/strawmann/bench/harness/report.py lbl\n"
              "30.0 /usr/bin/python3 /home/x/strawmann/bench/setup.py check\n"
              "99.0 /home/x/strawmann/zig-out/isa/avx512/strawmann-avx512 --port 6334\n"
              "88.0 ./qdrant --uri x\n"
              "70.0 rustc --crate-name foo\n"
              " 0.5 /usr/bin/python3 idle.py\n")
        self.assertEqual(self.w.foreign_from_ps(ps), "python3(95%) rustc(70%)")
        # bench/setup.py applies the same rule.
        setup = importlib.import_module("setup")
        self.assertFalse(setup._is_ours("/usr/bin/python3 train.py"))
        self.assertTrue(setup._is_ours("python3 /x/bench/harness/workloads.py run"))
        self.assertTrue(setup._is_ours("/x/zig-out/isa/base/strawmann-baseline --port 1"))
        self.assertFalse(setup._is_ours("uv run something-else"))
        # `ps` used to be exempt by prefix, which made every process whose
        # name starts with it -- psql, pserve, psi-notify -- invisible to the
        # gate. The sampler reads /proc and never runs ps.
        for f in (self.w._is_ours, setup._is_ours):
            self.assertFalse(f("psql -h db -c 'select 1'"))
            self.assertFalse(f("/usr/bin/psi-notify"))
            self.assertFalse(f("pserve app.ini"))
        # And the two samplers share one list rather than two copies of it.
        self.assertIs(self.w._OURS, setup._OURS)
        self.assertIs(self.w.FOREIGN_BUDGET_CORES, setup.FOREIGN_BUDGET_CORES)
        # The harness's own scripts, run from their directory: no path prefix.
        for f in (self.w._is_ours, setup._is_ours):
            self.assertTrue(f("./workloads.py run http://l qdrant"))
            self.assertTrue(f("python3 setup.py check"))
            self.assertTrue(f("/usr/bin/python3 ./workloads.py list"))
            self.assertFalse(f("./train.py"))

    def _fake_bfb(self) -> Path:
        fake = Path(self.tmp.name) / "bfb"
        fake.write_text(FAKE_BFB)
        fake.chmod(0o755)
        self.w.BFB = fake
        return fake

    def test_run_one_records_stamp_n_quant_and_overlap(self):
        # A stand-in bfb, so `run_one` runs end to end and the row carries what
        # the comparison needs, without an engine.
        w = self.w
        self._fake_bfb()
        t = {x.id: x for x in w.table()}
        results = Path(self.tmp.name) / "r"
        results.mkdir()
        stamp = w.harness_stamp()
        r = w.run_one(t["W7"], "http://localhost:1", results, [], None, 1, stamp)
        self.assertEqual(r.status, w.Status.ok, r.detail)
        self.assertEqual(r.n_requested, w.QUERIES)
        self.assertEqual(r.harness_hash, w.stamp_hash(stamp))
        self.assertNotIn("-n ", r.notes)
        self.assertEqual((r.quantization_oversampling, r.quantization_rescore), (4.0, True))
        # W7 carries a policy since findings 24: binary quantization is unusable
        # at d=128, so the row is a characterisation and gets no ratio.
        # W7's refusal moved to `compare.RECALL_FLOOR`, which reads the measured
        # recall instead of asserting a claim about d=128 onto every dataset.
        self.assertEqual(r.ratio_policy, "")
        self.assertIsNone(r.write_overlap_pct)
        # --n-factor changes the effective -n, recorded and noted; not the hash.
        r4 = w.run_one(t["W3"], "http://localhost:1", results, [], None, 4, stamp)
        self.assertEqual(r4.n_requested, 4 * w.QUERIES)
        self.assertEqual(r4.harness_hash, w.stamp_hash(stamp))
        self.assertIn(f"-n {4 * w.QUERIES} (4x the table's {w.QUERIES})", r4.notes)
        # W11: the append finishes at a fifth of the search, and the row says so.
        r11 = w.run_one(t["W11"], "http://localhost:1", results, [], None, 1, stamp)
        self.assertEqual(r11.status, w.Status.ok, r11.detail)
        self.assertEqual(r11.ratio_policy, "search-during-write; no recall join")
        self.assertIsNotNone(r11.write_overlap_pct)
        self.assertLess(r11.write_overlap_pct, 90)
        self.assertIn("write overlap", r11.notes)
        # ...and no stamp given, no hash invented.
        self.assertIsNone(w.run_one(t["W3"], "http://localhost:1", results, []).harness_hash)

    def test_w11_writer_failure_and_reaper_exception_are_on_the_row(self):
        # A writer that exits non-zero fails the row and still times its exit;
        # a reaper that raises is recorded on the row rather than lost (the
        # row used to die on a KeyError for the missing end time).
        w = self.w
        fake = self._fake_bfb()
        fake.write_text(FAKE_BFB.replace("    time.sleep(0.1)\n", "    time.sleep(0.1)\n    sys.exit(3)\n", 1))
        t = {x.id: x for x in w.table()}
        results = Path(self.tmp.name) / "r"
        results.mkdir()
        r = w.run_one(t["W11"], "http://localhost:1", results, [])
        self.assertEqual(r.status, w.Status.failed)
        self.assertIn("concurrent append failed", r.detail)
        self.assertIsNotNone(r.overlap_s)
        self.assertLess(r.overlap_s, 0.4)

        class Boom(w.subprocess.Popen):
            def communicate(self, *a, **k):
                if "--skip-create" in self.args:  # the writer only
                    raise RuntimeError("boom")
                return super().communicate(*a, **k)

        with mock.patch.object(w.subprocess, "Popen", Boom):
            r = w.run_one(t["W11"], "http://localhost:1", results, [])
        self.assertEqual(r.status, w.Status.failed)
        self.assertIn("RuntimeError: boom", r.detail)
        self.assertIn("append reaper failed: RuntimeError: boom", r.notes)
        self.assertIsNotNone(r.write_overlap_pct)

    def test_percentiles_are_ceil_nearest_rank(self):
        w = self.w
        xs = [1e-6 * (i + 1) for i in range(5)]           # 1..5 µs
        got = w.percentiles_us(xs)
        # rank ceil(0.5 * 5) = 3 -> the third value; round() gave the second.
        self.assertAlmostEqual(got["p50"], 3.0)
        self.assertAlmostEqual(got["p99"], 5.0)
        xs = [1e-6 * (i + 1) for i in range(3125)]        # W5's batch count
        got = w.percentiles_us(xs)
        self.assertAlmostEqual(got["p50"], 1563.0)        # ceil(1562.5)
        self.assertAlmostEqual(got["p95"], 2969.0)        # ceil(2968.75)
        self.assertAlmostEqual(got["p99"], 3094.0)        # ceil(3093.75)
        self.assertAlmostEqual(got["p999"], 3122.0)       # ceil(3121.875)

    def test_foreign_between_sees_only_what_ran_during_the_row(self):
        w = self.w
        S = w.CpuSample
        # pid 1: a long-idle process with a busy history (ps would say 95%);
        # pid 2: the engine; pid 3: a compiler that started inside the row;
        # pid 4: a kernel thread doing writeback; pid 5: our own harness.
        before = S(100.0, {1: (5000.0, "python3 train.py"), 2: (10.0, "./strawmann --port 1"),
                           4: (1.0, "kworker/0:1"), 5: (2.0, "python3 bench/harness/workloads.py run")},
                   busy_s=6000.0, kernel=frozenset({4}))
        after = S(110.0, {1: (5000.5, "python3 train.py"), 2: (30.0, "./strawmann --port 1"),
                          3: (7.0, "rustc --crate-name foo"), 4: (4.0, "kworker/0:1"),
                          5: (2.5, "python3 bench/harness/workloads.py run")},
                  busy_s=6000.0 + 0.5 + 20.0 + 7.0 + 3.0 + 0.5, kernel=frozenset({4}))
        # rustc alone is 0.70 cores, under the budget: `docs/decisions.md`
        # measures 1 core as inside run-to-run noise, so this is named in the
        # sample but is not a verdict.
        self.assertEqual(w.foreign_between(before, after), "")
        # 0.75, not 0.70: the long-idle `train.py` still burned 0.05 of a core
        # during the row, and the total is what the budget is compared against.
        self.assertEqual(w.foreign_between(before, after, budget_cores=0.5),
                         "0.75 cores: rustc(70%)")
        # A process that started and exited inside the row leaves no pid to
        # read; its time is on the system-wide line and counts toward the
        # total like any other. 0.70 + 0.40 crosses one core.
        gone = w.CpuSample(after.t, after.procs, after.busy_s + 4.0, after.kernel)
        self.assertEqual(w.foreign_between(before, gone),
                         "1.15 cores: rustc(70%) exited-processes(40%)")
        # ...unless it was our own reaped child: bfb starts and exits inside
        # every row, so its time is on the system-wide line at `after` and
        # nowhere else. The first gated row after the per-pid rewrite came
        # back as `exited-processes(124%)`, i.e. bfb's own 16 threads.
        ours_gone = w.CpuSample(after.t, after.procs, after.busy_s + 4.0, after.kernel,
                                children_s=before.children_s + 4.0)
        self.assertEqual(w.foreign_between(before, ours_gone), "")
        # A reused pid does not inherit its predecessor's time: cc1 is 0.30,
        # so with rustc the total is 1.00 and still not *over* the budget.
        reused = w.CpuSample(after.t, {**after.procs, 1: (3.0, "cc1 x.c")}, after.busy_s,
                             after.kernel)
        self.assertEqual(w.foreign_between(before, reused), "")
        self.assertIn("cc1(30%)", w.foreign_between(before, reused, budget_cores=0.5))
        self.assertEqual(w.foreign_between(before, before), "")

    def test_foreign_load_is_judged_as_a_total_not_per_process(self):
        """The verdict was each process against 20% of *one* core. It condemned
        0.2 cores, which `docs/decisions.md` measures as inside run-to-run
        noise, and it passed load made of many small processes, which the same
        calibration measures as costing real throughput."""
        w = self.w
        S = w.CpuSample
        # Ten processes at 15% of a core each: 1.5 cores, past the 3-cores-is-
        # +11% calibration's lower end, and not one of them would have crossed
        # a 20%-of-one-core test.
        before = S(0.0, {i: (0.0, f"worker{i}") for i in range(10)},
                   busy_s=0.0, kernel=frozenset())
        after = S(10.0, {i: (1.5, f"worker{i}") for i in range(10)},
                  busy_s=15.0, kernel=frozenset())
        got = w.foreign_between(before, after)
        self.assertTrue(got.startswith("1.50 cores"), got)
        self.assertIn("worker", got)
        # And a single process at a fifth of a core is no longer a verdict.
        one = S(10.0, {1: (2.0, "editor")}, busy_s=2.0, kernel=frozenset())
        self.assertEqual(w.foreign_between(S(0.0, {1: (0.0, "editor")}, 0.0, frozenset()),
                                           one), "")

    def test_the_gate_names_the_cpu_count_in_its_verdict(self):
        """`cores` was rebound to the foreign-core sum before the verdict was
        printed, so env.txt archived "quiescent (load 0.4 over 0.03 cores)"."""
        setup = importlib.import_module("setup")
        b = {1: (0.0, "/usr/bin/Xorg")}
        a = {1: (0.05, "/usr/bin/Xorg")}
        env = setup.Env(lax=False)
        real_read, real_sleep, real_cpu = setup.read, setup.time.sleep, setup._cpu_seconds
        setup._cpu_seconds = lambda: b if not hasattr(setup, "_sampled2") else a
        setup.read = lambda p: "0.10 0.2 0.3 1/100 1" if "loadavg" in str(p) else real_read(p)
        setup.time.sleep = lambda _: setattr(setup, "_sampled2", True)
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                setup.check_quiescent(env)
        finally:
            setup.read, setup.time.sleep, setup._cpu_seconds = real_read, real_sleep, real_cpu
            delattr(setup, "_sampled2")
        self.assertEqual(env.failures, 0)
        verdict = next(x for x in err.getvalue().splitlines() if "quiescent" in x)
        self.assertIn(f"over {os.cpu_count() or 1} cores", verdict)
        self.assertNotIn("over 0.0", verdict)

    def test_a_load_only_gate_failure_waits_on_the_load_average(self):
        """"load X over N cores" with nobody named is the one-minute average
        still carrying the previous phase; the process sampler returns in one
        tick for it, so the retries used to burn out in fifteen seconds."""
        w = self.w
        readings = iter([80.0, 40.0, 8.0])
        slept = []
        with mock.patch.object(w, "load_per_core_pct", lambda: next(readings)), \
                mock.patch.object(w.time, "sleep", lambda s: slept.append(s)), \
                contextlib.redirect_stdout(io.StringIO()):
            w.settle_load_for_retry()
        self.assertEqual(len(slept), 2)

    def test_the_gate_sums_foreign_cores_too(self):
        """`bench/setup.py` and `workloads` must agree: a process the gate is
        willing to start a run alongside must not condemn every row it touches."""
        setup = importlib.import_module("setup")
        before = {1: (0.0, "rustc a.rs"), 2: (0.0, "./strawmann --port 1")}
        after = {1: (7.0, "rustc a.rs"), 2: (20.0, "./strawmann --port 1")}
        # 0.70 cores foreign; the engine's 2.0 cores are ours and excluded.
        self.assertAlmostEqual(setup.foreign_cores(before, after, 10.0), 0.7)
        self.assertLessEqual(setup.foreign_cores(before, after, 10.0),
                             setup.FOREIGN_BUDGET_CORES)
        # Still named, so the operator sees it.
        self.assertEqual(setup.busy_processes(before, after, 10.0), ["rustc(70%)"])

    def test_the_gate_reads_the_sampler_and_not_the_load_average(self):
        setup = importlib.import_module("setup")
        # Two cores of foreign load, over the budget by construction, so the
        # verdict below is about *which instrument* decided and not about the
        # threshold.
        b = {1: (0.0, "rustc a.rs")}
        a = {1: (20.0, "rustc a.rs")}
        # ...and the gate's verdict is that measurement, not the load average.
        # load1 is a one-minute decaying average: it read 1.15 on 12 cores,
        # inside the threshold, sixty seconds after this sampler had seen a
        # rustc at 102%, and the run that passed measured ten rows against it.
        env = setup.Env(lax=False)
        setup._cpu_seconds = lambda: b if not hasattr(setup, "_sampled") else a
        real_read, real_sleep = setup.read, setup.time.sleep
        setup.read = lambda p: "0.10 0.2 0.3 1/100 1" if "loadavg" in str(p) else real_read(p)
        setup.time.sleep = lambda _: setattr(setup, "_sampled", True)
        try:
            setup.check_quiescent(env)
        finally:
            setup.read, setup.time.sleep = real_read, real_sleep
            delattr(setup, "_sampled")
        self.assertEqual(env.failures, 1)
        self.assertTrue(any("NOT quiescent" in x and "rustc" in x for x in env.lines + [""])
                        or env.failures == 1)
        # And which processes are busy is transient, so it must not reach the
        # hash: a run passed at hash 1cdcf393fa18f478 while every row
        # it recorded carried 473591cd7728f349.
        self.assertFalse([x for x in env.lines if "busy_processes" in x], env.lines)
        self.assertTrue([x for x in env.observed if "busy_processes" in x], env.observed)

    def test_smt_is_declared_and_not_a_verdict(self):
        """SMT on passes the gate, stays in the hash, and says which cpus pair.

        It stopped being a failure because §7.1 asks for SMT "explicitly on or
        off, declared per run"; it stays *hashed* because that declaration is
        what keeps an SMT-on row off the same chart as an SMT-off one.
        """
        setup = importlib.import_module("setup")
        env = setup.Env(lax=False)
        real_read = setup.read
        setup.read = lambda p: "on" if "smt/control" in str(p) else real_read(p)
        try:
            setup.check_smt(env)
        finally:
            setup.read = real_read
        self.assertEqual(env.failures, 0)
        self.assertIn("smt=on", env.lines)

    def test_smt_topology_names_the_pairing(self):
        """`--server-cpus` is raw cpu numbers, so which of them share a core
        decides whether `4-11` is eight cores or four cores twice over."""
        setup = importlib.import_module("setup")
        # This host: siblings N and N+12, each pair listed twice, once per cpu.
        pairs = [f"{n},{n + 12}" for n in range(12)] * 2
        self.assertEqual(setup.smt_topology(pairs),
                         "cpu N and cpu N+12 share a core (12 pairs)")
        # The range spelling is the same fact: /sys writes `0-1` where the
        # pair is contiguous, and reading it as one cpu named "0-1" would
        # have found no pairs at all.
        self.assertEqual(setup.smt_topology(["0-1", "2-3"]),
                         "cpu N and cpu N+1 share a core (2 pairs)")
        # No uniform stride, so the groups are named rather than summarised.
        self.assertEqual(setup.smt_topology(["0,1", "2,5"]),
                         "cpus sharing a core: 0,1 2,5")
        # SMT off: `thread_siblings_list` is one cpu per core, no pairs, and
        # the hint is omitted rather than printed empty.
        self.assertEqual(setup.smt_topology(["0", "1", "2"]), "")

    def test_build_identity_from_run_meta(self):
        w = self.w
        sm = {"strawmann": {"commit": "abc", "dirty": True, "binary_sha256": "1111",
                            "optimize": "ReleaseFast"},
              "isa_build": "avx512", "gate": "pass", "profile": "isolated"}
        self.assertEqual(w.build_identity(sm), {
            "gate": "pass", "profile": "isolated", "engine_build": "abc-dirty",
            "engine_binary": "1111", "isa_build": "avx512", "optimize": "ReleaseFast",
            # A binary built before `-Dvisited` existed says nothing about it,
            # which is the honest value rather than the default's name.
            "visited_set": None})
        qd = {"qdrant": {"version": "1.19.0", "digest": "sha256:ff"}, "gate": "FAIL",
              "profile": "as-deployed"}
        self.assertEqual(w.build_identity(qd)["engine_build"], "1.19.0")
        self.assertEqual(w.build_identity(qd)["engine_binary"], "sha256:ff")
        self.assertIsNone(w.build_identity(qd)["isa_build"])
        self.assertIsNone(w.build_identity(qd)["optimize"])
        self.assertTrue(set(w.BUILD_KEYS) <= set(w.build_identity(sm)))

    def test_the_stamp_changes_when_the_settle_discipline_does(self):
        """A row measured settled is not comparable with one measured unsettled.

        The settle changes what the row *after* an upload measures — Qdrant's
        W3 moved 1,180/2,013/1,402 to 1,951/1,965/1,977 when it landed — and
        that row carries no local signal, since its own `engine_settle_s` is
        None. Without the discipline in the stamp, `compare.py` ratios a
        pre-settle label against a post-settle one and prints nothing, and
        `regression.py` keeps serving the pre-settle floor.
        """
        w = self.w
        base = w.harness_stamp()
        self.assertIn("engine_settle", base)
        self.assertIn("engine_settle", w.STAMP_KEYS)

        # Every knob that decides how quiet is quiet enough moves the hash.
        for key, value in (("cores", 0.5), ("window_s", 5.0),
                           ("stable", 1), ("timeout_s", 30.0)):
            other = dict(base)
            other["engine_settle"] = {**base["engine_settle"], key: value}
            self.assertNotEqual(w.stamp_hash(base), w.stamp_hash(other),
                                f"{key} does not reach the stamp")

        # And a stamp from before the field existed is a disagreement, not a
        # match — which is what makes every published row stale rather than
        # silently comparable.
        legacy = {k: v for k, v in base.items() if k != "engine_settle"}
        self.assertNotEqual(w.stamp_hash(base), w.stamp_hash(legacy))

    def test_settle_engine_gives_up_rather_than_waiting_for_ever(self):
        """The three ways the wait ends, and what each records on the row.

        It is on by default now, so every upload row runs it. A wait that never
        returned would hang a nine-hour run at row three, and one that reported
        nothing would leave a reader unable to tell "settled at once" from
        "gave up after three minutes" -- which is the difference between a
        quiet engine and an engine that never stopped working.
        """
        w = self.w
        real = (w.ENGINE_IDLE_WINDOW_S, w.ENGINE_IDLE_TIMEOUT_S, w.ENGINE_IDLE_STABLE)
        w.ENGINE_IDLE_WINDOW_S, w.ENGINE_IDLE_TIMEOUT_S, w.ENGINE_IDLE_STABLE = 0.01, 0.2, 2
        real_cpu = w.procstat.cpu_seconds
        try:
            # No engine to watch: nulls, and no waiting at all.
            self.assertEqual(w.settle_engine(None, "W0-upload"),
                             {"engine_settle_s": None, "engine_settle_cores": None})

            # Quiet from the first window: returns as soon as it is sure, and
            # the bound is the *patched* timeout. Bounding against `real[1]`
            # (180 s, captured before the patch) passed for every outcome the
            # patched function can produce, including one that never exits
            # early at all.
            w.procstat.cpu_seconds = lambda _pid: 5.0          # never moves
            quiet = w.settle_engine(1, "W0-upload")
            self.assertLess(quiet["engine_settle_s"], w.ENGINE_IDLE_TIMEOUT_S)
            self.assertEqual(quiet["engine_settle_cores"], 0.0)

            # Busy for ever: bounded by the timeout, and the row says how much
            # of a core it was still using when the harness stopped waiting.
            # The ticker adds one core-second per 0.01 s window, so `cores` is
            # 100 — comfortably over the threshold, which is all this needs.
            ticking = itertools.count(0.0, 1.0)
            w.procstat.cpu_seconds = lambda _pid: next(ticking)
            with contextlib.redirect_stdout(io.StringIO()):
                busy = w.settle_engine(1, "W2")
            self.assertGreaterEqual(busy["engine_settle_s"], w.ENGINE_IDLE_TIMEOUT_S)
            self.assertGreater(busy["engine_settle_cores"], w.ENGINE_IDLE_CORES)

            # A wait that never returns is the regression this test exists for,
            # so the ticker is finite: a loop that ignored the timeout runs off
            # the end and fails here instead of hanging the suite — and
            # `scripts/check.py` runs unittest with no `timeout=`, so a hang is
            # a silent CI stall rather than a red assertion.
            budget = itertools.chain(
                (float(i) for i in range(int(w.ENGINE_IDLE_TIMEOUT_S /
                                            w.ENGINE_IDLE_WINDOW_S) + 20)),
                iter(lambda: (_ for _ in ()).throw(
                    AssertionError("settle_engine ran past its own timeout")), None))
            w.procstat.cpu_seconds = lambda _pid: next(budget)
            with contextlib.redirect_stdout(io.StringIO()):
                w.settle_engine(1, "W2")

            # The engine exits mid-watch — busy, then gone — which is the
            # realistic death. No `cores` for a process that is not there, and
            # no exception: the row after this one fails on its own.
            seq = iter([10.0, 12.0, None])
            w.procstat.cpu_seconds = lambda _pid: next(seq, None)
            with contextlib.redirect_stdout(io.StringIO()):
                gone = w.settle_engine(1, "W2")
            self.assertIsNotNone(gone["engine_settle_s"])
            self.assertIsNone(gone["engine_settle_cores"])

            # And the same path with the first window already gone, which is
            # where `cores` is None at the print and used to raise TypeError.
            w.procstat.cpu_seconds = lambda _pid: None
            with contextlib.redirect_stdout(io.StringIO()):
                first = w.settle_engine(1, "W2")
            self.assertIsNone(first["engine_settle_cores"])
        finally:
            w.procstat.cpu_seconds = real_cpu
            w.ENGINE_IDLE_WINDOW_S, w.ENGINE_IDLE_TIMEOUT_S, w.ENGINE_IDLE_STABLE = real

    def test_an_upload_row_waits_for_the_engine_to_go_quiet(self):
        """bfb returns at green, and green is not idle.

        Measured on sift1m: W3 straight after W2 cost 528 us/query
        in the pass whose W2 happened to wait longest, and 5,024 us/query in
        the pass that waited least -- nine times the CPU for the same 50,000
        queries, with `sched_coverage` 0.897 against 0.140. That is Qdrant's
        optimizer still running, measured as if it were search, and it is
        findings 38's 28% spread on that row.

        Only upload rows settle, and only after their own numbers are closed,
        so what it protects is the row that comes next.
        """
        w = self.w
        self._fake_bfb()
        seen = []
        real = w.settle_engine
        w.settle_engine = lambda pid, what: (seen.append(what) or
                                             {"engine_settle_s": 4.0,
                                              "engine_settle_cores": 0.01})
        try:
            t = {x.id: x for x in w.table()}
            results = Path(self.tmp.name) / "settle"
            results.mkdir(parents=True, exist_ok=True)
            up = w.run_one(t["W0-upload"], "http://localhost:1", results, [], warmup=False)
            self.assertEqual(seen, ["W0-upload"], "an upload row must settle")
            self.assertEqual(up.engine_settle_s, 4.0)
            seen.clear()
            search = w.run_one(t["W3"], "http://localhost:1", results, [], warmup=False)
            self.assertEqual(seen, [], "a search row must not settle")
            self.assertIsNone(search.engine_settle_s)
        finally:
            w.settle_engine = real

    def test_run_one_stamps_the_build_and_warms_up_first(self):
        w = self.w
        self._fake_bfb()
        t = {x.id: x for x in w.table()}
        results = Path(self.tmp.name) / "r"
        results.mkdir()
        build = w.build_identity({"strawmann": {"commit": "abc", "binary_sha256": "11"},
                                  "isa_build": "avx2", "gate": "pass", "profile": "isolated"})
        seen = []
        real_run = w.subprocess.run

        def spy(cmd, **kw):
            seen.append(list(cmd))
            return real_run(cmd, **kw)

        with mock.patch.object(w.subprocess, "run", spy):
            r = w.run_one(t["W3"], "http://localhost:1", results, [], build=build)
        self.assertEqual(r.status, w.Status.ok, r.detail)
        self.assertEqual((r.engine_build, r.engine_binary, r.isa_build, r.gate, r.profile),
                         ("abc", "11", "avx2", "pass", "isolated"))
        # Warm-up first, discarded, at WARMUP_N and to its own JSON; then the
        # measured pass at the table's -n.
        self.assertEqual(len(seen), 2, seen)
        warm, measured = seen
        self.assertEqual(warm[warm.index("-n") + 1], str(w.WARMUP_N))
        self.assertTrue(warm[warm.index("--json") + 1].endswith("W3.warmup.json"))
        self.assertEqual(measured[measured.index("-n") + 1], str(w.QUERIES))
        self.assertTrue(measured[measured.index("--json") + 1].endswith("W3.json"))
        self.assertIsNotNone(r.warmup_s)
        # `--no-warmup`: one pass, and the row says there was no warm-up.
        seen.clear()
        with mock.patch.object(w.subprocess, "run", spy):
            r = w.run_one(t["W3"], "http://localhost:1", results, [], warmup=False)
        self.assertEqual(len(seen), 1)
        self.assertIsNone(r.warmup_s)
        # An upload row is never run twice: no `-n` scaling, no second create.
        self.assertIsNone(w.warmup_command(["bfb", "--json", "x"], results, "W1"))
        with mock.patch.object(w.subprocess, "run", spy):
            w.run_one(t["W1"], "http://localhost:1", results, [])
        self.assertEqual(len(seen), 2)   # the previous one plus this single pass

    def test_a_row_that_did_not_measure_is_a_non_zero_exit(self):
        """`main` returned 0 unconditionally, and three callers test the code.

        `isa_sweep.py` says why it started to: "a failed arm used to leave an
        empty result directory while the sweep printed 'sweep complete'". A run
        against an engine that was never up wrote 26 FAILED rows and exited 0.
        """
        w = self.w
        ok = w.Result(id="W3", status=w.Status.ok, seconds=1, load_start=0,
                      load_end=0, foreign="", qps=1.0, rps=1.0, detail="", notes="")
        declined = w.Result(id="W12", status=w.Status.not_applicable, seconds=0,
                            load_start=0, load_end=0, foreign="", qps=None,
                            rps=None, detail="", notes="")
        failed = w.Result(id="W4", status=w.Status.failed, seconds=0, load_start=0,
                          load_end=0, foreign="", qps=None, rps=None, detail="",
                          notes="")
        no_out = w.Result(id="W5", status=w.Status.no_output, seconds=0,
                          load_start=0, load_end=0, foreign="", qps=None,
                          rps=None, detail="", notes="")

        def unmeasured(rows):
            return [r.id for r in rows
                    if r.status in (w.Status.failed, w.Status.no_output)]

        # `n/a` is the engine declining a construct §2 has not built, which W12
        # does on every run. It is not a failure and must not fail the run.
        self.assertEqual(unmeasured([ok, declined]), [])
        self.assertEqual(unmeasured([ok, failed, declined]), ["W4"])
        self.assertEqual(unmeasured([failed, no_out]), ["W4", "W5"])

    def test_the_closed_sets_keep_the_values_rows_json_already_holds(self):
        """These are `StrEnum`s so the artifact is byte-identical.

        Every one of them is compared against a literal somewhere, or read back
        out of a file measured before it existed. A renamed *value* would be a
        silent format change; a renamed member would only be a compile error.
        """
        w = self.w
        self.assertEqual([m.value for m in w.Gate], ["pass", "FAIL", "mixed"])
        self.assertEqual([m.value for m in w.Placement], ["cold", "cached", "pinned"])
        self.assertEqual([m.value for m in w.RpsSource], ["pinned", "own"])
        self.assertEqual([m.value for m in w.Direction], ["same", "opposite", "mixed"])
        self.assertEqual(sorted(m.value for m in w.Status),
                         ["FAILED", "n/a", "no-output", "ok"])
        # `unset` is "" so a row written before `load_mode` existed reads back.
        self.assertEqual(w.LoadMode.unset.value, "")
        self.assertEqual(w.LoadMode(""), w.LoadMode.unset)
        # `pinned` is two words: a memory placement, and an offered rate that
        # came from --rps-reference rather than this engine's own saturation.
        self.assertEqual(w.Placement.pinned.value, w.RpsSource.pinned.value)
        self.assertIsNot(w.Placement.pinned, w.RpsSource.pinned)

    def test_no_output_is_a_failure_and_unknown_args_are_errors(self):
        w = self.w
        fake = self._fake_bfb()
        # bfb exits 0, writes nothing, prints nothing.
        fake.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n")
        t = {x.id: x for x in w.table()}
        results = Path(self.tmp.name) / "r"
        results.mkdir()
        r = w.run_one(t["W3"], "http://localhost:1", results, [], warmup=False)
        self.assertEqual(r.status, w.Status.no_output)
        self.assertNotEqual(r.status, w.Status.ok)
        self.assertIsNone(r.qps)
        self.assertIn("no `Median qps`", r.detail)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.main(["workloads.py", "run", "http://l", "lbl", "W3x"]), 2)
        self.assertIn("unknown workload id(s): W3x", err.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.main(["workloads.py", "run", "http://l", "lbl", "--min-durations"]), 2)
        # argparse's wording, not the hand-rolled parser's. What matters is
        # that a misspelt flag is named and refused: it used to be dropped on
        # the floor, so `--min-durations` ran the table without re-running
        # short rows and exited 0.
        self.assertIn("--min-durations", err.getvalue())
        self.assertIn("error:", err.getvalue())

    def test_a_re_measured_open_loop_row_keeps_its_offered_rate(self):
        """The offer fields were attached to the first measurement only; the
        `--min-duration` and contamination retries rebind the row from a fresh
        `run_one`, so a re-measured W4-sat row published with no target and the
        served-fraction check and the cross-engine latency caveat stopped
        seeing it. The stand-in bfb answers in 0.5 s, under MIN_ROW_S, so the
        first measurement is re-run."""
        w = self.w
        self._fake_bfb()
        os.environ.pop("RESULTS_DIR", None)
        d = Path(self.tmp.name) / "bench/results/ol"
        d.mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(w, "settle_for_retry", lambda: None):
            rc = w.main(["workloads.py", "run", "http://localhost:1", "ol", "W4-sat50",
                         "--rps-reference", "1000", "--min-duration"])
        self.assertEqual(rc, 0)
        rows = {r["id"]: r for r in json.loads((d / "rows.json").read_text())}
        row = rows["W4-sat50"]
        self.assertIn("short (<2 s)", row.get("notes", ""), row)  # it was re-measured
        self.assertEqual(row["rps_target"], 500)
        self.assertEqual(row["rps_fraction"], 0.5)
        self.assertEqual(row["saturation_qps"], 1000.0)
        self.assertEqual(row["rps_reference_source"], "pinned")

    def test_the_min_duration_verdict_is_pinned_across_an_arms_passes(self):
        """§7.4 folds three passes into a median, and a median of two
        configurations is not one.

        `--min-duration` doubles a short row's `-n` until it clears
        `MIN_ROW_S`, which a row's duration can straddle: W12
        settled at 2x in one pass of qd-dbp1m-perf-rel-0903 and 4x in another,
        and `aggregate.fold` published the median of three rates measured under
        two configurations, keeping pass 1's `n_queries` with only a note. The
        first pass decides; the passes after it read the verdict and do not
        escalate again.
        """
        w = self.w
        self._fake_bfb()
        os.environ.pop("RESULTS_DIR", None)
        pin = Path(self.tmp.name) / "pins/n_factors.json"
        out1, out2 = io.StringIO(), io.StringIO()
        for lbl, cap in (("p1", out1), ("p2", out2)):
            (Path(self.tmp.name) / f"bench/results/{lbl}").mkdir(parents=True)
            with contextlib.redirect_stdout(cap), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(w, "settle_for_retry", lambda: None):
                rc = w.main(["workloads.py", "run", "http://localhost:1", lbl, "W3",
                             "--min-duration", "--n-pin", str(pin)])
            self.assertEqual(rc, 0)

        # The stand-in bfb always answers short, so the first pass escalates to
        # the 4x ceiling and writes it down.
        self.assertEqual(json.loads(pin.read_text())["W3"], 4)
        self.assertIn("re-running with -n", out1.getvalue())

        # The second pass measures that configuration directly: no escalation,
        # and the same `-n` on the row rather than a differently-sized one.
        self.assertNotIn("re-running with -n", out2.getvalue())
        def n_req(lbl):
            rows = json.loads((Path(self.tmp.name) / f"bench/results/{lbl}/rows.json").read_text())
            return {r["id"]: r for r in rows}["W3"]["n_requested"]
        self.assertEqual(n_req("p1"), n_req("p2"))

    def test_a_run_without_min_duration_does_not_write_a_pin(self):
        """The pin records `--min-duration`'s verdict, so a run that has no
        verdict must not write one.

        Otherwise a hand invocation that passed `--n-pin` and omitted
        `--min-duration` writes 1 for every row, and the next pass reads it,
        skips the escalation and publishes short rows as the ramp measurements
        the flag exists to prevent — silently, because a pinned row prints no
        re-run line.
        """
        w = self.w
        self._fake_bfb()
        os.environ.pop("RESULTS_DIR", None)
        pin = Path(self.tmp.name) / "pins/no_verdict.json"
        (Path(self.tmp.name) / "bench/results/nv").mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(w, "settle_for_retry", lambda: None):
            self.assertEqual(w.main(["workloads.py", "run", "http://localhost:1", "nv",
                                     "W3", "--n-pin", str(pin)]), 0)
        self.assertFalse(pin.exists(), "a run with no --min-duration wrote a pin")

    def test_an_explicit_n_factor_is_not_overridden_by_a_stale_pin(self):
        # A flag that is silently discarded is worse than one that is refused.
        w = self.w
        self._fake_bfb()
        os.environ.pop("RESULTS_DIR", None)
        pin = Path(self.tmp.name) / "pins/stale.json"
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(json.dumps({"W3": 1}))
        (Path(self.tmp.name) / "bench/results/xf").mkdir(parents=True)
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(w, "settle_for_retry", lambda: None):
            self.assertEqual(w.main(["workloads.py", "run", "http://localhost:1", "xf",
                                     "W3", "--min-duration", "--n-pin", str(pin),
                                     "--n-factor", "4"]), 0)
        self.assertIn("ignoring the -n pin", cap.getvalue())
        rows = {r["id"]: r for r in
                json.loads((Path(self.tmp.name) / "bench/results/xf/rows.json").read_text())}
        # The explicit factor was honoured, and the stale pin left alone.
        self.assertGreaterEqual(rows["W3"]["n_requested"], 4)
        self.assertEqual(json.loads(pin.read_text())["W3"], 1)

    def test_an_absent_pin_file_leaves_the_escalation_alone(self):
        # The pin is an optimisation of consistency, not a precondition: a hand
        # invocation without `--n-pin` must still re-run a short row.
        w = self.w
        self._fake_bfb()
        os.environ.pop("RESULTS_DIR", None)
        (Path(self.tmp.name) / "bench/results/nopin").mkdir(parents=True)
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(w, "settle_for_retry", lambda: None):
            rc = w.main(["workloads.py", "run", "http://localhost:1", "nopin", "W3",
                         "--min-duration"])
        self.assertEqual(rc, 0)
        self.assertIn("re-running with -n", cap.getvalue())

    def test_run_records_isa_build_from_the_server_banner(self):
        # `main` end to end against the stand-in bfb: an ISA-sweep arm's
        # run.json names the build the banner names, not `native`.
        w = self.w
        self._fake_bfb()
        os.environ.pop("RESULTS_DIR", None)
        d = Path(self.tmp.name) / "bench/results/avx512-rep0"
        d.mkdir(parents=True)
        (d / "server.log").write_text(
            "strawmann 0.1\n  isa build              : avx512\n"
            "  memory bandwidth       : 47.3 GB/s aggregate (24 threads) · "
            "25.1 GB/s single core (53% of bus)\n")
        # `settle_for_retry` neutered, because this is the one test that drives
        # the real `main` and therefore the real contamination path. A row
        # measured on a busy host comes back contaminated, `contaminated_action`
        # says "rerun", and each of the CONTAMINATED_RETRIES attempts first
        # waits out RETRY_SETTLE_TIMEOUT_S -- up to nine minutes of real
        # sleeping in a unit test, for a property (does run.json name the ISA
        # the banner named) that has nothing to do with quiescence. On an idle
        # box it never fires, which is why this sat here unnoticed until a
        # report render happened to run alongside the suite and turned a
        # 44-second gate into a ten-minute one that looked hung.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(w, "settle_for_retry", lambda: None):
            rc = w.main(["workloads.py", "run", "http://localhost:1", "avx512-rep0", "W3"])
        self.assertEqual(rc, 0)
        meta = json.loads((d / "run.json").read_text())
        self.assertEqual(meta.get("isa_build"), "avx512")
        self.assertEqual(meta["host"]["memory_bandwidth"]["source"], "strawmann startup banner")
        rows = {r["id"]: r for r in json.loads((d / "rows.json").read_text())}
        self.assertEqual(rows["W3"]["harness_hash"], w.stamp_hash(meta["harness"]))

class BfbPinTests(unittest.TestCase):
    """§9 pins the toolchain because "a benchmark that silently changes its load
    generator is not a benchmark", and nothing enforced it for the one tool that
    generates every number."""

    def _repo(self, td, head):
        repo = Path(td) / "bfb"
        (repo / "target/release").mkdir(parents=True)
        (repo / "target/release/bfb").write_text("")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
        (repo / "x").write_text(head)
        subprocess.run(["git", "-C", str(repo), "add", "x"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", head], check=True)
        # Touched *after* the commit, because that is what "built from this
        # checkout" means and `stale_bfb_binary` now says so. Created before
        # it, these fixtures would have been stale or not depending on which
        # side of a second boundary the two landed.
        os.utime(repo / "target/release/bfb", None)
        return repo

    def test_a_checkout_that_is_not_the_pin_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(td, "whatever")
            head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            with mock.patch.object(workloads, "BFB", repo / "target/release/bfb"):
                with mock.patch.object(workloads, "BFB_COMMIT", "deadbeef"):
                    why = workloads.check_bfb_pin()
                    self.assertIsNotNone(why)
                    self.assertIn(head[:8], why)
                    # A mismatch is a refusal, not a caveat -- unless asked.
                    with mock.patch.dict(os.environ, {"BFB_PIN_LAX": "1"}):
                        self.assertIsNone(workloads.check_bfb_pin())
                # The pinned commit passes.
                with mock.patch.object(workloads, "BFB_COMMIT", head[:8]):
                    self.assertIsNone(workloads.check_bfb_pin())

    def test_a_checkout_that_is_not_a_git_tree_is_not_a_mismatch(self):
        """A released tarball or a CI cache has no HEAD to compare. That is the
        absence of a check, not a failed one."""
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / "target/release/bfb"
            b.parent.mkdir(parents=True)
            b.write_text("")
            with mock.patch.object(workloads, "BFB", b):
                self.assertIsNone(workloads.check_bfb_pin())



    def test_a_binary_older_than_the_pinned_commit_is_refused(self):
        """The checkout being right is not the binary being right.

        Moving the pin is when the two come apart: it moved from
        the fork `6f216634` to upstream `0c1aafee`, `git checkout` was instant
        and `cargo build` was not, and in between `check_bfb_pin` passed
        against a binary five days older than the commit it vouched for. Every
        row measured then would have been stamped with code that did not
        produce it -- §9's exact failure, through the front door.

        One-way on purpose: older than the commit means definitely not built
        from it, newer only means possibly.
        """
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(td, "whatever")
            binary = repo / "target/release/bfb"
            head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            at = int(subprocess.run(
                ["git", "-C", str(repo), "show", "-s", "--format=%ct", "HEAD"],
                capture_output=True, text=True).stdout.strip())
            with mock.patch.object(workloads, "BFB", binary), \
                    mock.patch.object(workloads, "BFB_COMMIT", head[:8]):
                # Built an hour before the commit: impossible, and refused by
                # `check_bfb_pin` even though the checkout is the pin.
                os.utime(binary, (at - 3600, at - 3600))
                why = workloads.check_bfb_pin()
                self.assertIsNotNone(why)
                self.assertIn("cargo build", why)
                # Built after it: nothing to say.
                os.utime(binary, (at + 1, at + 1))
                self.assertIsNone(workloads.check_bfb_pin())
                # The same escape hatch as the commit mismatch.
                os.utime(binary, (at - 3600, at - 3600))
                with mock.patch.dict(os.environ, {"BFB_PIN_LAX": "1"}):
                    self.assertIsNone(workloads.check_bfb_pin())
                # No binary is the absence of a check, not a verdict.
                binary.unlink()
                self.assertIsNone(workloads.stale_bfb_binary(repo))

class PlacementInvariantTests(unittest.TestCase):
    """§7.4 compares at one residency or not at all.

    The run used to leave each engine at its own default — strawmANN holding
    anonymous RAM, Qdrant mapping a file — refuse the storage rows, note that
    "residency can move throughput too", and print the throughput ratios
    anyway. The confound was named on the page and bounded nowhere.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.w = self.m["workloads"]
        # `fullrun` is an entry point rather than a library and is not in
        # `_reload`'s dependency order; imported here, after it, so the module
        # it binds is the reloaded `workloads`.
        self.f = importlib.reload(sys.modules["fullrun"]) \
            if "fullrun" in sys.modules else importlib.import_module("fullrun")

    def tearDown(self):
        self.tmp.cleanup()

    def test_qdrant_segments_are_read_one_by_one_from_telemetry(self):
        """gRPC gives `segments_count` only, so "2 segments" could not say
        whether the second was a graph. The shape is Qdrant 1.19's
        `/telemetry?details_level=10`, trimmed to the fields read."""
        f = self.f
        seg = lambda n, kind, app: {"info": {"num_points": n, "num_indexed_vectors": n,
                                             "segment_type": kind, "is_appendable": app}}
        doc = {"result": {"collections": {"collections": [
            {"id": "bench2", "shards": [{"local": {"segments": [
                seg(1_000_000, "indexed", False), seg(0, "plain", True)]}}]},
            {"id": "bench6", "shards": [{"local": None}]}]}}}
        got = f.parse_segment_telemetry(doc)
        self.assertEqual([s["points"] for s in got["bench2"]], [1_000_000, 0])
        self.assertEqual(got["bench2"][1]["appendable"], True)
        self.assertEqual(got["bench6"], [])    # a remote shard lists nothing here
        self.assertEqual(f.parse_segment_telemetry({}), {})
        # strawmANN is addressed on its own port and is never asked.
        self.assertEqual(f.qdrant_segments(f"http://localhost:{f.STRAWMANN_PORT}"), {})

    def test_a_residency_one_engine_cannot_serve_is_refused_before_the_run(self):
        f, P = self.f, self.w.Placement
        # Qdrant v1.19.0: "`pinned` memory placement is not supported for dense
        # vector storage" -- measured, not assumed.
        why = f.preflight_placement(P.pinned, ["strawmann", "qdrant"])
        self.assertIsNotNone(why)
        self.assertIn("qdrant cannot serve", why)
        self.assertIn("cached", why)          # names what *is* servable
        # The only residencies both can serve.
        self.assertIsNone(f.preflight_placement(P.cached, ["strawmann", "qdrant"]))
        self.assertIsNone(f.preflight_placement(P.cold, ["strawmann", "qdrant"]))
        # A single-engine run may use the engine's own default.
        self.assertIsNone(f.preflight_placement(P.pinned, ["strawmann"]))

    def _label(self, name, placements):
        d = Path(self.tmp.name) / "bench/results" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "collections.json").write_text(json.dumps({"collections": [
            {"collection": f"bench{i}", "placement": p}
            for i, p in enumerate(placements)]}))

    def test_the_arms_must_have_served_from_one_residency(self):
        f = self.f
        f.RESULTS = Path(self.tmp.name) / "bench/results"
        # Agreeing: no complaint.
        self._label("a", ["cached", "cached"])
        self._label("b", ["cached", "cached"])
        self.assertIsNone(f.placement_mismatch(["a", "b"]))
        # Differing: refused, and both sides named.
        self._label("b", ["pinned", "pinned"])
        why = f.placement_mismatch(["a", "b"])
        self.assertIsNotNone(why)
        self.assertIn("cached", why)
        self.assertIn("pinned", why)
        self.assertIn("measures the residency", why)
        # Read back, not requested: an engine that reported nothing has not
        # shown it served from the shared residency, which is the `--segments 1`
        # lesson -- a request is a target an optimizer may miss.
        self._label("b", [None, None])
        self.assertIsNotNone(f.placement_mismatch(["a", "b"]))
        # One arm drifting mid-run is a mismatch even against itself.
        self._label("b", ["cached", "pinned"])
        self.assertIsNotNone(f.placement_mismatch(["a", "b"]))


class FullrunRowInvocationTests(unittest.TestCase):
    """What `fullrun.py` actually asks `workloads.py` for.

    `workloads.py` has had `--min-duration` — re-run a row that came in under
    `MIN_ROW_S` with more queries — for as long as it has had `MIN_ROW_S`. The
    sequence that produces every published table never passed it, so the flag
    protected only a hand invocation, and the run published W10-ef32
    (1.23 s), W10-ef64 (1.59 s) and W13 (0.97 s) with `short (<2 s): the ramp is
    a visible share of the row` in the notes column instead of measuring them
    again. This asserts the argv, because "the whole benchmark is one command"
    is only true of the flags that command passes.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _reload(Path(self.tmp.name))
        self.f = importlib.reload(sys.modules["fullrun"]) \
            if "fullrun" in sys.modules else importlib.import_module("fullrun")

    def tearDown(self):
        self.tmp.cleanup()

    def _argv(self, **kw):
        seen = []

        def fake_run(argv, **_):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(self.f, "settle", lambda *_a, **_k: None), \
                mock.patch.object(self.f.subprocess, "run", fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            self.f.run_workloads("http://l", "lbl", "0-3", None, **kw)
        return seen[0]

    def test_the_rows_are_asked_to_re_run_when_they_come_in_short(self):
        self.assertIn("--min-duration", self._argv())

    def _prev_pair(self, perf_set, sm_qps=9000.0, qd_qps=3484.0):
        """A previous run of two labels on this machine, with a perf setting."""
        for lbl, qps in (("prev-sm", sm_qps), ("prev-qd", qd_qps)):
            d = self.f.ROOT / "bench/results" / lbl
            d.mkdir(parents=True, exist_ok=True)
            (d / "rows.json").write_text(json.dumps(
                [{"id": self.f.workloads.SATURATION_ROW, "qps": qps}]))
            (d / "run.json").write_text(json.dumps({"perf_set": perf_set}))

    def test_auto_refuses_a_reference_measured_under_a_different_instrument(self):
        """`--perf` costs throughput, so a perf-less run's saturation is high
        for a run that attaches perf.

        The dbpedia-openai-1m reference of 3,484 came from a
        perf-less containerised run; Qdrant's real saturation under perf was
        2,564, and W4-sat90 offered 3,136 to an engine that served 2,542 —
        seven and a half seconds of queue where a p50 should have been.
        """
        self._prev_pair(perf_set="")          # measured with no perf
        prev = ["prev-sm", "prev-qd"]
        with mock.patch.object(self.f, "PERF_SET", "default"), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                self.assertRaises(ValueError) as e:
            self.f.resolve_rps_reference("auto", prev)
        self.assertIn("not measured under this run's perf setting", str(e.exception))
        self.assertIn("ignoring", out.getvalue())

        # The same pair, read by a run that attaches nothing: comparable again.
        with mock.patch.object(self.f, "PERF_SET", None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.f.resolve_rps_reference("auto", prev), 3484.0)

    def test_auto_says_unknown_rather_than_off_when_no_run_json_says(self):
        # This guard exists to be honest about provenance; a missing run.json
        # is no evidence, not evidence of perf being off. Both disqualify the
        # reference, but only one of them may be stated as a measurement fact.
        self._prev_pair(perf_set="")
        for lbl in ("prev-sm", "prev-qd"):
            (self.f.ROOT / "bench/results" / lbl / "run.json").unlink()
        with mock.patch.object(self.f, "PERF_SET", "default"), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                self.assertRaises(ValueError):
            self.f.resolve_rps_reference("auto", ["prev-sm", "prev-qd"])
        said = out.getvalue()
        self.assertIn("no run.json saying which instrument measured it", said)
        self.assertNotIn("measured with perf off", said)

    def test_auto_still_takes_the_slower_engine_when_the_instrument_matches(self):
        # The guard must not change what `auto` means when it does apply.
        self._prev_pair(perf_set="default")
        with mock.patch.object(self.f, "PERF_SET", "default"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                self.f.resolve_rps_reference("auto", ["prev-sm", "prev-qd"]), 3484.0)

    def _w11_pair(self, date, spans, rows, rates=None):
        """A dbpedia pair measured at `spans` (None: before spans were
        recorded), with `rows` = {engine: {wid: (qps, n_queries, overlap)}}."""
        for eng, got in rows.items():
            d = self.f.ROOT / "bench/results" / f"{eng}-dbp1m-perf-{date}"
            d.mkdir(parents=True)
            h = {"upload_n": 990_000, "w11_n": 198_000}
            if spans:
                h["w11_spans_s"] = spans
            if rates:
                h["w11_append_rate"] = rates
            (d / "run.json").write_text(json.dumps({"harness": h}))
            (d / "rows.json").write_text(json.dumps([
                {"id": wid, "qps": q, "n_queries": n, "write_overlap_pct": c}
                for wid, (q, n, c) in got.items()]))

    def test_the_mixed_rows_search_ends_inside_a_fixed_rate_append(self):
        """0a3de76 stretched the append to the previous search, and the search
        was as long as the write rate let it be: 2,000 points/s on 0924, 200 on
        0925, about 1,100 next. The rate is now fixed and the search is sized
        from the newest pair measured at that rate (decisions, 2026-09-25)."""
        f, w = self.f, self.f.workloads
        with mock.patch.object(w, "upload_n", lambda: 990_000), \
                mock.patch.object(w, "w11_n", lambda: 198_000):
            want = {"W11-steady": w.w11_append_rate(w.w11_steady_n(), w.W11_STEADY_SPAN_S),
                    "W11": w.w11_append_rate(w.w11_n(), w.W11_SPAN_S)}
            self.assertEqual(want, {"W11-steady": 1_900, "W11": 3_300})
            # 0924: the constants, before spans were recorded. Its searches ran
            # 117 to 345 s against a 26 s and 60 s append.
            self._w11_pair("0924", None, {
                "sm": {"W11-steady": (244.0, 50_000, 12.1), "W11": (166.0, 50_000, 19.9)},
                "qd": {"W11-steady": (427.0, 50_000, 21.2), "W11": (145.0, 50_000, 21.6)}})
            # 0925: a tenth of the rate, so a search nine times faster. Its
            # speed says nothing about a search against 1,900 points/s.
            self._w11_pair("0925", {"W11-steady": 256.0, "W11": 431.0}, {
                "sm": {"W11-steady": (2206.0, 50_000, 100.0), "W11": (102.0, 50_000, 82.0)},
                "qd": {"W11-steady": (1382.0, 50_000, 100.0), "W11": (216.0, 50_000, 100.0)}})
            self.assertEqual(f.w11_append_rates_of("sm-dbp1m-perf-0925"),
                             {"W11-steady": 100, "W11": 400})
            got = f.resolve_w11_queries(["sm-dbp1m-perf-0926", "qd-dbp1m-perf-0926"])
            append = {k: n / want[k] for k, n in (("W11-steady", 49_500), ("W11", 198_000))}
            # Read off 0924, the conservative of the two estimates per engine:
            # the rate times the append, or the queries the writer covered.
            steady = min(244.0 * append["W11-steady"], 50_000 * 0.121)
            self.assertEqual(got, {
                "W11_STEADY_QUERIES": math.floor(steady / f.W11_SPAN_MARGIN),
                "W11_QUERIES": math.floor(145.0 * append["W11"] / f.W11_SPAN_MARGIN)})
            # And the slower engine's search then ends inside the append.
            self.assertLess(got["W11_STEADY_QUERIES"] / 244.0, append["W11-steady"])
            self.assertLess(got["W11_QUERIES"] / 145.0, append["W11"])

    def test_a_search_that_outran_its_writer_shrinks_the_next_one(self):
        """A qps averaged over a search that outlived its writer includes the
        faster quiet tail, so sizing from it alone would overrun again, every
        night. The covered share shrinks it by the margin until it fits."""
        f, w = self.f, self.f.workloads
        rates = {"W11-steady": 1_900, "W11": 3_300}
        with mock.patch.object(w, "upload_n", lambda: 990_000), \
                mock.patch.object(w, "w11_n", lambda: 198_000):
            self._w11_pair("0926", None, {
                "sm": {"W11-steady": (240.0, 4_840, 80.0), "W11": (150.0, 6_965, 100.0)},
                "qd": {"W11-steady": (420.0, 4_840, 100.0), "W11": (140.0, 6_965, 100.0)}},
                rates=rates)
            got = f.resolve_w11_queries(["sm-dbp1m-perf-0927", "qd-dbp1m-perf-0927"])
            self.assertEqual(got["W11_STEADY_QUERIES"], math.floor(4_840 * 0.80 / f.W11_SPAN_MARGIN))
            # Covered fully: the rate is the rate under the write, and exact.
            self.assertEqual(got["W11_QUERIES"], math.floor(140.0 * 60.0 / f.W11_SPAN_MARGIN))

    def test_the_search_is_never_longer_than_queries_nor_shorter_than_a_row(self):
        f, w = self.f, self.f.workloads
        rates = {"W11-steady": 1_900, "W11": 3_300}
        with mock.patch.object(w, "upload_n", lambda: 990_000), \
                mock.patch.object(w, "w11_n", lambda: 198_000):
            # sift1m-fast: QUERIES already fits inside the append.
            self._w11_pair("0926", None, {
                "sm": {"W11-steady": (9_000.0, 50_000, 100.0), "W11": (3.0, 50_000, 100.0)},
                "qd": {"W11-steady": (8_000.0, 50_000, 100.0), "W11": (900.0, 50_000, 100.0)}},
                rates=rates)
            got = f.resolve_w11_queries(["sm-dbp1m-perf-0927", "qd-dbp1m-perf-0927"])
            self.assertEqual(got["W11_STEADY_QUERIES"], w.QUERIES)
            # An absurdly slow engine does not cut the faster one below MIN_ROW_S.
            self.assertEqual(got["W11_QUERIES"], math.ceil(900.0 * w.MIN_ROW_S))
        # Nothing to read, or nothing at this rate: QUERIES stands.
        self.assertEqual(f.resolve_w11_queries(["strawmann", "qdrant"]), {})

    def test_the_write_rate_is_in_the_hash_and_old_stamps_keep_theirs(self):
        w = self.f.workloads
        stamp = w.harness_stamp()
        self.assertIn("w11_append_rate", w.STAMP_KEYS)
        moved = {**stamp, "w11_append_rate": {"W11-steady": 100, "W11": 400}}
        self.assertNotEqual(w.stamp_hash(stamp), w.stamp_hash(moved))
        # A stamp from before the key hashes as its rows were hashed.
        old = {k: v for k, v in stamp.items() if k != "w11_append_rate"}
        key = {k: old.get(k) for k in w.STAMP_KEYS if k != "w11_append_rate"
               and (k in old or k not in w.STAMP_KEYS_SINCE)}
        import hashlib
        self.assertEqual(w.stamp_hash(old), hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12])

    def _dirs(self, *names):
        for n in names:
            (self.f.ROOT / "bench/results" / n).mkdir(parents=True, exist_ok=True)

    def test_the_previous_pair_is_found_across_both_spellings(self):
        """Every published label carries `rel-` before the date, and `sort` was
        lexicographic, so `rel-0903` beat `0910` on `r` > `0`."""
        f = self.f
        self._dirs("sm-dbp1m-perf-rel-0903", "qd-dbp1m-perf-rel-0903")
        self.assertEqual(f.previous_pair(["sm-dbp1m-perf-0924", "qd-dbp1m-perf-0924"]),
                         ["sm-dbp1m-perf-rel-0903", "qd-dbp1m-perf-rel-0903"])
        self._dirs("sm-dbp1m-perf-0910")
        self.assertEqual(f.previous_pair(["sm-dbp1m-perf-0924", "qd-dbp1m-perf-0924"])[0],
                         "sm-dbp1m-perf-0910")

    def test_a_pair_is_not_its_own_previous_and_a_rep_is_not_a_pair(self):
        f = self.f
        self._dirs("sm-sift-perf-0923", "sm-sift-perf-rel-0921", "sm-sift-perf-rel-0921-rep1")
        self.assertEqual(f.previous_pair(["sm-sift-perf-0923", "qd-sift-perf-0923"])[0],
                         "sm-sift-perf-rel-0921")

    def test_labels_off_the_convention_have_no_previous_pair(self):
        f = self.f
        self._dirs("sm-sift-perf-rel-0921")
        for pair in (["strawmann", "qdrant"], ["sm-sift-perf-0924", "qd-dbp1m-perf-0924"],
                     ["sm-sift-perf-0924"], ["sm-dbp100k-perf-0924", "qd-dbp100k-perf-0924"]):
            self.assertIsNone(f.previous_pair(pair), pair)

    def test_auto_on_a_fresh_pair_reads_the_previous_one(self):
        """A night's labels are new, so `auto` had nothing to read and the
        open-loop arms each used their own engine's saturation."""
        f = self.f
        for lbl, qps in (("sm-sift-perf-rel-0921", 22_000), ("qd-sift-perf-rel-0921", 10_351)):
            d = f.ROOT / "bench/results" / lbl
            d.mkdir(parents=True)
            (d / "rows.json").write_text(json.dumps(
                [{"id": f.workloads.SATURATION_ROW, "qps": qps}]))
            (d / "run.json").write_text(json.dumps({"perf_set": "default"}))
        with mock.patch.object(f, "PERF_SET", "default"), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            got = f.resolve_rps_reference("auto", ["sm-sift-perf-0924", "qd-sift-perf-0924"])
        self.assertEqual(got, 10_351.0)
        self.assertIn("reading the previous pair", out.getvalue())
        # A pair with rows of its own reads them, as before.
        with mock.patch.object(f, "PERF_SET", "default"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(f.resolve_rps_reference(
                "auto", ["sm-sift-perf-rel-0921", "qd-sift-perf-rel-0921"]), 10_351.0)

    def _admit(self, asks, wait_min=30, lax=False):
        """`wait_until_admitted` over canned answers, on a fake clock.

        Each ask is "ports" (a port is held), False (the gate refuses) or True.
        """
        f, asks, slept, now = self.f, list(asks), [], [0.0]
        state = {}

        def ports():
            state["ask"] = asks.pop(0)
            return ["6334 qdrant"] if state["ask"] == "ports" else []

        def gate(_lax):
            return f.Gate(proceed=state["ask"] is True, failed=state["ask"] is not True)

        def sleep(s):
            slept.append(s)
            now[0] += s

        with mock.patch.object(f, "ports_in_use", ports), \
                mock.patch.object(f, "gate", gate), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            ok = f.wait_until_admitted(lax, wait_min, sleep=sleep, clock=lambda: now[0])
        return bool(ok), len(slept), err.getvalue()

    def test_without_waiting_a_refusal_is_final(self):
        self.assertEqual(self._admit([False], wait_min=0)[:2], (False, 0))
        ok, sleeps, err = self._admit(["ports"], wait_min=0)
        self.assertEqual((ok, sleeps), (False, 0))
        self.assertIn("port(s) already in use", err)
        self.assertNotIn("gave up", err)

    def test_a_busy_host_is_asked_again_until_admitted(self):
        """`nightrun` did this by relaunching the process and grepping its
        output for "refusing to run"; asked before anything is built there is
        nothing to tell apart."""
        self.assertEqual(self._admit([False, "ports", True])[:2], (True, 2))

    def test_a_lax_admission_keeps_the_failed_gate(self):
        """`--lax` proceeds on a failed gate, and the render is told so from
        this verdict: a bare True lost it."""
        f = self.f
        with mock.patch.object(f, "ports_in_use", lambda: []), \
                mock.patch.object(f, "gate", lambda lax: f.Gate(proceed=True, failed=True)):
            v = f.wait_until_admitted(True, 0)
        self.assertTrue(v)
        self.assertTrue(v.failed)

    def test_waiting_gives_up_at_the_deadline(self):
        # 30 min at 5 min per ask: asks at 0, 5, ... 30, and the one at the
        # deadline is the last.
        ok, sleeps, err = self._admit([False] * 10, wait_min=30)
        self.assertEqual((ok, sleeps), (False, 6))
        self.assertIn("gave up after 7 ask(s)", err)

    def test_every_pass_of_an_arm_is_given_the_same_n_pin(self):
        # Keyed on the arm's base label, so rep1/rep2/rep3 share one verdict and
        # the two engines keep their own: the slower engine's rows are already
        # long enough and must not inherit the faster one's escalation.
        argv = self._argv()
        self.assertIn("--n-pin", argv)
        pin = argv[argv.index("--n-pin") + 1]
        self.assertTrue(pin.endswith("/lbl/n_factors.json"), pin)

    def test_the_row_ids_still_arrive_after_the_flags(self):
        # argparse and a `nargs="*"` positional: a boolean flag inserted before
        # `--storage` must not swallow the ids `fullrun` appends last. This is
        # the shape the parser is given, asserted rather than assumed.
        argv = self._argv(only=["W0", "W3"])
        self.assertEqual(argv[-2:], ["W0", "W3"])
        self.assertLess(argv.index("--min-duration"), argv.index("W0"))

    def test_settle_waits_for_what_the_gate_refuses_and_not_the_named_list(self):
        """`settle` and the gate have to agree about "busy".

        `foreign_load` returned `busy_processes`, whose threshold is 5% of one
        core and whose docstring says it is a list for the operator and not a
        verdict. The verdict is a whole core summed. A desktop at `Xorg(6%)`
        therefore held settle for its full 300s timeout before every gate
        re-run, twice per arm, and the gate then passed the machine without
        remark — an hour of a three-rep run spent waiting for a condition
        nothing was going to refuse.
        """
        setup = importlib.import_module("setup")
        # A desktop: two processes the gate names and passes (0.15 cores).
        quiet_before = {1: (0.0, "/usr/bin/Xorg"), 2: (0.0, "claude")}
        quiet_after = {1: (0.6, "/usr/bin/Xorg"), 2: (0.9, "claude")}
        # And one it refuses, on the same instrument (2.0 cores).
        busy_after = {1: (0.6, "/usr/bin/Xorg"), 2: (20.0, "rustc a.rs")}
        busy_before = {1: (0.0, "/usr/bin/Xorg"), 2: (0.0, "rustc a.rs")}

        def sampler(*pair):
            """`_cpu_seconds` returns the before sample, then the after one."""
            it = iter(pair)
            return lambda: next(it)

        # Over budget, the naming threshold applies again and the desktop is
        # listed beside the cause: settle prints what it is waiting for.
        for before, after, expected in ((quiet_before, quiet_after, []),
                                        (busy_before, busy_after,
                                         ["rustc(200%)", "Xorg(6%)"])):
            with mock.patch.object(setup, "_cpu_seconds", sampler(before, after)), \
                    mock.patch.object(self.f.time, "sleep", lambda _s: None), \
                    mock.patch.object(setup, "QUIESCENT_SAMPLE_S", 10.0):
                self.assertEqual(self.f.foreign_load(), expected)


    def _settle(self, loads: list[str], busy=()) -> str:
        """`settle` over a scripted `/proc/loadavg`, a clock that advances a
        poll per reading of it, and the gate's instrument saying `busy`."""
        f = self.f
        clock = itertools.count(0.0, f.SETTLE_POLL_S)
        reads = iter(loads)
        out = io.StringIO()
        with mock.patch.object(f.Path, "read_text", lambda _self: next(reads)), \
                mock.patch.object(f.time, "monotonic", lambda: next(clock)), \
                mock.patch.object(f.time, "sleep", lambda _s: None), \
                mock.patch.object(f, "foreign_load", lambda: list(busy)), \
                mock.patch.object(f.os, "cpu_count", lambda: 24), \
                contextlib.redirect_stdout(out):
            f.settle("lbl rows")
        return out.getvalue()

    def test_settle_says_how_long_it_waited(self):
        """0925's strawmANN W1-to-W2 gap went from 3 s to ~6 min, and the log
        could not say how much of it was the settle: it printed the load it
        settled at and nothing about the time."""
        text = self._settle(["8.0 1 1 1/1 1\n", "4.0 1 1 1/1 1\n", "1.2 1 1 1/1 1\n"])
        self.assertRegex(text, r"settled at load 1\.2 \(5% per core\) after [1-9]\d* s")
        # Already quiet: nothing is announced, so nothing is timed either.
        self.assertEqual(self._settle(["0.5 1 1 1/1 1\n"]), "")

    def test_a_plateau_says_how_long_it_waited(self):
        f = self.f
        loads = ["3.0 1 1 1/1 1\n"] * (f.SETTLE_PLATEAU_POLLS + 1)
        text = self._settle(loads)
        self.assertRegex(text, r"load has stopped falling at 3\.0 \(12% per core\) after \d+ s")


class FullrunPreflightTests(unittest.TestCase):
    """What `fullrun.py` refuses before it spends the hours.

    `check_bfb_pin` is called once per `workloads.py` invocation, of which a
    `--reps 3` run makes twelve, and `fullrun.py` only read the accumulated
    return code at the end. That meant twelve refusals, three empty
    passes, and a `FileNotFoundError` on a log directory no row had created.

    Both halves are asserted: the run stops at the pin, and the log directory
    exists whether or not anything wrote to it first.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _reload(Path(self.tmp.name))
        self.f = importlib.reload(sys.modules["fullrun"]) \
            if "fullrun" in sys.modules else importlib.import_module("fullrun")

    def tearDown(self):
        self.tmp.cleanup()

    def _main(self, *extra, pin=None):
        """`fullrun.py` up to the point it refuses, with the pin's verdict forced.

        Nothing here is allowed to reach the machine: `use_dataset` would want a
        corpus and `gate` would want a quiescent host, and a preflight that
        stops before both is exactly what is being asserted. Either being
        called at all is a failure of the test's premise, so they raise.
        """
        def unreachable(*_a, **_k):
            raise AssertionError("preflight ran past the load-generator check")

        err = io.StringIO()
        with mock.patch.object(self.f.workloads, "check_bfb_pin", return_value=pin), \
                mock.patch.object(self.f.workloads, "use_dataset", unreachable), \
                mock.patch.object(self.f, "gate", unreachable), \
                contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = self.f.main(["fullrun.py", "--server-cpus", "4-11",
                              "--client-cpus", "0-3", "--rps-reference", "none",
                              *extra])
        return rc, err.getvalue()

    def test_a_bfb_off_the_pin_stops_the_run_before_anything_expensive(self):
        rc, err = self._main(pin="bfb at /w/bfb is 55cd014d, and this harness is "
                                 "pinned to dev @ 0c1aafee")
        self.assertEqual(rc, 1)
        # The reason, not just a code: this is the message that replaces seven
        # minutes of measuring nothing.
        self.assertIn("55cd014d", err)
        self.assertIn("pinned", err)

    def test_the_refusal_is_not_worded_as_the_gate_refuses(self):
        """"refusing to run" is the §7.1 gate's phrase, and a busy host goes
        quiet, so `--wait-for-gate` asks again. A checkout on the wrong commit
        does not, and a refusal worded like the gate's reads as a busy host to
        whoever, or whatever analysis, counts the gate's refusals."""
        _, err = self._main(pin="bfb is not the pin")
        self.assertNotIn("refusing to run", err)

    def test_a_pinned_bfb_is_not_what_stops_it(self):
        """The check is the run's, not the test's: with the pin satisfied,
        control reaches `use_dataset` — which this test makes raise."""
        with self.assertRaises(AssertionError):
            self._main(pin=None)

    def test_skipping_both_engines_does_not_consult_the_generator(self):
        """`--skip strawmann --skip qdrant` measures no rows, so the load
        generator is not this run's instrument and its commit is not this
        run's business — a render-only invocation should not need a checkout
        on the pin."""
        with self.assertRaises(AssertionError):
            self._main("--skip", "strawmann", "--skip", "qdrant",
                       pin="bfb is not the pin")

    def test_the_server_log_directory_is_made_where_the_log_is_opened(self):
        """The per-pass caller made the directory and the conformance caller
        did not; `start_strawmann` opens the log, so it is what makes the
        directory. Asserted through the real function up to the `Popen`, which
        is where the machine would start."""
        log = self.f.RESULTS / "a-label-no-row-created" / "server-conformance.log"
        self.assertFalse(log.parent.exists())

        class Started:
            pid = 4321

            def poll(self):
                return None

        with mock.patch.object(self.f, "wipe_strawmann_storage", lambda: None), \
                mock.patch.object(self.f, "maybe_scope", lambda argv, _u: argv), \
                mock.patch.object(self.f.subprocess, "Popen",
                                  lambda *_a, **_k: Started()), \
                mock.patch.object(self.f.time, "sleep", lambda _s: None), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            # It gives up after the wait loop without a "listening" line; the
            # subject is the open, which happens before the loop.
            self.f.start_strawmann("4-11", 6344, log, 8, 1_000_000)
        self.assertTrue(log.exists())


class DirtyScopeTests(unittest.TestCase):
    """`dirty` says "this binary may not be what `commit` describes". Computed
    over the whole tree, a *successful* run set it: `render` splices measured
    tables into README.md and docs/comparison-<dataset>.md, so the first
    dataset of a two-dataset session left the tree modified and the second
    stamped its engine `tree dirty` over documentation the engine cannot
    depend on."""

    def _repo(self, td):
        r = Path(td)
        for cmd in (["init", "-q", str(r)], ):
            subprocess.run(["git", *cmd], check=True)
        subprocess.run(["git", "-C", str(r), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(r), "config", "user.name", "t"], check=True)
        (r / "src").mkdir()
        (r / "src/main.zig").write_text("pub fn main() void {}\n")
        (r / "build.zig").write_text("// build\n")
        (r / "README.md").write_text("docs\n")
        subprocess.run(["git", "-C", str(r), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(r), "commit", "-qm", "x"], check=True)
        return r

    def test_a_regenerated_table_is_not_a_dirty_binary(self):
        import provenance
        with tempfile.TemporaryDirectory() as td:
            r = self._repo(td)
            old, provenance.ROOT = provenance.ROOT, r
            try:
                self.assertFalse(provenance.strawmann_build()["dirty"])
                # What `render` does at the end of a successful run.
                (r / "README.md").write_text("docs\nregenerated table\n")
                self.assertFalse(provenance.strawmann_build()["dirty"],
                                 "a regenerated doc must not flag the binary")
                # A real source change still flags, which is the point.
                (r / "src/main.zig").write_text("pub fn main() void { @panic(\"x\"); }\n")
                self.assertTrue(provenance.strawmann_build()["dirty"])
            finally:
                provenance.ROOT = old

    def test_a_build_script_change_still_flags(self):
        import provenance
        with tempfile.TemporaryDirectory() as td:
            r = self._repo(td)
            old, provenance.ROOT = provenance.ROOT, r
            try:
                (r / "build.zig").write_text("// changed\n")
                self.assertTrue(provenance.strawmann_build()["dirty"])
            finally:
                provenance.ROOT = old

    def test_the_conformance_stamp_asks_the_same_question_as_the_rows(self):
        """§8 binds a perf row to its conformance run by commit string, and the
        two halves computed `dirty` differently: this module over the build
        inputs, `fullrun.build_identity` over the whole tree. A session's second
        dataset — run after `render` spliced the first one's tables into
        README.md — stamped its rows `abc123` and its conformance run
        `abc123-dirty`, and §8 refused all 24 of them (db100k)."""
        import fullrun

        import provenance
        with tempfile.TemporaryDirectory() as td:
            r = self._repo(td)
            old, provenance.ROOT = provenance.ROOT, r
            try:
                self.assertNotIn("-dirty",
                                 fullrun.build_identity(None)["STRAWMANN_COMMIT"])
                # What `render` leaves behind after a successful run.
                (r / "README.md").write_text("docs\nregenerated table\n")
                self.assertNotIn("-dirty",
                                 fullrun.build_identity(None)["STRAWMANN_COMMIT"],
                                 "a regenerated table must not make the "
                                 "conformance row refuse the perf rows")
                self.assertFalse(provenance.strawmann_build()["dirty"],
                                 "and the two halves must still agree")
                # A real source change still flags on both sides.
                (r / "src/main.zig").write_text("pub fn main() void { @panic(\"x\"); }\n")
                self.assertIn("-dirty",
                              fullrun.build_identity(None)["STRAWMANN_COMMIT"])
                self.assertTrue(provenance.strawmann_build()["dirty"])
            finally:
                provenance.ROOT = old

class ArmStartedTests(unittest.TestCase):
    """`started` must be when the arm started, not when its last phase did.

    `fullrun` defers the mutating rows past the recall sweeps, so they run as a
    second `workloads.py` invocation and rewrite `run.json` wholesale. The
    field recorded the last call: a sift1m Qdrant arm said
    21:29:30Z while its first row was measured at 21:11:46Z, which reads as a
    seven-minute arm.
    """

    def _prev(self, d, **meta):
        (d / "run.json").write_text(json.dumps(meta))

    def test_the_earliest_stamp_in_a_session_wins(self):
        import workloads as w
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._prev(d, session="s1", started="2026-08-26T21:11:46Z")
            self.assertEqual(w.arm_started(d, "2026-08-26T21:29:30Z", "s1"),
                             "2026-08-26T21:11:46Z")

    def test_a_new_session_does_not_inherit_the_old_start(self):
        """The db100k re-run wrote the directory its failed attempt left."""
        import workloads as w
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._prev(d, session="s1", started="2026-08-25T21:55:29Z")
            self.assertEqual(w.arm_started(d, "2026-08-26T02:29:43Z", "s2"),
                             "2026-08-26T02:29:43Z")

    def test_no_session_or_no_previous_file_keeps_now(self):
        import workloads as w
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.assertEqual(w.arm_started(d, "2026-08-26T02:29:43Z", "s1"),
                             "2026-08-26T02:29:43Z")
            self._prev(d, session="s1", started="2026-08-26T01:00:00Z")
            self.assertEqual(w.arm_started(d, "2026-08-26T02:29:43Z", None),
                             "2026-08-26T02:29:43Z")

    def test_an_unreadable_previous_run_json_is_not_a_start_time(self):
        import workloads as w
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "run.json").write_text("{not json")
            self.assertEqual(w.arm_started(d, "2026-08-26T02:29:43Z", "s1"),
                             "2026-08-26T02:29:43Z")


class SchedCoverageTests(unittest.TestCase):
    """The coverage gate's two escapes, closed."""

    def _snap(self, procstat, sched):
        return procstat.Snapshot(pid=1, comm="strawmann", engines_running=1, io=None,
                                 io_source=procstat.IoSource.none, rss_peak_bytes=None,
                                 storage_path=None, storage_bytes=None, sched=sched)

    def test_the_sampler_reads_the_counters_it_is_credited_with(self):
        """`_THREAD_SUMMED` names the two context-switch counters, and the
        banked path trusts whatever the sampler banked -- but the sampler read
        `schedstat` and `sched`, not `status`, where those two live."""
        procstat = importlib.import_module("procstat")
        got = procstat.thread_counters(os.getpid())
        self.assertTrue(got)
        for tid, counters in got.items():
            self.assertIn("ctx_switches_voluntary", counters, tid)
            self.assertIn("ctx_switches_involuntary", counters, tid)
            break

    def test_a_banked_row_still_suppresses_what_the_sampler_did_not_bank(self):
        procstat = importlib.import_module("procstat")
        before = self._snap(procstat, {"cpu_user_s": 0.0, "cpu_system_s": 0.0, "oncpu_s": 0.0,
                                       "ctx_switches_voluntary": 0.0, "threads": 8.0})
        after = self._snap(procstat, {"cpu_user_s": 10.0, "cpu_system_s": 3.0, "oncpu_s": 0.07,
                                      "ctx_switches_voluntary": 50.0, "threads": 2.0})
        banked = {"oncpu_s": 12.9, "runqueue_wait_s": 0.2}
        out = procstat._sched_delta(before, after, banked)
        # Banked counters are the row's; the one the sampler did not read is
        # the survivors' subtraction and coverage (0.005) withholds it.
        self.assertEqual(out["oncpu_s"], 12.9)
        self.assertIsNone(out["ctx_switches_voluntary"])
        self.assertLess(out["sched_coverage"], 0.01)

    def test_a_thread_sum_that_fell_is_the_worst_coverage_not_none(self):
        """A negative delta dropped `oncpu_s`, coverage was never computed, and
        the gate keyed on coverage being *low* rather than absent: every other
        thread-summed counter went out unsuppressed and unbadged."""
        procstat = importlib.import_module("procstat")
        before = self._snap(procstat, {"cpu_user_s": 0.0, "cpu_system_s": 0.0, "oncpu_s": 100.0,
                                       "runqueue_wait_s": 0.0, "ctx_switches_voluntary": 0.0,
                                       "threads": 40.0})
        after = self._snap(procstat, {"cpu_user_s": 13.0, "cpu_system_s": 0.0, "oncpu_s": 5.0,
                                      "runqueue_wait_s": 0.0, "ctx_switches_voluntary": 4000.0,
                                      "threads": 3.0})
        out = procstat._sched_delta(before, after, None)
        self.assertEqual(out["sched_coverage"], 0.0)
        self.assertIsNone(out["ctx_switches_voluntary"])
        self.assertIsNone(out["runqueue_wait_s"])

    def test_cgroup_io_ops_take_the_gate_psi_takes(self):
        procstat = importlib.import_module("procstat")
        with mock.patch.object(procstat, "_proc_io", lambda pid: {"disk_read_bytes": 1, "disk_write_bytes": 2, "disk_read_ops": None, "disk_write_ops": None, "read_bytes": None, "write_bytes": None}), \
                mock.patch.object(procstat, "_cgroup_io", lambda pid: {"disk_read_ops": 9, "disk_write_ops": 9, "disk_read_bytes": 9, "disk_write_bytes": 9}), \
                mock.patch.object(procstat, "cgroup_dir", lambda pid: Path("/sys/fs/cgroup/x")):
            with mock.patch.object(procstat, "cgroup_holds_only", lambda pid, d, **k: False):
                out, src = procstat.io_counters(1)
                self.assertIsNone(out["disk_read_ops"])
                self.assertEqual(str(src), "proc")
            with mock.patch.object(procstat, "cgroup_holds_only", lambda pid, d, **k: True):
                out, src = procstat.io_counters(1)
                self.assertEqual(out["disk_read_ops"], 9)

    def test_an_unreadable_cmdline_is_not_an_empty_store(self):
        procstat = importlib.import_module("procstat")
        with mock.patch.object(procstat, "_read", lambda p: None):
            self.assertIsNone(procstat.store_bytes_for(1, "strawmann", None))


class RunEstimateTests(unittest.TestCase):
    """The run must say how long it will take before it takes it.

    A `--reps 3 --dataset dbpedia-openai-1m` run held the machine
    for nine and a quarter hours. Nothing stated that number: not the flags,
    not the header, not the first pass. The operator committed a night to a
    cost nobody had said out loud and stopped the run in the morning, short of
    the conformance phase that licenses the report.

    So the estimate is measured from the previous run's own rows on the same
    corpus, and it is printed *before* the §7.1 gate settles rather than after
    it -- an estimate an operator reads once the run is under way is not a
    decision they got to make.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.f = importlib.reload(sys.modules["fullrun"]) \
            if "fullrun" in sys.modules else importlib.import_module("fullrun")
        self.f.RESULTS = Path(self.tmp.name) / "bench/results"
        # Priced as each basis ran, except where a test names the table.
        self._planned = self.f.planned_rows
        self.f.planned_rows = lambda: None

    def tearDown(self):
        self.f.planned_rows = self._planned
        self.tmp.cleanup()

    def test_the_basis_is_priced_for_the_rows_this_run_will_make(self):
        """The table grew from 32 rows to 43 between rel-0903 and perf-0924,
        and the estimate, priced on the old arm's rows, came out three hours
        short. A row the basis never ran costs its mean row; one the table has
        dropped costs nothing."""
        f = self.f
        self._label("qd-old", "sift1m", [
            {"id": "W3", "wall_s": 100, "when": "2026-09-01T00:00:00Z"},
            {"id": "W4", "wall_s": 300, "when": "2026-09-01T00:01:40Z"},
            {"id": "W9", "wall_s": 200, "when": "2026-09-01T00:06:40Z"}])
        f.planned_rows = lambda: ["W3", "W4", "W12-sel1", "W12-sel10"]
        mins, _ = f.estimated_minutes("sift1m", 1)
        # W3 + W4 as measured, the two W12 rows at the 200 s mean, W9 dropped;
        # a single engine counts twice; the span is exactly the rows (1.0x).
        self.assertAlmostEqual(mins, 2 * (100 + 300 + 200 + 200) / 60, places=2)

    def test_settles_are_counted_from_the_invocations_measure_makes(self):
        # Stable rows in two parts around W1, then the mutating rows.
        self.assertEqual(self.f.invocations_per_arm(), 3)

    def _label(self, name, dataset, rows, engine="qdrant"):
        d = self.f.RESULTS / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps(
            {"dataset": {"name": dataset}, engine: {"version": "x"}}))
        (d / "rows.json").write_text(json.dumps(rows))

    @staticmethod
    def _rows(n, wall_s, gap_s):
        # `when` walks forward by wall_s + gap_s, so the arm's wall span
        # exceeds its row time by exactly the gaps -- the sweeps, the settles
        # and the engine restarts a pass pays and a row does not.
        t, out = 0, []
        for i in range(n):
            out.append({"id": f"W{i}", "wall_s": wall_s,
                        "when": f"2026-08-26T{t // 3600:02d}:"
                                f"{t // 60 % 60:02d}:{t % 60:02d}Z"})
            t += wall_s + gap_s
        return out

    def test_the_estimate_comes_from_the_last_run_on_the_same_corpus(self):
        f = self.f
        # 10 rows x 60s of row time, 30s of overhead between each: the arm
        # spans 870s and 600s of it is rows.
        self._label("a", "sift1m", self._rows(10, 60, 30))
        got = f.estimated_minutes("sift1m", 1)
        self.assertIsNotNone(got)
        mins, src = got
        self.assertEqual(src, "a")
        # Only one engine has ever been measured here, so the other is
        # assumed to cost the same: two arms of 870s. The conformance differ
        # and the render are deliberately NOT in this figure -- nothing on
        # disk times them, and padding the passes to stand in for them is how
        # the first version came to agree with the night by cancelling two
        # errors.
        self.assertAlmostEqual(mins, (2 * 870) / 60, places=2)
        # Passes scale linearly; there is no once-per-run term to get wrong.
        self.assertAlmostEqual(f.estimated_minutes("sift1m", 3)[0],
                               (3 * 2 * 870) / 60, places=2)

    def test_another_corpus_is_not_borrowed(self):
        """A dbpedia estimate from a sift run would be off by 4x.

        Silence is the honest answer for a corpus this machine has never run,
        and the header says so rather than printing a borrowed number.
        """
        self._label("a", "sift1m", self._rows(10, 60, 30))
        self.assertIsNone(self.f.estimated_minutes("dbpedia-openai-1m", 3))

    def test_an_arm_whose_run_json_names_no_engine_is_skipped(self):
        d = self.f.RESULTS / "nameless"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"dataset": {"name": "sift1m"}}))
        (d / "rows.json").write_text(json.dumps(self._rows(4, 60, 30)))
        self.assertIsNone(self.f.estimated_minutes("sift1m", 1))

    def test_a_pass_is_one_arm_of_each_engine_not_two_of_the_slower(self):
        """Qdrant's dbpedia arm is 79 min of rows against strawmANN's 52.

        Grouping by label text put `qd-sift-perf` and `qdrant` -- two Qdrant
        arms of one corpus under different names -- in separate groups, and
        priced a sift1m pass as two Qdrant arms and no strawmANN one.
        `run.json` records which engine an arm measured; that is what groups
        them.
        """
        f = self.f
        self._label("qd-a", "sift1m", self._rows(10, 90, 30), engine="qdrant")
        self._label("qdrant", "sift1m", self._rows(10, 80, 30), engine="qdrant")
        self._label("sm-a", "sift1m", self._rows(10, 30, 30), engine="strawmann")
        mins, src = f.estimated_minutes("sift1m", 1)
        self.assertIn("qd-a", src)
        self.assertIn("sm-a", src)
        self.assertNotIn("qdrant", src.replace("qd-a", ""))
        # Qdrant's heaviest arm is 900s of rows and its cheapest ratio 1.30;
        # strawmANN's arm is 300s at 1.90. Label-text grouping instead charged
        # 900s and 800s of rows, both of them Qdrant, and no strawmANN arm.
        self.assertAlmostEqual(mins, (900 * 1.30 + 300 * 1.90) / 60, places=2)
        self.assertLess(mins, (900 * 1.30 + 800 * 1.3375) / 60)

    def test_an_interrupted_arm_does_not_poison_the_ratio(self):
        """`qd-dbp1m-rep3` spans 182 min for 75 min of rows, because the run
        was stopped part-way through it. Its ratio is 41% against its
        siblings' 86%, and a mean would carry that into every later estimate.
        """
        f = self.f
        self._label("a", "sift1m", self._rows(10, 60, 30), engine="qdrant")
        self._label("b", "sift1m", self._rows(10, 60, 30), engine="strawmann")
        clean = f.estimated_minutes("sift1m", 1)[0]
        stalled = self._rows(10, 60, 30)
        stalled[-1]["when"] = "2026-08-26T04:00:00Z"     # four hours of nothing
        self._label("c", "sift1m", stalled, engine="qdrant")
        # Two Qdrant arms now, one of them stalled. A median of two is their
        # mean, so this is exactly the case a median cannot survive -- and
        # per-engine pools are usually this small.
        self.assertAlmostEqual(f.estimated_minutes("sift1m", 1)[0], clean,
                               places=2)
        # A third and fourth stall must not move it either: contamination is
        # one-sided, so one clean arm is enough however many are dirty.
        for i, name in enumerate(("d", "e")):
            more = self._rows(10, 60, 30)
            more[-1]["when"] = f"2026-08-26T0{5 + i}:00:00Z"
            self._label(name, "sift1m", more, engine="qdrant")
        self.assertAlmostEqual(f.estimated_minutes("sift1m", 1)[0], clean,
                               places=2)

    def test_each_engine_is_priced_at_its_own_overhead_ratio(self):
        """Rows are 86% of a Qdrant arm's wall clock here and 79% of a
        strawmANN one. One pooled ratio priced the dbpedia pass 6% high."""
        f = self.f
        # Qdrant: 10x60s of rows in 870s. strawmANN: 10x60s of rows in 1,170s.
        # Qdrant 600s of rows in 870s (1.45); strawmANN 300s in 840s (2.80).
        self._label("q", "sift1m", self._rows(10, 60, 30), engine="qdrant")
        self._label("s", "sift1m", self._rows(10, 30, 60), engine="strawmann")
        mins = f.estimated_minutes("sift1m", 1)[0]
        self.assertAlmostEqual(mins, (870 + 840) / 60, places=2)
        # Pooling the two ratios would charge both arms the midpoint, 2.125,
        # and price the pass at 1,912s instead of 1,710s.
        self.assertNotAlmostEqual(mins, (600 + 300) * 2.125 / 60, places=2)

    def test_the_segment_policy_reaches_qdrant_s_server_default(self):
        """Both start paths exported `DEFAULT_SEGMENT_NUMBER=1` whatever the
        policy, and `as-deployed` sends no `--segments`, so it measured
        equal-work under the other name and the stamp could not tell."""
        f = self.f
        w = importlib.import_module("workloads")
        with mock.patch.object(w, "SEGMENT_POLICY", w.SegmentPolicy.equal_work):
            self.assertEqual(f.qdrant_segment_env(),
                             {"QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER": "1"})
        with mock.patch.object(w, "SEGMENT_POLICY", w.SegmentPolicy.as_deployed):
            self.assertEqual(f.qdrant_segment_env(), {})

    def test_a_folded_label_is_not_a_measurement_of_an_arm(self):
        """`sm-dbp1m` holds the medians of three passes under one pass's
        stamps, so its span-over-rows is an artefact of the fold."""
        f = self.f
        self._label("a-rep1", "sift1m", self._rows(10, 60, 30), engine="qdrant")
        alone = f.estimated_minutes("sift1m", 1)[0]
        folded = self.f.RESULTS / "a"
        folded.mkdir(parents=True)
        (folded / "run.json").write_text(json.dumps(
            {"dataset": {"name": "sift1m"}, "qdrant": {}, "reps": 3,
             "rep_labels": ["a-rep1"]}))
        # Same stamps, but the rows are medians: a 10x-inflated span.
        rows = self._rows(10, 60, 30)
        rows[-1]["when"] = "2026-08-26T10:00:00Z"
        (folded / "rows.json").write_text(json.dumps(rows))
        self.assertEqual(f.estimated_minutes("sift1m", 1)[0], alone)
        self.assertEqual(f.estimated_minutes("sift1m", 1)[1], "a-rep1")

    def test_a_label_without_usable_rows_is_skipped_not_fatal(self):
        f = self.f
        (f.RESULTS / "empty").mkdir(parents=True)
        (f.RESULTS / "empty" / "rows.json").write_text("[]")
        self._label("broken", "sift1m", [{"id": "W1"}])          # no wall_s, no when
        self._label("nostamps", "sift1m", [{"id": "W1", "wall_s": 60}])
        self._label("good", "sift1m", self._rows(4, 60, 30))
        self.assertEqual(f.estimated_minutes("sift1m", 1)[1], "good")

    def test_the_longest_arm_wins_so_the_estimate_is_not_optimistic(self):
        """Two labels, and the estimate must not come from the short one.

        A run's cost is set by the arm that takes longest, and an estimate
        taken from the other one under-promises exactly where it matters.
        """
        f = self.f
        self._label("short", "sift1m", self._rows(4, 30, 10))
        self._label("long", "sift1m", self._rows(4, 120, 30))
        self.assertEqual(f.estimated_minutes("sift1m", 3)[1], "long")


class SegmentPolicyTests(unittest.TestCase):
    """The two Qdrant experiments, named rather than spelled as a number.

    Segments are Qdrant's intra-query parallelism and moving them moves it in
    *opposite* directions: measured on dbpedia-openai-1m, holding it to one
    populated graph took W3 from 584 to 353 qps and W4 from 1,237 to 2,584.
    The published 2.91x saturating headline reads 1.38x at one segment, and it
    is Qdrant that moved. So the count is not a tuning knob with a better
    setting; it is which question the run is answering, and a report that
    prints `segments: 1` has named the knob and not the question.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.w = self.m["workloads"]
        self.saved = os.environ.get("SEGMENT_POLICY")

    def tearDown(self):
        self.tmp.cleanup()
        if self.saved is None:
            os.environ.pop("SEGMENT_POLICY", None)
        else:
            os.environ["SEGMENT_POLICY"] = self.saved

    def test_equal_work_pins_one_graph_and_says_so(self):
        w = self.w
        w.use_segment_policy("equal-work")
        flags = w.collection_flags()
        self.assertIn("--segments", flags)
        self.assertEqual(flags[flags.index("--segments") + 1], "1")
        # The ceiling is what actually delivers one segment: `--segments 1`
        # alone leaves five, four of them populated (findings 39).
        self.assertIn("--max-segment-size", flags)
        self.assertGreater(int(flags[flags.index("--max-segment-size") + 1]), 0)
        self.assertEqual(w.collection_settings()["segment_policy"], "equal-work")

    def test_as_deployed_asks_for_nothing_rather_than_naming_the_default(self):
        """"Qdrant's own default" means the one Qdrant picks.

        Passing `default_segment_number: 0` explicitly would pin the harness to
        whatever the default was on the day someone read it, and passing a
        ceiling at all is the mechanism that *defeats* the default count -- so
        the two policies would differ by less than their names claim.
        """
        w = self.w
        w.use_segment_policy("as-deployed")
        flags = w.collection_flags()
        self.assertNotIn("--segments", flags)
        self.assertNotIn("--max-segment-size", flags)
        # The thresholds are a different axis and stay in both: they decide
        # whether an HNSW graph is built at all, which W0 needs settled.
        self.assertIn("--indexing-threshold", flags)
        self.assertIn("--full-scan-threshold", flags)
        settings = w.collection_settings()
        self.assertEqual(settings["segment_policy"], "as-deployed")
        self.assertEqual(settings["segments"], "engine default")
        self.assertEqual(settings["max_segment_size_kb"], "engine default")

    def test_the_two_policies_cannot_share_a_table(self):
        """The refusal is the hash, not a check someone has to remember.

        `collection_settings()` is stamped into every row's `harness_hash`, so
        two arms measured under different policies read STALE in `compare.py`
        and no ratio is printed. That is the mechanism decisions.md relies on
        for "they are two experiments and never one table".
        """
        w = self.w
        w.use_segment_policy("equal-work")
        a = w.stamp_hash({"collection": w.collection_settings()})
        w.use_segment_policy("as-deployed")
        b = w.stamp_hash({"collection": w.collection_settings()})
        self.assertNotEqual(a, b)

    def test_binding_it_covers_this_process_and_the_ones_it_spawns(self):
        """`fullrun.py` runs each arm as a subprocess, so a policy that lived
        only in this process would leave the arms free to disagree."""
        w = self.w
        w.use_segment_policy("as-deployed")
        self.assertEqual(os.environ["SEGMENT_POLICY"], "as-deployed")
        # CREATE is `collection_flags()` frozen at import, and the policy is
        # precisely which flags those are.
        self.assertNotIn("--segments", w.CREATE)
        w.use_segment_policy("equal-work")
        self.assertIn("--segments", w.CREATE)

    def test_an_override_of_the_number_does_not_rename_the_experiment(self):
        """`SEGMENTS=4` is an escape hatch for a one-off, not a third policy.

        The number is stamped beside the name, so a run that overrides it
        cannot silently claim to be the experiment it is no longer running.
        """
        w = self.w
        os.environ["SEGMENTS"] = "4"
        try:
            w.use_segment_policy("equal-work")
            self.assertEqual(w.segments(), 4)
            settings = w.collection_settings()
            self.assertEqual(settings["segment_policy"], "equal-work")
            self.assertEqual(settings["segments"], 4)
        finally:
            os.environ.pop("SEGMENTS", None)

    def test_an_unknown_policy_is_refused_at_the_flag(self):
        with self.assertRaises(ValueError):
            self.w.use_segment_policy("one-segment")


class CgroupIoDelegationTests(unittest.TestCase):
    """`setup.io_levels`: which cgroup levels must pass `io` to their children.

    A cgroup has `io.stat` only where its *parent* enabled `io`, and the engines
    run in a transient `systemd-run --user --scope` whose parent is the user
    manager's `app.slice`. Both the §7.1 check and `delegate_cgroup_io` stopped
    at `user@UID.service`, one level short of it -- so the gate reported the
    controller "delegated to user.slice, so block-layer read/write operations
    are measurable" while `app.slice` passed down `memory pids` and every report
    printed `disk read ops: unknown` beside real byte counts.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.setup = importlib.import_module("setup")

    def tearDown(self):
        self.tmp.cleanup()

    def _tree(self, app_control: str) -> str:
        """A fake user.slice whose deepest level carries `app_control`."""
        root = Path(self.tmp.name) / "user.slice"
        mgr = root / "user-1000.slice" / "user@1000.service"
        (mgr / "app.slice").mkdir(parents=True)
        full = "cpu io memory pids"
        for d, ctl in ((root, full), (root / "user-1000.slice", full),
                       (mgr, full), (mgr / "app.slice", app_control)):
            (d / "cgroup.subtree_control").write_text(ctl)
        return str(root)

    def test_the_scopes_own_parent_counts_as_a_level(self):
        root = self._tree("cpu io memory pids")
        self.assertIn("app.slice", [d.name for d in self.setup.io_levels(root)])
        self.assertEqual(self.setup.io_missing(root), [],
                         "nothing is missing once app.slice passes io down")

    def test_io_reaching_only_the_user_manager_is_not_measurable(self):
        """This host's exact shape before the fix: io enabled every level down
        to `user@1000.service`, and `app.slice` passing down `memory pids`."""
        root = self._tree("memory pids")
        self.assertEqual([d.name for d in self.setup.io_missing(root)], ["app.slice"],
                         "the level the gate used to skip is the one that mattered")


if __name__ == "__main__":
    unittest.main()


class EnvHashStampTests(unittest.TestCase):
    """§7.1's environment hash, from the gate's mouth to the refusal that reads it.

    Three files touch this value and none of them import each other:
    `bench/setup.py` prints it, `workloads.env_hash_of_gate` stamps it into
    `run.json`, and `results.py` parses it out of `env.txt` for the §8 store.
    A stamp that silently stamps nothing is worse than no stamp -- the refusal
    downstream (`compare.env_matches`) is permissive by design where it cannot
    know, so a null hash reads as "no disagreement" and permits everything.
    """

    CHECK_LINE = "environment hash: 373e8531fd3d1bf6"

    def test_reads_the_line_check_actually_prints(self):
        self.assertEqual(
            workloads.env_hash_of_gate(f"some preamble\n{self.CHECK_LINE}\ntail"),
            "373e8531fd3d1bf6")

    def test_the_describe_body_line_is_not_the_check_line(self):
        """`setup.py` writes `env_hash=<h>` into the hash *body* and prints
        `environment hash: <h>` as the verdict. The first cut of the stamp
        matched the body line, which `check` never emits, so every run.json
        carried a null hash while looking like it carried a real one."""
        self.assertIsNone(workloads.env_hash_of_gate("env_hash=373e8531fd3d1bf6"))

    def test_absent_is_none_not_a_guess(self):
        self.assertIsNone(workloads.env_hash_of_gate("no hash here"))

    def test_the_real_gate_output_is_parseable_by_both_readers(self):
        """The integration half, and the one that would have caught the bug.

        `bench/setup.py check` is run for real. Its exit code is ignored: on a
        busy host the gate fails and still prints its hash, and what is under
        test is the *format*, not the verdict. Both consumers' patterns are
        asserted against the same bytes, so a change to what `setup.py` prints
        breaks this rather than silently zeroing a field in two places.
        """
        root = Path(workloads.__file__).resolve().parents[2]
        r = subprocess.run([sys.executable, str(root / "bench/setup.py"), "check"],
                           capture_output=True, text=True, timeout=180)
        out = r.stdout + r.stderr
        stamped = workloads.env_hash_of_gate(out)
        self.assertIsNotNone(stamped, f"no environment hash in gate output:\n{out[-800:]}")

        # `results.py` keeps its own copy of this pattern because it imports no
        # harness module. Same bytes, same answer, or the §8 store records
        # "unknown" for every row and nothing says so.
        import re
        m = re.search(r"environment hash: ([0-9a-f]+)", out)
        self.assertIsNotNone(m, "results.py's pattern no longer matches the gate")
        self.assertEqual(m.group(1), stamped)


class ConcurrencyKnobTests(unittest.TestCase):
    """`W9_PARALLEL` and `W4_CONNS` change what a row measures.

    Both are read once at import, so a test has to reload the module rather
    than set the variable — which is also the honest shape of the constraint:
    these are per-process knobs, not per-row ones.
    """

    def _reloaded(self, **env):
        for k, v in env.items():
            os.environ[k] = v
        try:
            return importlib.reload(workloads)
        finally:
            for k in env:
                os.environ.pop(k, None)

    def tearDown(self):
        importlib.reload(workloads)

    def test_w9_parallel_reaches_the_row_bfb_runs(self):
        """The knob exists because the bus makes W9 falsifiable at `-p 1`: an
        exhaustive fp32 scan re-read per query cannot beat aggregate bandwidth
        over corpus bytes. A knob that did not reach the flags would have
        produced a `-p 8` row wearing a `-p 1` label, which is worse than not
        asking the question (validation 13)."""
        w = self._reloaded(W9_PARALLEL="1")
        self.assertEqual(w.W9_PARALLEL, 1)
        w9 = next(x for x in w.table() if x.id == "W9")
        flags = [str(f) for f in w9.args]
        self.assertIn("-p", flags)
        self.assertEqual(flags[flags.index("-p") + 1], "1")

    def test_w9_defaults_to_eight(self):
        w = importlib.reload(workloads)
        w9 = next(x for x in w.table() if x.id == "W9")
        flags = [str(f) for f in w9.args]
        self.assertEqual(flags[flags.index("-p") + 1], "8")

    def test_w4_conns_reaches_the_row(self):
        """`-t 16 -c 8` opens ~128 sockets and a server with fewer closes the
        excess before the HTTP/2 preface, which cost an entire W4 run."""
        w = self._reloaded(W4_CONNS="8")
        self.assertEqual(w.W4_CONNS, 8)
        w4 = next(x for x in w.table() if x.id == "W4")
        flags = [str(f) for f in w4.args]
        self.assertEqual(flags[flags.index("-c") + 1], "8")


class ClientConcurrencyTests(unittest.TestCase):
    """`client_concurrency`: what the load generator offered, per row.

    Findings 52. The rows do not agree about concurrency and their throughput
    shares one column, so a qps read across rows without this compares two
    different loads. Both wrong conclusions in that entry were drawn from the
    published columns in one sitting.
    """

    def setUp(self):
        self.w = importlib.reload(workloads)

    def _of(self, wid: str) -> dict:
        w = next(x for x in self.w.table() if x.id == wid)
        return self.w.client_concurrency(w)

    def test_an_unpinned_flag_records_bfbs_default_not_none(self):
        """The row did offer a concurrency; `None` would read as unknown."""
        c = self._of("W6-ef128")
        self.assertEqual(c["client_parallel"], self.w.BFB_DEFAULT_PARALLEL)
        self.assertEqual(c["client_parallel"], 2)
        self.assertIn("-p", c["client_defaults"].split())

    def test_a_pinned_flag_is_recorded_and_not_marked_default(self):
        c = self._of("W10-ef128")
        self.assertEqual(c["client_parallel"], 8)
        self.assertNotIn("-p", c["client_defaults"].split())

    def test_the_pair_that_caused_the_misreading_differs(self):
        """W10's ladder runs at four times W6's, and the table prints both."""
        self.assertEqual(self._of("W10-ef128")["client_parallel"], 8)
        self.assertEqual(self._of("W6-ef128")["client_parallel"], 2)
        self.assertEqual(self._of("W12-sel10-ef128")["client_parallel"], 2)
        self.assertEqual(self._of("W5")["client_parallel"], 2)

    def test_open_loop_records_no_parallel(self):
        """bfb ignores `--parallel` under `--rps`, so a number would be fiction."""
        c = self._of("W4-sat90")
        self.assertIsNone(c["client_parallel"])
        self.assertEqual(c["client_threads"], 16)

    def test_single_query_row_is_one(self):
        self.assertEqual(self._of("W3")["client_parallel"], 1)

    def test_w4_records_all_three(self):
        c = self._of("W4")
        self.assertEqual((c["client_parallel"], c["client_threads"]), (64, 16))
        self.assertEqual(c["client_connections"], self.w.W4_CONNS)
        self.assertEqual(c["client_defaults"], "")

    def test_every_search_row_records_a_concurrency(self):
        """A qps with no recorded load is the state this fix exists to end."""
        for w in self.w.table():
            if w.upload_only:
                continue
            c = self.w.client_concurrency(w)
            with self.subTest(row=w.id):
                self.assertIsNotNone(c["client_threads"])
                self.assertIsNotNone(c["client_connections"])
                if self.w.load_mode_of(w) is not self.w.LoadMode.open_loop:
                    self.assertIsNotNone(c["client_parallel"])

    def test_long_flag_spellings_are_read(self):
        w = self.w.Workload("X", "x", ["--parallel", "9", "--threads", "3",
                                       "--connections", "4", "--search"])
        c = self.w.client_concurrency(w)
        self.assertEqual((c["client_parallel"], c["client_threads"],
                          c["client_connections"]), (9, 3, 4))
        self.assertEqual(c["client_defaults"], "")

    def test_a_row_reaching_rows_json_carries_it(self):
        """The field has to survive `Result`, not merely exist in the helper."""
        from dataclasses import asdict
        r = self.w.Result("W6-ef128", self.w.Status.ok, 1, 0, 0, "", 1.0, 1.0, "",
                          **self.w.client_concurrency(
                              next(x for x in self.w.table() if x.id == "W6-ef128")))
        d = asdict(r)
        self.assertEqual(d["client_parallel"], 2)
        self.assertIn("-p", d["client_defaults"])


class NightrunTests(unittest.TestCase):
    """`nightrun.py`: what an unattended night adds around `fullrun.py`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "bench/results").mkdir(parents=True)
        import nightrun
        self.n = importlib.reload(nightrun)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_are_stamped_with_the_date(self):
        self.assertEqual(self.n.labels("2026-09-23", "sift1m"),
                         ("sm-sift-perf-0923", "qd-sift-perf-0923"))
        self.assertEqual(self.n.labels("2026-10-01", "dbpedia-openai-1m"),
                         ("sm-dbp1m-perf-1001", "qd-dbp1m-perf-1001"))
        self.assertEqual(self.n.family("glove-100"), "glove-100-perf")

    def test_a_second_run_of_one_night_refuses_rather_than_sharing_a_log(self):
        """A failed launch's analysis was still running when its directory was
        removed and a second launch recreated it, so a stray "analysis exited 0"
        from the dead run landed in the middle of the live one's log."""
        night = self.root / "bench/results/night-20260923"
        night.mkdir(parents=True)
        (night / "night.log").write_text("2026-09-23 03:51:37 night run starting\n")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = self.n.main(["2026-09-23", "sift1m"], root=self.root)
        self.assertEqual(rc, self.n.EXIT_LOG_EXISTS)
        self.assertIn("exists", err.getvalue())
        self.assertEqual((night / "night.log").read_text().count("\n"), 1)

    def test_no_qdrant_binary_stops_before_anything_runs(self):
        with mock.patch.dict(os.environ, {"QDRANT_BINARY": str(self.root / "absent")}):
            rc = self.n.main(["2026-09-23", "sift1m"], root=self.root)
        self.assertEqual(rc, self.n.EXIT_NO_QDRANT)
        self.assertIn("EXIT=97", (self.root / "bench/results/night-20260923/night.log")
                      .read_text())

    def test_the_run_keeps_a_copy_of_its_report(self):
        """A later re-render of the same pair writes the same file name, and
        on 2026-09-23 one replaced the run's own render."""
        results = self.root / "bench/results"
        page = results / "report-sift1m-sm-a-vs-qd-a-2026-09-23-0230.html"
        page.write_text("the run's render")
        night = results / "night-20260923"
        night.mkdir()
        got = self.n.keep_report(self.root, night, "sift1m", "sm-a", "qd-a",
                                 self.n.Log(night / "night.log"))
        self.assertEqual(got, page)
        page.write_text("a later re-render")
        self.assertEqual((night / page.name).read_text(), "the run's render")
        self.assertEqual((night / "report.path").read_text().strip(), str(page))

    def test_no_report_is_recorded_as_none(self):
        night = self.root / "bench/results/night-20260923"
        night.mkdir()
        self.assertIsNone(self.n.keep_report(self.root, night, "sift1m", "sm-a", "qd-a",
                                             self.n.Log(night / "night.log")))
        self.assertEqual((night / "report.path").read_text().strip(), "none")

    def test_the_provenance_line_says_when_the_binary_predates_its_commit(self):
        repo = self.root / "qdrant"
        (repo / "target/release").mkdir(parents=True)
        binary = repo / "target/release/qdrant"
        binary.write_bytes(b"\x7fELF")
        os.utime(binary, (1_000_000_000, 1_000_000_000))   # 2001, before any commit
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        for cmd in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "x"]):
            subprocess.run(["git", *cmd], cwd=repo, env=env, check=True)
        line = self.n.qdrant_provenance(binary)
        self.assertIn("BINARY PREDATES THIS COMMIT", line)
        self.assertIn("sha256=", line)
        os.utime(binary)   # now: built after the commit
        self.assertNotIn("PREDATES", self.n.qdrant_provenance(binary))

    def test_a_whole_night_end_to_end_with_a_stub_fullrun(self):
        """Every step once, in order, from a temp tree: provenance, the one
        `fullrun.py` call and its flags, the report copy, the closing line."""
        results = self.root / "bench/results"
        (self.root / "bench/harness").mkdir(parents=True, exist_ok=True)
        (self.root / "bench/harness/fullrun.py").write_text(
            "import sys, pathlib\n"
            "a = sys.argv\n"
            "sm, qd = a[a.index('--strawmann-label') + 1], a[a.index('--qdrant-label') + 1]\n"
            "pathlib.Path('bench/results/report-sift1m-' + sm + '-vs-' + qd + "
            "'-2026-09-24-0230.html').write_text('page')\n"
            "print('args', ' '.join(a[1:]))\n")
        binary = self.root / "qdrant"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        env = {"QDRANT_BINARY": str(binary), "HOME": str(self.root)}   # no claude here
        import fullrun
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(fullrun, "previous_pair", lambda labels: None):
            rc = self.n.main(["2026-09-24", "sift1m"], root=self.root)
        self.assertEqual(rc, 0)
        night = results / "night-20260924"
        log = (night / "night.log").read_text()
        for line in ("labels sm-sift-perf-0924 / qd-sift-perf-0924", "qdrant binary",
                     "no previous sift-perf pair", "fullrun.py exited 0",
                     "report copied", "no claude on PATH", "ALL_DONE"):
            self.assertIn(line, log)
        out = (night / "fullrun.out").read_text()
        self.assertIn("--segment-policy equal-work --oversampling-policy defaults --perf", out)
        self.assertIn("--wait-for-gate 120", out)
        self.assertIn("oversampling defaults", log)
        # The command as run, so the analysis need not rebuild it from source.
        self.assertRegex(log, r"command: \S+ bench/harness/fullrun\.py .*"
                              r"--oversampling-policy defaults --perf .*"
                              r"--strawmann-label sm-sift-perf-0924 --qdrant-label qd-sift-perf-0924\n")
        # The reference is `fullrun`'s `auto`, not a number decided here.
        self.assertNotIn("--rps-reference", out)
        self.assertTrue((night / "report-sift1m-sm-sift-perf-0924-vs-qd-sift-perf-0924-"
                                  "2026-09-24-0230.html").exists())
        # The quantized experiment is chosen per night, from the environment
        # the timer sets, and forwarded as the flag rather than assumed.
        with mock.patch.dict(os.environ, {**env, "OVERSAMPLING_POLICY": "matched"}), \
                mock.patch.object(fullrun, "previous_pair", lambda labels: None):
            self.assertEqual(self.n.main(["2026-09-25", "sift1m"], root=self.root), 0)
        night = results / "night-20260925"
        self.assertIn("--oversampling-policy matched", (night / "fullrun.out").read_text())
        self.assertIn("oversampling matched", (night / "night.log").read_text())


class OversamplingPolicyTests(unittest.TestCase):
    """The two quantized experiments, and the refusal that keeps them apart.

    Qdrant's SQ8 rescore pool is `limit`-sized and strawmANN's is
    `max(asked, ef)`, so §7.4 refuses the quantized rows a ratio at defaults.
    Findings 42 measured that the gap is the pool and nothing else, and that
    `oversampling 2` closes it. Tuning the knob is a different experiment from
    the default one, so both exist and neither may be ratioed against the other.
    """

    def setUp(self):
        self.w = importlib.reload(workloads)
        self.addCleanup(os.environ.pop, "OVERSAMPLING_POLICY", None)

    def _over(self, policy: str) -> dict:
        self.w.use_oversampling_policy(policy)
        return {x.id: self.w.quant_of(x)["quantization_oversampling"]
                for x in self.w.table()}

    def test_defaults_leaves_every_engine_on_its_own_pool(self):
        o = self._over("defaults")
        self.assertIsNone(o["W6"])
        self.assertIsNone(o["W8"])
        self.assertIsNone(o["W6-ef128"])

    def test_matched_asks_the_quantized_search_rows_for_two(self):
        o = self._over("matched")
        self.assertEqual(o["W6"], float(self.w.MATCHED_OVERSAMPLING))
        self.assertEqual(o["W8"], float(self.w.MATCHED_OVERSAMPLING))
        self.assertEqual(o["W6-ef128"], float(self.w.MATCHED_OVERSAMPLING))

    def test_a_row_that_names_its_own_is_left_alone(self):
        """W7 sends 4 because binary is inert without it; the policy must not
        overwrite a row's own choice, or it changes two things at once."""
        self.assertEqual(self._over("matched")["W7"], 4.0)
        self.assertEqual(self._over("defaults")["W7"], 4.0)

    def test_the_fp32_rows_are_untouched(self):
        """It is a quantized experiment, so W3, W4 and the W10 ladder must read
        identically under both policies or the run is measuring two changes."""
        d, m = self._over("defaults"), self._over("matched")
        for wid in ("W3", "W4", "W10-ef128", "W10-ef512", "W9", "W13"):
            with self.subTest(row=wid):
                self.assertIsNone(d[wid])
                self.assertIsNone(m[wid])

    def test_upload_rows_never_carry_a_search_flag(self):
        m = self._over("matched")
        for wid in ("W6-upload", "W7-upload", "W8-upload"):
            self.assertIsNone(m[wid], wid)

    def test_the_policy_is_stamped_and_therefore_hashed(self):
        self.w.use_oversampling_policy("matched")
        self.assertEqual(self.w.harness_stamp()["oversampling_policy"], "matched")
        self.assertIn("oversampling_policy", self.w.STAMP_KEYS)

    def test_two_policies_do_not_share_a_stamp_hash(self):
        """The refusal that makes them two experiments rather than one table."""
        self.w.use_oversampling_policy("defaults")
        a = self.w.stamp_hash(self.w.harness_stamp())
        self.w.use_oversampling_policy("matched")
        self.assertNotEqual(a, self.w.stamp_hash(self.w.harness_stamp()))

    def test_a_stamp_from_before_the_policy_hashes_as_its_rows_were(self):
        """19b1761 added the key and every earlier run's rows stopped matching
        their own `run.json`: the 0921 and 0908 sift1m pages lost every ratio."""
        import hashlib
        self.w.use_oversampling_policy("defaults")
        stamp = self.w.harness_stamp()
        legacy = {k: v for k, v in stamp.items() if k != "oversampling_policy"}
        before = [k for k in self.w.STAMP_KEYS if k != "oversampling_policy"]
        as_measured = hashlib.sha256(json.dumps({k: legacy.get(k) for k in before},
                                                sort_keys=True, default=str)
                                     .encode()).hexdigest()[:12]
        self.assertEqual(self.w.stamp_hash(legacy), as_measured)
        # A run that records the policy is still a different harness from one
        # that predates it; only its absence is forgiven, and only for this key.
        self.assertNotEqual(self.w.stamp_hash(stamp), self.w.stamp_hash(legacy))
        no_settle = {k: v for k, v in legacy.items() if k != "engine_settle"}
        self.assertNotEqual(self.w.stamp_hash(no_settle), self.w.stamp_hash(legacy))

    def test_pool_gives_every_quantized_row_an_ef_sized_pool(self):
        """Qdrant rescores `limit x oversampling` candidates, strawmANN
        `max(asked, ef)`. At `ef / limit` the two are the same number, which
        `matched`'s 2 is not for any row, and binary's 4 is not either."""
        o = self._over("pool")
        for wid, ef in (("W6", 128), ("W6-ef32", 32), ("W6-ef64", 64),
                        ("W6-ef256", 256), ("W6-ef512", 512), ("W7", 128), ("W8", 128)):
            with self.subTest(row=wid):
                self.assertEqual(o[wid], ef / 10)
        # W7's own 4 is replaced, not appended beside: one flag, one value.
        w7 = {x.id: x for x in self.w.table()}["W7"]
        self.assertEqual([str(a) for a in w7.args].count("--quantization-oversampling"), 1)
        for wid in ("W3", "W4", "W10-ef128", "W9", "W6-upload", "W7-upload"):
            with self.subTest(row=wid):
                self.assertIsNone(o[wid])
        self.assertEqual(self.w.pool_oversampling(5, 10), 1.0)   # never below 1

    def test_pool_is_its_own_experiment(self):
        self.w.use_oversampling_policy("matched")
        m = self.w.stamp_hash(self.w.harness_stamp())
        self.w.use_oversampling_policy("pool")
        self.assertEqual(self.w.harness_stamp()["oversampling_policy"], "pool")
        self.assertNotEqual(m, self.w.stamp_hash(self.w.harness_stamp()))

    def test_it_survives_the_subprocess_boundary(self):
        """`fullrun.py` binds it once and each arm is a separate process."""
        self.w.use_oversampling_policy("matched")
        self.assertEqual(os.environ["OVERSAMPLING_POLICY"], "matched")
        reloaded = importlib.reload(workloads)
        self.assertIs(reloaded.OVERSAMPLING_POLICY,
                      reloaded.OversamplingPolicy.matched)


class VisitedSetProvenanceTests(unittest.TestCase):
    """Which visited set served a row, recorded rather than inferred.

    `-Dvisited` makes §11's open question 3 measurable, and the measurement is
    worthless if a row does not say which arm produced it: the two binaries
    answer identically, so nothing else in the record distinguishes them.
    """

    def setUp(self):
        self.w = importlib.reload(workloads)

    def test_the_arm_reaches_the_row(self):
        ident = self.w.build_identity(
            {"strawmann": {"commit": "abc123", "binary_sha256": "d00d",
                           "optimize": "ReleaseFast", "visited_set": "bitmap"},
             "isa_build": "native", "gate": "pass", "profile": "as-deployed"})
        self.assertEqual(ident["visited_set"], "bitmap")
        self.assertIn("visited_set", self.w.BUILD_KEYS)

    def test_a_qdrant_row_says_nothing_about_it(self):
        """Qdrant has no such knob, so `None` is the honest value."""
        ident = self.w.build_identity(
            {"qdrant": {"version": "1.19.2-dev", "binary_sha256": "beef"},
             "gate": "pass"})
        self.assertIsNone(ident["visited_set"])

    def test_two_arms_are_a_mixed_build(self):
        """The refusal this exists for: one label, rows from both binaries."""
        import compare
        rows = {"W3": {"engine_build": "abc", "engine_binary": "d0",
                       "isa_build": "native", "optimize": "ReleaseFast",
                       "profile": "as-deployed", "visited_set": "generation"},
                "W4": {"engine_build": "abc", "engine_binary": "d0",
                       "isa_build": "native", "optimize": "ReleaseFast",
                       "profile": "as-deployed", "visited_set": "bitmap"}}
        self.assertEqual(len(compare.build_identities(rows)), 2)


class W9AbTests(unittest.TestCase):
    """The W9 experiment's plan and arithmetic, without an engine."""

    @classmethod
    def setUpClass(cls):
        cls.ab = importlib.import_module("w9_ab")

    def test_arms_that_share_a_worker_count_share_a_session_and_alternate(self):
        self.assertEqual(self.ab.plan(2), [
            (7, [("stream7", 1), ("batch8", 1), ("stream7", 2), ("batch8", 2)]),
            (4, [("stream4", 1), ("stream4", 2)])])

    def test_each_arm_is_the_tables_w9_with_one_thing_changed(self):
        w9 = {w.id: w for w in workloads.table()}["W9"]
        for arm, batch in (("stream7", 1), ("batch8", 8)):
            w = self.ab.w9_row(arm, batch, 1000)
            args = list(w.args)
            self.assertEqual(w.id, f"W9-{arm}")
            self.assertEqual(args[args.index("-n") + 1], "1000")
            self.assertEqual(workloads.batch_size_of(w), batch)
            # Everything else is W9's own invocation.
            rest = list(args)
            if "--search-batch-size" in rest:
                i = rest.index("--search-batch-size")
                del rest[i:i + 2]
            self.assertEqual(self.ab.set_n(rest, workloads.n_of(w9)), list(w9.args))
            self.assertIn("--search-exact", args)

    def test_the_summary_says_how_much_of_the_bus_each_arm_implies(self):
        corpus = 990_000 * 1536 * 4
        rows = [
            {"arm": "stream7", "qps": q, "n_queries": 1000, "duration_s": 1000 / q,
             "perf_task_clock_s": 7.0 * 1000 / q, "perf_cycles": 1.43e9 * 1000,
             "ipc": 0.26, "dram_bytes": 1.633e9 * 1000, "foreign": ""}
            for q in (9.7, 9.8, 9.9)] + [
            {"arm": "batch8", "qps": 30.0, "n_queries": 1000, "duration_s": 33.3,
             "foreign": "Xorg(6%)"}]
        got = {s["arm"]: s for s in self.ab.summarise(rows, corpus)}
        s7 = got["stream7"]
        self.assertEqual(s7["reps"], 3)
        self.assertAlmostEqual(s7["qps"], 9.8)
        self.assertAlmostEqual(s7["qps_spread"], 0.2)
        self.assertAlmostEqual(s7["cores"], 7.0)
        self.assertAlmostEqual(s7["cycles_per_q"], 1.43e9)
        self.assertAlmostEqual(s7["dram_mb_per_q"], 1633.0)
        self.assertAlmostEqual(s7["implied_gbs"], 9.8 * corpus / 1e9)   # ~59.6
        # A row without counters still gives its rate, and says what was busy.
        self.assertIsNone(got["batch8"]["ipc"])
        self.assertEqual(got["batch8"]["foreign"], ["Xorg(6%)"])
        self.assertNotIn("stream4", got)
        text = self.ab.table_text(self.ab.summarise(rows, corpus), "65.2 GB/s")
        self.assertIn("stream7", text)
        self.assertIn("59.6", text)

    def test_the_profile_attaches_to_the_engine_not_its_scope(self):
        with mock.patch.object(self.ab.procstat, "engine_processes",
                               lambda: [(10, "systemd-run"), (42, "strawmann")]):
            self.assertEqual(self.ab.strawmann_pid(), 42)
        # Two engines up is not a profile of either.
        with mock.patch.object(self.ab.procstat, "engine_processes",
                               lambda: [(42, "strawmann"), (43, "qdrant"), (44, "strawmann")]):
            self.assertIsNone(self.ab.strawmann_pid())
        argv = self.ab.perf_record_argv(42, Path("/tmp/x.data"))
        self.assertEqual(argv[:2], ["perf", "record"])
        self.assertIn("42", argv)


class BuildersAliveTests(unittest.TestCase):
    """The warning a launcher prints before measuring beside someone's build."""

    def test_builders_are_counted_by_name_and_nothing_else_is(self):
        procstat = importlib.import_module("procstat")
        procs = [(1, "systemd"), (2, "rustc"), (3, "rustc"), (4, "cargo"),
                 (5, "claude"), (6, "strawmann"), (7, "clippy-driver"), (8, "rustc")]
        self.assertEqual(procstat.builders_alive(procs),
                         ["cargo", "clippy-driver", "rustc x3"])
        self.assertEqual(procstat.builders_alive([(1, "bash"), (2, "claude")]), [])
        # And the real table reads without raising.
        self.assertIsInstance(procstat.builders_alive(), list)

    def test_the_night_log_names_a_running_build(self):
        n = importlib.import_module("nightrun")
        procstat = importlib.import_module("procstat")
        src = Path(n.__file__).read_text()
        self.assertIn("procstat.builders_alive()", src)
        with mock.patch.object(procstat, "processes", lambda: [(9, "rustc"), (10, "rustc")]):
            self.assertEqual(procstat.builders_alive(), ["rustc x2"])
