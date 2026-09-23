#!/usr/bin/env python3
"""Unit tests: the rendered report: its tables, charts and file name."""

from __future__ import annotations

import calendar
import contextlib
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from harness_fixtures import CONF_T2, CONF_T3, Fixture, _reload, _row, _sweep, good_stamp

from workloads import stamp_hash


class ReportHostTests(unittest.TestCase):
    """report.py: one machine is one Host card; the page opens with its bandwidth."""

    @classmethod
    def setUpClass(cls):
        try:
            # Imported to find out whether they are installed, not to use
            # them: this suite skips rather than fails outside the uv project.
            import jinja2  # noqa: F401
            import pandas  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as e:  # the uv project only
            raise unittest.SkipTest(
                "report.py needs the uv project (jinja2/pandas/plotly)") from e
        cls.report = importlib.import_module("report")

    ENV = ("§7.1 host discipline:\n  ok   CPU X (24 logical)\n  FAIL governor=powersave\n"
           "       a scaling governor makes throughput depend on how warm the run is\n"
           "  ok   thp=madvise\n  ok   machine is quiescent (load {load} over 24 cores)\n"
           "{busy}\nenvironment hash: {h}\n(every result row carries this)\n\n"
           "1 check(s) failed. §7.1: this script refuses to run if any check fails.\n"
           "for development. A run made with --lax MUST NOT produce published numbers.\n")

    def _run(self, label, load, busy, h, bw=None):
        env = self.ENV.format(load=load, busy=busy, h=h)
        meta = {"host": {"hostname": "box", "cpu": "CPU X",
                         "memory_bandwidth": bw}}
        return self.report.Run(label, [], env, False, h, meta=meta)

    def test_static_checks_exclude_the_moment_and_the_advice(self):
        r = self._run("a", 1.5, "       busy: rustc(100%)", "aaaa")
        texts = [c.text for c in r.static_checks]
        self.assertIn("CPU X (24 logical)", texts)
        self.assertIn("governor=powersave", texts)
        self.assertNotIn("machine is quiescent (load 1.5 over 24 cores)", texts)
        self.assertFalse(any("MUST NOT" in t or "--lax" in t for t in texts))
        self.assertEqual([c.text for c in r.moment_checks],
                         ["machine is quiescent (load 1.5 over 24 cores)", "busy: rustc(100%)"])

    def test_same_host_despite_differing_hashes_and_load(self):
        a = self._run("a", 1.5, "", "aaaa")
        b = self._run("b", 1.8, "       busy: rustc(100%)", "bbbb")
        self.assertTrue(self.report.same_host([a, b]))
        # A different static check is a different machine (or state).
        c = self._run("c", 1.8, "", "cccc")
        c.env = c.env.replace("thp=madvise", "thp=never")
        self.assertFalse(self.report.same_host([a, c]))

    def test_bandwidth_said_once_and_prefers_startup(self):
        probe = {"aggregate_gbps": 70.0, "threads": 24, "single_core_gbps": 39.0,
                 "single_core_pct_of_bus": 55, "source": "strawmann --probe at run start",
                 "measured_at": "t1"}
        boot = dict(probe, aggregate_gbps=72.0, source="strawmann startup banner",
                    measured_at="t0")
        a = self._run("a", 1.5, "", "aaaa", bw=probe)
        b = self._run("b", 1.8, "", "bbbb", bw=boot)
        out = self.report.bandwidth_of([a, b])
        self.assertEqual(out["aggregate_gbps"], 72.0)
        self.assertIn("startup banner", out["source"])
        self.assertIn("agrees within 10%", out["source"])
        # Disagreement past 10% shows both rather than picking one.
        b.meta["host"]["memory_bandwidth"] = dict(boot, aggregate_gbps=40.0)
        out = self.report.bandwidth_of([a, b])
        self.assertIn("a:", out["text"])
        self.assertIn("b:", out["text"])
        # Nothing recorded is said as such.
        self.assertIsNone(self.report.bandwidth_of([self._run("z", 1, "", "z")])["aggregate_gbps"])


    NOT_QUIESCENT = ("§7.1 host discipline:\n  ok   CPU X (24 logical)\n  FAIL governor=powersave\n"
                     "       a scaling governor makes throughput depend on how warm the run is\n"
                     "  ok   thp=madvise\n  FAIL machine is NOT quiescent: load 30.0 over 24 cores "
                     "(125% per core)\n       numbers taken now are not comparable to numbers taken idle,\n"
                     "       and the difference is invisible in the results.\n"
                     "       busy: llama-server(1008%)\nenvironment hash: dddd\n")

    def test_not_quiescent_is_the_moment_not_the_machine(self):
        meta = {"host": {"hostname": "box", "cpu": "CPU X", "memory_bandwidth": None}}
        busy = self.report.Run("d", [], self.NOT_QUIESCENT, False, "dddd", meta=meta)
        texts = [c.text for c in busy.static_checks]
        self.assertNotIn("machine is NOT quiescent: load 30.0 over 24 cores (125% per core)", texts)
        self.assertFalse(any("busy:" in t or "not comparable" in t for t in texts))
        self.assertEqual(next(c.kind for c in busy.moment_checks), "fail")
        self.assertIn("busy: llama-server(1008%)", [c.text for c in busy.moment_checks])
        # A quiescent run and a NOT-quiescent run of the same box are one host.
        idle = self._run("a", 1.5, "", "aaaa")
        self.assertTrue(self.report.same_host([idle, busy]))

    def _pair(self, tmp: Path, a_rows, b_rows, stamp_a=None, stamp_b=None,
              conf_a=CONF_T3, conf_b=CONF_T3, sw_a=None, sw_b=None,
              colls=None):
        tmp = Path(tempfile.mkdtemp(dir=tmp))   # a fresh root per pair
        fx = Fixture(tmp)
        mods = _reload(tmp)
        report = importlib.reload(self.report)
        sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
        fx.label("a", a_rows, stamp_a or good_stamp(), conf_a,
                 sw if sw_a is None else sw_a, collections=colls)
        fx.label("b", b_rows, stamp_b or good_stamp(), conf_b,
                 sw if sw_b is None else sw_b, collections=colls)
        mods["compare"]._RECALL_CACHE.clear()
        runs = [report.load_run("a"), report.load_run("b")]
        return report, runs, report.frame(runs)

    @staticmethod
    def _section(html: str, wid: str) -> str:
        return html.split(f'id="wl-{wid}"', 1)[1].split("</section>", 1)[0]

    def test_sections_and_headline_refuse_what_compare_refuses(self):
        import re
        w11 = dict(recall_joinable=False, ratio_policy="search-during-write; no recall join")
        with tempfile.TemporaryDirectory() as tmp:
            # W3 clean; W4 contaminated; W5 re-run under another stamp; W11 policy.
            report, runs, df = self._pair(
                Path(tmp),
                [_row("W3", 4000), _row("W4", 8000, foreign="rustc(100%)"), _row("W5", 800),
                 _row("W11", 100, **w11)],
                [_row("W3", 2000), _row("W4", 4000), _row("W5", 400, stamp=good_stamp(queries=1)),
                 _row("W11", 50, **w11)])
            html = report.workload_sections(runs, df)
            self.assertIn("2.00x", self._section(html, "W3"))
            for wid in ("W4", "W5", "W11"):
                sec = self._section(html, wid)
                self.assertIsNone(re.search(r"\d\.\d\dx", sec), f"{wid}: {sec[:400]}")
            # STALE pair (differing stamps): no `x` anywhere in the sections,
            # and the headline states the refusal rather than a range.
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)],
                                          stamp_b=good_stamp(metric="Cosine"))
            html = report.workload_sections(runs, df)
            self.assertIsNone(re.search(r"\d\.\d\dx", html), html[:600])
            summ = report.summary(runs, df)
            self.assertIsNone(summ["matched"])
            # The losses list is built from `compare.Row` like every other
            # ratio, so a STALE pair yields no "where it is slower" line either.
            self.assertEqual(summ["losses"], [])
            self.assertTrue(summ["matched_refusal"].startswith("STALE"), summ)
            self.assertIn("No matched-recall ratio", report.matched_recall_table(runs))
            # Unlicensed (T2): the same, naming the licence.
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)],
                                          conf_a=CONF_T2, conf_b=CONF_T2)
            self.assertIn("NOT A LICENSED", report.summary(runs, df)["matched_refusal"])
            # A contaminated W10 row refuses the matched-recall headline too.
            pts = [{"ef": e, "exact": False, "recall_at_10": r, "smoke_qps": 1.0}
                   for e, r in ((32, 0.90), (64, 0.95))]
            sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98, points=pts))]
            w10 = [_row("W10-ef32", 9000, ef=32), _row("W10-ef64", 6000, ef=64)]
            report, runs, df = self._pair(
                Path(tmp), w10 + [_row("W3", 4000)],
                [_row("W10-ef32", 4500, ef=32, foreign="rustc(100%)"), _row("W10-ef64", 3000, ef=64),
                 _row("W3", 2000)], sw_a=sw, sw_b=sw)
            summ = report.summary(runs, df)
            self.assertIsNone(summ["matched"])
            self.assertIn("contaminated", summ["matched_refusal"])
            self.assertIn("W10-ef32", summ["matched_refusal"])
            # And the clean version of the same fixture states the range.
            report, runs, df = self._pair(
                Path(tmp), w10 + [_row("W3", 4000)],
                [_row("W10-ef32", 4500, ef=32), _row("W10-ef64", 3000, ef=64), _row("W3", 2000)],
                sw_a=sw, sw_b=sw)
            summ = report.summary(runs, df)
            self.assertEqual(summ["matched_refusal"], "")
            self.assertIsNotNone(summ["matched"])
            self.assertAlmostEqual(summ["matched"]["lo"], 2.0)
            self.assertIn("2.00x", report.matched_recall_table(runs))
            self.assertEqual(summ["losses"], [])   # `a` is the faster engine here
            # Both sweeps carry the same smoke rate, so the single-client
            # ordering is "same" (ratio 1.0), and the headline must not claim
            # the opposite ordering the template used to state verbatim.
            self.assertEqual(summ["matched"]["smoke"]["verdict"], "same")
            pts_b = [{"ef": e, "exact": False, "recall_at_10": r, "smoke_qps": 2.0}
                     for e, r in ((32, 0.90), (64, 0.95))]
            sw_b = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98, points=pts_b))]
            report, runs, df = self._pair(
                Path(tmp), w10 + [_row("W3", 4000)],
                [_row("W10-ef32", 4500, ef=32), _row("W10-ef64", 3000, ef=64), _row("W3", 2000)],
                sw_a=sw, sw_b=sw_b)
            sm = report.smoke_ordering(runs[0], runs[1])
            self.assertEqual((sm["verdict"], sm["n"]), ("opposite", 2))
            self.assertAlmostEqual(sm["lo"], 0.5)
            pts_b[0]["smoke_qps"] = 0.5
            sw_b = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98, points=pts_b))]
            report, runs, df = self._pair(
                Path(tmp), w10 + [_row("W3", 4000)],
                [_row("W10-ef32", 4500, ef=32), _row("W10-ef64", 3000, ef=64), _row("W3", 2000)],
                sw_a=sw, sw_b=sw_b)
            self.assertEqual(report.smoke_ordering(runs[0], runs[1])["verdict"], "mixed")

    def test_no_ratio_across_the_smoke_rate_and_w10(self):
        with tempfile.TemporaryDirectory() as tmp:
            pts = [{"ef": e, "exact": False, "recall_at_10": r, "smoke_qps": 100.0}
                   for e, r in ((32, 0.90), (64, 0.95))]
            sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98, points=pts))]
            # b has the sweep (so a smoke rate per ef) but no W10 rows.
            report, runs, df = self._pair(
                Path(tmp), [_row("W10-ef32", 9000, ef=32), _row("W10-ef64", 6000, ef=64)],
                [_row("W3", 2000)], sw_a=sw, sw_b=sw)
            # The chart may still draw b's smoke curve, labelled as such...
            self.assertEqual({p["source"] for p in report.frontier_points(runs[1])},
                             {"conformance smoke rate"})
            self.assertEqual(report.frontier_points(runs[1], bfb_only=True), [])
            # ...but no ratio is formed across the two instruments.
            summ = report.summary(runs, df)
            self.assertIsNone(summ["matched"])
            self.assertIn("no W10 rows", summ["matched_refusal"])
            self.assertIn("No matched-recall ratio", report.matched_recall_table(runs))
            self.assertNotIn("x</td>", report.matched_recall_table(runs))

    def test_the_filtered_grade_gets_a_matched_recall_table_of_its_own(self):
        """`W12-sel10` is refused a per-row ratio because the engines land just
        outside the recall band at the one `ef` it measures, so the filtered
        comparison exists only at matched recall — and `frontier_points` needs
        the ladder rows and the *graded* sweep to build it (findings 48).

        The ladder was measured for a run and the page still showed a dash,
        because nothing called `frontier_points` with a `W12` prefix: both call
        sites used the W10/bench2 defaults. This is that call site.
        """
        with tempfile.TemporaryDirectory() as tmp:
            efs = ((128, 0.990), (256, 0.999))
            # The filtered table is composed onto the fp32 one, as the SQ8
            # table is, so the fixture is a whole run: bench2's sweep and W10
            # rows beside the filtered ones. A run without them is not one this
            # table has an opinion about.
            fp32 = [{"ef": e, "exact": False, "recall_at_10": r, "smoke_qps": 1.0}
                    for e, r in efs]
            filt = [{"ef": e, "exact": False, "recall_at_10": r, "smoke_qps": 1.0,
                     "n_matching": 19936} for e, r in efs]
            sw = [("recall.sift1m.bench2.json",
                   _sweep("bench2", "sift1m", 0.99, points=fp32)),
                  # Written under the grade, which is where `load_recall_json`
                  # looks for it.
                  ("recall.sift1m.bench12.sel10.json",
                   _sweep("bench12", "sift1m", 0.99, points=filt, grade="sel10"))]
            rows_a = [_row(f"W10-ef{e}", q, ef=e) for e, q in ((128, 9000), (256, 6000))] + \
                     [_row(f"W12-sel10-ef{e}", q, ef=e, collection="bench12")
                      for e, q in ((128, 400), (256, 380))]
            rows_b = [_row(f"W10-ef{e}", q, ef=e) for e, q in ((128, 4500), (256, 3000))] + \
                     [_row(f"W12-sel10-ef{e}", q, ef=e, collection="bench12")
                      for e, q in ((128, 2000), (256, 1400))]
            # A filtered row is refused a ratio unless the engine's payload
            # index was *read back* — flags say what was asked, not what was
            # built (docs/workloads.md W12 point 1). The new table inherits
            # that guard through `matched_recall_refusal`, which is why the
            # fixture has to carry `collections.json`: without it the refusal
            # is `W12-sel10-ef128: payload index state not read back`, and
            # correctly so.
            colls = [{"collection": "bench12", "payload_indexes": ["a"]}]
            report, runs, df = self._pair(Path(tmp), rows_a, rows_b,
                                          sw_a=sw, sw_b=sw, colls=colls)
            html = report.matched_recall_table(runs)
            self.assertIn("At matched recall, filtered to 10%", html)
            # And it says what the condition selected, so the reader is not
            # asked to take the grade's name for the size of the matching set.
            self.assertIn("19,936", html)

    def test_recall_ci_widens_the_matched_ratio(self):
        """The headline range treats each measured recall as exact; the recall
        CI the Recall section prints is carried through the interpolation.

        The frontier steepens toward recall 1.0, so the same CI half-width buys
        a much wider band at the top than at the bottom. That is the whole
        reason the column exists, and a band that did not widen with the slope
        would be decoration.
        """
        report = self.report
        # Two points an order of magnitude apart in throughput: a shallow
        # segment (0.90 -> 0.95) and a steep one (0.99 -> 0.999).
        other = [{"ef": 32, "recall": 0.90, "qps": 1000.0},
                 {"ef": 64, "recall": 0.95, "qps": 500.0},
                 {"ef": 128, "recall": 0.99, "qps": 250.0},
                 {"ef": 256, "recall": 0.999, "qps": 25.0}]
        shallow = [{"ef": 32, "recall": 0.92, "qps": 1600.0, "ci": (0.915, 0.925)}]
        steep = [{"ef": 256, "recall": 0.995, "qps": 1600.0, "ci": (0.990, 1.0)}]
        a = report.matched_ratios(shallow, other)[0]
        b = report.matched_ratios(steep, other)[0]
        for r in (a, b):
            lo, hi = r["band"]
            self.assertLess(lo, r["ratio"])
            self.assertGreater(hi, r["ratio"])
        wide = lambda r: r["band"][1] / r["band"][0]
        self.assertGreater(wide(b), 3 * wide(a))
        # No CI on the anchor is no band, not a band of zero width.
        self.assertIsNone(report.matched_ratios(
            [{"ef": 32, "recall": 0.92, "qps": 1600.0, "ci": (None, None)}], other)[0]["band"])
        # Outside the other engine's measured range there is no row at all,
        # because nothing is extrapolated.
        self.assertEqual(report.matched_ratios(
            [{"ef": 1, "recall": 0.10, "qps": 9.0, "ci": (0.09, 0.11)}], other), [])

    def test_summary_states_the_losses(self):
        """A summary that reports only the wins is an advertisement."""
        with tempfile.TemporaryDirectory() as tmp:
            # `a` loses W3 by 2x and wins W13; both clear the 2% minimum.
            report, runs, df = self._pair(
                Path(tmp), [_row("W3", 2000), _row("W13", 4000)],
                [_row("W3", 4000), _row("W13", 2000)])
            # `_pair` reloads the module onto a fresh root, so ROOT must be read
            # *after* it. Read before, this wrote over the real noise floor.
            noise = report.ROOT / "bench/results/noise.json"
            noise.parent.mkdir(parents=True, exist_ok=True)
            # `arms` names the engines the floor was measured on. Without it
            # the floor bands no ratio at all (findings 38), which is a
            # different refusal than the one this test is about.
            noise.write_text(json.dumps({"rsd": {"W3": 0.001, "W13": 0.001},
                                         "reps": {"W3": 6, "W13": 6},
                                         "source": "t", "discarded": [],
                                         "arms": ["a", "b"]}))
            got = {l["id"]: l["ratio"] for l in report.losses(runs)}
            self.assertEqual(list(got), ["W3"])
            self.assertTrue(got["W3"].startswith("0.5"), got)
            # A row inside the floor is not a loss, it is no measured
            # difference; and a row with no floor at all has not been judged,
            # so it is not one either.
            report, runs, df = self._pair(
                Path(tmp), [_row("W3", 3980)], [_row("W3", 4000)])
            self.assertEqual(report.losses(runs), [])          # no noise.json here
            noise = report.ROOT / "bench/results/noise.json"   # a fresh root per pair
            noise.parent.mkdir(parents=True, exist_ok=True)
            noise.write_text(json.dumps({"rsd": {"W3": 0.001}, "reps": {"W3": 6},
                                         "source": "t", "discarded": []}))
            self.assertEqual(report.losses(runs), [])          # 0.995x, inside the 2%

    def test_verdict_names_the_minimum_effect_apart_from_measured_spread(self):
        """`noise_band` returns `max(3*sqrt(2)*rsd, 2%)`, and calling the second
        of those "the measured noise floor" says 2% was measured here."""
        v = self.report.ratio_verdict
        self.assertIn("minimum reportable effect", v("W3", 1.5, {"W3": 0.001}))
        self.assertIn("measured noise floor", v("W3", 1.5, {"W3": 0.05}))
        self.assertIn("no noise floor measured", v("W3", 1.5, {}))

    def test_charts_refuse_what_compare_refuses(self):
        """The tables honoured `compare.py`'s refusals; the bar chart did not,
        so a skimmer got exactly the comparison every table declines to make."""
        w11 = dict(recall_joinable=False, ratio_policy="search-during-write; no recall join")
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(
                Path(tmp), [_row("W3", 4000), _row("W11", 100, **w11)],
                [_row("W3", 2000), _row("W11", 50, **w11)])
            self.assertEqual(set(report.refused_ids(runs)), {"W11"})
            c = report.chart_throughput(runs, df)
            self.assertIn("W11", c["note"])
            self.assertIn("search-during-write", c["note"])
            self.assertIn("not compared", c["html"])
            # W3 is comparable and must not be marked.
            self.assertNotIn("W3  search, fp32, single query  \u2014 not compared", c["html"])

    def test_every_series_trace_carries_the_theme_tag(self):
        """`THEME_JS` re-colours by this tag, so a trace whose colour means the
        engine must carry one — and one whose colour means something else must
        not.

        The ambient-load chart used to be the exception: its two colours meant
        clean and contaminated, so tagging it made the theme switch repaint
        Qdrant's clean bars crimson under a caption saying red means another
        process was on the box. It is not an exception any more, because its
        colour is now the engine and contamination is a hatch — which is also
        what let both engines' clean rows stop being the same green.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            self.assertIn('"series":"a"', report.chart_throughput(runs, df)["html"])
            load = report.chart_load(runs, df)["html"]
            self.assertIn("a clean", load)
            # Tagged, and tagged per engine rather than with one shared status
            # colour: the two runs must not come out the same hue.
            self.assertIn('"series":"a"', load)
            self.assertIn('"series":"b"', load)
            hues = set(re.findall(r'"color":"(#[0-9A-Fa-f]{6})"', load))
            self.assertGreaterEqual(len(hues), 2, "both engines drawn in one colour")

    def test_the_distribution_note_quotes_the_unclipped_tail(self):
        """The chart clips at p99.5; the note must not describe the clip.

        Quoting the drawn series' max prints the clip boundary as if it were
        the slowest request — 16.1 ms for a row whose slowest took 36.2 — and
        the clipped region is exactly where these two engines differ, so the
        chart alone reads as a win for whichever is left-shifted in the body.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # A tight body with one far outlier, on one engine only.
            a = [0.010] * 200 + [0.400]
            b = [0.011] * 200 + [0.012]
            report, runs, _ = self._pair(
                Path(tmp), [_row("W4", 4000)], [_row("W4", 2000)])
            # Per-request timings live in `<id>.json`, which the fixture does
            # not write; `Run.detail` is where they land once read.
            for run, vals in zip(runs, (a, b)):
                run.detail["W4"] = {"results": {"search": {"full_timings": vals}}}
            c = report.chart_latency_distribution(runs)
            self.assertIsNotNone(c)
            self.assertIn("400.0 ms", c["note"], "the note must quote the real max")
            self.assertNotIn("10.0 ms; ", c["note"], "not the clip boundary")

    def test_a_quantized_sweep_is_joined_on_the_params_it_sent(self):
        """The encoding chart must not borrow a sweep of a different search.

        `load_recall_json` refuses a sweep whose quantization parameters differ
        from the ones asked for, which is why the quantized curves were absent
        until the chart passed each collection's own. A chart that defaulted
        them would silently draw whichever sweep happened to be on disk.
        """
        report = importlib.reload(self.report)
        r = importlib.reload(importlib.import_module("recall"))
        # The pairing the chart relies on: each collection has its own params,
        # and they are not all the default.
        params = {c: r.quant_params_of(c) for c, _n, _h in report.ENCODINGS}
        self.assertEqual(params["bench2"], (None, None), "fp32 sends none")
        self.assertNotEqual(params["bench7"], (None, None),
                            "binary sends oversampling and rescore")
        self.assertEqual(len({c for c, _n, _h in report.ENCODINGS}), 4,
                         "four encodings, one per collection")

    def test_the_wait_chart_reads_positions_not_bar_lengths(self):
        """Five orders of magnitude, so a log axis — and therefore not bars.

        A bar encodes magnitude as length from zero; on a log axis a bar three
        quarters as long can be a thousandth of the value. Markers encode
        position, which a log axis leaves honest. This asserts the form, not
        the styling: the chart may be restyled, but it may not become bars.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(
                Path(tmp),
                [_row("W4", 4000, runqueue_wait_s=0.0078)],
                [_row("W4", 2000, runqueue_wait_s=60.6433)])
            c = report.chart_runqueue(runs, df)
            self.assertIsNotNone(c)
            self.assertIn('"type":"log"', c["html"])
            self.assertIn('"mode":"markers"', c["html"])
            self.assertNotIn('"type":"bar"', c["html"])
            # Both engines present and distinguishable.
            hues = set(re.findall(r'"color":"(#[0-9A-Fa-f]{6})"', c["html"]))
            self.assertGreaterEqual(len(hues), 2)

    def test_a_row_with_no_measured_wait_is_named_not_dropped(self):
        """A log axis cannot place a zero, and an absent row reads as "no wait"."""
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(
                Path(tmp),
                [_row("W4", 4000, runqueue_wait_s=0.0078), _row("W3", 1)],
                [_row("W4", 2000, runqueue_wait_s=60.6433), _row("W3", 1)])
            c = report.chart_runqueue(runs, df)
            self.assertIn("W3", c["note"], "a dropped row must be named in the note")

    def test_ingest_total_is_the_sum_the_section_asks_for(self):
        """"Compare the sum, not the upload line" was advice the table left as
        arithmetic for the reader."""
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(
                Path(tmp),
                [_row("W1", None, upload_s=2.0), _row("W2", None, index_wait_s=8.0)],
                [_row("W1", None, upload_s=5.0), _row("W2", None, index_wait_s=20.0)])
            html = report.ingest_table(runs, df)
            self.assertIn("W1 + W2", html)
            self.assertIn("10.0 s", html)
            self.assertIn("25.0 s", html)
            self.assertIn("2.50x", html)
            # And it says the ratio has nothing to be judged against.
            self.assertIn("no noise floor", html)

    def test_an_epsilon_from_the_floor_is_not_called_calibrated(self):
        """`conformance.json` said `epsilon_source: floor` and the banner said
        "calibrated" over it, on every sift1m report to date."""
        with tempfile.TemporaryDirectory() as tmp:
            conf = {**CONF_T3, "max_delta": 0.0, "p99_delta": 0.0,
                    "epsilon": 4.172e-7, "epsilon_relative": True,
                    "epsilon_source": "floor"}
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], conf_a=conf, conf_b=conf)
            self.assertEqual(report.licence_of(runs)["delta"]["source"], "floor")
            html = report.build(runs, "t")
            self.assertRegex(html, r"floor</b> relative \u03b5 of\s+4\.172e-07")
            self.assertNotRegex(html, r"calibrated relative \u03b5")
            self.assertIn("no calibrated cell", html)
            calibrated = {**conf, "epsilon_source": "calibrated"}
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)],
                                          conf_a=calibrated, conf_b=calibrated)
            self.assertRegex(report.build(runs, "t"), r"calibrated relative \u03b5")

    def test_a_licensed_verdict_is_green_and_a_refused_one_red(self):
        """The passing verdict was drawn in the caveat amber, so the one piece of
        good news on the page looked like the warnings under it."""
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            # The fixture's env.txt fails §7.1, which is red on its own account.
            self.assertIn('class="banner " id="verdict"', report.build(runs, "t"))
            for r in runs:
                r.gate_pass = True
            self.assertIn('class="banner ok" id="verdict"', report.build(runs, "t"))
            self.assertIn(".banner.ok{", report.CSS)
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)],
                                          conf_a=CONF_T2, conf_b=CONF_T2)
            for r in runs:
                r.gate_pass = True
            self.assertIn('class="banner " id="verdict"', report.build(runs, "t"))

    def test_every_chart_frame_the_template_uses_is_styled(self):
        """Four sections framed their charts in `.card`, which nothing defined."""
        tpl = (Path(self.report.__file__).parent / "templates" / self.report.TEMPLATE).read_text()
        for klass in set(re.findall(r'<(?:div|figure) class="(card|chart)"', tpl)):
            self.assertRegex(self.report.CSS, rf"(^|[,}}\s])(figure)?\.{klass}[,{{]", klass)

    def test_throughput_table_wraps_its_descriptions(self):
        """Held to one line, W11-steady's description pushed the ratio column past
        the card's edge and crushed the notes to one word wide."""
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            self.assertIn('<table class="throughput">', report.throughput_table(runs, df))
            self.assertIn("table.throughput td.desc{white-space:normal", report.CSS)

    def test_the_ingest_wait_is_hatched_in_the_engine_colour(self):
        """A page-coloured hatch drew near-white lines on a transparent fill:
        the wait-for-Green bars and their legend swatches were invisible."""
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(
                Path(tmp), [_row("W1", None, upload_s=1.0, index_wait_s=0.5)],
                [_row("W1", None, upload_s=10.0, index_wait_s=2.0)])
            chart = report.chart_build(runs, df)
            self.assertIsNotNone(chart)
            self.assertIn('"pattern":{"shape"', chart["html"])
            self.assertNotIn('"fgcolor"', chart["html"])

    def test_a_folded_label_measures_over_all_its_passes(self):
        """The fold copied pass 1's run.json, so the page's "measured" window
        was a third of the run and the record introduced itself as -rep1."""
        import importlib
        agg = importlib.import_module("aggregate")
        spans = agg.pass_spans(["x-rep1", "x-rep2"], [
            [{"id": "W3", "when": "2026-09-03T02:00:31Z"},
             {"id": "W4", "when": "2026-09-03T02:12:58Z"}],
            [{"id": "W3", "when": "2026-09-03T02:39:33Z"},
             {"id": "W4", "when": "2026-09-03T02:52:50Z"}]])
        # No run.json on disk for these: `started` is None, the rows still span.
        self.assertEqual([s["label"] for s in spans], ["x-rep1", "x-rep2"])
        self.assertEqual([s["last_row"] for s in spans],
                         ["2026-09-03T02:12:58Z", "2026-09-03T02:52:50Z"])
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            runs[0].meta["passes"] = [
                {"label": "a-rep1", "started": "2026-09-03T02:00:31Z",
                 "last_row": "2026-09-03T02:12:58Z"},
                {"label": "a-rep2", "started": "2026-09-03T02:39:33Z",
                 "last_row": "2026-09-03T02:52:50Z"}]
            html = report.build(runs, "t")
            self.assertIn("2026-09-03T02:00:31Z to 2026-09-03T02:52:50Z (2 passes)", html)

    def test_licence_carries_the_delta_behind_the_tier(self):
        """§8.9 lists max|Δscore| and p99|Δscore| as row fields. They were
        measured and then rendered into T1's prose, so the page carried the
        verdict and not the number behind it."""
        with tempfile.TemporaryDirectory() as tmp:
            conf = {**CONF_T3, "max_delta": 1.5e-6, "p99_delta": 2.0e-7,
                    "epsilon": 9.5e-7, "epsilon_relative": False}
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], conf_a=conf, conf_b=conf)
            lic = report.licence_of(runs)
            self.assertEqual(lic["delta"]["max"], 1.5e-6)
            self.assertEqual(lic["delta"]["p99"], 2.0e-7)
            self.assertFalse(lic["delta_missing"])
            html = report.build(runs, "t")
            self.assertIn("1.500e-06", html)
            # `epsilon_relative` is a bool saying which kind of tolerance ε is,
            # not a second ε. Formatted as a number it printed `True` as
            # `1.000e+00`, so the page claimed a 100% tolerance.
            self.assertEqual(lic["delta"]["kind"], "absolute")
            self.assertRegex(html, r"absolute \u03b5 of\s+9\.500e-07")
            self.assertNotIn("1.000e+00", html)
            rel = {**conf, "epsilon_relative": True}
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], conf_a=rel, conf_b=rel)
            self.assertEqual(report.licence_of(runs)["delta"]["kind"], "relative")
            # A differ run from before the fields existed says so; it does not
            # show a blank, and it does not show a zero.
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)])
            lic = report.licence_of(runs)
            self.assertIsNone(lic["delta"])
            self.assertTrue(lic["delta_missing"])
            self.assertIn("predates its structured fields", report.build(runs, "t"))
            # No conformance row at all is a third state, and not this one.
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], conf_a={}, conf_b={})
            self.assertFalse(report.licence_of(runs)["delta_missing"])

    def test_the_recall_table_says_what_the_builds_disagreed_by(self):
        """Two engines 0.0001 apart at the top of the frontier, one of them
        0.0028 apart from itself across two builds of the same collection. The
        table printed the folded recall alone, which states a draw as a
        property of the engine."""
        report = self.report

        def run(label, spread):
            pts = [{"ef": 512, "recall_at_1": 0.999, "recall_at_10": 0.998,
                    "recall_at_10_ci95_low": 0.9955, "recall_at_10_ci95_high": 0.9998,
                    "mean_relative_distance_error": 1e-5,
                    "rep_spread": spread, "reps": 3}]
            return report.Run(label, [], "", True, "h",
                              recall={"points": pts, "queries": 10000,
                                      "limit": 10, "epsilon": 1e-7})

        html = report.recall_table([run("strawmann", 0.0028), run("qdrant", 0.00007)])
        self.assertIn("±0.0014 over 3 builds", html)
        self.assertIn("at ef=512 spread by: strawmann 0.00280; qdrant 0.00007", html)
        # The asymmetry is the finding, and it is only claimed when it is one.
        self.assertIn("40x as much", html)
        even = report.recall_table([run("a", 0.001), run("b", 0.0009)])
        self.assertIn("independent builds", even)
        self.assertNotIn("as much as", even)
        # A single-pass run has nothing to say here and says nothing.
        one = report.Run("solo", [], "", True, "h", recall={
            "points": [{"ef": 512, "recall_at_10": 0.998}], "queries": 10000})
        self.assertNotIn("independent builds", report.recall_table([one]))
        self.assertNotIn("builds</span>", report.recall_table([one]))

    def test_the_fold_medians_every_measured_column_not_just_throughput(self):
        """`fold` copies pass 1 wholesale and medians only what `NUMERIC` names.

        That list was hand-kept and had fallen behind every counter added since
        it was written: the scheduler columns (`ctx_switches_voluntary` among
        them), all six PSI figures, the anon/file resident split and every
        hardware counter were absent. A `--reps 3` row would have carried pass
        one's IPC, DRAM traffic and stall time beside a median qps, with
        `reps: 3` on the row saying all of it was folded — the same shape as
        the recall bug in the test below, in the same function, one field set
        over.

        So the list is built from the modules that produce the fields, and this
        asserts the property rather than the list: anything a row carries as a
        measured number is medianed, and the strings that describe the capture
        are not.
        """
        import importlib
        agg = importlib.import_module("aggregate")
        procstat = importlib.import_module("procstat")
        perfstat = importlib.import_module("perfstat")

        measured = {
            "qps": 100.0, "perf_cycles": 1e9, "perf_instructions": 2e9,
            "perf_dram_fills": 1e6, "perf_dtlb_walks": 5e5,
            "psi_io_full_s": 0.01, "psi_mem_some_s": 0.02,
            "rss_anon_bytes": 100, "rss_peak_bytes": 200,
            "ctx_switches_voluntary": 10.0, "blkio_delay_s": 0.5,
            "runqueue_wait_s": 0.25,
        }

        def row(mult):
            r = {"id": "W3", "status": "ok", "latency": {},
                 "perf_set": "default", "psi_scope": "engine", "io_source": "proc"}
            r.update({k: v * mult for k, v in measured.items()})
            return r

        rows, _rsd, _notes = agg.fold([[row(1)], [row(2)], [row(3)]])
        got = rows[0]
        for key, base in measured.items():
            self.assertAlmostEqual(
                got[key], base * 2, places=6,
                msg=f"{key} was not medianed across the passes")
        # What describes the capture rather than the measurement stays put.
        self.assertEqual(got["perf_set"], "default")
        self.assertEqual(got["psi_scope"], "engine")
        self.assertEqual(got["reps"], 3)

        # No module's columns may be sitting on the configuration list, which
        # is the only way one of them now escapes the fold.
        for name, fields in (("procstat.SCHED_FIELDS", procstat.SCHED_FIELDS),
                             ("procstat.PSI_FIELDS", procstat.PSI_FIELDS),
                             ("procstat.RSS_FIELDS", procstat.RSS_FIELDS),
                             ("perfstat.PERF_FIELDS", perfstat.PERF_FIELDS)):
            escaped = [f for f in fields if f in agg.CONFIG_NUMERIC]
            self.assertEqual(escaped, [], f"{name} would not be folded: {escaped}")

    def test_the_fold_keeps_a_row_the_passes_agree_is_not_ok(self):
        """The status rule is agreement, and the statuses are `Status` values.

        W12 is `Status.not_applicable` on strawmANN in every pass and must
        still fold, or the report loses its "declined, not implemented" note.
        The first version of this test spelled the statuses `"declined"` and
        `"failed"` -- neither is a `Status` value ("declined" is note text
        derived from `n/a`), so it pinned strings no row carries and would have
        gone green through a change that dropped `n/a` from the fold.
        """
        import importlib
        agg = importlib.import_module("aggregate")
        workloads = importlib.import_module("workloads")

        def row(status, qps=None, detail=""):
            # A fresh dict per pass: `[row(...)] * 3` shares one object, so a
            # fold that took pass 3 instead of pass 1 would not be caught.
            return {"id": "W12", "status": status, "latency": {},
                    "qps": qps, "seconds": 10.0, "detail": detail}

        # Agreed non-ok: folded, still says what it was, and medians the
        # numeric column it carries.
        rows, _rsd, notes = agg.fold(
            [[row(workloads.Status.not_applicable, 5.0)] for _ in range(3)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], workloads.Status.not_applicable)
        self.assertEqual((rows[0]["reps"], rows[0]["seconds"]), (3, 10.0))
        self.assertEqual(notes, [])

        # Statuses that disagree are refused even when none of them is `ok` --
        # `n/a`, `FAILED`, `FAILED` used to fold into a clean-looking declined
        # row, because the check only fired when some pass was `ok`.
        rows, _rsd, notes = agg.fold([[row(workloads.Status.not_applicable)],
                                      [row(workloads.Status.failed)],
                                      [row(workloads.Status.failed)]])
        self.assertEqual(rows, [])
        self.assertTrue(any("statuses disagree" in n for n in notes), notes)

        # Two ok and one not is not two thirds of a measurement.
        rows, _rsd, notes = agg.fold([[row(workloads.Status.ok, 100.0)],
                                      [row(workloads.Status.ok, 102.0)],
                                      [row(workloads.Status.failed)]])
        self.assertEqual(rows, [])
        self.assertTrue(any("statuses disagree" in n for n in notes), notes)

        # A row missing from a pass gets its own refusal, not a median over
        # the passes that happened to have it.
        rows, _rsd, notes = agg.fold([[row(workloads.Status.ok, 100.0)],
                                      [row(workloads.Status.ok, 101.0)], []])
        self.assertEqual(rows, [])
        self.assertTrue(any("measured in 2 of 3 passes" in n for n in notes), notes)

    def test_a_column_only_one_pass_measured_is_not_published_as_a_median(self):
        """The column-level form of the refusal above.

        `_median` keeps only numbers, so a counter the scheduler sampler
        produced in one pass of three was medianed over that single value and
        published on a row stamped `reps: 3` — `qd-dbp100k-perf`'s W8-upload
        `ctx_switches_voluntary` is a shipped example. A reader had nothing to
        tell it apart from a three-pass median.
        """
        import importlib
        agg = importlib.import_module("aggregate")
        workloads = importlib.import_module("workloads")

        def row(**extra):
            r = {"id": "W2", "status": workloads.Status.ok, "latency": {},
                 "qps": 100.0}
            r.update(extra)
            return r

        rows, _rsd, notes = agg.fold([[row(ctx_switches_voluntary=2729.0)],
                                      [row()], [row()]])
        self.assertIsNone(rows[0]["ctx_switches_voluntary"])
        self.assertTrue(any("ctx_switches_voluntary measured in 1 of 3" in n
                            for n in notes), notes)
        # The column every pass did measure is unaffected.
        self.assertEqual(rows[0]["qps"], 100.0)

    def test_a_latency_percentile_one_pass_measured_is_not_published_either(self):
        """The same refusal, one field set over, where it was missing.

        The column loop refuses a partial column and says so; `latency` is a
        nested dict and went through a bare `_median` that drops non-numbers,
        so a percentile only one pass carried was published as that pass's raw
        value on a row stamped `reps: 3`. The percentiles are what the
        comparison tables print beside the throughput ratio, so an unmarked
        pass-1 p95 there is exactly the shipped `ctx_switches_voluntary` bug
        in the field a reader is most likely to quote.
        """
        import importlib
        agg = importlib.import_module("aggregate")
        workloads = importlib.import_module("workloads")

        def row(mult, **lat):
            base = {"p50": 1.0 * mult}
            base.update(lat)
            return {"id": "W6", "status": workloads.Status.ok,
                    "qps": 100.0 * mult, "latency": base}

        rows, _rsd, notes = agg.fold([[row(1, p95=9.0)], [row(2)], [row(3)]])
        self.assertIsNone(rows[0]["latency"]["p95"])
        self.assertTrue(any("latency p95 measured in 1 of 3" in n
                            for n in notes), notes)
        # The percentile every pass did measure is still medianed.
        self.assertEqual(rows[0]["latency"]["p50"], 2.0)

    def test_the_fold_medians_workloads_own_columns_too(self):
        """The half of the old list that no module could derive, and that drifted.

        `NUMERIC` was built from `procstat` and `perfstat` precisely so their
        columns could not fall out of it. `workloads.py`'s own were hand-typed
        in `_BASE_NUMERIC`, and by a folded row carried pass one's
        `seconds`, `warmup_s`, `load_start`, `load_end`, `qps_bfb_median` and
        every one of W11's write-overlap figures — 87 values in a 32-row label
        — beside a median qps, with `reps: 3` saying otherwise. The published
        W11 note read "write overlap 18.8%, append 21,117 points/s" from one
        pass next to a throughput folded from three.

        Asserted as the property: a column is folded because it was measured.
        """
        import importlib
        agg = importlib.import_module("aggregate")
        measured = {"seconds": 10.0, "warmup_s": 0.5, "load_start": 4.0,
                    "load_end": 6.0, "qps_bfb_median": 900.0,
                    "background_pps": 1000.0, "background_s": 2.0,
                    "overlap_s": 2.5, "write_overlap_pct": 12.0,
                    "engine_settle_s": 3.0, "engine_settle_cores": 0.02}

        def row(mult):
            r = {"id": "W11", "status": "ok", "latency": {},
                 # Configuration: shared by the passes, so pass 1's is right.
                 "ef": 128, "n_requested": 50_000, "rps_target": 5000.0,
                 # A flag is not a number to take the median of.
                 "recall_joinable": mult == 1}
            r.update({k: v * mult for k, v in measured.items()})
            return r

        rows, _rsd, notes = agg.fold([[row(1)], [row(2)], [row(3)]])
        got = rows[0]
        for key, base in measured.items():
            self.assertAlmostEqual(got[key], base * 2, places=6,
                                   msg=f"{key} was not medianed across the passes")
        self.assertEqual((got["ef"], got["n_requested"], got["rps_target"]),
                         (128, 50_000, 5000.0))
        self.assertIs(got["recall_joinable"], True)
        self.assertEqual(notes, [])

        # A "configuration" that moved between passes is reported, not folded
        # away: the passes did not share it, which is the interesting part.
        a, b = row(1), row(1)
        b["ef"] = 256
        _rows, _r, notes = agg.fold([[a], [b], [row(1)]])
        self.assertTrue(any("ef differs between passes" in n for n in notes), notes)

    def test_the_fold_medians_recall_too_and_keeps_the_build_spread(self):
        """`rows.json` carried the median qps of three passes while the recall
        sweeps were pass 1's, copied verbatim -- so the matched-recall table
        paired one build's recall with three builds' throughput. On strawmANN
        those are not interchangeable: two builds of the same collection with
        the same parameters measured 0.99957 and 0.99678 at ef 512, and mean
        relative distance errors a factor of 44 apart."""
        import importlib
        agg = importlib.import_module("aggregate")

        def sweep(r10, ci, mrde):
            return {"label": "x", "collection": "bench2", "queries": 10000,
                    "points": [{"ef": 512, "recall_at_1": r10, "recall_at_10": r10,
                                "recall_at_10_ci95_low": ci[0],
                                "recall_at_10_ci95_high": ci[1],
                                "mean_relative_distance_error": mrde}]}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old, agg.ROOT = agg.ROOT, root
            try:
                vals = [(0.99957, (0.99893, 0.99983), 8.61e-06),
                        (0.99678, (0.99546, 0.99772), 3.83e-04),
                        (0.99800, (0.99700, 0.99880), 9.00e-05)]
                for i, v in enumerate(vals, 1):
                    d = root / "bench/results" / f"r{i}"
                    d.mkdir(parents=True)
                    (d / "recall.sift1m.bench2.json").write_text(json.dumps(sweep(*v)))
                dest = root / "folded"
                dest.mkdir()
                notes = []
                agg.fold_recall(dest, ["r1", "r2", "r3"], notes)
                got = json.loads((dest / "recall.sift1m.bench2.json").read_text())
                pt = got["points"][0]
                # Median of the point estimates, not the first pass's.
                self.assertAlmostEqual(pt["recall_at_10"], 0.99800)
                self.assertAlmostEqual(pt["mean_relative_distance_error"], 9.00e-05)
                # The interval is the union of the passes', so it cannot come
                # out narrower than the values it was folded from.
                self.assertAlmostEqual(pt["recall_at_10_ci95_low"], 0.99546)
                self.assertAlmostEqual(pt["recall_at_10_ci95_high"], 0.99983)
                self.assertLessEqual(pt["recall_at_10_ci95_low"], min(v[0] for v in vals))
                self.assertGreaterEqual(pt["recall_at_10_ci95_high"], max(v[0] for v in vals))
                # The build variance is kept rather than folded away.
                self.assertAlmostEqual(pt["rep_spread"], 0.99957 - 0.99678)
                self.assertEqual(pt["reps"], 3)
                self.assertEqual(got["rep_labels"], ["r1", "r2", "r3"])
                self.assertEqual(notes, [])

                # A sweep one pass did not run is medianed over the rest, and
                # said so: a missing sweep is not a zero.
                (root / "bench/results/r3/recall.sift1m.bench6.json").write_text(
                    json.dumps(sweep(0.9, (0.89, 0.91), 1e-3)))
                notes = []
                agg.fold_recall(dest, ["r1", "r2", "r3"], notes)
                self.assertIn("present in 1 of 3 passes", notes[0])
            finally:
                agg.ROOT = old

    def test_aggregate_medians_and_refuses_partial_rows(self):
        """§7.4: "median of medians and the spread". A median so one unlucky
        pass moves nothing, and a refusal where the passes disagree about
        whether the row worked at all."""
        import importlib
        agg = importlib.import_module("aggregate")

        def row(wid, qps, status="ok", **extra):
            lat = {"client_p50_us": 1000.0 / qps} if qps else {}
            return {"id": wid, "status": status, "qps": qps, "latency": lat, **extra}

        # One wild pass must not move the answer: median of 1000/1100/9000.
        # Outlier in the middle pass, not the last: this asserts the median
        # resists it, and an outlier arriving last would also be a monotone
        # sequence, which `_rep_drift` reports separately (and rightly — a
        # spread that large is not a noise band). Kept apart so each test says
        # one thing.
        rows, rsd, notes = agg.fold([[row("W3", 1000.0)], [row("W3", 9000.0)],
                                     [row("W3", 1100.0)]])
        self.assertEqual(rows[0]["qps"], 1100.0)
        self.assertEqual(rows[0]["reps"], 3)
        self.assertGreater(rsd["W3"], 1.0)      # the spread reports the outlier
        self.assertEqual(notes, [])

        # ok twice, failed once: not two thirds of a measurement.
        rows, _, notes = agg.fold([[row("W3", 1000.0)], [row("W3", 1100.0)],
                                   [row("W3", None, status="FAILED")]])
        self.assertEqual(rows, [])
        # The refusal is now stated as disagreement rather than as a count of
        # `ok`s, because the count-based form only fired when some pass was
        # `ok` — three passes that all failed differently slipped through it.
        self.assertIn("statuses disagree", notes[0])
        self.assertIn("FAILED", notes[0])

        # Present in only some passes: also refused, and named.
        rows, _, notes = agg.fold([[row("W3", 1000.0)], [row("W3", 1100.0)], []])
        self.assertEqual(rows, [])
        self.assertIn("measured in 2 of 3 passes", notes[0])

        # Two passes give a difference, not a standard deviation.
        rows, rsd, _ = agg.fold([[row("W3", 1000.0)], [row("W3", 1100.0)]])
        self.assertEqual(rows[0]["qps"], 1050.0)
        self.assertEqual(rsd, {})
        self.assertNotIn("rep_rsd", rows[0])

        # Contamination in any pass is carried, named by pass.
        rows, _, _ = agg.fold([[row("W3", 1000.0)],
                               [row("W3", 1100.0, foreign="zig(99%)")],
                               [row("W3", 1050.0)]])
        self.assertIn("pass 2: zig(99%)", rows[0]["foreign"])

    def test_folded_run_is_not_reported_as_sequential(self):
        """`--reps N` alternates the arms, but the fold carries pass 1's
        timestamps, so the stamp test would read it as A-then-B and warn about
        a confound the run was built to remove."""
        report = self.report
        folded = [report.Run("a", [{"id": "W3", "when": "2026-01-01T00:00:00Z"}],
                             "", True, "h", meta={"reps": 3}),
                  report.Run("b", [{"id": "W3", "when": "2026-01-01T00:00:00Z"}],
                             "", True, "h", meta={"reps": 3})]
        iv = report.interleaving(folded)
        self.assertFalse(iv["sequential"])
        self.assertIn("folded from 3 alternated passes", iv["spans"][0])
        # A single-pass run still gets the stamp test.
        one = [report.Run("a", [{"id": "W3", "when": "2026-01-01T00:00:00Z"}],
                          "", True, "h", meta={"reps": 1}),
               report.Run("b", [{"id": "W3", "when": "2026-01-01T00:00:01Z"}],
                          "", True, "h", meta={"reps": 1})]
        self.assertTrue(report.interleaving(one)["sequential"])

    def test_interleaving_is_computed_from_the_row_stamps(self):
        """§7.2(5): A/B/A/B, not A-then-B, to control for drift."""
        with tempfile.TemporaryDirectory() as tmp:
            def at(wid, qps, when):
                return {**_row(wid, qps), "when": when}
            # A-then-B: every `a` row before every `b` row.
            report, runs, df = self._pair(
                Path(tmp), [at("W3", 4000, "2026-08-18T20:41:00Z"),
                            at("W4", 8000, "2026-08-18T20:55:00Z")],
                [at("W3", 2000, "2026-08-18T21:00:00Z"),
                 at("W4", 4000, "2026-08-18T21:21:00Z")])
            iv = report.interleaving(runs)
            self.assertEqual((iv["blocks"], iv["sequential"], iv["first"]), (2, True, "a"))
            self.assertIn("were not interleaved", report.build(runs, "t"))
            # A/B/A/B: four blocks, and no banner.
            report, runs, df = self._pair(
                Path(tmp), [at("W3", 4000, "2026-08-18T20:00:00Z"),
                            at("W4", 8000, "2026-08-18T20:20:00Z")],
                [at("W3", 2000, "2026-08-18T20:10:00Z"),
                 at("W4", 4000, "2026-08-18T20:30:00Z")])
            iv = report.interleaving(runs)
            self.assertEqual((iv["blocks"], iv["sequential"]), (4, False))
            self.assertNotIn("were not interleaved", report.build(runs, "t"))
            # Stamps that cannot answer it are unknown, not sequential, and
            # get no banner: `_row`'s rows all carry the same instant, so the
            # merge order would be the sort's rather than the run's.
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            self.assertEqual(report.interleaving(runs), {})
            self.assertNotIn("were not interleaved", report.build(runs, "t"))
            # One instant shared across the two engines is the same ambiguity
            # even when the rest of the stamps differ.
            report, runs, df = self._pair(
                Path(tmp), [at("W3", 4000, "2026-08-18T20:00:00Z"),
                            at("W4", 8000, "2026-08-18T20:20:00Z")],
                [at("W3", 2000, "2026-08-18T20:20:00Z"),
                 at("W4", 4000, "2026-08-18T20:30:00Z")])
            self.assertEqual(report.interleaving(runs), {})

    def test_a_pinned_reference_is_not_described_as_measured(self):
        """`--rps-reference 10313` pins's saturation for both arms;
        the row header called it "measured 10,313 qps" beside a W4 of 9,939."""
        with tempfile.TemporaryDirectory() as tmp:
            def arm(src):
                return {**_row("W4-sat90", 9282), "load_mode": "open-loop",
                        "rps_target": 9282, "rps_fraction": 0.9,
                        "saturation_qps": 10313.0, "rps_reference_source": src}
            report, runs, df = self._pair(Path(tmp), [arm("pinned")], [arm("pinned")])
            html = report.build(runs, "t")
            self.assertIn("pinned reference 10,313 qps", html)
            self.assertNotIn("measured 10,313 qps", html)
            report, runs, df = self._pair(Path(tmp), [arm("own")], [arm("own")])
            self.assertIn("measured 10,313 qps", report.build(runs, "t"))

    def test_open_loop_arms_at_different_rates_are_not_compared(self):
        """§4's fractions, taken per engine, are two different offered loads.
        docs/workloads.md: "comparing p99s taken at each engine's own
        saturation point compares two different offered loads"."""
        with tempfile.TemporaryDirectory() as tmp:
            def arm(qps, rate, src):
                return {**_row("W4-sat50", qps), "load_mode": "open-loop",
                        "latency": {"client_p50_us": 900.0, "client_p99_us": 1800.0,
                                    "server_p50_us": 800.0},
                        "rps_target": rate, "rps_fraction": 0.5,
                        "saturation_qps": rate * 2, "rps_reference_source": src}
            # Each engine at half of its own saturation: different rates, so
            # the percentiles are not side by side comparable.
            report, runs, df = self._pair(Path(tmp), [arm(11420, 11420, "own")],
                                          [arm(7489, 7489, "own")])
            msg = report.open_loop_mismatch(runs)
            self.assertIn("W4-sat50", msg)
            self.assertIn("--rps-reference", report.latency_table(runs))
            # One pinned reference: same rate on both, and no refusal.
            report, runs, df = self._pair(Path(tmp), [arm(7489, 7489, "pinned")],
                                          [arm(7489, 7489, "pinned")])
            self.assertEqual(report.open_loop_mismatch(runs), "")
            self.assertNotIn("Not compared", report.latency_table(runs))
            # Same rate reached without pinning is still not a refusal: the
            # test is the offered load, not the bookkeeping.
            report, runs, df = self._pair(Path(tmp), [arm(9000, 9000, "own")],
                                          [arm(9000, 9000, "own")])
            self.assertEqual(report.open_loop_mismatch(runs), "")

    def test_open_loop_row_short_of_its_offer_says_so(self):
        """"offered" reads as "both engines kept up". A row that delivered 93%
        of what it was asked for did not, and `client p50 > 1 s` does not catch
        it: at 7,490/s offered the arms served 6,972 and 6,373 with a p50 of
        700 ms, well under the saturation threshold."""
        with tempfile.TemporaryDirectory() as tmp:
            def arm(qps, target):
                return {**_row("W4-sat50", qps), "load_mode": "open-loop",
                        "rps_target": target, "rps_fraction": 0.5,
                        "latency": {"client_p50_us": 700_000.0,
                                    "server_p50_us": 8_000.0}}
            m = _reload(Path(tmp))["compare"]
            fx = Fixture(Path(tmp))
            fx.label("a", [arm(6972, 7490)], good_stamp(), CONF_T3, [])
            fx.label("b", [arm(6373, 7490)], good_stamp(), CONF_T3, [])
            r = m.joined("a", "b", m.load("a"), m.load("b"))[0]
            self.assertEqual(r.ratio, "short of offer")
            note = " ".join(r.notes)
            self.assertIn("offered rate not served", note)
            self.assertIn("6,972 of 7,490 (93%)", note)
            self.assertIn("6,373 of 7,490 (85%)", note)
            # A row that did serve its offer keeps the plain word.
            fx.label("c", [arm(7400, 7490)], good_stamp(), CONF_T3, [])
            fx.label("d", [arm(7450, 7490)], good_stamp(), CONF_T3, [])
            r = m.joined("c", "d", m.load("c"), m.load("d"))[0]
            self.assertEqual(r.ratio, "offered")
            self.assertNotIn("not served", " ".join(r.notes))

    def test_affinity_cell_distinguishes_per_thread_from_main_only(self):
        """The cores row is the comparison's central fairness fact, and the
        first version read `/proc/<pid>/status` — the main thread — so it
        printed `0-11` for an engine whose workers were pinned to `4-11`."""
        report = self.report
        run = report.Run("a", [], "", True, "h",
                         meta={"engine_affinity": "4-11 x8, 0-11 x2",
                               "server_cpus_requested": "4-11"})
        cell = report._affinity_cell(run)
        self.assertIn("requested by the harness", cell)
        self.assertIn("observed on 8 threads", cell)
        self.assertIn("observed on 2 threads", cell)
        self.assertNotIn("main thread only", cell)
        # Observed without a requested set cannot answer the fairness question,
        # and says so instead of leaving the reader to assume it did.
        bare = report.Run("a", [], "", True, "h",
                          meta={"engine_affinity": "4-11 x8"})
        self.assertIn("cannot be checked", report._affinity_cell(bare))
        # A capture from before per-thread masks were read is labelled, not
        # presented as though it described every thread.
        old = report.Run("a", [], "", True, "h", meta={"engine_affinity": "0-11"})
        self.assertIn("main thread only", report._affinity_cell(old))
        # And no capture at all is a third state.
        self.assertIn("not recorded",
                      report._affinity_cell(report.Run("a", [], "", True, "h")))

    def test_segment_claim_uses_the_read_back_not_the_cap(self):
        """The page said "capped at 8 segments" and "up to 1024 node visits" in
        three places while `collections.json`, displayed in the same report,
        said 5. The cap is what Qdrant might hold; the capture is what it did."""
        report = self.report

        def run(label, n):
            coll = {"collections": [{"collection": "bench2", "segments_count": n}]}
            return report.Run(label, [], "", True, "h", collections=coll)

        runs = [run("strawmann", 1), run("qdrant", 5)]
        self.assertEqual(report.segments_of(runs), {"strawmann": 1, "qdrant": 5})
        note = report.segment_note(runs)
        self.assertIn("qdrant held 5 segments", note)
        self.assertIn("up to 640 node visits", note)   # 128 x 5, not 128 x 8
        self.assertNotIn("1,024", note)
        self.assertNotIn("capped at 8", note)
        # No capture falls back to the cap, and says that is what it is quoting.
        bare = [report.Run("strawmann", [], "", True, "h"),
                report.Run("qdrant", [], "", True, "h")]
        self.assertEqual(report.segments_of(bare), {})
        fallback = report.segment_note(bare)
        self.assertIn("capped at 8", fallback)
        self.assertIn("the cap, not a measurement", fallback)
        # A single-segment Qdrant makes the whole caveat inapplicable, so it
        # must not claim a multiplier of 1 as though it were a confound.
        one = [run("strawmann", 1), run("qdrant", 1)]
        self.assertNotIn("node visits", report.segment_note(one))

    def test_an_empty_appendable_is_a_segment_and_not_a_graph(self):
        """sift1m's Qdrant "held 2 segments": one graph over every point and the
        empty appendable it keeps for writes. The note had to call the count an
        upper bound; read back per segment, it can say which."""
        report = self.report

        def run(label, n, populated=None):
            c = {"collection": "bench2", "segments_count": n}
            if populated is not None:
                c["populated_segments_count"] = populated
            return report.Run(label, [], "", True, "h",
                              collections={"collections": [c]})

        equal = [run("strawmann", 1), run("qdrant", 2, populated=1)]
        self.assertEqual(report.populated_of(equal), {"qdrant": 1})
        note = report.segment_note(equal)
        self.assertIn("ef is the same unit here", note)
        self.assertIn("one populated and 1 empty", note)
        self.assertNotIn("node visits", note)
        # Four graphs and an empty one multiply by four, not five.
        four = report.segment_note([run("strawmann", 1), run("qdrant", 5, populated=4)])
        self.assertIn("5 segments, 4 of them populated", four)
        self.assertIn("up to 512 node visits", four)
        # A run captured before the per-segment read-back reads as it did.
        self.assertIn("up to 256 node visits",
                      report.segment_note([run("strawmann", 1), run("qdrant", 2)]))

    def test_the_state_the_search_rows_saw_is_the_one_the_table_shows(self):
        """The survivor read-back happened once, at the end of the arm, after
        W11 had appended into bench2 -- so the record said 1,250,000 points
        beside recall and throughput taken at 1,000,000, and the page banners
        the mismatch. Captured before the mutating rows as well, the table is
        the state the search rows were measured in and the post-append state is
        a second fact rather than a warning about the first."""
        report = self.report
        coll = {"collections": [{"collection": "bench2", "points_count": 1_000_000,
                                 "segments_count": 1,
                                 "indexed_vectors_count": 1_000_000,
                                 "after_mutating_rows": {
                                     "collection": "bench2",
                                     "points_count": 1_250_000,
                                     "indexed_vectors_count": 1_050_000}}]}
        runs = [report.Run("strawmann", [], "", True, "h", collections=coll,
                           meta={"upload_n": 1_000_000, "harness": {"w11_n": 250_000}})]
        html = report.collection_table(runs)
        self.assertNotIn("Captured after the mutating rows", html)
        self.assertIn("After the mutating rows", html)
        self.assertIn("1,250,000 (+250,000)", html)
        self.assertIn("1,050,000 of them indexed", html)
        # The nested capture is a second look at bench2, not a second
        # collection, and not a metric row.
        self.assertEqual(html.count('<td class="wid">bench2</td>'), 1)
        self.assertNotIn("after_mutating_rows", html)
        # A run measured before the pre-mutation capture existed still gets the
        # banner, because for it the late read-back is all there is.
        old = {"collections": [{"collection": "bench2", "points_count": 1_250_000,
                                "segments_count": 1}]}
        self.assertIn("Captured after the mutating rows", report.collection_table(
            [report.Run("strawmann", [], "", True, "h", collections=old,
                        meta={"upload_n": 1_000_000})]))

    def test_a_band_too_wide_to_discriminate_says_so(self):
        """"within the ±64% band: no measured difference, not a small one" is the
        right sentence at ±5% and a false comfort at ±64%: nothing the row could
        plausibly have measured would have cleared it. The sift1m run
        printed exactly that, off a three-sample rsd where Qdrant's W3 read
        1,368, 2,063 and 1,580."""
        report = self.report
        v = report.ratio_verdict
        self.assertIn("clears", v("W3", 2.00, {"W3": 0.01}, 3))
        self.assertIn("inconclusive", v("W3", 1.01, {"W3": 0.03}, 3))
        wide = v("W3", 1.15, {"W3": 0.15}, 3)
        self.assertIn("too noisy to judge", wide)
        self.assertIn("over 3 passes", wide)
        self.assertNotIn("inconclusive", wide)
        # A ratio that clears even a wide band is still a result.
        self.assertIn("clears", v("W3", 5.0, {"W3": 0.15}, 3))
        # And an unmeasured row is a third thing, unchanged.
        self.assertIn("no noise floor", v("W3", 1.15, {}, 3))

    def test_declines_are_counted_apart_from_failures(self):
        """`25/27 ok` says two rows went wrong. Both were W12, which the engine
        refused by name because filtered search is not built -- and on dbpedia
        that count sat beside a *green* pill saying the gate passed, the number
        and the colour disagreeing about what happened."""
        report = self.report
        S = report.Status
        rows = ([{"id": f"W{i}", "status": S.ok} for i in range(25)]
                + [{"id": "W12", "status": S.not_applicable},
                   {"id": "W12-upload", "status": S.not_applicable}])
        r = report.Run("strawmann", rows, "", True, "h")
        self.assertEqual(r.n_ok, 25)
        self.assertEqual(r.n_declined, 2)
        self.assertEqual(r.n_measured, 25)          # 25/25, not 25/27
        # A genuine failure still counts against the measured total.
        rows[0] = {"id": "W0", "status": S.failed}
        r = report.Run("strawmann", rows, "", True, "h")
        self.assertEqual((r.n_ok, r.n_measured, r.n_declined), (24, 25, 2))
        # And a run with nothing declined reads exactly as before.
        clean = report.Run("qdrant", [{"id": "W0", "status": S.ok}], "", True, "h")
        self.assertEqual((clean.n_ok, clean.n_measured, clean.n_declined), (1, 1, 0))

    def test_an_unnamed_placement_is_reported_not_dropped(self):
        """Qdrant at its dense default reports no `memory` field. The harness
        used to ask it for `cached` so it would say something -- moving it off
        its default to fill in a report sentence -- and once that stopped, the
        banner listed only strawmANN and silently omitted the engine whose
        residency it is contrasting."""
        report = self.report

        def run(label, placement):
            c = {"collection": "bench2"}
            if placement:
                c["placement"] = placement
            return report.Run(label, [], "", True, "h",
                              collections={"collections": [c]})

        got = report.placements_of([run("strawmann", "pinned"), run("qdrant", None)])
        self.assertEqual(got, ["pinned", report.UNREPORTED_PLACEMENT])
        # No capture at all stays None: unknown and "captured, unnamed" are
        # different answers.
        self.assertEqual(
            report.placements_of([report.Run("x", [], "", True, "h")]), [None])
        # Both engines naming the same residency is not a split.
        same = report.placements_of([run("a", "cached"), run("b", "cached")])
        self.assertEqual(len(set(same)), 1)

    def test_a_folded_run_is_judged_against_its_own_measured_spread(self):
        """`aggregate.py --reps N` writes each folded label a `noise.json` --
        that engine's spread, on that corpus, in that session -- and nothing
        read it. Both consumers went to the global file, a sift1m fold measured
        on strawmANN alone, so a repeated dbpedia run measured its own floor and
        was still banded against another corpus's, or against none."""
        import json as _json

        import regression
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "bench/results").mkdir(parents=True)
            old_results, regression.RESULTS = regression.RESULTS, root / "bench/results"
            old_default, regression.DEFAULT_NOISE = (
                regression.DEFAULT_NOISE, root / "bench/results/noise.json")
            try:
                regression.DEFAULT_NOISE.write_text(_json.dumps(
                    {"rsd": {"W3": 0.10}, "dataset": "sift1m", "source": "borrowed"}))
                # No fold on disk: the global file, exactly as before.
                self.assertEqual(regression.floor_for(["a", "b"])["source"], "borrowed")
                for lab, rsd, n in (("a", 0.03, 3), ("b", 0.04, 3)):
                    d = regression.RESULTS / lab
                    d.mkdir()
                    (d / "noise.json").write_text(_json.dumps({
                        "rsd": {"W3": rsd, "W4": rsd}, "reps": {"W3": n, "W4": n},
                        "source": f"{lab}: median of {n}", "discarded": [],
                        "harness_hash": "h", "n_dirs": n, "dataset": "dbpedia"}))
                got = regression.floor_for(["a", "b"])
                # sqrt(mean(rsd^2)), so noise_band(., arms=2) is
                # 3*sqrt(rsd_a^2 + rsd_b^2) -- two measured spreads, not one
                # engine's used for both.
                self.assertAlmostEqual(got["rsd"]["W3"], ((0.03**2 + 0.04**2) / 2) ** 0.5)
                self.assertAlmostEqual(regression.noise_band(got["rsd"]["W3"], arms=2),
                                       3 * (0.03**2 + 0.04**2) ** 0.5)
                self.assertEqual(got["dataset"], "dbpedia")
                self.assertEqual(got["arms"], ["a", "b"])
                # A row only one arm measured is not the ratio's spread.
                (regression.RESULTS / "b" / "noise.json").write_text(_json.dumps(
                    {"rsd": {"W3": 0.04}, "reps": {"W3": 3}, "dataset": "dbpedia"}))
                self.assertEqual(set(regression.floor_for(["a", "b"])["rsd"]), {"W3"})
            finally:
                regression.RESULTS = old_results
                regression.DEFAULT_NOISE = old_default

    def test_segment_claim_finds_qdrant_by_engine_not_by_label(self):
        """A `--dataset` run must relabel both arms, and the split was
        `label != "strawmann"` — so on `sm-dbp100k` vs `qd-dbp100k` it read the
        strawmANN arm's segment count as Qdrant's and published "ef is the same
        unit here" on a run where Qdrant held two segments."""
        report = self.report

        def run(label, engine, n):
            coll = {"collections": [{"collection": "bench2", "segments_count": n}]}
            return report.Run(label, [], "", True, "h", collections=coll,
                              meta={engine: {"present": True}})

        runs = [run("sm-dbp100k", "strawmann", 1), run("qd-dbp100k", "qdrant", 2)]
        note = report.segment_note(runs)
        self.assertIn("qd-dbp100k held 2 segments", note)
        self.assertIn("strawmANN held 1", note)
        self.assertNotIn("same unit here", note)
        # Order on the command line must not decide which arm is which.
        self.assertEqual(report.segment_note(list(reversed(runs))), note)
        # One capture missing is not a measurement that both held one segment.
        half = [run("sm-dbp100k", "strawmann", 1),
                report.Run("qd-dbp100k", [], "", True, "h",
                           meta={"qdrant": {"present": True}})]
        self.assertIn("the cap, not a measurement", report.segment_note(half))

    def test_per_core_pins_collapse_to_the_range_they_cover(self):
        """strawmANN pins each worker to one core, so the real capture on this
        host is nine groups — `0-11 x1, 4 x1, 5 x1, ... 11 x1` — true and
        unreadable, and it buries what a reader wants: those eight workers
        cover exactly the `4-11` that was requested."""
        report = self.report
        raw = "0-11 x1, 4 x1, 5 x1, 6 x1, 7 x1, 8 x1, 9 x1, 10 x1, 11 x1"
        self.assertEqual(report._mask_groups(raw), [("4-11", 8, 8), ("0-11", 1, 12)])
        # A cpuset-bound engine has one group already and must survive intact.
        self.assertEqual(report._mask_groups("4-11 x8"), [("4-11", 8, 8)])
        # Non-contiguous pins keep their gap rather than being smoothed over.
        self.assertEqual(report._mask_groups("4 x1, 6 x1")[0][0], "4,6")
        cell = report._affinity_cell(report.Run(
            "strawmann", [], "", True, "h",
            meta={"engine_affinity": raw, "server_cpus_requested": "4-11"}))
        self.assertIn("observed on 8 threads", cell)
        self.assertNotIn("x1", cell)

    def test_collections_share_one_table_and_mark_a_disagreement(self):
        """One table per collection, seven on sift1m, made the one field that
        differs (Qdrant's segment count) something to find seven times."""
        report = self.report

        def run(label, segs):
            return report.Run(label, [], "", True, "h", meta={"upload_n": 1000},
                              collections={"collections": [
                                  {"collection": c, "points_count": 1000,
                                   "segments_count": segs, "hnsw_m": 16}
                                  for c in ("bench2", "bench6")]})

        html = report.collection_table([run("sm", 1), run("qd", 2)])
        self.assertEqual(html.count("<table"), 1)
        self.assertEqual(html.count("<tr>"), 3)             # header and two rows
        self.assertEqual(html.count('<td class="num differs">1 / 2</td>'), 2)
        self.assertEqual(html.count('<td class="num">1,000</td>'), 2)   # agreed
        self.assertIn("cell reads sm / qd", html)
        # A field no collection reports gets no column at all.
        self.assertNotIn("quantization", html)

    def test_capture_after_the_mutating_rows_is_flagged(self):
        """`collections.json` is read back last, so for a collection W11
        appends to it describes a state no search row was measured in: bench2
        read 1,200,000 points beside recall and throughput taken at 1,000,000."""
        report = self.report

        def run(label, points):
            return report.Run(
                label, [], "", True, "h",
                # `w11_n` lives only under `harness` in a real `run.json`,
                # which is where reading the top level alone lost the clause.
                meta={"upload_n": 1_000_000, "harness": {"w11_n": 200_000}},
                collections={"collections": [{"collection": "bench2",
                                              "points_count": points,
                                              "segments_count": 1}]})

        runs = [run("strawmann", 1_200_000), run("qdrant", 1_200_000)]
        html = report.collection_table(runs)
        self.assertIn("Captured after the mutating rows", html)
        self.assertIn("1,200,000 against 1,000,000 uploaded", html)
        self.assertIn("W11's 200,000 appended points", html)
        # A collection nothing appended to carries no banner.
        clean = [run("strawmann", 1_000_000), run("qdrant", 1_000_000)]
        self.assertNotIn("Captured after", report.collection_table(clean))
        # A surplus that is not W11's is still reported, without naming it.
        odd = [run("strawmann", 1_050_000), run("qdrant", 1_000_000)]
        h = report.collection_table(odd)
        self.assertIn("Captured after", h)
        self.assertNotIn("W11's", h)

    def test_apply_skips_offline_cpus(self):
        """`apply` printed twelve "Device or resource busy" lines on a host it
        had just configured correctly: SMT off leaves the sibling threads
        offline, and an offline CPU keeps its cpufreq directory, so
        `Path.exists()` passes and the write is refused."""
        import importlib
        setup = importlib.import_module("setup")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n, online in ((0, None), (1, "1"), (2, "0")):
                d = root / f"cpu{n}" / "cpufreq"
                d.mkdir(parents=True)
                (d / "scaling_governor").write_text("performance\n")
                if online is not None:
                    (root / f"cpu{n}" / "online").write_text(online + "\n")
            gov = lambda n: root / f"cpu{n}" / "cpufreq" / "scaling_governor"
            # cpu0 has no `online` file because it cannot be offlined.
            self.assertTrue(setup.cpu_is_online(gov(0)))
            self.assertTrue(setup.cpu_is_online(gov(1)))
            self.assertFalse(setup.cpu_is_online(gov(2)))

    def test_time_to_green_reads_index_wait(self):
        fmt = next(f for name, f, _, _ in self.report.WORKLOAD_METRICS if name == "time to Green")
        raw = next(r for name, _, r, _ in self.report.WORKLOAD_METRICS if name == "time to Green")
        # A real-shaped W2 row: the flag is a bool, the duration is index_wait_s.
        w2 = {"id": "W2", "status": "ok", "wall_s": 41.2, "upload_s": 38.1, "index_wait_s": 3.01,
              "time_to_green_floored": True}
        self.assertEqual(fmt(w2), "3.0 s (at bfb's polling floor)")
        self.assertEqual(raw(w2), 3.01)
        w2 = {**w2, "index_wait_s": 12.4, "time_to_green_floored": False}
        self.assertEqual(fmt(w2), "12.4 s")
        self.assertEqual(fmt({"id": "W3", "status": "ok"}), "-")

    def test_a_floor_from_another_environment_bands_nothing_on_the_page(self):
        """`compare.parity_band` refuses a floor whose `env_hash` disagrees
        with the rows (findings 46); the page did not, so it printed a bare
        ratio and "clears the noise floor" under it from the refused floor."""
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], sw_a=[], sw_b=[])
            noise = report.ROOT / "bench/results/noise.json"
            noise.parent.mkdir(parents=True, exist_ok=True)
            floor = {"rsd": {"W3": 0.001}, "reps": {"W3": 6}, "source": "t",
                     "discarded": [], "arms": ["a", "b"]}
            # `compare.env_matches` reads the arms' `run.json`, which the
            # fixture writes without a hash: stamp both arms.
            for r in runs:
                rj = report.ROOT / "bench/results" / r.label / "run.json"
                rj.write_text(json.dumps({**json.loads(rj.read_text()), "env_hash": "e1"}))
            noise.write_text(json.dumps({**floor, "env_hash": "e1"}))
            self.assertEqual(report.load_noise(runs), {"W3": 0.001})
            noise.write_text(json.dumps({**floor, "env_hash": "0000000000000000"}))
            self.assertEqual(report.load_noise(runs), {})
            self.assertIn("environment hash", report.noise_env_mismatch(runs))

    def test_two_conformance_rows_license_no_comparison_on_the_card(self):
        """`compare.licence` refuses a pair whose arms carry different rows;
        the card took the first arm's and said "licensed" under UNLICENSED."""
        with tempfile.TemporaryDirectory() as tmp:
            other = {**CONF_T3, "hash": "other-hash"}
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], conf_a=CONF_T3, conf_b=other)
            lic = report.licence_of(runs)
            self.assertFalse(lic["comparative"])
            self.assertIn("different conformance rows", lic["headline"])
            html = report.build(runs, "t")
            self.assertNotIn("comparison is licensed", html)

    def test_a_floor_from_one_engine_bands_nothing(self):
        """A ratio's band is both arms' spread, and one arm's is not it.

        Findings 38: `noise.json` here was six passes of strawmANN, giving W3
        an RSD of 0.46%, and it banded a strawmANN-vs-Qdrant ratio on a row
        where Qdrant's own spread over twelve runs is 13.16% — 934 to 1,806
        qps. Unlike a missing `dataset`, a missing `arms` cannot mean "maybe
        both": a floor is measured over one label's repetitions.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)],
                                          [_row("W3", 2000)], sw_a=[], sw_b=[])
            noise = report.ROOT / "bench/results/noise.json"
            noise.parent.mkdir(parents=True, exist_ok=True)
            floor = {"rsd": {"W3": 0.001}, "reps": {"W3": 6}, "source": "t",
                     "discarded": []}

            # Both arms: the floor applies.
            noise.write_text(json.dumps({**floor, "arms": ["a", "b"]}))
            self.assertEqual(report.load_noise(runs), {"W3": 0.001})
            self.assertIsNone(report.noise_engine_mismatch(runs))

            # One arm: refused, and the page says whose spread it was.
            noise.write_text(json.dumps({**floor, "arms": ["a"]}))
            self.assertEqual(report.load_noise(runs), {})
            self.assertIn("measured on a", report.noise_engine_mismatch(runs))

            # Unattributed: also refused.
            noise.write_text(json.dumps(floor))
            self.assertEqual(report.load_noise(runs), {})
            self.assertIn("does not say which engine",
                          report.noise_engine_mismatch(runs))

    def test_a_native_qdrant_is_identified_by_commit_not_reported_as_unknown(self):
        """A native binary is the best-identified engine this harness runs.

        `run.json` records its path, sha256, commit and dirty flag. Until
        the provenance table ignored all of it and printed "image
        unknown, digest unknown" -- which reads as "we do not know what ran"
        for the one kind of run where it is known best. `--perf` requires the
        binary, so every perf report said so.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            runs[1].meta["qdrant"] = {
                "image": None, "digest": None, "version": "1.19.1-dev",
                "network": "native", "binary": "/opt/qdrant/target/release/qdrant",
                "binary_sha256": "c64dc861c55772dc", "commit": "ff8f3da842d0",
                "dirty": False}
            html = report.build(runs, "t")
            self.assertIn("c64dc861c55772dc", html)
            self.assertIn("ff8f3da842d0", html)
            self.assertIn("pinned by commit", html)
            self.assertNotIn("image unknown", html)
            self.assertNotIn("userland proxy", html)
            # A dirty tree is a different build, and the commit says so.
            runs[1].meta["qdrant"]["dirty"] = True
            self.assertIn("ff8f3da842d0-dirty", report.build(runs, "t"))
            # The container path is unchanged: image and digest, no commit pill.
            runs[1].meta["qdrant"] = {
                "image": "qdrant/qdrant:v1.19.0", "digest": "sha256:abc",
                "version": "1.19.0", "network": "host"}
            html = report.build(runs, "t")
            self.assertIn("qdrant/qdrant:v1.19.0", html)
            self.assertNotIn("pinned by commit", html)

    def test_a_native_qdrant_shows_its_cargo_profile(self):
        """`target/perf/qdrant` is release without LTO; the page showed only
        the path, and the Qdrant arm lost 4-13% against a fat-LTO
        binary with nothing on the page saying which profile ran."""
        import importlib
        from unittest import mock
        provenance = importlib.import_module("provenance")
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "qdrant" / "target" / "perf" / "qdrant"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"#!/bin/true\n")
            # No REST probe: a Qdrant of the user's own may be up on :6333.
            with mock.patch.object(provenance.Path, "resolve", return_value=exe), \
                 mock.patch.object(provenance, "file_digest", return_value="ab" * 32), \
                 mock.patch.object(provenance, "_run", return_value=""), \
                 mock.patch.object(provenance.urllib.request, "urlopen",
                                   side_effect=OSError("no probe")):
                got = provenance.qdrant_native_build(1)
            self.assertEqual(got["cargo_profile"], "perf")
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)])
            runs[1].meta["qdrant"] = {
                "image": None, "digest": None, "version": "1.19.1-dev",
                "network": "native", "binary": str(exe), "cargo_profile": "perf",
                "binary_sha256": "c1876e7ee060723c", "commit": "c2bf88e0c34d"}
            self.assertIn("cargo profile <code>perf</code>", report.build(runs, "t"))

    def test_a_qdrant_binary_older_than_its_checkouts_head_is_recorded_as_such(self):
        """`commit` is read from the checkout at run time, not from the binary,
        so it names whatever HEAD happens to be — and that was a
        commit dated a day *after* `target/release/qdrant` was built. `dirty`
        was true and says "this may not be what the commit describes"; it cannot
        say "this certainly is not", and only `binary_sha256` showed the file
        was byte-identical to the previous run's.

        One-way, as `workloads.stale_bfb_binary` is: older than the commit means
        certainly not from it, newer means only possibly from it, and no
        timestamp means no verdict rather than a clean one.
        """
        import importlib
        from unittest import mock
        provenance = importlib.import_module("provenance")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "qdrant"
            (repo / ".git").mkdir(parents=True)
            exe = repo / "target" / "release" / "qdrant"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"#!/bin/true\n")
            os.utime(exe, (1_757_000_000, 1_757_000_000))

            def at(commit_ts):
                return mock.patch.object(provenance, "_run",
                                         lambda cmd, **kw: str(commit_ts)
                                         if "show" in cmd else "")

            # Committed after the binary was built: impossible provenance.
            with at(1_757_100_000):
                self.assertIs(provenance.qdrant_binary_predates_head(exe), True)
            # Committed before it: the timestamps allow it, which is not proof.
            with at(1_756_900_000):
                self.assertIs(provenance.qdrant_binary_predates_head(exe), False)
            # No timestamp to be had is the absence of a check.
            with mock.patch.object(provenance, "_run", return_value=""):
                self.assertIsNone(provenance.qdrant_binary_predates_head(exe))
            # A binary in no git tree gets no verdict either.
            loose = Path(tmp) / "qdrant-loose"
            loose.write_bytes(b"#!/bin/true\n")
            self.assertIsNone(provenance.qdrant_binary_predates_head(loose))

    def test_the_verdict_tool_refuses_a_floor_from_another_corpus_or_environment(self):
        """`cmd_compare` read the global floor with no dataset or env_hash and
        printed REGRESSION (30%) from a floor measured on another corpus."""
        import json as _json

        import regression
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "bench/results").mkdir(parents=True)
            old_results, regression.RESULTS = regression.RESULTS, root / "bench/results"
            try:
                for lab in ("base", "cand"):
                    d = regression.RESULTS / lab
                    d.mkdir()
                    (d / "run.json").write_text(_json.dumps(
                        {"dataset": {"name": "dbpedia"}, "env_hash": "e1", "engine_comm": "strawmann"}))
                    (d / "rows.json").write_text(_json.dumps([{"id": "W3", "qps": 100.0, "when": "2026-09-01T00:00:00Z"}]))
                n = regression.Noise.from_dict({"rsd": {"W3": 0.1}, "reps": {"W3": 3}, "source": "s",
                                                "dataset": "sift1m", "env_hash": "e1"})
                self.assertEqual(n.dataset, "sift1m")
                why = regression.floor_refusals(n, ["base", "cand"])
                self.assertEqual(len(why), 1)
                self.assertIn("sift1m", why[0])
                same = regression.Noise.from_dict({"rsd": {"W3": 0.1}, "reps": {}, "source": "s",
                                                   "dataset": "dbpedia", "env_hash": "e2"})
                self.assertIn("environment hash", regression.floor_refusals(same, ["base", "cand"])[0])
                ok = regression.Noise.from_dict({"rsd": {"W3": 0.1}, "reps": {}, "source": "s",
                                                 "dataset": "dbpedia", "env_hash": "e1"})
                self.assertEqual(regression.floor_refusals(ok, ["base", "cand"]), [])
                # And a measured floor records both keys, so it can be refused later.
                m = regression.measure_noise([regression.RESULTS / "base", regression.RESULTS / "cand"], "s")
                self.assertEqual((m.dataset, m.env_hash), ("dbpedia", "e1"))
            finally:
                regression.RESULTS = old_results

    def test_noise_text_derives_from_noise_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(Path(tmp), [_row("W3", 4000)], [_row("W3", 2000)],
                                          sw_a=[], sw_b=[])
            self.assertEqual(report.noise_provenance(runs), {})
            noise = report.ROOT / "bench/results/noise.json"
            noise.write_text(json.dumps({
                "rsd": {"W3": 0.02}, "reps": {"W3": 6}, "source": "nf (6 repetitions)",
                "discarded": [], "harness_hash": stamp_hash(good_stamp()), "n_dirs": 6,
                "arms": ["a", "b"]}))
            prov = report.noise_provenance(runs)
            self.assertEqual((prov["passes"], prov["discarded"]), (6, 0))
            self.assertEqual(prov["stamp_note"], "")
            html = report.build(runs, "t")
            self.assertIn("6 identical passes", html)
            self.assertIn("0 discarded as contaminated", html)
            self.assertNotIn("six identical passes, one discarded", html)
            noise.write_text(json.dumps({
                "rsd": {"W3": 0.02}, "reps": {"W3": 5}, "source": "nf",
                "discarded": ["rep3: W3"], "arms": ["a", "b"],
                "harness_hash": stamp_hash(good_stamp(metric="Cosine"))}))
            prov = report.noise_provenance(runs)
            self.assertEqual((prov["passes"], prov["discarded"]), (6, 1))
            self.assertIn("different harness stamp than a and b", prov["stamp_note"])

    def test_chart_labels_use_the_tables_purpose_and_the_note_is_not_doubled(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, runs, df = self._pair(
                Path(tmp), [_row("W5", 800), _row("W10-ef32", 9000, ef=32)],
                [_row("W5", 400), _row("W10-ef32", 4500, ef=32)])
            fig_html = report.chart_throughput(runs, df)["html"]
            self.assertIn("distinct dataset queries per request", fig_html)   # §4's own words
            note = report.chart_ef_sweep(runs, df)["note"]
            self.assertEqual(note.count("visits against strawmANN's 128"), 1)

    def test_throughput_table_prints_what_compare_prints(self):
        # Same fixture, two renderings: the HTML ratio is compare's, and its
        # refusals and banners are compare's too.
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            mods = _reload(Path(tmp))
            report = importlib.reload(self.report)
            sw = [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))]
            w11 = dict(recall_joinable=False, ratio_policy="search-during-write; no recall join")
            fx.label("a", [_row("W3", 4000), _row("W4", 8000), _row("W5", 800, foreign="rustc(100%)"),
                           _row("W11", 100, **w11)], good_stamp(), CONF_T3, sw)
            fx.label("b", [_row("W3", 2000), _row("W4", 4000, stamp=good_stamp(queries=1)),
                           _row("W5", 400), _row("W11", 50, **w11)], good_stamp(), CONF_T3, sw)
            mods["compare"]._RECALL_CACHE.clear()
            runs = [report.load_run("a"), report.load_run("b")]
            html = report.throughput_table(runs, report.frame(runs))
            cmp = mods["compare"]
            rows = {r.id: r for r in cmp.joined("a", "b", cmp.load("a"), cmp.load("b"))}
            self.assertEqual(rows["W3"].ratio, "2.00x")
            self.assertIn(">2.00x", html)
            self.assertEqual(rows["W4"].ratio, "-")
            self.assertNotIn("2.00x</td>", html.split("W4", 1)[1].split("</tr>", 1)[0])
            self.assertIn("re-run under a different harness", html)
            self.assertEqual(rows["W5"].ratio, "-")
            self.assertIn("a contaminated: rustc(100%)", html)
            self.assertIn("search-during-write; no recall join", html)
            self.assertNotIn("STALE", html)
            # A stale, unlicensed pair: no ratio anywhere, both banners above the table.
            fx.label("a", [_row("W3", 4000)], good_stamp(), CONF_T2, sw)
            fx.label("b", [_row("W3", 2000)], good_stamp(metric="Cosine"), CONF_T2, sw)
            mods["compare"]._RECALL_CACHE.clear()
            runs = [report.load_run("a"), report.load_run("b")]
            html = report.throughput_table(runs, report.frame(runs))
            self.assertIn("STALE", html)
            self.assertIn("NOT A LICENSED COMPARATIVE CLAIM", html)
            self.assertNotIn("2.00x", html)


class LevelSeedNoteTests(unittest.TestCase):
    """The page names the seed its graphs were drawn at.

    §8.7 provenance. The note once claimed the seed was worth 0.00216 of
    recall@10, from a table that did not reproduce; it now says what was
    re-measured, and must not repeat the old figure.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import jinja2  # noqa: F401
            import pandas  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(
                "report.py needs the uv project (jinja2/pandas/plotly)") from e
        cls.report = importlib.import_module("report")

    def _run(self, label, builds):
        return self.report.Run(label, [], "", False, "h",
                               meta={"graph_builds": builds})

    def test_the_seed_is_named(self):
        r = self._run("sm-x", [{"nodes": 10, "unreachable": 0, "seed": "57ea3111"},
                               {"nodes": 10, "unreachable": 1, "seed": "57ea3111"}])
        note = self.report._graph_quality_note([r])
        self.assertIn("Level seed: sm-x 0x57ea3111", note)
        self.assertIn("within 0.00003", note)
        self.assertNotIn("0.00216", note)

    def test_an_engine_that_records_no_seed_is_not_described(self):
        """Qdrant draws levels from a thread-local RNG seeded by the OS, so it
        has no seed to name and the note must not invent one."""
        r = self._run("qd-x", [{"nodes": 10, "unreachable": 0}])
        note = self.report._graph_quality_note([r])
        self.assertIn("unreachable nodes", note)
        self.assertNotIn("Level seed", note)

    def test_two_seeds_in_one_arm_are_called_out(self):
        r = self._run("sm-x", [{"nodes": 10, "unreachable": 0, "seed": "1"},
                               {"nodes": 10, "unreachable": 0, "seed": "2"}])
        note = self.report._graph_quality_note([r])
        self.assertIn("2 DIFFERENT SEEDS", note)


class ReportNamingTests(unittest.TestCase):
    """One page per (dataset, labels), and no page across two datasets.

    `report.html` was one path for every combination of engine, arm and corpus,
    so the second render silently replaced the first. `recall.json` made the
    same mistake one directory down and it cost real time, which is why
    `recall_path` is keyed by (label, dataset, collection, limit).
    """

    @classmethod
    def setUpClass(cls):
        try:
            # Imported to find out whether they are installed, not to use
            # them: this suite skips rather than fails outside the uv project.
            import jinja2  # noqa: F401
            import pandas  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as e:  # the uv project only
            raise unittest.SkipTest(
                "report.py needs the uv project (jinja2/pandas/plotly)") from e
        cls.report = importlib.import_module("report")

    def _runs(self, tmp: Path, **labels_to_dataset):
        fx = Fixture(tmp)
        mods = _reload(tmp)
        report = importlib.reload(self.report)
        for label, dataset in labels_to_dataset.items():
            fx.label(label, [_row("W3", 1000)], good_stamp(), CONF_T3,
                     [("recall.sift1m.bench2.json", _sweep("bench2", "sift1m", 0.98))],
                     dataset=dataset)
        mods["compare"]._RECALL_CACHE.clear()
        return report, [report.load_run(x) for x in labels_to_dataset]

    def test_the_name_carries_the_dataset_every_label_and_the_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="dbpedia-openai-1m", b="dbpedia-openai-1m")
            self.assertEqual(report.default_out(runs).name,
                             "report-dbpedia-openai-1m-a-vs-b-2026-08-20-0900.html")
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), solo="sift1m")
            self.assertEqual(report.default_out(runs).name,
                             "report-sift1m-solo-2026-08-20-0900.html")

    def test_the_same_labels_on_two_days_are_two_files(self):
        """The reason the day is in the name at all.

        `fullrun.py` defaults to `strawmann`/`qdrant` every night, so without
        the date a week of comparisons rendered to one path and each night
        silently replaced the one before it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="sift1m", b="sift1m")
            monday = report.default_out(runs).name
            for r in runs:
                r.meta["started"] = "2026-08-27T21:40:11Z"
            self.assertNotEqual(report.default_out(runs).name, monday)
            self.assertEqual(report.default_out(runs).name,
                             "report-sift1m-a-vs-b-2026-08-27-2140.html")

    def test_the_same_labels_twice_in_one_day_are_two_files(self):
        """The reason the minute is in the name, not just the day.

        A day holds more than one run of one pair: a settled SIFT
        pair was measured over the labels an unsettled pair had used that
        morning, and the second render would have replaced the first in place.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="sift1m", b="sift1m")
            for r in runs:
                r.meta["started"] = "2026-08-29T09:12:00Z"
            morning = report.default_out(runs).name
            for r in runs:
                r.meta["started"] = "2026-08-29T16:03:29Z"
            self.assertNotEqual(report.default_out(runs).name, morning)
            self.assertEqual(morning, "report-sift1m-a-vs-b-2026-08-29-0912.html")
            self.assertEqual(report.default_out(runs).name,
                             "report-sift1m-a-vs-b-2026-08-29-1603.html")

    def test_re_rendering_one_run_keeps_its_path(self):
        """The other half: the minute is the *run's*, so a render is idempotent.

        Keying on the clock would separate two runs too, and would also spray a
        fresh copy of one run on every render -- leaving the reader to guess
        which of five identical pages is current.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="sift1m", b="sift1m")
            self.assertEqual(report.default_out(runs), report.default_out(runs))

    def test_arms_straddling_midnight_are_filed_under_the_later_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="sift1m", b="sift1m")
            runs[0].meta["started"] = "2026-08-26T23:52:00Z"
            runs[1].meta["started"] = "2026-08-27T00:31:00Z"
            self.assertEqual(report.default_out(runs).name,
                             "report-sift1m-a-vs-b-2026-08-27-0031.html")

    def test_an_unstamped_run_is_dated_by_its_rows_not_by_today(self):
        """`started` postdates some result directories, and re-renders happen.

        Either would put August's numbers under the day someone re-rendered
        them, which is the confusion the date is here to remove.
        """
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="sift1m")
            del runs[0].meta["started"]
            rows = report.ROOT / "bench/results/a/rows.json"
            when = calendar.timegm(time.strptime("2026-07-04T18:00:00Z",
                                                 "%Y-%m-%dT%H:%M:%SZ"))
            os.utime(rows, (when, when))
            self.assertEqual(report.default_out(runs).name,
                             "report-sift1m-a-2026-07-04-1800.html")

    def test_a_run_predating_the_field_is_sift1m(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, runs = self._runs(Path(tmp), a="sift1m")
            del runs[0].meta["dataset"]
            self.assertEqual(report.datasets_of(runs), ["sift1m"])

    def test_two_datasets_on_one_page_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._runs(Path(tmp), a="sift1m", b="dbpedia-openai-1m")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = report.main(["report.py", "a", "b"])
            self.assertEqual(code, 2)
            self.assertIn("sift1m", err.getvalue())
            self.assertIn("dbpedia-openai-1m", err.getvalue())
            # Refused before writing: a rendered page would be read as a
            # comparison, and its ratios would be about the corpora.
            self.assertEqual(list((Path(tmp) / "bench/results").glob("*.html")), [])

    def test_one_dataset_renders_to_the_keyed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._runs(Path(tmp), a="sift1m", b="sift1m")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(report.main(["report.py", "a", "b"]), 0)
            written = report.ROOT / "bench/results/report-sift1m-a-vs-b-2026-08-20-0900.html"
            self.assertTrue(written.exists(), out.getvalue())
            self.assertIn("<title>strawmANN benchmark on sift1m: a vs b</title>",
                          written.read_text())
            # The path is printed, because `fullrun.py` no longer computes it.
            self.assertIn(str(written), out.getvalue())


class NetworkPathTests(unittest.TestCase):
    """Which engines are told they pay Docker's proxy tax, and which are not.

    Finding 26 measured what publishing a container port costs: a userland
    process relaying every gRPC byte, charged to neither engine. The pill that
    reports it is a claim about the request path, and getting it wrong in the
    permissive direction understates one engine's numbers.
    """

    def setUp(self):
        self.rep = importlib.import_module("report")

    def test_a_native_binary_is_not_behind_a_published_port(self):
        """`--perf` requires the native binary, so this fired on every perf run.

        `start_qdrant_binary` records `network: native`. Only `host` was
        special-cased, so `native` fell through to the published-port branch
        and the report told the reader a userland proxy relayed every request
        to a process that had no container anywhere near it.
        """
        got = self.rep._network_path("native")
        self.assertIn("pill ok", got)
        self.assertNotIn("userland proxy", got)
        self.assertNotIn("published port", got)

    def test_classify_network_covers_the_open_docker_value_space(self):
        """The raw value is open; the classification is the closed set.

        `run.json` carries `native` or whatever `docker inspect` reports, and
        that is any network name someone created. What every reader needs is
        the four cases that differ, so the mapping happens once here instead of
        as a string compare at each use — which is how `native` came to be
        reported as a published port.
        """
        import importlib
        prov = importlib.import_module("provenance")
        NP = prov.NetworkPath
        for raw, want in (("native", NP.native),
                          ("host", NP.host),
                          ("bridge", NP.published),
                          ("my-custom-net", NP.published),
                          (" host ", NP.host),          # docker pads its output
                          (None, NP.unknown),
                          ("", NP.unknown),
                          ("   ", NP.unknown)):
            self.assertIs(prov.classify_network(raw), want, f"{raw!r}")

        # Anything not `native` or `host` must classify as published rather
        # than as unknown: an unrecognised network is a mapped port, and
        # reporting it as "not recorded" would drop finding 26's caveat.
        self.assertIs(prov.classify_network("weird"), NP.published)

    def test_host_networking_and_a_published_port_still_read_as_before(self):
        host = self.rep._network_path("host")
        self.assertIn("pill ok", host)
        self.assertNotIn("userland proxy", host)
        # Anything else is a published port, which is the case finding 26 is
        # about and must keep saying so.
        for mode in ("bridge", "published"):
            bad = self.rep._network_path(mode)
            self.assertIn("pill bad", bad)
            self.assertIn("userland proxy", bad)
        # Not recorded at all: still bad, because every such run published one.
        unknown = self.rep._network_path(None)
        self.assertIn("pill bad", unknown)


class InstrumentTests(unittest.TestCase):
    """The counters beside the rate: switches, stalls and hardware.

    Every test here is about a *blank*. Each of these columns has more than one
    way of having no number — not collected by this kernel, not this engine's
    cgroup, not implemented by this part, multiplexed and therefore
    extrapolated, or simply measured before the column existed — and each of
    them, printed as a zero, is a specific and wrong claim about a machine.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import jinja2  # noqa: F401
            import pandas  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(
                "report.py needs the uv project (jinja2/pandas/plotly)") from e
        cls.report = importlib.import_module("report")

    def _runs(self, *rowsets):
        return [self.report.Run(f"e{i}", list(rows), "", True, "h")
                for i, rows in enumerate(rowsets)]

    def test_voluntary_switches_are_shown_beside_involuntary(self):
        """Both, because they mean opposite things and only the pair reads.

        The voluntary count has been on every row since `procstat` grew the
        scheduler fields, and the table printed only the involuntary one — so
        the free proxy for lock contention was measured, stored and never
        displayed.
        """
        report = self.report
        rows = [_row("W3", 100.0, ctx_switches_voluntary=196246,
                     ctx_switches_involuntary=51, cpu_user_s=8.0, wall_s=8.0)]
        html = report.scheduler_rows_table(self._runs(rows))
        self.assertIn("196,246 / 51", html)
        self.assertIn("switches vol/invol", html)
        # One measured and one not is not a zero for the missing one.
        half, _ = report._switches_cell({"ctx_switches_voluntary": 7})
        self.assertEqual(half, "7 / ?")
        self.assertEqual(report._switches_cell({}), ("-", False))

    def test_coverage_is_shown_only_where_it_explains_a_dash(self):
        """The cause, printed beside the effect, and nowhere else.

        A coverage badge on every healthy row is a column of noise that hides
        the one row where the thread-summed counters were withheld.
        """
        report = self.report
        text, _ = report._threads_cell({"threads": 10.0, "sched_coverage": 0.0085})
        self.assertIn("1%", text)
        clean, _ = report._threads_cell({"threads": 10.0, "sched_coverage": 1.0})
        self.assertEqual(clean, "10")

    def test_pressure_says_which_kind_of_blank_it_is(self):
        """Four answers, and a row measured before the column existed is one.

        Reporting `unrecorded` as "this kernel has no PSI" would make a claim
        about the host out of the absence of a field.
        """
        report = self.report
        runs = self._runs([_row("W3", 1.0, psi_scope="engine")],
                          [_row("W3", 1.0, psi_scope="shared")],
                          [_row("W3", 1.0)])
        self.assertEqual(list(report.psi_scopes(runs).values()),
                         ["engine", "shared", "unrecorded"])

    def test_stall_columns_withhold_what_they_could_not_measure(self):
        report = self.report
        measured = _row("W3", 1.0, psi_scope="engine", psi_io_some_s=0.25,
                        psi_io_full_s=0.0, blkio_delay_s=0.5)
        html = report.stalls_rows_table(self._runs([measured]))
        self.assertIn("250 / 0 ms", html)
        self.assertIn("500 ms", html)
        # `task_delayacct=0` is the common case and reads as a permanent zero
        # in /proc; the harness stores None and the table must not fill it in.
        unmeasured = _row("W3", 1.0, psi_scope="shared", blkio_delay_s=None,
                          runqueue_wait_s=0.01)
        html = report.stalls_rows_table(self._runs([unmeasured]))
        self.assertNotIn("0 ms</td>", html.replace("10.0 ms", ""))
        # A run with nothing to say produces no section at all, rather than a
        # table of dashes.
        self.assertEqual(report.stalls_rows_table(self._runs([_row("W3", 1.0)])), "")

    def test_hardware_table_is_per_query_and_absent_without_counters(self):
        report = self.report
        self.assertEqual(report.hardware_rows_table(self._runs([_row("W3", 1.0)])), "")
        # A real `-p 1` search row, counters and all.
        row = _row("W3", 1.0, n_queries=50_000, perf_set="default",
                   perf_cycles=13_135_706_338.0, perf_instructions=16_849_722_545.0,
                   perf_ref_cycles=13_037_924_580.0, perf_ref_hz=1.9958e9,
                   perf_task_clock_s=7.548, perf_branch_misses=89_997_470.0,
                   perf_dtlb_walks=33_837_357.0)
        html = report.hardware_rows_table(self._runs([row]))
        self.assertIn("1.28", html)          # IPC
        self.assertIn("2.01", html)          # effective GHz
        self.assertIn("1.007x nominal", html)  # the form §7.5's guard rail uses
        # The formula this column used to have. `task-clock` is time on a core
        # and `ref-cycles` counts only while the core is unhalted, so on a row
        # where the engine sleeps between queries the first averages halted
        # time into a frequency: 1.74 GHz against a true 2.01. It agrees on a
        # CPU-bound row, which is why it survived until a latency row was
        # measured with it.
        self.assertNotIn("1.74", html)
        self.assertIn("262,714", html)       # cycles per query
        self.assertIn("676.7", html)         # TLB walks per query
        # A load row has no queries, so a per-query column is a dash rather
        # than an absolute count wearing a per-query heading.
        load = _row("W1", None, n_queries=None, perf_set="default",
                    perf_cycles=1e9, perf_instructions=2e9, perf_task_clock_s=1.0)
        html = report.hardware_rows_table(self._runs([load]))
        self.assertIn("2.00", html)          # IPC still reads
        self.assertNotIn("1,000,000,000", html)
        # No reference counter on this row, so no frequency at all — not one
        # recomputed from the CPU time sitting right there.
        self.assertNotIn("nominal", html)

    def test_per_engine_tables_name_each_engine_once(self):
        """Every column header repeated the label: seven times per engine in the
        scheduler table, five in stalls, six in hardware, two in sharing."""
        report = self.report
        stall = _row("W3", 1.0, runqueue_wait_s=0.01)
        perf = _row("W3", 1.0, n_queries=1000, perf_set="default",
                    perf_cycles=1e9, perf_instructions=2e9, perf_task_clock_s=1.0)
        sched = _row("W3", 1.0, cpu_user_s=1.0, wall_s=2.0)
        share = _row("W3", 1.0, n_queries=1000, perf_fills_local_ccx=10.0,
                     perf_fills_all=100.0)
        for table, rows, cols in (
                (report.scheduler_rows_table, [sched], 7),
                (report.stalls_rows_table, [stall], 5),
                (report.hardware_rows_table, [perf], 6),
                (report.sharing_rows_table, [share], 2)):
            html = table(self._runs(rows, rows))
            head = html.split("</thead>", 1)[0]
            for label in ("e0", "e1"):
                self.assertEqual(head.count(f">{label}<"), 1, label)
                self.assertIn(f'colspan="{cols}">{label}</th>', head)
            self.assertNotIn("e0 ", head)
            # A rule opens each engine's group, in the header and on every row.
            self.assertEqual(head.count('class="num grp"'), 2)
            body = html.split("<tbody>", 1)[1]
            self.assertEqual(body.count('class="num grp"'), 2)

    def test_arms_measured_with_different_instruments_are_named(self):
        report = self.report
        runs = self._runs([_row("W3", 1.0, perf_set="default", perf_cycles=1e9)],
                          [_row("W3", 1.0)])
        self.assertEqual(list(report.perf_arms(runs).values()), ["default", ""])

    def test_an_arm_that_produced_no_counters_is_not_reported_as_measured(self):
        """`perf_set` is the request, and a request that failed leaves it set.

        Attaching to a Qdrant container fails outright — it runs as root, the
        harness does not, and `perf_event_open` on another user's task needs
        ptrace permission that `perf_event_paranoid` does not grant at any
        value. Read from the request, both arms say `default`, the mismatch
        check sees none, and the page shows one engine's cycles beside a column
        of dashes with nothing saying the other could not be counted.
        """
        report = self.report
        runs = self._runs(
            [_row("W3", 1.0, perf_set="default", perf_cycles=1.3e10,
                  perf_instructions=1.6e10)],
            [_row("W3", 1.0, perf_set="default",
                  perf_note="the engine runs as uid 0 and this harness as uid 1000")])
        # Asked for by both; measured by one.
        self.assertEqual(list(report.perf_arms(runs).values()), ["default", ""])
        refusals = report.perf_refusals(runs)
        self.assertEqual(len(refusals), 1, refusals)
        self.assertIn("produced none", refusals[0])
        self.assertIn("uid 0", refusals[0])
        # An arm nobody asked about is not a refusal.
        quiet = self._runs([_row("W3", 1.0)], [_row("W3", 1.0)])
        self.assertEqual(report.perf_refusals(quiet), [])

    def test_cache_to_cache_fills_sum_disjoint_sources(self):
        """The three distances are addable only because their umasks are.

        `ls_dmnd_fills_from_sys` is one event select with a umask per source,
        and `remote_cache` (0x14) is `far_cache | near_cache` rather than a
        third place a line can come from. Summing it with `near_cache` counts
        the near fills twice, by an amount that depends on the topology — and
        on a single-socket part the two counters read *identically*, so the
        error is a clean doubling with nothing on its face to show it.
        """
        report = self.report
        row = _row("W3", 1.0, n_queries=50_000, perf_set="sharing",
                   perf_fills_local_ccx=18_895_479.0,
                   perf_fills_near_cache=1_118_574.0,
                   perf_fills_far_cache=0.0,
                   perf_fills_all=117_748_527.0)
        self.assertEqual(report._c2c_fills(row), 20_014_053.0)
        # A row that also carries the overlapping spelling must not have it
        # added: 400.3 per query, not 422.7.
        html = report.sharing_rows_table(self._runs([dict(row, perf_fills_remote_cache=1_118_574.0)]))
        self.assertIn("400.3", html)
        self.assertIn("17.0%", html)
        # Not asked for, not rendered.
        self.assertEqual(report.sharing_rows_table(self._runs([_row("W3", 1.0)])), "")

    def test_the_new_figures_appear_only_when_they_say_something(self):
        """A chart of one dot, or of a counter nobody measured, is not a chart.

        Every one of these returns `None` rather than an empty figure, because
        `build_figures` filters falsy entries and the template loops over what
        survives — so "no data" has to mean "no card", not a card with an empty
        axis in it.
        """
        report = self.report
        plain = self._runs([_row("W3", 100.0), _row("W10-ef32", 200.0)])
        df = report.frame(plain)
        for fn in (report.chart_query_cost, report.chart_dram_per_query,
                   report.chart_ipc, report.chart_io_pressure,
                   report.chart_memory_pressure):
            self.assertIsNone(fn(plain, df), fn.__name__)

        # Two rows with counters: cycles/query and IPC both have something to
        # say, and the per-query charts exclude the load row that has no
        # queries to divide by.
        rows = [_row("W3", 100.0, n_queries=50_000, perf_cycles=13_135_706_338.0,
                     perf_instructions=16_849_722_545.0),
                _row("W10-ef32", 200.0, n_queries=50_000, perf_cycles=1.2e10,
                     perf_instructions=1.4e10),
                _row("W2", None, n_queries=None, perf_cycles=2.5e11,
                     perf_instructions=3.7e11)]
        runs = self._runs(rows)
        df = report.frame(runs)
        cost = report.chart_query_cost(runs, df)
        self.assertIsNotNone(cost)
        self.assertEqual(cost["section"], "hardware")
        # The load row is in the IPC chart and not in the per-query one.
        self.assertNotIn("W2", cost["html"])
        self.assertIn("W2", report.chart_ipc(runs, df)["html"])

        # One engine, one stalling row: a single marker on a log axis, which
        # the table states exactly two inches below.
        one = self._runs([_row("W2", None, psi_mem_full_s=0.017)])
        self.assertIsNone(report.chart_memory_pressure(one, report.frame(one)))
        # Two rows that stalled is a comparison, so it renders.
        two = self._runs([_row("W2", None, psi_mem_full_s=0.017),
                          _row("W3", 1.0, psi_mem_full_s=0.004)])
        fig = report.chart_memory_pressure(two, report.frame(two))
        self.assertIsNotNone(fig)
        self.assertEqual(fig["section"], "stalls")

    def test_the_load_generator_is_read_from_the_rows_not_the_checkout(self):
        """§9's pin travels with the measurement, or it pins nothing.

        This read the live bfb checkout's HEAD, so the page attributed its rows
        to whatever was checked out when the *report* was rendered. On the
        dbpedia-openai-1m report that meant claiming `dev @ 14d793d` for rows
        measured under `dev @ 6f216634` — the same fork patch rewritten onto
        eight newer upstream commits, which is a different generator, which is
        what the pin exists to notice.
        """
        report = self.report

        def run(label, pin):
            meta = {"harness": {"bfb_pin": pin}} if pin else {}
            return report.Run(label, [], "", True, "h", meta=meta)

        self.assertEqual(report.bfb_commit([run("a", "dev @ abc"), run("b", "dev @ abc")]),
                         "dev @ abc")
        # Two arms driven by different generators is not one comparison.
        both = report.bfb_commit([run("a", "dev @ abc"), run("b", "dev @ def")])
        self.assertIn("MISMATCH", both)
        self.assertIn("abc", both)
        self.assertIn("def", both)
        # A run that recorded no pin falls back to a checkout and says so,
        # rather than presenting today's HEAD as the row's provenance.
        #
        # `BFB_REPO` points at a repository this test makes, because the
        # fallback's other candidates are `../bfb` and `~/Workspace/bfb`:
        # asserting against those passes on a machine that happens to have a
        # bfb checkout and fails on one that does not, which is how this test
        # passed here for weeks and went red the first time CI ran it.
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q", tmp], check=True)
            subprocess.run(["git", "-C", tmp, "-c", "user.email=t@example.invalid",
                            "-c", "user.name=t", "commit", "-q", "--allow-empty",
                            "-m", "x"], check=True)
            from unittest import mock
            with mock.patch.dict(os.environ, {"BFB_REPO": tmp}):
                said = report.bfb_commit([run("a", None)])
        self.assertIn("recorded none", said)
        self.assertIn("checkout now", said)

    def test_the_page_escapes_prose_and_leaves_markup_alone(self):
        """Autoescaping, which was off for the only template there is.

        `select_autoescape` matches the *suffix* of the template name and the
        template is `report.html.j2`, so `["html"]` matched nothing. The
        symptom was on every unlicensed page: §8 names the file to produce as
        `bench/results/<label>/conformance.json`, the browser parsed `<label>`
        as an element and rendered nothing for it, and the reader was told to
        write a path with the placeholder missing.
        """
        report = self.report
        env = report.template_env()
        self.assertTrue(env.autoescape(report.TEMPLATE),
                        "autoescaping is off for the report template")
        # Prose is escaped...
        self.assertEqual(
            env.from_string("{{ x }}").render(x="bench/results/<label>/x.json"),
            "bench/results/&lt;label&gt;/x.json")
        # ...and the HTML the report builds for itself is not, which is what
        # every `|safe` in the template is for.
        self.assertEqual(env.from_string("{{ x|safe }}").render(x="<b>hi</b>"),
                         "<b>hi</b>")
        # Figure notes carry deliberate markup (`chart_latency_distribution`
        # bolds a sentence), so their four interpolations must be `|safe` too
        # or autoescaping turns that bold into visible tag text.
        tpl = (report.HERE / "templates" / report.TEMPLATE).read_text()
        self.assertNotIn('<figcaption class="note">{{ c.note }}</figcaption>', tpl)

    def test_the_two_instruments_are_checked_against_each_other(self):
        """perf and /proc count the same quantity two ways; a gap is a finding.

        Not about the engines — about whether one of the instruments covered
        the window, which is exactly what an exited thread breaks.
        """
        report = self.report
        agree = _row("W3", 1.0, perf_ctx_switches=168_930.0,
                     ctx_switches_voluntary=168_466.0, ctx_switches_involuntary=464.0)
        self.assertEqual(report.perf_crosscheck(self._runs([agree])), [])
        disagree = _row("W3", 1.0, perf_ctx_switches=168_930.0,
                        ctx_switches_voluntary=1_000.0, ctx_switches_involuntary=0.0)
        self.assertEqual(len(report.perf_crosscheck(self._runs([disagree]))), 1)
        # Small counts differ by rounding and by the bracket's idle ends, which
        # is not worth a paragraph under every table.
        tiny = _row("W3", 1.0, perf_ctx_switches=10.0,
                    ctx_switches_voluntary=4.0, ctx_switches_involuntary=0.0)
        self.assertEqual(report.perf_crosscheck(self._runs([tiny])), [])


class CoordinatedOmissionBannerTests(unittest.TestCase):
    """The banner that discards the open-loop rows must show its evidence.

    findings 32: bfb charges each request the wait between the slot its rate
    limiter planned and the moment it actually sent, so an open-loop row's
    client latency can be the generator's rather than the engine's. The
    signature is a gap that *collapses* as offered rate rises — 1,404 ms at
    2,000/s down to 340 ms at 20,000/s.

    On the dbpedia-openai-1m run the gate was `any fall at all`,
    the measured fall was two microseconds, and the report printed "0 ms
    unaccounted at 619/s against 0 ms at 1,114/s" while telling the reader
    these rows "do not measure the engine at all". They did.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = importlib.import_module("report")

    def _run(self, label, pts):
        rows = [{"id": f"W4-sat{i}", "rps_target": rps,
                 "latency": {"client_p50_us": c, "server_p50_us": srv}}
                for i, (rps, c, srv) in enumerate(pts)]
        return self.report.Run(label, rows, "", True, "h")

    def test_a_two_microsecond_drift_is_not_the_signature(self):
        # The measured dbpedia-openai-1m ladder, verbatim.
        r = self._run("sm-dbp1m", [(619, 1029.9, 923.9), (866, 990.0, 884.9),
                                   (1114, 988.6, 884.3)])
        self.assertEqual(self.report.open_loop_is_instrument_limited([r]), "")

    def test_a_gap_far_under_the_latency_cannot_be_what_the_client_saw(self):
        """Halved, but a tenth of a millisecond under a 0.9 ms p50."""
        r = self._run("a", [(600, 1024.0, 924.0), (1200, 974.0, 924.0)])
        self.assertEqual(self.report.open_loop_is_instrument_limited([r]), "")

    def test_the_real_collapse_is_still_reported_and_quoted_readably(self):
        # findings 32's own numbers, scaled to a p50 they dominate.
        r = self._run("a", [(2000, 1504_000.0, 100_000.0),
                            (20000, 440_000.0, 100_000.0)])
        out = self.report.open_loop_is_instrument_limited([r])
        self.assertIn("not latency measurements", out)
        self.assertIn("1,404 ms unaccounted at 2,000/s", out)
        self.assertIn("340 ms at 20,000/s", out)

    def test_a_gap_that_grows_with_load_is_a_queue_and_is_left_alone(self):
        r = self._run("a", [(600, 1000.0, 900.0), (1200, 9000.0, 900.0)])
        self.assertEqual(self.report.open_loop_is_instrument_limited([r]), "")

    def test_the_evidence_never_rounds_to_zero_in_its_own_sentence(self):
        """Sub-millisecond gaps that *do* collapse still print as numbers."""
        r = self._run("a", [(600, 1200.0, 400.0), (1200, 500.0, 400.0)])
        out = self.report.open_loop_is_instrument_limited([r])
        self.assertIn("0.800 ms unaccounted", out)
        self.assertIn("0.100 ms at 1,200/s", out)
        # A bare zero -- the sentence's own evidence rounded away.
        self.assertNotIn(" 0 ms", out)


class SegmentPolicyReportTests(unittest.TestCase):
    """The report must say which Qdrant experiment it is, and never guess.

    `as-deployed` is the second axis in this project to use that word --
    `setup.Profile.as_deployed` is the scheduler left as it ships, this is
    Qdrant's segment count left as it ships. The banner prints them in one
    sentence so neither has to be read from context.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = importlib.import_module("report")

    def _run(self, label, policy=None, segments=1):
        collection = {"segments": segments}
        if policy:
            collection["segment_policy"] = policy
        return self.report.Run(label, [], "", True, "h",
                               meta={"harness": {"collection": collection}})

    def test_the_policy_is_read_from_the_stamp(self):
        self.assertEqual(self.report.segment_policy_of(
            self._run("a", "as-deployed")), "as-deployed")
        self.assertEqual(self.report.segment_policy_of(
            self._run("a", "equal-work")), "equal-work")

    def test_a_run_predating_the_flag_is_unnamed_not_guessed(self):
        """It asked for `segments: 1` and a ceiling above the corpus, which is
        exactly what `equal-work` now means -- and calling it that would be the
        report deciding what an old run intended. A segment count is not a
        name; that is why the flag exists."""
        r = self._run("a", None, segments=1)
        self.assertEqual(self.report.segment_policy_of(r), "unnamed")
        # The banner prints the number instead, so the reader is not left
        # with nothing.
        self.assertEqual(self.report.segments_requested([r]), "1")

    def test_two_policies_are_reported_as_two_experiments(self):
        runs = [self._run("a", "equal-work"), self._run("b", "as-deployed")]
        self.assertEqual(self.report.segment_policies_of(runs),
                         ["as-deployed", "equal-work"])
        self.assertTrue(len(self.report.segment_policies_of(runs)) > 1)
        # One policy on both arms is the healthy case and raises nothing.
        same = [self._run("a", "equal-work"), self._run("b", "equal-work")]
        self.assertEqual(len(self.report.segment_policies_of(same)), 1)

    def test_a_run_with_no_harness_stamp_at_all_does_not_raise(self):
        bare = self.report.Run("a", [], "", True, "h")
        self.assertEqual(self.report.segment_policy_of(bare), "unnamed")
        self.assertEqual(self.report.segments_requested([bare]), "?")


if __name__ == "__main__":
    unittest.main()


class RowConfigAndStorageTests(unittest.TestCase):
    """What the page says a row ran at, and where its end state came from."""

    @classmethod
    def setUpClass(cls):
        try:
            import jinja2  # noqa: F401
            import pandas  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(
                "report.py needs the uv project (jinja2/pandas/plotly)") from e

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.m["report"] = importlib.reload(importlib.import_module("report"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_row_config_line_states_the_client_load(self):
        """Findings 52: two rows at different `-p` share one qps column."""
        rep = self.m["report"]
        line = rep._row_config([_row("W10-ef128", 1.0, client_parallel=8,
                                     client_threads=2, client_connections=1,
                                     client_defaults="-t -c")])
        self.assertIn("client -p 8", line)
        self.assertIn("bfb default", line)

    def test_a_defaulted_parallel_is_marked_as_one(self):
        rep = self.m["report"]
        line = rep._row_config([_row("W6-ef128", 1.0, client_parallel=2,
                                     client_threads=2, client_connections=1,
                                     client_defaults="-p -t -c")])
        self.assertIn("client -p 2", line)
        self.assertIn("-p, -t, -c bfb default", line)

    def test_an_open_loop_row_does_not_claim_a_parallel(self):
        rep = self.m["report"]
        line = rep._row_config([_row("W4-sat90", 1.0, load_mode="open-loop",
                                     client_parallel=None, client_threads=16,
                                     client_connections=2, client_defaults="")])
        self.assertIn("n/a under --rps", line)
        self.assertIn("-t 16", line)

    def test_a_row_without_the_fields_says_nothing_about_the_client(self):
        """Runs measured before this existed must not grow a fabricated line."""
        rep = self.m["report"]
        self.assertNotIn("client", rep._row_config([_row("W3", 1.0)]))

    def test_storage_level_skips_the_mutating_rows_and_says_so(self):
        """Findings 53: the last row is a rewrite in progress, not a footprint."""
        rep = self.m["report"]
        rows = [_row("W13", 2000.0, storage_bytes=3_700_000_000, rss_peak_bytes=10),
                _row("W11", 200.0, storage_bytes=10_900_000_000, rss_peak_bytes=99,
                     background_pps=3300.0)]
        html = rep.storage_table([rep.Run("a", rows, "", True, "h")])
        self.assertIn(rep.procstat.human_bytes(3_700_000_000), html)
        self.assertNotIn(rep.procstat.human_bytes(10_900_000_000), html)
        self.assertIn("before W11", html)
        # The peak is still the peak, so it keeps the mutating row.
        self.assertIn(rep.procstat.human_bytes(99), html)
