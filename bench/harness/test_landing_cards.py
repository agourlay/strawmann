#!/usr/bin/env python3
"""Unit tests: the landing page's report cards."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import landing_cards as lc

#: The sentences `templates/report.html.j2` writes, cut down to what the
#: cards read.
PAGE = """<html><dl><div><dt>Dataset</dt><dd>sift1m</dd></div></dl>
<div class="banner ok" id="verdict">
    <b>Host gate passed</b> for every run here.
    Conformance passed through tier T4: the two engines return the same
    answers at equal recall, so they may be compared.
    <a href="#conditions">Run conditions</a>
  </div>
<p class="verdict"><b>At equal recall, strawmANN serves
    1.78x to
    2.14x Qdrant's throughput.</b></p>
<p class="note">Every figure is the median of 3 passes per engine</p></html>"""

UNGATED = PAGE.replace("<b>Host gate passed</b> for every run here.",
                       "<b>Development-grade. Not publishable.</b> The host gate fails")

SIFT = "report-sift1m-sm-a-0926-vs-qd-b-0926-2026-09-25-2021.html"


class ParseTests(unittest.TestCase):
    def test_a_page_reads_back_its_own_header(self):
        c = lc.parse(SIFT, PAGE)
        self.assertEqual(c.dataset, "sift1m")
        self.assertEqual(c.labels, "sm-a-0926 vs qd-b-0926")
        self.assertEqual(c.measured, "2026-09-25 20:21")
        self.assertEqual(c.tier, "T4")
        self.assertTrue(c.gated)
        self.assertEqual(c.passes, 3)
        self.assertEqual(c.verdict,
                         "At equal recall, strawmANN serves 1.78x to 2.14x Qdrant's throughput.")

    def test_a_dataset_with_hyphens_is_cut_from_the_labels(self):
        page = PAGE.replace("<dd>sift1m</dd>", "<dd>dbpedia-openai-1m</dd>")
        c = lc.parse("report-dbpedia-openai-1m-sm-x-vs-qd-y-2026-09-24-2152.html", page)
        self.assertEqual(c.dataset, "dbpedia-openai-1m")
        self.assertEqual(c.labels, "sm-x vs qd-y")

    def test_a_missing_sentence_is_a_missing_chip_not_a_crash(self):
        c = lc.parse(SIFT, "<html></html>")
        self.assertEqual((c.tier, c.passes, c.verdict, c.gated), (None, None, None, False))
        self.assertIsNone(lc.parse("notes.html", PAGE))


class RenderTests(unittest.TestCase):
    def test_newest_first_and_only_the_newest_per_dataset_is_current(self):
        old = lc.parse("report-sift1m-sm-o-vs-qd-o-2026-09-20-1000.html", PAGE)
        new = lc.parse(SIFT, PAGE)
        dbp = lc.parse("report-sift1m2-sm-d-vs-qd-d-2026-09-22-1000.html",
                       PAGE.replace("<dd>sift1m</dd>", "<dd>sift1m2</dd>"))
        out = lc.render([old, dbp, new])
        self.assertLess(out.index(new.file), out.index(dbp.file))
        self.assertLess(out.index(dbp.file), out.index(old.file))
        self.assertEqual(out.count("chip now"), 2)
        old_card = out[out.index(old.file):]
        self.assertNotIn("chip now", old_card)

    def test_an_ungated_page_says_so(self):
        out = lc.render([lc.parse(SIFT, UNGATED)])
        self.assertIn("development-grade", out)
        self.assertNotIn("gate pass", out)

    def test_page_text_is_escaped(self):
        out = lc.render([lc.parse(SIFT, PAGE)])
        self.assertIn("Qdrant&#x27;s", out)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "docs/reports").mkdir(parents=True)
        self.page = self.root / "docs/reports/index.html"
        self.page.write_text(f"<main>\n{lc.BEGIN}\nold\n{lc.END}\n</main>\n")
        (self.root / "docs/reports" / SIFT).write_text(PAGE)
        self.patch = mock.patch.multiple(lc, ROOT=self.root, REPORTS=self.root / "docs/reports",
                                         PAGE=self.page)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _run(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return lc.main(["landing_cards.py", *args])

    def test_check_fails_until_written_then_passes(self):
        self.assertEqual(self._run("--check"), 1)
        self.assertIn("old", self.page.read_text())
        self.assertEqual(self._run(), 0)
        self.assertIn(SIFT, self.page.read_text())
        self.assertEqual(self._run("--check"), 0)
        # A retired page makes the cards stale again.
        (self.root / "docs/reports" / SIFT).unlink()
        self.assertEqual(self._run("--check"), 1)

    def test_a_page_without_markers_is_an_error(self):
        self.page.write_text("<main></main>")
        self.assertEqual(self._run(), 1)

    def test_the_committed_page_is_current(self):
        """The gate step's claim, held here too so a test run catches it."""
        self.patch.stop()
        try:
            self.assertEqual(self._run("--check"), 0)
        finally:
            self.patch.start()


if __name__ == "__main__":
    unittest.main()
