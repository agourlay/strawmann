#!/usr/bin/env python3
"""Unit tests: dataset selection, the cache roots and the doctor."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from harness_fixtures import HERE, _reload

from workloads import stamp_hash


class UseDatasetTests(unittest.TestCase):
    """Choosing a corpus has to reach this process, not only the ones it spawns.

    `fullrun.py` exported `$STRAWMANN_DATASET` and every arm it spawned was
    correct, while its own `required_capacity()` still answered for SIFT1M and
    sized the server from it. Half-applied is the dangerous state: nothing
    fails, and the server is configured for a corpus it never sees.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.w = self.m["workloads"]
        self.saved = os.environ.get("STRAWMANN_DATASET")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("STRAWMANN_DATASET", None)
        else:
            os.environ["STRAWMANN_DATASET"] = self.saved
        _reload(Path(self.tmp.name))
        self.tmp.cleanup()

    def test_strawmann_storage_is_a_config_key_like_the_other_three(self):
        """`_root("strawmann_storage", ...)` read the config, and `read_config`
        refused the key, so setting it killed every tool that imports paths."""
        import importlib
        import os
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "config.toml"
            cfg.write_text(f'[paths]\nstrawmann_storage = "{td}/sm"\n')
            old = os.environ.get("STRAWMANN_CONFIG")
            os.environ["STRAWMANN_CONFIG"] = str(cfg)
            try:
                sys.modules.pop("paths", None)
                paths = importlib.import_module("paths")
                self.assertEqual(paths.STRAWMANN_STORAGE, Path(td) / "sm")
                self.assertEqual(paths.source("strawmann_storage"), "config file")
                self.assertIn("strawmann", paths.describe())
            finally:
                if old is None:
                    os.environ.pop("STRAWMANN_CONFIG", None)
                else:
                    os.environ["STRAWMANN_CONFIG"] = old
                sys.modules.pop("paths", None)

    def test_it_moves_every_value_the_corpus_decides(self):
        w = self.w
        self.assertEqual((w.DATASET, w.DIM, w.METRIC), ("sift1m", 128, "Euclid"))
        w.use_dataset("dbpedia-openai-100K-1536-angular")
        self.assertEqual(w.DATASET, "dbpedia-openai-100K-1536-angular")
        self.assertEqual(w.DIM, 1536)
        self.assertEqual(w.METRIC, "Cosine")
        # CREATE is what a collection is created from; a stale one would make
        # the new corpus's collections under the old corpus's distance.
        self.assertEqual(w.CREATE[w.CREATE.index("--distance") + 1], "Cosine")
        for row in w.table():
            if "--fbin" in row.args:
                self.assertEqual(row.args[row.args.index("-d") + 1], "1536")

    def test_it_exports_for_the_processes_this_one_spawns(self):
        self.w.use_dataset("dbpedia-openai-1m")
        self.assertEqual(os.environ["STRAWMANN_DATASET"], "dbpedia-openai-1m")


class CorpusSizeTests(unittest.TestCase):
    """No row may ask bfb for more points than the corpus holds.

    bfb is *told* how many to upload and slices the mmap unchecked, so the
    overrun is a panic rather than a short read — which is what a 100k corpus
    asked for 1,000,000 produced on the first attempt at a second dataset.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        sys.path.insert(0, str(HERE))
        self.paths = importlib.import_module("paths")
        self.saved = (self.paths.DATA, os.environ.get("STRAWMANN_DATA"))
        self.data = Path(self.tmp.name) / "data"
        self.paths.use_data_dir(str(self.data))

    def tearDown(self):
        self.paths.DATA = self.saved[0]
        if self.saved[1] is None:
            os.environ.pop("STRAWMANN_DATA", None)
        else:
            os.environ["STRAWMANN_DATA"] = self.saved[1]
        self.tmp.cleanup()

    def _corpus(self, rows: int, dim: int = 128) -> None:
        """An fbin header only: `corpus_rows` reads the first eight bytes."""
        f = self.paths.dataset("sift1m")[0]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rows.to_bytes(4, "little") + dim.to_bytes(4, "little"))

    def _loading_rows(self, w):
        return [r for r in w.table() if "--fbin" in r.args]

    def test_a_smaller_corpus_caps_every_row_that_loads_it(self):
        w = self.m["workloads"]
        self._corpus(100_000)
        self.assertEqual(w.corpus_rows(), 100_000)
        self.assertEqual(w.upload_n(), 100_000, "capped at the corpus")
        self.assertEqual(w.UPLOAD_N, 1_000_000, "the request itself is unchanged")
        self.assertEqual(w.w12_n(), 100_000)
        rows = self._loading_rows(w)
        self.assertTrue(rows)
        for r in rows:
            n = int(r.args[r.args.index("-n") + 1])
            self.assertLessEqual(n, 100_000, f"{r.id} asks for more than the corpus holds")

    def test_no_upload_row_exceeds_the_capacity_the_server_is_given(self):
        """`--capacity` is `required_capacity()`, and §3 preallocates it.

        A row asking for more is refused by the engine mid-run, which is how
        W0 failed once the other rows were capped and the capacity followed
        them: it was exempted for being synthetic, and synthetic points occupy
        exactly as much preallocated capacity as real ones.
        """
        w = self.m["workloads"]
        self._corpus(100_000)
        cap = w.required_capacity()
        for r in w.table():
            if not r.upload_only:
                continue
            n = int(r.args[r.args.index("-n") + 1])
            self.assertLessEqual(n, cap, f"{r.id} uploads more than --capacity {cap:,}")
        # And the one row that appends rather than creates ends up exactly at it.
        w11 = next(r for r in w.table() if r.id == "W11")
        off = int(w11.background[w11.background.index("--offset") + 1])
        n = int(w11.background[w11.background.index("-n") + 1])
        self.assertLessEqual(off + n, cap, "W11's append ends past the preallocation")

    def test_every_searching_row_asks_at_its_collection_s_own_width(self):
        """A generated query is sized by `-d`, and bfb defaults it to 128.

        The general rule, not "W12 has a -d": a row must ask a collection for
        neighbours at the width that collection was created with. W0 is correct
        with `-d 4` because `bench0` is created at 4; W12 was wrong with no `-d`
        at all because `bench12` is created at `DIM`. On SIFT1M the default and
        the corpus agreed, so the missing flag was invisible.
        """
        w = self.m["workloads"]
        # A real 1536-wide corpus, not just a header claiming one: `DIM` comes
        # from the descriptor, so the dataset has to actually change for this
        # test to be about anything.
        saved = os.environ.get("STRAWMANN_DATASET")
        self.addCleanup(lambda: os.environ.pop("STRAWMANN_DATASET", None)
                        if saved is None else
                        os.environ.__setitem__("STRAWMANN_DATASET", saved))
        w.use_dataset("dbpedia-openai-100K-1536-angular")
        self.assertEqual(w.DIM, 1536)
        created = {}
        for r in w.table():
            a = [str(x) for x in r.args]
            if not r.upload_only or "-d" not in a:
                continue
            created[a[a.index("--collection-name") + 1]] = a[a.index("-d") + 1]
        checked = 0
        for r in w.table():
            a = [str(x) for x in r.args]
            # Only rows whose queries bfb generates; a row reading a query file
            # takes its width from the file.
            if "--search" not in a or r.query_collection:
                continue
            coll = a[a.index("--collection-name") + 1]
            self.assertIn("-d", a, f"{r.id} generates queries with no -d; bfb defaults to 128")
            self.assertEqual(a[a.index("-d") + 1], created[coll],
                             f"{r.id} queries {coll} at a width it was not created with")
            checked += 1
        # W0 alone since 2026-09-08: the filtered rows moved to the config
        # path, where the query file defines the width, which is exactly the
        # exclusion above.
        self.assertGreaterEqual(checked, 1, "W0 at least")
        # The two that must differ, so a single shared value cannot satisfy it.
        self.assertEqual(created["bench12"], "1536")
        self.assertEqual(created["bench0"], "4")

    def test_the_floor_row_is_measured_at_the_same_scale_as_the_rest(self):
        """W0 takes the distance out; it must not also take the graph size out."""
        w = self.m["workloads"]
        self._corpus(100_000)
        w0 = next(r for r in w.table() if r.id == "W0-upload")
        self.assertEqual(int(w0.args[w0.args.index("-n") + 1]), w.upload_n())
        self.assertEqual(w0.args[w0.args.index("-d") + 1], "4", "still the d=4 floor")

    def test_the_append_offset_follows_the_cap(self):
        """W11 appends past the end of what W2 loaded, not past a stale constant.

        An offset beyond the collection leaves a gap in the id space; one below
        it overwrites points W2 uploaded, which is the bug the offset was added
        to fix in the first place.
        """
        w = self.m["workloads"]
        self._corpus(100_000)
        # Two write rows append to bench2 in sequence, so their id ranges have
        # to abut: W11-steady starts at the end of the corpus and W11 starts at
        # the end of W11-steady. A gap leaves holes in the id space; an overlap
        # turns an append into the overwrite the offset exists to prevent.
        rows = {r.id: r for r in w.table()}
        steady, w11 = rows["W11-steady"], rows["W11"]
        s_off = int(steady.background[steady.background.index("--offset") + 1])
        w_off = int(w11.background[w11.background.index("--offset") + 1])
        self.assertEqual(s_off, w.upload_n())
        self.assertEqual(w_off, w.upload_n() + w.w11_steady_n())
        self.assertEqual(w.required_capacity(), w_off + w.w11_n(),
                         "the preallocation ends exactly where the last append does")

    def test_the_write_row_keeps_its_ratio_to_the_collection(self):
        """W11's note says the ratio is the experiment; `--dataset` moves it.

        Unscaled, this row appended 200,000 points to a 100,000-point
        collection and reported `write overlap 4%` — the append finished long
        before the search did, so the row called itself mixed read/write while
        its writer was absent for 96% of it (searching the rebuild the append
        had forced, which is not the quiet collection the note of the day
        claimed). The note above `W11_N` predicted
        exactly that ("a different and far heavier workload"); it expected a
        human to turn the knob, and `--dataset` turns the collection instead.
        """
        w = self.m["workloads"]
        self._corpus(100_000)
        self.assertEqual(w.upload_n(), 100_000)
        self.assertEqual(w.w11_n(), 20_000, "one fifth, as at full scale")
        self.assertAlmostEqual(w.w11_n() / w.upload_n(), w.W11_N / w.UPLOAD_N)
        w11 = next(r for r in w.table() if r.id == "W11")
        n = int(w11.background[w11.background.index("-n") + 1])
        self.assertEqual(n, w.w11_n(), "the row sends the scaled count")
        # bench2 takes W11's append and then W11-steady's, both of them.
        self.assertEqual(w.w11_steady_n(), 5_000)
        self.assertEqual(w.required_capacity(), 125_000)

    def test_a_corpus_that_fits_is_left_alone(self):
        """SIFT1M holds exactly 1,000,000, so the cap must be a no-op for it.

        The published rows are stamped with `upload_n`, and a cap that moved it
        would re-hash every one of them without any number having changed.
        """
        w = self.m["workloads"]
        self._corpus(1_000_000)
        self.assertEqual(w.upload_n(), 1_000_000)
        self.assertEqual(w.w12_n(), w.W12_N)
        self.assertEqual(w.w11_n(), w.W11_N, "and the write row is untouched too")
        stamp = w.harness_stamp()
        self.assertEqual(stamp["upload_n"], 1_000_000)
        self.assertEqual(stamp["w11_n"], 200_000)
        self.assertEqual(w.w11_steady_n(), 50_000,
                         "5% of the corpus, under rebuild_ratio's 10%")
        self.assertEqual(w.required_capacity(), 1_250_000)

    def test_an_absent_corpus_falls_back_to_the_request(self):
        """`workloads.py list` and a bare checkout must still build the table.

        A missing corpus is the preflight's error to report; making it this
        function's would turn `list` into something that needs 16 GB fetched.
        """
        w = self.m["workloads"]
        self.assertIsNone(w.corpus_rows())
        self.assertEqual(w.upload_n(), w.UPLOAD_N)
        self.assertEqual(w.w12_n(), w.W12_N)
        self.assertTrue(self._loading_rows(w), "the table still builds")


class LabelDatasetTests(unittest.TestCase):
    """A result label belongs to one corpus, and the harness enforces it.

    The failure is silent and unrecoverable: `rows.json` merges in place, so a
    second corpus written into a label leaves a directory that describes itself
    consistently and measures two different things. `bench/results` is
    gitignored, so the rows it overwrites are simply gone.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        self.w = self.m["workloads"]
        self.d = Path(self.tmp.name) / "bench/results/strawmann"
        self.d.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_json(self, dataset):
        body = {"label": "strawmann"}
        if dataset is not None:
            body["dataset"] = {"name": dataset}
        (self.d / "run.json").write_text(json.dumps(body))

    def test_a_label_holding_another_corpus_is_refused(self):
        self._run_json("sift1m")
        why = self.w.foreign_dataset(self.d, "dbpedia-openai-100K-1536-angular")
        self.assertIsNotNone(why)
        # The message has to carry both names and a way out, because the
        # directory it names is the one the published tables are built from.
        self.assertIn("sift1m", why)
        self.assertIn("dbpedia-openai-100K-1536-angular", why)
        self.assertIn("label", why)

    def test_the_same_corpus_is_allowed(self):
        self._run_json("sift1m")
        self.assertIsNone(self.w.foreign_dataset(self.d, "sift1m"))

    def test_an_empty_or_unstamped_label_is_not_a_conflict(self):
        """Every result set predating the field reads as the corpus it measured."""
        self.assertIsNone(self.w.foreign_dataset(self.d, "sift1m"), "no run.json")
        self._run_json(None)
        self.assertIsNone(self.w.foreign_dataset(self.d, "sift1m"), "no dataset field")
        (self.d / "run.json").write_text("{not json")
        self.assertIsNone(self.w.foreign_dataset(self.d, "sift1m"), "unreadable")

    def test_the_run_refuses_rather_than_merging(self):
        """End to end through `main`, since that is what actually writes."""
        self._run_json("sift1m")
        os.environ["STRAWMANN_DATASET"] = "dbpedia-openai-100K-1536-angular"
        try:
            w = _reload(Path(self.tmp.name))["workloads"]
            rc = w.main(["workloads.py", "run", "http://localhost:1", "strawmann"])
            self.assertEqual(rc, 2, "a refusal, not a warning")
            # And nothing was written on the way to refusing.
            self.assertFalse((self.d / "rows.json").exists())
            self.assertFalse((self.d / "env.txt").exists())
            self.assertEqual(
                json.loads((self.d / "run.json").read_text())["dataset"]["name"],
                "sift1m", "the existing stamp is untouched")
        finally:
            os.environ.pop("STRAWMANN_DATASET", None)
            _reload(Path(self.tmp.name))


class DoctorVerdictTests(unittest.TestCase):
    """The doctor's last line has to name the failure it actually found.

    It is the sentence someone acts on, and a wrong one is expensive in a
    specific direction: "a dataset failed its digest" printed for a dataset
    whose digests are fine sends the reader to re-fetch 14 GB, and says nothing
    about the thing that is broken.
    """

    def setUp(self):
        sys.path.insert(0, str(HERE.parents[1] / "scripts"))
        self.doc = importlib.import_module("doctor")

    def _verdict(self, datasets, gate=True):
        d = self.doc
        sections = [d.Section("Toolchain", [d.Item("zig", d.OK)]),
                    d.Section("Build artifacts", [d.Item("strawmann", d.OK)]),
                    d.Section("Datasets (§4.2)", datasets)]
        return "\n".join(d.verdict(sections, {"gate_pass": gate, "env_hash": "h"}, False))

    def test_a_dangling_link_is_not_reported_as_a_digest_failure(self):
        d = self.doc
        out = self._verdict([d.Item("sift1m", d.OK),
                             d.Item("dbpedia-openai-1m", d.BAD, "1 bfb shard link(s) dangle",
                                    kind="links")])
        self.assertIn("dbpedia-openai-1m", out)
        self.assertNotIn("failed its digest", out)
        # And it is scoped: SIFT1M is measurable while dbpedia's links dangle.
        self.assertIn("no other dataset is affected", out)

    def test_a_digest_failure_still_says_so(self):
        d = self.doc
        out = self._verdict([d.Item("sift1m", d.BAD, "digest mismatch", kind="digest")])
        self.assertIn("failed its digest", out)

    def test_both_causes_are_reported_when_both_are_present(self):
        d = self.doc
        out = self._verdict([d.Item("sift1m", d.BAD, "digest mismatch", kind="digest"),
                             d.Item("dbpedia-openai-1m", d.BAD, "links dangle", kind="links")])
        self.assertIn("failed its digest", out)
        self.assertIn("bfb's shard links dangle", out)

    def test_a_real_failure_hides_the_routine_fetch_advice(self):
        """`fetch` is the wrong next step while something is actually broken."""
        d = self.doc
        unfetched = d.Item("gist1m", d.WARN, "not fetched, 1.0GB to download")
        self.assertIn("datasets.py fetch", self._verdict([unfetched]))
        self.assertNotIn("datasets.py fetch",
                         self._verdict([unfetched,
                                        d.Item("x", d.BAD, "digest mismatch", kind="digest")]))


class DataDirTests(unittest.TestCase):
    """`--data-dir`: one override that every tool and every child process sees.

    The failure this guards against is silent and expensive. A dataset path
    captured into a module constant at import is bound before the flag is
    parsed, so the override reaches the tool that was given it and nothing
    else: the fetcher writes 16 GB into the chosen directory and the rows are
    then measured against whatever happens to sit in the default one.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = _reload(Path(self.tmp.name))
        sys.path.insert(0, str(HERE))
        self.paths = importlib.import_module("paths")
        self.saved = (self.paths.DATA, os.environ.get("STRAWMANN_DATA"))

    def tearDown(self):
        self.paths.DATA = self.saved[0]
        if self.saved[1] is None:
            os.environ.pop("STRAWMANN_DATA", None)
        else:
            os.environ["STRAWMANN_DATA"] = self.saved[1]
        self.tmp.cleanup()

    def test_override_moves_the_root_and_is_exported(self):
        before = self.paths.DATA
        self.assertEqual(self.paths.use_data_dir(None), before, "None is a no-op")
        got = self.paths.use_data_dir("/mnt/shared/datasets")
        self.assertEqual(got, Path("/mnt/shared/datasets"))
        self.assertEqual(self.paths.DATA, Path("/mnt/shared/datasets"))
        # Exported, because `fullrun.py` passes the choice on by spawning.
        self.assertEqual(os.environ["STRAWMANN_DATA"], "/mnt/shared/datasets")

    def test_the_corpus_trio_follows_the_override(self):
        self.paths.use_data_dir("/mnt/shared/datasets")
        base, queries, gt = self.paths.dataset("sift1m")
        self.assertEqual(base, Path("/mnt/shared/datasets/sift1m/sift1m.fbin"))
        self.assertEqual(queries, Path("/mnt/shared/datasets/sift1m/sift1m_query.fbin"))
        self.assertEqual(gt.parent, Path("/mnt/shared/datasets/sift1m/gt"))

    def test_each_dataset_carries_its_own_metric_dim_and_layout(self):
        """The ground truth name is the metric's, and the metric is the descriptor's.

        The failure this guards is silent: a corpus resolved for one dataset
        and scored against a `gt` file named for another metric either does not
        exist (loud) or does (the cosine-against-euclid read that stood in the
        published table for weeks).
        """
        self.paths.use_data_dir("/mnt/shared/datasets")
        base, queries, gt = self.paths.dataset("dbpedia-openai-1m")
        self.assertEqual(base, Path("/mnt/shared/datasets/dbpedia-openai-1m/base.fbin"))
        self.assertEqual(queries, Path("/mnt/shared/datasets/dbpedia-openai-1m/queries.fbin"))
        self.assertEqual(gt.name, "dbpedia-openai-1m.cosine.k100.gt.json")
        self.assertEqual(self.paths.metric("dbpedia-openai-1m"), "cosine")
        self.assertEqual(self.paths.dim("dbpedia-openai-1m"), 1536)
        self.assertEqual(self.paths.metric("sift1m"), "euclid")
        self.assertEqual(self.paths.dim("sift1m"), 128)
        # The bundle entries resolve the same way once `convert-npy` has run
        # over them, and take their width and metric from the same descriptor.
        base, queries, gt = self.paths.dataset("dbpedia-openai-100K-1536-angular")
        root = Path("/mnt/shared/datasets/dbpedia-openai-100K-1536-angular")
        self.assertEqual((base, queries), (root / "base.fbin", root / "queries.fbin"))
        self.assertEqual(gt.name, "dbpedia-openai-100K-1536-angular.cosine.k100.gt.json")
        self.assertEqual(self.paths.dim("dbpedia-openai-100K-1536-angular"), 1536)
        # Two datasets at the same width and metric, which is exactly the pair
        # `stale_reasons` has to keep apart — the metric cannot do it here.
        self.assertEqual(self.paths.metric("dbpedia-openai-100K-1536-angular"),
                         self.paths.metric("dbpedia-openai-1m"))
        # Fetched and extracted, but its shipped neighbours are computed under
        # each query's filter, so it has no unfiltered ground truth to score
        # against. It must not be offerable as a choice while that is true.
        self.assertNotIn("laion-small-clip", self.paths.runnable())
        with self.assertRaises(SystemExit):
            self.paths.dataset("laion-small-clip")

    def test_rows_are_built_against_the_overridden_root(self):
        w = self.m["workloads"]
        self.paths.use_data_dir("/mnt/shared/datasets")
        self.assertEqual(w.corpus(), Path("/mnt/shared/datasets/sift1m/sift1m.fbin"))
        self.assertIn(str(w.corpus_queries()), w.search_config("bench5", "from-start"))
        for row in w.table():
            if "--fbin" in row.args:
                self.assertEqual(row.args[row.args.index("--fbin") + 1],
                                 str(w.corpus()), f"{row.id} names the default root")

    def test_every_corpus_row_is_uploaded_at_the_datasets_own_width(self):
        """`-d` is what bfb sends, not what it reads out of the fbin.

        So a row carrying a width the corpus does not have uploads a corpus
        misparsed at that width and indexes it without complaint. The only rows
        allowed to differ are W0's, whose whole point is d=4.
        """
        w = self.m["workloads"]
        for row in w.table():
            if "--fbin" not in row.args:
                continue
            self.assertEqual(row.args[row.args.index("-d") + 1], str(w.DIM),
                             f"{row.id} uploads {w.DATASET} at the wrong width")
        self.assertEqual(w.DIM, 128, "sift1m is the default")
        self.assertEqual(w.METRIC, "Euclid", "and its metric comes from the descriptor")

    def test_the_dataset_choice_survives_the_spawn(self):
        """`fullrun.py --dataset` exports it; `workloads.py` is a subprocess.

        Everything the dataset decides has to move together. A run that picked
        up the new corpus and kept the old width, metric or query file would
        still produce a full table of numbers, all of them wrong in a way no
        row reports.
        """
        os.environ["STRAWMANN_DATASET"] = "dbpedia-openai-1m"
        try:
            w = _reload(Path(self.tmp.name))["workloads"]
            self.paths.use_data_dir("/mnt/shared/datasets")
            self.assertEqual(w.DATASET, "dbpedia-openai-1m")
            self.assertEqual(w.DIM, 1536)
            self.assertEqual(w.METRIC, "Cosine")
            self.assertEqual(w.corpus(),
                             Path("/mnt/shared/datasets/dbpedia-openai-1m/base.fbin"))
            self.assertIn("/mnt/shared/datasets/dbpedia-openai-1m/queries.fbin",
                          w.search_config("bench5", "from-start"))
            self.assertIn("--distance", w.CREATE)
            self.assertEqual(w.CREATE[w.CREATE.index("--distance") + 1], "Cosine")
            for row in w.table():
                if "--fbin" in row.args:
                    self.assertEqual(row.args[row.args.index("-d") + 1], "1536",
                                     f"{row.id} kept the old width")
            # Recorded in the stamp, and deliberately not a stamp key: the
            # refusal lives in `stale_reasons` (see the test below), and adding
            # it here would re-hash every row already measured into "re-run
            # under a different harness" without any number having changed.
            self.assertEqual(w.harness_stamp()["dataset"], "dbpedia-openai-1m")
            self.assertNotIn("dataset", w.STAMP_KEYS)
            self.assertEqual(stamp_hash(w.harness_stamp()),
                             stamp_hash(dict(w.harness_stamp(), dataset="sift1m")),
                             "the dataset must not invalidate rows already measured")
        finally:
            os.environ.pop("STRAWMANN_DATASET", None)
            _reload(Path(self.tmp.name))

    def test_the_flag_is_consumed_and_applied(self):
        w = self.m["workloads"]
        argv = ["workloads.py", "--data-dir", "/mnt/shared/datasets", "list"]
        self.assertEqual(w.main(argv), 0)
        self.assertEqual(self.paths.DATA, Path("/mnt/shared/datasets"))
        # The flag is understood before the subcommand, not left to be
        # rejected as unknown. It used to be spliced out of `argv` in place and
        # this asserted the splice; argparse consumes it by construction, so
        # what is worth asserting is that `list` accepted it at all — which the
        # exit code above already says.
        self.assertEqual(w.main(["workloads.py", "run", "uri", "lbl", "--data-dir"]), 2,
                         "a flag that takes a path must refuse to be given none")


class ConfigFileTests(unittest.TestCase):
    """`~/.config/strawmann/config.toml`: a root that sticks, still overridable.

    The four sources have to stay in that order — flag, environment, file,
    default — because each is a different scope: this run, this shell, this
    user, always. A file that quietly won over `--data-dir` would make a
    one-off run measure the wrong corpus, and one that quietly lost would make
    writing it down pointless.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = self.dir / "config.toml"
        self.saved = {k: os.environ.get(k)
                      for k in ("STRAWMANN_CONFIG", "STRAWMANN_DATA", "STRAWMANN_CACHE")}
        for k in self.saved:
            os.environ.pop(k, None)
        os.environ["STRAWMANN_CONFIG"] = str(self.cfg)
        sys.path.insert(0, str(HERE))
        self.paths = importlib.import_module("paths")

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(self.paths)   # back to this machine's real config
        self.tmp.cleanup()

    def _reload(self, body: str | None = None):
        if body is not None:
            self.cfg.write_text(body)
        return importlib.reload(self.paths)

    def test_the_file_moves_a_root_and_says_so(self):
        m = self._reload('[paths]\ndatasets = "/mnt/big/datasets"\n')
        self.assertEqual(m.DATA, Path("/mnt/big/datasets"))
        self.assertEqual(m.source("datasets"), "config file")
        # An absent key still falls back to the default layout.
        self.assertEqual(m.QDRANT_STORAGE, m.CACHE / "qdrant-storage")
        self.assertEqual(m.source("qdrant_storage"), "default")

    def test_a_config_cache_root_carries_the_datasets_dir_with_it(self):
        m = self._reload('[paths]\ncache = "/mnt/big/strawmann"\n')
        self.assertEqual(m.DATA, Path("/mnt/big/strawmann/datasets"))
        self.assertEqual(m.QDRANT_STORAGE, Path("/mnt/big/strawmann/qdrant-storage"))

    def test_environment_beats_the_file_and_the_flag_beats_both(self):
        os.environ["STRAWMANN_DATA"] = "/from/env"
        m = self._reload('[paths]\ndatasets = "/from/file"\n')
        self.assertEqual(m.DATA, Path("/from/env"))
        self.assertEqual(m.source("datasets"), "$STRAWMANN_DATA")
        m.use_data_dir("/from/flag")
        self.assertEqual(m.DATA, Path("/from/flag"))
        self.assertEqual(m.source("datasets"), "--data-dir")

    def test_a_home_relative_path_is_expanded(self):
        m = self._reload('[paths]\ndatasets = "~/Documents/datasets"\n')
        self.assertEqual(m.DATA, Path.home() / "Documents/datasets")

    def test_a_broken_file_is_refused_rather_than_ignored(self):
        # Falling back to the default root would look exactly like "nothing is
        # fetched" while the data sat where the file said.
        for body, why in (
            ('[paths]\ndatasets = ~/oops\n', "unquoted value"),
            ('[paths]\ndatsets = "/typo"\n', "misspelt key"),
            ('datasets = "/mnt/x"\n', "key outside [paths]"),
            ('paths = "not a table"\n', "[paths] is not a table"),
        ):
            with self.subTest(why=why), self.assertRaises(SystemExit) as e:
                self._reload(body)
            self.assertIn(str(self.cfg), str(e.exception), why)

    def test_no_file_is_not_an_error(self):
        m = self._reload()
        self.assertFalse(self.cfg.exists())
        self.assertEqual(m.CONFIG, {})
        self.assertEqual(m.source("datasets"), "default")

    def test_the_fetcher_resolves_the_same_root_as_the_harness(self):
        """`datasets.py` is stdlib-only and keeps its own copy of this rule.

        Two copies of one contract is the arrangement that lets the fetcher run
        on a fresh clone with no `bench/` on the path. It is also the
        arrangement that drifts, so the copies are checked against each other
        rather than trusted.
        """
        spec = importlib.util.spec_from_file_location(
            "strawmann_datasets_cli",
            HERE.parents[1] / "conformance/datasets/datasets.py")
        cli = importlib.util.module_from_spec(spec)
        # Registered before execution: its frozen dataclass resolves its own
        # module out of `sys.modules` while being defined.
        sys.modules[spec.name] = cli
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(cli)

        cases = [
            ({}, {}),
            ({}, {"datasets": "/from/file"}),
            ({}, {"cache": "/from/file/cache"}),
            ({"STRAWMANN_DATA": "/from/env"}, {"datasets": "/from/file"}),
            ({"STRAWMANN_CACHE": "/env/cache"}, {"cache": "/file/cache"}),
            ({"XDG_CACHE_HOME": "/xdg"}, {}),
        ]
        for env, cfg in cases:
            with self.subTest(env=env, cfg=cfg):
                for k in ("STRAWMANN_DATA", "STRAWMANN_CACHE", "XDG_CACHE_HOME"):
                    os.environ.pop(k, None)
                os.environ.update(env)
                self.cfg.write_text("[paths]\n" + "".join(
                    f'{k} = "{v}"\n' for k, v in cfg.items()))
                harness = importlib.reload(self.paths).DATA
                fetcher = cli.default_root(dict(os.environ), cfg)
                self.assertEqual(fetcher, harness)
        for k in ("XDG_CACHE_HOME", "STRAWMANN_DATA", "STRAWMANN_CACHE"):
            os.environ.pop(k, None)


class DoctorDatasetsTests(unittest.TestCase):
    """`doctor.py` reads `datasets.py list --json`. That contract is a test.

    It used to scrape columns out of the human table, where a formatting change
    was a silent parse failure: the section simply listed nothing and the report
    looked fine. Both halves are exercised here against a temporary root, so a
    key renamed on one side fails on the other.
    """

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "strawmann_doctor", HERE.parents[1] / "scripts/doctor.py")
        cls.spec, cls.doctor = spec, importlib.util.module_from_spec(spec)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved = {k: os.environ.get(k) for k in ("STRAWMANN_DATA", "STRAWMANN_CONFIG")}
        os.environ["STRAWMANN_DATA"] = str(self.root)
        # A config file that does not exist, so this machine's own cannot
        # decide what the test measures.
        os.environ["STRAWMANN_CONFIG"] = str(self.root / "absent.toml")
        sys.modules[self.spec.name] = self.doctor
        self.spec.loader.exec_module(self.doctor)

    def tearDown(self):
        sys.modules.pop(self.spec.name, None)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def _items(self):
        sec = self.doctor.section_datasets(quick=True)
        return {i.name: i for i in sec.items}

    def test_an_empty_root_costs_what_the_descriptor_says(self):
        items = self._items()
        self.assertEqual(items["sift1m"].status, self.doctor.WARN)
        self.assertIn("not fetched", items["sift1m"].detail)
        self.assertIn("160.5MB to download", items["sift1m"].detail)
        self.assertIn("declared", items["gist1m"].detail, "no files pinned, nothing to size")
        self.assertIn("0B", items["total"].detail)
        self.assertIn(str(self.root), items["total"].detail)
        self.assertIn("left to fetch", items["total"].detail)

    def test_a_present_dataset_is_measured_on_disk(self):
        # The declared size, as a sparse file: `--fast` checks size, not content,
        # and `on_disk` reads `st_size`, so this costs no disk and no seconds.
        d = self.root / "sift1m"
        d.mkdir()
        with (d / "sift.tar.gz").open("wb") as fh:
            fh.truncate(168280445)
        # Something we converted rather than downloaded, which is the reason to
        # measure the directory instead of adding up the descriptor.
        (d / "sift").mkdir()
        with (d / "sift" / "sift_base.fvecs").open("wb") as fh:
            fh.truncate(516000000)

        items = self._items()
        self.assertEqual(items["sift1m"].status, self.doctor.OK)
        self.assertIn("652.6MB", items["sift1m"].detail, "archive plus what came out of it")
        self.assertIn("on disk", items["sift1m"].detail)
        # `--quick` skips digests, so the files are present, not verified.
        self.assertIn("present", items["sift1m"].detail)
        self.assertNotIn("verified", items["sift1m"].detail)
        self.assertIn("652.6MB", items["total"].detail)


if __name__ == "__main__":
    unittest.main()
