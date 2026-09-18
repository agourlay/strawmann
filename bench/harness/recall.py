#!/usr/bin/env python3
"""§4.1's other half: recall for every collection a search row measures.

bfb generates load and knows nothing about relevance; the conformance binary
measures relevance and its request rate is a smoke test rather than a
benchmark. They meet on `ef`, and the join is only sound when both halves
describe the *same collection* and the *same queries*.

Names lining up is not the whole of it: `corpus_disagreement` reads the
violation counters the binary already computes, so a sweep that scored a
collection against ground truth from a *different* corpus is refused rather
than joined. Nothing here read them until 2026-09-01.

That was true for exactly one row before this script existed. The fp32 sweep
covered `bench2`, so W3/W4/W5/W10 could be joined; W6, W7 and W8 search
quantized collections with no sweep at all, so their rows carried a qps and no
recall. §7.4 calls that number meaningless, and the §8 sink refuses it, which
is the machinery working correctly against an incomplete measurement rather
than a wrong one.

Sweeping a quantized collection needs no new engine code: the collections bfb
built are already there, with the same points at the same ids, so `relevance
--skip-upload` scores them against the same fp64 ground truth. What comes back
is the recall the encoding actually delivers, per `ef`, next to the throughput
it delivers it at.

A sweep speaks for a row only if it searched the way the row did. W7 sends
`--quantization-oversampling 4 --quantization-rescore true`; a sweep of
`bench7` that sent neither measured a different search, and the join used to
accept it. The sweep now sends each collection's own quantization parameters
(from §4's table, or `--oversampling`/`--rescore`), records them in the JSON,
and the readers join only where they match the row's.

Usage:
    recall.py <label> [--engine http://localhost:6334] [--collections bench2,bench6]
    recall.py <label> --collections bench7 --oversampling 4 --rescore true
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))
CONF = ROOT / "conformance"

#: The collections §4's search rows actually query, and what each one is.
#: `bench2` is the fp32 collection W3/W4/W5/W9/W10 use; the rest are §6.7's
#: encodings, built by their own upload rows.
COLLECTIONS = {
    "bench2": "fp32",
    "bench6": "SQ8 scalar quantization",
    "bench7": "binary quantization",
    "bench8": "product quantization x16",
    # Swept once per grade rather than once: W12's rows differ only in
    # selectivity (`FILTER_GRADES`), and each grade needs its own restricted
    # ground truth. Absent from this map until 2026-09-08, which is why W12
    # was the row §8 refused for having a qps and no recall.
    "bench12": "fp32 with keyword payloads (filtered)",
}

#: The `ef` values W10 sweeps, plus the 128 every other search row pins. A
#: sweep speaks for a row only at an `ef` the row searched, so this list is the
#: benchmark's definition and not a preference. `--ef` overrides it for
#: exploration — asking "what recall can this engine reach at all", which W10
#: does not answer because it stops where its throughput rows stop.
EF = [32, 64, 128, 256, 512]

#: `bench12`'s payload field, and the values each W12 grade's sweep fixes.
#:
#: Deterministic and lowest-numbered, so the condition a published recall was
#: measured at is the same on every run and legible in the file name. The row
#: itself draws its keyword *per query* — bfb's config path has no way to fix
#: one — so the sweep and the row agree on the selectivity *grade* rather than
#: on the condition. `n_matching` in the sweep records what this particular
#: condition selected, which is what keeps the grade a measurement rather than
#: an assumption.
FILTER_FIELD = os.environ.get("FILTER_FIELD", "a")
FILTER_GRADES: dict[str, list[str]] = {
    "sel1": ["keyword_0"],
    "sel10": [f"keyword_{i}" for i in range(10)],
}

#: The collection the filtered grades search, and the rows they speak for.
FILTERED_COLLECTION = "bench12"

#: The dataset every sweep here scores against, and the metric its ground truth
#: was computed under. `headline.py` sweeps another one and writes it under its
#: own name; see `recall_path`.
#:
#: From the environment, so `fullrun.py --dataset` reaches this the same way
#: `--data-dir` does — and the metric is derived from it rather than written
#: out, because a sweep sent the wrong `--metric` scores a correct collection
#: against truth from a different geometry and still returns a plausible number.
DATASET = os.environ.get("STRAWMANN_DATASET", "sift1m")
METRIC = paths.metric(DATASET)

#: Held-out queries the sweep scores per `ef`.
#:
#: 2,000 until 2026-08-22, which is not enough to read the top of the frontier.
#: recall@10 is a proportion, and at 2,000 queries its 95% interval is about
#: ±0.003 wide — wider than the whole distance between the two engines above
#: `ef` 256. The sift1m sweep put strawmANN at 0.9965 and Qdrant at
#: 0.9994 there, a gap of 0.0029 inside an interval of ±0.003, and the
#: matched-recall table turned that unresolved difference into a printed
#: `0.96x` — the low end of the report's headline range, and an artefact of the
#: query count rather than a property of either engine. Every query the dataset
#: ships takes the interval to about ±0.0013 on sift1m, which resolves it.
#:
#: Costs seconds: the sweep is one client at `QueryBatch` 32, and its own
#: `smoke_qps` is thousands per second.
RECALL_QUERIES = int(os.environ.get("RECALL_QUERIES", 0)) or paths.n_queries(
    DATASET)


#: The `limit` every sweep sends unless told otherwise, and the one the W10
#: latency rows are joined at. `recall_path` leaves this one unsuffixed so the
#: files every existing reader opens keep their names.
DEFAULT_LIMIT = 10


def _params_suffix(oversampling: float | None, rescore: bool | None) -> str:
    """The quantization parameters, as a file-name component.

    Empty when neither was sent, so a sweep that asked for nothing keeps the
    name every existing reader opens.
    """
    parts = []
    if oversampling is not None:
        parts.append(f"ov{oversampling:g}")
    if rescore is not None:
        parts.append("rs1" if rescore else "rs0")
    return ("." + ".".join(parts)) if parts else ""


def recall_path(label: str, dataset: str, collection: str,
                limit: int = DEFAULT_LIMIT,
                oversampling: float | None = None,
                rescore: bool | None = None,
                grade: str | None = None) -> Path:
    """Where a sweep of `collection` on `dataset` at `limit` lives for `label`.

    Keyed by all five, because `load_recall_json` joins on all five: a sweep
    whose parameters differ from the row's is skipped. Any key left out of the
    path lets two sweeps collide, and the survivor is joined onto rows it did
    not measure.

    `limit` is not cosmetic — the engines search at `max(ef, limit)` (Qdrant's
    `graph_layers.rs`, our `handlers.effectiveEf`), so a sweep at `limit 100`
    and nominal `ef=32` searched at 100 and its `recall_at_10` is recall at
    ef=100. A standard `fullrun` never collides, since §4 gives each collection
    one parameter set; repeating a sweep at a second setting does.
    """
    suffix = "" if limit == DEFAULT_LIMIT else f".k{limit}"
    suffix += _params_suffix(oversampling, rescore)
    # The condition too, for the reason every other key is here: two grades
    # sweep one collection, and a name that omits which would let the second
    # overwrite the first and be joined onto the rows of both.
    if grade:
        suffix += f".{grade}"
    return ROOT / "bench/results" / label / f"recall.{dataset}.{collection}{suffix}.json"


def corpus_disagreement(sweep_json: dict) -> str | None:
    """Why this sweep's ground truth and the collection it scored disagree.

    The join this module exists to make sound pairs recall measured by
    `relevance --skip-upload` with throughput measured by bfb, on the strength
    of both halves describing the same collection and the same queries. `stamp`
    checks the *names* line up. These two counters check the *contents* do, and
    the binary has always computed them:

    - `impossible_scores` — the returned point's score, recomputed in fp64
      from the base vectors by `Rescorer` rather than taken from the engine,
      beats the k-th truth while the truth does not hold its id. The
      exhaustive scan would have listed it. `judge_at` calls this "a
      ground-truth defect rather than an engine one".
    - `unknown_ids` — the returned id is not a row of the base file at all
      (§4.3 makes ids row indices). The loudest form of the same fault.

    Either one non-zero means the collection was built from a different corpus
    than the ground truth describes, or the ground truth is stale. Recall
    joined from such a sweep is a number about two different datasets. Both
    are already counted as misses, so recall is understated rather than
    flattered — but understated by an unknown amount, which is not a result.

    Returns None when the sweep is clean, which is every sweep that has not
    gone wrong.
    """
    impossible = 0
    unknown = 0
    for pt in sweep_json.get("points", []):
        if not isinstance(pt, dict):
            continue
        impossible += int(pt.get("impossible_scores") or 0)
        unknown += int(pt.get("unknown_ids") or 0)
    parts = []
    if impossible:
        parts.append(f"impossible_scores={impossible} (points the fp64 oracle "
                     f"scores better than the k-th truth, absent from it)")
    if unknown:
        parts.append(f"unknown_ids={unknown} (returned ids that are not rows "
                     f"of the base file)")
    return "; ".join(parts) if parts else None


def stamp(path: Path, dataset: str, collection: str, metric: str,
          oversampling: float | None = None, rescore: bool | None = None,
          limit: int = DEFAULT_LIMIT, grade: str | None = None) -> bool:
    """Record inside the JSON what the sweep was of, and refuse a mismatch.

    The conformance binary writes `collection`, `metric` and `base_checksum`;
    `dataset` is added here so a reader can check the join key against the
    row it is joining to instead of trusting the file name. If the binary's
    own record of the collection disagrees with what was asked for, the file
    is renamed aside rather than left where a reader would join it.
    """
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    got = d.get("collection")
    if got is not None and got != collection:
        aside = path.with_suffix(".mismatch.json")
        path.rename(aside)
        print(f"  !! sweep says collection {got!r}, asked for {collection!r}; "
              f"moved to {aside.name} so nothing joins it", file=sys.stderr)
        return False
    bad = corpus_disagreement(d)
    if bad is not None:
        aside = path.with_suffix(".mismatch.json")
        path.rename(aside)
        print(f"  !! sweep of {collection} and its ground truth are not the "
              f"same corpus: {bad}; moved to {aside.name} so nothing joins it",
              file=sys.stderr)
        return False
    d.setdefault("collection", collection)
    d["dataset"] = dataset
    d.setdefault("metric", metric)
    # The quantization search parameters this sweep sent, part of the join
    # key beside dataset and collection. None means "none sent".
    d["oversampling"] = oversampling
    d["rescore"] = rescore
    # The binary records `limit` itself; setting it here too means a reader
    # that opens the file by name still learns the width the search really ran
    # at without having to know the naming convention.
    d.setdefault("limit", limit)
    # Which selectivity grade this is, beside the `condition` the binary
    # records. The condition is the exact expression; the grade is the name the
    # row carries, and a reader joins on the second.
    d["grade"] = grade
    path.write_text(json.dumps(d, indent=2) + "\n")
    return True


def load_recall_json(label: str, dataset: str, collection: str,
                     oversampling: float | None = None,
                     rescore: bool | None = None,
                     grade: str | None = None) -> dict:
    """The sweep file for (`dataset`, `collection`), whole, or `{}`.

    Only a file whose own `dataset` and `collection` fields say what the
    caller asked for, and whose contents do not contradict them
    (`corpus_disagreement`). The legacy names (`recall.json`,
    `recall.<coll>.json`) are read as well, on the same condition: the file name is where to look
    and the fields inside are the join key, so a `recall.json` from a sweep of
    the differ's own `relevance` collection does not speak for `bench2`, and a
    dbpedia cosine sweep under the old name does not speak for SIFT.
    """
    d = ROOT / "bench/results" / label
    # The parameterised name first, then the names sweeps used before it
    # existed. Every candidate is checked against its own fields below, so a
    # legacy file only answers for the parameters it actually recorded.
    if grade is not None:
        # Only the graded name. The legacy names predate conditions entirely,
        # and an unfiltered sweep of `bench12` answers a different question
        # from a filtered row — the whole point of W12 point 3.
        candidates = [recall_path(label, dataset, collection, oversampling=oversampling,
                                  rescore=rescore, grade=grade)]
    else:
        candidates = [recall_path(label, dataset, collection,
                                  oversampling=oversampling, rescore=rescore),
                      recall_path(label, dataset, collection),
                      d / (f"recall.{collection}.json" if collection != "bench2"
                           else "recall.json")]
    for path in candidates:
        if not path.exists():
            continue
        try:
            sweep_json = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(sweep_json, dict):
            continue
        if sweep_json.get("collection") != collection:
            continue
        if sweep_json.get("dataset", dataset) != dataset:
            continue
        # The grade too, in both directions: a sweep of another condition, or
        # an unfiltered one, does not speak for a filtered row and a filtered
        # sweep does not speak for an unfiltered one.
        if sweep_json.get("grade") != grade:
            continue
        # A sweep that sent other quantization parameters than the row did
        # (or none, where the row sent some) measured a different search.
        got = sweep_quant_params(sweep_json)
        if got is None or got != (oversampling, rescore):
            continue
        # `stamp` moves such a file aside at sweep time, so this catches the
        # ones it never saw: the legacy names among the candidates above, and
        # any file written before the check existed.
        if corpus_disagreement(sweep_json) is not None:
            continue
        return sweep_json
    return {}


class QuantParams(NamedTuple):
    """The two knobs that decide whether a sweep speaks for a search row.

    §6.7's join key: a sweep of `bench7` that sent no oversampling measured a
    different search than the row that sent `--quantization-oversampling 4`,
    and the join used to accept it. Named because the pair travelled unlabelled
    through four functions, and `(None, None)` says nothing about which knob is
    which.
    """

    oversampling: float | None = None
    rescore: bool | None = None


def sweep_quant_params(sweep_json: dict) -> QuantParams | None:
    """The quantization search parameters a sweep file says it sent.

    The binary writes its own `quantization_oversampling` /
    `quantization_rescore`; `stamp` adds `oversampling` / `rescore` from what
    the harness asked for. The binary's word is what was sent, so it wins,
    the stamp is the fallback for a file written before the binary recorded
    them, and a file where the two disagree is refused (None): a sweep whose
    own record contradicts its stamp speaks for no row.
    """
    own = ("quantization_oversampling" in sweep_json or "quantization_rescore" in sweep_json)
    stamped = ("oversampling" in sweep_json or "rescore" in sweep_json)
    if own:
        got = QuantParams(sweep_json.get("quantization_oversampling"),
                          sweep_json.get("quantization_rescore"))
        if stamped and got != (sweep_json.get("oversampling"), sweep_json.get("rescore")):
            return None
        return got
    return QuantParams(sweep_json.get("oversampling"), sweep_json.get("rescore"))


def load_recall(label: str, dataset: str, collection: str,
                oversampling: float | None = None,
                rescore: bool | None = None,
                grade: str | None = None) -> dict[int, dict]:
    """The sweep for (`dataset`, `collection`, quantization params, `grade`),
    keyed by `ef`, or `{}`."""
    out = {}
    for pt in load_recall_json(label, dataset, collection, oversampling,
                               rescore, grade).get("points", []):
        if pt.get("ef") is not None and not pt.get("exact"):
            out[int(pt["ef"])] = pt
    return out


def quant_params_of(collection: str) -> QuantParams:
    """The quantization parameters §4's search row for `collection` sends."""
    try:
        import workloads
    except ImportError:  # pragma: no cover
        return QuantParams()
    for w in workloads.table():
        if w.upload_only or workloads.collection_of(w) != collection:
            continue
        q = workloads.quant_of(w)
        return QuantParams(q["quantization_oversampling"], q["quantization_rescore"])
    return QuantParams()


class EfSplit(NamedTuple):
    """The `ef` values that mean what they say at this `limit`, and the rest.

    Two lists of the same type returned in a fixed order; naming them is the
    difference between sweeping the usable values and sweeping the skipped
    ones."""

    usable: list[int]
    skipped: list[int]


def usable_ef(ef: list[int], limit: int) -> EfSplit:
    """`ef` values that mean what they say at `limit`, and the ones that don't.

    Both engines resolve a query's traversal width as `max(ef, limit)` —
    Qdrant in `graph_layers.rs`, strawmANN in `handlers.effectiveEf`, written
    against that source. So at `limit 100` a nominal `ef=32` searches at 100,
    and a sweep that recorded it as `ef=32` would be publishing a point that
    was never measured: same width as ef=64, same width as ef=100, three rows
    of one measurement wearing three labels.

    Dropped rather than silently relabelled, because a recall curve whose low
    end is flat is exactly what an engine with a broken `ef` would also look
    like, and the two must not be indistinguishable.
    """
    return EfSplit(usable=[e for e in ef if e >= limit],
                   skipped=[e for e in ef if e < limit])


def grade_of(wid: str) -> str | None:
    """The selectivity grade a row id names, or `None` for an unfiltered row.

    `W12-sel1` searches under `FILTER_GRADES["sel1"]`, and its sweep is written
    to a file named by the same grade — so the row id *is* the join key, and no
    table mapping rows to conditions has to be kept in step by hand.
    """
    _, _, suffix = wid.partition("-")
    # An `ef` ladder's rows carry the grade *and* the width — `W12-sel10-ef128`
    # is the sel10 condition at ef=128 — and both are join keys: the grade
    # picks which sweep, the `ef` picks the point in it. Without stripping the
    # width every ladder row read as unfiltered, found no sweep, and would have
    # been refused for the recall it does have.
    suffix = re.sub(r"-ef\d+$", "", suffix)
    return suffix if suffix in FILTER_GRADES else None


def filtered_truth_path(label: str, dataset: str, collection: str, grade: str) -> Path:
    """Where this run's restricted ground truth for one grade lives.

    Under the *label*, not beside the dataset's shipped truth, because it is
    not cacheable across runs: bfb assigns keyword payloads from an unseeded
    RNG at every upload, so which ids satisfy `a in {keyword_0}` is a property
    of this run's `bench12` and no other. A file cached by (dataset, metric, k,
    condition) alone would be read back next run against a collection whose
    keywords had been redrawn.
    """
    return (ROOT / "bench/results" / label
            / f"gt.{dataset}.{collection}.{grade}.json")


def build_filtered_truth(engine: str, label: str, collection: str, grade: str,
                         values: list[str], base_n: int,
                         limit: int = DEFAULT_LIMIT) -> Path | None:
    """fp64 truth over the points this run's `collection` matches for `grade`.

    docs/workloads.md W12 point 3. Asks the engine which ids satisfy the
    condition — the only place that knows — then scores the corpus prefix the
    collection was built from.
    """
    out = filtered_truth_path(label, DATASET, collection, grade)
    out.parent.mkdir(parents=True, exist_ok=True)
    base, queries_file, _ = paths.dataset(DATASET)
    cmd = [
        "cargo", "run", "--release", "--quiet", "--",
        "filtered-truth",
        "--engine", engine,
        "--collection", collection,
        "--field", FILTER_FIELD,
        "--values", ",".join(values),
        "--base", str(base),
        "--queries", str(queries_file),
        "--metric", METRIC,
        "--k", str(max(limit, 100)),
        # The collection holds a *prefix* of the corpus, so the truth must be
        # over that prefix: scored against all 1M, every neighbour outside the
        # first 200k is a point the collection never held.
        "--limit", str(base_n),
        # No query cap: `relevance` checks the truth's `query_checksum` against
        # the whole query file and caps afterwards, so a truth over a truncated
        # query set is refused by §4.3 however right it otherwise is.
        "--out", str(out),
    ]
    print(f"\n=== {collection} filtered truth, {grade}: "
          f"{FILTER_FIELD} in {{{','.join(values)}}} ===", flush=True)
    r = subprocess.run(cmd, cwd=CONF)
    if r.returncode != 0:
        print(f"  !! could not build the {grade} ground truth; its rows will "
              f"carry no recall", file=sys.stderr)
        return None
    return out


def sweep(engine: str, label: str, collection: str, queries: int,
          oversampling: float | None = None, rescore: bool | None = None,
          ef: list[int] | None = None, limit: int = DEFAULT_LIMIT,
          grade: str | None = None, base_n: int | None = None) -> int:
    """One collection's recall curve, written where the report will find it.

    `--skip-upload` is what makes this measure the collection the benchmark
    measured, rather than a fresh one built with different parameters. The
    ground truth is keyed to the corpus checksum, so a collection built from a
    different file fails the assertion inside the binary rather than producing
    a plausible curve.
    """
    want = list(ef or EF)
    want, dropped = usable_ef(want, limit)
    if dropped:
        print(f"  ef {', '.join(str(e) for e in dropped)} dropped: at limit "
              f"{limit} the engines search at max(ef, limit), so these would "
              f"be recorded at a width they did not run", file=sys.stderr)
    if not want:
        print(f"  !! no ef >= limit {limit}; nothing to sweep", file=sys.stderr)
        return 1
    ef = want
    out = recall_path(label, DATASET, collection, limit,
                      oversampling=oversampling, rescore=rescore, grade=grade)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Read here rather than at import: `--data-dir` moves the root, and a
    # constant bound at import would still name the old one.
    base, queries_file, gt = paths.dataset(DATASET)
    # A filtered grade scores against a truth restricted to its own condition,
    # built from this run's collection. `relevance` refuses the mismatch in
    # both directions, so a mistake here fails loudly rather than producing a
    # recall against the wrong question.
    values: list[str] = []
    if grade is not None:
        if not base_n:
            print(f"  !! {collection} {grade}: no --filtered-base-n, so the "
                  f"restricted truth would cover the whole corpus while the "
                  f"collection holds a prefix of it — every neighbour outside "
                  f"that prefix would read as a miss. Skipping.", file=sys.stderr)
            return 1
        values = FILTER_GRADES[grade]
        built = build_filtered_truth(engine, label, collection, grade, values,
                                     base_n or 0, limit)
        if built is None:
            return 1
        gt = built
    cmd = [
        "cargo", "run", "--release", "--quiet", "--",
        "relevance",
        "--engine", engine,
        "--label", label,
        "--base", str(base),
        "--queries", str(queries_file),
        "--ground-truth", str(gt),
        "--metric", METRIC,
        "--limit", str(limit),
        "--limit-queries", str(queries),
        "--collection", collection,
        "--skip-upload",
        "--json", str(out),
        "--ef", *[str(e) for e in ef],
    ]
    if grade is not None:
        cmd += ["--filter-field", FILTER_FIELD, "--filter-values", ",".join(values)]
        # The collection is a prefix of the corpus, and the truth was built
        # over that prefix; the sweep has to read the same one or the checksum
        # guard inside the binary refuses the pair.
        if base_n:
            cmd += ["--limit-base", str(base_n)]
    if oversampling is not None:
        cmd += ["--quantization-oversampling", str(oversampling)]
    if rescore is not None:
        cmd += ["--quantization-rescore", "true" if rescore else "false"]
    what = COLLECTIONS.get(collection, "unknown")
    if grade is not None:
        what += f", {grade}: {FILTER_FIELD} in {{{','.join(values)}}}"
    print(f"\n=== {collection} ({what})"
          f" oversampling={oversampling} rescore={rescore} ===", flush=True)
    r = subprocess.run(cmd, cwd=CONF)
    if r.returncode != 0:
        print(f"  !! sweep failed for {collection}; its rows will carry no recall",
              file=sys.stderr)
        return 1
    if not stamp(out, DATASET, collection, METRIC, oversampling, rescore, limit,
                 grade):
        return 1
    print(f"  wrote {out}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("label", help="result directory the sweeps belong to")
    ap.add_argument("--engine", default="http://localhost:6334")
    ap.add_argument("--collections", default=",".join(COLLECTIONS))
    ap.add_argument("--queries", type=int, default=RECALL_QUERIES,
                    help=f"held-out queries per ef (§4.2's split); default is "
                         f"every query {DATASET} ships ({RECALL_QUERIES:,})")
    ap.add_argument("--oversampling", type=float, default=None,
                    help="quantization oversampling to send (default: what §4's "
                         "row for each collection sends)")
    ap.add_argument("--rescore", choices=["true", "false"], default=None,
                    help="quantization rescore to send (default: as above)")
    ap.add_argument("--ef", default=None,
                    help="comma-separated ef values to sweep instead of §4's "
                         "list; for exploration, since a sweep only speaks for "
                         "a throughput row at an ef that row searched")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"results per query (default {DEFAULT_LIMIT}). §8.9 lists "
                         f"recall@100, which needs 100. A non-default limit writes "
                         f"to its own `.k<limit>` file so the sweep the W10 rows "
                         f"join is not replaced, and ef values below the limit are "
                         f"dropped: the engines search at max(ef, limit)")
    ap.add_argument("--filtered-base-n", type=int, default=None,
                    help=f"how many corpus rows {FILTERED_COLLECTION} holds, so a "
                         f"filtered grade's ground truth is over the same prefix. "
                         f"`fullrun.py` passes `workloads.w12_n()`; without it a "
                         f"filtered sweep is skipped rather than scored against "
                         f"the whole corpus")
    paths.add_data_argument(ap)
    args = ap.parse_args(argv[1:])
    paths.use_data_dir(args.data_dir)

    for f in paths.dataset(DATASET):
        if not f.exists():
            # The same distinction `fullrun.py` makes: past SIFT1M the missing
            # file is the converted corpus or the fp64 ground truth, and
            # `fetch` produces neither.
            print(f"missing {f}\n"
                  f"the archives come from: conformance/datasets/datasets.py fetch {DATASET}\n"
                  f"the fbin and the ground truth are made by the dataset's own "
                  f"conversion (for dbpedia-openai-1m: bench/harness/headline.py)",
                  file=sys.stderr)
            return 1

    rc = 0
    for c in args.collections.split(","):
        c = c.strip()
        # Each flag overrides its own parameter only: `--rescore` alone keeps
        # the collection's default oversampling.
        ov, rs = quant_params_of(c)
        if args.oversampling is not None:
            ov = args.oversampling
        if args.rescore is not None:
            rs = args.rescore == "true"
        efs = [int(x) for x in args.ef.split(",")] if args.ef else None
        if c == FILTERED_COLLECTION:
            # One collection, one sweep per grade: W12's two rows differ only
            # in selectivity, so they search the same `bench12` under different
            # conditions and each needs its own restricted truth.
            for grade in FILTER_GRADES:
                rc |= sweep(args.engine, args.label, c, args.queries, ov, rs, efs,
                            args.limit, grade=grade, base_n=args.filtered_base_n)
            continue
        rc |= sweep(args.engine, args.label, c, args.queries, ov, rs, efs, args.limit)

    print("\nrecall written next to the throughput rows; `compare.py` and the HTML "
          "report join them on (dataset, collection, ef).")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
