#!/usr/bin/env python3
"""Where this project is allowed to put things, in one place.

Six files each carried their own `Path.home() / ".cache/strawmann/datasets"`,
none honouring `XDG_CACHE_HOME`, and a docstring recommending
`--storage ~/qdrant_storage` — which is how 1.6 GB ended up loose in a home
directory. Everything written outside the checkout now lives under one root:

    $STRAWMANN_CACHE                  default ${XDG_CACHE_HOME:-~/.cache}/strawmann
      datasets/                       $STRAWMANN_DATA
      qdrant-storage/                 $QDRANT_STORAGE

Each level stays individually overridable, because a 16 GB dataset directory is
the one thing a user may reasonably want on another disk. Results stay in the
checkout (`bench/results/`, gitignored): they belong to the tree that produced
them, not to the user's cache.

## Where a root comes from

Four sources, highest first, so a lasting choice is written once and still
overridable for one run:

| source | scope |
|---|---|
| `--data-dir` | this run |
| `$STRAWMANN_DATA`, `$STRAWMANN_CACHE`, `$QDRANT_STORAGE` | this shell |
| `$XDG_CONFIG_HOME/strawmann/config.toml`, a `[paths]` table | this user |
| the layout above | always |

`SOURCES` records which of the four each root came from, because a default and
an inherited variable produce identical output and are not the same thing to
debug (`scripts/doctor.py` reports it). A malformed config file is an error,
not a shrug: a typo in the file that says where 16 GB lives would otherwise
look exactly like "nothing is fetched".

## Overriding for one run

`use_data_dir` **also exports `$STRAWMANN_DATA`**, so the directory is chosen
once at the outermost tool and every spawned tool — `workloads.py`,
`recall.py`, bfb, the conformance binary — agrees without being told again.
That export is also what lets a directory be shared with bfb or
vector-db-benchmark, which use the same `<root>/<dataset>/...` layout.

Because the override lands after import, a dataset path must be read from
`paths.DATA` at use, not copied into a module constant at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import NamedTuple


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


#: `XDG_CACHE_HOME` first, as the spec says, and `~/.cache` only as its
#: documented fallback. Hardcoding the fallback is what made this unmovable.
XDG_CACHE = _env_path("XDG_CACHE_HOME", Path.home() / ".cache")
XDG_CONFIG = _env_path("XDG_CONFIG_HOME", Path.home() / ".config")

#: `$STRAWMANN_CONFIG` names another file, which is how the tests point this at
#: a temporary one without touching the user's.
CONFIG_FILE = _env_path("STRAWMANN_CONFIG", XDG_CONFIG / "strawmann" / "config.toml")

#: The `[paths]` keys, and the module global each one sets.
CONFIG_KEYS = ("cache", "datasets", "qdrant_storage", "strawmann_storage")


def read_config(path: Path) -> dict[str, str]:
    """The `[paths]` table of `path`, or `{}` when there is no file.

    Malformed input raises rather than falling back. This is the one file that
    says where 16 GB lives, and a silent fall back to the default root looks
    exactly like "nothing is fetched" — a confusing way to spend an afternoon.
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise SystemExit(f"{path}: {e}") from e
    stray = sorted(set(doc) & set(CONFIG_KEYS))
    if stray:
        raise SystemExit(f"{path}: {', '.join(stray)} must be under a [paths] table")
    table = doc.get("paths", {})
    if not isinstance(table, dict):
        raise SystemExit(f"{path}: [paths] must be a table")
    unknown = sorted(set(table) - set(CONFIG_KEYS))
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) under [paths]: {', '.join(unknown)}; "
                         f"known: {', '.join(CONFIG_KEYS)}")
    return {k: str(v) for k, v in table.items()}


CONFIG = read_config(CONFIG_FILE)

#: Where each root came from, for `scripts/doctor.py`. Keyed as `CONFIG_KEYS`.
SOURCES: dict[str, str] = {}


def _root(key: str, env: str, default: Path) -> Path:
    """One root, resolved environment first and config file second."""
    if os.environ.get(env):
        SOURCES[key] = f"${env}"
        return Path(os.environ[env]).expanduser()
    if key in CONFIG:
        SOURCES[key] = str(CONFIG_FILE)
        return Path(CONFIG[key]).expanduser()
    SOURCES[key] = "default"
    return default


CACHE = _root("cache", "STRAWMANN_CACHE", XDG_CACHE / "strawmann")
DATA = _root("datasets", "STRAWMANN_DATA", CACHE / "datasets")
QDRANT_STORAGE = _root("qdrant_storage", "QDRANT_STORAGE", CACHE / "qdrant-storage")
#: Where strawmANN maps `cached`/`cold` arenas from. Needed because §7.4's
#: comparison is run at one residency for both engines and the only residency
#: both can serve is a file-backed one — Qdrant refuses `pinned` for dense
#: vectors outright. Its own store, not Qdrant's, so the two engines' disk
#: counters describe their own files.
STRAWMANN_STORAGE = _root("strawmann_storage", "STRAWMANN_STORAGE",
                          CACHE / "strawmann-storage")


def use_data_dir(path: str | Path | None) -> Path:
    """Adopt `path` as the dataset root for this process *and its children*.

    `None` changes nothing, so a caller can hand it `args.data_dir` unguarded.

    Exporting `$STRAWMANN_DATA` is the point: the directory is chosen at the
    outermost tool and inherited by every subprocess below it, including ones
    that are not ours. Rebinding the module global is the other half, and it
    only reaches code that reads `paths.DATA` at call time.
    """
    global DATA
    if path is None:
        return DATA
    DATA = Path(path).expanduser()
    SOURCES["datasets"] = "--data-dir"
    os.environ["STRAWMANN_DATA"] = str(DATA)
    return DATA


def add_data_argument(ap: argparse.ArgumentParser) -> None:
    """`--data-dir`, spelled the same way by every tool that reads a dataset."""
    ap.add_argument("--data-dir", metavar="DIR", default=None,
                    help=f"dataset root to read and write (default {DATA}); "
                         f"exported as $STRAWMANN_DATA so every tool this one "
                         f"spawns reads the same directory")


#: The converted corpus of every dataset a benchmark row can read, relative to
#: that dataset's own directory.
#:
#: Written down rather than derived: the fetcher's output is not this. SIFT1M's
#: fvecs went through `convert-vecs` and dbpedia's parquet shards through
#: `convert-parquet`, on different days, and the two tools were told different
#: output names. Guessing one convention would report "missing file" for a
#: dataset that is fully fetched and converted, which reads as a fetch problem
#: and is not one.
#:
#: A dataset is in here when its corpus exists in a form a row can open, which
#: for the `tar` entries in `datasets.json` means `convert-npy` has been run
#: over the extracted bundle. The two that remain absent
#: (`laion-small-clip`, `h-and-m-2048-angular-filters`) are convertible the same
#: way, but their `tests.jsonl` ships neighbours computed *under each query's
#: filter*, so they have no unfiltered ground truth to be scored against yet.
_CORPUS = {
    "sift1m": ("sift1m.fbin", "sift1m_query.fbin"),
    "dbpedia-openai-1m": ("base.fbin", "queries.fbin"),
    "dbpedia-openai-100K-1536-angular": ("base.fbin", "queries.fbin"),
}

#: `datasets.json`, for the dimension and the metric. Read from the descriptor
#: rather than restated, so a dataset is defined in exactly one file.
_DESCRIPTOR = Path(__file__).resolve().parents[2] / "conformance/datasets/datasets.json"


def _entry(name: str) -> dict:
    try:
        doc = json.loads(_DESCRIPTOR.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read {_DESCRIPTOR}: {e}") from e
    for d in doc:
        if d["name"] == name:
            return d
    raise SystemExit(f"{name}: not in {_DESCRIPTOR}")


def runnable() -> list[str]:
    """Datasets a benchmark row can read, whether or not they are on this disk.

    The choice list for `--dataset`. Presence is a separate question and a
    separate error: "not a dataset this harness can run" and "that dataset is
    not converted yet" send the reader to different places.
    """
    return sorted(_CORPUS)


def metric(name: str) -> str:
    """The distance `name`'s ground truth was computed under.

    The descriptor's, always. A collection created under one metric and scored
    against ground truth computed under another is the exact failure the
    `METRIC` note in `workloads.py` records, and it survived review once
    because the recall it produced still looked plausible.
    """
    return _entry(name)["distance"]


def dim(name: str) -> int:
    """`name`'s vector width, for the `-d` every upload row sends."""
    return _entry(name)["vector_size"]


def n_queries(name: str) -> int:
    """How many held-out queries `name` ships, from the descriptor.

    The ceiling on a recall sweep. Asking for more than this silently measures
    fewer, and the recall a sweep reports is a proportion whose confidence
    interval is set by exactly this number.
    """
    return int(_entry(name)["n_queries"])


class Corpus(NamedTuple):
    """The three files a dataset is, by name rather than by position.

    A `NamedTuple` because these are three `Path`s of one type returned in a
    fixed order, unpacked by convention at every call site and indexed `[0]` at
    one. Transposing two is silent, and the wrong one of these is not a crash:
    scoring against `queries` as if it were `base`, or against another
    dataset's ground truth, produces a plausible recall number, which is the
    class of mistake this project has made most often.

    Still a tuple, so `base, queries, gt = paths.dataset(x)` and `[0]` keep
    working exactly as they did.
    """

    base: Path
    queries: Path
    ground_truth: Path


def dataset(name: str) -> Corpus:
    """`name`'s converted base, its held-out queries, and its cached ground truth.

    Resolved on call, so `use_data_dir` reaches it. `recall.py`, `workloads.py`
    and `fullrun.py` all need the trio and each used to spell it out from its
    own copy of `DATA`, which is three places for one layout to be written down.
    """
    if name not in _CORPUS:
        raise SystemExit(f"{name}: no converted corpus; harness datasets are "
                         f"{', '.join(runnable())}")
    base, queries = _CORPUS[name]
    d = DATA / name
    return Corpus(base=d / base, queries=d / queries,
                  ground_truth=d / "gt" / f"{name}.{metric(name)}.k100.gt.json")


def source(key: str) -> str:
    """Where `key`'s root came from, in words rather than as a path.

    `SOURCES` stores the config file by name so that two different files (a
    test's and the user's) stay distinguishable; every reader wants the short
    form, so it lives here rather than in each of them.
    """
    s = SOURCES.get(key, "default")
    return "config file" if s == str(CONFIG_FILE) else s


def describe() -> str:
    """The layout and where each part of it came from."""
    have = "present" if CONFIG_FILE.exists() else "not created"
    return (f"config         {CONFIG_FILE} ({have})\n"
            f"cache          {CACHE} ({source('cache')})\n"
            f"  datasets     {DATA} ({source('datasets')})\n"
            f"  qdrant       {QDRANT_STORAGE} ({source('qdrant_storage')})\n"
            f"  strawmann    {STRAWMANN_STORAGE} ({source('strawmann_storage')})")


if __name__ == "__main__":
    print(describe())
