#!/usr/bin/env python3
"""§4.2 dataset management: one descriptor, many datasets.

    datasets.py list                    inventory: declared, available, present
    datasets.py info sift1m             everything known about one dataset
    datasets.py fetch [name ...]        download and verify
    datasets.py verify [name ...]       re-checksum what is on disk, fetch nothing
    datasets.py extract [name ...]      unpack archives named in the descriptor
    datasets.py bfb-config <name>       emit a bfb-compatible dataset entry
    datasets.py add                     scaffold a new entry, with digests fetched

Destination, highest source first: `--data-dir`, then `$STRAWMANN_DATA`,
then `datasets` under the `[paths]` table of
`${XDG_CONFIG_HOME:-~/.config}/strawmann/config.toml`, then
`$STRAWMANN_CACHE/datasets`, itself defaulting to
`${XDG_CACHE_HOME:-~/.cache}/strawmann`.

That is `bench/harness/paths.py`'s rule, and this is a deliberate second
copy of it: the script is stdlib-only and standalone by design, so it runs
on a fresh clone with no `bench/` on the path. The *rule* is the contract,
not the module, and `bench/harness/test_paths.py` asserts the two copies resolve the
same root from the same inputs rather than trusting that they do.

`--data-dir` exports `$STRAWMANN_DATA` for anything this process spawns, and
exists so a directory can be *shared*. These corpora are large and public:
bfb and vector-db-benchmark keep the same files under the same
`<root>/<dataset>/` shape, so pointing every tool at one root fetches each
dataset once instead of once per tool. Where an entry's `extract.dir` is
given it is vector-db-benchmark's own path for that dataset, so a shared root
is genuinely shared rather than merely adjacent.

## Why this replaced manifest.tsv

Adding a dataset used to mean editing three places that could disagree:
`manifest.tsv` for files and digests, the `DATASETS` const in
`conformance/src/datasets/mod.rs` for dimensions and metric, and a hardcoded
converter for the layout. §4.2 lists six datasets; two were fetchable and the
other four existed only as Rust constants, so the spec's table and the
fetchable set had already drifted.

`datasets.json` is now the single source of truth, and it carries `status`:
`available` means every file is described and can be fetched, `declared` means
§4.2 names it but nobody has pinned its files yet. A drift between the two is
visible in `list` instead of being invisible across two files.

## Shape

Deliberately close to bfb's `src/dataset/config.rs`, which is itself compatible
with vector-db-benchmark's `datasets.json`: `name`, `format`, `vector_size`,
`distance`, `parts`. Staying close means a dataset defined here can drive bfb
directly, via `bfb-config`.

It is a superset, not a copy. bfb has no checksums, and §4.2 requires "static,
versioned, checksummed"; bfb's config also uses `deny_unknown_fields`, so our
extra keys would be rejected. `bfb-config` therefore emits a projection rather
than the file itself.

Digests are **upstream** Git-LFS object ids, not digests computed after
downloading. A self-computed digest only proves the file has not rotted on our
disk. The upstream one also proves we fetched what the publisher published.

Not every host publishes a sha256. Google Cloud Storage, which serves the
ann-filtered-benchmark archives, publishes an md5 (`x-goog-hash`, and the
ETag for a single-part object) and nothing else. Those entries carry both: an
`md5` that is upstream's, checked once as the bytes arrive, and a `sha256`
computed from the very bytes that satisfied it. The upstream anchor and the
rot check are then still two different claims, made by two different digests,
rather than one self-computed number pretending to be both.

## Sharded datasets

`parts` describes a numbered family read as one row space. bfb templates the
path with `{i}`; we cannot, because Hugging Face appends a content hash to each
shard name (`train-00007-of-00026-1cd8b3ca61f2ab2b.parquet`) and every shard
needs its own digest anyway. So `parts.files` is an explicit ordered list, and
the order is load-bearing: row order defines point ids, and point ids are what
ground truth indexes into.

## Archives

`extract.members` lists what an archive is expected to contain, by path and
size, so a half-unpacked directory is visible rather than mistaken for a
present dataset. They land in `extract.dir` under the root when the entry
names one, and in `<dataset>/` when it does not.

Stdlib only: this runs on a fresh clone, before `uv sync`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESCRIPTOR = HERE / "datasets.json"


def config_file(env: dict[str, str] | None = None) -> Path:
    """Where the per-user config lives; `$STRAWMANN_CONFIG` names another file."""
    env = os.environ if env is None else env
    return Path(env.get("STRAWMANN_CONFIG") or
                (Path(env.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
                 / "strawmann" / "config.toml")).expanduser()


def read_config(path: Path) -> dict[str, str]:
    """The `[paths]` table of `path`, or `{}` when there is no file.

    Malformed input raises rather than falling back to the default root, which
    would look exactly like "nothing is fetched" while 16 GB sat elsewhere.
    The accepted keys are `paths.py`'s, so a typo is refused by both.
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise SystemExit(f"{path}: {e}") from e
    keys = ("cache", "datasets", "qdrant_storage")
    stray = sorted(set(doc) & set(keys))
    if stray:
        raise SystemExit(f"{path}: {', '.join(stray)} must be under a [paths] table")
    table = doc.get("paths", {})
    if not isinstance(table, dict):
        raise SystemExit(f"{path}: [paths] must be a table")
    unknown = sorted(set(table) - set(keys))
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) under [paths]: {', '.join(unknown)}; "
                         f"known: {', '.join(keys)}")
    return {k: str(v) for k, v in table.items()}


def default_root(env: dict[str, str] | None = None,
                 config: dict[str, str] | None = None) -> Path:
    """The dataset root asked for, environment first and config file second."""
    env = os.environ if env is None else env
    config = read_config(config_file(env)) if config is None else config
    if env.get("STRAWMANN_DATA"):
        return Path(env["STRAWMANN_DATA"]).expanduser()
    if config.get("datasets"):
        return Path(config["datasets"]).expanduser()
    cache = (Path(env["STRAWMANN_CACHE"]) if env.get("STRAWMANN_CACHE")
             else Path(config["cache"]) if config.get("cache")
             else Path(env.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "strawmann")
    return (cache / "datasets").expanduser()


DATA_ROOT = default_root()


def use_data_dir(path: str | None) -> Path:
    """Adopt `path` as the root, and export it for anything we spawn.

    `None` changes nothing, so `args.data_dir` can be passed unguarded. The
    export is what makes the choice reach a second tool rather than stopping at
    this one; see the note on sharing a root at the top of this file.
    """
    global DATA_ROOT
    if path is None:
        return DATA_ROOT
    DATA_ROOT = Path(path).expanduser()
    os.environ["STRAWMANN_DATA"] = str(DATA_ROOT)
    return DATA_ROOT


GREEN, RED, DIM, YELLOW, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = YELLOW = OFF = ""


@dataclass(frozen=True)
class File:
    dataset: str
    path: str
    link: str
    bytes: int
    sha256: str
    role: str = "base"
    #: Upstream's own digest where the host publishes md5 rather than a sha256.
    #: Empty when `sha256` is already upstream's, as it is for Hugging Face.
    md5: str = ""

    def local(self, root: Path) -> Path:
        return root / self.dataset / self.path


#: §4.2's dataset states, mirrored by `datasets::Status` in the Rust
#: descriptor. Two readers, one closed set.
STATUSES = frozenset({"available", "declared"})


class Dataset:
    def __init__(self, raw: dict):
        self.raw = raw
        self.name: str = raw["name"]
        #: §4.2's two states. Validated rather than taken as written: the same
        #: word is compared against a literal here and in the Rust descriptor,
        #: and a typo used not to fail — it made the dataset invisible to every
        #: `status == "available"` filter on both sides while this script still
        #: listed the descriptor and the gate still passed. The Rust side
        #: rejects an unknown value at parse; so does this.
        self.status: str = raw.get("status", "declared")
        if self.status not in STATUSES:
            raise SystemExit(f"{DESCRIPTOR}: dataset {self.name!r} has status "
                             f"{self.status!r}; expected one of {', '.join(sorted(STATUSES))}")
        self.format: str = raw.get("format", "unknown")
        self.dim = raw.get("vector_size")
        self.distance = raw.get("distance")
        self.n = raw.get("n")
        self.n_queries = raw.get("n_queries")
        self.role: str = raw.get("role", "")
        self.gt_shipped: bool = raw.get("gt_shipped", False)
        #: `.npy` sources carry their own dtype and it is not always f32;
        #: reading float16 as float32 halves the row count without an error.
        self.dtype: str = raw.get("dtype", "")
        self.notes: str = raw.get("notes", "")

    @property
    def files(self) -> list[File]:
        out = [File(self.name, **f) for f in self.raw.get("files", [])]
        parts = self.raw.get("parts")
        if parts:
            for f in parts["files"]:
                out.append(File(self.name, role="base", **f))
        return out

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def extract_members(self) -> list[dict]:
        return self.raw.get("extract", {}).get("members", [])

    def extract_dir(self, root: Path) -> Path:
        """Where this dataset's archive members live once unpacked.

        `<root>/<name>` unless the entry names an `extract.dir`, which the
        filtered-benchmark entries do: theirs is vector-db-benchmark's path for
        the same dataset, so a root shared with that tool holds one copy of the
        files rather than two identical ones under different names.
        """
        return root / self.raw.get("extract", {}).get("dir", self.name)


def load(names: list[str] | None = None) -> list[Dataset]:
    if not DESCRIPTOR.exists():
        raise SystemExit(f"no descriptor at {DESCRIPTOR}")
    try:
        raw = json.loads(DESCRIPTOR.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{DESCRIPTOR}: {e}") from e
    ds = [Dataset(r) for r in raw]
    seen = set()
    for d in ds:
        if d.name in seen:
            raise SystemExit(f"{DESCRIPTOR}: duplicate dataset {d.name!r}")
        seen.add(d.name)
    if names:
        known = {d.name for d in ds}
        for n in names:
            if n not in known:
                raise SystemExit(f"unknown dataset {n!r}; known: {', '.join(sorted(known))}")
        ds = [d for d in ds if d.name in names]
    return ds


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def digest(path: Path, algo: str = "sha256") -> str:
    """Streamed: a 366 MB parquet costs about a second, memory stays flat."""
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class FileState:
    file: File
    present: bool
    wrong_digest: bool


def survey(d: Dataset, root: Path, check_digest: bool = True) -> list[FileState]:
    out = []
    for f in d.files:
        p = f.local(root)
        if not p.exists() or p.stat().st_size != f.bytes:
            out.append(FileState(f, False, False))
            continue
        if check_digest and digest(p) != f.sha256:
            out.append(FileState(f, False, True))
            continue
        out.append(FileState(f, True, False))
    return out


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def stale_links(d: Dataset, root: Path) -> list[Path]:
    """This dataset's bfb shard links that no longer resolve.

    `bfb-config --link` writes absolute symlinks against whatever root was
    current when it ran, so moving the datasets root leaves a farm pointing at
    the old one. `survey` cannot catch this: the shards themselves are present
    and still match their digests, so the dataset reads as complete while bfb
    opens the template and finds nothing. Re-run `bfb-config --link`.
    """
    link_dir = root / d.name / "bfb-parts"
    if not link_dir.is_dir():
        return []
    return sorted(p for p in link_dir.iterdir()
                  if p.is_symlink() and not p.exists())


def dir_bytes(path: Path) -> int:
    """Every byte under `path`, or 0 if it is not there."""
    total = 0
    stack = [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                stack.append(Path(e.path))
            elif e.is_file(follow_symlinks=False):
                total += e.stat().st_size
    return total


def on_disk(d: Dataset, root: Path) -> int:
    """What this dataset actually occupies, measured rather than declared.

    Measured, because the two numbers are different and both are true: the
    descriptor knows what was downloaded, the disk also holds what we unpacked
    and what we converted. SIFT1M is a 160 MB archive and 1.2 GB of directory —
    fvecs, the fbin the oracle read, and the cached ground truth.

    Two directories, because they can differ: the archive stays under the
    dataset's own name and `extract.dir` may put its contents somewhere the
    other tools already look. Deduplicated, or the common case where they are
    the same directory would count everything twice.
    """
    dirs = {root / d.name, d.extract_dir(root)}
    return sum(dir_bytes(p) for p in dirs)


def inventory(ds: list[Dataset], root: Path, check_digest: bool = True) -> list[dict]:
    """One row per dataset, for the table and for `--json` alike.

    Both renderings come from here so they cannot disagree about what `state`
    means — `scripts/doctor.py` reads the JSON and used to scrape the table.
    """
    rows = []
    for d in ds:
        row = {"name": d.name, "status": d.status, "format": d.format,
               "vector_size": d.dim, "distance": d.distance,
               "declared_bytes": 0, "on_disk_bytes": on_disk(d, root),
               "files_present": 0, "files_total": 0, "state": "declared",
               "stale_links": len(stale_links(d, root))}
        if d.status != "declared":
            st = survey(d, root, check_digest=check_digest)
            bad = sum(1 for x in st if x.wrong_digest)
            row["files_present"] = sum(1 for x in st if x.present)
            row["files_total"] = len(st)
            row["declared_bytes"] = d.total_bytes
            row["state"] = ("digest mismatch" if bad
                            else "complete" if row["files_present"] == len(st)
                            else "partial" if row["files_present"]
                            else "not fetched")
        rows.append(row)
    return rows


def cmd_list(args) -> int:
    ds = load(args.datasets or None)
    rows = inventory(ds, DATA_ROOT, check_digest=not args.fast)
    if args.json:
        print(json.dumps({"root": str(DATA_ROOT), "datasets": rows}, indent=2))
        return 0

    # Widened to fit, rather than pinned at 20: the filtered-benchmark names run
    # to 32 characters, and a name that overflows its column shifts every field
    # after it on that row only, which reads as corruption.
    w = max(20, max((len(d.name) for d in ds), default=20))
    print(f"{DIM}datasets root: {DATA_ROOT}{OFF}\n")
    print(f"  {'dataset':<{w}} {'status':<10} {'format':<8} {'dim':>5} {'metric':<7} "
          f"{'download':>8} {'on disk':>8}  state")
    print(f"  {'-' * w} {'-' * 10} {'-' * 8} {'-' * 5} {'-' * 7} {'-' * 8} {'-' * 8}  "
          f"{'-' * 22}")
    paint = {"declared": f"{DIM}§4.2 names it, no files pinned{OFF}",
             "complete": f"{GREEN}complete{OFF}",
             "partial": f"{YELLOW}partial{OFF}",
             "not fetched": f"{DIM}not fetched{OFF}"}
    for r in rows:
        state = paint.get(r["state"], f"{RED}{r['state'].upper()}{OFF}")
        if r["state"] == "complete":
            state += f" ({r['files_present']} files)"
        elif r["state"] == "partial":
            state += f" {r['files_present']}/{r['files_total']}"
        print(f"  {r['name']:<{w}} {r['status']:<10} {r['format']:<8} "
              f"{r['vector_size'] or '-'!s:>5} {(r['distance'] or '-'):<7} "
              f"{human(r['declared_bytes']) if r['declared_bytes'] else '-':>8} "
              f"{human(r['on_disk_bytes']) if r['on_disk_bytes'] else '-':>8}  {state}")
    grand = sum(r["declared_bytes"] for r in rows)
    disk = sum(r["on_disk_bytes"] for r in rows)
    print(f"\n  {len(rows)} dataset(s); {human(grand)} declared, {human(disk)} on disk "
          f"{DIM}(the archives, plus what was unpacked and converted from them){OFF}")
    if any(r["status"] == "declared" for r in rows):
        print(f"  {DIM}`declared` entries come from §4.2's table. Pin their files with "
              f"`datasets.py add`.{OFF}")
    return 0


def cmd_info(args) -> int:
    d = load([args.dataset])[0]
    print(f"{d.name}")
    print(f"  role        {d.role}")
    print(f"  status      {d.status}")
    print(f"  format      {d.format}")
    print(f"  vectors     n={d.n:,}  dim={d.dim}  metric={d.distance}"
          if d.n else f"  vectors     dim={d.dim}  metric={d.distance}")
    print(f"  queries     {d.n_queries:,}" if d.n_queries else "  queries     -")
    print(f"  ground truth{'shipped, and we diff against it' if d.gt_shipped else 'recomputed (§4.2)'}"
          .replace("truth", "truth  "))
    if d.dtype:
        print(f"  dtype       {d.dtype}  (the element type on disk, not what a reader assumes)")
    if d.raw.get("payload_schema"):
        fields = d.raw["payload_schema"]
        print(f"  payload     {len(fields)} field(s): "
              f"{', '.join(list(fields)[:4])}{', ...' if len(fields) > 4 else ''}")
    if d.raw.get("column"):
        print(f"  column      {d.raw['column']}  (the vector column inside each file)")
    if d.raw.get("held_out_queries"):
        print(f"  held out    {d.raw['held_out_queries']:,} rows become the query set, "
              f"so base and queries never overlap")
    parts = d.raw.get("parts")
    if parts:
        print(f"  parts       {parts['count']} shards from {parts['start']}, "
              f"read as one row space in listed order")
    if d.notes:
        print()
        for line in textwrap.wrap(d.notes, 74):
            print(f"  {line}")
    if not d.files:
        print(f"\n  {DIM}no files pinned yet{OFF}")
        return 0
    print(f"\n  {len(d.files)} file(s), {human(d.total_bytes)}:")
    st = survey(d, DATA_ROOT, check_digest=not args.fast)
    by_role: dict[str, list[FileState]] = {}
    for s in st:
        by_role.setdefault(s.file.role, []).append(s)
    for role, group in by_role.items():
        mark = (f"{GREEN}present{OFF}" if all(g.present for g in group)
                else f"{RED}missing{OFF}")
        if len(group) == 1:
            print(f"    {role:<14} {group[0].file.path:<58} "
                  f"{human(group[0].file.bytes):>9}  {mark}")
        else:
            n_ok = sum(1 for g in group if g.present)
            print(f"    {role:<14} {len(group)} shards, {n_ok} present"
                  f"{'':<38}{human(sum(g.file.bytes for g in group)):>9}  {mark}")
    ex_root = d.extract_dir(DATA_ROOT)
    for m in d.extract_members:
        p = ex_root / m["path"]
        ok = p.exists() and p.stat().st_size == m["bytes"]
        print(f"    {m['role']:<14} {m['path']:<58} {human(m['bytes']):>9}  "
              + (f"{GREEN}extracted{OFF}" if ok else f"{DIM}not extracted{OFF}"))
    if d.extract_members:
        print(f"\n  unpacked into {ex_root}")
    disk = on_disk(d, DATA_ROOT)
    if disk:
        print(f"  {human(disk)} on disk {DIM}(everything under this dataset's "
              f"directories, including anything converted from it){OFF}")
    return 0


def cmd_verify(args) -> int:
    ds = [d for d in load(args.datasets or None) if d.status == "available"]
    w = max((len(d.name) for d in ds), default=20)
    bad = missing = 0
    for d in ds:
        st = survey(d, DATA_ROOT)
        for s in st:
            if s.wrong_digest:
                print(f"  {RED}!!{OFF} {d.name}/{s.file.path}: right size, WRONG digest",
                      file=sys.stderr)
                bad += 1
            elif not s.present:
                missing += 1
        for m in d.extract_members:
            p = d.extract_dir(DATA_ROOT) / m["path"]
            if not (p.exists() and p.stat().st_size == m["bytes"]):
                print(f"  {YELLOW}..{OFF} {d.name}/{m['path']} not extracted "
                      f"(run `datasets.py extract {d.name}`)")
                missing += 1
        n_ok = sum(1 for s in st if s.present)
        mark = GREEN + "ok" + OFF if n_ok == len(st) and not bad else RED + "incomplete" + OFF
        print(f"  {d.name:<{w}} {n_ok}/{len(st)} files  {mark}")
    if bad or missing:
        print(f"\n{bad} corrupt, {missing} missing", file=sys.stderr)
        return 1
    print(f"\nall {len(ds)} dataset(s) verified against upstream digests")
    return 0


def fetch_file(f: File, root: Path, i: int, n: int) -> None:
    target = f.local(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{i:2d}/{n:2d}] {f.dataset}/{f.path} ({human(f.bytes)})", flush=True)
    # --location: HF's resolve endpoint 302s to a CDN.
    # --continue-at -: resume rather than restart, which matters at 9 GB.
    # --retry-all-errors: CDN hiccups on a multi-GB transfer are routine.
    r = subprocess.run(["curl", "--fail", "--location", "--continue-at", "-",
                        "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
                        "--output", str(target), f.link])
    if r.returncode != 0:
        raise SystemExit(f"curl failed for {f.link} (exit {r.returncode})")
    if f.md5:
        # Upstream's only published digest for this host, so it is checked at
        # the one moment it is a statement about the transfer. `verify` re-runs
        # the sha256 instead, which is the digest that catches rot on our disk.
        got_md5 = digest(target, "md5")
        if got_md5 != f.md5:
            target.unlink(missing_ok=True)
            raise SystemExit(f"upstream md5 mismatch for {target}\n  expected {f.md5}\n"
                             f"  got      {got_md5}\nremoved the corrupt file")
    got = digest(target)
    if got != f.sha256:
        # A corrupt file left in place would be treated as present by the next
        # run. §4.2's "checksummed" only means something if failure removes it.
        target.unlink(missing_ok=True)
        raise SystemExit(f"digest mismatch for {target}\n  expected {f.sha256}\n"
                         f"  got      {got}\nremoved the corrupt file")


def cmd_fetch(args) -> int:
    if not shutil.which("curl"):
        raise SystemExit("curl not found; it is required for the transfer")
    ds = load(args.datasets or None)
    declared = [d.name for d in ds if d.status == "declared"]
    if declared and args.datasets:
        raise SystemExit(f"{', '.join(declared)}: §4.2 names these but no files are "
                         f"pinned. Add them with `datasets.py add` first.")
    todo: list[File] = []
    total = 0
    for d in ds:
        if d.status != "available":
            continue
        total += d.total_bytes
        todo += [s.file for s in survey(d, DATA_ROOT) if not s.present]
    if not todo:
        print(f"all {human(total)} present and verified")
        return 0
    print(f"fetching {len(todo)} file(s), {human(sum(f.bytes for f in todo))} "
          f"to go -> {DATA_ROOT}")
    for i, f in enumerate(todo, 1):
        fetch_file(f, DATA_ROOT, i, len(todo))
    print(f"done: {human(total)} verified in {DATA_ROOT}")
    return cmd_extract(args) if not args.no_extract else 0


def cmd_extract(args) -> int:
    n = 0
    for d in load(args.datasets or None):
        ex = d.raw.get("extract")
        if not ex:
            continue
        members = d.extract_members
        root = d.extract_dir(DATA_ROOT)
        if all((root / m["path"]).exists()
               and (root / m["path"]).stat().st_size == m["bytes"] for m in members):
            continue
        archive = next((f for f in d.files if f.role == "archive"), None)
        if archive is None or not archive.local(DATA_ROOT).exists():
            print(f"  {d.name}: archive not fetched yet", file=sys.stderr)
            continue
        src = archive.local(DATA_ROOT)
        print(f"  extracting {d.name}/{archive.path} -> {root}")
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src) as tf:
            # `filter="data"` refuses absolute paths, `..` escapes, symlinks and
            # device nodes. Python 3.14 makes it the default; setting it keeps
            # 3.11 and 3.12 from silently using the permissive behaviour.
            tf.extractall(root, filter="data")
        for m in members:
            p = root / m["path"]
            if not p.exists():
                print(f"  {RED}!!{OFF} {m['path']} missing after extraction", file=sys.stderr)
                return 1
            if p.stat().st_size != m["bytes"]:
                print(f"  {RED}!!{OFF} {m['path']}: expected {m['bytes']} bytes, "
                      f"got {p.stat().st_size}", file=sys.stderr)
                return 1
        n += 1
    print(f"extracted {n} archive(s)" if n else "nothing to extract")
    return 0


def cmd_bfb_config(args) -> int:
    """Emit a bfb-compatible entry for this dataset.

    bfb's `DatasetConfig` uses `deny_unknown_fields`, so our digests, `status`,
    `role` and `extract` block cannot be handed over as-is. This projects only
    the fields bfb understands, which is what lets a dataset defined here drive
    bfb without maintaining the definition twice. Nothing explanatory is added
    to the JSON, because an extra key would make the output bfb rejects.

    Sharded datasets need `--link`. bfb wants a `{i}` path template, and Hugging
    Face appends a per-shard content hash (`train-00007-of-00026-1cd8b3ca...`)
    that no template can reproduce. `--link` builds a directory of symlinks with
    templatable names, so the emitted template is correct rather than
    approximate. Emitting a template that looks plausible and resolves to
    nothing would be the worse failure: bfb would report an empty dataset, not
    a broken path.
    """
    d = load([args.dataset])[0]
    if d.status != "available":
        raise SystemExit(f"{d.name} has no files pinned yet")

    out: dict = {"name": d.name}
    if d.format in ("parquet", "npy", "h5", "hdf5", "tar", "sparse"):
        out["format"] = {"hdf5": "h5"}.get(d.format, d.format)
    if d.dim:
        out["vector_size"] = d.dim
    if d.distance:
        out["distance"] = d.distance

    parts = d.raw.get("parts")
    if parts:
        files = parts["files"]
        suffix = Path(files[0]["path"]).suffix
        link_dir = DATA_ROOT / d.name / "bfb-parts"
        template = f"{d.name}/bfb-parts/part_{{i}}{suffix}"
        if args.link:
            link_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files, start=parts["start"]):
                target = DATA_ROOT / d.name / f["path"]
                if not target.exists():
                    raise SystemExit(f"{target} is not fetched; run "
                                     f"`datasets.py fetch {d.name}` first")
                link = link_dir / f"part_{i}{suffix}"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
            print(f"linked {len(files)} shard(s) into {link_dir}", file=sys.stderr)
        else:
            print(f"note: {d.name} is sharded and its shard names carry content "
                  f"hashes, so no {{i}} template can address them.\n"
                  f"      Re-run with --link to create templatable symlinks "
                  f"under {link_dir},\n"
                  f"      otherwise the emitted path resolves to nothing.",
                  file=sys.stderr)
        out["parts"] = {"count": parts["count"], "start": parts["start"],
                        "path": template}
    elif d.files:
        out["path"] = bfb_path(d)

    if d.raw.get("column"):
        out["columns"] = [d.raw["column"]]
    if d.raw.get("payload_schema"):
        # bfb understands `schema`, and a filtered dataset is only useful to it
        # with one: it is what tells bfb which payload fields to index.
        out["schema"] = d.raw["payload_schema"]
    print(json.dumps([out], indent=2))
    return 0


def bfb_path(d: Dataset) -> str:
    """Where bfb should look for this dataset, relative to the datasets root.

    An archive is not a dataset: bfb's tar reader opens the *directory* holding
    `vectors.npy`, and its fbin/parquet readers open a file. So an entry with an
    `extract` block resolves to where its base member lands, and one without
    resolves to the file we fetched. Emitting the `.tgz` we downloaded would
    hand bfb a path it can open and cannot read.
    """
    if d.extract_members:
        base = next((m for m in d.extract_members if m["role"] == "base"),
                    d.extract_members[0])
        return (d.extract_dir(Path(".")) / base["path"]).parent.as_posix()
    base_file = next((f for f in d.files if f.role == "base"), d.files[0])
    return f"{d.name}/{base_file.path}"


def digest_from_head(head_output: str) -> str:
    """The upstream sha256 in a `curl -sIL` transcript, or "" when there is none.

    Only the FINAL response's headers count: `-L` prints one header block per
    hop, and a redirecting hop's `ETag` (a CDN's opaque tag, an S3 md5) used to
    be taken as the digest, since the first line containing "etag" won. Hugging
    Face puts the LFS oid in `X-Linked-ETag` on the final response; anything
    that is not a 64-hex sha256 is not a digest, and the caller writes `TODO`.
    """
    blocks = [b for b in head_output.replace("\r", "").split("\n\n") if b.strip()]
    if not blocks:
        return ""
    candidates = []
    for ln in blocks[-1].splitlines():
        key, _, val = ln.partition(":")
        key = key.strip().lower()
        if key in ("x-linked-etag", "etag"):
            val = val.strip().strip('"').removeprefix("sha256:").strip('"')
            # x-linked-etag first: it is the LFS oid where both are present.
            candidates.insert(0 if key == "x-linked-etag" else len(candidates), val)
    hexdigits = set("0123456789abcdef")
    for c in candidates:
        if len(c) == 64 and set(c.lower()) <= hexdigits:
            return c.lower()
    return ""


def cmd_self_test(args) -> int:
    """Unit checks for the pure helpers; `scripts/check.py`-shaped."""
    hop = "HTTP/2 302\netag: \"abc123\"\nlocation: https://cdn/x\n\n"
    final_ok = ("HTTP/2 200\ncontent-length: 10\n"
                "x-linked-etag: \"" + "a" * 64 + "\"\netag: \"deadbeef\"\n\n")
    assert digest_from_head(hop + final_ok) == "a" * 64, "LFS oid on the final hop"
    assert digest_from_head(final_ok.replace("x-linked-etag", "x-other")) == "", \
        "an md5-looking ETag is not a sha256"
    final_sha = "HTTP/2 200\nETag: \"sha256:" + "B" * 64 + "\"\n\n"
    assert digest_from_head(hop + final_sha) == "b" * 64, "sha256: prefix stripped, lowercased"
    assert digest_from_head("HTTP/2 302\netag: \"" + "c" * 64 + "\"\n\n"
                            "HTTP/2 200\ncontent-length: 1\n\n") == "", \
        "a sha256-looking ETag on a redirect hop is not the file's"
    assert digest_from_head("") == ""

    env = {"STRAWMANN_DATA": "/a", "STRAWMANN_CACHE": "/b", "XDG_CACHE_HOME": "/c"}
    cfg = {"datasets": "/cfg", "cache": "/cfgcache"}
    assert default_root(env, cfg) == Path("/a"), "STRAWMANN_DATA wins"
    assert default_root({k: v for k, v in env.items() if k != "STRAWMANN_DATA"}, cfg) \
        == Path("/cfg"), "then the config file, over any cache root"
    assert default_root({"STRAWMANN_CACHE": "/b", "XDG_CACHE_HOME": "/c"}, {}) \
        == Path("/b/datasets"), "then the cache root"
    assert default_root({"XDG_CACHE_HOME": "/c"}, {"cache": "/cfgcache"}) \
        == Path("/cfgcache/datasets"), "a config cache root, when it names no datasets dir"
    assert default_root({"XDG_CACHE_HOME": "/c"}, {}) == Path("/c/strawmann/datasets"), \
        "then XDG, and the datasets dir is under the project's own root"
    assert config_file({"XDG_CONFIG_HOME": "/x"}) == Path("/x/strawmann/config.toml")
    assert config_file({"STRAWMANN_CONFIG": "/x/other.toml"}) == Path("/x/other.toml")
    assert read_config(Path("/nonexistent/config.toml")) == {}, "no file is not an error"

    plain = Dataset({"name": "d", "extract": {"members": [{"path": "vectors.npy",
                                                           "bytes": 1, "role": "base"}]}})
    assert plain.extract_dir(Path("/r")) == Path("/r/d"), "default: the dataset's own dir"
    assert bfb_path(plain) == "d", "bfb opens the directory holding vectors.npy"
    shared = Dataset({"name": "d", "extract": {"dir": "up/stream",
                                               "members": [{"path": "vectors.npy",
                                                            "bytes": 1, "role": "base"}]}})
    assert shared.extract_dir(Path("/r")) == Path("/r/up/stream"), "extract.dir wins"
    assert bfb_path(shared) == "up/stream"
    nested = Dataset({"name": "s", "extract": {"members": [{"path": "sift/base.fvecs",
                                                            "bytes": 1, "role": "base"}]}})
    assert bfb_path(nested) == "s/sift", "the member's own directory, not the archive's"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = Dataset({"name": "one", "status": "available",
                     "extract": {"dir": "shared/place",
                                 "members": [{"path": "vectors.npy", "bytes": 4,
                                              "role": "base"}]}})
        assert on_disk(d, root) == 0, "nothing on disk is not an error"
        (root / "one").mkdir()
        (root / "one" / "archive.tgz").write_bytes(b"x" * 10)
        (root / "shared" / "place").mkdir(parents=True)
        (root / "shared" / "place" / "vectors.npy").write_bytes(b"y" * 4)
        # Converted artefacts count: they are on the disk and the disk is what
        # is being reported.
        (root / "one" / "gt").mkdir()
        (root / "one" / "gt" / "k100.json").write_bytes(b"z" * 6)
        assert on_disk(d, root) == 20, "the archive, the extract dir, and what we made"
        same = Dataset({"name": "one", "status": "available",
                        "extract": {"members": [{"path": "v", "bytes": 1, "role": "base"}]}})
        assert on_disk(same, root) == 16, "one directory counted once, not twice"

        assert stale_links(d, root) == [], "no link farm is not a stale one"
        links = root / "one" / "bfb-parts"
        links.mkdir()
        (links / "part_0.parquet").symlink_to(root / "one" / "archive.tgz")
        (links / "part_1.parquet").symlink_to(root / "gone" / "moved.parquet")
        assert [p.name for p in stale_links(d, root)] == ["part_1.parquet"], \
            "only the link that does not resolve"
        # The shards are all present and verified in this case; the farm is what
        # broke. `inventory` has to carry it separately or the row reads clean.
        row = inventory([d], root, check_digest=False)[0]
        assert row["stale_links"] == 1, "inventory carries it to the doctor"

    # The descriptor itself: `check.py` calls this the consistency gate, so the
    # claims it makes about every entry are asserted rather than assumed.
    hexdigits = set("0123456789abcdef")
    for d in load():
        if d.status != "available":
            assert not d.files, f"{d.name}: declared, but files are pinned"
            continue
        assert d.files, f"{d.name}: available with no files"
        paths = [f.path for f in d.files]
        assert len(paths) == len(set(paths)), f"{d.name}: two files share a path"
        for f in d.files:
            assert len(f.sha256) == 64 and set(f.sha256) <= hexdigits, \
                f"{d.name}/{f.path}: sha256 is not 64 hex characters"
            assert not f.md5 or (len(f.md5) == 32 and set(f.md5) <= hexdigits), \
                f"{d.name}/{f.path}: md5 is not 32 hex characters"
            assert f.bytes > 0, f"{d.name}/{f.path}: no size, so `present` cannot mean anything"
        if d.extract_members:
            assert any(f.role == "archive" for f in d.files), \
                f"{d.name}: extract members but nothing to extract them from"
            assert any(m["role"] == "base" for m in d.extract_members), \
                f"{d.name}: no base member, so `bfb-config` cannot name a path"
    print("datasets.py self-test: ok")
    return 0


def cmd_add(args) -> int:
    """Scaffold a new entry, resolving sizes and digests from the URLs given."""
    ds = load()
    if any(d.name == args.name for d in ds):
        raise SystemExit(f"{args.name} already exists; edit {DESCRIPTOR}")
    entry: dict = {"name": args.name, "role": args.role or "", "status": "available",
                   "format": args.format, "vector_size": args.dim,
                   "distance": args.metric, "gt_shipped": False, "files": []}
    print(f"resolving {len(args.url)} URL(s) via HTTP HEAD and a streamed digest.")
    print(f"{DIM}Digests must be the upstream ones. For Hugging Face, the LFS oid is "
          f"authoritative and this only falls back to hashing a download.{OFF}")
    for u in args.url:
        name = u.rsplit("/", 1)[-1].split("?")[0]
        print(f"  {name} ...", flush=True)
        r = subprocess.run(["curl", "-sIL", u], capture_output=True, text=True)
        size = next((int(ln.split(":")[1]) for ln in reversed(r.stdout.splitlines())
                     if ln.lower().startswith("content-length")), 0)
        entry["files"].append({"path": name, "link": u, "bytes": size,
                               "sha256": digest_from_head(r.stdout) or "TODO",
                               "role": "base"})
    ds_raw = json.loads(DESCRIPTOR.read_text())
    ds_raw.append(entry)
    DESCRIPTOR.write_text(json.dumps(ds_raw, indent=2) + "\n")
    print(f"\nappended {args.name} to {DESCRIPTOR}")
    print("Review it: any `TODO` digest must be replaced with the upstream sha256 "
          "before this is trustworthy.")
    return 0


def main(argv: list[str]) -> int:
    # `--data-dir` is accepted on either side of the subcommand. It defaults to
    # SUPPRESS rather than None so that a subparser which did not see the flag
    # leaves the top-level parser's value alone: argparse otherwise writes the
    # subparser default over it, and `datasets.py --data-dir X fetch` would
    # silently fetch into the default root.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", metavar="DIR", default=argparse.SUPPRESS,
                        help="dataset root to read and write, exported as "
                             "$STRAWMANN_DATA for anything this spawns; point it "
                             "at another tool's dataset directory to share one copy")

    ap = argparse.ArgumentParser(
        parents=[common],
        description="§4.2 dataset management.",
        epilog=f"destination: {DATA_ROOT}\n"
               f"config:      {config_file()}"
               f"{'' if config_file().exists() else ' (not created)'}\n"
               f"order:       --data-dir, $STRAWMANN_DATA, the config file, the default",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def with_names(p, fast=False):
        p.add_argument("datasets", nargs="*", help="limit to these datasets")
        if fast:
            p.add_argument("--fast", action="store_true",
                           help="check size only, skip the digest")
        return p

    lst = with_names(sub.add_parser("list", help="inventory", parents=[common]), fast=True)
    lst.add_argument("--json", action="store_true",
                     help="machine-readable, including measured on-disk bytes; "
                          "what `scripts/doctor.py` reads")
    p = sub.add_parser("info", help="everything known about one dataset", parents=[common])
    p.add_argument("dataset")
    p.add_argument("--fast", action="store_true")
    with_names(sub.add_parser("verify", help="re-checksum, fetch nothing", parents=[common]))
    f = with_names(sub.add_parser("fetch", help="download and verify", parents=[common]))
    f.add_argument("--no-extract", action="store_true",
                   help="do not unpack archives afterwards")
    with_names(sub.add_parser("extract", help="unpack archives", parents=[common]))
    p = sub.add_parser("bfb-config", help="emit a bfb-compatible entry", parents=[common])
    p.add_argument("dataset")
    p.add_argument("--link", action="store_true",
                   help="create templatable symlinks for a sharded dataset, so "
                        "the emitted {i} path actually resolves")
    p = sub.add_parser("add", help="scaffold a new entry", parents=[common])
    p.add_argument("name")
    p.add_argument("url", nargs="+")
    p.add_argument("--format", default="fbin")
    p.add_argument("--dim", type=int)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--role", default="")
    sub.add_parser("self-test", help="unit checks for the pure helpers", parents=[common])

    args = ap.parse_args(argv[1:])
    if not args.cmd:
        # Keep the flags that were given: `datasets.py --data-dir X` with no
        # subcommand should still list X rather than the default root.
        args = ap.parse_args([*argv[1:], "list"])
    use_data_dir(getattr(args, "data_dir", None))
    if not hasattr(args, "fast"):
        args.fast = False
    if not hasattr(args, "json"):
        args.json = False
    if not hasattr(args, "no_extract"):
        args.no_extract = False
    if not hasattr(args, "link"):
        args.link = False
    return {"list": cmd_list, "info": cmd_info, "verify": cmd_verify,
            "fetch": cmd_fetch, "extract": cmd_extract,
            "bfb-config": cmd_bfb_config, "add": cmd_add,
            "self-test": cmd_self_test}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
