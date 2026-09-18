# Dataset management

§4.2 requires the dataset set be "static, versioned, checksummed". This is how
that is arranged, and how to add a new one.

## One descriptor

`conformance/datasets/datasets.json` is the single source of truth. It is read by:

- `conformance/datasets/datasets.py`, which fetches, verifies and extracts
- the Rust conformance binary, via `include_str!`, so `cargo run -- datasets`
  and the fp64 oracle agree with the fetcher about dimension and metric
- bfb, through `datasets.py bfb-config`

Before this there were three places to edit and they had already drifted: a
`manifest.tsv` pinning files for two datasets, a `DATASETS` const in Rust
declaring six, and a converter with dbpedia's layout hardcoded. Nothing
reported the gap, because nothing compared them.

Each entry carries a `status`:

| status | meaning |
|---|---|
| `available` | every file is pinned by size and upstream digest, and can be fetched |
| `declared` | §4.2 names it, nobody has pinned its files yet |

So the spec table and the fetchable set live in one file, and the difference
between them is visible in `datasets.py list` rather than invisible across two.

## What is pinned

| dataset | N | dim | metric | source | what it is for |
|---|---|---|---|---|---|
| `sift1m` | 1M | 128 | euclid | fvecs archive | §4.2's CI tier |
| `dbpedia-openai-1m` | 1M | 1536 | cosine | 26 parquet shards | §4.2's headline tier |
| `dbpedia-openai-100K-1536-angular` | 100k | 1536 | cosine | tgz | the headline dimensionality at a tenth of the size and a sixteenth of the download |
| `laion-small-clip` | 100k | 512 | cosine | tgz | filtered search: every query carries a float range condition |
| `h-and-m-2048-angular-filters` | 105k | 2048 | cosine | tgz | filtered search at high dim, over a 24-field product payload |

The last three come from [ann-filtered-benchmark][afb], which is also where
vector-db-benchmark takes them from. Each is one `.tgz` holding `vectors.npy`,
a `tests.jsonl` query set, and (except dbpedia-100K) a `payloads.jsonl`.

[afb]: https://github.com/qdrant/ann-filtering-benchmark-datasets

Two things about them are easy to get wrong, so `datasets.py info` prints both:

- **`laion-small-clip` is float16 on disk**, the only entry here that is not
  f32. A reader that assumes f32 gets half the rows at twice the dimension and
  no error to say so.
- **Their shipped ground truth is *filtered* ground truth**: `tests.jsonl`
  gives k=10 neighbours (25 for h-and-m) computed under that query's own
  condition. It is not §4.3's unfiltered k=100 and cannot be substituted for
  it.

What they are wired into today is bfb, through `bfb-config`: bfb's `tar`
reader takes the extracted directory and reads all three files, so these
datasets can drive load and filtered search now. They are **not** wired into
the relevance path. That is `fbin` plus our own fp64 ground truth (§4.3), and
nothing converts `.npy` yet. Recall on them needs that converter first.

## Choosing the dataset directory

These corpora are large, public, and used by more than one tool, so the root is
overridable from four places. Highest first, and each one is a different scope:

| source | scope |
|---|---|
| `--data-dir DIR` | this run |
| `$STRAWMANN_DATA` | this shell |
| `[paths] datasets` in `${XDG_CONFIG_HOME:-~/.config}/strawmann/config.toml` | this user |
| `$STRAWMANN_CACHE/datasets`, itself under `${XDG_CACHE_HOME:-~/.cache}` | always |

The config file is where a lasting choice goes:

```toml
# ~/.config/strawmann/config.toml
[paths]
cache = "~/.cache/strawmann"
datasets = "~/Documents/datasets"
qdrant_storage = "~/.cache/strawmann/qdrant-storage"
```

Every key is optional and an absent one falls back to the layout. A file that
cannot be parsed, or that carries a key nobody reads, is an **error**: a typo
in the one file that says where 16 GB lives would otherwise fall back to the
default root, which looks exactly like "nothing is fetched". `scripts/doctor.py`
prints each root and which of the four sources it came from.

For one run, the flag:

```sh
conformance/datasets/datasets.py --data-dir /mnt/big/datasets fetch
bench/harness/fullrun.py --data-dir /mnt/big/datasets --server-cpus 4-11 --client-cpus 0-3
```

`--data-dir` exports `$STRAWMANN_DATA`, so a tool that spawns another:
`fullrun.py` runs `workloads.py` and `recall.py`, which run bfb and the
conformance binary, passes the choice down without being asked to. Choosing
it once, at the outermost tool, is the whole point: the failure it prevents is
fetching 16 GB into one directory and measuring against another.

Where an entry has an `extract.dir`, that directory is vector-db-benchmark's
own path for the same dataset, so a root shared with that tool holds one copy
of the files rather than two identical ones under different names.

## Commands

```sh
datasets.py list                  # inventory, with what is on disk
datasets.py list --fast           # size only, skip digests
datasets.py list --json           # the same, machine-readable; the doctor reads this
datasets.py info sift1m           # everything known about one
datasets.py fetch                 # download and verify everything available
datasets.py fetch sift1m          # just one
datasets.py verify                # re-checksum, fetch nothing, fail if incomplete
datasets.py extract               # unpack archives named in the descriptor
datasets.py bfb-config <name>     # emit a bfb-compatible entry
datasets.py add <name> <url>...   # scaffold a new entry
datasets.py --data-dir DIR ...    # any of the above, against another root
```

`datasets.py --help` ends with the root it resolved, the config file it read,
and the order the two were consulted in.

`list` reports two sizes and they are different numbers on purpose: **download**
is what the descriptor pins, **on disk** is what the directories actually hold.
The archive, whatever was unpacked from it, and anything converted since. SIFT1M
is a 160 MB download and 1.2 GB of directory, because the fbin the fp64 oracle
read and the cached ground truth live there too. `scripts/doctor.py` shows the
same per dataset, with a total.

## Adding a dataset

```sh
conformance/datasets/datasets.py add my-dataset \
    https://host/base.fbin https://host/queries.fbin \
    --format fbin --dim 768 --metric cosine --role "why this dataset exists"
```

`add` resolves each URL's size and, where the host exposes it, the upstream
digest. Review the entry afterwards: **any `TODO` digest must be replaced with
the real upstream sha256 before the entry is trustworthy.**

Digests are the *upstream* Git-LFS object ids, not digests computed after
downloading. A self-computed digest only proves the file has not rotted on our
disk; the upstream one also proves we fetched what the publisher published.

Not every host publishes a sha256. Google Cloud Storage, which serves the
ann-filtered-benchmark archives, publishes an md5 and nothing else, so those
entries carry both: an `md5` that is upstream's, checked once as the bytes
arrive, and a `sha256` computed from the very bytes that satisfied it. The
upstream anchor and the on-disk rot check stay two separate claims made by two
separate digests, rather than one self-computed number pretending to be both.
`verify` re-checks the sha256, which is the one that is about our disk.

Then:

```sh
datasets.py fetch my-dataset
cargo run --release -- oracle --base ... --queries ... --metric cosine --k 100 \
    --dataset my-dataset --out gt.json
```

## Sharded datasets

`parts` describes a numbered family read as one row space:

```json
"parts": { "count": 26, "start": 0,
           "files": [ {"path": "...", "link": "...", "bytes": 0, "sha256": "..."} ] }
```

bfb templates shard paths with `{i}`. We cannot, because Hugging Face appends a
per-shard content hash (`train-00007-of-00026-1cd8b3ca61f2ab2b.parquet`) that no
template reproduces, and each shard needs its own digest anyway. So the list is
explicit, and **its order is load-bearing**: row order defines point ids, and
point ids are what ground truth indexes into. Sorting the directory listing
instead would make ids depend on how the files landed on disk.

`datasets.py bfb-config <name> --link` builds a directory of symlinks with
templatable names so the emitted `{i}` template resolves. Without `--link` it
warns rather than emitting a path that looks plausible and matches nothing: bfb
would report an empty dataset rather than a broken path, which is the harder
failure to notice.

## Relationship to bfb

The schema is deliberately close to bfb's `src/dataset/config.rs`, itself
compatible with vector-db-benchmark's `datasets.json`: `name`, `format`,
`vector_size`, `distance`, `parts`.

It is a superset. bfb has no checksums and §4.2 requires them; bfb's config also
uses `deny_unknown_fields`, so our extra keys would be rejected outright. That
is why `bfb-config` emits a projection instead of the file itself, and why it
adds no explanatory keys to its output.

bfb supports formats we have not needed yet: `h5`, `sparse`, `npy`, `jsonl`.
The descriptor records `format` for every entry, so wiring one up is a reader,
not a redesign.

An archive is not a dataset: bfb's `tar` reader opens the *directory* holding
`vectors.npy`, not the `.tgz` we fetched. `bfb-config` therefore emits where
the base member lands after extraction, which is also why `extract` records
every member by path and size. A half-unpacked directory is then visible
rather than mistaken for a present dataset.
