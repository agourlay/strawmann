#!/usr/bin/env python3
"""What was measured, on what, and when.

A benchmark result read by someone who did not run it is a claim about two
engines, a dataset and a machine. Without the identities of all four it is not
checkable, and an unchecked claim about someone else's engine is worth nothing
to them. This module collects those identities so that
`bench/harness/report.py` can put them on the page rather than in a person's
memory.

The questions an external reader has to be able to answer from the artifact
alone:

  * **which strawmann?** repository commit, whether the tree was dirty, and a
    hash of the binary that actually ran, because the commit describes the
    source and the hash describes the thing that was executed.
  * **which Qdrant?** image reference and digest, taken from the running
    container rather than from a note, plus the version string it reports.
  * **which data?** dataset name, the local file, and the upstream checksums
    the descriptor pins.
  * **when?** an absolute timestamp per row. `rows.json` merges re-measured
    rows in place, so without one a file can hold rows from different days and
    different binaries with nothing to tell them apart.

Everything here degrades to `None` rather than guessing. A field nobody can
determine is reported as unknown, which a reader can act on; a plausible
default is one they cannot.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
import urllib.request
from enum import StrEnum
from pathlib import Path

ROOT = Path(os.environ.get("STRAWMANN_ROOT", Path(__file__).resolve().parents[2]))


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 10) -> str | None:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def now_iso() -> str:
    """UTC, ISO 8601, seconds. Local time in an artifact that travels is a trap."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_digest(path: Path, limit: int | None = None) -> str | None:
    """sha256 of a file, or of its first `limit` bytes.

    The binary is hashed whole; a multi-gigabyte dataset is identified by its
    descriptor's upstream checksum instead, so nothing here reads 6 GB to
    produce a line of provenance.
    """
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            read = 0
            while chunk := fh.read(1 << 20):
                if limit is not None and read + len(chunk) > limit:
                    h.update(chunk[: limit - read])
                    break
                h.update(chunk)
                read += len(chunk)
        return h.hexdigest()
    except OSError:
        return None


def cmdline(pid: int | None) -> str | None:
    """How the engine was actually launched.

    Provenance, and the practical kind. strawmann's `--connections` decides
    whether W4 can run at all: bfb opens `threads × connections` sockets and a
    server with fewer closes the excess *before* the HTTP/2 preface, so the
    client reports a bare "transport error" with no status. A smoke run hit
    exactly that, and nothing in the result directory said how the server had
    been started, so the row read as an engine failure rather than a
    configuration one.
    """
    if pid is None:
        return None
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    parts = [p for p in raw.split(b"\0") if p]
    return " ".join(p.decode("utf-8", "replace") for p in parts) or None


def affinity(pid: int | None) -> str | None:
    """Which cores the kernel will actually run the engine on.

    Read back from `/proc/<pid>/status`, not taken from the request, and that
    is the whole point. `fullrun.py` gives both engines the same set — `--pin
    --cpus` for strawmANN and `--cpuset-cpus` for the container — but only the
    first of those survives into `engine_cmdline`, so the report showed one
    engine pinned to `4-11` and the other launched as a bare `./qdrant`. A
    reader had no way to tell whether the comparison was tilted, or which way.
    `Cpus_allowed_list` answers it for both, in the same units, because a
    cpuset cgroup constrains the task's allowed mask exactly as `sched_setaffinity`
    does.

    Every thread, grouped by mask and ordered by how many threads share it,
    because a per-thread-pinned engine does not have one answer: `--pin --cpus
    4-11` binds strawmANN's workers and leaves the main thread on every core.
    Reading `/proc/<pid>/status` alone reported that main thread and nothing
    else.

    None when the process is gone or the field is missing, which reads as
    "not recorded" rather than as "every core".
    """
    if pid is None:
        return None
    tasks = Path(f"/proc/{pid}/task")
    masks: dict[str, int] = {}
    try:
        tids = list(tasks.iterdir())
    except OSError:
        return None
    for t in tids:
        try:
            text = (t / "status").read_text()
        except OSError:
            continue          # a thread that exited while we were reading
        for line in text.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                m = line.split(":", 1)[1].strip()
                if m:
                    masks[m] = masks.get(m, 0) + 1
                break
    if not masks:
        return None
    # Most-populated mask first: that is where the work is. Reading only
    # `/proc/<pid>/status` gave the *main* thread's mask, and strawmANN pins
    # its workers individually while leaving the main thread alone — so it
    # reported `0-11` for an engine whose seven workers and io thread were on
    # `4-11`, against Qdrant's container cpuset that really does bind every
    # thread. The comparison's central fairness fact was stated backwards.
    def first_cpu(mask: str) -> int:
        """Lowest CPU in a list like `4-11` or `10`, for ordering."""
        head = mask.split(",")[0].split("-")[0]
        return int(head) if head.isdigit() else -1

    # Ordered by thread count then by *number*, not by string: strawmANN pins
    # each worker to one core, so the masks are "4", "5", ... "11", and a
    # lexicographic sort put 10 and 11 ahead of 4.
    ordered = sorted(masks.items(), key=lambda kv: (-kv[1], first_cpu(kv[0])))
    return ", ".join(f"{m} x{n}" for m, n in ordered)


def running_binary(pid: int | None) -> Path | None:
    """The executable behind `pid`, via `/proc/<pid>/exe`.

    The binary that *ran*, not the one at the default build path: the ISA
    sweep starts `zig-out/isa/<arm>/strawmann-<arm>`, and hashing
    `zig-out/bin/strawmann` while one of those served the rows recorded the
    identity of a binary that was not measured. Falls back to None when the
    pid is unknown or unreadable, so the caller can say "unrecorded" rather
    than guess.
    """
    if pid is None:
        return None
    try:
        target = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None
    # A replaced-on-disk binary reads as "<path> (deleted)"; the link still
    # resolves to the mapped image and can be hashed through /proc.
    return Path(target.replace(" (deleted)", ""))


#: The paths `zig build` reads, and so the only ones whose modification means
#: the binary may not be what `commit` describes.
BUILD_INPUTS = ("src", "build.zig", "build.zig.zon")


def tree_dirty() -> bool | None:
    """Is the working tree modified in a way that changes the binary?

    One definition, because there were two and they disagreed. `fullrun.
    build_identity` stamps the conformance row and asked `git status` over the
    whole tree; this module stamps the perf rows and asks it over BUILD_INPUTS.
    §8 binds a perf row to its conformance run by that string, so a session's
    second dataset — measured after `render` spliced the first one's tables
    into README.md — stamped its rows `abc123` and its conformance run
    `abc123-dirty`, and §8 refused all 24 of them. The refusal was correct; the
    disagreement it reported was not real.

    `None` when git could not be asked, which is not the same as clean.
    """
    status = _run(["git", "status", "--porcelain", "--", *BUILD_INPUTS], cwd=ROOT)
    return bool(status) if status is not None else None


def strawmann_build(pid: int | None = None) -> dict:
    """The engine binary that ran, and the source it came from.

    Both, not either. The commit says what the source was; the binary hash says
    what was executed, and the two disagree the moment someone rebuilds with a
    different `-Dcpu` or forgets to rebuild at all. `dirty` is the flag that
    turns "commit abc1234" from an identity into an approximation.

    With a `pid`, the binary is `/proc/<pid>/exe`, i.e. whatever is actually
    serving; without one it is the default build output and says so.
    """
    commit = _run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT)
    # Over the sources the binary is built from, not the whole tree.
    #
    # `dirty` exists to say "this binary may not be what `commit` describes".
    # Computed over everything, a *successful* run set it: `render` splices the
    # measured tables into README.md and docs/comparison-<dataset>.md, so the
    # first dataset of a two-dataset session left the tree modified and the
    # second stamped its engine `tree dirty` — over documentation the engine
    # cannot depend on, generated by the run that came before it.
    #
    # Scoped to `src/`, `build.zig` and `build.zig.zon`, which is exactly the
    # set `zig build` reads. A change under any of them still flags, which is
    # the case the flag is for; a regenerated table no longer does.
    status = tree_dirty()
    running = running_binary(pid)
    binary = running if running is not None else ROOT / "zig-out/bin/strawmann"
    # Hash through /proc when the pid is known so a binary deleted or rebuilt
    # since the server started is still the one that is hashed.
    hash_path = Path(f"/proc/{pid}/exe") if running is not None else binary
    exists = binary.exists() or (running is not None and hash_path.exists())
    digest = file_digest(hash_path) if exists else None
    out = {
        "commit": commit,
        "dirty": status,
        "binary": str(binary) if exists else None,
        "binary_source": (f"/proc/{pid}/exe" if running is not None
                          else "zig-out/bin/strawmann (default build path; no engine pid)"),
        "binary_sha256": digest[:16] if digest else None,
        "binary_mtime": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(binary.stat().st_mtime))
                         if binary.exists() else None),
        "zig": (_run(["zig", "version"]) or None),
    }
    return out


def probe_path_of(build: dict) -> Path | None:
    """The file to run `--probe` on, given `strawmann_build`'s record.

    `binary_sha256` is taken through `/proc/<pid>/exe`, which is the image
    that is serving whether or not the path it was started from has been
    rebuilt or deleted since; `isa_build` and `build_flags` used to probe the
    on-disk path and so could record the ISA and optimize mode of a binary
    that never served a row. With a pid the /proc link is used (it resolves
    for a deleted target too); without one, the recorded path is all there is.
    """
    src = build.get("binary_source") or ""
    if src.startswith("/proc/"):
        return Path(src)
    return Path(build["binary"]) if build.get("binary") else None


class NetworkPath(StrEnum):
    """How the load generator reached the engine, as the four cases that differ.

    `run.json` keeps the raw value -- `native`, or whatever `docker inspect`
    reports, which is an open set (`host`, `bridge`, a custom network name).
    What any reader actually needs is closed: is there a userland proxy
    relaying every request, which finding 26 measured as a per-request tax on
    one arm and not the other.

    Introduced after a bug this shape would have refused. `report._network_path`
    special-cased `"host"` and let everything else fall to the published-port
    branch, so a *native* binary -- which `--perf` requires, since perf cannot
    attach into a container -- was labelled "a userland proxy relays every
    request". Every perf report told its reader Qdrant paid a tax it was not
    paying. A string compare has no exhaustive form; classifying once and
    handling each case does.
    """

    #: No container at all. `--qdrant-binary`, which every `--perf` run uses.
    native = "native"
    #: A container on the host's network namespace: no mapping, no proxy.
    host = "host"
    #: A container with a published port, so Docker's proxy is in the path.
    published = "published"
    #: Not recorded. Runs before 2026-08-18 published a port, so this is not
    #: the same as "no proxy" and must not be reported as one.
    unknown = "unknown"


def classify_network(raw: str | None) -> NetworkPath:
    """The raw `run.json` value as one of the four cases above."""
    if raw is None or not str(raw).strip():
        return NetworkPath.unknown
    got = str(raw).strip()
    if got == NetworkPath.native:
        return NetworkPath.native
    if got == NetworkPath.host:
        return NetworkPath.host
    # Any other Docker network -- `bridge`, or one someone named -- means a
    # mapped port, which means the proxy.
    return NetworkPath.published


def _binary_predates_commit(repo: Path, binary: Path) -> bool | None:
    """Was this binary built before the commit the checkout is now on?

    True means `commit` cannot be what produced it. False means only that the
    timestamps allow it — a binary newer than the commit still might have come
    from an older one. `None` when either timestamp is unavailable, which is
    the absence of a check rather than a clean verdict.

    See the call site in `qdrant_build`, and `workloads.stale_bfb_binary`,
    which is the same test on the load generator.
    """
    committed = _run(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=repo)
    if not committed or not committed.strip().isdigit():
        return None
    try:
        built = binary.stat().st_mtime
    except OSError:
        return None
    return built < int(committed.strip())


def qdrant_binary_predates_head(binary: Path) -> bool | None:
    """`_binary_predates_commit` for a path, before any server is running.

    `qdrant_build` can only ask once there is a pid to read `/proc/<pid>/exe`
    from, which is after the arm has started. `fullrun.py` validates
    `--qdrant-binary` before the gate and the build, and that is where an
    operator can still act on the answer.
    """
    repo = Path(binary).parent
    for _ in range(4):
        if (repo / ".git").exists():
            return _binary_predates_commit(repo, Path(binary))
        if repo.parent == repo:
            break
        repo = repo.parent
    return None


def qdrant_build(pid: int | None) -> dict:
    """The container that is actually serving, by digest.

    §8.9 pins the comparison target by image digest. Reading it from the
    running container closes the gap between the pin and the process: a
    `docker pull` between the note and the run changes one and not the other.
    """
    out: dict = {"image": None, "digest": None, "version": None,
                 "container": None, "version_source": None, "network": None}
    ids = _run(["docker", "ps", "-q"])
    if not ids:
        # No containers at all. Either nothing is running or this run used
        # `--qdrant-binary`; the native path answers both, and returning here
        # left a `--perf` run — which is *always* the native path — with no
        # Qdrant build identity at all, so §8 refused its rows for carrying
        # none.
        return {**out, **qdrant_native_build(pid)}
    for cid in ids.split():
        info = _run(["docker", "inspect", "-f",
                     "{{.State.Pid}}|{{.Config.Image}}|{{.Image}}|{{.Name}}|"
                     "{{json .NetworkSettings.Ports}}|{{.HostConfig.NetworkMode}}", cid])
        if not info or "|" not in info:
            continue
        cpid, image, digest, name, ports, netmode = \
            (info.split("|") + ["", "", "", "", "", ""])[:6]
        # Against the process's ancestry, not the process: `State.Pid` is the
        # container's PID 1 and the engine visible on the host is its child.
        # See `procstat.ancestors`.
        if pid is not None:
            import procstat
            if not cpid.strip().isdigit() or int(cpid) not in procstat.ancestors(pid):
                continue
        if pid is None and "qdrant" not in image.lower():
            continue
        out.update(image=image, digest=digest, container=name.lstrip("/"),
                   network=netmode.strip() or None)
        out.update(_qdrant_version(cid, image, ports, netmode.strip()))
        return out
    # No container claims this process. Either nothing is running, or the run
    # used `--qdrant-binary` and the engine is a native process — which is the
    # only shape `perf stat` can attach to, and therefore the shape any run
    # with hardware counters has. Identify it the way the strawmANN side is
    # identified: the image that is actually serving, hashed through /proc.
    return {**out, **qdrant_native_build(pid)}


def qdrant_native_build(pid: int | None) -> dict:
    """A Qdrant started from a binary rather than an image.

    §8.9 pins the comparison target by image digest and this cannot: a path is
    not a digest, and nothing here says where the binary came from. So it
    records what it can — the sha256 of the file that is serving, read through
    `/proc/<pid>/exe` so a rebuild since start-up does not change the answer,
    and the version the server reports over REST — and `image` stays null so a
    reader can tell the two provenances apart rather than seeing a weaker one
    wearing the shape of the stronger.
    """
    out: dict = {}
    if pid is None:
        return out
    exe = Path(f"/proc/{pid}/exe")
    try:
        target = exe.resolve()
    except OSError:
        return out
    if "qdrant" not in target.name.lower():
        return out
    digest = file_digest(exe)
    out.update(binary=str(target), binary_sha256=digest[:16] if digest else None,
               container=None, network="native")
    # Where the binary came from, when it can be known. An image digest names
    # a build anyone can pull; a path names a file on one machine, and the
    # sha256 says only that it has not changed since. If the binary sits inside
    # a git tree — `target/release/qdrant` under a checkout is the usual shape —
    # then the commit is the missing half, and `dirty` is what turns it from an
    # identity into an approximation. Same treatment the strawmANN side gets,
    # for the same reason.
    # `target/<profile>/qdrant` is cargo's layout, and the profile is part of
    # the build identity: `perf` here is `release` without LTO and with 256
    # codegen units, which cost 20-30% more instructions per
    # query than the fat-LTO `release` binary it replaced.
    if target.parent.parent.name == "target":
        out["cargo_profile"] = target.parent.name
    repo = target.parent
    for _ in range(4):
        if (repo / ".git").exists():
            out["commit"] = _run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo)
            out["dirty"] = bool(_run(["git", "status", "--porcelain"], cwd=repo))
            out["source"] = str(repo)
            # And whether `commit` can be the binary's source at all. `dirty`
            # says "this may not be what the commit describes"; it cannot say
            # "this is certainly not", and the difference showed up on
            # 2026-09-07: `target/release/qdrant` was built 09-03 18:17 and the
            # checkout had since moved to a commit dated 09-04 16:57, so the
            # run recorded a commit that postdates the binary it names. Only
            # `binary_sha256` revealed the file was in fact the previous run's,
            # byte for byte.
            #
            # The same one-way test `workloads.stale_bfb_binary` applies to the
            # load generator, and cheap for the same reason: a binary older
            # than the commit definitely did not come from it, while a newer one
            # only might have. `None` when either timestamp cannot be had —
            # the absence of a check is not a verdict.
            out["binary_predates_commit"] = _binary_predates_commit(repo, target)
            break
        if repo.parent == repo:
            break
        repo = repo.parent
    for port in (6333,):
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:{port}/", timeout=5) as r:
                got = json.loads(r.read())
            out.update(version=str(got.get("version")) or None,
                       version_source=f"server REST :{port}")
            break
        except Exception:
            continue
    return out


def _qdrant_version(cid: str, image: str, ports: str, netmode: str = "") -> dict:
    """The running Qdrant's version, and how it was learnt.

    This used to `docker exec` a `wget`/`curl` inside the container. Neither
    exists in `qdrant/qdrant:v1.19.0` — it ships `sh` and nothing else — so the
    probe returned nothing every time, `version` stayed null, and §8 refused
    every Qdrant row for carrying no build identity. Silently: a null version
    looks the same as a container that was not running.

    So: ask over the published REST port from the host, where the tooling is
    ours, and fall back to the image tag. The tag is weaker evidence than the
    server's own answer, which is why `version_source` records which one this
    was rather than presenting them as the same fact. The digest is pinned
    either way (§8.9), so the fallback loses the server's self-report, not the
    identity of what ran.

    Under `--network host` there is no port mapping to read: the container binds
    the host's 6333 directly, and `NetworkSettings.Ports` is empty. Without the
    branch for it this function would find no published port, fall through to
    the tag, and quietly downgrade every Qdrant row's build identity the moment
    the proxy was removed -- the same silent-null failure this docstring was
    written about, reintroduced by the fix for something else.
    """
    host_port = "6333" if netmode == "host" else None
    try:
        mapping = json.loads(ports) if ports else None
        binds = (mapping or {}).get("6333/tcp") or []
        if binds and not host_port:
            host_port = binds[0].get("HostPort")
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass

    if host_port:
        try:
            with urllib.request.urlopen(f"http://localhost:{host_port}/", timeout=5) as r:
                v = json.loads(r.read().decode()).get("version")
            if v:
                where = ("REST / on the host's own port (--network host)"
                         if netmode == "host" else "REST / on the published port")
                return {"version": v, "version_source": where}
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    # `qdrant/qdrant:v1.19.0` -> `1.19.0`. Only a tag that looks like a version
    # is used: `latest` or `dev` identifies nothing and must stay null rather
    # than become a build identity that cannot be checked.
    tag = image.rpartition(":")[2] if ":" in image else ""
    m = re.fullmatch(r"v?(\d+\.\d+\.\d+)", tag.strip())
    if m:
        return {"version": m.group(1), "version_source": f"image tag {tag!r}"}
    return {"version": None, "version_source": None}


def dataset_identity(name: str = "sift1m") -> dict:
    """The corpus, by the descriptor's pinned upstream checksums.

    §4.3: "Checksums of base, query, and GT files are recorded in every result
    row." The descriptor is the one definition of those (`datasets.json`), so
    they are read from it rather than restated here.
    """
    out: dict = {"name": name, "checksums": None}
    desc = ROOT / "conformance/datasets/datasets.json"
    if not desc.exists():
        return out
    try:
        d = json.loads(desc.read_text())
    except json.JSONDecodeError:
        return out
    # The descriptor is a list of datasets; `datasets.py` is its only other
    # reader and this must not become a second, divergent understanding of it.
    entries = d if isinstance(d, list) else d.get("datasets", [])
    entry = next((x for x in entries
                  if isinstance(x, dict) and x.get("name") == name), None)
    if entry is None:
        return out

    out.update(dim=entry.get("vector_size"), metric=entry.get("distance"),
               n=entry.get("n"), n_queries=entry.get("n_queries"),
               gt_shipped=entry.get("gt_shipped"), format=entry.get("format"))
    files = entry.get("files") or entry.get("parts", {}).get("files") or []
    checksums = {f["path"]: f["sha256"] for f in files
                 if isinstance(f, dict) and f.get("sha256")}
    if checksums:
        # A 26-part dataset would put 26 lines of hex on the page; the count and
        # the first are enough to identify the pin, and `datasets.json` holds
        # the rest.
        out["checksums"] = dict(list(checksums.items())[:4])
        out["checksum_count"] = len(checksums)
    return out


# --------------------------------------------------------------------------
# Memory bandwidth: the one number every qps figure is bounded by
# --------------------------------------------------------------------------

#: The server's startup banner (`src/main.zig`), one line:
#:   memory bandwidth       : 47.3 GB/s aggregate (24 threads) · 25.1 GB/s single core (53% of bus)
BANNER_RE = re.compile(
    r"memory bandwidth\s*:\s*([0-9.]+) GB/s aggregate \((\d+) threads\)"
    r"\s*·\s*([0-9.]+) GB/s single core \((\d+)% of bus\)")
#: `strawmann --probe`, two lines:
#:   bandwidth aggregate    : 47.3 GB/s (24 threads)
#:   bandwidth single core  : 25.1 GB/s (53% of bus)
PROBE_AGG_RE = re.compile(r"bandwidth aggregate\s*:\s*([0-9.]+) GB/s \((\d+) threads\)")
PROBE_ONE_RE = re.compile(r"bandwidth single core\s*:\s*([0-9.]+) GB/s \((\d+)% of bus\)")
#: Both the banner and `--probe` print `isa build : <name>`; `isa_sweep.py`
#: checks the same line to confirm which arm is serving.
ISA_BUILD_RE = re.compile(r"isa build\s*:\s*(\S+)")
#: `build mode : ReleaseFast`. §9 quotes results from ReleaseFast only, and a
#: Debug binary is otherwise indistinguishable from a release one in every
#: record a run keeps: same path, same `isa build: native`, same commit.
BUILD_MODE_RE = re.compile(r"build mode\s*:\s*(\S+)")
#: `dispatch tier : avx512`. The arm's *name* is a build label; this is the tier
#: the kernels actually reached, which `dispatch.active` fixes at comptime.
DISPATCH_TIER_RE = re.compile(r"dispatch tier\s*:\s*(\S+)")
#: `vnni (u8xi8) : true` in the banner, `compiled vnni (u8xi8) : true` in
#: `--probe`; one pattern reads both.
VNNI_RE = re.compile(r"vnni \(u8xi8\)\s*:\s*(true|false)")


def parse_isa_build(text: str) -> str | None:
    m = ISA_BUILD_RE.search(text)
    return m.group(1) if m else None


def parse_build_flags(text: str) -> dict:
    """Build mode, dispatch tier and VNNI out of a banner or `--probe`.

    Absent keys rather than nulls: a binary built before these lines existed
    should read as "this run does not say", which the report renders as a
    warning, and not as "it was fine".
    """
    out: dict = {}
    for key, rx in (("optimize", BUILD_MODE_RE),
                    ("dispatch_tier", DISPATCH_TIER_RE)):
        m = rx.search(text)
        if m:
            out[key] = m.group(1)
    m = VNNI_RE.search(text)
    if m:
        out["vnni"] = m.group(1) == "true"
    return out


def _engine_texts(server_log: Path | None, binary: Path | None):
    """What the engine said about itself: the banner first, then `--probe`.

    A generator rather than one string, because "the log exists but predates
    the line I want" has to fall through to the probe exactly as it always did
    for `isa build`.
    """
    if server_log is not None and server_log.exists():
        try:
            yield server_log.read_text()
        except OSError:
            pass
    if binary is None or not binary.exists():
        return
    if str(binary) in _PROBE_TEXT:  # `memory_bandwidth` already probed this binary
        yield _PROBE_TEXT[str(binary)]
        return
    try:
        p = subprocess.run([str(binary), "--probe"], capture_output=True, text=True,
                           timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return
    yield p.stdout + p.stderr


def isa_build(server_log: Path | None = None, binary: Path | None = None) -> str | None:
    """Which ISA build served: the banner in `server_log`, else `binary --probe`.

    `results.py` files a run under `run.json`'s `isa_build` and defaults it to
    `native`, so an ISA-sweep arm whose run never said otherwise ingested as
    the native build. Read from the server rather than from a label.
    """
    for text in _engine_texts(server_log, binary):
        got = parse_isa_build(text)
        if got:
            return got
    return None


def build_flags(server_log: Path | None = None, binary: Path | None = None) -> dict:
    """Optimize mode, dispatch tier and VNNI, from the same two sources.

    Separate from `isa_build` because the arm name and the build are different
    claims: `isa build: native` was true of a Debug binary that served a whole
    run, and nothing in the run said the mode. §9 only licenses numbers from
    ReleaseFast, so the report needs the mode to be able to refuse them.
    """
    for text in _engine_texts(server_log, binary):
        got = parse_build_flags(text)
        if got:
            return got
    return {}


#: `strawmann --probe` output by binary path: `memory_bandwidth` runs it once
#: per run and `isa_build` reads the same output rather than probing again.
_PROBE_TEXT: dict[str, str] = {}


def _full_affinity() -> tuple[str, str | None]:
    """Widen this process's affinity to every CPU, for the probe to inherit.

    `fullrun.py` runs the harness under `taskset -c <client cpus>`, and a
    child inherits that, so `strawmann --probe` measured the client cpuset's
    bandwidth and recorded it as the machine's. The probe is a host property
    and has to see the host. Returns the cpuset actually in effect afterwards
    (a cpuset-limited container clips it) and the error if the widening
    failed, so the record says what the probe really ran on.
    """
    err = None
    try:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except (OSError, AttributeError) as e:
        err = f"{type(e).__name__}: {e}"
    return probe_cpuset(), err


def probe_cpuset() -> str:
    """The CPUs this process may run on, spelled as `taskset -c` spells them."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return f"0-{(os.cpu_count() or 1) - 1}"
    runs: list[str] = []
    for c in cpus:
        if runs and int(runs[-1].split("-")[-1]) == c - 1:
            runs[-1] = f"{runs[-1].split('-')[0]}-{c}"
        else:
            runs.append(str(c))
    return ",".join(runs)


def parse_bandwidth_banner(text: str) -> dict | None:
    """The bandwidth the *server* measured when it started, from its log."""
    m = BANNER_RE.search(text)
    if not m:
        return None
    return {"aggregate_gbps": float(m.group(1)), "threads": int(m.group(2)),
            "single_core_gbps": float(m.group(3)),
            "single_core_pct_of_bus": int(m.group(4))}


def parse_bandwidth_probe(text: str) -> dict | None:
    a, o = PROBE_AGG_RE.search(text), PROBE_ONE_RE.search(text)
    if not a:
        return None
    out = {"aggregate_gbps": float(a.group(1)), "threads": int(a.group(2))}
    if o:
        out.update(single_core_gbps=float(o.group(1)),
                   single_core_pct_of_bus=int(o.group(2)))
    return out


def memory_bandwidth(server_log: Path | None = None,
                     binary: Path | None = None) -> dict | None:
    """The machine's memory bandwidth, as the engine measured it.

    §7.2's hardware baseline: a qps figure without the bandwidth of the machine
    it ran on is not comparable to anything, so the report opens with it. It is
    a property of the host, not of the engine under test, which is why the
    Qdrant arm records it too: the same `sysinfo.probe` runs either from the
    strawmann server's own startup (`server.log`, when `fullrun.py` launched
    it, which is the measurement "at startup" in the strict sense) or from
    `strawmann --probe` at the start of the run, and the record says which.
    """
    if server_log is not None and server_log.exists():
        try:
            bw = parse_bandwidth_banner(server_log.read_text())
        except OSError:
            bw = None
        if bw:
            bw.update(source="strawmann startup banner", log=str(server_log),
                      measured_at=now_iso())
            return bw
    binary = binary or ROOT / "zig-out/bin/strawmann"
    if not binary.exists():
        return None
    # Widened in the parent and restored after, so the child inherits every
    # CPU and the record holds the affinity actually in effect.
    before = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    cpuset, aff_err = _full_affinity()
    try:
        p = subprocess.run([str(binary), "--probe"], capture_output=True, text=True,
                           timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        if before is not None:
            try:
                os.sched_setaffinity(0, before)
            except OSError:
                pass
    _PROBE_TEXT[str(binary)] = p.stdout + p.stderr
    bw = parse_bandwidth_probe(p.stdout + p.stderr)
    if bw:
        bw.update(source="strawmann --probe at run start", measured_at=now_iso(),
                  cpuset=cpuset)
        if aff_err:
            bw["affinity_error"] = aff_err
    return bw


def host() -> dict:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "memory_bytes": mem,
            "kernel": platform.release(), "hostname": platform.node()}


def collect(label: str, uri: str, engine_pid: int | None,
            engine_comm: str | None, dataset: str = "sift1m",
            results: Path | None = None) -> dict:
    """Everything an outside reader needs to identify this run.

    The engine side is filled in for whichever engine is running: asking Docker
    about strawmann or git about Qdrant would produce confident nonsense.

    `results` is the directory the run writes to. It defaults to
    `bench/results/<label>`, but `isa_sweep.py` and `qdrant_ab.py` point
    `RESULTS_DIR` elsewhere, and the server log looked for under the default
    path was never there for them, so their arms probed instead of reading
    the banner.
    """
    if results is None:
        results = Path(os.environ.get("RESULTS_DIR", ROOT / "bench/results" / label))
    out = {
        "label": label,
        "uri": uri,
        "started": now_iso(),
        "host": host(),
        "dataset": dataset_identity(dataset),
        "engine_comm": engine_comm,
    }
    probe = None
    if engine_comm and engine_comm.startswith("qdrant"):
        out["qdrant"] = qdrant_build(engine_pid)
    else:
        out["strawmann"] = strawmann_build(engine_pid)
        probe = probe_path_of(out["strawmann"])
    # `fullrun.py` writes the strawmann server's log beside its results; a
    # Qdrant arm has no such log and is probed instead. Same machine, same
    # number, said differently. The probe is the serving image where there is
    # one, so `isa_build` reads this run's output rather than probing again.
    out["host"]["memory_bandwidth"] = memory_bandwidth(server_log=results / "server.log",
                                                       binary=probe)
    out["engine_cmdline"] = cmdline(engine_pid)
    out["engine_affinity"] = affinity(engine_pid)
    # What the harness *asked* for, beside what the kernel reports. The two
    # answer different questions and a reader needs both: `engine_affinity` is
    # observed per thread and can be surprising (a per-thread-pinned engine
    # leaves its main thread unbound), while this is the one number
    # `fullrun.py` gave to both arms and therefore the statement that the
    # comparison was set up fairly. Absent for a run driven by something that
    # did not set it, which is not the same as "every core".
    cpus = os.environ.get("BENCH_SERVER_CPUS")
    if cpus:
        out["server_cpus_requested"] = cpus
    return out


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "strawmann"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "sift1m"
    print(json.dumps(collect(label, "http://localhost:6334", None, label, dataset), indent=2))
