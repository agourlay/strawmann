#!/usr/bin/env python3
"""Is this machine set up, and what can it actually produce?

    scripts/doctor.py           the full report
    scripts/doctor.py --quick   skip the bandwidth probe and dataset digests
    scripts/doctor.py --json    machine-readable
    scripts/doctor.py --data-dir DIR    report on another dataset root

The paths it reports come from `bench/harness/paths.py`, which resolves each
root from `--data-dir`, then the environment, then a `[paths]` table in
`${XDG_CONFIG_HOME:-~/.config}/strawmann/config.toml`, then the default layout.

Three questions, in order:

  1. **Installation.** Is every tool present, at the pinned version, and is
     everything built that needs building?
  2. **Host.** Does §7.1's discipline hold, and how fast is the memory system?
  3. **Verdict.** Given 1 and 2, what class of number can this host produce?

The third is the point, and it is why this is not just `check.py` with more
lines. `scripts/check.py` asks "is the code correct?" and `bench/setup.py check`
asks "is the host disciplined?". Neither answers "can I trust a number I measure
here today", which is the question someone actually has, and which depends on
the toolchain, the build, the host and the datasets at once.

The host figures come from `strawmann --probe`, not from a reimplementation
here: the binary already measures bandwidth at startup, and a second copy of
that logic would drift from the first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")

sys.path.insert(0, str(ROOT / "bench/harness"))
import paths  # noqa: E402  (after ROOT, and stdlib-only itself)
import workloads  # noqa: E402  (for BFB_PIN; stdlib-only, same as paths)

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = DIM = BOLD = OFF = ""

OK, WARN, BAD = "ok", "warn", "FAIL"
MARK = {OK: f"{GREEN}ok  {OFF}", WARN: f"{YELLOW}warn{OFF}", BAD: f"{RED}FAIL{OFF}"}

#: §9 pins the compiler: "a benchmark that silently changes compiler is not a
#: benchmark." Read from build.zig.zon so this file cannot disagree with it.
def pinned_zig() -> str:
    m = re.search(r"minimum_zig_version\s*=\s*\"([^\"]+)\"",
                  (ROOT / "build.zig.zon").read_text())
    return m.group(1) if m else "unknown"


@dataclass
class Item:
    name: str
    status: str
    detail: str = ""
    hint: str = ""
    #: Why this item is BAD, for a verdict line that has to name the cause.
    #: Matching on `detail` text would work until someone rewords it, and the
    #: wrong verdict here is worse than none: "a dataset failed its digest"
    #: printed for an intact dataset sends the reader to re-download 14 GB.
    kind: str = ""


@dataclass
class Section:
    title: str
    items: list[Item] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "",
            kind: str = "") -> None:
        self.items.append(Item(name, status, detail, hint, kind))

    @property
    def worst(self) -> str:
        if any(i.status == BAD for i in self.items):
            return BAD
        if any(i.status == WARN for i in self.items):
            return WARN
        return OK


def run(argv: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return r.returncode, ANSI.sub("", (r.stdout or "") + (r.stderr or ""))
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def tool_version(exe: str, *args: str) -> str | None:
    if not shutil.which(exe):
        return None
    code, out = run([exe, *args], timeout=30)
    return out.strip().splitlines()[0] if code == 0 and out.strip() else None


# --------------------------------------------------------------------------

def section_toolchain() -> Section:
    s = Section("Toolchain")
    want = pinned_zig()
    zig = tool_version("zig", "version")
    if zig is None:
        s.add("zig", BAD, "not on PATH", f"install {want} exactly (§9)")
    elif zig != want:
        # Not a warning. §9 makes an unpinned compiler disqualifying, because
        # LLVM's autovectoriser and its AVX-512 cost model both move between
        # releases, which turns the §7.5 ISA matrix into noise.
        s.add("zig", BAD, f"{zig}, pinned is {want}", "results from another zig are not comparable")
    else:
        s.add("zig", OK, zig)

    for exe, args, why, fatal in (
        ("cargo", ("--version",), "conformance harness (§8.5 needs the Rust qdrant-client)", True),
        ("uv", ("--version",), "report tooling (bench/pyproject.toml)", False),
        ("objdump", ("--version",), "§6.6.3 disassembly diffing", False),
        ("curl", ("--version",), "dataset fetch", True),
        ("docker", ("--version",), "Qdrant comparison arm", False),
    ):
        v = tool_version(exe, *args)
        if v:
            s.add(exe, OK, v[:60])
        else:
            s.add(exe, BAD if fatal else WARN, "not on PATH", why)
    return s


def section_build() -> Section:
    s = Section("Build artifacts")
    server = ROOT / "zig-out/bin/strawmann"
    if server.exists():
        s.add("strawmann", OK, f"{server.stat().st_size / 1e6:.1f} MB")
    else:
        s.add("strawmann", BAD, "not built", "zig build -Doptimize=ReleaseFast")

    isa_dir = ROOT / "zig-out/isa"
    arms = sorted(p.name for p in isa_dir.iterdir()) if isa_dir.is_dir() else []
    built = [a for a in arms if (isa_dir / a / f"strawmann-{a}").exists()]
    if len(built) >= 8:
        s.add("ISA arms", OK, f"{len(built)} arms: {', '.join(built)}")
    elif built:
        s.add("ISA arms", WARN, f"{len(built)} of 8", "zig build bench-isa")
    else:
        s.add("ISA arms", WARN, "none", "zig build bench-isa, for §7.5")

    bfb = os.environ.get("BFB")
    cands = [Path(bfb)] if bfb else []
    cands += [ROOT.parent / "bfb/target/release/bfb", Path.home() / "Workspace/bfb/target/release/bfb"]
    hit = next((c for c in cands if c.exists()), None)
    if hit:
        code, out = run(["git", "-C", str(hit.parents[2]), "rev-parse", "--short", "HEAD"], 20)
        rev = out.strip() if code == 0 else "?"
        # `workloads.BFB_COMMIT`, not a copy of it. This held the literal
        # `8b6dec0a`, the upstream commit the pin was based on while the
        # `--rps` reaping fix (findings 32) made the pin a one-patch fork. So
        # doctor warned "a different load generator is a different benchmark"
        # at the checkout `check_bfb_pin` requires, and would have called the
        # wrong one pinned — the failure mode a second source of truth has.
        pin = workloads.BFB_COMMIT
        if rev and pin.startswith(rev):
            s.add("bfb", OK, f"dev @ {rev} (pinned)")
        else:
            s.add("bfb", WARN, f"at {rev}, pinned is {pin[:len(rev) or 7]}",
                  "a different load generator is a different benchmark")
    else:
        s.add("bfb", BAD, "not built",
              f"git clone -b dev https://github.com/qdrant/bfb && "
              f"git checkout {workloads.BFB_COMMIT} && cargo build --release")

    venv = ROOT / "bench/.venv"
    s.add("bench venv", OK if venv.is_dir() else WARN,
          "present" if venv.is_dir() else "absent",
          "" if venv.is_dir() else "uv sync --project bench, for the HTML report")
    return s


def section_host(quick: bool) -> tuple[Section, dict]:
    s = Section("Host discipline (§7.1)")
    code, out = run([sys.executable, str(ROOT / "bench/setup.py"), "check"], 120)
    checks = 0
    def label(text: str) -> str:
        """A short name for a gate line.

        The gate prints free prose ("machine is NOT quiescent: load 8.22 ..."),
        so the first token is sometimes a word like "machine" and sometimes a
        `key=value`. Take the key when there is one, else the first few words.
        """
        head = text.split(":")[0].split(",")[0].strip()
        if "=" in head:
            return head.split("=")[0]
        return " ".join(head.split()[:3])[:22]

    for line in out.splitlines():
        t = line.strip()
        if t.startswith("ok "):
            s.add(label(t[3:]), OK, t[3:].strip()[:70])
            checks += 1
        elif t.startswith("FAIL"):
            s.add(label(t[4:]), BAD, t[4:].strip()[:70])
            checks += 1
    if not checks:
        s.add("gate", BAD, "bench/setup.py check produced nothing", out.strip()[:120])
    m = re.search(r"environment hash: ([0-9a-f]+)", out)
    return s, {"gate_pass": code == 0, "env_hash": m.group(1) if m else None}


def section_perf(quick: bool) -> tuple[Section, dict]:
    s = Section("Host performance")
    facts: dict = {}
    server = ROOT / "zig-out/bin/strawmann"
    if not server.exists():
        s.add("probe", WARN, "server not built", "zig build -Doptimize=ReleaseFast")
        return s, facts
    if quick:
        s.add("probe", WARN, "skipped (--quick)")
        return s, facts
    code, out = run([str(server), "--probe"], 180)
    if code != 0:
        s.add("probe", BAD, "strawmann --probe failed", out.strip()[:120])
        return s, facts
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        facts[k] = v
        s.add(k[:22], OK, v)

    agg = facts.get("bandwidth aggregate", "")
    single = facts.get("bandwidth single core", "")
    if agg and single:
        try:
            frac = float(re.search(r"\((\d+)%", single).group(1))
            # A single core reaching nearly all of the bus means the bus is
            # narrow relative to the core, so adding workers buys little. Worth
            # saying, because it changes what a throughput number means.
            if frac >= 80:
                s.add("scaling headroom", WARN, f"one core already reaches {frac:.0f}% of the bus",
                      "extra workers cannot buy much bandwidth on this host")
        except (AttributeError, ValueError):
            pass
    return s, facts


def section_paths() -> Section:
    """Where this project reads and writes, and what decided each of those.

    The doctor is the first thing run on a new machine, and two of the first
    questions are "where did the 16 GB go" and "what do I edit to move it".
    `bench/harness/paths.py` is the definition; this reports what it resolved,
    including `paths.SOURCES`, so the two cannot disagree.

    The source is the part that is invisible otherwise. A default, a value from
    the config file, an inherited `$STRAWMANN_DATA` and a `--data-dir` given to
    this run all print the same path and are four different things to debug.

    A malformed config file never reaches here: `paths.py` refuses it at import
    with the file and the parse error, which is a better failure than a report
    quietly describing the default root.
    """
    s = Section("Paths")
    cfg = paths.CONFIG_FILE
    if cfg.exists():
        keys = ", ".join(sorted(paths.CONFIG)) or "no [paths] keys"
        s.add("config file", OK, f"{cfg}  {DIM}(sets {keys}){OFF}")
    else:
        s.add("config file", OK, f"{cfg}  {DIM}(not created; write a [paths] table "
                                 f"there to make a root stick){OFF}")
    for name, path, key in (("cache root", paths.CACHE, "cache"),
                            ("datasets", paths.DATA, "datasets"),
                            ("qdrant storage", paths.QDRANT_STORAGE, "qdrant_storage")):
        source = paths.source(key)
        if source == "default" and key == "cache" and os.environ.get("XDG_CACHE_HOME"):
            source = "default, under $XDG_CACHE_HOME"
        # A directory that does not exist yet is not a fault: the fetcher and
        # the engine each create their own on first use.
        state = "" if path.is_dir() else "  not created yet"
        s.add(name, OK, f"{path}  {DIM}({source}){OFF}{state}")
    s.add("results", OK, f"{ROOT / 'bench/results'}  {DIM}(in the checkout, "
                         f"gitignored){OFF}")
    return s


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def section_datasets(quick: bool) -> Section:
    """What is fetched, and what it costs on this disk.

    `datasets.py list --json` rather than the table it prints for people: this
    used to scrape columns out of that table, which made a formatting change a
    silent parse failure. The measured `on_disk_bytes` is the reason to ask at
    all — the descriptor knows the download size, and the disk also holds what
    was unpacked and converted, which for SIFT1M is 160 MB against 1.2 GB.
    """
    s = Section("Datasets (§4.2)")
    ds = ROOT / "conformance/datasets/datasets.py"
    if not ds.exists():
        s.add("descriptor", BAD, "conformance/datasets/datasets.py missing")
        return s
    code, out = run([str(ds), "list", "--json", *(["--fast"] if quick else [])], 600)
    if code != 0:
        s.add("inventory", BAD, out.strip().splitlines()[-1][:100] if out.strip() else "failed")
        return s
    try:
        doc = json.loads(out)
    except json.JSONDecodeError:
        s.add("inventory", BAD, "datasets.py list --json did not produce JSON",
              out.strip().splitlines()[-1][:100] if out.strip() else "")
        return s

    # `--quick` skips the digests, so the files are present rather than verified.
    # Reporting them as verified would be the one lie this section could tell.
    checked = "present" if quick else "verified"
    for r in doc["datasets"]:
        size = human(r["on_disk_bytes"]) if r["on_disk_bytes"] else ""
        if r["status"] == "declared":
            s.add(r["name"], WARN, f"{'':>8}  declared: §4.2 names it, no files pinned")
        elif r["state"] == "complete":
            s.add(r["name"], OK, f"{size:>8}  {DIM}on disk, {r['files_present']} "
                                 f"file(s) {checked}{OFF}")
        elif r["state"] == "digest mismatch":
            s.add(r["name"], BAD, "digest mismatch", "datasets.py fetch will re-download",
                  kind="digest")
        elif r["state"] == "partial":
            s.add(r["name"], WARN, f"{size:>8}  {DIM}partial, {r['files_present']} of "
                                   f"{r['files_total']} file(s) {checked}{OFF}",
                  "datasets.py fetch")
        else:
            s.add(r["name"], WARN, f"{'':>8}  {DIM}not fetched, "
                                   f"{human(r['declared_bytes'])} to download{OFF}",
                  "datasets.py fetch")
        # Its own line, and not folded into `state`: the shards are complete and
        # verified, and it is only the templatable names bfb reads that dangle.
        # Reporting the dataset as incomplete would send the reader to `fetch`,
        # which re-downloads 14 GB and does not relink.
        if r.get("stale_links"):
            s.add(r["name"], BAD, f"{'':>8}  {r['stale_links']} bfb shard link(s) "
                                  f"dangle; the root moved after they were written",
                  f"datasets.py bfb-config {r['name']} --link", kind="links")

    disk = sum(r["on_disk_bytes"] for r in doc["datasets"])
    missing = sum(r["declared_bytes"] for r in doc["datasets"]
                  if r["state"] in ("not fetched", "partial"))
    rest = f", {human(missing)} left to fetch" if missing else ""
    s.add("total", OK, f"{human(disk):>8}  {DIM}in {doc['root']}{rest}{OFF}")
    return s


def verdict(sections: list[Section], host: dict, quick: bool) -> list[str]:
    """What class of number this host can produce, and why.

    The line that matters. Everything above is inputs; someone reading this file
    wants to know whether a number measured here can be quoted, and if not, what
    would have to change.
    """
    out: list[str] = []
    by = {s.title: s for s in sections}
    tool_bad = by["Toolchain"].worst == BAD
    build_bad = by["Build artifacts"].worst == BAD
    gate = host.get("gate_pass", False)

    if tool_bad or build_bad:
        out.append(f"{RED}Cannot measure.{OFF} A required tool or build artifact is missing; "
                   "fix the FAIL lines above first.")
        return out

    if gate:
        out.append(f"{GREEN}Publication-grade.{OFF} §7.1's host discipline holds, so a number "
                   "measured here may be quoted, with its environment hash "
                   f"({host.get('env_hash', '?')}) recorded beside it.")
    else:
        out.append(f"{YELLOW}Development-grade only.{OFF} §7.1's gate fails, so numbers measured "
                   "here describe this machine in its current state rather than the engine. "
                   "They are useful for spotting large effects and unusable for publication.")
        out.append(f"  {DIM}The settable checks: sudo bench/setup.py apply. "
                   f"isolcpus/nohz_full need a kernel cmdline change and a reboot.{OFF}")

    ds = by["Datasets (§4.2)"]
    bad = {i.kind for i in ds.items if i.status == BAD}
    if "digest" in bad:
        out.append(f"{RED}A dataset failed its digest.{OFF} Nothing measured against it means "
                   "anything until that is resolved.")
    if "links" in bad:
        # Scoped deliberately. The corpus is intact and every other dataset is
        # untouched, so this must not read as "the datasets are broken": the
        # only thing that cannot be trusted is a bfb run over the linked shards,
        # which opens the template and finds nothing.
        named = sorted(i.name for i in ds.items if i.kind == "links")
        out.append(f"{RED}bfb's shard links dangle for {', '.join(named)}.{OFF} The shards "
                   "themselves are present and verified, and no other dataset is affected — "
                   "but bfb reads that corpus through the links, so a row that loads it would "
                   "measure an empty collection. Relink before running it.")
    if not bad and any(i.status == WARN and "declared" not in i.detail
                       for i in ds.items):
        out.append("Some datasets are not fetched: conformance/datasets/datasets.py fetch")

    if by["Build artifacts"].worst == WARN:
        out.append("Some optional artifacts are missing; the relevant benchmarks cannot run "
                   "until they are built (see the warn lines).")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the bandwidth probe and dataset digests")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    paths.add_data_argument(ap)
    ap.add_argument("--exit-zero", action="store_true",
                    help="always exit 0. For the gate, which asks whether this "
                         "report runs, not whether the host is publication-ready: "
                         "a development machine legitimately fails §7.1 and that "
                         "must not fail the build.")
    args = ap.parse_args(argv[1:])
    # Exported, so the `datasets.py` this runs inventories the same root the
    # Paths section is about to print.
    paths.use_data_dir(args.data_dir)

    sections = [section_toolchain(), section_build()]
    host_sec, host = section_host(args.quick)
    sections.append(host_sec)
    perf_sec, perf = section_perf(args.quick)
    sections.append(perf_sec)
    sections.append(section_paths())
    sections.append(section_datasets(args.quick))

    if args.json:
        print(json.dumps({
            "sections": [{"title": s.title, "worst": s.worst,
                          "items": [vars(i) for i in s.items]} for s in sections],
            "host": host, "perf": perf,
        }, indent=2))
        return 0 if (args.exit_zero or all(s.worst != BAD for s in sections)) else 1

    print(f"\n{BOLD}strawmANN doctor{OFF}  {DIM}{ROOT}{OFF}")
    # Widened to the longest name rather than pinned at 24: the dataset names
    # run past it, and a name that overflows its column pushes that one line's
    # detail out of alignment with every other.
    w = max(24, max((len(i.name) for s in sections for i in s.items), default=24))
    for s in sections:
        print(f"\n{BOLD}{s.title}{OFF}")
        for i in s.items:
            print(f"  {MARK[i.status]} {i.name:<{w}} {i.detail}")
            if i.hint and i.status != OK:
                print(f"       {DIM}{i.hint}{OFF}")

    print(f"\n{BOLD}Verdict{OFF}")
    for line in verdict(sections, host, args.quick):
        print(f"  {line}")
    print()
    return 0 if (args.exit_zero or all(s.worst != BAD for s in sections)) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
