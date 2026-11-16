# Working in this repository

## Running things

- The Python harness runs under `bench/.venv/bin/python`. The system and pyenv
  interpreters have no pandas, and the report tests fail to import under them.
  - Harness tests: `cd bench/harness && ../.venv/bin/python -m unittest -q test_report test_workloads test_compare test_results test_paths`
  - Lint: `bench/.venv/bin/ruff check bench/harness`
- The full local gate is `scripts/check.py`; `--only '<step>'` runs one step,
  and CI runs the same steps by name.
- Conformance (Rust) is in `conformance/`: `cargo test --release`,
  `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check`, all
  gated.
- To find a hanging Zig test (ptrace is blocked by Yama, so no debugger
  attach), run the compiled `.zig-cache/o/*/test` binary directly under
  `timeout` in a loop. It prints `N/M name...` as each test starts, so the
  last line names the one that hung.

## Benchmarks

- A measured run needs a quiet machine. Another session's build (`cargo`,
  `rustc`) is foreign load the §7.1 gate counts, and its memory pressure can
  get the engine killed by `systemd-oomd`. So is an open Claude session's idle
  CPU. Schedule runs for the night with a `systemd-run --user` timer and close
  the other sessions first; `nightrun.py` and `w9_ab.py` warn about what is
  still running.
- To postpone a transient timer, read its settings first
  (`systemctl --user show <unit>.service -p Environment -p WorkingDirectory -p ExecStart`):
  stopping it deletes the unit, and the settings go with it.

## Setting up a new host

- Toolchains: Zig 0.17.1 (`build.zig.zon`); Rust is pinned in
  `conformance/rust-toolchain.toml` and rustup installs it on first `cargo`;
  `uv sync --project bench` creates `bench/.venv`. Night analyses call the
  `claude` CLI, so install it and log in.
- Paths: `~/.config/strawmann/config.toml`, a `[paths]` table with `cache`,
  `datasets`, `qdrant_storage` (and optionally `strawmann_storage`).
  `scripts/doctor.py` reports where each root came from and what is missing.
- Datasets: `conformance/datasets/datasets.py fetch`. Copying an existing
  datasets root is much faster: it holds the converted `.fbin` files and the
  ground truth, which take hours to recompute.
- Host: `bench/setup.py check`, and `apply` for what it can fix (cgroup I/O
  delegation among others). `apt-daily-upgrade.timer` re-executes systemd
  around 06:00 and takes the delegation back mid-run: mask it on run nights
  or make the delegation a permanent `Delegate=` drop-in. A night timer is a
  transient user unit, so on a server run `loginctl enable-linger` or it dies
  with the login session.
- bfb: build `workloads.BFB_COMMIT` in its own worktree, never by moving a
  working bfb checkout:
  `git -C <bfb repo> worktree add --detach ~/.cache/strawmann/bfb-pinned <commit>`
  then `cargo build --release` there, and pass `BFB=<that>/target/release/bfb`.
  The harness refuses a checkout off the pin and a binary older than it.
- Qdrant: build the pinned commit (`docs/spec.md` §8.9, `docs/decisions.md`)
  with `cargo build --release` and keep a sidecar `<binary>.json` beside it,
  `{"cargo_profile": "release", "commit": "<sha>", "built_at": "..."}`, which
  `nightrun.py` records. Pass `QDRANT_BINARY=<path>`. Quick development A/Bs
  of Qdrant changes use `--profile perf` (no LTO); pins use `release`.
- A new host is a new environment and a new series. Start with an empty
  `bench/results/`: the night runs take their load reference from the
  previous pair of the same name, and one measured on other hardware is the
  wrong reference. Noise floors from another environment are refused anyway.

## Night runs

- One corpus per `nightrun.py DATE DATASET` (three passes, `--perf`, the
  differ, the report, a headless analysis). Check the reference first with
  `nightrun.py --print-prev DATE DATASET` and `--print-ref`.
- Launch with a timer, several corpora chained with `;` in one `sh -c`:
  `systemd-run --user --unit=strawmann-night-MMDD --on-calendar='YYYY-MM-DD HH:MM' -p WorkingDirectory=$PWD --setenv=QDRANT_BINARY=... --setenv=BFB=... --setenv=OVERSAMPLING_POLICY=pool /bin/sh -c '/usr/bin/python3 bench/harness/nightrun.py YYYY-MM-DD sift1m; ...'`
- Each night leaves `bench/results/night-YYYYMMDD-<dataset>/` with
  `night.log`, `fullrun.out` and `analysis.md`. A failed analysis is redone
  with `nightrun.py --analysis-only DATE DATASET`.
- A/Bs between commits or flags: `bench/harness/commit_ab.py` (worktrees under
  the cache's `ab/`), summaries in `bench/results/<tag>/summary.txt`.
- Publishing: a licensed run writes `README.md`, `docs/comparison-*.md` and
  `docs/index.html`, and copies its report into `docs/reports/`. Read the
  analysis first, fix what it finds, re-render with
  `compare.py <sm-label> <qd-label> --write-readme`, retire the superseded
  report with `git mv`, update `docs/reports/README.md`, then commit.
- Stop engines by PID, never `pkill -f`: the pattern matches the harness and
  the night unit too.

## House rules

- Commit straight to `main` and push. No feature branches.
- No em or en dashes in prose, docs or commit messages; use periods, commas,
  colons or parentheses.
- Minimal fixes, each with a test.
- Logic goes in tested Python; shell only calls it.
- Run the Zig tests without `nice`, and one `--only` per `scripts/check.py`
  invocation.
- Two `test_workloads` tests fail while any `qdrant` or `strawmann` process
  runs on the host.
- Why things are as they are: `docs/decisions.md`. What is next:
  `docs/backlog.md`. What a night found: its `analysis.md`.
