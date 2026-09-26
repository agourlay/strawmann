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
