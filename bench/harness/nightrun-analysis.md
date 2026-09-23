You are analyzing a strawmANN benchmark run that finished overnight on this machine. Repository: the strawmann checkout you are started in (the current working directory). Do not modify any file; your written answer is the deliverable and is being saved to a Markdown file verbatim, so write GitHub-flavored Markdown with no preamble.

The run: `fullrun.py --dataset @DATASET@ --reps 3 --segment-policy equal-work --perf --qdrant-binary ...` into labels @SM@ / @QD@, launched by bench/harness/nightrun.py; the exact command line is in @DIR@/night.log and the first lines of the fullrun output.

Inputs to read:
- @DIR@/night.log: the waiter's log (provenance, the reference pair, exit code, timings).
- @DIR@/fullrun.out: fullrun.py's full stdout+stderr. Under `--wait-for-gate` it asks a busy host again every 5 minutes, so any gate refusals before admission are at its top, each followed by `--wait-for-gate: asking again`.
- @DIR@/report.path: the path of the HTML report; also @DIR@/report_by_hand.out if it exists.
- bench/results/@SM@*/ and bench/results/@QD@*/ (the folded labels and their -rep1..-rep3 passes): rows.json, run.json, conformance.json, any perf or gate files.
- The HTML report itself: its licence banner, refusals, notes, the Hardware counters section, and any row-level flags.
- For the meaning of gate, foreign load, contamination retries, refusals, licences and tiers: docs/spec.md (§7.1, §7.2, §7.4, §8), docs/validation.md, docs/findings.md, and the docstrings in bench/harness/fullrun.py, workloads.py, report.py, compare.py, perfstat.py.
- The previous run of this pair for comparison, if any: bench/results/@PREV@ and its qd- counterpart, and the newest bench/results/report-@DATASET@-*.html older than this run's.

Deliver, in this order:

1. **Outcome.** Did the run complete? Exit code, how many times the gate or a held port refused before admission and why, wall clock per phase (gate, build, strawmann arm, qdrant arm, conformance, render), whether README.md / docs/comparison-<dataset>.md were written or refused and the exact reasons printed, and `git status --short` of the checkout afterwards.

2. **Every warning and error, exhaustively.** Sweep the fullrun output, the waiter log, the report and the JSON files for: lines starting with `!!`, `warn`, `note`, `error`, `refus`, `contaminat`, `foreign`, `retry`, `gate`, `stale`, `unpublishable`, `lax`, `skipped`, `timeout`, `perf` sidecar messages, calibration, tracebacks, non-zero exits, and the report's banner and per-row flags. Group identical messages and count them. For each distinct item give: the quoted line, where it appeared (file and phase), what it means in this harness's vocabulary, the root cause as best you can determine from the code and docs (classify as: harness defect, property of the run as requested, host condition, engine behavior, or unknown), whether it affects the publishability or the licence tier of the result, and a recommended action. Do not skip low-severity notes; the request is to understand all of them.

3. **Headline numbers.** A short table of the per-workload medians with spread for both engines, the ratios, recall at the sweep points, the conformance tier reached, and the perf counters summary (IPC, frequency, MPKI, DRAM per query) per engine. Note anything that moved materially against the previous run of the same pair and whether a difference in the Qdrant binary (commit, or cargo profile as run.json records it) plausibly explains it.

4. **Things to fix or decide**, as a prioritized list, each pointing at the file and line that would change.

Quote lines verbatim; be concrete; do not speculate beyond what the code and files show, and say when something is unknown.
