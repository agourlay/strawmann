#!/usr/bin/env bash
# What "smoke test" means in this project, so it means one thing.
#
# `sift1m`, one pass per engine, into labels that are not the published ones.
# About 35 minutes of measurement plus the §8 differ; `fullrun.py` prints its
# own estimate before the gate settles, so the real number is on screen before
# any time is spent.
#
# It exists because the full run is not a feedback loop. dbpedia-openai-1m at
# `--reps 3` is 7.7 hours, of which 44% is rebuilding the corpus three times,
# and that is a publication artifact rather than something to run after a
# change. This is the other end: the fastest run that still exercises both
# engines end to end, every workload, and §8's differ.
#
# Three choices are fixed here rather than left to whoever types the command,
# which is the whole point of the file.
#
# **`--reps 1`, and it forfeits two things.** §7.2(5) wants A/B/A/B and gets
# A-then-B, so anything that drifts on this machine between the arms is
# confounded with the engine. And `aggregate._rsd` returns None below three
# passes -- "two passes give a spread that is really one difference" -- so no
# per-corpus noise floor is folded and no ratio on the report is banded. The
# report says both, twice, and that is correct: a smoke test is not a result.
# `--reps 3` is the smallest number that buys either, at three times the cost.
#
# **Labels `smoke-sm` / `smoke-qd`, never the defaults.** `fullrun.py` defaults
# to `strawmann` / `qdrant`, which on sift1m are the *published* pair behind
# the dated `report-sift1m-strawmann-vs-qdrant-*.html`. `rows.json` merges in place
# and `run.json` is overwritten by the last invocation, so a smoke test at the
# default labels does not fail loudly -- it quietly mixes one unbanded pass
# into a published result set and restamps it. Separate labels are the only
# thing standing between a five-second habit and that.
#
# **`--segment-policy equal-work`, stated rather than inherited.** It is
# `fullrun.py`'s default, so this changes nothing today -- and a standard that
# rides on a default is one that changes meaning the day the default does. It
# is also the policy that agrees with the licence: §8's differ builds its own
# collection at Qdrant's segment count and its corpus is small enough (n=100,000
# at d=128) to settle at one populated graph, so an `equal-work` run has rows
# and a T3 verdict describing the same Qdrant. Under `as-deployed` they would
# not, which for a run whose point is "does the differ still pass" is the wrong
# trade. `as-deployed` therefore goes unexercised by this loop; run
# `fullrun.py` directly for it, and see decisions.md.
#
# **`--perf` on, when a Qdrant binary can be found.** The sidecar costs a
# measured 4.25% of throughput against W3's 2.00% noise band (docs/validation.md)
# and is opt-in everywhere else for that reason -- but a smoke test folds no
# noise floor, so nothing on its page is banded and there is no number for the
# perturbation to spoil. What it buys is coverage: `perfstat.py` is a large
# piece of harness that would otherwise only ever run inside the eight-hour
# publication run, which is a bad place to find out the sidecar broke. It also
# makes these rows comparable by eye with `sm-sift-perf` / `qd-sift-perf`,
# which were measured the same way.
#
# `perf_event_open` cannot attach into the container -- the image runs as root
# and this harness does not -- so `--perf` requires a native Qdrant, and
# `fullrun.py` exits 2 on the pair rather than measuring one arm and blanking
# the other. There is no portable path for that binary the way `_find_bfb`
# has one, so this searches and *degrades*: found, and the run carries
# counters; not found, and it says so and measures without them. A smoke test
# that refuses to run because of an optional instrument is not a smoke test.
#
# **The conformance differ stays in.** It is the slowest phase and the first
# thing anyone would cut, and it is also the only part that checks the two
# engines still answer the same questions -- which is what a smoke test is for.
# A run that is 20% faster and no longer notices strawmANN returning wrong ids
# is not a smoke test. Pass `--skip conformance` if you truly only want speed.
#
# Usage:  bench/harness/smoke-test.sh [server-cpus] [client-cpus] [extra fullrun args...]
#
#   bench/harness/smoke-test.sh
#   bench/harness/smoke-test.sh 4-11 0-3 --perf --qdrant-binary ~/Workspace/qdrant/target/release/qdrant
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# The two cpu sets are positional and optional, so anything starting with `-`
# is a fullrun flag and not a cpu set. Without this check
# `smoke-test.sh --perf` silently became `--server-cpus --perf`, which fails
# somewhere far from the cause.
SERVER_CPUS=4-11
CLIENT_CPUS=0-3
case "${1:-}" in -*|"") ;; *) SERVER_CPUS=$1; shift ;; esac
case "${1:-}" in -*|"") ;; *) CLIENT_CPUS=$1; shift ;; esac

# The four choices above are the file's whole purpose, so an extra argument
# may not quietly undo one. argparse lets a later flag win, and a passed-through
# `--dataset dbpedia-openai-1m` would stamp `smoke-sm/run.json` with the other
# corpus -- after which `workloads.foreign_dataset` refuses every later plain
# smoke run with a message naming the label rather than the override, until the
# directories are deleted by hand.
for a in "$@"; do
  case "$a" in
    --dataset|--dataset=*|--reps|--reps=*|--strawmann-label*|--qdrant-label*|\
    --segment-policy|--segment-policy=*)
      echo "smoke-test.sh: $a is fixed by this script -- it is what \"smoke test\"" >&2
      echo "  means here. Call bench/harness/fullrun.py directly to vary it." >&2
      exit 2 ;;
  esac
done

# Left alone entirely if the caller said anything about the engine or counters:
# `--qdrant-docker` is mutually exclusive with `--qdrant-binary`, so injecting
# one beside it turns a clear choice into an argparse error.
PERF=()
engine_flag=0
for a in "$@"; do
  case "$a" in
    --perf|--perf=*|--qdrant-binary|--qdrant-binary=*|--qdrant-docker) engine_flag=1 ;;
  esac
done
if [ "$engine_flag" -eq 0 ]; then
  QDRANT=""
  for c in "${QDRANT_BINARY:-}" \
           "$ROOT/../qdrant/target/release/qdrant" \
           "$HOME/Workspace/qdrant/target/release/qdrant" \
           "$HOME/src/qdrant/target/release/qdrant"; do
    # Resolved, because this path is printed and lands in `run.json`; the
    # candidate below is written relative to $ROOT and reads as `.../../qdrant`.
    if [ -n "$c" ] && [ -x "$c" ]; then QDRANT=$(readlink -f "$c"); break; fi
  done
  if [ -n "$QDRANT" ]; then
    PERF=(--perf --qdrant-binary "$QDRANT")
  fi
fi

cd "$ROOT"
echo "smoke test: sift1m, equal-work (Qdrant at one populated graph),"
echo "  one pass per engine, labels smoke-sm / smoke-qd"
echo "  not a result: one pass forfeits §7.2(5) interleaving and folds no noise"
echo "  floor, so the report bands no ratio and says so. Use --reps 3 for that."
if [ "${#PERF[@]}" -gt 0 ]; then
  echo "  hardware counters on, Qdrant native from ${PERF[2]}"
elif [ "$engine_flag" -eq 0 ]; then
  echo "  no hardware counters: no qdrant binary found. perf_event_open cannot"
  echo "  attach into the container, so --perf needs a native one -- set"
  echo "  \$QDRANT_BINARY or build ../qdrant. Measuring without it."
fi
echo

exec python3 bench/harness/fullrun.py \
    --server-cpus "$SERVER_CPUS" --client-cpus "$CLIENT_CPUS" \
    --dataset sift1m \
    --reps 1 \
    --segment-policy equal-work \
    --strawmann-label smoke-sm --qdrant-label smoke-qd \
    ${PERF[@]+"${PERF[@]}"} \
    "$@"
