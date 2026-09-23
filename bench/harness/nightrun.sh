#!/usr/bin/env bash
# An unattended publication run, scheduled for a quiet hour.
#
#   bench/harness/nightrun.sh 2026-09-10 sift1m
#   systemd-run --user --on-calendar='2026-09-10 04:00:00' \
#       --unit=strawmann-night-20260910 -p WorkingDirectory=$PWD \
#       /usr/bin/bash bench/harness/nightrun.sh 2026-09-10 sift1m
#
# Why a systemd --user timer and not a Claude session's cron: the §7.1 gate
# counts the CLI's idle CPU as foreign load (findings 44), so the run must
# not need a session open. Why a waiter: the gate refuses a busy host with
# "refusing to run", and asking again every five minutes is what gets a run
# admitted on a laptop that also builds things. Any other failure is not
# retried -- a run that died mid-corpus must be read, not repeated.
#
# Fresh labels are stamped with the date so a night never merges into a
# published set. `--rps-reference` is resolved from the previous pair of the
# same family, since a fresh label gives `auto` nothing to read and the
# fallback would refuse the cross-engine latency read.
#
# Afterwards a headless `claude -p`, restricted to read-only tools, writes
# `analysis.md` beside the log: every warning, refusal and note, with its
# cause. The prompt is `nightrun-analysis.md` next to this script.
set -u
: "${HOME:?HOME must be set (systemd --user sets it)}"
export PATH=$HOME/.local/bin:$HOME/.pyenv/shims:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin

# `--print-prev` answers "which pair would this run take its reference from?"
# and exits, touching nothing. A scheduled run that silently found no previous
# pair loses its cross-engine latency read and says so only in the report, hours
# later; this is how to ask beforehand, and it is what the test drives.
PRINT_PREV=0
PRINT_REF=0
if [ "${1:-}" = "--print-prev" ]; then PRINT_PREV=1; shift; fi
# `--print-ref` answers "and what reference would it take from that pair?",
# which is the other half of the same question and was the other half of the
# same class of bug: the resolver prints its refusal on stdout, the helper
# captured that sentence as a number, and `fullrun.py` rejected the flag and
# exited 2 before the run began. Asking costs nothing and writes nothing.
if [ "${1:-}" = "--print-ref" ]; then PRINT_REF=1; shift; fi

DATE=${1:?usage: nightrun.sh [--print-prev] YYYY-MM-DD [dataset]}
DATASET=${2:-sift1m}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
D=$ROOT/bench/results/night-${DATE//-/}
LOG=$D/night.log
QD=${QDRANT_BINARY:-$HOME/Workspace/qdrant/target/release/qdrant}
SERVER_CPUS=${SERVER_CPUS:-4-11}
CLIENT_CPUS=${CLIENT_CPUS:-0-3}
REPS=${REPS:-3}
TAG=${DATE:5:2}${DATE:8:2}
case $DATASET in
  sift1m) FAM=sift-perf ;;
  dbpedia-openai-100K-1536-angular) FAM=dbp100k-perf ;;
  dbpedia-openai-1m) FAM=dbp1m-perf ;;
  *) FAM=${DATASET}-perf ;;
esac
SM=sm-$FAM-$TAG
QDL=qd-$FAM-$TAG
MAX_TRIES=${MAX_TRIES:-24}   # x ~5 min: two hours of asking the gate

# The previous pair of this family, for the open-loop reference: the slower
# engine's W4, as `fullrun.resolve_rps_reference` picks it.
#
# Two bugs in one line, both silent. The glob was `sm-$FAM-[0-9][0-9][0-9][0-9]`
# and every label this project has published carries a `rel-` before the date
# (`sm-dbp1m-perf-rel-0903`), so it matched nothing, `auto` had nothing to read,
# and the run's open-loop arms each used their own engine's saturation, which
# makes the report refuse the cross-engine latency read. And `sort` was
# lexicographic, so once a bare-date label existed beside a `rel-` one,
# `sm-dbp1m-perf-rel-0903` sorted *after* `sm-dbp1m-perf-0910` because `r` > `0`
# and the older pair won. Ordered by the trailing MMDD instead.
prev_pair() {
  ls -d "$ROOT"/bench/results/sm-"$FAM"-*[0-9][0-9][0-9][0-9] 2>/dev/null \
    | grep -v "/$SM\$" \
    | awk -F- '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2-
}

# A number or nothing. Both guards matter and both were learned here: set
# `PERF_SET` to what this run will pass, because the resolver compares a
# candidate against *its own module global* and a bare interpreter has `None`;
# and filter to digits, because a refusal is printed rather than raised.
resolve_ref() {
  local p=$1 pq=$2
  python3 -c "
import sys; sys.path.insert(0, 'bench/harness'); import fullrun
fullrun.PERF_SET = 'default'
try:
    v = fullrun.resolve_rps_reference('auto', ['$p', '$pq'])
except ValueError:
    v = None
print(int(round(v)) if v else '')" 2>/dev/null | grep -Ex '[0-9]+' | tail -1
}

if [ "$PRINT_PREV" = 1 ]; then prev_pair; exit 0; fi
if [ "$PRINT_REF" = 1 ]; then
  PREV=$(prev_pair)
  [ -z "$PREV" ] && exit 0
  cd "$ROOT" || exit 1
  P=$(basename "$PREV"); resolve_ref "$P" "qd-${P#sm-}"; exit 0
fi

# One run per night directory. Two instances sharing `$D` share `night.log`,
# `analysis.md` and `report.path`, and on 2026-09-23 that is exactly what
# happened: a failed 03:51 launch had its analysis still running when the
# directory was removed and a second launch recreated it, so a stray
# "analysis exited 0" from the dead run landed in the middle of the live one's
# log and its own analysis could only call the line unexplained. A run that
# finds a log already there stops rather than interleaving with whatever wrote
# it; `$D` is derived from the date alone, so a second run of one night needs
# its own directory (or the first one moved aside).
if [ -e "$LOG" ]; then
  echo "nightrun: $LOG exists; another run of $DATE has used this directory." >&2
  echo "  move it aside, or pass a different date, rather than sharing a log." >&2
  exit 96
fi
mkdir -p "$D"
cd "$ROOT" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "night run starting: $DATASET, labels $SM / $QDL, reps $REPS"
log "strawmann $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
if [ ! -x "$QD" ]; then log "EXIT=97 (no Qdrant binary at $QD; set QDRANT_BINARY)"; exit 97; fi
# The checkout's HEAD is not evidence about the binary, and saying it alone
# asserts a commit that may not have produced it: on 2026-09-23 this line read
# `commit=63c6a797d` for a binary built at an earlier one, and `run.json`
# carried the same claim. `fullrun.py` already computes the verdict; the log
# says it too, because the log is what a reader reaches for first.
QD_DIR=$(dirname "$QD")
QD_COMMIT=$(git -C "$QD_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')
QD_PREDATES=""
if [ "$QD_COMMIT" != "?" ]; then
  QD_HEAD_TS=$(git -C "$QD_DIR" log -1 --format=%ct 2>/dev/null || echo 0)
  QD_BIN_TS=$(stat -c %Y "$QD" 2>/dev/null || echo 0)
  [ "$QD_BIN_TS" -lt "$QD_HEAD_TS" ] 2>/dev/null &&     QD_PREDATES=" (BINARY PREDATES THIS COMMIT; sha256 is the identity that holds)"
fi
log "qdrant binary $QD sha256=$(sha256sum "$QD" | cut -c1-16) commit=$QD_COMMIT$QD_PREDATES"

PREV=$(prev_pair)
REF_ARGS=()
if [ -n "$PREV" ]; then
  P=$(basename "$PREV"); PQ=qd-${P#sm-}
  REF=$(resolve_ref "$P" "$PQ")
  if [ -n "$REF" ]; then REF_ARGS=(--rps-reference "$REF"); log "rps reference $REF from $P / $PQ"
  else log "no usable rps reference from $P / $PQ (see the resolver's refusal); each engine uses its own saturation"; fi
fi
[ ${#REF_ARGS[@]} -eq 0 ] && log "no previous $FAM pair: each engine uses its own saturation (report refuses the cross-engine latency read)"

# Sessions of the CLI are ambient load the gate sees (findings 44). Named,
# not killed: they are the user's.
CL=$(pgrep -x claude | wc -l)
[ "$CL" -gt 0 ] && log "warning: $CL claude process(es) alive; the gate counts their idle CPU as foreign load"

ports_busy() { ss -ltn 2>/dev/null | grep -qE ':(6333|6334|6344) '; }

run_until_admitted() {
  local try=0
  while :; do
    try=$((try+1))
    if ports_busy; then
      log "attempt $try: an engine port (6333/6334/6344) is already bound; waiting 5 min"
      [ "$try" -ge "$MAX_TRIES" ] && { log "EXIT=98 (ports stayed busy)"; return 98; }
      sleep 300; continue
    fi
    log "=== attempt $try ==="
    local out="$D/fullrun_attempt${try}.out"
    python3 bench/harness/fullrun.py \
        --server-cpus "$SERVER_CPUS" --client-cpus "$CLIENT_CPUS" \
        --dataset "$DATASET" --reps "$REPS" --segment-policy equal-work \
        --perf --qdrant-binary "$QD" "${REF_ARGS[@]}" \
        --strawmann-label "$SM" --qdrant-label "$QDL" > "$out" 2>&1
    local rc=$?
    log "attempt $try: fullrun.py exited $rc (output in $out)"
    if [ "$rc" -eq 0 ]; then log "EXIT=0"; return 0; fi
    if grep -q "refusing to run" "$out" && ! grep -q "^=== pass 1" "$out"; then
      log "gate refused (attempt $try); retrying in 5 min"
      [ "$try" -ge "$MAX_TRIES" ] && { log "EXIT=$rc (gave up after $try refusals)"; return "$rc"; }
      sleep 300; continue
    fi
    log "EXIT=$rc (not a gate refusal; not retried)"
    return "$rc"
  done
}

START_TS=$(date +%s)
run_until_admitted
RC=$?
log "measurement finished, rc=$RC, $(( ($(date +%s) - START_TS) / 60 )) min"

REPORT=$(ls -t "$ROOT"/bench/results/report-$DATASET-$SM-vs-$QDL-*.html 2>/dev/null | head -1)
if [ -z "$REPORT" ] && [ -f "$ROOT/bench/results/$SM/rows.json" ] && [ -f "$ROOT/bench/results/$QDL/rows.json" ]; then
  log "no report from render; running report.py by hand"
  uv run --project bench bench/harness/report.py "$SM" "$QDL" > "$D/report_by_hand.out" 2>&1
  log "report.py exited $?"
  REPORT=$(ls -t "$ROOT"/bench/results/report-$DATASET-$SM-vs-$QDL-*.html 2>/dev/null | head -1)
fi
log "report: ${REPORT:-none}"
echo "${REPORT:-none}" > "$D/report.path"

# A copy, not just the path. The stamped name comes from the newest arm's
# `started` (`report.run_stamp`), so any later re-render of the same pair
# writes the same file: on 2026-09-23 a session re-rendered at 07:04 and the
# 07:00 artifact was gone, with only `report.path` pointing at what was now a
# different file. The copy beside the log is what the run actually produced.
if [ -n "$REPORT" ] && [ -f "$REPORT" ]; then
  cp -p "$REPORT" "$D/$(basename "$REPORT")" && log "report copied into $D"
fi

if command -v claude >/dev/null; then
  log "analysis starting"
  PROMPT=$(sed -e "s|@SM@|$SM|g; s|@QD@|$QDL|g; s|@DIR@|bench/results/night-${DATE//-/}|g; s|@DATASET@|$DATASET|g; s|@PREV@|${PREV:+$(basename "$PREV")}|g" \
           "$ROOT/bench/harness/nightrun-analysis.md")
  claude -p "$PROMPT" --output-format text \
    --allowedTools "Read,Grep,Glob,Bash(cat:*),Bash(grep:*),Bash(head:*),Bash(tail:*),Bash(ls:*),Bash(wc:*),Bash(sed -n:*),Bash(python3:*),Bash(uv run:*),Bash(git status:*),Bash(git diff:*),Bash(git log:*),Bash(jq:*),Bash(ps:*),Bash(pgrep:*)" \
    > "$D/analysis.md" 2> "$D/analysis.err"
  # `wc -l` on its own would print an error and an empty count if the file
  # were gone, which is how a stray line from a dead run reads.
  AN_RC=$?
  log "analysis exited $AN_RC ($(wc -l < "$D/analysis.md" 2>/dev/null || echo '?') lines in $D/analysis.md)"
else
  log "no claude on PATH; analysis skipped"
fi
log "ALL_DONE"
exit "$RC"
