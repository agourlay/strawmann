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

DATE=${1:?usage: nightrun.sh YYYY-MM-DD [dataset]}
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

mkdir -p "$D"
cd "$ROOT" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "night run starting: $DATASET, labels $SM / $QDL, reps $REPS"
log "strawmann $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
if [ ! -x "$QD" ]; then log "EXIT=97 (no Qdrant binary at $QD; set QDRANT_BINARY)"; exit 97; fi
log "qdrant binary $QD sha256=$(sha256sum "$QD" | cut -c1-16) commit=$(git -C "$(dirname "$QD")" rev-parse --short HEAD 2>/dev/null || echo '?')"

# The previous pair of this family, for the open-loop reference: the slower
# engine's W4, as `fullrun.resolve_rps_reference` picks it.
PREV=$(ls -d "$ROOT"/bench/results/sm-$FAM-[0-9][0-9][0-9][0-9] 2>/dev/null | grep -v "$SM" | sort | tail -1)
REF_ARGS=()
if [ -n "$PREV" ]; then
  P=$(basename "$PREV"); PQ=qd-${P#sm-}
  REF=$(python3 -c "
import sys; sys.path.insert(0, 'bench/harness'); import fullrun
print(int(round(fullrun.resolve_rps_reference('auto', ['$P', '$PQ']))))" 2>/dev/null | tail -1)
  if [ -n "$REF" ]; then REF_ARGS=(--rps-reference "$REF"); log "rps reference $REF from $P / $PQ"; fi
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

if command -v claude >/dev/null; then
  log "analysis starting"
  PROMPT=$(sed -e "s|@SM@|$SM|g; s|@QD@|$QDL|g; s|@DIR@|bench/results/night-${DATE//-/}|g; s|@DATASET@|$DATASET|g; s|@PREV@|${PREV:+$(basename "$PREV")}|g" \
           "$ROOT/bench/harness/nightrun-analysis.md")
  claude -p "$PROMPT" --output-format text \
    --allowedTools "Read,Grep,Glob,Bash(cat:*),Bash(grep:*),Bash(head:*),Bash(tail:*),Bash(ls:*),Bash(wc:*),Bash(sed -n:*),Bash(python3:*),Bash(uv run:*),Bash(git status:*),Bash(git diff:*),Bash(git log:*),Bash(jq:*),Bash(ps:*),Bash(pgrep:*)" \
    > "$D/analysis.md" 2> "$D/analysis.err"
  log "analysis exited $? ($(wc -l < "$D/analysis.md") lines in $D/analysis.md)"
else
  log "no claude on PATH; analysis skipped"
fi
log "ALL_DONE"
exit "$RC"
