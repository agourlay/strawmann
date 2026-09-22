#!/usr/bin/env bash
# P2 item 6: where strawmANN's 322k cycles per query of unsaturated overhead go.
#
# rel-0921 measures 972k cycles per query on W3 (`-p 1`) against 650k on W4 at
# saturation, on the same collection at the same ef. W3 is the one search row
# strawmANN loses (0.85x), and overhead that saturation amortises away is a wake
# or a spin rather than search. This asks the profiler which.
#
# Three samples, because the answer is a *difference* between two load levels
# and a single flat profile of either cannot show it:
#   1. W3   `-p 1`, the row that loses
#   2. W4   `-p 64`, the same queries saturated
#   3. idle, the engine serving nothing, so a spin shows up with no query at all
#
# Needs a QUIET HOST, and `perf_event_paranoid` already permits this (the gate
# checks it).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
OUT=${OUT:-$ROOT/bench/results/unsaturated-$(date +%Y%m%d-%H%M)}
URI=http://localhost:6334
FREQ=${FREQ:-2000}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/profile.log"; }

pkill -x strawmann 2>/dev/null; sleep 2
zig build -Doptimize=ReleaseFast >> "$OUT/profile.log" 2>&1 || exit 1
CAPACITY=${CAPACITY:-$(python3 -c "import sys; sys.path.insert(0, 'bench/harness'); import workloads; print(workloads.required_capacity())")}
# Only the arena files, and the directory stays: `fullrun.wipe_strawmann_storage`
# unlinks `*.vectors.bin` rather than removing the tree, and the engine does not
# create the directory itself. `rm -rf` on it made every upload fail with
# "could not create or map the arena file for this placement".
STORE="${STRAWMANN_CACHE:-$HOME/.cache/strawmann}/strawmann-storage"
mkdir -p "$STORE" && rm -f "$STORE"/*.vectors.bin
taskset -c 4-11 ./zig-out/bin/strawmann --port 6334 \
    --capacity "$CAPACITY" --connections 64 --workers 7 --io-threads 1 \
    --pin --cpus 4-11 \
    --data-dir "$STORE" \
    --default-placement cached > "$OUT/server.log" 2>&1 &
SRV=$!
sleep 3
PID=$(pgrep -x strawmann | head -1)
log "engine pid $PID"

log "loading bench2"
taskset -c 0-3 python3 bench/harness/workloads.py run "$URI" prof-load W2 \
    >> "$OUT/profile.log" 2>&1

# Idle first: whatever the engine burns with no query in flight is the floor,
# and if the wake path is a spin it is visible here and nowhere else.
log "idle sample, 20 s"
perf record -F "$FREQ" -g -p "$PID" -o "$OUT/idle.data" -- sleep 20 \
    >> "$OUT/profile.log" 2>&1

for row in W3 W4; do
  log "profiling $row"
  perf record -F "$FREQ" -g -p "$PID" -o "$OUT/$row.data" -- \
      taskset -c 0-3 python3 bench/harness/workloads.py run "$URI" "prof-$row" "$row" \
      >> "$OUT/profile.log" 2>&1
done

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
for f in idle W3 W4; do
  log "=== $f, top 25 by self time ==="
  perf report -i "$OUT/$f.data" --stdio --sort symbol --percent-limit 0.5 2>/dev/null \
      | head -35 | tee -a "$OUT/profile.log"
done
log "done: $OUT"
