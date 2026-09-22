#!/usr/bin/env bash
# §11 open question 3, measured: does the bitmap visited set beat the stamped
# array at 1M points?
#
# The traffic decomposition that motivates it (rel-0921, per query at ef=128):
# the quantized row moves 538 KiB, of which 64 KiB is the rescore pool, 107 KiB
# the quantized codes and 367 KiB the walk itself. §6.5's visited set is 4 MB
# per worker and takes one random cache line per neighbour probed (findings 28),
# so it is the prime suspect for that 367 KiB. The bitmap is 21x smaller and
# pays a reset proportional to what the query touched, which the decomposition
# does not price. This prices it.
#
# Two ReleaseFast binaries, same commit, same corpus, alternated A/B/A/B so a
# drifting host cannot be read as a difference between them. The rows are the
# two the decomposition is about (W10-ef128 fp32, W6-ef128 SQ8) plus W4, which
# is the headline and the one that must not regress.
#
# Needs a QUIET HOST: the gate counts foreign load, and this is a comparison of
# two numbers a few percent apart.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
OUT=${OUT:-$ROOT/bench/results/visited-ab-$(date +%Y%m%d-%H%M)}
ROWS=${ROWS:-"W2 W4 W10-ef128 W6-upload W6-ef128"}
REPS=${REPS:-3}
# What the largest collection needs, from the workload table rather than a
# constant that drifts from it.
CAPACITY=${CAPACITY:-$(python3 -c "import sys; sys.path.insert(0, 'bench/harness'); import workloads; print(workloads.required_capacity())")}
URI=http://localhost:6334
mkdir -p "$OUT"

log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/ab.log"; }

for arm in generation bitmap; do
  log "building $arm"
  zig build -Doptimize=ReleaseFast -Dvisited="$arm" >> "$OUT/ab.log" 2>&1 || exit 1
  cp zig-out/bin/strawmann "$OUT/strawmann-$arm" || exit 1
done

for rep in $(seq 1 "$REPS"); do
  for arm in generation bitmap; do
    log "=== pass $rep, $arm ==="
    pkill -x strawmann 2>/dev/null; sleep 2
    # The same flags `fullrun.start_strawmann` uses, so these rows are
    # comparable with a published run rather than only with each other:
    # 64 connections (bfb opens threads x connections and a short server closes
    # the excess before the preface), one io thread, seven workers on the pin,
    # and the storage dir wiped so the run maps its own arenas.
    rm -rf "${STRAWMANN_CACHE:-$HOME/.cache/strawmann}/strawmann-storage"
    taskset -c 4-11 "$OUT/strawmann-$arm" --port 6334 \
        --capacity "$CAPACITY" --connections 64 --workers 7 --io-threads 1 \
        --pin --cpus 4-11 \
        --data-dir "${STRAWMANN_CACHE:-$HOME/.cache/strawmann}/strawmann-storage" \
        --default-placement cached > "$OUT/server-$arm-$rep.log" 2>&1 &
    SRV=$!
    sleep 3
    # The banner is what says which arm served, and the row records it.
    grep -m1 "visited set" "$OUT/server-$arm-$rep.log" | tee -a "$OUT/ab.log"
    taskset -c 0-3 python3 bench/harness/workloads.py run "$URI" \
        "vis-$arm-rep$rep" $ROWS >> "$OUT/ab.log" 2>&1
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  done
done

log "done. Fold with:"
log "  for a in generation bitmap; do uv run --project bench bench/harness/aggregate.py vis-\$a vis-\$a-rep1 vis-\$a-rep2 vis-\$a-rep3; done"
log "  uv run --project bench bench/harness/compare.py vis-generation vis-bitmap"
