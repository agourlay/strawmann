#!/usr/bin/env bash
# The noise floor, as a sequence rather than as folklore.
#
# `report.py` renders no verdict on a ratio without one: it cannot tell 1.15x
# from run-to-run scatter, so every unmeasured row reads "no noise floor
# measured for this row" and the reader has to take the number on faith. This
# builds one: one engine, one corpus, N identical search passes, and the
# per-row relative standard deviation between them.
#
# Two things here are load-bearing, and each was learned by getting it wrong.
#
# **It settles before every pass.** `workloads.py` runs the §7.1 gate at the
# *start* of an invocation and `load1` is a one-minute average, so a pass
# launched straight after a 1M-point upload reads that upload's load and stamps
# FAIL -- the measurement's own load, with no foreign process named. fullrun.py
# already solved this with `settle()`; a driver that calls workloads.py directly
# bypasses it, which is what the first three attempts at this floor all did.
#
# **It aborts on the first failed gate**, rather than completing every
# repetition and discovering at the end that none of them counts.
#
# W11 is excluded on purpose: it appends 200k to bench2, so a second pass is not
# a repetition of the first.
#
# Usage:  bench/harness/noisefloor.sh [reps] [server-cpus] [client-cpus]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPS=${1:-6}
SERVER_CPUS=${2:-4-11}
CLIENT_CPUS=${3:-0-3}
PORT=6484
OUT="$ROOT/bench/results/nf"
cd "$ROOT"

WORKERS=$(( $(python3 -c "
import sys
n=0
for part in sys.argv[1].split(','):
    a,_,b = part.partition('-')
    n += int(b or a) - int(a) + 1
print(n)" "$SERVER_CPUS") - 1 ))

say()    { echo; echo "=== $* $(date -u +%H:%M:%S)"; }
settle() { python3 -c "
import sys; sys.path.insert(0, '$ROOT/bench/harness')
import fullrun; fullrun.settle(sys.argv[1])" "$1"; }

say "noise floor: $REPS repetitions, engine on $SERVER_CPUS, client on $CLIENT_CPUS"
rm -rf "$OUT"; mkdir -p "$OUT"

PIDF="$OUT/engine.pid"
setsid bash -c 'echo $$ > "$1"; exec "$2" --port "$3" --capacity 1200000 \
    --connections 64 --workers "$4" --io-threads 1 --pin --cpus "$5" \
    --no-bandwidth-probe' _ "$PIDF" "$ROOT/zig-out/bin/strawmann" "$PORT" \
    "$WORKERS" "$SERVER_CPUS" > "$OUT/engine.log" 2>&1 < /dev/null &
sleep 5
EPID=$(cat "$PIDF") || { echo "engine did not start; see $OUT/engine.log"; exit 1; }
echo "engine pid $EPID"
trap 'kill "$EPID" 2>/dev/null' EXIT
URI="http://localhost:$PORT"

say "building the corpus once"
settle "the corpus build"
taskset -c "$CLIENT_CPUS" bench/harness/workloads.py run "$URI" nf/build \
  W0-upload W1 W2 W6-upload W7-upload W8-upload 2>&1 | grep -E "^    ok|FAILED" | tail -8

SEARCH="W0 W3 W4 W5 W6 W7 W8 W9 W10-ef32 W10-ef64 W10-ef128 W10-ef256 W10-ef512 W13"
for i in $(seq 1 "$REPS"); do
  say "repetition $i/$REPS"
  settle "noise repetition $i"
  taskset -c "$CLIENT_CPUS" bench/harness/workloads.py run "$URI" "nf/rep$i" $SEARCH \
    2>&1 | grep -cE "^    ok" | sed 's/^/    ok rows: /'
  python3 -c "
import json, sys
g = json.load(open('$OUT/rep$i/run.json'))['gate']
print(f'    gate: {g}')
sys.exit(0 if g == 'pass' else 1)" \
    || { echo "!! repetition $i failed the host gate; the floor it would"; \
         echo "   produce would be interference, not noise. Aborting."; exit 1; }
done

kill "$EPID" 2>/dev/null; trap - EXIT; sleep 3
say "measuring the floor"
exec bench/harness/regression.py --measure-noise "$OUT"
