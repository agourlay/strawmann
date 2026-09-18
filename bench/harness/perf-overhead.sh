#!/usr/bin/env bash
# What the hardware sidecar costs the engine it is measuring.
#
# §7.1's discipline, turned on the instrument. `workloads.py run --perf`
# attaches `perf stat` to the engine for the length of every row, and a
# counter that changes the row is not measuring the row. `perfstat.py
# --overhead` answers this for a single-threaded Python spin, which is a floor
# and not a bound: the sidecar's cost is a function of how often the engine
# switches, faults and forks, and a spin loop does none of those. This answers
# it for a real row against a real engine.
#
# Three things here are load-bearing, and two of them were learned by getting
# the measurement wrong on a busy box.
#
# **The arms alternate ABBA, not ABAB.** Run `off` first in every pair and any
# monotonic drift — a warming cache, a thermal ramp, an index settling —
# reports as a cost of the second arm. Measured that way the sidecar looked
# like a consistent 4.0-4.7%; the ordering was doing an unknown share of it.
# §7.2(5) interleaves the two *engines* for the same reason, and the instrument
# deserves the same treatment.
#
# **It settles before every pass**, for the reason `noisefloor.sh` gives: the
# gate reads a one-minute load average at the *start* of an invocation, so a
# pass launched straight after another row stamps that row's load as its own.
#
# **A contaminated row is discarded, not averaged in.** The effect being looked
# for is a few percent, and a single foreign process is worth more than that.
# The script refuses to state a figure when fewer than two clean pairs survive,
# rather than reporting the median of whatever the machine allowed — which is
# how the first attempt at this produced 4.5% on a box that was compiling.
#
# Usage:  bench/harness/perf-overhead.sh [pairs] [row] [server-cpus] [client-cpus]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAIRS=${1:-4}
ROW=${2:-W3}
SERVER_CPUS=${3:-4-11}
CLIENT_CPUS=${4:-0-3}
PORT=6485
OUT="$ROOT/bench/results/perf-ab"
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

say "perf sidecar overhead: $PAIRS pairs of $ROW, engine on $SERVER_CPUS"
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
taskset -c "$CLIENT_CPUS" bench/harness/workloads.py run "$URI" perf-ab/build \
  W2 2>&1 | grep -E "^    ok|FAILED" | tail -3

# ABBA: off,on,on,off,off,on,on,off,... so a linear drift cancels across a pair
# of pairs rather than accumulating against one arm.
slot=0
for p in $(seq 1 "$PAIRS"); do
  if [ $((p % 2)) -eq 1 ]; then ORDER="off on"; else ORDER="on off"; fi
  for arm in $ORDER; do
    slot=$((slot + 1))
    FLAG=""; [ "$arm" = "on" ] && FLAG="--perf"
    settle "$ROW pair $p ($arm)"
    RESULTS_DIR="$OUT/$slot-$arm" taskset -c "$CLIENT_CPUS" \
      bench/harness/workloads.py run "$URI" "perf-ab/$slot-$arm" "$ROW" $FLAG \
      >/dev/null 2>&1
    python3 -c "
import json, sys
try:
    r = [x for x in json.load(open('$OUT/$slot-$arm/rows.json')) if x['id'] == '$ROW'][0]
except (OSError, IndexError):
    print('    slot $slot  perf $arm   no row written'); sys.exit(0)
dirty = r.get('foreign') or ''
print(f\"    slot $slot  perf $arm   {r['qps'] or 0:8.1f} qps   \"
      + (f'DISCARDED: {dirty}' if dirty else 'clean'))"
  done
done

kill "$EPID" 2>/dev/null; trap - EXIT
say "verdict"
exec python3 -c "
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, '$ROOT/bench/harness')
import regression

arms = {'on': [], 'off': []}
dirty = 0
for d in sorted(Path('$OUT').glob('*-o*')):
    rows = d / 'rows.json'
    if not rows.exists():
        continue
    r = [x for x in json.loads(rows.read_text()) if x['id'] == '$ROW']
    if not r or r[0].get('qps') is None:
        continue
    if r[0].get('foreign'):
        dirty += 1
        continue
    arms[d.name.rsplit('-', 1)[1]].append(r[0]['qps'])

pairs = min(len(arms['on']), len(arms['off']))
print(f'  clean rows: {len(arms[\"off\"])} without the sidecar, {len(arms[\"on\"])} with; '
      f'{dirty} discarded for foreign load')
if pairs < 2:
    print()
    print('  NO FIGURE. Fewer than two clean pairs survived, and the effect being')
    print('  looked for is a few percent — smaller than one foreign process. Run')
    print('  this on a quiet host; §7.1 is the definition of one.')
    raise SystemExit(1)

off, on = statistics.median(arms['off']), statistics.median(arms['on'])
delta = (on - off) / off * 100
spread = (max(arms['off']) - min(arms['off'])) / off * 100
rsd = (json.loads(Path('$ROOT/bench/results/noise.json').read_text())
       .get('rsd', {}).get('$ROW'))
band = regression.noise_band(rsd) * 100 if rsd else None
print(f'  without the sidecar: {off:,.1f} qps (median of {len(arms[\"off\"])}, '
      f'spread {spread:.2f}%)')
print(f'  with the sidecar:    {on:,.1f} qps (median of {len(arms[\"on\"])})')
print(f'  sidecar cost:        {-delta:+.2f}% of throughput')
if band is None:
    print(f'  -> no measured noise floor for $ROW, so this cannot be called '
          f'signal or scatter')
elif abs(delta) <= band:
    print(f'  -> within \${ROW}\\'s {band:.2f}% noise band: not distinguishable from '
          f'run-to-run scatter, and rows measured with and without the sidecar '
          f'may share a chart')
else:
    print(f'  -> OUTSIDE \${ROW}\\'s {band:.2f}% noise band: the sidecar is visible in '
          f'the measurement. --perf stays opt-in, and a row measured with it is '
          f'not comparable against one measured without it')
"
