#!/usr/bin/env bash
# TASK-040 two-hour RSS growth check for realsense_online.
# Samples resident memory every 60 seconds and fails if growth exceeds 5%.
# Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Writes samples to /tmp/memory_2h.csv for later inspection.
# Requires D435i + atlas for a real run; no-arg path only validates usage.

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <map_atlas.osa>" >&2
}

if [ "$#" -ne 1 ]; then
  usage
  exit 1
fi

ATLAS_PATH=$1
if [ ! -f "$ATLAS_PATH" ]; then
  echo "atlas file not found: $ATLAS_PATH" >&2
  exit 1
fi

CSV_PATH=/tmp/memory_2h.csv
echo "elapsed_seconds,rss_kb" > "$CSV_PATH"

BINARY=/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online
VOCAB=/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt
SETTINGS=/home/jzq/MyJob/DexSlide/umi_mono/config/RealSense_D435i_online.yaml

"$BINARY" \
  -v "$VOCAB" \
  -s "$SETTINGS" \
  -l "$ATLAS_PATH" \
  --publisher stdout >/dev/null 2>&1 &
PID=$!

cleanup() {
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill -INT "$PID" >/dev/null 2>&1 || true
    wait "$PID" || true
  fi
}
trap cleanup EXIT

sleep 1
if ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "TASK-040 FAIL: realsense_online exited early" >&2
  exit 1
fi

INITIAL_RSS=$(ps -o rss= -p "$PID" | awk '{print $1}')
if [ -z "$INITIAL_RSS" ] || [ "$INITIAL_RSS" -le 0 ]; then
  echo "TASK-040 FAIL: could not read initial RSS" >&2
  exit 1
fi
echo "0,$INITIAL_RSS" >> "$CSV_PATH"

for minute in $(seq 1 120); do
  sleep 60
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "TASK-040 FAIL: process died before 2 hours" >&2
    exit 1
  fi
  RSS=$(ps -o rss= -p "$PID" | awk '{print $1}')
  echo "$((minute * 60)),$RSS" >> "$CSV_PATH"
done

FINAL_RSS=$(ps -o rss= -p "$PID" | awk '{print $1}')
kill -INT "$PID" >/dev/null 2>&1 || true
wait "$PID" || true
trap - EXIT

GROWTH_PERCENT=$(
  /usr/bin/python3 - <<PY
initial = float("$INITIAL_RSS")
final = float("$FINAL_RSS")
print(((final - initial) / initial) * 100.0)
PY
)

if /usr/bin/python3 - <<PY
growth = float("$GROWTH_PERCENT")
import sys
sys.exit(0 if growth < 5.0 else 1)
PY
then
  echo "TASK-040 PASS: RSS growth = ${GROWTH_PERCENT}%"
  exit 0
fi

echo "TASK-040 FAIL: RSS growth = ${GROWTH_PERCENT}% (samples: $CSV_PATH)" >&2
exit 1
