#!/usr/bin/env bash
# TASK-043 manual recovery-time check for realsense_online.
# Human interaction required: operator occludes the D435i lens for 3 seconds.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Measures time from occlusion release to first subsequent pose line in the log.
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

FIFO_PATH=$(mktemp -u /tmp/recovery_stream.XXXXXX)
LOG_PATH=/tmp/recovery_log.txt
rm -f "$FIFO_PATH" "$LOG_PATH"
mkfifo "$FIFO_PATH"

BINARY=/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online
VOCAB=/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt
SETTINGS=/data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml

/usr/bin/python3 -u - "$FIFO_PATH" "$LOG_PATH" <<'PY' &
import sys
import time

fifo_path = sys.argv[1]
log_path = sys.argv[2]
with open(fifo_path, "r", encoding="utf-8", errors="replace") as src, open(log_path, "w", encoding="utf-8") as dst:
    for line in src:
        dst.write(f"{time.time():.6f}\t{line}")
        dst.flush()
PY
LOGGER_PID=$!

"$BINARY" \
  -v "$VOCAB" \
  -s "$SETTINGS" \
  -l "$ATLAS_PATH" \
  --publisher stdout >"$FIFO_PATH" 2>&1 &
BINARY_PID=$!

cleanup() {
  kill -INT "$BINARY_PID" >/dev/null 2>&1 || true
  wait "$BINARY_PID" || true
  kill "$LOGGER_PID" >/dev/null 2>&1 || true
  wait "$LOGGER_PID" || true
  rm -f "$FIFO_PATH"
}
trap cleanup EXIT

sleep 10
echo 'Operator: please occlude the D435i lens NOW for 3 seconds, then release.'
sleep 6
T_RELEASE=$(date +%s.%N)

T_RECOVER=$(
  /usr/bin/python3 - "$LOG_PATH" "$T_RELEASE" <<'PY'
import sys
import time
from pathlib import Path

log_path = Path(sys.argv[1])
t_release = float(sys.argv[2])
deadline = time.time() + 30.0

while time.time() < deadline:
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "\tpose " not in line:
                    continue
                try:
                    receipt = float(line.split("\t", 1)[0])
                except ValueError:
                    continue
                if receipt > t_release:
                    print(f"{receipt:.6f}")
                    sys.exit(0)
    time.sleep(0.2)

sys.exit(1)
PY
) || {
  echo "TASK-043 FAIL: no recovered pose observed within 30s after release" >&2
  exit 1
}

RECOVERY_SECONDS=$(
  /usr/bin/python3 - <<PY
release_t = float("$T_RELEASE")
recover_t = float("$T_RECOVER")
print(recover_t - release_t)
PY
)

kill -INT "$BINARY_PID" >/dev/null 2>&1 || true
wait "$BINARY_PID" || true
kill "$LOGGER_PID" >/dev/null 2>&1 || true
wait "$LOGGER_PID" || true
rm -f "$FIFO_PATH"
trap - EXIT

if /usr/bin/python3 - <<PY
recovery = float("$RECOVERY_SECONDS")
import sys
sys.exit(0 if recovery <= 5.0 else 1)
PY
then
  echo "TASK-043 PASS: recovery time = ${RECOVERY_SECONDS}s"
  exit 0
fi

echo "TASK-043 FAIL: recovery time = ${RECOVERY_SECONDS}s" >&2
exit 1
