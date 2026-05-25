#!/usr/bin/env bash
# TASK-039 headless runtime check for realsense_online.
# Runs localization-only mode with DISPLAY unset and rejects X11/display errors.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Timeout exit 124 is acceptable; clean exit 0 is also acceptable.
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

LOG_PATH=/tmp/headless_run.log
BINARY=/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online
VOCAB=/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt
SETTINGS=/data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml

set +e
(
  unset DISPLAY
  timeout 30 "$BINARY" \
    -v "$VOCAB" \
    -s "$SETTINGS" \
    -l "$ATLAS_PATH" \
    --publisher stdout 2>&1
) | tee "$LOG_PATH"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e

RUN_EXIT=${PIPE_STATUS[0]}

if grep -Eiq 'X11|Xlib|DISPLAY|cannot open display' "$LOG_PATH"; then
  echo "TASK-039 FAIL: display-related error found in $LOG_PATH" >&2
  exit 1
fi

if [ "$RUN_EXIT" -ne 0 ] && [ "$RUN_EXIT" -ne 124 ]; then
  echo "TASK-039 FAIL: realsense_online exit code $RUN_EXIT" >&2
  exit 1
fi

echo "TASK-039 PASS"
exit 0
