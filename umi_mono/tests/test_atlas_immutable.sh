#!/usr/bin/env bash
# TASK-023 integration check for atlas immutability in localization-only mode.
# Verifies that realsense_online with --load_map does not mutate the atlas file.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Leaves runtime logs on disk for diagnosis on failure.
# Safe for no-device hosts: atlas hash must remain unchanged either way.

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <atlas.osa> [duration_seconds]" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 1
fi

ATLAS_PATH=$1
DURATION=${2:-60}

if [ ! -f "$ATLAS_PATH" ]; then
  echo "atlas file not found: $ATLAS_PATH" >&2
  exit 1
fi

if [[ ! "$DURATION" =~ ^[1-9][0-9]*$ ]]; then
  echo "duration must be a positive integer: $DURATION" >&2
  exit 1
fi

PRE_HASH=$(sha256sum "$ATLAS_PATH" | awk '{print $1}')
LOG_PATH=$(mktemp /data/codes/DexSlide/umi_mono/tests/test_atlas_immutable.XXXXXX.log)

set +e
timeout "$DURATION" /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i.yaml \
  -l "$ATLAS_PATH" >"$LOG_PATH" 2>&1
RUN_EXIT=$?
set -e

if [ "$RUN_EXIT" -ne 0 ] && [ "$RUN_EXIT" -ne 124 ]; then
  echo "realsense_online failed with exit code $RUN_EXIT; log: $LOG_PATH" >&2
  cat "$LOG_PATH" >&2
  exit 1
fi

POST_HASH=$(sha256sum "$ATLAS_PATH" | awk '{print $1}')

if [ "$PRE_HASH" = "$POST_HASH" ]; then
  echo "PASS: atlas SHA-256 unchanged ($PRE_HASH)"
  exit 0
fi

echo "FAIL: atlas mutated $PRE_HASH -> $POST_HASH" >&2
echo "Run log: $LOG_PATH" >&2
exit 1
