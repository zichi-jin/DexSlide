#!/usr/bin/env bash
# TASK-042 vocab mismatch failure harness for realsense_online.
# Verifies that an obviously wrong ORB vocabulary fails gracefully and quickly.
# Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Uses a temporary fake vocab under /tmp and expects a non-zero binary exit.
# Leaves the captured log on disk only on failure for diagnosis.

set -euo pipefail

FAKE_VOCAB=/tmp/fake_vocab.txt
LOG_PATH=$(mktemp /home/jzq/MyJob/DexSlide/umi_mono/tests/test_vocab_mismatch.XXXXXX.log)
SETTINGS=/home/jzq/MyJob/DexSlide/umi_mono/config/RealSense_D435i_online.yaml
BINARY=/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online

echo 'not orb voc' > "$FAKE_VOCAB"

set +e
timeout 6 "$BINARY" -v "$FAKE_VOCAB" -s "$SETTINGS" --publisher stdout >"$LOG_PATH" 2>&1
RUN_EXIT=$?
set -e

if [ "$RUN_EXIT" -eq 0 ]; then
  echo "TASK-042 FAIL: realsense_online unexpectedly exited 0 with fake vocab" >&2
  cat "$LOG_PATH" >&2
  exit 1
fi

if grep -Eq 'Vocabulary file size suspicious|Failed to open vocabulary|vocabulary|Wrong path to vocabulary|TASK-012 smoke setup failed' "$LOG_PATH"; then
  echo "TASK-042 PASS: wrong vocab failed gracefully (exit $RUN_EXIT)"
  exit 0
fi

echo "TASK-042 FAIL: fake vocab produced no recognizable error" >&2
cat "$LOG_PATH" >&2
exit 1
