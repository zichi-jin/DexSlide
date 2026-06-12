#!/usr/bin/env bash
# TASK-005 D435i smoke-test runner for online tracking mode.
# Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Compiles the local librealsense2 smoke test against the installed dev package.
# Requires librealsense2-dev to be present and never uses sudo.
# Prints the test exit code and propagates it to the caller.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="${SCRIPT_DIR}/test_d435i_streams.cpp"
BIN_FILE="${SCRIPT_DIR}/test_d435i_streams"

if ! pkg-config --modversion realsense2 >/dev/null 2>&1; then
  echo "Run install_librealsense.sh --apply first"
  exit 1
fi

g++ -std=c++17 -O2 -o "${BIN_FILE}" "${SRC_FILE}" $(pkg-config --cflags --libs realsense2) -lpthread

set +e
"${BIN_FILE}"
rc=$?
set -e

echo "exit ${rc}"
exit "${rc}"
