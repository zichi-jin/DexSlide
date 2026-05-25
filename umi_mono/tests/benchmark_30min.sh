#!/usr/bin/env bash
# TASK-038 30-minute stdout pose benchmark for realsense_online.
# Measures inter-message timing and capture-to-publish latency from stdout pose lines.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Writes summary JSON to /tmp/benchmark_30min_YYYYMMDD_HHMMSS.json.
# Requires D435i + atlas for a real run; no-arg path is safe on any host.

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <map_atlas.osa> [duration_minutes]" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 1
fi

ATLAS_PATH=$1
DURATION_MINUTES=${2:-30}
if [ ! -f "$ATLAS_PATH" ]; then
  echo "atlas file not found: $ATLAS_PATH" >&2
  exit 1
fi
if [[ ! "$DURATION_MINUTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "duration_minutes must be a positive integer: $DURATION_MINUTES" >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
JSON_PATH=/tmp/benchmark_30min_${STAMP}.json
export DEXSLIDE_BENCHMARK_JSON="$JSON_PATH"

DURATION_SECONDS=$((DURATION_MINUTES * 60))
BINARY=/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online
VOCAB=/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt
SETTINGS=/data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml

set +e
timeout "${DURATION_SECONDS}" "$BINARY" \
  -v "$VOCAB" \
  -s "$SETTINGS" \
  -l "$ATLAS_PATH" \
  --publisher stdout | /usr/bin/python3 - <<'PY'
import json
import math
import os
import sys
import time


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    alpha = pos - lo
    return float(sorted_values[lo] * (1.0 - alpha) + sorted_values[hi] * alpha)


inter_ms = []
latency_ms = []
hist = {"under_16ms": 0, "16_33ms": 0, "33_50ms": 0, "over_50ms": 0}
last_wall = None
last_report = time.monotonic()

for line in sys.stdin:
    if not line.startswith("pose "):
        continue
    parts = line.strip().split()
    if len(parts) < 3:
        continue
    recv_wall = time.time()
    try:
        ts = float(parts[1])
    except ValueError:
        continue

    if last_wall is not None:
        delta_ms = (recv_wall - last_wall) * 1000.0
        inter_ms.append(delta_ms)
        if delta_ms < 16.0:
            hist["under_16ms"] += 1
        elif delta_ms < 33.0:
            hist["16_33ms"] += 1
        elif delta_ms <= 50.0:
            hist["33_50ms"] += 1
        else:
            hist["over_50ms"] += 1
    last_wall = recv_wall

    latency_ms.append((recv_wall - ts) * 1000.0)

    now = time.monotonic()
    if now - last_report >= 60.0 and inter_ms and latency_ms:
        inter_sorted = sorted(inter_ms)
        lat_sorted = sorted(latency_ms)
        print(
            "rolling 60s: inter p50={:.3f}ms p99={:.3f}ms | latency p50={:.3f}ms p99={:.3f}ms".format(
                percentile(inter_sorted, 0.5),
                percentile(inter_sorted, 0.99),
                percentile(lat_sorted, 0.5),
                percentile(lat_sorted, 0.99),
            )
        )
        last_report = now

inter_sorted = sorted(inter_ms)
lat_sorted = sorted(latency_ms)
summary = {
    "samples": len(latency_ms),
    "inter_message_ms": {
        "p50": percentile(inter_sorted, 0.5),
        "p99": percentile(inter_sorted, 0.99),
        "p999": percentile(inter_sorted, 0.999),
    },
    "capture_to_publish_ms": {
        "p50": percentile(lat_sorted, 0.5),
        "p99": percentile(lat_sorted, 0.99),
        "p999": percentile(lat_sorted, 0.999),
    },
    "histogram": hist,
}

json_path = os.environ["DEXSLIDE_BENCHMARK_JSON"]
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(json.dumps(summary, indent=2, sort_keys=True))

inter_p99 = summary["inter_message_ms"]["p99"]
lat_p99 = summary["capture_to_publish_ms"]["p99"]
if inter_p99 is None or lat_p99 is None:
    sys.exit(1)
sys.exit(0 if inter_p99 <= 50.0 and lat_p99 <= 33.0 else 1)
PY
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e

RUN_EXIT=${PIPE_STATUS[0]}
ANALYSIS_EXIT=${PIPE_STATUS[1]}

if [ "$RUN_EXIT" -ne 0 ] && [ "$RUN_EXIT" -ne 124 ]; then
  echo "benchmark run failed with exit code $RUN_EXIT" >&2
  exit 1
fi

if [ "$ANALYSIS_EXIT" -ne 0 ]; then
  echo "TASK-038 FAIL: thresholds violated; stats at $JSON_PATH" >&2
  exit 1
fi

echo "TASK-038 PASS: stats written to $JSON_PATH"
exit 0
