#!/usr/bin/env bash
# Smoke test for realsense_topic_slam_node:
#  1. Launch the SLAM node with a given .osa atlas
#  2. Replay a given ROS2 bag
#  3. Count PoseStamped messages on /dexslide/slam/pose
#  4. Report metrics, exit 0 on pass / 1 on fail
#
# Usage:
#   bash scripts/test_realsense_topic_slam.sh \
#       --map  /data/codes/umi_mono_data/demos/mapping/map_atlas.osa \
#       --bag  /data/codes/umi_mono_data/aurco
#
# Expected on aurco bag (14s): pose count ≥ 200 (typical 280-320).

set -e
set -o pipefail

# ----------------------------- CLI parsing -----------------------------
MAP=""
BAG=""
MIN_POSES=100
WAIT_LOAD_S=10
WAIT_AFTER_BAG_S=2
POSE_TOPIC="/dexslide/slam/pose"
WORKSPACE_SETUP="/home/jzq/MyJob/DexSlide/umi_mono/ros2_ws/install/setup.bash"
ROS_SETUP="/opt/ros/jazzy/setup.bash"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map)        MAP="$2"; shift 2 ;;
    --bag)        BAG="$2"; shift 2 ;;
    --min-poses)  MIN_POSES="$2"; shift 2 ;;
    --pose-topic) POSE_TOPIC="$2"; shift 2 ;;
    --workspace-setup) WORKSPACE_SETUP="$2"; shift 2 ;;
    --ros-setup)  ROS_SETUP="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2 ;;
  esac
done

if [[ -z "$MAP" || -z "$BAG" ]]; then
  echo "ERROR: --map and --bag are required." >&2
  echo "Try: bash $0 --help" >&2
  exit 2
fi

if [[ ! -f "$MAP" ]]; then
  echo "ERROR: map atlas not found: $MAP" >&2
  exit 2
fi
if [[ ! -d "$BAG" ]]; then
  echo "ERROR: bag directory not found: $BAG" >&2
  exit 2
fi
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ERROR: ROS setup not found at $ROS_SETUP" >&2
  exit 2
fi
if [[ ! -f "$WORKSPACE_SETUP" ]]; then
  echo "ERROR: workspace setup not found at $WORKSPACE_SETUP" >&2
  echo "Did you run colcon build --packages-select dexslide_slam_publisher ?" >&2
  exit 2
fi

# ----------------------------- environment -----------------------------
# shellcheck source=/dev/null
source "$ROS_SETUP"
# shellcheck source=/dev/null
source "$WORKSPACE_SETUP"

TMP_DIR="$(mktemp -d -t realsense_topic_slam_test.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
SLAM_LOG="$TMP_DIR/slam.log"
POSE_LOG="$TMP_DIR/poses.txt"
BAG_LOG="$TMP_DIR/bag.log"

# ----------------------------- run --------------------------------------
PIDS=()
cleanup_procs() {
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
  done
  pkill -9 -f realsense_topic_slam_node 2>/dev/null || true
  pkill -9 -f 'ros2 bag play' 2>/dev/null || true
  pkill -9 -f 'ros2 topic echo' 2>/dev/null || true
}
trap 'cleanup_procs; rm -rf "$TMP_DIR"' EXIT INT TERM

echo "Step 1/5  Launching SLAM node..."
nohup ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  "map_atlas:=$MAP" > "$SLAM_LOG" 2>&1 &
LAUNCH_PID=$!
disown "$LAUNCH_PID" 2>/dev/null || true
PIDS+=("$LAUNCH_PID")

echo "Step 2/5  Waiting ${WAIT_LOAD_S}s for atlas load..."
sleep "$WAIT_LOAD_S"

if ! grep -qE 'Atlas loaded' "$SLAM_LOG"; then
  echo "ERROR: atlas did not load within ${WAIT_LOAD_S}s. Tail of log:" >&2
  tail -30 "$SLAM_LOG" >&2
  exit 1
fi
echo "  → Atlas loaded."

echo "Step 3/5  Starting pose subscriber..."
# Explicit QoS to match the publisher's rclcpp::SensorDataQoS (BEST_EFFORT +
# VOLATILE). `ros2 topic echo` defaults are usually compatible via auto-match,
# but the auto-match has a discovery race that occasionally returns 0 messages.
# Forcing the QoS removes that source of flakiness.
nohup bash -c "
  source '$ROS_SETUP' 2>/dev/null
  source '$WORKSPACE_SETUP' 2>/dev/null
  exec ros2 topic echo '$POSE_TOPIC' \
      --field header.stamp.sec \
      --no-arr \
      --qos-reliability best_effort \
      --qos-durability volatile \
      2>&1
" > "$POSE_LOG" &
ECHO_PID=$!
disown "$ECHO_PID" 2>/dev/null || true
PIDS+=("$ECHO_PID")

sleep 2  # let echo finish DDS discovery before bag starts

echo "Step 4/5  Replaying bag $BAG..."
nohup ros2 bag play "$BAG" > "$BAG_LOG" 2>&1 &
BAG_PID=$!
disown "$BAG_PID" 2>/dev/null || true
PIDS+=("$BAG_PID")

# Parse bag duration via `ros2 bag info`, which is the supported public API
# across rosbag2 versions; fall back to a hard error rather than guessing.
BAG_INFO_OUT="$TMP_DIR/bag_info.txt"
if ! ros2 bag info "$BAG" > "$BAG_INFO_OUT" 2>&1; then
  echo "ERROR: 'ros2 bag info $BAG' failed:" >&2
  cat "$BAG_INFO_OUT" >&2
  exit 1
fi
BAG_DURATION_S=$(awk '
  /^Duration:/ {
    val=$2
    if (val ~ /s$/) { sub("s$", "", val); printf "%d\n", val + 0.5; exit }
  }' "$BAG_INFO_OUT")
if [[ -z "$BAG_DURATION_S" || "$BAG_DURATION_S" -le 0 ]]; then
  echo "ERROR: could not parse Duration from 'ros2 bag info':" >&2
  cat "$BAG_INFO_OUT" >&2
  exit 1
fi
WAIT_S=$(( BAG_DURATION_S + WAIT_AFTER_BAG_S ))
echo "  → Bag duration = ${BAG_DURATION_S}s, waiting ${WAIT_S}s."
sleep "$WAIT_S"

# Drain echo buffers cleanly before counting: stop bag (if still running),
# stop echo, then read the log.
if kill -0 "$BAG_PID" 2>/dev/null; then
  kill -INT "$BAG_PID" 2>/dev/null || true
fi
if kill -0 "$ECHO_PID" 2>/dev/null; then
  kill -INT "$ECHO_PID" 2>/dev/null || true
fi
# Give the OS up to 3s to flush pipe buffers + write to disk.
for _ in 1 2 3; do
  if ! kill -0 "$ECHO_PID" 2>/dev/null; then break; fi
  sleep 1
done
sync

echo "Step 5/5  Collecting metrics..."
# Only count stamps that are non-zero (a header.stamp of 0 means the publisher
# never set it, which would mask broken pipelines as "passing").
POSE_COUNT=$(grep -cE '^[1-9][0-9]*$' "$POSE_LOG" || true)
ZERO_STAMPS=$(grep -cE '^0$' "$POSE_LOG" || true)
RELOC_COUNT=$(grep -c 'Relocalized!!' "$SLAM_LOG" || true)
ATLAS_COUNT=$(grep -c 'Atlas loaded' "$SLAM_LOG" || true)
SESSION_ORIGIN=$(grep -m1 'Session timestamp origin captured' "$SLAM_LOG" || echo "(not captured)")
# grep -c outputs "0\n" when there are no matches AND we used `|| true`; that's
# already a clean integer. Strip any stray whitespace just in case.
POSE_COUNT=${POSE_COUNT//[!0-9]/}
ZERO_STAMPS=${ZERO_STAMPS//[!0-9]/}
RELOC_COUNT=${RELOC_COUNT//[!0-9]/}
ATLAS_COUNT=${ATLAS_COUNT//[!0-9]/}
: "${POSE_COUNT:=0}" "${ZERO_STAMPS:=0}" "${RELOC_COUNT:=0}" "${ATLAS_COUNT:=0}"

echo ""
echo "================ Results ================"
echo "  pose count (sec>0)    : $POSE_COUNT"
echo "  pose count (sec==0)   : $ZERO_STAMPS"
echo "  relocalized events    : $RELOC_COUNT"
echo "  atlas-loaded events   : $ATLAS_COUNT"
echo "  session origin line   : $SESSION_ORIGIN"
echo "========================================="
echo ""

if [[ "$ZERO_STAMPS" -gt 0 ]]; then
  echo "FAIL: $ZERO_STAMPS pose messages had header.stamp.sec == 0 (broken timing)" >&2
  exit 1
fi

if [[ "$POSE_COUNT" -lt "$MIN_POSES" ]]; then
  echo "FAIL: pose count $POSE_COUNT < threshold $MIN_POSES" >&2
  echo "" >&2
  echo "POSE_LOG size = $(wc -c < "$POSE_LOG" 2>/dev/null || echo 0) bytes, $(wc -l < "$POSE_LOG" 2>/dev/null || echo 0) lines" >&2
  echo "POSE_LOG head (first 10 lines):" >&2
  head -10 "$POSE_LOG" >&2 2>/dev/null || true
  echo "..." >&2
  echo "POSE_LOG tail (last 10 lines):" >&2
  tail -10 "$POSE_LOG" >&2 2>/dev/null || true
  echo "" >&2
  if [[ "$POSE_COUNT" -eq 0 ]] && [[ "$(wc -c < "$POSE_LOG" 2>/dev/null || echo 0)" -eq 0 ]]; then
    echo "DIAGNOSIS: POSE_LOG is empty — the subscriber received NO messages." >&2
    echo "  Common causes:" >&2
    echo "  1. QoS mismatch: publisher is BEST_EFFORT (SensorDataQoS), subscriber must match." >&2
    echo "     Verify: ros2 topic info $POSE_TOPIC --verbose | grep Reliability" >&2
    echo "  2. DDS discovery never completed within the bag window." >&2
    echo "     Try: ROS_DOMAIN_ID=0 (or matching the publisher's domain)." >&2
    echo "  3. SLAM is publishing but never produced a valid pose (check SLAM log below)." >&2
  fi
  echo "Last 30 lines of SLAM log:" >&2
  tail -30 "$SLAM_LOG" >&2
  exit 1
fi

echo "PASS: pose count $POSE_COUNT ≥ threshold $MIN_POSES"
exit 0
