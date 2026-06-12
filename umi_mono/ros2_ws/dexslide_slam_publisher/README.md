# dexslide_slam_publisher

ROS2 Humble `ament_cmake` package providing executables for online SLAM pose publication and optional ArUco world-frame localization:

| Executable | Source | Use case |
|---|---|---|
| `realsense_topic_slam_node` | `src/realsense_topic_slam_node.cpp` | **NEW**. Subscribes to RealSense `image_raw` + `accel/sample` + `gyro/sample` ROS2 topics, runs ORB-SLAM3 against a pre-built `.osa` atlas, publishes `PoseStamped` + tf2. Works with either live `realsense2_camera` or `ros2 bag play`. |
| `aruco_world_pose_node.py` | `scripts/aruco_world_pose_node.py` | Optional. Subscribes to camera images + `/dexslide/slam/pose`, detects `target_marker_id`, applies `tx_slam_tag.json`, and publishes the marker pose in the calibrated world frame. |
| `pose_publisher_node` | `src/pose_publisher_node.cpp` | Legacy. ZMQ → ROS2 bridge for the librealsense-direct `realsense_online` binary in the ORB-SLAM3 fork. |

Pose output topic: `/dexslide/slam/pose` (`geometry_msgs/PoseStamped`)
tf2 broadcast: `map` → `camera_color_optical_frame`
Optional ArUco world-pose topic: `/dexslide/aruco/world_pose`
Optional ArUco tf2 broadcast: `world` → `aruco_marker`

## Build

```bash
source /opt/ros/humble/setup.bash
cd /data/codes/DexSlide/umi_mono/ros2_ws
colcon build --packages-select dexslide_slam_publisher
source install/setup.bash
```

Prerequisites: Pangolin + Sophus + ORB-SLAM3 fork must already be built (see
`docs/setup_phase0_environment.md` → `setup_phase1_native_build.md` →
`setup_phase2_realsense_online.md`).

## Run (topic-driven node, recommended)

```bash
# Launch SLAM node with a map
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/path/to/map_atlas.osa

# Then either replay a bag:
ros2 bag play /path/to/bag
# Or run a real D435i:
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true unite_imu_method:=0
```

To also publish a specified ArUco marker in the calibrated world frame:

```bash
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/path/to/map_atlas.osa \
  enable_aruco_world:=true \
  tx_slam_tag:=/path/to/demos/aurco/tx_slam_tag.json \
  target_marker_id:=10 \
  aruco_yaml:=/data/codes/DexSlide/umi_mono/example/calibration/aruco_config_wrist.yaml
```

See [docs/realsense_topic_slam_usage.md](../../docs/realsense_topic_slam_usage.md)
for the full guide, parameter table, environment requirements, and troubleshooting.

## Run (legacy ZMQ bridge)

```bash
# Start the librealsense-direct binary first:
ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py \
  map_atlas:=/path/to/map_atlas.osa
```

This launches both `realsense_online` and `pose_publisher_node` together.

## Test

```bash
bash /data/codes/DexSlide/umi_mono/scripts/test_realsense_topic_slam.sh \
  --map /data/codes/umi_mono_data/demos/mapping/map_atlas.osa \
  --bag /data/codes/umi_mono_data/aurco
```

Expected on the aurco bag (14 s): pose count ≥ 200 (typical 280-320). Exit code 0 = pass.

## Python requirements

The core SLAM runtime is C++. The optional ArUco world-pose node and auxiliary
Python validators need:

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

Use the **system** Python (`/usr/bin/python3`, 3.10) — anaconda Python is
incompatible with `rclpy` on this fork.
