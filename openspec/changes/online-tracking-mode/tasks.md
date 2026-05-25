## 1. Prep & build environment

- [ ] 1.1 Pin a specific commit SHA of `cheng-chi/ORB_SLAM3` in `umi_mono/external/ORB_SLAM3_fork/.git_pinned_sha` and document it in `umi_mono/CLAUDE.md`
- [ ] 1.2 Install host build deps: `libeigen3-dev`, `libopencv-dev (>=4.6)`, `libboost-serialization-dev`, `libboost-system-dev`, `libpangolin-dev` (or build from source), `librealsense2-dev (>=2.55)`, `libsophus-dev`
- [ ] 1.3 Install ROS2 Humble (`ros-humble-desktop`, `ros-humble-cv-bridge`, `ros-humble-tf2-ros`, `ros-humble-geometry-msgs`, `ros-humble-sensor-msgs`)
- [ ] 1.4 Clone the fork as a git submodule at `umi_mono/external/ORB_SLAM3_fork/` pinned to the SHA from 1.1
- [ ] 1.5 Verify native build of the fork's existing `gopro_slam` binary on the host (no Docker) and confirm it produces a trajectory matching the Docker output on a known recording

## 2. Online tracker binary — minimal stdout version

- [ ] 2.1 Add `Examples/Monocular-Inertial/realsense_online.cc` in the fork: CLI parsing for `--vocabulary`, `--setting`, `--load_map` (settings YAML override path), `--publisher {stdout,ros2,zmq}`, `--exposure-us`, `--max_lost_frames`
- [ ] 2.2 Add `Examples/Monocular-Inertial/imu_ring_buffer.hpp`: header-only lock-free SPSC ring buffer with `push(IMU::Point)` and `drain_until(timestamp) -> vector<IMU::Point>`
- [ ] 2.3 Implement RealSense2 pipeline setup: color 960×540 BGR8 @30Hz + accel @200Hz + gyro @200Hz, `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`, fixed exposure
- [ ] 2.4 Implement IMU callback wiring (accel+gyro frames → `IMU::Point` constructor → ring buffer)
- [ ] 2.5 Construct `System` with settings YAML (which contains `System.LoadAtlasFromFile`), `IMU_MONOCULAR`, `bUseViewer=false`; verify atlas loads
- [ ] 2.6 Call `SLAM.ActivateLocalizationMode()` immediately after construction and log `Localization mode active, atlas read-only`
- [ ] 2.7 Implement main tracking loop: `pipe.wait_for_frames(100)` → drain IMU → `SLAM.TrackMonocular(im, t, vImu)` → print pose to stdout
- [ ] 2.8 Add startup skew measurement: median image/IMU timestamp delta over first 100 frames; abort if > 5 ms
- [ ] 2.9 Add MD5 check on vocab vs atlas vocab fingerprint; abort with clear message on mismatch
- [ ] 2.10 Add atlas-byte-identity guard: compute SHA-256 of atlas before/after a 60 s session in a smoke test
- [ ] 2.11 Wire `CMakeLists.txt` target `realsense_online` linking ORB_SLAM3, librealsense2, Sophus, Pangolin (link-only), OpenCV

## 3. ROS2 publisher backend

- [ ] 3.1 Add `Examples/Monocular-Inertial/ros2_pose_publisher.hpp` exposing `class Ros2PosePublisher` with `publish(Sophus::SE3f Tcw, double t)` and a tf2 broadcaster
- [ ] 3.2 Add `Examples/Monocular-Inertial/zmq_pose_publisher.hpp` exposing the same interface over ZMQ PUB on `tcp://127.0.0.1:5555`
- [ ] 3.3 Refactor `realsense_online.cc` to dispatch on `--publisher` flag to the chosen backend
- [ ] 3.4 Add CMake option `BUILD_ROS2_PUBLISHER=ON` (default ON if `rclcpp` found); fall back to stdout+zmq only when OFF
- [ ] 3.5 Verify `/dexslide/slam/pose` (`geometry_msgs/PoseStamped`) and tf2 `map → camera_color_optical_frame` appear in `ros2 topic list` / `ros2 run tf2_tools view_frames`

## 4. Tracking loss handling

- [ ] 4.1 Set `max_lost_frames` default to 900 (~30 s) in `realsense_online.cc`
- [ ] 4.2 Replace fork's exit-on-exceed behavior with log-and-continue in this binary (no `exit()` on tracking loss)
- [ ] 4.3 Publish nothing on `/dexslide/slam/pose` while tracking is `LOST` / `NOT_INITIALIZED`
- [ ] 4.4 Publish a `/dexslide/slam/diagnostics` (`std_msgs/String` JSON) at 1 Hz with tracker state and uptime
- [ ] 4.5 Integration test: occlude camera 3 s in mapped area, verify pose resumes within 5 s and PID unchanged

## 5. Python consumer (`dexslide.world_pose`)

- [ ] 5.1 Create `dexslide/world_pose/__init__.py` with `__all__ = ["SlamPoseSubscriber"]`
- [ ] 5.2 Implement `dexslide/world_pose/slam_pose_subscriber.py` with constructor `(node_name='dexslide_consumer', topic='/dexslide/slam/pose', stale_after_seconds=0.2, buffer_size=300)`
- [ ] 5.3 Implement bounded `collections.deque` ring buffer with `threading.Lock`, storing `(t, T_world_camera_np)` tuples
- [ ] 5.4 Implement `latest()`, `is_tracking()`, `get_T_world_camera(t=None)` (with linear-translation + SLERP-rotation interpolation; return None outside ±100 ms)
- [ ] 5.5 Implement `spin_in_thread()` daemon thread running its own `SingleThreadedExecutor`
- [ ] 5.6 Add unit test `tests/test_slam_pose_subscriber.py` with mocked rclpy node verifying concurrent access invariants and stale flag transitions

## 6. Launch + config

- [ ] 6.1 Add `launch/dexslide_slam_online.launch.py` accepting parameters `vocab`, `settings_yaml`, `pose_topic`, `image_topic` (unused, kept for future), `imu_topic` (unused, kept for future), `publisher`, `exposure_us`
- [ ] 6.2 Add `config/RealSense_D435i_online.yaml` derived from `RealSense_D435i.yaml` with `System.LoadAtlasFromFile: <override-at-launch>` and viewer fields zeroed
- [ ] 6.3 Document the launch usage in `umi_mono/SLAM_readme_mono.txt` under a new section `## 7. 在线追踪 (Online tracking)`

## 7. Validation & SLO verification

- [ ] 7.1 Smoke test on a 30-min stationary recording: assert `realsense_online --playback-mode` ATE vs Docker batch < 2 cm
- [ ] 7.2 Live test in mapped area: 30-min run, verify p99 inter-message interval ≤ 50 ms and p99 capture-to-publish latency < 33 ms
- [ ] 7.3 Headless test: `unset DISPLAY && realsense_online ...` runs without X11 errors and publishes pose
- [ ] 7.4 Memory test: 2-hour run, assert RSS growth < 5%
- [ ] 7.5 Atlas immutability test: SHA-256 of `map_atlas.osa` unchanged before/after 60 s session
- [ ] 7.6 Vocabulary mismatch test: deliberately use a wrong vocab and assert clear error + non-zero exit within 5 s
- [ ] 7.7 Recovery test: occlude 3 s and 30 min separately; verify behaviors per spec scenarios
- [ ] 7.8 Skew abort test: simulate bad timestamp domain config and verify abort within 4 s

## 8. Documentation & handoff

- [ ] 8.1 Update `umi_mono/CLAUDE.md` with: pinned fork SHA, native build commands, launch invocation, supported failure modes
- [ ] 8.2 Update `umi_mono/docs/online_tracking_research.md` "Implementation status" section once 7.x are green
- [ ] 8.3 Update root `CLAUDE.md` (`/data/codes/DexSlide/CLAUDE.md`) `World-frame localization (planned)` line to point at the new launch file
- [ ] 8.4 Open a follow-up OpenSpec change `migrate-mapping-to-native` for moving stage 02 / 03 off Docker (out of scope here)
