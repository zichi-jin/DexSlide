## Why

DexSlide's data glove records 20-DOF hand kinematics, but the world-frame palm pose currently relies on `umi_mono`'s ORB-SLAM3 pipeline running offline in Docker — frames are buffered to disk, then SLAM is post-processed against a pre-built `map_atlas.osa`. For live demonstrations, teleoperation, and closed-loop policy evaluation, we need 6-DoF camera pose at 30 Hz with sub-33 ms latency from a RealSense D435i. The algorithm side is already capable (cheng-chi/ORB_SLAM3 fork ships `--load_map` and preserves upstream `ActivateLocalizationMode()` + `TrackMonocular()` APIs); only the IO layer is batch-bound. See `umi_mono/docs/online_tracking_research.md` for the full investigation.

## What Changes

- Add a new native-Linux executable `realsense_online` inside the `cheng-chi/ORB_SLAM3` fork's `Examples/Monocular-Inertial/` directory that replaces file IO with live RealSense2 streaming.
- New executable explicitly calls `System::ActivateLocalizationMode()` after construction to prevent the loaded map from being mutated (closes a latent bug in `gopro_slam.cc` where the constructor comment "we want to localize" is aspirational and the call is missing).
- Publish 6-DoF camera pose at the camera frame rate via ROS2 `geometry_msgs/PoseStamped` on `/dexslide/slam/pose` and a tf2 broadcast `map → camera_color_optical_frame`.
- Wire a Python consumer (`dexslide.world_pose.SlamPoseSubscriber`) so the existing Vedo viewer / live hand reconstruction can read the world-frame palm transform.
- Add a launch file `dexslide_slam_online.launch.py` accepting `vocab`, `settings_yaml` (with `System.LoadAtlasFromFile` field), `image_topic`, `imu_topic`, `pose_topic` parameters.
- **BREAKING (for users of the offline pipeline)**: none. The existing Docker batch path (`02_create_map.py`, `03_batch_slam.py`) remains untouched. Online tracking is an additive capability.

## Capabilities

### New Capabilities
- `slam-online-tracker`: Live monocular-inertial SLAM tracker that loads a pre-built `.osa` atlas, runs ORB-SLAM3 in localization-only mode against a live RealSense D435i feed, and publishes 6-DoF pose at ≥30 Hz with bounded latency. Owns the realsense_online binary, launch file, and ROS2 publishing contract.
- `slam-pose-bridge`: Python-side consumer of the ROS2 pose topic that exposes a thread-safe `get_T_world_camera(t)` API to the DexSlide kinematics / viewer stack. Owns the rclpy subscriber, ring buffer, and time-aligned interpolation behavior.

### Modified Capabilities
- (none — no existing OpenSpec specs in this repo yet)

## Impact

- **Code**:
  - New: ~600 LOC in cheng-chi/ORB_SLAM3 fork (`Examples/Monocular-Inertial/realsense_online.cc`, `realsense_online_imu.hpp`, CMakeLists target), ~250 LOC in DexSlide (`dexslide/world_pose/slam_pose_subscriber.py`, launch file).
  - Touched: `umi_mono/CLAUDE.md` (document the new online path), `umi_mono/docs/online_tracking_research.md` (already written, referenced).
  - Untouched: `umi_mono/run_slam_pipeline.py`, `02_create_map.py`, `03_batch_slam.py` — offline pipeline remains the source of `map_atlas.osa`.
- **Dependencies**:
  - New native-host build deps: `librealsense2-dev` (≥2.55), `libpangolin-dev` or build-from-source, `libeigen3-dev`, `libopencv-dev (≥4.6)`, `libboost-serialization-dev`, `libsophus-dev`, ROS2 Humble (`rclcpp`, `cv_bridge`, `tf2_ros`, `geometry_msgs`, `sensor_msgs`).
  - Python: `rclpy`, already implied by ROS2 install.
  - Removed: at runtime, Docker is no longer required for tracking (still required for offline mapping until a future change migrates that too).
- **APIs**:
  - New ROS2 topic `/dexslide/slam/pose` (PoseStamped, 30 Hz), new tf2 broadcast `map → camera_color_optical_frame`.
  - New Python module `dexslide.world_pose` with `SlamPoseSubscriber` class.
- **Systems**:
  - Native build of cheng-chi/ORB_SLAM3 fork on Ubuntu 22.04 (new build artifact path on the host).
  - Linker discipline: ORB-SLAM3 binary stays in fork's build dir; DexSlide does not vendor or relink its sources.
- **Performance / SLOs**:
  - p99 per-frame latency < 33 ms; p99 publish-to-subscriber latency < 50 ms; sustained 30 Hz for ≥ 1 hour without memory growth > 5% RSS.
  - Relocalization recovery time from tracking lost < 5 s in known environment.
- **Risks**:
  - cheng-chi fork is a snapshot (0 open issues, no active upstream maintenance) — future ORB-SLAM3 fixes won't propagate automatically. Mitigation: pin a Git SHA; isolate node code from fork internals.
  - RealSense IMU/image timestamp domains must be reconciled (`RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME` or `HARDWARE_CLOCK`); skew > 10 ms breaks IMU preintegration.
  - Headless Pangolin viewer crashes if `bUseViewer=true` without `$DISPLAY`. Mitigation: force `bUseViewer=false`; consider `-DUSE_PANGOLIN=OFF` build flag.
