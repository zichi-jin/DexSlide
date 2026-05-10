# ROS2 SLAM Pipeline Migration Plan (Non-intrusive)

## Scope
- Keep existing ROS1 pipeline code unchanged.
- Implement a parallel ROS2 pipeline in new files/directories.
- Update `SLAM_readme_mono.txt` with ROS2 validated workflow and caveats.

## Constraints & Findings
- Current pipeline hard-depends on ROS1 `rosbag` + `cv_bridge` in:
  - `scripts_slam_pipeline_ours/00_process_videos.py`
  - `scripts_slam_pipeline_ours/07_storage_hdf5.py`
- ROS2 `ros2 bag record` output is a **directory** (`metadata.yaml` + `.db3`/`.mcap`), not ROS1 single-file `.bag`.
- RealSense ROS2 topics for D435i are commonly:
  - `/camera/camera/color/image_raw`
  - `/camera/camera/accel/sample`
  - `/camera/camera/gyro/sample`
- Therefore, direct复用 ROS1 bag reader不可行；需要二选一：
  1. 改读 ROS2 bag（推荐）
  2. 先把 rosbag2 转成 rosbag1 再跑老脚本

## Design Decision
- Primary path: implement ROS2-native readers (based on `rosbags` Python lib) for Stage 00 and Stage 07.
- Keep Stage 02~06 unchanged and reused (they operate on generated files, not raw bag API).
- Add optional documented conversion fallback (`rosbags-convert`) for compatibility/emergency use.

## Execution Steps
1. Create new ROS2 pipeline entry script: `run_slam_pipeline_ros2.py`.
2. Create new folder `scripts_slam_pipeline_ros2/`.
3. Implement shared ROS2 bag utility module:
   - discover ROS2 bag directories
   - read messages via `AnyReader`
   - decode `sensor_msgs/Image` without ROS1 `cv_bridge`
4. Implement ROS2 Stage 00:
   - parse color/IMU topics from rosbag2
   - export `raw_video.mp4` + `imu_data.json`
   - keep existing naming behavior for `demos/` and `mapping`
5. Implement ROS2 Stage 07:
   - parse images (+ optional `/sensor_data`)
   - keep output HDF5 schema compatible with existing training flow
6. Update `SLAM_readme_mono.txt`:
   - ROS2 recording command and naming conventions
   - dependency notes (`rosbags`)
   - answer “是否能直接录成 ROS1 bag”的结论与可选转换命令
7. Validate:
   - Python syntax check for new files
   - dry-run style checks for argument parsing and path discovery

## Risks & Mitigations
- `rosbags` not preinstalled:
  - Mitigation: explicit install command and runtime import error hint.
- Image encoding mismatch (`rgb8`/`bgr8`/`mono8`/`bgra8`):
  - Mitigation: robust conversion handling in utility layer.
- IMU may be unified topic (`/camera/camera/imu`) on some setups:
  - Mitigation: support both split accel/gyro and unified imu topic.

## Deliverables
- `docs/ros2_slam_pipeline_plan.md` (this plan)
- `run_slam_pipeline_ros2.py`
- `scripts_slam_pipeline_ros2/` new implementation files
- updated `SLAM_readme_mono.txt`
