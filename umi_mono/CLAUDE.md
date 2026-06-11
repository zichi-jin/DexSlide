# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is the **mono-camera variant** of the Universal Manipulation Interface (UMI) data pipeline. It consumes wrist-camera demonstrations recorded as ROS1 `.bag` files or ROS2 bag directories (RealSense D435i color + IMU), runs ORB_SLAM3 in a Docker container, calibrates the SLAM frame to a world ArUco tag, and emits per-demo HDF5 files used downstream for diffusion-policy training.

The original multi-camera UMI README (`README.md`) describes the upstream Stanford project; the mono pipeline used here is documented in `SLAM_readme_mono.txt` (中文/English mix) and `IL_readme_mono.txt`. Treat `SLAM_readme_mono.txt` as the source of truth for the current workflow — `README.md` describes upstream defaults that no longer match this fork.

Important note on working-tree state: `diffusion_policy/`, `eval_real.py`, `calib.py`, and `assets/` are deleted from the working tree but still tracked by git (visible as `D` in `git status`). Code that imports `diffusion_policy.*` (including `tests/test_pose_util.py` and `train.py`) will not run as-is until those deletions are reverted or the modules are re-added.

## Two Parallel Pipelines

The pipeline has two entry points that share most stages but use different bag readers for stages 00 and 07:

- **ROS1**: `python run_slam_pipeline.py <session_dir>` — reads single-file `.bag` via `rosbag` + `cv_bridge`. Stages live in `scripts_slam_pipeline_ours/`.
- **ROS2**: `python run_slam_pipeline_ros2.py <session_dir>` — reads ROS2 bag directories (`metadata.yaml` + `.db3`/`.mcap`) via the `rosbags` Python library. Overrides stages 00 and 07 with files in `scripts_slam_pipeline_ros2/`; reuses stages 02–06 from `scripts_slam_pipeline_ours/`.

Default ROS2 topics target RealSense D435i: `/camera/camera/color/image_raw`, `/camera/camera/accel/sample`, `/camera/camera/gyro/sample`, optional unified `/camera/camera/imu`, plus `/sensor_data` for stage 07.

`ros2 bag record` cannot directly emit ROS1 `.bag` files. For ROS2 captures either use `run_slam_pipeline_ros2.py` (preferred) or convert with `rosbags-convert <ros2_dir> --dst out.bag` and run the ROS1 entry. The design rationale is in `docs/ros2_slam_pipeline_plan.md`.

## Session Directory Layout

Both pipelines expect the same on-disk shape (the `aurco` spelling is intentional and matches the code):

```
<session_dir>/
  aurco.bag | aurco/        # ArUco-13 base-tag calibration recording
  data_001.bag | data_*/    # demonstration recordings (ROS1 files OR ROS2 dirs)
  ...
```

After stage 00 runs, the pipeline materialises:

```
<session_dir>/demos/
  mapping/                  # promoted longest demo, used to build map_atlas.osa
  aurco/                    # base ArUco-13 calibration
  data_*/                   # one folder per demo with raw_video.mp4 + imu_data.json
```

Key intermediate outputs: `demos/mapping/map_atlas.osa`, `demos/aurco/tx_slam_tag.json`, `demos/data*/tag_detection_wrist.pkl`, `<session>/episode_*.hdf5`.

## Pipeline Stages

The orchestrator scripts call these stages in order. When debugging, you can re-run any individual stage script with the same arguments the orchestrator passes (each stage is idempotent on already-completed outputs and skips if the output exists, e.g. `02_create_map.py` skips when `map_atlas.osa` already exists).

| Stage | Script | Purpose |
|-------|--------|---------|
| 00 | `00_process_videos[_ros2].py` | Decode bag → `raw_video.mp4` + `imu_data.json` per demo; promote longest demo to `mapping/` |
| 02 | `02_create_map.py` | Run ORB_SLAM3 in Docker on mapping video → `map_atlas.osa` |
| 03 | `03_batch_slam.py` | Localise each demo against the map → `camera_trajectory.csv` |
| 04 | `04_detect_aruco.py` | Detect ArUco-13 (base) on `aurco/` demo and ArUco-10 (wrist) on each `data_*` |
| 05 | `05_run_calibrations.py` | Solve SLAM↔world tag transform → `tx_slam_tag.json` |
| 06 | `06_generate_dataset_plan.py` | Filter SLAM trajectories into valid contiguous segments per demo |
| 07 | `07_storage_hdf5[_ros2].py` | Re-read bag + segment plan → per-demo `episode_*.hdf5` |

Stage 02 requires Docker and pulls `chicheng/orb_slam3:latest` by default. Stage 05 calls `scripts/calibrate_slam_tag.py`. The legacy `07_generate_replay_buffer.py` and `08_generate_replay_buffer_from_rosbag.py` produce zarr replay buffers but are not in the active orchestration path (stage 08 is commented out in `run_slam_pipeline.py`).

## Setup

```bash
mamba env create -f conda_environment.yaml
conda activate umi
# ROS2 path also needs:
pip install rosbags
# Stage 02 needs Docker installed and access to chicheng/orb_slam3:latest
```

System packages required by the conda env: `libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf` (and `libspnav-dev spacenavd` for spacemouse-based eval).

## Common Commands

```bash
# Run full mono SLAM pipeline (ROS1)
python run_slam_pipeline.py <session_dir>

# Run full mono SLAM pipeline (ROS2)
python run_slam_pipeline_ros2.py <session_dir>
# With non-default topics:
python run_slam_pipeline_ros2.py <session_dir> \
    --image_topic /camera/camera/color/image_raw \
    --accel_topic /camera/camera/accel/sample \
    --gyro_topic /camera/camera/gyro/sample

# Override calibration directory (defaults to ./example/calibration)
python run_slam_pipeline.py <session_dir> --calibration_dir <path>

# Re-run a single stage manually (uses the cwd-as-repo-root convention)
python scripts_slam_pipeline_ours/03_batch_slam.py --input_dir <session>/demos --map_path <session>/demos/mapping/map_atlas.osa
```

`train.py` is the upstream diffusion-policy entry point but currently references the deleted `diffusion_policy/` package; do not assume it runs without restoring those files.

## Code Layout

- `umi/common/` — pose math, ORB_SLAM trajectory parsing, interpolation, timecode/IMU helpers. `pose_util_ours.py` is a mono-pipeline-specific extension of upstream `pose_util.py`.
- `umi/real_world/` — robot/gripper drivers (UR/Franka, WSG50), spacemouse, multi-cam plumbing. Used by upstream eval scripts.
- `umi/shared_memory/` — shared-memory queues/ring buffers used between camera/control processes.
- `umi/traj_eval/` — trajectory alignment and ATE/RPE error utilities.
- `umi/pipeline/aruco_detection.py` — ArUco detection logic invoked by stage 04.
- `scripts/` — calibration utilities (`calibrate_slam_tag.py`, hand-eye, latency) and one-off helpers; called by stage scripts via subprocess.
- `scripts_slam_pipeline_ours/` — ROS1 stages 00–07 plus legacy 08.
- `scripts_slam_pipeline_ros2/` — ROS2 stages 00 and 07 plus shared `ros2_bag_utils.py` (uses `rosbags.highlevel.AnyReader`).
- `example/calibration/` — `d435i_960_540.json` intrinsics, `aruco_config.yaml` (base, tag id 13), `aruco_config_wrist.yaml` (wrist, tag id 10).
- `config/RealSense_D435i*.yaml` — ORB_SLAM3 camera/IMU config consumed inside the Docker container.

Each stage script sets `ROOT_DIR = parent.parent` and `os.chdir(ROOT_DIR)` at import time, so all paths inside scripts assume the repo root as cwd. Preserve that pattern when adding new stages.

## Conventions Specific to This Fork

- `aurco` (sic) is the base-calibration directory name — keep this spelling; multiple stages and configs depend on it.
- Wrist ArUco tag id is 10, base ArUco tag id is 13; both are hardcoded in default configs and stage 07 invocations.
- New pipeline stages should keep the numbered prefix pattern (e.g. `06_generate_dataset_plan.py`).
- `black` is pinned in `requirements.txt`; format touched Python files with `python -m black <paths>`.
- Don't commit run artefacts: `data/`, `data_local/`, `outputs/`, `wandb/`, `example_demo_session/`.
- Tests are script-style (`tests/test_<feature>.py`); run with `python tests/test_pose_util.py`. Hardware tests under `tests/` need devices attached.

## When Adding ROS2 Support to a New Stage

If extending the ROS2 pipeline to read bag data, route through `scripts_slam_pipeline_ros2/ros2_bag_utils.py` rather than importing `rosbags` directly — it already handles `metadata.yaml` discovery, `AnyReader` setup, image-encoding fallbacks (`rgb8`/`bgr8`/`mono8`/`bgra8`), and split-vs-unified IMU topics. Match the existing pattern of overriding only the affected stage (00/07) and reusing ROS1 stages for everything else.

## Online tracking mode (added 2026-05)

Live 6-DoF pose @ 30Hz from D435i, using offline-built `.osa` atlas. ORB-SLAM3 fork at
`external/ORB_SLAM3_fork/` (pinned SHA `b741dca39015330ef4bcc3a85f89493503ade04b`), NOT a submodule.

### Build commands

```bash
bash scripts/check_host_env.sh
bash scripts/install_apt_deps.sh --apply           # one-time
bash scripts/build_pangolin.sh                     # one-time
bash scripts/build_sophus.sh                       # one-time
bash scripts/install_librealsense.sh --apply       # one-time
bash scripts/setup_orbslam3_fork.sh                # one-time
bash scripts/build_orbslam3_native.sh              # incremental
```

Build also touches the `ament_cmake` bridge package:

```bash
cd ros2_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select dexslide_slam_publisher
```

### Run commands

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py map_atlas:=<path.osa>
```

### Known failure modes

- System Python is 3.13 (anaconda) → `rclpy` NOT compatible. Use `/usr/bin/python3` (3.10) for Python tests.
- `cheng-chi` fork uses C++14 + `Thirdparty/Pangolin` empty dir → `external/Pangolin v0.8` via `CMAKE_PREFIX_PATH`.
- Vocabulary MD5: `5420bad0713bc97034dd2a9b2f0cc387` on this host's `ORBvoc.txt` (informational only).
- `librealsense2` dev pkg from Intel PPA, runtime resolves to `/opt/ros/jazzy`'s bundled `2.57`.

### Docs index

- `docs/online_tracking_research.md`     ← feasibility study
- `docs/online_tracking_implementation.md`  ← TASK tracker (43 tasks across 7 phases)
- `docs/setup_phase0_environment.md`     ← host prep
- `docs/setup_phase1_native_build.md`    ← fork integration
- `docs/setup_phase2_realsense_online.md`  ← binary implementation
- `docs/setup_phase3_4_publishers_robustness.md`  ← ZMQ + ROS2 + skew/MD5/recovery
- `docs/setup_phase5_python_consumer.md` ← Python `SlamPoseSubscriber`
