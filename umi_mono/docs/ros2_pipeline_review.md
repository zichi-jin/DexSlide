# `run_slam_pipeline_ros2.py` Pipeline Review

## Scope of review
The orchestrator `run_slam_pipeline_ros2.py` itself is mostly a thin click + `subprocess.run` driver. Most real risks live in the ROS2-specific stages it invokes:
- `scripts_slam_pipeline_ros2/00_process_videos_ros2.py`
- `scripts_slam_pipeline_ros2/07_storage_hdf5_ros2.py`
- `scripts_slam_pipeline_ros2/ros2_bag_utils.py`

So the report covers the orchestrator + the call graph it triggers.

---

## A. High-severity issues (fix before next run)

### A1. IMU sample duplication when bag has both unified `/imu` and split `accel/gyro`
**Where:** `00_process_videos_ros2.py:73-120`
**What:** The loop has independent `elif` branches for `accel_topic`, `gyro_topic`, and `imu_topic`. `expected_topics = {image_topic, accel_topic, gyro_topic, imu_topic}` matches all four if all four are present in the bag — and they can be: with `unite_imu_method >= 1` the realsense2 driver publishes the merged `/camera/camera/imu` AND keeps `/camera/camera/accel/sample` + `/camera/camera/gyro/sample`. ([RealSense ROS2 docs](https://dev.realsenseai.com/docs/ros2-wrapper/), [Issue #598](https://github.com/realsenseai/realsense-ros/issues/598))
**Impact:** `accel_samples` / `gyro_samples` end up with each timestamp duplicated; sorted, sent to ORB_SLAM3 via `imu_data.json`, this looks like a doubled IMU rate. ORB_SLAM3 IMU pre-integration is sensitive to monotonically increasing, non-duplicate timestamps and will produce broken biases or refuse the frame ([UZ-SLAMLab/ORB_SLAM3#67](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/67)).
**Trigger:** any user who turns on `unite_imu_method` (we hint at it implicitly by accepting an `--imu_topic` default).
**Fix sketch (do NOT apply yet):** prefer `imu_topic` if present and skip split topics; otherwise use split. I.e., make it an either/or selection done before iteration.

### A2. Failed-bag leaves a zombie output directory; re-runs then silently skip it
**Where:** `00_process_videos_ros2.py:45` (`output_dir.mkdir(...)`) + `00_process_videos_ros2.py:229-231` (`if output_dir.exists(): skipping`).
**What:** `process_ros2_bag` creates the destination directory before doing any work. If anything later raises (no images, unsupported encoding, OpenCV codec failure on `out.release()` etc.), `process_single_bag` catches the exception and prints "Error processing", but the empty directory persists. On the next run, `process_single_bag` short-circuits with "already exists, skipping" and never reprocesses that bag.
**Impact:** silent permanent loss of a demo across runs. The user cannot tell from the orchestrator log that the bag is broken — only that "Successfully processed N out of M".
**Fix sketch:** create dir only after successful read, or move outputs to a temp dir then atomically rename, or check for both `raw_video.mp4` and `imu_data.json` instead of "directory exists".

### A3. mp4v constant-rate VideoWriter desyncs vs. real timestamps fed to IMU JSON
**Where:** `00_process_videos_ros2.py:154-172`.
**What:** `actual_fps` is computed as `(n-1)/duration` (an *average*), then fed to `cv2.VideoWriter(..., fourcc='mp4v', ...)`. mp4v emits a constant-rate header — frame timestamps in the .mp4 are reconstructed by ORB_SLAM3 as `i / fps`, not from real bag timestamps. Meanwhile `imu_data.json["cts"]` is written from *actual* receipt timestamps `1000 * (ts - first_ts)`. ([opencv/opencv#23403](https://github.com/opencv/opencv/issues/23403), [opencv/opencv#25637](https://github.com/opencv/opencv/issues/25637))
**Impact:** any frame-rate jitter on the camera (very common on D435i with USB load) becomes camera↔IMU misalignment of tens to hundreds of ms accumulated over a 60s mapping run. Mono-inertial ORB_SLAM3 is documented to fail with "Frame with a timestamp older than previous frame" or silent IMU bias drift in this regime ([UZ-SLAMLab/ORB_SLAM3#67](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/67), [UZ-SLAMLab/ORB_SLAM3#346](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/346)).
**Note:** the ROS1 stage has the same flaw — but ROS1 was where the original UMI dataset was tuned, so the failure mode may be more visible on a ROS2 setup that has different USB scheduling.
**Fix sketch:** write a per-frame timestamp sidecar (e.g. JSON list of float seconds) alongside `raw_video.mp4`, and have the ORB_SLAM3 wrapper use that instead of frame_index/fps. Or use a constant-rate codec only after explicit resampling to that rate.

### A4. Image–IMU origin mismatch in `imu_data.json["cts"]`
**Where:** `00_process_videos_ros2.py:128-148`.
**What:** `first_ts` is set to the *first gyro sample's* timestamp. `cts` of each accel/gyro sample is `1000 * (ts - first_ts)`. But `raw_video.mp4` frame 0 is whatever timestamp was on the first image, which is generally `≠ first_gyro_ts` (RealSense gyro 200 Hz, accel 62 Hz, image 30 Hz — they start at different sub-second offsets after enabling streams; ([Medium 99p-labs guide](https://medium.com/99p-labs/navigating-depth-perception-f9d4874dd88f))).
**Impact:** ORB_SLAM3 receives mp4 at frame_idx · 1/fps and IMU at `cts/1000`, but the two clocks aren't co-zeroed. Constant offset Δ = `first_image_ts − first_gyro_ts` is folded into IMU pre-integration as a calibration error.
**Note:** identical to ROS1; flagging because ROS2 makes the splitting more visible (separate accel/gyro/imu topics with independent driver activation latencies).
**Fix sketch:** compute `first_ts = min(first_image_ts, first_gyro_ts, first_accel_ts)` and zero everything (including image timestamps written to a sidecar) against that.

### A5. Out-of-memory risk in stage 07 — full image stack in RAM
**Where:** `07_storage_hdf5_ros2.py:166-182` and `:218`.
**What:** `wrist_imgs.append(image_msg_to_array(msg))` then `np.stack(wrist_imgs)`. For a 5-min demo at 30 FPS at 960×540 BGR uint8 that's ~13 GB. Multiplied by `max_workers` threads it OOMs.
**Trigger:** longer demos and/or higher resolution. Currently the orchestrator forces `--max_workers 1` for stage 07 (`run_slam_pipeline_ros2.py:28, :228`), which mitigates the multiplier — but a single long demo still has the per-bag risk.
**Note:** ROS1 stage 07 has the same flaw; not regressed.
**Fix sketch:** stream images directly into an HDF5 chunked dataset, growing it as messages arrive.

---

## B. Medium-severity issues (correctness / UX)

### B1. `aurco/` is eligible to be moved into `mapping/`
**Where:** `00_process_videos_ros2.py:175-221` (`move_largest_video_to_mapping`).
**What:** `demo_dirs = [x for x in demos_dir.iterdir() if x.is_dir() and x.name != "mapping"]` — only `mapping` is excluded. The ArUco-13 base-calibration demo lives in `<demos>/aurco/` and is an unfiltered candidate. If a user happens to record `aurco/` longer than every demo (rare but possible), it will be selected as the mapping demo: the source `<session>/aurco/` ROS2 bag is moved to `<session>/mapping/`, the wrist video is moved to `<demos>/mapping/raw_video.mp4`, and `<demos>/aurco/` is `rmtree`d.
**Impact:** stages 04a, 05, 06, 07 all dereference `<demos>/aurco/` and will fail with `FileNotFoundError` / `assert aurco_dir.is_dir()` (run_slam_pipeline_ros2.py:131-132).
**Fix sketch:** add `and x.name != "aurco"` to the candidate filter (matches ROS1 spelling).

### B2. Empty/no-IMU bag silently produces a placeholder `imu_data.json`
**Where:** `00_process_videos_ros2.py:67-71` and `:128-148`.
**What:** `matched_topics` only fails if NONE of `{image, accel, gyro, imu}` are present. If only the image topic matches, the loop produces 0 accel + 0 gyro samples, and `first_ts = image_timestamps[0]` is used as fallback. `imu_data.json` is then written with empty stream lists, and the pipeline proceeds.
**Impact:** stage 02 (ORB_SLAM3 monocular-inertial via Docker) will then either crash with no-IMU error or silently fall back to monocular-only without telling the user. Hard to diagnose downstream.
**Fix sketch:** explicit warning + non-zero exit if IMU streams ended up empty in mono-inertial mode (which the YAML config in `config/RealSense_D435i.yaml` implies).

### B3. Two reader passes over each bag in stage 00 (and similar in 07)
**Where:** `00_process_videos_ros2.py:64` (`get_available_topics(bag_dir)`) followed by `:73` (`iter_deserialized_messages(bag_dir, ...)`).
**What:** Each call constructs a fresh `AnyReader([bag_dir])` (`ros2_bag_utils.py:42, :52`). For multi-GB bags this doubles I/O and metadata parsing time.
**Impact:** wall-clock cost only, no correctness issue. ([rosbags AnyReader docs](https://ternaris.gitlab.io/rosbags/topics/highlevel.html) confirm streaming inside one `with` block is the supported pattern.)
**Fix sketch:** open `AnyReader` once, snapshot `connections` -> topics, then iterate `messages(connections=...)` without reopening.

### B4. Stage 03 / stage 04b iterate everything in `demos/` minus `aurco|mapping`, including the `aurco`-derived demo dir if mapping move skipped it
**Where:** `run_slam_pipeline_ros2.py:108-119` (stage 03) and `:157-179` (stage 04b).
**What:** stage 03's `--input_dir demo_dir` causes `03_batch_slam.py` to receive *everything* under `demos/` (its filtering logic isn't visible from the orchestrator). Stage 04b explicitly excludes `aurco|mapping`. If 03's filter is weaker than 04b's, SLAM may run on `aurco/`.
**Impact:** wasted compute (running ORB_SLAM3 on the 60-frame ArUco demo), and possibly an `aurco/camera_trajectory.csv` that isn't used. No hard failure — just confusing logs.
**Fix sketch:** verify `03_batch_slam.py`'s skip-list matches `04b`'s. (Out of scope of the ROS2 file itself, but the orchestrator could pass an explicit allow-list.)

### B5. ROS2 bag receipt timestamp ≠ source `header.stamp`
**Where:** `ros2_bag_utils.py:58-62`.
**What:** `iter_deserialized_messages` yields `float(timestamp_ns) * 1e-9`. Per the `rosbags` API and rosbag2 conventions, that timestamp is the *recorder receipt time*, not the publisher's `header.stamp`. ([Robotics SE: rosbag time vs publish time](https://robotics.stackexchange.com/questions/73077/rosbag-and-timestamp-received-vs-timestamp-published), [answers.ros.org #414352](https://answers.ros.org/question/414352))
**Impact:** ROS2 has higher and more variable transport latency than ROS1's localhost rosbag. Two sensors that fire at the same hardware moment but arrive in different orders/jitter at the recorder will look temporally desynchronized, compounding A3/A4 above. Existing ROS1 code uses the same convention, so this is a *bigger* practical problem on ROS2 than ROS1.
**Fix sketch:** prefer `msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9` for sensor topics that have a header; keep receipt time only as fallback.

### B6. Stage 04a's call doesn't pass `--video raw_video.mp4`, relying on the script's default
**Where:** `run_slam_pipeline_ros2.py:133-145` vs `04_detect_aruco.py` defaults.
**What:** This matches the ROS1 path, but is fragile — any future change to `04_detect_aruco.py`'s default video filename will silently break the ROS2 pipeline. Stage 04b *does* pass `--video raw_video.mp4` explicitly, so the asymmetry is suspicious.
**Impact:** brittleness, not a current bug.
**Fix sketch:** pass `--video raw_video.mp4` explicitly in the 04a call too.

---

## C. Low-severity / nits

- **`run_slam_pipeline_ros2.py:51`** `for session in session_dir`: with `nargs=-1` and no sessions provided, the script silently exits with rc=0. Add a "no sessions specified" warning.
- **`run_slam_pipeline_ros2.py:14`** `os.chdir(ROOT_DIR)` at import time. Standard pattern in this repo, but it's a global side effect that surprises any wrapper that imports this module.
- **No subprocess timeout anywhere.** A hung Docker stage 02 hangs the entire pipeline indefinitely (matches ROS1).
- **No top-level `try/except`** around individual stages, so one failed session aborts the whole batch instead of moving on.
- **`ros2_bag_utils.py:1`** imports `Iterable` from `typing` but never uses it; minor cleanup.
- **`00_process_videos_ros2.py:154-163`** prints "Actual video FPS: …" only inside the `if duration > 0` branch — when there's exactly 1 frame the print is suppressed but `actual_fps=30.0` is silently used. Consider a single explicit log line in all paths.
- **`ros2_bag_utils.image_msg_to_bgr` only handles `bgr8/rgb8/bgra8/rgba8/mono8`.** RealSense rarely emits anything else by default, but `yuv422`/`nv12` and `image_transport/Compressed*` will raise. Documented as "Unsupported image encoding" — acceptable, but worth listing in `SLAM_readme_mono.txt`.
- **`07_storage_hdf5_ros2.py:212-221`** When sensor data is missing, `interp_sensor` is built only over `wrist_times` length; later `cropped_sensor` is `np.resize`d to `crop_len`. `np.resize` *cycles* values, not pads with zero — for a sensor stream of ≥ crop_len that's a no-op, but for shorter or empty inputs it produces nonsense gripper-width values. Inherited bug pattern from ROS1.
- **`run_slam_pipeline_ros2.py:34, :228`** `--imu_topic` is accepted at the orchestrator level but only forwarded to stage 00. Stage 07 has no IMU consumption, so this is by design — but consider documenting why the parameter set differs between stages so users don't expect symmetry.
- **`shutil.move(...)` of the source ROS2 bag dir into `<session>/mapping/`** is destructive. If the user re-runs stage 00 after deleting `<demos>/mapping/raw_video.mp4` to redo it, the original bag is no longer where they expect; it's now under `<session>/mapping/`. Worth a one-line message at move time.

---

## D. What I did NOT find that you might have suspected

- The `aurco` (sic) spelling is consistently used everywhere, so no spelling-related dispatch bug.
- Click `--max_workers` for stages 00 and 07 are wired through the orchestrator correctly.
- The `tx_slam_tag.json` plumbing (stage 05 → stage 06 → stage 07) matches the ROS1 path verbatim.
- `discover_ros2_bag_dirs` correctly distinguishes ROS2 bag directories by `metadata.yaml` and excludes `<session>/demos/` (which has no metadata.yaml).
- `image_msg_to_bgr` correctly handles `step != width*channels` row padding.

---

## Suggested triage order if/when you want fixes
1. **A1** (IMU duplication) — small change, prevents hard-to-debug downstream SLAM failure.
2. **A2** (zombie dir) — small change, prevents silent dataset loss.
3. **B1** (`aurco/` mapping promotion) — one-line filter fix.
4. **A3 + A4 + B5** (timestamp model) — coordinated change, bigger but most impactful for SLAM quality.
5. **A5** (stage 07 streaming) — only matters at scale.
6. **Everything else** as cleanup.

---

## Sources consulted
- [Realsense ROS2 wrapper: unite_imu_method docs](https://dev.realsenseai.com/docs/ros2-wrapper/)
- [realsense-ros#598: D435i unified IMU vs. split topics](https://github.com/realsenseai/realsense-ros/issues/598)
- [Medium 99p-labs: D435i IMU rates and methods](https://medium.com/99p-labs/navigating-depth-perception-f9d4874dd88f)
- [rosbags AnyReader high-level API](https://ternaris.gitlab.io/rosbags/topics/highlevel.html)
- [rosbag receipt-time vs header.stamp](https://robotics.stackexchange.com/questions/73077/rosbag-and-timestamp-received-vs-timestamp-published)
- [answers.ros.org: ros2bag time field semantics](https://answers.ros.org/question/414352)
- [opencv/opencv#23403: VideoWriter constant FPS limitation](https://github.com/opencv/opencv/issues/23403)
- [opencv/opencv#25637: VideoWriter ignores frame timestamps](https://github.com/opencv/opencv/issues/25637)
- [UZ-SLAMLab/ORB_SLAM3#67: timestamp-older-than-previous error](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/67)
- [UZ-SLAMLab/ORB_SLAM3#346: Mono-Inertial issues with D455](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/346)
