## Context

DexSlide's world-frame palm pose is sourced from `umi_mono`'s ORB-SLAM3 pipeline (cheng-chi/ORB_SLAM3 Docker), which today runs strictly batch: video + IMU files in, trajectory CSV out. The fork already ships `--load_map`/`--save_map` paths and preserves upstream `ActivateLocalizationMode()` + `TrackMonocular()`. The algorithmic capability for offline-map + live-tracking already exists; the IO boundary is the limiter. Full investigation in `umi_mono/docs/online_tracking_research.md`.

Stakeholders:
- DexSlide live demo / teleoperation: needs sub-33 ms pose for closed-loop control.
- DexSlide policy evaluation: needs deterministic pose stream synchronized with joint angles.
- Calibration / data collection: continues to use the offline batch pipeline (out of scope here).

Constraints:
- Native Ubuntu 22.04 host, no Docker for the online tracker.
- Existing offline pipeline must remain untouched (additive change).
- Camera fixed as RealSense D435i, 960×540 BGR @30 Hz + IMU @200 Hz.
- Map artifact stays in `map_atlas.osa` format produced by the existing offline stages.

## Goals / Non-Goals

**Goals:**
- Publish 6-DoF camera pose at 30 Hz, p99 per-frame latency < 33 ms, p99 publish-to-subscriber latency < 50 ms.
- Load a pre-built `.osa` atlas and run ORB-SLAM3 in localization-only mode (zero map mutation).
- Provide a Python consumer (`SlamPoseSubscriber`) exposing thread-safe time-aligned pose lookup to the existing Vedo viewer / kinematics stack.
- Sustained 1-hour run without memory growth > 5% RSS or FPS degradation > 5%.
- Recovery from tracking loss within 5 s in a known environment without process restart.
- Isolation: existing Docker batch pipeline (`02_create_map.py`, `03_batch_slam.py`) is not modified.

**Non-Goals:**
- Online mapping (map updates from the live stream) — explicitly forbidden by localization-only mode.
- GPU acceleration — ORB-SLAM3 is CPU-bound; not needed for 30 Hz.
- Cross-SLAM map portability (e.g., stella_vslam msgpack, RTAB-Map .db).
- Migrating the offline mapping stage off Docker — future change.
- Multi-camera tracking — single D435i only in this iteration.
- Web/IK-side consumers — only `SlamPoseSubscriber` API is contracted here; downstream usage is owned by future DexSlide changes.

## Decisions

### D1: Extend the fork with a new binary, not a separate ROS2 package
**Choice**: Add `Examples/Monocular-Inertial/realsense_online.cc` inside the cheng-chi/ORB_SLAM3 fork. Build via the fork's CMakeLists.

**Why**: Reuses the fork's already-resolved DBoW2/Pangolin/Boost.Serialization/Sophus linker setup. The atlas-load constructor parameter (`System.LoadAtlasFromFile` in settings YAML) and `LocalizeMonocular` extension are fork-specific; a standalone package would re-wire those.

**Alternatives considered**:
- *Separate ROS2 package consuming ORB-SLAM3 as an installed library*: cleaner long-term, but the fork's CMakeLists doesn't currently `install()` headers/.so files; switching to an installed-library model is its own work-item. Defer to a follow-on change.
- *v4l2loopback + IMU JSON tail under existing Docker*: rejected — `cv::VideoCapture(... CAP_FFMPEG)` and one-shot `LoadTelemetry` don't survive streaming; would require Docker USB passthrough.

### D2: Localization mode is enforced at the binary, not assumed from atlas-load
**Choice**: Call `SLAM.ActivateLocalizationMode()` immediately after `System` construction in `realsense_online.cc`.

**Why**: cheng-chi fork's `System.cc` constructor contains the comment `// when loading a file from disk, we want to localize` but does **not** actually flip `mbActivateLocalizationMode`. Relying on the comment leaks tracking results into the loaded atlas. Calling `ActivateLocalizationMode()` post-construction is the documented, supported path.

**Alternatives considered**:
- *Patch the fork's `System.cc` to auto-activate localization on atlas load*: rejected — modifies behavior of `gopro_slam.cc` (batch) which relies on the current semantics; would require coordinated changes.

### D3: RealSense `GLOBAL_TIME` timestamp domain, single monotonic clock
**Choice**: Configure both color and IMU streams with `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`; ensure firmware supports it (D435i does from FW 5.12+).

**Why**: Image and IMU must share a clock for IMU preintegration. The default `RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME` puts USB-stamp into wallclock and is jittery. `HARDWARE_CLOCK` is even tighter but requires firmware + driver flag.

**Alternatives**: software-side cross-correlation alignment (rejected — fragile, drifts over hours).

### D4: ROS2 PoseStamped + tf2 as the publish channel
**Choice**: Publish `geometry_msgs/PoseStamped` on `/dexslide/slam/pose` and tf2 `map → camera_color_optical_frame`.

**Why**: `umi_mono` already has a ROS2 path (`run_slam_pipeline_ros2.py`), DexSlide can consume via `rclpy`. Pose latency through ROS2 intra-process or DDS over loopback is ~2–5 ms — fits the 50 ms publish-to-subscriber budget.

**Alternatives**:
- *ZMQ PUB/SUB on `tcp://127.0.0.1:5555`*: lower setup cost, no DDS dep. Keep as a `--publisher` CLI option for cases where ROS2 is unavailable.
- *Shared-memory ring buffer*: sub-ms latency but adds a producer/consumer contract to maintain. Defer.

### D5: IMU buffering — lock-free ring buffer, drain-by-timestamp
**Choice**: Pre-allocated lock-free ring buffer (`boost::lockfree::spsc_queue` or hand-rolled SPMC) of size ≥ 200 (1 s of IMU at 200 Hz). Per-frame: `drain_until(image_timestamp)` returns the slice `[last_frame_t, image_t]` for `TrackMonocular(im, t, vImuMeas)`.

**Why**: Heap allocations in a 30 Hz loop cause GC-style jitter that violates the latency budget. SPSC (single-producer-single-consumer) matches the librealsense callback → main-loop pattern.

**Alternatives**: mutex-protected `std::deque` (rejected — measured > 1 ms median lock contention on contention; tail latency spikes).

### D6: `max_lost_frames` re-interpreted as soft reset
**Choice**: In `realsense_online.cc`, set `max_lost_frames = 900` (~30 s at 30 Hz) and convert the fork's behavior of process-exit-on-exceed into "log + continue feeding frames so the internal `RELOCALIZATION` state machine can recover." The process never exits on tracking loss.

**Why**: Batch mode terminates on long loss because there's nothing to recover toward. Online mode must keep feeding frames so the place-recognition module can re-anchor on the loaded atlas.

**Alternatives considered**: hard-restart the whole `System` after N seconds lost (rejected — atlas-load is ~1–3 s; would block live consumers).

### D7: Headless build, viewer off by default
**Choice**: Construct `System(..., bUseViewer=false, ...)`. Build with `-DUSE_PANGOLIN=OFF` is a future option but not required for this change.

**Why**: 30 Hz loop tolerates Pangolin linked-in-but-not-shown. Hard requirement is that no DISPLAY-dependent code runs when viewer is off.

### D8: Python consumer interface
**Choice**: `dexslide.world_pose.SlamPoseSubscriber` exposes:
```python
def get_T_world_camera(self, t: Optional[float] = None) -> Optional[np.ndarray]:
    """4x4 SE(3) at time t (s, monotonic) via linear interp on translation +
    SLERP on rotation. Returns None if no pose within ±100 ms of t."""

def latest(self) -> Optional[tuple[float, np.ndarray]]: ...

def is_tracking(self) -> bool: ...  # True iff a pose was received within last 200 ms
```

**Why**: Vedo viewer's frame rate is decoupled from the SLAM frame rate; consumers want time-aligned lookups, not raw stream callbacks. Mirrors the API style already present in `dexslide/serial_reader.py`.

## Risks / Trade-offs

- **[Risk] cheng-chi fork is a snapshot (0 open issues, no commits 2023+)** → Mitigation: pin a Git SHA in build script; document the SHA in `umi_mono/CLAUDE.md`; isolate `realsense_online.cc` from fork internals so future migration to upstream-or-different-fork stays bounded to one file.
- **[Risk] Image/IMU timestamp skew > 10 ms breaks IMU preintegration** → Mitigation: D3 (GLOBAL_TIME domain) + startup sanity check that logs measured skew over first 100 frames and aborts if > 5 ms median.
- **[Risk] Pangolin viewer crash in headless** → Mitigation: D7 (`bUseViewer=false` enforced); also test with `unset DISPLAY` in CI smoke job.
- **[Risk] Tracking loss in dynamic lighting (windows, lamps)** → Mitigation: lock exposure to a calibrated range (`RS2_OPTION_ENABLE_AUTO_EXPOSURE=0`, fixed exposure value per environment); document in `CLAUDE.md`.
- **[Risk] Vocabulary MD5 mismatch on cross-machine map share** → Mitigation: bake the vocab file into the binary install path; check MD5 at startup; fail loud, not silent.
- **[Trade-off] GPL-3 boundary**: cheng-chi/ORB_SLAM3 is GPL-3. Keeping the SLAM as a separate process and consuming pose over ROS2/ZMQ keeps the GPL boundary clean for any DexSlide consumer code not yet committed to GPL.
- **[Risk] Map drift over weeks if the rig is recalibrated**: not the SLAM's problem, but worth flagging — when D435i extrinsics shift, the saved map's metric scale and `T_imu_camera` go stale. Mitigation: re-run mapping after any physical recalibration; expose the map fingerprint (MD5 + timestamp) in pose topic header.metadata.
- **[Trade-off] No GPU**: locks us out of dense reconstruction or learning-based VO (DPVO, MASt3R-SLAM). Acceptable — current task is pose tracking, not dense scene.
- **[Trade-off] D5 lock-free dep**: `boost::lockfree` is header-only in Boost ≥ 1.71 (Ubuntu 22.04 has 1.74). Acceptable.

## Migration Plan

This is purely additive — no existing user-facing surface changes.

1. Build the new binary on a dev machine without touching the Docker image.
2. Validate on a known recording: run `realsense_online` against a playback of `raw_video.mp4` + `imu_data.json` (via a `--playback-mode` CLI flag) and compare ATE vs the Docker batch output; require ATE < 2 cm.
3. Validate on live camera against a known ArUco-13 ground truth (the existing `aurco` calibration path supplies this).
4. Roll out to the demo workstation; offline pipeline keeps running for non-live data collection.

**Rollback**: delete the binary, ignore the topic. The Docker batch path is unaffected, so any consumer that hasn't subscribed to the new topic continues to work.

## Open Questions

- Should the pose topic emit identity / `NaN` on tracking loss, or skip messages entirely? **Tentative**: skip; let `SlamPoseSubscriber.is_tracking()` flag drive consumer behavior. Confirm with one downstream consumer (likely the Vedo viewer) before locking the contract.
- Do we publish the body-frame (IMU) pose as well as the camera pose? `IMU.T_b_c1` is in `RealSense_D435i.yaml`; cheap to add but requires deciding the `map → imu_body_frame` semantics. **Tentative**: publish only camera pose now; add IMU body frame only if a consumer needs it.
- Should the build run inside an OpenSpec-managed workspace or stay inside `umi_mono/`? **Tentative**: keep build inside `umi_mono/external/ORB_SLAM3_fork/` (git submodule); the OpenSpec change lives at DexSlide root and orchestrates.
- Settings YAML: do we ship a `RealSense_D435i_online.yaml` separate from the existing `RealSense_D435i.yaml`, or add a `System.LoadAtlasFromFile` override path? **Tentative**: ship a separate `_online.yaml` that diffs only on the atlas-load and viewer fields, to avoid breaking the offline pipeline if someone shares config.
