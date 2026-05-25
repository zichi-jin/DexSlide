## ADDED Requirements

### Requirement: Online tracker SHALL load a pre-built atlas without mutating it
The `realsense_online` binary SHALL load an existing `map_atlas.osa` via the `System.LoadAtlasFromFile` settings key and SHALL enter ORB-SLAM3's localization-only mode before processing the first frame. The on-disk atlas file SHALL be byte-identical before and after a tracking session.

#### Scenario: Atlas is not mutated by a tracking session
- **WHEN** the operator runs `realsense_online --settings <yaml-with-LoadAtlasFromFile> --vocab ORBvoc.txt` and lets it track for 60 s
- **THEN** the SHA-256 of `map_atlas.osa` after the session equals the SHA-256 before the session
- **AND** the binary writes to stdout the message `Localization mode active, atlas read-only`

#### Scenario: Missing atlas fails loudly
- **WHEN** the settings YAML's `System.LoadAtlasFromFile` points to a non-existent file
- **THEN** the binary exits within 2 s with a non-zero exit code and a `stderr` line containing `Atlas file not found: <path>`

#### Scenario: Vocabulary MD5 mismatch fails loudly
- **WHEN** the vocabulary file MD5 does not match the MD5 stored in the atlas
- **THEN** the binary exits within 5 s with a non-zero exit code and a `stderr` line beginning with `Vocabulary mismatch:`

### Requirement: Tracker SHALL publish pose at ≥ 30 Hz with bounded per-frame latency
The `realsense_online` binary SHALL publish 6-DoF camera pose messages on the configured pose topic at a sustained rate ≥ 30 Hz when tracking is OK, with p99 wall-clock latency from image capture-timestamp to message-publish-timestamp < 33 ms on the reference workstation (i7-12700 or better, RealSense D435i over USB 3.x).

#### Scenario: Sustained 30 Hz publish rate
- **WHEN** the tracker runs for 30 minutes against a stationary D435i in a previously-mapped environment
- **THEN** the count of `PoseStamped` messages published is ≥ 30 × 60 × 30 × 0.95 (5% loss budget)
- **AND** the p99 inter-message interval is ≤ 50 ms

#### Scenario: Bounded per-frame latency
- **WHEN** the tracker is processing live frames and tracking is OK
- **THEN** the p99 difference `header.stamp.publish_wallclock - image.capture.timestamp` is < 33 ms over any contiguous 10-minute window

#### Scenario: No publish during tracking lost
- **WHEN** the internal tracker state is `LOST` or `NOT_INITIALIZED`
- **THEN** no `PoseStamped` message is published, but a separate diagnostic counter on `/dexslide/slam/diagnostics` increments

### Requirement: Tracker SHALL recover from tracking loss without process restart
When tracking is lost, the tracker SHALL continue ingesting frames, attempt relocalization against the loaded atlas, and resume publishing pose within 5 s once relocalization succeeds. The process SHALL NOT exit on tracking loss regardless of duration.

#### Scenario: Recover from camera occlusion in known area
- **WHEN** the camera is fully occluded for 3 s in a previously-mapped area, then uncovered
- **THEN** the tracker publishes a valid `PoseStamped` within 5 s of being uncovered
- **AND** the process is still alive (PID unchanged)

#### Scenario: Indefinite loss does not exit
- **WHEN** the camera is occluded for 30 minutes
- **THEN** the process is still alive at the end of the period
- **AND** zero `PoseStamped` messages were published during the occlusion
- **AND** `/dexslide/slam/diagnostics` reports state `LOST` for the full duration

### Requirement: IMU/image timestamp domains SHALL be reconciled and validated at startup
The tracker SHALL configure both color and IMU streams with `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`, measure the median timestamp skew between paired image and IMU samples over the first 100 frames, and abort if the median skew exceeds 5 ms.

#### Scenario: Skew within budget
- **WHEN** the D435i firmware supports GLOBAL_TIME and the bus is healthy
- **THEN** measured median skew over the first 100 frames is < 5 ms
- **AND** the tracker proceeds to normal operation

#### Scenario: Skew out of budget aborts startup
- **WHEN** measured median skew over the first 100 frames is ≥ 5 ms
- **THEN** the tracker exits with a non-zero exit code within the 4th second of operation
- **AND** logs `Timestamp skew out of budget: <measured> ms (max 5 ms)`

### Requirement: Tracker SHALL run headless without Pangolin viewer
The tracker SHALL be invocable with no `$DISPLAY` environment variable set, and SHALL construct the ORB-SLAM3 `System` with `bUseViewer=false`. No Pangolin window SHALL be created in this mode.

#### Scenario: Headless run
- **WHEN** the tracker is launched with `unset DISPLAY` in the environment
- **THEN** the process runs normally and publishes pose
- **AND** no X11 errors are logged on stderr
- **AND** no Pangolin window is created (verified via `wmctrl -l` showing no `ORB-SLAM3` entry)

### Requirement: Tracker CLI SHALL accept the same atlas, vocabulary, and settings paths as the offline binary
The `realsense_online` binary SHALL accept `--vocabulary`, `--setting`, and (optionally) `--load_map` CLI flags compatible with the existing `gopro_slam` invocation so that the offline-produced `map_atlas.osa` is consumed unchanged.

#### Scenario: Offline-produced atlas works online
- **WHEN** an `map_atlas.osa` produced by `02_create_map.py` (via Docker `gopro_slam`) is fed to `realsense_online` on the host
- **THEN** the tracker initializes successfully and publishes pose against that atlas
- **AND** the published trajectory's ATE vs the same recording's Docker-batch trajectory is < 2 cm
