## ADDED Requirements

### Requirement: Python consumer SHALL expose thread-safe time-aligned pose lookup
The `dexslide.world_pose.SlamPoseSubscriber` class SHALL subscribe to the SLAM pose topic, maintain a bounded ring buffer of recent poses, and expose `get_T_world_camera(t)` returning the interpolated SE(3) transform at timestamp `t` (monotonic seconds). The method SHALL be safe to call from any thread.

#### Scenario: Time-aligned lookup returns interpolated pose
- **WHEN** the buffer contains poses at timestamps `t1 < t < t2` with `t2 - t1 < 100 ms`
- **THEN** `get_T_world_camera(t)` returns a 4x4 numpy array
- **AND** the translation component equals the linear interpolation of `t1.translation` and `t2.translation` at `t`
- **AND** the rotation component equals the SLERP of `t1.rotation` and `t2.rotation` at `t`

#### Scenario: Out-of-range lookup returns None
- **WHEN** `t` is more than 100 ms outside any pose in the buffer
- **THEN** `get_T_world_camera(t)` returns `None`

#### Scenario: Concurrent access does not corrupt
- **WHEN** 10 threads call `get_T_world_camera(t)` concurrently while the rclpy executor pushes new poses to the buffer at 30 Hz for 60 s
- **THEN** no exception is raised
- **AND** every returned value is either a valid 4x4 SE(3) matrix or `None`

### Requirement: Subscriber SHALL expose a tracking-health flag
The subscriber SHALL expose `is_tracking() -> bool` returning `True` iff at least one pose message was received in the past 200 ms.

#### Scenario: Tracking flag goes false on stream stop
- **WHEN** the SLAM publisher stops emitting messages
- **AND** 200 ms have passed since the last received message
- **THEN** `is_tracking()` returns `False`

#### Scenario: Tracking flag goes true on first message
- **WHEN** the subscriber has just constructed and one pose message has arrived
- **THEN** `is_tracking()` returns `True` within the next 10 ms

### Requirement: Subscriber SHALL not block when no rclpy executor is running
Constructing `SlamPoseSubscriber` SHALL not require the caller to have started an rclpy executor. The class SHALL provide a `spin_in_thread()` method that starts a background daemon thread running its own SingleThreadedExecutor.

#### Scenario: Standalone use in a Python script
- **WHEN** a script constructs `SlamPoseSubscriber(node_name='dexslide_consumer')` and calls `subscriber.spin_in_thread()`
- **THEN** within 100 ms `is_tracking()` returns `True` provided the publisher is running
- **AND** the main thread is not blocked

### Requirement: Subscriber SHALL expose a stale-detection threshold
The subscriber SHALL accept a constructor parameter `stale_after_seconds: float = 0.2` that controls the `is_tracking()` threshold and the freshness check inside `get_T_world_camera(t)` with `t=None`.

#### Scenario: Default stale threshold
- **WHEN** the subscriber is constructed without arguments
- **THEN** `is_tracking()` returns `False` exactly when the last message age exceeds 0.2 s

#### Scenario: Custom stale threshold
- **WHEN** the subscriber is constructed with `stale_after_seconds=0.5`
- **AND** the last message age is 0.3 s
- **THEN** `is_tracking()` returns `True`
