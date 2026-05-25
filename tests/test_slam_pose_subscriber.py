import math

import numpy as np

from dexslide.world_pose import slam_pose_subscriber as sps


def _assert_quaternion_close(q_actual: np.ndarray, q_expected: np.ndarray, atol: float = 1e-7) -> None:
    actual = np.asarray(q_actual, dtype=np.float64)
    expected = np.asarray(q_expected, dtype=np.float64)
    if np.dot(actual, expected) < 0.0:
        actual = -actual
    np.testing.assert_allclose(actual, expected, atol=atol)


def test_quaternion_to_rotmat_identity():
    R = sps._quaternion_to_rotmat(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


def test_quaternion_to_rotmat_90deg_z():
    q = np.array([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)], dtype=np.float64)
    R = sps._quaternion_to_rotmat(q)
    rotated = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-7)


def test_slerp_endpoints():
    q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q2 = np.array([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)], dtype=np.float64)

    _assert_quaternion_close(sps._slerp(q1, q2, 0.0), q1)
    _assert_quaternion_close(sps._slerp(q1, q2, 1.0), q2)

    q_mid = sps._slerp(q1, q2, 0.5)
    R_mid = sps._quaternion_to_rotmat(q_mid)
    rotated = R_mid @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(
        rotated,
        np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0], dtype=np.float64),
        atol=1e-7,
    )


def test_slerp_antipodal_handling():
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q2 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_mid = sps._slerp(q1, q2, 0.5)
    assert np.all(np.isfinite(q_mid))
    _assert_quaternion_close(q_mid, q1)


def test_buffer_appends_and_latest(subscriber_factory, fake_pose_msg):
    sub = subscriber_factory()
    sub._on_pose(fake_pose_msg(10.0))

    latest = sub.latest()
    assert latest is not None
    stamp, T = latest
    assert stamp == 10.0
    np.testing.assert_allclose(T, np.eye(4), atol=1e-7)


def test_buffer_bounded(subscriber_factory, fake_pose_msg):
    sub = subscriber_factory(buffer_size=5)
    for i in range(55):
        sub._on_pose(fake_pose_msg(float(i), tx=float(i)))

    with sub._lock:
        buffered = list(sub._buf)

    assert len(buffered) == 5
    assert buffered[0][0] == 50.0
    assert buffered[-1][0] == 54.0
    assert buffered[0][1][0, 3] == 50.0


def test_get_T_world_camera_interpolation(subscriber_factory, fake_pose_msg):
    sub = subscriber_factory()
    sub._on_pose(fake_pose_msg(10.0))
    sub._on_pose(
        fake_pose_msg(
            10.05,
            tx=1.0,
            qz=math.sin(math.pi / 4.0),
            qw=math.cos(math.pi / 4.0),
        )
    )

    T = sub.get_T_world_camera(10.025)
    assert T is not None
    np.testing.assert_allclose(T[:3, 3], np.array([0.5, 0.0, 0.0]), atol=1e-7)
    rotated = T[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(
        rotated,
        np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0], dtype=np.float64),
        atol=1e-7,
    )


def test_get_T_world_camera_out_of_range(subscriber_factory, fake_pose_msg):
    sub = subscriber_factory()
    sub._on_pose(fake_pose_msg(10.0))
    sub._on_pose(fake_pose_msg(10.05, tx=1.0))

    assert sub.get_T_world_camera(12.0) is None
    assert sub.get_T_world_camera(9.0) is None
    T = sub.get_T_world_camera(10.0)
    assert T is not None
    np.testing.assert_allclose(T, np.eye(4), atol=1e-7)


def test_is_tracking_default_threshold(subscriber_factory, fake_pose_msg, monkeypatch):
    sub = subscriber_factory()
    now = [100.0]
    monkeypatch.setattr(sps.time, "monotonic", lambda: now[0])

    sub._on_pose(fake_pose_msg(10.0))
    assert sub.is_tracking() is True

    now[0] = 100.25
    assert sub.is_tracking() is False


def test_stale_after_seconds_custom(subscriber_factory, fake_pose_msg, monkeypatch):
    sub = subscriber_factory(stale_after_seconds=0.5)
    now = [200.0]
    monkeypatch.setattr(sps.time, "monotonic", lambda: now[0])

    sub._on_pose(fake_pose_msg(20.0))
    now[0] = 200.3
    assert sub.is_tracking() is True

    now[0] = 200.6
    assert sub.is_tracking() is False
