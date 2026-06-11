"""TASK-026..TASK-030 world-pose subscriber utilities for DexSlide.

Provides a ROS2 jazzy PoseStamped subscriber with time-aligned SE(3) lookup.
Manual SLERP is implemented with numpy because scipy is not installed on
system /usr/bin/python3.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

LOGGER = logging.getLogger(__name__)


def _ensure_ros_log_dir() -> None:
    if "ROS_LOG_DIR" in os.environ:
        return
    ros_log_dir = "/tmp/roslog"
    os.makedirs(ros_log_dir, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = ros_log_dir


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    return q / norm


def _quaternion_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = _normalize_quaternion(q)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return _normalize_quaternion(np.array([qx, qy, qz, qw], dtype=np.float64))


def _slerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    qa = _normalize_quaternion(q1)
    qb = _normalize_quaternion(q2)
    cos_omega = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if cos_omega < 0.0:
        qb = -qb
        cos_omega = -cos_omega
    if cos_omega > 0.9995:
        return _normalize_quaternion((1.0 - alpha) * qa + alpha * qb)

    omega = float(np.arccos(cos_omega))
    sin_omega = float(np.sin(omega))
    scale_1 = np.sin((1.0 - alpha) * omega) / sin_omega
    scale_2 = np.sin(alpha * omega) / sin_omega
    return _normalize_quaternion(scale_1 * qa + scale_2 * qb)


def _se3_from_pose(pose: PoseStamped().pose.__class__) -> np.ndarray:
    quat = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quaternion_to_rotmat(quat)
    T[:3, 3] = np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    return T


class SlamPoseSubscriber:
    def __init__(
        self,
        node_name: str = "dexslide_consumer",
        topic: str = "/dexslide/slam/pose",
        stale_after_seconds: float = 0.2,
        buffer_size: int = 300,
    ):
        if not rclpy.ok():
            _ensure_ros_log_dir()
            rclpy.init()

        self._buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stale_after_seconds = stale_after_seconds
        self._node: Node = Node(node_name)
        self._subscription = self._node.create_subscription(
            PoseStamped, topic, self._on_pose, qos_profile_sensor_data
        )
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._started = False
        self._last_recv_monotonic: Optional[float] = None

    def _on_pose(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            T = _se3_from_pose(msg.pose)
        except ValueError as exc:
            LOGGER.warning("Dropping invalid pose message: %s", exc)
            return

        recv_mono = time.monotonic()
        with self._lock:
            self._buf.append((t, T))
            self._last_recv_monotonic = recv_mono

    def latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            if not self._buf:
                return None
            t, T = self._buf[-1]
            return t, T.copy()

    def is_tracking(self) -> bool:
        with self._lock:
            last_recv_monotonic = self._last_recv_monotonic
        if last_recv_monotonic is None:
            return False
        return time.monotonic() - last_recv_monotonic <= self._stale_after_seconds

    def get_T_world_camera(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        with self._lock:
            entries: List[Tuple[float, np.ndarray]] = [
                (stamp, T.copy()) for stamp, T in self._buf
            ]
            last_recv_monotonic = self._last_recv_monotonic

        if t is None:
            if not entries or last_recv_monotonic is None:
                return None
            if time.monotonic() - last_recv_monotonic > self._stale_after_seconds:
                return None
            return entries[-1][1]

        if not entries:
            return None

        times = [stamp for stamp, _ in entries]
        if t < times[0] - 0.1 or t > times[-1] + 0.1:
            return None

        import bisect

        idx = bisect.bisect_left(times, t)
        if idx < len(entries) and abs(times[idx] - t) < 1e-9:
            return entries[idx][1]
        if idx == 0 or idx >= len(entries):
            return None

        t1, T1 = entries[idx - 1]
        t2, T2 = entries[idx]
        if t < t1 or t > t2 or (t2 - t1) >= 0.1:
            return None

        alpha = (t - t1) / (t2 - t1)
        translation = (1.0 - alpha) * T1[:3, 3] + alpha * T2[:3, 3]
        q1 = _rotmat_to_quaternion(T1[:3, :3])
        q2 = _rotmat_to_quaternion(T2[:3, :3])
        quat = _slerp(q1, q2, alpha)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quaternion_to_rotmat(quat)
        T[:3, 3] = translation
        return T

    def spin_in_thread(self) -> None:
        if self._started:
            return

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="slam-pose-subscriber", daemon=True
        )
        self._spin_thread.start()
        self._started = True

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        if self._executor is not None:
            self._executor.remove_node(self._node)
        self._node.destroy_node()
        self._executor = None
        self._spin_thread = None
        self._started = False
        if rclpy.ok():
            rclpy.shutdown()
