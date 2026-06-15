#!/usr/bin/env python3
import json
import os
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from matplotlib.animation import FuncAnimation
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def pose_to_matrix(msg: PoseStamped) -> np.ndarray:
    q = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("PoseStamped contains a zero quaternion")
    q /= norm
    x, y, z, w = q
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ]
    return mat


class PoseSeries:
    def __init__(self, label: str, color: str, history_size: int):
        self.label = label
        self.color = color
        self.latest: Optional[np.ndarray] = None
        self.history: Deque[np.ndarray] = deque(maxlen=history_size)
        self.stamp: Optional[float] = None


class WorldPosePlotter(Node):
    def __init__(self):
        super().__init__("dexslide_world_pose_plotter")
        self.world_topic = self.declare_parameter(
            "world_topic", "/dexslide/aruco/world_pose"
        ).value
        self.slam_topic = self.declare_parameter(
            "slam_topic", "/dexslide/slam/pose"
        ).value
        self.axis_length = float(self.declare_parameter("axis_length", 0.08).value)
        self.history_size = int(self.declare_parameter("history_size", 300).value)
        self.update_hz = float(self.declare_parameter("update_hz", 15.0).value)
        self.range_padding = float(self.declare_parameter("range_padding", 0.15).value)
        self.tx_slam_tag = self.declare_parameter("tx_slam_tag", "").value
        self.t_world_slam = self._load_world_slam(self.tx_slam_tag)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.series: Dict[str, PoseSeries] = {
            "world": PoseSeries("/dexslide/aruco/world_pose", "tab:orange", self.history_size),
            "slam": PoseSeries("/dexslide/slam/pose", "tab:blue", self.history_size),
        }
        self.lock = threading.Lock()

        self.create_subscription(PoseStamped, self.world_topic, self.on_world_pose, qos)
        self.create_subscription(PoseStamped, self.slam_topic, self.on_slam_pose, qos)

    def _load_world_slam(self, tx_slam_tag: str) -> Optional[np.ndarray]:
        if not tx_slam_tag:
            return None
        path = Path(tx_slam_tag).expanduser().absolute()
        if not path.is_file():
            raise FileNotFoundError(f"tx_slam_tag file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        if "tx_slam_tag" not in data:
            raise KeyError(f"'tx_slam_tag' missing in {path}")
        t_slam_tag = np.asarray(data["tx_slam_tag"], dtype=np.float64)
        if t_slam_tag.shape != (4, 4):
            raise ValueError(f"Expected 4x4 tx_slam_tag in {path}, got {t_slam_tag.shape}")
        return np.linalg.inv(t_slam_tag)

    def _store_pose(self, key: str, msg: PoseStamped) -> None:
        try:
            mat = pose_to_matrix(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        if key == "slam" and self.t_world_slam is not None:
            mat = self.t_world_slam @ mat
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self.lock:
            series = self.series[key]
            series.latest = mat
            series.history.append(mat[:3, 3].copy())
            series.stamp = stamp

    def on_world_pose(self, msg: PoseStamped) -> None:
        self._store_pose("world", msg)

    def on_slam_pose(self, msg: PoseStamped) -> None:
        self._store_pose("slam", msg)


def draw_frame(ax, mat: np.ndarray, axis_length: float, prefix: str) -> None:
    origin = mat[:3, 3]
    rot = mat[:3, :3]
    colors = ["r", "g", "b"]
    labels = [f"{prefix}-x", f"{prefix}-y", f"{prefix}-z"]
    for idx in range(3):
        axis_vec = rot[:, idx] * axis_length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_vec[0],
            axis_vec[1],
            axis_vec[2],
            color=colors[idx],
            linewidth=2.0,
            arrow_length_ratio=0.18,
        )
        ax.text(
            origin[0] + axis_vec[0],
            origin[1] + axis_vec[1],
            origin[2] + axis_vec[2],
            labels[idx],
            color=colors[idx],
            fontsize=8,
        )


def set_equal_limits(ax, points: np.ndarray, padding: float) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def main(args=None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    rclpy.init(args=args)
    node = WorldPosePlotter()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(_frame_idx: int) -> None:
        ax.cla()
        ax.set_title("DexSlide Pose 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        with node.lock:
            snapshot = {
                key: {
                    "latest": None if value.latest is None else value.latest.copy(),
                    "history": np.array(value.history, dtype=np.float64)
                    if value.history
                    else np.empty((0, 3), dtype=np.float64),
                    "label": value.label,
                    "color": value.color,
                    "stamp": value.stamp,
                }
                for key, value in node.series.items()
            }

        plotted_points = [np.zeros(3, dtype=np.float64)]
        for key in ("world", "slam"):
            item = snapshot[key]
            history = item["history"]
            latest = item["latest"]
            if history.size:
                ax.plot(
                    history[:, 0],
                    history[:, 1],
                    history[:, 2],
                    color=item["color"],
                    linewidth=1.5,
                    alpha=0.75,
                    label=f"{item['label']} traj",
                )
                plotted_points.extend(history)
            if latest is not None:
                draw_frame(ax, latest, node.axis_length, key)
                pos = latest[:3, 3]
                ax.scatter(
                    [pos[0]],
                    [pos[1]],
                    [pos[2]],
                    color=item["color"],
                    s=48,
                    label=item["label"],
                )
                if item["stamp"] is not None:
                    ax.text(
                        pos[0],
                        pos[1],
                        pos[2],
                        f"{key}@{item['stamp']:.2f}",
                        fontsize=8,
                    )
                plotted_points.append(pos)

        ax.scatter([0.0], [0.0], [0.0], color="k", s=24, label="world origin")
        all_points = np.vstack(plotted_points)
        set_equal_limits(ax, all_points, node.range_padding)
        ax.legend(loc="upper left")
        ax.grid(True)

    anim = FuncAnimation(
        fig,
        update,
        interval=max(1, int(1000.0 / max(node.update_hz, 1.0))),
        cache_frame_data=False,
    )
    fig._anim = anim

    try:
        plt.show()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
