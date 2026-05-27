#!/usr/bin/env python3
"""Subscribe /camera/camera/color/image_raw, count messages, report FPS."""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image


class RgbFpsCounter(Node):
    def __init__(self, duration_s: float, use_reliable: bool) -> None:
        super().__init__("rgb_fps_counter")
        self.duration_s = duration_s
        self.first_stamp_ns: int | None = None
        self.last_stamp_ns: int | None = None
        self.first_recv_ns: float | None = None
        self.last_recv_ns: float | None = None
        self.count = 0
        self.periods_ms: list[float] = []
        self.prev_stamp_ns: int | None = None
        qos = (
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            if use_reliable
            else qos_profile_sensor_data
        )
        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.on_image,
            qos,
        )

    def on_image(self, msg: Image) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        recv_ns = time.monotonic_ns()
        if self.first_stamp_ns is None:
            self.first_stamp_ns = stamp_ns
            self.first_recv_ns = recv_ns
        if self.prev_stamp_ns is not None:
            self.periods_ms.append((stamp_ns - self.prev_stamp_ns) / 1e6)
        self.prev_stamp_ns = stamp_ns
        self.last_stamp_ns = stamp_ns
        self.last_recv_ns = recv_ns
        self.count += 1


def main() -> int:
    duration_s = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    use_reliable = "--reliable" in sys.argv
    rclpy.init()
    node = RgbFpsCounter(duration_s, use_reliable)
    deadline = time.monotonic() + duration_s
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.count < 2:
        print(f"only {node.count} frames received", flush=True)
        rclpy.shutdown()
        return 1
    span_stamp_s = (node.last_stamp_ns - node.first_stamp_ns) / 1e9
    span_recv_s = (node.last_recv_ns - node.first_recv_ns) / 1e9
    fps_stamp = (node.count - 1) / span_stamp_s if span_stamp_s > 0 else 0.0
    fps_recv = (node.count - 1) / span_recv_s if span_recv_s > 0 else 0.0
    periods_sorted = sorted(node.periods_ms)
    median = periods_sorted[len(periods_sorted) // 2]
    p99 = periods_sorted[int(0.99 * (len(periods_sorted) - 1))]
    print(
        f"frames={node.count} duration_stamp={span_stamp_s:.3f}s "
        f"fps_by_stamp={fps_stamp:.3f} fps_by_recv={fps_recv:.3f} "
        f"period_ms median={median:.3f} p99={p99:.3f} max={periods_sorted[-1]:.3f}",
        flush=True,
    )
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
