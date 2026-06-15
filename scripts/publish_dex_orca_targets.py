#!/usr/bin/env python3
"""Publish live DexSlide retargeting outputs to the OrcaHand ROS 2 topic."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dexslide.live import live_listener
from dexslide.paths import (
    DEFAULT_GLOVE_CALIBRATION_FILE,
    DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE,
    DEFAULT_SKELETON_FILE,
)
from dexslide.retargeting import create_dex_retargeter
from dexslide.retargeting.live_bridge import (
    FAULTY_GLOVE_JOINT_OVERRIDES_RAD,
    apply_glove_joint_overrides,
    format_named_values,
)
from dexslide.serial_angles import pick_default_port


def _create_ros_publisher(topic_name: str):
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
    except Exception as ex:
        raise RuntimeError(
            "ROS 2 publisher setup failed. `rclpy` must be imported from a Python interpreter "
            "that matches your ROS 2 installation ABI. If you are inside the `dexslide` conda "
            "environment and see `_rclpy_pybind11` import errors, run this publisher under the "
            "system ROS Python instead."
        ) from ex

    class DexToOrcaPublisher(Node):
        def __init__(self, ros_topic_name: str) -> None:
            super().__init__("dex_to_orca_targets")
            self.publisher = self.create_publisher(Float32MultiArray, ros_topic_name, 10)

        def publish_joint_targets(self, joint_targets: np.ndarray) -> None:
            message = Float32MultiArray(data=np.asarray(joint_targets, dtype=np.float32).tolist())
            self.publisher.publish(message)

    return rclpy, DexToOrcaPublisher, topic_name


def main() -> None:
    parser = argparse.ArgumentParser(description="DexSlide live retarget publisher for OrcaHand")
    parser.add_argument("--port", default=pick_default_port(), help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--mode", choices=["raw", "angles"], default="raw")
    parser.add_argument("--topic", default="/orca_hand/joint_targets")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE))
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--retarget-config", default=str(DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE))
    parser.add_argument(
        "--hand",
        choices=["left", "right", "auto"],
        default="auto",
        help="Physical glove/skeleton side used for human-hand reconstruction.",
    )
    parser.add_argument(
        "--mirror-reconstruction",
        action="store_true",
        help="Mirror the reconstructed human landmarks across the palm to treat a left glove as a right hand.",
    )
    parser.add_argument(
        "--dex-retargeting-python",
        default=None,
        help="Optional override interpreter for retarget runtime deps. Prefer the current project environment.",
    )
    args = parser.parse_args()

    if not args.port:
        raise SystemExit("No serial port found. Use --port /dev/ttyACM0")

    listener = live_listener(
        port=args.port,
        baud=args.baud,
        mode=args.mode,
        calib_file=args.calib_file,
    )
    retargeter = create_dex_retargeter(
        config_file=args.retarget_config,
        skeleton_file=args.skeleton_file,
        hand=args.hand,
        mirror_reconstruction=args.mirror_reconstruction,
        dex_retargeting_python=args.dex_retargeting_python,
    )

    period = 1.0 / max(args.fps, 1e-6)
    rclpy, publisher_cls, topic_name = _create_ros_publisher(args.topic)
    rclpy.init()
    node = publisher_cls(topic_name)
    human_joint_names = list(retargeter.human_model.joint_names)
    print(f"[retarget] outgoing joint unit: deg")
    print(f"[retarget] outgoing joint order: {', '.join(retargeter.joint_ids)}")
    print(f"[retarget] glove hand={args.hand}, mirror_reconstruction={args.mirror_reconstruction}")
    if FAULTY_GLOVE_JOINT_OVERRIDES_RAD:
        override_text = ", ".join(
            f"{name}={float(np.rad2deg(value)):6.2f} deg"
            for name, value in FAULTY_GLOVE_JOINT_OVERRIDES_RAD.items()
        )
        print(f"[retarget] temporary glove joint overrides enabled: {override_text}")
    try:
        while rclpy.ok():
            human_joint_angles, timestamp, _raw_line = listener.snapshot_rad20()
            if timestamp <= 0.0:
                time.sleep(period)
                continue
            overridden_human_joint_map = apply_glove_joint_overrides(
                human_joint_angles,
                human_joint_names,
            )
            robot_joint_angles_rad = retargeter.retarget(overridden_human_joint_map)
            robot_joint_angles_deg = np.rad2deg(robot_joint_angles_rad)
            print(
                f"[send] topic={args.topic} unit=deg t={timestamp:.3f} "
                f"{format_named_values(retargeter.joint_ids, robot_joint_angles_deg, unit='deg')}"
            )
            node.publish_joint_targets(robot_joint_angles_deg)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        retargeter.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
