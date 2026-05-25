from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


DEFAULT_ORCA_CORE_PATH = Path(__file__).resolve().parents[2] / "orca_dependencies"


class OrcaHandNode(Node):
    """Subscribe to joint targets and forward them to orca_core."""

    def __init__(self) -> None:
        super().__init__("orca_hand_node")

        self.declare_parameter("orca_core_path", str(DEFAULT_ORCA_CORE_PATH))
        self.declare_parameter("config_path", "")
        self.declare_parameter("topic_name", "/orca_hand/joint_targets")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("init_joints", True)

        orca_core_path = self.get_parameter("orca_core_path").value
        if orca_core_path:
            sys.path.insert(0, str(Path(orca_core_path).expanduser()))

        from orca_core import OrcaHand, OrcaJointPositions

        self.OrcaJointPositions = OrcaJointPositions
        self.dry_run = bool(self.get_parameter("dry_run").value)

        config_path = str(self.get_parameter("config_path").value or "")
        self.hand = OrcaHand(config_path=config_path or None)
        self.joint_ids = list(self.hand.config.joint_ids)

        if not self.dry_run:
            success, message = self.hand.connect()
            if not success:
                raise RuntimeError(f"Failed to connect OrcaHand: {message}")
            if bool(self.get_parameter("init_joints").value):
                self.hand.init_joints(move_to_neutral=False)

        topic_name = str(self.get_parameter("topic_name").value)
        self.create_subscription(Float32MultiArray, topic_name, self.on_target, 10)

        self.get_logger().info(f"Subscribed to {topic_name}")
        self.get_logger().info("Joint order: " + ", ".join(self.joint_ids))
        if self.dry_run:
            self.get_logger().warn("dry_run=true: not sending commands to hardware")

    def on_target(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != len(self.joint_ids):
            self.get_logger().error(
                f"Expected {len(self.joint_ids)} joint values, got {len(msg.data)}"
            )
            return

        joint_pos = self.OrcaJointPositions.from_ndarray(
            np.asarray(msg.data, dtype=np.float64),
            joint_ids=self.joint_ids,
        )

        if self.dry_run:
            self.get_logger().info(f"Target: {joint_pos.as_dict()}")
            return

        self.hand.set_joint_positions(joint_pos)

    def destroy_node(self) -> bool:
        if hasattr(self, "hand") and not self.dry_run:
            self.hand.disconnect()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OrcaHandNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
