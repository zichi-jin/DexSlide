# orca_hand_ros

ROS 2 wrapper for the local OrcaHand `orca_core` dependency.

The node subscribes to:

- `/orca_hand/joint_targets` (`std_msgs/msg/Float32MultiArray`)

Message data must be ordered exactly as the active OrcaHand `config.yaml` `joint_ids`.
Values are passed to `OrcaHand.set_joint_positions()` without unit conversion, so use
the same unit convention as the OrcaHand config and calibration files.

## Build

From a ROS 2 workspace containing this package:

```bash
colcon build --packages-select orca_hand_ros
source install/setup.bash
```

## Run Without Hardware

```bash
ros2 run orca_hand_ros orca_hand_node --ros-args -p dry_run:=true
```

## Run With Hardware

```bash
ros2 run orca_hand_ros orca_hand_node
```

Useful parameters:

- `orca_core_path`: path containing the `orca_core` Python package.
- `config_path`: path to OrcaHand `config.yaml`; default is the local `models/v1/orcahand_right/config.yaml`.
- `topic_name`: target topic, default `/orca_hand/joint_targets`.
- `dry_run`: validate/log targets without connecting to hardware.
- `init_joints`: call `hand.init_joints(move_to_neutral=False)` after connect.

Example publisher:

```bash
ros2 topic pub --once /orca_hand/joint_targets std_msgs/msg/Float32MultiArray \
  "{data: [0, 50, 0, 20, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20]}"
```
