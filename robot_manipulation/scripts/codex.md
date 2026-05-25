2026-05-21 最终状态总结
=======================

目标
----

今天的目标是做一个 JAKA + OrcaHand 的 ROS 2 上位机 UI，并提供 VS Code 内一键启动方式：

- 启动 JAKA driver。
- 启动 OrcaHand ROS node。
- 启动 Tkinter 上位机 UI。
- UI 通过 slider 控制 OrcaHand。
- UI 通过按钮或键盘调用 JAKA `/jaka_driver/linear_move` service。
- 避免要求用户在命令行手动输入续行符。
- 避免 `(dexslide)` Conda Python 3.11 和 ROS Jazzy Python 3.12 的 `rclpy` ABI 不匹配问题。

当前脚本说明
------------

- `scripts/setup_dexslide_ros2.bash`
  - 统一 ROS 2 环境入口。
  - source `/opt/ros/jazzy/setup.bash`。
  - source JAKA ROS 2 workspace install。
  - source OrcaHand ROS package install。
  - 设置 `ORCA_CORE_PATH`、`ORCA_HAND_CONFIG` 和相关 `PYTHONPATH`。

- `scripts/teleop_jaka_orca_ros2.py`
  - Tkinter UI 主程序。
  - 内部创建 ROS 2 node：`jaka_orca_teleop`。
  - 发布 OrcaHand 目标到 `/orca_hand/joint_targets`。
  - 调用 JAKA service：`/jaka_driver/linear_move`。
  - 订阅 JAKA 当前 TCP 位姿：`/jaka_driver/tool_position`。
  - 已加 JAKA in-flight 防重入保护：上一条 JAKA motion 没返回前，不再发下一条 motion。
  - 已加 OrcaHand slider 发布节流：默认 `--hand-publish-interval-ms 50`。

- `scripts/run_teleop_jaka_orca_ros2.bash`
  - 单独启动 UI 的推荐入口。
  - 自动 source `scripts/setup_dexslide_ros2.bash`。
  - 强制使用 `/usr/bin/python3` 启动 UI，避免 Conda Python 加载 ROS Jazzy `rclpy` extension 失败。

- `scripts/run_teleop_jaka_orca_ros2_stack.bash`
  - 完整 stack 启动入口。
  - 先启动 JAKA driver。
  - 再启动 OrcaHand ROS node。
  - 等待短暂启动时间后启动 UI。
  - 关闭 UI 时尝试清理后台启动的两个 ROS 进程。

- `.vscode/tasks.json`
  - VS Code task 配置。
  - `DexSlide: Start ROS nodes + Teleop UI`：启动完整 stack。
  - `DexSlide: Teleop UI only`：只启动 UI，适合两个 ROS node 已经手动启动时使用。
  - 也保留单独启动 `DexSlide: JAKA driver` 和 `DexSlide: OrcaHand node` 的 task。

推荐启动方式
------------

VS Code：

```text
Ctrl+Shift+P -> Tasks: Run Task -> DexSlide: Start ROS nodes + Teleop UI
```

单行终端命令：

```bash
bash scripts/run_teleop_jaka_orca_ros2_stack.bash
```

如果 JAKA driver 和 OrcaHand node 已经启动，只启动 UI：

```bash
bash scripts/run_teleop_jaka_orca_ros2.bash --orca-config /home/jzq/MyJob/DexSlide/robot_manipulation/orca_control/orca_dependencies/orca_core/models/v1/orcahand_right/config.yaml
```

不要使用：

```bash
bash scripts/run_teleop_jaka_orca_ros2.bash \ --orca-config /path/to/config.yaml
```

原因：`\ ` 会把空格转义进参数，Python 收到的是 `" --orca-config"`，`argparse` 无法识别。

手动分步启动方式
----------------

初始化环境：

```bash
conda activate dexslide
cd /home/jzq/MyJob/DexSlide/robot_manipulation
source scripts/setup_dexslide_ros2.bash
```

启动 JAKA driver：

```bash
ros2 launch jaka_driver robot_start.launch.py ip:=192.168.99.44
```

启动 OrcaHand ROS node：

```bash
ros2 run orca_hand_ros orca_hand_node --ros-args -p config_path:=/home/jzq/MyJob/DexSlide/robot_manipulation/orca_control/orca_dependencies/orca_core/models/v1/orcahand_right/config.yaml
```

启动 UI：

```bash
bash scripts/run_teleop_jaka_orca_ros2.bash --orca-config /home/jzq/MyJob/DexSlide/robot_manipulation/orca_control/orca_dependencies/orca_core/models/v1/orcahand_right/config.yaml
```

UI 控制方式
-----------

- OrcaHand：拖动 UI slider，向 `/orca_hand/joint_targets` 发布 `std_msgs/msg/Float32MultiArray`。
- JAKA：点击按钮或使用键盘，调用 `/jaka_driver/linear_move`。
- 普通按键：`w/s/a/d/Up/Down` 控制 `x/y/z` 平移。
- 按住空格：同样按键切换为控制 `rx/ry/rz` 旋转。
- 默认平移步长：`--xyz-step-mm 1.0`。
- 默认旋转步长：`--rpy-step-rad 0.01`。
- 默认 JAKA 速度和加速度：`--jaka-velocity 20.0 --jaka-acceleration 20.0`。
- 默认 OrcaHand 发布节流：`--hand-publish-interval-ms 50`。

已验证内容
----------

- `/usr/bin/python3 -m py_compile scripts/teleop_jaka_orca_ros2.py` 通过。
- `bash scripts/run_teleop_jaka_orca_ros2.bash --help` 通过。
- `.vscode/tasks.json` JSON 语法通过。
- `scripts/run_teleop_jaka_orca_ros2_stack.bash` bash 语法通过。
- UI wrapper 确认会使用 `/usr/bin/python3`，不再触发 Conda Python 3.11 与 ROS Jazzy Python 3.12 的 `rclpy._rclpy_pybind11` ABI 错误。

当前已知问题
------------

截至 2026-05-21 晚上，功能仍未达到可用状态：

- JAKA 仍不能通过 UI 正常移动。
  - 用户反馈仍然出现：
    `error occurred:ERR_MOTION_ABNORMAL`
  - 已尝试在 UI 侧避免连续 service 堆叠，但问题仍存在。
  - 因此问题可能不只是 UI 高频触发，也可能与目标位姿格式、当前 TCP 位姿来源、姿态表达、运动模式、机器人状态、限位、碰撞/异常状态、JAKA driver 的 `linear_move` 封装有关。

- OrcaHand 不是“慢”，而是“卡”。
  - 已尝试 slider 发布节流，但用户反馈卡顿仍存在。
  - 这说明问题可能不只是 ROS topic 发布频率，也可能在 `orca_hand_node` 到实际手部控制链路、`orca_core` 调用、硬件通信、阻塞式控制 API、或 UI 与 ROS callback 的交互方式上。

JAKA 相关观察
-------------

`JAKA_control/jaka_ros2/src/jaka_driver/src/jaka_driver.cpp` 中 `/jaka_driver/linear_move` 的核心逻辑：

- service 名称：`/jaka_driver/linear_move`。
- request 类型：`jaka_msgs/srv/Move`。
- 使用 `request->pose[0:6]` 作为目标位姿。
- `pose[0:3]` 被作为 Cartesian translation。
- `pose[3:6]` 被当成 angle-axis 向量，经 `Angaxis2Rot()` 和 `rot_matrix_to_rpy()` 转成 JAKA SDK 的 RPY。
- 调用 `robot.linear_move(&end_pose, MoveMode::ABS, TRUE, speed, accel, tol, option_cond)`。
- `TRUE` 表示 SDK 调用是阻塞运动。
- 非 0 返回码会映射为错误字符串，例如 `-12 -> ERR_MOTION_ABNORMAL`。

后续重点确认：

- `/jaka_driver/tool_position` 发布的 angular 分量到底是 degree、radian、RPY，还是别的表示。
- UI 中 `_on_tool_position()` 当前把 angular 分量做了 `math.radians()`，这可能和 driver 的 `linear_move` 姿态输入约定不一致。
- `linear_move` service 期望的 `pose[3:6]` 是 angle-axis，但 `/tool_position` 未必提供 angle-axis。
- 当前 UI 以 `target_pose += step` 的方式更新姿态，可能对 angle-axis 表示不成立。
- 下一次建议先只测试纯 `x/y/z` 小步移动，并保持姿态完全等于 driver/client 示例中的已知合法姿态。

OrcaHand 后续排查方向
---------------------

- 单独绕过 UI，用 `ros2 topic pub --once` 或低频循环发布固定目标，确认卡顿是否来自 UI。
- 在 `orca_hand_node` 中加时间戳日志，测量：
  - 收到 ROS topic 的时间。
  - 调用 `orca_core` 的开始/结束时间。
  - 硬件控制 API 返回时间。
- 判断 `orca_core` 或硬件通信是否阻塞。
- 如果 node 内部控制调用耗时明显，应考虑在 `orca_hand_node` 内做目标缓存和固定频率控制循环，而不是每收到一个 topic 就同步控制硬件。

下一次建议顺序
--------------

1. 先脱离 UI，用 `ros2 service call /jaka_driver/linear_move ...` 复现一个最小 JAKA 运动。
2. 确认 JAKA 当前位姿 topic 的单位和姿态表示。
3. 暂时禁用 UI 姿态控制，只保留纯平移，并复用当前真实姿态。
4. 对 OrcaHand 做 UI 绕过测试，确认卡顿来自 UI、ROS node、`orca_core` 还是硬件通信。
5. 如果 OrcaHand 卡顿在 node 内部，重构为“ROS topic 更新目标缓存 + 固定频率硬件控制 loop”。
