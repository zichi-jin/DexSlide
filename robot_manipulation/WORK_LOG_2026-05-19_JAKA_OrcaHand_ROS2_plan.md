# 工作日志与接续计划：JAKA S5 + OrcaHand ROS 2 控制

日期：2026-05-19
路径：`~/MyJob/DexSlide/robot_manipulation`

## 今天完成的状态

今天围绕 JAKA S5 + OrcaHand 的 ROS 2 控制平台做了环境确认、负载补偿方案调查和初步脚本准备。

### JAKA ROS 2 环境

用户今天已成功运行：

```bash
cd ~/MyJob/DexSlide/robot_manipulation/JAKA_control/jaka_ros2
ros2 launch jaka_driver robot_start.launch.py ip:=192.168.99.44
```

该命令现在可以启动 JAKA 机械臂 ROS 2 driver。

这说明 JAKA S5 的 ROS 2 环境基本已搭建完成。此前 `ros2 pkg list` 里似乎只有 `jaka_msgs`，现在已经能看到包括 `jaka_driver` 在内的约 25 个 JAKA 相关包。

### JAKA 控制器 IP

当前 JAKA S5 控制器 IP：

```text
192.168.99.44
```

之前误用了 `192.168.99.31`，后续不要再使用这个地址。

### JAKA 负载识别 API 调查

已确认 JAKA 官方支持未知末端负载识别，不应该手填 OrcaHand 官网重量。

官方流程对应 API：

- `start_torq_sensor_payload_identify(jointPosition)`：开始传感器末端负载辨识，会触发机械臂运动到指定关节终点。
- `get_torq_sensor_identify_staus()`：获取辨识状态。
  - `0`：辨识完成，结果可以读取。
  - `1`：辨识中，暂无结果。
  - `2`：辨识失败。
- `get_torq_sensor_payload_identify_result()`：读取识别出的 `mass` 和 `centroid`。
  - `mass` 单位：kg。
  - `centroid` 单位：mm。
- `set_torq_sensor_tool_payload()` / 官方 TCP 指令 `set_tool_payload`：设置传感器末端负载，影响力控 / 力矩补偿。
- `set_payload()`：设置机器人动力学 payload，影响机械臂动力学 / 重力补偿。

注意：`payload` 和 `tool_payload` 含义不同，后续脚本应在辨识完成后同时设置两者。

### 已写脚本

已创建脚本：

```text
JAKA_control/jk_scripts/identify_and_apply_payload.py
```

用途：自动识别未知 JAKA S5 末端负载，并把识别结果同时写入：

- `set_torq_sensor_tool_payload()`：力控 / 力矩补偿。
- `set_payload()`：机器人动力学 / 重力补偿。

运行形式：

```bash
python3 JAKA_control/jk_scripts/identify_and_apply_payload.py \
  --end-joint J1 J2 J3 J4 J5 J6
```

其中 `J1~J6` 是负载辨识终点，单位 rad。

脚本已做过语法检查和 `--help` 校验，但没有实际连接或移动机器人。

## 今天未完成的点

### 负载识别和补偿未完成

目前没有运行 `identify_and_apply_payload.py`。

原因：不知道 `J1~J6` 该设置成什么安全值。该命令会触发机械臂运动，如果贸然设置终点，存在碰撞风险。

下一步需要先确定一个 JAKA S5 负载辨识安全终点，然后再运行脚本完成真实末端负载识别。

未完成项：

1. 查 JAKA 力控产品使用手册，确认负载辨识终点设置要求。
2. 根据当前工位、JAKA S5、OrcaHand 安装姿态选择一个安全的 `J1~J6`。
3. 运行：

```bash
python3 JAKA_control/jk_scripts/identify_and_apply_payload.py \
  --end-joint J1 J2 J3 J4 J5 J6
```

4. 确认脚本输出的 `mass` 和 `centroid` 合理。
5. 确认已同时写入 `set_torq_sensor_tool_payload()` 和 `set_payload()`。

## 明天计划：JAKA + OrcaHand ROS 2 协同控制测试

目标：写一个 JAKA 和 OrcaHand 通过 ROS 2 协同控制的测试脚本。

### 整体控制设想

机器人控制流程预想：

1. 机械臂先移动至某个用户认为“能够代表此刻自己姿态”的动作姿态。
2. 以此机械臂姿态作为零点开始跟踪。
3. 后续 SLAM 获得用户腕部位姿，JAKA 机械臂已有末端位姿。
4. 两者天然具有相同单位，因此可以尝试让 SLAM 末端位姿 1:1 传导到 JAKA 末端位姿。
5. 当前阶段先不接 retarget 和视觉 SLAM，先做人工控制闭环测试。

### 预想测试脚本功能

伪代码方向：

```python
import jkrc
import orca_core
import ros2
import tkinter as tk

robot_hand = orca_core.Hand(port="/dev/ttyUSB0", config_path="..")
robot_arm = jkrc.robot(ip="192.168.99.44")

robot_arm.start()
robot_hand.connect()

# JAKA: 键盘控制末端位姿
# w/s/a/d/up/down: x+/x-/y+/y-/z+/z-
# space + w/s/a/d/up/down: rotx+/rotx-/roty+/roty-/rotz+/rotz-

# OrcaHand: 参考 orca_core/scripts 里的 slider_joints-like UI 控制手指

tk.TK()
while True:
    key_command = keyboard.get()
    target_jaka_pos += key2mov_dict[key_command] * 0.001
    ros2.publish(node="jaka_driver/linear_move", target_jaka_pos)

    target_orca_joints = orcahand.get_from(slider_UI)
    ros2.publish(node="orca_core/joint", target_orca_joints)
```

以上只是方向，明天需要按本仓库实际 ROS 2 service / topic 类型重写。

### JAKA 侧明天要做

1. 确认 `/jaka_driver/linear_move` 的服务类型和字段。
2. 写一个 Python ROS 2 client，能发送很小的末端线性移动。
3. 确认单位：位置应该是 mm 或 m，姿态是 rad，必须读 `jaka_msgs/srv/Move.srv` 和 driver 实现确认。
4. 先不要大幅移动，只测试极小增量。
5. 后续把键盘输入映射成目标末端位姿。

### OrcaHand 侧明天要做

用户今天已把 OrcaHand 官方仓库 `orca_core` 复制进项目：

```text
orca_control/orca_dependencies
```

但目前还不清楚：

1. 如何正确实例化一只 OrcaHand。
2. 使用哪个 `config.yaml`。
3. 真实硬件端口是否是 `/dev/ttyUSB0`。
4. 官方是否已有类似 `jaka_driver` 的 ROS 2 底层驱动包。
5. 如果没有，就需要自己创建 ROS 2 node / topic wrapper。

明天应先做：

1. 阅读 `orca_control/orca_dependencies/README.md`。
2. 阅读 `orca_control/orca_dependencies/scripts/slider_joint.py`。
3. 阅读 `orca_control/orca_dependencies/orca_core/hardware_hand.py`。
4. 找到正确的 OrcaHand 实例化方式。
5. 单独运行或复用官方 demo，让手能动。
6. 如果没有官方 ROS 2 driver，创建一个 ROS 2 wrapper node，例如：

```text
/orca_hand/joint_targets
```

消息内容可以先用 `Float32MultiArray` 或更清晰的自定义结构，后续再规范化。

### 最小闭环建议

明天不要一上来接 SLAM / retarget。建议顺序：

1. 启动 JAKA driver：

```bash
cd ~/MyJob/DexSlide/robot_manipulation/JAKA_control/jaka_ros2
ros2 launch jaka_driver robot_start.launch.py ip:=192.168.99.44
```

2. 写并运行一个最小 JAKA ROS 2 client，发送小幅 `linear_move`。
3. 单独确认 OrcaHand 可以通过 `orca_core` 动起来。
4. 给 OrcaHand 写 ROS 2 wrapper node。
5. 写统一键盘 + slider UI 测试脚本，同时控制 JAKA 和 OrcaHand。
6. 之后同学的 retarget 和视觉 SLAM 完成后，只需要新增 ROS 2 节点，并让机器人侧订阅对应 topic 即可衔接。

## 当前风险

1. JAKA 负载辨识终点未知，不能贸然运行。
2. OrcaHand 的 ROS 2 driver 是否存在未知。
3. OrcaHand 具体硬件端口和配置文件未知。
4. JAKA `/linear_move` 单位和姿态字段必须确认后才能控制。
5. 协同控制前必须先单独验证 JAKA 和 OrcaHand 两侧都可控。
