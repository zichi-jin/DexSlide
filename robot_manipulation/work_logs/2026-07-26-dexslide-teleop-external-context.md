# DexSlide -> JAKA / OrcaHand 增量遥操外部背景交接

日期：2026-07-26

## 1. 文档用途

这份文档提供给负责 `robot_manipulation` 遥操实现的新对话，集中说明它不一定会主动查到的 DexSlide 外部上下文：

1. DexSlide 手套、相机和 marker body 的设备入口。
2. table ArUco 世界系和 marker body 位姿链。
3. DexAlign 的 `marker2hand`、skeleton 和 joint calibration 结果语义。
4. 手套工具系到 OrcaHand 工具系的固定姿态关系。
5. 左手控制右手时的世界系运动镜像。
6. JAKA 柔顺增量接口的已确认行为和安全启用顺序。

本文档不是遥操代码实现，也不处理 OrcaHand 手指关节映射。当前第一阶段只实现 JAKA 末端工具空间的增量遥操。

为避免把推测当成事实，后文使用以下标签：

- **硬约束**：用户已经明确指定，不应在实现中擅自改变。
- **仓库事实**：已经从当前代码或资产核对。
- **待验证**：实现前或真机测试时仍需确认。

## 2. 系统背景与当前范围

系统由以下部分组成：

1. 左手 DexSlide 外骨骼手套。机构和传感器主要位于手背，输出 20 维关节角。
2. 固结在手套背板上的多面 ArUco marker body。它和手套刚性连接，可由相机实时估计位姿。
3. Intel RealSense D435I，相机同时观察桌面参考 ArUco 和手套 marker body。
4. JAKA S5 机械臂，末端安装右手 OrcaHand。
5. 桌面参考 ArUco `id=0`，定义手套侧的世界坐标系。

当前任务范围：

1. 只做 JAKA TCP 的位置和姿态增量遥操。
2. 遥操一定是 **incremental teleoperation**，不是绝对姿态的影子模仿。
3. OrcaHand 手指关节映射暂缓，等 OrcaHand 本体和关节映射修好后再接。
4. 手套世界系必须是 table ArUco 系，禁止以 camera frame 作为遥操世界系。

## 3. 坐标系、变换方向和单位

### 3.1 记号

本文统一使用：

- `T`：table ArUco 世界系。
- `C`：camera frame。
- `M`：手套 marker body frame。
- `H`：DexSlide hand / wrist frame，即手套工具系。
- `B`：JAKA base / robot world frame。
- `O`：OrcaHand 所在的 JAKA tool frame。

`^A T_B` 表示“把 B 系坐标表达为 A 系坐标”的 4x4 齐次变换。

当前项目采用列向量和左乘变换：

```text
p_A = ^A T_B @ p_B
^A T_C = ^A T_B @ ^B T_C
```

长度单位需要明确区分：

- ArUco、marker body、`marker2hand` 和 JAKA 外部位姿链通常使用 `m`。
- JAKA `servo_p` / TCP 平移量使用 `mm`。
- JAKA `rx/ry/rz` 在 Python SDK 调用中使用 `rad`。
- DexAlign 内部 skeleton 长度使用 `mm`。
- `DexSlideHumanModel` 默认 `unit_scale=0.001`，输出 21 个点时单位是 `m`。

### 3.2 table、camera 和 marker body

Direct ArUco 检测器中的命名已经固定：

```text
frame_result["table_in_camera"]["matrix"]  = ^C T_T
frame_result["camera_in_table"]["matrix"] = ^T T_C
```

单个目标 marker 的 table 位姿为：

```text
^T T_target = inverse(^C T_T) @ ^C T_target
```

marker body 融合结果的数据类型是 `CubePoseEstimate`，虽然名称还保留了历史上的 `cube`，但手套场景下：

```text
CubePoseEstimate.transform_table_cube = ^T T_M
```

因此完整的手套 wrist pose 为：

```text
^T T_H = ^T T_M @ ^M T_H
```

这是遥操姿态输入的唯一正确主链。table marker 丢失时，不得退回使用 `^C T_M` 继续遥操，否则世界轴会变成相机视角轴。

### 3.3 table 系和 robot world 的关系

用户计划把 table ArUco 按机器人工作台零点和轴方向放到操作桌上，因此第一版可配置为：

```text
^B T_T = I
```

但这仍然必须作为显式配置保存，而不是把 identity 隐藏在代码中。现场需要验证 table 的 `x/y/z` 轴方向确实与 JAKA base/world frame 一致；若不一致，应只修改 `^B T_T`，不要在后续映射中继续叠加零散符号补丁。

## 4. table ArUco 与 marker body 资产

### 4.1 table ArUco

资产：

```text
assets/calibration/direct_aruco/table_aruco.yaml
```

当前配置：

```text
dictionary: DICT_4X4_50
table marker id: 0
marker size: 0.195 m
```

相关实现：

```text
dexslide/world_pose/direct_aruco_tracker.py
```

该模块会在实际分辨率变化时缩放相机内参。默认内参文件名仍是：

```text
assets/calibration/direct_aruco/d435i_960_540.json
```

虽然文件名是 `960_540`，当前运行分辨率默认是 `1280x720`，代码会通过 `_convert_fisheye_intrinsics_resolution(...)` 按实际画面尺寸换算内参。

### 4.2 左手 marker body

资产：

```text
assets/calibration/direct_aruco/left_tags2marker.json
```

当前配置事实：

```text
hand: left
dictionary: DICT_4X4_1000
marker ids: 1..18
marker square size: 40 mm
ArUco detection bound size: 30 mm
```

每个 marker 都记录了固定的 body-to-marker 旋转 `rot` 和位置 `p_mm`。

该文件包含 `//` 注释，不是严格标准 JSON。不要直接使用 `json.load(...)`。使用现有入口：

```python
from dexslide.world_pose.hand_cube_overlay import HandCubeOverlayConfig

config = HandCubeOverlayConfig.load(
    "assets/calibration/direct_aruco/left_tags2marker.json"
)
```

marker body 求解已有两种模式：

- `joint_pnp`：多 marker 角点联合 PnP，带重投影误差剔除；当前默认模式。
- `marker_average`：各 marker 独立 pose 反推 body pose 后融合。

现有 tracker：

```text
dexslide/world_pose/marker_body_pose_tracker.py
dexslide/world_pose/hand_cube_overlay.py
```

`MarkerBodyPoseTracker.update(...)` 会返回 raw 和 smoothed pose。没有可用 marker body pose 时，当前实现会清空旧 pose，而不是无限冻结上一帧。

## 5. DexAlign 外参与结果消费方式

### 5.1 DexAlign 解决什么问题

DexAlign 用 RGB-D MediaPipe 21 点、marker body pose 和手套 20 维角度离线校准：

1. palm 上 wrist 到五个 base 点的方向。
2. 手套角度到 DexSlide 手模型自由度的 affine mapping。
3. skeleton 长度和 `T_marker2hand` 平移。

DexAlign 2.0 中 `R_marker2hand` 固定为初始外参旋转，只优化 `T_marker2hand` 的平移。因此遥操所用 hand frame 姿态仍继承初始 `marker2hand` 的固定旋转。

### 5.2 当前应显式绑定的 session

当前 session：

```text
assets/calibration/dexalign/test_left_001/
```

主要产物：

```text
optimized_marker2hand.json
optimized_skeleton.json
optimized_joint_calibration.json
optimization_report.json
```

遥操第一阶段对工具空间 pose 真正必需的是：

```text
assets/calibration/dexalign/test_left_001/optimized_marker2hand.json
```

如果后续需要显示 DexSlide 21 点或接入关节映射，再同时使用该 session 中的 skeleton 和 joint calibration。

### 5.3 当前 `^M T_H`

`optimized_marker2hand.json` 的 `result` 为：

```text
translation_m = [
  0.08667014617290807,
  0.10673669112051717,
 -0.11812700253841443
]

rotation =
[[ 0,  1, 0],
 [-1,  0, 0],
 [ 0,  0, 1]]
```

完整矩阵：

```text
^M T_H =
[[ 0,  1, 0,  0.0866701462],
 [-1,  0, 0,  0.1067366911],
 [ 0,  0, 1, -0.1181270025],
 [ 0,  0, 0,  1.0         ]]
```

`marker_to_wrist_asset_transforms(...)` 的选择规则是优先 `result`，没有 `result` 时才回退 `initial_guess`。

### 5.4 一个关键加载陷阱

`left_tags2marker.json` 内部的：

```text
marker2wrist_path: left_marker2wrist.json
```

指向的是原始 direct-ArUco 外参，而不是 DexAlign session 的 optimized 外参。

因此，如果只调用 `HandCubeOverlayConfig.load(left_tags2marker.json)`，得到的 body-to-wrist 默认来自：

```text
assets/calibration/direct_aruco/left_marker2wrist.json
```

遥操实现必须随后显式加载选定 session 的 `optimized_marker2hand.json`，并覆盖 config：

```python
from dexslide.world_pose.hand_cube_overlay import (
    HandCubeOverlayConfig,
    marker_to_wrist_asset_transforms,
)

config = HandCubeOverlayConfig.load(tags_to_marker_path)
_initial, _result, active = marker_to_wrist_asset_transforms(marker2hand_path)
config.set_body_to_wrist_transform(active)
```

不要让 teleop 进程在不知情的情况下继续使用原始 `left_marker2wrist.json`。

### 5.5 禁止按“最新文件”分别选择结果

当前 `scripts/view_direct_aruco_overlay.py` 的默认逻辑会分别按修改时间寻找：

1. 最新 `optimized_skeleton.json`
2. 最新 `optimized_marker2hand.json`
3. 最新 `optimized_joint_calibration.json`

这三个文件理论上可能来自不同 session。遥操程序不要复用这个默认逻辑，应在一个配置中显式指定同一个 session 的全部路径，并校验它们的父目录一致。

### 5.6 joint calibration 的语义

DexAlign joint calibration 公式：

```text
q_calibrated = joint_scale * q_raw + joint_bias_rad
```

20 维顺序固定为：

```text
thumb:  DIP, PIP, MCP_front, MCP_back
index:  DIP, PIP, MCP_front, MCP_back
middle: DIP, PIP, MCP_front, MCP_back
ring:   DIP, PIP, MCP_front, MCP_back
pinky:  DIP, PIP, MCP_front, MCP_back
```

这些参数只是在 DexSlide human hand model 语境下校准 glove angle，不是 DexSlide -> OrcaHand 的最终关节映射。当前工具空间遥操阶段不要把它误当作 OrcaHand joint command。

## 6. DexSlide 全局通信配置

唯一通信配置：

```text
assets/dexslide_communications.json
```

当前左手关节通信：

```text
port: /dev/ttyACM0
stable_port: /dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_5CE284523731-if00
baud: 115200
mode: raw
startup_timeout_sec: 3.0
max_sample_age_sec: 0.5
```

当前相机通信：

```text
backend: realsense
model: Intel RealSense D435I
serial: 332522073507
opencv_source: /dev/video4
stable_opencv_source: /dev/v4l/by-path/pci-0000:00:14.0-usb-0:4:1.3-video-index0
width: 1280
height: 720
fps: 30
```

应使用：

```python
from dexslide.communications import (
    camera_communication,
    hand_joint_communication,
    resolve_camera_source,
    resolve_joint_port,
    resolve_realsense_serial,
)
```

禁止重新引入“扫描所有 USB/ACM 端口后随便选择一个”的 fallback。历史上这种行为曾把没有真实变化的默认 A-pose 当作有效 S2 数据录入，而画面显示使用了另一个串口，导致错误数据长时间未被发现。

`resolve_joint_port("left")` 会优先使用存在的 stable by-id 路径，否则使用配置中的 `/dev/ttyACM0`；它不执行随机设备扫描。

## 7. 手套工具系与 OrcaHand 工具系的固定姿态关系

### 7.1 两个工具系的物理定义

左手 DexSlide hand frame：

```text
x+：沿四指伸直方向
y+：与拇指方向相反
z+：从手掌指向手背
```

JAKA 上右手 OrcaHand tool frame：

```text
z+：沿四指方向
y+：拇指正向向掌心转过 45° 的方向
x+：从小指侧掌根向外后，再向掌心转过 45° 的方向
```

两者都是右手系。

### 7.2 同一物理姿态下的固定旋转

用户给出的等价关系为：

```text
R_orca = R_glove @ Ry(+90°) @ Rz(-45°)
```

定义：

```text
F = Ry(+90°) @ Rz(-45°)
```

则：

```text
R_orca_same_pose = R_glove @ F
```

数值矩阵：

```text
F =
[[ 0,          0,          1],
 [-0.70710678, 0.70710678, 0],
 [-0.70710678,-0.70710678, 0]]
```

这里是右乘，表示固连在 glove body 上的 intrinsic / body-fixed 旋转。不要擅自改为左乘。

这个固定变换主要用于：

1. 解释两个工具 frame 的同姿态关系。
2. 初始化检查。
3. 可视化对比。

它不意味着启动时要把机器人绝对姿态强制跳到 `R_glove @ F`。

## 8. 增量遥操的正确映射

### 8.1 为什么不能发绝对 glove pose

JAKA 必须先处于经验证安全的下垂姿态。遥操启动时应把当时的 glove pose 和 robot actual pose 同时设为 anchor，之后只映射相对 anchor 的运动。

如果直接把 `^T T_H` 转成绝对机器人目标：

1. 相机首次检测或重捕获时可能发生大跳变。
2. 机器人会试图离开安全初始姿态，追逐手套的绝对安装姿态。
3. 柔顺模式可能因初始姿态不合适而强烈震颤并进入安全模式。

### 8.2 左手控制右手的世界系镜像

运动关于世界系 `xOz` 平面镜像：

```text
glove y+     -> robot y-
glove rot_x+ -> robot rot_x-
glove rot_z+ -> robot rot_z-
```

定义反射矩阵：

```text
S = diag(1, -1, 1)
```

平移向量按普通向量变换：

```text
[dx, dy, dz] -> [dx, -dy, dz]
```

旋转向量是 axial vector，在反射下多一个 `det(S)=-1`，因此小角度符号为：

```text
[rx, ry, rz] -> [-rx, ry, -rz]
```

这和用户指定的 `rot_x`、`rot_z` 反向一致。

### 8.3 推荐的 anchor 公式

在遥操正式 arm 的时刻 `t0`，记录：

```text
p_H0, R_H0：glove wrist 在 table frame 下的位置和旋转
p_O0, R_O0：JAKA actual TCP 在 robot base 下的位置和旋转
```

手套平移增量：

```text
delta_p_T = p_H(t) - p_H0
```

若 `^B T_T = I`，机器人目标位置为：

```text
p_O_des = p_O0 + translation_scale * S @ delta_p_T
```

手套的世界系旋转增量：

```text
Delta_R_T = R_H(t) @ R_H0.T
```

镜像后的机器人旋转增量：

```text
Delta_R_B = S @ Delta_R_T @ S
R_O_des   = Delta_R_B @ R_O0
```

如果 `^B T_T` 不是 identity，令其旋转部分为 `R_BT`：

```text
delta_p_B = R_BT @ S @ delta_p_T
Delta_R_B = R_BT @ (S @ Delta_R_T @ S) @ R_BT.T
```

固定工具旋转 `F` 在这种世界系相对旋转中会抵消：

```text
(R_H(t) @ F) @ (R_H0 @ F).T
= R_H(t) @ R_H0.T
```

因此增量遥操中不要再把 `F` 重复叠加到每个旋转增量上。

### 8.4 连续下发时的增量

上式给出相对启动 anchor 的 desired pose。JAKA 使用 `servo_p(INCR)` 时，发送线程应根据当前 robot pose 或上一条已接受的目标，计算一个受限的小步增量，而不是每个周期重复发送从 `t0` 到当前时刻的完整累计位移。

建议逻辑：

1. pose estimator 持续更新 `desired pose`。
2. robot sender 以固定周期读取最新 desired pose。
3. 计算 `desired - actual/last_commanded` 的短步误差。
4. 经过 deadband、速度限制、每周期步长限制后调用 `servo_p(INCR)`。
5. 视觉数据丢失时不累计离线运动；恢复后必须重新 anchor 或由操作者重新 arm。

## 9. JAKA 柔顺增量接口的当前事实

现有参考脚本：

```text
robot_manipulation/scripts/jaka_compliant_teleop_ui.py
```

该脚本已经验证的命令形式：

```python
robot.servo_move_enable(True)
robot.servo_p(delta6, MoveMode.INCR, servo_step_num)
```

`delta6` 的当前单位：

```text
[dx, dy, dz]：mm
[rx, ry, rz]：rad
```

现有 UI 对单轴旋转按钮的处理是把 degree 转成 rad，然后只设置 `delta6[3/4/5]` 中的一个分量。它证明了“小幅单轴 `rx/ry/rz` 增量可以通过 `servo_p(INCR)` 下发”，但还不能单独证明任意组合旋转在 SDK 中究竟按 rotation vector、Euler 增量还是控制器特定规则组合。

**待验证**：正式连续遥操前，应在低步长、远离奇异和碰撞的位置，分别测试组合 `rx/ry/rz` 的左右乘语义。不要直接假定将一个 3D rotation vector 填进 `delta6[3:6]` 就一定等价于 `Delta_R_B`。

现有柔顺初始化使用：

```text
set_torque_sensor_mode(1)
load/write saved payload（若可用）
zero_end_sensor()
set_ft_ctrl_frame(0)                  # tool frame
set_admit_ctrl_config(axis, ...)
servo_move_enable(True)
set_compliant_type(1, 0)              # init
等待
set_compliant_type(0, 1)              # constant-force compliance
```

退出时已有顺序：

```text
set_compliant_type(0, 0)
disable_force_control()（若存在）
servo_move_enable(False)
logout / optional power off
```

`jaka_admittance_motion_trial.py` 中记录的 SDK servo 基准周期是 `8 ms`。这不代表视觉、估计和 Python 发送线程必须都以 125 Hz 运行，但 sender 的周期、`step_num` 和限速必须匹配 JAKA 控制器预期，不能让低频视觉更新直接产生大步阶跃。

## 10. 柔顺模式安全顺序

用户已经真机确认：JAKA 柔顺模式对初始姿态很敏感。如果手不够下垂，开启 compliant 后可能强烈震颤并摔入安全模式。

经验证安全的用户级初始 pose：

```text
move -100 400 300 -180 0 -45
```

解释：

```text
x/y/z = [-100, 400, 300] mm
rx/ry/rz = [-180, 0, -45] deg
```

写入 JAKA Python SDK 时，角度应转为 rad。

必须遵守以下顺序：

1. 连接 JAKA，检查 robot state、报警和急停状态。
2. 在未启用 compliant 的前提下，使用 `linear_move(ABS)` 移动到安全初始 pose。
3. 读取 actual TCP，确认位置和姿态均进入允许误差范围。
4. 确认 table ArUco、marker body 和 `^T T_H` 连续稳定。
5. 记录 glove anchor 和 robot anchor。
6. 执行力传感器、payload、servo 和 compliant 初始化。
7. compliant 已确认就绪后，才打开 teleop 数据通道。
8. 停止时先关闭 teleop 数据通道，再退出 compliant，最后关闭 servo。

现有 `jaka_compliant_teleop_ui.py` 本身没有“先移动并验证上述安全初始 pose”的逻辑，因此不能仅因为它能发送按钮增量，就直接把它原样当成完整遥操入口。

## 11. 必需的运行时状态机和 fail-safe

建议最少使用以下状态：

```text
DISCONNECTED
READY_FOR_SAFE_MOVE
SAFE_POSE_CONFIRMED
TRACKING_READY
COMPLIANT_READY
TELEOP_ACTIVE
FAULT / PAUSED
```

任何下列情况都应立即停止发送新的运动增量，并要求重新确认或重新 anchor：

1. table marker 丢失或超过 stale timeout。
2. marker body pose 丢失或超过 stale timeout。
3. 相邻 pose 出现超过阈值的平移或旋转跳变。
4. 相机时间戳倒退或数据长期不更新。
5. JAKA 报警、servo/compliant 状态退出或 SDK 调用失败。
6. 实际 TCP 与 commanded target 的误差持续过大。
7. 操作者释放 dead-man / enable 控制。

暂停期间不要继续积累 glove 位移；否则重新检测后会一次性补发大跳变。

## 12. `workspace_axis_mapping.json` 的当前状态与建议内容

目标资产：

```text
assets/teleop_robot_mappings/workspace_axis_mapping.json
```

当前文件只是占位，并且末尾有 trailing comma，属于无效标准 JSON，现阶段不要直接加载。实现遥操时需要把它重写成合法 schema。

建议至少保存：

```json
{
  "schema_version": 1,
  "mode": "incremental",
  "world_frame": "table_aruco",
  "robot_T_table": {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ]
  },
  "glove_to_orca_tool_rotation": [
    [0.0, 0.0, 1.0],
    [-0.70710678, 0.70710678, 0.0],
    [-0.70710678, -0.70710678, 0.0]
  ],
  "world_reflection_matrix": [
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "rotation_vector_sign": [-1.0, 1.0, -1.0],
  "translation_scale": 1.0,
  "rotation_scale": 1.0,
  "safe_start_pose": {
    "xyz_mm": [-100.0, 400.0, 300.0],
    "jaka_rx_ry_rz_deg": [-180.0, 0.0, -45.0]
  },
  "dexalign": {
    "session_dir": "assets/calibration/dexalign/test_left_001",
    "marker2hand_file": "assets/calibration/dexalign/test_left_001/optimized_marker2hand.json",
    "skeleton_file": "assets/calibration/dexalign/test_left_001/optimized_skeleton.json",
    "joint_calibration_file": "assets/calibration/dexalign/test_left_001/optimized_joint_calibration.json"
  }
}
```

实际 schema 还应补充：

1. translation / rotation deadband。
2. 每周期最大平移和旋转步长。
3. 最大平移和旋转速度。
4. pose stale timeout。
5. marker 重捕获后的 re-arm 策略。
6. safe pose 的位置和姿态容差。
7. `servo_step_num` 和 sender 周期。

不要同时保存 `rotation_vector_sign` 和 `world_reflection_matrix` 后又在代码中重复应用两遍。前者可作为说明或小角度诊断字段，真正矩阵映射应以 `S @ Delta_R @ S` 为唯一来源。

## 13. 推荐复用的 DexSlide API

通信：

```text
dexslide.communications.hand_joint_communication
dexslide.communications.resolve_joint_port
dexslide.communications.camera_communication
dexslide.communications.resolve_camera_source
dexslide.communications.resolve_realsense_serial
```

ArUco 世界系：

```text
dexslide.world_pose.direct_aruco_tracker
```

marker body 配置和变换：

```text
dexslide.world_pose.hand_cube_overlay.HandCubeOverlayConfig
dexslide.world_pose.hand_cube_overlay.marker_to_wrist_asset_transforms
dexslide.world_pose.hand_cube_overlay.estimate_cube_pose_in_table
dexslide.world_pose.hand_cube_overlay.make_transform
dexslide.world_pose.hand_cube_overlay.invert_transform
```

marker body 时间跟踪：

```text
dexslide.world_pose.marker_body_pose_tracker.MarkerBodyPoseTracker
```

如果需要关节流：

```text
dexslide.live.live_listener
dexslide.serial_angles.make_joint_order
```

如果需要 21 点手模型：

```text
dexslide.retargeting.human_model.DexSlideHumanModel
```

不要在 `robot_manipulation/scripts` 中复制一套 transform、串口扫描或 marker body 融合逻辑。也不要通过每个脚本临时修改 `sys.path` 解决 import；应从仓库根目录运行、安装当前 package，或提供统一 package entry point。

## 14. 机器人侧实现建议拆分

为了降低安全风险，建议把实现拆成以下独立职责：

1. `DexSlidePoseSource`：输出带 timestamp 和 quality 的 `^T T_H`，只负责视觉与外参链。
2. `IncrementalTeleopMapper`：管理 anchor、工具固定变换、镜像、倍率和 pose limits，不接触 JAKA SDK。
3. `JakaCompliantController`：负责 safe move、状态确认、compliant/servo 生命周期和 `servo_p(INCR)`。
4. `TeleopSupervisor`：管理 arm/pause/fault/re-anchor 和 stale timeout。
5. UI：只显示状态、质量、当前增量和 enable/dead-man，不应绕过 supervisor 直接发指令。

第一轮验收应先关闭实际运动，只记录以下量：

```text
^T T_M
^M T_H
^T T_H
glove anchor delta
mirrored robot delta
JAKA actual TCP
```

确认轴向和符号后，再以很小倍率开启单轴平移测试，然后是单轴旋转，最后才允许六维组合运动。

## 15. 实现时禁止事项

1. 禁止把 glove absolute pose 直接下发为 robot absolute pose。
2. 禁止 table marker 丢失后静默切换到 camera frame。
3. 禁止从不同 DexAlign session 混选 skeleton、marker2hand 和 joint calibration。
4. 禁止只加载 `left_tags2marker.json` 后误以为已经使用 optimized marker2hand。
5. 禁止随机扫描串口或相机设备作为无提示 fallback。
6. 禁止在 tracking 丢失期间积累运动，并在恢复时一次性补发。
7. 禁止在安全初始 pose 未到位时开启 compliant。
8. 禁止未经小步真机验证就假设 JAKA 组合 `rx/ry/rz` 的矩阵语义。
9. 禁止把 DexAlign joint calibration 当作 OrcaHand joint mapping。
10. 禁止把 `F`、镜像符号和其他轴变换散落硬编码在多个脚本中。

## 16. 最短接手结论

新对话开始实现时，应以以下事实为起点：

```text
glove world pose:
^T T_H = ^T T_M @ ^M T_H

same-pose tool frame relation:
R_orca = R_glove @ Ry(90 deg) @ Rz(-45 deg)

incremental translation mirror:
[dx, dy, dz] -> [dx, -dy, dz]

incremental world rotation mirror:
Delta_R_robot = S @ Delta_R_glove @ S
S = diag(1, -1, 1)

safe order:
safe linear move -> verify actual pose -> stable tracking/anchor
-> enable servo/compliant -> enable teleop data
```

当前唯一应绑定的 DexAlign pose 外参是：

```text
assets/calibration/dexalign/test_left_001/optimized_marker2hand.json
```

当前唯一 DexSlide 通信配置是：

```text
assets/dexslide_communications.json
```

当前 table 世界系配置是：

```text
assets/calibration/direct_aruco/table_aruco.yaml
```

在这些约束没有被明确替换前，不要重新发明另一套 frame、设备选择或 absolute teleop 逻辑。
