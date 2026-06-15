# DexSlide

DexSlide 是一个外骨骼数据手套项目仓库，当前主仓库聚焦上位机 Python 软件，包含 3 条主线：

- `20 DOF` 手套关节角采集后的手部重建与可视化。
- 基于桌面参考 ArUco 与目标 ArUco 的直接世界位姿估计。
- DexSlide 人手重建结果到 OrcaHand 的 retargeting 与 ROS 2 发布桥接。

STM32 固件、PCB、BOM 和装配资料已拆分到独立仓库 `dexslide_infra`。本仓库不再承担下位机固件构建。

## 当前推荐路线

视觉位姿这条线目前推荐使用 `direct ArUco` 方案，而不是 `umi_mono + ORB-SLAM3`：

- 固定桌面参考 tag 作为 `world/table` 原点。
- 在同一帧中同时检测桌面 tag 和目标 tag。
- 分别做两次 `PnP`，再用
  `T_table_target = inv(T_camera_table) @ T_camera_target`
  直接得到目标在世界系下的位姿。

这条路线已经在仓库里有完整实现和可视化脚本，部署复杂度明显低于 SLAM。

## 仓库内容

```text
.
├── README.md
├── main.py
├── requirements.txt
├── requirements-retargeting.txt
├── assets/
│   ├── calibration/
│   │   ├── glove_calibration.json
│   │   └── direct_aruco/
│   ├── robot_hands/
│   │   └── orcahand_description/
│   └── skeletons/
├── dexslide/
│   ├── calibration/
│   ├── kinematics/
│   ├── retargeting/
│   ├── vision/
│   ├── visualization/
│   ├── world_pose/
│   └── live.py
├── dex_retargeting/
├── robot_manipulation/
├── scripts/
├── tests/
├── third_party/
└── umi_mono/
```

说明：

- `dexslide/` 是当前主 Python 包。
- `dexslide/world_pose/` 放 direct ArUco 世界位姿跟踪。
- `dexslide/retargeting/` 放 DexSlide -> OrcaHand 的人手模型与 retarget bridge。
- `dex_retargeting/` 是随仓库一起分发的 vendored retargeting 代码。
- `assets/robot_hands/orcahand_description/` 是 vendored OrcaHand 资产。
- `umi_mono/` 仍保留历史单目 SLAM 与 ROS 2 实验链路，但不再是当前首选方案。

## 环境安装

### 基础环境

用于手套重建、Matplotlib 可视化和 direct ArUco 世界位姿工具：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

基础依赖包括：

- `pyserial`
- `opencv-contrib-python`
- `PyYAML`
- `mediapipe`
- `numpy`
- `matplotlib`
- `vedo`

### 可选：retargeting 额外依赖

DexSlide -> OrcaHand 的 retargeting 需要更重的运动学依赖，单独放在：

```bash
pip install -r requirements-retargeting.txt
```

这会安装 `pin`、`nlopt`、`pytransform3d`、`trimesh`、`torch` 等依赖。

## 工作流 A：手套骨骼与实时重建

### 1. 准备下位机数据源

DexSlide 手套的 STM32 固件与 ADC 标定脚本位于 `dexslide_infra`。本仓库默认你已经能从串口读到 20 路关节相关数据。

### 2. 离线提取 skeleton

把裸手放在 A4 纸上拍多张图，默认放到 `assets/photos/`，然后运行：

```bash
python main.py calibrate-skeleton
```

常用参数：

```bash
python main.py calibrate-skeleton --show-debug
python main.py calibrate-skeleton --reuse-a4
python main.py calibrate-skeleton --input-dir assets/photos --skeleton-aggregate median
```

默认输出：

```text
assets/skeletons/skeleton.json
assets/skeletons/offline_bone_mm_results.json
```

### 3. 标定 ADC 到角度映射

这个步骤在 `dexslide_infra` 仓库执行，结果写回当前仓库：

```bash
cd ../dexslide_infra
python scripts/glove_calibrate.py \
  --port /dev/ttyACM0 \
  --out ../DexSlide/assets/calibration/glove_calibration.json
```

### 4. 实时 3D 重建

```bash
python main.py run --port /dev/ttyACM0
```

旧入口仍保留：

```bash
python scripts/glove_live_3d.py --port /dev/ttyACM0
```

如果串口输入已经是角度流，也可以使用：

```bash
python main.py run --port /dev/ttyACM0 --mode angles
```

## 工作流 B：direct ArUco 世界位姿

### 场景假设

- 桌面参考 ArUco 固定不动。
- 相机和目标 ArUco 都可以运动。
- 当前实现要求同一帧里至少看到：
  - `table marker`
  - 一个或多个 `target marker`

### 默认本地标定资源

仓库已经自带一套默认 direct ArUco 配置：

```text
assets/calibration/direct_aruco/d435i_960_540.json
assets/calibration/direct_aruco/table_aruco_4x4_120mm.yaml
assets/calibration/direct_aruco/target_aruco_4x4_50mm.yaml
```

当前默认口径：

- 字典：`DICT_4X4_50`
- 桌面参考 tag：`id=0`，边长 `120 mm`
- 目标 tag：默认边长 `50 mm`

### 1. 3D 轨迹查看

```bash
python scripts/plot_aruco_relative_pose_3d.py \
  --source /dev/video4 \
  --target-marker-ids 5
```

这个脚本会在 `table/world` 坐标系下显示：

- 相机轨迹
- 目标轨迹
- 当前各自坐标轴

### 2. 相机画面 overlay 查看

```bash
python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --target-marker-ids 5
```

这个脚本会在实时相机图像上直接画出：

- 桌面 tag 轮廓与坐标轴
- 目标 tag 轮廓与坐标轴
- 当前检测状态与相对位姿信息

它是检查“坐标系是否真正粘在实物上”的首选工具。

### 3. 代码入口

- 跟踪器：`dexslide/world_pose/direct_aruco_tracker.py`
- 底层检测：`dexslide/vision/aruco_pose_tracker.py`
- 3D plot：`scripts/plot_aruco_relative_pose_3d.py`
- 相机 overlay：`scripts/view_direct_aruco_overlay.py`

### 已知约束

- 如果桌面参考 tag 完全出画，则 `world/table` 系下的目标位姿会中断。
- 快速运动时主要瓶颈仍可能来自曝光时间和 motion blur，而不只是检测参数。
- 若单个桌面 tag 视野受限，建议升级为多 table tags 布局，而不是继续依赖 SLAM。

## 工作流 C：DexSlide -> OrcaHand retargeting

### 1. 实时对比查看

安装完 `requirements-retargeting.txt` 后，可以用下面的 viewer 对比：

```bash
python scripts/glove_live_retarget_compare.py \
  --port /dev/ttyACM0 \
  --mode raw
```

它会同时显示：

- 由 DexSlide 20 维关节角重建出的人手姿态
- retarget 到 OrcaHand 后的机器人手姿态

### 2. 发布到 OrcaHand ROS 2 话题

```bash
python scripts/publish_dex_orca_targets.py \
  --port /dev/ttyACM0 \
  --topic /orca_hand/joint_targets
```

注意：

- 这个发布脚本依赖 `rclpy`。
- 如果你在 conda 环境里遇到 `_rclpy_pybind11` ABI 问题，通常需要改用与你本机 ROS 2 安装匹配的 Python 解释器运行。
- 下游话题与单位约定请同步查看 `robot_manipulation/orca_control/orca_hand_ros/README.md`。

### 3. 相关入口

- 共享串口监听：`dexslide/live.py`
- retarget API：`dexslide/retargeting/`
- 发布脚本：`scripts/publish_dex_orca_targets.py`
- 可视化对比：`scripts/glove_live_retarget_compare.py`

## `main.py` 当前命令

```bash
python main.py calibrate-skeleton --help
python main.py run --help
python main.py raw --help
```

顶层入口当前保留 3 个子命令：

- `calibrate-skeleton`
- `run`
- `raw`

## `umi_mono` 的位置

`umi_mono/` 仍保留：

- 单目 SLAM / ROS 2 实验链路
- `dexslide_slam_publisher`
- 历史标定与数据处理脚本

当前它更适合作为历史实验与备份路径，而不是日常首选部署方案。

## 测试

纯 Python 单元测试推荐用下面方式运行，避免环境里的 ROS 2 `pytest` 插件污染：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_direct_aruco_tracker.py -q
```

其余测试可按需在对应环境里执行。

## 后续方向

当前仓库后续重点仍然是 4 条：

- DexSlide 20 DOF 手套数据的稳定采集与重建。
- 更稳健的世界位姿方案，优先 direct ArUco / 多参考 tag。
- DexSlide 到 OrcaHand 的实时 retarget bridge。
- 机器人侧集成，包括 OrcaHand 与 JAKA/Orca 控制链路。

## License

TBD
