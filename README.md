# DexSlide

DexSlide 是一个外骨骼数据手套项目仓库，当前主仓库聚焦上位机 Python 软件，包含 2 条主线：

- `20 DOF` 手套关节角采集后的手部重建与可视化。
- 基于桌面参考 ArUco 与目标 ArUco 的直接世界位姿估计。

STM32 固件、PCB、BOM 和装配资料已拆分到独立仓库 `dexslide_infra`。本仓库不再承担下位机固件构建。

## 当前推荐路线

视觉位姿使用 `direct ArUco` 方案：

- 固定桌面参考 tag 作为 `world/table` 原点。
- 在同一帧中同时检测桌面 tag 和目标 tag。
- 对普通单目标 tag，分别做两次 `PnP`，再用
  `T_table_target = inv(T_camera_table) @ T_camera_target`
  直接得到目标在世界系下的位姿。
- 对手背 `18` 面 marker body，则把全部可见 marker 的角点一起做联合 `PnP`，直接解出融合后的 marker body 位姿。

这条路线已经在仓库里有完整实现和可视化脚本。

## 仓库内容

```text
.
├── README.md
├── main.py
├── requirements.txt
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
│   ├── vision/
│   ├── visualization/
│   └── live.py
├── robot_manipulation/
│   ├── JAKA_control/
│   ├── orca_control/
│   ├── assets/
│   └── scripts/
├── scripts/
├── tests/
├── third_party/
└── third_party/
```

说明：

- `dexslide/` 是当前主 Python 包。
- `dexslide/vision/` 包含 direct ArUco、Marker Body 和底层视觉 Pose 跟踪代码。
- `robot_manipulation/assets/orca_hand/` 是 OrcaHand 的模型、retargeting 和配置资产。
- `robot_manipulation/assets/jaka/configs/` 保存 JAKA 的 Workspace Mapping 和 Payload Identification 结果。

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

## 全局通信配置

除机器人控制连接外，DexSlide 的手套串口和主相机默认参数统一由下面这个文件提供：

```text
assets/dexslide_communications.json
```

当前配置包含：

- 左手 joints 串口、稳定 `by-id` 路径、baud、stream mode、启动超时和最大 sample age。
- 左手 tactile 预留项。
- 右手 joints / tactile 预留项。
- 主相机 backend、RealSense serial、OpenCV source、稳定 `by-path` 路径和 `1280x720@30` 参数。

仓库中的 DexSlide viewer、标定和采集脚本都使用这份配置作为默认值，不再自行枚举 `ttyACM/ttyUSB` 或扫描其他 `/dev/video*`。CLI 参数只用于明确的临时覆盖；修改日常连接参数时应只改这一个 JSON。

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
assets/skeletons/photos2skeletons_dataset.json
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
assets/calibration/direct_aruco/table_aruco.yaml
assets/calibration/direct_aruco/left_tags2marker.json
assets/calibration/direct_aruco/left_marker2wrist.json
assets/calibration/direct_aruco/left_marker2wrist_dataset.json
```

当前默认口径：

- 字典：`DICT_4X4_50`
- 桌面参考 tag：`id=0`，边长 `120 mm`
- 目标 tag：默认按 `20 mm` 的 ArUco 黑边界做单码 PnP
- 手部载体：默认使用 `18` 面 marker body，几何关系由 `left_tags2marker.json` 提供

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

更完整的使用说明、参数解释、运行时交互和稳定性调参建议，见：

- [docs/direct_aruco_overlay_usage.md](docs/direct_aruco_overlay_usage.md)

这个脚本会在实时相机图像上直接画出：

- 桌面 tag 轮廓与坐标轴
- 目标 tag 轮廓与坐标轴
- 当前检测状态与相对位姿信息

它是检查“坐标系是否真正粘在实物上”的首选工具。

如果你现在使用的是手背 `18` 面 marker body，并且希望叠加 DexSlide 骨骼手，推荐直接运行：

```bash
python scripts/view_direct_aruco_overlay.py \
  --enable-hand-overlay \
  --hand left \
  --corner-refine-mode apriltag \
  --body-smoothing 0.35 \
  --body-outlier-threshold-mm 15 \
  --body-reprojection-threshold-px 4.0
```

这条命令会：

- 只在画面中显示各 marker 边框。
- 用联合 `PnP` 解算融合后的 marker body 坐标系。
- 把实时骨骼手刚性绑定到 marker body 上做 AR 投影。
- 把运行状态输出到终端，避免往画面堆文字。

### 3. 代码入口

- 跟踪器：`dexslide/vision/direct_aruco_tracker.py`
- 底层检测：`dexslide/vision/aruco_pose_tracker.py`
- 3D plot：`scripts/plot_aruco_relative_pose_3d.py`
- 相机 overlay：`scripts/view_direct_aruco_overlay.py`

### 已知约束

- 如果桌面参考 tag 完全出画，则 `world/table` 系下的目标位姿会中断。
- 快速运动时主要瓶颈仍可能来自曝光时间和 motion blur，而不只是检测参数。

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

## 测试

纯 Python 单元测试可用下面方式运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_direct_aruco_tracker.py -q
```

其余测试可按需在对应环境里执行。

## 后续方向

当前仓库后续重点仍然是 4 条：

- DexSlide 20 DOF 手套数据的稳定采集与重建。
- 更稳健的世界位姿方案，优先 direct ArUco / 多参考 tag。
- 机器人侧集成，包括 JAKA 控制链路。

## License

TBD
