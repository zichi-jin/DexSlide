# DexSlide User Guide

本文档是 DexSlide 当前仓库的使用入口和能力边界说明。它回答四个问题：

1. 现在能用 DexSlide 做什么；
2. 应该从哪个命令或 Python API 进入；
3. 每条链路输入、输出什么数据；
4. 哪些脚本是主线、调试工具、硬件专用工具或尚未完成的占位入口。

本文档按当前工作区代码整理，最后核对日期为 `2026-08-04`。参数的最终事实来源始终是对应命令的 `--help`。

## 1. 先看结论：日常应该使用哪些入口

| 目的 | 推荐入口 | 状态 | 主要输入 | 主要输出 |
|---|---|---|---|---|
| 实时获取手套关节角、手腕位姿和视觉状态 | `python main.py stream` | 主线 | 相机、手套串口、streaming 配置 | `DexSlideSceneSample`；默认不滚动打印 |
| 查看主线 AR 画面 | `python main.py stream --show-overlay` | 主线 | 同上 | marker 边框、table/body 坐标系、手骨架 |
| 在 table frame 查看 Plot3D 骨架和位姿 | `python main.py stream --show-plot3d` | 主线 | 同上 | table、marker-body、wrist 坐标轴和多手骨架 |
| 录制可回读的数据集 | `python main.py stream --save-dir <目录>` | 主线 | 同上 | session metadata、配置快照、分块 NPZ、首张有效图像 |
| 在自己的 Python/遥操程序里消费实时数据 | `DexSlideScene` | 主线 API | streaming 配置 | Python sample 对象 |
| 单独检查 direct ArUco 和 marker body | `python scripts/view_direct_aruco_overlay.py` | 视觉诊断主入口 | 相机、ArUco 配置，可选手套串口 | OpenCV overlay 和诊断日志 |
| 兼容方式启动 Plot3D | `python main.py run` | 兼容别名 | streaming 配置 | 等价于 `stream --show-plot3d` |
| 查看串口原始 20 路数值 | `python main.py raw` | 调试 | 手套串口 | 终端数值 |
| 从 A4 照片提取初始手骨架 | `python main.py calibrate-skeleton` | 标定 | 手部照片 | `skeleton.json` 和逐图测量结果 |
| DexAlign 离线优化 | `python -m dexslide.calibration.dexalign.optimize_alignment` | 可用，离线 | S1/S2 NPZ 数据集 | 优化后的 skeleton、marker2hand、joint calibration 和报告 |
| DexAlign RGB-D 数据采集 | `collect_alignment_dataset` | RealSense 专用；当前 OpenCV 配置下不能直接启动 | RealSense RGB-D、手套、marker body | S1/S2 NPZ 数据集 |
| JAKA 手腕增量遥操 | `jaka_dexslide_incremental_teleop.py` | 机器人主入口 | `DexSlideScene`、workspace mapping、JAKA SDK | JAKA `servo_p` 增量命令；可选 AR 和录制 |

如果只是想取得“当前手在哪里、20 个关节是多少”，不要从 `scripts/` 目录找脚本，直接使用 `main.py stream` 或 `DexSlideScene`。

## 2. 系统能力边界

当前主线数据流是：

```text
手套串口 ── AngleStreamReader ──┐
                                ├── DexSlideScene ── Python API / AR / Plot3D / Recorder / Teleop
相机 ── ArUco + marker body ────┘
              │
              └── 固定 table marker 建立 world/table 坐标系
```

### 已经具备的能力

- 单手或配置驱动的多手关节流接入；每只手拥有独立串口、标定和时间戳。
- 20 DOF 关节角读取、ADC calibration 和 DexAlign 线性 joint calibration。
- 共享相机帧中的 table ArUco、手背多 marker body 检测和联合 PnP。
- `table/world -> hand/wrist` 4x4 位姿输出。
- table PnP 双解分支稳定和 SE(3) 位姿滤波。
- 关节流与相机帧按 wall-clock 时间做最近邻匹配，并报告 sample age。
- 无窗口实时 API、可选 AR、可选数据集录制、可选遥操可同时组合。
- 数据集配置快照、SHA-256 provenance、分块 NPZ 和 reader 回读。
- JAKA 手腕增量遥操、柔顺控制配置与 dry-run。

### 当前不应假定具备的能力

- `tactile` 仍是配置预留项，没有进入 `DexSlideSceneSample`。
- `robot_manipulation/scripts/dexslide_teleop_orcahand.py` 和 `dexslide_teleop_jaka.py` 是空文件，不是可用入口。
- 当前仓库没有统一的 OrcaHand 实机遥操主程序；已有的是模型、映射资产和 JAKA 手腕遥操。
- RGB-D MediaPipe 标定工具仍依赖 RealSense。当前 `assets/dexslide_communications.json` 把主相机配置成 OpenCV，因此部分 RealSense 工具在构造参数默认值时就会拒绝启动。
- 视觉 table marker 完全丢失时，系统不会把 camera frame 冒充 world frame；`table_valid=False`，手部 world pose 也会无效。
- 数据集 chunk 当前保存 table pose、手腕 pose 和关节数据，不保存完整视频流，也不保存每个手部 marker 的逐帧角点。

## 3. 配置文件及其职责

### `assets/dexslide_communications.json`

只描述“硬件从哪里接入”：

- 左右手 joints 串口、baud、输入模式、启动超时和最大 sample age；
- camera backend 和 OpenCV source；
- tactile 的预留位置。

日常更换串口或相机设备入口，优先修改这个文件。

### `assets/dexslide_streaming.json`

描述一次实时场景如何组装：

- camera 请求分辨率、FPS、FOURCC 和内参；
- table marker id 和 ArUco 配置；
- joint unit、joint mode、marker-body solver 和检测参数；
- 每只手使用的 skeleton、glove calibration、tags-to-marker、marker-to-hand 和 DexAlign joint calibration。

当前默认相机请求是 `1920x1192@60`、`MJPG`。相机返回什么画面由相机设备自身决定；主线不切割左右眼。

### 重要标定资产

| 文件 | 含义 |
|---|---|
| `assets/calibration/glove_calibration.json` | ADC/raw sensor 到基础关节角的 calibration |
| `assets/skeletons/skeleton.json` | 初始手部骨段和 palm 几何 |
| `assets/calibration/direct_aruco/d435i_intrinsic.json` | 当前主线使用的相机内参文件名；内容必须与实际图像模式匹配 |
| `assets/calibration/direct_aruco/table_aruco.yaml` | table ArUco 字典和实际边长；当前 id `0` 为 `0.075 m` |
| `assets/calibration/direct_aruco/left_tags2marker.json` | 手背各 marker 与 marker-body 几何关系 |
| `assets/calibration/dexalign/<session>/optimized_marker2hand.json` | marker body 到 hand/wrist 的最终外参 |
| `assets/calibration/dexalign/<session>/optimized_joint_calibration.json` | `scale * raw + bias` 的 20 通道 calibration |
| `assets/calibration/dexalign/<session>/optimized_skeleton.json` | DexAlign 优化后的个体化 skeleton |

## 4. 主入口：`main.py`

查看总入口：

```bash
python main.py --help
```

当前有四个子命令：`stream`、`run`、`raw`、`calibrate-skeleton`。

### 4.1 `stream`：实时场景主线

最小启动：

```bash
python main.py stream
```

默认行为：

- 启动相机和配置中的全部手套串口；
- 实时产生 scene samples；
- 默认使用 `joint_mode=dexalign`、`joint_unit=deg`；
- 默认不保存；
- 默认不开窗口；
- 默认不向终端连续打印 sample；
- 会输出一次性的启动信息，例如配置、相机模式、手 ID 和录制目录。

常用组合：

```bash
# AR 画面，默认包含手骨架
python main.py stream --show-overlay

# table frame 下的 Plot3D 骨架和位姿
python main.py stream --show-plot3d

# 同时打开 AR 和 Plot3D
python main.py stream --show-overlay --show-plot3d

# viewer 不画骨架，只保留位姿坐标轴
python main.py stream --show-plot3d --no-skeleton

# 录制数据
python main.py stream --save-dir assets/datasets --session-id demo_001

# 同时显示和录制
python main.py stream --show-overlay --save-dir assets/datasets

# 显式输出紧凑 JSONL
python main.py stream --stdout

# 使用 raw joint calibration 结果并以 rad 输出
python main.py stream --joint-mode raw --joint-unit rad

# 临时关闭主线中的全部位姿平滑层
python main.py stream --no-pose-filter

# 即使配置关闭，也可从 CLI 临时重新开启
python main.py stream --pose-filter

# 限时或限样本运行
python main.py stream --duration-sec 30
python main.py stream --max-samples 1000
```

参数：

| 参数 | 含义 |
|---|---|
| `--config` | streaming JSON，默认 `assets/dexslide_streaming.json` |
| `--joint-unit deg\|rad` | 对外关节角单位，默认由配置决定，目前为 `deg` |
| `--joint-mode dexalign\|raw` | 主输出使用 DexAlign calibration 还是基础 glove calibration |
| `--pose-filter` / `--no-pose-filter` | 覆盖配置，统一开启/关闭 table、marker-body 和 wrist 的全部位姿平滑层 |
| `--rate-hz` | 可选的软件输出/采样限频；不传时尽可能快运行 |
| `--duration-sec` | 到时自动结束 |
| `--max-samples` | 达到样本数后结束 |
| `--stdout` | 开启连续紧凑 JSONL；默认关闭 |
| `--no-stdout` | 显式关闭连续输出，和默认行为相同 |
| `--show-overlay` | 附加 AR viewer，不改变 sample 内容 |
| `--show-plot3d` | 附加 table-frame Plot3D viewer，不改变 sample 内容 |
| `--plot-fps` | Plot3D 最大刷新率，默认 `20`；不会改变采集频率 |
| `--plot-range-m` | Plot3D 初始坐标半径，默认 `0.45 m`，必要时自动扩大 |
| `--no-skeleton` | AR 和 Plot3D viewer 均不绘制重建手骨架 |
| `--save-dir` | 附加 recorder，指定 session 根目录 |
| `--session-id` | session 子目录名；省略时自动生成 |
| `--chunk-size` | 每个压缩 NPZ 的 scene sample 数，默认 `1000` |

`--stdout` 输出不是完整 API，只是方便 shell 管道的紧凑格式：

```json
{"timestamp":1785484438.727,"T_hand":[[1,0,0,0.10],[0,1,0,0.20],[0,0,1,0.40],[0,0,0,1]],"joint_angles":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

实际 `T_hand` 是完整 `4x4` 矩阵，`joint_angles` 是 20 个圆整整数；无效位姿时 `T_hand=null`。需要所有 validity、raw/dexalign 角度和 marker 信息时应使用 Python API。

### 4.2 `run`：Plot3D 兼容别名

```bash
python main.py run
```

`run` 现在等价于 `stream --show-plot3d`，内部使用同一个 `DexSlideScene`，相机、串口、skeleton、marker body 和 DexAlign calibration 全部来自 `--config`。它不再创建独立 camera、serial reader 或 `ArucoPoseTracker`。

可用参数包括 `--config`、`--joint-unit`、`--joint-mode`、`--rate-hz/--fps`、`--plot-fps`、`--plot-range-m` 和 `--no-skeleton`。新代码应优先使用 `stream --show-plot3d` 或 `DexSlidePlot3DViewer`。


### 4.3 `raw`：串口原始数据诊断

```bash
python main.py raw
```

该入口只读取 communications 配置中的手套串口并打印原始数据，不启动相机、视觉、骨架重建或 recorder。常用参数是 `--port` 和 `--baud`。

### 4.4 `calibrate-skeleton`：照片骨架初值

```bash
python main.py calibrate-skeleton --input-dir assets/photos
```

它从 A4 参考照片生成基础 `skeleton.json`，属于离线标定，不使用 `DexSlideScene`。DexAlign 优化后的 skeleton 应通过 `assets/dexslide_streaming.json` 配入主线。

## 5. Python 实时 API

主入口是 `dexslide.streaming.DexSlideScene`：

```python
from dexslide.streaming import DexSlideScene

with DexSlideScene.from_file("assets/dexslide_streaming.json") as scene:
    for sample in scene.samples(rate_hz=30):
        left = sample.hands["left"]
        controller.update(
            transform_table_hand=left.transform_table_hand,
            joint_angles=left.joint_angles,
            valid=left.valid,
        )
```

所有主线位姿平滑共用一个 master switch。配置入口是
`stream.pose_filter_enabled`，Python override 为：

```python
with DexSlideScene.from_file(
    "assets/dexslide_streaming.json",
    pose_filter_enabled=False,
) as scene:
    sample = scene.sample()
```

关闭时会绕过 table 的时序连续性选择与 SE(3) 平滑、marker-body tracker 的
One Euro 平滑，以及 wrist/body 的后置 robust median + SE(3) 低通。ArUco
正反面、相机朝向和右手系候选筛选属于位姿合法性判别，不受该开关影响。

`DexSlideSceneSample` 包含共享的 `timestamp`、`camera_T_table`、`table_valid`、图像尺寸和各手 sample。每个 `DexSlideHandSample` 包含：

- `transform_table_body`：marker ball 质心/body 位姿；
- `transform_table_hand`：应用 marker-to-hand calibration 后的 wrist 位姿；
- `joint_angles_raw`、`joint_angles_dexalign`、`joint_angles`；
- `pose_valid`、`joints_valid`、`joint_age_sec`；
- marker ID、角点和重投影误差。

可选消费者均不拥有采集设备，可以任意组合：

```python
from dexslide.recording import DexSlideRecorder
from dexslide.visualization import DexSlideARViewer, DexSlidePlot3DViewer

with DexSlideScene.from_file("assets/dexslide_streaming.json") as scene:
    ar = DexSlideARViewer(scene)
    plot3d = DexSlidePlot3DViewer(scene)
    try:
        for sample in scene.samples():
            if not ar.update(sample) or not plot3d.update(sample):
                break
    finally:
        ar.close()
        plot3d.close()
```

`DexSlidePlot3DViewer` 在 table frame 中显示 table/world、marker-body 和 wrist 坐标轴，并用校正后的 20 DOF 角度重建多手骨架。默认只以 `20 Hz` 刷新 Matplotlib，不会改变 Scene 的采集频率或 sample 内容。

## 6. 视觉与诊断工具

### 6.1 `view_direct_aruco_overlay.py`

```bash
python scripts/view_direct_aruco_overlay.py
```

用于检查相机、table marker、marker ball、body/wrist 坐标系和投影骨架。它是视觉诊断入口；业务程序仍应直接消费 `DexSlideScene`。

### 6.2 `plot_aruco_relative_pose_3d.py`

```bash
python scripts/plot_aruco_relative_pose_3d.py
```

这是 direct ArUco 的专项位姿/轨迹诊断工具，只显示 table、camera 和单个 tag 位姿，不重建 DexSlide marker body 或手骨架。日常查看完整手部 Plot3D 应使用 `main.py stream --show-plot3d`。

主要参数包括相机输入、table/target ArUco 文件、`--target-marker-ids`、`--plot-fps`、`--history-size`、`--axis-length` 和视角参数。

### 6.3 `camera_probe.py`

```bash
python -m dexslide.vision.camera.camera_probe --index 0 --width 1920 --height 1192 --fps 60
```

只打开原始相机流，报告设备实际返回的分辨率、FOURCC 和 FPS。它不切割双目画面，也不运行 ArUco。用于确认相机设备自主选择了哪个输出模式。

### 6.4 其他视觉/手模型脚本

| 脚本 | 状态 | 说明 |
|---|---|---|
| `scripts/glove_live_3d.py` | 兼容入口 | 转发到 `main.py stream --show-plot3d`，使用统一 Scene API |
| `scripts/glove_live_mano.py` | 实验性 | 将关节近似映射到 MANO mesh；需要 MANO assets，可静态或 `--live` |
| `dexslide/visualization/demo_20dof_matplotlib.py` | 演示 | 不依赖真实硬件的 20 DOF 可视化/交互示例 |
| `dexslide/calibration/tags2marker_check.py` | 当前损坏 | 当前 import 指向已迁移的 symbol，运行时会 `ImportError`；不要作为正式入口 |

## 7. 标定工作流

### 7.1 基础 skeleton

使用 `main.py calibrate-skeleton`，生成初始个体几何。这个结果随后可以作为 DexAlign 的初值。

### 7.2 DexAlign 2.0

设计说明见 `docs/DexAlign.md`。完整流程理论上包括：

1. 采集 S1：手掌几何；
2. 采集 S2：关节角和 3D 关键点；
3. 离线优化；
4. 把同一个 session 的 skeleton、marker2hand 和 joint calibration 配入 streaming。

#### 数据采集

入口：

```bash
python -m dexslide.calibration.dexalign.collect_alignment_dataset --capture-kind s1
python -m dexslide.calibration.dexalign.collect_alignment_dataset --capture-kind s2
```

主要参数包括 `--hand`、`--capture-kind`、`--session-id`、`--marker-body-config`、`--marker2hand-file`、`--glove-*`、`--camera-serial`、RGB-D 模式、关键点阈值和 marker-body 阈值。

当前限制：该入口依赖 RealSense RGB-D，并且当前主通信配置为 OpenCV camera，因此 parser 默认值调用 `resolve_realsense_serial()` 时会报错。它目前不是“换一个 CLI 参数就能运行”的状态；要使用它，需要切回带 RealSense serial 的 communications 配置或先修正该工具的默认值解析。

#### 离线优化

```bash
python -m dexslide.calibration.dexalign.optimize_alignment \
  --session-dir assets/calibration/dexalign/test_left_001
```

可用参数：

- `--session-dir`：包含 `dataset_s1.npz` 和 `dataset_s2.npz` 的 session；
- `--dataset-s1/--meta-s1`、`--dataset-s2/--meta-s2`：显式数据路径；
- `--skeleton-file`、`--marker2hand-file`：初值；
- `--output-dir`：输出目录；
- `--max-nfev-step2`、`--max-nfev-step3`：优化次数；
- `--optimize-thumb-base-rx`：额外开放 thumb base X 轴旋转；
- `--no-plots`：不生成诊断图。

典型输出：

- `optimized_skeleton.json`；
- `optimized_marker2hand.json`；
- `optimized_joint_calibration.json`；
- `optimization_report.json`；
- keypoint/frame error plots。

#### 数据预览

`scripts/view_dexalign_capture_3d.py` 也是 RealSense RGB-D 专用，并受当前 OpenCV communications 配置的相同限制。

### 7.3 marker-body 到 wrist 的 RGB-D 标定

入口：

```bash
python -m dexslide.calibration.marker_wrist_offset
```

它使用 RealSense 深度和 MediaPipe palm triangle 估计 body-to-wrist 外参，输出 marker-to-wrist JSON 和采样报告。当前同样属于 RealSense 专用工具，在 OpenCV 主相机配置下不能直接启动。

## 8. 机器人控制入口

机器人代码位于 `robot_manipulation/`，不要把它和通用 DexSlide API 混入 `dexslide/`。

### 8.1 JAKA DexSlide 增量遥操

推荐先 dry-run：

```bash
python robot_manipulation/scripts/jaka_dexslide_incremental_teleop.py --dry-run --show-overlay
```

连接实机前必须检查：

- JAKA IP、SDK 和机械臂状态；
- `workspace_axis_mapping.json` 中的 table-to-robot、mirror、scale 和 safe start；
- `safe_start_pose_mmdeg` 是否适合当前现场；
- table marker/world frame 是否稳定；
- marker body、DexAlign session 和 joint stream 是否有效。

数据链：

```text
DexSlideScene.transform_table_hand
  -> SE(3) 滤波
  -> 建立 glove/robot anchor
  -> table delta 映射为 robot delta
  -> deadband / step limit / jump rejection
  -> JAKA servo_p incremental command
```

关键参数：

- 安全：`--dry-run`、`--power-off-on-exit`、safe-start tolerances；
- 机器人：`--ip`、sensor/admittance 参数、`--servo-step-num`、`--loop-hz`；
- DexSlide：`--stream-config`、`--joint-unit`、`--joint-mode`、`--dexalign-session`；
- 映射：`--mapping-file`；
- 可选消费者：`--show-overlay`、`--save-dir`、`--session-id`；
- 视觉覆盖：camera/table/tags 和 marker-body solver 参数。

shell 包装入口：

```bash
robot_manipulation/scripts/run_jaka_dexslide_incremental_teleop.sh --dry-run
```

注意：包装脚本固定调用 `/usr/bin/python3`，如果依赖只安装在 Conda 环境中，应直接使用对应环境的 Python 运行 `.py` 文件。

### 8.2 JAKA 柔顺相关工具

| 入口 | 用途 |
|---|---|
| `jaka_compliant_teleop_ui.py` | 手动六维柔顺末端控制 UI，不读取 DexSlide |
| `jaka_set_admittance.py` | 配置 JAKA admittance/compliance；支持 `--dry-run` |
| `jaka_admittance_motion_trial.py` | payload/admittance/servo 增量轨迹综合测试 |

这些工具会改变机器人状态。第一次运行应阅读对应 `--help` 和 `robot_manipulation/work_logs/`，并优先使用 dry-run 或无动作模式。

### 8.3 OrcaHand 直接关节遥操

OrcaHand 手指遥操只使用 `DexSlideScene` 的 20 维 raw joint angles，单位固定为 `deg`，不使用 DexAlign joint calibration。

先分两阶段完成校准：

```bash
python -m robot_manipulation.scripts.dexslide_orcahand_calibrate \
  --stage dexslide

python -m robot_manipulation.scripts.dexslide_orcahand_calibrate \
  --stage orca
```

单独验证 OrcaHand：

```bash
python robot_manipulation/scripts/dexslide_teleop_orcahand.py --dry-run
```

最终单臂组合入口：

```bash
python robot_manipulation/scripts/dexslide_jaka_orcahand_teleop.py \
  --jaka-dry-run --orca-dry-run
```

组合入口只负责创建一个共享 `DexSlideScene` 并把 sample 分发给 JAKA 和 OrcaHand endpoint；映射、安全边界和硬件生命周期分别位于各自控制模块中。

## 9. 脚本状态索引

### 推荐主线

- `main.py stream`
- `dexslide.streaming.DexSlideScene`
- `dexslide.recording.DexSlideRecorder`
- `dexslide.recording.DexSlideDatasetReader`
- `scripts/view_direct_aruco_overlay.py`
- `robot_manipulation/scripts/jaka_dexslide_incremental_teleop.py`

### 兼容或专项工具

- `main.py run`
- `main.py raw`
- `scripts/glove_live_3d.py`
- `scripts/plot_aruco_relative_pose_3d.py`
- `scripts/glove_live_mano.py`
- `dexslide/vision/camera/camera_probe.py`
- `dexslide/calibration/plot_compare_skeletons.py`

### RealSense 专用、当前 OpenCV 配置下不能直接运行

- `dexslide.calibration.dexalign.collect_alignment_dataset`
- `scripts/view_dexalign_capture_3d.py`
- `dexslide.calibration.marker_wrist_offset`
- `dexslide.calibration.realsense_hand_preview`

### 当前不可用或占位

- `dexslide/calibration/tags2marker_check.py`：import 已失配；
- `robot_manipulation/scripts/dexslide_teleop_orcahand.py`：OrcaHand 直接关节遥操入口；
- `robot_manipulation/scripts/dexslide_teleop_jaka.py`：JAKA 增量遥操兼容别名。

## 10. 常见选择

### “我要把实时数据交给自己的控制器”

使用 `DexSlideScene`，不要解析终端文字：

```python
with DexSlideScene.from_file("assets/dexslide_streaming.json") as scene:
    for sample in scene.samples():
        hand = sample.hands["left"]
        controller.update(sample.timestamp, hand.transform_table_hand, hand.joint_angles, hand.valid)
```

### “我要确认新相机是否真的达到目标模式”

先运行 `camera_probe`，再运行 `main.py stream --show-overlay`。前者验证设备输出，后者验证完整视觉链路。

### “我要检查 marker ball 几何，不关心录制和遥操”

使用 `scripts/view_direct_aruco_overlay.py --diagnose-marker-body`。

### “我要收集训练或实验数据”

使用 `main.py stream --save-dir ...`。不要自行把终端 JSONL 当作完整数据集，因为它有意省略 validity、raw/dexalign 分项、配置 provenance 和多数视觉字段。

### “我要只读关节角，不启动相机”

使用 `live_listen()`、`AngleStreamReader` 或 `main.py raw`。`main.py run` 现在会启动完整 Scene（包括相机）。

### “我要给另一个机器人写遥操”

消费 `DexSlideSceneSample`，把机器人专用映射、安全限制和 SDK 代码放进 `robot_manipulation/<robot>/`。不要在 `DexSlideScene` 中加入机器人逻辑，也不要让机器人脚本自己再创建第二套相机/ArUco pipeline。

## 11. 目录职责

```text
dexslide/
├── calibration/       离线/在线标定算法和 DexAlign
├── kinematics/        手模型几何、SE(3) 数学和滤波
├── retargeting/       20 DOF 到 21 landmarks 的人体手模型接口
├── vision/
│   ├── camera/        相机设备配置、打开和原始帧读取
│   └── ...            ArUco、PnP、marker-body 和 scene vision
├── visualization/     viewer 和绘制消费者
├── streaming.py       实时场景公共 API
├── recording.py       数据集 writer/reader
├── live.py            仅关节串口的便捷 API
└── serial_angles.py   串口解析、calibration 和时间缓冲

scripts/               用户可运行的通用查看/诊断薄入口
robot_manipulation/    机器人 SDK、映射、资产和遥操脚本
assets/                配置、标定结果、模型和数据
tests/                 无硬件单元测试
docs/                  用户和算法文档
```

架构约束：

- 相机设备特有逻辑只应位于 `dexslide/vision/camera/`；
- ArUco 和 marker-body 视觉计算位于 `dexslide/vision/`；
- `streaming.py` 只编排实时数据，不承担相机特有裁剪或窗口绘制；
- viewer、recorder 和 teleop 都是 scene 的可选消费者；
- 机器人专用代码不得进入通用 `dexslide/` 包。

## 12. 自检命令

```bash
# 查看主线参数
python main.py stream --help

# 检查相机原始输出
python -m dexslide.vision.camera.camera_probe --help

# 检查 direct ArUco 参数
python scripts/view_direct_aruco_overlay.py --help

# 运行无硬件 streaming 单元测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_streaming.py -q
```

如果命令和本文档不一致，以当前命令的 `--help`、配置文件和对应 Python dataclass 为准，并同步修订本文档。
