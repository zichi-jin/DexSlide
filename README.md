# dexslide

dexslide 是一款开源外骨骼数据手套，用于采集人手 20 个关节角度并实时重建手部姿态。当前仓库只保留上位机 Python 标定、可视化、运动学和实验脚本；固件、PCB 资料、芯片资料和 pinout 文档已独立到 `dexslide_infra`。

硬件模型、PCB、BOM 和装配资料后续会统一发布在：<https://dexslide.org/hardware>。

## 使用流程

### 1. 准备硬件

组装 dexslide 机械结构、编码器、ADS1115 ADC 板和 STM32F103 控制板。当前主线下位机工程已独立到 `dexslide_infra` 仓库，负责读取 5 个 ADS1115 上的 20 路编码器数据，并通过 USB CDC 串口输出。

固件编译与烧录：

```bash
cd ../dexslide_infra
cmake --preset Debug
cmake --build build/Debug
st-flash write build/Debug/dexslide_stm32.bin 0x08000000
```

这次仓库改名只调整工程目录、CMake project 名称和输出文件名，不改变固件业务逻辑；如果你已经烧录过可用固件，不需要因为重命名重新烧写。

### 2. 安装上位机环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 拍摄裸手骨骼照片

把手平放在 A4 纸上拍摄多张 RGB 图片，建议包含不同角度但保持手掌和 A4 纸尽量共面。默认图片目录是：

```text
assets/photos/
```

### 4. 提取 skeleton

运行离线骨骼提取。程序会让你按 TL、TR、BR、BL 顺序点击 A4 四角，然后用 MediaPipe 提取 21 个手部关键点，并把像素坐标映射到 A4 平面毫米坐标。

```bash
python main.py calibrate-skeleton
```

默认输出：

```text
assets/skeletons/skeleton.json
assets/skeletons/offline_bone_mm_results.json
```

常用参数：

```bash
python main.py calibrate-skeleton --show-debug
python main.py calibrate-skeleton --reuse-a4
python main.py calibrate-skeleton --input-dir assets/photos --skeleton-aggregate median
```

提取完成后会自动打开 2D 骨架检查图。也可以手动对比 `assets/skeletons/` 下所有 skeleton：

```bash
python -m dexslide.calibration.plot_compare_skeletons --show
```

### 5. 标定下位机 ADC 到关节角

拍完骨骼后，还需要告诉上位机“串口 ADC 数值如何换算成角度”。这个过程由 `dexslide_infra` 仓库里的脚本完成，结果写回本仓库的上位机资产目录。

1. 逐关节标定：

```bash
cd ../dexslide_infra
python scripts/glove_calibrate.py --port /dev/ttyACM0 --out ../DexSlide/assets/calibration/glove_calibration.json
```

脚本会按 20 个关节顺序引导你采集 0 度和参考角度。大多数关节使用 90 度参考；部分 `MCP_back` 可以输入自定义参考角，或在机械动作不方便时选择默认比例。标定结果默认保存为：

```text
assets/calibration/glove_calibration.json
```

2. 检查原始串口数据：

```bash
python scripts/ads_live_monitor.py --port /dev/ttyACM0
```

3. 检查映射后的角度是否合理：

```bash
python scripts/ads_live_monitor.py --port /dev/ttyACM0 --angles --calib-file ../DexSlide/assets/calibration/glove_calibration.json
```

如果直接运行 `main.py run` 但找不到 `glove_calibration.json`，程序会提示先执行这一步。

### 6. 实时 3D 重建

```bash
python main.py run --port /dev/ttyACM0
```

`main.py run` 会读取：

```text
assets/skeletons/skeleton.json
assets/calibration/glove_calibration.json
```

并启动 Matplotlib 3D 实时重建。旧入口仍可用：

```bash
python scripts/glove_live_3d.py --port /dev/ttyACM0
```

如果你已经用 `ads_live_monitor.py --angles` 产生角度文本流，也可以使用：

```bash
python main.py run --port /dev/ttyACM0 --mode angles
```

### 7. 离线动画与可视化

20 DOF 轨迹动画脚本已经放到 visualization 模块：

```bash
python -m dexslide.visualization.demo_20dof_matplotlib
python -m dexslide.visualization.demo_20dof_matplotlib --trajectory-csv your_motion.csv --degrees
```

`dexslide/calibration/` 只保留骨骼计算和骨骼结果检查相关功能；Leap Motion 相机相关代码已删除，因为 Leap 相机硬件路线已经废弃。

## `main.py` 命令

```text
python main.py calibrate-skeleton
  --input-dir INPUT_DIR
  --results-file RESULTS_FILE
  --skeleton-file SKELETON_FILE
  --min-conf MIN_CONF
  --reuse-a4
  --show-debug
  --skeleton-aggregate {median,mean}
  --no-skeleton-plot

python main.py run
  --port PORT
  --baud BAUD
  --mode {raw,angles}
  --skeleton-file SKELETON_FILE
  --calib-file CALIB_FILE
  --hand {auto,left,right}
  --fps FPS

python main.py raw
  --port PORT
```

`calibrate-zero` 已从 `main.py` 移除。当前角度零位和比例关系由 `dexslide_infra/scripts/glove_calibrate.py` 统一完成。

## 文件结构

```text
.
├── README.md
├── main.py
├── requirements.txt
├── assets/
│   ├── calibration/
│   │   └── glove_calibration.json
│   ├── photos/
│   ├── skeletons/
│   │   ├── skeleton.json
│   │   └── offline_bone_mm_results.json
│   └── mano/
├── dexslide/
│   ├── calibration/
│   ├── kinematics/
│   ├── visualization/
│   ├── paths.py
│   ├── serial_angles.py
│   └── serial_reader.py
└── scripts/
    ├── glove_live_3d.py
    └── glove_live_mano.py
```

## 后续计划

后续 dexslide 会加入 4 个方向：

- 触觉：在 `dexslide_infra` 中整理和烧写触觉采样代码，配合 System B 硬件。
- 机器人输入端：使用头戴相机持续追踪手套腕部位姿，同时读取 dexslide 的 20 个关节角。
- 中间层：保存腕部位姿，并使用成熟 retarget 方案把手指姿态映射到机器人灵巧手。
- 输出端：驱动一只或一对 JAKA S5 机械臂和 OrcaHand 灵巧手。

## License

TBD
