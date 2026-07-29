# DexAlign

这个目录放的是 DexAlign 的离线标定脚本、数据结构和优化辅助模块。

## 状态说明

截至 `2026-07-22`：

1. 目录下现有脚本、采集工具和预览工具仍可继续使用。
2. 算法主线已经切到新的 **DexAlign 2.0 三步方案**。
3. 新方案的正式设计规格以 [docs/DexAlign.md](/home/jzq/MyJob/DexSlide/docs/DexAlign.md) 为准。
4. 后续代码改动应继续遵守这套三步流程与输出契约，不回退到旧版联合优化思路。

也就是说，这个 README 现在主要回答两件事：

1. 当前哪些脚本还能继续用于采集、预览和结果查看。
2. 当前 DexAlign 2.0 已经按哪条路线落地。

## DexAlign 2.0 的新路线

新方案不再把所有量一次性塞进同一个联合优化里，而是拆成 3 步：

1. 第一步：用 `S1` 学 palm 上 `wrist -> 五指 base` 的 `5` 条方向。
2. 第二步：用 `S2` 学 `20` 个关节角通道到模型自由度的线性映射 `w_q`。
3. 第三步：在方向固定后，再拟合全部长度和 `T_marker2hand` 的平移。

当前实现里，`S1/S2` 只是采集与存盘层面的区分。真正进入优化时，脚本会先把帧拼成统一池，再按每帧有没有关节角、有没有可用 wrist / palm 观测来决定能参与 `Step 1 / Step 2 / Step 3` 的哪一部分。

关键约定：

1. `R_marker2hand` 固定为 `left_marker2wrist.json` 中的 `initial_guess` 旋转。
2. 不再使用 `thumb_cmc` 插补，也不再保留那条拇指牵连自由度拟合线。
3. 采集人工开关保留，但交互目标简化为只保留 `SPACE` 一个开关键。
4. Step 3 可通过 `--optimize-thumb-base-rx` 额外放开一个严格的 x 轴旋转自由度。该自由度同时旋转 `thumb_base` 方向和后续 pp/ip/dp 局部平面，默认关闭。

详细数学定义、数据集 `S1/S2`、每步目标函数和输出格式，都写在 [docs/DexAlign.md](/home/jzq/MyJob/DexSlide/docs/DexAlign.md)。

## 当前目录内容

- [types.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/types.py)：内部数据结构定义。
- [skeleton_param.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/skeleton_param.py)：旧版 skeleton 参数展平与回填逻辑。后续切换 2.0 时需要重构。
- [io_utils.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/io_utils.py)：数据集读写、session 目录、`marker2hand` 资产读写；现已支持 `dataset_s1 / dataset_s2`。
- [collect_alignment_dataset.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/collect_alignment_dataset.py)：离线数据采集；现已支持 `--capture-kind {s1,s2}` 和仅 `SPACE` 人工开关。
- [objective.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/objective.py)：旧版联合残差与误差评估辅助；当前主要保留给诊断和测试。
- [pipeline_v2.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/pipeline_v2.py)：DexAlign 2.0 的 `Step 1 / Step 2 / Step 3` 核心实现。
- [optimize_alignment.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/optimize_alignment.py)：DexAlign 2.0 离线优化入口，负责读取 `S1/S2`、写回最终结果以及阶段性报告。
- [visualization.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/visualization.py)：优化前后误差图。
- [scripts/view_dexalign_capture_3d.py](/home/jzq/MyJob/DexSlide/scripts/view_dexalign_capture_3d.py)：采集前的独立 3D 观测预览工具。

## 当前仍可继续使用的脚本

### 1. 3D 观测预览

这个脚本继续保留，用来在真正采数据前确认：

1. marker pose 是否稳定
2. wrist / palm / 21 个 3D 点是否可信
3. table frame 切换是否正常

命令示例：

```bash
python scripts/view_dexalign_capture_3d.py \
  --marker-body-config assets/calibration/direct_aruco/left_tags2marker.json \
  --marker2hand-file assets/calibration/direct_aruco/left_marker2wrist.json
```

### 2. 离线数据采集

采集脚本继续保留，后续新方案仍会复用它来收集 `S1` 和 `S2`。

当前要求保留人工开关，不允许脚本一运行就无差别录制。

交互目标按新方案统一成：

- `SPACE`：采集开 / 关

相机窗口和 3D 窗口都接收 `SPACE`，不再要求鼠标必须聚焦到 3D plot。

通信参数不再由采集脚本自行猜测，唯一默认来源是：

```text
assets/dexslide_communications.json
```

S2 启动时必须从配置指定的手套串口收到首条完整 20-DOF raw frame。采集中若 raw line 为空、timestamp 无效、数值非有限或 sample age 超限，该帧会被拒绝，不会以静止 A-pose 写入数据集。

也就是说：

1. 先看实况画面。
2. 摆到想采的姿态。
3. 再用 `SPACE` 让这段数据进库或暂停。

命令示例：

```bash
python -m dexslide.calibration.dexalign.collect_alignment_dataset \
  --hand left \
  --capture-kind s2 \
  --session-id test_left_001 \
  --marker-body-config assets/calibration/direct_aruco/left_tags2marker.json \
  --glove-calib-file assets/calibration/glove_calibration.json \
  --target-kept-frames 120
```

### 3. DexAlign 2.0 优化入口

[optimize_alignment.py](/home/jzq/MyJob/DexSlide/dexslide/calibration/dexalign/optimize_alignment.py) 现在已经切到 DexAlign 2.0 三步流程。

推荐用法是直接给一个 session 目录，让它自动读取：

- `dataset_s1.npz`
- `dataset_s2.npz`

命令示例：

```bash
python -m dexslide.calibration.dexalign.optimize_alignment \
  --session-dir assets/calibration/dexalign/test_left_001 \
  --skeleton-file assets/skeletons/skeleton.json \
  --marker2hand-file assets/calibration/direct_aruco/left_marker2wrist.json \
  --optimize-thumb-base-rx
```

`--optimize-thumb-base-rx` 是实验性可选项。开启后，Step 1 和 Step 2 完全不变；Step 3 在原有 thumb base 长度之外新增一个 `thumb_base_rx`，默认 bounds 为 `±45°`。最终角度写入 `optimized_skeleton.json` 的 `palm.thumb_chain_rx_rad`，主线加载该 skeleton 后会复现同样的 thumb 运动链。

## 后续代码改动时的边界

当前实现和后续演进都应遵守这些边界：

1. 保留现有采集和预览链路，不重写整套设备接入。
2. 保留人工采集开关，交互收敛到仅 `SPACE`。
3. `R_marker2hand` 固定，不参与优化。
4. `thumb_cmc` 插补逻辑不再引入。
5. `optimize_alignment.py` 继续维持 `Step 1 / Step 2 / Step 3` 的分阶段结构。
6. 第二步待校参数维度是 `20 * 2`，不是 `15 * 2`。

## 输出路径契约

由于主线 `view_direct_aruco_overlay.py` 已经会指向 DexAlign 优化结果，因此后续代码切换到 2.0 时，**最终优化结果的输出路径、目录结构和命名不能改**。

需要保持兼容的目标形式是：

- `assets/calibration/dexalign/<session_id>/`
- 例如：`assets/calibration/dexalign/test_left_001/`

该目录下的结果文件命名也要保持兼容，至少包括：

- `optimized_skeleton.json`
- `optimized_marker2hand.json`
- `optimization_report.json`

主线脚本当前实际依赖的是这套“session 目录 + 固定文件名”接口，而不是旧版求解器内部细节。换句话说，后续就算改成 DexAlign 2.0 的三步流程，也必须继续产出例如：

- `assets/calibration/dexalign/test_left_001/optimized_skeleton.json`
- `assets/calibration/dexalign/test_left_001/optimized_marker2hand.json`
- `assets/calibration/dexalign/test_left_001/optimization_report.json`

也就是说，即使后续继续迭代 DexAlign 2.0 的内部细节：

1. session 目录仍然落在 `assets/calibration/dexalign/` 下。
2. 最终产物仍然写进同一个 session 目录。
3. 对外暴露给主线脚本消费的文件名不变。

内部中间产物、阶段性缓存、诊断文件可以新增，但不能破坏这套最终输出接口。

## 当前结论

这个目录当前已经以 [docs/DexAlign.md](/home/jzq/MyJob/DexSlide/docs/DexAlign.md) 里的新三步方案为主线，后续改动应继续沿这条路线前进。
