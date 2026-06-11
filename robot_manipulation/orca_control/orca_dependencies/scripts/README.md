# ORCA Core — scripts 目录说明

本文件为 `scripts/` 目录下脚本的详细说明（中文），包含每个脚本的用途、常用参数、注意事项与示例用法。目的在于方便装配、调试与日常使用。若需要，我可以把部分内容同步到顶层 `README.md`。

---

## 使用说明（通用）
- 这些脚本通常在仓库根目录下运行（工作目录为项目根）。
- 大多数脚本接受可选的 `config_path` 参数来指定手模型的 `config.yaml`，默认为仓库内绑定的模型。
- 对于直接控制真实电机的脚本，请先确保串口权限和电源连接正确。Linux 用户若遇到权限问题可运行：

```bash
sudo usermod -aG dialout $USER
# 然后重新登录会话
```

---

### `check_motor.py`
- 位置：[scripts/check_motor.py](scripts/check_motor.py#L1)
- 作用：对单个电机进行循环读位与写位测试，方便实时调试与微调。可读取当前位置并写入小幅增量位置，用于观察电机响应与稳定性。
- 主要参数：`--port`（串口），`--baudrate`，`--motor_id`，`--wrist`（若为腕部使用位置控制模式），`--reverse`（反向微调）。
- 使用示例：
```
python scripts/check_motor.py --port /dev/ttyUSB0 --motor_id 2
```
- 注意：连接真实电机时会持续写入位置，请在需要时按 Ctrl+C 停止并断开。

### `common.py`
- 位置：[scripts/common.py](scripts/common.py#L1)
- 作用：脚本间共用的辅助函数与工具集合，包括：参数解析（`add_hand_arguments`）、创建手实例（`create_hand`，支持 `--mock` 注入 mock SDK）、连接/关闭手、文件路径解析与录制输出目录准备。
- 建议脚本通过 `add_hand_arguments()` 复用该模块，以实现参数一致性与 mock 测试方便性。

### `configure_motor_chain.py`
- 位置：[scripts/configure_motor_chain.py](scripts/configure_motor_chain.py#L1)
- 作用：交互式将出厂默认的电机（ID=1，波特率=57600）逐一配置为目标 ID 与目标波特率，并验证总线上的 ID 顺序与型号（XC330 / XC430）。适合组装新手或者需要批量重设 ID 的场景。
- 流程要点：预扫描已配置电机 → 按顺序提示连接下一个出厂电机 → 自动改 ID → 自动改波特率 → 验证链路完整性。
- 注意：该脚本会在检测到“错误配置”时退出并提示将电机恢复出厂设置；务必在有足够电源与串口权限时运行。

### `debug_overload.py`
- 位置：[scripts/debug_overload.py](scripts/debug_overload.py#L1)
- 作用：分步骤调试单个电机出现过载（overload）时的检测与恢复。包含读硬件错误码、发送 reboot、等待并重试读错误码、重设模式并重新启用扭矩。
- 主要参数：`--port`、`--baudrate`、`--motor_id`。
- 使用场景：仅在排查过载无法自动清除或想观察恢复时使用，脚本直接操作底层客户端接口。

### `jitter.py`
- 位置：[scripts/jitter.py](scripts/jitter.py#L1)
- 作用：对指定电机或一组电机执行小幅度、高频率的“抖动”动作（tendon seating），帮助去除线缆初始松弛，使张紧更均匀。
- 参数：`--motor`（可重复，选择多个电机）、`--duration`、`--amplitude`（角度/幅度）、`--frequency`、`--include-wrist`。
- 注意：运行前会启用扭矩并设置为 `current_based_position` 模式，请在安全场景下操作。

### `main_demo_abduction.py`
- 位置：[scripts/main_demo_abduction.py](scripts/main_demo_abduction.py#L1)
- 作用：演示手指外展、扇形展开与 spread_grasp 等姿态，循环播放预设命名姿态，适合作为功能演示或验收展示。
- 参数：`--cycles`、`--num-steps`、`--step-size`，并支持 `--mock` 通过 `common.add_hand_arguments`。

### `main_demo.py`
- 位置：[scripts/main_demo.py](scripts/main_demo.py#L1)
- 作用：基本演示脚本：open / power grasp / pinch 与 neutral 的循环演示，便于快速查看手的运动范围与握持形态。
- 参数同 `main_demo_abduction.py`。

### `overload_demo.py`
- 位置：[scripts/overload_demo.py](scripts/overload_demo.py#L1)
- 作用：将指定手指驱动到机械极限以故意触发过载（位置控制，无电流限制），用于验证电机的过载检测与恢复机制。
- 参数：`--finger`（thumb/index/...）、`--duration`（持续时间）、`config_path`。
- 警告：会使单指持续推向硬碰撞，务必保证测试环境安全且远离手指附近人员或敏感物体。

### `record_angles.py`
- 位置：[scripts/record_angles.py](scripts/record_angles.py#L1)
- 作用：手动离散记录关节位姿 waypoint（按回车保存每个快照），生成包含 `waypoints` 与 `metadata` 的 YAML 文件，供 `replay_angles.py` 回放。
- 交互：运行后禁用扭矩，手动摆放姿态，按 Enter 捕获，Ctrl+C 结束并写入文件。

### `record_continuous.py`
- 位置：[scripts/record_continuous.py](scripts/record_continuous.py#L1)
- 作用：以固定采样频率持续记录关节角度并保存为 YAML（每帧为一个关节值列表），用于后续精确回放或数据分析。
- 参数：`--frequency`、`--duration`、`--output-dir`。

### `replay_angles.py`
- 位置：[scripts/replay_angles.py](scripts/replay_angles.py#L1)
- 作用：回放 `record_angles.py` 生成的离散 waypoint 文件，支持插值过渡、循环播放与不同缓动函数（linear / ease_in_out）。
- 重要参数：`--replay-file`（必需）、`--step-time`、`--transition-time`、`--mode`、`--loop`。

### `replay_continuous.py`
- 位置：[scripts/replay_continuous.py](scripts/replay_continuous.py#L1)
- 作用：按录制时的采样频率回放 `record_continuous.py` 的逐帧数据，严格按时间戳间隔发送关节命令以复现动作流。
- 参数：`--replay-file`（必需）。

### `setup.py`
- 位置：[scripts/setup.py](scripts/setup.py#L1)
- 作用：全流程装配/调试脚本，串联多轮 `tension` → `calibrate` → motion test → 验证（默认 3 轮），并在每步提供交互提示与跳过选项。
- 建议用于新装配的 ORCA Hand 或完整复位后的验证流程，运行时应有助力人员配合（张力调节等）。

### `slider_joint.py`
- 位置：[scripts/slider_joint.py](scripts/slider_joint.py#L1)
- 作用：基于 Tkinter 的 GUI，按关节显示滑条，用于单关节的实时控制；包含启用/禁用扭矩按钮，并实时将滑条值写入对应关节。
- 适合需要视觉/交互调试的场景，运行在带 GUI 的机器上（如笔记本），需确保 X11/Wayland 可用。

### `slider_motor.py`
- 位置：[scripts/slider_motor.py](scripts/slider_motor.py#L1)
- 作用：基于 Tkinter 的电机级滑条界面，每个滑条以当前编码器位置为中心、微幅范围用于精细调整电机编码器位置（非常适合微调张力或微调编码器零点）。
- 注意：滑动会直接调用 `_set_motor_pos` 写位置，使用时请小心并在安全环境下操作。

### `stress_test.py`
- 位置：[scripts/stress_test.py](scripts/stress_test.py#L1)
- 作用：循环开/合手并同时监控电机温度，终端以彩色表格显示各电机温度与占比，支持设置循环次数或无限循环，用于热稳态与过热检测。
- 参数：`--cycles`（0 表示无限），脚本内有保守的温度阈值提示（默认 70°C）。

### `test_motor_latency.py`
- 位置：[scripts/test_motor_latency.py](scripts/test_motor_latency.py#L1)
- 作用：对单电机或多电机执行一系列延迟基准测试（读、写、读写往返、顺序读、sync read、sync write、全手控制循环等），支持 Dynamixel 与 Feetech，两者接口均已实现。
- 常用参数：`--iterations/-n`、`--baudrate`、`--amplitude`、`--optimize`（对 Dynamixel 应用 Return Delay=0、Status Return=READ_ONLY、启用 fast sync read）。
- 输出为统计数据（均值/最小/最大/95% 等），适合评估通信性能与决定控制循环频率。

### `test_overload.py`
- 位置：[scripts/test_overload.py](scripts/test_overload.py#L1)
- 作用：缓慢递增单电机目标位置，物理阻挡手指以触发过载，观察硬件错误标志与自动恢复流程。用于验证过载检测的灵敏度与恢复路径。

### `zero.py`
- 位置：[scripts/zero.py](scripts/zero.py#L1)
- 作用：将所有关节移动到零位（`set_zero_position`），常用于重置姿态或做初始对齐。支持 `--force-calibrate`、`--num-steps`、`--step-size` 等参数。

---

如果你希望我：
- 把这份 `scripts/README.md` 添加到 Git（创建 commit），我可以代为生成 commit 命令并执行；
- 或者把其中某几个脚本扩展为更详细的子文档（例如增加常见故障排查、示例输出、图示），我也可以继续补充。
