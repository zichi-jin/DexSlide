# DexSlide 项目摘要（提供给外部 AI / RAL 论文协作者）

> 本文档是对当前工作区代码的事实性说明，重点解释“系统已经实现了什么、数据如何流动、哪些部分可作为论文系统描述、哪些结论仍不能直接宣称”。硬件机构的学术创新点由项目成员另行定义；本文不替代机械设计、实验统计或论文贡献论证。

## 1. 一句话概括

DexSlide 是一个面向机器人手部遥操作的 20-DoF 外骨骼数据手套系统。手套通过编码器测量佩戴者的 20 个手指关节相关自由度；手背刚性安装的多 ArUco marker body 提供手部 6-DoF 位姿锚点；上位机用 Python 将串口关节流、相机 ArUco 位姿、个体化手部 skeleton 和机器人映射组合成统一的实时 scene API，并可驱动 JAKA 机械臂腕部和 OrcaHand 灵巧手。

主仓库 `/home/jzq/MyJob/DexSlide` 负责上位机算法与遥操；下位机仓库 `/home/jzq/MyJob/DexSlide_Infra` 负责 STM32 固件、ADC/USB CDC 采集、校准脚本和触觉数据链路。

## 2. 系统分层

| 层级 | 主要实现 | 输入 / 输出 |
|---|---|---|
| 外骨骼关节采集 | `DexSlide_Infra/firmware/joints_f103` | 20 路电阻式编码器 → ADC counts → USB CDC ASCII 流 |
| 触觉采集 | `DexSlide_Infra/firmware/tactile_h562` | 11 路 UART 电子皮肤 → DMA → 带 CRC 的 USB CDC 聚合记录 |
| 串口与配置 | `dexslide/serial_angles.py`、`dexslide/live.py`、`dexslide/communications.py` | raw counts 或角度文本 → 20 维角度向量（deg/rad） |
| 初始 hand model | `dexslide/calibration/offline_a4_bone_mm.py`、`assets/skeletons/` | A4 参考照片 + MediaPipe → mm 单位 skeleton |
| 视觉位姿 | `dexslide/vision/` | 相机图像 + ArUco 内参/几何 → table/world 到 hand/wrist 的 SE(3) |
| DexAlign | `dexslide/calibration/dexalign/` | 同步 marker pose、20 角编码器、RGB-D/MediaPipe 3D 点 → 个体化 skeleton/外参/角度映射 |
| 实时场景 | `dexslide/streaming.py`、`dexslide/recording.py` | 统一 `DexSlideSceneSample`，供 AR、录制和 teleop 消费 |
| 机器人遥操 | `robot_manipulation/` | JAKA wrist incremental teleop；OrcaHand direct joint mapping |

## 3. 关节采集硬件与通信

### 3.1 F103 关节板

- MCU：`STM32F103RCT6`。
- ADC：5 个 `ADS1115`，每个 4 个 single-ended channel，共 20 路。
- 总线/地址映射：
  - `I2C1@0x48`：thumb 的 4 路；
  - `I2C1@0x49`：index 的 4 路；
  - `I2C2@0x48`：middle 的 4 路；
  - `I2C2@0x49`：ring 的 4 路；
  - `I2C2@0x4B`：pinky 的 4 路。
- 每个 ADS1115 通道配置为单次转换、`±4.096 V` PGA、`860 SPS`；固件轮询有效器件，并约每 `20 ms` 输出一行。
- 当前固件输出格式是可读的 ASCII 行，例如：

  ```text
  I2C1@0x48[A0:...,A1:...,A2:...,A3:...] | I2C1@0x49[...] | ...
  ```

- USB 端点使用 CDC，默认主机脚本以 `115200` 打开；实际 CDC 传输不依赖传统 UART 波特率，但脚本保留该参数作为端口配置约定。
- 固件具有 ADS1115 地址发现、掉线重发现、I2C 错误报告和总线复位逻辑。

### 3.2 ADC 到角度

`DexSlide_Infra/scripts/glove_calibrate.py` 通过“0° + 参考角”两点采样，为 20 个通道生成 `offset/d0`、`rate`、`k`、`b` 等参数。主机端在 `dexslide/serial_angles.py` 中使用：

```text
angle_deg = angle0 + normalize_delta(raw_count - offset) / rate
```

其中编码器一圈约 `24370 counts`，跨越半圈时做环绕修正。角度语义按 5 根手指 × 4 通道组织：`DIP, PIP, MCP_front, MCP_back`。主机可将基础角度再应用 DexAlign 的 20 维 affine 校正：

```text
q_dexalign = joint_scale * q_raw + joint_bias_rad
```

注意：仓库中 `dexslide/serial_reader.py` 仍保留一套旧的二进制 `0xAA55 + 20x uint16 + XOR + 0x0D` 解析器；当前 F103 固件和主线 `AngleStreamReader` 使用的是 ASCII raw/angle 行。这是历史兼容代码，不能把二进制格式误写成当前主通信协议。

## 4. 触觉下位机链路

`firmware/tactile_h562` 使用 `STM32H562VGT6`，将 11 个连接器（`CN1...CN11`）的传感器 UART 接收链路统一聚合到 USB CDC：

- 11 路 UART 均使用 DMA，单路 `1024 B` 环形/双半缓冲；固件按完整 DMA half 形成数据记录。
- MCU→PC 记录头以 `0xD3 0x5C` 同步，包含 version、record type、port id、payload length，尾部为 `CRC16-CCITT`。
- 原始传感器帧仍保留 `0xA5 ... 0x5A` 包络，并使用 `CRC16-Modbus`。
- 主机 `scripts/tactile_api.py` 可解出 ADC、normal force、shear-x、shear-y 矩阵；当前电子皮肤测量帧支持 `ADC_DATA (0x20)`、`FORCE_DATA (0x21)`、`COMBINED_DATA (0x22)`。
- 每个测量包含时间戳、阵列行列数和 sequence；系统状态包每秒提供每路 overflow/UART/DMA 错误、USB 队列水位和丢包计数。
- `tactile_h562_raw` 是 CN2/USART2 的透明 USB-UART 调试桥，不是 11 路聚合格式。

当前主仓库 `DexSlideSceneSample` 的 tactile 字段仍是配置预留项，触觉尚未进入主线实时遥操闭环。论文中可以把它描述为独立的多模态扩展链路，不能声称现有 scene API 已同步融合触觉。

## 5. 初始手部模型与 20-DoF FK

离线 skeleton 提取使用 A4 尺寸参考和 MediaPipe 关键点，输出 `assets/skeletons/skeleton.json`。运行时模型位于 `dexslide/kinematics/live_hand.py`：

- 内部使用 21 个手部节点；单位主要是 `mm`。
- palm 包含 wrist、thumb base、index/middle/ring/pinky MCP 等基点。
- thumb 为 `metacarpal/proximal/distal` 三段；其他四指各为 `proximal/middle/distal` 三段。
- 20 个外骨骼通道驱动 FK。thumb 的 MCP 前后两个自由度采用特殊 frame 构造；四指 MCP 的 zero convention 和 handedness 在代码中显式处理。
- 初始 skeleton 来自照片几何，不等同于最终个体化模型。

## 6. ArUco 与手部位姿

系统以固定桌面参考 marker 建立 `table/world` frame。主线视觉链路为：

1. 相机检测 table marker 和手背 marker body 的多个 marker；
2. 使用相机内参和 marker 实际边长做 PnP；
3. 对平面 marker 处理 IPPE 双解并进行时序分支稳定；
4. 对手背多个 marker 进行联合几何求解/重投影筛选/平滑，得到 marker body pose；
5. 用标定的 `body/marker → wrist/hand` 固定变换输出 `T_table_hand (4×4)`。

独立的 `dexslide/vision/direct_aruco_tracker.py` 也实现“同一帧 table marker + target marker”的直接相对位姿：

```text
T_table_target = inv(T_camera_table) @ T_camera_target
```

如果 table marker 丢失，系统将 `table_valid=False`，不会伪造 world pose。快速运动时主要风险是曝光和 motion blur；marker 数量过少时联合 PnP 的姿态质量会下降。

## 7. DexAlign：离线个体化标定

DexAlign 的研究问题是：给定同步的 20 维编码器角度 `q_t`、marker pose `^C T_M` 和 RGB-D/MediaPipe 21 个 3D 点 `y_t`，估计佩戴者的 skeleton 参数和 marker-to-hand 外参，使 FK 预测与视觉观测一致。

基本链路为：

```text
x_hat_t = FK(q_t; theta)
p_hat_t = ^C T_M · ^M T_H · x_hat_t
```

### DexAlign 1.0

对 skeleton 几何和 marker-to-hand 6-DoF 外参做多帧 robust nonlinear least-squares 联合优化，残差是带 confidence/depth mask/关键点类别权重的 3D 点位误差。

### DexAlign 2.0

将耦合问题拆为三个阶段：

- S1：固定 marker-to-hand 旋转，用多帧 palm/wrist 观测估计 5 条 palm base 射线方向；
- S2：对 20 个通道估计 `q_hat_j = a_j q_j + b_j`，主要比较 15 条骨段方向；
- S3：固定前两阶段结果，优化骨长和 marker-to-hand 平移，同时使用点位残差和 wrist 平移样本残差。

S1/S2 是采集语义，不应被当成互斥优化流水线；实际代码按每帧是否有 q、wrist、palm、marker 等模态做 gating。

### 当前最重要的科学限制

DexAlign 的主要瓶颈被代码和实验记录归纳为 observation-model mismatch：FK 节点更接近内部关节中心，而 MediaPipe + depth 更接近手表面点。弯曲时表面点相对旋转中心的偏移会随姿态变化，静态骨长 FK 无法完全解释。marker 固定在手背外骨骼上，而 hand root/wrist 是手部内部语义，二者之间的常量外参容易吸收视觉系统误差。

已有诊断（见 `docs/DexAlign_Academic_Summary.md` 和 `docs/DexAlign_AI_Handoff.md`）：约 600 帧数据中，21 点平均有效约 `20.92/21`，marker 平均重投影误差约 `0.84 px`；平均点位误差可从约 `36.89 mm` 降到 `19.09 mm`，但仍不够好；允许每帧独立自由刚体时可到约 `12.37 mm`，说明优化器在工作，但“常量外参 + 当前观测语义”未解释真实误差。

因此，其他 AI 在讨论论文方法时应优先区分：

- 硬件测量创新（外骨骼、模块化、20-DoF、marker/tactile 集成）；
- 工程系统贡献（多模态同步、统一 scene API、可记录/可回放、机器人端映射）；
- 仍在研究的建模问题（表面 landmark 与内部 FK 节点的语义错配、外参漂移和先验约束）。

## 8. 实时 scene API 与数据流

推荐入口是 `python main.py stream` 或 `DexSlideScene.from_file(...)`，而不是旧 viewer。`DexSlideScene` 同时拥有相机和每只手的独立关节 reader，输出不可变的 `DexSlideSceneSample`：

- scene 级：timestamp、`camera_T_table`、table validity、图像尺寸、table marker 角点；
- hand 级：`transform_table_hand`、raw/DexAlign 两套 20 维角度、pose/joint timestamp、joint age、marker ids、重投影误差和 validity。

关节样本放在带时间戳的环形缓冲中，场景读取时按 wall-clock 最近邻匹配，并显式报告 sample age。Recorder 保存 session metadata、配置快照、SHA-256 provenance、分块 NPZ 和首张有效图像，但当前不保存完整视频，也不保存每帧所有 marker 角点的完整历史。

这种架构把 acquisition、recording、visualization 和 teleoperation 解耦：同一份 sample 可以同时被 AR viewer、数据集录制器和机器人 endpoint 消费。

## 9. 机器人遥操路径

### JAKA 腕部

`robot_manipulation/JAKA_control/teleop_endpoint.py` 只消费 hand pose：

- marker/table pose 经过 `GlovePoseFilter`；
- 首次稳定跟踪建立 glove anchor 与 robot anchor；
- 使用 workspace axis mapping 将 table frame 的平移/旋转增量映射到机器人坐标；
- 有 translation/rotation deadband、步长限幅、安全初始姿态检查和 tracking lost 暂停策略；
- JAKA 端启用 torque sensor、admittance/compliance 和 `servo_p` 增量命令；支持 dry-run。

### OrcaHand 手指

`robot_manipulation/orca_control/teleop_endpoint.py` 使用 `DirectJointMapper`：

- 20 个 DexSlide 源关节通过配置文件做 degree-space affine mapping；
- OrcaHand 目标为 16 个非 wrist 关节（thumb 3 个、四指各 3 个加 abduction 等）；
- 每个目标有 target/motor limit，超限会记录 clip reason；
- `OrcaHandAdapter` 负责连接真实硬件或 mock；joint stale 时不发命令。

组合入口是 `robot_manipulation/scripts/dexslide_jaka_orcahand_teleop.py`，共享一个 `DexSlideTeleopRuntime`，以约 `20 Hz` 将同一 scene sample 分发给 JAKA 和 OrcaHand endpoint。

## 10. 代码现状与论文措辞边界

可以安全描述的事实：

- 20-DoF 外骨骼角度采集和主机端校准链路已经实现；
- 多 marker body + table marker 的视觉位姿链路已经实现；
- 主仓库提供统一的多手 scene、录制、AR 和机器人 endpoint 抽象；
- JAKA 腕部增量遥操和 OrcaHand 直接关节映射都有代码、配置和 dry-run 路径；
- 触觉下位机具备 11 路 DMA、USB CDC 聚合、双层 CRC、状态诊断和主机端矩阵解码。

需要谨慎或明确限定的说法：

- tactile 尚未进入主仓库 `DexSlideSceneSample` 的实时融合闭环；
- MediaPipe/RGB-D 的 DexAlign 标定数据采集仍有 RealSense 专用入口，不能把所有 OpenCV 配置描述成已完成 RGB-D 标定系统；
- 旧脚本 `robot_manipulation/scripts/dexslide_teleop_orcahand.py`、`dexslide_teleop_jaka.py` 曾为空/兼容占位，实际组合入口是新的 `dexslide_jaka_orcahand_teleop.py`；
- `serial_reader.py` 的二进制帧格式是历史兼容实现，不代表当前 F103 ASCII 输出；
- 目前仓库没有足够证据证明 DexAlign 已达到高精度 anatomical joint-center 标定，论文应报告误差、同步、遮挡、marker 可见性和观测语义限制。

## 11. 建议其他 AI 优先回答的问题

1. 如何将外骨骼的 20 个机械自由度、模块化硬件和 marker/tactile 接口转化为可验证的 RAL contribution，而不是仅列功能？
2. 触觉链路如何与现有 scene timestamp、validity、recording schema 对齐，才能形成真正的多模态遥操实验？
3. DexAlign 是否应从“FK 节点拟合 MediaPipe 点”改为 surface-aware observation model、joint-center latent model 或带解剖先验的优化？
4. 实验应如何分别评估：关节角准确率、hand pose 误差、机器人末端误差、闭环延迟、丢帧/过期样本、marker 遮挡和触觉吞吐？
5. 哪些模块应在论文中作为系统设计，哪些仅作为实现细节或 future work？

## 12. 关键文件索引

- 总入口和命令：`main.py`、`docs/USER_GUIDE.md`、`README.md`
- 串口/角度：`dexslide/serial_angles.py`、`dexslide/live.py`、`dexslide/communications.py`
- 实时场景：`dexslide/streaming.py`、`dexslide/recording.py`
- FK：`dexslide/kinematics/live_hand.py`
- ArUco/marker body：`dexslide/vision/aruco_pose_tracker.py`、`dexslide/vision/direct_aruco_tracker.py`、`dexslide/vision/scene_backend.py`
- DexAlign：`dexslide/calibration/dexalign/`、`docs/DexAlign_Academic_Summary.md`、`docs/DexAlign_AI_Handoff.md`
- 机器人：`robot_manipulation/teleop/`、`robot_manipulation/JAKA_control/`、`robot_manipulation/orca_control/`
- 关节下位机：`DexSlide_Infra/firmware/joints_f103/Core/Src/main.c`、`DexSlide_Infra/scripts/glove_calibrate.py`、`DexSlide_Infra/scripts/ads_live_monitor.py`
- 触觉下位机：`DexSlide_Infra/firmware/tactile_h562/Core/Src/tactile_stream.c`、`DexSlide_Infra/firmware/tactile_h562/Core/Inc/tactile_stream_protocol.h`、`DexSlide_Infra/scripts/tactile_api.py`

