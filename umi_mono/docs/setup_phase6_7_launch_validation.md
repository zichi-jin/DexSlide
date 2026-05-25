# Phase 6 + Phase 7 — Launch、配置、验证脚本（操作说明）

> 关联 tracker: `umi_mono/docs/online_tracking_implementation.md`
> 前置：Phase 0~5 全部通过
> Phase 6 把整条流水线封进一键 launch；Phase 7 写 8 个测试脚本（其中 2 个可立刻跑、6 个等 D435i）

---

## 0. Phase 6 + 7 新增/修改文件

```
umi_mono/ros2_ws/dexslide_slam_publisher/
├── CMakeLists.txt                              # +1 行: install(DIRECTORY launch ...)
└── launch/
    └── dexslide_slam_online.launch.py          # 新增 (TASK-032)

umi_mono/config/
└── RealSense_D435i_online.yaml                 # 新增 (TASK-033)，派生自 RealSense_D435i.yaml

umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/
└── realsense_online.cc                          # 修改 (TASK-036): +playback mode +vocab preflight

umi_mono/SLAM_readme_mono.txt                    # 追加 (TASK-034) 「## 7. 在线追踪」
umi_mono/CLAUDE.md                               # 追加 (TASK-035) Online tracking mode

umi_mono/tests/                                  # Phase 7 测试脚本
├── test_atlas_immutable.sh                      # TASK-023/041 (Phase 4 已建)
├── test_vocab_mismatch.sh                       # TASK-042 (PASS 已验证)
├── test_ate_vs_docker.py                        # TASK-037 (Python, numpy only)
├── benchmark_30min.sh                           # TASK-038 (含 inline Python stats)
├── test_headless.sh                             # TASK-039
├── test_memory_2h.sh                            # TASK-040
└── test_recovery_3s.sh                          # TASK-043 (半自动)
```

---

## 1. 一键 launch（最终用法）

```bash
# 主流程
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash

ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py \
  map_atlas:=/path/to/map_atlas.osa \
  exposure_us:=8000
```

**Launch 参数**：

| 参数 | 默认 | 含义 |
|------|------|------|
| `vocab` | `<fork>/Vocabulary/ORBvoc.txt` | ORB vocab 路径 |
| `settings` | `<umi_mono>/config/RealSense_D435i_online.yaml` | SLAM yaml |
| `map_atlas` | (空) | `.osa` atlas 路径；空则走在线建图 |
| `exposure_us` | 0 (auto) | D435i color sensor 曝光 |
| `pose_topic` | `/dexslide/slam/pose` | ROS2 topic |
| `zmq_endpoint` | `tcp://127.0.0.1:5555` | realsense_online ↔ bridge ZMQ |
| `log_to_file` | false | true 时 stdout 走 log |

启动后另一终端验证：
```bash
ros2 topic hz /dexslide/slam/pose       # ~30 Hz
ros2 topic echo /dexslide/slam/pose     # PoseStamped 实时流
ros2 run tf2_ros tf2_echo map camera_color_optical_frame
```

---

## 2. realsense_online 离线 playback 模式（TASK-036）

跑已录制的 mp4 + IMU json，不需要 D435i：
```bash
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v <vocab> \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml \
  -l <map_atlas.osa> \
  --playback_video <session>/demos/data_001/raw_video.mp4 \
  --playback_imu   <session>/demos/data_001/imu_data.json \
  --publisher stdout
```

- 录制和 IMU json 格式与离线 stage 02/03 一致（`gopro_slam` 输入格式）
- 同样跑 vocab MD5 / skew validation / ActivateLocalizationMode / SIGINT
- mp4 EOF → 自动 'playback complete' 退出 0

用途：**TASK-037 ATE vs Docker** 测试用它生成原生 trajectory，与 Docker batch 输出 ATE 对比。

---

## 3. 测试脚本快速参考

| 脚本 | 何时跑 | 期望输出 |
|------|--------|---------|
| `test_vocab_mismatch.sh` | **可立即跑** | `TASK-042 PASS: wrong vocab failed gracefully` |
| `test_atlas_immutable.sh <map.osa> [duration]` | 有 .osa 后 | `PASS: atlas SHA-256 unchanged (<hash>)` |
| `test_ate_vs_docker.py --recording <dir> --tolerance_cm 2` | 有 recording + .osa | `TASK-037 PASS: ATE = X cm` |
| `benchmark_30min.sh <map.osa> [minutes]` | D435i + atlas | `p99 inter-msg ≤ 50ms, p99 latency ≤ 33ms` + JSON |
| `test_headless.sh <map.osa>` | D435i + atlas | `TASK-039 PASS`，无 X11 错 |
| `test_memory_2h.sh <map.osa>` | D435i + atlas (耐心) | `TASK-040 PASS: RSS growth = X%` |
| `test_recovery_3s.sh <map.osa>` | D435i + 操作员 | 遮挡 3s 后 ≤ 5s 恢复 |

---

## 4. Phase 7 SLO 总表

| SLO | 阈值 | 验证 |
|-----|------|------|
| 帧率 | ≥ 30 Hz @ p99 | TASK-038 |
| 端到端延迟 | < 33 ms @ p99 | TASK-038 |
| 内存稳定性 | RSS 2h growth < 5% | TASK-040 |
| Atlas 不变性 | SHA-256 不变 | TASK-023/041 |
| 重定位恢复 | < 5s 从遮挡 | TASK-043 |
| 离线一致性 | ATE < 2 cm vs Docker | TASK-037 |
| 错误处理 | wrong vocab 5s 内退出 | TASK-042 ✓ |
| Headless | DISPLAY 缺失不报错 | TASK-039 |

---

## 5. 关键决策记录

### 5.1 为什么 realsense_online 加 playback mode 而不是单独写一个工具
- 复用 90%+ 的代码（CLI / System / pose dispatch / SIGINT / skew check）
- live vs playback 差异只在 frame source；分一个 binary 反而 maintenance 翻倍
- 缺点：binary 变大、CLI 多了 2 个 flag。Acceptable

### 5.2 为什么 launch 文件用 ExecuteProcess 启 realsense_online 而不是 Node
- realsense_online 不是 rclcpp 节点（它发 ZMQ，不是 ROS2 topic）
- ExecuteProcess 简单透明；ROS2 launch 体系也支持

### 5.3 为什么 D435i_online.yaml 派生自 D435i.yaml 而不是新建
- 离线 / 在线共享相机内参 + IMU 噪声参数
- 只差在 viewer / atlas load 字段
- 派生方便相机标定更新时手动同步两边（注释提醒）

### 5.4 为什么 ATE 测试用 Python 不用 C++
- Python + numpy 对轨迹处理更简洁
- 不需要重新编译；改阈值/对齐方式快速迭代
- 性能不是关键（一次性后处理）

### 5.5 为什么 vocab mismatch 加 preflight rejection
- Codex 试 fake vocab 时发现 ORB-SLAM3 在加载小文件会 segfault（未 catch 的 boost.serialization 异常）
- 我们的 preflight：< 1MB 或 > 200MB → 拒绝前置，避免 segfault
- 真实 ORBvoc.txt 是 145MB，远在范围内
- 副作用：将来用 ORB vocab 压缩版（< 1MB 的二进制 vocab）需要调整阈值

---

## 6. 失败时如何修复

### `ros2 launch ... map_atlas:=<path>` 报 'launch file not found'
- 没 `source ros2_ws/install/setup.bash`
- 或者 colcon 没构建过：`cd ros2_ws && colcon build --packages-select dexslide_slam_publisher`

### `ros2 topic hz` 显示 0
- 检查 realsense_online 是否真在跑：`ps -ef | grep realsense_online`
- 检查 publisher 是否 `zmq` 或 `ros2`：launch 自动设 zmq

### benchmark p99 > 33ms
- USB 3 接口确认
- 关闭桌面 effects / GPU compositing
- 锁 CPU governor 到 performance：`sudo cpupower frequency-set -g performance`
- 减小 ORB extractor.nFeatures（D435i_online.yaml 默认 1250，降到 1000 试试）

### TASK-040 RSS 增长 > 5%
- ORB-SLAM3 内部 KeyFrames pool 累积？不该，定位模式 mapping 关闭
- 我们的 ring buffer 泄漏？长跑前先跑 valgrind
- Pangolin 即使 bUseViewer=false 也吃内存？检查 `-DUSE_PANGOLIN=OFF` 编译开关

### test_recovery 失败 / 恢复 > 5s
- atlas 与场景不一致（旧地图、移动太多）
- 重新建图后再测
- 调大 `--max_lost_frames` 到 1800

---

## 7. 项目状态总览

**43/43 编码任务完成。** 等以下硬件/数据齐备后做最后一波 live 验证：

- ✅ Phase 0 (5) — host env
- ✅ Phase 1 (4) — fork build (gopro_slam ✓ + realsense_online 框架)
- ✅ Phase 2 (8) — realsense_online 核心 (CLI + RsCapture + System + 主循环 + NaN + SIGINT)
- ✅ Phase 3 (3) — publishers (stdout + ZMQ + ROS2 bridge)
- ✅ Phase 4 (5) — robustness (skew + MD5 + atlas + soft reset + exposure)
- ✅ Phase 5 (6) — Python SlamPoseSubscriber + 10 pytest 全过
- ✅ Phase 6 (4) — launch + yaml + docs
- ✅ Phase 7 (8) — playback + 7 测试脚本

**待 live 验证（D435i + .osa）**：TASK-005 (烟囱), TASK-008 (ATE), TASK-017 (live), TASK-023/041 (atlas SHA), TASK-037 (ATE), TASK-038 (30min), TASK-039 (headless), TASK-040 (2h), TASK-043 (recovery)
