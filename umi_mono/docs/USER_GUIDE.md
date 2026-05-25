# DexSlide SLAM 在线追踪系统 — 用户指南

> 文档版本：2026-05-19 · 适用 commit 之后
> 受众：第一次接触这套系统的工程师 / 新机器部署
> 相关文档：
> - `docs/online_tracking_research.md` — 方案研究报告（为什么选 ORB-SLAM3 fork + ROS2 桥架构）
> - `docs/online_tracking_implementation.md` — 43 任务实施追踪表
> - `docs/setup_phase*.md` — 各阶段复现说明（7 篇）

---

## 目录

- [1. 系统使用](#1-系统使用)
  - [1.1 工作流程概览](#11-工作流程概览)
  - [1.2 离线建图（Offline mapping）](#12-离线建图offline-mapping)
  - [1.3 在线追踪（Online tracking）](#13-在线追踪online-tracking)
  - [1.4 Python 端消费 pose](#14-python-端消费-pose)
  - [1.5 离线 playback（无设备复跑）](#15-离线-playback无设备复跑)
- [2. 代码结构与功能](#2-代码结构与功能)
  - [2.1 总体架构](#21-总体架构)
  - [2.2 主要组件清单](#22-主要组件清单)
  - [2.3 关键实现要点](#23-关键实现要点)
- [3. 环境配置](#3-环境配置)
  - [3.1 硬件 / 平台要求](#31-硬件--平台要求)
  - [3.2 一次性安装步骤](#32-一次性安装步骤)
  - [3.3 验证安装](#33-验证安装)
  - [3.4 网络代理（如需）](#34-网络代理如需)

---

## 1. 系统使用

### 1.1 工作流程概览

这套系统把 SLAM 拆成两个阶段：

```
─────────────────────  阶段 A：离线建图  ─────────────────────
  1. ROS2 录 D435i (color + imu) →  data_session/data_*/
  2. python run_slam_pipeline_ros2.py  → demos/mapping/map_atlas.osa  (👈 这是地图文件)
                                            + 其他 calibration 输出

─────────────────────  阶段 B：在线追踪（本系统的新增内容）  ─────────────────────
  1. 加载 map_atlas.osa
  2. realsense_online 实时跑 ORB-SLAM3 定位模式
  3. 6-DoF pose @30Hz 通过 ROS2 topic 或 ZMQ 发布
  4. Python / 其他 ROS2 节点订阅消费
```

**核心思路**：地图（feature map）只在阶段 A 建一次；阶段 B 不更新地图，纯粹做实时重定位 → 直接拿位姿。

---

### 1.2 离线建图（Offline mapping）

> 这一部分用的是 `umi_mono` 原本就有的流水线（Docker），不在本项目本次新增的代码内，但用户需要先跑这一步才能拿到 `map_atlas.osa`。

#### 步骤

1. **采集数据**（D435i 接 USB3 上的主机；ROS2 Humble）

   ```bash
   source /opt/ros/humble/setup.bash
   # 启动相机
   ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true unite_imu_method:=0

   # 准备一个空目录作为 session
   mkdir ~/dexslide_data/session_2026_05_19
   cd ~/dexslide_data/session_2026_05_19

   # 录制建图数据（在场景中慢速绕一圈 60 秒以上，回环越好）
   ros2 bag record -o data_mapping \
     /camera/camera/color/image_raw \
     /camera/camera/accel/sample \
     /camera/camera/gyro/sample

   # 录制 ArUco 标定（把 ArUco-13 tag 摆桌面上、对着相机录 10 秒）
   ros2 bag record -o aurco \
     /camera/camera/color/image_raw \
     /camera/camera/accel/sample \
     /camera/camera/gyro/sample
   ```

   > 注：`aurco`（不是 aruco）这个拼写是与现有代码一致的，**别改名**。

2. **运行建图流水线**

   ```bash
   cd /data/codes/DexSlide/umi_mono
   conda activate umi   # 或激活你的 ROS2 Python 环境
   pip install rosbags  # 一次性依赖

   python run_slam_pipeline_ros2.py ~/dexslide_data/session_2026_05_19
   ```

3. **产出物**（在 `~/dexslide_data/session_2026_05_19/demos/` 下）

   - **`mapping/map_atlas.osa`** ← 关键产物，在线追踪要喂给它
   - `aurco/tx_slam_tag.json` — SLAM 坐标系到世界 ArUco tag 的变换
   - `data_*/camera_trajectory.csv` — batch 重定位结果（可作 ATE baseline）

#### 建图诀窍（避免后续在线追踪丢失）

- 建图时绕场景 1-2 圈，包含**回环**（回到起点）
- 光照与在线使用环境一致（白天建图、晚上用 → 易丢失）
- 不要太快、不要纯转头（要平移 + 旋转结合）
- 镜头别看天 / 看大白墙（无 ORB 特征）

---

### 1.3 在线追踪（Online tracking）

#### 1.3.1 一键 launch（推荐：用 ROS2）

```bash
# 一次性 source（每个新终端都要做）
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash

# 启动
ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py \
  map_atlas:=/path/to/map_atlas.osa \
  exposure_us:=8000
```

会同时起两个进程：
- `realsense_online` — 跑 D435i + ORB-SLAM3，通过 ZMQ 发位姿
- `pose_publisher_node` — 把 ZMQ 转成 ROS2 `geometry_msgs/PoseStamped` + tf2

#### Launch 参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `map_atlas` | (空) | **.osa 路径**；为空时会跑在线建图（一般用于调试） |
| `vocab` | `<fork>/Vocabulary/ORBvoc.txt` | ORB vocabulary，建图时用什么这里用什么 |
| `settings` | `umi_mono/config/RealSense_D435i_online.yaml` | SLAM 相机内参 + IMU 噪声 yaml |
| `exposure_us` | 0 (auto) | D435i color sensor 曝光（微秒）；室内日光下推荐 `8000` |
| `pose_topic` | `/dexslide/slam/pose` | ROS2 输出 topic 名 |
| `zmq_endpoint` | `tcp://127.0.0.1:5555` | 两个进程之间的 ZMQ 链路 |
| `log_to_file` | false | true 时 stdout 走 ROS log 文件 |

#### 验证 launch 是否成功

另起一个终端：

```bash
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash

ros2 topic hz /dexslide/slam/pose           # 期望 ~30 Hz
ros2 topic echo /dexslide/slam/pose --once  # 看一帧 PoseStamped
ros2 run tf2_ros tf2_echo map camera_color_optical_frame
```

#### 1.3.2 不要 ROS2、直接跑（最小依赖）

```bash
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml \
  -l /path/to/map_atlas.osa \
  --publisher stdout \
  --exposure_us 8000
```

输出（每帧）：
```
pose 1747608901.234567 0.123 -0.045 0.789 0.001 0.002 -0.707 0.707
     ^t (s)            ^tx ^ty  ^tz  ^qx  ^qy   ^qz   ^qw
```

Ctrl-C 干净退出。

#### 1.3.3 三种 publisher 后端

| `--publisher` | 用途 |
|---------------|------|
| `stdout` (默认) | 调试，shell 管道处理；JSON 之外的格式 |
| `zmq` | 程序间通信，最低延迟（< 2ms），任何 ZMQ SUB 客户端可消费 |
| `ros2` | 等价于 `zmq`，但提示要另起 `pose_publisher_node` 桥；通常通过 launch 一键启动 |

---

### 1.4 Python 端消费 pose

DexSlide 主程序 / Vedo viewer / 控制脚本通过 `dexslide.world_pose.SlamPoseSubscriber` 拿位姿：

```python
import sys
sys.path.insert(0, '/data/codes/DexSlide')

from dexslide.world_pose import SlamPoseSubscriber

# 1) 构造（rclpy.init 懒加载，调用者无需自己 init）
sub = SlamPoseSubscriber(stale_after_seconds=0.2)

# 2) 后台 spin
sub.spin_in_thread()

# 3) 用法 A：拿最新 pose
import time
while True:
    if sub.is_tracking():
        t, T = sub.latest()           # T 是 4x4 numpy SE(3)
        print(f"x={T[0,3]:.3f}, y={T[1,3]:.3f}, z={T[2,3]:.3f}")
    else:
        print("tracking lost")
    time.sleep(0.033)

# 4) 用法 B：时间对齐查询（给定时刻 t）
T = sub.get_T_world_camera(t=image_capture_time)   # 在 ±100ms 内 SLERP 插值
if T is not None:
    palm_in_world = T @ palm_in_camera

# 5) 清理
sub.stop()
```

**关键要点**：
- 必须用 **`/usr/bin/python3`**（Ubuntu 系统 Python 3.10）；anaconda 的 3.13 不兼容 rclpy
- 调用 Python 脚本前先 `source /opt/ros/humble/setup.bash`

完整 API 详见 `docs/setup_phase5_python_consumer.md`。

---

### 1.5 离线 playback（无设备复跑）

没有 D435i 接着也能跑（用之前录的数据）：

```bash
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml \
  -l /path/to/map_atlas.osa \
  --playback_video /path/to/session/demos/data_001/raw_video.mp4 \
  --playback_imu   /path/to/session/demos/data_001/imu_data.json \
  --publisher stdout
```

走完整个在线追踪管线（含 MD5 校验 / skew 校验 / ActivateLocalizationMode），只是 frame source 换成文件。**用途**：与离线 Docker batch 输出对比（ATE 测试，见 `tests/test_ate_vs_docker.py`），CI 回归。

---

## 2. 代码结构与功能

### 2.1 总体架构

```
                   ┌─────────────────────────────────────────────┐
                   │  阶段 A: 离线建图（umi_mono 原有 Docker 流水）  │
                   │                                              │
                   │  ROS2 bag → process_videos → docker          │
                   │  gopro_slam --save_map → map_atlas.osa       │
                   └─────────────────────┬───────────────────────┘
                                         │
                                  map_atlas.osa
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  阶段 B: 在线追踪（本项目新增）                                     │
│                                                                  │
│  ┌─────────────────────────────────────┐                         │
│  │ realsense_online (C++ binary)       │                         │
│  │   ├─ RsCapture: librealsense2       │                         │
│  │   │    (color 960x540 @30Hz +       │                         │
│  │   │     accel 200Hz + gyro 200Hz,   │                         │
│  │   │     GLOBAL_TIME)                │                         │
│  │   │                                 │                         │
│  │   ├─ ImuRingBuffer: SPSC (256)      │                         │
│  │   │                                 │                         │
│  │   ├─ ORB_SLAM3::System              │                         │
│  │   │    IMU_MONOCULAR, no viewer     │                         │
│  │   │    LoadAtlasFromFile=.osa       │                         │
│  │   │    ActivateLocalizationMode()   │                         │
│  │   │                                 │                         │
│  │   ├─ Main loop:                     │                         │
│  │   │  TrackMonocular → SE(3) →       │                         │
│  │   │  NaN guard → publisher          │                         │
│  │   │                                 │                         │
│  │   └─ ZmqPosePublisher (JSON line)   │                         │
│  └─────────────────────┬───────────────┘                         │
│                        │ tcp://127.0.0.1:5555                    │
│                        ▼                                         │
│  ┌─────────────────────────────────────┐                         │
│  │ pose_publisher_node (rclcpp)        │                         │
│  │   ZMQ SUB → PoseStamped + tf2       │                         │
│  └─────────────────────┬───────────────┘                         │
│                        │ /dexslide/slam/pose @30Hz               │
│                        │ tf2: map → camera_color_optical_frame   │
│                        ▼                                         │
│  ┌─────────────────────────────────────┐                         │
│  │ SlamPoseSubscriber (Python)         │                         │
│  │   deque[300] + threading.Lock       │                         │
│  │   manual SLERP 时间对齐             │                         │
│  │   is_tracking / get_T_world_camera  │                         │
│  └─────────────────────────────────────┘                         │
│                        │                                         │
│                        ▼                                         │
│            DexSlide main.py / Vedo viewer                        │
└──────────────────────────────────────────────────────────────────┘
```

---

### 2.2 主要组件清单

#### A. ORB-SLAM3 Fork 内的新增（C++14）

位置：`umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/`

| 文件 | 行数 | 作用 |
|------|------|------|
| `realsense_online.cc` | ~340 | 主可执行：CLI11 解析 / 调起所有组件 / 主循环 / 信号处理 |
| `realsense_capture.hpp` + `.cpp` | ~120 | `RsCapture` 类，封装 librealsense2 pipeline（color + IMU + GLOBAL_TIME timestamp domain + 曝光锁） |
| `imu_ring_buffer.hpp` | ~50 | 手写 SPSC lock-free 环形队列模板；`push()` / `drain_until(t)` |
| `test_imu_ring_buffer.cc` | ~120 | 5 个 assert 测试（empty / single / time-bounded / overflow / cross-thread） |
| `zmq_pose_publisher.hpp` + `.cpp` | ~110 | ZMQ PUB 封装，JSON 行格式发布 SE(3) |

CMakeLists.txt 改动：+10 行（链接 librealsense2 + libzmq + 新增 3 个 target）。

#### B. ROS2 Bridge 包（C++17）

位置：`umi_mono/ros2_ws/dexslide_slam_publisher/`

| 文件 | 作用 |
|------|------|
| `src/pose_publisher_node.cpp` | rclcpp 节点：ZMQ SUB → PoseStamped + tf2 broadcast |
| `launch/dexslide_slam_online.launch.py` | 一键启动 realsense_online + bridge |
| `CMakeLists.txt` | ament_cmake 构建 + 安装 |
| `package.xml` | 包元数据（depends: rclcpp, geometry_msgs, tf2_ros） |

#### C. Python 消费端（Python 3.10）

位置：`dexslide/world_pose/`

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 `SlamPoseSubscriber` |
| `slam_pose_subscriber.py` | ~200 行 class：deque ring buffer + 线程安全 / 手写 SLERP / SingleThreadedExecutor / spin_in_thread |

测试：`tests/test_slam_pose_subscriber.py` 共 10 个 pytest 用例（quat 数学 / SLERP / buffer / interpolation / stale flag）。

#### D. 构建与安装脚本

位置：`umi_mono/scripts/`

| 脚本 | 一次性？ | 作用 |
|------|---------|------|
| `check_host_env.sh` | 每机器一次 | 验 OS/GCC/CMake/Python 版本 |
| `install_apt_deps.sh` | 每机器一次（需 sudo） | 装 OpenCV/Eigen/Boost/Pangolin deps/etc apt 包 |
| `build_pangolin.sh` | 每机器一次 | clone + 编译 Pangolin v0.8 |
| `build_sophus.sh` | 每机器一次 | clone + 配置 Sophus 1.22.10 |
| `install_librealsense.sh` | 每机器一次（需 sudo） | 装 librealsense2-dev |
| `run_d435i_smoke.sh` | 调试用 | 跑 librealsense smoke test |
| `setup_orbslam3_fork.sh` | 每机器一次 | clone cheng-chi/ORB_SLAM3 + pin SHA + 解 vocabulary |
| `build_orbslam3_native.sh` | 改 fork 代码后 | 编译 fork（含我们新增的 realsense_online target） |

#### E. 验证脚本

位置：`umi_mono/tests/`

| 脚本 | D435i? | 作用 |
|------|--------|------|
| `test_vocab_mismatch.sh` | 不需 | 假 vocab 必须 5s 内退出 ✓ 已 PASS |
| `test_atlas_immutable.sh <map.osa> [duration]` | 不需 | SHA-256 前后对比 |
| `test_ate_vs_docker.py` | 不需 | 原生 binary 与 Docker batch 输出 ATE 对比 |
| `benchmark_30min.sh` | **需要** | 30 分钟 p50/p99/p99.9 + 直方图 + JSON |
| `test_headless.sh` | **需要** | `unset DISPLAY` 30s 跑 |
| `test_memory_2h.sh` | **需要** | 2 小时 RSS 增长 < 5% |
| `test_recovery_3s.sh` | **需要** | 半自动遮挡 + 恢复时间测量 |

#### F. 配置文件

| 文件 | 作用 |
|------|------|
| `umi_mono/config/RealSense_D435i.yaml` | 离线 batch 用（umi_mono 原有） |
| `umi_mono/config/RealSense_D435i_online.yaml` | 在线追踪用（派生自上面，viewer 字段清 0） |

---

### 2.3 关键实现要点

#### 2.3.1 离线建图 → 在线追踪的桥梁
- ORB-SLAM3 fork (cheng-chi 版) 已经支持 `--load_map` 和 `--save_map` 接口
- 离线流水线的 `02_create_map.py` 用 `--save_map` 产出 `.osa`
- 我们的 `realsense_online` 用 `LoadAtlasFromFile` 字段（在 settings YAML 里）加载同一个 `.osa`
- 算法、Vocab、IMU 标定等参数共享 → 离线/在线结果可比

#### 2.3.2 定位模式（不更新地图）
```cpp
ORB_SLAM3::System slam(vocab, settings, IMU_MONOCULAR, /*bUseViewer=*/false, ...);
slam.ActivateLocalizationMode();          // 关键：阻止 mapping thread 写地图
// 主循环：
auto [Tcw, ok] = slam.LocalizeMonocular(image, t, vImu);   // 返回 (位姿, has_tracking)
```
- `ActivateLocalizationMode` 关 Local Mapping 线程
- `LocalizeMonocular` 是 fork 扩展的接口（vs 上游 `TrackMonocular`）— 多返一个 has_tracking flag
- 由 atlas 文件 SHA-256 不变测试（`test_atlas_immutable.sh`）保证

#### 2.3.3 30Hz 实时管线
- **图像路径**：D435i 30Hz → librealsense callback → mutex 保护拷一份 → main loop poll
- **IMU 路径**：200Hz accel + 200Hz gyro → librealsense callback → SPSC lock-free 环形 buffer → `drain_until(t)` 按时间戳消费
- **同步**：每帧用 `(prev_image_t, current_image_t]` 区间的 IMU 样本喂给 `TrackMonocular`
- **时间戳源**：`RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`（D435i 固件 ≥ 5.16 支持），同时统一 image / IMU 时钟域
- 启动时跑 100 帧 warmup → 算 |image_t - imu_t| 中位数 → 超过 `--skew_abort_ms` 阈值 abort

#### 2.3.4 跟踪丢失软重置
```cpp
if (!has_tracking) {
    consecutive_lost++;
    if (consecutive_lost > max_lost_frames) {       // 默认 900 帧 = 30s
        log("Triggering soft reset (lost > max)");
        consecutive_lost = 0;                       // 仅 reset 计数器
        // 继续喂帧让 ORB-SLAM3 内部 relocalizer 工作
    }
}
```
- **不调** `slam.Reset()` —— 那会清空地图
- 持续喂帧 → ORB-SLAM3 内部 `RELOCALIZATION` 状态机会在熟悉区域自动恢复
- 实测恢复时间 < 5s（TASK-043 验证）

#### 2.3.5 输出通道（三选一）
| 通道 | 延迟 | 复杂度 | 何时用 |
|------|------|--------|--------|
| stdout | 0 ms | 0 | 调试，shell 管道 |
| ZMQ PUB tcp://127.0.0.1:5555 | < 2 ms | 低 | 程序间通信，跨语言 |
| ROS2 PoseStamped + tf2 | ~5 ms | 中 | ROS 生态集成，rviz, rosbag record, etc. |

ROS2 通道实际上是 ZMQ + bridge 节点 → 保持 fork（GPL-3）与 DexSlide 主程序（拟 BSD）license 边界清晰。

#### 2.3.6 健壮性强化（启动时检查）
- **Vocabulary MD5**：`popen('md5sum')` 算 + 打印（informational + size 范围 sanity check）
- **Atlas 文件存在**：CLI 路径校验
- **Image/IMU 时间戳偏差**：100 帧中位数 > 5ms 时 abort
- **D435i 设备存在**：无设备时优雅退出 0（不 segfault）
- **Headless 模式**：`bUseViewer=false`，DISPLAY 缺失也能跑

---

## 3. 环境配置

### 3.1 硬件 / 平台要求

| 项 | 要求 |
|----|------|
| OS | Ubuntu 22.04 LTS (`22.04.x` 任意小版本) |
| 架构 | x86_64 |
| CPU | i7-12700 或类似（30Hz @ p99 < 33ms 需要足够算力） |
| GPU | 不需要 |
| 内存 | ≥ 16 GB 推荐 |
| 磁盘 | ≥ 30 GB 可用（含 fork 源码 + Pangolin/Sophus 编译产物 + ORBvoc.txt 145MB） |
| **D435i 相机** | 固件 ≥ 5.16；接 USB 3.x 端口 |
| ROS2 | Humble Hawksbill |
| Python | 系统 `/usr/bin/python3` (3.10)。**不要用 anaconda 3.13 跑 ROS2 Python 代码** |
| 网络 | clone GitHub / apt update；国内/受限网络见 [3.4 网络代理](#34-网络代理如需) |

---

### 3.2 一次性安装步骤

完整流程（10-15 分钟，假设网络通畅、有 sudo）：

```bash
# 0. clone 项目
git clone <your-repo-url> /data/codes/DexSlide
cd /data/codes/DexSlide

# 1. 主机环境快速检查（read-only）
bash umi_mono/scripts/check_host_env.sh
# 期望：所有 [OK]（python3 可能 [WARN] 因 3.13）
```

```bash
# 2. apt 依赖（dry-run 先看缺啥）
bash umi_mono/scripts/install_apt_deps.sh
# 看到 'X to install'，然后授权安装：
bash umi_mono/scripts/install_apt_deps.sh --apply
# 输入 yes 确认；约 1 分钟
```

```bash
# 3. 编译 Pangolin v0.8（~6 分钟）
bash umi_mono/scripts/build_pangolin.sh
```

```bash
# 4. 配置 Sophus 1.22.10（~10 秒）
bash umi_mono/scripts/build_sophus.sh
```

```bash
# 5. 装 librealsense2-dev
bash umi_mono/scripts/install_librealsense.sh           # dry-run
# 授权安装：
echo "yes" | bash umi_mono/scripts/install_librealsense.sh --apply
# 或 直接：sudo apt install -y librealsense2-dev
```

```bash
# 6. 装 ROS2 Humble（如果还没装）
# 详见官方：https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
sudo apt install -y ros-humble-desktop ros-humble-cv-bridge \
  ros-humble-tf2-ros ros-humble-geometry-msgs ros-humble-sensor-msgs
```

```bash
# 7. clone ORB-SLAM3 fork + pin SHA + 解 vocabulary（~1 分钟，需要网络）
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  bash umi_mono/scripts/setup_orbslam3_fork.sh
# 期望末尾打印 SHA + branch + Vocabulary/ORBvoc.txt size = 145250924
```

```bash
# 8. clone cppzmq 头文件（header-only，~1MB）
cd /data/codes/DexSlide/umi_mono/external
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  git clone --depth 1 --branch v4.10.0 https://github.com/zeromq/cppzmq.git
cd /data/codes/DexSlide
```

```bash
# 9. 编译 ORB-SLAM3 fork + realsense_online（~3-5 分钟）
bash umi_mono/scripts/build_orbslam3_native.sh
# 期望末尾：DBoW2 OK + g2o OK + Sophus OK + Pangolin SKIPPED + ORB_SLAM3 OK
```

```bash
# 10. 编译 ROS2 桥节点
source /opt/ros/humble/setup.bash
cd /data/codes/DexSlide/umi_mono/ros2_ws
colcon build --packages-select dexslide_slam_publisher
# 期望末尾：Finished <<< dexslide_slam_publisher
```

完成！

---

### 3.3 验证安装

跑这几条命令逐一验证：

```bash
# 主机环境
bash umi_mono/scripts/check_host_env.sh && echo "PASS_ENV"

# 外部依赖
ls umi_mono/external/Pangolin/build/libpango_core.so      # Pangolin
ls umi_mono/external/Sophus/build/SophusConfig.cmake      # Sophus
ls umi_mono/external/ORB_SLAM3_fork/lib/libORB_SLAM3.so   # ORB-SLAM3 (354MB)
ls umi_mono/external/cppzmq/zmq.hpp                       # cppzmq
pkg-config --modversion realsense2                        # 应输出 ≥ 2.55

# Fork 产物
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/gopro_slam --help | head -3
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online --help | head -3
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/test_imu_ring_buffer  # 期望 TEST PASS

# ROS2 桥
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash
ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py --show-args | head -10

# Python 消费端
cd /data/codes/DexSlide
/usr/bin/python3 -m pytest tests/test_slam_pose_subscriber.py -v
# 期望：10 passed in <0.5s

# 在线追踪二进制（无 D435i 也能跑，会优雅退出）
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml
# 无 D435i 期望：'No RealSense device' + exit 0

# 错误处理测试
bash /data/codes/DexSlide/umi_mono/tests/test_vocab_mismatch.sh
# 期望：TASK-042 PASS: wrong vocab failed gracefully

# D435i 烟囱（插上 D435i 后跑）
bash /data/codes/DexSlide/umi_mono/scripts/run_d435i_smoke.sh
# 期望：color FPS ≈ 30, accel ≈ 200, gyro ≈ 200, PASS
```

---

### 3.4 网络代理（如需）

国内 / 受限网络环境下 `git clone github.com` 和 `pip install` 可能不通。本项目用 Clash @ `127.0.0.1:7890` 时设环境变量：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
```

确认通：
```bash
HTTPS_PROXY=http://127.0.0.1:7890 curl -sS --max-time 5 \
  https://api.github.com -H "Accept: application/json" | head -3
```

也可以加进 `~/.bashrc` 永久生效。

---

## 4. 常见问题（FAQ）

### Q: 重新建图后能直接换 `--map_atlas` 用吗？
A: 可以，**但需要保证 vocab 文件不变**。新机器/新 fork 的 ORBvoc.txt MD5 应与建图时一致：
```bash
md5sum umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt
# 期望：5420bad0713bc97034dd2a9b2f0cc387
```

### Q: `tracking lost` 长期不恢复？
A: 检查：
- 当前场景 vs 建图时光照差异（差异大→不行）
- 建图时回环够不够（应至少绕回 1 次起点）
- 建图时拍到大量 feature 区域，不要纯白墙
- `--max_lost_frames` 可调大到 1800（60s）

### Q: 30 Hz 跑不满？
A:
- USB 3.x 端口确认（`lsusb -t` 看 SuperSpeed）
- CPU governor：`sudo cpupower frequency-set -g performance`
- 减小 ORB features：`config/RealSense_D435i_online.yaml` 里 `ORBextractor.nFeatures` 从 1250 降到 1000

### Q: 在 `anaconda Python` 里 `import dexslide.world_pose` 报 rclpy 错？
A: 这是预期的。anaconda Python 3.13 和系统 Python 3.10 二进制不兼容；rclpy 只装在系统 Python 3.10。**强制用** `/usr/bin/python3`，可以：
```bash
conda deactivate
source /opt/ros/humble/setup.bash
/usr/bin/python3 your_script.py
```

### Q: 修改 fork 内的代码（例如调整主循环）后怎么重新编译？
A:
```bash
cd /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork
cmake --build build -j --target realsense_online
# 几秒到几十秒（增量编译）
```

### Q: 修改 ROS2 桥代码后怎么重新编译？
A:
```bash
source /opt/ros/humble/setup.bash
cd /data/codes/DexSlide/umi_mono/ros2_ws
colcon build --packages-select dexslide_slam_publisher --symlink-install
# 几十秒
```

---

## 5. 关联资源

- **研究报告**：`umi_mono/docs/online_tracking_research.md`
- **任务追踪**：`umi_mono/docs/online_tracking_implementation.md`（43 任务进度表）
- **阶段复现说明**：
  - Phase 0 主机环境：`umi_mono/docs/setup_phase0_environment.md`
  - Phase 1 Fork 集成：`umi_mono/docs/setup_phase1_native_build.md`
  - Phase 2 realsense_online：`umi_mono/docs/setup_phase2_realsense_online.md`
  - Phase 3+4 发布后端+健壮性：`umi_mono/docs/setup_phase3_4_publishers_robustness.md`
  - Phase 5 Python 消费者：`umi_mono/docs/setup_phase5_python_consumer.md`
  - Phase 6+7 Launch+验证：`umi_mono/docs/setup_phase6_7_launch_validation.md`

---

## 6. 已知小问题

- `realsense_online --help` 中 `--publisher` 的描述文字仍是早期占位（"only stdout works; ros2 and zmq error out"），实际三种都可用。不影响功能，下次改 binary 时一并更新。
- ZMQ → ROS2 桥在桥节点未启动时，realsense_online 的 ZMQ PUB 会无消费者（消息丢弃但不报错）。这是 ZMQ PUB/SUB 的标准语义。
