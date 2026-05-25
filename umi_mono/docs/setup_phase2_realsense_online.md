# Phase 2 — 实现 `realsense_online.cc`（操作说明）

> 关联 tracker: `umi_mono/docs/online_tracking_implementation.md`
> 前置：`setup_phase0_environment.md` + `setup_phase1_native_build.md` 全部通过
> 完成 8 个 TASK 后：`realsense_online` binary 可以加载 .osa atlas、跑 ORB-SLAM3 定位模式、对 D435i 实时帧调用 TrackMonocular、stdout 输出位姿，并响应 SIGINT/SIGTERM 干净关停

---

## 0. Phase 2 改了 fork 的哪些文件

```
external/ORB_SLAM3_fork/
├── CMakeLists.txt                                                  # +6 行
└── Examples/Monocular-Inertial/
    ├── realsense_online.cc                                         # 新增 ~240 行
    ├── realsense_capture.hpp                                       # 新增（RsCapture 接口）
    ├── realsense_capture.cpp                                       # 新增（pipeline 实现）
    ├── imu_ring_buffer.hpp                                         # 新增（header-only SPSC 模板）
    └── test_imu_ring_buffer.cc                                     # 新增（5 个 assert 测试）
```

CMakeLists.txt 改动（unified diff，line numbers 近似）：

```diff
@@ around line 45 @@
+find_package(realsense2 QUIET CONFIG)
+if(NOT realsense2_FOUND)
+  find_package(PkgConfig REQUIRED)
+  pkg_check_modules(REALSENSE2 REQUIRED realsense2)
+  add_library(realsense2::realsense2 INTERFACE IMPORTED)
+  set_target_properties(realsense2::realsense2 PROPERTIES
+    INTERFACE_INCLUDE_DIRECTORIES "${REALSENSE2_INCLUDE_DIRS}"
+    INTERFACE_LINK_LIBRARIES "${REALSENSE2_LINK_LIBRARIES}")
+endif()

@@ near gopro_slam target @@
-add_executable(realsense_online
-Examples/Monocular-Inertial/realsense_online.cc)
-target_link_libraries(realsense_online ${PROJECT_NAME})
+add_executable(realsense_online Examples/Monocular-Inertial/realsense_online.cc Examples/Monocular-Inertial/realsense_capture.cpp)
+target_link_libraries(realsense_online ${PROJECT_NAME} realsense2::realsense2)
+
+add_executable(test_imu_ring_buffer Examples/Monocular-Inertial/test_imu_ring_buffer.cc)
+target_link_libraries(test_imu_ring_buffer pthread)
```

---

## 1. 一行复现命令

```bash
# 编译（Phase 1 已完成的前提下）
cmake --build external/ORB_SLAM3_fork/build -j --target realsense_online
cmake --build external/ORB_SLAM3_fork/build -j --target test_imu_ring_buffer

# 跑 IMU ring buffer 单元测试（独立，无需 D435i）
external/ORB_SLAM3_fork/Examples/Monocular-Inertial/test_imu_ring_buffer
# 期望：TEST PASS, exit 0

# 跑 realsense_online --help
external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online --help

# 跑 realsense_online（无 D435i 也能跑，会优雅退出）
external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s umi_mono/config/RealSense_D435i.yaml
# 无 D435i：打印 'No RealSense device' + exit 0
# 有 D435i：构造 SLAM、调 ActivateLocalizationMode、5 秒主循环、stdout 'pose ...'

# 带 atlas（要先有 map_atlas.osa）
external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s umi_mono/config/RealSense_D435i.yaml \
  -l /path/to/map_atlas.osa
```

---

## 2. CLI 全表

| Flag | 类型 | 默认 | 含义 |
|------|------|------|------|
| `-v, --vocabulary` | path | (必填) | ORB vocabulary 文件，典型 `<fork>/Vocabulary/ORBvoc.txt` |
| `-s, --setting` | path | (必填) | SLAM settings YAML，本项目用 `umi_mono/config/RealSense_D435i.yaml` |
| `-l, --load_map` | path | (空) | 加载已有 .osa atlas；空 = 从零起建图（生产环境必填） |
| `--publisher` | enum | `stdout` | 输出后端：`stdout` / `ros2` / `zmq`；后两者 TASK-018/019 实现 |
| `--exposure_us` | int | 0 | RealSense 曝光微秒；0 = 自动；TASK-025 锁定 |
| `--max_lost_frames` | int | 900 | tracking 丢失多少帧后 soft reset；30Hz×30s = 900 |
| `--skew_abort_ms` | int | 5 | image/IMU 时间戳偏差超 N ms 则启动期 abort；TASK-021 实现 |

---

## 3. 主循环数据流

```
RealSense SDK
  │
  ├── color callback (30Hz, BGR8 960x540)
  │     → mutex-protected copy into last_image (cv::Mat) + last_image_t
  │
  └── motion callback (~400Hz combined accel+gyro @ 200Hz each)
        → ImuSample{a, w, t} pushed into ImuRingBuffer<>

Main loop (5s timer + g_should_exit flag)
  │
  ├── cap.poll_image(100ms)              # blocks for next frame
  ├── if has_new under mutex:
  │     copy image out + clear flag
  ├── imu_buf.drain_until(current_t)     # consumes IMU samples with t ≤ current_t
  ├── filter samples (prev_t, current_t]; build vector<IMU::Point>
  ├── slam.TrackMonocular(image, current_t, vImu)
  ├── is_pose_valid(Tcw)?
  │     yes → cout << 'pose t x y z qx qy qz qw'
  │     no  → n_invalid++ (TASK-015)
  └── prev_t = current_t

Signal handler (TASK-016)
  SIGINT/SIGTERM → g_should_exit = true → loop exits next iter

Cleanup
  cap.stop() → slam.Shutdown() → return 0
```

---

## 4. 关键决策记录（避免新机器复现踩坑）

### 4.1 为什么不用 `boost::lockfree::spsc_queue` 而是手写 SPSC
- `boost::lockfree::spsc_queue<T>` 要求 `T` 默认可构造
- ORB-SLAM3 的 `IMU::Point` **没有默认构造函数**（只有 2 个带参 ctor）
- 我们的 `ImuSample`（POD）虽然可以默认构造，但 `boost::lockfree::spsc_queue` **不支持 peek**，做 `drain_until(timestamp)`（消费到某时间戳为止）会被迫读取下一个 sample 才知道要不要 stop —— 多读了一个就回不去
- 手写 SPSC ring（std::array + atomic head/tail，~50 LOC header）支持 peek，干净实现 `drain_until` 和 `drain_in_range`

### 4.2 为什么把图像走 mutex / IMU 走 lock-free
- IMU 频率 200Hz，但是是低延迟、不能阻塞的 SDK callback：lock-free SPSC 必要
- 图像 30Hz，回调线程拷一份 `cv::Mat` 到主缓冲（main loop 消费）就够；mutex 持有时间约几十微秒，不会阻塞 SDK

### 4.3 为什么 fork 用 -std=c++14 不是 C++17
- fork 的 CMakeLists 强制 C++14（继承自 ORB-SLAM3 上游）
- 我们的 `realsense_online.cc` 用 `std::filesystem`（C++17）；解决方案：`#if __cplusplus < 201703L` 加 `<sys/stat.h>` fallback shim
- 不直接改 fork 的 CMake 标准（避免破坏其他 example）

### 4.4 vendored CLI11.hpp 没 `add_set`
- fork 的 `include/CLI11.hpp` 是个老版本，没 `app.add_set(name, choices, ...)`
- 改用 `add_option(name, var, desc)->check(CLI::IsMember({...}))` —— 等价语义
- 行为完全相同，只是 API 写法不同

### 4.5 librealsense2.so 实际链接到 ROS2 Humble 自带版本
- `ldd realsense_online | grep realsense2` 显示 `librealsense2.so.2.57 => /opt/ros/humble/lib/x86_64-linux-gnu/librealsense2.so.2.57`
- 不是 `/usr/lib/...` 上的 Intel PPA 版本
- 原因：ROS2 setup script 把 `/opt/ros/humble/lib` 加到 `LD_LIBRARY_PATH` 前面
- **不是问题**：两个版本都是 2.57.7，ABI 一致

### 4.6 `Sophus::SE3f` 失败时的语义
- `TrackMonocular` 跟踪失败时返回**默认构造的 SE3f**（identity，不是空）
- 我们用 `std::isfinite()` 检查每个分量 —— 但其实 identity 也是 finite。这意味着如果 SLAM 真的返回 identity 我们会当作 "valid" 打印
- TASK-015 的 NaN guard 主要防范的是**未初始化/损坏的 pose**（罕见）
- 真正的"无效 pose"判定要看 ORB-SLAM3 内部 tracking state；fork 的 `LocalizeMonocular` 返回 `pair<SE3f, bool>`（has_tracking flag），TASK-024 会切换到这个 API

---

## 5. 失败时如何修复

### `cmake --build` 报 `'CLI::App' has no member named 'add_set'`
fork 的 vendored CLI11.hpp 是老版本。改用 `->check(CLI::IsMember({...}))`。已在 TASK-010 实施。

### `librealsense2.so: cannot open shared object file`
ROS2 没 source 进 env：
```bash
source /opt/ros/humble/setup.bash
```
或者直接：
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 主循环里 `slam.TrackMonocular()` 在 D435i 朝外（无 ORB 特征）的环境下，pose 一直是 NaN
预期行为：TASK-015 会跳过；进生产环境必填 `--load_map` 加载已建图区域。

### SIGINT 没响应
确认信号处理是 free function 不是 lambda（C++ lambda with capture 不能转 C 函数指针）。已在 TASK-016 用 `static void on_signal(int)` 实现。

### 编译警告 `comparison of integer expressions of different signedness`
warn-not-error，可忽略；fork 上游遗留代码。

---

## 6. 还没验证的部分（等 D435i 插上）

- **TASK-017 live smoke**：把 D435i 朝向已建图区域跑 60s，期望：
  - ≥1500 行 valid pose（30Hz × 60s × 0.83 容差）
  - n_invalid 比例 < 10%
  - 'Localization mode active' 立即出现
  - SIGINT 5 秒内退出
- **TASK-013 atlas 加载**：用真实 `map_atlas.osa` 跑，确认 'Atlas loaded from: ...' + KeyFrames 数 > 0
- **TASK-021 timestamp skew**：跑前 100 帧统计 image/IMU 时间戳差

---

## 7. 产物清单

| 文件 | 用途 |
|------|------|
| `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online` | 在线追踪 binary (~17 MB) |
| `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/test_imu_ring_buffer` | IMU buffer 单元测试 binary |
| `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online.cc` | 主 source |
| `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_capture.hpp/.cpp` | RealSense pipeline 封装 |
| `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/imu_ring_buffer.hpp` | SPSC ring 模板 |
| `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/test_imu_ring_buffer.cc` | 单元测试 |

---

## 8. 下一阶段

**Phase 3 — 发布后端**（TASK-018 ~ TASK-020）

- TASK-018: ZMQ pose publisher (`tcp://127.0.0.1:5555`, line-JSON)
- TASK-019: ROS2 桥节点（订 ZMQ → 发 `geometry_msgs/PoseStamped` + tf2）
- TASK-020: `--publisher {stdout,zmq,ros2}` flag 实际 dispatch

预估 1 天。Phase 3 完成后，DexSlide 的 Python 侧 Vedo viewer 就能消费 pose。
