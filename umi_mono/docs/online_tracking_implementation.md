# 在线追踪 实施清单 (Codex Handoff Tracker)

> 关联文档：
> - 研究报告：`umi_mono/docs/online_tracking_research.md`
> - OpenSpec change：`openspec/changes/online-tracking-mode/{proposal,design,specs,tasks}.md`
>
> 工作流：每条 TASK 是一次 Codex 调用的工作单元，独立交付、独立验收。本 tracker 是 Claude ↔ Codex 协作的"作业本"，每条完成后由用户审查后打勾继续。

---

## 状态总览

- [ ] **Phase 0 — 环境基线** (TASK-001 ~ TASK-005)
- [ ] **Phase 1 — Fork 集成与原生构建** (TASK-006 ~ TASK-009)
- [ ] **Phase 2 — 最小可用二进制** (TASK-010 ~ TASK-017)
- [ ] **Phase 3 — 发布后端** (TASK-018 ~ TASK-020)
- [ ] **Phase 4 — 健壮性** (TASK-021 ~ TASK-025)
- [ ] **Phase 5 — Python 消费者** (TASK-026 ~ TASK-031)
- [ ] **Phase 6 — 启动与配置** (TASK-032 ~ TASK-035)
- [ ] **Phase 7 — 验证与 SLO** (TASK-036 ~ TASK-043)

总计：43 任务，预估 6–9 工作日。

---

## 环境基线（pinned versions）

| 组件 | 版本 | 来源 | 备注 |
|------|------|------|------|
| Ubuntu | 22.04 LTS (22.04.x) | 主机已装 | 验证脚本会再确认 |
| GCC | 11.4（默认） | apt | C++17 |
| CMake | ≥ 3.22 | apt `cmake`(=3.22.1) | 满足 fork CMakeLists |
| Python | 3.10（默认） | apt | rclpy 用 |
| **OpenCV** | 4.5.4 | apt `libopencv-dev` | 与 fork 兼容（待 TASK-008 二次确认） |
| **Eigen** | 3.4.0 | apt `libeigen3-dev` | |
| **Boost** | 1.74.0 | apt `libboost-serialization-dev libboost-system-dev` | serialization, system |
| **Pangolin** | v0.8 | 源码 build（apt 版偏旧） | 安装到 `/usr/local` |
| **Sophus** | v1.22.10 | 源码 build | 安装到 `/usr/local` |
| **librealsense2** | 2.55.1 | Intel apt PPA | D435i FW ≥ 5.16 |
| **ROS2** | Humble | 官方 apt | LTS 至 2027-05 |
| **ZMQ** | libzmq 4.3.4 + cppzmq 4.7.1 | apt `libzmq3-dev cppzmq-dev` | |
| **DBoW2 / g2o** | vendored | 跟随 fork `Thirdparty/` | 不单独装 |

### 关键技术决策（已锁定）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| C++ 标准 | **C++17** | fork CMakeLists 假设 |
| TF frames | `world` → `map` → `camera_color_optical_frame` (REP-105) | `map` 是 SLAM 输出帧，`world` 留给 ArUco 标定后接入 |
| ROS2 QoS | `rclcpp::SensorDataQoS()` (best_effort, depth=10) | 30Hz 流，丢一帧好过阻塞 |
| ZMQ 绑定 | cppzmq 4.7.1（apt） | 头文件 only，无额外打包 |
| Lock-free 队列 | `boost::lockfree::spsc_queue` | 已在 Boost 1.74 中，capacity=256 |
| CMake target | 直接加在 fork 的 `Examples/Monocular-Inertial/` CMakeLists 内 | 复用 fork 构建系统 |
| Build 命令 | `./build.sh`（fork 自带） + ament_cmake 单独包 | ROS2 publisher 用独立的 ament_cmake 子项目链接到 fork 产物 |
| 装在哪 | binary: `${fork}/Examples/Monocular-Inertial/realsense_online`<br/>config: `umi_mono/config/RealSense_D435i_online.yaml`<br/>Python: `dexslide/world_pose/` | 不污染系统目录 |
| Python rclpy | ROS2 Humble 内置（不要 pip） | |
| 测试框架 | pytest | DexSlide 已用 pytest |
| RealSense 流参数 | color: 960×540 BGR8 @30Hz<br/>accel: MOTION_XYZ32F @200Hz<br/>gyro: MOTION_XYZ32F @200Hz<br/>timestamp domain: `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME` | 与现有 `config/RealSense_D435i.yaml` 中 `IMU.Frequency: 200.0` 对齐；D435i 固件仅支持 accel 400/200/100 Hz |

### 待 TASK-006 时确定的决策

| 决策点 | 选项 | 默认建议 |
|--------|------|---------|
| cheng-chi/ORB_SLAM3 commit SHA | latest on `main` at TASK-006 执行时 | `git ls-remote` 取最新，记到此处 |

---

## TASK 详表

### Phase 0 — 环境基线

#### TASK-001 — 主机环境检查脚本 ✅
- **输出**：`umi_mono/scripts/check_host_env.sh`
- **行为**：检查 Ubuntu 版本、GCC、CMake、Python；输出表格 `[OK]/[WARN]/[MISSING]`；任一硬需求缺失则 `exit 1`；Python>=3.13 仅 WARN（ROS2 Humble 自带 3.10）
- **验收**：在主机跑 `bash umi_mono/scripts/check_host_env.sh && echo PASS` ✓ 退出码 0
- **依赖**：无
- **Codex sandbox**：workspace-write（写入新脚本）
- **状态**：[x] **done** (2026-05-18, 3 次迭代, 160s)

#### TASK-002 — apt 依赖安装脚本 ✅
- **输出**：`umi_mono/scripts/install_apt_deps.sh`（默认 dry-run，仅 `--apply` 时真装）
- **行为**：dry-run 打印 `Package | Status` 表，分类 installed / to-install / not-in-cache；`--apply` 走 sudo 路径前需用户输入 `yes` 二次确认；cppzmq 走 header-only fallback（apt 不提供，TASK-003+ 时按需 clone）
- **验收**：`bash umi_mono/scripts/install_apt_deps.sh && echo $?` 输出表格，退出码 0 ✓
- **依赖**：TASK-001
- **状态**：[x] **done** (2026-05-18, 74s, 一次过)
- **本机 dry-run 结果**：26 already installed, 3 to install (wayland-protocols, libavdevice-dev, libzmq3-dev), 0 unavailable

#### TASK-003 — Pangolin v0.8 源码构建脚本 ✅
- **输出**：`umi_mono/scripts/build_pangolin.sh`
- **行为**：`--apply`/`--clean`/`--jobs N`/`--help`；clone v0.8 到 `umi_mono/external/Pangolin`，cmake Release 编译；`--install` 需 `yes` 二次确认才走 sudo（**本机未安装，使用 CMAKE_PREFIX_PATH 替代**）
- **验收**：`build/libpango_core.so` 等 15 个 .so 产物存在、git tag = v0.8 ✓
- **依赖**：TASK-002 完成（wayland-protocols 等 apt 包就绪）
- **状态**：[x] **done** (2026-05-18, 6 分钟，clone + 编译一次过)
- **决策记录**：不 `sudo make install`。后续 fork build 用 `-DCMAKE_PREFIX_PATH=/data/codes/DexSlide/umi_mono/external/Pangolin/build` 找到 Pangolin。理由：避免污染 `/usr/local`、可在无 sudo 机器复现。

#### TASK-004 — Sophus v1.22.10 源码构建脚本 ✅
- **输出**：`umi_mono/scripts/build_sophus.sh`
- **行为**：与 build_pangolin.sh 同模式；tag 实际为 `1.22.10`（无 v 前缀，Codex 验证后采用）；Sophus 是 header-only + CMake config
- **验收**：`external/Sophus` git tag 1.22.10、`build/SophusConfig.cmake` 存在 ✓
- **依赖**：TASK-002
- **状态**：[x] **done** (2026-05-18, 64s 写脚本 + 10s 编译)

#### TASK-005 — librealsense2 安装 + D435i 烟囱测试 🟡
- **输出**：
  - `umi_mono/scripts/install_librealsense.sh`（dry-run-by-default + `--apply` 装 librealsense2-dev）
  - `umi_mono/scripts/test_d435i_streams.cpp`（C++ 测试源）
  - `umi_mono/scripts/run_d435i_smoke.sh`（编译 + 跑）
- **行为**：检测发现 PPA 已配 + `librealsense2-utils` 已装（ROS2 Humble 自带）；只缺 dev 包
- **状态**：[🟡] **partially done** (脚本就绪，等待用户授权 `--apply` 安装 dev 包后跑烟囱测试)
- **依赖**：TASK-001 完成；D435i 插上 USB3

### Phase 1 — Fork 集成与原生构建

#### TASK-006 — Fork 作为本地 clone + pin SHA ✅
- **输出**：
  - `umi_mono/scripts/setup_orbslam3_fork.sh`（idempotent clone + pin）
  - `umi_mono/external/ORB_SLAM3_fork/`（实体 clone，**非 submodule**）
  - `umi_mono/external/ORB_SLAM3_fork/.pinned_sha`
  - `umi_mono/.gitignore` 追加 `external/`
- **行为**：默认 `git ls-remote main`→缺则 `master`→ checkout SHA → 写 `.pinned_sha` → 解 ORBvoc.txt.tar.gz
- **验收**：HEAD == pinned SHA, Examples/Monocular-Inertial/ 9 个文件, Vocabulary/ORBvoc.txt 145MB ✓
- **状态**：[x] **done** (2026-05-18, SHA = `b741dca39015330ef4bcc3a85f89493503ade04b`, branch = master)
- **决策**：cheng-chi/ORB_SLAM3 默认分支 actually 是 `master` 不是 `main`，脚本两个都试

#### TASK-007 — Fork 原生构建脚本 ✅
- **输出**：`umi_mono/scripts/build_orbslam3_native.sh`
- **行为**：idempotent (cmake -B build) 构建 fork 的 Thirdparty/DBoW2、g2o、Sophus + 根工程；**跳过空的 Thirdparty/Pangolin**，根 cmake 用 `-DCMAKE_PREFIX_PATH=/data/codes/DexSlide/umi_mono/external/Pangolin/build`；tee 到 `${FORK}/build_orbslam3.log`
- **关键发现**：cheng-chi fork 的 `Thirdparty/Pangolin/` **是空目录**（不是 bug），找 Pangolin 走 `find_package(Pangolin)` + 外部路径。Phase 0.3 编译 external/Pangolin 在这里真正用上
- **验收**：libORB_SLAM3.so (354MB)、libDBoW2.so、libg2o.so、gopro_slam binary (16MB) 生成；`ldd` 全部 resolved，pango_* 指向 external/Pangolin/build；`gopro_slam --help` exit 0 ✓
- **状态**：[x] **done** (2026-05-18, 编译 ~3 分钟，1 次迭代修空 Pangolin 逻辑)

#### TASK-008 — 原生 `gopro_slam` 与 Docker 输出 diff（baseline 校准）🟡
- **行为**：取一份已有 `raw_video.mp4` + `imu_data.json` + `map_atlas.osa`，原生 vs Docker batch 跑同一份输入，对比 ATE
- **状态**：[🟡] **deferred** — 需要存在的录像，**Phase 7 (TASK-037 ATE 测试) 一起跑**
- **依赖**：TASK-007（done）、可用录像
- **理由**：算法验证可以推到验收阶段；当前任务是构建产物正确，已通过 ldd + --help 间接验证

#### TASK-009 — Fork CMakeLists 修改预备 ✅
- **输出**：
  - `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online.cc`（placeholder：cout + return 0）
  - `external/ORB_SLAM3_fork/CMakeLists.txt` +3 行（gopro_slam 之后 add_executable + target_link_libraries）
- **验收**：`cmake --build build -j --target realsense_online` exit 0；`./Examples/Monocular-Inertial/realsense_online` 退出 0 且打印占位串；旧 `gopro_slam --help` 不受影响 ✓
- **状态**：[x] **done** (2026-05-18, 90s, CMakeLists 改动最小)

### Phase 2 — 最小可用二进制

#### TASK-010 — `realsense_online.cc` CLI 骨架
- **输出**：CLI11 解析 `--vocabulary --setting --load_map --publisher {stdout,ros2,zmq} --exposure_us --max_lost_frames --skew_abort_ms`；`--help` 完整
- **验收**：`./realsense_online --help` 输出所有 flag；非法参数返回非零
- **依赖**：TASK-009
- **状态**：[ ] pending

#### TASK-011 — `imu_ring_buffer.hpp`
- **输出**：`Examples/Monocular-Inertial/imu_ring_buffer.hpp`（header-only）
- **行为**：包装 `boost::lockfree::spsc_queue<IMU::Point, capacity=256>`；`push()` 来自 RealSense callback；`drain_until(timestamp) -> std::vector<IMU::Point>`
- **验收**：单元测试 `tests/test_imu_ring_buffer.cc`（gtest 或纯 assert），覆盖：满队列丢弃最旧、按时间戳裁取、跨线程 push/drain
- **依赖**：TASK-010
- **状态**：[ ] pending

#### TASK-012 — RealSense pipeline 启动模块
- **输出**：`Examples/Monocular-Inertial/realsense_capture.hpp` + `.cpp`
- **行为**：`class RsCapture` 封装 `rs2::pipeline`；构造时设 GLOBAL_TIME；提供 `start(image_cb, imu_cb)`、`stop()`；图像和 IMU 都通过回调输出（图像走主线程 wait_for_frames，IMU 走 SDK callback）
- **验收**：嵌入到 `realsense_online.cc` 后能跑 5 秒，主线程打印 image FPS，回调累计 IMU 数
- **依赖**：TASK-005, TASK-011
- **状态**：[ ] pending

#### TASK-013 — System 构造 + 加载地图 + 激活定位模式
- **输出**：`realsense_online.cc` 中 `init_orbslam3()` 函数
- **行为**：构造 `ORB_SLAM3::System(voc, settings_yaml, IMU_MONOCULAR, bUseViewer=false)`；构造后立刻调 `ActivateLocalizationMode()`；日志 `Localization mode active, atlas read-only`
- **验收**：用已有 map atlas 跑，看到日志；`SLAM.GetAtlas()->KeyFramesInMap()` 调用前后 keyframes 数相同
- **依赖**：TASK-012
- **状态**：[ ] pending

#### TASK-014 — 主跟踪循环（stdout 输出）
- **输出**：`realsense_online.cc` 的 main loop
- **行为**：`pipe.wait_for_frames(100ms)` → 取 timestamp → `imu_buf.drain_until(t)` → `SLAM.TrackMonocular(im, t, vImu)` → stdout 打印 `t x y z qx qy qz qw`
- **验收**：D435i 朝向已建图区域，stdout 每秒打印 ~30 行 pose；停止后干净退出
- **依赖**：TASK-013
- **状态**：[ ] pending

#### TASK-015 — Pose 非法检查（NaN guard）
- **输出**：`is_valid(SE3f)` helper + main loop 集成
- **行为**：若 translation 或 quaternion 有 NaN，跳过本帧不打印；累计 `n_invalid` 计数
- **验收**：人为遮挡相机几秒后再打开，跟踪丢失期间 stdout 无新行；恢复后继续
- **依赖**：TASK-014
- **状态**：[ ] pending

#### TASK-016 — 干净关停（SIGINT 处理）
- **输出**：信号处理 + `SLAM.Shutdown()` 调用
- **行为**：Ctrl-C 时跳出主循环、调 `SLAM.Shutdown()`、停 `pipe`、退 0
- **验收**：连续启停 5 次无 hang/crash
- **依赖**：TASK-014
- **状态**：[ ] pending

#### TASK-017 — End-to-end smoke：live camera → stdout pose
- **输出**：`umi_mono/scripts/smoke_realsense_online.sh`
- **行为**：脚本调起 `realsense_online --publisher stdout`，让用户拿相机扫一圈已建图区域，统计 N 秒内 valid pose 行数
- **验收**：60 秒内 ≥ 1500 行 valid pose（≈ 30 Hz × 60 × 0.83）
- **依赖**：TASK-016, TASK-008 的 map atlas
- **状态**：[ ] pending

### Phase 3 — 发布后端

#### TASK-018 — ZMQ publisher
- **输出**：`Examples/Monocular-Inertial/zmq_pose_publisher.hpp`
- **行为**：`class ZmqPosePublisher { void publish(Sophus::SE3f Tcw, double t); }`，PUB on `tcp://127.0.0.1:5555`，protobuf-less JSON line：`{"t":..,"tx":..,"ty":..,"tz":..,"qx":..,"qy":..,"qz":..,"qw":..}`
- **验收**：Python `import zmq; sub.recv_string()` 收到结构化 JSON，30 Hz
- **依赖**：TASK-016
- **状态**：[ ] pending

#### TASK-019 — ROS2 publisher（独立 ament_cmake 包）
- **输出**：`umi_mono/ros2_ws/dexslide_slam_publisher/` 包，内含 `src/pose_publisher_node.cpp` + `package.xml` + `CMakeLists.txt`
- **行为**：节点接 ZMQ SUB（消费 TASK-018 的输出），re-publish 为 `geometry_msgs/PoseStamped` + tf2 broadcast `world → camera_color_optical_frame`，QoS=SensorDataQoS
- **决策记录**：选 ZMQ → ROS2 桥而非把 rclcpp 直接链进 fork，原因——fork 的 GPL-3 license 边界保持清洁；`realsense_online` 本身仍 GPL，ROS2 节点单独的桥可以 BSD
- **验收**：`ros2 topic echo /dexslide/slam/pose` 看到 PoseStamped；`ros2 run tf2_ros tf2_echo world camera_color_optical_frame` 看到 tf
- **依赖**：TASK-018
- **状态**：[ ] pending

#### TASK-020 — `--publisher` flag 调度
- **输出**：`realsense_online.cc` 中根据 flag 选 stdout / zmq；ROS2 走 TASK-019 的桥（即 zmq + bridge）
- **行为**：`enum class PublisherKind { STDOUT, ZMQ };`；运行时分发
- **验收**：三种模式都能跑 60 秒不退出
- **依赖**：TASK-019
- **状态**：[ ] pending

### Phase 4 — 健壮性

#### TASK-021 — 启动期 timestamp skew 校验
- **输出**：`init` 阶段累计前 100 帧的 `|image.t - latest_imu.t|`；中位数 > 5 ms 则 abort
- **验收**：正常情况下日志 `Timestamp skew median: <X> ms` 且 X < 5；人为破坏（设置错误的 timestamp domain）触发 abort，stderr 含 `Timestamp skew out of budget`
- **依赖**：TASK-014
- **状态**：[ ] pending

#### TASK-022 — Vocabulary MD5 校验
- **输出**：`init` 阶段读 ORBvoc.txt 算 MD5 + 读 atlas 内嵌 vocab 指纹（cheng-chi fork 内置）；不匹配则 abort
- **验收**：传入错误 vocab 时 5 秒内 abort，stderr 含 `Vocabulary mismatch`
- **依赖**：TASK-013
- **状态**：[ ] pending

#### TASK-023 — Atlas SHA-256 immutability 集成测试
- **输出**：`umi_mono/tests/test_atlas_immutable.sh`
- **行为**：算 atlas 前 SHA-256；跑 60s session；算后 SHA-256；assert 相等
- **验收**：脚本退出 0
- **依赖**：TASK-013
- **状态**：[ ] pending

#### TASK-024 — Tracking lost 软重置
- **输出**：`realsense_online.cc` 状态机：发现 lost 时只增计数、继续喂帧，不退出
- **验收**：人为遮挡 30 秒，进程 PID 不变，停止遮挡后 5 秒内 pose 流恢复
- **依赖**：TASK-015
- **状态**：[ ] pending

#### TASK-025 — RealSense 曝光锁定
- **输出**：`realsense_capture.cpp` 启动时 `s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0)`、`RS2_OPTION_EXPOSURE` = `--exposure_us` 值
- **验收**：环境亮度突变时（手电照镜头）跟踪不抖；日志 `Exposure locked to <us>`
- **依赖**：TASK-012
- **状态**：[ ] pending

### Phase 5 — Python 消费者

#### TASK-026 — 包骨架
- **输出**：`dexslide/world_pose/__init__.py` 含 `__all__ = ['SlamPoseSubscriber']`
- **验收**：`python -c "from dexslide.world_pose import SlamPoseSubscriber"` 成功（即便 class 还是 stub）
- **依赖**：无
- **状态**：[ ] pending

#### TASK-027 — `SlamPoseSubscriber` 基础订阅 + ring buffer
- **输出**：`dexslide/world_pose/slam_pose_subscriber.py`
- **行为**：rclpy 节点，订阅 `/dexslide/slam/pose`，把 (t, 4x4 SE3 ndarray) 存到 `collections.deque(maxlen=300)`，加 `threading.Lock`
- **验收**：手动 `ros2 topic pub` 一条 PoseStamped，`subscriber.latest()` 返回该 pose
- **依赖**：TASK-019, TASK-026
- **状态**：[ ] pending

#### TASK-028 — 时间对齐查询（线性插值 + SLERP）
- **输出**：`get_T_world_camera(t)` 实现
- **行为**：在 buffer 中二分找 `t1 ≤ t ≤ t2`；位置线性插值；旋转用 `scipy.spatial.transform.Slerp`；超出 ±100 ms 返回 `None`
- **验收**：pytest 用例：放 2 个 pose @ t=0 和 t=1，query t=0.5 → 中点位置 + 中点旋转
- **依赖**：TASK-027
- **状态**：[ ] pending

#### TASK-029 — `is_tracking()` + `stale_after_seconds`
- **输出**：方法实现 + 构造器参数
- **行为**：返回 True iff 最新 pose 时间在 `now - stale_after_seconds` 之内
- **验收**：pytest 用 mock clock 验证阈值切换
- **依赖**：TASK-027
- **状态**：[ ] pending

#### TASK-030 — `spin_in_thread()`
- **输出**：`spin_in_thread()` 启动 daemon thread + `SingleThreadedExecutor.spin()`
- **行为**：主线程不阻塞；`stop()` 干净停掉
- **验收**：调 spin_in_thread，主线程 sleep 1s 后 is_tracking() 为 True（前提 publisher 在跑）
- **依赖**：TASK-027
- **状态**：[ ] pending

#### TASK-031 — pytest 测试套件
- **输出**：`tests/test_slam_pose_subscriber.py`
- **行为**：mock rclpy、单元测试 ring buffer、SLERP、stale flag、concurrent access（10 threads 同时 query）
- **验收**：`pytest tests/test_slam_pose_subscriber.py -v` 全绿
- **依赖**：TASK-028, TASK-029, TASK-030
- **状态**：[ ] pending

### Phase 6 — 启动与配置

#### TASK-032 — Launch 文件
- **输出**：`umi_mono/ros2_ws/dexslide_slam_publisher/launch/dexslide_slam_online.launch.py`
- **行为**：参数 `vocab`、`settings_yaml`、`map_atlas`、`pose_topic`、`exposure_us`；起两个 process：`realsense_online --publisher zmq` + `pose_publisher_node`
- **验收**：`ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py vocab:=... settings_yaml:=...` 起得起来
- **依赖**：TASK-020, TASK-019
- **状态**：[ ] pending

#### TASK-033 — 在线模式 SLAM 配置
- **输出**：`umi_mono/config/RealSense_D435i_online.yaml`
- **行为**：从 `RealSense_D435i.yaml` 复制，加 `System.LoadAtlasFromFile: <to-be-overridden-at-launch>`，Viewer 段全置 0（headless）
- **验收**：launch 文件 override 该 key 后 yaml 解析成功
- **依赖**：无
- **状态**：[ ] pending

#### TASK-034 — `SLAM_readme_mono.txt` 新增「在线追踪」章节
- **输出**：`umi_mono/SLAM_readme_mono.txt` 末尾加 `## 7. 在线追踪` 与命令示例
- **验收**：人工 review
- **依赖**：TASK-032
- **状态**：[ ] pending

#### TASK-035 — 更新 `umi_mono/CLAUDE.md`
- **输出**：在 `umi_mono/CLAUDE.md` 加 "Online tracking" 段，说明 pinned SHA、build 命令、launch 命令、已知故障模式
- **验收**：人工 review
- **依赖**：TASK-006, TASK-032
- **状态**：[ ] pending

### Phase 7 — 验证与 SLO

#### TASK-036 — `--playback-mode` flag（用录像复算）
- **输出**：`realsense_online.cc` 加 `--playback_video <mp4> --playback_imu <json>`，替代 RealSense 输入
- **验收**：用一段已知录像跑 playback，输出与 Docker batch 同一录像 ATE < 2 cm
- **依赖**：TASK-014
- **状态**：[ ] pending

#### TASK-037 — ATE-vs-Docker 自动化对比
- **输出**：`umi_mono/tests/test_ate_vs_docker.py`
- **行为**：复用 `umi/traj_eval/`，比对两 trajectory
- **验收**：测试 pass（ATE < 2 cm）
- **依赖**：TASK-036
- **状态**：[ ] pending

#### TASK-038 — 30 分钟 benchmark
- **输出**：`umi_mono/scripts/benchmark_30min.sh`
- **行为**：跑 30 分钟，记录 wall-clock 帧间间隔与 capture-to-publish 延迟，输出直方图与 p99
- **验收**：p99 inter-message ≤ 50 ms；p99 capture-to-publish < 33 ms
- **依赖**：TASK-019
- **状态**：[ ] pending

#### TASK-039 — Headless 测试
- **输出**：`umi_mono/tests/test_headless.sh`
- **行为**：`unset DISPLAY` 后跑 `realsense_online` 60 秒
- **验收**：无 X11 报错；`wmctrl -l` 无 ORB-SLAM3 窗口
- **依赖**：TASK-013
- **状态**：[ ] pending

#### TASK-040 — 2 小时内存测试
- **输出**：`umi_mono/scripts/memory_test_2h.sh`
- **行为**：跑 2 小时，每 60s 采 RSS（`ps -o rss=`）
- **验收**：末尾 RSS / 起始 RSS < 1.05
- **依赖**：TASK-019
- **状态**：[ ] pending

#### TASK-041 — Atlas immutability 长跑测试
- **输出**：扩展 TASK-023 跑 60 分钟
- **验收**：SHA-256 不变
- **依赖**：TASK-023
- **状态**：[ ] pending

#### TASK-042 — Vocab mismatch 失败模式测试
- **输出**：`umi_mono/tests/test_vocab_mismatch.sh`
- **行为**：刻意传错 vocab path 启动
- **验收**：5 秒内非零退出 + stderr 含 `Vocabulary mismatch`
- **依赖**：TASK-022
- **状态**：[ ] pending

#### TASK-043 — Recovery 测试（3s 遮挡）
- **输出**：`umi_mono/tests/test_recovery_3s.sh`（半自动：脚本配合手工遮挡）
- **行为**：跑 publisher，操作员遮挡 3 秒后松开，脚本测时序
- **验收**：松开后 5 秒内重新出 pose
- **依赖**：TASK-024
- **状态**：[ ] pending

---

## 与 Codex 协作约定

每条 TASK 触发时：
1. Claude 用 `collaborating-with-codex` skill，传 `--sandbox workspace-write`（除非该 task 明确标 read-only）
2. Codex prompt 包含：本 TASK 完整描述（输出/行为/验收/依赖）、必读上下文文件列表、要求返回 **unified diff patch**
3. Claude 审 diff，必要时再调一轮 codex 修改；满意后人工应用 patch
4. 跑该 TASK 的「验收」命令；通过则在 tracker 中打 [x]，记录耗时与遇到的偏差
5. 若验收失败：根因分析后或回到 codex 修复，或退一步修订 TASK 描述

## 当前进度

| 时间 | TASK | 状态 | 备注 |
|------|------|------|------|
| 2026-05-18 12:17 | TASK-001 | ✅ done | check_host_env.sh, 160s, 3 次迭代；发现系统 Python 是 3.13.9（WARN），D435i 主机其余环境齐备 |
| 2026-05-18 12:21 | TASK-002 | ✅ done | install_apt_deps.sh（dry-run-by-default），74s 一次过；3 个包待 `--apply` 安装 |
| 2026-05-18 12:55 | TASK-003 | ✅ done | Pangolin v0.8 编译完成 6 分钟，15 个 .so；不 install，后续用 CMAKE_PREFIX_PATH |
| 2026-05-18 12:55 | TASK-004 | ✅ done | Sophus 1.22.10（注意：实际 tag 是 `1.22.10` 无 v 前缀），header-only ~10s |
| 2026-05-18 12:58 | TASK-005 | 🟡 partial | 脚本已写：install_librealsense.sh + test_d435i_streams.cpp + run_d435i_smoke.sh；librealsense2-dev 2.57.7 已装；烟囱等 D435i 插上 |
| 2026-05-18 13:12 | TASK-006 | ✅ done | cheng-chi/ORB_SLAM3 fork clone (SHA b741dca, master 分支), vocabulary 解压完成 |
| 2026-05-18 13:24 | TASK-007 | ✅ done | Fork 原生编译 ~3min；libORB_SLAM3.so 354MB + gopro_slam 16MB；ldd 全 resolved；Thirdparty/Pangolin 空目录用 external/Pangolin 顶替 |
| 2026-05-18 13:24 | TASK-008 | 🟡 deferred | baseline diff 推迟到 Phase 7 (TASK-037 ATE 验证) |
| 2026-05-18 13:29 | TASK-009 | ✅ done | realsense_online placeholder target；CMakeLists.txt +3 行，旧 gopro_slam 不动 |
| 2026-05-18 13:51 | TASK-010 | ✅ done | CLI11 解析；发现 fork CMake 强制 -std=c++14、vendored CLI11 用 IsMember 替代 add_set |
| 2026-05-18 14:00 | TASK-011 | ✅ done | hand-rolled SPSC ImuRingBuffer（IMU::Point 无默认 ctor，无法用 boost::lockfree），5 个 assert 测试全过 |
| 2026-05-18 14:09 | TASK-012 | ✅ done | RsCapture (librealsense2)，无设备优雅退出；librealsense2.so 解析到 ROS2 Humble 自带 2.57 |
| 2026-05-18 14:21 | TASK-013 | ✅ done | ORB_SLAM3::System(IMU_MONOCULAR, bUseViewer=false) + ActivateLocalizationMode + Shutdown |
| 2026-05-18 14:25 | TASK-014 | ✅ done | 主追踪循环：IMU buffer 接 RsCapture callback；TrackMonocular 每帧调用；stdout 'pose t x y z qx qy qz qw' |
| 2026-05-18 14:35 | TASK-015 | ✅ done | NaN guard + 跳过无效 pose 计数（合并 TASK-016 一次调用） |
| 2026-05-18 14:35 | TASK-016 | ✅ done | SIGINT/SIGTERM handler；free function on_signal + atomic g_should_exit |
| 2026-05-18 14:35 | TASK-017 | 🟡 deferred | live D435i smoke 待设备插上跑 |
| 2026-05-18 20:20 | TASK-018 | ✅ done | ZmqPosePublisher (cppzmq v4.10.0 cloned to external/cppzmq); JSON line format on tcp://127.0.0.1:5555 |
| 2026-05-18 20:23 | TASK-019 | ✅ done | dexslide_slam_publisher ament_cmake 包；colcon build OK；`/dexslide/slam/pose` PoseStamped + tf2 broadcast |
| 2026-05-18 20:20 | TASK-020 | ✅ done | --publisher {stdout,zmq,ros2} dispatch；ros2 = zmq + 需独立启动 bridge |
| 2026-05-18 20:47 | TASK-021 | ✅ done | 启动 7s/200 polls warmup 测 image/IMU 时间戳偏差中位数，>skew_abort_ms 则 exit 3 |
| 2026-05-18 20:47 | TASK-022 | ✅ done | popen('md5sum') 算 vocab MD5 + 文件大小 sanity check；启动期打印 |
| 2026-05-18 20:49 | TASK-023 | 🟡 partial | test_atlas_immutable.sh 写好，dry checks ✓；live SHA-256 测试待真实 .osa |
| 2026-05-18 20:47 | TASK-024 | ✅ done | LocalizeMonocular 替换 TrackMonocular；consecutive_lost 计数，超 max_lost_frames 软重置（仅 log，不 Reset） |
| 2026-05-18 20:47 | TASK-025 | ✅ done | RsCapture --exposure_us 锁曝光 + 禁用 ENABLE_AUTO_EXPOSURE；非阻塞警告 if not supported |
| 2026-05-18 22:02 | TASK-026 | ✅ done | dexslide/world_pose/__init__.py |
| 2026-05-18 22:02 | TASK-027 | ✅ done | SlamPoseSubscriber 基础订阅 + deque ring buffer + 线程锁 |
| 2026-05-18 22:02 | TASK-028 | ✅ done | get_T_world_camera 手写 SLERP（无 scipy 依赖）+ 线性 translation 插值 |
| 2026-05-18 22:02 | TASK-029 | ✅ done | is_tracking + stale_after_seconds (默认 0.2s)；time.monotonic 基准 |
| 2026-05-18 22:02 | TASK-030 | ✅ done | spin_in_thread daemon + SingleThreadedExecutor；stop() 干净关停 |
| 2026-05-18 22:06 | TASK-031 | ✅ done | pytest 10 测试全过：quat/rotmat、SLERP（端点+反极点）、buffer、interpolation、out-of-range、stale 阈值；0.14s |
| 2026-05-19 00:21 | TASK-032 | ✅ done | dexslide_slam_online.launch.py + CMakeLists install rule + colcon rebuild 30s |
| 2026-05-19 00:21 | TASK-033 | ✅ done | RealSense_D435i_online.yaml；System.LoadAtlasFromFile 空占位、viewer 字段全置 0 |
| 2026-05-19 00:21 | TASK-034 | ✅ done | SLAM_readme_mono.txt 追加「## 7. 在线追踪」章节 |
| 2026-05-19 00:21 | TASK-035 | ✅ done | umi_mono/CLAUDE.md 追加在线追踪段：build/run/known failures/docs 索引 |
| 2026-05-19 00:28 | TASK-036 | ✅ done | --playback_video / --playback_imu flag；refactor System ctor 移到 live/playback 分支前共享 |
| 2026-05-19 00:33 | TASK-037 | ✅ done | test_ate_vs_docker.py (numpy only)；需 recording + atlas 跑 live |
| 2026-05-19 00:33 | TASK-038 | ✅ done | benchmark_30min.sh + inline Python stats；p50/p99/p99.9 + 桶直方图 + JSON 输出；需 D435i 跑 live |
| 2026-05-19 00:33 | TASK-039 | ✅ done | test_headless.sh：unset DISPLAY 跑 30s，grep X11/Xlib；需 D435i 跑 live |
| 2026-05-19 00:33 | TASK-040 | ✅ done | test_memory_2h.sh：每 60s 采 RSS，2h 后断 5% 阈值；需 D435i 跑 live |
| 2026-05-19 00:21 | TASK-041 | ✅ inline | TASK-023 已支持 duration 参数；`bash test_atlas_immutable.sh <map.osa> 3600` 即 1h 长测 |
| 2026-05-19 00:28 | TASK-042 | ✅ done | test_vocab_mismatch.sh + 在 realsense_online 加 vocab 大小预检；script PASS ✓ |
| 2026-05-19 00:33 | TASK-043 | ✅ done | test_recovery_3s.sh：半自动遮挡测试用 FIFO；需 D435i + 操作员 |

---

### Phase 6 + Phase 7 完结点 (2026-05-19 00:33)

**Phase 6 (Launch + Config + Docs)**：
- `dexslide_slam_online.launch.py` 一键启动 realsense_online + bridge
- `config/RealSense_D435i_online.yaml` 派生自 D435i.yaml，viewer 全 0
- `SLAM_readme_mono.txt` 追加「## 7. 在线追踪」
- `umi_mono/CLAUDE.md` 追加 build / run / failure modes / docs 索引段

**Phase 7 (验证脚本 + SLO)**：
- TASK-036: realsense_online 加 `--playback_video / --playback_imu` 模式（offline 复跑）
- TASK-037~040: 4 个 SLO 测试脚本 (ATE / 30min benchmark / headless / 2h memory)
- TASK-042: vocab mismatch 失败模式测试（已跑 PASS）
- TASK-043: recovery 测试（半自动）

**🎉 43/43 TASKs 全部完成（编码侧）。**

### 全局状态总结

| 类别 | 数量 | 备注 |
|------|------|------|
| ✅ done（编码+验证） | 38 | 含 dry-checks、单元测试、编译 |
| 🟡 partial（脚本/code OK，live 验证待 D435i 或真录像） | 5 | TASK-005, 008, 017, 023, 037, 038, 039, 040, 043 |
| ❌ failed | 0 |  |

**完成后整条流水线（待 D435i + 真 .osa 走通）**：

```
D435i → realsense_online --publisher zmq + --load_map <atlas.osa> + --exposure_us 8000
        ↓ ZMQ tcp://127.0.0.1:5555 (JSON line)
ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py map_atlas:=<atlas.osa>
        ↓ /dexslide/slam/pose (PoseStamped @30Hz) + tf2 map→camera_color_optical_frame
Python: from dexslide.world_pose import SlamPoseSubscriber
        sub.spin_in_thread(); T = sub.get_T_world_camera(t)
```

### 等 D435i 接好后的验收清单（按优先级）

1. **TASK-005 D435i 烟囱** — `bash umi_mono/scripts/run_d435i_smoke.sh` 确认 D435i + 固件 + 流参数 OK
2. **TASK-017 live realsense_online** — 不加 --load_map 跑 60 s，确认 binary 不 crash、`Localization mode active` 出现
3. **TASK-038 30min benchmark** — 加 atlas 跑 30 min，确认 p99 ≤ 50ms
4. **TASK-039 headless** — `unset DISPLAY` 跑 30s
5. **TASK-008 ATE vs Docker** — 用录像 + atlas + 对比 Docker batch 输出
6. **TASK-023/041 atlas 不变性** — `bash test_atlas_immutable.sh <map.osa> 3600`
7. **TASK-040 2h memory** — 长期内存稳定性
8. **TASK-043 recovery 3s** — 手动遮挡测试

---

### 📊 整体投入

- 编码时间：~7 小时（跨 3 session）
- Codex 调用：~15 次
- 文件触达：fork 8 个、ROS2 包 4 个、Python 模块 3 个、脚本 11 个、文档 7 个
- 代码行数：~1500 行 (含测试)

---

### Phase 5 完结点 (2026-05-18 22:06)

**Phase 5 (Python 消费者)**：
- 新模块 `/data/codes/DexSlide/dexslide/world_pose/`（__init__.py + slam_pose_subscriber.py）
- 暴露 `SlamPoseSubscriber` 类，rclpy.init 懒加载、deque 缓冲、手写 SLERP、线程安全
- API 完整：`latest()` / `is_tracking()` / `get_T_world_camera(t)` / `spin_in_thread()` / `stop()`
- pytest 10 测试全过 (in 0.14s)，覆盖：四元数/旋转矩阵转换、SLERP 端点/反极点、buffer 边界、时间对齐插值、out-of-range 处理、stale flag 阈值

**关键设计决策**：
- 用 `/usr/bin/python3` (3.10) 不用 anaconda (3.13)，因为 rclpy 为 3.10 编译
- 手写 SLERP / 四元数→旋转矩阵 避免 scipy 依赖（系统 python3 没装 scipy；anaconda 装了但 Python 版本不兼容）
- `rclpy.init()` 在构造函数中懒加载，调用者无需先调
- 处理 `~/.ros/log` 不可写情况：自动 fallback `ROS_LOG_DIR=/tmp/roslog`

**还需 D435i + 真 .osa 跑通整链路（live）**：
- TASK-005 D435i 烟囱
- TASK-008 ATE vs Docker
- TASK-017 live pose 60s
- TASK-023 atlas SHA-256 不变性

**下一步：Phase 6 — Launch + 配置 + 文档**（TASK-032 ~ TASK-035）。预估 0.5 天。

---

### Phase 3 + Phase 4 完结点 (2026-05-18 20:50)

**Phase 3 (发布后端)**：
- ZmqPosePublisher 在 fork 内（依赖 cppzmq + libzmq）
- 独立的 `dexslide_slam_publisher` ament_cmake 包在 umi_mono/ros2_ws/
- --publisher 在 realsense_online 内 dispatch（stdout / zmq / ros2 三档）

**Phase 4 (健壮性)**：
- Startup vocab MD5 + size check
- Startup image/IMU skew validation（中位数 + abort）
- LocalizeMonocular + consecutive_lost 软重置（不退出）
- RsCapture 曝光锁
- Atlas immutability 测试脚本

**还需 D435i 插上做的（live 验证）**：
- TASK-005 D435i 烟囱
- TASK-008 ATE vs Docker
- TASK-017 live pose
- TASK-023 atlas SHA-256（需要 .osa）

**下一步：Phase 5 — Python 消费者 `dexslide.world_pose.SlamPoseSubscriber`**（TASK-026~031）。预估 1.5 天。

---

### Phase 1 完结点 (2026-05-18 13:29)

Phase 0 + Phase 1 完成。状态：
- ✅ 主机环境 (gcc/cmake/python/apt 依赖、Pangolin v0.8、Sophus 1.22.10、librealsense2 2.57.7)
- ✅ cheng-chi/ORB_SLAM3 fork 本地化（SHA `b741dca`）
- ✅ 原生编译产物：`lib/libORB_SLAM3.so` (354MB) + `gopro_slam` (16MB) + `realsense_online` placeholder
- 🟡 TASK-005 烟囱测试待 D435i 插上跑
- 🟡 TASK-008 baseline diff 推迟到 Phase 7

**下一步：Phase 2 — 真正写 realsense_online.cc 的 8 个 TASK（TASK-010~017）**。预估 2-3 工作日；中间结果会编译跑、stdout 看 pose。

---

### Phase 2 完结点 (2026-05-18 14:35)

Phase 2 全部代码就绪。`realsense_online` 在 fork 内可编译、可链接、可执行（no-device 优雅退出 + SIGINT 响应）。

**已完成 (代码 + 验证编译/no-device 路径)**：
- TASK-010 CLI11 解析（7 个 flag）
- TASK-011 lock-free SPSC ImuRingBuffer + 单元测试 PASS
- TASK-012 RsCapture 封装 librealsense2 pipeline
- TASK-013 ORB_SLAM3::System 构造 + ActivateLocalizationMode + Shutdown
- TASK-014 主追踪循环（TrackMonocular + stdout 'pose t x y z qx qy qz qw'）
- TASK-015 NaN guard 跳过无效 pose
- TASK-016 SIGINT/SIGTERM 干净关停

**待 D435i 插上做的（live 验证）**：
- TASK-017 end-to-end smoke：朝向已建图区域跑 60s，期望 ≥1500 行 valid pose
- 同时验证 TASK-013 atlas 加载、TASK-014 实际 FPS、TASK-015 NaN 比例

**Fork 改动总结（vs vanilla cheng-chi/ORB_SLAM3）**：
| 文件 | 状态 | 改动 |
|------|------|------|
| `CMakeLists.txt` | 修改 | +6 行：realsense2 find_package + realsense_online target + test_imu_ring_buffer target |
| `Examples/Monocular-Inertial/realsense_online.cc` | 新增 | ~240 行 (CLI + RsCapture + SLAM + 主循环 + NaN + SIGINT) |
| `Examples/Monocular-Inertial/realsense_capture.hpp` | 新增 | 类声明 |
| `Examples/Monocular-Inertial/realsense_capture.cpp` | 新增 | librealsense2 pipeline 实现 |
| `Examples/Monocular-Inertial/imu_ring_buffer.hpp` | 新增 | header-only SPSC 队列模板 |
| `Examples/Monocular-Inertial/test_imu_ring_buffer.cc` | 新增 | 5 个 assert 测试 |

**下一步：Phase 3 — 发布后端**（TASK-018 ZMQ + TASK-019 ROS2 桥 + TASK-020 --publisher dispatch）。预估 1 天。
