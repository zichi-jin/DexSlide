# 离线建图 + 在线追踪方案研究报告

> 作者：研究调研产物 · 日期：2026-05-16
> 范围：将 `umi_mono` 当前的离线 SLAM 流水线（Docker / `chicheng/orb_slam3:latest`）改造为「离线建图 + 在线追踪」架构
> 部署约束：本机 Linux Ubuntu 22.04，无 Docker，单目+IMU（RealSense D435i 960×540 @30Hz + IMU @200Hz），延迟预算 < 33 ms / 帧

---

## 0. TL;DR（一句话结论）

**可以**继续使用当前的 ORB-SLAM3（cheng-chi fork）算法，**不需要更换 SLAM**。最小改造路径是：**在 fork 内新增一个本机可执行文件 `realsense_online`**，复用现有 `--load_map` 路径，调用 `System::ActivateLocalizationMode()` + `System::TrackMonocular()` 公开 API，发布 `Sophus::SE3f` 位姿到 ROS2/共享内存。**估算工作量 ~600 LOC、2–4 天可工程可用。**不推荐改 Docker 直接喂流，也不推荐换 stella_vslam（IMU 支持 5 年未合并）。

---

## 1. 现状分析

### 1.1 当前算法栈

| 组件 | 版本/来源 | 备注 |
|------|----------|------|
| SLAM 算法 | **ORB-SLAM3** (Monocular-Inertial) | 视觉 + IMU 紧耦合 |
| Fork | `cheng-chi/ORB_SLAM3` ← `urbste/ORB_SLAM3` ← `UZ-SLAMLab/ORB_SLAM3` | cheng-chi 是 snapshot 分支，0 open issues |
| 二进制 | `/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam` | 自定义 C++ 入口 |
| 部署 | Docker `chicheng/orb_slam3:latest` | 通过 `subprocess.run(['docker', 'run', ...])` 调起 |
| 地图格式 | `map_atlas.osa`（Boost binary archive） | 多地图 Atlas |
| 相机 | RealSense D435i, 960×540 @30Hz + IMU @200Hz | `config/RealSense_D435i.yaml` |

### 1.2 当前 Pipeline 流程（关键的两个阶段）

```
Stage 02: 02_create_map.py
  → docker run gopro_slam --input_video raw_video.mp4 --input_imu_json imu_data.json
                           --save_map map_atlas.osa --output_trajectory_csv ...
Stage 03: 03_batch_slam.py（batch 模式跑多个 demo）
  → docker run gopro_slam --input_video raw_video.mp4 --input_imu_json imu_data.json
                           --load_map map_atlas.osa --max_lost_frames 600
                           --output_trajectory_csv camera_trajectory.csv
```

**Key insight**：cheng-chi fork **已经支持 `--load_map`**，并在 Stage 03 中已用于「**对已有地图做离线重定位**」——也就是说算法本身已经具备"加载地图后追踪"的能力，只是入口被锁死在「文件输入 → CSV 输出」的批处理形态。我们要做的不是新增算法能力，**而是替换 IO 层**。

### 1.3 `gopro_slam.cc` 实际 API 调用链

（来自 cheng-chi fork 源码确认）

```cpp
// 入口
cv::VideoCapture cap(input_video, cv::CAP_FFMPEG);           // ← 文件输入
LoadTelemetry(input_imu_json, ...);                          // ← 文件输入
ORB_SLAM3::System SLAM(voc, settings, IMU_MONOCULAR, gui,
                       /*initFr=*/0, /*sequence=*/"",
                       /*load_atlas_path=*/load_map,          // ← 地图加载
                       /*save_atlas_path=*/save_map, ...);
// 注意：构造函数注释 "we want to localize" 是 ASPIRATIONAL，
// 实际上构造函数并不会自动 ActivateLocalizationMode！
// gopro_slam.cc 也没显式调用，意味着 batch 模式下加载地图后仍会改写地图
// （在 batch 场景影响小，因为不保存；在线场景必须显式调用）

for (frame in cap) {
    auto [pose, ok] = SLAM.LocalizeMonocular(im, t, vImuMeas);   // ← cheng-chi 扩展 API
    write_csv(pose);
}
```

cheng-chi fork **保留了所有上游 ORB-SLAM3 公开 API**（`ActivateLocalizationMode`、`TrackMonocular`、`Shutdown` 等）。我们要做的是绕开 `gopro_slam.cc` 的 main 函数，直接用 `System` 类的公开 API。

---

## 2. 核心问题：能不能简单改造现有 SLAM 实现在线追踪？

**答案：能。** 改造点集中在三个地方，**不需要碰算法核心**：

1. **输入侧**：`cv::VideoCapture(file)` → RealSense2 capture loop（`rs2::pipeline`）
2. **IMU 侧**：`LoadTelemetry(json)` → RealSense2 IMU callback + 时间戳同步的环形缓冲
3. **输出侧**：`SaveTrajectoryCSV()` → ROS2 publisher（`geometry_msgs/PoseStamped` + tf2）或 ZMQ/共享内存

**算法侧零改动**——只需在 `System` 构造后**立即**调用 `SLAM.ActivateLocalizationMode()`（这是 cheng-chi fork 的一个隐性陷阱，下面 §5 详述）。

---

## 3. 三种方案对比

| 方案 | 描述 | 工作量 | 维护性 | 风险 |
|------|------|--------|--------|------|
| **A. 新增 `realsense_online` C++ 可执行文件**（推荐 ★） | 在 cheng-chi fork 的 `Examples/Monocular-Inertial/` 下新增 `.cc`，链接 ORB-SLAM3 + librealsense2 + rclcpp。复用 fork 的 CMakeLists，本机编译 | ~600 LOC，2–4 天 | 中（依赖 cheng-chi snapshot fork） | 低 |
| **B. 新写 ROS2 节点，调 ORB-SLAM3 库** | 不在 fork 内改动，单独建 `dexslide_slam_ros2` 包，subscribe `/camera/.../image_raw` + `/camera/.../imu`，publish `/orb_slam3/pose` + tf | ~300–500 LOC，3–5 天 | 高（与上游解耦） | 低 |
| **C. v4l2loopback + IMU JSON 持续 tail** | 把 `gopro_slam` 当黑盒，伪造视频设备和 IMU 文件喂给 Docker | "看似零代码"，实际很多陷阱 | 低 | 高（EOF/缓冲/同步） |
| **D. 换 SLAM**（stella_vslam / VINS-Fusion / MASt3R-SLAM） | 见 §6 | 高 | — | 见 §6 |

### 3.1 为什么不推荐 C（v4l2loopback + Docker）

- `gopro_slam.cc` 使用 `cv::VideoCapture(... CAP_FFMPEG)` 读视频，依赖 EOF 终结；流式输入会卡在 buffer 满。
- `LoadTelemetry` 是一次性 JSON 解析，不支持追加。
- Docker 内 USB 直通对 RealSense 不友好，且不符合"本机部署"的约束。
- 解决一个问题制造三个问题。

### 3.2 A vs B 的取舍

| 维度 | A（fork 内新增 `.cc`） | B（独立 ROS2 包） |
|------|----------------------|-----------------|
| 第一次跑起来速度 | 快——可复用 fork 现有 CMakeLists、`LoadTelemetry` 风格的工具函数 | 慢——重新写 IMU buffer、frame sync |
| 与 fork 解耦 | 弱——绑定 cheng-chi snapshot | 强——`System` 公开 API 跨上游/各 fork 通用 |
| 与 DexSlide 主线集成 | 需要桥接（pose 输出到 ROS2 或共享内存） | 天然 ROS2，Python 侧 `rclpy.Subscriber` 直接消费 |
| Pangolin Viewer | fork 默认带 GUI，需要 `bUseViewer=false` | 同样需要禁用，但更易控制 |
| 代码腐烂风险 | fork 的 CMakeLists 已包含 Pangolin / Boost.Serialization / DBoW2，将来 ORB-SLAM3 上游修 bug 不能 free 拿到 | 算法库依赖锁版本，节点代码独立演进 |

**最终倾向 A → 渐进式过渡到 B**：
- **第一阶段（A）**：在 fork 内做最少改动验证可行性。优势是几乎不需要重写 IMU 缓冲、初始化、地图加载等"已经被 gopro_slam.cc 解决过一次"的细节。
- **第二阶段（B，可选）**：当 A 跑通且需要把 SLAM 节点纳入 ROS2 launch 体系时，把 A 的代码从 fork 抽出来包装成 ROS2 节点。算法库当 binary 依赖，源代码不再 fork。

---

## 4. 推荐方案详细设计（方案 A）

### 4.1 新文件结构

在 `cheng-chi/ORB_SLAM3` fork 内新增：

```
Examples/Monocular-Inertial/
├── gopro_slam.cc               # 保留不动
├── realsense_online.cc         # 新增：本机在线追踪入口
├── realsense_online_imu.hpp    # 新增：IMU 时间同步环形缓冲
└── ros2_pose_publisher.hpp     # 新增（可选）：ROS2 publisher 抽出
CMakeLists.txt                  # 添加新 target
```

### 4.2 核心代码骨架（约 200 LOC 核心 + 400 LOC 配套）

```cpp
// realsense_online.cc 伪代码（不要照抄，仅示意结构）

int main(int argc, char** argv) {
    // 1. CLI: --vocabulary --setting --load_map [--ros2-domain N]
    parse_args();

    // 2. ORB-SLAM3 系统初始化（关键：load_atlas_path 通过 settings YAML 的
    //    System.LoadAtlasFromFile 传入；构造函数本身不接受 load 路径参数）
    ORB_SLAM3::System SLAM(voc_path, settings_path,
                           ORB_SLAM3::System::IMU_MONOCULAR,
                           /*bUseViewer=*/false);

    // 3. ⚠️ 关键：必须显式激活定位模式，否则会改写已加载的地图
    SLAM.ActivateLocalizationMode();

    // 4. RealSense 启动
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 960, 540, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
    cfg.enable_stream(RS2_STREAM_GYRO,  RS2_FORMAT_MOTION_XYZ32F, 200);
    auto profile = pipe.start(cfg, [&imu_buf](const rs2::frame& f) {
        if (auto m = f.as<rs2::motion_frame>()) {
            imu_buf.push(IMU::Point{...});   // lock-free ring buffer
        }
    });

    // 5. 主循环
    while (running) {
        auto frames = pipe.wait_for_frames(100 /*ms*/);
        auto cf = frames.get_color_frame();
        double t = cf.get_timestamp() / 1000.0;   // ms → s

        cv::Mat im(cv::Size(960, 540), CV_8UC3,
                   (void*)cf.get_data(), cv::Mat::AUTO_STEP);

        // 取 (last_frame_t, t] 之间的 IMU 样本
        std::vector<IMU::Point> vImu = imu_buf.drain_until(t);

        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, t, vImu);
        // 或：auto [Tcw, ok] = SLAM.LocalizeMonocular(im, t, vImu);

        if (!Tcw.translation().array().isNaN().any()) {
            publish_pose(Tcw, t);  // ROS2 / ZMQ / shm
        }
        last_frame_t = t;
    }
    SLAM.Shutdown();
}
```

### 4.3 与上游 DexSlide 的集成

DexSlide 主程序是 Python（`main.py` + Vedo viewer）。Pose 输出有三种通道可选：

| 通道 | 延迟 | 复杂度 | 推荐场景 |
|------|------|--------|----------|
| **ROS2 topic** `/orb_slam3/pose` | ~2–5 ms | 中 | 已有 ROS2 工具链时首选 |
| **ZMQ PUB/SUB** TCP `tcp://*:5555` | ~1–3 ms | 低 | 简单部署，跨进程 |
| **共享内存** mmap + 双缓冲 | < 1 ms | 高 | 极致延迟，需要同机 |

考虑到 `umi_mono` 已有 ROS2 路径（`run_slam_pipeline_ros2.py` 已存在），**推荐 ROS2 topic**。

---

## 5. 风险清单与缓解（live streaming 特有的陷阱）

> 这些是 batch 模式被"隐藏"的问题，转在线后会暴露。

| 风险 | 影响 | 缓解 |
|------|------|------|
| **1. 图像 / IMU 时钟域不一致** | IMU 在 SDK 内带本地时钟，图像可能用 USB 时间戳；偏差超过 10 ms 会让 IMU 预积分发散 | 用 RealSense `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME` 或 `HARDWARE_CLOCK`，并在节点启动时打印两者偏差 |
| **2. IMU 200Hz / 图像 30Hz 缓冲管理** | 直接堆 vector 会让 GC 抖动 | Lock-free ring buffer，预分配大小 ≥ 200 |
| **3. 帧抖动 / USB 丢帧** | `gopro_slam.cc` 的 `max_lost_frames=60`（≈2s @30Hz）会让在线追踪被频繁强制重置 | 在线模式设 `max_lost_frames=900` 或改成"软重置"——只触发重定位，不退出 |
| **4. Tracking lost + 自动恢复** | batch 模式可以直接 SLAM.Reset()，在线必须保持地图不变继续尝试重定位 | 用 `LocalizeMonocular` 返回的 `ok` 标志做状态机；持续投喂帧让 ORB-SLAM3 内部的 `RELOCALIZATION` 状态自然恢复 |
| **5. 曝光/自动白平衡突变** | D435i 进入光照变化区时曝光跳变会引起 ORB 特征质量骤降 | 锁定曝光（`RS2_OPTION_EXPOSURE` + `RS2_OPTION_ENABLE_AUTO_EXPOSURE=0`）或限制 AE 范围 |
| **6. MJPEG 解码抖动** | 默认 BGR8 raw 比 MJPEG 稳，避免库内解码 | 上述配置已用 `RS2_FORMAT_BGR8` raw |
| **7. Vocabulary MD5 mismatch** | 跨机器/版本运行可能因 vocab 文件不一致而 LoadAtlas 失败 | 用 fork 仓库自带 `ORBvoc.txt`，并在脚本中校验 MD5 |
| **8. Pangolin 无头模式 crash** | fork 默认编译 Pangolin viewer；headless 主机无 DISPLAY 时构造会阻塞 / segfault | `System(..., bUseViewer=false, ...)`；编译时也可加 `-DUSE_PANGOLIN=OFF`（cheng-chi fork 已支持） |
| **9. `LoadAtlas` 是 private** | 不能在 `System` 构造后再 load map；必须在构造时通过 settings YAML 的 `System.LoadAtlasFromFile` 字段传入 | 在 ROS2 节点的 parameter 里允许指定 settings yaml override |
| **10. cheng-chi fork 已 snapshot** | 0 open issues、几乎不更新 | 维护策略：a) 不轻易升级；b) 如需 bugfix 自己 cherry-pick 上游；c) 把节点代码做成方案 B 的样子隔离 |

---

## 6. 其他方案（已调研，未推荐）

| 方案 | 状态 | 适用判定 | 否决理由 |
|------|------|---------|---------|
| **stella_vslam** (OpenVSLAM 后继) | 主线纯视觉，定位模式成熟（`--disable-mapping`），msgpack map | 仅视觉 OK | **IMU 支持自 2021 年 issue #235 至今未合并到 main**——丢掉 D435i IMU 等于丢掉 metric scale 和抗快速运动能力 |
| **VINS-Fusion** + `loop_fusion` save/load | HKUST 上游停更（最后大改 2020）；ROS2 仅社区 fork (`zinuok`, `cannnnxu/jazzy`) | 备选 | GPL-3 license 传染；社区 ROS2 fork 维护性弱于本机 build ORB-SLAM3 |
| **MASt3R-SLAM** (CVPR 2025) | 学习式单目实时 SLAM，无 IMU；需要 RTX 级 GPU；~15 FPS RTX4090 | 不适配 | 30 Hz 难达；非商用 license；无 ActivateLocalization 等价物 |
| **OKVIS2** (ETH MRL 2024) | 优秀的 mono+IMU VIO；BSD-3；ROS2 支持 | 不适配 | **不支持持久化地图 + 重定位**；每次启动都是新会话——和"离线建图+在线追踪"模式不匹配 |
| **RTAB-Map** | 极成熟；BSD-3；定位模式 (`Mem/IncrementalMemory=false`)；ROS2 一流 | 备选 | mono+IMU 需要外接 VIO frontend（如 VINS）才能进入 RTAB-Map，引入两层堆栈 |
| **HLoc** (image retrieval) | 学习式定位，COLMAP map | 不适配 | 单 query ~0.3–1 Hz，连续追踪还需 VIO 串联 |
| **DROID-SLAM / DPVO / DPV-SLAM** | 学习式 VO，无 IMU | 不适配 | 无持久化重定位 |
| **NeRF/3DGS SLAM (MonoGS, NICE-SLAM)** | 研究为主 | 不适配 | <10 FPS，重 GPU |

### 6.1 ROS 包装层调研结论

| 项目 | 状态 | 适配本场景？ |
|------|------|------------|
| `thien94/orb_slam3_ros` (ROS1) | 2023.12 最后更新，最完整的 ORB-SLAM3 ROS 包装 | ✅ 作为**移植模板**；本身是 ROS1 |
| `Mechazo11/ros2_orb_slam3` | 2025.6 活跃，bare-bones | ⚠️ 仅单目（mono+IMU 是 TODO），无 load_map |
| `zang09/ORB_SLAM3_ROS2` | 2023.2 停更 | ❌ 不支持 mono+IMU、无 load_map |
| 上游 `UZ-SLAMLab/ORB_SLAM3/Examples/ROS` | 2021.12 frozen | ❌ 无 ActivateLocalizationMode、ROS1 |

**结论**：没有"开箱即用"的 ROS2 包装满足 (load `.osa`) + (localization-only) + (mono+IMU) + (≥30 Hz) 四项；最近的模板是 thien94/orb_slam3_ros（ROS1），手工移植到 ROS2 ~1–2 天。

---

## 7. 实施路线图

### 阶段 1：可行性验证（1–2 天）
- [ ] 本机源码编译 cheng-chi/ORB_SLAM3 fork（验证 Pangolin/Boost/DBoW2/OpenCV/Sophus 链接）
- [ ] 用现有 `map_atlas.osa` + 录制的 `raw_video.mp4` 跑一遍 `gopro_slam --load_map`，确认结果与 Docker 一致（baseline）
- [ ] 用 RealSense Viewer 确认 D435i 在 960×540@30Hz + IMU@200Hz 配置下稳定输出
- [ ] 实测时间戳偏差（`RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`）

### 阶段 2：最小可用版本（2–3 天）
- [ ] 新增 `Examples/Monocular-Inertial/realsense_online.cc`（裸 stdout 打印 pose）
- [ ] 实现 IMU ring buffer + `drain_until(timestamp)` 逻辑
- [ ] 加 `ActivateLocalizationMode()` 调用
- [ ] 端到端跑通：相机 → 追踪 → stdout pose，<33 ms/frame

### 阶段 3：ROS2 集成（1–2 天）
- [ ] 加 rclcpp publisher，发布 `geometry_msgs/PoseStamped` + tf2
- [ ] 加 launch 文件（参数：vocab、settings、map_path、image_topic、imu_topic）
- [ ] Python 侧测试 subscriber 接 Vedo viewer

### 阶段 4：稳健性强化（1–2 天）
- [ ] tracking lost 状态机（不退出，持续尝试重定位）
- [ ] 异常 IMU/图像跳变检测 + 日志
- [ ] `max_lost_frames` 改成软重置 + 计数器
- [ ] 锁定曝光范围
- [ ] benchmark：连续运行 1 小时，统计 FPS / latency / lost ratio

**总计：5–9 工作日**

---

## 8. 验收标准（PBT 性质 / invariants）

| # | 不变量 | 检验方式 |
|---|--------|---------|
| INV-1 | 追踪状态为 OK 时，连续两帧 pose 平移差 < 物理速度上限 × Δt | 在线 outlier 检测 |
| INV-2 | 启动→第一帧成功重定位时间 < 5 s（与 batch 模式重定位一致） | 端到端计时 |
| INV-3 | 加载地图后，地图 keyframes 数量与保存时一致（地图未被改写） | `SLAM.GetAtlas()->KeyFramesInMap()` 对比 |
| INV-4 | 在已建图区域，p99 帧延迟 < 33 ms（30 Hz 达标） | 连续 30 min benchmark |
| INV-5 | 追踪丢失 → 重定位回归后，pose 与 ground truth 偏差 < 5 cm（同一物理点） | ArUco 13 标定 |
| INV-6 | IMU 时间戳与图像时间戳差异 < 5 ms | 启动期 sanity check + 持续日志 |
| INV-7 | 连续运行 2 小时无内存泄漏（< 5% RSS 增长） | valgrind / massif 离线 + 在线 RSS 监控 |
| INV-8 | 同一段录制：在线追踪 trajectory 与 batch 追踪 trajectory ATE < 2 cm | 离线对照 |

---

## 9. 决定矩阵

| 你最关心的 | 选择 |
|----------|------|
| **最低工作量、最快出活** | 方案 A（fork 内新增 `realsense_online.cc`） |
| **长期维护性、可移植** | 方案 B（独立 ROS2 节点 + ORB-SLAM3 binary 依赖） |
| **不想动 C++** | 暂时无解——ORB-SLAM3 没有官方 Python 绑定支持 ActivateLocalizationMode 路径 |
| **想换更现代的 SLAM** | 推迟到 stella_vslam 的 IMU 支持合并（监控 issue #235） |
| **想要 GPU 加速** | 与算法无关——ORB-SLAM3 是 CPU 密集，单核就够 30 Hz |

**默认推荐：方案 A → 三个月后视使用情况升级到方案 B。**

---

## 10. 参考资料

### 代码
- cheng-chi/ORB_SLAM3 (本项目当前 fork): https://github.com/cheng-chi/ORB_SLAM3
- urbste/ORB_SLAM3 (上游 fork): https://github.com/urbste/ORB_SLAM3
- UZ-SLAMLab/ORB_SLAM3 (原仓库): https://github.com/UZ-SLAMLab/ORB_SLAM3
- thien94/orb_slam3_ros (ROS1 包装，最佳移植模板): https://github.com/thien94/orb_slam3_ros
- Mechazo11/ros2_orb_slam3 (ROS2 起点): https://github.com/Mechazo11/ros2_orb_slam3
- stella-cv/stella_vslam: https://github.com/stella-cv/stella_vslam
- HKUST-Aerial-Robotics/VINS-Fusion: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion

### 论文
- ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM (Campos et al., T-RO 2021)
- OKVIS2 (Leutenegger, 2022) — mono-inertial VIO 参考
- MASt3R-SLAM (CVPR 2025) — learning-based real-time mono SLAM

### 上游 API（关键函数签名）
```cpp
// include/System.h（cheng-chi fork 与上游一致）
void ActivateLocalizationMode();
void DeactivateLocalizationMode();
Sophus::SE3f TrackMonocular(const cv::Mat& im, const double& timestamp,
                            const vector<IMU::Point>& vImuMeas = {},
                            string filename = "");
// cheng-chi 扩展
std::pair<Sophus::SE3f, bool> LocalizeMonocular(...);
```

---

## 11. 给评审者的一页摘要

**Q: 当前 SLAM 离线建图离线追踪，能否改成离线建图在线追踪？**
A: 能。算法侧无需替换——cheng-chi/ORB_SLAM3 fork 已实现 `--load_map`，且保留上游 `ActivateLocalizationMode()` + `TrackMonocular()` API。

**Q: 改造最小路径是什么？**
A: 在 fork 内新增 `Examples/Monocular-Inertial/realsense_online.cc`：（1）用 librealsense2 替换文件输入；（2）构造 `System` 后立刻调 `ActivateLocalizationMode()`；（3）通过 ROS2 topic 发布 `Sophus::SE3f` 位姿。**~600 LOC、2–4 天**。

**Q: 该用 ROS2 还是 ZMQ 发布？**
A: ROS2（umi_mono 已有 ROS2 路径），但用 launch 参数允许切换 ZMQ 后端。

**Q: 30 Hz 能保证吗？**
A: 能。ORB-SLAM3 单线程 tracking 在 i7 上典型 < 25 ms/frame；预算余量足够。

**Q: 要不要换 SLAM？**
A: 不要。stella_vslam IMU 支持 5 年未合并；OKVIS2 不支持持久化地图；MASt3R-SLAM 是 GPU 重负载学习式方法。当前 fork 的算法选型在你的硬件和精度需求下仍是合理的。
