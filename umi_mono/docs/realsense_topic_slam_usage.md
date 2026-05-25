# `realsense_topic_slam_node` 使用与测试指南

> 离线建图 + 在线追踪：使用 ROS2 RealSense topic 输入的 ORB-SLAM3 定位节点。
> 与已有的 librealsense 直接驱动二进制 `realsense_online` 互为补充。
> 
> 适用场景：
> 
> 1. 由 `ros2 launch realsense2_camera rs_launch.py` 启动 D435i，节点订阅其发布的 topic 实时解算 pose；
> 2. 用 `ros2 bag play` 离线复跑已经录制的 bag（无需相机），同样能流式产出 pose。

---

## 1. 环境要求

| 项              | 版本                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| OS             | Ubuntu 22.04.5 LTS (jammy)                                                                                                     |
| 内核             | Linux 6.x x86_64                                                                                                               |
| ROS2           | Humble Hawksbill                                                                                                               |
| 系统 Python      | `/usr/bin/python3` = 3.10.12（**不要用 anaconda 3.x**，rclpy ABI 不兼容）                                                               |
| 编译器            | gcc/g++ ≥ 11.4.0（apt 默认 jammy 即可）                                                                                              |
| CMake          | ≥ 3.22                                                                                                                         |
| OpenCV         | 4.5.4 (apt: `libopencv-dev`)                                                                                                   |
| Eigen3         | 3.4.0 (apt: `libeigen3-dev`)                                                                                                   |
| ZMQ            | libzmq3 (apt: `libzmq3-dev`) — 仅为兼容旧 `pose_publisher_node`                                                                     |
| Pangolin       | 0.8 已编译至 `umi_mono/external/Pangolin/build/`（一次性脚本生成）                                                                          |
| Sophus         | 1.22.10 已配置至 `umi_mono/external/Sophus/`（一次性脚本生成）                                                                              |
| ORB-SLAM3 fork | `umi_mono/external/ORB_SLAM3_fork/`（cheng-chi snapshot；`bash scripts/setup_orbslam3_fork.sh + build_orbslam3_native.sh` 已编译完成） |
| RealSense 驱动   | `ros-humble-realsense2-camera` ≥ 4.57.7（**仅 live 模式需要**；bag 回放不依赖它）                                                            |
| 词袋 MD5         | `5420bad0713bc97034dd2a9b2f0cc387`（informational）                                                                              |

### 已经安装好的 apt 包（关键项）

```bash
ros-humble-rclcpp
ros-humble-geometry-msgs
ros-humble-sensor-msgs
ros-humble-tf2-ros
ros-humble-cv-bridge
ros-humble-realsense2-camera           # 实机使用时
python3-colcon-common-extensions
libopencv-dev libeigen3-dev libzmq3-dev
```

### 一次性主机准备（如尚未做）

按 `docs/setup_phase0_environment.md` → `docs/setup_phase1_native_build.md` → `docs/setup_phase2_realsense_online.md` 顺序：

```bash
bash umi_mono/scripts/check_host_env.sh
bash umi_mono/scripts/install_apt_deps.sh --apply
bash umi_mono/scripts/build_pangolin.sh
bash umi_mono/scripts/build_sophus.sh
bash umi_mono/scripts/install_librealsense.sh --apply   # 仅 live 模式需要
bash umi_mono/scripts/setup_orbslam3_fork.sh
bash umi_mono/scripts/build_orbslam3_native.sh
```

之后只需要构建新的 ROS2 节点（增量编译，几秒钟）。

---

## 2. 构建新节点

```bash
source /opt/ros/humble/setup.bash
cd /data/codes/DexSlide/umi_mono/ros2_ws
colcon build --packages-select dexslide_slam_publisher
```

成功标志：

```
Finished <<< dexslide_slam_publisher [11s]
Summary: 1 package finished
```

二进制位置：`install/dexslide_slam_publisher/lib/dexslide_slam_publisher/realsense_topic_slam_node`

---

## 3. 运行方式

启动前在 **每个新终端** 先 source 环境：

```bash
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash
```

### 3.1 用 `ros2 bag play` 离线回放（推荐先用这个跑通流程）

```bash
# 终端 1：启动 SLAM 节点（加载已有 .osa 地图）
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/data/codes/umi_mono_data/demos/mapping/map_atlas.osa

# 终端 2：回放 aurco bag（14 秒）
ros2 bag play /data/codes/umi_mono_data/aurco

# 终端 3：验证
ros2 topic hz /dexslide/slam/pose
ros2 topic echo /dexslide/slam/pose --once
```

终端 1 预期日志：

```
Atlas loaded!
Localization mode NOT activated (matches gopro_slam.cc); ...
Atlas loaded from: /data/codes/umi_mono_data/demos/mapping/map_atlas.osa
Session timestamp origin captured: 1779443333.752 s
... Relocalized!! ...
```

### 3.2 用真实 D435i live 模式

```bash
# 终端 A：启动 D435i 驱动（含 color + accel + gyro，不要 unite imu）
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true \
  unite_imu_method:=0 \
  rgb_camera.color_profile:=960x540x30 \
  gyro_fps:=200 accel_fps:=200 \
  initial_reset:=true

# 终端 B：启动 SLAM 节点
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/path/to/your/map_atlas.osa

# 终端 C：验证
ros2 topic hz /dexslide/slam/pose          # 期望 ~30 Hz（视觉/IMU 都健康时）
ros2 run tf2_ros tf2_echo map camera_color_optical_frame
```

### 3.3 Launch 参数全表

| 参数                           | 类型     | 默认值                                           | 含义                                                                                     |
| ---------------------------- | ------ | --------------------------------------------- | -------------------------------------------------------------------------------------- |
| `vocab`                      | string | `<fork>/Vocabulary/ORBvoc.txt`                | ORB 词袋路径                                                                               |
| `settings`                   | string | `umi_mono/config/RealSense_D435i_online.yaml` | 相机内参 + IMU 噪声 yaml                                                                     |
| `map_atlas`                  | string | （空）                                           | `.osa` 地图路径，空则从零起建图（调试用）                                                               |
| `image_topic`                | string | `/camera/camera/color/image_raw`              | 订阅彩色图像 topic                                                                           |
| `accel_topic`                | string | `/camera/camera/accel/sample`                 | 订阅加速度计 topic                                                                           |
| `gyro_topic`                 | string | `/camera/camera/gyro/sample`                  | 订阅角速度计 topic                                                                           |
| `pose_topic`                 | string | `/dexslide/slam/pose`                         | 输出 PoseStamped topic 名                                                                 |
| `map_frame`                  | string | `map`                                         | tf 父坐标系名                                                                               |
| `camera_frame`               | string | `camera_color_optical_frame`                  | tf 子坐标系名                                                                               |
| `max_lost_frames`            | int    | 900                                           | 连续丢失多少帧后触发 soft reset（仅清零计数器，不重置地图）                                                    |
| `accel_gyro_pair_window_s`   | double | 0.020                                         | accel 与 gyro 时戳差 ≤ 此值时配对为 IMU::Point                                                   |
| `activate_localization_mode` | bool   | `false`                                       | 是否调用 `ActivateLocalizationMode()`。默认 false 与 `gopro_slam.cc` 一致；地图修改只在内存里，磁盘 `.osa` 不变 |

---

## 4. 自动化烟囱测试

仓库提供脚本 `scripts/test_realsense_topic_slam.sh`：

```bash
bash /data/codes/DexSlide/umi_mono/scripts/test_realsense_topic_slam.sh \
  --map /data/codes/umi_mono_data/demos/mapping/map_atlas.osa \
  --bag /data/codes/umi_mono_data/aurco
```

脚本会：

1. 后台启动 SLAM 节点
2. 等待 atlas 加载（10s）
3. 后台启动 `ros2 topic echo` 统计 pose 数量
4. 后台启动 `ros2 bag play` 回放
5. 等待 16s（bag 14s + 余量）
6. 打印 pose 计数、relocalization 次数、session origin、SLAM 关键事件
7. 清理所有后台进程

预期结果（aurco 14s bag）：

```
Pose count: ≥ 200  (典型 280~320)
Relocalized events: ≥ 10
Atlas loaded events: ≥ 1
Session origin captured: <Unix-epoch in seconds>
```

退出码：0 = 通过（pose count > 100），1 = 失败。

---

## 5. Python 端订阅 pose 示例

新节点直接发布 `geometry_msgs/PoseStamped`，因此可以用标准 `rclpy` 订阅（必须用系统 Python 3.10）：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 -c "
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class Sub(Node):
    def __init__(self):
        super().__init__('pose_sniffer')
        self.create_subscription(PoseStamped, '/dexslide/slam/pose',
                                  lambda m: print(m.pose.position), 10)
rclpy.init(); rclpy.spin(Sub())
"
```

如果使用 DexSlide 已有的 `SlamPoseSubscriber`（含时间对齐 SLERP），无需任何改动——它已经订阅 `/dexslide/slam/pose`。

---

## 6. 常见问题

### Q：`ros2 launch` 启动后日志只到 `Loading ORB Vocabulary` 就卡住

A：词袋加载需要约 2 秒，是正常的；之后会打印 `Vocabulary loaded!` → `Atlas loaded!`。

### Q：`Atlas loaded` 后没有 pose 输出

A：检查 bag 是否真的有 `/camera/camera/color/image_raw` 流：

```bash
ros2 bag info /data/codes/umi_mono_data/aurco
```

并确认相机帧的 `header.stamp` 不是全零。

### Q：节点崩溃 `pangolin/pangolin.h: No such file`

A：编译期没找到 Pangolin。确认 `external/Pangolin/build/PangolinConfig.cmake` 存在；如果不存在，重跑 `bash scripts/build_pangolin.sh`。

### Q：link error `libcurl.so.4: undefined reference to curl_easy_*@CURL_OPENSSL_4`

A：CMakeLists.txt 已加 `-Wl,--allow-shlib-undefined` 绕过；如果重新编译后又冒出来，检查这个 linker option 是否仍在。

### Q：`Relocalized!!` 频繁出现，pose 率只有 ~15 Hz

A：aurco bag 是 ArUco 标定时近距离拍摄的，跟用来建图的远景不一致，召回率自然低。换用与建图同场景的 bag 或 live 数据测试，预期 ~30 Hz。

### Q：想要避免地图在内存里被修改

A：launch 时加 `activate_localization_mode:=true`。注意此设置在某些 dataset 上可能触发 "IMU is not or recently initialized. Reseting active map" 警告——这是 ORB-SLAM3 上游 bug，并不影响磁盘上的 `.osa`。

### Q：用真机 D435i 跑，发现 `unite_imu_method` 怎么设？

A：**必须设 0**（split accel/gyro topic）；本节点不订阅合并的 `/camera/camera/imu`。设 ≥ 1 会让 driver 同时发布合并和分裂版本（参见 `docs/ros2_pipeline_review.md` A1 隐患）。

---

## 7. 已知限制

- **CMakeLists 用了 `-Wl,--allow-shlib-undefined`**：用来绕过 Pangolin → libgdal → libcurl@CURL_OPENSSL_4 这条传递依赖链上的版本化符号冲突（Ubuntu 22.04 自带的 `libgdal.so.30` 引用了与本机 `libcurl.so.4` 不一致的版本化符号）。此选项会屏蔽**真正的**链接错误。验证手段：每次发版前手动跑一次 `ldd -r install/dexslide_slam_publisher/lib/dexslide_slam_publisher/realsense_topic_slam_node` 检查 `undefined symbol`；只要 `libpang_*.so` 链上的 curl 符号是唯一的 undefined 就属于已知项。
- **`activate_localization_mode:=true` 在 aurco bag 上可能触发 ORB-SLAM3 自动 reset active map**。这是上游 ORB-SLAM3 在 IMU_MONOCULAR + loaded atlas + inliers 不足时的固有行为。如果发生，磁盘 `.osa` 仍然不变，但内存里的活跃地图被重置 → 后续 pose 输出会大幅减少。建议在 close-up / 与建图视角差异大的场景下用默认 `false`。
- **MultiThreadedExecutor 假设 image 处理时间 < 1.28 s**：IMU ring buffer 容量是 256，gyro 200 Hz 满载下 1.28 s 写满。超过这个时长会丢失最早的 IMU 样本（drop-oldest 策略）。正常 `LocalizeMonocular` 单次 10–30 ms，远远低于这个上限。

---

## 8. 相关文档

- 离线建图流水：`docs/USER_GUIDE.md` 1.2 节
- 老的 librealsense 直接驱动二进制（实时模式 v1）：`docs/setup_phase2_realsense_online.md`
- ROS2 桥架构（ZMQ → PoseStamped）：`docs/setup_phase3_4_publishers_robustness.md`
- 上游 SLAM 流水检视：`docs/ros2_pipeline_review.md`
