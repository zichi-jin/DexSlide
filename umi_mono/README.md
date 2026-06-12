# DexSlide 实时 SLAM 系统：部署与使用手册（整合版）

> **版本**：2026-05 整合版 · 当前方案 = `realsense_topic_slam_node`（ROS2 topic 订阅 + 内嵌 ORB-SLAM3 fork + PoseStamped 直接发布）
> **整合自**：`USER_GUIDE.md`、`setup_phase0~7_*.md`、`realsense_topic_slam_usage.md`、`VALIDATION_GUIDE.md`、`SLAM_readme_mono.txt`
>
> 全套流程分两阶段：
>
> - **阶段 A（离线建图）**：用 ROS2 录的 bag 跑 `run_mapping_and_desk_aruco_ros2.py` 流水线 → 产出 `map_atlas.osa`
> - **阶段 B（实时 SLAM）**：用 `realsense_topic_slam_node` 加载 `.osa`，订阅 RealSense ROS2 topic（实机或 bag 回放），实时发布 `geometry_msgs/PoseStamped`；可选启动 ArUco world-pose 节点，实时输出指定 marker 在桌面标定 world frame 下的位姿
>
>   **<mark>注意请自行修改代碼/命令的路径位置，以及不同ubuntu版本的对应修改</mark>**

---

## 0. 系统总览

```
┌─────────────── 阶段 A：离线建图 (Docker 流水线，一次性) ───────────────┐
│ 1) ros2 bag record (D435i color + accel + gyro)  → mapping bag         │
│ 2) python run_mapping_and_desk_aruco_ros2.py <session>                 │
│    → demos/mapping/map_atlas.osa  (Atlas 二进制)                       │
│    → demos/aurco/tx_slam_tag.json (SLAM↔ArUco-13 桌面 tag 标定)        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  map_atlas.osa
                                ▼
┌─────────────── 阶段 B：实时 SLAM（推荐 = topic 方案）─────────────────┐
│  数据源：                                                             │
│   • 实机：ros2 launch realsense2_camera rs_launch.py …               │
│   • 离线：ros2 bag play <bag>                                         │
│                  ↓ /camera/camera/color/image_raw                    │
│                  ↓ /camera/camera/accel/sample                       │
│                  ↓ /camera/camera/gyro/sample                        │
│  ───────────────────────────────────                                  │
│  realsense_topic_slam_node  (rclcpp 节点, MultiThreadedExecutor)      │
│   • cv_bridge 解码 image                                              │
│   • accel/gyro 在 20 ms 内配对 → IMU::Point → 环形 buffer             │
│   • 内嵌 ORB_SLAM3::System (cheng-chi fork, IMU_MONOCULAR)            │
│     LoadAtlasFromFile=map_atlas.osa                                   │
│     LocalizeMonocular(image, t, vImu) → Sophus::SE3f                  │
│   • header.stamp 零对齐 → 防 IMU 预积分 float 精度问题                │
│                  ↓                                                    │
│   geometry_msgs/PoseStamped @ /dexslide/slam/pose  (BEST_EFFORT)      │
│   tf2: map → camera_color_optical_frame                               │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 1. 环境配置与部署

### 1.1 硬件 / 平台要求

| 项            | 要求                                                          |
| ------------ | ----------------------------------------------------------- |
| OS           | Ubuntu 22.04 LTS（22.04.x 任意小版本）                             |
| 内存           | ≥ 16 GB                                                     |
| **D435i 相机** | 固件 ≥ 5.16；USB 3.x 端口（实机模式必需，bag 回放不需要）                      |
| ROS2         | jazzy Hawksbill                                            |
| 系统 Python    | `/usr/bin/python3` = 3.10.x（**不要用 anaconda**，rclpy ABI 不兼容） |
| Docker       | 仅阶段 A 离线建图需要（`chicheng/orb_slam3:latest`）                   |
| 网络           | 能 clone GitHub + apt update；国内需代理                           |

### 1.2 不在 GitHub 仓库里的文件 / 数据 / 工件清单

以下都 **不**会被 `git clone` 拉下来（`.gitignore` 已排除 `external/`），需要自备或现场生成：

#### A. 外部第三方代码 / 库（一次性安装到 `umi_mono/external/`）

| 路径                                              | 来源                                                               | 大小                                             | 谁负责拉                                                          |
| ----------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------- |
| `external/Pangolin/`                            | `https://github.com/stevenlovegrove/Pangolin` v0.8               | ~500 MB build 后                                | `scripts/build_pangolin.sh`                                   |
| `external/Sophus/`                              | `https://github.com/strasdat/Sophus.git` 1.22.10                 | header-only                                    | `scripts/build_sophus.sh`                                     |
| `external/ORB_SLAM3_fork/`                      | `https://github.com/cheng-chi/ORB_SLAM3` pinned SHA `b741dca...` | ~400 MB                                        | `scripts/setup_orbslam3_fork.sh` + `build_orbslam3_native.sh` |
| `external/cppzmq/`                              | `https://github.com/zeromq/cppzmq.git` v4.10.0                   | ~1 MB                                          | 一行 `git clone --depth 1 --branch v4.10.0 ...`（脚本未自动化）         |
| `external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt` | fork 内置 tar.gz 解压                                                | 145 MB（md5=`5420bad0713bc97034dd2a9b2f0cc387`） | 由 `setup_orbslam3_fork.sh` 自动解                                |
| `external/ORB_SLAM3_fork/lib/libORB_SLAM3.so`   | 编译产物                                                             | 354 MB                                         | 由 `build_orbslam3_native.sh` 生成                               |

#### B. ROS2 工作空间构建产物（gitignored）

| 路径                 | 内容                     | 谁负责生成              |
| ------------------ | ---------------------- | ------------------ |
| `ros2_ws/build/`   | colcon 中间产物            | `colcon build ...` |
| `ros2_ws/install/` | 二进制 + share + 环境 setup | `colcon build ...` |
| `ros2_ws/log/`     | 编译日志                   | colcon             |

#### C. 用户运行数据（阶段A生成，绝对路径与你的部署相关）

| 路径                                                        | 内容                                      | 怎么来的                                 |
| --------------------------------------------------------- | --------------------------------------- | ------------------------------------ |
| `<session>/`                                              | ROS2 bag 录制目录（`metadata.yaml` + `.db3`） | `ros2 bag record`                    |
| `<session>/demos/mapping/map_atlas.osa`                   | **建图产物（关键）**                            | `run_mapping_and_desk_aruco_ros2.py` |
| `<session>/demos/mapping/raw_video.mp4` + `imu_data.json` | 中间产物                                    | 同上，阶段 00                             |
| `<session>/demos/aurco/tx_slam_tag.json`                  | 桌面 ArUco-13 → SLAM 坐标系                  | 同上，阶段 05                             |
| `<session>/demos/mapping/slam_mask.png`                   | 可选 mask                                 | 用户自备（用于遮挡建图时无关像素）                    |

#### D. 例子里参考的具体路径（本机）

| 路径                                                      | 当前内容                      |
| ------------------------------------------------------- | ------------------------- |
| `/data/codes/umi_mono_data/mapping/`                    | 一段 58 s 建图 bag            |
| `/data/codes/umi_mono_data/aurco/`                      | 一段 14 s ArUco-13 桌面标定 bag |
| `/data/codes/umi_mono_data/demos/mapping/map_atlas.osa` | 已经生成的地图（验证用）              |

### 1.3 一次性安装步骤

> 假设网络通畅、有 sudo`。
> 国内环境先 `export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890`。

```bash
# 0. 拉项目
git clone <your-repo-url> /home/jzq/MyJob/DexSlide
cd /home/jzq/MyJob/DexSlide

# 1. 主机环境快速检查（只读）
bash umi_mono/scripts/check_host_env.sh
#  期望：所有 [OK]（system python3 显示 anaconda 时 [WARN] 也无所谓）

# 2. apt 依赖（先 dry-run，看缺什么）
bash umi_mono/scripts/install_apt_deps.sh
#  之后 --apply（提示时输入 yes，约 1 分钟）
bash umi_mono/scripts/install_apt_deps.sh --apply

# 3. Pangolin v0.8 源码编译（~6 分钟）
bash umi_mono/scripts/build_pangolin.sh

# 4. Sophus 1.22.10 配置（~10 秒）
bash umi_mono/scripts/build_sophus.sh

# 5. librealsense2-dev（实机模式必装，bag 模式可跳）
bash umi_mono/scripts/install_librealsense.sh             # dry-run
echo yes | bash umi_mono/scripts/install_librealsense.sh --apply

# 6. ROS2 jazzy + 关键包（如未装）
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-cv-bridge \
  ros-jazzy-tf2-ros \
  ros-jazzy-geometry-msgs \
  ros-jazzy-sensor-msgs \
  ros-jazzy-realsense2-camera \
  python3-colcon-common-extensions

# 7. ORB-SLAM3 fork（cheng-chi snapshot）clone + SHA pin + Vocabulary 解压
bash umi_mono/scripts/setup_orbslam3_fork.sh

# 8. cppzmq 头文件（脚本里没自动化，手动一次性 clone）
cd /home/jzq/MyJob/DexSlide/umi_mono/external
git clone --depth 1 --branch v4.10.0 https://github.com/zeromq/cppzmq.git
cd /home/jzq/MyJob/DexSlide

# 9. 编译 ORB-SLAM3 fork（含 libORB_SLAM3.so 与 legacy realsense_online）
bash umi_mono/scripts/build_orbslam3_native.sh

# 10. 编译 ROS2 节点（含 realsense_topic_slam_node + 老的 pose_publisher_node）
source /opt/ros/jazzy/setup.bash
cd /home/jzq/MyJob/DexSlide/umi_mono/ros2_ws
colcon build --packages-select dexslide_slam_publisher

# 11. 阶段 A 才需要：Docker + 镜像 pull
docker pull chicheng/orb_slam3:latest          # 仅在你要重新建图时

# 12. 阶段 A 才需要：Python 依赖（最小集合，详见 §1.5）
#     系统 Python 3.10 安装（推荐；阶段 B 也用这个 Python）：
/usr/bin/python3 -m pip install --user -r umi_mono/requirements_mapping.txt
```

完成。后续日常增量编译（改了 `realsense_topic_slam_node.cpp` 后）：

```bash
source /opt/ros/jazzy/setup.bash
cd /home/jzq/MyJob/DexSlide/umi_mono/ros2_ws
colcon build --packages-select dexslide_slam_publisher
```

### 1.4 Python 依赖（精简版）

> **不要**用上游 `conda_environment.yaml` —— 那是给 GoPro + diffusion policy 训练用的（pytorch / diffusers / accelerate / ~30 GB），跟当前 SLAM 方案没关系。当前方案的 Python 依赖只有以下范围：

| 阶段 / 角色 | Python 解释器 | 需要的包 | 装法 |
|---|---|---|---|
| **阶段 A 建图流水线**（`run_mapping_and_desk_aruco_ros2.py` 调用 stage 00/02/04/05） | 任何 Python ≥ 3.9（系统 3.10 / conda / venv 都可） | `click`、`numpy`、`scipy`、`opencv-python`、`PyYAML`、`tqdm`、`rosbags` | `pip install -r umi_mono/requirements_mapping.txt` |
| **阶段 B 实时 SLAM 节点**（`realsense_topic_slam_node` 本身） | **不需要 Python** | — | C++ 二进制，C++17 |
| **阶段 B launch 文件** | `/usr/bin/python3` 3.10（由 ROS2 jazzy 提供） | `launch`、`launch_ros`、`rclpy`、`ament_index_python` | apt 装 `ros-jazzy-desktop` 时自动给 |
| **阶段 B 可选 ArUco world-pose 节点**（`aruco_world_pose_node.py`） | `/usr/bin/python3` 3.10 | `rclpy`、`cv_bridge`（apt）+ `numpy`、`opencv-python`、`PyYAML` | apt + `pip install -r umi_mono/ros2_ws/dexslide_slam_publisher/requirements.txt` |
| **下游 Python 消费 pose**（`rclpy.Subscriber` / DexSlide 主程序） | `/usr/bin/python3` 3.10 | `rclpy`（apt 提供）+ 你自己的业务包 | 同上 |
| **本仓库的 pytest 单元测试**（如 `tests/test_slam_pose_subscriber.py`） | `/usr/bin/python3` 3.10 | `pytest`（apt 提供） + `numpy` | 同上 |

#### 关键约定

1. **跑阶段 B 时必须用 `/usr/bin/python3`**（rclpy 只跟系统 Python 3.10 二进制兼容；anaconda 3.x 跑 launch 文件会 `ModuleNotFoundError: rclpy`）。
2. **跑阶段 A 不强制用系统 Python** —— 因为它走的是 `rosbags`（纯 Python，无 ABI 绑定），任何 ≥3.9 的 Python 都行。
3. **anaconda 用户**：若用 conda env 跑阶段 A，**进入阶段 B 前先 `conda deactivate`**，再 source ROS2 jazzy。
4. **`pip install --user -r requirements_mapping.txt`** 装在 `~/.local/lib/python3.10/site-packages`，不污染系统包，也不需要 sudo。

#### 与上游 README 的区别

| 上游 `README.md` 提到的                                                                                              | 当前方案是否需要                                                             |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `mamba env create -f conda_environment.yaml`（含 pytorch / accelerate / diffusers / wandb / mujoco / k3d / gym 等） | ❌ 完全不需要                                                              |
| `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`                                             | ❌ 只有上游 diffusion policy viewer 需要                                    |
| `sudo apt install libspnav-dev spacenavd` (SpaceMouse)                                                          | ❌ 只有上游 `eval_real.py` 需要                                             |
| `run_slam_pipeline.py example_demo_session`（GoPro 全流程）                                                          | ❌ 我们用 `run_mapping_and_desk_aruco_ros2.py`（仅 mapping + ArUco 标定 5 步） |
| `train.py` + `accelerate ... train.py`（diffusion policy 训练）                                                     | ❌ 不在当前方案范围                                                           |
| `eval_real.py` + UR5/Franka/WSG50/GoPro/SpaceMouse 硬件配置                                                         | ❌ 不在当前方案范围                                                           |

### 1.5 安装验证（一次性自检 checklist）

每一条都跑一遍，全过才说明部署成功：

```bash
# A. 外部依赖到位
ls umi_mono/external/Pangolin/build/libpango_core.so
ls umi_mono/external/Sophus/build/SophusConfig.cmake
ls umi_mono/external/ORB_SLAM3_fork/lib/libORB_SLAM3.so   # 354 MB
ls umi_mono/external/cppzmq/zmq.hpp
md5sum umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt
#   期望: 5420bad0713bc97034dd2a9b2f0cc387

# B. Python 依赖（阶段 A 用）
/usr/bin/python3 -c "import click, numpy, scipy, cv2, yaml, tqdm, rosbags; print('OK')"
#   期望: OK；任何 ModuleNotFoundError 都重跑 pip install -r umi_mono/requirements_mapping.txt

# C. ROS2 节点构建产物
ls umi_mono/ros2_ws/install/dexslide_slam_publisher/lib/dexslide_slam_publisher/realsense_topic_slam_node
ls umi_mono/ros2_ws/install/dexslide_slam_publisher/share/dexslide_slam_publisher/launch/dexslide_slam_topics.launch.py

# D. Launch 文件 args 可被识别
source /opt/ros/jazzy/setup.bash
source umi_mono/ros2_ws/install/setup.bash
ROS_LOG_DIR=/tmp/ros-log ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py --show-args
#   期望：能看到 vocab/settings/map_atlas/image_topic/.../aruco_world_pose_topic/.../activate_localization_mode 共 21 个参数

# E. （实机模式）D435i 烟囱测试
bash umi_mono/scripts/run_d435i_smoke.sh
#   期望：color FPS ≈ 30, accel FPS ≈ 200, gyro FPS ≈ 200, PASS

# F. （bag 模式）自动化端到端测试 — 这是最强信号
bash umi_mono/scripts/test_realsense_topic_slam.sh \
  --map /data/codes/umi_mono_data/demos/mapping/map_atlas.osa \
  --bag /data/codes/umi_mono_data/aurco
#   期望：pose count (sec>0) ≥ 100; PASS; exit 0
```

> **注意**：如果 E 步骤的 `pose count (sec>0) == 0` 但 `ros2 topic hz /dexslide/slam/pose` 在独立终端能跑出 ~20 Hz，**不是 SLAM bug**，而是 `ros2 topic echo` QoS 自动匹配竞态。脚本已修，老版本要更新到 2026-05-24 以后的版本。

---

## 2. 阶段 A：离线建图（命令与操作）

> 此阶段产物是一个 `.osa` 地图文件 + 一个 `tx_slam_tag.json` 桌面标定。建一次可以反复在阶段 B 用。
> 需要 **Docker + `chicheng/orb_slam3:latest`**（Docker 跑离线 ORB-SLAM3）。

### 2.1 录建图与桌面标定数据

新开一个 session 目录：

```bash
mkdir -p ~/dexslide_data/session_$(date +%Y%m%d)
cd ~/dexslide_data/session_$(date +%Y%m%d)

# 启动 D435i 驱动
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true \
  unite_imu_method:=0 \
  rgb_camera.color_profile:=960x540x30 \
  gyro_fps:=200 accel_fps:=200 \
  initial_reset:=true &
```

在新终端录两段 bag（注意 `aurco` 拼写是项目硬编码，**不要写成 aruco**）：

```bash
# 1) 建图 bag —— 围绕场景慢速绕一圈 60 s 以上，要有回环
source /opt/ros/jazzy/setup.bash
ros2 bag record -o mapping \
  /camera/camera/color/image_raw \
  /camera/camera/accel/sample \
  /camera/camera/gyro/sample

# 2) ArUco-13 桌面 tag 标定 bag —— 把 tag 摆桌面对着镜头 10 秒
ros2 bag record -o aurco \
  /camera/camera/color/image_raw \
  /camera/camera/accel/sample \
  /camera/camera/gyro/sample
```

录完目录形如：

```
~/dexslide_data/session_$(date +%Y%m%d)/
├── mapping/
│   ├── metadata.yaml
│   └── mapping_0.db3
└── aurco/
    ├── metadata.yaml
    └── aurco_0.db3
```

### 2.2 跑建图流水线（5 阶段）

```bash
cd /home/jzq/MyJob/DexSlide/umi_mono
conda deactivate 2>/dev/null      # 防止 anaconda 干扰
source /opt/ros/jazzy/setup.bash # 仅当你需要 ros2 bag 解析时也可（流水线本身用 Docker）

python run_mapping_and_desk_aruco_ros2.py ~/dexslide_data/session_$(date +%Y%m%d)
```

这条命令会按顺序跑完：

| 阶段         | 做什么                                                     | 产物                                                                                        |
| ---------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 00         | 解码 bag → mp4 + IMU json；把最长那段 demo 提升为 `demos/mapping/` | `demos/mapping/raw_video.mp4`、`demos/mapping/imu_data.json`、`demos/aurco/raw_video.mp4` 等 |
| 02         | Docker 里跑 ORB-SLAM3 建图 → 保存 atlas                       | **`demos/mapping/map_atlas.osa`**                                                         |
| 03 (aurco) | 在 `aurco/` 上 load 这张地图重定位                               | `demos/aurco/camera_trajectory.csv`                                                       |
| 04         | 桌面 ArUco-13 检测                                          | `demos/aurco/tag_detection.pkl`                                                           |
| 05         | 解算 SLAM↔桌面 tag 变换                                       | **`demos/aurco/tx_slam_tag.json`**                                                        |

### 2.3 阶段 A 产物验证

```bash
SESSION=~/dexslide_data/session_$(date +%Y%m%d)

# 关键产物存在？
ls -la $SESSION/demos/mapping/map_atlas.osa            # 几十~几百 MB
ls -la $SESSION/demos/aurco/tx_slam_tag.json           # < 1 KB JSON
ls -la $SESSION/demos/aurco/camera_trajectory.csv      # CSV，应有几百行

# 地图加载 sanity check（用 topic SLAM 节点试加载，5 秒就能看出来）
source /opt/ros/jazzy/setup.bash
source /home/jzq/MyJob/DexSlide/umi_mono/ros2_ws/install/setup.bash
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=$SESSION/demos/mapping/map_atlas.osa &
LAUNCH_PID=$!
sleep 10
# 看日志关键字
ros2 topic info /dexslide/slam/pose --verbose | head -8
kill -SIGINT $LAUNCH_PID
#   期望：节点日志里能看到 'Atlas loaded!' + 'Atlas loaded from: <path>'
```

#### 建图诀窍（避免阶段 B 反复丢失）

- 围着场景走 **1–2 圈**，回到起点（含回环）
- 光照与之后实机使用时一致（白天建图、晚上用 → 容易丢）
- **慢速移动 + 平移与旋转结合**；不要纯转头
- 镜头别看天 / 大白墙 / 玻璃（ORB 特征不足）
- 用 D435i 固件 ≥ 5.16，USB 3.x（USB 2 带宽不够）

---

## 3. 阶段 B：实时 SLAM（命令与操作）

> 推荐路径 = **`realsense_topic_slam_node`**（订阅 ROS2 topic + 内嵌 ORB-SLAM3 + 直接发 PoseStamped）。
> 老路径 = `realsense_online` 二进制 + ZMQ + `pose_publisher_node` 桥，保留为 fallback，不在本节展开（见 §6 文档索引）。

每个新终端都先 source：

```bash
source /opt/ros/jazzy/setup.bash
source /home/jzq/MyJob/DexSlide/umi_mono/ros2_ws/install/setup.bash
```

### 3.1 启动 SLAM 节点（不管数据源）

```bash
# 终端 1
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/path/to/your/map_atlas.osa
```

启动期日志中你应该看到：

```
Field                  | Value
-----------------------+------------------------------------
vocabulary             | <fork>/Vocabulary/ORBvoc.txt
setting                | <umi_mono>/config/RealSense_D435i_online.yaml
load_map               | /path/to/your/map_atlas.osa
image_topic            | /camera/camera/color/image_raw
accel_topic            | /camera/camera/accel/sample
gyro_topic             | /camera/camera/gyro/sample
pose_topic             | /dexslide/slam/pose
...
Loading ORB Vocabulary. This could take a while...
Vocabulary loaded!
Atlas loaded!
Localization mode NOT activated (matches gopro_slam.cc); ...
Atlas loaded from: /path/to/your/map_atlas.osa
```

之后等数据源（实机或 bag）一接入，就会立刻开始算 pose。

### 3.2 数据源 A：实机 D435i -- (待验证)

```bash
# 终端 2 —— 启动 RealSense ROS2 驱动
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true \
  unite_imu_method:=0 \
  rgb_camera.color_profile:=960x540x30 \
  gyro_fps:=200 accel_fps:=200 \
  initial_reset:=true
```

关键 launch 参数说明：

- `enable_gyro / enable_accel` — 必须打开
- `unite_imu_method:=0` — **必须为 0**，节点订阅的是 split 的 `accel/sample` + `gyro/sample`，不订 `/imu`
- `rgb_camera.color_profile:=960x540x30` — 与 `config/RealSense_D435i_online.yaml` 里的内参对应

### 3.3 数据源 B：ros2 bag play 回放 -- (已验证)

无相机也能跑（用之前录的 bag）：

```bash
# 终端 2 —— 回放 bag
ros2 bag play /path/to/bag_directory
```

需要 bag 里有这三个 topic：

- `/camera/camera/color/image_raw` (`sensor_msgs/Image`)
- `/camera/camera/accel/sample` (`sensor_msgs/Imu`)
- `/camera/camera/gyro/sample` (`sensor_msgs/Imu`)

可以加 `--rate 0.5` 慢放或 `--rate 2.0` 加速。**`--clock` 不需要加**，节点用 `header.stamp` 不依赖 ROS sim 时间。

### 3.4 验证 pose 输出

```bash
# 终端 3 —— 验证
ros2 topic hz /dexslide/slam/pose
#   实机健康场景：~30 Hz； bag 回放 + 难场景（如 aurco close-up）：~15-25 Hz

ros2 topic echo /dexslide/slam/pose --once \
    --qos-reliability best_effort --qos-durability volatile
#   QoS flag 强制匹配，避免发现期竞态。看到一帧完整 PoseStamped 就 OK

ros2 run tf2_ros tf2_echo map camera_color_optical_frame
#   持续打印 4x4 transform
```

> **重要 — 关于 `ros2 topic echo` 的 QoS 坑**：节点 publisher 是 `BEST_EFFORT`（`SensorDataQoS`）。
> `ros2 topic echo` 在 jazzy 上默认会尝试自动匹配，但 **discovery 窗口偶尔会输给** publisher 已发但 subscriber 还没识别的竞态，结果一条都收不到。
> **建议**：手动加 `--qos-reliability best_effort --qos-durability volatile` 永久避免这个坑。`ros2 topic hz` 默认就 OK，不用加 flag。

### 3.5 实时检测指定 ArUco 的 world 坐标

`enable_aruco_world` 是实时 ArUco 检测开关。默认 `false` 时，阶段 B 只运行原来的 SLAM localize：发布 `/dexslide/slam/pose` 和 `map -> camera_color_optical_frame`，不会检测任何 ArUco，也不会读取 `tx_slam_tag.json`。只有启动时显式传 `enable_aruco_world:=true`，才会额外启动 ArUco world-pose 节点 /dexslide/aruco/world_pose。

阶段 A 的 `demos/aurco/tx_slam_tag.json` 只负责定义 world frame：它记录桌面/base ArUco 在 SLAM map 里的位姿，实时阶段会用它计算 `T_world_slam = inv(T_slam_base_tag)`。

实时要检测哪个 ArUco 由 `target_marker_id` 和 `aruco_yaml` 决定。默认检测的是 wrist marker：

- `target_marker_id:=10`
- `aruco_yaml:=/home/jzq/MyJob/DexSlide/umi_mono/example/calibration/aruco_config_wrist.yaml`

也就是说，默认不是检测标定时那块桌面 ArUco，而是检测 ArUco-10，并输出它在标定 world frame 下的 pose。内部坐标链路为：

```text
T_world_marker = inv(T_slam_base_tag) @ T_slam_camera @ T_camera_marker
```

启动方式：

```bash
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/path/to/map_atlas.osa \
  enable_aruco_world:=true \
  tx_slam_tag:=/path/to/demos/aurco/tx_slam_tag.json \
  target_marker_id:=10 \
  aruco_yaml:=/home/jzq/MyJob/DexSlide/umi_mono/example/calibration/aruco_config_wrist.yaml
```

查看输出：

```bash
ros2 topic echo /dexslide/aruco/world_pose --once \
    --qos-reliability best_effort --qos-durability volatile

ros2 run tf2_ros tf2_echo world aruco_marker
```

如果你要实时检测“和标定时同一个桌面 ArUco”，把 config 和 id 切到桌面标定配置。例如标定 tag 是 13 时：

```bash
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=/path/to/map_atlas.osa \
  enable_aruco_world:=true \
  tx_slam_tag:=/path/to/demos/aurco/tx_slam_tag.json \
  target_marker_id:=13 \
  aruco_yaml:=/home/jzq/MyJob/DexSlide/umi_mono/example/calibration/aruco_config.yaml
```

这种情况下输出的 `world -> aruco_marker` 理论上应接近 identity；实际会有 ArUco 检测噪声和实时 SLAM 抖动。

### 3.6 Launch 参数全表

| 参数                           | 类型     | 默认                                              | 说明                                                                                      |
| ---------------------------- | ------ | ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| `vocab`                      | string | `<fork>/Vocabulary/ORBvoc.txt`                  | ORB 词袋；阶段 A 和 B 必须用同一份                                                                  |
| `settings`                   | string | `<umi_mono>/config/RealSense_D435i_online.yaml` | 相机内参 + IMU 噪声                                                                           |
| `map_atlas`                  | string | （空）                                             | `.osa` 地图。空 = 从零起建图（调试用，生产必填）                                                           |
| `image_topic`                | string | `/camera/camera/color/image_raw`                | 订彩色图 topic                                                                              |
| `accel_topic`                | string | `/camera/camera/accel/sample`                   | 订加速度计                                                                                   |
| `gyro_topic`                 | string | `/camera/camera/gyro/sample`                    | 订陀螺仪                                                                                    |
| `pose_topic`                 | string | `/dexslide/slam/pose`                           | 发 PoseStamped topic 名                                                                   |
| `map_frame`                  | string | `map`                                           | tf 父坐标系                                                                                 |
| `camera_frame`               | string | `camera_color_optical_frame`                    | tf 子坐标系                                                                                 |
| `enable_aruco_world`         | bool   | `false`                                         | 实时 ArUco 检测总开关；默认 false 时只跑原 SLAM localize，不检测 ArUco、不读取 `tx_slam_tag`                  |
| `tx_slam_tag`                | string | （空）                                             | 阶段 A 产物 `demos/aurco/tx_slam_tag.json`；启用 `enable_aruco_world` 时必填                      |
| `camera_intrinsics`          | string | `example/calibration/d435i_960_540.json`        | ArUco 检测使用的相机内参                                                                         |
| `aruco_yaml`                 | string | `example/calibration/aruco_config_wrist.yaml`   | ArUco 字典和 marker 尺寸配置；默认用于 wrist ArUco-10                                               |
| `target_marker_id`           | int    | `10`                                            | 实时检测并输出 world pose 的 marker id                                                          |
| `world_frame`                | string | `world`                                         | ArUco world-pose 输出的父坐标系，由 `tx_slam_tag` 中的桌面/base ArUco 定义                             |
| `marker_frame`               | string | `aruco_marker`                                  | ArUco world-pose TF 子坐标系                                                                |
| `aruco_world_pose_topic`     | string | `/dexslide/aruco/world_pose`                    | 指定 ArUco 在 world frame 下的 `PoseStamped` 输出 topic                                        |
| `aruco_max_pose_dt`          | double | `0.1`                                           | 图像帧与 SLAM pose 的最大时间匹配差，单位秒                                                             |
| `max_lost_frames`            | int    | `900`                                           | 连续丢失 N 帧后 soft reset（清计数器，不重置地图）                                                        |
| `accel_gyro_pair_window_s`   | double | `0.020`                                         | accel/gyro 时戳差 ≤ 此值时配对                                                                  |
| `activate_localization_mode` | bool   | `false`                                         | 调 `ActivateLocalizationMode()`。默认 false 与 `gopro_slam.cc` 一致；地图修改只发生在内存里，磁盘 `.osa` 始终不变 |

---

## 4. 自动化烟囱测试

一条命令端到端验证（无需 D435i，用 bag 回放）：

```bash
bash /home/jzq/MyJob/DexSlide/umi_mono/scripts/test_realsense_topic_slam.sh \
  --map /data/codes/umi_mono_data/demos/mapping/map_atlas.osa \
  --bag /data/codes/umi_mono_data/aurco
```

脚本会：

1. 后台启 SLAM 节点
2. 等 10 s 让 atlas 加载
3. 后台启 `ros2 topic echo` 订阅 pose（带强制 QoS）
4. 后台跑 `ros2 bag play`，自动用 `ros2 bag info` 算 bag 时长
5. 等 bag 跑完 + 2 s 余量
6. SIGINT 干净停 bag + echo，sync 刷盘
7. 打印 pose 数 / relocalization 数 / atlas load 数 / session 原点
8. 判定通过：`pose count (sec>0) ≥ 100`（默认阈值，可 `--min-poses N` 修改）

退出码：`0 = PASS`，`1 = FAIL`。

预期：在 `aurco` 14 s bag 上 pose count 一般在 **120–320** 之间浮动（取决于重定位起步快慢）。

---

## 5. Python 端消费 pose

节点发的是标准 `geometry_msgs/PoseStamped`，下游可以直接 `rclpy` 订阅。**必须用系统 Python 3.10**：

```bash
conda deactivate 2>/dev/null
source /opt/ros/jazzy/setup.bash

/usr/bin/python3 - <<'PY'
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped

class Sub(Node):
    def __init__(self):
        super().__init__('pose_sniffer')
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            PoseStamped, '/dexslide/slam/pose',
            lambda m: print(f"t={m.header.stamp.sec}.{m.header.stamp.nanosec:09d} "
                            f"pos=({m.pose.position.x:.3f},"
                            f"{m.pose.position.y:.3f},"
                            f"{m.pose.position.z:.3f})"),
            qos)

rclpy.init(); rclpy.spin(Sub())
PY
```

如果你用 DexSlide 主程序里已有的 `dexslide.world_pose.SlamPoseSubscriber`（带 SLERP 时间对齐 + ring buffer），直接 import 即可，无须改：

```python
from dexslide.world_pose import SlamPoseSubscriber
sub = SlamPoseSubscriber()
sub.spin_in_thread()
T = sub.get_T_world_camera(t=capture_time)   # 4x4 numpy SE(3)，±100 ms 内 SLERP 插值
```

---

## 6. FAQ / 故障排查

### Q1：节点启动后日志只到 `Loading ORB Vocabulary` 就卡住

正常的。词袋 145 MB，加载 ~2 s 后会打 `Vocabulary loaded!` → `Atlas loaded!`。如果 60 s 还没动 → vocab 文件损坏，重跑 `setup_orbslam3_fork.sh`。

### Q2：`Atlas loaded` 之后 pose 一直不出

排查顺序：

1. `ros2 bag info <bag>` 确认 bag 里有那三个 topic 且消息数不为 0；
2. `ros2 topic hz /camera/camera/color/image_raw` 确认 image 流真的在动；
3. 看 SLAM 节点日志里有没有 `Session timestamp origin captured` —— 没有就说明数据根本没进来；
4. 检查 D435i 的 `unite_imu_method` 是否为 0。

### Q3：`ros2 topic echo /dexslide/slam/pose` 一条都收不到，但 `ros2 topic hz` 同时能看到 22 Hz

**QoS 自动匹配竞态。** 加上 flag：

```bash
ros2 topic echo /dexslide/slam/pose \
    --qos-reliability best_effort --qos-durability volatile
```

也可以用：`--qos-profile sensor_data` 一键设全。

### Q4：测试脚本 `pose count (sec>0): 0`

基本同 Q3。脚本 2026-05-24 之后的版本已经显式传 QoS 修好了；如果还在用老脚本，更新或手动加 flag。

### Q5：节点频繁打 `Fail to track local map` / `Relocalized!!`

- aurco / 桌面 close-up 之类**与建图视角差异很大**的场景固有现象；切换到与建图同视角的 bag / 实机就会稳定 30 Hz。
- 建图本身回环不足 / 光照变化大 → 重新建图。
- 关于 ORB-SLAM3 自身判定（`mnMatchesInliers<15 && isImuInitialized()`）触发 reset：默认 `activate_localization_mode:=false` 已经避开了 ATLAS reset 的最坏情况；不要轻易开 true。

### Q6：实机模式下帧率不到 30 Hz

- USB 3.x 端口确认（`lsusb -t` 看 `SuperSpeed`）
- CPU governor 锁高：`sudo cpupower frequency-set -g performance`
- 减 ORB features：`config/RealSense_D435i_online.yaml` 里 `ORBextractor.nFeatures` 从 1250 降到 1000
- 别在同机器跑 GPU 渲染 / rviz2 重叠

### Q7：链接错误 `libcurl.so.4: undefined reference to curl_easy_*@CURL_OPENSSL_4`

`CMakeLists.txt` 已加 `-Wl,--allow-shlib-undefined` 规避 Pangolin → libgdal → libcurl 传递依赖链上的版本化符号冲突。如果你改了 link flag 又出现，把这个选项加回去。

### Q8：rclpy import error

在用 anaconda Python。`conda deactivate` 后用 `/usr/bin/python3`。

### Q9：地图 `.osa` 跨机器搬过去能用吗

能。**前提**：两台机器的 `ORBvoc.txt` md5 一致（`5420bad0713bc97034dd2a9b2f0cc387`）；ORB-SLAM3 fork SHA 一致。

### Q10：怎么知道我的地图被修改了没（即使没 `--save_map`）

`md5sum` 跑前后对比：

```bash
md5sum map_atlas.osa     # 跑实时 SLAM 前
# ... 跑一段时间 ...
md5sum map_atlas.osa     # 后；md5 应一致
```

节点设计上从不写回 `.osa`（不传 save 路径），跑多久磁盘文件都不会变。

---

## 7. 已知限制 & 设计取舍

| 项                                                  | 现状                                    | 影响 / 缓解                                                                                                 |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Publisher QoS = BEST_EFFORT                        | `SensorDataQoS`                       | 上游 echo / 自定义订阅者需显式匹配；见 Q3                                                                              |
| `--allow-shlib-undefined` linker flag              | 加在 CMakeLists                         | 屏蔽 Pangolin→libgdal→libcurl 版本化符号；可能也会屏蔽真实错误，发版前手动跑 `ldd -r ...realsense_topic_slam_node`               |
| `activate_localization_mode` 默认 OFF                | 与 `gopro_slam.cc` 一致                  | 内存里地图可能被改但磁盘 `.osa` 永远不变；ORB-SLAM3 在 IMU_MONOCULAR + loaded atlas + inliers<15 时**不会**触发 ResetActiveMap |
| IMU ring buffer 容量 256                             | gyro 200 Hz 最多缓冲 1.28 s               | `LocalizeMonocular` 单帧 10–30 ms，远低于阈值                                                                   |
| 时间戳零对齐                                             | 第一帧 `header.stamp` 作原点（mutex 保护）      | 防 ORB-SLAM3 IMU 预积分 float 精度问题；发布时加回原点恢复 Unix 时间                                                        |
| MultiThreadedExecutor，image / IMU 双 callback group | image 回调里同步跑 SLAM；IMU 在另一线程灌环形 buffer | image 处理时 IMU 不阻塞                                                                                       |

---

## 8. 项目结构与代码位置（速查）

```
umi_mono/
├── config/
│   ├── RealSense_D435i.yaml              # 阶段 A Docker batch 用
│   └── RealSense_D435i_online.yaml       # 阶段 B 实时用（节点默认加载）
├── docs/
│   ├── DEPLOYMENT_AND_USAGE.md           # ← 本文件（整合版）
│   ├── USER_GUIDE.md                     # 上一版整体指南
│   ├── realsense_topic_slam_usage.md     # topic 方案详解
│   ├── setup_phase0~7_*.md               # 7 个阶段的细化复现说明
│   ├── VALIDATION_GUIDE.md               # 详细验证 SLO 表
│   ├── online_tracking_research.md       # 方案研究
│   ├── online_tracking_implementation.md # 43 任务追踪
│   ├── ros2_slam_pipeline_plan.md        # 离线 ROS2 流水线设计
│   └── ros2_pipeline_review.md           # 离线流水线 review（含已知坑）
├── external/                              # gitignored，需自行 build / clone
│   ├── Pangolin/                         # 0.8
│   ├── Sophus/                           # 1.22.10
│   ├── ORB_SLAM3_fork/                   # cheng-chi snapshot
│   │   ├── lib/libORB_SLAM3.so           # 354 MB
│   │   ├── Vocabulary/ORBvoc.txt         # 145 MB
│   │   └── Examples/Monocular-Inertial/
│   │       ├── gopro_slam                # 阶段 A Docker 也用同一份源码
│   │       ├── realsense_online          # 老的 librealsense-直驱二进制（legacy）
│   │       └── imu_ring_buffer.hpp       # 被新节点 #include 复用
│   └── cppzmq/                           # 仅老 pose_publisher_node 用
├── ros2_ws/
│   └── dexslide_slam_publisher/
│       ├── src/
│       │   ├── realsense_topic_slam_node.cpp    # ← 当前方案核心 (~434 行)
│       │   └── pose_publisher_node.cpp          # legacy ZMQ→ROS2 桥
│       ├── scripts/
│       │   └── aruco_world_pose_node.py         # 可选：实时 ArUco camera→SLAM→world pose
│       ├── launch/
│       │   ├── dexslide_slam_topics.launch.py   # ← 当前方案 launch
│       │   └── dexslide_slam_online.launch.py   # legacy
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── requirements.txt                     # Python 验证脚本可选依赖
│       └── README.md
├── scripts/
│   ├── check_host_env.sh                        # 一次性环境自检
│   ├── install_apt_deps.sh                      # apt 依赖
│   ├── build_pangolin.sh
│   ├── build_sophus.sh
│   ├── install_librealsense.sh
│   ├── setup_orbslam3_fork.sh                   # clone fork + pin SHA
│   ├── build_orbslam3_native.sh                 # 编译 libORB_SLAM3.so
│   ├── run_d435i_smoke.sh                       # D435i 实机烟囱测试
│   └── test_realsense_topic_slam.sh             # ← 端到端 bag 回放测试
├── scripts_slam_pipeline_ours/                  # 阶段 A: ROS1 stage 02-06
├── scripts_slam_pipeline_ros2/                  # 阶段 A: ROS2 stage 00, 07
├── run_mapping_and_desk_aruco_ros2.py           # ← 阶段 A 入口
└── ...
```

---

## 9. 相关文档（深读）

按职责区分：

- **整体使用**：`USER_GUIDE.md`（中文，已被本文件大部分整合）
- **当前方案细节**：`realsense_topic_slam_usage.md`（中文，节点专属手册）
- **环境准备**：`setup_phase0_environment.md`（主机基线、apt、Pangolin、Sophus、librealsense）
- **Fork 集成**：`setup_phase1_native_build.md`（ORB-SLAM3 fork 拉取与原生编译）
- **老路径实现**：`setup_phase2_realsense_online.md` ~ `setup_phase6_7_launch_validation.md`（5 篇）
- **Python 消费者**：`setup_phase5_python_consumer.md` + `dexslide/world_pose/`
- **验证 SLO**：`VALIDATION_GUIDE.md`（详细测试表 + 8 个 SLO 阈值）
- **离线流水线已知坑**：`ros2_pipeline_review.md`（IMU 重复、mp4 恒帧率漂移、stage 07 OOM 等）
- **方案选型背景**：`online_tracking_research.md`（为什么是 ORB-SLAM3 fork 而不是 stella_vslam / VINS / OKVIS2）

---

## 10. 命令速查卡（贴墙打印用）

```bash
# 一次性环境
bash umi_mono/scripts/check_host_env.sh
bash umi_mono/scripts/install_apt_deps.sh --apply
bash umi_mono/scripts/build_pangolin.sh
bash umi_mono/scripts/build_sophus.sh
bash umi_mono/scripts/setup_orbslam3_fork.sh
bash umi_mono/scripts/build_orbslam3_native.sh
/usr/bin/python3 -m pip install --user -r umi_mono/requirements_mapping.txt
source /opt/ros/jazzy/setup.bash
(cd umi_mono/ros2_ws && colcon build --packages-select dexslide_slam_publisher)

# 阶段 A：录数据 + 建图
ros2 bag record -o mapping /camera/camera/color/image_raw \
                           /camera/camera/accel/sample /camera/camera/gyro/sample
ros2 bag record -o aurco   /camera/camera/color/image_raw \
                           /camera/camera/accel/sample /camera/camera/gyro/sample
python run_mapping_and_desk_aruco_ros2.py <session_dir>

# 阶段 B：实时 SLAM
source /opt/ros/jazzy/setup.bash
source umi_mono/ros2_ws/install/setup.bash
ros2 launch dexslide_slam_publisher dexslide_slam_topics.launch.py \
  map_atlas:=<session>/demos/mapping/map_atlas.osa
# 另一终端：实机
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true unite_imu_method:=0 \
  rgb_camera.color_profile:=960x540x30
# 或：bag 回放
ros2 bag play <bag_dir>
# 验证
ros2 topic hz /dexslide/slam/pose
ros2 topic echo /dexslide/slam/pose --once \
  --qos-reliability best_effort --qos-durability volatile

# 一键自动化测试
bash umi_mono/scripts/test_realsense_topic_slam.sh \
  --map <session>/demos/mapping/map_atlas.osa \
  --bag <session_or_aurco_bag>
```
