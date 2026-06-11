# Mono SLAM Pipeline 使用说明（ROS1 + ROS2）

## 0. 核心结论（先看这个）
1. 当前 `run_slam_pipeline.py` 是 ROS1 路线，依赖 ROS1 `rosbag` 读取 `.bag`。
2. ROS2 `ros2 bag record` 产物是 **目录**（含 `metadata.yaml` + `.db3/.mcap`），不是 ROS1 单文件 `.bag`。
3. 因此 ROS2 场景有两条路：
   - 推荐：使用本仓库新增的 ROS2 流水线 `run_slam_pipeline_ros2.py`（直接读取 ROS2 bag 目录）。
   - 备选：先把 ROS2 bag 转成 ROS1 `.bag`，再走旧脚本。

## 1. 采集流程

### 1.1 环境准备
- 安装 ROS/ROS2、realsense2 SDK、realsense-ros、docker。
- 配置 conda 环境。
- ROS2 流水线额外依赖：
```bash
pip install rosbags
```

### 1.2 ROS1 采集（旧流程）
```bash        
roslaunch realsense2_camera rs_camera.launch
rostopic echo /camera/color/metadata
rostopic echo /camera/accel/sample
rostopic echo /camera/gyro/sample
rqt_image_view
rosbag record -o data /camera/color/image_raw /camera/accel/sample /camera/gyro/sample
```
- 建议先录制一个较长 mapping 数据（约 60s）。
- 再录一个用于基座标定的包，并命名为 `aurco.bag`（代码里是这个拼写）。

### 1.3 ROS2 采集（新流程）
启动 RealSense（示例）：
```bash
ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true unite_imu_method:=0
```
检查 topic：
```bash
ros2 topic list
ros2 topic hz /camera/camera/color/image_raw
ros2 topic echo /camera/camera/accel/sample
ros2 topic echo /camera/camera/gyro/sample
```
录制（你给的 D435i topic）：
```bash
ros2 bag record -o data /camera/camera/color/image_raw /camera/camera/accel/sample /camera/camera/gyro/sample
```
查看 bag：
```bash
ros2 bag info <bag_dir>


```

## 2. 会话目录命名规范

### 2.1 ROS1
```text
<session_dir>/
  aurco.bag
  data_001.bag
  data_002.bag
  ...
```

### 2.2 ROS2（推荐）
`ros2 bag record -o <name>` 会生成目录。建议：
```text
<session_dir>/
  aurco/                 # 用于 04a + 05 的 ArUco-13 标定录制
    metadata.yaml
    *.db3 / *.mcap
  data_2026_.../         # 演示数据
    metadata.yaml
    *.db3 / *.mcap
  data_2026_.../
    ...
```
说明：
- `aurco` 目录名要保留这个拼写（与现有代码一致）。
- `data*` 前缀会被后续阶段自动扫描。

ros2 run rqt_image_view rqt_image_view

## 3. 运行方式

### 3.1 ROS1 旧入口（不变）
```bash
python run_slam_pipeline.py <session_dir> --calibration_dir /data/codes/umi_mono/example/calibration
```

### 3.2 ROS2 新入口（新增）
```bash
python run_slam_pipeline_ros2.py <session_dir> --calibration_dir /data/codes/umi_mono/example/calibration
```
可选 topic 参数（默认已按 D435i ROS2 常见命名）：
```bash
python run_slam_pipeline_ros2.py <session_dir> \
  --image_topic /camera/camera/color/image_raw \
  --accel_topic /camera/camera/accel/sample \
  --gyro_topic /camera/camera/gyro/sample \
  --sensor_topic /sensor_data
```

## 4. 是否能“录成 ROS1 rosbag”再直接复用老代码？
结论：`ros2 bag record` 本身不能直接产出 ROS1 `.bag`。

### 方案 A（推荐）
直接使用本仓库新增 ROS2 读取流程（`run_slam_pipeline_ros2.py`）。

### 方案 B（备选）
先把 ROS2 bag 目录转 ROS1 `.bag`：
```bash
rosbags-convert /path/to/ros2_bag_dir --dst /path/to/output.bag
```
注意：
- `rosbags-convert` 支持 rosbag1/rosbag2 双向转换。
- 从 rosbag2 转 rosbag1 时，仅支持 ROS2 默认消息类型。

## 5. 关键输出
- `demos/mapping/map_atlas.osa`
- `demos/aurco/tx_slam_tag.json`
- `demos/data*/tag_detection_wrist.pkl`
- `<session_dir>/episode_*.hdf5`

## 6. 现有相机与标定配置
- `example/calibration/d435i_960_540.json`
- `example/calibration/aruco_config.yaml`
- `example/calibration/aruco_config_wrist.yaml`
- `config/RealSense_D435i.yaml`

## 7. 在线追踪 (Online tracking)

从 2026-05 起新增。把已建好的 .osa atlas 用于实时 SLAM 定位，输出位姿到 ROS2 / ZMQ。

前置：
- Phase 0~5 已部署。参考 docs/setup_phase{0..5}*.md。
- D435i 接 USB3，固件 >= 5.16。
- .osa atlas 已通过离线流水线（02_create_map.py + 03_batch_slam.py）建好。

### 7.1 一键 launch

```bash
source /opt/ros/jazzy/setup.bash
source /home/jzq/MyJob/DexSlide/umi_mono/ros2_ws/install/setup.bash
ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py \
  map_atlas:=/path/to/map_atlas.osa
```

另起终端验证：
```bash
ros2 topic hz /dexslide/slam/pose  # 应该 ~30 Hz
```

### 7.2 单独跑（不要 ROS2）

```bash
/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /home/jzq/MyJob/DexSlide/umi_mono/config/RealSense_D435i_online.yaml \
  -l /path/to/map_atlas.osa \
  --publisher stdout
```

### 7.3 Python 消费者

See docs/setup_phase5_python_consumer.md.

### 7.4 故障排查

- tracking 一直 lost: 检查 atlas 是否对应当前环境；vocab MD5 是否匹配
- skew abort: 升级 D435i 固件
- ZMQ 连不上: 确认 realsense_online 用 --publisher zmq 或 ros2

Full reference: docs/online_tracking_implementation.md + docs/online_tracking_research.md
