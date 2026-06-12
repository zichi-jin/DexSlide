# 2026-06-06 UMI Mono Docker / Jazzy / SLAM 排障工作日志

## 目标

跑通以下命令：

```bash
python run_mapping_and_desk_aruco_ros2.py ~/dexslide_data/session_20260606
```

期望产出：

- `~/dexslide_data/session_20260606/demos/mapping/map_atlas.osa`
- `~/dexslide_data/session_20260606/demos/aurco/tx_slam_tag.json`

## 今日完成的工作

### 1. 定位并修复 Docker daemon 无法连接

最初报错表现为：

- `Cannot connect to the Docker daemon at unix:///var/run/docker.sock`
- `docker pull`、`docker run` 全部失败。

排查结论：

- Docker CLI 已安装。
- 当前用户已在 `docker` 组中。
- `containerd.service` 正常。
- `docker.service` 启动失败。

进一步查日志后，发现根因是：

- `/etc/docker/daemon.json` 不是合法 JSON。
- 文件中混入了 `# ...` 注释，导致 `dockerd` 启动时报：
  - `invalid character '#' after object key:value pair`

处理结果：

- 修正了 Docker daemon 配置。
- Docker service 可以正常启动。

### 2. 修复 Docker 无法通过代理拉镜像

Docker service 启动后，新的失败表现为：

- `docker pull hello-world`
- `docker pull chicheng/orb_slam3:latest`

均访问 `registry-1.docker.io:443` 超时。

排查结论：

- 本机 HTTP 代理 `http://127.0.0.1:8889` 可用。
- 终端里的代理对 Docker daemon 无效。
- 需要给 `dockerd` 自身配置 systemd 代理。

处理结果：

- 为 `docker.service` 配置了：
  - `HTTP_PROXY=http://127.0.0.1:8889`
  - `HTTPS_PROXY=http://127.0.0.1:8889`
  - `NO_PROXY=localhost,127.0.0.1,::1`
- 验证通过：
  - `docker pull hello-world` 成功
  - `docker pull chicheng/orb_slam3:latest` 成功

结论：

- Docker 侧问题已经全部打通。
- 当前阻塞点已不在 Docker 网络层。

### 3. 确认当前机器 ROS2 发行版为 Jazzy，而不是 jazzy

运行环境核对结果：

- 机器实际存在：`/opt/ros/jazzy`
- 不存在：`/opt/ros/jazzy`

这意味着仓库中大量写死 `source /opt/ros/jazzy/setup.bash` 的文档与脚本说明，在本机上并不成立。

说明：

- 这会影响 `ros2 bag info`、测试脚本、README 操作步骤。
- 但它不是 `02_create_map` 崩溃的直接根因。

### 4. 修复 ROS2 Jazzy bag 解析 bug

在重新从原始 bag 生成 `demos/` 时，`stage00` 最初报错：

- `Error processing mapping: ROS2_jazzy`
- `Error processing aurco: ROS2_jazzy`

定位结果：

- 问题在 `umi_mono/scripts_slam_pipeline_ros2/ros2_bag_utils.py`
- `rosbags.typesys.Stores` 被错误写成：
  - `Stores.ROS2_jazzy`
- 当前 `rosbags` 正确枚举名是：
  - `Stores.ROS2_JAZZY`

今日代码修复：

- 将 typestore 选择逻辑改为兼容 fallback：
  - 优先 `ROS2_JAZZY`
  - 再退 `ROS2_jazzy`
  - 再退 `LATEST`

相关文件：

- `umi_mono/scripts_slam_pipeline_ros2/ros2_bag_utils.py`

修复后验证：

- `get_available_topics(...)` 可以正常读取 `mapping` bag
- `iter_deserialized_messages(...)` 可以正常读出 `sensor_msgs/Image`
- `stage00` 能重新生成：
  - `demos/mapping/raw_video.mp4`
  - `demos/mapping/imu_data.json`
  - `demos/aurco/raw_video.mp4`
  - `demos/aurco/imu_data.json`

### 5. 备份旧中间产物并重跑 session

为排除旧中间产物污染，执行了：

- 备份旧目录：
  - `~/dexslide_data/session_20260606/demos.bak_20260606_204057`
- 删除当前路径上的旧 `demos` 引用后重新跑：
  - `python run_mapping_and_desk_aruco_ros2.py /home/jzq/dexslide_data/session_20260606`

结论：

- 旧 `demos` 污染不是当前核心阻塞点。
- 即便重新从原始 bag 生成，中间产物重建后仍会卡在 `02_create_map`。

## 当前状态

### 已解决部分

以下问题均已解决：

- Docker daemon 无法启动
- Docker daemon 无法通过代理访问 Docker Hub
- `hello-world` 镜像拉取失败
- `chicheng/orb_slam3:latest` 镜像拉取失败
- `stage00` 无法解析 ROS2 Jazzy bag
- 旧 `demos` 中间产物污染的不确定性

### 当前真实阻塞点

当前唯一的核心阻塞点是：

- `02_create_map.py` 调用的 `gopro_slam` 在建图阶段崩溃

表现为：

- Docker 路径：`returncode=139`
- native 路径：同样复现崩溃，并最终出现 `Sophus ... omega: nan nan nan`

这说明：

- 问题不是 Docker 独有的。
- 问题不是代理、镜像、daemon 或用户权限。
- 问题已经进入 `ORB_SLAM3 / gopro_slam` 与当前 `mapping` 数据本身的匹配问题。

### 当前 session_20260606 的输入数据特征

已确认：

- 原始 `mapping` bag 为 ROS2 Jazzy / mcap
- 时长约：`257.23 s`
- 图像帧数：`6152`
- 重建视频实际 FPS：`23.92`
- IMU 计数：
  - accel：`40419`
  - gyro：`40012`

而当前 `RealSense_D435i.yaml` / pipeline 预期更接近：

- color：`30 FPS`
- IMU：`200 Hz`

### 当前判断

综合 Docker、native、日志三侧证据，当前更像是以下问题之一：

- 这次录制的 `mapping` bag 本身不适合当前 `gopro_slam` 建图链路。
- 图像 / IMU 时序或采样质量不稳定，导致 ORB_SLAM3 初始化与惯导融合多次 reset。
- `RealSense_D435i.yaml` 与当前这批录制数据特征不完全匹配。

当前没有证据表明：

- 单纯改 Docker 配置还能继续推进。
- 单纯换 jazzy / Jazzy 就能解决 `02_create_map` 崩溃。

## 关键日志现象

native 与 Docker 路径都出现过以下现象：

- `Failed to init`
- `Fail to track local map!`
- `IMU is not or recently initialized. Reseting active map...`
- `Empty IMU measurements vector!!!`
- 最终在 native 路径下出现：
  - `Sophus ensure failed`
  - `SO3::exp failed! omega: nan nan nan`

这些现象说明：

- 地图初始化会偶尔成功，但随即丢失 / reset。
- 惯导链路在当前数据上不稳定。
- 问题更偏向输入数据或 SLAM 参数层，而不是部署层。

## 明天建议优先做的事

### 方案 A：优先重录一套新的 mapping / aurco 数据（推荐）

这是最值得先做的路线。

建议步骤：

1. 使用 Jazzy 环境启动 RealSense：

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true \
  unite_imu_method:=0 \
  rgb_camera.color_profile:=960x540x30 \
  gyro_fps:=200 accel_fps:=200 \
  initial_reset:=true
```

2. 新建全新 session：

```bash
mkdir -p ~/dexslide_data/session_YYYYMMDD
cd ~/dexslide_data/session_YYYYMMDD
```

3. 重录两段 bag：

```bash
ros2 bag record -o mapping \
  /camera/camera/color/image_raw \
  /camera/camera/accel/sample \
  /camera/camera/gyro/sample

ros2 bag record -o aurco \
  /camera/camera/color/image_raw \
  /camera/camera/accel/sample \
  /camera/camera/gyro/sample
```

录制要求：

- `mapping` 至少 60 秒以上。
- 运动要慢，避免猛晃。
- 必须有明显回环。
- 场景要有纹理，避免大面积白墙 / 桌面纯色。
- 录制后先用 `ros2 bag info` 检查 color FPS 与 IMU 数量是否接近预期。

4. 对新 session 直接执行：

```bash
python run_mapping_and_desk_aruco_ros2.py ~/dexslide_data/session_YYYYMMDD
```

### 方案 B：如果不想重录，继续做参数 / 数据侧调试

如果必须继续使用 `session_20260606`，明天建议按这个顺序排：

1. 检查 `mapping` 录制时的相机输出实际是否稳定在 `960x540x30`。
2. 检查为什么重建视频只有 `23.92 FPS`。
3. 抽样检查 `imu_data.json` 中图像与 IMU 时间轴的对齐质量。
4. 比较 `RealSense_D435i.yaml` 是否需要按当前录制特征调整。
5. 在 `02_create_map.py` 中增加更强的失败诊断信息，而不是只看 `returncode=139`。

### 方案 C：整理环境文档与脚本

这部分不影响当前核心建图问题，但很值得做，避免再次踩坑：

1. 将仓库中所有写死 `jazzy` 的说明改为可配置或改为 `Jazzy`。
2. 修正测试脚本中默认的 ROS setup 路径。
3. 在 README 中单独注明：
   - Docker 需要 daemon 代理，而不是只给 shell 配代理。
   - ROS2 Jazzy 需要 `rosbags` typestore 支持。

## 今日产出

- 修复 Docker daemon 启动配置问题
- 修复 Docker daemon 代理配置问题
- 成功拉取：
  - `hello-world`
  - `chicheng/orb_slam3:latest`
- 备份旧中间产物：
  - `~/dexslide_data/session_20260606/demos.bak_20260606_204057`
- 修复 `Jazzy` bag typestore 兼容问题：
  - `umi_mono/scripts_slam_pipeline_ros2/ros2_bag_utils.py`
- 验证 `stage00` 重新生成成功
- 确认当前唯一核心阻塞点是 `gopro_slam` 在 `session_20260606` 的 `mapping` 数据上崩溃

## 接手时的最短结论

如果明天继续接手本任务，先记住这三点：

1. Docker 已经修好，不要再回头查 daemon / proxy。
2. Jazzy bag 解析 bug 已经修好，`stage00` 现在能跑。
3. 真正卡住的是 `stage02` 建图，优先考虑重录新的 `mapping` / `aurco` 数据，而不是继续纠缠当前这份 `session_20260606`。
