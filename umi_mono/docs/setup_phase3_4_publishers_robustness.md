# Phase 3 + Phase 4 — 发布后端与健壮性（操作说明）

> 关联 tracker: `umi_mono/docs/online_tracking_implementation.md`
> 前置：Phase 0/1/2 全部通过；`realsense_online` 可编译运行
> Phase 3 装上三种发布后端（stdout/zmq/ros2）；Phase 4 加上启动时校验与丢失追踪软重置

---

## 0. Phase 3 + 4 改了哪些文件

```
external/ORB_SLAM3_fork/
├── CMakeLists.txt                                         # +10 行 (libzmq pkg + realsense_online 加 zmq 源 + 链接)
├── Examples/Monocular-Inertial/
│   ├── realsense_online.cc                                # +120 行 (MD5/skew/LocalizeMonocular/dispatch)
│   ├── realsense_capture.hpp                              # +1 ctor 参数 (exposure_us) + 新方法
│   ├── realsense_capture.cpp                              # +configure_color_exposure
│   ├── zmq_pose_publisher.hpp                             # 新增 (TASK-018)
│   └── zmq_pose_publisher.cpp                             # 新增 (TASK-018)
external/cppzmq/                                           # 新增 git clone v4.10.0 header-only
umi_mono/ros2_ws/dexslide_slam_publisher/                  # 新增 ament_cmake 包 (TASK-019)
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── src/pose_publisher_node.cpp
│   └── README.md
umi_mono/tests/test_atlas_immutable.sh                     # 新增 (TASK-023)
```

---

## 1. Phase 3 — 三种发布后端

### 1.1 `--publisher stdout`（默认）
每帧 valid pose 打印：
```
pose <timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
```
用法：调试 / shell 管道处理。

### 1.2 `--publisher zmq`
绑定 `tcp://127.0.0.1:5555` 的 ZMQ PUB socket，每帧发 JSON line：
```json
{"t":<double>,"tx":<f>,"ty":<f>,"tz":<f>,"qx":<f>,"qy":<f>,"qz":<f>,"qw":<f>}
```
用法：任何能读 ZMQ SUB 的进程都能消费（Python `import zmq` 例子见下文）。

```python
# 简单 ZMQ SUB 消费者
import zmq, json
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt(zmq.SUBSCRIBE, b"")
while True:
    msg = sock.recv_string()
    d = json.loads(msg)
    print(d["t"], d["tx"], d["ty"], d["tz"])
```

### 1.3 `--publisher ros2`
等价于 `--publisher zmq`，但 stdout 提示要单独启动 bridge：

```bash
# 终端 1：bridge
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash
ros2 run dexslide_slam_publisher pose_publisher_node

# 终端 2：实际生产 pose 的进程
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v <vocab> -s <settings> -l <atlas> --publisher ros2

# 终端 3：消费
ros2 topic echo /dexslide/slam/pose
ros2 run tf2_ros tf2_echo world camera_color_optical_frame
```

Bridge 参数（节点 declare 的）：
| 参数 | 默认 | 含义 |
|------|------|------|
| `zmq_endpoint` | `tcp://127.0.0.1:5555` | ZMQ SUB 连接地址 |
| `pose_topic` | `/dexslide/slam/pose` | ROS2 publish topic |
| `world_frame` | `world` | （预留，bridge 当前不用） |
| `map_frame` | `map` | header.frame_id + tf parent |
| `camera_frame` | `camera_color_optical_frame` | tf child（REP-105） |
| `recv_timeout_ms` | 100 | ZMQ 非阻塞 recv 超时 |

更改示例：
```bash
ros2 run dexslide_slam_publisher pose_publisher_node \
  --ros-args -p pose_topic:=/dexslide/slam/pose_alt -p map_frame:=slam_map
```

---

## 2. Phase 4 — 健壮性增强

### 2.1 Vocabulary MD5 + 大小校验（TASK-022）
启动期打印：
```
Vocabulary MD5: 5420bad0713bc97034dd2a9b2f0cc387
```
若 ORBvoc.txt 文件大小 < 1MB 或 > 200MB，stderr 警告。**当前 ORBvoc.txt 的 MD5 = `5420bad0713bc97034dd2a9b2f0cc387`**（145MB）。新机器跑出不一致就要排查。

### 2.2 Startup timestamp skew 校验（TASK-021）
RsCapture.start() 成功后，跑 200 polls / 7 秒的预热阶段：
- 收集每帧的 (image_t, latest_imu_t) 对
- 计算 |image_t - imu_t| 的中位数
- 若中位数 > `--skew_abort_ms`（默认 5ms），stderr 打印 'Timestamp skew out of budget: X ms' + Shutdown + exit 3
- OK 则打印 `Startup skew median: X ms (budget Y ms) [OK]`

调试不通时尝试：
- `--skew_abort_ms 50` 给宽限度
- 检查 D435i 固件 ≥ 5.16（rs-fw-update）
- 在 RsCapture 里强制 `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`（已默认）

### 2.3 LocalizeMonocular + soft reset（TASK-024）
- `slam.LocalizeMonocular(...)` 替代 `slam.TrackMonocular(...)`
- 返回 `pair<SE3f, has_tracking>`
- 当 `has_tracking == false`：consecutive_lost++ + 不发 pose
- 当 consecutive_lost > `--max_lost_frames`（默认 900 = 30s @30Hz）：log `Triggering soft reset (lost > max)` + 重置计数器，**继续喂帧**让 ORB-SLAM3 内部 relocalizer 工作；不调 `slam.Reset()`，地图保持原样

### 2.4 RsCapture 曝光锁（TASK-025）
`--exposure_us N`（N > 0）：
- 找到 sensor name 含 "RGB" 或 "Color" 的传感器
- `set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0)`
- `set_option(RS2_OPTION_EXPOSURE, N)`
- 打印 `Exposure locked to Nus`
- 设备不支持时仅 stderr 警告，不退出

D435i 典型曝光值：室内日光下 `--exposure_us 8000`（8ms）即可。

### 2.5 Atlas 不变性测试脚本（TASK-023）
```bash
bash umi_mono/tests/test_atlas_immutable.sh <path/to/map_atlas.osa> [duration_seconds]
```
- 默认跑 60 秒，可改
- 算前 SHA-256 → 跑 realsense_online --load_map → 算后 SHA-256
- 相等 → PASS，否则 FAIL
- timeout 124 的退出码视为正常（脚本内 kill -TERM）

---

## 3. ZMQ → ROS2 链路调试

排查顺序：
1. **realsense_online 真在 PUB**：另起 Python `recv_string()` 看到 JSON 才算 OK
2. **bridge 真连上**：`ros2 topic info /dexslide/slam/pose` → Publisher count: 1
3. **bridge 在 republish**：`ros2 topic hz /dexslide/slam/pose` → 应该接近 30Hz
4. **TF 在 broadcast**：`ros2 run tf2_ros tf2_echo map camera_color_optical_frame` 应该看到位姿

若 ZMQ 不通：
- 确认 realsense_online 是 BIND（默认 endpoint），bridge 是 CONNECT
- 检查 `127.0.0.1` 是否被防火墙挡（一般本机不会）

---

## 4. 关键决策记录

### 4.1 为什么 ZMQ 是 PUB/SUB 而不是 REQ/REP
- 30Hz 单向数据流，订阅者多个时只要起多个 SUB 就行
- PUB/SUB 自然处理「订阅者掉线 → 重连」语义；REQ/REP 强配对，订阅者掉线就死了

### 4.2 为什么 bridge 是 SUB connect 而不是 bind
- 标准 ZMQ 模式：稳定的一方 bind，灵活的一方 connect
- realsense_online 是「pose 生产者」，**应当稳定地存在**（用户启动）
- bridge 是「消费/转发者」，**可以随时启停**
- 因此 PUB BIND + SUB CONNECT；如果反过来，bridge 重启时 PUB 端会丢消息

### 4.3 为什么 ROS2 bridge 是独立的 ament_cmake 包
- fork 是 GPL-3（继承自 ORB-SLAM3 上游 UZ-SLAMLab）
- ROS2 桥保持 BSD/MIT license 边界清洁 —— 我们不希望整个 DexSlide 被传染
- 也方便后续单独 release 或 ros2 launch 编排

### 4.4 为什么 vocab MD5 校验只是 informational 而不是 fail
- ORBvoc.txt 在不同源/不同年份的 fork 可能 MD5 不一致但内容等价（行尾、注释等差异）
- 真要严格比对，需要解析为 BoW 表后比较，不是 MD5 能做的
- 当前打印 MD5 让人能横向对比（一台机器与另一台），但不强制

### 4.5 为什么 LocalizeMonocular 而不是 TrackMonocular
- cheng-chi fork 扩展了 `LocalizeMonocular`，返回 `pair<SE3f, bool>` 二元组
- `bool` 反映了 ORB-SLAM3 内部的 `mState == OK || RECENTLY_LOST`
- 上游 TrackMonocular 单返 SE3f，无法显式区分 "valid pose" 和 "tracking lost identity"

---

## 5. 失败时如何修复

### bridge 报 `[zmq error] No such device or address`
- ZMQ endpoint 错误。确认 realsense_online 的 publisher 是 `zmq` 或 `ros2`
- 防火墙/IPv6 问题：尝试 `--ros-args -p zmq_endpoint:=tcp://0.0.0.0:5555`

### `ros2 topic hz` 显示 0Hz
- realsense_online 没在发：检查它是否真跑（`ps -ef | grep realsense_online`）
- 或者它在发但 bridge 没 connect 上：`ss -tnlp | grep 5555` 应该看到 PUB 在 listen

### `Triggering soft reset` 一直触发
- 相机朝着空白墙 / 太暗：换姿态、加灯光
- max_lost_frames 太小：调大 `--max_lost_frames 1800` (60s @30Hz)
- ORBvoc.txt 与 atlas 不匹配（用错版本）：检查 MD5

### `Timestamp skew out of budget`
- D435i 固件旧：rs-fw-update 升级到 ≥ 5.16
- USB 2.0 接口（应该用 3.x）
- `--skew_abort_ms 50` 临时放宽诊断

---

## 6. 已知限制 / 还没验证

| 项 | 状态 | 何时验证 |
|----|------|---------|
| TASK-017 live D435i 60s smoke | 🟡 待设备 | D435i 插上后 |
| TASK-023 atlas SHA-256 真正不变性 | 🟡 待数据 | 有 map_atlas.osa 后 |
| ZMQ 帧率（应该 30Hz）| 🟡 | live 时 `ros2 topic hz` |
| skew abort 真触发路径 | 🟡 | live 时 / 故意配错 timestamp domain |
| 曝光锁实际生效 | 🟡 | live 时 `rs-sensor-control` 验证 |

---

## 7. 下一阶段

**Phase 5 — Python 消费者**（TASK-026 ~ TASK-031）

- TASK-026: `dexslide/world_pose/__init__.py`
- TASK-027: `SlamPoseSubscriber` ring buffer + 基础订阅
- TASK-028: SLERP 时间对齐查询
- TASK-029: `is_tracking()` + stale 阈值
- TASK-030: `spin_in_thread()` daemon
- TASK-031: pytest 测试

预估 1.5 天。Phase 5 完成后，DexSlide Python 侧 Vedo viewer / 控制脚本可以读到 `T_world_camera`。
