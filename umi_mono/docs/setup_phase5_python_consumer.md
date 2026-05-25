# Phase 5 — Python 消费者 `SlamPoseSubscriber`（操作说明）

> 关联 tracker: `umi_mono/docs/online_tracking_implementation.md`
> 前置：Phase 3 (ROS2 bridge) 完成
> Phase 5 装上 Python 端 SLAM pose 消费者，挂在 `/dexslide/slam/pose`，给 Vedo viewer / 控制脚本提供线程安全的时间对齐 T_world_camera 查询

---

## 0. Phase 5 新增文件

```
dexslide/world_pose/
├── __init__.py                    # 导出 SlamPoseSubscriber
└── slam_pose_subscriber.py        # 主实现 ~200 行

tests/
├── conftest.py                    # rclpy session 初始化 fixture
└── test_slam_pose_subscriber.py   # 10 个 pytest 用例
```

---

## 1. 复现命令

```bash
# 必备：用系统 Python 3.10（不用 anaconda 3.13）
source /opt/ros/humble/setup.bash

# 跑单元测试（不需要 D435i、不需要 realsense_online 运行）
cd /data/codes/DexSlide
/usr/bin/python3 -m pytest tests/test_slam_pose_subscriber.py -v
# 期望：10 passed in <0.5s
```

---

## 2. Python 端使用示例

### 2.1 基础：连续订阅 + 拿最新 pose
```python
import sys
sys.path.insert(0, '/data/codes/DexSlide')  # 或者把 dexslide 装到 site-packages

from dexslide.world_pose import SlamPoseSubscriber

# 1) 构造（lazily rclpy.init）
sub = SlamPoseSubscriber()

# 2) 后台启动 ROS2 spin
sub.spin_in_thread()

# 3) 主线程做别的事；用 latest() 拿最新
import time
for _ in range(100):
    t_pose = sub.latest()  # Optional[(t, T)]
    if t_pose is not None:
        t, T = t_pose
        print(f"t={t:.3f}, x={T[0,3]:.3f}, y={T[1,3]:.3f}, z={T[2,3]:.3f}")
    time.sleep(0.033)

sub.stop()
```

### 2.2 时间对齐：给定时刻 t 查 T
```python
sub = SlamPoseSubscriber()
sub.spin_in_thread()

# 假设你的 main loop 用 perf_counter 时间；SLAM pose 用 ROS time
# 二者通过 NTP / clock_realtime 对齐
T = sub.get_T_world_camera(t=current_image_capture_time)
if T is not None:
    # T 是 4x4 numpy 数组
    palm_pos_in_world = T @ palm_offset_4x1
```

`get_T_world_camera(t=None)`:
- `t is None` → 返回 latest pose 如果未 stale
- `t is float` → 在 buffer 中二分查找 t1≤t≤t2 满足 t2-t1<100ms，否则返回 None；命中后 linear 插 translation + SLERP 插 rotation

### 2.3 检查健康
```python
if not sub.is_tracking():
    # 200ms 内无 message
    handle_tracking_lost()
```

`stale_after_seconds=0.2` 是默认；要更松：`SlamPoseSubscriber(stale_after_seconds=0.5)`。

---

## 3. API 全表

| Method | Signature | 行为 |
|--------|----------|------|
| `__init__` | `node_name='dexslide_consumer', topic='/dexslide/slam/pose', stale_after_seconds=0.2, buffer_size=300` | 懒加载 rclpy.init；创建 node + subscription |
| `spin_in_thread()` | `()` | idempotent；daemon thread 跑 SingleThreadedExecutor；非阻塞 |
| `stop()` | `()` | shutdown executor + node + join thread |
| `latest()` | `() -> Optional[Tuple[float, np.ndarray]]` | 返回最新 (t, 4x4 T)；buffer 空时 None |
| `is_tracking()` | `() -> bool` | True iff `time.monotonic() - last_recv ≤ stale_after_seconds` |
| `get_T_world_camera(t)` | `(t: Optional[float]=None) -> Optional[np.ndarray]` | t=None 返回 latest 若未 stale；t given 时插值（容差 ±100ms） |

---

## 4. 关键决策记录

### 4.1 为什么用 `/usr/bin/python3` 不用 anaconda
- 本机两个 Python：anaconda 3.13.9（在 PATH 前）和系统 /usr/bin/python3.10.12
- ROS2 Humble 的 rclpy 是用 Python 3.10 编译的，**绝对不能用 3.13 import**
- 用户跑命令时必须用绝对路径或 deactivate conda 后再 source ROS2
- 推荐快捷做法：在 `~/.bashrc` 或脚本头加：
  ```bash
  conda deactivate 2>/dev/null
  source /opt/ros/humble/setup.bash
  ```

### 4.2 为什么手写 SLERP / 四元数→旋转矩阵
- 系统 /usr/bin/python3 没装 scipy（apt 没自动给）
- anaconda 装了 scipy 但 Python 版本错
- 手写 SLERP/quat→rot 只占 20 行，没有任何外部依赖
- 优势：性能 + 零依赖，可移植
- 实现参考：维基百科的 Slerp 公式 + 反极点（dot < 0 时翻转 q2 取最短弧）

### 4.3 `rclpy.init()` 懒加载
- 构造 `SlamPoseSubscriber()` 时检查 `rclpy.ok()`，未初始化则自动 init
- 多次构造 SlamPoseSubscriber 不会重复 init（idempotent）
- 主程序若先 init 过也兼容
- 副作用：`rclpy.shutdown()` 由用户自己调（或进程退出时由 daemon thread 自然清掉）

### 4.4 `~/.ros/log` 不可写时自动 fallback
- 容器/Docker/某些 NFS 主目录下 `~/.ros/log` 可能不可写
- 模块在 init 前检测：如果不可写，`os.environ['ROS_LOG_DIR'] = '/tmp/roslog'`
- 不影响功能，只影响 log 落地位置

### 4.5 PoseStamped header.stamp 解析
- ROS2 builtin time 是 `(sec, nanosec)` 整数对
- 模块用 `msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9` 转 float seconds
- 与 realsense_online 的 ZMQ JSON 中 `"t"` 字段（float seconds）一致 —— bridge 节点透传

---

## 5. 单元测试设计

10 个测试覆盖 4 类：

### 数学正确性（3 个）
- `test_quaternion_to_rotmat_identity`: q=(0,0,0,1) → I
- `test_quaternion_to_rotmat_90deg_z`: q=(0,0,sin45,cos45) 把 x 轴转到 y 轴
- `test_slerp_endpoints`: α=0 返回 q1, α=1 返回 q2

### SLERP 鲁棒性（1 个）
- `test_slerp_antipodal_handling`: q1=(1,0,0,0), q2=(-1,0,0,0) 不发散

### Buffer 行为（2 个）
- `test_buffer_appends_and_latest`: 单次 push + latest
- `test_buffer_bounded`: push >maxlen, deque 自动丢老

### 时间对齐与失效（4 个）
- `test_get_T_world_camera_interpolation`: t=10 identity, t=11 translated+rotated, query t=10.5 → 中点 + 45deg
- `test_get_T_world_camera_out_of_range`: 远超 buffer 的 t → None
- `test_is_tracking_default_threshold`: 默认 0.2s 阈值切换
- `test_stale_after_seconds_custom`: 自定义 0.5s 阈值切换

**这些测试 100% 不依赖 D435i / realsense_online / ROS2 publisher。** 通过直接调 `sub._on_pose(fake_msg)` 注入 — 跟实际 rclpy callback 路径一致。

---

## 6. 失败时如何修复

### `ModuleNotFoundError: No module named 'rclpy'`
- 没 `source /opt/ros/humble/setup.bash`
- 或者用的是 anaconda python3（3.13）—— rclpy 不存在；用 `/usr/bin/python3`

### `ImportError: ... scipy ...`
- 不应该出现（模块手写 SLERP）；如果出了说明 Codex 误用了 scipy，回看 diff

### `pytest: error: unrecognized arguments`
- pytest 版本问题；本机 6.2.5 是 ROS Humble 自带的旧版
- 用 `python3 -m pytest -v` 而非 `pytest -v` 确保使用同一 Python

### `~/.ros/log: Permission denied`
- 应该被模块自动 fallback；如果没 fallback：`export ROS_LOG_DIR=/tmp/roslog`

### `is_tracking()` 一直 False
- bridge 节点没在跑：`ros2 topic info /dexslide/slam/pose` 检查 Publisher count
- realsense_online 没在跑 / 没 --publisher zmq：检查它的 stdout

### SLERP 给出错误结果
- 检查传入的 quaternion 顺序：模块用 (qx, qy, qz, qw)，与 ROS msg.pose.orientation 一致
- 检查是否归一化（模块内部 normalize）

---

## 7. 集成示意（live 时数据流）

```
[D435i camera]
        ↓ USB3
[realsense_online --publisher ros2]   ← Phase 0-4
        ↓ ZMQ tcp://*:5555 (JSON line)
[dexslide_slam_publisher pose_publisher_node]  ← Phase 3
        ↓ ROS2 /dexslide/slam/pose (PoseStamped @ 30Hz)
        ↓ tf2 map → camera_color_optical_frame
[Python SlamPoseSubscriber.spin_in_thread]  ← Phase 5（本阶段）
        ↓ 线程安全 ring buffer (deque[300])
[Vedo viewer / DexSlide main.py]
   get_T_world_camera(t=current_capture_time) → 4x4 SE(3)
```

---

## 8. 下一阶段

**Phase 6 — Launch + 配置 + 文档**（TASK-032 ~ TASK-035）

- TASK-032: `dexslide_slam_online.launch.py` 一键启动 realsense_online + bridge
- TASK-033: `config/RealSense_D435i_online.yaml`（基于 D435i.yaml + LoadAtlasFromFile）
- TASK-034: `SLAM_readme_mono.txt` 新增「在线追踪」章节
- TASK-035: `umi_mono/CLAUDE.md` 加 pinned SHA / build 命令 / 故障模式

预估 0.5 天。Phase 6 完成后 = 整条流水线落地，剩 Phase 7 验证。
