# DexSlide SLAM 在线追踪 — Live 验证清单

> 文档版本：2026-05-19
> 受众：D435i 已接好、准备做完整 end-to-end 验证的人
> 相关文档：`USER_GUIDE.md`、`online_tracking_implementation.md`

本文档以**正确的执行顺序**列出所有需要 live 验证的步骤。**没有 `.osa` 地图就不能跑大部分验证**——所以中间有一步「先建图」是必须的，不能跳过。

---

## 目录

- [0. 验证决策树](#0-验证决策树)
- [1. 阶段 A — 不需要 D435i / 不需要地图（dry checks）](#1-阶段-a--不需要-d435i--不需要地图dry-checks)
- [2. 阶段 B — 只需要 D435i（不需要地图）](#2-阶段-b--只需要-d435i不需要地图)
- [3. 阶段 C — 录建图数据 + 跑离线流水线（前置）](#3-阶段-c--录建图数据--跑离线流水线前置)
- [4. 阶段 D — 在线追踪 live 验证（需要 D435i + 地图）](#4-阶段-d--在线追踪-live-验证需要-d435i--地图)
- [5. 阶段 E — 离线 playback 验证（需要录像 + 地图，不需要 D435i 实时）](#5-阶段-e--离线-playback-验证需要录像--地图不需要-d435i-实时)
- [6. 阶段 F — 长跑 / 半自动测试（需要 D435i + 地图 + 时间/操作员）](#6-阶段-f--长跑--半自动测试需要-d435i--地图--时间操作员)
- [7. 失败时怎么定位](#7-失败时怎么定位)
- [8. 总览速查表](#8-总览速查表)

---

## 0. 验证决策树

```
                    手头有什么？
                        │
        ┌───────────────┴────────────────┐
        没有 D435i                    有 D435i
        │                                │
        阶段 A                          阶段 A → 阶段 B
        （ dry checks ）              （ 加 D435i 流参数 ）
                                          │
                                  有没有跑过建图？
                                          │
                              ┌───────────┴───────────┐
                              没有                       有
                              │                          │
                              阶段 C（先录数据 + 跑       直接进 阶段 D
                              离线建图流水线）             和 E、F
                                          │
                                          ▼
                                      阶段 D, E, F
```

**总时间预算**：

- 阶段 A：~5 分钟
- 阶段 B：~5 分钟
- 阶段 C：~20-40 分钟（含录像 + 离线建图）
- 阶段 D：~5 分钟（不算 30min benchmark）
- 阶段 E：~10-30 分钟（取决于录像长度）
- 阶段 F：30 分钟 ~ 2 小时

---

## 1. 阶段 A — 不需要 D435i / 不需要地图（dry checks）

### 1.1 主机环境

```bash
bash /data/codes/DexSlide/umi_mono/scripts/check_host_env.sh && echo PASS_A1
```

**期望**：表格所有行 `[OK]` 或 `[WARN]`（Python 3.13 是 WARN，无所谓），末尾 `PASS — host environment OK`，最后 echo `PASS_A1`。

**失败**：脚本会列出每条 `[MISSING]` 对应的 `sudo apt install` 命令，按提示装。

---

### 1.2 编译产物存在

```bash
ls /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/lib/libORB_SLAM3.so
ls /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online
ls /data/codes/DexSlide/umi_mono/external/Pangolin/build/libpango_core.so
ls /data/codes/DexSlide/umi_mono/ros2_ws/install/dexslide_slam_publisher/lib/dexslide_slam_publisher/pose_publisher_node
pkg-config --modversion realsense2          # ≥ 2.55
echo "PASS_A2 if all above succeeded"
```

**失败**：哪个缺就回 `USER_GUIDE.md §3.2` 那一步重跑。

---

### 1.3 二进制可启动 + 无设备优雅退出

```bash
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online --help | head -10
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/test_imu_ring_buffer && echo PASS_A3
```

**期望**：`--help` 输出 9 个 flag；`test_imu_ring_buffer` 末尾 `TEST PASS`。

---

### 1.4 Python 单元测试

```bash
source /opt/ros/humble/setup.bash
cd /data/codes/DexSlide
/usr/bin/python3 -m pytest tests/test_slam_pose_subscriber.py -v 2>&1 | tail -5
```

**期望**：`10 passed in <0.5s`。

---

### 1.5 错误处理（vocab mismatch）

```bash
bash /data/codes/DexSlide/umi_mono/tests/test_vocab_mismatch.sh
```

**期望**：`TASK-042 PASS: wrong vocab failed gracefully (exit 255)`。

---

### 1.6 ROS2 launch dry-run

```bash
source /opt/ros/humble/setup.bash
source /data/codes/DexSlide/umi_mono/ros2_ws/install/setup.bash
ROS_LOG_DIR=/tmp/roslog ros2 launch dexslide_slam_publisher dexslide_slam_online.launch.py --show-args 2>&1 | head -20
```

**期望**：列出 `vocab` / `settings` / `map_atlas` / `exposure_us` / `pose_topic` / `zmq_endpoint` / `log_to_file` 7 个 LaunchArgument。

---

**阶段 A 完成判定**：以上 1.1–1.6 全部 PASS。**这一阶段不需要任何硬件**，是"代码完整性"自检。

---

## 2. 阶段 B — 只需要 D435i（不需要地图）

> 前置：D435i 接 USB 3.x、固件 ≥ 5.16、`rs-enumerate-devices` 能看到。

### 2.1 TASK-005 D435i 烟囱

```bash
bash /data/codes/DexSlide/umi_mono/scripts/run_d435i_smoke.sh
```

**期望**：

```
color FPS=29.x, accel FPS=199.x, gyro FPS=199.x, timestamp_domain=GLOBAL_TIME
PASS
```

**失败定位**：

- `No device found` → 检查 `lsusb | grep Intel`、换 USB3 端口
- color FPS < 27 → USB 带宽问题、USB Hub 不行、换直连
- timestamp_domain ≠ GLOBAL_TIME → D435i 固件旧，跑 `rs-fw-update -l` 升到 ≥ 5.16

---

### 2.2 realsense_online live（无地图）

```bash
timeout 12 /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml \
  --publisher stdout 2>&1 | tail -15
```

**期望**：

- 看到 `Vocabulary MD5: 5420bad0713bc97034dd2a9b2f0cc387`
- 看到 `Exposure: auto`
- 看到 `Localization mode active, atlas read-only`
- 看到 `Startup skew median: ... ms (budget 5 ms) [OK]`
- **没看到 pose 行**（因为没 atlas、SLAM 不会初始化）—— **正常**
- 12 秒后超时退出

> 这一步只验证 binary live 跑起来不 crash、各种启动期检查都过。**不期望出 pose**——只有有 atlas + ActivateLocalizationMode 关掉 mapping 前提下，并且已经在已建图区域中，才会出 pose。

**失败定位**：

- `Timestamp skew out of budget` → 临时改 `--skew_abort_ms 50`，然后查为何 D435i 时钟域有问题
- segfault → 把 stderr 贴出来诊断

---

**阶段 B 完成判定**：D435i 流参数正常、`realsense_online` live 跑起不 crash。

---

## 3. 阶段 C — 录建图数据 + 跑离线流水线（前置）

> 这一步**是必需的**：没有 `.osa` 地图，阶段 D/E/F 大部分都跑不了。

### 3.1 准备 ArUco-13 标定板

到 `umi_mono/example/calibration/aruco_config.yaml` 看 tag id 和 size（默认 id=13、size=0.16 m）。打印 16cm × 16cm 的 ArUco-13 marker（白纸黑边都行）。

### 3.2 录制 session

```bash
source /opt/ros/humble/setup.bash

# 启动 D435i ROS2 driver
  ros2 launch realsense2_camera rs_launch.py \
    enable_gyro:=true enable_accel:=true \
    unite_imu_method:=0 \
    rgb_camera.color_profile:=960x540x30 \
    gyro_fps:=200 accel_fps:=200 \
    enable_depth:=false enable_infra1:=false enable_infra2:=false
```

另起一个终端：

```bash
# 准备 session 目录
SESSION=~/dexslide_data/session_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SESSION"
cd "$SESSION"

# 1) 先录 ArUco 标定（10 秒；把打印好的 tag 摆桌面、相机对着 tag 慢慢晃）
ros2 bag record -o aurco \
  /camera/camera/color/image_raw \
  /camera/camera/accel/sample \
  /camera/camera/gyro/sample
# 按 Ctrl-C 停止录制（默认 ~10s 就够）

# 2) 再录建图（60-90 秒；在场景中绕一圈、回到起点形成回环）
ros2 bag record -o data_001 \
  /camera/camera/color/image_raw \
  /camera/camera/accel/sample \
  /camera/camera/gyro/sample
# Ctrl-C 停止

# 看下录的什么
ls -la
# 期望：aurco/  data_001/  两个目录，里面都有 metadata.yaml 和 .db3 文件
```

**建图诀窍**：

- 慢速移动（< 0.5 m/s）
- 平移 + 旋转结合，不要纯转头
- 回到起点形成回环
- 避免大白墙、避免相机看天

---

### 3.3 跑离线流水线

```bash
# 装 rosbags（一次性）
pip install rosbags

# 跑流水线
cd /data/codes/DexSlide
conda deactivate
run_mapping_and_desk_aruco_ros2.py ~/dexslide_data/session_<...>
```

这一步会调起 Docker `chicheng/orb_slam3:latest`。**第一次运行会 docker pull**（~300 MB，国内网络可能要代理）。

完整流水线包括 stage 00–07。建图核心是 **stage 02** 和 **stage 03**。

---

### 3.4 验证地图产出

```bash
SESSION=~/dexslide_data/session_<...>
ls -la "$SESSION/demos/mapping/map_atlas.osa"
# 期望：文件存在，大小 > 1 MB（典型 5-50 MB）
ls "$SESSION/demos/mapping/camera_trajectory.csv"
# 期望：stage 03 的 baseline trajectory，给 TASK-037 ATE 用
ls "$SESSION/demos/data_001/camera_trajectory.csv"
# 期望：data demo 的 batch 重定位结果

# 记下 atlas 路径，后续 D/E/F 都用它
export MAP_ATLAS=$SESSION/demos/mapping/map_atlas.osa
echo "MAP_ATLAS = $MAP_ATLAS"
```

---

### 3.5 常见建图失败

| 现象                                                | 原因               | 修复                                                  |
| ------------------------------------------------- | ---------------- | --------------------------------------------------- |
| `docker pull` 卡 / 超时                              | 国内网络             | 设代理：`export HTTPS_PROXY=http://127.0.0.1:7890`      |
| `map_atlas.osa` 没生成                               | stage 02 失败      | 查 `mapping/slam_stderr.txt`：常见是 IMU 不够 / 回环没拍到 / 太快 |
| `mapping/slam_stderr.txt` 报 "IMU NOT INITIALIZED" | IMU 数据不够 / 太短    | 重新录，至少 60s 含明显运动                                    |
| atlas 生成但 trajectory 长度短                          | 大段 tracking lost | 重新录、慢一点、加照明                                         |

---

**阶段 C 完成判定**：`$MAP_ATLAS` 文件存在，`demos/data_001/camera_trajectory.csv` 也存在（后者给阶段 E 的 ATE 测试）。

---

## 4. 阶段 D — 在线追踪 live 验证（需要 D435i + 地图）

> 前置：阶段 C 完成（已有 `$MAP_ATLAS`）。把 D435i 拿到**建图时**走过的同一片区域。

### 4.1 TASK-017 live realsense_online + atlas（60s）

```bash
timeout 65 /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online \
  -v /data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt \
  -s /data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml \
  -l "$MAP_ATLAS" \
  --publisher stdout 2>&1 | tee /tmp/task017.log | grep -c "^pose " &
LIVE_PID=$!

# 在另一终端，拿着 D435i 慢慢扫已建图区域；60 秒后回来看结果
wait $LIVE_PID

# 数 pose 行数
N=$(grep -c "^pose " /tmp/task017.log)
echo "valid pose lines: $N (target: ≥ 1500)"
[ "$N" -ge 1500 ] && echo "TASK-017 PASS" || echo "TASK-017 FAIL"
```

**期望**：60 秒内 ≥ 1500 行 `pose ...`，每行有 8 个数。

**失败定位**：

- < 100 行：tracking 一直 lost。检查相机是否对准了**建图时的同一片区域**（光照、视角差异都敏感）
- ~500 行：偶尔 lost。可以接受，但 SLO 不达标。降速、加照明
- > 1500：✅

---

### 4.2 TASK-038 30 分钟 benchmark

```bash
bash /data/codes/DexSlide/umi_mono/tests/benchmark_30min.sh "$MAP_ATLAS" 30
```

**期望**：

- 末尾打印 `TASK-038 PASS: p99 inter-msg = X ms (≤50), p99 latency = Y ms (≤33)`
- `/tmp/benchmark_30min_<timestamp>.json` 生成

**实测时长**：30 分钟（**长**）。期间相机最好保持在建图区域内、避免长期 lost。

**失败定位**：

- p99 inter-msg > 50ms：USB 带宽或 CPU 不够。USB3 接直连、`cpupower frequency-set -g performance`、减小 `ORBextractor.nFeatures` 到 1000
- p99 latency > 33ms：同上 + 检查 SLAM thread 调度（`top -H -p <pid>` 看每线程负载）

---

### 4.3 TASK-039 headless 测试

```bash
bash /data/codes/DexSlide/umi_mono/tests/test_headless.sh "$MAP_ATLAS"
```

**期望**：30 秒后打印 `TASK-039 PASS`，`/tmp/headless_run.log` 中无 X11/Xlib/DISPLAY 字样。

**失败定位**：

- log 里出现 X11 错误 → Pangolin 在 viewer 关闭情况下还在尝试创建窗口；重新编译时加 `-DUSE_PANGOLIN=OFF`

---

**阶段 D 完成判定**：TASK-017 / 038 / 039 全部 PASS。

---

## 5. 阶段 E — 离线 playback 验证（需要录像 + 地图，不需要 D435i 实时）

> 前置：阶段 C 完成，有 `$MAP_ATLAS` + `$SESSION/demos/data_001/{raw_video.mp4, imu_data.json}`。
> 这一阶段拔下 D435i 也能跑（用 playback mode）。

### 5.1 TASK-037 / TASK-008 ATE vs Docker

```bash
SESSION=~/dexslide_data/session_<...>

source /opt/ros/humble/setup.bash
cd /data/codes/DexSlide
/usr/bin/python3 umi_mono/tests/test_ate_vs_docker.py \
  --recording "$SESSION" \
  --duration 60 \
  --tolerance_cm 2.0
```

**期望**：`TASK-037 PASS: ATE = X cm (tolerance 2.0)`，X 应该 < 2 cm。

**失败定位**：

- ATE > 2 cm：原生 binary 跟 Docker 输出有显著差异 → 检查是不是 ActivateLocalizationMode 被意外跳过（看 stderr 是否有 `Localization mode active` 这行）
- 脚本说 "no overlap" → 时间戳对不齐，可能 playback mp4 起始时间 ≠ Docker 输出起始时间；查 stage 03 输出的 csv 第一列

---

### 5.2 TASK-023 / TASK-041 atlas SHA-256 不变性

```bash
# 短测（60s）
bash /data/codes/DexSlide/umi_mono/tests/test_atlas_immutable.sh "$MAP_ATLAS" 60

# 长测（1 小时，TASK-041）
bash /data/codes/DexSlide/umi_mono/tests/test_atlas_immutable.sh "$MAP_ATLAS" 3600
```

**期望**：`PASS: atlas SHA-256 unchanged (<hash>)`。

**失败**：`FAIL: atlas mutated <pre> -> <post>` → 严重问题。说明 ActivateLocalizationMode 没生效或被绕过；检查 fork commit SHA 是否对、settings yaml 是否对。

---

**阶段 E 完成判定**：TASK-037、TASK-023/041 全部 PASS。

---

## 6. 阶段 F — 长跑 / 半自动测试（需要 D435i + 地图 + 时间/操作员）

### 6.1 TASK-040 2 小时内存测试

```bash
bash /data/codes/DexSlide/umi_mono/tests/test_memory_2h.sh "$MAP_ATLAS"
```

**期望**：2 小时后 `TASK-040 PASS: RSS growth = X%`，X < 5%。`/tmp/memory_2h.csv` 有 121 行采样。

**实测时长**：2 小时。可以挂着 ssh 跑，相机摆固定位置即可（lost 也无所谓，只测内存）。

**失败定位**：

- 增长 > 5%：用 `valgrind --tool=massif` 或 heaptrack 在短时间内对比；常见嫌疑是 ORB-SLAM3 Local Mapping 残留 KeyFrame
- 进程崩了：看 `/tmp/memory_2h.csv` 末尾时间戳，与 `dmesg` 对比

---

### 6.2 TASK-043 recovery 3 秒测试（半自动，需要操作员）

```bash
bash /data/codes/DexSlide/umi_mono/tests/test_recovery_3s.sh "$MAP_ATLAS"
```

脚本会：

1. 启动 `realsense_online`，让 SLAM 稳定 10 秒
2. 提示你"现在用手遮住镜头 3 秒，然后放开"
3. 测量从放开到再出 pose 的时间
4. 时长 < 5s → PASS

**实测时长**：~30 秒（含等待）。

**失败定位**：

- 恢复 > 5s → 当前场景与建图差太远 / 建图不够鲁棒。可以试加大 `--max_lost_frames 1800`，但根本治法是补建图

---

**阶段 F 完成判定**：TASK-040 + TASK-043 PASS。

---

## 7. 失败时怎么定位

| 失败现象                              | 第一步查                                                             | 第二步查                                                        |
| --------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| `No device found`                 | `lsusb \| grep Intel`                                            | 换 USB 端口、查 udev                                             |
| `Vocabulary file size suspicious` | 是不是传错了 vocab 路径                                                  | `md5sum <vocab>` 跟 `5420bad...` 比对                          |
| `Timestamp skew out of budget`    | D435i 固件版本（`rs-enumerate-devices`）                               | 升固件、临时调 `--skew_abort_ms 50`                                |
| `No atlas to load`                | `ls -la $MAP_ATLAS`                                              | 重跑 stage 02                                                 |
| `Atlas mutated`                   | settings yaml 是不是用了 `_online.yaml` 派生版                           | 重新编译 fork（pinned SHA）                                       |
| tracking 一直 lost                  | 当前场景 vs 建图区域差异                                                   | 重新建图，确保有回环                                                  |
| p99 latency > 33ms                | USB 接口、CPU governor                                              | 减 `nFeatures` / 锁 CPU 频率                                    |
| ROS2 topic 0Hz                    | `realsense_online` 是不是用 `--publisher zmq` / 是不是 launch 没起 bridge | `ps -ef \| grep -E "realsense_online\|pose_publisher_node"` |
| Python 端 `import rclpy` 错         | 用的不是 `/usr/bin/python3`                                          | `conda deactivate && /usr/bin/python3 ...`                  |

---

## 8. 总览速查表

| #      | 阶段       | TASK          | 命令                                              | 时间            | 前置                 |
| ------ | -------- | ------------- | ----------------------------------------------- | ------------- | ------------------ |
| A1     | dry      | env           | `check_host_env.sh`                             | 1s            | 装                  |
| A2     | dry      | binary 存在     | `ls .../{libORB_SLAM3.so,realsense_online,...}` | 1s            | 编译                 |
| A3     | dry      | binary 启动     | `realsense_online --help; test_imu_ring_buffer` | 5s            | A2                 |
| A4     | dry      | Python 单元     | `pytest tests/test_slam_pose_subscriber.py`     | 1s            | env                |
| A5     | dry      | vocab 错处理     | `test_vocab_mismatch.sh`                        | 10s           | A2                 |
| A6     | dry      | launch 模拟     | `ros2 launch ... --show-args`                   | 2s            | colcon build       |
| **B1** | D435i    | **TASK-005**  | `run_d435i_smoke.sh`                            | 10s           | D435i              |
| B2     | D435i    | live no-atlas | `timeout 12 realsense_online ...`               | 12s           | D435i              |
| **C**  | **建图**   | **离线流水线**     | **`run_slam_pipeline_ros2.py`**                 | **20-40 min** | **D435i + Docker** |
| **D1** | live     | **TASK-017**  | `realsense_online -l $MAP_ATLAS` 60s            | 60s           | C                  |
| **D2** | live     | **TASK-038**  | `benchmark_30min.sh $MAP_ATLAS 30`              | 30 min        | C                  |
| **D3** | live     | **TASK-039**  | `test_headless.sh $MAP_ATLAS`                   | 30s           | C                  |
| **E1** | playback | **TASK-037**  | `test_ate_vs_docker.py --recording $SESSION`    | 1-3 min       | C 录像               |
| **E2** | playback | **TASK-023**  | `test_atlas_immutable.sh $MAP_ATLAS 60`         | 60s           | C                  |
| E3     | playback | **TASK-041**  | `test_atlas_immutable.sh $MAP_ATLAS 3600`       | 1 hr          | C                  |
| **F1** | long     | **TASK-040**  | `test_memory_2h.sh $MAP_ATLAS`                  | 2 hr          | C                  |
| **F2** | manual   | **TASK-043**  | `test_recovery_3s.sh $MAP_ATLAS`（操作员遮挡）         | 30s           | C + 人              |

**最小可行验证集**（约 1 小时）：A1-A6 → B1 → C → D1 → E2 → F2

**完整验证集**（约 4 小时）：上面 + D2 + D3 + E1 + E3 + F1

---

## 9. 验证通过后

把以下信息存档（写一个 `validation_report_<date>.md`）：

- D435i 序列号 + 固件版本
- 建图 session 路径 + 时长 + 场景描述
- 各 TASK PASS/FAIL + 关键指标（ATE 多少、p99 延迟多少、RSS 增长多少）
- 失败的 TASK 的复现路径

这个报告下次升级 fork / 改 Pangolin / 换机器时可以直接做 regression 对比。
