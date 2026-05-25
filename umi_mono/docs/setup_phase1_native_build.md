# Phase 1 — Fork 集成与原生构建（操作说明）

> 适用：DexSlide 在线追踪 (`online-tracking-mode`) 部署
> 关联 tracker: `umi_mono/docs/online_tracking_implementation.md`
> 前置：`setup_phase0_environment.md` 全部通过
> Phase 1 把 cheng-chi/ORB_SLAM3 fork clone 到本地、用 Phase 0 的外部 Pangolin 编译出 `gopro_slam` 和 `realsense_online` placeholder

---

## 0. 适用平台 / 前置

- Phase 0 全部 [OK]（特别是 `external/Pangolin/build/PangolinConfig.cmake` 存在）
- 网络：`HTTPS_PROXY=http://127.0.0.1:7890`（首次 clone fork 需要访问 GitHub）

---

## 1. 复现步骤

```bash
# 1. clone fork + pin SHA + 解压 vocabulary
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 \
  bash umi_mono/scripts/setup_orbslam3_fork.sh
# 期望末尾打印：SHA + branch + Vocabulary/ORBvoc.txt size = 145250924
# 注：cheng-chi/ORB_SLAM3 默认分支是 master，不是 main

# 2. 原生编译（首次 ~3 分钟，i7 + 20 cores 并行）
bash umi_mono/scripts/build_orbslam3_native.sh
# 期望末尾打印 Step | Status：
#   DBoW2     | OK | <size>
#   g2o       | OK | <size>
#   Sophus    | OK | <size>
#   Pangolin  | SKIPPED | external build
#   ORB_SLAM3 | OK | 353803512  (354MB libORB_SLAM3.so)

# 3. 验证 gopro_slam 跑得起来
external/ORB_SLAM3_fork/Examples/Monocular-Inertial/gopro_slam --help | head -25

# 4. 验证 realsense_online placeholder
external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online
# 期望：打印 "realsense_online placeholder (TASK-009). Future: live D435i + ORB-SLAM3 tracking."
```

---

## 2. Phase 1 完成判定

| TASK | 验收命令 | 期望 |
|------|----------|------|
| 006 | `git -C external/ORB_SLAM3_fork rev-parse HEAD; cat external/ORB_SLAM3_fork/.pinned_sha` | 两者相等 |
| 007 | `ls -la external/ORB_SLAM3_fork/lib/libORB_SLAM3.so` | 354MB |
| 007 | `ls external/ORB_SLAM3_fork/Examples/Monocular-Inertial/gopro_slam` | 可执行 |
| 007 | `ldd external/ORB_SLAM3_fork/Examples/Monocular-Inertial/gopro_slam \| grep "not found"` | 无输出 |
| 009 | `external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online; echo $?` | exit 0 + placeholder 串 |

---

## 3. 关键决策记录

### 3.1 为什么用 `cheng-chi/ORB_SLAM3` snapshot fork
研究报告 `online_tracking_research.md` 已论证。要点：
- 已经支持 `--load_map` + `--max_lost_frames`（这是 stage 03 batch 重定位用的 flag）
- 保留上游 `System::ActivateLocalizationMode()` + `TrackMonocular()` 公开 API
- 加了一个扩展 `LocalizeMonocular(im, t, vImuMeas)` 返回 `(pose, ok_flag)` 二元组
- 已经在 chicheng/orb_slam3 Docker 镜像里用过，原生编译只是把 Docker 内的 `gopro_slam` 搬出来

### 3.2 fork 不用 git submodule
原因：
- fork 是 snapshot（0 open issues、近 2 年无更新），不需要 `git submodule update --remote`
- 我们会在 fork 里直接加新 `.cc` 和 patch CMakeLists；submodule 跟踪规则会让 `git status` 变 messy
- `.pinned_sha` 文件 + `setup_orbslam3_fork.sh --sha <SHA>` 已经能复现版本控制

### 3.3 fork 的 `Thirdparty/Pangolin/` 是空目录
意外发现，但是 by design：fork 走 `find_package(Pangolin)`，期望外部提供。Phase 0.3 编译的 `external/Pangolin/build/` 在 Phase 1 build 时通过 `CMAKE_PREFIX_PATH` 接入。CMake config-mode lookup 找到 `PangolinConfig.cmake` 即可。

### 3.4 默认分支是 `master` 不是 `main`
GitHub 上 `cheng-chi/ORB_SLAM3` 的默认分支显示为 `master`（不是常见的 `main`）。`setup_orbslam3_fork.sh` 先试 `main`，缺则 fallback `master`。

### 3.5 编译用 `cmake -B build -S .`（不是 fork 自带的 build.sh）
fork 自带 `build.sh` 用 `mkdir build; cd build; cmake ..` —— 第二次跑会 `mkdir: cannot create directory 'build': File exists` 报错。我们的 wrapper 用 `cmake -B build` 形式，**幂等**，可以多次跑。

### 3.6 `--target realsense_online` 而不是 `--target all`
TASK-010+ 开发期间，我们只关心 `realsense_online`，可以 `cmake --build build -j --target realsense_online` 单独编译，比每次重新编全部快得多。

---

## 4. 失败时如何修复

### `setup_orbslam3_fork.sh` 报 "Could not resolve host: github.com"
- 检查 Clash 代理是否在跑：`ss -tnlp | grep 7890`
- 检查 env：`echo $HTTPS_PROXY`（如果空，重新 `export`）
- 测试：`HTTPS_PROXY=http://127.0.0.1:7890 git ls-remote https://github.com/cheng-chi/ORB_SLAM3`

### `build_orbslam3_native.sh` 报 "External Pangolin v0.8 missing"
回到 Phase 0：`bash umi_mono/scripts/build_pangolin.sh`。Pangolin v0.8 必须先于 fork 编译。

### libORB_SLAM3.so 链接出错（undefined reference to Eigen::xxx）
- 检查 Eigen 版本：`pkg-config --modversion eigen3` 应当 ≥ 3.3.0
- fork 假设 Eigen 3.x（22.04 默认 3.4.0 OK）
- 如果是 Sophus 链接错，确认用 `Thirdparty/Sophus`（fork 自带）而不是 external/Sophus。fork 的 cmake 应当自动选 Thirdparty 优先

### `ldd gopro_slam` 报某个 `*.so not found`
- pango_*.so 找不到：`export LD_LIBRARY_PATH=/data/codes/DexSlide/umi_mono/external/Pangolin/build:$LD_LIBRARY_PATH`
- 但 build 时 cmake 应当 RPATH 已经写好（看 `readelf -d gopro_slam | grep RUNPATH`）

### `realsense_online` 链接错
TASK-009 只链了 `${PROJECT_NAME}`（即 ORB_SLAM3）。后续 TASK-011/012 加 librealsense2 + cppzmq 链接时会更新 target_link_libraries。

---

## 附录 A — 编译产物地图

```
/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/
├── .pinned_sha                                  # b741dca...
├── lib/
│   └── libORB_SLAM3.so                          # 354MB (debug symbols)
├── Thirdparty/
│   ├── DBoW2/lib/libDBoW2.so                    # 75KB
│   ├── g2o/lib/libg2o.so                        # 780KB
│   ├── Sophus/build/SophusConfig.cmake          # header-only
│   └── Pangolin/                                # 空目录 (by design)
├── Examples/Monocular-Inertial/
│   ├── gopro_slam                               # 16MB, batch SLAM 用
│   ├── realsense_online                         # placeholder (TASK-009)
│   └── ... 其他 batch examples
├── Vocabulary/
│   └── ORBvoc.txt                               # 145MB, 解压自 ORBvoc.txt.tar.gz
├── build/                                       # CMake build dir
└── build_orbslam3.log                           # tee 的完整编译日志
```

---

## 附录 B — Codex 与 fork 协作的 lesson learned

实施过程中遇到 fork 的两个意外：
1. **Thirdparty/Pangolin 是空目录** → 第一版 `build_orbslam3_native.sh` 报 "missing CMakeLists" 错误；改为优雅 skip + `CMAKE_PREFIX_PATH` 接入
2. **默认分支是 master** → `setup_orbslam3_fork.sh` 加 main→master fallback

经验：对 snapshot fork 不能盲信 README 描述的步骤，要先 `git clone` 看一眼再写脚本。Codex 在 sandbox 内可以读到 clone 后的 fork 内容，但要给它充分的探索时间（同一 session 内 multi-turn）。

---

## 附录 C — 下一阶段

**Phase 2 — 真正写 `realsense_online.cc`**（TASK-010 ~ TASK-017）

- TASK-010: CLI 骨架（CLI11 解析所有 flag）
- TASK-011: `imu_ring_buffer.hpp` lock-free SPSC 队列 + 单元测试
- TASK-012: `realsense_capture.hpp/cpp` librealsense2 pipeline 封装
- TASK-013: ORB-SLAM3 System 构造 + 加载地图 + `ActivateLocalizationMode()`
- TASK-014: 主追踪循环（stdout 打印 pose）
- TASK-015: pose NaN 校验
- TASK-016: SIGINT 干净关停
- TASK-017: End-to-end smoke（D435i live → stdout pose）

Phase 2 完结后会有可用的 **stdout-only 在线追踪 binary**。
