# Phase 0 — 主机环境基线（操作说明）

> 适用：DexSlide 在线追踪 (`online-tracking-mode`) 部署
> 关联 tracker: `umi_mono/docs/online_tracking_implementation.md`
> Phase 0 完成 5 个 TASK，把主机从 vanilla Ubuntu 22.04 准备到可以编译 ORB-SLAM3 fork 的状态

---

## 0. 适用平台

- **OS**: Ubuntu 22.04 LTS（22.04.x 任意小版本）
- **架构**: x86_64
- **特权**: 当前用户能 `sudo`
- **网络**:
  - 出网：能访问 GitHub、Ubuntu apt 源、Intel librealsense PPA
  - 国内/受限网络：用 Clash @ `127.0.0.1:7890`（详见附录 A）
- **D435i**：USB3 接好（可在 Phase 0 末尾再插）

---

## 1. 从 0 到 1 的复现步骤

```bash
# 0. clone 项目（如果新机器还没有）
git clone <repo-url> /data/codes/DexSlide
cd /data/codes/DexSlide

# 1. 主机环境快速检查
bash umi_mono/scripts/check_host_env.sh
# 期望：所有 [OK]，最多 python3 一行 [WARN]；退出码 0

# 2. apt 依赖（先 dry-run 看缺啥）
bash umi_mono/scripts/install_apt_deps.sh
# 看到 'N to install' 后，授权安装：
bash umi_mono/scripts/install_apt_deps.sh --apply   # 提示时输入 yes
# 期望末尾 '0 to install'

# 3. Pangolin v0.8 源码编译（~6 分钟）
bash umi_mono/scripts/build_pangolin.sh
# 期望产物：umi_mono/external/Pangolin/build/libpango_core.so 等 15 个 .so

# 4. Sophus 1.22.10 源码配置（header-only，~10 秒）
bash umi_mono/scripts/build_sophus.sh
# 期望产物：umi_mono/external/Sophus/build/SophusConfig.cmake

# 5. librealsense2 (D435i SDK) 安装
bash umi_mono/scripts/install_librealsense.sh           # dry-run, 看是否需要装
echo "yes" | bash umi_mono/scripts/install_librealsense.sh --apply   # 实装（仅当 dev 缺失）

# 5a. D435i 烟囱测试（D435i 必须插上）
bash umi_mono/scripts/run_d435i_smoke.sh
# 期望输出：color FPS in [27,33], accel FPS in [200,280], gyro FPS in [180,220], PASS
```

---

## 2. Phase 0 完成判定

| TASK | 产物 | 验收命令 | 期望 |
|------|------|---------|------|
| 001 | `scripts/check_host_env.sh` | `bash ... && echo $?` | exit 0, 表格全 OK/WARN |
| 002 | `scripts/install_apt_deps.sh` | `bash ...` | `0 to install` |
| 003 | `external/Pangolin/build/libpango_core.so` | `ls ...` | 15 个 .so |
| 004 | `external/Sophus/build/SophusConfig.cmake` | `ls ...` | exists |
| 005 | `pkg-config --modversion realsense2` | `pkg-config --modversion realsense2` | `>= 2.55` |
| 005a | D435i 烟囱 | `bash scripts/run_d435i_smoke.sh` | `PASS` |

---

## 3. 关键决策记录（避免新机器复现踩坑）

### 3.1 为什么不 `sudo make install` Pangolin / Sophus？
源码构建后的 `build/` 目录就能用，CMake 通过 `-DCMAKE_PREFIX_PATH` 指过去即可。`sudo make install` 会把 `.so` 和头文件写入 `/usr/local/`，会污染系统、新机器复现时多一步 sudo、不同项目用不同版本时冲突。所有外部依赖都保持在 `/data/codes/DexSlide/umi_mono/external/<pkg>/build/`。

### 3.2 为什么 `apt install` 默认 dry-run？
`install_apt_deps.sh --apply` 需要 sudo 才会真装。未授权前只查 `dpkg -s` 状态、列出待装清单。在企业 / CI 机器上 dry-run 可以列依赖却不动系统，便于审计。

### 3.3 关于 Python 版本
本机 Python 3.13.9 比 Ubuntu 22.04 默认 3.10 新。脚本对 ≥3.13 报 `[WARN]` 但不 fail —— **ROS2 Humble 安装时自带 `/opt/ros/humble/lib/python3.10`**，本项目的 ROS2 节点用的是 Humble 的 Python；系统 python 仅给非 ROS 脚本。

### 3.4 cppzmq 头文件
`cppzmq-dev` 在 Ubuntu 22.04 apt 不可得。`install_apt_deps.sh` 检测到不可用时打一行 note 并跳过，**Phase 3 (TASK-018) 会按需 clone**。

### 3.5 librealsense2 版本
本机装到 `2.57.7-0~realsense.18833`（PPA 最新）；tracker pin 的 `2.55.1` 是下限，2.57 兼容。**D435i 固件需要 ≥5.16**（rs-fw-update 检查）。

---

## 4. 失败时如何修复

### check_host_env.sh 报 MISSING
脚本末尾会打 `sudo apt install <pkg>` 提示，逐条装即可。

### Pangolin 编译失败
- 常见：`wayland-protocols` / `libxkbcommon-dev` 缺 → 跑 `install_apt_deps.sh --apply`
- 常见：`-DBUILD_PANGOLIN_PYTHON=OFF` 不够 → 检查脚本中 cmake flag 是否齐全
- 实在不行：`bash scripts/build_pangolin.sh --clean` 重来

### Sophus 报 `Eigen3 not found`
应已被 `libeigen3-dev` 解决；若未解决检查 `pkg-config --modversion eigen3`。

### librealsense2 安装失败
PPA 可能掉了。检查 `/etc/apt/sources.list.d/librealsense.list` 是否存在；若不存在按 [Intel 官方指南](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) 重新配置。

### D435i 烟囱测试失败 (`No device connected`)
- 确认 USB3 接口（USB2 不够带宽，会丢帧）
- `lsusb | grep Intel` 看到 `RealSense D435I`
- `rs-enumerate-devices` 看到 device
- 如果 `dmesg | tail` 报 USB 错误，换线/换 USB 口

---

## 附录 A — Codex / OpenAI API 通过 Clash 代理

实施期间通过 Codex CLI 协作。本机直连 OpenAI 不通，但 Clash 在 `127.0.0.1:7890` 听着。调 codex 前设：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
```

确认通：

```bash
HTTPS_PROXY=http://127.0.0.1:7890 curl -sS --max-time 5 \
  https://api.openai.com/v1/models -H "Authorization: Bearer dummy" | head -3
# 期望：invalid_request_error（API 可达，只是 key 错——网络层 OK）
```

同样的代理也用于 `git ls-remote` 和 `git clone github.com`（TASK-006 起需要）。

---

## 附录 B — 本阶段产物清单

| 文件 | 用途 |
|------|------|
| `umi_mono/scripts/check_host_env.sh` | TASK-001 主机环境检查 |
| `umi_mono/scripts/install_apt_deps.sh` | TASK-002 apt 依赖安装（dry-run-by-default） |
| `umi_mono/scripts/build_pangolin.sh` | TASK-003 Pangolin v0.8 源码编译 |
| `umi_mono/scripts/build_sophus.sh` | TASK-004 Sophus 1.22.10 配置 |
| `umi_mono/scripts/install_librealsense.sh` | TASK-005 librealsense2-dev 安装 |
| `umi_mono/scripts/test_d435i_streams.cpp` | TASK-005 D435i 烟囱测试源码 |
| `umi_mono/scripts/run_d435i_smoke.sh` | TASK-005 编译 + 跑烟囱 |
| `umi_mono/external/Pangolin/` | clone + 编译产物（gitignored） |
| `umi_mono/external/Sophus/` | clone + 编译产物（gitignored） |

## 附录 C — 下一阶段

**Phase 1 — Fork 集成与原生构建**（TASK-006 ~ TASK-009）

- TASK-006 ✅ Fork clone + SHA pin (`b741dca...`)
- TASK-007: Fork 原生构建脚本（Thirdparty/DBoW2、Thirdparty/g2o、根工程；用 CMAKE_PREFIX_PATH 指向 Pangolin/Sophus build）
- TASK-008: 原生 `gopro_slam` 与 Docker 输出 diff（验证 baseline 一致）
- TASK-009: 在 fork 的 CMakeLists.txt 加 `realsense_online` 空 target

详见 `setup_phase1_native_build.md`（TASK-007 完成后落地）。
