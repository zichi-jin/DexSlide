# 2026-07-04 Direct ArUco / Hand Overlay / 手眼标定交接日志

## 今日目标

- 放弃 `umi_mono + ORB_SLAM3` 的世界位姿路线，改用桌面参考 ArUco `id=0` + 手背 marker body 的 direct ArUco 视觉链路。
- 在相机画面中直接观察：
  - 桌面坐标系；
  - marker body 融合位姿；
  - 绑定在 marker body 上的 DexSlide 骨骼手 AR 投影。
- 解决近期最明显的两个稳定性问题：
  - 单个 ArUco 的 `z` 轴朝内 / 朝外高频翻转；
  - marker body / 手骨移出画面后仍残留在屏幕上，且偶尔“冻住不跟”。

## 今日完成

### 1. 彻底转向 direct ArUco 路线

- 不再继续把主要精力放在 `umi_mono` 的单目 SLAM 世界位姿链路上。
- 当前主入口已经变成：
  - `scripts/view_direct_aruco_overlay.py`
- 相关核心模块：
  - `dexslide/vision/aruco_pose_tracker.py`
  - `dexslide/world_pose/direct_aruco_tracker.py`
  - `dexslide/world_pose/hand_cube_overlay.py`
  - `dexslide/world_pose/marker_body_pose_tracker.py`

### 2. 完成多面 marker body 的 YAML 几何描述与融合位姿解算

- 手背载体已经不再是早期的 `cube` 交互配置流程，而是离线 YAML：
  - `assets/calibration/direct_aruco/hand_marker_body_left.yaml`
- 运行时不再询问每个面的轴向映射，全部依赖 YAML。
- marker body 当前走的是：
  - 单帧检测全部可见 marker；
  - 由 YAML 给出的 marker 安装几何反推出 marker body 位姿；
  - 再把 DexSlide 的骨骼手刚性绑定到 marker body 上做 AR 投影。

### 3. 修正单个 ArUco 的双解翻转问题

- 根因确认：单个正方形 ArUco 的 `PnP` 存在平面双解，不是单纯角点噪声。
- 代码改动：
  - `dexslide/vision/aruco_pose_tracker.py` 现在会保留单 marker 的候选位姿分支。
  - `dexslide/world_pose/hand_cube_overlay.py` 增加了 marker body 级别的分支选择逻辑。
- 当前选择策略：
  - 优先使用多 marker 之间的几何一致性选出同一簇分支；
  - 当可见 marker 很少时，可参考上一帧 body 位姿做分支消歧。
- 效果：
  - `--show-target-axes` 下，原来经常朝内 / 朝外翻的那些蓝色 `z` 轴明显稳定了。

### 4. 删除“自由冻结视觉对齐”方案

- 运行时的自由冻结对齐方案已经判定效果差，且不再作为后续路线。
- 已删除功能：
  - `1 / 2 / 3` 对齐模式；
  - `f` 冻结；
  - `g` 应用并写回 YAML；
  - `x` 取消。
- 对应地，`view_direct_aruco_overlay.py` 和使用说明里都已移除这套逻辑。
- 结论：
  - 后续不再走“人眼手动挪对”的在线校准。
  - 以后改走定点、约束更强的手眼标定方案。

### 5. 查明并修复“过度冻结 / 出画残留”问题

- 现象：
  - 手移出画面后，marker body 坐标系和手骨不会立刻消失，而是停在画面里某处；
  - 手明明没按任何键，也会出现“偶尔冻结不跟”的感觉。
- 排查结论：
  - 不是 `f/g` 旧逻辑残留；
  - 不是 YAML 里写回的 offset；
  - 而是 `dexslide/world_pose/marker_body_pose_tracker.py` 里那套“拒收可疑新观测时继续沿用上一帧位姿”的 tracker 门控造成的。
- 已做处理：
  - 去掉了“无观测时保留旧位姿”；
  - 去掉了“jump gate 拒收后继续冻结上一帧”的主逻辑；
  - 现在只要 `raw_pose` 丢失，就直接让 marker body / 手骨消失。
- 这条修复是今天最后一轮的重要结论。

## 当前状态

- 当前 direct ArUco 视觉链路已经能稳定输出：
  - 桌面 `id=0` 坐标系；
  - 手背 marker body 的融合位姿；
  - 绑定到 marker body 上的 DexSlide 骨骼手 AR 投影。
- 当前已知最重要的已完成修复：
  - 单 marker 双解翻转；
  - 自由冻结对齐删除；
  - 过度冻结 / 出画残留修复。
- 当前不再建议继续投入：
  - `umi_mono` 的 SLAM 世界位姿路线；
  - 运行时自由冻结视觉对齐。

## 当前还需要用户现场再验证的点

- `marker_body_pose_tracker.py` 的最后一轮“出画即消失”修复已经通过单元测试，但仍建议明天真机再看一遍：
  - 手背 marker body 完全移出画面后，坐标系和手骨是否立刻消失；
  - 重新进画后是否正常恢复；
  - 是否还存在“偶发卡住不跟”的体感。

## 明天计划

明天打算切到更强约束的定点对齐方案，命名就叫“手眼标定”。

核心思路：

- 在桌面参考 ArUco `id=0` 的纸张上额外打印一个指定手形标记区域。
- 让操作者把手掌按到该手形区域。
- 把这个姿态定义为手的零位 / 标定姿态。
- 利用“手形标记相对于 `id=0` 的固定关系”，建立：
  - 桌面坐标系；
  - 手零位坐标系；
  - marker body / 骨骼手之间的固定外参。

这条路线的目的：

- 放弃自由拖拽式的人眼对齐；
- 改成一次性、约束明确、可重复的桌面手眼标定。

## 建议明天继续时优先看的文件

- 主脚本：
  - `scripts/view_direct_aruco_overlay.py`
- 单 marker 检测与候选 pose：
  - `dexslide/vision/aruco_pose_tracker.py`
- marker body 几何与分支选择：
  - `dexslide/world_pose/hand_cube_overlay.py`
- marker body 时间跟踪：
  - `dexslide/world_pose/marker_body_pose_tracker.py`
- 当前 marker body 几何配置：
  - `assets/calibration/direct_aruco/hand_marker_body_left.yaml`
- 使用说明：
  - `docs/direct_aruco_overlay_usage.md`

## 当前推荐启动命令

### 1. 只看视觉位姿

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --show-target-axes
```

### 2. 看 marker body + 骨骼手 AR 叠加

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --enable-hand-overlay \
  --source /dev/video4 \
  --hand left \
  --glove-port /dev/ttyACM0
```

### 3. 诊断某个 marker 是否仍与其他面不一致

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --hand left \
  --diagnose-marker-body \
  --show-target-axes
```

## 今天最后的测试结论

- 最近一次和本链路直接相关的测试已通过：
  - `36 passed`
- 覆盖范围包含：
  - direct ArUco tracker；
  - hand cube / marker body 几何；
  - marker body pose tracker；
  - overlay helper；
  - capture helper。

如果明天换新对话窗口，新的 AI 应该默认从这份日志继续，而不是重新回到 `umi_mono` 或在线冻结对齐路线。
