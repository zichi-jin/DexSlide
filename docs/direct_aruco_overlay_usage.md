# `view_direct_aruco_overlay.py` 使用说明

这个脚本的作用是：

- 用桌面参考 ArUco `id=0` 建立 `table/world` 坐标系。
- 在同一帧中检测手背 marker body 上的一个或多个 target ArUco。
- 对普通 target，直接计算 `target in table` 位姿。
- 对手背 `18` 面 marker body，把全部可见 marker 角点做一次联合 `PnP`，直接解出融合后的 marker body 位姿。
- 在相机画面上显示 target 边框和融合后的 marker body 质心坐标系。
- 可选地把 DexSlide 手套骨骼手重建结果绑定到 marker body 上，做 AR 叠加。

当前推荐用途有 3 个：

1. 检查 direct ArUco 世界位姿是否稳定。
2. 检查 marker body 质心坐标系是否真正粘在手背载体上。
3. 检查骨骼手 AR 投影是否和真实手套位置一致。

## 一、启动前提

需要准备：

- 一个固定在桌面上的参考 ArUco，默认 `id=0`，边长 `120 mm`。
- 一个贴在手背 marker body 上的 target ArUco 集合。默认按 `20 mm` 的 ArUco 黑边界解释各 marker 的角点几何。
- 一份离线编辑的 tags->marker JSON。默认左手路径是 `assets/calibration/direct_aruco/left_tags2marker.json`。
- 一台能读到 RGB 画面的相机，通常用 `/dev/video4`。
- 如果启用手骨骼 AR 叠加，还需要 DexSlide 手套串口，例如 `/dev/ttyACM0`。

默认依赖的标定文件：

```text
assets/calibration/direct_aruco/d435i_960_540.json
assets/calibration/direct_aruco/table_aruco.yaml
assets/calibration/direct_aruco/left_tags2marker.json
assets/calibration/direct_aruco/left_marker2wrist.json
assets/calibration/direct_aruco/left_marker2wrist_dataset.json
assets/calibration/glove_calibration.json
assets/skeletons/skeleton.json
```

## 二、最常用启动方式

### 1. 纯 direct ArUco overlay

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --target-marker-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
```

用途：

- 只看桌面参考 tag 和 target marker body 的视觉跟踪效果。
- 不叠加骨骼手。

当前画面行为：

- `table` 和 `target` 只显示边框。
- 不再把大量文字堆到画面上。
- `marker body` 只显示融合后的三轴坐标系。
- 运行状态滚动输出在终端，而不是画在画面里。

### 2. 开启骨骼手 AR 叠加

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --enable-hand-overlay \
  --source /dev/video4 \
  --hand left \
  --glove-port /dev/ttyACM0
```

用途：

- 把 DexSlide 的实时骨骼手投影到相机画面。
- 骨骼手相对于手背 marker body 保持刚性绑定。
- target id 会直接从 `left_tags2marker.json` 读取，不需要再手写一遍。

### 3. 观察单个 target marker 的三轴方向

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --target-marker-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 \
  --show-target-axes
```

这个模式用于离线编辑 tags->marker 几何：

1. 常规情况下，画面里每个 target 只显示四角框。
2. 打开 `--show-target-axes` 后，每个 target 会额外显示自己的局部 `xyz` 三轴和 id。
3. 观察结果后，手动修改 `left_tags2marker.json` 里的 `marker_face_id` 矩阵。

### 4. 检查某个 marker 是否“写得合法但方向写错”

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --hand left \
  --diagnose-marker-body \
  --show-target-axes
```

这个模式会：

- 从 `left_tags2marker.json` 读取全部 marker id 和几何关系。
- 让每个可见 marker 单独反推出一份 marker body 位姿。
- 再把这些单独解彼此比较，并和联合 `PnP` 的融合解对比。
- 在终端输出可疑 id，例如某个码的单独解总是和其他码显著不一致。

若终端出现类似：

```text
[diag] visible=[1,2,14] suspicious=[14] id14:peer=3.2mm/87.5deg fused=2.8mm/81.4deg reproj=0.42px
```

通常可这样理解：

- `peer=...mm/...deg`：这个 marker 单独反推出来的 body 位姿，和其他 marker 的平均两两不一致度。
- `fused=...mm/...deg`：这个 marker 单独解和联合 `PnP` 融合解之间的偏差。
- `reproj=...px`：在当前 YAML 假设下，它相对于融合解的重投影误差。

经验上：

- 如果 `peer` / `fused` 很大，但 `reproj` 不大，优先怀疑 YAML 方向写错。
- 如果 `peer` / `fused` 很大，同时 `reproj` 也大，优先怀疑当前帧角点检测错了、遮挡了、或 motion blur 太重。

## 三、运行时交互

- `q` / `Esc`：退出。

终端会持续滚动输出：

- 当前检测到的 marker id 集合。
- 当前用于融合 marker body 的 marker id 集合。
- marker body `spread`，单位 `mm`。
- 联合 PnP 的平均重投影误差 `reproj`，单位 `px`。
- 当前 marker body 求解器模式，例如 `joint_pnp`。
- 手套流延迟。

## 四、参数总表

下面是当前脚本的全部入口参数。

### A. 输入源与相机

#### `--source`

- 默认值：`0`
- 含义：OpenCV 采集源，可以是数字索引，也可以是 `/dev/video4` 这种设备路径。
- 备注：如果传入的源打不开，脚本会自动扫描 `/dev/video*` 兜底。

#### `--width`

- 默认值：`1280`
- 含义：请求的采集宽度。

#### `--height`

- 默认值：`720`
- 含义：请求的采集高度。

#### `--fps`

- 默认值：`None`
- 含义：请求的采集帧率。

#### `--buffer-size`

- 默认值：`2`
- 含义：OpenCV `VideoCapture` 缓冲区大小。
- 建议：如果你更关心实时性而不是稳定性，可以尝试 `1`。

#### `--num-workers`

- 默认值：`2`
- 含义：OpenCV 内部线程数。

### B. ArUco 与标定文件

#### `--camera-intrinsics`

- 默认值：`assets/calibration/direct_aruco/d435i_960_540.json`
- 含义：相机内参 JSON。

#### `--table-aruco-yaml`

- 默认值：`assets/calibration/direct_aruco/table_aruco.yaml`
- 含义：桌面参考 tag 的 ArUco 配置。

#### `--target-aruco-yaml`

- 默认值：`assets/calibration/direct_aruco/table_aruco.yaml`
- 含义：普通 target 的 ArUco 配置。默认与桌面配置共用同一个文件，并使用其中的 `default` marker size。
- 备注：启用 `--enable-hand-overlay` 后，这个参数会被 `left_tags2marker.json` 里的 marker 几何信息覆盖。

#### `--table-marker-id`

- 默认值：`0`
- 含义：桌面参考 tag 的 id。

#### `--target-marker-ids`

- 默认值：空字符串
- 含义：需要跟踪的 target id 列表，例如 `1,2,3,...,18`。
- 备注：为空时，脚本会跟踪所有被检测到的非 table marker；如果启用了 hand overlay，则优先跟踪 YAML 里配置过的 marker id。

### C. 检测器行为

#### `--no-refine-subpix`

- 默认值：关闭。
- 含义：关闭后，角点亚像素优化将被禁用。

#### `--corner-refine-mode`

- 可选值：`apriltag` / `subpix` / `contour` / `none`
- 默认值：空，也就是默认使用 `apriltag`；如果同时传了 `--no-refine-subpix`，则默认退化到 `none`。
- 含义：指定角点 refinement 策略。
- 建议：当前优先用 `apriltag`，因为它对手背 marker body 附近的黑色干扰结构更稳。

#### `--strict-detector`

- 默认值：关闭，也就是默认使用 motion-tolerant 检测配置。
- 含义：打开后，使用更严格的检测器参数。
- 建议：如果手持运动较快，一般不要开。

#### `--body-reprojection-threshold-px` / `--cube-reprojection-threshold-px`

- 默认值：`5.0`
- 含义：联合 PnP 后，如果某个 marker 的平均重投影误差超过这个阈值，就把它踢掉再重解一次。
- 作用：抑制某个 marker 四角抓错时，把整只虚拟手一起带偏。

### D. 画面显示相关

#### `--table-axis-length`

- 默认值：`0.08`
- 含义：桌面 tag 坐标轴长度，单位 `m`。
- 备注：当前主 overlay 画面默认不再显示 table 坐标轴，这个参数主要对早期调试或局部流程保留。

#### `--target-axis-length`

- 默认值：`0.04`
- 含义：target tag 坐标轴长度，单位 `m`。
- 备注：只有在显式传入 `--show-target-axes` 时才会显示。

#### `--body-axis-length` / `--cube-axis-length`

- 默认值：`0.05`
- 含义：融合后的 marker body 坐标轴长度，单位 `m`。

#### `--hand-axis-length`

- 默认值：`0.04`
- 含义：手腕坐标轴长度，单位 `m`。
- 备注：当前主 overlay 画面默认不再显示手腕坐标轴，因此这个参数目前基本处于保留状态。

### E. 手骨骼 AR 叠加

#### `--enable-hand-overlay`

- 默认值：关闭。
- 含义：打开后，叠加 DexSlide 的实时骨骼手。

#### `--glove-port`

- 默认值：`/dev/ttyACM0`
- 含义：DexSlide 手套串口。

#### `--glove-baud`

- 默认值：`115200`
- 含义：DexSlide 手套串口波特率。

#### `--glove-mode`

- 可选值：`raw` / `angles`
- 默认值：`raw`
- 含义：串口输入流模式。

#### `--glove-calib-file`

- 默认值：`assets/calibration/glove_calibration.json`
- 含义：DexSlide 手套标定文件。

#### `--skeleton-file`

- 默认值：`assets/skeletons/skeleton.json`
- 含义：骨骼手长度模型。

#### `--hand`

- 可选值：`auto` / `left` / `right`
- 默认值：`left`
- 含义：手型选择。

#### `--hand-overlay-config`

- 默认值：空
- 含义：显式指定 marker body 几何体 YAML 路径。
- 备注：为空时，会根据 `--hand` 自动选默认路径。

#### `--show-target-axes`

- 默认值：关闭。
- 含义：为每个已检测 target 画出自己的局部三轴和 id。
- 用途：离线检查 marker 贴装方向，随后手动修改 YAML。

### F. Marker Body 融合稳定性

#### `--body-outlier-threshold-mm` / `--cube-outlier-threshold-mm`

- 默认值：`20.0`
- 含义：当同帧可见 marker 数大于等于 3 时，若某个候选 marker body 种子中心偏离中位位置超过这个阈值，就将其踢掉。
- 建议：如果你看到特定角度会跳，可以尝试改成 `15` 甚至 `10`。

#### `--body-smoothing` / `--cube-smoothing`

- 默认值：`0.55`
- 含义：marker body 位姿时间平滑系数。
- 解释：越接近 `1.0` 越跟手，越接近 `0.0` 越平滑。
- 备注：`1.0` 等价于不做平滑。

## 五、推荐命令模板

### 1. 左手，常规稳定观察

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --enable-hand-overlay \
  --source /dev/video4 \
  --hand left \
  --glove-port /dev/ttyACM0 \
  --corner-refine-mode apriltag \
  --body-smoothing 0.35 \
  --body-outlier-threshold-mm 15 \
  --body-reprojection-threshold-px 4.0
```

### 2. 只看视觉位姿，不接手套

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --target-marker-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
```

### 3. 观察 target 三轴并离线改 YAML

```bash
/home/jzq/anaconda3/envs/dexslide/bin/python scripts/view_direct_aruco_overlay.py \
  --source /dev/video4 \
  --target-marker-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 \
  --show-target-axes
```

## 六、已知现象与排查建议

### 1. 启动时提示缺少 marker body 配置文件

原因通常是：`assets/calibration/direct_aruco/left_tags2marker.json` 不存在，或者你在 `--hand right` 下没有提供对应的 tags->marker JSON。

### 2. marker body 坐标系在特定角度抖动

优先检查：

- 某些 target marker 是否快出画。
- marker body 边缘是否遮挡太多。
- 相机是否离太近导致透视畸变和 motion blur 变重。

优先调：

- `--body-outlier-threshold-mm 15`
- `--body-smoothing 0.35`
- `--corner-refine-mode apriltag`
- `--body-reprojection-threshold-px 4.0`

### 3. 手骨骼和真实手套整体有固定偏差

这通常不是检测坏了，而是 `marker body -> wrist` 外参本身还需要重新离线标定。

### 4. 相机打不开

优先把 `--source` 改成明确设备路径，例如：

```bash
--source /dev/video4
```

脚本也会自动尝试扫描 `/dev/video*`。

## 七、相关代码入口

- 主脚本：`scripts/view_direct_aruco_overlay.py`
- direct ArUco 跟踪：`dexslide/vision/direct_aruco_tracker.py`
- marker 检测：`dexslide/vision/aruco_pose_tracker.py`
- marker body 融合与手部挂载：`dexslide/vision/hand_cube_overlay.py`
- 单独 3D 世界轨迹查看：`scripts/plot_aruco_relative_pose_3d.py`
