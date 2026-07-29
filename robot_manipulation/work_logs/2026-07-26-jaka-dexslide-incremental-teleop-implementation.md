# 2026-07-26 DexSlide -> JAKA 增量遥操实现记录

## 本次落地内容

当前已经在 `robot_manipulation` 下落了一条可运行的第一阶段遥操管线，入口脚本为：

```text
scripts/jaka_dexslide_incremental_teleop.py
```

辅助启动脚本：

```text
scripts/run_jaka_dexslide_incremental_teleop.sh
```

默认静态映射配置：

```text
assets/teleop_robot_mappings/workspace_axis_mapping.json
```

注意这里的 `assets` 指的是上级 DexSlide 资产目录，也就是：

```text
/home/jzq/MyJob/DexSlide/assets/teleop_robot_mappings/workspace_axis_mapping.json
```

## 当前脚本实际做了什么

### 1. 输入链

脚本按以下固定链路获取手套 wrist pose：

```text
table ArUco -> marker body pose -> DexAlign marker2hand -> glove wrist pose
```

具体约束如下：

- 通信配置来自 `assets/dexslide_communications.json`。
- table 世界系来自 `assets/calibration/direct_aruco/table_aruco.yaml`。
- marker body 几何来自 `assets/calibration/direct_aruco/left_tags2marker.json`。
- `marker2hand` 强制来自 `assets/calibration/dexalign/test_left_001/optimized_marker2hand.json`。
- 不会继续使用 `left_tags2marker.json` 里默认指向的旧 `left_marker2wrist.json`。

### 2. 增量遥操映射

脚本不是绝对 shadowing，而是显式 anchor 模式：

1. tracking 恢复后先等 `anchor_stable_frames` 帧。
2. 记录当下 glove anchor pose。
3. 同时记录 JAKA 当前 TCP pose 作为 robot anchor。
4. 每一帧都先根据 glove 相对 anchor 的位姿，计算一个 `desired robot pose`。
5. 再从 `previous_desired_robot_pose` 到 `current_desired_robot_pose` 提取一小步 `servo_p(INCR)` 增量。

这样满足了：

- 机器人不会追 glove 的绝对安装姿态。
- tracking 丢失后会 pause 并重新建 anchor。
- 下发给 JAKA 的是连续小步，不是每一拍都重复发从 `t0` 到当前的全量累计位移。

### 3. 世界系镜像和固定旋转

配置中已固化：

- 平移镜像：`[dx, dy, dz] -> [dx, -dy, dz]`
- 旋转镜像：`[rx, ry, rz] -> [-rx, ry, -rz]`
- 同姿态 glove/tool 到 Orca/tool 的固定旋转：`Ry(+90) @ Rz(-45)`

当前实现里：

- 固定旋转只保存在配置中用于说明和后续扩展。
- 在增量旋转里不会重复叠加这项固定旋转。

### 4. JAKA 侧安全约束

在真正开 compliant 之前，脚本会：

1. 上电。
2. 使能。
3. 检查当前 TCP 是否已在安全下垂初始姿态附近。
4. 如果不在，则先执行线性移动到：

```text
move -100 400 300 -180 0 -45
```

5. 确认到位后，才继续：
   - `set_torque_sensor_mode(1)`
   - 可选写回 saved payload
   - `zero_end_sensor()`
   - `set_admit_ctrl_config(...)`
   - `servo_move_enable(True)`
   - `set_compliant_type(1, 0)`
   - `set_compliant_type(0, 1)`

### 5. OrcaHand 预留口

当前阶段不发手指命令，但已经留了结构化占位：

- `DexAlign` 的 `optimized_skeleton.json`
- `optimized_joint_calibration.json`
- glove joint 串口路径

都在启动时显式加载或打印。

脚本里有一个：

```python
OrcaHandTeleopPlaceholder
```

后续要把真实的手指映射接进去，优先从这里扩展，而不是重新写一套并行入口。

## 当前默认配置

当前 `workspace_axis_mapping.json` 中默认值为：

- `^B T_T = I`
- 平移倍率：`[1, 1, 1]`
- 旋转倍率：`[1, 1, 1]`
- 平移 deadband：`0.8 mm`
- 旋转 deadband：`0.6 deg`
- 单步最大平移：`3.0 mm`
- 单步最大旋转：`2.0 deg`
- tracking 跳变拒绝阈值：`25 mm / 18 deg`
- 建 anchor 需要稳定帧数：`5`

这些都只是第一轮保守值，真机联调时可能还要继续改。

## 运行方式

### 仅看位姿链和映射，不碰机器人

```bash
scripts/run_jaka_dexslide_incremental_teleop.sh --dry-run
```

### 真机运行

```bash
scripts/run_jaka_dexslide_incremental_teleop.sh
```

如果你不想把本地 payload 缓存写回控制器：

```bash
scripts/run_jaka_dexslide_incremental_teleop.sh --no-saved-payload
```

如果要允许转动轴柔顺：

```bash
scripts/run_jaka_dexslide_incremental_teleop.sh --enable-rotation-compliance
```

## 已做的离线验证

已经验证：

- `python3 -m py_compile scripts/jaka_dexslide_incremental_teleop.py tests/test_jaka_dexslide_incremental_teleop.py`
- `conda run -n dexslide python -c "import os, pytest; os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; raise SystemExit(pytest.main(['tests/test_jaka_dexslide_incremental_teleop.py','-q']))"`
- `conda run -n dexslide python scripts/jaka_dexslide_incremental_teleop.py --help`

当前单测覆盖了：

- 固定旋转在相对旋转里抵消。
- 平移 / 小角度旋量镜像。
- anchor 下的 desired robot pose 构造。
- `previous_desired -> current_desired` 的增量提取。
- deadband 和单步限幅。

## 仍待真机验证的项

以下事项还没有在当前会话里做真机验证：

- JAKA 实际 TCP 的 `Rpy` 约定是否与脚本里采用的 `R = Rz(rz) @ Ry(ry) @ Rx(rx)` 完全一致。
- compliant 模式里连续 `servo_p(INCR)` 和当前 `desired pose` 生成方式的手感是否稳定。
- table marker 丢失、重捕获、marker body 部分遮挡时的 pause / re-anchor 行为是否足够稳。
- 当机械臂姿态接近水平举起时，当前 deadband / 限幅参数是否还会诱发震颤。

## 目前结论

这条管线在代码结构上已经满足第一阶段目标：

- 只做末端位姿遥操。
- 明确使用 table 世界系。
- 明确绑定 DexAlign session。
- 明确使用增量遥操。
- 在 JAKA compliant 模式前强制执行安全初始姿态和传感器置零。
- 给 OrcaHand 后续关节遥操留了清晰的接入点。

但它还不是“已经真机证明完全可靠”的最终版。下一步主要是拿这版去做实机联调，然后根据实际 jitter、丢 marker、机械臂姿态敏感区间，再继续收参数和保护逻辑。
