---
title: "Learnings"
readMode: optional
priority: medium
category: learning
keywords:
  - bug
  - lesson
  - gotcha
  - learning
---

# Learnings

Add entries with: `/spec-add learning <description>`

## Entries

<spec-entry id="INS-20260726-001" category="learning" keywords="orcahand,dynamixel,xc330,pretension,antagonistic-tendon,observability" date="2026-07-26" source=".workflow/knowhow/KNW-investigate-orcahand-current-pretension/report.md">
ORCA 非腕关节使用 flexor/extensor 双拮抗腱。单个电机的 current/torque 主要观测两根腱的张力差，而共同预紧取决于张力和；因此 motor current 可以做 differential torque/compliance control，却不能唯一测量或闭环设定 common-mode tendon pretension。真正自动预紧需要独立 top-spool 调节自由度、弹性张紧元件或额外张力传感。
</spec-entry>

<spec-entry id="INS-20260726-002" category="debug" keywords="xc330,current-scale,goal-current,present-current,orca-core" date="2026-07-26" source="orca_control/orca_dependencies/orca_core/hardware/dynamixel_client.py:59">
XC330-T288 官方 Goal Current 与 Present Current 单位约为 1 mA/count；orca_core 当前沿用 DEFAULT_CUR_SCALE=1.34 读取 Present Current，却直接把调用值写入 Goal Current，读写标度不一致。任何基于 current threshold 的新控制前都应改为 model-specific conversion，并分别暴露 goal current 与 current limit 语义。
</spec-entry>

<spec-entry id="INS-20260726-003" category="learning" keywords="orcahand,calibration,current,position-stability,slack,hysteresis" date="2026-07-26" source="orca_control/orca_dependencies/orca_core/hardware_hand.py:642">
OrcaHand 自动 calibration 的机械限位判据仅使用 motor position buffer 的稳定性；current_log 虽被采集但未参与判定。预紧不一致、tendon slack、backlash、torque-off rebound 和 routing friction 会直接改变限位记录，因此改进精度应优先量化 slack/hysteresis，而不是仅调 current telemetry。
</spec-entry>
