#!/usr/bin/env python3
"""记录 OrcaHand 16 个手指电机的角度上下限。

脚本只关闭 torque 并读取电机当前位置，不调用自动校准或目标位置写入。
默认逐关节记录；也可选择逐手指同时记录一根手指的全部电机位置。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / "orca_control" / "orca_dependencies"
if str(DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(DEPENDENCIES))

from orca_core import OrcaHand  # noqa: E402


DEFAULT_CONFIG = (
    DEPENDENCIES / "orca_core" / "models" / "v1" / "orcahand_right" / "config.yaml"
)
DEFAULT_SOURCE_CALIBRATION = DEFAULT_CONFIG.with_name("calibration.yaml")
DEFAULT_OUTPUT = ROOT / "assets" / "orca_hand" / "configs" / "calibration.yaml"


def _capture_finger(
    hand: OrcaHand,
    *,
    finger: str,
    joints: list[str],
    bound: str,
) -> dict[int, float] | None:
    summary = ", ".join(
        f"{joint}={float(hand.config.joint_roms_dict[joint][0 if bound == '下限' else 1]):g} deg"
        for joint in joints
    )
    answer = input(
        f"请将 {finger} 的全部关节手动摆到角度{bound}（{summary}），"
        "按 Enter 记录；输入 s 跳过当前手指："
    )
    if answer.strip().lower() == "s":
        return None
    positions = hand.get_motor_pos(as_dict=True)
    result = {int(hand.config.joint_to_motor_map[joint]): float(positions[int(hand.config.joint_to_motor_map[joint])]) for joint in joints}
    print(
        "已记录："
        + ", ".join(
            f"{joint}/M{hand.config.joint_to_motor_map[joint]}={result[int(hand.config.joint_to_motor_map[joint])]:.6f} rad"
            for joint in joints
        )
    )
    return result


def _capture_joint(
    hand: OrcaHand,
    *,
    joint: str,
    motor_id: int,
    angle_deg: float,
    bound: str,
) -> float | None:
    answer = input(
        f"请将 {joint}（M{motor_id}）手动摆到角度{bound} {angle_deg:g} deg，"
        "按 Enter 记录；输入 s 跳过当前关节："
    )
    if answer.strip().lower() == "s":
        return None
    position = float(hand.get_motor_pos(as_dict=True)[motor_id])
    print(f"已记录：{joint} / M{motor_id} = {position:.6f} rad")
    return position


def _load_existing_endpoints(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    endpoints = payload.get("manual_endpoints", {})
    return endpoints if isinstance(endpoints, dict) else {}


def _reuse_existing(
    endpoints: dict[str, object],
    joints: list[str],
    joint_to_motor_map: dict[str, int],
) -> tuple[dict[int, float], dict[int, float]] | None:
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    for joint in joints:
        item = endpoints.get(joint)
        if not isinstance(item, dict):
            return None
        lower_value = item.get("motor_at_angle_lower_rad")
        upper_value = item.get("motor_at_angle_upper_rad")
        if lower_value is None or upper_value is None:
            return None
        motor_id = int(joint_to_motor_map[joint])
        lower[motor_id] = float(lower_value)
        upper[motor_id] = float(upper_value)
    return lower, upper


def _build_calibration(
    config,
    angle_lower: dict[int, float],
    angle_upper: dict[int, float],
    source: Path,
) -> dict:
    source_data = yaml.safe_load(source.read_text(encoding="utf-8")) if source.exists() else {}
    source_data = source_data or {}
    old_limits = source_data.get("motor_limits", {})
    old_ratios = source_data.get("joint_to_motor_ratios", {})

    motor_limits: dict[int, list[float | None]] = {}
    ratios: dict[int, float] = {}
    endpoints: dict[str, dict[str, float | int]] = {}

    for joint in config.joint_ids:
        motor_id = int(config.joint_to_motor_map[joint])
        if joint == "wrist":
            motor_limits[motor_id] = old_limits.get(motor_id, old_limits.get(str(motor_id), [None, None]))
            ratios[motor_id] = float(old_ratios.get(motor_id, old_ratios.get(str(motor_id), 0.0)))
            continue

        joint_lower, joint_upper = map(float, config.joint_roms_dict[joint])
        motor_at_lower = float(angle_lower[motor_id])
        motor_at_upper = float(angle_upper[motor_id])
        span = abs(motor_at_upper - motor_at_lower)
        if span <= 1e-6:
            raise ValueError(f"{joint} / motor {motor_id} 两个角度端点位置没有变化，不能生成映射")

        motor_limits[motor_id] = [min(motor_at_lower, motor_at_upper), max(motor_at_lower, motor_at_upper)]
        # OrcaHand 当前实现直接使用 config.yaml 中的 UI 数值（deg）。
        ratios[motor_id] = span / (joint_upper - joint_lower)
        endpoints[joint] = {
            "motor_id": motor_id,
            "angle_lower_deg": joint_lower,
            "angle_upper_deg": joint_upper,
            "motor_at_angle_lower_rad": motor_at_lower,
            "motor_at_angle_upper_rad": motor_at_upper,
        }

    complete = all(
        limits[0] is not None and limits[1] is not None and ratios[mid] != 0.0
        for mid, limits in motor_limits.items()
    )
    return {
        "calibrated": bool(complete),
        "wrist_calibrated": bool(config.joint_to_motor_map.get("wrist") in motor_limits),
        "motor_limits": motor_limits,
        "joint_to_motor_ratios": ratios,
        "manual_endpoints": endpoints,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-calibration", type=Path, default=DEFAULT_SOURCE_CALIBRATION)
    parser.add_argument(
        "--capture-mode",
        choices=("joint", "finger"),
        default="joint",
        help="记录粒度：joint（默认）逐关节；finger 逐手指",
    )
    args = parser.parse_args()

    hand = OrcaHand(config_path=str(args.config), calibration_path=str(args.output))
    success, message = hand.connect()
    if not success:
        print(message, file=sys.stderr)
        return 1

    try:
        hand.disable_torque()
        finger_joints = [joint for joint in hand.config.joint_ids if joint != "wrist"]
        if len(finger_joints) != 16:
            raise RuntimeError(f"配置中应有 16 个非 wrist 关节，实际为 {len(finger_joints)} 个")
        print("已连接 OrcaHand，16 个电机 torque 已关闭；现在可以手动掰动。")
        existing_endpoints = _load_existing_endpoints(args.output)
        angle_lower: dict[int, float] = {}
        angle_upper: dict[int, float] = {}
        if args.capture_mode == "joint":
            for joint in finger_joints:
                motor_id = int(hand.config.joint_to_motor_map[joint])
                lower, upper = map(float, hand.config.joint_roms_dict[joint])
                print(f"\n[{joint}] motor=M{motor_id}，角度范围 [{lower:g}, {upper:g}] deg")
                while True:
                    captured_lower = _capture_joint(
                        hand, joint=joint, motor_id=motor_id, angle_deg=lower, bound="下限"
                    )
                    captured_upper = (
                        None
                        if captured_lower is None
                        else _capture_joint(
                            hand, joint=joint, motor_id=motor_id, angle_deg=upper, bound="上限"
                        )
                    )
                    if captured_lower is not None and captured_upper is not None:
                        angle_lower[motor_id] = captured_lower
                        angle_upper[motor_id] = captured_upper
                        break
                    reused = _reuse_existing(existing_endpoints, [joint], hand.config.joint_to_motor_map)
                    if reused is None:
                        print("该关节没有已有的完整记录，不能跳过。")
                        continue
                    angle_lower.update(reused[0])
                    angle_upper.update(reused[1])
                    print(f"已跳过 {joint}，沿用已有记录。")
                    break
        else:
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                joints = [joint for joint in finger_joints if joint.startswith(f"{finger}_")]
                print(f"\n[{finger}] 包含关节：{', '.join(joints)}")
                while True:
                    captured_lower = _capture_finger(hand, finger=finger, joints=joints, bound="下限")
                    captured_upper = (
                        None
                        if captured_lower is None
                        else _capture_finger(hand, finger=finger, joints=joints, bound="上限")
                    )
                    if captured_lower is not None and captured_upper is not None:
                        angle_lower.update(captured_lower)
                        angle_upper.update(captured_upper)
                        break
                    reused = _reuse_existing(existing_endpoints, joints, hand.config.joint_to_motor_map)
                    if reused is None:
                        print("该手指没有已有的完整记录，不能跳过。")
                        continue
                    angle_lower.update(reused[0])
                    angle_upper.update(reused[1])
                    print(f"已跳过 {finger}，沿用已有记录。")
                    break
        result = _build_calibration(hand.config, angle_lower, angle_upper, args.source_calibration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"校准已写入：{args.output}")
        return 0
    finally:
        # 仅断开串口，避免 hand.disconnect() 额外发送 disable-torque 指令。
        client = getattr(hand, "_motor_client", None)
        if client is not None:
            client.disconnect()
            hand._motor_client = None


if __name__ == "__main__":
    raise SystemExit(main())
