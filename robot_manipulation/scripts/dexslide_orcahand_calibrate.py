#!/usr/bin/env python3
"""逐指记录 DexSlide 角度上下限，生成 DexSlide→OrcaHand 映射。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROBOT_MANIP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROBOT_MANIP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dexslide.paths import DEFAULT_DEXSLIDE_STREAMING_FILE
from dexslide.streaming import DexSlideScene
from robot_manipulation.orca_control.direct_joint_calibration import (
    FINGERS,
    build_fingerwise_calibration,
    save_fingerwise_calibration,
)
from robot_manipulation.orca_control.direct_joint_mapping import TARGET_JOINTS, load_static_joint_map
from robot_manipulation.orca_control.paths import (
    DEFAULT_DEX_TO_ORCA_DIRECT_JOINT_MAP_FILE,
    DEFAULT_DIRECT_JOINT_CALIBRATION_FILE,
    DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE,
    ORCA_DEPENDENCIES_DIR,
)

if str(ORCA_DEPENDENCIES_DIR) not in sys.path:
    sys.path.insert(0, str(ORCA_DEPENDENCIES_DIR))

from orca_core.hand_config import OrcaHandConfig  # noqa: E402


DEFAULT_MANUAL_MOTOR_CALIBRATION = (
    ROBOT_MANIP_ROOT / "assets" / "orca_hand" / "configs" / "calibration.yaml"
)


def capture_endpoint(
    scene: DexSlideScene,
    *,
    hand_id: str,
    source_order: tuple[str, ...],
    sample_count: int,
    sample_rate_hz: float,
    max_std_deg: float,
) -> dict[str, float]:
    samples: list[np.ndarray] = []
    while len(samples) < sample_count:
        hand = scene.sample().hands[hand_id]
        if hand.joints_valid:
            values = np.asarray(hand.joint_angles_raw, dtype=np.float64).reshape(-1)
            if values.size != len(source_order) or not np.isfinite(values).all():
                raise RuntimeError("DexSlide joint vector is invalid")
            samples.append(values.copy())
        time.sleep(1.0 / sample_rate_hz)
    stacked = np.vstack(samples)
    deviation = float(np.max(np.std(stacked, axis=0)))
    if deviation > max_std_deg:
        raise RuntimeError(f"采样不稳定：最大标准差 {deviation:.2f} deg，超过 {max_std_deg:.2f} deg")
    median = np.median(stacked, axis=0)
    return {joint: float(median[index]) for index, joint in enumerate(source_order)}


def load_existing_finger_endpoints(
    path: str | Path,
    source_order: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.exists():
        return {}
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    raw_endpoints = payload.get("finger_endpoints_deg", {})
    if not isinstance(raw_endpoints, dict):
        return {}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for finger in FINGERS:
        item = raw_endpoints.get(finger)
        if not isinstance(item, dict):
            continue
        left, right = item.get("left"), item.get("right")
        finger_sources = [joint for joint in source_order if joint.startswith(f"{finger}.")]
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if not all(joint in left and joint in right for joint in finger_sources):
            continue
        result[finger] = {
            "left": {joint: float(left[joint]) for joint in finger_sources},
            "right": {joint: float(right[joint]) for joint in finger_sources},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    parser.add_argument("--hand-id", default="left")
    parser.add_argument("--output", default=str(DEFAULT_DIRECT_JOINT_CALIBRATION_FILE))
    parser.add_argument("--static-map", default=str(DEFAULT_DEX_TO_ORCA_DIRECT_JOINT_MAP_FILE))
    parser.add_argument("--orca-config", default=str(DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE))
    parser.add_argument("--motor-calibration", default=str(DEFAULT_MANUAL_MOTOR_CALIBRATION))
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--sample-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-std-deg", type=float, default=3.0)
    args = parser.parse_args()
    if args.sample_count < 1 or args.sample_rate_hz <= 0.0 or args.max_std_deg <= 0.0:
        raise SystemExit("采样参数必须为正数")

    static_map = load_static_joint_map(args.static_map)
    config = OrcaHandConfig.from_config_path(args.orca_config)
    motor_calibration_path = Path(args.motor_calibration).expanduser().resolve()
    if not motor_calibration_path.exists():
        raise SystemExit(f"找不到手动 Orca motor calibration：{motor_calibration_path}")
    motor_calibration = yaml.safe_load(motor_calibration_path.read_text(encoding="utf-8")) or {}

    scene = DexSlideScene.from_file(args.stream_config)
    try:
        scene.start()
        source_order = tuple(str(value) for value in scene.joint_order)
        required = set(static_map["dex_joint_ids"])
        missing = sorted(required - set(source_order))
        if missing:
            raise RuntimeError(f"DexSlide 缺少关节：{missing}")

        endpoints: dict[str, dict[str, dict[str, float]]] = {}
        existing_endpoints = load_existing_finger_endpoints(args.output, source_order)
        for finger in FINGERS:
            targets = [joint for joint in TARGET_JOINTS if joint.startswith(f"{finger}_")]
            left_rom = ", ".join(f"{joint}={config.joint_roms_dict[joint][0]:g}" for joint in targets)
            right_rom = ", ".join(f"{joint}={config.joint_roms_dict[joint][1]:g}" for joint in targets)
            while True:
                print(f"\n[{finger}] Orca 角度下限：{left_rom}")
                lower_answer = input("请仅摆好这一根手指，按 Enter 记录；输入 s 跳过当前手指： ")
                if lower_answer.strip().lower() == "s":
                    if finger not in existing_endpoints:
                        print("该手指没有已有的完整记录，不能跳过。")
                        continue
                    endpoints[finger] = existing_endpoints[finger]
                    print(f"[{finger}] 已跳过，沿用已有记录。")
                    break
                left = capture_endpoint(
                    scene, hand_id=args.hand_id, source_order=source_order,
                    sample_count=args.sample_count, sample_rate_hz=args.sample_rate_hz,
                    max_std_deg=args.max_std_deg,
                )
                print(f"[{finger}] Orca 角度上限：{right_rom}")
                upper_answer = input("请摆到同一根手指的角度上限，按 Enter 记录；输入 s 跳过当前手指： ")
                if upper_answer.strip().lower() == "s":
                    if finger not in existing_endpoints:
                        print("该手指没有已有的完整记录，不能跳过。")
                        continue
                    endpoints[finger] = existing_endpoints[finger]
                    print(f"[{finger}] 已跳过，沿用已有记录。")
                    break
                right = capture_endpoint(
                    scene, hand_id=args.hand_id, source_order=source_order,
                    sample_count=args.sample_count, sample_rate_hz=args.sample_rate_hz,
                    max_std_deg=args.max_std_deg,
                )
                endpoints[finger] = {"left": left, "right": right}
                print(f"[{finger}] 已完成。")
                break

        payload = build_fingerwise_calibration(
            source_joint_order=source_order,
            finger_endpoints_deg=endpoints,
            static_map_file=args.static_map,
            joint_roms_deg=config.joint_roms_dict,
            joint_to_motor_map=config.joint_to_motor_map,
            joint_inversion=config.joint_inversion_dict,
            motor_calibration=motor_calibration,
        )
        save_fingerwise_calibration(args.output, payload)
        print(f"\nDexSlide→Orca 映射已写入：{Path(args.output).expanduser().resolve()}")
        return 0
    finally:
        scene.close()


if __name__ == "__main__":
    raise SystemExit(main())
