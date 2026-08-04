#!/usr/bin/env python3
"""Collect independent Orca pose and DexSlide pose pairs without teleoperation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

ROBOT_MANIPULATION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROBOT_MANIPULATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dexslide.paths import DEFAULT_DEXSLIDE_STREAMING_FILE
from dexslide.streaming import DexSlideScene
from robot_manipulation.orca_control.direct_joint_mapping import TARGET_JOINTS
from robot_manipulation.orca_control.hand_adapter import (
    DEFAULT_MANUAL_CALIBRATION_PATH,
    OrcaHandAdapter,
)
from robot_manipulation.orca_control.paths import (
    DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE,
    ORCA_CONFIG_DIR,
)

DEFAULT_PAIRS_FILE = ORCA_CONFIG_DIR / "orca_teleop_pairs.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_payload(source_order: tuple[str, ...], motor_calibration: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_unit": "deg",
        "target_unit": "deg",
        "source_joint_order": list(source_order),
        "target_joint_order": list(TARGET_JOINTS),
        "motor_calibration_file": str(motor_calibration),
        "motor_calibration_sha256": _file_sha256(motor_calibration)
        if motor_calibration.exists()
        else None,
        "pairs": [],
    }


def _load_payload(
    path: Path,
    source_order: tuple[str, ...],
    motor_calibration: Path,
) -> dict[str, object]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return _new_payload(source_order, motor_calibration)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported pose-pair data schema: {path}")
    if tuple(str(value) for value in payload.get("source_joint_order", ())) != source_order:
        raise ValueError("Existing pose-pair file uses a different DexSlide joint order")
    if tuple(str(value) for value in payload.get("target_joint_order", ())) != TARGET_JOINTS:
        raise ValueError("Existing pose-pair file uses a different Orca target joint order")
    if not isinstance(payload.get("pairs"), list):
        raise ValueError("Existing pose-pair file contains invalid pairs")
    return dict(payload)


def _save_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _capture_dexslide_pose(
    scene: DexSlideScene,
    *,
    hand_id: str,
    source_order: tuple[str, ...],
    sample_count: int,
    sample_rate_hz: float,
    max_std_deg: float,
) -> tuple[dict[str, float], dict[str, float]]:
    samples: list[np.ndarray] = []
    while len(samples) < sample_count:
        sample = scene.sample().hands[hand_id]
        if sample.joints_valid:
            vector = np.asarray(sample.joint_angles_raw, dtype=np.float64).reshape(-1)
            if vector.size != len(source_order) or not np.isfinite(vector).all():
                raise RuntimeError("DexSlide joint vector is invalid")
            samples.append(vector.copy())
        time.sleep(1.0 / sample_rate_hz)
    stacked = np.vstack(samples)
    std = np.std(stacked, axis=0)
    maximum = float(np.max(std))
    if maximum > max_std_deg:
        raise RuntimeError(
            f"DexSlide sample is unstable: max std {maximum:.2f} deg exceeds {max_std_deg:.2f} deg"
        )
    median = np.median(stacked, axis=0)
    return (
        {joint: float(median[index]) for index, joint in enumerate(source_order)},
        {joint: float(std[index]) for index, joint in enumerate(source_order)},
    )


def _orca_pose(adapter: OrcaHandAdapter) -> dict[str, object]:
    snapshot = adapter.snapshot(include_wrist=False)
    missing = [joint for joint in TARGET_JOINTS if joint not in snapshot.joint_positions_deg]
    if missing:
        raise RuntimeError(f"Orca did not report all 16 target joints: {missing}")
    return {
        "joint_positions_deg": {
            joint: float(snapshot.joint_positions_deg[joint])
            for joint in TARGET_JOINTS
        },
        "motor_positions_rad": snapshot.motor_positions_rad,
        "motor_currents_ma": snapshot.motor_currents_ma,
        "motor_temperatures_c": snapshot.motor_temperatures_c,
        "moving_status": snapshot.moving_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    parser.add_argument("--hand-id", default="left")
    parser.add_argument("--orca-config", default=str(DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE))
    parser.add_argument("--motor-calibration", default=str(DEFAULT_MANUAL_CALIBRATION_PATH))
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_FILE)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--sample-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-std-deg", type=float, default=3.0)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.sample_count < 1 or args.sample_rate_hz <= 0.0 or args.max_std_deg <= 0.0:
        raise SystemExit("Sampling values must be positive")

    scene = DexSlideScene.from_file(args.stream_config)
    calibration_path = Path(args.motor_calibration).expanduser().resolve()
    adapter = OrcaHandAdapter(
        args.orca_config,
        calibration_path=calibration_path,
        mock=args.mock,
    )
    try:
        scene.start()
        source_order = tuple(str(value) for value in scene.joint_order)
        if len(source_order) != 20:
            raise RuntimeError(f"DexSlide must provide 20 joints, received {len(source_order)}")
        pairs_path = args.pairs.expanduser().resolve()
        payload = _load_payload(pairs_path, source_order, calibration_path)
        adapter.connect()
        adapter.initialize(move_to_neutral=False)
        adapter.disable_torque()
        print("DexSlide is sampling. Orca torque is disabled and may be moved by hand.")
        print("For each pair: set Orca pose A, freeze it, set glove pose A prime, then record.")

        while True:
            pair_number = len(payload["pairs"]) + 1
            answer = input(
                f"\n[{pair_number}] Move Orca to pose A. Press Enter to freeze and record, or q to finish: "
            ).strip().lower()
            if answer == "q":
                break
            adapter.enable_torque_safe()
            orca = _orca_pose(adapter)
            print(f"[{pair_number}] Orca pose is frozen and recorded.")
            answer = input(
                f"[{pair_number}] Put on the glove and make pose A prime. Press Enter to record, or q to finish: "
            ).strip().lower()
            if answer == "q":
                adapter.disable_torque()
                break
            try:
                dexslide, std = _capture_dexslide_pose(
                    scene,
                    hand_id=args.hand_id,
                    source_order=source_order,
                    sample_count=args.sample_count,
                    sample_rate_hz=args.sample_rate_hz,
                    max_std_deg=args.max_std_deg,
                )
            except RuntimeError as exc:
                adapter.disable_torque()
                print(f"[{pair_number}] {exc}. This pair was not saved.")
                continue
            payload["pairs"].append(
                {
                    "id": pair_number,
                    "label": f"pair_{pair_number}",
                    "created_at_unix": time.time(),
                    "dexslide": {
                        "joint_angles_deg": dexslide,
                        "sample_count": args.sample_count,
                        "std_deg": std,
                    },
                    "orca": orca,
                }
            )
            _save_payload(pairs_path, payload)
            adapter.disable_torque()
            print(f"[{pair_number}] Saved to {pairs_path}. Orca torque is disabled for the next pose.")
    finally:
        try:
            adapter.disable_torque()
        except Exception:
            pass
        adapter.close()
        scene.close()

    pair_count = len(payload["pairs"])
    print(f"Collected {pair_count} pose pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
