"""OrcaHand direct-joint teleoperation endpoint."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from dexslide.streaming import DexSlideSceneSample

from .pair_curve_mapping import load_orca_teleop_mapper
from .hand_adapter import OrcaHandAdapter


@dataclass(frozen=True)
class OrcaTeleopConfig:
    source_hand: str = "left"
    calibration_file: str = ""
    orca_config_file: str = ""
    loop_hz: float | None = None
    max_sample_age_sec: float = 0.5
    dry_run: bool = False
    mock: bool = False
    warn_interval_s: float = 1.0


class OrcaTeleopEndpoint:
    def __init__(self, config: OrcaTeleopConfig) -> None:
        self.config = config
        self.mapper = load_orca_teleop_mapper(config.calibration_file)
        self.hand = (
            None
            if config.dry_run
            else OrcaHandAdapter(config.orca_config_file, mock=config.mock)
        )
        self.last_status = "starting"
        self.last_clip_reasons: dict[str, tuple[str, ...]] = {}
        self.last_warning_time = 0.0
        self.last_joint_age_sec = float("inf")
        self.sent_frames = 0

    @property
    def name(self) -> str:
        return "orcahand"

    def start(self) -> None:
        if self.hand is None:
            print("[orcahand] dry-run mode")
            return
        self.hand.connect()
        self.hand.initialize(move_to_neutral=False)
        print("[orcahand] direct joint teleop ready")

    def consume(self, sample: DexSlideSceneSample) -> None:
        hand_sample = sample.hands[self.config.source_hand]
        self.last_joint_age_sec = float(hand_sample.joint_age_sec)
        if (
            not hand_sample.joints_valid
            or hand_sample.joint_age_sec > float(self.config.max_sample_age_sec)
        ):
            self.last_status = "joint_stale"
            return
        result = self.mapper.map(
            hand_sample.joint_angles_raw,
            source_joint_order=self.mapper.calibration.source_joint_order,
        )
        self.last_clip_reasons = result.clip_reasons
        if result.clip_reasons and time.monotonic() - self.last_warning_time >= self.config.warn_interval_s:
            print(f"[orcahand] clipped joints: {result.clip_reasons}")
            self.last_warning_time = time.monotonic()
        if self.hand is None:
            self.last_status = "dry_run"
            return
        self.hand.set_joint_positions(result.target_positions_deg)
        self.sent_frames += 1
        self.last_status = "clipped" if result.clipped else "ok"

    def status_lines(self) -> list[str]:
        return [
            f"status={self.last_status}",
            f"joint_age={self.last_joint_age_sec:.3f}s",
            f"sent={self.sent_frames}",
        ]

    def close(self) -> None:
        if self.hand is not None:
            self.hand.close()


__all__ = ["OrcaTeleopConfig", "OrcaTeleopEndpoint"]
