"""Thin OrcaHand hardware adapter with a stable diagnostic snapshot API."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .paths import ORCA_DEPENDENCIES_DIR

if str(ORCA_DEPENDENCIES_DIR) not in sys.path:
    sys.path.insert(0, str(ORCA_DEPENDENCIES_DIR))

from orca_core import OrcaHand  # noqa: E402


DEFAULT_MANUAL_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "orca_hand"
    / "configs"
    / "calibration.yaml"
)


def _install_fake_dynamixel_sdk() -> None:
    """Provide the tiny SDK surface used by MockDynamixelClient."""
    if "dynamixel_sdk" in sys.modules:
        return
    module = types.ModuleType("dynamixel_sdk")

    class PortHandler:
        def __init__(self, port: str):
            self.port_name = port
            self.is_using = False

    class PacketHandler:
        def __init__(self, protocol_version: float):
            self.protocol_version = protocol_version

    class GroupBulkRead:
        def __init__(self, port_handler: PortHandler, packet_handler: PacketHandler):
            self.port_handler = port_handler
            self.packet_handler = packet_handler

        def addParam(self, motor_id: int, address: int, size: int) -> bool:
            return True

        def txRxPacket(self) -> int:
            return 0

        def isAvailable(self, motor_id: int, address: int, size: int) -> bool:
            return True

        def getData(self, motor_id: int, address: int, size: int) -> int:
            return 0

    module.PortHandler = PortHandler
    module.PacketHandler = PacketHandler
    module.GroupBulkRead = GroupBulkRead
    sys.modules["dynamixel_sdk"] = module


@dataclass(frozen=True)
class OrcaDiagnosticSnapshot:
    joint_positions_deg: dict[str, float]
    motor_positions_rad: dict[int, float]
    motor_currents_ma: dict[int, float]
    motor_temperatures_c: dict[int, float]
    moving_status: dict[int, int | None]


class OrcaHandAdapter:
    """Keep vendor-specific/private access in one small boundary module."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        calibration_path: str | Path | None = None,
        mock: bool = False,
    ) -> None:
        if mock:
            _install_fake_dynamixel_sdk()
            from orca_core.hardware_hand import MockOrcaHand

            self.hand = MockOrcaHand(config_path=str(config_path))
        else:
            selected_calibration = (
                Path(calibration_path).expanduser().resolve()
                if calibration_path is not None
                else DEFAULT_MANUAL_CALIBRATION_PATH
            )
            self.hand = OrcaHand(
                config_path=str(config_path),
                calibration_path=str(selected_calibration),
            )
        self.config_path = Path(config_path).expanduser().resolve()

    @property
    def joint_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.hand.config.joint_ids)

    @property
    def motor_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.hand.config.motor_ids)

    def connect(self) -> None:
        success, message = self.hand.connect()
        if not success:
            raise RuntimeError(message)

    def initialize(self, *, move_to_neutral: bool = True) -> None:
        self.hand.init_joints(force_calibrate=False, move_to_neutral=move_to_neutral)

    def enable_torque_safe(self) -> None:
        """Re-enable torque without replaying an old goal position."""
        self.hand.disable_torque()
        current = np.asarray(self.hand.get_motor_pos(), dtype=np.float64).copy()
        compute_offsets = getattr(self.hand, "_compute_wrap_offsets_dict", None)
        if compute_offsets is not None:
            compute_offsets()
        set_motor = getattr(self.hand, "_set_motor_pos", None)
        if set_motor is None:
            raise AttributeError("OrcaHand does not expose its motor position writer")
        set_motor(current)
        self.hand.enable_torque()

    def disable_torque(self) -> None:
        self.hand.disable_torque()

    def set_joint_positions(self, positions_deg: Mapping[str, float]) -> None:
        self.hand.set_joint_positions(dict(positions_deg))

    def preview_motor_positions(self, positions_deg: Mapping[str, float]) -> dict[int, float]:
        converter = getattr(self.hand, "_joint_to_motor_pos", None)
        if converter is None:
            raise AttributeError("OrcaHand does not expose joint-to-motor conversion")
        values = converter(dict(positions_deg))
        result: dict[int, float] = {}
        for index, value in enumerate(values):
            if value is not None:
                result[int(self.hand.config.motor_ids[index])] = float(value)
        return result

    def snapshot(self, *, include_wrist: bool = True) -> OrcaDiagnosticSnapshot:
        joint_positions = {
            str(joint): float(value)
            for joint, value in self.hand.get_joint_position().as_dict().items()
            if value is not None and (include_wrist or joint != "wrist")
        }
        motor_positions = {
            int(motor): float(value)
            for motor, value in self.hand.get_motor_pos(as_dict=True).items()
        }
        motor_currents = {
            int(motor): float(value)
            for motor, value in self.hand.get_motor_current(as_dict=True).items()
        }
        motor_temperatures = {
            int(motor): float(value)
            for motor, value in self.hand.get_motor_temp(as_dict=True).items()
        }
        moving_status: dict[int, int | None] = {motor: None for motor in motor_positions}
        client = getattr(self.hand, "_motor_client", None)
        reader = getattr(client, "read_status_is_done_moving", None)
        if reader is not None:
            values = np.asarray(reader()).reshape(-1)
            moving_status = {
                int(motor): int(values[index]) if index < values.size else None
                for index, motor in enumerate(motor_positions)
            }
        return OrcaDiagnosticSnapshot(
            joint_positions_deg=joint_positions,
            motor_positions_rad=motor_positions,
            motor_currents_ma=motor_currents,
            motor_temperatures_c=motor_temperatures,
            moving_status=moving_status,
        )

    def close(self) -> None:
        if self.hand is None:
            return
        try:
            self.hand.disconnect()
        finally:
            self.hand = None  # type: ignore[assignment]


__all__ = ["OrcaDiagnosticSnapshot", "OrcaHandAdapter"]
