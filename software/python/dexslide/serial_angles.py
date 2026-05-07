"""Serial parsing and ADC-count-to-angle mapping for DexSlide."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_OFFSET = {"thumb": 0, "index": 4, "middle": 8, "ring": 12, "pinky": 16}
JOINT_LABELS = ("DIP", "PIP", "MCP_front", "MCP_back")
SENSOR_ORDER = (
    ("I2C1", 0x48, "thumb"),
    ("I2C1", 0x49, "index"),
    ("I2C2", 0x48, "middle"),
    ("I2C2", 0x49, "ring"),
    ("I2C2", 0x4B, "pinky"),
)

COUNTS_PER_TURN = 24370.0
DEFAULT_RATE = COUNTS_PER_TURN / 360.0
HALF_TURN = COUNTS_PER_TURN / 2.0

RAW_FRAME_RE = re.compile(
    r"(I2C[12])@0x([0-9A-Fa-f]{2})\[A0:(-?\d+),A1:(-?\d+),A2:(-?\d+),A3:(-?\d+)\]"
)
ANGLE_LINE_RE = re.compile(r"([a-z]+\.(?:DIP|PIP|MCP_front|MCP_back)):(-?\d+(?:\.\d+)?)")


def pick_default_port() -> str:
    try:
        from serial.tools import list_ports
    except ImportError:
        return ""

    ports = [p.device for p in list_ports.comports()]
    if not ports:
        return ""
    for candidate in ports:
        if "ttyACM" in candidate or "ttyUSB" in candidate:
            return candidate
    return ports[0]


def make_joint_order() -> list[dict[str, object]]:
    order = []
    for bus, addr, finger in SENSOR_ORDER:
        sensor_key = f"{bus}@0x{addr:02X}"
        for channel_idx, joint_name in enumerate(JOINT_LABELS):
            order.append(
                {
                    "id": f"{finger}.{joint_name}",
                    "finger": finger,
                    "joint": joint_name,
                    "sensor": sensor_key,
                    "channel": channel_idx,
                }
            )
    return order


def parse_raw_sensor_line(line: str) -> dict[str, list[int]]:
    sensor_map = {}
    for match in RAW_FRAME_RE.finditer(line):
        sensor_key = f"{match.group(1)}@0x{int(match.group(2), 16):02X}"
        sensor_map[sensor_key] = [
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
        ]
    return sensor_map


def parse_angle_line(line: str) -> dict[str, float]:
    return {match.group(1): float(match.group(2)) for match in ANGLE_LINE_RE.finditer(line)}


def normalize_delta(delta: float) -> float:
    if delta > HALF_TURN:
        return delta - COUNTS_PER_TURN
    if delta < -HALF_TURN:
        return delta + COUNTS_PER_TURN
    return delta


def build_default_calibration(joint_order: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        str(joint["id"]): {"offset": 0.0, "angle0": 0.0, "rate": DEFAULT_RATE}
        for joint in joint_order
    }


def load_calibration(
    path: Path | None,
    joint_order: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    calibration = build_default_calibration(joint_order)
    if path is None or not path.exists():
        return calibration

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    for key, value in loaded.items():
        if not isinstance(value, dict):
            continue
        rate = float(value.get("rate", 0.0))
        if abs(rate) < 1e-9:
            k = float(value.get("k", 0.0))
            rate = 1.0 / k if abs(k) > 1e-9 else DEFAULT_RATE
        calibration[key] = {
            "offset": float(value.get("offset", value.get("d0", 0.0))),
            "angle0": float(value.get("angle0", 0.0)),
            "rate": rate,
        }
    return calibration


class AngleStreamReader:
    """Background serial reader that exposes the latest 20 joint angles in radians."""

    def __init__(
        self,
        port: str,
        baud: int,
        mode: str,
        joint_order: list[dict[str, object]],
        calibration: dict[str, dict[str, float]],
    ):
        self.port = port
        self.baud = baud
        self.mode = mode
        self.joint_order = joint_order
        self.calibration = calibration
        self.latest_deg = {str(j["id"]): 0.0 for j in joint_order}
        self.alignment_offsets = {str(j["id"]): 0.0 for j in joint_order}
        self.latest_time = 0.0
        self.latest_line = ""
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def snapshot_rad20(self) -> tuple[np.ndarray, float, str]:
        with self.lock:
            vec = np.zeros(20, dtype=np.float64)
            for joint in self.joint_order:
                finger = str(joint["finger"])
                channel = int(joint["channel"])
                joint_id = str(joint["id"])
                idx = FINGER_OFFSET[finger] + channel
                deg = float(self.latest_deg.get(joint_id, 0.0))
                offset = float(self.alignment_offsets.get(joint_id, 0.0))
                vec[idx] = np.deg2rad(deg - offset)
            return vec, self.latest_time, self.latest_line

    def _worker(self) -> None:
        import serial

        with serial.Serial(self.port, self.baud, timeout=0.5) as ser:
            time.sleep(0.2)
            while self.running:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                if self.mode == "angles":
                    self._update_angle_line(line)
                else:
                    self._update_raw_line(line)

    def _update_angle_line(self, line: str) -> None:
        values = parse_angle_line(line)
        if not values:
            return
        with self.lock:
            for key, value in values.items():
                if key in self.latest_deg:
                    self.latest_deg[key] = value
            self.latest_line = line
            self.latest_time = time.time()

    def _update_raw_line(self, line: str) -> None:
        sensor_map = parse_raw_sensor_line(line)
        if not sensor_map:
            return
        with self.lock:
            for joint in self.joint_order:
                sensor_values = sensor_map.get(str(joint["sensor"]))
                if sensor_values is None:
                    continue
                raw_count = float(sensor_values[int(joint["channel"])])
                conf = self.calibration[str(joint["id"])]
                angle_deg = conf["angle0"] + (
                    normalize_delta(raw_count - conf["offset"]) / conf["rate"]
                )
                self.latest_deg[str(joint["id"])] = angle_deg
            self.latest_line = line
            self.latest_time = time.time()
