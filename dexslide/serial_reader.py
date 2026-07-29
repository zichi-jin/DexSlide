"""
Serial reader for DexSlide data frames.

Reads 20 joint angles from STM32 over USB CDC (Virtual COM Port).

Frame format (44 bytes):
  [0xAA55] [20x uint16_t angles] [XOR checksum] [0x0D]
"""

import struct
import serial

from dexslide.communications import hand_joint_communication, resolve_joint_port

FRAME_HEADER = 0xAA55
FRAME_SIZE = 44  # 2 + 40 + 1 + 1
TOTAL_JOINTS = 20

# Joint name mapping
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
JOINT_NAMES = ["DIP", "PIP", "MCP_front", "MCP_back"]


def joint_index(finger: int, joint: int) -> int:
    """Get flat index from finger (0-4) and joint (0-3)."""
    return finger * 4 + joint


def joint_label(index: int) -> str:
    """Human-readable label for joint index (0-19)."""
    finger = index // 4
    joint = index % 4
    return f"{FINGER_NAMES[finger]}_{JOINT_NAMES[joint]}"


class SerialReader:
    """Reads and parses DexSlide joint data frames from USB serial."""

    def __init__(
        self,
        port: str | None = None,
        baudrate: int | None = None,
        *,
        hand: str = "left",
    ):
        communication = hand_joint_communication(hand)
        self.port = resolve_joint_port(hand) if port is None else str(port)
        self.baudrate = int(communication["baud"] if baudrate is None else baudrate)
        self.ser = None

    def open(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def read_frame(self) -> list[int] | None:
        """
        Read one data frame. Returns list of 20 uint16 angle values,
        or None if frame is invalid / timed out.
        """
        if not self.ser or not self.ser.is_open:
            return None

        # Sync to header bytes 0x55 0xAA (little-endian 0xAA55)
        while True:
            b = self.ser.read(1)
            if len(b) == 0:
                return None  # timeout
            if b[0] == 0x55:
                b2 = self.ser.read(1)
                if len(b2) == 0:
                    return None
                if b2[0] == 0xAA:
                    break

        # Read remaining: 40 bytes angles + 1 checksum + 1 tail
        payload = self.ser.read(FRAME_SIZE - 2)
        if len(payload) < FRAME_SIZE - 2:
            return None

        # Parse 20 uint16 values (little-endian)
        angles = list(struct.unpack("<20H", payload[:40]))

        # Verify checksum
        expected_cs = payload[40]
        computed_cs = 0
        for byte in payload[:40]:
            computed_cs ^= byte
        if computed_cs != expected_cs:
            return None  # checksum mismatch

        # Verify tail
        if payload[41] != 0x0D:
            return None

        return angles
