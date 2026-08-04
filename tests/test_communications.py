from __future__ import annotations

import json
import time

import pytest

from dexslide.communications import (
    camera_communication,
    hand_joint_communication,
    load_communications,
    resolve_camera_source,
    resolve_joint_port,
    resolve_realsense_serial,
)
from dexslide.serial_angles import AngleStreamReader, make_joint_order


def _write_config(path, *, stable_port: str, stable_camera: str) -> None:
    payload = {
        "schema_version": 1,
        "left": {
            "joints": {
                "port": "/dev/ttyACM0",
                "stable_port": stable_port,
                "baud": 115200,
                "mode": "raw",
                "startup_timeout_sec": 3.0,
                "max_sample_age_sec": 0.5,
            },
            "tactile": None,
        },
        "right": {"joints": None, "tactile": None},
        "camera": {
            "primary": {
                "backend": "realsense",
                "serial": "332522073507",
                "opencv_source": "/dev/video4",
                "stable_opencv_source": stable_camera,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_communications_match_connected_dexslide_devices() -> None:
    joints = hand_joint_communication("left")
    camera = camera_communication("primary")

    assert joints["port"] == "/dev/ttyACM0"
    assert joints["baud"] == 115200
    assert joints["mode"] == "raw"
    assert camera["backend"] == "opencv"
    assert camera["opencv_source"] == "0"
    assert "width" not in camera
    assert "height" not in camera
    assert "fps" not in camera
    with pytest.raises(ValueError, match="not configured with the RealSense backend"):
        resolve_realsense_serial("primary")


def test_camera_stream_profile_comes_from_intrinsics_file() -> None:
    from dexslide.camera_profile import load_camera_stream_profile
    from dexslide.paths import DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE

    payload = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    profile = load_camera_stream_profile()

    assert (profile.width, profile.height, profile.fps) == (
        int(payload["image_width"]),
        int(payload["image_height"]),
        float(payload["fps"]),
    )


def test_resolvers_prefer_existing_stable_device_paths(tmp_path) -> None:
    stable_port = tmp_path / "serial-by-id"
    stable_camera = tmp_path / "video-by-path"
    stable_port.touch()
    stable_camera.touch()
    config_path = tmp_path / "communications.json"
    _write_config(
        config_path,
        stable_port=str(stable_port),
        stable_camera=str(stable_camera),
    )

    assert resolve_joint_port("left", path=config_path) == str(stable_port)
    assert resolve_camera_source("primary", path=config_path) == str(stable_camera)


def test_unconfigured_right_hand_fails_loudly(tmp_path) -> None:
    config_path = tmp_path / "communications.json"
    _write_config(
        config_path,
        stable_port=str(tmp_path / "missing-port"),
        stable_camera=str(tmp_path / "missing-camera"),
    )

    with pytest.raises(ValueError, match="No joints communication configured"):
        hand_joint_communication("right", path=config_path)


def test_invalid_schema_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "communications.json"
    config_path.write_text('{"schema_version": 99}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_communications(config_path)


def test_serial_reader_reports_worker_error_before_old_sample() -> None:
    joint_order = make_joint_order()
    reader = AngleStreamReader(
        port="/dev/null",
        baud=115200,
        mode="raw",
        joint_order=joint_order,
        calibration={str(item["id"]): {"offset": 0.0, "angle0": 0.0, "rate": 1.0} for item in joint_order},
    )
    reader.latest_time = time.time()
    reader.last_error = "device disconnected"

    with pytest.raises(RuntimeError, match="device disconnected"):
        reader.wait_for_first_sample(0.05)


def test_partial_raw_sensor_line_does_not_publish_a_sample() -> None:
    joint_order = make_joint_order()
    reader = AngleStreamReader(
        port="/dev/null",
        baud=115200,
        mode="raw",
        joint_order=joint_order,
        calibration={
            str(item["id"]): {"offset": 0.0, "angle0": 0.0, "rate": 1.0}
            for item in joint_order
        },
    )

    reader._update_raw_line("I2C1@0x48[A0:1,A1:2,A2:3,A3:4]")

    assert reader.latest_time == 0.0
    assert reader.latest_line == ""
