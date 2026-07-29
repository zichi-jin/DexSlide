"""Single source of truth for DexSlide device communication settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dexslide.paths import DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE


def _required_text(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing {context}.{key} in DexSlide communications config")
    return value


def _positive_int(payload: dict[str, Any], key: str, *, context: str) -> int:
    value = int(payload.get(key, 0))
    if value <= 0:
        raise ValueError(f"{context}.{key} must be positive in DexSlide communications config")
    return value


def load_communications(path: str | Path = DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"DexSlide communications config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"DexSlide communications config must contain a JSON object: {config_path}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported DexSlide communications schema_version in {config_path}")
    return payload


def hand_joint_communication(
    hand: str = "left",
    *,
    path: str | Path = DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
) -> dict[str, Any]:
    requested = str(hand).strip().lower()
    if requested == "auto":
        requested = "left"
    payload = load_communications(path)
    hand_payload = payload.get(requested)
    if not isinstance(hand_payload, dict):
        raise ValueError(f"No communications entry for hand={requested!r}")
    joints = hand_payload.get("joints")
    if not isinstance(joints, dict):
        raise ValueError(f"No joints communication configured for hand={requested!r}")
    result = dict(joints)
    context = f"{requested}.joints"
    _required_text(result, "port", context=context)
    result["baud"] = _positive_int(result, "baud", context=context)
    result["mode"] = _required_text(result, "mode", context=context)
    if result["mode"] not in {"raw", "angles"}:
        raise ValueError(f"{context}.mode must be 'raw' or 'angles'")
    result["startup_timeout_sec"] = float(result.get("startup_timeout_sec", 3.0))
    result["max_sample_age_sec"] = float(result.get("max_sample_age_sec", 0.5))
    if result["startup_timeout_sec"] <= 0.0:
        raise ValueError(f"{context}.startup_timeout_sec must be positive")
    if result["max_sample_age_sec"] <= 0.0:
        raise ValueError(f"{context}.max_sample_age_sec must be positive")
    return result


def camera_communication(
    name: str = "primary",
    *,
    path: str | Path = DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
) -> dict[str, Any]:
    payload = load_communications(path)
    cameras = payload.get("camera")
    if not isinstance(cameras, dict):
        raise ValueError("DexSlide communications config has no camera section")
    camera = cameras.get(str(name))
    if not isinstance(camera, dict):
        raise ValueError(f"No camera communication configured for name={name!r}")
    result = dict(camera)
    context = f"camera.{name}"
    result["backend"] = _required_text(result, "backend", context=context)
    return result


def resolve_joint_port(
    hand: str = "left",
    *,
    path: str | Path = DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
) -> str:
    joints = hand_joint_communication(hand, path=path)
    stable_port = str(joints.get("stable_port", "")).strip()
    fallback_port = str(joints.get("port", "")).strip()
    if stable_port and Path(stable_port).exists():
        return stable_port
    if fallback_port:
        return fallback_port
    raise ValueError(f"No serial port configured for hand={hand!r}")


def resolve_camera_source(
    name: str = "primary",
    *,
    path: str | Path = DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
) -> str:
    camera = camera_communication(name, path=path)
    stable_source = str(camera.get("stable_opencv_source", "")).strip()
    fallback_source = str(camera.get("opencv_source", "")).strip()
    if stable_source and Path(stable_source).exists():
        return stable_source
    if fallback_source:
        return fallback_source
    raise ValueError(f"No OpenCV source configured for camera={name!r}")


def resolve_realsense_serial(
    name: str = "primary",
    *,
    path: str | Path = DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
) -> str:
    camera = camera_communication(name, path=path)
    if str(camera.get("backend", "")).strip().lower() != "realsense":
        raise ValueError(f"Camera {name!r} is not configured with the RealSense backend")
    return _required_text(camera, "serial", context=f"camera.{name}")
