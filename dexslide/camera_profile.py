"""Camera stream profile loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexslide.paths import DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE


@dataclass(frozen=True)
class CameraStreamProfile:
    width: int
    height: int
    fps: float


def _positive_number(payload: dict[str, Any], key: str, *, path: Path) -> float:
    try:
        value = float(payload[key])
    except KeyError as exc:
        raise ValueError(f"Camera intrinsics `{path}` is missing `{key}`.") from exc
    if value <= 0.0:
        raise ValueError(f"Camera intrinsics `{path}` field `{key}` must be positive.")
    return value


def load_camera_stream_profile(
    path: str | Path = DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
) -> CameraStreamProfile:
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Camera intrinsics `{profile_path}` must contain a JSON object.")
    return CameraStreamProfile(
        width=int(round(_positive_number(payload, "image_width", path=profile_path))),
        height=int(round(_positive_number(payload, "image_height", path=profile_path))),
        fps=float(_positive_number(payload, "fps", path=profile_path)),
    )
