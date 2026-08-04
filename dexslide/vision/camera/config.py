"""Resolution of camera-device configuration for DexSlide scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dexslide.communications import camera_communication, resolve_camera_source


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _required_positive_number(payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(
            f"Missing camera.{key} in streaming config; capture mode must not "
            "come from the intrinsics file"
        )
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera.{key} must be numeric") from exc
    if value <= 0.0:
        raise ValueError(f"camera.{key} must be positive")
    return value


def resolve_camera_config(
    payload: Mapping[str, Any],
    *,
    base_dir: Path,
    communications_file: Path,
) -> dict[str, Any]:
    """Resolve device source and mode without exposing camera details upstream."""

    camera_name = str(payload.get("name", "primary")).strip()
    raw_intrinsics = str(payload.get("intrinsics_file", "")).strip()
    if not raw_intrinsics:
        raise ValueError("Missing camera.intrinsics_file")
    intrinsics_file = _resolve_path(raw_intrinsics, base_dir=base_dir)
    if not intrinsics_file.is_file():
        raise FileNotFoundError(f"Configured camera intrinsics file does not exist: {intrinsics_file}")

    communication = camera_communication(camera_name, path=communications_file)
    return {
        **dict(payload),
        "name": camera_name,
        "intrinsics_file": intrinsics_file,
        "source": payload.get(
            "source",
            resolve_camera_source(camera_name, path=communications_file),
        ),
        "width": int(_required_positive_number(payload, "width")),
        "height": int(_required_positive_number(payload, "height")),
        "fps": _required_positive_number(payload, "fps"),
        "fourcc": payload.get("fourcc"),
        "backend": communication["backend"],
    }
