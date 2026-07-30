"""Configuration and calibration resolution for the direct ArUco overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE, DEFAULT_LEFT_MARKER_TO_WRIST_FILE,
    DEFAULT_SKELETON_FILE, DEXALIGN_CALIBRATION_DIR, DIRECT_ARUCO_CALIBRATION_DIR,
)
from dexslide.vision.direct_aruco_tracker import _normalize_target_marker_ids
from dexslide.vision.marker_body_model import HandCubeOverlayConfig, try_load_hand_cube_overlay_config

def _latest_dexalign_result_path(filename: str) -> Path | None:
    base_dir = Path(DEXALIGN_CALIBRATION_DIR).expanduser().resolve()
    if not base_dir.exists():
        return None

    candidates: list[Path] = []
    for session_dir in base_dir.iterdir():
        if not session_dir.is_dir():
            continue
        candidate = session_dir / str(filename)
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _default_overlay_skeleton_file() -> Path:
    optimized = _latest_dexalign_result_path("optimized_skeleton.json")
    return optimized if optimized is not None else DEFAULT_SKELETON_FILE


def _default_overlay_marker2hand_file() -> Path:
    optimized = _latest_dexalign_result_path("optimized_marker2hand.json")
    return optimized if optimized is not None else DEFAULT_LEFT_MARKER_TO_WRIST_FILE


def _default_overlay_joint_calibration_file() -> str:
    optimized = _latest_dexalign_result_path("optimized_joint_calibration.json")
    return "" if optimized is None else str(optimized)


def _load_overlay_joint_calibration(path: str | Path | None) -> tuple[np.ndarray, np.ndarray, Path | None]:
    if path is None:
        return np.ones(20, dtype=np.float64), np.zeros(20, dtype=np.float64), None
    text = str(path).strip()
    if not text:
        return np.ones(20, dtype=np.float64), np.zeros(20, dtype=np.float64), None

    config_path = Path(text).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid joint calibration JSON `{config_path}`: expected object root.")

    joint_scale = np.asarray(payload.get("joint_scale"), dtype=np.float64).reshape(-1)
    joint_bias = np.asarray(payload.get("joint_bias_rad"), dtype=np.float64).reshape(-1)
    if joint_scale.shape[0] != 20 or joint_bias.shape[0] != 20:
        raise ValueError(
            f"Invalid joint calibration JSON `{config_path}`: expected 20 scales and 20 biases, got "
            f"{joint_scale.shape[0]} and {joint_bias.shape[0]}."
        )
    return joint_scale, joint_bias, config_path


def _apply_overlay_joint_calibration(
    joint_angles: np.ndarray,
    *,
    joint_scale: np.ndarray,
    joint_bias: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(joint_angles, dtype=np.float64).reshape(20)
    scale = np.asarray(joint_scale, dtype=np.float64).reshape(20)
    bias = np.asarray(joint_bias, dtype=np.float64).reshape(20)
    return (scale * raw) + bias


def _marker_id_compact_text(marker_ids: list[int]) -> str:
    return "-" if not marker_ids else "".join(str(int(marker_id)) for marker_id in marker_ids)


def _default_hand_overlay_config_path(hand: str) -> Path:
    requested = str(hand).strip().lower()
    if requested in {"left", "right"}:
        return DIRECT_ARUCO_CALIBRATION_DIR / f"{requested}_tags2marker.json"

    fallback_candidates = [
        DIRECT_ARUCO_CALIBRATION_DIR / "left_tags2marker.json",
        DIRECT_ARUCO_CALIBRATION_DIR / "right_tags2marker.json",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    return fallback_candidates[0]


def _parse_target_marker_ids(raw_value: str) -> list[int] | None:
    text = str(raw_value).strip()
    if not text:
        return None
    marker_ids: list[int] = []
    for chunk in text.replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        marker_ids.append(int(item))
    return marker_ids or None


def _ensure_hand_cube_config(
    *,
    config_path: Path,
    hand: str,
) -> HandCubeOverlayConfig:
    cfg = try_load_hand_cube_overlay_config(config_path)
    if cfg is None:
        raise SystemExit(
            "未找到可用的 marker body 配置文件："
            f"{config_path}\n"
            "请先准备 tags->marker JSON，并写入 18 个 marker 面的几何关系；"
            "当前版本已移除运行时逐面交互配置。"
        )
    requested = str(hand).strip().lower()
    if requested in {"left", "right"}:
        cfg.hand = requested
    elif str(cfg.hand).strip().lower() not in {"left", "right"}:
        cfg.hand = "left"
    return cfg


def _resolve_target_marker_ids_for_overlay(
    parsed_target_ids: list[int] | None,
    hand_overlay_enabled: bool,
    hand_overlay_config: HandCubeOverlayConfig | None,
    table_marker_id: int,
) -> list[int] | None:
    if hand_overlay_enabled:
        configured = [] if hand_overlay_config is None else hand_overlay_config.marker_ids()
        selected = configured if parsed_target_ids is None else parsed_target_ids
        return _normalize_target_marker_ids(selected, table_marker_id)
    return _normalize_target_marker_ids(parsed_target_ids, table_marker_id)


