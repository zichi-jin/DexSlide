"""Realtime overlay of reconstructed human hand and retargeted Orca hand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dexslide.kinematics.live_hand import FINGERS, palm_edges
from dexslide.live import live_listener, shutdown_live_listeners
from dexslide.paths import DEFAULT_ORCAHAND_RIGHT_URDF_FILE
from dexslide.retargeting import HUMAN_LANDMARK_NAMES, create_dex_retargeter
from dexslide.retargeting.engine import _load_retarget_document
from dexslide.retargeting.live_bridge import (
    FAULTY_GLOVE_JOINT_OVERRIDES_RAD,
    apply_glove_joint_overrides,
)

COLORS = {
    "thumb": "#e76f51",
    "index": "#3a86ff",
    "middle": "#06d6a0",
    "ring": "#ffbe0b",
    "pinky": "#8338ec",
}

ROBOT_LINK_NAMES_BY_HUMAN_LANDMARK = [
    "right_palm",
    "right_thumb_mp",
    "right_thumb_pp",
    "right_thumb_ip",
    "right_thumb_fingertip",
    "right_index_mp",
    "right_index_pp",
    "right_index_ip",
    "right_index_fingertip",
    "right_middle_mp",
    "right_middle_pp",
    "right_middle_ip",
    "right_middle_fingertip",
    "right_ring_mp",
    "right_ring_pp",
    "right_ring_ip",
    "right_ring_fingertip",
    "right_pinky_mp",
    "right_pinky_pp",
    "right_pinky_ip",
    "right_pinky_fingertip",
]

FINGER_LANDMARK_SLICES = {
    "thumb": slice(1, 5),
    "index": slice(5, 9),
    "middle": slice(9, 13),
    "ring": slice(13, 17),
    "pinky": slice(17, 21),
}

PALM_LANDMARK_INDEX = {
    "wrist": 0,
    "thumb": 1,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}


def _require_robot_wrapper():
    try:
        from dex_retargeting.robot_wrapper import RobotWrapper
    except Exception as ex:
        raise RuntimeError(
            "Realtime retarget compare needs local OrcaHand forward kinematics in the current "
            "DexSlide interpreter. Install `pip install -r requirements-retargeting.txt` first. "
            "`--dex-retargeting-python` only serves as a temporary fallback for the optimizer subprocess."
        ) from ex
    return RobotWrapper


def _unique_flattened_indices(indices: np.ndarray) -> list[int]:
    flat = np.asarray(indices, dtype=int).reshape(-1)
    ordered: list[int] = []
    seen: set[int] = set()
    for value in flat.tolist():
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _best_fit_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    mode: Literal["rigid", "similarity"],
) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 clouds, got {src.shape} and {dst.shape}")

    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    src_zero = src - src_center[None, :]
    dst_zero = dst - dst_center[None, :]
    covariance = src_zero.T @ dst_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    scale = 1.0
    if mode == "similarity":
        denom = float(np.sum(src_zero * src_zero))
        if denom > 1e-12:
            scale = float(np.sum(singular_values) / denom)

    translation = dst_center - scale * (rotation @ src_center)
    return scale, rotation, translation


def _apply_transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation[None, :]


def _link_positions(robot: Any, qpos: np.ndarray, link_ids: np.ndarray) -> np.ndarray:
    robot.compute_forward_kinematics(np.asarray(qpos, dtype=np.float64))
    return np.asarray([robot.get_link_pose(int(link_id))[:3, 3] for link_id in link_ids], dtype=np.float64)


def _set_polyline(line, points: np.ndarray) -> None:
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def run_live_retarget_compare_viewer(
    *,
    port: str,
    baud: int,
    mode: str,
    calib_file: str | Path,
    skeleton_file: str | Path,
    retarget_config: str | Path,
    dex_retargeting_python: str | None = None,
    fps: float = 30.0,
    hand: str = "auto",
    mirror_reconstruction: bool = False,
    align_mode: Literal["rigid", "similarity"] = "similarity",
) -> None:
    listener = live_listener(port=port, baud=baud, mode=mode, calib_file=calib_file)
    retargeter = create_dex_retargeter(
        config_file=retarget_config,
        skeleton_file=skeleton_file,
        hand=hand,
        mirror_reconstruction=mirror_reconstruction,
        dex_retargeting_python=dex_retargeting_python,
    )

    document = _load_retarget_document(retarget_config)
    align_indices = _unique_flattened_indices(np.asarray(document["retargeting"]["target_link_human_indices"], dtype=int))

    RobotWrapper = _require_robot_wrapper()
    robot = RobotWrapper(str(DEFAULT_ORCAHAND_RIGHT_URDF_FILE))
    robot_link_names = list(ROBOT_LINK_NAMES_BY_HUMAN_LANDMARK)
    if len(robot_link_names) != len(HUMAN_LANDMARK_NAMES):
        raise RuntimeError(
            "Robot landmark list does not match human landmark layout: "
            f"robot={len(robot_link_names)}, human={len(HUMAN_LANDMARK_NAMES)}"
        )
    robot_link_ids = np.asarray([robot.get_link_index(name) for name in robot_link_names], dtype=int)
    backend_joint_names = list(retargeter.backend_joint_names)
    robot_joint_names = list(robot.dof_joint_names)
    if set(backend_joint_names) != set(robot_joint_names):
        raise RuntimeError(
            "Retarget backend joint names do not match Orca URDF DOF names. "
            f"backend={backend_joint_names}, urdf={robot_joint_names}"
        )
    robot_qpos_reorder = np.asarray([backend_joint_names.index(name) for name in robot_joint_names], dtype=int)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig = plt.figure("DexSlide Retarget Compare", figsize=(11, 8), dpi=plt.rcParams.get("figure.dpi", 100))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("DexSlide Human vs Retargeted Orca Overlay")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.4, 1.0, 0.9))
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-35, 185)
    ax.set_ylim(-95, 95)
    ax.set_zlim(-125, 55)

    human_lines = {}
    robot_lines = {}
    for finger in FINGERS:
        (human_line,) = ax.plot([], [], [], "-o", lw=2.8, ms=4.8, color=COLORS[finger], alpha=0.28)
        (robot_line,) = ax.plot([], [], [], "-o", lw=3.6, ms=5.8, color=COLORS[finger], alpha=0.96, label=finger)
        human_lines[finger] = human_line
        robot_lines[finger] = robot_line

    human_edge_lines = []
    robot_edge_lines = []
    for _ in palm_edges():
        (human_edge,) = ax.plot([], [], [], "-", lw=2.0, color="#888888", alpha=0.20)
        (robot_edge,) = ax.plot([], [], [], "-", lw=2.4, color="#444444", alpha=0.85)
        human_edge_lines.append(human_edge)
        robot_edge_lines.append(robot_edge)
    ax.legend(loc="upper right")
    txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=11)

    print(f"Opening compare viewer on {port} @ {baud}, mode={mode}, align={align_mode}")
    print("Human hand is semi-transparent; Orca hand is opaque.")
    print(f"Glove hand={hand}, mirror_reconstruction={mirror_reconstruction}")
    if FAULTY_GLOVE_JOINT_OVERRIDES_RAD:
        override_text = ", ".join(
            f"{name}={float(np.rad2deg(value)):6.2f} deg"
            for name, value in FAULTY_GLOVE_JOINT_OVERRIDES_RAD.items()
        )
        print(f"Temporary glove joint overrides enabled: {override_text}")

    def update(_frame_idx: int):
        human_joint_angles, timestamp, raw_line = listener.snapshot_rad20()
        if timestamp <= 0.0:
            return (
                tuple(human_lines.values())
                + tuple(robot_lines.values())
                + tuple(human_edge_lines)
                + tuple(robot_edge_lines)
                + (txt,)
            )

        human_joint_map = apply_glove_joint_overrides(
            human_joint_angles,
            retargeter.human_model.joint_names,
        )
        human_landmarks_mm = retargeter.human_landmarks(human_joint_map) * 1000.0
        robot_full_qpos = retargeter.retarget_full_qpos(human_joint_map)
        robot_qpos_for_fk = robot_full_qpos[robot_qpos_reorder]
        robot_landmarks_mm = _link_positions(robot, robot_qpos_for_fk, robot_link_ids) * 1000.0

        scale, rotation, translation = _best_fit_transform(
            robot_landmarks_mm[align_indices],
            human_landmarks_mm[align_indices],
            mode=align_mode,
        )
        robot_landmarks_aligned = _apply_transform(robot_landmarks_mm, scale, rotation, translation)

        for finger in FINGERS:
            finger_slice = FINGER_LANDMARK_SLICES[finger]
            human_points = np.vstack([human_landmarks_mm[0], human_landmarks_mm[finger_slice]])
            robot_points = np.vstack([robot_landmarks_aligned[0], robot_landmarks_aligned[finger_slice]])
            _set_polyline(human_lines[finger], human_points)
            _set_polyline(robot_lines[finger], robot_points)

        for human_line, robot_line, (a_name, b_name) in zip(human_edge_lines, robot_edge_lines, palm_edges()):
            a_idx = PALM_LANDMARK_INDEX[a_name]
            b_idx = PALM_LANDMARK_INDEX[b_name]
            human_pair = human_landmarks_mm[[a_idx, b_idx], :]
            robot_pair = robot_landmarks_aligned[[a_idx, b_idx], :]
            _set_polyline(human_line, human_pair)
            _set_polyline(robot_line, robot_pair)

        age_ms = (time.time() - timestamp) * 1000.0
        sent_deg = np.rad2deg(robot_full_qpos[retargeter.orca_indices])
        txt.set_text(
            f"stream_age={age_ms:6.1f} ms  align={align_mode}  display_scale={scale:5.3f}\n"
            f"thumb_tip_err={np.linalg.norm(robot_landmarks_aligned[4] - human_landmarks_mm[4]):6.1f} mm  "
            f"index_tip_err={np.linalg.norm(robot_landmarks_aligned[8] - human_landmarks_mm[8]):6.1f} mm\n"
            f"orca_deg: {retargeter.joint_ids[0]}={sent_deg[0]:6.1f}, {retargeter.joint_ids[1]}={sent_deg[1]:6.1f}, "
            f"{retargeter.joint_ids[2]}={sent_deg[2]:6.1f}, {retargeter.joint_ids[3]}={sent_deg[3]:6.1f}\n"
            f"{raw_line[:100]}"
        )
        return (
            tuple(human_lines.values())
            + tuple(robot_lines.values())
            + tuple(human_edge_lines)
            + tuple(robot_edge_lines)
            + (txt,)
        )

    anim = FuncAnimation(
        fig,
        update,
        interval=1000.0 / max(1e-6, fps),
        blit=False,
        repeat=True,
    )
    fig._anim = anim  # type: ignore[attr-defined]

    try:
        plt.tight_layout()
        plt.show()
    finally:
        retargeter.close()
        shutdown_live_listeners()
