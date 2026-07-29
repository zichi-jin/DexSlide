"""Matplotlib live 3D visualizer for DexSlide serial angle streams."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dexslide.kinematics.live_hand import (
    FINGER_OFFSET,
    FINGERS,
    apply_handedness,
    canonicalize_palm_xoy,
    finger_points,
    palm_edges,
    thumb_chain_rx_rad,
    thumb_pp_frame,
)
from dexslide.serial_angles import AngleStreamReader, load_calibration, make_joint_order

COLORS = {
    "thumb": "#e76f51",
    "index": "#3a86ff",
    "middle": "#06d6a0",
    "ring": "#ffbe0b",
    "pinky": "#8338ec",
}


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = (rotvec / theta).astype(np.float64)
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def _apply_rigid(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (points @ rot.T) + trans[None, :]


def _resolve_group_for_hand(
    groups: dict,
    aruco_hand_group: str | None,
    aruco_group_hand_map: dict[str, str] | None,
    aruco_hand_slot: str,
) -> str | None:
    if not groups:
        return None

    if aruco_hand_group is not None:
        return aruco_hand_group if aruco_hand_group in groups else None

    if aruco_group_hand_map:
        candidates = [g for g, h in aruco_group_hand_map.items() if h == aruco_hand_slot and g in groups]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return sorted(candidates)[0]

    if "all" in groups:
        return "all"
    if len(groups) == 1:
        return list(groups.keys())[0]
    return None


def _set_line3d(line, origin: np.ndarray, direction: np.ndarray, scale: float) -> None:
    line.set_data(
        [origin[0], origin[0] + direction[0] * scale],
        [origin[1], origin[1] + direction[1] * scale],
    )
    line.set_3d_properties([origin[2], origin[2] + direction[2] * scale])


def run_live_viewer(
    port: str,
    baud: int,
    mode: str,
    skeleton_file: str | Path,
    calib_file: str | Path,
    hand: str = "left",
    fps: float = 30.0,
    aruco_pose_tracker=None,
    aruco_hand_group: str | None = None,
    aruco_group_hand_map: dict[str, str] | None = None,
    aruco_hand_slot: str = "hand1",
    aruco_pose_hold_sec: float = 0.5,
) -> None:
    with Path(skeleton_file).open("r", encoding="utf-8") as handle:
        skeleton = json.load(handle)

    palm = apply_handedness(canonicalize_palm_xoy(skeleton), hand)
    joint_order = make_joint_order()
    calibration = load_calibration(Path(calib_file), joint_order)

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

    print(f"Opening {port} @ {baud}, mode={mode}")
    reader = AngleStreamReader(port, baud, mode, joint_order, calibration)
    reader.start()

    fig = plt.figure("DexSlide Live 3D", figsize=(10, 8), dpi=plt.rcParams.get("figure.dpi", 100))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("DexSlide Live Hand Reconstruction")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.4, 1.0, 0.9))
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-35, 185)
    ax.set_ylim(-95, 95)
    ax.set_zlim(-125, 55)

    finger_lines = {}
    for name in FINGERS:
        (line,) = ax.plot([], [], [], "-o", lw=3.6, ms=6.0, color=COLORS[name], label=name)
        finger_lines[name] = line

    edge_lines = []
    for _ in palm_edges():
        (line,) = ax.plot([], [], [], "-", lw=2.6, color="#666666", alpha=0.95)
        edge_lines.append(line)
    ax.legend(loc="upper right")
    txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=11)

    palm_origin = palm["wrist"].copy()
    palm_axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    palm_axis_lines = []
    palm_axis_texts = []
    for axis, color in zip(palm_axes, ("r", "g", "b")):
        (line,) = ax.plot([], [], [], lw=3.2, color=color)
        palm_axis_lines.append(line)
        _set_line3d(line, palm_origin, axis, 30.0)
        palm_axis_texts.append(ax.text(0.0, 0.0, 0.0, "", color=color))

    pp_axis_lines = []
    for color in ("r", "g", "b"):
        (line,) = ax.plot([], [], [], lw=3.2, ls="--", color=color)
        pp_axis_lines.append(line)

    last_valid_pose = {
        "rot": np.eye(3, dtype=np.float64),
        "trans": np.zeros(3, dtype=np.float64),
        "time_wall": -1e9,
        "group": None,
    }

    def update(_frame_idx: int):
        hand_rot = np.eye(3, dtype=np.float64)
        hand_trans = np.zeros(3, dtype=np.float64)
        pose_group = None
        pose_mode = "disabled"
        pose_age_ms = -1.0

        if aruco_pose_tracker is not None:
            pose_mode = "waiting"
            snap = aruco_pose_tracker.snapshot()
            if snap is not None:
                groups = snap.get("groups", {})
                pose_group = _resolve_group_for_hand(
                    groups=groups,
                    aruco_hand_group=aruco_hand_group,
                    aruco_group_hand_map=aruco_group_hand_map,
                    aruco_hand_slot=aruco_hand_slot,
                )
                if pose_group is None:
                    pose_mode = "group_unresolved"
                else:
                    group_data = groups.get(pose_group, {})
                    fusion = group_data.get("fusion", {})
                    fused_pos = fusion.get("fused_position")
                    rep_rvec = fusion.get("rep_rvec")
                    pose_age_ms = (time.time() - float(snap.get("time_wall", 0.0))) * 1000.0
                    if (
                        fused_pos is not None
                        and rep_rvec is not None
                        and pose_age_ms <= float(aruco_pose_hold_sec) * 1000.0
                    ):
                        hand_rot = _rotvec_to_matrix(np.asarray(rep_rvec, dtype=np.float64).reshape(3))
                        hand_trans = np.asarray(fused_pos, dtype=np.float64).reshape(3) * 1000.0
                        pose_mode = str(fusion.get("mode", "ok"))
                        last_valid_pose["rot"] = hand_rot.copy()
                        last_valid_pose["trans"] = hand_trans.copy()
                        last_valid_pose["time_wall"] = time.time()
                        last_valid_pose["group"] = pose_group
                    else:
                        pose_mode = "invalid_or_stale"

        hold_age_ms = (time.time() - float(last_valid_pose["time_wall"])) * 1000.0
        if pose_mode != "disabled" and pose_mode != "waiting":
            if hold_age_ms <= float(aruco_pose_hold_sec) * 1000.0 and float(last_valid_pose["time_wall"]) > 0:
                hand_rot = np.asarray(last_valid_pose["rot"], dtype=np.float64)
                hand_trans = np.asarray(last_valid_pose["trans"], dtype=np.float64)
                if pose_mode not in ("single", "mean_position_only"):
                    pose_mode = "hold_last_valid"
                if pose_group is None:
                    pose_group = last_valid_pose["group"]

        q, stamp, raw_line = reader.snapshot_rad20()
        for name in FINGERS:
            start = FINGER_OFFSET[name]
            pts_local = finger_points(name, q[start : start + 4], skeleton, palm, hand)
            pts = _apply_rigid(pts_local, hand_rot, hand_trans)
            line = finger_lines[name]
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

        thumb_origin_local, x_pp_local, y_pp_local, z_pp_local = thumb_pp_frame(
            q[0:4],
            palm,
            hand,
            thumb_chain_rx_rad(skeleton),
        )
        thumb_origin = (hand_rot @ thumb_origin_local) + hand_trans
        x_pp = hand_rot @ x_pp_local
        y_pp = hand_rot @ y_pp_local
        z_pp = hand_rot @ z_pp_local
        _set_line3d(pp_axis_lines[0], thumb_origin, x_pp, 20.0)
        _set_line3d(pp_axis_lines[1], thumb_origin, y_pp, 20.0)
        _set_line3d(pp_axis_lines[2], thumb_origin, z_pp, 20.0)

        for line, (a, b) in zip(edge_lines, palm_edges()):
            pa = (hand_rot @ palm[a]) + hand_trans
            pb = (hand_rot @ palm[b]) + hand_trans
            line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            line.set_3d_properties([pa[2], pb[2]])

        palm_origin_world = (hand_rot @ palm["wrist"]) + hand_trans
        for axis_line, axis, axis_text, label in zip(palm_axis_lines, palm_axes, palm_axis_texts, ("Px", "Py", "Pz")):
            axis_world = hand_rot @ axis
            _set_line3d(axis_line, palm_origin_world, axis_world, 30.0)
            pos = palm_origin_world + axis_world * 32.0
            axis_text.set_position((float(pos[0]), float(pos[1])))
            axis_text.set_3d_properties(float(pos[2]))
            axis_text.set_text(label)

        age_ms = (time.time() - stamp) * 1000.0 if stamp > 0 else -1.0
        aruco_line = ""
        if aruco_pose_tracker is not None:
            aruco_line = (
                f"aruco={pose_mode} slot={aruco_hand_slot} "
                f"group={pose_group} age={pose_age_ms:6.1f} ms\n"
            )
        txt.set_text(
            f"stream_age={age_ms:6.1f} ms  mode={mode}\n"
            f"{aruco_line}"
            f"thumb.MCP_front={float(np.rad2deg(q[2])):6.1f} deg  "
            f"thumb.MCP_back={float(np.rad2deg(q[3])):6.1f} deg\n"
            f"{raw_line[:100]}"
        )
        return (
            tuple(finger_lines.values())
            + tuple(edge_lines)
            + tuple(pp_axis_lines)
            + tuple(palm_axis_lines)
            + tuple(palm_axis_texts)
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

    def _on_key(event) -> None:
        if event.key != " ":
            return
        with reader.lock:
            for joint in joint_order:
                joint_id = str(joint["id"])
                desired = 0.0
                if joint["finger"] == "thumb" and joint["joint"] == "MCP_front":
                    desired = 60.0
                elif joint["finger"] == "thumb" and joint["joint"] == "MCP_back":
                    desired = 90.0
                current = float(reader.latest_deg.get(joint_id, 0.0))
                reader.alignment_offsets[joint_id] = current - desired
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    try:
        plt.tight_layout()
        plt.show()
    finally:
        reader.stop()
