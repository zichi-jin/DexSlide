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
    for axis, color in zip(palm_axes, ("r", "g", "b")):
        (line,) = ax.plot([], [], [], lw=3.2, color=color)
        palm_axis_lines.append(line)
        _set_line3d(line, palm_origin, axis, 30.0)
    ax.text(*(palm_origin + palm_axes[0] * 32.0), "Px", color="r")
    ax.text(*(palm_origin + palm_axes[1] * 32.0), "Py", color="g")
    ax.text(*(palm_origin + palm_axes[2] * 32.0), "Pz", color="b")

    pp_axis_lines = []
    for color in ("r", "g", "b"):
        (line,) = ax.plot([], [], [], lw=3.2, ls="--", color=color)
        pp_axis_lines.append(line)

    def update(_frame_idx: int):
        q, stamp, raw_line = reader.snapshot_rad20()
        for name in FINGERS:
            start = FINGER_OFFSET[name]
            pts = finger_points(name, q[start : start + 4], skeleton, palm, hand)
            line = finger_lines[name]
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

        thumb_origin, x_pp, y_pp, z_pp = thumb_pp_frame(q[0:4], palm, hand)
        _set_line3d(pp_axis_lines[0], thumb_origin, x_pp, 20.0)
        _set_line3d(pp_axis_lines[1], thumb_origin, y_pp, 20.0)
        _set_line3d(pp_axis_lines[2], thumb_origin, z_pp, 20.0)

        for line, (a, b) in zip(edge_lines, palm_edges()):
            pa, pb = palm[a], palm[b]
            line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            line.set_3d_properties([pa[2], pb[2]])

        age_ms = (time.time() - stamp) * 1000.0 if stamp > 0 else -1.0
        txt.set_text(
            f"stream_age={age_ms:6.1f} ms  mode={mode}\n"
            f"thumb.MCP_front={float(np.rad2deg(q[2])):6.1f} deg  "
            f"thumb.MCP_back={float(np.rad2deg(q[3])):6.1f} deg\n"
            f"{raw_line[:100]}"
        )
        return tuple(finger_lines.values()) + tuple(edge_lines) + tuple(pp_axis_lines) + (txt,)

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
                    desired = 90.0
                elif joint["finger"] == "thumb" and joint["joint"] == "MCP_back":
                    desired = 60.0
                current = float(reader.latest_deg.get(joint_id, 0.0))
                reader.alignment_offsets[joint_id] = current - desired
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    try:
        plt.tight_layout()
        plt.show()
    finally:
        reader.stop()
