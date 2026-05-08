#!/usr/bin/env python3
"""
Play a 20-DOF hand trajectory with matplotlib 3D.

Geometric conventions (as requested):
  - Palm lies on XOY plane (z = 0 for palm base points)
  - Four fingers (index/middle/ring/pinky) point to +X
  - Thumb opens toward -Y for left hand, +Y for right hand
  - Each finger bends in its own plane:
      MCP_front / PIP / DIP are parallel-axis flexion in one plane
      MCP_back is a small in-plane swing (ab/ad), limited for four fingers

CSV trajectory format:
  - Shape: N x 20
  - Joint order: [thumb4, index4, middle4, ring4, pinky4]
  - Per finger 4 values: [DIP, PIP, MCP_front, MCP_back]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ensure parent of the `dexslide` package is on `sys.path` so the script
# can be executed directly (e.g. `python demo_20dof_matplotlib.py`) from
# the `visualization/` directory. If you prefer, run the module instead:
#   python -m dexslide.visualization.demo_20dof_matplotlib
import os
import sys

ROOT_PARENT = str(Path(__file__).resolve().parents[2])
if ROOT_PARENT not in sys.path:
    sys.path.insert(0, ROOT_PARENT)

import numpy as np

from dexslide.paths import DEFAULT_SKELETON_FILE


FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_OFFSET = {"thumb": 0, "index": 4, "middle": 8, "ring": 12, "pinky": 16}
COLORS = {
    "thumb": "#e76f51",
    "index": "#3a86ff",
    "middle": "#06d6a0",
    "ring": "#ffbe0b",
    "pinky": "#8338ec",
}
FINGER_MCP_BACK_LIMIT_RAD = np.deg2rad(12.0)
THUMB_MCP_BACK_LIMIT_RAD = np.deg2rad(35.0)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-9:
        return fallback.astype(np.float64).copy()
    return (v / n).astype(np.float64)


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _extract_palm_points(skeleton: dict) -> dict[str, np.ndarray]:
    """Extract wrist/thumb/index/middle/ring/pinky from skeleton palm vertices."""
    verts = skeleton.get("palm", {}).get("vertices", [])
    if isinstance(verts, list) and len(verts) >= 6:
        pts = np.asarray(verts[:6], dtype=np.float64)
        return {
            "wrist": pts[0],
            "thumb": pts[1],
            "index": pts[2],
            "middle": pts[3],
            "ring": pts[4],
            "pinky": pts[5],
        }

    # Fallback synthetic palm layout in mm.
    return {
        "wrist": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "thumb": np.array([8.0, 35.0, 0.0], dtype=np.float64),
        "index": np.array([20.0, -22.0, 0.0], dtype=np.float64),
        "middle": np.array([25.0, 0.0, 0.0], dtype=np.float64),
        "ring": np.array([22.0, 20.0, 0.0], dtype=np.float64),
        "pinky": np.array([16.0, 38.0, 0.0], dtype=np.float64),
    }


def _canonicalize_palm_xoy(skeleton: dict) -> dict[str, np.ndarray]:
    """
    Rigid-align palm points to requested frame:
      +X: four-finger direction
      +Y: thumb opening side when inferred positive; sign can be flipped by handedness
      palm plane -> XOY (z=0 for base points)
    """
    p = _extract_palm_points(skeleton)
    wrist = p["wrist"]
    four_center = 0.25 * (p["index"] + p["middle"] + p["ring"] + p["pinky"])

    x_raw = _unit(four_center - wrist, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    thumb_side = p["thumb"] - four_center
    z_raw = np.cross(x_raw, thumb_side)
    if _norm(z_raw) < 1e-6:
        z_raw = np.cross(x_raw, p["index"] - p["pinky"])
    z_raw = _unit(z_raw, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y_raw = _unit(np.cross(z_raw, x_raw), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    z_raw = _unit(np.cross(x_raw, y_raw), np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # Columns are basis vectors in source frame.
    R = np.stack([x_raw, y_raw, z_raw], axis=1)  # source->canonical via R.T
    out = {}
    for k, v in p.items():
        out[k] = (R.T @ (v - wrist)).astype(np.float64)

    # Force palm points onto XOY plane (static palm web).
    for k in out:
        out[k][2] = 0.0
    return out


def _apply_handedness(
    palm: dict[str, np.ndarray],
    hand: str,
) -> dict[str, np.ndarray]:
    """
    Align thumb-side sign by hand:
      left  -> thumb -Y
      right -> thumb +Y
      auto  -> keep inferred orientation
    """
    if hand == "auto":
        return {k: v.copy() for k, v in palm.items()}

    desired = -1.0 if hand == "left" else 1.0
    current = 1.0 if palm["thumb"][1] >= 0.0 else -1.0
    out = {k: v.copy() for k, v in palm.items()}
    if current != desired:
        for k in out:
            out[k][1] *= -1.0
    return out


def _finger_lengths(name: str, skeleton: dict) -> list[float]:
    f = skeleton.get(name, {})
    if name == "thumb":
        return [
            float(f.get("metacarpal", 30.0)),
            float(f.get("proximal", 24.0)),
            float(f.get("distal", 18.0)),
        ]
    return [
        float(f.get("proximal", 25.0)),
        float(f.get("middle", 18.0)),
        float(f.get("distal", 14.0)),
    ]


def _finger_points(
    name: str,
    raw4: np.ndarray,
    skeleton: dict,
    palm: dict[str, np.ndarray],
    hand: str = "auto",
    thumb_link_mode: str = "fixed",
) -> np.ndarray:
    """
    Compute one finger polyline in 3D.
    raw4 order: [DIP, PIP, MCP_front, MCP_back]
    """
    dip, pip, mcp_front, mcp_back = [float(v) for v in raw4]
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if name == "thumb":
        base = palm["thumb"].copy()
        mcp_back = float(mcp_back)
        thumb_sign = -1.0 if hand == "left" else 1.0
        if hand == "auto":
            thumb_sign = -1.0 if base[1] < 0.0 else 1.0
        f0 = np.array([0.0, thumb_sign, 0.0], dtype=np.float64)
    else:
        base = palm[name].copy()
        mcp_back = float(mcp_back)
        f0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # four fingers nominal +X

    f = _rot_z(mcp_back) @ f0
    lengths = _finger_lengths(name, skeleton)
    a1 = mcp_front
    a2 = mcp_front + pip
    a3 = mcp_front + pip + dip
    ang = [a1, a2, a3]

    pts = [base]
    p = base.copy()
    for L, th in zip(lengths, ang):
        # Finger bends in its own plane spanned by (f, z).
        v = np.cos(th) * f - np.sin(th) * z_axis
        p = p + float(max(0.0, L)) * v
        pts.append(p.copy())

    return np.asarray(pts, dtype=np.float64)


def _generate_demo_trajectory(num_frames: int, cycle_sec: float, fps: float) -> np.ndarray:
    t = np.arange(num_frames, dtype=np.float64) / max(1e-6, fps)
    w = 2.0 * np.pi / max(1e-6, cycle_sec)
    q = np.zeros((num_frames, 20), dtype=np.float64)

    phase = {"thumb": 0.1, "index": 0.0, "middle": 0.35, "ring": 0.7, "pinky": 1.0}
    for name in FINGERS:
        s = FINGER_OFFSET[name]
        p = phase[name]

        flex = 0.5 * (1.0 + np.sin(w * t + p))  # 0..1
        q[:, s + 0] = 0.55 * flex  # DIP
        q[:, s + 1] = 0.85 * flex  # PIP
        q[:, s + 2] = 1.05 * flex  # MCP_front

        if name == "thumb":
            q[:, s + 3] = 0.42 * np.sin(0.6 * w * t + 0.8)  # opposition spread (larger)
        else:
            q[:, s + 3] = 0.09 * np.sin(w * t + p + 0.3)  # realistic small ab/ad
    return q


def _load_csv(path: Path, degrees: bool) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 20:
        raise ValueError(f"Trajectory must be N x 20, got {data.shape}")
    if degrees:
        data = np.deg2rad(data)
    return data


def _palm_edges() -> list[tuple[str, str]]:
    # Spider-web style edges from wrist + MCP polygon edges.
    return [
        ("wrist", "thumb"),
        ("wrist", "index"),
        ("wrist", "middle"),
        ("wrist", "ring"),
        ("wrist", "pinky"),
        ("thumb", "index"),
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Matplotlib 3D 20-DOF hand demo")
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--trajectory-csv", default=None, help="Optional N x 20 CSV")
    parser.add_argument("--degrees", action="store_true", help="CSV is in degrees")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=8.0, help="Used for demo trajectory")
    parser.add_argument("--cycle-sec", type=float, default=3.0, help="Gesture cycle period")
    parser.add_argument(
        "--hand",
        choices=["auto", "left", "right"],
        default="left",
        help="Handedness for thumb side (-Y for left, +Y for right)",
    )
    parser.add_argument("--thumb-link-mode", choices=["fixed", "dynamic"], default="fixed", help="Thumb first-link mode: fixed (legacy) or dynamic (experimental)")
    parser.add_argument("--no-show", action="store_true", help="Build geometry and exit")
    args = parser.parse_args()

    with open(args.skeleton_file, "r", encoding="utf-8") as f:
        skeleton = json.load(f)

    palm = _apply_handedness(_canonicalize_palm_xoy(skeleton), args.hand)

    if args.trajectory_csv:
        traj = _load_csv(Path(args.trajectory_csv), degrees=args.degrees)
    else:
        num_frames = max(2, int(args.duration * args.fps))
        traj = _generate_demo_trajectory(num_frames, args.cycle_sec, args.fps)

    if args.no_show:
        print(f"trajectory shape: {traj.shape}")
        print("canonical palm points (mm):")
        for k in ["wrist", "thumb", "index", "middle", "ring", "pinky"]:
            v = palm[k]
            print(f"  {k:6s}: [{v[0]:8.2f}, {v[1]:8.2f}, {v[2]:8.2f}]")
        sample = traj[0]
        idx_pts = _finger_points("index", sample[4:8], skeleton, palm, args.hand, args.thumb_link_mode)
        th_pts = _finger_points("thumb", sample[0:4], skeleton, palm, args.hand, args.thumb_link_mode)
        print(f"index sample:\n{idx_pts}")
        print(f"thumb sample:\n{th_pts}")
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure("DexSlide 20DOF Demo", figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("20-DOF Scripted Hand Motion (Palm on XOY)")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.4, 1.0, 0.9))
    ax.grid(True, alpha=0.25)

    # Axis range tuned to forward +X motion.
    ax.set_xlim(-35, 185)
    ax.set_ylim(-95, 95)
    ax.set_zlim(-125, 55)

    finger_lines = {}
    for name in FINGERS:
        (line,) = ax.plot([], [], [], "-o", lw=3.0, ms=4.5, color=COLORS[name], label=name)
        finger_lines[name] = line

    edge_lines = []
    for _ in _palm_edges():
        (line,) = ax.plot([], [], [], "-", lw=2.0, color="#666666", alpha=0.9)
        edge_lines.append(line)

    txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(frame_idx: int):
        q = traj[frame_idx]

        for name in FINGERS:
            s = FINGER_OFFSET[name]
            pts = _finger_points(name, q[s : s + 4], skeleton, palm, args.hand, args.thumb_link_mode)
            line = finger_lines[name]
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

        # Static palm spider-web from skeleton-derived MCP positions.
        for line, (a, b) in zip(edge_lines, _palm_edges()):
            pa, pb = palm[a], palm[b]
            line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            line.set_3d_properties([pa[2], pb[2]])

        txt.set_text(f"frame: {frame_idx + 1}/{len(traj)}")
        return tuple(finger_lines.values()) + tuple(edge_lines) + (txt,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(traj),
        interval=1000.0 / max(1e-6, args.fps),
        blit=False,
        repeat=True,
    )
    # Keep a live reference so matplotlib does not garbage-collect the animation.
    fig._anim = anim  # type: ignore[attr-defined]
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
