#!/usr/bin/env python3
from __future__ import annotations

"""
Simple mapper: render MANO mesh and overlay DexSlide joint positions (live or CSV/demo).

Behavior:
 - Loads MANO_LEFT/RIGHT.pkl from assets if available.
 - Uses package-level DexSlide kinematics to compute finger joint world positions.
 - Displays MANO mesh (translated to palm wrist) and animated joint markers.

This is an approximate visualization (no full MANO skinning). It aligns the MANO mesh
centroid to the glove palm wrist and overlays joint markers computed from the glove kinematics.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dexslide.kinematics.live_hand import (
    apply_handedness,
    canonicalize_palm_xoy,
    finger_points,
)
from dexslide.paths import DEFAULT_FIRMWARE_CALIBRATION_FILE, DEFAULT_SKELETON_FILE
from dexslide.serial_angles import AngleStreamReader, load_calibration, make_joint_order, pick_default_port


def load_mano_pkl(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    # Typical keys: 'v_template', 'f' or 'faces'
    verts = np.array(data.get("v_template") or data.get("v_template"), dtype=np.float64)
    faces = data.get("f") or data.get("faces") or data.get("facedata")
    if faces is None:
        raise RuntimeError("MANO file missing faces key")
    faces = np.asarray(faces, dtype=np.int32)
    return verts, faces


def main():
    parser = argparse.ArgumentParser(description="Map DexSlide live motion onto MANO mesh (approx)")
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--mano-dir", default=str(Path(__file__).resolve().parents[1] / "assets" / "mano" / "models"))
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--live", action="store_true", help="Use live serial stream via AngleStreamReader")
    parser.add_argument("--port", default=pick_default_port())
    parser.add_argument("--calib-file", default=str(DEFAULT_FIRMWARE_CALIBRATION_FILE))
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    mano_file = Path(args.mano_dir) / ("MANO_LEFT.pkl" if args.hand == "left" else "MANO_RIGHT.pkl")
    if not mano_file.exists():
        raise SystemExit(f"MANO model not found: {mano_file}")

    verts, faces = load_mano_pkl(mano_file)

    with open(args.skeleton_file, "r", encoding="utf-8") as f:
        skeleton = json.load(f)
    palm = apply_handedness(canonicalize_palm_xoy(skeleton), args.hand)

    # Scale: MANO verts are in meters (usually); glove skeleton in mm -> convert mm->m
    mesh_verts_m = verts.copy()

    # Compute mesh centroid and translate to palm wrist
    mesh_centroid = mesh_verts_m.mean(axis=0)
    wrist_m = palm["wrist"] / 1000.0
    mesh_offset = wrist_m - mesh_centroid

    # Setup live reader if requested
    reader = None
    joint_order = make_joint_order()
    calibration = load_calibration(Path(args.calib_file), joint_order)
    if args.live:
        reader = AngleStreamReader(args.port, 115200, "raw", joint_order, calibration)
        reader.start()

    # Demo trajectory generator (simple sinusoid) for offline testing
    def gen_demo(num_frames=300):
        t = np.linspace(0.0, 2.0 * np.pi, num_frames)
        q = np.zeros((num_frames, 20), dtype=np.float64)
        for i, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            s = i * 4
            q[:, s + 0] = 0.4 * np.sin(t + i)
            q[:, s + 1] = 0.7 * np.sin(t + i)
            q[:, s + 2] = 0.9 * np.sin(t + i)
            q[:, s + 3] = 0.2 * np.sin(0.5 * t + i)
        return q

    demo_traj = gen_demo(300)

    fig = plt.figure("DexSlide MANO", figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("MANO Overlay")
    ax.set_box_aspect((1.2, 1.0, 1.0))

    # Prepare mesh collection
    face_verts = [mesh_verts_m[f] + mesh_offset for f in faces]
    mesh = Poly3DCollection(face_verts, facecolor=(0.8, 0.8, 0.9), edgecolor=(0.3, 0.3, 0.3), alpha=0.9)
    ax.add_collection3d(mesh)

    # Joint markers (we will animate positions in meters)
    joint_markers = []
    marker_colors = ["r", "g", "b", "k", "m", "c", "y"]
    for i in range(21):
        (ln,) = ax.plot([], [], [], "o", ms=5, color=marker_colors[i % len(marker_colors)])
        joint_markers.append(ln)

    ax.auto_scale_xyz([mesh_verts_m[:, 0].min() + mesh_offset[0], mesh_verts_m[:, 0].max() + mesh_offset[0]],
                      [mesh_verts_m[:, 1].min() + mesh_offset[1], mesh_verts_m[:, 1].max() + mesh_offset[1]],
                      [mesh_verts_m[:, 2].min() + mesh_offset[2], mesh_verts_m[:, 2].max() + mesh_offset[2]])

    def update(frame_idx):
        if reader is not None:
            q, tstamp, _ = reader.snapshot_rad20()
        else:
            q = demo_traj[frame_idx % len(demo_traj)]

        # build joint point list: wrist + 4 joints per finger (thumb,index,middle,ring,pinky)
        points = []
        # wrist
        points.append(palm["wrist"]/1000.0)
        for name in ["thumb", "index", "middle", "ring", "pinky"]:
            s = {"thumb":0,"index":4,"middle":8,"ring":12,"pinky":16}[name]
            pts = finger_points(name, q[s:s+4], skeleton, palm, args.hand)
            for p in pts:
                points.append(p / 1000.0)

        pts_m = np.asarray(points)

        # Update joint markers
        for ln, p in zip(joint_markers, pts_m[: len(joint_markers)]):
            ln.set_data([p[0]], [p[1]])
            ln.set_3d_properties([p[2]])

        return tuple(joint_markers) + (mesh,)

    anim = FuncAnimation(fig, update, frames=1000, interval=1000.0/args.fps, blit=False)
    fig._anim = anim

    try:
        plt.show()
    finally:
        if reader is not None:
            reader.stop()


if __name__ == "__main__":
    main()
