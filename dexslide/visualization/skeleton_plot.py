"""
Skeleton Plot — 2D palm-flattened review plot after Phase 1 calibration.

This view is designed for quick human inspection of bone lengths and finger order.
"""

from __future__ import annotations

import numpy as np

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_KEYS = {
    "thumb": ["metacarpal", "proximal", "distal"],
    "index": ["proximal", "middle", "distal"],
    "middle": ["proximal", "middle", "distal"],
    "ring": ["proximal", "middle", "distal"],
    "pinky": ["proximal", "middle", "distal"],
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([0.0, 1.0], dtype=np.float64)
    return v / n


def _flatten_palm_points(skeleton: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    palm = skeleton.get("palm", {})
    verts = palm.get("vertices", []) if isinstance(palm, dict) else []
    if isinstance(verts, list) and len(verts) >= 6:
        pts3 = np.asarray(verts[:6], dtype=np.float64)  # wrist, thumb, index, middle, ring, pinky
        wrist3 = pts3[0]

        # Build a palm plane basis from MCP arc and wrist direction.
        index3, middle3, ring3, pinky3 = pts3[2], pts3[3], pts3[4], pts3[5]
        x3 = pinky3 - index3
        if np.linalg.norm(x3) < 1e-6:
            x3 = ring3 - middle3
        x3 = x3 / max(np.linalg.norm(x3), 1e-6)

        y3 = (index3 + middle3 + ring3 + pinky3) * 0.25 - wrist3
        y3 = y3 - x3 * float(np.dot(y3, x3))
        y3 = y3 / max(np.linalg.norm(y3), 1e-6)

        def project(p3: np.ndarray) -> np.ndarray:
            d = p3 - wrist3
            return np.array([float(np.dot(d, x3)), float(np.dot(d, y3))], dtype=np.float64)

        wrist2 = np.array([0.0, 0.0], dtype=np.float64)
        mcp2 = {
            "thumb": project(pts3[1]),
            "index": project(index3),
            "middle": project(middle3),
            "ring": project(ring3),
            "pinky": project(pinky3),
        }
        return wrist2, mcp2

    # Fallback when palm vertices are unavailable.
    wrist2 = np.array([0.0, 0.0], dtype=np.float64)
    mcp2 = {
        "thumb": np.array([-35.0, 10.0], dtype=np.float64),
        "index": np.array([-24.0, 30.0], dtype=np.float64),
        "middle": np.array([-8.0, 34.0], dtype=np.float64),
        "ring": np.array([8.0, 31.0], dtype=np.float64),
        "pinky": np.array([24.0, 24.0], dtype=np.float64),
    }
    return wrist2, mcp2


def _chain_points_2d(start: np.ndarray, direction: np.ndarray, lengths: list[float]) -> list[np.ndarray]:
    pts = [start]
    d = _normalize(direction)
    p = start.copy()
    for l in lengths:
        p = p + d * float(max(0.0, l))
        pts.append(p.copy())
    return pts


def show_skeleton_plot(skeleton: dict, title: str = "DexSlide Skeleton Review (2D flattened)"):
    """
    Render a palm-flattened 2D skeleton preview for sanity-check.
    """
    import matplotlib.pyplot as plt

    wrist2, mcp2 = _flatten_palm_points(skeleton)

    colors = {
        "thumb": "#FF6B6B",
        "index": "#4D96FF",
        "middle": "#6BCB77",
        "ring": "#FFD93D",
        "pinky": "#A66CFF",
    }

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title)
    ax.set_xlabel("Palm-X (mm)")
    ax.set_ylabel("Palm-Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    # Draw palm fan edges from wrist to MCP points.
    for finger in FINGER_ORDER:
        if finger not in mcp2:
            continue
        p = mcp2[finger]
        ax.plot([wrist2[0], p[0]], [wrist2[1], p[1]], color="#999999", linewidth=1.2, alpha=0.9)

    # Draw MCP arc order: index->middle->ring->pinky (requested palm edges)
    for a, b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        if a in mcp2 and b in mcp2:
            pa, pb = mcp2[a], mcp2[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#666666", linewidth=2.2)

    # Draw finger chains in plane.
    for finger in FINGER_ORDER:
        if finger not in skeleton or finger not in mcp2:
            continue
        lengths = [float(skeleton[finger].get(k, 0.0)) for k in FINGER_KEYS[finger]]
        direction = mcp2[finger] - wrist2
        # Compact skeleton stores MCP bases in palm.vertices, so every finger chain starts there.
        chain_start = mcp2[finger]
        pts = _chain_points_2d(chain_start, direction, lengths)
        arr = np.asarray(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=colors[finger], linewidth=2.8)
        ax.scatter(arr[:, 0], arr[:, 1], color=colors[finger], s=20)
        tip = arr[-1]
        ax.text(tip[0], tip[1], finger, color=colors[finger], fontsize=9)

    ax.scatter([wrist2[0]], [wrist2[1]], color="black", s=28)
    ax.text(wrist2[0], wrist2[1], "wrist", color="black", fontsize=9)

    plt.tight_layout()
    plt.show()
