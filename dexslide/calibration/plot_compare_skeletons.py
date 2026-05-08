from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dexslide.paths import SKELETONS_DIR
from dexslide.visualization.skeleton_plot import (
    FINGER_KEYS,
    FINGER_ORDER,
    _chain_points_2d,
    _flatten_palm_points,
)


def _load_skeleton(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _draw_one(ax, skeleton: dict, label: str, color: str, style: str) -> None:
    wrist2, mcp2 = _flatten_palm_points(skeleton)

    for finger in FINGER_ORDER:
        if finger not in skeleton or finger not in mcp2:
            continue
        lengths = [float(skeleton[finger].get(k, 0.0)) for k in FINGER_KEYS[finger]]
        direction = mcp2[finger] - wrist2
        start = mcp2[finger] if finger == "thumb" else wrist2
        pts = _chain_points_2d(start, direction, lengths)
        arr = np.asarray(pts)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=color,
            linestyle=style,
            linewidth=2.1,
            alpha=0.9,
        )
        ax.scatter(arr[:, 0], arr[:, 1], color=color, s=12, alpha=0.8)

    for a, b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        if a in mcp2 and b in mcp2:
            pa, pb = mcp2[a], mcp2[b]
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                color=color,
                linestyle=style,
                linewidth=1.6,
                alpha=0.8,
            )

    ax.scatter([wrist2[0]], [wrist2[1]], color=color, s=26, alpha=0.9, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay multiple skeleton.json files in one 2D flattened plot."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Skeleton json files to overlay",
    )
    parser.add_argument(
        "--output",
        default="skeleton_compare_overlay.png",
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window after saving",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = (
        sorted(SKELETONS_DIR.glob("*.json"))
        if args.files is None
        else [Path(p).resolve() for p in args.files]
    )
    if not files:
        raise SystemExit(f"No skeleton JSON files found in {SKELETONS_DIR}")
    colors = ["#ff4d4f", "#1677ff", "#22c55e", "#f59e0b", "#8b5cf6", "#14b8a6"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Skeleton Overlay Comparison (2D flattened)")
    ax.set_xlabel("Palm-X (mm)")
    ax.set_ylabel("Palm-Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)

    for idx, path in enumerate(files):
        skeleton = _load_skeleton(path)
        _draw_one(
            ax,
            skeleton=skeleton,
            label=path.stem,
            color=colors[idx % len(colors)],
            style=styles[idx % len(styles)],
        )

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved overlay to: {out}")


if __name__ == "__main__":
    main()
