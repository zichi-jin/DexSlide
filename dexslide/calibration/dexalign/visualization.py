from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dexslide.retargeting.human_model import HUMAN_LANDMARK_NAMES

from .objective import AlignmentEvaluation


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array), array, 0.0)


def _select_sample_indices(used_frame_mask: np.ndarray, *, count: int) -> list[int]:
    used = np.flatnonzero(np.asarray(used_frame_mask, dtype=bool))
    if used.size == 0:
        return []
    if used.size <= count:
        return used.tolist()
    sample_positions = np.linspace(0, used.size - 1, count, dtype=int)
    return used[sample_positions].tolist()


def _set_axes_equal(ax: plt.Axes, points_xyz: np.ndarray) -> None:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.55, 30.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_hand_lines(ax: plt.Axes, points_xyz: np.ndarray, *, color: str, alpha: float) -> None:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    for parent_idx, child_idx in HAND_CONNECTIONS:
        segment = points[[parent_idx, child_idx]]
        if not np.isfinite(segment).all():
            continue
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, alpha=alpha, linewidth=1.4)


def save_alignment_plots(
    output_dir: str | Path,
    before_eval: AlignmentEvaluation,
    after_eval: AlignmentEvaluation,
    *,
    sample_count: int = 4,
) -> dict[str, str]:
    base_dir = Path(output_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = base_dir / "frame_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    keypoint_path = base_dir / "plots_keypoint_error.png"
    frame_path = base_dir / "plots_frame_error.png"

    labels = HUMAN_LANDMARK_NAMES
    x = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.18, _finite_or_zero(before_eval.keypoint_mean_error_mm), width=0.36, label="before", color="#d97706")
    ax.bar(x + 0.18, _finite_or_zero(after_eval.keypoint_mean_error_mm), width=0.36, label="after", color="#15803d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Error (mm)")
    ax.set_title("DexAlign Keypoint Error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(keypoint_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(_finite_or_zero(before_eval.frame_mean_error_mm), label="before", color="#d97706", linewidth=1.4)
    ax.plot(_finite_or_zero(after_eval.frame_mean_error_mm), label="after", color="#15803d", linewidth=1.4)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Mean Error (mm)")
    ax.set_title("DexAlign Frame Mean Error")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(frame_path, dpi=180)
    plt.close(fig)

    sample_paths: list[str] = []
    for frame_idx in _select_sample_indices(after_eval.used_frame_mask, count=sample_count):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        observed = after_eval.observed_keypoints_camera_mm[frame_idx]
        predicted_before = before_eval.predicted_keypoints_camera_mm[frame_idx]
        predicted_after = after_eval.predicted_keypoints_camera_mm[frame_idx]
        metric_mask = after_eval.metric_mask[frame_idx]

        observed_plot = np.where(metric_mask[:, None], observed, np.nan)
        ax.scatter(observed_plot[:, 0], observed_plot[:, 1], observed_plot[:, 2], color="#2563eb", s=18, label="obs")
        ax.scatter(predicted_before[:, 0], predicted_before[:, 1], predicted_before[:, 2], color="#d97706", s=14, label="before")
        ax.scatter(predicted_after[:, 0], predicted_after[:, 1], predicted_after[:, 2], color="#15803d", s=14, label="after")
        _draw_hand_lines(ax, observed_plot, color="#2563eb", alpha=0.7)
        _draw_hand_lines(ax, predicted_before, color="#d97706", alpha=0.5)
        _draw_hand_lines(ax, predicted_after, color="#15803d", alpha=0.7)

        stacked = np.concatenate([observed_plot, predicted_before, predicted_after], axis=0)
        _set_axes_equal(ax, stacked)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"Frame {frame_idx}")
        ax.legend(loc="upper left")
        fig.tight_layout()
        sample_path = sample_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(sample_path, dpi=180)
        plt.close(fig)
        sample_paths.append(str(sample_path))

    return {
        "plots_keypoint_error": str(keypoint_path),
        "plots_frame_error": str(frame_path),
        "frame_samples_dir": str(sample_dir),
        "frame_samples": sample_paths,
    }
