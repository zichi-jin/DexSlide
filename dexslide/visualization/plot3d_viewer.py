"""Matplotlib Plot3D consumer for :mod:`dexslide.streaming`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexslide.kinematics.transforms import transform_points
from dexslide.retargeting.human_model import DexSlideHumanModel
from dexslide.streaming import DexSlideScene, DexSlideSceneSample


_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
_PALM_EDGES = (
    (0, 1),
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
)
_FINGER_COLORS = ("#e76f51", "#3a86ff", "#06d6a0", "#ffbe0b", "#8338ec")
_AXIS_COLORS = ("r", "g", "b")


@dataclass
class _HandArtists:
    finger_lines: tuple[Any, ...]
    palm_lines: tuple[Any, ...]
    body_axis_lines: tuple[Any, ...]
    wrist_axis_lines: tuple[Any, ...]

    def all_lines(self) -> tuple[Any, ...]:
        return self.finger_lines + self.palm_lines + self.body_axis_lines + self.wrist_axis_lines


def _set_line3d(line: Any, points: np.ndarray) -> None:
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    line.set_data(coords[:, 0], coords[:, 1])
    line.set_3d_properties(coords[:, 2])


def _set_axes(lines: tuple[Any, ...], transform: np.ndarray, length_m: float) -> None:
    pose = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    origin = pose[:3, 3]
    rotation = pose[:3, :3]
    for axis_index, line in enumerate(lines):
        endpoint = origin + rotation[:, axis_index] * float(length_m)
        _set_line3d(line, np.stack([origin, endpoint]))
        line.set_visible(True)


def _set_visible(lines: tuple[Any, ...], visible: bool) -> None:
    for line in lines:
        line.set_visible(bool(visible))


class DexSlidePlot3DViewer:
    """Render scene poses and reconstructed hand skeletons in the table frame."""

    def __init__(
        self,
        scene: DexSlideScene,
        *,
        window_name: str = "DexSlide Plot3D",
        show_skeleton: bool = True,
        plot_range_m: float = 0.45,
        axis_length_m: float = 0.06,
        max_refresh_hz: float = 20.0,
    ) -> None:
        if plot_range_m <= 0.0:
            raise ValueError("plot_range_m must be positive")
        if axis_length_m <= 0.0:
            raise ValueError("axis_length_m must be positive")
        if max_refresh_hz < 0.0:
            raise ValueError("max_refresh_hz must be non-negative")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("DexSlidePlot3DViewer requires matplotlib") from exc

        self.scene = scene
        self.show_skeleton = bool(show_skeleton)
        self.plot_range_m = float(plot_range_m)
        self.axis_length_m = float(axis_length_m)
        self.max_refresh_hz = float(max_refresh_hz)
        self.closed = False
        self._next_refresh_time = 0.0
        self._plt = plt
        self._models: dict[str, DexSlideHumanModel] = {}
        self._artists: dict[str, _HandArtists] = {}

        skeleton_files = getattr(scene, "hand_skeleton_files", {})
        handedness = getattr(scene, "hand_handedness", {})
        for hand_id, skeleton_file in skeleton_files.items():
            try:
                self._models[str(hand_id)] = DexSlideHumanModel(
                    skeleton_file,
                    hand=str(handedness.get(hand_id, hand_id)),
                )
            except Exception:
                continue

        plt.ion()
        self.figure = plt.figure(str(window_name), figsize=(9, 8))
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._interactive_canvas = (
            self.figure.canvas.__class__.__module__ != "matplotlib.backends.backend_agg"
        )
        self.axes.set_title("DexSlide — table frame")
        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")
        self.axes.set_zlabel("Z (m)")
        self.axes.set_box_aspect((1.0, 1.0, 1.0))
        self.axes.view_init(elev=24.0, azim=-58.0)
        self.axes.grid(True, alpha=0.25)
        self._set_plot_limits(self.plot_range_m)

        self._table_axis_lines = tuple(
            self.axes.plot([], [], [], color=color, linewidth=3.0)[0]
            for color in _AXIS_COLORS
        )
        _set_axes(self._table_axis_lines, np.eye(4, dtype=np.float64), self.axis_length_m)
        if self._interactive_canvas:
            self.figure.show()

    def _on_close(self, _event: Any) -> None:
        self.closed = True

    def _set_plot_limits(self, radius_m: float) -> None:
        radius = float(radius_m)
        self.axes.set_xlim(-radius, radius)
        self.axes.set_ylim(-radius, radius)
        self.axes.set_zlim(-radius, radius)

    def _ensure_hand_artists(self, hand_id: str, hand_index: int) -> _HandArtists:
        existing = self._artists.get(hand_id)
        if existing is not None:
            return existing
        line_style = "-" if hand_index % 2 == 0 else "--"
        finger_lines = tuple(
            self.axes.plot(
                [],
                [],
                [],
                color=color,
                linestyle=line_style,
                linewidth=2.8,
                marker="o",
                markersize=4.0,
            )[0]
            for color in _FINGER_COLORS
        )
        palm_lines = tuple(
            self.axes.plot([], [], [], color="#666666", linestyle=line_style, linewidth=2.0)[0]
            for _ in _PALM_EDGES
        )
        body_axis_lines = tuple(
            self.axes.plot([], [], [], color=color, linestyle=":", linewidth=2.0)[0]
            for color in _AXIS_COLORS
        )
        wrist_axis_lines = tuple(
            self.axes.plot([], [], [], color=color, linestyle=line_style, linewidth=2.8)[0]
            for color in _AXIS_COLORS
        )
        artists = _HandArtists(
            finger_lines=finger_lines,
            palm_lines=palm_lines,
            body_axis_lines=body_axis_lines,
            wrist_axis_lines=wrist_axis_lines,
        )
        self._artists[hand_id] = artists
        return artists

    def _update_skeleton(
        self,
        hand_id: str,
        hand: Any,
        artists: _HandArtists,
    ) -> np.ndarray | None:
        model = self._models.get(hand_id)
        if not self.show_skeleton or not hand.joints_valid or model is None:
            _set_visible(artists.finger_lines + artists.palm_lines, False)
            return None
        angles = np.asarray(hand.joint_angles, dtype=np.float64)
        if hand.joint_unit == "deg":
            angles = np.deg2rad(angles)
        landmarks_local = model.landmarks_from_angles(angles)
        landmarks_table = transform_points(
            np.asarray(hand.transform_table_hand, dtype=np.float64),
            landmarks_local,
        )
        for line, indices in zip(artists.finger_lines, _FINGER_CHAINS):
            _set_line3d(line, landmarks_table[np.asarray(indices, dtype=np.int64)])
            line.set_visible(True)
        for line, (first, second) in zip(artists.palm_lines, _PALM_EDGES):
            _set_line3d(line, landmarks_table[[first, second]])
            line.set_visible(True)
        return landmarks_table

    def update(self, sample: DexSlideSceneSample) -> bool:
        """Draw the latest sample and return ``False`` after the window closes."""

        if self.closed or not self._plt.fignum_exists(self.figure.number):
            self.closed = True
            return False
        now = time.monotonic()
        if self.max_refresh_hz > 0.0 and now < self._next_refresh_time:
            return True
        if self.max_refresh_hz > 0.0:
            self._next_refresh_time = now + 1.0 / self.max_refresh_hz

        visible_points = [np.zeros((1, 3), dtype=np.float64)]
        for hand_index, (hand_id, hand) in enumerate(sample.hands.items()):
            artists = self._ensure_hand_artists(hand_id, hand_index)
            if not hand.pose_valid:
                _set_visible(artists.all_lines(), False)
                continue
            body_transform = (
                hand.transform_table_body
                if hand.transform_table_body is not None
                else hand.transform_table_hand
            )
            _set_axes(artists.body_axis_lines, body_transform, self.axis_length_m)
            _set_axes(artists.wrist_axis_lines, hand.transform_table_hand, self.axis_length_m)
            visible_points.append(np.asarray(body_transform, dtype=np.float64)[:3, 3].reshape(1, 3))
            visible_points.append(
                np.asarray(hand.transform_table_hand, dtype=np.float64)[:3, 3].reshape(1, 3)
            )
            skeleton_points = self._update_skeleton(hand_id, hand, artists)
            if skeleton_points is not None:
                visible_points.append(skeleton_points)

        max_abs = float(np.max(np.abs(np.vstack(visible_points))))
        if max_abs > self.plot_range_m * 0.90:
            self.plot_range_m = max(self.plot_range_m, max_abs * 1.25)
            self._set_plot_limits(self.plot_range_m)
        self.figure.canvas.draw_idle()
        if self._interactive_canvas:
            self.figure.canvas.flush_events()
            self._plt.pause(0.001)
        else:
            self.figure.canvas.draw()
        return not self.closed

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._plt.close(self.figure)


__all__ = ["DexSlidePlot3DViewer"]
