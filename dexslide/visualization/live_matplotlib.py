"""Compatibility API for the scene-backed Plot3D viewer.

New code should import :class:`DexSlidePlot3DViewer` directly.
"""

from __future__ import annotations

from dexslide.streaming import DexSlideScene
from dexslide.visualization.plot3d_viewer import DexSlidePlot3DViewer


def run_live_viewer(
    scene: DexSlideScene,
    *,
    rate_hz: float | None = None,
    show_skeleton: bool = True,
    plot_range_m: float = 0.45,
    plot_fps: float = 20.0,
) -> None:
    """Consume a shared scene with the Plot3D viewer until its window closes."""

    viewer = DexSlidePlot3DViewer(
        scene,
        show_skeleton=show_skeleton,
        plot_range_m=plot_range_m,
        max_refresh_hz=plot_fps,
    )
    try:
        with scene:
            for sample in scene.samples(rate_hz=rate_hz):
                if not viewer.update(sample):
                    break
    finally:
        viewer.close()


__all__ = ["DexSlidePlot3DViewer", "run_live_viewer"]
