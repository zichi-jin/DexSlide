"""Backward-compatible facade for marker-body pose and overlay helpers."""

from dexslide.vision import marker_body_pose as _impl
from dexslide.vision.marker_body_pose import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)

