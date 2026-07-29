"""Stable CLI facade for marker-to-wrist calibration."""

from . import marker_wrist_offset as _impl
from .marker_wrist_offset import BodyPoseEstimate, main

__all__ = ["BodyPoseEstimate", "main"]


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

