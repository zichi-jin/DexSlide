"""Small public contracts shared by robot-specific teleop endpoints."""

from __future__ import annotations

from typing import Protocol

from dexslide.streaming import DexSlideSceneSample


class TeleopEndpoint(Protocol):
    """A robot-specific consumer of immutable DexSlide scene samples."""

    @property
    def name(self) -> str: ...

    def start(self) -> None: ...

    def consume(self, sample: DexSlideSceneSample) -> None: ...

    def status_lines(self) -> list[str]: ...

    def close(self) -> None: ...


__all__ = ["TeleopEndpoint"]
