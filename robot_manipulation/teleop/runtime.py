"""Shared DexSlide sample loop for one or more robot endpoints."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

from dexslide.recording import DexSlideRecorder
from dexslide.streaming import DexSlideScene
from dexslide.visualization import DexSlideARViewer

from .contracts import TeleopEndpoint


class DexSlideTeleopRuntime:
    """Own one scene and dispatch every sample to independent endpoints."""

    def __init__(
        self,
        scene: DexSlideScene,
        endpoints: Sequence[TeleopEndpoint],
        *,
        loop_hz: float | None = None,
        print_interval_s: float = 1.0,
        recorder: DexSlideRecorder | None = None,
        viewer: DexSlideARViewer | None = None,
    ) -> None:
        if loop_hz is not None and float(loop_hz) <= 0.0:
            raise ValueError("loop_hz must be positive when provided")
        if float(print_interval_s) <= 0.0:
            raise ValueError("print_interval_s must be positive")
        if not endpoints:
            raise ValueError("at least one teleop endpoint is required")
        self.scene = scene
        self.endpoints = list(endpoints)
        self.loop_hz = None if loop_hz is None else float(loop_hz)
        self.print_interval_s = float(print_interval_s)
        self.recorder = recorder
        self.viewer = viewer

    def run(self) -> None:
        started: list[TeleopEndpoint] = []
        scene_started = False
        last_status_time = 0.0
        confirmation_thread: threading.Thread | None = None
        confirmation_error: BaseException | None = None
        period_s = 0.0 if self.loop_hz is None else 1.0 / self.loop_hz
        try:
            pre_scene_endpoints = [
                endpoint
                for endpoint in self.endpoints
                if bool(getattr(endpoint, "start_before_scene", False))
            ]
            for endpoint in pre_scene_endpoints:
                endpoint.start()
                started.append(endpoint)

            self.scene.start()
            scene_started = True
            for endpoint in self.endpoints:
                if endpoint not in started:
                    endpoint.start()
                    started.append(endpoint)

            gated_endpoint = next(
                (
                    endpoint
                    for endpoint in self.endpoints
                    if bool(getattr(endpoint, "requires_user_confirmation", False))
                ),
                None,
            )

            while True:
                loop_start = time.monotonic()
                sample = self.scene.sample()
                if self.recorder is not None:
                    self.recorder.write(sample)

                if confirmation_error is not None:
                    raise confirmation_error
                if (
                    gated_endpoint is not None
                    and not bool(getattr(gated_endpoint, "teleop_armed", False))
                ):
                    gated_endpoint.consume(sample)
                    if (
                        bool(getattr(gated_endpoint, "input_ready", False))
                        and confirmation_thread is None
                    ):
                        waiter = getattr(
                            gated_endpoint,
                            "wait_for_user_teleop_confirmation",
                            None,
                        )
                        if waiter is None:
                            raise RuntimeError(
                                "Gated teleop endpoint lacks a user confirmation method"
                            )

                        def confirm() -> None:
                            nonlocal confirmation_error
                            try:
                                waiter()
                            except BaseException as exc:  # propagate on the loop thread
                                confirmation_error = exc

                        confirmation_thread = threading.Thread(
                            target=confirm,
                            name="teleop-confirmation",
                            daemon=True,
                        )
                        confirmation_thread.start()
                else:
                    for endpoint in self.endpoints:
                        endpoint.consume(sample)

                if self.viewer is not None and not self.viewer.update(sample):
                    print("[teleop] overlay requested exit")
                    break

                now = time.monotonic()
                if now - last_status_time >= self.print_interval_s:
                    for endpoint in self.endpoints:
                        lines = endpoint.status_lines()
                        print(f"[{endpoint.name}] " + (" | ".join(lines) if lines else "ok"))
                    last_status_time = now

                remaining = period_s - (time.monotonic() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("\n[teleop] interrupted by user")
        finally:
            for endpoint in reversed(started):
                try:
                    endpoint.close()
                except Exception as exc:
                    print(f"[{endpoint.name}] close failed: {exc}")
            if self.viewer is not None:
                self.viewer.close()
            if self.recorder is not None:
                self.recorder.close()
            if scene_started:
                self.scene.close()


__all__ = ["DexSlideTeleopRuntime"]
