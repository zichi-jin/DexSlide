from __future__ import annotations

from dataclasses import dataclass

from robot_manipulation.teleop.runtime import DexSlideTeleopRuntime


class FakeScene:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.count = 0

    def start(self):
        self.started = True
        return self

    def sample(self):
        self.count += 1
        return self.count

    def close(self):
        self.closed = True


@dataclass
class FakeEndpoint:
    consumed: list[int]
    started: bool = False
    closed: bool = False

    @property
    def name(self) -> str:
        return "fake"

    def start(self) -> None:
        self.started = True

    def consume(self, sample: int) -> None:
        self.consumed.append(sample)
        if sample >= 2:
            raise KeyboardInterrupt

    def status_lines(self) -> list[str]:
        return []

    def close(self) -> None:
        self.closed = True


def test_runtime_closes_scene_and_endpoint_on_interrupt() -> None:
    scene = FakeScene()
    endpoint = FakeEndpoint([])
    DexSlideTeleopRuntime(scene, [endpoint], loop_hz=1000.0).run()
    assert scene.started is True
    assert scene.closed is True
    assert endpoint.started is True
    assert endpoint.closed is True
    assert endpoint.consumed == [1, 2]


class GatedEndpoint:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.start_before_scene = True
        self.requires_user_confirmation = True
        self.teleop_armed = False
        self.input_ready = False
        self.samples: list[int] = []

    @property
    def name(self) -> str:
        return "gated"

    def start(self) -> None:
        self.events.append("gated.start")

    def consume(self, sample: int) -> None:
        self.samples.append(sample)
        self.events.append(f"gated.consume.{sample}")
        self.input_ready = True

    def wait_for_user_teleop_confirmation(self) -> None:
        self.events.append("confirm")
        self.teleop_armed = True

    def status_lines(self) -> list[str]:
        return []

    def close(self) -> None:
        self.events.append("gated.close")


class OrderedScene(FakeScene):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def start(self):
        self.events.append("scene.start")
        return super().start()

    def sample(self):
        sample = super().sample()
        if sample >= 3:
            raise KeyboardInterrupt
        return sample

    def close(self):
        self.events.append("scene.close")
        super().close()


def test_runtime_starts_gated_endpoint_before_scene_and_arms_after_stable_sample() -> None:
    events: list[str] = []
    scene = OrderedScene(events)
    gated = GatedEndpoint(events)

    DexSlideTeleopRuntime(scene, [gated], loop_hz=1000.0).run()

    assert events[:3] == ["gated.start", "scene.start", "gated.consume.1"]
    assert "confirm" in events
    assert events.index("confirm") < events.index("gated.consume.2")
    assert events[-2:] == ["gated.close", "scene.close"]
