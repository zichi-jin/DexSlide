from __future__ import annotations

import numpy as np
import pytest

from dexslide.retargeting.live_bridge import (
    apply_glove_joint_overrides,
    format_named_values,
    joint_vector_to_map,
)


def test_joint_vector_to_map_uses_joint_order() -> None:
    joint_map = joint_vector_to_map(np.array([0.1, -0.2, 0.3], dtype=np.float64), ["a", "b", "c"])
    assert joint_map == {"a": 0.1, "b": -0.2, "c": 0.3}


def test_apply_glove_joint_overrides_overrides_requested_joints() -> None:
    joint_map = apply_glove_joint_overrides(
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
        ["thumb.DIP", "middle.DIP", "index.MCP_back"],
        overrides_rad={"middle.DIP": 0.0, "index.MCP_back": 0.5},
    )
    assert joint_map == {"thumb.DIP": 0.1, "middle.DIP": 0.0, "index.MCP_back": 0.5}


def test_format_named_values_includes_unit_and_name() -> None:
    text = format_named_values(["joint_a", "joint_b"], np.array([12.345, -6.78]), unit="deg")
    assert "joint_a=" in text
    assert "joint_b=" in text
    assert "deg" in text


def test_format_named_values_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected 2 joint values"):
        format_named_values(["joint_a", "joint_b"], np.array([1.0]), unit="deg")
