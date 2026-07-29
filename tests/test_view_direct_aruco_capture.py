from __future__ import annotations

import numpy as np

import scripts.view_direct_aruco_overlay as overlay
from dexslide.visualization import aruco_overlay
from dexslide.kinematics.transforms import make_transform
from dexslide.world_pose.marker_body_pose_tracker import _smooth_marker_body_pose


def test_default_overlay_result_paths_prefer_latest_dexalign_outputs(tmp_path, monkeypatch) -> None:
    older = tmp_path / "older_session"
    newer = tmp_path / "newer_session"
    older.mkdir()
    newer.mkdir()
    (older / "optimized_skeleton.json").write_text("{}", encoding="utf-8")
    (older / "optimized_marker2hand.json").write_text("{}", encoding="utf-8")
    (older / "optimized_joint_calibration.json").write_text("{}", encoding="utf-8")
    (newer / "optimized_skeleton.json").write_text("{}", encoding="utf-8")
    (newer / "optimized_marker2hand.json").write_text("{}", encoding="utf-8")
    (newer / "optimized_joint_calibration.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(overlay, "DEXALIGN_CALIBRATION_DIR", tmp_path)

    skeleton_default = overlay._default_overlay_skeleton_file()
    marker_default = overlay._default_overlay_marker2hand_file()
    joint_calib_default = overlay._default_overlay_joint_calibration_file()

    assert skeleton_default == newer / "optimized_skeleton.json"
    assert marker_default == newer / "optimized_marker2hand.json"
    assert joint_calib_default == str(newer / "optimized_joint_calibration.json")


def test_use_realsense_backend_decouples_color_backend_from_wrist_align() -> None:
    assert (
        overlay._use_realsense_backend(
            camera_backend="opencv",
            wrist_align_enabled=False,
            use_realsense_rgb=False,
        )
        is False
    )
    assert (
        overlay._use_realsense_backend(
            camera_backend="realsense",
            wrist_align_enabled=False,
            use_realsense_rgb=False,
        )
        is True
    )
    assert (
        overlay._use_realsense_backend(
            camera_backend="opencv",
            wrist_align_enabled=True,
            use_realsense_rgb=False,
        )
        is True
    )
    assert (
        overlay._use_realsense_backend(
            camera_backend="opencv",
            wrist_align_enabled=False,
            use_realsense_rgb=True,
        )
        is True
    )


def test_iter_capture_candidates_for_numeric_source_does_not_scan_linux_devices() -> None:
    assert overlay._iter_capture_candidates("0") == [0]


def test_iter_capture_candidates_for_file_source_does_not_scan_devices() -> None:
    assert overlay._iter_capture_candidates("sample.mp4") == ["sample.mp4"]


def test_open_capture_with_fallback_only_uses_configured_source(monkeypatch, capsys) -> None:
    class FakeCapture:
        def __init__(self, source, backend=None):
            self.source = source
            self.backend = backend
            self.props: dict[int, float] = {}
            self.released = False

        def set(self, prop: int, value: float) -> bool:
            self.props[prop] = value
            return True

        def isOpened(self) -> bool:
            return self.source == 0

        def read(self):
            if self.source == 0:
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(overlay.cv2, "VideoCapture", FakeCapture)

    cap, selected = overlay._open_capture_with_fallback(
        source="0",
        width=640,
        height=480,
        fps=30.0,
        buffer_size=1,
        purpose="test",
    )

    assert selected == 0
    assert isinstance(cap, FakeCapture)
    assert cap.props[overlay.cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert cap.props[overlay.cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert cap.props[overlay.cv2.CAP_PROP_FPS] == 30.0
    assert cap.props[overlay.cv2.CAP_PROP_BUFFERSIZE] == 1
    assert "selected=0" in capsys.readouterr().out


def test_project_points_makes_fisheye_inputs_contiguous(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_project_points(obj, rvec, tvec, k, d):
        captured["obj"] = obj
        captured["rvec"] = rvec
        captured["tvec"] = tvec
        captured["k"] = k
        captured["d"] = d
        return np.zeros((obj.shape[0], 1, 2), dtype=np.float64), None

    monkeypatch.setattr(aruco_overlay.cv2.fisheye, "projectPoints", fake_project_points)

    rvec = np.arange(6, dtype=np.float64)[::2]
    tvec = np.arange(1, 7, dtype=np.float64)[::2]
    object_points = np.arange(18, dtype=np.float64).reshape(2, 3, 3)[:, 0, :]
    intr = {
        "K": np.eye(3, dtype=np.float64),
        "D": np.array([[0.1], [0.01], [0.001], [0.0001]], dtype=np.float64),
    }

    projected = aruco_overlay.project_points(object_points, rvec, tvec, intr)

    assert projected.shape == (2, 2)
    assert captured["obj"].flags["C_CONTIGUOUS"]
    assert captured["rvec"].flags["C_CONTIGUOUS"]
    assert captured["tvec"].flags["C_CONTIGUOUS"]
    assert captured["k"].flags["C_CONTIGUOUS"]
    assert captured["d"].flags["C_CONTIGUOUS"]


def test_smooth_marker_body_pose_blends_position_and_rotation() -> None:
    prev = overlay.CubePoseEstimate(
        transform_table_cube=make_transform(np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 0.4])),
        source_marker_ids=[1, 2],
        max_position_deviation_m=0.001,
    )
    angle = np.deg2rad(30.0)
    rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    curr = overlay.CubePoseEstimate(
        transform_table_cube=make_transform(rot_z, np.array([0.2, -0.1, 0.8])),
        source_marker_ids=[2, 5],
        max_position_deviation_m=0.004,
    )

    smoothed = _smooth_marker_body_pose(prev, curr, alpha=0.25)

    assert smoothed is not None
    np.testing.assert_allclose(smoothed.transform_table_cube[:3, 3], np.array([0.05, -0.025, 0.5]), atol=1e-9)
    assert smoothed.source_marker_ids == [2, 5]
    assert smoothed.max_position_deviation_m == curr.max_position_deviation_m
    assert smoothed.solver_mode == curr.solver_mode
    assert float(smoothed.transform_table_cube[0, 0]) > float(curr.transform_table_cube[0, 0])
