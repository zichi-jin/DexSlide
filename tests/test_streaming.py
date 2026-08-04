from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexslide.paths import DEFAULT_DEXSLIDE_STREAMING_FILE
from dexslide.recording import DexSlideDatasetReader, DexSlideRecorder
from dexslide.serial_angles import AngleStreamReader, make_joint_order
from dexslide.streaming import (
    DexSlideScene,
    VisionHandPose,
    VisionSceneFrame,
)
from dexslide.vision.scene_backend import OpenCVSceneVision
from dexslide.visualization import DexSlideARViewer, DexSlidePlot3DViewer


class FakeJointReader:
    def __init__(self, samples: list[tuple[float, np.ndarray]]) -> None:
        self.samples = [(float(ts), np.asarray(values, dtype=np.float64)) for ts, values in samples]
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def wait_for_first_sample(self, timeout_sec: float) -> None:
        return None

    def snapshot_nearest_rad20(self, timestamp: float):
        if not self.samples:
            return np.zeros(20, dtype=np.float64), 0.0, ""
        sample_time, values = min(self.samples, key=lambda item: abs(item[0] - timestamp))
        return values.copy(), sample_time, "fake"


class FakeVision:
    def __init__(self, frames: list[VisionSceneFrame]) -> None:
        self.frames = frames
        self.index = 0
        self.started = False
        self.intrinsics = {
            "K": np.array([[500.0, 0.0, 4.0], [0.0, 500.0, 3.0], [0.0, 0.0, 1.0]]),
            "D": np.zeros(5),
        }

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read(self) -> VisionSceneFrame:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame


def _vision_frame(
    timestamp: float,
    *,
    table_valid: bool = True,
    hand_ids: tuple[str, ...] = ("left",),
) -> VisionSceneFrame:
    camera_t_table = np.eye(4, dtype=np.float64)
    camera_t_table[:3, 3] = [0.1, 0.2, 0.8]
    transform_table_hand = np.eye(4, dtype=np.float64)
    transform_table_hand[:3, 3] = [0.3, -0.1, 0.2]
    return VisionSceneFrame(
        timestamp=float(timestamp),
        frame_bgr=np.full((6, 8, 3), 127, dtype=np.uint8),
        image_size=(8, 6),
        camera_T_table=(
            camera_t_table
            if table_valid
            else np.full((4, 4), np.nan, dtype=np.float64)
        ),
        table_valid=table_valid,
        table_marker_corners_px=(
            np.array([[1, 1], [6, 1], [6, 5], [1, 5]], dtype=np.float64)
            if table_valid
            else None
        ),
        hands={
            hand_id: VisionHandPose(
                transform_table_hand=transform_table_hand,
                valid=True,
                marker_ids=(1, 2),
                reprojection_error_px=0.75,
            )
            for hand_id in hand_ids
        },
    )


def _default_scene(
    *,
    reader: FakeJointReader,
    vision: FakeVision,
    joint_unit: str | None = None,
    joint_mode: str | None = None,
    pose_filter_enabled: bool | None = None,
) -> DexSlideScene:
    return DexSlideScene.from_file(
        DEFAULT_DEXSLIDE_STREAMING_FILE,
        joint_unit=joint_unit,
        joint_mode=joint_mode,
        pose_filter_enabled=pose_filter_enabled,
        joint_readers={"left": reader},
        vision_backend=vision,
    )


def test_angle_stream_reader_selects_nearest_buffered_sample() -> None:
    joint_order = make_joint_order()
    reader = AngleStreamReader(
        port="/dev/null",
        baud=115200,
        mode="angles",
        joint_order=joint_order,
        calibration={
            str(item["id"]): {"offset": 0.0, "angle0": 0.0, "rate": 1.0}
            for item in joint_order
        },
        buffer_size=2,
    )
    with reader.lock:
        reader.latest_deg = {key: 10.0 for key in reader.latest_deg}
        reader._publish_sample_locked("first", timestamp=1.0)
        reader.latest_deg = {key: 20.0 for key in reader.latest_deg}
        reader._publish_sample_locked("second", timestamp=3.0)

    values, timestamp, raw_line = reader.snapshot_nearest_rad20(2.8)

    np.testing.assert_allclose(values, np.deg2rad(np.full(20, 20.0)))
    assert timestamp == 3.0
    assert raw_line == "second"


def test_scene_defaults_to_degrees_and_emits_raw_and_dexalign_values() -> None:
    raw_rad = np.linspace(-0.2, 0.3, 20)
    scene = _default_scene(
        reader=FakeJointReader([(10.0, raw_rad)]),
        vision=FakeVision([_vision_frame(10.05)]),
    )
    with scene:
        sample = scene.sample()

    hand = sample.hands["left"]
    calibration = scene.effective_calibration["left"]
    scale = np.asarray(calibration["effective_joint_scale"])
    bias_rad = np.asarray(calibration["effective_joint_bias_rad"])
    np.testing.assert_allclose(hand.joint_angles_raw, np.rad2deg(raw_rad))
    np.testing.assert_allclose(
        hand.joint_angles_dexalign,
        np.rad2deg(scale * raw_rad + bias_rad),
    )
    np.testing.assert_allclose(hand.joint_angles, hand.joint_angles_dexalign)
    assert sample.units["joint_angles"] == "deg"
    assert json.loads(sample.to_json())["units"]["translation"] == "m"
    assert hand.valid is True


def test_scene_pose_filter_config_and_api_override() -> None:
    default_scene = _default_scene(
        reader=FakeJointReader([]),
        vision=FakeVision([_vision_frame(1.0)]),
    )
    assert default_scene.pose_filter_enabled is True

    payload = json.loads(DEFAULT_DEXSLIDE_STREAMING_FILE.read_text(encoding="utf-8"))
    payload["stream"]["pose_filter_enabled"] = False
    scene = DexSlideScene(
        payload,
        config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
        joint_readers={"left": FakeJointReader([])},
        vision_backend=FakeVision([_vision_frame(1.0)]),
    )
    assert scene.pose_filter_enabled is False
    assert scene.config["stream"]["pose_filter_enabled"] is False

    enabled_override = DexSlideScene(
        payload,
        config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
        pose_filter_enabled=True,
        joint_readers={"left": FakeJointReader([])},
        vision_backend=FakeVision([_vision_frame(1.0)]),
    )
    assert enabled_override.pose_filter_enabled is True
    assert enabled_override.config["stream"]["pose_filter_enabled"] is True

    overridden = _default_scene(
        reader=FakeJointReader([]),
        vision=FakeVision([_vision_frame(1.0)]),
        pose_filter_enabled=False,
    )
    assert overridden.pose_filter_enabled is False
    assert overridden.config["stream"]["pose_filter_enabled"] is False

    payload["stream"].pop("pose_filter_enabled")
    backward_compatible = DexSlideScene(
        payload,
        config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
        joint_readers={"left": FakeJointReader([])},
        vision_backend=FakeVision([_vision_frame(1.0)]),
    )
    assert backward_compatible.pose_filter_enabled is True


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_scene_rejects_non_boolean_pose_filter_config(invalid: object) -> None:
    payload = json.loads(DEFAULT_DEXSLIDE_STREAMING_FILE.read_text(encoding="utf-8"))
    payload["stream"]["pose_filter_enabled"] = invalid
    with pytest.raises(ValueError, match="stream.pose_filter_enabled must be a boolean"):
        DexSlideScene(
            payload,
            config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
            joint_readers={"left": FakeJointReader([])},
            vision_backend=FakeVision([_vision_frame(1.0)]),
        )


def test_scene_rejects_non_boolean_pose_filter_override() -> None:
    with pytest.raises(ValueError, match="pose_filter_enabled override"):
        DexSlideScene.from_file(
            DEFAULT_DEXSLIDE_STREAMING_FILE,
            pose_filter_enabled="false",  # type: ignore[arg-type]
            joint_readers={"left": FakeJointReader([])},
            vision_backend=FakeVision([_vision_frame(1.0)]),
        )


def test_scene_backend_master_switch_disables_all_smoothing_state() -> None:
    runtime = SimpleNamespace(marker_config=object())
    vision = OpenCVSceneVision(
        camera=object(),  # type: ignore[arg-type]
        intrinsics_file=Path("intrinsics.json"),
        table_aruco_file=Path("table.yaml"),
        table_marker_id=0,
        hands={"left": runtime},
        num_workers=1,
        pose_solver="joint_pnp",
        smoothing_alpha=0.35,
        pose_filter_enabled=False,
        outlier_threshold_m=0.02,
        reprojection_error_threshold_px=5.0,
        detection_scale=1.0,
    )

    assert vision._pose_filter_enabled is False
    assert vision._pose_trackers["left"]._smoothing_alpha == 1.0
    assert vision._hand_pose_filters == {}
    assert vision._table_pose_filter is None


def test_scene_raw_radians_and_stale_joint_semantics() -> None:
    raw_rad = np.linspace(0.0, 0.19, 20)
    scene = _default_scene(
        reader=FakeJointReader([(1.0, raw_rad)]),
        vision=FakeVision([_vision_frame(2.0)]),
        joint_unit="rad",
        joint_mode="raw",
    )
    with scene:
        sample = scene.sample()

    hand = sample.hands["left"]
    np.testing.assert_allclose(hand.joint_angles, raw_rad)
    assert hand.joint_unit == "rad"
    assert hand.joint_age_sec == pytest.approx(1.0)
    assert hand.joints_valid is False
    assert hand.pose_valid is True
    assert hand.valid is False


def test_table_loss_invalidates_all_hand_poses_without_camera_fallback() -> None:
    scene = _default_scene(
        reader=FakeJointReader([(5.0, np.zeros(20))]),
        vision=FakeVision([_vision_frame(5.0, table_valid=False)]),
    )
    with scene:
        sample = scene.sample()
        hand = sample.hands["left"]

    assert hand.pose_valid is False
    assert np.isnan(hand.transform_table_hand).all()
    payload = json.loads(sample.to_json())
    assert payload["camera_T_table"][0][0] is None


def test_dexalign_mode_rejects_missing_joint_calibration() -> None:
    payload = json.loads(DEFAULT_DEXSLIDE_STREAMING_FILE.read_text(encoding="utf-8"))
    payload["hands"]["left"].pop("joint_calibration_file")

    with pytest.raises(ValueError, match="required for joint_mode='dexalign'"):
        DexSlideScene(
            payload,
            config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
            joint_readers={"left": FakeJointReader([])},
            vision_backend=FakeVision([_vision_frame(1.0)]),
        )


def test_same_dictionary_marker_overlap_across_hands_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DEXSLIDE_STREAMING_FILE.read_text(encoding="utf-8"))
    communications_path = tmp_path / "communications.json"
    communications = json.loads(
        (DEFAULT_DEXSLIDE_STREAMING_FILE.parent / "dexslide_communications.json").read_text(
            encoding="utf-8"
        )
    )
    communications["right"]["joints"] = copy.deepcopy(communications["left"]["joints"])
    communications_path.write_text(json.dumps(communications), encoding="utf-8")
    payload["communications_file"] = str(communications_path)
    payload["hands"]["right"] = copy.deepcopy(payload["hands"]["left"])
    payload["hands"]["right"]["communication_hand"] = "right"

    with pytest.raises(ValueError, match="cannot share marker IDs"):
        DexSlideScene(payload, config_path=DEFAULT_DEXSLIDE_STREAMING_FILE)


def test_table_marker_id_cannot_overlap_a_hand_marker() -> None:
    payload = json.loads(DEFAULT_DEXSLIDE_STREAMING_FILE.read_text(encoding="utf-8"))
    payload["world"]["table_marker_id"] = 1

    with pytest.raises(ValueError, match="table marker ID"):
        DexSlideScene(
            payload,
            config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
            joint_readers={"left": FakeJointReader([])},
            vision_backend=FakeVision([_vision_frame(1.0)]),
        )


def _two_hand_config(tmp_path: Path) -> dict:
    payload = json.loads(DEFAULT_DEXSLIDE_STREAMING_FILE.read_text(encoding="utf-8"))
    communications_path = tmp_path / "communications.json"
    communications = json.loads(
        (DEFAULT_DEXSLIDE_STREAMING_FILE.parent / "dexslide_communications.json").read_text(
            encoding="utf-8"
        )
    )
    communications["right"]["joints"] = copy.deepcopy(communications["left"]["joints"])
    communications_path.write_text(json.dumps(communications), encoding="utf-8")
    payload["communications_file"] = str(communications_path)

    source_session = (
        DEFAULT_DEXSLIDE_STREAMING_FILE.parent / "calibration/dexalign/test_left_001"
    )
    right_session = tmp_path / "right_session"
    right_session.mkdir()
    for filename in (
        "optimized_skeleton.json",
        "optimized_marker2hand.json",
        "optimized_joint_calibration.json",
    ):
        shutil.copy2(source_session / filename, right_session / filename)
    tags = json.loads(
        (
            DEFAULT_DEXSLIDE_STREAMING_FILE.parent
            / "calibration/direct_aruco/left_tags2marker.json"
        ).read_text(encoding="utf-8")
    )
    tags["hand"] = "right"
    tags["aruco_dict"]["predefined"] = "DICT_5X5_1000"
    tags["marker2wrist_path"] = str(right_session / "optimized_marker2hand.json")
    tags_path = tmp_path / "right_tags2marker.json"
    tags_path.write_text(json.dumps(tags), encoding="utf-8")
    payload["hands"]["right"] = {
        "communication_hand": "right",
        "skeleton_file": str(right_session / "optimized_skeleton.json"),
        "glove_calibration_file": str(
            DEFAULT_DEXSLIDE_STREAMING_FILE.parent / "calibration/glove_calibration.json"
        ),
        "tags_to_marker_file": str(tags_path),
        "marker_to_hand_file": str(right_session / "optimized_marker2hand.json"),
        "joint_calibration_file": str(right_session / "optimized_joint_calibration.json"),
        "dexalign_session": str(right_session),
    }
    return payload


def test_two_hands_match_joints_independently_at_different_rates(tmp_path: Path) -> None:
    payload = _two_hand_config(tmp_path)
    scene = DexSlideScene(
        payload,
        config_path=DEFAULT_DEXSLIDE_STREAMING_FILE,
        joint_readers={
            "left": FakeJointReader([(10.0, np.ones(20))]),
            "right": FakeJointReader([(8.0, np.full(20, 2.0))]),
        },
        vision_backend=FakeVision([_vision_frame(10.1, hand_ids=("left", "right"))]),
    )
    with scene:
        sample = scene.sample()

    assert sample.hands["left"].joints_valid is True
    assert sample.hands["right"].joints_valid is False
    assert sample.hands["left"].joint_timestamp == 10.0
    assert sample.hands["right"].joint_timestamp == 8.0


def test_recorder_snapshots_configs_chunks_and_first_valid_frame(tmp_path: Path) -> None:
    scene = _default_scene(
        reader=FakeJointReader([(20.0, np.linspace(0.0, 0.2, 20))]),
        vision=FakeVision([_vision_frame(20.0), _vision_frame(20.1), _vision_frame(20.2)]),
    )
    with scene:
        with DexSlideRecorder(tmp_path, scene, session_id="demo", chunk_size=2) as recorder:
            samples = [scene.sample(), scene.sample(), scene.sample()]
            for sample in samples:
                recorder.write(sample)

    session_dir = tmp_path / "demo"
    meta = json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["sample_count"] == 3
    assert [chunk["sample_count"] for chunk in meta["chunks"]] == [2, 1]
    assert meta["first_valid_frame"]["valid_hands"] == ["left"]
    assert (session_dir / "first_valid_frame.jpg").is_file()
    for provenance in meta["configs"].values():
        copied = session_dir / provenance["session_copy"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == provenance["sha256"]

    loaded = list(DexSlideDatasetReader(session_dir).iter_samples())
    assert len(loaded) == 3
    np.testing.assert_allclose(
        loaded[-1].hands["left"].joint_angles,
        samples[-1].hands["left"].joint_angles,
    )
    assert loaded[0].units == samples[0].units


def test_recorder_rejects_unit_changes_and_omits_invalid_photo(tmp_path: Path) -> None:
    scene = _default_scene(
        reader=FakeJointReader([]),
        vision=FakeVision([_vision_frame(30.0, table_valid=False)]),
    )
    with scene:
        sample = scene.sample()
        with DexSlideRecorder(tmp_path, scene, session_id="invalid", chunk_size=5) as recorder:
            recorder.write(sample)
            with pytest.raises(ValueError, match="units cannot change"):
                recorder.write(replace(sample, joint_unit="rad"))

    session_dir = tmp_path / "invalid"
    meta = json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["first_valid_frame"] is None
    assert not (session_dir / "first_valid_frame.jpg").exists()


def test_ar_viewer_does_not_mutate_scene_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = _default_scene(
        reader=FakeJointReader([(40.0, np.zeros(20))]),
        vision=FakeVision([_vision_frame(40.0)]),
    )
    with scene:
        sample = scene.sample()
        before = sample.to_json()
        monkeypatch.setattr("cv2.imshow", lambda *args, **kwargs: None)
        monkeypatch.setattr("cv2.waitKey", lambda *args, **kwargs: -1)
        monkeypatch.setattr("cv2.getWindowProperty", lambda *args, **kwargs: 1.0)
        viewer = DexSlideARViewer(scene)
        assert viewer.update(sample) is True
        viewer.closed = True

    assert sample.to_json() == before


def test_plot3d_viewer_uses_scene_pose_without_mutating_sample() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    scene = _default_scene(
        reader=FakeJointReader([(41.0, np.zeros(20))]),
        vision=FakeVision([_vision_frame(41.0)]),
    )
    with scene:
        sample = scene.sample()
        before = sample.to_json()
        viewer = DexSlidePlot3DViewer(scene, max_refresh_hz=0.0)
        assert viewer.update(sample) is True
        assert "left" in viewer._artists
        assert all(line.get_visible() for line in viewer._artists["left"].wrist_axis_lines)
        viewer.close()

    assert sample.to_json() == before
