from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

import numpy as np

from dexslide.calibration.dexalign.collect_alignment_dataset import (
    CollectionStats,
    FrameCollectionResult,
    KeypointDeprojection,
    MarkerPoseObservation,
    RuntimeFrame,
    _apply_manual_capture_gate,
    _build_parser,
    collect_alignment_dataset,
)
from dexslide.calibration.dexalign.io_utils import load_alignment_dataset, save_alignment_dataset
from dexslide.calibration.dexalign.types import AlignmentDataset, AlignmentFrame
from dexslide.world_pose.hand_cube_overlay import make_transform


@dataclass
class _FakeDetector:
    detections: list[tuple[np.ndarray, np.ndarray] | None]
    index: int = 0

    def detect(self, _color_bgr: np.ndarray):
        detection = self.detections[self.index]
        self.index += 1
        return detection


@dataclass
class _FakeGloveReader:
    joint_vectors: list[np.ndarray]
    index: int = 0
    raw_line: str = "fake"
    sample_age_seconds: float = 0.0

    def snapshot_rad20(self):
        value = self.joint_vectors[self.index]
        self.index += 1
        return value, float(self.index), self.raw_line

    def sample_age_sec(self) -> float:
        return float(self.sample_age_seconds)


def _make_pose(translation_mm: np.ndarray) -> MarkerPoseObservation:
    return MarkerPoseObservation(
        camera_T_marker_mm=make_transform(np.eye(3, dtype=np.float64), np.asarray(translation_mm, dtype=np.float64)),
        marker_ids_used=(1, 2),
        marker_reproj_error_px=1.5,
    )


def _make_candidate_frame(timestamp: float = 0.0) -> AlignmentFrame:
    return AlignmentFrame(
        timestamp=float(timestamp),
        camera_T_marker=np.eye(4, dtype=np.float64),
        q_encoder_rad20=np.zeros(20, dtype=np.float64),
        keypoints_camera_mm=np.zeros((21, 3), dtype=np.float64),
        keypoint_confidence=np.ones(21, dtype=np.float64),
        keypoint_valid_mask=np.ones(21, dtype=bool),
    )


def test_manual_capture_gate_only_keeps_frames_when_enabled() -> None:
    frame_result = FrameCollectionResult(
        detection=None,
        marker_pose=None,
        deprojection=None,
        q_encoder_rad20=np.zeros(20, dtype=np.float64),
        candidate_frame=_make_candidate_frame(),
        skip_reason=None,
    )
    paused_stats = CollectionStats()
    paused_kept_frames: list[AlignmentFrame] = []

    paused_decision = _apply_manual_capture_gate(
        frame_result=frame_result,
        capture_enabled=False,
        kept_frames=paused_kept_frames,
        stats=paused_stats,
    )

    assert paused_decision == "paused:ready"
    assert paused_stats.frames_kept == 0
    assert paused_stats.skip_counts["capture_paused"] == 1
    assert paused_kept_frames == []

    active_stats = CollectionStats()
    active_kept_frames: list[AlignmentFrame] = []
    active_decision = _apply_manual_capture_gate(
        frame_result=frame_result,
        capture_enabled=True,
        kept_frames=active_kept_frames,
        stats=active_stats,
    )

    assert active_decision == "kept"
    assert active_stats.frames_kept == 1
    assert len(active_kept_frames) == 1


def test_collect_alignment_dataset_writes_dataset_without_motion_filter(tmp_path) -> None:
    keypoints_2d = np.stack([np.array([10.0 + idx, 20.0 + idx], dtype=np.float64) for idx in range(21)], axis=0)
    confidence = np.ones(21, dtype=np.float64)
    valid_mask = np.zeros(21, dtype=bool)
    valid_mask[:12] = True
    points_mm = np.full((21, 3), np.nan, dtype=np.float64)
    points_mm[:12] = np.stack(
        [np.array([float(idx), float(idx + 1), 500.0 + float(idx)], dtype=np.float64) for idx in range(12)],
        axis=0,
    )
    depth_mm = np.where(valid_mask, 500.0, np.nan)
    frames = [
        RuntimeFrame(timestamp=0.0, color_bgr=np.zeros((8, 8, 3), dtype=np.uint8), depth_frame=None),
        RuntimeFrame(timestamp=1.0, color_bgr=np.zeros((8, 8, 3), dtype=np.uint8), depth_frame=None),
        RuntimeFrame(timestamp=2.0, color_bgr=np.zeros((8, 8, 3), dtype=np.uint8), depth_frame=None),
    ]
    detections = [(keypoints_2d, confidence), (keypoints_2d, confidence), (keypoints_2d, confidence)]
    poses = [_make_pose(np.array([0.0, 0.0, 500.0])), _make_pose(np.array([0.0, 0.0, 500.0])), _make_pose(np.array([0.0, 0.0, 510.0]))]
    q0 = np.zeros(20, dtype=np.float64)
    q2 = q0.copy()
    q2[3] = np.deg2rad(12.0)

    def marker_pose_estimator(frame: RuntimeFrame, _previous_pose: np.ndarray | None):
        return poses[int(frame.timestamp)]

    def deproject(_frame: RuntimeFrame, _keypoints: np.ndarray) -> KeypointDeprojection:
        return KeypointDeprojection(keypoints_camera_mm=points_mm, valid_mask=valid_mask, depth_mm=depth_mm)

    dataset, stats = collect_alignment_dataset(
        frames=frames,
        detector=_FakeDetector(detections),
        marker_pose_estimator=marker_pose_estimator,
        glove_reader=_FakeGloveReader([q0, q0, q2]),
        deproject_keypoints=deproject,
        hand="left",
        capture_kind="s2",
        max_kept_frames=None,
    )

    assert dataset.num_frames == 3
    assert dataset.capture_kind == "s2"
    assert "insufficient_motion" not in stats.skip_counts

    dataset_path, meta_path = save_alignment_dataset(tmp_path, dataset)
    loaded = load_alignment_dataset(dataset_path, meta_path)

    assert loaded.num_frames == 3
    assert dataset_path.name == "dataset_s2.npz"
    np.testing.assert_allclose(loaded.frames[2].q_encoder_rad20, q2, atol=1e-9)


def test_collect_alignment_dataset_applies_required_filters() -> None:
    keypoints_2d = np.stack([np.array([10.0 + idx, 20.0 + idx], dtype=np.float64) for idx in range(21)], axis=0)
    confidence = np.ones(21, dtype=np.float64)
    frames = [
        RuntimeFrame(timestamp=0.0, color_bgr=np.zeros((4, 4, 3), dtype=np.uint8), depth_frame=None),
        RuntimeFrame(timestamp=1.0, color_bgr=np.zeros((4, 4, 3), dtype=np.uint8), depth_frame=None),
        RuntimeFrame(timestamp=2.0, color_bgr=np.zeros((4, 4, 3), dtype=np.uint8), depth_frame=None),
        RuntimeFrame(timestamp=3.0, color_bgr=np.zeros((4, 4, 3), dtype=np.uint8), depth_frame=None),
    ]
    valid_11 = np.zeros(21, dtype=bool)
    valid_11[:11] = True
    valid_missing_anchor = np.zeros(21, dtype=bool)
    valid_missing_anchor[:12] = True
    valid_missing_anchor[9] = False
    valid_missing_anchor[12] = True
    valid_ok = np.zeros(21, dtype=bool)
    valid_ok[:12] = True
    points_template = np.full((21, 3), np.nan, dtype=np.float64)
    depth_template = np.full(21, np.nan, dtype=np.float64)

    def _projection(mask: np.ndarray) -> KeypointDeprojection:
        points = points_template.copy()
        depths = depth_template.copy()
        for idx in np.flatnonzero(mask):
            points[idx] = np.array([float(idx), float(idx), 450.0], dtype=np.float64)
            depths[idx] = 450.0
        return KeypointDeprojection(points, mask, depths)

    projections = [None, _projection(valid_11), _projection(valid_missing_anchor), _projection(valid_ok)]

    def marker_pose_estimator(frame: RuntimeFrame, _previous_pose: np.ndarray | None):
        if int(frame.timestamp) == 0:
            return None
        return _make_pose(np.array([0.0, 0.0, 500.0 + 10.0 * frame.timestamp], dtype=np.float64))

    def deproject(frame: RuntimeFrame, _keypoints: np.ndarray) -> KeypointDeprojection:
        return projections[int(frame.timestamp)]

    dataset, stats = collect_alignment_dataset(
        frames=frames,
        detector=_FakeDetector([(keypoints_2d, confidence)] * 4),
        marker_pose_estimator=marker_pose_estimator,
        glove_reader=_FakeGloveReader([np.zeros(20, dtype=np.float64)] * 3),
        deproject_keypoints=deproject,
        hand="left",
        capture_kind="s2",
        max_kept_frames=None,
    )

    assert dataset.num_frames == 1
    assert stats.skip_counts["marker_pose_unavailable"] == 1
    assert stats.skip_counts["too_few_valid_keypoints"] == 1
    assert stats.skip_counts["missing_anchor_keypoints"] == 1




def test_collect_alignment_dataset_supports_s1_without_glove_reader(tmp_path) -> None:
    keypoints_2d = np.stack([np.array([15.0 + idx, 30.0 + idx], dtype=np.float64) for idx in range(21)], axis=0)
    confidence = np.ones(21, dtype=np.float64)
    valid_mask = np.zeros(21, dtype=bool)
    valid_mask[:12] = True
    points_mm = np.full((21, 3), np.nan, dtype=np.float64)
    for idx in np.flatnonzero(valid_mask):
        points_mm[idx] = np.array([float(idx), float(idx + 2), 600.0 + float(idx)], dtype=np.float64)
    depth_mm = np.where(valid_mask, 600.0, np.nan)
    frames = [
        RuntimeFrame(timestamp=0.0, color_bgr=np.zeros((8, 8, 3), dtype=np.uint8), depth_frame=None),
        RuntimeFrame(timestamp=1.0, color_bgr=np.zeros((8, 8, 3), dtype=np.uint8), depth_frame=None),
    ]

    def marker_pose_estimator(frame: RuntimeFrame, _previous_pose: np.ndarray | None):
        return _make_pose(np.array([0.0, 0.0, 500.0 + 5.0 * frame.timestamp], dtype=np.float64))

    def deproject(_frame: RuntimeFrame, _keypoints: np.ndarray) -> KeypointDeprojection:
        return KeypointDeprojection(keypoints_camera_mm=points_mm, valid_mask=valid_mask, depth_mm=depth_mm)

    dataset, _stats = collect_alignment_dataset(
        frames=frames,
        detector=_FakeDetector([(keypoints_2d, confidence)] * 2),
        marker_pose_estimator=marker_pose_estimator,
        glove_reader=None,
        deproject_keypoints=deproject,
        hand="left",
        capture_kind="s1",
        max_kept_frames=None,
    )

    assert dataset.capture_kind == "s1"
    assert dataset.num_frames == 2
    assert not dataset.has_finite_glove_angles()

    dataset_path, meta_path = save_alignment_dataset(tmp_path, dataset)
    loaded = load_alignment_dataset(dataset_path, meta_path)

    assert dataset_path.name == "dataset_s1.npz"
    assert loaded.capture_kind == "s1"
    assert not loaded.has_finite_glove_angles()
    assert np.isnan(loaded.frames[0].q_encoder_rad20).all()


def test_collect_alignment_dataset_rejects_empty_glove_raw_line() -> None:
    keypoints_2d = np.zeros((21, 2), dtype=np.float64)
    confidence = np.ones(21, dtype=np.float64)
    valid_mask = np.ones(21, dtype=bool)
    points_mm = np.zeros((21, 3), dtype=np.float64)
    depth_mm = np.full(21, 500.0, dtype=np.float64)

    dataset, stats = collect_alignment_dataset(
        frames=[
            RuntimeFrame(
                timestamp=1.0,
                color_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
                depth_frame=None,
            )
        ],
        detector=_FakeDetector([(keypoints_2d, confidence)]),
        marker_pose_estimator=lambda _frame, _previous: _make_pose(
            np.array([0.0, 0.0, 500.0], dtype=np.float64)
        ),
        glove_reader=_FakeGloveReader([np.zeros(20, dtype=np.float64)], raw_line=""),
        deproject_keypoints=lambda _frame, _points: KeypointDeprojection(
            points_mm,
            valid_mask,
            depth_mm,
        ),
        hand="left",
        capture_kind="s2",
    )

    assert dataset.num_frames == 0
    assert stats.skip_counts["glove_sample_unavailable"] == 1


def test_collect_alignment_dataset_rejects_stale_glove_sample() -> None:
    keypoints_2d = np.zeros((21, 2), dtype=np.float64)
    confidence = np.ones(21, dtype=np.float64)
    valid_mask = np.ones(21, dtype=bool)
    points_mm = np.zeros((21, 3), dtype=np.float64)
    depth_mm = np.full(21, 500.0, dtype=np.float64)

    dataset, stats = collect_alignment_dataset(
        frames=[
            RuntimeFrame(
                timestamp=1.0,
                color_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
                depth_frame=None,
            )
        ],
        detector=_FakeDetector([(keypoints_2d, confidence)]),
        marker_pose_estimator=lambda _frame, _previous: _make_pose(
            np.array([0.0, 0.0, 500.0], dtype=np.float64)
        ),
        glove_reader=_FakeGloveReader(
            [np.zeros(20, dtype=np.float64)],
            sample_age_seconds=1.0,
        ),
        deproject_keypoints=lambda _frame, _points: KeypointDeprojection(
            points_mm,
            valid_mask,
            depth_mm,
        ),
        hand="left",
        capture_kind="s2",
        max_glove_sample_age_sec=0.5,
    )

    assert dataset.num_frames == 0
    assert stats.skip_counts["glove_sample_stale"] == 1


def test_load_alignment_dataset_rejects_npz_meta_frame_count_mismatch(tmp_path) -> None:
    dataset = AlignmentDataset(
        hand="left",
        capture_kind="s1",
        frames=(_make_candidate_frame(0.0), _make_candidate_frame(1.0)),
    )
    dataset_path, meta_path = save_alignment_dataset(tmp_path, dataset)
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_payload["frame_count"] = 99
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="meta records"):
        load_alignment_dataset(dataset_path, meta_path)


def test_load_alignment_dataset_rejects_npz_meta_frame_count_mismatch(tmp_path) -> None:
    dataset = AlignmentDataset(
        hand="left",
        capture_kind="s1",
        frames=(_make_candidate_frame(timestamp=1.0), _make_candidate_frame(timestamp=2.0)),
    )
    dataset_path, meta_path = save_alignment_dataset(tmp_path, dataset)
    broken_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    broken_meta["frame_count"] = 99
    meta_path.write_text(json.dumps(broken_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="meta records"):
        load_alignment_dataset(dataset_path, meta_path)
