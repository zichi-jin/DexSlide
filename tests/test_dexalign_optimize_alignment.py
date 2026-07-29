from __future__ import annotations

import json
import pytest

import numpy as np

from dexslide.calibration.dexalign.io_utils import load_marker2hand_asset_mm, load_runtime_skeleton
from dexslide.calibration.dexalign.io_utils import save_alignment_dataset
from dexslide.calibration.dexalign.optimize_alignment import _resolve_dataset_pair, run_alignment_optimization
from dexslide.calibration.dexalign.pipeline_v2 import (
    _build_frame_pool,
    _make_runtime_shaped_skeleton,
    _palm_radii_and_directions,
    _runtime_landmarks,
    run_step3_lengths_and_translation,
)
from dexslide.calibration.dexalign.types import AlignmentDataset, AlignmentFrame
from dexslide.kinematics.transforms import make_transform, transform_points
from dexslide.paths import DEFAULT_SKELETON_FILE


def _rot_x(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rot_z(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    return value / np.linalg.norm(value)


def _make_true_skeleton(initial_skeleton: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    palm_radii0, base_dirs0 = _palm_radii_and_directions(initial_skeleton, "left")
    dirs_true = np.stack(
        [
            _unit(base_dirs0[0] + np.array([0.02, -0.03, 0.04], dtype=np.float64)),
            _unit(base_dirs0[1] + np.array([0.03, 0.01, 0.02], dtype=np.float64)),
            _unit(base_dirs0[2] + np.array([-0.02, 0.01, 0.03], dtype=np.float64)),
            _unit(base_dirs0[3] + np.array([-0.01, -0.02, 0.02], dtype=np.float64)),
            _unit(base_dirs0[4] + np.array([0.01, -0.01, 0.01], dtype=np.float64)),
        ],
        axis=0,
    )
    palm_radii_true = palm_radii0 + np.array([2.5, -1.0, 2.0, 1.2, -0.8], dtype=np.float64)
    finger_lengths_true = np.asarray(
        [
            49.0,
            28.5,
            20.3,
            36.0,
            23.2,
            18.4,
            38.8,
            25.7,
            19.7,
            35.9,
            23.9,
            18.5,
            29.5,
            18.8,
            16.3,
        ],
        dtype=np.float64,
    )
    true_skeleton = _make_runtime_shaped_skeleton(
        initial_skeleton,
        hand="left",
        base_radii_mm=palm_radii_true,
        base_directions=dirs_true,
        finger_lengths_mm=finger_lengths_true,
    )
    return true_skeleton, palm_radii_true, finger_lengths_true


def _make_s1_dataset(true_skeleton: dict, marker2hand_true: np.ndarray) -> AlignmentDataset:
    frames: list[AlignmentFrame] = []
    for idx in range(10):
        camera_T_marker = make_transform(
            _rot_z(-0.04 * idx) @ _rot_x(0.015 * idx),
            np.array([10.0 * idx, -3.0 * idx, 620.0 + 2.0 * idx], dtype=np.float64),
        )
        hand_points = _runtime_landmarks(true_skeleton, np.zeros(20, dtype=np.float64), "left")
        observed = transform_points(camera_T_marker @ marker2hand_true, hand_points)
        frames.append(
            AlignmentFrame(
                timestamp=float(idx),
                camera_T_marker=camera_T_marker,
                q_encoder_rad20=np.full(20, np.nan, dtype=np.float64),
                keypoints_camera_mm=observed,
                keypoint_confidence=np.ones(21, dtype=np.float64),
                keypoint_valid_mask=np.ones(21, dtype=bool),
            )
        )
    return AlignmentDataset(hand="left", frames=tuple(frames), capture_kind="s1")


def _make_s2_dataset(
    true_skeleton: dict,
    marker2hand_true: np.ndarray,
    joint_scale_true: np.ndarray,
    joint_bias_true: np.ndarray,
) -> AlignmentDataset:
    frames: list[AlignmentFrame] = []
    for idx in range(24):
        q_raw = np.zeros(20, dtype=np.float64)
        phase = float(idx) / 23.0
        for joint_idx in range(20):
            sign = -1.0 if joint_idx % 2 else 1.0
            q_raw[joint_idx] = np.deg2rad(sign * (5.0 + 18.0 * phase + 0.35 * joint_idx))
        q_true = joint_scale_true * q_raw + joint_bias_true
        hand_points = _runtime_landmarks(true_skeleton, q_true, "left")
        camera_T_marker = make_transform(
            _rot_z(-0.02 * idx) @ _rot_x(0.01 * idx),
            np.array([6.0 * idx, -2.5 * idx, 610.0 + 1.5 * idx], dtype=np.float64),
        )
        observed = transform_points(camera_T_marker @ marker2hand_true, hand_points)
        frames.append(
            AlignmentFrame(
                timestamp=float(idx),
                camera_T_marker=camera_T_marker,
                q_encoder_rad20=q_raw,
                keypoints_camera_mm=observed,
                keypoint_confidence=np.ones(21, dtype=np.float64),
                keypoint_valid_mask=np.ones(21, dtype=bool),
            )
        )
    return AlignmentDataset(hand="left", frames=tuple(frames), capture_kind="s2")


def _make_mixed_modality_s2_dataset(
    true_skeleton: dict,
    marker2hand_true: np.ndarray,
    joint_scale_true: np.ndarray,
    joint_bias_true: np.ndarray,
) -> AlignmentDataset:
    base_dataset = _make_s2_dataset(true_skeleton, marker2hand_true, joint_scale_true, joint_bias_true)
    mixed_frames = list(base_dataset.frames)
    for idx in range(3):
        camera_T_marker = make_transform(
            _rot_z(-0.03 * idx) @ _rot_x(0.02 * idx),
            np.array([8.0 * idx, -2.0 * idx, 640.0 + 3.0 * idx], dtype=np.float64),
        )
        hand_points = _runtime_landmarks(true_skeleton, np.zeros(20, dtype=np.float64), "left")
        observed = transform_points(camera_T_marker @ marker2hand_true, hand_points)
        mixed_frames.append(
            AlignmentFrame(
                timestamp=1000.0 + float(idx),
                camera_T_marker=camera_T_marker,
                q_encoder_rad20=np.full(20, np.nan, dtype=np.float64),
                keypoints_camera_mm=observed,
                keypoint_confidence=np.ones(21, dtype=np.float64),
                keypoint_valid_mask=np.ones(21, dtype=bool),
            )
        )
    return AlignmentDataset(hand="left", frames=tuple(mixed_frames), capture_kind="s2")


def test_run_alignment_optimization_recovers_dexalign2_synthetic_parameters(tmp_path) -> None:
    initial_skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)
    true_skeleton, _palm_radii_true, finger_lengths_true = _make_true_skeleton(initial_skeleton)

    fixed_rotation = _rot_z(0.08) @ _rot_x(-0.12)
    initial_marker2hand = make_transform(
        fixed_rotation,
        np.array([45.0, 84.0, -96.0], dtype=np.float64),
    )
    true_marker2hand = make_transform(
        fixed_rotation,
        np.array([60.0, 94.0, -128.0], dtype=np.float64),
    )
    joint_scale_true = np.linspace(0.92, 1.08, 20, dtype=np.float64)
    joint_bias_true = np.deg2rad(np.linspace(-4.0, 5.0, 20, dtype=np.float64))

    dataset_s1 = _make_s1_dataset(true_skeleton, true_marker2hand)
    dataset_s2 = _make_s2_dataset(true_skeleton, true_marker2hand, joint_scale_true, joint_bias_true)

    result = run_alignment_optimization(
        dataset_s1=dataset_s1,
        dataset_s2=dataset_s2,
        initial_skeleton=initial_skeleton,
        initial_marker2hand_mm=initial_marker2hand,
        output_dir=tmp_path,
        dataset_s1_source_path="synthetic_s1.npz",
        dataset_s2_source_path="synthetic_s2.npz",
        write_outputs=True,
        generate_plots=False,
        max_nfev_step2=240,
        max_nfev_step3=260,
    )

    step2_summary = result.summary_metrics["step2"]
    step3_summary = result.summary_metrics["step3"]

    assert step3_summary["final_mean_error_mm"] < step3_summary["initial_mean_error_mm"]
    # DexAlign 2.0 keeps step2/step3 regularization and fixed marker rotation,
    # so the synthetic case is expected to recover to sub-millimeter accuracy,
    # but not necessarily to an almost exact zero-residual fit.
    assert step3_summary["final_mean_error_mm"] < 0.15
    np.testing.assert_allclose(result.optimized_marker2hand[:3, :3], fixed_rotation, atol=1e-9)
    np.testing.assert_allclose(result.optimized_marker2hand[:3, 3], true_marker2hand[:3, 3], atol=2.0)
    assert abs(float(result.optimized_skeleton["thumb"]["metacarpal"]) - float(finger_lengths_true[0])) < 2.0
    assert abs(float(result.optimized_skeleton["middle"]["middle"]) - float(finger_lengths_true[7])) < 2.0
    assert abs(float(step2_summary["joint_scale"][0]) - float(joint_scale_true[0])) < 0.05
    # Non-thumb joints are regularized more tightly toward scale=1, so allow a
    # slightly wider tolerance here than the thumb channel above.
    assert abs(float(step2_summary["joint_scale"][19]) - float(joint_scale_true[19])) < 0.08
    assert abs(float(step2_summary["joint_bias_rad"][0]) - float(joint_bias_true[0])) < np.deg2rad(1.5)
    assert abs(float(step2_summary["joint_bias_rad"][19]) - float(joint_bias_true[19])) < np.deg2rad(1.5)
    assert result.summary_metrics["loss_weights"]["step2"]["joint_scale_lower_bound"] == 0.5
    assert result.summary_metrics["loss_weights"]["step2"]["joint_scale_upper_bound"] == 1.5
    assert result.summary_metrics["loss_weights"]["step2"]["joint_bias_bound_deg"] == 35.0
    assert result.optimized_skeleton["palm"]["coordinate_mode"] == "runtime"
    assert (tmp_path / "optimized_skeleton.json").exists()
    assert (tmp_path / "optimized_marker2hand.json").exists()
    assert (tmp_path / "optimization_report.json").exists()
    assert (tmp_path / "optimized_joint_calibration.json").exists()

    joint_payload = json.loads((tmp_path / "optimized_joint_calibration.json").read_text(encoding="utf-8"))
    assert "joint_scale" not in joint_payload["summary"]
    assert "joint_bias_rad" not in joint_payload["summary"]


def test_load_marker2hand_asset_mm_falls_back_to_initial_guess(tmp_path) -> None:
    payload = {
        "hand": "left",
        "initial_guess": {
            "trans": [0.01, 0.02, -0.03],
            "rot": np.eye(3, dtype=np.float64).tolist(),
            "trans_unit": "m",
        },
        "note": "test",
    }
    asset_path = tmp_path / "marker2hand.json"
    asset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    initial_mm, result_mm, active_mm = load_marker2hand_asset_mm(asset_path)

    assert result_mm is None
    np.testing.assert_allclose(initial_mm[:3, 3], np.array([10.0, 20.0, -30.0], dtype=np.float64), atol=1e-9)
    np.testing.assert_allclose(active_mm[:3, 3], initial_mm[:3, 3], atol=1e-9)


def test_run_alignment_optimization_uses_per_frame_modalities_not_dataset_identity(tmp_path) -> None:
    initial_skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)
    true_skeleton, _palm_radii_true, _finger_lengths_true = _make_true_skeleton(initial_skeleton)

    fixed_rotation = _rot_z(0.08) @ _rot_x(-0.12)
    initial_marker2hand = make_transform(
        fixed_rotation,
        np.array([45.0, 84.0, -96.0], dtype=np.float64),
    )
    true_marker2hand = make_transform(
        fixed_rotation,
        np.array([60.0, 94.0, -128.0], dtype=np.float64),
    )
    joint_scale_true = np.linspace(0.92, 1.08, 20, dtype=np.float64)
    joint_bias_true = np.deg2rad(np.linspace(-4.0, 5.0, 20, dtype=np.float64))

    mixed_dataset_s2 = _make_mixed_modality_s2_dataset(
        true_skeleton,
        true_marker2hand,
        joint_scale_true,
        joint_bias_true,
    )

    result = run_alignment_optimization(
        dataset_s1=None,
        dataset_s2=mixed_dataset_s2,
        initial_skeleton=initial_skeleton,
        initial_marker2hand_mm=initial_marker2hand,
        output_dir=tmp_path,
        dataset_s2_source_path="synthetic_mixed_s2.npz",
        write_outputs=False,
        generate_plots=False,
        max_nfev_step2=240,
        max_nfev_step3=260,
    )

    assert result.summary_metrics["frame_pool_mode"] == "merged_frames_gated_by_modalities"
    assert result.summary_metrics["capture_fallback"] == "dataset_s1_missing_reused_s2"
    assert result.summary_metrics["frame_pool_summary"]["total_frames"] == mixed_dataset_s2.num_frames
    assert result.summary_metrics["frame_pool_summary"]["frames_without_finite_q"] == 3
    assert result.summary_metrics["step3"]["translation_sample_count"] == mixed_dataset_s2.num_frames


def test_step3_optional_thumb_base_rx_recovers_x_axis_rotation() -> None:
    initial_skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)
    palm_radii, base_directions = _palm_radii_and_directions(initial_skeleton, "left")
    finger_lengths = np.asarray(
        [
            float(initial_skeleton[finger][bone_name])
            for finger, bone_name in (
                ("thumb", "metacarpal"),
                ("thumb", "proximal"),
                ("thumb", "distal"),
                ("index", "proximal"),
                ("index", "middle"),
                ("index", "distal"),
                ("middle", "proximal"),
                ("middle", "middle"),
                ("middle", "distal"),
                ("ring", "proximal"),
                ("ring", "middle"),
                ("ring", "distal"),
                ("pinky", "proximal"),
                ("pinky", "middle"),
                ("pinky", "distal"),
            )
        ],
        dtype=np.float64,
    )
    skeleton_l1 = _make_runtime_shaped_skeleton(
        initial_skeleton,
        hand="left",
        base_radii_mm=palm_radii,
        base_directions=base_directions,
        finger_lengths_mm=finger_lengths,
    )

    true_angle = np.deg2rad(14.0)
    true_directions = base_directions.copy()
    true_directions[0] = _rot_x(true_angle) @ true_directions[0]
    true_skeleton = _make_runtime_shaped_skeleton(
        initial_skeleton,
        hand="left",
        base_radii_mm=palm_radii,
        base_directions=true_directions,
        finger_lengths_mm=finger_lengths,
    )
    true_skeleton["palm"]["thumb_chain_rx_rad"] = float(true_angle)

    marker2hand = make_transform(np.eye(3, dtype=np.float64), np.array([35.0, 70.0, -110.0]))
    dataset_s2 = _make_s2_dataset(
        true_skeleton,
        marker2hand,
        np.ones(20, dtype=np.float64),
        np.zeros(20, dtype=np.float64),
    )
    frame_pool = _build_frame_pool(None, dataset_s2)

    fixed_result = run_step3_lengths_and_translation(
        frame_pool,
        initial_skeleton,
        skeleton_l1,
        marker2hand,
        np.ones(20, dtype=np.float64),
        np.zeros(20, dtype=np.float64),
        max_nfev=180,
        optimize_thumb_base_rx=False,
    )
    result = run_step3_lengths_and_translation(
        frame_pool,
        initial_skeleton,
        skeleton_l1,
        marker2hand,
        np.ones(20, dtype=np.float64),
        np.zeros(20, dtype=np.float64),
        max_nfev=180,
        optimize_thumb_base_rx=True,
    )

    assert result.optimize_thumb_base_rx is True
    assert fixed_result.optimize_thumb_base_rx is False
    assert fixed_result.thumb_base_rx_delta_rad == 0.0
    assert "thumb_chain_rx_rad" not in fixed_result.optimized_skeleton["palm"]
    assert abs(result.thumb_base_rx_delta_rad - true_angle) < np.deg2rad(0.5)
    assert abs(float(result.optimized_skeleton["palm"]["thumb_chain_rx_rad"]) - true_angle) < np.deg2rad(0.5)
    assert float(np.nanmean(result.final_eval.frame_mean_error_mm)) < 0.1
    assert float(np.nanmean(result.final_eval.frame_mean_error_mm)) < float(
        np.nanmean(fixed_result.final_eval.frame_mean_error_mm)
    )


def test_resolve_dataset_pair_rejects_broken_empty_s1_with_nonzero_meta(tmp_path) -> None:
    dataset_s2 = AlignmentDataset(hand="left", capture_kind="s2", frames=(_make_s2_dataset(load_runtime_skeleton(DEFAULT_SKELETON_FILE), np.eye(4), np.ones(20), np.zeros(20)).frames[0],))
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    save_alignment_dataset(session_dir, dataset_s2, extra_meta={"capture_summary": {"frames_kept": 1}})

    np.savez_compressed(
        session_dir / "dataset_s1.npz",
        timestamps=np.zeros((0,), dtype=np.float64),
        camera_T_marker=np.zeros((0, 4, 4), dtype=np.float64),
        q_encoder_rad20=np.zeros((0, 20), dtype=np.float64),
        keypoints_camera_mm=np.zeros((0, 21, 3), dtype=np.float64),
        keypoint_confidence=np.zeros((0, 21), dtype=np.float64),
        keypoint_valid_mask=np.zeros((0, 21), dtype=np.uint8),
        keypoints_uv=np.zeros((0, 21, 2), dtype=np.float64),
        depth_mm=np.zeros((0, 21), dtype=np.float64),
    )
    (session_dir / "dataset_s1_meta.json").write_text(
        json.dumps({"hand": "left", "capture_kind": "s1", "frame_count": 12}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="meta records|contains 0 frames"):
        _resolve_dataset_pair(
            session_dir=session_dir,
            dataset_s1=None,
            meta_s1=None,
            dataset_s2=None,
            meta_s2=None,
            dataset_legacy=None,
            meta_legacy=None,
        )
