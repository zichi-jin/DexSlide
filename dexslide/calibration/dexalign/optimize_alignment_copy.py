from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dexslide.paths import DEFAULT_LEFT_MARKER_TO_WRIST_FILE, DEFAULT_SKELETON_FILE

from dexslide.calibration.dexalign.io_utils import (
    describe_dataset,
    ensure_session_dir,
    load_alignment_dataset,
    load_marker2hand_asset_mm,
    load_runtime_skeleton,
    save_marker2hand_result,
    save_runtime_skeleton,
)
from dexslide.calibration.dexalign.pipeline_v2 import _palm_radii_and_directions, run_dexalign_v2
from dexslide.calibration.dexalign.skeleton_param import FINGER_BONE_LAYOUT
from dexslide.calibration.dexalign.types import AlignmentDataset, OptimizationResult


def _safe_nanmean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _resolve_dataset_pair(
    *,
    session_dir: str | Path | None,
    dataset_s1: str | Path | None,
    meta_s1: str | Path | None,
    dataset_s2: str | Path | None,
    meta_s2: str | Path | None,
    dataset_legacy: str | Path | None,
    meta_legacy: str | Path | None,
) -> tuple[AlignmentDataset | None, str | None, AlignmentDataset, str, Path]:
    if session_dir is not None:
        resolved_session_dir = Path(session_dir).expanduser().resolve()
        s1_npz = resolved_session_dir / "dataset_s1.npz"
        s2_npz = resolved_session_dir / "dataset_s2.npz"
        legacy_npz = resolved_session_dir / "dataset.npz"
        if s2_npz.exists():
            dataset_s2 = s2_npz
            meta_s2 = resolved_session_dir / "dataset_s2_meta.json"
        elif legacy_npz.exists():
            dataset_s2 = legacy_npz
            meta_s2 = resolved_session_dir / "dataset_meta.json"
        else:
            raise FileNotFoundError(f"No S2 dataset found under session dir: {resolved_session_dir}")
        if s1_npz.exists():
            dataset_s1 = s1_npz
            meta_s1 = resolved_session_dir / "dataset_s1_meta.json"
        output_dir = resolved_session_dir
    else:
        if dataset_s2 is None and dataset_legacy is not None:
            dataset_s2 = dataset_legacy
            meta_s2 = meta_legacy
        if dataset_s2 is None:
            raise ValueError("Provide --session-dir or --dataset-s2 (or legacy --dataset).")
        output_dir = Path(dataset_s2).expanduser().resolve().parent

    resolved_s2_path = Path(dataset_s2).expanduser().resolve()
    resolved_meta_s2 = None if meta_s2 is None else Path(meta_s2).expanduser().resolve()
    loaded_s2 = load_alignment_dataset(resolved_s2_path, resolved_meta_s2)

    loaded_s1 = None
    resolved_s1_path = None
    if dataset_s1 is not None:
        resolved_s1_path = Path(dataset_s1).expanduser().resolve()
        resolved_meta_s1 = None if meta_s1 is None else Path(meta_s1).expanduser().resolve()
        loaded_s1 = load_alignment_dataset(resolved_s1_path, resolved_meta_s1)
    return (
        loaded_s1,
        None if resolved_s1_path is None else str(resolved_s1_path),
        loaded_s2,
        str(resolved_s2_path),
        output_dir,
    )


def _build_result_payload(
    *,
    run_summary: dict[str, Any],
    dataset_s1: AlignmentDataset | None,
    dataset_s1_path: str | None,
    dataset_s2: AlignmentDataset,
    dataset_s2_path: str,
    initial_marker2hand_mm: np.ndarray,
    optimized_marker2hand_mm: np.ndarray,
) -> dict[str, Any]:
    input_summary: dict[str, Any] = {
        "dataset_s2": {
            "path": dataset_s2_path,
            "summary": describe_dataset(dataset_s2),
        },
        "marker2hand_rotation_source": "initial_guess",
    }
    if dataset_s1 is not None:
        input_summary["dataset_s1"] = {
            "path": dataset_s1_path,
            "summary": describe_dataset(dataset_s1),
        }
    else:
        input_summary["dataset_s1"] = None

    marker_rotation_delta = optimized_marker2hand_mm[:3, :3] - initial_marker2hand_mm[:3, :3]
    run_summary = dict(run_summary)
    run_summary["input_datasets"] = input_summary
    run_summary["marker2hand_rotation_fixed"] = bool(np.allclose(marker_rotation_delta, 0.0, atol=1e-12))
    return run_summary


def run_alignment_optimization(
    *,
    dataset_s1: AlignmentDataset | None,
    dataset_s2: AlignmentDataset,
    initial_skeleton: dict[str, Any] | str | Path,
    initial_marker2hand_mm: np.ndarray,
    output_dir: str | Path | None = None,
    dataset_s1_source_path: str | None = None,
    dataset_s2_source_path: str | None = None,
    write_outputs: bool = True,
    generate_plots: bool = True,
    max_nfev_step2: int = 200,
    max_nfev_step3: int = 250,
) -> OptimizationResult:
    initial_skeleton_dict = (
        load_runtime_skeleton(initial_skeleton)
        if isinstance(initial_skeleton, (str, Path))
        else dict(initial_skeleton)
    )
    run_result = run_dexalign_v2(
        dataset_s1=dataset_s1,
        dataset_s2=dataset_s2,
        initial_skeleton=initial_skeleton_dict,
        marker2hand_initial_mm=initial_marker2hand_mm,
        max_nfev_step2=int(max_nfev_step2),
        max_nfev_step3=int(max_nfev_step3),
    )

    optimized_skeleton = run_result.step3.optimized_skeleton
    optimized_marker2hand = run_result.step3.optimized_marker2hand
    total_residual_final = np.concatenate(
        [
            run_result.step3.point_residual_vector_final,
            run_result.step3.translation_sample_residual_final,
        ],
        axis=0,
    )
    final_cost = float(0.5 * np.dot(total_residual_final, total_residual_final))
    summary_metrics = _build_result_payload(
        run_summary=run_result.summary,
        dataset_s1=dataset_s1,
        dataset_s1_path=dataset_s1_source_path,
        dataset_s2=dataset_s2,
        dataset_s2_path=str(dataset_s2_source_path or ""),
        initial_marker2hand_mm=initial_marker2hand_mm,
        optimized_marker2hand_mm=optimized_marker2hand,
    )

    output_path: Path | None = None
    if write_outputs or generate_plots:
        if output_dir is None:
            _session_id, output_path = ensure_session_dir()
        else:
            output_path = Path(output_dir).expanduser().resolve()
            output_path.mkdir(parents=True, exist_ok=True)

    if generate_plots and output_path is not None:
        try:
            from dexslide.calibration.dexalign.visualization import save_alignment_plots

            plot_outputs = save_alignment_plots(
                output_path,
                run_result.step3.initial_eval,
                run_result.step3.final_eval,
            )
            summary_metrics["plot_outputs"] = plot_outputs
        except ImportError as exc:
            summary_metrics["plot_outputs"] = {"skipped_reason": f"matplotlib unavailable: {exc}"}

    if write_outputs and output_path is not None:
        skeleton_path = output_path / "optimized_skeleton.json"
        marker_path = output_path / "optimized_marker2hand.json"
        report_path = output_path / "optimization_report.json"
        joint_calib_path = output_path / "optimized_joint_calibration.json"
        summary_metrics["output_paths"] = {
            "optimized_skeleton": str(skeleton_path),
            "optimized_marker2hand": str(marker_path),
            "optimization_report": str(report_path),
            "optimized_joint_calibration": str(joint_calib_path),
        }

        skeleton_payload = dict(optimized_skeleton)
        skeleton_payload["note"] = (
            "This skeleton was optimized by DexAlign 2.0: step1 palm direction, step2 joint calibration, step3 lengths+translation."
        )
        save_runtime_skeleton(skeleton_path, skeleton_payload)
        save_marker2hand_result(
            marker_path,
            hand=dataset_s2.hand,
            initial_transform_mm=initial_marker2hand_mm,
            optimized_transform_mm=optimized_marker2hand,
            source_dataset_path=dataset_s2_source_path,
            note="DexAlign 2.0 optimized marker2hand translation with fixed rotation from initial_guess.",
        )
        with joint_calib_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "hand": dataset_s2.hand,
                    "joint_scale": run_result.step2.joint_scale.tolist(),
                    "joint_bias_rad": run_result.step2.joint_bias.tolist(),
                    "summary": summary_metrics["step2"],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    optimized_palm_radii, _optimized_base_dirs = _palm_radii_and_directions(optimized_skeleton, dataset_s2.hand)
    optimized_finger_lengths = np.asarray(
        [float(optimized_skeleton[finger][bone_name]) for finger, bone_name in FINGER_BONE_LAYOUT],
        dtype=np.float64,
    )
    x_opt = np.concatenate(
        [
            optimized_palm_radii,
            optimized_finger_lengths,
            run_result.step2.joint_scale,
            run_result.step2.joint_bias,
            run_result.step3.optimized_marker2hand[:3, 3],
        ],
        axis=0,
    )
    return OptimizationResult(
        x_opt=x_opt,
        optimized_skeleton=optimized_skeleton,
        optimized_marker2hand=optimized_marker2hand,
        final_cost=final_cost,
        num_frames_used=int(np.count_nonzero(run_result.step3.final_eval.used_frame_mask)),
        summary_metrics=summary_metrics,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DexAlign 2.0 offline optimization from S1/S2 datasets.")
    parser.add_argument("--session-dir", default=None, help="Session directory containing dataset_s1.npz / dataset_s2.npz.")
    parser.add_argument("--dataset-s1", default=None, help="Path to dataset_s1.npz.")
    parser.add_argument("--meta-s1", default=None, help="Optional dataset_s1_meta.json path.")
    parser.add_argument("--dataset-s2", default=None, help="Path to dataset_s2.npz.")
    parser.add_argument("--meta-s2", default=None, help="Optional dataset_s2_meta.json path.")
    parser.add_argument("--dataset", default=None, help="Legacy alias for a single dataset, treated as S2.")
    parser.add_argument("--meta", default=None, help="Optional legacy dataset_meta.json path.")
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--marker2hand-file", default=str(DEFAULT_LEFT_MARKER_TO_WRIST_FILE))
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to the session dir.")
    parser.add_argument("--max-nfev-step2", type=int, default=200)
    parser.add_argument("--max-nfev-step3", type=int, default=250)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_s1, dataset_s1_path, dataset_s2, dataset_s2_path, default_output_dir = _resolve_dataset_pair(
        session_dir=args.session_dir,
        dataset_s1=args.dataset_s1,
        meta_s1=args.meta_s1,
        dataset_s2=args.dataset_s2,
        meta_s2=args.meta_s2,
        dataset_legacy=args.dataset,
        meta_legacy=args.meta,
    )
    initial_skeleton = load_runtime_skeleton(args.skeleton_file)
    initial_marker_mm, _result_marker_mm, active_marker_mm = load_marker2hand_asset_mm(args.marker2hand_file)
    marker2hand_initial_mm = active_marker_mm if initial_marker_mm is None else initial_marker_mm
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir
    )

    result = run_alignment_optimization(
        dataset_s1=dataset_s1,
        dataset_s2=dataset_s2,
        initial_skeleton=initial_skeleton,
        initial_marker2hand_mm=marker2hand_initial_mm,
        output_dir=output_dir,
        dataset_s1_source_path=dataset_s1_path,
        dataset_s2_source_path=dataset_s2_path,
        write_outputs=True,
        generate_plots=not args.no_plots,
        max_nfev_step2=int(args.max_nfev_step2),
        max_nfev_step3=int(args.max_nfev_step3),
    )
    final_mean_error_mm = _safe_nanmean(
        np.asarray(result.summary_metrics["step3"]["per_frame_mean_error_mm_after"], dtype=np.float64)
    )
    print(
        f"[dexalign2-optimize] used_frames={result.num_frames_used} "
        f"final_mean_error_mm={final_mean_error_mm:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
