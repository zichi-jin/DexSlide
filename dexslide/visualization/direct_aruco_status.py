"""Status and diagnostic output for direct ArUco overlay."""

from __future__ import annotations

import sys
import time

from dexslide.vision.hand_cube_overlay import CubePoseEstimate
from dexslide.visualization.direct_aruco_config import _marker_id_compact_text

def _emit_runtime_status(
    *,
    frame_idx: int,
    frame_result: dict[str, object],
    cube_pose: CubePoseEstimate | None,
    raw_joint_stamp: float,
    glove_port: str,
) -> None:
    detected = _marker_id_compact_text(list(frame_result.get("detected_ids", [])))
    body_ids = _marker_id_compact_text([] if cube_pose is None else cube_pose.source_marker_ids)
    spread_mm = -1.0 if cube_pose is None else float(cube_pose.max_position_deviation_m) * 1000.0
    reproj_px = -1.0 if cube_pose is None else float(cube_pose.mean_reprojection_error_px)
    solver_mode = "-" if cube_pose is None else str(cube_pose.solver_mode)
    hand_age_ms = (time.time() - raw_joint_stamp) * 1000.0 if raw_joint_stamp > 0.0 else -1.0
    line = (
        f"frame={frame_idx} table={'Y' if frame_result.get('table_detected') else 'N'} "
        f"detected={detected} body={body_ids} spread={spread_mm:.1f}mm reproj={reproj_px:.2f}px solver={solver_mode} "
        f"hand_port={glove_port} hand_age={hand_age_ms:.1f}ms "
        "keys=q"
    )
    sys.stdout.write("\r" + line.ljust(180))
    sys.stdout.flush()


def _emit_marker_body_diagnostic(
    *,
    report: object | None,
    position_threshold_mm: float,
    rotation_threshold_deg: float,
) -> None:
    if report is None or not getattr(report, "items", None):
        return

    suspicious = [
        item
        for item in report.items
        if float(item.peer_position_error_m) * 1000.0 >= float(position_threshold_mm)
        or float(item.peer_rotation_error_deg) >= float(rotation_threshold_deg)
    ]
    visible_text = ",".join(str(marker_id) for marker_id in report.marker_ids)
    if suspicious:
        details: list[str] = []
        for item in suspicious[:3]:
            reproj_text = (
                "-"
                if item.reprojection_mean_error_px is None
                else f"{float(item.reprojection_mean_error_px):.2f}px"
            )
            fused_pos_mm = -1.0 if item.fused_position_error_m is None else float(item.fused_position_error_m) * 1000.0
            fused_rot_deg = -1.0 if item.fused_rotation_error_deg is None else float(item.fused_rotation_error_deg)
            details.append(
                f"id{item.marker_id}:peer={float(item.peer_position_error_m)*1000.0:.1f}mm/{float(item.peer_rotation_error_deg):.1f}deg "
                f"fused={fused_pos_mm:.1f}mm/{fused_rot_deg:.1f}deg reproj={reproj_text}"
            )
        print(
            "\n[diag] visible=["
            f"{visible_text}] suspicious=[{','.join(str(item.marker_id) for item in suspicious)}] "
            + " | ".join(details)
        )
        return

    worst = report.items[0]
    print(
        "\n[diag] visible=["
        f"{visible_text}] consistent "
        f"worst=id{worst.marker_id} peer={float(worst.peer_position_error_m)*1000.0:.1f}mm/{float(worst.peer_rotation_error_deg):.1f}deg"
    )

