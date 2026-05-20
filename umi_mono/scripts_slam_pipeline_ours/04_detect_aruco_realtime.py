"""
README - realtime ArUco detection with per-group fusion
========================================================

What this script does
---------------------
1) Realtime detect one or multiple ArUco markers from camera/video stream.
2) Estimate pose (rvec/tvec) for each marker in camera frame.
3) Apply offset along marker center negative z-axis:
   tvec_offset = tvec - offset_scale * z_axis(marker_in_camera)
4) Fuse only positions (rotation is NOT fused):
   - if one marker in group: use that marker directly
   - if multiple markers in group and all pairwise distances <= threshold:
     use mean position
   - else: emit warning for that group and do not output fused position
5) Support multiple fusion groups via YAML config.

Naming conventions used in this file
------------------------------------
- rvec / tvec:
  OpenCV Rodrigues rotation vector and translation vector (meters).
- tvec_offset:
  Position after offset along marker negative z-axis.
- marker_entries:
  Per-frame list of detected markers, one dict per marker.
- fusion:
  Fusion result dict for one marker set/group.
- group_entries_map:
  Mapping: group_name -> detected marker entries belonging to that group.
- frame_result:
  Per-frame output record (written to jsonl when enabled).
- mode:
  Fusion status string:
  no_marker | single | mean_position_only | disagree

Group config file format (--fusion_groups_yaml)
-----------------------------------------------
Format A:
groups:
  left_tool: [10, 11, 12]
  right_tool: [20, 21]

Format B:
- name: left_tool
  ids: [10, 11, 12]
- name: right_tool
  ids: [20, 21]

Only markers inside the same group are evaluated/fused together.
No cross-group fusion is performed.

CLI parameters
--------------
- --source:
  Capture source. Examples: "0", "/dev/video4", rtsp/http URL, or video file path.
- --camera_intrinsics:
  Camera intrinsics json path.
- --aruco_yaml:
  ArUco dictionary and marker-size config yaml path.
- --offset_scale:
  Offset distance in meters along marker negative z-axis.
- --merge_pos_threshold:
  Max allowed pairwise distance (meters) to enable position averaging in a group.
- --fusion_groups_yaml:
  Optional group config yaml. If omitted, script uses one implicit group "all".
- --warning_cooldown_sec:
  Cooldown seconds for repeated "group disagree" warning print.
- --num_workers:
  OpenCV thread count.
- --width / --height:
  Requested capture resolution.
- --fps:
  Requested capture fps.
- --buffer_size:
  Capture buffer size (backend dependent).
- --display / --no-display:
  Enable/disable realtime visualization window.
- --draw_axes / --no-draw_axes:
  Draw ArUco axes for each detected marker on visualization.
- --save_jsonl:
  Optional output path. One json object per frame.
- --log_interval_sec:
  Console summary print interval.
- --max_frames:
  Max number of frames to process. 0 means run until stream ends or user quits.

Output (jsonl) overview
-----------------------
Each line is one frame with:
- markers: raw per-marker pose and offset position
- groups.<group_name>.fusion: group-specific fusion result
- fused: alias to groups.all.fusion only when group config is not used
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import cv2
import numpy as np
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def parse_capture_source(source: str) -> Union[int, str]:
    src = str(source).strip()
    if src.lstrip("-").isdigit():
        return int(src)
    return os.path.expanduser(src)


def offset_along_marker_negative_z(
    rvec: np.ndarray, tvec: np.ndarray, offset_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    rot_mat, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    z_axis_cam = rot_mat[:, 2]
    tvec_offset = tvec - offset_scale * z_axis_cam
    return tvec_offset, z_axis_cam


def eval_and_fuse_positions(
    marker_entries: List[Dict], merge_pos_threshold: float
) -> Dict:
    n = len(marker_entries)
    if n == 0:
        return {
            "within_threshold": False,
            "max_pairwise_dist": None,
            "fused_position": None,
            "rep_rvec": None,
            "rep_marker_id": None,
            "mode": "no_marker",
        }

    if n == 1:
        entry = marker_entries[0]
        return {
            "within_threshold": True,
            "max_pairwise_dist": 0.0,
            "fused_position": entry["tvec_offset"].copy(),
            "rep_rvec": entry["rvec"].copy(),
            "rep_marker_id": entry["id"],
            "mode": "single",
        }

    positions = np.stack([e["tvec_offset"] for e in marker_entries], axis=0)
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    max_pairwise_dist = float(np.max(dists))
    within_threshold = max_pairwise_dist <= merge_pos_threshold

    if within_threshold:
        rep_entry = marker_entries[0]
        return {
            "within_threshold": True,
            "max_pairwise_dist": max_pairwise_dist,
            "fused_position": np.mean(positions, axis=0),
            "rep_rvec": rep_entry["rvec"].copy(),
            "rep_marker_id": rep_entry["id"],
            "mode": "mean_position_only",
        }

    return {
        "within_threshold": False,
        "max_pairwise_dist": max_pairwise_dist,
        "fused_position": None,
        "rep_rvec": None,
        "rep_marker_id": None,
        "mode": "disagree",
    }


def to_float_list(array_like: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(array_like).reshape(-1)]


def parse_fusion_groups_config(fusion_groups_yaml: str) -> Dict[str, List[int]]:
    path = os.path.expanduser(fusion_groups_yaml)
    cfg = yaml.safe_load(open(path, "r"))
    if cfg is None:
        return dict()

    groups_obj = cfg
    if isinstance(cfg, dict) and ("groups" in cfg):
        groups_obj = cfg["groups"]

    groups: Dict[str, List[int]] = dict()
    if isinstance(groups_obj, dict):
        raw_items = list(groups_obj.items())
    elif isinstance(groups_obj, list):
        raw_items = []
        for i, item in enumerate(groups_obj):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid fusion group entry at index {i}: expected dict with keys 'name' and 'ids'."
                )
            if ("name" not in item) or ("ids" not in item):
                raise ValueError(
                    f"Invalid fusion group entry at index {i}: missing 'name' or 'ids'."
                )
            raw_items.append((item["name"], item["ids"]))
    else:
        raise ValueError(
            "Invalid fusion groups config. Expected dict/list, e.g. "
            "{groups: {g1: [10,11], g2: [20,21]}}."
        )

    for raw_name, raw_ids in raw_items:
        group_name = str(raw_name)
        if group_name in groups:
            raise ValueError(f"Duplicate fusion group name: {group_name}")

        if isinstance(raw_ids, (int, np.integer)):
            id_list = [int(raw_ids)]
        elif isinstance(raw_ids, (list, tuple, set)):
            id_list = [int(x) for x in raw_ids]
        else:
            raise ValueError(
                f"Invalid ids for group '{group_name}'. Expected int/list/tuple/set, got {type(raw_ids)}."
            )

        id_list = sorted(set(id_list))
        if len(id_list) == 0:
            raise ValueError(f"Fusion group '{group_name}' has no marker ids.")
        groups[group_name] = id_list

    return groups


def fusion_to_output_dict(fusion: Dict) -> Dict[str, Any]:
    return {
        "mode": fusion["mode"],
        "within_threshold": bool(fusion["within_threshold"]),
        "max_pairwise_dist": (
            None if fusion["max_pairwise_dist"] is None else float(fusion["max_pairwise_dist"])
        ),
        "rep_marker_id": (None if fusion["rep_marker_id"] is None else int(fusion["rep_marker_id"])),
        "fused_position": (
            None if fusion["fused_position"] is None else to_float_list(fusion["fused_position"])
        ),
        "rep_rvec": (None if fusion["rep_rvec"] is None else to_float_list(fusion["rep_rvec"])),
    }


@click.command()
@click.option(
    "-s",
    "--source",
    default="0",
    show_default=True,
    help="Capture source: camera index (e.g. 0), /dev/videoX, rtsp/http URL, or video path.",
)
@click.option(
    "-ci",
    "--camera_intrinsics",
    required=True,
    help="Camera intrinsics json file.",
)
@click.option("-ac", "--aruco_yaml", required=True, help="Aruco config yaml file.")
@click.option(
    "--offset_scale",
    default=0.0,
    type=float,
    show_default=True,
    help="Offset in meters along marker center negative z-axis.",
)
@click.option(
    "--merge_pos_threshold",
    default=0.03,
    type=float,
    show_default=True,
    help="If all pairwise distances of offset positions are within this threshold (m), fuse by averaging positions.",
)
@click.option(
    "--fusion_groups_yaml",
    type=str,
    default=None,
    help=(
        "Optional YAML file for marker-id groups. "
        "Fusion is only performed inside each group, and supports multiple groups."
    ),
)
@click.option(
    "--warning_cooldown_sec",
    default=1.0,
    type=float,
    show_default=True,
    help="Minimum warning interval when multiple markers disagree.",
)
@click.option("-n", "--num_workers", type=int, default=1, show_default=True)
@click.option("--width", type=int, default=None, help="Requested capture width.")
@click.option("--height", type=int, default=None, help="Requested capture height.")
@click.option("--fps", type=float, default=None, help="Requested capture FPS.")
@click.option("--buffer_size", type=int, default=2, show_default=True, help="Capture buffer size.")
@click.option(
    "--display/--no-display",
    default=True,
    show_default=True,
    help="Show realtime visualization window.",
)
@click.option(
    "--draw_axes/--no-draw_axes",
    default=True,
    show_default=True,
    help="Draw marker axes in visualization.",
)
@click.option(
    "--save_jsonl",
    type=str,
    default=None,
    help="Optional output path. One JSON line per frame.",
)
@click.option(
    "--log_interval_sec",
    type=float,
    default=0.5,
    show_default=True,
    help="Console print interval.",
)
@click.option(
    "--max_frames",
    type=int,
    default=0,
    show_default=True,
    help="0 means run until stream ends or user exits.",
)
def main(
    source,
    camera_intrinsics,
    aruco_yaml,
    offset_scale,
    merge_pos_threshold,
    fusion_groups_yaml,
    warning_cooldown_sec,
    num_workers,
    width,
    height,
    fps,
    buffer_size,
    display,
    draw_axes,
    save_jsonl,
    log_interval_sec,
    max_frames,
):
    cv2.setNumThreads(num_workers)

    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, "r")))
    aruco_dict = aruco_config["aruco_dict"]
    marker_size_map = aruco_config["marker_size_map"]
    raw_intr = parse_fisheye_intrinsics(json.load(open(camera_intrinsics, "r")))

    fusion_groups: Optional[Dict[str, List[int]]] = None
    if fusion_groups_yaml is not None:
        fusion_groups = parse_fusion_groups_config(fusion_groups_yaml)
        if len(fusion_groups) == 0:
            raise ValueError(f"No valid groups found in: {fusion_groups_yaml}")
        print(f"Loaded fusion groups from {os.path.expanduser(fusion_groups_yaml)}")
        for group_name, group_ids in fusion_groups.items():
            print(f"  - {group_name}: {group_ids}")

    cap_source = parse_capture_source(source)
    if isinstance(cap_source, str) and cap_source.startswith("/dev/video"):
        cap = cv2.VideoCapture(cap_source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(cap_source)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open capture source: {source}")

    jsonl_fp = None
    if save_jsonl is not None:
        save_path = os.path.expanduser(save_jsonl)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        jsonl_fp = open(save_path, "w")
        print(f"Writing realtime results to: {save_path}")

    intr = None
    frame_idx = 0
    last_warning_time_by_group: Dict[str, float] = dict()
    last_log_time = -1e9
    t_start = time.monotonic()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Stream ended or failed to grab frame.")
                break

            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            if intr is None:
                intr = convert_fisheye_intrinsics_resolution(
                    opencv_intr_dict=raw_intr, target_resolution=(w, h)
                )
                print(f"Using runtime resolution: {w}x{h}")

            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=intr,
                refine_subpix=True,
            )

            marker_entries = []
            for marker_id in sorted(tag_dict.keys()):
                tag = tag_dict[marker_id]
                rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
                tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
                tvec_offset, z_axis_cam = offset_along_marker_negative_z(
                    rvec=rvec, tvec=tvec, offset_scale=offset_scale
                )
                marker_entries.append(
                    {
                        "id": int(marker_id),
                        "rvec": rvec,
                        "tvec": tvec,
                        "tvec_offset": tvec_offset,
                        "z_axis_cam": z_axis_cam,
                        "corners": np.asarray(tag["corners"], dtype=np.float32),
                    }
                )

            marker_entries_by_id = {m["id"]: m for m in marker_entries}

            if fusion_groups is None:
                group_entries_map: Dict[str, List[Dict]] = {"all": marker_entries}
                group_config_ids: Dict[str, Optional[List[int]]] = {"all": None}
            else:
                group_entries_map = dict()
                group_config_ids = dict()
                for group_name, configured_ids in fusion_groups.items():
                    group_config_ids[group_name] = configured_ids
                    group_entries_map[group_name] = [
                        marker_entries_by_id[mid] for mid in configured_ids if mid in marker_entries_by_id
                    ]

            group_fusions: Dict[str, Dict] = dict()
            now_monotonic = time.monotonic()
            for group_name, group_entries in group_entries_map.items():
                fusion = eval_and_fuse_positions(
                    marker_entries=group_entries,
                    merge_pos_threshold=merge_pos_threshold,
                )
                group_fusions[group_name] = fusion

                if len(group_entries) > 1 and (not fusion["within_threshold"]):
                    last_t = last_warning_time_by_group.get(group_name, -1e9)
                    if now_monotonic - last_t >= warning_cooldown_sec:
                        ids = [m["id"] for m in group_entries]
                        print(
                            f"⚠️ group[{group_name}] marker offset positions disagree: "
                            f"ids={ids}, max_pairwise_dist={fusion['max_pairwise_dist']:.4f}m, "
                            f"threshold={merge_pos_threshold:.4f}m"
                        )
                        last_warning_time_by_group[group_name] = now_monotonic

            frame_result = {
                "frame_idx": int(frame_idx),
                "time_wall": float(time.time()),
                "n_markers": int(len(marker_entries)),
                "offset_scale": float(offset_scale),
                "merge_pos_threshold": float(merge_pos_threshold),
                "fusion_groups_enabled": bool(fusion_groups is not None),
                "markers": [
                    {
                        "id": int(m["id"]),
                        "rvec": to_float_list(m["rvec"]),
                        "tvec": to_float_list(m["tvec"]),
                        "tvec_offset": to_float_list(m["tvec_offset"]),
                    }
                    for m in marker_entries
                ],
                "groups": {},
            }
            for group_name, fusion in group_fusions.items():
                detected_ids = [m["id"] for m in group_entries_map[group_name]]
                frame_result["groups"][group_name] = {
                    "configured_ids": group_config_ids[group_name],
                    "detected_ids": detected_ids,
                    "n_detected": int(len(detected_ids)),
                    "fusion": fusion_to_output_dict(fusion),
                }
            frame_result["fused"] = (
                frame_result["groups"]["all"]["fusion"] if "all" in frame_result["groups"] else None
            )

            if jsonl_fp is not None:
                jsonl_fp.write(json.dumps(frame_result, ensure_ascii=False) + "\n")
                jsonl_fp.flush()

            now = time.monotonic()
            if now - last_log_time >= log_interval_sec:
                if "all" in frame_result["groups"]:
                    all_fusion = frame_result["groups"]["all"]["fusion"]
                    if all_fusion["fused_position"] is None:
                        pos_str = "None"
                    else:
                        p = all_fusion["fused_position"]
                        pos_str = f"[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]"
                    print(
                        f"frame={frame_idx:06d} markers={len(marker_entries)} "
                        f"mode={all_fusion['mode']} "
                        f"pos={pos_str}"
                    )
                else:
                    group_parts = []
                    for group_name in sorted(frame_result["groups"].keys()):
                        g = frame_result["groups"][group_name]
                        group_parts.append(
                            f"{group_name}:{g['fusion']['mode']}/{g['n_detected']}"
                        )
                    print(
                        f"frame={frame_idx:06d} markers={len(marker_entries)} groups="
                        + ", ".join(group_parts)
                    )
                last_log_time = now

            if display:
                vis = frame_bgr.copy()
                if len(marker_entries) > 0:
                    corners = [m["corners"].reshape(1, 4, 2) for m in marker_entries]
                    ids = np.array([m["id"] for m in marker_entries], dtype=np.int32).reshape(-1, 1)
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                for m in marker_entries:
                    if draw_axes:
                        axis_len = float(marker_size_map.get(m["id"], 0.05)) * 0.6
                        cv2.drawFrameAxes(
                            vis,
                            intr["K"],
                            np.zeros((1, 5), dtype=np.float64),
                            m["rvec"].reshape(3, 1),
                            m["tvec"].reshape(3, 1),
                            axis_len,
                            2,
                        )
                    uv, _ = cv2.projectPoints(
                        m["tvec_offset"].reshape(1, 3),
                        np.zeros((3, 1), dtype=np.float64),
                        np.zeros((3, 1), dtype=np.float64),
                        intr["K"],
                        np.zeros((1, 5), dtype=np.float64),
                    )
                    u, v = uv.reshape(2)
                    cv2.circle(vis, (int(round(u)), int(round(v))), 4, (0, 255, 255), -1)
                    cv2.putText(
                        vis,
                        f"id={m['id']}",
                        (int(round(u)) + 4, int(round(v)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if "all" in frame_result["groups"]:
                    g_all = frame_result["groups"]["all"]["fusion"]
                    cv2.putText(
                        vis,
                        f"mode={g_all['mode']}  markers={len(marker_entries)}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if g_all["within_threshold"] else (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if g_all["fused_position"] is not None:
                        px, py, pz = g_all["fused_position"]
                        cv2.putText(
                            vis,
                            f"pos=[{px:.3f}, {py:.3f}, {pz:.3f}] m",
                            (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    y = 30
                    for group_name in sorted(frame_result["groups"].keys()):
                        g = frame_result["groups"][group_name]
                        gf = g["fusion"]
                        line = f"{group_name}: {gf['mode']} ({g['n_detected']})"
                        if gf["fused_position"] is not None:
                            px, py, pz = gf["fused_position"]
                            line += f" [{px:.3f},{py:.3f},{pz:.3f}]"
                        cv2.putText(
                            vis,
                            line,
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0) if gf["within_threshold"] else (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        y += 26

                cv2.imshow("aruco_realtime", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Exit requested by user (q).")
                    break

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                print(f"Reached max_frames={max_frames}, stopping.")
                break

    finally:
        cap.release()
        if jsonl_fp is not None:
            jsonl_fp.close()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    fps_est = frame_idx / max(elapsed, 1e-9)
    print(f"Done. processed_frames={frame_idx}, elapsed={elapsed:.2f}s, avg_fps={fps_est:.2f}")


if __name__ == "__main__":
    main()
