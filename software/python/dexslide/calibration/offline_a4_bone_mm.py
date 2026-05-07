from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dexslide.calibration.landmark_detector import LandmarkDetector
from dexslide.paths import DEFAULT_RESULTS_FILE, DEFAULT_SKELETON_FILE


# A4 in millimeters, click order must be:
# 1) TL, 2) TR, 3) BR, 4) BL
A4_WORLD_PTS = np.array(
    [[0.0, 0.0], [210.0, 0.0], [210.0, 297.0], [0.0, 297.0]],
    dtype=np.float32,
)


BONE_SPECS: list[tuple[str, int, int]] = [
    ("thumb_base", 0, 1),
    ("thumb_metacarpal", 1, 2),
    ("thumb_proximal", 2, 3),
    ("thumb_distal", 3, 4),
    ("index_metacarpal", 0, 5),
    ("index_proximal", 5, 6),
    ("index_middle", 6, 7),
    ("index_distal", 7, 8),
    ("middle_metacarpal", 0, 9),
    ("middle_proximal", 9, 10),
    ("middle_middle", 10, 11),
    ("middle_distal", 11, 12),
    ("ring_metacarpal", 0, 13),
    ("ring_proximal", 13, 14),
    ("ring_middle", 14, 15),
    ("ring_distal", 15, 16),
    ("pinky_metacarpal", 0, 17),
    ("pinky_proximal", 17, 18),
    ("pinky_middle", 18, 19),
    ("pinky_distal", 19, 20),
    ("palm_mcp_index_middle", 5, 9),
    ("palm_mcp_middle_ring", 9, 13),
    ("palm_mcp_ring_pinky", 13, 17),
]

SKELETON_CONNECTIONS = [(p, c) for _, p, c in BONE_SPECS]
PALM_POINT_ORDER = [0, 1, 5, 9, 13, 17]  # wrist, thumb_cmc, index, middle, ring, pinky
DEFAULT_WINDOW_SIZE = (1680, 980)

SKELETON_BONE_MAP: dict[str, dict[str, str]] = {
    "thumb": {
        "metacarpal": "thumb_metacarpal",
        "proximal": "thumb_proximal",
        "distal": "thumb_distal",
    },
    "index": {
        "metacarpal": "index_metacarpal",
        "proximal": "index_proximal",
        "middle": "index_middle",
        "distal": "index_distal",
    },
    "middle": {
        "metacarpal": "middle_metacarpal",
        "proximal": "middle_proximal",
        "middle": "middle_middle",
        "distal": "middle_distal",
    },
    "ring": {
        "metacarpal": "ring_metacarpal",
        "proximal": "ring_proximal",
        "middle": "ring_middle",
        "distal": "ring_distal",
    },
    "pinky": {
        "metacarpal": "pinky_metacarpal",
        "proximal": "pinky_proximal",
        "middle": "pinky_middle",
        "distal": "pinky_distal",
    },
}
PALM_MCP_MAP = {
    "index_middle": "palm_mcp_index_middle",
    "middle_ring": "palm_mcp_middle_ring",
    "ring_pinky": "palm_mcp_ring_pinky",
}


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return imgs


def read_image_unicode(path: Path) -> np.ndarray | None:
    """
    Robust image reader for paths that may include non-ASCII characters.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _safe_window_image_rect(win: str) -> tuple[int, int, int, int] | None:
    try:
        x, y, w, h = cv2.getWindowImageRect(win)
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def _prepare_window(win: str, window_state: dict[str, int] | None = None) -> None:
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if window_state and "w" in window_state and "h" in window_state:
        cv2.resizeWindow(win, int(window_state["w"]), int(window_state["h"]))
        if "x" in window_state and "y" in window_state:
            cv2.moveWindow(win, int(window_state["x"]), int(window_state["y"]))
        return
    cv2.resizeWindow(win, int(DEFAULT_WINDOW_SIZE[0]), int(DEFAULT_WINDOW_SIZE[1]))


def _remember_window(win: str, window_state: dict[str, int] | None = None) -> None:
    if window_state is None:
        return
    rect = _safe_window_image_rect(win)
    if rect is None:
        return
    x, y, w, h = rect
    window_state.update({"x": x, "y": y, "w": w, "h": h})


def collect_a4_points(
    image_bgr: np.ndarray,
    title: str,
    window_state: dict[str, int] | None = None,
) -> tuple[np.ndarray | None, str]:
    """
    Returns:
      (4,2) float32 points in required order OR None
      status: "ok" | "skip" | "quit"
    Controls:
      mouse-left:
        - if points < 4: add next corner
        - if points == 4: confirm and continue
      mouse-right:
        - undo previous point
      mouse-middle:
        - skip current image
      close window (X):
        - quit all
    """
    win = "A4_Marking"
    points: list[tuple[int, int]] = []
    base = image_bgr.copy()

    def redraw() -> np.ndarray:
        canvas = base.copy()
        hint = "Left:add/confirm  Right:undo  Middle:skip  Close window:quit"
        cv2.putText(canvas, hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 220, 255), 2)
        cv2.putText(
            canvas,
            f"Image: {title}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 80),
            1,
        )
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
        if len(points) >= 2:
            pts = np.asarray(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 200, 255), 2)
        if len(points) == 4:
            pts = np.asarray(points + [points[0]], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)
        return canvas

    action = {"cmd": None}

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((int(x), int(y)))
            else:
                action["cmd"] = "confirm"
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:
            action["cmd"] = "skip"

    _prepare_window(win, window_state)
    # Some OpenCV+Qt builds need one show/poll cycle before mouse callback is valid.
    cv2.imshow(win, redraw())
    cv2.waitKey(1)
    try:
        cv2.setMouseCallback(win, on_mouse)
    except cv2.error as ex:
        print(f"[mark] ERROR: failed to set mouse callback ({ex}).")
        print("[mark] Tip: use X11 desktop session and avoid remote/headless terminal.")
        return None, "skip"
    try:
        print(f"[mark] {title}")
        print("[mark] Left:add/confirm, Right:undo, Middle:skip, Close window:quit")
        last_n = -1
        while True:
            cv2.imshow(win, redraw())
            cv2.waitKey(20)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None, "quit"

            if len(points) != last_n:
                last_n = len(points)
                print(f"[mark] points: {last_n}/4")
            cmd = action["cmd"]
            action["cmd"] = None
            if cmd is None:
                continue
            if cmd == "confirm":
                if len(points) == 4:
                    return np.asarray(points, dtype=np.float32), "ok"
                print("[mark] Need exactly 4 points before confirm.")
            elif cmd == "skip":
                return None, "skip"
    finally:
        _remember_window(win, window_state)
        cv2.destroyWindow(win)


def wait_debug_mouse(image: np.ndarray, title: str, window_state: dict[str, int] | None = None) -> str:
    """
    Show debug image and wait for mouse decision.
    Controls:
      left click: continue
      right click: quit
      close window: quit
    Returns: "continue" | "quit"
    """
    win = "MM_Debug"
    cmd = {"value": None}

    def on_mouse(event, _x, _y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            cmd["value"] = "continue"
        elif event == cv2.EVENT_RBUTTONDOWN:
            cmd["value"] = "quit"

    _prepare_window(win, window_state)
    cv2.imshow(win, image)
    cv2.waitKey(1)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            canvas = image.copy()
            cv2.putText(
                canvas,
                f"Debug: {title} | Left:next  Right:quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (40, 220, 255),
                2,
            )
            cv2.imshow(win, canvas)
            cv2.waitKey(20)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            if cmd["value"] is not None:
                return cmd["value"]
    finally:
        _remember_window(win, window_state)
        cv2.destroyWindow(win)


def homography_img_to_world_mm(points_img_4x2: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(points_img_4x2.astype(np.float32), A4_WORLD_PTS)


def transform_points_h(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = points_xy.astype(np.float64)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    world_h = (H.astype(np.float64) @ homog.T).T
    z = np.clip(world_h[:, 2:3], 1e-9, None)
    world_xy = world_h[:, :2] / z
    return world_xy


def compute_bone_lengths_mm(
    keypoints_mm: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p, c in BONE_SPECS:
        if bool(valid[p]) and bool(valid[c]):
            out[name] = float(np.linalg.norm(keypoints_mm[p] - keypoints_mm[c]))
        else:
            out[name] = float("nan")
    return out


def render_mm_debug_canvas(
    keypoints_mm: np.ndarray,
    valid: np.ndarray,
    bone_lengths: dict[str, float],
) -> np.ndarray:
    """
    Render 2D debug view in A4 world coordinates (mm).
    """
    h_px, w_px = 820, 620
    margin = 60
    scale = min((w_px - 2 * margin) / 210.0, (h_px - 2 * margin) / 297.0)
    canvas = np.full((h_px, w_px, 3), 245, dtype=np.uint8)

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        x = int(round(margin + x_mm * scale))
        y = int(round(margin + y_mm * scale))
        return x, y

    # A4 border
    p0 = to_px(0.0, 0.0)
    p1 = to_px(210.0, 0.0)
    p2 = to_px(210.0, 297.0)
    p3 = to_px(0.0, 297.0)
    cv2.polylines(canvas, [np.array([p0, p1, p2, p3], dtype=np.int32)], True, (80, 80, 80), 2)
    cv2.putText(canvas, "A4 world plane (mm)", (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # skeleton lines
    for p, c in SKELETON_CONNECTIONS:
        if bool(valid[p]) and bool(valid[c]):
            pp = to_px(float(keypoints_mm[p, 0]), float(keypoints_mm[p, 1]))
            cc = to_px(float(keypoints_mm[c, 0]), float(keypoints_mm[c, 1]))
            cv2.line(canvas, pp, cc, (60, 60, 60), 2)

    # points
    for i in range(21):
        if bool(valid[i]):
            pt = to_px(float(keypoints_mm[i, 0]), float(keypoints_mm[i, 1]))
            cv2.circle(canvas, pt, 4, (255, 80, 80), -1)
            cv2.putText(canvas, str(i), (pt[0] + 5, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

    # show a few key lengths
    show_names = ["index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"]
    y0 = 55
    for name in show_names:
        v = bone_lengths.get(name, float("nan"))
        txt = f"{name}: {v:.1f} mm" if np.isfinite(v) else f"{name}: nan"
        cv2.putText(canvas, txt, (350, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        y0 += 22
    return canvas


def print_stats(all_lengths: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    print("\n================ Bone Length Stats (mm) ================")
    for name, vals in all_lengths.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            stats[name] = {"count": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
            print(f"{name:24s}: count=0, mean=nan, std=nan, median=nan")
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        med = float(np.median(arr))
        stats[name] = {"count": int(arr.size), "mean": mean, "std": std, "median": med}
        print(f"{name:24s}: count={arr.size:3d}, mean={mean:7.2f}, std={std:6.2f}, median={med:7.2f}")
    print("========================================================\n")
    return stats


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except Exception:
        return float(default)
    return f if np.isfinite(f) else float(default)


def _aggregate_scalar(stats: dict[str, dict[str, float]], key: str, method: str) -> float:
    row = stats.get(key, {})
    metric = "mean" if method == "mean" else "median"
    val = _safe_float(row.get(metric, float("nan")), default=float("nan"))
    if np.isfinite(val):
        return float(max(0.0, val))
    fallback = _safe_float(row.get("mean", float("nan")), default=0.0)
    return float(max(0.0, fallback))


def _aggregate_points(points: list[np.ndarray], method: str) -> np.ndarray:
    if not points:
        return np.zeros(2, dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if method == "mean":
        return np.mean(arr, axis=0)
    return np.median(arr, axis=0)


def _build_local_palm_samples(
    keypoints_mm: np.ndarray,
    valid: np.ndarray,
) -> dict[int, np.ndarray] | None:
    """
    Convert per-image palm landmarks into a wrist-local 2D frame.
    This removes global translation/rotation before cross-image aggregation.
    """
    required = [0, 5, 9, 13, 17]  # wrist + 4 MCP anchors
    if any((not bool(valid[idx])) for idx in required):
        return None
    if any((not np.isfinite(keypoints_mm[idx]).all()) for idx in required):
        return None

    wrist = keypoints_mm[0]
    index = keypoints_mm[5]
    middle = keypoints_mm[9]
    ring = keypoints_mm[13]
    pinky = keypoints_mm[17]

    x_axis = pinky - index
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm

    y_axis = (index + middle + ring + pinky) * 0.25 - wrist
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        return None
    y_axis /= y_norm

    out: dict[int, np.ndarray] = {}
    for idx in PALM_POINT_ORDER:
        if not bool(valid[idx]):
            continue
        p = keypoints_mm[idx]
        if not np.isfinite(p).all():
            continue
        d = p - wrist
        out[idx] = np.array(
            [float(np.dot(d, x_axis)), float(np.dot(d, y_axis))],
            dtype=np.float64,
        )
    return out


def build_skeleton_dict(
    stats: dict[str, dict[str, float]],
    results: dict[str, dict[str, Any]],
    aggregate: str = "median",
) -> dict[str, Any]:
    skeleton: dict[str, Any] = {}

    for finger, mapping in SKELETON_BONE_MAP.items():
        skeleton[finger] = {
            field: _aggregate_scalar(stats, bone_key, aggregate)
            for field, bone_key in mapping.items()
        }

    point_samples: dict[int, list[np.ndarray]] = {idx: [] for idx in PALM_POINT_ORDER}
    for entry in results.values():
        kp = np.asarray(entry.get("keypoints_mm", []), dtype=np.float64)
        valid = np.asarray(entry.get("valid_mask", []), dtype=bool)
        if kp.shape != (21, 2) or valid.shape[0] != 21:
            continue
        local = _build_local_palm_samples(kp, valid)
        if local is None:
            continue
        for idx, pt in local.items():
            point_samples[idx].append(pt)

    palm_vertices: list[list[float]] = []
    for idx in PALM_POINT_ORDER:
        p = _aggregate_points(point_samples[idx], aggregate)
        palm_vertices.append([_safe_float(p[0]), _safe_float(p[1]), 0.0])

    # If thumb CMC is not reliably visible in photos, synthesize a stable local placeholder.
    # Order: [wrist, thumb_cmc, index_mcp, middle_mcp, ring_mcp, pinky_mcp]
    if len(point_samples[1]) == 0:
        index_pt = np.asarray(palm_vertices[2][:2], dtype=np.float64)
        pinky_pt = np.asarray(palm_vertices[5][:2], dtype=np.float64)
        thumb_guess = index_pt * 0.85 - (pinky_pt - index_pt) * 0.25
        palm_vertices[1] = [_safe_float(thumb_guess[0]), _safe_float(thumb_guess[1]), 0.0]

    skeleton["palm"] = {
        "vertices": palm_vertices,
        "mcp_distances": {
            k: _aggregate_scalar(stats, v, aggregate) for k, v in PALM_MCP_MAP.items()
        },
        "thickness": 25.0,
    }
    return skeleton


def run_offline_pipeline(
    input_dir: Path,
    output_json: Path,
    skeleton_json: Path,
    min_conf: float = 0.35,
    show_debug: bool = False,
    reuse_a4: bool = False,
    skeleton_aggregate: str = "median",
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    image_paths = list_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    detector = LandmarkDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    results: dict[str, dict] = {}
    all_lengths: dict[str, list[float]] = {name: [] for name, _, _ in BONE_SPECS}

    cached_a4_pts: np.ndarray | None = None
    window_state: dict[str, int] = {}

    try:
        for idx, img_path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")
            img = read_image_unicode(img_path)
            if img is None:
                print("  -> skip: image read failed.")
                continue

            if reuse_a4 and cached_a4_pts is not None:
                pts_img = cached_a4_pts.copy()
                mark_status = "ok"
            else:
                pts_img, mark_status = collect_a4_points(
                    img,
                    img_path.name,
                    window_state=window_state,
                )

            if mark_status == "quit":
                print("User requested quit.")
                break
            if mark_status == "skip" or pts_img is None:
                print("  -> skip: user skipped A4 marking.")
                continue

            if reuse_a4 and cached_a4_pts is None:
                cached_a4_pts = pts_img.copy()

            H = homography_img_to_world_mm(pts_img)
            detected = detector.detect(img)
            if detected is None:
                print("  -> skip: MediaPipe found no hand.")
                continue

            keypoints_px, confidence = detected
            keypoints_px = keypoints_px.astype(np.float64)
            confidence = confidence.astype(np.float64)
            valid = np.isfinite(keypoints_px).all(axis=1) & (confidence >= float(min_conf))

            keypoints_mm = transform_points_h(keypoints_px, H)
            bone_lengths = compute_bone_lengths_mm(keypoints_mm, valid)

            for name in all_lengths:
                all_lengths[name].append(float(bone_lengths.get(name, float("nan"))))

            image_entry = {
                "a4_points_px_tl_tr_br_bl": pts_img.astype(float).tolist(),
                "homography_img_to_world": H.astype(float).tolist(),
                "keypoints_px": keypoints_px.astype(float).tolist(),
                "keypoints_mm": keypoints_mm.astype(float).tolist(),
                "confidence": confidence.astype(float).tolist(),
                "valid_mask": valid.astype(bool).tolist(),
                "bone_lengths_mm": bone_lengths,
            }
            results[img_path.name] = image_entry

            # quick console output for core bones
            ip = bone_lengths.get("index_proximal", float("nan"))
            mp = bone_lengths.get("middle_proximal", float("nan"))
            print(f"  index_proximal={ip:.2f} mm, middle_proximal={mp:.2f} mm")

            if show_debug:
                dbg = render_mm_debug_canvas(keypoints_mm, valid, bone_lengths)
                dbg_cmd = wait_debug_mouse(dbg, img_path.name, window_state=window_state)
                if dbg_cmd == "quit":
                    print("User requested quit from debug window.")
                    break

        stats = print_stats(all_lengths)
        skeleton = build_skeleton_dict(
            stats=stats,
            results=results,
            aggregate=skeleton_aggregate,
        )
        payload = {
            "meta": {
                "input_dir": str(input_dir),
                "a4_world_mm_tl_tr_br_bl": A4_WORLD_PTS.astype(float).tolist(),
                "min_conf": float(min_conf),
                "reuse_a4": bool(reuse_a4),
                "skeleton_aggregate": skeleton_aggregate,
            },
            "stats": stats,
            "results": results,
        }
        output_path = Path(output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {output_path}")
        skeleton_path = Path(skeleton_json).resolve()
        skeleton_path.parent.mkdir(parents=True, exist_ok=True)
        with skeleton_path.open("w", encoding="utf-8") as f:
            json.dump(skeleton, f, ensure_ascii=False, indent=2)
        print(f"Saved skeleton to: {skeleton_path}")
        return payload, skeleton
    finally:
        detector.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "assets" / "photos"
    parser = argparse.ArgumentParser(
        description="Offline A4-homography hand bone measurement (mm).",
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input),
        help="Folder containing input images (default: assets/photos)",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_RESULTS_FILE),
        help="Detailed output JSON path",
    )
    parser.add_argument(
        "--skeleton-json",
        default=str(DEFAULT_SKELETON_FILE),
        help="Compact skeleton JSON path",
    )
    parser.add_argument("--min-conf", type=float, default=0.35, help="Min keypoint confidence [0..1]")
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Show per-image mm debug canvas (left click next, right click quit)",
    )
    parser.add_argument(
        "--reuse-a4",
        action="store_true",
        help="Optional: reuse first image A4 points for all images (same camera setup).",
    )
    parser.add_argument(
        "--skeleton-aggregate",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate per-image bones into compact skeleton.json.",
    )
    args = parser.parse_args()

    run_offline_pipeline(
        input_dir=Path(args.input_dir),
        output_json=Path(args.output_json),
        skeleton_json=Path(args.skeleton_json),
        min_conf=float(args.min_conf),
        show_debug=bool(args.show_debug),
        reuse_a4=bool(args.reuse_a4),
        skeleton_aggregate=str(args.skeleton_aggregate),
    )


if __name__ == "__main__":
    main()
