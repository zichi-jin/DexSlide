"""20-DOF geometric hand reconstruction used by live and demo visualizers."""

from __future__ import annotations

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_OFFSET = {"thumb": 0, "index": 4, "middle": 8, "ring": 12, "pinky": 16}
RUNTIME_PALM_COORDINATE_MODE = "runtime"
FOUR_FINGER_MCP_FRONT_ZERO_RAD = 0.0
FOUR_FINGER_MCP_BACK_ZERO_RAD = 0.0
FOUR_FINGER_FORWARD_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
THUMB_CHAIN_RX_FIELD = "thumb_chain_rx_rad"


def _thumb_sign(base: np.ndarray, hand: str) -> float:
    if hand == "left":
        return -1.0
    if hand == "right":
        return 1.0
    return -1.0 if float(base[1]) < 0.0 else 1.0


def thumb_chain_rx_rad(skeleton: dict) -> float:
    value = float(skeleton.get("palm", {}).get(THUMB_CHAIN_RX_FIELD, 0.0))
    return value if np.isfinite(value) else 0.0


def thumb_pp_frame(
    raw4: np.ndarray,
    palm: dict[str, np.ndarray],
    hand: str,
    thumb_base_rx_rad: float = 0.0,
):
    _dip, _pip, mcp_front, mcp_back = [float(v) for v in raw4]
    base = palm["thumb"].copy()
    r_pp = (
        rot_x(float(thumb_base_rx_rad))
        @ rot_x(np.deg2rad(75.0))
        @ rot_z(np.deg2rad(-90.0) + mcp_back)
        @ rot_y(np.deg2rad(90.0) + mcp_front)
        @ rot_x(np.deg2rad(-5.0))
    )
    return base, r_pp[:, 0], r_pp[:, 1], r_pp[:, 2]



# def thumb_pp_frame(raw4: np.ndarray, palm: dict[str, np.ndarray], hand: str):
#     _dip, _pip, mcp_front, mcp_back = [float(v) for v in raw4]
#     base = palm["thumb"].copy()
#     thumb_sign = _thumb_sign(base, hand)
#     z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#     spread_dir = rot_z(mcp_back) @ np.array([0.0, thumb_sign, 0.0], dtype=np.float64)
#     bend_axis = _unit(np.cross(z_axis, spread_dir), np.array([1.0, 0.0, 0.0], dtype=np.float64))
#     x_pp = _unit(
#         np.cos(mcp_front) * spread_dir - np.sin(mcp_front) * z_axis,
#         spread_dir,
#     )
#     y_pp = bend_axis
#     z_pp = _unit(np.cross(x_pp, y_pp), z_axis)
#     return base, x_pp, y_pp, z_pp



def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-9:
        return fallback.astype(np.float64).copy()
    return (v / n).astype(np.float64)


def rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_axis(axis: np.ndarray, rad: float) -> np.ndarray:
    a = _unit(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def extract_palm_points(skeleton: dict) -> dict[str, np.ndarray]:
    verts = skeleton.get("palm", {}).get("vertices", [])
    if isinstance(verts, list) and len(verts) >= 6:
        pts = np.asarray(verts[:6], dtype=np.float64)
        return {
            "wrist": pts[0],
            "thumb": pts[1],
            "index": pts[2],
            "middle": pts[3],
            "ring": pts[4],
            "pinky": pts[5],
        }
    return {
        "wrist": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "thumb": np.array([8.0, 35.0, 0.0], dtype=np.float64),
        "index": np.array([20.0, -22.0, 0.0], dtype=np.float64),
        "middle": np.array([25.0, 0.0, 0.0], dtype=np.float64),
        "ring": np.array([22.0, 20.0, 0.0], dtype=np.float64),
        "pinky": np.array([16.0, 38.0, 0.0], dtype=np.float64),
    }


def canonicalize_palm_xoy(skeleton: dict) -> dict[str, np.ndarray]:
    palm = extract_palm_points(skeleton)
    wrist = palm["wrist"]
    four_center = 0.25 * (palm["index"] + palm["middle"] + palm["ring"] + palm["pinky"])
    x_raw = _unit(four_center - wrist, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    thumb_side = palm["thumb"] - four_center
    z_raw = np.cross(x_raw, thumb_side)
    if _norm(z_raw) < 1e-6:
        z_raw = np.cross(x_raw, palm["index"] - palm["pinky"])
    z_raw = _unit(z_raw, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y_raw = _unit(np.cross(z_raw, x_raw), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    z_raw = _unit(np.cross(x_raw, y_raw), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    rotation = np.stack([x_raw, y_raw, z_raw], axis=1)

    out = {}
    for key, value in palm.items():
        out[key] = (rotation.T @ (value - wrist)).astype(np.float64)
    return out


def apply_handedness(palm: dict[str, np.ndarray], hand: str) -> dict[str, np.ndarray]:
    if hand == "auto":
        return {key: value.copy() for key, value in palm.items()}
    desired = -1.0 if hand == "left" else 1.0
    current = 1.0 if palm["thumb"][1] >= 0.0 else -1.0
    out = {key: value.copy() for key, value in palm.items()}
    if current != desired:
        for key in out:
            out[key][1] *= -1.0
    return out


def runtime_palm_points(skeleton: dict, hand: str) -> dict[str, np.ndarray]:
    palm_payload = skeleton.get("palm", {}) if isinstance(skeleton, dict) else {}
    coordinate_mode = str(palm_payload.get("coordinate_mode", "")).strip().lower()
    if coordinate_mode == RUNTIME_PALM_COORDINATE_MODE:
        palm = extract_palm_points(skeleton)
    else:
        palm = canonicalize_palm_xoy(skeleton)
    return apply_handedness(palm, hand)


def finger_lengths(name: str, skeleton: dict) -> list[float]:
    finger = skeleton.get(name, {})
    if name == "thumb":
        return [
            float(finger.get("metacarpal", 30.0)),
            float(finger.get("proximal", 24.0)),
            float(finger.get("distal", 18.0)),
        ]
    return [
        float(finger.get("proximal", 25.0)),
        float(finger.get("middle", 18.0)),
        float(finger.get("distal", 14.0)),
    ]



def finger_points(
    name: str,
    raw4: np.ndarray,
    skeleton: dict,
    palm: dict[str, np.ndarray],
    hand: str,
) -> np.ndarray:
    dip, pip, mcp_front, mcp_back = [float(v) for v in raw4]
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if name == "thumb":
        base = palm["thumb"].copy()
        _origin, x_pp, y_pp, _z_pp = thumb_pp_frame(
            np.array([dip, pip, mcp_front, mcp_back]),
            palm,
            hand,
            thumb_chain_rx_rad(skeleton),
        )
        lengths = finger_lengths(name, skeleton)
        pts = [base]
        point = base.copy()
        point = point + float(max(0.0, lengths[0])) * x_pp
        pts.append(point.copy())
        point = point + float(max(0.0, lengths[1])) * (rot_axis(y_pp, pip) @ x_pp)
        pts.append(point.copy())
        point = point + float(max(0.0, lengths[2])) * (rot_axis(y_pp, pip + dip) @ x_pp)
        pts.append(point.copy())
        return np.asarray(pts, dtype=np.float64)

    base = palm[name].copy()
    f = rot_z(FOUR_FINGER_MCP_BACK_ZERO_RAD + mcp_back) @ FOUR_FINGER_FORWARD_AXIS
    lengths = finger_lengths(name, skeleton)
    angles = [
        FOUR_FINGER_MCP_FRONT_ZERO_RAD + mcp_front,
        FOUR_FINGER_MCP_FRONT_ZERO_RAD + mcp_front + pip,
        FOUR_FINGER_MCP_FRONT_ZERO_RAD + mcp_front + pip + dip,
    ]
    pts = [base]
    point = base.copy()
    for length, theta in zip(lengths, angles):
        direction = np.cos(theta) * f - np.sin(theta) * z_axis
        point = point + float(max(0.0, length)) * direction
        pts.append(point.copy())
    return np.asarray(pts, dtype=np.float64)


def palm_edges() -> list[tuple[str, str]]:
    return [
        ("wrist", "thumb"),
        ("wrist", "index"),
        ("wrist", "middle"),
        ("wrist", "ring"),
        ("wrist", "pinky"),
        ("thumb", "index"),
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
    ]
