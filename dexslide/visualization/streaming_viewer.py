"""Optional OpenCV consumer for :mod:`dexslide.streaming`."""

from __future__ import annotations

import cv2
import numpy as np

from dexslide.kinematics.transforms import transform_points, transform_to_rvec_tvec
from dexslide.retargeting.human_model import DexSlideHumanModel
from dexslide.streaming import DexSlideScene, DexSlideSceneSample
from dexslide.visualization.aruco_overlay import (
    draw_axes,
    draw_marker_outline,
    draw_projected_hand,
    marker_id_to_color,
)


class DexSlideARViewer:
    """Render the shared scene overlay without owning acquisition."""

    def __init__(
        self,
        scene: DexSlideScene,
        *,
        window_name: str = "DexSlide Streaming",
        table_axis_length_m: float = 0.08,
        hand_axis_length_m: float = 0.05,
        show_skeleton: bool = True,
    ) -> None:
        self.scene = scene
        self.window_name = str(window_name)
        self.table_axis_length_m = float(table_axis_length_m)
        self.hand_axis_length_m = float(hand_axis_length_m)
        self.show_skeleton = bool(show_skeleton)
        self._human_models: dict[str, DexSlideHumanModel] = {}
        # Keep this optional so custom/mock scenes used by API clients still work.
        skeleton_files = getattr(scene, "hand_skeleton_files", {})
        handedness = getattr(scene, "hand_handedness", {})
        for hand_id, skeleton_file in skeleton_files.items():
            try:
                self._human_models[str(hand_id)] = DexSlideHumanModel(
                    skeleton_file,
                    hand=str(handedness.get(hand_id, hand_id)),
                )
            except Exception:
                continue
        self.closed = False

    def update(self, sample: DexSlideSceneSample) -> bool:
        """Draw one sample and return False after the user closes the window."""

        if self.closed:
            return False
        frame = self.scene.latest_color_frame()
        intrinsics = self.scene.latest_intrinsics()
        if frame is None or intrinsics is None:
            return True
        image = frame.copy()
        if sample.table_marker_corners_px is not None:
            draw_marker_outline(
                image,
                sample.table_marker_corners_px,
                (0, 215, 255),
                None,
            )
        for hand in sample.hands.values():
            for marker_id, corners in hand.marker_corners_px.items():
                draw_marker_outline(image, corners, marker_id_to_color(int(marker_id)), None)
        if sample.table_valid:
            table_rvec, table_tvec = transform_to_rvec_tvec(sample.camera_T_table)
            draw_axes(
                image,
                intrinsics,
                table_rvec,
                table_tvec,
                self.table_axis_length_m,
                None,
                (0, 215, 255),
            )
            for index, (hand_id, hand) in enumerate(sample.hands.items()):
                color = (80, 255, 80) if index % 2 == 0 else (255, 180, 80)
                if not hand.pose_valid:
                    continue
                camera_t_table = np.asarray(sample.camera_T_table, dtype=np.float64)
                camera_t_body = camera_t_table @ np.asarray(
                    hand.transform_table_body if hand.transform_table_body is not None else hand.transform_table_hand
                )
                body_rvec, body_tvec = transform_to_rvec_tvec(camera_t_body)
                draw_axes(
                    image,
                    intrinsics,
                    body_rvec,
                    body_tvec,
                    self.hand_axis_length_m,
                    None,
                    color,
                )
                if self.show_skeleton and hand.joints_valid:
                    model = self._human_models.get(hand_id)
                    if model is not None:
                        angles = np.asarray(hand.joint_angles, dtype=np.float64)
                        if hand.joint_unit == "deg":
                            angles = np.deg2rad(angles)
                        landmarks_wrist = model.landmarks_from_angles(angles)
                        camera_landmarks = transform_points(
                            camera_t_table @ np.asarray(hand.transform_table_hand, dtype=np.float64),
                            landmarks_wrist,
                        )
                        draw_projected_hand(
                            image,
                            intrinsics,
                            camera_landmarks,
                            draw_axes_enabled=False,
                            axis_rvec=None,
                            axis_tvec=None,
                            axis_length_m=self.hand_axis_length_m,
                        )
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            self.close()
            return False
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 1.0
        if visible < 1.0:
            self.close()
            return False
        return True

    def close(self) -> None:
        if self.closed:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self.closed = True


__all__ = ["DexSlideARViewer"]
