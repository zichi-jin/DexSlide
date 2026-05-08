"""
Landmark Detector — 2D Only

Uses MediaPipe Hands to detect 21 hand landmarks as pixel coordinates
with per-landmark confidence. No 3D lifting — all 3D reasoning is
deferred to the joint optimizer.
"""

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import cv2
except ImportError:
    cv2 = None


class LandmarkDetector:
    """Detects 2D hand landmarks using MediaPipe Hands."""

    # MediaPipe landmark indices for each finger chain
    FINGER_LANDMARKS = {
        "thumb": [0, 1, 2, 3, 4],
        "index": [0, 5, 6, 7, 8],
        "middle": [0, 9, 10, 11, 12],
        "ring": [0, 13, 14, 15, 16],
        "pinky": [0, 17, 18, 19, 20],
    }

    PALM_INDICES = [0, 5, 9, 13, 17]

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        if mp is None:
            raise ImportError(
                "mediapipe is required. Install with: pip install mediapipe"
            )

        hands_api = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            hands_api = mp.solutions.hands
        else:
            # Some mediapipe builds do not expose `solutions` at top-level.
            try:
                from mediapipe.python.solutions import hands as mp_hands

                hands_api = mp_hands
            except Exception as ex:
                raise ImportError(
                    "Unable to access MediaPipe Hands API. "
                    "Tried `mediapipe.solutions.hands` and "
                    "`mediapipe.python.solutions.hands`."
                ) from ex

        self.hands = hands_api.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(
        self, color_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Detect hand landmarks in a BGR image.

        Args:
            color_bgr: H×W×3 BGR image (uint8).

        Returns:
            (keypoints_2d, confidence) where:
                keypoints_2d: (21, 2) pixel coordinates [u, v]
                confidence: (21,) per-landmark visibility/confidence [0..1]
            Returns None if no hand detected.
        """
        h, w = color_bgr.shape[:2]
        rgb = color_bgr[:, :, ::-1]  # BGR → RGB

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        hand_score = 1.0
        if hasattr(results, "multi_handedness") and results.multi_handedness:
            try:
                hand_score = float(results.multi_handedness[0].classification[0].score)
            except Exception:
                hand_score = 1.0

        keypoints_2d = np.zeros((21, 2), dtype=np.float32)
        confidence = np.zeros(21, dtype=np.float32)

        for i, lm in enumerate(hand.landmark):
            keypoints_2d[i, 0] = lm.x * w  # u (pixel)
            keypoints_2d[i, 1] = lm.y * h  # v (pixel)
            # Hand landmarks often don't provide reliable per-landmark visibility.
            # Prefer explicit per-point scores when meaningful, otherwise use hand score.
            c = None
            if hasattr(lm, "presence"):
                try:
                    c = float(lm.presence)
                except Exception:
                    c = None
            if (c is None or c <= 0.0) and hasattr(lm, "visibility"):
                try:
                    c = float(lm.visibility)
                except Exception:
                    c = None
            if c is None or c <= 0.0:
                c = hand_score
            confidence[i] = np.float32(max(0.0, min(1.0, c)))

        # If all confidences collapse near zero, use hand-level confidence.
        if float(confidence.mean()) < 1e-3:
            confidence[:] = np.float32(max(0.2, min(1.0, hand_score)))

        return keypoints_2d, confidence

    def draw_landmarks(
        self, color_bgr: np.ndarray, keypoints_2d: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        """
        Draw detected landmarks on image for preview.

        Returns a copy with landmarks drawn.
        """
        if cv2 is None:
            return color_bgr

        img = color_bgr.copy()

        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17),
        ]

        for p, c in connections:
            pt1 = tuple(keypoints_2d[p].astype(int))
            pt2 = tuple(keypoints_2d[c].astype(int))
            cv2.line(img, pt1, pt2, (200, 200, 200), 1)

        # Draw landmarks with confidence-based color
        for i in range(21):
            pt = tuple(keypoints_2d[i].astype(int))
            conf = confidence[i]
            # Green = high confidence, Red = low
            color = (0, int(255 * conf), int(255 * (1 - conf)))
            radius = 5 if i in self.PALM_INDICES else 3
            cv2.circle(img, pt, radius, color, -1)

        return img

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
