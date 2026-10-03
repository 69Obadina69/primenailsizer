"""
Hand + fingertip landmark detection using MediaPipe Hands.

IMPORTANT: landmarks give us positional priors (where each fingertip is,
which direction each finger points) — they are NOT the nail boundary
themselves. Nail segmentation (nail_detector.py) uses these as a starting
region, but the actual nail polygon comes from segmentation, not from
landmark geometry alone (spec section 13).
"""
from __future__ import annotations
import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

# The current mediapipe build in this environment only ships the new
# Tasks API (mediapipe.tasks), not the legacy mp.solutions.hands API.
# The Tasks API requires a downloaded model bundle (hand_landmarker.task,
# ~8MB) that is NOT bundled with the pip package and must be fetched once
# at deploy/build time, e.g.:
#   wget -O models/hand_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "hand_landmarker.task",
)

FINGER_TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
FINGER_DIP_IDS = {  # joint just below the tip, used to derive pointing direction
    "thumb": 3,
    "index": 7,
    "middle": 11,
    "ring": 15,
    "pinky": 19,
}


class HandDetector:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.6,
                 model_path: str = DEFAULT_MODEL_PATH):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not installed in this environment")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at {model_path}. "
                "Download it once during deployment (network required, see "
                "comment at top of hand_detector.py) — this sandbox has no "
                "network access so it could not be fetched during testing."
            )
        base_options = BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray):
        """Returns list of hands; each hand is a dict with 'handedness',
        'confidence', and 'fingers': {name: {tip, dip, direction_unit}}.
        Coordinates are in pixel space of image_bgr.
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        hands_out = []
        if not result.hand_landmarks:
            return hands_out

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks])
            fingers = {}
            for name, tip_id in FINGER_TIP_IDS.items():
                dip_id = FINGER_DIP_IDS[name]
                tip = pts[tip_id]
                dip = pts[dip_id]
                direction = tip - dip
                norm = np.linalg.norm(direction)
                direction_unit = direction / norm if norm > 1e-6 else np.array([0.0, -1.0])
                fingers[name] = {
                    "tip": tip.tolist(),
                    "dip": dip.tolist(),
                    "direction_unit": direction_unit.tolist(),
                }

            label = "unknown"
            confidence = 0.0
            if result.handedness and i < len(result.handedness):
                classification = result.handedness[i][0]
                label = classification.category_name  # "Left" / "Right" (mirrored, as seen by camera)
                confidence = float(classification.score)

            hands_out.append({
                "handedness": label,
                "confidence": confidence,
                "landmarks_px": pts.tolist(),
                "fingers": fingers,
            })

        return hands_out

    def close(self):
        self._detector.close()
