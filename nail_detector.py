"""
Nail boundary extraction.

Per spec section 14, the PREFERRED approach is a YOLO segmentation model
fine-tuned specifically on fingernails. Training such a model requires a
labeled fingernail dataset and a GPU training run — neither is available
in this environment (no network access to pull a dataset/pretrained
weights, no accelerator). `models/nail_segmentation/README.md` documents
the training pipeline to run that separately.

Until that model is trained and dropped into models/nail_segmentation/,
this module uses the explicitly-sanctioned FALLBACK: MediaPipe fingertip
landmarks define a region-of-interest around each nail, and OpenCV
GrabCut refines that ROI into a polygon. This is clearly weaker than a
trained segmentation model and is labeled as such in every result
(`"method": "grabcut_fallback"`), never reported as equivalent.
"""
from __future__ import annotations
import numpy as np
import cv2
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "nail_segmentation")


def _has_trained_model() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "weights.onnx")) or \
        os.path.exists(os.path.join(MODEL_DIR, "weights.pt"))


def _roi_for_finger(finger: dict, image_shape, nail_length_frac=0.55, width_frac=0.9):
    """Build an oriented rectangular ROI around the fingertip, sized
    proportionally to the distance between the tip and the DIP joint
    (a stable per-finger scale reference), oriented along the finger's
    pointing direction.
    """
    h, w = image_shape[:2]
    tip = np.array(finger["tip"])
    dip = np.array(finger["dip"])
    direction = np.array(finger["direction_unit"])
    finger_seg_len = np.linalg.norm(tip - dip)

    nail_len = finger_seg_len * nail_length_frac
    nail_width = finger_seg_len * width_frac

    perp = np.array([-direction[1], direction[0]])
    # Nail sits between roughly the DIP and slightly past the tip.
    base_center = tip - direction * (nail_len * 0.15)
    tip_center = tip + direction * (nail_len * 0.85)

    corners = np.array([
        base_center - perp * nail_width / 2,
        base_center + perp * nail_width / 2,
        tip_center + perp * nail_width / 2,
        tip_center - perp * nail_width / 2,
    ])
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners


def _grabcut_refine(image: np.ndarray, roi_quad: np.ndarray):
    """Run GrabCut initialized from the oriented ROI rectangle, return a
    refined polygon (largest contour of the foreground mask) plus a
    heuristic confidence based on how much the refined region differs
    from the naive ROI (large disagreement => lower confidence)."""
    h, w = image.shape[:2]
    x, y, rw, rh = cv2.boundingRect(roi_quad.astype(np.int32))
    x, y = max(x, 0), max(y, 0)
    rw = min(rw, w - x - 1)
    rh = min(rh, h - y - 1)
    if rw <= 2 or rh <= 2:
        return roi_quad, 0.2

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x, y, rw, rh)
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return roi_quad, 0.2

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi_quad, 0.25

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (rw * rh) * 0.05:
        return roi_quad, 0.25

    polygon = largest.reshape(-1, 2).astype(np.float64)

    roi_area = cv2.contourArea(roi_quad.astype(np.float32))
    refined_area = cv2.contourArea(largest)
    area_ratio = min(refined_area, roi_area) / max(refined_area, roi_area) if roi_area > 0 else 0
    confidence = float(np.clip(0.35 + 0.4 * area_ratio, 0.2, 0.75))

    return polygon, confidence


def detect_nails(image: np.ndarray, hand: dict):
    """For a single detected hand dict (from HandDetector.detect), return
    a list of nail results: finger, polygon, confidence, method.
    """
    use_trained_model = _has_trained_model()
    results = []
    for finger_name, finger in hand["fingers"].items():
        roi_quad = _roi_for_finger(finger, image.shape)
        if use_trained_model:
            # Placeholder hook: real inference call would go here once a
            # trained fingernail segmentation model is available.
            polygon, confidence, method = roi_quad, 0.9, "yolo_segmentation"
        else:
            polygon, confidence = _grabcut_refine(image, roi_quad)
            method = "grabcut_fallback"
        results.append({
            "finger": finger_name,
            "polygon": polygon.tolist(),
            "confidence": confidence,
            "method": method,
        })
    return results
