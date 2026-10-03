"""Pre-flight image quality checks. Reject clearly unsuitable images before
running the (expensive) detection pipeline.
"""
from __future__ import annotations
import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(gray: np.ndarray) -> dict:
    mean = float(gray.mean())
    # fraction of pixels clipped at extremes
    clipped_dark = float((gray < 8).mean())
    clipped_bright = float((gray > 247).mean())
    return {"mean_brightness": mean, "clipped_dark": clipped_dark, "clipped_bright": clipped_bright}


def assess_quality(image: np.ndarray, min_width=800, min_height=800,
                    blur_threshold=60.0, dark_mean_threshold=50.0,
                    bright_mean_threshold=225.0) -> dict:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    reasons = []
    if w < min_width or h < min_height:
        reasons.append("resolution_too_low")

    b = blur_score(gray)
    if b < blur_threshold:
        reasons.append("image_too_blurry")

    exp = exposure_stats(gray)
    if exp["mean_brightness"] < dark_mean_threshold:
        reasons.append("image_too_dark")
    if exp["mean_brightness"] > bright_mean_threshold:
        reasons.append("image_too_bright")

    # Normalize a rough 0-1 quality score for downstream confidence use.
    blur_component = min(b / (blur_threshold * 3), 1.0)
    exposure_component = 1.0 - min(
        abs(exp["mean_brightness"] - 140) / 140, 1.0
    )
    resolution_component = min((w * h) / (1600 * 1600), 1.0)
    score = float(np.clip(0.5 * blur_component + 0.3 * exposure_component + 0.2 * resolution_component, 0, 1))

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "score": round(score, 3),
        "blur_score": round(b, 1),
        "exposure": exp,
        "width": w,
        "height": h,
    }
