"""
Turns detected reference geometry into a physical mm-per-pixel scale, and
cross-validates multiple references against each other when available.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from . import perspective as persp


@dataclass
class Calibration:
    mm_per_px: float
    method: str            # "card_homography" | "coin" | "bottle_cap" | "custom"
    confidence: float
    perspective_corrected: bool
    warped_image: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None


def calibrate_from_card(image: np.ndarray, corners_px: np.ndarray,
                         card_confidence: float,
                         card_width_mm: float = 85.60,
                         card_height_mm: float = 53.98,
                         px_per_mm_output: float = 10.0) -> Calibration:
    H, out_w, out_h = persp.compute_homography(
        corners_px, card_width_mm, card_height_mm, px_per_mm_output)
    warped, H_shifted = persp.warp_full_image(image, H, out_w, out_h)
    mm_per_px = 1.0 / px_per_mm_output
    return Calibration(
        mm_per_px=mm_per_px,
        method="card_homography",
        confidence=card_confidence,
        perspective_corrected=True,
        warped_image=warped,
        homography=H_shifted,
    )


def original_space_mm_per_px_from_card(corners_px: np.ndarray,
                                        card_width_mm: float = 85.60,
                                        card_height_mm: float = 53.98) -> float:
    """mm-per-pixel of the card AS SEEN IN THE ORIGINAL (unwarped) photo,
    from raw corner distances. This is NOT what's used for the actual
    rectified measurement (calibrate_from_card's mm_per_px is in the
    warped canvas' own coordinate space, at a fixed output resolution,
    and is not comparable to any other calibration's mm_per_px). This
    helper exists purely so a card calibration can be cross-validated
    against a same-image circular reference (coin/cap), which is
    necessarily measured in original-image pixels.
    """
    ordered = persp.order_points(corners_px)
    tl, tr, br, bl = ordered
    top_w_px = np.linalg.norm(tr - tl)
    bottom_w_px = np.linalg.norm(br - bl)
    left_h_px = np.linalg.norm(bl - tl)
    right_h_px = np.linalg.norm(br - tr)

    mm_per_px_w = card_width_mm / ((top_w_px + bottom_w_px) / 2)
    mm_per_px_h = card_height_mm / ((left_h_px + right_h_px) / 2)
    return float((mm_per_px_w + mm_per_px_h) / 2)


def calibrate_from_circle(diameter_px: float, known_diameter_mm: float,
                           detection_confidence: float,
                           method: str = "coin") -> Calibration:
    """Simple planar calibration from a circular reference. This does NOT
    perspective-correct the whole scene (a circle alone can't define a full
    homography), so perspective_corrected=False and confidence is capped
    lower to reflect that residual tilt error is possible.
    """
    mm_per_px = known_diameter_mm / diameter_px
    # Cap confidence since we can't fully rectify perspective from a circle.
    confidence = min(detection_confidence, 0.9)
    return Calibration(
        mm_per_px=mm_per_px,
        method=method,
        confidence=confidence,
        perspective_corrected=False,
    )


def cross_validate(cal_a: Calibration, cal_b: Calibration,
                    tolerance_pct: float = 5.0):
    """Compare two independent calibrations. Returns dict describing
    agreement. Per spec section 20: if they disagree beyond tolerance,
    calibration should be flagged as unreliable rather than silently
    averaged.
    """
    a, b = cal_a.mm_per_px, cal_b.mm_per_px
    diff_pct = abs(a - b) / ((a + b) / 2) * 100.0
    agrees = diff_pct <= tolerance_pct
    return {
        "method_a": cal_a.method,
        "method_b": cal_b.method,
        "mm_per_px_a": a,
        "mm_per_px_b": b,
        "difference_pct": diff_pct,
        "agrees": agrees,
        "message": (
            "Calibration verified across both references."
            if agrees else
            "Reference objects disagree. Make sure both objects are flat "
            "and on the same surface, then retake the photo."
        ),
    }
