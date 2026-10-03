"""
Combines per-stage signals into a final confidence score and an estimated
physical uncertainty (± mm) for each nail measurement. Never report a bare
number without an accompanying confidence/uncertainty.
"""
from __future__ import annotations
import numpy as np


def nail_confidence(*, segmentation_confidence: float, reference_confidence: float,
                     image_quality_score: float, perspective_corrected: bool,
                     calibration_agreement_pct: float | None = None) -> float:
    """Weighted combination of independent confidence signals, all in [0,1]."""
    perspective_factor = 1.0 if perspective_corrected else 0.85

    calibration_factor = 1.0
    if calibration_agreement_pct is not None:
        # Linearly discount confidence as calibration disagreement grows;
        # fully discounted (to 0.5x) at 10% disagreement or worse.
        calibration_factor = float(np.clip(1.0 - calibration_agreement_pct / 20.0, 0.5, 1.0))

    combined = (
        0.40 * segmentation_confidence +
        0.30 * reference_confidence +
        0.20 * image_quality_score +
        0.10 * 1.0  # boundary-quality placeholder until a dedicated check exists
    )
    combined *= perspective_factor * calibration_factor
    return float(np.clip(combined, 0.0, 0.99))


def estimate_uncertainty_mm(*, mm_per_px: float, segmentation_confidence: float,
                             reference_confidence: float, perspective_corrected: bool,
                             measurement_px: float) -> float:
    """Rough uncertainty model: each imperfect stage contributes an
    independent pixel-error term; errors combine in quadrature and are
    converted to mm. This is intentionally conservative (spec explicitly
    forbids pretending every measurement is exact).
    """
    seg_error_px = (1 - segmentation_confidence) * 6.0       # segmentation boundary jitter
    ref_error_px = (1 - reference_confidence) * 3.0          # reference edge localization error
    persp_error_px = 0.5 if perspective_corrected else 3.0   # residual scale error w/o full rectification
    boundary_noise_px = 1.0                                  # baseline pixel-quantization noise

    combined_px_error = np.sqrt(
        seg_error_px ** 2 + ref_error_px ** 2 + persp_error_px ** 2 + boundary_noise_px ** 2
    )
    uncertainty_mm = combined_px_error * mm_per_px
    # Floor so we never claim sub-0.1mm certainty from a phone camera.
    return float(max(round(uncertainty_mm, 2), 0.1))
