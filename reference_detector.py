"""
Detects reference calibration objects: a standard ID-1 card, a coin, or a
bottle cap, anywhere in the image (never assumed to be in a fixed position).
"""
from __future__ import annotations
import numpy as np
import cv2

CARD_ASPECT = 85.60 / 53.98  # ~1.586


def detect_card(image: np.ndarray, expected_aspect: float = CARD_ASPECT,
                 aspect_tolerance: float = 0.18):
    """Find the best quadrilateral candidate matching a card's aspect ratio.

    Returns dict with keys: corners (4x2 float32, order arbitrary),
    confidence (0-1), or None if nothing plausible found.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(w * h)
    best = None
    best_score = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * 0.01 or area > image_area * 0.95:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        aspect = long_side / short_side

        aspect_error = abs(aspect - expected_aspect) / expected_aspect
        if aspect_error > aspect_tolerance:
            continue

        rect_area = rw * rh
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.85:
            continue

        # Score: prioritize good aspect match, high rectangularity, larger area.
        area_frac = area / image_area
        score = (1 - aspect_error / aspect_tolerance) * 0.5 \
            + rectangularity * 0.3 \
            + min(area_frac / 0.25, 1.0) * 0.2

        if score > best_score:
            best_score = score
            best = pts

    if best is None:
        return None

    confidence = float(np.clip(best_score, 0, 1))
    return {"corners": best, "confidence": confidence}


def detect_coin(image: np.ndarray, min_radius_frac: float = 0.02,
                 max_radius_frac: float = 0.35):
    """Detect the most plausible circular coin using Hough circles,
    cross-checked against contour circularity (never trust Hough alone).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_blur = cv2.medianBlur(gray, 5)

    min_r = int(min(h, w) * min_radius_frac)
    max_r = int(min(h, w) * max_radius_frac)
    min_r = max(min_r, 5)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) / 4,
        param1=100, param2=35, minRadius=min_r, maxRadius=max_r,
    )

    # Build contour-based circularity map for cross-validation.
    edges = cv2.Canny(gray_blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_circles = []
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0 or area < np.pi * (min_r ** 2) * 0.5:
            continue
        circularity = 4 * np.pi * area / (peri ** 2)
        if circularity < 0.7:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if min_r <= r <= max_r:
            contour_circles.append(((cx, cy), r, circularity))

    candidates = []
    if circles is not None:
        for x, y, r in circles[0]:
            # cross-validate against contour circularity list
            best_match_circularity = 0.0
            for (ccx, ccy), cr, circ in contour_circles:
                dist = np.hypot(ccx - x, ccy - y)
                if dist < r * 0.3 and abs(cr - r) / r < 0.3:
                    best_match_circularity = max(best_match_circularity, circ)
            confidence = 0.6 + 0.4 * best_match_circularity if best_match_circularity > 0 else 0.45
            candidates.append({"center": (float(x), float(y)), "radius_px": float(r),
                                "confidence": float(min(confidence, 0.99))})

    if not candidates:
        # Fall back to pure contour circularity if Hough found nothing.
        for (cx, cy), r, circ in contour_circles:
            candidates.append({"center": (cx, cy), "radius_px": r,
                                "confidence": float(min(0.5 + 0.3 * circ, 0.85))})

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    best = candidates[0]
    best["diameter_px"] = best["radius_px"] * 2
    return best
