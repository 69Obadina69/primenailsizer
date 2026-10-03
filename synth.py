"""Synthetic image generators used to test calibration/perspective/geometry
against KNOWN ground truth, since we don't have a physical camera in CI.
"""
import numpy as np
import cv2


def make_card_image(canvas_size=(900, 1200), card_width_mm=85.60, card_height_mm=53.98,
                     px_per_mm=6.0, tilt_deg=0.0, center=None, bg_color=(40, 40, 40),
                     card_color=(230, 230, 230)):
    """Draws a card-shaped rectangle (optionally rotated in-plane) on a
    background, and returns (image, ground_truth_corners_xy_in_TL_TR_BR_BL_order).
    Pure in-plane rotation (no true 3D tilt) - good enough to test corner
    ordering + aspect scoring + homography scale recovery.
    """
    h, w = canvas_size
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    card_w_px = card_width_mm * px_per_mm
    card_h_px = card_height_mm * px_per_mm

    if center is None:
        center = (w / 2, h / 2)

    rect = ((center[0], center[1]), (card_w_px, card_h_px), tilt_deg)
    box = cv2.boxPoints(rect)  # 4x2, order: varies but consistent rectangle

    cv2.fillConvexPoly(img, box.astype(np.int32), card_color)
    # subtle border so edge detector has an edge to find
    cv2.polylines(img, [box.astype(np.int32)], True, (10, 10, 10), 2)
    img = _add_texture(img)

    return img, box


def _add_texture(img, sigma=6.0, seed=0):
    """Real photos always have sensor noise/texture; a perfectly flat
    synthetic render would incorrectly fail the blur-detection quality
    gate. Add mild noise so synthetic test images behave like real photos."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_coin_image(canvas_size=(800, 800), diameter_mm=24.26, px_per_mm=6.0,
                     center=None, bg_color=(150, 150, 150), coin_color=(210, 180, 100)):
    h, w = canvas_size
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    radius_px = (diameter_mm * px_per_mm) / 2
    if center is None:
        center = (w / 2, h / 2)
    cv2.circle(img, (int(center[0]), int(center[1])), int(round(radius_px)), coin_color, -1)
    cv2.circle(img, (int(center[0]), int(center[1])), int(round(radius_px)), (60, 45, 20), 2)
    img = _add_texture(img)
    return img, center, radius_px


def make_nail_polygon(width_mm=13.0, length_mm=14.0, px_per_mm=10.0, curvature=0.15):
    """Generate a synthetic nail-shaped polygon (rounded-rect-ish / oval)
    with KNOWN width and length in the rectified px space, for testing the
    geometry module independent of segmentation.
    """
    w_px = width_mm * px_per_mm
    l_px = length_mm * px_per_mm
    t = np.linspace(0, 2 * np.pi, 200)
    # superellipse-ish nail shape: width along x, length along y
    x = (w_px / 2) * np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** curvature)
    y = (l_px / 2) * np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** curvature)
    pts = np.stack([x, y], axis=1)
    return pts
