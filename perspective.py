"""
Perspective correction utilities.

A tilted camera means different parts of a flat plane project to the image
sensor at different apparent scales. We never compute a single global
pixels_per_mm from a raw (unwarped) reference. Instead, when four ordered
corners of a planar reference (e.g. a card) are available, we compute a
homography that maps those corners to a known physical rectangle, warp the
region of interest into that rectified coordinate system, and perform all
downstream measurement in the rectified (mm-linear) space.
"""
from __future__ import annotations
import numpy as np
import cv2


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Works regardless of the order the contour points were discovered in,
    and regardless of rotation, using the sum/diff trick.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left     -> smallest x+y
    ordered[2] = pts[np.argmax(s)]  # bottom-right  -> largest x+y

    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]  # top-right    -> smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # bottom-left  -> largest y-x

    return ordered


def compute_homography(corners_px: np.ndarray, physical_width_mm: float,
                        physical_height_mm: float, px_per_mm: float = 10.0):
    """Compute a homography that rectifies a planar reference to a
    top-down view, at a chosen output resolution of px_per_mm pixels per mm.

    Returns (H, output_width_px, output_height_px).
    """
    corners_px = order_points(corners_px)
    out_w = int(round(physical_width_mm * px_per_mm))
    out_h = int(round(physical_height_mm * px_per_mm))

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners_px, dst)
    return H, out_w, out_h


def warp_full_image(image: np.ndarray, H: np.ndarray, out_w: int, out_h: int,
                     canvas_scale: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Warp the ENTIRE image (not just the reference) into the rectified
    coordinate system defined by H, so that the hand/nails next to the
    reference are also rectified onto the same physical mm-grid.

    Because the hand may extend outside the reference's own rectangle, we
    warp onto a larger canvas (canvas_scale x the reference size) with the
    reference placed in a corner offset, then return the adjusted transform
    so pixel coordinates in the warped canvas can be converted to mm by
    dividing by px_per_mm.
    """
    canvas_w = int(out_w * canvas_scale)
    canvas_h = int(out_h * canvas_scale)

    # Shift destination points so the reference sits inset from the canvas
    # origin, leaving room around it for the hand.
    offset_x = out_w * (canvas_scale - 1) / 2
    offset_y = out_h * (canvas_scale - 1) / 2

    # Recompute H with the offset baked into destination points.
    # We assume H currently maps src -> [0,out_w] x [0,out_h]; adjust by
    # composing with a translation.
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1],
    ], dtype=np.float64)
    H_shifted = T @ H

    warped = cv2.warpPerspective(image, H_shifted, (canvas_w, canvas_h))
    return warped, H_shifted


def px_to_mm_scale(out_w_px: int, physical_width_mm: float) -> float:
    """mm per pixel in the rectified coordinate system (uniform in x & y,
    since the rectification already normalizes aspect)."""
    return physical_width_mm / out_w_px
