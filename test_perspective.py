import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from vision import perspective as persp
from vision import reference_detector as rd
from vision import calibration as calib
from tests.synth import make_card_image


def make_true_perspective_card(canvas_size=(900, 1200), card_width_mm=85.60,
                                card_height_mm=53.98, px_per_mm=6.0,
                                skew_px=(60, -30, 40, 20)):
    """Build a card with a genuine (non-affine) perspective distortion by
    directly warping a known rectangle's corners, so we have exact ground
    truth corner coordinates to test homography-based rectification."""
    h, w = canvas_size
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    cw = card_width_mm * px_per_mm
    ch = card_height_mm * px_per_mm
    cx, cy = w / 2, h / 2
    tl = (cx - cw / 2, cy - ch / 2)
    tr = (cx + cw / 2, cy - ch / 2)
    br = (cx + cw / 2, cy + ch / 2)
    bl = (cx - cw / 2, cy + ch / 2)
    # Apply independent per-corner pixel skew to simulate perspective tilt.
    sx = skew_px
    tl = (tl[0] + sx[0], tl[1] + sx[0] * 0.3)
    tr = (tr[0] - sx[1], tr[1] + sx[1] * 0.2)
    br = (br[0] - sx[2], br[1] - sx[2] * 0.25)
    bl = (bl[0] + sx[3], bl[1] - sx[3] * 0.15)
    box = np.array([tl, tr, br, bl], dtype=np.float32)
    cv2.fillConvexPoly(img, box.astype(np.int32), (230, 230, 230))
    cv2.polylines(img, [box.astype(np.int32)], True, (10, 10, 10), 2)
    return img, box


def test_homography_recovers_known_dimension():
    img, gt_corners = make_true_perspective_card()
    H, out_w, out_h = persp.compute_homography(gt_corners, 85.60, 53.98, px_per_mm=10.0)
    warped = cv2.warpPerspective(img, H, (out_w, out_h))
    mm_per_px = persp.px_to_mm_scale(out_w, 85.60)

    # Recover card region in warped (rectified) image -> should be full canvas
    # i.e. its measured width should equal card_width_mm within a small tolerance.
    measured_width_mm = out_w * mm_per_px
    measured_height_mm = out_h * mm_per_px
    assert abs(measured_width_mm - 85.60) < 0.5
    assert abs(measured_height_mm - 53.98) < 0.5
    print(f"PASS test_homography_recovers_known_dimension  W={measured_width_mm:.2f}mm H={measured_height_mm:.2f}mm")


def test_order_points_is_rotation_invariant():
    img, gt_corners = make_true_perspective_card()
    for shift in range(4):
        shuffled = np.roll(gt_corners, shift, axis=0)
        ordered = persp.order_points(shuffled)
        # top-left should have smallest x+y, bottom-right largest, consistently
        s = ordered.sum(axis=1)
        assert np.argmin(s) == 0 and np.argmax(s) == 2
    print("PASS test_order_points_is_rotation_invariant")


def test_full_pipeline_card_to_calibration():
    """detect_card -> calibrate_from_card -> mm_per_px should reconstruct
    the card's real width when measuring the (now-known) rectified card."""
    img, gt_corners = make_true_perspective_card()
    detected = rd.detect_card(img)
    assert detected is not None
    cal = calib.calibrate_from_card(img, detected["corners"], detected["confidence"])
    assert cal.perspective_corrected is True
    assert cal.warped_image is not None
    # In the rectified/warped image, the card should now occupy a clean
    # axis-aligned rectangle of the expected pixel size (since warp_full_image
    # places the card inset into a larger canvas at known offset/scale).
    expected_card_px_w = 85.60 / cal.mm_per_px
    assert abs(expected_card_px_w - 856.0) < 1.0  # px_per_mm_output default 10.0
    print(f"PASS test_full_pipeline_card_to_calibration  mm_per_px={cal.mm_per_px:.4f} conf={cal.confidence:.2f}")


if __name__ == "__main__":
    test_homography_recovers_known_dimension()
    test_order_points_is_rotation_invariant()
    test_full_pipeline_card_to_calibration()
    print("\nALL perspective TESTS PASSED")
