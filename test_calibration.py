import sys, os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import calibration as calib
from vision import reference_detector as rd
from tests.synth import make_coin_image


def test_coin_calibration_known_diameter():
    known_mm = 24.26
    img, center, radius_px = make_coin_image(diameter_mm=known_mm, px_per_mm=6.0)
    detected = rd.detect_coin(img)
    assert detected is not None
    cal = calib.calibrate_from_circle(detected["diameter_px"], known_mm, detected["confidence"])
    expected_mm_per_px = 1.0 / 6.0
    err_pct = abs(cal.mm_per_px - expected_mm_per_px) / expected_mm_per_px * 100
    assert err_pct < 8, f"calibration error too high: {err_pct:.2f}%"
    assert cal.perspective_corrected is False
    print(f"PASS test_coin_calibration_known_diameter  mm_per_px={cal.mm_per_px:.4f} err={err_pct:.2f}%")


def test_cross_validation_agrees():
    cal_a = calib.Calibration(mm_per_px=0.1000, method="card_homography", confidence=0.9, perspective_corrected=True)
    cal_b = calib.Calibration(mm_per_px=0.1010, method="coin", confidence=0.8, perspective_corrected=False)
    result = calib.cross_validate(cal_a, cal_b)
    assert result["agrees"] is True
    assert result["difference_pct"] < 5
    print(f"PASS test_cross_validation_agrees  diff={result['difference_pct']:.2f}%")


def test_cross_validation_disagrees():
    cal_a = calib.Calibration(mm_per_px=0.1000, method="card_homography", confidence=0.9, perspective_corrected=True)
    cal_b = calib.Calibration(mm_per_px=0.1300, method="coin", confidence=0.8, perspective_corrected=False)
    result = calib.cross_validate(cal_a, cal_b)
    assert result["agrees"] is False
    assert "disagree" in result["message"]
    print(f"PASS test_cross_validation_disagrees  diff={result['difference_pct']:.2f}%")


def test_card_and_coin_same_scene_agree_in_original_space():
    """Regression test for a real bug found during dev: comparing a card's
    WARPED-canvas mm_per_px (fixed by output resolution, e.g. always 0.1 at
    px_per_mm_output=10) directly against a coin's ORIGINAL-image mm_per_px
    is apples-to-oranges and falsely reports large disagreement even on a
    perfectly consistent scene. original_space_mm_per_px_from_card() fixes
    this by measuring the card in the same (original-image) coordinate
    space as the coin, so a geometrically consistent scene should agree.
    """
    from tests.synth import _add_texture
    canvas = np.full((1400, 1800, 3), 150, dtype=np.uint8)
    px_per_mm = 8.0
    card_w, card_h = 85.60 * px_per_mm, 53.98 * px_per_mm
    cv2.rectangle(canvas, (100, 100), (int(100 + card_w), int(100 + card_h)), (230, 230, 230), -1)
    cv2.rectangle(canvas, (100, 100), (int(100 + card_w), int(100 + card_h)), (10, 10, 10), 2)
    coin_r = (24.26 * px_per_mm) / 2
    coin_center = (int(100 + card_w + 150), int(100 + card_h / 2))
    cv2.circle(canvas, coin_center, int(coin_r), (210, 180, 100), -1)
    cv2.circle(canvas, coin_center, int(coin_r), (60, 45, 20), 2)
    canvas = _add_texture(canvas)

    coin = rd.detect_coin(canvas)
    coin_cal = calib.calibrate_from_circle(coin["diameter_px"], 24.26, coin["confidence"], method="coin")

    card = rd.detect_card(canvas)
    card_mm_per_px_orig = calib.original_space_mm_per_px_from_card(card["corners"], 85.60, 53.98)
    proxy = calib.Calibration(mm_per_px=card_mm_per_px_orig, method="card_homography",
                               confidence=card["confidence"], perspective_corrected=False)

    cross = calib.cross_validate(coin_cal, proxy)
    assert cross["agrees"] is True, f"expected agreement on a consistent scene, got {cross}"
    assert cross["difference_pct"] < 5
    print(f"PASS test_card_and_coin_same_scene_agree_in_original_space  diff={cross['difference_pct']:.2f}%")


if __name__ == "__main__":
    test_coin_calibration_known_diameter()
    test_cross_validation_agrees()
    test_cross_validation_disagrees()
    test_card_and_coin_same_scene_agree_in_original_space()
    print("\nALL calibration TESTS PASSED")
