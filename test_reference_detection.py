import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vision import reference_detector as rd
from tests.synth import make_card_image, make_coin_image


def test_detect_card_axis_aligned():
    img, gt_box = make_card_image(tilt_deg=0.0)
    result = rd.detect_card(img)
    assert result is not None, "card should be detected"
    assert result["confidence"] > 0.5, f"confidence too low: {result['confidence']}"
    # Check recovered aspect ratio is close to real card aspect.
    corners = result["corners"]
    d = np.linalg.norm(corners[0] - corners[1])
    d2 = np.linalg.norm(corners[1] - corners[2])
    aspect = max(d, d2) / min(d, d2)
    assert abs(aspect - rd.CARD_ASPECT) / rd.CARD_ASPECT < 0.1, f"aspect off: {aspect}"
    print("PASS test_detect_card_axis_aligned  confidence=%.2f aspect=%.3f" % (result["confidence"], aspect))


def test_detect_card_rotated():
    img, gt_box = make_card_image(tilt_deg=27.0)
    result = rd.detect_card(img)
    assert result is not None, "rotated card should still be detected"
    assert result["confidence"] > 0.4
    print("PASS test_detect_card_rotated  confidence=%.2f" % result["confidence"])


def test_detect_card_rejects_wrong_aspect():
    # A square should NOT be accepted as a card.
    img = np.full((600, 600, 3), 40, dtype=np.uint8)
    import cv2
    cv2.rectangle(img, (150, 150), (450, 450), (230, 230, 230), -1)
    cv2.rectangle(img, (150, 150), (450, 450), (10, 10, 10), 2)
    result = rd.detect_card(img)
    assert result is None, "square should not match card aspect ratio"
    print("PASS test_detect_card_rejects_wrong_aspect")


def test_detect_coin():
    known_diameter_mm = 24.26
    img, center, radius_px = make_coin_image(diameter_mm=known_diameter_mm, px_per_mm=6.0)
    result = rd.detect_coin(img)
    assert result is not None, "coin should be detected"
    err_pct = abs(result["diameter_px"] - radius_px * 2) / (radius_px * 2) * 100
    assert err_pct < 8, f"diameter error too high: {err_pct:.2f}%"
    print("PASS test_detect_coin  diameter_px=%.1f (gt=%.1f) err=%.2f%% conf=%.2f" %
          (result["diameter_px"], radius_px * 2, err_pct, result["confidence"]))


if __name__ == "__main__":
    test_detect_card_axis_aligned()
    test_detect_card_rotated()
    test_detect_card_rejects_wrong_aspect()
    test_detect_coin()
    print("\nALL reference_detection TESTS PASSED")
