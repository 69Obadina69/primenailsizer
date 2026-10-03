import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vision import geometry
from tests.synth import make_nail_polygon


def test_geometry_recovers_known_width_length_unrotated():
    poly = make_nail_polygon(width_mm=13.0, length_mm=15.0, px_per_mm=10.0)
    mm_per_px = 0.1
    result = geometry.measure_width_length(poly, mm_per_px)
    assert abs(result["width_mm"] - 13.0) < 0.6, result
    assert abs(result["length_mm"] - 15.0) < 0.6, result
    print(f"PASS unrotated  width={result['width_mm']} length={result['length_mm']}")


def test_geometry_rotated_nail_not_bbox():
    """A nail rotated 35 degrees should still measure correctly via the
    principal-axis method, whereas a naive axis-aligned bounding box would
    over-estimate both width and length."""
    poly = make_nail_polygon(width_mm=12.0, length_mm=16.0, px_per_mm=10.0)
    theta = np.radians(35)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = poly @ R.T

    mm_per_px = 0.1
    result = geometry.measure_width_length(rotated, mm_per_px)

    # naive bbox measurement for comparison
    bbox_w_px = rotated[:, 0].max() - rotated[:, 0].min()
    bbox_h_px = rotated[:, 1].max() - rotated[:, 1].min()
    bbox_w_mm = bbox_w_px * mm_per_px
    bbox_h_mm = bbox_h_px * mm_per_px

    assert abs(result["width_mm"] - 12.0) < 0.7, result
    assert abs(result["length_mm"] - 16.0) < 0.7, result
    # bbox method should be measurably worse (larger error) at this rotation
    bbox_err = abs(min(bbox_w_mm, bbox_h_mm) - 12.0)
    axis_err = abs(result["width_mm"] - 12.0)
    assert axis_err <= bbox_err + 0.05
    print(f"PASS rotated  axis_width={result['width_mm']} (err={axis_err:.2f}) "
          f"vs bbox_width~{min(bbox_w_mm,bbox_h_mm):.2f} (err={bbox_err:.2f})")


def test_geometry_handles_curved_nail():
    poly = make_nail_polygon(width_mm=11.0, length_mm=13.0, px_per_mm=10.0, curvature=0.3)
    result = geometry.measure_width_length(poly, 0.1)
    assert result["width_mm"] > 0 and result["length_mm"] > 0
    assert abs(result["width_mm"] - 11.0) < 1.2
    print(f"PASS curved  width={result['width_mm']} length={result['length_mm']}")


if __name__ == "__main__":
    test_geometry_recovers_known_width_length_unrotated()
    test_geometry_rotated_nail_not_bbox()
    test_geometry_handles_curved_nail()
    print("\nALL measurement/geometry TESTS PASSED")
