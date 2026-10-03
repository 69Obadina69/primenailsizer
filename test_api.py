import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
import numpy as np
import cv2
from PIL import Image

from main import create_app
from tests.synth import make_card_image, make_coin_image


def _png_bytes(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return buf


def get_client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_health():
    c = get_client()
    r = c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    print("PASS test_health")


def test_references_list():
    c = get_client()
    r = c.get("/api/v1/references")
    assert r.status_code == 200
    data = r.get_json()
    assert "coins" in data and "card" in data
    assert any(c_["id"] == "us_quarter" for c_ in data["coins"])
    print("PASS test_references_list  (%d coins listed)" % len(data["coins"]))


def test_measure_no_image():
    c = get_client()
    r = c.post("/api/v1/measure", data={})
    assert r.status_code == 400
    body = r.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_IMAGE"
    print("PASS test_measure_no_image")


def test_measure_too_dark_rejected():
    c = get_client()
    # Dark but with texture/noise so the blur check doesn't also trigger -
    # isolates the exposure check specifically.
    rng = np.random.default_rng(0)
    dark_img = (rng.normal(loc=8, scale=4, size=(1000, 1000, 3))
                .clip(0, 255).astype(np.uint8))
    r = c.post("/api/v1/measure", data={
        "image": (_png_bytes(dark_img), "dark.png"),
        "reference_type": "credit_card",
    }, content_type="multipart/form-data")
    body = r.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "IMAGE_TOO_DARK"
    print("PASS test_measure_too_dark_rejected ->", body["error"]["code"])


def test_measure_no_card_found():
    c = get_client()
    # Well-exposed, sharp-ish image but with NO card in it.
    img = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.3)
    r = c.post("/api/v1/measure", data={
        "image": (_png_bytes(img), "noise.png"),
        "reference_type": "credit_card",
    }, content_type="multipart/form-data")
    body = r.get_json()
    assert body["success"] is False
    assert body["error"]["code"] in ("REFERENCE_NOT_FOUND", "IMAGE_TOO_BLURRY", "IMAGE_TOO_DARK", "IMAGE_TOO_BRIGHT")
    print("PASS test_measure_no_card_found ->", body["error"]["code"])


def test_measure_card_found_hand_model_unavailable():
    """This sandbox has no network access to download the MediaPipe
    hand_landmarker.task model, so the pipeline should get all the way
    through image-quality + card detection + calibration, then fail
    gracefully with a clear, structured error rather than crashing."""
    c = get_client()
    img, gt_box = make_card_image(canvas_size=(1400, 1800), tilt_deg=8.0, px_per_mm=8.0)
    r = c.post("/api/v1/measure", data={
        "image": (_png_bytes(img), "card.png"),
        "reference_type": "credit_card",
    }, content_type="multipart/form-data")
    body = r.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "HAND_MODEL_UNAVAILABLE", body
    print("PASS test_measure_card_found_hand_model_unavailable -> pipeline reached hand-detection stage "
          "(quality+card-detection+calibration all succeeded) before failing cleanly on missing model")


def test_measure_coin_without_diameter_rejected():
    c = get_client()
    img, center, r_px = make_coin_image(diameter_mm=24.26, px_per_mm=6.0)
    r = c.post("/api/v1/measure", data={
        "image": (_png_bytes(img), "coin.png"),
        "reference_type": "coin",
    }, content_type="multipart/form-data")
    body = r.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "REFERENCE_DIAMETER_UNKNOWN"
    print("PASS test_measure_coin_without_diameter_rejected")


def test_measure_coin_with_reference_id_reaches_hand_stage():
    c = get_client()
    img, center, r_px = make_coin_image(diameter_mm=24.26, px_per_mm=6.0)
    r = c.post("/api/v1/measure", data={
        "image": (_png_bytes(img), "coin.png"),
        "reference_type": "coin",
        "reference_id": "us_quarter",
    }, content_type="multipart/form-data")
    body = r.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "HAND_MODEL_UNAVAILABLE", body
    print("PASS test_measure_coin_with_reference_id_reaches_hand_stage")


if __name__ == "__main__":
    test_health()
    test_references_list()
    test_measure_no_image()
    test_measure_too_dark_rejected()
    test_measure_no_card_found()
    test_measure_card_found_hand_model_unavailable()
    test_measure_coin_without_diameter_rejected()
    test_measure_coin_with_reference_id_reaches_hand_stage()
    print("\nALL API integration TESTS PASSED")
