import io
import traceback
import numpy as np
import cv2
from flask import Blueprint, request, jsonify
from PIL import Image

from vision import quality, reference_detector as rd, calibration as calib
from vision import hand_detector as hd_module
from vision import nail_detector as nd
from vision import geometry, confidence as conf_module, sizing

measure_bp = Blueprint("measure", __name__)

CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98

# Hand detector is expensive to initialize (loads a model) — build once,
# lazily, and reuse across requests. If the model file is missing (see
# hand_detector.py), we surface a clear structured error instead of
# crashing the whole process.
_hand_detector = None
_hand_detector_error = None


def _get_hand_detector():
    global _hand_detector, _hand_detector_error
    if _hand_detector is not None:
        return _hand_detector
    if _hand_detector_error is not None:
        raise _hand_detector_error
    try:
        _hand_detector = hd_module.HandDetector()
        return _hand_detector
    except Exception as e:  # noqa: BLE001
        _hand_detector_error = e
        raise


def _error(code: str, message: str, status: int = 400):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status


def _read_image_from_request():
    if "image" not in request.files:
        return None
    file = request.files["image"]
    data = file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@measure_bp.route("/api/v1/measure", methods=["POST"])
def measure():
    image = _read_image_from_request()
    if image is None:
        return _error("INVALID_IMAGE", "Could not read the uploaded image.")

    reference_type = request.form.get("reference_type", "credit_card")
    reference_id = request.form.get("reference_id")
    custom_diameter_mm = request.form.get("custom_reference_diameter_mm", type=float)
    custom_width_mm = request.form.get("custom_reference_width_mm", type=float)
    custom_height_mm = request.form.get("custom_reference_height_mm", type=float)

    # ---- 1. Image quality gate ----
    q = quality.assess_quality(image)
    if not q["valid"]:
        reason = q["reasons"][0]
        friendly = {
            "resolution_too_low": "Please use a higher-resolution photo.",
            "image_too_blurry": "The photo looks blurry. Hold the camera steady and retake it.",
            "image_too_dark": "The photo is too dark. Use brighter, even lighting.",
            "image_too_bright": "The photo is overexposed. Reduce glare or direct light.",
        }.get(reason, "The photo quality is too low for a reliable measurement.")
        return _error(reason.upper(), friendly)

    # ---- 2. Reference detection + calibration ----
    calibrations = []
    if reference_type == "credit_card":
        card = rd.detect_card(image)
        if card is None:
            return _error("REFERENCE_NOT_FOUND", "We couldn't detect the reference card. "
                          "Make sure the entire card is visible, flat, and well lit.")
        cal = calib.calibrate_from_card(image, card["corners"], card["confidence"],
                                         CARD_WIDTH_MM, CARD_HEIGHT_MM)
        calibrations.append(cal)

    elif reference_type == "coin":
        coin = rd.detect_coin(image)
        if coin is None:
            return _error("REFERENCE_NOT_FOUND", "We couldn't detect the coin. "
                          "Place it flat, fully visible, beside your nails.")
        diameter_mm = custom_diameter_mm
        if diameter_mm is None and reference_id:
            import json, os
            data_path = "data/references.json"
            try:
                with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)) as f:
                    refs = json.load(f)
                match = next((c for c in refs["coins"] if c["id"] == reference_id), None)
                if match:
                    diameter_mm = match["diameter_mm"]
            except FileNotFoundError:
                pass
        if not diameter_mm:
            return _error("REFERENCE_DIAMETER_UNKNOWN",
                          "Select a coin or enter its diameter in millimeters.")
        cal = calib.calibrate_from_circle(coin["diameter_px"], diameter_mm, coin["confidence"], method="coin")
        calibrations.append(cal)

    elif reference_type == "bottle_cap":
        if not custom_diameter_mm:
            return _error("REFERENCE_DIAMETER_UNKNOWN", "Enter the bottle cap's diameter in millimeters.")
        cap = rd.detect_coin(image)  # circular detector reused for caps
        if cap is None:
            return _error("REFERENCE_NOT_FOUND", "We couldn't detect the bottle cap's circular shape.")
        cal = calib.calibrate_from_circle(cap["diameter_px"], custom_diameter_mm, cap["confidence"], method="bottle_cap")
        calibrations.append(cal)

    elif reference_type == "custom":
        if not custom_width_mm or not custom_height_mm:
            return _error("REFERENCE_DIMENSIONS_UNKNOWN", "Enter the reference object's width and height in millimeters.")
        card_like = rd.detect_card(image, expected_aspect=custom_width_mm / custom_height_mm)
        if card_like is None:
            return _error("REFERENCE_NOT_FOUND", "We couldn't detect the custom reference object's outline.")
        cal = calib.calibrate_from_card(image, card_like["corners"], card_like["confidence"],
                                         custom_width_mm, custom_height_mm)
        calibrations.append(cal)
    else:
        return _error("UNKNOWN_REFERENCE_TYPE", f"Unknown reference_type '{reference_type}'.")

    primary_cal = calibrations[0]
    calibration_agreement_pct = None
    secondary_calibration_info = None

    # ---- 2b. Opportunistic secondary-reference cross-validation ----
    # If the primary reference was a coin/bottle-cap, a credit card has a
    # FIXED, known physical size (ISO/IEC 7810 ID-1) — so if one happens to
    # also be visible in frame we can detect it and cross-validate for free,
    # with no extra input from the user. We do NOT do the reverse (opportunistically
    # guessing a coin's identity when the primary reference is a card) because an
    # unselected coin's exact diameter is unknown and guessing it would be exactly
    # the kind of "assume all coins are the same size" mistake the spec forbids.
    if reference_type in ("coin", "bottle_cap"):
        secondary_card = rd.detect_card(image)
        if secondary_card is not None and secondary_card["confidence"] > 0.5:
            # Compare in a common coordinate space: the ORIGINAL image.
            # primary_cal (coin/cap) is measured in original-image pixels;
            # calibrate_from_card's mm_per_px is in its own warped-canvas
            # space and is NOT comparable to that directly (see
            # calibration.original_space_mm_per_px_from_card docstring).
            card_mm_per_px_original = calib.original_space_mm_per_px_from_card(
                secondary_card["corners"], CARD_WIDTH_MM, CARD_HEIGHT_MM)
            proxy_card_cal = calib.Calibration(
                mm_per_px=card_mm_per_px_original,
                method="card_homography",
                confidence=secondary_card["confidence"],
                perspective_corrected=False,  # comparison is in original space, pre-rectification
            )
            cross = calib.cross_validate(primary_cal, proxy_card_cal)
            calibration_agreement_pct = cross["difference_pct"]
            secondary_calibration_info = cross
            if cross["agrees"]:
                # A verified card also gives us full perspective correction,
                # so prefer its (properly warped) calibration for the actual
                # measurement once cross-validated as trustworthy.
                secondary_cal = calib.calibrate_from_card(
                    image, secondary_card["corners"], secondary_card["confidence"],
                    CARD_WIDTH_MM, CARD_HEIGHT_MM,
                )
                primary_cal = secondary_cal

    working_image = primary_cal.warped_image if primary_cal.warped_image is not None else image
    working_mm_per_px = primary_cal.mm_per_px

    # ---- 3. Hand detection ----
    try:
        detector = _get_hand_detector()
    except FileNotFoundError as e:
        return _error(
            "HAND_MODEL_UNAVAILABLE",
            "The hand-detection model isn't installed on this server yet. "
            "Run the one-time model download step described in "
            "backend/vision/hand_detector.py, then retry.",
            status=503,
        )
    except Exception:
        traceback.print_exc()
        return _error("HAND_DETECTOR_ERROR", "Hand detection is temporarily unavailable.", status=503)

    hands = detector.detect(working_image)
    if not hands:
        return _error("HAND_NOT_FOUND", "We couldn't detect a hand in the photo. "
                      "Make sure your whole hand is visible and well lit.")

    hand = hands[0]

    # ---- 4. Nail segmentation + geometry ----
    nail_results = nd.detect_nails(working_image, hand)
    if len(nail_results) < 5:
        return _error("NAILS_NOT_VISIBLE", "We couldn't find all five nails. "
                      "Make sure every fingertip is visible and not curled under.")

    output_nails = []
    for nr in nail_results:
        polygon = np.array(nr["polygon"])
        geo = geometry.measure_width_length(polygon, working_mm_per_px)

        final_confidence = conf_module.nail_confidence(
            segmentation_confidence=nr["confidence"],
            reference_confidence=primary_cal.confidence,
            image_quality_score=q["score"],
            perspective_corrected=primary_cal.perspective_corrected,
            calibration_agreement_pct=calibration_agreement_pct,
        )
        uncertainty_mm = conf_module.estimate_uncertainty_mm(
            mm_per_px=working_mm_per_px,
            segmentation_confidence=nr["confidence"],
            reference_confidence=primary_cal.confidence,
            perspective_corrected=primary_cal.perspective_corrected,
            measurement_px=geo["width_px"],
        )

        output_nails.append({
            "finger": nr["finger"],
            "width_mm": round(geo["width_mm"], 1),
            "length_mm": round(geo["length_mm"], 1),
            "confidence": round(final_confidence, 2),
            "uncertainty_mm": uncertainty_mm,
            "segmentation_method": nr["method"],
        })

    # ---- 5. Confidence gate: never present an overconfident bad result ----
    low_confidence_nails = [n for n in output_nails if n["confidence"] < 0.5]
    if len(low_confidence_nails) >= 3:
        return _error(
            "LOW_CONFIDENCE_MEASUREMENT",
            "We couldn't get a reliable measurement from this photo. "
            "Move the camera directly above your hand, keep the reference "
            "flat beside your nails, and improve lighting, then try again.",
            status=200,
        )

    # ---- 6. Sizing ----
    profiles = sizing.load_profiles()
    profile = profiles.get("standard")
    for n in output_nails:
        n["recommended_size"] = sizing.recommend_size(n["width_mm"], profile)

    return jsonify({
        "success": True,
        "image_quality": {"score": q["score"]},
        "calibration": {
            "reference": primary_cal.method,
            "confidence": round(primary_cal.confidence, 2),
            "perspective_corrected": primary_cal.perspective_corrected,
            "mm_per_px": primary_cal.mm_per_px,
            "cross_validation": secondary_calibration_info,
        },
        "handedness": hand["handedness"],
        "nails": output_nails,
    })
