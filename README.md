# Fingernail Segmentation Model

This directory is where a trained YOLO segmentation model for fingernails
should live (`weights.pt` or exported `weights.onnx`). `vision/nail_detector.py`
automatically uses it instead of the GrabCut fallback as soon as one of
those files is present.

## Why it isn't here yet

Training this model requires two things this build environment does not
have:
1. **A labeled fingernail dataset** (images + polygon masks per nail).
   There is no bundled dataset, and no network access to download one
   (e.g. from Roboflow Universe or similar).
2. **A GPU** for a reasonable training time, and network access to
   `pip install ultralytics` (not preinstalled here).

## Training pipeline (to run in an environment with GPU + network)

```bash
pip install ultralytics

# 1. Collect/label data: images of hands with polygon masks around each
#    visible fingernail, in YOLO segmentation format (class 0 = "nail").
#    Roboflow (roboflow.com) is a reasonable place to label + export.

# 2. dataset.yaml
#    train: /path/to/train/images
#    val:   /path/to/val/images
#    names: ["nail"]

# 3. Fine-tune a YOLOv8 segmentation model
yolo segment train model=yolov8n-seg.pt data=dataset.yaml epochs=100 imgsz=640

# 4. Export to ONNX for portable inference
yolo export model=runs/segment/train/weights/best.pt format=onnx

# 5. Copy the result here
cp best.onnx models/nail_segmentation/weights.onnx
```

## Until then

`vision/nail_detector.py` uses MediaPipe hand landmarks to build an
oriented region-of-interest around each fingertip, then refines it with
OpenCV GrabCut. This is explicitly a **fallback**, not a substitute —
every result it produces is tagged `"segmentation_method": "grabcut_fallback"`
and given a capped, lower confidence score, per the project spec's
requirement to never present a heuristic as equivalent to a trained model.
