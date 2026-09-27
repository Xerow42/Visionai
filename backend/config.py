"""
Central configuration, env-driven so the same code runs locally, in tests,
and in Docker without modification.
"""
import os

# Model backend selection.
#   "yolo"     -> real YOLO26n (falls back to YOLO11n) via ultralytics.
#                 This is the production path.
#   "fallback" -> OpenCV Haar-cascade based detector. NOT YOLO, NOT trained
#                 on COCO, NOT the model described in the README's "Model"
#                 section. It exists only so the API/pipeline can be
#                 exercised end-to-end in environments where `ultralytics`
#                 and PyTorch cannot be installed (see README "Known
#                 limitations"). Every response produced by the fallback
#                 engine is labeled with "engine": "opencv_haarcascade_fallback"
#                 so a caller can never mistake it for a real YOLO result.
MODEL_BACKEND = os.environ.get("VISIONAI_MODEL_BACKEND", "yolo")

# If MODEL_BACKEND="yolo" but ultralytics/torch aren't installed, should we
# silently degrade to the fallback engine? Default: no (fail loudly) --
# production should never silently swap models. Local dev/test can opt in.
ALLOW_FALLBACK = os.environ.get("VISIONAI_ALLOW_FALLBACK", "false").lower() == "true"

# Per Decision #9: try YOLO26n first, fall back to YOLO11n only on genuine
# install/compatibility failure, and log exactly why.
YOLO_MODEL_CANDIDATES = ["yolo26n.pt", "yolo11n.pt"]

# Inference input size (pixels, square, letterboxed). Standard YOLO default.
INFERENCE_IMAGE_SIZE = 640

# Upload validation
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Detection confidence threshold (only applies to the real YOLO engine --
# the fallback engine does not produce calibrated confidence scores, see
# fallback_engine.py).
CONFIDENCE_THRESHOLD = float(os.environ.get("VISIONAI_CONFIDENCE_THRESHOLD", "0.25"))
