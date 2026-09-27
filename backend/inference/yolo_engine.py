"""
Real production detector: YOLO26n (falls back to YOLO11n only on genuine
install/compatibility failure -- Decision #9), pretrained on COCO (80
classes), via the `ultralytics` package.

Requires `pip install ultralytics` (pulls in torch). On first use,
`ultralytics` downloads the chosen weights file automatically from
Ultralytics' release assets (needs network access once; the file is then
cached locally, e.g. ~/.cache or the project's working directory,
depending on `ultralytics` version).

This module could not be executed in the sandbox this project was built
in (no outbound network access to install `ultralytics`/`torch` or
download weights -- see README "Known limitations"). The code below
follows the documented `ultralytics` API and the standard pattern for
loading a model once and reusing it across requests; it has not been
run end-to-end here. Verify it works with:
    python -c "from backend.inference.yolo_engine import YoloDetector; d = YoloDetector(); print(d.engine_name)"
once `ultralytics` is installed on a machine with network access.
"""
import logging
import time
from typing import List

import numpy as np

from .. import config
from .base import Detection

logger = logging.getLogger("visionai.yolo_engine")


class YoloDetector:
    is_real_model = True

    def __init__(self):
        try:
            from ultralytics import YOLO  # noqa: imported lazily -- heavy dependency
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` "
                "(see requirements.txt), or set VISIONAI_ALLOW_FALLBACK=true "
                "for local development without it (uses a non-YOLO OpenCV "
                "fallback -- see fallback_engine.py)."
            ) from exc

        last_error = None
        for weights_name in config.YOLO_MODEL_CANDIDATES:
            try:
                logger.info("Loading %s ...", weights_name)
                self._model = YOLO(weights_name)  # auto-downloads on first use
                self.engine_name = weights_name.replace(".pt", "")
                if weights_name != config.YOLO_MODEL_CANDIDATES[0]:
                    logger.warning(
                        "Fell back to %s because %s failed to load: %s",
                        weights_name, config.YOLO_MODEL_CANDIDATES[0], last_error,
                    )
                return
            except Exception as exc:  # noqa: broad -- genuinely any load failure should try the next candidate
                last_error = exc
                logger.warning("Failed to load %s: %s", weights_name, exc)

        raise RuntimeError(
            f"Could not load any YOLO model from {config.YOLO_MODEL_CANDIDATES}. "
            f"Last error: {last_error}"
        )

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=image_bgr,
            imgsz=config.INFERENCE_IMAGE_SIZE,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names  # {class_id: class_name}
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(
                class_name=names[class_id],
                confidence=round(confidence, 4),
                bbox=[x1, y1, x2, y2],
            ))

        return detections
