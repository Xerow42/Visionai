"""
OpenCV Haar-cascade fallback detector.

THIS IS NOT THE YOLO26n MODEL. It exists only because this project was
built in a sandboxed environment with no outbound network access, so
`pip install ultralytics torch` and the automatic download of YOLO's
pretrained weights were both impossible (see README "Known limitations").
OpenCV ships these Haar-cascade classifier files locally with the
`opencv-python` package -- no download required -- which makes them the
only real, runnable pretrained detector available in that environment.

Differences from the real YOLO26n engine, stated plainly:
  - Detects only 2 classes (face, full_body) via classical Haar features,
    not the 80 COCO classes a real YOLO model would detect.
  - Haar cascades do not produce a calibrated probability. The
    "confidence" returned here is a heuristic derived from OpenCV's
    `detectMultiScale` neighbor count, min-max normalized to fall in the
    displayable 0-1 range. It is NOT a model confidence score and must
    never be presented or compared as one.
  - Every Detection this engine returns is traceable back to
    `engine_name = "opencv_haarcascade_fallback"`, and every API response
    that used it includes an explicit warning field -- see
    backend/routers/predict.py.

Use `backend/inference/yolo_engine.py` for the real, production detector.
"""
import logging
from typing import List

import cv2
import numpy as np

from .base import Detection

logger = logging.getLogger("visionai.fallback_engine")


class HaarCascadeFallbackDetector:
    engine_name = "opencv_haarcascade_fallback"
    is_real_model = False

    def __init__(self):
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        body_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"

        self._face_cascade = cv2.CascadeClassifier(face_path)
        self._body_cascade = cv2.CascadeClassifier(body_path)

        if self._face_cascade.empty() or self._body_cascade.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar cascade files -- opencv-python "
                "installation may be incomplete."
            )
        logger.warning(
            "Using OpenCV Haar-cascade FALLBACK detector, not YOLO26n. "
            "This is expected only in environments without network access "
            "to install ultralytics/torch. See fallback_engine.py docstring."
        )

    def _neighbors_to_pseudo_confidence(self, neighbors: int, max_expected: int = 15) -> float:
        """Heuristic only -- not a calibrated probability. See module docstring."""
        return round(min(neighbors, max_expected) / max_expected, 3)

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detections: List[Detection] = []

        faces, _, face_neighbors = self._face_cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), outputRejectLevels=True
        )
        for (x, y, w, h), neighbors in zip(faces, face_neighbors):
            detections.append(Detection(
                class_name="face",
                confidence=self._neighbors_to_pseudo_confidence(int(neighbors)),
                bbox=[int(x), int(y), int(x + w), int(y + h)],
            ))

        bodies = self._body_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 100))
        for (x, y, w, h) in bodies:
            detections.append(Detection(
                class_name="full_body",
                confidence=0.5,  # detectMultiScale (2-value form) exposes no neighbor count
                bbox=[int(x), int(y), int(x + w), int(y + h)],
            ))

        return detections
