"""
Draws bounding boxes + labels onto a copy of the original image, and
base64-encodes it for the JSON API response. Real, runnable, tested code.
"""
import base64
from typing import List

import cv2
import numpy as np

from .inference.base import Detection

_BOX_COLOR = (46, 204, 113)   # BGR -- a readable green
_TEXT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        label = f"{det.class_name} {det.confidence * 100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        label_y = max(y1, text_h + 6)
        cv2.rectangle(annotated, (x1, label_y - text_h - 6), (x1 + text_w + 6, label_y), _BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y - 4), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)

    return annotated


def encode_image_base64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode annotated image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")
