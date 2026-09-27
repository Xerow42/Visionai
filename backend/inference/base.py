"""
Detector interface every inference engine implements, so routers/predict.py
never needs to know whether it's talking to the real YOLO engine or the
OpenCV fallback.
"""
from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class Detection:
    class_name: str
    confidence: float  # 0.0-1.0. See fallback_engine.py for what this means
                        # when the fallback engine is active.
    bbox: List[int]     # [x1, y1, x2, y2] in original image pixel coordinates


class Detector(Protocol):
    engine_name: str    # e.g. "yolo26n", "yolo11n", "opencv_haarcascade_fallback"
    is_real_model: bool  # False only for the fallback engine

    def predict(self, image_bgr) -> List[Detection]:
        """image_bgr: numpy array (H, W, 3), BGR order (OpenCV convention)."""
        ...
