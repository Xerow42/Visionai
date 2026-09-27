"""
Selects and caches the active detector so the model is loaded exactly once
per process (Decision #3), not per request.
"""
import logging

from .. import config
from .base import Detector

logger = logging.getLogger("visionai.factory")

_detector_instance: Detector = None


def get_detector() -> Detector:
    global _detector_instance
    if _detector_instance is not None:
        return _detector_instance

    if config.MODEL_BACKEND == "fallback":
        from .fallback_engine import HaarCascadeFallbackDetector
        _detector_instance = HaarCascadeFallbackDetector()
        return _detector_instance

    try:
        from .yolo_engine import YoloDetector
        _detector_instance = YoloDetector()
    except RuntimeError as exc:
        if config.ALLOW_FALLBACK:
            logger.warning("YOLO engine unavailable (%s); using OpenCV fallback engine because "
                            "VISIONAI_ALLOW_FALLBACK=true.", exc)
            from .fallback_engine import HaarCascadeFallbackDetector
            _detector_instance = HaarCascadeFallbackDetector()
        else:
            raise

    return _detector_instance


def reset_detector_for_tests():
    """Test-only helper: clears the cached singleton so tests can swap backends."""
    global _detector_instance
    _detector_instance = None
