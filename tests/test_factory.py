"""Tests the model-backend selection logic (real, runnable -- no ML deps needed)."""
import os
import unittest

from backend.inference import factory


class TestFactory(unittest.TestCase):
    def setUp(self):
        factory.reset_detector_for_tests()
        self._orig_backend = os.environ.get("VISIONAI_MODEL_BACKEND")
        self._orig_allow = os.environ.get("VISIONAI_ALLOW_FALLBACK")

    def tearDown(self):
        factory.reset_detector_for_tests()
        # restore env
        for key, val in [("VISIONAI_MODEL_BACKEND", self._orig_backend),
                          ("VISIONAI_ALLOW_FALLBACK", self._orig_allow)]:
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_explicit_fallback_backend_loads_and_is_labeled_honestly(self):
        from backend import config
        config.MODEL_BACKEND = "fallback"
        try:
            detector = factory.get_detector()
            self.assertEqual(detector.engine_name, "opencv_haarcascade_fallback")
            self.assertFalse(detector.is_real_model)
        finally:
            config.MODEL_BACKEND = "yolo"

    def test_detector_is_cached_singleton(self):
        from backend import config
        config.MODEL_BACKEND = "fallback"
        try:
            d1 = factory.get_detector()
            d2 = factory.get_detector()
            self.assertIs(d1, d2)  # loaded once, not per call (Decision #3)
        finally:
            config.MODEL_BACKEND = "yolo"
    def test_yolo_backend_fails_loudly_without_fallback_opt_in(self):
        from backend import config
        from unittest.mock import patch
        
        config.MODEL_BACKEND = "yolo"
        config.ALLOW_FALLBACK = False
        
        # Mock ultralytics YOLO to raise an error on init (simulating load failure)
        with patch("ultralytics.YOLO", side_effect=Exception("Simulated load failure")):
            try:
                with self.assertRaises(RuntimeError):
                    factory.get_detector()
            finally:
                pass        
 

if __name__ == "__main__":
    unittest.main()
