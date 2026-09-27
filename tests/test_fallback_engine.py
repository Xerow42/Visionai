"""
Tests the OpenCV Haar-cascade fallback engine for real (it needs no
external download -- see fallback_engine.py docstring for why it exists
and its documented limitations vs. the real YOLO26n model).
"""
import time
import unittest

import cv2
import numpy as np

from backend.inference.fallback_engine import HaarCascadeFallbackDetector


class TestFallbackEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = HaarCascadeFallbackDetector()

    def test_engine_metadata_is_honest(self):
        self.assertEqual(self.detector.engine_name, "opencv_haarcascade_fallback")
        self.assertFalse(self.detector.is_real_model)  # must never claim to be YOLO

    def test_runs_without_error_on_synthetic_image(self):
        img = np.full((300, 400, 3), 210, dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (150, 150), (0, 0, 0), -1)
        detections = self.detector.predict(img)
        self.assertIsInstance(detections, list)
        # A synthetic geometric image has no real face/body -- 0 detections
        # is the correct, honest result, not a bug.
        self.assertEqual(len(detections), 0)

    def test_inference_time_is_measurable_and_fast(self):
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        t0 = time.perf_counter()
        self.detector.predict(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 2000)  # sanity bound, not a marketing claim

    def test_detection_confidence_in_valid_range(self):
        # Even though it's a heuristic (see module docstring), it must stay
        # within the 0-1 range the API contract promises.
        pseudo = self.detector._neighbors_to_pseudo_confidence(20, max_expected=15)
        self.assertGreaterEqual(pseudo, 0.0)
        self.assertLessEqual(pseudo, 1.0)


if __name__ == "__main__":
    unittest.main()
