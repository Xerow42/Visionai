"""
API tests using FastAPI's TestClient. Require fastapi + pydantic +
python-multipart to be installed; skipped automatically if not (the case
in the sandbox this project was built in -- see README "Known limitations").
"""
import os
os.environ["VISIONAI_MODEL_BACKEND"] = "fallback"

import unittest
from backend.inference import factory
factory.reset_detector_for_tests()
try:
    import fastapi  # noqa
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["VISIONAI_MODEL_BACKEND"] = "fallback"  # no network needed for tests

        from backend.inference import factory
        factory.reset_detector_for_tests()

        from fastapi.testclient import TestClient
        from backend.main import app
        cls.client = TestClient(app)

    def _synthetic_jpeg(self):
        import cv2
        import numpy as np
        img = np.full((300, 400, 3), 200, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_predict_valid_image(self):
        files = {"file": ("test.jpg", self._synthetic_jpeg(), "image/jpeg")}
        r = self.client.post("/predict", files=files)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("detections", body)
        self.assertIn("inference_time_ms", body)
        self.assertFalse(body["is_real_model"])  # fallback engine in tests
        self.assertIsNotNone(body["warning"])

    def test_predict_rejects_empty_upload(self):
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        r = self.client.post("/predict", files=files)
        self.assertEqual(r.status_code, 400)

    def test_predict_rejects_unsupported_format(self):
        files = {"file": ("test.gif", b"not a real gif either", "image/gif")}
        r = self.client.post("/predict", files=files)
        self.assertEqual(r.status_code, 400)

    def test_predict_rejects_invalid_image_bytes(self):
        files = {"file": ("test.jpg", b"definitely not a jpeg", "image/jpeg")}
        r = self.client.post("/predict", files=files)
        self.assertEqual(r.status_code, 400)

    def test_analyze_valid_image(self):
        files = {"file": ("test.jpg", self._synthetic_jpeg(), "image/jpeg")}
        r = self.client.post("/analyze", files=files)
        self.assertEqual(r.status_code, 200)
        self.assertIn("image_stats", r.json())


if __name__ == "__main__":
    unittest.main()
