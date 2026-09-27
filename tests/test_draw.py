import unittest

import numpy as np

from backend.draw import draw_detections, encode_image_base64
from backend.inference.base import Detection


class TestDraw(unittest.TestCase):
    def test_draw_detections_modifies_image(self):
        img = np.full((300, 400, 3), 220, dtype=np.uint8)
        dets = [Detection(class_name="face", confidence=0.873, bbox=[50, 60, 180, 220])]
        annotated = draw_detections(img, dets)
        self.assertFalse(np.array_equal(img, annotated))
        self.assertEqual(annotated.shape, img.shape)

    def test_no_detections_returns_unmodified_copy(self):
        img = np.full((300, 400, 3), 220, dtype=np.uint8)
        annotated = draw_detections(img, [])
        self.assertTrue(np.array_equal(img, annotated))
        self.assertIsNot(img, annotated)  # must be a copy, not the same array

    def test_encode_base64_roundtrip(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        b64 = encode_image_base64(img)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 0)

        import base64
        raw = base64.b64decode(b64)
        self.assertGreater(len(raw), 0)
        self.assertEqual(raw[:2], b"\xff\xd8")  # JPEG magic bytes


if __name__ == "__main__":
    unittest.main()
