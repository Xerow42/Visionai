import unittest

import cv2
import numpy as np

from backend.preprocess import (
    InvalidImageError, validate_upload, decode_image,
    compute_image_stats, letterbox_resize,
)
from backend import config


def _synthetic_jpeg_bytes(w=640, h=480):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (0, 128, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestValidateUpload(unittest.TestCase):
    def test_rejects_empty_file(self):
        with self.assertRaises(InvalidImageError):
            validate_upload(b"", "test.jpg", "image/jpeg")

    def test_rejects_oversized_file(self):
        big = b"0" * (config.MAX_UPLOAD_SIZE_BYTES + 1)
        with self.assertRaises(InvalidImageError):
            validate_upload(big, "test.jpg", "image/jpeg")

    def test_rejects_unsupported_content_type(self):
        with self.assertRaises(InvalidImageError):
            validate_upload(_synthetic_jpeg_bytes(), "test.gif", "image/gif")

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(InvalidImageError):
            validate_upload(_synthetic_jpeg_bytes(), "test.bmp", "image/jpeg")

    def test_accepts_valid_jpeg(self):
        validate_upload(_synthetic_jpeg_bytes(), "test.jpg", "image/jpeg")  # no raise


class TestDecodeImage(unittest.TestCase):
    def test_decodes_real_jpeg(self):
        img = decode_image(_synthetic_jpeg_bytes(320, 240))
        self.assertEqual(img.shape, (240, 320, 3))

    def test_rejects_garbage_bytes_despite_jpg_extension(self):
        # Simulates a file renamed to .jpg that isn't actually a valid image.
        with self.assertRaises(InvalidImageError):
            decode_image(b"this is not an image, just text pretending to be one")


class TestImageStats(unittest.TestCase):
    def test_stats_shape_matches_image(self):
        img = decode_image(_synthetic_jpeg_bytes(640, 480))
        stats = compute_image_stats(img, "JPEG")
        self.assertEqual(stats.width, 640)
        self.assertEqual(stats.height, 480)
        self.assertEqual(stats.channels, 3)
        self.assertGreaterEqual(stats.brightness_mean, 0)
        self.assertLessEqual(stats.brightness_mean, 255)


class TestLetterboxResize(unittest.TestCase):
    def test_output_is_square_target_size(self):
        img = decode_image(_synthetic_jpeg_bytes(640, 480))
        resized, scale, pad = letterbox_resize(img, target_size=640)
        self.assertEqual(resized.shape, (640, 640, 3))

    def test_aspect_ratio_preserved_via_scale(self):
        img = decode_image(_synthetic_jpeg_bytes(1280, 480))  # wide image
        resized, scale, (pad_x, pad_y) = letterbox_resize(img, target_size=640)
        self.assertAlmostEqual(scale, 640 / 1280, places=4)
        self.assertGreater(pad_y, 0)  # vertical padding expected for a wide image
        self.assertEqual(pad_x, 0)


if __name__ == "__main__":
    unittest.main()
