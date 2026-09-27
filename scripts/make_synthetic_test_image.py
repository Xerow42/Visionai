"""
Generates a purely synthetic test image (drawn shapes -- no photographic
content, no copyright concerns) for exercising the upload/preprocess/API
pipeline. It will NOT trigger a meaningful YOLO/Haar-cascade detection
since it contains no real objects -- see sample_images/README.md for where
to get a real photo for an actual demo.
"""
import os

import cv2
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_images", "synthetic_test.jpg")


def build():
    img = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (260, 300), (0, 128, 255), -1)
    cv2.circle(img, (450, 200), 90, (255, 100, 0), -1)
    cv2.putText(img, "SYNTHETIC TEST IMAGE - NOT A PHOTO", (40, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(OUT_PATH, img)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
