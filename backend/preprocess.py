"""
Image validation, decoding, and preprocessing. Real, runnable code --
covered by automated tests in this environment (unlike the FastAPI layer,
which needs packages this sandbox couldn't install).

OpenCV (cv2) is used here specifically for: decoding raw upload bytes into
a pixel array (`cv2.imdecode`), color-space conversion (BGR<->RGB, and
BGR->gray for the fallback detector), and resizing -- the standard,
well-documented tool for each of those operations.
"""
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from . import config


class InvalidImageError(ValueError):
    """Raised for any upload that fails validation. Safe to show to the client."""
    pass


@dataclass
class ImageStats:
    width: int
    height: int
    channels: int
    format_guess: str
    brightness_mean: float   # 0-255, mean pixel intensity
    blur_variance: float     # Laplacian variance -- lower often means blurrier


def validate_upload(data: bytes, filename: str, content_type: str) -> None:
    if not data:
        raise InvalidImageError("Uploaded file is empty.")

    if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
        raise InvalidImageError(
            f"File is {len(data) / 1_000_000:.1f} MB, exceeds the "
            f"{config.MAX_UPLOAD_SIZE_BYTES / 1_000_000:.0f} MB limit."
        )

    if content_type not in config.ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"Unsupported content type '{content_type}'. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_CONTENT_TYPES))}."
        )

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}."
        )


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR numpy array (OpenCV convention). Raises InvalidImageError
    if the bytes don't decode as a real image (protects against a file with
    a spoofed .jpg extension that isn't actually valid image data)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("File could not be decoded as a valid image.")
    return img


def compute_image_stats(image_bgr: np.ndarray, format_guess: str) -> ImageStats:
    h, w = image_bgr.shape[:2]
    channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return ImageStats(
        width=w,
        height=h,
        channels=channels,
        format_guess=format_guess,
        brightness_mean=round(float(gray.mean()), 2),
        blur_variance=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    )


def letterbox_resize(image_bgr: np.ndarray, target_size: int = None) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize preserving aspect ratio, padding to a square (target_size x
    target_size) with grey borders -- the standard YOLO preprocessing step.
    Returns (resized_image, scale_factor, (pad_x, pad_y)) so callers can map
    detection boxes back to original image coordinates.
    """
    target_size = target_size or config.INFERENCE_IMAGE_SIZE
    h, w = image_bgr.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # grey pad, YOLO convention
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded, scale, (pad_x, pad_y)
