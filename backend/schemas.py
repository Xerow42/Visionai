"""Pydantic response models for the API."""
from typing import List, Optional

from pydantic import BaseModel


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]


class ImageStatsOut(BaseModel):
    width: int
    height: int
    channels: int
    format_guess: str
    brightness_mean: float
    blur_variance: float


class PredictResponse(BaseModel):
    detections: List[DetectionOut]
    inference_time_ms: float
    image_stats: ImageStatsOut
    annotated_image_base64: str
    engine: str
    is_real_model: bool
    warning: Optional[str] = None


class AnalyzeResponse(BaseModel):
    image_stats: ImageStatsOut


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    engine: Optional[str] = None
