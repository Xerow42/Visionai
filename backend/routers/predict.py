"""
POST /predict, POST /analyze.

Errors are always returned as a structured {"detail": "..."} JSON body via
HTTPException -- never a raw stack trace (security requirement).
"""
import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import config, draw, preprocess, schemas
from ..inference.factory import get_detector

logger = logging.getLogger("visionai.predict")
router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=schemas.PredictResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    try:
        preprocess.validate_upload(data, file.filename or "", file.content_type or "")
        image_bgr = preprocess.decode_image(data)
    except preprocess.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stats = preprocess.compute_image_stats(image_bgr, file.content_type or "unknown")

    detector = get_detector()

    t0 = time.perf_counter()
    try:
        detections = detector.predict(image_bgr)
    except Exception as exc:  # noqa: broad -- never leak internals to the client
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. Please try a different image.")
    inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    annotated = draw.draw_detections(image_bgr, detections)
    annotated_b64 = draw.encode_image_base64(annotated)

    warning = None
    if not detector.is_real_model:
        warning = (
            "This response was produced by the OpenCV Haar-cascade FALLBACK "
            "detector, NOT the real YOLO26n model -- see fallback_engine.py. "
            "Detections are limited to 'face'/'full_body' and confidence "
            "values are a heuristic, not a calibrated probability."
        )

    return schemas.PredictResponse(
        detections=[schemas.DetectionOut(class_name=d.class_name, confidence=d.confidence, bbox=d.bbox)
                    for d in detections],
        inference_time_ms=inference_time_ms,
        image_stats=schemas.ImageStatsOut(**vars(stats)),
        annotated_image_base64=annotated_b64,
        engine=detector.engine_name,
        is_real_model=detector.is_real_model,
        warning=warning,
    )


@router.post("/analyze", response_model=schemas.AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """Lightweight endpoint: image stats only, no model inference."""
    data = await file.read()

    try:
        preprocess.validate_upload(data, file.filename or "", file.content_type or "")
        image_bgr = preprocess.decode_image(data)
    except preprocess.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stats = preprocess.compute_image_stats(image_bgr, file.content_type or "unknown")
    return schemas.AnalyzeResponse(image_stats=schemas.ImageStatsOut(**vars(stats)))
