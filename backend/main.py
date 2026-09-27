"""
FastAPI entrypoint.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import schemas
from .inference.factory import get_detector
from .routers import predict

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="VisionAI API",
    description="Object detection via YOLO26n (pretrained on COCO), with an "
                 "OpenCV fallback engine for environments without network "
                 "access to install ultralytics/torch. See README.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.on_event("startup")
def load_model_once():
    """Model is loaded once here at process startup (Decision #3), not
    per-request. If this fails (e.g. ultralytics not installed and
    VISIONAI_ALLOW_FALLBACK is not set), the app still starts -- /health
    will report model_loaded: false, and /predict will return a clear 500
    rather than crashing the whole server."""
    try:
        get_detector()
    except RuntimeError as exc:
        logging.getLogger("visionai.main").error("Model failed to load at startup: %s", exc)


@app.get("/health", response_model=schemas.HealthResponse)
def health():
    from .inference.factory import _detector_instance
    return schemas.HealthResponse(
        status="ok",
        model_loaded=_detector_instance is not None,
        engine=_detector_instance.engine_name if _detector_instance else None,
    )
