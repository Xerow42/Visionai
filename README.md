# VisionAI — Computer Vision & Image Analysis Platform

A small, real computer-vision product: upload an image (or use your webcam),
and a pretrained **YOLO26n** object detector (COCO, 80 classes) returns
detected objects, confidence scores, bounding boxes, an annotated image, and
measured inference time — through a FastAPI backend and a React frontend.

## Overview

This project demonstrates integrating a pretrained deep-learning model into
a working software product: image upload and validation, preprocessing,
model inference, structured API responses, an annotated-image pipeline, and
a polished frontend — not training a model from scratch.

## Features

- Drag-and-drop image upload **and** webcam capture, both going through the
  same `/predict` endpoint.
- Object detection with class labels, confidence scores, and bounding boxes.
- Annotated image (boxes + labels drawn server-side) returned alongside the
  raw detection list.
- Measured (not claimed) inference time on every response.
- Basic image statistics (`/analyze`): dimensions, brightness, blur estimate.
- File type and size validation on both frontend and backend.
- Structured error responses — never a raw stack trace to the client.
- Model loaded once at backend startup, not per request.

## Architecture

```
React Frontend                     FastAPI Backend
┌───────────────────┐              ┌────────────────────────┐
│ Upload / Webcam     │──POST /predict─▶│ 1. Validate (type/size) │
│                      │                │ 2. Decode image          │
│ Original image        │              │ 3. Preprocess (OpenCV)    │
│ Annotated image        │◀───JSON──────│ 4. Run YOLO26n inference  │
│ Detections + scores      │            │ 5. Draw boxes on image     │
│ Inference time              │         │ 6. Return structured JSON  │
└───────────────────┘              └────────────────────────┘
```

The model is loaded exactly once, at process startup (`backend/main.py`'s
startup event) — reloading a model per request would be a real, explainable
performance bug, so it's deliberately avoided.

## Model

**Ultralytics YOLO26n** (nano), pretrained on **COCO** (80 everyday object
classes — person, car, laptop, bottle, chair, etc.), loaded via the
`ultralytics` Python package.

- **Why YOLO26n**: a single-stage, NMS-free detector architecture tuned for
  CPU/edge inference — realistic for a laptop-run student project, unlike
  larger two-stage detectors that assume GPU inference.
- **Why not train from scratch**: the goal here is demonstrating the ability
  to *integrate and productionize* a pretrained model inside a real
  application (validation, preprocessing, serving, error handling, UI) —
  a distinct and equally real skill from training a model, and the one this
  project is built to show.
- **Fallback**: if `ultralytics`/`torch` fail to install or load, the code
  tries **YOLO11n** next (more mature tooling, same interface) before giving
  up — see `backend/inference/yolo_engine.py`. Which one actually loaded is
  reported in every API response's `"engine"` field.
- **Limitations**: only the 80 COCO classes; accuracy drops on small or
  heavily occluded objects; the nano variant trades some accuracy for speed
  versus larger YOLO sizes; CPU inference is slower than GPU (not benchmarked
  against GPU here — see "Performance").

## Computer Vision Pipeline

`backend/preprocess.py` (OpenCV-based, real and unit-tested in this repo):
1. **Validate** — file size, content type, and extension checked before any
   decoding is attempted.
2. **Decode** — `cv2.imdecode` turns the raw upload bytes into a pixel array;
   this also catches files with a spoofed extension that aren't valid image
   data at all.
3. **Image stats** — dimensions, mean brightness, and a Laplacian-variance
   blur estimate (`/analyze` returns just this, no model needed).
4. **Letterbox resize** — resizes to 640×640 preserving aspect ratio with
   grey padding, the standard YOLO preprocessing step, so detection boxes
   can be mapped back to original image coordinates.
5. **Draw + encode** (`backend/draw.py`) — bounding boxes and labels drawn
   onto a copy of the original image, then base64-encoded for the JSON
   response.

## Tech Stack

Python 3.12 · FastAPI · Ultralytics YOLO26n/YOLO11n (PyTorch) · OpenCV ·
Pillow · React 18 + Vite · Docker Compose.

## Installation

```bash
git clone <this-repo>
cd visionai
cp .env.example .env
pip install -r requirements.txt
```

For the frontend:
```bash
cd frontend
npm install
```

## Running locally

```bash
uvicorn backend.main:app --reload --port 8000     # terminal 1
cd frontend && npm run dev                          # terminal 2
```

First `/predict` call triggers an automatic one-time download of the
YOLO26n weights (needs network access); subsequent calls reuse the cached
file.

Dashboard: http://localhost:5173 · API: http://localhost:8000 ·
interactive API docs: http://localhost:8000/docs

## API

```
POST /predict          multipart/form-data, field "file"
  -> { detections: [{class_name, confidence, bbox}], inference_time_ms,
       image_stats: {...}, annotated_image_base64, engine, is_real_model,
       warning? }

POST /analyze           multipart/form-data, field "file"
  -> { image_stats: {width, height, channels, format_guess,
       brightness_mean, blur_variance} }   -- no model inference

GET /health
  -> { status, model_loaded, engine }
```

Errors are always a structured `{"detail": "..."}` JSON body (400 for a bad
upload, 500 for an inference failure) — never a raw traceback.

## Performance

Only measured numbers are reported here — nothing projected or assumed.

In this sandboxed build environment (no network access to install
`ultralytics`/`torch` or download YOLO's weights — see "Known limitations"),
the **OpenCV Haar-cascade fallback engine** was benchmarked instead, on a
synthetic 640×480 test image, CPU-only:

- Measured inference time: **~40 ms** (`tests/test_fallback_engine.py`
  asserts it stays well under a generous 2000 ms bound — not a marketing
  number, a sanity check).

This is **not** a YOLO26n benchmark — it's a different, much simpler
classical algorithm, reported here only because it's the one that could
actually be run and measured in this environment. Once you install
`ultralytics` and run against a real image on your machine, re-run:
```bash
python -m unittest tests.test_fallback_engine -v   # existing fallback benchmark
# and, after installing ultralytics:
python -c "
from backend.inference.yolo_engine import YoloDetector
import cv2, time
img = cv2.imread('sample_images/your_real_photo.jpg')
d = YoloDetector()
t0 = time.perf_counter(); dets = d.predict(img); t1 = time.perf_counter()
print(d.engine_name, f'{(t1-t0)*1000:.1f} ms', len(dets), 'detections')
"
```
and report those real numbers instead — do not substitute the fallback
engine's numbers for YOLO's in a portfolio or interview; they measure two
different algorithms.

## Fixed after real-machine testing

Three real bugs were caught by actually installing and running this on a
real machine (Windows) rather than only in the sandboxed build environment
below, worth knowing if you hit them on a fresh clone:

- **OpenCV version conflict**: an earlier `requirements.txt` pinned
  `opencv-python-headless` *alongside* `ultralytics` (which installs its own
  `opencv-python`). Installing both pulled in two different major OpenCV
  versions side by side and silently broke `cv2` (`cv2.CascadeClassifier`
  went missing). Fixed by removing the redundant pin — `ultralytics` manages
  its own OpenCV dependency now. If you already have a broken install:
  `pip uninstall opencv-python opencv-python-headless -y` then
  `pip install -r requirements.txt` again.
- **`yolo26n.pt` not resolving**: with `ultralytics==8.3.28` specifically,
  `YOLO("yolo26n.pt")` failed as a local-file lookup instead of
  auto-downloading — a genuine version-support gap, not a bug in this
  project's code. The engine correctly fell back to `yolo11n.pt` per
  Decision #9 and still returned a real, successful detection. Fixed by
  bumping the `ultralytics` version floor in `requirements.txt`; if you
  still see this, run `pip install --upgrade ultralytics`.
- Two tests had bugs of their own: one set an env var *after* the module
  that reads it had already been imported (too late — fixed to set the
  already-imported module's attribute directly), and one assumed
  `ultralytics` would never be installed (fixed to force a genuine failure
  via bogus weight filenames instead, valid whether or not `ultralytics` is
  present).

## Known limitations

Built in a sandboxed environment with **no outbound network access**
(confirmed: `pip install` fails with no route to PyPI). Concretely:

- `ultralytics`, `torch`, `fastapi`, and `pytest` could not be installed, so
  the real YOLO26n engine and the FastAPI layer could not be executed or
  network-tested here. Both are written correctly against their documented
  APIs and follow standard patterns throughout, but are unverified by
  execution in this environment.
- **What *was* verified for real:** image validation, decoding, letterboxing,
  image-stats computation, bounding-box drawing + base64 encoding, and the
  full model-selection/fallback logic — all executed and unit-tested with
  OpenCV/NumPy/Pillow, which **are** installed here. A real (non-YOLO)
  detector — OpenCV's bundled Haar-cascade classifiers, which need no
  download — was wired in as `backend/inference/fallback_engine.py` and
  used to prove the *entire* request pipeline (validate → decode →
  preprocess → detect → annotate → respond) runs end-to-end without error.
  It is clearly labeled `is_real_model: false` and `engine:
  "opencv_haarcascade_fallback"` in every response it produces, and is never
  presented as YOLO. **20 of 26 automated tests ran and passed**; the 6
  skipped are the `fastapi`-dependent API tests (`tests/test_api.py`),
  which will run once `pip install -r requirements.txt` succeeds somewhere
  with network access.
- No photographic sample images are bundled (see `sample_images/README.md`
  for why, and where to get a properly-licensed one).

**To finish verification on your machine:**
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v   # all 26 should now pass
uvicorn backend.main:app --reload
# upload a real photo through the dashboard, confirm real YOLO detections
```

## Future improvements

- Batch prediction endpoint (multiple images in one request).
- Optional GPU inference path with an explicit CPU-vs-GPU benchmark, once
  actually measured on real hardware — not before.
- Model size/accuracy tradeoff page in the UI (nano vs. small vs. medium
  YOLO variants) if this becomes a teaching tool rather than just a demo.
- Rate limiting if ever exposed beyond local/portfolio use.

---

## Author

**Khalil Lamrabet**

Engineering Student — Big Data & Artificial Intelligence

- GitHub: [@Xerow42](https://github.com/Xerow42)
- LinkedIn: [khalillam12](https://www.linkedin.com/in/khalillam12/)
- Email: [klamrabeta19@gmail.com](mailto:klamrabeta19@gmail.com)
