# VisionAI — Interview Prep

**CNNs** — Convolutional layers learn spatially-local filters (edges,
textures, then higher-level shapes as depth increases) with shared weights
across the image, which is why CNNs generalize to object position far
better than a fully-connected network would, and with far fewer parameters.

**Object detection vs. classification** — Classification answers "what is
in this image" (one label). Detection answers "what is in this image, and
where" (label + bounding box, potentially multiple objects). This project
is deliberately detection, not classification, because the brief wanted
localization (bounding boxes), a materially different and harder problem.

**Confidence scores** — YOLO's confidence combines objectness (is there an
object here at all) and class probability (which class). It's a threshold
I control (`VISIONAI_CONFIDENCE_THRESHOLD`, default 0.25) — trading off
false positives against missed detections. I can explain that raising the
threshold reduces false positives but risks missing real objects, and vice
versa.

**Inference** — Preprocess (letterbox resize to 640×640, normalize) → single
forward pass through the network → decode raw predictions into boxes +
class scores → threshold by confidence → (YOLO26 specifically is NMS-free
by design, unlike older YOLO versions which needed a separate
non-max-suppression step to remove duplicate overlapping boxes).

**Preprocessing** — Letterbox resize (preserve aspect ratio, pad with grey
to a square) rather than a naive stretch-resize, because stretching distorts
object proportions and can hurt detection accuracy; letterboxing keeps
proportions correct at the cost of some unused padding area.

**Model selection** — Chose YOLO26n specifically for CPU-friendly nano-size
inference speed over larger/more accurate variants, because this runs on a
laptop, not a GPU server — a deliberate speed/accuracy tradeoff I can
justify, not a default I didn't think about. Documented a real fallback
plan (YOLO11n) for genuine compatibility issues, decided in advance rather
than improvised.

**Overfitting** — Not directly a concern here since I'm using pretrained
COCO weights, not training my own model — but I can explain it generally:
a model that memorizes training data patterns (including noise) rather than
learning generalizable features, showing as a gap between training and
validation/test performance.

**Transfer learning** — What I'm doing here is closer to "off-the-shelf
inference" than fine-tuning, but I can articulate the difference: transfer
learning would mean taking YOLO's pretrained weights and fine-tuning on a
custom, smaller dataset for classes COCO doesn't cover — a natural next
step for this project I chose not to do here to keep scope honest.

**CPU vs. GPU** — CPU inference is what I actually measured (see README
"Performance" — with the honest caveat about which engine was actually
benchmarked in the sandboxed build environment). GPU would parallelize the
convolution operations far more aggressively; the nano model size was
specifically chosen because it's usable on CPU without GPU access, a
deliberate constraint-driven choice.

**Latency** — Reported only measured numbers, never projected ones (see
README). Real product-level latency work would include batching, using
ONNX Runtime or TensorRT for faster CPU/GPU execution respectively, and
reducing input resolution if accuracy allows.

**Model optimization** — Beyond model *size* selection (nano vs. larger
YOLO variants), real next steps would be quantization (INT8) and exporting
to ONNX or TensorRT for faster inference — deliberately out of scope here
to keep the project's claims honest and matched to what was actually done.
