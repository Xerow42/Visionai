# Sample images

No photographic sample images are bundled in this repository.

Why: this project's brief specifically asked to "include sample images
only when their licenses permit redistribution." Verifying a redistribution
license for a photo (rather than a purely synthetic image) properly needs
checking the original source and its exact license terms case by case --
that wasn't practical to do reliably for this project, so the safer choice
was to include none rather than risk including one without a clearly
verified license.

For testing:
- `scripts/make_synthetic_test_image.py` generates a purely synthetic
  image (drawn shapes, no photographic content, no copyright concerns) --
  useful for exercising the upload/preprocessing/API pipeline, though it
  won't produce a meaningful YOLO detection since it contains no real
  objects.
- For a real demo with actual detections, use any photo you have the
  rights to (your own photo, or a properly-licensed stock photo from a
  source like Unsplash or Pexels, both of which permit redistribution
  under their respective free licenses -- check the specific photo's
  license before using it in anything beyond local testing).
