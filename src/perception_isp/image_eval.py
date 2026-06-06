"""Real-image evaluation samples using CameraE2E RAW generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from .camerae2e_bridge import raw_from_camerae2e_rgb
from .detectors import DetectorAdapter
from .eval_types import BoundingBox, EvaluationSample


DEFAULT_SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def make_yolo_pseudo_label_sample(
    detector: DetectorAdapter,
    *,
    url: str = DEFAULT_SAMPLE_IMAGE_URL,
    cache_dir: str | Path = "data/sample_images",
    width: int = 320,
    height: int = 240,
    cfa_pattern: str = "auto",
) -> EvaluationSample:
    """Download one real image and use detector outputs as pseudo labels.

    This is a detector-consistency smoke test, not ground-truth benchmarking.
    Use KITTI/BDD labels for performance claims.
    """

    image_path = _download(url, Path(cache_dir))
    rgb = _load_rgb(image_path, width=width, height=height)
    pseudo = detector.detect(rgb, input_name="reference_rgb")
    boxes = tuple(BoundingBox(det.box.xyxy, label=det.box.label) for det in pseudo.detections)
    raw = raw_from_camerae2e_rgb(rgb, width=width, height=height, cfa_pattern=cfa_pattern)
    raw.metadata = replace(raw.metadata, camera_id="camerae2e_real_image_bridge", module_serial=f"source={url}")
    return EvaluationSample(
        sample_id="sample_image_yolo_pseudo_0000",
        raw=raw,
        ground_truth=boxes,
        source="sample_image_yolo_pseudo",
        metadata={
            "url": url,
            "image_path": str(image_path),
            "pseudo_label_detector": detector.name,
            "pseudo_label_count": len(boxes),
            "width": int(width),
            "height": int(height),
            "requested_cfa_pattern": cfa_pattern,
            "cfa_pattern": raw.metadata.cfa_pattern,
            "raw_provenance": dict(raw.provenance),
            "ground_truth_warning": "pseudo labels from detector, not human annotations",
        },
        reference_rgb=rgb,
    )


def _download(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = Path(url.split("?")[0]).name or "sample.jpg"
    path = cache_dir / name
    if not path.exists():
        urlretrieve(url, path)
    return path


def _load_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float64) / 255.0
