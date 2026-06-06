"""KITTI-format detection dataset adapter.

Supported layouts:

```
dataset/
  training/
    image_2/*.png
    label_2/*.txt
```

and compact subsets:

```
dataset/
  image_2/*.png
  label_2/*.txt
```

KITTI labels are expected in the native object-detection text format:
``type truncated occluded alpha left top right bottom ...``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .camerae2e_bridge import raw_from_camerae2e_rgb, raw_from_rgb_direct
from .eval_types import BoundingBox, EvaluationSample


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
KITTI_TO_COMMON_LABEL = {
    "Car": "car",
    "Van": "van",
    "Truck": "truck",
    "Pedestrian": "person",
    "Person_sitting": "person",
    "Cyclist": "cyclist",
    "Tram": "tram",
    "Misc": "object",
}


def load_kitti_detection_samples(
    dataset: str | Path,
    *,
    split: str = "training",
    limit: Optional[int] = None,
    offset: int = 0,
    width: int = 320,
    height: int = 240,
    cfa_pattern: str = "auto",
    use_camerae2e: bool = True,
    include_dontcare: bool = False,
    progress_interval: int = 0,
    progress_label: str = "load_kitti_detection_samples",
) -> Tuple[EvaluationSample, ...]:
    """Load KITTI object-detection labels as PerceptionISP samples."""

    root = Path(dataset).expanduser().resolve()
    image_dir, label_dir = _resolve_kitti_dirs(root, split)
    image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    start = max(int(offset), 0)
    if start:
        image_paths = image_paths[start:]
    if limit is not None:
        image_paths = image_paths[: max(int(limit), 0)]

    samples: List[EvaluationSample] = []
    total = int(len(image_paths))
    interval = max(int(progress_interval), 0)
    started = time.perf_counter()
    for index, image_path in enumerate(image_paths):
        rgb, original_size = _load_rgb(image_path, width=int(width), height=int(height))
        label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
        boxes = _read_kitti_boxes(
            label_path,
            original_size=original_size,
            target_size=(int(width), int(height)),
            include_dontcare=include_dontcare,
        )
        raw = (
            raw_from_camerae2e_rgb(rgb, width=int(width), height=int(height), cfa_pattern=cfa_pattern)
            if use_camerae2e
            else raw_from_rgb_direct(rgb, width=int(width), height=int(height), cfa_pattern=cfa_pattern)
        )
        raw.metadata = replace(
            raw.metadata,
            frame_counter=start + index,
            timestamp_us=33333.0 * (start + index),
            camera_id="camerae2e_kitti_bridge" if use_camerae2e else "direct_kitti_bridge",
            module_serial=str(image_path),
        )
        samples.append(
            EvaluationSample(
                sample_id=image_path.stem,
                raw=raw,
                ground_truth=boxes,
                source="kitti_detection_dataset_camerae2e" if use_camerae2e else "kitti_detection_dataset_direct",
                metadata={
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "original_width": int(original_size[0]),
                    "original_height": int(original_size[1]),
                    "width": int(width),
                    "height": int(height),
                    "requested_cfa_pattern": cfa_pattern,
                    "cfa_pattern": raw.metadata.cfa_pattern,
                    "split": split,
                    "offset": int(start),
                    "dataset_index": int(start + index),
                    "use_camerae2e": bool(use_camerae2e),
                    "include_dontcare": bool(include_dontcare),
                    "raw_provenance": dict(raw.provenance),
                },
                reference_rgb=rgb,
            )
        )
        sample_index = index + 1
        if interval and (sample_index == total or sample_index % interval == 0):
            _print_load_progress(str(progress_label), sample_index, total, started)
    return tuple(samples)


def _print_load_progress(label: str, sample_index: int, total: int, started: float) -> None:
    elapsed = max(time.perf_counter() - float(started), 1.0e-9)
    rate = float(sample_index / elapsed)
    remaining = float((int(total) - int(sample_index)) / max(rate, 1.0e-12))
    print(
        f"[{label}] loaded {sample_index}/{total} samples "
        f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={remaining:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _resolve_kitti_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    if root.name == "image_2":
        image_dir = root
        label_dir = root.parent / "label_2"
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir

    candidates = (
        (root / split / "image_2", root / split / "label_2"),
        (root / "image_2" / split, root / "label_2" / split),
        (root / "image_2", root / "label_2"),
    )
    for image_dir, label_dir in candidates:
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir
    tried = ", ".join(f"{image_dir} + {label_dir}" for image_dir, label_dir in candidates)
    raise FileNotFoundError(f"KITTI image_2/label_2 directories not found. Tried: {tried}")


def _load_rgb(path: Path, *, width: int, height: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float64) / 255.0, original_size


def _read_kitti_boxes(
    path: Path,
    *,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    include_dontcare: bool,
) -> Tuple[BoundingBox, ...]:
    if not path.exists():
        return ()
    original_w, original_h = float(original_size[0]), float(original_size[1])
    target_w, target_h = float(target_size[0]), float(target_size[1])
    scale_x = target_w / max(original_w, 1.0)
    scale_y = target_h / max(original_h, 1.0)
    boxes: List[BoundingBox] = []
    for raw_line in path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 8:
            continue
        raw_label = parts[0]
        if raw_label == "DontCare" and not include_dontcare:
            continue
        left, top, right, bottom = (float(value) for value in parts[4:8])
        x1 = left * scale_x
        y1 = top * scale_y
        x2 = right * scale_x
        y2 = bottom * scale_y
        x1 = max(0.0, min(target_w - 1.0, x1))
        x2 = max(0.0, min(target_w - 1.0, x2))
        y1 = max(0.0, min(target_h - 1.0, y1))
        y2 = max(0.0, min(target_h - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label = KITTI_TO_COMMON_LABEL.get(raw_label, raw_label.lower())
        boxes.append(BoundingBox((x1, y1, x2, y2), label=label))
    return tuple(boxes)
