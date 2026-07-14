"""YOLO-format detection dataset adapter.

This loader supports the common layout used by Ultralytics/KITTI conversions:

```
dataset/
  data.yaml
  images/train/*.jpg
  labels/train/*.txt
  images/val/*.jpg
  labels/val/*.txt
```

Labels are expected as ``class cx cy width height`` in normalized YOLO format.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from perception_isp.core.camerae2e_bridge import CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION, raw_from_camerae2e_rgb, raw_from_rgb_direct
from perception_isp.core.task_types import BoundingBox, EvaluationSample
from perception_isp.core.sample_cache import load_cached_sample, sample_cache_key, save_cached_sample


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_detection_samples(
    dataset: str | Path,
    *,
    split: str = "val",
    limit: Optional[int] = None,
    offset: int = 0,
    width: int = 320,
    height: int = 240,
    cfa_pattern: str = "auto",
    use_camerae2e: bool = True,
    progress_interval: int = 0,
    progress_label: str = "load_yolo_detection_samples",
    cache_dir: str | Path | None = None,
) -> Tuple[EvaluationSample, ...]:
    """Load a YOLO-format detection dataset as PerceptionISP samples."""

    root, config = _resolve_dataset(dataset)
    names = _class_names(config, root)
    image_dir = _split_path(root, config, split, "images")
    label_dir = _label_dir_for_image_dir(root, image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

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
        label_path = _label_path_for_image(label_dir, image_dir, image_path)
        cache_key = sample_cache_key(
            namespace="yolo_detection_dataset",
            image_path=image_path,
            label_path=label_path,
            params={
                "split": split,
                "dataset_index": int(start + index),
                "width": int(width),
                "height": int(height),
                "cfa_pattern": str(cfa_pattern),
                "use_camerae2e": bool(use_camerae2e),
                "camerae2e_native_cfa_bridge_version": CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION if use_camerae2e else "direct",
            },
        )
        sample = load_cached_sample(cache_dir, cache_key)
        if sample is None:
            boxes = _read_yolo_boxes(label_path, names, original_size, (int(width), int(height)))
            raw = (
                raw_from_camerae2e_rgb(rgb, width=int(width), height=int(height), cfa_pattern=cfa_pattern)
                if use_camerae2e
                else raw_from_rgb_direct(rgb, width=int(width), height=int(height), cfa_pattern=cfa_pattern)
            )
            raw.metadata = replace(
                raw.metadata,
                frame_counter=start + index,
                timestamp_us=33333.0 * (start + index),
                camera_id="camerae2e_yolo_dataset_bridge" if use_camerae2e else "direct_yolo_dataset_bridge",
                module_serial=str(image_path),
            )
            sample = EvaluationSample(
                sample_id=image_path.stem,
                raw=raw,
                ground_truth=boxes,
                source="yolo_detection_dataset_camerae2e" if use_camerae2e else "yolo_detection_dataset_direct",
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
                    "raw_provenance": dict(raw.provenance),
                },
                reference_rgb=rgb,
            )
            save_cached_sample(cache_dir, cache_key, sample)
        samples.append(sample)
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


def _resolve_dataset(dataset: str | Path) -> Tuple[Path, Dict[str, Any]]:
    path = Path(dataset).expanduser()
    if path.is_file():
        config = _read_yaml_like(path)
        root = Path(config.get("path", path.parent)).expanduser()
        if not root.is_absolute():
            root = path.parent if root == Path(path.parent.name) else path.parent / root
        return root.resolve(), config
    config_path = path / "data.yaml"
    if config_path.exists():
        config = _read_yaml_like(config_path)
        root = Path(config.get("path", path)).expanduser()
        if not root.is_absolute():
            root = path if root == Path(path.name) else path / root
    else:
        config = {}
        root = path.expanduser()
    return root.resolve(), config


def _read_yaml_like(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text()) or {}
        return dict(payload)
    except Exception:
        return _read_simple_yaml(path)


def _read_simple_yaml(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            payload[key] = items
        else:
            payload[key] = value
    return payload


def _class_names(config: Mapping[str, Any], root: Path | None = None) -> Dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, Mapping):
        parsed = {int(key): str(value) for key, value in names.items()}
        if parsed:
            return parsed
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        parsed = {index: str(value) for index, value in enumerate(names)}
        if parsed:
            return parsed
    if root is not None and str(root.name).lower().startswith("coco"):
        return {index: value for index, value in enumerate(COCO80_CLASS_NAMES)}
    if root is not None and str(root.name).lower().startswith("kitti"):
        return {index: value for index, value in enumerate(KITTI_CLASS_NAMES)}
    return {}


KITTI_CLASS_NAMES = (
    "car",
    "van",
    "truck",
    "pedestrian",
    "Person_sitting",
    "cyclist",
    "tram",
    "misc",
)


COCO80_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _split_path(root: Path, config: Mapping[str, Any], split: str, kind: str) -> Path:
    value = config.get(split)
    if value is None:
        return root / kind / split
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0] if value else f"{kind}/{split}"
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if path.name == split and path.parent.name == kind:
        return path
    if path.name == split and kind not in path.parts:
        return root / kind / split
    return path


def _label_dir_for_image_dir(root: Path, image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    return root / "labels" / image_dir.name


def _label_path_for_image(label_dir: Path, image_dir: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    return (label_dir / relative).with_suffix(".txt")


def _load_rgb(path: Path, *, width: int, height: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float64) / 255.0, original_size


def _read_yolo_boxes(
    path: Path,
    names: Mapping[int, str],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
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
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(value) for value in parts[1:5])
        x1 = (cx - bw / 2.0) * original_w * scale_x
        y1 = (cy - bh / 2.0) * original_h * scale_y
        x2 = (cx + bw / 2.0) * original_w * scale_x
        y2 = (cy + bh / 2.0) * original_h * scale_y
        label = str(names.get(class_id, class_id))
        boxes.append(BoundingBox((x1, y1, x2, y2), label=label))
    return tuple(boxes)
