"""PASCALRAW downsampled PNG loader for HumanISP vs PerceptionISP evaluation."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .camerae2e_bridge import CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION, raw_from_camerae2e_rgb, raw_from_rgb_direct
from .eval_types import BoundingBox, EvaluationSample


def load_pascalraw_detection_samples(
    dataset_root: str | Path,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    width: int = 320,
    height: int = 240,
    cfa_pattern: str = "auto",
    use_camerae2e: bool = True,
    progress_interval: int = 0,
    progress_label: str = "load_pascalraw_detection_samples",
) -> Tuple[EvaluationSample, ...]:
    """Load PASCALRAW's downsampled RAW-derived PNG subset as evaluation samples.

    This loader intentionally labels the bridge as RAW-derived remosaic rather
    than native RAW. The public 1 GB PASCALRAW subset is already a PNG image
    release, so it is useful for fast annotated detector sanity checks but not
    for native Bayer/CFA claims.
    """

    root = Path(dataset_root).expanduser().resolve()
    rows = _load_manifest(manifest)
    start = max(int(offset), 0)
    selected = rows[start:]
    if limit is not None:
        selected = selected[: max(int(limit), 0)]

    samples: List[EvaluationSample] = []
    total = int(len(selected))
    interval = max(int(progress_interval), 0)
    started = time.perf_counter()
    for index, row in enumerate(selected):
        sample = load_pascalraw_manifest_row(
            root,
            row,
            dataset_index=start + index,
            target_width=int(width),
            target_height=int(height),
            cfa_pattern=cfa_pattern,
            use_camerae2e=bool(use_camerae2e),
        )
        samples.append(sample)
        sample_index = index + 1
        if interval and (sample_index == total or sample_index % interval == 0):
            _print_load_progress(str(progress_label), sample_index, total, started)
    return tuple(samples)


def load_pascalraw_manifest_row(
    dataset_root: str | Path,
    row: Mapping[str, Any],
    *,
    dataset_index: int = 0,
    target_width: int = 320,
    target_height: int = 240,
    cfa_pattern: str = "auto",
    use_camerae2e: bool = True,
) -> EvaluationSample:
    root = Path(dataset_root).expanduser().resolve()
    image_relative = str(row.get("expected_image_relative_path", "")).strip()
    if not image_relative:
        raise ValueError("PASCALRAW manifest row is missing expected_image_relative_path")
    image_path = root / image_relative
    if not image_path.is_file():
        raise FileNotFoundError(f"PASCALRAW extracted image not found: {image_path}")
    rgb, original_size = _load_rgb(image_path, width=target_width, height=target_height)
    boxes = _scale_boxes(row, original_size=original_size, target_size=(target_width, target_height))
    raw = (
        raw_from_camerae2e_rgb(rgb, width=target_width, height=target_height, cfa_pattern=cfa_pattern)
        if use_camerae2e
        else raw_from_rgb_direct(rgb, width=target_width, height=target_height, cfa_pattern=cfa_pattern)
    )
    bridge_name = "camerae2e_pascalraw_png_bridge" if use_camerae2e else "direct_pascalraw_png_bridge"
    raw.metadata = replace(
        raw.metadata,
        frame_counter=int(dataset_index),
        timestamp_us=33333.0 * int(dataset_index),
        camera_id=bridge_name,
        module_serial=str(image_path),
    )
    raw.provenance.update(
        {
            "dataset": "PASCALRAW",
            "bridge_dataset": "pascalraw_downsampled_png",
            "raw_derived_png_input": True,
            "native_raw_input": False,
            "camerae2e_native_cfa_bridge_version": CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION if use_camerae2e else "direct",
        }
    )
    return EvaluationSample(
        sample_id=str(row.get("sample_id") or image_path.stem),
        raw=raw,
        ground_truth=boxes,
        source="pascalraw_downsampled_png_camerae2e" if use_camerae2e else "pascalraw_downsampled_png_direct",
        metadata={
            "dataset": "PASCALRAW",
            "dataset_index": int(dataset_index),
            "image_path": str(image_path),
            "file_name": str(row.get("file_name", image_path.name)),
            "selection_condition": str(row.get("selection_condition", "")),
            "tags": list(row.get("tags", ()) or ()),
            "original_width": int(original_size[0]),
            "original_height": int(original_size[1]),
            "width": int(target_width),
            "height": int(target_height),
            "requested_cfa_pattern": cfa_pattern,
            "cfa_pattern": raw.metadata.cfa_pattern,
            "use_camerae2e": bool(use_camerae2e),
            "raw_derived_png_input": True,
            "native_raw_input": False,
            "raw_provenance": dict(raw.provenance),
        },
        reference_rgb=rgb,
    )


def _load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Path)):
        return [dict(row) for row in manifest if isinstance(row, Mapping)]
    path = Path(manifest).expanduser()
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("manifest", ())
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"PASCALRAW manifest must be a JSON list or contain a manifest list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _load_rgb(path: Path, *, width: int, height: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float64) / 255.0, original_size


def _scale_boxes(
    row: Mapping[str, Any],
    *,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[BoundingBox, ...]:
    annotation_w = float(row.get("width", original_size[0]) or original_size[0])
    annotation_h = float(row.get("height", original_size[1]) or original_size[1])
    target_w, target_h = float(target_size[0]), float(target_size[1])
    scale_x = target_w / max(annotation_w, 1.0)
    scale_y = target_h / max(annotation_h, 1.0)
    boxes = []
    for payload in row.get("boxes", ()) or ():
        if not isinstance(payload, Mapping):
            continue
        xyxy = payload.get("xyxy", ())
        if not isinstance(xyxy, Sequence) or isinstance(xyxy, (str, bytes)) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        x1 = max(0.0, min(target_w - 1.0, x1 * scale_x))
        x2 = max(0.0, min(target_w - 1.0, x2 * scale_x))
        y1 = max(0.0, min(target_h - 1.0, y1 * scale_y))
        y2 = max(0.0, min(target_h - 1.0, y2 * scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(BoundingBox((x1, y1, x2, y2), label=str(payload.get("label", "object"))))
    return tuple(boxes)


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
