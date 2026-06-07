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
from .types import CalibrationProfile, RawFrame, SensorMetadata


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


def load_pascalraw_native_detection_samples(
    dataset_root: str | Path,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    progress_interval: int = 0,
    progress_label: str = "load_pascalraw_native_detection_samples",
) -> Tuple[EvaluationSample, ...]:
    """Load full-resolution PASCALRAW NEF files as true Bayer RAW samples."""

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
        sample = load_pascalraw_native_manifest_row(
            root,
            row,
            dataset_index=start + index,
            target_width=width,
            target_height=height,
        )
        samples.append(sample)
        sample_index = index + 1
        if interval and (sample_index == total or sample_index % interval == 0):
            _print_load_progress(str(progress_label), sample_index, total, started)
    return tuple(samples)


def load_pascalraw_native_manifest_row(
    dataset_root: str | Path,
    row: Mapping[str, Any],
    *,
    dataset_index: int = 0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> EvaluationSample:
    root = Path(dataset_root).expanduser().resolve()
    sample_id = str(row.get("sample_id") or Path(str(row.get("file_name", ""))).stem).strip()
    if not sample_id:
        raise ValueError("PASCALRAW native row is missing sample_id")
    raw_relative = str(
        row.get("expected_native_raw_relative_path")
        or row.get("expected_raw_relative_path")
        or f"PASCALRAW/original/raw/{sample_id}.nef"
    )
    raw_path = root / raw_relative
    if not raw_path.is_file():
        raise FileNotFoundError(f"PASCALRAW native NEF not found: {raw_path}")

    raw_mosaic, raw_meta = _load_nef_mosaic(
        raw_path,
        target_width=target_width,
        target_height=target_height,
    )
    height, width = int(raw_mosaic.shape[0]), int(raw_mosaic.shape[1])
    pattern = str(raw_meta["cfa_pattern"])
    black_level = float(raw_meta["black_level"])
    white_level = float(raw_meta["white_level"])
    calibration = CalibrationProfile(cfa_pattern=pattern, black_level=black_level, white_level=white_level)
    metadata = SensorMetadata(
        camera_id="pascalraw_native_nef_loader",
        sensor_id="pascalraw_nikon_rawpy",
        module_serial=str(raw_path),
        calibration_id="pascalraw_rawpy_levels_v1",
        isp_profile_id="perception_isp_reference_v1",
        frame_counter=int(dataset_index),
        timestamp_us=33333.0 * int(dataset_index),
        cfa_pattern=pattern,
        hdr_mode="single",
        exposure_times_us=(8000.0,),
        hdr_ratios=(1.0,),
        line_time_us=33333.0 / float(max(height, 1)),
    )
    boxes = _scale_boxes(row, original_size=(int(row.get("width", width)), int(row.get("height", height))), target_size=(width, height))
    reference_rgb = _load_optional_reference_rgb(root, row, sample_id=sample_id, width=width, height=height)
    provenance = {
        "bridge": "pascalraw_native_nef",
        "raw_source_key": "PASCALRAW/original/raw/*.nef",
        "raw_path": str(raw_path),
        "raw_relative_path": raw_relative,
        "raw_input_shape": list(raw_meta["raw_input_shape"]),
        "target_shape": [height, width],
        "annotation_size": [int(row.get("height", height)), int(row.get("width", width))],
        "source_cfa_pattern": pattern,
        "requested_cfa_pattern": "sensor_native",
        "target_cfa_pattern": pattern,
        "pattern_remapped": False,
        "true_sensor_cfa_mosaic": True,
        "native_raw_input": True,
        "raw_derived_png_input": False,
        "native_resolution_matches_target": bool(raw_meta["native_resolution_matches_target"]),
        "native_resolution_at_least_target": bool(raw_meta["native_resolution_at_least_target"]),
        "raw_resize_mode": str(raw_meta["raw_resize_mode"]),
        "camerae2e_used": False,
        "rawpy_color_desc": str(raw_meta["rawpy_color_desc"]),
        "rawpy_raw_pattern": raw_meta["rawpy_raw_pattern"],
        "black_level_source": "rawpy_black_level_per_channel",
        "white_level_source": "rawpy_white_level",
    }
    return EvaluationSample(
        sample_id=sample_id,
        raw=RawFrame(data=raw_mosaic, metadata=metadata, calibration=calibration, provenance=provenance),
        ground_truth=boxes,
        source="pascalraw_native_nef",
        metadata={
            "dataset": "PASCALRAW",
            "dataset_index": int(dataset_index),
            "file_name": str(row.get("file_name", f"{sample_id}.png")),
            "selection_condition": str(row.get("selection_condition", "native_nef")),
            "tags": [*list(row.get("tags", ()) or ()), "native_nef", "true_sensor_cfa_mosaic"],
            "raw_path": str(raw_path),
            "reference_rgb_path": "" if reference_rgb is None else str(_reference_rgb_path(root, row, sample_id)),
            "original_width": int(row.get("width", width)),
            "original_height": int(row.get("height", height)),
            "width": width,
            "height": height,
            "cfa_pattern": pattern,
            "raw_resize_mode": str(raw_meta["raw_resize_mode"]),
            "native_resolution_matches_target": bool(raw_meta["native_resolution_matches_target"]),
            "raw_provenance": provenance,
        },
        reference_rgb=reference_rgb,
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


def _load_nef_mosaic(
    path: Path,
    *,
    target_width: Optional[int],
    target_height: Optional[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import rawpy  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional local dataset tooling
        raise ImportError("rawpy is required for --pascalraw-native-raw") from exc

    with rawpy.imread(str(path)) as raw:
        mosaic = np.asarray(raw.raw_image_visible, dtype=np.float64)
        raw_shape = tuple(int(value) for value in mosaic.shape)
        pattern = _rawpy_cfa_pattern(raw)
        color_desc = raw.color_desc.decode("ascii", errors="replace") if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
        raw_pattern = [[int(value) for value in row] for row in np.asarray(raw.raw_pattern).tolist()]
        black_levels = [float(value) for value in getattr(raw, "black_level_per_channel", ()) or ()]
        black_level = float(min(black_levels)) if black_levels else 0.0
        white_level = float(getattr(raw, "white_level", 0.0) or np.max(mosaic) or 1.0)

    prepared, resize_meta = _prepare_native_mosaic(
        mosaic,
        target_width=target_width,
        target_height=target_height,
    )
    return prepared, {
        **resize_meta,
        "raw_input_shape": raw_shape,
        "cfa_pattern": pattern,
        "black_level": black_level,
        "white_level": white_level,
        "rawpy_color_desc": color_desc,
        "rawpy_raw_pattern": raw_pattern,
    }


def _rawpy_cfa_pattern(raw: Any) -> str:
    color_desc = raw.color_desc.decode("ascii", errors="replace") if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
    pattern = np.asarray(raw.raw_pattern)
    chars = []
    for value in pattern.reshape(-1):
        index = int(value)
        chars.append(color_desc[index] if 0 <= index < len(color_desc) else "?")
    cfa = "".join(chars).upper()
    if len(cfa) != 4 or "?" in cfa:
        raise ValueError(f"Unsupported rawpy CFA pattern: {cfa!r}")
    return cfa


def _prepare_native_mosaic(
    mosaic: np.ndarray,
    *,
    target_width: Optional[int],
    target_height: Optional[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    native_h, native_w = int(mosaic.shape[0]), int(mosaic.shape[1])
    requested_h = native_h if target_height is None or int(target_height) <= 0 else int(target_height)
    requested_w = native_w if target_width is None or int(target_width) <= 0 else int(target_width)
    if requested_h < 2 or requested_w < 2:
        raise ValueError("PASCALRAW native target dimensions must be at least 2x2")
    if (requested_h, requested_w) == (native_h, native_w):
        return mosaic.astype(np.float64, copy=False), {
            "raw_resize_mode": "native",
            "native_resolution_matches_target": True,
            "native_resolution_at_least_target": True,
        }
    resized = _resize_cfa_nearest(mosaic, height=requested_h, width=requested_w)
    return resized, {
        "raw_resize_mode": "nearest_cfa_preserve_parity",
        "native_resolution_matches_target": False,
        "native_resolution_at_least_target": bool(native_h >= requested_h and native_w >= requested_w),
    }


def _resize_cfa_nearest(mosaic: np.ndarray, *, height: int, width: int) -> np.ndarray:
    source = np.asarray(mosaic, dtype=np.float64)
    source_h, source_w = int(source.shape[0]), int(source.shape[1])
    target_h, target_w = int(height), int(width)
    row_base = np.floor(np.arange(target_h, dtype=np.float64) * source_h / float(target_h)).astype(int)
    col_base = np.floor(np.arange(target_w, dtype=np.float64) * source_w / float(target_w)).astype(int)
    rows = _match_parity(row_base, np.arange(target_h), source_h)
    cols = _match_parity(col_base, np.arange(target_w), source_w)
    return source[np.ix_(rows, cols)].astype(np.float64, copy=False)


def _match_parity(source_indices: np.ndarray, target_indices: np.ndarray, limit: int) -> np.ndarray:
    adjusted = np.asarray(source_indices, dtype=int).copy()
    target = np.asarray(target_indices, dtype=int)
    mismatch = (adjusted % 2) != (target % 2)
    adjusted[mismatch] += 1
    high = adjusted >= int(limit)
    adjusted[high] -= 2
    adjusted = np.clip(adjusted, 0, max(int(limit) - 1, 0))
    return adjusted


def _reference_rgb_path(root: Path, row: Mapping[str, Any], sample_id: str) -> Path:
    reference_relative = str(
        row.get("expected_reference_rgb_relative_path")
        or row.get("expected_original_jpg_relative_path")
        or f"PASCALRAW/original/jpg/{sample_id}.jpg"
    )
    return root / reference_relative


def _load_optional_reference_rgb(
    root: Path,
    row: Mapping[str, Any],
    *,
    sample_id: str,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    path = _reference_rgb_path(root, row, sample_id)
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGB").resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float64) / 255.0


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
