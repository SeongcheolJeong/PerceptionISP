"""AODRaw image loader for HumanISP vs PerceptionISP evaluation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .eval_types import BoundingBox, EvaluationSample
from .types import CalibrationProfile, RawFrame, SensorMetadata


DEFAULT_AODRAW_CFA_PATTERN = "RGGB"


def load_aodraw_detection_samples(
    dataset_root: str | Path,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cfa_pattern: str = DEFAULT_AODRAW_CFA_PATTERN,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
    require_srgb: bool = False,
    progress_interval: int = 0,
    progress_label: str = "load_aodraw_detection_samples",
) -> Tuple[EvaluationSample, ...]:
    """Load an AODRaw manifest as PerceptionISP ``EvaluationSample`` objects.

    The official AODRaw downsample/slice RAW files are expected as ``.npy``
    Bayer mosaics. If ``width``/``height`` differ from the RAW size, the loader
    uses a parity-preserving nearest resample so the CFA pattern remains valid.
    That is useful for fast smoke tests, but native-resolution claim gates
    should keep the RAW size unchanged or use a separately validated binning
    protocol.
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
        sample = load_aodraw_manifest_row(
            root,
            row,
            dataset_index=start + index,
            target_width=width,
            target_height=height,
            cfa_pattern=cfa_pattern,
            black_level=black_level,
            white_level=white_level,
            require_srgb=require_srgb,
        )
        samples.append(sample)
        sample_index = index + 1
        if interval and (sample_index == total or sample_index % interval == 0):
            _print_load_progress(str(progress_label), sample_index, total, started)
    return tuple(samples)


def load_aodraw_manifest_row(
    dataset_root: str | Path,
    row: Mapping[str, Any],
    *,
    dataset_index: int = 0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    cfa_pattern: str = DEFAULT_AODRAW_CFA_PATTERN,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
    require_srgb: bool = False,
) -> EvaluationSample:
    root = Path(dataset_root).expanduser().resolve()
    raw_relative = str(row.get("expected_raw_relative_path", "")).strip()
    if not raw_relative:
        raise ValueError("AODRaw manifest row is missing expected_raw_relative_path")
    raw_path = root / raw_relative
    if not raw_path.is_file():
        raise FileNotFoundError(f"AODRaw RAW file not found: {raw_path}")
    raw_array = np.load(raw_path)
    pattern = _normalize_cfa_pattern(cfa_pattern)
    raw_mosaic, raw_meta = _prepare_raw_mosaic(
        raw_array,
        target_width=target_width,
        target_height=target_height,
        cfa_pattern=pattern,
    )
    height, width = int(raw_mosaic.shape[0]), int(raw_mosaic.shape[1])
    levels = _infer_levels(raw_mosaic, black_level=black_level, white_level=white_level)
    calibration = CalibrationProfile(
        cfa_pattern=pattern,
        black_level=float(levels["black_level"]),
        white_level=float(levels["white_level"]),
    )
    metadata = SensorMetadata(
        camera_id="aodraw_native_loader",
        sensor_id="aodraw_unknown_sensor",
        module_serial=str(raw_path),
        calibration_id="aodraw_inferred_levels_v1",
        isp_profile_id="perception_isp_reference_v1",
        cfa_pattern=pattern,
        hdr_mode="single",
        exposure_times_us=(8000.0,),
        hdr_ratios=(1.0,),
        line_time_us=33333.0 / float(max(height, 1)),
    )
    srgb, srgb_meta = _load_optional_srgb(
        root,
        row,
        target_width=width,
        target_height=height,
        require_srgb=require_srgb,
    )
    boxes = _scale_boxes(row, target_size=(width, height))
    provenance = {
        "bridge": "aodraw_native_raw",
        "raw_source_key": "aodraw.npy",
        "raw_path": str(raw_path),
        "raw_relative_path": raw_relative,
        "raw_input_shape": [int(value) for value in np.asarray(raw_array).shape],
        "target_shape": [height, width],
        "annotation_size": [int(row.get("height", height)), int(row.get("width", width))],
        "source_cfa_pattern": pattern,
        "requested_cfa_pattern": str(cfa_pattern),
        "target_cfa_pattern": pattern,
        "pattern_remapped": False,
        "raw_storage_layout": str(row.get("raw_storage_layout", raw_meta["raw_storage_layout"])),
        "raw_loader_layout": str(raw_meta["raw_storage_layout"]),
        "true_sensor_cfa_mosaic": bool(raw_meta["true_sensor_cfa_mosaic"]),
        "synthetic_cfa_from_rgb": bool(raw_meta["synthetic_cfa_from_rgb"]),
        "native_resolution_matches_target": bool(raw_meta["native_resolution_matches_target"]),
        "native_resolution_at_least_target": bool(raw_meta["native_resolution_at_least_target"]),
        "raw_resize_mode": str(raw_meta["raw_resize_mode"]),
        "camerae2e_used": False,
        "black_level_source": str(levels["black_level_source"]),
        "white_level_source": str(levels["white_level_source"]),
    }
    if srgb_meta:
        provenance.update(srgb_meta)
    return EvaluationSample(
        sample_id=Path(str(row.get("file_name", raw_path.stem))).stem,
        raw=RawFrame(data=raw_mosaic, metadata=metadata, calibration=calibration, provenance=provenance),
        ground_truth=boxes,
        source="aodraw_native_raw_manifest",
        metadata={
            "dataset": "AODRaw",
            "dataset_index": int(dataset_index),
            "image_id": int(row.get("image_id", -1)),
            "file_name": str(row.get("file_name", "")),
            "selection_condition": str(row.get("selection_condition", "")),
            "tags": list(row.get("tags", ()) or ()),
            "raw_path": str(raw_path),
            "srgb_path": srgb_meta.get("srgb_path", "") if srgb_meta else "",
            "original_width": int(row.get("width", width)),
            "original_height": int(row.get("height", height)),
            "width": width,
            "height": height,
            "cfa_pattern": pattern,
            "raw_resize_mode": str(raw_meta["raw_resize_mode"]),
            "native_resolution_matches_target": bool(raw_meta["native_resolution_matches_target"]),
            "raw_storage_layout": str(raw_meta["raw_storage_layout"]),
            "true_sensor_cfa_mosaic": bool(raw_meta["true_sensor_cfa_mosaic"]),
            "raw_provenance": provenance,
        },
        reference_rgb=srgb,
    )


def _load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Path)):
        return [dict(row) for row in manifest if isinstance(row, Mapping)]
    path = Path(manifest).expanduser()
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("manifest", ())
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"AODRaw manifest must be a JSON list or contain a manifest list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _prepare_raw_mosaic(
    raw_array: Any,
    *,
    target_width: Optional[int],
    target_height: Optional[int],
    cfa_pattern: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mosaic, layout_meta = _extract_single_mosaic(raw_array, cfa_pattern=cfa_pattern)
    native_h, native_w = int(mosaic.shape[0]), int(mosaic.shape[1])
    requested_h = native_h if target_height is None or int(target_height) <= 0 else int(target_height)
    requested_w = native_w if target_width is None or int(target_width) <= 0 else int(target_width)
    if requested_h < 2 or requested_w < 2:
        raise ValueError("AODRaw target dimensions must be at least 2x2")
    if (requested_h, requested_w) == (native_h, native_w):
        return mosaic.astype(np.float64, copy=False), {
            **layout_meta,
            "raw_resize_mode": "native",
            "native_resolution_matches_target": True,
            "native_resolution_at_least_target": True,
        }
    resized = _resize_cfa_nearest(mosaic, height=requested_h, width=requested_w)
    return resized, {
        **layout_meta,
        "raw_resize_mode": "nearest_cfa_preserve_parity",
        "native_resolution_matches_target": False,
        "native_resolution_at_least_target": bool(native_h >= requested_h and native_w >= requested_w),
    }


def _extract_single_mosaic(raw_array: Any, *, cfa_pattern: str) -> tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(raw_array)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float64), {
            "raw_storage_layout": "single_channel_bayer_mosaic",
            "true_sensor_cfa_mosaic": True,
            "synthetic_cfa_from_rgb": False,
        }
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.asarray(arr[:, :, 0], dtype=np.float64), {
            "raw_storage_layout": "single_channel_bayer_mosaic_hwc1",
            "true_sensor_cfa_mosaic": True,
            "synthetic_cfa_from_rgb": False,
        }
    if arr.ndim == 3 and arr.shape[0] == 1:
        return np.asarray(arr[0], dtype=np.float64), {
            "raw_storage_layout": "single_channel_bayer_mosaic_chw1",
            "true_sensor_cfa_mosaic": True,
            "synthetic_cfa_from_rgb": False,
        }
    if arr.ndim == 3 and arr.shape[0] == 4:
        return _unpack_aodraw_packed_bayer_chw(arr), {
            "raw_storage_layout": "packed_bayer_chw4",
            "true_sensor_cfa_mosaic": True,
            "synthetic_cfa_from_rgb": False,
        }
    if arr.ndim == 3 and arr.shape[0] == 3:
        return _rgb_to_cfa_mosaic(np.moveaxis(arr, 0, -1), pattern=cfa_pattern), {
            "raw_storage_layout": "demosaiced_rgb_chw3_to_synthetic_bayer",
            "true_sensor_cfa_mosaic": False,
            "synthetic_cfa_from_rgb": True,
        }
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _unpack_aodraw_packed_bayer_chw(np.moveaxis(arr, -1, 0)), {
            "raw_storage_layout": "packed_bayer_hwc4",
            "true_sensor_cfa_mosaic": True,
            "synthetic_cfa_from_rgb": False,
        }
    if arr.ndim == 3 and arr.shape[2] == 3:
        return _rgb_to_cfa_mosaic(arr, pattern=cfa_pattern), {
            "raw_storage_layout": "demosaiced_rgb_hwc3_to_synthetic_bayer",
            "true_sensor_cfa_mosaic": False,
            "synthetic_cfa_from_rgb": True,
        }
    raise ValueError(
        "AODRaw loader expects a 2-D Bayer mosaic, official 4-channel packed Bayer, "
        f"or official 3-channel demosaiced RAW-RGB .npy, got shape {arr.shape}"
    )


def _unpack_aodraw_packed_bayer_chw(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] != 4:
        raise ValueError(f"AODRaw packed Bayer must be 4xHxW, got {arr.shape}")
    _, h, w = arr.shape
    out = np.zeros((int(h) * 2, int(w) * 2), dtype=np.float64)
    out[0::2, 0::2] = arr[0]
    out[0::2, 1::2] = arr[1]
    out[1::2, 1::2] = arr[2]
    out[1::2, 0::2] = arr[3]
    return out


def _rgb_to_cfa_mosaic(values: Any, *, pattern: str) -> np.ndarray:
    rgb = np.asarray(values, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"AODRaw RGB-to-CFA conversion expects HxWx3, got {rgb.shape}")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    mosaic = np.zeros((h, w), dtype=np.float64)
    tiles = {
        "RGGB": ((0, 1), (1, 2)),
        "GRBG": ((1, 0), (2, 1)),
        "GBRG": ((1, 2), (0, 1)),
        "BGGR": ((2, 1), (1, 0)),
    }
    tile = tiles.get(_normalize_cfa_pattern(pattern), tiles["RGGB"])
    for row_offset in range(2):
        for col_offset in range(2):
            mosaic[row_offset::2, col_offset::2] = rgb[row_offset::2, col_offset::2, tile[row_offset][col_offset]]
    return mosaic


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


def _infer_levels(raw_array: Any, *, black_level: Optional[float], white_level: Optional[float]) -> Dict[str, Any]:
    arr = np.asarray(raw_array)
    arr_float = arr.astype(np.float64, copy=False)
    finite = arr_float[np.isfinite(arr_float)]
    observed_max = float(np.max(finite)) if finite.size else 1.0
    if black_level is None:
        inferred_black = 0.0
        black_source = "inferred_zero"
    else:
        inferred_black = float(black_level)
        black_source = "user"
    if white_level is None:
        if observed_max <= 1.5:
            inferred_white = 1.0
            white_source = "observed_unit_range"
        elif observed_max <= 255.0:
            inferred_white = 255.0
            white_source = "observed_8bit_range"
        elif observed_max <= 4095.0:
            inferred_white = 4095.0
            white_source = "observed_12bit_range"
        elif observed_max <= 16383.0:
            inferred_white = 16383.0
            white_source = "observed_14bit_range"
        elif np.issubdtype(arr.dtype, np.integer):
            inferred_white = float(np.iinfo(arr.dtype).max)
            white_source = f"dtype_{arr.dtype}"
        else:
            inferred_white = observed_max
            white_source = "observed_max"
    else:
        inferred_white = float(white_level)
        white_source = "user"
    if inferred_white <= inferred_black:
        raise ValueError("AODRaw white_level must be greater than black_level")
    return {
        "black_level": inferred_black,
        "white_level": inferred_white,
        "black_level_source": black_source,
        "white_level_source": white_source,
    }


def _load_optional_srgb(
    root: Path,
    row: Mapping[str, Any],
    *,
    target_width: int,
    target_height: int,
    require_srgb: bool,
) -> tuple[np.ndarray | None, Dict[str, Any]]:
    relative = str(row.get("expected_srgb_relative_path", "")).strip()
    if not relative:
        if require_srgb:
            raise ValueError("AODRaw manifest row is missing expected_srgb_relative_path")
        return None, {}
    path = root / relative
    if not path.is_file():
        if require_srgb:
            raise FileNotFoundError(f"AODRaw sRGB file not found: {path}")
        return None, {"srgb_path": str(path), "srgb_available": False}
    image = Image.open(path).convert("RGB")
    original_size = image.size
    if original_size != (int(target_width), int(target_height)):
        image = image.resize((int(target_width), int(target_height)), resample=Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return rgb, {
        "srgb_path": str(path),
        "srgb_relative_path": relative,
        "srgb_available": True,
        "srgb_original_size": [int(original_size[0]), int(original_size[1])],
    }


def _scale_boxes(row: Mapping[str, Any], *, target_size: tuple[int, int]) -> Tuple[BoundingBox, ...]:
    target_w, target_h = float(target_size[0]), float(target_size[1])
    source_w = float(row.get("width", target_w) or target_w)
    source_h = float(row.get("height", target_h) or target_h)
    scale_x = target_w / max(source_w, 1.0)
    scale_y = target_h / max(source_h, 1.0)
    boxes: List[BoundingBox] = []
    for item in row.get("boxes", ()) or ():
        if not isinstance(item, Mapping):
            continue
        coords = item.get("xyxy")
        if coords is None or len(coords) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in coords)
        x1 = max(0.0, min(target_w - 1.0, x1 * scale_x))
        x2 = max(0.0, min(target_w - 1.0, x2 * scale_x))
        y1 = max(0.0, min(target_h - 1.0, y1 * scale_y))
        y2 = max(0.0, min(target_h - 1.0, y2 * scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(BoundingBox((x1, y1, x2, y2), label=str(item.get("label", "object"))))
    return tuple(boxes)


def _normalize_cfa_pattern(value: str) -> str:
    pattern = str(value or DEFAULT_AODRAW_CFA_PATTERN).upper().replace("-", "").replace("_", "")
    if pattern not in {"RGGB", "GRBG", "BGGR", "GBRG"}:
        raise ValueError(f"Unsupported AODRaw CFA pattern: {value}")
    return pattern


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
