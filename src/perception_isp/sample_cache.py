"""Disk cache for dataset EvaluationSample RAW payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .eval_types import BoundingBox, EvaluationSample
from .synthetic import make_synthetic_raw
from .types import RawFrame, SensorMetadata, json_ready, metadata_to_dict


CACHE_VERSION = 1


def sample_cache_key(*, namespace: str, image_path: str | Path, label_path: str | Path, params: Mapping[str, Any]) -> str:
    image = Path(image_path)
    label = Path(label_path)
    payload = {
        "cache_version": CACHE_VERSION,
        "namespace": str(namespace),
        "image_path": str(image.resolve()),
        "image_stat": _stat_payload(image),
        "label_path": str(label.resolve()),
        "label_stat": _stat_payload(label),
        "params": json_ready(dict(params)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_cached_sample(cache_dir: str | Path | None, key: str) -> EvaluationSample | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["payload_json"].item()))
        raw_data = np.asarray(payload["raw_data"], dtype=np.float64)
        reference_rgb = np.asarray(payload["reference_rgb"], dtype=np.float64) if bool(meta.get("has_reference_rgb")) else None
    raw_metadata = _sensor_metadata_from_dict(meta["raw_metadata"])
    raw = RawFrame(
        data=raw_data,
        metadata=raw_metadata,
        calibration=_synthetic_calibration_for_raw(raw_data, raw_metadata.cfa_pattern),
        provenance=dict(meta.get("raw_provenance", {})),
    )
    return EvaluationSample(
        sample_id=str(meta["sample_id"]),
        raw=raw,
        ground_truth=tuple(BoundingBox(tuple(item["xyxy"]), label=str(item["label"])) for item in meta.get("ground_truth", ())),
        source=str(meta["source"]),
        metadata=dict(meta.get("metadata", {})),
        reference_rgb=reference_rgb,
    )


def save_cached_sample(cache_dir: str | Path | None, key: str, sample: EvaluationSample) -> None:
    if cache_dir is None:
        return
    path = _cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "sample_id": sample.sample_id,
        "source": sample.source,
        "ground_truth": [box.to_dict() for box in sample.ground_truth],
        "metadata": dict(sample.metadata),
        "raw_metadata": metadata_to_dict(sample.raw.metadata),
        "raw_provenance": dict(sample.raw.provenance),
        "has_reference_rgb": sample.reference_rgb is not None,
    }
    arrays = {
        "payload_json": np.array(json.dumps(json_ready(payload), sort_keys=True)),
        "raw_data": np.asarray(sample.raw.data),
        "reference_rgb": np.asarray(sample.reference_rgb if sample.reference_rgb is not None else np.zeros((0,), dtype=np.float32)),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir).expanduser() / f"{key}.npz"


def _stat_payload(path: Path) -> Mapping[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False}
    return {
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sensor_metadata_from_dict(payload: Mapping[str, Any]) -> SensorMetadata:
    values = dict(payload)
    for key in ("exposure_times_us", "analog_gains", "digital_gains", "hdr_ratios"):
        if key in values:
            values[key] = tuple(float(value) for value in values[key])
    if "frame_counter" in values:
        values["frame_counter"] = int(values["frame_counter"])
    for key in ("timestamp_us", "temperature_c", "rolling_shutter_time_us", "line_time_us"):
        if key in values:
            values[key] = float(values[key])
    return SensorMetadata(**values)


def _synthetic_calibration_for_raw(raw_data: np.ndarray, cfa_pattern: str):
    array = np.asarray(raw_data)
    if array.ndim == 3 and array.shape[0] <= 8:
        height, width = int(array.shape[1]), int(array.shape[2])
    else:
        height, width = int(array.shape[0]), int(array.shape[1])
    return make_synthetic_raw(width=width, height=height, cfa_pattern=str(cfa_pattern)).calibration
