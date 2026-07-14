"""SID Sony RAW loader for low-light native RAW diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
from PIL import Image

from perception_isp.core.task_types import EvaluationSample
from perception_isp.core.types import CalibrationProfile, RawFrame, SensorMetadata


DEFAULT_SID_SONY_ZIP = "data/raw_datasets/sid/downloads/Sony2025.zip"
DEFAULT_SID_CACHE_DIR = "data/raw_datasets/sid/extracted_samples"

_SID_NAME_RE = re.compile(r"^Sony/(?P<kind>short|long)/(?P<scene>\d{5})_(?P<shot>\d{2})_(?P<exposure>[0-9.]+)s\.ARW$")


@dataclass(frozen=True)
class SIDPair:
    """One SID short/long RAW pair."""

    scene_id: str
    short_member: str
    long_member: str
    short_exposure_s: float
    long_exposure_s: float


def build_sid_sony_pairs(zip_path: str | Path, *, exposure_s: float | None = 0.1) -> tuple[SIDPair, ...]:
    """Index SID Sony archive entries and return short/long pairs."""

    zip_file = Path(zip_path).expanduser()
    if not zip_file.is_file():
        raise FileNotFoundError(f"SID Sony archive not found: {zip_file}")
    with ZipFile(zip_file) as archive:
        names = archive.namelist()
    long_by_scene: dict[str, tuple[str, float]] = {}
    short_by_scene: dict[str, list[tuple[str, float]]] = {}
    for name in names:
        match = _SID_NAME_RE.match(name)
        if not match:
            continue
        scene = str(match.group("scene"))
        exposure = float(match.group("exposure"))
        if match.group("kind") == "long":
            previous = long_by_scene.get(scene)
            if previous is None or exposure > previous[1]:
                long_by_scene[scene] = (name, exposure)
        else:
            short_by_scene.setdefault(scene, []).append((name, exposure))

    pairs: list[SIDPair] = []
    for scene in sorted(set(long_by_scene).intersection(short_by_scene)):
        short_candidates = sorted(short_by_scene[scene], key=lambda item: (abs(item[1] - float(exposure_s or item[1])), item[0]))
        if exposure_s is None:
            short_candidates = sorted(short_by_scene[scene], key=lambda item: (-item[1], item[0]))
        short_member, short_exposure = short_candidates[0]
        long_member, long_exposure = long_by_scene[scene]
        pairs.append(
            SIDPair(
                scene_id=scene,
                short_member=short_member,
                long_member=long_member,
                short_exposure_s=float(short_exposure),
                long_exposure_s=float(long_exposure),
            )
        )
    return tuple(pairs)


def load_sid_sony_scene_edge_samples(
    zip_path: str | Path = DEFAULT_SID_SONY_ZIP,
    *,
    count: int = 1,
    offset: int = 0,
    width: int = 640,
    height: int = 426,
    scene_scale: float = 2.0,
    exposure_s: float | None = 0.1,
    cache_dir: str | Path = DEFAULT_SID_CACHE_DIR,
) -> tuple[EvaluationSample, ...]:
    """Load SID Sony pairs as EvaluationSamples.

    The short-exposure RAW becomes the native sensor input. The paired
    long-exposure RAW is post-processed into a high-information reference RGB
    for scene-edge proxy metrics.
    """

    all_pairs = build_sid_sony_pairs(zip_path, exposure_s=exposure_s)
    start = max(int(offset), 0)
    selected = all_pairs[start : start + max(int(count), 0)]
    if not selected:
        raise ValueError(f"no SID Sony pairs selected from {zip_path}")
    zip_file = Path(zip_path).expanduser()
    cache_root = Path(cache_dir).expanduser()
    samples = [
        _load_pair(
            zip_file=zip_file,
            pair=pair,
            cache_root=cache_root,
            width=width,
            height=height,
            scene_scale=scene_scale,
        )
        for pair in selected
    ]
    return tuple(samples)


def _load_pair(
    *,
    zip_file: Path,
    pair: SIDPair,
    cache_root: Path,
    width: int,
    height: int,
    scene_scale: float,
) -> EvaluationSample:
    short_path = _extract_member(zip_file, pair.short_member, cache_root)
    long_path = _extract_member(zip_file, pair.long_member, cache_root)
    raw_data, metadata, calibration, raw_shape = _read_short_raw(short_path, pair)
    raw_data = _resize_bayer_preserving_pattern(raw_data, width=width, height=height)
    reference_rgb = _read_reference_rgb(
        long_path,
        width=max(int(round(raw_data.shape[1] * max(float(scene_scale), 1.0))), int(raw_data.shape[1])),
        height=max(int(round(raw_data.shape[0] * max(float(scene_scale), 1.0))), int(raw_data.shape[0])),
    )
    raw = RawFrame(
        data=raw_data,
        metadata=metadata,
        calibration=calibration,
        provenance={
            "dataset": "SID",
            "sensor": "Sony",
            "true_sensor_cfa_mosaic": True,
            "pattern_remapped": False,
            "source_cfa_pattern": metadata.cfa_pattern,
            "target_cfa_pattern": metadata.cfa_pattern,
            "short_member": pair.short_member,
            "long_member": pair.long_member,
            "original_raw_shape": raw_shape,
            "resized_raw_shape": tuple(int(v) for v in raw_data.shape),
        },
    )
    return EvaluationSample(
        sample_id=f"sid_sony_{pair.scene_id}_{pair.short_exposure_s:g}s",
        raw=raw,
        ground_truth=(),
        source="sid_sony_raw",
        metadata={
            "dataset": "SID",
            "sensor": "Sony",
            "scene_id": pair.scene_id,
            "short_member": pair.short_member,
            "long_member": pair.long_member,
            "short_exposure_s": pair.short_exposure_s,
            "long_exposure_s": pair.long_exposure_s,
            "scene_width": int(reference_rgb.shape[1]),
            "scene_height": int(reference_rgb.shape[0]),
            "width": int(raw_data.shape[1]),
            "height": int(raw_data.shape[0]),
            "cfa_pattern": metadata.cfa_pattern,
        },
        reference_rgb=reference_rgb,
    )


def _extract_member(zip_file: Path, member: str, cache_root: Path) -> Path:
    target = cache_root / member
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_file) as archive:
        with archive.open(member) as src, target.open("wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
    return target


def _read_short_raw(path: Path, pair: SIDPair) -> tuple[np.ndarray, SensorMetadata, CalibrationProfile, tuple[int, int]]:
    rawpy = _import_rawpy()

    with rawpy.imread(str(path)) as raw:
        visible = np.asarray(raw.raw_image_visible, dtype=np.float64).copy()
        pattern = _rawpy_pattern(raw)
        black = float(_mean_black(raw.black_level_per_channel))
        white = float(raw.white_level)
    metadata = SensorMetadata(
        camera_id="sid_sony",
        sensor_id="sony_a7s2_sid",
        module_serial=pair.scene_id,
        calibration_id="sid_sony_rawpy",
        cfa_pattern=pattern,
        exposure_times_us=(pair.short_exposure_s * 1_000_000.0,),
        analog_gains=(1.0,),
        digital_gains=(1.0,),
    )
    calibration = CalibrationProfile(
        cfa_pattern=pattern,
        black_level=black,
        white_level=white,
        shot_noise_coeff=0.0065,
        read_noise_var=0.00025,
        dark_current_coeff=0.000002,
    )
    return visible, metadata, calibration, tuple(int(v) for v in visible.shape[:2])


def _read_reference_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    rawpy = _import_rawpy()

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
    rgb = np.asarray(rgb16, dtype=np.float64) / 65535.0
    rgb = _robust_rgb_stretch(rgb)
    target_h = _even_at_least_two(min(int(height), int(rgb.shape[0])))
    target_w = _even_at_least_two(min(int(width), int(rgb.shape[1])))
    return _resize_rgb(rgb, target_h, target_w)


def _resize_bayer_preserving_pattern(raw: np.ndarray, *, width: int, height: int) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    values = values[: values.shape[0] - values.shape[0] % 2, : values.shape[1] - values.shape[1] % 2]
    target_h = _even_at_least_two(min(int(height), int(values.shape[0])))
    target_w = _even_at_least_two(min(int(width), int(values.shape[1])))
    if values.shape == (target_h, target_w):
        return values.copy()
    plane_h = target_h // 2
    plane_w = target_w // 2
    out = np.zeros((target_h, target_w), dtype=np.float64)
    out[0::2, 0::2] = _resize_gray(values[0::2, 0::2], plane_h, plane_w)
    out[0::2, 1::2] = _resize_gray(values[0::2, 1::2], plane_h, plane_w)
    out[1::2, 0::2] = _resize_gray(values[1::2, 0::2], plane_h, plane_w)
    out[1::2, 1::2] = _resize_gray(values[1::2, 1::2], plane_h, plane_w)
    return out


def _resize_gray(values: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32))
    return np.asarray(image.resize((int(width), int(height)), Image.Resampling.BOX), dtype=np.float64)


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype("uint8"))
    resized = image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64) / 255.0


def _rawpy_pattern(raw: Any) -> str:
    desc = raw.color_desc.decode("ascii", errors="ignore") if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
    chars = []
    for row in np.asarray(raw.raw_pattern)[:2, :2]:
        for index in row:
            idx = int(index)
            chars.append(desc[idx] if 0 <= idx < len(desc) else "G")
    return "".join("G" if value == "g" else value.upper() for value in chars)


def _import_rawpy() -> Any:
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError("rawpy is required for SID Sony RAW loading; install perception-isp[raw] or run pip install rawpy") from exc
    return rawpy


def _mean_black(values: Iterable[Any]) -> float:
    arr = np.asarray(tuple(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else 0.0


def _robust_rgb_stretch(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    low = float(np.percentile(values, 0.1))
    high = float(np.percentile(values, 99.7))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.clip(values, 0.0, 1.0)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _even_at_least_two(value: int) -> int:
    result = max(int(value), 2)
    return result if result % 2 == 0 else result - 1
