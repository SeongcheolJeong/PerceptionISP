"""Synthetic RAW/HDR input generator for fast Perception ISP experiments."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .types import CalibrationProfile, RawFrame, SensorMetadata


def make_synthetic_raw(
    width: int = 320,
    height: int = 180,
    exposures: Sequence[float] = (1.0, 0.25, 0.0625),
    cfa_pattern: str = "RGGB",
    frame_counter: int = 0,
    timestamp_us: float = 0.0,
    seed: int = 7,
) -> RawFrame:
    """Create a small automotive-like HDR RAW frame.

    The scene includes road texture, lane markings, a bright LED-like signal,
    a dark pedestrian-like shape, and a saturated headlight patch.
    """

    width = max(int(width), 16)
    height = max(int(height), 16)
    rng = np.random.default_rng(int(seed) + int(frame_counter))
    rgb = _synthetic_linear_rgb(height, width, rng)
    pattern = str(cfa_pattern).upper().replace("-", "")
    exposure_values = tuple(float(v) for v in exposures)
    planes = []
    for index, exposure_scale in enumerate(exposure_values):
        mosaic = _mosaic(rgb, pattern)
        noisy = np.clip(mosaic * float(exposure_scale), 0.0, 1.0)
        shot = rng.normal(0.0, np.sqrt(np.maximum(noisy, 0.0)) * 0.012, size=noisy.shape)
        read = rng.normal(0.0, 0.003 + index * 0.0005, size=noisy.shape)
        noisy = np.clip(noisy + shot + read, 0.0, 1.0)
        raw = np.round(noisy * (4095.0 - 64.0) + 64.0)
        planes.append(raw.astype(np.float64))
    metadata = SensorMetadata(
        camera_id="synthetic_front",
        sensor_id="synthetic_hdr_sensor",
        module_serial="SIM0001",
        calibration_id="synthetic_calibration_v1",
        isp_profile_id="perception_isp_reference_v1",
        frame_counter=int(frame_counter),
        timestamp_us=float(timestamp_us),
        exposure_times_us=tuple(8000.0 * value for value in exposure_values),
        analog_gains=tuple(1.0 for _ in exposure_values),
        digital_gains=tuple(1.0 for _ in exposure_values),
        hdr_mode="multi_exposure" if len(exposure_values) > 1 else "single",
        hdr_ratios=exposure_values,
        cfa_pattern=pattern,
        rolling_shutter_time_us=33333.0,
        line_time_us=33333.0 / float(height),
    )
    calibration = CalibrationProfile(
        cfa_pattern=pattern,
        black_level=64.0,
        white_level=4095.0,
        defect_pixels=((height // 3, width // 4), (height // 2, width // 2)),
        lens_shading_gain=_lens_gain(height, width),
        prnu_gain=_prnu_gain(height, width),
        mtf_confidence_map=_mtf_map(height, width),
        distortion_coeffs=(-0.08, 0.015, 0.0, 0.0, 0.0),
    )
    return RawFrame(data=np.stack(planes, axis=0), metadata=metadata, calibration=calibration)


def make_synthetic_scene_rgb(
    width: int = 320,
    height: int = 180,
    *,
    frame_counter: int = 0,
    seed: int = 7,
) -> np.ndarray:
    """Return the linear RGB scene used by the synthetic RAW generator."""

    width = max(int(width), 16)
    height = max(int(height), 16)
    rng = np.random.default_rng(int(seed) + int(frame_counter))
    return _synthetic_linear_rgb(height, width, rng)


def _synthetic_linear_rgb(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    sky = np.stack([0.20 + 0.18 * yy, 0.30 + 0.12 * yy, 0.42 + 0.18 * yy], axis=2)
    road = np.stack([0.12 + 0.16 * yy, 0.12 + 0.15 * yy, 0.12 + 0.13 * yy], axis=2)
    image = np.where((yy[:, :, None] < 0.42), sky, road)
    road_mask = yy > 0.42
    center = 0.5
    lane_left = np.abs(xx - (center - 0.18 * (yy - 0.42))) < 0.012
    lane_right = np.abs(xx - (center + 0.18 * (yy - 0.42))) < 0.012
    yellow_lane = lane_left & road_mask
    white_lane = lane_right & road_mask
    image[yellow_lane] = np.array([0.95, 0.78, 0.18])
    image[white_lane] = np.array([0.92, 0.92, 0.86])

    car = (yy > 0.60) & (yy < 0.78) & (xx > 0.58) & (xx < 0.76)
    image[car] = np.array([0.05, 0.06, 0.07])
    image[(yy > 0.63) & (yy < 0.68) & (xx > 0.62) & (xx < 0.72)] = np.array([0.22, 0.28, 0.35])
    headlight = ((xx - 0.61) ** 2 + (yy - 0.75) ** 2 < 0.0012) | ((xx - 0.73) ** 2 + (yy - 0.75) ** 2 < 0.0012)
    image[headlight] = np.array([1.8, 1.7, 1.45])

    pedestrian = (yy > 0.56) & (yy < 0.82) & (xx > 0.36) & (xx < 0.41)
    image[pedestrian] = np.array([0.03, 0.035, 0.04])
    image[((xx - 0.385) ** 2 + (yy - 0.535) ** 2) < 0.002] = np.array([0.05, 0.04, 0.035])

    traffic_light = ((xx - 0.80) ** 2 + (yy - 0.23) ** 2) < 0.001
    image[traffic_light] = np.array([1.6, 0.08, 0.05])
    texture = rng.normal(0.0, 0.018, size=(height, width, 1))
    image = image + texture * (road_mask[:, :, None] * 0.8 + 0.2)
    return np.clip(image, 0.0, 2.0)


def _mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.float64)
    pattern = pattern.upper().replace("-", "")
    if pattern == "RGGB":
        tile = ((0, 1), (1, 2))
    elif pattern == "BGGR":
        tile = ((2, 1), (1, 0))
    elif pattern == "GRBG":
        tile = ((1, 0), (2, 1))
    elif pattern == "GBRG":
        tile = ((1, 2), (0, 1))
    elif pattern == "RCCB":
        clear = np.mean(rgb, axis=2) * 1.25
        out[0::2, 0::2] = rgb[0::2, 0::2, 0]
        out[0::2, 1::2] = clear[0::2, 1::2]
        out[1::2, 0::2] = clear[1::2, 0::2]
        out[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return out
    elif pattern in {"RGBIR", "RGBIR2X2"}:
        ir = np.mean(rgb, axis=2) * 0.45 + rgb[:, :, 0] * 0.08
        out[0::2, 0::2] = rgb[0::2, 0::2, 0]
        out[0::2, 1::2] = rgb[0::2, 1::2, 1]
        out[1::2, 0::2] = rgb[1::2, 0::2, 2]
        out[1::2, 1::2] = ir[1::2, 1::2]
        return out
    elif pattern in {"MONO", "MONOCHROME", "THERMAL"}:
        return np.mean(rgb, axis=2)
    else:
        tile = ((0, 1), (1, 2))
    for r in range(2):
        for c in range(2):
            out[r::2, c::2] = rgb[r::2, c::2, tile[r][c]]
    return out


def _lens_gain(height: int, width: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    return 1.0 + 0.45 * np.clip(radius, 0.0, 1.4) ** 2


def _prnu_gain(height: int, width: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return 1.0 + 0.01 * np.sin(xx / 13.0) + 0.006 * np.cos(yy / 17.0)


def _mtf_map(height: int, width: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    return np.clip(1.0 - 0.35 * radius, 0.45, 1.0)
