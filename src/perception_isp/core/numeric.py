"""Small numpy-only image helpers used by the reference ISP."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]
EPS = 1.0e-12


def as_float_array(values: object) -> ArrayF:
    return np.asarray(values, dtype=np.float64)


def clip01(values: object) -> ArrayF:
    return np.clip(as_float_array(values), 0.0, 1.0)


def normalize_image(values: object) -> ArrayF:
    array = as_float_array(values)
    if array.size == 0:
        raise ValueError("image array must not be empty")
    finite = np.isfinite(array)
    if not bool(np.any(finite)):
        return np.zeros_like(array, dtype=np.float64)
    low = float(np.nanmin(array[finite]))
    high = float(np.nanmax(array[finite]))
    if high <= low:
        return np.zeros_like(array, dtype=np.float64)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def ensure_exposure_first(raw: object) -> ArrayF:
    array = as_float_array(raw)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError("raw input must be HxW, ExHxW, or HxWxE")
    if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
        return array
    if array.shape[-1] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
        return np.moveaxis(array, -1, 0)
    raise ValueError("cannot infer exposure axis; expected <=4 exposures")


def resize_nearest(values: object, shape: Tuple[int, int]) -> ArrayF:
    array = as_float_array(values)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("resize_nearest expects a 2-D map")
    rows, cols = int(shape[0]), int(shape[1])
    if array.shape == (rows, cols):
        return array.astype(np.float64, copy=True)
    y = np.linspace(0, array.shape[0] - 1, rows)
    x = np.linspace(0, array.shape[1] - 1, cols)
    yi = np.clip(np.round(y).astype(int), 0, array.shape[0] - 1)
    xi = np.clip(np.round(x).astype(int), 0, array.shape[1] - 1)
    return array[np.ix_(yi, xi)].astype(np.float64, copy=True)


def pad_edge_2d(values: object, radius: int) -> ArrayF:
    array = as_float_array(values)
    return np.pad(array, int(radius), mode="edge")


def box_filter(values: object, radius: int = 1) -> ArrayF:
    """Mean filter using an integral image, no scipy dependency."""

    array = as_float_array(values)
    if radius <= 0:
        return array.copy()
    if array.ndim == 3:
        return np.stack([box_filter(array[:, :, c], radius) for c in range(array.shape[2])], axis=2)
    if array.ndim != 2:
        raise ValueError("box_filter expects a 2-D or HxWxC array")
    pad = int(radius)
    padded = np.pad(array, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    size = 2 * pad + 1
    bottom = np.arange(size, size + array.shape[0])
    top = np.arange(0, array.shape[0])
    right = np.arange(size, size + array.shape[1])
    left = np.arange(0, array.shape[1])
    total = (
        integral[np.ix_(bottom, right)]
        - integral[np.ix_(top, right)]
        - integral[np.ix_(bottom, left)]
        + integral[np.ix_(top, left)]
    )
    return total / float(size * size)


def weighted_interpolate(samples: object, mask: object, radius: int = 2) -> ArrayF:
    values = as_float_array(samples)
    weights = as_float_array(mask)
    if values.shape != weights.shape:
        raise ValueError("samples and mask must have the same shape")
    weighted = box_filter(values * weights, radius)
    support = box_filter(weights, radius)
    fallback = float(np.sum(values * weights) / max(float(np.sum(weights)), EPS)) if np.any(weights) else 0.0
    return np.where(support > EPS, weighted / np.maximum(support, EPS), fallback)


def gradient_xy(values: object) -> Tuple[ArrayF, ArrayF]:
    array = as_float_array(values)
    if array.ndim != 2:
        raise ValueError("gradient_xy expects a 2-D array")
    gx = np.empty_like(array, dtype=np.float64)
    gy = np.empty_like(array, dtype=np.float64)
    gx[:, 1:-1] = (array[:, 2:] - array[:, :-2]) * 0.5
    gx[:, 0] = array[:, min(1, array.shape[1] - 1)] - array[:, 0]
    gx[:, -1] = array[:, -1] - array[:, max(array.shape[1] - 2, 0)]
    gy[1:-1, :] = (array[2:, :] - array[:-2, :]) * 0.5
    gy[0, :] = array[min(1, array.shape[0] - 1), :] - array[0, :]
    gy[-1, :] = array[-1, :] - array[max(array.shape[0] - 2, 0), :]
    return gx, gy


def block_reduce_mean(values: object, factor: int) -> ArrayF:
    array = as_float_array(values)
    factor = max(int(factor), 1)
    if factor == 1:
        return array.copy()
    if array.ndim == 2:
        rows = array.shape[0] // factor
        cols = array.shape[1] // factor
        if rows <= 0 or cols <= 0:
            return array.copy()
        cropped = array[: rows * factor, : cols * factor]
        return cropped.reshape(rows, factor, cols, factor).mean(axis=(1, 3))
    if array.ndim == 3:
        return np.stack([block_reduce_mean(array[:, :, c], factor) for c in range(array.shape[2])], axis=2)
    raise ValueError("block_reduce_mean expects a 2-D or HxWxC array")


def apply_matrix_rgb(image: object, matrix: object) -> ArrayF:
    rgb = as_float_array(image)
    mat = as_float_array(matrix).reshape(3, 3)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("apply_matrix_rgb expects HxWx3 input")
    return np.tensordot(rgb[:, :, :3], mat.T, axes=([-1], [0]))


def safe_log_ratio(numerator: object, denominator: object) -> ArrayF:
    return np.log(np.maximum(as_float_array(numerator), EPS) / np.maximum(as_float_array(denominator), EPS))


def percentile(values: object, q: float) -> float:
    array = as_float_array(values).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, float(q)))


def gamma_encode(linear: object, gamma: float = 2.2) -> ArrayF:
    return np.power(np.clip(as_float_array(linear), 0.0, 1.0), 1.0 / max(float(gamma), EPS))


def log_tonemap(linear: object, strength: float = 8.0) -> ArrayF:
    values = np.maximum(as_float_array(linear), 0.0)
    return np.log1p(float(strength) * values) / np.log1p(float(strength))


def edge_aware_denoise(image: object, edge_confidence: object, strength: float) -> ArrayF:
    values = as_float_array(image)
    smooth = box_filter(values, radius=1)
    edge = np.clip(as_float_array(edge_confidence), 0.0, 1.0)
    if values.ndim == 3:
        edge = edge[:, :, None]
    blend = np.clip(float(strength), 0.0, 1.0) * (1.0 - edge)
    return values * (1.0 - blend) + smooth * blend


def row_timestamp_map(
    rows: int,
    timestamp_us: float,
    line_time_us: float,
    readout_direction: str = "top_to_bottom",
) -> ArrayF:
    offsets = np.arange(int(rows), dtype=np.float64) * float(line_time_us)
    direction = str(readout_direction).lower().replace("-", "_")
    if direction == "bottom_to_top":
        offsets = offsets[::-1]
    elif direction != "top_to_bottom":
        raise ValueError(f"unsupported readout direction: {readout_direction!r}")
    return float(timestamp_us) + offsets


def nearest_dewarp(image: object, distortion_coeffs: Sequence[float]) -> ArrayF:
    """Very small radial distortion correction for accurate-path prototyping."""

    values = as_float_array(image)
    rows, cols = values.shape[:2]
    k1, k2, p1, p2, k3 = (list(distortion_coeffs) + [0.0] * 5)[:5]
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, rows), np.linspace(-1.0, 1.0, cols), indexing="ij")
    r2 = xx * xx + yy * yy
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    src_x = xx * radial + 2.0 * p1 * xx * yy + p2 * (r2 + 2.0 * xx * xx)
    src_y = yy * radial + p1 * (r2 + 2.0 * yy * yy) + 2.0 * p2 * xx * yy
    xi = np.clip(np.round((src_x + 1.0) * 0.5 * (cols - 1)).astype(int), 0, cols - 1)
    yi = np.clip(np.round((src_y + 1.0) * 0.5 * (rows - 1)).astype(int), 0, rows - 1)
    if values.ndim == 2:
        return values[yi, xi]
    return values[yi, xi, :]
