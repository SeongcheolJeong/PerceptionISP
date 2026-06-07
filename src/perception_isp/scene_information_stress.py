"""Scene-information stress tests for sensor/ISP information loss."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .pipeline import PerceptionISPPipeline
from .types import CalibrationProfile, PerceptionISPConfig, RawFrame, SensorMetadata, json_ready


SCENE_INFORMATION_SUMMARY = "scene_information_stress_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a scene-information stress report for PerceptionISP.")
    parser.add_argument("--sensor-width", type=int, default=160)
    parser.add_argument("--sensor-height", type=int, default=96)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--cfa", default="RGGB")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-dir", default="reports/perception_scene_information_stress_synthetic")
    args = parser.parse_args(argv)

    summary = build_scene_information_stress(
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
        oversample=int(args.oversample),
        cfa_pattern=str(args.cfa),
        seed=int(args.seed),
    )
    html_path = write_scene_information_stress(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SCENE_INFORMATION_SUMMARY),
                    "status": summary["status"],
                    "case_count": len(summary["cases"]),
                    "check_count": len(summary["checks"]),
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_scene_information_stress(
    *,
    sensor_width: int = 160,
    sensor_height: int = 96,
    oversample: int = 8,
    cfa_pattern: str = "RGGB",
    seed: int = 29,
    config: PerceptionISPConfig | None = None,
) -> Dict[str, Any]:
    width = max(int(sensor_width), 16)
    height = max(int(sensor_height), 16)
    scale = max(int(oversample), 2)
    pattern = str(cfa_pattern).upper().replace("-", "")
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    cases = [
        _run_case("resolved_reference", "Resolved broad edges and objects.", "box", width, height, scale, pattern, seed, pipeline=pipeline),
        _run_case("supersampled_thin_detail", "High-resolution luminance detail finer than the sensor pitch, integrated away by the sensor.", "box", width, height, scale, pattern, seed, pipeline=pipeline),
        _run_case("cfa_chroma_alias", "Same-luma red/green high-frequency chroma pattern sampled by the CFA.", "point", width, height, scale, pattern, seed, pipeline=pipeline),
        _run_case("subpixel_signal", "Sub-pixel traffic-light-like signal with high scene contrast but low sensor fill factor.", "box", width, height, scale, pattern, seed, pipeline=pipeline),
    ]
    by_id = {case["id"]: case for case in cases}
    checks = [
        _case_metric_check(
            "latent_high_frequency_detail_loss",
            "Scene detail above sensor sampling should be visible in the scene oracle but mostly absent after sensor integration.",
            by_id["supersampled_thin_detail"],
            (
                ("scene_luma_gradient_p90", "minimum", 0.20),
                ("sensor_luma_gradient_p90", "maximum", 0.08),
                ("luma_detail_retention_p90", "maximum", 0.20),
            ),
        ),
        _case_metric_check(
            "cfa_chroma_alias_color_confidence_drop",
            "High-frequency chroma content should be visible in the scene oracle while PerceptionISP reports low color confidence after CFA sampling.",
            by_id["cfa_chroma_alias"],
            (
                ("scene_chroma_gradient_p90", "minimum", 0.50),
                ("sensor_chroma_gradient_p90", "maximum", 0.05),
                ("color_confidence_mean", "maximum", 0.10),
            ),
        ),
        _case_metric_check(
            "subpixel_signal_fill_factor_loss",
            "A sub-pixel bright signal should have high scene contrast but much lower sensor contrast after pixel integration.",
            by_id["subpixel_signal"],
            (
                ("scene_signal_contrast", "minimum", 1.00),
                ("sensor_signal_contrast", "maximum", 0.35),
                ("signal_contrast_retention", "maximum", 0.25),
            ),
        ),
    ]
    return {
        "sensor_width": width,
        "sensor_height": height,
        "oversample": scale,
        "scene_width": width * scale,
        "scene_height": height * scale,
        "cfa_pattern": pattern,
        "seed": int(seed),
        "cases": cases,
        "checks": checks,
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "interpretation": (
            "This suite separates high-information scene content from the lower-resolution CFA RAW measurement. "
            "It shows where information is lost before any ISP can recover it and where PerceptionISP confidence maps expose remaining CFA/color uncertainty."
        ),
        "claim_boundary": (
            "The scene oracle proves sensor-front-end information loss in controlled synthetic cases. "
            "It is not detector-performance evidence and it does not prove recovery of information absent from RAW."
        ),
    }


def write_scene_information_stress(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _materialize_assets(summary, destination)
    (destination / SCENE_INFORMATION_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _run_case(
    case_id: str,
    description: str,
    sample_mode: str,
    width: int,
    height: int,
    oversample: int,
    cfa_pattern: str,
    seed: int,
    *,
    pipeline: PerceptionISPPipeline,
) -> Dict[str, Any]:
    scene, roi = _make_high_information_scene(case_id, height, width, oversample)
    sensor_rgb = _sample_scene(scene, height, width, mode=sample_mode)
    raw = _raw_from_sensor_rgb(sensor_rgb, cfa_pattern=cfa_pattern, seed=seed, provenance={"scene_case": case_id, "scene_sample_mode": sample_mode, "scene_oversample": oversample})
    result = pipeline.run(raw)
    metrics = _metrics(scene, sensor_rgb, result, roi)
    metrics["finite_outputs"] = bool(
        np.isfinite(result.vision_rgb).all()
        and (result.human_rgb is None or np.isfinite(result.human_rgb).all())
        and all(np.isfinite(value).all() for value in result.maps.values())
    )
    return {
        "id": str(case_id),
        "description": str(description),
        "sample_mode": str(sample_mode),
        "cfa_pattern": str(result.metadata.get("frame", {}).get("cfa_pattern", cfa_pattern)),
        "metrics": metrics,
        "health": dict(result.health),
        "_assets_source": {
            "scene_preview": _sample_scene(scene, height, width, mode="box"),
            "sensor_rgb": sensor_rgb,
            "human_rgb": np.asarray(result.human_rgb if result.human_rgb is not None else result.vision_rgb, dtype=np.float64),
            "vision_rgb": np.asarray(result.vision_rgb, dtype=np.float64),
            "edge_confidence": np.asarray(result.maps["edge_confidence"], dtype=np.float64),
            "demosaic_confidence": np.asarray(result.maps["demosaic_confidence"], dtype=np.float64),
            "color_confidence": np.asarray(result.maps["color_confidence"], dtype=np.float64),
        },
    }


def _make_high_information_scene(case_id: str, sensor_height: int, sensor_width: int, oversample: int) -> Tuple[np.ndarray, Mapping[str, Any]]:
    height = sensor_height * oversample
    width = sensor_width * oversample
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    image = np.ones((height, width, 3), dtype=np.float64) * 0.16
    roi: Dict[str, Any] = {}
    if case_id == "resolved_reference":
        image[:] = 0.12
        image[(xx > 0.20) & (xx < 0.25)] = np.array([0.85, 0.85, 0.82])
        image[(xx > 0.45) & (xx < 0.58) & (yy > 0.35) & (yy < 0.72)] = np.array([0.05, 0.05, 0.06])
        image[((xx - 0.70) ** 2 + (yy - 0.32) ** 2) < 0.015] = np.array([1.60, 0.08, 0.05])
    elif case_id == "supersampled_thin_detail":
        local_x = np.arange(width) % max(int(oversample), 1)
        stripes = ((local_x // 2) % 2) == 0
        image[:, stripes, :] = 0.92
        image[:, ~stripes, :] = 0.04
    elif case_id == "cfa_chroma_alias":
        local_x = np.arange(width) % max(int(oversample), 1)
        center_phase = ((max(int(oversample), 1) // 2) // 2) % 2
        stripes = ((local_x // 2) % 2) == center_phase
        image[:, stripes, :] = np.array([0.90, 0.04, 0.04])
        image[:, ~stripes, :] = np.array([0.04, 0.90, 0.04])
    elif case_id == "subpixel_signal":
        image[:] = 0.08
        image[(xx > 0.48) & (xx < 0.505) & (yy > 0.25) & (yy < 0.75)] = 0.18
        center_x = 0.50
        center_y = 0.245
        radius = 0.30 / max(float(sensor_height), float(sensor_width))
        signal = ((xx - center_x) ** 2 + (yy - center_y) ** 2) < radius * radius
        image[signal] = np.array([2.0, 0.02, 0.01])
        roi = {"center_x": center_x, "center_y": center_y, "radius": radius, "mask": signal}
    else:
        raise ValueError(f"unsupported scene information case: {case_id}")
    return np.clip(image, 0.0, 2.0), roi


def _sample_scene(scene: np.ndarray, sensor_height: int, sensor_width: int, *, mode: str) -> np.ndarray:
    source_h, source_w = scene.shape[:2]
    if mode == "point":
        ys = np.clip(((np.arange(sensor_height) + 0.5) * source_h / sensor_height).astype(int), 0, source_h - 1)
        xs = np.clip(((np.arange(sensor_width) + 0.5) * source_w / sensor_width).astype(int), 0, source_w - 1)
        return np.asarray(scene[np.ix_(ys, xs)], dtype=np.float64)
    factor_y = max(source_h // sensor_height, 1)
    factor_x = max(source_w // sensor_width, 1)
    cropped = scene[: sensor_height * factor_y, : sensor_width * factor_x]
    return np.asarray(cropped.reshape(sensor_height, factor_y, sensor_width, factor_x, 3).mean(axis=(1, 3)), dtype=np.float64)


def _raw_from_sensor_rgb(
    rgb: np.ndarray,
    *,
    cfa_pattern: str,
    seed: int,
    exposures: Sequence[float] = (1.0, 0.25, 0.0625),
    provenance: Mapping[str, Any] | None = None,
) -> RawFrame:
    height, width = rgb.shape[:2]
    pattern = str(cfa_pattern).upper().replace("-", "")
    rng = np.random.default_rng(int(seed))
    planes = []
    for index, exposure in enumerate(tuple(float(value) for value in exposures)):
        mosaic = _mosaic_rgb(rgb, pattern)
        signal = np.clip(mosaic * exposure, 0.0, 1.0)
        read_noise = rng.normal(0.0, 0.0012 + 0.0002 * index, size=signal.shape)
        signal = np.clip(signal + read_noise, 0.0, 1.0)
        planes.append(np.round(signal * (4095.0 - 64.0) + 64.0).astype(np.float64))
    metadata = SensorMetadata(
        camera_id="scene_information_synthetic",
        sensor_id="supersampled_scene_sensor",
        module_serial="SIM_SCENE_INFO",
        calibration_id="scene_information_calibration_v1",
        isp_profile_id="perception_isp_reference_v1",
        exposure_times_us=tuple(8000.0 * float(value) for value in exposures),
        analog_gains=tuple(1.0 for _ in exposures),
        digital_gains=tuple(1.0 for _ in exposures),
        hdr_mode="multi_exposure" if len(tuple(exposures)) > 1 else "single",
        hdr_ratios=tuple(float(value) for value in exposures),
        cfa_pattern=pattern,
        line_time_us=33333.0 / float(height),
    )
    calibration = CalibrationProfile(
        cfa_pattern=pattern,
        black_level=64.0,
        white_level=4095.0,
        defect_pixels=(),
        lens_shading_gain=np.ones((height, width), dtype=np.float64),
        prnu_gain=np.ones((height, width), dtype=np.float64),
        mtf_confidence_map=np.ones((height, width), dtype=np.float64),
    )
    return RawFrame(data=np.stack(planes, axis=0), metadata=metadata, calibration=calibration, provenance=dict(provenance or {}))


def _mosaic_rgb(rgb: np.ndarray, pattern: str) -> np.ndarray:
    height, width = rgb.shape[:2]
    out = np.zeros((height, width), dtype=np.float64)
    normalized = str(pattern).upper().replace("-", "")
    if normalized == "RGGB":
        tile = ((0, 1), (1, 2))
    elif normalized == "BGGR":
        tile = ((2, 1), (1, 0))
    elif normalized == "GRBG":
        tile = ((1, 0), (2, 1))
    elif normalized == "GBRG":
        tile = ((1, 2), (0, 1))
    elif normalized == "RCCB":
        clear = np.mean(rgb, axis=2) * 1.25
        out[0::2, 0::2] = rgb[0::2, 0::2, 0]
        out[0::2, 1::2] = clear[0::2, 1::2]
        out[1::2, 0::2] = clear[1::2, 0::2]
        out[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return out
    elif normalized in {"RGBIR", "RGBIR2X2"}:
        ir = np.mean(rgb, axis=2) * 0.45 + rgb[:, :, 0] * 0.08
        out[0::2, 0::2] = rgb[0::2, 0::2, 0]
        out[0::2, 1::2] = rgb[0::2, 1::2, 1]
        out[1::2, 0::2] = rgb[1::2, 0::2, 2]
        out[1::2, 1::2] = ir[1::2, 1::2]
        return out
    elif normalized in {"MONO", "MONOCHROME", "THERMAL"}:
        return np.mean(rgb, axis=2)
    else:
        tile = ((0, 1), (1, 2))
    for row in range(2):
        for col in range(2):
            out[row::2, col::2] = rgb[row::2, col::2, tile[row][col]]
    return out


def _metrics(scene: np.ndarray, sensor_rgb: np.ndarray, result: Any, roi: Mapping[str, Any]) -> Dict[str, Any]:
    scene_luma = np.mean(scene, axis=2)
    sensor_luma = np.mean(sensor_rgb, axis=2)
    scene_luma_grad = _gradient_stats(scene_luma)
    sensor_luma_grad = _gradient_stats(sensor_luma)
    scene_chroma_grad = _chroma_gradient_stats(scene)
    sensor_chroma_grad = _chroma_gradient_stats(sensor_rgb)
    maps = result.maps
    metrics = {
        "scene_luma_gradient_mean": scene_luma_grad["mean"],
        "scene_luma_gradient_p90": scene_luma_grad["p90"],
        "scene_luma_gradient_p99": scene_luma_grad["p99"],
        "sensor_luma_gradient_mean": sensor_luma_grad["mean"],
        "sensor_luma_gradient_p90": sensor_luma_grad["p90"],
        "sensor_luma_gradient_p99": sensor_luma_grad["p99"],
        "luma_detail_retention_p90": sensor_luma_grad["p90"] / max(scene_luma_grad["p90"], 1.0e-12),
        "scene_chroma_gradient_p90": scene_chroma_grad["p90"],
        "sensor_chroma_gradient_p90": sensor_chroma_grad["p90"],
        "edge_strength_mean": _mean(maps["edge_strength"]),
        "edge_confidence_mean": _mean(maps["edge_confidence"]),
        "demosaic_confidence_mean": _mean(maps["demosaic_confidence"]),
        "color_confidence_mean": _mean(maps["color_confidence"]),
        "focus_confidence": float(result.health.get("focus_confidence", 0.0)),
        "visibility_confidence": float(result.health.get("visibility_confidence", 0.0)),
        "over_exposure_fraction": float(result.health.get("over_exposure_fraction", 0.0)),
    }
    metrics.update(_signal_metrics(scene, sensor_rgb, roi))
    return metrics


def _gradient_stats(values: np.ndarray) -> Dict[str, float]:
    gy, gx = np.gradient(np.asarray(values, dtype=np.float64))
    magnitude = np.sqrt(gx * gx + gy * gy)
    return {"mean": _mean(magnitude), "p90": float(np.percentile(magnitude, 90.0)), "p99": float(np.percentile(magnitude, 99.0))}


def _chroma_gradient_stats(rgb: np.ndarray) -> Dict[str, float]:
    rg = np.asarray(rgb[:, :, 0] - rgb[:, :, 1], dtype=np.float64)
    bg = np.asarray(rgb[:, :, 2] - rgb[:, :, 1], dtype=np.float64)
    rg_stats = _gradient_stats(rg)
    bg_stats = _gradient_stats(bg)
    return {"mean": 0.5 * (rg_stats["mean"] + bg_stats["mean"]), "p90": max(rg_stats["p90"], bg_stats["p90"]), "p99": max(rg_stats["p99"], bg_stats["p99"])}


def _signal_metrics(scene: np.ndarray, sensor_rgb: np.ndarray, roi: Mapping[str, Any]) -> Dict[str, float]:
    if not roi:
        return {"scene_signal_contrast": 0.0, "sensor_signal_contrast": 0.0, "signal_contrast_retention": 0.0}
    mask = np.asarray(roi.get("mask"), dtype=bool)
    background = np.logical_not(mask)
    scene_signal = float(np.max(scene[:, :, 0][mask])) if bool(np.any(mask)) else 0.0
    scene_background = float(np.median(scene[:, :, 0][background])) if bool(np.any(background)) else 0.0
    center_x = float(roi.get("center_x", 0.5))
    center_y = float(roi.get("center_y", 0.5))
    height, width = sensor_rgb.shape[:2]
    cx = int(round(center_x * float(width)))
    cy = int(round(center_y * float(height)))
    signal_roi = sensor_rgb[max(cy - 2, 0) : min(cy + 3, height), max(cx - 2, 0) : min(cx + 3, width), 0]
    bg_roi = sensor_rgb[max(cy - 10, 0) : max(cy - 5, 1), max(cx - 10, 0) : max(cx - 5, 1), 0]
    sensor_signal = float(np.max(signal_roi)) if signal_roi.size else 0.0
    sensor_background = float(np.median(bg_roi)) if bg_roi.size else float(np.median(sensor_rgb[:, :, 0]))
    scene_contrast = max(scene_signal - scene_background, 0.0)
    sensor_contrast = max(sensor_signal - sensor_background, 0.0)
    return {
        "scene_signal_contrast": scene_contrast,
        "sensor_signal_contrast": sensor_contrast,
        "signal_contrast_retention": sensor_contrast / max(scene_contrast, 1.0e-12),
    }


def _case_metric_check(check_id: str, description: str, case: Mapping[str, Any], criteria: Sequence[Tuple[str, str, float]]) -> Dict[str, Any]:
    metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), Mapping) else {}
    rows = []
    for metric, direction, threshold in criteria:
        value = float(metrics.get(metric, 0.0))
        passed = value >= float(threshold) if direction == "minimum" else value <= float(threshold)
        rows.append({"metric": str(metric), "direction": str(direction), "value": value, "threshold": float(threshold), "pass": bool(passed)})
    return {"id": str(check_id), "description": str(description), "case": str(case.get("id", "")), "status": "pass" if all(row["pass"] for row in rows) else "fail", "criteria": rows}


def _delta_check(
    check_id: str,
    description: str,
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    criteria: Sequence[Tuple[str, str, float]],
) -> Dict[str, Any]:
    base_metrics = baseline.get("metrics", {}) if isinstance(baseline.get("metrics"), Mapping) else {}
    target_metrics = target.get("metrics", {}) if isinstance(target.get("metrics"), Mapping) else {}
    rows = []
    for metric, direction, threshold in criteria:
        base = float(base_metrics.get(metric, 0.0))
        value = float(target_metrics.get(metric, 0.0))
        delta = value - base
        passed = delta >= float(threshold) if direction == "minimum_delta" else delta <= float(threshold)
        rows.append({"metric": str(metric), "direction": str(direction), "baseline": base, "target": value, "delta": delta, "threshold": float(threshold), "pass": bool(passed)})
    return {
        "id": str(check_id),
        "description": str(description),
        "baseline_case": str(baseline.get("id", "")),
        "target_case": str(target.get("id", "")),
        "status": "pass" if all(row["pass"] for row in rows) else "fail",
        "criteria": rows,
    }


def _materialize_assets(summary: Mapping[str, Any], destination: Path) -> None:
    assets_dir = destination / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for case in summary.get("cases", ()):
        if not isinstance(case, dict):
            continue
        source = case.pop("_assets_source", None)
        if not isinstance(source, Mapping):
            continue
        assets: Dict[str, str] = {}
        for name, image in source.items():
            filename = f"assets/{_safe_name(case.get('id', 'case'))}_{_safe_name(name)}.png"
            _save_png(np.asarray(image, dtype=np.float64), destination / filename)
            assets[str(name)] = filename
        case["assets"] = assets


def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        out = _to_uint8_gray(image)
    elif image.ndim == 3:
        out = _to_uint8_rgb(image)
    else:
        raise ValueError("asset image must be 2-D or HxWxC")
    Image.fromarray(out).save(path)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.shape[-1] == 1:
        return np.repeat(_to_uint8_gray(rgb[:, :, 0])[:, :, None], 3, axis=2)
    rgb = rgb[:, :, :3]
    if float(np.nanmax(rgb)) <= 1.5:
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.clip(rgb / max(float(np.nanpercentile(rgb, 99.0)), 1.0e-12), 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(gray)
    if not bool(np.any(finite)):
        return np.zeros(gray.shape, dtype=np.uint8)
    low = float(np.nanmin(gray[finite]))
    high = float(np.nanmax(gray[finite]))
    if high <= low:
        out = np.zeros_like(gray, dtype=np.float64)
    elif low >= 0.0 and high <= 1.0:
        out = np.clip(gray, 0.0, 1.0)
    else:
        out = np.clip((gray - low) / (high - low), 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()))
    case_rows = "".join(_case_row(row, destination) for row in summary.get("cases", ()))
    status = str(summary.get("status", ""))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Scene Information Stress</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    img {{ width: 88px; height: auto; margin-right: 5px; border: 1px solid #d8ded7; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Scene Information Stress</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</code>. Scene: {int(summary.get('scene_width', 0))} x {int(summary.get('scene_height', 0))}; sensor: {int(summary.get('sensor_width', 0))} x {int(summary.get('sensor_height', 0))}; CFA: <code>{html_lib.escape(str(summary.get('cfa_pattern', '')))}</code>.</p>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Status</th><th>Check</th><th>Description</th><th>Criteria</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>
  <h2>Cases</h2>
  <table>
    <thead><tr><th>Case</th><th>Visuals</th><th>Scene Luma P90</th><th>Sensor Luma P90</th><th>Retention</th><th>Edge Conf</th><th>Demosaic Conf</th><th>Color Conf</th><th>Scene Signal</th><th>Sensor Signal</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{SCENE_INFORMATION_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    criteria = []
    for item in row.get("criteria", ()):
        if not isinstance(item, Mapping):
            continue
        if item.get("delta") is not None:
            criteria.append(f"{item.get('metric')} d={_fmt(item.get('delta'), signed=True)} threshold={_fmt(item.get('threshold'), signed=True)}")
        else:
            criteria.append(f"{item.get('metric')} value={_fmt(item.get('value'))} threshold={_fmt(item.get('threshold'))}")
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape('; '.join(criteria))}</td>"
        "</tr>"
    )


def _case_row(row: Mapping[str, Any], destination: Path) -> str:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    assets = row.get("assets", {}) if isinstance(row.get("assets"), Mapping) else {}
    thumbs = " ".join(
        _asset_img(path, destination)
        for path in (
            assets.get("scene_preview"),
            assets.get("sensor_rgb"),
            assets.get("human_rgb"),
            assets.get("vision_rgb"),
            assets.get("edge_confidence"),
            assets.get("demosaic_confidence"),
            assets.get("color_confidence"),
        )
        if path
    )
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code><br>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{thumbs}</td>"
        f"<td>{_fmt(metrics.get('scene_luma_gradient_p90'))}</td>"
        f"<td>{_fmt(metrics.get('sensor_luma_gradient_p90'))}</td>"
        f"<td>{_fmt(metrics.get('luma_detail_retention_p90'))}</td>"
        f"<td>{_fmt(metrics.get('edge_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('demosaic_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('color_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('scene_signal_contrast'))}</td>"
        f"<td>{_fmt(metrics.get('sensor_signal_contrast'))}</td>"
        "</tr>"
    )


def _asset_img(path: str, destination: Path) -> str:
    relative = os.path.relpath(str(destination / path), start=str(destination))
    return f"<img src=\"{html_lib.escape(relative)}\" alt=\"{html_lib.escape(os.path.basename(path))}\">"


def _mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
