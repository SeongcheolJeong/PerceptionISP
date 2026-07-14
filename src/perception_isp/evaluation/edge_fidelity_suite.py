"""Synthetic object-edge fidelity validation for PerceptionISP."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from perception_isp.core.pipeline import PerceptionISPPipeline
from perception_isp.core.types import CalibrationProfile, PerceptionISPConfig, RawFrame, SensorMetadata, json_ready


EDGE_FIDELITY_SUMMARY = "edge_fidelity_suite_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a PerceptionISP object-edge fidelity report.")
    parser.add_argument("--sensor-width", type=int, default=160)
    parser.add_argument("--sensor-height", type=int, default=96)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--cfa", action="append", default=None)
    parser.add_argument("--psf-sigma", action="append", type=float, default=None, help="Lens PSF sigma in sensor pixels. Repeatable.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output-dir", default="reports/perception_edge_fidelity_suite_synthetic")
    args = parser.parse_args(argv)

    summary = build_edge_fidelity_suite(
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
        oversample=int(args.oversample),
        cfa_patterns=tuple(args.cfa or ("RGGB", "GRBG", "BGGR", "GBRG", "RCCB", "RGBIR", "MONO")),
        psf_sigmas=tuple(float(value) for value in (args.psf_sigma or (0.0, 0.8, 1.6))),
        seed=int(args.seed),
    )
    html_path = write_edge_fidelity_suite(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / EDGE_FIDELITY_SUMMARY),
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


def build_edge_fidelity_suite(
    *,
    sensor_width: int = 160,
    sensor_height: int = 96,
    oversample: int = 6,
    cfa_patterns: Sequence[str] = ("RGGB", "GRBG", "BGGR", "GBRG", "RCCB", "RGBIR", "MONO"),
    psf_sigmas: Sequence[float] = (0.0, 0.8, 1.6),
    seed: int = 41,
    config: PerceptionISPConfig | None = None,
) -> Dict[str, Any]:
    width = max(int(sensor_width), 24)
    height = max(int(sensor_height), 16)
    scale = max(int(oversample), 2)
    cfas = tuple(dict.fromkeys(str(value).upper().replace("-", "") for value in cfa_patterns))
    sigmas = tuple(float(value) for value in psf_sigmas)
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    scene, labels = _make_object_edge_scene(height, width, scale)
    object_edge = _downsample_bool(_label_boundary(labels), height, width)

    cases = []
    for sigma in sigmas:
        optical_scene = _gaussian_blur(scene, sigma=float(sigma) * float(scale))
        sensor_rgb = _sample_scene(optical_scene, height, width)
        sensor_edge_strength_abs = _edge_strength_abs(_luma(sensor_rgb))
        sensor_edge_strength = _normalize(sensor_edge_strength_abs)
        sensor_edge = _edge_mask(sensor_edge_strength)
        for cfa in cfas:
            cases.append(
                _run_case(
                    cfa_pattern=cfa,
                    psf_sigma=float(sigma),
                    sensor_rgb=sensor_rgb,
                    sensor_edge=sensor_edge,
                    sensor_edge_strength=sensor_edge_strength,
                    sensor_edge_strength_abs=sensor_edge_strength_abs,
                    object_edge=object_edge,
                    seed=seed,
                    pipeline=pipeline,
                )
            )
    checks = _checks(cases, sigmas=sigmas)
    return {
        "sensor_width": width,
        "sensor_height": height,
        "oversample": scale,
        "scene_width": width * scale,
        "scene_height": height * scale,
        "cfa_patterns": cfas,
        "psf_sigmas": sigmas,
        "seed": int(seed),
        "cases": cases,
        "checks": checks,
        "psf_visibility": _psf_visibility(cases),
        "rankings": _rankings(cases),
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "interpretation": (
            "This suite compares object-boundary and sensor-edge oracles against HumanISP RGB, "
            "PerceptionISP RGB, and PerceptionISP aux edge maps across CFA patterns and LensPSF blur."
        ),
        "claim_boundary": (
            "The metrics are front-end edge-fidelity diagnostics. They do not prove downstream detector "
            "accuracy, but they show where CFA choice and optical blur change edge evidence available to perception."
        ),
    }


def write_edge_fidelity_suite(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _materialize_assets(summary, destination)
    (destination / EDGE_FIDELITY_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _run_case(
    *,
    cfa_pattern: str,
    psf_sigma: float,
    sensor_rgb: np.ndarray,
    sensor_edge: np.ndarray,
    sensor_edge_strength: np.ndarray,
    sensor_edge_strength_abs: np.ndarray,
    object_edge: np.ndarray,
    seed: int,
    pipeline: PerceptionISPPipeline,
) -> Dict[str, Any]:
    raw = _raw_from_sensor_rgb(
        sensor_rgb,
        cfa_pattern=cfa_pattern,
        psf_sigma=psf_sigma,
        seed=seed,
        provenance={"edge_fidelity_psf_sigma": float(psf_sigma), "edge_fidelity_cfa_pattern": str(cfa_pattern)},
    )
    result = pipeline.run(raw)
    human_rgb = np.asarray(result.human_rgb if result.human_rgb is not None else result.vision_rgb, dtype=np.float64)
    vision_rgb = np.asarray(result.vision_rgb, dtype=np.float64)
    maps = result.maps
    human_edge_strength = _edge_strength(_luma(human_rgb))
    vision_edge_strength = _edge_strength(_luma(vision_rgb))
    aux_edge_strength = _normalize(np.asarray(maps["edge_strength"], dtype=np.float64))
    human_edge = _edge_mask(human_edge_strength)
    vision_edge = _edge_mask(vision_edge_strength)
    aux_edge = _edge_mask(aux_edge_strength)
    edge_confidence = np.asarray(maps["edge_confidence"], dtype=np.float64)
    demosaic_confidence = np.asarray(maps["demosaic_confidence"], dtype=np.float64)
    metrics = {
        "sensor_edge_strength_p95": float(np.percentile(sensor_edge_strength_abs, 95.0)),
        "object_edge_fraction": float(np.mean(object_edge)),
        "sensor_edge_fraction": float(np.mean(sensor_edge)),
        "human_object_edge_f1": _f1_metrics(human_edge, object_edge)["f1"],
        "perception_object_edge_f1": _f1_metrics(vision_edge, object_edge)["f1"],
        "aux_object_edge_f1": _f1_metrics(aux_edge, object_edge)["f1"],
        "human_sensor_edge_f1": _f1_metrics(human_edge, sensor_edge)["f1"],
        "perception_sensor_edge_f1": _f1_metrics(vision_edge, sensor_edge)["f1"],
        "aux_sensor_edge_f1": _f1_metrics(aux_edge, sensor_edge)["f1"],
        "edge_confidence_on_object_edge": _masked_mean(edge_confidence, object_edge),
        "edge_confidence_off_object_edge": _masked_mean(edge_confidence, np.logical_not(_dilate(object_edge, radius=2))),
        "demosaic_confidence_on_object_edge": _masked_mean(demosaic_confidence, object_edge),
        "finite_outputs": bool(
            np.isfinite(human_rgb).all()
            and np.isfinite(vision_rgb).all()
            and all(np.isfinite(value).all() for value in maps.values())
        ),
    }
    metrics["edge_confidence_separation"] = float(metrics["edge_confidence_on_object_edge"] - metrics["edge_confidence_off_object_edge"])
    for prefix, pred, ref in (
        ("human_object", human_edge, object_edge),
        ("perception_object", vision_edge, object_edge),
        ("aux_object", aux_edge, object_edge),
        ("human_sensor", human_edge, sensor_edge),
        ("perception_sensor", vision_edge, sensor_edge),
        ("aux_sensor", aux_edge, sensor_edge),
    ):
        values = _f1_metrics(pred, ref)
        metrics[f"{prefix}_precision"] = values["precision"]
        metrics[f"{prefix}_recall"] = values["recall"]
    return {
        "id": f"psf_{psf_sigma:.2f}_{cfa_pattern}",
        "psf_sigma": float(psf_sigma),
        "cfa_pattern": str(cfa_pattern),
        "metrics": metrics,
        "health": dict(result.health),
        "_assets_source": {
            "sensor_rgb": sensor_rgb,
            "object_edge": object_edge.astype(np.float64),
            "sensor_edge": sensor_edge.astype(np.float64),
            "human_rgb": human_rgb,
            "vision_rgb": vision_rgb,
            "human_edge": human_edge_strength,
            "vision_edge": vision_edge_strength,
            "aux_edge_strength": aux_edge_strength,
            "edge_confidence": edge_confidence,
        },
    }


def _make_object_edge_scene(sensor_height: int, sensor_width: int, oversample: int) -> Tuple[np.ndarray, np.ndarray]:
    height = sensor_height * oversample
    width = sensor_width * oversample
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    image = np.zeros((height, width, 3), dtype=np.float64)
    labels = np.zeros((height, width), dtype=np.int32)
    image[:] = np.array([0.12, 0.16, 0.18])
    road = yy > 0.48
    image[road] = np.array([0.18, 0.18, 0.17])
    sky = yy <= 0.48
    image[sky] = np.array([0.18, 0.28, 0.38])
    lane_left = np.abs(xx - (0.36 + 0.14 * (yy - 0.5))) < 0.006
    lane_right = np.abs(xx - (0.64 - 0.14 * (yy - 0.5))) < 0.006
    image[(lane_left | lane_right) & road] = np.array([0.85, 0.82, 0.70])

    vehicle = (xx > 0.34) & (xx < 0.67) & (yy > 0.58) & (yy < 0.82)
    windshield = (xx > 0.42) & (xx < 0.58) & (yy > 0.61) & (yy < 0.69)
    image[vehicle] = np.array([0.04, 0.07, 0.10])
    image[windshield] = np.array([0.16, 0.28, 0.34])
    labels[vehicle] = 1

    person_body = (xx > 0.18) & (xx < 0.23) & (yy > 0.56) & (yy < 0.83)
    person_head = ((xx - 0.205) ** 2 + (yy - 0.525) ** 2) < 0.020**2
    image[person_body | person_head] = np.array([0.82, 0.18, 0.10])
    labels[person_body | person_head] = 2

    sign = (np.abs(xx - 0.78) + np.abs(yy - 0.45)) < 0.070
    image[sign] = np.array([0.95, 0.86, 0.04])
    labels[sign] = 3

    pole = (xx > 0.82) & (xx < 0.835) & (yy > 0.36) & (yy < 0.83)
    lamp = ((xx - 0.828) ** 2 + (yy - 0.315) ** 2) < 0.018**2
    image[pole] = np.array([0.08, 0.08, 0.07])
    image[lamp] = np.array([1.2, 0.03, 0.02])
    labels[pole | lamp] = 4

    return np.clip(image, 0.0, 1.2), labels


def _raw_from_sensor_rgb(
    rgb: np.ndarray,
    *,
    cfa_pattern: str,
    psf_sigma: float = 0.0,
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
        signal = np.clip(signal + rng.normal(0.0, 0.001 + 0.0002 * index, size=signal.shape), 0.0, 1.0)
        planes.append(np.round(signal * (4095.0 - 64.0) + 64.0).astype(np.float64))
    metadata = SensorMetadata(
        camera_id="edge_fidelity_synthetic",
        sensor_id="object_edge_sensor",
        module_serial="SIM_EDGE_FIDELITY",
        calibration_id="edge_fidelity_calibration_v1",
        isp_profile_id="perception_isp_reference_v1",
        exposure_times_us=tuple(8000.0 * float(value) for value in exposures),
        analog_gains=tuple(1.0 for _ in exposures),
        digital_gains=tuple(1.0 for _ in exposures),
        hdr_mode="multi_exposure",
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
        psf_sigma_map=np.full((height, width), max(float(psf_sigma), 0.0), dtype=np.float64),
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


def _checks(cases: Sequence[Mapping[str, Any]], *, sigmas: Sequence[float]) -> list[Dict[str, Any]]:
    finite = all(bool(case.get("metrics", {}).get("finite_outputs")) for case in cases if isinstance(case.get("metrics"), Mapping))
    valid_rows = []
    for case in cases:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), Mapping) else {}
        valid_rows.append(
            all(
                0.0 <= float(metrics.get(name, 0.0)) <= 1.0
                for name in (
                    "human_object_edge_f1",
                    "perception_object_edge_f1",
                    "aux_object_edge_f1",
                    "human_sensor_edge_f1",
                    "perception_sensor_edge_f1",
                    "aux_sensor_edge_f1",
                    "edge_confidence_on_object_edge",
                    "demosaic_confidence_on_object_edge",
                )
            )
        )
    object_edges = [float(case.get("metrics", {}).get("object_edge_fraction", 0.0)) for case in cases if isinstance(case.get("metrics"), Mapping)]
    sensor_edges = [float(case.get("metrics", {}).get("sensor_edge_fraction", 0.0)) for case in cases if isinstance(case.get("metrics"), Mapping)]
    psf_response = _psf_response(cases, sigmas=sigmas)
    return [
        {
            "id": "finite_edge_fidelity_outputs",
            "description": "All HumanISP, PerceptionISP, and aux edge outputs are finite.",
            "status": "pass" if finite else "fail",
            "criteria": [{"metric": "finite_outputs", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "object_and_sensor_edge_oracles_present",
            "description": "Object-boundary and sensor-edge oracle masks are non-empty.",
            "status": "pass" if min(object_edges or [0.0]) > 0.005 and min(sensor_edges or [0.0]) > 0.005 else "fail",
            "criteria": [
                {"metric": "min_object_edge_fraction", "value": min(object_edges or [0.0]), "threshold": 0.005, "pass": min(object_edges or [0.0]) > 0.005},
                {"metric": "min_sensor_edge_fraction", "value": min(sensor_edges or [0.0]), "threshold": 0.005, "pass": min(sensor_edges or [0.0]) > 0.005},
            ],
        },
        {
            "id": "edge_fidelity_metrics_bounded",
            "description": "Fidelity and confidence metrics are bounded in [0, 1].",
            "status": "pass" if all(valid_rows) else "fail",
            "criteria": [{"metric": "bounded_case_count", "value": sum(1 for value in valid_rows if value), "threshold": len(valid_rows), "pass": all(valid_rows)}],
        },
        {
            "id": "lens_psf_reduces_sensor_edge_contrast",
            "description": "Increasing LensPSF blur should lower the sensor-edge gradient oracle.",
            "status": "pass" if bool(psf_response.get("pass")) else "fail",
            "criteria": [
                {
                    "metric": "sensor_edge_strength_p95_ratio",
                    "baseline": psf_response.get("baseline"),
                    "target": psf_response.get("target"),
                    "delta": psf_response.get("delta"),
                    "value": psf_response.get("ratio"),
                    "threshold": 0.85,
                    "pass": bool(psf_response.get("pass")),
                }
            ],
        },
    ]


def _psf_response(cases: Sequence[Mapping[str, Any]], *, sigmas: Sequence[float]) -> Dict[str, Any]:
    if not sigmas:
        return {"pass": False, "baseline": 0.0, "target": 0.0, "delta": 0.0}
    low = min(float(value) for value in sigmas)
    high = max(float(value) for value in sigmas)
    by_sigma: Dict[float, list[float]] = {low: [], high: []}
    for case in cases:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), Mapping) else {}
        sigma = float(case.get("psf_sigma", 0.0))
        if sigma in by_sigma:
            by_sigma[sigma].append(float(metrics.get("sensor_edge_strength_p95", 0.0)))
    baseline = float(np.mean(by_sigma[low])) if by_sigma[low] else 0.0
    target = float(np.mean(by_sigma[high])) if by_sigma[high] else 0.0
    delta = target - baseline
    ratio = target / max(baseline, 1.0e-12)
    return {"pass": ratio <= 0.85, "baseline": baseline, "target": target, "delta": delta, "ratio": ratio}


def _psf_visibility(cases: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    by_sigma: Dict[float, list[Mapping[str, Any]]] = {}
    for case in cases:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), Mapping) else {}
        by_sigma.setdefault(float(case.get("psf_sigma", 0.0)), []).append(metrics)

    rows: list[Dict[str, Any]] = []
    baseline: float | None = None
    previous: float | None = None
    for sigma, metrics_rows in sorted(by_sigma.items()):
        sensor_edge_p95 = _mean_metric(metrics_rows, "sensor_edge_strength_p95")
        if baseline is None:
            baseline = sensor_edge_p95
        ratio = sensor_edge_p95 / max(float(baseline), 1.0e-12)
        delta_previous = None if previous is None else sensor_edge_p95 - previous
        rows.append(
            {
                "psf_sigma": float(sigma),
                "sensor_edge_strength_p95_mean": float(sensor_edge_p95),
                "sensor_edge_strength_p95_ratio_vs_nominal": float(ratio),
                "sensor_edge_strength_p95_delta_vs_previous": None if delta_previous is None else float(delta_previous),
                "mean_human_object_edge_f1": _mean_metric(metrics_rows, "human_object_edge_f1"),
                "mean_perception_object_edge_f1": _mean_metric(metrics_rows, "perception_object_edge_f1"),
                "mean_aux_object_edge_f1": _mean_metric(metrics_rows, "aux_object_edge_f1"),
                "visibility_status": _psf_visibility_status(float(sigma), float(ratio)),
            }
        )
        previous = sensor_edge_p95
    return rows


def _mean_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = [float(row.get(metric, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _psf_visibility_status(sigma: float, ratio_vs_nominal: float) -> str:
    if sigma <= 0.0:
        return "nominal"
    if ratio_vs_nominal >= 0.95:
        return "weak_or_subpixel"
    if ratio_vs_nominal <= 0.85:
        return "visible"
    return "moderate"


def _rankings(cases: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    by_sigma: Dict[float, list[Mapping[str, Any]]] = {}
    for case in cases:
        by_sigma.setdefault(float(case.get("psf_sigma", 0.0)), []).append(case)
    rankings = []
    for sigma, rows in sorted(by_sigma.items()):
        ranked = sorted(
            (
                {
                    "cfa_pattern": str(row.get("cfa_pattern", "")),
                    "aux_object_edge_f1": float(row.get("metrics", {}).get("aux_object_edge_f1", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                    "perception_object_edge_f1": float(row.get("metrics", {}).get("perception_object_edge_f1", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                    "edge_confidence_separation": float(row.get("metrics", {}).get("edge_confidence_separation", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                }
                for row in rows
            ),
            key=lambda item: (item["aux_object_edge_f1"], item["perception_object_edge_f1"], item["edge_confidence_separation"]),
            reverse=True,
        )
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
        rankings.append({"psf_sigma": sigma, "ranked_cfas": ranked})
    return rankings


def _sample_scene(scene: np.ndarray, sensor_height: int, sensor_width: int) -> np.ndarray:
    source_h, source_w = scene.shape[:2]
    factor_y = max(source_h // sensor_height, 1)
    factor_x = max(source_w // sensor_width, 1)
    cropped = scene[: sensor_height * factor_y, : sensor_width * factor_x]
    return np.asarray(cropped.reshape(sensor_height, factor_y, sensor_width, factor_x, 3).mean(axis=(1, 3)), dtype=np.float64)


def _downsample_bool(mask: np.ndarray, sensor_height: int, sensor_width: int) -> np.ndarray:
    source_h, source_w = mask.shape[:2]
    factor_y = max(source_h // sensor_height, 1)
    factor_x = max(source_w // sensor_width, 1)
    cropped = np.asarray(mask[: sensor_height * factor_y, : sensor_width * factor_x], dtype=bool)
    return np.any(cropped.reshape(sensor_height, factor_y, sensor_width, factor_x), axis=(1, 3))


def _label_boundary(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int32)
    boundary = np.zeros_like(values, dtype=bool)
    boundary[:-1, :] |= values[:-1, :] != values[1:, :]
    boundary[1:, :] |= values[1:, :] != values[:-1, :]
    boundary[:, :-1] |= values[:, :-1] != values[:, 1:]
    boundary[:, 1:] |= values[:, 1:] != values[:, :-1]
    return boundary & (values > 0)


def _edge_strength(values: np.ndarray) -> np.ndarray:
    return _normalize(_edge_strength_abs(values))


def _edge_strength_abs(values: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(values, dtype=np.float64))
    return np.sqrt(gx * gx + gy * gy)


def _edge_mask(strength: np.ndarray, percentile: float = 88.0) -> np.ndarray:
    values = np.asarray(strength, dtype=np.float64)
    threshold = max(float(np.percentile(values, float(percentile))), float(np.max(values)) * 0.10, 1.0e-6)
    return values >= threshold


def _f1_metrics(predicted: np.ndarray, reference: np.ndarray, *, tolerance: int = 1) -> Dict[str, float]:
    pred = np.asarray(predicted, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if not bool(np.any(pred)) and not bool(np.any(ref)):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not bool(np.any(pred)):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not bool(np.any(ref)):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ref_dilated = _dilate(ref, radius=tolerance)
    pred_dilated = _dilate(pred, radius=tolerance)
    precision = float(np.sum(pred & ref_dilated) / max(float(np.sum(pred)), 1.0))
    recall = float(np.sum(ref & pred_dilated) / max(float(np.sum(ref)), 1.0))
    f1 = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


def _dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return values.copy()
    padded = np.pad(values, int(radius), mode="constant", constant_values=False)
    out = np.zeros_like(values, dtype=bool)
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            out |= padded[int(radius) + dy : int(radius) + dy + values.shape[0], int(radius) + dx : int(radius) + dx + values.shape[1]]
    return out


def _gaussian_blur(image: np.ndarray, *, sigma: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if sigma <= 0.0:
        return arr.copy()
    radius = max(int(np.ceil(float(sigma) * 3.0)), 1)
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coords / max(float(sigma), 1.0e-6)) ** 2)
    kernel /= np.sum(kernel)
    blurred = _convolve_axis(arr, kernel, axis=0)
    return np.clip(_convolve_axis(blurred, kernel, axis=1), 0.0, 1.2)


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def _luma(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.zeros_like(arr, dtype=np.float64)
    low = float(np.nanmin(arr[finite]))
    high = float(np.nanmax(arr[finite]))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    keep = np.asarray(mask, dtype=bool)
    if not bool(np.any(keep)):
        return 0.0
    return float(np.mean(arr[keep]))


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
    psf_visibility_rows = "".join(_psf_visibility_row(row) for row in summary.get("psf_visibility", ()))
    ranking_rows = "".join(_ranking_row(row) for row in summary.get("rankings", ()))
    status = str(summary.get("status", ""))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Object Edge Fidelity</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    img {{ width: 72px; height: auto; margin-right: 4px; border: 1px solid #d8ded7; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Object Edge Fidelity</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</code>. Scene: {int(summary.get('scene_width', 0))} x {int(summary.get('scene_height', 0))}; sensor: {int(summary.get('sensor_width', 0))} x {int(summary.get('sensor_height', 0))}; CFA: <code>{html_lib.escape(', '.join(str(value) for value in summary.get('cfa_patterns', ())))}</code>; LensPSF sigma: <code>{html_lib.escape(', '.join(str(value) for value in summary.get('psf_sigmas', ())))}</code>.</p>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Status</th><th>Check</th><th>Description</th><th>Criteria</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>
  <h2>LensPSF Visibility</h2>
  <div class=\"note\">PSF sigma is expressed in sensor pixels and is applied on the high-resolution scene plane before sensor sampling. If the PSF footprint is far below one sensor pixel, the edge contrast change can be weak or invisible after sampling.</div>
  <table>
    <thead><tr><th>LensPSF Sigma</th><th>Sensor Edge P95 Mean</th><th>Ratio vs Nominal</th><th>Delta vs Previous</th><th>Human Obj F1 Mean</th><th>Perception Obj F1 Mean</th><th>Aux Obj F1 Mean</th><th>Status</th></tr></thead>
    <tbody>{psf_visibility_rows}</tbody>
  </table>
  <h2>CFA Rankings</h2>
  <table>
    <thead><tr><th>LensPSF Sigma</th><th>Ranking</th></tr></thead>
    <tbody>{ranking_rows}</tbody>
  </table>
  <h2>Cases</h2>
  <table>
    <thead><tr><th>Case</th><th>Visuals</th><th>Human Obj F1</th><th>Perception Obj F1</th><th>Aux Obj F1</th><th>Human Sensor F1</th><th>Perception Sensor F1</th><th>Aux Sensor F1</th><th>Edge Conf Sep</th><th>Sensor Edge P95</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{EDGE_FIDELITY_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    criteria = []
    for item in row.get("criteria", ()):
        if not isinstance(item, Mapping):
            continue
        parts = [f"{item.get('metric')}"]
        if item.get("value") is not None:
            parts.append(f"value={_fmt(item.get('value')) if not isinstance(item.get('value'), bool) else item.get('value')}")
        if item.get("delta") is not None:
            parts.append(f"delta={_fmt(item.get('delta'), signed=True)}")
        if item.get("threshold") is not None:
            parts.append(f"threshold={_fmt(item.get('threshold'), signed=isinstance(item.get('threshold'), float) and float(item.get('threshold')) < 0.0)}")
        parts.append(f"pass={bool(item.get('pass'))}")
        criteria.append(" ".join(str(part) for part in parts))
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape('; '.join(criteria))}</td>"
        "</tr>"
    )


def _ranking_row(row: Mapping[str, Any]) -> str:
    ranked = []
    for item in row.get("ranked_cfas", ()):
        if not isinstance(item, Mapping):
            continue
        ranked.append(
            f"{int(item.get('rank', 0))}. {item.get('cfa_pattern')} "
            f"auxObjF1={_fmt(item.get('aux_object_edge_f1'))} "
            f"perObjF1={_fmt(item.get('perception_object_edge_f1'))} "
            f"confSep={_fmt(item.get('edge_confidence_separation'), signed=True)}"
        )
    return f"<tr><td>{_fmt(row.get('psf_sigma'))}</td><td>{html_lib.escape(' | '.join(ranked))}</td></tr>"


def _psf_visibility_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("visibility_status", ""))
    return (
        "<tr>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{_fmt(row.get('sensor_edge_strength_p95_mean'))}</td>"
        f"<td>{_fmt(row.get('sensor_edge_strength_p95_ratio_vs_nominal'))}</td>"
        f"<td>{_fmt(row.get('sensor_edge_strength_p95_delta_vs_previous'), signed=True)}</td>"
        f"<td>{_fmt(row.get('mean_human_object_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('mean_perception_object_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('mean_aux_object_edge_f1'))}</td>"
        f"<td><code>{html_lib.escape(status)}</code></td>"
        "</tr>"
    )


def _case_row(row: Mapping[str, Any], destination: Path) -> str:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    assets = row.get("assets", {}) if isinstance(row.get("assets"), Mapping) else {}
    thumbs = " ".join(
        _asset_img(path, destination)
        for path in (
            assets.get("sensor_rgb"),
            assets.get("object_edge"),
            assets.get("sensor_edge"),
            assets.get("human_rgb"),
            assets.get("vision_rgb"),
            assets.get("aux_edge_strength"),
            assets.get("edge_confidence"),
        )
        if path
    )
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code><br>CFA <code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code>, PSF {_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{thumbs}</td>"
        f"<td>{_fmt(metrics.get('human_object_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('perception_object_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('aux_object_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('human_sensor_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('perception_sensor_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('aux_sensor_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('edge_confidence_separation'), signed=True)}</td>"
        f"<td>{_fmt(metrics.get('sensor_edge_strength_p95'))}</td>"
        "</tr>"
    )


def _asset_img(path: str, destination: Path) -> str:
    relative = os.path.relpath(str(destination / path), start=str(destination))
    return f"<img src=\"{html_lib.escape(relative)}\" alt=\"{html_lib.escape(os.path.basename(path))}\">"


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
