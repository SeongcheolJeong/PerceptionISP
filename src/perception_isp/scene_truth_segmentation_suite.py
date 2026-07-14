"""Scene-truth object-mask edge evaluation for HumanISP vs PerceptionISP.

The key difference from RGB-annotated segmentation benchmarks is that object
masks are created before camera simulation. The rendered RGB scene, CFA RAW, ISP
outputs, and aux maps are all evaluated against that pre-sensor object truth.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .camerae2e_bridge import raw_from_camerae2e_rgb, raw_from_rgb_direct
from .comparison import build_pipeline_images
from .object_boundary_edge_suite import _dilate, _edge_mask, _edge_strength, _f1_metrics, _luma, _masked_mean, _normalize
from .types import PerceptionISPConfig, RawFrame, json_ready


SUMMARY_FILENAME = "scene_truth_segmentation_summary.json"
SIGNALS = (
    "human_rgb_edge",
    "perception_rgb_edge",
    "aux_edge_strength",
    "aux_edge_confidence",
    "aux_edge_evidence",
    "aux_strength_gated_confidence",
)


@dataclass(frozen=True)
class SceneTruthObject:
    label: str
    mask_high: np.ndarray
    shape_tag: str


@dataclass(frozen=True)
class SceneTruthCase:
    case_id: str
    rgb_high: np.ndarray
    objects: Tuple[SceneTruthObject, ...]
    cfa: str
    psf_sigma: float
    raw: RawFrame
    metadata: Mapping[str, Any]


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ISP edge evidence against pre-sensor scene object masks.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--scene-scale", type=float, default=3.0)
    parser.add_argument("--cfas", default="RGGB,GRBG")
    parser.add_argument("--psf-sigmas", default="0.0,1.2")
    parser.add_argument("--use-camerae2e", action="store_true")
    parser.add_argument("--scene-luminance", type=float, default=100.0)
    parser.add_argument("--boundary-thickness", type=int, default=1)
    parser.add_argument("--context-radius", type=int, default=8)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_segmentation_smoke_v1")
    args = parser.parse_args(argv)

    cases = make_scene_truth_cases(
        count=int(args.count),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfas=_parse_csv(args.cfas),
        psf_sigmas=tuple(float(v) for v in _parse_csv(args.psf_sigmas)),
        use_camerae2e=bool(args.use_camerae2e),
        scene_luminance=float(args.scene_luminance),
    )
    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
    )
    summary = build_scene_truth_segmentation_suite(
        cases,
        config=config,
        boundary_thickness=int(args.boundary_thickness),
        context_radius=int(args.context_radius),
    )
    html_path = write_scene_truth_segmentation_suite(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "case_count": summary["case_count"],
                    "object_count": summary["object_count"],
                    "aggregate": summary["aggregate"],
                }
            ),
            indent=2,
        )
    )
    return 0


def make_scene_truth_cases(
    *,
    count: int,
    width: int,
    height: int,
    scene_scale: float,
    cfas: Sequence[str],
    psf_sigmas: Sequence[float],
    use_camerae2e: bool,
    scene_luminance: float,
) -> Tuple[SceneTruthCase, ...]:
    scene_w = max(int(round(int(width) * max(float(scene_scale), 1.0))), int(width))
    scene_h = max(int(round(int(height) * max(float(scene_scale), 1.0))), int(height))
    cases: list[SceneTruthCase] = []
    for index in range(max(int(count), 1)):
        rgb_high, objects = _make_vector_scene(scene_w, scene_h, seed=101 + index * 17, frame_index=index)
        for cfa in cfas:
            pattern = str(cfa).upper().replace("-", "").replace("_", "")
            for sigma in psf_sigmas:
                rgb_for_camera = _blur_rgb(rgb_high, sigma=max(float(sigma), 0.0) * max(float(scene_scale), 1.0))
                if use_camerae2e:
                    raw = raw_from_camerae2e_rgb(
                        rgb_for_camera,
                        width=width,
                        height=height,
                        cfa_pattern=pattern,
                        scene_luminance=scene_luminance,
                        resize_scene_to_target=False,
                    )
                else:
                    raw = raw_from_rgb_direct(rgb_for_camera, width=width, height=height, cfa_pattern=pattern)
                raw = _with_psf_metadata(raw, psf_sigma=float(sigma))
                case_id = f"scene_{index:03d}_{pattern}_psf{float(sigma):.2f}"
                cases.append(
                    SceneTruthCase(
                        case_id=case_id,
                        rgb_high=rgb_high,
                        objects=tuple(objects),
                        cfa=pattern,
                        psf_sigma=float(sigma),
                        raw=raw,
                        metadata={
                            "width": int(width),
                            "height": int(height),
                            "scene_width": int(scene_w),
                            "scene_height": int(scene_h),
                            "scene_scale": float(scene_scale),
                            "cfa": pattern,
                            "psf_sigma": float(sigma),
                            "use_camerae2e": bool(use_camerae2e),
                            "raw_provenance": dict(raw.provenance),
                        },
                    )
                )
    return tuple(cases)


def build_scene_truth_segmentation_suite(
    cases: Sequence[SceneTruthCase],
    *,
    config: PerceptionISPConfig,
    boundary_thickness: int = 1,
    context_radius: int = 8,
) -> Dict[str, Any]:
    destination_assets: list[tuple[str, Mapping[str, np.ndarray]]] = []
    case_rows = []
    object_rows = []
    for case in cases:
        row, objects, assets = _run_case(
            case,
            config=config,
            boundary_thickness=boundary_thickness,
            context_radius=context_radius,
        )
        case_rows.append(row)
        object_rows.extend(objects)
        destination_assets.append((case.case_id, assets))

    checks = _checks(case_rows, object_rows)
    aggregate = _aggregate(case_rows, object_rows)
    return {
        "name": "Scene-truth object-mask edge evaluation",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "case_count": len(case_rows),
        "object_count": len(object_rows),
        "checks": checks,
        "aggregate": aggregate,
        "by_cfa": _breakdown(object_rows, "cfa"),
        "by_psf": _breakdown(object_rows, "psf_sigma"),
        "by_shape": _breakdown(object_rows, "shape_tag"),
        "cases": case_rows,
        "objects": object_rows,
        "boundary_thickness": int(boundary_thickness),
        "context_radius": int(context_radius),
        "assets_pending": destination_assets,
        "interpretation": (
            "Object masks are generated as vector scene truth before CFA/PSF/raw/ISP. "
            "Metrics compare HumanISP RGB edge, PerceptionISP RGB edge, and PerceptionISP aux maps against the pre-sensor object boundary."
        ),
        "claim_boundary": (
            "This is a scene-truth edge/segmentation smoke test, not a trained segmentation-detector mAP benchmark. "
            "It directly addresses RGB-human-GT bias but uses synthetic vector scenes."
        ),
    }


def write_scene_truth_segmentation_suite(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    asset_dir = destination / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(summary)
    assets_pending = serializable.pop("assets_pending", [])
    asset_manifest = []
    for case_id, assets in assets_pending:
        case_assets = {}
        for name, image in assets.items():
            filename = f"{case_id}_{name}.png"
            _save_image(asset_dir / filename, image)
            case_assets[name] = f"assets/{filename}"
        asset_manifest.append({"case_id": case_id, "assets": case_assets})
    serializable["asset_manifest"] = asset_manifest
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(serializable), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(serializable), encoding="utf-8")
    return html_path


def _run_case(
    case: SceneTruthCase,
    *,
    config: PerceptionISPConfig,
    boundary_thickness: int,
    context_radius: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Mapping[str, np.ndarray]]:
    images = build_pipeline_images(_as_eval_sample(case), config=config)
    human_rgb = np.asarray(images.human_rgb, dtype=np.float64)
    perception_rgb = np.asarray(images.perception_rgb, dtype=np.float64)
    aux_maps = {name: np.asarray(value, dtype=np.float64) for name, value in images.aux_maps.items()}
    shape = human_rgb.shape[:2]
    signals = _signals(human_rgb, perception_rgb, aux_maps)

    object_rows = []
    object_boundaries = []
    object_areas = []
    for obj in case.objects:
        area = _resize_bool_any(obj.mask_high, shape)
        boundary = _resize_bool_any(_mask_boundary(obj.mask_high, thickness=boundary_thickness), shape)
        if not bool(np.any(boundary)):
            boundary = _mask_boundary(area, thickness=1)
        context = _context_mask(area, boundary, radius=context_radius)
        metrics = _boundary_metrics(signals, boundary, context)
        bbox = _mask_bbox(area)
        object_rows.append(
            {
                "case_id": case.case_id,
                "label": obj.label,
                "shape_tag": obj.shape_tag,
                "cfa": case.cfa,
                "psf_sigma": float(case.psf_sigma),
                "area_fraction": float(np.mean(area)),
                "boundary_fraction": float(np.mean(boundary)),
                "aspect_ratio": _bbox_aspect_ratio(bbox),
                **metrics,
            }
        )
        object_boundaries.append(boundary)
        object_areas.append(area)

    union_boundary = np.logical_or.reduce(object_boundaries) if object_boundaries else np.zeros(shape, dtype=bool)
    union_area = np.logical_or.reduce(object_areas) if object_areas else np.zeros(shape, dtype=bool)
    union_context = _context_mask(union_area, union_boundary, radius=context_radius)
    case_metrics = _boundary_metrics(signals, union_boundary, union_context)
    case_metrics.update(
        {
            "object_count": int(len(case.objects)),
            "boundary_fraction": float(np.mean(union_boundary)),
            "finite_outputs": bool(
                np.isfinite(human_rgb).all()
                and np.isfinite(perception_rgb).all()
                and all(np.isfinite(value).all() for value in aux_maps.values())
            ),
            "raw_pattern_remapped": bool(case.raw.provenance.get("pattern_remapped", False)),
        }
    )
    row = {
        "case_id": case.case_id,
        "cfa": case.cfa,
        "psf_sigma": float(case.psf_sigma),
        "metadata": dict(case.metadata),
        "metrics": case_metrics,
    }
    assets = {
        "scene_truth_rgb": _resize_rgb(np.clip(case.rgb_high, 0.0, 1.0), shape),
        "scene_truth_boundary": union_boundary.astype(np.float64),
        "human_rgb": np.clip(human_rgb, 0.0, 1.0),
        "human_edge": signals["human_rgb_edge"],
        "perception_rgb": np.clip(perception_rgb, 0.0, 1.0),
        "perception_edge": signals["perception_rgb_edge"],
        "aux_edge_confidence": signals["aux_edge_confidence"],
        "aux_edge_strength": signals["aux_edge_strength"],
        "aux_edge_evidence": signals["aux_edge_evidence"],
    }
    return row, object_rows, assets


def _as_eval_sample(case: SceneTruthCase) -> Any:
    from .eval_types import EvaluationSample

    return EvaluationSample(
        sample_id=case.case_id,
        raw=case.raw,
        ground_truth=(),
        source="scene_truth_vector_camerae2e" if case.metadata.get("use_camerae2e") else "scene_truth_vector_direct",
        metadata=dict(case.metadata),
        reference_rgb=case.rgb_high,
    )


def _make_vector_scene(width: int, height: int, *, seed: int, frame_index: int) -> tuple[np.ndarray, Tuple[SceneTruthObject, ...]]:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    sky = np.stack([0.18 + 0.10 * yy, 0.24 + 0.10 * yy, 0.34 + 0.15 * yy], axis=2)
    road = np.stack([0.13 + 0.14 * yy, 0.13 + 0.13 * yy, 0.12 + 0.12 * yy], axis=2)
    rgb = np.where((yy[:, :, None] < 0.44), sky, road)
    texture = rng.normal(0.0, 0.018, size=(height, width, 1))
    rgb = np.clip(rgb + texture * (0.35 + 0.65 * (yy[:, :, None] > 0.44)), 0.0, 1.0)
    lane_left = (np.abs(xx - (0.42 - 0.10 * (yy - 0.44))) < 0.006) & (yy > 0.44)
    lane_right = (np.abs(xx - (0.60 + 0.10 * (yy - 0.44))) < 0.006) & (yy > 0.44)
    rgb[lane_left] = np.array([0.85, 0.70, 0.18])
    rgb[lane_right] = np.array([0.86, 0.86, 0.82])

    dx = 0.018 * np.sin(float(frame_index) * 0.7)
    objects: list[SceneTruthObject] = []

    car = _poly_mask(width, height, [(0.55 + dx, 0.61), (0.77 + dx, 0.60), (0.82 + dx, 0.75), (0.51 + dx, 0.78)])
    rgb[car] = np.array([0.055, 0.060, 0.068])
    window = _poly_mask(width, height, [(0.61 + dx, 0.63), (0.73 + dx, 0.63), (0.75 + dx, 0.68), (0.59 + dx, 0.69)])
    rgb[window] = np.array([0.18, 0.23, 0.30])
    objects.append(SceneTruthObject("car", car, "medium_polygon"))

    person_body = _rect_mask(width, height, (0.355 - dx * 0.5, 0.56, 0.395 - dx * 0.5, 0.82))
    person_head = _ellipse_mask(width, height, (0.362 - dx * 0.5, 0.515, 0.390 - dx * 0.5, 0.555))
    person = person_body | person_head
    rgb[person] = np.array([0.035, 0.040, 0.046])
    objects.append(SceneTruthObject("person", person, "thin_small"))

    pole = _rect_mask(width, height, (0.845, 0.22, 0.855, 0.78))
    rgb[pole] = np.array([0.18, 0.18, 0.16])
    objects.append(SceneTruthObject("pole", pole, "thin_long"))

    traffic = _ellipse_mask(width, height, (0.815, 0.205, 0.842, 0.250))
    rgb[traffic] = np.array([1.0, 0.08, 0.04])
    objects.append(SceneTruthObject("traffic_light", traffic, "tiny_bright"))

    cable = _poly_mask(width, height, [(0.12, 0.34), (0.52, 0.36), (0.52, 0.372), (0.12, 0.352)])
    rgb[cable] = np.array([0.10, 0.10, 0.105])
    objects.append(SceneTruthObject("cable", cable, "thin_long"))
    return np.clip(rgb, 0.0, 1.0), tuple(objects)


def _signals(human_rgb: np.ndarray, perception_rgb: np.ndarray, aux_maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    edge_confidence = np.asarray(aux_maps.get("edge_confidence", np.zeros(human_rgb.shape[:2])), dtype=np.float64)
    edge_strength = _normalize(np.asarray(aux_maps.get("edge_strength", np.zeros_like(edge_confidence)), dtype=np.float64))
    if "edge_evidence" in aux_maps:
        edge_evidence = np.clip(np.asarray(aux_maps["edge_evidence"], dtype=np.float64), 0.0, 1.0)
    else:
        edge_evidence = np.sqrt(np.clip(edge_strength * np.clip(edge_confidence, 0.0, 1.0), 0.0, 1.0))
    return {
        "human_rgb_edge": _edge_strength(_luma(human_rgb)),
        "perception_rgb_edge": _edge_strength(_luma(perception_rgb)),
        "aux_edge_strength": edge_strength,
        "aux_edge_confidence": np.clip(edge_confidence, 0.0, 1.0),
        "aux_edge_evidence": edge_evidence,
        "aux_strength_gated_confidence": np.clip(edge_strength * (0.25 + 0.75 * np.clip(edge_confidence, 0.0, 1.0)), 0.0, 1.0),
    }


def _boundary_metrics(signals: Mapping[str, np.ndarray], boundary: np.ndarray, context: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for signal_name in SIGNALS:
        signal = np.asarray(signals[signal_name], dtype=np.float64)
        predicted = _edge_mask(signal)
        f1 = _f1_metrics(predicted, boundary, tolerance=2)
        on = _masked_mean(signal, boundary)
        off = _masked_mean(signal, context)
        metrics[f"{signal_name}_boundary_precision"] = f1["precision"]
        metrics[f"{signal_name}_boundary_recall"] = f1["recall"]
        metrics[f"{signal_name}_boundary_f1"] = f1["f1"]
        metrics[f"{signal_name}_on_boundary"] = on
        metrics[f"{signal_name}_off_boundary_context"] = off
        metrics[f"{signal_name}_boundary_separation"] = float(on - off)
    human = metrics["human_rgb_edge_boundary_f1"]
    for signal_name in SIGNALS:
        if signal_name == "human_rgb_edge":
            continue
        metrics[f"{signal_name}_minus_human_boundary_f1"] = float(metrics[f"{signal_name}_boundary_f1"] - human)
    metrics["perception_rgb_minus_human_boundary_f1"] = metrics["perception_rgb_edge_minus_human_boundary_f1"]
    metrics["aux_strength_minus_human_boundary_f1"] = metrics["aux_edge_strength_minus_human_boundary_f1"]
    metrics["aux_confidence_minus_human_boundary_f1"] = metrics["aux_edge_confidence_minus_human_boundary_f1"]
    return metrics


def _aggregate(case_rows: Sequence[Mapping[str, Any]], object_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in object_rows if isinstance(row, Mapping)]
    out: Dict[str, Any] = {"case_count": len(case_rows), "object_count": len(rows)}
    for signal_name in SIGNALS:
        for suffix in ("boundary_f1", "boundary_precision", "boundary_recall", "boundary_separation"):
            key = f"{signal_name}_{suffix}"
            out[f"{key}_mean"] = _mean([float(row.get(key, 0.0)) for row in rows])
    for key in ("perception_rgb_minus_human_boundary_f1", "aux_strength_minus_human_boundary_f1", "aux_confidence_minus_human_boundary_f1"):
        values = [float(row.get(key, 0.0)) for row in rows]
        out[f"{key}_mean"] = _mean(values)
        out[f"{key}_win_rate"] = _win_rate(values)
    for signal_name in SIGNALS:
        if signal_name == "human_rgb_edge":
            continue
        key = f"{signal_name}_minus_human_boundary_f1"
        if f"{key}_mean" in out:
            continue
        values = [float(row.get(key, 0.0)) for row in rows]
        out[f"{key}_mean"] = _mean(values)
        out[f"{key}_win_rate"] = _win_rate(values)
    return out


def _breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> list[Dict[str, Any]]:
    groups: Dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    out = []
    for value, group in sorted(groups.items()):
        payload: Dict[str, Any] = {key: value, "object_count": len(group)}
        for signal_name in SIGNALS:
            payload[f"{signal_name}_boundary_f1_mean"] = _mean([float(row.get(f"{signal_name}_boundary_f1", 0.0)) for row in group])
            payload[f"{signal_name}_boundary_separation_mean"] = _mean([float(row.get(f"{signal_name}_boundary_separation", 0.0)) for row in group])
        payload["perception_rgb_minus_human_boundary_f1_mean"] = _mean([float(row.get("perception_rgb_minus_human_boundary_f1", 0.0)) for row in group])
        payload["aux_confidence_minus_human_boundary_f1_mean"] = _mean([float(row.get("aux_confidence_minus_human_boundary_f1", 0.0)) for row in group])
        payload["aux_edge_evidence_minus_human_boundary_f1_mean"] = _mean([float(row.get("aux_edge_evidence_minus_human_boundary_f1", 0.0)) for row in group])
        payload["aux_strength_gated_confidence_minus_human_boundary_f1_mean"] = _mean(
            [float(row.get("aux_strength_gated_confidence_minus_human_boundary_f1", 0.0)) for row in group]
        )
        out.append(payload)
    return out


def _checks(case_rows: Sequence[Mapping[str, Any]], object_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    case_metrics = [row.get("metrics", {}) for row in case_rows if isinstance(row.get("metrics"), Mapping)]
    finite = all(bool(row.get("finite_outputs")) for row in case_metrics)
    boundaries = [float(row.get("boundary_fraction", 0.0)) for row in case_metrics]
    bounded = all(
        0.0 <= float(row.get(f"{signal_name}_boundary_f1", 0.0)) <= 1.0
        for row in object_rows
        for signal_name in SIGNALS
    )
    remaps = [bool(row.get("metrics", {}).get("raw_pattern_remapped", False)) for row in case_rows]
    return [
        {
            "id": "finite_outputs",
            "status": "pass" if finite else "fail",
            "description": "HumanISP RGB, PerceptionISP RGB, and aux maps are finite.",
            "criteria": [{"metric": "finite_outputs", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "scene_truth_boundaries_present",
            "status": "pass" if object_rows and min(boundaries or [0.0]) > 0.0 else "fail",
            "description": "Pre-sensor object mask boundaries exist after resizing to sensor resolution.",
            "criteria": [
                {"metric": "object_count", "value": len(object_rows), "threshold": 1, "pass": bool(object_rows)},
                {"metric": "min_boundary_fraction", "value": min(boundaries or [0.0]), "threshold": 0.0, "pass": min(boundaries or [0.0]) > 0.0},
            ],
        },
        {
            "id": "bounded_metrics",
            "status": "pass" if bounded else "fail",
            "description": "All object-boundary F1 metrics are bounded.",
            "criteria": [{"metric": "bounded", "value": bool(bounded), "pass": bool(bounded)}],
        },
        {
            "id": "native_cfa_not_remapped",
            "status": "pass" if not any(remaps) else "warn",
            "description": "CameraE2E native CFA should be preserved when available.",
            "criteria": [{"metric": "pattern_remapped_count", "value": sum(1 for value in remaps if value), "threshold": 0, "pass": not any(remaps)}],
        },
    ]


def _with_psf_metadata(raw: RawFrame, *, psf_sigma: float) -> RawFrame:
    height, width = _raw_height_width(raw.data)
    sigma = max(float(psf_sigma), 0.0)
    calibration = replace(raw.calibration, psf_sigma_map=np.full((height, width), sigma, dtype=np.float64))
    metadata = replace(
        raw.metadata,
        calibration_id=f"{raw.metadata.calibration_id}_scene_truth_psf_{sigma:.2f}",
        lens_profile_id=f"{raw.metadata.lens_profile_id}_scene_truth_psf_{sigma:.2f}",
    )
    return replace(raw, metadata=metadata, calibration=calibration, provenance={**dict(raw.provenance), "scene_truth_psf_sigma": sigma})


def _raw_height_width(raw_data: Any) -> tuple[int, int]:
    arr = np.asarray(raw_data)
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3 and arr.shape[0] <= 8:
        return int(arr.shape[1]), int(arr.shape[2])
    if arr.ndim == 3:
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"unsupported RAW shape: {arr.shape}")


def _rect_mask(width: int, height: int, box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).rectangle((x1 * width, y1 * height, x2 * width, y2 * height), fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _ellipse_mask(width: int, height: int, box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).ellipse((x1 * width, y1 * height, x2 * width, y2 * height), fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _poly_mask(width: int, height: int, points: Sequence[tuple[float, float]]) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([(x * width, y * height) for x, y in points], fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _mask_boundary(mask: np.ndarray, *, thickness: int = 1) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if not bool(np.any(values)):
        return values.copy()
    eroded = values.copy()
    for _ in range(max(int(thickness), 1)):
        eroded = _erode(eroded)
    return values & np.logical_not(eroded)


def _erode(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1, mode="constant", constant_values=False)
    out = np.ones_like(values, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return out


def _context_mask(area: np.ndarray, boundary: np.ndarray, *, radius: int) -> np.ndarray:
    near = _dilate(area, radius=max(int(radius), 1))
    blocked = _dilate(boundary, radius=2)
    context = near & np.logical_not(blocked)
    if not bool(np.any(context)):
        context = np.logical_not(blocked)
    return context


def _resize_bool_any(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    resized = image.resize((width, height), resample=Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.uint8) > 0


def _resize_rgb(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.shape[:2] == (height, width):
        return arr[:, :, :3]
    image = Image.fromarray(np.round(np.clip(arr[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float64) / 255.0


def _blur_rgb(rgb: np.ndarray, *, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.asarray(rgb, dtype=np.float64)
    image = Image.fromarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float64) / 255.0


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs) or not len(ys):
        return (0.0, 0.0, 0.0, 0.0)
    height, width = mask.shape
    return (float(xs.min() / width), float(ys.min() / height), float((xs.max() + 1) / width), float((ys.max() + 1) / height))


def _bbox_aspect_ratio(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return float(max(w / h, h / w))


def _save_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=2)
    else:
        rgb = arr[:, :, :3]
    Image.fromarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _win_rate(values: Sequence[float]) -> float | None:
    return float(np.mean([float(value) > 0.0 for value in values])) if values else None


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split(",") if token.strip())


def _render_html(summary: Mapping[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    rows = []
    for row in summary.get("by_shape", []):
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('shape_tag', '')))}</td>"
            f"<td>{int(row.get('object_count', 0))}</td>"
            f"<td>{_fmt(row.get('human_rgb_edge_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('perception_rgb_edge_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('aux_edge_evidence_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('perception_rgb_minus_human_boundary_f1_mean'), signed=True)}</td>"
            f"<td>{_fmt(row.get('aux_edge_evidence_minus_human_boundary_f1_mean'), signed=True)}</td>"
            "</tr>"
        )
    cfa_rows = []
    for row in summary.get("by_cfa", []):
        cfa_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('cfa', '')))}</td>"
            f"<td>{int(row.get('object_count', 0))}</td>"
            f"<td>{_fmt(row.get('human_rgb_edge_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('perception_rgb_edge_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('aux_edge_evidence_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('aux_edge_evidence_minus_human_boundary_f1_mean'), signed=True)}</td>"
            "</tr>"
        )
    assets = summary.get("asset_manifest", [])[:4]
    asset_blocks = []
    for item in assets:
        paths = item.get("assets", {})
        cols = []
        for name in ("scene_truth_rgb", "scene_truth_boundary", "human_edge", "perception_edge", "aux_edge_evidence"):
            if name in paths:
                cols.append(f"<div><b>{html_lib.escape(name)}</b><img src='{html_lib.escape(paths[name])}'></div>")
        asset_blocks.append(f"<h3>{html_lib.escape(str(item.get('case_id', '')))}</h3><div class='assets'>{''.join(cols)}</div>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Segmentation Edge Evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #16202a; }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .tile {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 10px 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .assets {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 22px; }}
    .assets img {{ width: 100%; border: 1px solid #d8dee4; background: #000; image-rendering: auto; }}
  </style>
</head>
<body>
  <h1>Scene-truth Segmentation Edge Evaluation</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <div class="grid">
    <div class="tile"><b>Status</b><br>{html_lib.escape(str(summary.get('status', '')))}</div>
    <div class="tile"><b>Cases</b><br>{int(summary.get('case_count', 0))}</div>
    <div class="tile"><b>Objects</b><br>{int(summary.get('object_count', 0))}</div>
    <div class="tile"><b>Aux evidence dF1</b><br>{_fmt(aggregate.get('aux_edge_evidence_minus_human_boundary_f1_mean'), signed=True)}</div>
  </div>
  <h2>Aggregate</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Human RGB boundary F1 mean</td><td>{_fmt(aggregate.get('human_rgb_edge_boundary_f1_mean'))}</td></tr>
      <tr><td>Perception RGB boundary F1 mean</td><td>{_fmt(aggregate.get('perception_rgb_edge_boundary_f1_mean'))}</td></tr>
      <tr><td>Aux confidence boundary F1 mean</td><td>{_fmt(aggregate.get('aux_edge_confidence_boundary_f1_mean'))}</td></tr>
      <tr><td>Aux evidence boundary F1 mean</td><td>{_fmt(aggregate.get('aux_edge_evidence_boundary_f1_mean'))}</td></tr>
      <tr><td>Aux gated strength boundary F1 mean</td><td>{_fmt(aggregate.get('aux_strength_gated_confidence_boundary_f1_mean'))}</td></tr>
      <tr><td>Perception RGB minus Human F1 mean</td><td>{_fmt(aggregate.get('perception_rgb_minus_human_boundary_f1_mean'), signed=True)}</td></tr>
      <tr><td>Aux confidence minus Human F1 mean</td><td>{_fmt(aggregate.get('aux_confidence_minus_human_boundary_f1_mean'), signed=True)}</td></tr>
      <tr><td>Aux evidence minus Human F1 mean</td><td>{_fmt(aggregate.get('aux_edge_evidence_minus_human_boundary_f1_mean'), signed=True)}</td></tr>
      <tr><td>Aux gated strength minus Human F1 mean</td><td>{_fmt(aggregate.get('aux_strength_gated_confidence_minus_human_boundary_f1_mean'), signed=True)}</td></tr>
    </tbody>
  </table>
  <h2>Shape Slices</h2>
  <table><thead><tr><th>Shape</th><th>N</th><th>Human F1</th><th>Perception RGB F1</th><th>Aux Evidence F1</th><th>Perception dF1</th><th>Aux Evidence dF1</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>CFA Slices</h2>
  <table><thead><tr><th>CFA</th><th>N</th><th>Human F1</th><th>Perception RGB F1</th><th>Aux Evidence F1</th><th>Aux Evidence dF1</th></tr></thead><tbody>{''.join(cfa_rows)}</tbody></table>
  <h2>Visual Cases</h2>
  {''.join(asset_blocks)}
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return html_lib.escape(str(value))
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
