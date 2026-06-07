"""Scene-edge confidence comparison for HumanISP and PerceptionISP."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from .camerae2e_bridge import raw_from_camerae2e_rgb, raw_from_rgb_direct
from .eval_types import EvaluationSample
from .pipeline import PerceptionISPPipeline
from .types import PerceptionISPConfig, json_ready


SCENE_EDGE_CONFIDENCE_SUMMARY = "scene_edge_confidence_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compare scene-edge confidence from HumanISP RGB and PerceptionISP aux maps.")
    parser.add_argument("--source", choices=("sample-image", "camerae2e-synthetic", "yolo-dataset", "kitti-dataset"), default="sample-image")
    parser.add_argument("--image-path", default="data/sample_images/bus.jpg")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--scene-scale", type=float, default=2.0, help="Reference scene scale relative to sensor output for source-edge oracle.")
    parser.add_argument("--cfa", default="auto")
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--output-dir", default="reports/perception_scene_edge_confidence_sample_image")
    args = parser.parse_args(argv)

    samples = load_scene_edge_samples(
        source=str(args.source),
        image_path=args.image_path,
        dataset=args.dataset,
        split=str(args.split),
        count=int(args.count),
        offset=int(args.offset),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfa_pattern=str(args.cfa),
        use_camerae2e=not bool(args.no_camerae2e),
    )
    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    summary = build_scene_edge_confidence_suite(samples, config=config)
    html_path = write_scene_edge_confidence_suite(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SCENE_EDGE_CONFIDENCE_SUMMARY),
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


def load_scene_edge_samples(
    *,
    source: str,
    image_path: str | Path = "data/sample_images/bus.jpg",
    dataset: str | None = None,
    split: str = "val",
    count: int = 1,
    offset: int = 0,
    width: int = 320,
    height: int = 240,
    scene_scale: float = 1.0,
    cfa_pattern: str = "auto",
    use_camerae2e: bool = True,
) -> Tuple[EvaluationSample, ...]:
    normalized = str(source)
    if normalized == "sample-image":
        return (_sample_image(Path(image_path), width=width, height=height, scene_scale=scene_scale, cfa_pattern=cfa_pattern, use_camerae2e=use_camerae2e),)
    if normalized == "camerae2e-synthetic":
        from .synthetic_eval import make_camerae2e_synthetic_evaluation_samples, make_synthetic_evaluation_samples

        if use_camerae2e:
            return make_camerae2e_synthetic_evaluation_samples(count=count, width=width, height=height, cfa_pattern=cfa_pattern)
        samples = make_synthetic_evaluation_samples(count=count, width=width, height=height, cfa_pattern="RGGB" if cfa_pattern == "auto" else cfa_pattern)
        return tuple(replace(sample, reference_rgb=np.asarray(sample.raw.data[0], dtype=np.float64)) for sample in samples)
    if normalized == "yolo-dataset":
        if not dataset:
            raise ValueError("--dataset is required for --source yolo-dataset")
        from .yolo_dataset import load_yolo_detection_samples

        return load_yolo_detection_samples(
            dataset,
            split=split,
            limit=count,
            offset=offset,
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            use_camerae2e=use_camerae2e,
        )
    if normalized == "kitti-dataset":
        if not dataset:
            raise ValueError("--dataset is required for --source kitti-dataset")
        from .kitti_dataset import load_kitti_detection_samples

        return load_kitti_detection_samples(
            dataset,
            split=split,
            limit=count,
            offset=offset,
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            use_camerae2e=use_camerae2e,
        )
    raise ValueError(f"unsupported scene-edge source: {source!r}")


def build_scene_edge_confidence_suite(
    samples: Sequence[EvaluationSample],
    *,
    config: PerceptionISPConfig | None = None,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("scene-edge confidence suite needs at least one sample")
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    cases = [_run_case(sample, pipeline=pipeline) for sample in samples]
    checks = _checks(cases)
    return {
        "sample_count": len(cases),
        "cases": cases,
        "checks": checks,
        "aggregate": _aggregate(cases),
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "interpretation": (
            "This suite uses the source scene RGB edge map as a proxy oracle, then compares HumanISP RGB edge evidence, "
            "PerceptionISP RGB edge evidence, and PerceptionISP native aux edge confidence against it. HumanISP does not "
            "produce a native confidence map here; its row is an RGB-derived edge-strength proxy."
        ),
        "claim_boundary": (
            "This is front-end scene-edge confidence evidence. It is not object-boundary ground truth and it is not detector-performance evidence."
        ),
    }


def write_scene_edge_confidence_suite(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _materialize_assets(summary, destination)
    (destination / SCENE_EDGE_CONFIDENCE_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _sample_image(path: Path, *, width: int, height: int, scene_scale: float, cfa_pattern: str, use_camerae2e: bool) -> EvaluationSample:
    scene_w = max(int(round(float(width) * max(float(scene_scale), 1.0))), int(width))
    scene_h = max(int(round(float(height) * max(float(scene_scale), 1.0))), int(height))
    rgb = _load_rgb(path, width=scene_w, height=scene_h)
    raw = (
        raw_from_camerae2e_rgb(rgb, width=width, height=height, cfa_pattern=cfa_pattern, resize_scene_to_target=False)
        if use_camerae2e
        else raw_from_rgb_direct(rgb, width=width, height=height, cfa_pattern="RGGB" if cfa_pattern == "auto" else cfa_pattern)
    )
    raw.metadata = replace(raw.metadata, camera_id="scene_edge_confidence_sample", module_serial=str(path))
    return EvaluationSample(
        sample_id=path.stem,
        raw=raw,
        ground_truth=(),
        source="sample_image_camerae2e" if use_camerae2e else "sample_image_direct",
        metadata={
            "image_path": str(path),
            "scene_width": int(scene_w),
            "scene_height": int(scene_h),
            "width": int(width),
            "height": int(height),
            "scene_scale": float(scene_scale),
            "requested_cfa_pattern": str(cfa_pattern),
            "cfa_pattern": raw.metadata.cfa_pattern,
            "raw_provenance": dict(raw.provenance),
        },
        reference_rgb=rgb,
    )


def _run_case(sample: EvaluationSample, *, pipeline: PerceptionISPPipeline) -> Dict[str, Any]:
    result = pipeline.run(sample.raw)
    human_rgb = np.asarray(result.human_rgb if result.human_rgb is not None else result.vision_rgb, dtype=np.float64)
    perception_rgb = np.asarray(result.vision_rgb, dtype=np.float64)
    reference_rgb, source_strength, source_edge = _source_edge_proxy(sample, human_rgb.shape[:2])
    human_strength = _edge_strength(_luma(human_rgb))
    perception_strength = _edge_strength(_luma(perception_rgb))
    aux_confidence = np.asarray(result.maps["edge_confidence"], dtype=np.float64)
    aux_strength = np.asarray(result.maps["edge_strength"], dtype=np.float64)
    metrics = _case_metrics(
        source_strength=source_strength,
        source_edge=source_edge,
        human_strength=human_strength,
        perception_strength=perception_strength,
        aux_confidence=aux_confidence,
        aux_strength=aux_strength,
    )
    metrics["finite_outputs"] = bool(
        np.isfinite(reference_rgb).all()
        and np.isfinite(human_rgb).all()
        and np.isfinite(perception_rgb).all()
        and all(np.isfinite(value).all() for value in result.maps.values())
    )
    metrics["raw_pattern_remapped"] = bool(sample.raw.provenance.get("pattern_remapped", False))
    return {
        "id": str(sample.sample_id),
        "source": str(sample.source),
        "cfa_pattern": str(result.metadata.get("frame", {}).get("cfa_pattern", sample.raw.metadata.cfa_pattern)),
        "raw_provenance": dict(sample.raw.provenance),
        "metrics": metrics,
        "_assets_source": {
            "reference_rgb": reference_rgb,
            "source_edge": source_edge.astype(np.float64),
            "human_rgb": human_rgb,
            "human_edge_proxy": human_strength,
            "perception_rgb": perception_rgb,
            "perception_edge_proxy": perception_strength,
            "aux_edge_confidence": aux_confidence,
            "aux_edge_strength": aux_strength,
        },
    }


def _source_edge_proxy(sample: EvaluationSample, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample.reference_rgb is None:
        raise ValueError(f"sample {sample.sample_id!r} does not contain reference_rgb; scene-edge proxy cannot be computed")
    rgb = np.asarray(sample.reference_rgb, dtype=np.float64)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"sample {sample.sample_id!r} reference_rgb must be HxWx3")
    high_rgb = np.clip(rgb[:, :, :3], 0.0, 1.0)
    high_strength = _edge_strength(_luma(high_rgb))
    high_edge = _edge_mask(high_strength)
    height, width = int(shape[0]), int(shape[1])
    display_rgb = _resize_rgb(high_rgb, height, width)
    source_strength = _resize_gray(high_strength, height, width)
    source_edge = _resize_bool_any(high_edge, height, width)
    return display_rgb, source_strength, source_edge


def _case_metrics(
    *,
    source_strength: np.ndarray,
    source_edge: np.ndarray,
    human_strength: np.ndarray,
    perception_strength: np.ndarray,
    aux_confidence: np.ndarray,
    aux_strength: np.ndarray,
) -> Dict[str, float]:
    off_source = np.logical_not(_dilate(source_edge, radius=2))
    metrics: Dict[str, float] = {
        "source_edge_fraction": float(np.mean(source_edge)),
        "source_edge_strength_mean": _mean(source_strength),
    }
    for prefix, signal in (
        ("human_rgb_proxy", human_strength),
        ("perception_rgb_proxy", perception_strength),
        ("perception_aux_confidence", aux_confidence),
        ("perception_aux_strength", aux_strength),
    ):
        edge = _edge_mask(signal)
        on = _masked_mean(signal, source_edge)
        off = _masked_mean(signal, off_source)
        metrics[f"{prefix}_on_source_edge"] = on
        metrics[f"{prefix}_off_source_edge"] = off
        metrics[f"{prefix}_scene_edge_separation"] = float(on - off)
        metrics[f"{prefix}_source_edge_correlation"] = _correlation(signal, source_strength)
        metrics[f"{prefix}_source_edge_f1"] = _f1_metrics(edge, source_edge)["f1"]
    metrics["aux_confidence_minus_human_proxy_separation"] = float(
        metrics["perception_aux_confidence_scene_edge_separation"] - metrics["human_rgb_proxy_scene_edge_separation"]
    )
    return metrics


def _checks(cases: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    metrics_rows = [case.get("metrics", {}) for case in cases if isinstance(case.get("metrics"), Mapping)]
    finite = all(bool(row.get("finite_outputs")) for row in metrics_rows)
    source_fractions = [float(row.get("source_edge_fraction", 0.0)) for row in metrics_rows]
    aux_separations = [float(row.get("perception_aux_confidence_scene_edge_separation", 0.0)) for row in metrics_rows]
    human_separations = [float(row.get("human_rgb_proxy_scene_edge_separation", 0.0)) for row in metrics_rows]
    perception_separations = [float(row.get("perception_rgb_proxy_scene_edge_separation", 0.0)) for row in metrics_rows]
    raw_remaps = [bool(row.get("raw_pattern_remapped")) for row in metrics_rows]
    bounded = all(
        0.0 <= float(row.get(name, 0.0)) <= 1.0
        for row in metrics_rows
        for name in (
            "human_rgb_proxy_source_edge_f1",
            "perception_rgb_proxy_source_edge_f1",
            "perception_aux_confidence_source_edge_f1",
            "perception_aux_strength_source_edge_f1",
        )
    )
    return [
        {
            "id": "finite_scene_edge_outputs",
            "description": "Reference RGB, HumanISP RGB, PerceptionISP RGB, and aux maps are finite.",
            "status": "pass" if finite else "fail",
            "criteria": [{"metric": "finite_outputs", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "reference_scene_edges_present",
            "description": "The source scene RGB edge proxy is non-empty.",
            "status": "pass" if min(source_fractions or [0.0]) > 0.005 else "fail",
            "criteria": [
                {"metric": "min_source_edge_fraction", "value": min(source_fractions or [0.0]), "threshold": 0.005, "pass": min(source_fractions or [0.0]) > 0.005}
            ],
        },
        {
            "id": "scene_edge_metrics_bounded",
            "description": "Scene-edge F1 metrics are bounded in [0, 1].",
            "status": "pass" if bounded else "fail",
            "criteria": [{"metric": "bounded_case_count", "value": sum(1 for _ in metrics_rows), "threshold": len(metrics_rows), "pass": bool(bounded)}],
        },
        {
            "id": "human_and_perception_edges_track_scene_edges",
            "description": "HumanISP RGB proxy, PerceptionISP RGB proxy, and PerceptionISP aux confidence should be higher on source-scene edges than off edges.",
            "status": "pass" if min(human_separations or [0.0]) > 0.0 and min(perception_separations or [0.0]) > 0.0 and min(aux_separations or [0.0]) > 0.0 else "fail",
            "criteria": [
                {"metric": "min_human_rgb_proxy_scene_edge_separation", "value": min(human_separations or [0.0]), "threshold": 0.0, "pass": min(human_separations or [0.0]) > 0.0},
                {"metric": "min_perception_rgb_proxy_scene_edge_separation", "value": min(perception_separations or [0.0]), "threshold": 0.0, "pass": min(perception_separations or [0.0]) > 0.0},
                {"metric": "min_perception_aux_confidence_scene_edge_separation", "value": min(aux_separations or [0.0]), "threshold": 0.0, "pass": min(aux_separations or [0.0]) > 0.0},
            ],
        },
        {
            "id": "camerae2e_cfa_pattern_preserved",
            "description": "CameraE2E true CFA mosaic evidence should avoid source/target CFA remapping.",
            "status": "pass" if not any(raw_remaps) else "fail",
            "criteria": [{"metric": "pattern_remapped_count", "value": sum(1 for value in raw_remaps if value), "threshold": 0, "pass": not any(raw_remaps)}],
        },
    ]


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics_rows = [case.get("metrics", {}) for case in cases if isinstance(case.get("metrics"), Mapping)]
    keys = (
        "source_edge_fraction",
        "human_rgb_proxy_scene_edge_separation",
        "perception_rgb_proxy_scene_edge_separation",
        "perception_aux_confidence_scene_edge_separation",
        "perception_aux_strength_scene_edge_separation",
        "human_rgb_proxy_source_edge_f1",
        "perception_rgb_proxy_source_edge_f1",
        "perception_aux_confidence_source_edge_f1",
        "perception_aux_strength_source_edge_f1",
    )
    return {f"{key}_mean": _mean([float(row.get(key, 0.0)) for row in metrics_rows]) for key in keys}


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


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()))
    case_rows = "".join(_case_row(row, destination) for row in summary.get("cases", ()))
    aggregate_rows = "".join(_aggregate_row(key, value) for key, value in sorted((summary.get("aggregate", {}) or {}).items()))
    status = str(summary.get("status", ""))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Scene Edge Confidence</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    img {{ width: 84px; height: auto; margin-right: 4px; border: 1px solid #d8ded7; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Scene Edge Confidence</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</code>. Samples: {int(summary.get('sample_count', 0))}.</p>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Status</th><th>Check</th><th>Description</th><th>Criteria</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>
  <h2>Aggregate</h2>
  <table>
    <thead><tr><th>Metric</th><th>Mean</th></tr></thead>
    <tbody>{aggregate_rows}</tbody>
  </table>
  <h2>Cases</h2>
  <table>
    <thead><tr><th>Sample</th><th>Visuals</th><th>Human Sep</th><th>Perception RGB Sep</th><th>Aux Conf Sep</th><th>Human F1</th><th>Perception RGB F1</th><th>Aux Conf F1</th><th>CFA</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{SCENE_EDGE_CONFIDENCE_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    criteria = []
    for item in row.get("criteria", ()):
        if not isinstance(item, Mapping):
            continue
        parts = [str(item.get("metric", ""))]
        if item.get("value") is not None:
            parts.append(f"value={_fmt(item.get('value')) if not isinstance(item.get('value'), bool) else item.get('value')}")
        if item.get("threshold") is not None:
            parts.append(f"threshold={_fmt(item.get('threshold'))}")
        parts.append(f"pass={bool(item.get('pass'))}")
        criteria.append(" ".join(parts))
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape('; '.join(criteria))}</td>"
        "</tr>"
    )


def _aggregate_row(key: str, value: Any) -> str:
    return f"<tr><td><code>{html_lib.escape(str(key))}</code></td><td>{_fmt(value)}</td></tr>"


def _case_row(row: Mapping[str, Any], destination: Path) -> str:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    assets = row.get("assets", {}) if isinstance(row.get("assets"), Mapping) else {}
    thumbs = " ".join(
        _asset_img(path, destination)
        for path in (
            assets.get("reference_rgb"),
            assets.get("source_edge"),
            assets.get("human_rgb"),
            assets.get("human_edge_proxy"),
            assets.get("perception_rgb"),
            assets.get("perception_edge_proxy"),
            assets.get("aux_edge_confidence"),
            assets.get("aux_edge_strength"),
        )
        if path
    )
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code><br>{html_lib.escape(str(row.get('source', '')))}</td>"
        f"<td>{thumbs}</td>"
        f"<td>{_fmt(metrics.get('human_rgb_proxy_scene_edge_separation'), signed=True)}</td>"
        f"<td>{_fmt(metrics.get('perception_rgb_proxy_scene_edge_separation'), signed=True)}</td>"
        f"<td>{_fmt(metrics.get('perception_aux_confidence_scene_edge_separation'), signed=True)}</td>"
        f"<td>{_fmt(metrics.get('human_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('perception_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(metrics.get('perception_aux_confidence_source_edge_f1'))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        "</tr>"
    )


def _save_png(image: np.ndarray, path: Path) -> None:
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


def _asset_img(path: str, destination: Path) -> str:
    relative = os.path.relpath(str(destination / path), start=str(destination))
    return f"<img src=\"{html_lib.escape(relative)}\" alt=\"{html_lib.escape(os.path.basename(path))}\">"


def _load_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float64) / 255.0


def _resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.shape[:2] == (height, width):
        return np.clip(rgb, 0.0, 1.0)
    pil = Image.fromarray(_to_uint8_rgb(rgb))
    resized = pil.resize((int(width), int(height)))
    return np.asarray(resized, dtype=np.float64) / 255.0


def _resize_gray(image: np.ndarray, height: int, width: int) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float64)
    if gray.shape[:2] == (height, width):
        return np.clip(gray, 0.0, 1.0)
    pil = Image.fromarray(_to_uint8_gray(gray))
    resized = pil.resize((int(width), int(height)))
    return np.asarray(resized, dtype=np.float64) / 255.0


def _resize_bool_any(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape[:2] == (height, width):
        return values.copy()
    source_h, source_w = values.shape[:2]
    if source_h >= height and source_w >= width and source_h % height == 0 and source_w % width == 0:
        factor_y = source_h // height
        factor_x = source_w // width
        cropped = values[: height * factor_y, : width * factor_x]
        return np.any(cropped.reshape(height, factor_y, width, factor_x), axis=(1, 3))
    pil = Image.fromarray(values.astype(np.uint8) * 255)
    resized = pil.resize((int(width), int(height)))
    return np.asarray(resized, dtype=np.uint8) > 0


def _luma(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def _edge_strength(values: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(values, dtype=np.float64))
    return _normalize(np.sqrt(gx * gx + gy * gy))


def _edge_mask(strength: np.ndarray, percentile: float = 88.0) -> np.ndarray:
    values = np.asarray(strength, dtype=np.float64)
    threshold = max(float(np.percentile(values, float(percentile))), float(np.max(values)) * 0.10, 1.0e-6)
    return values >= threshold


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


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size <= 1:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.clip(np.sum(x * y) / denom, -1.0, 1.0))


def _f1_metrics(predicted: np.ndarray, reference: np.ndarray, *, tolerance: int = 1) -> Dict[str, float]:
    pred = np.asarray(predicted, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if not bool(np.any(pred)) and not bool(np.any(ref)):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not bool(np.any(pred)) or not bool(np.any(ref)):
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


def _mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
