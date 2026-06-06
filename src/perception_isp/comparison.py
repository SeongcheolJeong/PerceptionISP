"""HumanISP vs PerceptionISP comparison runner."""

from __future__ import annotations

import json
import html as html_lib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .detectors import AuxMapRiskDetector, DetectorAdapter, NumpyRiskObjectDetector, fuse_rgb_aux_results
from .eval_types import BoundingBox, DetectorResult, EvaluationSample, PipelineImageSet
from .metrics import aggregate_metric_rows, evaluate_detections
from .pipeline import PerceptionISPPipeline
from .types import PerceptionISPConfig, json_ready


INPUT_ORDER = ("reference_rgb", "human_rgb", "perception_rgb", "perception_fusion_rgb_aux", "perception_aux_rgb")
PIPELINE_INPUT_NAMES = ("human_rgb", "perception_rgb", "perception_aux_rgb")
AREA_BUCKETS = ("small", "medium", "large")


def build_pipeline_images(sample: EvaluationSample, config: PerceptionISPConfig | None = None) -> PipelineImageSet:
    result = PerceptionISPPipeline(config=config).run(sample.raw)
    human = result.human_rgb if result.human_rgb is not None else result.vision_rgb
    aux_rgb = np.stack(
        [
            np.asarray(result.maps["edge_strength"], dtype=np.float64),
            np.asarray(result.maps["saturation"], dtype=np.float64),
            np.asarray(result.maps["snr_map"], dtype=np.float64),
        ],
        axis=2,
    )
    fast_preview = _fast_tensor_to_rgb(result.fast.tensor, result.fast.channels)
    return PipelineImageSet(
        human_rgb=np.asarray(human, dtype=np.float64),
        perception_rgb=np.asarray(result.vision_rgb, dtype=np.float64),
        perception_aux_rgb=np.asarray(aux_rgb, dtype=np.float64),
        fast_tensor_preview=fast_preview,
        metadata=result.metadata,
    )


def compare_sample(
    sample: EvaluationSample,
    *,
    rgb_detector: DetectorAdapter | None = None,
    aux_detector: DetectorAdapter | None = None,
    config: PerceptionISPConfig | None = None,
    label_agnostic: bool = True,
    include_images: bool = False,
    include_fusion: bool = True,
) -> Dict[str, Any]:
    images = build_pipeline_images(sample, config=config)
    rgb_model = rgb_detector or NumpyRiskObjectDetector()
    aux_model = aux_detector or AuxMapRiskDetector()
    detector_results = []
    if sample.reference_rgb is not None:
        detector_results.append(rgb_model.detect(sample.reference_rgb, input_name="reference_rgb"))
    human_result = rgb_model.detect(images.human_rgb, input_name="human_rgb")
    perception_result = rgb_model.detect(images.perception_rgb, input_name="perception_rgb")
    aux_result = aux_model.detect(images.perception_aux_rgb, input_name="perception_aux_rgb")
    detector_results.extend((human_result, perception_result))
    if include_fusion:
        detector_results.append(
            fuse_rgb_aux_results(
                perception_result,
                aux_result,
                images.perception_aux_rgb,
                input_name="perception_fusion_rgb_aux",
            )
        )
    detector_results.append(aux_result)
    metric_rows = {}
    for result in detector_results:
        metric_rows[result.input_name] = evaluate_detections(
            result.detections,
            sample.ground_truth,
            label_agnostic=label_agnostic,
        )
    payload: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "ground_truth": [box.to_dict() for box in sample.ground_truth],
        "metadata": dict(sample.metadata),
        "isp_metadata": images.metadata,
        "detectors": [item.to_dict() for item in detector_results],
        "metrics": metric_rows,
        "breakdown": _sample_breakdown(detector_results, sample.ground_truth, label_agnostic=label_agnostic),
    }
    if include_images:
        visuals = {
            "human_rgb": images.human_rgb,
            "perception_rgb": images.perception_rgb,
            "perception_aux_rgb": images.perception_aux_rgb,
            "fast_tensor_preview": images.fast_tensor_preview,
        }
        if include_fusion:
            visuals["perception_fusion_rgb_aux"] = images.perception_rgb
        if sample.reference_rgb is not None:
            visuals["reference_rgb"] = sample.reference_rgb
        payload["_visuals"] = visuals
    return payload


def compare_dataset(
    samples: Sequence[EvaluationSample],
    *,
    rgb_detector: DetectorAdapter | None = None,
    aux_detector: DetectorAdapter | None = None,
    config: PerceptionISPConfig | None = None,
    label_agnostic: bool = True,
    include_images: bool = False,
    include_fusion: bool = True,
) -> Dict[str, Any]:
    sample_results = [
        compare_sample(
            sample,
            rgb_detector=rgb_detector,
            aux_detector=aux_detector,
            config=config,
            label_agnostic=label_agnostic,
            include_images=include_images,
            include_fusion=include_fusion,
        )
        for sample in samples
    ]
    aggregate: Dict[str, Any] = {}
    for input_name in _input_names_from_samples(sample_results):
        rows = [result["metrics"][input_name] for result in sample_results if input_name in result.get("metrics", {})]
        aggregate[input_name] = aggregate_metric_rows(rows)
    return {
        "sample_count": int(len(samples)),
        "aggregate": aggregate,
        "breakdown": _aggregate_breakdown(sample_results),
        "samples": sample_results,
    }


def write_comparison_report(result: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    report_model = _materialize_visual_assets(result, destination)
    json_path = destination / "comparison_summary.json"
    json_path.write_text(json.dumps(json_ready(_strip_private(report_model)), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(report_model))
    return html_path


def _fast_tensor_to_rgb(tensor: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    arr = np.asarray(tensor, dtype=np.float64)
    lookup = {name: idx for idx, name in enumerate(channels)}
    edge = arr[:, :, lookup.get("edge_strength", 0)]
    sat = arr[:, :, lookup.get("saturation", 0)]
    noise = arr[:, :, lookup.get("noise_variance", 0)]
    reliability = 1.0 - noise / max(float(np.max(noise)), 1.0e-12)
    return np.clip(np.stack([edge, sat, reliability], axis=2), 0.0, 1.0)


def _sample_breakdown(
    detector_results: Sequence[DetectorResult],
    ground_truth: Sequence[BoundingBox],
    *,
    label_agnostic: bool,
) -> Dict[str, Any]:
    labels = sorted({box.label for box in ground_truth})
    payload: Dict[str, Any] = {}
    for result in detector_results:
        label_metrics = {}
        for label in labels:
            label_gt = tuple(box for box in ground_truth if box.label == label)
            label_detections = tuple(item for item in result.detections if item.box.label == label)
            label_metrics[label] = evaluate_detections(label_detections, label_gt, label_agnostic=False)
        area_metrics = {}
        for bucket in AREA_BUCKETS:
            area_gt = tuple(box for box in ground_truth if _area_bucket(box.area) == bucket)
            area_detections = tuple(item for item in result.detections if _area_bucket(item.box.area) == bucket)
            area_metrics[bucket] = evaluate_detections(
                area_detections,
                area_gt,
                label_agnostic=label_agnostic,
            )
        payload[result.input_name] = {
            "labels": label_metrics,
            "areas": area_metrics,
        }
    return payload


def _aggregate_breakdown(sample_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for input_name in _input_names_from_samples(sample_results):
        labels = sorted(
            {
                label
                for sample in sample_results
                for label in sample.get("breakdown", {}).get(input_name, {}).get("labels", {})
            }
        )
        label_payload = {}
        for label in labels:
            rows = [
                sample.get("breakdown", {}).get(input_name, {}).get("labels", {}).get(label)
                for sample in sample_results
            ]
            label_payload[label] = aggregate_metric_rows([row for row in rows if row and int(row.get("gt_count", 0)) > 0])

        area_payload = {}
        for bucket in AREA_BUCKETS:
            rows = [
                sample.get("breakdown", {}).get(input_name, {}).get("areas", {}).get(bucket)
                for sample in sample_results
            ]
            area_payload[bucket] = aggregate_metric_rows([row for row in rows if row and int(row.get("gt_count", 0)) > 0])
        payload[input_name] = {
            "labels": label_payload,
            "areas": area_payload,
        }
    return payload


def _input_names_from_samples(sample_results: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    names = {name for sample in sample_results for name in sample.get("metrics", {})}
    ordered = [name for name in INPUT_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in INPUT_ORDER))
    return tuple(ordered)


def _area_bucket(area: float) -> str:
    if float(area) <= 32.0 * 32.0:
        return "small"
    if float(area) <= 96.0 * 96.0:
        return "medium"
    return "large"


def _materialize_visual_assets(result: Mapping[str, Any], destination: Path) -> Dict[str, Any]:
    samples_out: List[Dict[str, Any]] = []
    assets_dir = destination / "assets"
    for sample_index, sample in enumerate(result.get("samples", [])):
        sample_map = dict(sample)
        visual_sources = sample_map.pop("_visuals", None)
        if visual_sources:
            assets_dir.mkdir(parents=True, exist_ok=True)
            sample_id = _safe_filename(str(sample_map.get("sample_id", f"sample_{sample_index}")))
            visual_rows = []
            for input_name, title in (
                ("reference_rgb", "Reference RGB"),
                ("human_rgb", "HumanISP RGB"),
                ("perception_rgb", "PerceptionISP RGB"),
                ("perception_fusion_rgb_aux", "Perception RGB+Aux Fusion"),
                ("perception_aux_rgb", "Perception Aux Maps"),
            ):
                image = visual_sources.get(input_name)
                if image is None:
                    continue
                detector = _detector_for_input(sample_map, input_name)
                filename = f"assets/{sample_index:03d}_{sample_id}_{input_name}.png"
                _save_overlay_png(
                    image,
                    sample_map.get("ground_truth", ()),
                    detector.get("detections", ()),
                    destination / filename,
                )
                visual_rows.append(
                    {
                        "input_name": input_name,
                        "title": title,
                        "image": filename,
                        "detector": detector.get("detector_name", ""),
                        "detection_count": len(detector.get("detections", ())),
                    }
                )
            fast_preview = visual_sources.get("fast_tensor_preview")
            if fast_preview is not None:
                filename = f"assets/{sample_index:03d}_{sample_id}_fast_tensor_preview.png"
                _save_plain_png(fast_preview, destination / filename)
                visual_rows.append(
                    {
                        "input_name": "fast_tensor_preview",
                        "title": "Fast Tensor Preview",
                        "image": filename,
                        "detector": "",
                        "detection_count": 0,
                    }
                )
            sample_map["visuals"] = visual_rows
        samples_out.append(sample_map)
    model = {str(key): value for key, value in result.items() if str(key) != "samples"}
    model["samples"] = samples_out
    return model


def _strip_private(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strip_private(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_strip_private(item) for item in value]
    return value


def _detector_for_input(sample: Mapping[str, Any], input_name: str) -> Mapping[str, Any]:
    for detector in sample.get("detectors", ()):
        if detector.get("input_name") == input_name:
            return detector
    return {"input_name": input_name, "detector_name": "", "detections": ()}


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _save_plain_png(image: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_rgb(image)).save(path)


def _save_overlay_png(image: Any, ground_truth: Sequence[Mapping[str, Any]], detections: Sequence[Mapping[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.fromarray(_to_uint8_rgb(image)).convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    stroke = max(2, int(round(min(width, height) / 140.0)))
    for box in ground_truth:
        _draw_box(draw, box, width, height, (22, 163, 74, 255), stroke, "GT")
    for detection in detections:
        _draw_box(draw, detection, width, height, (220, 38, 38, 255), stroke, "DET")
    canvas.convert("RGB").save(path)


def _to_uint8_rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError("visual image must be HxW or HxWxC")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.nan_to_num(arr[:, :, :3], nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.max(arr)) > 1.5:
        arr = arr / 255.0
    return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_box(draw: Any, payload: Mapping[str, Any], image_width: int, image_height: int, color: Tuple[int, int, int, int], stroke: int, prefix: str) -> None:
    box_payload = payload.get("box", payload)
    coords = box_payload.get("xyxy")
    if coords is None or len(coords) != 4:
        return
    x1, y1, x2, y2 = (float(value) for value in coords)
    x1 = max(0.0, min(float(image_width - 1), x1))
    x2 = max(0.0, min(float(image_width - 1), x2))
    y1 = max(0.0, min(float(image_height - 1), y1))
    y2 = max(0.0, min(float(image_height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
    label = str(box_payload.get("label", "object"))
    if "score" in payload:
        label = f"{label} {float(payload['score']):.2f}"
    text = f"{prefix} {label}"
    text_width = max(6 * len(text), 24)
    text_height = 12
    y_text = max(0.0, y1 - text_height - 2)
    draw.rectangle((x1, y_text, min(float(image_width - 1), x1 + text_width + 5), y_text + text_height + 2), fill=color)
    draw.text((x1 + 3, y_text + 1), text, fill=(255, 255, 255, 255))


def _render_html(result: Mapping[str, Any]) -> str:
    aggregate = result.get("aggregate", {})
    rows = []
    for name in _input_names_from_aggregate(aggregate):
        metrics = aggregate.get(name, {})
        safe_name = html_lib.escape(name)
        rows.append(
            f"<tr><td>{safe_name}</td>"
            f"<td>{metrics.get('mean_recall_mean', 0.0):.3f}</td>"
            f"<td>{metrics.get('mean_precision_mean', 0.0):.3f}</td>"
            f"<td>{metrics.get('small_recall@0.50_mean', 0.0):.3f}</td>"
            f"<td>{metrics.get('det_count_mean', 0.0):.2f}</td></tr>"
        )
    sample_rows = []
    visual_sections = []
    for sample in result.get("samples", []):
        sample_id = html_lib.escape(str(sample.get("sample_id")))
        source = html_lib.escape(str(sample.get("source")))
        sample_rows.append(
            f"<tr><td>{sample_id}</td>"
            f"<td>{source}</td>"
            f"<td>{len(sample.get('ground_truth', []))}</td>"
            f"<td>{sample.get('metrics', {}).get('human_rgb', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{sample.get('metrics', {}).get('perception_rgb', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{sample.get('metrics', {}).get('perception_fusion_rgb_aux', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{sample.get('metrics', {}).get('perception_aux_rgb', {}).get('recall@0.50', 0.0):.3f}</td></tr>"
        )
        visuals = sample.get("visuals", ())
        if visuals:
            figures = []
            for item in visuals:
                title = html_lib.escape(str(item.get("title", item.get("input_name", ""))))
                image_path = html_lib.escape(str(item.get("image", "")))
                detector = html_lib.escape(str(item.get("detector", "")))
                det_count = int(item.get("detection_count", 0))
                caption = f"{title}"
                if detector:
                    caption += f" · {detector} · {det_count} det"
                figures.append(
                    f"<figure><img src=\"{image_path}\" alt=\"{title}\">"
                    f"<figcaption>{caption}</figcaption></figure>"
                )
            visual_sections.append(
                f"<section class=\"sample-visual\"><h3>{sample_id}</h3>"
                f"<div class=\"visual-grid\">{''.join(figures)}</div></section>"
            )
    class_rows = _render_class_breakdown_rows(result.get("breakdown", {}))
    area_rows = _render_area_breakdown_rows(result.get("breakdown", {}))
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>HumanISP vs PerceptionISP Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #5b6472; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 10px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    h2 {{ margin-top: 28px; }}
    h3 {{ margin: 12px 0; }}
    .note {{ border-left: 5px solid #b45309; background: #fff4df; padding: 12px 14px; margin: 16px 0; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .sample-visual {{ margin-top: 20px; }}
    .visual-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d8ded7; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px 10px; font-size: 13px; color: #394150; }}
  </style>
</head>
<body>
  <h1>HumanISP vs PerceptionISP Comparison</h1>
  <p>이 리포트는 현재 harness가 같은 RAW에서 Human RGB, Perception RGB, Perception auxiliary RGB를 만들고 detector metric을 계산한 결과입니다.</p>
  <div class=\"note\">현재 기본 detector는 순수 numpy smoke detector입니다. 성능 주장에는 Ultralytics YOLO 또는 학습된 aux-map detector 결과가 필요합니다.</div>
  <h2>Aggregate</h2>
  <table>
    <thead><tr><th>Input</th><th>Mean Recall</th><th>Mean Precision</th><th>Small Recall@0.50</th><th>Detections / sample</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Samples</h2>
  <table>
    <thead><tr><th>Sample</th><th>Source</th><th>GT</th><th>Human Recall@0.50</th><th>Perception RGB Recall@0.50</th><th>Fusion Recall@0.50</th><th>Aux Recall@0.50</th></tr></thead>
    <tbody>{''.join(sample_rows)}</tbody>
  </table>
  <h2>Class Breakdown</h2>
  <table>
    <thead><tr><th>Input</th><th>Class</th><th>GT samples</th><th>Recall@0.50</th><th>Precision@0.50</th><th>Detections / sample</th></tr></thead>
    <tbody>{class_rows if class_rows else '<tr><td colspan="6">No class breakdown available.</td></tr>'}</tbody>
  </table>
  <h2>Area Breakdown</h2>
  <table>
    <thead><tr><th>Input</th><th>Area</th><th>GT samples</th><th>Recall@0.50</th><th>Precision@0.50</th><th>Detections / sample</th></tr></thead>
    <tbody>{area_rows if area_rows else '<tr><td colspan="6">No area breakdown available.</td></tr>'}</tbody>
  </table>
  <h2>Visual Evidence</h2>
  <p>Green boxes are ground truth. Red boxes are detector outputs on each ISP image. The RGB+Aux Fusion view keeps RGB detector labels and uses aux-map support for conservative score/filtering. The auxiliary preview uses edge, saturation, and reliability channels as RGB.</p>
  {''.join(visual_sections) if visual_sections else '<p>No visual assets were captured for this run.</p>'}
  <p>Raw JSON: <code>comparison_summary.json</code></p>
</body>
</html>
"""


def _render_class_breakdown_rows(breakdown: Mapping[str, Any]) -> str:
    rows = []
    for input_name in _input_names_from_breakdown(breakdown):
        labels = breakdown.get(input_name, {}).get("labels", {})
        for label, metrics in labels.items():
            if int(metrics.get("sample_count", 0)) <= 0:
                continue
            rows.append(
                _breakdown_row(
                    input_name=input_name,
                    group=str(label),
                    metrics=metrics,
                )
            )
    return "".join(rows)


def _render_area_breakdown_rows(breakdown: Mapping[str, Any]) -> str:
    rows = []
    for input_name in _input_names_from_breakdown(breakdown):
        areas = breakdown.get(input_name, {}).get("areas", {})
        for area in AREA_BUCKETS:
            metrics = areas.get(area, {})
            if int(metrics.get("sample_count", 0)) <= 0:
                continue
            rows.append(_breakdown_row(input_name=input_name, group=area, metrics=metrics))
    return "".join(rows)


def _breakdown_row(*, input_name: str, group: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"<tr><td>{html_lib.escape(str(input_name))}</td>"
        f"<td>{html_lib.escape(str(group))}</td>"
        f"<td>{metrics.get('sample_count', 0)}</td>"
        f"<td>{metrics.get('recall@0.50_mean', 0.0):.3f}</td>"
        f"<td>{metrics.get('precision@0.50_mean', 0.0):.3f}</td>"
        f"<td>{metrics.get('det_count_mean', 0.0):.2f}</td></tr>"
    )


def _input_names_from_aggregate(aggregate: Mapping[str, Any]) -> Tuple[str, ...]:
    names = set(str(name) for name in aggregate)
    ordered = [name for name in INPUT_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in INPUT_ORDER))
    return tuple(ordered)


def _input_names_from_breakdown(breakdown: Mapping[str, Any]) -> Tuple[str, ...]:
    names = set(str(name) for name in breakdown)
    ordered = [name for name in INPUT_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in INPUT_ORDER))
    return tuple(ordered)
