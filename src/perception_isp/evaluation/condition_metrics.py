"""Condition-specific metrics over saved comparison reports."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from perception_isp.evaluation.comparison import INPUT_ORDER
from perception_isp.core.task_types import BoundingBox, Detection
from perception_isp.evaluation.metrics import aggregate_metric_rows, evaluate_detections
from perception_isp.core.types import json_ready


DEFAULT_CONDITION_KEYS = (
    "condition",
    "conditions",
    "condition_tags",
    "tags",
    "weather",
    "lighting",
    "time_of_day",
    "scene_condition",
    "scenario",
    "scenario_tags",
    "adverse_condition",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compute condition-specific metrics from a saved comparison report.")
    parser.add_argument("report", help="Comparison report directory or comparison_summary.json")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--inputs", default=None, help="Comma-separated input names. Default uses all report inputs.")
    parser.add_argument("--condition-key", action="append", default=[], help="Metadata key to treat as an explicit condition tag. Repeats are allowed.")
    parser.add_argument("--no-derived", action="store_true", help="Disable derived sensor/health condition tags.")
    parser.add_argument("--low-visibility-threshold", type=float, default=0.50)
    parser.add_argument("--under-exposure-threshold", type=float, default=0.25)
    parser.add_argument("--over-exposure-threshold", type=float, default=0.01)
    parser.add_argument("--low-focus-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="reports/perception_condition_metrics")
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    summary = build_condition_metrics(
        report,
        source_report=report_path,
        baseline_input=str(args.baseline_input),
        inputs=parse_list(args.inputs),
        condition_keys=tuple(args.condition_key) or DEFAULT_CONDITION_KEYS,
        include_derived=not bool(args.no_derived),
        low_visibility_threshold=float(args.low_visibility_threshold),
        under_exposure_threshold=float(args.under_exposure_threshold),
        over_exposure_threshold=float(args.over_exposure_threshold),
        low_focus_threshold=float(args.low_focus_threshold),
    )
    html_path = write_condition_metrics(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "condition_metrics_summary.json"),
                    "input_count": len(summary["inputs"]),
                    "condition_count": len(summary["conditions"]),
                }
            ),
            indent=2,
        )
    )
    return 0


def parse_list(value: str | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return items or None


def build_condition_metrics(
    report: Mapping[str, Any],
    *,
    source_report: str | Path | None = None,
    baseline_input: str = "human_rgb",
    inputs: Sequence[str] | None = None,
    condition_keys: Sequence[str] = DEFAULT_CONDITION_KEYS,
    include_derived: bool = True,
    low_visibility_threshold: float = 0.50,
    under_exposure_threshold: float = 0.25,
    over_exposure_threshold: float = 0.01,
    low_focus_threshold: float = 0.50,
) -> Dict[str, Any]:
    selected_inputs = tuple(inputs) if inputs else _input_names(report)
    label_agnostic = bool(report.get("run_config", {}).get("label_agnostic", True))
    samples = tuple(sample for sample in report.get("samples", ()) if isinstance(sample, Mapping))
    sample_tags = [
        _condition_tags(
            sample,
            condition_keys=condition_keys,
            include_derived=include_derived,
            low_visibility_threshold=float(low_visibility_threshold),
            under_exposure_threshold=float(under_exposure_threshold),
            over_exposure_threshold=float(over_exposure_threshold),
            low_focus_threshold=float(low_focus_threshold),
        )
        for sample in samples
    ]
    condition_names = _ordered_conditions(sample_tags)
    metrics: Dict[str, Dict[str, Any]] = {}
    for input_name in selected_inputs:
        metrics[input_name] = {}
        for condition in condition_names:
            indices = tuple(index for index, tags in enumerate(sample_tags) if condition in tags)
            metrics[input_name][condition] = _evaluate_condition(
                samples,
                input_name=input_name,
                indices=indices,
                condition=condition,
                label_agnostic=label_agnostic,
            )
    _attach_deltas(metrics, baseline_input=str(baseline_input))
    return {
        "source_report": "" if source_report is None else str(source_report),
        "baseline_input": str(baseline_input),
        "inputs": list(selected_inputs),
        "conditions": [
            {
                "name": condition,
                "sample_count": int(sum(1 for tags in sample_tags if condition in tags)),
                "derived": _is_derived_condition(condition),
            }
            for condition in condition_names
        ],
        "condition_keys": list(condition_keys),
        "include_derived": bool(include_derived),
        "thresholds": {
            "low_visibility": float(low_visibility_threshold),
            "under_exposure": float(under_exposure_threshold),
            "over_exposure": float(over_exposure_threshold),
            "low_focus": float(low_focus_threshold),
        },
        "sample_count": int(len(samples)),
        "label_agnostic": bool(label_agnostic),
        "metrics": metrics,
        "interpretation": (
            "Condition metrics are computed from saved detections and sample metadata. "
            "They expose adverse-condition slices before making broad HumanISP-vs-PerceptionISP claims."
        ),
    }


def write_condition_metrics(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "condition_metrics_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _evaluate_condition(
    samples: Sequence[Mapping[str, Any]],
    *,
    input_name: str,
    indices: Sequence[int],
    condition: str,
    label_agnostic: bool,
) -> Dict[str, Any]:
    rows = []
    positive_sample_count = 0
    for index in indices:
        sample = samples[int(index)]
        detector = _detector_for_input(sample, input_name)
        if detector is None:
            continue
        gt = _boxes_from_payload(sample.get("ground_truth", ()))
        detections = _detections_from_payload(detector.get("detections", ()))
        if gt:
            positive_sample_count += 1
        rows.append(evaluate_detections(detections, gt, label_agnostic=label_agnostic))
    aggregate = aggregate_metric_rows(rows)
    aggregate["condition"] = str(condition)
    aggregate["condition_sample_count"] = int(len(indices))
    aggregate["positive_sample_count"] = int(positive_sample_count)
    aggregate["gt_count_total"] = int(sum(int(row.get("gt_count", 0)) for row in rows))
    aggregate["det_count_total"] = int(sum(int(row.get("det_count", 0)) for row in rows))
    return aggregate


def _attach_deltas(metrics: Dict[str, Dict[str, Any]], *, baseline_input: str) -> None:
    baseline = metrics.get(str(baseline_input), {})
    for input_name, conditions in metrics.items():
        if input_name == str(baseline_input):
            continue
        for condition, row in conditions.items():
            base = baseline.get(condition)
            if not isinstance(base, Mapping):
                continue
            for key in (
                "precision@0.50_mean",
                "recall@0.50_mean",
                "recall@0.75_mean",
                "small_recall@0.50_mean",
                "fp@0.50_mean",
                "det_count_mean",
            ):
                row[f"delta_{key}"] = float(row.get(key, 0.0)) - float(base.get(key, 0.0))


def _condition_tags(
    sample: Mapping[str, Any],
    *,
    condition_keys: Sequence[str],
    include_derived: bool,
    low_visibility_threshold: float,
    under_exposure_threshold: float,
    over_exposure_threshold: float,
    low_focus_threshold: float,
) -> frozenset[str]:
    tags: set[str] = {"all"}
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), Mapping) else {}
    for key in condition_keys:
        for source in (sample, metadata):
            if isinstance(source, Mapping) and key in source:
                tags.update(_tags_from_value(key, source.get(key)))
    if include_derived:
        tags.update(
            _derived_tags(
                sample,
                low_visibility_threshold=float(low_visibility_threshold),
                under_exposure_threshold=float(under_exposure_threshold),
                over_exposure_threshold=float(over_exposure_threshold),
                low_focus_threshold=float(low_focus_threshold),
            )
        )
    return frozenset(tag for tag in tags if tag)


def _tags_from_value(key: str, value: Any) -> Tuple[str, ...]:
    normalized_key = _normalize_tag(key)
    if value is None or value is False:
        return ()
    if isinstance(value, Mapping):
        tags = []
        for sub_key, sub_value in value.items():
            if sub_value is False or sub_value is None:
                continue
            tag = _normalize_tag(sub_key) if sub_value is True else f"{_normalize_tag(sub_key)}:{_normalize_tag(sub_value)}"
            tags.append(tag)
        return tuple(tags)
    if isinstance(value, (list, tuple, set)):
        return tuple(tag for item in value for tag in _tags_from_value(key, item))
    if isinstance(value, str) and normalized_key in {"tags", "conditions", "condition_tags", "scenario_tags"}:
        return tuple(_normalize_tag(token) for token in value.replace(";", ",").replace("|", ",").split(",") if token.strip())
    if normalized_key in {"tags", "conditions", "condition_tags", "scenario_tags"}:
        return (_normalize_tag(value),)
    return (f"{normalized_key}:{_normalize_tag(value)}",)


def _derived_tags(
    sample: Mapping[str, Any],
    *,
    low_visibility_threshold: float,
    under_exposure_threshold: float,
    over_exposure_threshold: float,
    low_focus_threshold: float,
) -> Tuple[str, ...]:
    tags = []
    isp_metadata = sample.get("isp_metadata", {}) if isinstance(sample.get("isp_metadata"), Mapping) else {}
    health = isp_metadata.get("health", {}) if isinstance(isp_metadata.get("health"), Mapping) else {}
    raw_provenance = _raw_provenance(sample, isp_metadata)
    if bool(raw_provenance.get("true_sensor_cfa_mosaic")):
        tags.append("true_cfa_mosaic")
    if bool(raw_provenance.get("camerae2e_used")):
        tags.append("camerae2e_raw")
    if _number_le(health.get("visibility_confidence"), low_visibility_threshold):
        tags.append("low_visibility_proxy")
    if _number_ge(health.get("under_exposure_fraction"), under_exposure_threshold):
        tags.append("low_light_proxy")
    if _number_ge(health.get("over_exposure_fraction"), over_exposure_threshold):
        tags.append("glare_or_over_exposure_proxy")
    if _number_le(health.get("focus_confidence"), low_focus_threshold):
        tags.append("low_focus_proxy")
    warnings = health.get("warnings", ())
    if isinstance(warnings, (list, tuple, set)):
        tags.extend(f"warning:{_normalize_tag(value)}" for value in warnings)
    return tuple(tags)


def _raw_provenance(sample: Mapping[str, Any], isp_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), Mapping) else {}
    if isinstance(metadata.get("raw_provenance"), Mapping):
        return metadata["raw_provenance"]
    if isinstance(isp_metadata.get("raw_provenance"), Mapping):
        return isp_metadata["raw_provenance"]
    return {}


def _ordered_conditions(sample_tags: Sequence[frozenset[str]]) -> Tuple[str, ...]:
    names = sorted({tag for tags in sample_tags for tag in tags})
    if "all" in names:
        names.remove("all")
        return ("all", *names)
    return tuple(names)


def _is_derived_condition(condition: str) -> bool:
    return str(condition) in {
        "camerae2e_raw",
        "true_cfa_mosaic",
        "low_visibility_proxy",
        "low_light_proxy",
        "glare_or_over_exposure_proxy",
        "low_focus_proxy",
    } or str(condition).startswith("warning:")


def _normalize_tag(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", ":"}).strip("_")


def _number_ge(value: Any, threshold: float) -> bool:
    try:
        return float(value) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _number_le(value: Any, threshold: float) -> bool:
    try:
        return float(value) <= float(threshold)
    except (TypeError, ValueError):
        return False


def _summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "comparison_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"comparison summary not found: {candidate}")
    return candidate


def _input_names(report: Mapping[str, Any]) -> Tuple[str, ...]:
    names = set(str(name) for name in report.get("aggregate", {}))
    if not names:
        names = {str(detector.get("input_name")) for sample in report.get("samples", ()) for detector in sample.get("detectors", ())}
    ordered = [name for name in INPUT_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in INPUT_ORDER))
    return tuple(ordered)


def _detector_for_input(sample: Mapping[str, Any], input_name: str) -> Mapping[str, Any] | None:
    for detector in sample.get("detectors", ()):
        if str(detector.get("input_name")) == str(input_name):
            return detector
    return None


def _boxes_from_payload(items: Sequence[Any]) -> Tuple[BoundingBox, ...]:
    boxes = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        coords = item.get("xyxy")
        if coords is None and isinstance(item.get("box"), Mapping):
            coords = item.get("box", {}).get("xyxy")
        if coords is None:
            continue
        box_payload = item.get("box", item)
        boxes.append(BoundingBox(tuple(float(value) for value in coords), label=str(box_payload.get("label", "object"))))
    return tuple(boxes)


def _detections_from_payload(items: Sequence[Any]) -> Tuple[Detection, ...]:
    detections = []
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("box"), Mapping):
            continue
        boxes = _boxes_from_payload((item,))
        if not boxes:
            continue
        detections.append(Detection(boxes[0], score=float(item.get("score", 1.0)), metadata=item.get("metadata", {})))
    return tuple(detections)


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    rows = []
    for input_name in summary.get("inputs", ()):
        condition_metrics = summary.get("metrics", {}).get(input_name, {})
        for condition in summary.get("conditions", ()):
            condition_name = str(condition.get("name", ""))
            metrics = condition_metrics.get(condition_name, {})
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(input_name))}</td>"
                f"<td>{html_lib.escape(condition_name)}</td>"
                f"<td>{int(condition.get('sample_count', 0))}</td>"
                f"<td>{int(metrics.get('gt_count_total', 0))}</td>"
                f"<td>{int(metrics.get('det_count_total', 0))}</td>"
                f"<td>{_fmt(metrics.get('precision@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.75_mean'))}</td>"
                f"<td>{_fmt(metrics.get('small_recall@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('fp@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('delta_precision@0.50_mean'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_recall@0.50_mean'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_recall@0.75_mean'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_fp@0.50_mean'), signed=True)}</td>"
                "</tr>"
            )
    source_link = _source_link(summary, destination)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Condition Metrics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Condition Metrics</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <p>Baseline input: <code>{html_lib.escape(str(summary.get('baseline_input', 'human_rgb')))}</code>. Source: {source_link}</p>
  <table>
    <thead><tr><th>Input</th><th>Condition</th><th>Samples</th><th>GT</th><th>Det</th><th>P@0.50</th><th>R@0.50</th><th>R@0.75</th><th>Small R@0.50</th><th>FP/sample</th><th>dP@0.50</th><th>dR@0.50</th><th>dR@0.75</th><th>dFP/sample</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>condition_metrics_summary.json</code></p>
</body>
</html>
"""


def _source_link(summary: Mapping[str, Any], destination: Path) -> str:
    source = str(summary.get("source_report", ""))
    if not source:
        return ""
    source_path = Path(source)
    html_path = source_path.with_name("index.html")
    if html_path.exists():
        relative = os.path.relpath(str(html_path), start=str(destination))
        return f"<a href=\"{html_lib.escape(relative)}\">open</a>"
    return f"<code>{html_lib.escape(source)}</code>"


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
