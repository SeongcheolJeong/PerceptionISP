"""Task-oriented metrics over saved comparison reports."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .comparison import INPUT_ORDER
from .eval_types import BoundingBox, Detection
from .metrics import evaluate_detections
from .types import json_ready


DEFAULT_LABEL_GROUPS = {
    "vru": ("person", "pedestrian", "bicycle", "cyclist"),
    "person": ("person", "pedestrian"),
    "cyclist": ("bicycle", "cyclist"),
    "vehicle": ("car", "van", "truck", "bus", "train"),
    "traffic_light": ("traffic_light", "traffic light"),
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compute task-oriented group metrics from a saved comparison report.")
    parser.add_argument("report", help="Comparison report directory or comparison_summary.json")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--inputs", default=None, help="Comma-separated input names. Default uses all report inputs.")
    parser.add_argument("--group", action="append", default=[], help="Custom label group as name=label1,label2. Repeats are allowed.")
    parser.add_argument("--no-default-groups", action="store_true")
    parser.add_argument("--small-area-threshold", type=float, default=32.0 * 32.0)
    parser.add_argument("--output-dir", default="reports/perception_task_metrics")
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    summary = build_task_metrics(
        report,
        source_report=report_path,
        baseline_input=str(args.baseline_input),
        inputs=parse_list(args.inputs),
        label_groups=parse_label_groups(args.group, include_defaults=not bool(args.no_default_groups)),
        small_area_threshold=float(args.small_area_threshold),
    )
    html_path = write_task_metrics(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "task_metrics_summary.json"),
                    "input_count": len(summary["inputs"]),
                    "group_count": len(summary["groups"]),
                }
            ),
            indent=2,
        )
    )
    return 0


def build_task_metrics(
    report: Mapping[str, Any],
    *,
    source_report: str | Path | None = None,
    baseline_input: str = "human_rgb",
    inputs: Sequence[str] | None = None,
    label_groups: Mapping[str, Sequence[str]] | None = None,
    small_area_threshold: float = 32.0 * 32.0,
) -> Dict[str, Any]:
    selected_inputs = tuple(inputs) if inputs else _input_names(report)
    resolved_label_groups = DEFAULT_LABEL_GROUPS if label_groups is None else label_groups
    groups = _group_specs(resolved_label_groups, small_area_threshold=float(small_area_threshold))
    label_agnostic = bool(report.get("run_config", {}).get("label_agnostic", True))
    rows: Dict[str, Dict[str, Any]] = {}
    for input_name in selected_inputs:
        rows[input_name] = {}
        for group in groups:
            rows[input_name][group["name"]] = _evaluate_group(report, input_name=input_name, group=group, label_agnostic=label_agnostic)
    _attach_deltas(rows, baseline_input=str(baseline_input))
    return {
        "source_report": "" if source_report is None else str(source_report),
        "baseline_input": str(baseline_input),
        "inputs": list(selected_inputs),
        "groups": [{key: value for key, value in group.items() if key != "predicate"} for group in groups],
        "small_area_threshold": float(small_area_threshold),
        "label_agnostic": bool(label_agnostic),
        "metrics": rows,
        "interpretation": "Task metrics are computed from saved detections and ground truth; they do not rerun CameraE2E, ISP, or the detector.",
    }


def write_task_metrics(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "task_metrics_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def parse_list(value: str | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return items or None


def parse_label_groups(values: Sequence[str], *, include_defaults: bool = True) -> Dict[str, Tuple[str, ...]]:
    groups: Dict[str, Tuple[str, ...]] = dict(DEFAULT_LABEL_GROUPS) if include_defaults else {}
    for value in values:
        if "=" not in str(value):
            raise ValueError("group must be formatted as name=label1,label2")
        name, raw_labels = str(value).split("=", 1)
        labels = tuple(token.strip() for token in raw_labels.split(",") if token.strip())
        if not name.strip() or not labels:
            raise ValueError("group must contain a name and at least one label")
        groups[name.strip()] = labels
    return groups


def _group_specs(label_groups: Mapping[str, Sequence[str]], *, small_area_threshold: float) -> Tuple[Dict[str, Any], ...]:
    specs = []
    for name, labels in label_groups.items():
        label_set = frozenset(str(label) for label in labels)
        specs.append(
            {
                "name": str(name),
                "kind": "label",
                "labels": tuple(sorted(label_set)),
                "area_threshold": None,
                "predicate": lambda box, label_set=label_set: str(box.label) in label_set,
            }
        )
    specs.append(
        {
            "name": "small_all",
            "kind": "area",
            "labels": (),
            "area_threshold": float(small_area_threshold),
            "predicate": lambda box, threshold=float(small_area_threshold): float(box.area) <= threshold,
        }
    )
    return tuple(specs)


def _evaluate_group(
    report: Mapping[str, Any],
    *,
    input_name: str,
    group: Mapping[str, Any],
    label_agnostic: bool,
) -> Dict[str, Any]:
    sample_count = 0
    positive_sample_count = 0
    gt_total = 0
    det_total = 0
    tp50 = fp50 = fn50 = 0
    tp75 = fp75 = fn75 = 0
    for sample in report.get("samples", ()):
        detector = _detector_for_input(sample, input_name)
        if detector is None:
            continue
        sample_count += 1
        predicate = group["predicate"]
        gt = tuple(box for box in _boxes_from_payload(sample.get("ground_truth", ())) if predicate(box))
        detections = tuple(item for item in _detections_from_payload(detector.get("detections", ())) if predicate(item.box))
        if gt:
            positive_sample_count += 1
        metrics = evaluate_detections(detections, gt, label_agnostic=label_agnostic)
        gt_total += int(metrics.get("gt_count", 0))
        det_total += int(metrics.get("det_count", 0))
        tp50 += int(metrics.get("tp@0.50", 0))
        fp50 += int(metrics.get("fp@0.50", 0))
        fn50 += int(metrics.get("fn@0.50", 0))
        tp75 += int(metrics.get("tp@0.75", 0))
        fp75 += int(metrics.get("fp@0.75", 0))
        fn75 += int(metrics.get("fn@0.75", 0))
    return {
        "group": str(group["name"]),
        "kind": str(group["kind"]),
        "labels": list(group.get("labels", ())),
        "area_threshold": group.get("area_threshold"),
        "sample_count": int(sample_count),
        "positive_sample_count": int(positive_sample_count),
        "gt_count": int(gt_total),
        "det_count": int(det_total),
        "tp@0.50": int(tp50),
        "fp@0.50": int(fp50),
        "fn@0.50": int(fn50),
        "precision@0.50": _precision(tp50, fp50),
        "recall@0.50": _recall(tp50, gt_total),
        "fp@0.50_per_sample": float(fp50 / max(sample_count, 1)),
        "tp@0.75": int(tp75),
        "fp@0.75": int(fp75),
        "fn@0.75": int(fn75),
        "precision@0.75": _precision(tp75, fp75),
        "recall@0.75": _recall(tp75, gt_total),
        "fp@0.75_per_sample": float(fp75 / max(sample_count, 1)),
    }


def _attach_deltas(rows: Dict[str, Dict[str, Any]], *, baseline_input: str) -> None:
    baseline = rows.get(str(baseline_input), {})
    for input_name, groups in rows.items():
        if input_name == str(baseline_input):
            continue
        for group_name, metrics in groups.items():
            base = baseline.get(group_name)
            if not isinstance(base, Mapping):
                continue
            for key in ("precision@0.50", "recall@0.50", "recall@0.75", "fp@0.50_per_sample", "det_count"):
                metrics[f"delta_{key}"] = float(metrics.get(key, 0.0)) - float(base.get(key, 0.0))


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
        box = boxes[0]
        detections.append(Detection(box, score=float(item.get("score", 1.0)), metadata=item.get("metadata", {})))
    return tuple(detections)


def _precision(tp: int, fp: int) -> float:
    return float(tp / max(int(tp) + int(fp), 1))


def _recall(tp: int, gt_count: int) -> float:
    return float(tp / max(int(gt_count), 1))


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    rows = []
    baseline_input = str(summary.get("baseline_input", "human_rgb"))
    for input_name in summary.get("inputs", ()):
        group_metrics = summary.get("metrics", {}).get(input_name, {})
        for group in summary.get("groups", ()):
            group_name = str(group.get("name", ""))
            metrics = group_metrics.get(group_name, {})
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(input_name))}</td>"
                f"<td>{html_lib.escape(group_name)}</td>"
                f"<td>{html_lib.escape(str(group.get('kind', '')))}</td>"
                f"<td>{int(metrics.get('gt_count', 0))}</td>"
                f"<td>{int(metrics.get('det_count', 0))}</td>"
                f"<td>{_fmt(metrics.get('precision@0.50'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.50'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.75'))}</td>"
                f"<td>{_fmt(metrics.get('fp@0.50_per_sample'))}</td>"
                f"<td>{_fmt(metrics.get('delta_precision@0.50'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_recall@0.50'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_recall@0.75'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_fp@0.50_per_sample'), signed=True)}</td>"
                "</tr>"
            )
    source_link = _source_link(summary, destination)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Task Metrics</title>
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
  <h1>PerceptionISP Task Metrics</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <p>Baseline input: <code>{html_lib.escape(baseline_input)}</code>. Source: {source_link}</p>
  <table>
    <thead><tr><th>Input</th><th>Group</th><th>Kind</th><th>GT</th><th>Det</th><th>P@0.50</th><th>R@0.50</th><th>R@0.75</th><th>FP/sample</th><th>dP@0.50</th><th>dR@0.50</th><th>dR@0.75</th><th>dFP/sample</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>task_metrics_summary.json</code></p>
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
