"""Post-hoc score threshold sweep for saved comparison reports."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .eval_types import BoundingBox, Detection
from .metrics import aggregate_metric_rows, evaluate_detections
from .types import json_ready


DEFAULT_INPUTS = ("perception_rgb", "perception_fusion_rgb_aux")
TRACKED_METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "det_count_mean",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep score thresholds on a saved comparison report.")
    parser.add_argument("report", help="Report directory or comparison_summary.json")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS))
    parser.add_argument("--thresholds", default="0.25:0.60:0.025", help="Comma values or start:stop:step range.")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--recall-delta-floor", type=float, default=-0.001)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    inputs = parse_csv(args.inputs)
    thresholds = parse_thresholds(args.thresholds)
    summary = build_threshold_sweep(
        report,
        inputs=inputs,
        thresholds=thresholds,
        baseline_input=str(args.baseline_input),
        recall_delta_floor=float(args.recall_delta_floor),
        source_report=report_path,
    )
    destination = Path(args.output_dir).expanduser() if args.output_dir else report_path.parent / "threshold_sweep"
    html_path = write_threshold_sweep(summary, destination)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "threshold_sweep_summary.json"),
                    "best": summary.get("best", {}),
                }
            ),
            indent=2,
        )
    )
    return 0


def parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not items:
        raise ValueError("at least one input is required")
    return items


def parse_thresholds(value: str) -> Tuple[float, ...]:
    text = str(value).strip()
    if ":" in text:
        start_text, stop_text, step_text = text.split(":", 2)
        start, stop, step = float(start_text), float(stop_text), float(step_text)
        if step <= 0.0:
            raise ValueError("threshold step must be positive")
        thresholds = []
        current = start
        while current <= stop + 1.0e-12:
            thresholds.append(round(float(current), 6))
            current += step
        return tuple(thresholds)
    return tuple(float(token) for token in parse_csv(text))


def build_threshold_sweep(
    report: Mapping[str, Any],
    *,
    inputs: Sequence[str],
    thresholds: Sequence[float],
    baseline_input: str = "human_rgb",
    recall_delta_floor: float = -0.001,
    source_report: str | Path | None = None,
) -> Dict[str, Any]:
    label_agnostic = bool(report.get("run_config", {}).get("label_agnostic", True))
    baseline = report.get("aggregate", {}).get(baseline_input, {})
    rows: List[Dict[str, Any]] = []
    for input_name in inputs:
        for threshold in thresholds:
            metrics = _aggregate_filtered(report, input_name=input_name, threshold=float(threshold), label_agnostic=label_agnostic)
            rows.append(
                {
                    "input": str(input_name),
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "delta_vs_baseline": _deltas(metrics, baseline),
                }
            )
    return {
        "source_report": "" if source_report is None else str(source_report),
        "sample_count": int(report.get("sample_count", 0)),
        "baseline_input": str(baseline_input),
        "baseline_metrics": dict(baseline),
        "inputs": list(inputs),
        "thresholds": [float(value) for value in thresholds],
        "recall_delta_floor": float(recall_delta_floor),
        "best": {
            "max_recall_delta": _best_by(rows, key="recall@0.50_mean"),
            "max_precision_with_recall_floor": _best_precision_with_floor(rows, recall_delta_floor=float(recall_delta_floor)),
            "min_fp_with_recall_floor": _best_min_fp_with_floor(rows, recall_delta_floor=float(recall_delta_floor)),
        },
        "rows": rows,
    }


def write_threshold_sweep(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "threshold_sweep_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _summary_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "comparison_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"comparison summary not found: {path}")
    return path


def _aggregate_filtered(report: Mapping[str, Any], *, input_name: str, threshold: float, label_agnostic: bool) -> Dict[str, Any]:
    metric_rows = []
    for sample in report.get("samples", ()):
        gt = tuple(_box_from_dict(item) for item in sample.get("ground_truth", ()))
        detections = tuple(
            detection
            for detection in _detections_for_sample(sample, input_name)
            if float(detection.score) >= float(threshold)
        )
        metric_rows.append(evaluate_detections(detections, gt, label_agnostic=label_agnostic))
    return aggregate_metric_rows(metric_rows)


def _detections_for_sample(sample: Mapping[str, Any], input_name: str) -> Tuple[Detection, ...]:
    for detector in sample.get("detectors", ()):
        if str(detector.get("input_name")) != str(input_name):
            continue
        return tuple(_detection_from_dict(item) for item in detector.get("detections", ()))
    return ()


def _box_from_dict(payload: Mapping[str, Any]) -> BoundingBox:
    return BoundingBox(tuple(payload.get("xyxy", (0.0, 0.0, 0.0, 0.0))), label=str(payload.get("label", "object")))


def _detection_from_dict(payload: Mapping[str, Any]) -> Detection:
    return Detection(
        box=_box_from_dict(payload.get("box", {})),
        score=float(payload.get("score", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _deltas(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in TRACKED_METRICS
    }


def _best_by(rows: Sequence[Mapping[str, Any]], *, key: str) -> Dict[str, Any]:
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("delta_vs_baseline", {}).get(key, -1.0e9)))
    return _compact_best(best)


def _best_precision_with_floor(rows: Sequence[Mapping[str, Any]], *, recall_delta_floor: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if float(row.get("delta_vs_baseline", {}).get("recall@0.50_mean", -1.0e9)) >= float(recall_delta_floor)
    ]
    if not candidates:
        return {}
    best = max(candidates, key=lambda row: float(row.get("metrics", {}).get("precision@0.50_mean", -1.0e9)))
    return _compact_best(best)


def _best_min_fp_with_floor(rows: Sequence[Mapping[str, Any]], *, recall_delta_floor: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if float(row.get("delta_vs_baseline", {}).get("recall@0.50_mean", -1.0e9)) >= float(recall_delta_floor)
    ]
    if not candidates:
        return {}
    best = min(candidates, key=lambda row: float(row.get("metrics", {}).get("fp@0.50_mean", 1.0e9)))
    return _compact_best(best)


def _compact_best(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "input": row.get("input"),
        "threshold": float(row.get("threshold", 0.0)),
        "metrics": dict(row.get("metrics", {})),
        "delta_vs_baseline": dict(row.get("delta_vs_baseline", {})),
    }


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for row in summary.get("rows", ()):
        metrics = row.get("metrics", {})
        delta = row.get("delta_vs_baseline", {})
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('input', '')))}</td>"
            f"<td>{float(row.get('threshold', 0.0)):.3f}</td>"
            f"<td>{_fmt(metrics.get('precision@0.50_mean'))}</td>"
            f"<td>{_fmt(metrics.get('recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('recall@0.50_mean'))}\">{_fmt_delta(delta.get('recall@0.50_mean'))}</td>"
            f"<td>{_fmt(metrics.get('recall@0.75_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('recall@0.75_mean'))}\">{_fmt_delta(delta.get('recall@0.75_mean'))}</td>"
            f"<td>{_fmt(metrics.get('small_recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('small_recall@0.50_mean'))}\">{_fmt_delta(delta.get('small_recall@0.50_mean'))}</td>"
            f"<td>{_fmt(metrics.get('fp@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(-float(delta.get('fp@0.50_mean', 0.0)))}\">{_fmt_delta(delta.get('fp@0.50_mean'))}</td>"
            "</tr>"
        )
    best_items = []
    for name, item in summary.get("best", {}).items():
        if not item:
            continue
        delta = item.get("delta_vs_baseline", {})
        best_items.append(
            f"<li><strong>{html_lib.escape(str(name))}</strong>: "
            f"{html_lib.escape(str(item.get('input')))} @ {float(item.get('threshold', 0.0)):.3f}, "
            f"delta R50={float(delta.get('recall@0.50_mean', 0.0)):+.4f}, "
            f"delta P50={float(delta.get('precision@0.50_mean', 0.0)):+.4f}, "
            f"delta FP50={float(delta.get('fp@0.50_mean', 0.0)):+.4f}</li>"
        )
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Threshold Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 9px; text-align: left; font-size: 14px; }}
    th {{ background: #e8f3f1; position: sticky; top: 0; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pos {{ color: #047857; font-weight: 650; }}
    .neg {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Threshold Sweep</h1>
  <div class=\"note\">This report re-filters saved detections by score threshold. It cannot recover detections below the detector confidence used in the source run.</div>
  <p><strong>Source:</strong> <code>{html_lib.escape(str(summary.get('source_report', '')))}</code></p>
  <p><strong>Baseline:</strong> <code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code>, samples={int(summary.get('sample_count', 0))}, recall floor={float(summary.get('recall_delta_floor', 0.0)):+.4f}</p>
  <h2>Best</h2>
  <ul>{''.join(best_items)}</ul>
  <table>
    <thead><tr><th>Input</th><th>Threshold</th><th>P50</th><th>R50</th><th>Delta R50</th><th>R75</th><th>Delta R75</th><th>Small R50</th><th>Delta Small</th><th>FP50</th><th>Delta FP50</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>threshold_sweep_summary.json</code></p>
</body>
</html>
"""


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


def _delta_class(value: Any) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if numeric > 0.0:
        return "pos"
    if numeric < 0.0:
        return "neg"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
