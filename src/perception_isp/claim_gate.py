"""Metric gate for HumanISP-vs-PerceptionISP claim readiness."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .types import json_ready


CRITERIA = (
    ("precision@0.50_mean", "minimum_delta", "min_precision_delta", 0.0),
    ("recall@0.50_mean", "minimum_delta", "min_recall_delta", 0.0),
    ("recall@0.75_mean", "minimum_delta", "min_recall75_delta", 0.0),
    ("small_recall@0.50_mean", "minimum_delta", "min_small_recall_delta", 0.0),
    ("fp@0.50_mean", "maximum_delta", "max_fp_delta", 0.0),
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether a target input passes a conservative comparison claim gate.")
    parser.add_argument("report", help="Report directory or comparison_summary.json")
    parser.add_argument("--target-input", default="perception_calibrated_fusion_rgb_aux")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--min-precision-delta", type=float, default=0.0)
    parser.add_argument("--min-recall-delta", type=float, default=0.0)
    parser.add_argument("--min-recall75-delta", type=float, default=0.0)
    parser.add_argument("--min-small-recall-delta", type=float, default=0.0)
    parser.add_argument("--max-fp-delta", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    summary = build_claim_gate(
        report,
        target_input=str(args.target_input),
        baseline_input=str(args.baseline_input),
        thresholds={
            "min_precision_delta": float(args.min_precision_delta),
            "min_recall_delta": float(args.min_recall_delta),
            "min_recall75_delta": float(args.min_recall75_delta),
            "min_small_recall_delta": float(args.min_small_recall_delta),
            "max_fp_delta": float(args.max_fp_delta),
            "min_samples": int(args.min_samples),
        },
        source_report=report_path,
    )
    destination = Path(args.output_dir).expanduser() if args.output_dir else report_path.parent / f"claim_gate_{_safe_name(args.target_input)}_vs_{_safe_name(args.baseline_input)}"
    html_path = write_claim_gate(summary, destination)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "claim_gate_summary.json"),
                    "pass": summary["pass"],
                    "verdict": summary["verdict"],
                    "failed": [item["metric"] for item in summary["criteria"] if not item["pass"]],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_claim_gate(
    report: Mapping[str, Any],
    *,
    target_input: str,
    baseline_input: str = "human_rgb",
    thresholds: Mapping[str, Any] | None = None,
    source_report: str | Path | None = None,
) -> Dict[str, Any]:
    threshold_values = dict(thresholds or {})
    aggregate = report.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        raise ValueError("comparison report aggregate is missing or invalid")
    target = aggregate.get(target_input)
    baseline = aggregate.get(baseline_input)
    if not isinstance(target, Mapping):
        raise ValueError(f"target input not found in aggregate: {target_input}")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"baseline input not found in aggregate: {baseline_input}")

    criteria = []
    for metric, direction, threshold_key, default_value in CRITERIA:
        target_value = _metric_value(target, metric)
        baseline_value = _metric_value(baseline, metric)
        available = target_value is not None and baseline_value is not None
        delta = None if not available else target_value - baseline_value
        threshold = float(threshold_values.get(threshold_key, default_value))
        passed = bool(available and (delta >= threshold if direction == "minimum_delta" else delta <= threshold))
        criteria.append(
            {
                "metric": metric,
                "direction": direction,
                "threshold_key": threshold_key,
                "threshold": threshold,
                "baseline": baseline_value,
                "target": target_value,
                "delta": delta,
                "available": bool(available),
                "pass": bool(passed),
            }
        )

    sample_count = int(report.get("sample_count", target.get("sample_count", baseline.get("sample_count", 0))))
    min_samples = int(threshold_values.get("min_samples", 1))
    sample_gate = {
        "metric": "sample_count",
        "direction": "minimum_value",
        "threshold_key": "min_samples",
        "threshold": int(min_samples),
        "target": int(sample_count),
        "delta": int(sample_count - min_samples),
        "pass": bool(sample_count >= min_samples),
    }
    criteria.append(sample_gate)
    passed = all(bool(item["pass"]) for item in criteria)
    verdict = "metric_gate_pass" if passed else "metric_gate_fail"
    return {
        "source_report": "" if source_report is None else str(source_report),
        "target_input": str(target_input),
        "baseline_input": str(baseline_input),
        "sample_count": int(sample_count),
        "thresholds": {
            "min_precision_delta": float(threshold_values.get("min_precision_delta", 0.0)),
            "min_recall_delta": float(threshold_values.get("min_recall_delta", 0.0)),
            "min_recall75_delta": float(threshold_values.get("min_recall75_delta", 0.0)),
            "min_small_recall_delta": float(threshold_values.get("min_small_recall_delta", 0.0)),
            "max_fp_delta": float(threshold_values.get("max_fp_delta", 0.0)),
            "min_samples": int(min_samples),
        },
        "baseline_metrics": {key: baseline.get(key) for key, *_ in CRITERIA},
        "target_metrics": {key: target.get(key) for key, *_ in CRITERIA},
        "criteria": criteria,
        "pass": bool(passed),
        "verdict": verdict,
        "interpretation": _interpretation(passed),
    }


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    if key not in metrics or metrics.get(key) is None:
        return None
    return float(metrics[key])


def write_claim_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "claim_gate_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
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


def _interpretation(passed: bool) -> str:
    if passed:
        return "The target passes this metric-only gate. This is still not a safety or product claim by itself."
    return "The target does not pass this metric-only gate, so a broad superiority claim is not supported."


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for item in summary.get("criteria", ()):
        status = "PASS" if bool(item.get("pass")) else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(item.get('metric', '')))}</td>"
            f"<td>{html_lib.escape(str(item.get('direction', '')))}</td>"
            f"<td>{_fmt(item.get('baseline'))}</td>"
            f"<td>{_fmt(item.get('target'))}</td>"
            f"<td>{_fmt(item.get('delta'), signed=True)}</td>"
            f"<td>{_fmt(item.get('threshold'), signed=True)}</td>"
            f"<td class=\"{status.lower()}\">{status}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Claim Gate</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 9px 10px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Claim Gate</h1>
  <p>Target <code>{html_lib.escape(str(summary.get('target_input', '')))}</code> vs baseline <code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code>.</p>
  <p><strong>Verdict:</strong> <code>{html_lib.escape(str(summary.get('verdict', '')))}</code></p>
  <p>{html_lib.escape(str(summary.get('interpretation', '')))}</p>
  <table>
    <thead><tr><th>Metric</th><th>Rule</th><th>Baseline</th><th>Target</th><th>Delta</th><th>Threshold</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>claim_gate_summary.json</code></p>
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
