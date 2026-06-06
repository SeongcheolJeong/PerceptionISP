"""Condition-slice robustness gate for PerceptionISP claims."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .claim_gate import CLAIM_PROFILES, CLAIM_PROFILE_DESCRIPTIONS
from .types import json_ready


CONDITION_GATE_SUMMARY = "condition_gate_summary.json"
CONDITION_METRICS_SUMMARY = "condition_metrics_summary.json"

CRITERIA = (
    ("precision@0.50_mean", "minimum_delta", "min_precision_delta", 0.0),
    ("recall@0.50_mean", "minimum_delta", "min_recall_delta", 0.0),
    ("recall@0.75_mean", "minimum_delta", "min_recall75_delta", 0.0),
    ("small_recall@0.50_mean", "minimum_delta", "min_small_recall_delta", 0.0),
    ("fp@0.50_mean", "maximum_delta", "max_fp_delta", 0.0),
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate condition-specific metrics against a conservative claim profile.")
    parser.add_argument("condition_metrics", help="Condition metrics report directory or condition_metrics_summary.json")
    parser.add_argument("--profile", default="fp_reducer", choices=sorted(CLAIM_PROFILES), help="Named metric gate profile.")
    parser.add_argument("--target-input", required=True)
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--condition", action="append", default=[], help="Condition name to evaluate. Default uses all conditions.")
    parser.add_argument("--exclude-condition", action="append", default=[], help="Condition name to skip even if present.")
    parser.add_argument("--min-condition-samples", type=int, default=30)
    parser.add_argument("--min-covered-conditions", type=int, default=1)
    parser.add_argument("--min-precision-delta", type=float, default=None)
    parser.add_argument("--min-recall-delta", type=float, default=None)
    parser.add_argument("--min-recall75-delta", type=float, default=None)
    parser.add_argument("--min-small-recall-delta", type=float, default=None)
    parser.add_argument("--max-fp-delta", type=float, default=None)
    parser.add_argument("--fail-on-fail", action="store_true", help="Return exit code 1 when the condition gate fails.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    summary_path = _summary_path(args.condition_metrics)
    condition_summary = json.loads(summary_path.read_text())
    thresholds = _cli_thresholds(args)
    summary = build_condition_gate(
        condition_summary,
        target_input=str(args.target_input),
        baseline_input=str(args.baseline_input),
        thresholds=thresholds,
        conditions=tuple(args.condition),
        exclude_conditions=tuple(args.exclude_condition),
        min_condition_samples=int(args.min_condition_samples),
        min_covered_conditions=int(args.min_covered_conditions),
        source_report=summary_path,
    )
    destination = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else summary_path.parent / f"condition_gate_{_safe_name(args.target_input)}_vs_{_safe_name(args.baseline_input)}"
    )
    html_path = write_condition_gate(summary, destination)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / CONDITION_GATE_SUMMARY),
                    "profile": summary["profile"],
                    "pass": summary["pass"],
                    "verdict": summary["verdict"],
                    "failed_conditions": [row["condition"] for row in summary["conditions"] if row["status"] == "fail"],
                    "skipped_conditions": [row["condition"] for row in summary["conditions"] if row["status"] == "skipped"],
                }
            ),
            indent=2,
        )
    )
    return 1 if bool(args.fail_on_fail) and not bool(summary["pass"]) else 0


def build_condition_gate(
    condition_summary: Mapping[str, Any],
    *,
    target_input: str,
    baseline_input: str = "human_rgb",
    thresholds: Mapping[str, Any] | None = None,
    conditions: Sequence[str] = (),
    exclude_conditions: Sequence[str] = (),
    min_condition_samples: int = 30,
    min_covered_conditions: int = 1,
    source_report: str | Path | None = None,
) -> Dict[str, Any]:
    threshold_values = _resolve_thresholds(thresholds)
    profile = str(threshold_values.pop("profile", "custom"))
    metrics = condition_summary.get("metrics", {}) if isinstance(condition_summary.get("metrics"), Mapping) else {}
    target = metrics.get(target_input)
    baseline = metrics.get(baseline_input)
    if not isinstance(target, Mapping):
        raise ValueError(f"target input not found in condition metrics: {target_input}")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"baseline input not found in condition metrics: {baseline_input}")

    selected = _selected_conditions(condition_summary, target, conditions=conditions, exclude_conditions=exclude_conditions)
    rows = []
    evaluated_count = 0
    for condition in selected:
        target_row = target.get(condition, {})
        baseline_row = baseline.get(condition, {})
        sample_count = int(
            _metric_value(target_row, "condition_sample_count")
            or _metric_value(target_row, "sample_count")
            or _condition_sample_count(condition_summary, condition)
        )
        if sample_count < int(min_condition_samples):
            rows.append(
                {
                    "condition": str(condition),
                    "sample_count": int(sample_count),
                    "status": "skipped",
                    "pass": None,
                    "criteria": [],
                    "reason": f"sample_count {sample_count} < min_condition_samples {int(min_condition_samples)}",
                }
            )
            continue
        evaluated_count += 1
        criteria = [
            _condition_criterion(
                target_row,
                baseline_row,
                metric=metric,
                direction=direction,
                threshold_key=threshold_key,
                threshold=float(threshold_values.get(threshold_key, default_value)),
            )
            for metric, direction, threshold_key, default_value in CRITERIA
        ]
        passed = all(bool(item["pass"]) for item in criteria)
        rows.append(
            {
                "condition": str(condition),
                "sample_count": int(sample_count),
                "status": "pass" if passed else "fail",
                "pass": bool(passed),
                "criteria": criteria,
                "reason": "",
            }
        )

    coverage_gate = {
        "condition": "__coverage__",
        "sample_count": int(evaluated_count),
        "status": "pass" if evaluated_count >= int(min_covered_conditions) else "fail",
        "pass": bool(evaluated_count >= int(min_covered_conditions)),
        "criteria": [
            {
                "metric": "evaluated_condition_count",
                "direction": "minimum_value",
                "threshold": int(min_covered_conditions),
                "target": int(evaluated_count),
                "delta": int(evaluated_count - int(min_covered_conditions)),
                "pass": bool(evaluated_count >= int(min_covered_conditions)),
            }
        ],
        "reason": "",
    }
    all_pass = bool(coverage_gate["pass"]) and all(row["status"] != "fail" for row in rows)
    verdict = "condition_gate_pass" if all_pass else "condition_gate_fail"
    return {
        "source_report": "" if source_report is None else str(source_report),
        "profile": profile,
        "profile_description": CLAIM_PROFILE_DESCRIPTIONS.get(profile, CLAIM_PROFILE_DESCRIPTIONS["custom"]),
        "target_input": str(target_input),
        "baseline_input": str(baseline_input),
        "min_condition_samples": int(min_condition_samples),
        "min_covered_conditions": int(min_covered_conditions),
        "condition_count": int(len(rows)),
        "evaluated_condition_count": int(evaluated_count),
        "skipped_condition_count": int(sum(1 for row in rows if row["status"] == "skipped")),
        "failed_condition_count": int(sum(1 for row in rows if row["status"] == "fail")),
        "thresholds": {
            "min_precision_delta": float(threshold_values.get("min_precision_delta", 0.0)),
            "min_recall_delta": float(threshold_values.get("min_recall_delta", 0.0)),
            "min_recall75_delta": float(threshold_values.get("min_recall75_delta", 0.0)),
            "min_small_recall_delta": float(threshold_values.get("min_small_recall_delta", 0.0)),
            "max_fp_delta": float(threshold_values.get("max_fp_delta", 0.0)),
        },
        "conditions": rows,
        "coverage_gate": coverage_gate,
        "pass": bool(all_pass),
        "verdict": verdict,
        "interpretation": _interpretation(all_pass, profile),
    }


def write_condition_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / CONDITION_GATE_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _condition_criterion(
    target: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    metric: str,
    direction: str,
    threshold_key: str,
    threshold: float,
) -> Dict[str, Any]:
    target_value = _metric_value(target, metric)
    baseline_value = _metric_value(baseline, metric)
    available = target_value is not None and baseline_value is not None
    delta_key = f"delta_{metric}"
    delta = _metric_value(target, delta_key)
    if delta is None and available:
        delta = float(target_value) - float(baseline_value)
    mean_pass = bool(available and delta is not None and (delta >= threshold if direction == "minimum_delta" else delta <= threshold))
    return {
        "metric": metric,
        "direction": direction,
        "threshold_key": threshold_key,
        "threshold": float(threshold),
        "baseline": baseline_value,
        "target": target_value,
        "delta": delta,
        "available": bool(available),
        "pass": bool(mean_pass),
    }


def _selected_conditions(
    condition_summary: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    conditions: Sequence[str],
    exclude_conditions: Sequence[str],
) -> Tuple[str, ...]:
    excludes = {str(value) for value in exclude_conditions}
    if conditions:
        return tuple(str(value) for value in conditions if str(value) not in excludes)
    names = [str(row.get("name", "")) for row in condition_summary.get("conditions", ()) if isinstance(row, Mapping)]
    if not names:
        names = sorted(str(value) for value in target)
    return tuple(name for name in names if name and name not in excludes)


def _condition_sample_count(condition_summary: Mapping[str, Any], condition: str) -> int:
    for row in condition_summary.get("conditions", ()):
        if isinstance(row, Mapping) and str(row.get("name", "")) == str(condition):
            return int(row.get("sample_count", 0))
    return 0


def _cli_thresholds(args: Any) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {"profile": str(args.profile), **CLAIM_PROFILES[str(args.profile)]}
    for key, value in (
        ("min_precision_delta", args.min_precision_delta),
        ("min_recall_delta", args.min_recall_delta),
        ("min_recall75_delta", args.min_recall75_delta),
        ("min_small_recall_delta", args.min_small_recall_delta),
        ("max_fp_delta", args.max_fp_delta),
    ):
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def _resolve_thresholds(thresholds: Mapping[str, Any] | None) -> Dict[str, Any]:
    raw = dict(thresholds or {})
    profile = str(raw.get("profile", "custom"))
    if profile != "custom" and profile not in CLAIM_PROFILES:
        raise ValueError(f"unknown claim profile: {profile}")
    values: Dict[str, Any] = dict(CLAIM_PROFILES.get(profile, {}))
    values.update({key: value for key, value in raw.items() if key != "profile"})
    values["profile"] = profile
    return values


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    if not isinstance(metrics, Mapping) or key not in metrics or metrics.get(key) is None:
        return None
    return float(metrics[key])


def _summary_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / CONDITION_METRICS_SUMMARY
    if not path.exists():
        raise FileNotFoundError(f"condition metrics summary not found: {path}")
    return path


def _interpretation(passed: bool, profile: str) -> str:
    if passed:
        if profile == "fp_reducer":
            return "All evaluated condition slices pass the recall-budgeted FP-reduction profile."
        return "All evaluated condition slices pass the configured condition robustness profile."
    return "One or more evaluated condition slices fail the configured robustness profile."


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for condition in summary.get("conditions", ()):
        criteria = condition.get("criteria", ()) if isinstance(condition, Mapping) else ()
        if not criteria:
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(condition.get('condition', '')))}</td>"
                f"<td>{int(condition.get('sample_count', 0))}</td>"
                f"<td class=\"skipped\">SKIPPED</td>"
                "<td colspan=\"6\"></td>"
                f"<td>{html_lib.escape(str(condition.get('reason', '')))}</td>"
                "</tr>"
            )
            continue
        for item in criteria:
            status = "PASS" if bool(item.get("pass")) else "FAIL"
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(condition.get('condition', '')))}</td>"
                f"<td>{int(condition.get('sample_count', 0))}</td>"
                f"<td class=\"{status.lower()}\">{status}</td>"
                f"<td>{html_lib.escape(str(item.get('metric', '')))}</td>"
                f"<td>{_fmt(item.get('baseline'))}</td>"
                f"<td>{_fmt(item.get('target'))}</td>"
                f"<td>{_fmt(item.get('delta'), signed=True)}</td>"
                f"<td>{_fmt(item.get('threshold'), signed=True)}</td>"
                f"<td>{html_lib.escape(str(item.get('direction', '')))}</td>"
                "<td></td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Condition Gate</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
    .skipped {{ color: #6b7280; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Condition Gate</h1>
  <p>Target <code>{html_lib.escape(str(summary.get('target_input', '')))}</code> vs baseline <code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code>.</p>
  <p><strong>Profile:</strong> <code>{html_lib.escape(str(summary.get('profile', '')))}</code> - {html_lib.escape(str(summary.get('profile_description', '')))}</p>
  <p><strong>Verdict:</strong> <code>{html_lib.escape(str(summary.get('verdict', '')))}</code>. {html_lib.escape(str(summary.get('interpretation', '')))}</p>
  <p>Evaluated conditions: {int(summary.get('evaluated_condition_count', 0))}; skipped: {int(summary.get('skipped_condition_count', 0))}; failed: {int(summary.get('failed_condition_count', 0))}.</p>
  <table>
    <thead><tr><th>Condition</th><th>Samples</th><th>Status</th><th>Metric</th><th>Baseline</th><th>Target</th><th>Delta</th><th>Threshold</th><th>Rule</th><th>Reason</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>{CONDITION_GATE_SUMMARY}</code></p>
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
