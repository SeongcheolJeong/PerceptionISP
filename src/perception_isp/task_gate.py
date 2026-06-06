"""Task-group metric gate for PerceptionISP claim readiness."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import json_ready


TASK_GATE_SUMMARY = "task_gate_summary.json"
TASK_METRICS_SUMMARY = "task_metrics_summary.json"

TASK_PROFILES = {
    "recall_improvement": {
        "min_precision_delta": 0.0,
        "min_recall_delta": 0.0,
        "min_recall75_delta": 0.0,
        "max_fp_delta": 0.0,
    },
    "fp_reducer": {
        "min_precision_delta": 0.0,
        "min_recall_delta": -0.01,
        "min_recall75_delta": -0.01,
        "max_fp_delta": 0.0,
    },
}

TASK_PROFILE_DESCRIPTIONS = {
    "recall_improvement": "Task recall gate: target must be no worse than baseline for precision, R50, R75, and FP/sample in every evaluated task group.",
    "fp_reducer": "Task FP-reducer gate: target may spend up to 0.01 absolute recall while keeping precision non-worse and FP/sample non-increasing in every evaluated task group.",
    "custom": "Custom task gate assembled from explicit threshold arguments.",
}

CRITERIA = (
    ("precision@0.50", "minimum_delta", "min_precision_delta", 0.0),
    ("recall@0.50", "minimum_delta", "min_recall_delta", 0.0),
    ("recall@0.75", "minimum_delta", "min_recall75_delta", 0.0),
    ("fp@0.50_per_sample", "maximum_delta", "max_fp_delta", 0.0),
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate task-group metrics against a configured claim profile.")
    parser.add_argument("task_metrics", help="Task metrics report directory or task_metrics_summary.json")
    parser.add_argument("--profile", default="recall_improvement", choices=sorted(TASK_PROFILES), help="Named task gate profile.")
    parser.add_argument("--target-input", required=True)
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--group", action="append", default=[], help="Task group to evaluate. Default uses all groups.")
    parser.add_argument("--exclude-group", action="append", default=[], help="Task group to skip even if present.")
    parser.add_argument("--min-group-gt", type=int, default=1)
    parser.add_argument("--min-covered-groups", type=int, default=1)
    parser.add_argument("--min-precision-delta", type=float, default=None)
    parser.add_argument("--min-recall-delta", type=float, default=None)
    parser.add_argument("--min-recall75-delta", type=float, default=None)
    parser.add_argument("--max-fp-delta", type=float, default=None)
    parser.add_argument("--fail-on-fail", action="store_true", help="Return exit code 1 when the task gate fails.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    summary_path = _summary_path(args.task_metrics)
    task_summary = json.loads(summary_path.read_text())
    summary = build_task_gate(
        task_summary,
        target_input=str(args.target_input),
        baseline_input=str(args.baseline_input),
        thresholds=_cli_thresholds(args),
        groups=tuple(args.group),
        exclude_groups=tuple(args.exclude_group),
        min_group_gt=int(args.min_group_gt),
        min_covered_groups=int(args.min_covered_groups),
        source_report=summary_path,
    )
    destination = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else summary_path.parent / f"task_gate_{_safe_name(args.target_input)}_vs_{_safe_name(args.baseline_input)}"
    )
    html_path = write_task_gate(summary, destination)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / TASK_GATE_SUMMARY),
                    "profile": summary["profile"],
                    "pass": summary["pass"],
                    "verdict": summary["verdict"],
                    "failed_groups": [row["group"] for row in summary["groups"] if row["status"] == "fail"],
                    "skipped_groups": [row["group"] for row in summary["groups"] if row["status"] == "skipped"],
                }
            ),
            indent=2,
        )
    )
    return 1 if bool(args.fail_on_fail) and not bool(summary["pass"]) else 0


def build_task_gate(
    task_summary: Mapping[str, Any],
    *,
    target_input: str,
    baseline_input: str = "human_rgb",
    thresholds: Mapping[str, Any] | None = None,
    groups: Sequence[str] = (),
    exclude_groups: Sequence[str] = (),
    min_group_gt: int = 1,
    min_covered_groups: int = 1,
    source_report: str | Path | None = None,
) -> Dict[str, Any]:
    threshold_values = _resolve_thresholds(thresholds)
    profile = str(threshold_values.pop("profile", "custom"))
    metrics = task_summary.get("metrics", {}) if isinstance(task_summary.get("metrics"), Mapping) else {}
    target = metrics.get(target_input)
    baseline = metrics.get(baseline_input)
    if not isinstance(target, Mapping):
        raise ValueError(f"target input not found in task metrics: {target_input}")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"baseline input not found in task metrics: {baseline_input}")

    selected = _selected_groups(task_summary, target, groups=groups, exclude_groups=exclude_groups)
    rows = []
    evaluated_count = 0
    for group in selected:
        target_row = target.get(group, {})
        baseline_row = baseline.get(group, {})
        gt_count = int(_metric_value(target_row, "gt_count") or _metric_value(baseline_row, "gt_count") or 0)
        if gt_count < int(min_group_gt):
            rows.append(
                {
                    "group": str(group),
                    "gt_count": int(gt_count),
                    "status": "skipped",
                    "pass": None,
                    "criteria": [],
                    "reason": f"gt_count {gt_count} < min_group_gt {int(min_group_gt)}",
                }
            )
            continue
        evaluated_count += 1
        criteria = [
            _task_criterion(
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
                "group": str(group),
                "gt_count": int(gt_count),
                "status": "pass" if passed else "fail",
                "pass": bool(passed),
                "criteria": criteria,
                "reason": "",
            }
        )

    coverage_gate = {
        "group": "__coverage__",
        "gt_count": int(evaluated_count),
        "status": "pass" if evaluated_count >= int(min_covered_groups) else "fail",
        "pass": bool(evaluated_count >= int(min_covered_groups)),
        "criteria": [
            {
                "metric": "evaluated_group_count",
                "direction": "minimum_value",
                "threshold": int(min_covered_groups),
                "target": int(evaluated_count),
                "delta": int(evaluated_count - int(min_covered_groups)),
                "pass": bool(evaluated_count >= int(min_covered_groups)),
            }
        ],
        "reason": "",
    }
    all_pass = bool(coverage_gate["pass"]) and all(row["status"] != "fail" for row in rows)
    verdict = "task_gate_pass" if all_pass else "task_gate_fail"
    return {
        "source_report": "" if source_report is None else str(source_report),
        "profile": profile,
        "profile_description": TASK_PROFILE_DESCRIPTIONS.get(profile, TASK_PROFILE_DESCRIPTIONS["custom"]),
        "target_input": str(target_input),
        "baseline_input": str(baseline_input),
        "min_group_gt": int(min_group_gt),
        "min_covered_groups": int(min_covered_groups),
        "group_count": int(len(rows)),
        "evaluated_group_count": int(evaluated_count),
        "skipped_group_count": int(sum(1 for row in rows if row["status"] == "skipped")),
        "failed_group_count": int(sum(1 for row in rows if row["status"] == "fail")),
        "thresholds": {
            "min_precision_delta": float(threshold_values.get("min_precision_delta", 0.0)),
            "min_recall_delta": float(threshold_values.get("min_recall_delta", 0.0)),
            "min_recall75_delta": float(threshold_values.get("min_recall75_delta", 0.0)),
            "max_fp_delta": float(threshold_values.get("max_fp_delta", 0.0)),
        },
        "groups": rows,
        "coverage_gate": coverage_gate,
        "pass": bool(all_pass),
        "verdict": verdict,
        "interpretation": _interpretation(all_pass, profile),
    }


def write_task_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / TASK_GATE_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _task_criterion(
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
    passed = bool(available and delta is not None and (delta >= threshold if direction == "minimum_delta" else delta <= threshold))
    return {
        "metric": metric,
        "direction": direction,
        "threshold_key": threshold_key,
        "threshold": float(threshold),
        "baseline": baseline_value,
        "target": target_value,
        "delta": delta,
        "available": bool(available),
        "pass": bool(passed),
    }


def _selected_groups(
    task_summary: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    groups: Sequence[str],
    exclude_groups: Sequence[str],
) -> Tuple[str, ...]:
    excludes = {str(value) for value in exclude_groups}
    if groups:
        return tuple(str(value) for value in groups if str(value) not in excludes)
    names = [str(row.get("name", "")) for row in task_summary.get("groups", ()) if isinstance(row, Mapping)]
    if not names:
        names = sorted(str(value) for value in target)
    return tuple(name for name in names if name and name not in excludes)


def _cli_thresholds(args: Any) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {"profile": str(args.profile), **TASK_PROFILES[str(args.profile)]}
    for key, value in (
        ("min_precision_delta", args.min_precision_delta),
        ("min_recall_delta", args.min_recall_delta),
        ("min_recall75_delta", args.min_recall75_delta),
        ("max_fp_delta", args.max_fp_delta),
    ):
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def _resolve_thresholds(thresholds: Mapping[str, Any] | None) -> Dict[str, Any]:
    raw = dict(thresholds or {})
    profile = str(raw.get("profile", "custom"))
    if profile != "custom" and profile not in TASK_PROFILES:
        raise ValueError(f"unknown task gate profile: {profile}")
    values: Dict[str, Any] = dict(TASK_PROFILES.get(profile, {}))
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
        path = path / TASK_METRICS_SUMMARY
    if not path.exists():
        raise FileNotFoundError(f"task metrics summary not found: {path}")
    return path


def _interpretation(passed: bool, profile: str) -> str:
    if passed:
        if profile == "recall_improvement":
            return "All evaluated task groups pass the task recall-improvement profile."
        if profile == "fp_reducer":
            return "All evaluated task groups pass the task FP-reducer profile."
        return "All evaluated task groups pass the configured task gate profile."
    return "One or more evaluated task groups fail the configured task gate profile."


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for group in summary.get("groups", ()):
        criteria = group.get("criteria", ()) if isinstance(group, Mapping) else ()
        if not criteria:
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(group.get('group', '')))}</td>"
                f"<td>{int(group.get('gt_count', 0))}</td>"
                f"<td class=\"skipped\">SKIPPED</td>"
                "<td colspan=\"6\"></td>"
                f"<td>{html_lib.escape(str(group.get('reason', '')))}</td>"
                "</tr>"
            )
            continue
        for item in criteria:
            status = "PASS" if bool(item.get("pass")) else "FAIL"
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(group.get('group', '')))}</td>"
                f"<td>{int(group.get('gt_count', 0))}</td>"
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
  <title>PerceptionISP Task Gate</title>
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
  <h1>PerceptionISP Task Gate</h1>
  <p>Target <code>{html_lib.escape(str(summary.get('target_input', '')))}</code> vs baseline <code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code>.</p>
  <p><strong>Profile:</strong> <code>{html_lib.escape(str(summary.get('profile', '')))}</code> - {html_lib.escape(str(summary.get('profile_description', '')))}</p>
  <p><strong>Verdict:</strong> <code>{html_lib.escape(str(summary.get('verdict', '')))}</code>. {html_lib.escape(str(summary.get('interpretation', '')))}</p>
  <p>Evaluated groups: {int(summary.get('evaluated_group_count', 0))}; skipped: {int(summary.get('skipped_group_count', 0))}; failed: {int(summary.get('failed_group_count', 0))}.</p>
  <table>
    <thead><tr><th>Group</th><th>GT</th><th>Status</th><th>Metric</th><th>Baseline</th><th>Target</th><th>Delta</th><th>Threshold</th><th>Rule</th><th>Reason</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>{TASK_GATE_SUMMARY}</code></p>
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
