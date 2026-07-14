"""Task-specific gates over an adverse native RAW slice."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from perception_isp.evaluation.task_gate import TASK_PROFILES, build_task_gate
from perception_isp.evaluation.task_metrics import build_task_metrics, parse_label_groups, parse_list
from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "adverse_task_slice_summary.json"
ADVERSE_NATIVE_SLICE_SUMMARY = "adverse_native_slice_summary.json"
DEFAULT_TARGET_INPUT = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compute task-specific gates for each condition in an adverse native RAW slice.")
    parser.add_argument("adverse_report", help="Adverse native RAW slice directory or adverse_native_slice_summary.json")
    parser.add_argument("--target-input", default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--profile", default="fp_reducer", choices=sorted(TASK_PROFILES))
    parser.add_argument("--inputs", default=None, help="Comma-separated inputs. Default uses baseline and target.")
    parser.add_argument("--group", action="append", default=[], help="Custom label group as name=label1,label2. Repeats are allowed.")
    parser.add_argument("--no-default-groups", action="store_true")
    parser.add_argument("--exclude-group", action="append", default=[], help="Task group to skip in the gate.")
    parser.add_argument("--small-area-threshold", type=float, default=32.0 * 32.0)
    parser.add_argument("--min-group-gt", type=int, default=1)
    parser.add_argument("--min-covered-groups", type=int, default=1)
    parser.add_argument("--output-dir", default="reports/perception_adverse_task_slice")
    args = parser.parse_args(argv)

    adverse_path = _adverse_summary_path(args.adverse_report)
    adverse_summary = json.loads(adverse_path.read_text())
    summary = build_adverse_task_slice(
        adverse_summary,
        adverse_summary_path=adverse_path,
        target_input=str(args.target_input),
        baseline_input=str(args.baseline_input),
        profile=str(args.profile),
        inputs=parse_list(args.inputs) or (str(args.baseline_input), str(args.target_input)),
        label_groups=parse_label_groups(args.group, include_defaults=not bool(args.no_default_groups)),
        exclude_groups=tuple(args.exclude_group),
        small_area_threshold=float(args.small_area_threshold),
        min_group_gt=int(args.min_group_gt),
        min_covered_groups=int(args.min_covered_groups),
    )
    html_path = write_adverse_task_slice(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "adverse_passed_condition_count": summary["aggregate"]["adverse_passed_condition_count"],
                    "adverse_condition_count": summary["aggregate"]["adverse_condition_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_adverse_task_slice(
    adverse_summary: Mapping[str, Any],
    *,
    adverse_summary_path: str | Path,
    target_input: str = DEFAULT_TARGET_INPUT,
    baseline_input: str = "human_rgb",
    profile: str = "fp_reducer",
    inputs: Sequence[str] | None = None,
    label_groups: Mapping[str, Sequence[str]] | None = None,
    exclude_groups: Sequence[str] = (),
    small_area_threshold: float = 32.0 * 32.0,
    min_group_gt: int = 1,
    min_covered_groups: int = 1,
) -> Dict[str, Any]:
    root = Path(adverse_summary_path).expanduser().parent
    selected_inputs = tuple(inputs) if inputs else (str(baseline_input), str(target_input))
    condition_rows = []
    for run in adverse_summary.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        report_path = _condition_comparison_path(root, run)
        report = json.loads(report_path.read_text())
        task_summary = build_task_metrics(
            report,
            source_report=report_path,
            baseline_input=str(baseline_input),
            inputs=selected_inputs,
            label_groups=label_groups,
            small_area_threshold=float(small_area_threshold),
        )
        gate = build_task_gate(
            task_summary,
            target_input=str(target_input),
            baseline_input=str(baseline_input),
            thresholds={"profile": str(profile)},
            exclude_groups=tuple(exclude_groups),
            min_group_gt=int(min_group_gt),
            min_covered_groups=int(min_covered_groups),
            source_report=report_path,
        )
        condition_rows.append(_condition_row(run, report_path, task_summary, gate, target_input=str(target_input), baseline_input=str(baseline_input)))
    aggregate = _aggregate_condition_rows(condition_rows)
    checks = _checks(condition_rows, adverse_summary=adverse_summary, aggregate=aggregate)
    status = "pass" if all(row["status"] in {"pass", "warning"} for row in checks) else "fail"
    return {
        "name": "Adverse-condition task-specific slice",
        "source_adverse_native_slice": str(adverse_summary_path),
        "source_adverse_native_slice_html": str(root / "index.html") if (root / "index.html").exists() else "",
        "status": status,
        "claim_status": _claim_status(aggregate),
        "profile": str(profile),
        "target_input": str(target_input),
        "baseline_input": str(baseline_input),
        "inputs": list(selected_inputs),
        "small_area_threshold": float(small_area_threshold),
        "min_group_gt": int(min_group_gt),
        "min_covered_groups": int(min_covered_groups),
        "condition_count": int(len(condition_rows)),
        "expected_condition_count": int(adverse_summary.get("run_count", adverse_summary.get("expected_run_count", 0))),
        "adverse_conditions": [str(row.get("condition")) for row in condition_rows if str(row.get("condition")) != "nominal"],
        "cfa_pattern": str(adverse_summary.get("cfa_pattern", "")),
        "psf_sigma": adverse_summary.get("psf_sigma"),
        "aggregate": aggregate,
        "checks": checks,
        "conditions": condition_rows,
        "group_summary": _group_summary(condition_rows),
        "interpretation": (
            "This report reuses saved adverse native RAW condition detections and evaluates task groups "
            "such as VRU, person, cyclist, vehicle, and small objects. It does not rerun CameraE2E, ISP, or the detector."
        ),
        "claim_boundary": (
            "Use this as simulated adverse task-slice evidence. It is not a real adverse dataset gate, "
            "and skipped low-GT groups should not be treated as passing task evidence."
        ),
    }


def write_adverse_task_slice(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _condition_row(
    run: Mapping[str, Any],
    report_path: Path,
    task_summary: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    target_input: str,
    baseline_input: str,
) -> Dict[str, Any]:
    target_metrics = task_summary.get("metrics", {}).get(target_input, {}) if isinstance(task_summary.get("metrics"), Mapping) else {}
    baseline_metrics = task_summary.get("metrics", {}).get(baseline_input, {}) if isinstance(task_summary.get("metrics"), Mapping) else {}
    group_rows = []
    for group in gate.get("groups", ()):
        if not isinstance(group, Mapping):
            continue
        name = str(group.get("group", ""))
        target = target_metrics.get(name, {}) if isinstance(target_metrics, Mapping) else {}
        baseline = baseline_metrics.get(name, {}) if isinstance(baseline_metrics, Mapping) else {}
        group_rows.append(
            {
                "group": name,
                "status": str(group.get("status", "")),
                "pass": group.get("pass"),
                "gt_count": int(group.get("gt_count", 0)),
                "baseline_recall@0.50": _maybe_float(baseline.get("recall@0.50")) if isinstance(baseline, Mapping) else None,
                "target_recall@0.50": _maybe_float(target.get("recall@0.50")) if isinstance(target, Mapping) else None,
                "target_precision@0.50": _maybe_float(target.get("precision@0.50")) if isinstance(target, Mapping) else None,
                "target_fp@0.50_per_sample": _maybe_float(target.get("fp@0.50_per_sample")) if isinstance(target, Mapping) else None,
                "delta_precision@0.50": _maybe_float(target.get("delta_precision@0.50")) if isinstance(target, Mapping) else None,
                "delta_recall@0.50": _maybe_float(target.get("delta_recall@0.50")) if isinstance(target, Mapping) else None,
                "delta_recall@0.75": _maybe_float(target.get("delta_recall@0.75")) if isinstance(target, Mapping) else None,
                "delta_fp@0.50_per_sample": _maybe_float(target.get("delta_fp@0.50_per_sample")) if isinstance(target, Mapping) else None,
                "reason": str(group.get("reason", "")),
            }
        )
    failed = [row["group"] for row in group_rows if row["status"] == "fail"]
    skipped = [row["group"] for row in group_rows if row["status"] == "skipped"]
    return {
        "condition": str(run.get("condition", "")),
        "run_id": str(run.get("run_id", "")),
        "report": str(report_path),
        "sample_count": int(run.get("sample_count", 0)),
        "verdict": str(gate.get("verdict", "")),
        "pass": bool(gate.get("pass")),
        "evaluated_group_count": int(gate.get("evaluated_group_count", 0)),
        "failed_group_count": int(gate.get("failed_group_count", 0)),
        "skipped_group_count": int(gate.get("skipped_group_count", 0)),
        "failed_groups": failed,
        "skipped_groups": skipped,
        "groups": group_rows,
    }


def _aggregate_condition_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    adverse = [row for row in rows if str(row.get("condition", "")) != "nominal"]
    passed = [row for row in rows if bool(row.get("pass"))]
    adverse_passed = [row for row in adverse if bool(row.get("pass"))]
    return {
        "condition_count": int(len(rows)),
        "adverse_condition_count": int(len(adverse)),
        "passed_condition_count": int(len(passed)),
        "failed_condition_count": int(len(rows) - len(passed)),
        "adverse_passed_condition_count": int(len(adverse_passed)),
        "adverse_failed_condition_count": int(len(adverse) - len(adverse_passed)),
        "evaluated_group_count": int(sum(int(row.get("evaluated_group_count", 0)) for row in rows)),
        "failed_group_count": int(sum(int(row.get("failed_group_count", 0)) for row in rows)),
        "skipped_group_count": int(sum(int(row.get("skipped_group_count", 0)) for row in rows)),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    groups = sorted({str(group.get("group", "")) for row in rows for group in row.get("groups", ()) if isinstance(group, Mapping)})
    summaries = []
    for name in groups:
        group_rows = [
            group
            for row in rows
            for group in row.get("groups", ())
            if isinstance(group, Mapping) and str(group.get("group", "")) == name
        ]
        evaluated = [row for row in group_rows if str(row.get("status", "")) != "skipped"]
        deltas_r = [_maybe_float(row.get("delta_recall@0.50")) for row in evaluated]
        deltas_fp = [_maybe_float(row.get("delta_fp@0.50_per_sample")) for row in evaluated]
        deltas_p = [_maybe_float(row.get("delta_precision@0.50")) for row in evaluated]
        deltas_r = [value for value in deltas_r if value is not None]
        deltas_fp = [value for value in deltas_fp if value is not None]
        deltas_p = [value for value in deltas_p if value is not None]
        summaries.append(
            {
                "group": name,
                "condition_count": int(len(group_rows)),
                "evaluated_condition_count": int(len(evaluated)),
                "pass_condition_count": int(sum(1 for row in group_rows if str(row.get("status", "")) == "pass")),
                "fail_condition_count": int(sum(1 for row in group_rows if str(row.get("status", "")) == "fail")),
                "skipped_condition_count": int(sum(1 for row in group_rows if str(row.get("status", "")) == "skipped")),
                "gt_count_total": int(sum(int(row.get("gt_count", 0)) for row in group_rows)),
                "mean_delta_precision@0.50": _mean(deltas_p),
                "mean_delta_recall@0.50": _mean(deltas_r),
                "mean_delta_fp@0.50_per_sample": _mean(deltas_fp),
                "worst_delta_recall@0.50": min(deltas_r) if deltas_r else None,
                "worst_delta_fp@0.50_per_sample": max(deltas_fp) if deltas_fp else None,
            }
        )
    return tuple(summaries)


def _checks(rows: Sequence[Mapping[str, Any]], *, adverse_summary: Mapping[str, Any], aggregate: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    expected = int(adverse_summary.get("run_count", adverse_summary.get("expected_run_count", 0)))
    adverse = int(aggregate.get("adverse_condition_count", 0))
    adverse_passed = int(aggregate.get("adverse_passed_condition_count", 0))
    majority_threshold = adverse // 2 + 1
    return (
        {
            "id": "condition_reports_available",
            "status": "pass" if len(rows) == expected and expected > 0 else "fail",
            "evidence": f"conditions={len(rows)} expected={expected}",
        },
        {
            "id": "task_groups_evaluated",
            "status": "pass" if int(aggregate.get("evaluated_group_count", 0)) > 0 else "fail",
            "evidence": f"evaluated_groups={int(aggregate.get('evaluated_group_count', 0))}",
        },
        {
            "id": "adverse_conditions_evaluated",
            "status": "pass" if adverse > 0 else "fail",
            "evidence": f"adverse_conditions={adverse}",
        },
        {
            "id": "adverse_task_gate_pass_observed",
            "status": "pass" if adverse_passed > 0 else "warning",
            "evidence": f"adverse_passed={adverse_passed}/{adverse}",
        },
        {
            "id": "adverse_task_gate_majority",
            "status": "pass" if adverse > 0 and adverse_passed >= majority_threshold else "warning",
            "evidence": f"adverse_passed={adverse_passed}/{adverse}",
        },
    )


def _claim_status(aggregate: Mapping[str, Any]) -> str:
    adverse = int(aggregate.get("adverse_condition_count", 0))
    passed = int(aggregate.get("adverse_passed_condition_count", 0))
    if adverse > 0 and passed == adverse:
        return "adverse_task_gate_supported"
    if adverse > 0 and passed >= adverse // 2 + 1:
        return "adverse_task_gate_partially_supported"
    if passed > 0:
        return "adverse_task_gate_mixed"
    return "adverse_task_diagnostic_only"


def _condition_comparison_path(root: Path, run: Mapping[str, Any]) -> Path:
    report = Path(str(run.get("report", "")))
    candidate = report if report.is_absolute() else root / report
    if candidate.name == "index.html":
        candidate = candidate.with_name("comparison_summary.json")
    if candidate.is_dir():
        candidate = candidate / "comparison_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"condition comparison summary not found: {candidate}")
    return candidate


def _adverse_summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / ADVERSE_NATIVE_SLICE_SUMMARY
    if not candidate.exists():
        raise FileNotFoundError(f"adverse native slice summary not found: {candidate}")
    return candidate


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if summary.get("status") == "pass" else "not_supported"
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    group_rows = "".join(_group_row(row) for row in summary.get("group_summary", ()) if isinstance(row, Mapping))
    condition_rows = "".join(_condition_html_row(row, destination) for row in summary.get("conditions", ()) if isinstance(row, Mapping))
    if not check_rows:
        check_rows = '<tr><td colspan="3">No checks were available.</td></tr>'
    if not group_rows:
        group_rows = '<tr><td colspan="10">No group summaries were available.</td></tr>'
    if not condition_rows:
        condition_rows = '<tr><td colspan="10">No condition rows were available.</td></tr>'
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Adverse Task Slice</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .supported {{ color: #047857; font-weight: 700; }}
    .not_supported {{ color: #b91c1c; font-weight: 700; }}
    .warning, .diagnostic {{ color: #a16207; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Adverse Task Slice</h1>
  <p>Status: <code class=\"{status_class}\">{html_lib.escape(str(summary.get('status', '')))}</code>;
  claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.
  {html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <table><thead><tr><th>Profile</th><th>Baseline</th><th>Target</th><th>CFA</th><th>PSF</th><th>Conditions</th><th>Adverse Pass</th><th>Failed Groups</th><th>Skipped Groups</th></tr></thead><tbody><tr>
    <td><code>{html_lib.escape(str(summary.get('profile', '')))}</code></td>
    <td><code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code></td>
    <td><code>{html_lib.escape(str(summary.get('target_input', '')))}</code></td>
    <td><code>{html_lib.escape(str(summary.get('cfa_pattern', '')))}</code></td>
    <td>{_fmt(summary.get('psf_sigma'))}</td>
    <td>{int(summary.get('condition_count', 0))}/{int(summary.get('expected_condition_count', 0))}</td>
    <td>{int(aggregate.get('adverse_passed_condition_count', 0))}/{int(aggregate.get('adverse_condition_count', 0))}</td>
    <td>{int(aggregate.get('failed_group_count', 0))}</td>
    <td>{int(aggregate.get('skipped_group_count', 0))}</td>
  </tr></tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Task Groups</h2>
  <table><thead><tr><th>Group</th><th>Evaluated</th><th>Pass</th><th>Fail</th><th>Skipped</th><th>GT</th><th>Mean dP50</th><th>Mean dR50</th><th>Mean dFP50/sample</th><th>Worst dR50</th></tr></thead><tbody>{group_rows}</tbody></table>
  <h2>Conditions</h2>
  <table><thead><tr><th>Condition</th><th>Report</th><th>Samples</th><th>Verdict</th><th>Evaluated</th><th>Failed</th><th>Skipped</th><th>Failed Groups</th><th>Skipped Groups</th></tr></thead><tbody>{condition_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    status_class = "supported" if status == "pass" else "warning" if status == "warning" else "not_supported"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{status_class}\">{html_lib.escape(status)}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _group_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('group', '')))}</code></td>"
        f"<td>{int(row.get('evaluated_condition_count', 0))}</td>"
        f"<td>{int(row.get('pass_condition_count', 0))}</td>"
        f"<td>{int(row.get('fail_condition_count', 0))}</td>"
        f"<td>{int(row.get('skipped_condition_count', 0))}</td>"
        f"<td>{int(row.get('gt_count_total', 0))}</td>"
        f"<td>{_fmt(row.get('mean_delta_precision@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('mean_delta_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('mean_delta_fp@0.50_per_sample'), signed=True)}</td>"
        f"<td>{_fmt(row.get('worst_delta_recall@0.50'), signed=True)}</td>"
        "</tr>"
    )


def _condition_html_row(row: Mapping[str, Any], destination: Path) -> str:
    report = Path(str(row.get("report", "")))
    link = html_lib.escape(str(row.get("run_id", "")) or str(row.get("condition", "")))
    html_path = report.with_name("index.html")
    if html_path.exists():
        relative = os.path.relpath(str(html_path), start=str(destination))
        link = f"<a href=\"{html_lib.escape(relative)}\">{link}</a>"
    status_class = "supported" if bool(row.get("pass")) else "not_supported"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('condition', '')))}</code></td>"
        f"<td>{link}</td>"
        f"<td>{int(row.get('sample_count', 0))}</td>"
        f"<td class=\"{status_class}\">{html_lib.escape(str(row.get('verdict', '')))}</td>"
        f"<td>{int(row.get('evaluated_group_count', 0))}</td>"
        f"<td>{int(row.get('failed_group_count', 0))}</td>"
        f"<td>{int(row.get('skipped_group_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('failed_groups', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('skipped_groups', ())) or 'none')}</td>"
        "</tr>"
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
