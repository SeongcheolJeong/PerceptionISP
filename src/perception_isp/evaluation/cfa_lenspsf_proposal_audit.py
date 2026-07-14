"""Proposal-edge audit for CFA/LensPSF detector sweep reports."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from perception_isp.evaluation.aux_contribution_audit import sample_bridge_from_comparison_report
from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "cfa_lenspsf_proposal_audit_summary.json"
SWEEP_SUMMARY_FILENAME = "cfa_lenspsf_detector_sweep_summary.json"
DEFAULT_BASELINE_INPUT = "perception_fusion_rgb_aux"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit proposal edge/scene-edge support across a CFA/LensPSF detector sweep.")
    parser.add_argument("sweep", help="CFA/LensPSF detector sweep summary path or directory.")
    parser.add_argument("--baseline-input", default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--target-input", default=None, help="Target input. Defaults to the first calibrated input in each condition.")
    parser.add_argument("--output-dir", default="reports/perception_cfa_lenspsf_proposal_audit")
    args = parser.parse_args(argv)

    summary = build_cfa_lenspsf_proposal_audit_from_path(
        args.sweep,
        baseline_input=str(args.baseline_input),
        target_input=None if args.target_input is None else str(args.target_input),
    )
    html_path = write_cfa_lenspsf_proposal_audit(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "condition_count": summary["condition_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_cfa_lenspsf_proposal_audit_from_path(
    sweep: str | Path,
    *,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str | None = None,
) -> Dict[str, Any]:
    summary_path = _summary_path(sweep, SWEEP_SUMMARY_FILENAME)
    sweep_summary = json.loads(summary_path.read_text())
    return build_cfa_lenspsf_proposal_audit(
        sweep_summary,
        sweep_summary_path=summary_path,
        baseline_input=baseline_input,
        target_input=target_input,
    )


def build_cfa_lenspsf_proposal_audit(
    sweep_summary: Mapping[str, Any],
    *,
    sweep_summary_path: str | Path,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str | None = None,
) -> Dict[str, Any]:
    sweep_path = Path(sweep_summary_path).expanduser()
    sweep_dir = sweep_path.parent
    conditions = []
    for run in sweep_summary.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        selected_target = _select_target_input(run, preferred=target_input)
        report_path = _condition_report_path(sweep_dir, run)
        bridge = (
            sample_bridge_from_comparison_report(
                report_path,
                baseline_input=str(baseline_input),
                target_input=selected_target,
            )
            if selected_target and report_path.exists()
            else None
        )
        conditions.append(_condition_summary(run, report_path=report_path, baseline_input=str(baseline_input), target_input=selected_target, bridge=bridge))
    checks = _checks(conditions, expected_condition_count=int(sweep_summary.get("run_count", 0)))
    aggregate = _aggregate(conditions)
    return {
        "name": "CFA/LensPSF proposal-edge audit",
        "source_sweep": str(sweep_path),
        "source_sweep_html": str(sweep_dir / "index.html") if (sweep_dir / "index.html").exists() else "",
        "baseline_input": str(baseline_input),
        "target_input": "" if target_input is None else str(target_input),
        "condition_count": len(conditions),
        "expected_condition_count": int(sweep_summary.get("run_count", 0)),
        "cfa_patterns": [str(value) for value in sweep_summary.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in sweep_summary.get("psf_sigmas", ())],
        "checks": checks,
        "aggregate": aggregate,
        "conditions": conditions,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This audit reuses saved CFA/LensPSF detector sweep reports and compares baseline RGB+Aux fusion proposals "
            "against the calibrated downstream target on the same samples. It measures whether removed FP proposals have "
            "lower aux-edge and source-scene-edge support than kept TP proposals under each condition."
        ),
        "claim_boundary": (
            "This is post-hoc proposal-level diagnostic evidence for a calibrated proposal path. It is not an incremental aux-only "
            "ablation, not a trained RGB+Aux DNN result, and not native sensor-CFA proof when the source CFA was remapped."
        ),
    }


def write_cfa_lenspsf_proposal_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _select_target_input(run: Mapping[str, Any], *, preferred: str | None) -> str:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}
    if preferred and preferred in metrics:
        return str(preferred)
    for name in metrics:
        if str(name).startswith("perception_calibrated"):
            return str(name)
    if "perception_fusion_rgb_aux" in metrics:
        return "perception_fusion_rgb_aux"
    return next(iter(metrics), "")


def _condition_report_path(sweep_dir: Path, run: Mapping[str, Any]) -> Path:
    raw_report = str(run.get("report", ""))
    if not raw_report:
        return sweep_dir / str(run.get("run_id", "")) / "comparison_summary.json"
    path = sweep_dir / raw_report
    if path.name == "index.html":
        return path.with_name("comparison_summary.json")
    if path.is_dir():
        return path / "comparison_summary.json"
    return path


def _condition_summary(
    run: Mapping[str, Any],
    *,
    report_path: Path,
    baseline_input: str,
    target_input: str,
    bridge: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if bridge is None:
        return {
            "run_id": str(run.get("run_id", "")),
            "report": str(report_path),
            "cfa_pattern": str(run.get("cfa_pattern", "")),
            "psf_sigma": _maybe_float(run.get("psf_sigma")),
            "baseline_input": baseline_input,
            "target_input": target_input,
            "status": "missing_bridge",
        }
    edge = _correlation(bridge, feature="edge_support")
    scene = _correlation(bridge, feature="scene_edge_support")
    support_deltas = bridge.get("support_deltas", {}) if isinstance(bridge.get("support_deltas"), Mapping) else {}
    return {
        "run_id": str(run.get("run_id", "")),
        "report": str(report_path),
        "cfa_pattern": str(run.get("cfa_pattern", "")),
        "psf_sigma": _maybe_float(run.get("psf_sigma")),
        "baseline_input": baseline_input,
        "target_input": target_input,
        "status": "pass",
        "sample_count": int(bridge.get("compared_sample_count", 0)),
        "baseline_detection_count": int(bridge.get("baseline_detection_count", 0)),
        "target_detection_count": int(bridge.get("target_detection_count", 0)),
        "removed_fp_count": int(bridge.get("removed_fp_count", 0)),
        "removed_tp_count": int(bridge.get("removed_tp_count", 0)),
        "added_fp_count": int(bridge.get("added_fp_count", 0)),
        "added_tp_count": int(bridge.get("added_tp_count", 0)),
        "fp_delta_count": int(bridge.get("fp_delta_count", 0)),
        "tp_delta_count": int(bridge.get("tp_delta_count", 0)),
        "removed_fp_fraction": _maybe_float(bridge.get("removed_fp_fraction")),
        "removed_fp_to_tp_ratio": _maybe_float(bridge.get("removed_fp_to_tp_ratio")),
        "edge_support_delta_removed_fp_minus_kept_tp": _maybe_float(support_deltas.get("removed_fp_minus_kept_tp_edge_support_mean")),
        "scene_edge_support_delta_removed_fp_minus_kept_tp": _maybe_float(support_deltas.get("removed_fp_minus_kept_tp_scene_edge_support_mean")),
        "edge_auc_low_predicts_removed_fp": _maybe_float(edge.get("auc_low_feature_predicts_positive") if edge else None),
        "scene_edge_auc_low_predicts_removed_fp": _maybe_float(scene.get("auc_low_feature_predicts_positive") if scene else None),
        "edge_lower_predicts_removed_fp": bool(edge.get("lower_feature_predicts_positive")) if edge else False,
        "scene_edge_lower_predicts_removed_fp": bool(scene.get("lower_feature_predicts_positive")) if scene else False,
    }


def _correlation(bridge: Mapping[str, Any], *, feature: str) -> Mapping[str, Any] | None:
    proposal_correlation = bridge.get("proposal_correlation", {}) if isinstance(bridge.get("proposal_correlation"), Mapping) else {}
    for row in proposal_correlation.get("rows", ()):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("comparison", "")) == "removed_fp_vs_kept_tp" and str(row.get("feature", "")) == str(feature):
            return row
    return None


def _checks(conditions: Sequence[Mapping[str, Any]], *, expected_condition_count: int) -> Tuple[Dict[str, Any], ...]:
    pass_conditions = [row for row in conditions if str(row.get("status", "")) == "pass"]
    removed_fp_total = sum(int(row.get("removed_fp_count", 0)) for row in pass_conditions)
    removed_tp_total = sum(int(row.get("removed_tp_count", 0)) for row in pass_conditions)
    scene_auc_rows = [
        row for row in pass_conditions if row.get("scene_edge_auc_low_predicts_removed_fp") is not None
    ]
    scene_auc_positive = [
        row for row in scene_auc_rows if bool(row.get("scene_edge_lower_predicts_removed_fp")) and float(row.get("scene_edge_auc_low_predicts_removed_fp", 0.0)) > 0.5
    ]
    edge_auc_rows = [row for row in pass_conditions if row.get("edge_auc_low_predicts_removed_fp") is not None]
    edge_auc_positive = [
        row for row in edge_auc_rows if bool(row.get("edge_lower_predicts_removed_fp")) and float(row.get("edge_auc_low_predicts_removed_fp", 0.0)) > 0.5
    ]
    scene_delta_negative = _negative_condition_count(scene_auc_rows, "scene_edge_support_delta_removed_fp_minus_kept_tp")
    edge_delta_negative = _negative_condition_count(edge_auc_rows, "edge_support_delta_removed_fp_minus_kept_tp")
    scene_auc_mean = _condition_mean(scene_auc_rows, "scene_edge_auc_low_predicts_removed_fp")
    edge_auc_mean = _condition_mean(edge_auc_rows, "edge_auc_low_predicts_removed_fp")
    scene_delta_mean = _condition_mean(scene_auc_rows, "scene_edge_support_delta_removed_fp_minus_kept_tp")
    edge_delta_mean = _condition_mean(edge_auc_rows, "edge_support_delta_removed_fp_minus_kept_tp")
    edge_majority_threshold = 0 if not edge_auc_rows else (len(edge_auc_rows) // 2) + 1
    return (
        {
            "id": "condition_bridges_available",
            "status": "pass" if len(pass_conditions) == int(expected_condition_count) and expected_condition_count > 0 else "fail",
            "evidence": f"bridges={len(pass_conditions)} expected={expected_condition_count}",
        },
        {
            "id": "removed_fp_observed_across_conditions",
            "status": "pass" if removed_fp_total > removed_tp_total and removed_fp_total > 0 else "fail",
            "evidence": f"removed_fp={removed_fp_total} removed_tp={removed_tp_total}",
        },
        {
            "id": "source_scene_edge_predicts_removed_fp_in_some_conditions",
            "status": "pass" if scene_auc_positive else "fail",
            "evidence": f"positive_conditions={len(scene_auc_positive)}/{len(scene_auc_rows)}",
        },
        {
            "id": "source_scene_edge_consistent_across_conditions",
            "status": "pass"
            if scene_auc_rows
            and len(scene_auc_positive) == len(scene_auc_rows)
            and scene_delta_negative == len(scene_auc_rows)
            and scene_auc_mean is not None
            and scene_auc_mean > 0.5
            and scene_delta_mean is not None
            and scene_delta_mean < 0.0
            else "fail",
            "evidence": (
                f"auc_positive={len(scene_auc_positive)}/{len(scene_auc_rows)} "
                f"delta_negative={scene_delta_negative}/{len(scene_auc_rows)} "
                f"mean_delta={_fmt(scene_delta_mean, signed=True)} mean_auc={_fmt(scene_auc_mean)}"
            ),
        },
        {
            "id": "aux_edge_predicts_removed_fp_in_some_conditions",
            "status": "pass" if edge_auc_positive else "fail",
            "evidence": f"positive_conditions={len(edge_auc_positive)}/{len(edge_auc_rows)}",
        },
        {
            "id": "aux_edge_consistent_across_majority_conditions",
            "status": "pass"
            if edge_auc_rows
            and len(edge_auc_positive) >= edge_majority_threshold
            and edge_delta_negative >= edge_majority_threshold
            and edge_auc_mean is not None
            and edge_auc_mean > 0.5
            and edge_delta_mean is not None
            and edge_delta_mean < 0.0
            else "fail",
            "evidence": (
                f"auc_positive={len(edge_auc_positive)}/{len(edge_auc_rows)} "
                f"delta_negative={edge_delta_negative}/{len(edge_auc_rows)} "
                f"threshold={edge_majority_threshold} "
                f"mean_delta={_fmt(edge_delta_mean, signed=True)} mean_auc={_fmt(edge_auc_mean)}"
            ),
        },
    )


def _aggregate(conditions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pass_conditions = [row for row in conditions if str(row.get("status", "")) == "pass"]
    best_scene = _best_auc(pass_conditions, "scene_edge_auc_low_predicts_removed_fp")
    best_edge = _best_auc(pass_conditions, "edge_auc_low_predicts_removed_fp")
    return {
        "condition_count": len(pass_conditions),
        "sample_count": sum(int(row.get("sample_count", 0)) for row in pass_conditions),
        "baseline_detection_count": sum(int(row.get("baseline_detection_count", 0)) for row in pass_conditions),
        "target_detection_count": sum(int(row.get("target_detection_count", 0)) for row in pass_conditions),
        "removed_fp_count": sum(int(row.get("removed_fp_count", 0)) for row in pass_conditions),
        "removed_tp_count": sum(int(row.get("removed_tp_count", 0)) for row in pass_conditions),
        "added_fp_count": sum(int(row.get("added_fp_count", 0)) for row in pass_conditions),
        "added_tp_count": sum(int(row.get("added_tp_count", 0)) for row in pass_conditions),
        "fp_delta_count": sum(int(row.get("fp_delta_count", 0)) for row in pass_conditions),
        "tp_delta_count": sum(int(row.get("tp_delta_count", 0)) for row in pass_conditions),
        "scene_edge_positive_condition_count": sum(1 for row in pass_conditions if bool(row.get("scene_edge_lower_predicts_removed_fp"))),
        "edge_positive_condition_count": sum(1 for row in pass_conditions if bool(row.get("edge_lower_predicts_removed_fp"))),
        "scene_edge_delta_negative_condition_count": _negative_condition_count(pass_conditions, "scene_edge_support_delta_removed_fp_minus_kept_tp"),
        "edge_delta_negative_condition_count": _negative_condition_count(pass_conditions, "edge_support_delta_removed_fp_minus_kept_tp"),
        "scene_edge_auc_condition_mean": _condition_mean(pass_conditions, "scene_edge_auc_low_predicts_removed_fp"),
        "edge_auc_condition_mean": _condition_mean(pass_conditions, "edge_auc_low_predicts_removed_fp"),
        "scene_edge_support_delta_condition_mean": _condition_mean(pass_conditions, "scene_edge_support_delta_removed_fp_minus_kept_tp"),
        "edge_support_delta_condition_mean": _condition_mean(pass_conditions, "edge_support_delta_removed_fp_minus_kept_tp"),
        "scene_edge_auc_removed_fp_weighted_mean": _weighted_condition_mean(pass_conditions, "scene_edge_auc_low_predicts_removed_fp", "removed_fp_count"),
        "edge_auc_removed_fp_weighted_mean": _weighted_condition_mean(pass_conditions, "edge_auc_low_predicts_removed_fp", "removed_fp_count"),
        "scene_edge_support_delta_removed_fp_weighted_mean": _weighted_condition_mean(pass_conditions, "scene_edge_support_delta_removed_fp_minus_kept_tp", "removed_fp_count"),
        "edge_support_delta_removed_fp_weighted_mean": _weighted_condition_mean(pass_conditions, "edge_support_delta_removed_fp_minus_kept_tp", "removed_fp_count"),
        "best_scene_edge_auc_condition": best_scene,
        "best_edge_auc_condition": best_edge,
    }


def _condition_mean(conditions: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_maybe_float(row.get(key)) for row in conditions if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _weighted_condition_mean(conditions: Sequence[Mapping[str, Any]], key: str, weight_key: str) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for row in conditions:
        if row.get(key) is None:
            continue
        weight = float(row.get(weight_key, 0) or 0)
        if weight <= 0.0:
            continue
        weighted_sum += float(row.get(key)) * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return float(weighted_sum / weight_sum)


def _negative_condition_count(conditions: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in conditions if row.get(key) is not None and float(row.get(key, 0.0)) < 0.0)


def _best_auc(conditions: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    rows = [row for row in conditions if row.get(key) is not None]
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get(key, 0.0)))
    return {
        "run_id": best.get("run_id"),
        "cfa_pattern": best.get("cfa_pattern"),
        "psf_sigma": best.get("psf_sigma"),
        key: _maybe_float(best.get(key)),
    }


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{html_lib.escape(str(row.get('status', '')))}\">{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in summary.get("checks", ())
        if isinstance(row, Mapping)
    )
    condition_rows = "".join(_condition_row(row, destination) for row in summary.get("conditions", ()))
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP CFA/LensPSF Proposal Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e8f3f1; position: sticky; top: 0; }}
    a {{ color: #155e75; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pass, .pos {{ color: #047857; font-weight: 650; }}
    .fail, .warning, .neg {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP CFA/LensPSF Proposal Audit</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>. Conditions={int(summary.get('condition_count', 0))}/{int(summary.get('expected_condition_count', 0))}.
  Source sweep: {_source_sweep_link(summary, destination)}</p>
  <h2>Aggregate</h2>
  <table><tbody>
    <tr><th>Removed FP</th><td>{int(aggregate.get('removed_fp_count', 0))}</td><th>Removed TP</th><td>{int(aggregate.get('removed_tp_count', 0))}</td></tr>
    <tr><th>Net FP Delta</th><td>{int(aggregate.get('fp_delta_count', 0))}</td><th>Net TP Delta</th><td>{int(aggregate.get('tp_delta_count', 0))}</td></tr>
    <tr><th>Scene-Edge d/AUC Mean</th><td>{_fmt(aggregate.get('scene_edge_support_delta_condition_mean'), signed=True)} / {_fmt(aggregate.get('scene_edge_auc_condition_mean'))}</td><th>Aux-Edge d/AUC Mean</th><td>{_fmt(aggregate.get('edge_support_delta_condition_mean'), signed=True)} / {_fmt(aggregate.get('edge_auc_condition_mean'))}</td></tr>
    <tr><th>Scene-Edge Negative d Conditions</th><td>{int(aggregate.get('scene_edge_delta_negative_condition_count', 0))}</td><th>Aux-Edge Negative d Conditions</th><td>{int(aggregate.get('edge_delta_negative_condition_count', 0))}</td></tr>
    <tr><th>Scene-Edge Removed-FP Weighted d/AUC</th><td>{_fmt(aggregate.get('scene_edge_support_delta_removed_fp_weighted_mean'), signed=True)} / {_fmt(aggregate.get('scene_edge_auc_removed_fp_weighted_mean'))}</td><th>Aux-Edge Removed-FP Weighted d/AUC</th><td>{_fmt(aggregate.get('edge_support_delta_removed_fp_weighted_mean'), signed=True)} / {_fmt(aggregate.get('edge_auc_removed_fp_weighted_mean'))}</td></tr>
    <tr><th>Best Scene-Edge AUC</th><td>{html_lib.escape(str(aggregate.get('best_scene_edge_auc_condition', {})))}</td><th>Best Aux-Edge AUC</th><td>{html_lib.escape(str(aggregate.get('best_edge_auc_condition', {})))}</td></tr>
  </tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Condition Bridges</h2>
  <table>
    <thead><tr><th>Run</th><th>CFA</th><th>PSF</th><th>Samples</th><th>Removed FP</th><th>Removed TP</th><th>FP Delta</th><th>Edge d</th><th>Edge AUC</th><th>Scene d</th><th>Scene AUC</th></tr></thead>
    <tbody>{condition_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _condition_row(row: Mapping[str, Any], destination: Path) -> str:
    report = str(row.get("report", ""))
    link = html_lib.escape(str(row.get("run_id", "")))
    if report:
        html_path = Path(report).with_name("index.html")
        relative = os.path.relpath(str(html_path), start=str(destination))
        link = f"<a href=\"{html_lib.escape(relative)}\">{html_lib.escape(str(row.get('run_id', '')))}</a>"
    return (
        "<tr>"
        f"<td>{link}</td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{int(row.get('sample_count', 0))}</td>"
        f"<td>{int(row.get('removed_fp_count', 0))}</td>"
        f"<td>{int(row.get('removed_tp_count', 0))}</td>"
        f"<td>{int(row.get('fp_delta_count', 0))}</td>"
        f"<td>{_fmt(row.get('edge_support_delta_removed_fp_minus_kept_tp'), signed=True)}</td>"
        f"<td>{_fmt(row.get('edge_auc_low_predicts_removed_fp'))}</td>"
        f"<td>{_fmt(row.get('scene_edge_support_delta_removed_fp_minus_kept_tp'), signed=True)}</td>"
        f"<td>{_fmt(row.get('scene_edge_auc_low_predicts_removed_fp'))}</td>"
        "</tr>"
    )


def _source_sweep_link(summary: Mapping[str, Any], destination: Path) -> str:
    html_path = str(summary.get("source_sweep_html", ""))
    if not html_path:
        return ""
    relative = os.path.relpath(html_path, start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


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
