"""Assemble PerceptionISP claim-readiness evidence into one report."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import json_ready


CLAIM_GATE_SUMMARY = "claim_gate_summary.json"
TRAINING_ROLLUP_SUMMARY = "training_rollup_summary.json"
COMPARISON_ROLLUP_SUMMARY = "rollup_summary.json"
TASK_METRICS_SUMMARY = "task_metrics_summary.json"
TASK_GATE_SUMMARY = "task_gate_summary.json"
PROTOCOL_COVERAGE_SUMMARY = "protocol_coverage_summary.json"
MECHANISM_VALIDATION_SUMMARY = "mechanism_validation_summary.json"
CFA_STRESS_SWEEP_SUMMARY = "cfa_stress_sweep_summary.json"
EDGE_CONFIDENCE_SUMMARY = "edge_confidence_suite_summary.json"
EDGE_FIDELITY_SUMMARY = "edge_fidelity_suite_summary.json"
SCENE_EDGE_CONFIDENCE_SUMMARY = "scene_edge_confidence_summary.json"
SCENE_INFORMATION_STRESS_SUMMARY = "scene_information_stress_summary.json"
AUX_CONTRIBUTION_AUDIT_SUMMARY = "aux_contribution_audit_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a consolidated PerceptionISP claim-readiness dashboard.")
    parser.add_argument("--claim-gate", action="append", default=[], help="Claim gate summary path/dir, optionally name=path.")
    parser.add_argument("--training-rollup", default=None, help="RGB+aux training rollup summary path/dir.")
    parser.add_argument("--task-metrics", default=None, help="Task metrics summary path/dir.")
    parser.add_argument("--task-gate", default=None, help="Task gate summary path/dir.")
    parser.add_argument("--protocol-coverage", default=None, help="Benchmark protocol coverage summary path/dir.")
    parser.add_argument("--mechanism-validation", default=None, help="Mechanism validation summary path/dir.")
    parser.add_argument("--cfa-stress-sweep", default=None, help="CFA stress sweep summary path/dir.")
    parser.add_argument("--edge-confidence-suite", default=None, help="Edge-confidence suite summary path/dir.")
    parser.add_argument("--edge-fidelity-suite", default=None, help="Object edge-fidelity suite summary path/dir.")
    parser.add_argument("--scene-edge-confidence", default=None, help="Scene-edge confidence summary path/dir.")
    parser.add_argument("--scene-information-stress", default=None, help="Scene-information stress summary path/dir.")
    parser.add_argument("--aux-contribution-audit", default=None, help="Aux contribution audit summary path/dir.")
    parser.add_argument("--comparison-rollup", action="append", default=[], help="Comparison rollup summary path/dir, optionally name=path.")
    parser.add_argument("--output-dir", default="reports/perception_claim_readiness_dashboard")
    args = parser.parse_args(argv)

    dashboard = build_claim_dashboard(
        claim_gate_specs=args.claim_gate,
        training_rollup=args.training_rollup,
        task_metrics=args.task_metrics,
        task_gate=args.task_gate,
        protocol_coverage=args.protocol_coverage,
        mechanism_validation=args.mechanism_validation,
        cfa_stress_sweep=args.cfa_stress_sweep,
        edge_confidence_suite=args.edge_confidence_suite,
        edge_fidelity_suite=args.edge_fidelity_suite,
        scene_edge_confidence=args.scene_edge_confidence,
        scene_information_stress=args.scene_information_stress,
        aux_contribution_audit=args.aux_contribution_audit,
        comparison_rollup_specs=args.comparison_rollup,
    )
    html_path = write_claim_dashboard(dashboard, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "claim_dashboard_summary.json"),
                    "decision_count": len(dashboard["decisions"]),
                    "claim_count": len(dashboard["claims"]),
                }
            ),
            indent=2,
        )
    )
    return 0


def build_claim_dashboard(
    *,
    claim_gate_specs: Sequence[str | Path],
    training_rollup: str | Path | None = None,
    task_metrics: str | Path | None = None,
    task_gate: str | Path | None = None,
    protocol_coverage: str | Path | None = None,
    mechanism_validation: str | Path | None = None,
    cfa_stress_sweep: str | Path | None = None,
    edge_confidence_suite: str | Path | None = None,
    edge_fidelity_suite: str | Path | None = None,
    scene_edge_confidence: str | Path | None = None,
    scene_information_stress: str | Path | None = None,
    aux_contribution_audit: str | Path | None = None,
    comparison_rollup_specs: Sequence[str | Path] = (),
) -> Dict[str, Any]:
    claims = [_load_claim_gate(spec) for spec in claim_gate_specs]
    training = _load_training_rollup(training_rollup) if training_rollup is not None else None
    task = _load_task_metrics(task_metrics, claims=claims) if task_metrics is not None else None
    task_gate_data = _load_task_gate(task_gate) if task_gate is not None else None
    protocol = _load_protocol_coverage(protocol_coverage) if protocol_coverage is not None else None
    mechanism = _load_mechanism_validation(mechanism_validation) if mechanism_validation is not None else None
    cfa_stress = _load_cfa_stress_sweep(cfa_stress_sweep) if cfa_stress_sweep is not None else None
    edge_confidence = _load_edge_confidence_suite(edge_confidence_suite) if edge_confidence_suite is not None else None
    edge_fidelity = _load_edge_fidelity_suite(edge_fidelity_suite) if edge_fidelity_suite is not None else None
    scene_edge = _load_scene_edge_confidence(scene_edge_confidence) if scene_edge_confidence is not None else None
    scene_information = _load_scene_information_stress(scene_information_stress) if scene_information_stress is not None else None
    aux_contribution = _load_aux_contribution_audit(aux_contribution_audit) if aux_contribution_audit is not None else None
    comparison_rollups = [_load_comparison_rollup(spec) for spec in comparison_rollup_specs]
    decisions = _claim_decisions(
        claims,
        training,
        task,
        task_gate_data,
        protocol,
        mechanism,
        cfa_stress,
        edge_confidence,
        edge_fidelity,
        scene_edge,
        scene_information,
        aux_contribution,
    )
    return {
        "claims": claims,
        "training": training,
        "task_metrics": task,
        "task_gate": task_gate_data,
        "protocol_coverage": protocol,
        "mechanism_validation": mechanism,
        "cfa_stress_sweep": cfa_stress,
        "edge_confidence_suite": edge_confidence,
        "edge_fidelity_suite": edge_fidelity,
        "scene_edge_confidence": scene_edge,
        "scene_information_stress": scene_information,
        "aux_contribution_audit": aux_contribution,
        "comparison_rollups": comparison_rollups,
        "decisions": decisions,
        "interpretation": "This dashboard separates supported engineering claims from claims that the current evidence does not support.",
    }


def write_claim_dashboard(dashboard: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "claim_dashboard_summary.json").write_text(json.dumps(json_ready(dashboard), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(dashboard, destination))
    return html_path


def _load_claim_gate(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CLAIM_GATE_SUMMARY)
    data = json.loads(summary_path.read_text())
    failed = [item for item in data.get("criteria", ()) if not bool(item.get("pass"))]
    metrics = {item.get("metric"): _criterion_summary(item) for item in data.get("criteria", ()) if item.get("metric") != "sample_count"}
    profile = str(data.get("profile", "custom"))
    passed = bool(data.get("pass"))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "profile": profile,
        "verdict": str(data.get("verdict", "")),
        "pass": passed,
        "sample_count": int(data.get("sample_count", 0)),
        "target_input": str(data.get("target_input", "")),
        "baseline_input": str(data.get("baseline_input", "")),
        "claim": _claim_text(profile, passed),
        "failed_metrics": [str(item.get("metric", "")) for item in failed],
        "metrics": metrics,
        "interpretation": str(data.get("interpretation", "")),
    }


def _criterion_summary(item: Mapping[str, Any]) -> Dict[str, Any]:
    paired = item.get("paired_delta", {}) if isinstance(item.get("paired_delta"), Mapping) else {}
    return {
        "delta": _maybe_float(item.get("delta")),
        "threshold": _maybe_float(item.get("threshold")),
        "ci_low": _maybe_float(paired.get("ci_low")),
        "ci_high": _maybe_float(paired.get("ci_high")),
        "pass": bool(item.get("pass")),
    }


def _load_training_rollup(path: str | Path) -> Dict[str, Any]:
    summary_path = _summary_path(path, TRAINING_ROLLUP_SUMMARY)
    data = json.loads(summary_path.read_text())
    runs = list(data.get("runs", ()))
    train_runs = [run for run in runs if str(run.get("kind", "")).startswith("train")]
    dense_eval_runs = [run for run in runs if str(run.get("kind", "")) == "dense_eval"]
    best_train = _max_by(train_runs, "throughput")
    best_recall = _max_metric(dense_eval_runs, "recall@0.50_mean")
    lowest_fp = _min_metric(dense_eval_runs, "fp@0.50_mean")
    max_precision = _max_metric(dense_eval_runs, "precision@0.50_mean")
    status = _training_status(best_recall, lowest_fp, max_precision, train_runs, dense_eval_runs)
    return {
        "name": _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "run_count": int(data.get("run_count", len(runs))),
        "status": status,
        "best_train_rate": _training_run_brief(best_train),
        "best_recall_eval": _dense_eval_brief(best_recall),
        "lowest_fp_eval": _dense_eval_brief(lowest_fp),
        "max_precision_eval": _dense_eval_brief(max_precision),
        "interpretation": _training_interpretation(status),
    }


def _load_comparison_rollup(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, COMPARISON_ROLLUP_SUMMARY)
    data = json.loads(summary_path.read_text())
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "run_count": int(data.get("run_count", 0)),
        "baseline_input": str(data.get("baseline_input", "")),
    }


def _load_task_metrics(spec: str | Path, *, claims: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, TASK_METRICS_SUMMARY)
    data = json.loads(summary_path.read_text())
    baseline_input = str(data.get("baseline_input", ""))
    target_input = _select_task_target(data, claims=claims, baseline_input=baseline_input)
    rows = _task_group_rows(data, target_input=target_input)
    status = _task_metrics_status(rows, target_input=target_input)
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "baseline_input": baseline_input,
        "target_input": target_input,
        "input_count": len(data.get("inputs", ())),
        "group_count": len(data.get("groups", ())),
        "label_agnostic": bool(data.get("label_agnostic", True)),
        "status": status,
        "rows": rows,
        "interpretation": _task_metrics_interpretation(status, baseline_input=baseline_input, target_input=target_input),
    }


def _load_task_gate(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, TASK_GATE_SUMMARY)
    data = json.loads(summary_path.read_text())
    groups = [row for row in data.get("groups", ()) if isinstance(row, Mapping)]
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "profile": str(data.get("profile", "")),
        "verdict": str(data.get("verdict", "")),
        "pass": bool(data.get("pass")),
        "target_input": str(data.get("target_input", "")),
        "baseline_input": str(data.get("baseline_input", "")),
        "evaluated_group_count": int(data.get("evaluated_group_count", 0)),
        "failed_group_count": int(data.get("failed_group_count", 0)),
        "skipped_group_count": int(data.get("skipped_group_count", 0)),
        "failed_groups": [str(row.get("group", "")) for row in groups if row.get("status") == "fail"],
        "skipped_groups": [str(row.get("group", "")) for row in groups if row.get("status") == "skipped"],
        "interpretation": str(data.get("interpretation", "")),
    }


def _load_protocol_coverage(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, PROTOCOL_COVERAGE_SUMMARY)
    data = json.loads(summary_path.read_text())
    requirements = [row for row in data.get("requirements", ()) if isinstance(row, Mapping)]
    missing = [row for row in requirements if str(row.get("status", "")) != "covered"]
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": str(data.get("status", "")),
        "coverage_status": str(data.get("coverage_status", "")),
        "metric_claim_status": str(data.get("metric_claim_status", "")),
        "claim_gate_outcomes": dict(data.get("claim_gate_outcomes", {})) if isinstance(data.get("claim_gate_outcomes"), Mapping) else {},
        "missing_required": list(data.get("missing_required", ())),
        "missing_raw_claim": list(data.get("missing_raw_claim", ())),
        "missing_count": len(missing),
        "requirements": requirements,
        "interpretation": str(data.get("interpretation", "")),
    }


def _load_mechanism_validation(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, MECHANISM_VALIDATION_SUMMARY)
    data = json.loads(summary_path.read_text())
    mechanisms = [row for row in data.get("mechanisms", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in mechanisms if str(row.get("status", "")) != "pass"]
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "mechanism_count": len(mechanisms),
        "failed_mechanisms": failed,
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "interpretation": str(data.get("interpretation", "")),
    }


def _load_cfa_stress_sweep(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CFA_STRESS_SWEEP_SUMMARY)
    data = json.loads(summary_path.read_text())
    support = data.get("support", {}) if isinstance(data.get("support"), Mapping) else {}
    rankings = [row for row in data.get("condition_rankings", ()) if isinstance(row, Mapping)]
    top_rows = []
    for ranking in rankings:
        ranked = ranking.get("ranked_cfas", ()) if isinstance(ranking.get("ranked_cfas", ()), Sequence) else ()
        first = next((row for row in ranked if isinstance(row, Mapping)), None)
        if first is None:
            continue
        top_rows.append(
            {
                "condition": str(ranking.get("condition", "")),
                "cfa_pattern": str(first.get("cfa_pattern", "")),
                "condition_score": _maybe_float(first.get("condition_score")),
                "score_definition": str(ranking.get("score_definition", "")),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass",
        "case_count": int(support.get("case_count", len(data.get("cases", ())))),
        "condition_count": len(rankings),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "top_rows": top_rows,
        "interpretation": str(data.get("interpretation", "")),
    }


def _load_edge_confidence_suite(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, EDGE_CONFIDENCE_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    status = str(data.get("status", ""))
    key_deltas = []
    for check in checks:
        criteria = check.get("criteria", ()) if isinstance(check.get("criteria", ()), Sequence) else ()
        for criterion in criteria:
            if not isinstance(criterion, Mapping):
                continue
            key_deltas.append(
                {
                    "check": str(check.get("id", "")),
                    "metric": str(criterion.get("metric", "")),
                    "delta": _maybe_float(criterion.get("delta")),
                    "threshold": _maybe_float(criterion.get("threshold")),
                    "pass": bool(criterion.get("pass")),
                }
            )
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "failed_checks": failed,
        "key_deltas": key_deltas[:8],
        "interpretation": str(data.get("interpretation", "")),
    }


def _load_edge_fidelity_suite(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, EDGE_FIDELITY_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    top_rows = []
    for ranking in data.get("rankings", ()):
        if not isinstance(ranking, Mapping):
            continue
        ranked = ranking.get("ranked_cfas", ()) if isinstance(ranking.get("ranked_cfas", ()), Sequence) else ()
        first = next((row for row in ranked if isinstance(row, Mapping)), None)
        if first is None:
            continue
        top_rows.append(
            {
                "psf_sigma": _maybe_float(ranking.get("psf_sigma")),
                "cfa_pattern": str(first.get("cfa_pattern", "")),
                "aux_object_edge_f1": _maybe_float(first.get("aux_object_edge_f1")),
                "perception_object_edge_f1": _maybe_float(first.get("perception_object_edge_f1")),
                "edge_confidence_separation": _maybe_float(first.get("edge_confidence_separation")),
            }
        )
    cases = []
    for row in data.get("cases", ()):
        if not isinstance(row, Mapping):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
        cases.append(
            {
                "id": str(row.get("id", "")),
                "psf_sigma": _maybe_float(row.get("psf_sigma")),
                "cfa_pattern": str(row.get("cfa_pattern", "")),
                "human_object_edge_f1": _maybe_float(metrics.get("human_object_edge_f1")),
                "perception_object_edge_f1": _maybe_float(metrics.get("perception_object_edge_f1")),
                "aux_object_edge_f1": _maybe_float(metrics.get("aux_object_edge_f1")),
                "edge_confidence_separation": _maybe_float(metrics.get("edge_confidence_separation")),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "failed_checks": failed,
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_maybe_float(value) for value in data.get("psf_sigmas", ())],
        "top_rows": top_rows,
        "cases": cases[:12],
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _load_scene_edge_confidence(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, SCENE_EDGE_CONFIDENCE_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    cases = []
    for row in data.get("cases", ()):
        if not isinstance(row, Mapping):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
        cases.append(
            {
                "id": str(row.get("id", "")),
                "source": str(row.get("source", "")),
                "cfa_pattern": str(row.get("cfa_pattern", "")),
                "human_rgb_proxy_source_edge_f1": _maybe_float(metrics.get("human_rgb_proxy_source_edge_f1")),
                "perception_rgb_proxy_source_edge_f1": _maybe_float(metrics.get("perception_rgb_proxy_source_edge_f1")),
                "perception_aux_strength_source_edge_f1": _maybe_float(metrics.get("perception_aux_strength_source_edge_f1")),
                "perception_aux_confidence_source_edge_f1": _maybe_float(metrics.get("perception_aux_confidence_source_edge_f1")),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "failed_checks": failed,
        "human_rgb_proxy_source_edge_f1_mean": _maybe_float(aggregate.get("human_rgb_proxy_source_edge_f1_mean")),
        "perception_rgb_proxy_source_edge_f1_mean": _maybe_float(aggregate.get("perception_rgb_proxy_source_edge_f1_mean")),
        "perception_aux_strength_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_strength_source_edge_f1_mean")),
        "perception_aux_confidence_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_confidence_source_edge_f1_mean")),
        "cases": cases[:12],
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _load_scene_information_stress(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, SCENE_INFORMATION_STRESS_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    cases = []
    for row in data.get("cases", ()):
        if not isinstance(row, Mapping):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
        cases.append(
            {
                "id": str(row.get("id", "")),
                "sample_mode": str(row.get("sample_mode", "")),
                "scene_luma_gradient_p90": _maybe_float(metrics.get("scene_luma_gradient_p90")),
                "sensor_luma_gradient_p90": _maybe_float(metrics.get("sensor_luma_gradient_p90")),
                "luma_detail_retention_p90": _maybe_float(metrics.get("luma_detail_retention_p90")),
                "scene_chroma_gradient_p90": _maybe_float(metrics.get("scene_chroma_gradient_p90")),
                "color_confidence_mean": _maybe_float(metrics.get("color_confidence_mean")),
                "signal_contrast_retention": _maybe_float(metrics.get("signal_contrast_retention")),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "failed_checks": failed,
        "sensor_width": int(data.get("sensor_width", 0)),
        "sensor_height": int(data.get("sensor_height", 0)),
        "scene_width": int(data.get("scene_width", 0)),
        "scene_height": int(data.get("scene_height", 0)),
        "cfa_pattern": str(data.get("cfa_pattern", "")),
        "cases": cases,
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _load_aux_contribution_audit(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, AUX_CONTRIBUTION_AUDIT_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    comparisons = []
    for row in data.get("comparisons", ()):
        if not isinstance(row, Mapping):
            continue
        deltas = row.get("deltas", {}) if isinstance(row.get("deltas"), Mapping) else {}
        comparisons.append(
            {
                "id": str(row.get("id", "")),
                "target_input": str(row.get("target_input", "")),
                "baseline_input": str(row.get("baseline_input", "")),
                "delta_precision@0.50": _maybe_float(deltas.get("precision@0.50_mean")),
                "delta_recall@0.50": _maybe_float(deltas.get("recall@0.50_mean")),
                "delta_fp@0.50": _maybe_float(deltas.get("fp@0.50_mean")),
            }
        )
    status = str(data.get("status", ""))
    feature_audit = data.get("feature_audit", {}) if isinstance(data.get("feature_audit"), Mapping) else {}
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "check_count": len(checks),
        "failed_checks": failed,
        "comparisons": comparisons,
        "aux_feature_count": int(feature_audit.get("aux_feature_count", 0)),
        "aux_features": [str(value) for value in feature_audit.get("aux_features", ())],
        "interpretation": str(data.get("interpretation", "")),
    }


def _select_task_target(data: Mapping[str, Any], *, claims: Sequence[Mapping[str, Any]], baseline_input: str) -> str:
    available = {str(value) for value in data.get("inputs", ())}
    for profile in ("fp_reducer", "broad_superiority"):
        for claim in claims:
            if str(claim.get("profile", "")) != profile:
                continue
            target = str(claim.get("target_input", ""))
            if target in available and target != baseline_input:
                return target
    for input_name in reversed([str(value) for value in data.get("inputs", ())]):
        if input_name and input_name != baseline_input:
            return input_name
    return ""


def _task_group_rows(data: Mapping[str, Any], *, target_input: str) -> list[Dict[str, Any]]:
    if not target_input:
        return []
    metrics_by_input = data.get("metrics", {}) if isinstance(data.get("metrics"), Mapping) else {}
    target_metrics = metrics_by_input.get(target_input, {}) if isinstance(metrics_by_input.get(target_input), Mapping) else {}
    group_specs = data.get("groups", ()) if isinstance(data.get("groups", ()), Sequence) else ()
    group_names = [str(group.get("name", "")) for group in group_specs if isinstance(group, Mapping)]
    ordered_names = [name for name in ("vru", "person", "cyclist", "vehicle", "traffic_light", "small_all") if name in group_names]
    ordered_names.extend(name for name in group_names if name not in ordered_names)
    rows = []
    for group_name in ordered_names:
        metrics = target_metrics.get(group_name, {}) if isinstance(target_metrics.get(group_name), Mapping) else {}
        rows.append(
            {
                "group": group_name,
                "gt_count": int(metrics.get("gt_count", 0)),
                "det_count": int(metrics.get("det_count", 0)),
                "precision@0.50": _maybe_float_or_none(metrics.get("precision@0.50")),
                "recall@0.50": _maybe_float_or_none(metrics.get("recall@0.50")),
                "recall@0.75": _maybe_float_or_none(metrics.get("recall@0.75")),
                "fp@0.50_per_sample": _maybe_float_or_none(metrics.get("fp@0.50_per_sample")),
                "delta_precision@0.50": _maybe_float_or_none(metrics.get("delta_precision@0.50")),
                "delta_recall@0.50": _maybe_float_or_none(metrics.get("delta_recall@0.50")),
                "delta_recall@0.75": _maybe_float_or_none(metrics.get("delta_recall@0.75")),
                "delta_fp@0.50_per_sample": _maybe_float_or_none(metrics.get("delta_fp@0.50_per_sample")),
            }
        )
    return rows


def _task_metrics_status(rows: Sequence[Mapping[str, Any]], *, target_input: str) -> str:
    if not target_input:
        return "missing_target"
    if not rows:
        return "missing_groups"
    priority_rows = [row for row in rows if row.get("group") in {"vru", "person", "vehicle", "small_all"}]
    evaluated = priority_rows or list(rows)
    recall_drops = [row for row in evaluated if _signed_metric(row, "delta_recall@0.50") is not None and (_signed_metric(row, "delta_recall@0.50") or 0.0) < 0.0]
    if recall_drops:
        return "recall_tradeoff"
    fp_reductions = [row for row in evaluated if _signed_metric(row, "delta_fp@0.50_per_sample") is not None and (_signed_metric(row, "delta_fp@0.50_per_sample") or 0.0) < 0.0]
    if fp_reductions:
        return "candidate_needs_gate"
    return "diagnostic_only"


def _task_metrics_interpretation(status: str, *, baseline_input: str, target_input: str) -> str:
    if status == "recall_tradeoff":
        return (
            f"Task groups for {target_input} show lower FP in places, but at least one priority recall group is below "
            f"{baseline_input}; do not claim VRU/person/task recall improvement."
        )
    if status == "candidate_needs_gate":
        return "Task-group metrics look directionally useful, but need a configured held-out task gate before promotion to a claim."
    if status == "missing_target":
        return "No non-baseline target input was available in the task metrics summary."
    if status == "missing_groups":
        return "No task groups were available for the selected target input."
    return "Task metrics are diagnostic evidence only."


def _claim_decisions(
    claims: Sequence[Mapping[str, Any]],
    training: Mapping[str, Any] | None,
    task_metrics: Mapping[str, Any] | None,
    task_gate: Mapping[str, Any] | None,
    protocol_coverage: Mapping[str, Any] | None,
    mechanism_validation: Mapping[str, Any] | None,
    cfa_stress_sweep: Mapping[str, Any] | None,
    edge_confidence_suite: Mapping[str, Any] | None,
    edge_fidelity_suite: Mapping[str, Any] | None,
    scene_edge_confidence: Mapping[str, Any] | None,
    scene_information_stress: Mapping[str, Any] | None,
    aux_contribution_audit: Mapping[str, Any] | None,
) -> list[Dict[str, Any]]:
    decisions: list[Dict[str, Any]] = []
    broad_claims = [claim for claim in claims if claim.get("profile") == "broad_superiority"]
    fp_claims = [claim for claim in claims if claim.get("profile") == "fp_reducer"]
    if any(bool(claim.get("pass")) for claim in broad_claims):
        decisions.append({"status": "supported", "claim": "Broad metric superiority versus HumanISP is supported by the configured gate."})
    elif broad_claims:
        decisions.append({"status": "not_supported", "claim": "Broad HumanISP superiority is not supported by the current gate evidence."})
    passing_fp_claim = next((claim for claim in fp_claims if bool(claim.get("pass"))), None)
    if passing_fp_claim is not None:
        baseline = _baseline_claim_name(str(passing_fp_claim.get("baseline_input", "")))
        decisions.append({"status": "supported", "claim": f"Recall-budgeted FP reduction versus {baseline} is supported."})
    elif fp_claims:
        baseline = _baseline_claim_name(str(fp_claims[0].get("baseline_input", "")))
        decisions.append({"status": "not_supported", "claim": f"Recall-budgeted FP reduction versus {baseline} is not supported by the current gate evidence."})
    if training is not None:
        status = str(training.get("status", "unknown"))
        if status == "diagnostic_only":
            decisions.append({"status": "not_supported", "claim": "The learned RGB+Aux DNN path is implemented and trainable, but current dense-detector metrics are not claim-quality."})
        elif status == "candidate_needs_gate":
            decisions.append({"status": "needs_gate", "claim": "The learned RGB+Aux DNN path has candidate metrics, but still needs a held-out claim gate."})
        elif status == "training_path_only":
            decisions.append({"status": "needs_eval", "claim": "The RGB+Aux DNN training path exists, but direct held-out detector evaluation is still missing."})
    if mechanism_validation is not None:
        if bool(mechanism_validation.get("pass")):
            decisions.append({"status": "supported", "claim": "PerceptionISP front-end mechanism validation passed; aux/confidence maps respond to controlled sensor stressors."})
        else:
            failed = ", ".join(str(value) for value in mechanism_validation.get("failed_mechanisms", ())) or "configured mechanism checks"
            decisions.append({"status": "not_supported", "claim": f"PerceptionISP front-end mechanism validation failed for {failed}; do not use aux maps as feasibility evidence yet."})
    if cfa_stress_sweep is not None:
        if bool(cfa_stress_sweep.get("pass")):
            decisions.append({"status": "diagnostic", "claim": "CFA stress sweep is available as diagnostic evidence for condition-dependent front-end signals; it is not detector-performance evidence."})
        else:
            decisions.append({"status": "not_supported", "claim": "CFA stress sweep failed or is incomplete; do not use it as CFA feasibility evidence yet."})
    if edge_confidence_suite is not None:
        if bool(edge_confidence_suite.get("pass")):
            decisions.append({"status": "diagnostic", "claim": "Edge-confidence suite passed; PerceptionISP confidence maps respond to difficult-edge stressors, but this is not detector-performance evidence."})
        else:
            failed = ", ".join(str(value) for value in edge_confidence_suite.get("failed_checks", ())) or "configured edge-confidence checks"
            decisions.append({"status": "not_supported", "claim": f"Edge-confidence suite failed for {failed}; do not use edge confidence as feasibility evidence yet."})
    if edge_fidelity_suite is not None:
        if bool(edge_fidelity_suite.get("pass")):
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": "Object edge-fidelity suite passed; HumanISP, PerceptionISP, and aux edge maps are compared against object/sensor edge oracles across CFA and LensPSF, but this is not detector-performance evidence.",
                }
            )
        else:
            failed = ", ".join(str(value) for value in edge_fidelity_suite.get("failed_checks", ())) or "configured object edge-fidelity checks"
            decisions.append({"status": "not_supported", "claim": f"Object edge-fidelity suite failed for {failed}; do not use edge fidelity as feasibility evidence yet."})
    if scene_edge_confidence is not None:
        if bool(scene_edge_confidence.get("pass")):
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": "Scene edge-confidence suite passed; HumanISP RGB, PerceptionISP RGB, aux edge-strength, and aux edge-confidence are compared against a high-information scene-edge proxy, but this is not detector-performance evidence.",
                }
            )
        else:
            failed = ", ".join(str(value) for value in scene_edge_confidence.get("failed_checks", ())) or "configured scene edge-confidence checks"
            decisions.append({"status": "not_supported", "claim": f"Scene edge-confidence suite failed for {failed}; do not use scene-edge confidence as feasibility evidence yet."})
    if scene_information_stress is not None:
        if bool(scene_information_stress.get("pass")):
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": "Scene-information stress suite passed; high-information scene loss and CFA/color uncertainty are covered as diagnostic evidence, not detector-performance evidence.",
                }
            )
        else:
            failed = ", ".join(str(value) for value in scene_information_stress.get("failed_checks", ())) or "configured scene-information checks"
            decisions.append({"status": "not_supported", "claim": f"Scene-information stress suite failed for {failed}; do not use RGB-scene pass-through tests as feasibility evidence."})
    if aux_contribution_audit is not None:
        if bool(aux_contribution_audit.get("pass")):
            decisions.append({"status": "diagnostic", "claim": "Aux contribution audit passed; aux features add proposal-scoring FP reduction within the recall budget, but this is calibration evidence rather than DNN performance."})
        else:
            failed = ", ".join(str(value) for value in aux_contribution_audit.get("failed_checks", ())) or "configured aux contribution checks"
            decisions.append({"status": "not_supported", "claim": f"Aux contribution audit failed for {failed}; do not claim aux helps proposal scoring yet."})
    if task_gate is not None:
        profile = str(task_gate.get("profile", "task"))
        if bool(task_gate.get("pass")):
            decisions.append({"status": "supported", "claim": f"Task-level `{profile}` gate passed for the evaluated groups."})
        else:
            failed = ", ".join(str(value) for value in task_gate.get("failed_groups", ())) or "configured groups"
            decisions.append({"status": "not_supported", "claim": f"Task-level `{profile}` gate failed for {failed}; do not promote that task-level claim."})
    elif task_metrics is not None:
        status = str(task_metrics.get("status", "unknown"))
        if status == "recall_tradeoff":
            decisions.append({"status": "not_supported", "claim": "Task-level VRU/person recall improvement versus HumanISP is not supported; the current evidence supports only the narrower FP-reduction claim."})
        elif status == "candidate_needs_gate":
            decisions.append({"status": "needs_gate", "claim": "Task-group metrics are candidate evidence, but need a configured held-out task gate before a task-level claim."})
        elif status in {"missing_target", "missing_groups"}:
            decisions.append({"status": "needs_eval", "claim": "Task-group evidence is incomplete, so task-level claims are not ready."})
    if protocol_coverage is not None:
        status = str(protocol_coverage.get("status", "unknown"))
        coverage_status = str(protocol_coverage.get("coverage_status") or ("coverage_complete" if status == "claim_ready" else "coverage_incomplete"))
        if coverage_status == "coverage_complete":
            metric_status = str(protocol_coverage.get("metric_claim_status", "unknown"))
            suffix = "" if metric_status in {"", "unknown"} else f" Metric claim status: {metric_status}."
            decisions.append({"status": "supported", "claim": "The configured benchmark protocol evidence coverage is complete." + suffix})
        else:
            missing = list(protocol_coverage.get("missing_required", ())) + list(protocol_coverage.get("missing_raw_claim", ()))
            suffix = "" if not missing else f" Missing: {', '.join(str(value) for value in missing)}."
            decisions.append({"status": "not_supported", "claim": "Benchmark protocol coverage is incomplete; do not make a broad HumanISP or RAW/sensor-native superiority claim." + suffix})
    return decisions


def _baseline_claim_name(input_name: str) -> str:
    if input_name == "human_rgb":
        return "HumanISP"
    if input_name == "perception_fusion_rgb_aux":
        return "the RGB+Aux fusion baseline"
    if input_name:
        return f"`{input_name}`"
    return "the configured baseline"


def _training_status(
    best_recall: Mapping[str, Any] | None,
    lowest_fp: Mapping[str, Any] | None,
    max_precision: Mapping[str, Any] | None,
    train_runs: Sequence[Mapping[str, Any]],
    dense_eval_runs: Sequence[Mapping[str, Any]],
) -> str:
    if dense_eval_runs:
        precision = _metric(max_precision, "precision@0.50_mean")
        fp = _metric(lowest_fp, "fp@0.50_mean")
        if precision is not None and fp is not None and (precision < 0.05 or fp > 5.0):
            return "diagnostic_only"
        return "candidate_needs_gate"
    if train_runs:
        return "training_path_only"
    return "missing_training"


def _training_interpretation(status: str) -> str:
    if status == "diagnostic_only":
        return "Training is fast enough for iteration, but the compact dense detector currently has weak precision/FP behavior."
    if status == "candidate_needs_gate":
        return "Direct RGB+Aux DNN metrics should be promoted only after a held-out claim gate."
    if status == "training_path_only":
        return "Training summaries exist, but direct dense detector eval has not been rolled up."
    return "No RGB+Aux training evidence was available."


def _claim_text(profile: str, passed: bool) -> str:
    if profile == "broad_superiority":
        return "Broad HumanISP superiority gate passed." if passed else "Broad HumanISP superiority gate failed."
    if profile == "fp_reducer":
        return "Recall-budgeted FP-reduction gate passed." if passed else "Recall-budgeted FP-reduction gate failed."
    return "Custom claim gate passed." if passed else "Custom claim gate failed."


def _split_named_path(value: str | Path) -> Tuple[str, Path]:
    text = str(value)
    if "=" in text:
        name, raw_path = text.split("=", 1)
        return name.strip(), Path(raw_path).expanduser()
    return "", Path(text).expanduser()


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _sibling_html(summary_path: Path) -> str | None:
    html_path = summary_path.with_name("index.html")
    return str(html_path) if html_path.exists() else None


def _default_name(summary_path: Path) -> str:
    return summary_path.parent.name


def _max_by(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any] | None:
    values = [row for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values, key=lambda row: float(row.get(key, 0.0)))


def _max_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    values = [row for row in rows if _metric(row, metric) is not None]
    if not values:
        return None
    return max(values, key=lambda row: float(_metric(row, metric) or 0.0))


def _min_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    values = [row for row in rows if _metric(row, metric) is not None]
    if not values:
        return None
    return min(values, key=lambda row: float(_metric(row, metric) or 0.0))


def _metric(row: Mapping[str, Any] | None, metric: str) -> float | None:
    if not row:
        return None
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping) or metrics.get(metric) is None:
        return None
    return float(metrics.get(metric))


def _training_run_brief(row: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not row:
        return None
    return {
        "name": row.get("name"),
        "channel_mode": row.get("channel_mode"),
        "sample_count": row.get("sample_count"),
        "epochs": row.get("epochs"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "throughput": row.get("throughput"),
    }


def _dense_eval_brief(row: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not row:
        return None
    return {
        "name": row.get("name"),
        "channel_mode": row.get("channel_mode"),
        "sample_count": row.get("sample_count"),
        "precision@0.50_mean": _metric(row, "precision@0.50_mean"),
        "recall@0.50_mean": _metric(row, "recall@0.50_mean"),
        "fp@0.50_mean": _metric(row, "fp@0.50_mean"),
        "det_count_mean": _metric(row, "det_count_mean"),
    }


def _render_html(dashboard: Mapping[str, Any], destination: Path) -> str:
    decision_rows = "".join(_decision_row(item) for item in dashboard.get("decisions", ()))
    claim_rows = "".join(_claim_row(item, destination) for item in dashboard.get("claims", ()))
    training = dashboard.get("training")
    training_html = _training_html(training, destination) if isinstance(training, Mapping) else "<p>No RGB+Aux training rollup was provided.</p>"
    aux_contribution = dashboard.get("aux_contribution_audit")
    aux_contribution_html = _aux_contribution_html(aux_contribution, destination) if isinstance(aux_contribution, Mapping) else "<p>No aux contribution audit was provided.</p>"
    task_metrics = dashboard.get("task_metrics")
    task_metrics_html = _task_metrics_html(task_metrics, destination) if isinstance(task_metrics, Mapping) else "<p>No task metrics summary was provided.</p>"
    task_gate = dashboard.get("task_gate")
    task_gate_html = _task_gate_html(task_gate, destination) if isinstance(task_gate, Mapping) else "<p>No task gate summary was provided.</p>"
    mechanism = dashboard.get("mechanism_validation")
    mechanism_html = _mechanism_html(mechanism, destination) if isinstance(mechanism, Mapping) else "<p>No mechanism validation summary was provided.</p>"
    cfa_stress = dashboard.get("cfa_stress_sweep")
    cfa_stress_html = _cfa_stress_html(cfa_stress, destination) if isinstance(cfa_stress, Mapping) else "<p>No CFA stress sweep summary was provided.</p>"
    edge_confidence = dashboard.get("edge_confidence_suite")
    edge_confidence_html = _edge_confidence_html(edge_confidence, destination) if isinstance(edge_confidence, Mapping) else "<p>No edge-confidence suite summary was provided.</p>"
    edge_fidelity = dashboard.get("edge_fidelity_suite")
    edge_fidelity_html = _edge_fidelity_html(edge_fidelity, destination) if isinstance(edge_fidelity, Mapping) else "<p>No object edge-fidelity suite summary was provided.</p>"
    scene_edge = dashboard.get("scene_edge_confidence")
    scene_edge_html = _scene_edge_html(scene_edge, destination) if isinstance(scene_edge, Mapping) else "<p>No scene edge-confidence summary was provided.</p>"
    scene_information = dashboard.get("scene_information_stress")
    scene_information_html = (
        _scene_information_html(scene_information, destination)
        if isinstance(scene_information, Mapping)
        else "<p>No scene-information stress summary was provided.</p>"
    )
    protocol = dashboard.get("protocol_coverage")
    protocol_html = _protocol_html(protocol, destination) if isinstance(protocol, Mapping) else "<p>No benchmark protocol coverage summary was provided.</p>"
    comparison_rows = "".join(_comparison_row(item, destination) for item in dashboard.get("comparison_rollups", ()))
    comparison_body = comparison_rows if comparison_rows else '<tr><td colspan="4">No comparison rollups were provided.</td></tr>'
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Claim Readiness Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .supported {{ color: #047857; font-weight: 700; }}
    .not_supported {{ color: #b91c1c; font-weight: 700; }}
    .needs_gate, .needs_eval {{ color: #a16207; font-weight: 700; }}
    .diagnostic {{ color: #2563eb; font-weight: 700; }}
    .good_delta {{ color: #047857; }}
    .bad_delta {{ color: #b91c1c; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Claim Readiness Dashboard</h1>
  <div class=\"note\">{html_lib.escape(str(dashboard.get('interpretation', '')))}</div>
  <h2>Decision Summary</h2>
  <table>
    <thead><tr><th>Status</th><th>Claim</th></tr></thead>
    <tbody>{decision_rows}</tbody>
  </table>
  <h2>Claim Gates</h2>
  <table>
    <thead><tr><th>Name</th><th>Report</th><th>Claim</th><th>Profile</th><th>Verdict</th><th>Baseline</th><th>Target</th><th>Samples</th><th>P50 d/CI</th><th>R50 d/CI</th><th>Small R50 d/CI</th><th>FP d/CI</th><th>Failed</th></tr></thead>
    <tbody>{claim_rows}</tbody>
  </table>
  <h2>RGB+Aux DNN Training</h2>
  {training_html}
  <h2>Aux Contribution Audit</h2>
  {aux_contribution_html}
  <h2>Task Metrics</h2>
  {task_metrics_html}
  <h2>Task Gate</h2>
  {task_gate_html}
  <h2>Mechanism Validation</h2>
  {mechanism_html}
  <h2>CFA Stress Sweep</h2>
  {cfa_stress_html}
  <h2>Edge Confidence Suite</h2>
  {edge_confidence_html}
  <h2>Object Edge Fidelity</h2>
  {edge_fidelity_html}
  <h2>Scene Edge Confidence</h2>
  {scene_edge_html}
  <h2>Scene Information Stress</h2>
  {scene_information_html}
  <h2>Benchmark Protocol Coverage</h2>
  {protocol_html}
  <h2>Supporting Rollups</h2>
  <table>
    <thead><tr><th>Name</th><th>Report</th><th>Runs</th><th>Baseline</th></tr></thead>
    <tbody>{comparison_body}</tbody>
  </table>
  <p>Raw JSON: <code>claim_dashboard_summary.json</code></p>
</body>
</html>
"""


def _decision_row(item: Mapping[str, Any]) -> str:
    status = str(item.get("status", ""))
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td>{html_lib.escape(str(item.get('claim', '')))}</td>"
        "</tr>"
    )


def _claim_row(item: Mapping[str, Any], destination: Path) -> str:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), Mapping) else {}
    failed = ", ".join(str(value) for value in item.get("failed_metrics", ()))
    verdict_class = "supported" if bool(item.get("pass")) else "not_supported"
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(item.get('name', '')))}</td>"
        f"<td>{_report_link(item, destination)}</td>"
        f"<td>{html_lib.escape(str(item.get('claim', '')))}</td>"
        f"<td>{html_lib.escape(str(item.get('profile', '')))}</td>"
        f"<td class=\"{verdict_class}\">{html_lib.escape(str(item.get('verdict', '')))}</td>"
        f"<td><code>{html_lib.escape(str(item.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(item.get('target_input', '')))}</code></td>"
        f"<td>{int(item.get('sample_count', 0))}</td>"
        f"<td>{_metric_cell(metrics.get('precision@0.50_mean'))}</td>"
        f"<td>{_metric_cell(metrics.get('recall@0.50_mean'))}</td>"
        f"<td>{_metric_cell(metrics.get('small_recall@0.50_mean'))}</td>"
        f"<td>{_metric_cell(metrics.get('fp@0.50_mean'))}</td>"
        f"<td>{html_lib.escape(failed or 'none')}</td>"
        "</tr>"
    )


def _training_html(training: Mapping[str, Any], destination: Path) -> str:
    best_rate = training.get("best_train_rate") if isinstance(training.get("best_train_rate"), Mapping) else {}
    best_recall = training.get("best_recall_eval") if isinstance(training.get("best_recall_eval"), Mapping) else {}
    lowest_fp = training.get("lowest_fp_eval") if isinstance(training.get("lowest_fp_eval"), Mapping) else {}
    return (
        f"<p>Status: <code>{html_lib.escape(str(training.get('status', '')))}</code>. {html_lib.escape(str(training.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Runs</th><th>Best Train Rate</th><th>Best R50 Eval</th><th>Lowest FP Eval</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(training, destination)}</td>"
        f"<td>{int(training.get('run_count', 0))}</td>"
        f"<td>{html_lib.escape(str(best_rate.get('name', '')))}<br>{_fmt(best_rate.get('throughput'))} sample-epochs/s</td>"
        f"<td>{html_lib.escape(str(best_recall.get('name', '')))}<br>R50 {_fmt(best_recall.get('recall@0.50_mean'))}, P50 {_fmt(best_recall.get('precision@0.50_mean'))}, FP {_fmt(best_recall.get('fp@0.50_mean'))}</td>"
        f"<td>{html_lib.escape(str(lowest_fp.get('name', '')))}<br>FP {_fmt(lowest_fp.get('fp@0.50_mean'))}, R50 {_fmt(lowest_fp.get('recall@0.50_mean'))}</td>"
        "</tr></tbody></table>"
    )


def _aux_contribution_html(aux_contribution: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(aux_contribution.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in aux_contribution.get("failed_checks", ())) or "none"
    rows = "".join(_aux_contribution_row(row) for row in aux_contribution.get("comparisons", ()))
    if not rows:
        rows = '<tr><td colspan="6">No aux contribution comparisons were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(aux_contribution.get('status', '')))}</code>. "
        f"{html_lib.escape(str(aux_contribution.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Checks</th><th>Failed</th><th>Aux Feature Count</th><th>Aux Features</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(aux_contribution, destination)}</td>"
        f"<td>{int(aux_contribution.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        f"<td>{int(aux_contribution.get('aux_feature_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in aux_contribution.get('aux_features', ())) or 'none')}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Comparison</th><th>Target</th><th>Baseline</th><th>dP@0.50</th><th>dR@0.50</th><th>dFP@0.50</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _aux_contribution_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('target_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('baseline_input', '')))}</code></td>"
        f"<td>{_fmt(row.get('delta_precision@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_fp@0.50'), signed=True)}</td>"
        "</tr>"
    )


def _task_metrics_html(task_metrics: Mapping[str, Any], destination: Path) -> str:
    rows = "".join(_task_metric_row(row) for row in task_metrics.get("rows", ()))
    if not rows:
        rows = '<tr><td colspan="11">No task group rows were available.</td></tr>'
    status = str(task_metrics.get("status", ""))
    status_class = "not_supported" if status == "recall_tradeoff" else "needs_gate" if status == "candidate_needs_gate" else "needs_eval"
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(status)}</code>. {html_lib.escape(str(task_metrics.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Baseline</th><th>Target</th><th>Groups</th><th>Label Mode</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(task_metrics, destination)}</td>"
        f"<td><code>{html_lib.escape(str(task_metrics.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(task_metrics.get('target_input', '')))}</code></td>"
        f"<td>{int(task_metrics.get('group_count', 0))}</td>"
        f"<td>{'label agnostic' if bool(task_metrics.get('label_agnostic', True)) else 'label aware'}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Group</th><th>GT</th><th>Det</th><th>P@0.50</th><th>R@0.50</th><th>R@0.75</th><th>FP/sample</th><th>dP@0.50</th><th>dR@0.50</th><th>dR@0.75</th><th>dFP/sample</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _task_metric_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('group', '')))}</td>"
        f"<td>{int(row.get('gt_count', 0))}</td>"
        f"<td>{int(row.get('det_count', 0))}</td>"
        f"<td>{_fmt(row.get('precision@0.50'))}</td>"
        f"<td>{_fmt(row.get('recall@0.50'))}</td>"
        f"<td>{_fmt(row.get('recall@0.75'))}</td>"
        f"<td>{_fmt(row.get('fp@0.50_per_sample'))}</td>"
        f"<td>{_task_delta_cell(row.get('delta_precision@0.50'), lower_is_better=False)}</td>"
        f"<td>{_task_delta_cell(row.get('delta_recall@0.50'), lower_is_better=False)}</td>"
        f"<td>{_task_delta_cell(row.get('delta_recall@0.75'), lower_is_better=False)}</td>"
        f"<td>{_task_delta_cell(row.get('delta_fp@0.50_per_sample'), lower_is_better=True)}</td>"
        "</tr>"
    )


def _task_gate_html(task_gate: Mapping[str, Any], destination: Path) -> str:
    verdict_class = "supported" if bool(task_gate.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in task_gate.get("failed_groups", ())) or "none"
    skipped = ", ".join(str(value) for value in task_gate.get("skipped_groups", ())) or "none"
    return (
        f"<p>Verdict: <code class=\"{verdict_class}\">{html_lib.escape(str(task_gate.get('verdict', '')))}</code>. "
        f"{html_lib.escape(str(task_gate.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Profile</th><th>Baseline</th><th>Target</th><th>Evaluated</th><th>Failed</th><th>Skipped</th><th>Failed Groups</th><th>Skipped Groups</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(task_gate, destination)}</td>"
        f"<td><code>{html_lib.escape(str(task_gate.get('profile', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(task_gate.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(task_gate.get('target_input', '')))}</code></td>"
        f"<td>{int(task_gate.get('evaluated_group_count', 0))}</td>"
        f"<td>{int(task_gate.get('failed_group_count', 0))}</td>"
        f"<td>{int(task_gate.get('skipped_group_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        f"<td>{html_lib.escape(skipped)}</td>"
        "</tr></tbody></table>"
    )


def _mechanism_html(mechanism: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(mechanism.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in mechanism.get("failed_mechanisms", ())) or "none"
    cfas = ", ".join(str(value) for value in mechanism.get("cfa_patterns", ())) or "none"
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(mechanism.get('status', '')))}</code>. "
        f"{html_lib.escape(str(mechanism.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Mechanisms</th><th>CFA Patterns</th><th>Failed</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(mechanism, destination)}</td>"
        f"<td>{int(mechanism.get('mechanism_count', 0))}</td>"
        f"<td>{html_lib.escape(cfas)}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        "</tr></tbody></table>"
    )


def _cfa_stress_html(cfa_stress: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(cfa_stress.get("pass")) else "not_supported"
    cfas = ", ".join(str(value) for value in cfa_stress.get("cfa_patterns", ())) or "none"
    rows = "".join(_cfa_stress_row(row) for row in cfa_stress.get("top_rows", ()))
    if not rows:
        rows = '<tr><td colspan="4">No condition rankings were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(cfa_stress.get('status', '')))}</code>. "
        f"{html_lib.escape(str(cfa_stress.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Cases</th><th>Conditions</th><th>CFA Patterns</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(cfa_stress, destination)}</td>"
        f"<td>{int(cfa_stress.get('case_count', 0))}</td>"
        f"<td>{int(cfa_stress.get('condition_count', 0))}</td>"
        f"<td>{html_lib.escape(cfas)}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Condition</th><th>Top CFA</th><th>Score</th><th>Score Definition</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _cfa_stress_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('condition', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(row.get('condition_score'))}</td>"
        f"<td>{html_lib.escape(str(row.get('score_definition', '')))}</td>"
        "</tr>"
    )


def _edge_confidence_html(edge_confidence: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(edge_confidence.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in edge_confidence.get("failed_checks", ())) or "none"
    rows = "".join(_edge_confidence_delta_row(row) for row in edge_confidence.get("key_deltas", ()))
    if not rows:
        rows = '<tr><td colspan="5">No edge-confidence deltas were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(edge_confidence.get('status', '')))}</code>. "
        f"{html_lib.escape(str(edge_confidence.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Cases</th><th>Checks</th><th>Failed</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(edge_confidence, destination)}</td>"
        f"<td>{int(edge_confidence.get('case_count', 0))}</td>"
        f"<td>{int(edge_confidence.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Check</th><th>Metric</th><th>Delta</th><th>Threshold</th><th>Pass</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _edge_confidence_delta_row(row: Mapping[str, Any]) -> str:
    status_class = "supported" if bool(row.get("pass")) else "not_supported"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('check', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('metric', '')))}</code></td>"
        f"<td>{_fmt(row.get('delta'))}</td>"
        f"<td>{_fmt(row.get('threshold'))}</td>"
        f"<td class=\"{status_class}\">{html_lib.escape(str(bool(row.get('pass'))))}</td>"
        "</tr>"
    )


def _edge_fidelity_html(edge_fidelity: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(edge_fidelity.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in edge_fidelity.get("failed_checks", ())) or "none"
    ranking_rows = "".join(_edge_fidelity_ranking_row(row) for row in edge_fidelity.get("top_rows", ()))
    if not ranking_rows:
        ranking_rows = '<tr><td colspan="5">No object edge-fidelity rankings were available.</td></tr>'
    case_rows = "".join(_edge_fidelity_case_row(row) for row in edge_fidelity.get("cases", ()))
    if not case_rows:
        case_rows = '<tr><td colspan="6">No object edge-fidelity case rows were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(edge_fidelity.get('status', '')))}</code>. "
        f"{html_lib.escape(str(edge_fidelity.get('interpretation', '')))} "
        f"{html_lib.escape(str(edge_fidelity.get('claim_boundary', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Cases</th><th>Checks</th><th>CFA Patterns</th><th>LensPSF Sigma</th><th>Failed</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(edge_fidelity, destination)}</td>"
        f"<td>{int(edge_fidelity.get('case_count', 0))}</td>"
        f"<td>{int(edge_fidelity.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in edge_fidelity.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(_fmt(value) for value in edge_fidelity.get('psf_sigmas', ()) if value is not None) or 'none')}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>LensPSF Sigma</th><th>Top CFA</th><th>Aux Obj F1</th><th>Perception Obj F1</th><th>Conf Sep</th></tr></thead>"
        f"<tbody>{ranking_rows}</tbody></table>"
        "<table>"
        "<thead><tr><th>Case</th><th>CFA</th><th>LensPSF</th><th>Human Obj F1</th><th>Perception Obj F1</th><th>Aux Obj F1</th></tr></thead>"
        f"<tbody>{case_rows}</tbody></table>"
    )


def _edge_fidelity_ranking_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(row.get('aux_object_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_object_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('edge_confidence_separation'), signed=True)}</td>"
        "</tr>"
    )


def _edge_fidelity_case_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{_fmt(row.get('human_object_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_object_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('aux_object_edge_f1'))}</td>"
        "</tr>"
    )


def _scene_edge_html(scene_edge: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(scene_edge.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in scene_edge.get("failed_checks", ())) or "none"
    case_rows = "".join(_scene_edge_case_row(row) for row in scene_edge.get("cases", ()))
    if not case_rows:
        case_rows = '<tr><td colspan="7">No scene edge-confidence case rows were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(scene_edge.get('status', '')))}</code>. "
        f"{html_lib.escape(str(scene_edge.get('interpretation', '')))} "
        f"{html_lib.escape(str(scene_edge.get('claim_boundary', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Cases</th><th>Checks</th><th>Failed</th><th>Human F1</th><th>Perception RGB F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(scene_edge, destination)}</td>"
        f"<td>{int(scene_edge.get('case_count', 0))}</td>"
        f"<td>{int(scene_edge.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        f"<td>{_fmt(scene_edge.get('human_rgb_proxy_source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_rgb_proxy_source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_aux_strength_source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_aux_confidence_source_edge_f1_mean'))}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Case</th><th>Source</th><th>CFA</th><th>Human F1</th><th>Perception RGB F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th></tr></thead>"
        f"<tbody>{case_rows}</tbody></table>"
    )


def _scene_edge_case_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('source', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(row.get('human_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_aux_strength_source_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_aux_confidence_source_edge_f1'))}</td>"
        "</tr>"
    )


def _scene_information_html(scene_information: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(scene_information.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in scene_information.get("failed_checks", ())) or "none"
    rows = "".join(_scene_information_case_row(row) for row in scene_information.get("cases", ()))
    if not rows:
        rows = '<tr><td colspan="8">No scene-information cases were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(scene_information.get('status', '')))}</code>. "
        f"{html_lib.escape(str(scene_information.get('interpretation', '')))} "
        f"{html_lib.escape(str(scene_information.get('claim_boundary', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Scene</th><th>Sensor</th><th>CFA</th><th>Cases</th><th>Checks</th><th>Failed</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(scene_information, destination)}</td>"
        f"<td>{int(scene_information.get('scene_width', 0))} x {int(scene_information.get('scene_height', 0))}</td>"
        f"<td>{int(scene_information.get('sensor_width', 0))} x {int(scene_information.get('sensor_height', 0))}</td>"
        f"<td><code>{html_lib.escape(str(scene_information.get('cfa_pattern', '')))}</code></td>"
        f"<td>{int(scene_information.get('case_count', 0))}</td>"
        f"<td>{int(scene_information.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Case</th><th>Sample</th><th>Scene Luma P90</th><th>Sensor Luma P90</th><th>Retention</th><th>Scene Chroma P90</th><th>Color Conf</th><th>Signal Retention</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _scene_information_case_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('sample_mode', '')))}</code></td>"
        f"<td>{_fmt(row.get('scene_luma_gradient_p90'))}</td>"
        f"<td>{_fmt(row.get('sensor_luma_gradient_p90'))}</td>"
        f"<td>{_fmt(row.get('luma_detail_retention_p90'))}</td>"
        f"<td>{_fmt(row.get('scene_chroma_gradient_p90'))}</td>"
        f"<td>{_fmt(row.get('color_confidence_mean'))}</td>"
        f"<td>{_fmt(row.get('signal_contrast_retention'))}</td>"
        "</tr>"
    )


def _protocol_html(protocol: Mapping[str, Any], destination: Path) -> str:
    rows = "".join(_protocol_row(row) for row in protocol.get("requirements", ()))
    if not rows:
        rows = '<tr><td colspan="5">No protocol coverage rows were available.</td></tr>'
    status = str(protocol.get("status", ""))
    coverage_status = str(protocol.get("coverage_status") or ("coverage_complete" if status == "claim_ready" else "coverage_incomplete"))
    metric_status = str(protocol.get("metric_claim_status", "unknown"))
    status_class = "supported" if coverage_status == "coverage_complete" else "not_supported"
    return (
        f"<p>Coverage status: <code class=\"{status_class}\">{html_lib.escape(coverage_status)}</code>. "
        f"Metric claim status: <code>{html_lib.escape(metric_status)}</code>. "
        f"Legacy status: <code>{html_lib.escape(status)}</code>. "
        f"{html_lib.escape(str(protocol.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Missing Required</th><th>Missing RAW Claim</th><th>Missing Rows</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(protocol, destination)}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in protocol.get('missing_required', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in protocol.get('missing_raw_claim', ())) or 'none')}</td>"
        f"<td>{int(protocol.get('missing_count', 0))}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Status</th><th>Scope</th><th>Requirement</th><th>Evidence</th><th>Missing Reason</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _protocol_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    status_class = "supported" if status == "covered" else "not_supported"
    return (
        "<tr>"
        f"<td class=\"{status_class}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('scope', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('label', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('missing_reason', '')))}</td>"
        "</tr>"
    )


def _comparison_row(item: Mapping[str, Any], destination: Path) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(item.get('name', '')))}</td>"
        f"<td>{_report_link(item, destination)}</td>"
        f"<td>{int(item.get('run_count', 0))}</td>"
        f"<td><code>{html_lib.escape(str(item.get('baseline_input', '')))}</code></td>"
        "</tr>"
    )


def _metric_cell(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    ci_low = item.get("ci_low")
    ci_high = item.get("ci_high")
    ci_text = "" if ci_low is None or ci_high is None else f"<br>CI [{_fmt(ci_low, signed=True)}, {_fmt(ci_high, signed=True)}]"
    status = "supported" if bool(item.get("pass")) else "not_supported"
    return f"<span class=\"{status}\">{_fmt(item.get('delta'), signed=True)}</span>{ci_text}"


def _task_delta_cell(value: Any, *, lower_is_better: bool) -> str:
    if value is None:
        return ""
    number = float(value)
    good = number < 0.0 if lower_is_better else number >= 0.0
    css_class = "good_delta" if good else "bad_delta"
    return f"<span class=\"{css_class}\">{_fmt(number, signed=True)}</span>"


def _report_link(item: Mapping[str, Any], destination: Path) -> str:
    html_path = item.get("html_path")
    if not html_path:
        return ""
    relative = os.path.relpath(str(html_path), start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _signed_metric(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    return None if value is None else float(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
