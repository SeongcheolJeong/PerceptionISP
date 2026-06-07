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
CFA_LENSPSF_DETECTOR_SWEEP_SUMMARY = "cfa_lenspsf_detector_sweep_summary.json"
CFA_LENSPSF_PROPOSAL_AUDIT_SUMMARY = "cfa_lenspsf_proposal_audit_summary.json"
CFA_LENSPSF_NATIVE_AUDIT_SUMMARY = "cfa_lenspsf_native_audit_summary.json"
CASEBOOK_SUMMARY = "casebook_summary.json"


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
    parser.add_argument("--scene-edge-confidence", action="append", default=[], help="Scene-edge confidence summary path/dir. Repeatable.")
    parser.add_argument("--scene-information-stress", default=None, help="Scene-information stress summary path/dir.")
    parser.add_argument("--aux-contribution-audit", default=None, help="Aux contribution audit summary path/dir.")
    parser.add_argument("--cfa-lenspsf-detector-sweep", default=None, help="CFA/LensPSF detector sweep summary path/dir.")
    parser.add_argument("--cfa-lenspsf-proposal-audit", default=None, help="CFA/LensPSF proposal-edge audit summary path/dir.")
    parser.add_argument("--cfa-lenspsf-native-audit", default=None, help="CFA/LensPSF native-CFA separation audit summary path/dir.")
    parser.add_argument("--casebook", default=None, help="Success/failure casebook summary path/dir.")
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
        cfa_lenspsf_detector_sweep=args.cfa_lenspsf_detector_sweep,
        cfa_lenspsf_proposal_audit=args.cfa_lenspsf_proposal_audit,
        cfa_lenspsf_native_audit=args.cfa_lenspsf_native_audit,
        casebook=args.casebook,
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
    scene_edge_confidence: str | Path | Sequence[str | Path] | None = None,
    scene_information_stress: str | Path | None = None,
    aux_contribution_audit: str | Path | None = None,
    cfa_lenspsf_detector_sweep: str | Path | None = None,
    cfa_lenspsf_proposal_audit: str | Path | None = None,
    cfa_lenspsf_native_audit: str | Path | None = None,
    casebook: str | Path | None = None,
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
    scene_edge = _load_scene_edge_confidence(scene_edge_confidence) if _as_path_specs(scene_edge_confidence) else None
    scene_information = _load_scene_information_stress(scene_information_stress) if scene_information_stress is not None else None
    aux_contribution = _load_aux_contribution_audit(aux_contribution_audit) if aux_contribution_audit is not None else None
    cfa_lenspsf_detector = (
        _load_cfa_lenspsf_detector_sweep(cfa_lenspsf_detector_sweep)
        if cfa_lenspsf_detector_sweep is not None
        else None
    )
    cfa_lenspsf_proposal = (
        _load_cfa_lenspsf_proposal_audit(cfa_lenspsf_proposal_audit)
        if cfa_lenspsf_proposal_audit is not None
        else None
    )
    cfa_lenspsf_native = (
        _load_cfa_lenspsf_native_audit(cfa_lenspsf_native_audit)
        if cfa_lenspsf_native_audit is not None
        else None
    )
    casebook_data = _load_casebook(casebook) if casebook is not None else None
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
        cfa_lenspsf_detector,
        cfa_lenspsf_proposal,
        cfa_lenspsf_native,
        casebook_data,
    )
    evidence_map = _build_evidence_map(
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
        cfa_lenspsf_detector,
        cfa_lenspsf_proposal,
        cfa_lenspsf_native,
        casebook_data,
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
        "cfa_lenspsf_detector_sweep": cfa_lenspsf_detector,
        "cfa_lenspsf_proposal_audit": cfa_lenspsf_proposal,
        "cfa_lenspsf_native_audit": cfa_lenspsf_native,
        "casebook": casebook_data,
        "comparison_rollups": comparison_rollups,
        "decisions": decisions,
        "evidence_map": evidence_map,
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


def _load_scene_edge_confidence(spec: str | Path | Sequence[str | Path]) -> Dict[str, Any]:
    specs = _as_path_specs(spec)
    reports = [_load_scene_edge_confidence_one(item) for item in specs]
    if len(reports) == 1:
        return reports[0]
    case_count = sum(int(report.get("case_count", 0)) for report in reports)
    check_count = sum(int(report.get("check_count", 0)) for report in reports)
    failed = [str(value) for report in reports for value in report.get("failed_checks", ())]
    cfa_patterns = sorted({str(value) for report in reports for value in report.get("cfa_patterns", ())})
    psf_sigmas = sorted({float(value) for report in reports for value in report.get("psf_sigmas", ()) if value is not None})
    cases = [case for report in reports for case in report.get("cases", ())]
    first = reports[0]
    return {
        "name": "Scene edge confidence evidence",
        "summary_path": first.get("summary_path"),
        "html_path": first.get("html_path"),
        "status": "pass" if all(bool(report.get("pass")) for report in reports) else "fail",
        "pass": all(bool(report.get("pass")) for report in reports),
        "report_count": len(reports),
        "check_count": check_count,
        "case_count": case_count,
        "failed_checks": failed,
        "cfa_patterns": cfa_patterns,
        "psf_sigmas": psf_sigmas,
        "human_rgb_proxy_source_edge_f1_mean": _weighted_report_mean(reports, "human_rgb_proxy_source_edge_f1_mean"),
        "perception_rgb_proxy_source_edge_f1_mean": _weighted_report_mean(reports, "perception_rgb_proxy_source_edge_f1_mean"),
        "perception_aux_strength_source_edge_f1_mean": _weighted_report_mean(reports, "perception_aux_strength_source_edge_f1_mean"),
        "perception_aux_confidence_source_edge_f1_mean": _weighted_report_mean(reports, "perception_aux_confidence_source_edge_f1_mean"),
        "perception_rgb_minus_human_source_edge_f1_mean": _weighted_report_mean(reports, "perception_rgb_minus_human_source_edge_f1_mean"),
        "perception_aux_strength_minus_human_source_edge_f1_mean": _weighted_report_mean(reports, "perception_aux_strength_minus_human_source_edge_f1_mean"),
        "perception_rgb_source_edge_f1_win_rate": _weighted_report_mean(reports, "perception_rgb_source_edge_f1_win_rate"),
        "perception_aux_strength_source_edge_f1_win_rate": _weighted_report_mean(reports, "perception_aux_strength_source_edge_f1_win_rate"),
        "cases": cases[:12],
        "reports": reports,
        "interpretation": "Multiple scene edge-confidence summaries are aggregated here.",
        "claim_boundary": "This is front-end scene-edge confidence evidence. It is not object-boundary ground truth and it is not detector-performance evidence.",
    }


def _load_scene_edge_confidence_one(spec: str | Path) -> Dict[str, Any]:
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
                "perception_rgb_minus_human_source_edge_f1": _maybe_float(metrics.get("perception_rgb_minus_human_source_edge_f1")),
                "perception_aux_strength_source_edge_f1": _maybe_float(metrics.get("perception_aux_strength_source_edge_f1")),
                "perception_aux_strength_minus_human_source_edge_f1": _maybe_float(metrics.get("perception_aux_strength_minus_human_source_edge_f1")),
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
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_maybe_float(value) for value in data.get("psf_sigmas", ())],
        "human_rgb_proxy_source_edge_f1_mean": _maybe_float(aggregate.get("human_rgb_proxy_source_edge_f1_mean")),
        "perception_rgb_proxy_source_edge_f1_mean": _maybe_float(aggregate.get("perception_rgb_proxy_source_edge_f1_mean")),
        "perception_aux_strength_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_strength_source_edge_f1_mean")),
        "perception_aux_confidence_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_confidence_source_edge_f1_mean")),
        "perception_rgb_minus_human_source_edge_f1_mean": _maybe_float(aggregate.get("perception_rgb_minus_human_source_edge_f1_mean")),
        "perception_aux_strength_minus_human_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_strength_minus_human_source_edge_f1_mean")),
        "perception_rgb_source_edge_f1_win_rate": _maybe_float(aggregate.get("perception_rgb_source_edge_f1_win_rate")),
        "perception_aux_strength_source_edge_f1_win_rate": _maybe_float(aggregate.get("perception_aux_strength_source_edge_f1_win_rate")),
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
    sample_bridge = _load_aux_sample_bridge(data.get("sample_bridge"))
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
        "sample_bridge": sample_bridge,
        "interpretation": str(data.get("interpretation", "")),
    }


def _load_cfa_lenspsf_detector_sweep(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CFA_LENSPSF_DETECTOR_SWEEP_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    runs = []
    for row in data.get("runs", ()):
        if not isinstance(row, Mapping):
            continue
        raw_summary = row.get("raw_condition_summary", {}) if isinstance(row.get("raw_condition_summary"), Mapping) else {}
        primary = _dashboard_primary_downstream_input(row)
        primary_metrics = row.get("metrics", {}).get(primary, {}) if isinstance(row.get("metrics"), Mapping) else {}
        primary_delta = row.get("delta_vs_human", {}).get(primary, {}) if isinstance(row.get("delta_vs_human"), Mapping) else {}
        row_report = str(row.get("report", ""))
        runs.append(
            {
                "run_id": str(row.get("run_id", "")),
                "report": row_report,
                "html_path": str(summary_path.parent / row_report) if row_report else "",
                "cfa_pattern": str(row.get("cfa_pattern", "")),
                "psf_sigma": _maybe_float(row.get("psf_sigma")),
                "sample_count": int(row.get("sample_count", 0)),
                "primary_input": primary,
                "precision@0.50_mean": _maybe_float(primary_metrics.get("precision@0.50_mean")),
                "recall@0.50_mean": _maybe_float(primary_metrics.get("recall@0.50_mean")),
                "fp@0.50_mean": _maybe_float(primary_metrics.get("fp@0.50_mean")),
                "delta_precision@0.50_mean": _maybe_float(primary_delta.get("precision@0.50_mean")),
                "delta_recall@0.50_mean": _maybe_float(primary_delta.get("recall@0.50_mean")),
                "delta_fp@0.50_mean": _maybe_float(primary_delta.get("fp@0.50_mean")),
                "pattern_remapped_fraction": _maybe_float(raw_summary.get("pattern_remapped_fraction")),
                "true_sensor_cfa_mosaic_fraction": _maybe_float(raw_summary.get("true_sensor_cfa_mosaic_fraction")),
                "psf_recorded_fraction": _maybe_float(raw_summary.get("psf_recorded_fraction")),
                "camerae2e_camera_types": list((raw_summary.get("camerae2e_camera_types", {}) if isinstance(raw_summary.get("camerae2e_camera_types"), Mapping) else {}).keys()),
                "camerae2e_native_cfa_bridge_versions": list((raw_summary.get("camerae2e_native_cfa_bridge_versions", {}) if isinstance(raw_summary.get("camerae2e_native_cfa_bridge_versions"), Mapping) else {}).keys()),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "run_count": int(data.get("run_count", len(runs))),
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "count": int(data.get("count", 0)),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_maybe_float(value) for value in data.get("psf_sigmas", ())],
        "use_camerae2e": bool(data.get("use_camerae2e", False)),
        "checks": checks,
        "failed_checks": failed,
        "rankings": data.get("rankings", {}) if isinstance(data.get("rankings"), Mapping) else {},
        "runs": runs,
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _load_cfa_lenspsf_proposal_audit(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CFA_LENSPSF_PROPOSAL_AUDIT_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    conditions = []
    for row in data.get("conditions", ()):
        if not isinstance(row, Mapping):
            continue
        conditions.append(
            {
                "run_id": str(row.get("run_id", "")),
                "report": str(row.get("report", "")),
                "cfa_pattern": str(row.get("cfa_pattern", "")),
                "psf_sigma": _maybe_float(row.get("psf_sigma")),
                "sample_count": int(row.get("sample_count", 0)),
                "removed_fp_count": int(row.get("removed_fp_count", 0)),
                "removed_tp_count": int(row.get("removed_tp_count", 0)),
                "fp_delta_count": int(row.get("fp_delta_count", 0)),
                "edge_auc_low_predicts_removed_fp": _maybe_float(row.get("edge_auc_low_predicts_removed_fp")),
                "scene_edge_auc_low_predicts_removed_fp": _maybe_float(row.get("scene_edge_auc_low_predicts_removed_fp")),
                "edge_support_delta_removed_fp_minus_kept_tp": _maybe_float(row.get("edge_support_delta_removed_fp_minus_kept_tp")),
                "scene_edge_support_delta_removed_fp_minus_kept_tp": _maybe_float(row.get("scene_edge_support_delta_removed_fp_minus_kept_tp")),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "condition_count": int(data.get("condition_count", len(conditions))),
        "expected_condition_count": int(data.get("expected_condition_count", 0)),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_maybe_float(value) for value in data.get("psf_sigmas", ())],
        "checks": checks,
        "failed_checks": failed,
        "aggregate": aggregate,
        "conditions": conditions[:16],
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _load_cfa_lenspsf_native_audit(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CFA_LENSPSF_NATIVE_AUDIT_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) not in {"pass", "warning"}]
    groups: Dict[str, Any] = {}
    for name, group in (data.get("groups", {}) if isinstance(data.get("groups"), Mapping) else {}).items():
        if not isinstance(group, Mapping):
            continue
        groups[str(name)] = {
            "run_count": int(group.get("run_count", 0)),
            "sample_count": int(group.get("sample_count", 0)),
            "cfa_patterns": [str(value) for value in group.get("cfa_patterns", ())],
            "psf_sigmas": [_maybe_float(value) for value in group.get("psf_sigmas", ())],
            "mean_delta_precision@0.50": _maybe_float(group.get("mean_delta_precision@0.50")),
            "mean_delta_recall@0.50": _maybe_float(group.get("mean_delta_recall@0.50")),
            "mean_delta_small_recall@0.50": _maybe_float(group.get("mean_delta_small_recall@0.50")),
            "mean_delta_fp@0.50": _maybe_float(group.get("mean_delta_fp@0.50")),
            "best_delta_fp@0.50": group.get("best_delta_fp@0.50", {}) if isinstance(group.get("best_delta_fp@0.50"), Mapping) else {},
            "best_delta_recall@0.50": group.get("best_delta_recall@0.50", {}) if isinstance(group.get("best_delta_recall@0.50"), Mapping) else {},
        }
    rows = []
    for row in data.get("runs", ())[:16]:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "run_id": str(row.get("run_id", "")),
                "html_path": str(row.get("html_path", "")),
                "native_status": str(row.get("native_status", "")),
                "cfa_pattern": str(row.get("cfa_pattern", "")),
                "psf_sigma": _maybe_float(row.get("psf_sigma")),
                "sample_count": int(row.get("sample_count", 0)),
                "pattern_remapped_count": int(row.get("pattern_remapped_count", 0)),
                "pattern_remapped_fraction": _maybe_float(row.get("pattern_remapped_fraction")),
                "source_cfa_patterns": row.get("source_cfa_patterns", {}) if isinstance(row.get("source_cfa_patterns"), Mapping) else {},
                "target_cfa_patterns": row.get("target_cfa_patterns", {}) if isinstance(row.get("target_cfa_patterns"), Mapping) else {},
                "delta_precision@0.50_mean": _maybe_float(row.get("delta_precision@0.50_mean")),
                "delta_recall@0.50_mean": _maybe_float(row.get("delta_recall@0.50_mean")),
                "delta_small_recall@0.50_mean": _maybe_float(row.get("delta_small_recall@0.50_mean")),
                "delta_fp@0.50_mean": _maybe_float(row.get("delta_fp@0.50_mean")),
            }
        )
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "run_count": int(data.get("run_count", 0)),
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_maybe_float(value) for value in data.get("psf_sigmas", ())],
        "groups": groups,
        "runs": rows,
        "checks": checks,
        "failed_checks": failed,
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _load_casebook(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CASEBOOK_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    categories: Dict[str, Any] = {}
    for name, payload in (data.get("categories", {}) if isinstance(data.get("categories"), Mapping) else {}).items():
        if not isinstance(payload, Mapping):
            continue
        cases = []
        for row in payload.get("cases", ())[:8]:
            if not isinstance(row, Mapping):
                continue
            cases.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "category": str(row.get("category", "")),
                    "visual_path": str(row.get("visual_path", "")),
                    "tp_delta@0.50": int(row.get("tp_delta@0.50", 0)),
                    "fp_delta@0.50": int(row.get("fp_delta@0.50", 0)),
                    "baseline_tp@0.50": int(row.get("baseline_tp@0.50", 0)),
                    "baseline_fp@0.50": int(row.get("baseline_fp@0.50", 0)),
                    "target_tp@0.50": int(row.get("target_tp@0.50", 0)),
                    "target_fp@0.50": int(row.get("target_fp@0.50", 0)),
                    "cfa_pattern": str(row.get("cfa_pattern", "")),
                    "pattern_remapped": bool(row.get("pattern_remapped", False)),
                }
            )
        categories[str(name)] = {
            "case_count": int(payload.get("case_count", 0)),
            "selected_case_count": int(payload.get("selected_case_count", 0)),
            "cases": cases,
        }
    status = str(data.get("status", ""))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "status": status,
        "pass": status == "pass" and not failed,
        "sample_count": int(data.get("sample_count", 0)),
        "selected_case_count": int(data.get("selected_case_count", 0)),
        "baseline_input": str(data.get("baseline_input", "")),
        "target_input": str(data.get("target_input", "")),
        "aggregate": data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {},
        "checks": checks,
        "failed_checks": failed,
        "categories": categories,
        "interpretation": str(data.get("interpretation", "")),
        "claim_boundary": str(data.get("claim_boundary", "")),
    }


def _dashboard_primary_downstream_input(run: Mapping[str, Any]) -> str:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}
    for input_name in metrics:
        if str(input_name).startswith("perception_calibrated"):
            return str(input_name)
    if "perception_fusion_rgb_aux" in metrics:
        return "perception_fusion_rgb_aux"
    if "perception_rgb" in metrics:
        return "perception_rgb"
    return next(iter(metrics), "")


def _load_aux_sample_bridge(payload: Any) -> Dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    return {
        "status": str(payload.get("status", "")),
        "baseline_input": str(payload.get("baseline_input", "")),
        "target_input": str(payload.get("target_input", "")),
        "compared_sample_count": int(payload.get("compared_sample_count", 0)),
        "baseline_detection_count": int(payload.get("baseline_detection_count", 0)),
        "target_detection_count": int(payload.get("target_detection_count", 0)),
        "baseline_only_detection_count": int(payload.get("baseline_only_detection_count", 0)),
        "target_only_detection_count": int(payload.get("target_only_detection_count", 0)),
        "removed_fp_count": int(payload.get("removed_fp_count", 0)),
        "removed_tp_count": int(payload.get("removed_tp_count", 0)),
        "added_fp_count": int(payload.get("added_fp_count", 0)),
        "added_tp_count": int(payload.get("added_tp_count", 0)),
        "fp_delta_count": int(payload.get("fp_delta_count", 0)),
        "tp_delta_count": int(payload.get("tp_delta_count", 0)),
        "removed_fp_fraction": _maybe_float_or_none(payload.get("removed_fp_fraction")),
        "removed_fp_to_tp_ratio": _maybe_float_or_none(payload.get("removed_fp_to_tp_ratio")),
        "support_means": payload.get("support_means", {}) if isinstance(payload.get("support_means"), Mapping) else {},
        "support_deltas": payload.get("support_deltas", {}) if isinstance(payload.get("support_deltas"), Mapping) else {},
        "proposal_correlation": _load_proposal_correlation(payload.get("proposal_correlation")),
        "interpretation": str(payload.get("interpretation", "")),
    }


def _load_proposal_correlation(payload: Any) -> Dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    rows = []
    for row in payload.get("rows", ()):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "comparison": str(row.get("comparison", "")),
                "feature": str(row.get("feature", "")),
                "positive_status": str(row.get("positive_status", "")),
                "negative_status": str(row.get("negative_status", "")),
                "positive_count": int(row.get("positive_count", 0)),
                "negative_count": int(row.get("negative_count", 0)),
                "positive_mean": _maybe_float_or_none(row.get("positive_mean")),
                "negative_mean": _maybe_float_or_none(row.get("negative_mean")),
                "delta": _maybe_float_or_none(row.get("delta")),
                "point_biserial": _maybe_float_or_none(row.get("point_biserial")),
                "auc_low_feature_predicts_positive": _maybe_float_or_none(row.get("auc_low_feature_predicts_positive")),
                "lower_feature_predicts_positive": bool(row.get("lower_feature_predicts_positive")),
            }
        )
    return {
        "status": str(payload.get("status", "")),
        "baseline_proposal_count": int(payload.get("baseline_proposal_count", 0)),
        "rows": rows,
        "key_results": payload.get("key_results", {}) if isinstance(payload.get("key_results"), Mapping) else {},
        "interpretation": str(payload.get("interpretation", "")),
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


def _build_evidence_map(
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
    cfa_lenspsf_detector_sweep: Mapping[str, Any] | None,
    cfa_lenspsf_proposal_audit: Mapping[str, Any] | None,
    cfa_lenspsf_native_audit: Mapping[str, Any] | None,
    casebook: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    current: list[Dict[str, Any]] = []
    broad = _first_claim(claims, "broad_superiority")
    if broad is not None:
        current.append(
            {
                "area": "Broad HumanISP superiority",
                "status": "supported" if bool(broad.get("pass")) else "not_supported",
                "claim_strength": "claim_ready" if bool(broad.get("pass")) else "blocked_by_gate",
                "evidence": _claim_metric_evidence(broad),
                "claim_boundary": (
                    "Can be claimed only if precision, recall, small-object recall, FP, sample scale, and CI gates pass together."
                    if bool(broad.get("pass"))
                    else "Do not claim broad HumanISP superiority; at least one configured broad metric gate failed."
                ),
                "next_evidence": "Improve recall/small-object recall without losing the FP gain, then rerun the held-out broad claim gate.",
            }
        )

    fp_claim = _first_claim(claims, "fp_reducer")
    if fp_claim is not None:
        baseline = _baseline_claim_name(str(fp_claim.get("baseline_input", "")))
        current.append(
            {
                "area": "Recall-budgeted FP reduction",
                "status": "supported" if bool(fp_claim.get("pass")) else "not_supported",
                "claim_strength": "claim_ready" if bool(fp_claim.get("pass")) else "blocked_by_gate",
                "evidence": _claim_metric_evidence(fp_claim),
                "claim_boundary": f"Use this as a narrow FP-reduction claim versus {baseline}, not as a full detector superiority claim.",
                "next_evidence": "Add condition/class slices that preserve recall while showing the same FP reduction direction.",
            }
        )

    if task_gate is not None or task_metrics is not None:
        task_source = task_gate if task_gate is not None else task_metrics
        status = "supported" if task_gate is not None and bool(task_gate.get("pass")) else "not_supported"
        current.append(
            {
                "area": "Task-level recall claim",
                "status": status,
                "claim_strength": "claim_ready" if status == "supported" else "not_claim_ready",
                "evidence": _task_evidence(task_metrics, task_gate),
                "claim_boundary": "Task-level claims need the configured task gate to pass for priority groups.",
                "next_evidence": "Tune or train the perception path for VRU/person/small-object recall, then rerun the task gate.",
                "source": str(task_source.get("name", "")) if isinstance(task_source, Mapping) else "",
            }
        )

    if protocol_coverage is not None:
        coverage_status = str(protocol_coverage.get("coverage_status") or "")
        protocol_status = "supported" if coverage_status == "coverage_complete" else "not_supported"
        current.append(
            {
                "area": "Benchmark protocol coverage",
                "status": protocol_status,
                "claim_strength": str(protocol_coverage.get("metric_claim_status") or protocol_coverage.get("status", "")),
                "evidence": _protocol_evidence(protocol_coverage),
                "claim_boundary": "Coverage can be complete while the metric claim remains narrow; do not expand beyond the metric claim status.",
                "next_evidence": "Keep all protocol rows covered when adding new CFA/PSF or training evidence.",
            }
        )

    if mechanism_validation is not None:
        current.append(
            {
                "area": "Front-end mechanism validation",
                "status": "supported" if bool(mechanism_validation.get("pass")) else "not_supported",
                "claim_strength": "front_end_feasibility",
                "evidence": _mechanism_evidence(mechanism_validation),
                "claim_boundary": "Shows aux/confidence maps respond to controlled stressors; not detector-performance evidence.",
                "next_evidence": "Tie the same mechanisms to detector TP/FP changes on matched samples.",
            }
        )

    if cfa_stress_sweep is not None:
        current.append(
            {
                "area": "CFA-dependent front-end behavior",
                "status": "diagnostic" if bool(cfa_stress_sweep.get("pass")) else "not_supported",
                "claim_strength": "diagnostic",
                "evidence": _cfa_evidence(cfa_stress_sweep),
                "claim_boundary": "Use for CFA/condition sensitivity, not for downstream detector superiority.",
                "next_evidence": "Run the same CFA patterns through CameraE2E, HumanISP, PerceptionISP, and a fixed detector.",
            }
        )

    if edge_confidence_suite is not None:
        current.append(
            {
                "area": "Difficult-edge confidence response",
                "status": "diagnostic" if bool(edge_confidence_suite.get("pass")) else "not_supported",
                "claim_strength": "diagnostic",
                "evidence": _edge_confidence_evidence(edge_confidence_suite),
                "claim_boundary": "Shows confidence changes under low light, glare, and low MTF; not object detection evidence.",
                "next_evidence": "Measure whether low-confidence edge regions correlate with detector mistakes and aux-based suppression.",
            }
        )

    if edge_fidelity_suite is not None:
        current.append(
            {
                "area": "Object edge fidelity by CFA/LensPSF",
                "status": "diagnostic" if bool(edge_fidelity_suite.get("pass")) else "not_supported",
                "claim_strength": "diagnostic",
                "evidence": _edge_fidelity_evidence(edge_fidelity_suite),
                "claim_boundary": "Front-end edge fidelity evidence only; object-boundary fidelity is not detector accuracy.",
                "next_evidence": "Convert edge-fidelity deltas into detector confidence/FP/TP deltas per object and CFA.",
            }
        )

    if cfa_lenspsf_detector_sweep is not None:
        current.append(
            {
                "area": "CFA/LensPSF detector condition sweep",
                "status": "diagnostic" if bool(cfa_lenspsf_detector_sweep.get("pass")) else "not_supported",
                "claim_strength": "condition_detector_diagnostic",
                "evidence": _cfa_lenspsf_detector_evidence(cfa_lenspsf_detector_sweep),
                "claim_boundary": "Condition-level detector evidence only; do not treat remapped CFA rows as native sensor-CFA proof.",
                "next_evidence": "Scale the sweep to more held-out samples and add same-sample proposal edge correlation per CFA/LensPSF condition.",
            }
        )

    if cfa_lenspsf_proposal_audit is not None:
        current.append(
            {
                "area": "CFA/LensPSF proposal-edge bridge",
                "status": "diagnostic" if bool(cfa_lenspsf_proposal_audit.get("pass")) else "not_supported",
                "claim_strength": "condition_proposal_bridge",
                "evidence": _cfa_lenspsf_proposal_evidence(cfa_lenspsf_proposal_audit),
                "claim_boundary": "Condition-level calibrated proposal evidence; not an incremental aux-only ablation, trained DNN result, or native CFA proof.",
                "next_evidence": "Scale the same proposal bridge to larger held-out condition sweeps and separate native-CFA simulation from bridge remap sensitivity.",
            }
        )

    if cfa_lenspsf_native_audit is not None:
        current.append(
            {
                "area": "CFA/LensPSF native-CFA separation",
                "status": "diagnostic" if bool(cfa_lenspsf_native_audit.get("pass")) else "not_supported",
                "claim_strength": "claim_boundary_guardrail",
                "evidence": _cfa_lenspsf_native_evidence(cfa_lenspsf_native_audit),
                "claim_boundary": "Native group can support native-CFA evidence; remapped group is bridge/remap sensitivity only.",
                "next_evidence": "Rerun larger CameraE2E sweeps with native_bayer_v1 and a fresh raw cache for each target Bayer CFA, or keep historical remapped rows explicitly labeled as remap sensitivity.",
            }
        )

    if scene_edge_confidence is not None:
        current.append(
            {
                "area": "High-information scene edge similarity",
                "status": "diagnostic" if bool(scene_edge_confidence.get("pass")) else "not_supported",
                "claim_strength": "diagnostic",
                "evidence": _scene_edge_evidence(scene_edge_confidence),
                "claim_boundary": "Shows scene-edge tracking against a high-information proxy; not object-boundary ground truth or detector accuracy.",
                "next_evidence": "Use the scene-edge proxy around detector boxes to explain which proposals are kept or removed.",
            }
        )

    if scene_information_stress is not None:
        current.append(
            {
                "area": "Scene-to-sensor information loss",
                "status": "diagnostic" if bool(scene_information_stress.get("pass")) else "not_supported",
                "claim_strength": "diagnostic",
                "evidence": _scene_information_evidence(scene_information_stress),
                "claim_boundary": "Shows what the sensor loses or aliases before ISP; it does not prove recovery of information absent from RAW.",
                "next_evidence": "Add real high-resolution scenes where the scene oracle has more detail/color information than the sensor RAW.",
            }
        )

    if aux_contribution_audit is not None:
        current.append(
            {
                "area": "Aux evidence used downstream",
                "status": "diagnostic" if bool(aux_contribution_audit.get("pass")) else "not_supported",
                "claim_strength": "proposal_level_bridge",
                "evidence": _aux_contribution_evidence(aux_contribution_audit),
                "claim_boundary": "This is proposal-level scoring/filtering evidence; it is not a trained RGB+Aux DNN detector claim.",
                "next_evidence": "Train or adapt a DNN that consumes RGB+Aux tensors, then rerun held-out detector gates.",
            }
        )

    if casebook is not None:
        current.append(
            {
                "area": "Visual success/failure casebook",
                "status": "diagnostic" if bool(casebook.get("pass")) else "not_supported",
                "claim_strength": "qualitative_review",
                "evidence": _casebook_evidence(casebook),
                "claim_boundary": "Qualitative case review only; does not replace held-out gates, native RAW/CFA coverage, or DNN evaluation.",
                "next_evidence": "Expand the casebook with native-CFA/adverse-condition slices and reviewable TP-loss counterexamples.",
            }
        )

    if training is not None:
        training_status = str(training.get("status", ""))
        current.append(
            {
                "area": "RGB+Aux DNN training path",
                "status": "needs_eval" if training_status in {"diagnostic_only", "training_path_only"} else "needs_gate",
                "claim_strength": training_status,
                "evidence": _training_evidence(training),
                "claim_boundary": "Use as implementation/resource evidence until held-out DNN detector metrics pass a gate.",
                "next_evidence": "Run a controlled RGB-only versus RGB+Aux fine-tune with enough samples for a held-out gate.",
            }
        )

    return {
        "claim_posture": _claim_posture(claims, protocol_coverage),
        "current_evidence": current,
        "future_evidence": _future_evidence_rows(
            scene_edge_confidence=scene_edge_confidence,
            aux_contribution_audit=aux_contribution_audit,
            edge_fidelity_suite=edge_fidelity_suite,
            cfa_stress_sweep=cfa_stress_sweep,
            cfa_lenspsf_detector_sweep=cfa_lenspsf_detector_sweep,
            cfa_lenspsf_proposal_audit=cfa_lenspsf_proposal_audit,
            cfa_lenspsf_native_audit=cfa_lenspsf_native_audit,
            casebook=casebook,
            training=training,
        ),
    }


def _first_claim(claims: Sequence[Mapping[str, Any]], profile: str) -> Mapping[str, Any] | None:
    for claim in claims:
        if str(claim.get("profile", "")) == profile:
            return claim
    return None


def _claim_posture(claims: Sequence[Mapping[str, Any]], protocol_coverage: Mapping[str, Any] | None) -> Dict[str, Any]:
    broad = _first_claim(claims, "broad_superiority")
    fp_claim = _first_claim(claims, "fp_reducer")
    protocol_metric = str(protocol_coverage.get("metric_claim_status", "")) if protocol_coverage is not None else ""
    if fp_claim is not None and bool(fp_claim.get("pass")):
        recommended = "Use a narrow recall-budgeted FP-reduction claim, with front-end/aux evidence as feasibility support."
    else:
        recommended = "Do not make a performance claim yet; use the current evidence as diagnostic feasibility only."
    if broad is not None and not bool(broad.get("pass")):
        blocked = "Do not claim broad HumanISP superiority."
    else:
        blocked = ""
    return {
        "recommended_claim": recommended,
        "blocked_claim": blocked,
        "metric_claim_status": protocol_metric,
    }


def _claim_metric_evidence(claim: Mapping[str, Any]) -> str:
    metrics = claim.get("metrics", {}) if isinstance(claim.get("metrics"), Mapping) else {}
    parts = [f"samples={int(claim.get('sample_count', 0))}"]
    for label, metric in (
        ("dP50", "precision@0.50_mean"),
        ("dR50", "recall@0.50_mean"),
        ("dSmallR50", "small_recall@0.50_mean"),
        ("dFP50", "fp@0.50_mean"),
    ):
        row = metrics.get(metric)
        if not isinstance(row, Mapping) or row.get("delta") is None:
            continue
        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
        ci = "" if ci_low is None or ci_high is None else f" CI[{_fmt(ci_low, signed=True)}, {_fmt(ci_high, signed=True)}]"
        parts.append(f"{label}={_fmt(row.get('delta'), signed=True)}{ci}")
    failed = ", ".join(str(value) for value in claim.get("failed_metrics", ())) or "none"
    parts.append(f"failed={failed}")
    return "; ".join(parts)


def _task_evidence(task_metrics: Mapping[str, Any] | None, task_gate: Mapping[str, Any] | None) -> str:
    parts: list[str] = []
    if task_gate is not None:
        failed = ", ".join(str(value) for value in task_gate.get("failed_groups", ())) or "none"
        parts.append(
            f"task gate {task_gate.get('verdict', '')}; evaluated={int(task_gate.get('evaluated_group_count', 0))}; failed={failed}"
        )
    if task_metrics is not None:
        rows = [row for row in task_metrics.get("rows", ()) if isinstance(row, Mapping)]
        recall_drops = [str(row.get("group", "")) for row in rows if (_signed_metric(row, "delta_recall@0.50") or 0.0) < 0.0]
        fp_reductions = [str(row.get("group", "")) for row in rows if (_signed_metric(row, "delta_fp@0.50_per_sample") or 0.0) < 0.0]
        parts.append(
            f"task metrics status={task_metrics.get('status', '')}; recall_drop_groups={', '.join(recall_drops) or 'none'}; fp_reduction_groups={', '.join(fp_reductions) or 'none'}"
        )
    return "; ".join(parts) if parts else "No task evidence was provided."


def _protocol_evidence(protocol: Mapping[str, Any]) -> str:
    missing = list(protocol.get("missing_required", ())) + list(protocol.get("missing_raw_claim", ()))
    return (
        f"coverage={protocol.get('coverage_status', '') or protocol.get('status', '')}; "
        f"metric_claim_status={protocol.get('metric_claim_status', '') or 'unknown'}; "
        f"missing={', '.join(str(value) for value in missing) or 'none'}"
    )


def _mechanism_evidence(mechanism: Mapping[str, Any]) -> str:
    cfas = ", ".join(str(value) for value in mechanism.get("cfa_patterns", ())) or "none"
    failed = ", ".join(str(value) for value in mechanism.get("failed_mechanisms", ())) or "none"
    return f"status={mechanism.get('status', '')}; mechanisms={int(mechanism.get('mechanism_count', 0))}; CFA={cfas}; failed={failed}"


def _cfa_evidence(cfa_stress: Mapping[str, Any]) -> str:
    top = "; ".join(
        f"{row.get('condition', '')}:{row.get('cfa_pattern', '')}@{_fmt(row.get('condition_score'))}"
        for row in cfa_stress.get("top_rows", ())
        if isinstance(row, Mapping)
    )
    return f"cases={int(cfa_stress.get('case_count', 0))}; conditions={int(cfa_stress.get('condition_count', 0))}; top={top or 'none'}"


def _cfa_lenspsf_detector_evidence(sweep: Mapping[str, Any]) -> str:
    cfas = ", ".join(str(value) for value in sweep.get("cfa_patterns", ())) or "none"
    psf = ", ".join(_fmt(value) for value in sweep.get("psf_sigmas", ()) if value is not None) or "none"
    ranking = sweep.get("rankings", {}) if isinstance(sweep.get("rankings"), Mapping) else {}
    fp_rows = ranking.get("calibrated_or_fusion_by_delta_fp@0.50", ())
    best_fp = next((row for row in fp_rows if isinstance(row, Mapping)), None)
    best_clause = (
        "no downstream FP ranking"
        if best_fp is None
        else (
            f"best dFP={_fmt(best_fp.get('delta'), signed=True)} "
            f"at {best_fp.get('cfa_pattern', '')}/psf={_fmt(best_fp.get('psf_sigma'))} "
            f"input={best_fp.get('input', '')}"
        )
    )
    remap_values = [
        _maybe_float_or_none(row.get("pattern_remapped_fraction"))
        for row in sweep.get("runs", ())
        if isinstance(row, Mapping)
    ]
    true_cfa_values = [
        _maybe_float_or_none(row.get("true_sensor_cfa_mosaic_fraction"))
        for row in sweep.get("runs", ())
        if isinstance(row, Mapping)
    ]
    max_remap = max((value for value in remap_values if value is not None), default=None)
    min_true_cfa = min((value for value in true_cfa_values if value is not None), default=None)
    bridge_versions = sorted(
        {
            str(version)
            for row in sweep.get("runs", ())
            if isinstance(row, Mapping)
            for version in row.get("camerae2e_native_cfa_bridge_versions", ())
        }
    )
    return (
        f"runs={int(sweep.get('run_count', 0))}/{int(sweep.get('expected_run_count', 0))}; "
        f"samples/run={int(sweep.get('count', 0))}; CFA={cfas}; LensPSF={psf}; "
        f"{best_clause}; max_remap={_fmt(max_remap)}; min_true_cfa={_fmt(min_true_cfa)}; "
        f"bridge={', '.join(bridge_versions) or 'unknown'}"
    )


def _cfa_lenspsf_proposal_evidence(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate"), Mapping) else {}
    best_scene = aggregate.get("best_scene_edge_auc_condition", {}) if isinstance(aggregate.get("best_scene_edge_auc_condition"), Mapping) else {}
    best_edge = aggregate.get("best_edge_auc_condition", {}) if isinstance(aggregate.get("best_edge_auc_condition"), Mapping) else {}
    return (
        f"conditions={int(audit.get('condition_count', 0))}/{int(audit.get('expected_condition_count', 0))}; "
        f"removedFP={int(aggregate.get('removed_fp_count', 0))}; removedTP={int(aggregate.get('removed_tp_count', 0))}; "
        f"dFP={int(aggregate.get('fp_delta_count', 0))}; "
        f"scenePositive={int(aggregate.get('scene_edge_positive_condition_count', 0))}; "
        f"edgePositive={int(aggregate.get('edge_positive_condition_count', 0))}; "
        f"bestSceneAUC={best_scene.get('run_id', '')}@{_fmt(best_scene.get('scene_edge_auc_low_predicts_removed_fp'))}; "
        f"bestEdgeAUC={best_edge.get('run_id', '')}@{_fmt(best_edge.get('edge_auc_low_predicts_removed_fp'))}"
    )


def _cfa_lenspsf_native_evidence(audit: Mapping[str, Any]) -> str:
    groups = audit.get("groups", {}) if isinstance(audit.get("groups"), Mapping) else {}
    native = groups.get("native", {}) if isinstance(groups.get("native"), Mapping) else {}
    remapped = groups.get("remapped", {}) if isinstance(groups.get("remapped"), Mapping) else {}
    partial = groups.get("partial_remap", {}) if isinstance(groups.get("partial_remap"), Mapping) else {}
    native_best = native.get("best_delta_fp@0.50", {}) if isinstance(native.get("best_delta_fp@0.50"), Mapping) else {}
    remapped_best = remapped.get("best_delta_fp@0.50", {}) if isinstance(remapped.get("best_delta_fp@0.50"), Mapping) else {}
    remapped_count = int(remapped.get("run_count", 0))
    remapped_clause = (
        "remapped=0 runs/0 samples"
        if remapped_count == 0
        else (
            f"remapped={remapped_count} runs/{int(remapped.get('sample_count', 0))} samples "
            f"CFA={', '.join(str(value) for value in remapped.get('cfa_patterns', ())) or 'none'} "
            f"mean dFP={_fmt(remapped.get('mean_delta_fp@0.50'), signed=True)} "
            f"best={remapped_best.get('run_id', '')}@{_fmt(remapped_best.get('delta'), signed=True)}"
        )
    )
    return (
        f"runs={int(audit.get('run_count', 0))}/{int(audit.get('expected_run_count', 0))}; "
        f"native={int(native.get('run_count', 0))} runs/{int(native.get('sample_count', 0))} samples "
        f"CFA={', '.join(str(value) for value in native.get('cfa_patterns', ())) or 'none'} "
        f"mean dFP={_fmt(native.get('mean_delta_fp@0.50'), signed=True)} "
        f"best={native_best.get('run_id', '')}@{_fmt(native_best.get('delta'), signed=True)}; "
        f"{remapped_clause}; "
        f"partial={int(partial.get('run_count', 0))}"
    )


def _casebook_evidence(casebook: Mapping[str, Any]) -> str:
    categories = casebook.get("categories", {}) if isinstance(casebook.get("categories"), Mapping) else {}
    parts = [
        f"samples={int(casebook.get('sample_count', 0))}",
        f"selected={int(casebook.get('selected_case_count', 0))}",
    ]
    for name in ("fp_reduction_success", "recall_tradeoff", "recall_loss_failure", "fp_regression_failure"):
        payload = categories.get(name, {}) if isinstance(categories.get(name), Mapping) else {}
        parts.append(f"{name}={int(payload.get('case_count', 0))}/{int(payload.get('selected_case_count', 0))}")
    aggregate = casebook.get("aggregate", {}) if isinstance(casebook.get("aggregate"), Mapping) else {}
    parts.append(f"dTP={int(aggregate.get('tp_delta_count', 0))}")
    parts.append(f"dFP={int(aggregate.get('fp_delta_count', 0))}")
    return "; ".join(parts)


def _edge_confidence_evidence(edge_confidence: Mapping[str, Any]) -> str:
    deltas = "; ".join(
        f"{row.get('metric', '')} {row.get('check', '')}={_fmt(row.get('delta'), signed=True)}"
        for row in edge_confidence.get("key_deltas", ())[:3]
        if isinstance(row, Mapping)
    )
    failed = ", ".join(str(value) for value in edge_confidence.get("failed_checks", ())) or "none"
    return f"cases={int(edge_confidence.get('case_count', 0))}; checks={int(edge_confidence.get('check_count', 0))}; failed={failed}; {deltas}"


def _edge_fidelity_evidence(edge_fidelity: Mapping[str, Any]) -> str:
    cfas = ", ".join(str(value) for value in edge_fidelity.get("cfa_patterns", ())) or "none"
    psf = ", ".join(_fmt(value) for value in edge_fidelity.get("psf_sigmas", ()) if value is not None) or "none"
    top = "; ".join(
        f"psf={_fmt(row.get('psf_sigma'))} top={row.get('cfa_pattern', '')} auxF1={_fmt(row.get('aux_object_edge_f1'))}"
        for row in edge_fidelity.get("top_rows", ())
        if isinstance(row, Mapping)
    )
    return f"cases={int(edge_fidelity.get('case_count', 0))}; CFA={cfas}; LensPSF={psf}; {top or 'no top rows'}"


def _scene_edge_evidence(scene_edge: Mapping[str, Any]) -> str:
    cfas = ", ".join(str(value) for value in scene_edge.get("cfa_patterns", ())) or "none"
    psf = ", ".join(_fmt(value) for value in scene_edge.get("psf_sigmas", ()) if value is not None) or "none"
    return (
        f"reports={int(scene_edge.get('report_count', 1))}; cases={int(scene_edge.get('case_count', 0))}; "
        f"CFA={cfas}; LensPSF={psf}; "
        f"RGB dF1={_fmt(scene_edge.get('perception_rgb_minus_human_source_edge_f1_mean'), signed=True)}; "
        f"aux-strength dF1={_fmt(scene_edge.get('perception_aux_strength_minus_human_source_edge_f1_mean'), signed=True)}; "
        f"RGB win={_fmt(scene_edge.get('perception_rgb_source_edge_f1_win_rate'))}; "
        f"aux win={_fmt(scene_edge.get('perception_aux_strength_source_edge_f1_win_rate'))}"
    )


def _scene_information_evidence(scene_information: Mapping[str, Any]) -> str:
    return (
        f"scene={int(scene_information.get('scene_width', 0))}x{int(scene_information.get('scene_height', 0))}; "
        f"sensor={int(scene_information.get('sensor_width', 0))}x{int(scene_information.get('sensor_height', 0))}; "
        f"CFA={scene_information.get('cfa_pattern', '')}; cases={int(scene_information.get('case_count', 0))}; "
        f"checks={int(scene_information.get('check_count', 0))}"
    )


def _aux_contribution_evidence(aux_contribution: Mapping[str, Any]) -> str:
    fp_delta = _preferred_aux_fp_delta(aux_contribution.get("comparisons", ()))
    bridge = aux_contribution.get("sample_bridge")
    bridge_text = ""
    if isinstance(bridge, Mapping):
        support_deltas = bridge.get("support_deltas", {}) if isinstance(bridge.get("support_deltas"), Mapping) else {}
        edge_delta = _maybe_float_or_none(support_deltas.get("removed_fp_minus_kept_tp_edge_support_mean"))
        edge_correlation = _proposal_correlation_lookup(
            bridge.get("proposal_correlation"),
            feature="edge_support",
            comparison="removed_fp_vs_kept_tp",
        )
        scene_edge_correlation = _proposal_correlation_lookup(
            bridge.get("proposal_correlation"),
            feature="scene_edge_support",
            comparison="removed_fp_vs_kept_tp",
        )
        edge_auc = _maybe_float_or_none(edge_correlation.get("auc_low_feature_predicts_positive")) if isinstance(edge_correlation, Mapping) else None
        scene_edge_delta = _maybe_float_or_none(scene_edge_correlation.get("delta")) if isinstance(scene_edge_correlation, Mapping) else None
        scene_edge_auc = (
            _maybe_float_or_none(scene_edge_correlation.get("auc_low_feature_predicts_positive"))
            if isinstance(scene_edge_correlation, Mapping)
            else None
        )
        bridge_text = (
            f"; same-sample removed_fp={int(bridge.get('removed_fp_count', 0))}, removed_tp={int(bridge.get('removed_tp_count', 0))}, "
            f"fp_delta_count={int(bridge.get('fp_delta_count', 0))}, removed_fp_fraction={_fmt(bridge.get('removed_fp_fraction'))}, "
            f"removedFP-edge-vs-keptTP={_fmt(edge_delta, signed=True)}, edgeAUC_low={_fmt(edge_auc)}, "
            f"sourceSceneEdgeDelta={_fmt(scene_edge_delta, signed=True)}, sourceSceneEdgeAUC_low={_fmt(scene_edge_auc)}"
        )
    return (
        f"aux_features={int(aux_contribution.get('aux_feature_count', 0))}; "
        f"incremental dFP50={_fmt(fp_delta, signed=True)}{bridge_text}"
    )


def _training_evidence(training: Mapping[str, Any]) -> str:
    best = training.get("best_train_rate") if isinstance(training.get("best_train_rate"), Mapping) else {}
    lowest_fp = training.get("lowest_fp_eval") if isinstance(training.get("lowest_fp_eval"), Mapping) else {}
    best_recall = training.get("best_recall_eval") if isinstance(training.get("best_recall_eval"), Mapping) else {}
    return (
        f"status={training.get('status', '')}; runs={int(training.get('run_count', 0))}; "
        f"best_train={best.get('name', '')} throughput={_fmt(best.get('throughput'))}; "
        f"best_recall R50={_fmt(best_recall.get('recall@0.50_mean'))}; "
        f"lowest_fp FP50={_fmt(lowest_fp.get('fp@0.50_mean'))}"
    )


def _future_evidence_rows(
    *,
    scene_edge_confidence: Mapping[str, Any] | None,
    aux_contribution_audit: Mapping[str, Any] | None,
    edge_fidelity_suite: Mapping[str, Any] | None,
    cfa_stress_sweep: Mapping[str, Any] | None,
    cfa_lenspsf_detector_sweep: Mapping[str, Any] | None,
    cfa_lenspsf_proposal_audit: Mapping[str, Any] | None,
    cfa_lenspsf_native_audit: Mapping[str, Any] | None,
    casebook: Mapping[str, Any] | None,
    training: Mapping[str, Any] | None,
) -> list[Dict[str, Any]]:
    edge_correlation = _sample_bridge_edge_correlation(aux_contribution_audit)
    scene_edge_correlation = _sample_bridge_scene_edge_correlation(aux_contribution_audit)
    has_edge_correlation = isinstance(edge_correlation, Mapping) and bool(edge_correlation.get("lower_feature_predicts_positive"))
    has_scene_edge_correlation = isinstance(scene_edge_correlation, Mapping) and bool(scene_edge_correlation.get("lower_feature_predicts_positive"))
    if cfa_lenspsf_native_audit is not None:
        scene_aux_gap = "Native/remap CFA separation exists; next step is larger scale native_bayer_v1 reruns with same-sample scene-edge proposal correlation."
    elif cfa_lenspsf_proposal_audit is not None:
        scene_aux_gap = "CFA/LensPSF proposal-edge bridge exists; next step is larger scale and native-CFA separation."
    elif has_scene_edge_correlation:
        scene_aux_gap = "Same-sample source scene-edge proposal correlation exists; it is not yet swept across CFA/LensPSF conditions."
    elif has_edge_correlation:
        scene_aux_gap = "Proposal-level aux edge correlation exists; high-information scene-edge oracle is not yet joined to each detector box."
    elif scene_edge_confidence is not None and aux_contribution_audit is not None:
        scene_aux_gap = "Scene-edge evidence and aux proposal bridge both exist, but they are not yet joined per proposal/object."
    else:
        scene_aux_gap = "Need both scene-edge evidence and aux proposal bridge before correlation can be measured."
    if cfa_lenspsf_native_audit is not None:
        cfa_gap = "Detector sweep exists and native/remap rows are separated; rerun at larger sample scale with native_bayer_v1 and fresh raw caches for all target Bayer patterns."
    elif cfa_lenspsf_detector_sweep is not None:
        cfa_gap = "Detector sweep exists; expand sample scale and connect each condition to same-sample proposal edge correlation."
    elif cfa_stress_sweep is not None or edge_fidelity_suite is not None:
        cfa_gap = "CFA/LensPSF front-end diagnostics exist, but detector metrics are not yet swept per CFA/LensPSF condition."
    else:
        cfa_gap = "Need CFA/LensPSF front-end diagnostics first."
    training_status = str(training.get("status", "")) if training is not None else "missing"
    casebook_gap = (
        "Visual success/failure casebook exists; next step is native-CFA/adverse-condition expansion."
        if casebook is not None
        else "Dashboard has gates and aggregate slices, but not a curated visual failure/success report."
    )
    return [
        {
            "priority": "P0",
            "evidence": "Scene-edge proposal correlation across CFA/LensPSF",
            "why": "Tests whether the new same-sample scene-edge proposal evidence holds when CFA pattern and optical blur change.",
            "current_gap": scene_aux_gap,
            "implementation_path": "Run matched CFA/LensPSF cases and keep the same proposal-level source scene-edge, aux-edge, reliability, and TP/FP/removed flags.",
        },
        {
            "priority": "P0",
            "evidence": "CFA/LensPSF detector sweep",
            "why": "Shows whether the PerceptionISP advantage depends on source CFA pattern and optical blur, rather than only on synthetic front-end metrics.",
            "current_gap": cfa_gap,
            "implementation_path": "Run matched RGGB/GRBG/BGGR/GBRG plus PSF cases through CameraE2E, HumanISP, PerceptionISP, and a fixed detector.",
        },
        {
            "priority": "P1",
            "evidence": "RGB+Aux DNN fine-tune gate",
            "why": "Needed before claiming the aux tensor improves a learned detector rather than only proposal calibration.",
            "current_gap": f"Current DNN training status is {training_status}.",
            "implementation_path": "Fine-tune matched RGB-only and RGB+Aux models on the same split, then evaluate a held-out claim gate.",
        },
        {
            "priority": "P1",
            "evidence": "High-information real-scene expansion",
            "why": "The strongest PerceptionISP argument needs scenes with more information than the sensor can directly sample.",
            "current_gap": "Current high-information scene evidence is useful but still limited in scene diversity.",
            "implementation_path": "Build a small real-scene set with high-resolution source edges/color detail, then simulate lower-resolution CFA RAW measurements.",
        },
        {
            "priority": "P1",
            "evidence": "Failure and slice report",
            "why": "Makes the feasibility claim more credible by showing where PerceptionISP helps, where it is neutral, and where it hurts.",
            "current_gap": casebook_gap,
            "implementation_path": "Collect representative FP removals, TP losses, CFA/PSF wins, and counterexamples into a visual casebook.",
        },
    ]


def _sample_bridge_edge_correlation(aux_contribution_audit: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if aux_contribution_audit is None:
        return None
    sample_bridge = aux_contribution_audit.get("sample_bridge")
    if not isinstance(sample_bridge, Mapping):
        return None
    return _proposal_correlation_lookup(
        sample_bridge.get("proposal_correlation"),
        feature="edge_support",
        comparison="removed_fp_vs_kept_tp",
    )


def _sample_bridge_scene_edge_correlation(aux_contribution_audit: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if aux_contribution_audit is None:
        return None
    sample_bridge = aux_contribution_audit.get("sample_bridge")
    if not isinstance(sample_bridge, Mapping):
        return None
    return _proposal_correlation_lookup(
        sample_bridge.get("proposal_correlation"),
        feature="scene_edge_support",
        comparison="removed_fp_vs_kept_tp",
    )


def _proposal_correlation_lookup(payload: Any, *, feature: str, comparison: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    for row in payload.get("rows", ()):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("feature", "")) == str(feature) and str(row.get("comparison", "")) == str(comparison):
            return row
    return None


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
    cfa_lenspsf_detector_sweep: Mapping[str, Any] | None,
    cfa_lenspsf_proposal_audit: Mapping[str, Any] | None,
    cfa_lenspsf_native_audit: Mapping[str, Any] | None,
    casebook: Mapping[str, Any] | None,
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
    if cfa_lenspsf_detector_sweep is not None:
        if bool(cfa_lenspsf_detector_sweep.get("pass")):
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": "CFA/LensPSF detector sweep is available as condition-level detector evidence; use it for sensitivity analysis, not broad superiority.",
                }
            )
        else:
            failed = ", ".join(str(value) for value in cfa_lenspsf_detector_sweep.get("failed_checks", ())) or "configured CFA/LensPSF detector checks"
            decisions.append({"status": "not_supported", "claim": f"CFA/LensPSF detector sweep failed for {failed}; do not use it as condition detector evidence yet."})
    if cfa_lenspsf_proposal_audit is not None:
        if bool(cfa_lenspsf_proposal_audit.get("pass")):
            aggregate = cfa_lenspsf_proposal_audit.get("aggregate", {}) if isinstance(cfa_lenspsf_proposal_audit.get("aggregate"), Mapping) else {}
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": (
                        "CFA/LensPSF proposal-edge audit passed as condition-level bridge evidence: "
                        f"removed FP {int(aggregate.get('removed_fp_count', 0))}, removed TP {int(aggregate.get('removed_tp_count', 0))}, "
                        f"source scene-edge positive conditions {int(aggregate.get('scene_edge_positive_condition_count', 0))}."
                    ),
                }
            )
        else:
            failed = ", ".join(str(value) for value in cfa_lenspsf_proposal_audit.get("failed_checks", ())) or "configured CFA/LensPSF proposal checks"
            decisions.append({"status": "not_supported", "claim": f"CFA/LensPSF proposal-edge audit failed for {failed}; do not use it as proposal bridge evidence yet."})
    if cfa_lenspsf_native_audit is not None:
        if bool(cfa_lenspsf_native_audit.get("pass")):
            groups = cfa_lenspsf_native_audit.get("groups", {}) if isinstance(cfa_lenspsf_native_audit.get("groups"), Mapping) else {}
            native = groups.get("native", {}) if isinstance(groups.get("native"), Mapping) else {}
            remapped = groups.get("remapped", {}) if isinstance(groups.get("remapped"), Mapping) else {}
            remapped_count = int(remapped.get("run_count", 0))
            remap_clause = (
                "All audited rows are native CameraE2E CFA rows."
                if remapped_count == 0
                else "Use remapped rows only as bridge sensitivity evidence."
            )
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": (
                        "CFA/LensPSF native-CFA audit passed: "
                        f"native rows {int(native.get('run_count', 0))}, remapped rows {remapped_count}. "
                        f"{remap_clause}"
                    ),
                }
            )
        else:
            failed = ", ".join(str(value) for value in cfa_lenspsf_native_audit.get("failed_checks", ())) or "configured native-CFA separation checks"
            decisions.append({"status": "not_supported", "claim": f"CFA/LensPSF native-CFA audit failed for {failed}; do not make native CFA claims yet."})
    if aux_contribution_audit is not None:
        if bool(aux_contribution_audit.get("pass")):
            decisions.append({"status": "diagnostic", "claim": "Aux contribution audit passed; aux features add proposal-scoring FP reduction within the recall budget, but this is calibration evidence rather than DNN performance."})
            sample_bridge = aux_contribution_audit.get("sample_bridge")
            if isinstance(sample_bridge, Mapping) and int(sample_bridge.get("removed_fp_count", 0)) > int(sample_bridge.get("removed_tp_count", 0)):
                support_deltas = sample_bridge.get("support_deltas", {}) if isinstance(sample_bridge.get("support_deltas"), Mapping) else {}
                edge_delta = _maybe_float_or_none(support_deltas.get("removed_fp_minus_kept_tp_edge_support_mean"))
                edge_clause = "" if edge_delta is None else f" Removed-FP edge support delta vs kept TP {_fmt(edge_delta, signed=True)}."
                edge_correlation = _proposal_correlation_lookup(
                    sample_bridge.get("proposal_correlation"),
                    feature="edge_support",
                    comparison="removed_fp_vs_kept_tp",
                )
                edge_auc = _maybe_float_or_none(edge_correlation.get("auc_low_feature_predicts_positive")) if isinstance(edge_correlation, Mapping) else None
                scene_edge_correlation = _proposal_correlation_lookup(
                    sample_bridge.get("proposal_correlation"),
                    feature="scene_edge_support",
                    comparison="removed_fp_vs_kept_tp",
                )
                scene_edge_auc = (
                    _maybe_float_or_none(scene_edge_correlation.get("auc_low_feature_predicts_positive"))
                    if isinstance(scene_edge_correlation, Mapping)
                    else None
                )
                edge_auc_clause = "" if edge_auc is None else f" Low-edge AUC for removed FP vs kept TP {_fmt(edge_auc)}."
                scene_edge_clause = "" if scene_edge_auc is None else f" Source scene-edge AUC {_fmt(scene_edge_auc)}."
                decisions.append(
                    {
                        "status": "diagnostic",
                        "claim": (
                            "Same-sample aux bridge passed: incremental aux scoring removed "
                            f"{int(sample_bridge.get('removed_fp_count', 0))} FP and "
                            f"{int(sample_bridge.get('removed_tp_count', 0))} TP proposals "
                            f"with net FP delta {int(sample_bridge.get('fp_delta_count', 0))}."
                            f"{edge_clause}{edge_auc_clause}{scene_edge_clause} "
                            "This is proposal-level evidence, not a trained DNN detector claim."
                        ),
                    }
                )
        else:
            failed = ", ".join(str(value) for value in aux_contribution_audit.get("failed_checks", ())) or "configured aux contribution checks"
            decisions.append({"status": "not_supported", "claim": f"Aux contribution audit failed for {failed}; do not claim aux helps proposal scoring yet."})
    if casebook is not None:
        if bool(casebook.get("pass")):
            categories = casebook.get("categories", {}) if isinstance(casebook.get("categories"), Mapping) else {}
            success = categories.get("fp_reduction_success", {}) if isinstance(categories.get("fp_reduction_success"), Mapping) else {}
            counterexamples = sum(
                int((categories.get(name, {}) if isinstance(categories.get(name), Mapping) else {}).get("selected_case_count", 0))
                for name in ("recall_tradeoff", "recall_loss_failure", "fp_regression_failure")
            )
            decisions.append(
                {
                    "status": "diagnostic",
                    "claim": (
                        "Visual success/failure casebook is available for qualitative review: "
                        f"selected FP-reduction successes {int(success.get('selected_case_count', 0))}, selected counterexamples {counterexamples}."
                    ),
                }
            )
        else:
            failed = ", ".join(str(value) for value in casebook.get("failed_checks", ())) or "configured casebook checks"
            decisions.append({"status": "not_supported", "claim": f"Visual success/failure casebook failed for {failed}; do not use it as review evidence yet."})
    bridge = _scene_aux_downstream_bridge(scene_edge_confidence, aux_contribution_audit)
    if bridge is not None:
        decisions.append(bridge)
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


def _scene_aux_downstream_bridge(
    scene_edge_confidence: Mapping[str, Any] | None,
    aux_contribution_audit: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    if scene_edge_confidence is None or aux_contribution_audit is None:
        return None
    if not bool(scene_edge_confidence.get("pass")) or not bool(aux_contribution_audit.get("pass")):
        return None
    rgb_delta = _maybe_float_or_none(scene_edge_confidence.get("perception_rgb_minus_human_source_edge_f1_mean"))
    aux_strength_delta = _maybe_float_or_none(scene_edge_confidence.get("perception_aux_strength_minus_human_source_edge_f1_mean"))
    fp_delta = _preferred_aux_fp_delta(aux_contribution_audit.get("comparisons", ()))
    if rgb_delta is None or aux_strength_delta is None or fp_delta is None:
        return None
    if rgb_delta <= 0.0 or aux_strength_delta <= 0.0 or fp_delta >= 0.0:
        return None
    return {
        "status": "diagnostic",
        "claim": (
            "Front-end/downstream bridge is directionally positive: "
            f"scene-edge RGB F1 delta {_fmt(rgb_delta, signed=True)}, "
            f"aux edge-strength delta {_fmt(aux_strength_delta, signed=True)}, "
            f"and incremental aux proposal-scoring dFP@0.50 {_fmt(fp_delta, signed=True)}. "
            "This is co-observed evidence, not same-sample causal correlation."
        ),
    }


def _preferred_aux_fp_delta(comparisons: Any) -> float | None:
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)):
        return None
    rows = [row for row in comparisons if isinstance(row, Mapping)]
    for comparison_id in ("score_label_aux_vs_score_label", "score_aux_vs_fusion"):
        for row in rows:
            if str(row.get("id", "")) != comparison_id:
                continue
            value = _maybe_float_or_none(row.get("delta_fp@0.50"))
            if value is not None:
                return value
    values = [_maybe_float_or_none(row.get("delta_fp@0.50")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return min(values)


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
    evidence_map = dashboard.get("evidence_map")
    evidence_map_html = _evidence_map_html(evidence_map) if isinstance(evidence_map, Mapping) else ""
    claim_rows = "".join(_claim_row(item, destination) for item in dashboard.get("claims", ()))
    training = dashboard.get("training")
    training_html = _training_html(training, destination) if isinstance(training, Mapping) else "<p>No RGB+Aux training rollup was provided.</p>"
    aux_contribution = dashboard.get("aux_contribution_audit")
    aux_contribution_html = _aux_contribution_html(aux_contribution, destination) if isinstance(aux_contribution, Mapping) else "<p>No aux contribution audit was provided.</p>"
    casebook = dashboard.get("casebook")
    casebook_html = _casebook_html(casebook, destination) if isinstance(casebook, Mapping) else "<p>No success/failure casebook was provided.</p>"
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
    cfa_lenspsf_detector = dashboard.get("cfa_lenspsf_detector_sweep")
    cfa_lenspsf_detector_html = (
        _cfa_lenspsf_detector_html(cfa_lenspsf_detector, destination)
        if isinstance(cfa_lenspsf_detector, Mapping)
        else "<p>No CFA/LensPSF detector sweep summary was provided.</p>"
    )
    cfa_lenspsf_proposal = dashboard.get("cfa_lenspsf_proposal_audit")
    cfa_lenspsf_proposal_html = (
        _cfa_lenspsf_proposal_html(cfa_lenspsf_proposal, destination)
        if isinstance(cfa_lenspsf_proposal, Mapping)
        else "<p>No CFA/LensPSF proposal-edge audit summary was provided.</p>"
    )
    cfa_lenspsf_native = dashboard.get("cfa_lenspsf_native_audit")
    cfa_lenspsf_native_html = (
        _cfa_lenspsf_native_html(cfa_lenspsf_native, destination)
        if isinstance(cfa_lenspsf_native, Mapping)
        else "<p>No CFA/LensPSF native-CFA audit summary was provided.</p>"
    )
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
  {evidence_map_html}
  <h2>Claim Gates</h2>
  <table>
    <thead><tr><th>Name</th><th>Report</th><th>Claim</th><th>Profile</th><th>Verdict</th><th>Baseline</th><th>Target</th><th>Samples</th><th>P50 d/CI</th><th>R50 d/CI</th><th>Small R50 d/CI</th><th>FP d/CI</th><th>Failed</th></tr></thead>
    <tbody>{claim_rows}</tbody>
  </table>
  <h2>RGB+Aux DNN Training</h2>
  {training_html}
  <h2>Aux Contribution Audit</h2>
  {aux_contribution_html}
  <h2>Success/Failure Casebook</h2>
  {casebook_html}
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
  <h2>CFA/LensPSF Detector Sweep</h2>
  {cfa_lenspsf_detector_html}
  <h2>CFA/LensPSF Proposal Edge Bridge</h2>
  {cfa_lenspsf_proposal_html}
  <h2>CFA/LensPSF Native-CFA Separation</h2>
  {cfa_lenspsf_native_html}
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


def _evidence_map_html(evidence_map: Mapping[str, Any]) -> str:
    posture = evidence_map.get("claim_posture", {}) if isinstance(evidence_map.get("claim_posture"), Mapping) else {}
    current_rows = "".join(_evidence_current_row(row) for row in evidence_map.get("current_evidence", ()) if isinstance(row, Mapping))
    if not current_rows:
        current_rows = '<tr><td colspan="6">No current evidence rows were available.</td></tr>'
    future_rows = "".join(_evidence_future_row(row) for row in evidence_map.get("future_evidence", ()) if isinstance(row, Mapping))
    if not future_rows:
        future_rows = '<tr><td colspan="5">No future evidence rows were available.</td></tr>'
    return (
        "<h2>Performance Evidence Map</h2>"
        "<table>"
        "<thead><tr><th>Recommended Claim</th><th>Blocked Claim</th><th>Metric Claim Status</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{html_lib.escape(str(posture.get('recommended_claim', '')))}</td>"
        f"<td>{html_lib.escape(str(posture.get('blocked_claim', '')) or 'none')}</td>"
        f"<td><code>{html_lib.escape(str(posture.get('metric_claim_status', '')) or 'unknown')}</code></td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Evidence Area</th><th>Status</th><th>Claim Strength</th><th>Current Evidence</th><th>Claim Boundary</th><th>Next Evidence</th></tr></thead>"
        f"<tbody>{current_rows}</tbody></table>"
        "<h3>What More Evidence To Build</h3>"
        "<table>"
        "<thead><tr><th>Priority</th><th>Evidence</th><th>Why It Matters</th><th>Current Gap</th><th>Implementation Path</th></tr></thead>"
        f"<tbody>{future_rows}</tbody></table>"
    )


def _evidence_current_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('area', '')))}</td>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('claim_strength', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('claim_boundary', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('next_evidence', '')))}</td>"
        "</tr>"
    )


def _evidence_future_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('why', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('current_gap', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('implementation_path', '')))}</td>"
        "</tr>"
    )


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
    sample_bridge = aux_contribution.get("sample_bridge")
    sample_bridge_html = _aux_sample_bridge_html(sample_bridge) if isinstance(sample_bridge, Mapping) else "<p>No same-sample aux bridge was available.</p>"
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
        "<h3>Same-Sample Aux Bridge</h3>"
        f"{sample_bridge_html}"
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


def _casebook_html(casebook: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(casebook.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in casebook.get("failed_checks", ())) or "none"
    aggregate = casebook.get("aggregate", {}) if isinstance(casebook.get("aggregate"), Mapping) else {}
    category_rows = "".join(
        _casebook_category_row(name, payload)
        for name, payload in (casebook.get("categories", {}) if isinstance(casebook.get("categories"), Mapping) else {}).items()
        if isinstance(payload, Mapping)
    )
    if not category_rows:
        category_rows = '<tr><td colspan="5">No case categories were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(casebook.get('status', '')))}</code>. "
        f"{html_lib.escape(str(casebook.get('interpretation', '')))} "
        f"{html_lib.escape(str(casebook.get('claim_boundary', '')))}</p>"
        "<table><thead><tr><th>Report</th><th>Samples</th><th>Selected Cases</th><th>Baseline</th><th>Target</th><th>dTP</th><th>dFP</th><th>Failed</th></tr></thead><tbody>"
        f"<tr><td>{_report_link(casebook, destination)}</td>"
        f"<td>{int(casebook.get('sample_count', 0))}</td>"
        f"<td>{int(casebook.get('selected_case_count', 0))}</td>"
        f"<td><code>{html_lib.escape(str(casebook.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(casebook.get('target_input', '')))}</code></td>"
        f"<td>{int(aggregate.get('tp_delta_count', 0))}</td>"
        f"<td>{int(aggregate.get('fp_delta_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td></tr></tbody></table>"
        "<table><thead><tr><th>Category</th><th>Total Samples</th><th>Selected</th><th>Example Samples</th><th>Meaning</th></tr></thead>"
        f"<tbody>{category_rows}</tbody></table>"
    )


def _casebook_category_row(name: str, payload: Mapping[str, Any]) -> str:
    examples = ", ".join(
        str(row.get("sample_id", ""))
        for row in payload.get("cases", ())[:5]
        if isinstance(row, Mapping)
    )
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{int(payload.get('case_count', 0))}</td>"
        f"<td>{int(payload.get('selected_case_count', 0))}</td>"
        f"<td>{html_lib.escape(examples or 'none')}</td>"
        f"<td>{html_lib.escape(_casebook_category_description(str(name)))}</td>"
        "</tr>"
    )


def _casebook_category_description(name: str) -> str:
    return {
        "fp_reduction_success": "Target reduces FP while preserving same-sample TP count.",
        "recall_tradeoff": "Target reduces FP but loses same-sample TP count.",
        "recall_loss_failure": "Target loses same-sample TP without an FP reduction.",
        "fp_regression_failure": "Target adds same-sample FP versus baseline.",
    }.get(name, "")


def _aux_sample_bridge_html(sample_bridge: Mapping[str, Any]) -> str:
    support_deltas = sample_bridge.get("support_deltas", {}) if isinstance(sample_bridge.get("support_deltas"), Mapping) else {}
    edge_correlation = _proposal_correlation_lookup(
        sample_bridge.get("proposal_correlation"),
        feature="edge_support",
        comparison="removed_fp_vs_kept_tp",
    )
    scene_edge_correlation = _proposal_correlation_lookup(
        sample_bridge.get("proposal_correlation"),
        feature="scene_edge_support",
        comparison="removed_fp_vs_kept_tp",
    )
    edge_auc = _maybe_float_or_none(edge_correlation.get("auc_low_feature_predicts_positive")) if isinstance(edge_correlation, Mapping) else None
    edge_point_biserial = _maybe_float_or_none(edge_correlation.get("point_biserial")) if isinstance(edge_correlation, Mapping) else None
    scene_edge_delta = _maybe_float_or_none(scene_edge_correlation.get("delta")) if isinstance(scene_edge_correlation, Mapping) else None
    scene_edge_auc = (
        _maybe_float_or_none(scene_edge_correlation.get("auc_low_feature_predicts_positive"))
        if isinstance(scene_edge_correlation, Mapping)
        else None
    )
    return (
        f"<p>{html_lib.escape(str(sample_bridge.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Baseline</th><th>Target</th><th>Samples</th><th>Removed FP</th><th>Removed TP</th><th>FP Delta</th><th>TP Delta</th><th>Removed FP Fraction</th><th>Removed FP Edge Delta vs Kept TP</th><th>Low-Edge AUC</th><th>Edge Point-Biserial</th><th>Source Scene Edge Delta</th><th>Source Scene Edge AUC</th></tr></thead>"
        "<tbody><tr>"
        f"<td><code>{html_lib.escape(str(sample_bridge.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(sample_bridge.get('target_input', '')))}</code></td>"
        f"<td>{int(sample_bridge.get('compared_sample_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('removed_fp_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('removed_tp_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('fp_delta_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('tp_delta_count', 0))}</td>"
        f"<td>{_fmt(sample_bridge.get('removed_fp_fraction'))}</td>"
        f"<td>{_fmt(support_deltas.get('removed_fp_minus_kept_tp_edge_support_mean'), signed=True)}</td>"
        f"<td>{_fmt(edge_auc)}</td>"
        f"<td>{_fmt(edge_point_biserial, signed=True)}</td>"
        f"<td>{_fmt(scene_edge_delta, signed=True)}</td>"
        f"<td>{_fmt(scene_edge_auc)}</td>"
        "</tr></tbody></table>"
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


def _cfa_lenspsf_detector_html(sweep: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(sweep.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in sweep.get("failed_checks", ())) or "none"
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in sweep.get("checks", ())
        if isinstance(row, Mapping)
    )
    run_rows = "".join(_cfa_lenspsf_detector_run_row(row, destination) for row in sweep.get("runs", ()))
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(sweep.get('status', '')))}</code>. "
        f"{html_lib.escape(str(sweep.get('interpretation', '')))} "
        f"{html_lib.escape(str(sweep.get('claim_boundary', '')))}</p>"
        "<table><thead><tr><th>Report</th><th>Runs</th><th>Samples/Run</th><th>CFA</th><th>PSF</th><th>CameraE2E</th><th>Failed</th></tr></thead><tbody>"
        f"<tr><td>{_report_link(sweep, destination)}</td>"
        f"<td>{int(sweep.get('run_count', 0))}/{int(sweep.get('expected_run_count', 0))}</td>"
        f"<td>{int(sweep.get('count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in sweep.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(_fmt(value) for value in sweep.get('psf_sigmas', ()) if value is not None) or 'none')}</td>"
        f"<td>{html_lib.escape(str(bool(sweep.get('use_camerae2e'))))}</td>"
        f"<td>{html_lib.escape(failed)}</td></tr></tbody></table>"
        "<h3>CFA/LensPSF Checks</h3>"
        f"<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>"
        "<h3>Primary Downstream Input By Condition</h3>"
        "<table><thead><tr><th>Run</th><th>CFA</th><th>PSF</th><th>Input</th><th>P50</th><th>R50</th><th>FP50</th><th>dP50</th><th>dR50</th><th>dFP50</th><th>Remap</th><th>True CFA</th><th>PSF Rec</th><th>Camera Type</th><th>Bridge</th></tr></thead>"
        f"<tbody>{run_rows}</tbody></table>"
    )


def _cfa_lenspsf_detector_run_row(row: Mapping[str, Any], destination: Path) -> str:
    link = ""
    if row.get("html_path"):
        relative = os.path.relpath(str(row.get("html_path")), start=str(destination))
        link = f"<a href=\"{html_lib.escape(relative)}\">{html_lib.escape(str(row.get('run_id', '')))}</a>"
    return (
        "<tr>"
        f"<td>{link or html_lib.escape(str(row.get('run_id', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('primary_input', '')))}</code></td>"
        f"<td>{_fmt(row.get('precision@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('recall@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('fp@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('delta_precision@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_fp@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('pattern_remapped_fraction'))}</td>"
        f"<td>{_fmt(row.get('true_sensor_cfa_mosaic_fraction'))}</td>"
        f"<td>{_fmt(row.get('psf_recorded_fraction'))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('camerae2e_camera_types', ())) or 'n/a')}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('camerae2e_native_cfa_bridge_versions', ())) or 'n/a')}</td>"
        "</tr>"
    )


def _cfa_lenspsf_proposal_html(audit: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(audit.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in audit.get("failed_checks", ())) or "none"
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate"), Mapping) else {}
    best_scene = aggregate.get("best_scene_edge_auc_condition", {}) if isinstance(aggregate.get("best_scene_edge_auc_condition"), Mapping) else {}
    best_edge = aggregate.get("best_edge_auc_condition", {}) if isinstance(aggregate.get("best_edge_auc_condition"), Mapping) else {}
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in audit.get("checks", ())
        if isinstance(row, Mapping)
    )
    if not check_rows:
        check_rows = '<tr><td colspan="3">No proposal-edge checks were available.</td></tr>'
    condition_rows = "".join(_cfa_lenspsf_proposal_condition_row(row, destination) for row in audit.get("conditions", ()))
    if not condition_rows:
        condition_rows = '<tr><td colspan="10">No condition bridge rows were available.</td></tr>'
    cfas = ", ".join(str(value) for value in audit.get("cfa_patterns", ())) or "none"
    psf = ", ".join(_fmt(value) for value in audit.get("psf_sigmas", ()) if value is not None) or "none"
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(audit.get('status', '')))}</code>. "
        f"{html_lib.escape(str(audit.get('interpretation', '')))} "
        f"{html_lib.escape(str(audit.get('claim_boundary', '')))}</p>"
        "<table><thead><tr><th>Report</th><th>Conditions</th><th>CFA</th><th>PSF</th><th>Removed FP</th><th>Removed TP</th><th>Net FP Delta</th><th>Scene Positive</th><th>Aux Edge Positive</th><th>Failed</th></tr></thead><tbody>"
        f"<tr><td>{_report_link(audit, destination)}</td>"
        f"<td>{int(audit.get('condition_count', 0))}/{int(audit.get('expected_condition_count', 0))}</td>"
        f"<td>{html_lib.escape(cfas)}</td>"
        f"<td>{html_lib.escape(psf)}</td>"
        f"<td>{int(aggregate.get('removed_fp_count', 0))}</td>"
        f"<td>{int(aggregate.get('removed_tp_count', 0))}</td>"
        f"<td>{int(aggregate.get('fp_delta_count', 0))}</td>"
        f"<td>{int(aggregate.get('scene_edge_positive_condition_count', 0))}</td>"
        f"<td>{int(aggregate.get('edge_positive_condition_count', 0))}</td>"
        f"<td>{html_lib.escape(failed)}</td></tr></tbody></table>"
        "<table><thead><tr><th>Best Source Scene-Edge AUC</th><th>Best Aux-Edge AUC</th></tr></thead><tbody><tr>"
        f"<td><code>{html_lib.escape(str(best_scene.get('run_id', '')))}</code> {_fmt(best_scene.get('scene_edge_auc_low_predicts_removed_fp'))}</td>"
        f"<td><code>{html_lib.escape(str(best_edge.get('run_id', '')))}</code> {_fmt(best_edge.get('edge_auc_low_predicts_removed_fp'))}</td>"
        "</tr></tbody></table>"
        "<h3>CFA/LensPSF Proposal Checks</h3>"
        f"<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>"
        "<h3>Proposal Edge Bridge By Condition</h3>"
        "<table><thead><tr><th>Run</th><th>CFA</th><th>PSF</th><th>Samples</th><th>Removed FP</th><th>Removed TP</th><th>dFP</th><th>Aux Edge d/AUC</th><th>Scene Edge d/AUC</th><th>Report</th></tr></thead>"
        f"<tbody>{condition_rows}</tbody></table>"
    )


def _cfa_lenspsf_proposal_condition_row(row: Mapping[str, Any], destination: Path) -> str:
    report = str(row.get("report", ""))
    report_link = ""
    if report:
        html_path = Path(report).with_name("index.html")
        relative = os.path.relpath(str(html_path), start=str(destination))
        report_link = f"<a href=\"{html_lib.escape(relative)}\">open</a>"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('run_id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{int(row.get('sample_count', 0))}</td>"
        f"<td>{int(row.get('removed_fp_count', 0))}</td>"
        f"<td>{int(row.get('removed_tp_count', 0))}</td>"
        f"<td>{int(row.get('fp_delta_count', 0))}</td>"
        f"<td>{_fmt(row.get('edge_support_delta_removed_fp_minus_kept_tp'), signed=True)} / {_fmt(row.get('edge_auc_low_predicts_removed_fp'))}</td>"
        f"<td>{_fmt(row.get('scene_edge_support_delta_removed_fp_minus_kept_tp'), signed=True)} / {_fmt(row.get('scene_edge_auc_low_predicts_removed_fp'))}</td>"
        f"<td>{report_link}</td>"
        "</tr>"
    )


def _cfa_lenspsf_native_html(audit: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(audit.get("pass")) else "not_supported"
    failed = ", ".join(str(value) for value in audit.get("failed_checks", ())) or "none"
    group_rows = "".join(
        _cfa_lenspsf_native_group_row(name, group)
        for name, group in (audit.get("groups", {}) if isinstance(audit.get("groups"), Mapping) else {}).items()
        if isinstance(group, Mapping)
    )
    if not group_rows:
        group_rows = '<tr><td colspan="9">No native/remap groups were available.</td></tr>'
    run_rows = "".join(_cfa_lenspsf_native_run_row(row, destination) for row in audit.get("runs", ()))
    if not run_rows:
        run_rows = '<tr><td colspan="9">No native/remap run rows were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(audit.get('status', '')))}</code>. "
        f"{html_lib.escape(str(audit.get('interpretation', '')))} "
        f"{html_lib.escape(str(audit.get('claim_boundary', '')))}</p>"
        "<table><thead><tr><th>Report</th><th>Runs</th><th>CFA</th><th>PSF</th><th>Failed</th></tr></thead><tbody>"
        f"<tr><td>{_report_link(audit, destination)}</td>"
        f"<td>{int(audit.get('run_count', 0))}/{int(audit.get('expected_run_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in audit.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(_fmt(value) for value in audit.get('psf_sigmas', ()) if value is not None) or 'none')}</td>"
        f"<td>{html_lib.escape(failed)}</td></tr></tbody></table>"
        "<h3>Native/Remap Groups</h3>"
        "<table><thead><tr><th>Group</th><th>Runs</th><th>Samples</th><th>CFA</th><th>Mean dP50</th><th>Mean dR50</th><th>Mean dSmallR50</th><th>Mean dFP50</th><th>Best dFP50</th></tr></thead>"
        f"<tbody>{group_rows}</tbody></table>"
        "<h3>Native/Remap Runs</h3>"
        "<table><thead><tr><th>Run</th><th>Status</th><th>CFA</th><th>PSF</th><th>Samples</th><th>Remap</th><th>dP50</th><th>dR50</th><th>dFP50</th></tr></thead>"
        f"<tbody>{run_rows}</tbody></table>"
    )


def _cfa_lenspsf_native_group_row(name: str, group: Mapping[str, Any]) -> str:
    best = group.get("best_delta_fp@0.50", {}) if isinstance(group.get("best_delta_fp@0.50"), Mapping) else {}
    best_text = "" if not best else f"{best.get('run_id', '')} {_fmt(best.get('delta'), signed=True)}"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{int(group.get('run_count', 0))}</td>"
        f"<td>{int(group.get('sample_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in group.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{_fmt(group.get('mean_delta_precision@0.50'), signed=True)}</td>"
        f"<td>{_fmt(group.get('mean_delta_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(group.get('mean_delta_small_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(group.get('mean_delta_fp@0.50'), signed=True)}</td>"
        f"<td>{html_lib.escape(best_text)}</td>"
        "</tr>"
    )


def _cfa_lenspsf_native_run_row(row: Mapping[str, Any], destination: Path) -> str:
    link = html_lib.escape(str(row.get("run_id", "")))
    if row.get("html_path"):
        relative = os.path.relpath(str(row.get("html_path")), start=str(destination))
        link = f"<a href=\"{html_lib.escape(relative)}\">{html_lib.escape(str(row.get('run_id', '')))}</a>"
    return (
        "<tr>"
        f"<td>{link}</td>"
        f"<td><code>{html_lib.escape(str(row.get('native_status', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{int(row.get('sample_count', 0))}</td>"
        f"<td>{int(row.get('pattern_remapped_count', 0))} ({_fmt(row.get('pattern_remapped_fraction'))})</td>"
        f"<td>{_fmt(row.get('delta_precision@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_fp@0.50_mean'), signed=True)}</td>"
        "</tr>"
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
    reports = scene_edge.get("reports", ())
    report_rows = "".join(_scene_edge_report_row(row, destination) for row in reports if isinstance(row, Mapping))
    if not report_rows:
        report_rows = _scene_edge_report_row(scene_edge, destination)
    case_rows = "".join(_scene_edge_case_row(row) for row in scene_edge.get("cases", ()))
    if not case_rows:
        case_rows = '<tr><td colspan="9">No scene edge-confidence case rows were available.</td></tr>'
    return (
        f"<p>Status: <code class=\"{status_class}\">{html_lib.escape(str(scene_edge.get('status', '')))}</code>. "
        f"{html_lib.escape(str(scene_edge.get('interpretation', '')))} "
        f"{html_lib.escape(str(scene_edge.get('claim_boundary', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Cases</th><th>Checks</th><th>CFA</th><th>LensPSF</th><th>Failed</th><th>Human F1</th><th>Perception RGB F1</th><th>RGB Delta</th><th>RGB Win Rate</th><th>Aux Strength F1</th><th>Aux Strength Delta</th><th>Aux Strength Win Rate</th><th>Aux Confidence F1</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(scene_edge, destination)}</td>"
        f"<td>{int(scene_edge.get('case_count', 0))}</td>"
        f"<td>{int(scene_edge.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in scene_edge.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(_fmt(value) for value in scene_edge.get('psf_sigmas', ()) if value is not None) or 'none')}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        f"<td>{_fmt(scene_edge.get('human_rgb_proxy_source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_rgb_proxy_source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_rgb_minus_human_source_edge_f1_mean'), signed=True)}</td>"
        f"<td>{_fmt(scene_edge.get('perception_rgb_source_edge_f1_win_rate'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_aux_strength_source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_aux_strength_minus_human_source_edge_f1_mean'), signed=True)}</td>"
        f"<td>{_fmt(scene_edge.get('perception_aux_strength_source_edge_f1_win_rate'))}</td>"
        f"<td>{_fmt(scene_edge.get('perception_aux_confidence_source_edge_f1_mean'))}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Evidence Report</th><th>Report</th><th>Cases</th><th>Checks</th><th>CFA</th><th>LensPSF</th><th>RGB Delta</th><th>Aux Strength Delta</th><th>Status</th></tr></thead>"
        f"<tbody>{report_rows}</tbody></table>"
        "<table>"
        "<thead><tr><th>Case</th><th>Source</th><th>CFA</th><th>Human F1</th><th>Perception RGB F1</th><th>RGB Delta</th><th>Aux Strength F1</th><th>Aux Strength Delta</th><th>Aux Confidence F1</th></tr></thead>"
        f"<tbody>{case_rows}</tbody></table>"
    )


def _scene_edge_report_row(row: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(row.get("pass")) else "not_supported"
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('name', 'scene edge')))}</td>"
        f"<td>{_report_link(row, destination)}</td>"
        f"<td>{int(row.get('case_count', 0))}</td>"
        f"<td>{int(row.get('check_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(_fmt(value) for value in row.get('psf_sigmas', ()) if value is not None) or 'none')}</td>"
        f"<td>{_fmt(row.get('perception_rgb_minus_human_source_edge_f1_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('perception_aux_strength_minus_human_source_edge_f1_mean'), signed=True)}</td>"
        f"<td class=\"{status_class}\">{html_lib.escape(str(row.get('status', '')))}</td>"
        "</tr>"
    )


def _scene_edge_case_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('source', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(row.get('human_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_rgb_minus_human_source_edge_f1'), signed=True)}</td>"
        f"<td>{_fmt(row.get('perception_aux_strength_source_edge_f1'))}</td>"
        f"<td>{_fmt(row.get('perception_aux_strength_minus_human_source_edge_f1'), signed=True)}</td>"
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


def _as_path_specs(path: str | Path | Sequence[str | Path] | None) -> Tuple[str | Path, ...]:
    if path is None:
        return ()
    if isinstance(path, (str, Path)):
        return (path,)
    return tuple(path)


def _weighted_report_mean(reports: Sequence[Mapping[str, Any]], key: str) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for report in reports:
        value = report.get(key)
        if value is None:
            continue
        weight = max(float(report.get("case_count", 0)), 1.0)
        weighted_sum += float(value) * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return float(weighted_sum / weight_sum)


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
