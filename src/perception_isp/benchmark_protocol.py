"""Benchmark protocol coverage for defensible PerceptionISP claims."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import json_ready


COMPARISON_SUMMARY = "comparison_summary.json"
COMPARISON_ROLLUP_SUMMARY = "rollup_summary.json"
TRAINING_ROLLUP_SUMMARY = "training_rollup_summary.json"
CLAIM_GATE_SUMMARY = "claim_gate_summary.json"
TASK_METRICS_SUMMARY = "task_metrics_summary.json"
TASK_GATE_SUMMARY = "task_gate_summary.json"
CONDITION_METRICS_SUMMARY = "condition_metrics_summary.json"
CONDITION_GATE_SUMMARY = "condition_gate_summary.json"
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

HUMAN_INPUTS = {"human_rgb"}
PERCEPTION_INPUTS = {
    "perception_rgb",
    "perception_fusion_rgb_aux",
    "perception_calibrated_fusion_rgb_aux",
    "perception_calibrated_score_label_fusion_rgb_aux",
    "perception_calibrated_score_label_aux_fusion_rgb_aux",
    "perception_rgb_aux_dnn",
}
AUX_INPUTS = {
    "perception_fusion_rgb_aux",
    "perception_calibrated_fusion_rgb_aux",
    "perception_calibrated_score_label_fusion_rgb_aux",
    "perception_calibrated_score_label_aux_fusion_rgb_aux",
    "perception_rgb_aux_dnn",
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether PerceptionISP evidence covers the minimum benchmark protocol.")
    parser.add_argument("--comparison-report", action="append", default=[], help="comparison_summary.json path/dir. Repeatable.")
    parser.add_argument("--comparison-rollup", action="append", default=[], help="rollup_summary.json path/dir. Repeatable.")
    parser.add_argument("--training-rollup", default=None, help="training_rollup_summary.json path/dir.")
    parser.add_argument("--claim-gate", action="append", default=[], help="claim_gate_summary.json path/dir. Repeatable.")
    parser.add_argument("--task-metrics", default=None, help="task_metrics_summary.json path/dir.")
    parser.add_argument("--task-gate", default=None, help="task_gate_summary.json path/dir.")
    parser.add_argument("--condition-metrics", default=None, help="condition_metrics_summary.json path/dir.")
    parser.add_argument("--condition-gate", default=None, help="condition_gate_summary.json path/dir.")
    parser.add_argument("--mechanism-validation", default=None, help="mechanism_validation_summary.json path/dir.")
    parser.add_argument("--cfa-stress-sweep", default=None, help="cfa_stress_sweep_summary.json path/dir.")
    parser.add_argument("--edge-confidence-suite", default=None, help="edge_confidence_suite_summary.json path/dir.")
    parser.add_argument("--edge-fidelity-suite", default=None, help="edge_fidelity_suite_summary.json path/dir.")
    parser.add_argument("--scene-edge-confidence", action="append", default=[], help="scene_edge_confidence_summary.json path/dir. Repeatable.")
    parser.add_argument("--scene-information-stress", default=None, help="scene_information_stress_summary.json path/dir.")
    parser.add_argument("--aux-contribution-audit", default=None, help="aux_contribution_audit_summary.json path/dir.")
    parser.add_argument("--cfa-lenspsf-detector-sweep", default=None, help="cfa_lenspsf_detector_sweep_summary.json path/dir.")
    parser.add_argument("--cfa-lenspsf-proposal-audit", default=None, help="cfa_lenspsf_proposal_audit_summary.json path/dir.")
    parser.add_argument("--cfa-lenspsf-native-audit", default=None, help="cfa_lenspsf_native_audit_summary.json path/dir.")
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="reports/perception_benchmark_protocol")
    args = parser.parse_args(argv)

    summary = build_protocol_coverage(
        comparison_reports=args.comparison_report,
        comparison_rollups=args.comparison_rollup,
        training_rollup=args.training_rollup,
        claim_gates=args.claim_gate,
        task_metrics=args.task_metrics,
        task_gate=args.task_gate,
        condition_metrics=args.condition_metrics,
        condition_gate=args.condition_gate,
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
        min_samples=int(args.min_samples),
    )
    html_path = write_protocol_coverage(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "status": summary["status"],
                    "coverage_status": summary["coverage_status"],
                    "metric_claim_status": summary["metric_claim_status"],
                    "missing_required": summary["missing_required"],
                    "missing_raw_claim": summary["missing_raw_claim"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_protocol_coverage(
    *,
    comparison_reports: Sequence[str | Path] = (),
    comparison_rollups: Sequence[str | Path] = (),
    training_rollup: str | Path | None = None,
    claim_gates: Sequence[str | Path] = (),
    task_metrics: str | Path | None = None,
    task_gate: str | Path | None = None,
    condition_metrics: str | Path | None = None,
    condition_gate: str | Path | None = None,
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
    min_samples: int = 1000,
) -> Dict[str, Any]:
    evidence = _collect_evidence(
        comparison_reports=comparison_reports,
        comparison_rollups=comparison_rollups,
        training_rollup=training_rollup,
        claim_gates=claim_gates,
        task_metrics=task_metrics,
        task_gate=task_gate,
        condition_metrics=condition_metrics,
        condition_gate=condition_gate,
        mechanism_validation=mechanism_validation,
        cfa_stress_sweep=cfa_stress_sweep,
        edge_confidence_suite=edge_confidence_suite,
        edge_fidelity_suite=edge_fidelity_suite,
        scene_edge_confidence=scene_edge_confidence,
        scene_information_stress=scene_information_stress,
        aux_contribution_audit=aux_contribution_audit,
        cfa_lenspsf_detector_sweep=cfa_lenspsf_detector_sweep,
        cfa_lenspsf_proposal_audit=cfa_lenspsf_proposal_audit,
        cfa_lenspsf_native_audit=cfa_lenspsf_native_audit,
    )
    requirements = _requirements(evidence, min_samples=int(min_samples))
    missing_required = [row["id"] for row in requirements if row["scope"] == "claim_required" and row["status"] != "covered"]
    missing_raw_claim = [row["id"] for row in requirements if row["scope"] == "raw_claim_required" and row["status"] != "covered"]
    coverage_complete = not missing_required and not missing_raw_claim
    coverage_status = "coverage_complete" if coverage_complete else "coverage_incomplete"
    # Keep the historical status string for compatibility with older reports and tests.
    status = "claim_ready" if coverage_complete else "not_claim_ready"
    gate_outcomes = _claim_gate_outcomes(evidence.get("claim_gates", ()))
    metric_claim_status = _metric_claim_status(gate_outcomes)
    return {
        "status": status,
        "coverage_status": coverage_status,
        "metric_claim_status": metric_claim_status,
        "claim_gate_outcomes": gate_outcomes,
        "min_samples": int(min_samples),
        "missing_required": missing_required,
        "missing_raw_claim": missing_raw_claim,
        "requirements": requirements,
        "evidence": evidence,
        "interpretation": _interpretation(
            coverage_status,
            metric_claim_status=metric_claim_status,
            missing_required=missing_required,
            missing_raw_claim=missing_raw_claim,
        ),
    }


def write_protocol_coverage(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "protocol_coverage_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _collect_evidence(
    *,
    comparison_reports: Sequence[str | Path],
    comparison_rollups: Sequence[str | Path],
    training_rollup: str | Path | None,
    claim_gates: Sequence[str | Path],
    task_metrics: str | Path | None,
    task_gate: str | Path | None,
    condition_metrics: str | Path | None,
    condition_gate: str | Path | None,
    mechanism_validation: str | Path | None,
    cfa_stress_sweep: str | Path | None,
    edge_confidence_suite: str | Path | None,
    edge_fidelity_suite: str | Path | None,
    scene_edge_confidence: str | Path | Sequence[str | Path] | None,
    scene_information_stress: str | Path | None,
    aux_contribution_audit: str | Path | None,
    cfa_lenspsf_detector_sweep: str | Path | None,
    cfa_lenspsf_proposal_audit: str | Path | None,
    cfa_lenspsf_native_audit: str | Path | None,
) -> Dict[str, Any]:
    input_names: set[str] = set()
    run_configs: list[Dict[str, Any]] = []
    sample_counts: list[int] = []
    label_agnostic_values: list[bool] = []
    comparison_paths: list[str] = []
    rollup_paths: list[str] = []

    for raw_path in comparison_reports:
        path = _summary_path(raw_path, COMPARISON_SUMMARY)
        report = json.loads(path.read_text())
        aggregate = report.get("aggregate", {}) if isinstance(report.get("aggregate"), Mapping) else {}
        input_names.update(str(name) for name in aggregate)
        sample_counts.append(_sample_count(report))
        run_config = report.get("run_config", {}) if isinstance(report.get("run_config"), Mapping) else {}
        run_configs.append(dict(run_config))
        if run_config.get("label_agnostic") is not None:
            label_agnostic_values.append(bool(run_config.get("label_agnostic")))
        comparison_paths.append(str(path))

    for raw_path in comparison_rollups:
        path = _summary_path(raw_path, COMPARISON_ROLLUP_SUMMARY)
        rollup = json.loads(path.read_text())
        for run in rollup.get("runs", ()):
            if not isinstance(run, Mapping):
                continue
            inputs = run.get("inputs", {}) if isinstance(run.get("inputs"), Mapping) else {}
            input_names.update(str(name) for name in inputs)
            sample_counts.append(int(run.get("sample_count", 0)))
            run_config = run.get("run_config", {}) if isinstance(run.get("run_config"), Mapping) else {}
            run_configs.append(dict(run_config))
            if run_config.get("label_agnostic") is not None:
                label_agnostic_values.append(bool(run_config.get("label_agnostic")))
        rollup_paths.append(str(path))

    training = _load_training(training_rollup)
    gates = [_load_gate(path) for path in claim_gates]
    task = _load_task_metrics(task_metrics)
    task_gate_data = _load_task_gate(task_gate)
    condition = _load_condition_metrics(condition_metrics)
    condition_gate_data = _load_condition_gate(condition_gate)
    mechanism = _load_mechanism_validation(mechanism_validation)
    cfa_stress = _load_cfa_stress_sweep(cfa_stress_sweep)
    edge_confidence = _load_edge_confidence_suite(edge_confidence_suite)
    edge_fidelity = _load_edge_fidelity_suite(edge_fidelity_suite)
    scene_edge = _load_scene_edge_confidence(scene_edge_confidence)
    scene_information = _load_scene_information_stress(scene_information_stress)
    aux_contribution = _load_aux_contribution_audit(aux_contribution_audit)
    cfa_lenspsf_detector = _load_cfa_lenspsf_detector_sweep(cfa_lenspsf_detector_sweep)
    cfa_lenspsf_proposal = _load_cfa_lenspsf_proposal_audit(cfa_lenspsf_proposal_audit)
    cfa_lenspsf_native = _load_cfa_lenspsf_native_audit(cfa_lenspsf_native_audit)

    return {
        "comparison_reports": comparison_paths,
        "comparison_rollups": rollup_paths,
        "input_names": sorted(input_names),
        "sample_count_max": max(sample_counts) if sample_counts else 0,
        "sample_count_total_reported": sum(sample_counts),
        "run_config_count": len(run_configs),
        "run_config_consistency": _run_config_consistency(run_configs),
        "run_config_features": _run_config_features(run_configs),
        "label_aware_present": any(value is False for value in label_agnostic_values),
        "training": training,
        "claim_gates": gates,
        "task_metrics": task,
        "task_gate": task_gate_data,
        "condition_metrics": condition,
        "condition_gate": condition_gate_data,
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
    }


def _requirements(evidence: Mapping[str, Any], *, min_samples: int) -> list[Dict[str, Any]]:
    inputs = {str(value) for value in evidence.get("input_names", ())}
    training = evidence.get("training", {}) if isinstance(evidence.get("training"), Mapping) else {}
    features = evidence.get("run_config_features", {}) if isinstance(evidence.get("run_config_features"), Mapping) else {}
    consistency = evidence.get("run_config_consistency", {}) if isinstance(evidence.get("run_config_consistency"), Mapping) else {}
    gates = evidence.get("claim_gates", ()) if isinstance(evidence.get("claim_gates", ()), Sequence) else ()
    task = evidence.get("task_metrics", {}) if isinstance(evidence.get("task_metrics"), Mapping) else {}
    task_gate = evidence.get("task_gate", {}) if isinstance(evidence.get("task_gate"), Mapping) else {}
    condition = evidence.get("condition_metrics", {}) if isinstance(evidence.get("condition_metrics"), Mapping) else {}
    condition_gate = evidence.get("condition_gate", {}) if isinstance(evidence.get("condition_gate"), Mapping) else {}
    mechanism = evidence.get("mechanism_validation", {}) if isinstance(evidence.get("mechanism_validation"), Mapping) else {}
    cfa_stress = evidence.get("cfa_stress_sweep", {}) if isinstance(evidence.get("cfa_stress_sweep"), Mapping) else {}
    edge_confidence = evidence.get("edge_confidence_suite", {}) if isinstance(evidence.get("edge_confidence_suite"), Mapping) else {}
    edge_fidelity = evidence.get("edge_fidelity_suite", {}) if isinstance(evidence.get("edge_fidelity_suite"), Mapping) else {}
    scene_edge = evidence.get("scene_edge_confidence", {}) if isinstance(evidence.get("scene_edge_confidence"), Mapping) else {}
    scene_information = evidence.get("scene_information_stress", {}) if isinstance(evidence.get("scene_information_stress"), Mapping) else {}
    aux_contribution = evidence.get("aux_contribution_audit", {}) if isinstance(evidence.get("aux_contribution_audit"), Mapping) else {}
    cfa_lenspsf_detector = evidence.get("cfa_lenspsf_detector_sweep", {}) if isinstance(evidence.get("cfa_lenspsf_detector_sweep"), Mapping) else {}
    cfa_lenspsf_proposal = evidence.get("cfa_lenspsf_proposal_audit", {}) if isinstance(evidence.get("cfa_lenspsf_proposal_audit"), Mapping) else {}
    cfa_lenspsf_native = evidence.get("cfa_lenspsf_native_audit", {}) if isinstance(evidence.get("cfa_lenspsf_native_audit"), Mapping) else {}

    return [
        _row(
            "paired_human_baseline",
            "Paired HumanISP RGB baseline",
            "claim_required",
            bool(inputs & HUMAN_INPUTS),
            f"inputs: {_matched(inputs, HUMAN_INPUTS)}",
            "A HumanISP RGB baseline from the same samples is required.",
        ),
        _row(
            "paired_perception_stream",
            "Paired PerceptionISP stream",
            "claim_required",
            bool(inputs & PERCEPTION_INPUTS),
            f"inputs: {_matched(inputs, PERCEPTION_INPUTS)}",
            "A PerceptionISP output from the same samples is required.",
        ),
        _row(
            "held_out_scale",
            f"Held-out sample scale >= {int(min_samples)}",
            "claim_required",
            int(evidence.get("sample_count_max", 0)) >= int(min_samples),
            f"max report sample_count: {int(evidence.get('sample_count_max', 0))}",
            "The largest available held-out report is too small for a broad claim.",
        ),
        _row(
            "fixed_detector_recipe",
            "Fixed detector and recipe across ablations",
            "claim_required",
            bool(consistency.get("stable")) and int(evidence.get("run_config_count", 0)) > 0,
            str(consistency.get("summary", "")),
            "Detector/model/confidence/label mode must be held fixed across compared inputs.",
        ),
        _row(
            "claim_gate_with_ci",
            "Claim gate with paired confidence intervals",
            "claim_required",
            _has_ci_gate(gates),
            _gate_evidence(gates),
            "A configured claim gate should require paired bootstrap CIs before promotion.",
        ),
        _row(
            "task_metrics",
            "Task-level metrics available",
            "claim_required",
            bool(task.get("available")),
            str(task.get("summary", "missing")),
            "Task-level groups are required before VRU/person/small-object claims.",
        ),
        _row(
            "task_gate",
            "Task-level gate evaluated",
            "claim_required",
            bool(task_gate.get("available")),
            str(task_gate.get("summary", "missing")),
            "Task-level metrics need a gate before promotion because raw group rows alone do not prove VRU/person/small-object claims.",
        ),
        _row(
            "condition_metrics",
            "Condition-specific metrics available",
            "claim_required",
            bool(condition.get("available")),
            str(condition.get("summary", "missing")),
            "Condition-specific metrics are required before broad RAW/sensor-native claims because aggregate averages can hide low-light, HDR, weather, or focus regressions.",
        ),
        _row(
            "condition_gate",
            "Condition robustness gate passed",
            "claim_required",
            bool(condition_gate.get("available")) and bool(condition_gate.get("pass")),
            str(condition_gate.get("summary", "missing")),
            "Condition-specific metrics need a robustness gate before promotion because coverage alone does not prove the target is acceptable on adverse slices.",
        ),
        _row(
            "front_end_mechanism_validation",
            "Front-end mechanism validation passed",
            "raw_claim_required",
            bool(mechanism.get("available")) and bool(mechanism.get("pass")),
            str(mechanism.get("summary", "missing")),
            "Sensor-native map claims need controlled low-light, glare/HDR, MTF, and CFA-response evidence before detector results are interpreted.",
        ),
        _row(
            "naive_raw_baseline",
            "Naive RAW or minimally adapted RAW baseline",
            "raw_claim_required",
            bool(features.get("naive_raw_like")),
            str(features.get("naive_raw_like_evidence", "")),
            "RAW/sensor-native claims need a naive RAW baseline because literature shows naive RAW can underperform badly.",
        ),
        _row(
            "classical_lightweight_transform",
            "Classical lightweight RAW transform baseline",
            "raw_claim_required",
            bool(features.get("classical_transform")),
            str(features.get("classical_transform_evidence", "")),
            "A demosaic + gamma/log style baseline is needed to separate ISP design from raw-data value.",
        ),
        _row(
            "task_aware_or_aux_adapter",
            "Task-aware adapter or aux-assisted path",
            "raw_claim_required",
            bool(inputs & AUX_INPUTS) or bool(training.get("trainable")),
            _adapter_evidence(inputs, training),
            "A perception-specific stream, fusion adapter, or learned RGB+aux path is needed.",
        ),
        _row(
            "aux_ablation",
            "RGB/RGB+aux/aux-only ablation",
            "recommended",
            _has_aux_ablation(training),
            _aux_ablation_evidence(training),
            "Channel ablations are recommended to prove aux maps add signal rather than only changing thresholds.",
        ),
        _row(
            "extended_sensor_native_tensor",
            "Extended sensor-native aux tensor exercised",
            "recommended",
            "rgb_aux_extended_chw" in {str(value) for value in training.get("tensor_keys", ())},
            f"tensor_keys: {', '.join(str(value) for value in training.get('tensor_keys', ())) or 'none'}",
            "The extended tensor should be included when claiming sensor-native side information.",
        ),
        _row(
            "cfa_stress_sweep",
            "CFA-dependent stress sweep available",
            "recommended",
            bool(cfa_stress.get("available")) and bool(cfa_stress.get("pass")),
            str(cfa_stress.get("summary", "missing")),
            "A CFA stress sweep helps separate sensor-native signal feasibility from detector-performance claims.",
        ),
        _row(
            "edge_confidence_suite",
            "Difficult-edge confidence suite available",
            "recommended",
            bool(edge_confidence.get("available")) and bool(edge_confidence.get("pass")),
            str(edge_confidence.get("summary", "missing")),
            "An edge-confidence suite helps show confidence maps react to low light, glare, and low-MTF stress before detector fine-tuning.",
        ),
        _row(
            "edge_fidelity_suite",
            "Object edge-fidelity suite available",
            "recommended",
            bool(edge_fidelity.get("available")) and bool(edge_fidelity.get("pass")),
            str(edge_fidelity.get("summary", "missing")),
            "An object edge-fidelity suite compares HumanISP, PerceptionISP, and aux edge maps against object/sensor edge oracles across CFA and LensPSF.",
        ),
        _row(
            "scene_edge_confidence",
            "High-information scene edge-confidence suite available",
            "recommended",
            bool(scene_edge.get("available")) and bool(scene_edge.get("pass")),
            str(scene_edge.get("summary", "missing")),
            "A scene edge-confidence suite compares HumanISP/PerceptionISP edge evidence against a higher-resolution scene-edge proxy after CameraE2E sampling.",
        ),
        _row(
            "scene_information_stress",
            "High-information scene stress suite available",
            "recommended",
            bool(scene_information.get("available")) and bool(scene_information.get("pass")),
            str(scene_information.get("summary", "missing")),
            "A scene-information stress suite verifies scene-to-sensor information loss and prevents overclaiming RGB-scene pass-through tests.",
        ),
        _row(
            "native_cfa_lenspsf_detector_sweep",
            "Native CFA/LensPSF detector sweep available",
            "raw_claim_required",
            bool(cfa_lenspsf_detector.get("available")) and bool(cfa_lenspsf_detector.get("native_clean")),
            str(cfa_lenspsf_detector.get("summary", "missing")),
            "RAW/sensor-native CFA claims need detector evidence with source CFA equal to target CFA, true CFA mosaics, and no bridge remapping.",
        ),
        _row(
            "native_cfa_lenspsf_audit",
            "Native CFA/LensPSF audit passed",
            "raw_claim_required",
            bool(cfa_lenspsf_native.get("available")) and bool(cfa_lenspsf_native.get("all_native")),
            str(cfa_lenspsf_native.get("summary", "missing")),
            "Native CFA claims need an audit that separates true native rows from remapped rows.",
        ),
        _row(
            "cfa_lenspsf_proposal_bridge",
            "CFA/LensPSF proposal-edge bridge available",
            "recommended",
            bool(cfa_lenspsf_proposal.get("available")) and bool(cfa_lenspsf_proposal.get("pass")),
            str(cfa_lenspsf_proposal.get("summary", "missing")),
            "A CFA/LensPSF proposal bridge is recommended to connect condition sweeps to proposal-level edge and scene-edge support.",
        ),
        _row(
            "aux_contribution_audit",
            "Aux contribution audit available",
            "recommended",
            bool(aux_contribution.get("available")) and bool(aux_contribution.get("pass")),
            str(aux_contribution.get("summary", "missing")),
            "An aux contribution audit checks whether aux features add proposal-scoring value beyond score/label calibration.",
        ),
    ]


def _row(requirement_id: str, label: str, scope: str, covered: bool, evidence: str, missing_reason: str) -> Dict[str, Any]:
    return {
        "id": requirement_id,
        "label": label,
        "scope": scope,
        "status": "covered" if covered else "missing",
        "evidence": evidence,
        "missing_reason": "" if covered else missing_reason,
    }


def _load_training(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "trainable": False, "summary": "missing"}
    summary_path = _summary_path(path, TRAINING_ROLLUP_SUMMARY)
    data = json.loads(summary_path.read_text())
    runs = [run for run in data.get("runs", ()) if isinstance(run, Mapping)]
    train_runs = [run for run in runs if str(run.get("kind", "")).startswith("train")]
    dense_eval_runs = [run for run in runs if str(run.get("kind", "")) == "dense_eval"]
    tensor_keys = sorted({str(run.get("tensor_key")) for run in runs if run.get("tensor_key")})
    channel_modes = sorted({str(run.get("channel_mode")) for run in runs if run.get("channel_mode")})
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "run_count": int(data.get("run_count", len(runs))),
        "trainable": bool(train_runs),
        "dense_eval_count": len(dense_eval_runs),
        "tensor_keys": tensor_keys,
        "channel_modes": channel_modes,
        "summary": f"{len(train_runs)} train runs, {len(dense_eval_runs)} dense eval runs",
    }


def _load_gate(path: str | Path) -> Dict[str, Any]:
    summary_path = _summary_path(path, CLAIM_GATE_SUMMARY)
    data = json.loads(summary_path.read_text())
    thresholds = data.get("thresholds", {}) if isinstance(data.get("thresholds"), Mapping) else {}
    return {
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "profile": str(data.get("profile", "")),
        "pass": bool(data.get("pass")),
        "require_ci": bool(thresholds.get("require_ci")),
        "sample_count": int(data.get("sample_count", 0)),
    }


def _load_task_metrics(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "summary": "missing"}
    summary_path = _summary_path(path, TASK_METRICS_SUMMARY)
    data = json.loads(summary_path.read_text())
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "input_count": len(data.get("inputs", ())),
        "group_count": len(data.get("groups", ())),
        "label_agnostic": bool(data.get("label_agnostic", True)),
        "summary": f"{len(data.get('groups', ()))} groups, {'label agnostic' if bool(data.get('label_agnostic', True)) else 'label aware'}",
    }


def _load_task_gate(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, TASK_GATE_SUMMARY)
    data = json.loads(summary_path.read_text())
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "profile": str(data.get("profile", "")),
        "pass": bool(data.get("pass")),
        "verdict": str(data.get("verdict", "")),
        "evaluated_group_count": int(data.get("evaluated_group_count", 0)),
        "failed_group_count": int(data.get("failed_group_count", 0)),
        "skipped_group_count": int(data.get("skipped_group_count", 0)),
        "summary": (
            f"{str(data.get('verdict', ''))}, profile={str(data.get('profile', ''))}, "
            f"evaluated={int(data.get('evaluated_group_count', 0))}, "
            f"failed={int(data.get('failed_group_count', 0))}, "
            f"skipped={int(data.get('skipped_group_count', 0))}"
        ),
    }


def _load_condition_metrics(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "summary": "missing"}
    summary_path = _summary_path(path, CONDITION_METRICS_SUMMARY)
    data = json.loads(summary_path.read_text())
    conditions = [row for row in data.get("conditions", ()) if isinstance(row, Mapping)]
    non_all = [row for row in conditions if str(row.get("name", "")) != "all"]
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "condition_count": len(conditions),
        "non_all_condition_count": len(non_all),
        "label_agnostic": bool(data.get("label_agnostic", True)),
        "summary": f"{len(conditions)} condition slices, {len(non_all)} non-all, {'label agnostic' if bool(data.get('label_agnostic', True)) else 'label aware'}",
    }


def _load_condition_gate(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, CONDITION_GATE_SUMMARY)
    data = json.loads(summary_path.read_text())
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "profile": str(data.get("profile", "")),
        "pass": bool(data.get("pass")),
        "verdict": str(data.get("verdict", "")),
        "evaluated_condition_count": int(data.get("evaluated_condition_count", 0)),
        "failed_condition_count": int(data.get("failed_condition_count", 0)),
        "skipped_condition_count": int(data.get("skipped_condition_count", 0)),
        "summary": (
            f"{str(data.get('verdict', ''))}, profile={str(data.get('profile', ''))}, "
            f"evaluated={int(data.get('evaluated_condition_count', 0))}, "
            f"failed={int(data.get('failed_condition_count', 0))}, "
            f"skipped={int(data.get('skipped_condition_count', 0))}"
        ),
    }


def _load_mechanism_validation(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, MECHANISM_VALIDATION_SUMMARY)
    data = json.loads(summary_path.read_text())
    mechanisms = [row for row in data.get("mechanisms", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in mechanisms if str(row.get("status", "")) != "pass"]
    cfas = ", ".join(str(value) for value in data.get("cfa_patterns", ()))
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "mechanism_count": len(mechanisms),
        "failed_mechanisms": failed,
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "summary": f"{status}, mechanisms={len(mechanisms)}, failed={len(failed)}, cfa={cfas or 'none'}",
    }


def _load_cfa_stress_sweep(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, CFA_STRESS_SWEEP_SUMMARY)
    data = json.loads(summary_path.read_text())
    support = data.get("support", {}) if isinstance(data.get("support"), Mapping) else {}
    rankings = [row for row in data.get("condition_rankings", ()) if isinstance(row, Mapping)]
    top = []
    for ranking in rankings:
        ranked = ranking.get("ranked_cfas", ()) if isinstance(ranking.get("ranked_cfas", ()), Sequence) else ()
        first = next((row for row in ranked if isinstance(row, Mapping)), None)
        if first is not None:
            top.append(f"{ranking.get('condition')}={first.get('cfa_pattern')}")
    status = str(data.get("status", ""))
    case_count = int(support.get("case_count", len(data.get("cases", ()))))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass",
        "status": status,
        "case_count": case_count,
        "condition_count": len(rankings),
        "top_cfas": top,
        "summary": f"{status}, cases={case_count}, top={', '.join(top) or 'none'}",
    }


def _load_edge_confidence_suite(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, EDGE_CONFIDENCE_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "case_count": len(data.get("cases", ())),
        "check_count": len(checks),
        "failed_checks": failed,
        "summary": f"{status}, cases={len(data.get('cases', ()))}, checks={len(checks)}, failed={len(failed)}",
    }


def _load_edge_fidelity_suite(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, EDGE_FIDELITY_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "case_count": len(data.get("cases", ())),
        "check_count": len(checks),
        "failed_checks": failed,
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in data.get("psf_sigmas", ())],
        "summary": (
            f"{status}, cases={len(data.get('cases', ()))}, checks={len(checks)}, failed={len(failed)}, "
            f"cfa={', '.join(str(value) for value in data.get('cfa_patterns', ())) or 'none'}, "
            f"psf={', '.join(str(value) for value in data.get('psf_sigmas', ())) or 'none'}"
        ),
    }


def _load_scene_edge_confidence(path: str | Path | Sequence[str | Path] | None) -> Dict[str, Any]:
    specs = _as_path_specs(path)
    if not specs:
        return {"available": False, "pass": False, "summary": "missing"}
    reports = [_load_scene_edge_confidence_one(spec) for spec in specs]
    case_count = sum(int(report.get("case_count", 0)) for report in reports)
    check_count = sum(int(report.get("check_count", 0)) for report in reports)
    failed = [str(value) for report in reports for value in report.get("failed_checks", ())]
    cfa_patterns = sorted({str(value) for report in reports for value in report.get("cfa_patterns", ())})
    psf_sigmas = sorted({float(value) for report in reports for value in report.get("psf_sigmas", ()) if value is not None})
    pass_all = all(bool(report.get("pass")) for report in reports)
    status = "pass" if pass_all else "fail"
    human_f1 = _weighted_report_mean(reports, "human_rgb_proxy_source_edge_f1_mean")
    perception_f1 = _weighted_report_mean(reports, "perception_rgb_proxy_source_edge_f1_mean")
    aux_strength_f1 = _weighted_report_mean(reports, "perception_aux_strength_source_edge_f1_mean")
    aux_confidence_f1 = _weighted_report_mean(reports, "perception_aux_confidence_source_edge_f1_mean")
    perception_delta = _weighted_report_mean(reports, "perception_rgb_minus_human_source_edge_f1_mean")
    aux_strength_delta = _weighted_report_mean(reports, "perception_aux_strength_minus_human_source_edge_f1_mean")
    perception_win_rate = _weighted_report_mean(reports, "perception_rgb_source_edge_f1_win_rate")
    aux_strength_win_rate = _weighted_report_mean(reports, "perception_aux_strength_source_edge_f1_win_rate")
    first = reports[0]
    return {
        "available": True,
        "summary_path": first.get("summary_path"),
        "html_path": first.get("html_path"),
        "pass": pass_all,
        "status": status,
        "report_count": len(reports),
        "case_count": case_count,
        "check_count": check_count,
        "failed_checks": failed,
        "cfa_patterns": cfa_patterns,
        "psf_sigmas": psf_sigmas,
        "human_rgb_proxy_source_edge_f1_mean": human_f1,
        "perception_rgb_proxy_source_edge_f1_mean": perception_f1,
        "perception_aux_strength_source_edge_f1_mean": aux_strength_f1,
        "perception_aux_confidence_source_edge_f1_mean": aux_confidence_f1,
        "perception_rgb_minus_human_source_edge_f1_mean": perception_delta,
        "perception_aux_strength_minus_human_source_edge_f1_mean": aux_strength_delta,
        "perception_rgb_source_edge_f1_win_rate": perception_win_rate,
        "perception_aux_strength_source_edge_f1_win_rate": aux_strength_win_rate,
        "reports": reports,
        "summary": (
            f"{status}, reports={len(reports)}, cases={case_count}, checks={check_count}, failed={len(failed)}, "
            f"cfa={', '.join(cfa_patterns) or 'none'}, "
            f"psf={', '.join(str(value) for value in psf_sigmas) or 'none'}, "
            f"humanF1={_fmt_optional(human_f1)}, "
            f"perRgbF1={_fmt_optional(perception_f1)}, "
            f"rgbDelta={_fmt_optional(perception_delta)}, "
            f"auxStrengthF1={_fmt_optional(aux_strength_f1)}, "
            f"auxDelta={_fmt_optional(aux_strength_delta)}"
        ),
    }


def _load_scene_edge_confidence_one(path: str | Path) -> Dict[str, Any]:
    summary_path = _summary_path(path, SCENE_EDGE_CONFIDENCE_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "case_count": len(data.get("cases", ())),
        "check_count": len(checks),
        "failed_checks": failed,
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in data.get("psf_sigmas", ())],
        "human_rgb_proxy_source_edge_f1_mean": _maybe_float(aggregate.get("human_rgb_proxy_source_edge_f1_mean")),
        "perception_rgb_proxy_source_edge_f1_mean": _maybe_float(aggregate.get("perception_rgb_proxy_source_edge_f1_mean")),
        "perception_aux_strength_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_strength_source_edge_f1_mean")),
        "perception_aux_confidence_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_confidence_source_edge_f1_mean")),
        "perception_rgb_minus_human_source_edge_f1_mean": _maybe_float(aggregate.get("perception_rgb_minus_human_source_edge_f1_mean")),
        "perception_aux_strength_minus_human_source_edge_f1_mean": _maybe_float(aggregate.get("perception_aux_strength_minus_human_source_edge_f1_mean")),
        "perception_rgb_source_edge_f1_win_rate": _maybe_float(aggregate.get("perception_rgb_source_edge_f1_win_rate")),
        "perception_aux_strength_source_edge_f1_win_rate": _maybe_float(aggregate.get("perception_aux_strength_source_edge_f1_win_rate")),
        "summary": (
            f"{status}, cases={len(data.get('cases', ()))}, checks={len(checks)}, failed={len(failed)}, "
            f"cfa={', '.join(str(value) for value in data.get('cfa_patterns', ())) or 'none'}, "
            f"psf={', '.join(str(value) for value in data.get('psf_sigmas', ())) or 'none'}, "
            f"humanF1={_fmt_optional(aggregate.get('human_rgb_proxy_source_edge_f1_mean'))}, "
            f"perRgbF1={_fmt_optional(aggregate.get('perception_rgb_proxy_source_edge_f1_mean'))}, "
            f"rgbDelta={_fmt_optional(aggregate.get('perception_rgb_minus_human_source_edge_f1_mean'))}, "
            f"auxStrengthF1={_fmt_optional(aggregate.get('perception_aux_strength_source_edge_f1_mean'))}, "
            f"auxDelta={_fmt_optional(aggregate.get('perception_aux_strength_minus_human_source_edge_f1_mean'))}"
        ),
    }


def _load_scene_information_stress(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, SCENE_INFORMATION_STRESS_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "case_count": len(data.get("cases", ())),
        "check_count": len(checks),
        "failed_checks": failed,
        "scene_width": int(data.get("scene_width", 0)),
        "scene_height": int(data.get("scene_height", 0)),
        "sensor_width": int(data.get("sensor_width", 0)),
        "sensor_height": int(data.get("sensor_height", 0)),
        "cfa_pattern": str(data.get("cfa_pattern", "")),
        "summary": (
            f"{status}, scene={int(data.get('scene_width', 0))}x{int(data.get('scene_height', 0))}, "
            f"sensor={int(data.get('sensor_width', 0))}x{int(data.get('sensor_height', 0))}, "
            f"cases={len(data.get('cases', ()))}, checks={len(checks)}, failed={len(failed)}"
        ),
    }


def _load_aux_contribution_audit(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, AUX_CONTRIBUTION_AUDIT_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    status = str(data.get("status", ""))
    feature_audit = data.get("feature_audit", {}) if isinstance(data.get("feature_audit"), Mapping) else {}
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "check_count": len(checks),
        "failed_checks": failed,
        "aux_feature_count": int(feature_audit.get("aux_feature_count", 0)),
        "summary": f"{status}, checks={len(checks)}, failed={len(failed)}, aux_features={int(feature_audit.get('aux_feature_count', 0))}",
    }


def _load_cfa_lenspsf_detector_sweep(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "native_clean": False, "summary": "missing"}
    summary_path = _summary_path(path, CFA_LENSPSF_DETECTOR_SWEEP_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    remap_values = []
    true_cfa_values = []
    sample_counts = []
    bridge_versions = set()
    camera_types = set()
    for run in data.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        raw_summary = run.get("raw_condition_summary", {}) if isinstance(run.get("raw_condition_summary"), Mapping) else {}
        remap = _maybe_float(raw_summary.get("pattern_remapped_fraction"))
        true_cfa = _maybe_float(raw_summary.get("true_sensor_cfa_mosaic_fraction"))
        if remap is not None:
            remap_values.append(remap)
        if true_cfa is not None:
            true_cfa_values.append(true_cfa)
        sample_count = _maybe_float(raw_summary.get("sample_count", run.get("sample_count")))
        if sample_count is not None:
            sample_counts.append(int(sample_count))
        camera_types.update(str(value) for value in (raw_summary.get("camerae2e_camera_types", {}) if isinstance(raw_summary.get("camerae2e_camera_types"), Mapping) else {}).keys())
        bridge_versions.update(str(value) for value in (raw_summary.get("camerae2e_native_cfa_bridge_versions", {}) if isinstance(raw_summary.get("camerae2e_native_cfa_bridge_versions"), Mapping) else {}).keys())
    max_remap = max(remap_values) if remap_values else None
    min_true_cfa = min(true_cfa_values) if true_cfa_values else None
    run_count = int(data.get("run_count", 0))
    count_per_condition = int(data.get("count", 0))
    total_samples = sum(sample_counts) if sample_counts else run_count * count_per_condition
    native_clean = (
        str(data.get("status", "")) == "pass"
        and not failed
        and max_remap == 0.0
        and min_true_cfa == 1.0
        and bool(bridge_versions)
    )
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "native_clean": native_clean,
        "status": str(data.get("status", "")),
        "run_count": run_count,
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "count": count_per_condition,
        "sample_count": total_samples,
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_maybe_float(value) for value in data.get("psf_sigmas", ())],
        "max_remap_fraction": max_remap,
        "min_true_cfa_fraction": min_true_cfa,
        "camerae2e_camera_types": sorted(camera_types),
        "camerae2e_native_cfa_bridge_versions": sorted(bridge_versions),
        "failed_checks": failed,
        "summary": (
            f"{str(data.get('status', ''))}, runs={run_count}/{int(data.get('expected_run_count', 0))}, "
            f"samples={total_samples}, size={int(data.get('width', 0))}x{int(data.get('height', 0))}, "
            f"max_remap={_fmt_optional(max_remap)}, min_true_cfa={_fmt_optional(min_true_cfa)}, "
            f"bridge={', '.join(sorted(bridge_versions)) or 'none'}"
        ),
    }


def _load_cfa_lenspsf_proposal_audit(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "summary": "missing"}
    summary_path = _summary_path(path, CFA_LENSPSF_PROPOSAL_AUDIT_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "status": status,
        "condition_count": int(data.get("condition_count", 0)),
        "expected_condition_count": int(data.get("expected_condition_count", 0)),
        "removed_fp_count": int(aggregate.get("removed_fp_count", 0)),
        "removed_tp_count": int(aggregate.get("removed_tp_count", 0)),
        "scene_edge_positive_condition_count": int(aggregate.get("scene_edge_positive_condition_count", 0)),
        "edge_positive_condition_count": int(aggregate.get("edge_positive_condition_count", 0)),
        "failed_checks": failed,
        "summary": (
            f"{status}, conditions={int(data.get('condition_count', 0))}/{int(data.get('expected_condition_count', 0))}, "
            f"removedFP={int(aggregate.get('removed_fp_count', 0))}, removedTP={int(aggregate.get('removed_tp_count', 0))}, "
            f"scenePositive={int(aggregate.get('scene_edge_positive_condition_count', 0))}, "
            f"edgePositive={int(aggregate.get('edge_positive_condition_count', 0))}"
        ),
    }


def _load_cfa_lenspsf_native_audit(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "all_native": False, "summary": "missing"}
    summary_path = _summary_path(path, CFA_LENSPSF_NATIVE_AUDIT_SUMMARY)
    data = json.loads(summary_path.read_text())
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [str(row.get("id", "")) for row in checks if str(row.get("status", "")) != "pass"]
    groups = data.get("groups", {}) if isinstance(data.get("groups"), Mapping) else {}
    native = groups.get("native", {}) if isinstance(groups.get("native"), Mapping) else {}
    remapped = groups.get("remapped", {}) if isinstance(groups.get("remapped"), Mapping) else {}
    partial = groups.get("partial_remap", {}) if isinstance(groups.get("partial_remap"), Mapping) else {}
    run_count = int(data.get("run_count", 0))
    native_run_count = int(native.get("run_count", 0))
    remapped_run_count = int(remapped.get("run_count", 0))
    partial_run_count = int(partial.get("run_count", 0))
    all_native = run_count > 0 and native_run_count == run_count and remapped_run_count == 0 and partial_run_count == 0
    status = str(data.get("status", ""))
    return {
        "available": True,
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "pass": status == "pass" and not failed,
        "all_native": status == "pass" and not failed and all_native,
        "status": status,
        "run_count": run_count,
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "native_run_count": native_run_count,
        "native_sample_count": int(native.get("sample_count", 0)),
        "remapped_run_count": remapped_run_count,
        "partial_run_count": partial_run_count,
        "cfa_patterns": [str(value) for value in native.get("cfa_patterns", ())],
        "failed_checks": failed,
        "summary": (
            f"{status}, native={native_run_count}/{run_count}, remapped={remapped_run_count}, partial={partial_run_count}, "
            f"samples={int(native.get('sample_count', 0))}, cfa={', '.join(str(value) for value in native.get('cfa_patterns', ())) or 'none'}"
        ),
    }


def _sample_count(report: Mapping[str, Any]) -> int:
    if report.get("sample_count") is not None:
        return int(report.get("sample_count", 0))
    aggregate = report.get("aggregate", {}) if isinstance(report.get("aggregate"), Mapping) else {}
    counts = [int(metrics.get("sample_count", 0)) for metrics in aggregate.values() if isinstance(metrics, Mapping)]
    return max(counts) if counts else 0


def _run_config_consistency(configs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not configs:
        return {"stable": False, "summary": "no run_config evidence"}
    keys = ("source", "dataset", "split", "rgb_detector", "rgb_detector_model", "rgb_detector_confidence", "label_agnostic")
    unstable = []
    for key in keys:
        values = {json.dumps(config.get(key), sort_keys=True) for config in configs if key in config}
        if len(values) > 1:
            unstable.append(key)
    if unstable:
        return {"stable": False, "summary": f"unstable keys: {', '.join(unstable)}"}
    detector = configs[0].get("rgb_detector_model", configs[0].get("rgb_detector", "unknown"))
    return {"stable": True, "summary": f"{len(configs)} run configs, detector {detector}"}


def _run_config_features(configs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    naive = []
    classical = []
    for config in configs:
        tone = str(config.get("tone_mapping", "")).lower()
        demosaic = str(config.get("demosaic_method", "")).lower()
        denoise = float(config.get("denoise_strength", 0.0) or 0.0)
        name = _config_name(config)
        if tone in {"linear", "raw", "none"} and demosaic in {"bilinear", "nearest", ""} and denoise <= 0.05:
            naive.append(name)
        if tone in {"log", "gamma", "detector_log", "srgb"} and demosaic in {"bilinear", "edge_aware", ""}:
            classical.append(name)
    return {
        "naive_raw_like": bool(naive),
        "naive_raw_like_evidence": ", ".join(naive),
        "classical_transform": bool(classical),
        "classical_transform_evidence": ", ".join(classical),
    }


def _config_name(config: Mapping[str, Any]) -> str:
    source = str(config.get("source", "report"))
    count = config.get("count")
    tone = str(config.get("tone_mapping", ""))
    demosaic = str(config.get("demosaic_method", ""))
    return f"{source} {count} tone={tone} demosaic={demosaic}" if count is not None else f"{source} tone={tone} demosaic={demosaic}"


def _has_ci_gate(gates: Sequence[Any]) -> bool:
    return any(isinstance(gate, Mapping) and bool(gate.get("require_ci")) for gate in gates)


def _gate_evidence(gates: Sequence[Any]) -> str:
    rows = [f"{gate.get('profile')} require_ci={gate.get('require_ci')}" for gate in gates if isinstance(gate, Mapping)]
    return ", ".join(rows) if rows else "missing"


def _claim_gate_outcomes(gates: Sequence[Any]) -> Dict[str, Any]:
    broad_gates = [gate for gate in gates if isinstance(gate, Mapping) and str(gate.get("profile", "")) == "broad_superiority"]
    fp_gates = [gate for gate in gates if isinstance(gate, Mapping) and str(gate.get("profile", "")) == "fp_reducer"]
    return {
        "broad_superiority_evaluated": bool(broad_gates),
        "broad_superiority_pass": any(bool(gate.get("pass")) for gate in broad_gates),
        "fp_reducer_evaluated": bool(fp_gates),
        "fp_reducer_pass": any(bool(gate.get("pass")) for gate in fp_gates),
        "ci_gate_present": _has_ci_gate(gates),
        "gate_count": len([gate for gate in gates if isinstance(gate, Mapping)]),
    }


def _metric_claim_status(outcomes: Mapping[str, Any]) -> str:
    if bool(outcomes.get("broad_superiority_pass")):
        return "broad_superiority_supported"
    if bool(outcomes.get("broad_superiority_evaluated")):
        if bool(outcomes.get("fp_reducer_pass")):
            return "fp_reducer_only"
        return "broad_superiority_not_supported"
    if bool(outcomes.get("fp_reducer_pass")):
        return "fp_reducer_only"
    if bool(outcomes.get("gate_count")):
        return "metric_claim_not_supported"
    return "metric_claim_not_evaluated"


def _has_aux_ablation(training: Mapping[str, Any]) -> bool:
    modes = {str(value) for value in training.get("channel_modes", ())}
    return {"rgb_aux", "rgb_only", "aux_only"}.issubset(modes)


def _aux_ablation_evidence(training: Mapping[str, Any]) -> str:
    modes = ", ".join(str(value) for value in training.get("channel_modes", ()))
    return f"channel_modes: {modes or 'none'}"


def _adapter_evidence(inputs: set[str], training: Mapping[str, Any]) -> str:
    matched = _matched(inputs, AUX_INPUTS)
    trainable = "trainable RGB+aux path" if bool(training.get("trainable")) else ""
    return ", ".join(value for value in (matched, trainable) if value) or "missing"


def _matched(inputs: set[str], candidates: set[str]) -> str:
    values = sorted(inputs & candidates)
    return ", ".join(values) if values else "none"


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = _path_from_spec(path)
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _path_from_spec(path: str | Path) -> Path:
    text = str(path)
    if "=" in text:
        _, raw_path = text.split("=", 1)
        return Path(raw_path).expanduser()
    return Path(path).expanduser()


def _sibling_html(summary_path: Path) -> str | None:
    html_path = summary_path.with_name("index.html")
    return str(html_path) if html_path.exists() else None


def _interpretation(
    coverage_status: str,
    *,
    metric_claim_status: str,
    missing_required: Sequence[str],
    missing_raw_claim: Sequence[str],
) -> str:
    if coverage_status == "coverage_complete":
        if metric_claim_status == "broad_superiority_supported":
            return "The provided evidence covers the configured benchmark protocol and the broad-superiority gate passed."
        if metric_claim_status == "fp_reducer_only":
            return (
                "The provided evidence covers the configured benchmark protocol, but the metric evidence supports only the narrower "
                "recall-budgeted FP-reduction claim, not broad HumanISP superiority."
            )
        if metric_claim_status == "broad_superiority_not_supported":
            return "The provided evidence covers the configured benchmark protocol, but the broad-superiority gate did not pass."
        return "The provided evidence covers the configured benchmark protocol. Metric claim support still depends on a passing claim gate."
    missing = list(missing_required) + list(missing_raw_claim)
    return "The provided evidence is not sufficient for a broad HumanISP-vs-PerceptionISP or RAW/sensor-native claim. Missing: " + ", ".join(missing)


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    rows = "".join(_requirement_row(row) for row in summary.get("requirements", ()))
    evidence = summary.get("evidence", {}) if isinstance(summary.get("evidence"), Mapping) else {}
    training = evidence.get("training", {}) if isinstance(evidence.get("training"), Mapping) else {}
    task = evidence.get("task_metrics", {}) if isinstance(evidence.get("task_metrics"), Mapping) else {}
    task_gate = evidence.get("task_gate", {}) if isinstance(evidence.get("task_gate"), Mapping) else {}
    condition = evidence.get("condition_metrics", {}) if isinstance(evidence.get("condition_metrics"), Mapping) else {}
    condition_gate = evidence.get("condition_gate", {}) if isinstance(evidence.get("condition_gate"), Mapping) else {}
    mechanism = evidence.get("mechanism_validation", {}) if isinstance(evidence.get("mechanism_validation"), Mapping) else {}
    cfa_stress = evidence.get("cfa_stress_sweep", {}) if isinstance(evidence.get("cfa_stress_sweep"), Mapping) else {}
    edge_confidence = evidence.get("edge_confidence_suite", {}) if isinstance(evidence.get("edge_confidence_suite"), Mapping) else {}
    edge_fidelity = evidence.get("edge_fidelity_suite", {}) if isinstance(evidence.get("edge_fidelity_suite"), Mapping) else {}
    scene_edge = evidence.get("scene_edge_confidence", {}) if isinstance(evidence.get("scene_edge_confidence"), Mapping) else {}
    scene_information = evidence.get("scene_information_stress", {}) if isinstance(evidence.get("scene_information_stress"), Mapping) else {}
    aux_contribution = evidence.get("aux_contribution_audit", {}) if isinstance(evidence.get("aux_contribution_audit"), Mapping) else {}
    cfa_lenspsf_detector = evidence.get("cfa_lenspsf_detector_sweep", {}) if isinstance(evidence.get("cfa_lenspsf_detector_sweep"), Mapping) else {}
    cfa_lenspsf_proposal = evidence.get("cfa_lenspsf_proposal_audit", {}) if isinstance(evidence.get("cfa_lenspsf_proposal_audit"), Mapping) else {}
    cfa_lenspsf_native = evidence.get("cfa_lenspsf_native_audit", {}) if isinstance(evidence.get("cfa_lenspsf_native_audit"), Mapping) else {}
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Benchmark Protocol Coverage</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .covered {{ color: #047857; font-weight: 700; }}
    .missing {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Benchmark Protocol Coverage</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <table>
    <thead><tr><th>Status</th><th>Scope</th><th>Requirement</th><th>Evidence</th><th>Missing Reason</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Evidence Summary</h2>
  <table>
    <tbody>
      <tr><th>Inputs</th><td>{html_lib.escape(', '.join(str(value) for value in evidence.get('input_names', ())) or 'none')}</td></tr>
      <tr><th>Coverage status</th><td><code>{html_lib.escape(str(summary.get('coverage_status', summary.get('status', ''))))}</code></td></tr>
      <tr><th>Metric claim status</th><td><code>{html_lib.escape(str(summary.get('metric_claim_status', 'unknown')))}</code></td></tr>
      <tr><th>Legacy status</th><td><code>{html_lib.escape(str(summary.get('status', '')))}</code></td></tr>
      <tr><th>Max samples</th><td>{int(evidence.get('sample_count_max', 0))}</td></tr>
      <tr><th>Run config consistency</th><td>{html_lib.escape(str(evidence.get('run_config_consistency', {}).get('summary', '')))}</td></tr>
      <tr><th>Claim gates</th><td>{html_lib.escape(_gate_outcome_text(summary.get('claim_gate_outcomes', {})))}</td></tr>
      <tr><th>Training</th><td>{_optional_link(training, destination)} {html_lib.escape(str(training.get('summary', 'missing')))}</td></tr>
      <tr><th>Task metrics</th><td>{_optional_link(task, destination)} {html_lib.escape(str(task.get('summary', 'missing')))}</td></tr>
      <tr><th>Task gate</th><td>{_optional_link(task_gate, destination)} {html_lib.escape(str(task_gate.get('summary', 'missing')))}</td></tr>
      <tr><th>Condition metrics</th><td>{_optional_link(condition, destination)} {html_lib.escape(str(condition.get('summary', 'missing')))}</td></tr>
      <tr><th>Condition gate</th><td>{_optional_link(condition_gate, destination)} {html_lib.escape(str(condition_gate.get('summary', 'missing')))}</td></tr>
      <tr><th>Mechanism validation</th><td>{_optional_link(mechanism, destination)} {html_lib.escape(str(mechanism.get('summary', 'missing')))}</td></tr>
      <tr><th>CFA stress sweep</th><td>{_optional_link(cfa_stress, destination)} {html_lib.escape(str(cfa_stress.get('summary', 'missing')))}</td></tr>
      <tr><th>Edge-confidence suite</th><td>{_optional_link(edge_confidence, destination)} {html_lib.escape(str(edge_confidence.get('summary', 'missing')))}</td></tr>
      <tr><th>Object edge-fidelity suite</th><td>{_optional_link(edge_fidelity, destination)} {html_lib.escape(str(edge_fidelity.get('summary', 'missing')))}</td></tr>
      <tr><th>Scene edge-confidence suite</th><td>{_optional_link(scene_edge, destination)} {html_lib.escape(str(scene_edge.get('summary', 'missing')))}</td></tr>
      <tr><th>Scene-information stress</th><td>{_optional_link(scene_information, destination)} {html_lib.escape(str(scene_information.get('summary', 'missing')))}</td></tr>
      <tr><th>CFA/LensPSF detector sweep</th><td>{_optional_link(cfa_lenspsf_detector, destination)} {html_lib.escape(str(cfa_lenspsf_detector.get('summary', 'missing')))}</td></tr>
      <tr><th>CFA/LensPSF native audit</th><td>{_optional_link(cfa_lenspsf_native, destination)} {html_lib.escape(str(cfa_lenspsf_native.get('summary', 'missing')))}</td></tr>
      <tr><th>CFA/LensPSF proposal bridge</th><td>{_optional_link(cfa_lenspsf_proposal, destination)} {html_lib.escape(str(cfa_lenspsf_proposal.get('summary', 'missing')))}</td></tr>
      <tr><th>Aux contribution audit</th><td>{_optional_link(aux_contribution, destination)} {html_lib.escape(str(aux_contribution.get('summary', 'missing')))}</td></tr>
    </tbody>
  </table>
  <p>Raw JSON: <code>protocol_coverage_summary.json</code></p>
</body>
</html>
"""


def _requirement_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('scope', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('label', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('missing_reason', '')))}</td>"
        "</tr>"
    )


def _gate_outcome_text(value: Any) -> str:
    outcomes = value if isinstance(value, Mapping) else {}
    return (
        f"broad_evaluated={bool(outcomes.get('broad_superiority_evaluated'))}, "
        f"broad_pass={bool(outcomes.get('broad_superiority_pass'))}, "
        f"fp_evaluated={bool(outcomes.get('fp_reducer_evaluated'))}, "
        f"fp_pass={bool(outcomes.get('fp_reducer_pass'))}, "
        f"ci_gate_present={bool(outcomes.get('ci_gate_present'))}"
    )


def _optional_link(item: Mapping[str, Any], destination: Path) -> str:
    html_path = item.get("html_path")
    if not html_path:
        return ""
    relative = os.path.relpath(str(html_path), start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _maybe_float(value: Any) -> float | None:
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


if __name__ == "__main__":
    raise SystemExit(main())
