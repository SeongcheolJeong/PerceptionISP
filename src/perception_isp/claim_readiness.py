"""One-shot claim-readiness report orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .aux_training_rollup import build_training_rollup, write_training_rollup
from .benchmark_protocol import build_protocol_coverage, write_protocol_coverage
from .claim_dashboard import build_claim_dashboard, write_claim_dashboard
from .claim_gate import build_claim_gate, write_claim_gate
from .condition_gate import build_condition_gate, write_condition_gate
from .condition_metrics import build_condition_metrics, write_condition_metrics
from .task_gate import build_task_gate, write_task_gate
from .task_metrics import build_task_metrics, write_task_metrics
from .types import json_ready


DEFAULT_TARGET_INPUT = "perception_calibrated_score_label_aux_fusion_rgb_aux"
DEFAULT_HUMAN_BASELINE_INPUT = "human_rgb"
DEFAULT_FUSION_BASELINE_INPUT = "perception_fusion_rgb_aux"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild PerceptionISP claim gates, optional training rollup, and dashboard in one command.")
    parser.add_argument("comparison_report", help="Comparison report directory or comparison_summary.json for the calibrated target.")
    parser.add_argument("--target-input", default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--human-baseline-input", default=DEFAULT_HUMAN_BASELINE_INPUT)
    parser.add_argument("--fusion-baseline-input", default=DEFAULT_FUSION_BASELINE_INPUT)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", default="claim_readiness")
    parser.add_argument("--no-require-ci", action="store_true", help="Do not require paired bootstrap CIs to satisfy each claim gate.")
    parser.add_argument("--training-summary", action="append", default=[], help="RGB+aux export/train/eval summary path/dir. Repeat to build a training rollup.")
    parser.add_argument("--training-rollup", default=None, help="Existing training rollup summary path/dir. Ignored when --training-summary is used.")
    parser.add_argument("--rgb-aux-dnn-gate", default=None, help="Existing RGB+Aux versus RGB-only DNN gate summary path/dir.")
    parser.add_argument("--rgb-aux-dnn-sweep", default=None, help="Existing RGB+Aux versus RGB-only DNN confidence sweep summary path/dir.")
    parser.add_argument("--dense-select-test-gate", default=None, help="Existing selection/test-separated RGB+Aux dense detector gate summary path/dir.")
    parser.add_argument("--dense-input-ablation-gate", default=None, help="Existing dense RGB+Aux DNN input-ablation gate summary path/dir.")
    parser.add_argument("--comparison-rollup", action="append", default=[], help="Existing comparison rollup path/dir, optionally name=path.")
    parser.add_argument("--protocol-comparison-report", action="append", default=[], help="Additional comparison report used only for benchmark-protocol coverage, for example a naive RAW baseline.")
    parser.add_argument("--mechanism-validation", default=None, help="Mechanism validation summary path/dir used for RAW/sensor-native claim coverage.")
    parser.add_argument("--cfa-stress-sweep", default=None, help="CFA stress sweep summary path/dir used as diagnostic CFA evidence.")
    parser.add_argument("--edge-confidence-suite", default=None, help="Edge-confidence suite summary path/dir used as diagnostic difficult-edge evidence.")
    parser.add_argument("--edge-fidelity-suite", default=None, help="Object edge-fidelity suite summary path/dir used as diagnostic CFA/LensPSF edge evidence.")
    parser.add_argument("--object-boundary-edge", default=None, help="Object-box-boundary edge summary path/dir used as diagnostic KITTI box-boundary evidence.")
    parser.add_argument("--object-boundary-detection-bridge", default=None, help="Object-box-boundary edge to detector TP/miss bridge summary path/dir.")
    parser.add_argument("--detector-box-support-audit", default=None, help="Detector-box FP/TP support audit summary path/dir.")
    parser.add_argument("--scene-edge-confidence", action="append", default=[], help="Scene-edge confidence summary path/dir used as high-information scene edge evidence. Repeatable.")
    parser.add_argument("--scene-information-stress", default=None, help="Scene-information stress summary path/dir used as diagnostic scene-to-sensor evidence.")
    parser.add_argument("--aux-contribution-audit", default=None, help="Aux contribution audit summary path/dir used as diagnostic downstream aux evidence.")
    parser.add_argument("--adverse-native-slice", default=None, help="Adverse-condition native RAW slice summary path/dir used as simulated adverse evidence.")
    parser.add_argument("--adverse-task-slice", default=None, help="Adverse-condition task-specific slice summary path/dir used as simulated task-slice evidence.")
    parser.add_argument("--cfa-lenspsf-detector-sweep", default=None, help="CFA/LensPSF detector sweep summary path/dir used as condition detector evidence.")
    parser.add_argument("--cfa-lenspsf-proposal-audit", default=None, help="CFA/LensPSF proposal-edge audit summary path/dir used as condition proposal bridge evidence.")
    parser.add_argument("--cfa-lenspsf-native-audit", default=None, help="CFA/LensPSF native-CFA separation audit summary path/dir used as native/remap boundary evidence.")
    parser.add_argument("--cfa-lenspsf-casebook", default=None, help="CFA/LensPSF visual casebook summary path/dir used as condition review evidence.")
    parser.add_argument("--cfa-lenspsf-aux-ablation", default=None, help="CFA/LensPSF score-label vs score-label-aux ablation summary path/dir.")
    parser.add_argument("--casebook", default=None, help="Success/failure casebook summary path/dir used as qualitative review evidence.")
    parser.add_argument("--output-dir", default="reports/perception_claim_readiness")
    args = parser.parse_args(argv)

    summary = run_claim_readiness(
        comparison_report=args.comparison_report,
        target_input=str(args.target_input),
        human_baseline_input=str(args.human_baseline_input),
        fusion_baseline_input=str(args.fusion_baseline_input),
        min_samples=int(args.min_samples),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_confidence=float(args.bootstrap_confidence),
        bootstrap_seed=str(args.bootstrap_seed),
        require_ci=not bool(args.no_require_ci),
        training_summaries=args.training_summary,
        training_rollup=args.training_rollup,
        rgb_aux_dnn_gate=args.rgb_aux_dnn_gate,
        rgb_aux_dnn_sweep=args.rgb_aux_dnn_sweep,
        dense_select_test_gate=args.dense_select_test_gate,
        dense_input_ablation_gate=args.dense_input_ablation_gate,
        comparison_rollups=args.comparison_rollup,
        protocol_comparison_reports=args.protocol_comparison_report,
        mechanism_validation=args.mechanism_validation,
        cfa_stress_sweep=args.cfa_stress_sweep,
        edge_confidence_suite=args.edge_confidence_suite,
        edge_fidelity_suite=args.edge_fidelity_suite,
        object_boundary_edge=args.object_boundary_edge,
        object_boundary_detection_bridge=args.object_boundary_detection_bridge,
        detector_box_support_audit=args.detector_box_support_audit,
        scene_edge_confidence=args.scene_edge_confidence,
        scene_information_stress=args.scene_information_stress,
        aux_contribution_audit=args.aux_contribution_audit,
        adverse_native_slice=args.adverse_native_slice,
        adverse_task_slice=args.adverse_task_slice,
        cfa_lenspsf_detector_sweep=args.cfa_lenspsf_detector_sweep,
        cfa_lenspsf_proposal_audit=args.cfa_lenspsf_proposal_audit,
        cfa_lenspsf_native_audit=args.cfa_lenspsf_native_audit,
        cfa_lenspsf_casebook=args.cfa_lenspsf_casebook,
        cfa_lenspsf_aux_ablation=args.cfa_lenspsf_aux_ablation,
        casebook=args.casebook,
        output_dir=args.output_dir,
    )
    print(json.dumps(json_ready(_compact_summary(summary)), indent=2))
    return 0


def run_claim_readiness(
    *,
    comparison_report: str | Path,
    target_input: str = DEFAULT_TARGET_INPUT,
    human_baseline_input: str = DEFAULT_HUMAN_BASELINE_INPUT,
    fusion_baseline_input: str = DEFAULT_FUSION_BASELINE_INPUT,
    min_samples: int = 1000,
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: str = "claim_readiness",
    require_ci: bool = True,
    training_summaries: Sequence[str | Path] = (),
    training_rollup: str | Path | None = None,
    rgb_aux_dnn_gate: str | Path | None = None,
    rgb_aux_dnn_sweep: str | Path | None = None,
    dense_select_test_gate: str | Path | None = None,
    dense_input_ablation_gate: str | Path | None = None,
    comparison_rollups: Sequence[str | Path] = (),
    protocol_comparison_reports: Sequence[str | Path] = (),
    mechanism_validation: str | Path | None = None,
    cfa_stress_sweep: str | Path | None = None,
    edge_confidence_suite: str | Path | None = None,
    edge_fidelity_suite: str | Path | None = None,
    object_boundary_edge: str | Path | None = None,
    object_boundary_detection_bridge: str | Path | None = None,
    detector_box_support_audit: str | Path | None = None,
    scene_edge_confidence: str | Path | Sequence[str | Path] | None = None,
    scene_information_stress: str | Path | None = None,
    aux_contribution_audit: str | Path | None = None,
    adverse_native_slice: str | Path | None = None,
    adverse_task_slice: str | Path | None = None,
    cfa_lenspsf_detector_sweep: str | Path | None = None,
    cfa_lenspsf_proposal_audit: str | Path | None = None,
    cfa_lenspsf_native_audit: str | Path | None = None,
    cfa_lenspsf_casebook: str | Path | None = None,
    cfa_lenspsf_aux_ablation: str | Path | None = None,
    casebook: str | Path | None = None,
    output_dir: str | Path = "reports/perception_claim_readiness",
) -> Dict[str, Any]:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = _comparison_summary_path(comparison_report)
    report = json.loads(report_path.read_text())

    broad_dir = destination / "broad_superiority_vs_human"
    broad_summary = build_claim_gate(
        report,
        target_input=str(target_input),
        baseline_input=str(human_baseline_input),
        thresholds=_claim_thresholds(
            profile="broad_superiority",
            min_samples=min_samples,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=f"{bootstrap_seed}:human",
            require_ci=require_ci,
        ),
        source_report=report_path,
    )
    broad_html = write_claim_gate(broad_summary, broad_dir)

    fp_dir = destination / "fp_reducer_vs_fusion"
    fp_summary = build_claim_gate(
        report,
        target_input=str(target_input),
        baseline_input=str(fusion_baseline_input),
        thresholds=_claim_thresholds(
            profile="fp_reducer",
            min_samples=min_samples,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=f"{bootstrap_seed}:fusion",
            require_ci=require_ci,
        ),
        source_report=report_path,
    )
    fp_html = write_claim_gate(fp_summary, fp_dir)

    task_metrics_dir = destination / "task_metrics"
    task_summary = build_task_metrics(
        report,
        source_report=report_path,
        baseline_input=str(human_baseline_input),
        inputs=tuple(dict.fromkeys((str(human_baseline_input), str(fusion_baseline_input), str(target_input)))),
    )
    task_html = write_task_metrics(task_summary, task_metrics_dir)

    task_gate_dir = destination / "task_gate"
    task_gate_summary = build_task_gate(
        task_summary,
        target_input=str(target_input),
        baseline_input=str(human_baseline_input),
        thresholds={"profile": "recall_improvement"},
        min_group_gt=1,
        source_report=task_html.parent / "task_metrics_summary.json",
    )
    task_gate_html = write_task_gate(task_gate_summary, task_gate_dir)

    condition_metrics_dir = destination / "condition_metrics"
    condition_summary = build_condition_metrics(
        report,
        source_report=report_path,
        baseline_input=str(human_baseline_input),
        inputs=tuple(dict.fromkeys((str(human_baseline_input), str(fusion_baseline_input), str(target_input)))),
    )
    condition_html = write_condition_metrics(condition_summary, condition_metrics_dir)

    condition_gate_dir = destination / "condition_gate"
    condition_gate_profile = "broad_superiority" if bool(broad_summary.get("pass")) else "fp_reducer"
    condition_min_samples = max(1, min(30, int(min_samples)))
    condition_gate_summary = build_condition_gate(
        condition_summary,
        target_input=str(target_input),
        baseline_input=str(human_baseline_input),
        thresholds={"profile": condition_gate_profile},
        min_condition_samples=condition_min_samples,
        source_report=condition_html.parent / "condition_metrics_summary.json",
    )
    condition_gate_html = write_condition_gate(condition_gate_summary, condition_gate_dir)

    training_rollup_path = None
    if training_summaries:
        training_summary = build_training_rollup(training_summaries)
        training_html = write_training_rollup(training_summary, destination / "rgb_aux_training_rollup")
        training_rollup_path = training_html.parent
    elif training_rollup is not None:
        training_rollup_path = Path(training_rollup).expanduser()

    protocol_dir = destination / "benchmark_protocol"
    protocol_reports = [report_path, *[Path(path).expanduser() for path in protocol_comparison_reports]]
    protocol_summary = build_protocol_coverage(
        comparison_reports=protocol_reports,
        comparison_rollups=comparison_rollups,
        training_rollup=training_rollup_path,
        claim_gates=[broad_dir, fp_dir],
        task_metrics=task_metrics_dir,
        task_gate=task_gate_dir,
        condition_metrics=condition_metrics_dir,
        condition_gate=condition_gate_dir,
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
        cfa_lenspsf_casebook=cfa_lenspsf_casebook,
        min_samples=int(min_samples),
    )
    protocol_html = write_protocol_coverage(protocol_summary, protocol_dir)

    dashboard_dir = destination / "dashboard"
    dashboard = build_claim_dashboard(
        claim_gate_specs=[
            f"Human broad superiority={broad_dir}",
            f"FP reducer vs RGB+Aux Fusion={fp_dir}",
        ],
        training_rollup=training_rollup_path,
        rgb_aux_dnn_gate=rgb_aux_dnn_gate,
        rgb_aux_dnn_sweep=rgb_aux_dnn_sweep,
        dense_select_test_gate=dense_select_test_gate,
        dense_input_ablation_gate=dense_input_ablation_gate,
        task_metrics=task_metrics_dir,
        task_gate=task_gate_dir,
        protocol_coverage=protocol_dir,
        mechanism_validation=mechanism_validation,
        cfa_stress_sweep=cfa_stress_sweep,
        edge_confidence_suite=edge_confidence_suite,
        edge_fidelity_suite=edge_fidelity_suite,
        object_boundary_edge=object_boundary_edge,
        object_boundary_detection_bridge=object_boundary_detection_bridge,
        detector_box_support_audit=detector_box_support_audit,
        scene_edge_confidence=scene_edge_confidence,
        scene_information_stress=scene_information_stress,
        aux_contribution_audit=aux_contribution_audit,
        adverse_native_slice=adverse_native_slice,
        adverse_task_slice=adverse_task_slice,
        cfa_lenspsf_detector_sweep=cfa_lenspsf_detector_sweep,
        cfa_lenspsf_proposal_audit=cfa_lenspsf_proposal_audit,
        cfa_lenspsf_native_audit=cfa_lenspsf_native_audit,
        cfa_lenspsf_casebook=cfa_lenspsf_casebook,
        cfa_lenspsf_aux_ablation=cfa_lenspsf_aux_ablation,
        casebook=casebook,
        comparison_rollup_specs=comparison_rollups,
    )
    dashboard_html = write_claim_dashboard(dashboard, dashboard_dir)

    summary = {
        "summary_json": str(destination / "claim_readiness_summary.json"),
        "comparison_report": str(report_path),
        "protocol_comparison_reports": [str(path) for path in protocol_reports],
        "target_input": str(target_input),
        "human_baseline_input": str(human_baseline_input),
        "fusion_baseline_input": str(fusion_baseline_input),
        "require_ci": bool(require_ci),
        "min_samples": int(min_samples),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_confidence": float(bootstrap_confidence),
        "bootstrap_seed": str(bootstrap_seed),
        "broad_superiority": {
            "report": str(broad_html),
            "summary_json": str(broad_html.parent / "claim_gate_summary.json"),
            "pass": bool(broad_summary.get("pass")),
            "verdict": broad_summary.get("verdict"),
            "failed": [item.get("metric") for item in broad_summary.get("criteria", ()) if not bool(item.get("pass"))],
        },
        "fp_reducer": {
            "report": str(fp_html),
            "summary_json": str(fp_html.parent / "claim_gate_summary.json"),
            "pass": bool(fp_summary.get("pass")),
            "verdict": fp_summary.get("verdict"),
            "failed": [item.get("metric") for item in fp_summary.get("criteria", ()) if not bool(item.get("pass"))],
        },
        "training_rollup": "" if training_rollup_path is None else str(training_rollup_path),
        "rgb_aux_dnn_gate": _rgb_aux_dnn_gate_summary(rgb_aux_dnn_gate),
        "rgb_aux_dnn_sweep": _rgb_aux_dnn_sweep_summary(rgb_aux_dnn_sweep),
        "dense_select_test_gate": _dense_select_test_gate_summary(dense_select_test_gate),
        "dense_input_ablation_gate": _dense_input_ablation_gate_summary(dense_input_ablation_gate),
        "task_metrics": {
            "report": str(task_html),
            "summary_json": str(task_html.parent / "task_metrics_summary.json"),
        },
        "task_gate": {
            "report": str(task_gate_html),
            "summary_json": str(task_gate_html.parent / "task_gate_summary.json"),
            "pass": bool(task_gate_summary.get("pass")),
            "verdict": task_gate_summary.get("verdict"),
            "profile": task_gate_summary.get("profile"),
            "failed_groups": [
                row.get("group")
                for row in task_gate_summary.get("groups", ())
                if isinstance(row, Mapping) and row.get("status") == "fail"
            ],
        },
        "condition_metrics": {
            "report": str(condition_html),
            "summary_json": str(condition_html.parent / "condition_metrics_summary.json"),
            "condition_count": len(condition_summary.get("conditions", ())),
        },
        "condition_gate": {
            "report": str(condition_gate_html),
            "summary_json": str(condition_gate_html.parent / "condition_gate_summary.json"),
            "pass": bool(condition_gate_summary.get("pass")),
            "verdict": condition_gate_summary.get("verdict"),
            "profile": condition_gate_summary.get("profile"),
            "min_condition_samples": condition_gate_summary.get("min_condition_samples"),
            "failed_conditions": [
                row.get("condition")
                for row in condition_gate_summary.get("conditions", ())
                if isinstance(row, Mapping) and row.get("status") == "fail"
            ],
        },
        "mechanism_validation": _mechanism_validation_summary(mechanism_validation),
        "cfa_stress_sweep": _cfa_stress_sweep_summary(cfa_stress_sweep),
        "edge_confidence_suite": _edge_confidence_suite_summary(edge_confidence_suite),
        "edge_fidelity_suite": _edge_fidelity_suite_summary(edge_fidelity_suite),
        "object_boundary_edge": _object_boundary_edge_summary(object_boundary_edge),
        "object_boundary_detection_bridge": _object_boundary_detection_bridge_summary(object_boundary_detection_bridge),
        "detector_box_support_audit": _detector_box_support_audit_summary(detector_box_support_audit),
        "scene_edge_confidence": _scene_edge_confidence_summary(scene_edge_confidence),
        "scene_information_stress": _scene_information_stress_summary(scene_information_stress),
        "aux_contribution_audit": _aux_contribution_audit_summary(aux_contribution_audit),
        "adverse_native_slice": _adverse_native_slice_summary(adverse_native_slice),
        "adverse_task_slice": _adverse_task_slice_summary(adverse_task_slice),
        "cfa_lenspsf_detector_sweep": _cfa_lenspsf_detector_sweep_summary(cfa_lenspsf_detector_sweep),
        "cfa_lenspsf_proposal_audit": _cfa_lenspsf_proposal_audit_summary(cfa_lenspsf_proposal_audit),
        "cfa_lenspsf_native_audit": _cfa_lenspsf_native_audit_summary(cfa_lenspsf_native_audit),
        "cfa_lenspsf_casebook": _cfa_lenspsf_casebook_summary(cfa_lenspsf_casebook),
        "cfa_lenspsf_aux_ablation": _cfa_lenspsf_aux_ablation_summary(cfa_lenspsf_aux_ablation),
        "casebook": _casebook_summary(casebook),
        "benchmark_protocol": {
            "report": str(protocol_html),
            "summary_json": str(protocol_html.parent / "protocol_coverage_summary.json"),
            "status": protocol_summary.get("status"),
            "coverage_status": protocol_summary.get("coverage_status"),
            "metric_claim_status": protocol_summary.get("metric_claim_status"),
            "claim_gate_outcomes": protocol_summary.get("claim_gate_outcomes"),
            "missing_required": protocol_summary.get("missing_required"),
            "missing_raw_claim": protocol_summary.get("missing_raw_claim"),
        },
        "dashboard": {
            "report": str(dashboard_html),
            "summary_json": str(dashboard_html.parent / "claim_dashboard_summary.json"),
            "decisions": dashboard.get("decisions", ()),
        },
    }
    (destination / "claim_readiness_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


def _claim_thresholds(
    *,
    profile: str,
    min_samples: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: str,
    require_ci: bool,
) -> Dict[str, Any]:
    return {
        "profile": str(profile),
        "min_samples": int(min_samples),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_confidence": float(bootstrap_confidence),
        "bootstrap_seed": str(bootstrap_seed),
        "require_ci": bool(require_ci),
    }


def _comparison_summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "comparison_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"comparison summary not found: {candidate}")
    return candidate


def _mechanism_validation_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "mechanism_validation_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"mechanism validation summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    mechanisms = [row for row in data.get("mechanisms", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in mechanisms if str(row.get("status", "")) != "pass"]
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_mechanisms": failed,
        "mechanism_count": len(mechanisms),
    }


def _cfa_stress_sweep_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "cfa_stress_sweep_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CFA stress sweep summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    support = data.get("support", {}) if isinstance(data.get("support"), Mapping) else {}
    rankings = [row for row in data.get("condition_rankings", ()) if isinstance(row, Mapping)]
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass",
        "status": data.get("status"),
        "case_count": int(support.get("case_count", len(data.get("cases", ())))),
        "condition_count": len(rankings),
    }


def _edge_confidence_suite_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "edge_confidence_suite_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"edge-confidence suite summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
    }


def _edge_fidelity_suite_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "edge_fidelity_suite_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"object edge-fidelity suite summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in data.get("psf_sigmas", ())],
    }


def _object_boundary_edge_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "object_boundary_edge_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"object-box-boundary edge summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": bool(data.get("pass")) and not failed,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "sample_count": int(data.get("sample_count", 0)),
        "box_count": int(data.get("box_count", 0)),
        "included_labels": [str(value) for value in data.get("included_labels", ())],
        "human_rgb_edge_boundary_f1_mean": _optional_float(aggregate.get("human_rgb_edge_boundary_f1_mean")),
        "perception_rgb_edge_boundary_f1_mean": _optional_float(aggregate.get("perception_rgb_edge_boundary_f1_mean")),
        "aux_edge_strength_boundary_f1_mean": _optional_float(aggregate.get("aux_edge_strength_boundary_f1_mean")),
        "aux_edge_confidence_boundary_f1_mean": _optional_float(aggregate.get("aux_edge_confidence_boundary_f1_mean")),
        "aux_confidence_minus_human_boundary_f1_mean": _optional_float(aggregate.get("aux_confidence_minus_human_boundary_f1_mean")),
        "aux_confidence_minus_human_boundary_f1_win_rate": _optional_float(aggregate.get("aux_confidence_minus_human_boundary_f1_win_rate")),
    }


def _object_boundary_detection_bridge_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "object_boundary_detection_bridge_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"object-boundary detection bridge summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": bool(data.get("pass")) and not failed,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "sample_count": int(data.get("sample_count", 0)),
        "object_count": int(data.get("object_count", 0)),
        "baseline_input": data.get("baseline_input"),
        "target_input": data.get("target_input"),
        "baseline_recall_proxy": _optional_float(aggregate.get("baseline_recall_proxy")),
        "target_recall_proxy": _optional_float(aggregate.get("target_recall_proxy")),
        "target_minus_baseline_recall_proxy": _optional_float(aggregate.get("target_minus_baseline_recall_proxy")),
        "target_only_detected_count": int(aggregate.get("target_only_detected_count", 0)),
        "baseline_only_detected_count": int(aggregate.get("baseline_only_detected_count", 0)),
    }


def _detector_box_support_audit_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "detector_box_support_audit_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"detector-box support audit summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    target = data.get("target_summary", {}) if isinstance(data.get("target_summary"), Mapping) else {}
    target_corr = target.get("correlations", {}) if isinstance(target.get("correlations"), Mapping) else {}
    bridge = data.get("transition_bridge", {}) if isinstance(data.get("transition_bridge"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": bool(data.get("pass")) and not failed,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "sample_count": int(data.get("sample_count", 0)),
        "target_input": data.get("target_input"),
        "target_detection_count": int(target.get("detection_count", 0)),
        "target_tp_count": int(target.get("tp_count", 0)),
        "target_fp_count": int(target.get("fp_count", 0)),
        "target_score_auc_low_fp": _detector_box_summary_auc(target_corr, "score"),
        "target_edge_auc_low_fp": _detector_box_summary_auc(target_corr, "edge_support"),
        "target_edge_delta_fp_minus_tp": _detector_box_summary_delta(target_corr, "edge_support"),
        "removed_fp_count": int(bridge.get("removed_fp_count", 0)),
        "removed_tp_count": int(bridge.get("removed_tp_count", 0)),
        "fp_delta_count": int(bridge.get("fp_delta_count", 0)),
        "tp_delta_count": int(bridge.get("tp_delta_count", 0)),
    }


def _detector_box_summary_auc(correlations: Mapping[str, Any], feature: str) -> float | None:
    row = _detector_box_summary_correlation(correlations, feature)
    return _optional_float(row.get("auc_low_feature_predicts_fp")) if row is not None else None


def _detector_box_summary_delta(correlations: Mapping[str, Any], feature: str) -> float | None:
    row = _detector_box_summary_correlation(correlations, feature)
    return _optional_float(row.get("delta_fp_minus_tp")) if row is not None else None


def _detector_box_summary_correlation(correlations: Mapping[str, Any], feature: str) -> Mapping[str, Any] | None:
    for row in correlations.get("rows", ()):
        if isinstance(row, Mapping) and str(row.get("feature", "")) == str(feature):
            return row
    return None


def _scene_edge_confidence_summary(path: str | Path | Sequence[str | Path] | None) -> Dict[str, Any]:
    specs = _as_path_specs(path)
    if not specs:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    reports = [_scene_edge_confidence_summary_one(spec) for spec in specs]
    case_count = sum(int(report.get("case_count", 0)) for report in reports)
    check_count = sum(int(report.get("check_count", 0)) for report in reports)
    failed = [str(value) for report in reports for value in report.get("failed_checks", ())]
    cfa_patterns = sorted({str(value) for report in reports for value in report.get("cfa_patterns", ())})
    psf_sigmas = sorted({float(value) for report in reports for value in report.get("psf_sigmas", ()) if value is not None})
    pass_all = all(bool(report.get("pass")) for report in reports)
    first = reports[0]
    return {
        "report": first.get("report", ""),
        "summary_json": first.get("summary_json", ""),
        "pass": pass_all,
        "status": "pass" if pass_all else "fail",
        "report_count": len(reports),
        "failed_checks": failed,
        "check_count": check_count,
        "case_count": case_count,
        "cfa_patterns": cfa_patterns,
        "psf_sigmas": psf_sigmas,
        "human_rgb_proxy_source_edge_f1_mean": _weighted_report_mean(reports, "human_rgb_proxy_source_edge_f1_mean"),
        "perception_rgb_proxy_source_edge_f1_mean": _weighted_report_mean(reports, "perception_rgb_proxy_source_edge_f1_mean"),
        "perception_aux_strength_source_edge_f1_mean": _weighted_report_mean(reports, "perception_aux_strength_source_edge_f1_mean"),
        "perception_aux_confidence_source_edge_f1_mean": _weighted_report_mean(reports, "perception_aux_confidence_source_edge_f1_mean"),
        "perception_rgb_minus_human_source_edge_f1_mean": _weighted_report_mean(reports, "perception_rgb_minus_human_source_edge_f1_mean"),
        "perception_aux_strength_minus_human_source_edge_f1_mean": _weighted_report_mean(reports, "perception_aux_strength_minus_human_source_edge_f1_mean"),
        "perception_aux_confidence_minus_human_source_edge_f1_mean": _weighted_report_mean(
            reports, "perception_aux_confidence_minus_human_source_edge_f1_mean"
        ),
        "perception_rgb_source_edge_f1_win_rate": _weighted_report_mean(reports, "perception_rgb_source_edge_f1_win_rate"),
        "perception_aux_strength_source_edge_f1_win_rate": _weighted_report_mean(reports, "perception_aux_strength_source_edge_f1_win_rate"),
        "perception_aux_confidence_source_edge_f1_win_rate": _weighted_report_mean(reports, "perception_aux_confidence_source_edge_f1_win_rate"),
        "reports": reports,
    }


def _scene_edge_confidence_summary_one(path: str | Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "scene_edge_confidence_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"scene-edge confidence summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in data.get("psf_sigmas", ())],
        "human_rgb_proxy_source_edge_f1_mean": _optional_float(aggregate.get("human_rgb_proxy_source_edge_f1_mean")),
        "perception_rgb_proxy_source_edge_f1_mean": _optional_float(aggregate.get("perception_rgb_proxy_source_edge_f1_mean")),
        "perception_aux_strength_source_edge_f1_mean": _optional_float(aggregate.get("perception_aux_strength_source_edge_f1_mean")),
        "perception_aux_confidence_source_edge_f1_mean": _optional_float(aggregate.get("perception_aux_confidence_source_edge_f1_mean")),
        "perception_rgb_minus_human_source_edge_f1_mean": _optional_float(aggregate.get("perception_rgb_minus_human_source_edge_f1_mean")),
        "perception_aux_strength_minus_human_source_edge_f1_mean": _optional_float(aggregate.get("perception_aux_strength_minus_human_source_edge_f1_mean")),
        "perception_aux_confidence_minus_human_source_edge_f1_mean": _optional_float(
            aggregate.get("perception_aux_confidence_minus_human_source_edge_f1_mean")
        ),
        "perception_rgb_source_edge_f1_win_rate": _optional_float(aggregate.get("perception_rgb_source_edge_f1_win_rate")),
        "perception_aux_strength_source_edge_f1_win_rate": _optional_float(aggregate.get("perception_aux_strength_source_edge_f1_win_rate")),
        "perception_aux_confidence_source_edge_f1_win_rate": _optional_float(aggregate.get("perception_aux_confidence_source_edge_f1_win_rate")),
    }


def _scene_information_stress_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "scene_information_stress_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"scene-information stress summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "case_count": len(data.get("cases", ())),
        "scene_width": int(data.get("scene_width", 0)),
        "scene_height": int(data.get("scene_height", 0)),
        "sensor_width": int(data.get("sensor_width", 0)),
        "sensor_height": int(data.get("sensor_height", 0)),
        "cfa_pattern": data.get("cfa_pattern"),
    }


def _aux_contribution_audit_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "aux_contribution_audit_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"aux contribution audit summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    feature_audit = data.get("feature_audit", {}) if isinstance(data.get("feature_audit"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "check_count": len(checks),
        "aux_feature_count": int(feature_audit.get("aux_feature_count", 0)),
    }


def _rgb_aux_dnn_gate_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "rgb_aux_dnn_gate_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"RGB+Aux DNN gate summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    criteria = [row for row in data.get("criteria", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in criteria if str(row.get("status", "")) != "pass"]
    deltas = data.get("deltas", {}) if isinstance(data.get("deltas"), Mapping) else {}
    primary_name = str(data.get("primary_run", "rgb_aux"))
    primary = next(
        (
            row
            for row in data.get("runs", ())
            if isinstance(row, Mapping) and str(row.get("name", "")) == primary_name
        ),
        {},
    )
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": bool(data.get("pass")),
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "profile": data.get("profile"),
        "sample_count": int((primary if isinstance(primary, Mapping) else {}).get("sample_count", 0)),
        "failed_criteria": failed,
        "delta_recall@0.50": _optional_float(deltas.get("recall@0.50_mean")),
        "delta_fp@0.50": _optional_float(deltas.get("fp@0.50_mean")),
    }


def _rgb_aux_dnn_sweep_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "rgb_aux_dnn_sweep_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"RGB+Aux DNN sweep summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    best = data.get("best_recall_positive_delta_row", {}) if isinstance(data.get("best_recall_positive_delta_row"), Mapping) else {}
    lowest = (
        data.get("lowest_fp_positive_recall_delta_row", {})
        if isinstance(data.get("lowest_fp_positive_recall_delta_row"), Mapping)
        else {}
    )
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": bool(data.get("pass")),
        "metric_pass": bool(data.get("metric_pass")),
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "profile": data.get("profile"),
        "row_count": int(data.get("row_count", 0)),
        "best_recall_confidence": _optional_float(best.get("confidence")),
        "lowest_fp_positive_confidence": _optional_float(lowest.get("confidence")),
    }


def _dense_select_test_gate_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"dense select/test gate summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    mean_test = data.get("mean_test_deltas", {}) if isinstance(data.get("mean_test_deltas"), Mapping) else {}
    seed_count = int(data.get("seed_count", 0))
    pass_count = int(data.get("pass_test_seed_count", 0))
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and seed_count > 0 and pass_count == seed_count,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "seed_count": seed_count,
        "pass_test_seed_count": pass_count,
        "selection_sample_count": int(data.get("selection_sample_count", 0)),
        "test_sample_count": int(data.get("test_sample_count", 0)),
        "delta_precision": _optional_float(mean_test.get("precision")),
        "delta_recall": _optional_float(mean_test.get("recall")),
        "delta_fp": _optional_float(mean_test.get("fp")),
        "delta_det_count": _optional_float(mean_test.get("det_count")),
    }


def _dense_input_ablation_gate_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "dense_input_ablation_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"dense input-ablation gate summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed_required = [
        row.get("id")
        for row in checks
        if bool(row.get("required")) and str(row.get("status", "")) != "pass"
    ]
    means = data.get("mean_by_mode", {}) if isinstance(data.get("mean_by_mode"), Mapping) else {}
    deltas = data.get("deltas_vs_none", {}) if isinstance(data.get("deltas_vs_none"), Mapping) else {}
    full = means.get("none", {}) if isinstance(means.get("none"), Mapping) else {}
    zero_aux_delta = deltas.get("zero_aux", {}) if isinstance(deltas.get("zero_aux"), Mapping) else {}
    zero_rgb_delta = deltas.get("zero_rgb", {}) if isinstance(deltas.get("zero_rgb"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed_required,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "seed_count": int(data.get("seed_count", 0)),
        "test_sample_count": int(data.get("test_sample_count", 0)),
        "failed_required_checks": failed_required,
        "full_precision": _optional_float(full.get("precision")),
        "full_recall": _optional_float(full.get("recall")),
        "full_fp": _optional_float(full.get("fp")),
        "zero_aux_delta_recall": _optional_float(zero_aux_delta.get("recall")),
        "zero_aux_delta_precision": _optional_float(zero_aux_delta.get("precision")),
        "zero_aux_delta_fp": _optional_float(zero_aux_delta.get("fp")),
        "zero_rgb_delta_recall": _optional_float(zero_rgb_delta.get("recall")),
    }


def _adverse_native_slice_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "adverse_native_slice_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"adverse native RAW slice summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) not in {"pass", "warning"}]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    native_count = 0
    remapped_count = 0
    for row in data.get("runs", ()):
        if not isinstance(row, Mapping):
            continue
        raw_summary = row.get("raw_condition_summary", {}) if isinstance(row.get("raw_condition_summary"), Mapping) else {}
        native_count += int(raw_summary.get("true_sensor_cfa_mosaic_count", 0))
        remapped_count += int(raw_summary.get("pattern_remapped_count", 0))
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "failed_checks": failed,
        "run_count": int(data.get("run_count", 0)),
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "sample_count": int(aggregate.get("sample_count", 0)),
        "adverse_condition_count": int(aggregate.get("adverse_condition_count", 0)),
        "adverse_fp_win_count": int(aggregate.get("adverse_fp_win_count", 0)),
        "adverse_recall_preserved_count": int(aggregate.get("adverse_recall_preserved_count", 0)),
        "mean_adverse_delta_recall@0.50": _optional_float(aggregate.get("mean_adverse_delta_recall@0.50")),
        "mean_adverse_delta_fp@0.50": _optional_float(aggregate.get("mean_adverse_delta_fp@0.50")),
        "cfa_pattern": data.get("cfa_pattern"),
        "native_count": native_count,
        "remapped_count": remapped_count,
    }


def _adverse_task_slice_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "adverse_task_slice_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"adverse task slice summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) not in {"pass", "warning"}]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "failed_checks": failed,
        "profile": data.get("profile"),
        "condition_count": int(data.get("condition_count", 0)),
        "expected_condition_count": int(data.get("expected_condition_count", 0)),
        "adverse_condition_count": int(aggregate.get("adverse_condition_count", 0)),
        "adverse_passed_condition_count": int(aggregate.get("adverse_passed_condition_count", 0)),
        "adverse_failed_condition_count": int(aggregate.get("adverse_failed_condition_count", 0)),
        "failed_group_count": int(aggregate.get("failed_group_count", 0)),
        "skipped_group_count": int(aggregate.get("skipped_group_count", 0)),
        "cfa_pattern": data.get("cfa_pattern"),
    }


def _cfa_lenspsf_detector_sweep_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "cfa_lenspsf_detector_sweep_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CFA/LensPSF detector sweep summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "run_count": int(data.get("run_count", 0)),
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "count": int(data.get("count", 0)),
        "cfa_patterns": [str(value) for value in data.get("cfa_patterns", ())],
        "psf_sigmas": [_optional_float(value) for value in data.get("psf_sigmas", ())],
    }


def _cfa_lenspsf_proposal_audit_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "cfa_lenspsf_proposal_audit_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CFA/LensPSF proposal audit summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "condition_count": int(data.get("condition_count", 0)),
        "expected_condition_count": int(data.get("expected_condition_count", 0)),
        "removed_fp_count": int(aggregate.get("removed_fp_count", 0)),
        "removed_tp_count": int(aggregate.get("removed_tp_count", 0)),
        "fp_delta_count": int(aggregate.get("fp_delta_count", 0)),
        "scene_edge_positive_condition_count": int(aggregate.get("scene_edge_positive_condition_count", 0)),
        "edge_positive_condition_count": int(aggregate.get("edge_positive_condition_count", 0)),
        "scene_edge_auc_condition_mean": _optional_float(aggregate.get("scene_edge_auc_condition_mean")),
        "edge_auc_condition_mean": _optional_float(aggregate.get("edge_auc_condition_mean")),
        "scene_edge_support_delta_condition_mean": _optional_float(aggregate.get("scene_edge_support_delta_condition_mean")),
        "edge_support_delta_condition_mean": _optional_float(aggregate.get("edge_support_delta_condition_mean")),
    }


def _cfa_lenspsf_native_audit_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "cfa_lenspsf_native_audit_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CFA/LensPSF native audit summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) not in {"pass", "warning"}]
    groups = data.get("groups", {}) if isinstance(data.get("groups"), Mapping) else {}
    native = groups.get("native", {}) if isinstance(groups.get("native"), Mapping) else {}
    remapped = groups.get("remapped", {}) if isinstance(groups.get("remapped"), Mapping) else {}
    partial = groups.get("partial_remap", {}) if isinstance(groups.get("partial_remap"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "run_count": int(data.get("run_count", 0)),
        "expected_run_count": int(data.get("expected_run_count", 0)),
        "native_run_count": int(native.get("run_count", 0)),
        "native_sample_count": int(native.get("sample_count", 0)),
        "native_cfa_patterns": [str(value) for value in native.get("cfa_patterns", ())],
        "remapped_run_count": int(remapped.get("run_count", 0)),
        "remapped_sample_count": int(remapped.get("sample_count", 0)),
        "remapped_cfa_patterns": [str(value) for value in remapped.get("cfa_patterns", ())],
        "partial_remap_run_count": int(partial.get("run_count", 0)),
    }


def _cfa_lenspsf_casebook_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "cfa_lenspsf_casebook_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CFA/LensPSF casebook summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    category_totals = data.get("category_totals", {}) if isinstance(data.get("category_totals"), Mapping) else {}
    success = category_totals.get("fp_reduction_success", {}) if isinstance(category_totals.get("fp_reduction_success"), Mapping) else {}
    counterexamples = sum(
        int((category_totals.get(name, {}) if isinstance(category_totals.get(name), Mapping) else {}).get("selected_case_count", 0))
        for name in ("recall_tradeoff", "recall_loss_failure", "fp_regression_failure")
    )
    native_condition_count = sum(
        1
        for row in data.get("conditions", ())
        if isinstance(row, Mapping)
        and row.get("pattern_remapped_fraction") == 0.0
        and row.get("true_sensor_cfa_mosaic_fraction") == 1.0
    )
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "condition_count": int(data.get("condition_count", 0)),
        "expected_condition_count": int(data.get("expected_condition_count", 0)),
        "selected_case_count": int(data.get("selected_case_count", 0)),
        "selected_fp_reduction_success_count": int(success.get("selected_case_count", 0)),
        "selected_counterexample_count": counterexamples,
        "native_condition_count": native_condition_count,
    }


def _cfa_lenspsf_aux_ablation_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "cfa_lenspsf_aux_ablation_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CFA/LensPSF aux ablation summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) not in {"pass", "warning"}]
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "claim_status": data.get("claim_status"),
        "failed_checks": failed,
        "condition_count": int(data.get("condition_count", 0)),
        "expected_condition_count": int(data.get("expected_condition_count", 0)),
        "sample_count": int(aggregate.get("sample_count", 0)),
        "aux_recall_win_count": int(aggregate.get("aux_recall_win_count", 0)),
        "aux_fp_win_count": int(aggregate.get("aux_fp_win_count", 0)),
        "mean_aux_minus_no_aux_recall@0.50": _optional_float(aggregate.get("mean_aux_minus_no_aux_recall@0.50")),
        "mean_aux_minus_no_aux_fp@0.50": _optional_float(aggregate.get("mean_aux_minus_no_aux_fp@0.50")),
    }


def _casebook_summary(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"report": "", "summary_json": "", "pass": False, "status": "missing"}
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "casebook_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"casebook summary not found: {candidate}")
    data = json.loads(candidate.read_text())
    html_path = candidate.with_name("index.html")
    checks = [row for row in data.get("checks", ()) if isinstance(row, Mapping)]
    failed = [row.get("id") for row in checks if str(row.get("status", "")) != "pass"]
    categories = data.get("categories", {}) if isinstance(data.get("categories"), Mapping) else {}
    return {
        "report": str(html_path) if html_path.exists() else "",
        "summary_json": str(candidate),
        "pass": str(data.get("status", "")) == "pass" and not failed,
        "status": data.get("status"),
        "failed_checks": failed,
        "sample_count": int(data.get("sample_count", 0)),
        "selected_case_count": int(data.get("selected_case_count", 0)),
        "fp_reduction_success_count": int((categories.get("fp_reduction_success", {}) if isinstance(categories.get("fp_reduction_success"), Mapping) else {}).get("case_count", 0)),
        "recall_tradeoff_count": int((categories.get("recall_tradeoff", {}) if isinstance(categories.get("recall_tradeoff"), Mapping) else {}).get("case_count", 0)),
        "recall_loss_failure_count": int((categories.get("recall_loss_failure", {}) if isinstance(categories.get("recall_loss_failure"), Mapping) else {}).get("case_count", 0)),
        "fp_regression_failure_count": int((categories.get("fp_regression_failure", {}) if isinstance(categories.get("fp_regression_failure"), Mapping) else {}).get("case_count", 0)),
    }


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "summary_json": summary.get("summary_json"),
        "dashboard": summary.get("dashboard", {}).get("report") if isinstance(summary.get("dashboard"), Mapping) else "",
        "broad_superiority": summary.get("broad_superiority"),
        "fp_reducer": summary.get("fp_reducer"),
        "task_metrics": summary.get("task_metrics"),
        "task_gate": summary.get("task_gate"),
        "condition_metrics": summary.get("condition_metrics"),
        "condition_gate": summary.get("condition_gate"),
        "mechanism_validation": summary.get("mechanism_validation"),
        "cfa_stress_sweep": summary.get("cfa_stress_sweep"),
        "edge_confidence_suite": summary.get("edge_confidence_suite"),
        "edge_fidelity_suite": summary.get("edge_fidelity_suite"),
        "object_boundary_edge": summary.get("object_boundary_edge"),
        "object_boundary_detection_bridge": summary.get("object_boundary_detection_bridge"),
        "detector_box_support_audit": summary.get("detector_box_support_audit"),
        "scene_edge_confidence": summary.get("scene_edge_confidence"),
        "scene_information_stress": summary.get("scene_information_stress"),
        "aux_contribution_audit": summary.get("aux_contribution_audit"),
        "rgb_aux_dnn_gate": summary.get("rgb_aux_dnn_gate"),
        "rgb_aux_dnn_sweep": summary.get("rgb_aux_dnn_sweep"),
        "dense_select_test_gate": summary.get("dense_select_test_gate"),
        "dense_input_ablation_gate": summary.get("dense_input_ablation_gate"),
        "adverse_native_slice": summary.get("adverse_native_slice"),
        "adverse_task_slice": summary.get("adverse_task_slice"),
        "cfa_lenspsf_detector_sweep": summary.get("cfa_lenspsf_detector_sweep"),
        "cfa_lenspsf_proposal_audit": summary.get("cfa_lenspsf_proposal_audit"),
        "cfa_lenspsf_native_audit": summary.get("cfa_lenspsf_native_audit"),
        "cfa_lenspsf_casebook": summary.get("cfa_lenspsf_casebook"),
        "cfa_lenspsf_aux_ablation": summary.get("cfa_lenspsf_aux_ablation"),
        "casebook": summary.get("casebook"),
        "benchmark_protocol": summary.get("benchmark_protocol"),
        "protocol_comparison_reports": summary.get("protocol_comparison_reports"),
        "decisions": summary.get("dashboard", {}).get("decisions") if isinstance(summary.get("dashboard"), Mapping) else [],
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_path_specs(path: str | Path | Sequence[str | Path] | None) -> tuple[str | Path, ...]:
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
