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
    parser.add_argument("--comparison-rollup", action="append", default=[], help="Existing comparison rollup path/dir, optionally name=path.")
    parser.add_argument("--protocol-comparison-report", action="append", default=[], help="Additional comparison report used only for benchmark-protocol coverage, for example a naive RAW baseline.")
    parser.add_argument("--mechanism-validation", default=None, help="Mechanism validation summary path/dir used for RAW/sensor-native claim coverage.")
    parser.add_argument("--cfa-stress-sweep", default=None, help="CFA stress sweep summary path/dir used as diagnostic CFA evidence.")
    parser.add_argument("--edge-confidence-suite", default=None, help="Edge-confidence suite summary path/dir used as diagnostic difficult-edge evidence.")
    parser.add_argument("--aux-contribution-audit", default=None, help="Aux contribution audit summary path/dir used as diagnostic downstream aux evidence.")
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
        comparison_rollups=args.comparison_rollup,
        protocol_comparison_reports=args.protocol_comparison_report,
        mechanism_validation=args.mechanism_validation,
        cfa_stress_sweep=args.cfa_stress_sweep,
        edge_confidence_suite=args.edge_confidence_suite,
        aux_contribution_audit=args.aux_contribution_audit,
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
    comparison_rollups: Sequence[str | Path] = (),
    protocol_comparison_reports: Sequence[str | Path] = (),
    mechanism_validation: str | Path | None = None,
    cfa_stress_sweep: str | Path | None = None,
    edge_confidence_suite: str | Path | None = None,
    aux_contribution_audit: str | Path | None = None,
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
        aux_contribution_audit=aux_contribution_audit,
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
        task_metrics=task_metrics_dir,
        task_gate=task_gate_dir,
        protocol_coverage=protocol_dir,
        mechanism_validation=mechanism_validation,
        cfa_stress_sweep=cfa_stress_sweep,
        edge_confidence_suite=edge_confidence_suite,
        aux_contribution_audit=aux_contribution_audit,
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
        "aux_contribution_audit": _aux_contribution_audit_summary(aux_contribution_audit),
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
        "aux_contribution_audit": summary.get("aux_contribution_audit"),
        "benchmark_protocol": summary.get("benchmark_protocol"),
        "protocol_comparison_reports": summary.get("protocol_comparison_reports"),
        "decisions": summary.get("dashboard", {}).get("decisions") if isinstance(summary.get("dashboard"), Mapping) else [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
