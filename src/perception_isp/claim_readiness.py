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

    training_rollup_path = None
    if training_summaries:
        training_summary = build_training_rollup(training_summaries)
        training_html = write_training_rollup(training_summary, destination / "rgb_aux_training_rollup")
        training_rollup_path = training_html.parent
    elif training_rollup is not None:
        training_rollup_path = Path(training_rollup).expanduser()

    protocol_dir = destination / "benchmark_protocol"
    protocol_summary = build_protocol_coverage(
        comparison_reports=[report_path],
        comparison_rollups=comparison_rollups,
        training_rollup=training_rollup_path,
        claim_gates=[broad_dir, fp_dir],
        task_metrics=task_metrics_dir,
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
        protocol_coverage=protocol_dir,
        comparison_rollup_specs=comparison_rollups,
    )
    dashboard_html = write_claim_dashboard(dashboard, dashboard_dir)

    summary = {
        "summary_json": str(destination / "claim_readiness_summary.json"),
        "comparison_report": str(report_path),
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
        "benchmark_protocol": {
            "report": str(protocol_html),
            "summary_json": str(protocol_html.parent / "protocol_coverage_summary.json"),
            "status": protocol_summary.get("status"),
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


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "summary_json": summary.get("summary_json"),
        "dashboard": summary.get("dashboard", {}).get("report") if isinstance(summary.get("dashboard"), Mapping) else "",
        "broad_superiority": summary.get("broad_superiority"),
        "fp_reducer": summary.get("fp_reducer"),
        "task_metrics": summary.get("task_metrics"),
        "benchmark_protocol": summary.get("benchmark_protocol"),
        "decisions": summary.get("dashboard", {}).get("decisions") if isinstance(summary.get("dashboard"), Mapping) else [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
