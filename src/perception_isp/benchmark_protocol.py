"""Benchmark protocol coverage for defensible PerceptionISP claims."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


COMPARISON_SUMMARY = "comparison_summary.json"
COMPARISON_ROLLUP_SUMMARY = "rollup_summary.json"
TRAINING_ROLLUP_SUMMARY = "training_rollup_summary.json"
CLAIM_GATE_SUMMARY = "claim_gate_summary.json"
TASK_METRICS_SUMMARY = "task_metrics_summary.json"
CONDITION_METRICS_SUMMARY = "condition_metrics_summary.json"

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
    parser.add_argument("--condition-metrics", default=None, help="condition_metrics_summary.json path/dir.")
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="reports/perception_benchmark_protocol")
    args = parser.parse_args(argv)

    summary = build_protocol_coverage(
        comparison_reports=args.comparison_report,
        comparison_rollups=args.comparison_rollup,
        training_rollup=args.training_rollup,
        claim_gates=args.claim_gate,
        task_metrics=args.task_metrics,
        condition_metrics=args.condition_metrics,
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
    condition_metrics: str | Path | None = None,
    min_samples: int = 1000,
) -> Dict[str, Any]:
    evidence = _collect_evidence(
        comparison_reports=comparison_reports,
        comparison_rollups=comparison_rollups,
        training_rollup=training_rollup,
        claim_gates=claim_gates,
        task_metrics=task_metrics,
        condition_metrics=condition_metrics,
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
    condition_metrics: str | Path | None,
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
    condition = _load_condition_metrics(condition_metrics)

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
        "condition_metrics": condition,
    }


def _requirements(evidence: Mapping[str, Any], *, min_samples: int) -> list[Dict[str, Any]]:
    inputs = {str(value) for value in evidence.get("input_names", ())}
    training = evidence.get("training", {}) if isinstance(evidence.get("training"), Mapping) else {}
    features = evidence.get("run_config_features", {}) if isinstance(evidence.get("run_config_features"), Mapping) else {}
    consistency = evidence.get("run_config_consistency", {}) if isinstance(evidence.get("run_config_consistency"), Mapping) else {}
    gates = evidence.get("claim_gates", ()) if isinstance(evidence.get("claim_gates", ()), Sequence) else ()
    task = evidence.get("task_metrics", {}) if isinstance(evidence.get("task_metrics"), Mapping) else {}
    condition = evidence.get("condition_metrics", {}) if isinstance(evidence.get("condition_metrics"), Mapping) else {}

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
            "condition_metrics",
            "Condition-specific metrics available",
            "claim_required",
            bool(condition.get("available")),
            str(condition.get("summary", "missing")),
            "Condition-specific metrics are required before broad RAW/sensor-native claims because aggregate averages can hide low-light, HDR, weather, or focus regressions.",
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
    condition = evidence.get("condition_metrics", {}) if isinstance(evidence.get("condition_metrics"), Mapping) else {}
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
      <tr><th>Condition metrics</th><td>{_optional_link(condition, destination)} {html_lib.escape(str(condition.get('summary', 'missing')))}</td></tr>
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


if __name__ == "__main__":
    raise SystemExit(main())
