"""Metric gate for HumanISP-vs-PerceptionISP claim readiness."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .types import json_ready


CRITERIA = (
    ("precision@0.50_mean", "minimum_delta", "min_precision_delta", 0.0),
    ("recall@0.50_mean", "minimum_delta", "min_recall_delta", 0.0),
    ("recall@0.75_mean", "minimum_delta", "min_recall75_delta", 0.0),
    ("small_recall@0.50_mean", "minimum_delta", "min_small_recall_delta", 0.0),
    ("fp@0.50_mean", "maximum_delta", "max_fp_delta", 0.0),
)

SAMPLE_METRIC_MAP = {
    "precision@0.50_mean": "precision@0.50",
    "recall@0.50_mean": "recall@0.50",
    "recall@0.75_mean": "recall@0.75",
    "small_recall@0.50_mean": "small_recall@0.50",
    "fp@0.50_mean": "fp@0.50",
}

CLAIM_PROFILES = {
    "broad_superiority": {
        "min_precision_delta": 0.0,
        "min_recall_delta": 0.0,
        "min_recall75_delta": 0.0,
        "min_small_recall_delta": 0.0,
        "max_fp_delta": 0.0,
    },
    "fp_reducer": {
        "min_precision_delta": 0.0,
        "min_recall_delta": -0.01,
        "min_recall75_delta": -0.01,
        "min_small_recall_delta": -0.01,
        "max_fp_delta": -0.10,
    },
}

CLAIM_PROFILE_DESCRIPTIONS = {
    "broad_superiority": "Conservative metric gate: target must be no worse than baseline on precision, recall, small-object recall, and FP/sample.",
    "fp_reducer": "Recall-budgeted FP reducer gate: target must reduce FP/sample substantially while keeping precision non-worse and recall losses within budget.",
    "custom": "Custom metric gate assembled from explicit threshold arguments.",
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether a target input passes a conservative comparison claim gate.")
    parser.add_argument("report", help="Report directory or comparison_summary.json")
    parser.add_argument("--profile", default="broad_superiority", choices=sorted(CLAIM_PROFILES), help="Named metric gate profile.")
    parser.add_argument("--target-input", default="perception_calibrated_fusion_rgb_aux")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--min-precision-delta", type=float, default=None)
    parser.add_argument("--min-recall-delta", type=float, default=None)
    parser.add_argument("--min-recall75-delta", type=float, default=None)
    parser.add_argument("--min-small-recall-delta", type=float, default=None)
    parser.add_argument("--max-fp-delta", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Paired bootstrap resamples for sample-level delta confidence intervals.")
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", default="claim_gate")
    parser.add_argument("--require-ci", action="store_true", help="Require the paired bootstrap CI to satisfy each delta threshold.")
    parser.add_argument("--fail-on-fail", action="store_true", help="Return exit code 1 when the metric gate fails.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    thresholds = _cli_thresholds(args)
    summary = build_claim_gate(
        report,
        target_input=str(args.target_input),
        baseline_input=str(args.baseline_input),
        thresholds=thresholds,
        source_report=report_path,
    )
    destination = Path(args.output_dir).expanduser() if args.output_dir else report_path.parent / f"claim_gate_{_safe_name(args.target_input)}_vs_{_safe_name(args.baseline_input)}"
    html_path = write_claim_gate(summary, destination)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "claim_gate_summary.json"),
                    "profile": summary["profile"],
                    "pass": summary["pass"],
                    "verdict": summary["verdict"],
                    "failed": [item["metric"] for item in summary["criteria"] if not item["pass"]],
                }
            ),
            indent=2,
        )
    )
    return 1 if bool(args.fail_on_fail) and not bool(summary["pass"]) else 0


def build_claim_gate(
    report: Mapping[str, Any],
    *,
    target_input: str,
    baseline_input: str = "human_rgb",
    thresholds: Mapping[str, Any] | None = None,
    source_report: str | Path | None = None,
) -> Dict[str, Any]:
    threshold_values = _resolve_thresholds(thresholds)
    profile = str(threshold_values.pop("profile", "custom"))
    bootstrap_samples = max(int(threshold_values.get("bootstrap_samples", 1000)), 0)
    bootstrap_confidence = float(threshold_values.get("bootstrap_confidence", 0.95))
    bootstrap_seed = str(threshold_values.get("bootstrap_seed", "claim_gate"))
    require_ci = bool(threshold_values.get("require_ci", False))
    aggregate = report.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        raise ValueError("comparison report aggregate is missing or invalid")
    target = aggregate.get(target_input)
    baseline = aggregate.get(baseline_input)
    if not isinstance(target, Mapping):
        raise ValueError(f"target input not found in aggregate: {target_input}")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"baseline input not found in aggregate: {baseline_input}")

    criteria = []
    for metric, direction, threshold_key, default_value in CRITERIA:
        target_value = _metric_value(target, metric)
        baseline_value = _metric_value(baseline, metric)
        available = target_value is not None and baseline_value is not None
        delta = None if not available else target_value - baseline_value
        threshold = float(threshold_values.get(threshold_key, default_value))
        paired = _paired_delta_interval(
            report,
            baseline_input=baseline_input,
            target_input=target_input,
            aggregate_metric=metric,
            bootstrap_samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=f"{bootstrap_seed}:{metric}",
        )
        mean_passed = bool(available and (delta >= threshold if direction == "minimum_delta" else delta <= threshold))
        ci_passed = _ci_passes(paired, direction=direction, threshold=threshold) if require_ci else None
        passed = bool(mean_passed and (ci_passed if ci_passed is not None else True))
        criteria.append(
            {
                "metric": metric,
                "direction": direction,
                "threshold_key": threshold_key,
                "threshold": threshold,
                "baseline": baseline_value,
                "target": target_value,
                "delta": delta,
                "available": bool(available),
                "mean_pass": bool(mean_passed),
                "ci_required": bool(require_ci),
                "ci_pass": ci_passed,
                "paired_delta": paired,
                "pass": bool(passed),
            }
        )

    sample_count = int(report.get("sample_count", target.get("sample_count", baseline.get("sample_count", 0))))
    min_samples = int(threshold_values.get("min_samples", 1))
    sample_gate = {
        "metric": "sample_count",
        "direction": "minimum_value",
        "threshold_key": "min_samples",
        "threshold": int(min_samples),
        "target": int(sample_count),
        "delta": int(sample_count - min_samples),
        "pass": bool(sample_count >= min_samples),
    }
    criteria.append(sample_gate)
    passed = all(bool(item["pass"]) for item in criteria)
    verdict = "metric_gate_pass" if passed else "metric_gate_fail"
    return {
        "source_report": "" if source_report is None else str(source_report),
        "profile": profile,
        "profile_description": _profile_description(profile),
        "target_input": str(target_input),
        "baseline_input": str(baseline_input),
        "sample_count": int(sample_count),
        "thresholds": {
            "min_precision_delta": float(threshold_values.get("min_precision_delta", 0.0)),
            "min_recall_delta": float(threshold_values.get("min_recall_delta", 0.0)),
            "min_recall75_delta": float(threshold_values.get("min_recall75_delta", 0.0)),
            "min_small_recall_delta": float(threshold_values.get("min_small_recall_delta", 0.0)),
            "max_fp_delta": float(threshold_values.get("max_fp_delta", 0.0)),
            "min_samples": int(min_samples),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_confidence": float(bootstrap_confidence),
            "bootstrap_seed": str(bootstrap_seed),
            "require_ci": bool(require_ci),
        },
        "baseline_metrics": {key: baseline.get(key) for key, *_ in CRITERIA},
        "target_metrics": {key: target.get(key) for key, *_ in CRITERIA},
        "criteria": criteria,
        "pass": bool(passed),
        "verdict": verdict,
        "interpretation": _interpretation(passed, profile),
    }


def _cli_thresholds(args: Any) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {"profile": str(args.profile), **CLAIM_PROFILES[str(args.profile)]}
    for key, value in (
        ("min_precision_delta", args.min_precision_delta),
        ("min_recall_delta", args.min_recall_delta),
        ("min_recall75_delta", args.min_recall75_delta),
        ("min_small_recall_delta", args.min_small_recall_delta),
        ("max_fp_delta", args.max_fp_delta),
    ):
        if value is not None:
            thresholds[key] = float(value)
    thresholds.update(
        {
            "min_samples": int(args.min_samples),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "bootstrap_seed": str(args.bootstrap_seed),
            "require_ci": bool(args.require_ci),
        }
    )
    return thresholds


def _resolve_thresholds(thresholds: Mapping[str, Any] | None) -> Dict[str, Any]:
    raw = dict(thresholds or {})
    profile = str(raw.get("profile", "custom"))
    if profile != "custom" and profile not in CLAIM_PROFILES:
        raise ValueError(f"unknown claim profile: {profile}")
    values: Dict[str, Any] = dict(CLAIM_PROFILES.get(profile, {}))
    values.update({key: value for key, value in raw.items() if key != "profile"})
    values["profile"] = profile
    return values


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    if key not in metrics or metrics.get(key) is None:
        return None
    return float(metrics[key])


def _paired_delta_interval(
    report: Mapping[str, Any],
    *,
    baseline_input: str,
    target_input: str,
    aggregate_metric: str,
    bootstrap_samples: int,
    confidence: float,
    seed: str,
) -> Dict[str, Any] | None:
    sample_metric = SAMPLE_METRIC_MAP.get(aggregate_metric)
    if not sample_metric:
        return None
    deltas = []
    for sample in report.get("samples", ()):
        metrics = sample.get("metrics", {}) if isinstance(sample, Mapping) else {}
        baseline = metrics.get(baseline_input, {}) if isinstance(metrics, Mapping) else {}
        target = metrics.get(target_input, {}) if isinstance(metrics, Mapping) else {}
        if not isinstance(baseline, Mapping) or not isinstance(target, Mapping):
            continue
        baseline_value = _metric_value(baseline, sample_metric)
        target_value = _metric_value(target, sample_metric)
        if baseline_value is None or target_value is None:
            continue
        deltas.append(float(target_value - baseline_value))
    if not deltas:
        return None
    values = np.asarray(deltas, dtype=np.float64)
    result: Dict[str, Any] = {
        "sample_metric": sample_metric,
        "sample_count": int(values.size),
        "mean": float(np.mean(values)),
    }
    if bootstrap_samples <= 0 or values.size == 1:
        result["ci_low"] = None
        result["ci_high"] = None
        result["confidence"] = float(confidence)
        result["bootstrap_samples"] = int(bootstrap_samples)
        return result
    normalized_confidence = min(max(float(confidence), 0.0), 1.0)
    alpha = 1.0 - normalized_confidence
    rng = np.random.default_rng(_stable_seed(seed))
    indices = rng.integers(0, values.size, size=(int(bootstrap_samples), values.size))
    means = np.mean(values[indices], axis=1)
    result["ci_low"] = float(np.quantile(means, alpha / 2.0))
    result["ci_high"] = float(np.quantile(means, 1.0 - alpha / 2.0))
    result["confidence"] = float(normalized_confidence)
    result["bootstrap_samples"] = int(bootstrap_samples)
    return result


def _ci_passes(paired: Mapping[str, Any] | None, *, direction: str, threshold: float) -> bool:
    if not paired:
        return False
    if direction == "minimum_delta":
        value = paired.get("ci_low")
        return bool(value is not None and float(value) >= float(threshold))
    value = paired.get("ci_high")
    return bool(value is not None and float(value) <= float(threshold))


def _stable_seed(value: str) -> int:
    # Avoid Python's process-randomized hash so reports are reproducible.
    total = 0
    for char in str(value):
        total = (total * 131 + ord(char)) % (2**32)
    return int(total)


def write_claim_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "claim_gate_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _summary_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "comparison_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"comparison summary not found: {path}")
    return path


def _profile_description(profile: str) -> str:
    return CLAIM_PROFILE_DESCRIPTIONS.get(str(profile), CLAIM_PROFILE_DESCRIPTIONS["custom"])


def _interpretation(passed: bool, profile: str) -> str:
    if profile == "fp_reducer":
        if passed:
            return "The target passes the recall-budgeted FP-reduction gate. This does not support a broad HumanISP superiority claim by itself."
        return "The target does not pass the recall-budgeted FP-reduction gate."
    if passed:
        return "The target passes this conservative metric-only gate. This is still not a safety or product claim by itself."
    return "The target does not pass this conservative metric-only gate, so a broad superiority claim is not supported."


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for item in summary.get("criteria", ()):
        status = "PASS" if bool(item.get("pass")) else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(item.get('metric', '')))}</td>"
            f"<td>{html_lib.escape(str(item.get('direction', '')))}</td>"
            f"<td>{_fmt(item.get('baseline'))}</td>"
            f"<td>{_fmt(item.get('target'))}</td>"
            f"<td>{_fmt(item.get('delta'), signed=True)}</td>"
            f"<td>{_fmt_nested(item.get('paired_delta'), 'ci_low', signed=True)}</td>"
            f"<td>{_fmt_nested(item.get('paired_delta'), 'ci_high', signed=True)}</td>"
            f"<td>{_fmt(item.get('threshold'), signed=True)}</td>"
            f"<td class=\"{status.lower()}\">{status}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Claim Gate</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 9px 10px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Claim Gate</h1>
  <p>Target <code>{html_lib.escape(str(summary.get('target_input', '')))}</code> vs baseline <code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code>.</p>
  <p><strong>Profile:</strong> <code>{html_lib.escape(str(summary.get('profile', '')))}</code> - {html_lib.escape(str(summary.get('profile_description', '')))}</p>
  <p><strong>Verdict:</strong> <code>{html_lib.escape(str(summary.get('verdict', '')))}</code></p>
  <p>{html_lib.escape(str(summary.get('interpretation', '')))}</p>
  <table>
    <thead><tr><th>Metric</th><th>Rule</th><th>Baseline</th><th>Target</th><th>Delta</th><th>CI Low</th><th>CI High</th><th>Threshold</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>claim_gate_summary.json</code></p>
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _fmt_nested(value: Any, key: str, *, signed: bool = False) -> str:
    if not isinstance(value, Mapping):
        return ""
    return _fmt(value.get(key), signed=signed)


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
