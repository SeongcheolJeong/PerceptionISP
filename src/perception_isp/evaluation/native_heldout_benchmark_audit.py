"""Audit large held-out native RAW/CFA benchmark provenance and metrics."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "native_heldout_benchmark_audit_summary.json"
COMPARISON_SUMMARY = "comparison_summary.json"
DEFAULT_BASELINE_INPUT = "human_rgb"
DEFAULT_TARGET_INPUT = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001"
DEFAULT_MIN_PRECISION_DELTA = 0.01
DEFAULT_MIN_FP_REDUCTION = 0.10
DEFAULT_MAX_RECALL_DROP = 0.01
THRESHOLD_EPS = 1.0e-12
TRACKED_METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "fp@0.75_mean",
    "det_count_mean",
    "tp@0.50_mean",
    "fn@0.50_mean",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a large held-out native RAW/CFA comparison report.")
    parser.add_argument("comparison_report", help="comparison_summary.json path or report directory.")
    parser.add_argument("--baseline-input", default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--target-input", default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--min-precision-delta", type=float, default=DEFAULT_MIN_PRECISION_DELTA)
    parser.add_argument("--min-fp-reduction", type=float, default=DEFAULT_MIN_FP_REDUCTION)
    parser.add_argument("--max-recall-drop", type=float, default=DEFAULT_MAX_RECALL_DROP)
    parser.add_argument("--output-dir", default="reports/perception_native_heldout_benchmark_audit")
    args = parser.parse_args(argv)

    summary_path = _summary_path(args.comparison_report, COMPARISON_SUMMARY)
    comparison = json.loads(summary_path.read_text())
    summary = build_native_heldout_benchmark_audit(
        comparison,
        baseline_input=str(args.baseline_input),
        target_input=str(args.target_input),
        min_samples=int(args.min_samples),
        min_precision_delta=float(args.min_precision_delta),
        min_fp_reduction=float(args.min_fp_reduction),
        max_recall_drop=float(args.max_recall_drop),
        source_report_path=summary_path,
    )
    html_path = write_native_heldout_benchmark_audit(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "sample_count": summary["sample_count"],
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_native_heldout_benchmark_audit(
    comparison: Mapping[str, Any],
    *,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str = DEFAULT_TARGET_INPUT,
    min_samples: int = 1000,
    min_precision_delta: float = DEFAULT_MIN_PRECISION_DELTA,
    min_fp_reduction: float = DEFAULT_MIN_FP_REDUCTION,
    max_recall_drop: float = DEFAULT_MAX_RECALL_DROP,
    source_report_path: str | Path | None = None,
) -> Dict[str, Any]:
    samples = [row for row in comparison.get("samples", ()) if isinstance(row, Mapping)]
    provenance = _provenance_summary(samples)
    metric_summary = _metric_summary(comparison, baseline_input=baseline_input, target_input=target_input)
    thresholds = {
        "min_precision_delta": float(min_precision_delta),
        "min_fp_reduction": float(min_fp_reduction),
        "max_recall_drop": float(max_recall_drop),
    }
    claim_status = _claim_status(
        metric_summary,
        provenance,
        sample_count=len(samples),
        min_samples=int(min_samples),
        thresholds=thresholds,
    )
    checks = _checks(
        sample_count=len(samples),
        min_samples=int(min_samples),
        provenance=provenance,
        metric_summary=metric_summary,
        thresholds=thresholds,
    )
    return {
        "name": "Large held-out native RAW benchmark audit",
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "pass": bool(checks) and all(row["status"] == "pass" for row in checks),
        "claim_status": claim_status,
        "source_comparison_summary": "" if source_report_path is None else str(source_report_path),
        "source_comparison_html": _html_sibling(source_report_path),
        "baseline_input": str(baseline_input),
        "target_input": str(target_input),
        "sample_count": len(samples),
        "min_samples": int(min_samples),
        "thresholds": thresholds,
        "provenance": provenance,
        "metric_summary": metric_summary,
        "checks": checks,
        "interpretation": (
            "This audit verifies whether a large held-out comparison report is backed by native RAW/CFA rows "
            "without bridge remapping, then records HumanISP-vs-PerceptionISP detector deltas. "
            "The claim gate requires a minimum precision gain, minimum FP reduction, and bounded recall loss."
        ),
        "claim_boundary": (
            "This proves large held-out native RAW provenance only for the CFA pattern represented by the report. "
            "It does not prove all-CFA coverage, real adverse RAW coverage, broad detector superiority, or trained RGB+Aux DNN improvement."
        ),
    }


def write_native_heldout_benchmark_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _provenance_summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counters: Dict[str, Counter[Any]] = {
        "source_cfa_patterns": Counter(),
        "target_cfa_patterns": Counter(),
        "requested_cfa_patterns": Counter(),
        "bridges": Counter(),
        "raw_source_keys": Counter(),
        "source_shapes": Counter(),
        "target_shapes": Counter(),
    }
    count = len(samples)
    camerae2e_count = 0
    true_cfa_count = 0
    remapped_count = 0
    source_target_match_count = 0
    native_resolution_match_count = 0
    native_resolution_at_least_count = 0
    native_raw_source_accepted_count = 0
    for sample in samples:
        metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), Mapping) else {}
        raw = metadata.get("raw_provenance", {}) if isinstance(metadata.get("raw_provenance"), Mapping) else {}
        source_cfa = str(raw.get("source_cfa_pattern", metadata.get("cfa_pattern", ""))).upper()
        target_cfa = str(raw.get("target_cfa_pattern", metadata.get("cfa_pattern", ""))).upper()
        requested_cfa = str(raw.get("requested_cfa_pattern", metadata.get("requested_cfa_pattern", ""))).upper()
        source_shape = tuple(raw.get("source_shape", raw.get("raw_input_shape", ())) or ())
        target_shape = tuple(raw.get("target_shape", ()) or ())
        true_cfa = bool(raw.get("true_sensor_cfa_mosaic", False))
        remapped = bool(raw.get("pattern_remapped", False))
        cfa_match = bool(source_cfa and target_cfa and source_cfa == target_cfa)
        _bump(counters["source_cfa_patterns"], source_cfa)
        _bump(counters["target_cfa_patterns"], target_cfa)
        _bump(counters["requested_cfa_patterns"], requested_cfa)
        _bump(counters["bridges"], raw.get("bridge"))
        _bump(counters["raw_source_keys"], raw.get("raw_source_key"))
        _bump(counters["source_shapes"], source_shape)
        _bump(counters["target_shapes"], target_shape)
        camerae2e_count += int(bool(raw.get("camerae2e_used", False)))
        true_cfa_count += int(true_cfa)
        remapped_count += int(remapped)
        source_target_match_count += int(cfa_match)
        native_resolution_match_count += int(bool(raw.get("native_resolution_matches_target", False)))
        native_resolution_at_least_count += int(bool(raw.get("native_resolution_at_least_target", False)))
        native_raw_source_accepted_count += int(true_cfa and not remapped and cfa_match)
    return {
        "sample_count": count,
        "camerae2e_used_count": camerae2e_count,
        "camerae2e_used_fraction": _fraction(camerae2e_count, count),
        "native_raw_source_accepted_count": native_raw_source_accepted_count,
        "native_raw_source_accepted_fraction": _fraction(native_raw_source_accepted_count, count),
        "true_sensor_cfa_mosaic_count": true_cfa_count,
        "true_sensor_cfa_mosaic_fraction": _fraction(true_cfa_count, count),
        "pattern_remapped_count": remapped_count,
        "pattern_remapped_fraction": _fraction(remapped_count, count),
        "source_target_cfa_match_count": source_target_match_count,
        "source_target_cfa_match_fraction": _fraction(source_target_match_count, count),
        "native_resolution_matches_target_count": native_resolution_match_count,
        "native_resolution_matches_target_fraction": _fraction(native_resolution_match_count, count),
        "native_resolution_at_least_target_count": native_resolution_at_least_count,
        "native_resolution_at_least_target_fraction": _fraction(native_resolution_at_least_count, count),
        "source_cfa_patterns": _counter_dict(counters["source_cfa_patterns"]),
        "target_cfa_patterns": _counter_dict(counters["target_cfa_patterns"]),
        "requested_cfa_patterns": _counter_dict(counters["requested_cfa_patterns"]),
        "bridges": _counter_dict(counters["bridges"]),
        "raw_source_keys": _counter_dict(counters["raw_source_keys"]),
        "source_shapes": _counter_dict(counters["source_shapes"]),
        "target_shapes": _counter_dict(counters["target_shapes"]),
    }


def _metric_summary(comparison: Mapping[str, Any], *, baseline_input: str, target_input: str) -> Dict[str, Any]:
    aggregate = comparison.get("aggregate", {}) if isinstance(comparison.get("aggregate"), Mapping) else {}
    baseline = aggregate.get(str(baseline_input), {}) if isinstance(aggregate.get(str(baseline_input)), Mapping) else {}
    target = aggregate.get(str(target_input), {}) if isinstance(aggregate.get(str(target_input)), Mapping) else {}
    deltas = {
        metric: _maybe_float(target.get(metric)) - _maybe_float(baseline.get(metric))
        for metric in TRACKED_METRICS
        if target.get(metric) is not None and baseline.get(metric) is not None
    }
    return {
        "baseline_input": str(baseline_input),
        "target_input": str(target_input),
        "baseline": {metric: _maybe_float(baseline.get(metric)) for metric in TRACKED_METRICS if baseline.get(metric) is not None},
        "target": {metric: _maybe_float(target.get(metric)) for metric in TRACKED_METRICS if target.get(metric) is not None},
        "deltas": deltas,
        "metrics_present": bool(baseline) and bool(target),
    }


def _claim_status(
    metric_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    sample_count: int,
    min_samples: int,
    thresholds: Mapping[str, Any],
) -> str:
    native_ok = (
        sample_count >= int(min_samples)
        and _maybe_float(provenance.get("native_raw_source_accepted_fraction")) == 1.0
        and _maybe_float(provenance.get("true_sensor_cfa_mosaic_fraction")) == 1.0
        and int(provenance.get("pattern_remapped_count", 0)) == 0
        and _maybe_float(provenance.get("source_target_cfa_match_fraction")) == 1.0
    )
    deltas = metric_summary.get("deltas", {}) if isinstance(metric_summary.get("deltas"), Mapping) else {}
    fp_delta = _maybe_float(deltas.get("fp@0.50_mean"))
    recall_delta = _maybe_float(deltas.get("recall@0.50_mean"))
    precision_delta = _maybe_float(deltas.get("precision@0.50_mean"))
    min_precision_delta = _maybe_float(thresholds.get("min_precision_delta"))
    min_fp_reduction = _maybe_float(thresholds.get("min_fp_reduction"))
    max_recall_drop = _maybe_float(thresholds.get("max_recall_drop"))
    effect_ok = (
        precision_delta >= min_precision_delta - THRESHOLD_EPS
        and fp_delta <= -min_fp_reduction + THRESHOLD_EPS
        and recall_delta >= -max_recall_drop - THRESHOLD_EPS
    )
    if not native_ok:
        return "native_heldout_benchmark_not_supported"
    if effect_ok and recall_delta < 0.0:
        return "large_native_fp_reducer_with_recall_tradeoff"
    if effect_ok and recall_delta >= 0.0:
        return "large_native_fp_reducer_supported"
    return "large_native_benchmark_diagnostic"


def _checks(
    *,
    sample_count: int,
    min_samples: int,
    provenance: Mapping[str, Any],
    metric_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    deltas = metric_summary.get("deltas", {}) if isinstance(metric_summary.get("deltas"), Mapping) else {}
    min_precision_delta = _maybe_float(thresholds.get("min_precision_delta"))
    min_fp_reduction = _maybe_float(thresholds.get("min_fp_reduction"))
    max_recall_drop = _maybe_float(thresholds.get("max_recall_drop"))
    fp_delta = _maybe_float(deltas.get("fp@0.50_mean"))
    precision_delta = _maybe_float(deltas.get("precision@0.50_mean"))
    recall_delta = _maybe_float(deltas.get("recall@0.50_mean"))
    return [
        {
            "id": "large_heldout_sample_count",
            "status": "pass" if int(sample_count) >= int(min_samples) else "fail",
            "evidence": f"samples={sample_count} min={min_samples}",
        },
        {
            "id": "native_raw_source_for_all_samples",
            "status": "pass" if _maybe_float(provenance.get("native_raw_source_accepted_fraction")) == 1.0 else "fail",
            "evidence": (
                f"native_raw_source={int(provenance.get('native_raw_source_accepted_count', 0))}/{sample_count}; "
                f"camerae2e={int(provenance.get('camerae2e_used_count', 0))}/{sample_count}"
            ),
        },
        {
            "id": "true_sensor_cfa_mosaic_for_all_samples",
            "status": "pass" if _maybe_float(provenance.get("true_sensor_cfa_mosaic_fraction")) == 1.0 else "fail",
            "evidence": f"true_cfa={int(provenance.get('true_sensor_cfa_mosaic_count', 0))}/{sample_count}",
        },
        {
            "id": "no_pattern_remap",
            "status": "pass" if int(provenance.get("pattern_remapped_count", 0)) == 0 else "fail",
            "evidence": f"remapped={int(provenance.get('pattern_remapped_count', 0))}/{sample_count}",
        },
        {
            "id": "source_and_target_cfa_match",
            "status": "pass" if _maybe_float(provenance.get("source_target_cfa_match_fraction")) == 1.0 else "fail",
            "evidence": f"match={int(provenance.get('source_target_cfa_match_count', 0))}/{sample_count}; source={provenance.get('source_cfa_patterns', {})}; target={provenance.get('target_cfa_patterns', {})}",
        },
        {
            "id": "baseline_and_target_metrics_present",
            "status": "pass" if bool(metric_summary.get("metrics_present")) else "fail",
            "evidence": f"baseline={metric_summary.get('baseline_input')} target={metric_summary.get('target_input')}",
        },
        {
            "id": "native_fp_reduction_observed",
            "status": "pass" if fp_delta <= -min_fp_reduction + THRESHOLD_EPS else "fail",
            "evidence": f"dFP50={_fmt(fp_delta, signed=True)} threshold<=-{_fmt(min_fp_reduction)}",
        },
        {
            "id": "native_precision_gain_effect_size",
            "status": "pass" if precision_delta >= min_precision_delta - THRESHOLD_EPS else "fail",
            "evidence": f"dP50={_fmt(precision_delta, signed=True)} threshold>={_fmt(min_precision_delta, signed=True)}",
        },
        {
            "id": "native_recall_drop_within_budget",
            "status": "pass" if recall_delta >= -max_recall_drop - THRESHOLD_EPS else "fail",
            "evidence": f"dR50={_fmt(recall_delta, signed=True)} budget>=-{_fmt(max_recall_drop)}",
        },
    ]


def _render_html(summary: Mapping[str, Any]) -> str:
    provenance = summary.get("provenance", {}) if isinstance(summary.get("provenance"), Mapping) else {}
    metrics = summary.get("metric_summary", {}) if isinstance(summary.get("metric_summary"), Mapping) else {}
    deltas = metrics.get("deltas", {}) if isinstance(metrics.get("deltas"), Mapping) else {}
    thresholds = summary.get("thresholds", {}) if isinstance(summary.get("thresholds"), Mapping) else {}
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    metric_rows = "".join(_metric_row(metric, metrics) for metric in TRACKED_METRICS)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP Native Held-Out Benchmark Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pass {{ color: #047857; font-weight: 650; }}
    .fail, .warning {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Native Held-Out Benchmark Audit</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class="{html_lib.escape(str(summary.get('status', '')))}">{html_lib.escape(str(summary.get('status', '')))}</code>; claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>; samples={int(summary.get('sample_count', 0))}.</p>
  <h2>Native Provenance</h2>
  <table><tbody>
    <tr><th>Native RAW Source</th><td>{int(provenance.get('native_raw_source_accepted_count', 0))}/{int(summary.get('sample_count', 0))}</td><th>True CFA Mosaic</th><td>{int(provenance.get('true_sensor_cfa_mosaic_count', 0))}/{int(summary.get('sample_count', 0))}</td><th>CameraE2E</th><td>{int(provenance.get('camerae2e_used_count', 0))}/{int(summary.get('sample_count', 0))}</td></tr>
    <tr><th>Remapped</th><td>{int(provenance.get('pattern_remapped_count', 0))}</td><th>Source/target CFA match</th><td>{int(provenance.get('source_target_cfa_match_count', 0))}/{int(summary.get('sample_count', 0))}</td><th></th><td></td></tr>
    <tr><th>Source CFA</th><td>{html_lib.escape(str(provenance.get('source_cfa_patterns', {})))}</td><th>Target CFA</th><td>{html_lib.escape(str(provenance.get('target_cfa_patterns', {})))}</td><th>Shapes</th><td>{html_lib.escape(str(provenance.get('target_shapes', {})))}</td></tr>
    <tr><th>Bridge</th><td>{html_lib.escape(str(provenance.get('bridges', {})))}</td><th>Raw Source</th><td>{html_lib.escape(str(provenance.get('raw_source_keys', {})))}</td><th>Native Resolution Match</th><td>{_fmt(provenance.get('native_resolution_matches_target_fraction'))}</td></tr>
  </tbody></table>
  <h2>Checks</h2>
  <p>Effect-size gate: min dP50={_fmt(thresholds.get('min_precision_delta'), signed=True)}, min FP50 reduction={_fmt(thresholds.get('min_fp_reduction'))}, max recall drop={_fmt(thresholds.get('max_recall_drop'))}.</p>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Metrics</h2>
  <p>Baseline <code>{html_lib.escape(str(metrics.get('baseline_input', '')))}</code> vs target <code>{html_lib.escape(str(metrics.get('target_input', '')))}</code>. dP50={_fmt(deltas.get('precision@0.50_mean'), signed=True)}, dR50={_fmt(deltas.get('recall@0.50_mean'), signed=True)}, dFP50={_fmt(deltas.get('fp@0.50_mean'), signed=True)}.</p>
  <table><thead><tr><th>Metric</th><th>Baseline</th><th>Target</th><th>Delta</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _metric_row(metric: str, metrics: Mapping[str, Any]) -> str:
    baseline = metrics.get("baseline", {}) if isinstance(metrics.get("baseline"), Mapping) else {}
    target = metrics.get("target", {}) if isinstance(metrics.get("target"), Mapping) else {}
    deltas = metrics.get("deltas", {}) if isinstance(metrics.get("deltas"), Mapping) else {}
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(metric)}</code></td>"
        f"<td>{_fmt(baseline.get(metric))}</td>"
        f"<td>{_fmt(target.get(metric))}</td>"
        f"<td>{_fmt(deltas.get(metric), signed=True)}</td>"
        "</tr>"
    )


def _bump(counter: Counter[Any], value: Any) -> None:
    if value in (None, "", ()):
        counter["missing"] += 1
        return
    counter[str(value)] += 1


def _counter_dict(counter: Counter[Any]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count / total)


def _html_sibling(path: str | Path | None) -> str:
    if path is None:
        return ""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        html_path = candidate / "index.html"
    else:
        html_path = candidate.with_name("index.html")
    return str(html_path) if html_path.exists() else ""


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _maybe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
