"""Audit detector-box FP/TP support evidence in saved comparison reports."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .aux_contribution_audit import (
    _box_iou,
    _box_payload,
    _detections_for_sample,
    _ground_truth_for_sample,
    _positive_detection_indices,
    _scene_edge_context_for_sample,
    _support_values,
    sample_bridge_from_comparison_report,
)
from .types import json_ready


SUMMARY_FILENAME = "detector_box_support_audit_summary.json"
COMPARISON_SUMMARY = "comparison_summary.json"
DEFAULT_TARGET_INPUT = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001"
DEFAULT_TRANSITION_BASELINE_INPUT = "perception_fusion_rgb_aux"
DEFAULT_AUDIT_INPUTS = ("human_rgb", DEFAULT_TRANSITION_BASELINE_INPUT, DEFAULT_TARGET_INPUT)
SUPPORT_FEATURES = (
    "score",
    "aux_support",
    "edge_support",
    "saturation_support",
    "reliability_support",
    "aux_box_iou",
    "scene_edge_support",
    "scene_edge_fraction",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit detector-box FP/TP support evidence from a comparison report.")
    parser.add_argument("comparison_report", help="comparison_summary.json path or report directory.")
    parser.add_argument("--audit-input", action="append", default=None, help="Input name to audit. May be repeated.")
    parser.add_argument("--target-input", default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--transition-baseline-input", default=DEFAULT_TRANSITION_BASELINE_INPUT)
    parser.add_argument("--output-dir", default="reports/perception_detector_box_support_audit")
    args = parser.parse_args(argv)

    report_path = _summary_path(args.comparison_report, COMPARISON_SUMMARY)
    report = json.loads(report_path.read_text())
    summary = build_detector_box_support_audit(
        report,
        audit_inputs=tuple(args.audit_input or DEFAULT_AUDIT_INPUTS),
        target_input=str(args.target_input),
        transition_baseline_input=str(args.transition_baseline_input),
        source_report_path=report_path,
    )
    html_path = write_detector_box_support_audit(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "sample_count": summary["sample_count"],
                    "target_input": summary["target_input"],
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_detector_box_support_audit(
    comparison: Mapping[str, Any],
    *,
    audit_inputs: Sequence[str] = DEFAULT_AUDIT_INPUTS,
    target_input: str = DEFAULT_TARGET_INPUT,
    transition_baseline_input: str = DEFAULT_TRANSITION_BASELINE_INPUT,
    source_report_path: str | Path | None = None,
) -> Dict[str, Any]:
    samples = [row for row in comparison.get("samples", ()) if isinstance(row, Mapping)]
    run_config = comparison.get("run_config", {}) if isinstance(comparison.get("run_config"), Mapping) else {}
    label_agnostic = bool(run_config.get("label_agnostic", False))
    scene_edge_cache: Dict[str, Dict[str, Any] | None] = {}
    input_names = _dedupe_inputs((*audit_inputs, target_input, transition_baseline_input))
    input_summaries = []
    for input_name in input_names:
        rows = _detector_rows_for_input(
            samples,
            str(input_name),
            label_agnostic=label_agnostic,
            scene_edge_cache=scene_edge_cache,
        )
        input_summaries.append(_input_summary(str(input_name), rows))
    by_input = {row["input_name"]: row for row in input_summaries}
    target_summary = by_input.get(str(target_input), _empty_input_summary(str(target_input)))
    transition_bridge = None
    if source_report_path is not None:
        transition_bridge = sample_bridge_from_comparison_report(
            source_report_path,
            baseline_input=str(transition_baseline_input),
            target_input=str(target_input),
        )
    checks = _checks(
        input_summaries,
        target_summary=target_summary,
        transition_bridge=transition_bridge,
        target_input=str(target_input),
        transition_baseline_input=str(transition_baseline_input),
    )
    return {
        "name": "Detector-box support audit",
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "pass": bool(checks) and all(row["status"] == "pass" for row in checks),
        "claim_status": "detector_box_support_diagnostic",
        "source_comparison_summary": "" if source_report_path is None else str(source_report_path),
        "source_comparison_html": _html_sibling(source_report_path),
        "sample_count": len(samples),
        "label_agnostic": label_agnostic,
        "audit_inputs": [row["input_name"] for row in input_summaries],
        "target_input": str(target_input),
        "transition_baseline_input": str(transition_baseline_input),
        "input_summaries": input_summaries,
        "target_summary": target_summary,
        "transition_bridge": transition_bridge,
        "checks": checks,
        "interpretation": (
            "This diagnostic audit classifies saved detector boxes as TP or FP against GT boxes, then compares score, "
            "aux/edge support metadata, reliability support, and source-scene edge support between FP and TP boxes. "
            "It also reuses the same-sample transition bridge to show which fusion proposals are removed by the "
            "calibrated target."
        ),
        "claim_boundary": (
            "This uses detector-box support metadata and a source-scene edge proxy from saved reports. It is not a "
            "true box-boundary contour metric, not a HumanISP/PerceptionISP edge-map recomputation, and not trained "
            "RGB+Aux DNN proof. If global FP boxes have higher edge support than TP boxes, do not claim edge support "
            "alone separates all false positives."
        ),
    }


def write_detector_box_support_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _detector_rows_for_input(
    samples: Sequence[Mapping[str, Any]],
    input_name: str,
    *,
    label_agnostic: bool,
    scene_edge_cache: Dict[str, Dict[str, Any] | None],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", sample_index))
        detections = _detections_for_sample(sample, input_name)
        if not detections:
            continue
        ground_truth = _ground_truth_for_sample(sample)
        positives = _positive_detection_indices(detections, ground_truth, label_agnostic=label_agnostic)
        scene_edge_context = _scene_edge_context_for_sample(sample, scene_edge_cache)
        for detection_index, detection in enumerate(detections):
            box = _box_payload(detection)
            support = _available_support_values(detection, scene_edge_context=scene_edge_context)
            is_tp = detection_index in positives
            rows.append(
                {
                    "sample_id": sample_id,
                    "input_name": str(input_name),
                    "label": str(box.get("label", "")),
                    "xyxy": [float(value) for value in box.get("xyxy", ())],
                    "area": _box_area(box),
                    "is_tp": bool(is_tp),
                    "is_fp": not bool(is_tp),
                    "best_gt_iou": _best_gt_iou(box, ground_truth, label_agnostic=label_agnostic),
                    **support,
                }
            )
    return rows


def _available_support_values(detection: Mapping[str, Any], *, scene_edge_context: Mapping[str, Any] | None) -> Dict[str, float]:
    raw = _support_values(detection, scene_edge_context=scene_edge_context)
    values: Dict[str, float] = {"score": _maybe_float(detection.get("score"))}
    metadata = detection.get("metadata", {}) if isinstance(detection.get("metadata"), Mapping) else {}
    fusion = metadata.get("fusion", {}) if isinstance(metadata.get("fusion"), Mapping) else {}
    for key in ("aux_support", "edge_support", "saturation_support", "reliability_support", "aux_box_iou"):
        if key in fusion:
            values[key] = _maybe_float(fusion.get(key))
    for key in ("scene_edge_support", "scene_edge_fraction"):
        if key in raw:
            values[key] = _maybe_float(raw.get(key))
    return values


def _input_summary(input_name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fp_rows = [row for row in rows if bool(row.get("is_fp"))]
    tp_rows = [row for row in rows if bool(row.get("is_tp"))]
    correlations = _feature_correlations(fp_rows, tp_rows)
    return {
        "input_name": str(input_name),
        "detection_count": len(rows),
        "tp_count": len(tp_rows),
        "fp_count": len(fp_rows),
        "precision_proxy": (len(tp_rows) / len(rows)) if rows else 0.0,
        "fp_per_sample_with_detections": (len(fp_rows) / max(len({str(row.get("sample_id", "")) for row in rows}), 1)) if rows else 0.0,
        "support_means": {
            "tp": _support_mean(tp_rows),
            "fp": _support_mean(fp_rows),
            "all": _support_mean(rows),
        },
        "support_deltas": _support_deltas(fp_rows, tp_rows),
        "correlations": correlations,
        "examples": _examples(fp_rows, tp_rows),
    }


def _empty_input_summary(input_name: str) -> Dict[str, Any]:
    return {
        "input_name": str(input_name),
        "detection_count": 0,
        "tp_count": 0,
        "fp_count": 0,
        "precision_proxy": 0.0,
        "fp_per_sample_with_detections": 0.0,
        "support_means": {"tp": {"count": 0}, "fp": {"count": 0}, "all": {"count": 0}},
        "support_deltas": {},
        "correlations": {"status": "missing", "rows": [], "key_results": {}},
        "examples": [],
    }


def _checks(
    input_summaries: Sequence[Mapping[str, Any]],
    *,
    target_summary: Mapping[str, Any],
    transition_bridge: Mapping[str, Any] | None,
    target_input: str,
    transition_baseline_input: str,
) -> list[Dict[str, Any]]:
    target_correlations = target_summary.get("correlations", {}) if isinstance(target_summary.get("correlations"), Mapping) else {}
    target_score = _correlation(target_correlations, "score")
    target_edge = _correlation(target_correlations, "edge_support")
    bridge_edge = _bridge_correlation(transition_bridge, "edge_support") if transition_bridge is not None else None
    bridge_scene = _bridge_correlation(transition_bridge, "scene_edge_support") if transition_bridge is not None else None
    removed_fp = int((transition_bridge or {}).get("removed_fp_count", 0)) if isinstance(transition_bridge, Mapping) else 0
    removed_tp = int((transition_bridge or {}).get("removed_tp_count", 0)) if isinstance(transition_bridge, Mapping) else 0
    return [
        {
            "id": "audit_inputs_have_detector_rows",
            "status": "pass" if any(int(row.get("detection_count", 0)) > 0 for row in input_summaries) else "fail",
            "evidence": f"inputs={len(input_summaries)} detections={sum(int(row.get('detection_count', 0)) for row in input_summaries)}",
        },
        {
            "id": "target_fp_tp_rows_available",
            "status": "pass" if int(target_summary.get("fp_count", 0)) > 0 and int(target_summary.get("tp_count", 0)) > 0 else "fail",
            "evidence": f"{target_input}: fp={int(target_summary.get('fp_count', 0))} tp={int(target_summary.get('tp_count', 0))}",
        },
        {
            "id": "target_score_separates_fp_from_tp",
            "status": "pass" if _lower_feature_predicts_fp(target_score) else "fail",
            "evidence": _correlation_evidence(target_score, "score"),
        },
        {
            "id": "target_aux_edge_global_result_recorded",
            "status": "pass" if target_edge is not None else "fail",
            "evidence": _correlation_evidence(target_edge, "edge_support"),
        },
        {
            "id": "transition_bridge_available",
            "status": "pass" if transition_bridge is not None and int(transition_bridge.get("compared_sample_count", 0)) > 0 else "fail",
            "evidence": (
                "missing"
                if transition_bridge is None
                else f"{transition_baseline_input}->{target_input}; samples={int(transition_bridge.get('compared_sample_count', 0))}"
            ),
        },
        {
            "id": "transition_removes_more_fp_than_tp",
            "status": "pass" if removed_fp > removed_tp and removed_fp > 0 else "fail",
            "evidence": f"removed_fp={removed_fp} removed_tp={removed_tp}",
        },
        {
            "id": "removed_fp_has_lower_aux_edge_support",
            "status": "pass" if _lower_feature_predicts_fp(bridge_edge) else "fail",
            "evidence": _correlation_evidence(bridge_edge, "edge_support"),
        },
        {
            "id": "removed_fp_has_lower_source_scene_edge_support",
            "status": "pass" if _lower_feature_predicts_fp(bridge_scene) else "fail",
            "evidence": _correlation_evidence(bridge_scene, "scene_edge_support"),
        },
    ]


def _feature_correlations(fp_rows: Sequence[Mapping[str, Any]], tp_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for feature in SUPPORT_FEATURES:
        row = _feature_correlation(fp_rows, tp_rows, feature)
        if row is not None:
            rows.append(row)
    key_results = {
        f"fp_vs_tp_{row['feature']}_{name}": row[name]
        for row in rows
        for name in ("fp_mean", "tp_mean", "delta_fp_minus_tp", "point_biserial", "auc_low_feature_predicts_fp")
        if row.get(name) is not None
    }
    edge = _correlation({"rows": rows}, "edge_support")
    return {
        "status": "pass" if edge is not None else "diagnostic",
        "rows": rows,
        "key_results": key_results,
    }


def _feature_correlation(fp_rows: Sequence[Mapping[str, Any]], tp_rows: Sequence[Mapping[str, Any]], feature: str) -> Dict[str, Any] | None:
    fp_values = [_maybe_float(row.get(feature)) for row in fp_rows if feature in row]
    tp_values = [_maybe_float(row.get(feature)) for row in tp_rows if feature in row]
    if not fp_values or not tp_values:
        return None
    fp_mean = sum(fp_values) / len(fp_values)
    tp_mean = sum(tp_values) / len(tp_values)
    delta = fp_mean - tp_mean
    auc_low = _binary_auc([-value for value in fp_values], [-value for value in tp_values])
    return {
        "comparison": "fp_vs_tp",
        "feature": str(feature),
        "fp_count": len(fp_values),
        "tp_count": len(tp_values),
        "fp_mean": fp_mean,
        "tp_mean": tp_mean,
        "delta_fp_minus_tp": delta,
        "point_biserial": _point_biserial(fp_values, tp_values),
        "auc_low_feature_predicts_fp": auc_low,
        "lower_feature_predicts_fp": delta < 0.0 and auc_low > 0.5,
    }


def _support_mean(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float | int]:
    result: Dict[str, float | int] = {"count": len(rows)}
    for key in SUPPORT_FEATURES:
        values = [_maybe_float(row.get(key)) for row in rows if key in row]
        if values:
            result[f"{key}_mean"] = sum(values) / len(values)
    return result


def _support_deltas(fp_rows: Sequence[Mapping[str, Any]], tp_rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    fp_mean = _support_mean(fp_rows)
    tp_mean = _support_mean(tp_rows)
    if int(fp_mean.get("count", 0)) <= 0 or int(tp_mean.get("count", 0)) <= 0:
        return {}
    return {
        f"fp_minus_tp_{key}_mean": _maybe_float(fp_mean.get(f"{key}_mean")) - _maybe_float(tp_mean.get(f"{key}_mean"))
        for key in SUPPORT_FEATURES
        if fp_mean.get(f"{key}_mean") is not None and tp_mean.get(f"{key}_mean") is not None
    }


def _examples(fp_rows: Sequence[Mapping[str, Any]], tp_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    examples: list[Dict[str, Any]] = []
    for label, rows in (("fp_high_score", sorted(fp_rows, key=lambda row: _maybe_float(row.get("score")), reverse=True)), ("tp_high_score", sorted(tp_rows, key=lambda row: _maybe_float(row.get("score")), reverse=True))):
        for row in rows[:6]:
            examples.append(
                {
                    "kind": label,
                    "sample_id": row.get("sample_id"),
                    "label": row.get("label"),
                    "score": row.get("score"),
                    "edge_support": row.get("edge_support"),
                    "aux_support": row.get("aux_support"),
                    "scene_edge_support": row.get("scene_edge_support"),
                    "best_gt_iou": row.get("best_gt_iou"),
                }
            )
    return examples


def _correlation(correlations: Mapping[str, Any], feature: str) -> Mapping[str, Any] | None:
    for row in correlations.get("rows", ()):
        if isinstance(row, Mapping) and str(row.get("feature", "")) == str(feature):
            return row
    return None


def _bridge_correlation(transition_bridge: Mapping[str, Any] | None, feature: str) -> Mapping[str, Any] | None:
    if transition_bridge is None:
        return None
    proposal_correlation = transition_bridge.get("proposal_correlation", {}) if isinstance(transition_bridge.get("proposal_correlation"), Mapping) else {}
    for row in proposal_correlation.get("rows", ()):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("comparison", "")) == "removed_fp_vs_kept_tp" and str(row.get("feature", "")) == str(feature):
            return {
                "feature": str(feature),
                "fp_count": int(row.get("positive_count", 0)),
                "tp_count": int(row.get("negative_count", 0)),
                "fp_mean": _maybe_float(row.get("positive_mean")),
                "tp_mean": _maybe_float(row.get("negative_mean")),
                "delta_fp_minus_tp": _maybe_float(row.get("delta")),
                "point_biserial": _maybe_float(row.get("point_biserial")),
                "auc_low_feature_predicts_fp": _maybe_float(row.get("auc_low_feature_predicts_positive")),
                "lower_feature_predicts_fp": bool(row.get("lower_feature_predicts_positive")),
            }
    return None


def _lower_feature_predicts_fp(row: Mapping[str, Any] | None) -> bool:
    return bool(row) and bool(row.get("lower_feature_predicts_fp")) and _maybe_float(row.get("auc_low_feature_predicts_fp")) > 0.5


def _correlation_evidence(row: Mapping[str, Any] | None, feature: str) -> str:
    if row is None:
        return f"{feature}=missing"
    return (
        f"{feature}: fpMean={_fmt(row.get('fp_mean'))}; tpMean={_fmt(row.get('tp_mean'))}; "
        f"dFPminusTP={_fmt(row.get('delta_fp_minus_tp'), signed=True)}; "
        f"aucLowPredictsFP={_fmt(row.get('auc_low_feature_predicts_fp'))}; "
        f"lowerPredictsFP={bool(row.get('lower_feature_predicts_fp'))}"
    )


def _best_gt_iou(box: Mapping[str, Any], ground_truth: Sequence[Mapping[str, Any]], *, label_agnostic: bool) -> float:
    best = 0.0
    for gt in ground_truth:
        gt_box = _box_payload(gt)
        if not label_agnostic and str(box.get("label", "")) != str(gt_box.get("label", "")):
            continue
        best = max(best, _box_iou(box, gt_box))
    return float(best)


def _box_area(box: Mapping[str, Any]) -> float:
    x1, y1, x2, y2 = tuple(float(value) for value in box.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _binary_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.5
    ranked = sorted([(float(score), 1) for score in positive_scores] + [(float(score), 0) for score in negative_scores], key=lambda item: item[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        tie_end = index + 1
        while tie_end < len(ranked) and ranked[tie_end][0] == ranked[index][0]:
            tie_end += 1
        average_rank = (rank + rank + (tie_end - index) - 1) / 2.0
        positives_in_tie = sum(label for _, label in ranked[index:tie_end])
        rank_sum_positive += average_rank * positives_in_tie
        rank += tie_end - index
        index = tie_end
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    return (rank_sum_positive - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)


def _point_biserial(positive_values: Sequence[float], negative_values: Sequence[float]) -> float:
    if not positive_values or not negative_values:
        return 0.0
    values = [*positive_values, *negative_values]
    mean_all = sum(values) / len(values)
    variance = sum((value - mean_all) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return 0.0
    mean_pos = sum(positive_values) / len(positive_values)
    mean_neg = sum(negative_values) / len(negative_values)
    n_pos = len(positive_values)
    n_neg = len(negative_values)
    n_total = n_pos + n_neg
    return (mean_pos - mean_neg) / (variance**0.5) * ((n_pos * n_neg) / float(n_total * n_total)) ** 0.5


def _render_html(summary: Mapping[str, Any]) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    input_rows = "".join(_input_row(row) for row in summary.get("input_summaries", ()) if isinstance(row, Mapping))
    target = summary.get("target_summary", {}) if isinstance(summary.get("target_summary"), Mapping) else {}
    target_rows = "".join(_correlation_row(row) for row in (target.get("correlations", {}) if isinstance(target.get("correlations"), Mapping) else {}).get("rows", ()) if isinstance(row, Mapping))
    transition_html = _transition_html(summary.get("transition_bridge"))
    source_link = _source_link(summary)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP Detector-Box Support Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pass, .supported {{ color: #047857; font-weight: 650; }}
    .fail, .warning {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f1; padding: 2px 4px; border-radius: 4px; }}
    a {{ color: #155e75; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Detector-Box Support Audit</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class="{html_lib.escape(str(summary.get('status', '')))}">{html_lib.escape(str(summary.get('status', '')))}</code>; claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>; samples={int(summary.get('sample_count', 0))}; source={source_link}</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Input Summary</h2>
  <table><thead><tr><th>Input</th><th>Detections</th><th>TP</th><th>FP</th><th>Precision Proxy</th><th>FP Score d/AUC</th><th>FP Edge d/AUC</th><th>FP Scene Edge d/AUC</th></tr></thead><tbody>{input_rows}</tbody></table>
  <h2>Target FP vs TP Correlations</h2>
  <table><thead><tr><th>Feature</th><th>FP Count</th><th>TP Count</th><th>FP Mean</th><th>TP Mean</th><th>dFP-TP</th><th>AUC Low Predicts FP</th><th>Lower Predicts FP</th></tr></thead><tbody>{target_rows}</tbody></table>
  <h2>Fusion To Calibrated Transition</h2>
  {transition_html}
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


def _input_row(row: Mapping[str, Any]) -> str:
    correlations = row.get("correlations", {}) if isinstance(row.get("correlations"), Mapping) else {}
    score = _correlation(correlations, "score")
    edge = _correlation(correlations, "edge_support")
    scene = _correlation(correlations, "scene_edge_support")
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('input_name', '')))}</code></td>"
        f"<td>{int(row.get('detection_count', 0))}</td>"
        f"<td>{int(row.get('tp_count', 0))}</td>"
        f"<td>{int(row.get('fp_count', 0))}</td>"
        f"<td>{_fmt(row.get('precision_proxy'))}</td>"
        f"<td>{_delta_auc(score)}</td>"
        f"<td>{_delta_auc(edge)}</td>"
        f"<td>{_delta_auc(scene)}</td>"
        "</tr>"
    )


def _correlation_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('feature', '')))}</code></td>"
        f"<td>{int(row.get('fp_count', 0))}</td>"
        f"<td>{int(row.get('tp_count', 0))}</td>"
        f"<td>{_fmt(row.get('fp_mean'))}</td>"
        f"<td>{_fmt(row.get('tp_mean'))}</td>"
        f"<td>{_fmt(row.get('delta_fp_minus_tp'), signed=True)}</td>"
        f"<td>{_fmt(row.get('auc_low_feature_predicts_fp'))}</td>"
        f"<td>{bool(row.get('lower_feature_predicts_fp'))}</td>"
        "</tr>"
    )


def _transition_html(transition_bridge: Any) -> str:
    if not isinstance(transition_bridge, Mapping):
        return "<p>No transition bridge was available.</p>"
    edge = _bridge_correlation(transition_bridge, "edge_support")
    scene = _bridge_correlation(transition_bridge, "scene_edge_support")
    aux = _bridge_correlation(transition_bridge, "aux_support")
    return (
        "<table><tbody>"
        f"<tr><th>Samples</th><td>{int(transition_bridge.get('compared_sample_count', 0))}</td><th>Baseline Detections</th><td>{int(transition_bridge.get('baseline_detection_count', 0))}</td><th>Target Detections</th><td>{int(transition_bridge.get('target_detection_count', 0))}</td></tr>"
        f"<tr><th>Removed FP</th><td>{int(transition_bridge.get('removed_fp_count', 0))}</td><th>Removed TP</th><td>{int(transition_bridge.get('removed_tp_count', 0))}</td><th>FP Delta</th><td>{int(transition_bridge.get('fp_delta_count', 0))}</td></tr>"
        f"<tr><th>Removed FP Aux d/AUC</th><td>{_delta_auc(aux)}</td><th>Removed FP Edge d/AUC</th><td>{_delta_auc(edge)}</td><th>Removed FP Scene d/AUC</th><td>{_delta_auc(scene)}</td></tr>"
        "</tbody></table>"
    )


def _delta_auc(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "n/a"
    return f"{_fmt(row.get('delta_fp_minus_tp'), signed=True)} / {_fmt(row.get('auc_low_feature_predicts_fp'))}"


def _source_link(summary: Mapping[str, Any]) -> str:
    html_path = str(summary.get("source_comparison_html", ""))
    if not html_path:
        return html_lib.escape(str(summary.get("source_comparison_summary", "")))
    return f"<a href=\"{html_lib.escape(html_path)}\">comparison</a>"


def _dedupe_inputs(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


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
