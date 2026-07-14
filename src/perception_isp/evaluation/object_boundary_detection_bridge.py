"""Bridge object-boundary edge evidence to detector TP/miss outcomes."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from perception_isp.core.types import json_ready


OBJECT_BOUNDARY_DETECTION_BRIDGE_SUMMARY = "object_boundary_detection_bridge_summary.json"
OBJECT_BOUNDARY_EDGE_SUMMARY = "object_boundary_edge_summary.json"
COMPARISON_SUMMARY = "comparison_summary.json"
FEATURES = (
    "human_rgb_edge_boundary_f1",
    "perception_rgb_edge_boundary_f1",
    "aux_edge_strength_boundary_f1",
    "aux_edge_confidence_boundary_f1",
    "perception_rgb_minus_human_boundary_f1",
    "aux_strength_minus_human_boundary_f1",
    "aux_confidence_minus_human_boundary_f1",
    "human_rgb_edge_boundary_separation",
    "perception_rgb_edge_boundary_separation",
    "aux_edge_strength_boundary_separation",
    "aux_edge_confidence_boundary_separation",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge object-box-boundary edge metrics to detector TP/miss outcomes.")
    parser.add_argument("--object-boundary-edge", required=True, help="object_boundary_edge_summary.json path or report dir.")
    parser.add_argument("--comparison-report", required=True, help="comparison_summary.json path or comparison report dir.")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--target-input", default="perception_calibrated_score_label_aux_fusion_rgb_aux_t001")
    parser.add_argument("--label-agnostic", action="store_true", help="Match detections to GT without requiring labels.")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="reports/perception_object_boundary_detection_bridge")
    args = parser.parse_args(argv)

    object_boundary_path = _summary_path(args.object_boundary_edge, OBJECT_BOUNDARY_EDGE_SUMMARY)
    comparison_path = _summary_path(args.comparison_report, COMPARISON_SUMMARY)
    object_boundary = json.loads(object_boundary_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    summary = build_object_boundary_detection_bridge(
        object_boundary,
        comparison,
        baseline_input=str(args.baseline_input),
        target_input=str(args.target_input),
        label_agnostic=bool(args.label_agnostic),
        iou_threshold=float(args.iou_threshold),
        object_boundary_summary_path=object_boundary_path,
        comparison_summary_path=comparison_path,
    )
    html_path = write_object_boundary_detection_bridge(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / OBJECT_BOUNDARY_DETECTION_BRIDGE_SUMMARY),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "object_count": summary["object_count"],
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_object_boundary_detection_bridge(
    object_boundary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    baseline_input: str = "human_rgb",
    target_input: str = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
    label_agnostic: bool | None = None,
    iou_threshold: float = 0.50,
    object_boundary_summary_path: str | Path | None = None,
    comparison_summary_path: str | Path | None = None,
) -> Dict[str, Any]:
    comparison_samples = [row for row in comparison.get("samples", ()) if isinstance(row, Mapping)]
    comparison_by_id = {str(row.get("sample_id", index)): row for index, row in enumerate(comparison_samples)}
    run_config = comparison.get("run_config", {}) if isinstance(comparison.get("run_config"), Mapping) else {}
    resolved_label_agnostic = bool(run_config.get("label_agnostic", False)) if label_agnostic is None else bool(label_agnostic)
    rows: list[Dict[str, Any]] = []
    compared_sample_ids: set[str] = set()
    missing_sample_ids: list[str] = []
    baseline_input_sample_count = 0
    target_input_sample_count = 0
    for case in object_boundary.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        sample_id = str(case.get("id", ""))
        comparison_sample = comparison_by_id.get(sample_id)
        if comparison_sample is None:
            missing_sample_ids.append(sample_id)
            continue
        compared_sample_ids.add(sample_id)
        baseline_detections = _detections_for_sample(comparison_sample, baseline_input)
        target_detections = _detections_for_sample(comparison_sample, target_input)
        if baseline_detections:
            baseline_input_sample_count += 1
        if target_detections:
            target_input_sample_count += 1
        box_rows = [row for row in case.get("box_rows", ()) if isinstance(row, Mapping)]
        baseline_matches = _gt_detection_matches(
            box_rows,
            baseline_detections,
            label_agnostic=resolved_label_agnostic,
            iou_threshold=float(iou_threshold),
        )
        target_matches = _gt_detection_matches(
            box_rows,
            target_detections,
            label_agnostic=resolved_label_agnostic,
            iou_threshold=float(iou_threshold),
        )
        for box_index, box_row in enumerate(box_rows):
            if not isinstance(box_row, Mapping):
                continue
            rows.append(
                _bridge_row(
                    box_row,
                    baseline_match=baseline_matches.get(box_index),
                    target_match=target_matches.get(box_index),
                    baseline_input=baseline_input,
                    target_input=target_input,
                )
            )
    checks = _checks(
        rows,
        compared_sample_count=len(compared_sample_ids),
        missing_sample_count=len(missing_sample_ids),
        baseline_input_sample_count=baseline_input_sample_count,
        target_input_sample_count=target_input_sample_count,
    )
    aggregate = _aggregate(rows)
    return {
        "name": "Object-boundary detection bridge",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "pass": all(row["status"] == "pass" for row in checks),
        "claim_status": "object_boundary_detection_bridge_diagnostic",
        "source_object_boundary_summary": "" if object_boundary_summary_path is None else str(object_boundary_summary_path),
        "source_comparison_summary": "" if comparison_summary_path is None else str(comparison_summary_path),
        "baseline_input": str(baseline_input),
        "target_input": str(target_input),
        "label_agnostic": resolved_label_agnostic,
        "iou_threshold": float(iou_threshold),
        "sample_count": len(compared_sample_ids),
        "object_count": len(rows),
        "missing_sample_count": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids[:20],
        "checks": checks,
        "aggregate": aggregate,
        "group_breakdown": _group_breakdown(rows),
        "label_breakdown": _label_breakdown(rows),
        "correlations": _correlations(rows),
        "examples": _examples(rows),
        "interpretation": (
            "This diagnostic bridge joins object-box-boundary edge metrics with detector true-positive/miss outcomes "
            "on the same samples. It tests whether stronger HumanISP, PerceptionISP, or aux edge evidence correlates "
            "with object recall decisions."
        ),
        "claim_boundary": (
            "The bridge uses KITTI/YOLO boxes, not segmentation contours. It explains GT-object TP/miss behavior; "
            "it is not a false-positive detector-box boundary audit and not a trained RGB+Aux DNN claim."
        ),
    }


def write_object_boundary_detection_bridge(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / OBJECT_BOUNDARY_DETECTION_BRIDGE_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _bridge_row(
    box_row: Mapping[str, Any],
    *,
    baseline_match: Mapping[str, Any] | None,
    target_match: Mapping[str, Any] | None,
    baseline_input: str,
    target_input: str,
) -> Dict[str, Any]:
    baseline_detected = baseline_match is not None
    target_detected = target_match is not None
    if baseline_detected and target_detected:
        status = "both_detected"
    elif target_detected:
        status = "target_only_detected"
    elif baseline_detected:
        status = "baseline_only_detected"
    else:
        status = "missed_by_both"
    row = {
        "sample_id": str(box_row.get("sample_id", "")),
        "label": str(box_row.get("label", "")),
        "area": _maybe_float(box_row.get("area")),
        "area_bucket": str(box_row.get("area_bucket", "")),
        "xyxy": [float(value) for value in box_row.get("xyxy", ())],
        "status": status,
        "baseline_input": str(baseline_input),
        "target_input": str(target_input),
        "baseline_detected": baseline_detected,
        "target_detected": target_detected,
        "baseline_score": _maybe_float((baseline_match or {}).get("score")),
        "target_score": _maybe_float((target_match or {}).get("score")),
        "baseline_iou": _maybe_float((baseline_match or {}).get("iou")),
        "target_iou": _maybe_float((target_match or {}).get("iou")),
        "score_delta": _maybe_float((target_match or {}).get("score")) - _maybe_float((baseline_match or {}).get("score")),
    }
    for feature in FEATURES:
        row[feature] = _maybe_float(box_row.get(feature))
    return row


def _checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    compared_sample_count: int,
    missing_sample_count: int,
    baseline_input_sample_count: int,
    target_input_sample_count: int,
) -> list[Dict[str, Any]]:
    finite = all(np.isfinite(_maybe_float(row.get(feature))) for row in rows for feature in FEATURES)
    target_positive = sum(1 for row in rows if bool(row.get("target_detected")))
    target_negative = len(rows) - target_positive
    baseline_positive = sum(1 for row in rows if bool(row.get("baseline_detected")))
    baseline_negative = len(rows) - baseline_positive
    return [
        {
            "id": "object_boundary_rows_matched_to_comparison_samples",
            "description": "Object-boundary rows should overlap the comparison report samples.",
            "status": "pass" if rows and compared_sample_count > 0 and missing_sample_count == 0 else "fail",
            "criteria": [
                {"metric": "object_count", "value": len(rows), "threshold": 1, "pass": len(rows) > 0},
                {"metric": "compared_sample_count", "value": int(compared_sample_count), "threshold": 1, "pass": int(compared_sample_count) > 0},
                {"metric": "missing_sample_count", "value": int(missing_sample_count), "threshold": 0, "pass": int(missing_sample_count) == 0},
            ],
        },
        {
            "id": "baseline_and_target_inputs_present",
            "description": "Both baseline and target detector inputs should be present on matched samples.",
            "status": "pass" if baseline_input_sample_count > 0 and target_input_sample_count > 0 else "fail",
            "criteria": [
                {"metric": "baseline_input_sample_count", "value": int(baseline_input_sample_count), "threshold": 1, "pass": int(baseline_input_sample_count) > 0},
                {"metric": "target_input_sample_count", "value": int(target_input_sample_count), "threshold": 1, "pass": int(target_input_sample_count) > 0},
            ],
        },
        {
            "id": "object_boundary_bridge_features_finite",
            "description": "All joined edge features should be finite.",
            "status": "pass" if finite else "fail",
            "criteria": [{"metric": "finite_features", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "target_detection_correlation_computable",
            "description": "Target detected/missed groups should both be present so edge correlation can be measured.",
            "status": "pass" if target_positive > 0 and target_negative > 0 else "fail",
            "criteria": [
                {"metric": "target_detected_count", "value": int(target_positive), "threshold": 1, "pass": target_positive > 0},
                {"metric": "target_missed_count", "value": int(target_negative), "threshold": 1, "pass": target_negative > 0},
            ],
        },
        {
            "id": "baseline_detection_correlation_computable",
            "description": "Baseline detected/missed groups should both be present so edge correlation can be measured.",
            "status": "pass" if baseline_positive > 0 and baseline_negative > 0 else "fail",
            "criteria": [
                {"metric": "baseline_detected_count", "value": int(baseline_positive), "threshold": 1, "pass": baseline_positive > 0},
                {"metric": "baseline_missed_count", "value": int(baseline_negative), "threshold": 1, "pass": baseline_negative > 0},
            ],
        },
    ]


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    baseline_detected = [row for row in rows if bool(row.get("baseline_detected"))]
    target_detected = [row for row in rows if bool(row.get("target_detected"))]
    target_only = [row for row in rows if row.get("status") == "target_only_detected"]
    baseline_only = [row for row in rows if row.get("status") == "baseline_only_detected"]
    missed = [row for row in rows if row.get("status") == "missed_by_both"]
    both = [row for row in rows if row.get("status") == "both_detected"]
    return {
        "object_count": total,
        "baseline_detected_count": len(baseline_detected),
        "target_detected_count": len(target_detected),
        "both_detected_count": len(both),
        "target_only_detected_count": len(target_only),
        "baseline_only_detected_count": len(baseline_only),
        "missed_by_both_count": len(missed),
        "baseline_recall_proxy": len(baseline_detected) / max(float(total), 1.0),
        "target_recall_proxy": len(target_detected) / max(float(total), 1.0),
        "target_minus_baseline_recall_proxy": (len(target_detected) - len(baseline_detected)) / max(float(total), 1.0),
        "mean_baseline_score": _mean([_maybe_float(row.get("baseline_score")) for row in baseline_detected]),
        "mean_target_score": _mean([_maybe_float(row.get("target_score")) for row in target_detected]),
        "mean_score_delta_on_both_detected": _mean([_maybe_float(row.get("score_delta")) for row in both]),
    }


def _group_breakdown(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups = ("both_detected", "target_only_detected", "baseline_only_detected", "missed_by_both")
    return [_feature_summary(group, [row for row in rows if row.get("status") == group], key_name="status") for group in groups]


def _label_breakdown(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    labels = sorted({str(row.get("label", "")) for row in rows})
    return [_feature_summary(label, [row for row in rows if str(row.get("label", "")) == label], key_name="label") for label in labels]


def _feature_summary(name: str, rows: Sequence[Mapping[str, Any]], *, key_name: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {key_name: str(name), "object_count": len(rows)}
    payload["baseline_detected_count"] = sum(1 for row in rows if bool(row.get("baseline_detected")))
    payload["target_detected_count"] = sum(1 for row in rows if bool(row.get("target_detected")))
    payload["target_recall_proxy"] = payload["target_detected_count"] / max(float(len(rows)), 1.0)
    for feature in FEATURES:
        payload[f"{feature}_mean"] = _mean([_maybe_float(row.get(feature)) for row in rows])
    return payload


def _correlations(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result_rows: list[Dict[str, Any]] = []
    comparisons = (
        ("target_detected_vs_missed", "target_detected"),
        ("baseline_detected_vs_missed", "baseline_detected"),
        ("target_only_vs_baseline_only", "target_only"),
    )
    for feature in FEATURES:
        for comparison_id, mode in comparisons:
            row = _correlation_row(rows, feature, comparison_id, mode)
            if row is not None:
                result_rows.append(row)
    key_results = {
        f"{row['comparison']}_{row['feature']}_{field}": row[field]
        for row in result_rows
        for field in ("positive_mean", "negative_mean", "delta", "auc_high_feature_predicts_positive", "point_biserial")
    }
    return {
        "status": "pass" if result_rows else "missing",
        "rows": result_rows,
        "key_results": key_results,
        "interpretation": "AUC is computed so higher feature values predicting the positive group score above 0.5.",
    }


def _correlation_row(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
    comparison_id: str,
    mode: str,
) -> Dict[str, Any] | None:
    if mode == "target_detected":
        positives = [_maybe_float(row.get(feature)) for row in rows if bool(row.get("target_detected"))]
        negatives = [_maybe_float(row.get(feature)) for row in rows if not bool(row.get("target_detected"))]
    elif mode == "baseline_detected":
        positives = [_maybe_float(row.get(feature)) for row in rows if bool(row.get("baseline_detected"))]
        negatives = [_maybe_float(row.get(feature)) for row in rows if not bool(row.get("baseline_detected"))]
    elif mode == "target_only":
        positives = [_maybe_float(row.get(feature)) for row in rows if row.get("status") == "target_only_detected"]
        negatives = [_maybe_float(row.get(feature)) for row in rows if row.get("status") == "baseline_only_detected"]
    else:
        return None
    positives = [value for value in positives if np.isfinite(value)]
    negatives = [value for value in negatives if np.isfinite(value)]
    if not positives or not negatives:
        return None
    positive_mean = _mean(positives)
    negative_mean = _mean(negatives)
    delta = positive_mean - negative_mean
    auc = _binary_auc(positives, negatives)
    return {
        "comparison": comparison_id,
        "feature": feature,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "delta": delta,
        "point_biserial": _point_biserial(positives, negatives),
        "auc_high_feature_predicts_positive": auc,
        "higher_feature_predicts_positive": delta > 0.0 and auc > 0.5,
    }


def _examples(rows: Sequence[Mapping[str, Any]], limit: int = 16) -> list[Dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row.get("status") in ("target_only_detected", "baseline_only_detected") else 1,
            -abs(_maybe_float(row.get("aux_confidence_minus_human_boundary_f1"))),
        ),
    )
    return [dict(row) for row in ordered[: int(limit)]]


def _gt_detection_matches(
    gt_rows: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    label_agnostic: bool,
    iou_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    candidates: list[tuple[float, float, int, int]] = []
    for gt_index, gt in enumerate(gt_rows):
        gt_box = _box_payload(gt)
        for detection_index, detection in enumerate(detections):
            det_box = _box_payload(detection)
            if not label_agnostic and str(det_box.get("label", "")) != str(gt_box.get("label", "")):
                continue
            iou = _box_iou(gt_box, det_box)
            if iou >= float(iou_threshold):
                candidates.append((iou, _maybe_float(detection.get("score")), gt_index, detection_index))
    matches: Dict[int, Dict[str, Any]] = {}
    used_detections: set[int] = set()
    used_gt: set[int] = set()
    for iou, _score, gt_index, detection_index in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
        if gt_index in used_gt or detection_index in used_detections:
            continue
        detection = detections[detection_index]
        det_box = _box_payload(detection)
        matches[gt_index] = {
            "score": _maybe_float(detection.get("score")),
            "iou": iou,
            "label": str(det_box.get("label", "")),
        }
        used_gt.add(gt_index)
        used_detections.add(detection_index)
    return matches


def _detections_for_sample(sample: Mapping[str, Any], input_name: str) -> list[Mapping[str, Any]]:
    for detector in sample.get("detectors", ()):
        if isinstance(detector, Mapping) and str(detector.get("input_name")) == str(input_name):
            return [row for row in detector.get("detections", ()) if isinstance(row, Mapping)]
    return []


def _box_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    box = payload.get("box", payload)
    if not isinstance(box, Mapping):
        box = {}
    values = tuple(float(value) for value in box.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    if len(values) != 4:
        values = (0.0, 0.0, 0.0, 0.0)
    return {"xyxy": values, "label": str(box.get("label", "object"))}


def _box_iou(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = tuple(float(value) for value in a.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    bx1, by1, bx2, by2 = tuple(float(value) for value in b.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    intersection_w = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    intersection_h = max(min(ay2, by2) - max(ay1, by1), 0.0)
    intersection = intersection_w * intersection_h
    union = _box_area(a) + _box_area(b) - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


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
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    group_rows = "".join(_breakdown_row(row, "status") for row in summary.get("group_breakdown", ()) if isinstance(row, Mapping))
    label_rows = "".join(_breakdown_row(row, "label") for row in summary.get("label_breakdown", ()) if isinstance(row, Mapping))
    correlation_rows = "".join(_correlation_html_row(row) for row in summary.get("correlations", {}).get("rows", ()) if isinstance(row, Mapping))
    example_rows = "".join(_example_row(row) for row in summary.get("examples", ()) if isinstance(row, Mapping))
    status_class = "supported" if bool(summary.get("pass")) else "not_supported"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Object Boundary Detection Bridge</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .supported {{ color: #047857; font-weight: 700; }}
    .not_supported {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Object Boundary Detection Bridge</h1>
  <p>Status: <code class=\"{status_class}\">{html_lib.escape(str(summary.get('status', '')))}</code>;
  claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.
  {html_lib.escape(str(summary.get('interpretation', '')))}
  {html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <table><thead><tr><th>Samples</th><th>Objects</th><th>Baseline</th><th>Target</th><th>Baseline Recall Proxy</th><th>Target Recall Proxy</th><th>Delta</th><th>Target Only</th><th>Baseline Only</th><th>Missed Both</th></tr></thead><tbody><tr>
    <td>{int(summary.get('sample_count', 0))}</td>
    <td>{int(summary.get('object_count', 0))}</td>
    <td><code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code></td>
    <td><code>{html_lib.escape(str(summary.get('target_input', '')))}</code></td>
    <td>{_fmt(aggregate.get('baseline_recall_proxy'))}</td>
    <td>{_fmt(aggregate.get('target_recall_proxy'))}</td>
    <td>{_fmt(aggregate.get('target_minus_baseline_recall_proxy'), signed=True)}</td>
    <td>{int(aggregate.get('target_only_detected_count', 0))}</td>
    <td>{int(aggregate.get('baseline_only_detected_count', 0))}</td>
    <td>{int(aggregate.get('missed_by_both_count', 0))}</td>
  </tr></tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>ID</th><th>Status</th><th>Description</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Outcome Groups</h2>
  <table><thead><tr><th>Status</th><th>Objects</th><th>Target Recall Proxy</th><th>Human F1</th><th>Perception F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th><th>Aux Confidence Delta</th></tr></thead><tbody>{group_rows}</tbody></table>
  <h2>Label Breakdown</h2>
  <table><thead><tr><th>Label</th><th>Objects</th><th>Target Recall Proxy</th><th>Human F1</th><th>Perception F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th><th>Aux Confidence Delta</th></tr></thead><tbody>{label_rows}</tbody></table>
  <h2>Feature Correlations</h2>
  <table><thead><tr><th>Comparison</th><th>Feature</th><th>Positive/Negative</th><th>Positive Mean</th><th>Negative Mean</th><th>Delta</th><th>AUC</th><th>Point Biserial</th></tr></thead><tbody>{correlation_rows}</tbody></table>
  <h2>Examples</h2>
  <table><thead><tr><th>Sample</th><th>Label</th><th>Status</th><th>Baseline Score</th><th>Target Score</th><th>Human F1</th><th>Aux Confidence F1</th><th>Aux Confidence Delta</th></tr></thead><tbody>{example_rows}</tbody></table>
  <p>Raw JSON: <code>{OBJECT_BOUNDARY_DETECTION_BRIDGE_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{'supported' if status == 'pass' else 'not_supported'}\">{html_lib.escape(status)}</td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        "</tr>"
    )


def _breakdown_row(row: Mapping[str, Any], key: str) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get(key, '')))}</code></td>"
        f"<td>{int(row.get('object_count', 0))}</td>"
        f"<td>{_fmt(row.get('target_recall_proxy'))}</td>"
        f"<td>{_fmt(row.get('human_rgb_edge_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('perception_rgb_edge_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('aux_edge_strength_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('aux_edge_confidence_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('aux_confidence_minus_human_boundary_f1_mean'), signed=True)}</td>"
        "</tr>"
    )


def _correlation_html_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('comparison', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('feature', '')))}</code></td>"
        f"<td>{int(row.get('positive_count', 0))}/{int(row.get('negative_count', 0))}</td>"
        f"<td>{_fmt(row.get('positive_mean'))}</td>"
        f"<td>{_fmt(row.get('negative_mean'))}</td>"
        f"<td>{_fmt(row.get('delta'), signed=True)}</td>"
        f"<td>{_fmt(row.get('auc_high_feature_predicts_positive'))}</td>"
        f"<td>{_fmt(row.get('point_biserial'), signed=True)}</td>"
        "</tr>"
    )


def _example_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('label', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('status', '')))}</code></td>"
        f"<td>{_fmt(row.get('baseline_score'))}</td>"
        f"<td>{_fmt(row.get('target_score'))}</td>"
        f"<td>{_fmt(row.get('human_rgb_edge_boundary_f1'))}</td>"
        f"<td>{_fmt(row.get('aux_edge_confidence_boundary_f1'))}</td>"
        f"<td>{_fmt(row.get('aux_confidence_minus_human_boundary_f1'), signed=True)}</td>"
        "</tr>"
    )


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else 0.0


def _maybe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = _maybe_float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
