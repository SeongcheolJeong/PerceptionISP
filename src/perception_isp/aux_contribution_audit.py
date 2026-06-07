"""Audit whether aux evidence contributes to proposal score calibration."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


AUX_CONTRIBUTION_SUMMARY = "aux_contribution_audit_summary.json"
DEFAULT_BASELINE_INPUT = "perception_fusion_rgb_aux"
DEFAULT_SCORE_AUX_INPUT = "perception_calibrated_score_aux_fusion_rgb_aux"
DEFAULT_SCORE_LABEL_INPUT = "perception_calibrated_score_label_fusion_rgb_aux"
DEFAULT_SCORE_LABEL_AUX_INPUT = "perception_calibrated_score_label_aux_fusion_rgb_aux"
TRACKED_METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "det_count_mean",
)
AUX_FEATURE_NAMES = {
    "aux_support",
    "edge_support",
    "saturation_support",
    "reliability_support",
    "aux_box_iou",
    "score_x_aux",
    "score_x_reliability",
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit aux contribution in proposal calibration feature ablations.")
    parser.add_argument("comparison_rollup", help="rollup_summary.json path or report-rollup directory.")
    parser.add_argument("--calibration-summary", default=None, help="Optional proposal_calibration_summary.json path/dir.")
    parser.add_argument("--baseline-input", default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--score-aux-input", default=DEFAULT_SCORE_AUX_INPUT)
    parser.add_argument("--score-label-input", default=DEFAULT_SCORE_LABEL_INPUT)
    parser.add_argument("--score-label-aux-input", default=DEFAULT_SCORE_LABEL_AUX_INPUT)
    parser.add_argument("--recall-floor", type=float, default=-0.005)
    parser.add_argument("--min-fp-reduction", type=float, default=0.02)
    parser.add_argument("--min-precision-delta", type=float, default=0.0)
    parser.add_argument("--output-dir", default="reports/perception_aux_contribution_audit")
    args = parser.parse_args(argv)

    summary = build_aux_contribution_audit_from_paths(
        args.comparison_rollup,
        calibration_summary=args.calibration_summary,
        baseline_input=str(args.baseline_input),
        score_aux_input=str(args.score_aux_input),
        score_label_input=str(args.score_label_input),
        score_label_aux_input=str(args.score_label_aux_input),
        recall_floor=float(args.recall_floor),
        min_fp_reduction=float(args.min_fp_reduction),
        min_precision_delta=float(args.min_precision_delta),
    )
    html_path = write_aux_contribution_audit(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / AUX_CONTRIBUTION_SUMMARY),
                    "status": summary["status"],
                    "check_count": len(summary["checks"]),
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_aux_contribution_audit_from_paths(
    comparison_rollup: str | Path,
    *,
    calibration_summary: str | Path | None = None,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    score_aux_input: str = DEFAULT_SCORE_AUX_INPUT,
    score_label_input: str = DEFAULT_SCORE_LABEL_INPUT,
    score_label_aux_input: str = DEFAULT_SCORE_LABEL_AUX_INPUT,
    recall_floor: float = -0.005,
    min_fp_reduction: float = 0.02,
    min_precision_delta: float = 0.0,
) -> Dict[str, Any]:
    rollup_path = _summary_path(comparison_rollup, "rollup_summary.json")
    rollup = json.loads(rollup_path.read_text())
    calibration = None
    calibration_path = ""
    if calibration_summary is not None:
        path = _summary_path(calibration_summary, "proposal_calibration_summary.json")
        calibration = json.loads(path.read_text())
        calibration_path = str(path)
    return build_aux_contribution_audit(
        rollup,
        calibration_summary=calibration,
        source_rollup=rollup_path,
        calibration_summary_path=calibration_path,
        baseline_input=baseline_input,
        score_aux_input=score_aux_input,
        score_label_input=score_label_input,
        score_label_aux_input=score_label_aux_input,
        recall_floor=recall_floor,
        min_fp_reduction=min_fp_reduction,
        min_precision_delta=min_precision_delta,
    )


def build_aux_contribution_audit(
    rollup: Mapping[str, Any],
    *,
    calibration_summary: Mapping[str, Any] | None = None,
    source_rollup: str | Path | None = None,
    calibration_summary_path: str | Path | None = None,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    score_aux_input: str = DEFAULT_SCORE_AUX_INPUT,
    score_label_input: str = DEFAULT_SCORE_LABEL_INPUT,
    score_label_aux_input: str = DEFAULT_SCORE_LABEL_AUX_INPUT,
    recall_floor: float = -0.005,
    min_fp_reduction: float = 0.02,
    min_precision_delta: float = 0.0,
) -> Dict[str, Any]:
    inputs = _collect_inputs(rollup)
    required_inputs = (baseline_input, score_aux_input, score_label_input, score_label_aux_input)
    missing = [name for name in required_inputs if name not in inputs]
    comparisons = []
    checks = []
    if not missing:
        comparisons = [
            _comparison("score_aux_vs_fusion", score_aux_input, baseline_input, inputs[score_aux_input], inputs[baseline_input]),
            _comparison("score_label_vs_fusion", score_label_input, baseline_input, inputs[score_label_input], inputs[baseline_input]),
            _comparison("score_label_aux_vs_fusion", score_label_aux_input, baseline_input, inputs[score_label_aux_input], inputs[baseline_input]),
            _comparison("score_label_aux_vs_score_label", score_label_aux_input, score_label_input, inputs[score_label_aux_input], inputs[score_label_input]),
        ]
        by_id = {row["id"]: row for row in comparisons}
        checks.extend(
            [
                _metric_check(
                    "score_aux_uses_aux_for_fp_reduction",
                    "Score+aux calibration should reduce false positives versus uncalibrated RGB+Aux fusion within the recall budget.",
                    by_id["score_aux_vs_fusion"],
                    recall_floor=recall_floor,
                    min_fp_reduction=min_fp_reduction,
                    min_precision_delta=min_precision_delta,
                ),
                _metric_check(
                    "aux_adds_incremental_value_over_score_label",
                    "Adding aux features to score+label calibration should reduce false positives versus score+label alone within the recall budget.",
                    by_id["score_label_aux_vs_score_label"],
                    recall_floor=recall_floor,
                    min_fp_reduction=min_fp_reduction,
                    min_precision_delta=min_precision_delta,
                ),
            ]
        )
    else:
        checks.append(
            {
                "id": "required_calibration_inputs_present",
                "description": "Rollup must contain baseline, score+aux, score+label, and score+label+aux inputs.",
                "status": "fail",
                "criteria": [{"metric": "missing_inputs", "value": ", ".join(missing), "pass": False}],
            }
        )
    feature_audit = _feature_audit(calibration_summary) if calibration_summary is not None else None
    if feature_audit is not None:
        checks.append(
            {
                "id": "score_label_aux_model_contains_aux_features",
                "description": "The score_label_aux proposal calibrator should include explicit aux feature columns.",
                "status": "pass" if feature_audit["has_score_label_aux_model"] and feature_audit["aux_feature_count"] > 0 else "fail",
                "criteria": [
                    {"metric": "has_score_label_aux_model", "value": bool(feature_audit["has_score_label_aux_model"]), "pass": bool(feature_audit["has_score_label_aux_model"])},
                    {"metric": "aux_feature_count", "value": int(feature_audit["aux_feature_count"]), "threshold": 1, "pass": int(feature_audit["aux_feature_count"]) > 0},
                ],
            }
        )
    sample_bridge = _sample_bridge_from_inputs(inputs, baseline_input=score_label_input, target_input=score_label_aux_input)
    if sample_bridge is not None:
        checks.extend(_sample_bridge_checks(sample_bridge))
    return {
        "source_rollup": "" if source_rollup is None else str(source_rollup),
        "calibration_summary": "" if calibration_summary_path is None else str(calibration_summary_path),
        "baseline_input": str(baseline_input),
        "score_aux_input": str(score_aux_input),
        "score_label_input": str(score_label_input),
        "score_label_aux_input": str(score_label_aux_input),
        "thresholds": {
            "recall_floor": float(recall_floor),
            "min_fp_reduction": float(min_fp_reduction),
            "min_precision_delta": float(min_precision_delta),
        },
        "available_inputs": sorted(inputs),
        "missing_inputs": missing,
        "comparisons": comparisons,
        "feature_audit": feature_audit,
        "sample_bridge": sample_bridge,
        "checks": checks,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "fail",
        "interpretation": (
            "This audit checks whether aux features add measurable value to detector-side proposal scoring/filtering. "
            "It is downstream calibration evidence for aux usage, not a trained DNN or broad detector-performance claim."
        ),
    }


def write_aux_contribution_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / AUX_CONTRIBUTION_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _collect_inputs(rollup: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    inputs: Dict[str, Dict[str, Any]] = {}
    for run in rollup.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        run_inputs = run.get("inputs", {}) if isinstance(run.get("inputs"), Mapping) else {}
        for name, metrics in run_inputs.items():
            if not isinstance(metrics, Mapping):
                continue
            key = str(name)
            if key in inputs and len(metrics) < len(inputs[key].get("metrics", {})):
                continue
            inputs[key] = {
                "input": key,
                "metrics": {metric: _maybe_float(metrics.get(metric)) for metric in TRACKED_METRICS if metrics.get(metric) is not None},
                "run_name": str(run.get("name", "")),
                "summary_path": str(run.get("summary_path", "")),
                "html_path": run.get("html_path"),
                "sample_count": int(run.get("sample_count", 0)),
            }
    return inputs


def _comparison(
    comparison_id: str,
    target_input: str,
    baseline_input: str,
    target: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    target_metrics = target.get("metrics", {}) if isinstance(target.get("metrics"), Mapping) else {}
    baseline_metrics = baseline.get("metrics", {}) if isinstance(baseline.get("metrics"), Mapping) else {}
    deltas = {
        metric: _maybe_float(target_metrics.get(metric)) - _maybe_float(baseline_metrics.get(metric))
        for metric in TRACKED_METRICS
        if target_metrics.get(metric) is not None and baseline_metrics.get(metric) is not None
    }
    return {
        "id": str(comparison_id),
        "target_input": str(target_input),
        "baseline_input": str(baseline_input),
        "target": dict(target),
        "baseline": dict(baseline),
        "deltas": deltas,
    }


def _metric_check(
    check_id: str,
    description: str,
    comparison: Mapping[str, Any],
    *,
    recall_floor: float,
    min_fp_reduction: float,
    min_precision_delta: float,
) -> Dict[str, Any]:
    deltas = comparison.get("deltas", {}) if isinstance(comparison.get("deltas"), Mapping) else {}
    precision_delta = _maybe_float(deltas.get("precision@0.50_mean"))
    recall_delta = _maybe_float(deltas.get("recall@0.50_mean"))
    fp_delta = _maybe_float(deltas.get("fp@0.50_mean"))
    criteria = [
        {
            "metric": "precision@0.50_mean",
            "direction": "minimum_delta",
            "delta": precision_delta,
            "threshold": float(min_precision_delta),
            "pass": precision_delta >= float(min_precision_delta),
        },
        {
            "metric": "recall@0.50_mean",
            "direction": "minimum_delta",
            "delta": recall_delta,
            "threshold": float(recall_floor),
            "pass": recall_delta >= float(recall_floor),
        },
        {
            "metric": "fp@0.50_mean",
            "direction": "maximum_delta",
            "delta": fp_delta,
            "threshold": -float(min_fp_reduction),
            "pass": fp_delta <= -float(min_fp_reduction),
        },
    ]
    return {
        "id": str(check_id),
        "description": str(description),
        "comparison": comparison.get("id"),
        "status": "pass" if all(bool(row["pass"]) for row in criteria) else "fail",
        "criteria": criteria,
    }


def _feature_audit(summary: Mapping[str, Any]) -> Dict[str, Any]:
    models = [row for row in summary.get("models", ()) if isinstance(row, Mapping)]
    score_label_aux = next((row for row in models if str(row.get("feature_set", "")) == "score_label_aux"), None)
    feature_names = [str(name) for name in score_label_aux.get("feature_names", ())] if isinstance(score_label_aux, Mapping) else []
    aux_features = [name for name in feature_names if name in AUX_FEATURE_NAMES]
    weights = list(score_label_aux.get("weights", ())) if isinstance(score_label_aux, Mapping) else []
    weighted_aux = []
    for index, name in enumerate(feature_names):
        if name not in AUX_FEATURE_NAMES or index >= len(weights):
            continue
        weighted_aux.append({"feature": name, "weight": _maybe_float(weights[index])})
    return {
        "has_score_label_aux_model": score_label_aux is not None,
        "feature_count": len(feature_names),
        "aux_feature_count": len(aux_features),
        "aux_features": aux_features,
        "weighted_aux_features": weighted_aux,
    }


def _sample_bridge_from_inputs(inputs: Mapping[str, Mapping[str, Any]], *, baseline_input: str, target_input: str) -> Dict[str, Any] | None:
    baseline_report = _load_comparison_summary(inputs.get(str(baseline_input), {}).get("summary_path") if isinstance(inputs.get(str(baseline_input)), Mapping) else None)
    target_report = _load_comparison_summary(inputs.get(str(target_input), {}).get("summary_path") if isinstance(inputs.get(str(target_input)), Mapping) else None)
    if baseline_report is None or target_report is None:
        return None
    baseline_samples = [row for row in baseline_report.get("samples", ()) if isinstance(row, Mapping)]
    target_samples = [row for row in target_report.get("samples", ()) if isinstance(row, Mapping)]
    if not baseline_samples or not target_samples:
        return None
    target_by_id = {str(row.get("sample_id", index)): row for index, row in enumerate(target_samples)}
    label_agnostic = bool(
        (target_report.get("run_config", {}) if isinstance(target_report.get("run_config"), Mapping) else {}).get(
            "label_agnostic",
            (baseline_report.get("run_config", {}) if isinstance(baseline_report.get("run_config"), Mapping) else {}).get("label_agnostic", False),
        )
    )
    counts = {
        "compared_sample_count": 0,
        "baseline_detection_count": 0,
        "target_detection_count": 0,
        "common_detection_count": 0,
        "baseline_only_detection_count": 0,
        "target_only_detection_count": 0,
        "baseline_fp_count": 0,
        "target_fp_count": 0,
        "baseline_tp_count": 0,
        "target_tp_count": 0,
        "removed_fp_count": 0,
        "removed_tp_count": 0,
        "added_fp_count": 0,
        "added_tp_count": 0,
    }
    support_groups: Dict[str, list[Dict[str, float]]] = {
        "removed_fp": [],
        "removed_tp": [],
        "kept_fp": [],
        "kept_tp": [],
        "added_fp": [],
        "added_tp": [],
    }
    examples: list[Dict[str, Any]] = []
    for index, baseline_sample in enumerate(baseline_samples):
        sample_id = str(baseline_sample.get("sample_id", index))
        target_sample = target_by_id.get(sample_id)
        if target_sample is None:
            continue
        baseline_detections = _detections_for_sample(baseline_sample, baseline_input)
        target_detections = _detections_for_sample(target_sample, target_input)
        ground_truth = _ground_truth_for_sample(baseline_sample)
        baseline_positive = _positive_detection_indices(baseline_detections, ground_truth, label_agnostic=label_agnostic)
        target_positive = _positive_detection_indices(target_detections, ground_truth, label_agnostic=label_agnostic)
        counts["compared_sample_count"] += 1
        counts["baseline_detection_count"] += len(baseline_detections)
        counts["target_detection_count"] += len(target_detections)
        counts["baseline_tp_count"] += len(baseline_positive)
        counts["target_tp_count"] += len(target_positive)
        counts["baseline_fp_count"] += len(baseline_detections) - len(baseline_positive)
        counts["target_fp_count"] += len(target_detections) - len(target_positive)
        matched_targets: set[int] = set()
        for detection_index, detection in enumerate(baseline_detections):
            target_index = _matching_detection_index(detection, target_detections)
            is_tp = detection_index in baseline_positive
            if target_index >= 0:
                counts["common_detection_count"] += 1
                matched_targets.add(target_index)
                support_groups["kept_tp" if is_tp else "kept_fp"].append(_support_values(detection))
                continue
            counts["baseline_only_detection_count"] += 1
            if is_tp:
                counts["removed_tp_count"] += 1
                support_groups["removed_tp"].append(_support_values(detection))
            else:
                counts["removed_fp_count"] += 1
                support_groups["removed_fp"].append(_support_values(detection))
            if len(examples) < 12:
                examples.append(
                    {
                        "sample_id": sample_id,
                        "label": str((detection.get("box", {}) if isinstance(detection.get("box"), Mapping) else {}).get("label", "object")),
                        "baseline_score": _maybe_float(detection.get("score")),
                        "target_status": "removed_tp" if is_tp else "removed_fp",
                        **_support_values(detection),
                    }
                )
        for target_index, detection in enumerate(target_detections):
            if target_index in matched_targets:
                continue
            is_tp = target_index in target_positive
            counts["target_only_detection_count"] += 1
            if is_tp:
                counts["added_tp_count"] += 1
                support_groups["added_tp"].append(_support_values(detection))
            else:
                counts["added_fp_count"] += 1
                support_groups["added_fp"].append(_support_values(detection))
    if counts["compared_sample_count"] <= 0:
        return None
    removed_count = counts["baseline_only_detection_count"]
    counts["fp_delta_count"] = counts["target_fp_count"] - counts["baseline_fp_count"]
    counts["tp_delta_count"] = counts["target_tp_count"] - counts["baseline_tp_count"]
    counts["removed_fp_fraction"] = counts["removed_fp_count"] / max(float(removed_count), 1.0)
    counts["removed_tp_fraction"] = counts["removed_tp_count"] / max(float(removed_count), 1.0)
    counts["removed_fp_to_tp_ratio"] = counts["removed_fp_count"] / max(float(counts["removed_tp_count"]), 1.0)
    return {
        "status": "pass",
        "baseline_input": str(baseline_input),
        "target_input": str(target_input),
        "label_agnostic": label_agnostic,
        **counts,
        "support_means": {name: _support_mean(rows) for name, rows in support_groups.items()},
        "examples": examples,
        "interpretation": (
            "This same-sample bridge compares score_label and score_label_aux proposal outputs on matched samples. "
            "It shows which detections aux features remove or add, and whether those detections are true positives or false positives."
        ),
    }


def _sample_bridge_checks(sample_bridge: Mapping[str, Any]) -> list[Dict[str, Any]]:
    removed_count = int(sample_bridge.get("baseline_only_detection_count", 0))
    removed_fp = int(sample_bridge.get("removed_fp_count", 0))
    removed_tp = int(sample_bridge.get("removed_tp_count", 0))
    fp_delta = int(sample_bridge.get("fp_delta_count", 0))
    return [
        {
            "id": "same_sample_aux_bridge_available",
            "description": "Score_label and score_label_aux sample reports should overlap and expose per-detection aux metadata.",
            "status": "pass" if removed_count > 0 else "fail",
            "criteria": [
                {"metric": "compared_sample_count", "value": int(sample_bridge.get("compared_sample_count", 0)), "threshold": 1, "pass": int(sample_bridge.get("compared_sample_count", 0)) > 0},
                {"metric": "baseline_only_detection_count", "value": removed_count, "threshold": 1, "pass": removed_count > 0},
            ],
        },
        {
            "id": "incremental_aux_removes_more_fp_than_tp",
            "description": "Detections removed by adding aux features should be mostly false positives.",
            "status": "pass" if removed_fp > removed_tp else "fail",
            "criteria": [
                {"metric": "removed_fp_count", "value": removed_fp, "threshold": removed_tp, "pass": removed_fp > removed_tp},
                {"metric": "removed_fp_fraction", "value": _maybe_float(sample_bridge.get("removed_fp_fraction")), "threshold": 0.5, "pass": _maybe_float(sample_bridge.get("removed_fp_fraction")) > 0.5},
            ],
        },
        {
            "id": "incremental_aux_net_fp_reduction_same_sample",
            "description": "Same-sample target output should contain fewer false positives than score_label baseline output.",
            "status": "pass" if fp_delta < 0 else "fail",
            "criteria": [{"metric": "fp_delta_count", "value": fp_delta, "threshold": 0, "pass": fp_delta < 0}],
        },
    ]


def _load_comparison_summary(path_value: Any) -> Dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_dir():
        path = path / "comparison_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None


def _detections_for_sample(sample: Mapping[str, Any], input_name: str) -> list[Mapping[str, Any]]:
    for detector in sample.get("detectors", ()):
        if isinstance(detector, Mapping) and str(detector.get("input_name")) == str(input_name):
            return [row for row in detector.get("detections", ()) if isinstance(row, Mapping)]
    return []


def _ground_truth_for_sample(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in sample.get("ground_truth", ()) if isinstance(row, Mapping)]


def _positive_detection_indices(detections: Sequence[Mapping[str, Any]], ground_truth: Sequence[Mapping[str, Any]], *, label_agnostic: bool) -> set[int]:
    ordered = sorted(range(len(detections)), key=lambda index: _maybe_float(detections[index].get("score")), reverse=True)
    used_gt: set[int] = set()
    positives: set[int] = set()
    for detection_index in ordered:
        detection_box = _box_payload(detections[detection_index])
        best_iou = 0.0
        best_gt = -1
        for gt_index, gt_payload in enumerate(ground_truth):
            if gt_index in used_gt:
                continue
            gt_box = _box_payload(gt_payload)
            if not label_agnostic and str(detection_box.get("label", "")) != str(gt_box.get("label", "")):
                continue
            value = _box_iou(detection_box, gt_box)
            if value > best_iou:
                best_iou = value
                best_gt = gt_index
        if best_iou >= 0.50 and best_gt >= 0:
            positives.add(detection_index)
            used_gt.add(best_gt)
    return positives


def _matching_detection_index(detection: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> int:
    box = _box_payload(detection)
    for index, target in enumerate(targets):
        target_box = _box_payload(target)
        if str(box.get("label", "")) == str(target_box.get("label", "")) and _box_iou(box, target_box) >= 0.99:
            return index
    return -1


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


def _support_values(detection: Mapping[str, Any]) -> Dict[str, float]:
    metadata = detection.get("metadata", {}) if isinstance(detection.get("metadata"), Mapping) else {}
    fusion = metadata.get("fusion", {}) if isinstance(metadata.get("fusion"), Mapping) else {}
    return {
        "score": _maybe_float(detection.get("score")),
        "aux_support": _maybe_float(fusion.get("aux_support")),
        "edge_support": _maybe_float(fusion.get("edge_support")),
        "saturation_support": _maybe_float(fusion.get("saturation_support")),
        "reliability_support": _maybe_float(fusion.get("reliability_support")),
        "aux_box_iou": _maybe_float(fusion.get("aux_box_iou")),
    }


def _support_mean(rows: Sequence[Mapping[str, float]]) -> Dict[str, float | int]:
    if not rows:
        return {"count": 0}
    keys = ("score", "aux_support", "edge_support", "saturation_support", "reliability_support", "aux_box_iou")
    return {"count": len(rows), **{f"{key}_mean": sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}}


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()))
    comparison_rows = "".join(_comparison_row(row, destination) for row in summary.get("comparisons", ()))
    if not comparison_rows:
        comparison_rows = '<tr><td colspan="10">No comparisons were available.</td></tr>'
    feature_audit = summary.get("feature_audit")
    feature_html = _feature_html(feature_audit) if isinstance(feature_audit, Mapping) else "<p>No calibration-summary feature audit was provided.</p>"
    sample_bridge = summary.get("sample_bridge")
    sample_bridge_html = _sample_bridge_html(sample_bridge) if isinstance(sample_bridge, Mapping) else "<p>No same-sample bridge audit was available.</p>"
    status = str(summary.get("status", ""))
    status_class = "pass" if status == "pass" else "fail"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Aux Contribution Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Aux Contribution Audit</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <p>Status: <code class=\"{status_class}\">{html_lib.escape(status)}</code>. Baseline: <code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code>.</p>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Status</th><th>Check</th><th>Description</th><th>Criteria</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>
  <h2>Feature-Set Comparisons</h2>
  <table>
    <thead><tr><th>Comparison</th><th>Target</th><th>Baseline</th><th>dP@0.50</th><th>dR@0.50</th><th>dR@0.75</th><th>dSmallR@0.50</th><th>dFP@0.50</th><th>Target Report</th><th>Baseline Report</th></tr></thead>
    <tbody>{comparison_rows}</tbody>
  </table>
  <h2>Aux Feature Audit</h2>
  {feature_html}
  <h2>Same-Sample Bridge</h2>
  {sample_bridge_html}
  <p>Raw JSON: <code>{AUX_CONTRIBUTION_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    criteria = "; ".join(
        f"{item.get('metric')} d={_fmt(item.get('delta'), signed=True)} threshold={_fmt(item.get('threshold'), signed=True)} pass={bool(item.get('pass'))}"
        if item.get("delta") is not None
        else f"{item.get('metric')} value={item.get('value')} pass={bool(item.get('pass'))}"
        for item in row.get("criteria", ())
        if isinstance(item, Mapping)
    )
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape(criteria)}</td>"
        "</tr>"
    )


def _comparison_row(row: Mapping[str, Any], destination: Path) -> str:
    deltas = row.get("deltas", {}) if isinstance(row.get("deltas"), Mapping) else {}
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('target_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('baseline_input', '')))}</code></td>"
        f"<td>{_fmt(deltas.get('precision@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('recall@0.75_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('small_recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('fp@0.50_mean'), signed=True)}</td>"
        f"<td>{_report_link(row.get('target', {}), destination)}</td>"
        f"<td>{_report_link(row.get('baseline', {}), destination)}</td>"
        "</tr>"
    )


def _feature_html(feature_audit: Mapping[str, Any]) -> str:
    aux_rows = "".join(
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('feature', '')))}</code></td>"
        f"<td>{_fmt(row.get('weight'), signed=True)}</td>"
        "</tr>"
        for row in feature_audit.get("weighted_aux_features", ())
        if isinstance(row, Mapping)
    )
    if not aux_rows:
        aux_rows = '<tr><td colspan="2">No weighted aux features were available.</td></tr>'
    return (
        "<table>"
        "<thead><tr><th>Has score_label_aux model</th><th>Feature Count</th><th>Aux Feature Count</th><th>Aux Features</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{html_lib.escape(str(bool(feature_audit.get('has_score_label_aux_model'))))}</td>"
        f"<td>{int(feature_audit.get('feature_count', 0))}</td>"
        f"<td>{int(feature_audit.get('aux_feature_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in feature_audit.get('aux_features', ())) or 'none')}</td>"
        "</tr></tbody></table>"
        "<table><thead><tr><th>Aux Feature</th><th>Weight</th></tr></thead>"
        f"<tbody>{aux_rows}</tbody></table>"
    )


def _sample_bridge_html(sample_bridge: Mapping[str, Any]) -> str:
    support_rows = "".join(
        _sample_bridge_support_row(name, values)
        for name, values in (sample_bridge.get("support_means", {}) if isinstance(sample_bridge.get("support_means"), Mapping) else {}).items()
        if isinstance(values, Mapping)
    )
    if not support_rows:
        support_rows = '<tr><td colspan="7">No support means were available.</td></tr>'
    example_rows = "".join(_sample_bridge_example_row(row) for row in sample_bridge.get("examples", ()) if isinstance(row, Mapping))
    if not example_rows:
        example_rows = '<tr><td colspan="7">No removed detection examples were available.</td></tr>'
    return (
        f"<p>{html_lib.escape(str(sample_bridge.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Baseline</th><th>Target</th><th>Samples</th><th>Baseline Det</th><th>Target Det</th><th>Removed FP</th><th>Removed TP</th><th>FP Delta</th><th>TP Delta</th></tr></thead>"
        "<tbody><tr>"
        f"<td><code>{html_lib.escape(str(sample_bridge.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(sample_bridge.get('target_input', '')))}</code></td>"
        f"<td>{int(sample_bridge.get('compared_sample_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('baseline_detection_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('target_detection_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('removed_fp_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('removed_tp_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('fp_delta_count', 0))}</td>"
        f"<td>{int(sample_bridge.get('tp_delta_count', 0))}</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Group</th><th>Count</th><th>Score</th><th>Aux Support</th><th>Edge Support</th><th>Reliability</th><th>Aux Box IoU</th></tr></thead>"
        f"<tbody>{support_rows}</tbody></table>"
        "<table>"
        "<thead><tr><th>Sample</th><th>Status</th><th>Label</th><th>Score</th><th>Aux Support</th><th>Edge Support</th><th>Reliability</th></tr></thead>"
        f"<tbody>{example_rows}</tbody></table>"
    )


def _sample_bridge_support_row(name: str, values: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{int(values.get('count', 0))}</td>"
        f"<td>{_fmt(values.get('score_mean'))}</td>"
        f"<td>{_fmt(values.get('aux_support_mean'))}</td>"
        f"<td>{_fmt(values.get('edge_support_mean'))}</td>"
        f"<td>{_fmt(values.get('reliability_support_mean'))}</td>"
        f"<td>{_fmt(values.get('aux_box_iou_mean'))}</td>"
        "</tr>"
    )


def _sample_bridge_example_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('target_status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('label', '')))}</td>"
        f"<td>{_fmt(row.get('baseline_score'))}</td>"
        f"<td>{_fmt(row.get('aux_support'))}</td>"
        f"<td>{_fmt(row.get('edge_support'))}</td>"
        f"<td>{_fmt(row.get('reliability_support'))}</td>"
        "</tr>"
    )


def _report_link(item: Any, destination: Path) -> str:
    if not isinstance(item, Mapping):
        return ""
    html_path = item.get("html_path")
    if not html_path:
        return ""
    relative = os.path.relpath(str(html_path), start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


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
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
