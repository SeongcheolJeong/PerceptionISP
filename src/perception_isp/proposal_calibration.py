"""Train a lightweight proposal score calibrator from saved comparison reports."""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .eval_types import BoundingBox, Detection
from .metrics import aggregate_metric_rows, box_iou, evaluate_detections
from .threshold_sweep import parse_thresholds
from .types import json_ready


DEFAULT_FEATURE_SETS = ("score_aux", "score_label", "score_label_aux")
TRACKED_METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "det_count_mean",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate proposal score calibration on a saved comparison report.")
    parser.add_argument("report", help="Report directory or comparison_summary.json")
    parser.add_argument("--input", default="perception_fusion_rgb_aux", help="Detector input to calibrate.")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS), help="Comma-separated: score, score_aux, score_label, score_label_aux.")
    parser.add_argument("--thresholds", default="0.05:0.95:0.05", help="Comma values or start:stop:step range for calibrated score.")
    parser.add_argument("--baseline-input", default="human_rgb")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--split-strategy", default="hash", choices=["hash", "sequential"])
    parser.add_argument("--seed", default="perception_isp_calibration")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--recall-delta-floor", type=float, default=-0.001)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    summary = build_proposal_calibration(
        report,
        input_name=str(args.input),
        feature_sets=parse_csv(args.feature_sets),
        thresholds=parse_thresholds(args.thresholds),
        baseline_input=str(args.baseline_input),
        train_fraction=float(args.train_fraction),
        split_strategy=str(args.split_strategy),
        seed=str(args.seed),
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        l2=float(args.l2),
        recall_delta_floor=float(args.recall_delta_floor),
        source_report=report_path,
    )
    destination = Path(args.output_dir).expanduser() if args.output_dir else report_path.parent / "proposal_calibration"
    html_path = write_proposal_calibration(summary, destination)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "proposal_calibration_summary.json"),
                    "best": summary.get("best", {}),
                    "models": _compact_models(summary.get("models", [])),
                }
            ),
            indent=2,
        )
    )
    return 0


def parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not items:
        raise ValueError("at least one value is required")
    return items


def build_proposal_calibration(
    report: Mapping[str, Any],
    *,
    input_name: str,
    feature_sets: Sequence[str],
    thresholds: Sequence[float],
    baseline_input: str = "human_rgb",
    train_fraction: float = 0.70,
    split_strategy: str = "hash",
    seed: str = "perception_isp_calibration",
    epochs: int = 800,
    learning_rate: float = 0.05,
    l2: float = 1.0e-3,
    recall_delta_floor: float = -0.001,
    source_report: str | Path | None = None,
) -> Dict[str, Any]:
    samples = tuple(report.get("samples", ()))
    if not samples:
        raise ValueError("report contains no samples")
    label_agnostic = bool(report.get("run_config", {}).get("label_agnostic", True))
    train_indices, eval_indices = split_sample_indices(
        samples,
        train_fraction=float(train_fraction),
        strategy=str(split_strategy),
        seed=str(seed),
    )
    baseline_eval = _aggregate_original(report, input_name=baseline_input, indices=eval_indices, label_agnostic=label_agnostic)
    original_eval = _aggregate_original(report, input_name=input_name, indices=eval_indices, label_agnostic=label_agnostic)
    train_gt_labels = _ground_truth_labels(samples, train_indices)
    label_names = _detection_labels(samples, train_indices, input_name)

    rows: List[Dict[str, Any]] = []
    models: List[Dict[str, Any]] = []
    for feature_set in feature_sets:
        feature_names = _feature_names(feature_set, label_names)
        train_records = _records_for_indices(samples, train_indices, input_name=input_name, label_agnostic=label_agnostic)
        eval_records = _records_for_indices(samples, eval_indices, input_name=input_name, label_agnostic=label_agnostic)
        train_x = _feature_matrix(train_records, feature_names=feature_names, train_gt_labels=train_gt_labels)
        train_y = np.asarray([float(record["target"]) for record in train_records], dtype=np.float64)
        eval_x = _feature_matrix(eval_records, feature_names=feature_names, train_gt_labels=train_gt_labels)
        eval_y = np.asarray([float(record["target"]) for record in eval_records], dtype=np.float64)
        model = _fit_logistic(
            train_x,
            train_y,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            l2=float(l2),
        )
        model["feature_set"] = str(feature_set)
        model["feature_names"] = list(feature_names)
        model["train_loss"] = _logistic_loss(train_x, train_y, model)
        model["eval_loss"] = _logistic_loss(eval_x, eval_y, model) if len(eval_records) else None
        model["train_positive_count"] = int(np.sum(train_y))
        model["train_negative_count"] = int(len(train_y) - np.sum(train_y))
        model["eval_positive_count"] = int(np.sum(eval_y))
        model["eval_negative_count"] = int(len(eval_y) - np.sum(eval_y))
        models.append(model)
        for threshold in thresholds:
            metrics = _aggregate_calibrated(
                report,
                input_name=input_name,
                indices=eval_indices,
                model=model,
                threshold=float(threshold),
                label_agnostic=label_agnostic,
                train_gt_labels=train_gt_labels,
            )
            rows.append(
                {
                    "feature_set": str(feature_set),
                    "input": str(input_name),
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "delta_vs_baseline": _deltas(metrics, baseline_eval),
                    "delta_vs_original": _deltas(metrics, original_eval),
                }
            )

    return {
        "source_report": "" if source_report is None else str(source_report),
        "input": str(input_name),
        "baseline_input": str(baseline_input),
        "label_agnostic": bool(label_agnostic),
        "sample_count": int(len(samples)),
        "train_sample_count": int(len(train_indices)),
        "eval_sample_count": int(len(eval_indices)),
        "train_indices": [int(index) for index in train_indices],
        "eval_indices": [int(index) for index in eval_indices],
        "train_fraction": float(train_fraction),
        "split_strategy": str(split_strategy),
        "seed": str(seed),
        "thresholds": [float(value) for value in thresholds],
        "recall_delta_floor": float(recall_delta_floor),
        "baseline_metrics": baseline_eval,
        "original_input_metrics": original_eval,
        "original_delta_vs_baseline": _deltas(original_eval, baseline_eval),
        "train_gt_labels": list(train_gt_labels),
        "label_names": list(label_names),
        "models": models,
        "best": {
            "max_recall_delta": _best_by(rows, key="recall@0.50_mean"),
            "max_precision_with_recall_floor": _best_precision_with_floor(rows, recall_delta_floor=float(recall_delta_floor)),
            "min_fp_with_recall_floor": _best_min_fp_with_floor(rows, recall_delta_floor=float(recall_delta_floor)),
            "max_precision_vs_original_with_recall_floor": _best_precision_vs_original_with_floor(rows, recall_delta_floor=float(recall_delta_floor)),
        },
        "rows": rows,
    }


def write_proposal_calibration(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "proposal_calibration_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def split_sample_indices(
    samples: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float,
    strategy: str,
    seed: str,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    fraction = min(max(float(train_fraction), 0.05), 0.95)
    total = int(len(samples))
    if total < 2:
        return (0,), ()
    normalized = str(strategy or "hash").lower()
    if normalized == "sequential":
        train_count = min(max(int(round(total * fraction)), 1), total - 1)
        return tuple(range(train_count)), tuple(range(train_count, total))
    train: List[int] = []
    eval_: List[int] = []
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", index))
        digest = hashlib.sha256(f"{seed}:{sample_id}:{index}".encode("utf-8")).hexdigest()
        value = int(digest[:12], 16) / float(16**12)
        if value < fraction:
            train.append(index)
        else:
            eval_.append(index)
    if not train or not eval_:
        train_count = min(max(int(round(total * fraction)), 1), total - 1)
        return tuple(range(train_count)), tuple(range(train_count, total))
    return tuple(train), tuple(eval_)


def _summary_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "comparison_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"comparison summary not found: {path}")
    return path


def _aggregate_original(
    report: Mapping[str, Any],
    *,
    input_name: str,
    indices: Sequence[int],
    label_agnostic: bool,
) -> Dict[str, Any]:
    rows = []
    samples = tuple(report.get("samples", ()))
    for index in indices:
        sample = samples[int(index)]
        rows.append(
            evaluate_detections(
                _detections_for_sample(sample, input_name),
                _ground_truth_for_sample(sample),
                label_agnostic=label_agnostic,
            )
        )
    return aggregate_metric_rows(rows)


def _aggregate_calibrated(
    report: Mapping[str, Any],
    *,
    input_name: str,
    indices: Sequence[int],
    model: Mapping[str, Any],
    threshold: float,
    label_agnostic: bool,
    train_gt_labels: Sequence[str],
) -> Dict[str, Any]:
    rows = []
    samples = tuple(report.get("samples", ()))
    feature_names = tuple(str(name) for name in model.get("feature_names", ()))
    for index in indices:
        sample = samples[int(index)]
        kept = []
        for detection in _detections_for_sample(sample, input_name):
            score = _predict_detection_score(detection, sample, model, feature_names=feature_names, train_gt_labels=train_gt_labels)
            if score < float(threshold):
                continue
            metadata = dict(detection.metadata)
            metadata["proposal_calibration"] = {
                "source_score": float(detection.score),
                "calibrated_score": float(score),
                "feature_set": str(model.get("feature_set", "")),
            }
            kept.append(Detection(detection.box, score=float(score), metadata=metadata))
        rows.append(evaluate_detections(tuple(kept), _ground_truth_for_sample(sample), label_agnostic=label_agnostic))
    return aggregate_metric_rows(rows)


def _records_for_indices(
    samples: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    input_name: str,
    label_agnostic: bool,
) -> Tuple[Dict[str, Any], ...]:
    records: List[Dict[str, Any]] = []
    for index in indices:
        sample = samples[int(index)]
        detections = _detections_for_sample(sample, input_name)
        positives = _positive_detection_indices(detections, _ground_truth_for_sample(sample), label_agnostic=label_agnostic)
        for detection_index, detection in enumerate(detections):
            records.append(
                {
                    "sample_index": int(index),
                    "sample_id": str(sample.get("sample_id", index)),
                    "detection_index": int(detection_index),
                    "detection": detection,
                    "sample": sample,
                    "target": 1 if detection_index in positives else 0,
                }
            )
    return tuple(records)


def _positive_detection_indices(
    detections: Sequence[Detection],
    ground_truth: Sequence[BoundingBox],
    *,
    label_agnostic: bool,
) -> set[int]:
    ordered = sorted(enumerate(detections), key=lambda item: float(item[1].score), reverse=True)
    used_gt: set[int] = set()
    positives: set[int] = set()
    for detection_index, detection in ordered:
        best_iou = 0.0
        best_gt = -1
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in used_gt:
                continue
            if not label_agnostic and detection.box.label != gt.label:
                continue
            iou = box_iou(detection.box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_iou >= 0.50 and best_gt >= 0:
            positives.add(int(detection_index))
            used_gt.add(best_gt)
    return positives


def _feature_names(feature_set: str, label_names: Sequence[str]) -> Tuple[str, ...]:
    normalized = str(feature_set).lower()
    names = [
        "score",
        "rgb_score",
        "score_logit",
        "area_norm",
        "sqrt_area_norm",
        "width_norm",
        "height_norm",
        "aspect_log",
        "center_x_norm",
        "center_y_norm",
    ]
    if "aux" in normalized:
        names.extend(
            [
                "aux_support",
                "edge_support",
                "saturation_support",
                "reliability_support",
                "aux_box_iou",
                "score_delta_from_rgb",
                "score_x_aux",
                "score_x_reliability",
            ]
        )
    if "label" in normalized:
        names.append("label_seen_gt")
        names.extend(f"label={label}" for label in label_names)
    return tuple(names)


def _feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    train_gt_labels: Sequence[str],
) -> np.ndarray:
    matrix = np.zeros((len(records), len(feature_names)), dtype=np.float64)
    for row, record in enumerate(records):
        values = _feature_values(
            record["detection"],
            record["sample"],
            train_gt_labels=train_gt_labels,
        )
        for col, name in enumerate(feature_names):
            matrix[row, col] = float(values.get(str(name), 0.0))
    return matrix


def _feature_values(detection: Detection, sample: Mapping[str, Any], *, train_gt_labels: Sequence[str]) -> Dict[str, float]:
    box = detection.box
    width, height = _sample_size(sample)
    x1, y1, x2, y2 = box.xyxy
    box_w = max(float(x2 - x1), 0.0)
    box_h = max(float(y2 - y1), 0.0)
    area_norm = (box_w * box_h) / max(float(width * height), 1.0)
    fusion = detection.metadata.get("fusion", {}) if isinstance(detection.metadata, Mapping) else {}
    if not isinstance(fusion, Mapping):
        fusion = {}
    score = _clip_probability(float(detection.score))
    rgb_score = _clip_probability(float(fusion.get("rgb_score", score)))
    aux_support = float(fusion.get("aux_support", 0.0))
    reliability = float(fusion.get("reliability_support", 0.0))
    values = {
        "score": score,
        "rgb_score": rgb_score,
        "score_logit": _logit(score),
        "area_norm": area_norm,
        "sqrt_area_norm": math.sqrt(max(area_norm, 0.0)),
        "width_norm": box_w / max(float(width), 1.0),
        "height_norm": box_h / max(float(height), 1.0),
        "aspect_log": math.log(max(box_w, 1.0) / max(box_h, 1.0)),
        "center_x_norm": (0.5 * (x1 + x2)) / max(float(width), 1.0),
        "center_y_norm": (0.5 * (y1 + y2)) / max(float(height), 1.0),
        "aux_support": aux_support,
        "edge_support": float(fusion.get("edge_support", 0.0)),
        "saturation_support": float(fusion.get("saturation_support", 0.0)),
        "reliability_support": reliability,
        "aux_box_iou": float(fusion.get("aux_box_iou", 0.0)),
        "score_delta_from_rgb": score - rgb_score,
        "score_x_aux": score * aux_support,
        "score_x_reliability": score * reliability,
        "label_seen_gt": 1.0 if box.label in set(str(label) for label in train_gt_labels) else 0.0,
        f"label={box.label}": 1.0,
    }
    return {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in values.items()}


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> Dict[str, Any]:
    if x.shape[0] <= 0:
        raise ValueError("no proposal records available for calibration")
    positives = int(np.sum(y > 0.5))
    negatives = int(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("calibration requires both positive and negative proposal records")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    z = np.clip(np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    sample_weights = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    denom = max(float(np.sum(sample_weights)), 1.0e-12)
    lr = float(learning_rate)
    l2_value = float(l2)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = 0.0
    v_b = 0.0
    beta1, beta2 = 0.9, 0.999
    history = []
    for step in range(1, max(int(epochs), 1) + 1):
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0))
        pred = _sigmoid(_linear_response(z, weights, bias))
        error = (pred - y) * sample_weights
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            grad_w = (z.T @ error) / denom + l2_value * weights
        grad_b = float(np.sum(error) / denom)
        grad_w = np.clip(np.nan_to_num(grad_w, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        grad_b = float(np.clip(np.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0))
        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)
        weights -= lr * (m_w / (1.0 - beta1**step)) / (np.sqrt(v_w / (1.0 - beta2**step)) + 1.0e-8)
        bias -= lr * (m_b / (1.0 - beta1**step)) / (math.sqrt(v_b / (1.0 - beta2**step)) + 1.0e-8)
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0))
        if step == 1 or step == int(epochs) or step % max(int(epochs) // 5, 1) == 0:
            history.append({"epoch": int(step), "loss": _weighted_bce(z, y, weights, bias, sample_weights, l2_value)})
    return {
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epochs": int(max(int(epochs), 1)),
        "learning_rate": float(learning_rate),
        "l2": l2_value,
        "history": history,
    }


def _predict_detection_score(
    detection: Detection,
    sample: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    train_gt_labels: Sequence[str],
) -> float:
    record = {"detection": detection, "sample": sample}
    x = _feature_matrix((record,), feature_names=feature_names, train_gt_labels=train_gt_labels)
    return float(_predict_matrix(x, model)[0])


def _predict_matrix(x: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    weights = np.clip(np.nan_to_num(np.asarray(model.get("weights", ()), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    mean = np.asarray(model.get("mean", ()), dtype=np.float64)
    std = np.asarray(model.get("std", ()), dtype=np.float64)
    z = np.clip(np.nan_to_num((x - mean) / np.where(std < 1.0e-6, 1.0, std), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    return _sigmoid(_linear_response(z, weights, float(model.get("bias", 0.0))))


def _logistic_loss(x: np.ndarray, y: np.ndarray, model: Mapping[str, Any]) -> float | None:
    if x.shape[0] <= 0:
        return None
    positives = max(int(np.sum(y > 0.5)), 1)
    negatives = max(int(len(y) - int(np.sum(y > 0.5))), 1)
    sample_weights = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    weights = np.asarray(model.get("weights", ()), dtype=np.float64)
    mean = np.asarray(model.get("mean", ()), dtype=np.float64)
    std = np.asarray(model.get("std", ()), dtype=np.float64)
    z = np.clip(np.nan_to_num((x - mean) / np.where(std < 1.0e-6, 1.0, std), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    return _weighted_bce(z, y, weights, float(model.get("bias", 0.0)), sample_weights, float(model.get("l2", 0.0)))


def _weighted_bce(z: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, sample_weights: np.ndarray, l2: float) -> float:
    weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0))
    pred = np.clip(_sigmoid(_linear_response(z, weights, bias)), 1.0e-8, 1.0 - 1.0e-8)
    loss = -sample_weights * (y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred))
    return float(np.sum(loss) / max(float(np.sum(sample_weights)), 1.0e-12) + 0.5 * float(l2) * float(np.sum(weights * weights)))


def _linear_response(z: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    safe_z = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    safe_weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    safe_bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        response = safe_z @ safe_weights + safe_bias
    return np.clip(np.nan_to_num(response, nan=0.0, posinf=40.0, neginf=-40.0), -40.0, 40.0)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _logit(value: float) -> float:
    clipped = _clip_probability(value)
    return float(math.log(clipped / (1.0 - clipped)))


def _clip_probability(value: float) -> float:
    return float(min(max(float(value), 1.0e-6), 1.0 - 1.0e-6))


def _sample_size(sample: Mapping[str, Any]) -> Tuple[int, int]:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("width") and metadata.get("height"):
        return int(metadata["width"]), int(metadata["height"])
    max_x = 1.0
    max_y = 1.0
    for item in sample.get("ground_truth", ()):
        coords = item.get("xyxy", (0.0, 0.0, 1.0, 1.0))
        max_x = max(max_x, float(coords[2]))
        max_y = max(max_y, float(coords[3]))
    return int(math.ceil(max_x)), int(math.ceil(max_y))


def _detections_for_sample(sample: Mapping[str, Any], input_name: str) -> Tuple[Detection, ...]:
    for detector in sample.get("detectors", ()):
        if str(detector.get("input_name")) == str(input_name):
            return tuple(_detection_from_dict(item) for item in detector.get("detections", ()))
    return ()


def _ground_truth_for_sample(sample: Mapping[str, Any]) -> Tuple[BoundingBox, ...]:
    return tuple(_box_from_dict(item) for item in sample.get("ground_truth", ()))


def _box_from_dict(payload: Mapping[str, Any]) -> BoundingBox:
    return BoundingBox(tuple(float(value) for value in payload.get("xyxy", (0.0, 0.0, 0.0, 0.0))), label=str(payload.get("label", "object")))


def _detection_from_dict(payload: Mapping[str, Any]) -> Detection:
    return Detection(
        box=_box_from_dict(payload.get("box", {})),
        score=float(payload.get("score", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _ground_truth_labels(samples: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Tuple[str, ...]:
    labels = set()
    for index in indices:
        for item in samples[int(index)].get("ground_truth", ()):
            labels.add(str(item.get("label", "object")))
    return tuple(sorted(labels))


def _detection_labels(samples: Sequence[Mapping[str, Any]], indices: Sequence[int], input_name: str) -> Tuple[str, ...]:
    labels = set()
    for index in indices:
        for detection in _detections_for_sample(samples[int(index)], input_name):
            labels.add(str(detection.box.label))
    return tuple(sorted(labels))


def _deltas(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in TRACKED_METRICS
    }


def _best_by(rows: Sequence[Mapping[str, Any]], *, key: str) -> Dict[str, Any]:
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("delta_vs_baseline", {}).get(key, -1.0e9)))
    return _compact_row(best)


def _best_precision_with_floor(rows: Sequence[Mapping[str, Any]], *, recall_delta_floor: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if float(row.get("delta_vs_baseline", {}).get("recall@0.50_mean", -1.0e9)) >= float(recall_delta_floor)
    ]
    if not candidates:
        return {}
    return _compact_row(max(candidates, key=lambda row: float(row.get("metrics", {}).get("precision@0.50_mean", -1.0e9))))


def _best_min_fp_with_floor(rows: Sequence[Mapping[str, Any]], *, recall_delta_floor: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if float(row.get("delta_vs_baseline", {}).get("recall@0.50_mean", -1.0e9)) >= float(recall_delta_floor)
    ]
    if not candidates:
        return {}
    return _compact_row(min(candidates, key=lambda row: float(row.get("metrics", {}).get("fp@0.50_mean", 1.0e9))))


def _best_precision_vs_original_with_floor(rows: Sequence[Mapping[str, Any]], *, recall_delta_floor: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if float(row.get("delta_vs_original", {}).get("recall@0.50_mean", -1.0e9)) >= float(recall_delta_floor)
    ]
    if not candidates:
        return {}
    return _compact_row(max(candidates, key=lambda row: float(row.get("metrics", {}).get("precision@0.50_mean", -1.0e9))))


def _compact_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "feature_set": row.get("feature_set"),
        "threshold": float(row.get("threshold", 0.0)),
        "metrics": dict(row.get("metrics", {})),
        "delta_vs_baseline": dict(row.get("delta_vs_baseline", {})),
        "delta_vs_original": dict(row.get("delta_vs_original", {})),
    }


def _compact_models(models: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for model in models:
        compact.append(
            {
                "feature_set": model.get("feature_set"),
                "feature_count": len(model.get("feature_names", ())),
                "train_loss": model.get("train_loss"),
                "eval_loss": model.get("eval_loss"),
                "train_positive_count": model.get("train_positive_count"),
                "train_negative_count": model.get("train_negative_count"),
                "eval_positive_count": model.get("eval_positive_count"),
                "eval_negative_count": model.get("eval_negative_count"),
            }
        )
    return compact


def _render_html(summary: Mapping[str, Any]) -> str:
    baseline = summary.get("baseline_metrics", {})
    original = summary.get("original_input_metrics", {})
    model_rows = []
    for model in summary.get("models", ()):
        model_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(model.get('feature_set', '')))}</td>"
            f"<td>{len(model.get('feature_names', ()))}</td>"
            f"<td>{int(model.get('train_positive_count', 0))}/{int(model.get('train_negative_count', 0))}</td>"
            f"<td>{_fmt(model.get('train_loss'))}</td>"
            f"<td>{_fmt(model.get('eval_loss'))}</td>"
            "</tr>"
        )
    best_items = []
    for name, item in summary.get("best", {}).items():
        if not item:
            continue
        metrics = item.get("metrics", {})
        delta = item.get("delta_vs_baseline", {})
        best_items.append(
            f"<li><strong>{html_lib.escape(str(name))}</strong>: "
            f"{html_lib.escape(str(item.get('feature_set')))} @ {float(item.get('threshold', 0.0)):.3f}, "
            f"P50={_fmt(metrics.get('precision@0.50_mean'))}, R50={_fmt(metrics.get('recall@0.50_mean'))}, "
            f"delta R50={_fmt_delta(delta.get('recall@0.50_mean'))}, delta FP={_fmt_delta(delta.get('fp@0.50_mean'))}</li>"
        )
    result_rows = []
    for row in summary.get("rows", ()):
        metrics = row.get("metrics", {})
        delta = row.get("delta_vs_baseline", {})
        original_delta = row.get("delta_vs_original", {})
        result_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('feature_set', '')))}</td>"
            f"<td>{float(row.get('threshold', 0.0)):.3f}</td>"
            f"<td>{_fmt(metrics.get('precision@0.50_mean'))}</td>"
            f"<td>{_fmt(metrics.get('recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('recall@0.50_mean'))}\">{_fmt_delta(delta.get('recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(original_delta.get('recall@0.50_mean'))}\">{_fmt_delta(original_delta.get('recall@0.50_mean'))}</td>"
            f"<td>{_fmt(metrics.get('small_recall@0.50_mean'))}</td>"
            f"<td>{_fmt(metrics.get('fp@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(-float(delta.get('fp@0.50_mean', 0.0)))}\">{_fmt_delta(delta.get('fp@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(-float(original_delta.get('fp@0.50_mean', 0.0)))}\">{_fmt_delta(original_delta.get('fp@0.50_mean'))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Proposal Calibration</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 9px; text-align: left; font-size: 14px; }}
    th {{ background: #e8f3f1; position: sticky; top: 0; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pos {{ color: #047857; font-weight: 650; }}
    .neg {{ color: #b91c1c; font-weight: 650; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Proposal Calibration</h1>
  <div class=\"note\">This report trains a lightweight score calibrator on saved detector proposals. It does not rerun CameraE2E or YOLO.</div>
  <p><strong>Source:</strong> <code>{html_lib.escape(str(summary.get('source_report', '')))}</code></p>
  <p><strong>Input:</strong> <code>{html_lib.escape(str(summary.get('input', '')))}</code>, train/eval samples: {int(summary.get('train_sample_count', 0))}/{int(summary.get('eval_sample_count', 0))}, label agnostic: {bool(summary.get('label_agnostic', False))}.</p>
  <p><strong>Human eval baseline:</strong> P50={_fmt(baseline.get('precision@0.50_mean'))}, R50={_fmt(baseline.get('recall@0.50_mean'))}, FP50={_fmt(baseline.get('fp@0.50_mean'))}.</p>
  <p><strong>Original input eval:</strong> P50={_fmt(original.get('precision@0.50_mean'))}, R50={_fmt(original.get('recall@0.50_mean'))}, FP50={_fmt(original.get('fp@0.50_mean'))}.</p>
  <h2>Models</h2>
  <table>
    <thead><tr><th>Feature Set</th><th>Features</th><th>Train Pos/Neg</th><th>Train Loss</th><th>Eval Loss</th></tr></thead>
    <tbody>{''.join(model_rows)}</tbody>
  </table>
  <h2>Best</h2>
  <ul>{''.join(best_items)}</ul>
  <h2>Threshold Rows</h2>
  <table>
    <thead><tr><th>Feature Set</th><th>Threshold</th><th>P50</th><th>R50</th><th>Delta R50 vs Human</th><th>Delta R50 vs Original</th><th>Small R50</th><th>FP50</th><th>Delta FP vs Human</th><th>Delta FP vs Original</th></tr></thead>
    <tbody>{''.join(result_rows)}</tbody>
  </table>
  <p>Summary JSON: <code>proposal_calibration_summary.json</code></p>
</body>
</html>
"""


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


def _delta_class(value: Any) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if numeric > 0.0:
        return "pos"
    if numeric < 0.0:
        return "neg"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
