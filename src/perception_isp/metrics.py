"""Small object-detection metrics for A/B ISP comparisons."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .eval_types import BoundingBox, Detection


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = a.area + b.area - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def match_detections(
    detections: Sequence[Detection],
    ground_truth: Sequence[BoundingBox],
    *,
    iou_threshold: float = 0.5,
    label_agnostic: bool = False,
) -> Dict[str, Any]:
    """Greedy one-to-one detection matching."""

    sorted_detections = sorted(detections, key=lambda item: float(item.score), reverse=True)
    used_gt: set[int] = set()
    matches: List[Dict[str, Any]] = []
    false_positive = 0
    for det_index, detection in enumerate(sorted_detections):
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
        if best_iou >= float(iou_threshold) and best_gt >= 0:
            used_gt.add(best_gt)
            matches.append(
                {
                    "det_index": det_index,
                    "gt_index": best_gt,
                    "iou": float(best_iou),
                    "score": float(detection.score),
                    "label": detection.box.label,
                }
            )
        else:
            false_positive += 1
    true_positive = len(matches)
    false_negative = max(len(ground_truth) - true_positive, 0)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(len(ground_truth), 1)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "matches": matches,
    }


def evaluate_detections(
    detections: Sequence[Detection],
    ground_truth: Sequence[BoundingBox],
    *,
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    small_area_threshold: float = 32.0 * 32.0,
    label_agnostic: bool = False,
) -> Dict[str, Any]:
    """Return compact metrics across IoU thresholds and small objects."""

    metrics: Dict[str, Any] = {
        "gt_count": int(len(ground_truth)),
        "det_count": int(len(detections)),
        "iou_thresholds": [float(v) for v in iou_thresholds],
    }
    recalls = []
    precisions = []
    for threshold in iou_thresholds:
        matched = match_detections(
            detections,
            ground_truth,
            iou_threshold=float(threshold),
            label_agnostic=label_agnostic,
        )
        metrics[f"precision@{threshold:.2f}"] = float(matched["precision"])
        metrics[f"recall@{threshold:.2f}"] = float(matched["recall"])
        metrics[f"tp@{threshold:.2f}"] = int(matched["true_positive"])
        metrics[f"fp@{threshold:.2f}"] = int(matched["false_positive"])
        metrics[f"fn@{threshold:.2f}"] = int(matched["false_negative"])
        precisions.append(float(matched["precision"]))
        recalls.append(float(matched["recall"]))
    metrics["mean_precision"] = float(np.mean(precisions)) if precisions else 0.0
    metrics["mean_recall"] = float(np.mean(recalls)) if recalls else 0.0

    small_gt = tuple(gt for gt in ground_truth if gt.area <= float(small_area_threshold))
    small_match = match_detections(
        detections,
        small_gt,
        iou_threshold=0.5,
        label_agnostic=label_agnostic,
    )
    metrics["small_gt_count"] = int(len(small_gt))
    metrics["small_recall@0.50"] = float(small_match["recall"]) if small_gt else 0.0
    return metrics


def aggregate_metric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    numeric_keys = sorted(
        key
        for key in rows[0]
        if isinstance(rows[0].get(key), (int, float)) and not isinstance(rows[0].get(key), bool)
    )
    aggregate: Dict[str, Any] = {"sample_count": int(len(rows))}
    for key in numeric_keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_min"] = float(np.min(values))
        aggregate[f"{key}_max"] = float(np.max(values))
    return aggregate
