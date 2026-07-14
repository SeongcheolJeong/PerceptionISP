"""Calibrate YOLO proposal scores with PerceptionISP aux-map box evidence."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .types import json_ready
from .yolo_hard_object_eval import (
    DEFAULT_CONF_THRESHOLDS,
    DEFAULT_IOU_THRESHOLDS,
    BoxRecord,
    _average_precision,
    _box_in_filter,
    _fixed_conf_metrics,
    _image_paths,
    _load_data_config,
    _load_gt,
    _load_image_array,
    _predict,
)


DEFAULT_FEATURE_SET = (
    "bias",
    "score_logit",
    "score",
    "box_area",
    "box_aspect_log",
    "edge_strength_mean",
    "edge_strength_p90",
    "edge_strength_max",
    "edge_strength_border_mean",
    "edge_strength_border_ratio",
    "edge_evidence_mean",
    "edge_evidence_p90",
    "edge_evidence_border_mean",
    "psf_edge_mean",
    "psf_edge_p90",
    "psf_edge_border_mean",
)

THRESHOLD_SWEEP = tuple(float(value) for value in np.linspace(0.001, 0.999, 500))
RECALL_FLOOR_TOLERANCE = 0.005


@dataclass(frozen=True)
class FeatureRecord:
    stem: str
    cls: int
    xyxy: tuple[float, float, float, float]
    original_conf: float
    target: int
    ignored: bool
    features: Mapping[str, float]


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate aux-map proposal calibration for YOLO detections.")
    parser.add_argument("--rgb-model", required=True, help="RGB detector checkpoint.")
    parser.add_argument("--rgb-data", required=True, help="RGB-only YOLO data.yaml.")
    parser.add_argument("--aux-data", required=True, help="RGB+Aux YOLO data.yaml with matching stems.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--predict-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--small-area", type=float, default=0.02)
    parser.add_argument("--thin-aspect", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--keep-threshold-sweep", action="store_true")
    parser.add_argument(
        "--train-filter",
        default="all",
        choices=["all", "small", "thin", "small_or_thin"],
        help="Target GT slice used to assign positives. Non-target matched GT are ignored.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    summary = run_aux_box_calibration(
        rgb_model=Path(args.rgb_model),
        rgb_data=Path(args.rgb_data),
        aux_data=Path(args.aux_data),
        train_split=str(args.train_split),
        eval_split=str(args.eval_split),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        predict_conf=float(args.predict_conf),
        predict_iou=float(args.predict_iou),
        max_det=int(args.max_det),
        match_iou=float(args.match_iou),
        small_area=float(args.small_area),
        thin_aspect=float(args.thin_aspect),
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        l2=float(args.l2),
        positive_weight=float(args.positive_weight),
        negative_weight=float(args.negative_weight),
        train_filter=str(args.train_filter),
        keep_threshold_sweep=bool(args.keep_threshold_sweep),
        out=Path(args.out),
    )
    print(json.dumps(json_ready(_compact_summary(summary)), indent=2))
    return 0


def run_aux_box_calibration(
    *,
    rgb_model: Path,
    rgb_data: Path,
    aux_data: Path,
    train_split: str = "train",
    eval_split: str = "val",
    imgsz: int = 512,
    batch: int = 8,
    device: str = "mps",
    workers: int = 0,
    predict_conf: float = 0.001,
    predict_iou: float = 0.70,
    max_det: int = 300,
    match_iou: float = 0.50,
    small_area: float = 0.02,
    thin_aspect: float = 3.0,
    epochs: int = 800,
    learning_rate: float = 0.05,
    l2: float = 1.0e-3,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    train_filter: str = "all",
    keep_threshold_sweep: bool = False,
    out: Path,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is required for YOLO aux box calibration") from exc

    out.mkdir(parents=True, exist_ok=True)
    rgb_root, rgb_config = _load_data_config(rgb_data)
    aux_root, aux_config = _load_data_config(aux_data)

    model = YOLO(str(rgb_model))
    train_image_paths = _image_paths(root=rgb_root, split=str(rgb_config.get(train_split, f"images/{train_split}")))
    eval_image_paths = _image_paths(root=rgb_root, split=str(rgb_config.get(eval_split, f"images/{eval_split}")))
    train_gt = _load_gt(
        root=rgb_root,
        split=str(rgb_config.get(train_split, f"images/{train_split}")),
        small_area=small_area,
        thin_aspect=thin_aspect,
    )
    eval_gt = _load_gt(
        root=rgb_root,
        split=str(rgb_config.get(eval_split, f"images/{eval_split}")),
        small_area=small_area,
        thin_aspect=thin_aspect,
    )
    train_predictions = _predict(
        model,
        image_paths=train_image_paths,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        predict_conf=predict_conf,
        predict_iou=predict_iou,
        max_det=max_det,
    )
    eval_predictions = _predict(
        model,
        image_paths=eval_image_paths,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        predict_conf=predict_conf,
        predict_iou=predict_iou,
        max_det=max_det,
    )

    train_aux_dir = aux_root / str(aux_config.get(train_split, f"images/{train_split}"))
    eval_aux_dir = aux_root / str(aux_config.get(eval_split, f"images/{eval_split}"))
    train_records = build_feature_records(
        predictions=train_predictions,
        gt_by_stem=train_gt,
        aux_image_dir=train_aux_dir,
        match_iou=match_iou,
        small_area=small_area,
        thin_aspect=thin_aspect,
        target_filter=train_filter,
    )
    eval_records = build_feature_records(
        predictions=eval_predictions,
        gt_by_stem=eval_gt,
        aux_image_dir=eval_aux_dir,
        match_iou=match_iou,
        small_area=small_area,
        thin_aspect=thin_aspect,
        target_filter="all",
    )
    train_fit_records = [record for record in train_records if not record.ignored]
    if not train_fit_records:
        raise ValueError("no non-ignored training proposal records were produced")
    feature_names = tuple(name for name in DEFAULT_FEATURE_SET if name != "bias")
    x_train, y_train, weights, normalizer = _matrix(
        train_fit_records,
        feature_names=feature_names,
        fit_normalizer=True,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
    )
    model_payload = _fit_logistic(
        x_train,
        y_train,
        weights=weights,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    model_payload.update(
        {
            "feature_names": list(feature_names),
            "normalizer": normalizer,
            "train_filter": str(train_filter),
            "match_iou": float(match_iou),
            "positive_weight": float(positive_weight),
            "negative_weight": float(negative_weight),
        }
    )
    calibrated_eval_predictions = _score_predictions(eval_records, model_payload, feature_names=feature_names)
    original_eval_predictions = _records_to_predictions(eval_records, score_key="original")

    summary = {
        "run_config": {
            "rgb_model": str(rgb_model),
            "rgb_data": str(rgb_data),
            "aux_data": str(aux_data),
            "train_split": str(train_split),
            "eval_split": str(eval_split),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "device": str(device),
            "workers": int(workers),
            "predict_conf": float(predict_conf),
            "predict_iou": float(predict_iou),
            "max_det": int(max_det),
            "match_iou": float(match_iou),
            "small_area": float(small_area),
            "thin_aspect": float(thin_aspect),
            "train_filter": str(train_filter),
            "epochs": int(epochs),
            "lr": float(learning_rate),
            "l2": float(l2),
            "out": str(out),
        },
        "model": {
            **model_payload,
            "train_loss": _logistic_loss(x_train, y_train, weights, model_payload),
            "train_positive_count": int(np.sum(y_train)),
            "train_negative_count": int(len(y_train) - np.sum(y_train)),
            "train_record_count": int(len(train_fit_records)),
            "train_ignored_count": int(sum(1 for record in train_records if record.ignored)),
            "eval_record_count": int(len(eval_records)),
        },
        "runs": {
            "rgb_original": _evaluate_prediction_set(
                gt_by_stem=eval_gt,
                predictions=original_eval_predictions,
                small_area=small_area,
                thin_aspect=thin_aspect,
            ),
            "rgb_aux_calibrated": _evaluate_prediction_set(
                gt_by_stem=eval_gt,
                predictions=calibrated_eval_predictions,
                small_area=small_area,
                thin_aspect=thin_aspect,
            ),
        },
    }
    summary["delta"] = _delta_summary(summary["runs"]["rgb_original"], summary["runs"]["rgb_aux_calibrated"])
    if not keep_threshold_sweep:
        _strip_threshold_sweeps(summary)

    (out / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (out / "model.json").write_text(json.dumps(json_ready(model_payload), indent=2) + "\n")
    (out / "index.html").write_text(_render_html(summary))
    return summary


def _strip_threshold_sweeps(summary: Mapping[str, Any]) -> None:
    runs = summary.get("runs", {})
    if not isinstance(runs, dict):
        return
    for run in runs.values():
        if not isinstance(run, dict):
            continue
        filters = run.get("filters", {})
        if not isinstance(filters, dict):
            continue
        for metrics in filters.values():
            if isinstance(metrics, dict):
                metrics.pop("threshold_sweep", None)


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    compact_runs: Dict[str, Any] = {}
    for run_name, run in summary.get("runs", {}).items():
        compact_runs[run_name] = {}
        for filter_name, metrics in run.get("filters", {}).items():
            compact_runs[run_name][filter_name] = {
                "mAP50": metrics.get("mAP50"),
                "mAP50_95": metrics.get("mAP50_95"),
                "fixed_conf": metrics.get("fixed_conf", {}),
            }
    return {
        "out": summary.get("run_config", {}).get("out"),
        "run_config": summary.get("run_config", {}),
        "model": {
            key: summary.get("model", {}).get(key)
            for key in ("train_filter", "train_loss", "train_positive_count", "train_negative_count", "train_record_count", "train_ignored_count", "eval_record_count")
        },
        "runs": compact_runs,
        "delta": summary.get("delta", {}),
    }


def build_feature_records(
    *,
    predictions: Mapping[str, Sequence[BoxRecord]],
    gt_by_stem: Mapping[str, Sequence[BoxRecord]],
    aux_image_dir: Path,
    match_iou: float,
    small_area: float,
    thin_aspect: float,
    target_filter: str,
) -> List[FeatureRecord]:
    aux_cache: Dict[str, np.ndarray] = {}
    records: List[FeatureRecord] = []
    target_by_stem: Dict[str, List[BoxRecord]] = {}
    ignore_by_stem: Dict[str, List[BoxRecord]] = {}
    for stem, gt_rows in gt_by_stem.items():
        target_rows = [row for row in gt_rows if _box_in_filter(row, target_filter, small_area=small_area, thin_aspect=thin_aspect)]
        target_by_stem[stem] = target_rows
        ignore_by_stem[stem] = [row for row in gt_rows if row not in target_rows]

    matched: Dict[str, set[int]] = {stem: set() for stem in gt_by_stem}
    for pred in sorted(_all_predictions(predictions), key=lambda row: row.conf, reverse=True):
        stem = pred.stem
        target_rows = target_by_stem.get(stem, ())
        best_iou, best_index = _best_match(pred, target_rows, matched.get(stem, set()))
        target = 0
        ignored = False
        if best_index >= 0 and best_iou >= match_iou:
            matched.setdefault(stem, set()).add(best_index)
            target = 1
        else:
            ignore_iou, _ = _best_match(pred, ignore_by_stem.get(stem, ()), set())
            ignored = bool(ignore_iou >= match_iou)
        aux_image = aux_cache.get(stem)
        if aux_image is None:
            aux_path = aux_image_dir / f"{stem}.npy"
            if not aux_path.exists():
                raise FileNotFoundError(f"missing aux tensor for {stem}: {aux_path}")
            aux_image = _load_image_array(aux_path)
            aux_cache[stem] = aux_image
        records.append(
            FeatureRecord(
                stem=stem,
                cls=int(pred.cls),
                xyxy=tuple(float(v) for v in pred.xyxy),
                original_conf=float(pred.conf),
                target=int(target),
                ignored=bool(ignored),
                features=_box_features(pred, aux_image),
            )
        )
    return records


def _box_features(pred: BoxRecord, image: np.ndarray) -> Dict[str, float]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 6:
        raise ValueError(f"expected HWC aux tensor with at least 6 channels, got {array.shape}")
    h, w = array.shape[:2]
    x1, y1, x2, y2 = pred.xyxy
    ix1 = int(np.clip(math.floor(x1 * w), 0, w - 1))
    iy1 = int(np.clip(math.floor(y1 * h), 0, h - 1))
    ix2 = int(np.clip(math.ceil(x2 * w), ix1 + 1, w))
    iy2 = int(np.clip(math.ceil(y2 * h), iy1 + 1, h))
    crop = array[iy1:iy2, ix1:ix2, 3:6] / 255.0
    border = _border_pixels(crop)
    area = max(0.0, (x2 - x1) * (y2 - y1))
    aspect = max((x2 - x1) / max(y2 - y1, 1e-6), (y2 - y1) / max(x2 - x1, 1e-6))
    features: Dict[str, float] = {
        "score": float(np.clip(pred.conf, 1e-6, 1.0 - 1e-6)),
        "score_logit": float(math.log(np.clip(pred.conf, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(pred.conf, 1e-6, 1.0 - 1e-6)))),
        "box_area": float(area),
        "box_aspect_log": float(math.log(max(aspect, 1.0))),
    }
    names = ("edge_strength", "edge_evidence", "psf_edge")
    for idx, name in enumerate(names):
        channel = crop[..., idx]
        border_channel = border[..., idx] if border.size else channel.reshape(-1)
        mean = float(np.mean(channel)) if channel.size else 0.0
        border_mean = float(np.mean(border_channel)) if border_channel.size else mean
        features[f"{name}_mean"] = mean
        features[f"{name}_p90"] = float(np.percentile(channel, 90.0)) if channel.size else 0.0
        features[f"{name}_max"] = float(np.max(channel)) if channel.size else 0.0
        features[f"{name}_border_mean"] = border_mean
        features[f"{name}_border_ratio"] = float(border_mean / max(mean, 1.0e-6))
    return features


def _border_pixels(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop.reshape(0, crop.shape[-1] if crop.ndim == 3 else 0)
    if crop.shape[0] <= 2 or crop.shape[1] <= 2:
        return crop.reshape(-1, crop.shape[-1])
    top = crop[0, :, :]
    bottom = crop[-1, :, :]
    left = crop[1:-1, 0, :]
    right = crop[1:-1, -1, :]
    return np.concatenate([top, bottom, left, right], axis=0)


def _matrix(
    records: Sequence[FeatureRecord],
    *,
    feature_names: Sequence[str],
    fit_normalizer: bool,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    normalizer: Mapping[str, Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[float]]]:
    raw = np.asarray([[float(record.features.get(name, 0.0)) for name in feature_names] for record in records], dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray([float(record.target) for record in records], dtype=np.float64)
    weights = np.where(y > 0.5, float(positive_weight), float(negative_weight)).astype(np.float64)
    if fit_normalizer:
        mean = np.mean(raw, axis=0)
        std = np.std(raw, axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        norm = {"mean": mean.tolist(), "std": std.tolist()}
    else:
        if normalizer is None:
            raise ValueError("normalizer is required when fit_normalizer is false")
        mean = np.asarray(normalizer["mean"], dtype=np.float64)
        std = np.asarray(normalizer["std"], dtype=np.float64)
        norm = {"mean": mean.tolist(), "std": std.tolist()}
    x = (raw - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -8.0, 8.0)
    x = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x, y, weights, norm


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> Dict[str, Any]:
    beta = np.zeros(x.shape[1], dtype=np.float64)
    weight_sum = max(float(np.sum(weights)), 1.0)
    for _ in range(int(epochs)):
        with np.errstate(all="ignore"):
            logits = x @ beta
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        probs = _sigmoid(logits)
        with np.errstate(all="ignore"):
            grad = (x.T @ ((probs - y) * weights)) / weight_sum
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad = np.clip(grad, -10.0, 10.0)
        reg = np.r_[0.0, beta[1:]] * float(l2)
        beta -= float(learning_rate) * (grad + reg)
    return {"intercept": float(beta[0]), "weights": beta[1:].tolist()}


def _logistic_loss(x: np.ndarray, y: np.ndarray, weights: np.ndarray, model: Mapping[str, Any]) -> float:
    beta = np.asarray([float(model["intercept"]), *[float(value) for value in model["weights"]]], dtype=np.float64)
    with np.errstate(all="ignore"):
        logits = x @ beta
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    probs = np.clip(_sigmoid(logits), 1.0e-9, 1.0 - 1.0e-9)
    loss = -(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    return float(np.sum(loss * weights) / max(float(np.sum(weights)), 1.0))


def _score_predictions(
    records: Sequence[FeatureRecord],
    model: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
) -> Dict[str, List[BoxRecord]]:
    x, _, _, _ = _matrix(
        records,
        feature_names=feature_names,
        fit_normalizer=False,
        normalizer=model["normalizer"],
    )
    beta = np.asarray([float(model["intercept"]), *[float(value) for value in model["weights"]]], dtype=np.float64)
    with np.errstate(all="ignore"):
        logits = x @ beta
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    scores = _sigmoid(logits)
    predictions: Dict[str, List[BoxRecord]] = {}
    for record, score in zip(records, scores):
        predictions.setdefault(record.stem, []).append(
            BoxRecord(cls=record.cls, xyxy=record.xyxy, conf=float(score), stem=record.stem)
        )
    return predictions


def _records_to_predictions(records: Sequence[FeatureRecord], *, score_key: str) -> Dict[str, List[BoxRecord]]:
    predictions: Dict[str, List[BoxRecord]] = {}
    for record in records:
        score = record.original_conf if score_key == "original" else float(record.features[score_key])
        predictions.setdefault(record.stem, []).append(
            BoxRecord(cls=record.cls, xyxy=record.xyxy, conf=float(score), stem=record.stem)
        )
    return predictions


def _evaluate_prediction_set(
    *,
    gt_by_stem: Mapping[str, Sequence[BoxRecord]],
    predictions: Mapping[str, Sequence[BoxRecord]],
    small_area: float,
    thin_aspect: float,
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for filter_name in ("all", "small", "thin", "small_or_thin"):
        target_by_stem: Dict[str, List[BoxRecord]] = {}
        ignore_by_stem: Dict[str, List[BoxRecord]] = {}
        for stem, gt_rows in gt_by_stem.items():
            targets = [row for row in gt_rows if _box_in_filter(row, filter_name, small_area=small_area, thin_aspect=thin_aspect)]
            target_by_stem[stem] = targets
            ignore_by_stem[stem] = [row for row in gt_rows if row not in targets]
        aps = []
        ap_by_iou: Dict[str, float] = {}
        for iou_threshold in DEFAULT_IOU_THRESHOLDS:
            ap = _average_precision(
                target_by_stem=target_by_stem,
                ignore_by_stem=ignore_by_stem,
                predictions=predictions,
                iou_threshold=float(iou_threshold),
            )
            ap_by_iou[f"{iou_threshold:.2f}"] = ap
            aps.append(ap)
        target_count = sum(len(rows) for rows in target_by_stem.values())
        filters[filter_name] = {
            "target_gt_count": int(target_count),
            "image_count_with_target": int(sum(1 for rows in target_by_stem.values() if rows)),
            "ignored_gt_count": int(sum(len(rows) for rows in ignore_by_stem.values())),
            "mAP50": float(ap_by_iou["0.50"]) if target_count else None,
            "mAP50_95": float(np.nanmean(aps)) if target_count else None,
            "ap_by_iou": ap_by_iou,
            "fixed_conf": {
                f"conf_{threshold:.2f}_iou_0.50": _fixed_conf_metrics(
                    target_by_stem=target_by_stem,
                    ignore_by_stem=ignore_by_stem,
                    predictions=predictions,
                    conf_threshold=float(threshold),
                    iou_threshold=0.50,
                )
                for threshold in DEFAULT_CONF_THRESHOLDS
            },
            "threshold_sweep": _threshold_sweep(
                target_by_stem=target_by_stem,
                ignore_by_stem=ignore_by_stem,
                predictions=predictions,
                thresholds=THRESHOLD_SWEEP,
                iou_threshold=0.50,
            ),
        }
    return {"filters": filters}


def _delta_summary(original: Mapping[str, Any], calibrated: Mapping[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for filter_name, base in original.get("filters", {}).items():
        other = calibrated.get("filters", {}).get(filter_name, {})
        item: Dict[str, Any] = {}
        for key in ("mAP50", "mAP50_95"):
            if base.get(key) is not None and other.get(key) is not None:
                item[f"delta_{key}"] = float(other[key]) - float(base[key])
        for threshold_key, base_fixed in base.get("fixed_conf", {}).items():
            other_fixed = other.get("fixed_conf", {}).get(threshold_key, {})
            item[threshold_key] = {
                "delta_precision": _optional_delta(other_fixed.get("precision"), base_fixed.get("precision")),
                "delta_recall": _optional_delta(other_fixed.get("recall"), base_fixed.get("recall")),
                "delta_fp": int(other_fixed.get("fp", 0)) - int(base_fixed.get("fp", 0)),
                "delta_tp": int(other_fixed.get("tp", 0)) - int(base_fixed.get("tp", 0)),
            }
            item[f"{threshold_key}_matched_recall"] = _best_sweep_point_at_recall(
                other.get("threshold_sweep", ()),
                recall_floor=(float(base_fixed.get("recall") or 0.0) - RECALL_FLOOR_TOLERANCE),
                baseline_fixed=base_fixed,
            )
        delta[filter_name] = item
    return delta


def _threshold_sweep(
    *,
    target_by_stem: Mapping[str, Sequence[BoxRecord]],
    ignore_by_stem: Mapping[str, Sequence[BoxRecord]],
    predictions: Mapping[str, Sequence[BoxRecord]],
    thresholds: Sequence[float],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    return [
        {
            "threshold": float(threshold),
            **_fixed_conf_metrics(
                target_by_stem=target_by_stem,
                ignore_by_stem=ignore_by_stem,
                predictions=predictions,
                conf_threshold=float(threshold),
                iou_threshold=float(iou_threshold),
            ),
        }
        for threshold in thresholds
    ]


def _best_sweep_point_at_recall(
    sweep: Sequence[Mapping[str, Any]],
    *,
    recall_floor: float,
    baseline_fixed: Mapping[str, Any],
) -> Dict[str, Any] | None:
    candidates = [
        point
        for point in sweep
        if point.get("precision") is not None
        and point.get("recall") is not None
        and float(point["recall"]) >= float(recall_floor)
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda point: (
            int(point.get("fp", 0)),
            -float(point.get("precision") or 0.0),
            -float(point.get("recall") or 0.0),
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "tp": int(best.get("tp", 0)),
        "fp": int(best.get("fp", 0)),
        "fn": int(best.get("fn", 0)),
        "delta_precision": _optional_delta(best.get("precision"), baseline_fixed.get("precision")),
        "delta_recall": _optional_delta(best.get("recall"), baseline_fixed.get("recall")),
        "delta_fp": int(best.get("fp", 0)) - int(baseline_fixed.get("fp", 0)),
        "delta_tp": int(best.get("tp", 0)) - int(baseline_fixed.get("tp", 0)),
        "recall_floor": float(recall_floor),
    }


def _optional_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def _all_predictions(predictions: Mapping[str, Sequence[BoxRecord]]) -> Iterable[BoxRecord]:
    for rows in predictions.values():
        yield from rows


def _best_match(pred: BoxRecord, targets: Sequence[BoxRecord], matched_indices: set[int]) -> tuple[float, int]:
    best_iou = 0.0
    best_index = -1
    for index, target in enumerate(targets):
        if index in matched_indices or int(pred.cls) != int(target.cls):
            continue
        iou = _iou(pred.xyxy, target.xyxy)
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_iou, best_index


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for run_name, run in summary.get("runs", {}).items():
        for filter_name, metrics in run.get("filters", {}).items():
            fixed05 = metrics.get("fixed_conf", {}).get("conf_0.05_iou_0.50", {})
            fixed25 = metrics.get("fixed_conf", {}).get("conf_0.25_iou_0.50", {})
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(run_name))}</td>"
                f"<td>{html.escape(str(filter_name))}</td>"
                f"<td>{metrics.get('target_gt_count')}</td>"
                f"<td>{_fmt(metrics.get('mAP50'))}</td>"
                f"<td>{_fmt(metrics.get('mAP50_95'))}</td>"
                f"<td>{_fmt(fixed05.get('precision'))}</td>"
                f"<td>{_fmt(fixed05.get('recall'))}</td>"
                f"<td>{fixed05.get('fp')}</td>"
                f"<td>{_fmt(fixed25.get('precision'))}</td>"
                f"<td>{_fmt(fixed25.get('recall'))}</td>"
                f"<td>{fixed25.get('fp')}</td>"
                "</tr>"
            )
    delta_rows = []
    for filter_name, values in summary.get("delta", {}).items():
        threshold = values.get("conf_0.05_iou_0.50", {})
        delta_rows.append(
            "<tr>"
            f"<td>{html.escape(str(filter_name))}</td>"
            f"<td>{_fmt(values.get('delta_mAP50'))}</td>"
            f"<td>{_fmt(values.get('delta_mAP50_95'))}</td>"
            f"<td>{_fmt(threshold.get('delta_precision'))}</td>"
            f"<td>{_fmt(threshold.get('delta_recall'))}</td>"
            f"<td>{threshold.get('delta_fp')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLO Aux Box Calibration</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 18px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d7dde5; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f5f7fa; }}
    code {{ background: #f5f7fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>YOLO Aux Box Calibration</h1>
  <p>RGB detector proposals are unchanged. Only proposal scores are recalibrated using RGB score, box geometry, and PerceptionISP aux-map statistics inside each box.</p>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Run</th><th>Filter</th><th>GT</th><th>AP50</th><th>mAP50-95</th><th>P@0.05</th><th>R@0.05</th><th>FP@0.05</th><th>P@0.25</th><th>R@0.25</th><th>FP@0.25</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Delta: Calibrated - Original</h2>
  <table>
    <thead><tr><th>Filter</th><th>Delta AP50</th><th>Delta mAP50-95</th><th>Delta P@0.05</th><th>Delta R@0.05</th><th>Delta FP@0.05</th></tr></thead>
    <tbody>{''.join(delta_rows)}</tbody>
  </table>
</body>
</html>
"""


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.4f}"
    except Exception:
        return html.escape(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
