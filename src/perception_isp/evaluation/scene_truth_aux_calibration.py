"""Train a small aux-map boundary calibrator on scene-truth masks.

This is intentionally lightweight: it learns a logistic pixel classifier from
PerceptionISP aux maps to pre-sensor object-boundary truth, then evaluates on a
held-out scene split. It is a fast engineering gate before committing GPU time
to a larger RGB+Aux segmentation DNN.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from perception_isp.evaluation.comparison import build_pipeline_images
from perception_isp.evaluation.object_boundary_edge_suite import _edge_mask, _edge_strength, _f1_metrics, _luma, _normalize
from perception_isp.evaluation.scene_truth_segmentation_suite import (
    SceneTruthCase,
    _as_eval_sample,
    _boundary_metrics,
    _context_mask,
    _mask_boundary,
    _resize_bool_any,
    _resize_rgb,
    _signals,
    make_scene_truth_cases,
)
from perception_isp.core.types import PerceptionISPConfig, json_ready


SUMMARY_FILENAME = "scene_truth_aux_calibration_summary.json"
AUX_FEATURES = (
    "aux_edge_strength",
    "aux_edge_confidence",
    "aux_edge_evidence",
    "aux_strength_gated_confidence",
    "saturation",
    "snr_map",
    "psf_blur_confidence",
)
RGB_AUX_FEATURES = AUX_FEATURES + ("perception_rgb_edge",)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate aux-map scene-truth boundary calibration.")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--scene-scale", type=float, default=3.0)
    parser.add_argument("--cfas", default="RGGB,GRBG,BGGR,GBRG")
    parser.add_argument("--psf-sigmas", default="0.0,1.2")
    parser.add_argument("--use-camerae2e", action="store_true")
    parser.add_argument("--scene-luminance", type=float, default=100.0)
    parser.add_argument("--boundary-thickness", type=int, default=1)
    parser.add_argument("--context-radius", type=int, default=8)
    parser.add_argument("--samples-per-case", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_aux_calibration_v1")
    args = parser.parse_args(argv)

    cfas = _parse_csv(args.cfas)
    psf_sigmas = tuple(float(value) for value in _parse_csv(args.psf_sigmas))
    train_cases = make_scene_truth_cases(
        count=int(args.train_count),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfas=cfas,
        psf_sigmas=psf_sigmas,
        use_camerae2e=bool(args.use_camerae2e),
        scene_luminance=float(args.scene_luminance),
    )
    val_cases_all = make_scene_truth_cases(
        count=int(args.train_count) + int(args.val_count),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfas=cfas,
        psf_sigmas=psf_sigmas,
        use_camerae2e=bool(args.use_camerae2e),
        scene_luminance=float(args.scene_luminance),
    )
    train_scene_ids = {f"scene_{index:03d}" for index in range(max(int(args.train_count), 1))}
    val_cases = tuple(case for case in val_cases_all if case.case_id.split("_", 2)[0] + "_" + case.case_id.split("_", 2)[1] not in train_scene_ids)
    if not val_cases:
        val_cases = tuple(val_cases_all[-max(len(cfas) * len(psf_sigmas), 1) :])

    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
    )
    summary = run_aux_calibration(
        train_cases=train_cases,
        val_cases=val_cases,
        config=config,
        boundary_thickness=int(args.boundary_thickness),
        context_radius=int(args.context_radius),
        samples_per_case=int(args.samples_per_case),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
    )
    html_path = write_aux_calibration_report(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "train_case_count": summary["train_case_count"],
                    "val_case_count": summary["val_case_count"],
                    "aggregate": summary["aggregate"],
                }
            ),
            indent=2,
        )
    )
    return 0


def run_aux_calibration(
    *,
    train_cases: Sequence[SceneTruthCase],
    val_cases: Sequence[SceneTruthCase],
    config: PerceptionISPConfig,
    boundary_thickness: int,
    context_radius: int,
    samples_per_case: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    train_prepared = [_prepare_case(case, config=config, boundary_thickness=boundary_thickness, context_radius=context_radius) for case in train_cases]
    val_prepared = [_prepare_case(case, config=config, boundary_thickness=boundary_thickness, context_radius=context_radius) for case in val_cases]

    aux_x, aux_y = _sample_training_matrix(train_prepared, AUX_FEATURES, samples_per_case=samples_per_case, rng=rng)
    rgb_aux_x, rgb_aux_y = _sample_training_matrix(train_prepared, RGB_AUX_FEATURES, samples_per_case=samples_per_case, rng=rng)
    aux_model = _fit_logistic(aux_x, aux_y, epochs=epochs, lr=lr, l2=l2)
    rgb_aux_model = _fit_logistic(rgb_aux_x, rgb_aux_y, epochs=epochs, lr=lr, l2=l2)

    aux_train_scores = _case_scores(train_prepared, AUX_FEATURES, aux_model)
    rgb_aux_train_scores = _case_scores(train_prepared, RGB_AUX_FEATURES, rgb_aux_model)
    aux_threshold = _best_threshold(aux_train_scores)
    rgb_aux_threshold = _best_threshold(rgb_aux_train_scores)

    train_eval = _evaluate_prepared_cases(
        train_prepared,
        learned_signals={
            "aux_learned_boundary": (AUX_FEATURES, aux_model, aux_threshold),
            "rgb_aux_learned_boundary": (RGB_AUX_FEATURES, rgb_aux_model, rgb_aux_threshold),
        },
    )
    val_eval = _evaluate_prepared_cases(
        val_prepared,
        learned_signals={
            "aux_learned_boundary": (AUX_FEATURES, aux_model, aux_threshold),
            "rgb_aux_learned_boundary": (RGB_AUX_FEATURES, rgb_aux_model, rgb_aux_threshold),
        },
    )
    checks = _checks(train_eval, val_eval)
    return {
        "name": "Scene-truth aux boundary calibration",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "checks": checks,
        "train_case_count": len(train_prepared),
        "val_case_count": len(val_prepared),
        "train_object_count": len(train_eval["objects"]),
        "val_object_count": len(val_eval["objects"]),
        "feature_sets": {
            "aux_only": list(AUX_FEATURES),
            "rgb_aux": list(RGB_AUX_FEATURES),
        },
        "training": {
            "samples_per_case": int(samples_per_case),
            "epochs": int(epochs),
            "lr": float(lr),
            "l2": float(l2),
            "seed": int(seed),
            "aux_threshold": float(aux_threshold),
            "rgb_aux_threshold": float(rgb_aux_threshold),
            "aux_weights": _model_to_dict(aux_model, AUX_FEATURES),
            "rgb_aux_weights": _model_to_dict(rgb_aux_model, RGB_AUX_FEATURES),
        },
        "train": train_eval,
        "val": val_eval,
        "aggregate": val_eval["aggregate"],
        "interpretation": (
            "A logistic calibrator is trained on pre-sensor vector object-boundary truth and evaluated on held-out scenes. "
            "The aux-only model uses only PerceptionISP aux maps; rgb_aux additionally uses PerceptionISP RGB edge strength."
        ),
        "claim_boundary": (
            "This is learned boundary calibration, not a full segmentation DNN. It is a fast gate to decide whether larger RGB+Aux segmentation training is worth running."
        ),
    }


def write_aux_calibration_report(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    asset_dir = destination / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(summary)
    for split in ("train", "val"):
        payload = dict(serializable[split])
        assets = payload.pop("assets_pending", [])
        manifest = []
        for case_id, case_assets in assets[:8]:
            rendered = {}
            for name, image in case_assets.items():
                filename = f"{split}_{case_id}_{name}.png"
                _save_image(asset_dir / filename, image)
                rendered[name] = f"assets/{filename}"
            manifest.append({"case_id": case_id, "assets": rendered})
        payload["asset_manifest"] = manifest
        serializable[split] = payload
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(serializable), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(serializable), encoding="utf-8")
    return html_path


def _prepare_case(
    case: SceneTruthCase,
    *,
    config: PerceptionISPConfig,
    boundary_thickness: int,
    context_radius: int,
) -> Dict[str, Any]:
    images = build_pipeline_images(_as_eval_sample(case), config=config)
    human_rgb = np.asarray(images.human_rgb, dtype=np.float64)
    perception_rgb = np.asarray(images.perception_rgb, dtype=np.float64)
    aux_maps = {name: np.asarray(value, dtype=np.float64) for name, value in images.aux_maps.items()}
    base_signals = _signals(human_rgb, perception_rgb, aux_maps)
    shape = human_rgb.shape[:2]
    object_rows = []
    object_boundaries = []
    object_areas = []
    for obj in case.objects:
        area = _resize_bool_any(obj.mask_high, shape)
        boundary = _resize_bool_any(_mask_boundary(obj.mask_high, thickness=boundary_thickness), shape)
        if not bool(np.any(boundary)):
            boundary = _mask_boundary(area, thickness=1)
        context = _context_mask(area, boundary, radius=context_radius)
        object_rows.append(
            {
                "label": obj.label,
                "shape_tag": obj.shape_tag,
                "boundary": boundary,
                "area": area,
                "context": context,
            }
        )
        object_boundaries.append(boundary)
        object_areas.append(area)
    union_boundary = np.logical_or.reduce(object_boundaries) if object_boundaries else np.zeros(shape, dtype=bool)
    union_area = np.logical_or.reduce(object_areas) if object_areas else np.zeros(shape, dtype=bool)
    union_context = _context_mask(union_area, union_boundary, radius=context_radius)
    features = _feature_maps(base_signals, aux_maps)
    return {
        "case": case,
        "shape": shape,
        "human_rgb": np.clip(human_rgb, 0.0, 1.0),
        "perception_rgb": np.clip(perception_rgb, 0.0, 1.0),
        "base_signals": base_signals,
        "features": features,
        "objects": object_rows,
        "union_boundary": union_boundary,
        "union_context": union_context,
        "finite_outputs": bool(
            np.isfinite(human_rgb).all()
            and np.isfinite(perception_rgb).all()
            and all(np.isfinite(value).all() for value in aux_maps.values())
        ),
    }


def _feature_maps(base_signals: Mapping[str, np.ndarray], aux_maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    zeros = np.zeros_like(next(iter(base_signals.values())))
    features = {
        "aux_edge_strength": _clean_feature(base_signals["aux_edge_strength"]),
        "aux_edge_confidence": _clean_feature(base_signals["aux_edge_confidence"]),
        "aux_edge_evidence": _clean_feature(base_signals["aux_edge_evidence"]),
        "aux_strength_gated_confidence": _clean_feature(base_signals["aux_strength_gated_confidence"]),
        "perception_rgb_edge": _clean_feature(base_signals["perception_rgb_edge"]),
        "saturation": _clean_feature(aux_maps.get("saturation", zeros)),
        "snr_map": _clean_feature(_normalize(np.nan_to_num(np.asarray(aux_maps.get("snr_map", zeros), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))),
        "psf_blur_confidence": _clean_feature(aux_maps.get("psf_blur_confidence", np.ones_like(zeros))),
    }
    return features


def _clean_feature(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _sample_training_matrix(
    prepared_cases: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    samples_per_case: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in prepared_cases:
        boundary = np.asarray(row["union_boundary"], dtype=bool)
        context = np.asarray(row["union_context"], dtype=bool)
        pos = np.flatnonzero(boundary.reshape(-1))
        neg = np.flatnonzero((context & np.logical_not(boundary)).reshape(-1))
        if not len(pos) or not len(neg):
            continue
        half = max(int(samples_per_case) // 2, 1)
        pos_idx = rng.choice(pos, size=min(len(pos), half), replace=len(pos) < half)
        neg_idx = rng.choice(neg, size=min(len(neg), half), replace=len(neg) < half)
        indices = np.concatenate([pos_idx, neg_idx])
        labels = np.concatenate([np.ones(len(pos_idx), dtype=np.float64), np.zeros(len(neg_idx), dtype=np.float64)])
        features = _stack_features(row["features"], feature_names).reshape(-1, len(feature_names))[indices]
        xs.append(features)
        ys.append(labels)
    if not xs:
        raise ValueError("no trainable pixels found")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order]


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, epochs: int, lr: float, l2: float) -> Dict[str, Any]:
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1.0e-6
    z = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(max(int(epochs), 1)):
        logits = _safe_linear(z, weights, bias)
        pred = _sigmoid(logits)
        err = pred - y
        with np.errstate(all="ignore"):
            grad_w = z.T @ err / max(float(len(y)), 1.0) + float(l2) * weights
        grad_b = float(np.mean(err))
        grad_w = np.clip(np.nan_to_num(grad_w, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
        grad_b = float(np.clip(np.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0))
        weights -= float(lr) * grad_w
        bias -= float(lr) * grad_b
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0))
    logits = _safe_linear(z, weights, bias)
    loss = -np.mean(y * np.log(_sigmoid(logits) + 1.0e-9) + (1.0 - y) * np.log(1.0 - _sigmoid(logits) + 1.0e-9))
    return {"weights": weights, "bias": float(bias), "mean": mean, "std": std, "loss": float(loss)}


def _case_scores(prepared_cases: Sequence[Mapping[str, Any]], feature_names: Sequence[str], model: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for row in prepared_cases:
        x = _stack_features(row["features"], feature_names)
        score = _predict_logistic(x, model)
        out.append((score, np.asarray(row["union_boundary"], dtype=bool)))
    return out


def _best_threshold(scores_and_masks: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.05, 0.95, 37)
    for threshold in thresholds:
        f1s = []
        for score, mask in scores_and_masks:
            f1s.append(_f1_metrics(score >= threshold, mask, tolerance=2)["f1"])
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = float(threshold)
    return best_threshold


def _evaluate_prepared_cases(
    prepared_cases: Sequence[Mapping[str, Any]],
    *,
    learned_signals: Mapping[str, tuple[Sequence[str], Mapping[str, Any], float]],
) -> Dict[str, Any]:
    case_rows = []
    object_rows = []
    assets_pending = []
    for row in prepared_cases:
        case: SceneTruthCase = row["case"]
        signals = dict(row["base_signals"])
        for signal_name, (feature_names, model, _threshold) in learned_signals.items():
            signals[signal_name] = _predict_logistic(_stack_features(row["features"], feature_names), model)
        case_metrics = _metrics_with_learned_thresholds(
            signals,
            row["union_boundary"],
            row["union_context"],
            learned_thresholds={name: spec[2] for name, spec in learned_signals.items()},
        )
        case_metrics.update({"finite_outputs": bool(row["finite_outputs"]), "object_count": len(row["objects"])})
        case_rows.append({"case_id": case.case_id, "cfa": case.cfa, "psf_sigma": float(case.psf_sigma), "metrics": case_metrics})
        for obj in row["objects"]:
            metrics = _metrics_with_learned_thresholds(
                signals,
                obj["boundary"],
                obj["context"],
                learned_thresholds={name: spec[2] for name, spec in learned_signals.items()},
            )
            object_rows.append(
                {
                    "case_id": case.case_id,
                    "label": obj["label"],
                    "shape_tag": obj["shape_tag"],
                    "cfa": case.cfa,
                    "psf_sigma": float(case.psf_sigma),
                    **metrics,
                }
            )
        assets_pending.append(
            (
                case.case_id,
                {
                    "scene_truth_rgb": _resize_rgb(case.rgb_high, row["shape"]),
                    "scene_truth_boundary": np.asarray(row["union_boundary"], dtype=np.float64),
                    "human_edge": row["base_signals"]["human_rgb_edge"],
                    "aux_edge_evidence": row["base_signals"]["aux_edge_evidence"],
                    "aux_learned_boundary": signals["aux_learned_boundary"],
                    "rgb_aux_learned_boundary": signals["rgb_aux_learned_boundary"],
                },
            )
        )
    return {
        "case_count": len(case_rows),
        "object_count": len(object_rows),
        "aggregate": _aggregate(object_rows),
        "by_shape": _breakdown(object_rows, "shape_tag"),
        "by_cfa": _breakdown(object_rows, "cfa"),
        "by_psf": _breakdown(object_rows, "psf_sigma"),
        "cases": case_rows,
        "objects": object_rows,
        "assets_pending": assets_pending,
    }


def _metrics_with_learned_thresholds(
    signals: Mapping[str, np.ndarray],
    boundary: np.ndarray,
    context: np.ndarray,
    *,
    learned_thresholds: Mapping[str, float],
) -> Dict[str, float]:
    metrics = _boundary_metrics(
        {name: value for name, value in signals.items() if name in ("human_rgb_edge", "perception_rgb_edge", "aux_edge_strength", "aux_edge_confidence", "aux_edge_evidence", "aux_strength_gated_confidence")},
        boundary,
        context,
    )
    human_f1 = float(metrics["human_rgb_edge_boundary_f1"])
    perception_f1 = float(metrics["perception_rgb_edge_boundary_f1"])
    for name, threshold in learned_thresholds.items():
        score = np.asarray(signals[name], dtype=np.float64)
        pred = score >= float(threshold)
        f1 = _f1_metrics(pred, boundary, tolerance=2)
        metrics[f"{name}_boundary_precision"] = f1["precision"]
        metrics[f"{name}_boundary_recall"] = f1["recall"]
        metrics[f"{name}_boundary_f1"] = f1["f1"]
        metrics[f"{name}_threshold"] = float(threshold)
        metrics[f"{name}_minus_human_boundary_f1"] = float(f1["f1"] - human_f1)
        metrics[f"{name}_minus_perception_rgb_boundary_f1"] = float(f1["f1"] - perception_f1)
    return metrics


def _aggregate(object_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"object_count": len(object_rows)}
    metric_names = (
        "human_rgb_edge_boundary_f1",
        "perception_rgb_edge_boundary_f1",
        "aux_edge_evidence_boundary_f1",
        "aux_learned_boundary_boundary_f1",
        "rgb_aux_learned_boundary_boundary_f1",
        "perception_rgb_minus_human_boundary_f1",
        "aux_edge_evidence_minus_human_boundary_f1",
        "aux_learned_boundary_minus_human_boundary_f1",
        "rgb_aux_learned_boundary_minus_human_boundary_f1",
        "aux_learned_boundary_minus_perception_rgb_boundary_f1",
        "rgb_aux_learned_boundary_minus_perception_rgb_boundary_f1",
    )
    for name in metric_names:
        values = [float(row.get(name, 0.0)) for row in object_rows]
        out[f"{name}_mean"] = float(np.mean(values)) if values else None
        if "minus" in name:
            out[f"{name}_win_rate"] = float(np.mean([value > 0.0 for value in values])) if values else None
    return out


def _breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    out = []
    for value, group in sorted(grouped.items()):
        payload = {key: value, "object_count": len(group)}
        payload.update(_aggregate(group))
        out.append(payload)
    return out


def _checks(train_eval: Mapping[str, Any], val_eval: Mapping[str, Any]) -> list[Dict[str, Any]]:
    aggregate = val_eval.get("aggregate", {})
    aux_delta = aggregate.get("aux_learned_boundary_minus_human_boundary_f1_mean")
    rgb_aux_delta = aggregate.get("rgb_aux_learned_boundary_minus_perception_rgb_boundary_f1_mean")
    finite = bool(np.isfinite(float(aux_delta or 0.0)) and np.isfinite(float(rgb_aux_delta or 0.0)))
    return [
        {
            "id": "train_val_present",
            "status": "pass" if train_eval.get("case_count", 0) > 0 and val_eval.get("case_count", 0) > 0 else "fail",
            "description": "Train and held-out validation scene sets are both non-empty.",
            "criteria": [
                {"metric": "train_case_count", "value": int(train_eval.get("case_count", 0)), "threshold": 1, "pass": train_eval.get("case_count", 0) > 0},
                {"metric": "val_case_count", "value": int(val_eval.get("case_count", 0)), "threshold": 1, "pass": val_eval.get("case_count", 0) > 0},
            ],
        },
        {
            "id": "finite_calibration_metrics",
            "status": "pass" if finite else "fail",
            "description": "Held-out calibration deltas are finite.",
            "criteria": [{"metric": "finite", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "heldout_aux_mean_beats_human",
            "status": "pass" if aux_delta is not None and float(aux_delta) > 0.0 else "fail",
            "description": "Aux-only learned boundary signal should beat Human RGB edge F1 on held-out scene truth before claiming Aux superiority.",
            "criteria": [{"metric": "aux_learned_minus_human_f1_mean", "value": aux_delta, "threshold": 0.0, "pass": aux_delta is not None and float(aux_delta) > 0.0}],
        },
        {
            "id": "heldout_rgb_aux_mean_beats_perception_rgb",
            "status": "pass" if rgb_aux_delta is not None and float(rgb_aux_delta) > 0.0 else "fail",
            "description": "RGB+Aux learned boundary signal should beat Perception RGB edge F1 on held-out scene truth.",
            "criteria": [
                {
                    "metric": "rgb_aux_learned_minus_perception_rgb_f1_mean",
                    "value": rgb_aux_delta,
                    "threshold": 0.0,
                    "pass": rgb_aux_delta is not None and float(rgb_aux_delta) > 0.0,
                }
            ],
        },
    ]


def _stack_features(features: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.stack([_clean_feature(features[name]) for name in names], axis=2)


def _predict_logistic(x: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    shape = arr.shape[:-1]
    flat = arr.reshape(-1, arr.shape[-1])
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    z = np.nan_to_num((flat - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    logits = _safe_linear(z, weights, float(model["bias"]))
    return _sigmoid(logits).reshape(shape)


def _safe_linear(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        logits = x @ weights + float(bias)
    return np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _model_to_dict(model: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    weights = np.asarray(model["weights"], dtype=np.float64)
    return {
        "features": list(names),
        "weights": {name: float(weights[index]) for index, name in enumerate(names)},
        "bias": float(model["bias"]),
        "mean": {name: float(np.asarray(model["mean"])[index]) for index, name in enumerate(names)},
        "std": {name: float(np.asarray(model["std"])[index]) for index, name in enumerate(names)},
        "loss": float(model.get("loss", 0.0)),
    }


def _save_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=2)
    else:
        rgb = arr[:, :, :3]
    Image.fromarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split(",") if token.strip())


def _render_html(summary: Mapping[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    train = summary.get("train", {})
    val = summary.get("val", {})
    shape_rows = []
    for row in val.get("by_shape", []):
        agg = row
        shape_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('shape_tag', '')))}</td>"
            f"<td>{int(row.get('object_count', 0))}</td>"
            f"<td>{_fmt(agg.get('human_rgb_edge_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(agg.get('aux_edge_evidence_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(agg.get('aux_learned_boundary_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(agg.get('rgb_aux_learned_boundary_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(agg.get('aux_learned_boundary_minus_human_boundary_f1_mean'), signed=True)}</td>"
            "</tr>"
        )
    cfa_rows = []
    for row in val.get("by_cfa", []):
        cfa_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('cfa', '')))}</td>"
            f"<td>{int(row.get('object_count', 0))}</td>"
            f"<td>{_fmt(row.get('human_rgb_edge_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('aux_learned_boundary_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('rgb_aux_learned_boundary_boundary_f1_mean'))}</td>"
            f"<td>{_fmt(row.get('aux_learned_boundary_minus_human_boundary_f1_mean'), signed=True)}</td>"
            "</tr>"
        )
    asset_blocks = []
    for item in val.get("asset_manifest", [])[:4]:
        cols = []
        for name, path in item.get("assets", {}).items():
            cols.append(f"<div><b>{html_lib.escape(str(name))}</b><img src='{html_lib.escape(str(path))}'></div>")
        asset_blocks.append(f"<h3>{html_lib.escape(str(item.get('case_id', '')))}</h3><div class='assets'>{''.join(cols)}</div>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Aux Boundary Calibration</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #16202a; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .tile {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 10px 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .assets {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-bottom: 22px; }}
    .assets img {{ width: 100%; border: 1px solid #d8dee4; background: #000; }}
  </style>
</head>
<body>
  <h1>Scene-truth Aux Boundary Calibration</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <div class="grid">
    <div class="tile"><b>Status</b><br>{html_lib.escape(str(summary.get('status', '')))}</div>
    <div class="tile"><b>Train/Val cases</b><br>{int(summary.get('train_case_count', 0))} / {int(summary.get('val_case_count', 0))}</div>
    <div class="tile"><b>Aux learned dF1 vs Human</b><br>{_fmt(aggregate.get('aux_learned_boundary_minus_human_boundary_f1_mean'), signed=True)}</div>
    <div class="tile"><b>RGB+Aux dF1 vs RGB</b><br>{_fmt(aggregate.get('rgb_aux_learned_boundary_minus_perception_rgb_boundary_f1_mean'), signed=True)}</div>
  </div>
  <h2>Held-out Aggregate</h2>
  <table><tbody>
    <tr><td>Human RGB edge F1</td><td>{_fmt(aggregate.get('human_rgb_edge_boundary_f1_mean'))}</td></tr>
    <tr><td>Perception RGB edge F1</td><td>{_fmt(aggregate.get('perception_rgb_edge_boundary_f1_mean'))}</td></tr>
    <tr><td>Raw aux evidence F1</td><td>{_fmt(aggregate.get('aux_edge_evidence_boundary_f1_mean'))}</td></tr>
    <tr><td>Aux learned F1</td><td>{_fmt(aggregate.get('aux_learned_boundary_boundary_f1_mean'))}</td></tr>
    <tr><td>RGB+Aux learned F1</td><td>{_fmt(aggregate.get('rgb_aux_learned_boundary_boundary_f1_mean'))}</td></tr>
    <tr><td>Aux learned minus Human F1</td><td>{_fmt(aggregate.get('aux_learned_boundary_minus_human_boundary_f1_mean'), signed=True)}</td></tr>
    <tr><td>RGB+Aux learned minus Perception RGB F1</td><td>{_fmt(aggregate.get('rgb_aux_learned_boundary_minus_perception_rgb_boundary_f1_mean'), signed=True)}</td></tr>
  </tbody></table>
  <h2>Held-out Shape Slices</h2>
  <table><thead><tr><th>Shape</th><th>N</th><th>Human</th><th>Raw Aux Evidence</th><th>Aux Learned</th><th>RGB+Aux Learned</th><th>Aux dF1</th></tr></thead><tbody>{''.join(shape_rows)}</tbody></table>
  <h2>Held-out CFA Slices</h2>
  <table><thead><tr><th>CFA</th><th>N</th><th>Human</th><th>Aux Learned</th><th>RGB+Aux Learned</th><th>Aux dF1</th></tr></thead><tbody>{''.join(cfa_rows)}</tbody></table>
  <h2>Visual Cases</h2>
  {''.join(asset_blocks)}
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return html_lib.escape(str(value))
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
