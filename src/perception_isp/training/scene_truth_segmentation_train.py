"""Train compact segmentation heads on scene-truth masks.

This module is a fast DNN gate for the PerceptionISP question: if the ground
truth object mask is defined before camera simulation, does a learned
RGB+Aux segmentation head outperform RGB-only heads on held-out scenes?
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html as html_lib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from perception_isp.evaluation.comparison import build_pipeline_images
from perception_isp.evaluation.object_boundary_edge_suite import _dilate, _f1_metrics
from perception_isp.evaluation.scene_truth_aux_calibration import _clean_feature
from perception_isp.evaluation.scene_truth_segmentation_suite import (
    SceneTruthCase,
    _as_eval_sample,
    _mask_boundary,
    _resize_bool_any,
    _resize_rgb,
    _signals,
    make_scene_truth_cases,
)
from perception_isp.core.types import PerceptionISPConfig, RawFrame, json_ready


SUMMARY_FILENAME = "scene_truth_segmentation_train_summary.json"
RUNS = ("human_rgb", "perception_rgb", "perception_rgb_aux")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train compact RGB vs RGB+Aux segmentation heads on scene-truth masks.")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--scene-scale", type=float, default=3.0)
    parser.add_argument("--cfas", default="RGGB,GRBG")
    parser.add_argument("--psf-sigmas", default="0.0,1.2")
    parser.add_argument("--adverse", default="none", choices=("none", "lowlight_noise"))
    parser.add_argument("--adverse-severity", type=float, default=1.0)
    parser.add_argument("--adverse-severities", default="")
    parser.add_argument("--adverse-seed", type=int, default=1009)
    parser.add_argument("--use-camerae2e", action="store_true")
    parser.add_argument("--scene-luminance", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--model-variant", default="tiny_unet", choices=("tiny_unet", "highres_side", "detail_side", "aux_detail_side"))
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-shape-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-shapes", default="thin_small,tiny_bright,thin_long")
    parser.add_argument("--hard-object-crop-dice-weight", type=float, default=0.0)
    parser.add_argument("--hard-object-crop-bce-weight", type=float, default=0.0)
    parser.add_argument("--hard-object-crop-pad", type=int, default=3)
    parser.add_argument("--rgb-aux-boundary-loss-weight", type=float, default=None)
    parser.add_argument("--rgb-aux-hard-shape-loss-weight", type=float, default=None)
    parser.add_argument("--rgb-aux-hard-shapes", default="")
    parser.add_argument("--rgb-aux-hard-object-crop-dice-weight", type=float, default=None)
    parser.add_argument("--rgb-aux-hard-object-crop-bce-weight", type=float, default=None)
    parser.add_argument("--rgb-aux-hard-object-crop-pad", type=int, default=None)
    parser.add_argument("--aux-dropout", type=float, default=0.0)
    parser.add_argument("--rgb-aux-init", default="random", choices=("random", "rgb_preserve_zero_aux"))
    parser.add_argument("--rgb-aux-lr-scale", type=float, default=1.0)
    parser.add_argument("--rgb-aux-distill-weight", type=float, default=0.0)
    parser.add_argument("--threshold-metric", default="mask_iou_mean", choices=("mask_iou_mean", "boundary_f1_mean", "mask_iou_boundary_f1_mean"))
    parser.add_argument("--restore-best-train-loss", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_segmentation_train_v1")
    args = parser.parse_args(argv)

    cfas = _parse_csv(args.cfas)
    psf_sigmas = tuple(float(value) for value in _parse_csv(args.psf_sigmas))
    adverse_specs = _parse_adverse_specs(str(args.adverse), str(args.adverse_severities), default_severity=float(args.adverse_severity))
    base_train_cases = make_scene_truth_cases(
        count=int(args.train_count),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfas=cfas,
        psf_sigmas=psf_sigmas,
        use_camerae2e=bool(args.use_camerae2e),
        scene_luminance=float(args.scene_luminance),
    )
    train_cases = apply_adverse_specs(base_train_cases, specs=adverse_specs, seed=int(args.adverse_seed))
    base_all_cases = make_scene_truth_cases(
        count=int(args.train_count) + int(args.val_count),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfas=cfas,
        psf_sigmas=psf_sigmas,
        use_camerae2e=bool(args.use_camerae2e),
        scene_luminance=float(args.scene_luminance),
    )
    all_cases = apply_adverse_specs(base_all_cases, specs=adverse_specs, seed=int(args.adverse_seed))
    train_scene_ids = {f"scene_{index:03d}" for index in range(max(int(args.train_count), 1))}
    val_cases = tuple(case for case in all_cases if _scene_id(case.case_id) not in train_scene_ids)
    if not val_cases:
        val_cases = tuple(all_cases[-max(len(cfas) * len(psf_sigmas), 1) :])

    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
    )
    summary = run_segmentation_training(
        train_cases=train_cases,
        val_cases=val_cases,
        config=config,
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        base_channels=int(args.base_channels),
        model_variant=str(args.model_variant),
        boundary_loss_weight=float(args.boundary_loss_weight),
        hard_shape_loss_weight=float(args.hard_shape_loss_weight),
        hard_shapes=_parse_csv(args.hard_shapes),
        hard_object_crop_dice_weight=float(args.hard_object_crop_dice_weight),
        hard_object_crop_bce_weight=float(args.hard_object_crop_bce_weight),
        hard_object_crop_pad=int(args.hard_object_crop_pad),
        rgb_aux_boundary_loss_weight=args.rgb_aux_boundary_loss_weight,
        rgb_aux_hard_shape_loss_weight=args.rgb_aux_hard_shape_loss_weight,
        rgb_aux_hard_shapes=_parse_csv(args.rgb_aux_hard_shapes) if str(args.rgb_aux_hard_shapes).strip() else None,
        rgb_aux_hard_object_crop_dice_weight=args.rgb_aux_hard_object_crop_dice_weight,
        rgb_aux_hard_object_crop_bce_weight=args.rgb_aux_hard_object_crop_bce_weight,
        rgb_aux_hard_object_crop_pad=args.rgb_aux_hard_object_crop_pad,
        aux_dropout=float(args.aux_dropout),
        rgb_aux_init=str(args.rgb_aux_init),
        rgb_aux_lr_scale=float(args.rgb_aux_lr_scale),
        rgb_aux_distill_weight=float(args.rgb_aux_distill_weight),
        threshold_metric=str(args.threshold_metric),
        restore_best_train_loss=bool(args.restore_best_train_loss),
        device_name=str(args.device),
        seed=int(args.seed),
    )
    summary["case_generation"] = {
        "train_count": int(args.train_count),
        "val_count": int(args.val_count),
        "width": int(args.width),
        "height": int(args.height),
        "scene_scale": float(args.scene_scale),
        "cfas": list(cfas),
        "psf_sigmas": list(psf_sigmas),
        "use_camerae2e": bool(args.use_camerae2e),
        "scene_luminance": float(args.scene_luminance),
        "adverse": str(args.adverse),
        "adverse_severity": float(args.adverse_severity),
        "adverse_severities": [dict(spec) for spec in adverse_specs],
        "adverse_seed": int(args.adverse_seed),
    }
    html_path = write_segmentation_training_report(summary, args.output_dir)
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


def run_segmentation_training(
    *,
    train_cases: Sequence[SceneTruthCase],
    val_cases: Sequence[SceneTruthCase],
    config: PerceptionISPConfig,
    epochs: int,
    batch: int,
    lr: float,
    base_channels: int,
    device_name: str,
    seed: int,
    model_variant: str = "tiny_unet",
    boundary_loss_weight: float = 0.0,
    hard_shape_loss_weight: float = 0.0,
    hard_shapes: Sequence[str] = (),
    hard_object_crop_dice_weight: float = 0.0,
    hard_object_crop_bce_weight: float = 0.0,
    hard_object_crop_pad: int = 3,
    rgb_aux_boundary_loss_weight: float | None = None,
    rgb_aux_hard_shape_loss_weight: float | None = None,
    rgb_aux_hard_shapes: Sequence[str] | None = None,
    rgb_aux_hard_object_crop_dice_weight: float | None = None,
    rgb_aux_hard_object_crop_bce_weight: float | None = None,
    rgb_aux_hard_object_crop_pad: int | None = None,
    aux_dropout: float = 0.0,
    rgb_aux_init: str = "random",
    rgb_aux_lr_scale: float = 1.0,
    rgb_aux_distill_weight: float = 0.0,
    threshold_metric: str = "mask_iou_mean",
    restore_best_train_loss: bool = False,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _set_seed(torch, seed)
    device = _select_device(torch, device_name)
    train_items = [_prepare_item(case, config=config) for case in train_cases]
    val_items = [_prepare_item(case, config=config) for case in val_cases]
    started = time.perf_counter()
    run_summaries: Dict[str, Any] = {}
    trained_states: Dict[str, Mapping[str, Any]] = {}
    for run_name in RUNS:
        run_boundary_loss_weight = (
            float(rgb_aux_boundary_loss_weight)
            if run_name == "perception_rgb_aux" and rgb_aux_boundary_loss_weight is not None
            else float(boundary_loss_weight)
        )
        run_hard_shape_loss_weight = (
            float(rgb_aux_hard_shape_loss_weight)
            if run_name == "perception_rgb_aux" and rgb_aux_hard_shape_loss_weight is not None
            else float(hard_shape_loss_weight)
        )
        run_hard_shapes = (
            list(rgb_aux_hard_shapes)
            if run_name == "perception_rgb_aux" and rgb_aux_hard_shapes is not None
            else list(hard_shapes)
        )
        run_hard_object_crop_dice_weight = (
            float(rgb_aux_hard_object_crop_dice_weight)
            if run_name == "perception_rgb_aux" and rgb_aux_hard_object_crop_dice_weight is not None
            else float(hard_object_crop_dice_weight)
        )
        run_hard_object_crop_bce_weight = (
            float(rgb_aux_hard_object_crop_bce_weight)
            if run_name == "perception_rgb_aux" and rgb_aux_hard_object_crop_bce_weight is not None
            else float(hard_object_crop_bce_weight)
        )
        run_hard_object_crop_pad = (
            int(rgb_aux_hard_object_crop_pad)
            if run_name == "perception_rgb_aux" and rgb_aux_hard_object_crop_pad is not None
            else int(hard_object_crop_pad)
        )
        channels = _input_channels(train_items[0], run_name)
        model = TinySegNet(channels, base_channels=max(int(base_channels), 4), model_variant=str(model_variant)).to(device)
        init_applied = False
        teacher_model = None
        if run_name == "perception_rgb_aux":
            init_applied = _initialize_rgb_aux_from_rgb_state(torch, model, trained_states.get("perception_rgb"), mode=str(rgb_aux_init))
            if float(rgb_aux_distill_weight) > 0.0 and "perception_rgb" in trained_states:
                teacher_model = TinySegNet(3, base_channels=max(int(base_channels), 4), model_variant=str(model_variant)).to(device)
                teacher_model.load_state_dict(trained_states["perception_rgb"])
                teacher_model.eval()
                for parameter in teacher_model.parameters():
                    parameter.requires_grad_(False)
        run_lr = float(lr) * (float(rgb_aux_lr_scale) if run_name == "perception_rgb_aux" else 1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=run_lr, weight_decay=1.0e-4)
        pos_weight = _positive_weight(train_items)
        history = []
        best_loss = float("inf")
        best_state = None
        for epoch in range(max(int(epochs), 1)):
            model.train()
            losses = []
            for batch_items in _batches(train_items, batch_size=max(int(batch), 1), seed=seed + epoch):
                x = torch.from_numpy(np.stack([_input_tensor(item, run_name) for item in batch_items])).float().to(device)
                if run_name == "perception_rgb_aux":
                    x = _apply_aux_dropout(torch, x, dropout=float(aux_dropout))
                y = torch.from_numpy(np.stack([item["mask"][None, :, :] for item in batch_items])).float().to(device)
                boundary = torch.from_numpy(np.stack([np.asarray(item["boundary"], dtype=np.float32)[None, :, :] for item in batch_items])).float().to(device)
                hard_shape = torch.from_numpy(np.stack([_hard_shape_mask(item, run_hard_shapes)[None, :, :] for item in batch_items])).float().to(device)
                logits = model(x)
                loss = _segmentation_loss(
                    F,
                    torch,
                    logits,
                    y,
                    boundary,
                    hard_shape,
                    pos_weight=pos_weight,
                    boundary_loss_weight=float(run_boundary_loss_weight),
                    hard_shape_loss_weight=float(run_hard_shape_loss_weight),
                )
                if float(run_hard_object_crop_dice_weight) > 0.0:
                    loss = loss + float(run_hard_object_crop_dice_weight) * _hard_object_crop_dice_loss(
                        torch,
                        torch.sigmoid(logits),
                        batch_items,
                        run_hard_shapes,
                        crop_pad=int(run_hard_object_crop_pad),
                    )
                if float(run_hard_object_crop_bce_weight) > 0.0:
                    loss = loss + float(run_hard_object_crop_bce_weight) * _hard_object_crop_bce_loss(
                        F,
                        torch,
                        logits,
                        batch_items,
                        run_hard_shapes,
                        crop_pad=int(run_hard_object_crop_pad),
                    )
                if teacher_model is not None and float(rgb_aux_distill_weight) > 0.0:
                    with torch.no_grad():
                        teacher_score = torch.sigmoid(teacher_model(x[:, :3, :, :]))
                    loss = loss + float(rgb_aux_distill_weight) * F.mse_loss(torch.sigmoid(logits), teacher_score)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses)) if losses else None
            if bool(restore_best_train_loss) and epoch_loss is not None and epoch_loss < best_loss:
                best_loss = float(epoch_loss)
                best_state = _state_dict_cpu_copy(model)
            if epoch == 0 or epoch == max(int(epochs), 1) - 1 or (epoch + 1) % max(int(epochs) // 3, 1) == 0:
                history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if bool(restore_best_train_loss) and best_state is not None:
            model.load_state_dict(best_state)
        train_threshold, train_threshold_curve = _best_threshold(torch, model, train_items, run_name, device, metric=str(threshold_metric))
        val_eval = _evaluate_model(torch, model, val_items, run_name, device, threshold=train_threshold)
        train_eval = _evaluate_model(torch, model, train_items, run_name, device, threshold=train_threshold, include_assets=False)
        run_summaries[run_name] = {
            "channels": int(channels),
            "threshold": float(train_threshold),
            "train_threshold_curve": train_threshold_curve,
            "training_config": {
                "lr": float(run_lr),
                "model_variant": str(model_variant),
                "pos_weight": float(pos_weight),
                "boundary_loss_weight": float(run_boundary_loss_weight),
                "hard_shape_loss_weight": float(run_hard_shape_loss_weight),
                "hard_shapes": list(run_hard_shapes),
                "hard_object_crop_dice_weight": float(run_hard_object_crop_dice_weight),
                "hard_object_crop_bce_weight": float(run_hard_object_crop_bce_weight),
                "hard_object_crop_pad": int(run_hard_object_crop_pad),
                "aux_dropout": float(aux_dropout) if run_name == "perception_rgb_aux" else 0.0,
                "rgb_aux_init": str(rgb_aux_init) if run_name == "perception_rgb_aux" else "n/a",
                "rgb_aux_init_applied": bool(init_applied) if run_name == "perception_rgb_aux" else False,
                "rgb_aux_distill_weight": float(rgb_aux_distill_weight) if run_name == "perception_rgb_aux" else 0.0,
                "threshold_metric": str(threshold_metric),
                "restore_best_train_loss": bool(restore_best_train_loss),
                "best_train_loss": None if best_state is None else float(best_loss),
            },
            "history": history,
            "train": train_eval,
            "val": val_eval,
        }
        trained_states[run_name] = _state_dict_cpu_copy(model)
    comparison = _compare_runs(run_summaries)
    checks = _checks(run_summaries, comparison)
    return {
        "name": "Scene-truth compact segmentation training",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "checks": checks,
        "train_case_count": len(train_items),
        "val_case_count": len(val_items),
        "train_object_count": sum(len(item["objects"]) for item in train_items),
        "val_object_count": sum(len(item["objects"]) for item in val_items),
        "training": {
            "epochs": int(epochs),
            "batch": int(batch),
            "lr": float(lr),
            "base_channels": int(base_channels),
            "model_variant": str(model_variant),
            "boundary_loss_weight": float(boundary_loss_weight),
            "hard_shape_loss_weight": float(hard_shape_loss_weight),
            "hard_shapes": list(hard_shapes),
            "hard_object_crop_dice_weight": float(hard_object_crop_dice_weight),
            "hard_object_crop_bce_weight": float(hard_object_crop_bce_weight),
            "hard_object_crop_pad": int(hard_object_crop_pad),
            "rgb_aux_boundary_loss_weight": None if rgb_aux_boundary_loss_weight is None else float(rgb_aux_boundary_loss_weight),
            "rgb_aux_hard_shape_loss_weight": None if rgb_aux_hard_shape_loss_weight is None else float(rgb_aux_hard_shape_loss_weight),
            "rgb_aux_hard_shapes": None if rgb_aux_hard_shapes is None else list(rgb_aux_hard_shapes),
            "rgb_aux_hard_object_crop_dice_weight": (
                None if rgb_aux_hard_object_crop_dice_weight is None else float(rgb_aux_hard_object_crop_dice_weight)
            ),
            "rgb_aux_hard_object_crop_bce_weight": (
                None if rgb_aux_hard_object_crop_bce_weight is None else float(rgb_aux_hard_object_crop_bce_weight)
            ),
            "rgb_aux_hard_object_crop_pad": None if rgb_aux_hard_object_crop_pad is None else int(rgb_aux_hard_object_crop_pad),
            "aux_dropout": float(aux_dropout),
            "rgb_aux_init": str(rgb_aux_init),
            "rgb_aux_lr_scale": float(rgb_aux_lr_scale),
            "rgb_aux_distill_weight": float(rgb_aux_distill_weight),
            "threshold_metric": str(threshold_metric),
            "restore_best_train_loss": bool(restore_best_train_loss),
            "device": str(device),
            "seed": int(seed),
            "elapsed_s": float(time.perf_counter() - started),
        },
        "runs": run_summaries,
        "comparison": comparison,
        "aggregate": comparison,
        "interpretation": (
            "Compact CNN segmentation heads are trained on pre-sensor vector object masks. "
            "The comparison isolates whether Perception RGB+Aux input helps a learned segmentation head on held-out scene-truth masks."
        ),
        "claim_boundary": (
            "This is a compact binary segmentation gate, not a large YOLO/Mask2Former-style production model. "
            "Positive held-out deltas justify scaling the RGB+Aux segmentation training."
        ),
    }


def _parse_adverse_specs(adverse: str, severities_text: str, *, default_severity: float) -> tuple[Dict[str, Any], ...]:
    values = _parse_csv(severities_text)
    mode = str(adverse or "none").lower()
    if not values:
        return ({"adverse": mode, "severity": None if mode == "none" else float(default_severity), "label": mode if mode == "none" else f"{mode}_{float(default_severity):.2f}"},)
    specs: list[Dict[str, Any]] = []
    seen = set()
    for raw_value in values:
        text = str(raw_value).strip().lower()
        if text in {"none", "nominal", "clean"}:
            spec = {"adverse": "none", "severity": None, "label": "nominal"}
        else:
            severity = float(text)
            if severity <= 0.0:
                spec = {"adverse": "none", "severity": None, "label": "nominal"}
            else:
                if mode == "none":
                    raise ValueError("--adverse-severities with numeric values requires --adverse lowlight_noise")
                spec = {"adverse": mode, "severity": float(severity), "label": f"{mode}_{float(severity):.2f}"}
        key = (spec["adverse"], spec["severity"])
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    if not specs:
        return ({"adverse": mode, "severity": None if mode == "none" else float(default_severity), "label": mode if mode == "none" else f"{mode}_{float(default_severity):.2f}"},)
    return tuple(specs)


def apply_adverse_specs(cases: Sequence[SceneTruthCase], *, specs: Sequence[Mapping[str, Any]], seed: int) -> tuple[SceneTruthCase, ...]:
    if len(specs) == 1:
        spec = specs[0]
        return apply_adverse_condition(
            cases,
            adverse=str(spec.get("adverse", "none")),
            severity=0.0 if spec.get("severity") is None else float(spec.get("severity")),
            seed=int(seed),
        )
    out: list[SceneTruthCase] = []
    for spec in specs:
        mode = str(spec.get("adverse", "none"))
        severity_raw = spec.get("severity")
        if mode == "none" or severity_raw is None:
            for case in cases:
                out.append(
                    replace(
                        case,
                        case_id=f"{case.case_id}_nominal",
                        metadata={
                            **dict(case.metadata),
                            "adverse": "none",
                            "adverse_severity": None,
                            "adverse_mix_label": "nominal",
                        },
                    )
                )
            continue
        label = str(spec.get("label", f"{mode}_{float(severity_raw):.2f}"))
        conditioned = apply_adverse_condition(cases, adverse=mode, severity=float(severity_raw), seed=int(seed))
        for case in conditioned:
            out.append(replace(case, metadata={**dict(case.metadata), "adverse_mix_label": label}))
    return tuple(out)


def apply_adverse_condition(
    cases: Sequence[SceneTruthCase],
    *,
    adverse: str,
    severity: float,
    seed: int,
) -> tuple[SceneTruthCase, ...]:
    mode = str(adverse or "none").lower()
    if mode == "none":
        return tuple(cases)
    if mode != "lowlight_noise":
        raise ValueError(f"unsupported adverse condition: {adverse!r}")
    conditioned = []
    for case in cases:
        case_seed = _stable_case_seed(case.case_id, seed)
        raw = _lowlight_noise_raw(case.raw, severity=max(float(severity), 0.0), seed=case_seed)
        metadata = {
            **dict(case.metadata),
            "adverse": mode,
            "adverse_severity": float(severity),
            "adverse_seed": int(seed),
        }
        conditioned.append(
            replace(
                case,
                case_id=f"{case.case_id}_lowlight_noise_{float(severity):.2f}",
                raw=raw,
                metadata=metadata,
            )
        )
    return tuple(conditioned)


def _lowlight_noise_raw(raw: RawFrame, *, severity: float, seed: int) -> RawFrame:
    rng = np.random.default_rng(int(seed))
    calibration = raw.calibration
    black = float(calibration.black_level)
    white = float(calibration.white_level)
    span = max(white - black, 1.0)
    data = np.asarray(raw.data, dtype=np.float64)
    signal = np.clip((data - black) / span, 0.0, 1.0)
    sev = max(float(severity), 0.0)
    light_scale = float(np.clip(0.38 / max(sev, 1.0e-6), 0.08, 0.75))
    dark = np.clip(signal * light_scale, 0.0, 1.0)
    shot_std = (0.028 + 0.022 * sev) * np.sqrt(np.maximum(dark, 0.0))
    read_std = 0.008 + 0.010 * sev
    noisy = dark + rng.normal(0.0, shot_std, size=dark.shape) + rng.normal(0.0, read_std, size=dark.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    adverse_data = np.round(noisy * span + black).astype(data.dtype)
    raw_metadata = replace(
        raw.metadata,
        noise_model_id=f"{raw.metadata.noise_model_id}_lowlight_noise_{sev:.2f}",
    )
    raw_calibration = replace(
        raw.calibration,
        shot_noise_coeff=float(raw.calibration.shot_noise_coeff * (1.0 + 4.0 * sev)),
        read_noise_var=float(raw.calibration.read_noise_var * (1.0 + 12.0 * sev)),
    )
    return replace(
        raw,
        data=adverse_data,
        metadata=raw_metadata,
        calibration=raw_calibration,
        provenance={
            **dict(raw.provenance),
            "adverse": "lowlight_noise",
            "adverse_severity": float(sev),
            "adverse_light_scale": float(light_scale),
            "adverse_seed": int(seed),
        },
    )


def _stable_case_seed(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def write_segmentation_training_report(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    asset_dir = destination / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(json_ready(summary)))
    for run_name, run in serializable.get("runs", {}).items():
        assets = run.get("val", {}).pop("assets_pending", [])
        manifest = []
        for case_id, case_assets in assets[:6]:
            rendered = {}
            for name, image in case_assets.items():
                filename = f"{run_name}_{case_id}_{name}.png"
                _save_image(asset_dir / filename, np.asarray(image, dtype=np.float64))
                rendered[name] = f"assets/{filename}"
            manifest.append({"case_id": case_id, "assets": rendered})
        run["val"]["asset_manifest"] = manifest
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(serializable), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(serializable), encoding="utf-8")
    return html_path


class TinySegNet:  # placeholder for type checkers; replaced after torch import
    pass


def _make_model_class() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _TinySegNet(nn.Module):
        def __init__(self, in_channels: int, *, base_channels: int = 16, model_variant: str = "tiny_unet") -> None:
            super().__init__()
            b = int(base_channels)
            self.model_variant = str(model_variant)
            if self.model_variant not in {"tiny_unet", "highres_side", "detail_side", "aux_detail_side"}:
                raise ValueError(f"unsupported model_variant: {self.model_variant!r}")
            self.enc1 = nn.Sequential(nn.Conv2d(in_channels, b, 3, padding=1), nn.BatchNorm2d(b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.SiLU())
            self.down = nn.Conv2d(b, b * 2, 3, stride=2, padding=1)
            self.enc2 = nn.Sequential(nn.BatchNorm2d(b * 2), nn.SiLU(), nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.SiLU())
            self.mid = nn.Sequential(nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.SiLU(), nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.SiLU())
            self.dec = nn.Sequential(nn.Conv2d(b * 3, b, 3, padding=1), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.SiLU())
            self.out = nn.Conv2d(b, 1, 1)
            self.side = None
            self.side_gain = None
            self.detail = None
            self.detail_gain = None
            self.detail_input_start = 0
            if self.model_variant == "highres_side":
                self.side = nn.Sequential(nn.Conv2d(b, b, 3, padding=1), nn.SiLU(), nn.Conv2d(b, 1, 1))
                self.side_gain = nn.Parameter(torch.tensor(0.25))
            elif self.model_variant == "detail_side":
                detail_channels = max(b // 2, 4)
                self.detail = nn.Sequential(
                    nn.Conv2d(in_channels, detail_channels, 1),
                    nn.SiLU(),
                    nn.Conv2d(detail_channels, detail_channels, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(detail_channels, 1, 1),
                )
                self.detail_gain = nn.Parameter(torch.tensor(0.15))
            elif self.model_variant == "aux_detail_side" and int(in_channels) > 3:
                detail_channels = max(b // 2, 4)
                self.detail_input_start = 3
                self.detail = nn.Sequential(
                    nn.Conv2d(int(in_channels) - self.detail_input_start, detail_channels, 1),
                    nn.SiLU(),
                    nn.Conv2d(detail_channels, detail_channels, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(detail_channels, 1, 1),
                )
                self.detail_gain = nn.Parameter(torch.tensor(0.20))

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.down(e1))
            mid = self.mid(e2)
            up = F.interpolate(mid, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            logits = self.out(self.dec(torch.cat([e1, up], dim=1)))
            if self.side is not None and self.side_gain is not None:
                logits = logits + self.side_gain * self.side(e1)
            if self.detail is not None and self.detail_gain is not None:
                logits = logits + self.detail_gain * self.detail(x[:, self.detail_input_start :, :, :])
            return logits

    return _TinySegNet


try:
    TinySegNet = _make_model_class()
except ImportError:  # Keep CLI help available in the lightweight installation.
    class TinySegNet:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required; install perception-isp[ml]")


def _prepare_item(case: SceneTruthCase, *, config: PerceptionISPConfig) -> Dict[str, Any]:
    images = build_pipeline_images(_as_eval_sample(case), config=config)
    human_rgb = np.clip(np.asarray(images.human_rgb, dtype=np.float64), 0.0, 1.0)
    perception_rgb = np.clip(np.asarray(images.perception_rgb, dtype=np.float64), 0.0, 1.0)
    aux_maps = {name: np.asarray(value, dtype=np.float64) for name, value in images.aux_maps.items()}
    base_signals = _signals(human_rgb, perception_rgb, aux_maps)
    shape = human_rgb.shape[:2]
    object_masks = []
    object_rows = []
    for obj in case.objects:
        area = _resize_bool_any(obj.mask_high, shape)
        boundary = _resize_bool_any(_mask_boundary(obj.mask_high, thickness=1), shape)
        if not bool(np.any(boundary)):
            boundary = _mask_boundary(area, thickness=1)
        object_masks.append(area)
        object_rows.append({"label": obj.label, "shape_tag": obj.shape_tag, "mask": area, "boundary": boundary})
    union_mask = np.logical_or.reduce(object_masks) if object_masks else np.zeros(shape, dtype=bool)
    union_boundary = _mask_boundary(union_mask, thickness=1)
    aux_channels = np.stack(
        [
            _clean_feature(base_signals["aux_edge_strength"]),
            _clean_feature(base_signals["aux_edge_confidence"]),
            _clean_feature(base_signals["aux_edge_evidence"]),
            _clean_feature(base_signals["aux_strength_gated_confidence"]),
            _clean_feature(aux_maps.get("snr_map", np.zeros(shape, dtype=np.float64))),
            _clean_feature(aux_maps.get("saturation", np.zeros(shape, dtype=np.float64))),
            _clean_feature(aux_maps.get("psf_blur_confidence", np.ones(shape, dtype=np.float64))),
        ],
        axis=2,
    )
    return {
        "case_id": case.case_id,
        "cfa": case.cfa,
        "psf_sigma": float(case.psf_sigma),
        "condition": _condition_label(case),
        "scene_truth_rgb": _resize_rgb(case.rgb_high, shape),
        "human_rgb": human_rgb,
        "perception_rgb": perception_rgb,
        "aux": aux_channels,
        "mask": union_mask.astype(np.float32),
        "boundary": union_boundary.astype(bool),
        "objects": object_rows,
    }


def _condition_label(case: SceneTruthCase) -> str:
    metadata = dict(case.metadata)
    if metadata.get("adverse_mix_label"):
        return str(metadata["adverse_mix_label"])
    adverse = str(metadata.get("adverse", "none"))
    severity = metadata.get("adverse_severity")
    if adverse == "none" or severity is None:
        return "nominal"
    return f"{adverse}_{float(severity):.2f}"


def _input_channels(item: Mapping[str, Any], run_name: str) -> int:
    return int(_input_tensor(item, run_name).shape[0])


def _input_tensor(item: Mapping[str, Any], run_name: str) -> np.ndarray:
    if run_name == "human_rgb":
        arr = np.asarray(item["human_rgb"], dtype=np.float32)
    elif run_name == "perception_rgb":
        arr = np.asarray(item["perception_rgb"], dtype=np.float32)
    elif run_name == "perception_rgb_aux":
        arr = np.concatenate([np.asarray(item["perception_rgb"], dtype=np.float32), np.asarray(item["aux"], dtype=np.float32)], axis=2)
    else:
        raise ValueError(f"unsupported run: {run_name}")
    return np.ascontiguousarray(np.moveaxis(arr, 2, 0))


def _positive_weight(items: Sequence[Mapping[str, Any]]) -> float:
    pos = sum(float(np.sum(item["mask"])) for item in items)
    total = sum(float(np.asarray(item["mask"]).size) for item in items)
    neg = max(total - pos, 1.0)
    return float(np.clip(np.sqrt(neg / max(pos, 1.0)), 1.0, 6.0))


def _segmentation_loss(
    F: Any,
    torch: Any,
    logits: Any,
    target: Any,
    boundary: Any,
    hard_shape: Any,
    *,
    pos_weight: float,
    boundary_loss_weight: float,
    hard_shape_loss_weight: float,
) -> Any:
    pixel_weight = None
    if float(boundary_loss_weight) > 0.0 or float(hard_shape_loss_weight) > 0.0:
        pixel_weight = 1.0 + float(boundary_loss_weight) * boundary + float(hard_shape_loss_weight) * hard_shape
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.tensor(float(pos_weight), device=target.device),
        weight=pixel_weight,
    )
    dice = _dice_loss(torch.sigmoid(logits), target)
    return bce + dice


def _hard_shape_mask(item: Mapping[str, Any], hard_shapes: Sequence[str]) -> np.ndarray:
    tags = {str(value) for value in hard_shapes}
    mask = np.zeros_like(np.asarray(item["mask"], dtype=np.float32), dtype=np.float32)
    if not tags:
        return mask
    for obj in item.get("objects", []):
        if str(obj.get("shape_tag")) in tags:
            mask = np.maximum(mask, np.asarray(obj.get("mask"), dtype=np.float32))
    return mask


def _hard_object_crop_dice_loss(
    torch: Any,
    pred: Any,
    batch_items: Sequence[Mapping[str, Any]],
    hard_shapes: Sequence[str],
    *,
    crop_pad: int,
) -> Any:
    tags = {str(value) for value in hard_shapes}
    losses = []
    for batch_index, item in enumerate(batch_items):
        for obj in item.get("objects", []):
            if str(obj.get("shape_tag")) not in tags:
                continue
            mask = np.asarray(obj.get("mask"), dtype=np.float32)
            bbox = _mask_bbox(mask, pad=int(crop_pad))
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox
            target = torch.from_numpy(mask[y0:y1, x0:x1][None, None, :, :]).float().to(pred.device)
            crop = pred[batch_index : batch_index + 1, :, y0:y1, x0:x1]
            losses.append(_dice_loss(crop, target))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def _hard_object_crop_bce_loss(
    F: Any,
    torch: Any,
    logits: Any,
    batch_items: Sequence[Mapping[str, Any]],
    hard_shapes: Sequence[str],
    *,
    crop_pad: int,
) -> Any:
    tags = {str(value) for value in hard_shapes}
    losses = []
    for batch_index, item in enumerate(batch_items):
        for obj in item.get("objects", []):
            if str(obj.get("shape_tag")) not in tags:
                continue
            mask = np.asarray(obj.get("mask"), dtype=np.float32)
            bbox = _mask_bbox(mask, pad=int(crop_pad))
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox
            target = torch.from_numpy(mask[y0:y1, x0:x1][None, None, :, :]).float().to(logits.device)
            crop_logits = logits[batch_index : batch_index + 1, :, y0:y1, x0:x1]
            pos = torch.clamp(target.sum(), min=1.0)
            total = torch.tensor(float(target.numel()), device=target.device)
            neg = torch.clamp(total - pos, min=1.0)
            local_pos_weight = torch.clamp(torch.sqrt(neg / pos), min=1.0, max=12.0)
            losses.append(F.binary_cross_entropy_with_logits(crop_logits, target, pos_weight=local_pos_weight))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _mask_bbox(mask: np.ndarray, *, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask) > 0.0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape[:2]
    y0 = max(int(np.min(ys)) - int(pad), 0)
    y1 = min(int(np.max(ys)) + int(pad) + 1, height)
    x0 = max(int(np.min(xs)) - int(pad), 0)
    x1 = min(int(np.max(xs)) + int(pad) + 1, width)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _apply_aux_dropout(torch: Any, x: Any, *, dropout: float) -> Any:
    probability = float(dropout)
    if probability <= 0.0 or x.shape[1] <= 3:
        return x
    probability = min(max(probability, 0.0), 0.95)
    keep = (torch.rand((x.shape[0], x.shape[1] - 3, 1, 1), device=x.device) >= probability).float()
    aux = x[:, 3:, :, :] * keep / max(1.0 - probability, 1.0e-6)
    return torch.cat([x[:, :3, :, :], aux], dim=1)


def _initialize_rgb_aux_from_rgb_state(torch: Any, model: Any, source_state: Mapping[str, Any] | None, *, mode: str) -> bool:
    if str(mode) == "random" or not source_state:
        return False
    if str(mode) != "rgb_preserve_zero_aux":
        raise ValueError(f"unsupported rgb_aux_init: {mode!r}")
    with torch.no_grad():
        target_state = model.state_dict()
        for name, source_tensor in source_state.items():
            if name in {"enc1.0.weight", "detail.0.weight"}:
                continue
            if name in target_state and tuple(target_state[name].shape) == tuple(source_tensor.shape):
                target_state[name].copy_(source_tensor.to(device=target_state[name].device, dtype=target_state[name].dtype))
        model.load_state_dict(target_state)
        source_weight = source_state["enc1.0.weight"].to(device=model.enc1[0].weight.device, dtype=model.enc1[0].weight.dtype)
        model.enc1[0].weight.zero_()
        model.enc1[0].weight[:, : source_weight.shape[1], :, :].copy_(source_weight)
        if "detail.0.weight" in source_state and getattr(model, "detail", None) is not None:
            source_detail = source_state["detail.0.weight"].to(device=model.detail[0].weight.device, dtype=model.detail[0].weight.dtype)
            model.detail[0].weight.zero_()
            model.detail[0].weight[:, : source_detail.shape[1], :, :].copy_(source_detail)
    return True


def _state_dict_cpu_copy(model: Any) -> Dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _batches(items: Sequence[Mapping[str, Any]], *, batch_size: int, seed: int) -> list[list[Mapping[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(items))
    return [[items[int(i)] for i in indices[start : start + batch_size]] for start in range(0, len(indices), batch_size)]


def _dice_loss(pred: Any, target: Any) -> Any:
    import torch

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inter = torch.sum(pred_flat * target_flat, dim=1)
    denom = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1)
    return 1.0 - torch.mean((2.0 * inter + 1.0) / (denom + 1.0))


def _best_threshold(
    torch: Any,
    model: Any,
    items: Sequence[Mapping[str, Any]],
    run_name: str,
    device: Any,
    *,
    metric: str = "mask_iou_mean",
) -> tuple[float, list[Dict[str, float]]]:
    scores = _predict_scores(torch, model, items, run_name, device)
    curve = _threshold_curve(scores, items)
    best = _best_curve_row(curve, metric)
    return float(best.get("threshold", 0.5)), curve


def _threshold_curve(scores: Sequence[np.ndarray], items: Sequence[Mapping[str, Any]]) -> list[Dict[str, float]]:
    curve = []
    for threshold in np.linspace(0.05, 0.995, 64):
        rows = [
            _seg_metrics(score >= float(threshold), score, np.asarray(item["mask"], dtype=bool), np.asarray(item["boundary"], dtype=bool))
            for score, item in zip(scores, items)
        ]
        aggregate = _aggregate(rows)
        curve.append(
            {
                "threshold": float(threshold),
                "mask_iou_mean": float(aggregate.get("mask_iou_mean") or 0.0),
                "mask_dice_mean": float(aggregate.get("mask_dice_mean") or 0.0),
                "mask_precision_mean": float(aggregate.get("mask_precision_mean") or 0.0),
                "mask_recall_mean": float(aggregate.get("mask_recall_mean") or 0.0),
                "boundary_f1_mean": float(aggregate.get("boundary_f1_mean") or 0.0),
                "mask_iou_boundary_f1_mean": float(
                    0.5 * float(aggregate.get("mask_iou_mean") or 0.0)
                    + 0.5 * float(aggregate.get("boundary_f1_mean") or 0.0)
                ),
            }
        )
    return curve


def _object_oracle_by_slice(scores: Sequence[np.ndarray], items: Sequence[Mapping[str, Any]], *, key: str) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[tuple[np.ndarray, Mapping[str, Any]]]] = {}
    for score, item in zip(scores, items):
        for obj in item.get("objects", []):
            if key == "cfa":
                value = str(item.get("cfa", ""))
            elif key == "psf_sigma":
                value = str(item.get("psf_sigma", ""))
            elif key == "condition":
                value = str(item.get("condition", ""))
            else:
                value = str(obj.get(key, ""))
            grouped.setdefault(value, []).append((score, obj))
    rows = []
    for value, pairs in sorted(grouped.items()):
        curve = []
        for threshold in np.linspace(0.05, 0.995, 64):
            metrics = [
                _seg_metrics(
                    score >= float(threshold),
                    score,
                    np.asarray(obj["mask"], dtype=bool),
                    np.asarray(obj["boundary"], dtype=bool),
                )
                for score, obj in pairs
            ]
            aggregate = _aggregate(metrics)
            curve.append(
                {
                    "threshold": float(threshold),
                    "mask_iou_mean": float(aggregate.get("mask_iou_mean") or 0.0),
                    "boundary_f1_mean": float(aggregate.get("boundary_f1_mean") or 0.0),
                    "score_boundary_separation_mean": float(aggregate.get("score_boundary_separation_mean") or 0.0),
                }
            )
        rows.append(
            {
                key: value,
                "object_count": len(pairs),
                "oracle_by_mask_iou": _best_curve_row(curve, "mask_iou_mean"),
                "oracle_by_boundary_f1": _best_curve_row(curve, "boundary_f1_mean"),
            }
        )
    return rows


def _best_curve_row(curve: Sequence[Mapping[str, float]], metric: str) -> Dict[str, float]:
    if not curve:
        return {"threshold": 0.5, metric: 0.0}
    return dict(max(curve, key=lambda row: float(row.get(metric, 0.0))))


def _predict_scores(torch: Any, model: Any, items: Sequence[Mapping[str, Any]], run_name: str, device: Any) -> list[np.ndarray]:
    model.eval()
    scores = []
    with torch.no_grad():
        for item in items:
            x = torch.from_numpy(_input_tensor(item, run_name)[None, :, :, :]).float().to(device)
            score = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            scores.append(np.asarray(score, dtype=np.float64))
    return scores


def _evaluate_model(
    torch: Any,
    model: Any,
    items: Sequence[Mapping[str, Any]],
    run_name: str,
    device: Any,
    *,
    threshold: float,
    include_assets: bool = True,
) -> Dict[str, Any]:
    scores = _predict_scores(torch, model, items, run_name, device)
    threshold_curve = _threshold_curve(scores, items)
    oracle_by_mask_iou = _best_curve_row(threshold_curve, "mask_iou_mean")
    oracle_by_boundary_f1 = _best_curve_row(threshold_curve, "boundary_f1_mean")
    object_oracle_by_shape = _object_oracle_by_slice(scores, items, key="shape_tag")
    object_oracle_by_cfa = _object_oracle_by_slice(scores, items, key="cfa")
    object_oracle_by_psf = _object_oracle_by_slice(scores, items, key="psf_sigma")
    object_oracle_by_condition = _object_oracle_by_slice(scores, items, key="condition")
    case_rows = []
    object_rows = []
    assets = []
    for item, score in zip(items, scores):
        pred = score >= float(threshold)
        mask = np.asarray(item["mask"], dtype=bool)
        boundary = np.asarray(item["boundary"], dtype=bool)
        case_metrics = _seg_metrics(pred, score, mask, boundary)
        case_rows.append(
            {
                "case_id": item["case_id"],
                "cfa": item["cfa"],
                "psf_sigma": float(item["psf_sigma"]),
                "condition": item.get("condition", ""),
                "metrics": case_metrics,
            }
        )
        for obj in item["objects"]:
            object_metrics = _seg_metrics(pred, score, np.asarray(obj["mask"], dtype=bool), np.asarray(obj["boundary"], dtype=bool))
            local_metrics = _local_object_metrics(
                pred,
                score,
                np.asarray(obj["mask"], dtype=bool),
                np.asarray(obj["boundary"], dtype=bool),
                pad=4,
            )
            object_rows.append(
                {
                    "case_id": item["case_id"],
                    "cfa": item["cfa"],
                    "psf_sigma": float(item["psf_sigma"]),
                    "condition": item.get("condition", ""),
                    "label": obj["label"],
                    "shape_tag": obj["shape_tag"],
                    **object_metrics,
                    **local_metrics,
                }
            )
        if include_assets:
            assets.append(
                (
                    item["case_id"],
                    {
                        "scene_truth_rgb": item["scene_truth_rgb"],
                        "target_mask": mask.astype(np.float64),
                        "score": score,
                        "prediction": pred.astype(np.float64),
                    },
                )
            )
    return {
        "case_count": len(case_rows),
        "object_count": len(object_rows),
        "threshold": float(threshold),
        "threshold_curve": threshold_curve,
        "oracle_by_mask_iou": oracle_by_mask_iou,
        "oracle_by_boundary_f1": oracle_by_boundary_f1,
        "object_oracle_by_shape": object_oracle_by_shape,
        "object_oracle_by_cfa": object_oracle_by_cfa,
        "object_oracle_by_psf": object_oracle_by_psf,
        "object_oracle_by_condition": object_oracle_by_condition,
        "case_aggregate": _aggregate([row["metrics"] for row in case_rows]),
        "aggregate": _aggregate(object_rows),
        "by_shape": _breakdown(object_rows, "shape_tag"),
        "by_cfa": _breakdown(object_rows, "cfa"),
        "by_psf": _breakdown(object_rows, "psf_sigma"),
        "by_condition": _breakdown(object_rows, "condition"),
        "cases": case_rows,
        "objects": object_rows,
        "assets_pending": assets,
    }


def _seg_metrics(pred: np.ndarray, score: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    boundary = np.asarray(boundary, dtype=bool)
    inter = float(np.sum(pred & mask))
    union = float(np.sum(pred | mask))
    precision = inter / max(float(np.sum(pred)), 1.0)
    recall = inter / max(float(np.sum(mask)), 1.0)
    iou = inter / union if union > 0.0 else 1.0
    dice = 2.0 * inter / max(float(np.sum(pred) + np.sum(mask)), 1.0)
    pred_boundary = _mask_boundary(pred, thickness=1)
    boundary_f1 = _f1_metrics(pred_boundary, boundary, tolerance=2)
    on = float(np.mean(score[boundary])) if bool(np.any(boundary)) else 0.0
    off_mask = np.logical_not(_dilate(boundary, radius=2))
    off = float(np.mean(score[off_mask])) if bool(np.any(off_mask)) else 0.0
    return {
        "mask_iou": float(iou),
        "mask_dice": float(dice),
        "mask_precision": float(precision),
        "mask_recall": float(recall),
        "boundary_f1": float(boundary_f1["f1"]),
        "boundary_precision": float(boundary_f1["precision"]),
        "boundary_recall": float(boundary_f1["recall"]),
        "score_boundary_separation": float(on - off),
    }


def _local_object_metrics(pred: np.ndarray, score: np.ndarray, mask: np.ndarray, boundary: np.ndarray, *, pad: int) -> Dict[str, float]:
    bbox = _mask_bbox(np.asarray(mask, dtype=np.float32), pad=int(pad))
    if bbox is None:
        metrics = _seg_metrics(pred, score, mask, boundary)
    else:
        y0, y1, x0, x1 = bbox
        metrics = _seg_metrics(pred[y0:y1, x0:x1], score[y0:y1, x0:x1], mask[y0:y1, x0:x1], boundary[y0:y1, x0:x1])
    return {f"local_{key}": float(value) for key, value in metrics.items()}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"object_count": len(rows)}
    for key in (
        "mask_iou",
        "mask_dice",
        "mask_precision",
        "mask_recall",
        "boundary_f1",
        "score_boundary_separation",
        "local_mask_iou",
        "local_mask_dice",
        "local_mask_precision",
        "local_mask_recall",
        "local_boundary_f1",
        "local_score_boundary_separation",
    ):
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        out[f"{key}_mean"] = float(np.mean(values)) if values else None
    return out


def _breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return [{key: value, **_aggregate(group)} for value, group in sorted(grouped.items())]


def _compare_runs(runs: Mapping[str, Any]) -> Dict[str, Any]:
    val = {name: payload["val"]["case_aggregate"] for name, payload in runs.items()}
    object_val = {name: payload["val"]["aggregate"] for name, payload in runs.items()}
    def delta(candidate: str, baseline: str, metric: str) -> float | None:
        c = val.get(candidate, {}).get(metric)
        b = val.get(baseline, {}).get(metric)
        return None if c is None or b is None else float(c) - float(b)

    return {
        "perception_rgb_minus_human_rgb": {
            "delta_mask_iou_mean": delta("perception_rgb", "human_rgb", "mask_iou_mean"),
            "delta_boundary_f1_mean": delta("perception_rgb", "human_rgb", "boundary_f1_mean"),
        },
        "perception_rgb_aux_minus_perception_rgb": {
            "delta_mask_iou_mean": delta("perception_rgb_aux", "perception_rgb", "mask_iou_mean"),
            "delta_boundary_f1_mean": delta("perception_rgb_aux", "perception_rgb", "boundary_f1_mean"),
        },
        "perception_rgb_aux_minus_human_rgb": {
            "delta_mask_iou_mean": delta("perception_rgb_aux", "human_rgb", "mask_iou_mean"),
            "delta_boundary_f1_mean": delta("perception_rgb_aux", "human_rgb", "boundary_f1_mean"),
        },
        "val_metrics": val,
        "val_object_metrics": object_val,
    }


def _checks(runs: Mapping[str, Any], comparison: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rgb_aux = comparison.get("perception_rgb_aux_minus_perception_rgb", {})
    aux_iou = rgb_aux.get("delta_mask_iou_mean")
    aux_boundary = rgb_aux.get("delta_boundary_f1_mean")
    finite = all(np.isfinite(float(value)) for value in (aux_iou or 0.0, aux_boundary or 0.0))
    return [
        {
            "id": "all_runs_completed",
            "status": "pass" if all(name in runs for name in RUNS) else "fail",
            "description": "Human RGB, Perception RGB, and Perception RGB+Aux segmentation heads trained and evaluated.",
            "criteria": [{"metric": "run_count", "value": len(runs), "threshold": len(RUNS), "pass": all(name in runs for name in RUNS)}],
        },
        {
            "id": "finite_heldout_metrics",
            "status": "pass" if finite else "fail",
            "description": "Held-out RGB+Aux deltas are finite.",
            "criteria": [{"metric": "finite", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "rgb_aux_beats_perception_rgb_mask_iou",
            "status": "pass" if aux_iou is not None and float(aux_iou) > 0.0 else "fail",
            "description": "RGB+Aux segmentation should beat Perception RGB-only mask IoU on held-out scene truth.",
            "criteria": [{"metric": "delta_mask_iou_mean", "value": aux_iou, "threshold": 0.0, "pass": aux_iou is not None and float(aux_iou) > 0.0}],
        },
        {
            "id": "rgb_aux_beats_perception_rgb_boundary_f1",
            "status": "pass" if aux_boundary is not None and float(aux_boundary) > 0.0 else "fail",
            "description": "RGB+Aux segmentation should beat Perception RGB-only boundary F1 on held-out scene truth.",
            "criteria": [{"metric": "delta_boundary_f1_mean", "value": aux_boundary, "threshold": 0.0, "pass": aux_boundary is not None and float(aux_boundary) > 0.0}],
        },
    ]


def _save_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=2)
    else:
        rgb = arr[:, :, :3]
    Image.fromarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _render_html(summary: Mapping[str, Any]) -> str:
    comparison = summary.get("comparison", {})
    val_metrics = comparison.get("val_metrics", {})
    rows = []
    for name in RUNS:
        metric = val_metrics.get(name, {})
        run = summary.get("runs", {}).get(name, {})
        oracle_iou = run.get("val", {}).get("oracle_by_mask_iou", {})
        oracle_boundary = run.get("val", {}).get("oracle_by_boundary_f1", {})
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(name)}</td>"
            f"<td>{_fmt(run.get('threshold'))}</td>"
            f"<td>{_fmt(metric.get('mask_iou_mean'))}</td>"
            f"<td>{_fmt(metric.get('mask_dice_mean'))}</td>"
            f"<td>{_fmt(metric.get('boundary_f1_mean'))}</td>"
            f"<td>{_fmt(metric.get('score_boundary_separation_mean'), signed=True)}</td>"
            f"<td>{_fmt(oracle_iou.get('threshold'))} / {_fmt(oracle_iou.get('mask_iou_mean'))}</td>"
            f"<td>{_fmt(oracle_boundary.get('threshold'))} / {_fmt(oracle_boundary.get('boundary_f1_mean'))}</td>"
            "</tr>"
        )
    shape_rows = []
    rgb_aux_shapes = summary.get("runs", {}).get("perception_rgb_aux", {}).get("val", {}).get("by_shape", [])
    for row in rgb_aux_shapes:
        shape_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('shape_tag', '')))}</td>"
            f"<td>{int(row.get('object_count', 0))}</td>"
            f"<td>{_fmt(row.get('mask_iou_mean'))}</td>"
            f"<td>{_fmt(row.get('boundary_f1_mean'))}</td>"
            "</tr>"
        )
    assets = summary.get("runs", {}).get("perception_rgb_aux", {}).get("val", {}).get("asset_manifest", [])[:4]
    asset_blocks = []
    for item in assets:
        cols = []
        for name, path in item.get("assets", {}).items():
            cols.append(f"<div><b>{html_lib.escape(str(name))}</b><img src='{html_lib.escape(str(path))}'></div>")
        asset_blocks.append(f"<h3>{html_lib.escape(str(item.get('case_id', '')))}</h3><div class='assets'>{''.join(cols)}</div>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Segmentation Training</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #16202a; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .tile {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 10px 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .assets {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 22px; }}
    .assets img {{ width: 100%; border: 1px solid #d8dee4; background: #000; }}
  </style>
</head>
<body>
  <h1>Scene-truth Segmentation Training</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <div class="grid">
    <div class="tile"><b>Status</b><br>{html_lib.escape(str(summary.get('status', '')))}</div>
    <div class="tile"><b>Train/Val cases</b><br>{int(summary.get('train_case_count', 0))} / {int(summary.get('val_case_count', 0))}</div>
    <div class="tile"><b>RGB+Aux dIoU</b><br>{_fmt(comparison.get('perception_rgb_aux_minus_perception_rgb', {}).get('delta_mask_iou_mean'), signed=True)}</div>
    <div class="tile"><b>RGB+Aux dBoundary</b><br>{_fmt(comparison.get('perception_rgb_aux_minus_perception_rgb', {}).get('delta_boundary_f1_mean'), signed=True)}</div>
  </div>
  <h2>Held-out Metrics</h2>
  <table><thead><tr><th>Run</th><th>Train Threshold</th><th>Mask IoU</th><th>Dice</th><th>Boundary F1</th><th>Score Boundary Separation</th><th>Val Oracle IoU Thr/IoU</th><th>Val Oracle Boundary Thr/F1</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>RGB+Aux Shape Slices</h2>
  <table><thead><tr><th>Shape</th><th>N</th><th>Mask IoU</th><th>Boundary F1</th></tr></thead><tbody>{''.join(shape_rows)}</tbody></table>
  <h2>RGB+Aux Visual Cases</h2>
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


def _scene_id(case_id: str) -> str:
    parts = str(case_id).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(case_id)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split(",") if token.strip())


def _set_seed(torch: Any, seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "manual_seed_all", None)):
        torch.cuda.manual_seed_all(int(seed))


def _select_device(torch: Any, value: str) -> Any:
    requested = str(value or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
