"""Compact class-aware RGB+aux detector training.

This is a practical reference path for learning from PerceptionISP auxiliary
maps without depending on a full YOLO training fork. It predicts objectness,
box coordinates, and class labels on a fixed grid over the RGB+aux tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from perception_isp.core.aux_dnn import (
    RGB_AUX_TENSOR_KEY,
    apply_channel_mask,
    channel_mask_for_mode,
    channels_for_tensor_key,
    chw_tensor_key,
    labels_from_manifest,
    make_aux_dense_detector_model,
    make_torch_dataset,
    normalize_channel_mode,
)
from perception_isp.core.types import json_ready


DEFAULT_BOX_ENCODING = "cell_center_size"
BOX_ENCODINGS = ("cell_center_size", "xyxy")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train a compact class-aware detector on PerceptionISP RGB+aux tensors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per optimizer/eval step.")
    parser.add_argument("--grid", default="15x20", help="Detector grid as HxW.")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument(
        "--model-architecture",
        default="early_fusion",
        choices=["early_fusion", "early_fusion_multiscale", "late_fusion"],
        help="Dense detector architecture.",
    )
    parser.add_argument("--box-encoding", default=DEFAULT_BOX_ENCODING, choices=BOX_ENCODINGS)
    parser.add_argument(
        "--positive-cell-radius",
        type=int,
        default=0,
        help="For xyxy box encoding, also train neighboring grid cells around each object center.",
    )
    parser.add_argument("--tensor-key", default=RGB_AUX_TENSOR_KEY, help="Tensor key to train on: rgb_aux_chw or rgb_aux_extended_chw.")
    parser.add_argument("--channel-mode", default="rgb_aux", choices=["rgb_aux", "rgb_only", "aux_only"], help="Input ablation mode.")
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--split-strategy", default="coverage", choices=["coverage", "hash", "sequential"])
    parser.add_argument("--include-labels", default=None, help="Comma-separated class labels to train/evaluate; default uses all labels.")
    parser.add_argument("--no-object-weight", type=float, default=0.15)
    parser.add_argument(
        "--object-loss-mode",
        default="balanced_positive_negative",
        choices=["balanced_positive_negative", "negative_focal"],
        help="Objectness loss mode. negative_focal focuses the no-object term on hard false-positive cells.",
    )
    parser.add_argument("--negative-focal-gamma", type=float, default=2.0, help="Gamma for --object-loss-mode negative_focal.")
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument(
        "--small-object-weight",
        type=float,
        default=1.0,
        help="Positive-cell loss multiplier for boxes at or below --small-object-area-threshold.",
    )
    parser.add_argument("--small-object-area-threshold", type=float, default=32.0 * 32.0)
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed for dense training.")
    parser.add_argument("--initial-checkpoint", default=None, help="Optional dense-detector checkpoint to warm-start from.")
    parser.add_argument(
        "--zero-aux-input-weights",
        action="store_true",
        help="After warm-start loading, zero the first early-fusion conv weights for aux channels so RGB-only behavior is preserved initially.",
    )
    parser.add_argument("--save-epoch-checkpoints", action="store_true", help="Save one dense-detector checkpoint per completed epoch.")
    parser.add_argument("--estimate-samples", default="100,1000,10000")
    parser.add_argument("--output-dir", default="exports/perception_rgb_aux_train_dense")
    args = parser.parse_args(argv)

    summary = train_dense(
        manifest_path=args.manifest,
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        device_name=str(args.device),
        batch_size=int(args.batch_size),
        grid_size=parse_grid_size(args.grid),
        base_channels=int(args.base_channels),
        model_architecture=str(args.model_architecture),
        box_encoding=str(args.box_encoding),
        positive_cell_radius=int(args.positive_cell_radius),
        tensor_key=str(args.tensor_key),
        channel_mode=str(args.channel_mode),
        eval_fraction=float(args.eval_fraction),
        split_strategy=str(args.split_strategy),
        include_labels=parse_label_list(args.include_labels),
        no_object_weight=float(args.no_object_weight),
        object_loss_mode=str(args.object_loss_mode),
        negative_focal_gamma=float(args.negative_focal_gamma),
        box_weight=float(args.box_weight),
        class_weight=float(args.class_weight),
        small_object_weight=float(args.small_object_weight),
        small_object_area_threshold=float(args.small_object_area_threshold),
        seed=args.seed,
        initial_checkpoint=args.initial_checkpoint,
        zero_aux_input_weights=bool(args.zero_aux_input_weights),
        save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        estimate_samples=parse_estimate_samples(args.estimate_samples),
        output_dir=args.output_dir,
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def train_dense(
    *,
    manifest_path: str | Path,
    epochs: int = 20,
    learning_rate: float = 1.0e-3,
    device_name: str = "auto",
    batch_size: int = 1,
    grid_size: Tuple[int, int] = (15, 20),
    base_channels: int = 24,
    model_architecture: str = "early_fusion",
    box_encoding: str = DEFAULT_BOX_ENCODING,
    positive_cell_radius: int = 0,
    tensor_key: str = RGB_AUX_TENSOR_KEY,
    channel_mode: str = "rgb_aux",
    eval_fraction: float = 0.25,
    split_strategy: str = "coverage",
    include_labels: Sequence[str] | None = None,
    no_object_weight: float = 0.15,
    object_loss_mode: str = "balanced_positive_negative",
    negative_focal_gamma: float = 2.0,
    box_weight: float = 5.0,
    class_weight: float = 1.0,
    small_object_weight: float = 1.0,
    small_object_area_threshold: float = 32.0 * 32.0,
    seed: int | None = None,
    initial_checkpoint: str | Path | None = None,
    zero_aux_input_weights: bool = False,
    save_epoch_checkpoints: bool = False,
    estimate_samples: Sequence[int] = (100, 1000, 10000),
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    seed_info = _set_training_seed(seed, torch)
    resolved_tensor_key = chw_tensor_key(tensor_key)
    channels = channels_for_tensor_key(resolved_tensor_key)
    dataset = make_torch_dataset(manifest_path, tensor_key=resolved_tensor_key)
    if len(dataset) <= 0:
        raise ValueError("manifest contains no samples")
    class_names = _selected_class_names(labels_from_manifest(manifest_path), include_labels)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    normalized_channel_mode = normalize_channel_mode(channel_mode)
    normalized_box_encoding = _normalize_box_encoding(box_encoding)
    effective_positive_cell_radius = _normalize_positive_cell_radius(
        positive_cell_radius,
        box_encoding=normalized_box_encoding,
    )
    channel_mask = channel_mask_for_mode(normalized_channel_mode, channels=channels)
    device = _select_device(torch, device_name)
    effective_batch_size = max(int(batch_size), 1)
    train_indices, eval_indices = _split_indices(dataset, eval_fraction, strategy=split_strategy, allowed_labels=set(class_names))
    train_class_names = _class_names_for_indices(dataset, train_indices, allowed_labels=set(class_names))
    eval_class_names = _class_names_for_indices(dataset, eval_indices, allowed_labels=set(class_names))
    missing_eval_class_names = tuple(sorted(set(eval_class_names) - set(train_class_names)))

    model = make_aux_dense_detector_model(
        num_classes=len(class_names),
        grid_size=grid_size,
        base_channels=int(base_channels),
        in_channels=len(channels),
        architecture=str(model_architecture),
    ).to(device)
    initialization = _apply_initial_checkpoint(
        model,
        initial_checkpoint,
        torch,
        input_channels=len(channels),
        zero_aux_input_weights=bool(zero_aux_input_weights),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=5.0e-4)
    history = []
    first_loss = None
    last_loss = None
    total_steps = 0
    best_state = None
    best_epoch = -1
    best_loss = None
    best_loss_kind = "eval_loss" if eval_indices else "train_loss"
    saved_epoch_states = []
    epoch_count = max(int(epochs), 1)
    start_time = time.perf_counter()
    initial_eval_loss = _evaluate_loss(
        model,
        dataset,
        eval_indices,
        class_to_index,
        device,
        torch,
        F,
        no_object_weight=no_object_weight,
        object_loss_mode=object_loss_mode,
        negative_focal_gamma=negative_focal_gamma,
        box_weight=box_weight,
        class_weight=class_weight,
        small_object_weight=small_object_weight,
        small_object_area_threshold=small_object_area_threshold,
        channel_mask=channel_mask,
        batch_size=effective_batch_size,
        box_encoding=normalized_box_encoding,
        positive_cell_radius=effective_positive_cell_radius,
    )
    for epoch in range(epoch_count):
        model.train()
        epoch_loss = 0.0
        for batch_indices in _batched_indices(train_indices, effective_batch_size):
            loss = _batch_loss(
                model,
                dataset,
                batch_indices,
                class_to_index,
                device,
                torch,
                F,
                no_object_weight=no_object_weight,
                object_loss_mode=object_loss_mode,
                negative_focal_gamma=negative_focal_gamma,
                box_weight=box_weight,
                class_weight=class_weight,
                small_object_weight=small_object_weight,
                small_object_area_threshold=small_object_area_threshold,
                channel_mask=channel_mask,
                box_encoding=normalized_box_encoding,
                positive_cell_radius=effective_positive_cell_radius,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            epoch_loss += value * len(batch_indices)
            total_steps += len(batch_indices)
        train_mean = float(epoch_loss / max(len(train_indices), 1))
        eval_mean = _evaluate_loss(
            model,
            dataset,
            eval_indices,
            class_to_index,
            device,
            torch,
            F,
            no_object_weight=no_object_weight,
            object_loss_mode=object_loss_mode,
            negative_focal_gamma=negative_focal_gamma,
            box_weight=box_weight,
            class_weight=class_weight,
            small_object_weight=small_object_weight,
            small_object_area_threshold=small_object_area_threshold,
            channel_mask=channel_mask,
            batch_size=effective_batch_size,
            box_encoding=normalized_box_encoding,
            positive_cell_radius=effective_positive_cell_radius,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_mean,
                "eval_loss": eval_mean,
            }
        )
        if bool(save_epoch_checkpoints):
            saved_epoch_states.append(
                {
                    "epoch": int(epoch),
                    "train_loss": train_mean,
                    "eval_loss": eval_mean,
                    "model_state": _clone_state_dict_cpu(model),
                }
            )
        selected_loss = eval_mean if eval_mean is not None else train_mean
        if selected_loss is not None and (best_loss is None or float(selected_loss) < float(best_loss)):
            best_loss = float(selected_loss)
            best_epoch = int(epoch)
            best_state = _clone_state_dict_cpu(model)
    elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-9)
    sample_epochs_per_second = float(total_steps / elapsed_seconds)
    summary = {
        "manifest": str(manifest_path),
        "sample_count": int(len(dataset)),
        "train_sample_count": int(len(train_indices)),
        "eval_sample_count": int(len(eval_indices)),
        "device": str(device),
        "epochs": int(epoch_count),
        "learning_rate": float(learning_rate),
        "batch_size": int(effective_batch_size),
        "grid_size": [int(grid_size[0]), int(grid_size[1])],
        "base_channels": int(base_channels),
        "model_architecture": str(model_architecture),
        "box_encoding": normalized_box_encoding,
        "positive_cell_radius": int(effective_positive_cell_radius),
        "tensor_key": resolved_tensor_key,
        "input_channels": len(channels),
        "channel_mode": normalized_channel_mode,
        "channel_mask": [float(value) for value in channel_mask],
        "channels": list(channels),
        "split_strategy": str(split_strategy),
        "eval_fraction": float(eval_fraction),
        "include_labels": list(class_names),
        "train_indices": [int(index) for index in train_indices],
        "eval_indices": [int(index) for index in eval_indices],
        "class_names": list(class_names),
        "class_count": int(len(class_names)),
        "train_class_names": list(train_class_names),
        "eval_class_names": list(eval_class_names),
        "missing_eval_class_names": list(missing_eval_class_names),
        "no_object_weight": float(no_object_weight),
        "object_loss_mode": str(object_loss_mode),
        "negative_focal_gamma": float(negative_focal_gamma),
        "box_weight": float(box_weight),
        "class_weight": float(class_weight),
        "small_object_weight": float(small_object_weight),
        "small_object_area_threshold": float(small_object_area_threshold),
        "seed": None if seed is None else int(seed),
        "seed_info": seed_info,
        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
        "initialization": initialization,
        "zero_aux_input_weights": bool(zero_aux_input_weights),
        "save_epoch_checkpoints": bool(save_epoch_checkpoints),
        "epoch_checkpoints": [],
        "first_loss": float(first_loss if first_loss is not None else 0.0),
        "last_loss": float(last_loss if last_loss is not None else 0.0),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": history[-1]["eval_loss"] if history else initial_eval_loss,
        "checkpoint_epoch": int(best_epoch),
        "checkpoint_loss": best_loss,
        "checkpoint_loss_kind": best_loss_kind,
        "elapsed_seconds": float(elapsed_seconds),
        "total_sample_epochs": int(total_steps),
        "sample_epochs_per_second": sample_epochs_per_second,
        "seconds_per_sample_epoch": float(1.0 / max(sample_epochs_per_second, 1.0e-12)),
        "time_estimates": _time_estimates(
            sample_epochs_per_second=sample_epochs_per_second,
            epochs=epoch_count,
            sample_counts=estimate_samples,
        ),
        "history": history,
        "purpose": "compact class-aware RGB+aux detector; useful for pipeline validation, not yet a production detector",
    }
    if output_dir is not None:
        destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        final_state = _clone_state_dict_cpu(model)
        epoch_checkpoint_rows = []
        checkpoint = {
            "model_type": "rgb_aux_dense_detector_v1",
            "model_state": best_state if best_state is not None else _clone_state_dict_cpu(model),
            "channels": list(channels),
            "tensor_key": resolved_tensor_key,
            "input_channels": len(channels),
            "channel_mode": normalized_channel_mode,
            "channel_mask": [float(value) for value in channel_mask],
            "grid_size": [int(grid_size[0]), int(grid_size[1])],
            "base_channels": int(base_channels),
            "model_architecture": str(model_architecture),
            "class_names": list(class_names),
            "box_encoding": normalized_box_encoding,
            "summary": summary,
        }
        if bool(save_epoch_checkpoints):
            epoch_dir = destination / "epoch_checkpoints"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for epoch_state in saved_epoch_states:
                epoch = int(epoch_state["epoch"])
                epoch_checkpoint = dict(checkpoint)
                epoch_checkpoint["model_state"] = epoch_state["model_state"]
                epoch_checkpoint["checkpoint_kind"] = "epoch"
                epoch_checkpoint["epoch"] = epoch
                epoch_checkpoint["train_loss"] = epoch_state["train_loss"]
                epoch_checkpoint["eval_loss"] = epoch_state["eval_loss"]
                epoch_path = epoch_dir / f"epoch_{epoch:03d}.pt"
                torch.save(epoch_checkpoint, epoch_path)
                epoch_checkpoint_rows.append(
                    {
                        "epoch": epoch,
                        "checkpoint": str(epoch_path),
                        "train_loss": epoch_state["train_loss"],
                        "eval_loss": epoch_state["eval_loss"],
                    }
                )
            summary["epoch_checkpoints"] = epoch_checkpoint_rows
        checkpoint_path = destination / "rgb_aux_dense_detector.pt"
        torch.save(checkpoint, checkpoint_path)
        summary["checkpoint"] = str(checkpoint_path)
        final_checkpoint = dict(checkpoint)
        final_checkpoint["model_state"] = final_state
        final_checkpoint["checkpoint_kind"] = "final"
        final_checkpoint_path = destination / "rgb_aux_dense_detector_final.pt"
        torch.save(final_checkpoint, final_checkpoint_path)
        summary["final_checkpoint"] = str(final_checkpoint_path)
        (destination / "train_dense_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


def parse_grid_size(value: str) -> Tuple[int, int]:
    normalized = str(value or "").lower().replace(",", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError("grid must be formatted as HxW")
    rows, cols = int(parts[0]), int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("grid dimensions must be positive")
    return rows, cols


def parse_estimate_samples(value: str) -> Tuple[int, ...]:
    samples = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if count > 0:
            samples.append(count)
    return tuple(samples) or (100, 1000, 10000)


def parse_label_list(value: str | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    labels = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return labels or None


def _selected_class_names(all_labels: Sequence[str], include_labels: Sequence[str] | None) -> Tuple[str, ...]:
    available = tuple(str(value) for value in all_labels)
    if include_labels is None:
        return available
    requested = {str(value) for value in include_labels}
    selected = tuple(label for label in available if label in requested)
    if not selected:
        raise ValueError("include_labels did not match any labels in manifest")
    return selected


def _normalize_box_encoding(value: str | None) -> str:
    normalized = str(value or DEFAULT_BOX_ENCODING).lower().replace("-", "_")
    if normalized not in BOX_ENCODINGS:
        raise ValueError(f"unsupported box_encoding: {value!r}")
    return normalized


def _normalize_positive_cell_radius(value: int | float | str | None, *, box_encoding: str) -> int:
    radius = max(int(value or 0), 0)
    if radius > 0 and str(box_encoding) != "xyxy":
        raise ValueError("positive_cell_radius > 0 requires box_encoding='xyxy'")
    return radius


def _sample_loss(
    model: Any,
    dataset: Any,
    index: int,
    class_to_index: Mapping[str, int],
    device: Any,
    torch: Any,
    functional: Any,
    *,
    no_object_weight: float,
    object_loss_mode: str,
    negative_focal_gamma: float,
    box_weight: float,
    class_weight: float,
    small_object_weight: float = 1.0,
    small_object_area_threshold: float = 32.0 * 32.0,
    channel_mask: Sequence[float],
    box_encoding: str = DEFAULT_BOX_ENCODING,
    positive_cell_radius: int = 0,
) -> Any:
    return _batch_loss(
        model,
        dataset,
        (int(index),),
        class_to_index,
        device,
        torch,
        functional,
        no_object_weight=no_object_weight,
        object_loss_mode=object_loss_mode,
        negative_focal_gamma=negative_focal_gamma,
        box_weight=box_weight,
        class_weight=class_weight,
        small_object_weight=small_object_weight,
        small_object_area_threshold=small_object_area_threshold,
        channel_mask=channel_mask,
        box_encoding=box_encoding,
        positive_cell_radius=positive_cell_radius,
    )


def _batch_loss(
    model: Any,
    dataset: Any,
    indices: Sequence[int],
    class_to_index: Mapping[str, int],
    device: Any,
    torch: Any,
    functional: Any,
    *,
    no_object_weight: float,
    object_loss_mode: str,
    negative_focal_gamma: float,
    box_weight: float,
    class_weight: float,
    small_object_weight: float = 1.0,
    small_object_area_threshold: float = 32.0 * 32.0,
    channel_mask: Sequence[float],
    box_encoding: str = DEFAULT_BOX_ENCODING,
    positive_cell_radius: int = 0,
) -> Any:
    if not indices:
        raise ValueError("batch loss requires at least one index")
    batch = [dataset[int(index)] for index in indices]
    tensors = torch.stack([item[0] for item in batch], dim=0).to(device)
    x = apply_channel_mask(tensors, channel_mask)
    pred = model(x)
    grid_size = (int(pred.shape[2]), int(pred.shape[3]))
    targets = [
        _dense_targets(
            target,
            class_to_index,
            grid_size,
            device,
            torch,
            box_encoding=box_encoding,
            positive_cell_radius=positive_cell_radius,
            small_object_weight=small_object_weight,
            small_object_area_threshold=small_object_area_threshold,
        )
        for _, target in batch
    ]
    object_target = torch.cat([item[0] for item in targets], dim=0)
    box_target = torch.stack([item[1] for item in targets], dim=0)
    class_target = torch.stack([item[2] for item in targets], dim=0)
    positive_weight = torch.stack([item[3] for item in targets], dim=0)
    return _dense_prediction_loss(
        pred,
        object_target,
        box_target,
        class_target,
        positive_weight,
        torch,
        functional,
        no_object_weight=no_object_weight,
        object_loss_mode=object_loss_mode,
        negative_focal_gamma=negative_focal_gamma,
        box_weight=box_weight,
        class_weight=class_weight,
    )


def _dense_prediction_loss(
    pred: Any,
    object_target: Any,
    box_target: Any,
    class_target: Any,
    positive_weight: Any,
    torch: Any,
    functional: Any,
    *,
    no_object_weight: float,
    object_loss_mode: str,
    negative_focal_gamma: float,
    box_weight: float,
    class_weight: float,
) -> Any:
    object_logits = pred[:, 0, :, :]
    box_pred = torch.sigmoid(pred[:, 1:5, :, :])
    class_logits = pred[:, 5:, :, :]

    positive = object_target > 0.5
    object_loss_raw = functional.binary_cross_entropy_with_logits(object_logits, object_target, reduction="none")
    negative = object_target <= 0.5
    if bool(torch.any(positive)):
        positive_object_loss = _weighted_mean(object_loss_raw[positive], positive_weight[positive], torch)
    else:
        positive_object_loss = object_logits.sum() * 0.0
    if bool(torch.any(negative)):
        negative_losses = object_loss_raw[negative]
        if str(object_loss_mode) == "negative_focal":
            negative_probs = torch.sigmoid(object_logits[negative])
            focal_weight = torch.pow(negative_probs.clamp(0.0, 1.0), float(negative_focal_gamma))
            negative_object_loss = (focal_weight * negative_losses).mean()
        elif str(object_loss_mode) == "balanced_positive_negative":
            negative_object_loss = negative_losses.mean()
        else:
            raise ValueError(f"unsupported object_loss_mode: {object_loss_mode}")
    else:
        negative_object_loss = object_logits.sum() * 0.0
    object_loss = positive_object_loss + float(no_object_weight) * negative_object_loss

    if bool(torch.any(positive)):
        pred_boxes = box_pred.permute(0, 2, 3, 1)[positive].contiguous()
        target_boxes = box_target.permute(0, 2, 3, 1)[positive].contiguous()
        box_loss_per_cell = functional.smooth_l1_loss(pred_boxes, target_boxes, reduction="none").mean(dim=1)
        positive_weights = positive_weight[positive].contiguous()
        box_loss = _weighted_mean(box_loss_per_cell, positive_weights, torch)
        pred_classes = class_logits.permute(0, 2, 3, 1)[positive].contiguous()
        target_classes = class_target[positive].contiguous()
        cls_loss_per_cell = functional.cross_entropy(pred_classes, target_classes, reduction="none")
        cls_loss = _weighted_mean(cls_loss_per_cell, positive_weights, torch)
    else:
        box_loss = box_pred.sum() * 0.0
        cls_loss = class_logits.sum() * 0.0
    return object_loss + float(box_weight) * box_loss + float(class_weight) * cls_loss


def _weighted_mean(values: Any, weights: Any, torch: Any) -> Any:
    safe_weights = weights.to(dtype=values.dtype, device=values.device).clamp_min(0.0)
    denominator = torch.clamp(torch.ones_like(values).sum(), min=1.0e-12)
    return (values * safe_weights).sum() / denominator


def _dense_targets(
    target: Mapping[str, Any],
    class_to_index: Mapping[str, int],
    grid_size: Tuple[int, int],
    device: Any,
    torch: Any,
    *,
    box_encoding: str = DEFAULT_BOX_ENCODING,
    positive_cell_radius: int = 0,
    small_object_weight: float = 1.0,
    small_object_area_threshold: float = 32.0 * 32.0,
) -> Tuple[Any, Any, Any, Any]:
    rows, cols = int(grid_size[0]), int(grid_size[1])
    encoding = _normalize_box_encoding(box_encoding)
    radius = _normalize_positive_cell_radius(positive_cell_radius, box_encoding=encoding)
    object_target = torch.zeros((1, rows, cols), dtype=torch.float32, device=device)
    box_target = torch.zeros((4, rows, cols), dtype=torch.float32, device=device)
    class_target = torch.zeros((rows, cols), dtype=torch.long, device=device)
    positive_weight = torch.ones((rows, cols), dtype=torch.float32, device=device)
    boxes = target.get("boxes_normalized")
    absolute_boxes = target.get("boxes")
    labels = tuple(str(value) for value in target.get("labels_text", ()))
    occupied: set[Tuple[int, int]] = set()
    if boxes is None or boxes.numel() <= 0:
        return object_target, box_target, class_target, positive_weight
    for box_index, box in enumerate(boxes.detach().cpu()):
        if box_index >= len(labels):
            continue
        class_index = class_to_index.get(labels[box_index])
        if class_index is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        row = min(max(int(center_y * rows), 0), rows - 1)
        col = min(max(int(center_x * cols), 0), cols - 1)
        cells = _target_cells(row, col, rows, cols, occupied, radius=radius)
        if not cells:
            continue
        width = max(x2 - x1, 1.0e-6)
        height = max(y2 - y1, 1.0e-6)
        cell_weight = _box_positive_weight(
            absolute_boxes,
            box_index=box_index,
            small_object_weight=small_object_weight,
            small_object_area_threshold=small_object_area_threshold,
        )
        for cell_row, cell_col in cells:
            occupied.add((cell_row, cell_col))
            object_target[0, cell_row, cell_col] = 1.0
            if encoding == "xyxy":
                encoded_box = (x1, y1, x2, y2)
            else:
                offset_x = max(0.0, min(1.0, center_x * cols - cell_col))
                offset_y = max(0.0, min(1.0, center_y * rows - cell_row))
                encoded_box = (offset_x, offset_y, width, height)
            box_target[:, cell_row, cell_col] = torch.tensor(encoded_box, dtype=torch.float32, device=device)
            class_target[cell_row, cell_col] = int(class_index)
            positive_weight[cell_row, cell_col] = float(cell_weight)
    return object_target, box_target, class_target, positive_weight


def _box_positive_weight(
    absolute_boxes: Any,
    *,
    box_index: int,
    small_object_weight: float,
    small_object_area_threshold: float,
) -> float:
    multiplier = max(float(small_object_weight), 0.0)
    if multiplier <= 1.0:
        return 1.0
    if absolute_boxes is None or not hasattr(absolute_boxes, "numel") or absolute_boxes.numel() <= 0:
        return 1.0
    if int(box_index) >= int(absolute_boxes.shape[0]):
        return 1.0
    x1, y1, x2, y2 = [float(value) for value in absolute_boxes[int(box_index)].detach().cpu()]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width * height <= max(float(small_object_area_threshold), 0.0):
        return multiplier
    return 1.0


def _target_cells(
    row: int,
    col: int,
    rows: int,
    cols: int,
    occupied: set[Tuple[int, int]],
    *,
    radius: int,
) -> Tuple[Tuple[int, int], ...]:
    center = _nearest_free_cell(row, col, rows, cols, occupied)
    if center is None:
        return ()
    if int(radius) <= 0:
        return (center,)
    candidates = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            rr, cc = row + dy, col + dx
            if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in occupied:
                candidates.append((abs(dy) + abs(dx), abs(dy), abs(dx), rr, cc))
    if not candidates:
        return (center,)
    return tuple((rr, cc) for _, _, _, rr, cc in sorted(candidates))


def _nearest_free_cell(
    row: int,
    col: int,
    rows: int,
    cols: int,
    occupied: set[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if (row, col) not in occupied:
        return row, col
    max_radius = max(rows, cols)
    for radius in range(1, max_radius + 1):
        candidates = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dy), abs(dx)) != radius:
                    continue
                rr, cc = row + dy, col + dx
                if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in occupied:
                    candidates.append((abs(dy) + abs(dx), rr, cc))
        if candidates:
            _, rr, cc = min(candidates)
            return rr, cc
    return None


def _evaluate_loss(
    model: Any,
    dataset: Any,
    indices: Sequence[int],
    class_to_index: Mapping[str, int],
    device: Any,
    torch: Any,
    functional: Any,
    *,
    no_object_weight: float,
    object_loss_mode: str,
    negative_focal_gamma: float,
    box_weight: float,
    class_weight: float,
    small_object_weight: float = 1.0,
    small_object_area_threshold: float = 32.0 * 32.0,
    channel_mask: Sequence[float],
    batch_size: int = 1,
    box_encoding: str = DEFAULT_BOX_ENCODING,
    positive_cell_radius: int = 0,
) -> Optional[float]:
    if not indices:
        return None
    was_training = bool(model.training)
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_indices in _batched_indices(indices, max(int(batch_size), 1)):
            loss = _batch_loss(
                model,
                dataset,
                batch_indices,
                class_to_index,
                device,
                torch,
                functional,
                no_object_weight=no_object_weight,
                object_loss_mode=object_loss_mode,
                negative_focal_gamma=negative_focal_gamma,
                box_weight=box_weight,
                class_weight=class_weight,
                small_object_weight=small_object_weight,
                small_object_area_threshold=small_object_area_threshold,
                channel_mask=channel_mask,
                box_encoding=box_encoding,
                positive_cell_radius=positive_cell_radius,
            )
            losses.extend([float(loss.detach().cpu())] * len(batch_indices))
    if was_training:
        model.train()
    return float(sum(losses) / max(len(losses), 1))


def _batched_indices(indices: Sequence[int], batch_size: int) -> Tuple[Tuple[int, ...], ...]:
    values = tuple(int(index) for index in indices)
    size = max(int(batch_size), 1)
    return tuple(tuple(values[start : start + size]) for start in range(0, len(values), size))


def _apply_initial_checkpoint(
    model: Any,
    checkpoint_path: str | Path | None,
    torch: Any,
    *,
    input_channels: int,
    zero_aux_input_weights: bool,
) -> Dict[str, Any]:
    if checkpoint_path is None:
        return {
            "loaded": False,
            "checkpoint": None,
            "zero_aux_input_weights": bool(zero_aux_input_weights),
        }
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"initial checkpoint must be a mapping: {path}")
    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError(f"initial checkpoint does not contain model_state: {path}")
    model.load_state_dict(state, strict=True)
    zero_result = None
    if bool(zero_aux_input_weights):
        zero_result = _zero_aux_input_channel_weights(
            model,
            torch,
            input_channels=int(input_channels),
            aux_start_channel=3,
        )
    return {
        "loaded": True,
        "checkpoint": str(path),
        "checkpoint_channel_mode": checkpoint.get("channel_mode"),
        "checkpoint_tensor_key": checkpoint.get("tensor_key"),
        "checkpoint_model_architecture": checkpoint.get("model_architecture"),
        "checkpoint_input_channels": checkpoint.get("input_channels"),
        "zero_aux_input_weights": bool(zero_aux_input_weights),
        "zero_aux_input_weight_result": zero_result,
    }


def _set_training_seed(seed: int | None, torch: Any) -> Dict[str, Any]:
    if seed is None:
        return {"set": False, "seed": None}
    value = int(seed)
    random.seed(value)
    try:
        import numpy as np

        np.random.seed(value)
        numpy_seeded = True
    except Exception:
        numpy_seeded = False
    torch.manual_seed(value)
    cuda_seeded = False
    if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "manual_seed_all", None)):
        try:
            torch.cuda.manual_seed_all(value)
            cuda_seeded = True
        except Exception:
            cuda_seeded = False
    return {
        "set": True,
        "seed": value,
        "python_random": True,
        "numpy": bool(numpy_seeded),
        "torch": True,
        "torch_cuda": bool(cuda_seeded),
    }


def _zero_aux_input_channel_weights(
    model: Any,
    torch: Any,
    *,
    input_channels: int,
    aux_start_channel: int = 3,
) -> Dict[str, Any]:
    total_channels = int(input_channels)
    start_channel = int(aux_start_channel)
    if total_channels <= start_channel:
        return {
            "status": "skipped",
            "reason": "input channel count does not include aux channels",
            "input_channels": total_channels,
            "aux_start_channel": start_channel,
        }
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or int(getattr(weight, "ndim", 0)) != 4:
            continue
        if int(weight.shape[1]) != total_channels:
            continue
        with torch.no_grad():
            before_abs_sum = float(weight[:, start_channel:, :, :].detach().abs().sum().cpu())
            weight[:, start_channel:, :, :].zero_()
            after_abs_sum = float(weight[:, start_channel:, :, :].detach().abs().sum().cpu())
        return {
            "status": "zeroed",
            "module": str(name),
            "input_channels": total_channels,
            "aux_start_channel": start_channel,
            "zeroed_channel_count": int(total_channels - start_channel),
            "abs_sum_before": before_abs_sum,
            "abs_sum_after": after_abs_sum,
        }
    return {
        "status": "not_found",
        "reason": "no early-fusion convolution consumes the full input channel set",
        "input_channels": total_channels,
        "aux_start_channel": start_channel,
    }


def _clone_state_dict_cpu(model: Any) -> Dict[str, Any]:
    return {str(key): value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _class_names_for_indices(dataset: Any, indices: Sequence[int], *, allowed_labels: set[str] | None = None) -> Tuple[str, ...]:
    labels = set()
    for index in indices:
        _, target = dataset[int(index)]
        values = {str(value) for value in target.get("labels_text", ())}
        if allowed_labels is not None:
            values &= allowed_labels
        labels.update(values)
    return tuple(sorted(labels))


def _split_indices(
    dataset: Any,
    eval_fraction: float,
    *,
    strategy: str = "coverage",
    allowed_labels: set[str] | None = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    count = int(len(dataset))
    indices = tuple(range(count))
    fraction = max(float(eval_fraction), 0.0)
    if count <= 1 or fraction <= 0.0:
        return indices, ()
    eval_count = min(max(int(round(count * min(fraction, 0.9))), 1), count - 1)
    normalized_strategy = str(strategy).lower()
    if normalized_strategy == "coverage":
        return _coverage_split_indices(dataset, indices, eval_count, allowed_labels=allowed_labels)
    if normalized_strategy == "hash":
        hash_order = tuple(sorted(indices, key=_stable_index_hash))
        return _coverage_split_indices(dataset, hash_order, eval_count, allowed_labels=allowed_labels)
    return indices[:-eval_count], indices[-eval_count:]


def _stable_index_hash(index: int) -> str:
    return hashlib.sha1(str(int(index)).encode("utf-8")).hexdigest()


def _coverage_split_indices(
    dataset: Any,
    indices: Sequence[int],
    eval_count: int,
    *,
    allowed_labels: set[str] | None = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Keep eval labels covered by train whenever the manifest allows it."""

    sample_labels = {int(index): _labels_for_index(dataset, int(index), allowed_labels=allowed_labels) for index in indices}
    train = set(int(index) for index in indices)
    eval_selected: list[int] = []
    positive_candidates = tuple(int(index) for index in reversed(indices) if sample_labels[int(index)])
    background_candidates = tuple(int(index) for index in reversed(indices) if not sample_labels[int(index)])
    for candidates in (positive_candidates, background_candidates):
        for candidate in candidates:
            if len(eval_selected) >= int(eval_count):
                break
            if candidate not in train:
                continue
            labels = sample_labels[candidate]
            if labels:
                remaining_labels = set()
                for index in train:
                    if index == candidate:
                        continue
                    remaining_labels.update(sample_labels[index])
                if not labels.issubset(remaining_labels):
                    continue
            train.remove(candidate)
            eval_selected.append(candidate)
        if len(eval_selected) >= int(eval_count):
            break
    return tuple(sorted(train)), tuple(sorted(eval_selected))


def _labels_for_index(dataset: Any, index: int, *, allowed_labels: set[str] | None = None) -> set[str]:
    _, target = dataset[int(index)]
    labels = {str(value) for value in target.get("labels_text", ())}
    return labels if allowed_labels is None else labels & allowed_labels


def _time_estimates(*, sample_epochs_per_second: float, epochs: int, sample_counts: Sequence[int]) -> list[Dict[str, Any]]:
    speed = max(float(sample_epochs_per_second), 1.0e-12)
    payload = []
    for sample_count in sample_counts:
        seconds = float(int(sample_count) * int(epochs) / speed)
        payload.append(
            {
                "samples": int(sample_count),
                "epochs": int(epochs),
                "estimated_seconds": seconds,
                "estimated_minutes": seconds / 60.0,
                "estimated_hours": seconds / 3600.0,
            }
        )
    return payload


def _select_device(torch: Any, value: str) -> Any:
    requested = str(value or "auto").lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
