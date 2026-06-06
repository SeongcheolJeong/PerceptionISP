"""Compact class-aware RGB+aux detector training.

This is a practical reference path for learning from PerceptionISP auxiliary
maps without depending on a full YOLO training fork. It predicts objectness,
box coordinates, and class labels on a fixed grid over the RGB+aux tensor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .aux_dnn import RGB_AUX_CHANNELS, labels_from_manifest, make_aux_dense_detector_model, make_torch_dataset
from .types import json_ready


BOX_ENCODING = "cell_center_size"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train a compact class-aware detector on PerceptionISP RGB+aux tensors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--grid", default="15x20", help="Detector grid as HxW.")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--split-strategy", default="coverage", choices=["coverage", "sequential"])
    parser.add_argument("--no-object-weight", type=float, default=0.15)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--estimate-samples", default="100,1000,10000")
    parser.add_argument("--output-dir", default="exports/perception_rgb_aux_train_dense")
    args = parser.parse_args(argv)

    summary = train_dense(
        manifest_path=args.manifest,
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        device_name=str(args.device),
        grid_size=parse_grid_size(args.grid),
        base_channels=int(args.base_channels),
        eval_fraction=float(args.eval_fraction),
        split_strategy=str(args.split_strategy),
        no_object_weight=float(args.no_object_weight),
        box_weight=float(args.box_weight),
        class_weight=float(args.class_weight),
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
    grid_size: Tuple[int, int] = (15, 20),
    base_channels: int = 24,
    eval_fraction: float = 0.25,
    split_strategy: str = "coverage",
    no_object_weight: float = 0.15,
    box_weight: float = 5.0,
    class_weight: float = 1.0,
    estimate_samples: Sequence[int] = (100, 1000, 10000),
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    dataset = make_torch_dataset(manifest_path)
    if len(dataset) <= 0:
        raise ValueError("manifest contains no samples")
    class_names = labels_from_manifest(manifest_path)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    device = _select_device(torch, device_name)
    train_indices, eval_indices = _split_indices(dataset, eval_fraction, strategy=split_strategy)
    train_class_names = _class_names_for_indices(dataset, train_indices)
    eval_class_names = _class_names_for_indices(dataset, eval_indices)
    missing_eval_class_names = tuple(sorted(set(eval_class_names) - set(train_class_names)))

    model = make_aux_dense_detector_model(
        num_classes=len(class_names),
        grid_size=grid_size,
        base_channels=int(base_channels),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=5.0e-4)
    history = []
    first_loss = None
    last_loss = None
    total_steps = 0
    best_state = None
    best_epoch = -1
    best_loss = None
    best_loss_kind = "eval_loss" if eval_indices else "train_loss"
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
        box_weight=box_weight,
        class_weight=class_weight,
    )
    for epoch in range(epoch_count):
        model.train()
        epoch_loss = 0.0
        for index in train_indices:
            loss = _sample_loss(
                model,
                dataset,
                int(index),
                class_to_index,
                device,
                torch,
                F,
                no_object_weight=no_object_weight,
                box_weight=box_weight,
                class_weight=class_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            epoch_loss += value
            total_steps += 1
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
            box_weight=box_weight,
            class_weight=class_weight,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_mean,
                "eval_loss": eval_mean,
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
        "grid_size": [int(grid_size[0]), int(grid_size[1])],
        "base_channels": int(base_channels),
        "split_strategy": str(split_strategy),
        "eval_fraction": float(eval_fraction),
        "train_indices": [int(index) for index in train_indices],
        "eval_indices": [int(index) for index in eval_indices],
        "class_names": list(class_names),
        "class_count": int(len(class_names)),
        "train_class_names": list(train_class_names),
        "eval_class_names": list(eval_class_names),
        "missing_eval_class_names": list(missing_eval_class_names),
        "no_object_weight": float(no_object_weight),
        "box_weight": float(box_weight),
        "class_weight": float(class_weight),
        "first_loss": float(first_loss if first_loss is not None else 0.0),
        "last_loss": float(last_loss if last_loss is not None else 0.0),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": history[-1]["eval_loss"] if history else initial_eval_loss,
        "checkpoint_epoch": int(best_epoch),
        "checkpoint_loss": best_loss,
        "checkpoint_loss_kind": best_loss_kind,
        "box_encoding": BOX_ENCODING,
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
        checkpoint = {
            "model_type": "rgb_aux_dense_detector_v1",
            "model_state": best_state if best_state is not None else _clone_state_dict_cpu(model),
            "channels": list(RGB_AUX_CHANNELS),
            "grid_size": [int(grid_size[0]), int(grid_size[1])],
            "base_channels": int(base_channels),
            "class_names": list(class_names),
            "box_encoding": BOX_ENCODING,
            "summary": summary,
        }
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
    box_weight: float,
    class_weight: float,
) -> Any:
    tensor, target = dataset[int(index)]
    x = tensor.unsqueeze(0).to(device)
    pred = model(x)
    grid_size = (int(pred.shape[2]), int(pred.shape[3]))
    object_target, box_target, class_target = _dense_targets(
        target,
        class_to_index,
        grid_size,
        device,
        torch,
    )
    object_logits = pred[:, 0, :, :]
    box_pred = torch.sigmoid(pred[:, 1:5, :, :])
    class_logits = pred[:, 5:, :, :]

    object_loss_raw = functional.binary_cross_entropy_with_logits(object_logits, object_target, reduction="none")
    object_weights = torch.where(
        object_target > 0.5,
        torch.ones_like(object_target),
        torch.full_like(object_target, float(no_object_weight)),
    )
    object_loss = (object_loss_raw * object_weights).mean()

    positive = object_target[0] > 0.5
    if bool(torch.any(positive)):
        pred_boxes = box_pred[0, :, positive].transpose(0, 1).contiguous()
        target_boxes = box_target[:, positive].transpose(0, 1).contiguous()
        box_loss = functional.smooth_l1_loss(pred_boxes, target_boxes)
        pred_classes = class_logits[0, :, positive].transpose(0, 1).contiguous()
        target_classes = class_target[positive].contiguous()
        cls_loss = functional.cross_entropy(pred_classes, target_classes)
    else:
        box_loss = box_pred.sum() * 0.0
        cls_loss = class_logits.sum() * 0.0
    return object_loss + float(box_weight) * box_loss + float(class_weight) * cls_loss


def _dense_targets(
    target: Mapping[str, Any],
    class_to_index: Mapping[str, int],
    grid_size: Tuple[int, int],
    device: Any,
    torch: Any,
) -> Tuple[Any, Any, Any]:
    rows, cols = int(grid_size[0]), int(grid_size[1])
    object_target = torch.zeros((1, rows, cols), dtype=torch.float32, device=device)
    box_target = torch.zeros((4, rows, cols), dtype=torch.float32, device=device)
    class_target = torch.zeros((rows, cols), dtype=torch.long, device=device)
    boxes = target.get("boxes_normalized")
    labels = tuple(str(value) for value in target.get("labels_text", ()))
    occupied: set[Tuple[int, int]] = set()
    if boxes is None or boxes.numel() <= 0:
        return object_target, box_target, class_target
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
        cell = _nearest_free_cell(row, col, rows, cols, occupied)
        if cell is None:
            continue
        row, col = cell
        occupied.add(cell)
        object_target[0, row, col] = 1.0
        width = max(x2 - x1, 1.0e-6)
        height = max(y2 - y1, 1.0e-6)
        offset_x = max(0.0, min(1.0, center_x * cols - col))
        offset_y = max(0.0, min(1.0, center_y * rows - row))
        box_target[:, row, col] = torch.tensor((offset_x, offset_y, width, height), dtype=torch.float32, device=device)
        class_target[row, col] = int(class_index)
    return object_target, box_target, class_target


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
    box_weight: float,
    class_weight: float,
) -> Optional[float]:
    if not indices:
        return None
    was_training = bool(model.training)
    model.eval()
    losses = []
    with torch.no_grad():
        for index in indices:
            losses.append(
                float(
                    _sample_loss(
                        model,
                        dataset,
                        int(index),
                        class_to_index,
                        device,
                        torch,
                        functional,
                        no_object_weight=no_object_weight,
                        box_weight=box_weight,
                        class_weight=class_weight,
                    )
                    .detach()
                    .cpu()
                )
            )
    if was_training:
        model.train()
    return float(sum(losses) / max(len(losses), 1))


def _clone_state_dict_cpu(model: Any) -> Dict[str, Any]:
    return {str(key): value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _class_names_for_indices(dataset: Any, indices: Sequence[int]) -> Tuple[str, ...]:
    labels = set()
    for index in indices:
        _, target = dataset[int(index)]
        labels.update(str(value) for value in target.get("labels_text", ()))
    return tuple(sorted(labels))


def _split_indices(dataset: Any, eval_fraction: float, *, strategy: str = "coverage") -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    count = int(len(dataset))
    indices = tuple(range(count))
    fraction = max(float(eval_fraction), 0.0)
    if count <= 1 or fraction <= 0.0:
        return indices, ()
    eval_count = min(max(int(round(count * min(fraction, 0.9))), 1), count - 1)
    if str(strategy).lower() == "coverage":
        return _coverage_split_indices(dataset, indices, eval_count)
    return indices[:-eval_count], indices[-eval_count:]


def _coverage_split_indices(dataset: Any, indices: Sequence[int], eval_count: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Keep eval labels covered by train whenever the manifest allows it."""

    sample_labels = {int(index): _labels_for_index(dataset, int(index)) for index in indices}
    train = set(int(index) for index in indices)
    eval_selected: list[int] = []
    for candidate in reversed(indices):
        if len(eval_selected) >= int(eval_count):
            break
        candidate = int(candidate)
        labels = sample_labels[candidate]
        if not labels:
            train.remove(candidate)
            eval_selected.append(candidate)
            continue
        remaining_labels = set()
        for index in train:
            if index == candidate:
                continue
            remaining_labels.update(sample_labels[index])
        if labels.issubset(remaining_labels):
            train.remove(candidate)
            eval_selected.append(candidate)
    return tuple(sorted(train)), tuple(sorted(eval_selected))


def _labels_for_index(dataset: Any, index: int) -> set[str]:
    _, target = dataset[int(index)]
    return {str(value) for value in target.get("labels_text", ())}


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
