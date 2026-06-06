"""Tiny RGB+aux training smoke test.

This is not a detector training recipe. It proves that exported PerceptionISP
RGB+aux tensors can be consumed by a PyTorch model and optimized end to end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .aux_dnn import RGB_AUX_TENSOR_KEY, channels_for_tensor_key, chw_tensor_key, make_aux_smoke_detector_model, make_torch_dataset
from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny PyTorch smoke training loop on exported RGB+aux tensors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--tensor-key", default=RGB_AUX_TENSOR_KEY, help="Tensor key to train on: rgb_aux_chw or rgb_aux_extended_chw.")
    parser.add_argument("--estimate-samples", default="10,100,1000,10000")
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    parser.add_argument("--output-dir", default="exports/perception_rgb_aux_train_smoke")
    args = parser.parse_args(argv)

    summary = train_smoke(
        manifest_path=args.manifest,
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        device_name=str(args.device),
        tensor_key=str(args.tensor_key),
        estimate_samples=parse_estimate_samples(args.estimate_samples),
        eval_fraction=float(args.eval_fraction),
        output_dir=args.output_dir,
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def train_smoke(
    *,
    manifest_path: str | Path,
    epochs: int = 2,
    learning_rate: float = 1.0e-3,
    device_name: str = "auto",
    tensor_key: str = RGB_AUX_TENSOR_KEY,
    estimate_samples: Sequence[int] = (10, 100, 1000, 10000),
    eval_fraction: float = 0.0,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    resolved_tensor_key = chw_tensor_key(tensor_key)
    channels = channels_for_tensor_key(resolved_tensor_key)
    dataset = make_torch_dataset(manifest_path, tensor_key=resolved_tensor_key)
    if len(dataset) <= 0:
        raise ValueError("manifest contains no samples")
    device = _select_device(torch, device_name)
    train_indices, eval_indices = _split_indices(len(dataset), eval_fraction)

    model = make_aux_smoke_detector_model(stem_channels=16, in_channels=len(channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    history = []
    first_loss = None
    last_loss = None
    epoch_count = max(int(epochs), 1)
    total_steps = 0
    start_time = time.perf_counter()
    initial_eval_loss = _evaluate_loss(model, dataset, eval_indices, device, torch, F)
    for epoch in range(epoch_count):
        epoch_loss = 0.0
        for index in train_indices:
            loss = _sample_loss(model, dataset, index, device, torch, F)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            epoch_loss += value
            total_steps += 1
        train_mean = float(epoch_loss / max(len(train_indices), 1))
        eval_mean = _evaluate_loss(model, dataset, eval_indices, device, torch, F)
        history.append(
            {
                "epoch": int(epoch),
                "mean_loss": train_mean,
                "train_loss": train_mean,
                "eval_loss": eval_mean,
            }
        )
    elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-9)
    sample_epochs_per_second = float(total_steps / elapsed_seconds)

    summary = {
        "manifest": str(manifest_path),
        "sample_count": int(len(dataset)),
        "train_sample_count": int(len(train_indices)),
        "eval_sample_count": int(len(eval_indices)),
        "device": str(device),
        "tensor_key": resolved_tensor_key,
        "input_channels": len(channels),
        "channels": list(channels),
        "epochs": int(epoch_count),
        "learning_rate": float(learning_rate),
        "first_loss": float(first_loss if first_loss is not None else 0.0),
        "last_loss": float(last_loss if last_loss is not None else 0.0),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": history[-1]["eval_loss"] if history else initial_eval_loss,
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
        "purpose": "smoke test only; not a detector performance claim",
    }
    if output_dir is not None:
        destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "train_smoke_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
        checkpoint = {
            "model_type": "rgb_aux_smoke_detector_v1",
            "model_state": model.state_dict(),
            "channels": list(channels),
            "tensor_key": resolved_tensor_key,
            "input_channels": len(channels),
            "stem_channels": 16,
            "summary": summary,
        }
        torch.save(checkpoint, destination / "rgb_aux_smoke_detector.pt")
        summary["checkpoint"] = str(destination / "rgb_aux_smoke_detector.pt")
        (destination / "train_smoke_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


def parse_estimate_samples(value: str) -> tuple[int, ...]:
    samples = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if count > 0:
            samples.append(count)
    return tuple(samples) or (10, 100, 1000)


def _split_indices(sample_count: int, eval_fraction: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = int(sample_count)
    indices = tuple(range(count))
    fraction = max(float(eval_fraction), 0.0)
    if count <= 1 or fraction <= 0.0:
        return indices, ()
    eval_count = min(max(int(round(count * min(fraction, 0.9))), 1), count - 1)
    return indices[:-eval_count], indices[-eval_count:]


def _sample_loss(model: Any, dataset: Any, index: int, device: Any, torch: Any, functional: Any) -> Any:
    tensor, target = dataset[int(index)]
    x = tensor.unsqueeze(0).to(device)
    boxes = target["boxes_normalized"].to(device)
    objectness = torch.ones((1,), dtype=torch.float32, device=device) if boxes.numel() > 0 else torch.zeros((1,), dtype=torch.float32, device=device)
    box_target = boxes[0].view(1, 4) if boxes.numel() > 0 else torch.zeros((1, 4), dtype=torch.float32, device=device)
    pred = model(x)
    object_loss = functional.binary_cross_entropy_with_logits(pred[:, 0], objectness)
    box_loss = functional.mse_loss(torch.sigmoid(pred[:, 1:5]), box_target) if boxes.numel() > 0 else pred[:, 1:5].abs().mean() * 0.0
    return object_loss + box_loss


def _evaluate_loss(model: Any, dataset: Any, indices: Sequence[int], device: Any, torch: Any, functional: Any) -> Optional[float]:
    if not indices:
        return None
    was_training = bool(model.training)
    model.eval()
    losses = []
    with torch.no_grad():
        for index in indices:
            losses.append(float(_sample_loss(model, dataset, int(index), device, torch, functional).detach().cpu()))
    if was_training:
        model.train()
    return float(sum(losses) / max(len(losses), 1))


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
