"""Tiny RGB+aux training smoke test.

This is not a detector training recipe. It proves that exported PerceptionISP
RGB+aux tensors can be consumed by a PyTorch model and optimized end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .aux_dnn import make_aux_early_fusion_stem, make_torch_dataset
from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny PyTorch smoke training loop on exported RGB+aux tensors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", default="exports/perception_rgb_aux_train_smoke")
    args = parser.parse_args(argv)

    summary = train_smoke(
        manifest_path=args.manifest,
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        device_name=str(args.device),
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
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dataset = make_torch_dataset(manifest_path)
    if len(dataset) <= 0:
        raise ValueError("manifest contains no samples")
    device = _select_device(torch, device_name)

    model = nn.Sequential(
        make_aux_early_fusion_stem(in_channels=6, out_channels=16),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 5),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    history = []
    first_loss = None
    last_loss = None
    for epoch in range(max(int(epochs), 1)):
        epoch_loss = 0.0
        for index in range(len(dataset)):
            tensor, target = dataset[index]
            x = tensor.unsqueeze(0).to(device)
            boxes = target["boxes_normalized"].to(device)
            objectness = torch.ones((1,), dtype=torch.float32, device=device) if boxes.numel() > 0 else torch.zeros((1,), dtype=torch.float32, device=device)
            box_target = boxes[0].view(1, 4) if boxes.numel() > 0 else torch.zeros((1, 4), dtype=torch.float32, device=device)
            pred = model(x)
            object_loss = F.binary_cross_entropy_with_logits(pred[:, 0], objectness)
            box_loss = F.mse_loss(torch.sigmoid(pred[:, 1:5]), box_target) if boxes.numel() > 0 else pred[:, 1:5].abs().mean() * 0.0
            loss = object_loss + box_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            epoch_loss += value
        history.append({"epoch": int(epoch), "mean_loss": float(epoch_loss / max(len(dataset), 1))})

    summary = {
        "manifest": str(manifest_path),
        "sample_count": int(len(dataset)),
        "device": str(device),
        "epochs": int(max(int(epochs), 1)),
        "learning_rate": float(learning_rate),
        "first_loss": float(first_loss if first_loss is not None else 0.0),
        "last_loss": float(last_loss if last_loss is not None else 0.0),
        "history": history,
        "purpose": "smoke test only; not a detector performance claim",
    }
    if output_dir is not None:
        destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "train_smoke_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


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
