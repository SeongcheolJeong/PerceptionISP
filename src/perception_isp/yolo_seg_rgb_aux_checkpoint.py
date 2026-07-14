"""Create YOLO-seg checkpoints that accept RGB plus Aux input channels."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand a YOLO segmentation checkpoint from RGB to RGB+Aux.")
    parser.add_argument("--source", required=True, help="Source YOLO segmentation .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output RGB+Aux checkpoint path.")
    parser.add_argument("--channels", type=int, default=4, help="Target input channel count.")
    parser.add_argument(
        "--aux-init",
        choices=("mean_rgb", "zero"),
        default="mean_rgb",
        help="Initialization for newly added input channels.",
    )
    args = parser.parse_args(argv)
    summary = build_rgb_aux_seg_checkpoint(
        source=Path(args.source),
        output=Path(args.output),
        channels=int(args.channels),
        aux_init=str(args.aux_init),
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_rgb_aux_seg_checkpoint(
    *,
    source: Path,
    output: Path,
    channels: int = 4,
    aux_init: str = "mean_rgb",
) -> Dict[str, Any]:
    if channels < 4:
        raise ValueError("RGB+Aux checkpoint needs at least 4 input channels")
    if aux_init not in {"mean_rgb", "zero"}:
        raise ValueError(f"unsupported aux_init: {aux_init}")

    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel

    source = source.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(source))
    source_model = yolo.model.cpu().float()
    source_yaml = copy.deepcopy(source_model.yaml)
    nc = int(source_yaml.get("nc", len(getattr(source_model, "names", {})) or 80))

    target_model = SegmentationModel(cfg=source_yaml, ch=channels, nc=nc, verbose=False).cpu().float()
    transfer = _transfer_state_dict(source_model.state_dict(), target_model.state_dict(), aux_init=aux_init)
    missing, unexpected = target_model.load_state_dict(transfer["state_dict"], strict=False)
    target_model.names = copy.deepcopy(getattr(source_model, "names", {i: str(i) for i in range(nc)}))
    target_model.args = copy.deepcopy(getattr(source_model, "args", {}))
    target_model.task = "segment"

    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    ckpt["model"] = copy.deepcopy(target_model).half()
    ckpt["ema"] = None
    ckpt["optimizer"] = None
    ckpt["epoch"] = -1
    ckpt["train_args"] = {**ckpt.get("train_args", {}), "channels": channels}
    ckpt["perception_isp_rgb_aux"] = {
        "source": str(source),
        "channels": channels,
        "aux_init": aux_init,
        "expanded_keys": transfer["expanded_keys"],
        "copied_keys": transfer["copied_keys"],
        "skipped_keys": transfer["skipped_keys"],
    }
    torch.save(ckpt, output)

    # Reload once so failures surface before training starts.
    reloaded = YOLO(str(output)).model
    first_conv_shape = _first_conv_weight_shape(reloaded)
    summary = {
        "status": "pass",
        "source": str(source),
        "output": str(output),
        "channels": channels,
        "aux_init": aux_init,
        "missing_after_load": list(missing),
        "unexpected_after_load": list(unexpected),
        "copied_keys": transfer["copied_keys"],
        "expanded_keys": transfer["expanded_keys"],
        "skipped_keys": transfer["skipped_keys"],
        "first_conv_weight_shape": first_conv_shape,
    }
    (output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _transfer_state_dict(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    *,
    aux_init: str,
) -> Dict[str, Any]:
    copied_keys = []
    expanded_keys = []
    skipped_keys = []
    transferred: Dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped_keys.append({"key": key, "reason": "missing_in_source", "target_shape": list(target_tensor.shape)})
            transferred[key] = target_tensor
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            transferred[key] = source_tensor
            copied_keys.append(key)
            continue
        if _can_expand_input_conv(source_tensor, target_tensor):
            expanded = target_tensor.clone()
            old_channels = source_tensor.shape[1]
            expanded[:, :old_channels, :, :] = source_tensor
            if aux_init == "mean_rgb":
                aux_value = source_tensor.mean(dim=1, keepdim=True)
                expanded[:, old_channels:, :, :] = aux_value.expand(-1, target_tensor.shape[1] - old_channels, -1, -1)
            else:
                expanded[:, old_channels:, :, :] = 0
            transferred[key] = expanded
            expanded_keys.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                    "aux_init": aux_init,
                }
            )
            continue
        transferred[key] = target_tensor
        skipped_keys.append(
            {
                "key": key,
                "reason": "shape_mismatch",
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
        )
    return {
        "state_dict": transferred,
        "copied_keys": copied_keys,
        "expanded_keys": expanded_keys,
        "skipped_keys": skipped_keys,
    }


def _can_expand_input_conv(source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> bool:
    return (
        source_tensor.ndim == 4
        and target_tensor.ndim == 4
        and source_tensor.shape[0] == target_tensor.shape[0]
        and source_tensor.shape[2:] == target_tensor.shape[2:]
        and source_tensor.shape[1] == 3
        and target_tensor.shape[1] > source_tensor.shape[1]
    )


def _first_conv_weight_shape(model: Any) -> list[int] | None:
    for module in model.modules():
        conv = getattr(module, "conv", None)
        weight = getattr(conv, "weight", None)
        if weight is not None and weight.ndim == 4:
            return list(weight.shape)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
