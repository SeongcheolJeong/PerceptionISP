"""Ultralytics YOLO training helpers for PerceptionISP RGB+Aux datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO on PerceptionISP RGB+Aux NPY datasets.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="outputs/yolo_aux_train")
    parser.add_argument("--name", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--zero-aux-input-weights", action="store_true")
    parser.add_argument("--aux-start-channel", type=int, default=3)
    parser.add_argument("--disable-augment", action="store_true", help="Disable color/geometric augmentations for multi-channel smoke runs.")
    args = parser.parse_args(argv)

    summary = train_yolo_aux(
        data=str(args.data),
        model_name=str(args.model),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        seed=int(args.seed),
        exist_ok=bool(args.exist_ok),
        plots=bool(args.plots),
        zero_aux_input_weights=bool(args.zero_aux_input_weights),
        aux_start_channel=int(args.aux_start_channel),
        disable_augment=bool(args.disable_augment),
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def train_yolo_aux(
    *,
    data: str,
    model_name: str = "yolo11n.pt",
    epochs: int = 1,
    imgsz: int = 512,
    batch: int = 4,
    device: str = "cpu",
    workers: int = 0,
    project: str = "outputs/yolo_aux_train",
    name: str = "train",
    seed: int = 0,
    exist_ok: bool = False,
    plots: bool = False,
    zero_aux_input_weights: bool = False,
    aux_start_channel: int = 3,
    disable_augment: bool = True,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is not installed; install it to run YOLO RGB+Aux training") from exc

    zero_result: Dict[str, Any] = {"enabled": bool(zero_aux_input_weights), "status": "not_requested"}
    yolo = YOLO(str(model_name))
    if zero_aux_input_weights:

        def _zero_aux_callback(trainer: Any) -> None:
            nonlocal zero_result
            zero_result = zero_aux_input_weights_for_model(trainer.model, aux_start_channel=aux_start_channel)
            ema = getattr(getattr(trainer, "ema", None), "ema", None)
            if ema is not None:
                zero_result["ema"] = zero_aux_input_weights_for_model(ema, aux_start_channel=aux_start_channel)

        yolo.add_callback("on_pretrain_routine_end", _zero_aux_callback)

    train_kwargs: Dict[str, Any] = {
        "data": str(data),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": str(device),
        "workers": int(workers),
        "project": str(project),
        "name": str(name),
        "seed": int(seed),
        "exist_ok": bool(exist_ok),
        "plots": bool(plots),
        "cache": False,
        "amp": False,
    }
    if disable_augment:
        train_kwargs.update(
            {
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
            }
        )
    metrics = yolo.train(**train_kwargs)
    save_dir = Path(getattr(getattr(yolo, "trainer", None), "save_dir", ""))
    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    summary = {
        "data": str(data),
        "model": str(model_name),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": str(device),
        "project": str(project),
        "name": str(name),
        "seed": int(seed),
        "save_dir": str(save_dir),
        "zero_aux_input_weights": zero_result,
        "results_dict": results_dict,
    }
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "perception_yolo_aux_train_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


def zero_aux_input_weights_for_model(model: Any, *, aux_start_channel: int = 3) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to zero YOLO aux input weights") from exc

    start = max(int(aux_start_channel), 0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and int(module.in_channels) > start:
            with torch.no_grad():
                before = float(module.weight[:, start:, :, :].detach().abs().sum().cpu())
                module.weight[:, start:, :, :].zero_()
                after = float(module.weight[:, start:, :, :].detach().abs().sum().cpu())
            return {
                "enabled": True,
                "status": "zeroed",
                "module": str(name),
                "input_channels": int(module.in_channels),
                "aux_start_channel": int(start),
                "zeroed_channel_count": int(module.in_channels - start),
                "abs_sum_before": before,
                "abs_sum_after": after,
            }
    return {
        "enabled": True,
        "status": "not_found",
        "reason": "no Conv2d with input channels beyond aux_start_channel",
        "aux_start_channel": int(start),
    }


if __name__ == "__main__":
    raise SystemExit(main())
