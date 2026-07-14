"""Ultralytics YOLO training helpers for PerceptionISP RGB+Aux datasets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from .types import json_ready

_CANONICAL_MODULE = "perception_isp.yolo_aux_train"
if __name__ == "__main__":
    sys.modules.setdefault(_CANONICAL_MODULE, sys.modules[__name__])

try:  # Keep module importable in environments where training dependencies are absent.
    import torch.nn as _torch_nn
except Exception:  # pragma: no cover - optional dependency
    _torch_nn = None


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
    parser.add_argument("--optimizer", default=None, help="Optional Ultralytics optimizer override, e.g. AdamW.")
    parser.add_argument("--lr0", type=float, default=None, help="Optional initial learning rate override.")
    parser.add_argument("--lrf", type=float, default=None, help="Optional final LR fraction override.")
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--freeze", type=int, default=None, help="Optional Ultralytics freeze setting.")
    parser.add_argument("--fraction", type=float, default=None, help="Optional Ultralytics dataset fraction for quick smoke runs.")
    parser.add_argument("--amp", action="store_true", help="Enable Ultralytics AMP to match RGB baselines that use amp=True.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--zero-aux-input-weights", action="store_true")
    parser.add_argument(
        "--aux-input-init",
        choices=("none", "zero", "mean_rgb"),
        default="none",
        help="Optional first-conv aux channel initialization. mean_rgb copies the RGB filter mean into aux channels.",
    )
    parser.add_argument(
        "--aux-feature-adapter",
        default=None,
        help="Optional aux feature distillation checkpoint. Copies its learned conv weights into first-conv aux channels.",
    )
    parser.add_argument(
        "--aux-adapter-scale",
        type=float,
        default=1.0,
        help="Scale applied when copying aux feature adapter conv weights into the detector first conv.",
    )
    parser.add_argument("--aux-start-channel", type=int, default=3)
    parser.add_argument(
        "--aux-stem-mode",
        choices=("none", "gated_sum", "gated_norm_sum"),
        default="none",
        help=(
            "Optional RGB/Aux separated first-stem fusion. gated_sum keeps an RGB branch and adds a gated Aux branch; "
            "gated_norm_sum also normalizes the Aux branch before fusion."
        ),
    )
    parser.add_argument("--aux-stem-aux-channels", type=int, default=3)
    parser.add_argument(
        "--aux-stem-init",
        choices=("zero", "mean_rgb", "adapter"),
        default="adapter",
        help="Aux branch initialization when --aux-stem-mode is enabled.",
    )
    parser.add_argument(
        "--aux-stem-gate-init",
        type=float,
        default=-2.0,
        help="Initial gate logit for Aux branch. -2.0 starts with roughly 12% Aux contribution.",
    )
    parser.add_argument(
        "--aux-stem-freeze-rgb-branch",
        action="store_true",
        help="Freeze only the RGB branch weights inside the gated stem.",
    )
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
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        freeze=args.freeze,
        fraction=args.fraction,
        amp=bool(args.amp),
        exist_ok=bool(args.exist_ok),
        plots=bool(args.plots),
        zero_aux_input_weights=bool(args.zero_aux_input_weights),
        aux_input_init=str(args.aux_input_init),
        aux_feature_adapter=args.aux_feature_adapter,
        aux_adapter_scale=float(args.aux_adapter_scale),
        aux_start_channel=int(args.aux_start_channel),
        aux_stem_mode=str(args.aux_stem_mode),
        aux_stem_aux_channels=int(args.aux_stem_aux_channels),
        aux_stem_init=str(args.aux_stem_init),
        aux_stem_gate_init=float(args.aux_stem_gate_init),
        aux_stem_freeze_rgb_branch=bool(args.aux_stem_freeze_rgb_branch),
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
    optimizer: str | None = None,
    lr0: float | None = None,
    lrf: float | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    warmup_epochs: float | None = None,
    freeze: int | None = None,
    fraction: float | None = None,
    amp: bool = False,
    exist_ok: bool = False,
    plots: bool = False,
    zero_aux_input_weights: bool = False,
    aux_input_init: str = "none",
    aux_feature_adapter: str | None = None,
    aux_adapter_scale: float = 1.0,
    aux_start_channel: int = 3,
    aux_stem_mode: str = "none",
    aux_stem_aux_channels: int = 3,
    aux_stem_init: str = "adapter",
    aux_stem_gate_init: float = -2.0,
    aux_stem_freeze_rgb_branch: bool = False,
    disable_augment: bool = True,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is not installed; install it to run YOLO RGB+Aux training") from exc

    requested_aux_init = "zero" if zero_aux_input_weights else str(aux_input_init or "none")
    aux_init_result: Dict[str, Any] = {"enabled": requested_aux_init != "none", "mode": requested_aux_init, "status": "not_requested"}
    adapter_result: Dict[str, Any] = {
        "enabled": bool(aux_feature_adapter),
        "checkpoint": str(aux_feature_adapter) if aux_feature_adapter else None,
        "scale": float(aux_adapter_scale),
        "status": "not_requested",
    }
    stem_result: Dict[str, Any] = {
        "enabled": str(aux_stem_mode or "none") != "none",
        "mode": str(aux_stem_mode or "none"),
        "status": "not_requested",
    }
    zero_result: Dict[str, Any] = {"enabled": bool(zero_aux_input_weights), "status": "not_requested"}
    yolo = YOLO(str(model_name))
    source_gated_stem = _clone_loaded_gated_stem(yolo) if str(aux_stem_mode or "none") != "none" else None
    trainer_cls: Any | None = None
    if str(aux_stem_mode or "none") != "none":
        base_trainer_cls = yolo._smart_load("trainer")

        class PerceptionAuxStemTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
            def setup_model(self) -> Any:
                nonlocal stem_result
                ckpt = super().setup_model()
                if source_gated_stem is not None:
                    stem_result = restore_first_stem_from_gated_source(
                        self.model,
                        source_stem=source_gated_stem,
                        freeze_rgb_branch=aux_stem_freeze_rgb_branch,
                    )
                else:
                    stem_result = replace_first_stem_with_gated_aux_stem(
                        self.model,
                        aux_start_channel=aux_start_channel,
                        aux_channels=aux_stem_aux_channels,
                        init_mode=aux_stem_init,
                        adapter_checkpoint=aux_feature_adapter,
                        adapter_scale=aux_adapter_scale,
                        gate_init=aux_stem_gate_init,
                        freeze_rgb_branch=aux_stem_freeze_rgb_branch,
                        mode=str(aux_stem_mode or "gated_sum"),
                    )
                stem_result["setup_phase"] = "setup_model_before_optimizer"
                return ckpt

        trainer_cls = PerceptionAuxStemTrainer
        if aux_stem_freeze_rgb_branch:

            def _aux_stem_freeze_callback(trainer: Any) -> None:
                nonlocal stem_result
                stem_result["freeze_enforced_after_ultralytics_freeze"] = freeze_gated_stem_rgb_branch(
                    trainer.model
                )
                ema = getattr(getattr(trainer, "ema", None), "ema", None)
                if ema is not None:
                    stem_result["freeze_enforced_after_ultralytics_freeze_ema"] = freeze_gated_stem_rgb_branch(ema)

            yolo.add_callback("on_pretrain_routine_end", _aux_stem_freeze_callback)

    if str(aux_stem_mode or "none") == "none" and (requested_aux_init != "none" or aux_feature_adapter):

        def _aux_init_callback(trainer: Any) -> None:
            nonlocal adapter_result, aux_init_result, stem_result, zero_result
            ema = getattr(getattr(trainer, "ema", None), "ema", None)
            if requested_aux_init != "none":
                aux_init_result = initialize_aux_input_weights_for_model(
                    trainer.model,
                    aux_start_channel=aux_start_channel,
                    mode=requested_aux_init,
                )
            if requested_aux_init != "none" and ema is not None:
                aux_init_result["ema"] = initialize_aux_input_weights_for_model(
                    ema,
                    aux_start_channel=aux_start_channel,
                    mode=requested_aux_init,
                )
            if requested_aux_init == "zero":
                zero_result = dict(aux_init_result)
            if aux_feature_adapter:
                adapter_result = initialize_aux_input_weights_from_adapter(
                    trainer.model,
                    adapter_checkpoint=str(aux_feature_adapter),
                    aux_start_channel=aux_start_channel,
                    scale=aux_adapter_scale,
                )
                if ema is not None:
                    adapter_result["ema"] = initialize_aux_input_weights_from_adapter(
                        ema,
                        adapter_checkpoint=str(aux_feature_adapter),
                        aux_start_channel=aux_start_channel,
                        scale=aux_adapter_scale,
                    )

        yolo.add_callback("on_pretrain_routine_end", _aux_init_callback)

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
        "amp": bool(amp),
    }
    train_overrides = _optional_train_overrides(
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        freeze=freeze,
        fraction=fraction,
    )
    train_kwargs.update(train_overrides)
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
    metrics = yolo.train(trainer=trainer_cls, **train_kwargs)
    trainer = getattr(yolo, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", ""))
    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    aux_stem_state = inspect_gated_stem_state(getattr(trainer, "model", None))
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
        "amp": bool(amp),
        "train_overrides": train_overrides,
        "save_dir": str(save_dir),
        "aux_input_weight_init": aux_init_result,
        "aux_feature_adapter_init": adapter_result,
        "aux_stem": stem_result,
        "aux_stem_state_after_train": aux_stem_state,
        "zero_aux_input_weights": zero_result,
        "results_dict": results_dict,
    }
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "perception_yolo_aux_train_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


def _optional_train_overrides(
    *,
    optimizer: str | None = None,
    lr0: float | None = None,
    lrf: float | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    warmup_epochs: float | None = None,
    freeze: int | None = None,
    fraction: float | None = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if optimizer:
        overrides["optimizer"] = str(optimizer)
    if lr0 is not None:
        overrides["lr0"] = float(lr0)
    if lrf is not None:
        overrides["lrf"] = float(lrf)
    if momentum is not None:
        overrides["momentum"] = float(momentum)
    if weight_decay is not None:
        overrides["weight_decay"] = float(weight_decay)
    if warmup_epochs is not None:
        overrides["warmup_epochs"] = float(warmup_epochs)
    if freeze is not None:
        overrides["freeze"] = int(freeze)
    if fraction is not None:
        overrides["fraction"] = float(fraction)
    return overrides


class RgbAuxGatedStem(_torch_nn.Module if _torch_nn is not None else object):
    """First YOLO stem that preserves an RGB branch and adds gated Aux evidence."""

    def __init__(
        self,
        original_stem: Any,
        *,
        aux_start_channel: int,
        aux_channels: int,
        init_mode: str,
        adapter_checkpoint: str | None,
        adapter_scale: float,
        gate_init: float,
        freeze_rgb_branch: bool,
        aux_norm: bool = False,
    ) -> Any:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required for gated RGB/Aux YOLO stem") from exc

        nn.Module.__init__(self)
        original_conv = getattr(original_stem, "conv", None)
        original_bn = getattr(original_stem, "bn", None)
        original_act = getattr(original_stem, "act", None)
        if not isinstance(original_conv, nn.Conv2d):
            raise TypeError("original YOLO stem does not expose a Conv2d as .conv")
        if original_bn is None or original_act is None:
            raise TypeError("original YOLO stem must expose .bn and .act")

        start = int(aux_start_channel)
        aux_count = int(aux_channels)
        if original_conv.in_channels < start:
            raise ValueError(f"original stem has {original_conv.in_channels} channels, need at least {start} RGB channels")
        self.aux_start_channel = start
        self.aux_channels = aux_count
        self.aux_norm_enabled = bool(aux_norm)
        self.rgb_conv = nn.Conv2d(
            start,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None,
            padding_mode=original_conv.padding_mode,
        )
        self.aux_conv = nn.Conv2d(
            aux_count,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=1,
            bias=False,
            padding_mode=original_conv.padding_mode,
        )
        self.aux_bn = nn.BatchNorm2d(original_conv.out_channels) if self.aux_norm_enabled else nn.Identity()
        self.bn = copy.deepcopy(original_bn)
        self.act = copy.deepcopy(original_act)
        self.gate_logit = nn.Parameter(
            torch.full(
                (1, int(original_conv.out_channels), 1, 1),
                float(gate_init),
                dtype=original_conv.weight.dtype,
                device=original_conv.weight.device,
            )
        )
        self.conv = self.rgb_conv
        self.rgb_conv.to(device=original_conv.weight.device, dtype=original_conv.weight.dtype)
        self.aux_conv.to(device=original_conv.weight.device, dtype=original_conv.weight.dtype)
        self.aux_bn.to(device=original_conv.weight.device, dtype=original_conv.weight.dtype)
        self.bn.to(device=original_conv.weight.device, dtype=original_conv.weight.dtype)

        with torch.no_grad():
            self.rgb_conv.weight.copy_(original_conv.weight[:, :start, :, :])
            if self.rgb_conv.bias is not None and original_conv.bias is not None:
                self.rgb_conv.bias.copy_(original_conv.bias)
            _initialize_aux_branch(
                self.aux_conv,
                original_conv=original_conv,
                init_mode=init_mode,
                adapter_checkpoint=adapter_checkpoint,
                adapter_scale=adapter_scale,
            )
        if freeze_rgb_branch:
            for parameter in self.rgb_conv.parameters():
                parameter.requires_grad_(False)

        for attribute in ("i", "f", "type", "np"):
            if hasattr(original_stem, attribute):
                setattr(self, attribute, getattr(original_stem, attribute))

    def forward(self, x: Any) -> Any:
        import torch

        rgb = x[:, : self.aux_start_channel, :, :]
        aux = x[:, self.aux_start_channel : self.aux_start_channel + self.aux_channels, :, :]
        aux_features = self.aux_conv(aux)
        aux_bn = getattr(self, "aux_bn", None)
        if aux_bn is not None:
            aux_features = aux_bn(aux_features)
        fused = self.rgb_conv(rgb) + torch.sigmoid(self.gate_logit) * aux_features
        return self.act(self.bn(fused))


RgbAuxGatedStem.__module__ = _CANONICAL_MODULE


def replace_first_stem_with_gated_aux_stem(
    model: Any,
    *,
    aux_start_channel: int = 3,
    aux_channels: int = 3,
    init_mode: str = "adapter",
    adapter_checkpoint: str | None = None,
    adapter_scale: float = 1.0,
    gate_init: float = -2.0,
    freeze_rgb_branch: bool = False,
    mode: str = "gated_sum",
) -> Dict[str, Any]:
    init_mode = str(init_mode or "adapter")
    normalized_mode = str(mode or "gated_sum")
    aux_norm = normalized_mode == "gated_norm_sum"
    modules = getattr(model, "model", None)
    if modules is None:
        return {"enabled": True, "mode": normalized_mode, "status": "not_found", "reason": "model.model is missing"}
    if len(modules) < 1:
        return {"enabled": True, "mode": normalized_mode, "status": "not_found", "reason": "model.model is empty"}

    module = modules[0]
    if _is_rgb_aux_gated_stem(module):
        return {
            "enabled": True,
            "mode": _gated_stem_mode(module),
            "status": "already_initialized",
            "module_index": 0,
            "aux_start_channel": int(getattr(module, "aux_start_channel", aux_start_channel)),
            "aux_channel_count": int(getattr(module, "aux_channels", aux_channels)),
            "requested_aux_start_channel": int(aux_start_channel),
            "requested_aux_channel_count": int(aux_channels),
            "matches_requested": bool(
                int(getattr(module, "aux_start_channel", aux_start_channel)) == int(aux_start_channel)
                and int(getattr(module, "aux_channels", aux_channels)) == int(aux_channels)
            ),
            "gate_mean_sigmoid": _maybe_gate_mean_sigmoid(module),
            "freeze_rgb_branch": bool(freeze_rgb_branch),
            "aux_norm": bool(getattr(module, "aux_norm_enabled", False)),
        }

    if init_mode == "adapter" and not adapter_checkpoint:
        raise ValueError("--aux-stem-init adapter requires --aux-feature-adapter")

    conv = getattr(module, "conv", None)
    if conv is not None:
        in_channels = int(getattr(conv, "in_channels", 0))
        if in_channels >= int(aux_start_channel):
            gated_stem = RgbAuxGatedStem(
                module,
                aux_start_channel=aux_start_channel,
                aux_channels=aux_channels,
                init_mode=init_mode,
                adapter_checkpoint=adapter_checkpoint,
                adapter_scale=adapter_scale,
                gate_init=gate_init,
                freeze_rgb_branch=freeze_rgb_branch,
                aux_norm=aux_norm,
            )
            modules[0] = gated_stem
            return {
                "enabled": True,
                "mode": normalized_mode,
                "status": "initialized",
                "module_index": 0,
                "input_channels": int(in_channels),
                "expanded_from_rgb_stem": bool(in_channels < int(aux_start_channel) + int(aux_channels)),
                "aux_start_channel": int(aux_start_channel),
                "aux_channel_count": int(aux_channels),
                "init_mode": str(init_mode),
                "adapter_checkpoint": str(adapter_checkpoint) if adapter_checkpoint else None,
                "adapter_scale": float(adapter_scale),
                "gate_init_logit": float(gate_init),
                "gate_init_sigmoid": float(1.0 / (1.0 + math.exp(-float(gate_init)))),
                "freeze_rgb_branch": bool(freeze_rgb_branch),
                "aux_norm": bool(aux_norm),
            }
        return {
            "enabled": True,
            "mode": normalized_mode,
            "status": "not_found",
            "reason": "first stem has fewer RGB channels than aux_start_channel",
            "module_index": 0,
            "input_channels": int(in_channels),
            "aux_start_channel": int(aux_start_channel),
            "aux_channel_count": int(aux_channels),
        }
    return {
        "enabled": True,
        "mode": normalized_mode,
        "status": "not_found",
        "reason": "first stem does not expose .conv",
        "module_index": 0,
        "aux_start_channel": int(aux_start_channel),
        "aux_channel_count": int(aux_channels),
    }


def _clone_loaded_gated_stem(yolo: Any) -> Any | None:
    modules = getattr(getattr(yolo, "model", None), "model", None)
    if modules is None or len(modules) < 1:
        return None
    module = modules[0]
    if not _is_rgb_aux_gated_stem(module):
        return None
    return copy.deepcopy(module)


def restore_first_stem_from_gated_source(
    model: Any,
    *,
    source_stem: Any,
    freeze_rgb_branch: bool = False,
) -> Dict[str, Any]:
    modules = getattr(model, "model", None)
    if modules is None:
        return {"enabled": True, "mode": _gated_stem_mode(source_stem), "status": "not_found", "reason": "model.model is missing"}
    if len(modules) < 1:
        return {"enabled": True, "mode": _gated_stem_mode(source_stem), "status": "not_found", "reason": "model.model is empty"}
    if not _is_rgb_aux_gated_stem(source_stem):
        return {"enabled": True, "mode": "gated_sum", "status": "not_found", "reason": "source stem is not gated"}

    restored = copy.deepcopy(source_stem)
    current_conv = getattr(modules[0], "conv", None)
    weight = getattr(current_conv, "weight", None)
    if weight is None:
        weight = getattr(getattr(restored, "rgb_conv", None), "weight", None)
    if weight is not None:
        restored.to(device=weight.device, dtype=weight.dtype)
    if freeze_rgb_branch and hasattr(restored, "rgb_conv"):
        for parameter in restored.rgb_conv.parameters():
            parameter.requires_grad_(False)
    modules[0] = restored
    return {
        "enabled": True,
        "mode": _gated_stem_mode(restored),
        "status": "restored_from_checkpoint",
        "module_index": 0,
        "aux_start_channel": int(getattr(restored, "aux_start_channel", 3)),
        "aux_channel_count": int(getattr(restored, "aux_channels", 3)),
        "gate_mean_sigmoid": _maybe_gate_mean_sigmoid(restored),
        "freeze_rgb_branch": bool(freeze_rgb_branch),
        "aux_norm": bool(getattr(restored, "aux_norm_enabled", False)),
    }


def freeze_gated_stem_rgb_branch(model: Any) -> Dict[str, Any]:
    modules = getattr(model, "model", None)
    if modules is None:
        return {"status": "not_found", "reason": "model.model is missing"}
    if len(modules) < 1:
        return {"status": "not_found", "reason": "model.model is empty"}
    module = modules[0]
    if not _is_rgb_aux_gated_stem(module):
        return {"status": "not_found", "reason": "first stem is not gated"}
    rgb_conv = getattr(module, "rgb_conv", None)
    parameters = list(rgb_conv.parameters()) if rgb_conv is not None else []
    for parameter in parameters:
        parameter.requires_grad_(False)
    return {
        "status": "frozen",
        "module_index": 0,
        "parameter_count": int(len(parameters)),
        "requires_grad_after": [bool(parameter.requires_grad) for parameter in parameters],
    }


def inspect_gated_stem_state(model: Any) -> Dict[str, Any]:
    modules = getattr(model, "model", None)
    if modules is None or len(modules) < 1:
        return {"status": "not_found"}
    module = modules[0]
    if not _is_rgb_aux_gated_stem(module):
        return {"status": "not_found"}
    result: Dict[str, Any] = {
        "status": "present",
        "mode": _gated_stem_mode(module),
        "aux_start_channel": int(getattr(module, "aux_start_channel", 3)),
        "aux_channel_count": int(getattr(module, "aux_channels", 3)),
        "aux_norm": bool(getattr(module, "aux_norm_enabled", False)),
    }
    try:
        import torch

        with torch.no_grad():
            gate = module.gate_logit.detach().float()
            gate_sigmoid = torch.sigmoid(gate)
            result.update(
                {
                    "gate_logit_mean": float(gate.mean().item()),
                    "gate_sigmoid_mean": float(gate_sigmoid.mean().item()),
                    "gate_sigmoid_min": float(gate_sigmoid.min().item()),
                    "gate_sigmoid_max": float(gate_sigmoid.max().item()),
                }
            )
    except Exception:
        pass
    rgb_conv = getattr(module, "rgb_conv", None)
    if rgb_conv is not None:
        parameters = list(rgb_conv.parameters())
        result["rgb_branch_requires_grad"] = [bool(parameter.requires_grad) for parameter in parameters]
    return result


def _is_rgb_aux_gated_stem(module: Any) -> bool:
    return bool(
        hasattr(module, "rgb_conv")
        and hasattr(module, "aux_conv")
        and hasattr(module, "gate_logit")
        and hasattr(module, "aux_start_channel")
        and hasattr(module, "aux_channels")
    )


def _maybe_gate_mean_sigmoid(module: Any) -> float | None:
    try:
        import torch

        return float(torch.sigmoid(module.gate_logit).mean().detach().cpu())
    except Exception:
        return None


def _gated_stem_mode(module: Any) -> str:
    return "gated_norm_sum" if bool(getattr(module, "aux_norm_enabled", False)) else "gated_sum"


def _initialize_aux_branch(
    aux_conv: Any,
    *,
    original_conv: Any,
    init_mode: str,
    adapter_checkpoint: str | None,
    adapter_scale: float,
) -> None:
    import torch

    mode = str(init_mode or "zero").lower()
    if mode == "zero":
        aux_conv.weight.zero_()
        return
    if mode == "mean_rgb":
        rgb_mean = original_conv.weight[:, :3, :, :].mean(dim=1, keepdim=True)
        aux_conv.weight.copy_(rgb_mean.repeat(1, int(aux_conv.in_channels), 1, 1))
        return
    if mode == "adapter":
        checkpoint_path = Path(str(adapter_checkpoint))
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("adapter_state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        adapter_weight = state.get("0.weight") if isinstance(state, Mapping) else None
        if adapter_weight is None and isinstance(state, Mapping):
            adapter_weight = state.get("conv.weight")
        if adapter_weight is None:
            raise KeyError(f"{checkpoint_path} does not contain adapter conv weights")
        if tuple(adapter_weight.shape) != tuple(aux_conv.weight.shape):
            raise ValueError(
                f"adapter weight shape {tuple(adapter_weight.shape)} does not match aux branch {tuple(aux_conv.weight.shape)}"
            )
        aux_conv.weight.copy_(adapter_weight.to(device=aux_conv.weight.device, dtype=aux_conv.weight.dtype) * float(adapter_scale))
        return
    raise ValueError(f"unsupported aux stem init mode: {init_mode}")


def initialize_aux_input_weights_for_model(model: Any, *, aux_start_channel: int = 3, mode: str = "zero") -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to initialize YOLO aux input weights") from exc

    init_mode = str(mode or "zero").lower()
    if init_mode not in {"zero", "mean_rgb"}:
        raise ValueError(f"unsupported aux input init mode: {mode}")
    start = max(int(aux_start_channel), 0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and int(module.in_channels) > start:
            with torch.no_grad():
                aux_weight = module.weight[:, start:, :, :]
                before = float(aux_weight.detach().abs().sum().cpu())
                if init_mode == "zero":
                    aux_weight.zero_()
                elif init_mode == "mean_rgb":
                    rgb_weight = module.weight[:, :start, :, :]
                    rgb_mean = rgb_weight.mean(dim=1, keepdim=True)
                    aux_weight.copy_(rgb_mean.repeat(1, int(module.in_channels) - start, 1, 1))
                after = float(aux_weight.detach().abs().sum().cpu())
                rgb_after = float(module.weight[:, :start, :, :].detach().abs().sum().cpu()) if start > 0 else 0.0
            return {
                "enabled": True,
                "mode": init_mode,
                "status": "zeroed" if init_mode == "zero" else "initialized",
                "module": str(name),
                "input_channels": int(module.in_channels),
                "aux_start_channel": int(start),
                "aux_channel_count": int(module.in_channels - start),
                "zeroed_channel_count": int(module.in_channels - start) if init_mode == "zero" else 0,
                "abs_sum_before": before,
                "abs_sum_after": after,
                "rgb_abs_sum_after": rgb_after,
            }
    return {
        "enabled": True,
        "status": "not_found",
        "reason": "no Conv2d with input channels beyond aux_start_channel",
        "aux_start_channel": int(start),
    }


def initialize_aux_input_weights_from_adapter(
    model: Any,
    *,
    adapter_checkpoint: str,
    aux_start_channel: int = 3,
    scale: float = 1.0,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to initialize YOLO aux input weights from an adapter") from exc

    checkpoint_path = Path(adapter_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"aux feature adapter checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("adapter_state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping):
        raise ValueError(f"unsupported aux feature adapter checkpoint format: {checkpoint_path}")
    adapter_weight = state.get("0.weight")
    if adapter_weight is None:
        adapter_weight = state.get("conv.weight")
    if adapter_weight is None:
        raise KeyError(f"{checkpoint_path} does not contain adapter conv weights under '0.weight' or 'conv.weight'")
    if adapter_weight.ndim != 4:
        raise ValueError(f"adapter conv weight must be OIHW, got shape {tuple(adapter_weight.shape)}")

    start = max(int(aux_start_channel), 0)
    aux_channels = int(adapter_weight.shape[1])
    scaled_weight = adapter_weight.detach() * float(scale)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if int(module.in_channels) < start + aux_channels:
            continue
        if int(module.out_channels) != int(adapter_weight.shape[0]):
            continue
        if tuple(module.weight.shape[2:]) != tuple(adapter_weight.shape[2:]):
            continue
        with torch.no_grad():
            target = module.weight[:, start : start + aux_channels, :, :]
            before = float(target.detach().abs().sum().cpu())
            target.copy_(scaled_weight.to(device=target.device, dtype=target.dtype))
            after = float(target.detach().abs().sum().cpu())
            rgb_after = float(module.weight[:, :start, :, :].detach().abs().sum().cpu()) if start > 0 else 0.0
        return {
            "enabled": True,
            "mode": "feature_adapter",
            "status": "initialized",
            "checkpoint": str(checkpoint_path),
            "scale": float(scale),
            "module": str(name),
            "input_channels": int(module.in_channels),
            "aux_start_channel": int(start),
            "aux_channel_count": int(aux_channels),
            "adapter_weight_shape": [int(value) for value in adapter_weight.shape],
            "abs_sum_before": before,
            "abs_sum_after": after,
            "rgb_abs_sum_after": rgb_after,
        }
    return {
        "enabled": True,
        "mode": "feature_adapter",
        "status": "not_found",
        "checkpoint": str(checkpoint_path),
        "scale": float(scale),
        "reason": "no compatible Conv2d found for adapter weight shape",
        "adapter_weight_shape": [int(value) for value in adapter_weight.shape],
        "aux_start_channel": int(start),
    }


def zero_aux_input_weights_for_model(model: Any, *, aux_start_channel: int = 3) -> Dict[str, Any]:
    return initialize_aux_input_weights_for_model(model, aux_start_channel=aux_start_channel, mode="zero")


if __name__ == "__main__":
    raise SystemExit(main())
