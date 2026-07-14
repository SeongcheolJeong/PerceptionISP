"""Train an aux-only adapter to mimic an RGB YOLO early feature map."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from perception_isp.core.types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Distill RGB YOLO early features into an aux-only adapter.")
    parser.add_argument("--data", required=True, help="RGB+Aux YOLO data.yaml containing NPY images.")
    parser.add_argument("--teacher", required=True, help="RGB-only YOLO checkpoint used as the frozen teacher.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--aux-start-channel", type=int, default=3)
    parser.add_argument("--aux-channels", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--project", default="outputs/yolo_aux_feature_distill")
    parser.add_argument("--name", default="aux_first_feature")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args(argv)

    summary = train_aux_feature_distill(
        data=str(args.data),
        teacher=str(args.teacher),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        aux_start_channel=int(args.aux_start_channel),
        aux_channels=int(args.aux_channels),
        max_train=args.max_train,
        max_val=args.max_val,
        project=str(args.project),
        name=str(args.name),
        seed=int(args.seed),
        exist_ok=bool(args.exist_ok),
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def train_aux_feature_distill(
    *,
    data: str,
    teacher: str,
    epochs: int = 3,
    batch: int = 16,
    device: str = "auto",
    workers: int = 0,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    aux_start_channel: int = 3,
    aux_channels: int = 3,
    max_train: int | None = None,
    max_val: int | None = None,
    project: str = "outputs/yolo_aux_feature_distill",
    name: str = "aux_first_feature",
    seed: int = 0,
    exist_ok: bool = False,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and ultralytics are required for feature distillation") from exc

    _seed_everything(seed)
    resolved_device = _resolve_device(device, torch)
    root, config = _load_data_config(Path(data))
    train_set = _RgbAuxNpyDataset(
        root=root,
        split=str(config.get("train", "images/train")),
        aux_start_channel=aux_start_channel,
        aux_channels=aux_channels,
        max_samples=max_train,
        dataset_cls=Dataset,
    )
    val_set = _RgbAuxNpyDataset(
        root=root,
        split=str(config.get("val", "images/val")),
        aux_start_channel=aux_start_channel,
        aux_channels=aux_channels,
        max_samples=max_val,
        dataset_cls=Dataset,
    )
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=workers)

    teacher_model = YOLO(str(teacher)).model.to(resolved_device).eval()
    teacher_first = teacher_model.model[0].eval()
    for parameter in teacher_first.parameters():
        parameter.requires_grad_(False)

    teacher_conv = teacher_first.conv
    teacher_bn = teacher_first.bn
    adapter = AuxFirstFeatureAdapter(
        aux_channels=aux_channels,
        output_channels=int(teacher_conv.out_channels),
        stride=int(teacher_conv.stride[0]),
        eps=float(teacher_bn.eps),
        momentum=float(teacher_bn.momentum),
    ).to(resolved_device)
    _initialize_adapter_from_teacher_mean(adapter, teacher_first, aux_channels=aux_channels)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(
            adapter=adapter,
            teacher_first=teacher_first,
            loader=train_loader,
            device=resolved_device,
            torch=torch,
            F=F,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            adapter=adapter,
            teacher_first=teacher_first,
            loader=val_loader,
            device=resolved_device,
            torch=torch,
            F=F,
            optimizer=None,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['loss']:.6f} train_cos={train_metrics['cosine']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_cos={val_metrics['cosine']:.6f}",
            flush=True,
        )

    save_dir = Path(project) / name
    if save_dir.exists() and not exist_ok:
        raise FileExistsError(f"{save_dir} exists; pass --exist-ok to overwrite summary/checkpoint")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "aux_first_feature_adapter.pt"
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "teacher": str(teacher),
            "data": str(data),
            "aux_start_channel": int(aux_start_channel),
            "aux_channels": int(aux_channels),
            "history": history,
        },
        checkpoint_path,
    )
    summary = {
        "data": str(data),
        "teacher": str(teacher),
        "epochs": int(epochs),
        "batch": int(batch),
        "device": str(resolved_device),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "aux_start_channel": int(aux_start_channel),
        "aux_channels": int(aux_channels),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "save_dir": str(save_dir),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "final": history[-1] if history else {},
    }
    (save_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


class AuxFirstFeatureAdapter:
    def __new__(cls, *, aux_channels: int, output_channels: int, stride: int, eps: float, momentum: float) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(int(aux_channels), int(output_channels), kernel_size=3, stride=int(stride), padding=1, bias=False),
            nn.BatchNorm2d(int(output_channels), eps=float(eps), momentum=float(momentum), affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
        )


class _RgbAuxNpyDataset:
    def __new__(
        cls,
        *,
        root: Path,
        split: str,
        aux_start_channel: int,
        aux_channels: int,
        max_samples: int | None,
        dataset_cls: Any,
    ) -> Any:
        class DatasetImpl(dataset_cls):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                image_dir = root / split
                self.paths = sorted(image_dir.glob("*.npy"))
                if max_samples is not None:
                    self.paths = self.paths[: int(max_samples)]
                if not self.paths:
                    raise FileNotFoundError(f"no .npy images found under {image_dir}")
                self.aux_start_channel = int(aux_start_channel)
                self.aux_channels = int(aux_channels)

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, index: int) -> Dict[str, Any]:
                import torch

                path = self.paths[index]
                array = np.load(path, allow_pickle=False)
                if array.ndim != 3:
                    raise ValueError(f"expected HWC or CHW tensor in {path}, got shape {array.shape}")
                if array.shape[0] in {3, 4, 5, 6, 15} and array.shape[-1] not in {3, 4, 5, 6, 15}:
                    array = np.moveaxis(array, 0, -1)
                channel_count = array.shape[-1]
                aux_end = self.aux_start_channel + self.aux_channels
                if channel_count < aux_end:
                    raise ValueError(f"{path} has {channel_count} channels, need at least {aux_end}")
                tensor = torch.from_numpy(np.ascontiguousarray(array)).float().permute(2, 0, 1) / 255.0
                return {
                    "rgb": tensor[:3],
                    "aux": tensor[self.aux_start_channel : aux_end],
                    "path": str(path),
                }

        return DatasetImpl()


def _run_epoch(
    *,
    adapter: Any,
    teacher_first: Any,
    loader: Iterable[Mapping[str, Any]],
    device: str,
    torch: Any,
    F: Any,
    optimizer: Any | None,
) -> Dict[str, float]:
    training = optimizer is not None
    adapter.train(training)
    totals = {"loss": 0.0, "smooth_l1": 0.0, "mse": 0.0, "cosine": 0.0, "count": 0.0}
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=False)
        aux = batch["aux"].to(device, non_blocking=False)
        with torch.no_grad():
            target = teacher_first(rgb)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = adapter(aux)
        smooth_l1 = F.smooth_l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        cosine = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean()
        loss = smooth_l1 + 0.1 * (1.0 - cosine)
        if training:
            loss.backward()
            optimizer.step()
        size = float(rgb.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * size
        totals["smooth_l1"] += float(smooth_l1.detach().cpu()) * size
        totals["mse"] += float(mse.detach().cpu()) * size
        totals["cosine"] += float(cosine.detach().cpu()) * size
        totals["count"] += size
    count = max(totals.pop("count"), 1.0)
    return {key: value / count for key, value in totals.items()}


def _initialize_adapter_from_teacher_mean(adapter: Any, teacher_first: Any, *, aux_channels: int) -> None:
    import torch

    with torch.no_grad():
        teacher_weight = teacher_first.conv.weight.detach()
        adapter[0].weight.copy_(teacher_weight.mean(dim=1, keepdim=True).repeat(1, int(aux_channels), 1, 1))
        adapter[1].weight.copy_(teacher_first.bn.weight.detach())
        adapter[1].bias.copy_(teacher_first.bn.bias.detach())
        adapter[1].running_mean.copy_(teacher_first.bn.running_mean.detach())
        adapter[1].running_var.copy_(teacher_first.bn.running_var.detach())


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _resolve_device(device: str, torch: Any) -> str:
    if device != "auto":
        return str(device)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_data_config(path: Path) -> tuple[Path, Dict[str, Any]]:
    config_path = path if path.is_file() else path / "data.yaml"
    config = _read_yaml_like(config_path)
    root = Path(config.get("path", config_path.parent)).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve(), config


def _read_yaml_like(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text()) or {}
        return dict(payload)
    except Exception:
        payload: Dict[str, Any] = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip().strip("'\"")
        return payload


if __name__ == "__main__":
    raise SystemExit(main())
