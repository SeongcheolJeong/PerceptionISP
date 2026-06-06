"""DNN-facing adapters for PerceptionISP RGB + auxiliary maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .eval_types import BoundingBox, PipelineImageSet


RGB_AUX_CHANNELS = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "aux_edge_strength",
    "aux_saturation",
    "aux_reliability",
)
CHANNEL_MODES = ("rgb_aux", "rgb_only", "aux_only")


def build_rgb_aux_tensor(
    images: PipelineImageSet,
    *,
    layout: str = "hwc",
    dtype: Any = np.float32,
) -> np.ndarray:
    """Return a DNN-ready tensor from Perception RGB and aux maps."""

    rgb = np.asarray(images.perception_rgb, dtype=np.float64)
    aux = np.asarray(images.perception_aux_rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("perception_rgb must be HxWx3")
    if aux.ndim != 3 or aux.shape[2] < 3:
        raise ValueError("perception_aux_rgb must be HxWx3")
    if rgb.shape[:2] != aux.shape[:2]:
        raise ValueError("perception_rgb and perception_aux_rgb must have the same HxW shape")
    tensor = np.concatenate([rgb[:, :, :3], aux[:, :, :3]], axis=2)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = np.clip(tensor, 0.0, 1.0).astype(dtype, copy=False)
    normalized_layout = str(layout or "hwc").lower()
    if normalized_layout == "hwc":
        return tensor
    if normalized_layout == "chw":
        return np.transpose(tensor, (2, 0, 1))
    raise ValueError("layout must be 'hwc' or 'chw'")


def normalize_channel_mode(value: str | None) -> str:
    normalized = str(value or "rgb_aux").lower().replace("-", "_")
    if normalized in {"all", "full", "rgbaux", "rgb_aux"}:
        return "rgb_aux"
    if normalized in {"rgb", "rgb_only"}:
        return "rgb_only"
    if normalized in {"aux", "aux_only"}:
        return "aux_only"
    raise ValueError(f"unsupported RGB+aux channel mode: {value!r}")


def channel_mask_for_mode(value: str | None) -> Tuple[float, ...]:
    mode = normalize_channel_mode(value)
    if mode == "rgb_only":
        return (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    if mode == "aux_only":
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def apply_channel_mask(tensor: Any, channel_mask: Sequence[float]):
    mask_values = tuple(float(value) for value in channel_mask)
    if len(mask_values) != len(RGB_AUX_CHANNELS):
        raise ValueError("channel mask must have one value per RGB+aux channel")
    if hasattr(tensor, "new_tensor"):
        mask = tensor.new_tensor(mask_values)
        if getattr(tensor, "ndim", 0) == 4:
            return tensor * mask.view(1, -1, 1, 1)
        if getattr(tensor, "ndim", 0) == 3:
            return tensor * mask.view(-1, 1, 1)
        raise ValueError("torch RGB+aux tensor must be CHW or NCHW")
    arr = np.asarray(tensor, dtype=np.float32)
    mask_np = np.asarray(mask_values, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[1] != len(RGB_AUX_CHANNELS):
            raise ValueError("numpy RGB+aux tensor must be NCHW")
        return arr * mask_np.reshape(1, -1, 1, 1)
    if arr.ndim == 3:
        if arr.shape[0] == len(RGB_AUX_CHANNELS):
            return arr * mask_np.reshape(-1, 1, 1)
        if arr.shape[2] == len(RGB_AUX_CHANNELS):
            return arr * mask_np.reshape(1, 1, -1)
    raise ValueError("numpy RGB+aux tensor must be CHW, HWC, or NCHW")


def boxes_xyxy_array(boxes: Sequence[BoundingBox]) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray([box.xyxy for box in boxes], dtype=np.float32).reshape(-1, 4)


def normalized_boxes_xyxy_array(boxes: Sequence[BoundingBox], *, width: int, height: int) -> np.ndarray:
    arr = boxes_xyxy_array(boxes).astype(np.float32, copy=True)
    if arr.size == 0:
        return arr
    scale = np.asarray([max(float(width), 1.0), max(float(height), 1.0), max(float(width), 1.0), max(float(height), 1.0)], dtype=np.float32)
    return np.clip(arr / scale[None, :], 0.0, 1.0)


def label_payload(boxes: Sequence[BoundingBox]) -> Dict[str, Any]:
    return {
        "boxes": [box.to_dict() for box in boxes],
        "labels": [box.label for box in boxes],
        "box_count": int(len(boxes)),
    }


def load_manifest(path: str | Path) -> Tuple[Dict[str, Any], ...]:
    manifest_path = Path(path).expanduser()
    entries = []
    for line in manifest_path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return tuple(entries)


def make_torch_dataset(manifest_path: str | Path):
    """Return a PyTorch Dataset that loads exported RGB+aux tensors.

    This intentionally imports torch lazily so the rest of the package remains
    usable without a deep-learning runtime.
    """

    import torch
    from torch.utils.data import Dataset

    manifest = load_manifest(manifest_path)
    base = Path(manifest_path).expanduser().parent

    class RGBAuxTensorDataset(Dataset):
        def __len__(self) -> int:
            return len(manifest)

        def __getitem__(self, index: int):
            item = manifest[int(index)]
            tensor_path = base / str(item["tensor_path"])
            label_path = base / str(item["label_path"])
            with np.load(tensor_path) as payload:
                tensor = np.asarray(payload["rgb_aux_chw"], dtype=np.float32)
            labels = json.loads(label_path.read_text())
            boxes = torch.as_tensor(labels.get("boxes_xyxy", []), dtype=torch.float32)
            boxes_normalized = torch.as_tensor(labels.get("boxes_xyxy_normalized", []), dtype=torch.float32)
            class_names = tuple(str(value) for value in labels.get("labels", []))
            return torch.from_numpy(tensor), {
                "boxes": boxes,
                "boxes_normalized": boxes_normalized,
                "labels_text": class_names,
                "sample_id": str(item.get("sample_id", index)),
            }

    return RGBAuxTensorDataset()


def make_aux_early_fusion_stem(*, in_channels: int = 6, out_channels: int = 24):
    """Create a small PyTorch stem proving a DNN can consume RGB+aux channels."""

    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(out_channels)),
        nn.SiLU(inplace=True),
        nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(int(out_channels)),
        nn.SiLU(inplace=True),
    )


def make_aux_smoke_detector_model(*, stem_channels: int = 16):
    """Create the tiny RGB+aux objectness/box model used by smoke tests."""

    import torch.nn as nn

    return nn.Sequential(
        make_aux_early_fusion_stem(in_channels=6, out_channels=int(stem_channels)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(int(stem_channels), 5),
    )


def make_aux_dense_detector_model(
    *,
    num_classes: int,
    grid_size: Tuple[int, int] = (15, 20),
    base_channels: int = 24,
):
    """Create a compact class-aware RGB+aux grid detector."""

    import torch.nn as nn

    class RGBAuxDenseDetectorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(base_channels)
            self.grid_size = (int(grid_size[0]), int(grid_size[1]))
            self.num_classes = int(num_classes)
            self.features = nn.Sequential(
                nn.Conv2d(6, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels * 4),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels * 4),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(self.grid_size),
            )
            self.head = nn.Conv2d(channels * 4, 5 + self.num_classes, kernel_size=1)

        def forward(self, value: Any) -> Any:
            return self.head(self.features(value))

    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    return RGBAuxDenseDetectorModel()


def labels_from_manifest(manifest_path: str | Path) -> Tuple[str, ...]:
    """Return sorted class labels referenced by an exported RGB+aux manifest."""

    manifest = load_manifest(manifest_path)
    base = Path(manifest_path).expanduser().parent
    labels = set()
    for item in manifest:
        label_path = base / str(item["label_path"])
        payload = json.loads(label_path.read_text())
        labels.update(str(value) for value in payload.get("labels", ()))
    return tuple(sorted(labels)) or ("object",)


def tensor_stats(tensor: np.ndarray) -> Mapping[str, Any]:
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == len(RGB_AUX_CHANNELS):
        channel_axis = 0
    elif arr.ndim == 3 and arr.shape[-1] == len(RGB_AUX_CHANNELS):
        channel_axis = 2
    else:
        raise ValueError("expected a 6-channel RGB+aux tensor")
    means = np.mean(arr, axis=(1, 2)) if channel_axis == 0 else np.mean(arr, axis=(0, 1))
    mins = np.min(arr, axis=(1, 2)) if channel_axis == 0 else np.min(arr, axis=(0, 1))
    maxs = np.max(arr, axis=(1, 2)) if channel_axis == 0 else np.max(arr, axis=(0, 1))
    return {
        "channels": list(RGB_AUX_CHANNELS),
        "shape": [int(value) for value in arr.shape],
        "mean": [float(value) for value in means],
        "min": [float(value) for value in mins],
        "max": [float(value) for value in maxs],
    }
