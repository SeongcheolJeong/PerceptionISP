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
RGB_AUX_EXTENDED_CHANNELS = RGB_AUX_CHANNELS + (
    "aux_noise_risk",
    "aux_clipping_distance",
    "aux_demosaic_confidence",
    "aux_hdr_confidence",
    "aux_lens_gain",
    "aux_color_confidence",
    "aux_blur_focus_confidence",
    "aux_edge_evidence",
    "aux_psf_blur_confidence",
    "aux_psf_edge_likelihood",
)
RGB_AUX_TENSOR_KEY = "rgb_aux_chw"
RGB_AUX_EXTENDED_TENSOR_KEY = "rgb_aux_extended_chw"
CHANNEL_MODES = ("rgb_aux", "rgb_only", "aux_only")


def build_rgb_aux_tensor(
    images: PipelineImageSet,
    *,
    layout: str = "hwc",
    dtype: Any = np.float32,
    channels: Sequence[str] | None = None,
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
    channel_names = tuple(channels or RGB_AUX_CHANNELS)
    tensor = np.stack([_dnn_channel(name, rgb=rgb, aux=aux, aux_maps=images.aux_maps) for name in channel_names], axis=2)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = np.clip(tensor, 0.0, 1.0).astype(dtype, copy=False)
    normalized_layout = str(layout or "hwc").lower()
    if normalized_layout == "hwc":
        return tensor
    if normalized_layout == "chw":
        return np.transpose(tensor, (2, 0, 1))
    raise ValueError("layout must be 'hwc' or 'chw'")


def build_rgb_aux_extended_tensor(
    images: PipelineImageSet,
    *,
    layout: str = "hwc",
    dtype: Any = np.float32,
) -> np.ndarray:
    """Return the sensor-native extended RGB+aux tensor without changing the 6-channel default path."""

    return build_rgb_aux_tensor(images, layout=layout, dtype=dtype, channels=RGB_AUX_EXTENDED_CHANNELS)


def channels_for_tensor_key(tensor_key: str | None) -> Tuple[str, ...]:
    normalized = str(tensor_key or RGB_AUX_TENSOR_KEY)
    if normalized in {"rgb_aux_chw", "rgb_aux_hwc", "stable", "six"}:
        return RGB_AUX_CHANNELS
    if normalized in {"rgb_aux_extended_chw", "rgb_aux_extended_hwc", "extended"}:
        return RGB_AUX_EXTENDED_CHANNELS
    raise ValueError(f"unsupported RGB+aux tensor key: {tensor_key!r}")


def hwc_tensor_key(tensor_key: str | None) -> str:
    normalized = str(tensor_key or RGB_AUX_TENSOR_KEY)
    if normalized in {"rgb_aux_chw", "rgb_aux_hwc", "stable", "six"}:
        return "rgb_aux_hwc"
    if normalized in {"rgb_aux_extended_chw", "rgb_aux_extended_hwc", "extended"}:
        return "rgb_aux_extended_hwc"
    raise ValueError(f"unsupported RGB+aux tensor key: {tensor_key!r}")


def chw_tensor_key(tensor_key: str | None) -> str:
    normalized = str(tensor_key or RGB_AUX_TENSOR_KEY)
    if normalized in {"rgb_aux_chw", "rgb_aux_hwc", "stable", "six"}:
        return "rgb_aux_chw"
    if normalized in {"rgb_aux_extended_chw", "rgb_aux_extended_hwc", "extended"}:
        return "rgb_aux_extended_chw"
    raise ValueError(f"unsupported RGB+aux tensor key: {tensor_key!r}")


def _dnn_channel(name: str, *, rgb: np.ndarray, aux: np.ndarray, aux_maps: Mapping[str, Any]) -> np.ndarray:
    normalized = str(name)
    if normalized == "rgb_r":
        return rgb[:, :, 0]
    if normalized == "rgb_g":
        return rgb[:, :, 1]
    if normalized == "rgb_b":
        return rgb[:, :, 2]
    shape = rgb.shape[:2]
    fallback_zero = np.zeros(shape, dtype=np.float64)
    if normalized == "aux_edge_strength":
        return _map_or_fallback(aux_maps, "edge_strength", aux[:, :, 0], shape)
    if normalized == "aux_saturation":
        return _map_or_fallback(aux_maps, "saturation", aux[:, :, 1], shape)
    if normalized in {"aux_reliability", "aux_snr"}:
        return _map_or_fallback(aux_maps, "snr_map", aux[:, :, 2], shape)
    if normalized == "aux_noise_risk":
        return _noise_risk(_map_or_fallback(aux_maps, "noise_variance", fallback_zero, shape))
    if normalized == "aux_clipping_distance":
        saturation = _map_or_fallback(aux_maps, "saturation", aux[:, :, 1], shape)
        return _map_or_fallback(aux_maps, "clipping_distance", 1.0 - saturation, shape)
    if normalized == "aux_demosaic_confidence":
        return _map_or_fallback(aux_maps, "demosaic_confidence", np.ones(shape, dtype=np.float64), shape)
    if normalized == "aux_hdr_confidence":
        return _map_or_fallback(aux_maps, "hdr_confidence", np.ones(shape, dtype=np.float64), shape)
    if normalized == "aux_lens_gain":
        return _unit_map(_map_or_fallback(aux_maps, "lens_gain", np.ones(shape, dtype=np.float64), shape))
    if normalized == "aux_color_confidence":
        return _map_or_fallback(aux_maps, "color_confidence", np.ones(shape, dtype=np.float64), shape)
    if normalized == "aux_blur_focus_confidence":
        return _map_or_fallback(aux_maps, "blur_focus_confidence", np.ones(shape, dtype=np.float64), shape)
    if normalized == "aux_edge_evidence":
        fallback_edge = _map_or_fallback(aux_maps, "edge_confidence", aux[:, :, 0], shape)
        return _map_or_fallback(aux_maps, "edge_evidence", fallback_edge, shape)
    if normalized == "aux_psf_blur_confidence":
        return _map_or_fallback(aux_maps, "psf_blur_confidence", np.ones(shape, dtype=np.float64), shape)
    if normalized == "aux_psf_edge_likelihood":
        return _map_or_fallback(aux_maps, "psf_edge_likelihood", np.ones(shape, dtype=np.float64), shape)
    raise ValueError(f"unsupported RGB+aux channel: {name!r}")


def _map_or_fallback(aux_maps: Mapping[str, Any], key: str, fallback: Any, shape: tuple[int, int]) -> np.ndarray:
    value = aux_maps.get(key, fallback) if isinstance(aux_maps, Mapping) else fallback
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"aux map {key!r} has shape {arr.shape}, expected {shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _noise_risk(value: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    if not bool(np.any(arr > 0.0)):
        return np.zeros_like(arr)
    scale = max(float(np.percentile(arr, 95.0)), 1.0e-12)
    return np.clip(arr / scale, 0.0, 1.0)


def _unit_map(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)


def normalize_channel_mode(value: str | None) -> str:
    normalized = str(value or "rgb_aux").lower().replace("-", "_")
    if normalized in {"all", "full", "rgbaux", "rgb_aux"}:
        return "rgb_aux"
    if normalized in {"rgb", "rgb_only"}:
        return "rgb_only"
    if normalized in {"aux", "aux_only"}:
        return "aux_only"
    raise ValueError(f"unsupported RGB+aux channel mode: {value!r}")


def channel_mask_for_mode(value: str | None, *, channels: Sequence[str] | None = None) -> Tuple[float, ...]:
    mode = normalize_channel_mode(value)
    channel_count = len(tuple(channels or RGB_AUX_CHANNELS))
    if mode == "rgb_only":
        return tuple(1.0 if index < 3 else 0.0 for index in range(channel_count))
    if mode == "aux_only":
        return tuple(0.0 if index < 3 else 1.0 for index in range(channel_count))
    return tuple(1.0 for _ in range(channel_count))


def apply_channel_mask(tensor: Any, channel_mask: Sequence[float]):
    mask_values = tuple(float(value) for value in channel_mask)
    if not mask_values:
        raise ValueError("channel mask must have at least one value")
    if hasattr(tensor, "new_tensor"):
        mask = tensor.new_tensor(mask_values)
        if getattr(tensor, "ndim", 0) == 4:
            if int(tensor.shape[1]) != len(mask_values):
                raise ValueError("torch RGB+aux tensor channel count does not match channel mask")
            return tensor * mask.view(1, -1, 1, 1)
        if getattr(tensor, "ndim", 0) == 3:
            if int(tensor.shape[0]) != len(mask_values):
                raise ValueError("torch RGB+aux tensor channel count does not match channel mask")
            return tensor * mask.view(-1, 1, 1)
        raise ValueError("torch RGB+aux tensor must be CHW or NCHW")
    arr = np.asarray(tensor, dtype=np.float32)
    mask_np = np.asarray(mask_values, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[1] != len(mask_values):
            raise ValueError("numpy RGB+aux tensor must be NCHW")
        return arr * mask_np.reshape(1, -1, 1, 1)
    if arr.ndim == 3:
        if arr.shape[0] == len(mask_values):
            return arr * mask_np.reshape(-1, 1, 1)
        if arr.shape[2] == len(mask_values):
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


def make_torch_dataset(manifest_path: str | Path, *, tensor_key: str = RGB_AUX_TENSOR_KEY):
    """Return a PyTorch Dataset that loads exported RGB+aux tensors.

    This intentionally imports torch lazily so the rest of the package remains
    usable without a deep-learning runtime.
    """

    import torch
    from torch.utils.data import Dataset

    manifest = load_manifest(manifest_path)
    base = Path(manifest_path).expanduser().parent
    resolved_tensor_key = chw_tensor_key(tensor_key)
    channel_count = len(channels_for_tensor_key(resolved_tensor_key))

    class RGBAuxTensorDataset(Dataset):
        def __len__(self) -> int:
            return len(manifest)

        def __getitem__(self, index: int):
            item = manifest[int(index)]
            tensor_path = base / str(item["tensor_path"])
            label_path = base / str(item["label_path"])
            with np.load(tensor_path) as payload:
                if resolved_tensor_key not in payload:
                    raise KeyError(f"tensor payload does not contain {resolved_tensor_key!r}: {tensor_path}")
                tensor = np.asarray(payload[resolved_tensor_key], dtype=np.float32)
                if tensor.ndim != 3:
                    raise ValueError(f"RGB+aux tensor must be 3D CHW or HWC: {tensor_path}")
                if tensor.shape[0] == channel_count:
                    pass
                elif tensor.shape[-1] == channel_count:
                    tensor = np.transpose(tensor, (2, 0, 1))
                else:
                    raise ValueError(f"RGB+aux tensor channel count does not match {channel_count}: {tensor_path}")
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


def make_aux_smoke_detector_model(*, stem_channels: int = 16, in_channels: int = 6):
    """Create the tiny RGB+aux objectness/box model used by smoke tests."""

    import torch.nn as nn

    return nn.Sequential(
        make_aux_early_fusion_stem(in_channels=int(in_channels), out_channels=int(stem_channels)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(int(stem_channels), 5),
    )


def make_aux_dense_detector_model(
    *,
    num_classes: int,
    grid_size: Tuple[int, int] = (15, 20),
    base_channels: int = 24,
    in_channels: int = 6,
    architecture: str = "early_fusion",
):
    """Create a compact class-aware RGB+aux grid detector."""

    import torch.nn as nn

    normalized_architecture = str(architecture or "early_fusion").lower().replace("-", "_")
    if normalized_architecture not in {"early_fusion", "late_fusion"}:
        raise ValueError(f"unsupported RGB+aux dense detector architecture: {architecture!r}")

    class RGBAuxDenseDetectorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(base_channels)
            self.grid_size = (int(grid_size[0]), int(grid_size[1]))
            self.num_classes = int(num_classes)
            self.features = nn.Sequential(
                nn.Conv2d(int(in_channels), channels, kernel_size=3, stride=2, padding=1, bias=False),
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

    class RGBLateAuxDenseDetectorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(base_channels)
            rgb_channels = min(int(in_channels), 3)
            aux_channels = max(int(in_channels) - rgb_channels, 1)
            self.rgb_channels = rgb_channels
            self.aux_channels = aux_channels
            self.grid_size = (int(grid_size[0]), int(grid_size[1]))
            self.num_classes = int(num_classes)
            self.rgb_features = nn.Sequential(
                nn.Conv2d(rgb_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.SiLU(inplace=True),
            )
            self.aux_features = nn.Sequential(
                nn.Conv2d(aux_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.SiLU(inplace=True),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels * 4),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(self.grid_size),
            )
            self.head = nn.Conv2d(channels * 4, 5 + self.num_classes, kernel_size=1)

        def forward(self, value: Any) -> Any:
            import torch

            rgb = value[:, : self.rgb_channels, :, :].contiguous()
            aux = value[:, self.rgb_channels :, :, :].contiguous()
            if int(aux.shape[1]) <= 0:
                aux = aux.new_zeros((int(aux.shape[0]), self.aux_channels, int(aux.shape[2]), int(aux.shape[3])))
            return self.head(self.fusion(torch.cat((self.rgb_features(rgb), self.aux_features(aux)), dim=1)))

    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    if normalized_architecture == "late_fusion":
        return RGBLateAuxDenseDetectorModel()
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


def tensor_stats(tensor: np.ndarray, *, channels: Sequence[str] | None = None) -> Mapping[str, Any]:
    channel_names = tuple(channels or RGB_AUX_CHANNELS)
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == len(channel_names):
        channel_axis = 0
    elif arr.ndim == 3 and arr.shape[-1] == len(channel_names):
        channel_axis = 2
    else:
        raise ValueError(f"expected a {len(channel_names)}-channel RGB+aux tensor")
    means = np.mean(arr, axis=(1, 2)) if channel_axis == 0 else np.mean(arr, axis=(0, 1))
    mins = np.min(arr, axis=(1, 2)) if channel_axis == 0 else np.min(arr, axis=(0, 1))
    maxs = np.max(arr, axis=(1, 2)) if channel_axis == 0 else np.max(arr, axis=(0, 1))
    return {
        "channels": list(channel_names),
        "shape": [int(value) for value in arr.shape],
        "mean": [float(value) for value in means],
        "min": [float(value) for value in mins],
        "max": [float(value) for value in maxs],
    }
