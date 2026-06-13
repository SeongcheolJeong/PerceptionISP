"""Build Ultralytics YOLO datasets from exported PerceptionISP RGB+aux tensors."""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from .aux_dnn import load_manifest, normalize_channel_mode
from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Convert PerceptionISP RGB+aux tensor exports to a YOLO NPY dataset.")
    parser.add_argument("--manifest", required=True, help="PerceptionISP aux_export manifest.jsonl.")
    parser.add_argument("--tensor-key", default="rgb_aux_extended_hwc", help="Tensor key inside each exported NPZ.")
    parser.add_argument("--channel-mode", default="rgb_aux", choices=["rgb_aux", "rgb_only", "aux_only"])
    parser.add_argument("--include-labels", default=None, help="Comma-separated labels to keep and class-order.")
    parser.add_argument("--count", type=int, default=None, help="Maximum manifest rows to export after offset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--split-strategy", default="hash", choices=["hash", "sequential"])
    parser.add_argument("--output-dir", default="exports/perception_yolo_aux_dataset")
    args = parser.parse_args(argv)

    summary = export_yolo_aux_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        tensor_key=str(args.tensor_key),
        channel_mode=str(args.channel_mode),
        include_labels=_parse_label_list(args.include_labels),
        count=args.count,
        offset=int(args.offset),
        eval_fraction=float(args.eval_fraction),
        split_strategy=str(args.split_strategy),
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def export_yolo_aux_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    tensor_key: str = "rgb_aux_extended_hwc",
    channel_mode: str = "rgb_aux",
    include_labels: Sequence[str] | None = None,
    count: int | None = None,
    offset: int = 0,
    eval_fraction: float = 0.25,
    split_strategy: str = "hash",
) -> Dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser()
    rows = list(load_manifest(manifest_file))
    start = max(int(offset), 0)
    selected_rows = rows[start:]
    if count is not None:
        selected_rows = selected_rows[: max(int(count), 0)]
    if not selected_rows:
        raise ValueError("no manifest rows selected")

    label_order = tuple(str(value) for value in include_labels) if include_labels else _infer_labels(manifest_file, selected_rows)
    if not label_order:
        raise ValueError("no class labels found; pass --include-labels for an explicit class list")
    label_to_id = {label: index for index, label in enumerate(label_order)}

    train_indices, val_indices = _split_indices(selected_rows, eval_fraction=eval_fraction, strategy=split_strategy)
    destination = Path(output_dir).expanduser()
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

    exported: List[Dict[str, Any]] = []
    for split, indices in (("train", train_indices), ("val", val_indices)):
        for local_index in indices:
            row = selected_rows[int(local_index)]
            exported.append(
                _export_one(
                    manifest_file=manifest_file,
                    row=row,
                    row_index=int(local_index) + start,
                    split=split,
                    destination=destination,
                    tensor_key=tensor_key,
                    channel_mode=channel_mode,
                    label_to_id=label_to_id,
                )
            )

    channel_count = int(exported[0]["channels"])
    data_yaml = _data_yaml(destination=destination, label_order=label_order, channel_count=channel_count)
    (destination / "data.yaml").write_text(data_yaml)
    summary = {
        "manifest": str(manifest_file),
        "output_dir": str(destination),
        "tensor_key": str(tensor_key),
        "channel_mode": normalize_channel_mode(channel_mode),
        "channels": channel_count,
        "class_names": list(label_order),
        "class_count": int(len(label_order)),
        "source_count": int(len(selected_rows)),
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "eval_fraction": float(eval_fraction),
        "split_strategy": str(split_strategy),
        "data_yaml": str(destination / "data.yaml"),
        "exported": exported,
        "purpose": "Ultralytics YOLO fine-tuning dataset with optional RGB+Aux multi-channel NPY images.",
    }
    (destination / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / "index.html").write_text(_render_html(summary))
    return summary


def _export_one(
    *,
    manifest_file: Path,
    row: Mapping[str, Any],
    row_index: int,
    split: str,
    destination: Path,
    tensor_key: str,
    channel_mode: str,
    label_to_id: Mapping[str, int],
) -> Dict[str, Any]:
    tensor_path = _resolve_manifest_path(manifest_file, row.get("tensor_path"))
    label_path = _resolve_manifest_path(manifest_file, row.get("label_path"))
    tensor_npz = np.load(tensor_path)
    tensor = _tensor_hwc(tensor_npz, tensor_key=tensor_key)
    tensor = _select_channels(tensor, mode=channel_mode)
    yolo_tensor = _to_uint8_hwc(tensor, channel_mode=channel_mode)

    stem = _safe_stem(str(row.get("sample_id", f"sample_{row_index:05d}")), row_index)
    image_path = destination / "images" / split / f"{stem}.png"
    npy_path = image_path.with_suffix(".npy")
    yolo_label_path = destination / "labels" / split / f"{stem}.txt"

    preview = yolo_tensor[:, :, :3]
    if preview.shape[2] == 1:
        preview = np.repeat(preview, 3, axis=2)
    Image.fromarray(preview[:, :, :3].astype(np.uint8)).save(image_path)
    np.save(npy_path, yolo_tensor, allow_pickle=False)
    yolo_lines, kept, dropped = _yolo_label_lines(label_path, label_to_id)
    yolo_label_path.write_text("".join(yolo_lines))

    return {
        "sample_id": str(row.get("sample_id", stem)),
        "split": str(split),
        "image": str(image_path.relative_to(destination)),
        "npy": str(npy_path.relative_to(destination)),
        "label": str(yolo_label_path.relative_to(destination)),
        "channels": int(yolo_tensor.shape[2]),
        "height": int(yolo_tensor.shape[0]),
        "width": int(yolo_tensor.shape[1]),
        "kept_boxes": int(kept),
        "dropped_boxes": int(dropped),
    }


def _resolve_manifest_path(manifest_file: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("manifest row is missing a required path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_file.parent / path
    return path


def _tensor_hwc(npz: Any, *, tensor_key: str) -> np.ndarray:
    if tensor_key not in npz:
        raise KeyError(f"tensor key {tensor_key!r} not found in NPZ")
    arr = np.asarray(npz[tensor_key], dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"tensor {tensor_key!r} must be 3D, got shape {arr.shape}")
    key_lower = str(tensor_key).lower()
    if key_lower.endswith("_chw") or (arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 64):
        arr = np.transpose(arr, (1, 2, 0))
    return np.asarray(arr, dtype=np.float32)


def _select_channels(tensor: np.ndarray, *, mode: str) -> np.ndarray:
    normalized = normalize_channel_mode(mode)
    if normalized == "rgb_only":
        if tensor.shape[2] < 3:
            raise ValueError("rgb_only export requires at least 3 tensor channels")
        return tensor[:, :, :3]
    if normalized == "aux_only":
        if tensor.shape[2] <= 3:
            raise ValueError("aux_only export requires aux channels after RGB")
        return tensor[:, :, 3:]
    return tensor


def _to_uint8_hwc(tensor: np.ndarray, *, channel_mode: str) -> np.ndarray:
    arr = np.clip(np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    out = np.round(arr * 255.0).astype(np.uint8)
    # Ultralytics reverses exactly 3-channel arrays from BGR to RGB in Format.
    if normalize_channel_mode(channel_mode) == "rgb_only" and out.shape[2] == 3:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _yolo_label_lines(label_path: Path, label_to_id: Mapping[str, int]) -> Tuple[List[str], int, int]:
    payload = json.loads(label_path.read_text())
    labels = [str(value) for value in payload.get("labels", ())]
    boxes = payload.get("boxes_xyxy_normalized", ())
    lines: List[str] = []
    dropped = 0
    for label, box in zip(labels, boxes):
        if label not in label_to_id:
            dropped += 1
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width <= 0.0 or height <= 0.0:
            dropped += 1
            continue
        cx = x1 + 0.5 * width
        cy = y1 + 0.5 * height
        lines.append(f"{int(label_to_id[label])} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}\n")
    return lines, len(lines), dropped


def _infer_labels(manifest_file: Path, rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    labels = set()
    for row in rows:
        label_path = _resolve_manifest_path(manifest_file, row.get("label_path"))
        payload = json.loads(label_path.read_text())
        labels.update(str(value) for value in payload.get("labels", ()))
    return tuple(sorted(labels))


def _split_indices(rows: Sequence[Mapping[str, Any]], *, eval_fraction: float, strategy: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    count = int(len(rows))
    indices = tuple(range(count))
    if count <= 1 or float(eval_fraction) <= 0.0:
        return indices, ()
    val_count = min(max(int(round(count * min(float(eval_fraction), 0.9))), 1), count - 1)
    if str(strategy).lower() == "hash":
        ordered = tuple(sorted(indices, key=lambda idx: _stable_hash(str(rows[idx].get("sample_id", idx)))))
    else:
        ordered = indices
    val = tuple(sorted(ordered[:val_count]))
    train = tuple(index for index in indices if index not in set(val))
    return train, val


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _safe_stem(sample_id: str, index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(sample_id)).strip("_")
    if not safe:
        safe = "sample"
    return f"{int(index):05d}_{safe}"


def _parse_label_list(value: str | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    labels = tuple(item.strip() for item in str(value).split(",") if item.strip())
    return labels or None


def _data_yaml(*, destination: Path, label_order: Sequence[str], channel_count: int) -> str:
    names = ", ".join(repr(str(label)) for label in label_order)
    return (
        f"path: {destination.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"channels: {int(channel_count)}\n"
        f"names: [{names}]\n"
    )


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html_lib.escape(str(key))}</th><td>{html_lib.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key != "exported"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>YOLO RGB+Aux Dataset</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;"
        "color:#17202a}table{border-collapse:collapse}th,td{border:1px solid #d6dbdf;padding:8px 10px;"
        "text-align:left}th{background:#f4f6f7}</style></head><body>"
        "<h1>YOLO RGB+Aux Dataset</h1><table>"
        f"{rows}</table></body></html>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
