"""Evaluate a trained RGB+aux dense detector directly on exported tensors."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .aux_dnn import hwc_tensor_key, load_manifest
from .detectors import rgb_aux_detector_from_checkpoint
from .eval_types import BoundingBox
from .metrics import aggregate_metric_rows, evaluate_detections
from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RGB+aux dense detector on an exported tensor manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "eval"])
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None, help="Optional dense-detector NMS IoU override.")
    parser.add_argument("--max-detections", type=int, default=None, help="Optional dense-detector top-k cap after NMS.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--label-aware", action="store_true")
    parser.add_argument("--include-labels", default=None, help="Comma-separated class labels to evaluate; default uses checkpoint labels.")
    parser.add_argument("--indices-file", default=None, help="Optional JSON file containing explicit sample indices to evaluate.")
    parser.add_argument("--indices-key", default=None, help="Key to read when --indices-file is a JSON object.")
    parser.add_argument(
        "--input-ablation",
        default="none",
        choices=["none", "zero_aux", "zero_rgb", "shuffle_aux"],
        help="Optional inference-time input ablation for RGB+Aux contribution checks.",
    )
    parser.add_argument("--ablation-seed", type=int, default=0, help="Deterministic seed for shuffle_aux input ablation.")
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_dense_eval")
    args = parser.parse_args(argv)

    summary = evaluate_dense_manifest(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=str(args.split),
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        max_detections=args.max_detections,
        device=str(args.device),
        label_agnostic=not bool(args.label_aware),
        include_labels=parse_label_list(args.include_labels),
        indices=_load_indices_file(args.indices_file, args.indices_key),
        input_ablation=str(args.input_ablation),
        ablation_seed=int(args.ablation_seed),
        output_dir=args.output_dir,
    )
    print(json.dumps(json_ready(_compact_summary(summary)), indent=2))
    return 0


def evaluate_dense_manifest(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "all",
    confidence: float | None = None,
    nms_iou: float | None = None,
    max_detections: int | None = None,
    device: str = "auto",
    label_agnostic: bool = False,
    include_labels: Sequence[str] | None = None,
    indices: Sequence[int] | None = None,
    input_ablation: str = "none",
    ablation_seed: int = 0,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    import torch

    start = time.perf_counter()
    manifest = load_manifest(manifest_path)
    manifest_root = Path(manifest_path).expanduser().parent
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_summary = checkpoint.get("summary", {}) if isinstance(checkpoint, Mapping) else {}
    tensor_key = hwc_tensor_key(checkpoint.get("tensor_key", checkpoint_summary.get("tensor_key", "rgb_aux_chw")) if isinstance(checkpoint, Mapping) else "rgb_aux_chw")
    eval_labels = _eval_labels(checkpoint, checkpoint_summary, include_labels)
    selected_indices = _indices_for_split(
        split,
        sample_count=len(manifest),
        summary=checkpoint_summary if isinstance(checkpoint_summary, Mapping) else {},
    ) if indices is None else tuple(int(index) for index in indices)
    _validate_indices(selected_indices, sample_count=len(manifest))
    ablation_mode = _normalize_input_ablation(input_ablation)
    shuffle_indices = _shuffle_indices(selected_indices, seed=int(ablation_seed)) if ablation_mode == "shuffle_aux" else {}
    detector = rgb_aux_detector_from_checkpoint(
        str(checkpoint_path),
        confidence=confidence,
        nms_iou=nms_iou,
        max_detections=max_detections,
        device=device,
    )
    sample_rows = []
    metric_rows = []
    for index in selected_indices:
        item = manifest[int(index)]
        tensor_path = manifest_root / str(item["tensor_path"])
        label_path = manifest_root / str(item["label_path"])
        with np.load(tensor_path) as payload:
            if tensor_key not in payload:
                raise KeyError(f"tensor payload does not contain {tensor_key!r}: {tensor_path}")
            tensor = np.asarray(payload[tensor_key], dtype=np.float32)
        aux_source_tensor = None
        aux_source_index = None
        if ablation_mode == "shuffle_aux":
            aux_source_index = int(shuffle_indices[int(index)])
            aux_source_item = manifest[aux_source_index]
            aux_source_path = manifest_root / str(aux_source_item["tensor_path"])
            with np.load(aux_source_path) as payload:
                if tensor_key not in payload:
                    raise KeyError(f"tensor payload does not contain {tensor_key!r}: {aux_source_path}")
                aux_source_tensor = np.asarray(payload[tensor_key], dtype=np.float32)
        tensor = _apply_input_ablation(tensor, mode=ablation_mode, aux_source_tensor=aux_source_tensor)
        ground_truth = _filter_boxes(_read_boxes(label_path), eval_labels)
        result = detector.detect(tensor, input_name="perception_rgb_aux_dnn")
        detections = _filter_boxes(result.detections, eval_labels)
        metrics = evaluate_detections(
            detections,
            ground_truth,
            label_agnostic=label_agnostic,
        )
        metric_rows.append(metrics)
        sample_rows.append(
            {
                "index": int(index),
                "sample_id": str(item.get("sample_id", index)),
                "tensor_path": str(item["tensor_path"]),
                "label_path": str(item["label_path"]),
                "gt_count": int(len(ground_truth)),
                "input_ablation": ablation_mode,
                "aux_source_index": aux_source_index,
                "detector": {**result.to_dict(), "detections": [box.to_dict() for box in detections]},
                "metrics": metrics,
            }
        )
    aggregate = aggregate_metric_rows(metric_rows)
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    summary = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "split": str(split),
        "label_agnostic": bool(label_agnostic),
        "eval_labels": list(eval_labels),
        "sample_count": int(len(sample_rows)),
        "selected_indices": [int(index) for index in selected_indices],
        "indices_override": indices is not None,
        "input_ablation": ablation_mode,
        "ablation_seed": int(ablation_seed),
        "detector_name": detector.name,
        "detector_config": {
            "confidence": None if confidence is None else float(confidence),
            "nms_iou": None if nms_iou is None else float(nms_iou),
            "max_detections": None if max_detections is None else int(max_detections),
        },
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(len(sample_rows) / elapsed) if sample_rows else 0.0,
        "aggregate": aggregate,
        "checkpoint_summary": {
            "sample_count": checkpoint_summary.get("sample_count"),
            "train_sample_count": checkpoint_summary.get("train_sample_count"),
            "eval_sample_count": checkpoint_summary.get("eval_sample_count"),
            "split_strategy": checkpoint_summary.get("split_strategy"),
            "missing_eval_class_names": checkpoint_summary.get("missing_eval_class_names"),
            "channel_mode": checkpoint.get("channel_mode") if isinstance(checkpoint, Mapping) else None,
            "model_architecture": checkpoint.get("model_architecture") if isinstance(checkpoint, Mapping) else None,
            "channel_mask": checkpoint.get("channel_mask") if isinstance(checkpoint, Mapping) else None,
            "tensor_key": checkpoint.get("tensor_key") if isinstance(checkpoint, Mapping) else None,
            "input_channels": checkpoint.get("input_channels") if isinstance(checkpoint, Mapping) else None,
        },
        "samples": sample_rows,
    }
    if output_dir is not None:
        destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "dense_eval_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
        (destination / "index.html").write_text(_render_html(summary))
    return summary


def _indices_for_split(split: str, *, sample_count: int, summary: Mapping[str, Any]) -> Tuple[int, ...]:
    normalized = str(split or "all").lower()
    if normalized == "train" and summary.get("train_indices") is not None:
        return tuple(int(index) for index in summary.get("train_indices", ()))
    if normalized == "eval" and summary.get("eval_indices") is not None:
        return tuple(int(index) for index in summary.get("eval_indices", ()))
    if normalized == "train":
        return tuple(range(int(summary.get("train_sample_count", sample_count))))
    if normalized == "eval":
        return ()
    return tuple(range(int(sample_count)))


def _normalize_input_ablation(value: str | None) -> str:
    normalized = str(value or "none").lower().replace("-", "_")
    if normalized in {"none", "off", "identity"}:
        return "none"
    if normalized in {"zero_aux", "aux_zero", "no_aux"}:
        return "zero_aux"
    if normalized in {"zero_rgb", "rgb_zero", "no_rgb"}:
        return "zero_rgb"
    if normalized in {"shuffle_aux", "aux_shuffle", "shuffled_aux"}:
        return "shuffle_aux"
    raise ValueError(f"unsupported input_ablation: {value!r}")


def _shuffle_indices(indices: Sequence[int], *, seed: int) -> Dict[int, int]:
    values = [int(index) for index in indices]
    if not values:
        return {}
    shuffled = list(values)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled)
    if len(values) > 1 and all(left == right for left, right in zip(values, shuffled)):
        shuffled = shuffled[1:] + shuffled[:1]
    return {int(index): int(source) for index, source in zip(values, shuffled)}


def _apply_input_ablation(
    tensor: np.ndarray,
    *,
    mode: str,
    aux_source_tensor: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(tensor, dtype=np.float32).copy()
    if arr.ndim != 3:
        raise ValueError("RGB+aux tensor must be HWC for dense eval")
    if arr.shape[2] < 4 and mode in {"zero_aux", "shuffle_aux"}:
        raise ValueError("RGB+aux tensor must have at least one aux channel")
    if arr.shape[2] < 3 and mode == "zero_rgb":
        raise ValueError("RGB+aux tensor must have at least three RGB channels")
    if mode == "none":
        return arr
    if mode == "zero_aux":
        arr[:, :, 3:] = 0.0
        return arr
    if mode == "zero_rgb":
        arr[:, :, :3] = 0.0
        return arr
    if mode == "shuffle_aux":
        if aux_source_tensor is None:
            raise ValueError("shuffle_aux requires aux_source_tensor")
        source = np.asarray(aux_source_tensor, dtype=np.float32)
        if source.shape != arr.shape:
            raise ValueError(f"shuffle_aux source shape {source.shape} does not match tensor shape {arr.shape}")
        arr[:, :, 3:] = source[:, :, 3:]
        return arr
    raise ValueError(f"unsupported input ablation mode: {mode!r}")


def _load_indices_file(path: str | Path | None, key: str | None = None) -> Tuple[int, ...] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text())
    if isinstance(payload, Mapping):
        selected_key = str(key) if key is not None else "indices"
        if selected_key not in payload:
            available = ", ".join(str(value) for value in sorted(payload.keys()))
            raise KeyError(f"indices file does not contain key {selected_key!r}; available keys: {available}")
        payload = payload[selected_key]
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError("indices payload must be a JSON array of integer sample indices")
    return tuple(int(index) for index in payload)


def _validate_indices(indices: Sequence[int], *, sample_count: int) -> None:
    limit = int(sample_count)
    for index in indices:
        value = int(index)
        if value < 0 or value >= limit:
            raise IndexError(f"sample index out of range: {value} for sample_count={limit}")


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest": summary.get("manifest"),
        "checkpoint": summary.get("checkpoint"),
        "split": summary.get("split"),
        "label_agnostic": summary.get("label_agnostic"),
        "eval_labels": summary.get("eval_labels"),
        "sample_count": summary.get("sample_count"),
        "indices_override": summary.get("indices_override"),
        "input_ablation": summary.get("input_ablation"),
        "ablation_seed": summary.get("ablation_seed"),
        "detector_name": summary.get("detector_name"),
        "detector_config": summary.get("detector_config"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "samples_per_second": summary.get("samples_per_second"),
        "aggregate": summary.get("aggregate"),
        "checkpoint_summary": summary.get("checkpoint_summary"),
    }


def parse_label_list(value: str | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    labels = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return labels or None


def _eval_labels(
    checkpoint: Any,
    checkpoint_summary: Mapping[str, Any],
    include_labels: Sequence[str] | None,
) -> Tuple[str, ...]:
    checkpoint_labels = tuple(str(value) for value in checkpoint.get("class_names", ())) if isinstance(checkpoint, Mapping) else ()
    if include_labels is not None:
        labels = tuple(str(value) for value in include_labels)
        if checkpoint_labels:
            missing = tuple(label for label in labels if label not in checkpoint_labels)
            if missing:
                raise ValueError(f"include_labels not present in checkpoint class_names: {', '.join(missing)}")
        return tuple(dict.fromkeys(labels))
    if checkpoint_labels:
        return checkpoint_labels
    summary_labels = checkpoint_summary.get("class_names", ()) if isinstance(checkpoint_summary, Mapping) else ()
    if summary_labels:
        return tuple(str(value) for value in summary_labels)
    return ("object",)


def _filter_boxes(items: Sequence[Any], labels: Sequence[str]) -> Tuple[Any, ...]:
    allowed = {str(label) for label in labels}
    if not allowed:
        return tuple(items)
    filtered = []
    for item in items:
        label = getattr(item, "label", None)
        if label is None and getattr(item, "box", None) is not None:
            label = getattr(item.box, "label", None)
        if str(label) in allowed:
            filtered.append(item)
    return tuple(filtered)


def _read_boxes(path: Path) -> Tuple[BoundingBox, ...]:
    payload = json.loads(path.read_text())
    boxes = []
    for item in payload.get("boxes", ()):
        boxes.append(BoundingBox(tuple(float(value) for value in item["xyxy"]), label=str(item.get("label", "object"))))
    return tuple(boxes)


def _render_html(summary: Mapping[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    rows = []
    for sample in summary.get("samples", ()):
        metrics = sample.get("metrics", {})
        detector = sample.get("detector", {})
        rows.append(
            "<tr>"
            f"<td>{int(sample.get('index', 0))}</td>"
            f"<td>{html_lib.escape(str(sample.get('sample_id', '')))}</td>"
            f"<td>{int(sample.get('gt_count', 0))}</td>"
            f"<td>{len(detector.get('detections', ()))}</td>"
            f"<td>{float(metrics.get('precision@0.50', 0.0)):.3f}</td>"
            f"<td>{float(metrics.get('recall@0.50', 0.0)):.3f}</td>"
            f"<td>{float(metrics.get('mean_recall', 0.0)):.3f}</td>"
            "</tr>"
        )
    missing = ", ".join(str(value) for value in summary.get("checkpoint_summary", {}).get("missing_eval_class_names") or ())
    eval_labels = ", ".join(str(value) for value in summary.get("eval_labels") or ())
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP RGB+Aux Dense Eval</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 10px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP RGB+Aux Dense Eval</h1>
  <div class=\"note\">Direct tensor-manifest evaluation. CameraE2E and ISP are not recomputed in this report.</div>
  <p>Split: <code>{html_lib.escape(str(summary.get('split', '')))}</code>, samples: {int(summary.get('sample_count', 0))}, label agnostic: {bool(summary.get('label_agnostic', False))}.</p>
  <p>Channel mode: <code>{html_lib.escape(str(summary.get('checkpoint_summary', {}).get('channel_mode') or 'rgb_aux'))}</code>.</p>
  <p>Input ablation: <code>{html_lib.escape(str(summary.get('input_ablation', 'none')))}</code>, seed: <code>{int(summary.get('ablation_seed', 0))}</code>.</p>
  <p>Eval labels: <code>{html_lib.escape(eval_labels or 'none')}</code>.</p>
  <p>Recall@0.50 mean: {float(aggregate.get('recall@0.50_mean', 0.0)):.3f}, precision@0.50 mean: {float(aggregate.get('precision@0.50_mean', 0.0)):.3f}, detections/sample: {float(aggregate.get('det_count_mean', 0.0)):.2f}.</p>
  <p>Missing eval classes from train: <code>{html_lib.escape(missing or 'none')}</code>.</p>
  <table>
    <thead><tr><th>Index</th><th>Sample</th><th>GT</th><th>Det</th><th>P@0.50</th><th>R@0.50</th><th>Mean R</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Summary JSON: <code>dense_eval_summary.json</code></p>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
