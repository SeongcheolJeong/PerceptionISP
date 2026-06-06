"""Evaluate a trained RGB+aux dense detector directly on exported tensors."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .aux_dnn import load_manifest
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--label-aware", action="store_true")
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_dense_eval")
    args = parser.parse_args(argv)

    summary = evaluate_dense_manifest(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=str(args.split),
        confidence=args.confidence,
        device=str(args.device),
        label_agnostic=not bool(args.label_aware),
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
    device: str = "auto",
    label_agnostic: bool = False,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    import torch

    start = time.perf_counter()
    manifest = load_manifest(manifest_path)
    manifest_root = Path(manifest_path).expanduser().parent
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_summary = checkpoint.get("summary", {}) if isinstance(checkpoint, Mapping) else {}
    indices = _indices_for_split(
        split,
        sample_count=len(manifest),
        summary=checkpoint_summary if isinstance(checkpoint_summary, Mapping) else {},
    )
    detector = rgb_aux_detector_from_checkpoint(
        str(checkpoint_path),
        confidence=confidence,
        device=device,
    )
    sample_rows = []
    metric_rows = []
    for index in indices:
        item = manifest[int(index)]
        tensor_path = manifest_root / str(item["tensor_path"])
        label_path = manifest_root / str(item["label_path"])
        with np.load(tensor_path) as payload:
            tensor = np.asarray(payload["rgb_aux_hwc"], dtype=np.float32)
        ground_truth = _read_boxes(label_path)
        result = detector.detect(tensor, input_name="perception_rgb_aux_dnn")
        metrics = evaluate_detections(
            result.detections,
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
                "detector": result.to_dict(),
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
        "sample_count": int(len(sample_rows)),
        "selected_indices": [int(index) for index in indices],
        "detector_name": detector.name,
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(len(sample_rows) / elapsed) if sample_rows else 0.0,
        "aggregate": aggregate,
        "checkpoint_summary": {
            "sample_count": checkpoint_summary.get("sample_count"),
            "train_sample_count": checkpoint_summary.get("train_sample_count"),
            "eval_sample_count": checkpoint_summary.get("eval_sample_count"),
            "split_strategy": checkpoint_summary.get("split_strategy"),
            "missing_eval_class_names": checkpoint_summary.get("missing_eval_class_names"),
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


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest": summary.get("manifest"),
        "checkpoint": summary.get("checkpoint"),
        "split": summary.get("split"),
        "label_agnostic": summary.get("label_agnostic"),
        "sample_count": summary.get("sample_count"),
        "detector_name": summary.get("detector_name"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "samples_per_second": summary.get("samples_per_second"),
        "aggregate": summary.get("aggregate"),
        "checkpoint_summary": summary.get("checkpoint_summary"),
    }


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
