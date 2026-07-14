"""Export PerceptionISP RGB+aux tensors for downstream DNN training."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from perception_isp.core.aux_dnn import (
    RGB_AUX_CHANNELS,
    RGB_AUX_EXTENDED_CHANNELS,
    boxes_xyxy_array,
    build_rgb_aux_extended_tensor,
    build_rgb_aux_tensor,
    label_payload,
    normalized_boxes_xyxy_array,
    tensor_stats,
)
from perception_isp.evaluation.comparison import build_pipeline_images
from perception_isp.core.task_types import EvaluationSample
from perception_isp.core.types import PerceptionISPConfig, json_ready


EDGE6_CHANNELS = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "aux_edge_strength",
    "aux_edge_evidence",
    "aux_psf_edge_likelihood",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export PerceptionISP RGB+aux tensors for DNN training.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "camerae2e-synthetic", "yolo-dataset", "kitti-dataset", "pascalraw-dataset"],
        default="synthetic",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--pascalraw-manifest", default=None, help="PASCALRAW subset manifest JSON for --source pascalraw-dataset.")
    parser.add_argument("--pascalraw-native-raw", action="store_true", help="Use full PASCALRAW NEF native Bayer RAW instead of downsampled PNG remosaic.")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many dataset images before applying --count.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--cfa", default="auto")
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--load-progress-interval", type=int, default=0, help="Print dataset loading progress every N samples; 0 disables progress logging.")
    parser.add_argument("--raw-cache-dir", default=None, help="Optional directory for cached dataset RAW samples.")
    parser.add_argument("--tone-mapping", default="srgb")
    parser.add_argument("--denoise-strength", type=float, default=0.18)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=["edge_aware", "bilinear"])
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.35)
    parser.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated base tensor channels to store in rgb_aux_hwc/chw. "
            "Aliases: stable6, edge6. Use edge6 to keep edge_evidence without exporting all 16 extended channels."
        ),
    )
    parser.add_argument("--minimal", action="store_true", help="Export only 6-channel RGB+aux tensors needed for fast gate training/eval.")
    parser.add_argument("--no-preview", action="store_true", help="Skip preview RGB/aux arrays while keeping trainable RGB+aux tensors.")
    parser.add_argument("--no-compress", action="store_true", help="Use faster uncompressed NPZ writes instead of np.savez_compressed.")
    parser.add_argument("--output-dir", default="exports/perception_rgb_aux_dataset")
    args = parser.parse_args(argv)

    samples = _load_samples(
        source=args.source,
        dataset=args.dataset,
        split=args.split,
        count=args.count,
        offset=int(args.offset),
        width=args.width,
        height=args.height,
        cfa_pattern=args.cfa,
        use_camerae2e=not bool(args.no_camerae2e),
        progress_interval=int(args.load_progress_interval),
        cache_dir=args.raw_cache_dir,
        pascalraw_manifest=args.pascalraw_manifest,
        pascalraw_native_raw=bool(args.pascalraw_native_raw),
    )
    config = PerceptionISPConfig(
        tone_mapping=args.tone_mapping,
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    summary = export_aux_dataset(
        samples,
        args.output_dir,
        config=config,
        channels=_parse_channel_list(args.channels),
        include_extended=not bool(args.minimal),
        include_preview=not bool(args.minimal) and not bool(args.no_preview),
        compress=not bool(args.no_compress),
        run_config={
            "source": args.source,
            "dataset": args.dataset,
            "split": args.split,
            "pascalraw_manifest": args.pascalraw_manifest,
            "pascalraw_native_raw": bool(args.pascalraw_native_raw),
            "count": int(args.count),
            "offset": int(args.offset),
            "width": int(args.width),
            "height": int(args.height),
            "cfa": args.cfa,
            "use_camerae2e": not bool(args.no_camerae2e),
            "load_progress_interval": int(args.load_progress_interval),
            "raw_cache_dir": args.raw_cache_dir,
            "tone_mapping": args.tone_mapping,
            "denoise_strength": float(args.denoise_strength),
            "demosaic_method": str(args.demosaic_method),
            "demosaic_artifact_suppression": float(args.demosaic_artifact_suppression),
            "channels": list(_parse_channel_list(args.channels) or RGB_AUX_CHANNELS),
            "minimal": bool(args.minimal),
            "no_preview": bool(args.no_preview),
            "include_preview": not bool(args.minimal) and not bool(args.no_preview),
            "compress": not bool(args.no_compress),
        },
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def export_aux_dataset(
    samples: Sequence[EvaluationSample],
    output_dir: str | Path,
    *,
    config: PerceptionISPConfig | None = None,
    channels: Sequence[str] | None = None,
    include_extended: bool = True,
    include_preview: bool = True,
    compress: bool = True,
    run_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    destination = Path(output_dir).expanduser()
    tensor_dir = destination / "tensors"
    label_dir = destination / "labels"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    base_channels = tuple(str(value) for value in (channels or RGB_AUX_CHANNELS))

    manifest_rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        images = build_pipeline_images(sample, config=config)
        rgb_aux_hwc = build_rgb_aux_tensor(images, layout="hwc", channels=base_channels)
        rgb_aux_chw = build_rgb_aux_tensor(images, layout="chw", channels=base_channels)
        height, width = rgb_aux_hwc.shape[:2]
        safe_id = _safe_id(sample.sample_id, index)
        tensor_name = f"{index:05d}_{safe_id}.npz"
        label_name = f"{index:05d}_{safe_id}.json"
        tensor_path = tensor_dir / tensor_name
        label_path = label_dir / label_name

        boxes_xyxy = boxes_xyxy_array(sample.ground_truth)
        boxes_xyxy_norm = normalized_boxes_xyxy_array(sample.ground_truth, width=width, height=height)
        tensor_payload = {
            "rgb_aux_hwc": rgb_aux_hwc,
            "rgb_aux_chw": rgb_aux_chw,
            "boxes_xyxy": boxes_xyxy,
            "boxes_xyxy_normalized": boxes_xyxy_norm,
            "channel_names": np.asarray(base_channels),
        }
        if include_extended:
            tensor_payload["rgb_aux_extended_hwc"] = build_rgb_aux_extended_tensor(images, layout="hwc")
            tensor_payload["rgb_aux_extended_chw"] = build_rgb_aux_extended_tensor(images, layout="chw")
            tensor_payload["extended_channel_names"] = np.asarray(RGB_AUX_EXTENDED_CHANNELS)
        if include_preview:
            tensor_payload["perception_rgb_hwc"] = np.asarray(images.perception_rgb, dtype=np.float32)
            tensor_payload["perception_aux_hwc"] = np.asarray(images.perception_aux_rgb, dtype=np.float32)
        _save_npz(tensor_path, tensor_payload, compress=compress)
        labels = label_payload(sample.ground_truth)
        labels["boxes_xyxy"] = boxes_xyxy.tolist()
        labels["boxes_xyxy_normalized"] = boxes_xyxy_norm.tolist()
        labels["sample_id"] = str(sample.sample_id)
        labels["source"] = str(sample.source)
        label_path.write_text(json.dumps(json_ready(labels), indent=2) + "\n")

        raw_provenance = _raw_provenance(sample, images.metadata)
        row = {
            "sample_id": str(sample.sample_id),
            "source": str(sample.source),
            "tensor_path": str(Path("tensors") / tensor_name),
            "label_path": str(Path("labels") / label_name),
            "width": int(width),
            "height": int(height),
            "channels": list(base_channels),
            "extended_channels": list(RGB_AUX_EXTENDED_CHANNELS) if include_extended else [],
            "box_count": int(len(sample.ground_truth)),
            "tensor_stats": tensor_stats(rgb_aux_chw, channels=base_channels),
            "raw_provenance": raw_provenance,
            "metadata": dict(sample.metadata),
            "isp_processing": dict(images.metadata.get("processing", {})) if isinstance(images.metadata, Mapping) else {},
        }
        if include_extended:
            row["extended_tensor_stats"] = tensor_stats(tensor_payload["rgb_aux_extended_chw"], channels=RGB_AUX_EXTENDED_CHANNELS)
        manifest_rows.append(row)

    manifest_path = destination / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(json_ready(row)) + "\n" for row in manifest_rows))
    elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-9)
    summary = _summary(
        manifest_rows,
        run_config=run_config,
        elapsed_seconds=elapsed_seconds,
        channels=base_channels,
        include_extended=include_extended,
        include_preview=include_preview,
        compress=compress,
    )
    (destination / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / "index.html").write_text(_render_html(summary, manifest_rows))
    return summary


def _save_npz(path: Path, payload: Mapping[str, Any], *, compress: bool) -> None:
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _load_samples(
    *,
    source: str,
    dataset: str | None,
    split: str,
    count: int,
    offset: int = 0,
    width: int,
    height: int,
    cfa_pattern: str,
    use_camerae2e: bool,
    progress_interval: int = 0,
    cache_dir: str | Path | None = None,
    pascalraw_manifest: str | Path | None = None,
    pascalraw_native_raw: bool = False,
) -> Sequence[EvaluationSample]:
    if source == "synthetic":
        from perception_isp.evaluation.synthetic_eval import make_synthetic_evaluation_samples

        return make_synthetic_evaluation_samples(count=count, width=width, height=height, cfa_pattern="RGGB" if cfa_pattern == "auto" else cfa_pattern)
    if source == "camerae2e-synthetic":
        from perception_isp.evaluation.synthetic_eval import make_camerae2e_synthetic_evaluation_samples

        return make_camerae2e_synthetic_evaluation_samples(count=count, width=width, height=height, cfa_pattern=cfa_pattern)
    if source == "kitti-dataset":
        if not dataset:
            raise ValueError("--dataset is required for --source kitti-dataset")
        from perception_isp.datasets.kitti_dataset import load_kitti_detection_samples

        return load_kitti_detection_samples(
            dataset,
            split=split,
            limit=count,
            offset=offset,
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            use_camerae2e=use_camerae2e,
            progress_interval=progress_interval,
            progress_label=f"load:kitti-dataset:{offset}+{count}",
            cache_dir=cache_dir,
        )
    if source == "pascalraw-dataset":
        if not dataset:
            raise ValueError("--dataset is required for --source pascalraw-dataset")
        if not pascalraw_manifest:
            raise ValueError("--pascalraw-manifest is required for --source pascalraw-dataset")
        if bool(pascalraw_native_raw):
            from perception_isp.datasets.pascalraw_loader import load_pascalraw_native_detection_samples

            return load_pascalraw_native_detection_samples(
                dataset,
                pascalraw_manifest,
                limit=count,
                offset=offset,
                width=width,
                height=height,
                progress_interval=progress_interval,
                progress_label=f"load:pascalraw-dataset:native:{offset}+{count}",
            )
        from perception_isp.datasets.pascalraw_loader import load_pascalraw_detection_samples

        return load_pascalraw_detection_samples(
            dataset,
            pascalraw_manifest,
            limit=count,
            offset=offset,
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            use_camerae2e=use_camerae2e,
            progress_interval=progress_interval,
            progress_label=f"load:pascalraw-dataset:{offset}+{count}",
        )
    if not dataset:
        raise ValueError("--dataset is required for --source yolo-dataset")
    from perception_isp.datasets.yolo_dataset import load_yolo_detection_samples

    return load_yolo_detection_samples(
        dataset,
        split=split,
        limit=count,
        offset=offset,
        width=width,
        height=height,
        cfa_pattern=cfa_pattern,
        use_camerae2e=use_camerae2e,
        progress_interval=progress_interval,
        progress_label=f"load:yolo-dataset:{offset}+{count}",
        cache_dir=cache_dir,
    )


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_config: Mapping[str, Any] | None,
    elapsed_seconds: float,
    channels: Sequence[str] | None = None,
    include_extended: bool = True,
    include_preview: bool = True,
    compress: bool = True,
) -> Dict[str, Any]:
    raw = [row.get("raw_provenance", {}) for row in rows]
    sample_count = int(len(rows))
    seconds = max(float(elapsed_seconds), 1.0e-9)
    tensor_layouts = ["rgb_aux_hwc", "rgb_aux_chw"]
    if include_extended:
        tensor_layouts.extend(["rgb_aux_extended_hwc", "rgb_aux_extended_chw"])
    if include_preview:
        tensor_layouts.extend(["perception_rgb_hwc", "perception_aux_hwc"])
    base_channels = tuple(str(value) for value in (channels or RGB_AUX_CHANNELS))
    return {
        "sample_count": sample_count,
        "channels": list(base_channels),
        "extended_channels": list(RGB_AUX_EXTENDED_CHANNELS) if include_extended else [],
        "tensor_layouts": tensor_layouts,
        "export_options": {
            "include_extended": bool(include_extended),
            "include_preview": bool(include_preview),
            "compress": bool(compress),
        },
        "manifest": "manifest.jsonl",
        "run_config": dict(run_config or {}),
        "total_boxes": int(sum(int(row.get("box_count", 0)) for row in rows)),
        "elapsed_seconds": seconds,
        "samples_per_second": float(sample_count / seconds) if sample_count else 0.0,
        "seconds_per_sample": float(seconds / sample_count) if sample_count else None,
        "raw_provenance": {
            "true_sensor_cfa_mosaic_count": sum(1 for item in raw if isinstance(item, Mapping) and item.get("true_sensor_cfa_mosaic")),
            "pattern_remapped_count": sum(1 for item in raw if isinstance(item, Mapping) and item.get("pattern_remapped")),
            "source_patterns": sorted({str(item.get("source_cfa_pattern")) for item in raw if isinstance(item, Mapping)}),
            "target_patterns": sorted({str(item.get("target_cfa_pattern")) for item in raw if isinstance(item, Mapping)}),
        },
    }


def _render_html(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        provenance = row.get("raw_provenance", {})
        table_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('sample_id', '')))}</td>"
            f"<td>{html_lib.escape(str(row.get('width', '')))}x{html_lib.escape(str(row.get('height', '')))}</td>"
            f"<td>{int(row.get('box_count', 0))}</td>"
            f"<td>{html_lib.escape(str(provenance.get('source_cfa_pattern', '')))}</td>"
            f"<td>{html_lib.escape(str(provenance.get('target_cfa_pattern', '')))}</td>"
            f"<td>{html_lib.escape(str(provenance.get('pattern_remapped', '')))}</td>"
            f"<td><code>{html_lib.escape(str(row.get('tensor_path', '')))}</code></td>"
            "</tr>"
        )
    channels = ", ".join(str(value) for value in summary.get("channels", ()))
    extended_channels = ", ".join(str(value) for value in summary.get("extended_channels", ()))
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP RGB+Aux DNN Export</title>
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
  <h1>PerceptionISP RGB+Aux DNN Export</h1>
  <div class=\"note\">This export is the DNN-facing path. The base tensor is stored as HWC/CHW for training: <code>{html_lib.escape(channels)}</code>. When enabled, a sensor-native extended tensor is also stored for broader aux-aware experiments: <code>{html_lib.escape(extended_channels)}</code>.</div>
  <p>Samples: {int(summary.get('sample_count', 0))}, boxes: {int(summary.get('total_boxes', 0))}, export: {float(summary.get('elapsed_seconds', 0.0)):.2f}s ({float(summary.get('samples_per_second', 0.0)):.2f} samples/s). Manifest: <code>manifest.jsonl</code>.</p>
  <table>
    <thead><tr><th>Sample</th><th>Shape</th><th>Boxes</th><th>Source CFA</th><th>Target CFA</th><th>Remapped</th><th>Tensor</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
  <p>Summary JSON: <code>summary.json</code></p>
</body>
</html>
"""


def _raw_provenance(sample: EvaluationSample, isp_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = sample.metadata
    if isinstance(metadata, Mapping) and isinstance(metadata.get("raw_provenance"), Mapping):
        return metadata["raw_provenance"]
    if isinstance(isp_metadata, Mapping) and isinstance(isp_metadata.get("raw_provenance"), Mapping):
        return isp_metadata["raw_provenance"]
    return {}


def _safe_id(value: str, index: int) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or f"sample_{index}"


def _parse_channel_list(value: str | None) -> Sequence[str] | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    alias = normalized.lower().replace("-", "_")
    if alias in {"stable", "stable6", "default", "rgb_aux"}:
        return RGB_AUX_CHANNELS
    if alias in {"edge", "edge6", "edge_evidence6"}:
        return EDGE6_CHANNELS
    return tuple(part.strip() for part in normalized.split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
