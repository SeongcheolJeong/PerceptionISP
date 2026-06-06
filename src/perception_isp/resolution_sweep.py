"""Resolution sweep runner for CameraE2E-backed perception evaluation."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .comparison import compare_dataset, write_comparison_report
from .detectors import detector_from_name
from .types import PerceptionISPConfig, json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run HumanISP vs PerceptionISP across scene/sensor resolutions.")
    parser.add_argument("--source", choices=["yolo-dataset", "kitti-dataset"], default="yolo-dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--resolutions", default="640x480,1280x960")
    parser.add_argument("--cfa", default="auto")
    parser.add_argument("--rgb-detector", default="yolo")
    parser.add_argument("--aux-detector", default="aux")
    parser.add_argument("--tone-mapping", default="srgb")
    parser.add_argument("--denoise-strength", type=float, default=0.18)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=["edge_aware", "bilinear"])
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.35)
    parser.add_argument("--label-aware", action="store_true")
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--load-progress-interval", type=int, default=0)
    parser.add_argument("--output-dir", default="reports/perception_resolution_sweep")
    args = parser.parse_args(argv)

    resolutions = parse_resolutions(args.resolutions)
    rgb_detector = detector_from_name(args.rgb_detector)
    aux_detector = detector_from_name(args.aux_detector)
    config = PerceptionISPConfig(
        tone_mapping=args.tone_mapping,
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    destination = Path(args.output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for width, height in resolutions:
        samples = _load_samples(
            source=args.source,
            dataset=args.dataset,
            split=args.split,
            count=args.count,
            width=width,
            height=height,
            cfa_pattern=args.cfa,
            use_camerae2e=not bool(args.no_camerae2e),
            progress_interval=int(args.load_progress_interval),
            progress_label=f"load:{args.source}:{width}x{height}",
        )
        result = compare_dataset(
            samples,
            rgb_detector=rgb_detector,
            aux_detector=aux_detector,
            config=config,
            label_agnostic=not bool(args.label_aware),
            include_images=not bool(args.no_visuals),
            include_fusion=not bool(args.no_fusion),
        )
        run_dir = destination / f"{width}x{height}"
        result["run_config"] = {
            "source": args.source,
            "dataset": args.dataset,
            "split": args.split,
            "count": int(args.count),
            "width": int(width),
            "height": int(height),
            "cfa": args.cfa,
            "use_camerae2e": not bool(args.no_camerae2e),
            "rgb_detector": rgb_detector.name,
            "aux_detector": aux_detector.name,
            "label_agnostic": not bool(args.label_aware),
            "visuals": not bool(args.no_visuals),
            "fusion": not bool(args.no_fusion),
            "load_progress_interval": int(args.load_progress_interval),
            "tone_mapping": args.tone_mapping,
            "denoise_strength": float(args.denoise_strength),
            "demosaic_method": str(args.demosaic_method),
            "demosaic_artifact_suppression": float(args.demosaic_artifact_suppression),
        }
        report_path = write_comparison_report(result, run_dir)
        runs.append(summarize_run(result, report_path.relative_to(destination)))

    summary = {
        "run_count": len(runs),
        "dataset": args.dataset,
        "source": args.source,
        "split": args.split,
        "count": int(args.count),
        "runs": runs,
    }
    (destination / "sweep_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / "index.html").write_text(_render_sweep_html(summary))
    print(json.dumps(json_ready({"report": str(destination / "index.html"), "summary": summary}), indent=2))
    return 0


def parse_resolutions(value: str) -> Tuple[Tuple[int, int], ...]:
    resolutions: List[Tuple[int, int]] = []
    for item in str(value).split(","):
        token = item.strip().lower().replace("*", "x")
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"Resolution must be WIDTHxHEIGHT: {item!r}")
        width_text, height_text = token.split("x", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError(f"Resolution dimensions must be positive: {item!r}")
        resolutions.append((width, height))
    if not resolutions:
        raise ValueError("At least one resolution is required")
    return tuple(resolutions)


def summarize_run(result: Mapping[str, Any], report_path: Path) -> Dict[str, Any]:
    run_config = dict(result.get("run_config", {}))
    aggregate = result.get("aggregate", {})
    raw_checks = [_raw_provenance_summary(sample) for sample in result.get("samples", ())]
    return {
        "resolution": f"{run_config.get('width')}x{run_config.get('height')}",
        "width": int(run_config.get("width", 0)),
        "height": int(run_config.get("height", 0)),
        "report": str(report_path),
        "metrics": {
            name: {
                "recall@0.50": float(aggregate.get(name, {}).get("recall@0.50_mean", 0.0)),
                "precision@0.50": float(aggregate.get(name, {}).get("precision@0.50_mean", 0.0)),
                "mean_recall": float(aggregate.get(name, {}).get("mean_recall_mean", 0.0)),
                "det_count": float(aggregate.get(name, {}).get("det_count_mean", 0.0)),
            }
            for name in ("reference_rgb", "human_rgb", "perception_rgb", "perception_fusion_rgb_aux", "perception_aux_rgb")
            if name in aggregate
        },
        "raw_provenance": {
            "sample_count": len(raw_checks),
            "true_sensor_cfa_mosaic_count": sum(1 for item in raw_checks if item.get("true_sensor_cfa_mosaic")),
            "pattern_remapped_count": sum(1 for item in raw_checks if item.get("pattern_remapped")),
            "native_resolution_matches_target_count": sum(1 for item in raw_checks if item.get("native_resolution_matches_target")),
            "native_resolution_at_least_target_count": sum(1 for item in raw_checks if item.get("native_resolution_at_least_target")),
            "source_shapes": sorted({str(item.get("source_shape")) for item in raw_checks}),
            "requested_patterns": sorted({str(item.get("requested_cfa_pattern")) for item in raw_checks}),
            "source_patterns": sorted({str(item.get("source_cfa_pattern")) for item in raw_checks}),
            "target_patterns": sorted({str(item.get("target_cfa_pattern")) for item in raw_checks}),
        },
    }


def _load_samples(
    *,
    source: str,
    dataset: str,
    split: str,
    count: int,
    width: int,
    height: int,
    cfa_pattern: str,
    use_camerae2e: bool,
    progress_interval: int = 0,
    progress_label: str = "load_resolution_sweep",
) -> Sequence[Any]:
    if source == "kitti-dataset":
        from .kitti_dataset import load_kitti_detection_samples

        return load_kitti_detection_samples(
            dataset,
            split=split,
            limit=count,
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            use_camerae2e=use_camerae2e,
            progress_interval=int(progress_interval),
            progress_label=str(progress_label),
        )
    from .yolo_dataset import load_yolo_detection_samples

    return load_yolo_detection_samples(
        dataset,
        split=split,
        limit=count,
        width=width,
        height=height,
        cfa_pattern=cfa_pattern,
        use_camerae2e=use_camerae2e,
        progress_interval=int(progress_interval),
        progress_label=str(progress_label),
    )


def _raw_provenance_summary(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, Mapping):
        provenance = metadata.get("raw_provenance", {})
        if isinstance(provenance, Mapping):
            return provenance
    isp_metadata = sample.get("isp_metadata", {})
    if isinstance(isp_metadata, Mapping):
        provenance = isp_metadata.get("raw_provenance", {})
        if isinstance(provenance, Mapping):
            return provenance
    return {}


def _render_sweep_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for run in summary.get("runs", ()):
        metrics = run.get("metrics", {})
        provenance = run.get("raw_provenance", {})
        link = html_lib.escape(str(run.get("report", "")))
        rows.append(
            "<tr>"
            f"<td><a href=\"{link}\">{html_lib.escape(str(run.get('resolution', '')))}</a></td>"
            f"<td>{metrics.get('reference_rgb', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{metrics.get('human_rgb', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{metrics.get('perception_rgb', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{metrics.get('perception_fusion_rgb_aux', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{metrics.get('perception_aux_rgb', {}).get('recall@0.50', 0.0):.3f}</td>"
            f"<td>{provenance.get('true_sensor_cfa_mosaic_count', 0)}/{provenance.get('sample_count', 0)}</td>"
            f"<td>{html_lib.escape(', '.join(provenance.get('source_patterns', ())))}</td>"
            f"<td>{html_lib.escape(', '.join(provenance.get('target_patterns', ())))}</td>"
            f"<td>{provenance.get('pattern_remapped_count', 0)}/{provenance.get('sample_count', 0)}</td>"
            f"<td>{provenance.get('native_resolution_matches_target_count', 0)}/{provenance.get('sample_count', 0)}</td>"
            f"<td>{provenance.get('native_resolution_at_least_target_count', 0)}/{provenance.get('sample_count', 0)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Resolution Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 10px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    a {{ color: #155e75; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Resolution Sweep</h1>
  <div class=\"note\">This report compares detector metrics across scene/sensor resolutions. For CameraE2E-backed evidence, True CFA and Native >= Target should equal sample count, and Remapped should normally be 0 when using sensor-native CFA.</div>
  <table>
    <thead><tr><th>Resolution</th><th>Reference Recall@0.50</th><th>Human Recall@0.50</th><th>Perception Recall@0.50</th><th>Fusion Recall@0.50</th><th>Aux Recall@0.50</th><th>True CFA</th><th>Source CFA</th><th>Target CFA</th><th>Remapped</th><th>Native Exact</th><th>Native >= Target</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>sweep_summary.json</code></p>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
