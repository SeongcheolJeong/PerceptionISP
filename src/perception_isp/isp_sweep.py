"""ISP tuning sweep for HumanISP-vs-PerceptionISP detector metrics."""

from __future__ import annotations

import argparse
import html as html_lib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .comparison import compare_dataset, write_comparison_report
from .detectors import UltralyticsYOLODetector, detector_from_name, rgb_aux_detector_from_checkpoint
from .eval_cli import parse_label_map, remap_sample_labels
from .eval_types import EvaluationSample
from .types import PerceptionISPConfig, json_ready


TRACKED_INPUTS = ("reference_rgb", "human_rgb", "perception_rgb", "perception_fusion_rgb_aux", "perception_aux_rgb")
TRACKED_METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "det_count_mean",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep PerceptionISP settings against a fixed HumanISP baseline.")
    parser.add_argument("--source", choices=["yolo-dataset", "kitti-dataset"], default="yolo-dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cfa", default="auto")
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--rgb-detector", default="yolo")
    parser.add_argument("--rgb-detector-model", default="yolo11n.pt")
    parser.add_argument("--rgb-detector-confidence", type=float, default=0.25)
    parser.add_argument("--aux-detector", default="aux")
    parser.add_argument("--rgb-aux-detector-checkpoint", default=None)
    parser.add_argument("--rgb-aux-detector-confidence", type=float, default=None)
    parser.add_argument("--tone-mappings", default="log,srgb,linear")
    parser.add_argument("--denoise-strengths", default="0.0,0.18,0.30")
    parser.add_argument("--demosaic-methods", default="edge_aware")
    parser.add_argument("--demosaic-artifact-suppressions", default="0.20")
    parser.add_argument("--human-tone-mapping", default="log")
    parser.add_argument("--human-denoise-strength", type=float, default=0.18)
    parser.add_argument("--human-demosaic-method", default="edge_aware", choices=["edge_aware", "bilinear"])
    parser.add_argument("--human-demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--label-aware", action="store_true")
    parser.add_argument("--ground-truth-label-map", default=None, help="Comma-separated src=dst labels, or preset 'kitti-coco'.")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--load-progress-interval", type=int, default=0)
    parser.add_argument("--raw-cache-dir", default=None)
    parser.add_argument("--output-dir", default="reports/perception_isp_sweep")
    args = parser.parse_args(argv)

    samples = _load_samples(
        source=args.source,
        dataset=args.dataset,
        split=args.split,
        count=int(args.count),
        offset=int(args.offset),
        width=int(args.width),
        height=int(args.height),
        cfa_pattern=str(args.cfa),
        use_camerae2e=not bool(args.no_camerae2e),
        progress_interval=int(args.load_progress_interval),
        progress_label=f"load:{args.source}:{args.offset}+{args.count}",
        cache_dir=args.raw_cache_dir,
    )
    label_map = parse_label_map(args.ground_truth_label_map)
    if label_map:
        samples = remap_sample_labels(samples, label_map)

    rgb_detector = _make_rgb_detector(
        name=str(args.rgb_detector),
        model=str(args.rgb_detector_model),
        confidence=float(args.rgb_detector_confidence),
    )
    aux_detector = detector_from_name(str(args.aux_detector))
    rgb_aux_detector = (
        rgb_aux_detector_from_checkpoint(
            args.rgb_aux_detector_checkpoint,
            confidence=args.rgb_aux_detector_confidence,
        )
        if args.rgb_aux_detector_checkpoint
        else None
    )
    human_config = PerceptionISPConfig(
        tone_mapping=str(args.human_tone_mapping),
        denoise_strength=float(args.human_denoise_strength),
        demosaic_method=str(args.human_demosaic_method),
        demosaic_artifact_suppression=float(args.human_demosaic_artifact_suppression),
    )
    configs = tuple(
        iter_sweep_configs(
            tone_mappings=parse_csv(args.tone_mappings),
            denoise_strengths=parse_float_csv(args.denoise_strengths),
            demosaic_methods=parse_csv(args.demosaic_methods),
            demosaic_artifact_suppressions=parse_float_csv(args.demosaic_artifact_suppressions),
        )
    )
    destination = Path(args.output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for run_index, config in enumerate(configs, start=1):
        run_id = config_id(config)
        run_dir = destination / f"{run_index:03d}_{run_id}"
        result = compare_dataset(
            samples,
            rgb_detector=rgb_detector,
            aux_detector=aux_detector,
            rgb_aux_detector=rgb_aux_detector,
            config=config,
            human_config=human_config,
            label_agnostic=not bool(args.label_aware),
            include_images=not bool(args.no_visuals),
            include_fusion=not bool(args.no_fusion),
            progress_interval=int(args.progress_interval),
            progress_label=f"isp-sweep:{run_index}/{len(configs)}:{run_id}",
        )
        result["run_config"] = _run_config(args, config, human_config, label_map, run_index, len(configs))
        report_path = write_comparison_report(result, run_dir)
        runs.append(summarize_run(result, report_path.relative_to(destination)))

    summary = build_sweep_summary(args, runs, label_map, human_config)
    (destination / "isp_sweep_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / "index.html").write_text(_render_sweep_html(summary))
    print(json.dumps(json_ready({"report": str(destination / "index.html"), "summary": summary}), indent=2))
    return 0


def parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not items:
        raise ValueError("at least one value is required")
    return items


def parse_float_csv(value: str) -> Tuple[float, ...]:
    return tuple(float(token) for token in parse_csv(value))


def iter_sweep_configs(
    *,
    tone_mappings: Sequence[str],
    denoise_strengths: Sequence[float],
    demosaic_methods: Sequence[str],
    demosaic_artifact_suppressions: Sequence[float],
) -> Tuple[PerceptionISPConfig, ...]:
    configs = []
    for tone_mapping, denoise_strength, demosaic_method, suppression in itertools.product(
        tone_mappings,
        denoise_strengths,
        demosaic_methods,
        demosaic_artifact_suppressions,
    ):
        configs.append(
            PerceptionISPConfig(
                tone_mapping=str(tone_mapping),
                denoise_strength=float(denoise_strength),
                demosaic_method=str(demosaic_method),
                demosaic_artifact_suppression=float(suppression),
            )
        )
    return tuple(configs)


def config_id(config: PerceptionISPConfig) -> str:
    return (
        f"tone-{_slug(config.tone_mapping)}"
        f"_denoise-{float(config.denoise_strength):.2f}"
        f"_demosaic-{_slug(config.demosaic_method)}"
        f"_artifact-{float(config.demosaic_artifact_suppression):.2f}"
    )


def summarize_run(result: Mapping[str, Any], report_path: Path) -> Dict[str, Any]:
    run_config = dict(result.get("run_config", {}))
    aggregate = result.get("aggregate", {})
    metrics = {
        input_name: {
            metric_name: float(aggregate.get(input_name, {}).get(metric_name, 0.0))
            for metric_name in TRACKED_METRICS
        }
        for input_name in TRACKED_INPUTS
        if input_name in aggregate
    }
    return {
        "run_index": int(run_config.get("run_index", 0)),
        "run_id": str(run_config.get("run_id", report_path.parent.name)),
        "report": str(report_path),
        "perception_config": dict(run_config.get("perception_config", {})),
        "human_baseline_config": dict(run_config.get("human_baseline_config", {})),
        "metrics": metrics,
        "delta_vs_human": _deltas_vs_human(metrics),
    }


def build_sweep_summary(
    args: Any,
    runs: Sequence[Mapping[str, Any]],
    label_map: Mapping[str, str],
    human_config: PerceptionISPConfig,
) -> Dict[str, Any]:
    run_list = [dict(run) for run in runs]
    return {
        "run_count": len(run_list),
        "source": str(args.source),
        "dataset": str(args.dataset),
        "split": str(args.split),
        "count": int(args.count),
        "offset": int(args.offset),
        "width": int(args.width),
        "height": int(args.height),
        "cfa": str(args.cfa),
        "use_camerae2e": not bool(args.no_camerae2e),
        "load_progress_interval": int(args.load_progress_interval),
        "raw_cache_dir": args.raw_cache_dir,
        "label_agnostic": not bool(args.label_aware),
        "ground_truth_label_map": dict(label_map),
        "human_baseline_config": _config_dict(human_config),
        "best": {
            "perception_rgb_by_delta_recall@0.50": _best_run(run_list, "perception_rgb", "recall@0.50_mean"),
            "perception_rgb_by_delta_small_recall@0.50": _best_run(run_list, "perception_rgb", "small_recall@0.50_mean"),
            "fusion_by_delta_recall@0.50": _best_run(run_list, "perception_fusion_rgb_aux", "recall@0.50_mean"),
        },
        "runs": run_list,
    }


def _load_samples(
    *,
    source: str,
    dataset: str,
    split: str,
    count: int,
    offset: int,
    width: int,
    height: int,
    cfa_pattern: str,
    use_camerae2e: bool,
    progress_interval: int = 0,
    progress_label: str = "load_samples",
    cache_dir: str | Path | None = None,
) -> Sequence[EvaluationSample]:
    if source == "kitti-dataset":
        from .kitti_dataset import load_kitti_detection_samples

        return load_kitti_detection_samples(
            dataset,
            split=split,
            limit=count,
            offset=offset,
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            use_camerae2e=use_camerae2e,
            progress_interval=int(progress_interval),
            progress_label=str(progress_label),
            cache_dir=cache_dir,
        )
    from .yolo_dataset import load_yolo_detection_samples

    return load_yolo_detection_samples(
        dataset,
        split=split,
        limit=count,
        offset=offset,
        width=width,
        height=height,
        cfa_pattern=cfa_pattern,
        use_camerae2e=use_camerae2e,
        progress_interval=int(progress_interval),
        progress_label=str(progress_label),
        cache_dir=cache_dir,
    )


def _make_rgb_detector(*, name: str, model: str, confidence: float):
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"yolo", "ultralytics", "yolo11n"}:
        return UltralyticsYOLODetector(str(model), confidence=float(confidence))
    return detector_from_name(name)


def _run_config(
    args: Any,
    config: PerceptionISPConfig,
    human_config: PerceptionISPConfig,
    label_map: Mapping[str, str],
    run_index: int,
    run_count: int,
) -> Dict[str, Any]:
    return {
        "source": args.source,
        "dataset": args.dataset,
        "split": args.split,
        "count": int(args.count),
        "offset": int(args.offset),
        "width": int(args.width),
        "height": int(args.height),
        "cfa": str(args.cfa),
        "use_camerae2e": not bool(args.no_camerae2e),
        "rgb_detector": str(args.rgb_detector),
        "rgb_detector_model": str(args.rgb_detector_model),
        "rgb_detector_confidence": float(args.rgb_detector_confidence),
        "aux_detector": str(args.aux_detector),
        "rgb_aux_detector_checkpoint": args.rgb_aux_detector_checkpoint,
        "rgb_aux_detector_confidence": args.rgb_aux_detector_confidence,
        "label_agnostic": not bool(args.label_aware),
        "visuals": not bool(args.no_visuals),
        "fusion": not bool(args.no_fusion),
        "load_progress_interval": int(args.load_progress_interval),
        "raw_cache_dir": args.raw_cache_dir,
        "ground_truth_label_map": dict(label_map),
        "run_index": int(run_index),
        "run_count": int(run_count),
        "run_id": config_id(config),
        "perception_config": _config_dict(config),
        "human_baseline_config": _config_dict(human_config),
    }


def _config_dict(config: PerceptionISPConfig) -> Dict[str, Any]:
    return {
        "tone_mapping": str(config.tone_mapping),
        "denoise_strength": float(config.denoise_strength),
        "demosaic_method": str(config.demosaic_method),
        "demosaic_artifact_suppression": float(config.demosaic_artifact_suppression),
    }


def _deltas_vs_human(metrics: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    human = metrics.get("human_rgb", {})
    deltas: Dict[str, Dict[str, float]] = {}
    for input_name, input_metrics in metrics.items():
        if input_name == "human_rgb":
            continue
        deltas[input_name] = {
            metric_name: float(input_metrics.get(metric_name, 0.0) - human.get(metric_name, 0.0))
            for metric_name in TRACKED_METRICS
        }
    return deltas


def _best_run(runs: Sequence[Mapping[str, Any]], input_name: str, metric_name: str) -> Dict[str, Any]:
    best = None
    best_value = None
    for run in runs:
        value = run.get("delta_vs_human", {}).get(input_name, {}).get(metric_name)
        if value is None:
            continue
        numeric = float(value)
        if best_value is None or numeric > best_value:
            best = run
            best_value = numeric
    if best is None:
        return {}
    return {
        "run_id": best.get("run_id"),
        "report": best.get("report"),
        "delta": float(best_value),
        "perception_config": dict(best.get("perception_config", {})),
    }


def _render_sweep_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for run in summary.get("runs", ()):
        metrics = run.get("metrics", {})
        delta = run.get("delta_vs_human", {})
        perception = metrics.get("perception_rgb", {})
        fusion = metrics.get("perception_fusion_rgb_aux", {})
        human = metrics.get("human_rgb", {})
        reference = metrics.get("reference_rgb", {})
        config = run.get("perception_config", {})
        report = html_lib.escape(str(run.get("report", "")))
        rows.append(
            "<tr>"
            f"<td><a href=\"{report}\">{html_lib.escape(str(run.get('run_id', '')))}</a></td>"
            f"<td>{html_lib.escape(_config_label(config))}</td>"
            f"<td>{_fmt(reference.get('recall@0.50_mean'))}</td>"
            f"<td>{_fmt(human.get('recall@0.50_mean'))}</td>"
            f"<td>{_fmt(perception.get('precision@0.50_mean'))}</td>"
            f"<td>{_fmt(perception.get('recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('perception_rgb', {}).get('recall@0.50_mean'))}\">{_fmt_delta(delta.get('perception_rgb', {}).get('recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('perception_rgb', {}).get('small_recall@0.50_mean'))}\">{_fmt_delta(delta.get('perception_rgb', {}).get('small_recall@0.50_mean'))}</td>"
            f"<td>{_fmt(fusion.get('recall@0.50_mean'))}</td>"
            f"<td class=\"{_delta_class(delta.get('perception_fusion_rgb_aux', {}).get('recall@0.50_mean'))}\">{_fmt_delta(delta.get('perception_fusion_rgb_aux', {}).get('recall@0.50_mean'))}</td>"
            f"<td>{_fmt(perception.get('det_count_mean'))}</td>"
            f"<td>{_fmt(perception.get('fp@0.50_mean'))}</td>"
            "</tr>"
        )
    best = summary.get("best", {})
    best_items = []
    for label, item in best.items():
        if not item:
            continue
        best_items.append(
            f"<li><strong>{html_lib.escape(str(label))}</strong>: "
            f"<a href=\"{html_lib.escape(str(item.get('report', '')))}\">{html_lib.escape(str(item.get('run_id', '')))}</a> "
            f"delta={float(item.get('delta', 0.0)):+.4f}</li>"
        )
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP ISP Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 9px; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #e8f3f1; position: sticky; top: 0; }}
    a {{ color: #155e75; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pos {{ color: #047857; font-weight: 650; }}
    .neg {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP ISP Sweep</h1>
  <div class=\"note\">HumanISP is evaluated with a fixed baseline config while PerceptionISP settings are swept. Use this as tuning evidence, not as a final performance claim until the winning setting is rerun on the full validation set.</div>
  <p><strong>Dataset:</strong> {html_lib.escape(str(summary.get('dataset', '')))} / {html_lib.escape(str(summary.get('split', '')))}, samples={int(summary.get('count', 0))}, size={int(summary.get('width', 0))}x{int(summary.get('height', 0))}, CameraE2E={bool(summary.get('use_camerae2e'))}</p>
  <h2>Best Runs</h2>
  <ul>{''.join(best_items)}</ul>
  <table>
    <thead><tr><th>Run</th><th>Perception Config</th><th>Reference R50</th><th>Human R50</th><th>Perception P50</th><th>Perception R50</th><th>Delta R50</th><th>Delta Small R50</th><th>Fusion R50</th><th>Fusion Delta R50</th><th>Det Count</th><th>FP50</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>isp_sweep_summary.json</code></p>
</body>
</html>
"""


def _config_label(config: Mapping[str, Any]) -> str:
    return (
        f"tone={config.get('tone_mapping')}, "
        f"denoise={float(config.get('denoise_strength', 0.0)):.2f}, "
        f"demosaic={config.get('demosaic_method')}, "
        f"artifact={float(config.get('demosaic_artifact_suppression', 0.0)):.2f}"
    )


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


def _delta_class(value: Any) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if numeric > 0.0:
        return "pos"
    if numeric < 0.0:
        return "neg"
    return ""


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value).lower()).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
