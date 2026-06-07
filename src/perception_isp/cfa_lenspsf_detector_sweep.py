"""CFA/LensPSF detector-condition sweep for PerceptionISP evidence."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .comparison import compare_dataset, write_comparison_report
from .detectors import UltralyticsYOLODetector, detector_from_name, rgb_aux_detector_from_checkpoint
from .eval_cli import apply_psf_sigma_to_samples, parse_label_map, remap_sample_labels
from .eval_types import EvaluationSample
from .proposal_calibration import load_proposal_calibration_artifact, proposal_calibration_run_config
from .types import PerceptionISPConfig, json_ready


SUMMARY_FILENAME = "cfa_lenspsf_detector_sweep_summary.json"
TRACKED_METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "det_count_mean",
)
INPUT_ORDER = (
    "reference_rgb",
    "human_rgb",
    "perception_rgb",
    "perception_fusion_rgb_aux",
    "perception_calibrated_fusion_rgb_aux",
    "perception_calibrated_score_aux_fusion_rgb_aux",
    "perception_calibrated_score_label_fusion_rgb_aux",
    "perception_calibrated_score_label_aux_fusion_rgb_aux",
    "perception_rgb_aux_dnn",
    "perception_aux_rgb",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run detector comparisons across CFA and LensPSF conditions.")
    parser.add_argument("--source", choices=("yolo-dataset", "kitti-dataset"), default="yolo-dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cfa", action="append", default=None, help="CFA condition. Repeatable; defaults to auto.")
    parser.add_argument("--psf-sigma", action="append", type=float, default=None, help="LensPSF sigma. Repeatable; defaults to 0.")
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--rgb-detector", default="yolo")
    parser.add_argument("--rgb-detector-model", default="yolo11n.pt")
    parser.add_argument("--rgb-detector-confidence", type=float, default=0.25)
    parser.add_argument("--aux-detector", default="aux")
    parser.add_argument("--rgb-aux-detector-checkpoint", default=None)
    parser.add_argument("--rgb-aux-detector-confidence", type=float, default=None)
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--human-tone-mapping", default="log")
    parser.add_argument("--human-denoise-strength", type=float, default=0.18)
    parser.add_argument("--human-demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--human-demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--label-aware", action="store_true")
    parser.add_argument("--ground-truth-label-map", default=None, help="Comma-separated src=dst labels, or preset 'kitti-coco'.")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--proposal-calibration-model", default=None)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--load-progress-interval", type=int, default=0)
    parser.add_argument("--raw-cache-dir", default=None)
    parser.add_argument("--output-dir", default="reports/perception_cfa_lenspsf_detector_sweep")
    args = parser.parse_args(argv)
    if args.proposal_calibration_model and bool(args.no_fusion):
        raise ValueError("--proposal-calibration-model requires fusion; remove --no-fusion")

    cfa_patterns = parse_cfa_patterns(args.cfa or ("auto",))
    psf_sigmas = parse_psf_sigmas(args.psf_sigma or (0.0,))
    label_map = parse_label_map(args.ground_truth_label_map)
    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    human_config = PerceptionISPConfig(
        tone_mapping=str(args.human_tone_mapping),
        denoise_strength=float(args.human_denoise_strength),
        demosaic_method=str(args.human_demosaic_method),
        demosaic_artifact_suppression=float(args.human_demosaic_artifact_suppression),
    )
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
    proposal_calibration_artifact = (
        load_proposal_calibration_artifact(args.proposal_calibration_model)
        if args.proposal_calibration_model
        else None
    )

    destination = Path(args.output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    condition_count = len(cfa_patterns) * len(psf_sigmas)
    runs = []
    condition_index = 0
    for cfa_pattern in cfa_patterns:
        base_samples = _load_samples(
            source=str(args.source),
            dataset=str(args.dataset),
            split=str(args.split),
            count=int(args.count),
            offset=int(args.offset),
            width=int(args.width),
            height=int(args.height),
            cfa_pattern=str(cfa_pattern),
            use_camerae2e=not bool(args.no_camerae2e),
            progress_interval=int(args.load_progress_interval),
            progress_label=f"load:cfa-psf:{cfa_pattern}",
            cache_dir=args.raw_cache_dir,
        )
        if label_map:
            base_samples = remap_sample_labels(base_samples, label_map)
        for psf_sigma in psf_sigmas:
            condition_index += 1
            run_id = condition_id(cfa_pattern, psf_sigma)
            run_dir = destination / f"{condition_index:03d}_{run_id}"
            samples = apply_psf_sigma_to_samples(base_samples, psf_sigma)
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
                proposal_calibration_artifact=proposal_calibration_artifact,
                progress_interval=int(args.progress_interval),
                progress_label=f"cfa-psf:{condition_index}/{condition_count}:{run_id}",
            )
            result["run_config"] = _run_config(
                args=args,
                cfa_pattern=str(cfa_pattern),
                psf_sigma=float(psf_sigma),
                condition_index=condition_index,
                condition_count=condition_count,
                config=config,
                human_config=human_config,
                label_map=label_map,
            )
            if proposal_calibration_artifact is not None:
                result["run_config"]["proposal_calibration"] = proposal_calibration_run_config(proposal_calibration_artifact)
            report_path = write_comparison_report(result, run_dir)
            runs.append(summarize_condition_run(result, report_path.relative_to(destination)))

    summary = build_sweep_summary(
        args=args,
        runs=runs,
        cfa_patterns=cfa_patterns,
        psf_sigmas=psf_sigmas,
        label_map=label_map,
        config=config,
        human_config=human_config,
    )
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / "index.html").write_text(_render_html(summary, destination))
    print(json.dumps(json_ready({"report": str(destination / "index.html"), "summary_json": str(destination / SUMMARY_FILENAME), "status": summary["status"]}), indent=2))
    return 0


def parse_cfa_patterns(values: Sequence[str]) -> Tuple[str, ...]:
    patterns = []
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        patterns.append("auto" if normalized.lower() == "auto" else normalized.upper().replace("-", ""))
    if not patterns:
        raise ValueError("at least one CFA pattern is required")
    return tuple(dict.fromkeys(patterns))


def parse_psf_sigmas(values: Sequence[float]) -> Tuple[float, ...]:
    sigmas = tuple(dict.fromkeys(max(float(value), 0.0) for value in values))
    if not sigmas:
        raise ValueError("at least one PSF sigma is required")
    return sigmas


def condition_id(cfa_pattern: str, psf_sigma: float) -> str:
    sigma = f"{float(psf_sigma):.2f}".replace(".", "p")
    return f"cfa-{_slug(cfa_pattern)}_psf-{sigma}"


def summarize_condition_run(result: Mapping[str, Any], report_path: Path) -> Dict[str, Any]:
    run_config = dict(result.get("run_config", {}))
    aggregate = result.get("aggregate", {})
    metrics = {
        input_name: {
            metric_name: float(aggregate.get(input_name, {}).get(metric_name, 0.0))
            for metric_name in TRACKED_METRICS
            if metric_name in aggregate.get(input_name, {})
        }
        for input_name in _input_names(aggregate)
    }
    return {
        "condition_index": int(run_config.get("condition_index", 0)),
        "run_id": str(run_config.get("run_id", report_path.parent.name)),
        "report": str(report_path),
        "cfa_pattern": str(run_config.get("cfa", "")),
        "psf_sigma": None if run_config.get("psf_sigma") is None else float(run_config.get("psf_sigma", 0.0)),
        "sample_count": int(result.get("sample_count", run_config.get("count", 0))),
        "raw_condition_summary": raw_condition_summary(result.get("samples", ())),
        "metrics": metrics,
        "delta_vs_human": _deltas_vs_human(metrics),
    }


def raw_condition_summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source_cfas = {}
    target_cfas = {}
    requested_cfas = {}
    psf_values = {}
    camerae2e_camera_types = {}
    native_cfa_bridge_versions = {}
    pattern_remapped_count = 0
    true_sensor_cfa_mosaic_count = 0
    psf_recorded_count = 0
    for sample in samples:
        metadata = sample.get("metadata", {})
        isp_metadata = sample.get("isp_metadata", {})
        raw_provenance = isp_metadata.get("raw_provenance", {})
        if not isinstance(raw_provenance, Mapping):
            raw_provenance = metadata.get("raw_provenance", {})
        _bump(source_cfas, raw_provenance.get("source_cfa_pattern"))
        _bump(target_cfas, raw_provenance.get("target_cfa_pattern", metadata.get("cfa_pattern")))
        _bump(requested_cfas, raw_provenance.get("requested_cfa_pattern", metadata.get("requested_cfa_pattern")))
        psf = raw_provenance.get("eval_psf_sigma", metadata.get("psf_sigma"))
        if psf is not None:
            psf_recorded_count += 1
            _bump(psf_values, f"{float(psf):.2f}")
        _bump(camerae2e_camera_types, raw_provenance.get("camerae2e_camera_type"))
        _bump(native_cfa_bridge_versions, raw_provenance.get("camerae2e_native_cfa_bridge_version"))
        pattern_remapped_count += int(bool(raw_provenance.get("pattern_remapped", False)))
        true_sensor_cfa_mosaic_count += int(bool(raw_provenance.get("true_sensor_cfa_mosaic", False)))
    total = int(len(samples))
    return {
        "sample_count": total,
        "source_cfa_patterns": source_cfas,
        "target_cfa_patterns": target_cfas,
        "requested_cfa_patterns": requested_cfas,
        "camerae2e_camera_types": camerae2e_camera_types,
        "camerae2e_native_cfa_bridge_versions": native_cfa_bridge_versions,
        "true_sensor_cfa_mosaic_count": int(true_sensor_cfa_mosaic_count),
        "true_sensor_cfa_mosaic_fraction": 0.0 if total <= 0 else float(true_sensor_cfa_mosaic_count / total),
        "psf_sigmas": psf_values,
        "pattern_remapped_count": int(pattern_remapped_count),
        "pattern_remapped_fraction": 0.0 if total <= 0 else float(pattern_remapped_count / total),
        "psf_recorded_count": int(psf_recorded_count),
        "psf_recorded_fraction": 0.0 if total <= 0 else float(psf_recorded_count / total),
    }


def build_sweep_summary(
    *,
    args: Any,
    runs: Sequence[Mapping[str, Any]],
    cfa_patterns: Sequence[str],
    psf_sigmas: Sequence[float],
    label_map: Mapping[str, str],
    config: PerceptionISPConfig,
    human_config: PerceptionISPConfig,
) -> Dict[str, Any]:
    run_list = [dict(run) for run in runs]
    checks = _checks(run_list, expected_run_count=len(cfa_patterns) * len(psf_sigmas))
    return {
        "name": "CFA/LensPSF detector sweep",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "warning",
        "run_count": len(run_list),
        "expected_run_count": int(len(cfa_patterns) * len(psf_sigmas)),
        "source": str(args.source),
        "dataset": str(args.dataset),
        "split": str(args.split),
        "count": int(args.count),
        "offset": int(args.offset),
        "width": int(args.width),
        "height": int(args.height),
        "cfa_patterns": list(cfa_patterns),
        "psf_sigmas": [float(value) for value in psf_sigmas],
        "use_camerae2e": not bool(args.no_camerae2e),
        "raw_cache_dir": args.raw_cache_dir,
        "label_agnostic": not bool(args.label_aware),
        "ground_truth_label_map": dict(label_map),
        "proposal_calibration_model": getattr(args, "proposal_calibration_model", None),
        "perception_config": _config_dict(config),
        "human_baseline_config": _config_dict(human_config),
        "checks": checks,
        "best": _best_summary(run_list),
        "rankings": _rankings(run_list),
        "runs": run_list,
        "interpretation": (
            "This report runs the same detector recipe across CFA and LensPSF conditions. "
            "It is condition-sensitivity evidence for detector outputs, not a broad HumanISP superiority proof."
        ),
        "claim_boundary": (
            "Use this as a detector-side CFA/LensPSF sweep. Explicit Bayer CFA requests can use native CameraE2E "
            "camera types, but older cached rows can still be bridge-remapped; check pattern_remapped_count, "
            "true_sensor_cfa_mosaic_fraction, and camerae2e_native_cfa_bridge_versions before making sensor-CFA claims."
        ),
    }


def _checks(runs: Sequence[Mapping[str, Any]], *, expected_run_count: int) -> Tuple[Dict[str, Any], ...]:
    total_samples = sum(int(run.get("raw_condition_summary", {}).get("sample_count", 0)) for run in runs)
    psf_recorded = sum(int(run.get("raw_condition_summary", {}).get("psf_recorded_count", 0)) for run in runs)
    return (
        {
            "id": "condition_grid_complete",
            "status": "pass" if int(len(runs)) == int(expected_run_count) else "fail",
            "evidence": f"runs={len(runs)} expected={expected_run_count}",
        },
        {
            "id": "psf_sigma_recorded_in_raw_provenance",
            "status": "pass" if total_samples > 0 and int(psf_recorded) == int(total_samples) else "fail",
            "evidence": f"recorded={psf_recorded} samples={total_samples}",
        },
        {
            "id": "detector_metrics_available",
            "status": "pass" if all(run.get("metrics", {}).get("human_rgb") for run in runs) else "fail",
            "evidence": f"metric_runs={sum(1 for run in runs if run.get('metrics', {}).get('human_rgb'))}/{len(runs)}",
        },
    )


def _best_summary(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    input_names = sorted({name for run in runs for name in run.get("delta_vs_human", {})})
    return {
        input_name: {
            "best_delta_recall@0.50": _best_run(runs, input_name, "recall@0.50_mean", higher_is_better=True),
            "best_delta_fp@0.50": _best_run(runs, input_name, "fp@0.50_mean", higher_is_better=False),
            "best_delta_precision@0.50": _best_run(runs, input_name, "precision@0.50_mean", higher_is_better=True),
        }
        for input_name in input_names
    }


def _rankings(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "calibrated_or_fusion_by_delta_fp@0.50": _rank_conditions(runs, _primary_downstream_input, "fp@0.50_mean", higher_is_better=False),
        "calibrated_or_fusion_by_delta_recall@0.50": _rank_conditions(runs, _primary_downstream_input, "recall@0.50_mean", higher_is_better=True),
        "perception_rgb_by_delta_recall@0.50": _rank_conditions(runs, lambda _: "perception_rgb", "recall@0.50_mean", higher_is_better=True),
    }


def _best_run(
    runs: Sequence[Mapping[str, Any]],
    input_name: str,
    metric_name: str,
    *,
    higher_is_better: bool,
) -> Dict[str, Any]:
    ranked = _rank_conditions(runs, lambda _: input_name, metric_name, higher_is_better=higher_is_better)
    return ranked[0] if ranked else {}


def _rank_conditions(
    runs: Sequence[Mapping[str, Any]],
    input_selector: Any,
    metric_name: str,
    *,
    higher_is_better: bool,
) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for run in runs:
        input_name = str(input_selector(run))
        value = run.get("delta_vs_human", {}).get(input_name, {}).get(metric_name)
        if value is None:
            continue
        rows.append(
            {
                "run_id": run.get("run_id"),
                "report": run.get("report"),
                "input": input_name,
                "cfa_pattern": run.get("cfa_pattern"),
                "psf_sigma": run.get("psf_sigma"),
                "delta": float(value),
            }
        )
    return tuple(sorted(rows, key=lambda row: float(row["delta"]), reverse=bool(higher_is_better)))


def _primary_downstream_input(run: Mapping[str, Any]) -> str:
    metrics = run.get("metrics", {})
    for name in _input_names(metrics):
        if name.startswith("perception_calibrated"):
            return name
    if "perception_fusion_rgb_aux" in metrics:
        return "perception_fusion_rgb_aux"
    return "perception_rgb"


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
    progress_interval: int,
    progress_label: str,
    cache_dir: str | Path | None,
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
            progress_interval=progress_interval,
            progress_label=progress_label,
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
        progress_interval=progress_interval,
        progress_label=progress_label,
        cache_dir=cache_dir,
    )


def _run_config(
    *,
    args: Any,
    cfa_pattern: str,
    psf_sigma: float,
    condition_index: int,
    condition_count: int,
    config: PerceptionISPConfig,
    human_config: PerceptionISPConfig,
    label_map: Mapping[str, str],
) -> Dict[str, Any]:
    run_id = condition_id(cfa_pattern, psf_sigma)
    return {
        "source": args.source,
        "dataset": args.dataset,
        "split": args.split,
        "count": int(args.count),
        "offset": int(args.offset),
        "width": int(args.width),
        "height": int(args.height),
        "cfa": str(cfa_pattern),
        "psf_sigma": float(psf_sigma),
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
        "proposal_calibration_model": getattr(args, "proposal_calibration_model", None),
        "load_progress_interval": int(args.load_progress_interval),
        "raw_cache_dir": args.raw_cache_dir,
        "ground_truth_label_map": dict(label_map),
        "condition_index": int(condition_index),
        "condition_count": int(condition_count),
        "run_id": run_id,
        "perception_config": _config_dict(config),
        "human_baseline_config": _config_dict(human_config),
    }


def _make_rgb_detector(*, name: str, model: str, confidence: float):
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"yolo", "ultralytics", "yolo11n"}:
        return UltralyticsYOLODetector(str(model), confidence=float(confidence))
    return detector_from_name(name)


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
            if metric_name in input_metrics and metric_name in human
        }
    return deltas


def _input_names(aggregate: Mapping[str, Any]) -> Tuple[str, ...]:
    names = {str(name) for name in aggregate}
    ordered = [name for name in INPUT_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in INPUT_ORDER))
    return tuple(ordered)


def _bump(counter: Dict[str, int], value: Any) -> None:
    if value is None:
        return
    key = str(value)
    counter[key] = int(counter.get(key, 0)) + 1


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{html_lib.escape(str(row.get('status', '')))}\">{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in summary.get("checks", ())
    )
    metric_rows = []
    for run in summary.get("runs", ()):
        raw_summary = run.get("raw_condition_summary", {})
        report = _relative_report_link(str(run.get("report", "")), destination)
        for input_name, metrics in run.get("metrics", {}).items():
            delta = run.get("delta_vs_human", {}).get(input_name, {})
            metric_rows.append(
                "<tr>"
                f"<td><a href=\"{report}\">{html_lib.escape(str(run.get('run_id', '')))}</a></td>"
                f"<td>{html_lib.escape(str(run.get('cfa_pattern', '')))}</td>"
                f"<td>{_fmt(run.get('psf_sigma'))}</td>"
                f"<td>{int(run.get('sample_count', 0))}</td>"
                f"<td>{_fmt(raw_summary.get('pattern_remapped_fraction'))}</td>"
                f"<td>{_fmt(raw_summary.get('true_sensor_cfa_mosaic_fraction'))}</td>"
                f"<td>{_fmt(raw_summary.get('psf_recorded_fraction'))}</td>"
                f"<td>{html_lib.escape(', '.join(str(value) for value in raw_summary.get('camerae2e_camera_types', {}).keys()) or 'n/a')}</td>"
                f"<td>{html_lib.escape(str(input_name))}</td>"
                f"<td>{_fmt(metrics.get('precision@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('small_recall@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('fp@0.50_mean'))}</td>"
                f"<td class=\"{_delta_class(delta.get('precision@0.50_mean'))}\">{_fmt_delta(delta.get('precision@0.50_mean'))}</td>"
                f"<td class=\"{_delta_class(delta.get('recall@0.50_mean'))}\">{_fmt_delta(delta.get('recall@0.50_mean'))}</td>"
                f"<td class=\"{_delta_class(delta.get('fp@0.50_mean'), lower_is_better=True)}\">{_fmt_delta(delta.get('fp@0.50_mean'))}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP CFA/LensPSF Detector Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e8f3f1; position: sticky; top: 0; }}
    a {{ color: #155e75; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pass, .pos {{ color: #047857; font-weight: 650; }}
    .fail, .warning, .neg {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP CFA/LensPSF Detector Sweep</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p><strong>Status:</strong> <code>{html_lib.escape(str(summary.get('status', '')))}</code>.
  Dataset: <code>{html_lib.escape(str(summary.get('dataset', '')))}</code> / <code>{html_lib.escape(str(summary.get('split', '')))}</code>,
  samples={int(summary.get('count', 0))}, size={int(summary.get('width', 0))}x{int(summary.get('height', 0))},
  CFA=<code>{html_lib.escape(', '.join(str(value) for value in summary.get('cfa_patterns', ())))}</code>,
  PSF=<code>{html_lib.escape(', '.join(str(value) for value in summary.get('psf_sigmas', ())))}</code>.</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Condition Metrics</h2>
  <table>
    <thead><tr><th>Run</th><th>CFA</th><th>PSF</th><th>Samples</th><th>Remap Frac</th><th>True CFA Frac</th><th>PSF Rec Frac</th><th>CameraE2E Type</th><th>Input</th><th>P50</th><th>R50</th><th>Small R50</th><th>FP50</th><th>dP50</th><th>dR50</th><th>dFP50</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _relative_report_link(report: str, destination: Path) -> str:
    if not report:
        return ""
    return html_lib.escape(os.path.relpath(str(destination / report), start=str(destination)))


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


def _delta_class(value: Any, *, lower_is_better: bool = False) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if numeric == 0.0:
        return ""
    positive = numeric > 0.0
    if lower_is_better:
        positive = not positive
    return "pos" if positive else "neg"


def _slug(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value).lower()).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
