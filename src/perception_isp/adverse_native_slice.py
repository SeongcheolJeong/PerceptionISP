"""Adverse-condition native RAW slice for PerceptionISP evidence."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .camerae2e_bridge import (
    CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION,
    raw_from_camerae2e_rgb,
    raw_from_rgb_direct,
)
from .cfa_lenspsf_detector_sweep import raw_condition_summary
from .comparison import compare_dataset, write_comparison_report
from .detectors import LabelMapDetector, UltralyticsYOLODetector, detector_from_name, rgb_aux_detector_from_checkpoint
from .eval_cli import apply_psf_sigma_to_samples, parse_label_map, remap_sample_labels
from .eval_types import EvaluationSample
from .proposal_calibration import load_proposal_calibration_artifact, proposal_calibration_run_config
from .sample_cache import load_cached_sample, sample_cache_key, save_cached_sample
from .types import PerceptionISPConfig, json_ready


SUMMARY_FILENAME = "adverse_native_slice_summary.json"
TRANSFORM_VERSION = "adverse_native_slice_v1"
DEFAULT_CONDITIONS = ("nominal", "night", "fog", "glare", "low_mtf", "hdr")
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
    parser = argparse.ArgumentParser(description="Run native CameraE2E RAW detector slices under simulated adverse conditions.")
    parser.add_argument("--source", choices=("yolo-dataset", "kitti-dataset"), default="yolo-dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--cfa", default="GRBG")
    parser.add_argument("--condition", action="append", default=None, help="Adverse condition. Repeatable.")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--psf-sigma", type=float, default=0.0)
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--rgb-detector", default="yolo")
    parser.add_argument("--rgb-detector-model", default="yolo11n.pt")
    parser.add_argument("--rgb-detector-confidence", type=float, default=0.25)
    parser.add_argument("--aux-detector", default="aux")
    parser.add_argument("--rgb-aux-detector-checkpoint", default=None)
    parser.add_argument("--rgb-aux-detector-confidence", type=float, default=None)
    parser.add_argument("--rgb-aux-detector-nms-iou", type=float, default=None)
    parser.add_argument("--rgb-aux-detector-max-detections", type=int, default=None)
    parser.add_argument("--rgb-aux-detector-device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--rgb-aux-detector-label-map",
        default=None,
        help="Optional label map for RGB+Aux detector outputs, using the same syntax as --ground-truth-label-map.",
    )
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--human-tone-mapping", default="log")
    parser.add_argument("--human-denoise-strength", type=float, default=0.18)
    parser.add_argument("--human-demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--human-demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--label-aware", action="store_true")
    parser.add_argument("--ground-truth-label-map", default=None)
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--proposal-calibration-model", default=None)
    parser.add_argument(
        "--primary-input",
        default=None,
        help=(
            "Input name used for aggregate adverse claim deltas. "
            "Defaults to calibrated fusion when available, then fusion, then perception_rgb."
        ),
    )
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--load-progress-interval", type=int, default=0)
    parser.add_argument("--raw-cache-dir", default=None, help="Cache for base dataset RAW samples.")
    parser.add_argument("--condition-raw-cache-dir", default=None, help="Cache for conditioned CameraE2E RAW samples.")
    parser.add_argument("--output-dir", default="reports/perception_adverse_native_slice")
    args = parser.parse_args(argv)
    if args.proposal_calibration_model and bool(args.no_fusion):
        raise ValueError("--proposal-calibration-model requires fusion; remove --no-fusion")

    conditions = parse_conditions(args.condition or DEFAULT_CONDITIONS)
    label_map = parse_label_map(args.ground_truth_label_map)
    rgb_aux_detector_label_map = parse_label_map(args.rgb_aux_detector_label_map)
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
            nms_iou=args.rgb_aux_detector_nms_iou,
            max_detections=args.rgb_aux_detector_max_detections,
            device=str(args.rgb_aux_detector_device),
        )
        if args.rgb_aux_detector_checkpoint
        else None
    )
    if rgb_aux_detector is not None and rgb_aux_detector_label_map:
        rgb_aux_detector = LabelMapDetector(rgb_aux_detector, rgb_aux_detector_label_map)
    proposal_calibration_artifact = (
        load_proposal_calibration_artifact(args.proposal_calibration_model)
        if args.proposal_calibration_model
        else None
    )

    base_samples = _load_base_samples(args)
    if label_map:
        base_samples = remap_sample_labels(base_samples, label_map)

    destination = Path(args.output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    for condition_index, condition in enumerate(conditions, start=1):
        run_id = condition_id(condition)
        run_dir = destination / f"{condition_index:03d}_{run_id}"
        conditioned = condition_samples(
            base_samples,
            condition=condition,
            cfa_pattern=str(args.cfa),
            width=int(args.width),
            height=int(args.height),
            severity=float(args.severity),
            use_camerae2e=not bool(args.no_camerae2e),
            cache_dir=args.condition_raw_cache_dir,
            progress_interval=int(args.load_progress_interval),
            progress_label=f"condition:{condition}",
        )
        conditioned = apply_psf_sigma_to_samples(conditioned, float(args.psf_sigma))
        result = compare_dataset(
            conditioned,
            rgb_detector=rgb_detector,
            aux_detector=aux_detector,
            rgb_aux_detector=rgb_aux_detector,
            config=config,
            human_config=human_config,
            label_agnostic=not bool(args.label_aware),
            include_images=not bool(args.no_visuals),
            include_fusion=not bool(args.no_fusion),
            progress_interval=int(args.progress_interval),
            progress_label=f"adverse:{condition_index}/{len(conditions)}:{condition}",
            proposal_calibration_artifact=proposal_calibration_artifact,
        )
        result["run_config"] = _run_config(
            args=args,
            condition=condition,
            condition_index=condition_index,
            condition_count=len(conditions),
            config=config,
            human_config=human_config,
            label_map=label_map,
        )
        if proposal_calibration_artifact is not None:
            result["run_config"]["proposal_calibration"] = proposal_calibration_run_config(proposal_calibration_artifact)
        report_path = write_comparison_report(result, run_dir)
        runs.append(summarize_condition_run(result, report_path.relative_to(destination)))

    summary = build_adverse_summary(
        args=args,
        runs=runs,
        conditions=conditions,
        label_map=label_map,
        rgb_aux_detector_label_map=rgb_aux_detector_label_map,
        config=config,
        human_config=human_config,
        primary_input=args.primary_input,
    )
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / "index.html").write_text(_render_html(summary, destination))
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(destination / "index.html"),
                    "summary_json": str(destination / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                }
            ),
            indent=2,
        )
    )
    return 0


def parse_conditions(values: Sequence[str]) -> Tuple[str, ...]:
    conditions = []
    for value in values:
        normalized = str(value).strip().lower().replace("-", "_")
        if not normalized:
            continue
        if normalized not in {"nominal", "night", "low_light", "fog", "glare", "low_mtf", "hdr", "rain"}:
            raise ValueError(f"unsupported adverse condition: {value}")
        conditions.append(normalized)
    if not conditions:
        raise ValueError("at least one adverse condition is required")
    return tuple(dict.fromkeys(conditions))


def condition_id(condition: str) -> str:
    return "condition-" + "".join(ch if ch.isalnum() else "-" for ch in str(condition).lower()).strip("-")


def condition_samples(
    samples: Sequence[EvaluationSample],
    *,
    condition: str,
    cfa_pattern: str,
    width: int,
    height: int,
    severity: float = 1.0,
    use_camerae2e: bool = True,
    cache_dir: str | Path | None = None,
    progress_interval: int = 0,
    progress_label: str = "condition_samples",
) -> Tuple[EvaluationSample, ...]:
    output = []
    total = int(len(samples))
    interval = max(int(progress_interval), 0)
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        conditioned = _condition_sample(
            sample,
            condition=condition,
            cfa_pattern=cfa_pattern,
            width=width,
            height=height,
            severity=severity,
            use_camerae2e=use_camerae2e,
            cache_dir=cache_dir,
            index=index,
        )
        output.append(conditioned)
        sample_index = index + 1
        if interval and (sample_index == total or sample_index % interval == 0):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            rate = float(sample_index / elapsed)
            remaining = float((total - sample_index) / max(rate, 1.0e-12))
            print(
                f"[{progress_label}] conditioned {sample_index}/{total} samples "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={remaining:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return tuple(output)


def apply_adverse_condition(rgb: Any, condition: str, *, severity: float = 1.0, seed: int = 0) -> np.ndarray:
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    normalized = str(condition).lower().replace("-", "_")
    sev = max(float(severity), 0.0)
    rng = np.random.default_rng(int(seed))
    if normalized == "nominal":
        return arr
    if normalized in {"night", "low_light"}:
        dark = arr * max(0.18, 0.30 / max(sev, 1.0e-6))
        noise = rng.normal(0.0, 0.018 * sev, size=arr.shape)
        return np.clip(dark + noise + 0.012, 0.0, 1.0)
    if normalized == "fog":
        height = int(arr.shape[0])
        haze = np.linspace(0.58, 0.20, height, dtype=np.float64)[:, None, None]
        haze = np.clip(haze * sev, 0.0, 0.82)
        veil = np.array([0.78, 0.80, 0.82], dtype=np.float64)
        return np.clip(arr * (1.0 - haze) + veil * haze, 0.0, 1.0)
    if normalized == "glare":
        height, width = int(arr.shape[0]), int(arr.shape[1])
        yy, xx = np.mgrid[0:height, 0:width]
        cx = width * 0.72
        cy = height * 0.34
        sigma = max(min(width, height) * 0.17, 1.0)
        spot = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)))[:, :, None]
        streak = np.exp(-((yy - cy) ** 2) / (2.0 * (sigma * 0.18) ** 2))[:, :, None] * 0.35
        return np.clip(arr + (spot * 0.85 + streak * 0.45) * sev, 0.0, 1.0)
    if normalized == "low_mtf":
        return _blur_rgb(arr, radius=1.4 * sev)
    if normalized == "hdr":
        tone = np.clip((arr - 0.38) * (1.0 + 0.9 * sev) + 0.38, 0.0, 1.0)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        yy, xx = np.mgrid[0:height, 0:width]
        cx = width * 0.18
        cy = height * 0.22
        sigma = max(min(width, height) * 0.11, 1.0)
        sun = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)))[:, :, None]
        return np.clip(tone + sun * 0.72 * sev, 0.0, 1.0)
    if normalized == "rain":
        rainy = np.clip(arr * 0.64 + 0.05, 0.0, 1.0)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        overlay = np.zeros_like(rainy)
        for _ in range(max(24, int(width * height / 2100))):
            x = int(rng.integers(0, max(width, 1)))
            y = int(rng.integers(0, max(height, 1)))
            length = int(max(6, min(width, height) * 0.08 * sev))
            for step in range(length):
                yy = y + step
                xx = x + step // 3
                if 0 <= yy < height and 0 <= xx < width:
                    overlay[yy, xx, :] = 1.0
        return np.clip(rainy + _blur_rgb(overlay, radius=0.55) * 0.42, 0.0, 1.0)
    raise ValueError(f"unsupported adverse condition: {condition}")


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
        "condition": str(run_config.get("adverse_condition", "")),
        "sample_count": int(result.get("sample_count", run_config.get("count", 0))),
        "raw_condition_summary": raw_condition_summary(result.get("samples", ())),
        "metrics": metrics,
        "delta_vs_human": _deltas_vs_human(metrics),
    }


def build_adverse_summary(
    *,
    args: Any,
    runs: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    label_map: Mapping[str, str],
    config: PerceptionISPConfig,
    human_config: PerceptionISPConfig,
    rgb_aux_detector_label_map: Mapping[str, str] | None = None,
    primary_input: str | None = None,
) -> Dict[str, Any]:
    run_list = [dict(run) for run in runs]
    selected_primary_input = _normalize_primary_input(primary_input)
    _validate_primary_input(run_list, selected_primary_input)
    aggregate = _aggregate_advantage(run_list, primary_input=selected_primary_input)
    checks = _checks(
        run_list,
        expected_run_count=len(conditions),
        aggregate=aggregate,
        primary_input=selected_primary_input,
    )
    return {
        "name": "Adverse-condition native RAW detector slice",
        "status": "pass" if all(row["status"] in {"pass", "warning"} for row in checks) else "fail",
        "claim_status": _claim_status(aggregate),
        "run_count": len(run_list),
        "expected_run_count": int(len(conditions)),
        "source": str(args.source),
        "dataset": str(args.dataset),
        "split": str(args.split),
        "count": int(args.count),
        "offset": int(args.offset),
        "width": int(args.width),
        "height": int(args.height),
        "cfa_pattern": str(args.cfa).upper().replace("-", ""),
        "conditions": list(conditions),
        "severity": float(args.severity),
        "psf_sigma": float(args.psf_sigma),
        "use_camerae2e": not bool(args.no_camerae2e),
        "raw_cache_dir": args.raw_cache_dir,
        "condition_raw_cache_dir": args.condition_raw_cache_dir,
        "label_agnostic": not bool(args.label_aware),
        "ground_truth_label_map": dict(label_map),
        "rgb_aux_detector_label_map": dict(rgb_aux_detector_label_map or {}),
        "proposal_calibration_model": getattr(args, "proposal_calibration_model", None),
        "primary_input": selected_primary_input or "auto",
        "perception_config": _config_dict(config),
        "human_baseline_config": _config_dict(human_config),
        "aggregate": aggregate,
        "checks": checks,
        "rankings": _rankings(run_list, primary_input=selected_primary_input),
        "runs": run_list,
        "interpretation": (
            "This report applies simulated adverse scene conditions before CameraE2E native CFA RAW generation, "
            "then runs the same HumanISP, PerceptionISP, and fixed detector recipe per condition."
        ),
        "claim_boundary": (
            "Use this as a simulated adverse native-RAW slice, not as proof on real night/rain/fog datasets. "
            "The ground truth boxes are inherited from the original labeled image; severe transforms can create label-visibility ambiguity."
        ),
    }


def _condition_sample(
    sample: EvaluationSample,
    *,
    condition: str,
    cfa_pattern: str,
    width: int,
    height: int,
    severity: float,
    use_camerae2e: bool,
    cache_dir: str | Path | None,
    index: int,
) -> EvaluationSample:
    if sample.reference_rgb is None:
        raise ValueError("adverse native slice requires reference_rgb in each sample")
    metadata = dict(sample.metadata)
    image_path = metadata.get("image_path", f"{sample.sample_id}.png")
    label_path = metadata.get("label_path", f"{sample.sample_id}.txt")
    key = sample_cache_key(
        namespace="adverse_native_slice",
        image_path=str(image_path),
        label_path=str(label_path),
        params={
            "condition": str(condition),
            "severity": float(severity),
            "width": int(width),
            "height": int(height),
            "cfa_pattern": str(cfa_pattern),
            "use_camerae2e": bool(use_camerae2e),
            "transform_version": TRANSFORM_VERSION,
            "camerae2e_native_cfa_bridge_version": CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION if use_camerae2e else "direct",
        },
    )
    cached = load_cached_sample(cache_dir, key)
    if cached is not None:
        return cached
    conditioned_rgb = apply_adverse_condition(
        sample.reference_rgb,
        condition,
        severity=severity,
        seed=int(metadata.get("dataset_index", index)),
    )
    raw = (
        raw_from_camerae2e_rgb(conditioned_rgb, width=width, height=height, cfa_pattern=cfa_pattern)
        if use_camerae2e
        else raw_from_rgb_direct(conditioned_rgb, width=width, height=height, cfa_pattern=cfa_pattern)
    )
    raw.metadata = replace(
        raw.metadata,
        frame_counter=int(getattr(sample.raw.metadata, "frame_counter", index)),
        timestamp_us=float(getattr(sample.raw.metadata, "timestamp_us", 33333.0 * index)),
        camera_id="camerae2e_adverse_native_slice" if use_camerae2e else "direct_adverse_native_slice",
        module_serial=f"{metadata.get('image_path', sample.sample_id)}:{condition}",
    )
    raw.provenance = {
        **dict(raw.provenance),
        "adverse_condition": str(condition),
        "adverse_severity": float(severity),
        "adverse_transform_version": TRANSFORM_VERSION,
    }
    conditioned_sample = EvaluationSample(
        sample_id=f"{sample.sample_id}_{condition}",
        raw=raw,
        ground_truth=tuple(sample.ground_truth),
        source=f"{sample.source}_adverse_native",
        metadata={
            **metadata,
            "sample_id_original": sample.sample_id,
            "adverse_condition": str(condition),
            "adverse_severity": float(severity),
            "adverse_transform_version": TRANSFORM_VERSION,
            "requested_cfa_pattern": str(cfa_pattern),
            "cfa_pattern": raw.metadata.cfa_pattern,
            "use_camerae2e": bool(use_camerae2e),
            "raw_provenance": dict(raw.provenance),
            "ground_truth_visibility_warning": "GT boxes inherited from original scene after simulated adverse transform",
        },
        reference_rgb=conditioned_rgb,
    )
    save_cached_sample(cache_dir, key, conditioned_sample)
    return conditioned_sample


def _load_base_samples(args: Any) -> Tuple[EvaluationSample, ...]:
    if str(args.source) == "kitti-dataset":
        from .kitti_dataset import load_kitti_detection_samples

        return load_kitti_detection_samples(
            args.dataset,
            split=str(args.split),
            limit=int(args.count),
            offset=int(args.offset),
            width=int(args.width),
            height=int(args.height),
            cfa_pattern=str(args.cfa),
            use_camerae2e=not bool(args.no_camerae2e),
            progress_interval=int(args.load_progress_interval),
            progress_label="load:adverse-base",
            cache_dir=args.raw_cache_dir,
        )
    from .yolo_dataset import load_yolo_detection_samples

    return load_yolo_detection_samples(
        args.dataset,
        split=str(args.split),
        limit=int(args.count),
        offset=int(args.offset),
        width=int(args.width),
        height=int(args.height),
        cfa_pattern=str(args.cfa),
        use_camerae2e=not bool(args.no_camerae2e),
        progress_interval=int(args.load_progress_interval),
        progress_label="load:adverse-base",
        cache_dir=args.raw_cache_dir,
    )


def _aggregate_advantage(runs: Sequence[Mapping[str, Any]], *, primary_input: str | None = None) -> Dict[str, Any]:
    rows = [_primary_delta_row(run, primary_input=primary_input) for run in runs]
    rows = [row for row in rows if row]
    adverse_rows = [row for row in rows if str(row.get("condition", "")) != "nominal"]
    fp_wins = [row for row in adverse_rows if float(row.get("delta_fp@0.50", 0.0)) < 0.0]
    recall_preserved = [row for row in adverse_rows if float(row.get("delta_recall@0.50", 0.0)) >= -0.01]
    joint = [
        row
        for row in adverse_rows
        if float(row.get("delta_fp@0.50", 0.0)) < 0.0 and float(row.get("delta_recall@0.50", 0.0)) >= -0.01
    ]
    return {
        "condition_count": int(len(rows)),
        "adverse_condition_count": int(len(adverse_rows)),
        "sample_count": int(sum(int(run.get("sample_count", 0)) for run in runs)),
        "primary_rows": rows,
        "adverse_fp_win_count": int(len(fp_wins)),
        "adverse_recall_preserved_count": int(len(recall_preserved)),
        "adverse_joint_fp_recall_win_count": int(len(joint)),
        "mean_adverse_delta_precision@0.50": _mean(adverse_rows, "delta_precision@0.50"),
        "mean_adverse_delta_recall@0.50": _mean(adverse_rows, "delta_recall@0.50"),
        "mean_adverse_delta_small_recall@0.50": _mean(adverse_rows, "delta_small_recall@0.50"),
        "mean_adverse_delta_fp@0.50": _mean(adverse_rows, "delta_fp@0.50"),
    }


def _primary_delta_row(run: Mapping[str, Any], *, primary_input: str | None = None) -> Dict[str, Any]:
    input_name = _primary_downstream_input(run, primary_input=primary_input)
    delta = run.get("delta_vs_human", {}).get(input_name)
    if not isinstance(delta, Mapping):
        return {}
    return {
        "condition": str(run.get("condition", "")),
        "run_id": str(run.get("run_id", "")),
        "input": input_name,
        "delta_precision@0.50": float(delta.get("precision@0.50_mean", 0.0)),
        "delta_recall@0.50": float(delta.get("recall@0.50_mean", 0.0)),
        "delta_small_recall@0.50": float(delta.get("small_recall@0.50_mean", 0.0)),
        "delta_fp@0.50": float(delta.get("fp@0.50_mean", 0.0)),
    }


def _checks(
    runs: Sequence[Mapping[str, Any]],
    *,
    expected_run_count: int,
    aggregate: Mapping[str, Any],
    primary_input: str | None = None,
) -> Tuple[Dict[str, Any], ...]:
    total_samples = sum(int(run.get("raw_condition_summary", {}).get("sample_count", 0)) for run in runs)
    true_native = sum(int(run.get("raw_condition_summary", {}).get("true_sensor_cfa_mosaic_count", 0)) for run in runs)
    remapped = sum(int(run.get("raw_condition_summary", {}).get("pattern_remapped_count", 0)) for run in runs)
    primary_available = _primary_input_available_count(runs, primary_input)
    return (
        {
            "id": "condition_grid_complete",
            "status": "pass" if int(len(runs)) == int(expected_run_count) else "fail",
            "evidence": f"runs={len(runs)} expected={expected_run_count}",
        },
        {
            "id": "native_camerae2e_raw_preserved",
            "status": "pass" if total_samples > 0 and true_native == total_samples and remapped == 0 else "fail",
            "evidence": f"true_native={true_native}/{total_samples} remapped={remapped}",
        },
        {
            "id": "detector_metrics_available",
            "status": "pass" if all(run.get("metrics", {}).get("human_rgb") for run in runs) else "fail",
            "evidence": f"metric_runs={sum(1 for run in runs if run.get('metrics', {}).get('human_rgb'))}/{len(runs)}",
        },
        {
            "id": "primary_input_metrics_available",
            "status": "pass" if primary_available == len(runs) else "fail",
            "evidence": (
                f"primary_input={primary_input or 'auto'} "
                f"metric_runs={primary_available}/{len(runs)}"
            ),
        },
        {
            "id": "adverse_conditions_included",
            "status": "pass" if int(aggregate.get("adverse_condition_count", 0)) > 0 else "fail",
            "evidence": f"adverse_conditions={int(aggregate.get('adverse_condition_count', 0))}",
        },
        {
            "id": "adverse_fp_reduction_observed",
            "status": "pass" if int(aggregate.get("adverse_fp_win_count", 0)) > 0 else "warning",
            "evidence": f"fp_win_conditions={int(aggregate.get('adverse_fp_win_count', 0))}/{int(aggregate.get('adverse_condition_count', 0))}",
        },
    )


def _claim_status(aggregate: Mapping[str, Any]) -> str:
    adverse = int(aggregate.get("adverse_condition_count", 0))
    joint = int(aggregate.get("adverse_joint_fp_recall_win_count", 0))
    fp_wins = int(aggregate.get("adverse_fp_win_count", 0))
    if adverse > 0 and joint >= max(1, adverse // 2):
        return "adverse_fp_reducer_supported"
    if fp_wins > 0:
        return "adverse_fp_recall_tradeoff"
    return "adverse_diagnostic_only"


def _rankings(runs: Sequence[Mapping[str, Any]], *, primary_input: str | None = None) -> Dict[str, Any]:
    return {
        "primary_by_delta_fp@0.50": _rank_conditions(
            runs,
            "fp@0.50_mean",
            input_name=primary_input,
            higher_is_better=False,
        ),
        "primary_by_delta_recall@0.50": _rank_conditions(
            runs,
            "recall@0.50_mean",
            input_name=primary_input,
            higher_is_better=True,
        ),
        "perception_rgb_by_delta_recall@0.50": _rank_conditions(runs, "recall@0.50_mean", input_name="perception_rgb", higher_is_better=True),
    }


def _rank_conditions(
    runs: Sequence[Mapping[str, Any]],
    metric_name: str,
    *,
    input_name: str | None = None,
    higher_is_better: bool,
) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for run in runs:
        selected = input_name or _primary_downstream_input(run)
        value = run.get("delta_vs_human", {}).get(selected, {}).get(metric_name)
        if value is None:
            continue
        rows.append(
            {
                "run_id": run.get("run_id"),
                "report": run.get("report"),
                "condition": run.get("condition"),
                "input": selected,
                "delta": float(value),
            }
        )
    return tuple(sorted(rows, key=lambda row: float(row["delta"]), reverse=bool(higher_is_better)))


def _run_config(
    *,
    args: Any,
    condition: str,
    condition_index: int,
    condition_count: int,
    config: PerceptionISPConfig,
    human_config: PerceptionISPConfig,
    label_map: Mapping[str, str],
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
        "psf_sigma": float(args.psf_sigma),
        "adverse_condition": str(condition),
        "adverse_severity": float(args.severity),
        "adverse_transform_version": TRANSFORM_VERSION,
        "use_camerae2e": not bool(args.no_camerae2e),
        "rgb_detector": str(args.rgb_detector),
        "rgb_detector_model": str(args.rgb_detector_model),
        "rgb_detector_confidence": float(args.rgb_detector_confidence),
        "aux_detector": str(args.aux_detector),
        "rgb_aux_detector_checkpoint": args.rgb_aux_detector_checkpoint,
        "rgb_aux_detector_confidence": args.rgb_aux_detector_confidence,
        "rgb_aux_detector_nms_iou": args.rgb_aux_detector_nms_iou,
        "rgb_aux_detector_max_detections": args.rgb_aux_detector_max_detections,
        "rgb_aux_detector_device": args.rgb_aux_detector_device,
        "rgb_aux_detector_label_map": getattr(args, "rgb_aux_detector_label_map", None),
        "label_agnostic": not bool(args.label_aware),
        "visuals": not bool(args.no_visuals),
        "fusion": not bool(args.no_fusion),
        "proposal_calibration_model": getattr(args, "proposal_calibration_model", None),
        "primary_input": getattr(args, "primary_input", None),
        "load_progress_interval": int(args.load_progress_interval),
        "raw_cache_dir": args.raw_cache_dir,
        "condition_raw_cache_dir": args.condition_raw_cache_dir,
        "ground_truth_label_map": dict(label_map),
        "condition_index": int(condition_index),
        "condition_count": int(condition_count),
        "run_id": condition_id(condition),
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


def _primary_downstream_input(run: Mapping[str, Any], *, primary_input: str | None = None) -> str:
    metrics = run.get("metrics", {})
    if primary_input:
        return str(primary_input)
    for name in _input_names(metrics):
        if name.startswith("perception_calibrated"):
            return name
    if "perception_fusion_rgb_aux" in metrics:
        return "perception_fusion_rgb_aux"
    return "perception_rgb"


def _normalize_primary_input(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized or normalized.lower() in {"auto", "default"}:
        return None
    return normalized


def _primary_input_available_count(runs: Sequence[Mapping[str, Any]], primary_input: str | None) -> int:
    if primary_input:
        return sum(1 for run in runs if isinstance(run.get("metrics"), Mapping) and primary_input in run.get("metrics", {}))
    return sum(
        1
        for run in runs
        if isinstance(run.get("delta_vs_human", {}).get(_primary_downstream_input(run)), Mapping)
    )


def _validate_primary_input(runs: Sequence[Mapping[str, Any]], primary_input: str | None) -> None:
    if not primary_input:
        return
    missing = [
        str(run.get("run_id", run.get("condition", index)))
        for index, run in enumerate(runs)
        if not isinstance(run.get("metrics"), Mapping) or primary_input not in run.get("metrics", {})
    ]
    if missing:
        preview = ", ".join(missing[:6])
        suffix = "" if len(missing) <= 6 else f", ... ({len(missing)} total)"
        raise ValueError(f"primary input {primary_input!r} missing from condition metrics: {preview}{suffix}")


def _input_names(aggregate: Mapping[str, Any]) -> Tuple[str, ...]:
    names = {str(name) for name in aggregate}
    ordered = [name for name in INPUT_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in INPUT_ORDER))
    return tuple(ordered)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows)) if rows else 0.0


def _blur_rgb(arr: np.ndarray, *, radius: float) -> np.ndarray:
    from PIL import Image, ImageFilter

    image = Image.fromarray(np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(float(radius), 0.0)))
    return np.asarray(blurred, dtype=np.float64) / 255.0


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
                f"<td>{html_lib.escape(str(run.get('condition', '')))}</td>"
                f"<td>{int(run.get('sample_count', 0))}</td>"
                f"<td>{_fmt(raw_summary.get('pattern_remapped_fraction'))}</td>"
                f"<td>{_fmt(raw_summary.get('true_sensor_cfa_mosaic_fraction'))}</td>"
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
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Adverse Native RAW Slice</title>
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
  <h1>PerceptionISP Adverse Native RAW Slice</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p><strong>Status:</strong> <code>{html_lib.escape(str(summary.get('status', '')))}</code>;
  claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.
  Dataset: <code>{html_lib.escape(str(summary.get('dataset', '')))}</code> / <code>{html_lib.escape(str(summary.get('split', '')))}</code>,
  samples/run={int(summary.get('count', 0))}, size={int(summary.get('width', 0))}x{int(summary.get('height', 0))},
  CFA=<code>{html_lib.escape(str(summary.get('cfa_pattern', '')))}</code>,
  primary=<code>{html_lib.escape(str(summary.get('primary_input', 'auto')))}</code>,
  conditions=<code>{html_lib.escape(', '.join(str(value) for value in summary.get('conditions', ())))}</code>.</p>
  <h2>Aggregate Adverse Primary Delta</h2>
  <table><tbody>
    <tr><th>Adverse conditions</th><td>{int(aggregate.get('adverse_condition_count', 0))}</td></tr>
    <tr><th>Adverse FP-win conditions</th><td>{int(aggregate.get('adverse_fp_win_count', 0))}</td></tr>
    <tr><th>Adverse recall-preserved conditions</th><td>{int(aggregate.get('adverse_recall_preserved_count', 0))}</td></tr>
    <tr><th>Mean adverse dP50</th><td>{_fmt_delta(aggregate.get('mean_adverse_delta_precision@0.50'))}</td></tr>
    <tr><th>Mean adverse dR50</th><td>{_fmt_delta(aggregate.get('mean_adverse_delta_recall@0.50'))}</td></tr>
    <tr><th>Mean adverse dFP50</th><td>{_fmt_delta(aggregate.get('mean_adverse_delta_fp@0.50'))}</td></tr>
  </tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Condition Metrics</h2>
  <table>
    <thead><tr><th>Run</th><th>Condition</th><th>Samples</th><th>Remap Frac</th><th>True CFA Frac</th><th>Input</th><th>P50</th><th>R50</th><th>Small R50</th><th>FP50</th><th>dP50</th><th>dR50</th><th>dFP50</th></tr></thead>
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


if __name__ == "__main__":
    raise SystemExit(main())
