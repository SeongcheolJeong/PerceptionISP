"""CLI for HumanISP vs PerceptionISP perception comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .comparison import compare_dataset, write_comparison_report
from .detectors import UltralyticsYOLODetector, detector_from_name, rgb_aux_detector_from_checkpoint
from .eval_types import BoundingBox, EvaluationSample
from .proposal_calibration import load_proposal_calibration_artifact, proposal_calibration_run_config
from .synthetic_eval import make_camerae2e_synthetic_evaluation_samples, make_synthetic_evaluation_samples
from .types import PerceptionISPConfig, json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compare HumanISP and PerceptionISP perception outputs.")
    parser.add_argument(
        "--source",
        choices=[
            "synthetic",
            "camerae2e-synthetic",
            "sample-image",
            "yolo-dataset",
            "kitti-dataset",
            "aodraw-dataset",
            "pascalraw-dataset",
        ],
        default="synthetic",
    )
    parser.add_argument("--image-url", default=None, help="Image URL for --source sample-image.")
    parser.add_argument("--dataset", default=None, help="Dataset root or data.yaml for dataset sources.")
    parser.add_argument("--split", default="val", help="Dataset split for --source yolo-dataset.")
    parser.add_argument("--aodraw-manifest", default=None, help="AODRaw subset manifest JSON for --source aodraw-dataset.")
    parser.add_argument("--aodraw-cfa", default="RGGB", help="AODRaw Bayer CFA pattern. Use verified sensor metadata for claim runs.")
    parser.add_argument("--aodraw-black-level", type=float, default=None, help="Optional AODRaw black level override.")
    parser.add_argument("--aodraw-white-level", type=float, default=None, help="Optional AODRaw white level override.")
    parser.add_argument("--aodraw-require-srgb", action="store_true", help="Require paired AODRaw sRGB files for reference RGB metrics.")
    parser.add_argument("--pascalraw-manifest", default=None, help="PASCALRAW subset manifest JSON for --source pascalraw-dataset.")
    parser.add_argument("--no-camerae2e", action="store_true", help="Use direct RGB remosaic instead of CameraE2E for dataset sources.")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many dataset images before applying --count.")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--cfa", default="auto", help="auto uses CameraE2E sensor-native CFA when available.")
    parser.add_argument(
        "--psf-sigma",
        type=float,
        default=None,
        help="Optional LensPSF sigma in sensor pixels injected into RAW calibration.",
    )
    parser.add_argument("--rgb-detector", default="numpy", help="numpy or yolo")
    parser.add_argument("--rgb-detector-model", default="yolo11n.pt", help="Model path/name for --rgb-detector yolo.")
    parser.add_argument("--rgb-detector-confidence", type=float, default=0.25, help="Confidence threshold for --rgb-detector yolo.")
    parser.add_argument("--aux-detector", default="aux", help="aux or numpy")
    parser.add_argument("--rgb-aux-detector-checkpoint", default=None, help="Optional RGB+aux detector checkpoint.")
    parser.add_argument("--rgb-aux-detector-confidence", type=float, default=None)
    parser.add_argument("--tone-mapping", default="log", help="log, detector_log, srgb, gamma, or linear.")
    parser.add_argument("--denoise-strength", type=float, default=0.18)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=["edge_aware", "bilinear"], help="Bayer demosaic method.")
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--human-tone-mapping", default=None, help="Optional fixed HumanISP baseline tone mapping.")
    parser.add_argument("--human-denoise-strength", type=float, default=None, help="Optional fixed HumanISP baseline denoise strength.")
    parser.add_argument("--human-demosaic-method", default=None, choices=["edge_aware", "bilinear"], help="Optional fixed HumanISP baseline demosaic method.")
    parser.add_argument("--human-demosaic-artifact-suppression", type=float, default=None, help="Optional fixed HumanISP baseline demosaic artifact suppression.")
    parser.add_argument("--output-dir", default="reports/perception_compare")
    parser.add_argument("--label-aware", action="store_true", help="Require exact class labels during metric matching.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip overlay PNG generation in the HTML report.")
    parser.add_argument("--no-fusion", action="store_true", help="Skip the Perception RGB+Aux fusion metric.")
    parser.add_argument("--proposal-calibration-model", default=None, help="Optional proposal_calibration_model.json for calibrated RGB+Aux fusion.")
    parser.add_argument("--fusion-low-score-threshold", type=float, default=0.42, help="Suppress RGB detections below this score when aux support is weak.")
    parser.add_argument("--fusion-low-support-threshold", type=float, default=0.16, help="Weak aux support threshold for low-score RGB suppression.")
    parser.add_argument("--fusion-score-gain", type=float, default=0.08, help="Score gain for detections with strong aux support.")
    parser.add_argument("--fusion-score-penalty", type=float, default=0.06, help="Score penalty for low-score detections with weak aux support.")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print progress every N samples to stderr; 0 disables progress logging.")
    parser.add_argument("--load-progress-interval", type=int, default=0, help="Print dataset loading progress every N samples; 0 disables progress logging.")
    parser.add_argument("--raw-cache-dir", default=None, help="Optional directory for cached dataset RAW samples.")
    parser.add_argument("--ground-truth-label-map", default=None, help="Comma-separated src=dst labels, or preset 'kitti-coco'.")
    parser.add_argument(
        "--ground-truth-label-keep",
        default=None,
        help="Comma-separated labels to keep after mapping, or preset 'coco80'/'aodraw-coco-overlap'. Empty samples are dropped.",
    )
    args = parser.parse_args(argv)
    if args.proposal_calibration_model and bool(args.no_fusion):
        raise ValueError("--proposal-calibration-model requires fusion; remove --no-fusion")

    rgb_detector = (
        UltralyticsYOLODetector(str(args.rgb_detector_model), confidence=float(args.rgb_detector_confidence))
        if str(args.rgb_detector).lower().replace("-", "_") in {"yolo", "ultralytics", "yolo11n"}
        else detector_from_name(args.rgb_detector)
    )
    aux_detector = detector_from_name(args.aux_detector)
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

    if args.source == "camerae2e-synthetic":
        samples = make_camerae2e_synthetic_evaluation_samples(
            count=args.count,
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
        )
    elif args.source == "sample-image":
        from .image_eval import DEFAULT_SAMPLE_IMAGE_URL, make_yolo_pseudo_label_sample

        samples = (
            make_yolo_pseudo_label_sample(
                rgb_detector,
                url=args.image_url or DEFAULT_SAMPLE_IMAGE_URL,
                width=args.width,
                height=args.height,
                cfa_pattern=args.cfa,
            ),
        )
    elif args.source == "yolo-dataset":
        if not args.dataset:
            raise ValueError("--dataset is required when --source yolo-dataset")
        from .yolo_dataset import load_yolo_detection_samples

        samples = load_yolo_detection_samples(
            args.dataset,
            split=args.split,
            limit=args.count,
            offset=int(args.offset),
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
            use_camerae2e=not bool(args.no_camerae2e),
            progress_interval=int(args.load_progress_interval),
            progress_label=f"load:{args.source}:{args.offset}+{args.count}",
            cache_dir=args.raw_cache_dir,
        )
    elif args.source == "kitti-dataset":
        if not args.dataset:
            raise ValueError("--dataset is required when --source kitti-dataset")
        from .kitti_dataset import load_kitti_detection_samples

        samples = load_kitti_detection_samples(
            args.dataset,
            split=args.split,
            limit=args.count,
            offset=int(args.offset),
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
            use_camerae2e=not bool(args.no_camerae2e),
            progress_interval=int(args.load_progress_interval),
            progress_label=f"load:{args.source}:{args.offset}+{args.count}",
            cache_dir=args.raw_cache_dir,
        )
    elif args.source == "aodraw-dataset":
        if not args.dataset:
            raise ValueError("--dataset is required when --source aodraw-dataset")
        if not args.aodraw_manifest:
            raise ValueError("--aodraw-manifest is required when --source aodraw-dataset")
        from .aodraw_loader import load_aodraw_detection_samples

        samples = load_aodraw_detection_samples(
            args.dataset,
            args.aodraw_manifest,
            limit=args.count,
            offset=int(args.offset),
            width=args.width,
            height=args.height,
            cfa_pattern=args.aodraw_cfa,
            black_level=args.aodraw_black_level,
            white_level=args.aodraw_white_level,
            require_srgb=bool(args.aodraw_require_srgb),
            progress_interval=int(args.load_progress_interval),
            progress_label=f"load:{args.source}:{args.offset}+{args.count}",
        )
    elif args.source == "pascalraw-dataset":
        if not args.dataset:
            raise ValueError("--dataset is required when --source pascalraw-dataset")
        if not args.pascalraw_manifest:
            raise ValueError("--pascalraw-manifest is required when --source pascalraw-dataset")
        from .pascalraw_loader import load_pascalraw_detection_samples

        samples = load_pascalraw_detection_samples(
            args.dataset,
            args.pascalraw_manifest,
            limit=args.count,
            offset=int(args.offset),
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
            use_camerae2e=not bool(args.no_camerae2e),
            progress_interval=int(args.load_progress_interval),
            progress_label=f"load:{args.source}:{args.offset}+{args.count}",
        )
    else:
        samples = make_synthetic_evaluation_samples(
            count=args.count,
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
        )
    label_map = parse_label_map(args.ground_truth_label_map)
    if label_map:
        samples = remap_sample_labels(samples, label_map)
    label_keep = parse_label_keep(args.ground_truth_label_keep)
    if label_keep:
        samples = filter_sample_labels(samples, label_keep)
        if not samples:
            raise ValueError("--ground-truth-label-keep removed every sample; choose a broader label set")
    samples = apply_psf_sigma_to_samples(samples, args.psf_sigma)

    config = PerceptionISPConfig(
        tone_mapping=args.tone_mapping,
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    human_config = human_config_from_args(args)
    fusion_options = {
        "low_score_threshold": float(args.fusion_low_score_threshold),
        "low_support_threshold": float(args.fusion_low_support_threshold),
        "score_gain": float(args.fusion_score_gain),
        "score_penalty": float(args.fusion_score_penalty),
    }
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
        fusion_options=fusion_options,
        progress_interval=int(args.progress_interval),
        progress_label=f"{args.source}:{args.offset}+{args.count}",
        proposal_calibration_artifact=proposal_calibration_artifact,
    )
    result["run_config"] = {
        "source": args.source,
        "count": int(args.count),
        "offset": int(args.offset),
        "width": int(args.width),
        "height": int(args.height),
        "cfa": str(args.cfa),
        "psf_sigma": None if args.psf_sigma is None else max(float(args.psf_sigma), 0.0),
        "dataset": args.dataset,
        "split": args.split,
        "use_camerae2e": False if args.source == "aodraw-dataset" else not bool(args.no_camerae2e),
        "aodraw_manifest": args.aodraw_manifest,
        "aodraw_cfa": str(args.aodraw_cfa),
        "aodraw_black_level": args.aodraw_black_level,
        "aodraw_white_level": args.aodraw_white_level,
        "aodraw_require_srgb": bool(args.aodraw_require_srgb),
        "pascalraw_manifest": args.pascalraw_manifest,
        "rgb_detector": rgb_detector.name,
        "rgb_detector_model": args.rgb_detector_model,
        "rgb_detector_confidence": float(args.rgb_detector_confidence),
        "aux_detector": aux_detector.name,
        "rgb_aux_detector": None if rgb_aux_detector is None else rgb_aux_detector.name,
        "rgb_aux_detector_checkpoint": args.rgb_aux_detector_checkpoint,
        "rgb_aux_detector_confidence": args.rgb_aux_detector_confidence,
        "label_agnostic": not bool(args.label_aware),
        "visuals": not bool(args.no_visuals),
        "fusion": not bool(args.no_fusion),
        "fusion_options": fusion_options,
        "proposal_calibration_model": args.proposal_calibration_model,
        "progress_interval": int(args.progress_interval),
        "load_progress_interval": int(args.load_progress_interval),
        "raw_cache_dir": args.raw_cache_dir,
        "ground_truth_label_map": label_map,
        "ground_truth_label_keep": list(label_keep),
        "tone_mapping": args.tone_mapping,
        "denoise_strength": float(args.denoise_strength),
        "demosaic_method": str(args.demosaic_method),
        "demosaic_artifact_suppression": float(args.demosaic_artifact_suppression),
        "human_baseline_config": None if human_config is None else config_to_dict(human_config),
    }
    if proposal_calibration_artifact is not None:
        result["run_config"]["proposal_calibration"] = proposal_calibration_run_config(proposal_calibration_artifact)
    html_path = write_comparison_report(result, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "aggregate": result["aggregate"], "run_config": result["run_config"]}), indent=2))
    return 0


def parse_label_map(value: str | None) -> dict[str, str]:
    if value is None or not str(value).strip():
        return {}
    normalized = str(value).strip()
    if normalized.lower().replace("_", "-") in {"kitti-coco", "kitti-to-coco"}:
        return {
            "car": "car",
            "van": "car",
            "truck": "truck",
            "pedestrian": "person",
            "person_sitting": "person",
            "Person_sitting": "person",
            "cyclist": "bicycle",
            "tram": "train",
        }
    if normalized.lower().replace("_", "-") in {"aodraw-coco", "aodraw-to-coco"}:
        return {
            "traffic_light": "traffic light",
            "fire_hydrant": "fire hydrant",
            "stop_sign": "stop sign",
            "parking_meter": "parking meter",
            "cell_phone": "cell phone",
            "dining_table": "dining table",
            "potted_plant": "potted plant",
            "teddy_bear": "teddy bear",
            "hair_drier": "hair drier",
            "sports_ball": "sports ball",
            "baseball_bat": "baseball bat",
            "baseball_glove": "baseball glove",
            "tennis_racket": "tennis racket",
            "wine_glass": "wine glass",
            "hot_dog": "hot dog",
        }
    mapping: dict[str, str] = {}
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError("label map entries must be formatted as src=dst")
        source, target = (part.strip() for part in token.split("=", 1))
        if not source or not target:
            raise ValueError("label map entries must have non-empty src and dst labels")
        mapping[source] = target
    return mapping


def parse_label_keep(value: str | None) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return ()
    normalized = str(value).strip()
    preset = normalized.lower().replace("_", "-")
    if preset in {"coco80", "coco-80", "aodraw-coco-overlap", "aodraw-to-coco-overlap"}:
        from .yolo_dataset import COCO80_CLASS_NAMES

        return tuple(str(label) for label in COCO80_CLASS_NAMES)
    labels = []
    seen = set()
    for token in normalized.split(","):
        label = token.strip()
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return tuple(labels)


def human_config_from_args(args: Any) -> PerceptionISPConfig | None:
    if (
        args.human_tone_mapping is None
        and args.human_denoise_strength is None
        and args.human_demosaic_method is None
        and args.human_demosaic_artifact_suppression is None
    ):
        return None
    defaults = PerceptionISPConfig()
    return PerceptionISPConfig(
        tone_mapping=str(args.human_tone_mapping or defaults.tone_mapping),
        denoise_strength=float(defaults.denoise_strength if args.human_denoise_strength is None else args.human_denoise_strength),
        demosaic_method=str(args.human_demosaic_method or defaults.demosaic_method),
        demosaic_artifact_suppression=float(
            defaults.demosaic_artifact_suppression
            if args.human_demosaic_artifact_suppression is None
            else args.human_demosaic_artifact_suppression
        ),
    )


def config_to_dict(config: PerceptionISPConfig) -> dict[str, Any]:
    return {
        "tone_mapping": str(config.tone_mapping),
        "denoise_strength": float(config.denoise_strength),
        "demosaic_method": str(config.demosaic_method),
        "demosaic_artifact_suppression": float(config.demosaic_artifact_suppression),
    }


def remap_sample_labels(samples: Sequence[EvaluationSample], label_map: Mapping[str, str]) -> tuple[EvaluationSample, ...]:
    if not label_map:
        return tuple(samples)
    remapped = []
    for sample in samples:
        boxes = []
        changed = 0
        for box in sample.ground_truth:
            label = str(label_map.get(box.label, box.label))
            changed += int(label != box.label)
            boxes.append(BoundingBox(box.xyxy, label=label))
        metadata = dict(sample.metadata)
        metadata["ground_truth_label_map"] = dict(label_map)
        metadata["ground_truth_label_remapped_count"] = int(changed)
        remapped.append(replace(sample, ground_truth=tuple(boxes), metadata=metadata))
    return tuple(remapped)


def filter_sample_labels(
    samples: Sequence[EvaluationSample],
    keep_labels: Sequence[str],
    *,
    keep_empty: bool = False,
) -> tuple[EvaluationSample, ...]:
    keep = {str(label) for label in keep_labels}
    if not keep:
        return tuple(samples)
    filtered = []
    for sample in samples:
        boxes = tuple(box for box in sample.ground_truth if box.label in keep)
        removed = len(sample.ground_truth) - len(boxes)
        if not boxes and not keep_empty:
            continue
        metadata = dict(sample.metadata)
        metadata["ground_truth_label_keep"] = sorted(keep)
        metadata["ground_truth_label_filtered_count"] = int(removed)
        filtered.append(replace(sample, ground_truth=boxes, metadata=metadata))
    return tuple(filtered)


def apply_psf_sigma_to_samples(
    samples: Sequence[EvaluationSample],
    psf_sigma: float | None,
) -> tuple[EvaluationSample, ...]:
    if psf_sigma is None:
        return tuple(samples)
    sigma = max(float(psf_sigma), 0.0)
    conditioned = []
    for sample in samples:
        raw = sample.raw
        height, width = raw_height_width(raw.data)
        raw_metadata = replace(
            raw.metadata,
            calibration_id=f"{raw.metadata.calibration_id}_psf_{sigma:.2f}",
            lens_profile_id=f"{raw.metadata.lens_profile_id}_psf_{sigma:.2f}",
        )
        raw_calibration = replace(
            raw.calibration,
            psf_sigma_map=np.full((int(height), int(width)), sigma, dtype=np.float64),
        )
        conditioned_raw = replace(
            raw,
            metadata=raw_metadata,
            calibration=raw_calibration,
            provenance={**dict(raw.provenance), "eval_psf_sigma": sigma},
        )
        conditioned.append(
            replace(
                sample,
                raw=conditioned_raw,
                metadata={**dict(sample.metadata), "psf_sigma": sigma},
            )
        )
    return tuple(conditioned)


def raw_height_width(raw_data: Any) -> tuple[int, int]:
    arr = np.asarray(raw_data)
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        if arr.shape[2] <= 8 and arr.shape[1] > 8:
            return int(arr.shape[0]), int(arr.shape[1])
        if arr.shape[0] <= 8:
            return int(arr.shape[1]), int(arr.shape[2])
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"unsupported RAW shape for PSF conditioning: {arr.shape}")


if __name__ == "__main__":
    raise SystemExit(main())
