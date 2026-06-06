"""CLI for HumanISP vs PerceptionISP perception comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .comparison import compare_dataset, write_comparison_report
from .detectors import UltralyticsYOLODetector, detector_from_name, rgb_aux_detector_from_checkpoint
from .eval_types import BoundingBox, EvaluationSample
from .proposal_calibration import load_proposal_calibration_artifact, proposal_calibration_run_config
from .synthetic_eval import make_camerae2e_synthetic_evaluation_samples, make_synthetic_evaluation_samples
from .types import PerceptionISPConfig, json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compare HumanISP and PerceptionISP perception outputs.")
    parser.add_argument("--source", choices=["synthetic", "camerae2e-synthetic", "sample-image", "yolo-dataset", "kitti-dataset"], default="synthetic")
    parser.add_argument("--image-url", default=None, help="Image URL for --source sample-image.")
    parser.add_argument("--dataset", default=None, help="YOLO dataset root or data.yaml for --source yolo-dataset.")
    parser.add_argument("--split", default="val", help="Dataset split for --source yolo-dataset.")
    parser.add_argument("--no-camerae2e", action="store_true", help="Use direct RGB remosaic instead of CameraE2E for dataset sources.")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many dataset images before applying --count.")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--cfa", default="auto", help="auto uses CameraE2E sensor-native CFA when available.")
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

    config = PerceptionISPConfig(
        tone_mapping=args.tone_mapping,
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
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
        "dataset": args.dataset,
        "split": args.split,
        "use_camerae2e": not bool(args.no_camerae2e),
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
        "tone_mapping": args.tone_mapping,
        "denoise_strength": float(args.denoise_strength),
        "demosaic_method": str(args.demosaic_method),
        "demosaic_artifact_suppression": float(args.demosaic_artifact_suppression),
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


if __name__ == "__main__":
    raise SystemExit(main())
