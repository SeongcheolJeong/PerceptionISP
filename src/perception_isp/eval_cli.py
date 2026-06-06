"""CLI for HumanISP vs PerceptionISP perception comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .comparison import compare_dataset, write_comparison_report
from .detectors import AuxMapRiskDetector, NumpyRiskObjectDetector, RGBAuxTorchSmokeDetector, detector_from_name
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
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--cfa", default="auto", help="auto uses CameraE2E sensor-native CFA when available.")
    parser.add_argument("--rgb-detector", default="numpy", help="numpy or yolo")
    parser.add_argument("--aux-detector", default="aux", help="aux or numpy")
    parser.add_argument("--rgb-aux-detector-checkpoint", default=None, help="Optional RGB+aux smoke detector checkpoint.")
    parser.add_argument("--tone-mapping", default="log", help="log, srgb, gamma, or linear.")
    parser.add_argument("--denoise-strength", type=float, default=0.18)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=["edge_aware", "bilinear"], help="Bayer demosaic method.")
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--output-dir", default="reports/perception_compare")
    parser.add_argument("--label-aware", action="store_true", help="Require exact class labels during metric matching.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip overlay PNG generation in the HTML report.")
    parser.add_argument("--no-fusion", action="store_true", help="Skip the Perception RGB+Aux fusion metric.")
    args = parser.parse_args(argv)

    rgb_detector = detector_from_name(args.rgb_detector)
    aux_detector = detector_from_name(args.aux_detector)
    rgb_aux_detector = (
        RGBAuxTorchSmokeDetector(args.rgb_aux_detector_checkpoint)
        if args.rgb_aux_detector_checkpoint
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
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
            use_camerae2e=not bool(args.no_camerae2e),
        )
    elif args.source == "kitti-dataset":
        if not args.dataset:
            raise ValueError("--dataset is required when --source kitti-dataset")
        from .kitti_dataset import load_kitti_detection_samples

        samples = load_kitti_detection_samples(
            args.dataset,
            split=args.split,
            limit=args.count,
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
            use_camerae2e=not bool(args.no_camerae2e),
        )
    else:
        samples = make_synthetic_evaluation_samples(
            count=args.count,
            width=args.width,
            height=args.height,
            cfa_pattern=args.cfa,
        )

    config = PerceptionISPConfig(
        tone_mapping=args.tone_mapping,
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    result = compare_dataset(
        samples,
        rgb_detector=rgb_detector,
        aux_detector=aux_detector,
        rgb_aux_detector=rgb_aux_detector,
        config=config,
        label_agnostic=not bool(args.label_aware),
        include_images=not bool(args.no_visuals),
        include_fusion=not bool(args.no_fusion),
    )
    result["run_config"] = {
        "source": args.source,
        "count": int(args.count),
        "width": int(args.width),
        "height": int(args.height),
        "cfa": str(args.cfa),
        "dataset": args.dataset,
        "split": args.split,
        "use_camerae2e": not bool(args.no_camerae2e),
        "rgb_detector": rgb_detector.name,
        "aux_detector": aux_detector.name,
        "rgb_aux_detector": None if rgb_aux_detector is None else rgb_aux_detector.name,
        "rgb_aux_detector_checkpoint": args.rgb_aux_detector_checkpoint,
        "label_agnostic": not bool(args.label_aware),
        "visuals": not bool(args.no_visuals),
        "fusion": not bool(args.no_fusion),
        "tone_mapping": args.tone_mapping,
        "denoise_strength": float(args.denoise_strength),
        "demosaic_method": str(args.demosaic_method),
        "demosaic_artifact_suppression": float(args.demosaic_artifact_suppression),
    }
    html_path = write_comparison_report(result, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "aggregate": result["aggregate"], "run_config": result["run_config"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
