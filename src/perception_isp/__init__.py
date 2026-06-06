"""Perception-oriented automotive ISP software reference."""

from .camerae2e_bridge import camerae2e_or_synthetic_raw, raw_from_camerae2e, raw_from_camerae2e_rgb
from .aux_dnn import RGB_AUX_CHANNELS, RGB_AUX_EXTENDED_CHANNELS, build_rgb_aux_extended_tensor, build_rgb_aux_tensor
from .comparison import build_pipeline_images, compare_dataset, compare_sample, write_comparison_report
from .detectors import AuxMapRiskDetector, NumpyRiskObjectDetector, RGBAuxTorchSmokeDetector, UltralyticsYOLODetector
from .eval_types import BoundingBox, Detection, EvaluationSample
from .image_eval import DEFAULT_SAMPLE_IMAGE_URL, make_yolo_pseudo_label_sample
from .kitti_dataset import load_kitti_detection_samples
from .yolo_dataset import load_yolo_detection_samples
from .pipeline import PerceptionISPPipeline, RuntimeController
from .synthetic import make_synthetic_raw, make_synthetic_scene_rgb
from .synthetic_eval import make_camerae2e_synthetic_evaluation_samples, make_synthetic_evaluation_samples
from .types import (
    AccuratePathOutput,
    CalibrationProfile,
    EdgePacket,
    FastPathOutput,
    PerceptionISPConfig,
    PerceptionISPResult,
    PreviousFrameState,
    RawFrame,
    RuntimeControlSuggestion,
    SensorMetadata,
)

__all__ = [
    "AccuratePathOutput",
    "CalibrationProfile",
    "EdgePacket",
    "FastPathOutput",
    "PerceptionISPConfig",
    "PerceptionISPResult",
    "PerceptionISPPipeline",
    "PreviousFrameState",
    "RawFrame",
    "RuntimeControlSuggestion",
    "RuntimeController",
    "SensorMetadata",
    "AuxMapRiskDetector",
    "BoundingBox",
    "DEFAULT_SAMPLE_IMAGE_URL",
    "Detection",
    "EvaluationSample",
    "NumpyRiskObjectDetector",
    "RGB_AUX_CHANNELS",
    "RGB_AUX_EXTENDED_CHANNELS",
    "RGBAuxTorchSmokeDetector",
    "UltralyticsYOLODetector",
    "build_pipeline_images",
    "build_rgb_aux_extended_tensor",
    "build_rgb_aux_tensor",
    "camerae2e_or_synthetic_raw",
    "compare_dataset",
    "compare_sample",
    "make_camerae2e_synthetic_evaluation_samples",
    "make_synthetic_evaluation_samples",
    "make_synthetic_raw",
    "make_synthetic_scene_rgb",
    "make_yolo_pseudo_label_sample",
    "raw_from_camerae2e",
    "raw_from_camerae2e_rgb",
    "load_kitti_detection_samples",
    "load_yolo_detection_samples",
    "write_comparison_report",
]
