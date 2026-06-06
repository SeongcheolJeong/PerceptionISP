"""Evaluation contracts for HumanISP vs PerceptionISP experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .types import RawFrame


@dataclass(frozen=True)
class BoundingBox:
    """Continuous ``xyxy`` bounding box."""

    xyxy: Tuple[float, float, float, float]
    label: str = "object"

    def __post_init__(self) -> None:
        coords = tuple(float(v) for v in self.xyxy)
        if len(coords) != 4:
            raise ValueError("xyxy must contain four values")
        if coords[2] < coords[0] or coords[3] < coords[1]:
            raise ValueError("xyxy must be ordered as x1, y1, x2, y2")
        if not np.all(np.isfinite(coords)):
            raise ValueError("xyxy values must be finite")
        object.__setattr__(self, "xyxy", coords)
        object.__setattr__(self, "label", str(self.label))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {"xyxy": list(self.xyxy), "label": self.label, "area": self.area}


@dataclass(frozen=True)
class Detection:
    """Detector output."""

    box: BoundingBox
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }


@dataclass
class EvaluationSample:
    """One sample for ISP and perception comparison."""

    sample_id: str
    raw: RawFrame
    ground_truth: Tuple[BoundingBox, ...]
    source: str = "synthetic"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_rgb: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PipelineImageSet:
    """Images/tensors generated from one RAW sample."""

    human_rgb: np.ndarray
    perception_rgb: np.ndarray
    perception_aux_rgb: np.ndarray
    fast_tensor_preview: np.ndarray
    metadata: Mapping[str, Any]
    human_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorResult:
    """Named detector output for one pipeline image."""

    detector_name: str
    input_name: str
    detections: Tuple[Detection, ...]
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "input_name": self.input_name,
            "elapsed_ms": float(self.elapsed_ms),
            "detections": [item.to_dict() for item in self.detections],
        }
