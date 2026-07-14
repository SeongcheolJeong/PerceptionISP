"""Synthetic labeled samples for quick ISP/perception smoke tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from perception_isp.core.task_types import BoundingBox, EvaluationSample
from perception_isp.core.synthetic import make_synthetic_raw, make_synthetic_scene_rgb


def make_synthetic_evaluation_samples(
    count: int = 4,
    *,
    width: int = 320,
    height: int = 180,
    cfa_pattern: str = "RGGB",
) -> Tuple[EvaluationSample, ...]:
    """Return deterministic synthetic driving samples with approximate labels."""

    samples = []
    for index in range(max(int(count), 1)):
        raw = make_synthetic_raw(
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            frame_counter=index,
            timestamp_us=33333.0 * index,
            seed=7 + index * 11,
        )
        raw.metadata = replace(raw.metadata, camera_id="synthetic_eval_front")
        gt = _synthetic_boxes(width, height)
        samples.append(
            EvaluationSample(
                sample_id=f"synthetic_{index:04d}",
                raw=raw,
                ground_truth=gt,
                source="synthetic_eval",
                metadata={"width": int(width), "height": int(height), "cfa_pattern": cfa_pattern},
            )
        )
    return tuple(samples)


def make_camerae2e_synthetic_evaluation_samples(
    count: int = 2,
    *,
    width: int = 160,
    height: int = 90,
    cfa_pattern: str = "auto",
) -> Tuple[EvaluationSample, ...]:
    """Use CameraE2E on labeled synthetic RGB scenes, preserving GT boxes."""

    from perception_isp.core.camerae2e_bridge import raw_from_camerae2e_rgb

    samples = []
    for index in range(max(int(count), 1)):
        rgb = make_synthetic_scene_rgb(width=width, height=height, frame_counter=index, seed=31 + index * 13)
        raw = raw_from_camerae2e_rgb(rgb, width=width, height=height, cfa_pattern=cfa_pattern)
        raw.metadata = replace(
            raw.metadata,
            frame_counter=index,
            timestamp_us=33333.0 * index,
        )
        samples.append(
            EvaluationSample(
                sample_id=f"camerae2e_synthetic_{index:04d}",
                raw=raw,
                ground_truth=_synthetic_boxes(width, height),
                source="camerae2e_synthetic_eval",
                metadata={
                    "width": int(width),
                    "height": int(height),
                    "requested_cfa_pattern": cfa_pattern,
                    "cfa_pattern": raw.metadata.cfa_pattern,
                    "raw_provenance": dict(raw.provenance),
                },
                reference_rgb=rgb,
            )
        )
    return tuple(samples)


def _synthetic_boxes(width: int, height: int) -> Tuple[BoundingBox, ...]:
    w, h = float(width), float(height)
    return (
        BoundingBox((0.58 * w, 0.60 * h, 0.76 * w, 0.78 * h), label="car"),
        BoundingBox((0.36 * w, 0.535 * h, 0.41 * w, 0.82 * h), label="person"),
        BoundingBox((0.775 * w, 0.205 * h, 0.825 * w, 0.255 * h), label="traffic_light"),
    )
