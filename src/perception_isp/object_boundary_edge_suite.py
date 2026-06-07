"""Object-box-boundary edge evidence for HumanISP and PerceptionISP."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .comparison import build_pipeline_images
from .eval_types import BoundingBox, EvaluationSample
from .types import PerceptionISPConfig, json_ready


OBJECT_BOUNDARY_EDGE_SUMMARY = "object_boundary_edge_summary.json"
SIGNALS = (
    "human_rgb_edge",
    "perception_rgb_edge",
    "aux_edge_strength",
    "aux_edge_confidence",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure edge evidence around GT object box boundaries.")
    parser.add_argument("--source", choices=("camerae2e-synthetic", "yolo-dataset", "kitti-dataset"), default="yolo-dataset")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--cfa", default="auto")
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--raw-cache-dir", default=None)
    parser.add_argument("--load-progress-interval", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--include-labels", default=None, help="Comma-separated labels to include, for example car,pedestrian,cyclist.")
    parser.add_argument("--boundary-thickness", type=int, default=2)
    parser.add_argument("--context-radius", type=int, default=10)
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--output-dir", default="reports/perception_object_boundary_edge_kitti")
    args = parser.parse_args(argv)

    samples = load_object_boundary_samples(
        source=str(args.source),
        dataset=args.dataset,
        split=str(args.split),
        count=int(args.count),
        offset=int(args.offset),
        width=int(args.width),
        height=int(args.height),
        cfa_pattern=str(args.cfa),
        use_camerae2e=not bool(args.no_camerae2e),
        cache_dir=args.raw_cache_dir,
        progress_interval=int(args.load_progress_interval),
    )
    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    summary = build_object_boundary_edge_suite(
        samples,
        config=config,
        include_labels=parse_label_list(args.include_labels),
        boundary_thickness=int(args.boundary_thickness),
        context_radius=int(args.context_radius),
        progress_interval=int(args.progress_interval),
    )
    html_path = write_object_boundary_edge_suite(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / OBJECT_BOUNDARY_EDGE_SUMMARY),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "sample_count": summary["sample_count"],
                    "box_count": summary["box_count"],
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def load_object_boundary_samples(
    *,
    source: str,
    dataset: str | None = None,
    split: str = "val",
    count: int = 64,
    offset: int = 0,
    width: int = 640,
    height: int = 192,
    cfa_pattern: str = "auto",
    use_camerae2e: bool = True,
    cache_dir: str | Path | None = None,
    progress_interval: int = 0,
) -> Tuple[EvaluationSample, ...]:
    normalized = str(source)
    if normalized == "camerae2e-synthetic":
        from .synthetic_eval import make_camerae2e_synthetic_evaluation_samples, make_synthetic_evaluation_samples

        if use_camerae2e:
            return make_camerae2e_synthetic_evaluation_samples(count=count, width=width, height=height, cfa_pattern=cfa_pattern)
        return make_synthetic_evaluation_samples(count=count, width=width, height=height, cfa_pattern="RGGB" if cfa_pattern == "auto" else cfa_pattern)
    if normalized == "yolo-dataset":
        if not dataset:
            raise ValueError("--dataset is required for --source yolo-dataset")
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
            cache_dir=cache_dir,
            progress_interval=progress_interval,
            progress_label=f"load:object-boundary:yolo:{offset}+{count}",
        )
    if normalized == "kitti-dataset":
        if not dataset:
            raise ValueError("--dataset is required for --source kitti-dataset")
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
            cache_dir=cache_dir,
            progress_interval=progress_interval,
            progress_label=f"load:object-boundary:kitti:{offset}+{count}",
        )
    raise ValueError(f"unsupported object-boundary source: {source!r}")


def build_object_boundary_edge_suite(
    samples: Sequence[EvaluationSample],
    *,
    config: PerceptionISPConfig | None = None,
    include_labels: Sequence[str] | None = None,
    boundary_thickness: int = 2,
    context_radius: int = 10,
    progress_interval: int = 0,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("object-boundary edge suite needs at least one sample")
    allowed = None if include_labels is None else {str(label) for label in include_labels}
    cases: list[Dict[str, Any]] = []
    box_rows: list[Dict[str, Any]] = []
    started = time.perf_counter()
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        case = _run_case(
            sample,
            config=config,
            include_labels=allowed,
            boundary_thickness=boundary_thickness,
            context_radius=context_radius,
        )
        cases.append(case)
        box_rows.extend(case.get("box_rows", ()))
        if progress_interval and (index == total or index % max(int(progress_interval), 1) == 0):
            _print_progress(index, total, started)
    checks = _checks(cases, box_rows)
    return {
        "name": "Object-boundary edge evidence",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "pass": all(row["status"] == "pass" for row in checks),
        "claim_status": "object_boundary_edge_diagnostic",
        "sample_count": len(cases),
        "box_count": len(box_rows),
        "included_labels": sorted(allowed) if allowed is not None else sorted({str(row.get("label", "")) for row in box_rows}),
        "boundary_thickness": int(boundary_thickness),
        "context_radius": int(context_radius),
        "checks": checks,
        "aggregate": _aggregate(cases, box_rows),
        "label_breakdown": _breakdown(box_rows, "label"),
        "area_breakdown": _breakdown(box_rows, "area_bucket"),
        "cases": cases,
        "interpretation": (
            "This suite compares HumanISP RGB edge evidence, PerceptionISP RGB edge evidence, "
            "PerceptionISP aux edge strength, and PerceptionISP aux edge confidence around GT object box boundaries."
        ),
        "claim_boundary": (
            "KITTI/YOLO labels provide boxes, not segmentation contours. This is a box-boundary proxy for object-edge evidence, "
            "not true object contour accuracy and not detector-performance proof."
        ),
    }


def write_object_boundary_edge_suite(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / OBJECT_BOUNDARY_EDGE_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def parse_label_list(value: str | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    labels = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return labels or None


def _run_case(
    sample: EvaluationSample,
    *,
    config: PerceptionISPConfig | None,
    include_labels: set[str] | None,
    boundary_thickness: int,
    context_radius: int,
) -> Dict[str, Any]:
    images = build_pipeline_images(sample, config=config)
    human_rgb = np.asarray(images.human_rgb, dtype=np.float64)
    perception_rgb = np.asarray(images.perception_rgb, dtype=np.float64)
    aux_maps = {key: np.asarray(value, dtype=np.float64) for key, value in images.aux_maps.items()}
    shape = human_rgb.shape[:2]
    boxes = tuple(box for box in sample.ground_truth if include_labels is None or str(box.label) in include_labels)
    boundary = _boxes_boundary_mask(boxes, shape, thickness=boundary_thickness)
    area = _boxes_area_mask(boxes, shape)
    context = _context_mask(area, boundary, radius=context_radius)
    signals = _signals(human_rgb, perception_rgb, aux_maps)
    metrics = _boundary_metrics(signals, boundary, context)
    metrics.update(
        {
            "box_count": int(len(boxes)),
            "boundary_fraction": float(np.mean(boundary)),
            "context_fraction": float(np.mean(context)),
            "finite_outputs": _finite_outputs(human_rgb, perception_rgb, aux_maps),
            "raw_pattern_remapped": bool(sample.raw.provenance.get("pattern_remapped", False)),
        }
    )
    box_rows = [
        _box_row(
            sample,
            box,
            signals,
            shape=shape,
            boundary_thickness=boundary_thickness,
            context_radius=context_radius,
        )
        for box in boxes
    ]
    return {
        "id": str(sample.sample_id),
        "source": str(sample.source),
        "box_count": int(len(boxes)),
        "cfa_pattern": str(sample.raw.metadata.cfa_pattern),
        "raw_provenance": dict(sample.raw.provenance),
        "metrics": metrics,
        "box_rows": box_rows,
    }


def _box_row(
    sample: EvaluationSample,
    box: BoundingBox,
    signals: Mapping[str, np.ndarray],
    *,
    shape: Tuple[int, int],
    boundary_thickness: int,
    context_radius: int,
) -> Dict[str, Any]:
    boundary = _boxes_boundary_mask((box,), shape, thickness=boundary_thickness)
    area = _boxes_area_mask((box,), shape)
    context = _context_mask(area, boundary, radius=context_radius)
    metrics = _boundary_metrics(signals, boundary, context)
    return {
        "sample_id": str(sample.sample_id),
        "label": str(box.label),
        "area": float(box.area),
        "area_bucket": _area_bucket(box.area),
        "xyxy": [float(value) for value in box.xyxy],
        **metrics,
    }


def _signals(human_rgb: np.ndarray, perception_rgb: np.ndarray, aux_maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    edge_confidence = np.asarray(aux_maps.get("edge_confidence", np.zeros(human_rgb.shape[:2])), dtype=np.float64)
    return {
        "human_rgb_edge": _edge_strength(_luma(human_rgb)),
        "perception_rgb_edge": _edge_strength(_luma(perception_rgb)),
        "aux_edge_strength": _normalize(np.asarray(aux_maps.get("edge_strength", np.zeros_like(edge_confidence)), dtype=np.float64)),
        "aux_edge_confidence": np.clip(edge_confidence, 0.0, 1.0),
    }


def _boundary_metrics(signals: Mapping[str, np.ndarray], boundary: np.ndarray, context: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name in SIGNALS:
        signal = np.asarray(signals[name], dtype=np.float64)
        predicted = _edge_mask(signal)
        values = _f1_metrics(predicted, boundary, tolerance=2)
        on = _masked_mean(signal, boundary)
        off = _masked_mean(signal, context)
        metrics[f"{name}_boundary_precision"] = values["precision"]
        metrics[f"{name}_boundary_recall"] = values["recall"]
        metrics[f"{name}_boundary_f1"] = values["f1"]
        metrics[f"{name}_on_boundary"] = on
        metrics[f"{name}_off_boundary_context"] = off
        metrics[f"{name}_boundary_separation"] = float(on - off)
    human_f1 = metrics["human_rgb_edge_boundary_f1"]
    metrics["perception_rgb_minus_human_boundary_f1"] = float(metrics["perception_rgb_edge_boundary_f1"] - human_f1)
    metrics["aux_strength_minus_human_boundary_f1"] = float(metrics["aux_edge_strength_boundary_f1"] - human_f1)
    metrics["aux_confidence_minus_human_boundary_f1"] = float(metrics["aux_edge_confidence_boundary_f1"] - human_f1)
    return metrics


def _checks(cases: Sequence[Mapping[str, Any]], box_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    case_metrics = [case.get("metrics", {}) for case in cases if isinstance(case.get("metrics"), Mapping)]
    finite = all(bool(row.get("finite_outputs")) for row in case_metrics)
    box_count = len(box_rows)
    boundaries = [float(row.get("boundary_fraction", 0.0)) for row in case_metrics]
    bounded = all(
        0.0 <= float(row.get(f"{signal}_boundary_f1", 0.0)) <= 1.0
        for row in list(case_metrics) + list(box_rows)
        for signal in SIGNALS
    )
    finite_metrics = all(
        np.isfinite(float(row.get(metric, 0.0)))
        for row in list(case_metrics) + list(box_rows)
        for metric in (
            "perception_rgb_minus_human_boundary_f1",
            "aux_strength_minus_human_boundary_f1",
            "aux_confidence_minus_human_boundary_f1",
        )
    )
    remaps = [bool(row.get("raw_pattern_remapped")) for row in case_metrics]
    return [
        {
            "id": "finite_object_boundary_outputs",
            "description": "HumanISP RGB, PerceptionISP RGB, and aux maps are finite.",
            "status": "pass" if finite else "fail",
            "criteria": [{"metric": "finite_outputs", "value": bool(finite), "pass": bool(finite)}],
        },
        {
            "id": "object_box_boundaries_present",
            "description": "At least one evaluated object box boundary exists and the union boundary mask is non-empty.",
            "status": "pass" if box_count > 0 and max(boundaries or [0.0]) > 0.0 else "fail",
            "criteria": [
                {"metric": "box_count", "value": box_count, "threshold": 1, "pass": box_count > 0},
                {"metric": "max_boundary_fraction", "value": max(boundaries or [0.0]), "threshold": 0.0, "pass": max(boundaries or [0.0]) > 0.0},
            ],
        },
        {
            "id": "object_boundary_metrics_bounded",
            "description": "Boundary precision/recall/F1 metrics are bounded in [0, 1].",
            "status": "pass" if bounded and finite_metrics else "fail",
            "criteria": [
                {"metric": "bounded_f1_metrics", "value": bool(bounded), "pass": bool(bounded)},
                {"metric": "finite_delta_metrics", "value": bool(finite_metrics), "pass": bool(finite_metrics)},
            ],
        },
        {
            "id": "camerae2e_cfa_pattern_preserved",
            "description": "CameraE2E true CFA mosaic evidence should avoid source/target CFA remapping.",
            "status": "pass" if not any(remaps) else "fail",
            "criteria": [{"metric": "pattern_remapped_count", "value": sum(1 for value in remaps if value), "threshold": 0, "pass": not any(remaps)}],
        },
    ]


def _aggregate(cases: Sequence[Mapping[str, Any]], box_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [row.get("metrics", {}) for row in cases if isinstance(row.get("metrics"), Mapping)]
    box_metric_rows = [row for row in box_rows if isinstance(row, Mapping)]
    aggregate: Dict[str, Any] = {
        "case_count": len(rows),
        "box_count": len(box_metric_rows),
        "mean_boxes_per_sample": _mean([float(row.get("box_count", 0.0)) for row in rows]),
        "boundary_fraction_mean": _mean([float(row.get("boundary_fraction", 0.0)) for row in rows]),
    }
    for prefix in SIGNALS:
        for suffix in ("boundary_f1", "boundary_precision", "boundary_recall", "boundary_separation"):
            key = f"{prefix}_{suffix}"
            aggregate[f"{key}_mean"] = _mean([float(row.get(key, 0.0)) for row in box_metric_rows])
    for key in (
        "perception_rgb_minus_human_boundary_f1",
        "aux_strength_minus_human_boundary_f1",
        "aux_confidence_minus_human_boundary_f1",
    ):
        values = [float(row.get(key, 0.0)) for row in box_metric_rows]
        aggregate[f"{key}_mean"] = _mean(values)
        aggregate[f"{key}_win_rate"] = _win_rate(values)
    return aggregate


def _breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    out = []
    for value, group in sorted(grouped.items()):
        payload: Dict[str, Any] = {key: value, "box_count": len(group)}
        for signal in SIGNALS:
            payload[f"{signal}_boundary_f1_mean"] = _mean([float(row.get(f"{signal}_boundary_f1", 0.0)) for row in group])
            payload[f"{signal}_boundary_separation_mean"] = _mean([float(row.get(f"{signal}_boundary_separation", 0.0)) for row in group])
        payload["perception_rgb_minus_human_boundary_f1_mean"] = _mean([float(row.get("perception_rgb_minus_human_boundary_f1", 0.0)) for row in group])
        payload["aux_strength_minus_human_boundary_f1_mean"] = _mean([float(row.get("aux_strength_minus_human_boundary_f1", 0.0)) for row in group])
        payload["aux_confidence_minus_human_boundary_f1_mean"] = _mean([float(row.get("aux_confidence_minus_human_boundary_f1", 0.0)) for row in group])
        out.append(payload)
    return out


def _boxes_boundary_mask(boxes: Sequence[BoundingBox], shape: Tuple[int, int], *, thickness: int) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    t = max(int(thickness), 1)
    for box in boxes:
        x1, y1, x2, y2 = _clip_box(box.xyxy, width=width, height=height)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1 : min(y1 + t, y2 + 1), x1 : x2 + 1] = True
        mask[max(y2 - t + 1, y1) : y2 + 1, x1 : x2 + 1] = True
        mask[y1 : y2 + 1, x1 : min(x1 + t, x2 + 1)] = True
        mask[y1 : y2 + 1, max(x2 - t + 1, x1) : x2 + 1] = True
    return mask


def _boxes_area_mask(boxes: Sequence[BoundingBox], shape: Tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = _clip_box(box.xyxy, width=width, height=height)
        if x2 > x1 and y2 > y1:
            mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def _clip_box(xyxy: Sequence[float], *, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    return (
        max(0, min(width - 1, int(np.floor(x1)))),
        max(0, min(height - 1, int(np.floor(y1)))),
        max(0, min(width - 1, int(np.ceil(x2)))),
        max(0, min(height - 1, int(np.ceil(y2)))),
    )


def _context_mask(area: np.ndarray, boundary: np.ndarray, *, radius: int) -> np.ndarray:
    near = _dilate(area, radius=max(int(radius), 1))
    blocked = _dilate(boundary, radius=2)
    context = near & np.logical_not(blocked)
    if not bool(np.any(context)):
        context = np.logical_not(blocked)
    return context


def _edge_strength(values: np.ndarray) -> np.ndarray:
    return _normalize(_edge_strength_abs(values))


def _edge_strength_abs(values: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(values, dtype=np.float64))
    return np.sqrt(gx * gx + gy * gy)


def _edge_mask(strength: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    values = np.asarray(strength, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    threshold = max(float(np.percentile(values, float(percentile))), float(np.max(values)) * 0.10, 1.0e-6)
    return values >= threshold


def _f1_metrics(predicted: np.ndarray, reference: np.ndarray, *, tolerance: int = 1) -> Dict[str, float]:
    pred = np.asarray(predicted, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if not bool(np.any(pred)) and not bool(np.any(ref)):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not bool(np.any(pred)) or not bool(np.any(ref)):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ref_dilated = _dilate(ref, radius=tolerance)
    pred_dilated = _dilate(pred, radius=tolerance)
    precision = float(np.sum(pred & ref_dilated) / max(float(np.sum(pred)), 1.0))
    recall = float(np.sum(ref & pred_dilated) / max(float(np.sum(ref)), 1.0))
    f1 = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


def _dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return values.copy()
    padded = np.pad(values, int(radius), mode="constant", constant_values=False)
    out = np.zeros_like(values, dtype=bool)
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            out |= padded[int(radius) + dy : int(radius) + dy + values.shape[0], int(radius) + dx : int(radius) + dx + values.shape[1]]
    return out


def _luma(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.zeros_like(arr, dtype=np.float64)
    low = float(np.nanmin(arr[finite]))
    high = float(np.nanmax(arr[finite]))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    keep = np.asarray(mask, dtype=bool)
    if not bool(np.any(keep)):
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)[keep]))


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else 0.0


def _win_rate(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean([value > 0.0 for value in clean])) if clean else 0.0


def _finite_outputs(human_rgb: np.ndarray, perception_rgb: np.ndarray, aux_maps: Mapping[str, np.ndarray]) -> bool:
    return bool(
        np.isfinite(human_rgb).all()
        and np.isfinite(perception_rgb).all()
        and all(np.isfinite(value).all() for value in aux_maps.values())
    )


def _area_bucket(area: float) -> str:
    value = float(area)
    if value < 32.0 * 32.0:
        return "small"
    if value < 96.0 * 96.0:
        return "medium"
    return "large"


def _print_progress(index: int, total: int, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    rate = float(index / elapsed)
    eta = float((int(total) - int(index)) / max(rate, 1.0e-12))
    print(f"[object-boundary-edge] {index}/{total} elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s", file=sys.stderr, flush=True)


def _render_html(summary: Mapping[str, Any]) -> str:
    checks = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    labels = "".join(_breakdown_row(row, "label") for row in summary.get("label_breakdown", ()) if isinstance(row, Mapping))
    areas = "".join(_breakdown_row(row, "area_bucket") for row in summary.get("area_breakdown", ()) if isinstance(row, Mapping))
    cases = "".join(_case_row(row) for row in summary.get("cases", ()) if isinstance(row, Mapping))
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    status_class = "supported" if bool(summary.get("pass")) else "not_supported"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Object Boundary Edge Evidence</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .supported {{ color: #047857; font-weight: 700; }}
    .not_supported {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Object Boundary Edge Evidence</h1>
  <p>Status: <code class=\"{status_class}\">{html_lib.escape(str(summary.get('status', '')))}</code>;
  claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.
  {html_lib.escape(str(summary.get('interpretation', '')))}
  {html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <table>
    <thead><tr><th>Samples</th><th>Boxes</th><th>Human F1</th><th>Perception F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th><th>Perception Delta</th><th>Aux Strength Delta</th><th>Aux Confidence Delta</th><th>Aux Confidence Win</th></tr></thead>
    <tbody><tr>
      <td>{int(summary.get('sample_count', 0))}</td>
      <td>{int(summary.get('box_count', 0))}</td>
      <td>{_fmt(aggregate.get('human_rgb_edge_boundary_f1_mean'))}</td>
      <td>{_fmt(aggregate.get('perception_rgb_edge_boundary_f1_mean'))}</td>
      <td>{_fmt(aggregate.get('aux_edge_strength_boundary_f1_mean'))}</td>
      <td>{_fmt(aggregate.get('aux_edge_confidence_boundary_f1_mean'))}</td>
      <td>{_fmt(aggregate.get('perception_rgb_minus_human_boundary_f1_mean'), signed=True)}</td>
      <td>{_fmt(aggregate.get('aux_strength_minus_human_boundary_f1_mean'), signed=True)}</td>
      <td>{_fmt(aggregate.get('aux_confidence_minus_human_boundary_f1_mean'), signed=True)}</td>
      <td>{_fmt(aggregate.get('aux_confidence_minus_human_boundary_f1_win_rate'))}</td>
    </tr></tbody>
  </table>
  <h2>Checks</h2>
  <table><thead><tr><th>ID</th><th>Status</th><th>Description</th></tr></thead><tbody>{checks}</tbody></table>
  <h2>Label Breakdown</h2>
  <table><thead><tr><th>Label</th><th>Boxes</th><th>Human F1</th><th>Perception F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th><th>Perception Delta</th><th>Aux Strength Delta</th><th>Aux Confidence Delta</th></tr></thead><tbody>{labels}</tbody></table>
  <h2>Area Breakdown</h2>
  <table><thead><tr><th>Area</th><th>Boxes</th><th>Human F1</th><th>Perception F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th><th>Perception Delta</th><th>Aux Strength Delta</th><th>Aux Confidence Delta</th></tr></thead><tbody>{areas}</tbody></table>
  <h2>Samples</h2>
  <table><thead><tr><th>Sample</th><th>CFA</th><th>Boxes</th><th>Human F1</th><th>Perception F1</th><th>Aux Strength F1</th><th>Aux Confidence F1</th><th>Pattern Remapped</th></tr></thead><tbody>{cases}</tbody></table>
  <p>Raw JSON: <code>{OBJECT_BOUNDARY_EDGE_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{'supported' if status == 'pass' else 'not_supported'}\">{html_lib.escape(status)}</td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        "</tr>"
    )


def _breakdown_row(row: Mapping[str, Any], key: str) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get(key, '')))}</code></td>"
        f"<td>{int(row.get('box_count', 0))}</td>"
        f"<td>{_fmt(row.get('human_rgb_edge_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('perception_rgb_edge_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('aux_edge_strength_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('aux_edge_confidence_boundary_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('perception_rgb_minus_human_boundary_f1_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('aux_strength_minus_human_boundary_f1_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('aux_confidence_minus_human_boundary_f1_mean'), signed=True)}</td>"
        "</tr>"
    )


def _case_row(row: Mapping[str, Any]) -> str:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{int(row.get('box_count', 0))}</td>"
        f"<td>{_fmt(metrics.get('human_rgb_edge_boundary_f1'))}</td>"
        f"<td>{_fmt(metrics.get('perception_rgb_edge_boundary_f1'))}</td>"
        f"<td>{_fmt(metrics.get('aux_edge_strength_boundary_f1'))}</td>"
        f"<td>{_fmt(metrics.get('aux_edge_confidence_boundary_f1'))}</td>"
        f"<td>{html_lib.escape(str(metrics.get('raw_pattern_remapped', '')))}</td>"
        "</tr>"
    )


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
