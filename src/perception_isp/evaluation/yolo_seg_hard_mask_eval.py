"""Hard-slice instance-mask evaluation for YOLO segmentation runs.

This evaluator is intended for YOLO-seg exports where labels are polygons:

    class x1 y1 x2 y2 ...

It also tolerates regular YOLO boxes:

    class xc yc w h

The main use is comparing RGB-only and RGB+Aux segmentation checkpoints on
small/thin object masks and boundaries.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from perception_isp.core.types import json_ready
from perception_isp.training.yolo_aux_train import RgbAuxGatedStem as RgbAuxGatedStem  # noqa: F401 - checkpoint pickle lookup


DEFAULT_IOU_THRESHOLDS = tuple(float(x) for x in np.arange(0.50, 0.96, 0.05))
DEFAULT_CONF_THRESHOLDS = (0.05, 0.10, 0.25)


@dataclass(frozen=True)
class MaskRecord:
    cls: int
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    conf: float = 1.0
    stem: str = ""

    @property
    def box_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return float(max(w / h if h > 0 else np.inf, h / w if w > 0 else np.inf))


@dataclass(frozen=True)
class MatchRecord:
    mask_iou: float
    boundary_f1: float


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-seg runs on small/thin mask and boundary slices.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec formatted as name|model.pt|data.yaml. Repeat for multiple segmenters.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--predict-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--small-area", type=float, default=0.02)
    parser.add_argument("--thin-aspect", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    summary = evaluate_runs(
        run_specs=[str(value) for value in args.run],
        split=str(args.split),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        predict_conf=float(args.predict_conf),
        predict_iou=float(args.predict_iou),
        max_det=int(args.max_det),
        match_iou=float(args.match_iou),
        boundary_radius=int(args.boundary_radius),
        small_area=float(args.small_area),
        thin_aspect=float(args.thin_aspect),
        out=Path(args.out),
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def evaluate_runs(
    *,
    run_specs: Sequence[str],
    split: str = "val",
    imgsz: int = 512,
    batch: int = 4,
    device: str = "mps",
    workers: int = 0,
    predict_conf: float = 0.001,
    predict_iou: float = 0.70,
    max_det: int = 300,
    match_iou: float = 0.50,
    boundary_radius: int = 2,
    small_area: float = 0.02,
    thin_aspect: float = 3.0,
    out: Path,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is required for YOLO segmentation hard-mask evaluation") from exc

    out.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Any] = {}
    for spec in run_specs:
        name, model_path, data_path = _parse_run_spec(spec)
        root, config = _load_data_config(Path(data_path))
        image_paths = _image_paths(root=root, split=str(config.get(split, f"images/{split}")))
        image_shapes = {path.stem: _image_hw(path) for path in image_paths}
        gt_by_stem = _load_gt(root=root, split=str(config.get(split, f"images/{split}")), image_shapes=image_shapes)
        predictions = _predict(
            YOLO(str(model_path)),
            image_paths=image_paths,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            predict_conf=predict_conf,
            predict_iou=predict_iou,
            max_det=max_det,
        )
        run_summary = {
            "name": name,
            "model": str(model_path),
            "data": str(data_path),
            "image_count": len(image_paths),
            "gt_count": int(sum(len(rows) for rows in gt_by_stem.values())),
            "pred_count_raw": int(sum(len(rows) for rows in predictions.values())),
            "filters": {},
        }
        for filter_name in ("all", "small", "thin", "small_or_thin"):
            run_summary["filters"][filter_name] = _evaluate_filter(
                gt_by_stem=gt_by_stem,
                predictions=predictions,
                filter_name=filter_name,
                small_area=small_area,
                thin_aspect=thin_aspect,
                iou_thresholds=DEFAULT_IOU_THRESHOLDS,
                conf_thresholds=DEFAULT_CONF_THRESHOLDS,
                match_iou=match_iou,
                boundary_radius=boundary_radius,
            )
        runs[name] = run_summary

    summary = {
        "run_config": {
            "split": str(split),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "device": str(device),
            "workers": int(workers),
            "predict_conf": float(predict_conf),
            "predict_iou": float(predict_iou),
            "max_det": int(max_det),
            "match_iou": float(match_iou),
            "boundary_radius": int(boundary_radius),
            "small_area": float(small_area),
            "thin_aspect": float(thin_aspect),
        },
        "runs": runs,
        "comparison": _compare_runs(runs),
    }
    (out / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (out / "index.html").write_text(_render_html(summary), encoding="utf-8")
    return summary


def _parse_run_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"run spec must be name|model.pt|data.yaml, got: {spec}")
    return parts[0], parts[1], parts[2]


def _image_paths(*, root: Path, split: str) -> List[Path]:
    image_dir = root / split
    paths = sorted(image_dir.glob("*.npy"))
    if not paths:
        paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(sorted(image_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"no images found under {image_dir}")
    return paths


def _image_hw(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(f"expected HWC/CHW .npy image, got {array.shape} in {path}")
        if array.shape[0] in {1, 3, 4, 5, 6, 15} and array.shape[-1] not in {1, 3, 4, 5, 6, 15}:
            array = np.moveaxis(array, 0, -1)
        return int(array.shape[0]), int(array.shape[1])
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def _load_gt(*, root: Path, split: str, image_shapes: Mapping[str, tuple[int, int]]) -> Dict[str, List[MaskRecord]]:
    label_dir = root / split.replace("images/", "labels/", 1)
    gt_by_stem: Dict[str, List[MaskRecord]] = {}
    for stem, (height, width) in sorted(image_shapes.items()):
        label_path = label_dir / f"{stem}.txt"
        rows: List[MaskRecord] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                record = _parse_gt_line(line, width=width, height=height, stem=stem)
                if record is not None:
                    rows.append(record)
        gt_by_stem[stem] = rows
    return gt_by_stem


def _parse_gt_line(line: str, *, width: int, height: int, stem: str) -> MaskRecord | None:
    if not line.strip():
        return None
    values = [float(value) for value in line.split()]
    if len(values) < 5:
        return None
    cls = int(values[0])
    if len(values) > 5 and (len(values) - 1) % 2 == 0:
        coords = np.asarray(values[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0], 0.0, 1.0) * max(width - 1, 1)
        coords[:, 1] = np.clip(coords[:, 1], 0.0, 1.0) * max(height - 1, 1)
        mask = _rasterize_polygon(coords, width=width, height=height)
    else:
        _, xc, yc, w, h = values[:5]
        x1 = max(0.0, xc - w / 2.0) * width
        y1 = max(0.0, yc - h / 2.0) * height
        x2 = min(1.0, xc + w / 2.0) * width
        y2 = min(1.0, yc + h / 2.0) * height
        mask = _rasterize_box((x1, y1, x2, y2), width=width, height=height)
    if not mask.any():
        return None
    return MaskRecord(cls=cls, mask=mask, bbox=_mask_bbox(mask), stem=stem)


def _rasterize_polygon(coords: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.new("L", (int(width), int(height)), 0)
    points = [(float(x), float(y)) for x, y in coords]
    ImageDraw.Draw(image).polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _rasterize_box(box: Sequence[float], *, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    image = Image.new("L", (int(width), int(height)), 0)
    ImageDraw.Draw(image).rectangle((x1, y1, x2, y2), outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _predict(
    model: Any,
    *,
    image_paths: Sequence[Path],
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    predict_conf: float,
    predict_iou: float,
    max_det: int,
) -> Dict[str, List[MaskRecord]]:
    predictions: Dict[str, List[MaskRecord]] = {}
    for start in range(0, len(image_paths), max(1, int(batch))):
        batch_paths = list(image_paths[start : start + max(1, int(batch))])
        arrays = [_load_image_array(path) for path in batch_paths]
        results = model.predict(
            source=arrays,
            imgsz=int(imgsz),
            batch=len(arrays),
            device=str(device),
            workers=int(workers),
            conf=float(predict_conf),
            iou=float(predict_iou),
            max_det=int(max_det),
            retina_masks=True,
            verbose=False,
            stream=False,
        )
        for path, array, result in zip(batch_paths, arrays, results):
            height, width = int(array.shape[0]), int(array.shape[1])
            rows: List[MaskRecord] = []
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is not None and masks is not None and len(boxes) > 0:
                cls_values = boxes.cls.detach().cpu().numpy().astype(int)
                conf_values = boxes.conf.detach().cpu().numpy()
                mask_values = masks.data.detach().cpu().numpy()
                for index, cls_value in enumerate(cls_values):
                    mask = np.asarray(mask_values[index] > 0.5, dtype=bool)
                    if mask.shape != (height, width):
                        mask = _resize_mask(mask, width=width, height=height)
                    if not mask.any():
                        continue
                    rows.append(
                        MaskRecord(
                            cls=int(cls_value),
                            mask=mask,
                            bbox=_mask_bbox(mask),
                            conf=float(conf_values[index]),
                            stem=path.stem,
                        )
                    )
            predictions[path.stem] = rows
    return predictions


def _load_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(f"expected HWC/CHW .npy image, got {array.shape} in {path}")
        if array.shape[0] in {1, 3, 4, 5, 6, 15} and array.shape[-1] not in {1, 3, 4, 5, 6, 15}:
            array = np.moveaxis(array, 0, -1)
        return np.ascontiguousarray(array)
    return np.asarray(Image.open(path).convert("RGB"))


def _evaluate_filter(
    *,
    gt_by_stem: Mapping[str, Sequence[MaskRecord]],
    predictions: Mapping[str, Sequence[MaskRecord]],
    filter_name: str,
    small_area: float,
    thin_aspect: float,
    iou_thresholds: Sequence[float],
    conf_thresholds: Sequence[float],
    match_iou: float,
    boundary_radius: int,
) -> Dict[str, Any]:
    target_by_stem: Dict[str, List[MaskRecord]] = {}
    ignore_by_stem: Dict[str, List[MaskRecord]] = {}
    for stem, gt_rows in gt_by_stem.items():
        targets = [row for row in gt_rows if _record_in_filter(row, filter_name, small_area=small_area, thin_aspect=thin_aspect)]
        target_ids = {id(row) for row in targets}
        ignores = [row for row in gt_rows if id(row) not in target_ids]
        target_by_stem[stem] = targets
        ignore_by_stem[stem] = ignores

    aps = []
    ap_by_iou: Dict[str, float] = {}
    for iou_threshold in iou_thresholds:
        ap = _average_precision(
            target_by_stem=target_by_stem,
            ignore_by_stem=ignore_by_stem,
            predictions=predictions,
            iou_threshold=float(iou_threshold),
        )
        ap_by_iou[f"{iou_threshold:.2f}"] = ap
        aps.append(ap)
    fixed = {
        f"conf_{threshold:.2f}_iou_{match_iou:.2f}": _fixed_conf_metrics(
            target_by_stem=target_by_stem,
            ignore_by_stem=ignore_by_stem,
            predictions=predictions,
            conf_threshold=float(threshold),
            iou_threshold=float(match_iou),
            boundary_radius=int(boundary_radius),
        )
        for threshold in conf_thresholds
    }
    target_count = sum(len(rows) for rows in target_by_stem.values())
    return {
        "target_gt_count": int(target_count),
        "image_count_with_target": int(sum(1 for rows in target_by_stem.values() if rows)),
        "ignored_gt_count": int(sum(len(rows) for rows in ignore_by_stem.values())),
        "mask_mAP50": float(ap_by_iou["0.50"]) if target_count else None,
        "mask_mAP50_95": float(np.nanmean(aps)) if target_count else None,
        "mask_ap_by_iou": ap_by_iou,
        "fixed_conf": fixed,
    }


def _record_in_filter(row: MaskRecord, filter_name: str, *, small_area: float, thin_aspect: float) -> bool:
    small = bool(row.box_area < float(small_area))
    thin = bool(row.aspect_ratio > float(thin_aspect))
    if filter_name == "all":
        return True
    if filter_name == "small":
        return small
    if filter_name == "thin":
        return thin
    if filter_name == "small_or_thin":
        return small or thin
    raise ValueError(f"unsupported filter: {filter_name}")


def _average_precision(
    *,
    target_by_stem: Mapping[str, Sequence[MaskRecord]],
    ignore_by_stem: Mapping[str, Sequence[MaskRecord]],
    predictions: Mapping[str, Sequence[MaskRecord]],
    iou_threshold: float,
) -> float:
    target_count = sum(len(rows) for rows in target_by_stem.values())
    if target_count <= 0:
        return float("nan")
    pred_rows = sorted(_all_predictions(predictions), key=lambda row: row.conf, reverse=True)
    matched: Dict[str, set[int]] = {stem: set() for stem in target_by_stem}
    tp: List[float] = []
    fp: List[float] = []
    for pred in pred_rows:
        stem = pred.stem
        best_iou, best_index = _best_match(pred, target_by_stem.get(stem, ()), matched.get(stem, set()))
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.setdefault(stem, set()).add(best_index)
            tp.append(1.0)
            fp.append(0.0)
            continue
        ignore_iou, _ = _best_match(pred, ignore_by_stem.get(stem, ()), set())
        if ignore_iou >= iou_threshold:
            continue
        tp.append(0.0)
        fp.append(1.0)
    if not tp:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp, dtype=float))
    fp_cum = np.cumsum(np.asarray(fp, dtype=float))
    recalls = tp_cum / max(float(target_count), 1.0)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return float(_integrate_ap(recalls, precisions))


def _fixed_conf_metrics(
    *,
    target_by_stem: Mapping[str, Sequence[MaskRecord]],
    ignore_by_stem: Mapping[str, Sequence[MaskRecord]],
    predictions: Mapping[str, Sequence[MaskRecord]],
    conf_threshold: float,
    iou_threshold: float,
    boundary_radius: int,
) -> Dict[str, Any]:
    matched: Dict[str, set[int]] = {stem: set() for stem in target_by_stem}
    tp = 0
    fp = 0
    ignored_predictions = 0
    matches: List[MatchRecord] = []
    for pred in sorted(_all_predictions(predictions), key=lambda row: row.conf, reverse=True):
        if pred.conf < conf_threshold:
            continue
        stem = pred.stem
        target_rows = target_by_stem.get(stem, ())
        best_iou, best_index = _best_match(pred, target_rows, matched.get(stem, set()))
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.setdefault(stem, set()).add(best_index)
            tp += 1
            matches.append(
                MatchRecord(
                    mask_iou=best_iou,
                    boundary_f1=_boundary_f1(pred.mask, target_rows[best_index].mask, radius=boundary_radius),
                )
            )
            continue
        ignore_iou, _ = _best_match(pred, ignore_by_stem.get(stem, ()), set())
        if ignore_iou >= iou_threshold:
            ignored_predictions += 1
            continue
        fp += 1
    target_count = sum(len(rows) for rows in target_by_stem.values())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / target_count if target_count else None
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(max(target_count - tp, 0)),
        "ignored_predictions": int(ignored_predictions),
        "precision": precision,
        "recall": recall,
        "mean_mask_iou_tp": _mean([row.mask_iou for row in matches]),
        "mean_boundary_f1_tp": _mean([row.boundary_f1 for row in matches]),
    }


def _all_predictions(predictions: Mapping[str, Sequence[MaskRecord]]) -> Iterable[MaskRecord]:
    for rows in predictions.values():
        yield from rows


def _best_match(pred: MaskRecord, targets: Sequence[MaskRecord], matched_indices: set[int]) -> tuple[float, int]:
    best_iou = 0.0
    best_index = -1
    for index, target in enumerate(targets):
        if index in matched_indices or int(pred.cls) != int(target.cls):
            continue
        if _bbox_intersection_area(pred.bbox, target.bbox) <= 0.0:
            continue
        iou = _mask_iou(pred.mask, target.mask)
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_iou, best_index


def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(float(ax1), float(bx1))
    y1 = max(float(ay1), float(by1))
    x2 = min(float(ax2), float(bx2))
    y2 = min(float(ay2), float(by2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        a = _resize_mask(a, width=b.shape[1], height=b.shape[0])
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, *, radius: int) -> float:
    pred_b = _mask_boundary(pred)
    gt_b = _mask_boundary(gt)
    pred_count = int(pred_b.sum())
    gt_count = int(gt_b.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    gt_d = _dilate(gt_b, radius=radius)
    pred_d = _dilate(pred_b, radius=radius)
    precision = float(np.logical_and(pred_b, gt_d).sum() / max(pred_count, 1))
    recall = float(np.logical_and(gt_b, pred_d).sum() / max(gt_count, 1))
    return float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        src = mask.astype(np.uint8)
        eroded = cv2.erode(src, np.ones((3, 3), np.uint8), iterations=1)
        return (src > 0) & (eroded == 0)
    except Exception:
        padded = np.pad(mask, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        neighbors = (
            padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, :-2]
            & padded[:-2, 2:]
            & padded[2:, :-2]
            & padded[2:, 2:]
        )
        return center & ~neighbors


def _dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    try:
        import cv2

        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        image = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(size=2 * radius + 1))
        return np.asarray(image, dtype=np.uint8) > 0


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        resized = cv2.resize(mask.astype(np.uint8), (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)
    except Exception:
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        return np.asarray(image.resize((int(width), int(height)), Image.Resampling.NEAREST), dtype=np.uint8) > 0


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if not len(xs) or not len(ys):
        return (0.0, 0.0, 0.0, 0.0)
    height, width = mask.shape
    return (
        float(xs.min() / max(width, 1)),
        float(ys.min() / max(height, 1)),
        float((xs.max() + 1) / max(width, 1)),
        float((ys.max() + 1) / max(height, 1)),
    )


def _integrate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _load_data_config(path: Path) -> tuple[Path, Dict[str, Any]]:
    config_path = path if path.is_file() else path / "data.yaml"
    config = _read_yaml_like(config_path)
    root = Path(config.get("path", config_path.parent)).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve(), config


def _read_yaml_like(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text()) or {}
        return dict(payload)
    except Exception:
        payload: Dict[str, Any] = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip().strip("'\"")
        return payload


def _compare_runs(runs: Mapping[str, Any]) -> Dict[str, Any]:
    names = list(runs)
    if len(names) < 2:
        return {}
    baseline_name = names[0]
    baseline = runs[baseline_name]
    comparisons: Dict[str, Any] = {}
    for candidate_name in names[1:]:
        candidate = runs[candidate_name]
        by_filter: Dict[str, Any] = {}
        for filter_name, base_metrics in baseline.get("filters", {}).items():
            cand_metrics = candidate.get("filters", {}).get(filter_name, {})
            base_fixed = base_metrics.get("fixed_conf", {}).get("conf_0.25_iou_0.50", {})
            cand_fixed = cand_metrics.get("fixed_conf", {}).get("conf_0.25_iou_0.50", {})
            by_filter[filter_name] = {
                "delta_mask_mAP50": _delta(cand_metrics.get("mask_mAP50"), base_metrics.get("mask_mAP50")),
                "delta_mask_mAP50_95": _delta(cand_metrics.get("mask_mAP50_95"), base_metrics.get("mask_mAP50_95")),
                "delta_conf025_precision": _delta(cand_fixed.get("precision"), base_fixed.get("precision")),
                "delta_conf025_recall": _delta(cand_fixed.get("recall"), base_fixed.get("recall")),
                "delta_conf025_mean_mask_iou_tp": _delta(cand_fixed.get("mean_mask_iou_tp"), base_fixed.get("mean_mask_iou_tp")),
                "delta_conf025_mean_boundary_f1_tp": _delta(
                    cand_fixed.get("mean_boundary_f1_tp"), base_fixed.get("mean_boundary_f1_tp")
                ),
            }
        comparisons[f"{candidate_name}_minus_{baseline_name}"] = by_filter
    return comparisons


def _delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for run_name, run in summary.get("runs", {}).items():
        for filter_name, metrics in run.get("filters", {}).items():
            fixed = metrics.get("fixed_conf", {}).get("conf_0.25_iou_0.50", {})
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(run_name))}</td>"
                f"<td>{html.escape(str(filter_name))}</td>"
                f"<td>{_fmt(metrics.get('target_gt_count'))}</td>"
                f"<td>{_fmt(metrics.get('mask_mAP50'))}</td>"
                f"<td>{_fmt(metrics.get('mask_mAP50_95'))}</td>"
                f"<td>{_fmt(fixed.get('precision'))}</td>"
                f"<td>{_fmt(fixed.get('recall'))}</td>"
                f"<td>{_fmt(fixed.get('mean_mask_iou_tp'))}</td>"
                f"<td>{_fmt(fixed.get('mean_boundary_f1_tp'))}</td>"
                "</tr>"
            )
    comp_rows = []
    for name, filters in summary.get("comparison", {}).items():
        for filter_name, metrics in filters.items():
            comp_rows.append(
                "<tr>"
                f"<td>{html.escape(str(name))}</td>"
                f"<td>{html.escape(str(filter_name))}</td>"
                f"<td>{_fmt(metrics.get('delta_mask_mAP50'))}</td>"
                f"<td>{_fmt(metrics.get('delta_mask_mAP50_95'))}</td>"
                f"<td>{_fmt(metrics.get('delta_conf025_recall'))}</td>"
                f"<td>{_fmt(metrics.get('delta_conf025_mean_boundary_f1_tp'))}</td>"
                "</tr>"
            )
    config = summary.get("run_config", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLO Seg Hard Mask Evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1 {{ margin-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 30px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px 14px; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .tile {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 10px 12px; background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>YOLO Seg Hard Mask Evaluation</h1>
  <div class="note">This report compares actual instance masks and matched-mask boundary F1. Small/thin filters are derived from GT polygon bounding boxes. It is a fast engineering gate, not a final native-RAW claim.</div>
  <div class="grid">
    <div class="tile"><b>split</b><br>{html.escape(str(config.get("split", "")))}</div>
    <div class="tile"><b>match IoU</b><br>{html.escape(str(config.get("match_iou", "")))}</div>
    <div class="tile"><b>boundary radius</b><br>{html.escape(str(config.get("boundary_radius", "")))}</div>
    <div class="tile"><b>small/thin</b><br>area &lt; {html.escape(str(config.get("small_area", "")))}, aspect &gt; {html.escape(str(config.get("thin_aspect", "")))}</div>
  </div>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Run</th><th>Slice</th><th>GT</th><th>Mask AP50</th><th>Mask AP50-95</th><th>P@0.25</th><th>R@0.25</th><th>TP Mask IoU</th><th>TP Boundary F1</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Delta vs First Run</h2>
  <table>
    <thead><tr><th>Comparison</th><th>Slice</th><th>d Mask AP50</th><th>d Mask AP50-95</th><th>d R@0.25</th><th>d TP Boundary F1</th></tr></thead>
    <tbody>{''.join(comp_rows)}</tbody>
  </table>
</body>
</html>
"""


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
