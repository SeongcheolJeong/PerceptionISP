"""Object-level hard-slice evaluation for YOLO RGB vs RGB+Aux detectors."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .types import json_ready
from .yolo_aux_train import RgbAuxGatedStem as RgbAuxGatedStem  # noqa: F401 - legacy checkpoint pickle lookup


DEFAULT_IOU_THRESHOLDS = tuple(float(x) for x in np.arange(0.50, 0.96, 0.05))
DEFAULT_CONF_THRESHOLDS = (0.05, 0.10, 0.25)


@dataclass(frozen=True)
class BoxRecord:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0
    area: float = 0.0
    aspect_ratio: float = 1.0
    stem: str = ""


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate detectors on small/thin object-level GT slices.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec formatted as name|model.pt|data.yaml. Repeat for multiple detectors.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--predict-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
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
    batch: int = 8,
    device: str = "mps",
    workers: int = 0,
    predict_conf: float = 0.001,
    predict_iou: float = 0.70,
    max_det: int = 300,
    small_area: float = 0.02,
    thin_aspect: float = 3.0,
    out: Path,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is required for YOLO hard-object evaluation") from exc

    out.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Any] = {}
    for spec in run_specs:
        name, model_path, data_path = _parse_run_spec(spec)
        root, config = _load_data_config(Path(data_path))
        image_paths = _image_paths(root=root, split=str(config.get(split, f"images/{split}")))
        gt_by_stem = _load_gt(root=root, split=str(config.get(split, f"images/{split}")), small_area=small_area, thin_aspect=thin_aspect)
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
            "predict_conf": float(predict_conf),
            "predict_iou": float(predict_iou),
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
            "small_area": float(small_area),
            "thin_aspect": float(thin_aspect),
        },
        "runs": runs,
    }
    (out / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (out / "index.html").write_text(_render_html(summary))
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


def _load_gt(*, root: Path, split: str, small_area: float, thin_aspect: float) -> Dict[str, List[BoxRecord]]:
    label_dir = root / split.replace("images/", "labels/", 1)
    gt_by_stem: Dict[str, List[BoxRecord]] = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        rows: List[BoxRecord] = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 5:
                continue
            cls = values[0]
            if len(values) > 5 and (len(values) - 1) % 2 == 0:
                coords = np.asarray(values[1:], dtype=float).reshape(-1, 2)
                x1 = float(np.clip(coords[:, 0].min(), 0.0, 1.0))
                y1 = float(np.clip(coords[:, 1].min(), 0.0, 1.0))
                x2 = float(np.clip(coords[:, 0].max(), 0.0, 1.0))
                y2 = float(np.clip(coords[:, 1].max(), 0.0, 1.0))
            else:
                _, xc, yc, w, h = values[:5]
                x1 = max(0.0, xc - w / 2.0)
                y1 = max(0.0, yc - h / 2.0)
                x2 = min(1.0, xc + w / 2.0)
                y2 = min(1.0, yc + h / 2.0)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue
            area = w * h
            aspect = max(w / h if h > 0 else np.inf, h / w if w > 0 else np.inf)
            rows.append(BoxRecord(cls=int(cls), xyxy=(x1, y1, x2, y2), area=area, aspect_ratio=float(aspect), stem=label_path.stem))
        gt_by_stem[label_path.stem] = rows
    return gt_by_stem


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
) -> Dict[str, List[BoxRecord]]:
    predictions: Dict[str, List[BoxRecord]] = {}
    for start in range(0, len(image_paths), int(batch)):
        batch_paths = list(image_paths[start : start + int(batch)])
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
            verbose=False,
            stream=False,
        )
        for path, result in zip(batch_paths, results):
            stem = path.stem
            boxes = getattr(result, "boxes", None)
            rows: List[BoxRecord] = []
            if boxes is not None and len(boxes) > 0:
                xyxyn = boxes.xyxyn.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy()
                conf = boxes.conf.detach().cpu().numpy()
                for coords, cls_value, conf_value in zip(xyxyn, cls, conf):
                    x1, y1, x2, y2 = [float(v) for v in coords[:4]]
                    rows.append(BoxRecord(cls=int(cls_value), xyxy=(x1, y1, x2, y2), conf=float(conf_value), stem=stem))
            predictions[stem] = rows
    return predictions


def _load_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(f"expected HWC/CHW .npy image, got {array.shape} in {path}")
        if array.shape[0] in {1, 3, 4, 5, 6, 15} and array.shape[-1] not in {1, 3, 4, 5, 6, 15}:
            array = np.moveaxis(array, 0, -1)
        return np.ascontiguousarray(array)
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Pillow is required to load non-NPY images") from exc
    return np.asarray(Image.open(path).convert("RGB"))


def _evaluate_filter(
    *,
    gt_by_stem: Mapping[str, Sequence[BoxRecord]],
    predictions: Mapping[str, Sequence[BoxRecord]],
    filter_name: str,
    small_area: float,
    thin_aspect: float,
    iou_thresholds: Sequence[float],
    conf_thresholds: Sequence[float],
) -> Dict[str, Any]:
    target_by_stem: Dict[str, List[BoxRecord]] = {}
    ignore_by_stem: Dict[str, List[BoxRecord]] = {}
    for stem, gt_rows in gt_by_stem.items():
        targets = [row for row in gt_rows if _box_in_filter(row, filter_name, small_area=small_area, thin_aspect=thin_aspect)]
        ignores = [row for row in gt_rows if row not in targets]
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
        f"conf_{threshold:.2f}_iou_0.50": _fixed_conf_metrics(
            target_by_stem=target_by_stem,
            ignore_by_stem=ignore_by_stem,
            predictions=predictions,
            conf_threshold=float(threshold),
            iou_threshold=0.50,
        )
        for threshold in conf_thresholds
    }
    target_count = sum(len(rows) for rows in target_by_stem.values())
    image_count_with_target = sum(1 for rows in target_by_stem.values() if rows)
    return {
        "target_gt_count": int(target_count),
        "image_count_with_target": int(image_count_with_target),
        "ignored_gt_count": int(sum(len(rows) for rows in ignore_by_stem.values())),
        "mAP50": float(ap_by_iou["0.50"]) if target_count else None,
        "mAP50_95": float(np.nanmean(aps)) if target_count else None,
        "ap_by_iou": ap_by_iou,
        "fixed_conf": fixed,
    }


def _box_in_filter(row: BoxRecord, filter_name: str, *, small_area: float, thin_aspect: float) -> bool:
    small = bool(row.area < float(small_area))
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
    target_by_stem: Mapping[str, Sequence[BoxRecord]],
    ignore_by_stem: Mapping[str, Sequence[BoxRecord]],
    predictions: Mapping[str, Sequence[BoxRecord]],
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
        target_rows = target_by_stem.get(stem, ())
        best_iou, best_index = _best_match(pred, target_rows, matched.get(stem, set()))
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
    target_by_stem: Mapping[str, Sequence[BoxRecord]],
    ignore_by_stem: Mapping[str, Sequence[BoxRecord]],
    predictions: Mapping[str, Sequence[BoxRecord]],
    conf_threshold: float,
    iou_threshold: float,
) -> Dict[str, Any]:
    matched: Dict[str, set[int]] = {stem: set() for stem in target_by_stem}
    tp = 0
    fp = 0
    ignored_predictions = 0
    for pred in sorted(_all_predictions(predictions), key=lambda row: row.conf, reverse=True):
        if pred.conf < conf_threshold:
            continue
        stem = pred.stem
        best_iou, best_index = _best_match(pred, target_by_stem.get(stem, ()), matched.get(stem, set()))
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.setdefault(stem, set()).add(best_index)
            tp += 1
            continue
        ignore_iou, _ = _best_match(pred, ignore_by_stem.get(stem, ()), set())
        if ignore_iou >= iou_threshold:
            ignored_predictions += 1
            continue
        fp += 1
    target_count = sum(len(rows) for rows in target_by_stem.values())
    recall = tp / target_count if target_count else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(max(target_count - tp, 0)),
        "ignored_predictions": int(ignored_predictions),
        "precision": precision,
        "recall": recall,
    }


def _all_predictions(predictions: Mapping[str, Sequence[BoxRecord]]) -> Iterable[BoxRecord]:
    for rows in predictions.values():
        yield from rows


def _best_match(pred: BoxRecord, targets: Sequence[BoxRecord], matched_indices: set[int]) -> tuple[float, int]:
    best_iou = 0.0
    best_index = -1
    for index, target in enumerate(targets):
        if index in matched_indices or int(pred.cls) != int(target.cls):
            continue
        iou = _iou(pred.xyxy, target.xyxy)
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_iou, best_index


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _integrate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


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


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for run_name, run in summary.get("runs", {}).items():
        for filter_name, metrics in run.get("filters", {}).items():
            fixed = metrics.get("fixed_conf", {}).get("conf_0.25_iou_0.50", {})
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(run_name))}</td>"
                f"<td>{html.escape(str(filter_name))}</td>"
                f"<td>{metrics.get('target_gt_count')}</td>"
                f"<td>{_fmt(metrics.get('mAP50'))}</td>"
                f"<td>{_fmt(metrics.get('mAP50_95'))}</td>"
                f"<td>{_fmt(fixed.get('precision'))}</td>"
                f"<td>{_fmt(fixed.get('recall'))}</td>"
                f"<td>{fixed.get('tp')}</td>"
                f"<td>{fixed.get('fp')}</td>"
                f"<td>{fixed.get('fn')}</td>"
                "</tr>"
            )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLO Hard Object Evaluation</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }
    table { border-collapse: collapse; width: 100%; font-size: 14px; }
    th, td { border-bottom: 1px solid #d7dde5; padding: 8px 10px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { background: #f5f7fa; }
    code { background: #f5f7fa; padding: 2px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>YOLO Hard Object Evaluation</h1>
  <p>Object-level metrics. For small/thin filters, non-target GT boxes are ignored when matched, so normal-object detections are not counted as false positives.</p>
  <table>
    <thead><tr><th>Run</th><th>Filter</th><th>GT</th><th>AP50</th><th>mAP50-95</th><th>P@0.25</th><th>R@0.25</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
    <tbody>
""" + "\n".join(rows) + """
    </tbody>
  </table>
</body>
</html>
"""


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.4f}"
    except Exception:
        return html.escape(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
