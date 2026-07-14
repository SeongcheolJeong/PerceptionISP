"""LIS segmentation smoke evaluation for RGB/RAW front-end variants.

The packaged LIS RAW archives are RGB PNG images, not native CFA mosaics.
This evaluator therefore measures whether image-derived front-end evidence
helps a pretrained segmentation model; it does not prove sensor/CFA-aware ISP
benefit by itself.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


YOLO_TO_LIS = {
    "bicycle": "bicycle",
    "chair": "chair",
    "dining table": "diningtable",
    "diningtable": "diningtable",
    "bottle": "bottle",
    "motorcycle": "motorbike",
    "motorbike": "motorbike",
    "car": "car",
    "tv": "tvmonitor",
    "tvmonitor": "tvmonitor",
    "bus": "bus",
}

DEFAULT_EVAL_CONFS = (0.05, 0.10, 0.25)
DEFAULT_RAW_TRANSFORMS = ("edge_contrast",)
SCREEN_RAW_TRANSFORMS = (
    "clahe",
    "gamma_bright",
    "unsharp",
    "edge_contrast_mild",
    "edge_contrast",
    "edge_contrast_strong",
)
TRANSFORM_LABELS = {
    "clahe": "RAW-dark PNG + luma CLAHE",
    "gamma_bright": "RAW-dark PNG + gamma brightening",
    "unsharp": "RAW-dark PNG + unsharp luma",
    "edge_contrast_mild": "RAW-dark PNG + mild edge/contrast boost",
    "edge_contrast": "RAW-dark PNG + rule-based edge/contrast boost",
    "edge_contrast_strong": "RAW-dark PNG + strong edge/contrast boost",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    image_dir: Path
    annotation_path: Path
    transform: str = "none"


@dataclass
class MaskRecord:
    stem: str
    category: str
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    conf: float = 1.0
    source: str = "gt"


@dataclass
class MatchRecord:
    stem: str
    category: str
    conf: float
    mask_iou: float
    boundary_f1: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LIS segmentation variants with a COCO pretrained YOLO segmenter.")
    parser.add_argument("--subset-root", default="data/raw_datasets/lis/subsets/dark_test32")
    parser.add_argument("--output-dir", default="reports/perception_lis_segmentation_dark_test32_yolo11n_v1")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--predict-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--eval-conf", action="append", type=float, default=[])
    parser.add_argument("--selection-conf", type=float, default=0.10)
    parser.add_argument(
        "--raw-transform",
        action="append",
        default=[],
        help="RAW-dark transform to evaluate. Repeat for multiple transforms. Default: edge_contrast.",
    )
    parser.add_argument(
        "--screen-transforms",
        action="store_true",
        help="Evaluate the built-in transform screen on the RAW-dark PNG branch.",
    )
    parser.add_argument("--no-edge-boost", action="store_true")
    args = parser.parse_args(argv)

    eval_confs = tuple(args.eval_conf) if args.eval_conf else DEFAULT_EVAL_CONFS
    raw_transforms: Sequence[str]
    if args.no_edge_boost:
        raw_transforms = ()
    elif args.screen_transforms:
        raw_transforms = SCREEN_RAW_TRANSFORMS
    elif args.raw_transform:
        raw_transforms = tuple(str(value) for value in args.raw_transform)
    else:
        raw_transforms = DEFAULT_RAW_TRANSFORMS
    summary = evaluate_lis_segmentation(
        subset_root=Path(args.subset_root),
        output_dir=Path(args.output_dir),
        model_name=str(args.model),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        predict_conf=float(args.predict_conf),
        predict_iou=float(args.predict_iou),
        max_det=int(args.max_det),
        match_iou=float(args.match_iou),
        boundary_radius=int(args.boundary_radius),
        preview_count=int(args.preview_count),
        eval_confs=eval_confs,
        raw_transforms=raw_transforms,
        selection_conf=float(args.selection_conf),
    )
    print(json.dumps(_json_ready(summary), indent=2))
    return 0


def evaluate_lis_segmentation(
    *,
    subset_root: Path,
    output_dir: Path,
    model_name: str,
    imgsz: int,
    batch: int,
    device: str,
    predict_conf: float,
    predict_iou: float,
    max_det: int,
    match_iou: float,
    boundary_radius: int,
    preview_count: int,
    eval_confs: Sequence[float],
    raw_transforms: Sequence[str] = DEFAULT_RAW_TRANSFORMS,
    selection_conf: float = 0.10,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is required for LIS segmentation evaluation") from exc

    subset_root = subset_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    variants = _variant_specs(subset_root, raw_transforms=raw_transforms, cache_dir=cache_dir)
    model = YOLO(model_name)
    yolo_names = _model_names(model)
    requested_device = _resolve_device(device)

    variant_summaries: Dict[str, Any] = {}
    for variant in variants:
        dataset = _load_lis_coco(variant.annotation_path)
        original_image_paths = _variant_image_paths(variant, dataset["images"])
        source_stems = [path.stem for path in original_image_paths]
        image_paths = list(original_image_paths)
        if variant.transform != "none":
            image_paths = [_transform_image(path, cache_dir / variant.name, variant.transform) for path in image_paths]

        predictions = _predict_with_fallback(
            model=model,
            image_paths=image_paths,
            source_stems=source_stems,
            yolo_names=yolo_names,
            imgsz=imgsz,
            batch=batch,
            device=requested_device,
            predict_conf=predict_conf,
            predict_iou=predict_iou,
            max_det=max_det,
        )
        gt_by_stem = dataset["gt_by_stem"]
        image_metrics = _image_level_metrics(
            gt_by_stem=gt_by_stem,
            predictions=predictions,
            eval_conf=min(float(c) for c in eval_confs),
            match_iou=match_iou,
            boundary_radius=boundary_radius,
        )
        metrics_by_conf = {
            f"{float(conf):.2f}": _evaluate_predictions(
                gt_by_stem=gt_by_stem,
                predictions=predictions,
                conf_threshold=float(conf),
                match_iou=match_iou,
                boundary_radius=boundary_radius,
            )
            for conf in eval_confs
        }
        preview_paths = _write_previews(
            variant=variant,
            original_image_paths=original_image_paths,
            rendered_image_paths=image_paths,
            gt_by_stem=gt_by_stem,
            predictions=predictions,
            out_dir=preview_dir,
            conf_threshold=min(float(c) for c in eval_confs),
            max_count=preview_count,
        )
        variant_summaries[variant.name] = {
            "label": variant.label,
            "annotation_path": str(variant.annotation_path),
            "image_dir": str(variant.image_dir),
            "transform": variant.transform,
            "image_count": len(image_paths),
            "gt_count": int(sum(len(rows) for rows in gt_by_stem.values())),
            "pred_count_raw": int(sum(len(rows) for rows in predictions.values())),
            "metrics_by_conf": metrics_by_conf,
            "image_level": image_metrics,
            "preview_paths": [str(path) for path in preview_paths],
        }

    selection_conf_key = f"{float(selection_conf):.2f}"
    comparison = _compare_to_baseline(variant_summaries, baseline="rgb_dark")
    ranking = _rank_variants(variant_summaries, baseline="rgb_dark", conf_key=selection_conf_key)
    summary = {
        "title": "LIS Dark Segmentation Front-End Smoke",
        "scope": {
            "what_it_tests": (
                "COCO-pretrained instance segmentation on LIS dark images, comparing the dataset RGB branch, "
                "the packaged RAW PNG branch, and a rule-based edge/contrast-enhanced RAW PNG front end."
            ),
            "important_caveat": (
                "The downloaded LIS packaged RAW files are RGB PNG images, not native CFA mosaics. "
                "This is not a native PerceptionISP CFA/PSF proof; it is a fast segmentation gate for "
                "image-derived edge/contrast evidence."
            ),
            "subset_root": str(subset_root),
            "model": model_name,
            "model_classes_used": sorted(set(YOLO_TO_LIS.values())),
        },
        "run_config": {
            "imgsz": int(imgsz),
            "batch": int(batch),
            "device_requested": str(device),
            "device_used_initial": requested_device,
            "predict_conf": float(predict_conf),
            "predict_iou": float(predict_iou),
            "max_det": int(max_det),
            "match_iou": float(match_iou),
            "boundary_radius": int(boundary_radius),
            "eval_confs": [float(c) for c in eval_confs],
            "selection_conf": float(selection_conf),
        },
        "variants": variant_summaries,
        "comparison_to_rgb_dark": comparison,
        "variant_ranking": ranking,
        "next_decision": _next_decision(comparison, ranking=ranking),
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    (output_dir / "index.html").write_text(_render_html(summary, output_dir), encoding="utf-8")
    return summary


def _variant_specs(subset_root: Path, *, raw_transforms: Sequence[str], cache_dir: Path) -> List[VariantSpec]:
    annotations = subset_root / "annotations"
    variants = [
        VariantSpec(
            name="rgb_dark",
            label="LIS RGB-dark JPG",
            image_dir=subset_root / "RGB-dark" / "JPEGImages",
            annotation_path=_find_subset_annotation(annotations, prefix="lis_coco_JPG_test+1"),
        ),
        VariantSpec(
            name="raw_dark_png",
            label="LIS packaged RAW-dark PNG",
            image_dir=subset_root / "RAW-dark" / "JPEGImages",
            annotation_path=_find_subset_annotation(annotations, prefix="lis_coco_png_test+1"),
        ),
    ]
    for transform in raw_transforms:
        if transform not in TRANSFORM_LABELS:
            raise ValueError(f"unsupported RAW transform: {transform}")
        variants.append(
            VariantSpec(
                name=f"raw_dark_{transform}",
                label=TRANSFORM_LABELS[transform],
                image_dir=subset_root / "RAW-dark" / "JPEGImages",
                annotation_path=_find_subset_annotation(annotations, prefix="lis_coco_png_test+1"),
                transform=transform,
            )
        )
    for variant in variants:
        if not variant.annotation_path.exists():
            raise FileNotFoundError(f"missing annotation file for {variant.name}: {variant.annotation_path}")
        if not variant.image_dir.exists():
            raise FileNotFoundError(f"missing image directory for {variant.name}: {variant.image_dir}")
    cache_dir.mkdir(exist_ok=True)
    return variants


def _find_subset_annotation(annotations_dir: Path, *, prefix: str) -> Path:
    exact_dark32 = annotations_dir / f"{prefix}_dark_test32.json"
    if exact_dark32.exists():
        return exact_dark32
    matches = sorted(path for path in annotations_dir.glob(f"{prefix}*.json") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"missing LIS annotation under {annotations_dir}: {prefix}*.json")
    subset_matches = [path for path in matches if path.name != f"{prefix}.json"]
    return subset_matches[0] if subset_matches else matches[0]


def _load_lis_coco(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    cat_by_id = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
    images = list(payload.get("images", []))
    image_by_id = {int(img["id"]): img for img in images}
    gt_by_stem: Dict[str, List[MaskRecord]] = {Path(str(img["file_name"])).stem: [] for img in images}
    for ann in payload.get("annotations", []):
        if int(ann.get("iscrowd", 0)) != 0 or int(ann.get("ignore", 0)) != 0:
            continue
        image = image_by_id.get(int(ann["image_id"]))
        if image is None:
            continue
        width = int(image["width"])
        height = int(image["height"])
        stem = Path(str(image["file_name"])).stem
        mask = _rasterize_segmentation(ann.get("segmentation"), width=width, height=height, bbox=ann.get("bbox"))
        if not mask.any():
            continue
        bbox = _bbox_xyxy(ann.get("bbox", [0, 0, 0, 0]))
        gt_by_stem.setdefault(stem, []).append(
            MaskRecord(
                stem=stem,
                category=cat_by_id.get(int(ann["category_id"]), str(ann["category_id"])),
                mask=mask,
                bbox=bbox,
                source="gt",
            )
        )
    return {"images": images, "gt_by_stem": gt_by_stem, "categories": cat_by_id}


def _variant_image_paths(variant: VariantSpec, images: Sequence[Mapping[str, Any]]) -> List[Path]:
    paths = []
    for image in images:
        file_name = Path(str(image["file_name"])).name
        path = variant.image_dir / file_name
        if not path.exists():
            matches = sorted(variant.image_dir.glob(Path(file_name).stem + ".*"))
            if not matches:
                raise FileNotFoundError(f"missing image for {variant.name}: {path}")
            path = matches[0]
        paths.append(path)
    return paths


def _rasterize_segmentation(segmentation: Any, *, width: int, height: int, bbox: Any = None) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
            draw.polygon(points, outline=1, fill=1)
    if not np.asarray(mask, dtype=np.uint8).any() and bbox is not None:
        x, y, w, h = [float(value) for value in bbox[:4]]
        draw.rectangle((x, y, x + w, y + h), outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8).astype(bool)


def _bbox_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in bbox[:4]]
    return (x, y, x + w, y + h)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _model_names(model: Any) -> Dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _predict_with_fallback(
    *,
    model: Any,
    image_paths: Sequence[Path],
    source_stems: Sequence[str],
    yolo_names: Mapping[int, str],
    imgsz: int,
    batch: int,
    device: str,
    predict_conf: float,
    predict_iou: float,
    max_det: int,
) -> Dict[str, List[MaskRecord]]:
    try:
        return _predict(
            model=model,
            image_paths=image_paths,
            source_stems=source_stems,
            yolo_names=yolo_names,
            imgsz=imgsz,
            batch=batch,
            device=device,
            predict_conf=predict_conf,
            predict_iou=predict_iou,
            max_det=max_det,
        )
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"prediction failed on device={device}; retrying on cpu: {exc}")
        return _predict(
            model=model,
            image_paths=image_paths,
            source_stems=source_stems,
            yolo_names=yolo_names,
            imgsz=imgsz,
            batch=batch,
            device="cpu",
            predict_conf=predict_conf,
            predict_iou=predict_iou,
            max_det=max_det,
        )


def _predict(
    *,
    model: Any,
    image_paths: Sequence[Path],
    source_stems: Sequence[str],
    yolo_names: Mapping[int, str],
    imgsz: int,
    batch: int,
    device: str,
    predict_conf: float,
    predict_iou: float,
    max_det: int,
) -> Dict[str, List[MaskRecord]]:
    if len(image_paths) != len(source_stems):
        raise ValueError("image_paths and source_stems must have the same length")
    predictions: Dict[str, List[MaskRecord]] = {}
    for start in range(0, len(image_paths), max(1, int(batch))):
        batch_paths = list(image_paths[start : start + max(1, int(batch))])
        batch_stems = list(source_stems[start : start + max(1, int(batch))])
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=int(imgsz),
            batch=len(batch_paths),
            device=str(device),
            conf=float(predict_conf),
            iou=float(predict_iou),
            max_det=int(max_det),
            retina_masks=True,
            verbose=False,
            stream=False,
        )
        for path, stem, result in zip(batch_paths, batch_stems, results):
            image = Image.open(path)
            width, height = image.size
            rows: List[MaskRecord] = []
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is not None and masks is not None and len(boxes) > 0:
                cls_values = boxes.cls.detach().cpu().numpy().astype(int)
                conf_values = boxes.conf.detach().cpu().numpy()
                mask_values = masks.data.detach().cpu().numpy()
                for index, cls_value in enumerate(cls_values):
                    yolo_name = yolo_names.get(int(cls_value), str(cls_value))
                    category = YOLO_TO_LIS.get(yolo_name.lower())
                    if category is None:
                        continue
                    mask = np.asarray(mask_values[index] > 0.5, dtype=bool)
                    if mask.shape != (height, width):
                        mask = _resize_mask(mask, width=width, height=height)
                    if not mask.any():
                        continue
                    rows.append(
                        MaskRecord(
                            stem=stem,
                            category=category,
                            mask=mask,
                            bbox=_mask_bbox(mask),
                            conf=float(conf_values[index]),
                            source="pred",
                        )
                    )
            predictions[stem] = rows
    return predictions


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
    if ys.size == 0 or xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _transform_image(path: Path, cache_root: Path, transform: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{path.stem}_{transform}.jpg"
    if out_path.exists() and out_path.stat().st_mtime >= path.stat().st_mtime:
        return out_path
    image = Image.open(path).convert("RGB")
    if transform == "edge_contrast":
        out = _edge_contrast_boost(np.asarray(image), clip_limit=1.8, sharp_amount=0.35, edge_gain=0.18)
    elif transform == "edge_contrast_mild":
        out = _edge_contrast_boost(np.asarray(image), clip_limit=1.4, sharp_amount=0.22, edge_gain=0.10)
    elif transform == "edge_contrast_strong":
        out = _edge_contrast_boost(np.asarray(image), clip_limit=2.3, sharp_amount=0.55, edge_gain=0.28)
    elif transform == "clahe":
        out = _clahe_luma(np.asarray(image), clip_limit=1.8)
    elif transform == "gamma_bright":
        out = _gamma_luma(np.asarray(image), gamma=0.72)
    elif transform == "unsharp":
        out = _unsharp_luma(np.asarray(image), sharp_amount=0.45)
    else:
        raise ValueError(f"unsupported transform: {transform}")
    Image.fromarray(out).save(out_path, quality=94)
    return out_path


def _edge_contrast_boost(
    rgb: np.ndarray,
    *,
    clip_limit: float,
    sharp_amount: float,
    edge_gain: float,
) -> np.ndarray:
    try:
        import cv2

        arr = np.asarray(rgb, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        l_eq = clahe.apply(l_chan)
        blur = cv2.GaussianBlur(l_eq, (0, 0), 1.0)
        l_sharp = cv2.addWeighted(l_eq, 1.0 + float(sharp_amount), blur, -float(sharp_amount), 0)
        edges = cv2.Canny(l_sharp, 40, 120)
        edges = cv2.GaussianBlur(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1), (0, 0), 0.8)
        gain_map = 1.0 + float(edge_gain) * (edges.astype(np.float32) / 255.0)
        boosted_l = np.clip(l_sharp.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((boosted_l, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    except Exception:
        return _unsharp_luma(_clahe_luma(rgb, clip_limit=clip_limit), sharp_amount=sharp_amount)


def _clahe_luma(rgb: np.ndarray, *, clip_limit: float) -> np.ndarray:
    try:
        import cv2

        arr = np.asarray(rgb, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l_chan), a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    except Exception:
        return np.asarray(Image.fromarray(rgb).convert("RGB"), dtype=np.uint8)


def _gamma_luma(rgb: np.ndarray, *, gamma: float) -> np.ndarray:
    try:
        import cv2

        arr = np.asarray(rgb, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        lut = np.asarray([np.clip((i / 255.0) ** float(gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        return cv2.cvtColor(cv2.merge((lut[l_chan], a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    except Exception:
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return np.clip(np.power(arr, float(gamma)) * 255.0, 0, 255).astype(np.uint8)


def _unsharp_luma(rgb: np.ndarray, *, sharp_amount: float) -> np.ndarray:
    try:
        import cv2

        arr = np.asarray(rgb, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        blur = cv2.GaussianBlur(l_chan, (0, 0), 1.0)
        l_sharp = cv2.addWeighted(l_chan, 1.0 + float(sharp_amount), blur, -float(sharp_amount), 0)
        return cv2.cvtColor(cv2.merge((l_sharp, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    except Exception:
        gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
        blur = np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
        detail = np.clip(gray + float(sharp_amount) * (gray - blur), 0, 255)
        scale = (detail + 1.0) / (gray + 1.0)
        return np.clip(rgb.astype(np.float32) * scale[..., None], 0, 255).astype(np.uint8)


def _evaluate_predictions(
    *,
    gt_by_stem: Mapping[str, Sequence[MaskRecord]],
    predictions: Mapping[str, Sequence[MaskRecord]],
    conf_threshold: float,
    match_iou: float,
    boundary_radius: int,
) -> Dict[str, Any]:
    totals = _empty_counts()
    per_class: Dict[str, Dict[str, Any]] = {}
    matches: List[MatchRecord] = []
    for stem, gt_rows in gt_by_stem.items():
        pred_rows = [row for row in predictions.get(stem, []) if row.conf >= conf_threshold]
        eval_result = _match_image(
            stem=stem,
            gt_rows=gt_rows,
            pred_rows=pred_rows,
            match_iou=match_iou,
            boundary_radius=boundary_radius,
        )
        _merge_counts(totals, eval_result["counts"])
        matches.extend(eval_result["matches"])
        for class_name, class_counts in eval_result["per_class"].items():
            _merge_counts(per_class.setdefault(class_name, _empty_counts()), class_counts)
    metrics = _counts_to_metrics(totals, matches)
    return {
        **metrics,
        "per_class": {name: _counts_to_metrics(counts, []) for name, counts in sorted(per_class.items())},
    }


def _image_level_metrics(
    *,
    gt_by_stem: Mapping[str, Sequence[MaskRecord]],
    predictions: Mapping[str, Sequence[MaskRecord]],
    eval_conf: float,
    match_iou: float,
    boundary_radius: int,
) -> Dict[str, Any]:
    rows = []
    for stem, gt_rows in gt_by_stem.items():
        pred_rows = [row for row in predictions.get(stem, []) if row.conf >= eval_conf]
        result = _match_image(
            stem=stem,
            gt_rows=gt_rows,
            pred_rows=pred_rows,
            match_iou=match_iou,
            boundary_radius=boundary_radius,
        )
        rows.append({"stem": stem, **_counts_to_metrics(result["counts"], result["matches"])})
    return {"conf": float(eval_conf), "rows": rows}


def _match_image(
    *,
    stem: str,
    gt_rows: Sequence[MaskRecord],
    pred_rows: Sequence[MaskRecord],
    match_iou: float,
    boundary_radius: int,
) -> Dict[str, Any]:
    counts = _empty_counts()
    per_class: Dict[str, Dict[str, Any]] = {}
    matched_gt: set[int] = set()
    matches: List[MatchRecord] = []
    for pred in sorted(pred_rows, key=lambda row: row.conf, reverse=True):
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_rows):
            if index in matched_gt or gt.category != pred.category:
                continue
            iou = _mask_iou(pred.mask, gt.mask)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        class_counts = per_class.setdefault(pred.category, _empty_counts())
        if best_index >= 0 and best_iou >= match_iou:
            matched_gt.add(best_index)
            counts["tp"] += 1
            class_counts["tp"] += 1
            bf1 = _boundary_f1(pred.mask, gt_rows[best_index].mask, radius=boundary_radius)
            matches.append(MatchRecord(stem=stem, category=pred.category, conf=pred.conf, mask_iou=best_iou, boundary_f1=bf1))
            continue
        counts["fp"] += 1
        class_counts["fp"] += 1
    for index, gt in enumerate(gt_rows):
        class_counts = per_class.setdefault(gt.category, _empty_counts())
        if index not in matched_gt:
            counts["fn"] += 1
            class_counts["fn"] += 1
    return {"counts": counts, "per_class": per_class, "matches": matches}


def _empty_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _merge_counts(target: MutableMapping[str, int], source: Mapping[str, int]) -> None:
    for key in ("tp", "fp", "fn"):
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))


def _counts_to_metrics(counts: Mapping[str, int], matches: Sequence[MatchRecord]) -> Dict[str, Any]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_mask_iou_tp": _mean([row.mask_iou for row in matches]),
        "mean_boundary_f1_tp": _mean([row.boundary_f1 for row in matches]),
    }


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


def _write_previews(
    *,
    variant: VariantSpec,
    original_image_paths: Sequence[Path],
    rendered_image_paths: Sequence[Path],
    gt_by_stem: Mapping[str, Sequence[MaskRecord]],
    predictions: Mapping[str, Sequence[MaskRecord]],
    out_dir: Path,
    conf_threshold: float,
    max_count: int,
) -> List[Path]:
    paths = []
    for original_path, rendered_path in list(zip(original_image_paths, rendered_image_paths))[: max(0, int(max_count))]:
        stem = original_path.stem
        image = np.asarray(Image.open(rendered_path).convert("RGB"), dtype=np.uint8)
        gt_edges = np.zeros(image.shape[:2], dtype=bool)
        for gt in gt_by_stem.get(stem, []):
            gt_edges |= _mask_boundary(gt.mask)
        pred_edges = np.zeros(image.shape[:2], dtype=bool)
        for pred in predictions.get(stem, []):
            if pred.conf >= conf_threshold:
                pred_edges |= _mask_boundary(pred.mask)
        overlay = image.copy()
        overlay[_dilate(gt_edges, radius=1)] = np.array([0, 215, 90], dtype=np.uint8)
        overlay[_dilate(pred_edges, radius=1)] = np.array([238, 60, 70], dtype=np.uint8)
        out_path = out_dir / f"{stem}_{variant.name}.jpg"
        Image.fromarray(overlay).save(out_path, quality=92)
        paths.append(out_path)
    return paths


def _compare_to_baseline(variant_summaries: Mapping[str, Any], *, baseline: str) -> Dict[str, Any]:
    base = variant_summaries.get(baseline)
    if not base:
        return {}
    comparison: Dict[str, Any] = {}
    for variant_name, variant in variant_summaries.items():
        if variant_name == baseline:
            continue
        rows: Dict[str, Any] = {}
        for conf_key, metrics in variant.get("metrics_by_conf", {}).items():
            base_metrics = base.get("metrics_by_conf", {}).get(conf_key, {})
            rows[conf_key] = {
                "delta_precision": _delta(metrics.get("precision"), base_metrics.get("precision")),
                "delta_recall": _delta(metrics.get("recall"), base_metrics.get("recall")),
                "delta_f1": _delta(metrics.get("f1"), base_metrics.get("f1")),
                "delta_mean_mask_iou_tp": _delta(metrics.get("mean_mask_iou_tp"), base_metrics.get("mean_mask_iou_tp")),
                "delta_mean_boundary_f1_tp": _delta(
                    metrics.get("mean_boundary_f1_tp"), base_metrics.get("mean_boundary_f1_tp")
                ),
                "delta_tp": int(metrics.get("tp", 0)) - int(base_metrics.get("tp", 0)),
                "delta_fp": int(metrics.get("fp", 0)) - int(base_metrics.get("fp", 0)),
                "delta_fn": int(metrics.get("fn", 0)) - int(base_metrics.get("fn", 0)),
            }
        comparison[variant_name] = rows
    return comparison


def _rank_variants(variant_summaries: Mapping[str, Any], *, baseline: str, conf_key: str) -> List[Dict[str, Any]]:
    rows = []
    base_metrics = variant_summaries.get(baseline, {}).get("metrics_by_conf", {}).get(conf_key, {})
    for variant_name, variant in variant_summaries.items():
        metrics = variant.get("metrics_by_conf", {}).get(conf_key, {})
        rows.append(
            {
                "variant": variant_name,
                "label": variant.get("label", variant_name),
                "conf": conf_key,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "mean_boundary_f1_tp": metrics.get("mean_boundary_f1_tp"),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "delta_f1_vs_baseline": _delta(metrics.get("f1"), base_metrics.get("f1")),
                "delta_recall_vs_baseline": _delta(metrics.get("recall"), base_metrics.get("recall")),
                "delta_boundary_f1_vs_baseline": _delta(
                    metrics.get("mean_boundary_f1_tp"),
                    base_metrics.get("mean_boundary_f1_tp"),
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_value(row.get("f1")),
            _sort_value(row.get("recall")),
            _sort_value(row.get("mean_boundary_f1_tp")),
            -float(row.get("fp") or 0),
        ),
        reverse=True,
    )


def _sort_value(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        number = float(value)
    except Exception:
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def _next_decision(comparison: Mapping[str, Any], *, ranking: Sequence[Mapping[str, Any]]) -> str:
    best = next((row for row in ranking if row.get("variant") != "rgb_dark"), {})
    delta_f1 = best.get("delta_f1_vs_baseline")
    delta_recall = best.get("delta_recall_vs_baseline")
    if delta_f1 is not None and delta_f1 > 0 and (delta_recall is None or delta_recall >= 0):
        return (
            f"{best.get('variant')} improved this segmentation gate; "
            "run the selected front-end on held-out LIS data and then train RGB+Aux segmentation."
        )
    if delta_recall is not None and delta_recall > 0:
        return "The best RAW-derived evidence increases recall but may trade precision; inspect failure cases before training."
    return "No segmentation improvement is proven by this smoke gate; prioritize case mining or native RAW/aux-input training."


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else None


def _render_html(summary: Mapping[str, Any], output_dir: Path) -> str:
    variants = summary.get("variants", {})
    comparison = summary.get("comparison_to_rgb_dark", {})
    ranking = summary.get("variant_ranking", [])
    metric_rows = []
    for variant_name, variant in variants.items():
        for conf_key, metrics in variant.get("metrics_by_conf", {}).items():
            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(str(variant_name))}</td>"
                f"<td>{html.escape(str(conf_key))}</td>"
                f"<td>{metrics.get('tp')}</td>"
                f"<td>{metrics.get('fp')}</td>"
                f"<td>{metrics.get('fn')}</td>"
                f"<td>{_fmt(metrics.get('precision'))}</td>"
                f"<td>{_fmt(metrics.get('recall'))}</td>"
                f"<td>{_fmt(metrics.get('f1'))}</td>"
                f"<td>{_fmt(metrics.get('mean_mask_iou_tp'))}</td>"
                f"<td>{_fmt(metrics.get('mean_boundary_f1_tp'))}</td>"
                "</tr>"
            )
    comparison_rows = []
    for variant_name, confs in comparison.items():
        for conf_key, row in confs.items():
            comparison_rows.append(
                "<tr>"
                f"<td>{html.escape(str(variant_name))}</td>"
                f"<td>{html.escape(str(conf_key))}</td>"
                f"<td>{_fmt(row.get('delta_precision'))}</td>"
                f"<td>{_fmt(row.get('delta_recall'))}</td>"
                f"<td>{_fmt(row.get('delta_f1'))}</td>"
                f"<td>{row.get('delta_tp')}</td>"
                f"<td>{row.get('delta_fp')}</td>"
                f"<td>{row.get('delta_fn')}</td>"
                "</tr>"
            )
    ranking_rows = []
    for rank, row in enumerate(ranking, start=1):
        ranking_rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(str(row.get('variant', '')))}</td>"
            f"<td>{_fmt(row.get('precision'))}</td>"
            f"<td>{_fmt(row.get('recall'))}</td>"
            f"<td>{_fmt(row.get('f1'))}</td>"
            f"<td>{_fmt(row.get('mean_boundary_f1_tp'))}</td>"
            f"<td>{_fmt(row.get('delta_f1_vs_baseline'))}</td>"
            f"<td>{_fmt(row.get('delta_recall_vs_baseline'))}</td>"
            f"<td>{_fmt(row.get('delta_boundary_f1_vs_baseline'))}</td>"
            f"<td>{row.get('tp')}</td>"
            f"<td>{row.get('fp')}</td>"
            "</tr>"
        )
    preview_rows = []
    stems = _preview_stems(variants)
    for stem in stems:
        cells = [f"<td><strong>{html.escape(stem)}</strong></td>"]
        for variant_name, variant in variants.items():
            path = _find_preview_path(variant.get("preview_paths", []), stem, variant_name)
            if path:
                rel = Path(path).resolve().relative_to(output_dir)
                cells.append(
                    f'<td><img src="{html.escape(str(rel))}" alt="{html.escape(stem + " " + variant_name)}"></td>'
                )
            else:
                cells.append("<td></td>")
        preview_rows.append("<tr>" + "".join(cells) + "</tr>")

    tabs = [
        ("summary", "Summary"),
        ("ranking", "Ranking"),
        ("metrics", "Metrics"),
        ("compare", "RGB Delta"),
        ("previews", "Visual Evidence"),
        ("caveats", "Caveats"),
    ]
    tab_buttons = "\n".join(
        f'<button class="tab-button{" active" if idx == 0 else ""}" data-tab="{tab_id}">{label}</button>'
        for idx, (tab_id, label) in enumerate(tabs)
    )
    scope = summary.get("scope", {})
    config = summary.get("run_config", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(str(summary.get("title", "LIS Segmentation Evaluation")))}</title>
  <style>
    :root {{ --ink:#1b2430; --muted:#5f6b7a; --line:#d8dee8; --soft:#f5f7fa; --accent:#1769aa; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #fff; }}
    header {{ padding: 28px 36px 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin-top: 0; font-size: 20px; }}
    p {{ line-height: 1.55; color: var(--muted); max-width: 1120px; }}
    code {{ background: var(--soft); padding: 2px 5px; border-radius: 4px; }}
    .tabs {{ display: flex; gap: 6px; padding: 12px 36px 0; border-bottom: 1px solid var(--line); background: #fff; position: sticky; top: 0; z-index: 2; }}
    .tab-button {{ border: 1px solid var(--line); border-bottom: 0; background: var(--soft); color: var(--ink); padding: 10px 14px; border-radius: 7px 7px 0 0; cursor: pointer; font-weight: 600; }}
    .tab-button.active {{ background: #fff; color: var(--accent); }}
    section {{ display: none; padding: 26px 36px 42px; }}
    section.active {{ display: block; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: var(--soft); font-weight: 700; }}
    .callout {{ border-left: 4px solid var(--accent); background: #f7fbff; padding: 12px 14px; margin: 14px 0 20px; max-width: 1120px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; max-width: 1120px; }}
    .tile {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }}
    .tile b {{ display: block; margin-bottom: 6px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid var(--line); }}
    .preview-table td {{ width: 25%; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(str(summary.get("title", "LIS Segmentation Evaluation")))}</h1>
    <p>{html.escape(str(scope.get("what_it_tests", "")))}</p>
  </header>
  <nav class="tabs">{tab_buttons}</nav>

  <section id="summary" class="active">
    <h2>Summary</h2>
    <div class="callout"><strong>Decision:</strong> {html.escape(str(summary.get("next_decision", "")))}</div>
    <div class="grid">
      <div class="tile"><b>Dataset</b><code>{html.escape(str(scope.get("subset_root", "")))}</code></div>
      <div class="tile"><b>Model</b><code>{html.escape(str(scope.get("model", "")))}</code></div>
      <div class="tile"><b>Gate</b>same-class mask IoU >= {html.escape(str(config.get("match_iou", "")))}, boundary radius {html.escape(str(config.get("boundary_radius", "")))} px</div>
    </div>
    <p>{html.escape(str(scope.get("important_caveat", "")))}</p>
  </section>

  <section id="ranking">
    <h2>Selection Ranking</h2>
    <p>Variants are ranked by F1 at the configured selection confidence, then recall and matched boundary F1. Deltas are relative to RGB-dark.</p>
    <table>
      <thead><tr><th>Rank</th><th>Variant</th><th>Precision</th><th>Recall</th><th>F1</th><th>Boundary F1</th><th>Delta F1</th><th>Delta Recall</th><th>Delta Boundary F1</th><th>TP</th><th>FP</th></tr></thead>
      <tbody>{''.join(ranking_rows)}</tbody>
    </table>
  </section>

  <section id="metrics">
    <h2>Metrics</h2>
    <p>Metrics are fixed-confidence, same-class instance-mask matching metrics. Boundary F1 is measured only on true-positive matched masks.</p>
    <table>
      <thead><tr><th>Variant</th><th>Conf</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mean Mask IoU</th><th>Mean Boundary F1</th></tr></thead>
      <tbody>{''.join(metric_rows)}</tbody>
    </table>
  </section>

  <section id="compare">
    <h2>Delta vs RGB-dark</h2>
    <p>Positive delta means the variant is higher than the LIS RGB-dark JPG baseline at the same confidence threshold.</p>
    <table>
      <thead><tr><th>Variant</th><th>Conf</th><th>Delta Precision</th><th>Delta Recall</th><th>Delta F1</th><th>Delta TP</th><th>Delta FP</th><th>Delta FN</th></tr></thead>
      <tbody>{''.join(comparison_rows)}</tbody>
    </table>
  </section>

  <section id="previews">
    <h2>Visual Evidence</h2>
    <p>Green lines are GT mask boundaries; red lines are predicted mask boundaries at the lowest evaluated confidence threshold.</p>
    <table class="preview-table">
      <thead><tr><th>Stem</th>{''.join(f'<th>{html.escape(str(name))}</th>' for name in variants.keys())}</tr></thead>
      <tbody>{''.join(preview_rows)}</tbody>
    </table>
  </section>

  <section id="caveats">
    <h2>Caveats</h2>
    <ul>
      <li>The LIS packaged RAW branch is decoded RGB PNG, not native Bayer/CFA raw.</li>
      <li>The segmenter is COCO-pretrained and is not fine-tuned on LIS or Aux maps.</li>
      <li>The edge/contrast variant is a rule-based image transform, not an RGB+Aux DNN input.</li>
      <li>A positive result here justifies expanding the LIS subset and training an RGB+Aux segmentation model; a negative result does not disprove native RAW/CFA-aware PerceptionISP.</li>
    </ul>
  </section>

  <script>
    for (const button of document.querySelectorAll('.tab-button')) {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
      }});
    }}
  </script>
</body>
</html>
"""


def _preview_stems(variants: Mapping[str, Any]) -> List[str]:
    stems: List[str] = []
    for variant in variants.values():
        for raw_path in variant.get("preview_paths", []):
            name = Path(str(raw_path)).name
            parts = name.split("_")
            if parts and parts[0] not in stems:
                stems.append(parts[0])
    return stems


def _find_preview_path(paths: Sequence[str], stem: str, variant_name: str) -> str | None:
    suffix = f"{stem}_{variant_name}.jpg"
    for path in paths:
        if str(path).endswith(suffix):
            return str(path)
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except Exception:
        return html.escape(str(value))
    if not math.isfinite(number):
        return ""
    return f"{number:.4f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (MatchRecord, MaskRecord, VariantSpec)):
        return _json_ready(value.__dict__)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
