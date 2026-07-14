"""Export LIS COCO polygon annotations to a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image

from perception_isp.evaluation.lis_segmentation_eval import TRANSFORM_LABELS, _clahe_luma, _edge_contrast_boost, _gamma_luma, _unsharp_luma


LIS_NAMES = ["bicycle", "chair", "diningtable", "bottle", "motorbike", "car", "tvmonitor", "bus"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create YOLO-seg train/val datasets from LIS subset roots.")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=("rgb_dark", "raw_dark_png", "raw_dark_clahe", "raw_dark_edge_contrast"),
        default="raw_dark_clahe",
    )
    parser.add_argument("--copy", action="store_true", help="Copy source images instead of hardlinking when possible.")
    args = parser.parse_args(argv)

    summary = export_lis_yolo_seg_dataset(
        train_root=Path(args.train_root),
        val_root=Path(args.val_root),
        output_dir=Path(args.output_dir),
        variant=str(args.variant),
        prefer_hardlink=not bool(args.copy),
    )
    print(json.dumps(summary, indent=2))
    return 0


def export_lis_yolo_seg_dataset(
    *,
    train_root: Path,
    val_root: Path,
    output_dir: Path,
    variant: str,
    prefer_hardlink: bool = True,
) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_summary = _export_split(train_root.resolve(), output_dir, split="train", variant=variant, prefer_hardlink=prefer_hardlink)
    val_summary = _export_split(val_root.resolve(), output_dir, split="val", variant=variant, prefer_hardlink=prefer_hardlink)
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(LIS_NAMES)],
                "",
            ]
        )
    )
    summary = {
        "status": "pass" if train_summary["status"] == "pass" and val_summary["status"] == "pass" else "fail",
        "variant": variant,
        "variant_label": _variant_label(variant),
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "train": train_summary,
        "val": val_summary,
        "caveat": "LIS packaged RAW images are RGB PNG files, not native CFA RAW.",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _export_split(
    subset_root: Path,
    output_dir: Path,
    *,
    split: str,
    variant: str,
    prefer_hardlink: bool,
) -> Dict[str, Any]:
    annotation_path = _annotation_path(subset_root, variant=variant)
    payload = json.loads(annotation_path.read_text())
    categories = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
    image_by_id = {int(image["id"]): image for image in payload.get("images", [])}
    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {image_id: [] for image_id in image_by_id}
    for ann in payload.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id in annotations_by_image and int(ann.get("ignore", 0)) == 0 and int(ann.get("iscrowd", 0)) == 0:
            annotations_by_image[image_id].append(ann)

    image_out = output_dir / "images" / split
    label_out = output_dir / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    linked = 0
    transformed = 0
    missing_images: List[str] = []
    skipped_annotations = 0
    label_rows = 0
    for image_id, image in sorted(image_by_id.items()):
        source = _source_image_path(subset_root, image, variant=variant)
        if not source.exists():
            missing_images.append(str(source))
            continue
        stem = source.stem
        suffix = ".jpg" if _requires_transform(variant) or source.suffix.lower() == ".jpg" else source.suffix.lower()
        dst_image = image_out / f"{stem}{suffix}"
        if _requires_transform(variant):
            _write_transformed_image(source, dst_image, variant=variant)
            transformed += 1
        else:
            link_result = _link_or_copy(source, dst_image, prefer_hardlink=prefer_hardlink)
            linked += int(link_result == "linked")
            copied += int(link_result == "copied")

        rows = []
        width = float(image["width"])
        height = float(image["height"])
        for ann in annotations_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in categories:
                skipped_annotations += 1
                continue
            polygon = _largest_polygon(ann.get("segmentation"))
            if len(polygon) < 6:
                skipped_annotations += 1
                continue
            values = _normalize_polygon(polygon, width=width, height=height)
            if len(values) < 6:
                skipped_annotations += 1
                continue
            rows.append(" ".join([str(cat_id), *[f"{value:.6f}" for value in values]]))
        (label_out / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        label_rows += len(rows)

    summary = {
        "status": "pass" if not missing_images else "fail",
        "subset_root": str(subset_root),
        "annotation_path": str(annotation_path),
        "image_count": len(image_by_id),
        "label_rows": label_rows,
        "missing_images": missing_images[:20],
        "skipped_annotations": skipped_annotations,
        "linked_images": linked,
        "copied_images": copied,
        "transformed_images": transformed,
    }
    return summary


def _annotation_path(subset_root: Path, *, variant: str) -> Path:
    annotations = subset_root / "annotations"
    prefix = "lis_coco_JPG_test+1" if variant == "rgb_dark" else "lis_coco_png_test+1"
    matches = sorted(path for path in annotations.glob(f"{prefix}*.json") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"missing annotation under {annotations}: {prefix}*.json")
    return matches[0]


def _source_image_path(subset_root: Path, image: Mapping[str, Any], *, variant: str) -> Path:
    branch = "RGB-dark" if variant == "rgb_dark" else "RAW-dark"
    file_name = Path(str(image["file_name"])).name
    source = subset_root / branch / "JPEGImages" / file_name
    if source.exists():
        return source
    matches = sorted((subset_root / branch / "JPEGImages").glob(Path(file_name).stem + ".*"))
    if not matches:
        return source
    return matches[0]


def _requires_transform(variant: str) -> bool:
    return variant.startswith("raw_dark_") and variant != "raw_dark_png"


def _write_transformed_image(source: Path, destination: Path, *, variant: str) -> None:
    if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
        return
    arr = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    if variant == "raw_dark_clahe":
        out = _clahe_luma(arr, clip_limit=1.8)
    elif variant == "raw_dark_edge_contrast":
        out = _edge_contrast_boost(arr, clip_limit=1.8, sharp_amount=0.35, edge_gain=0.18)
    elif variant == "raw_dark_gamma_bright":
        out = _gamma_luma(arr, gamma=0.72)
    elif variant == "raw_dark_unsharp":
        out = _unsharp_luma(arr, sharp_amount=0.45)
    else:
        raise ValueError(f"unsupported transformed variant: {variant}")
    Image.fromarray(out).save(destination, quality=94)


def _link_or_copy(source: Path, destination: Path, *, prefer_hardlink: bool) -> str:
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return "linked" if destination.stat().st_ino == source.stat().st_ino else "copied"
    if destination.exists():
        destination.unlink()
    if prefer_hardlink:
        try:
            os.link(source, destination)
            return "linked"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copied"


def _largest_polygon(segmentation: Any) -> List[float]:
    if not isinstance(segmentation, list):
        return []
    polygons = [poly for poly in segmentation if isinstance(poly, list) and len(poly) >= 6]
    if not polygons:
        return []
    return list(max(polygons, key=len))


def _normalize_polygon(polygon: Sequence[float], *, width: float, height: float) -> List[float]:
    values: List[float] = []
    for index in range(0, len(polygon) - 1, 2):
        x = min(max(float(polygon[index]) / max(width, 1.0), 0.0), 1.0)
        y = min(max(float(polygon[index + 1]) / max(height, 1.0), 0.0), 1.0)
        values.extend([x, y])
    return values if len(values) >= 6 else []


def _variant_label(variant: str) -> str:
    if variant == "rgb_dark":
        return "LIS RGB-dark JPG"
    if variant == "raw_dark_png":
        return "LIS packaged RAW-dark PNG"
    transform = variant.removeprefix("raw_dark_")
    return TRANSFORM_LABELS.get(transform, variant)


if __name__ == "__main__":
    raise SystemExit(main())
