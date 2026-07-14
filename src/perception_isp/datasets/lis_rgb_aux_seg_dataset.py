"""Build LIS YOLO-seg RGB+Aux datasets with 4-channel NumPy sidecars."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image


LIS_NAMES = ["bicycle", "chair", "diningtable", "bottle", "motorbike", "car", "tvmonitor", "bus"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a YOLO-seg RGB+Aux 4-channel sidecar dataset.")
    parser.add_argument("--rgb-yolo-root", required=True, help="Existing RGB YOLO-seg dataset root.")
    parser.add_argument("--aux-yolo-root", required=True, help="Existing transformed RAW/Aux YOLO-seg dataset root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aux-mode", choices=("sobel_luma", "luma"), default="sobel_luma")
    parser.add_argument("--copy", action="store_true", help="Copy RGB images/labels instead of hardlinking.")
    args = parser.parse_args(argv)

    summary = export_rgb_aux_dataset(
        rgb_yolo_root=Path(args.rgb_yolo_root),
        aux_yolo_root=Path(args.aux_yolo_root),
        output_dir=Path(args.output_dir),
        aux_mode=str(args.aux_mode),
        prefer_hardlink=not bool(args.copy),
    )
    print(json.dumps(summary, indent=2))
    return 0


def export_rgb_aux_dataset(
    *,
    rgb_yolo_root: Path,
    aux_yolo_root: Path,
    output_dir: Path,
    aux_mode: str = "sobel_luma",
    prefer_hardlink: bool = True,
) -> Dict[str, Any]:
    rgb_yolo_root = rgb_yolo_root.resolve()
    aux_yolo_root = aux_yolo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries = {}
    for split in ("train", "val"):
        split_summaries[split] = _export_split(
            rgb_yolo_root=rgb_yolo_root,
            aux_yolo_root=aux_yolo_root,
            output_dir=output_dir,
            split=split,
            aux_mode=aux_mode,
            prefer_hardlink=prefer_hardlink,
        )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/val",
                "channels: 4",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(LIS_NAMES)],
                "",
            ]
        )
    )

    summary = {
        "status": "pass" if all(v["status"] == "pass" for v in split_summaries.values()) else "fail",
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "rgb_yolo_root": str(rgb_yolo_root),
        "aux_yolo_root": str(aux_yolo_root),
        "channels": 4,
        "channel_order": "RGB + Aux",
        "aux_mode": aux_mode,
        "splits": split_summaries,
        "caveat": "LIS packaged RAW images are RGB PNG-derived images, not native CFA RAW.",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _export_split(
    *,
    rgb_yolo_root: Path,
    aux_yolo_root: Path,
    output_dir: Path,
    split: str,
    aux_mode: str,
    prefer_hardlink: bool,
) -> Dict[str, Any]:
    src_img_dir = rgb_yolo_root / "images" / split
    src_label_dir = rgb_yolo_root / "labels" / split
    aux_img_dir = aux_yolo_root / "images" / split
    dst_img_dir = output_dir / "images" / split
    dst_label_dir = output_dir / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    missing_aux = []
    linked_images = copied_images = linked_labels = copied_labels = npy_written = 0
    image_paths = sorted(path for path in src_img_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for rgb_path in image_paths:
        aux_path = _match_aux(aux_img_dir, rgb_path.stem)
        if aux_path is None:
            missing_aux.append(rgb_path.stem)
            continue

        dst_image = dst_img_dir / rgb_path.name
        image_result = _link_or_copy(rgb_path, dst_image, prefer_hardlink=prefer_hardlink)
        linked_images += int(image_result == "linked")
        copied_images += int(image_result == "copied")

        src_label = src_label_dir / f"{rgb_path.stem}.txt"
        dst_label = dst_label_dir / src_label.name
        label_result = _link_or_copy(src_label, dst_label, prefer_hardlink=prefer_hardlink)
        linked_labels += int(label_result == "linked")
        copied_labels += int(label_result == "copied")

        npy_path = dst_image.with_suffix(".npy")
        if not npy_path.exists() or npy_path.stat().st_mtime < max(rgb_path.stat().st_mtime, aux_path.stat().st_mtime):
            np.save(npy_path, _make_rgb_aux(rgb_path, aux_path, aux_mode=aux_mode), allow_pickle=False)
            npy_written += 1

    return {
        "status": "pass" if not missing_aux else "fail",
        "split": split,
        "image_count": len(image_paths),
        "missing_aux": missing_aux[:20],
        "linked_images": linked_images,
        "copied_images": copied_images,
        "linked_labels": linked_labels,
        "copied_labels": copied_labels,
        "npy_written": npy_written,
    }


def _match_aux(aux_img_dir: Path, stem: str) -> Path | None:
    matches = sorted(path for path in aux_img_dir.glob(stem + ".*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return matches[0] if matches else None


def _make_rgb_aux(rgb_path: Path, aux_path: Path, *, aux_mode: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; install perception-isp[ml]") from exc

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    aux_rgb = np.asarray(Image.open(aux_path).convert("RGB"), dtype=np.uint8)
    if aux_rgb.shape[:2] != rgb.shape[:2]:
        aux_rgb = cv2.resize(aux_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    aux_luma = cv2.cvtColor(aux_rgb, cv2.COLOR_RGB2GRAY)
    if aux_mode == "luma":
        aux = aux_luma
    elif aux_mode == "sobel_luma":
        grad_x = cv2.Sobel(aux_luma, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(aux_luma, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        aux = np.clip(mag / max(float(np.percentile(mag, 99.0)), 1.0) * 255.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unsupported aux_mode: {aux_mode}")
    return np.dstack([rgb, aux[..., None]]).astype(np.uint8, copy=False)


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


if __name__ == "__main__":
    raise SystemExit(main())
