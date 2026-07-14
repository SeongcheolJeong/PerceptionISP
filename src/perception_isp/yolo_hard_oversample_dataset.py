"""Create a YOLO dataset with small/thin-object train images oversampled."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Oversample YOLO train images that contain small or thin objects.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--hard-repeat", type=int, default=3, help="Total copies for hard train images, including the original.")
    parser.add_argument("--small-area", type=float, default=0.02)
    parser.add_argument("--thin-aspect", type=float, default=3.0)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = create_hard_oversample_dataset(
        source=Path(args.source),
        destination=Path(args.destination),
        hard_repeat=int(args.hard_repeat),
        small_area=float(args.small_area),
        thin_aspect=float(args.thin_aspect),
        copy=bool(args.copy),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def create_hard_oversample_dataset(
    *,
    source: Path,
    destination: Path,
    hard_repeat: int = 3,
    small_area: float = 0.02,
    thin_aspect: float = 3.0,
    copy: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} exists; pass --overwrite")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    source_yaml = _read_yaml_like(source / "data.yaml")
    channels = int(source_yaml.get("channels", 3))
    names = source_yaml.get("names", [])
    train_stats = _copy_split(
        source=source,
        destination=destination,
        split="train",
        hard_repeat=max(int(hard_repeat), 1),
        small_area=small_area,
        thin_aspect=thin_aspect,
        copy=copy,
    )
    val_stats = _copy_split(
        source=source,
        destination=destination,
        split="val",
        hard_repeat=1,
        small_area=small_area,
        thin_aspect=thin_aspect,
        copy=copy,
    )
    data_yaml = (
        f"path: {destination.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"channels: {channels}\n"
        f"names: {names}\n"
    )
    (destination / "data.yaml").write_text(data_yaml)
    summary = {
        "source": str(source),
        "destination": str(destination.resolve()),
        "hard_repeat": int(hard_repeat),
        "small_area": float(small_area),
        "thin_aspect": float(thin_aspect),
        "copy": bool(copy),
        "channels": int(channels),
        "names": names,
        "train": train_stats,
        "val": val_stats,
    }
    (destination / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary


def _copy_split(
    *,
    source: Path,
    destination: Path,
    split: str,
    hard_repeat: int,
    small_area: float,
    thin_aspect: float,
    copy: bool,
) -> Dict[str, Any]:
    src_images = source / "images" / split
    src_labels = source / "labels" / split
    dst_images = destination / "images" / split
    dst_labels = destination / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    hard_images = 0
    source_images = 0
    output_images = 0
    hard_box_count = 0
    total_box_count = 0
    for label_path in sorted(src_labels.glob("*.txt")):
        stem = label_path.stem
        image_files = sorted(src_images.glob(f"{stem}.*"))
        if not image_files:
            continue
        source_images += 1
        hard, hard_boxes, total_boxes = _is_hard_label(label_path, small_area=small_area, thin_aspect=thin_aspect)
        total_box_count += total_boxes
        hard_box_count += hard_boxes
        repeat = hard_repeat if hard else 1
        if hard:
            hard_images += 1
        for copy_index in range(repeat):
            suffix = "" if copy_index == 0 else f"_harddup{copy_index:02d}"
            out_stem = f"{stem}{suffix}"
            for image_path in image_files:
                _link_or_copy(image_path, dst_images / f"{out_stem}{image_path.suffix}", copy=copy)
            _link_or_copy(label_path, dst_labels / f"{out_stem}.txt", copy=copy)
            output_images += 1
    return {
        "source_images": int(source_images),
        "output_images": int(output_images),
        "hard_images": int(hard_images),
        "total_box_count": int(total_box_count),
        "hard_box_count": int(hard_box_count),
        "hard_fraction": float(hard_images / source_images) if source_images else 0.0,
    }


def _is_hard_label(label_path: Path, *, small_area: float, thin_aspect: float) -> tuple[bool, int, int]:
    hard_count = 0
    total = 0
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        _, _, _, w, h = [float(value) for value in line.split()[:5]]
        area = w * h
        aspect = max(w / h if h > 0 else np.inf, h / w if w > 0 else np.inf)
        if area < small_area or aspect > thin_aspect:
            hard_count += 1
    return hard_count > 0, hard_count, total


def _link_or_copy(source: Path, destination: Path, *, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


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
            key = key.strip()
            value = value.strip().strip("'\"")
            if value.startswith("[") and value.endswith("]"):
                payload[key] = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            else:
                payload[key] = value
        return payload


if __name__ == "__main__":
    raise SystemExit(main())
