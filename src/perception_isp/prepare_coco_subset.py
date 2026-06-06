"""Prepare a YOLO-format COCO val subset without downloading full COCO train."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .types import json_ready
from .yolo_dataset import COCO80_CLASS_NAMES


DEFAULT_LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
DEFAULT_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/{split}/{filename}"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format COCO val subset for PerceptionISP evaluation.")
    parser.add_argument("--output-dir", default="data/coco_val2017_1k")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--labels-url", default=DEFAULT_LABELS_URL)
    parser.add_argument("--labels-zip", default=None, help="Existing coco2017labels.zip path; skips label download when present.")
    parser.add_argument("--cache-dir", default="data/.cache/coco2017labels")
    parser.add_argument("--image-url-template", default=DEFAULT_IMAGE_URL_TEMPLATE)
    parser.add_argument("--image-source-dir", default=None, help="Optional local image directory for tests/offline preparation.")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    summary = prepare_coco_val_subset(
        output_dir=args.output_dir,
        count=int(args.count),
        split=str(args.split),
        labels_url=str(args.labels_url),
        labels_zip=args.labels_zip,
        cache_dir=args.cache_dir,
        image_url_template=str(args.image_url_template),
        image_source_dir=args.image_source_dir,
        threads=int(args.threads),
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def prepare_coco_val_subset(
    *,
    output_dir: str | Path,
    count: int = 1000,
    split: str = "val2017",
    labels_url: str = DEFAULT_LABELS_URL,
    labels_zip: str | Path | None = None,
    cache_dir: str | Path = "data/.cache/coco2017labels",
    image_url_template: str = DEFAULT_IMAGE_URL_TEMPLATE,
    image_source_dir: str | Path | None = None,
    threads: int = 8,
) -> Dict[str, Any]:
    start = time.perf_counter()
    destination = Path(output_dir).expanduser()
    image_dir = destination / "images" / split
    label_dir = destination / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = Path(labels_zip).expanduser() if labels_zip else cache_root.with_suffix(".zip")
    if not zip_path.exists():
        _download_file(labels_url, zip_path)
    _extract_labels_zip(zip_path, cache_root)

    selected = _selected_filenames(cache_root, split=split, count=count)
    source_label_dir = _find_label_dir(cache_root, split)
    copied_labels = 0
    for filename in selected:
        source_label = source_label_dir / Path(filename).with_suffix(".txt").name
        target_label = label_dir / source_label.name
        if source_label.exists():
            shutil.copy2(source_label, target_label)
            copied_labels += 1
        else:
            target_label.write_text("")

    image_results = _prepare_images(
        selected,
        split=split,
        image_dir=image_dir,
        image_url_template=image_url_template,
        image_source_dir=Path(image_source_dir).expanduser() if image_source_dir else None,
        threads=threads,
    )
    _write_data_yaml(destination)
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    summary = {
        "output_dir": str(destination),
        "split": split,
        "requested_count": int(count),
        "image_count": int(len(selected)),
        "label_file_count": int(copied_labels),
        "empty_label_file_count": int(len(selected) - copied_labels),
        "downloaded_image_count": int(sum(1 for item in image_results if item["status"] == "downloaded")),
        "copied_image_count": int(sum(1 for item in image_results if item["status"] == "copied")),
        "cached_image_count": int(sum(1 for item in image_results if item["status"] == "cached")),
        "failed_image_count": int(sum(1 for item in image_results if item["status"] == "failed")),
        "failed_images": [item for item in image_results if item["status"] == "failed"],
        "labels_zip": str(zip_path),
        "cache_dir": str(cache_root),
        "data_yaml": str(destination / "data.yaml"),
        "elapsed_seconds": float(elapsed),
        "images_per_second": float(len(selected) / elapsed),
    }
    (destination / "prepare_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    if summary["failed_image_count"]:
        raise RuntimeError(f"failed to prepare {summary['failed_image_count']} COCO images")
    return summary


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(str(url), temp_path)
    temp_path.replace(path)


def _extract_labels_zip(zip_path: Path, cache_root: Path) -> None:
    marker = cache_root / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_root)
    marker.write_text(str(zip_path) + "\n")


def _selected_filenames(cache_root: Path, *, split: str, count: int) -> Tuple[str, ...]:
    split_list = _find_split_list(cache_root, split)
    names = []
    for line in split_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        names.append(Path(line).name)
        if len(names) >= max(int(count), 0):
            break
    if not names:
        label_dir = _find_label_dir(cache_root, split)
        names = [path.with_suffix(".jpg").name for path in sorted(label_dir.glob("*.txt"))[: max(int(count), 0)]]
    return tuple(names)


def _find_split_list(cache_root: Path, split: str) -> Path:
    candidates = sorted(cache_root.rglob(f"{split}.txt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"{split}.txt not found under {cache_root}")


def _find_label_dir(cache_root: Path, split: str) -> Path:
    candidates = sorted(path for path in cache_root.rglob(split) if path.is_dir() and path.parent.name == "labels")
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"labels/{split} not found under {cache_root}")


def _prepare_images(
    filenames: Sequence[str],
    *,
    split: str,
    image_dir: Path,
    image_url_template: str,
    image_source_dir: Path | None,
    threads: int,
) -> Tuple[Dict[str, Any], ...]:
    with ThreadPoolExecutor(max_workers=max(int(threads), 1)) as executor:
        futures = [
            executor.submit(
                _prepare_one_image,
                filename,
                split=split,
                image_dir=image_dir,
                image_url_template=image_url_template,
                image_source_dir=image_source_dir,
            )
            for filename in filenames
        ]
        return tuple(future.result() for future in as_completed(futures))


def _prepare_one_image(
    filename: str,
    *,
    split: str,
    image_dir: Path,
    image_url_template: str,
    image_source_dir: Path | None,
) -> Dict[str, Any]:
    target = image_dir / Path(filename).name
    if target.exists() and target.stat().st_size > 0:
        return {"filename": filename, "status": "cached", "path": str(target)}
    try:
        if image_source_dir is not None:
            source = image_source_dir / Path(filename).name
            shutil.copy2(source, target)
            return {"filename": filename, "status": "copied", "path": str(target)}
        url = image_url_template.format(split=split, filename=Path(filename).name)
        _download_file(url, target)
        return {"filename": filename, "status": "downloaded", "path": str(target)}
    except Exception as exc:
        return {"filename": filename, "status": "failed", "error": str(exc), "path": str(target)}


def _write_data_yaml(destination: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(COCO80_CLASS_NAMES))
    (destination / "data.yaml").write_text(
        "path: .\n"
        "train: images/val2017\n"
        "val: images/val2017\n"
        "names:\n"
        f"{names}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
