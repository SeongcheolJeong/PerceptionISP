"""AODRaw annotation parsing and subset planning utilities."""

from __future__ import annotations

import argparse
import collections
import html as html_lib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .eval_types import BoundingBox
from .types import json_ready


DEFAULT_SPLIT = "test_annotations_downsample_scale3_bbox_min_size32.json"
DEFAULT_CONDITIONS = ("low_light", "rain", "fog")
SUMMARY_FILENAME = "aodraw_subset_plan_summary.json"
MANIFEST_FILENAME = "aodraw_subset_manifest.json"
HARD_CASE_CHOICES = ("none", "small", "thin", "small_or_thin", "small_and_thin")


@dataclass(frozen=True)
class AODRawAnnotationRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    tags: tuple[str, ...]
    boxes: tuple[BoundingBox, ...]
    ignored_annotation_count: int
    category_ids: tuple[int, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({box.label for box in self.boxes}))

    def hard_stats(self, *, small_area_frac: float, thin_aspect: float) -> Dict[str, Any]:
        image_area = max(float(self.width * self.height), 1.0)
        small_count = 0
        thin_count = 0
        small_or_thin_count = 0
        small_and_thin_count = 0
        min_area_frac = None
        max_aspect = 0.0
        for box in self.boxes:
            x1, y1, x2, y2 = box.xyxy
            width = max(float(x2 - x1), 0.0)
            height = max(float(y2 - y1), 0.0)
            area_frac = float(box.area) / image_area
            aspect = max(width / max(height, 1e-9), height / max(width, 1e-9)) if width > 0.0 and height > 0.0 else 0.0
            is_small = area_frac <= float(small_area_frac)
            is_thin = aspect >= float(thin_aspect)
            small_count += int(is_small)
            thin_count += int(is_thin)
            small_or_thin_count += int(is_small or is_thin)
            small_and_thin_count += int(is_small and is_thin)
            min_area_frac = area_frac if min_area_frac is None else min(float(min_area_frac), area_frac)
            max_aspect = max(max_aspect, aspect)
        return {
            "small_box_count": small_count,
            "thin_box_count": thin_count,
            "small_or_thin_box_count": small_or_thin_count,
            "small_and_thin_box_count": small_and_thin_count,
            "min_box_area_frac": 0.0 if min_area_frac is None else float(min_area_frac),
            "max_box_aspect": float(max_aspect),
        }

    def to_manifest_row(
        self,
        *,
        split: str,
        selection_condition: str | None = None,
        hard_case: str = "none",
        small_area_frac: float = 0.02,
        thin_aspect: float = 3.0,
    ) -> Dict[str, Any]:
        raw_dir, srgb_dir = _image_dirs_for_split(split)
        raw_file = _raw_file_name(self.file_name, split)
        hard_stats = self.hard_stats(small_area_frac=small_area_frac, thin_aspect=thin_aspect)
        raw_layout = _raw_storage_layout_for_split(split)
        native_cfa = _raw_layout_is_native_cfa(raw_layout)
        return {
            "image_id": int(self.image_id),
            "file_name": self.file_name,
            "selection_condition": "" if selection_condition is None else str(selection_condition),
            "hard_case": str(hard_case),
            "small_area_frac": float(small_area_frac),
            "thin_aspect": float(thin_aspect),
            **hard_stats,
            "tags": list(self.tags),
            "width": int(self.width),
            "height": int(self.height),
            "box_count": len(self.boxes),
            "ignored_annotation_count": int(self.ignored_annotation_count),
            "labels": list(self.labels),
            "srgb_file_name": self.file_name,
            "raw_file_name": raw_file,
            "expected_srgb_relative_path": f"{srgb_dir}/{self.file_name}",
            "expected_raw_relative_path": f"{raw_dir}/{raw_file}",
            "raw_storage_layout": raw_layout,
            "true_sensor_cfa_mosaic_expected": bool(native_cfa),
            "boxes": [box.to_dict() for box in self.boxes],
        }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a small AODRaw held-out subset from annotations.")
    parser.add_argument("annotations", help="AODRaw_annotations.zip, extracted annotations directory, or a split JSON.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--condition", action="append", default=[], help="Condition/tag to sample. Repeatable.")
    parser.add_argument("--per-condition", type=int, default=8)
    parser.add_argument("--max-total", type=int, default=24)
    parser.add_argument("--allow-empty-boxes", action="store_true")
    parser.add_argument("--hard-case", choices=HARD_CASE_CHOICES, default="none")
    parser.add_argument("--small-area-frac", type=float, default=0.02)
    parser.add_argument("--thin-aspect", type=float, default=3.0)
    parser.add_argument("--output-dir", default="reports/perception_aodraw_subset_plan")
    args = parser.parse_args(argv)

    summary = build_aodraw_subset_plan(
        args.annotations,
        split=str(args.split),
        conditions=tuple(args.condition) if args.condition else DEFAULT_CONDITIONS,
        per_condition=int(args.per_condition),
        max_total=int(args.max_total),
        require_boxes=not bool(args.allow_empty_boxes),
        hard_case=str(args.hard_case),
        small_area_frac=float(args.small_area_frac),
        thin_aspect=float(args.thin_aspect),
    )
    html_path = write_aodraw_subset_plan(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "manifest_json": str(html_path.parent / MANIFEST_FILENAME),
                    "status": summary["status"],
                    "selected_count": summary["selected_count"],
                    "split": summary["split"],
                    "conditions": summary["conditions"],
                }
            ),
            indent=2,
        )
    )
    return 0


def load_aodraw_annotation_records(annotations: str | Path, *, split: str = DEFAULT_SPLIT) -> tuple[AODRawAnnotationRecord, ...]:
    payload = _load_split_payload(Path(annotations).expanduser(), split)
    categories = {
        int(row.get("id", -1)): str(row.get("name", ""))
        for row in payload.get("categories", ())
        if isinstance(row, Mapping) and row.get("id") is not None
    }
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for ann in payload.get("annotations", ()):
        if not isinstance(ann, Mapping):
            continue
        annotations_by_image[int(ann.get("image_id", -1))].append(ann)

    records = []
    for image in payload.get("images", ()):
        if not isinstance(image, Mapping):
            continue
        image_id = int(image.get("id", -1))
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        boxes = []
        ignored = 0
        category_ids = []
        for ann in annotations_by_image.get(image_id, ()):
            box = _annotation_to_box(ann, categories, width=width, height=height)
            if box is None:
                ignored += 1
                continue
            boxes.append(box)
            category_ids.append(int(ann.get("category_id", -1)))
        records.append(
            AODRawAnnotationRecord(
                image_id=image_id,
                file_name=str(image.get("file_name", "")),
                width=width,
                height=height,
                tags=tuple(str(tag) for tag in (image.get("tag", ()) or ())),
                boxes=tuple(boxes),
                ignored_annotation_count=ignored,
                category_ids=tuple(category_ids),
            )
        )
    return tuple(sorted(records, key=lambda row: (row.file_name, row.image_id)))


def build_aodraw_subset_plan(
    annotations: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    per_condition: int = 8,
    max_total: int = 24,
    require_boxes: bool = True,
    hard_case: str = "none",
    small_area_frac: float = 0.02,
    thin_aspect: float = 3.0,
) -> Dict[str, Any]:
    normalized_hard_case = _normalize_hard_case(hard_case)
    records = load_aodraw_annotation_records(annotations, split=split)
    selected = _select_records(
        records,
        conditions=tuple(str(condition) for condition in conditions),
        per_condition=int(per_condition),
        max_total=int(max_total),
        require_boxes=bool(require_boxes),
        hard_case=normalized_hard_case,
        small_area_frac=float(small_area_frac),
        thin_aspect=float(thin_aspect),
    )
    manifest = [
        record.to_manifest_row(
            split=split,
            selection_condition=condition,
            hard_case=normalized_hard_case,
            small_area_frac=float(small_area_frac),
            thin_aspect=float(thin_aspect),
        )
        for condition, record in selected
    ]
    condition_counts = collections.Counter(str(row["selection_condition"]) for row in manifest)
    label_counts = collections.Counter(label for row in manifest for label in row["labels"])
    hard_counts = _hard_counts(manifest)
    checks = _checks(
        records,
        manifest,
        conditions=conditions,
        per_condition=per_condition,
        max_total=max_total,
        hard_case=normalized_hard_case,
    )
    return {
        "name": "AODRaw subset plan",
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "source": str(Path(annotations).expanduser()),
        "split": str(split),
        "conditions": [str(condition) for condition in conditions],
        "per_condition": int(per_condition),
        "max_total": int(max_total),
        "require_boxes": bool(require_boxes),
        "hard_case": normalized_hard_case,
        "small_area_frac": float(small_area_frac),
        "thin_aspect": float(thin_aspect),
        "record_count": len(records),
        "selected_count": len(manifest),
        "condition_counts": dict(condition_counts),
        "label_counts": dict(label_counts.most_common()),
        "hard_counts": hard_counts,
        "manifest": manifest,
        "checks": checks,
        "image_requirements": {
            "raw_directory": _image_dirs_for_split(split)[0],
            "srgb_directory": _image_dirs_for_split(split)[1],
            "raw_extension": Path(manifest[0]["raw_file_name"]).suffix if manifest else "",
            "srgb_extension": Path(manifest[0]["srgb_file_name"]).suffix if manifest else "",
            "raw_storage_layout": _raw_storage_layout_for_split(split),
            "true_sensor_cfa_mosaic_expected": _raw_layout_is_native_cfa(_raw_storage_layout_for_split(split)),
            "claim_note": _raw_layout_claim_note(_raw_storage_layout_for_split(split)),
        },
        "next_action": (
            "Acquire only the listed expected_raw_relative_path / expected_srgb_relative_path files first, then run an image-availability audit before detector evaluation."
        ),
        "claim_boundary": (
            "This is an annotation/subset manifest only. It does not prove image availability, RAW decoding, ISP behavior, or detector performance."
        ),
    }


def write_aodraw_subset_plan(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / MANIFEST_FILENAME).write_text(json.dumps(json_ready(summary.get("manifest", ())), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _load_split_payload(path: Path, split: str) -> Mapping[str, Any]:
    if path.is_file() and path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.is_dir():
        candidate = path / split
        if not candidate.exists():
            matches = list(path.rglob(split))
            if not matches:
                raise FileNotFoundError(f"AODRaw split not found under {path}: {split}")
            candidate = matches[0]
        return json.loads(candidate.read_text())
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            match = next((name for name in archive.namelist() if Path(name).name == split), None)
            if match is None:
                raise FileNotFoundError(f"AODRaw split not found in {path}: {split}")
            with archive.open(match) as handle:
                return json.load(handle)
    raise ValueError(f"Unsupported AODRaw annotation source: {path}")


def _annotation_to_box(
    annotation: Mapping[str, Any],
    categories: Mapping[int, str],
    *,
    width: int,
    height: int,
) -> BoundingBox | None:
    if int(annotation.get("iscrowd", 0)) == 1 or int(annotation.get("ignore", 0)) == 1:
        return None
    category_id = int(annotation.get("category_id", -1))
    label = str(categories.get(category_id, "object"))
    if label == "ignore":
        return None
    bbox = annotation.get("bbox", ())
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None
    x, y, w, h = (float(value) for value in bbox)
    if w <= 0.0 or h <= 0.0:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox((x1, y1, x2, y2), label=label)


def _select_records(
    records: Sequence[AODRawAnnotationRecord],
    *,
    conditions: Sequence[str],
    per_condition: int,
    max_total: int,
    require_boxes: bool,
    hard_case: str,
    small_area_frac: float,
    thin_aspect: float,
) -> list[tuple[str, AODRawAnnotationRecord]]:
    selected: list[tuple[str, AODRawAnnotationRecord]] = []
    selected_ids: set[int] = set()
    for condition in conditions:
        count = 0
        for record in records:
            if max_total > 0 and len(selected) >= max_total:
                return selected
            if record.image_id in selected_ids:
                continue
            if condition not in record.tags:
                continue
            if require_boxes and not record.boxes:
                continue
            if not _record_matches_hard_case(
                record,
                hard_case=hard_case,
                small_area_frac=small_area_frac,
                thin_aspect=thin_aspect,
            ):
                continue
            selected.append((str(condition), record))
            selected_ids.add(record.image_id)
            count += 1
            if count >= per_condition:
                break
    return selected


def _checks(
    records: Sequence[AODRawAnnotationRecord],
    manifest: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    per_condition: int,
    max_total: int,
    hard_case: str,
) -> list[Dict[str, Any]]:
    condition_counts = collections.Counter(str(row.get("selection_condition", "")) for row in manifest)
    missing_conditions = [str(condition) for condition in conditions if condition_counts.get(str(condition), 0) <= 0]
    empty_box_rows = [row for row in manifest if int(row.get("box_count", 0)) <= 0]
    duplicate_files = [
        file_name
        for file_name, count in collections.Counter(str(row.get("file_name", "")) for row in manifest).items()
        if count > 1
    ]
    hard_mismatch_rows = [row for row in manifest if not _manifest_row_matches_hard_case(row, hard_case=hard_case)]
    return [
        {
            "id": "source_records_available",
            "status": "pass" if len(records) > 0 else "fail",
            "evidence": f"records={len(records)}",
        },
        {
            "id": "conditions_represented",
            "status": "pass" if not missing_conditions else "fail",
            "evidence": f"condition_counts={dict(condition_counts)} missing={missing_conditions}",
        },
        {
            "id": "selected_rows_have_boxes",
            "status": "pass" if not empty_box_rows else "fail",
            "evidence": f"empty_box_rows={len(empty_box_rows)}",
        },
        {
            "id": "selected_rows_unique",
            "status": "pass" if not duplicate_files else "fail",
            "evidence": f"duplicates={duplicate_files}",
        },
        {
            "id": "selected_rows_match_hard_case",
            "status": "pass" if not hard_mismatch_rows else "fail",
            "evidence": f"hard_case={hard_case} mismatch_rows={len(hard_mismatch_rows)}",
        },
        {
            "id": "selected_count_within_limit",
            "status": "pass" if max_total <= 0 or len(manifest) <= max_total else "fail",
            "evidence": f"selected={len(manifest)} max_total={max_total} per_condition={per_condition}",
        },
    ]


def _normalize_hard_case(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in HARD_CASE_CHOICES:
        raise ValueError(f"Unsupported hard_case={value}; expected one of {HARD_CASE_CHOICES}")
    return normalized


def _record_matches_hard_case(
    record: AODRawAnnotationRecord,
    *,
    hard_case: str,
    small_area_frac: float,
    thin_aspect: float,
) -> bool:
    stats = record.hard_stats(small_area_frac=small_area_frac, thin_aspect=thin_aspect)
    return _stats_match_hard_case(stats, hard_case=hard_case)


def _manifest_row_matches_hard_case(row: Mapping[str, Any], *, hard_case: str) -> bool:
    return _stats_match_hard_case(row, hard_case=hard_case)


def _stats_match_hard_case(stats: Mapping[str, Any], *, hard_case: str) -> bool:
    if hard_case == "none":
        return True
    if hard_case == "small":
        return int(stats.get("small_box_count", 0)) > 0
    if hard_case == "thin":
        return int(stats.get("thin_box_count", 0)) > 0
    if hard_case == "small_or_thin":
        return int(stats.get("small_or_thin_box_count", 0)) > 0
    if hard_case == "small_and_thin":
        return int(stats.get("small_and_thin_box_count", 0)) > 0
    return False


def _hard_counts(manifest: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "small": sum(1 for row in manifest if int(row.get("small_box_count", 0)) > 0),
        "thin": sum(1 for row in manifest if int(row.get("thin_box_count", 0)) > 0),
        "small_or_thin": sum(1 for row in manifest if int(row.get("small_or_thin_box_count", 0)) > 0),
        "small_and_thin": sum(1 for row in manifest if int(row.get("small_and_thin_box_count", 0)) > 0),
    }


def _image_dirs_for_split(split: str) -> tuple[str, str]:
    lowered = str(split).lower()
    if "slice" in lowered:
        return "images_slice_raw", "images_slice_srgb"
    if "downsample" in lowered:
        return "images_downsampled_raw", "images_downsampled_srgb"
    return "images", "images"


def _raw_file_name(file_name: str, split: str) -> str:
    lowered = str(split).lower()
    path = Path(str(file_name))
    if "downsample" in lowered or "slice" in lowered:
        return str(path.with_suffix(".npy"))
    return str(path.with_suffix(".ARW"))


def _raw_storage_layout_for_split(split: str) -> str:
    lowered = str(split).lower()
    if "slice" in lowered:
        return "packed_bayer_4ch_npy_native_cfa"
    if "downsample" in lowered:
        return "demosaiced_rgb_3ch_npy_not_native_cfa"
    return "original_arw_native_cfa"


def _raw_layout_is_native_cfa(layout: str) -> bool:
    return "native_cfa" in str(layout) and "not_native_cfa" not in str(layout)


def _raw_layout_claim_note(layout: str) -> str:
    if str(layout) == "demosaiced_rgb_3ch_npy_not_native_cfa":
        return (
            "Official AODRaw downsampled RAW .npy files are demosaiced 3-channel RAW-RGB, "
            "not a native Bayer mosaic. They are useful for AODRaw detector smoke tests, "
            "but not sufficient for a strict CFA/demosaic PerceptionISP claim."
        )
    if str(layout) == "packed_bayer_4ch_npy_native_cfa":
        return "Official AODRaw sliced RAW .npy files are packed Bayer planes and can support native CFA/demosaic claims after unpacking."
    return "Original AODRaw .ARW files can support native CFA/demosaic claims if acquired and decoded with verified metadata."


def _render_html(summary: Mapping[str, Any]) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    manifest_rows = "".join(_manifest_row(row) for row in summary.get("manifest", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Subset Plan</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pass {{ color: #047857; font-weight: 650; }}
    .fail, .warning {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>AODRaw Subset Plan</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; split: <code>{html_lib.escape(str(summary.get('split', '')))}</code>; selected={int(summary.get('selected_count', 0))}/{int(summary.get('record_count', 0))}; hard_case=<code>{html_lib.escape(str(summary.get('hard_case', 'none')))}</code>.</p>
  <p><strong>RAW layout:</strong> <code>{html_lib.escape(str(summary.get('image_requirements', {}).get('raw_storage_layout', '')))}</code>. {html_lib.escape(str(summary.get('image_requirements', {}).get('claim_note', '')))}</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <h2>Image Requirements</h2>
  <pre>{html_lib.escape(json.dumps(json_ready(summary.get('image_requirements', {})), indent=2))}</pre>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Manifest</h2>
  <table><thead><tr><th>Condition</th><th>Hard</th><th>File</th><th>RAW Path</th><th>RAW Layout</th><th>sRGB Path</th><th>Tags</th><th>Boxes</th><th>Labels</th></tr></thead><tbody>{manifest_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code>, manifest: <code>{MANIFEST_FILENAME}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _manifest_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('selection_condition', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('hard_case', 'none')))}; small={int(row.get('small_box_count', 0))}, thin={int(row.get('thin_box_count', 0))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('file_name', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('expected_raw_relative_path', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('raw_storage_layout', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('expected_srgb_relative_path', '')))}</code></td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('tags', ())))}</td>"
        f"<td>{int(row.get('box_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('labels', ())))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
