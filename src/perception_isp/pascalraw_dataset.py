"""PASCALRAW annotation, availability, and extraction utilities."""

from __future__ import annotations

import argparse
import collections
import html as html_lib
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .eval_types import BoundingBox
from .types import json_ready


DEFAULT_CONDITION = "daylight_raw_derived_downsampled"
DEFAULT_IMAGE_ARCHIVE = "JPEGImages.zip"
DEFAULT_IMAGE_DIR = "images"
DEFAULT_IMAGE_EXTENSION = ".png"
SUMMARY_FILENAME = "pascalraw_subset_plan_summary.json"
MANIFEST_FILENAME = "pascalraw_subset_manifest.json"
AVAILABILITY_SUMMARY_FILENAME = "pascalraw_image_availability_summary.json"
EXTRACT_SUMMARY_FILENAME = "pascalraw_extract_summary.json"
REQUIRED_FILES_FILENAME = "pascalraw_required_files.txt"
MISSING_FILES_FILENAME = "pascalraw_missing_files.txt"


@dataclass(frozen=True)
class PascalRawAnnotationRecord:
    sample_id: str
    annotation_file_name: str
    image_file_name: str
    width: int
    height: int
    boxes: tuple[BoundingBox, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({box.label for box in self.boxes}))

    def to_manifest_row(
        self,
        *,
        image_dir: str = DEFAULT_IMAGE_DIR,
        selection_condition: str = DEFAULT_CONDITION,
    ) -> Dict[str, Any]:
        image_name = str(Path(self.image_file_name or f"{self.sample_id}{DEFAULT_IMAGE_EXTENSION}").name)
        return {
            "sample_id": self.sample_id,
            "file_name": image_name,
            "annotation_file_name": self.annotation_file_name,
            "selection_condition": str(selection_condition),
            "tags": [str(selection_condition), "pascalraw", "raw_derived_png"],
            "width": int(self.width),
            "height": int(self.height),
            "box_count": len(self.boxes),
            "labels": list(self.labels),
            "image_file_name": image_name,
            "zip_member": image_name,
            "expected_image_relative_path": f"{image_dir}/{image_name}",
            "expected_zip_member": image_name,
            "boxes": [box.to_dict() for box in self.boxes],
        }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Plan and audit PASCALRAW downsampled RAW-derived images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build a VOC XML subset manifest.")
    plan.add_argument("annotations", help="PASCALRAW VOC XML annotation directory, zip, or single XML file.")
    plan.add_argument("--max-total", type=int, default=128)
    plan.add_argument("--condition", default=DEFAULT_CONDITION)
    plan.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    plan.add_argument("--allow-empty-boxes", action="store_true")
    plan.add_argument("--output-dir", default="reports/perception_pascalraw_subset_plan")

    availability = subparsers.add_parser("availability", help="Audit required PASCALRAW image files.")
    availability.add_argument("manifest", help="PASCALRAW subset manifest JSON or summary JSON containing a manifest key.")
    availability.add_argument("--dataset-root", default="data/raw_datasets/pascalraw")
    availability.add_argument("--image-archive", default=DEFAULT_IMAGE_ARCHIVE)
    availability.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    availability.add_argument("--output-dir", default="reports/perception_pascalraw_image_availability")

    extract = subparsers.add_parser("extract", help="Extract selected PASCALRAW PNG images from JPEGImages.zip.")
    extract.add_argument("manifest", help="PASCALRAW subset manifest JSON or summary JSON containing a manifest key.")
    extract.add_argument("--dataset-root", default="data/raw_datasets/pascalraw")
    extract.add_argument("--image-archive", default=DEFAULT_IMAGE_ARCHIVE)
    extract.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    extract.add_argument("--output-dir", default="reports/perception_pascalraw_extract")

    args = parser.parse_args(argv)
    if args.command == "plan":
        summary = build_pascalraw_subset_plan(
            args.annotations,
            max_total=int(args.max_total),
            selection_condition=str(args.condition),
            image_dir=str(args.image_dir),
            require_boxes=not bool(args.allow_empty_boxes),
        )
        html_path = write_pascalraw_subset_plan(summary, args.output_dir)
        printed = {
            "report": str(html_path),
            "summary_json": str(html_path.parent / SUMMARY_FILENAME),
            "manifest_json": str(html_path.parent / MANIFEST_FILENAME),
            "status": summary["status"],
            "selected_count": summary["selected_count"],
        }
    elif args.command == "availability":
        summary = build_pascalraw_image_availability(
            args.manifest,
            dataset_root=args.dataset_root,
            image_archive=str(args.image_archive),
            image_dir=str(args.image_dir),
        )
        html_path = write_pascalraw_image_availability(summary, args.output_dir)
        printed = {
            "report": str(html_path),
            "summary_json": str(html_path.parent / AVAILABILITY_SUMMARY_FILENAME),
            "required_files_txt": str(html_path.parent / REQUIRED_FILES_FILENAME),
            "missing_files_txt": str(html_path.parent / MISSING_FILES_FILENAME),
            "status": summary["status"],
            "evaluation_ready": summary["evaluation_ready"],
            "ready_to_extract": summary["ready_to_extract"],
            "missing_file_count": summary["missing_file_count"],
        }
    else:
        summary = extract_pascalraw_images(
            args.manifest,
            dataset_root=args.dataset_root,
            image_archive=str(args.image_archive),
            image_dir=str(args.image_dir),
        )
        html_path = write_pascalraw_extract_summary(summary, args.output_dir)
        printed = {
            "report": str(html_path),
            "summary_json": str(html_path.parent / EXTRACT_SUMMARY_FILENAME),
            "status": summary["status"],
            "extracted_count": summary["extracted_count"],
            "missing_member_count": summary["missing_member_count"],
        }
    print(json.dumps(json_ready(printed), indent=2))
    return 0


def load_pascalraw_annotation_records(
    annotations: str | Path,
    *,
    include_difficult: bool = False,
) -> tuple[PascalRawAnnotationRecord, ...]:
    source = Path(annotations).expanduser()
    records = []
    for name, payload in _iter_xml_payloads(source):
        record = _record_from_xml(name, payload, include_difficult=include_difficult)
        if record is not None:
            records.append(record)
    return tuple(sorted(records, key=lambda row: row.sample_id))


def build_pascalraw_subset_plan(
    annotations: str | Path,
    *,
    max_total: int = 128,
    selection_condition: str = DEFAULT_CONDITION,
    image_dir: str = DEFAULT_IMAGE_DIR,
    require_boxes: bool = True,
) -> Dict[str, Any]:
    records = load_pascalraw_annotation_records(annotations)
    selected = []
    for record in records:
        if require_boxes and not record.boxes:
            continue
        selected.append(record)
        if max_total > 0 and len(selected) >= max_total:
            break
    manifest = [
        record.to_manifest_row(image_dir=image_dir, selection_condition=selection_condition)
        for record in selected
    ]
    label_counts = collections.Counter(label for row in manifest for label in row.get("labels", ()))
    checks = _subset_checks(records, manifest, max_total=max_total)
    return {
        "name": "PASCALRAW subset plan",
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "source": str(Path(annotations).expanduser()),
        "selection_condition": str(selection_condition),
        "max_total": int(max_total),
        "record_count": len(records),
        "selected_count": len(manifest),
        "label_counts": dict(label_counts.most_common()),
        "manifest": manifest,
        "checks": checks,
        "image_requirements": {
            "image_archive": DEFAULT_IMAGE_ARCHIVE,
            "image_directory": str(image_dir),
            "image_extension": DEFAULT_IMAGE_EXTENSION,
        },
        "next_action": "Run availability, extract the selected PNGs, then use this as a RAW-derived sanity benchmark.",
        "claim_boundary": (
            "PASCALRAW here is the 10x downsampled PNG release derived from RAW, not full native Bayer RAW. "
            "It is useful for fast annotated detector sanity checks, but weak evidence for CFA/PSF/native RAW claims."
        ),
    }


def build_pascalraw_image_availability(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path,
    image_archive: str = DEFAULT_IMAGE_ARCHIVE,
    image_dir: str = DEFAULT_IMAGE_DIR,
) -> Dict[str, Any]:
    rows = _load_manifest(manifest)
    root = Path(dataset_root).expanduser()
    archive_path = root / image_archive
    archive_members = _archive_members(archive_path)
    file_checks = _file_checks(rows, root=root, archive_members=archive_members, image_dir=image_dir)
    required_files = sorted({str(row["relative_path"]) for row in file_checks if row["kind"] == "image"})
    missing_files = [row for row in file_checks if row["kind"] == "image" and not bool(row["exists"])]
    missing_members = [row for row in file_checks if row["kind"] == "zip_member" and not bool(row["exists"])]
    checks = _availability_checks(rows, archive_path=archive_path, file_checks=file_checks)
    evaluation_ready = bool(rows) and not missing_files and not missing_members
    ready_to_extract = bool(rows) and archive_path.is_file() and not missing_members
    status = "pass" if evaluation_ready else "extractable" if ready_to_extract else "blocked"
    return {
        "name": "PASCALRAW image availability audit",
        "status": status,
        "evaluation_ready": evaluation_ready,
        "ready_to_extract": ready_to_extract,
        "dataset_root": str(root),
        "image_archive": str(archive_path),
        "manifest_source": str(manifest) if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, Path)) else "in_memory",
        "manifest_row_count": len(rows),
        "required_file_count": len(required_files),
        "available_file_count": len(file_checks) - len(missing_files) - len(missing_members),
        "missing_file_count": len(missing_files),
        "missing_zip_member_count": len(missing_members),
        "required_files": required_files,
        "missing_files": [dict(row) for row in missing_files],
        "missing_zip_members": [dict(row) for row in missing_members],
        "file_checks": file_checks,
        "checks": checks,
        "next_action": _availability_next_action(evaluation_ready=evaluation_ready, ready_to_extract=ready_to_extract),
        "claim_boundary": (
            "This is only a local file-availability gate for downsampled RAW-derived PNGs. "
            "It does not prove native RAW decoding, ISP behavior, or detector performance."
        ),
    }


def extract_pascalraw_images(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path,
    image_archive: str = DEFAULT_IMAGE_ARCHIVE,
    image_dir: str = DEFAULT_IMAGE_DIR,
) -> Dict[str, Any]:
    rows = _load_manifest(manifest)
    root = Path(dataset_root).expanduser()
    archive_path = root / image_archive
    destination = root / image_dir
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    missing = []
    with zipfile.ZipFile(archive_path) as archive:
        members = set(name.lstrip("/") for name in archive.namelist())
        for row in rows:
            member = str(row.get("expected_zip_member") or row.get("zip_member") or row.get("image_file_name") or "").lstrip("/")
            image_name = str(row.get("image_file_name") or Path(member).name)
            if member not in members:
                missing.append({"sample_id": str(row.get("sample_id", "")), "zip_member": member})
                continue
            output_path = destination / image_name
            with archive.open(member) as handle:
                output_path.write_bytes(handle.read())
            extracted.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "zip_member": member,
                    "relative_path": f"{image_dir}/{image_name}",
                    "size_bytes": int(output_path.stat().st_size),
                }
            )
    return {
        "name": "PASCALRAW extract",
        "status": "pass" if not missing and len(extracted) == len(rows) else "blocked",
        "dataset_root": str(root),
        "image_archive": str(archive_path),
        "image_dir": str(destination),
        "manifest_row_count": len(rows),
        "extracted_count": len(extracted),
        "missing_member_count": len(missing),
        "extracted": extracted,
        "missing_members": missing,
        "next_action": "Run the PASCALRAW image availability audit again, then detector evaluation.",
        "claim_boundary": (
            "Extraction only materializes selected downsampled RAW-derived PNGs. It does not prove PerceptionISP performance."
        ),
    }


def write_pascalraw_subset_plan(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / MANIFEST_FILENAME).write_text(json.dumps(json_ready(summary.get("manifest", ())), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_subset_html(summary))
    return html_path


def write_pascalraw_image_availability(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    required_files = [str(value) for value in summary.get("required_files", ())]
    missing_files = [str(row.get("relative_path", "")) for row in summary.get("missing_files", ()) if isinstance(row, Mapping)]
    (destination / AVAILABILITY_SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / REQUIRED_FILES_FILENAME).write_text("\n".join(required_files) + ("\n" if required_files else ""))
    (destination / MISSING_FILES_FILENAME).write_text("\n".join(missing_files) + ("\n" if missing_files else ""))
    html_path = destination / "index.html"
    html_path.write_text(_render_availability_html(summary))
    return html_path


def write_pascalraw_extract_summary(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / EXTRACT_SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_extract_html(summary))
    return html_path


def _iter_xml_payloads(source: Path) -> tuple[tuple[str, str], ...]:
    if source.is_dir():
        return tuple((str(path.name), path.read_text(errors="ignore")) for path in sorted(source.rglob("*.xml")))
    if source.is_file() and source.suffix.lower() == ".xml":
        return ((source.name, source.read_text(errors="ignore")),)
    if zipfile.is_zipfile(source):
        payloads = []
        with zipfile.ZipFile(source) as archive:
            for name in sorted(item for item in archive.namelist() if item.lower().endswith(".xml")):
                with archive.open(name) as handle:
                    payloads.append((Path(name).name, handle.read().decode("utf-8", errors="ignore")))
        return tuple(payloads)
    raise ValueError(f"Unsupported PASCALRAW annotation source: {source}")


def _record_from_xml(name: str, payload: str, *, include_difficult: bool) -> PascalRawAnnotationRecord | None:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return None
    filename = _xml_text(root, "filename") or f"{Path(name).stem}{DEFAULT_IMAGE_EXTENSION}"
    image_name = str(Path(filename).with_suffix(DEFAULT_IMAGE_EXTENSION).name)
    width = _xml_int(root, "size/width", 0)
    height = _xml_int(root, "size/height", 0)
    boxes = []
    for obj in root.findall("object"):
        difficult = _xml_int(obj, "difficult", 0)
        if difficult and not include_difficult:
            continue
        label = _normalize_label(_xml_text(obj, "name") or "object")
        box_node = obj.find("bndbox")
        if box_node is None:
            continue
        coords = (
            _xml_float(box_node, "xmin", 0.0),
            _xml_float(box_node, "ymin", 0.0),
            _xml_float(box_node, "xmax", 0.0),
            _xml_float(box_node, "ymax", 0.0),
        )
        box = _box_from_voc(coords, label=label, width=width, height=height)
        if box is not None:
            boxes.append(box)
    if width <= 0 and boxes:
        width = int(max(box.xyxy[2] for box in boxes))
    if height <= 0 and boxes:
        height = int(max(box.xyxy[3] for box in boxes))
    sample_id = Path(image_name).stem or Path(name).stem
    return PascalRawAnnotationRecord(
        sample_id=sample_id,
        annotation_file_name=name,
        image_file_name=image_name,
        width=int(width),
        height=int(height),
        boxes=tuple(boxes),
    )


def _box_from_voc(coords: Sequence[float], *, label: str, width: int, height: int) -> BoundingBox | None:
    x1, y1, x2, y2 = (float(value) for value in coords)
    max_x = float(width - 1) if width > 1 else max(x1, x2)
    max_y = float(height - 1) if height > 1 else max(y1, y2)
    x1 = max(0.0, min(max_x, x1))
    x2 = max(0.0, min(max_x, x2))
    y1 = max(0.0, min(max_y, y1))
    y2 = max(0.0, min(max_y, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox((x1, y1, x2, y2), label=label)


def _load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Path)):
        return [dict(row) for row in manifest if isinstance(row, Mapping)]
    path = Path(manifest).expanduser()
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("manifest", ())
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"PASCALRAW manifest must be a JSON list or contain a manifest list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _archive_members(path: Path) -> set[str]:
    if not path.is_file() or not zipfile.is_zipfile(path):
        return set()
    with zipfile.ZipFile(path) as archive:
        return {name.lstrip("/") for name in archive.namelist()}


def _file_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    archive_members: set[str],
    image_dir: str,
) -> list[Dict[str, Any]]:
    checks = []
    for index, row in enumerate(rows):
        image_name = str(row.get("image_file_name") or row.get("file_name") or "")
        relative_path = str(row.get("expected_image_relative_path") or f"{image_dir}/{image_name}")
        full_path = root / relative_path if relative_path else root
        exists = bool(relative_path) and full_path.is_file()
        checks.append(
            {
                "manifest_index": int(index),
                "sample_id": str(row.get("sample_id", "")),
                "selection_condition": str(row.get("selection_condition", "")),
                "kind": "image",
                "relative_path": relative_path,
                "absolute_path": str(full_path),
                "exists": exists,
                "size_bytes": int(full_path.stat().st_size) if exists else 0,
            }
        )
        member = str(row.get("expected_zip_member") or row.get("zip_member") or image_name).lstrip("/")
        checks.append(
            {
                "manifest_index": int(index),
                "sample_id": str(row.get("sample_id", "")),
                "selection_condition": str(row.get("selection_condition", "")),
                "kind": "zip_member",
                "relative_path": member,
                "absolute_path": "",
                "exists": bool(member) and member in archive_members,
                "size_bytes": 0,
            }
        )
    return checks


def _subset_checks(records: Sequence[PascalRawAnnotationRecord], manifest: Sequence[Mapping[str, Any]], *, max_total: int) -> list[Dict[str, Any]]:
    empty_box_rows = [row for row in manifest if int(row.get("box_count", 0)) <= 0]
    duplicates = [
        sample_id
        for sample_id, count in collections.Counter(str(row.get("sample_id", "")) for row in manifest).items()
        if count > 1
    ]
    return [
        {"id": "source_records_available", "status": "pass" if records else "fail", "evidence": f"records={len(records)}"},
        {"id": "selected_rows_have_boxes", "status": "pass" if not empty_box_rows else "fail", "evidence": f"empty_box_rows={len(empty_box_rows)}"},
        {"id": "selected_rows_unique", "status": "pass" if not duplicates else "fail", "evidence": f"duplicates={duplicates}"},
        {
            "id": "selected_count_within_limit",
            "status": "pass" if max_total <= 0 or len(manifest) <= max_total else "fail",
            "evidence": f"selected={len(manifest)} max_total={max_total}",
        },
    ]


def _availability_checks(rows: Sequence[Mapping[str, Any]], *, archive_path: Path, file_checks: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    missing_images = [row for row in file_checks if row.get("kind") == "image" and not bool(row.get("exists", False))]
    missing_members = [row for row in file_checks if row.get("kind") == "zip_member" and not bool(row.get("exists", False))]
    empty_boxes = [row for row in rows if int(row.get("box_count", 0)) <= 0]
    return [
        {"id": "manifest_loaded", "status": "pass" if rows else "fail", "evidence": f"rows={len(rows)}"},
        {"id": "image_archive_available", "status": "pass" if archive_path.is_file() else "fail", "evidence": str(archive_path)},
        {"id": "zip_members_available", "status": "pass" if not missing_members else "fail", "evidence": f"missing_zip_members={len(missing_members)}"},
        {"id": "extracted_images_available", "status": "pass" if not missing_images else "fail", "evidence": f"missing_images={len(missing_images)}"},
        {"id": "selected_rows_have_boxes", "status": "pass" if not empty_boxes else "fail", "evidence": f"empty_box_rows={len(empty_boxes)}"},
    ]


def _availability_next_action(*, evaluation_ready: bool, ready_to_extract: bool) -> str:
    if evaluation_ready:
        return "Run the detector/evaluation benchmark on the extracted PASCALRAW subset."
    if ready_to_extract:
        return "Run the PASCALRAW extract command to materialize the selected PNGs."
    return "Acquire JPEGImages.zip or fix missing zip members before extraction."


def _xml_text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    return "" if node is None or node.text is None else str(node.text).strip()


def _xml_int(root: ET.Element, path: str, default: int) -> int:
    try:
        return int(float(_xml_text(root, path)))
    except ValueError:
        return int(default)


def _xml_float(root: ET.Element, path: str, default: float) -> float:
    try:
        return float(_xml_text(root, path))
    except ValueError:
        return float(default)


def _normalize_label(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_") or "object"


def _render_subset_html(summary: Mapping[str, Any]) -> str:
    return _render_common_html(
        "PASCALRAW Subset Plan",
        summary,
        sections=[
            ("Image Requirements", f"<pre>{html_lib.escape(json.dumps(json_ready(summary.get('image_requirements', {})), indent=2))}</pre>"),
            ("Checks", _checks_table(summary.get("checks", ()))),
            ("Manifest", _manifest_table(summary.get("manifest", ()))),
        ],
        footer=f"Raw JSON: <code>{SUMMARY_FILENAME}</code>, manifest: <code>{MANIFEST_FILENAME}</code>",
    )


def _render_availability_html(summary: Mapping[str, Any]) -> str:
    missing_rows = "".join(_missing_row(row) for row in list(summary.get("missing_files", ()))[:80] if isinstance(row, Mapping))
    missing_table = f"<table><thead><tr><th>Sample</th><th>Expected Relative Path</th></tr></thead><tbody>{missing_rows}</tbody></table>"
    return _render_common_html(
        "PASCALRAW Image Availability Audit",
        summary,
        sections=[
            ("Checks", _checks_table(summary.get("checks", ()))),
            ("Missing Extracted Images", missing_table),
        ],
        footer=f"Raw JSON: <code>{AVAILABILITY_SUMMARY_FILENAME}</code>, required list: <code>{REQUIRED_FILES_FILENAME}</code>, missing list: <code>{MISSING_FILES_FILENAME}</code>",
    )


def _render_extract_html(summary: Mapping[str, Any]) -> str:
    extracted_rows = "".join(_extract_row(row) for row in list(summary.get("extracted", ()))[:80] if isinstance(row, Mapping))
    extracted_table = f"<table><thead><tr><th>Sample</th><th>Output</th><th>Bytes</th></tr></thead><tbody>{extracted_rows}</tbody></table>"
    return _render_common_html(
        "PASCALRAW Extract",
        summary,
        sections=[("Extracted Images", extracted_table)],
        footer=f"Raw JSON: <code>{EXTRACT_SUMMARY_FILENAME}</code>",
    )


def _render_common_html(title: str, summary: Mapping[str, Any], *, sections: Sequence[tuple[str, str]], footer: str) -> str:
    body_sections = "".join(f"<h2>{html_lib.escape(name)}</h2>{html}" for name, html in sections)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_lib.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .pass {{ color: #047857; font-weight: 650; }}
    .fail, .blocked, .warning, .extractable {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html_lib.escape(title)}</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>. {html_lib.escape(str(summary.get('next_action', '')))}</p>
  {body_sections}
  <p>{footer}</p>
</body>
</html>
"""


def _checks_table(rows: Any) -> str:
    check_rows = "".join(_check_row(row) for row in rows if isinstance(row, Mapping))
    return f"<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>"


def _check_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _manifest_table(rows: Any) -> str:
    manifest_rows = "".join(_manifest_row(row) for row in rows if isinstance(row, Mapping))
    return (
        "<table><thead><tr><th>Sample</th><th>Image Path</th><th>Zip Member</th><th>Boxes</th><th>Labels</th></tr></thead>"
        f"<tbody>{manifest_rows}</tbody></table>"
    )


def _manifest_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('expected_image_relative_path', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('expected_zip_member', '')))}</code></td>"
        f"<td>{int(row.get('box_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('labels', ())))}</td>"
        "</tr>"
    )


def _missing_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('relative_path', '')))}</code></td>"
        "</tr>"
    )


def _extract_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('relative_path', '')))}</code></td>"
        f"<td>{int(row.get('size_bytes', 0))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
