"""LOD low-light RAW dataset annotation and availability utilities."""

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


DEFAULT_ANNOTATION_DIR = "RAW-dark-Annotations"
DEFAULT_RAW_DIR = "RAW-dark-images"
DEFAULT_SRGB_DIR = "RGB-dark-images"
DEFAULT_RAW_EXTENSION = ".CR2"
DEFAULT_SRGB_EXTENSION = ".JPG"
DEFAULT_CONDITION = "low_light"
SUBSET_SUMMARY_FILENAME = "lod_subset_plan_summary.json"
MANIFEST_FILENAME = "lod_subset_manifest.json"
AVAILABILITY_SUMMARY_FILENAME = "lod_image_availability_summary.json"
LOCAL_READINESS_SUMMARY_FILENAME = "lod_local_readiness_summary.json"
REQUIRED_FILES_FILENAME = "lod_required_files.txt"
MISSING_FILES_FILENAME = "lod_missing_files.txt"


@dataclass(frozen=True)
class LODAnnotationRecord:
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
        raw_dir: str = DEFAULT_RAW_DIR,
        srgb_dir: str = DEFAULT_SRGB_DIR,
        raw_extension: str = DEFAULT_RAW_EXTENSION,
        srgb_extension: str = DEFAULT_SRGB_EXTENSION,
        selection_condition: str = DEFAULT_CONDITION,
    ) -> Dict[str, Any]:
        stem = Path(self.image_file_name or self.sample_id).stem
        srgb_name = _with_suffix(self.image_file_name or f"{stem}{srgb_extension}", srgb_extension)
        raw_name = f"{stem}{_normalize_extension(raw_extension)}"
        return {
            "sample_id": self.sample_id,
            "file_name": self.image_file_name,
            "annotation_file_name": self.annotation_file_name,
            "selection_condition": str(selection_condition),
            "tags": [str(selection_condition)],
            "width": int(self.width),
            "height": int(self.height),
            "box_count": len(self.boxes),
            "labels": list(self.labels),
            "srgb_file_name": srgb_name,
            "raw_file_name": raw_name,
            "expected_annotation_relative_path": f"{DEFAULT_ANNOTATION_DIR}/{self.annotation_file_name}",
            "expected_srgb_relative_path": f"{srgb_dir}/{srgb_name}",
            "expected_raw_relative_path": f"{raw_dir}/{raw_name}",
            "boxes": [box.to_dict() for box in self.boxes],
        }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Plan and audit LOD low-light RAW dataset files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build a LOD VOC XML subset manifest.")
    plan.add_argument("annotations", help="LOD annotation directory or zip containing VOC XML files.")
    plan.add_argument("--max-total", type=int, default=24)
    plan.add_argument("--condition", default=DEFAULT_CONDITION)
    plan.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    plan.add_argument("--srgb-dir", default=DEFAULT_SRGB_DIR)
    plan.add_argument("--raw-extension", default=DEFAULT_RAW_EXTENSION)
    plan.add_argument("--srgb-extension", default=DEFAULT_SRGB_EXTENSION)
    plan.add_argument("--allow-empty-boxes", action="store_true")
    plan.add_argument("--output-dir", default="reports/perception_lod_subset_plan")

    availability = subparsers.add_parser("availability", help="Audit image files required by a LOD subset manifest.")
    availability.add_argument("manifest", help="LOD subset manifest JSON or summary JSON containing a manifest key.")
    availability.add_argument("--dataset-root", default="data/raw_datasets/lod")
    availability.add_argument("--output-dir", default="reports/perception_lod_image_availability")

    readiness = subparsers.add_parser("readiness", help="Audit local LOD directory availability before ingest.")
    readiness.add_argument("--dataset-root", default="data/raw_datasets/lod")
    readiness.add_argument("--annotation-dir", default=DEFAULT_ANNOTATION_DIR)
    readiness.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    readiness.add_argument("--srgb-dir", default=DEFAULT_SRGB_DIR)
    readiness.add_argument("--output-dir", default="reports/perception_lod_local_readiness")

    args = parser.parse_args(argv)
    if args.command == "plan":
        summary = build_lod_subset_plan(
            args.annotations,
            max_total=int(args.max_total),
            selection_condition=str(args.condition),
            raw_dir=str(args.raw_dir),
            srgb_dir=str(args.srgb_dir),
            raw_extension=str(args.raw_extension),
            srgb_extension=str(args.srgb_extension),
            require_boxes=not bool(args.allow_empty_boxes),
        )
        html_path = write_lod_subset_plan(summary, args.output_dir)
        printed = {
            "report": str(html_path),
            "summary_json": str(html_path.parent / SUBSET_SUMMARY_FILENAME),
            "manifest_json": str(html_path.parent / MANIFEST_FILENAME),
            "status": summary["status"],
            "selected_count": summary["selected_count"],
        }
    elif args.command == "availability":
        summary = build_lod_image_availability(args.manifest, dataset_root=args.dataset_root)
        html_path = write_lod_image_availability(summary, args.output_dir)
        printed = {
            "report": str(html_path),
            "summary_json": str(html_path.parent / AVAILABILITY_SUMMARY_FILENAME),
            "required_files_txt": str(html_path.parent / REQUIRED_FILES_FILENAME),
            "missing_files_txt": str(html_path.parent / MISSING_FILES_FILENAME),
            "status": summary["status"],
            "evaluation_ready": summary["evaluation_ready"],
            "required_file_count": summary["required_file_count"],
            "missing_file_count": summary["missing_file_count"],
        }
    else:
        summary = build_lod_local_readiness(
            dataset_root=args.dataset_root,
            annotation_dir=str(args.annotation_dir),
            raw_dir=str(args.raw_dir),
            srgb_dir=str(args.srgb_dir),
        )
        html_path = write_lod_local_readiness(summary, args.output_dir)
        printed = {
            "report": str(html_path),
            "summary_json": str(html_path.parent / LOCAL_READINESS_SUMMARY_FILENAME),
            "status": summary["status"],
            "ready_for_subset_plan": summary["ready_for_subset_plan"],
            "ready_for_image_eval": summary["ready_for_image_eval"],
        }
    print(json.dumps(json_ready(printed), indent=2))
    return 0


def load_lod_annotation_records(
    annotations: str | Path,
    *,
    include_difficult: bool = False,
) -> tuple[LODAnnotationRecord, ...]:
    source = Path(annotations).expanduser()
    records = []
    for name, payload in _iter_xml_payloads(source):
        record = _record_from_xml(name, payload, include_difficult=include_difficult)
        if record is not None:
            records.append(record)
    return tuple(sorted(records, key=lambda row: row.sample_id))


def build_lod_subset_plan(
    annotations: str | Path,
    *,
    max_total: int = 24,
    selection_condition: str = DEFAULT_CONDITION,
    raw_dir: str = DEFAULT_RAW_DIR,
    srgb_dir: str = DEFAULT_SRGB_DIR,
    raw_extension: str = DEFAULT_RAW_EXTENSION,
    srgb_extension: str = DEFAULT_SRGB_EXTENSION,
    require_boxes: bool = True,
) -> Dict[str, Any]:
    records = load_lod_annotation_records(annotations)
    selected = []
    for record in records:
        if require_boxes and not record.boxes:
            continue
        selected.append(record)
        if max_total > 0 and len(selected) >= max_total:
            break
    manifest = [
        record.to_manifest_row(
            raw_dir=raw_dir,
            srgb_dir=srgb_dir,
            raw_extension=raw_extension,
            srgb_extension=srgb_extension,
            selection_condition=selection_condition,
        )
        for record in selected
    ]
    label_counts = collections.Counter(label for row in manifest for label in row.get("labels", ()))
    checks = _subset_checks(records, manifest, max_total=max_total)
    return {
        "name": "LOD subset plan",
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
            "annotation_directory": DEFAULT_ANNOTATION_DIR,
            "raw_directory": str(raw_dir),
            "srgb_directory": str(srgb_dir),
            "raw_extension": _normalize_extension(raw_extension),
            "srgb_extension": _normalize_extension(srgb_extension),
        },
        "next_action": (
            "Acquire the listed RAW-dark and RGB-dark files, then run the LOD image availability audit before detector evaluation."
        ),
        "claim_boundary": (
            "This is a VOC-annotation subset manifest only. It does not prove image availability, RAW decoding, ISP behavior, or detector performance."
        ),
    }


def build_lod_image_availability(manifest: str | Path | Sequence[Mapping[str, Any]], *, dataset_root: str | Path) -> Dict[str, Any]:
    rows = _load_manifest(manifest)
    root = Path(dataset_root).expanduser()
    file_checks = _file_checks(rows, root)
    required_files = sorted({str(row["relative_path"]) for row in file_checks})
    missing_files = [row for row in file_checks if not bool(row["exists"])]
    missing_raw = [row for row in missing_files if row["kind"] == "raw"]
    missing_srgb = [row for row in missing_files if row["kind"] == "srgb"]
    checks = _availability_checks(rows, file_checks, required_files)
    evaluation_ready = bool(rows) and not missing_files and all(row["status"] == "pass" for row in checks)
    return {
        "name": "LOD image availability audit",
        "status": "pass" if evaluation_ready else "blocked",
        "evaluation_ready": evaluation_ready,
        "dataset_root": str(root),
        "manifest_source": str(manifest) if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, Path)) else "in_memory",
        "manifest_row_count": len(rows),
        "required_file_count": len(required_files),
        "available_file_count": len(file_checks) - len(missing_files),
        "missing_file_count": len(missing_files),
        "missing_raw_count": len(missing_raw),
        "missing_srgb_count": len(missing_srgb),
        "required_files": required_files,
        "missing_files": [dict(row) for row in missing_files],
        "file_checks": file_checks,
        "checks": checks,
        "next_action": _lod_availability_next_action(missing_raw=missing_raw, missing_srgb=missing_srgb),
        "claim_boundary": (
            "This is only a local file-availability gate. It does not prove RAW decoding, HumanISP/PerceptionISP parity, or detector performance."
        ),
    }


def build_lod_local_readiness(
    *,
    dataset_root: str | Path,
    annotation_dir: str = DEFAULT_ANNOTATION_DIR,
    raw_dir: str = DEFAULT_RAW_DIR,
    srgb_dir: str = DEFAULT_SRGB_DIR,
) -> Dict[str, Any]:
    root = Path(dataset_root).expanduser()
    directories = []
    for kind, relative in (("annotation", annotation_dir), ("raw", raw_dir), ("srgb", srgb_dir)):
        path = root / relative
        file_count = len([item for item in path.rglob("*") if item.is_file()]) if path.is_dir() else 0
        xml_count = len([item for item in path.rglob("*.xml") if item.is_file()]) if path.is_dir() and kind == "annotation" else 0
        directories.append(
            {
                "kind": kind,
                "relative_path": str(relative),
                "absolute_path": str(path),
                "exists": path.is_dir(),
                "file_count": int(file_count),
                "xml_count": int(xml_count),
            }
        )
    annotation_ready = any(row["kind"] == "annotation" and row["exists"] and row["xml_count"] > 0 for row in directories)
    images_ready = all(any(row["kind"] == kind and row["exists"] and row["file_count"] > 0 for row in directories) for kind in ("raw", "srgb"))
    return {
        "name": "LOD local readiness",
        "status": "pass" if annotation_ready and images_ready else "blocked",
        "dataset_root": str(root),
        "directories": directories,
        "ready_for_subset_plan": bool(annotation_ready),
        "ready_for_image_eval": bool(annotation_ready and images_ready),
        "checks": [
            {
                "id": "voc_annotations_present",
                "status": "pass" if annotation_ready else "fail",
                "evidence": f"annotation_dir={annotation_dir}",
            },
            {
                "id": "raw_images_present",
                "status": "pass" if any(row["kind"] == "raw" and row["exists"] and row["file_count"] > 0 for row in directories) else "fail",
                "evidence": f"raw_dir={raw_dir}",
            },
            {
                "id": "srgb_images_present",
                "status": "pass" if any(row["kind"] == "srgb" and row["exists"] and row["file_count"] > 0 for row in directories) else "fail",
                "evidence": f"srgb_dir={srgb_dir}",
            },
        ],
        "next_action": (
            "Download LOD RAW-dark-Annotations(raw_version), RAW-dark-images, and RGB-dark-images into these directories. "
            "Then run the subset plan and image availability commands."
        ),
        "claim_boundary": (
            "This readiness report checks only local LOD file presence. It does not prove annotation quality, RAW decoding, or detector performance."
        ),
    }


def write_lod_subset_plan(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUBSET_SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / MANIFEST_FILENAME).write_text(json.dumps(json_ready(summary.get("manifest", ())), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_subset_html(summary))
    return html_path


def write_lod_image_availability(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
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


def write_lod_local_readiness(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / LOCAL_READINESS_SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_readiness_html(summary))
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
    raise ValueError(f"Unsupported LOD annotation source: {source}")


def _record_from_xml(name: str, payload: str, *, include_difficult: bool) -> LODAnnotationRecord | None:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return None
    filename = _xml_text(root, "filename") or f"{Path(name).stem}{DEFAULT_SRGB_EXTENSION}"
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
    sample_id = Path(filename).stem or Path(name).stem
    return LODAnnotationRecord(
        sample_id=sample_id,
        annotation_file_name=name,
        image_file_name=filename,
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


def _normalize_extension(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw if raw.startswith(".") else f".{raw}"


def _with_suffix(file_name: str, suffix: str) -> str:
    return str(Path(str(file_name)).with_suffix(_normalize_extension(suffix)))


def _load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Path)):
        return [dict(row) for row in manifest if isinstance(row, Mapping)]
    path = Path(manifest).expanduser()
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("manifest", ())
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"LOD manifest must be a JSON list or contain a manifest list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _file_checks(rows: Sequence[Mapping[str, Any]], root: Path) -> list[Dict[str, Any]]:
    checks = []
    for index, row in enumerate(rows):
        for kind, key in (("raw", "expected_raw_relative_path"), ("srgb", "expected_srgb_relative_path")):
            relative_path = str(row.get(key, "")).strip()
            full_path = root / relative_path if relative_path else root
            exists = bool(relative_path) and full_path.is_file()
            checks.append(
                {
                    "manifest_index": int(index),
                    "sample_id": str(row.get("sample_id", "")),
                    "selection_condition": str(row.get("selection_condition", "")),
                    "kind": kind,
                    "relative_path": relative_path,
                    "absolute_path": str(full_path),
                    "exists": exists,
                    "size_bytes": int(full_path.stat().st_size) if exists else 0,
                }
            )
    return checks


def _subset_checks(records: Sequence[LODAnnotationRecord], manifest: Sequence[Mapping[str, Any]], *, max_total: int) -> list[Dict[str, Any]]:
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


def _availability_checks(
    rows: Sequence[Mapping[str, Any]],
    file_checks: Sequence[Mapping[str, Any]],
    required_files: Sequence[str],
) -> list[Dict[str, Any]]:
    missing_raw = [row for row in file_checks if row.get("kind") == "raw" and not bool(row.get("exists", False))]
    missing_srgb = [row for row in file_checks if row.get("kind") == "srgb" and not bool(row.get("exists", False))]
    empty_boxes = [row for row in rows if int(row.get("box_count", 0)) <= 0]
    duplicate_count = len(file_checks) - len(required_files)
    return [
        {"id": "manifest_loaded", "status": "pass" if rows else "fail", "evidence": f"rows={len(rows)}"},
        {"id": "raw_files_available", "status": "pass" if not missing_raw else "fail", "evidence": f"missing_raw={len(missing_raw)}"},
        {"id": "srgb_files_available", "status": "pass" if not missing_srgb else "fail", "evidence": f"missing_srgb={len(missing_srgb)}"},
        {"id": "selected_rows_have_boxes", "status": "pass" if not empty_boxes else "fail", "evidence": f"empty_box_rows={len(empty_boxes)}"},
        {"id": "duplicate_requirements_collapsed", "status": "pass", "evidence": f"duplicate_file_references={duplicate_count}"},
    ]


def _lod_availability_next_action(*, missing_raw: Sequence[Mapping[str, Any]], missing_srgb: Sequence[Mapping[str, Any]]) -> str:
    if missing_raw or missing_srgb:
        return (
            "Acquire only the missing LOD RAW-dark/RGB-dark files listed in lod_missing_files.txt first, then rerun this audit."
        )
    return "Run the LOD HumanISP/PerceptionISP detector smoke test on this subset."


def _render_subset_html(summary: Mapping[str, Any]) -> str:
    return _render_common_html(
        "LOD Subset Plan",
        summary,
        sections=[
            ("Image Requirements", f"<pre>{html_lib.escape(json.dumps(json_ready(summary.get('image_requirements', {})), indent=2))}</pre>"),
            ("Checks", _checks_table(summary.get("checks", ()))),
            ("Manifest", _manifest_table(summary.get("manifest", ()))),
        ],
        footer=f"Raw JSON: <code>{SUBSET_SUMMARY_FILENAME}</code>, manifest: <code>{MANIFEST_FILENAME}</code>",
    )


def _render_availability_html(summary: Mapping[str, Any]) -> str:
    missing_rows = "".join(_missing_row(row) for row in list(summary.get("missing_files", ()))[:80] if isinstance(row, Mapping))
    missing_table = f"<table><thead><tr><th>Kind</th><th>Sample</th><th>Expected Relative Path</th></tr></thead><tbody>{missing_rows}</tbody></table>"
    return _render_common_html(
        "LOD Image Availability Audit",
        summary,
        sections=[
            ("Checks", _checks_table(summary.get("checks", ()))),
            ("Missing Files", missing_table),
        ],
        footer=f"Raw JSON: <code>{AVAILABILITY_SUMMARY_FILENAME}</code>, required list: <code>{REQUIRED_FILES_FILENAME}</code>, missing list: <code>{MISSING_FILES_FILENAME}</code>",
    )


def _render_readiness_html(summary: Mapping[str, Any]) -> str:
    directory_rows = "".join(_directory_row(row) for row in summary.get("directories", ()) if isinstance(row, Mapping))
    directory_table = (
        "<table><thead><tr><th>Kind</th><th>Relative Path</th><th>Exists</th><th>Files</th><th>XML</th></tr></thead>"
        f"<tbody>{directory_rows}</tbody></table>"
    )
    return _render_common_html(
        "LOD Local Readiness",
        summary,
        sections=[
            ("Directories", directory_table),
            ("Checks", _checks_table(summary.get("checks", ()))),
        ],
        footer=f"Raw JSON: <code>{LOCAL_READINESS_SUMMARY_FILENAME}</code>",
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
    .fail, .blocked, .warning {{ color: #b91c1c; font-weight: 650; }}
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
        "<table><thead><tr><th>Sample</th><th>RAW Path</th><th>sRGB Path</th><th>Boxes</th><th>Labels</th></tr></thead>"
        f"<tbody>{manifest_rows}</tbody></table>"
    )


def _manifest_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('expected_raw_relative_path', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('expected_srgb_relative_path', '')))}</code></td>"
        f"<td>{int(row.get('box_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('labels', ())))}</td>"
        "</tr>"
    )


def _missing_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('kind', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('sample_id', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('relative_path', '')))}</code></td>"
        "</tr>"
    )


def _directory_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('kind', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('relative_path', '')))}</code></td>"
        f"<td>{html_lib.escape(str(bool(row.get('exists', False))))}</td>"
        f"<td>{int(row.get('file_count', 0))}</td>"
        f"<td>{int(row.get('xml_count', 0))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
