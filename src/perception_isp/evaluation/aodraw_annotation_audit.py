"""Inspect AODRaw COCO-format annotation archives."""

from __future__ import annotations

import argparse
import collections
import html as html_lib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "aodraw_annotation_audit_summary.json"
REQUIRED_JSONS = (
    "train_annotations.json",
    "test_annotations.json",
    "train_annotations_downsample_scale3_bbox_min_size32.json",
    "test_annotations_downsample_scale3_bbox_min_size32.json",
)
RECOMMENDED_FIRST_SPLIT = "test_annotations_downsample_scale3_bbox_min_size32.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit AODRaw annotation archive readiness.")
    parser.add_argument("annotations", help="AODRaw_annotations.zip or extracted annotations directory.")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_annotation_audit")
    args = parser.parse_args(argv)

    summary = build_aodraw_annotation_audit(args.annotations)
    html_path = write_aodraw_annotation_audit(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "split_count": len(summary["splits"]),
                    "recommended_first_split": summary["recommended_first_split"],
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_aodraw_annotation_audit(annotations: str | Path) -> Dict[str, Any]:
    path = Path(annotations).expanduser()
    payloads = _load_json_payloads(path)
    splits = [_split_summary(name, payload) for name, payload in sorted(payloads.items())]
    checks = _checks(payloads, splits)
    return {
        "name": "AODRaw annotation audit",
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "source": str(path),
        "split_count": len(splits),
        "splits": splits,
        "checks": checks,
        "recommended_first_split": RECOMMENDED_FIRST_SPLIT,
        "recommended_first_reason": (
            "Use the held-out downsampled test annotations first: it keeps the official test split, avoids 6000x4000 original images, "
            "and avoids the much larger sliced patch set for the first ingest."
        ),
        "claim_boundary": (
            "This validates annotation readiness only. Real RAW performance still requires image download, RAW decoding, fixed HumanISP/PerceptionISP processing, and detector evaluation."
        ),
    }


def write_aodraw_annotation_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _load_json_payloads(path: Path) -> Dict[str, Mapping[str, Any]]:
    if path.is_dir():
        payloads = {}
        for json_path in path.rglob("*.json"):
            payloads[json_path.name] = json.loads(json_path.read_text())
        return payloads
    if zipfile.is_zipfile(path):
        payloads = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                with archive.open(name) as handle:
                    payloads[Path(name).name] = json.load(handle)
        return payloads
    raise ValueError(f"Unsupported AODRaw annotation path: {path}")


def _split_summary(name: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    images = [row for row in payload.get("images", ()) if isinstance(row, Mapping)]
    annotations = [row for row in payload.get("annotations", ()) if isinstance(row, Mapping)]
    categories = [row for row in payload.get("categories", ()) if isinstance(row, Mapping)]
    tags = collections.Counter()
    extensions = collections.Counter()
    dimensions = collections.Counter()
    for image in images:
        for tag in image.get("tag", ()) or ():
            tags[str(tag)] += 1
        extensions[Path(str(image.get("file_name", ""))).suffix.lower() or "missing"] += 1
        dimensions[f"{int(image.get('width', 0))}x{int(image.get('height', 0))}"] += 1
    category_names = {int(row.get("id", -1)): str(row.get("name", "")) for row in categories if row.get("id") is not None}
    category_counts = collections.Counter()
    invalid_bbox_count = 0
    ignored_invalid_bbox_count = 0
    active_invalid_bbox_count = 0
    for ann in annotations:
        category_counts[category_names.get(int(ann.get("category_id", -1)), "missing")] += 1
        bbox = ann.get("bbox", ())
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(float(value) < 0.0 for value in bbox)
            or float(bbox[2]) <= 0.0
            or float(bbox[3]) <= 0.0
        ):
            invalid_bbox_count += 1
            if int(ann.get("iscrowd", 0)) == 1 or int(ann.get("ignore", 0)) == 1:
                ignored_invalid_bbox_count += 1
            else:
                active_invalid_bbox_count += 1
    ignore_id = next((int(row.get("id")) for row in categories if str(row.get("name", "")) == "ignore"), None)
    return {
        "name": str(name),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "object_category_count": len([row for row in categories if str(row.get("name", "")) != "ignore"]),
        "ignore_category_id": ignore_id,
        "invalid_bbox_count": invalid_bbox_count,
        "ignored_invalid_bbox_count": ignored_invalid_bbox_count,
        "active_invalid_bbox_count": active_invalid_bbox_count,
        "tags": dict(tags.most_common()),
        "extensions": dict(extensions.most_common()),
        "dimensions": dict(dimensions.most_common()),
        "top_categories": dict(category_counts.most_common(12)),
        "sample_image": _sample_image(images),
    }


def _sample_image(images: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    first = next(iter(images), None)
    if first is None:
        return {}
    return {
        "id": first.get("id"),
        "file_name": first.get("file_name"),
        "width": first.get("width"),
        "height": first.get("height"),
        "tag": list(first.get("tag", ()) or ()),
    }


def _checks(payloads: Mapping[str, Mapping[str, Any]], splits: list[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    split_by_name = {str(row.get("name", "")): row for row in splits}
    missing = [name for name in REQUIRED_JSONS if name not in payloads]
    recommended = split_by_name.get(RECOMMENDED_FIRST_SPLIT, {})
    all_tags = {tag for row in splits for tag in (row.get("tags", {}) if isinstance(row.get("tags"), Mapping) else {})}
    ignore_ok = any(row.get("ignore_category_id") is not None for row in splits)
    invalid_bbox_count = sum(int(row.get("invalid_bbox_count", 0)) for row in splits)
    ignored_invalid_bbox_count = sum(int(row.get("ignored_invalid_bbox_count", 0)) for row in splits)
    active_invalid_bbox_count = sum(int(row.get("active_invalid_bbox_count", 0)) for row in splits)
    return [
        {
            "id": "required_annotation_jsons_present",
            "status": "pass" if not missing else "fail",
            "evidence": "missing=none" if not missing else f"missing={','.join(missing)}",
        },
        {
            "id": "heldout_downsample_test_available",
            "status": "pass" if int(recommended.get("image_count", 0)) > 0 else "fail",
            "evidence": f"{RECOMMENDED_FIRST_SPLIT} images={int(recommended.get('image_count', 0))} anns={int(recommended.get('annotation_count', 0))}",
        },
        {
            "id": "adverse_tags_present",
            "status": "pass" if {"low_light", "rain", "fog"}.issubset(all_tags) else "fail",
            "evidence": f"tags={','.join(sorted(all_tags))}",
        },
        {
            "id": "ignore_category_is_explicit",
            "status": "pass" if ignore_ok else "fail",
            "evidence": "ignore category present" if ignore_ok else "ignore category missing",
        },
        {
            "id": "bbox_schema_valid",
            "status": "pass" if active_invalid_bbox_count == 0 else "fail",
            "evidence": (
                f"active_invalid_bbox_count={active_invalid_bbox_count}; "
                f"ignored_invalid_bbox_count={ignored_invalid_bbox_count}; invalid_bbox_count={invalid_bbox_count}"
            ),
        },
    ]


def _render_html(summary: Mapping[str, Any]) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    split_rows = "".join(_split_row(row) for row in summary.get("splits", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Annotation Audit</title>
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
  <h1>AODRaw Annotation Audit</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; recommended first split: <code>{html_lib.escape(str(summary.get('recommended_first_split', '')))}</code>.</p>
  <p>{html_lib.escape(str(summary.get('recommended_first_reason', '')))}</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Splits</h2>
  <table><thead><tr><th>JSON</th><th>Images</th><th>Annotations</th><th>Categories</th><th>Ignore ID</th><th>Dimensions</th><th>Tags</th><th>Top Categories</th><th>Sample</th></tr></thead><tbody>{split_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
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


def _split_row(row: Mapping[str, Any]) -> str:
    sample = row.get("sample_image", {}) if isinstance(row.get("sample_image"), Mapping) else {}
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('name', '')))}</code></td>"
        f"<td>{int(row.get('image_count', 0))}</td>"
        f"<td>{int(row.get('annotation_count', 0))}</td>"
        f"<td>{int(row.get('object_category_count', 0))}+ignore</td>"
        f"<td>{html_lib.escape(str(row.get('ignore_category_id', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('dimensions', {})))}</td>"
        f"<td>{html_lib.escape(str(row.get('tags', {})))}</td>"
        f"<td>{html_lib.escape(str(row.get('top_categories', {})))}</td>"
        f"<td>{html_lib.escape(str(sample))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
