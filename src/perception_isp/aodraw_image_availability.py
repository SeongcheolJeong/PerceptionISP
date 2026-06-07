"""Audit local AODRaw image availability for a planned subset manifest."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "aodraw_image_availability_summary.json"
REQUIRED_FILES_FILENAME = "aodraw_required_files.txt"
MISSING_FILES_FILENAME = "aodraw_missing_files.txt"
MISSING_FILES_JSON_FILENAME = "aodraw_missing_files.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local AODRaw image files required by a subset manifest.")
    parser.add_argument("manifest", help="AODRaw subset manifest JSON, or subset-plan summary JSON containing a manifest key.")
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_image_availability")
    args = parser.parse_args(argv)

    summary = build_aodraw_image_availability(args.manifest, dataset_root=args.dataset_root)
    html_path = write_aodraw_image_availability(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "required_files_txt": str(html_path.parent / REQUIRED_FILES_FILENAME),
                    "missing_files_txt": str(html_path.parent / MISSING_FILES_FILENAME),
                    "status": summary["status"],
                    "evaluation_ready": summary["evaluation_ready"],
                    "required_file_count": summary["required_file_count"],
                    "missing_file_count": summary["missing_file_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_aodraw_image_availability(manifest: str | Path | Sequence[Mapping[str, Any]], *, dataset_root: str | Path) -> Dict[str, Any]:
    rows = _load_manifest(manifest)
    root = Path(dataset_root).expanduser()
    file_checks = _file_checks(rows, root)
    required_files = sorted({str(row["relative_path"]) for row in file_checks})
    missing_files = [row for row in file_checks if not bool(row["exists"])]
    missing_raw = [row for row in missing_files if row["kind"] == "raw"]
    missing_srgb = [row for row in missing_files if row["kind"] == "srgb"]
    total_size_bytes = sum(int(row["size_bytes"]) for row in file_checks if bool(row["exists"]))
    checks = _checks(rows, file_checks, required_files)
    evaluation_ready = bool(rows) and not missing_files and all(row["status"] == "pass" for row in checks)
    return {
        "name": "AODRaw image availability audit",
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
        "available_total_size_bytes": total_size_bytes,
        "required_files": required_files,
        "missing_files": [dict(row) for row in missing_files],
        "file_checks": file_checks,
        "condition_counts": _condition_counts(rows),
        "checks": checks,
        "next_action": _next_action(missing_raw=missing_raw, missing_srgb=missing_srgb),
        "claim_boundary": (
            "This is only a local file-availability gate. It does not prove RAW decoding, HumanISP/PerceptionISP parity, or detector performance."
        ),
    }


def write_aodraw_image_availability(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    required_files = [str(value) for value in summary.get("required_files", ())]
    missing_files = [str(row.get("relative_path", "")) for row in summary.get("missing_files", ()) if isinstance(row, Mapping)]
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / REQUIRED_FILES_FILENAME).write_text("\n".join(required_files) + ("\n" if required_files else ""))
    (destination / MISSING_FILES_FILENAME).write_text("\n".join(missing_files) + ("\n" if missing_files else ""))
    (destination / MISSING_FILES_JSON_FILENAME).write_text(json.dumps(json_ready(summary.get("missing_files", ())), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Path)):
        return [dict(row) for row in manifest if isinstance(row, Mapping)]
    path = Path(manifest).expanduser()
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("manifest", ())
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"AODRaw manifest must be a JSON list or contain a manifest list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _file_checks(rows: Sequence[Mapping[str, Any]], root: Path) -> list[Dict[str, Any]]:
    checks: list[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        for kind, key in (("raw", "expected_raw_relative_path"), ("srgb", "expected_srgb_relative_path")):
            relative_path = str(row.get(key, "")).strip()
            full_path = root / relative_path if relative_path else root
            exists = bool(relative_path) and full_path.is_file()
            size_bytes = full_path.stat().st_size if exists else 0
            checks.append(
                {
                    "manifest_index": int(index),
                    "image_id": int(row.get("image_id", -1)),
                    "file_name": str(row.get("file_name", "")),
                    "selection_condition": str(row.get("selection_condition", "")),
                    "kind": kind,
                    "relative_path": relative_path,
                    "absolute_path": str(full_path),
                    "exists": exists,
                    "size_bytes": int(size_bytes),
                }
            )
    return checks


def _checks(
    rows: Sequence[Mapping[str, Any]],
    file_checks: Sequence[Mapping[str, Any]],
    required_files: Sequence[str],
) -> list[Dict[str, Any]]:
    missing_paths = [row for row in file_checks if not bool(row.get("relative_path", ""))]
    missing_raw = [row for row in file_checks if row.get("kind") == "raw" and not bool(row.get("exists", False))]
    missing_srgb = [row for row in file_checks if row.get("kind") == "srgb" and not bool(row.get("exists", False))]
    empty_boxes = [row for row in rows if int(row.get("box_count", 0)) <= 0]
    duplicate_count = len(file_checks) - len(required_files)
    return [
        {
            "id": "manifest_loaded",
            "status": "pass" if rows else "fail",
            "evidence": f"rows={len(rows)}",
        },
        {
            "id": "manifest_rows_have_paths",
            "status": "pass" if not missing_paths else "fail",
            "evidence": f"empty_path_entries={len(missing_paths)}",
        },
        {
            "id": "raw_files_available",
            "status": "pass" if not missing_raw else "fail",
            "evidence": f"missing_raw={len(missing_raw)}",
        },
        {
            "id": "srgb_files_available",
            "status": "pass" if not missing_srgb else "fail",
            "evidence": f"missing_srgb={len(missing_srgb)}",
        },
        {
            "id": "selected_rows_have_boxes",
            "status": "pass" if not empty_boxes else "fail",
            "evidence": f"empty_box_rows={len(empty_boxes)}",
        },
        {
            "id": "duplicate_requirements_collapsed",
            "status": "pass",
            "evidence": f"duplicate_file_references={duplicate_count}",
        },
    ]


def _condition_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        condition = str(row.get("selection_condition", ""))
        counts[condition] = counts.get(condition, 0) + 1
    return counts


def _next_action(*, missing_raw: Sequence[Mapping[str, Any]], missing_srgb: Sequence[Mapping[str, Any]]) -> str:
    if missing_raw or missing_srgb:
        return (
            "Acquire only the missing paths listed in aodraw_missing_files.txt first. "
            "Avoid the full 223 GB downsampled RAW download unless partial/manual selection is impossible."
        )
    return "Run the AODRaw RAW decoding and HumanISP/PerceptionISP detector smoke test on this subset."


def _render_html(summary: Mapping[str, Any]) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    missing_rows = "".join(_missing_row(row) for row in list(summary.get("missing_files", ()))[:80] if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Image Availability Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .pass {{ color: #047857; font-weight: 650; }}
    .fail, .blocked {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>AODRaw Image Availability Audit</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; evaluation_ready=<code>{html_lib.escape(str(summary.get('evaluation_ready', '')))}</code>.</p>
  <p>Dataset root: <code>{html_lib.escape(str(summary.get('dataset_root', '')))}</code></p>
  <p>Required files: {int(summary.get('required_file_count', 0))}; missing: {int(summary.get('missing_file_count', 0))}; missing RAW: {int(summary.get('missing_raw_count', 0))}; missing sRGB: {int(summary.get('missing_srgb_count', 0))}.</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Missing Files</h2>
  <table><thead><tr><th>Kind</th><th>Condition</th><th>Source File</th><th>Expected Relative Path</th></tr></thead><tbody>{missing_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code>, required list: <code>{REQUIRED_FILES_FILENAME}</code>, missing list: <code>{MISSING_FILES_FILENAME}</code>.</p>
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


def _missing_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('kind', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('selection_condition', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('file_name', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('relative_path', '')))}</code></td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
