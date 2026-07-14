"""Import manually downloaded AODRaw files into the expected dataset layout."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "aodraw_download_import_summary.json"
MISSING_FILES_FILENAME = "aodraw_import_missing_files.txt"
IMPORTED_FILES_FILENAME = "aodraw_imported_files.txt"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Import manually downloaded AODRaw subset files from a download directory.")
    parser.add_argument("--manifest", default="reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json")
    parser.add_argument("--download-root", action="append", default=[], help="Directory or zip file to scan. Repeatable. Defaults to ~/Downloads.")
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw")
    parser.add_argument("--kind", choices=["all", "raw", "srgb"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_download_import_v1")
    args = parser.parse_args(argv)

    summary = import_aodraw_downloads(
        manifest=args.manifest,
        download_roots=tuple(args.download_root) if args.download_root else (Path.home() / "Downloads",),
        dataset_root=args.dataset_root,
        kind=str(args.kind),
        dry_run=bool(args.dry_run),
    )
    html_path = write_aodraw_download_import(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "dry_run": summary["dry_run"],
                    "required_file_count": summary["required_file_count"],
                    "resolved_file_count": summary["resolved_file_count"],
                    "imported_file_count": summary["imported_file_count"],
                    "missing_file_count": summary["missing_file_count"],
                    "source_scan_error_count": summary["source_scan_error_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def import_aodraw_downloads(
    *,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    download_roots: Sequence[str | Path],
    dataset_root: str | Path,
    kind: str = "all",
    dry_run: bool = False,
) -> Dict[str, Any]:
    rows = _load_manifest(manifest)
    required = _required_paths(rows, kind=kind)
    root = Path(dataset_root).expanduser()
    roots = tuple(Path(item).expanduser() for item in download_roots)
    source_index, source_scan_errors = _build_source_index(roots, required)
    actions = []
    for relative_path in required:
        target = root / relative_path
        if target.is_file():
            actions.append(_action(relative_path, target, status="already_present", source="", source_kind="target"))
            continue
        source = source_index.get(relative_path)
        if source is None:
            actions.append(_action(relative_path, target, status="missing", source="", source_kind=""))
            continue
        if dry_run:
            actions.append(_action(relative_path, target, status="would_import", source=source["source"], source_kind=source["kind"]))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source["kind"] == "file":
            shutil.copy2(source["path"], target)
        elif source["kind"] == "zip":
            with zipfile.ZipFile(source["archive"]) as archive:
                with archive.open(source["member"]) as handle, target.open("wb") as output:
                    shutil.copyfileobj(handle, output)
        else:
            raise ValueError(f"Unsupported source kind: {source['kind']}")
        actions.append(_action(relative_path, target, status="imported", source=source["source"], source_kind=source["kind"]))

    missing = [row for row in actions if row["status"] == "missing"]
    imported = [row for row in actions if row["status"] == "imported"]
    would_import = [row for row in actions if row["status"] == "would_import"]
    already = [row for row in actions if row["status"] == "already_present"]
    resolved_count = len(imported) + len(would_import) + len(already)
    return {
        "name": "AODRaw download import",
        "status": "pass" if not missing and not dry_run else ("ready_to_import" if not missing else "blocked"),
        "dry_run": bool(dry_run),
        "dataset_root": str(root),
        "download_roots": [str(item) for item in roots],
        "manifest_source": str(manifest) if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, Path)) else "in_memory",
        "kind": str(kind),
        "required_file_count": len(required),
        "resolved_file_count": int(resolved_count),
        "imported_file_count": len(imported),
        "would_import_file_count": len(would_import),
        "already_present_file_count": len(already),
        "missing_file_count": len(missing),
        "actions": actions,
        "missing_files": [row["relative_path"] for row in missing],
        "imported_files": [row["relative_path"] for row in imported],
        "source_scan_error_count": len(source_scan_errors),
        "source_scan_errors": source_scan_errors,
        "next_action": _next_action(dry_run=bool(dry_run), missing_count=len(missing), imported_count=len(imported)),
        "post_import_command": _post_import_command(kind=str(kind)),
        "claim_boundary": (
            "This importer only copies/extracts files into the expected local layout. It does not prove RAW decoding or detector performance until the availability and evaluation gates pass."
        ),
    }


def write_aodraw_download_import(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / MISSING_FILES_FILENAME).write_text("\n".join(str(path) for path in summary.get("missing_files", ())) + ("\n" if summary.get("missing_files") else ""))
    (destination / IMPORTED_FILES_FILENAME).write_text("\n".join(str(path) for path in summary.get("imported_files", ())) + ("\n" if summary.get("imported_files") else ""))
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Path)):
        return [dict(row) for row in manifest if isinstance(row, Mapping)]
    payload = json.loads(Path(manifest).expanduser().read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("manifest", ())
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("AODRaw manifest must be a JSON list or a summary with a manifest key")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _required_paths(rows: Sequence[Mapping[str, Any]], *, kind: str) -> list[str]:
    keys = []
    normalized = str(kind).lower()
    if normalized in {"all", "raw"}:
        keys.append("expected_raw_relative_path")
    if normalized in {"all", "srgb"}:
        keys.append("expected_srgb_relative_path")
    required = {
        str(row.get(key, "")).strip()
        for row in rows
        for key in keys
        if str(row.get(key, "")).strip()
    }
    return sorted(required)


def _build_source_index(roots: Sequence[Path], required: Sequence[str]) -> tuple[Dict[str, Dict[str, Any]], list[Dict[str, Any]]]:
    required_set = set(required)
    by_basename: dict[str, list[str]] = {}
    for relative_path in required:
        by_basename.setdefault(Path(relative_path).name, []).append(relative_path)
    sources: Dict[str, Dict[str, Any]] = {}
    scan_errors: list[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".zip":
            _index_zip(root, by_basename, required_set, sources, scan_errors)
            continue
        if root.is_dir():
            _index_directory(root, by_basename, required_set, sources)
            for archive in root.rglob("*.zip"):
                _index_zip(archive, by_basename, required_set, sources, scan_errors)
    return sources, scan_errors


def _index_directory(
    root: Path,
    by_basename: Mapping[str, Sequence[str]],
    required_set: set[str],
    sources: Dict[str, Dict[str, Any]],
) -> None:
    wanted_names = set(by_basename)
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in wanted_names:
            continue
        relative_match = _best_match(path, by_basename[path.name], required_set)
        if relative_match and relative_match not in sources:
            sources[relative_match] = {"kind": "file", "path": path, "source": str(path)}


def _index_zip(
    archive_path: Path,
    by_basename: Mapping[str, Sequence[str]],
    required_set: set[str],
    sources: Dict[str, Dict[str, Any]],
    scan_errors: list[Dict[str, Any]],
) -> None:
    wanted_names = set(by_basename)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                path = Path(member)
                if member.endswith("/") or path.name not in wanted_names:
                    continue
                relative_match = _best_match(path, by_basename[path.name], required_set)
                if relative_match and relative_match not in sources:
                    sources[relative_match] = {
                        "kind": "zip",
                        "archive": archive_path,
                        "member": member,
                        "source": f"{archive_path}!{member}",
                    }
    except (OSError, zipfile.BadZipFile) as exc:
        scan_errors.append(
            {
                "path": str(archive_path),
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "The archive is not usable yet. It may be an incomplete or failed browser download.",
            }
        )


def _best_match(path: Path, candidates: Sequence[str], required_set: set[str]) -> str:
    normalized_parts = [part for part in path.parts if part]
    normalized_suffix = "/".join(normalized_parts)
    for candidate in candidates:
        if normalized_suffix.endswith(candidate):
            return candidate
    if len(candidates) == 1 and candidates[0] in required_set:
        return str(candidates[0])
    return ""


def _action(relative_path: str, target: Path, *, status: str, source: str, source_kind: str) -> Dict[str, Any]:
    return {
        "relative_path": str(relative_path),
        "target_path": str(target),
        "status": str(status),
        "source": str(source),
        "source_kind": str(source_kind),
    }


def _next_action(*, dry_run: bool, missing_count: int, imported_count: int) -> str:
    if missing_count:
        return "Some subset files were not found. Download them first, then rerun the importer."
    if dry_run:
        return "Dry-run resolved all required files. Rerun without --dry-run to copy/extract them."
    if imported_count:
        return "Files were imported. Rerun the AODRaw image availability gate."
    return "All required files were already present. Run the AODRaw image availability gate."


def _post_import_command(*, kind: str) -> str:
    suffix = "_raw_only" if str(kind) == "raw" else ""
    return (
        "perception-isp data aodraw-availability "
        "reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json "
        f"--kind {kind} "
        "--dataset-root data/raw_datasets/aodraw "
        f"--output-dir reports/perception_aodraw_image_availability_test_downsample_adverse_24{suffix}_v1"
    )


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = "".join(_action_row(row) for row in summary.get("actions", ()) if isinstance(row, Mapping))
    error_rows = "".join(_scan_error_row(row) for row in summary.get("source_scan_errors", ()) if isinstance(row, Mapping))
    scan_error_section = (
        "<h2>Source Scan Errors</h2>"
        f"<table><thead><tr><th>Path</th><th>Error</th><th>Hint</th></tr></thead><tbody>{error_rows}</tbody></table>"
        if error_rows
        else ""
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Download Import</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .pass, .imported, .already_present, .would_import {{ color: #047857; font-weight: 650; }}
    .blocked, .missing {{ color: #b91c1c; font-weight: 650; }}
    .ready_to_import {{ color: #b45309; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    pre {{ background: #111827; color: #f9fafb; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>AODRaw Download Import</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; dry_run=<code>{html_lib.escape(str(summary.get('dry_run', '')))}</code>.</p>
  <p>Required={int(summary.get('required_file_count', 0))}; resolved={int(summary.get('resolved_file_count', 0))}; imported={int(summary.get('imported_file_count', 0))}; missing={int(summary.get('missing_file_count', 0))}.</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <h2>Actions</h2>
  <table><thead><tr><th>Status</th><th>Relative Path</th><th>Source</th><th>Target</th></tr></thead><tbody>{rows}</tbody></table>
  {scan_error_section}
  <h2>Post-import Command</h2>
  <pre>{html_lib.escape(str(summary.get('post_import_command', '')))}</pre>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code>, missing list: <code>{MISSING_FILES_FILENAME}</code>, imported list: <code>{IMPORTED_FILES_FILENAME}</code></p>
</body>
</html>
"""


def _action_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td><code>{html_lib.escape(str(row.get('relative_path', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('source', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(row.get('target_path', '')))}</code></td>"
        "</tr>"
    )


def _scan_error_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('path', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('error', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('hint', '')))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
