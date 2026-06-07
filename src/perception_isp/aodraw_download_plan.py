"""Create an actionable AODRaw partial-download plan."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "aodraw_download_plan_summary.json"
CHECKLIST_FILENAME = "aodraw_manual_download_checklist.md"
TARGETS_FILENAME = "aodraw_download_targets.txt"


AODRAW_OFFICIAL_URLS = {
    "official_repo": "https://github.com/lzyhha/AODRaw",
    "annotations_google_drive": "https://drive.google.com/file/d/1VEm1TRur7UgjzzEB2vx1kApC1xvceMzS/view?usp=sharing",
    "downsampled_srgb_baidu": "https://pan.baidu.com/s/1_56k-Tr1JGDI99xFugPGtQ?pwd=aerr",
    "downsampled_srgb_terabox": "https://terabox.com/s/1QerBpH6FaGCE05cXks2XxQ",
    "downsampled_raw_baidu": "https://pan.baidu.com/s/1QvqKuBPIgWXzdoABo-L-MQ?pwd=5v4a",
    "original_images_baidu": "https://pan.baidu.com/s/1WqPZz_E9godci3FHlx07EQ?pwd=i2dv",
    "original_images_terabox": "https://terabox.com/s/1QMnQ7z0V9Wy79pBylG5ZBw",
}

AODRAW_DOWNSAMPLED_RAW_ARCHIVES = {
    "test": {
        "filename": "AODRaw_test_downsampled_raw.zip",
        "size_gb": 58.94,
        "size_bytes": 58936290415,
        "fs_id": "39940515692639",
    },
    "train": {
        "filename": "AODRaw_train_downsampled_raw.zip",
        "size_gb": 137.19,
        "size_bytes": 137186166557,
        "fs_id": "472511554161908",
    },
}

RAW_DOWNLOAD_HEADROOM_GIB = 10.0


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create an AODRaw partial-download plan.")
    parser.add_argument("--manifest", default="reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json")
    parser.add_argument("--availability-summary", default="reports/perception_aodraw_image_availability_test_downsample_adverse_24_v1/aodraw_image_availability_summary.json")
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_download_plan_v1")
    args = parser.parse_args(argv)

    summary = build_aodraw_download_plan(
        manifest=args.manifest,
        availability_summary=args.availability_summary,
        dataset_root=args.dataset_root,
    )
    html_path = write_aodraw_download_plan(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "checklist": str(html_path.parent / CHECKLIST_FILENAME),
                    "targets": str(html_path.parent / TARGETS_FILENAME),
                    "status": summary["status"],
                    "recommended_first": summary["recommended_first"],
                    "disk_available_gib": summary["disk"]["available_gib"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_aodraw_download_plan(
    *,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    availability_summary: str | Path | Mapping[str, Any] | None,
    dataset_root: str | Path,
) -> Dict[str, Any]:
    root = Path(dataset_root).expanduser()
    manifest_rows = _load_manifest(manifest)
    availability = _load_availability_summary(availability_summary)
    raw_files = sorted({str(row.get("expected_raw_relative_path", "")) for row in manifest_rows if row.get("expected_raw_relative_path")})
    srgb_files = sorted({str(row.get("expected_srgb_relative_path", "")) for row in manifest_rows if row.get("expected_srgb_relative_path")})
    missing_files = [str(row.get("relative_path", "")) for row in availability.get("missing_files", ()) if isinstance(row, Mapping)]
    disk = _disk_summary(root)
    steps = _download_steps(root=root, raw_files=raw_files, srgb_files=srgb_files, disk=disk)
    checks = _checks(manifest_rows, raw_files, srgb_files, availability, disk)
    return {
        "name": "AODRaw partial-download plan",
        "status": "ready_for_manual_download" if _ready_for_raw_download(checks) else "blocked",
        "dataset_root": str(root),
        "manifest_source": str(manifest) if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, Path)) else "in_memory",
        "availability_source": "" if availability_summary is None else str(availability_summary),
        "sample_count": len(manifest_rows),
        "subset_raw_file_count": len(raw_files),
        "subset_srgb_file_count": len(srgb_files),
        "missing_file_count": len(missing_files),
        "official_urls": dict(AODRAW_OFFICIAL_URLS),
        "disk": disk,
        "download_steps": steps,
        "required_subset_files": sorted(set(raw_files + srgb_files)),
        "missing_files": missing_files,
        "downsampled_raw_archives": dict(AODRAW_DOWNSAMPLED_RAW_ARCHIVES),
        "recommended_first": _recommended_first(disk),
        "post_download_commands": _post_download_commands(),
        "checks": checks,
        "claim_boundary": (
            "This is a download/action plan only. It does not prove image availability or PerceptionISP performance until the files are present and the availability/evaluation gates pass."
        ),
    }


def write_aodraw_download_plan(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    (destination / CHECKLIST_FILENAME).write_text(_render_checklist(summary))
    targets = [str(path) for path in summary.get("required_subset_files", ())]
    (destination / TARGETS_FILENAME).write_text("\n".join(targets) + ("\n" if targets else ""))
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


def _load_availability_summary(value: str | Path | Mapping[str, Any] | None) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser()
    return json.loads(path.read_text()) if path.exists() else {}


def _disk_summary(root: Path) -> Dict[str, Any]:
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(probe),
        "total_gib": _bytes_to_gib(usage.total),
        "used_gib": _bytes_to_gib(usage.used),
        "available_gib": _bytes_to_gib(usage.free),
        "fits_downsampled_srgb_4p3gb": bool(usage.free >= int(4.3 * 1024**3)),
        "fits_downsampled_raw_test_zip_58p9gb": bool(usage.free >= int(AODRAW_DOWNSAMPLED_RAW_ARCHIVES["test"]["size_bytes"])),
        "fits_downsampled_raw_test_zip_with_headroom": bool(
            usage.free >= int(AODRAW_DOWNSAMPLED_RAW_ARCHIVES["test"]["size_bytes"]) + int(RAW_DOWNLOAD_HEADROOM_GIB * 1024**3)
        ),
        "fits_downsampled_raw_train_zip_137p2gb": bool(usage.free >= int(AODRAW_DOWNSAMPLED_RAW_ARCHIVES["train"]["size_bytes"])),
        "fits_downsampled_raw_all_zips_196p1gb": bool(
            usage.free
            >= int(AODRAW_DOWNSAMPLED_RAW_ARCHIVES["test"]["size_bytes"]) + int(AODRAW_DOWNSAMPLED_RAW_ARCHIVES["train"]["size_bytes"])
        ),
        "fits_downsampled_raw_223gb": bool(usage.free >= int(223.0 * 1024**3)),
        "fits_downsampled_srgb_plus_raw_test_zip": bool(usage.free >= int(AODRAW_DOWNSAMPLED_RAW_ARCHIVES["test"]["size_bytes"]) + int(4.3 * 1024**3)),
        "raw_test_zip_download_headroom_gib": float(RAW_DOWNLOAD_HEADROOM_GIB),
    }


def _bytes_to_gib(value: int) -> float:
    return round(float(value) / float(1024**3), 2)


def _download_steps(
    *,
    root: Path,
    raw_files: Sequence[str],
    srgb_files: Sequence[str],
    disk: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    return [
        {
            "priority": "P0",
            "name": "Download AODRaw test downsampled RAW zip",
            "target_directory": "~/Downloads or " + str(root / "downloads"),
            "expected_size_gb": AODRAW_DOWNSAMPLED_RAW_ARCHIVES["test"]["size_gb"],
            "source_urls": [AODRAW_OFFICIAL_URLS["downsampled_raw_baidu"]],
            "archive_filename": AODRAW_DOWNSAMPLED_RAW_ARCHIVES["test"]["filename"],
            "required_for_subset_files": [path for path in raw_files],
            "status": "recommended_now" if disk.get("fits_downsampled_raw_test_zip_with_headroom") else "disk_blocked",
            "why": (
                "This is the first real RAW archive to acquire; the importer can extract only the selected subset .npy files from this zip. "
                f"Keep at least {RAW_DOWNLOAD_HEADROOM_GIB:.0f} GiB free beyond the zip size to avoid incomplete browser downloads."
            ),
        },
        {
            "priority": "P0",
            "name": "Download downsampled sRGB zip or directory",
            "target_directory": str(root / "images_downsampled_srgb"),
            "expected_size_gb": 4.3,
            "source_urls": [AODRAW_OFFICIAL_URLS["downsampled_srgb_baidu"], AODRAW_OFFICIAL_URLS["downsampled_srgb_terabox"]],
            "required_for_subset_files": [path for path in srgb_files],
            "status": "recommended_now" if disk.get("fits_downsampled_srgb_4p3gb") else "disk_blocked",
            "why": "Small enough to download now and enables paired sRGB/reference checks for the selected adverse subset.",
        },
        {
            "priority": "P1",
            "name": "Download selected downsampled RAW files if the file browser supports partial selection",
            "target_directory": str(root / "images_downsampled_raw"),
            "expected_size_gb": None,
            "source_urls": [AODRAW_OFFICIAL_URLS["downsampled_raw_baidu"]],
            "required_for_subset_files": [path for path in raw_files],
            "status": "recommended_if_partial_selection_available",
            "why": "This would be faster than the test zip, but the current Baidu link appears to expose zip archives rather than individual files.",
        },
        {
            "priority": "P2",
            "name": "Download AODRaw train downsampled RAW zip",
            "target_directory": "~/Downloads or " + str(root / "downloads"),
            "expected_size_gb": AODRAW_DOWNSAMPLED_RAW_ARCHIVES["train"]["size_gb"],
            "source_urls": [AODRAW_OFFICIAL_URLS["downsampled_raw_baidu"]],
            "archive_filename": AODRAW_DOWNSAMPLED_RAW_ARCHIVES["train"]["filename"],
            "required_for_subset_files": [],
            "status": "possible_but_high_disk_cost" if disk.get("fits_downsampled_raw_train_zip_137p2gb") else "disk_blocked",
            "why": "Defer until the test RAW evaluation works and disk headroom is increased.",
        },
    ]


def _recommended_first(disk: Mapping[str, Any]) -> str:
    if disk.get("fits_downsampled_raw_test_zip_with_headroom"):
        return "AODRaw_test_downsampled_raw.zip 58.94GB via Baidu"
    if disk.get("fits_downsampled_srgb_4p3gb"):
        return f"Free at least {RAW_DOWNLOAD_HEADROOM_GIB:.0f} GiB more headroom, then download AODRaw_test_downsampled_raw.zip"
    return "Free disk space before downloading AODRaw images"


def _checks(
    rows: Sequence[Mapping[str, Any]],
    raw_files: Sequence[str],
    srgb_files: Sequence[str],
    availability: Mapping[str, Any],
    disk: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    return [
        {"id": "manifest_loaded", "status": "pass" if rows else "fail", "evidence": f"samples={len(rows)}"},
        {"id": "subset_raw_targets_listed", "status": "pass" if raw_files else "fail", "evidence": f"raw_files={len(raw_files)}"},
        {"id": "subset_srgb_targets_listed", "status": "pass" if srgb_files else "fail", "evidence": f"srgb_files={len(srgb_files)}"},
        {
            "id": "sufficient_disk_for_srgb",
            "status": "pass" if disk.get("fits_downsampled_srgb_4p3gb") else "fail",
            "evidence": f"available_gib={disk.get('available_gib')}",
        },
        {
            "id": "sufficient_disk_for_test_raw_zip",
            "status": "pass" if disk.get("fits_downsampled_raw_test_zip_58p9gb") else "fail",
            "evidence": f"available_gib={disk.get('available_gib')} raw_test_gb={AODRAW_DOWNSAMPLED_RAW_ARCHIVES['test']['size_gb']}",
        },
        {
            "id": "sufficient_disk_for_test_raw_zip_with_headroom",
            "status": "pass" if disk.get("fits_downsampled_raw_test_zip_with_headroom") else "fail",
            "evidence": (
                f"available_gib={disk.get('available_gib')} raw_test_gb={AODRAW_DOWNSAMPLED_RAW_ARCHIVES['test']['size_gb']} "
                f"required_headroom_gib={RAW_DOWNLOAD_HEADROOM_GIB:.0f}"
            ),
        },
        {
            "id": "sufficient_disk_for_train_raw_zip",
            "status": "pass" if disk.get("fits_downsampled_raw_train_zip_137p2gb") else "warning",
            "evidence": f"available_gib={disk.get('available_gib')} raw_train_gb={AODRAW_DOWNSAMPLED_RAW_ARCHIVES['train']['size_gb']}",
        },
        {
            "id": "current_availability_gate",
            "status": "pass" if bool(availability.get("evaluation_ready", False)) else "fail",
            "evidence": f"evaluation_ready={availability.get('evaluation_ready', False)} missing={availability.get('missing_file_count', 'unknown')}",
        },
    ]


def _post_download_commands() -> list[str]:
    return [
        (
            "PYTHONPATH=src python3 -m perception_isp.aodraw_image_availability "
            "reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json "
            "--kind raw "
            "--dataset-root data/raw_datasets/aodraw "
            "--output-dir reports/perception_aodraw_image_availability_test_downsample_adverse_24_raw_only_v1"
        ),
        (
            "PYTHONPATH=src python3 -m perception_isp.eval_cli "
            "--source aodraw-dataset "
            "--dataset data/raw_datasets/aodraw "
            "--aodraw-manifest reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json "
            "--count 24 --width 768 --height 512 --aodraw-cfa RGGB "
            "--rgb-detector yolo --rgb-detector-model yolo11n.pt --label-aware "
            "--ground-truth-label-map aodraw-coco --ground-truth-label-keep aodraw-coco-overlap "
            "--output-dir reports/perception_aodraw_compare_test_downsample_adverse_24_v1"
        ),
    ]


def _ready_for_raw_download(checks: Sequence[Mapping[str, Any]]) -> bool:
    statuses = {str(row.get("id", "")): str(row.get("status", "")) for row in checks}
    required = (
        "manifest_loaded",
        "subset_raw_targets_listed",
        "sufficient_disk_for_test_raw_zip",
        "sufficient_disk_for_test_raw_zip_with_headroom",
    )
    return all(statuses.get(item) == "pass" for item in required)


def _render_checklist(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AODRaw Manual Download Checklist",
        "",
        f"Dataset root: `{summary.get('dataset_root')}`",
        f"Recommended first: {summary.get('recommended_first')}",
        "",
        "## Official Links",
    ]
    for key, url in summary.get("official_urls", {}).items():
        lines.append(f"- `{key}`: {url}")
    lines.extend(["", "## Steps"])
    for step in summary.get("download_steps", ()):
        lines.append(f"- [{step.get('priority')}] {step.get('name')}")
        lines.append(f"  - Target: `{step.get('target_directory')}`")
        lines.append(f"  - Status: `{step.get('status')}`")
        lines.append(f"  - Why: {step.get('why')}")
    lines.extend(["", "## Subset Files"])
    for path in summary.get("required_subset_files", ()):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Post-download Commands"])
    for command in summary.get("post_download_commands", ()):
        lines.append(f"```bash\n{command}\n```")
    return "\n".join(lines) + "\n"


def _render_html(summary: Mapping[str, Any]) -> str:
    steps = "".join(_step_row(row) for row in summary.get("download_steps", ()) if isinstance(row, Mapping))
    checks = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    commands = "".join(f"<pre>{html_lib.escape(str(command))}</pre>" for command in summary.get("post_download_commands", ()))
    urls = "".join(
        f"<li><code>{html_lib.escape(str(key))}</code>: <a href=\"{html_lib.escape(str(url))}\">{html_lib.escape(str(url))}</a></li>"
        for key, url in summary.get("official_urls", {}).items()
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Partial Download Plan</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .pass, .recommended_now {{ color: #047857; font-weight: 650; }}
    .fail, .blocked, .disk_blocked {{ color: #b91c1c; font-weight: 650; }}
    .warning, .possible_but_high_disk_cost, .recommended_if_partial_selection_available {{ color: #b45309; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    pre {{ background: #111827; color: #f9fafb; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>AODRaw Partial Download Plan</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>. Recommended first: <b>{html_lib.escape(str(summary.get('recommended_first', '')))}</b>.</p>
  <p>Disk available: <code>{html_lib.escape(str(summary.get('disk', {}).get('available_gib', '')))} GiB</code>; subset samples: {int(summary.get('sample_count', 0))}; missing files: {int(summary.get('missing_file_count', 0))}.</p>
  <h2>Official Links</h2>
  <ul>{urls}</ul>
  <h2>Download Steps</h2>
  <table><thead><tr><th>Priority</th><th>Name</th><th>Status</th><th>Target</th><th>Size GB</th><th>Why</th></tr></thead><tbody>{steps}</tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{checks}</tbody></table>
  <h2>Post-download Commands</h2>
  {commands}
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code>, checklist: <code>{CHECKLIST_FILENAME}</code>, subset targets: <code>{TARGETS_FILENAME}</code></p>
</body>
</html>
"""


def _step_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('name', '')))}</td>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td><code>{html_lib.escape(str(row.get('target_directory', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('expected_size_gb', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('why', '')))}</td>"
        "</tr>"
    )


def _check_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
