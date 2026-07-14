"""One-command AODRaw acquisition runner after explicit cleanup approval."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from perception_isp.datasets.aodraw_download_plan import (
    AODRAW_OFFICIAL_URLS,
    CLEANUP_CONFIRM_TOKEN,
    SUMMARY_FILENAME as PLAN_SUMMARY_FILENAME,
    build_aodraw_download_plan,
    write_aodraw_download_plan,
)
from perception_isp.datasets.aodraw_download_watch import SUMMARY_FILENAME as WATCH_SUMMARY_FILENAME
from perception_isp.datasets.aodraw_download_watch import watch_aodraw_downloads, write_aodraw_download_watch
from perception_isp.datasets.aodraw_storage_cleanup import SUMMARY_FILENAME as CLEANUP_SUMMARY_FILENAME
from perception_isp.datasets.aodraw_storage_cleanup import build_aodraw_storage_cleanup, write_aodraw_storage_cleanup
from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "aodraw_acquisition_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local AODRaw acquisition sequence after explicit cleanup approval.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json")
    parser.add_argument(
        "--availability-summary",
        default="reports/perception_aodraw_image_availability_test_downsample_adverse_24_v1/aodraw_image_availability_summary.json",
    )
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw")
    parser.add_argument("--download-root", action="append", default=[], help="Directory or zip file to scan. Repeatable. Defaults to ~/Downloads.")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_acquisition_v1")
    parser.add_argument("--execute-cleanup", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--open-download", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not open Safari; pass dry-run through to watcher/import.")
    parser.add_argument("--watch-iterations", type=int, default=1)
    parser.add_argument("--watch-interval", type=float, default=30.0)
    args = parser.parse_args(argv)

    summary = run_aodraw_acquisition(
        project_root=args.project_root,
        manifest=args.manifest,
        availability_summary=args.availability_summary,
        dataset_root=args.dataset_root,
        download_roots=tuple(args.download_root) if args.download_root else (Path.home() / "Downloads",),
        execute_cleanup=bool(args.execute_cleanup),
        confirm_token=str(args.confirm_token),
        open_download=bool(args.open_download),
        watch=bool(args.watch),
        dry_run=bool(args.dry_run),
        watch_iterations=int(args.watch_iterations),
        watch_interval_seconds=float(args.watch_interval),
        output_dir=args.output_dir,
    )
    html_path = write_aodraw_acquisition(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "cleanup_status": summary["cleanup"]["status"],
                    "download_open_status": summary["download_open"]["status"],
                    "watch_status": summary["watch"]["status"],
                    "post_cleanup_disk_available_gib": summary["post_cleanup_plan"]["disk"]["available_gib"],
                }
            ),
            indent=2,
        )
    )
    return 0


def run_aodraw_acquisition(
    *,
    project_root: str | Path,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    availability_summary: str | Path | Mapping[str, Any] | None,
    dataset_root: str | Path,
    download_roots: Sequence[str | Path],
    execute_cleanup: bool = False,
    confirm_token: str = "",
    open_download: bool = False,
    watch: bool = False,
    dry_run: bool = False,
    watch_iterations: int = 1,
    watch_interval_seconds: float = 30.0,
    output_dir: str | Path = "reports/perception_aodraw_acquisition_v1",
    opener: Any = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser()
    project_path = Path(project_root).expanduser().resolve()
    dataset_path = _resolve_project_path(project_path, dataset_root)
    manifest_path = _resolve_project_path(project_path, manifest) if isinstance(manifest, (str, Path)) else manifest
    availability_path = _resolve_project_path(project_path, availability_summary) if isinstance(availability_summary, (str, Path)) else availability_summary

    initial_plan = build_aodraw_download_plan(
        manifest=manifest_path,
        availability_summary=availability_path,
        dataset_root=dataset_path,
        project_root=project_path,
    )
    plan_report = write_aodraw_download_plan(initial_plan, output_path / "initial_plan")
    cleanup_candidate = str(initial_plan.get("cleanup", {}).get("first_sufficient_candidate", "")).strip()

    cleanup = build_aodraw_storage_cleanup(
        project_root=project_path,
        dataset_root=dataset_path,
        requested_candidates=(cleanup_candidate,) if cleanup_candidate else (),
        execute=bool(execute_cleanup),
        confirm_token=str(confirm_token),
    )
    cleanup_report = write_aodraw_storage_cleanup(cleanup, output_path / "cleanup")

    post_cleanup_plan = build_aodraw_download_plan(
        manifest=manifest_path,
        availability_summary=availability_path,
        dataset_root=dataset_path,
        project_root=project_path,
    )
    post_plan_report = write_aodraw_download_plan(post_cleanup_plan, output_path / "post_cleanup_plan")

    download_open = _open_download_link(
        enabled=bool(open_download),
        dry_run=bool(dry_run),
        disk_ready=bool(post_cleanup_plan.get("disk", {}).get("fits_downsampled_raw_test_zip_with_headroom", False)),
        opener=opener,
    )

    watch_summary: Dict[str, Any]
    watch_report = ""
    if watch:
        watch_summary = watch_aodraw_downloads(
            manifest=manifest_path,
            download_roots=download_roots,
            dataset_root=dataset_path,
            kind="raw",
            interval_seconds=float(watch_interval_seconds),
            max_iterations=int(watch_iterations),
            dry_run=bool(dry_run),
            availability_output_dir=output_path / "availability",
        )
        watch_report = str(write_aodraw_download_watch(watch_summary, output_path / "watch"))
    else:
        watch_summary = {"status": "skipped", "evaluation_ready": False, "missing_file_count": None}

    status = _status(cleanup=cleanup, download_open=download_open, watch_summary=watch_summary, watch_requested=bool(watch))
    return {
        "name": "AODRaw acquisition runner",
        "status": status,
        "project_root": str(project_path),
        "output_dir": str(output_path),
        "dataset_root": str(dataset_path),
        "download_roots": [str(Path(item).expanduser()) for item in download_roots],
        "manifest_source": str(manifest_path) if not isinstance(manifest_path, Sequence) or isinstance(manifest_path, (str, bytes, Path)) else "in_memory",
        "initial_plan": initial_plan,
        "post_cleanup_plan": post_cleanup_plan,
        "cleanup": cleanup,
        "download_open": download_open,
        "watch": watch_summary,
        "reports": {
            "initial_plan": str(plan_report),
            "cleanup": str(cleanup_report),
            "post_cleanup_plan": str(post_plan_report),
            "watch": watch_report,
        },
        "next_action": _next_action(status, post_cleanup_plan=post_cleanup_plan, download_open=download_open, watch_summary=watch_summary),
        "claim_boundary": (
            "This runner only handles local acquisition plumbing. It is not PerceptionISP performance evidence until RAW files download and evaluation passes."
        ),
    }


def write_aodraw_acquisition(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _resolve_project_path(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _open_download_link(*, enabled: bool, dry_run: bool, disk_ready: bool, opener: Any = None) -> Dict[str, Any]:
    url = AODRAW_OFFICIAL_URLS["downsampled_raw_baidu"]
    command = ["open", "-a", "Safari", url]
    if not enabled:
        return {"status": "skipped", "url": url, "command": " ".join(command), "message": "Pass --open-download after disk headroom is available."}
    if not disk_ready:
        return {"status": "blocked_disk", "url": url, "command": " ".join(command), "message": "Disk headroom check failed after cleanup."}
    if dry_run:
        return {"status": "dry_run", "url": url, "command": " ".join(command), "message": "Dry-run only. Safari was not opened."}
    open_func = opener or _default_opener
    result = open_func(command)
    return {
        "status": "opened" if int(result.get("returncode", 1)) == 0 else "open_failed",
        "url": url,
        "command": " ".join(command),
        "returncode": int(result.get("returncode", 1)),
        "stdout": str(result.get("stdout", "")),
        "stderr": str(result.get("stderr", "")),
        "message": "Safari open command completed." if int(result.get("returncode", 1)) == 0 else "Safari open command failed.",
    }


def _default_opener(command: Sequence[str]) -> Dict[str, Any]:
    completed = subprocess.run(list(command), capture_output=True, text=True, check=False)
    return {"returncode": int(completed.returncode), "stdout": completed.stdout, "stderr": completed.stderr}


def _status(*, cleanup: Mapping[str, Any], download_open: Mapping[str, Any], watch_summary: Mapping[str, Any], watch_requested: bool) -> str:
    cleanup_status = str(cleanup.get("status", ""))
    open_status = str(download_open.get("status", ""))
    watch_status = str(watch_summary.get("status", ""))
    if cleanup_status == "blocked_confirmation_required":
        return "blocked_cleanup_confirmation"
    if open_status == "blocked_disk":
        return "blocked_disk"
    if watch_requested and bool(watch_summary.get("evaluation_ready", False)):
        return "ready_for_evaluation"
    if watch_requested and watch_status in {"waiting_for_files", "dry_run"}:
        return "waiting_for_download"
    if open_status in {"opened", "dry_run"}:
        return "download_link_opened"
    if cleanup_status == "executed":
        return "cleanup_executed"
    return "planned"


def _next_action(
    status: str,
    *,
    post_cleanup_plan: Mapping[str, Any],
    download_open: Mapping[str, Any],
    watch_summary: Mapping[str, Any],
) -> str:
    if status == "blocked_cleanup_confirmation":
        return f"Rerun with --confirm-token {CLEANUP_CONFIRM_TOKEN} only after accepting selected cleanup deletion."
    if status == "blocked_disk":
        return "Free more disk space before opening the AODRaw RAW Baidu link."
    if status == "ready_for_evaluation":
        return str(watch_summary.get("evaluation_command", "Run the RAW-only AODRaw pipeline."))
    if status == "waiting_for_download":
        return "Finish the Baidu/Safari RAW zip download, then rerun the watcher or acquisition runner with --watch."
    if status == "download_link_opened":
        return "Use Safari/Baidu to start or monitor the RAW zip download, then run the watcher."
    if bool(post_cleanup_plan.get("disk", {}).get("fits_downsampled_raw_test_zip_with_headroom", False)):
        return "Run with --open-download to open the official AODRaw RAW Baidu link."
    return "Run with --execute-cleanup and the confirmation token after accepting cleanup."


def _render_html(summary: Mapping[str, Any]) -> str:
    reports = summary.get("reports", {}) if isinstance(summary.get("reports"), Mapping) else {}
    report_links = "".join(
        f"<li><code>{html_lib.escape(str(key))}</code>: <a href=\"{html_lib.escape(_relative_report(summary, value))}\">{html_lib.escape(str(value))}</a></li>"
        for key, value in reports.items()
        if value
    )
    cleanup = summary.get("cleanup", {}) if isinstance(summary.get("cleanup"), Mapping) else {}
    download_open = summary.get("download_open", {}) if isinstance(summary.get("download_open"), Mapping) else {}
    watch = summary.get("watch", {}) if isinstance(summary.get("watch"), Mapping) else {}
    post_plan = summary.get("post_cleanup_plan", {}) if isinstance(summary.get("post_cleanup_plan"), Mapping) else {}
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Acquisition Runner</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    pre {{ background: #111827; color: #f9fafb; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>AODRaw Acquisition Runner</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>.</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <table><tbody>
    <tr><th>Cleanup</th><td><code>{html_lib.escape(str(cleanup.get('status', '')))}</code></td></tr>
    <tr><th>Download Open</th><td><code>{html_lib.escape(str(download_open.get('status', '')))}</code> {html_lib.escape(str(download_open.get('message', '')))}</td></tr>
    <tr><th>Watch</th><td><code>{html_lib.escape(str(watch.get('status', '')))}</code>; missing={html_lib.escape(str(watch.get('missing_file_count', '')))}</td></tr>
    <tr><th>Post-cleanup Disk</th><td><code>{html_lib.escape(str(post_plan.get('disk', {}).get('available_gib', '')))} GiB</code></td></tr>
  </tbody></table>
  <h2>Reports</h2>
  <ul>{report_links}</ul>
  <h2>Download Command</h2>
  <pre>{html_lib.escape(str(download_open.get('command', '')))}</pre>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code>; nested plan JSON: <code>initial_plan/{PLAN_SUMMARY_FILENAME}</code>; cleanup JSON: <code>cleanup/{CLEANUP_SUMMARY_FILENAME}</code>; watch JSON: <code>watch/{WATCH_SUMMARY_FILENAME}</code>.</p>
</body>
</html>
"""


def _relative_report(summary: Mapping[str, Any], value: Any) -> str:
    try:
        base = Path(str(summary.get("output_dir", "")))
        if str(base) and base.exists():
            return str(Path(str(value)).resolve().relative_to(base.resolve()))
    except Exception:
        pass
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
