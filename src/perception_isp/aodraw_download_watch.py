"""Watch manually downloaded AODRaw files and keep import/readiness reports current."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .aodraw_download_importer import import_aodraw_downloads
from .aodraw_image_availability import build_aodraw_image_availability, write_aodraw_image_availability
from .types import json_ready


SUMMARY_FILENAME = "aodraw_download_watch_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Watch AODRaw downloads, import found files, and update availability reports.")
    parser.add_argument("--manifest", default="reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json")
    parser.add_argument("--download-root", action="append", default=[], help="Directory or zip file to scan. Repeatable. Defaults to ~/Downloads.")
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw")
    parser.add_argument("--kind", choices=["all", "raw", "srgb"], default="all")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=1, help="Use 0 to watch until ready.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_download_watch_v1")
    parser.add_argument("--availability-output-dir", default="reports/perception_aodraw_image_availability_test_downsample_adverse_24_v1")
    args = parser.parse_args(argv)

    summary = watch_aodraw_downloads(
        manifest=args.manifest,
        download_roots=tuple(args.download_root) if args.download_root else (Path.home() / "Downloads",),
        dataset_root=args.dataset_root,
        kind=str(args.kind),
        interval_seconds=float(args.interval),
        max_iterations=int(args.max_iterations),
        dry_run=bool(args.dry_run),
        availability_output_dir=args.availability_output_dir,
    )
    html_path = write_aodraw_download_watch(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "dry_run": summary["dry_run"],
                    "iteration_count": summary["iteration_count"],
                    "evaluation_ready": summary["evaluation_ready"],
                    "missing_file_count": summary["missing_file_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def watch_aodraw_downloads(
    *,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    download_roots: Sequence[str | Path],
    dataset_root: str | Path,
    kind: str = "all",
    interval_seconds: float = 30.0,
    max_iterations: int = 1,
    dry_run: bool = False,
    availability_output_dir: str | Path | None = None,
    sleep_func: Any = time.sleep,
) -> Dict[str, Any]:
    iterations = []
    start = time.time()
    count = 0
    limit = max(int(max_iterations), 0)
    while True:
        count += 1
        import_summary = import_aodraw_downloads(
            manifest=manifest,
            download_roots=download_roots,
            dataset_root=dataset_root,
            kind=kind,
            dry_run=dry_run,
        )
        availability = build_aodraw_image_availability(manifest, dataset_root=dataset_root, kind=kind)
        if availability_output_dir is not None:
            write_aodraw_image_availability(availability, availability_output_dir)
        iterations.append(
            {
                "iteration": count,
                "timestamp_epoch": time.time(),
                "import_status": import_summary["status"],
                "imported_file_count": int(import_summary.get("imported_file_count", 0)),
                "would_import_file_count": int(import_summary.get("would_import_file_count", 0)),
                "resolved_file_count": int(import_summary.get("resolved_file_count", 0)),
                "import_missing_file_count": int(import_summary.get("missing_file_count", 0)),
                "availability_status": availability["status"],
                "evaluation_ready": bool(availability["evaluation_ready"]),
                "availability_missing_file_count": int(availability["missing_file_count"]),
            }
        )
        if bool(availability["evaluation_ready"]):
            break
        if limit and count >= limit:
            break
        sleep_func(max(float(interval_seconds), 0.0))

    last = iterations[-1] if iterations else {}
    ready = bool(last.get("evaluation_ready", False))
    return {
        "name": "AODRaw download watch",
        "status": "ready" if ready else ("dry_run" if dry_run else "waiting_for_files"),
        "dry_run": bool(dry_run),
        "dataset_root": str(Path(dataset_root).expanduser()),
        "download_roots": [str(Path(item).expanduser()) for item in download_roots],
        "manifest_source": str(manifest) if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, Path)) else "in_memory",
        "kind": str(kind),
        "interval_seconds": float(interval_seconds),
        "max_iterations": int(max_iterations),
        "iteration_count": len(iterations),
        "elapsed_seconds": round(time.time() - start, 3),
        "evaluation_ready": ready,
        "missing_file_count": int(last.get("availability_missing_file_count", 0)),
        "iterations": iterations,
        "next_action": _next_action(ready=ready, dry_run=bool(dry_run)),
        "availability_output_dir": "" if availability_output_dir is None else str(availability_output_dir),
        "evaluation_command": (
            "PYTHONPATH=src python3 -m perception_isp.eval_cli "
            "--source aodraw-dataset "
            "--dataset data/raw_datasets/aodraw "
            "--aodraw-manifest reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json "
            "--count 24 --width 768 --height 512 --aodraw-cfa RGGB "
            "--rgb-detector yolo --rgb-detector-model yolo11n.pt --label-aware "
            "--ground-truth-label-map aodraw-coco --ground-truth-label-keep aodraw-coco-overlap "
            "--output-dir reports/perception_aodraw_compare_test_downsample_adverse_24_v1"
        ),
        "claim_boundary": (
            "This watcher only automates local file import and availability checks. It does not download Baidu/TeraBox files or prove PerceptionISP performance."
        ),
    }


def write_aodraw_download_watch(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _next_action(*, ready: bool, dry_run: bool) -> str:
    if ready:
        return "AODRaw subset files are available. Run the evaluation command."
    if dry_run:
        return "Dry-run only. Download files or rerun without --dry-run to import resolved files."
    return "Keep the watcher running while Baidu/TeraBox downloads finish, or rerun after files land in the download directory."


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = "".join(_iteration_row(row) for row in summary.get("iterations", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Download Watch</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .ready, .pass {{ color: #047857; font-weight: 650; }}
    .waiting_for_files, .blocked {{ color: #b91c1c; font-weight: 650; }}
    .dry_run {{ color: #b45309; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    pre {{ background: #111827; color: #f9fafb; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>AODRaw Download Watch</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; evaluation_ready=<code>{html_lib.escape(str(summary.get('evaluation_ready', '')))}</code>; missing={int(summary.get('missing_file_count', 0))}.</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <h2>Iterations</h2>
  <table><thead><tr><th>#</th><th>Import</th><th>Imported</th><th>Resolved</th><th>Import Missing</th><th>Availability</th><th>Eval Ready</th><th>Availability Missing</th></tr></thead><tbody>{rows}</tbody></table>
  <h2>Evaluation Command</h2>
  <pre>{html_lib.escape(str(summary.get('evaluation_command', '')))}</pre>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _iteration_row(row: Mapping[str, Any]) -> str:
    import_status = html_lib.escape(str(row.get("import_status", "")))
    availability_status = html_lib.escape(str(row.get("availability_status", "")))
    return (
        "<tr>"
        f"<td>{int(row.get('iteration', 0))}</td>"
        f"<td class=\"{import_status}\">{import_status}</td>"
        f"<td>{int(row.get('imported_file_count', 0))}</td>"
        f"<td>{int(row.get('resolved_file_count', 0))}</td>"
        f"<td>{int(row.get('import_missing_file_count', 0))}</td>"
        f"<td class=\"{availability_status}\">{availability_status}</td>"
        f"<td>{html_lib.escape(str(bool(row.get('evaluation_ready', False))))}</td>"
        f"<td>{int(row.get('availability_missing_file_count', 0))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
