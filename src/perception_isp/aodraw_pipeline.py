"""Run the post-download AODRaw import, readiness, and optional evaluation flow."""

from __future__ import annotations

import argparse
import contextlib
import html as html_lib
import io
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .aodraw_download_importer import import_aodraw_downloads, write_aodraw_download_import
from .aodraw_image_availability import build_aodraw_image_availability, write_aodraw_image_availability
from .eval_cli import main as eval_cli_main
from .types import json_ready


SUMMARY_FILENAME = "aodraw_pipeline_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run AODRaw post-download import, availability, and optional evaluation.")
    parser.add_argument("--manifest", default="reports/perception_aodraw_subset_plan_test_downsample_adverse_24_v1/aodraw_subset_manifest.json")
    parser.add_argument("--download-root", action="append", default=[], help="Directory or zip file to scan. Repeatable. Defaults to ~/Downloads.")
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw")
    parser.add_argument("--kind", choices=["all", "raw", "srgb"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Resolve files without importing or running evaluation.")
    parser.add_argument("--skip-eval", action="store_true", help="Stop after import and availability checks.")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_pipeline_v1")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--aodraw-cfa", default="RGGB")
    parser.add_argument("--aodraw-require-srgb", action="store_true")
    parser.add_argument("--rgb-detector", default="yolo")
    parser.add_argument("--rgb-detector-model", default="yolo11n.pt")
    parser.add_argument("--rgb-detector-confidence", type=float, default=0.25)
    parser.add_argument("--label-aware", action="store_true", default=True)
    parser.add_argument("--no-label-aware", action="store_false", dest="label_aware")
    parser.add_argument("--no-visuals", action="store_true")
    args = parser.parse_args(argv)

    summary = run_aodraw_pipeline(
        manifest=args.manifest,
        download_roots=tuple(args.download_root) if args.download_root else (Path.home() / "Downloads",),
        dataset_root=args.dataset_root,
        kind=str(args.kind),
        dry_run=bool(args.dry_run),
        skip_eval=bool(args.skip_eval),
        output_dir=args.output_dir,
        count=int(args.count),
        width=int(args.width),
        height=int(args.height),
        aodraw_cfa=str(args.aodraw_cfa),
        require_srgb=bool(args.aodraw_require_srgb),
        rgb_detector=str(args.rgb_detector),
        rgb_detector_model=str(args.rgb_detector_model),
        rgb_detector_confidence=float(args.rgb_detector_confidence),
        label_aware=bool(args.label_aware),
        no_visuals=bool(args.no_visuals),
    )
    html_path = write_aodraw_pipeline(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "evaluation_ready": summary["evaluation_ready"],
                    "evaluation_status": summary["evaluation_status"],
                    "missing_file_count": summary["availability"]["missing_file_count"],
                    "import_status": summary["import"]["status"],
                }
            ),
            indent=2,
        )
    )
    return 0


def run_aodraw_pipeline(
    *,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    download_roots: Sequence[str | Path],
    dataset_root: str | Path,
    kind: str = "all",
    dry_run: bool = False,
    skip_eval: bool = False,
    output_dir: str | Path,
    count: int = 24,
    width: int = 768,
    height: int = 512,
    aodraw_cfa: str = "RGGB",
    require_srgb: bool = False,
    rgb_detector: str = "yolo",
    rgb_detector_model: str = "yolo11n.pt",
    rgb_detector_confidence: float = 0.25,
    label_aware: bool = True,
    no_visuals: bool = False,
) -> Dict[str, Any]:
    destination = Path(output_dir).expanduser()
    import_dir = destination / "import"
    availability_dir = destination / "availability"
    eval_dir = destination / "evaluation"

    import_summary = import_aodraw_downloads(
        manifest=manifest,
        download_roots=download_roots,
        dataset_root=dataset_root,
        kind=kind,
        dry_run=dry_run,
    )
    import_report = write_aodraw_download_import(import_summary, import_dir)
    availability = build_aodraw_image_availability(manifest, dataset_root=dataset_root)
    availability_report = write_aodraw_image_availability(availability, availability_dir)

    eval_result: Dict[str, Any] = {
        "status": "skipped",
        "exit_code": None,
        "report": "",
        "summary_json": "",
        "stdout_json": {},
        "error": "",
    }
    eval_args = _eval_args(
        manifest=manifest,
        dataset_root=dataset_root,
        output_dir=eval_dir,
        count=count,
        width=width,
        height=height,
        aodraw_cfa=aodraw_cfa,
        require_srgb=require_srgb,
        rgb_detector=rgb_detector,
        rgb_detector_model=rgb_detector_model,
        rgb_detector_confidence=rgb_detector_confidence,
        label_aware=label_aware,
        no_visuals=no_visuals,
    )
    if dry_run:
        eval_result["status"] = "dry_run_skipped"
    elif not bool(availability["evaluation_ready"]):
        eval_result["status"] = "blocked_by_availability"
    elif skip_eval:
        eval_result["status"] = "ready_skipped"
    else:
        eval_result = _run_eval(eval_args, eval_dir)

    status = _status(dry_run=bool(dry_run), availability_ready=bool(availability["evaluation_ready"]), skip_eval=bool(skip_eval), eval_status=str(eval_result["status"]))
    return {
        "name": "AODRaw post-download pipeline",
        "status": status,
        "dry_run": bool(dry_run),
        "skip_eval": bool(skip_eval),
        "dataset_root": str(Path(dataset_root).expanduser()),
        "download_roots": [str(Path(item).expanduser()) for item in download_roots],
        "manifest_source": str(manifest) if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, Path)) else "in_memory",
        "import": import_summary,
        "import_report": str(import_report),
        "availability": availability,
        "availability_report": str(availability_report),
        "evaluation_ready": bool(availability["evaluation_ready"]),
        "evaluation_status": str(eval_result["status"]),
        "evaluation": eval_result,
        "evaluation_args": eval_args,
        "next_action": _next_action(status),
        "claim_boundary": (
            "This pipeline only executes the local post-download workflow. It does not download Baidu/TeraBox files, and a successful smoke evaluation is not a broad PerceptionISP performance claim."
        ),
    }


def write_aodraw_pipeline(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _eval_args(
    *,
    manifest: str | Path | Sequence[Mapping[str, Any]],
    dataset_root: str | Path,
    output_dir: Path,
    count: int,
    width: int,
    height: int,
    aodraw_cfa: str,
    require_srgb: bool,
    rgb_detector: str,
    rgb_detector_model: str,
    rgb_detector_confidence: float,
    label_aware: bool,
    no_visuals: bool,
) -> list[str]:
    args = [
        "--source",
        "aodraw-dataset",
        "--dataset",
        str(dataset_root),
        "--aodraw-manifest",
        str(manifest),
        "--count",
        str(int(count)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--aodraw-cfa",
        str(aodraw_cfa),
        "--rgb-detector",
        str(rgb_detector),
        "--rgb-detector-model",
        str(rgb_detector_model),
        "--rgb-detector-confidence",
        str(float(rgb_detector_confidence)),
        "--ground-truth-label-map",
        "aodraw-coco",
        "--ground-truth-label-keep",
        "aodraw-coco-overlap",
        "--output-dir",
        str(output_dir),
    ]
    if require_srgb:
        args.append("--aodraw-require-srgb")
    if label_aware:
        args.append("--label-aware")
    if no_visuals:
        args.append("--no-visuals")
    return args


def _run_eval(eval_args: Sequence[str], eval_dir: Path) -> Dict[str, Any]:
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exit_code = eval_cli_main(list(eval_args))
        payload = _parse_stdout_json(stdout.getvalue())
        return {
            "status": "pass" if int(exit_code) == 0 else "fail",
            "exit_code": int(exit_code),
            "report": str(payload.get("report", "")),
            "summary_json": str(eval_dir / "comparison_summary.json"),
            "stdout_json": payload,
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "exit_code": None,
            "report": "",
            "summary_json": str(eval_dir / "comparison_summary.json"),
            "stdout_json": _parse_stdout_json(stdout.getvalue()),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_stdout_json(value: str) -> Dict[str, Any]:
    text = str(value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_stdout": text}
    return dict(payload) if isinstance(payload, Mapping) else {"stdout": payload}


def _status(*, dry_run: bool, availability_ready: bool, skip_eval: bool, eval_status: str) -> str:
    if dry_run:
        return "dry_run"
    if not availability_ready:
        return "waiting_for_files"
    if skip_eval:
        return "ready_for_evaluation"
    if eval_status == "pass":
        return "evaluation_pass"
    if eval_status == "error":
        return "evaluation_error"
    return "evaluation_failed"


def _next_action(status: str) -> str:
    if status == "dry_run":
        return "Download files or rerun without --dry-run to import resolved files."
    if status == "waiting_for_files":
        return "Download the missing AODRaw files, then rerun this pipeline."
    if status == "ready_for_evaluation":
        return "Rerun without --skip-eval to execute the AODRaw HumanISP/PerceptionISP evaluation."
    if status == "evaluation_pass":
        return "Review the generated evaluation report before making any claim."
    return "Inspect the evaluation error and fix the data or detector setup."


def _render_html(summary: Mapping[str, Any]) -> str:
    checks = "".join(_check_row(row) for row in summary.get("availability", {}).get("checks", ()) if isinstance(row, Mapping))
    eval_args = " ".join(str(item) for item in summary.get("evaluation_args", ()))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Post-download Pipeline</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .pass, .evaluation_pass, .ready_for_evaluation {{ color: #047857; font-weight: 650; }}
    .waiting_for_files, .blocked, .fail, .evaluation_failed, .evaluation_error {{ color: #b91c1c; font-weight: 650; }}
    .dry_run {{ color: #b45309; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    pre {{ background: #111827; color: #f9fafb; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>AODRaw Post-download Pipeline</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class="{html_lib.escape(str(summary.get('status', '')))}">{html_lib.escape(str(summary.get('status', '')))}</code>; evaluation_ready=<code>{html_lib.escape(str(summary.get('evaluation_ready', '')))}</code>; evaluation_status=<code>{html_lib.escape(str(summary.get('evaluation_status', '')))}</code>.</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <p>Import report: <code>{html_lib.escape(str(summary.get('import_report', '')))}</code></p>
  <p>Availability report: <code>{html_lib.escape(str(summary.get('availability_report', '')))}</code></p>
  <p>Evaluation report: <code>{html_lib.escape(str(summary.get('evaluation', {}).get('report', '')))}</code></p>
  <h2>Availability Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{checks}</tbody></table>
  <h2>Evaluation Args</h2>
  <pre>{html_lib.escape(eval_args)}</pre>
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


if __name__ == "__main__":
    raise SystemExit(main())
