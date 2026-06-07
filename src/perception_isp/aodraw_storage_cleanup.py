"""Dry-run-first storage cleanup helper for AODRaw acquisition."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .aodraw_download_plan import _cleanup_candidates, _cleanup_summary, _disk_summary
from .types import json_ready


SUMMARY_FILENAME = "aodraw_storage_cleanup_summary.json"
CONFIRM_TOKEN = "DELETE_AODRAW_CLEANUP_CANDIDATES"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute cleanup to make room for AODRaw RAW downloads.")
    parser.add_argument("--project-root", default=".", help="PerceptionISP project root containing cleanup candidate paths.")
    parser.add_argument("--dataset-root", default="data/raw_datasets/aodraw", help="Dataset root used for free-space probing.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate path or label to select. Defaults to the first sufficient candidate.")
    parser.add_argument("--execute", action="store_true", help="Delete selected candidates. Requires --confirm-token.")
    parser.add_argument("--confirm-token", default="", help=f"Required exact token for --execute: {CONFIRM_TOKEN}")
    parser.add_argument("--output-dir", default="reports/perception_aodraw_storage_cleanup_v1")
    args = parser.parse_args(argv)

    summary = build_aodraw_storage_cleanup(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        requested_candidates=tuple(args.candidate),
        execute=bool(args.execute),
        confirm_token=str(args.confirm_token),
    )
    html_path = write_aodraw_storage_cleanup(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "execute": summary["execute"],
                    "selected_candidate_count": summary["selected_candidate_count"],
                    "selected_total_gib": summary["selected_total_gib"],
                    "additional_free_gib_needed": summary["cleanup"]["additional_free_gib_needed"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_aodraw_storage_cleanup(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    requested_candidates: Sequence[str] = (),
    execute: bool = False,
    confirm_token: str = "",
) -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    dataset_path = Path(dataset_root).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    disk = _disk_summary(dataset_path)
    candidates = _cleanup_candidates(root)
    cleanup = _cleanup_summary(disk, candidates)
    selected = _select_candidates(candidates, requested_candidates=requested_candidates, default_path=str(cleanup.get("first_sufficient_candidate", "")))
    confirmed = bool(execute) and str(confirm_token) == CONFIRM_TOKEN
    actions = []
    for row in selected:
        actions.append(_cleanup_action(root, row, execute=bool(execute), confirmed=confirmed))
    selected_total_bytes = sum(int(row.get("size_bytes", 0)) for row in selected)
    status = _status(execute=bool(execute), confirmed=confirmed, selected=selected, actions=actions)
    return {
        "name": "AODRaw storage cleanup",
        "status": status,
        "execute": bool(execute),
        "confirmed": confirmed,
        "confirm_token_required": CONFIRM_TOKEN,
        "project_root": str(root),
        "dataset_root": str(dataset_path),
        "disk": disk,
        "cleanup": cleanup,
        "available_candidates": candidates,
        "requested_candidates": [str(item) for item in requested_candidates],
        "selected_candidates": selected,
        "selected_candidate_count": len(selected),
        "selected_total_gib": _bytes_to_gib(selected_total_bytes),
        "actions": actions,
        "execute_command": _execute_command(selected),
        "next_action": _next_action(status),
        "claim_boundary": (
            "This cleanup helper is only for freeing local disk before AODRaw acquisition. "
            "It does not download data or prove PerceptionISP performance."
        ),
    }


def write_aodraw_storage_cleanup(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    requested_candidates: Sequence[str],
    default_path: str,
) -> list[Dict[str, Any]]:
    requested = [str(item).strip() for item in requested_candidates if str(item).strip()]
    if not requested and default_path:
        requested = [default_path]
    selected = []
    seen = set()
    for needle in requested:
        for row in candidates:
            if _candidate_matches(row, needle) and str(row.get("path", "")) not in seen:
                selected.append(dict(row))
                seen.add(str(row.get("path", "")))
    return selected


def _candidate_matches(row: Mapping[str, Any], needle: str) -> bool:
    normalized = needle.strip()
    return normalized in {
        str(row.get("path", "")),
        str(row.get("absolute_path", "")),
        str(row.get("label", "")),
    }


def _cleanup_action(root: Path, row: Mapping[str, Any], *, execute: bool, confirmed: bool) -> Dict[str, Any]:
    relative = Path(str(row.get("path", "")))
    path = (root / relative).resolve()
    if not _is_relative_to(path, root):
        return _action(row, "blocked_outside_project", "Refusing to delete a path outside project_root.")
    if not path.exists():
        return _action(row, "missing", "Candidate path is already absent.")
    if not execute:
        return _action(row, "would_delete", "Dry-run only. No files were deleted.")
    if not confirmed:
        return _action(row, "blocked_confirmation_required", f"Pass --confirm-token {CONFIRM_TOKEN} to execute deletion.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return _action(row, "deleted", "Candidate was deleted.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _action(row: Mapping[str, Any], status: str, message: str) -> Dict[str, Any]:
    return {
        "path": str(row.get("path", "")),
        "size_gib": row.get("size_gib", 0.0),
        "status": status,
        "message": message,
    }


def _status(*, execute: bool, confirmed: bool, selected: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]]) -> str:
    if not selected:
        return "no_candidate_selected"
    if execute and not confirmed:
        return "blocked_confirmation_required"
    if execute:
        return "executed" if all(str(row.get("status")) in {"deleted", "missing"} for row in actions) else "partial"
    return "dry_run"


def _execute_command(selected: Sequence[Mapping[str, Any]]) -> str:
    candidates = " ".join(f"--candidate {json.dumps(str(row.get('path', '')))}" for row in selected)
    return (
        "PYTHONPATH=src python3 -m perception_isp.aodraw_storage_cleanup "
        f"{candidates} --execute --confirm-token {CONFIRM_TOKEN} "
        "--output-dir reports/perception_aodraw_storage_cleanup_execute_v1"
    ).replace("  ", " ").strip()


def _next_action(status: str) -> str:
    if status == "dry_run":
        return "Review selected candidates. Run the execute command only after explicitly accepting deletion."
    if status == "executed":
        return "Rerun the AODRaw download plan and then start the AODRaw test RAW zip download."
    if status == "blocked_confirmation_required":
        return "Deletion was not run because the confirmation token was missing or wrong."
    return "Select a cleanup candidate or free disk space manually."


def _bytes_to_gib(value: int) -> float:
    return round(float(value) / float(1024**3), 2)


def _render_html(summary: Mapping[str, Any]) -> str:
    available_rows = "".join(_candidate_row(row) for row in summary.get("available_candidates", ()) if isinstance(row, Mapping))
    action_rows = "".join(_action_row(row) for row in summary.get("actions", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AODRaw Storage Cleanup</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #b45309; background: #fff7ed; padding: 12px 14px; margin: 16px 0; }}
    .dry_run, .would_delete {{ color: #b45309; font-weight: 650; }}
    .executed, .deleted {{ color: #047857; font-weight: 650; }}
    .blocked_confirmation_required, .partial, .blocked_outside_project {{ color: #b91c1c; font-weight: 650; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    pre {{ background: #111827; color: #f9fafb; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>AODRaw Storage Cleanup</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; execute=<code>{html_lib.escape(str(summary.get('execute', '')))}</code>; confirmed=<code>{html_lib.escape(str(summary.get('confirmed', '')))}</code>.</p>
  <p>Additional free space needed: <code>{html_lib.escape(str(summary.get('cleanup', {}).get('additional_free_gib_needed', '')))} GiB</code>; selected total: <code>{html_lib.escape(str(summary.get('selected_total_gib', '')))} GiB</code>.</p>
  <p>{html_lib.escape(str(summary.get('next_action', '')))}</p>
  <h2>Selected Actions</h2>
  <table><thead><tr><th>Path</th><th>Size GiB</th><th>Status</th><th>Message</th></tr></thead><tbody>{action_rows}</tbody></table>
  <h2>Execute Command</h2>
  <pre>{html_lib.escape(str(summary.get('execute_command', '')))}</pre>
  <h2>Available Candidates</h2>
  <table><thead><tr><th>Path</th><th>Size GiB</th><th>Risk</th><th>Why</th><th>Verification</th></tr></thead><tbody>{available_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _candidate_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('path', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('size_gib', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('risk', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('why', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('verification', '')))}</td>"
        "</tr>"
    )


def _action_row(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "")))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('path', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('size_gib', '')))}</td>"
        f"<td class=\"{status}\">{status}</td>"
        f"<td>{html_lib.escape(str(row.get('message', '')))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
