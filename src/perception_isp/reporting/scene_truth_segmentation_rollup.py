"""Roll up scene-truth segmentation training runs across seeds."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "scene_truth_segmentation_rollup_summary.json"
TARGET_COMPARISON = "perception_rgb_aux_minus_perception_rgb"
METRICS = ("delta_mask_iou_mean", "delta_boundary_f1_mean")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Roll up scene-truth segmentation training summaries.")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Group spec formatted as name|summary_or_report_dir. Repeat for each run.",
    )
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_segmentation_rollup_v1")
    args = parser.parse_args(argv)
    summary = build_rollup([str(value) for value in args.group])
    html_path = write_rollup(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "groups": summary["groups"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_rollup(group_specs: Sequence[str]) -> Dict[str, Any]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for spec in group_specs:
        name, path = _parse_group_spec(spec)
        grouped.setdefault(name, []).append(_load_run(path))
    groups = {name: _summarize_group(name, rows) for name, rows in grouped.items()}
    checks = _checks(groups)
    return {
        "name": "Scene-truth segmentation multi-seed rollup",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "checks": checks,
        "groups": groups,
        "interpretation": (
            "This rollup measures whether compact RGB+Aux segmentation improvements over Perception RGB-only remain stable across random seeds."
        ),
        "claim_boundary": (
            "The rollup is still a compact synthetic scene-truth gate. It is stronger than a single-seed smoke test, but not a replacement for a large real RAW segmentation benchmark."
        ),
    }


def write_rollup(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return html_path


def _parse_group_spec(spec: str) -> tuple[str, Path]:
    parts = spec.split("|", 1)
    if len(parts) != 2:
        raise ValueError(f"group spec must be name|path, got {spec!r}")
    return parts[0], Path(parts[1]).expanduser()


def _load_run(path: Path) -> Dict[str, Any]:
    summary_path = path
    if path.is_dir():
        summary_path = path / "scene_truth_segmentation_train_summary.json"
    data = json.loads(summary_path.read_text())
    comparison = data.get("comparison", {}).get(TARGET_COMPARISON, {})
    val = data.get("comparison", {}).get("val_metrics", {})
    seed = data.get("training", {}).get("seed")
    return {
        "path": str(summary_path),
        "report": str(summary_path.parent / "index.html"),
        "status": data.get("status"),
        "seed": seed,
        "train_case_count": data.get("train_case_count"),
        "val_case_count": data.get("val_case_count"),
        "comparison": {metric: _optional_float(comparison.get(metric)) for metric in METRICS},
        "val_metrics": val,
    }


def _summarize_group(name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_summary = {}
    for metric in METRICS:
        values = [_optional_float(row.get("comparison", {}).get(metric)) for row in rows]
        numeric = [float(value) for value in values if value is not None]
        metric_summary[metric] = {
            "mean": _mean(numeric),
            "std": _std(numeric),
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "positive_count": sum(1 for value in numeric if value > 0.0),
            "positive_rate": float(np.mean([value > 0.0 for value in numeric])) if numeric else None,
            "values": numeric,
        }
    pass_all = all(row.get("status") == "pass" for row in rows)
    positive_all = all((metric_summary[metric]["positive_rate"] or 0.0) == 1.0 for metric in METRICS)
    return {
        "name": name,
        "status": "pass" if pass_all and positive_all else "fail",
        "run_count": len(rows),
        "seeds": [row.get("seed") for row in rows],
        "all_runs_pass": bool(pass_all),
        "all_target_deltas_positive": bool(positive_all),
        "metrics": metric_summary,
        "runs": list(rows),
    }


def _checks(groups: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {
            "id": "groups_present",
            "status": "pass" if groups else "fail",
            "description": "At least one rollup group is present.",
            "criteria": [{"metric": "group_count", "value": len(groups), "threshold": 1, "pass": bool(groups)}],
        },
        {
            "id": "all_groups_have_positive_rgb_aux_deltas",
            "status": "pass" if groups and all(group.get("all_target_deltas_positive") for group in groups.values()) else "fail",
            "description": "Every group should have positive RGB+Aux minus Perception RGB deltas for mask IoU and boundary F1 in every seed.",
            "criteria": [
                {
                    "metric": f"{name}_all_target_deltas_positive",
                    "value": bool(group.get("all_target_deltas_positive")),
                    "pass": bool(group.get("all_target_deltas_positive")),
                }
                for name, group in groups.items()
            ],
        },
    ]


def _render_html(summary: Mapping[str, Any]) -> str:
    group_rows = []
    run_rows = []
    for group_name, group in summary.get("groups", {}).items():
        mask = group.get("metrics", {}).get("delta_mask_iou_mean", {})
        boundary = group.get("metrics", {}).get("delta_boundary_f1_mean", {})
        group_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(group_name))}</td>"
            f"<td>{int(group.get('run_count', 0))}</td>"
            f"<td>{html_lib.escape(', '.join(str(seed) for seed in group.get('seeds', [])))}</td>"
            f"<td>{_fmt(mask.get('mean'), signed=True)}</td>"
            f"<td>{_fmt(mask.get('std'))}</td>"
            f"<td>{_fmt(mask.get('positive_rate'))}</td>"
            f"<td>{_fmt(boundary.get('mean'), signed=True)}</td>"
            f"<td>{_fmt(boundary.get('std'))}</td>"
            f"<td>{_fmt(boundary.get('positive_rate'))}</td>"
            "</tr>"
        )
        for row in group.get("runs", []):
            comp = row.get("comparison", {})
            run_rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(group_name))}</td>"
                f"<td>{html_lib.escape(str(row.get('seed')))}</td>"
                f"<td>{html_lib.escape(str(row.get('status')))}</td>"
                f"<td>{_fmt(comp.get('delta_mask_iou_mean'), signed=True)}</td>"
                f"<td>{_fmt(comp.get('delta_boundary_f1_mean'), signed=True)}</td>"
                f"<td><a href='{html_lib.escape(str(row.get('report')))}'>report</a></td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Segmentation Multi-seed Rollup</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #16202a; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Scene-truth Segmentation Multi-seed Rollup</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <b>{html_lib.escape(str(summary.get('status', '')))}</b></p>
  <h2>Group Summary</h2>
  <table><thead><tr><th>Group</th><th>Runs</th><th>Seeds</th><th>dIoU mean</th><th>dIoU std</th><th>dIoU positive</th><th>dBoundary mean</th><th>dBoundary std</th><th>dBoundary positive</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table>
  <h2>Runs</h2>
  <table><thead><tr><th>Group</th><th>Seed</th><th>Status</th><th>dIoU</th><th>dBoundary</th><th>Link</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table>
</body>
</html>
"""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=0)) if values else None


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return html_lib.escape(str(value))
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
