"""Build an audit report for scene-truth segmentation counterexamples."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "scene_truth_counterexample_audit_summary.json"
TRAIN_SUMMARY_FILENAME = "scene_truth_segmentation_train_summary.json"
RUNS = ("human_rgb", "perception_rgb", "perception_rgb_aux")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit scene-truth segmentation counterexamples.")
    parser.add_argument("--run", action="append", required=True, help="Run spec formatted as label|summary_or_report_dir.")
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_counterexample_audit_v1")
    args = parser.parse_args(argv)
    summary = build_audit([str(value) for value in args.run])
    html_path = write_audit(summary, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "summary_json": str(html_path.parent / SUMMARY_FILENAME), "status": summary["status"]}), indent=2))
    return 0


def build_audit(specs: Sequence[str]) -> dict[str, Any]:
    runs = [_load_run(spec) for spec in specs]
    counterexamples = [
        row
        for row in runs
        if (row.get("delta_mask_iou_mean") is not None and float(row["delta_mask_iou_mean"]) <= 0.0)
        or (row.get("delta_boundary_f1_mean") is not None and float(row["delta_boundary_f1_mean"]) <= 0.0)
    ]
    seed71 = next((row for row in runs if "seed71" in row["label"] and "diag" in row["label"]), None)
    diagnosis = _diagnose_seed71(seed71) if seed71 else "No diagnostic seed71 run was provided."
    return {
        "name": "Scene-truth segmentation counterexample audit",
        "status": "needs_action" if counterexamples else "pass",
        "purpose": (
            "Separate RGB-human annotation bias from scene-truth failures, and identify whether a failing RGB+Aux run is a threshold "
            "calibration issue or a real segmentation-head counterexample."
        ),
        "claim_boundary": (
            "RGB-human segmentation labels remain useful as compatibility/debug gates only. Strong PerceptionISP claims should use "
            "pre-sensor scene truth, native RAW with independent labels, or task labels not inherited from RGB-only renderings."
        ),
        "runs": runs,
        "counterexample_count": len(counterexamples),
        "diagnosis": diagnosis,
        "recommended_actions": [
            "Keep direct low-light/noise seed71 as an explicit counterexample; do not claim direct low-light seed-stability yet.",
            "Use CameraE2E low-light/noise and high-PSF scene-truth gates as the current positive evidence.",
            "Add RGB+Aux loss regularization or distillation so aux channels improve boundary localization without over-narrowing object masks.",
            "Promote RGB-human GT evaluations to compatibility checks, not final PerceptionISP proof.",
            "Scale the scene-truth segmentation gate with more high-information scenes and validation seeds before making broad claims.",
        ],
    }


def write_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(json_ready(summary)))
    (destination / SUMMARY_FILENAME).write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(serializable), encoding="utf-8")
    return html_path


def _load_run(spec: str) -> dict[str, Any]:
    label, raw_path = _split_spec(spec)
    path = Path(raw_path).expanduser()
    summary_path = path / TRAIN_SUMMARY_FILENAME if path.is_dir() else path
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = data.get("comparison", {}).get("perception_rgb_aux_minus_perception_rgb", {})
    run_payloads = data.get("runs", {})
    row: dict[str, Any] = {
        "label": label,
        "status": data.get("status"),
        "path": str(summary_path),
        "report": str(summary_path.parent / "index.html"),
        "delta_mask_iou_mean": comparison.get("delta_mask_iou_mean"),
        "delta_boundary_f1_mean": comparison.get("delta_boundary_f1_mean"),
        "thresholds": {name: run_payloads.get(name, {}).get("threshold") for name in RUNS},
        "case_metrics": {name: run_payloads.get(name, {}).get("val", {}).get("case_aggregate", {}) for name in RUNS},
        "oracles": {
            name: {
                "mask_iou": run_payloads.get(name, {}).get("val", {}).get("oracle_by_mask_iou"),
                "boundary_f1": run_payloads.get(name, {}).get("val", {}).get("oracle_by_boundary_f1"),
            }
            for name in RUNS
        },
    }
    return row


def _diagnose_seed71(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "No seed71 diagnostic payload is available."
    metrics = row.get("case_metrics", {})
    rgb = metrics.get("perception_rgb", {})
    aux = metrics.get("perception_rgb_aux", {})
    aux_oracle = row.get("oracles", {}).get("perception_rgb_aux", {})
    oracle_iou = aux_oracle.get("mask_iou") or {}
    oracle_boundary = aux_oracle.get("boundary_f1") or {}
    rgb_iou = _num(rgb.get("mask_iou_mean"))
    rgb_boundary = _num(rgb.get("boundary_f1_mean"))
    aux_iou = _num(aux.get("mask_iou_mean"))
    aux_boundary = _num(aux.get("boundary_f1_mean"))
    aux_oracle_iou = _num(oracle_iou.get("mask_iou_mean"))
    aux_oracle_boundary = _num(oracle_boundary.get("boundary_f1_mean"))
    score_sep = _num(aux.get("score_boundary_separation_mean"))
    rgb_score_sep = _num(rgb.get("score_boundary_separation_mean"))
    if aux_oracle_iou is not None and rgb_iou is not None and aux_oracle_iou < rgb_iou:
        return (
            "Seed71 is not solved by threshold calibration. RGB+Aux has stronger boundary score separation "
            f"({_fmt(score_sep)} vs RGB-only {_fmt(rgb_score_sep)}), but its best validation mask IoU remains "
            f"{_fmt(aux_oracle_iou)} below RGB-only {_fmt(rgb_iou)}, and best validation boundary F1 remains "
            f"{_fmt(aux_oracle_boundary)} below RGB-only {_fmt(rgb_boundary)}. Treat this as a true segmentation-head "
            "counterexample: the aux signal exists, but the compact model is converting it into worse masks."
        )
    return (
        "Seed71 may be threshold-sensitive. The diagnostic run did not show a clear oracle gap against RGB-only, so this needs "
        "a larger validation split or saved score-map analysis."
    )


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for row in summary.get("runs", []):
        thresholds = row.get("thresholds", {})
        aux_metrics = row.get("case_metrics", {}).get("perception_rgb_aux", {})
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('label', '')))}</td>"
            f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
            f"<td>{_fmt(row.get('delta_mask_iou_mean'), signed=True)}</td>"
            f"<td>{_fmt(row.get('delta_boundary_f1_mean'), signed=True)}</td>"
            f"<td>{_fmt(thresholds.get('perception_rgb'))}</td>"
            f"<td>{_fmt(thresholds.get('perception_rgb_aux'))}</td>"
            f"<td>{_fmt(aux_metrics.get('score_boundary_separation_mean'), signed=True)}</td>"
            f"<td><a href='{html_lib.escape(str(row.get('report', '')))}'>report</a></td>"
            "</tr>"
        )
    actions = "".join(f"<li>{html_lib.escape(str(action))}</li>" for action in summary.get("recommended_actions", []))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Counterexample Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    .note {{ border: 1px solid #f59e0b; background: #fffbeb; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .tile {{ border: 1px solid #d8dee4; background: #f8fafc; border-radius: 6px; padding: 10px 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 18px 0 28px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    li {{ margin: 7px 0; }}
  </style>
</head>
<body>
  <h1>Scene-truth Counterexample Audit</h1>
  <div class="note">{html_lib.escape(str(summary.get('purpose', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <div class="grid">
    <div class="tile"><b>Status</b><br>{html_lib.escape(str(summary.get('status', '')))}</div>
    <div class="tile"><b>Runs</b><br>{len(summary.get('runs', []))}</div>
    <div class="tile"><b>Counterexamples</b><br>{int(summary.get('counterexample_count', 0))}</div>
  </div>
  <h2>Run Comparison</h2>
  <table><thead><tr><th>Run</th><th>Status</th><th>dIoU</th><th>dBoundary</th><th>RGB Thr</th><th>RGB+Aux Thr</th><th>RGB+Aux Score Sep</th><th>Link</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>Diagnosis</h2>
  <p>{html_lib.escape(str(summary.get('diagnosis', '')))}</p>
  <h2>Recommended Actions</h2>
  <ul>{actions}</ul>
</body>
</html>
"""


def _split_spec(spec: str) -> tuple[str, str]:
    if "|" not in spec:
        path = Path(spec).expanduser()
        return path.parent.name if path.name == TRAIN_SUMMARY_FILENAME else path.name, str(path)
    label, path = spec.split("|", 1)
    return label.strip(), path.strip()


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Any, *, signed: bool = False) -> str:
    number = _num(value)
    if number is None:
        return "-"
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
