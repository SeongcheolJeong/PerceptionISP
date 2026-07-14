"""Roll up scene-truth segmentation results by CFA, PSF, condition, and shape slices."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .types import json_ready


SUMMARY_FILENAME = "scene_truth_segmentation_slice_rollup_summary.json"
DEFAULT_BASELINE = "perception_rgb"
DEFAULT_CANDIDATE = "perception_rgb_aux"
METRICS = (
    "mask_iou_mean",
    "boundary_f1_mean",
    "score_boundary_separation_mean",
    "local_mask_iou_mean",
    "local_boundary_f1_mean",
    "local_score_boundary_separation_mean",
)
NOMINAL_SATURATION_EPSILON = 0.005
NOMINAL_NO_REGRESSION_EPSILON = 0.005
PRIMARY_DELTA_METRICS = (
    "delta_mask_iou_mean",
    "delta_boundary_f1_mean",
    "delta_local_mask_iou_mean",
    "delta_local_boundary_f1_mean",
)
SLICE_SPECS = {
    "cfa": ("by_cfa", "cfa"),
    "psf": ("by_psf", "psf_sigma"),
    "condition": ("by_condition", "condition"),
    "shape": ("by_shape", "shape_tag"),
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Roll up scene-truth segmentation slice metrics across runs.")
    parser.add_argument("--run", action="append", required=True, help="Run spec formatted as label|summary_or_report_dir.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_segmentation_slice_rollup_v1")
    args = parser.parse_args(argv)
    summary = build_slice_rollup(
        [str(value) for value in args.run],
        baseline=str(args.baseline),
        candidate=str(args.candidate),
    )
    html_path = write_slice_rollup(summary, args.output_dir)
    print(
        json.dumps(
            json_ready({"report": str(html_path), "summary_json": str(html_path.parent / SUMMARY_FILENAME), "status": summary["status"]}),
            indent=2,
        )
    )
    return 0


def build_slice_rollup(specs: Sequence[str], *, baseline: str, candidate: str) -> Dict[str, Any]:
    runs = [_load_run(spec, baseline=baseline, candidate=candidate) for spec in specs]
    slices = {
        slice_name: _summarize_slice_runs(runs, slice_name=slice_name, source=source, key=key)
        for slice_name, (source, key) in SLICE_SPECS.items()
    }
    checks = _checks(slices)
    return {
        "name": "Scene-truth segmentation slice rollup",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "baseline": baseline,
        "candidate": candidate,
        "run_count": len(runs),
        "seeds": [run.get("seed") for run in runs],
        "checks": checks,
        "slices": slices,
        "runs": runs,
        "interpretation": (
            "This report rolls up slice-level candidate minus baseline deltas. It is intended to show whether RGB+Aux benefits "
            "hold across CFA patterns, PSF levels, operating conditions, and object shapes rather than only in the aggregate."
        ),
        "claim_boundary": (
            "The slices come from compact scene-truth segmentation runs. They are useful for engineering direction and CFA/PSF "
            "hypotheses, but still need larger real RAW validation before final product claims."
        ),
    }


def write_slice_rollup(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return html_path


def _load_run(spec: str, *, baseline: str, candidate: str) -> Dict[str, Any]:
    label, raw_path = _split_spec(spec)
    path = Path(raw_path).expanduser()
    summary_path = path / "scene_truth_segmentation_train_summary.json" if path.is_dir() else path
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "path": str(summary_path),
        "report": str(summary_path.parent / "index.html"),
        "status": data.get("status"),
        "seed": data.get("training", {}).get("seed"),
        "case_generation": data.get("case_generation", {}),
        "baseline": data.get("runs", {}).get(baseline, {}).get("val", {}),
        "candidate": data.get("runs", {}).get(candidate, {}).get("val", {}),
    }


def _summarize_slice_runs(runs: Sequence[Mapping[str, Any]], *, slice_name: str, source: str, key: str) -> Dict[str, Any]:
    by_value: Dict[str, list[Dict[str, Any]]] = {}
    for run in runs:
        baseline_rows = _index_rows(run.get("baseline", {}).get(source, []), key)
        candidate_rows = _index_rows(run.get("candidate", {}).get(source, []), key)
        for value in sorted(set(baseline_rows) & set(candidate_rows)):
            baseline = baseline_rows[value]
            candidate = candidate_rows[value]
            deltas = {
                f"delta_{metric}": _optional_float(candidate.get(metric)) - _optional_float(baseline.get(metric))
                for metric in METRICS
                if _optional_float(candidate.get(metric)) is not None and _optional_float(baseline.get(metric)) is not None
            }
            by_value.setdefault(value, []).append(
                {
                    "seed": run.get("seed"),
                    "label": run.get("label"),
                    "object_count": candidate.get("object_count"),
                    "baseline": {metric: baseline.get(metric) for metric in METRICS},
                    "candidate": {metric: candidate.get(metric) for metric in METRICS},
                    "deltas": deltas,
                }
            )
    values = []
    for value, rows in sorted(by_value.items()):
        metric_summary = {}
        for metric in METRICS:
            delta_key = f"delta_{metric}"
            vals = [float(row.get("deltas", {}).get(delta_key)) for row in rows if row.get("deltas", {}).get(delta_key) is not None]
            metric_summary[delta_key] = _summarize_values(vals)
        values.append(
            {
                "slice": slice_name,
                "key": key,
                "value": value,
                "run_count": len(rows),
                "seeds": [row.get("seed") for row in rows],
                "metrics": metric_summary,
                "runs": rows,
            }
        )
    for row in values:
        metrics = row.get("metrics", {})
        global_positive = _positive_rate(metrics, "delta_mask_iou_mean") == 1.0 and _positive_rate(metrics, "delta_boundary_f1_mean") == 1.0
        local_positive = _positive_rate(metrics, "delta_local_mask_iou_mean") == 1.0 and _positive_rate(metrics, "delta_local_boundary_f1_mean") == 1.0
        row["global_status"] = "pass" if global_positive else "fail"
        row["local_status"] = "pass" if local_positive else "fail"
        if global_positive:
            row["status"] = "pass"
        elif local_positive:
            row["status"] = "local_pass"
        elif slice_name == "condition" and _is_nominal_saturated(row):
            row["status"] = "saturated"
        elif slice_name == "condition" and _is_nominal_no_regression(row):
            row["status"] = "neutral"
        else:
            row["status"] = "fail"
    all_global_positive = all(row.get("global_status") == "pass" for row in values)
    all_local_positive = all(row.get("local_status") == "pass" for row in values)
    all_condition_acceptable = (
        slice_name == "condition"
        and bool(values)
        and all(row.get("global_status") == "pass" or row.get("status") in {"saturated", "neutral"} for row in values)
    )
    status = "pass" if values and all_global_positive else ("local_pass" if values and all_local_positive else "fail")
    if all_condition_acceptable:
        status = "pass"
    return {
        "source": source,
        "key": key,
        "status": status,
        "global_status": "pass" if values and all_global_positive else "fail",
        "local_status": "pass" if values and all_local_positive else "fail",
        "values": values,
    }


def _checks(slices: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    checks = [
        {
            "id": "slices_present",
            "status": "pass" if slices else "fail",
            "description": "At least one slice family was summarized.",
            "criteria": [{"metric": "slice_family_count", "value": len(slices), "pass": bool(slices)}],
        },
        {
            "id": "all_cfa_values_positive",
            "status": "pass" if slices.get("cfa", {}).get("status") == "pass" else "fail",
            "description": "Every CFA value should have positive RGB+Aux deltas for mask IoU and boundary F1 in every seed.",
            "criteria": [{"metric": "cfa_status", "value": slices.get("cfa", {}).get("status"), "pass": slices.get("cfa", {}).get("status") == "pass"}],
        },
    ]
    if "condition" in slices:
        condition_values = slices.get("condition", {}).get("values", [])
        adverse = [row for row in condition_values if str(row.get("value")) not in {"nominal", "none", "clean"}]
        adverse_pass = bool(adverse) and all(row.get("global_status") == "pass" for row in adverse)
        checks.append(
            {
                "id": "adverse_conditions_positive",
                "status": "pass" if adverse_pass else "fail",
                "description": "Every non-nominal condition should have positive RGB+Aux deltas for mask IoU and boundary F1 in every seed.",
                "criteria": [
                    {"metric": str(row.get("value")), "value": row.get("global_status"), "pass": row.get("global_status") == "pass"}
                    for row in adverse
                ],
            }
        )
    if "shape" in slices:
        shape_values = slices.get("shape", {}).get("values", [])
        hard_shapes = [row for row in shape_values if str(row.get("value")) in {"thin_long", "thin_small", "tiny_bright"}]
        local_pass = bool(hard_shapes) and all(row.get("local_status") == "pass" for row in hard_shapes)
        checks.append(
            {
                "id": "hard_shape_local_positive",
                "status": "pass" if local_pass else "fail",
                "description": "Thin/tiny object slices should have positive local mask IoU and local boundary F1 deltas in every seed.",
                "criteria": [
                    {"metric": str(row.get("value")), "value": row.get("local_status"), "pass": row.get("local_status") == "pass"}
                    for row in hard_shapes
                ],
            }
        )
    return checks


def _index_rows(rows: Any, key: str) -> Dict[str, Mapping[str, Any]]:
    out = {}
    for row in rows if isinstance(rows, list) else []:
        out[str(row.get(key))] = row
    return out


def _summarize_values(values: Sequence[float]) -> Dict[str, Any]:
    numeric = [float(value) for value in values]
    return {
        "mean": float(np.mean(numeric)) if numeric else None,
        "std": float(np.std(numeric)) if numeric else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "positive_count": sum(1 for value in numeric if value > 0.0),
        "positive_rate": float(np.mean([value > 0.0 for value in numeric])) if numeric else None,
        "values": numeric,
    }


def _positive_rate(metrics: Mapping[str, Any], metric: str) -> float | None:
    value = metrics.get(metric, {}).get("positive_rate")
    return None if value is None else float(value)


def _is_nominal_saturated(row: Mapping[str, Any]) -> bool:
    value = str(row.get("value"))
    if value not in {"nominal", "none", "clean"}:
        return False
    metrics = row.get("metrics", {})
    for metric in PRIMARY_DELTA_METRICS:
        mean = metrics.get(metric, {}).get("mean")
        if mean is None or abs(float(mean)) > NOMINAL_SATURATION_EPSILON:
            return False
    return True


def _is_nominal_no_regression(row: Mapping[str, Any]) -> bool:
    value = str(row.get("value"))
    if value not in {"nominal", "none", "clean"}:
        return False
    metrics = row.get("metrics", {})
    for metric in PRIMARY_DELTA_METRICS:
        summary = metrics.get(metric, {})
        mean = summary.get("mean")
        min_value = summary.get("min")
        if mean is None or min_value is None:
            return False
        if float(mean) < -NOMINAL_NO_REGRESSION_EPSILON or float(min_value) < -NOMINAL_NO_REGRESSION_EPSILON:
            return False
    return True


def _render_html(summary: Mapping[str, Any]) -> str:
    sections = []
    for slice_name, payload in summary.get("slices", {}).items():
        rows = []
        for row in payload.get("values", []):
            metrics = row.get("metrics", {})
            mask = metrics.get("delta_mask_iou_mean", {})
            boundary = metrics.get("delta_boundary_f1_mean", {})
            sep = metrics.get("delta_score_boundary_separation_mean", {})
            local_mask = metrics.get("delta_local_mask_iou_mean", {})
            local_boundary = metrics.get("delta_local_boundary_f1_mean", {})
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(row.get('value')))}</td>"
                f"<td>{html_lib.escape(str(row.get('status')))}</td>"
                f"<td>{int(row.get('run_count', 0))}</td>"
                f"<td>{html_lib.escape(', '.join(str(seed) for seed in row.get('seeds', [])))}</td>"
                f"<td>{_fmt(mask.get('mean'), signed=True)}</td>"
                f"<td>{_fmt(mask.get('min'), signed=True)}</td>"
                f"<td>{_fmt(mask.get('positive_rate'))}</td>"
                f"<td>{_fmt(boundary.get('mean'), signed=True)}</td>"
                f"<td>{_fmt(boundary.get('min'), signed=True)}</td>"
                f"<td>{_fmt(boundary.get('positive_rate'))}</td>"
                f"<td>{_fmt(local_mask.get('mean'), signed=True)}</td>"
                f"<td>{_fmt(local_boundary.get('mean'), signed=True)}</td>"
                f"<td>{_fmt(sep.get('mean'), signed=True)}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{html_lib.escape(str(slice_name).upper())} Slice</h2>"
            f"<p>Status: <b>{html_lib.escape(str(payload.get('status')))}</b> · Global: {html_lib.escape(str(payload.get('global_status')))} · Local: {html_lib.escape(str(payload.get('local_status')))}</p>"
            "<table><thead><tr><th>Value</th><th>Status</th><th>Runs</th><th>Seeds</th><th>dIoU mean</th><th>dIoU min</th><th>dIoU positive</th>"
            "<th>dBoundary mean</th><th>dBoundary min</th><th>dBoundary positive</th><th>dLocal IoU mean</th><th>dLocal Boundary mean</th><th>dScoreSep mean</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Segmentation Slice Rollup</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #16202a; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 30px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Scene-truth Segmentation Slice Rollup</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <div class="note">Shape slices report both global object metrics and local object-crop metrics. In binary union segmentation, global per-object metrics can penalize correct detections of other objects as false positives for a small object; local metrics are the primary signal for thin/tiny edge evidence.</div>
  <p>Status: <b>{html_lib.escape(str(summary.get('status')))}</b> · Baseline: {html_lib.escape(str(summary.get('baseline')))} · Candidate: {html_lib.escape(str(summary.get('candidate')))}</p>
  {''.join(sections)}
</body>
</html>
"""


def _split_spec(spec: str) -> tuple[str, str]:
    if "|" not in spec:
        path = Path(spec).expanduser()
        return path.parent.name if path.name == "scene_truth_segmentation_train_summary.json" else path.name, str(path)
    label, path = spec.split("|", 1)
    return label.strip(), path.strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
