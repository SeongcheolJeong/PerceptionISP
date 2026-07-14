"""Build a condition-sweep report from scene-truth segmentation rollups."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .types import json_ready


SUMMARY_FILENAME = "scene_truth_condition_sweep_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Roll up scene-truth segmentation results across operating conditions.")
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Condition spec formatted as label|severity|rollup_dir_or_json|slice_dir_or_json.",
    )
    parser.add_argument("--output-dir", default="reports/perception_scene_truth_condition_sweep_v1")
    args = parser.parse_args(argv)
    summary = build_condition_sweep([str(value) for value in args.condition])
    html_path = write_condition_sweep(summary, args.output_dir)
    print(
        json.dumps(
            json_ready({"report": str(html_path), "summary_json": str(html_path.parent / SUMMARY_FILENAME), "status": summary["status"]}),
            indent=2,
        )
    )
    return 0


def build_condition_sweep(specs: Sequence[str]) -> Dict[str, Any]:
    conditions = [_load_condition(spec) for spec in specs]
    conditions.sort(key=lambda row: row["severity_sort"])
    checks = _checks(conditions)
    return {
        "name": "Scene-truth segmentation condition sweep",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "review",
        "conditions": conditions,
        "checks": checks,
        "trend": _trend_summary(conditions),
        "interpretation": (
            "Nominal scenes are a saturation/no-regression reference. RGB+Aux gains become meaningful once low-light/noise degrades "
            "the RGB reconstruction path, and the recovery grows across the tested severity sweep."
        ),
        "claim_boundary": (
            "This is controlled scene-truth evidence. It supports condition-dependent robustness and recovery, not a blanket claim of "
            "large nominal-scene gains or final real-native-RAW product performance."
        ),
    }


def write_condition_sweep(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return html_path


def _load_condition(spec: str) -> Dict[str, Any]:
    label, severity_text, rollup_raw, slice_raw = _split_spec(spec)
    severity = _parse_severity(severity_text)
    rollup_path = _summary_path(Path(rollup_raw).expanduser(), "scene_truth_segmentation_rollup_summary.json")
    slice_path = _summary_path(Path(slice_raw).expanduser(), "scene_truth_segmentation_slice_rollup_summary.json")
    rollup = json.loads(rollup_path.read_text(encoding="utf-8"))
    slice_summary = json.loads(slice_path.read_text(encoding="utf-8"))
    group = _first_group(rollup)
    runs = list(group.get("runs", []))
    metrics = group.get("metrics", {})
    val_means = _val_metric_means(runs)
    aux_minus_human = _mean_run_delta(runs, "perception_rgb_aux", "human_rgb")
    shape_rows = _shape_rows(slice_summary)
    cfa_rows = _family_rows(slice_summary, "cfa")
    psf_rows = _family_rows(slice_summary, "psf")
    return {
        "label": label,
        "severity": severity,
        "severity_sort": -1.0 if severity is None else float(severity),
        "rollup_status": rollup.get("status"),
        "slice_status": slice_summary.get("status"),
        "run_count": group.get("run_count"),
        "seeds": group.get("seeds", []),
        "rollup_report": str(rollup_path.parent / "index.html"),
        "slice_report": str(slice_path.parent / "index.html"),
        "rgb_aux_minus_perception_rgb": {
            "mean_delta_mask_iou": _metric_mean(metrics, "delta_mask_iou_mean"),
            "mean_delta_boundary_f1": _metric_mean(metrics, "delta_boundary_f1_mean"),
            "positive_rate_mask_iou": _metric_positive_rate(metrics, "delta_mask_iou_mean"),
            "positive_rate_boundary_f1": _metric_positive_rate(metrics, "delta_boundary_f1_mean"),
            "values_delta_mask_iou": _metric_values(metrics, "delta_mask_iou_mean"),
            "values_delta_boundary_f1": _metric_values(metrics, "delta_boundary_f1_mean"),
        },
        "rgb_aux_minus_human_rgb": aux_minus_human,
        "absolute_metrics_mean": val_means,
        "saturation": {
            "perception_rgb_boundary_f1_saturated": bool((val_means.get("perception_rgb_boundary_f1") or 0.0) >= 0.999),
            "perception_rgb_mask_iou": val_means.get("perception_rgb_mask_iou"),
            "perception_rgb_boundary_f1": val_means.get("perception_rgb_boundary_f1"),
        },
        "shape": shape_rows,
        "cfa": cfa_rows,
        "psf": psf_rows,
    }


def _split_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"condition spec must be label|severity|rollup|slice, got {spec!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()


def _parse_severity(text: str) -> float | None:
    lowered = text.strip().lower()
    if lowered in {"none", "nominal", "na"}:
        return None
    return float(lowered)


def _summary_path(path: Path, filename: str) -> Path:
    return path / filename if path.is_dir() else path


def _first_group(rollup: Mapping[str, Any]) -> Mapping[str, Any]:
    groups = rollup.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        return {}
    return next(iter(groups.values()))


def _metric_mean(metrics: Mapping[str, Any], name: str) -> float | None:
    return _optional_float(metrics.get(name, {}).get("mean"))


def _metric_positive_rate(metrics: Mapping[str, Any], name: str) -> float | None:
    return _optional_float(metrics.get(name, {}).get("positive_rate"))


def _metric_values(metrics: Mapping[str, Any], name: str) -> list[float]:
    values = metrics.get(name, {}).get("values", [])
    return [float(value) for value in values if value is not None]


def _val_metric_means(runs: Sequence[Mapping[str, Any]]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for path_name in ("human_rgb", "perception_rgb", "perception_rgb_aux"):
        for metric in ("mask_iou_mean", "boundary_f1_mean"):
            values = [
                _optional_float(run.get("val_metrics", {}).get(path_name, {}).get(metric))
                for run in runs
                if _optional_float(run.get("val_metrics", {}).get(path_name, {}).get(metric)) is not None
            ]
            out[f"{path_name}_{metric.replace('_mean', '')}"] = _mean(values)
    return out


def _mean_run_delta(runs: Sequence[Mapping[str, Any]], candidate: str, baseline: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in ("mask_iou_mean", "boundary_f1_mean"):
        values = []
        for run in runs:
            val = run.get("val_metrics", {})
            cand = _optional_float(val.get(candidate, {}).get(metric))
            base = _optional_float(val.get(baseline, {}).get(metric))
            if cand is not None and base is not None:
                values.append(cand - base)
        out[f"mean_delta_{metric}"] = _mean(values)
        out[f"values_delta_{metric}"] = values
    return out


def _shape_rows(slice_summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in slice_summary.get("slices", {}).get("shape", {}).get("values", []):
        metrics = row.get("metrics", {})
        rows[str(row.get("value"))] = {
            "delta_global_mask_iou_mean": _metric_mean(metrics, "delta_mask_iou_mean"),
            "delta_global_mask_iou_positive_rate": _metric_positive_rate(metrics, "delta_mask_iou_mean"),
            "delta_global_boundary_f1_mean": _metric_mean(metrics, "delta_boundary_f1_mean"),
            "delta_global_boundary_f1_positive_rate": _metric_positive_rate(metrics, "delta_boundary_f1_mean"),
            "delta_local_mask_iou_mean": _metric_mean(metrics, "delta_local_mask_iou_mean"),
            "delta_local_mask_iou_positive_rate": _metric_positive_rate(metrics, "delta_local_mask_iou_mean"),
            "delta_local_boundary_f1_mean": _metric_mean(metrics, "delta_local_boundary_f1_mean"),
            "delta_local_boundary_f1_positive_rate": _metric_positive_rate(metrics, "delta_local_boundary_f1_mean"),
        }
    return rows


def _family_rows(slice_summary: Mapping[str, Any], family: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in slice_summary.get("slices", {}).get(family, {}).get("values", []):
        metrics = row.get("metrics", {})
        rows[str(row.get("value"))] = {
            "delta_mask_iou_mean": _metric_mean(metrics, "delta_mask_iou_mean"),
            "delta_mask_iou_positive_rate": _metric_positive_rate(metrics, "delta_mask_iou_mean"),
            "delta_boundary_f1_mean": _metric_mean(metrics, "delta_boundary_f1_mean"),
            "delta_boundary_f1_positive_rate": _metric_positive_rate(metrics, "delta_boundary_f1_mean"),
        }
    return rows


def _trend_summary(conditions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for condition in conditions:
        metrics = condition.get("rgb_aux_minus_perception_rgb", {})
        rows.append(
            {
                "label": condition.get("label"),
                "severity": condition.get("severity"),
                "mean_delta_mask_iou": metrics.get("mean_delta_mask_iou"),
                "mean_delta_boundary_f1": metrics.get("mean_delta_boundary_f1"),
                "perception_rgb_mask_iou": condition.get("absolute_metrics_mean", {}).get("perception_rgb_mask_iou"),
                "perception_rgb_boundary_f1": condition.get("absolute_metrics_mean", {}).get("perception_rgb_boundary_f1"),
            }
        )
    adverse_rows = [row for row in rows if row.get("severity") is not None]
    return {
        "rows": rows,
        "adverse_delta_mask_iou_monotonic_non_decreasing": _non_decreasing([row.get("mean_delta_mask_iou") for row in adverse_rows]),
        "adverse_delta_boundary_f1_monotonic_non_decreasing": _non_decreasing([row.get("mean_delta_boundary_f1") for row in adverse_rows]),
    }


def _checks(conditions: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    adverse = [row for row in conditions if row.get("severity") is not None]
    nominal = [row for row in conditions if row.get("severity") is None]
    return [
        {
            "id": "conditions_present",
            "status": "pass" if len(conditions) >= 2 else "fail",
            "description": "At least nominal and one adverse condition should be present.",
            "criteria": [{"metric": "condition_count", "value": len(conditions), "threshold": 2, "pass": len(conditions) >= 2}],
        },
        {
            "id": "adverse_conditions_positive",
            "status": "pass" if adverse and all(_condition_positive(row) for row in adverse) else "fail",
            "description": "Every adverse condition should have positive aggregate RGB+Aux minus RGB-only deltas.",
            "criteria": [
                {"metric": str(row.get("label")), "value": _condition_positive(row), "pass": _condition_positive(row)}
                for row in adverse
            ],
        },
        {
            "id": "nominal_saturated_or_positive",
            "status": "pass" if not nominal or all(_nominal_saturated_or_positive(row) for row in nominal) else "fail",
            "description": "Nominal conditions should either pass positive deltas or be explicitly saturated.",
            "criteria": [
                {
                    "metric": str(row.get("label")),
                    "value": _nominal_saturated_or_positive(row),
                    "pass": _nominal_saturated_or_positive(row),
                }
                for row in nominal
            ],
        },
    ]


def _condition_positive(condition: Mapping[str, Any]) -> bool:
    metrics = condition.get("rgb_aux_minus_perception_rgb", {})
    return (metrics.get("mean_delta_mask_iou") or 0.0) > 0.0 and (metrics.get("mean_delta_boundary_f1") or 0.0) > 0.0


def _nominal_saturated_or_positive(condition: Mapping[str, Any]) -> bool:
    metrics = condition.get("rgb_aux_minus_perception_rgb", {})
    saturated = bool(condition.get("saturation", {}).get("perception_rgb_boundary_f1_saturated"))
    return _condition_positive(condition) or (saturated and (metrics.get("mean_delta_mask_iou") or 0.0) >= 0.0)


def _render_html(summary: Mapping[str, Any]) -> str:
    condition_rows = []
    shape_rows = []
    max_delta = max(
        [float(row.get("rgb_aux_minus_perception_rgb", {}).get("mean_delta_mask_iou") or 0.0) for row in summary.get("conditions", [])]
        + [0.001]
    )
    for condition in summary.get("conditions", []):
        metrics = condition.get("rgb_aux_minus_perception_rgb", {})
        abs_metrics = condition.get("absolute_metrics_mean", {})
        label = str(condition.get("label"))
        mask = float(metrics.get("mean_delta_mask_iou") or 0.0)
        bar_width = min(100.0, max(0.0, mask / max_delta * 100.0))
        condition_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(label)}</td>"
            f"<td>{_fmt(condition.get('severity'))}</td>"
            f"<td>{_fmt(abs_metrics.get('perception_rgb_mask_iou'))}</td>"
            f"<td>{_fmt(abs_metrics.get('perception_rgb_boundary_f1'))}</td>"
            f"<td>{_fmt(metrics.get('mean_delta_mask_iou'), signed=True)}<div class='bar'><span style='width:{bar_width:.1f}%'></span></div></td>"
            f"<td>{_fmt(metrics.get('mean_delta_boundary_f1'), signed=True)}</td>"
            f"<td>{_fmt(metrics.get('positive_rate_mask_iou'))}</td>"
            f"<td>{_fmt(metrics.get('positive_rate_boundary_f1'))}</td>"
            f"<td><a href='{html_lib.escape(str(condition.get('rollup_report')))}'>rollup</a> · <a href='{html_lib.escape(str(condition.get('slice_report')))}'>slices</a></td>"
            "</tr>"
        )
        for shape in ("thin_small", "tiny_bright"):
            row = condition.get("shape", {}).get(shape, {})
            shape_rows.append(
                "<tr>"
                f"<td>{html_lib.escape(label)}</td>"
                f"<td>{html_lib.escape(shape)}</td>"
                f"<td>{_fmt(row.get('delta_local_mask_iou_mean'), signed=True)}</td>"
                f"<td>{_fmt(row.get('delta_local_mask_iou_positive_rate'))}</td>"
                f"<td>{_fmt(row.get('delta_local_boundary_f1_mean'), signed=True)}</td>"
                f"<td>{_fmt(row.get('delta_local_boundary_f1_positive_rate'))}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scene-truth Segmentation Condition Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #16202a; }}
    .note {{ background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 30px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .bar {{ height: 6px; background: #e5e7eb; border-radius: 999px; margin-top: 5px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: #2563eb; }}
  </style>
</head>
<body>
  <h1>Scene-truth Segmentation Condition Sweep</h1>
  <div class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <b>{html_lib.escape(str(summary.get('status')))}</b></p>
  <h2>Condition Trend</h2>
  <table>
    <thead><tr><th>Condition</th><th>Severity</th><th>RGB-only IoU</th><th>RGB-only boundary</th><th>RGB+Aux dIoU</th><th>RGB+Aux dBoundary</th><th>dIoU positive</th><th>dBoundary positive</th><th>Reports</th></tr></thead>
    <tbody>{''.join(condition_rows)}</tbody>
  </table>
  <h2>Hard Shape Local Metrics</h2>
  <table>
    <thead><tr><th>Condition</th><th>Shape</th><th>dLocal IoU</th><th>dLocal IoU positive</th><th>dLocal Boundary</th><th>dLocal Boundary positive</th></tr></thead>
    <tbody>{''.join(shape_rows)}</tbody>
  </table>
</body>
</html>
"""


def _non_decreasing(values: Sequence[Any]) -> bool:
    numeric = [float(value) for value in values if value is not None]
    return all(b + 1.0e-12 >= a for a, b in zip(numeric, numeric[1:]))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    val = float(value)
    if signed:
        return f"{val:+.4f}"
    return f"{val:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
