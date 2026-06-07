"""Visual success/failure casebook for PerceptionISP comparison reports."""

from __future__ import annotations

import argparse
import copy
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from PIL import Image, ImageDraw

from .types import json_ready


SUMMARY_FILENAME = "casebook_summary.json"
DEFAULT_BASELINE_INPUT = "human_rgb"
DEFAULT_TARGET_INPUT = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001"
CASE_CATEGORIES = (
    "fp_reduction_success",
    "recall_tradeoff",
    "recall_loss_failure",
    "fp_regression_failure",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a visual PerceptionISP success/failure casebook from a comparison report.")
    parser.add_argument("comparison_report", help="comparison_summary.json path or comparison report directory.")
    parser.add_argument("--baseline-input", default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--target-input", default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--max-cases-per-category", type=int, default=8)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output-dir", default="reports/perception_casebook")
    args = parser.parse_args(argv)

    summary = build_casebook_from_path(
        args.comparison_report,
        baseline_input=str(args.baseline_input),
        target_input=str(args.target_input),
        max_cases_per_category=int(args.max_cases_per_category),
        match_iou=float(args.match_iou),
    )
    html_path = write_casebook(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "sample_count": summary["sample_count"],
                    "selected_case_count": summary["selected_case_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_casebook_from_path(
    report: str | Path,
    *,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str = DEFAULT_TARGET_INPUT,
    max_cases_per_category: int = 8,
    match_iou: float = 0.50,
) -> Dict[str, Any]:
    report_path = _summary_path(report, "comparison_summary.json")
    data = json.loads(report_path.read_text())
    return build_casebook(
        data,
        source_report=report_path,
        baseline_input=baseline_input,
        target_input=target_input,
        max_cases_per_category=max_cases_per_category,
        match_iou=match_iou,
    )


def build_casebook(
    report: Mapping[str, Any],
    *,
    source_report: str | Path | None = None,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str = DEFAULT_TARGET_INPUT,
    max_cases_per_category: int = 8,
    match_iou: float = 0.50,
) -> Dict[str, Any]:
    samples = [row for row in report.get("samples", ()) if isinstance(row, Mapping)]
    run_config = report.get("run_config", {}) if isinstance(report.get("run_config"), Mapping) else {}
    label_agnostic = bool(run_config.get("label_agnostic", False))
    categories: Dict[str, Dict[str, Any]] = {
        name: {"case_count": 0, "selected_case_count": 0, "cases": []}
        for name in CASE_CATEGORIES
    }
    aggregate = {
        "baseline_tp@0.50": 0,
        "target_tp@0.50": 0,
        "baseline_fp@0.50": 0,
        "target_fp@0.50": 0,
        "baseline_fn@0.50": 0,
        "target_fn@0.50": 0,
    }
    candidate_cases: Dict[str, list[Dict[str, Any]]] = {name: [] for name in CASE_CATEGORIES}
    for index, sample in enumerate(samples):
        baseline_metrics = _metrics_for_sample(sample, baseline_input)
        target_metrics = _metrics_for_sample(sample, target_input)
        if not baseline_metrics or not target_metrics:
            continue
        baseline_tp = int(baseline_metrics.get("tp@0.50", 0))
        target_tp = int(target_metrics.get("tp@0.50", 0))
        baseline_fp = int(baseline_metrics.get("fp@0.50", 0))
        target_fp = int(target_metrics.get("fp@0.50", 0))
        baseline_fn = int(baseline_metrics.get("fn@0.50", 0))
        target_fn = int(target_metrics.get("fn@0.50", 0))
        aggregate["baseline_tp@0.50"] += baseline_tp
        aggregate["target_tp@0.50"] += target_tp
        aggregate["baseline_fp@0.50"] += baseline_fp
        aggregate["target_fp@0.50"] += target_fp
        aggregate["baseline_fn@0.50"] += baseline_fn
        aggregate["target_fn@0.50"] += target_fn
        fp_delta = target_fp - baseline_fp
        tp_delta = target_tp - baseline_tp
        category = _case_category(fp_delta=fp_delta, tp_delta=tp_delta)
        if category is None:
            continue
        categories[category]["case_count"] += 1
        candidate_cases[category].append(
            _case_summary(
                sample,
                sample_index=index,
                category=category,
                baseline_input=baseline_input,
                target_input=target_input,
                baseline_metrics=baseline_metrics,
                target_metrics=target_metrics,
                label_agnostic=label_agnostic,
                match_iou=match_iou,
            )
        )
    for category, rows in candidate_cases.items():
        selected = sorted(rows, key=_case_sort_key)[: max(int(max_cases_per_category), 0)]
        categories[category]["cases"] = selected
        categories[category]["selected_case_count"] = len(selected)
    aggregate["tp_delta_count"] = aggregate["target_tp@0.50"] - aggregate["baseline_tp@0.50"]
    aggregate["fp_delta_count"] = aggregate["target_fp@0.50"] - aggregate["baseline_fp@0.50"]
    selected_case_count = sum(int(row["selected_case_count"]) for row in categories.values())
    checks = _checks(categories, selected_case_count=selected_case_count)
    return {
        "name": "PerceptionISP success/failure casebook",
        "source_report": "" if source_report is None else str(source_report),
        "baseline_input": str(baseline_input),
        "target_input": str(target_input),
        "label_agnostic": label_agnostic,
        "match_iou": float(match_iou),
        "sample_count": len(samples),
        "selected_case_count": selected_case_count,
        "aggregate": aggregate,
        "categories": categories,
        "checks": checks,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This casebook selects representative sample-level successes and failures from the same comparison report used by the claim gates. "
            "It is a review artifact for explaining FP reduction, recall tradeoffs, and regressions."
        ),
        "claim_boundary": (
            "Use this as qualitative visual evidence only. It does not replace held-out claim gates, native RAW/CFA coverage, or trained RGB+Aux DNN evaluation."
        ),
    }


def write_casebook(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    assets_dir = destination / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    materialized: Dict[str, Any] = copy.deepcopy(dict(summary))
    for category, payload in (materialized.get("categories", {}) if isinstance(materialized.get("categories"), Mapping) else {}).items():
        if not isinstance(payload, Mapping):
            continue
        for index, case in enumerate(payload.get("cases", ())):
            if not isinstance(case, dict):
                continue
            asset_name = f"{_safe_id(str(category))}_{index:02d}_{_safe_id(str(case.get('sample_id', 'sample')))}.png"
            asset_path = assets_dir / asset_name
            if _render_case_image(case, asset_path):
                case["visual_path"] = str(asset_path)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(materialized), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(materialized, destination))
    return html_path


def _case_category(*, fp_delta: int, tp_delta: int) -> str | None:
    if fp_delta < 0 and tp_delta >= 0:
        return "fp_reduction_success"
    if fp_delta < 0 and tp_delta < 0:
        return "recall_tradeoff"
    if tp_delta < 0:
        return "recall_loss_failure"
    if fp_delta > 0:
        return "fp_regression_failure"
    return None


def _case_sort_key(case: Mapping[str, Any]) -> tuple[float, float, float, str]:
    fp_delta = float(case.get("fp_delta@0.50", 0.0))
    tp_delta = float(case.get("tp_delta@0.50", 0.0))
    category = str(case.get("category", ""))
    if category == "fp_reduction_success":
        return (fp_delta, -float(case.get("baseline_fp@0.50", 0.0)), 0.0, str(case.get("sample_id", "")))
    if category == "recall_tradeoff":
        return (tp_delta, fp_delta, 0.0, str(case.get("sample_id", "")))
    if category == "recall_loss_failure":
        return (tp_delta, -fp_delta, 0.0, str(case.get("sample_id", "")))
    return (-fp_delta, tp_delta, 0.0, str(case.get("sample_id", "")))


def _case_summary(
    sample: Mapping[str, Any],
    *,
    sample_index: int,
    category: str,
    baseline_input: str,
    target_input: str,
    baseline_metrics: Mapping[str, Any],
    target_metrics: Mapping[str, Any],
    label_agnostic: bool,
    match_iou: float,
) -> Dict[str, Any]:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), Mapping) else {}
    raw_provenance = metadata.get("raw_provenance", {}) if isinstance(metadata.get("raw_provenance"), Mapping) else {}
    baseline_detections = _detections_for_sample(sample, baseline_input)
    target_detections = _detections_for_sample(sample, target_input)
    ground_truth = _ground_truth_for_sample(sample)
    overlays = _overlay_rows(
        baseline_detections,
        target_detections,
        ground_truth,
        label_agnostic=label_agnostic,
        match_iou=match_iou,
    )
    baseline_tp = int(baseline_metrics.get("tp@0.50", 0))
    target_tp = int(target_metrics.get("tp@0.50", 0))
    baseline_fp = int(baseline_metrics.get("fp@0.50", 0))
    target_fp = int(target_metrics.get("fp@0.50", 0))
    return {
        "case_id": f"{category}:{sample.get('sample_id', sample_index)}",
        "sample_id": str(sample.get("sample_id", sample_index)),
        "category": category,
        "source": str(sample.get("source", "")),
        "image_path": str(metadata.get("image_path", "")),
        "width": int(metadata.get("width", 0)),
        "height": int(metadata.get("height", 0)),
        "original_width": int(metadata.get("original_width", 0)),
        "original_height": int(metadata.get("original_height", 0)),
        "cfa_pattern": str(metadata.get("cfa_pattern", raw_provenance.get("target_cfa_pattern", ""))),
        "psf_sigma": _maybe_float_or_none(metadata.get("psf_sigma")),
        "pattern_remapped": bool(raw_provenance.get("pattern_remapped", False)),
        "true_sensor_cfa_mosaic": bool(raw_provenance.get("true_sensor_cfa_mosaic", False)),
        "baseline_input": baseline_input,
        "target_input": target_input,
        "baseline_tp@0.50": baseline_tp,
        "target_tp@0.50": target_tp,
        "baseline_fp@0.50": baseline_fp,
        "target_fp@0.50": target_fp,
        "baseline_fn@0.50": int(baseline_metrics.get("fn@0.50", 0)),
        "target_fn@0.50": int(target_metrics.get("fn@0.50", 0)),
        "fp_delta@0.50": target_fp - baseline_fp,
        "tp_delta@0.50": target_tp - baseline_tp,
        "overlays": overlays,
        "overlay_counts": _overlay_counts(overlays),
    }


def _checks(categories: Mapping[str, Mapping[str, Any]], *, selected_case_count: int) -> list[Dict[str, Any]]:
    success_count = int(categories.get("fp_reduction_success", {}).get("selected_case_count", 0))
    counter_count = sum(
        int(categories.get(name, {}).get("selected_case_count", 0))
        for name in ("recall_tradeoff", "recall_loss_failure", "fp_regression_failure")
    )
    return [
        {
            "id": "casebook_has_selected_cases",
            "status": "pass" if selected_case_count > 0 else "fail",
            "evidence": f"selected_cases={selected_case_count}",
        },
        {
            "id": "casebook_includes_fp_reduction_successes",
            "status": "pass" if success_count > 0 else "fail",
            "evidence": f"selected_successes={success_count}",
        },
        {
            "id": "casebook_includes_counterexamples",
            "status": "pass" if counter_count > 0 else "fail",
            "evidence": f"selected_counterexamples={counter_count}",
        },
    ]


def _render_case_image(case: Mapping[str, Any], output_path: Path) -> bool:
    image_path = str(case.get("image_path", ""))
    if not image_path:
        return False
    path = Path(image_path).expanduser()
    if not path.exists():
        return False
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return False
    source_w, source_h = image.size
    target_w = int(case.get("width", 0)) or source_w
    target_h = int(case.get("height", 0)) or source_h
    scale_x = source_w / max(float(target_w), 1.0)
    scale_y = source_h / max(float(target_h), 1.0)
    max_width = 960
    display_scale = min(max_width / max(float(source_w), 1.0), 1.0)
    if display_scale < 1.0:
        image = image.resize((int(round(source_w * display_scale)), int(round(source_h * display_scale))))
    draw = ImageDraw.Draw(image)
    line_width = max(2, int(round(3 * display_scale)))
    for overlay in case.get("overlays", ()):
        if not isinstance(overlay, Mapping):
            continue
        xyxy = overlay.get("xyxy", ())
        if len(xyxy) != 4:
            continue
        coords = [
            float(xyxy[0]) * scale_x * display_scale,
            float(xyxy[1]) * scale_y * display_scale,
            float(xyxy[2]) * scale_x * display_scale,
            float(xyxy[3]) * scale_y * display_scale,
        ]
        color = _overlay_color(str(overlay.get("role", "")), bool(overlay.get("is_tp", False)))
        draw.rectangle(coords, outline=color, width=line_width)
        label = _overlay_label(overlay)
        if label:
            text_origin = (coords[0] + 2, max(coords[1] - 14, 0))
            draw.text(text_origin, label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return True


def _overlay_rows(
    baseline_detections: Sequence[Mapping[str, Any]],
    target_detections: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    label_agnostic: bool,
    match_iou: float,
) -> list[Dict[str, Any]]:
    baseline_positive = _positive_detection_indices(baseline_detections, ground_truth, label_agnostic=label_agnostic)
    target_positive = _positive_detection_indices(target_detections, ground_truth, label_agnostic=label_agnostic)
    overlays = [
        {
            "role": "ground_truth",
            "xyxy": list(_box_payload(gt).get("xyxy", ())),
            "label": str(_box_payload(gt).get("label", "")),
            "is_tp": True,
            "score": None,
        }
        for gt in ground_truth
    ]
    matched_targets: set[int] = set()
    for index, detection in enumerate(baseline_detections):
        target_index = _matching_detection_index(detection, target_detections, match_iou=match_iou, unavailable=matched_targets)
        is_tp = index in baseline_positive
        role = "common" if target_index >= 0 else "baseline_only"
        if target_index >= 0:
            matched_targets.add(target_index)
        overlays.append(_detection_overlay(detection, role=role, is_tp=is_tp))
    for index, detection in enumerate(target_detections):
        if index in matched_targets:
            continue
        overlays.append(_detection_overlay(detection, role="target_only", is_tp=index in target_positive))
    return overlays


def _detection_overlay(detection: Mapping[str, Any], *, role: str, is_tp: bool) -> Dict[str, Any]:
    box = _box_payload(detection)
    fusion = _fusion_payload(detection)
    return {
        "role": role,
        "xyxy": list(box.get("xyxy", ())),
        "label": str(box.get("label", "")),
        "score": _maybe_float_or_none(detection.get("score")),
        "is_tp": bool(is_tp),
        "edge_support": _maybe_float_or_none(fusion.get("edge_support")),
        "aux_support": _maybe_float_or_none(fusion.get("aux_support")),
        "reliability_support": _maybe_float_or_none(fusion.get("reliability_support")),
    }


def _overlay_counts(overlays: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in overlays:
        role = str(row.get("role", ""))
        key = f"{role}_{'tp' if bool(row.get('is_tp', False)) else 'fp'}" if role != "ground_truth" else "ground_truth"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    category_rows = "".join(_category_summary_row(name, payload) for name, payload in summary.get("categories", {}).items() if isinstance(payload, Mapping))
    if not category_rows:
        category_rows = '<tr><td colspan="4">No case categories were available.</td></tr>'
    case_sections = "\n".join(
        _category_case_section(name, payload, destination)
        for name, payload in summary.get("categories", {}).items()
        if isinstance(payload, Mapping)
    )
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in summary.get("checks", ())
        if isinstance(row, Mapping)
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Success/Failure Casebook</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f1; padding: 1px 5px; border-radius: 4px; }}
    img {{ max-width: 100%; border: 1px solid #d8ded7; background: #111827; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .case {{ margin: 18px 0 26px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Success/Failure Casebook</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <table><tbody>
    <tr><th>Status</th><td><code>{html_lib.escape(str(summary.get('status', '')))}</code></td><th>Samples</th><td>{int(summary.get('sample_count', 0))}</td></tr>
    <tr><th>Baseline</th><td><code>{html_lib.escape(str(summary.get('baseline_input', '')))}</code></td><th>Target</th><td><code>{html_lib.escape(str(summary.get('target_input', '')))}</code></td></tr>
    <tr><th>Net TP Delta</th><td>{int(aggregate.get('tp_delta_count', 0))}</td><th>Net FP Delta</th><td>{int(aggregate.get('fp_delta_count', 0))}</td></tr>
  </tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Category Summary</h2>
  <table><thead><tr><th>Category</th><th>Total Samples</th><th>Selected</th><th>Meaning</th></tr></thead><tbody>{category_rows}</tbody></table>
  <h2>Legend</h2>
  <p>Green: ground truth. Red/orange: baseline-only FP/TP. Blue/purple: target-only TP/FP. Gray: common detections.</p>
  {case_sections}
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _category_summary_row(name: str, payload: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{int(payload.get('case_count', 0))}</td>"
        f"<td>{int(payload.get('selected_case_count', 0))}</td>"
        f"<td>{html_lib.escape(_category_description(str(name)))}</td>"
        "</tr>"
    )


def _category_case_section(name: str, payload: Mapping[str, Any], destination: Path) -> str:
    rows = []
    for case in payload.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        visual = _case_visual(case, destination)
        rows.append(
            "<div class=\"case\">"
            f"<h3>{html_lib.escape(str(case.get('sample_id', '')))}: {html_lib.escape(str(name))}</h3>"
            "<table><tbody>"
            f"<tr><th>TP Delta</th><td>{int(case.get('tp_delta@0.50', 0))}</td><th>FP Delta</th><td>{int(case.get('fp_delta@0.50', 0))}</td></tr>"
            f"<tr><th>Baseline TP/FP</th><td>{int(case.get('baseline_tp@0.50', 0))}/{int(case.get('baseline_fp@0.50', 0))}</td><th>Target TP/FP</th><td>{int(case.get('target_tp@0.50', 0))}/{int(case.get('target_fp@0.50', 0))}</td></tr>"
            f"<tr><th>CFA</th><td>{html_lib.escape(str(case.get('cfa_pattern', '')))}</td><th>Pattern Remapped</th><td>{html_lib.escape(str(bool(case.get('pattern_remapped', False))))}</td></tr>"
            f"<tr><th>Overlay Counts</th><td colspan=\"3\">{html_lib.escape(str(case.get('overlay_counts', {})))}</td></tr>"
            "</tbody></table>"
            f"{visual}"
            "</div>"
        )
    if not rows:
        rows.append("<p>No selected cases for this category.</p>")
    return f"<h2>{html_lib.escape(str(name))}</h2>" + "\n".join(rows)


def _case_visual(case: Mapping[str, Any], destination: Path) -> str:
    visual_path = str(case.get("visual_path", ""))
    if not visual_path:
        return "<p>No visual asset was generated for this case.</p>"
    relative = os.path.relpath(visual_path, start=str(destination))
    alt = html_lib.escape(str(case.get("case_id", "case")))
    return f"<img src=\"{html_lib.escape(relative)}\" alt=\"{alt}\">"


def _category_description(name: str) -> str:
    return {
        "fp_reduction_success": "Target reduces FP while keeping TP count at least as high as baseline.",
        "recall_tradeoff": "Target reduces FP but loses TP on the same sample.",
        "recall_loss_failure": "Target loses TP without a same-sample FP reduction.",
        "fp_regression_failure": "Target adds FP relative to baseline.",
    }.get(name, "")


def _overlay_color(role: str, is_tp: bool) -> tuple[int, int, int]:
    if role == "ground_truth":
        return (22, 163, 74)
    if role == "baseline_only":
        return (234, 88, 12) if is_tp else (220, 38, 38)
    if role == "target_only":
        return (37, 99, 235) if is_tp else (147, 51, 234)
    return (75, 85, 99)


def _overlay_label(row: Mapping[str, Any]) -> str:
    role = str(row.get("role", ""))
    if role == "ground_truth":
        return f"GT {row.get('label', '')}"
    side = "B" if role == "baseline_only" else "T" if role == "target_only" else "C"
    kind = "TP" if bool(row.get("is_tp", False)) else "FP"
    score = row.get("score")
    return f"{side}-{kind}" if score is None else f"{side}-{kind} {float(score):.2f}"


def _metrics_for_sample(sample: Mapping[str, Any], input_name: str) -> Mapping[str, Any]:
    metrics = sample.get("metrics", {}) if isinstance(sample.get("metrics"), Mapping) else {}
    row = metrics.get(str(input_name), {})
    return row if isinstance(row, Mapping) else {}


def _detections_for_sample(sample: Mapping[str, Any], input_name: str) -> list[Mapping[str, Any]]:
    for detector in sample.get("detectors", ()):
        if isinstance(detector, Mapping) and str(detector.get("input_name")) == str(input_name):
            return [row for row in detector.get("detections", ()) if isinstance(row, Mapping)]
    return []


def _ground_truth_for_sample(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in sample.get("ground_truth", ()) if isinstance(row, Mapping)]


def _positive_detection_indices(detections: Sequence[Mapping[str, Any]], ground_truth: Sequence[Mapping[str, Any]], *, label_agnostic: bool) -> set[int]:
    ordered = sorted(range(len(detections)), key=lambda index: _maybe_float(detections[index].get("score")), reverse=True)
    used_gt: set[int] = set()
    positives: set[int] = set()
    for detection_index in ordered:
        detection_box = _box_payload(detections[detection_index])
        best_iou = 0.0
        best_gt = -1
        for gt_index, gt_payload in enumerate(ground_truth):
            if gt_index in used_gt:
                continue
            gt_box = _box_payload(gt_payload)
            if not label_agnostic and str(detection_box.get("label", "")) != str(gt_box.get("label", "")):
                continue
            value = _box_iou(detection_box, gt_box)
            if value > best_iou:
                best_iou = value
                best_gt = gt_index
        if best_iou >= 0.50 and best_gt >= 0:
            positives.add(detection_index)
            used_gt.add(best_gt)
    return positives


def _matching_detection_index(
    detection: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    match_iou: float,
    unavailable: set[int],
) -> int:
    box = _box_payload(detection)
    best_index = -1
    best_iou = 0.0
    for index, target in enumerate(targets):
        if index in unavailable:
            continue
        target_box = _box_payload(target)
        if str(box.get("label", "")) != str(target_box.get("label", "")):
            continue
        value = _box_iou(box, target_box)
        if value > best_iou:
            best_iou = value
            best_index = index
    return best_index if best_iou >= float(match_iou) else -1


def _box_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    box = payload.get("box", payload)
    if not isinstance(box, Mapping):
        box = {}
    values = tuple(float(value) for value in box.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    if len(values) != 4:
        values = (0.0, 0.0, 0.0, 0.0)
    return {"xyxy": values, "label": str(box.get("label", "object"))}


def _fusion_payload(detection: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = detection.get("metadata", {}) if isinstance(detection.get("metadata"), Mapping) else {}
    fusion = metadata.get("fusion", {}) if isinstance(metadata.get("fusion"), Mapping) else {}
    return fusion


def _box_iou(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = tuple(float(value) for value in a.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    bx1, by1, bx2, by2 = tuple(float(value) for value in b.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    intersection_w = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    intersection_h = max(min(ay2, by2) - max(ay1, by1), 0.0)
    intersection = intersection_w * intersection_h
    union = _box_area(a) + _box_area(b) - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _box_area(box: Mapping[str, Any]) -> float:
    x1, y1, x2, y2 = tuple(float(value) for value in box.get("xyxy", (0.0, 0.0, 0.0, 0.0)))
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "case"


def _maybe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _maybe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
