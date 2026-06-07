"""Visual casebook rollup for CFA/LensPSF detector sweeps."""

from __future__ import annotations

import argparse
import copy
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .casebook import CASE_CATEGORIES, build_casebook_from_path, write_casebook
from .types import json_ready


SUMMARY_FILENAME = "cfa_lenspsf_casebook_summary.json"
SWEEP_SUMMARY_FILENAME = "cfa_lenspsf_detector_sweep_summary.json"
DEFAULT_BASELINE_INPUT = "perception_fusion_rgb_aux"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a visual casebook across a CFA/LensPSF detector sweep.")
    parser.add_argument("sweep", help="CFA/LensPSF detector sweep summary path or directory.")
    parser.add_argument("--baseline-input", default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--target-input", default=None, help="Target input. Defaults to the first calibrated input in each condition.")
    parser.add_argument("--max-cases-per-category", type=int, default=2)
    parser.add_argument("--max-showcase-cases", type=int, default=48)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output-dir", default="reports/perception_cfa_lenspsf_casebook")
    args = parser.parse_args(argv)

    summary = build_cfa_lenspsf_casebook_from_path(
        args.sweep,
        baseline_input=str(args.baseline_input),
        target_input=None if args.target_input is None else str(args.target_input),
        max_cases_per_category=int(args.max_cases_per_category),
        max_showcase_cases=int(args.max_showcase_cases),
        match_iou=float(args.match_iou),
    )
    html_path = write_cfa_lenspsf_casebook(summary, args.output_dir)
    written = json.loads((html_path.parent / SUMMARY_FILENAME).read_text())
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": written["status"],
                    "condition_count": written["condition_count"],
                    "selected_case_count": written["selected_case_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_cfa_lenspsf_casebook_from_path(
    sweep: str | Path,
    *,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str | None = None,
    max_cases_per_category: int = 2,
    max_showcase_cases: int = 48,
    match_iou: float = 0.50,
) -> Dict[str, Any]:
    summary_path = _summary_path(sweep, SWEEP_SUMMARY_FILENAME)
    sweep_summary = json.loads(summary_path.read_text())
    return build_cfa_lenspsf_casebook(
        sweep_summary,
        sweep_summary_path=summary_path,
        baseline_input=baseline_input,
        target_input=target_input,
        max_cases_per_category=max_cases_per_category,
        max_showcase_cases=max_showcase_cases,
        match_iou=match_iou,
    )


def build_cfa_lenspsf_casebook(
    sweep_summary: Mapping[str, Any],
    *,
    sweep_summary_path: str | Path,
    baseline_input: str = DEFAULT_BASELINE_INPUT,
    target_input: str | None = None,
    max_cases_per_category: int = 2,
    max_showcase_cases: int = 48,
    match_iou: float = 0.50,
) -> Dict[str, Any]:
    sweep_path = Path(sweep_summary_path).expanduser()
    sweep_dir = sweep_path.parent
    conditions = []
    for run in sweep_summary.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        selected_target = _select_target_input(run, preferred=target_input)
        report_path = _condition_report_path(sweep_dir, run)
        casebook = (
            build_casebook_from_path(
                report_path,
                baseline_input=str(baseline_input),
                target_input=selected_target,
                max_cases_per_category=int(max_cases_per_category),
                match_iou=float(match_iou),
            )
            if selected_target and report_path.exists()
            else None
        )
        conditions.append(_condition_summary(run, report_path=report_path, baseline_input=str(baseline_input), target_input=selected_target, casebook=casebook))
    return _summary(
        source_sweep=sweep_path,
        source_sweep_html=sweep_dir / "index.html",
        sweep_summary=sweep_summary,
        baseline_input=baseline_input,
        target_input=target_input,
        max_cases_per_category=max_cases_per_category,
        max_showcase_cases=max_showcase_cases,
        match_iou=match_iou,
        conditions=conditions,
    )


def write_cfa_lenspsf_casebook(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    materialized: Dict[str, Any] = copy.deepcopy(dict(summary))
    conditions_dir = destination / "conditions"
    for index, condition in enumerate(materialized.get("conditions", ())):
        if not isinstance(condition, dict):
            continue
        casebook = condition.get("casebook")
        if not isinstance(casebook, Mapping):
            continue
        condition_dir = conditions_dir / f"{index + 1:03d}_{_safe_id(str(condition.get('run_id', 'condition')))}"
        html_path = write_casebook(casebook, condition_dir)
        written_summary_path = html_path.parent / "casebook_summary.json"
        written_summary = json.loads(written_summary_path.read_text())
        condition["casebook_html"] = str(html_path)
        condition["casebook_summary"] = str(written_summary_path)
        condition["casebook"] = written_summary
        condition["status"] = str(written_summary.get("status", condition.get("status", "")))
        condition["selected_case_count"] = int(written_summary.get("selected_case_count", condition.get("selected_case_count", 0)))
    source_sweep_html = materialized.get("source_sweep_html")
    refreshed = _summary(
        source_sweep=Path(str(materialized.get("source_sweep", ""))),
        source_sweep_html=Path(str(source_sweep_html)) if source_sweep_html else None,
        sweep_summary={
            "run_count": materialized.get("expected_condition_count", 0),
            "cfa_patterns": materialized.get("cfa_patterns", ()),
            "psf_sigmas": materialized.get("psf_sigmas", ()),
        },
        baseline_input=str(materialized.get("baseline_input", "")),
        target_input=str(materialized.get("target_input", "")) or None,
        max_cases_per_category=int(materialized.get("max_cases_per_category", 0)),
        max_showcase_cases=int(materialized.get("max_showcase_cases", 0)),
        match_iou=float(materialized.get("match_iou", 0.5)),
        conditions=[row for row in materialized.get("conditions", ()) if isinstance(row, Mapping)],
    )
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(refreshed), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(refreshed, destination))
    return html_path


def _summary(
    *,
    source_sweep: Path,
    source_sweep_html: Path | None,
    sweep_summary: Mapping[str, Any],
    baseline_input: str,
    target_input: str | None,
    max_cases_per_category: int,
    max_showcase_cases: int,
    match_iou: float,
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    category_totals = _category_totals(conditions)
    selected_case_count = sum(int(payload.get("selected_case_count", 0)) for payload in category_totals.values())
    checks = _checks(
        conditions,
        category_totals=category_totals,
        selected_case_count=selected_case_count,
        expected_condition_count=int(sweep_summary.get("run_count", 0)),
        expected_cfas=[str(value) for value in sweep_summary.get("cfa_patterns", ())],
        expected_psf=[float(value) for value in sweep_summary.get("psf_sigmas", ())],
    )
    return {
        "name": "CFA/LensPSF visual casebook",
        "source_sweep": str(source_sweep),
        "source_sweep_html": str(source_sweep_html) if source_sweep_html is not None and source_sweep_html.exists() else "",
        "baseline_input": str(baseline_input),
        "target_input": "" if target_input is None else str(target_input),
        "max_cases_per_category": int(max_cases_per_category),
        "max_showcase_cases": int(max_showcase_cases),
        "match_iou": float(match_iou),
        "condition_count": len(conditions),
        "expected_condition_count": int(sweep_summary.get("run_count", 0)),
        "selected_case_count": selected_case_count,
        "cfa_patterns": [str(value) for value in sweep_summary.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in sweep_summary.get("psf_sigmas", ())],
        "category_totals": category_totals,
        "checks": checks,
        "conditions": [dict(row) for row in conditions],
        "showcase_cases": _showcase_cases(conditions, max_count=int(max_showcase_cases)),
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This report builds reviewable visual casebooks for every CFA/LensPSF condition in the native detector sweep. "
            "It is meant to show representative FP reductions, TP losses, CFA/PSF wins, and counterexamples with the same detector recipe."
        ),
        "claim_boundary": (
            "This is qualitative condition-slice evidence. It helps explain the detector/proposal metrics, but it does not replace "
            "held-out claim gates, larger adverse-condition datasets, or a trained RGB+Aux DNN evaluation."
        ),
    }


def _condition_summary(
    run: Mapping[str, Any],
    *,
    report_path: Path,
    baseline_input: str,
    target_input: str,
    casebook: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    raw_summary = run.get("raw_condition_summary", {}) if isinstance(run.get("raw_condition_summary"), Mapping) else {}
    if casebook is None:
        return {
            "run_id": str(run.get("run_id", "")),
            "report": str(report_path),
            "cfa_pattern": str(run.get("cfa_pattern", "")),
            "psf_sigma": _maybe_float(run.get("psf_sigma")),
            "baseline_input": baseline_input,
            "target_input": target_input,
            "status": "missing_casebook",
            "selected_case_count": 0,
            "pattern_remapped_fraction": _maybe_float(raw_summary.get("pattern_remapped_fraction")),
            "true_sensor_cfa_mosaic_fraction": _maybe_float(raw_summary.get("true_sensor_cfa_mosaic_fraction")),
            "native_raw_input_fraction": _maybe_float(raw_summary.get("native_raw_input_fraction")),
            "raw_derived_png_input_fraction": _maybe_float(raw_summary.get("raw_derived_png_input_fraction")),
            "camerae2e_used_fraction": _maybe_float(raw_summary.get("camerae2e_used_fraction")),
        }
    aggregate = casebook.get("aggregate", {}) if isinstance(casebook.get("aggregate"), Mapping) else {}
    return {
        "run_id": str(run.get("run_id", "")),
        "report": str(report_path),
        "cfa_pattern": str(run.get("cfa_pattern", "")),
        "psf_sigma": _maybe_float(run.get("psf_sigma")),
        "baseline_input": baseline_input,
        "target_input": target_input,
        "status": str(casebook.get("status", "")),
        "sample_count": int(casebook.get("sample_count", run.get("sample_count", 0))),
        "selected_case_count": int(casebook.get("selected_case_count", 0)),
        "tp_delta_count": int(aggregate.get("tp_delta_count", 0)),
        "fp_delta_count": int(aggregate.get("fp_delta_count", 0)),
        "pattern_remapped_fraction": _maybe_float(raw_summary.get("pattern_remapped_fraction")),
        "true_sensor_cfa_mosaic_fraction": _maybe_float(raw_summary.get("true_sensor_cfa_mosaic_fraction")),
        "native_raw_input_fraction": _maybe_float(raw_summary.get("native_raw_input_fraction")),
        "raw_derived_png_input_fraction": _maybe_float(raw_summary.get("raw_derived_png_input_fraction")),
        "camerae2e_used_fraction": _maybe_float(raw_summary.get("camerae2e_used_fraction")),
        "casebook": dict(casebook),
    }


def _category_totals(conditions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    totals = {name: {"case_count": 0, "selected_case_count": 0} for name in CASE_CATEGORIES}
    for condition in conditions:
        casebook = condition.get("casebook", {}) if isinstance(condition.get("casebook"), Mapping) else {}
        categories = casebook.get("categories", {}) if isinstance(casebook.get("categories"), Mapping) else {}
        for name in CASE_CATEGORIES:
            payload = categories.get(name, {}) if isinstance(categories.get(name), Mapping) else {}
            totals[name]["case_count"] += int(payload.get("case_count", 0))
            totals[name]["selected_case_count"] += int(payload.get("selected_case_count", 0))
    return totals


def _checks(
    conditions: Sequence[Mapping[str, Any]],
    *,
    category_totals: Mapping[str, Mapping[str, Any]],
    selected_case_count: int,
    expected_condition_count: int,
    expected_cfas: Sequence[str],
    expected_psf: Sequence[float],
) -> Tuple[Dict[str, Any], ...]:
    condition_count = len(conditions)
    cfas = {str(row.get("cfa_pattern", "")) for row in conditions if row.get("cfa_pattern")}
    psf = {float(row.get("psf_sigma", 0.0)) for row in conditions if row.get("psf_sigma") is not None}
    success_count = int(category_totals.get("fp_reduction_success", {}).get("selected_case_count", 0))
    counter_count = sum(int(category_totals.get(name, {}).get("selected_case_count", 0)) for name in ("recall_tradeoff", "recall_loss_failure", "fp_regression_failure"))
    true_native_rows = [row for row in conditions if _is_true_native_condition(row)]
    simulated_native_rows = [row for row in conditions if _is_simulated_native_condition(row)]
    return (
        {
            "id": "condition_casebooks_available",
            "status": "pass" if condition_count == int(expected_condition_count) and expected_condition_count > 0 else "fail",
            "evidence": f"conditions={condition_count} expected={expected_condition_count}",
        },
        {
            "id": "casebook_covers_cfa_psf_grid",
            "status": "pass" if cfas == set(expected_cfas) and psf == {float(value) for value in expected_psf} else "fail",
            "evidence": f"cfa={','.join(sorted(cfas)) or 'none'} psf={','.join(_fmt(value) for value in sorted(psf)) or 'none'}",
        },
        {
            "id": "casebook_uses_native_cfa_rows",
            "status": "pass" if len(true_native_rows) == condition_count and condition_count > 0 else "fail",
            "evidence": f"true_native={len(true_native_rows)}/{condition_count} simulated_native={len(simulated_native_rows)}",
        },
        {
            "id": "casebook_separates_simulated_native_rows",
            "status": "pass" if not simulated_native_rows else "warning",
            "evidence": f"simulated_native={len(simulated_native_rows)}/{condition_count}",
        },
        {
            "id": "casebook_has_selected_cases",
            "status": "pass" if int(selected_case_count) > 0 else "fail",
            "evidence": f"selected_cases={int(selected_case_count)}",
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
    )


def _is_true_native_condition(row: Mapping[str, Any]) -> bool:
    if row.get("pattern_remapped_fraction") != 0.0 or row.get("true_sensor_cfa_mosaic_fraction") != 1.0:
        return False
    raw_derived = row.get("raw_derived_png_input_fraction")
    camerae2e = row.get("camerae2e_used_fraction")
    native_raw = row.get("native_raw_input_fraction")
    if raw_derived is None and camerae2e is None and native_raw is None:
        return True
    return float(native_raw or 0.0) == 1.0 and float(raw_derived or 0.0) == 0.0


def _is_simulated_native_condition(row: Mapping[str, Any]) -> bool:
    if row.get("pattern_remapped_fraction") != 0.0 or row.get("true_sensor_cfa_mosaic_fraction") != 1.0:
        return False
    return float(row.get("raw_derived_png_input_fraction") or 0.0) > 0.0 or float(row.get("camerae2e_used_fraction") or 0.0) > 0.0


def _showcase_cases(conditions: Sequence[Mapping[str, Any]], *, max_count: int) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for condition in conditions:
        casebook = condition.get("casebook", {}) if isinstance(condition.get("casebook"), Mapping) else {}
        categories = casebook.get("categories", {}) if isinstance(casebook.get("categories"), Mapping) else {}
        for category_name, category in categories.items():
            if not isinstance(category, Mapping):
                continue
            for case in category.get("cases", ()):
                if not isinstance(case, Mapping):
                    continue
                rows.append(
                    {
                        "run_id": condition.get("run_id"),
                        "condition_casebook_html": condition.get("casebook_html", ""),
                        "cfa_pattern": condition.get("cfa_pattern"),
                        "psf_sigma": condition.get("psf_sigma"),
                        "category": str(category_name),
                        "sample_id": str(case.get("sample_id", "")),
                        "visual_path": str(case.get("visual_path", "")),
                        "fp_delta@0.50": int(case.get("fp_delta@0.50", 0)),
                        "tp_delta@0.50": int(case.get("tp_delta@0.50", 0)),
                    }
                )
    rows.sort(key=lambda row: (str(row.get("category", "")), str(row.get("run_id", "")), str(row.get("sample_id", ""))))
    return tuple(rows[: max(int(max_count), 0)])


def _select_target_input(run: Mapping[str, Any], *, preferred: str | None) -> str:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}
    if preferred and preferred in metrics:
        return str(preferred)
    for name in metrics:
        if str(name).startswith("perception_calibrated"):
            return str(name)
    if "perception_fusion_rgb_aux" in metrics:
        return "perception_fusion_rgb_aux"
    return next(iter(metrics), "")


def _condition_report_path(sweep_dir: Path, run: Mapping[str, Any]) -> Path:
    raw_report = str(run.get("report", ""))
    if not raw_report:
        return sweep_dir / str(run.get("run_id", "")) / "comparison_summary.json"
    path = sweep_dir / raw_report
    if path.name == "index.html":
        return path.with_name("comparison_summary.json")
    if path.is_dir():
        return path / "comparison_summary.json"
    return path


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    aggregate_rows = "".join(_category_row(name, payload) for name, payload in summary.get("category_totals", {}).items() if isinstance(payload, Mapping))
    condition_rows = "".join(_condition_row(row, destination) for row in summary.get("conditions", ()) if isinstance(row, Mapping))
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{html_lib.escape(str(row.get('status', '')))}\">{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in summary.get("checks", ())
        if isinstance(row, Mapping)
    )
    showcase = "".join(_showcase_card(row, destination) for row in summary.get("showcase_cases", ()) if isinstance(row, Mapping))
    if not showcase:
        showcase = "<p>No showcase cases were selected.</p>"
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP CFA/LensPSF Visual Casebook</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e8f3f1; position: sticky; top: 0; }}
    a {{ color: #155e75; }}
    img {{ max-width: 100%; border: 1px solid #d8ded7; background: #111827; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    .pass {{ color: #047857; font-weight: 650; }}
    .fail, .warning {{ color: #b91c1c; font-weight: 650; }}
    .showcase {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .case {{ background: white; border: 1px solid #d8ded7; padding: 10px; }}
    code {{ background: #eef2f1; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP CFA/LensPSF Visual Casebook</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>.
  Conditions={int(summary.get('condition_count', 0))}/{int(summary.get('expected_condition_count', 0))}.
  Selected cases={int(summary.get('selected_case_count', 0))}. Source sweep: {_source_sweep_link(summary, destination)}</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Category Totals</h2>
  <table><thead><tr><th>Category</th><th>Total Cases</th><th>Selected</th></tr></thead><tbody>{aggregate_rows}</tbody></table>
  <h2>Condition Casebooks</h2>
  <table><thead><tr><th>Run</th><th>CFA</th><th>PSF</th><th>Status</th><th>Selected</th><th>dTP</th><th>dFP</th><th>Native CFA</th><th>Casebook</th></tr></thead><tbody>{condition_rows}</tbody></table>
  <h2>Showcase</h2>
  <div class=\"showcase\">{showcase}</div>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _category_row(name: str, payload: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{int(payload.get('case_count', 0))}</td>"
        f"<td>{int(payload.get('selected_case_count', 0))}</td>"
        "</tr>"
    )


def _condition_row(row: Mapping[str, Any], destination: Path) -> str:
    html_path = str(row.get("casebook_html", ""))
    link = ""
    if html_path:
        relative = os.path.relpath(html_path, start=str(destination))
        link = f"<a href=\"{html_lib.escape(relative)}\">open</a>"
    native = row.get("pattern_remapped_fraction") == 0.0 and row.get("true_sensor_cfa_mosaic_fraction") == 1.0
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('run_id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{int(row.get('selected_case_count', 0))}</td>"
        f"<td>{int(row.get('tp_delta_count', 0))}</td>"
        f"<td>{int(row.get('fp_delta_count', 0))}</td>"
        f"<td>{html_lib.escape(str(native))}</td>"
        f"<td>{link}</td>"
        "</tr>"
    )


def _showcase_card(row: Mapping[str, Any], destination: Path) -> str:
    visual_path = str(row.get("visual_path", ""))
    image = "<p>No visual asset.</p>"
    if visual_path:
        relative = os.path.relpath(visual_path, start=str(destination))
        image = f"<img src=\"{html_lib.escape(relative)}\" alt=\"{html_lib.escape(str(row.get('sample_id', 'case')))}\">"
    casebook_link = ""
    casebook_html = str(row.get("condition_casebook_html", ""))
    if casebook_html:
        relative = os.path.relpath(casebook_html, start=str(destination))
        casebook_link = f"<a href=\"{html_lib.escape(relative)}\">condition casebook</a>"
    return (
        "<div class=\"case\">"
        f"<strong>{html_lib.escape(str(row.get('run_id', '')))}</strong><br>"
        f"CFA={html_lib.escape(str(row.get('cfa_pattern', '')))} PSF={_fmt(row.get('psf_sigma'))}<br>"
        f"{html_lib.escape(str(row.get('category', '')))} sample={html_lib.escape(str(row.get('sample_id', '')))}<br>"
        f"dTP={int(row.get('tp_delta@0.50', 0))} dFP={int(row.get('fp_delta@0.50', 0))} {casebook_link}"
        f"{image}"
        "</div>"
    )


def _source_sweep_link(summary: Mapping[str, Any], destination: Path) -> str:
    html_path = str(summary.get("source_sweep_html", ""))
    if not html_path:
        return ""
    relative = os.path.relpath(html_path, start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value)).strip("-") or "item"


if __name__ == "__main__":
    raise SystemExit(main())
