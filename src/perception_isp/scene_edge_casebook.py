"""Visual casebook for scene-edge confidence summaries."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .scene_edge_confidence_suite import SCENE_EDGE_CONFIDENCE_SUMMARY
from .types import json_ready


SUMMARY_FILENAME = "scene_edge_casebook_summary.json"

CASE_CATEGORIES = (
    "rgb_edge_improvement",
    "aux_confidence_success",
    "aux_confidence_counterexample",
    "rgb_edge_regression",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a visual casebook from a scene-edge confidence summary.")
    parser.add_argument("scene_edge_report", help="scene_edge_confidence_summary.json path or report directory.")
    parser.add_argument("--max-cases-per-category", type=int, default=6)
    parser.add_argument("--output-dir", default="reports/perception_scene_edge_casebook")
    args = parser.parse_args(argv)

    summary = build_scene_edge_casebook_from_path(
        args.scene_edge_report,
        max_cases_per_category=int(args.max_cases_per_category),
    )
    html_path = write_scene_edge_casebook(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "source_case_count": summary["source_case_count"],
                    "selected_case_count": summary["selected_case_count"],
                    "category_counts": {key: value["selected_case_count"] for key, value in summary["categories"].items()},
                }
            ),
            indent=2,
        )
    )
    return 0


def build_scene_edge_casebook_from_path(
    report: str | Path,
    *,
    max_cases_per_category: int = 6,
) -> Dict[str, Any]:
    summary_path = _summary_path(report)
    data = json.loads(summary_path.read_text())
    return build_scene_edge_casebook(
        data,
        source_summary=summary_path,
        max_cases_per_category=max_cases_per_category,
    )


def build_scene_edge_casebook(
    summary: Mapping[str, Any],
    *,
    source_summary: str | Path | None = None,
    max_cases_per_category: int = 6,
) -> Dict[str, Any]:
    source_path = Path(source_summary).expanduser() if source_summary is not None else None
    source_dir = source_path.parent if source_path is not None else Path(".")
    source_cases = [row for row in summary.get("cases", ()) if isinstance(row, Mapping)]
    candidate_cases: Dict[str, list[Dict[str, Any]]] = {name: [] for name in CASE_CATEGORIES}
    for row in source_cases:
        case = _case_summary(row, source_dir=source_dir)
        for category in _case_categories(case):
            candidate_cases[category].append({**case, "category": category})

    categories: Dict[str, Dict[str, Any]] = {}
    for name in CASE_CATEGORIES:
        rows = sorted(candidate_cases[name], key=lambda item: _case_sort_key(name, item))
        selected = rows[: max(int(max_cases_per_category), 0)]
        categories[name] = {
            "case_count": len(candidate_cases[name]),
            "selected_case_count": len(selected),
            "cases": selected,
        }
    selected_case_count = sum(int(row["selected_case_count"]) for row in categories.values())
    checks = _checks(categories, selected_case_count=selected_case_count)
    return {
        "name": "Scene-edge confidence success/failure casebook",
        "source_summary": "" if source_path is None else str(source_path),
        "source_report": "" if source_path is None else _sibling_html(source_path),
        "source_status": str(summary.get("status", "")),
        "source_failed_checks": [str(row.get("id", "")) for row in summary.get("checks", ()) if isinstance(row, Mapping) and str(row.get("status", "")) != "pass"],
        "source_case_count": len(source_cases),
        "selected_case_count": selected_case_count,
        "categories": categories,
        "checks": checks,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This casebook selects reviewable scene-edge successes and counterexamples from the same scene-edge confidence summary. "
            "It highlights where PerceptionISP RGB edge evidence improves over HumanISP and where aux edge confidence fails to track the high-information reference."
        ),
        "claim_boundary": (
            "Use this as qualitative front-end evidence only. It is not object-detection accuracy and does not replace held-out native RAW gates."
        ),
    }


def write_scene_edge_casebook(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    materialized = json_ready(summary)
    for payload in (materialized.get("categories", {}) if isinstance(materialized, Mapping) else {}).values():
        if not isinstance(payload, Mapping):
            continue
        for case in payload.get("cases", ()):
            if isinstance(case, dict):
                _materialize_case_assets(case, destination)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(materialized, indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(materialized, destination))
    return html_path


def _summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / SCENE_EDGE_CONFIDENCE_SUMMARY
    if not candidate.exists():
        raise FileNotFoundError(f"scene-edge confidence summary not found: {candidate}")
    return candidate


def _case_summary(row: Mapping[str, Any], *, source_dir: Path) -> Dict[str, Any]:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    assets = row.get("assets", {}) if isinstance(row.get("assets"), Mapping) else {}
    return {
        "id": str(row.get("id", "")),
        "source": str(row.get("source", "")),
        "cfa_pattern": str(row.get("cfa_pattern", "")),
        "psf_sigma": _maybe_float(row.get("psf_sigma")),
        "source_edge_fraction": _maybe_float(metrics.get("source_edge_fraction")),
        "human_rgb_proxy_source_edge_f1": _maybe_float(metrics.get("human_rgb_proxy_source_edge_f1")),
        "perception_rgb_proxy_source_edge_f1": _maybe_float(metrics.get("perception_rgb_proxy_source_edge_f1")),
        "perception_rgb_minus_human_source_edge_f1": _maybe_float(metrics.get("perception_rgb_minus_human_source_edge_f1")),
        "perception_aux_confidence_source_edge_f1": _maybe_float(metrics.get("perception_aux_confidence_source_edge_f1")),
        "perception_aux_confidence_minus_human_source_edge_f1": _maybe_float(metrics.get("perception_aux_confidence_minus_human_source_edge_f1")),
        "perception_aux_confidence_scene_edge_separation": _maybe_float(metrics.get("perception_aux_confidence_scene_edge_separation")),
        "perception_aux_strength_source_edge_f1": _maybe_float(metrics.get("perception_aux_strength_source_edge_f1")),
        "perception_aux_strength_minus_human_source_edge_f1": _maybe_float(metrics.get("perception_aux_strength_minus_human_source_edge_f1")),
        "asset_sources": {
            str(name): str((source_dir / str(path)).resolve()) if path else ""
            for name, path in assets.items()
            if name
        },
    }


def _case_categories(case: Mapping[str, Any]) -> tuple[str, ...]:
    categories: list[str] = []
    rgb_delta = _float_or_zero(case.get("perception_rgb_minus_human_source_edge_f1"))
    aux_delta = _float_or_zero(case.get("perception_aux_confidence_minus_human_source_edge_f1"))
    aux_separation = _float_or_zero(case.get("perception_aux_confidence_scene_edge_separation"))
    if rgb_delta > 0.0:
        categories.append("rgb_edge_improvement")
    if aux_delta > 0.0 and aux_separation > 0.0:
        categories.append("aux_confidence_success")
    if aux_delta < 0.0 or aux_separation <= 0.0:
        categories.append("aux_confidence_counterexample")
    if rgb_delta < 0.0:
        categories.append("rgb_edge_regression")
    return tuple(categories)


def _case_sort_key(category: str, case: Mapping[str, Any]) -> tuple[float, str]:
    if category == "rgb_edge_improvement":
        return (-_float_or_zero(case.get("perception_rgb_minus_human_source_edge_f1")), str(case.get("id", "")))
    if category == "aux_confidence_success":
        score = _float_or_zero(case.get("perception_aux_confidence_minus_human_source_edge_f1")) + _float_or_zero(
            case.get("perception_aux_confidence_scene_edge_separation")
        )
        return (-score, str(case.get("id", "")))
    if category == "aux_confidence_counterexample":
        score = min(
            _float_or_zero(case.get("perception_aux_confidence_minus_human_source_edge_f1")),
            _float_or_zero(case.get("perception_aux_confidence_scene_edge_separation")),
        )
        return (score, str(case.get("id", "")))
    return (_float_or_zero(case.get("perception_rgb_minus_human_source_edge_f1")), str(case.get("id", "")))


def _checks(categories: Mapping[str, Any], *, selected_case_count: int) -> list[Dict[str, Any]]:
    aux_successes = int(categories.get("aux_confidence_success", {}).get("selected_case_count", 0)) if isinstance(categories.get("aux_confidence_success"), Mapping) else 0
    aux_counterexamples = (
        int(categories.get("aux_confidence_counterexample", {}).get("selected_case_count", 0))
        if isinstance(categories.get("aux_confidence_counterexample"), Mapping)
        else 0
    )
    rgb_improvements = int(categories.get("rgb_edge_improvement", {}).get("selected_case_count", 0)) if isinstance(categories.get("rgb_edge_improvement"), Mapping) else 0
    return [
        {
            "id": "scene_edge_casebook_has_selected_cases",
            "status": "pass" if selected_case_count > 0 else "fail",
            "evidence": f"selected_cases={selected_case_count}",
        },
        {
            "id": "scene_edge_casebook_includes_rgb_improvements",
            "status": "pass" if rgb_improvements > 0 else "fail",
            "evidence": f"rgb_improvements={rgb_improvements}",
        },
        {
            "id": "scene_edge_casebook_includes_aux_successes",
            "status": "pass" if aux_successes > 0 else "fail",
            "evidence": f"aux_successes={aux_successes}",
        },
        {
            "id": "scene_edge_casebook_includes_aux_counterexamples",
            "status": "pass" if aux_counterexamples > 0 else "fail",
            "evidence": f"aux_counterexamples={aux_counterexamples}",
        },
    ]


def _materialize_case_assets(case: Dict[str, Any], destination: Path) -> None:
    asset_sources = case.get("asset_sources", {}) if isinstance(case.get("asset_sources"), Mapping) else {}
    assets_dir = destination / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for name, source in asset_sources.items():
        source_path = Path(str(source))
        if not source_path.is_file():
            continue
        filename = f"{_safe_name(str(case.get('category', 'case')))}_{_safe_name(str(case.get('id', 'case')))}_{_safe_name(str(name))}.png"
        target = assets_dir / filename
        shutil.copy2(source_path, target)
        copied[str(name)] = str(target.relative_to(destination))
    case["assets"] = copied


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    status = str(summary.get("status", ""))
    category_rows = "".join(_category_row(name, payload) for name, payload in (summary.get("categories", {}) if isinstance(summary.get("categories"), Mapping) else {}).items())
    case_sections = "".join(
        _category_cases(name, payload, destination)
        for name, payload in (summary.get("categories", {}) if isinstance(summary.get("categories"), Mapping) else {}).items()
        if isinstance(payload, Mapping)
    )
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    source_link = _source_link(summary, destination)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Scene-Edge Casebook</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    img {{ width: 116px; height: auto; margin-right: 4px; border: 1px solid #d8ded7; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail, .warning {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Scene-Edge Casebook</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</code>. Source cases: {int(summary.get('source_case_count', 0))}. Selected rows: {int(summary.get('selected_case_count', 0))}. Source report: {source_link}</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Status</th><th>Check</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Categories</h2>
  <table><thead><tr><th>Category</th><th>Available</th><th>Selected</th><th>Meaning</th></tr></thead><tbody>{category_rows}</tbody></table>
  {case_sections}
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
        f"<td>{html_lib.escape(_category_description(name))}</td>"
        "</tr>"
    )


def _category_cases(name: str, payload: Mapping[str, Any], destination: Path) -> str:
    rows = "".join(_case_row(row, destination) for row in payload.get("cases", ()) if isinstance(row, Mapping))
    if not rows:
        rows = '<tr><td colspan="12">No selected cases.</td></tr>'
    return (
        f"<h2>{html_lib.escape(_category_title(name))}</h2>"
        "<table><thead><tr><th>Case</th><th>Visuals</th><th>Source Edge</th><th>Human F1</th><th>Perception RGB F1</th>"
        "<th>RGB Delta</th><th>Aux Confidence F1</th><th>Aux Confidence Delta</th><th>Aux Confidence Separation</th>"
        "<th>Aux Strength F1</th><th>Aux Strength Delta</th><th>CFA / PSF</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _case_row(case: Mapping[str, Any], destination: Path) -> str:
    assets = case.get("assets", {}) if isinstance(case.get("assets"), Mapping) else {}
    thumbs = " ".join(
        _asset_img(path, destination)
        for path in (
            assets.get("reference_rgb"),
            assets.get("source_edge"),
            assets.get("human_rgb"),
            assets.get("perception_rgb"),
            assets.get("aux_edge_confidence"),
            assets.get("aux_edge_strength"),
        )
        if path
    )
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(case.get('id', '')))}</code><br>{html_lib.escape(str(case.get('source', '')))}</td>"
        f"<td>{thumbs}</td>"
        f"<td>{_fmt(case.get('source_edge_fraction'))}</td>"
        f"<td>{_fmt(case.get('human_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(case.get('perception_rgb_proxy_source_edge_f1'))}</td>"
        f"<td>{_fmt(case.get('perception_rgb_minus_human_source_edge_f1'), signed=True)}</td>"
        f"<td>{_fmt(case.get('perception_aux_confidence_source_edge_f1'))}</td>"
        f"<td>{_fmt(case.get('perception_aux_confidence_minus_human_source_edge_f1'), signed=True)}</td>"
        f"<td>{_fmt(case.get('perception_aux_confidence_scene_edge_separation'), signed=True)}</td>"
        f"<td>{_fmt(case.get('perception_aux_strength_source_edge_f1'))}</td>"
        f"<td>{_fmt(case.get('perception_aux_strength_minus_human_source_edge_f1'), signed=True)}</td>"
        f"<td><code>{html_lib.escape(str(case.get('cfa_pattern', '')))}</code> / {_fmt(case.get('psf_sigma'))}</td>"
        "</tr>"
    )


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _asset_img(path: Any, destination: Path) -> str:
    raw = str(path)
    href = raw if os.path.isabs(raw) else os.path.relpath(str(destination / raw), start=str(destination))
    return f"<img src=\"{html_lib.escape(href)}\" alt=\"{html_lib.escape(Path(raw).stem)}\">"


def _source_link(summary: Mapping[str, Any], destination: Path) -> str:
    source = str(summary.get("source_report", ""))
    if not source:
        return "none"
    href = source if os.path.isabs(source) else os.path.relpath(source, start=str(destination))
    return f"<a href=\"{html_lib.escape(href)}\">source</a>"


def _sibling_html(path: Path) -> str:
    html_path = path.with_name("index.html")
    return str(html_path) if html_path.exists() else ""


def _category_title(name: str) -> str:
    return {
        "rgb_edge_improvement": "Perception RGB Edge Improvements",
        "aux_confidence_success": "Aux Confidence Successes",
        "aux_confidence_counterexample": "Aux Confidence Counterexamples",
        "rgb_edge_regression": "Perception RGB Edge Regressions",
    }.get(name, name.replace("_", " ").title())


def _category_description(name: str) -> str:
    return {
        "rgb_edge_improvement": "PerceptionISP RGB edge F1 is higher than the HumanISP RGB edge proxy.",
        "aux_confidence_success": "Aux confidence improves F1 over HumanISP and is higher on source edges than off edges.",
        "aux_confidence_counterexample": "Aux confidence loses F1 versus HumanISP or is not higher on source edges than off edges.",
        "rgb_edge_regression": "PerceptionISP RGB edge F1 is lower than the HumanISP RGB edge proxy.",
    }.get(name, "")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value)
    return safe.strip("_") or "item"


def _maybe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _float_or_zero(value: Any) -> float:
    result = _maybe_float(value)
    return 0.0 if result is None else float(result)


def _fmt(value: Any, *, signed: bool = False) -> str:
    number = _maybe_float(value)
    if number is None:
        return "n/a"
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
