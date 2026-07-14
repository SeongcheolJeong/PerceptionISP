"""Synthetic CFA stress sweep for PerceptionISP front-end signal analysis."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from perception_isp.core.pipeline import PerceptionISPPipeline
from perception_isp.core.synthetic import make_synthetic_raw
from perception_isp.core.types import PerceptionISPConfig, RawFrame, json_ready


CFA_STRESS_SUMMARY = "cfa_stress_sweep_summary.json"
DEFAULT_CFAS = ("RGGB", "GRBG", "RCCB", "RGBIR", "MONO")
DEFAULT_CONDITIONS = ("nominal_hdr", "low_light", "glare", "low_mtf")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic CFA stress-sweep report for PerceptionISP signals.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cfa", action="append", default=[], help="CFA pattern to include. Repeats are ignored.")
    parser.add_argument("--condition", action="append", default=[], help="Condition to include: nominal_hdr, low_light, glare, low_mtf.")
    parser.add_argument("--output-dir", default="reports/perception_cfa_stress_sweep_synthetic")
    args = parser.parse_args(argv)

    summary = build_cfa_stress_sweep(
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
        cfa_patterns=tuple(args.cfa) or DEFAULT_CFAS,
        conditions=tuple(args.condition) or DEFAULT_CONDITIONS,
    )
    html_path = write_cfa_stress_sweep(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / CFA_STRESS_SUMMARY),
                    "case_count": len(summary["cases"]),
                    "condition_count": len(summary["condition_rankings"]),
                    "status": summary["status"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_cfa_stress_sweep(
    *,
    width: int = 128,
    height: int = 72,
    seed: int = 13,
    cfa_patterns: Sequence[str] = DEFAULT_CFAS,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    config: PerceptionISPConfig | None = None,
) -> Dict[str, Any]:
    cfas = tuple(dict.fromkeys(str(value).upper().replace("-", "") for value in cfa_patterns))
    condition_ids = tuple(dict.fromkeys(str(value).lower().replace("-", "_") for value in conditions))
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    cases = [
        _run_case(cfa, condition, _make_condition_raw(cfa, condition, width=int(width), height=int(height), seed=int(seed)), pipeline=pipeline)
        for condition in condition_ids
        for cfa in cfas
    ]
    rankings = [_condition_ranking(condition, [case for case in cases if case["condition"] == condition]) for condition in condition_ids]
    support = _support_summary(cases)
    return {
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "cfa_patterns": list(cfas),
        "conditions": list(condition_ids),
        "cases": cases,
        "condition_rankings": rankings,
        "support": support,
        "status": "pass" if bool(support["all_finite"]) and bool(support["all_supported"]) else "fail",
        "interpretation": (
            "This synthetic CFA stress sweep ranks PerceptionISP front-end signal quality under controlled stressors. "
            "It is evidence about CFA-dependent aux/confidence behavior, not detector performance or a product sensor recommendation."
        ),
    }


def write_cfa_stress_sweep(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / CFA_STRESS_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _make_condition_raw(cfa: str, condition: str, *, width: int, height: int, seed: int) -> RawFrame:
    if condition == "nominal_hdr":
        return make_synthetic_raw(width=width, height=height, exposures=(1.0, 0.25, 0.0625), cfa_pattern=cfa, seed=seed)
    if condition == "low_light":
        return make_synthetic_raw(width=width, height=height, exposures=(0.08,), cfa_pattern=cfa, seed=seed)
    if condition == "glare":
        return make_synthetic_raw(width=width, height=height, exposures=(2.5, 0.5, 0.125), cfa_pattern=cfa, seed=seed)
    if condition == "low_mtf":
        raw = make_synthetic_raw(width=width, height=height, exposures=(1.0, 0.25, 0.0625), cfa_pattern=cfa, seed=seed)
        exposures = np.asarray(raw.data)
        shape = exposures.shape[-2:] if exposures.ndim == 3 else exposures.shape[:2]
        calibration = replace(raw.calibration, mtf_confidence_map=np.full(shape, 0.25, dtype=np.float64))
        return replace(raw, calibration=calibration, provenance={**dict(raw.provenance), "cfa_stress_low_mtf": True})
    raise ValueError(f"unsupported condition: {condition}")


def _run_case(cfa: str, condition: str, raw: RawFrame, *, pipeline: PerceptionISPPipeline) -> Dict[str, Any]:
    result = pipeline.run(raw)
    metrics = _metrics(result)
    score = _condition_score(condition, metrics)
    return {
        "id": f"{condition}_{cfa.lower()}",
        "condition": str(condition),
        "cfa_pattern": str(result.metadata.get("frame", {}).get("cfa_pattern", cfa)),
        "warnings": list(result.health.get("warnings", ())),
        "metrics": metrics,
        "condition_score": score,
    }


def _metrics(result: Any) -> Dict[str, Any]:
    maps = result.maps
    health = result.health
    clear_or_ir = np.maximum(np.asarray(maps["clear_channel"], dtype=np.float64), np.asarray(maps["ir_channel"], dtype=np.float64))
    finite_outputs = bool(np.isfinite(result.vision_rgb).all() and all(np.isfinite(value).all() for value in maps.values()))
    return {
        "finite_outputs": finite_outputs,
        "snr_map_mean": _mean(maps["snr_map"]),
        "snr_map_p10": _percentile(maps["snr_map"], 10.0),
        "visibility_confidence": float(health.get("visibility_confidence", 0.0)),
        "under_exposure_fraction": float(health.get("under_exposure_fraction", 0.0)),
        "saturation_mean": _mean(maps["saturation"]),
        "over_exposure_fraction": float(health.get("over_exposure_fraction", 0.0)),
        "clipping_distance_mean": _mean(maps["clipping_distance"]),
        "hdr_confidence_mean": _mean(maps["hdr_confidence"]),
        "edge_strength_mean": _mean(maps["edge_strength"]),
        "edge_confidence_mean": _mean(maps["edge_confidence"]),
        "demosaic_confidence_mean": _mean(maps["demosaic_confidence"]),
        "focus_confidence": float(health.get("focus_confidence", 0.0)),
        "color_confidence_mean": _mean(maps["color_confidence"]),
        "clear_or_ir_mean": _mean(clear_or_ir),
        "ir_contamination_mean": _mean(maps["ir_contamination"]),
    }


def _condition_score(condition: str, metrics: Mapping[str, Any]) -> float:
    if condition == "low_light":
        return _average(
            metrics.get("snr_map_mean"),
            metrics.get("visibility_confidence"),
            metrics.get("edge_confidence_mean"),
            metrics.get("demosaic_confidence_mean"),
            1.0 - float(metrics.get("under_exposure_fraction", 1.0)),
        )
    if condition == "glare":
        return _average(
            metrics.get("clipping_distance_mean"),
            metrics.get("visibility_confidence"),
            metrics.get("edge_confidence_mean"),
            metrics.get("demosaic_confidence_mean"),
            1.0 - float(metrics.get("saturation_mean", 1.0)),
            1.0 - float(metrics.get("over_exposure_fraction", 1.0)),
        )
    if condition == "low_mtf":
        return _average(metrics.get("edge_confidence_mean"), metrics.get("demosaic_confidence_mean"), metrics.get("focus_confidence"))
    return _average(
        metrics.get("snr_map_mean"),
        metrics.get("visibility_confidence"),
        metrics.get("edge_confidence_mean"),
        metrics.get("demosaic_confidence_mean"),
        metrics.get("color_confidence_mean"),
    )


def _condition_ranking(condition: str, cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(cases, key=lambda row: float(row.get("condition_score", 0.0)), reverse=True)
    return {
        "condition": str(condition),
        "score_definition": _score_definition(condition),
        "ranked_cfas": [
            {
                "rank": index + 1,
                "cfa_pattern": str(row.get("cfa_pattern", "")),
                "condition_score": float(row.get("condition_score", 0.0)),
                "snr_map_mean": float(row.get("metrics", {}).get("snr_map_mean", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                "visibility_confidence": float(row.get("metrics", {}).get("visibility_confidence", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                "saturation_mean": float(row.get("metrics", {}).get("saturation_mean", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                "edge_confidence_mean": float(row.get("metrics", {}).get("edge_confidence_mean", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
                "clear_or_ir_mean": float(row.get("metrics", {}).get("clear_or_ir_mean", 0.0)) if isinstance(row.get("metrics"), Mapping) else 0.0,
            }
            for index, row in enumerate(ordered)
        ],
    }


def _support_summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for case in cases:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), Mapping) else {}
        warnings = {str(value) for value in case.get("warnings", ())}
        rows.append(
            {
                "case": str(case.get("id", "")),
                "finite_outputs": bool(metrics.get("finite_outputs")),
                "warnings": sorted(warnings),
                "supported": bool(metrics.get("finite_outputs")) and "cfa_warning" not in warnings,
            }
        )
    return {
        "case_count": len(rows),
        "all_finite": bool(rows) and all(bool(row["finite_outputs"]) for row in rows),
        "all_supported": bool(rows) and all(bool(row["supported"]) for row in rows),
        "failed_cases": [row["case"] for row in rows if not bool(row["supported"])],
    }


def _score_definition(condition: str) -> str:
    if condition == "low_light":
        return "mean(SNR, visibility, edge confidence, demosaic confidence, inverse under-exposure)"
    if condition == "glare":
        return "mean(clipping distance, visibility, edge confidence, demosaic confidence, inverse saturation, inverse over-exposure)"
    if condition == "low_mtf":
        return "mean(edge confidence, demosaic confidence, focus confidence)"
    return "mean(SNR, visibility, edge confidence, demosaic confidence, color confidence)"


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    ranking_rows = "".join(_ranking_table(row) for row in summary.get("condition_rankings", ()))
    case_rows = "".join(_case_row(row) for row in summary.get("cases", ()))
    support = summary.get("support", {}) if isinstance(summary.get("support"), Mapping) else {}
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP CFA Stress Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP CFA Stress Sweep</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(str(summary.get('status', '')))}\">{html_lib.escape(str(summary.get('status', '')))}</code>.
  Cases: {int(support.get('case_count', 0))}. Size: {int(summary.get('width', 0))} x {int(summary.get('height', 0))}.</p>
  <h2>Condition Rankings</h2>
  {ranking_rows}
  <h2>Case Metrics</h2>
  <table>
    <thead><tr><th>Condition</th><th>CFA</th><th>Score</th><th>SNR</th><th>Visibility</th><th>Under Exp</th><th>Saturation</th><th>Clip Dist</th><th>Edge Conf</th><th>Demosaic Conf</th><th>Focus</th><th>Color Conf</th><th>Clear/IR</th><th>Warnings</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{CFA_STRESS_SUMMARY}</code></p>
</body>
</html>
"""


def _ranking_table(row: Mapping[str, Any]) -> str:
    body = "".join(_ranking_row(item) for item in row.get("ranked_cfas", ()))
    return (
        f"<h3>{html_lib.escape(str(row.get('condition', '')))}</h3>"
        f"<p><code>{html_lib.escape(str(row.get('score_definition', '')))}</code></p>"
        "<table>"
        "<thead><tr><th>Rank</th><th>CFA</th><th>Score</th><th>SNR</th><th>Visibility</th><th>Saturation</th><th>Edge Conf</th><th>Clear/IR</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _ranking_row(item: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{int(item.get('rank', 0))}</td>"
        f"<td><code>{html_lib.escape(str(item.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(item.get('condition_score'))}</td>"
        f"<td>{_fmt(item.get('snr_map_mean'))}</td>"
        f"<td>{_fmt(item.get('visibility_confidence'))}</td>"
        f"<td>{_fmt(item.get('saturation_mean'))}</td>"
        f"<td>{_fmt(item.get('edge_confidence_mean'))}</td>"
        f"<td>{_fmt(item.get('clear_or_ir_mean'))}</td>"
        "</tr>"
    )


def _case_row(row: Mapping[str, Any]) -> str:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    warnings = ", ".join(str(value) for value in row.get("warnings", ())) or "none"
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('condition', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code></td>"
        f"<td>{_fmt(row.get('condition_score'))}</td>"
        f"<td>{_fmt(metrics.get('snr_map_mean'))}</td>"
        f"<td>{_fmt(metrics.get('visibility_confidence'))}</td>"
        f"<td>{_fmt(metrics.get('under_exposure_fraction'))}</td>"
        f"<td>{_fmt(metrics.get('saturation_mean'))}</td>"
        f"<td>{_fmt(metrics.get('clipping_distance_mean'))}</td>"
        f"<td>{_fmt(metrics.get('edge_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('demosaic_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('focus_confidence'))}</td>"
        f"<td>{_fmt(metrics.get('color_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('clear_or_ir_mean'))}</td>"
        f"<td>{html_lib.escape(warnings)}</td>"
        "</tr>"
    )


def _mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: Any, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _average(*values: Any) -> float:
    numbers = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not numbers:
        return 0.0
    return float(np.mean(np.clip(numbers, 0.0, 1.0)))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
