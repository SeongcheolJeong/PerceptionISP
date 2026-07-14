"""Synthetic edge-confidence validation for PerceptionISP maps."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from perception_isp.core.pipeline import PerceptionISPPipeline
from perception_isp.core.synthetic import make_synthetic_raw
from perception_isp.core.types import PerceptionISPConfig, RawFrame, json_ready


EDGE_CONFIDENCE_SUMMARY = "edge_confidence_suite_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic PerceptionISP edge-confidence validation report.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cfa", default="RGGB")
    parser.add_argument("--output-dir", default="reports/perception_edge_confidence_suite_synthetic")
    args = parser.parse_args(argv)

    summary = build_edge_confidence_suite(width=int(args.width), height=int(args.height), seed=int(args.seed), cfa_pattern=str(args.cfa))
    html_path = write_edge_confidence_suite(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / EDGE_CONFIDENCE_SUMMARY),
                    "case_count": len(summary["cases"]),
                    "check_count": len(summary["checks"]),
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_edge_confidence_suite(
    *,
    width: int = 128,
    height: int = 72,
    seed: int = 23,
    cfa_pattern: str = "RGGB",
    config: PerceptionISPConfig | None = None,
) -> Dict[str, Any]:
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    cases = [
        _run_case("nominal_sharp", "Nominal HDR scene with normal MTF.", _raw_case("nominal_sharp", width, height, seed, cfa_pattern), pipeline=pipeline),
        _run_case("low_light", "Single low-exposure scene with lower SNR.", _raw_case("low_light", width, height, seed, cfa_pattern), pipeline=pipeline),
        _run_case("glare_saturated", "Over-exposed HDR stack with saturated edge regions.", _raw_case("glare_saturated", width, height, seed, cfa_pattern), pipeline=pipeline),
        _run_case("low_mtf", "Nominal scene with degraded MTF confidence calibration.", _raw_case("low_mtf", width, height, seed, cfa_pattern), pipeline=pipeline),
    ]
    by_id = {case["id"]: case for case in cases}
    checks = [
        _delta_check(
            "low_light_edge_confidence_drop",
            "Low light should lower edge confidence and visibility relative to nominal.",
            by_id["nominal_sharp"],
            by_id["low_light"],
            (
                ("edge_confidence_mean", "maximum_delta", -0.10),
                ("strong_edge_confidence_mean", "maximum_delta", -0.08),
                ("visibility_confidence", "maximum_delta", -0.15),
            ),
        ),
        _delta_check(
            "glare_edge_confidence_drop",
            "Saturation/glare should lower edge/demosaic confidence and raise over-exposure.",
            by_id["nominal_sharp"],
            by_id["glare_saturated"],
            (
                ("edge_confidence_mean", "maximum_delta", -0.10),
                ("demosaic_confidence_mean", "maximum_delta", -0.08),
                ("over_exposure_fraction", "minimum_delta", 0.10),
            ),
        ),
        _delta_check(
            "low_mtf_strong_edge_confidence_drop",
            "Low MTF should keep edge candidates visible but lower confidence and focus.",
            by_id["nominal_sharp"],
            by_id["low_mtf"],
            (
                ("edge_confidence_mean", "maximum_delta", -0.20),
                ("strong_edge_confidence_mean", "maximum_delta", -0.25),
                ("focus_confidence", "maximum_delta", -0.10),
            ),
        ),
    ]
    return {
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "cfa_pattern": str(cfa_pattern).upper().replace("-", ""),
        "cases": cases,
        "checks": checks,
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "interpretation": (
            "This synthetic suite checks whether PerceptionISP edge/confidence maps separate reliable edges from low-light, glare, and low-MTF edge stressors. "
            "It is front-end confidence evidence, not detector performance evidence."
        ),
    }


def write_edge_confidence_suite(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _materialize_assets(summary, destination)
    (destination / EDGE_CONFIDENCE_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _raw_case(case_id: str, width: int, height: int, seed: int, cfa_pattern: str) -> RawFrame:
    pattern = str(cfa_pattern).upper().replace("-", "")
    if case_id == "nominal_sharp":
        return make_synthetic_raw(width=width, height=height, exposures=(1.0, 0.25, 0.0625), cfa_pattern=pattern, seed=seed)
    if case_id == "low_light":
        return make_synthetic_raw(width=width, height=height, exposures=(0.08,), cfa_pattern=pattern, seed=seed)
    if case_id == "glare_saturated":
        return make_synthetic_raw(width=width, height=height, exposures=(2.5, 0.5, 0.125), cfa_pattern=pattern, seed=seed)
    if case_id == "low_mtf":
        raw = make_synthetic_raw(width=width, height=height, exposures=(1.0, 0.25, 0.0625), cfa_pattern=pattern, seed=seed)
        exposures = np.asarray(raw.data)
        shape = exposures.shape[-2:] if exposures.ndim == 3 else exposures.shape[:2]
        calibration = replace(raw.calibration, mtf_confidence_map=np.full(shape, 0.25, dtype=np.float64))
        return replace(raw, calibration=calibration, provenance={**dict(raw.provenance), "edge_confidence_low_mtf": True})
    raise ValueError(f"unsupported edge confidence case: {case_id}")


def _run_case(case_id: str, description: str, raw: RawFrame, *, pipeline: PerceptionISPPipeline) -> Dict[str, Any]:
    result = pipeline.run(raw)
    metrics = _metrics(result)
    metrics["finite_outputs"] = bool(np.isfinite(result.vision_rgb).all() and all(np.isfinite(value).all() for value in result.maps.values()))
    return {
        "id": str(case_id),
        "description": str(description),
        "cfa_pattern": str(result.metadata.get("frame", {}).get("cfa_pattern", raw.metadata.cfa_pattern)),
        "warnings": list(result.health.get("warnings", ())),
        "metrics": metrics,
        "_assets_source": {
            "vision_rgb": np.asarray(result.vision_rgb, dtype=np.float64),
            "edge_strength": np.asarray(result.maps["edge_strength"], dtype=np.float64),
            "edge_confidence": np.asarray(result.maps["edge_confidence"], dtype=np.float64),
            "demosaic_confidence": np.asarray(result.maps["demosaic_confidence"], dtype=np.float64),
        },
    }


def _metrics(result: Any) -> Dict[str, Any]:
    edge_strength = np.asarray(result.maps["edge_strength"], dtype=np.float64)
    edge_confidence = np.asarray(result.maps["edge_confidence"], dtype=np.float64)
    demosaic_confidence = np.asarray(result.maps["demosaic_confidence"], dtype=np.float64)
    threshold = float(np.percentile(edge_strength, 90.0))
    strong_mask = edge_strength >= threshold if threshold > 0.0 else np.zeros(edge_strength.shape, dtype=bool)
    if bool(np.any(strong_mask)):
        strong_confidence = float(np.mean(edge_confidence[strong_mask]))
        strong_demosaic = float(np.mean(demosaic_confidence[strong_mask]))
        strong_fraction = float(np.mean(strong_mask))
    else:
        strong_confidence = 0.0
        strong_demosaic = 0.0
        strong_fraction = 0.0
    unreliable_strong = strong_mask & (edge_confidence < 0.30)
    return {
        "edge_strength_mean": _mean(edge_strength),
        "edge_strength_p90": threshold,
        "edge_confidence_mean": _mean(edge_confidence),
        "strong_edge_fraction": strong_fraction,
        "strong_edge_confidence_mean": strong_confidence,
        "strong_edge_demosaic_confidence_mean": strong_demosaic,
        "unreliable_strong_edge_fraction": float(np.mean(unreliable_strong)),
        "demosaic_confidence_mean": _mean(demosaic_confidence),
        "focus_confidence": float(result.health.get("focus_confidence", 0.0)),
        "visibility_confidence": float(result.health.get("visibility_confidence", 0.0)),
        "over_exposure_fraction": float(result.health.get("over_exposure_fraction", 0.0)),
        "under_exposure_fraction": float(result.health.get("under_exposure_fraction", 0.0)),
        "confidence_weighted_edge_score": _mean(edge_strength * edge_confidence),
    }


def _delta_check(
    check_id: str,
    description: str,
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    criteria: Sequence[Tuple[str, str, float]],
) -> Dict[str, Any]:
    rows = []
    base_metrics = baseline.get("metrics", {}) if isinstance(baseline.get("metrics"), Mapping) else {}
    target_metrics = target.get("metrics", {}) if isinstance(target.get("metrics"), Mapping) else {}
    for metric, direction, threshold in criteria:
        base = float(base_metrics.get(metric, 0.0))
        value = float(target_metrics.get(metric, 0.0))
        delta = value - base
        passed = delta >= float(threshold) if direction == "minimum_delta" else delta <= float(threshold)
        rows.append(
            {
                "metric": str(metric),
                "direction": str(direction),
                "baseline": base,
                "target": value,
                "delta": delta,
                "threshold": float(threshold),
                "pass": bool(passed),
            }
        )
    return {
        "id": str(check_id),
        "description": str(description),
        "baseline_case": str(baseline.get("id", "")),
        "target_case": str(target.get("id", "")),
        "status": "pass" if all(bool(row["pass"]) for row in rows) else "fail",
        "criteria": rows,
    }


def _materialize_assets(summary: Mapping[str, Any], destination: Path) -> None:
    assets_dir = destination / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for case in summary.get("cases", ()):
        if not isinstance(case, dict):
            continue
        source = case.pop("_assets_source", None)
        if not isinstance(source, Mapping):
            continue
        assets: Dict[str, str] = {}
        for name, image in source.items():
            filename = f"assets/{_safe_name(case.get('id', 'case'))}_{_safe_name(name)}.png"
            _save_png(np.asarray(image, dtype=np.float64), destination / filename)
            assets[str(name)] = filename
        case["assets"] = assets


def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        out = _to_uint8_gray(image)
    elif image.ndim == 3:
        out = _to_uint8_rgb(image)
    else:
        raise ValueError("asset image must be 2-D or HxWxC")
    Image.fromarray(out).save(path)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.shape[-1] == 1:
        return np.repeat(_to_uint8_gray(rgb[:, :, 0])[:, :, None], 3, axis=2)
    rgb = rgb[:, :, :3]
    if float(np.nanmax(rgb)) <= 1.5:
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.clip(rgb / max(float(np.nanpercentile(rgb, 99.0)), 1.0e-12), 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(gray)
    if not bool(np.any(finite)):
        return np.zeros(gray.shape, dtype=np.uint8)
    low = float(np.nanmin(gray[finite]))
    high = float(np.nanmax(gray[finite]))
    if high <= low:
        out = np.zeros_like(gray, dtype=np.float64)
    elif low >= 0.0 and high <= 1.0:
        out = np.clip(gray, 0.0, 1.0)
    else:
        out = np.clip((gray - low) / (high - low), 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()))
    case_rows = "".join(_case_row(row, destination) for row in summary.get("cases", ()))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Edge Confidence Suite</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    img {{ width: 96px; height: auto; margin-right: 6px; border: 1px solid #d8ded7; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Edge Confidence Suite</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(str(summary.get('status', '')))}\">{html_lib.escape(str(summary.get('status', '')))}</code>.
  CFA: <code>{html_lib.escape(str(summary.get('cfa_pattern', '')))}</code>. Size: {int(summary.get('width', 0))} x {int(summary.get('height', 0))}.</p>
  <h2>Confidence Checks</h2>
  <table>
    <thead><tr><th>Status</th><th>Check</th><th>Description</th><th>Cases</th><th>Criteria</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>
  <h2>Case Metrics</h2>
  <table>
    <thead><tr><th>Case</th><th>Visuals</th><th>Edge Mean</th><th>Edge Conf</th><th>Strong Edge Conf</th><th>Unreliable Strong</th><th>Demosaic Conf</th><th>Focus</th><th>Visibility</th><th>Over Exp</th><th>Warnings</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{EDGE_CONFIDENCE_SUMMARY}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    criteria = "; ".join(
        f"{item.get('metric')} d={_fmt(item.get('delta'), signed=True)} threshold={_fmt(item.get('threshold'), signed=True)}"
        for item in row.get("criteria", ())
        if isinstance(item, Mapping)
    )
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('baseline_case', '')))} -> {html_lib.escape(str(row.get('target_case', '')))}</td>"
        f"<td>{html_lib.escape(criteria)}</td>"
        "</tr>"
    )


def _case_row(row: Mapping[str, Any], destination: Path) -> str:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
    assets = row.get("assets", {}) if isinstance(row.get("assets"), Mapping) else {}
    thumbs = " ".join(
        _asset_img(path, destination)
        for path in (assets.get("vision_rgb"), assets.get("edge_strength"), assets.get("edge_confidence"), assets.get("demosaic_confidence"))
        if path
    )
    warnings = ", ".join(str(value) for value in row.get("warnings", ())) or "none"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code><br>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{thumbs}</td>"
        f"<td>{_fmt(metrics.get('edge_strength_mean'))}</td>"
        f"<td>{_fmt(metrics.get('edge_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('strong_edge_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('unreliable_strong_edge_fraction'))}</td>"
        f"<td>{_fmt(metrics.get('demosaic_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('focus_confidence'))}</td>"
        f"<td>{_fmt(metrics.get('visibility_confidence'))}</td>"
        f"<td>{_fmt(metrics.get('over_exposure_fraction'))}</td>"
        f"<td>{html_lib.escape(warnings)}</td>"
        "</tr>"
    )


def _asset_img(path: str, destination: Path) -> str:
    relative = os.path.relpath(str(destination / path), start=str(destination))
    return f"<img src=\"{html_lib.escape(relative)}\" alt=\"{html_lib.escape(os.path.basename(path))}\">"


def _mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
