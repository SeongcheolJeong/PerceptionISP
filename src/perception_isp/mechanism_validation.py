"""Synthetic mechanism validation for PerceptionISP auxiliary maps."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .pipeline import PerceptionISPPipeline
from .synthetic import make_synthetic_raw
from .types import PerceptionISPConfig, RawFrame, json_ready


MECHANISM_SUMMARY = "mechanism_validation_summary.json"
DEFAULT_CFAS = ("RGGB", "GRBG", "RCCB", "RGBIR")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic PerceptionISP mechanism-validation report.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cfa", action="append", default=[], help="CFA pattern for the CFA-support sweep. Repeats are allowed.")
    parser.add_argument("--output-dir", default="reports/perception_mechanism_validation_synthetic")
    args = parser.parse_args(argv)

    summary = build_mechanism_validation(
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
        cfa_patterns=tuple(args.cfa) or DEFAULT_CFAS,
    )
    html_path = write_mechanism_validation(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / MECHANISM_SUMMARY),
                    "case_count": len(summary["cases"]),
                    "mechanism_count": len(summary["mechanisms"]),
                    "failed_mechanisms": [row["id"] for row in summary["mechanisms"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_mechanism_validation(
    *,
    width: int = 128,
    height: int = 72,
    seed: int = 7,
    cfa_patterns: Sequence[str] = DEFAULT_CFAS,
    config: PerceptionISPConfig | None = None,
) -> Dict[str, Any]:
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    case_specs = _case_specs(width=int(width), height=int(height), seed=int(seed), cfa_patterns=tuple(cfa_patterns))
    cases = [_run_case(case_id, raw, pipeline=pipeline, group=group, description=description) for case_id, group, description, raw in case_specs]
    by_id = {case["id"]: case for case in cases}
    mechanisms = [
        _mechanism_delta(
            "low_light_noise_response",
            "Low light reduces SNR/visibility and raises under-exposure risk.",
            by_id["nominal_rggb"],
            by_id["low_light_rggb"],
            (
                ("snr_map_mean", "maximum_delta", -0.15),
                ("visibility_confidence", "maximum_delta", -0.10),
                ("under_exposure_fraction", "minimum_delta", 0.05),
            ),
        ),
        _mechanism_delta(
            "glare_saturation_response",
            "Glare raises saturation/over-exposure and lowers clipping distance.",
            by_id["nominal_rggb"],
            by_id["glare_rggb"],
            (
                ("saturation_mean", "minimum_delta", 0.10),
                ("over_exposure_fraction", "minimum_delta", 0.10),
                ("clipping_distance_mean", "maximum_delta", -0.20),
            ),
        ),
        _mechanism_delta(
            "low_mtf_edge_confidence_response",
            "Low optical MTF lowers edge, demosaic, and focus confidence.",
            by_id["nominal_rggb"],
            by_id["low_mtf_rggb"],
            (
                ("edge_confidence_mean", "maximum_delta", -0.15),
                ("demosaic_confidence_mean", "maximum_delta", -0.10),
                ("focus_confidence", "maximum_delta", -0.10),
            ),
        ),
        _cfa_support_mechanism([case for case in cases if case["group"] == "cfa_support"], minimum_count=min(len(tuple(cfa_patterns)), 4)),
    ]
    return {
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "cfa_patterns": [str(value).upper().replace("-", "") for value in cfa_patterns],
        "cases": cases,
        "mechanisms": mechanisms,
        "status": "pass" if all(row["status"] == "pass" for row in mechanisms) else "fail",
        "interpretation": (
            "Synthetic mechanism validation checks whether PerceptionISP auxiliary/confidence maps respond to controlled sensor/ISP stressors. "
            "It is feasibility evidence for the front-end signals, not a detector performance claim."
        ),
    }


def write_mechanism_validation(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _materialize_assets(summary, destination)
    (destination / MECHANISM_SUMMARY).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _case_specs(*, width: int, height: int, seed: int, cfa_patterns: Tuple[str, ...]) -> Tuple[Tuple[str, str, str, RawFrame], ...]:
    nominal = make_synthetic_raw(width=width, height=height, exposures=(1.0, 0.25, 0.0625), cfa_pattern="RGGB", seed=seed)
    low_light = make_synthetic_raw(width=width, height=height, exposures=(0.08,), cfa_pattern="RGGB", seed=seed)
    glare = make_synthetic_raw(width=width, height=height, exposures=(2.5, 0.5, 0.125), cfa_pattern="RGGB", seed=seed)
    low_mtf = _with_low_mtf(nominal, mtf=0.25)
    specs = [
        ("nominal_rggb", "baseline", "Nominal HDR RGGB synthetic scene.", nominal),
        ("low_light_rggb", "mechanism", "Single low-exposure frame to exercise SNR and under-exposure maps.", low_light),
        ("glare_rggb", "mechanism", "Over-exposed HDR stack to exercise saturation and clipping-distance maps.", glare),
        ("low_mtf_rggb", "mechanism", "Nominal RAW with low MTF calibration to exercise edge/focus confidence.", low_mtf),
    ]
    for cfa in cfa_patterns:
        pattern = str(cfa).upper().replace("-", "")
        specs.append(
            (
                f"cfa_{_safe_name(pattern).lower()}",
                "cfa_support",
                f"{pattern} synthetic RAW to verify CFA-specific decode path and map validity.",
                make_synthetic_raw(width=width, height=height, exposures=(1.0, 0.25, 0.0625), cfa_pattern=pattern, seed=seed),
            )
        )
    return tuple(specs)


def _with_low_mtf(raw: RawFrame, *, mtf: float) -> RawFrame:
    exposures = np.asarray(raw.data)
    shape = exposures.shape[-2:] if exposures.ndim == 3 else exposures.shape[:2]
    calibration = replace(raw.calibration, mtf_confidence_map=np.full(shape, float(mtf), dtype=np.float64))
    return replace(raw, calibration=calibration, provenance={**dict(raw.provenance), "mechanism_low_mtf": True})


def _run_case(case_id: str, raw: RawFrame, *, pipeline: PerceptionISPPipeline, group: str, description: str) -> Dict[str, Any]:
    result = pipeline.run(raw)
    metrics = _case_metrics(result)
    metrics["finite_outputs"] = bool(np.isfinite(result.vision_rgb).all() and all(np.isfinite(value).all() for value in result.maps.values()))
    return {
        "id": str(case_id),
        "group": str(group),
        "description": str(description),
        "cfa_pattern": str(result.metadata.get("frame", {}).get("cfa_pattern", raw.metadata.cfa_pattern)),
        "warnings": list(result.health.get("warnings", ())),
        "metrics": metrics,
        "_assets_source": {
            "vision_rgb": np.asarray(result.vision_rgb, dtype=np.float64),
            "snr_map": np.asarray(result.maps["snr_map"], dtype=np.float64),
            "saturation": np.asarray(result.maps["saturation"], dtype=np.float64),
            "edge_confidence": np.asarray(result.maps["edge_confidence"], dtype=np.float64),
            "demosaic_confidence": np.asarray(result.maps["demosaic_confidence"], dtype=np.float64),
        },
    }


def _case_metrics(result: Any) -> Dict[str, Any]:
    maps = result.maps
    health = result.health
    return {
        "snr_map_mean": _mean(maps["snr_map"]),
        "snr_map_p10": _percentile(maps["snr_map"], 10.0),
        "noise_variance_mean": _mean(maps["noise_variance"]),
        "saturation_mean": _mean(maps["saturation"]),
        "clipping_distance_mean": _mean(maps["clipping_distance"]),
        "hdr_confidence_mean": _mean(maps["hdr_confidence"]),
        "edge_confidence_mean": _mean(maps["edge_confidence"]),
        "edge_strength_mean": _mean(maps["edge_strength"]),
        "demosaic_confidence_mean": _mean(maps["demosaic_confidence"]),
        "blur_focus_confidence_mean": _mean(maps["blur_focus_confidence"]),
        "edge_packet_count": int(len(result.fast.edge_packets)),
        "visibility_confidence": float(health.get("visibility_confidence", 0.0)),
        "dnn_input_validity": float(health.get("dnn_input_validity", 0.0)),
        "over_exposure_fraction": float(health.get("over_exposure_fraction", 0.0)),
        "under_exposure_fraction": float(health.get("under_exposure_fraction", 0.0)),
        "focus_confidence": float(health.get("focus_confidence", 0.0)),
        "color_tint_confidence": float(health.get("color_tint_confidence", 0.0)),
    }


def _mechanism_delta(
    mechanism_id: str,
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
    passed_all = all(bool(row["pass"]) for row in rows)
    return {
        "id": str(mechanism_id),
        "description": str(description),
        "baseline_case": str(baseline.get("id", "")),
        "target_case": str(target.get("id", "")),
        "status": "pass" if passed_all else "fail",
        "criteria": rows,
    }


def _cfa_support_mechanism(cases: Sequence[Mapping[str, Any]], *, minimum_count: int) -> Dict[str, Any]:
    rows = []
    for case in cases:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), Mapping) else {}
        warnings = {str(value) for value in case.get("warnings", ())}
        passed = bool(metrics.get("finite_outputs")) and "cfa_warning" not in warnings
        rows.append(
            {
                "case": str(case.get("id", "")),
                "cfa_pattern": str(case.get("cfa_pattern", "")),
                "finite_outputs": bool(metrics.get("finite_outputs")),
                "warnings": sorted(warnings),
                "pass": bool(passed),
            }
        )
    passed_all = len(rows) >= int(minimum_count) and all(bool(row["pass"]) for row in rows)
    return {
        "id": "cfa_variant_support",
        "description": "Configured CFA variants run through the PerceptionISP decode path with finite outputs and valid aux maps.",
        "baseline_case": "",
        "target_case": "",
        "status": "pass" if passed_all else "fail",
        "minimum_count": int(minimum_count),
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
    mechanism_rows = "".join(_mechanism_row(row) for row in summary.get("mechanisms", ()))
    case_rows = "".join(_case_row(row, destination) for row in summary.get("cases", ()))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Mechanism Validation</title>
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
  <h1>PerceptionISP Mechanism Validation</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(str(summary.get('status', '')))}\">{html_lib.escape(str(summary.get('status', '')))}</code>. Size: {int(summary.get('width', 0))} x {int(summary.get('height', 0))}.</p>
  <h2>Mechanism Checks</h2>
  <table>
    <thead><tr><th>Status</th><th>Mechanism</th><th>Description</th><th>Cases</th><th>Criteria</th></tr></thead>
    <tbody>{mechanism_rows}</tbody>
  </table>
  <h2>Case Metrics</h2>
  <table>
    <thead><tr><th>Case</th><th>CFA</th><th>Group</th><th>Visuals</th><th>SNR</th><th>Saturation</th><th>Clip Dist</th><th>Edge Conf</th><th>Demosaic Conf</th><th>Focus</th><th>Visibility</th><th>Warnings</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{MECHANISM_SUMMARY}</code></p>
</body>
</html>
"""


def _mechanism_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    if row.get("id") == "cfa_variant_support":
        criteria = "; ".join(
            f"{item.get('cfa_pattern')} finite={item.get('finite_outputs')} pass={item.get('pass')}"
            for item in row.get("criteria", ())
            if isinstance(item, Mapping)
        )
    else:
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
    thumbs = " ".join(_asset_img(path, destination) for path in (assets.get("vision_rgb"), assets.get("snr_map"), assets.get("saturation"), assets.get("edge_confidence")) if path)
    warnings = ", ".join(str(value) for value in row.get("warnings", ())) or "none"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code><br>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('group', '')))}</td>"
        f"<td>{thumbs}</td>"
        f"<td>{_fmt(metrics.get('snr_map_mean'))}</td>"
        f"<td>{_fmt(metrics.get('saturation_mean'))}</td>"
        f"<td>{_fmt(metrics.get('clipping_distance_mean'))}</td>"
        f"<td>{_fmt(metrics.get('edge_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('demosaic_confidence_mean'))}</td>"
        f"<td>{_fmt(metrics.get('focus_confidence'))}</td>"
        f"<td>{_fmt(metrics.get('visibility_confidence'))}</td>"
        f"<td>{html_lib.escape(warnings)}</td>"
        "</tr>"
    )


def _asset_img(path: str, destination: Path) -> str:
    relative = os.path.relpath(str(destination / path), start=str(destination))
    return f"<img src=\"{html_lib.escape(relative)}\" alt=\"{html_lib.escape(os.path.basename(path))}\">"


def _mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: Any, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
