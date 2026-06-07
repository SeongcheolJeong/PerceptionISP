"""Sweep derived aux edge-evidence maps against scene-edge proxies."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .eval_types import EvaluationSample
from .pipeline import PerceptionISPPipeline
from .scene_edge_confidence_suite import (
    _case_metrics,
    _edge_strength,
    _luma,
    _source_edge_proxy,
    load_scene_edge_sample_grid,
)
from .types import PerceptionISPConfig, json_ready


SUMMARY_FILENAME = "scene_edge_aux_sweep_summary.json"

DEFAULT_CANDIDATES = (
    "edge_confidence",
    "edge_strength",
    "mean_norm_conf_strength",
    "sqrt_norm_conf_strength",
    "confidence_gated_strength",
    "strength_gated_confidence",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep derived aux evidence maps for scene-edge confidence diagnostics.")
    parser.add_argument("--source", choices=("sample-image", "camerae2e-synthetic", "yolo-dataset", "kitti-dataset", "sid-sony-raw"), default="sample-image")
    parser.add_argument("--image-path", default="data/sample_images/bus.jpg")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--sid-zip", default="data/raw_datasets/sid/downloads/Sony2025.zip")
    parser.add_argument("--sid-cache-dir", default="data/raw_datasets/sid/extracted_samples")
    parser.add_argument("--sid-exposure", type=float, default=0.1)
    parser.add_argument("--cfa", action="append", default=None)
    parser.add_argument("--psf-sigma", action="append", type=float, default=None)
    parser.add_argument("--no-camerae2e", action="store_true")
    parser.add_argument("--tone-mapping", default="detector_log")
    parser.add_argument("--denoise-strength", type=float, default=0.30)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=("edge_aware", "bilinear"))
    parser.add_argument("--demosaic-artifact-suppression", type=float, default=0.20)
    parser.add_argument("--output-dir", default="reports/perception_scene_edge_aux_sweep")
    args = parser.parse_args(argv)

    samples = load_scene_edge_sample_grid(
        source=str(args.source),
        image_path=args.image_path,
        dataset=args.dataset,
        split=str(args.split),
        count=int(args.count),
        offset=int(args.offset),
        width=int(args.width),
        height=int(args.height),
        scene_scale=float(args.scene_scale),
        cfa_patterns=tuple(str(value) for value in (args.cfa or ("auto",))),
        psf_sigmas=tuple(float(value) for value in (args.psf_sigma or (0.0,))),
        use_camerae2e=not bool(args.no_camerae2e),
        sid_zip=str(args.sid_zip),
        sid_cache_dir=str(args.sid_cache_dir),
        sid_exposure=float(args.sid_exposure),
    )
    config = PerceptionISPConfig(
        tone_mapping=str(args.tone_mapping),
        denoise_strength=float(args.denoise_strength),
        demosaic_method=str(args.demosaic_method),
        demosaic_artifact_suppression=float(args.demosaic_artifact_suppression),
    )
    summary = build_scene_edge_aux_sweep(samples, config=config)
    html_path = write_scene_edge_aux_sweep(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "case_count": summary["case_count"],
                    "best_candidate": summary["best_candidate"],
                    "failed_checks": [row["id"] for row in summary["checks"] if row["status"] != "pass"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_scene_edge_aux_sweep(
    samples: Sequence[EvaluationSample],
    *,
    config: PerceptionISPConfig | None = None,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("scene-edge aux sweep needs at least one sample")
    pipeline = PerceptionISPPipeline(config=config or PerceptionISPConfig())
    cases = [_run_case(sample, pipeline=pipeline, candidate_names=candidates) for sample in samples]
    aggregate = _aggregate(cases, candidates)
    best = _best_candidate(aggregate)
    checks = _checks(cases, aggregate, best)
    return {
        "name": "Scene-edge aux evidence sweep",
        "case_count": len(cases),
        "candidate_names": list(candidates),
        "best_candidate": best,
        "cases": cases,
        "aggregate": aggregate,
        "checks": checks,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This sweep tests fixed, training-free combinations of PerceptionISP aux maps against the same high-information scene-edge proxy. "
            "It is meant to identify whether the current aux-confidence failures are due to missing edge evidence or to the way aux maps are combined."
        ),
        "claim_boundary": (
            "This is same-sample diagnostic tuning evidence. It is not a held-out claim and must be rerun on AODRaw or another object-detection RAW benchmark before performance claims."
        ),
    }


def write_scene_edge_aux_sweep(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _run_case(
    sample: EvaluationSample,
    *,
    pipeline: PerceptionISPPipeline,
    candidate_names: Sequence[str],
) -> Dict[str, Any]:
    result = pipeline.run(sample.raw)
    human_rgb = np.asarray(result.human_rgb if result.human_rgb is not None else result.vision_rgb, dtype=np.float64)
    reference_rgb, source_strength, source_edge = _source_edge_proxy(sample, human_rgb.shape[:2])
    human_strength = _edge_strength(_luma(human_rgb))
    confidence = np.asarray(result.maps["edge_confidence"], dtype=np.float64)
    strength = np.asarray(result.maps["edge_strength"], dtype=np.float64)
    signals = _candidate_signals(confidence=confidence, strength=strength)
    candidates = []
    for name in candidate_names:
        signal = np.asarray(signals[name], dtype=np.float64)
        metrics = _case_metrics(
            source_strength=source_strength,
            source_edge=source_edge,
            human_strength=human_strength,
            perception_strength=human_strength,
            aux_confidence=signal,
            aux_strength=signal,
        )
        candidates.append(
            {
                "name": str(name),
                "source_edge_f1": float(metrics["perception_aux_confidence_source_edge_f1"]),
                "minus_human_source_edge_f1": float(metrics["perception_aux_confidence_minus_human_source_edge_f1"]),
                "scene_edge_separation": float(metrics["perception_aux_confidence_scene_edge_separation"]),
                "source_edge_correlation": float(metrics["perception_aux_confidence_source_edge_correlation"]),
                "signal_mean": float(metrics["perception_aux_confidence_mean"]),
            }
        )
    return {
        "id": str(sample.sample_id),
        "source": str(sample.source),
        "cfa_pattern": str(result.metadata.get("frame", {}).get("cfa_pattern", sample.raw.metadata.cfa_pattern)),
        "psf_sigma": _psf_sigma(sample),
        "source_edge_fraction": float(np.mean(source_edge)),
        "human_rgb_proxy_source_edge_f1": _candidate_metric(candidates, "edge_confidence", "source_edge_f1")
        - _candidate_metric(candidates, "edge_confidence", "minus_human_source_edge_f1"),
        "candidates": candidates,
        "finite_outputs": bool(
            np.isfinite(reference_rgb).all()
            and np.isfinite(human_rgb).all()
            and np.isfinite(confidence).all()
            and np.isfinite(strength).all()
            and all(np.isfinite(signals[name]).all() for name in candidate_names)
        ),
    }


def _candidate_signals(*, confidence: np.ndarray, strength: np.ndarray) -> Dict[str, np.ndarray]:
    c = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    s = np.clip(np.asarray(strength, dtype=np.float64), 0.0, 1.0)
    cn = _robust_norm(c)
    sn = _robust_norm(s)
    return {
        "edge_confidence": c,
        "edge_strength": s,
        "mean_norm_conf_strength": 0.5 * cn + 0.5 * sn,
        "sqrt_norm_conf_strength": np.sqrt(np.clip(cn * sn, 0.0, 1.0)),
        "confidence_gated_strength": sn * (0.25 + 0.75 * cn),
        "strength_gated_confidence": cn * (0.25 + 0.75 * sn),
    }


def _robust_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.clip(arr, 0.0, 1.0)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _aggregate(cases: Sequence[Mapping[str, Any]], candidate_names: Sequence[str]) -> Dict[str, Any]:
    rows: Dict[str, list[Mapping[str, Any]]] = {str(name): [] for name in candidate_names}
    for case in cases:
        for row in case.get("candidates", ()):
            if isinstance(row, Mapping):
                rows.setdefault(str(row.get("name", "")), []).append(row)
    aggregate: Dict[str, Any] = {}
    for name in candidate_names:
        items = rows.get(str(name), [])
        deltas = [float(row.get("minus_human_source_edge_f1", 0.0)) for row in items]
        separations = [float(row.get("scene_edge_separation", 0.0)) for row in items]
        f1s = [float(row.get("source_edge_f1", 0.0)) for row in items]
        aggregate[str(name)] = {
            "case_count": len(items),
            "source_edge_f1_mean": _mean(f1s),
            "minus_human_source_edge_f1_mean": _mean(deltas),
            "source_edge_f1_win_rate": _win_rate(deltas),
            "scene_edge_separation_mean": _mean(separations),
            "min_scene_edge_separation": min(separations or [0.0]),
            "negative_scene_edge_separation_count": sum(1 for value in separations if value <= 0.0),
            "score": float(_mean(deltas) + 0.25 * _mean(separations) - 0.02 * sum(1 for value in separations if value <= 0.0)),
        }
    return aggregate


def _best_candidate(aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for name, payload in aggregate.items():
        if isinstance(payload, Mapping):
            rows.append({"name": str(name), **dict(payload)})
    rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return rows[0] if rows else {}


def _checks(cases: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], best: Mapping[str, Any]) -> list[Dict[str, Any]]:
    finite = all(bool(row.get("finite_outputs")) for row in cases)
    best_delta = float(best.get("minus_human_source_edge_f1_mean", 0.0)) if best else 0.0
    best_neg_sep = int(best.get("negative_scene_edge_separation_count", 0)) if best else 0
    return [
        {
            "id": "aux_sweep_outputs_finite",
            "status": "pass" if finite else "fail",
            "evidence": f"finite={finite}",
        },
        {
            "id": "aux_sweep_best_candidate_positive_delta",
            "status": "pass" if best_delta > 0.0 else "fail",
            "evidence": f"best={best.get('name', '')} dF1={best_delta:.4f}",
        },
        {
            "id": "aux_sweep_best_candidate_tracks_source_edges",
            "status": "pass" if best_neg_sep == 0 else "fail",
            "evidence": f"best={best.get('name', '')} negative_separation_count={best_neg_sep}",
        },
        {
            "id": "aux_sweep_includes_current_aux_confidence_baseline",
            "status": "pass" if "edge_confidence" in aggregate else "fail",
            "evidence": "edge_confidence baseline present",
        },
    ]


def _render_html(summary: Mapping[str, Any]) -> str:
    aggregate_rows = "".join(_aggregate_row(name, payload) for name, payload in (summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}).items())
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    case_rows = "".join(_case_row(row, summary.get("best_candidate", {}).get("name", "")) for row in summary.get("cases", ()) if isinstance(row, Mapping))
    status = str(summary.get("status", ""))
    best = summary.get("best_candidate", {}) if isinstance(summary.get("best_candidate"), Mapping) else {}
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Scene-Edge Aux Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail, .warning {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Scene-Edge Aux Sweep</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</code>. Cases: {int(summary.get('case_count', 0))}. Best candidate: <code>{html_lib.escape(str(best.get('name', '')))}</code>, dF1={_fmt(best.get('minus_human_source_edge_f1_mean'), signed=True)}, win={_fmt(best.get('source_edge_f1_win_rate'))}, negative separation={int(best.get('negative_scene_edge_separation_count', 0))}.</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Status</th><th>Check</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Candidates</h2>
  <table><thead><tr><th>Candidate</th><th>F1 Mean</th><th>dF1 Mean</th><th>Win Rate</th><th>Separation Mean</th><th>Min Separation</th><th>Negative Separation</th><th>Score</th></tr></thead><tbody>{aggregate_rows}</tbody></table>
  <h2>Cases</h2>
  <table><thead><tr><th>Case</th><th>CFA/PSF</th><th>Source Edge</th><th>Human F1</th><th>Best F1</th><th>Best dF1</th><th>Best Separation</th></tr></thead><tbody>{case_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _aggregate_row(name: str, payload: Any) -> str:
    row = payload if isinstance(payload, Mapping) else {}
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{_fmt(row.get('source_edge_f1_mean'))}</td>"
        f"<td>{_fmt(row.get('minus_human_source_edge_f1_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('source_edge_f1_win_rate'))}</td>"
        f"<td>{_fmt(row.get('scene_edge_separation_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('min_scene_edge_separation'), signed=True)}</td>"
        f"<td>{int(row.get('negative_scene_edge_separation_count', 0))}</td>"
        f"<td>{_fmt(row.get('score'), signed=True)}</td>"
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


def _case_row(row: Mapping[str, Any], best_name: Any) -> str:
    best = _candidate(row, str(best_name))
    human = float(row.get("human_rgb_proxy_source_edge_f1", 0.0))
    best_f1 = float(best.get("source_edge_f1", 0.0)) if best else 0.0
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code><br>{html_lib.escape(str(row.get('source', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('cfa_pattern', '')))}</code> / {_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{_fmt(row.get('source_edge_fraction'))}</td>"
        f"<td>{_fmt(human)}</td>"
        f"<td>{_fmt(best_f1)}</td>"
        f"<td>{_fmt(best_f1 - human, signed=True)}</td>"
        f"<td>{_fmt(best.get('scene_edge_separation') if best else None, signed=True)}</td>"
        "</tr>"
    )


def _candidate(row: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for item in row.get("candidates", ()):
        if isinstance(item, Mapping) and str(item.get("name", "")) == name:
            return item
    return None


def _candidate_metric(candidates: Sequence[Mapping[str, Any]], name: str, key: str) -> float:
    for item in candidates:
        if str(item.get("name", "")) == name:
            return float(item.get(key, 0.0))
    return 0.0


def _psf_sigma(sample: EvaluationSample) -> float:
    sigma_map = sample.raw.calibration.psf_sigma_map
    if sigma_map is None:
        return 0.0
    return float(np.mean(np.asarray(sigma_map, dtype=np.float64)))


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def _win_rate(deltas: Sequence[float]) -> float:
    arr = np.asarray(deltas, dtype=np.float64)
    return float(np.mean(arr > 0.0)) if arr.size else 0.0


def _fmt(value: Any, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
