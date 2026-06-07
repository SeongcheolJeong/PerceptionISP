"""Summarize RGB+Aux versus RGB-only dense detector confidence sweeps."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .rgb_aux_dnn_gate import DENSE_EVAL_SUMMARY, DNN_GATE_PROFILES
from .types import json_ready


SUMMARY_FILENAME = "rgb_aux_dnn_sweep_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize matched RGB+Aux/RGB-only dense detector confidence sweeps.")
    parser.add_argument("--confidence", action="append", type=float, required=True, help="Confidence threshold for a matched pair.")
    parser.add_argument("--rgb-aux", action="append", required=True, help="RGB+Aux dense_eval_summary path/dir. Repeat with --confidence.")
    parser.add_argument("--rgb-only", action="append", required=True, help="RGB-only dense_eval_summary path/dir. Repeat with --confidence.")
    parser.add_argument("--profile", default="claim_quality", choices=sorted(DNN_GATE_PROFILES))
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_dnn_sweep")
    args = parser.parse_args(argv)

    if not (len(args.confidence) == len(args.rgb_aux) == len(args.rgb_only)):
        raise SystemExit("--confidence, --rgb-aux, and --rgb-only must be repeated the same number of times")

    pairs = []
    for confidence, rgb_aux_spec, rgb_only_spec in zip(args.confidence, args.rgb_aux, args.rgb_only):
        rgb_aux_path, rgb_aux = _load_dense_eval(rgb_aux_spec)
        rgb_only_path, rgb_only = _load_dense_eval(rgb_only_spec)
        pairs.append(
            {
                "confidence": float(confidence),
                "rgb_aux": rgb_aux,
                "rgb_only": rgb_only,
                "rgb_aux_source": rgb_aux_path,
                "rgb_only_source": rgb_only_path,
            }
        )
    summary = build_rgb_aux_dnn_sweep(pairs, profile=str(args.profile))
    html_path = write_rgb_aux_dnn_sweep(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "profile": summary["profile"],
                    "pass": summary["pass"],
                    "metric_pass": summary["metric_pass"],
                    "best_metric_confidence": (
                        None
                        if summary.get("best_metric_row") is None
                        else summary["best_metric_row"].get("confidence")
                    ),
                }
            ),
            indent=2,
        )
    )
    return 0


def build_rgb_aux_dnn_sweep(
    pairs: Sequence[Mapping[str, Any]],
    *,
    profile: str = "claim_quality",
) -> Dict[str, Any]:
    thresholds = dict(DNN_GATE_PROFILES.get(str(profile), DNN_GATE_PROFILES["claim_quality"]))
    rows = [
        _sweep_row(pair, thresholds=thresholds)
        for pair in sorted(pairs, key=lambda item: float(item.get("confidence", 0.0)))
    ]
    passed_rows = [row for row in rows if bool(row.get("pass"))]
    metric_rows = [row for row in rows if bool(row.get("metric_pass"))]
    best_metric = _best_by(metric_rows, ("rgb_aux", "recall@0.50_mean"), lower_secondary=("rgb_aux", "fp@0.50_mean"))
    best_recall_positive = _best_by(
        [row for row in rows if _metric(row, "deltas", "fp@0.50_mean") <= 0.0 and _metric(row, "deltas", "recall@0.50_mean") >= 0.0],
        ("rgb_aux", "recall@0.50_mean"),
        lower_secondary=("rgb_aux", "fp@0.50_mean"),
    )
    lowest_fp_positive = _lowest_by(
        [row for row in rows if _metric(row, "deltas", "recall@0.50_mean") > 0.0],
        ("rgb_aux", "fp@0.50_mean"),
    )
    passed = bool(passed_rows)
    metric_pass = bool(metric_rows)
    return {
        "name": "RGB+Aux DNN confidence sweep",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "metric_pass": metric_pass,
        "claim_status": _claim_status(str(profile), passed=passed, metric_pass=metric_pass),
        "profile": str(profile),
        "thresholds": thresholds,
        "row_count": len(rows),
        "rows": rows,
        "best_passing_row": passed_rows[0] if passed_rows else None,
        "best_metric_row": best_metric,
        "best_recall_positive_delta_row": best_recall_positive,
        "lowest_fp_positive_recall_delta_row": lowest_fp_positive,
        "interpretation": _interpretation(str(profile), passed=passed, metric_pass=metric_pass),
        "claim_boundary": (
            "This sweep only changes dense-detector confidence on saved RGB+Aux/RGB-only compact DNN outputs. "
            "It does not retrain the model, increase held-out scale, or prove full detector superiority."
        ),
    }


def write_rgb_aux_dnn_sweep(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _sweep_row(pair: Mapping[str, Any], *, thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    rgb_aux = _run_row(pair.get("rgb_aux", {}), pair.get("rgb_aux_source"))
    rgb_only = _run_row(pair.get("rgb_only", {}), pair.get("rgb_only_source"))
    deltas = {
        "precision@0.50_mean": _delta(rgb_aux, rgb_only, "precision@0.50_mean"),
        "recall@0.50_mean": _delta(rgb_aux, rgb_only, "recall@0.50_mean"),
        "small_recall@0.50_mean": _delta(rgb_aux, rgb_only, "small_recall@0.50_mean"),
        "fp@0.50_mean": _delta(rgb_aux, rgb_only, "fp@0.50_mean"),
    }
    criteria = _criteria(rgb_aux, rgb_only, deltas, thresholds)
    metric_criteria = [row for row in criteria if str(row.get("id", "")) not in {"sample_count", "eval_classes_present"}]
    return {
        "confidence": float(pair.get("confidence", 0.0)),
        "rgb_aux": rgb_aux,
        "rgb_only": rgb_only,
        "deltas": deltas,
        "criteria": criteria,
        "pass": all(bool(row.get("pass")) for row in criteria),
        "metric_pass": all(bool(row.get("pass")) for row in metric_criteria),
        "failed_criteria": [str(row.get("id", "")) for row in criteria if not bool(row.get("pass"))],
        "failed_metric_criteria": [str(row.get("id", "")) for row in metric_criteria if not bool(row.get("pass"))],
    }


def _run_row(summary: Any, source: Any) -> Dict[str, Any]:
    data = summary if isinstance(summary, Mapping) else {}
    aggregate = data.get("aggregate", {}) if isinstance(data.get("aggregate"), Mapping) else {}
    checkpoint = data.get("checkpoint_summary", {}) if isinstance(data.get("checkpoint_summary"), Mapping) else {}
    return {
        "source": "" if source is None else str(source),
        "html_path": _sibling_html(source),
        "sample_count": int(data.get("sample_count", aggregate.get("sample_count", 0))),
        "channel_mode": str(checkpoint.get("channel_mode", "")),
        "precision@0.50_mean": _maybe_float(aggregate.get("precision@0.50_mean")),
        "recall@0.50_mean": _maybe_float(aggregate.get("recall@0.50_mean")),
        "small_recall@0.50_mean": _maybe_float(aggregate.get("small_recall@0.50_mean")),
        "fp@0.50_mean": _maybe_float(aggregate.get("fp@0.50_mean")),
        "det_count_mean": _maybe_float(aggregate.get("det_count_mean")),
        "missing_eval_class_names": _missing_classes(checkpoint),
    }


def _criteria(
    rgb_aux: Mapping[str, Any],
    rgb_only: Mapping[str, Any],
    deltas: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    missing = list(rgb_aux.get("missing_eval_class_names", ())) + list(rgb_only.get("missing_eval_class_names", ()))
    return [
        _minimum_value("sample_count", rgb_aux.get("sample_count"), thresholds.get("min_samples")),
        _minimum_value("absolute_precision", rgb_aux.get("precision@0.50_mean"), thresholds.get("min_precision")),
        _minimum_value("absolute_recall", rgb_aux.get("recall@0.50_mean"), thresholds.get("min_recall")),
        _maximum_value("absolute_fp_per_sample", rgb_aux.get("fp@0.50_mean"), thresholds.get("max_fp_per_sample")),
        _minimum_delta("precision_vs_rgb_only", deltas.get("precision@0.50_mean"), thresholds.get("min_precision_delta")),
        _minimum_delta("recall_vs_rgb_only", deltas.get("recall@0.50_mean"), thresholds.get("min_recall_delta")),
        _minimum_delta("small_recall_vs_rgb_only", deltas.get("small_recall@0.50_mean"), thresholds.get("min_small_recall_delta")),
        _maximum_delta("fp_vs_rgb_only", deltas.get("fp@0.50_mean"), thresholds.get("max_fp_delta")),
        {
            "id": "eval_classes_present",
            "target": len(missing),
            "threshold": 0,
            "delta": None,
            "status": "pass" if not missing else "fail",
            "pass": not missing,
        },
    ]


def _minimum_value(identifier: str, value: Any, threshold: Any) -> Dict[str, Any]:
    available = value is not None and threshold is not None
    passed = bool(available and float(value) >= float(threshold))
    return _criterion(identifier, value, threshold, None if not available else float(value) - float(threshold), passed)


def _maximum_value(identifier: str, value: Any, threshold: Any) -> Dict[str, Any]:
    available = value is not None and threshold is not None
    passed = bool(available and float(value) <= float(threshold))
    return _criterion(identifier, value, threshold, None if not available else float(value) - float(threshold), passed)


def _minimum_delta(identifier: str, delta: Any, threshold: Any) -> Dict[str, Any]:
    available = delta is not None and threshold is not None
    passed = bool(available and float(delta) >= float(threshold))
    return _criterion(identifier, delta, threshold, None if not available else float(delta) - float(threshold), passed)


def _maximum_delta(identifier: str, delta: Any, threshold: Any) -> Dict[str, Any]:
    available = delta is not None and threshold is not None
    passed = bool(available and float(delta) <= float(threshold))
    return _criterion(identifier, delta, threshold, None if not available else float(delta) - float(threshold), passed)


def _criterion(identifier: str, target: Any, threshold: Any, delta: Any, passed: bool) -> Dict[str, Any]:
    return {
        "id": identifier,
        "target": _maybe_float(target),
        "threshold": _maybe_float(threshold),
        "delta": _maybe_float(delta),
        "status": "pass" if passed else "fail",
        "pass": bool(passed),
    }


def _load_dense_eval(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    summary_path = _summary_path(path, DENSE_EVAL_SUMMARY)
    return summary_path, json.loads(summary_path.read_text())


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"dense eval summary not found: {candidate}")
    return candidate


def _sibling_html(source: Any) -> str:
    if source is None:
        return ""
    path = Path(source).expanduser()
    html_path = (path / "index.html") if path.is_dir() else path.with_name("index.html")
    return str(html_path) if html_path.exists() else ""


def _missing_classes(checkpoint: Mapping[str, Any]) -> list[str]:
    value = checkpoint.get("missing_eval_class_names", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _delta(target: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> float | None:
    target_value = target.get(key)
    baseline_value = baseline.get(key)
    if target_value is None or baseline_value is None:
        return None
    return float(target_value) - float(baseline_value)


def _best_by(rows: Sequence[Mapping[str, Any]], key: tuple[str, str], *, lower_secondary: tuple[str, str]) -> Dict[str, Any] | None:
    if not rows:
        return None
    return dict(
        max(
            rows,
            key=lambda row: (
                _metric(row, key[0], key[1]),
                -_metric(row, lower_secondary[0], lower_secondary[1]),
            ),
        )
    )


def _lowest_by(rows: Sequence[Mapping[str, Any]], key: tuple[str, str]) -> Dict[str, Any] | None:
    if not rows:
        return None
    return dict(min(rows, key=lambda row: _metric(row, key[0], key[1])))


def _metric(row: Mapping[str, Any], section: str, key: str) -> float:
    payload = row.get(section, {}) if isinstance(row.get(section), Mapping) else {}
    value = payload.get(key)
    if value is None:
        return float("-inf")
    return float(value)


def _claim_status(profile: str, *, passed: bool, metric_pass: bool) -> str:
    if profile == "diagnostic":
        return "rgb_aux_dnn_sweep_diagnostic_pass" if (passed or metric_pass) else "rgb_aux_dnn_sweep_diagnostic_no_operating_point"
    if passed:
        return "rgb_aux_dnn_sweep_claim_ready"
    if metric_pass:
        return "rgb_aux_dnn_sweep_needs_scale"
    return "rgb_aux_dnn_sweep_no_claim_operating_point"


def _interpretation(profile: str, *, passed: bool, metric_pass: bool) -> str:
    if profile == "diagnostic" and (passed or metric_pass):
        return "At least one confidence threshold passes the diagnostic RGB+Aux DNN sweep gate; this is operating-point evidence, not a claim-ready detector result."
    if passed:
        return "At least one confidence threshold passes the configured RGB+Aux DNN sweep gate."
    if metric_pass:
        return "At least one confidence threshold passes the metric criteria, but not the sample-scale gate."
    return "No confidence threshold passes the configured metric criteria; this is not a claim-ready learned RGB+Aux detector result."


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(summary.get("pass")) else "not_supported"
    rows = "".join(_row_html(row, destination) for row in summary.get("rows", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>RGB+Aux DNN Confidence Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .supported {{ color: #047857; font-weight: 700; }}
    .not_supported {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>RGB+Aux DNN Confidence Sweep</h1>
  <p>Status: <code class=\"{status_class}\">{html_lib.escape(str(summary.get('status', '')))}</code>;
  claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>;
  profile: <code>{html_lib.escape(str(summary.get('profile', '')))}</code>.
  {html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <table>
    <thead><tr><th>Conf</th><th>Status</th><th>Metric Pass</th><th>RGB+Aux P/R/Small/FP</th><th>RGB-only P/R/Small/FP</th><th>Deltas dP/dR/dSmall/dFP</th><th>Failed Criteria</th><th>Reports</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _row_html(row: Mapping[str, Any], destination: Path) -> str:
    rgb_aux = row.get("rgb_aux", {}) if isinstance(row.get("rgb_aux"), Mapping) else {}
    rgb_only = row.get("rgb_only", {}) if isinstance(row.get("rgb_only"), Mapping) else {}
    deltas = row.get("deltas", {}) if isinstance(row.get("deltas"), Mapping) else {}
    failed = ", ".join(str(value) for value in row.get("failed_criteria", ())) or "none"
    return (
        "<tr>"
        f"<td>{float(row.get('confidence', 0.0)):.3f}</td>"
        f"<td class=\"{'supported' if bool(row.get('pass')) else 'not_supported'}\">{html_lib.escape('pass' if bool(row.get('pass')) else 'fail')}</td>"
        f"<td>{html_lib.escape('yes' if bool(row.get('metric_pass')) else 'no')}</td>"
        f"<td>{_metric_group(rgb_aux)}</td>"
        f"<td>{_metric_group(rgb_only)}</td>"
        f"<td>{_fmt(deltas.get('precision@0.50_mean'), signed=True)} / {_fmt(deltas.get('recall@0.50_mean'), signed=True)} / {_fmt(deltas.get('small_recall@0.50_mean'), signed=True)} / {_fmt(deltas.get('fp@0.50_mean'), signed=True)}</td>"
        f"<td>{html_lib.escape(failed)}</td>"
        f"<td>{_source_link(rgb_aux, destination)} / {_source_link(rgb_only, destination)}</td>"
        "</tr>"
    )


def _metric_group(row: Mapping[str, Any]) -> str:
    return (
        f"{_fmt(row.get('precision@0.50_mean'))} / "
        f"{_fmt(row.get('recall@0.50_mean'))} / "
        f"{_fmt(row.get('small_recall@0.50_mean'))} / "
        f"{_fmt(row.get('fp@0.50_mean'))}"
    )


def _source_link(row: Mapping[str, Any], destination: Path) -> str:
    html_path = str(row.get("html_path", ""))
    label = html_lib.escape(str(row.get("channel_mode", "")) or "open")
    if not html_path:
        return label
    import os

    href = os.path.relpath(html_path, start=str(destination))
    return f"<a href=\"{html_lib.escape(href)}\">{label}</a>"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
