"""Gate RGB+Aux DNN eval summaries against claim-quality criteria."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import json_ready


SUMMARY_FILENAME = "rgb_aux_dnn_gate_summary.json"
DENSE_EVAL_SUMMARY = "dense_eval_summary.json"

DNN_GATE_PROFILES = {
    "claim_quality": {
        "min_samples": 1000,
        "min_precision": 0.05,
        "min_recall": 0.10,
        "max_fp_per_sample": 5.0,
        "min_precision_delta": 0.0,
        "min_recall_delta": 0.0,
        "min_small_recall_delta": 0.0,
        "max_fp_delta": 0.0,
    },
    "fp_reducer": {
        "min_samples": 1000,
        "min_precision": 0.05,
        "min_recall": 0.05,
        "max_fp_per_sample": 5.0,
        "min_precision_delta": 0.0,
        "min_recall_delta": -0.01,
        "min_small_recall_delta": -0.01,
        "max_fp_delta": -0.10,
    },
    "diagnostic": {
        "min_samples": 1,
        "min_precision": 0.0,
        "min_recall": 0.0,
        "max_fp_per_sample": 1.0e9,
        "min_precision_delta": -1.0,
        "min_recall_delta": -1.0,
        "min_small_recall_delta": -1.0,
        "max_fp_delta": 1.0e9,
    },
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RGB+Aux DNN dense eval summaries against a claim gate.")
    parser.add_argument("--rgb-aux", required=True, help="RGB+Aux dense_eval_summary path/dir.")
    parser.add_argument("--rgb-only", required=True, help="RGB-only dense_eval_summary path/dir.")
    parser.add_argument("--aux-only", default=None, help="Optional aux-only dense_eval_summary path/dir.")
    parser.add_argument("--extended-rgb-aux", default=None, help="Optional extended tensor RGB+Aux dense_eval_summary path/dir.")
    parser.add_argument("--profile", default="claim_quality", choices=sorted(DNN_GATE_PROFILES))
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--max-fp-per-sample", type=float, default=None)
    parser.add_argument("--min-precision-delta", type=float, default=None)
    parser.add_argument("--min-recall-delta", type=float, default=None)
    parser.add_argument("--min-small-recall-delta", type=float, default=None)
    parser.add_argument("--max-fp-delta", type=float, default=None)
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_dnn_gate")
    args = parser.parse_args(argv)

    rgb_aux_path, rgb_aux = _load_dense_eval(args.rgb_aux)
    rgb_only_path, rgb_only = _load_dense_eval(args.rgb_only)
    aux_only_pair = _load_dense_eval(args.aux_only) if args.aux_only is not None else None
    extended_pair = _load_dense_eval(args.extended_rgb_aux) if args.extended_rgb_aux is not None else None
    summary = build_rgb_aux_dnn_gate(
        rgb_aux,
        rgb_only,
        aux_only=aux_only_pair[1] if aux_only_pair else None,
        extended_rgb_aux=extended_pair[1] if extended_pair else None,
        rgb_aux_source=rgb_aux_path,
        rgb_only_source=rgb_only_path,
        aux_only_source=aux_only_pair[0] if aux_only_pair else None,
        extended_rgb_aux_source=extended_pair[0] if extended_pair else None,
        thresholds=_cli_thresholds(args),
    )
    html_path = write_rgb_aux_dnn_gate(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "pass": summary["pass"],
                    "failed": [row["id"] for row in summary["criteria"] if row["status"] == "fail"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_rgb_aux_dnn_gate(
    rgb_aux: Mapping[str, Any],
    rgb_only: Mapping[str, Any],
    *,
    aux_only: Mapping[str, Any] | None = None,
    extended_rgb_aux: Mapping[str, Any] | None = None,
    rgb_aux_source: str | Path | None = None,
    rgb_only_source: str | Path | None = None,
    aux_only_source: str | Path | None = None,
    extended_rgb_aux_source: str | Path | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    threshold_values = _resolve_thresholds(thresholds)
    profile = str(threshold_values.pop("profile", "claim_quality"))
    runs = [
        _run_row("rgb_aux", rgb_aux, source=rgb_aux_source),
        _run_row("rgb_only", rgb_only, source=rgb_only_source),
    ]
    if aux_only is not None:
        runs.append(_run_row("aux_only", aux_only, source=aux_only_source))
    if extended_rgb_aux is not None:
        runs.append(_run_row("extended_rgb_aux", extended_rgb_aux, source=extended_rgb_aux_source))
    primary = runs[0]
    baseline = runs[1]
    deltas = {
        "precision@0.50_mean": _delta(primary, baseline, "precision@0.50_mean"),
        "recall@0.50_mean": _delta(primary, baseline, "recall@0.50_mean"),
        "small_recall@0.50_mean": _delta(primary, baseline, "small_recall@0.50_mean"),
        "fp@0.50_mean": _delta(primary, baseline, "fp@0.50_mean"),
    }
    criteria = _criteria(primary, baseline, deltas, threshold_values)
    passed = all(row["status"] == "pass" for row in criteria)
    return {
        "name": "RGB+Aux DNN gate",
        "status": "pass" if passed else "fail",
        "pass": bool(passed),
        "claim_status": "rgb_aux_dnn_claim_ready" if passed else "rgb_aux_dnn_not_claim_ready",
        "profile": profile,
        "thresholds": threshold_values,
        "primary_run": "rgb_aux",
        "baseline_run": "rgb_only",
        "runs": runs,
        "deltas": deltas,
        "criteria": criteria,
        "interpretation": _interpretation(passed),
        "claim_boundary": (
            "This gate evaluates direct compact dense-detector outputs from exported tensors. "
            "Passing it would still be compact-DNN evidence, not full YOLO-scale RGB+Aux fine-tuning proof."
        ),
    }


def write_rgb_aux_dnn_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _criteria(
    primary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    deltas: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    missing = list(primary.get("missing_eval_class_names", ()))
    return (
        _minimum_value("sample_count", primary.get("sample_count"), thresholds.get("min_samples")),
        _minimum_value("absolute_precision", primary.get("precision@0.50_mean"), thresholds.get("min_precision")),
        _minimum_value("absolute_recall", primary.get("recall@0.50_mean"), thresholds.get("min_recall")),
        _maximum_value("absolute_fp_per_sample", primary.get("fp@0.50_mean"), thresholds.get("max_fp_per_sample")),
        _minimum_delta(
            "precision_vs_rgb_only",
            primary.get("precision@0.50_mean"),
            baseline.get("precision@0.50_mean"),
            deltas.get("precision@0.50_mean"),
            thresholds.get("min_precision_delta"),
        ),
        _minimum_delta(
            "recall_vs_rgb_only",
            primary.get("recall@0.50_mean"),
            baseline.get("recall@0.50_mean"),
            deltas.get("recall@0.50_mean"),
            thresholds.get("min_recall_delta"),
        ),
        _minimum_delta(
            "small_recall_vs_rgb_only",
            primary.get("small_recall@0.50_mean"),
            baseline.get("small_recall@0.50_mean"),
            deltas.get("small_recall@0.50_mean"),
            thresholds.get("min_small_recall_delta"),
        ),
        _maximum_delta(
            "fp_vs_rgb_only",
            primary.get("fp@0.50_mean"),
            baseline.get("fp@0.50_mean"),
            deltas.get("fp@0.50_mean"),
            thresholds.get("max_fp_delta"),
        ),
        {
            "id": "eval_classes_present",
            "metric": "missing_eval_class_names",
            "direction": "empty",
            "target": len(missing),
            "threshold": 0,
            "status": "pass" if not missing else "fail",
            "pass": not missing,
            "evidence": ", ".join(str(value) for value in missing) or "none",
        },
    )


def _minimum_value(identifier: str, value: Any, threshold: Any) -> Dict[str, Any]:
    available = value is not None and threshold is not None
    passed = bool(available and float(value) >= float(threshold))
    return {
        "id": identifier,
        "direction": "minimum_value",
        "target": _maybe_float(value),
        "threshold": _maybe_float(threshold),
        "delta": None if not available else float(value) - float(threshold),
        "status": "pass" if passed else "fail",
        "pass": passed,
    }


def _maximum_value(identifier: str, value: Any, threshold: Any) -> Dict[str, Any]:
    available = value is not None and threshold is not None
    passed = bool(available and float(value) <= float(threshold))
    return {
        "id": identifier,
        "direction": "maximum_value",
        "target": _maybe_float(value),
        "threshold": _maybe_float(threshold),
        "delta": None if not available else float(value) - float(threshold),
        "status": "pass" if passed else "fail",
        "pass": passed,
    }


def _minimum_delta(identifier: str, target: Any, baseline: Any, delta: Any, threshold: Any) -> Dict[str, Any]:
    available = delta is not None and threshold is not None
    passed = bool(available and float(delta) >= float(threshold))
    return {
        "id": identifier,
        "direction": "minimum_delta",
        "baseline": _maybe_float(baseline),
        "target": _maybe_float(target),
        "delta": _maybe_float(delta),
        "threshold": _maybe_float(threshold),
        "status": "pass" if passed else "fail",
        "pass": passed,
    }


def _maximum_delta(identifier: str, target: Any, baseline: Any, delta: Any, threshold: Any) -> Dict[str, Any]:
    available = delta is not None and threshold is not None
    passed = bool(available and float(delta) <= float(threshold))
    return {
        "id": identifier,
        "direction": "maximum_delta",
        "baseline": _maybe_float(baseline),
        "target": _maybe_float(target),
        "delta": _maybe_float(delta),
        "threshold": _maybe_float(threshold),
        "status": "pass" if passed else "fail",
        "pass": passed,
    }


def _run_row(name: str, summary: Mapping[str, Any], *, source: str | Path | None) -> Dict[str, Any]:
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    checkpoint = summary.get("checkpoint_summary", {}) if isinstance(summary.get("checkpoint_summary"), Mapping) else {}
    missing = checkpoint.get("missing_eval_class_names", ()) if isinstance(checkpoint.get("missing_eval_class_names", ()), Sequence) else ()
    return {
        "name": str(name),
        "source": "" if source is None else str(source),
        "html_path": _sibling_html(source),
        "split": str(summary.get("split", "")),
        "sample_count": int(summary.get("sample_count", aggregate.get("sample_count", 0))),
        "channel_mode": str(checkpoint.get("channel_mode", "")),
        "tensor_key": str(checkpoint.get("tensor_key", "")),
        "input_channels": _maybe_int(checkpoint.get("input_channels")),
        "missing_eval_class_names": [str(value) for value in missing],
        "precision@0.50_mean": _maybe_float(aggregate.get("precision@0.50_mean")),
        "recall@0.50_mean": _maybe_float(aggregate.get("recall@0.50_mean")),
        "recall@0.75_mean": _maybe_float(aggregate.get("recall@0.75_mean")),
        "small_recall@0.50_mean": _maybe_float(aggregate.get("small_recall@0.50_mean")),
        "fp@0.50_mean": _maybe_float(aggregate.get("fp@0.50_mean")),
        "det_count_mean": _maybe_float(aggregate.get("det_count_mean")),
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


def _sibling_html(source: str | Path | None) -> str:
    if source is None:
        return ""
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        html_path = candidate / "index.html"
    else:
        html_path = candidate.with_name("index.html")
    return str(html_path) if html_path.exists() else ""


def _resolve_thresholds(thresholds: Mapping[str, Any] | None) -> Dict[str, Any]:
    values = dict(DNN_GATE_PROFILES["claim_quality"])
    if thresholds is not None:
        profile = str(thresholds.get("profile", "claim_quality"))
        values = dict(DNN_GATE_PROFILES.get(profile, DNN_GATE_PROFILES["claim_quality"]))
        values["profile"] = profile
        for key, value in thresholds.items():
            if key == "profile" or value is None:
                continue
            values[key] = value
    else:
        values["profile"] = "claim_quality"
    return values


def _cli_thresholds(args: Any) -> Dict[str, Any]:
    values: Dict[str, Any] = {"profile": str(args.profile)}
    for key, attr in (
        ("min_samples", "min_samples"),
        ("min_precision", "min_precision"),
        ("min_recall", "min_recall"),
        ("max_fp_per_sample", "max_fp_per_sample"),
        ("min_precision_delta", "min_precision_delta"),
        ("min_recall_delta", "min_recall_delta"),
        ("min_small_recall_delta", "min_small_recall_delta"),
        ("max_fp_delta", "max_fp_delta"),
    ):
        value = getattr(args, attr)
        if value is not None:
            values[key] = value
    return values


def _delta(target: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> float | None:
    target_value = target.get(key)
    baseline_value = baseline.get(key)
    if target_value is None or baseline_value is None:
        return None
    return float(target_value) - float(baseline_value)


def _interpretation(passed: bool) -> str:
    if passed:
        return "The compact RGB+Aux DNN evaluation passes the configured gate versus RGB-only."
    return (
        "The compact RGB+Aux DNN evaluation does not pass the configured gate versus RGB-only. "
        "Do not claim learned RGB+Aux detector improvement from this evidence."
    )


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    status_class = "supported" if bool(summary.get("pass")) else "not_supported"
    run_rows = "".join(_run_html_row(row, destination) for row in summary.get("runs", ()) if isinstance(row, Mapping))
    criteria_rows = "".join(_criterion_html_row(row) for row in summary.get("criteria", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>RGB+Aux DNN Gate</title>
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
  <h1>RGB+Aux DNN Gate</h1>
  <p>Status: <code class=\"{status_class}\">{html_lib.escape(str(summary.get('status', '')))}</code>;
  claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.
  {html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <table><thead><tr><th>Profile</th><th>Primary</th><th>Baseline</th><th>dP50</th><th>dR50</th><th>dSmallR50</th><th>dFP50</th></tr></thead><tbody><tr>
    <td><code>{html_lib.escape(str(summary.get('profile', '')))}</code></td>
    <td><code>{html_lib.escape(str(summary.get('primary_run', '')))}</code></td>
    <td><code>{html_lib.escape(str(summary.get('baseline_run', '')))}</code></td>
    <td>{_fmt((summary.get('deltas') or {}).get('precision@0.50_mean'), signed=True)}</td>
    <td>{_fmt((summary.get('deltas') or {}).get('recall@0.50_mean'), signed=True)}</td>
    <td>{_fmt((summary.get('deltas') or {}).get('small_recall@0.50_mean'), signed=True)}</td>
    <td>{_fmt((summary.get('deltas') or {}).get('fp@0.50_mean'), signed=True)}</td>
  </tr></tbody></table>
  <h2>Gate Criteria</h2>
  <table><thead><tr><th>Criterion</th><th>Status</th><th>Target</th><th>Baseline</th><th>Delta</th><th>Threshold</th><th>Direction</th><th>Evidence</th></tr></thead><tbody>{criteria_rows}</tbody></table>
  <h2>Dense Eval Runs</h2>
  <table><thead><tr><th>Run</th><th>Report</th><th>Samples</th><th>Mode</th><th>Tensor</th><th>P50</th><th>R50</th><th>Small R50</th><th>FP50</th><th>Det/sample</th><th>Missing Classes</th></tr></thead><tbody>{run_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _criterion_html_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{'supported' if status == 'pass' else 'not_supported'}\">{html_lib.escape(status)}</td>"
        f"<td>{_fmt(row.get('target'))}</td>"
        f"<td>{_fmt(row.get('baseline'))}</td>"
        f"<td>{_fmt(row.get('delta'), signed=True)}</td>"
        f"<td>{_fmt(row.get('threshold'))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('direction', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _run_html_row(row: Mapping[str, Any], destination: Path) -> str:
    link = html_lib.escape(str(row.get("name", "")))
    if row.get("html_path"):
        relative = Path(str(row.get("html_path")))
        try:
            href = relative.relative_to(destination)
        except ValueError:
            href = Path(_relpath(relative, destination))
        link = f"<a href=\"{html_lib.escape(str(href))}\">{link}</a>"
    missing = ", ".join(str(value) for value in row.get("missing_eval_class_names", ())) or "none"
    tensor = str(row.get("tensor_key", "")) or "rgb_aux_chw"
    channels = row.get("input_channels")
    tensor_text = tensor if channels is None else f"{tensor} ({int(channels)}ch)"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('name', '')))}</code></td>"
        f"<td>{link}</td>"
        f"<td>{int(row.get('sample_count', 0))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('channel_mode', '')))}</code></td>"
        f"<td><code>{html_lib.escape(tensor_text)}</code></td>"
        f"<td>{_fmt(row.get('precision@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('recall@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('small_recall@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('fp@0.50_mean'))}</td>"
        f"<td>{_fmt(row.get('det_count_mean'))}</td>"
        f"<td>{html_lib.escape(missing)}</td>"
        "</tr>"
    )


def _relpath(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(str(path), start=str(start))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
