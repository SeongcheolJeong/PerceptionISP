"""Roll up multiple comparison reports into one compact summary."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import json_ready


DEFAULT_INPUTS = (
    "reference_rgb",
    "human_rgb",
    "perception_rgb",
    "perception_fusion_rgb_aux",
    "perception_calibrated_fusion_rgb_aux",
    "perception_rgb_aux_dnn",
)
METRIC_KEYS = ("precision@0.50_mean", "recall@0.50_mean", "recall@0.75_mean", "small_recall@0.50_mean", "fp@0.50_mean", "det_count_mean")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a rollup report from comparison_summary.json files.")
    parser.add_argument("reports", nargs="+", help="comparison_summary.json paths or report directories.")
    parser.add_argument("--baseline-input", default="human_rgb", help="Input name used for delta columns.")
    parser.add_argument("--output-dir", default="reports/perception_comparison_rollup")
    args = parser.parse_args(argv)

    rollup = build_rollup(args.reports, baseline_input=str(args.baseline_input))
    html_path = write_rollup_report(rollup, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "run_count": rollup["run_count"]}), indent=2))
    return 0


def build_rollup(paths: Sequence[str | Path], *, baseline_input: str = "human_rgb") -> Dict[str, Any]:
    runs = []
    for raw_path in paths:
        summary_path = _summary_path(raw_path)
        summary = json.loads(summary_path.read_text())
        aggregate = summary.get("aggregate", {})
        run_config = summary.get("run_config", {})
        baseline = aggregate.get(str(baseline_input), {})
        inputs = {}
        for input_name in _input_names(aggregate):
            metrics = aggregate.get(input_name, {})
            row = {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}
            if input_name != str(baseline_input) and baseline:
                row["delta_precision@0.50_mean"] = _delta(metrics, baseline, "precision@0.50_mean")
                row["delta_recall@0.50_mean"] = _delta(metrics, baseline, "recall@0.50_mean")
                row["delta_small_recall@0.50_mean"] = _delta(metrics, baseline, "small_recall@0.50_mean")
            inputs[input_name] = row
        runs.append(
            {
                "name": _run_name(summary_path, run_config),
                "summary_path": str(summary_path),
                "html_path": str(summary_path.with_name("index.html")) if summary_path.with_name("index.html").exists() else None,
                "sample_count": int(summary.get("sample_count", run_config.get("count", 0))),
                "run_config": dict(run_config),
                "inputs": inputs,
            }
        )
    _disambiguate_run_names(runs)
    return {
        "run_count": int(len(runs)),
        "baseline_input": str(baseline_input),
        "metric_keys": list(METRIC_KEYS),
        "runs": runs,
    }


def write_rollup_report(rollup: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "rollup_summary.json"
    json_path.write_text(json.dumps(json_ready(rollup), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(rollup, destination))
    return html_path


def _summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "comparison_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"comparison summary not found: {candidate}")
    return candidate


def _run_name(summary_path: Path, run_config: Mapping[str, Any]) -> str:
    count = run_config.get("count")
    source = run_config.get("source")
    suffix = _calibration_name_suffix(run_config)
    if count is not None and source:
        base = f"{source} {count}"
    else:
        base = summary_path.parent.name
    return f"{base} - {suffix}" if suffix else base


def _calibration_name_suffix(run_config: Mapping[str, Any]) -> str:
    calibration = run_config.get("proposal_calibration", {})
    if not isinstance(calibration, Mapping):
        return ""
    feature_set = str(calibration.get("feature_set", "") or "")
    output_input = str(calibration.get("output_input", "") or "")
    if feature_set:
        return f"calibrated {feature_set}"
    if output_input:
        return output_input
    return ""


def _disambiguate_run_names(runs: Sequence[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for run in runs:
        name = str(run.get("name", ""))
        counts[name] = counts.get(name, 0) + 1
    for run in runs:
        name = str(run.get("name", ""))
        if counts.get(name, 0) <= 1:
            continue
        summary_path = Path(str(run.get("summary_path", "")))
        qualifier = summary_path.parent.name if summary_path.name else ""
        if qualifier:
            run["name"] = f"{name} ({qualifier})"


def _input_names(aggregate: Mapping[str, Any]) -> Tuple[str, ...]:
    names = set(str(name) for name in aggregate)
    ordered = [name for name in DEFAULT_INPUTS if name in names]
    ordered.extend(sorted(name for name in names if name not in DEFAULT_INPUTS))
    return tuple(ordered)


def _delta(metrics: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> float:
    return float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))


def _render_html(rollup: Mapping[str, Any], destination: Path) -> str:
    rows = []
    baseline_input = str(rollup.get("baseline_input", "human_rgb"))
    for run in rollup.get("runs", ()):
        run_name = html_lib.escape(str(run.get("name", "")))
        sample_count = int(run.get("sample_count", 0))
        report_link = _relative_report_link(run, destination)
        for input_name, metrics in run.get("inputs", {}).items():
            rows.append(
                "<tr>"
                f"<td>{run_name}</td>"
                f"<td>{sample_count}</td>"
                f"<td>{report_link}</td>"
                f"<td>{html_lib.escape(str(input_name))}</td>"
                f"<td>{_fmt(metrics.get('precision@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('recall@0.75_mean'))}</td>"
                f"<td>{_fmt(metrics.get('small_recall@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('fp@0.50_mean'))}</td>"
                f"<td>{_fmt(metrics.get('delta_precision@0.50_mean'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_recall@0.50_mean'), signed=True)}</td>"
                f"<td>{_fmt(metrics.get('delta_small_recall@0.50_mean'), signed=True)}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Comparison Rollup</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 9px 10px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Comparison Rollup</h1>
  <p>Delta columns are computed against <code>{html_lib.escape(baseline_input)}</code> within each run.</p>
  <table>
    <thead><tr><th>Run</th><th>Samples</th><th>Report</th><th>Input</th><th>P@0.50</th><th>R@0.50</th><th>R@0.75</th><th>Small R@0.50</th><th>FP@0.50</th><th>dP@0.50</th><th>dR@0.50</th><th>dSmallR@0.50</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Raw JSON: <code>rollup_summary.json</code></p>
</body>
</html>
"""


def _relative_report_link(run: Mapping[str, Any], destination: Path) -> str:
    html_path = run.get("html_path")
    if not html_path:
        return ""
    relative = os.path.relpath(str(html_path), start=str(destination))
    safe_relative = html_lib.escape(relative)
    return f"<a href=\"{safe_relative}\">open</a>"


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
