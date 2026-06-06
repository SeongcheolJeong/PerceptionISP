"""Roll up RGB+aux export, training, and dense-eval summaries."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_CANDIDATES = (
    "train_dense_summary.json",
    "train_smoke_summary.json",
    "dense_eval_summary.json",
    "summary.json",
)

METRIC_KEYS = ("precision@0.50_mean", "recall@0.50_mean", "fp@0.50_mean", "det_count_mean")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a rollup report from RGB+aux export/training/eval summaries.")
    parser.add_argument("summaries", nargs="+", help="Summary JSON paths or directories containing a known summary file.")
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_training_rollup")
    args = parser.parse_args(argv)

    rollup = build_training_rollup(args.summaries)
    html_path = write_training_rollup(rollup, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "run_count": rollup["run_count"]}), indent=2))
    return 0


def build_training_rollup(paths: Sequence[str | Path]) -> Dict[str, Any]:
    runs = []
    for raw_path in paths:
        summary_path = _summary_path(raw_path)
        summary = json.loads(summary_path.read_text())
        runs.append(_run_row(summary_path, summary))
    return {
        "run_count": int(len(runs)),
        "metric_keys": list(METRIC_KEYS),
        "runs": runs,
        "interpretation": "Training rows quantify local RGB+aux data-path cost. Dense-eval rows are diagnostic only and are not detector-performance claims.",
    }


def write_training_rollup(rollup: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "training_rollup_summary.json").write_text(json.dumps(json_ready(rollup), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(rollup, destination))
    return html_path


def _summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        for filename in SUMMARY_CANDIDATES:
            summary = candidate / filename
            if summary.exists():
                return summary
        raise FileNotFoundError(f"known RGB+aux summary not found in: {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"RGB+aux summary not found: {candidate}")
    return candidate


def _run_row(summary_path: Path, summary: Mapping[str, Any]) -> Dict[str, Any]:
    kind = _summary_kind(summary_path, summary)
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    checkpoint_summary = summary.get("checkpoint_summary", {}) if isinstance(summary.get("checkpoint_summary"), Mapping) else {}
    channel_mode = summary.get("channel_mode", checkpoint_summary.get("channel_mode"))
    throughput_key = "sample_epochs_per_second" if summary.get("sample_epochs_per_second") is not None else "samples_per_second"
    time_estimates = summary.get("time_estimates", ())
    if not isinstance(time_estimates, (list, tuple)):
        time_estimates = ()
    row = {
        "name": summary_path.parent.name,
        "kind": kind,
        "summary_path": str(summary_path),
        "html_path": str(summary_path.with_name("index.html")) if summary_path.with_name("index.html").exists() else None,
        "sample_count": _maybe_int(summary.get("sample_count")),
        "train_sample_count": _maybe_int(summary.get("train_sample_count")),
        "eval_sample_count": _maybe_int(summary.get("eval_sample_count")),
        "epochs": _maybe_int(summary.get("epochs")),
        "device": summary.get("device"),
        "channel_mode": channel_mode,
        "elapsed_seconds": _maybe_float(summary.get("elapsed_seconds")),
        "throughput_key": throughput_key,
        "throughput": _maybe_float(summary.get(throughput_key)),
        "seconds_per_sample_epoch": _maybe_float(summary.get("seconds_per_sample_epoch")),
        "final_eval_loss": _maybe_float(summary.get("final_eval_loss")),
        "checkpoint_loss": _maybe_float(summary.get("checkpoint_loss")),
        "checkpoint_loss_kind": summary.get("checkpoint_loss_kind"),
        "checkpoint": summary.get("checkpoint"),
        "final_checkpoint": summary.get("final_checkpoint"),
        "time_estimates": list(time_estimates),
        "metrics": {key: _maybe_float(aggregate.get(key)) for key in METRIC_KEYS},
    }
    return row


def _summary_kind(summary_path: Path, summary: Mapping[str, Any]) -> str:
    name = summary_path.name
    if name == "train_dense_summary.json":
        return "train_dense"
    if name == "train_smoke_summary.json":
        return "train_smoke"
    if name == "dense_eval_summary.json":
        return "dense_eval"
    if summary.get("samples_per_second") is not None and summary.get("epochs") is None:
        return "export"
    return "summary"


def _render_html(rollup: Mapping[str, Any], destination: Path) -> str:
    rows = []
    estimate_rows = []
    for run in rollup.get("runs", ()):
        report_link = _relative_report_link(run, destination)
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(run.get('name', '')))}</td>"
            f"<td>{html_lib.escape(str(run.get('kind', '')))}</td>"
            f"<td>{report_link}</td>"
            f"<td>{_fmt(run.get('sample_count'), digits=0)}</td>"
            f"<td>{_fmt(run.get('train_sample_count'), digits=0)}</td>"
            f"<td>{_fmt(run.get('eval_sample_count'), digits=0)}</td>"
            f"<td>{_fmt(run.get('epochs'), digits=0)}</td>"
            f"<td>{html_lib.escape(str(run.get('device') or ''))}</td>"
            f"<td>{html_lib.escape(str(run.get('channel_mode') or ''))}</td>"
            f"<td>{_fmt(run.get('elapsed_seconds'))}</td>"
            f"<td>{html_lib.escape(str(run.get('throughput_key') or ''))}</td>"
            f"<td>{_fmt(run.get('throughput'))}</td>"
            f"<td>{_fmt(run.get('final_eval_loss'))}</td>"
            f"<td>{_fmt(run.get('metrics', {}).get('precision@0.50_mean'))}</td>"
            f"<td>{_fmt(run.get('metrics', {}).get('recall@0.50_mean'))}</td>"
            f"<td>{_fmt(run.get('metrics', {}).get('fp@0.50_mean'))}</td>"
            f"<td>{_fmt(run.get('metrics', {}).get('det_count_mean'))}</td>"
            "</tr>"
        )
        for estimate in run.get("time_estimates", ()):
            estimate_rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(run.get('name', '')))}</td>"
                f"<td>{_fmt(estimate.get('samples'), digits=0)}</td>"
                f"<td>{_fmt(estimate.get('epochs'), digits=0)}</td>"
                f"<td>{_fmt(estimate.get('estimated_minutes'))}</td>"
                f"<td>{_fmt(estimate.get('estimated_hours'))}</td>"
                "</tr>"
            )
    estimate_body = "".join(estimate_rows) if estimate_rows else '<tr><td colspan="5">No time estimates recorded.</td></tr>'
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP RGB+Aux Training Rollup</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP RGB+Aux Training Rollup</h1>
  <div class=\"note\">{html_lib.escape(str(rollup.get('interpretation', '')))}</div>
  <table>
    <thead><tr><th>Run</th><th>Kind</th><th>Report</th><th>Samples</th><th>Train</th><th>Eval</th><th>Epochs</th><th>Device</th><th>Channels</th><th>Seconds</th><th>Rate</th><th>Rate Value</th><th>Eval Loss</th><th>P@0.50</th><th>R@0.50</th><th>FP@0.50</th><th>Det/sample</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Time Estimates</h2>
  <table>
    <thead><tr><th>Run</th><th>Samples</th><th>Epochs</th><th>Minutes</th><th>Hours</th></tr></thead>
    <tbody>{estimate_body}</tbody>
  </table>
  <p>Raw JSON: <code>training_rollup_summary.json</code></p>
</body>
</html>
"""


def _relative_report_link(run: Mapping[str, Any], destination: Path) -> str:
    html_path = run.get("html_path")
    if not html_path:
        return ""
    relative = os.path.relpath(str(html_path), start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _fmt(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return ""
    if digits <= 0:
        return str(int(value))
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
