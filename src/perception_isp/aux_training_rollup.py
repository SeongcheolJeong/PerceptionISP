"""Roll up RGB+aux export, training, and dense-eval summaries."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import statistics
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
DEFAULT_PLANNING_SCENARIOS = (
    ("KITTI val-sized compact check", 1496, 5),
    ("KITTI train compact check", 5985, 5),
    ("KITTI train stronger compact run", 5985, 50),
    ("KITTI train exhaustive compact run", 5985, 100),
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a rollup report from RGB+aux export/training/eval summaries.")
    parser.add_argument("summaries", nargs="+", help="Summary JSON paths or directories containing a known summary file.")
    parser.add_argument(
        "--plan-scenario",
        action="append",
        default=[],
        help="Training-time scenario as name=samples,epochs. Defaults cover KITTI-sized compact runs.",
    )
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_training_rollup")
    args = parser.parse_args(argv)

    rollup = build_training_rollup(args.summaries, planning_scenarios=parse_planning_scenarios(args.plan_scenario))
    html_path = write_training_rollup(rollup, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "run_count": rollup["run_count"]}), indent=2))
    return 0


def build_training_rollup(
    paths: Sequence[str | Path],
    *,
    planning_scenarios: Sequence[tuple[str, int, int]] | None = None,
) -> Dict[str, Any]:
    runs = []
    for raw_path in paths:
        summary_path = _summary_path(raw_path)
        summary = json.loads(summary_path.read_text())
        runs.append(_run_row(summary_path, summary))
    planning = _training_time_plan(runs, planning_scenarios or DEFAULT_PLANNING_SCENARIOS)
    return {
        "run_count": int(len(runs)),
        "metric_keys": list(METRIC_KEYS),
        "runs": runs,
        "training_time_plan": planning,
        "interpretation": "Training rows quantify local RGB+aux data-path cost. Dense-eval rows are diagnostic only and are not detector-performance claims.",
    }


def write_training_rollup(rollup: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "training_rollup_summary.json").write_text(json.dumps(json_ready(rollup), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(rollup, destination))
    return html_path


def parse_planning_scenarios(values: Sequence[str]) -> tuple[tuple[str, int, int], ...]:
    if not values:
        return DEFAULT_PLANNING_SCENARIOS
    scenarios = []
    for value in values:
        if "=" not in str(value) or "," not in str(value):
            raise ValueError("plan scenario must be formatted as name=samples,epochs")
        name, raw_counts = str(value).split("=", 1)
        samples_text, epochs_text = raw_counts.split(",", 1)
        name = name.strip()
        samples = int(samples_text.strip())
        epochs = int(epochs_text.strip())
        if not name or samples <= 0 or epochs <= 0:
            raise ValueError("plan scenario must contain a name, positive samples, and positive epochs")
        scenarios.append((name, samples, epochs))
    return tuple(scenarios)


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
        "tensor_key": summary.get("tensor_key"),
        "input_channels": _maybe_int(summary.get("input_channels")),
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


def _training_time_plan(runs: Sequence[Mapping[str, Any]], scenarios: Sequence[tuple[str, int, int]]) -> Dict[str, Any]:
    train_rates = [
        float(run.get("throughput"))
        for run in runs
        if str(run.get("kind", "")).startswith("train") and run.get("throughput_key") == "sample_epochs_per_second" and run.get("throughput") is not None
    ]
    export_rates = [
        float(run.get("throughput"))
        for run in runs
        if run.get("kind") == "export" and run.get("throughput_key") == "samples_per_second" and run.get("throughput") is not None
    ]
    if not train_rates:
        return {
            "status": "missing_train_rate",
            "scenario_count": 0,
            "interpretation": "No train sample-epochs/sec measurements were available, so training-time planning is not possible.",
            "scenarios": [],
        }
    train_rate_summary = _rate_summary(train_rates)
    export_rate_summary = _rate_summary(export_rates) if export_rates else None
    rows = [
        _scenario_estimate(
            name=name,
            samples=samples,
            epochs=epochs,
            train_rate_summary=train_rate_summary,
            export_rate_summary=export_rate_summary,
        )
        for name, samples, epochs in scenarios
    ]
    return {
        "status": "estimated",
        "scenario_count": len(rows),
        "train_rate": train_rate_summary,
        "export_rate": export_rate_summary,
        "scenarios": rows,
        "interpretation": "Estimates use observed local throughput; compact-detector timing is not a claim-quality detector-training guarantee.",
    }


def _rate_summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(value) for value in values if float(value) > 0.0]
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "median": statistics.median(clean),
        "max": max(clean),
    }


def _scenario_estimate(
    *,
    name: str,
    samples: int,
    epochs: int,
    train_rate_summary: Mapping[str, Any],
    export_rate_summary: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    train_typical = _estimate_seconds(sample_epochs=int(samples) * int(epochs), rate=train_rate_summary.get("median"))
    train_conservative = _estimate_seconds(sample_epochs=int(samples) * int(epochs), rate=train_rate_summary.get("min"))
    train_optimistic = _estimate_seconds(sample_epochs=int(samples) * int(epochs), rate=train_rate_summary.get("max"))
    export_typical = _estimate_seconds(sample_epochs=int(samples), rate=None if export_rate_summary is None else export_rate_summary.get("median"))
    total_typical = None if train_typical is None else float(train_typical + (export_typical or 0.0))
    return {
        "name": str(name),
        "samples": int(samples),
        "epochs": int(epochs),
        "sample_epochs": int(samples) * int(epochs),
        "train_seconds_typical": train_typical,
        "train_minutes_typical": _seconds_to_minutes(train_typical),
        "train_hours_typical": _seconds_to_hours(train_typical),
        "train_seconds_conservative": train_conservative,
        "train_minutes_conservative": _seconds_to_minutes(train_conservative),
        "train_hours_conservative": _seconds_to_hours(train_conservative),
        "train_seconds_optimistic": train_optimistic,
        "train_minutes_optimistic": _seconds_to_minutes(train_optimistic),
        "train_hours_optimistic": _seconds_to_hours(train_optimistic),
        "export_seconds_typical": export_typical,
        "export_minutes_typical": _seconds_to_minutes(export_typical),
        "total_seconds_typical": total_typical,
        "total_minutes_typical": _seconds_to_minutes(total_typical),
        "total_hours_typical": _seconds_to_hours(total_typical),
    }


def _estimate_seconds(*, sample_epochs: int, rate: Any) -> float | None:
    if rate is None:
        return None
    rate_value = float(rate)
    if rate_value <= 0.0:
        return None
    return float(int(sample_epochs) / rate_value)


def _seconds_to_minutes(seconds: float | None) -> float | None:
    return None if seconds is None else float(seconds / 60.0)


def _seconds_to_hours(seconds: float | None) -> float | None:
    return None if seconds is None else float(seconds / 3600.0)


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
            f"<td>{html_lib.escape(str(run.get('tensor_key') or ''))}</td>"
            f"<td>{_fmt(run.get('input_channels'), digits=0)}</td>"
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
    plan_html = _training_time_plan_html(rollup.get("training_time_plan", {}))
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
    <thead><tr><th>Run</th><th>Kind</th><th>Report</th><th>Samples</th><th>Train</th><th>Eval</th><th>Epochs</th><th>Device</th><th>Tensor</th><th>Input Ch</th><th>Channels</th><th>Seconds</th><th>Rate</th><th>Rate Value</th><th>Eval Loss</th><th>P@0.50</th><th>R@0.50</th><th>FP@0.50</th><th>Det/sample</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Time Estimates</h2>
  <table>
    <thead><tr><th>Run</th><th>Samples</th><th>Epochs</th><th>Minutes</th><th>Hours</th></tr></thead>
    <tbody>{estimate_body}</tbody>
  </table>
  <h2>Training-Time Plan</h2>
  {plan_html}
  <p>Raw JSON: <code>training_rollup_summary.json</code></p>
</body>
</html>
"""


def _training_time_plan_html(plan: Any) -> str:
    if not isinstance(plan, Mapping) or plan.get("status") != "estimated":
        return "<p>No training-time plan is available.</p>"
    train_rate = plan.get("train_rate", {}) if isinstance(plan.get("train_rate"), Mapping) else {}
    export_rate = plan.get("export_rate", {}) if isinstance(plan.get("export_rate"), Mapping) else {}
    plan_rows = "".join(
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('name', '')))}</td>"
        f"<td>{_fmt(row.get('samples'), digits=0)}</td>"
        f"<td>{_fmt(row.get('epochs'), digits=0)}</td>"
        f"<td>{_fmt(row.get('train_minutes_typical'))}</td>"
        f"<td>{_fmt(row.get('train_minutes_conservative'))}</td>"
        f"<td>{_fmt(row.get('train_hours_typical'))}</td>"
        f"<td>{_fmt(row.get('export_minutes_typical'))}</td>"
        f"<td>{_fmt(row.get('total_minutes_typical'))}</td>"
        "</tr>"
        for row in plan.get("scenarios", ())
    )
    if not plan_rows:
        plan_rows = '<tr><td colspan="8">No planning scenarios were available.</td></tr>'
    return (
        f"<p>{html_lib.escape(str(plan.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Train Rate Count</th><th>Median Train Rate</th><th>Min Train Rate</th><th>Max Train Rate</th><th>Median Export Rate</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_fmt(train_rate.get('count'), digits=0)}</td>"
        f"<td>{_fmt(train_rate.get('median'))} sample-epochs/s</td>"
        f"<td>{_fmt(train_rate.get('min'))} sample-epochs/s</td>"
        f"<td>{_fmt(train_rate.get('max'))} sample-epochs/s</td>"
        f"<td>{_fmt(export_rate.get('median'))} samples/s</td>"
        "</tr></tbody></table>"
        "<table>"
        "<thead><tr><th>Scenario</th><th>Samples</th><th>Epochs</th><th>Typical Train Min</th><th>Conservative Train Min</th><th>Typical Train Hours</th><th>Typical Export Min</th><th>Typical Total Min</th></tr></thead>"
        f"<tbody>{plan_rows}</tbody></table>"
    )


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
