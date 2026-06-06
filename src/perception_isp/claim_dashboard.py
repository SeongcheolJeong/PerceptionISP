"""Assemble PerceptionISP claim-readiness evidence into one report."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import json_ready


CLAIM_GATE_SUMMARY = "claim_gate_summary.json"
TRAINING_ROLLUP_SUMMARY = "training_rollup_summary.json"
COMPARISON_ROLLUP_SUMMARY = "rollup_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a consolidated PerceptionISP claim-readiness dashboard.")
    parser.add_argument("--claim-gate", action="append", default=[], help="Claim gate summary path/dir, optionally name=path.")
    parser.add_argument("--training-rollup", default=None, help="RGB+aux training rollup summary path/dir.")
    parser.add_argument("--comparison-rollup", action="append", default=[], help="Comparison rollup summary path/dir, optionally name=path.")
    parser.add_argument("--output-dir", default="reports/perception_claim_readiness_dashboard")
    args = parser.parse_args(argv)

    dashboard = build_claim_dashboard(
        claim_gate_specs=args.claim_gate,
        training_rollup=args.training_rollup,
        comparison_rollup_specs=args.comparison_rollup,
    )
    html_path = write_claim_dashboard(dashboard, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "claim_dashboard_summary.json"),
                    "decision_count": len(dashboard["decisions"]),
                    "claim_count": len(dashboard["claims"]),
                }
            ),
            indent=2,
        )
    )
    return 0


def build_claim_dashboard(
    *,
    claim_gate_specs: Sequence[str | Path],
    training_rollup: str | Path | None = None,
    comparison_rollup_specs: Sequence[str | Path] = (),
) -> Dict[str, Any]:
    claims = [_load_claim_gate(spec) for spec in claim_gate_specs]
    training = _load_training_rollup(training_rollup) if training_rollup is not None else None
    comparison_rollups = [_load_comparison_rollup(spec) for spec in comparison_rollup_specs]
    decisions = _claim_decisions(claims, training)
    return {
        "claims": claims,
        "training": training,
        "comparison_rollups": comparison_rollups,
        "decisions": decisions,
        "interpretation": "This dashboard separates supported engineering claims from claims that the current evidence does not support.",
    }


def write_claim_dashboard(dashboard: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "claim_dashboard_summary.json").write_text(json.dumps(json_ready(dashboard), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(dashboard, destination))
    return html_path


def _load_claim_gate(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, CLAIM_GATE_SUMMARY)
    data = json.loads(summary_path.read_text())
    failed = [item for item in data.get("criteria", ()) if not bool(item.get("pass"))]
    metrics = {item.get("metric"): _criterion_summary(item) for item in data.get("criteria", ()) if item.get("metric") != "sample_count"}
    profile = str(data.get("profile", "custom"))
    passed = bool(data.get("pass"))
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "profile": profile,
        "verdict": str(data.get("verdict", "")),
        "pass": passed,
        "sample_count": int(data.get("sample_count", 0)),
        "target_input": str(data.get("target_input", "")),
        "baseline_input": str(data.get("baseline_input", "")),
        "claim": _claim_text(profile, passed),
        "failed_metrics": [str(item.get("metric", "")) for item in failed],
        "metrics": metrics,
        "interpretation": str(data.get("interpretation", "")),
    }


def _criterion_summary(item: Mapping[str, Any]) -> Dict[str, Any]:
    paired = item.get("paired_delta", {}) if isinstance(item.get("paired_delta"), Mapping) else {}
    return {
        "delta": _maybe_float(item.get("delta")),
        "threshold": _maybe_float(item.get("threshold")),
        "ci_low": _maybe_float(paired.get("ci_low")),
        "ci_high": _maybe_float(paired.get("ci_high")),
        "pass": bool(item.get("pass")),
    }


def _load_training_rollup(path: str | Path) -> Dict[str, Any]:
    summary_path = _summary_path(path, TRAINING_ROLLUP_SUMMARY)
    data = json.loads(summary_path.read_text())
    runs = list(data.get("runs", ()))
    train_runs = [run for run in runs if str(run.get("kind", "")).startswith("train")]
    dense_eval_runs = [run for run in runs if str(run.get("kind", "")) == "dense_eval"]
    best_train = _max_by(train_runs, "throughput")
    best_recall = _max_metric(dense_eval_runs, "recall@0.50_mean")
    lowest_fp = _min_metric(dense_eval_runs, "fp@0.50_mean")
    max_precision = _max_metric(dense_eval_runs, "precision@0.50_mean")
    status = _training_status(best_recall, lowest_fp, max_precision, train_runs, dense_eval_runs)
    return {
        "name": _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "run_count": int(data.get("run_count", len(runs))),
        "status": status,
        "best_train_rate": _training_run_brief(best_train),
        "best_recall_eval": _dense_eval_brief(best_recall),
        "lowest_fp_eval": _dense_eval_brief(lowest_fp),
        "max_precision_eval": _dense_eval_brief(max_precision),
        "interpretation": _training_interpretation(status),
    }


def _load_comparison_rollup(spec: str | Path) -> Dict[str, Any]:
    label, path = _split_named_path(spec)
    summary_path = _summary_path(path, COMPARISON_ROLLUP_SUMMARY)
    data = json.loads(summary_path.read_text())
    return {
        "name": label or _default_name(summary_path),
        "summary_path": str(summary_path),
        "html_path": _sibling_html(summary_path),
        "run_count": int(data.get("run_count", 0)),
        "baseline_input": str(data.get("baseline_input", "")),
    }


def _claim_decisions(claims: Sequence[Mapping[str, Any]], training: Mapping[str, Any] | None) -> list[Dict[str, Any]]:
    decisions: list[Dict[str, Any]] = []
    broad_claims = [claim for claim in claims if claim.get("profile") == "broad_superiority"]
    fp_claims = [claim for claim in claims if claim.get("profile") == "fp_reducer"]
    if any(bool(claim.get("pass")) for claim in broad_claims):
        decisions.append({"status": "supported", "claim": "Broad metric superiority versus HumanISP is supported by the configured gate."})
    elif broad_claims:
        decisions.append({"status": "not_supported", "claim": "Broad HumanISP superiority is not supported by the current gate evidence."})
    if any(bool(claim.get("pass")) for claim in fp_claims):
        decisions.append({"status": "supported", "claim": "Recall-budgeted FP reduction versus the RGB+Aux fusion baseline is supported."})
    elif fp_claims:
        decisions.append({"status": "not_supported", "claim": "Recall-budgeted FP reduction is not supported by the current gate evidence."})
    if training is not None:
        status = str(training.get("status", "unknown"))
        if status == "diagnostic_only":
            decisions.append({"status": "not_supported", "claim": "The learned RGB+Aux DNN path is implemented and trainable, but current dense-detector metrics are not claim-quality."})
        elif status == "candidate_needs_gate":
            decisions.append({"status": "needs_gate", "claim": "The learned RGB+Aux DNN path has candidate metrics, but still needs a held-out claim gate."})
        elif status == "training_path_only":
            decisions.append({"status": "needs_eval", "claim": "The RGB+Aux DNN training path exists, but direct held-out detector evaluation is still missing."})
    return decisions


def _training_status(
    best_recall: Mapping[str, Any] | None,
    lowest_fp: Mapping[str, Any] | None,
    max_precision: Mapping[str, Any] | None,
    train_runs: Sequence[Mapping[str, Any]],
    dense_eval_runs: Sequence[Mapping[str, Any]],
) -> str:
    if dense_eval_runs:
        precision = _metric(max_precision, "precision@0.50_mean")
        fp = _metric(lowest_fp, "fp@0.50_mean")
        if precision is not None and fp is not None and (precision < 0.05 or fp > 5.0):
            return "diagnostic_only"
        return "candidate_needs_gate"
    if train_runs:
        return "training_path_only"
    return "missing_training"


def _training_interpretation(status: str) -> str:
    if status == "diagnostic_only":
        return "Training is fast enough for iteration, but the compact dense detector currently has weak precision/FP behavior."
    if status == "candidate_needs_gate":
        return "Direct RGB+Aux DNN metrics should be promoted only after a held-out claim gate."
    if status == "training_path_only":
        return "Training summaries exist, but direct dense detector eval has not been rolled up."
    return "No RGB+Aux training evidence was available."


def _claim_text(profile: str, passed: bool) -> str:
    if profile == "broad_superiority":
        return "Broad HumanISP superiority gate passed." if passed else "Broad HumanISP superiority gate failed."
    if profile == "fp_reducer":
        return "Recall-budgeted FP-reduction gate passed." if passed else "Recall-budgeted FP-reduction gate failed."
    return "Custom claim gate passed." if passed else "Custom claim gate failed."


def _split_named_path(value: str | Path) -> Tuple[str, Path]:
    text = str(value)
    if "=" in text:
        name, raw_path = text.split("=", 1)
        return name.strip(), Path(raw_path).expanduser()
    return "", Path(text).expanduser()


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _sibling_html(summary_path: Path) -> str | None:
    html_path = summary_path.with_name("index.html")
    return str(html_path) if html_path.exists() else None


def _default_name(summary_path: Path) -> str:
    return summary_path.parent.name


def _max_by(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any] | None:
    values = [row for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values, key=lambda row: float(row.get(key, 0.0)))


def _max_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    values = [row for row in rows if _metric(row, metric) is not None]
    if not values:
        return None
    return max(values, key=lambda row: float(_metric(row, metric) or 0.0))


def _min_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    values = [row for row in rows if _metric(row, metric) is not None]
    if not values:
        return None
    return min(values, key=lambda row: float(_metric(row, metric) or 0.0))


def _metric(row: Mapping[str, Any] | None, metric: str) -> float | None:
    if not row:
        return None
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping) or metrics.get(metric) is None:
        return None
    return float(metrics.get(metric))


def _training_run_brief(row: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not row:
        return None
    return {
        "name": row.get("name"),
        "channel_mode": row.get("channel_mode"),
        "sample_count": row.get("sample_count"),
        "epochs": row.get("epochs"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "throughput": row.get("throughput"),
    }


def _dense_eval_brief(row: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not row:
        return None
    return {
        "name": row.get("name"),
        "channel_mode": row.get("channel_mode"),
        "sample_count": row.get("sample_count"),
        "precision@0.50_mean": _metric(row, "precision@0.50_mean"),
        "recall@0.50_mean": _metric(row, "recall@0.50_mean"),
        "fp@0.50_mean": _metric(row, "fp@0.50_mean"),
        "det_count_mean": _metric(row, "det_count_mean"),
    }


def _render_html(dashboard: Mapping[str, Any], destination: Path) -> str:
    decision_rows = "".join(_decision_row(item) for item in dashboard.get("decisions", ()))
    claim_rows = "".join(_claim_row(item, destination) for item in dashboard.get("claims", ()))
    training = dashboard.get("training")
    training_html = _training_html(training, destination) if isinstance(training, Mapping) else "<p>No RGB+Aux training rollup was provided.</p>"
    comparison_rows = "".join(_comparison_row(item, destination) for item in dashboard.get("comparison_rollups", ()))
    comparison_body = comparison_rows if comparison_rows else '<tr><td colspan="4">No comparison rollups were provided.</td></tr>'
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Claim Readiness Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .supported {{ color: #047857; font-weight: 700; }}
    .not_supported {{ color: #b91c1c; font-weight: 700; }}
    .needs_gate, .needs_eval {{ color: #a16207; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Claim Readiness Dashboard</h1>
  <div class=\"note\">{html_lib.escape(str(dashboard.get('interpretation', '')))}</div>
  <h2>Decision Summary</h2>
  <table>
    <thead><tr><th>Status</th><th>Claim</th></tr></thead>
    <tbody>{decision_rows}</tbody>
  </table>
  <h2>Claim Gates</h2>
  <table>
    <thead><tr><th>Name</th><th>Report</th><th>Claim</th><th>Profile</th><th>Verdict</th><th>Baseline</th><th>Target</th><th>Samples</th><th>P50 d/CI</th><th>R50 d/CI</th><th>Small R50 d/CI</th><th>FP d/CI</th><th>Failed</th></tr></thead>
    <tbody>{claim_rows}</tbody>
  </table>
  <h2>RGB+Aux DNN Training</h2>
  {training_html}
  <h2>Supporting Rollups</h2>
  <table>
    <thead><tr><th>Name</th><th>Report</th><th>Runs</th><th>Baseline</th></tr></thead>
    <tbody>{comparison_body}</tbody>
  </table>
  <p>Raw JSON: <code>claim_dashboard_summary.json</code></p>
</body>
</html>
"""


def _decision_row(item: Mapping[str, Any]) -> str:
    status = str(item.get("status", ""))
    return (
        "<tr>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td>{html_lib.escape(str(item.get('claim', '')))}</td>"
        "</tr>"
    )


def _claim_row(item: Mapping[str, Any], destination: Path) -> str:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), Mapping) else {}
    failed = ", ".join(str(value) for value in item.get("failed_metrics", ()))
    verdict_class = "supported" if bool(item.get("pass")) else "not_supported"
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(item.get('name', '')))}</td>"
        f"<td>{_report_link(item, destination)}</td>"
        f"<td>{html_lib.escape(str(item.get('claim', '')))}</td>"
        f"<td>{html_lib.escape(str(item.get('profile', '')))}</td>"
        f"<td class=\"{verdict_class}\">{html_lib.escape(str(item.get('verdict', '')))}</td>"
        f"<td><code>{html_lib.escape(str(item.get('baseline_input', '')))}</code></td>"
        f"<td><code>{html_lib.escape(str(item.get('target_input', '')))}</code></td>"
        f"<td>{int(item.get('sample_count', 0))}</td>"
        f"<td>{_metric_cell(metrics.get('precision@0.50_mean'))}</td>"
        f"<td>{_metric_cell(metrics.get('recall@0.50_mean'))}</td>"
        f"<td>{_metric_cell(metrics.get('small_recall@0.50_mean'))}</td>"
        f"<td>{_metric_cell(metrics.get('fp@0.50_mean'))}</td>"
        f"<td>{html_lib.escape(failed or 'none')}</td>"
        "</tr>"
    )


def _training_html(training: Mapping[str, Any], destination: Path) -> str:
    best_rate = training.get("best_train_rate") if isinstance(training.get("best_train_rate"), Mapping) else {}
    best_recall = training.get("best_recall_eval") if isinstance(training.get("best_recall_eval"), Mapping) else {}
    lowest_fp = training.get("lowest_fp_eval") if isinstance(training.get("lowest_fp_eval"), Mapping) else {}
    return (
        f"<p>Status: <code>{html_lib.escape(str(training.get('status', '')))}</code>. {html_lib.escape(str(training.get('interpretation', '')))}</p>"
        "<table>"
        "<thead><tr><th>Report</th><th>Runs</th><th>Best Train Rate</th><th>Best R50 Eval</th><th>Lowest FP Eval</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{_report_link(training, destination)}</td>"
        f"<td>{int(training.get('run_count', 0))}</td>"
        f"<td>{html_lib.escape(str(best_rate.get('name', '')))}<br>{_fmt(best_rate.get('throughput'))} sample-epochs/s</td>"
        f"<td>{html_lib.escape(str(best_recall.get('name', '')))}<br>R50 {_fmt(best_recall.get('recall@0.50_mean'))}, P50 {_fmt(best_recall.get('precision@0.50_mean'))}, FP {_fmt(best_recall.get('fp@0.50_mean'))}</td>"
        f"<td>{html_lib.escape(str(lowest_fp.get('name', '')))}<br>FP {_fmt(lowest_fp.get('fp@0.50_mean'))}, R50 {_fmt(lowest_fp.get('recall@0.50_mean'))}</td>"
        "</tr></tbody></table>"
    )


def _comparison_row(item: Mapping[str, Any], destination: Path) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(item.get('name', '')))}</td>"
        f"<td>{_report_link(item, destination)}</td>"
        f"<td>{int(item.get('run_count', 0))}</td>"
        f"<td><code>{html_lib.escape(str(item.get('baseline_input', '')))}</code></td>"
        "</tr>"
    )


def _metric_cell(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    ci_low = item.get("ci_low")
    ci_high = item.get("ci_high")
    ci_text = "" if ci_low is None or ci_high is None else f"<br>CI [{_fmt(ci_low, signed=True)}, {_fmt(ci_high, signed=True)}]"
    status = "supported" if bool(item.get("pass")) else "not_supported"
    return f"<span class=\"{status}\">{_fmt(item.get('delta'), signed=True)}</span>{ci_text}"


def _report_link(item: Mapping[str, Any], destination: Path) -> str:
    html_path = item.get("html_path")
    if not html_path:
        return ""
    relative = os.path.relpath(str(html_path), start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
