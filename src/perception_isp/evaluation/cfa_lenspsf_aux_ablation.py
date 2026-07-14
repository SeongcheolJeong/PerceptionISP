"""Compare CFA/LensPSF score-label and score-label-aux calibrated sweeps."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "cfa_lenspsf_aux_ablation_summary.json"
SWEEP_SUMMARY = "cfa_lenspsf_detector_sweep_summary.json"
METRICS = (
    "precision@0.50_mean",
    "recall@0.50_mean",
    "recall@0.75_mean",
    "small_recall@0.50_mean",
    "fp@0.50_mean",
    "det_count_mean",
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Compare no-aux and aux calibrated CFA/LensPSF detector sweeps.")
    parser.add_argument("no_aux_sweep", help="score/label-only CFA/LensPSF detector sweep dir or summary JSON.")
    parser.add_argument("aux_sweep", help="score/label/aux CFA/LensPSF detector sweep dir or summary JSON.")
    parser.add_argument("--no-aux-input", default=None, help="Input name in the no-aux sweep. Defaults to auto-detect.")
    parser.add_argument("--aux-input", default=None, help="Input name in the aux sweep. Defaults to auto-detect.")
    parser.add_argument("--output-dir", default="reports/perception_cfa_lenspsf_aux_ablation")
    args = parser.parse_args(argv)

    summary = build_aux_ablation(
        _load_sweep(args.no_aux_sweep),
        _load_sweep(args.aux_sweep),
        no_aux_source=_summary_path(args.no_aux_sweep),
        aux_source=_summary_path(args.aux_sweep),
        no_aux_input=args.no_aux_input,
        aux_input=args.aux_input,
    )
    html_path = write_aux_ablation(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "condition_count": summary["condition_count"],
                    "aux_fp_win_count": summary["aggregate"]["aux_fp_win_count"],
                    "aux_recall_win_count": summary["aggregate"]["aux_recall_win_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_aux_ablation(
    no_aux_sweep: Mapping[str, Any],
    aux_sweep: Mapping[str, Any],
    *,
    no_aux_source: str | Path | None = None,
    aux_source: str | Path | None = None,
    no_aux_input: str | None = None,
    aux_input: str | None = None,
) -> Dict[str, Any]:
    no_aux_name = no_aux_input or _detect_input(no_aux_sweep, prefer_aux=False)
    aux_name = aux_input or _detect_input(aux_sweep, prefer_aux=True)
    no_aux_by_run = _runs_by_id(no_aux_sweep)
    aux_by_run = _runs_by_id(aux_sweep)
    common_ids = tuple(run_id for run_id in no_aux_by_run if run_id in aux_by_run)
    rows = [
        _compare_run(no_aux_by_run[run_id], aux_by_run[run_id], no_aux_input=no_aux_name, aux_input=aux_name)
        for run_id in common_ids
    ]
    rows = [row for row in rows if row]
    aggregate = _aggregate(rows)
    cfa_groups = _group(rows, "cfa_pattern")
    psf_groups = _group(rows, "psf_sigma")
    expected = max(int(no_aux_sweep.get("expected_run_count", 0)), int(aux_sweep.get("expected_run_count", 0)), len(no_aux_by_run), len(aux_by_run))
    checks = [
        {
            "id": "matched_conditions_available",
            "status": "pass" if len(rows) == expected and expected > 0 else "fail",
            "evidence": f"matched={len(rows)} expected={expected}",
        },
        {
            "id": "aux_recall_tradeoff_measured",
            "status": "pass" if aggregate["aux_recall_win_count"] > 0 else "warning",
            "evidence": f"aux_recall_wins={aggregate['aux_recall_win_count']}/{len(rows)}",
        },
        {
            "id": "aux_fp_incremental_gain_majority",
            "status": "pass" if aggregate["aux_fp_win_count"] > len(rows) / 2.0 else "warning",
            "evidence": f"aux_fp_wins={aggregate['aux_fp_win_count']}/{len(rows)}",
        },
    ]
    claim_status = (
        "aux_incremental_fp_supported"
        if aggregate["aux_fp_win_count"] > len(rows) / 2.0 and aggregate["aux_recall_loss_count"] == 0
        else "aux_recall_fp_tradeoff"
    )
    return {
        "name": "CFA/LensPSF score-label vs score-label-aux ablation",
        "no_aux_source": "" if no_aux_source is None else str(no_aux_source),
        "aux_source": "" if aux_source is None else str(aux_source),
        "no_aux_input": no_aux_name,
        "aux_input": aux_name,
        "condition_count": int(len(rows)),
        "expected_condition_count": int(expected),
        "rows": rows,
        "aggregate": aggregate,
        "cfa_groups": cfa_groups,
        "psf_groups": psf_groups,
        "checks": checks,
        "status": "pass" if all(row["status"] in {"pass", "warning"} for row in checks) and rows else "fail",
        "claim_status": claim_status,
        "interpretation": (
            "This audit compares calibrated score-label proposals with calibrated score-label-aux proposals on the same "
            "native CFA/LensPSF conditions. It tests whether aux features add incremental condition-level value beyond "
            "score and class label calibration."
        ),
        "claim_boundary": (
            "Use this as an incremental calibration ablation. It is not a trained RGB+Aux DNN result and does not prove "
            "broad HumanISP superiority."
        ),
    }


def write_aux_ablation(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / SWEEP_SUMMARY
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _load_sweep(path: str | Path) -> Dict[str, Any]:
    return json.loads(_summary_path(path).read_text())


def _detect_input(sweep: Mapping[str, Any], *, prefer_aux: bool) -> str:
    runs = [row for row in sweep.get("runs", ()) if isinstance(row, Mapping)]
    if not runs:
        raise ValueError("sweep has no runs")
    names = {
        str(name)
        for run in runs
        for name in (run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}).keys()
    }
    candidates = [name for name in names if "calibrated" in name and ("score_label_aux" in name if prefer_aux else "score_label_aux" not in name and "score_label" in name)]
    if not candidates:
        candidates = [name for name in names if "calibrated" in name]
    if not candidates:
        raise ValueError("could not auto-detect calibrated input")
    return sorted(candidates)[0]


def _runs_by_id(sweep: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row.get("run_id", "")): row for row in sweep.get("runs", ()) if isinstance(row, Mapping) and row.get("run_id")}


def _compare_run(
    no_aux_run: Mapping[str, Any],
    aux_run: Mapping[str, Any],
    *,
    no_aux_input: str,
    aux_input: str,
) -> Dict[str, Any]:
    no_aux_metrics = _input_metrics(no_aux_run, no_aux_input)
    aux_metrics = _input_metrics(aux_run, aux_input)
    no_aux_delta = _input_delta(no_aux_run, no_aux_input)
    aux_delta = _input_delta(aux_run, aux_input)
    metric_delta = {metric: _float(aux_metrics.get(metric)) - _float(no_aux_metrics.get(metric)) for metric in METRICS}
    human_delta_delta = {metric: _float(aux_delta.get(metric)) - _float(no_aux_delta.get(metric)) for metric in METRICS}
    return {
        "run_id": str(no_aux_run.get("run_id", "")),
        "cfa_pattern": str(no_aux_run.get("cfa_pattern", aux_run.get("cfa_pattern", ""))),
        "psf_sigma": _float(no_aux_run.get("psf_sigma", aux_run.get("psf_sigma", 0.0))),
        "sample_count": min(int(no_aux_run.get("sample_count", 0)), int(aux_run.get("sample_count", 0))),
        "no_aux": {metric: _float(no_aux_metrics.get(metric)) for metric in METRICS},
        "aux": {metric: _float(aux_metrics.get(metric)) for metric in METRICS},
        "aux_minus_no_aux": metric_delta,
        "aux_minus_no_aux_delta_vs_human": human_delta_delta,
        "aux_precision_win": bool(metric_delta["precision@0.50_mean"] > 0.0),
        "aux_recall_win": bool(metric_delta["recall@0.50_mean"] > 0.0),
        "aux_small_recall_win": bool(metric_delta["small_recall@0.50_mean"] > 0.0),
        "aux_fp_win": bool(metric_delta["fp@0.50_mean"] < 0.0),
    }


def _input_metrics(run: Mapping[str, Any], input_name: str) -> Mapping[str, Any]:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}
    row = metrics.get(input_name)
    if not isinstance(row, Mapping):
        raise ValueError(f"input metrics not found for {input_name!r} in {run.get('run_id')}")
    return row


def _input_delta(run: Mapping[str, Any], input_name: str) -> Mapping[str, Any]:
    metrics = run.get("delta_vs_human", {}) if isinstance(run.get("delta_vs_human"), Mapping) else {}
    row = metrics.get(input_name)
    return row if isinstance(row, Mapping) else {}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "condition_count": int(len(rows)),
        "sample_count": int(sum(int(row.get("sample_count", 0)) for row in rows)),
        "aux_precision_win_count": int(sum(bool(row.get("aux_precision_win")) for row in rows)),
        "aux_recall_win_count": int(sum(bool(row.get("aux_recall_win")) for row in rows)),
        "aux_recall_loss_count": int(sum(_delta(row, "recall@0.50_mean") < 0.0 for row in rows)),
        "aux_small_recall_win_count": int(sum(bool(row.get("aux_small_recall_win")) for row in rows)),
        "aux_fp_win_count": int(sum(bool(row.get("aux_fp_win")) for row in rows)),
        "mean_aux_minus_no_aux_precision@0.50": _mean_delta(rows, "precision@0.50_mean"),
        "mean_aux_minus_no_aux_recall@0.50": _mean_delta(rows, "recall@0.50_mean"),
        "mean_aux_minus_no_aux_small_recall@0.50": _mean_delta(rows, "small_recall@0.50_mean"),
        "mean_aux_minus_no_aux_fp@0.50": _mean_delta(rows, "fp@0.50_mean"),
    }


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> list[Dict[str, Any]]:
    values = sorted({row.get(key) for row in rows}, key=lambda value: str(value))
    output = []
    for value in values:
        group_rows = [row for row in rows if row.get(key) == value]
        output.append({"group": value, **_aggregate(group_rows)})
    return output


def _mean_delta(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return sum(_delta(row, metric) for row in rows) / len(rows) if rows else 0.0


def _delta(row: Mapping[str, Any], metric: str) -> float:
    deltas = row.get("aux_minus_no_aux", {}) if isinstance(row.get("aux_minus_no_aux"), Mapping) else {}
    return _float(deltas.get(metric))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _render_html(summary: Mapping[str, Any]) -> str:
    checks = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), Mapping) else {}
    cfa_rows = "".join(_group_row(row) for row in summary.get("cfa_groups", ()) if isinstance(row, Mapping))
    psf_rows = "".join(_group_row(row) for row in summary.get("psf_groups", ()) if isinstance(row, Mapping))
    rows = "".join(_condition_row(row) for row in summary.get("rows", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CFA/LensPSF Aux Ablation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d7dee8; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #edf2f7; }}
    code {{ background: #f3f6fa; padding: 2px 4px; border-radius: 4px; }}
    .pass {{ color: #176b43; font-weight: 700; }}
    .warning {{ color: #9a5b00; font-weight: 700; }}
    .fail {{ color: #9b1c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>CFA/LensPSF Score-Label Aux Ablation</h1>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; claim status: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.</p>
  <p>{html_lib.escape(str(summary.get('interpretation', '')))}</p>
  <p>{html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <h2>Aggregate</h2>
  <table>
    <tr><th>Conditions</th><td>{int(aggregate.get('condition_count', 0))}</td></tr>
    <tr><th>Samples</th><td>{int(aggregate.get('sample_count', 0))}</td></tr>
    <tr><th>Aux precision wins</th><td>{int(aggregate.get('aux_precision_win_count', 0))}</td></tr>
    <tr><th>Aux recall wins</th><td>{int(aggregate.get('aux_recall_win_count', 0))}</td></tr>
    <tr><th>Aux FP wins</th><td>{int(aggregate.get('aux_fp_win_count', 0))}</td></tr>
    <tr><th>Mean aux-noaux dP50</th><td>{_fmt(aggregate.get('mean_aux_minus_no_aux_precision@0.50'), signed=True)}</td></tr>
    <tr><th>Mean aux-noaux dR50</th><td>{_fmt(aggregate.get('mean_aux_minus_no_aux_recall@0.50'), signed=True)}</td></tr>
    <tr><th>Mean aux-noaux dFP50</th><td>{_fmt(aggregate.get('mean_aux_minus_no_aux_fp@0.50'), signed=True)}</td></tr>
  </table>
  <h2>Checks</h2>
  <table><tr><th>ID</th><th>Status</th><th>Evidence</th></tr>{checks}</table>
  <h2>CFA Groups</h2>
  <table><tr><th>CFA</th><th>Conditions</th><th>dP50</th><th>dR50</th><th>dSmallR50</th><th>dFP50</th><th>Aux FP wins</th></tr>{cfa_rows}</table>
  <h2>PSF Groups</h2>
  <table><tr><th>PSF</th><th>Conditions</th><th>dP50</th><th>dR50</th><th>dSmallR50</th><th>dFP50</th><th>Aux FP wins</th></tr>{psf_rows}</table>
  <h2>Conditions</h2>
  <table><tr><th>Run</th><th>CFA</th><th>PSF</th><th>dP50</th><th>dR50</th><th>dSmallR50</th><th>dFP50</th><th>Aux FP win</th></tr>{rows}</table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return f"<tr><td>{html_lib.escape(str(row.get('id', '')))}</td><td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td><td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"


def _group_row(row: Mapping[str, Any]) -> str:
    return (
        f"<tr><td>{html_lib.escape(str(row.get('group', '')))}</td>"
        f"<td>{int(row.get('condition_count', 0))}</td>"
        f"<td>{_fmt(row.get('mean_aux_minus_no_aux_precision@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('mean_aux_minus_no_aux_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('mean_aux_minus_no_aux_small_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(row.get('mean_aux_minus_no_aux_fp@0.50'), signed=True)}</td>"
        f"<td>{int(row.get('aux_fp_win_count', 0))}</td></tr>"
    )


def _condition_row(row: Mapping[str, Any]) -> str:
    deltas = row.get("aux_minus_no_aux", {}) if isinstance(row.get("aux_minus_no_aux"), Mapping) else {}
    return (
        f"<tr><td>{html_lib.escape(str(row.get('run_id', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{_fmt(deltas.get('precision@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('small_recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(deltas.get('fp@0.50_mean'), signed=True)}</td>"
        f"<td>{html_lib.escape(str(bool(row.get('aux_fp_win'))))}</td></tr>"
    )


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _float(value)
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
