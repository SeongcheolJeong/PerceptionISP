"""Audit YOLO RGB/RGB+Aux evaluation evidence and flag channel provenance issues."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from perception_isp.core.types import json_ready

EDGE_EVIDENCE_CHANNELS = {"aux_edge_strength", "aux_edge_evidence", "aux_psf_edge_likelihood"}
TRAIN_FAIRNESS_KEYS = (
    "epochs",
    "imgsz",
    "optimizer",
    "lr0",
    "lrf",
    "weight_decay",
    "warmup_epochs",
    "mosaic",
    "mixup",
    "copy_paste",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "fliplr",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit YOLO PerceptionISP evidence summaries.")
    parser.add_argument("--input-root", default="outputs/yolo_hard_object_eval")
    parser.add_argument("--output-dir", default="reports/perception_yolo_edge6_evidence_audit_current_v1")
    args = parser.parse_args(argv)

    summary = build_yolo_evidence_audit(Path(args.input_root))
    html_path = write_yolo_evidence_audit(summary, Path(args.output_dir))
    print(json.dumps({"report": str(html_path), "summary_json": str(html_path.with_name("yolo_evidence_audit_summary.json"))}, indent=2))
    return 0


def build_yolo_evidence_audit(input_root: Path) -> dict[str, Any]:
    root = Path(input_root)
    summaries = sorted(root.rglob("summary.json"))
    evals = []
    run_rows = []
    comparison_rows = []
    invalid_rows = []
    valid_edge6_rows = []
    for summary_path in summaries:
        payload = _read_json(summary_path)
        if not isinstance(payload.get("runs"), Mapping):
            continue
        eval_row = _eval_row(summary_path, payload)
        evals.append(eval_row)
        runs = []
        for run_name, run in payload.get("runs", {}).items():
            if not isinstance(run, Mapping):
                continue
            row = _run_row(summary_path, str(run_name), run)
            runs.append(row)
            run_rows.append(row)
            if row["channel_status"] == "edge6_valid":
                valid_edge6_rows.append(row)
            if row["channel_status"] in {"edge6_name_without_edge_evidence", "aux_without_edge_evidence"}:
                invalid_rows.append(row)
        comparison_rows.extend(_comparison_rows(eval_row, runs))

    valid_comparisons = [
        row
        for row in comparison_rows
        if row.get("target_channel_status") == "edge6_valid" and row.get("baseline_channel_status") == "rgb_only"
    ]
    fair_valid_comparisons = [row for row in valid_comparisons if row.get("training_config_match")]
    invalid_named_edge6 = [row for row in run_rows if row.get("channel_status") == "edge6_name_without_edge_evidence"]
    all_filter_wins = sum(1 for row in fair_valid_comparisons if str(row.get("filter")) == "all" and float(row.get("delta_mAP50", 0.0)) > 0.0)
    hard_filter_wins = sum(
        1
        for row in fair_valid_comparisons
        if str(row.get("filter")) in {"small", "thin", "small_or_thin"} and float(row.get("delta_mAP50", 0.0)) > 0.0
    )
    summary = {
        "name": "YOLO edge6 evidence audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root),
        "status": "pass" if valid_edge6_rows else "blocked",
        "eval_summary_count": len(evals),
        "run_count": len(run_rows),
        "valid_edge6_run_count": len(valid_edge6_rows),
        "invalid_or_ambiguous_aux_run_count": len(invalid_rows),
        "edge6_name_without_edge_evidence_count": len(invalid_named_edge6),
        "valid_edge6_rgb_comparison_count": len(valid_comparisons),
        "fair_valid_edge6_rgb_comparison_count": len(fair_valid_comparisons),
        "training_mismatched_valid_edge6_rgb_comparison_count": len(valid_comparisons) - len(fair_valid_comparisons),
        "valid_edge6_all_filter_mAP50_win_count": all_filter_wins,
        "valid_edge6_hard_filter_mAP50_win_count": hard_filter_wins,
        "evals": evals,
        "runs": run_rows,
        "comparisons": comparison_rows,
        "claim_boundary": (
            "This audit checks whether YOLO evidence actually used the intended edge-evidence aux channels. "
            "It does not prove detector superiority by itself; it prevents mislabeled or ambiguous runs from being used as strong PerceptionISP claims."
        ),
        "interpretation": _interpretation(valid_comparisons, fair_valid_comparisons, invalid_named_edge6),
        "next_actions": _next_actions(valid_comparisons, fair_valid_comparisons, invalid_named_edge6),
    }
    summary["checks"] = _checks(summary)
    return json_ready(summary)


def write_yolo_evidence_audit(summary: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "yolo_evidence_audit_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    html_path = output_dir / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return html_path


def _eval_row(summary_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    config = payload.get("run_config") if isinstance(payload.get("run_config"), Mapping) else {}
    return {
        "summary_path": str(summary_path),
        "report_dir": str(summary_path.parent),
        "name": summary_path.parent.name,
        "imgsz": config.get("imgsz"),
        "batch": config.get("batch"),
        "device": config.get("device"),
        "split": config.get("split"),
        "run_count": len(payload.get("runs", {}) if isinstance(payload.get("runs"), Mapping) else {}),
    }


def _run_row(summary_path: Path, run_name: str, run: Mapping[str, Any]) -> dict[str, Any]:
    data_path = _resolve_path(summary_path.parent, run.get("data"))
    export_summary = _read_json(data_path.parent / "summary.json") if data_path else {}
    train_args = _training_args_for_model(run.get("model"))
    selected_channels = tuple(str(value) for value in export_summary.get("selected_channels", []) or [])
    channel_count = _to_int(export_summary.get("channels"), default=_to_int(_yaml_channels(data_path), default=0))
    channel_status = _channel_status(name=run_name, channel_count=channel_count, selected_channels=selected_channels, data_path=data_path)
    filters = run.get("filters") if isinstance(run.get("filters"), Mapping) else {}
    return {
        "eval_summary": str(summary_path),
        "eval_name": summary_path.parent.name,
        "name": run_name,
        "model": str(run.get("model", "")),
        "data": str(run.get("data", "")),
        "data_path": str(data_path) if data_path else "",
        "export_summary": str(data_path.parent / "summary.json") if data_path else "",
        "channel_count": channel_count,
        "selected_channels": list(selected_channels),
        "channel_status": channel_status,
        "train_args": train_args,
        "train_signature": _train_signature(train_args),
        "all_mAP50": _metric(filters, "all", "mAP50"),
        "all_mAP50_95": _metric(filters, "all", "mAP50_95"),
        "small_mAP50": _metric(filters, "small", "mAP50"),
        "thin_mAP50": _metric(filters, "thin", "mAP50"),
        "small_or_thin_mAP50": _metric(filters, "small_or_thin", "mAP50"),
        "all_precision_conf025": _fixed_metric(filters, "all", "conf_0.25_iou_0.50", "precision"),
        "all_recall_conf025": _fixed_metric(filters, "all", "conf_0.25_iou_0.50", "recall"),
        "small_or_thin_precision_conf025": _fixed_metric(filters, "small_or_thin", "conf_0.25_iou_0.50", "precision"),
        "small_or_thin_recall_conf025": _fixed_metric(filters, "small_or_thin", "conf_0.25_iou_0.50", "recall"),
    }


def _comparison_rows(eval_row: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = [row for row in runs if row.get("channel_status") == "rgb_only"]
    targets = [row for row in runs if row.get("channel_status") != "rgb_only"]
    if not baselines or not targets:
        return []
    baseline = baselines[0]
    rows = []
    for target in targets:
        for filter_name in ("all", "small", "thin", "small_or_thin"):
            rows.append(
                {
                    "eval_name": eval_row.get("name"),
                    "eval_summary": eval_row.get("summary_path"),
                    "baseline": baseline.get("name"),
                    "target": target.get("name"),
                    "baseline_channel_status": baseline.get("channel_status"),
                    "target_channel_status": target.get("channel_status"),
                    "target_selected_channels": target.get("selected_channels", []),
                    "training_config_match": _train_signature(target.get("train_args")) == _train_signature(baseline.get("train_args")),
                    "training_mismatch": _training_mismatch(baseline.get("train_args"), target.get("train_args")),
                    "filter": filter_name,
                    "baseline_mAP50": baseline.get(f"{filter_name}_mAP50"),
                    "target_mAP50": target.get(f"{filter_name}_mAP50"),
                    "delta_mAP50": _delta(target.get(f"{filter_name}_mAP50"), baseline.get(f"{filter_name}_mAP50")),
                }
            )
        rows.append(
            {
                "eval_name": eval_row.get("name"),
                "eval_summary": eval_row.get("summary_path"),
                "baseline": baseline.get("name"),
                "target": target.get("name"),
                "baseline_channel_status": baseline.get("channel_status"),
                "target_channel_status": target.get("channel_status"),
                "target_selected_channels": target.get("selected_channels", []),
                "training_config_match": _train_signature(target.get("train_args")) == _train_signature(baseline.get("train_args")),
                "training_mismatch": _training_mismatch(baseline.get("train_args"), target.get("train_args")),
                "filter": "all_conf025",
                "baseline_precision": baseline.get("all_precision_conf025"),
                "target_precision": target.get("all_precision_conf025"),
                "delta_precision": _delta(target.get("all_precision_conf025"), baseline.get("all_precision_conf025")),
                "baseline_recall": baseline.get("all_recall_conf025"),
                "target_recall": target.get("all_recall_conf025"),
                "delta_recall": _delta(target.get("all_recall_conf025"), baseline.get("all_recall_conf025")),
            }
        )
    return rows


def _channel_status(*, name: str, channel_count: int, selected_channels: Sequence[str], data_path: Path | None) -> str:
    lowered = f"{name} {data_path or ''}".lower()
    selected = set(selected_channels)
    if channel_count == 3 or "rgb_only" in lowered:
        return "rgb_only"
    if EDGE_EVIDENCE_CHANNELS.issubset(selected):
        return "edge6_valid"
    if "edge6" in lowered:
        return "edge6_name_without_edge_evidence"
    if channel_count > 3:
        return "aux_without_edge_evidence"
    return "unknown"


def _interpretation(
    valid_comparisons: Sequence[Mapping[str, Any]],
    fair_valid_comparisons: Sequence[Mapping[str, Any]],
    invalid_named_edge6: Sequence[Mapping[str, Any]],
) -> str:
    valid_all = [row for row in fair_valid_comparisons if row.get("filter") == "all"]
    best = sorted(valid_all, key=lambda row: float(row.get("delta_mAP50") or -999.0), reverse=True)[:1]
    parts = []
    if best:
        row = best[0]
        parts.append(
            f"Best valid edge6-vs-RGB all-filter mAP50 delta is {_fmt(row.get('delta_mAP50'), signed=True)} "
            f"in {row.get('eval_name')} ({row.get('target')} vs {row.get('baseline')})."
        )
    if invalid_named_edge6:
        parts.append(
            f"{len(invalid_named_edge6)} run(s) include 'edge6' in their name/path but do not contain aux_edge_evidence and must not be cited as edge-evidence results."
        )
    mismatched = len(valid_comparisons) - len(fair_valid_comparisons)
    if mismatched > 0:
        parts.append(
            f"{mismatched} valid edge6-vs-RGB comparison row(s) have training recipe mismatches, so their deltas are diagnostic rather than fair performance claims."
        )
    if not parts:
        parts.append("No fair valid edge6-vs-RGB comparison was found.")
    return " ".join(parts)


def _next_actions(
    valid_comparisons: Sequence[Mapping[str, Any]],
    fair_valid_comparisons: Sequence[Mapping[str, Any]],
    invalid_named_edge6: Sequence[Mapping[str, Any]],
) -> list[str]:
    actions = []
    if invalid_named_edge6:
        actions.append("Regenerate or relabel w640 edge6 datasets: current w640 edge6-named rows use stable aux channels, not aux_edge_evidence.")
    if not valid_comparisons:
        actions.append("Run a matched RGB-only vs true edge6 training/eval pair before using YOLO evidence in claims.")
    elif not fair_valid_comparisons:
        actions.append("Rerun matched RGB-only vs true edge6 training with the same optimizer/LR/augment schedule before claiming YOLO detector gain.")
    else:
        actions.append("Use only rows with channel_status=edge6_valid for edge-evidence claims.")
        actions.append("Repeat the best valid edge6 configuration across at least three seeds before claiming detector robustness.")
    actions.append("Prioritize hard slices (small/thin/low-MTF) because whole-val mAP can hide edge-evidence benefits.")
    return actions


def _checks(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "valid_edge6_runs_present",
            "status": "pass" if int(summary.get("valid_edge6_run_count", 0)) > 0 else "fail",
            "evidence": f"valid_edge6_run_count={summary.get('valid_edge6_run_count', 0)}",
        },
        {
            "id": "mislabeled_edge6_runs_flagged",
            "status": "pass",
            "evidence": f"edge6_name_without_edge_evidence_count={summary.get('edge6_name_without_edge_evidence_count', 0)}",
        },
        {
            "id": "valid_edge6_rgb_comparisons_present",
            "status": "pass" if int(summary.get("valid_edge6_rgb_comparison_count", 0)) > 0 else "fail",
            "evidence": f"valid_edge6_rgb_comparison_count={summary.get('valid_edge6_rgb_comparison_count', 0)}",
        },
        {
            "id": "fair_valid_edge6_rgb_comparisons_present",
            "status": "pass" if int(summary.get("fair_valid_edge6_rgb_comparison_count", 0)) > 0 else "fail",
            "evidence": f"fair_valid_edge6_rgb_comparison_count={summary.get('fair_valid_edge6_rgb_comparison_count', 0)}",
        },
    ]


def _render_html(summary: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(summary.get("status", "unknown")))
    valid_runs = [row for row in _list(summary.get("runs")) if row.get("channel_status") == "edge6_valid"]
    invalid_runs = [
        row
        for row in _list(summary.get("runs"))
        if row.get("channel_status") in {"edge6_name_without_edge_evidence", "aux_without_edge_evidence"}
    ]
    comparisons = _list(summary.get("comparisons"))
    valid_comparisons = [row for row in comparisons if row.get("target_channel_status") == "edge6_valid"]
    fair_valid_comparisons = [row for row in valid_comparisons if row.get("training_config_match")]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Edge6 Evidence Audit</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f6f8fb; }}
    header {{ background: #fff; border-bottom: 1px solid #d8e0e8; padding: 28px 32px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 14px; font-size: 21px; }}
    p {{ line-height: 1.55; }}
    .wrap {{ padding: 18px 32px 40px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .card {{ background: #fff; border: 1px solid #d8e0e8; border-radius: 8px; padding: 13px; }}
    .label {{ color: #5c6b7a; font-size: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 5px; }}
    .status {{ display: inline-block; border: 1px solid #d8e0e8; border-radius: 999px; padding: 2px 8px; background: #eef2f7; }}
    .pass, .edge6_valid, .rgb_only {{ color: #14804a; background: #eefaf2; border-color: #b7e3c7; }}
    .blocked, .edge6_name_without_edge_evidence, .aux_without_edge_evidence, .false {{ color: #9a6700; background: #fff8df; border-color: #ead18a; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; border-bottom: 1px solid #d8e0e8; }}
    .tab-button {{ border: 1px solid #d8e0e8; border-bottom: 0; background: #eef2f7; padding: 10px 13px; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 650; }}
    .tab-button.active {{ background: #fff; color: #1f6feb; }}
    .tab-panel {{ display: none; background: #fff; border: 1px solid #d8e0e8; border-top: 0; padding: 18px; border-radius: 0 0 8px 8px; overflow-x: auto; }}
    .tab-panel.active {{ display: block; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border: 1px solid #d8e0e8; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f9; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 1px 5px; }}
    .note {{ border-left: 4px solid #1f6feb; background: #f2f7ff; padding: 12px 14px; margin: 14px 0; }}
    @media (max-width: 1000px) {{ .cards {{ grid-template-columns: 1fr; }} header, .wrap {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>YOLO Edge6 Evidence Audit</h1>
  <p>This report separates true <code>aux_edge_evidence</code> YOLO evidence from mislabeled or ambiguous RGB+Aux runs.</p>
  <p>Status: <span class="status {status}">{status}</span>. {html_lib.escape(str(summary.get("interpretation", "")))}</p>
</header>
<div class="wrap">
  <section class="cards">
    {_metric_card("Eval reports", summary.get("eval_summary_count"))}
    {_metric_card("Runs", summary.get("run_count"))}
    {_metric_card("Valid edge6 runs", summary.get("valid_edge6_run_count"))}
    {_metric_card("Mislabeled edge6", summary.get("edge6_name_without_edge_evidence_count"))}
    {_metric_card("Fair comparisons", summary.get("fair_valid_edge6_rgb_comparison_count"))}
  </section>
  <nav class="tabs">
    {_tab_button("overview", "Overview", True)}
    {_tab_button("valid", "Valid Edge6 Runs", False)}
    {_tab_button("invalid", "Flagged Runs", False)}
    {_tab_button("deltas", "Deltas", False)}
    {_tab_button("next", "Next Actions", False)}
  </nav>
  <section id="overview" class="tab-panel active">
    <h2>Overview</h2>
    <div class="note">{html_lib.escape(str(summary.get("claim_boundary", "")))}</div>
    <table>{_checks_table(_list(summary.get("checks")))}</table>
  </section>
  <section id="valid" class="tab-panel">
    <h2>Valid Edge6 Runs</h2>
    <table>{_runs_table(valid_runs)}</table>
  </section>
  <section id="invalid" class="tab-panel">
    <h2>Flagged Runs</h2>
    <p>These rows use aux inputs but should not be cited as edge-evidence runs unless the selected channels are corrected.</p>
    <table>{_runs_table(invalid_runs)}</table>
  </section>
  <section id="deltas" class="tab-panel">
    <h2>Valid Edge6 vs RGB Deltas</h2>
    <table>{_comparisons_table(fair_valid_comparisons)}</table>
    <h2>Training-mismatched valid edge6 rows</h2>
    <p>These rows use true edge evidence, but RGB and Edge6 training recipes differ. Treat them as diagnostic only.</p>
    <table>{_comparisons_table([row for row in valid_comparisons if not row.get("training_config_match")])}</table>
  </section>
  <section id="next" class="tab-panel">
    <h2>Next Actions</h2>
    <ul>{''.join(f"<li>{html_lib.escape(str(item))}</li>" for item in _list(summary.get("next_actions")))}</ul>
  </section>
</div>
<script>
  const buttons = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));
  function activate(id) {{
    buttons.forEach((button) => button.classList.toggle('active', button.dataset.tab === id));
    panels.forEach((panel) => panel.classList.toggle('active', panel.id === id));
    history.replaceState(null, '', '#' + id);
  }}
  buttons.forEach((button) => button.addEventListener('click', () => activate(button.dataset.tab)));
  const initial = location.hash ? location.hash.slice(1) : 'overview';
  if (document.getElementById(initial)) activate(initial);
</script>
</body>
</html>
"""


def _metric_card(label: str, value: Any) -> str:
    return f"<article class=\"card\"><div class=\"label\">{html_lib.escape(label)}</div><div class=\"value\">{html_lib.escape(str(value))}</div></article>"


def _tab_button(tab_id: str, label: str, active: bool) -> str:
    cls = "tab-button active" if active else "tab-button"
    return f"<button class=\"{cls}\" type=\"button\" data-tab=\"{html_lib.escape(tab_id)}\">{html_lib.escape(label)}</button>"


def _checks_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{html_lib.escape(str(row.get('id', '')))}</td><td><span class=\"status {html_lib.escape(str(row.get('status', '')))}\">{html_lib.escape(str(row.get('status', '')))}</span></td><td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in rows
    )
    return f"<thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{body}</tbody>"


def _runs_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        status = html_lib.escape(str(row.get("channel_status", "")))
        body.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('eval_name', '')))}</td>"
            f"<td>{html_lib.escape(str(row.get('name', '')))}</td>"
            f"<td><span class=\"status {status}\">{status}</span></td>"
            f"<td>{html_lib.escape(str(row.get('channel_count', '')))}</td>"
            f"<td>{html_lib.escape(', '.join(str(v) for v in row.get('selected_channels', [])))}</td>"
            f"<td>{_fmt(row.get('all_mAP50'))}</td>"
            f"<td>{_fmt(row.get('small_or_thin_mAP50'))}</td>"
            "</tr>"
        )
    return "<thead><tr><th>Eval</th><th>Run</th><th>Channel status</th><th>Channels</th><th>Selected channels</th><th>All mAP50</th><th>Small/Thin mAP50</th></tr></thead><tbody>" + "".join(body) + "</tbody>"


def _comparisons_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in sorted(rows, key=lambda item: (str(item.get("eval_name")), str(item.get("target")), str(item.get("filter")))):
        match = bool(row.get("training_config_match"))
        match_label = "true" if match else "false"
        body.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('eval_name', '')))}</td>"
            f"<td>{html_lib.escape(str(row.get('baseline', '')))}</td>"
            f"<td>{html_lib.escape(str(row.get('target', '')))}</td>"
            f"<td><span class=\"status {match_label}\">{match_label}</span></td>"
            f"<td>{html_lib.escape(str(row.get('filter', '')))}</td>"
            f"<td>{_fmt(row.get('baseline_mAP50'))}</td>"
            f"<td>{_fmt(row.get('target_mAP50'))}</td>"
            f"<td>{_fmt(row.get('delta_mAP50'), signed=True)}</td>"
            f"<td>{html_lib.escape(_mismatch_text(row.get('training_mismatch')))}</td>"
            f"<td>{html_lib.escape(', '.join(str(v) for v in row.get('target_selected_channels', [])))}</td>"
            "</tr>"
        )
    return "<thead><tr><th>Eval</th><th>RGB baseline</th><th>Target</th><th>Train match</th><th>Filter</th><th>RGB mAP50</th><th>Target mAP50</th><th>Delta</th><th>Mismatch</th><th>Channels</th></tr></thead><tbody>" + "".join(body) + "</tbody>"


def _metric(filters: Mapping[str, Any], filter_name: str, metric_name: str) -> float | None:
    section = filters.get(filter_name)
    if not isinstance(section, Mapping):
        return None
    return _to_float(section.get(metric_name))


def _fixed_metric(filters: Mapping[str, Any], filter_name: str, key: str, metric_name: str) -> float | None:
    section = filters.get(filter_name)
    if not isinstance(section, Mapping):
        return None
    fixed = section.get("fixed_conf")
    if not isinstance(fixed, Mapping):
        return None
    item = fixed.get(key)
    if not isinstance(item, Mapping):
        return None
    return _to_float(item.get(metric_name))


def _training_args_for_model(model_value: Any) -> dict[str, Any]:
    if not model_value:
        return {}
    model_path = Path(str(model_value))
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    run_dir = model_path.parent.parent if model_path.parent.name == "weights" else model_path.parent
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return {}
    return _parse_simple_yaml(args_path)


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = _coerce_scalar(value.strip())
    return result


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except Exception:
        return value.strip("'\"")


def _train_signature(value: Any) -> tuple[tuple[str, str], ...]:
    args = value if isinstance(value, Mapping) else {}
    return tuple((key, _canonical_train_value(args.get(key))) for key in TRAIN_FAIRNESS_KEYS)


def _training_mismatch(baseline: Any, target: Any) -> dict[str, Any]:
    base = baseline if isinstance(baseline, Mapping) else {}
    other = target if isinstance(target, Mapping) else {}
    mismatch = {}
    for key in TRAIN_FAIRNESS_KEYS:
        left = _canonical_train_value(base.get(key))
        right = _canonical_train_value(other.get(key))
        if left != right:
            mismatch[key] = {"baseline": base.get(key), "target": other.get(key)}
    return mismatch


def _canonical_train_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _mismatch_text(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return ""
    parts = []
    for key, pair in value.items():
        if isinstance(pair, Mapping):
            parts.append(f"{key}: {pair.get('baseline')} -> {pair.get('target')}")
    return "; ".join(parts)


def _delta(target: Any, baseline: Any) -> float | None:
    t = _to_float(target)
    b = _to_float(baseline)
    if t is None or b is None:
        return None
    return float(t - b)


def _resolve_path(base: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


def _yaml_channels(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("channels:"):
            return _to_int(line.split(":", 1)[1].strip(), default=0)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Any, *, signed: bool = False) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.4f}"


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
