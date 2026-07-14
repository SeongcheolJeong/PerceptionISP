"""Input-ablation gate for trained RGB+Aux dense detectors."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from perception_isp.evaluation.aux_eval_dense import evaluate_dense_manifest, parse_label_list
from perception_isp.evaluation.dense_select_test_gate import METRIC_KEYS
from perception_isp.core.types import json_ready


SUMMARY_FILENAME = "dense_input_ablation_summary.json"
DEFAULT_MODES = ("none", "zero_aux", "shuffle_aux", "zero_rgb")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run an input-ablation gate for selected RGB+Aux dense-detector checkpoints.")
    parser.add_argument("--gate-summary", required=True, help="dense_select_test_gate summary.json path or report directory.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated input ablation modes.")
    parser.add_argument("--include-labels", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--label-agnostic", action="store_true", help="Use label-agnostic metrics instead of the label-aware gate default.")
    parser.add_argument("--min-zero-aux-recall-drop", type=float, default=0.05)
    parser.add_argument(
        "--min-zero-aux-precision-drop",
        type=float,
        default=0.0,
        help="Optional precision-drop requirement for zero_aux. Values <= 0 keep precision as a diagnostic.",
    )
    parser.add_argument("--min-zero-rgb-recall-drop", type=float, default=0.05)
    parser.add_argument("--output-dir", default="reports/perception_dense_input_ablation_gate")
    args = parser.parse_args(argv)

    summary = build_dense_input_ablation_gate(
        gate_summary=args.gate_summary,
        modes=parse_modes(args.modes),
        include_labels=parse_label_list(args.include_labels),
        device=str(args.device),
        label_agnostic=bool(args.label_agnostic),
        min_zero_aux_recall_drop=float(args.min_zero_aux_recall_drop),
        min_zero_aux_precision_drop=float(args.min_zero_aux_precision_drop),
        min_zero_rgb_recall_drop=float(args.min_zero_rgb_recall_drop),
    )
    html_path = write_dense_input_ablation_gate(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "mean_by_mode": summary["mean_by_mode"],
                    "deltas_vs_none": summary["deltas_vs_none"],
                    "failed_required_checks": [
                        row["id"]
                        for row in summary["checks"]
                        if bool(row.get("required")) and row.get("status") != "pass"
                    ],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_dense_input_ablation_gate(
    *,
    gate_summary: str | Path | Mapping[str, Any],
    modes: Sequence[str] = DEFAULT_MODES,
    include_labels: Sequence[str] | None = None,
    device: str = "auto",
    label_agnostic: bool = False,
    min_zero_aux_recall_drop: float = 0.05,
    min_zero_aux_precision_drop: float = 0.0,
    min_zero_rgb_recall_drop: float = 0.05,
) -> Dict[str, Any]:
    start = time.perf_counter()
    gate = _load_gate_summary(gate_summary)
    resolved_modes = parse_modes(modes)
    if "none" not in resolved_modes:
        raise ValueError("input ablation gate requires the 'none' reference mode")
    manifest = str(gate["manifest"])
    test_indices = tuple(int(index) for index in gate.get("test_indices", ()))
    if not test_indices:
        raise ValueError("gate summary must contain non-empty test_indices")
    gate_rows = tuple(row for row in gate.get("rows", ()) if isinstance(row, Mapping))
    if not gate_rows:
        raise ValueError("gate summary must contain selected seed rows")

    rows = []
    for gate_row in gate_rows:
        seed = int(gate_row["seed"])
        checkpoint = str(gate_row["aux_checkpoint"])
        confidence = float(gate_row["selected_confidence"])
        nms_iou = gate_row.get("selected_nms_iou")
        max_detections = gate_row.get("selected_max_detections")
        for mode in resolved_modes:
            result = evaluate_dense_manifest(
                manifest_path=manifest,
                checkpoint_path=checkpoint,
                split="eval",
                confidence=confidence,
                nms_iou=nms_iou,
                max_detections=max_detections,
                device=device,
                label_agnostic=bool(label_agnostic),
                include_labels=include_labels,
                indices=test_indices,
                input_ablation=mode,
                ablation_seed=seed,
                output_dir=None,
            )
            aggregate = result.get("aggregate", {}) if isinstance(result.get("aggregate"), Mapping) else {}
            rows.append(
                {
                    "seed": seed,
                    "mode": str(mode),
                    "checkpoint": checkpoint,
                    "confidence": confidence,
                    "nms_iou": None if nms_iou is None else float(nms_iou),
                    "max_detections": None if max_detections is None else int(max_detections),
                    "sample_count": int(result.get("sample_count", 0)),
                    **_metrics_from_aggregate(aggregate),
                }
            )

    mean_by_mode = {
        mode: _mean_metrics(row for row in rows if row["mode"] == mode)
        for mode in resolved_modes
    }
    reference = mean_by_mode["none"]
    deltas_vs_none = {
        mode: {
            metric: float(mean_by_mode[mode].get(metric, 0.0)) - float(reference.get(metric, 0.0))
            for metric in METRIC_KEYS
        }
        for mode in resolved_modes
        if mode != "none"
    }
    checks = _checks(
        mean_by_mode,
        deltas_vs_none,
        min_zero_aux_recall_drop=float(min_zero_aux_recall_drop),
        min_zero_aux_precision_drop=float(min_zero_aux_precision_drop),
        min_zero_rgb_recall_drop=float(min_zero_rgb_recall_drop),
    )
    required_pass = all(row.get("status") == "pass" for row in checks if bool(row.get("required")))
    return {
        "name": "PerceptionISP dense DNN input ablation gate",
        "source_gate_summary": _gate_source(gate_summary),
        "manifest": manifest,
        "test_sample_count": int(len(test_indices)),
        "seed_count": int(len(gate_rows)),
        "modes": list(resolved_modes),
        "label_agnostic": bool(label_agnostic),
        "include_labels": None if include_labels is None else [str(label) for label in include_labels],
        "thresholds": {
            "min_zero_aux_recall_drop": float(min_zero_aux_recall_drop),
            "min_zero_aux_precision_drop": float(min_zero_aux_precision_drop),
            "min_zero_rgb_recall_drop": float(min_zero_rgb_recall_drop),
        },
        "rows": rows,
        "mean_by_mode": mean_by_mode,
        "deltas_vs_none": deltas_vs_none,
        "checks": checks,
        "status": "pass" if required_pass else "mixed",
        "claim_status": "aux_input_used_by_dense_dnn" if required_pass else "aux_input_ablation_mixed",
        "elapsed_seconds": float(max(time.perf_counter() - start, 0.0)),
        "interpretation": (
            "The selected RGB+Aux checkpoint and operating point from the source gate are reused unchanged. "
            "Only inference-time input tensor channels are ablated, so this tests whether the trained dense DNN "
            "depends on RGB/Aux evidence rather than only changing thresholds."
        ),
        "claim_boundary": (
            "This proves input dependence for the compact dense detector under this held-out split. "
            "It is not yet a production-detector proof or a broad adverse-condition benchmark."
        ),
    }


def write_dense_input_ablation_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def parse_modes(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        tokens = tuple(token.strip().lower().replace("-", "_") for token in value.split(",") if token.strip())
    else:
        tokens = tuple(str(token).strip().lower().replace("-", "_") for token in value if str(token).strip())
    values = tuple(dict.fromkeys(tokens))
    return values or DEFAULT_MODES


def _load_gate_summary(value: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "summary.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"gate summary must be a JSON object: {path}")
    return dict(payload)


def _gate_source(value: str | Path | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        return "in_memory"
    return str(Path(value).expanduser())


def _metrics_from_aggregate(aggregate: Mapping[str, Any]) -> Dict[str, float]:
    return {
        metric: float(aggregate.get(source_key, 0.0))
        for metric, source_key in METRIC_KEYS.items()
    }


def _mean_metrics(rows: Sequence[Mapping[str, Any]] | Any) -> Dict[str, float]:
    values = tuple(rows)
    if not values:
        return {metric: 0.0 for metric in METRIC_KEYS}
    return {
        metric: float(statistics.mean(float(row.get(metric, 0.0)) for row in values))
        for metric in METRIC_KEYS
    }


def _checks(
    mean_by_mode: Mapping[str, Mapping[str, float]],
    deltas_vs_none: Mapping[str, Mapping[str, float]],
    *,
    min_zero_aux_recall_drop: float,
    min_zero_aux_precision_drop: float,
    min_zero_rgb_recall_drop: float,
) -> list[Dict[str, Any]]:
    checks: list[Dict[str, Any]] = []
    zero_aux_delta = deltas_vs_none.get("zero_aux")
    if zero_aux_delta is None:
        checks.append(
            {
                "id": "zero_aux_mode_present",
                "required": True,
                "status": "fail",
                "description": "zero_aux mode is required to test whether Aux channels carry useful signal.",
            }
        )
    else:
        recall_drop = -float(zero_aux_delta.get("recall", 0.0))
        precision_drop = -float(zero_aux_delta.get("precision", 0.0))
        precision_required = float(min_zero_aux_precision_drop) > 0.0
        recall_pass = recall_drop >= float(min_zero_aux_recall_drop)
        precision_pass = precision_drop >= float(min_zero_aux_precision_drop)
        checks.append(
            {
                "id": "zero_aux_reduces_dense_dnn_performance",
                "required": True,
                "status": "pass"
                if recall_pass
                and (not precision_required or precision_pass)
                else "fail",
                "description": (
                    "Zeroing Aux channels should degrade held-out dense-detector recall. Precision is diagnostic by "
                    "default because a disabled evidence path can reduce detections and raise precision while losing recall."
                ),
                "criteria": [
                    {
                        "metric": "recall_drop",
                        "value": recall_drop,
                        "threshold": float(min_zero_aux_recall_drop),
                        "required": True,
                        "pass": recall_pass,
                    },
                    {
                        "metric": "precision_drop",
                        "value": precision_drop,
                        "threshold": float(min_zero_aux_precision_drop),
                        "required": precision_required,
                        "pass": precision_pass,
                    },
                ],
            }
        )

    zero_rgb_delta = deltas_vs_none.get("zero_rgb")
    if zero_rgb_delta is not None:
        recall_drop = -float(zero_rgb_delta.get("recall", 0.0))
        checks.append(
            {
                "id": "zero_rgb_is_not_sufficient_for_selected_detector",
                "required": True,
                "status": "pass" if recall_drop >= float(min_zero_rgb_recall_drop) else "fail",
                "description": "Zeroing RGB should not outperform the full RGB+Aux input at the selected operating point.",
                "criteria": [
                    {
                        "metric": "recall_drop",
                        "value": recall_drop,
                        "threshold": float(min_zero_rgb_recall_drop),
                        "pass": recall_drop >= float(min_zero_rgb_recall_drop),
                    }
                ],
            }
        )

    shuffle_aux_delta = deltas_vs_none.get("shuffle_aux")
    if shuffle_aux_delta is not None:
        recall_drop = -float(shuffle_aux_delta.get("recall", 0.0))
        metric_change = max(abs(float(shuffle_aux_delta.get(metric, 0.0))) for metric in ("precision", "recall", "fp", "det_count"))
        checks.append(
            {
                "id": "shuffle_aux_spatial_sensitivity_diagnostic",
                "required": False,
                "status": "pass" if metric_change > 1.0e-9 else "warning",
                "description": "Shuffling Aux channels across samples is a diagnostic for spatial/sample alignment sensitivity.",
                "criteria": [
                    {"metric": "recall_drop", "value": recall_drop},
                    {"metric": "max_abs_metric_change", "value": metric_change},
                ],
            }
        )
    return checks


def _render_html(summary: Mapping[str, Any]) -> str:
    mean_rows = []
    for mode, metrics in summary.get("mean_by_mode", {}).items():
        mean_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(mode))}</td>"
            f"<td>{_fmt(metrics.get('precision'))}</td>"
            f"<td>{_fmt(metrics.get('recall'))}</td>"
            f"<td>{_fmt(metrics.get('fp'), digits=3)}</td>"
            f"<td>{_fmt(metrics.get('det_count'), digits=3)}</td>"
            "</tr>"
        )
    delta_rows = []
    for mode, metrics in summary.get("deltas_vs_none", {}).items():
        delta_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(mode))}</td>"
            f"<td class=\"{_delta_class(metrics.get('precision'), positive_good=True)}\">{_fmt(metrics.get('precision'), signed=True)}</td>"
            f"<td class=\"{_delta_class(metrics.get('recall'), positive_good=True)}\">{_fmt(metrics.get('recall'), signed=True)}</td>"
            f"<td class=\"{_delta_class(metrics.get('fp'), positive_good=False)}\">{_fmt(metrics.get('fp'), signed=True, digits=3)}</td>"
            f"<td>{_fmt(metrics.get('det_count'), signed=True, digits=3)}</td>"
            "</tr>"
        )
    check_rows = []
    for check in summary.get("checks", ()):
        status = str(check.get("status", ""))
        check_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(check.get('id', '')))}</td>"
            f"<td class=\"{'pos' if status == 'pass' else 'neg'}\">{html_lib.escape(status)}</td>"
            f"<td>{html_lib.escape('required' if bool(check.get('required')) else 'diagnostic')}</td>"
            f"<td><code>{html_lib.escape(json.dumps(json_ready(check), sort_keys=True))}</code></td>"
            "</tr>"
        )
    seed_rows = []
    for row in summary.get("rows", ()):
        seed_rows.append(
            "<tr>"
            f"<td>{int(row.get('seed', 0))}</td>"
            f"<td>{html_lib.escape(str(row.get('mode', '')))}</td>"
            f"<td>{_fmt(row.get('precision'))}</td>"
            f"<td>{_fmt(row.get('recall'))}</td>"
            f"<td>{_fmt(row.get('fp'), digits=3)}</td>"
            f"<td>{_fmt(row.get('det_count'), digits=3)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP Dense Input Ablation Gate</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; color: #17202a; background: #fff; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 44px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin-top: 28px; }}
    p {{ color: #5f6b7a; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin: 14px 0; }}
    th, td {{ border-bottom: 1px solid #d8dee8; padding: 9px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f5f7fa; }}
    .pos {{ color: #0b7a4b; font-weight: 700; }}
    .neg {{ color: #b42318; font-weight: 700; }}
    .note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 14px; color: #23395d; }}
    code {{ font-size: 12px; white-space: normal; }}
  </style>
</head>
<body>
<main>
  <h1>PerceptionISP Dense Input Ablation Gate</h1>
  <p>Status: <strong>{html_lib.escape(str(summary.get('status', '')))}</strong>; claim: <code>{html_lib.escape(str(summary.get('claim_status', '')))}</code>.</p>
  <p class="note">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</p>
  <p>Test samples: {int(summary.get('test_sample_count', 0))}; seeds: {int(summary.get('seed_count', 0))}; label agnostic: {bool(summary.get('label_agnostic', False))}; elapsed: {float(summary.get('elapsed_seconds', 0.0)):.1f}s.</p>
  <h2>Mean Metrics</h2>
  <table><thead><tr><th>Mode</th><th>Precision</th><th>Recall</th><th>FP</th><th>Det/sample</th></tr></thead><tbody>{''.join(mean_rows)}</tbody></table>
  <h2>Deltas vs Full RGB+Aux</h2>
  <table><thead><tr><th>Mode</th><th>Delta Precision</th><th>Delta Recall</th><th>Delta FP</th><th>Delta Det</th></tr></thead><tbody>{''.join(delta_rows)}</tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Type</th><th>Details</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table>
  <h2>Seed Rows</h2>
  <table><thead><tr><th>Seed</th><th>Mode</th><th>Precision</th><th>Recall</th><th>FP</th><th>Det/sample</th></tr></thead><tbody>{''.join(seed_rows)}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</main>
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False, digits: int = 4) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _delta_class(value: Any, *, positive_good: bool) -> str:
    number = float(value or 0.0)
    passed = number >= -1.0e-12 if positive_good else number <= 1.0e-9
    return "pos" if passed else "neg"


if __name__ == "__main__":
    raise SystemExit(main())
