"""Roll up LIS YOLO-seg fine-tuning runs.

This is a real-dataset compatibility gate for PerceptionISP segmentation work.
LIS packaged RAW images are PNG-derived images rather than native CFA RAW, so a
positive result is useful but not a native sensor/ISP proof.
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SUMMARY_FILENAME = "lis_segmentation_training_rollup_summary.json"
CORE_FAIRNESS_KEYS = ("epochs", "batch", "imgsz", "seed", "mosaic", "erasing", "copy_paste", "workers")
METRIC_KEYS = (
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll up LIS YOLO segmentation fine-tuning runs.")
    parser.add_argument("--baseline", required=True, help="Baseline spec: name|run_dir")
    parser.add_argument("--run", action="append", required=True, help="Candidate spec: name|run_dir. Repeatable.")
    parser.add_argument("--output-dir", default="reports/perception_lis_segmentation_training_rollup_v1")
    args = parser.parse_args(argv)

    summary = build_rollup(str(args.baseline), [str(value) for value in args.run])
    html_path = write_report(summary, Path(args.output_dir))
    print(
        json.dumps(
            {
                "report": str(html_path),
                "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                "status": summary["status"],
                "best_candidate": summary.get("best_candidate"),
            },
            indent=2,
        )
    )
    return 0


def build_rollup(baseline_spec: str, run_specs: Sequence[str]) -> Dict[str, Any]:
    baseline = _load_run(*_parse_spec(baseline_spec), role="baseline")
    candidates = [_load_run(*_parse_spec(spec), role="candidate") for spec in run_specs]
    rows = []
    for candidate in candidates:
        deltas = {
            key: _optional_float(candidate["metrics"].get(key)) - _optional_float(baseline["metrics"].get(key))
            for key in METRIC_KEYS
            if _optional_float(candidate["metrics"].get(key)) is not None
            and _optional_float(baseline["metrics"].get(key)) is not None
        }
        fair = _fairness(candidate.get("args", {}), baseline.get("args", {}))
        rows.append(
            {
                **candidate,
                "deltas": deltas,
                "fairness": fair,
                "decision": _decision(deltas, fair),
            }
        )
    best = max(rows, key=lambda row: float(row.get("metrics", {}).get("metrics/mAP50(M)", -1.0)), default=None)
    checks = _checks(rows)
    return {
        "title": "LIS YOLO Segmentation Training Rollup",
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "diagnostic",
        "baseline": baseline,
        "candidates": rows,
        "best_candidate": None if best is None else best.get("name"),
        "checks": checks,
        "claim_boundary": (
            "LIS packaged RAW images are RGB PNG-derived images, not native CFA RAW. "
            "LIS segmentation labels are also human/RGB-image annotations, so the benchmark naturally favors "
            "conventional RGB task compatibility over sensor-level scene-truth recovery. "
            "Use this as a real low-light segmentation compatibility gate, not as final PerceptionISP sensor proof."
        ),
    }


def write_report(summary: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    html_path = output_dir / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return html_path


def _parse_spec(spec: str) -> tuple[str, Path]:
    parts = spec.split("|", 1)
    if len(parts) != 2:
        raise ValueError(f"run spec must be name|run_dir, got {spec!r}")
    return parts[0], Path(parts[1]).expanduser()


def _load_run(name: str, run_dir: Path, *, role: str) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    metrics = _last_results_row(run_dir / "results.csv")
    args = _load_simple_yaml(run_dir / "args.yaml")
    return {
        "name": name,
        "role": role,
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
        "metrics": {key: _optional_float(metrics.get(key)) for key in METRIC_KEYS},
        "epoch": int(float(metrics.get("epoch", 0))) if metrics.get("epoch") not in {None, ""} else None,
        "time": _optional_float(metrics.get("time")),
        "args": args,
    }


def _last_results_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return {key.strip(): value.strip() for key, value in rows[-1].items()}


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        out: Dict[str, Any] = {}
        for line in path.read_text().splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split(":", 1)
            out[key.strip()] = _parse_scalar(value.strip())
        return out


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _fairness(candidate_args: Mapping[str, Any], baseline_args: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches = {}
    for key in CORE_FAIRNESS_KEYS:
        left = _canonical(candidate_args.get(key))
        right = _canonical(baseline_args.get(key))
        if left != right:
            mismatches[key] = {"candidate": left, "baseline": right}
    amp_match = _canonical(candidate_args.get("amp")) == _canonical(baseline_args.get("amp"))
    return {
        "core_config_match": not mismatches,
        "amp_match": amp_match,
        "mismatches": mismatches,
        "note": "AMP mismatch is reported separately because it may change numerics but is not the intended RGB/Aux variable.",
    }


def _decision(deltas: Mapping[str, float], fairness: Mapping[str, Any]) -> str:
    mask50 = float(deltas.get("metrics/mAP50(M)", 0.0))
    mask5095 = float(deltas.get("metrics/mAP50-95(M)", 0.0))
    recall = float(deltas.get("metrics/recall(M)", 0.0))
    if not fairness.get("core_config_match"):
        return "diagnostic_config_mismatch"
    if mask50 > 0.0 and mask5095 > 0.0 and recall >= -0.01:
        return "supports_aux_gain"
    if mask50 > 0.0 or mask5095 > 0.0:
        return "mixed_tradeoff"
    return "no_aux_gain"


def _checks(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    fair_rows = [row for row in rows if row.get("fairness", {}).get("core_config_match")]
    supporting = [row for row in fair_rows if row.get("decision") == "supports_aux_gain"]
    return [
        {
            "id": "candidate_runs_present",
            "status": "pass" if rows else "fail",
            "description": "At least one candidate run was summarized.",
            "criteria": [{"metric": "candidate_count", "value": len(rows), "pass": bool(rows)}],
        },
        {
            "id": "fair_candidates_present",
            "status": "pass" if fair_rows else "fail",
            "description": "At least one candidate uses the same core training recipe as the RGB baseline.",
            "criteria": [{"metric": "fair_candidate_count", "value": len(fair_rows), "pass": bool(fair_rows)}],
        },
        {
            "id": "aux_gain_supported",
            "status": "pass" if supporting else "fail",
            "description": "At least one fair RGB+Aux candidate should improve mask mAP50 and mAP50-95 without material recall loss.",
            "criteria": [{"metric": "supporting_candidate_count", "value": len(supporting), "pass": bool(supporting)}],
        },
    ]


def _render_html(summary: Mapping[str, Any]) -> str:
    baseline = summary.get("baseline", {})
    rows = []
    for row in summary.get("candidates", []):
        metrics = row.get("metrics", {})
        deltas = row.get("deltas", {})
        fairness = row.get("fairness", {})
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(row.get('name')))}</td>"
            f"<td>{html_lib.escape(str(row.get('decision')))}</td>"
            f"<td>{_fmt(metrics.get('metrics/mAP50(M)'))}</td>"
            f"<td>{_fmt(deltas.get('metrics/mAP50(M)'), signed=True)}</td>"
            f"<td>{_fmt(metrics.get('metrics/mAP50-95(M)'))}</td>"
            f"<td>{_fmt(deltas.get('metrics/mAP50-95(M)'), signed=True)}</td>"
            f"<td>{_fmt(metrics.get('metrics/recall(M)'))}</td>"
            f"<td>{_fmt(deltas.get('metrics/recall(M)'), signed=True)}</td>"
            f"<td>{_fmt(metrics.get('metrics/precision(M)'))}</td>"
            f"<td>{html_lib.escape(str(fairness.get('core_config_match')))}</td>"
            f"<td>{html_lib.escape(str(fairness.get('amp_match')))}</td>"
            f"<td><code>{html_lib.escape(str(row.get('run_dir')))}</code></td>"
            "</tr>"
        )
    check_rows = []
    for check in summary.get("checks", []):
        check_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(check.get('id')))}</td>"
            f"<td>{html_lib.escape(str(check.get('status')))}</td>"
            f"<td>{html_lib.escape(str(check.get('description')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_lib.escape(str(summary.get('title', 'LIS YOLO Segmentation Training Rollup')))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17212b; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 30px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:last-child, td:last-child {{ text-align: left; }}
    th {{ background: #f3f6f9; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px 14px; margin: 14px 0; }}
    code {{ background: #f5f7fa; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html_lib.escape(str(summary.get('title', 'LIS YOLO Segmentation Training Rollup')))}</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <b>{html_lib.escape(str(summary.get('status')))}</b></p>
  <h2>Baseline</h2>
  <p><b>{html_lib.escape(str(baseline.get('name')))}</b> mask mAP50={_fmt(baseline.get('metrics', {}).get('metrics/mAP50(M)'))}, mask mAP50-95={_fmt(baseline.get('metrics', {}).get('metrics/mAP50-95(M)'))}, mask recall={_fmt(baseline.get('metrics', {}).get('metrics/recall(M)'))}</p>
  <h2>Candidate Comparison</h2>
  <table><thead><tr><th>Run</th><th>Decision</th><th>Mask mAP50</th><th>dMask mAP50</th><th>Mask mAP50-95</th><th>dMask mAP50-95</th><th>Mask Recall</th><th>dRecall</th><th>Mask Precision</th><th>Core Match</th><th>AMP Match</th><th>Path</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Description</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table>
</body>
</html>
"""


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _canonical(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
