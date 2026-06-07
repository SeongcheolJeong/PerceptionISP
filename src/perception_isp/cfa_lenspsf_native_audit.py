"""Native-CFA separation audit for CFA/LensPSF detector sweeps."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "cfa_lenspsf_native_audit_summary.json"
SWEEP_SUMMARY_FILENAME = "cfa_lenspsf_detector_sweep_summary.json"


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Separate native-CFA and remapped rows in a CFA/LensPSF detector sweep.")
    parser.add_argument("sweep", help="CFA/LensPSF detector sweep summary path or directory.")
    parser.add_argument("--output-dir", default="reports/perception_cfa_lenspsf_native_audit")
    args = parser.parse_args(argv)

    summary = build_native_audit_from_path(args.sweep)
    html_path = write_native_audit(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "native_run_count": summary["groups"]["native"]["run_count"],
                    "remapped_run_count": summary["groups"]["remapped"]["run_count"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_native_audit_from_path(sweep: str | Path) -> Dict[str, Any]:
    summary_path = _summary_path(sweep, SWEEP_SUMMARY_FILENAME)
    sweep_summary = json.loads(summary_path.read_text())
    return build_native_audit(sweep_summary, sweep_summary_path=summary_path)


def build_native_audit(
    sweep_summary: Mapping[str, Any],
    *,
    sweep_summary_path: str | Path,
) -> Dict[str, Any]:
    sweep_path = Path(sweep_summary_path).expanduser()
    sweep_dir = sweep_path.parent
    rows = [_run_summary(run, sweep_dir=sweep_dir) for run in sweep_summary.get("runs", ()) if isinstance(run, Mapping)]
    groups = {
        "native": _group_summary([row for row in rows if row["native_status"] == "native"]),
        "partial_remap": _group_summary([row for row in rows if row["native_status"] == "partial_remap"]),
        "remapped": _group_summary([row for row in rows if row["native_status"] == "remapped"]),
    }
    checks = _checks(rows, groups)
    return {
        "name": "CFA/LensPSF native-CFA separation audit",
        "source_sweep": str(sweep_path),
        "source_sweep_html": str(sweep_dir / "index.html") if (sweep_dir / "index.html").exists() else "",
        "run_count": len(rows),
        "expected_run_count": int(sweep_summary.get("run_count", len(rows))),
        "cfa_patterns": [str(value) for value in sweep_summary.get("cfa_patterns", ())],
        "psf_sigmas": [float(value) for value in sweep_summary.get("psf_sigmas", ())],
        "groups": groups,
        "runs": rows,
        "checks": checks,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This audit separates detector-sweep rows whose CameraE2E source CFA already matches the ISP target CFA from rows "
            "that were bridge-remapped to a different requested CFA. It keeps native sensor-CFA evidence separate from remap sensitivity evidence."
        ),
        "claim_boundary": (
            "Only the native group can be used as native sensor-CFA evidence. Remapped rows can support bridge/remap sensitivity analysis, "
            "but they are not proof that the corresponding CFA was natively simulated by CameraE2E."
        ),
    }


def write_native_audit(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _run_summary(run: Mapping[str, Any], *, sweep_dir: Path) -> Dict[str, Any]:
    raw = run.get("raw_condition_summary", {}) if isinstance(run.get("raw_condition_summary"), Mapping) else {}
    primary = _primary_downstream_input(run)
    metrics = run.get("metrics", {}).get(primary, {}) if isinstance(run.get("metrics"), Mapping) else {}
    deltas = run.get("delta_vs_human", {}).get(primary, {}) if isinstance(run.get("delta_vs_human"), Mapping) else {}
    sample_count = int(raw.get("sample_count", run.get("sample_count", 0)))
    remapped_count = int(raw.get("pattern_remapped_count", 0))
    remapped_fraction = _optional_float(raw.get("pattern_remapped_fraction"))
    if remapped_fraction is None:
        remapped_fraction = 0.0 if sample_count <= 0 else float(remapped_count / max(sample_count, 1))
    native_status = "native" if remapped_count == 0 else "remapped" if remapped_count == sample_count and sample_count > 0 else "partial_remap"
    report = str(run.get("report", ""))
    return {
        "run_id": str(run.get("run_id", "")),
        "report": report,
        "html_path": str(sweep_dir / report) if report else "",
        "cfa_pattern": str(run.get("cfa_pattern", "")),
        "psf_sigma": _optional_float(run.get("psf_sigma")),
        "sample_count": int(sample_count),
        "source_cfa_patterns": {str(k): int(v) for k, v in raw.get("source_cfa_patterns", {}).items()} if isinstance(raw.get("source_cfa_patterns"), Mapping) else {},
        "target_cfa_patterns": {str(k): int(v) for k, v in raw.get("target_cfa_patterns", {}).items()} if isinstance(raw.get("target_cfa_patterns"), Mapping) else {},
        "pattern_remapped_count": remapped_count,
        "pattern_remapped_fraction": remapped_fraction,
        "native_status": native_status,
        "primary_input": primary,
        "precision@0.50_mean": _optional_float(metrics.get("precision@0.50_mean")),
        "recall@0.50_mean": _optional_float(metrics.get("recall@0.50_mean")),
        "small_recall@0.50_mean": _optional_float(metrics.get("small_recall@0.50_mean")),
        "fp@0.50_mean": _optional_float(metrics.get("fp@0.50_mean")),
        "delta_precision@0.50_mean": _optional_float(deltas.get("precision@0.50_mean")),
        "delta_recall@0.50_mean": _optional_float(deltas.get("recall@0.50_mean")),
        "delta_small_recall@0.50_mean": _optional_float(deltas.get("small_recall@0.50_mean")),
        "delta_fp@0.50_mean": _optional_float(deltas.get("fp@0.50_mean")),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cfas = sorted({str(row.get("cfa_pattern", "")) for row in rows if row.get("cfa_pattern")})
    psf = sorted({float(row.get("psf_sigma")) for row in rows if row.get("psf_sigma") is not None})
    sample_count = sum(int(row.get("sample_count", 0)) for row in rows)
    best_fp = _best_row(rows, "delta_fp@0.50_mean", lower_is_better=True)
    best_recall = _best_row(rows, "delta_recall@0.50_mean", lower_is_better=False)
    return {
        "run_count": len(rows),
        "sample_count": int(sample_count),
        "cfa_patterns": cfas,
        "psf_sigmas": psf,
        "mean_delta_precision@0.50": _weighted_mean(rows, "delta_precision@0.50_mean"),
        "mean_delta_recall@0.50": _weighted_mean(rows, "delta_recall@0.50_mean"),
        "mean_delta_small_recall@0.50": _weighted_mean(rows, "delta_small_recall@0.50_mean"),
        "mean_delta_fp@0.50": _weighted_mean(rows, "delta_fp@0.50_mean"),
        "best_delta_fp@0.50": best_fp,
        "best_delta_recall@0.50": best_recall,
    }


def _checks(rows: Sequence[Mapping[str, Any]], groups: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    native_runs = int(groups.get("native", {}).get("run_count", 0))
    remapped_runs = int(groups.get("remapped", {}).get("run_count", 0))
    partial_runs = int(groups.get("partial_remap", {}).get("run_count", 0))
    all_rows_native = bool(rows) and native_runs == len(rows) and remapped_runs == 0 and partial_runs == 0
    return [
        {
            "id": "sweep_rows_available",
            "status": "pass" if rows else "fail",
            "evidence": f"runs={len(rows)}",
        },
        {
            "id": "native_rows_identified",
            "status": "pass" if native_runs > 0 else "fail",
            "evidence": f"native_runs={native_runs}",
        },
        {
            "id": "remapped_rows_separated",
            "status": "pass" if all_rows_native or remapped_runs > 0 or partial_runs > 0 else "warning",
            "evidence": f"remapped_runs={remapped_runs} partial_runs={partial_runs}",
        },
    ]


def _primary_downstream_input(run: Mapping[str, Any]) -> str:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}
    for input_name in metrics:
        if str(input_name).startswith("perception_calibrated"):
            return str(input_name)
    if "perception_fusion_rgb_aux" in metrics:
        return "perception_fusion_rgb_aux"
    if "perception_rgb" in metrics:
        return "perception_rgb"
    return next(iter(metrics), "")


def _best_row(rows: Sequence[Mapping[str, Any]], key: str, *, lower_is_better: bool) -> Dict[str, Any]:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return {}
    best = min(candidates, key=lambda row: float(row.get(key, 0.0))) if lower_is_better else max(candidates, key=lambda row: float(row.get(key, 0.0)))
    return {
        "run_id": best.get("run_id"),
        "cfa_pattern": best.get("cfa_pattern"),
        "psf_sigma": best.get("psf_sigma"),
        "primary_input": best.get("primary_input"),
        "delta": best.get(key),
    }


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        weight = max(float(row.get("sample_count", 0)), 1.0)
        total += float(value) * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return float(total / weight_sum)


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    group_rows = "".join(_group_row(name, group) for name, group in summary.get("groups", {}).items() if isinstance(group, Mapping))
    run_rows = "".join(_run_row(row, destination) for row in summary.get("runs", ()) if isinstance(row, Mapping))
    check_rows = "".join(
        f"<tr><td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td></tr>"
        for row in summary.get("checks", ())
        if isinstance(row, Mapping)
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP CFA/LensPSF Native-CFA Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f1; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>PerceptionISP CFA/LensPSF Native-CFA Audit</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))} {html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>. Source sweep: {_source_sweep_link(summary, destination)}</p>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Native/Remap Group Summary</h2>
  <table><thead><tr><th>Group</th><th>Runs</th><th>Samples</th><th>CFA</th><th>PSF</th><th>Mean dP50</th><th>Mean dR50</th><th>Mean dSmallR50</th><th>Mean dFP50</th><th>Best dFP50</th><th>Best dR50</th></tr></thead><tbody>{group_rows}</tbody></table>
  <h2>Runs</h2>
  <table><thead><tr><th>Run</th><th>Status</th><th>CFA</th><th>PSF</th><th>Samples</th><th>Remap</th><th>Source CFA</th><th>Target CFA</th><th>dP50</th><th>dR50</th><th>dSmallR50</th><th>dFP50</th></tr></thead><tbody>{run_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _group_row(name: str, group: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(name))}</code></td>"
        f"<td>{int(group.get('run_count', 0))}</td>"
        f"<td>{int(group.get('sample_count', 0))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in group.get('cfa_patterns', ())) or 'none')}</td>"
        f"<td>{html_lib.escape(', '.join(_fmt(value) for value in group.get('psf_sigmas', ())) or 'none')}</td>"
        f"<td>{_fmt(group.get('mean_delta_precision@0.50'), signed=True)}</td>"
        f"<td>{_fmt(group.get('mean_delta_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(group.get('mean_delta_small_recall@0.50'), signed=True)}</td>"
        f"<td>{_fmt(group.get('mean_delta_fp@0.50'), signed=True)}</td>"
        f"<td>{_best_text(group.get('best_delta_fp@0.50'))}</td>"
        f"<td>{_best_text(group.get('best_delta_recall@0.50'))}</td>"
        "</tr>"
    )


def _run_row(row: Mapping[str, Any], destination: Path) -> str:
    link = html_lib.escape(str(row.get("run_id", "")))
    if row.get("html_path"):
        relative = os.path.relpath(str(row.get("html_path")), start=str(destination))
        link = f"<a href=\"{html_lib.escape(relative)}\">{html_lib.escape(str(row.get('run_id', '')))}</a>"
    return (
        "<tr>"
        f"<td>{link}</td>"
        f"<td><code>{html_lib.escape(str(row.get('native_status', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('cfa_pattern', '')))}</td>"
        f"<td>{_fmt(row.get('psf_sigma'))}</td>"
        f"<td>{int(row.get('sample_count', 0))}</td>"
        f"<td>{int(row.get('pattern_remapped_count', 0))} ({_fmt(row.get('pattern_remapped_fraction'))})</td>"
        f"<td>{html_lib.escape(str(row.get('source_cfa_patterns', {})))}</td>"
        f"<td>{html_lib.escape(str(row.get('target_cfa_patterns', {})))}</td>"
        f"<td>{_fmt(row.get('delta_precision@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_small_recall@0.50_mean'), signed=True)}</td>"
        f"<td>{_fmt(row.get('delta_fp@0.50_mean'), signed=True)}</td>"
        "</tr>"
    )


def _source_sweep_link(summary: Mapping[str, Any], destination: Path) -> str:
    html_path = str(summary.get("source_sweep_html", ""))
    if not html_path:
        return ""
    relative = os.path.relpath(html_path, start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open</a>"


def _best_text(payload: Any) -> str:
    if not isinstance(payload, Mapping) or not payload:
        return ""
    return (
        f"<code>{html_lib.escape(str(payload.get('run_id', '')))}</code> "
        f"{_fmt(payload.get('delta'), signed=True)}"
    )


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _optional_float(value: Any) -> float | None:
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
