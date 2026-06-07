"""Summarize defensible PerceptionISP claim language from a dashboard."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "claim_evidence_summary.json"
DASHBOARD_SUMMARY = "claim_dashboard_summary.json"


PERFORMANCE_AREAS = {
    "Broad HumanISP superiority",
    "Recall-budgeted FP reduction",
    "Task-level recall claim",
    "RGB+Aux DNN training path",
    "RGB+Aux DNN fine-tune gate",
    "RGB+Aux DNN operating-point sweep",
}
BOUNDARY_AREAS = {
    "Benchmark protocol coverage",
    "Large held-out native RAW benchmark",
    "CFA/LensPSF native-CFA separation",
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a concise claim-evidence summary from a claim dashboard.")
    parser.add_argument("dashboard", help="claim_dashboard_summary.json path or dashboard directory.")
    parser.add_argument("--output-dir", default="reports/perception_claim_evidence_summary")
    args = parser.parse_args(argv)

    summary = build_claim_evidence_summary_from_path(args.dashboard)
    html_path = write_claim_evidence_summary(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "claim_level": summary["claim_level"],
                    "supported_claim_count": len(summary["supported_performance_claims"]),
                    "unsupported_claim_count": len(summary["unsupported_performance_claims"]),
                }
            ),
            indent=2,
        )
    )
    return 0


def build_claim_evidence_summary_from_path(path: str | Path) -> Dict[str, Any]:
    summary_path = _summary_path(path, DASHBOARD_SUMMARY)
    dashboard = json.loads(summary_path.read_text())
    return build_claim_evidence_summary(dashboard, dashboard_summary_path=summary_path)


def build_claim_evidence_summary(dashboard: Mapping[str, Any], *, dashboard_summary_path: str | Path) -> Dict[str, Any]:
    dashboard_path = Path(dashboard_summary_path).expanduser()
    evidence_map = dashboard.get("evidence_map", {}) if isinstance(dashboard.get("evidence_map"), Mapping) else {}
    posture = evidence_map.get("claim_posture", {}) if isinstance(evidence_map.get("claim_posture"), Mapping) else {}
    current_rows = [row for row in evidence_map.get("current_evidence", ()) if isinstance(row, Mapping)]
    future_rows = [row for row in evidence_map.get("future_evidence", ()) if isinstance(row, Mapping)]
    claims = [claim for claim in dashboard.get("claims", ()) if isinstance(claim, Mapping)]
    decisions = [decision for decision in dashboard.get("decisions", ()) if isinstance(decision, Mapping)]
    supported = _supported_performance_claims(current_rows, claims)
    unsupported = _unsupported_performance_claims(current_rows)
    diagnostics = _diagnostic_rows(current_rows)
    boundaries = _claim_boundary_rows(current_rows)
    allowed_language = _allowed_language(supported, diagnostics, posture)
    disallowed_language = _disallowed_language(unsupported, decisions, posture)
    checks = _checks(
        posture=posture,
        supported=supported,
        unsupported=unsupported,
        diagnostics=diagnostics,
        boundaries=boundaries,
    )
    return {
        "name": "PerceptionISP claim evidence summary",
        "source_dashboard_summary": str(dashboard_path),
        "source_dashboard_html": _sibling_html(dashboard_path),
        "claim_level": _claim_level(posture, supported),
        "metric_claim_status": str(posture.get("metric_claim_status", "")),
        "recommended_claim": str(posture.get("recommended_claim", "")),
        "blocked_claim": str(posture.get("blocked_claim", "")),
        "allowed_language": allowed_language,
        "disallowed_language": disallowed_language,
        "supported_performance_claims": supported,
        "unsupported_performance_claims": unsupported,
        "diagnostic_support": diagnostics,
        "claim_boundaries": boundaries,
        "next_evidence": [_future_row(row) for row in future_rows],
        "checks": checks,
        "status": "pass" if checks and all(row["status"] == "pass" for row in checks) else "warning",
        "interpretation": (
            "This report turns the claim-readiness dashboard into reusable claim language. "
            "It separates supported performance claims from diagnostic evidence so feasibility evidence is not overstated."
        ),
    }


def write_claim_evidence_summary(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary, destination))
    return html_path


def _supported_performance_claims(rows: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    claim_by_area = {
        "Broad HumanISP superiority": _claim_by_profile(claims, "broad_superiority"),
        "Recall-budgeted FP reduction": _claim_by_profile(claims, "fp_reducer"),
    }
    supported = []
    for row in rows:
        area = str(row.get("area", ""))
        if area not in PERFORMANCE_AREAS:
            continue
        if str(row.get("status", "")) != "supported":
            continue
        supported.append(
            {
                "area": area,
                "claim_strength": str(row.get("claim_strength", "")),
                "evidence": str(row.get("evidence", "")),
                "claim_boundary": str(row.get("claim_boundary", "")),
                "talking_point": _talking_point(area, row, claim_by_area.get(area)),
                "source_claim": _claim_brief(claim_by_area.get(area)),
            }
        )
    return supported


def _unsupported_performance_claims(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    unsupported = []
    for row in rows:
        area = str(row.get("area", ""))
        if area not in PERFORMANCE_AREAS:
            continue
        if str(row.get("status", "")) == "supported":
            continue
        unsupported.append(
            {
                "area": area,
                "status": str(row.get("status", "")),
                "claim_strength": str(row.get("claim_strength", "")),
                "evidence": str(row.get("evidence", "")),
                "claim_boundary": str(row.get("claim_boundary", "")),
                "next_evidence": str(row.get("next_evidence", "")),
            }
        )
    return unsupported


def _diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    diagnostics = []
    for row in rows:
        area = str(row.get("area", ""))
        if area in PERFORMANCE_AREAS or area in BOUNDARY_AREAS:
            continue
        if str(row.get("status", "")) not in {"supported", "diagnostic"}:
            continue
        diagnostics.append(
            {
                "area": area,
                "status": str(row.get("status", "")),
                "claim_strength": str(row.get("claim_strength", "")),
                "evidence": str(row.get("evidence", "")),
                "claim_boundary": str(row.get("claim_boundary", "")),
            }
        )
    return diagnostics


def _claim_boundary_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    boundaries = []
    for row in rows:
        boundary = str(row.get("claim_boundary", ""))
        if not boundary:
            continue
        if str(row.get("status", "")) == "not_supported" and str(row.get("area", "")) not in PERFORMANCE_AREAS:
            continue
        boundaries.append(
            {
                "area": str(row.get("area", "")),
                "status": str(row.get("status", "")),
                "claim_boundary": boundary,
            }
        )
    return boundaries


def _allowed_language(
    supported: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    posture: Mapping[str, Any],
) -> list[str]:
    lines = []
    for row in supported:
        talking_point = str(row.get("talking_point", ""))
        if talking_point:
            lines.append(talking_point)
    if diagnostics:
        lines.append(
            "PerceptionISP front-end and aux evidence can be described as feasibility and mechanism support, not as broad detector superiority."
        )
    recommended = str(posture.get("recommended_claim", ""))
    if recommended:
        lines.append(recommended)
    return _dedupe(lines)


def _disallowed_language(
    unsupported: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    posture: Mapping[str, Any],
) -> list[str]:
    lines = []
    blocked = str(posture.get("blocked_claim", ""))
    if blocked:
        lines.append(blocked)
    for row in unsupported:
        area = str(row.get("area", ""))
        if area == "Task-level recall claim":
            lines.append("Do not claim task-level recall improvement for VRU/person/small-object groups until the task gate passes.")
        elif area == "RGB+Aux DNN training path":
            lines.append("Do not claim a trained RGB+Aux DNN detector improvement from the current compact dense-detector metrics.")
        elif area == "RGB+Aux DNN fine-tune gate":
            lines.append("Do not claim the aux tensor improves a learned detector until the RGB+Aux DNN gate passes.")
        elif area == "RGB+Aux DNN operating-point sweep":
            lines.append("Do not claim a learned RGB+Aux detector operating point until the confidence sweep finds a passing threshold.")
        elif area == "Broad HumanISP superiority":
            lines.append("Do not claim broad HumanISP superiority from the current gate evidence.")
    for decision in decisions:
        text = str(decision.get("claim", ""))
        if str(decision.get("status", "")) == "not_supported" and ("do not" in text.lower() or "not supported" in text.lower()):
            lines.append(text)
    return _dedupe(lines)


def _checks(
    *,
    posture: Mapping[str, Any],
    supported: Sequence[Mapping[str, Any]],
    unsupported: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Any], ...]:
    metric_status = str(posture.get("metric_claim_status", ""))
    unsupported_areas = {str(row.get("area", "")) for row in unsupported}
    return (
        {
            "id": "metric_claim_status_recorded",
            "status": "pass" if metric_status else "fail",
            "evidence": f"metric_claim_status={metric_status or 'missing'}",
        },
        {
            "id": "supported_claims_are_narrow_when_fp_reducer_only",
            "status": "pass" if metric_status != "fp_reducer_only" or _only_fp_reducer_supported(supported) else "fail",
            "evidence": f"metric_claim_status={metric_status}; supported={','.join(str(row.get('area', '')) for row in supported) or 'none'}",
        },
        {
            "id": "broad_superiority_blocked",
            "status": "pass" if "Broad HumanISP superiority" in unsupported_areas else "fail",
            "evidence": f"unsupported={','.join(sorted(unsupported_areas)) or 'none'}",
        },
        {
            "id": "diagnostic_evidence_has_boundaries",
            "status": "pass" if diagnostics and all(str(row.get("claim_boundary", "")) for row in diagnostics) else "fail",
            "evidence": f"diagnostics={len(diagnostics)}",
        },
        {
            "id": "claim_boundaries_available",
            "status": "pass" if len(boundaries) >= len(diagnostics) else "fail",
            "evidence": f"boundaries={len(boundaries)} diagnostics={len(diagnostics)}",
        },
    )


def _only_fp_reducer_supported(supported: Sequence[Mapping[str, Any]]) -> bool:
    areas = {str(row.get("area", "")) for row in supported}
    return areas == {"Recall-budgeted FP reduction"}


def _claim_level(posture: Mapping[str, Any], supported: Sequence[Mapping[str, Any]]) -> str:
    metric_status = str(posture.get("metric_claim_status", ""))
    if metric_status == "fp_reducer_only" and _only_fp_reducer_supported(supported):
        return "narrow_fp_reducer_claim_ready"
    if any(str(row.get("area", "")) == "Broad HumanISP superiority" for row in supported):
        return "broad_claim_ready"
    if supported:
        return "limited_claim_ready"
    return "diagnostic_only"


def _talking_point(area: str, row: Mapping[str, Any], claim: Mapping[str, Any] | None) -> str:
    if area == "Recall-budgeted FP reduction":
        metrics = claim.get("metrics", {}) if isinstance(claim, Mapping) and isinstance(claim.get("metrics"), Mapping) else {}
        samples = int(claim.get("sample_count", 0)) if isinstance(claim, Mapping) else 0
        return (
            "PerceptionISP supports a recall-budgeted false-positive reduction claim versus HumanISP"
            f" on {samples} held-out samples: "
            f"dP50={_metric_delta(metrics, 'precision@0.50_mean')}, "
            f"dR50={_metric_delta(metrics, 'recall@0.50_mean')}, "
            f"dFP50={_metric_delta(metrics, 'fp@0.50_mean')}."
        )
    if area == "Broad HumanISP superiority":
        return "Broad HumanISP superiority is supported by the configured gate."
    return str(row.get("evidence", ""))


def _metric_delta(metrics: Mapping[str, Any], name: str) -> str:
    row = metrics.get(name, {}) if isinstance(metrics.get(name), Mapping) else {}
    delta = row.get("delta")
    if delta is None:
        return "n/a"
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    ci = "" if ci_low is None or ci_high is None else f" CI[{_fmt(ci_low, signed=True)}, {_fmt(ci_high, signed=True)}]"
    return f"{_fmt(delta, signed=True)}{ci}"


def _claim_by_profile(claims: Sequence[Mapping[str, Any]], profile: str) -> Mapping[str, Any] | None:
    for claim in claims:
        if str(claim.get("profile", "")) == profile:
            return claim
    return None


def _claim_brief(claim: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if claim is None:
        return None
    return {
        "name": str(claim.get("name", "")),
        "profile": str(claim.get("profile", "")),
        "pass": bool(claim.get("pass")),
        "sample_count": int(claim.get("sample_count", 0)),
        "baseline_input": str(claim.get("baseline_input", "")),
        "target_input": str(claim.get("target_input", "")),
        "html_path": claim.get("html_path"),
    }


def _future_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "priority": str(row.get("priority", "")),
        "evidence": str(row.get("evidence", "")),
        "why": str(row.get("why", "")),
        "current_gap": str(row.get("current_gap", "")),
        "implementation_path": str(row.get("implementation_path", "")),
    }


def _render_html(summary: Mapping[str, Any], destination: Path) -> str:
    allowed_rows = "".join(f"<li>{html_lib.escape(str(row))}</li>" for row in summary.get("allowed_language", ()))
    disallowed_rows = "".join(f"<li>{html_lib.escape(str(row))}</li>" for row in summary.get("disallowed_language", ()))
    supported_rows = "".join(_claim_row(row) for row in summary.get("supported_performance_claims", ()) if isinstance(row, Mapping))
    if not supported_rows:
        supported_rows = '<tr><td colspan="4">No supported performance claims were found.</td></tr>'
    unsupported_rows = "".join(_unsupported_row(row) for row in summary.get("unsupported_performance_claims", ()) if isinstance(row, Mapping))
    if not unsupported_rows:
        unsupported_rows = '<tr><td colspan="5">No unsupported performance rows were found.</td></tr>'
    diagnostic_rows = "".join(_diagnostic_row(row) for row in summary.get("diagnostic_support", ()) if isinstance(row, Mapping))
    if not diagnostic_rows:
        diagnostic_rows = '<tr><td colspan="5">No diagnostic support rows were found.</td></tr>'
    boundary_rows = "".join(_boundary_row(row) for row in summary.get("claim_boundaries", ()) if isinstance(row, Mapping))
    next_rows = "".join(_next_row(row) for row in summary.get("next_evidence", ()) if isinstance(row, Mapping))
    check_rows = "".join(_check_row(row) for row in summary.get("checks", ()) if isinstance(row, Mapping))
    source_link = _source_link(summary, destination)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>PerceptionISP Claim Evidence Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; }}
    .pass, .supported {{ color: #047857; font-weight: 700; }}
    .fail, .not_supported {{ color: #b91c1c; font-weight: 700; }}
    .warning, .diagnostic, .needs_eval, .needs_gate {{ color: #a16207; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>PerceptionISP Claim Evidence Summary</h1>
  <div class=\"note\">{html_lib.escape(str(summary.get('interpretation', '')))}</div>
  <table><tbody>
    <tr><th>Status</th><td><code>{html_lib.escape(str(summary.get('status', '')))}</code></td></tr>
    <tr><th>Claim level</th><td><code>{html_lib.escape(str(summary.get('claim_level', '')))}</code></td></tr>
    <tr><th>Metric claim status</th><td><code>{html_lib.escape(str(summary.get('metric_claim_status', '')))}</code></td></tr>
    <tr><th>Recommended claim</th><td>{html_lib.escape(str(summary.get('recommended_claim', '')))}</td></tr>
    <tr><th>Blocked claim</th><td>{html_lib.escape(str(summary.get('blocked_claim', '')) or 'none')}</td></tr>
    <tr><th>Source dashboard</th><td>{source_link}</td></tr>
  </tbody></table>
  <h2>Allowed Language</h2>
  <ul>{allowed_rows}</ul>
  <h2>Do Not Claim</h2>
  <ul>{disallowed_rows}</ul>
  <h2>Supported Performance Claims</h2>
  <table><thead><tr><th>Area</th><th>Strength</th><th>Talking Point</th><th>Boundary</th></tr></thead><tbody>{supported_rows}</tbody></table>
  <h2>Unsupported Performance Claims</h2>
  <table><thead><tr><th>Area</th><th>Status</th><th>Strength</th><th>Evidence</th><th>Next Evidence</th></tr></thead><tbody>{unsupported_rows}</tbody></table>
  <h2>Diagnostic Support</h2>
  <table><thead><tr><th>Area</th><th>Status</th><th>Strength</th><th>Evidence</th><th>Boundary</th></tr></thead><tbody>{diagnostic_rows}</tbody></table>
  <h2>Claim Boundaries</h2>
  <table><thead><tr><th>Area</th><th>Status</th><th>Boundary</th></tr></thead><tbody>{boundary_rows}</tbody></table>
  <h2>Next Evidence</h2>
  <table><thead><tr><th>Priority</th><th>Evidence</th><th>Why</th><th>Current Gap</th><th>Implementation Path</th></tr></thead><tbody>{next_rows}</tbody></table>
  <h2>Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _claim_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('area', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('claim_strength', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('talking_point', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('claim_boundary', '')))}</td>"
        "</tr>"
    )


def _unsupported_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('area', '')))}</td>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('claim_strength', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('next_evidence', '')))}</td>"
        "</tr>"
    )


def _diagnostic_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('area', '')))}</td>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('claim_strength', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('claim_boundary', '')))}</td>"
        "</tr>"
    )


def _boundary_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('area', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('status', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('claim_boundary', '')))}</td>"
        "</tr>"
    )


def _next_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('why', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('current_gap', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('implementation_path', '')))}</td>"
        "</tr>"
    )


def _check_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td class=\"{html_lib.escape(status)}\">{html_lib.escape(status)}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
    )


def _source_link(summary: Mapping[str, Any], destination: Path) -> str:
    html_path = str(summary.get("source_dashboard_html", ""))
    if not html_path:
        return html_lib.escape(str(summary.get("source_dashboard_summary", "")))
    relative = os.path.relpath(html_path, start=str(destination))
    return f"<a href=\"{html_lib.escape(relative)}\">open dashboard</a>"


def _summary_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / filename
    if not candidate.exists():
        raise FileNotFoundError(f"summary not found: {candidate}")
    return candidate


def _sibling_html(summary_path: Path) -> str:
    html_path = summary_path.with_name("index.html")
    return str(html_path) if html_path.exists() else ""


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    prefix = "+" if signed and number >= 0.0 else ""
    return f"{prefix}{number:.4f}"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
