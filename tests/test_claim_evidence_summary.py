from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.reporting.claim_evidence_summary import (
    build_claim_evidence_summary_from_path,
    main as evidence_main,
    write_claim_evidence_summary,
)


class ClaimEvidenceSummaryTest(unittest.TestCase):
    def test_builds_claim_language_summary_from_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dashboard = _write_dashboard(root / "dashboard")

            summary = build_claim_evidence_summary_from_path(dashboard)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_level"], "narrow_fp_reducer_claim_ready")
            self.assertEqual(summary["metric_claim_status"], "fp_reducer_only")
            self.assertEqual([row["area"] for row in summary["supported_performance_claims"]], ["Recall-budgeted FP reduction"])
            unsupported_areas = {row["area"] for row in summary["unsupported_performance_claims"]}
            self.assertIn("Broad HumanISP superiority", unsupported_areas)
            self.assertIn("Task-level recall claim", unsupported_areas)
            self.assertIn("RGB+Aux DNN training path", unsupported_areas)
            self.assertTrue(any("false-positive reduction" in text for text in summary["allowed_language"]))
            self.assertTrue(any("Do not claim broad HumanISP superiority" in text for text in summary["disallowed_language"]))
            self.assertTrue(any(row["area"] == "CFA/LensPSF visual casebook" for row in summary["diagnostic_support"]))
            self.assertTrue(any(row["area"] == "Large held-out native RAW benchmark" for row in summary["claim_boundaries"]))
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["supported_claims_are_narrow_when_fp_reducer_only"], "pass")
            self.assertEqual(checks["broad_superiority_blocked"], "pass")

            html_path = write_claim_evidence_summary(summary, root / "summary")
            html = html_path.read_text()
            self.assertIn("Allowed Language", html)
            self.assertIn("Do Not Claim", html)
            self.assertIn("CFA/LensPSF visual casebook", html)
            self.assertTrue((html_path.parent / "claim_evidence_summary.json").exists())

    def test_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dashboard = _write_dashboard(root / "dashboard")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = evidence_main([str(dashboard), "--output-dir", str(root / "summary")])

            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["claim_level"], "narrow_fp_reducer_claim_ready")
            self.assertEqual(printed["supported_claim_count"], 1)
            self.assertTrue((root / "summary" / "claim_evidence_summary.json").exists())


def _write_dashboard(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "claims": [
            _claim("Aux broad vs Human", "broad_superiority", passed=False),
            _claim("Aux FP reducer vs Human", "fp_reducer", passed=True),
        ],
        "decisions": [
            {"status": "not_supported", "claim": "Broad HumanISP superiority is not supported by the current gate evidence."},
            {"status": "supported", "claim": "Recall-budgeted FP reduction versus HumanISP is supported."},
            {"status": "not_supported", "claim": "Task-level `recall_improvement` gate failed for vru, person; do not promote that task-level claim."},
        ],
        "evidence_map": {
            "claim_posture": {
                "recommended_claim": "Use a narrow recall-budgeted FP-reduction claim, with front-end/aux evidence as feasibility support.",
                "blocked_claim": "Do not claim broad HumanISP superiority.",
                "metric_claim_status": "fp_reducer_only",
            },
            "current_evidence": [
                {
                    "area": "Broad HumanISP superiority",
                    "status": "not_supported",
                    "claim_strength": "blocked_by_gate",
                    "evidence": "samples=1496; dR50=-0.0062; failed=recall@0.50_mean",
                    "claim_boundary": "Do not claim broad HumanISP superiority; at least one configured broad metric gate failed.",
                    "next_evidence": "Improve recall.",
                },
                {
                    "area": "Recall-budgeted FP reduction",
                    "status": "supported",
                    "claim_strength": "claim_ready",
                    "evidence": "samples=1496; dP50=+0.0241; dR50=-0.0062; dFP50=-0.2246; failed=none",
                    "claim_boundary": "Use this as a narrow FP-reduction claim versus HumanISP, not as a full detector superiority claim.",
                    "next_evidence": "Add condition slices.",
                },
                {
                    "area": "Task-level recall claim",
                    "status": "not_supported",
                    "claim_strength": "not_claim_ready",
                    "evidence": "task gate task_gate_fail; failed=vru, person",
                    "claim_boundary": "Task-level claims need the configured task gate to pass for priority groups.",
                    "next_evidence": "Tune recall.",
                },
                {
                    "area": "Benchmark protocol coverage",
                    "status": "supported",
                    "claim_strength": "fp_reducer_only",
                    "evidence": "coverage=coverage_complete; metric_claim_status=fp_reducer_only; missing=none",
                    "claim_boundary": "Coverage can be complete while the metric claim remains narrow.",
                    "next_evidence": "Keep protocol rows covered.",
                },
                {
                    "area": "Large held-out native RAW benchmark",
                    "status": "diagnostic",
                    "claim_strength": "large_native_fp_reducer_with_recall_tradeoff",
                    "evidence": "samples=1496/1000; dP50=+0.0300; dR50=-0.0060; dFP50=-0.3500",
                    "claim_boundary": "Large native RAW provenance only; not all-CFA or real adverse RAW proof.",
                    "next_evidence": "Repeat on real RAW datasets.",
                },
                {
                    "area": "CFA/LensPSF visual casebook",
                    "status": "diagnostic",
                    "claim_strength": "condition_qualitative_review",
                    "evidence": "conditions=12/12; selected=26; fp_success=24; counterexamples=2",
                    "claim_boundary": "Qualitative condition-slice review only.",
                    "next_evidence": "Expand selected cases.",
                },
                {
                    "area": "RGB+Aux DNN training path",
                    "status": "needs_eval",
                    "claim_strength": "diagnostic_only",
                    "evidence": "status=diagnostic_only; runs=10",
                    "claim_boundary": "Use as implementation/resource evidence until held-out DNN detector metrics pass a gate.",
                    "next_evidence": "Run RGB-only versus RGB+Aux fine-tune.",
                },
            ],
            "future_evidence": [
                {
                    "priority": "P0",
                    "evidence": "Scene-edge proposal correlation across CFA/LensPSF",
                    "why": "Tests whether proposal evidence holds across conditions.",
                    "current_gap": "Needs larger native_bayer_v1 reruns.",
                    "implementation_path": "Run matched CFA/LensPSF cases.",
                }
            ],
        },
        "interpretation": "unit dashboard",
    }
    (path / "claim_dashboard_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _claim(name: str, profile: str, *, passed: bool) -> dict:
    return {
        "name": name,
        "profile": profile,
        "pass": passed,
        "sample_count": 1496,
        "baseline_input": "human_rgb",
        "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
        "metrics": {
            "precision@0.50_mean": {"delta": 0.0241, "ci_low": 0.0179, "ci_high": 0.0308, "pass": True},
            "recall@0.50_mean": {"delta": -0.0062, "ci_low": -0.0097, "ci_high": -0.0028, "pass": passed},
            "fp@0.50_mean": {"delta": -0.2246, "ci_low": -0.2574, "ci_high": -0.1932, "pass": True},
        },
    }


if __name__ == "__main__":
    unittest.main()
