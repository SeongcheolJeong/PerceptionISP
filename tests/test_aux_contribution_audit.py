from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.aux_contribution_audit import (
    build_aux_contribution_audit,
    build_aux_contribution_audit_from_paths,
    main as aux_audit_main,
    write_aux_contribution_audit,
)


class AuxContributionAuditTest(unittest.TestCase):
    def test_build_aux_contribution_audit_passes_incremental_aux_checks(self) -> None:
        summary = build_aux_contribution_audit(_rollup(), calibration_summary=_calibration_summary())

        self.assertEqual(summary["status"], "pass")
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["score_aux_uses_aux_for_fp_reduction"]["status"], "pass")
        self.assertEqual(checks["aux_adds_incremental_value_over_score_label"]["status"], "pass")
        self.assertEqual(checks["score_label_aux_model_contains_aux_features"]["status"], "pass")

        comparisons = {row["id"]: row for row in summary["comparisons"]}
        incremental = comparisons["score_label_aux_vs_score_label"]["deltas"]
        self.assertGreater(incremental["precision@0.50_mean"], 0.0)
        self.assertLess(incremental["fp@0.50_mean"], -0.02)
        self.assertGreaterEqual(incremental["recall@0.50_mean"], -0.005)
        self.assertIn("aux_support", summary["feature_audit"]["aux_features"])

    def test_write_and_cli_output_json_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollup_dir = root / "rollup"
            calibration_dir = root / "calibration"
            rollup_dir.mkdir()
            calibration_dir.mkdir()
            (rollup_dir / "rollup_summary.json").write_text(json.dumps(_rollup()) + "\n")
            (calibration_dir / "proposal_calibration_summary.json").write_text(json.dumps(_calibration_summary()) + "\n")

            summary = build_aux_contribution_audit_from_paths(rollup_dir, calibration_summary=calibration_dir)
            html_path = write_aux_contribution_audit(summary, root / "audit")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Aux Contribution Audit", html_path.read_text())
            persisted = json.loads((html_path.parent / "aux_contribution_audit_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aux_audit_main(
                    [
                        str(rollup_dir),
                        "--calibration-summary",
                        str(calibration_dir),
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["failed_checks"], [])
            self.assertTrue((root / "cli" / "aux_contribution_audit_summary.json").exists())


def _rollup() -> dict:
    baseline = _metrics(0.60, 0.500, 0.300, 0.200, 1.000, 3.000)
    score_aux = _metrics(0.61, 0.497, 0.298, 0.198, 0.950, 2.950)
    score_label = _metrics(0.65, 0.496, 0.297, 0.199, 0.800, 2.800)
    score_label_aux = _metrics(0.66, 0.494, 0.296, 0.198, 0.750, 2.750)
    return {
        "run_count": 4,
        "baseline_input": "perception_fusion_rgb_aux",
        "runs": [
            _run("source", {"perception_fusion_rgb_aux": baseline}),
            _run("score_aux", {"perception_fusion_rgb_aux": baseline, "perception_calibrated_score_aux_fusion_rgb_aux": score_aux}),
            _run("score_label", {"perception_fusion_rgb_aux": baseline, "perception_calibrated_score_label_fusion_rgb_aux": score_label}),
            _run(
                "score_label_aux",
                {
                    "perception_fusion_rgb_aux": baseline,
                    "perception_calibrated_score_label_aux_fusion_rgb_aux": score_label_aux,
                },
            ),
        ],
    }


def _run(name: str, inputs: dict) -> dict:
    return {
        "name": name,
        "sample_count": 10,
        "summary_path": f"/tmp/{name}/comparison_summary.json",
        "html_path": f"/tmp/{name}/index.html",
        "inputs": inputs,
    }


def _metrics(precision: float, recall: float, recall75: float, small_recall: float, fp: float, det_count: float) -> dict:
    return {
        "precision@0.50_mean": precision,
        "recall@0.50_mean": recall,
        "recall@0.75_mean": recall75,
        "small_recall@0.50_mean": small_recall,
        "fp@0.50_mean": fp,
        "det_count_mean": det_count,
    }


def _calibration_summary() -> dict:
    return {
        "models": [
            {
                "feature_set": "score_label_aux",
                "feature_names": ["score", "aux_support", "edge_support", "label_seen_gt"],
                "weights": [0.4, -0.2, 0.1, 0.5],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
