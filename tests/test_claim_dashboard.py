from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.claim_dashboard import build_claim_dashboard, main as dashboard_main, write_claim_dashboard


class ClaimDashboardTest(unittest.TestCase):
    def test_dashboard_separates_supported_and_blocked_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = _write_claim_gate(root / "broad", profile="broad_superiority", passed=False)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            training = _write_training_rollup(root / "training")
            comparison = _write_comparison_rollup(root / "rollup")

            dashboard = build_claim_dashboard(
                claim_gate_specs=[f"Human superiority={broad}", f"FP reducer={fp}"],
                training_rollup=training,
                comparison_rollup_specs=[f"Calibration={comparison}"],
            )

            self.assertEqual(len(dashboard["claims"]), 2)
            statuses = [item["status"] for item in dashboard["decisions"]]
            self.assertIn("supported", statuses)
            self.assertIn("not_supported", statuses)
            self.assertEqual(dashboard["training"]["status"], "diagnostic_only")
            self.assertEqual(dashboard["comparison_rollups"][0]["name"], "Calibration")

            html_path = write_claim_dashboard(dashboard, root / "dashboard")
            html = html_path.read_text()
            self.assertIn("PerceptionISP Claim Readiness Dashboard", html)
            self.assertIn("Broad HumanISP superiority gate failed", html)
            self.assertIn("Recall-budgeted FP-reduction gate passed", html)
            self.assertTrue((html_path.parent / "claim_dashboard_summary.json").exists())

    def test_dashboard_cli_outputs_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = dashboard_main(["--claim-gate", str(fp), "--output-dir", str(root / "dashboard")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["claim_count"], 1)
            self.assertTrue((root / "dashboard" / "claim_dashboard_summary.json").exists())


def _write_claim_gate(path: Path, *, profile: str, passed: bool) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    criteria = [
        _criterion("precision@0.50_mean", 0.03, 0.0, passed=True),
        _criterion("recall@0.50_mean", -0.01 if not passed else -0.005, 0.0 if profile == "broad_superiority" else -0.01, passed=passed),
        _criterion("small_recall@0.50_mean", -0.002, 0.0 if profile == "broad_superiority" else -0.01, passed=passed),
        _criterion("fp@0.50_mean", -0.3, 0.0 if profile == "broad_superiority" else -0.10, passed=True),
        {"metric": "sample_count", "pass": True, "target": 1000, "threshold": 1000},
    ]
    (path / "claim_gate_summary.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "verdict": "metric_gate_pass" if passed else "metric_gate_fail",
                "pass": passed,
                "sample_count": 1000,
                "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                "baseline_input": "human_rgb" if profile == "broad_superiority" else "perception_fusion_rgb_aux",
                "criteria": criteria,
                "interpretation": "unit",
            }
        )
        + "\n"
    )
    return path


def _criterion(metric: str, delta: float, threshold: float, *, passed: bool) -> dict:
    return {
        "metric": metric,
        "delta": delta,
        "threshold": threshold,
        "pass": passed,
        "paired_delta": {"ci_low": delta - 0.001, "ci_high": delta + 0.001},
    }


def _write_training_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "training_rollup_summary.json").write_text(
        json.dumps(
            {
                "run_count": 2,
                "runs": [
                    {
                        "name": "train",
                        "kind": "train_dense",
                        "sample_count": 128,
                        "epochs": 5,
                        "channel_mode": "rgb_aux",
                        "elapsed_seconds": 8.0,
                        "throughput": 60.0,
                    },
                    {
                        "name": "eval",
                        "kind": "dense_eval",
                        "sample_count": 32,
                        "channel_mode": "rgb_aux",
                        "metrics": {
                            "precision@0.50_mean": 0.01,
                            "recall@0.50_mean": 0.09,
                            "fp@0.50_mean": 40.0,
                            "det_count_mean": 41.0,
                        },
                    },
                ],
            }
        )
        + "\n"
    )
    return path


def _write_comparison_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "rollup_summary.json").write_text(json.dumps({"run_count": 1, "baseline_input": "human_rgb", "runs": []}) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
