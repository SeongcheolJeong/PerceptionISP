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
            task_metrics = _write_task_metrics(root / "task_metrics")
            comparison = _write_comparison_rollup(root / "rollup")

            dashboard = build_claim_dashboard(
                claim_gate_specs=[f"Human superiority={broad}", f"FP reducer={fp}"],
                training_rollup=training,
                task_metrics=task_metrics,
                comparison_rollup_specs=[f"Calibration={comparison}"],
            )

            self.assertEqual(len(dashboard["claims"]), 2)
            statuses = [item["status"] for item in dashboard["decisions"]]
            self.assertIn("supported", statuses)
            self.assertIn("not_supported", statuses)
            self.assertEqual(dashboard["training"]["status"], "diagnostic_only")
            self.assertEqual(dashboard["task_metrics"]["status"], "recall_tradeoff")
            self.assertEqual(dashboard["comparison_rollups"][0]["name"], "Calibration")
            self.assertIn(
                "Task-level VRU/person recall improvement versus HumanISP is not supported; the current evidence supports only the narrower FP-reduction claim.",
                [item["claim"] for item in dashboard["decisions"]],
            )

            html_path = write_claim_dashboard(dashboard, root / "dashboard")
            html = html_path.read_text()
            self.assertIn("PerceptionISP Claim Readiness Dashboard", html)
            self.assertIn("Broad HumanISP superiority gate failed", html)
            self.assertIn("Recall-budgeted FP-reduction gate passed", html)
            self.assertIn("Task Metrics", html)
            self.assertIn("recall_tradeoff", html)
            self.assertTrue((html_path.parent / "claim_dashboard_summary.json").exists())

    def test_dashboard_cli_outputs_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            task_metrics = _write_task_metrics(root / "task_metrics")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = dashboard_main(["--claim-gate", str(fp), "--task-metrics", str(task_metrics), "--output-dir", str(root / "dashboard")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["claim_count"], 1)
            summary = json.loads((root / "dashboard" / "claim_dashboard_summary.json").read_text())
            self.assertEqual(summary["task_metrics"]["status"], "recall_tradeoff")
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


def _write_task_metrics(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "task_metrics_summary.json").write_text(
        json.dumps(
            {
                "baseline_input": "human_rgb",
                "inputs": [
                    "human_rgb",
                    "perception_fusion_rgb_aux",
                    "perception_calibrated_score_label_aux_fusion_rgb_aux",
                ],
                "groups": [
                    {"name": "vru", "kind": "label"},
                    {"name": "person", "kind": "label"},
                    {"name": "vehicle", "kind": "label"},
                    {"name": "small_all", "kind": "area"},
                ],
                "label_agnostic": True,
                "metrics": {
                    "human_rgb": {
                        "vru": {"recall@0.50": 0.30, "fp@0.50_per_sample": 0.24},
                        "person": {"recall@0.50": 0.40, "fp@0.50_per_sample": 0.18},
                    },
                    "perception_calibrated_score_label_aux_fusion_rgb_aux": {
                        "vru": {
                            "gt_count": 100,
                            "det_count": 70,
                            "precision@0.50": 0.59,
                            "recall@0.50": 0.29,
                            "recall@0.75": 0.08,
                            "fp@0.50_per_sample": 0.16,
                            "delta_precision@0.50": 0.09,
                            "delta_recall@0.50": -0.01,
                            "delta_recall@0.75": -0.004,
                            "delta_fp@0.50_per_sample": -0.08,
                        },
                        "person": {
                            "gt_count": 80,
                            "det_count": 60,
                            "precision@0.50": 0.60,
                            "recall@0.50": 0.38,
                            "recall@0.75": 0.10,
                            "fp@0.50_per_sample": 0.15,
                            "delta_precision@0.50": 0.02,
                            "delta_recall@0.50": -0.02,
                            "delta_recall@0.75": -0.006,
                            "delta_fp@0.50_per_sample": -0.03,
                        },
                        "vehicle": {
                            "gt_count": 120,
                            "det_count": 100,
                            "precision@0.50": 0.72,
                            "recall@0.50": 0.50,
                            "recall@0.75": 0.33,
                            "fp@0.50_per_sample": 0.85,
                            "delta_precision@0.50": 0.03,
                            "delta_recall@0.50": -0.01,
                            "delta_recall@0.75": -0.003,
                            "delta_fp@0.50_per_sample": -0.14,
                        },
                        "small_all": {
                            "gt_count": 90,
                            "det_count": 80,
                            "precision@0.50": 0.50,
                            "recall@0.50": 0.30,
                            "recall@0.75": 0.08,
                            "fp@0.50_per_sample": 0.80,
                            "delta_precision@0.50": 0.05,
                            "delta_recall@0.50": -0.002,
                            "delta_recall@0.75": -0.001,
                            "delta_fp@0.50_per_sample": -0.07,
                        },
                    },
                },
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
