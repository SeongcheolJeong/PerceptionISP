from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.claim_readiness import main as readiness_main, run_claim_readiness


class ClaimReadinessTest(unittest.TestCase):
    def test_run_claim_readiness_builds_gates_training_rollup_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            train_dir = _write_train_summary(root / "train")
            eval_dir = _write_eval_summary(root / "eval")
            rollup_dir = _write_comparison_rollup(root / "rollup")

            summary = run_claim_readiness(
                comparison_report=report_dir,
                min_samples=3,
                bootstrap_samples=16,
                bootstrap_seed="unit",
                training_summaries=[train_dir, eval_dir],
                comparison_rollups=[f"Calibration={rollup_dir}"],
                output_dir=root / "readiness",
            )

            self.assertFalse(summary["broad_superiority"]["pass"])
            self.assertTrue(summary["fp_reducer"]["pass"])
            self.assertTrue((root / "readiness" / "broad_superiority_vs_human" / "claim_gate_summary.json").exists())
            self.assertTrue((root / "readiness" / "fp_reducer_vs_fusion" / "claim_gate_summary.json").exists())
            self.assertTrue((root / "readiness" / "task_metrics" / "task_metrics_summary.json").exists())
            self.assertTrue((root / "readiness" / "rgb_aux_training_rollup" / "training_rollup_summary.json").exists())
            self.assertTrue((root / "readiness" / "dashboard" / "claim_dashboard_summary.json").exists())
            self.assertIn("task_metrics", summary)
            decisions = {item["claim"]: item["status"] for item in summary["dashboard"]["decisions"]}
            self.assertEqual(decisions["Broad HumanISP superiority is not supported by the current gate evidence."], "not_supported")
            self.assertEqual(decisions["Recall-budgeted FP reduction versus the RGB+Aux fusion baseline is supported."], "supported")

    def test_claim_readiness_cli_outputs_compact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = readiness_main(
                    [
                        str(report_dir),
                        "--min-samples",
                        "3",
                        "--bootstrap-samples",
                        "16",
                        "--bootstrap-seed",
                        "unit",
                        "--output-dir",
                        str(root / "readiness"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(printed["broad_superiority"]["pass"])
            self.assertTrue(printed["fp_reducer"]["pass"])
            self.assertIn("task_metrics", printed)
            self.assertTrue((root / "readiness" / "claim_readiness_summary.json").exists())


def _write_comparison_report(path: Path) -> Path:
    path.mkdir()
    samples = []
    for index in range(3):
        samples.append(
            {
                "sample_id": str(index),
                "metrics": {
                    "human_rgb": {
                        "precision@0.50": 0.60,
                        "recall@0.50": 0.470,
                        "recall@0.75": 0.300,
                        "small_recall@0.50": 0.280,
                        "fp@0.50": 1.30,
                    },
                    "perception_fusion_rgb_aux": {
                        "precision@0.50": 0.600,
                        "recall@0.50": 0.464,
                        "recall@0.75": 0.303,
                        "small_recall@0.50": 0.279,
                        "fp@0.50": 1.30,
                    },
                    "perception_calibrated_score_label_aux_fusion_rgb_aux": {
                        "precision@0.50": 0.630,
                        "recall@0.50": 0.459,
                        "recall@0.75": 0.299,
                        "small_recall@0.50": 0.276,
                        "fp@0.50": 1.00,
                    },
                },
            }
        )
    aggregate = {
        "human_rgb": _aggregate([sample["metrics"]["human_rgb"] for sample in samples]),
        "perception_fusion_rgb_aux": _aggregate([sample["metrics"]["perception_fusion_rgb_aux"] for sample in samples]),
        "perception_calibrated_score_label_aux_fusion_rgb_aux": _aggregate(
            [sample["metrics"]["perception_calibrated_score_label_aux_fusion_rgb_aux"] for sample in samples]
        ),
    }
    payload = {"sample_count": len(samples), "samples": samples, "aggregate": aggregate}
    (path / "comparison_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _aggregate(rows: list[dict]) -> dict:
    mapping = {
        "precision@0.50_mean": "precision@0.50",
        "recall@0.50_mean": "recall@0.50",
        "recall@0.75_mean": "recall@0.75",
        "small_recall@0.50_mean": "small_recall@0.50",
        "fp@0.50_mean": "fp@0.50",
    }
    return {out_key: sum(float(row[in_key]) for row in rows) / len(rows) for out_key, in_key in mapping.items()}


def _write_train_summary(path: Path) -> Path:
    path.mkdir()
    (path / "train_dense_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 3,
                "train_sample_count": 2,
                "eval_sample_count": 1,
                "epochs": 1,
                "device": "cpu",
                "channel_mode": "rgb_aux",
                "elapsed_seconds": 0.1,
                "sample_epochs_per_second": 30.0,
            }
        )
        + "\n"
    )
    return path


def _write_eval_summary(path: Path) -> Path:
    path.mkdir()
    (path / "dense_eval_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 1,
                "checkpoint_summary": {"channel_mode": "rgb_aux"},
                "aggregate": {
                    "precision@0.50_mean": 0.01,
                    "recall@0.50_mean": 0.09,
                    "fp@0.50_mean": 40.0,
                    "det_count_mean": 41.0,
                },
            }
        )
        + "\n"
    )
    return path


def _write_comparison_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "rollup_summary.json").write_text(json.dumps({"run_count": 1, "baseline_input": "human_rgb", "runs": []}) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
