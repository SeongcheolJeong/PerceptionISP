from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.condition_gate import build_condition_gate, main as condition_gate_main, write_condition_gate


class ConditionGateTest(unittest.TestCase):
    def test_condition_gate_passes_fp_reducer_and_skips_tiny_slice(self) -> None:
        summary = build_condition_gate(
            _condition_metrics(),
            target_input="target",
            baseline_input="human_rgb",
            thresholds={"profile": "fp_reducer"},
            min_condition_samples=30,
        )
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["verdict"], "condition_gate_pass")
        self.assertEqual(summary["evaluated_condition_count"], 2)
        self.assertEqual(summary["skipped_condition_count"], 1)
        by_condition = {row["condition"]: row for row in summary["conditions"]}
        self.assertEqual(by_condition["tiny_warning"]["status"], "skipped")
        self.assertTrue(by_condition["low_light"]["pass"])

    def test_condition_gate_fails_broad_superiority_recall_loss(self) -> None:
        summary = build_condition_gate(
            _condition_metrics(),
            target_input="target",
            baseline_input="human_rgb",
            thresholds={"profile": "broad_superiority"},
            min_condition_samples=30,
        )
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["verdict"], "condition_gate_fail")
        failed = [row for row in summary["conditions"] if row["status"] == "fail"]
        self.assertEqual([row["condition"] for row in failed], ["all", "low_light"])

    def test_condition_gate_write_json_html_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "condition_metrics"
            metrics_dir.mkdir()
            (metrics_dir / "condition_metrics_summary.json").write_text(json.dumps(_condition_metrics()) + "\n")
            summary = build_condition_gate(_condition_metrics(), target_input="target", baseline_input="human_rgb")
            html_path = write_condition_gate(summary, root / "gate")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Condition Gate", html_path.read_text())
            self.assertTrue((html_path.parent / "condition_gate_summary.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = condition_gate_main(
                    [
                        str(metrics_dir),
                        "--target-input",
                        "target",
                        "--baseline-input",
                        "human_rgb",
                        "--profile",
                        "fp_reducer",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(printed["pass"])
            self.assertEqual(printed["skipped_conditions"], ["tiny_warning"])
            self.assertTrue((root / "cli" / "condition_gate_summary.json").exists())


def _condition_metrics() -> dict:
    return {
        "baseline_input": "human_rgb",
        "inputs": ["human_rgb", "target"],
        "conditions": [
            {"name": "all", "sample_count": 100},
            {"name": "low_light", "sample_count": 80},
            {"name": "tiny_warning", "sample_count": 7},
        ],
        "metrics": {
            "human_rgb": {
                "all": _row(0.60, 0.50, 0.30, 0.20, 1.20, 100),
                "low_light": _row(0.55, 0.45, 0.25, 0.18, 1.10, 80),
                "tiny_warning": _row(1.0, 1.0, 1.0, 1.0, 0.0, 7),
            },
            "target": {
                "all": _row(0.64, 0.495, 0.295, 0.195, 1.00, 100),
                "low_light": _row(0.58, 0.442, 0.245, 0.178, 0.90, 80),
                "tiny_warning": _row(1.0, 1.0, 1.0, 1.0, 0.0, 7),
            },
        },
    }


def _row(precision: float, recall: float, recall75: float, small_recall: float, fp: float, samples: int) -> dict:
    return {
        "condition_sample_count": samples,
        "precision@0.50_mean": precision,
        "recall@0.50_mean": recall,
        "recall@0.75_mean": recall75,
        "small_recall@0.50_mean": small_recall,
        "fp@0.50_mean": fp,
    }


if __name__ == "__main__":
    unittest.main()
