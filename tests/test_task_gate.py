from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.task_gate import build_task_gate, main as task_gate_main, write_task_gate


class TaskGateTest(unittest.TestCase):
    def test_task_gate_fails_recall_improvement_and_skips_empty_group(self) -> None:
        summary = build_task_gate(
            _task_metrics(),
            target_input="target",
            baseline_input="human_rgb",
            thresholds={"profile": "recall_improvement"},
            min_group_gt=1,
        )
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["verdict"], "task_gate_fail")
        self.assertEqual(summary["evaluated_group_count"], 2)
        self.assertEqual(summary["skipped_group_count"], 1)
        by_group = {row["group"]: row for row in summary["groups"]}
        self.assertEqual(by_group["traffic_light"]["status"], "skipped")
        self.assertEqual(by_group["vru"]["status"], "fail")
        failed_metrics = [item["metric"] for item in by_group["vru"]["criteria"] if not item["pass"]]
        self.assertIn("recall@0.50", failed_metrics)

    def test_task_gate_passes_fp_reducer_profile(self) -> None:
        summary = build_task_gate(
            _task_metrics(),
            target_input="target",
            baseline_input="human_rgb",
            thresholds={"profile": "fp_reducer"},
            min_group_gt=1,
        )
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["verdict"], "task_gate_pass")
        self.assertEqual(summary["failed_group_count"], 0)

    def test_task_gate_write_json_html_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "task_metrics"
            metrics_dir.mkdir()
            (metrics_dir / "task_metrics_summary.json").write_text(json.dumps(_task_metrics()) + "\n")
            summary = build_task_gate(_task_metrics(), target_input="target", baseline_input="human_rgb")
            html_path = write_task_gate(summary, root / "gate")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Task Gate", html_path.read_text())
            self.assertTrue((html_path.parent / "task_gate_summary.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = task_gate_main(
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
            self.assertEqual(printed["skipped_groups"], ["traffic_light"])
            self.assertTrue((root / "cli" / "task_gate_summary.json").exists())


def _task_metrics() -> dict:
    return {
        "baseline_input": "human_rgb",
        "inputs": ["human_rgb", "target"],
        "groups": [
            {"name": "vru", "kind": "label"},
            {"name": "person", "kind": "label"},
            {"name": "traffic_light", "kind": "label"},
        ],
        "metrics": {
            "human_rgb": {
                "vru": _row(0.50, 0.30, 0.10, 0.24, 100),
                "person": _row(0.55, 0.40, 0.15, 0.18, 80),
                "traffic_light": _row(0.0, 0.0, 0.0, 0.0, 0),
            },
            "target": {
                "vru": _row(0.56, 0.294, 0.098, 0.20, 100),
                "person": _row(0.57, 0.394, 0.148, 0.17, 80),
                "traffic_light": _row(0.0, 0.0, 0.0, 0.0, 0),
            },
        },
    }


def _row(precision: float, recall: float, recall75: float, fp: float, gt_count: int) -> dict:
    return {
        "gt_count": gt_count,
        "precision@0.50": precision,
        "recall@0.50": recall,
        "recall@0.75": recall75,
        "fp@0.50_per_sample": fp,
    }


if __name__ == "__main__":
    unittest.main()
