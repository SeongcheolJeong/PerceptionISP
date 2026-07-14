from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.task_metrics import build_task_metrics, main as task_metrics_main, parse_label_groups, write_task_metrics


class TaskMetricsTest(unittest.TestCase):
    def test_task_metrics_compute_vru_micro_deltas(self) -> None:
        report = _report()
        summary = build_task_metrics(
            report,
            baseline_input="human_rgb",
            inputs=("human_rgb", "perception_calibrated_score_label_aux_fusion_rgb_aux"),
            label_groups={"vru": ("person", "bicycle")},
        )
        human = summary["metrics"]["human_rgb"]["vru"]
        target = summary["metrics"]["perception_calibrated_score_label_aux_fusion_rgb_aux"]["vru"]
        self.assertEqual(human["gt_count"], 2)
        self.assertEqual(human["tp@0.50"], 2)
        self.assertEqual(human["fp@0.50"], 0)
        self.assertEqual(target["tp@0.50"], 1)
        self.assertEqual(target["fp@0.50"], 1)
        self.assertAlmostEqual(target["recall@0.50"], 0.5)
        self.assertAlmostEqual(target["precision@0.50"], 0.5)
        self.assertAlmostEqual(target["delta_recall@0.50"], -0.5)
        self.assertAlmostEqual(target["delta_fp@0.50_per_sample"], 0.5)

    def test_task_metrics_include_small_area_group(self) -> None:
        summary = build_task_metrics(_report(), inputs=("human_rgb",), label_groups={})
        self.assertEqual([group["name"] for group in summary["groups"]], ["small_all"])
        small = summary["metrics"]["human_rgb"]["small_all"]
        self.assertEqual(small["kind"], "area")
        self.assertEqual(small["gt_count"], 2)
        self.assertEqual(small["tp@0.50"], 2)

    def test_parse_label_groups_extends_defaults(self) -> None:
        groups = parse_label_groups(["custom=foo,bar"])
        self.assertIn("vru", groups)
        self.assertEqual(groups["custom"], ("foo", "bar"))
        self.assertEqual(parse_label_groups(["custom=foo"], include_defaults=False), {"custom": ("foo",)})

    def test_task_metrics_write_json_html_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "report"
            report_dir.mkdir()
            (report_dir / "comparison_summary.json").write_text(json.dumps(_report()) + "\n")
            summary = build_task_metrics(_report(), source_report=report_dir / "comparison_summary.json")
            html_path = write_task_metrics(summary, root / "task")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Task Metrics", html_path.read_text())
            self.assertTrue((html_path.parent / "task_metrics_summary.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = task_metrics_main(
                    [
                        str(report_dir),
                        "--inputs",
                        "human_rgb,perception_calibrated_score_label_aux_fusion_rgb_aux",
                        "--group",
                        "custom=person",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["input_count"], 2)
            self.assertTrue((root / "cli" / "task_metrics_summary.json").exists())


def _report() -> dict:
    return {
        "run_config": {"label_agnostic": False},
        "sample_count": 2,
        "aggregate": {
            "human_rgb": {},
            "perception_calibrated_score_label_aux_fusion_rgb_aux": {},
        },
        "samples": [
            {
                "sample_id": "0",
                "ground_truth": [
                    {"xyxy": [10, 10, 30, 30], "label": "person"},
                    {"xyxy": [40, 40, 80, 80], "label": "car"},
                ],
                "detectors": [
                    {
                        "input_name": "human_rgb",
                        "detections": [
                            {"box": {"xyxy": [10, 10, 30, 30], "label": "person"}, "score": 0.9},
                            {"box": {"xyxy": [40, 40, 80, 80], "label": "car"}, "score": 0.9},
                        ],
                    },
                    {
                        "input_name": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                        "detections": [
                            {"box": {"xyxy": [10, 10, 30, 30], "label": "person"}, "score": 0.9},
                            {"box": {"xyxy": [0, 0, 8, 8], "label": "person"}, "score": 0.8},
                        ],
                    },
                ],
            },
            {
                "sample_id": "1",
                "ground_truth": [{"xyxy": [10, 10, 40, 40], "label": "bicycle"}],
                "detectors": [
                    {
                        "input_name": "human_rgb",
                        "detections": [{"box": {"xyxy": [10, 10, 40, 40], "label": "bicycle"}, "score": 0.9}],
                    },
                    {
                        "input_name": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                        "detections": [],
                    },
                ],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
