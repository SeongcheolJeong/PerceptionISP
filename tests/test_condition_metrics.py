from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.condition_metrics import build_condition_metrics, main as condition_metrics_main, write_condition_metrics


class ConditionMetricsTest(unittest.TestCase):
    def test_condition_metrics_compute_explicit_and_derived_slices(self) -> None:
        summary = build_condition_metrics(
            _report(),
            baseline_input="human_rgb",
            inputs=("human_rgb", "perception_calibrated_score_label_aux_fusion_rgb_aux"),
        )
        condition_names = {item["name"] for item in summary["conditions"]}
        self.assertIn("all", condition_names)
        self.assertIn("weather:rain", condition_names)
        self.assertIn("lighting:night", condition_names)
        self.assertIn("low_light_proxy", condition_names)
        self.assertIn("true_cfa_mosaic", condition_names)

        target = summary["metrics"]["perception_calibrated_score_label_aux_fusion_rgb_aux"]["weather:rain"]
        self.assertEqual(target["condition_sample_count"], 1)
        self.assertAlmostEqual(target["recall@0.50_mean"], 0.0)
        self.assertAlmostEqual(target["delta_recall@0.50_mean"], -1.0)
        self.assertAlmostEqual(target["delta_fp@0.50_mean"], 1.0)

    def test_condition_metrics_write_json_html_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "report"
            report_dir.mkdir()
            (report_dir / "comparison_summary.json").write_text(json.dumps(_report()) + "\n")
            summary = build_condition_metrics(_report(), source_report=report_dir / "comparison_summary.json")
            html_path = write_condition_metrics(summary, root / "condition")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Condition Metrics", html_path.read_text())
            self.assertTrue((html_path.parent / "condition_metrics_summary.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = condition_metrics_main(
                    [
                        str(report_dir),
                        "--inputs",
                        "human_rgb,perception_calibrated_score_label_aux_fusion_rgb_aux",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["input_count"], 2)
            self.assertGreaterEqual(printed["condition_count"], 4)
            self.assertTrue((root / "cli" / "condition_metrics_summary.json").exists())


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
                "sample_id": "rain_night",
                "metadata": {
                    "weather": "rain",
                    "lighting": "night",
                    "raw_provenance": {"true_sensor_cfa_mosaic": True, "camerae2e_used": True},
                },
                "isp_metadata": {
                    "health": {
                        "visibility_confidence": 0.40,
                        "under_exposure_fraction": 0.35,
                        "over_exposure_fraction": 0.0,
                        "focus_confidence": 0.80,
                        "warnings": [],
                    }
                },
                "ground_truth": [{"xyxy": [10, 10, 30, 30], "label": "person"}],
                "detectors": [
                    {
                        "input_name": "human_rgb",
                        "detections": [{"box": {"xyxy": [10, 10, 30, 30], "label": "person"}, "score": 0.9}],
                    },
                    {
                        "input_name": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                        "detections": [{"box": {"xyxy": [40, 40, 60, 60], "label": "person"}, "score": 0.8}],
                    },
                ],
            },
            {
                "sample_id": "clear_day",
                "metadata": {"weather": "clear", "lighting": "day"},
                "isp_metadata": {
                    "health": {
                        "visibility_confidence": 0.90,
                        "under_exposure_fraction": 0.0,
                        "over_exposure_fraction": 0.0,
                        "focus_confidence": 0.90,
                        "warnings": [],
                    }
                },
                "ground_truth": [{"xyxy": [50, 50, 80, 80], "label": "car"}],
                "detectors": [
                    {
                        "input_name": "human_rgb",
                        "detections": [{"box": {"xyxy": [50, 50, 80, 80], "label": "car"}, "score": 0.9}],
                    },
                    {
                        "input_name": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                        "detections": [{"box": {"xyxy": [50, 50, 80, 80], "label": "car"}, "score": 0.9}],
                    },
                ],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
