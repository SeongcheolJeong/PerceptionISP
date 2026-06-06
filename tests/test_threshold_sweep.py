from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perception_isp.threshold_sweep import build_threshold_sweep, parse_thresholds, write_threshold_sweep


class ThresholdSweepTest(unittest.TestCase):
    def test_parse_threshold_range_and_list(self) -> None:
        self.assertEqual(parse_thresholds("0.25:0.30:0.025"), (0.25, 0.275, 0.3))
        self.assertEqual(parse_thresholds("0.25,0.5"), (0.25, 0.5))

    def test_build_threshold_sweep_filters_saved_detections(self) -> None:
        report = {
            "sample_count": 1,
            "run_config": {"label_agnostic": False},
            "aggregate": {
                "human_rgb": {
                    "precision@0.50_mean": 1.0,
                    "recall@0.50_mean": 1.0,
                    "recall@0.75_mean": 1.0,
                    "small_recall@0.50_mean": 1.0,
                    "fp@0.50_mean": 0.0,
                    "det_count_mean": 1.0,
                }
            },
            "samples": [
                {
                    "ground_truth": [{"xyxy": [0, 0, 10, 10], "label": "car"}],
                    "detectors": [
                        {
                            "input_name": "perception_rgb",
                            "detections": [
                                {"box": {"xyxy": [0, 0, 10, 10], "label": "car"}, "score": 0.40, "metadata": {}},
                                {"box": {"xyxy": [20, 20, 30, 30], "label": "car"}, "score": 0.30, "metadata": {}},
                            ],
                        }
                    ],
                }
            ],
        }

        summary = build_threshold_sweep(
            report,
            inputs=("perception_rgb",),
            thresholds=(0.25, 0.35),
            baseline_input="human_rgb",
            recall_delta_floor=0.0,
        )

        self.assertEqual(len(summary["rows"]), 2)
        self.assertAlmostEqual(summary["rows"][0]["metrics"]["precision@0.50_mean"], 0.5)
        self.assertAlmostEqual(summary["rows"][1]["metrics"]["precision@0.50_mean"], 1.0)
        self.assertEqual(summary["best"]["max_precision_with_recall_floor"]["threshold"], 0.35)

    def test_write_threshold_sweep_outputs_json_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            html_path = write_threshold_sweep(
                {
                    "sample_count": 0,
                    "source_report": "unit",
                    "baseline_input": "human_rgb",
                    "recall_delta_floor": 0.0,
                    "best": {},
                    "rows": [],
                },
                Path(tmp),
            )
            self.assertTrue(html_path.exists())
            self.assertTrue((Path(tmp) / "threshold_sweep_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
