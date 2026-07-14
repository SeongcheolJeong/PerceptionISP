from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.reporting.report_rollup import build_rollup, write_rollup_report


class ReportRollupTest(unittest.TestCase):
    def test_rollup_computes_human_deltas_and_writes_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "run_a"
            report_dir.mkdir()
            (report_dir / "index.html").write_text("<html></html>")
            (report_dir / "comparison_summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 2,
                        "run_config": {"source": "synthetic", "count": 2},
                        "aggregate": {
                            "human_rgb": {
                                "precision@0.50_mean": 0.5,
                                "recall@0.50_mean": 0.4,
                                "recall@0.75_mean": 0.3,
                                "small_recall@0.50_mean": 0.2,
                                "fp@0.50_mean": 1.0,
                                "det_count_mean": 2.0,
                            },
                            "perception_rgb": {
                                "precision@0.50_mean": 0.6,
                                "recall@0.50_mean": 0.45,
                                "recall@0.75_mean": 0.35,
                                "small_recall@0.50_mean": 0.25,
                                "fp@0.50_mean": 0.8,
                                "det_count_mean": 2.1,
                            },
                            "perception_calibrated_fusion_rgb_aux": {
                                "precision@0.50_mean": 0.7,
                                "recall@0.50_mean": 0.39,
                                "recall@0.75_mean": 0.29,
                                "small_recall@0.50_mean": 0.21,
                                "fp@0.50_mean": 0.6,
                                "det_count_mean": 1.8,
                            },
                        },
                    }
                )
                + "\n"
            )
            rollup = build_rollup([report_dir])
            self.assertEqual(rollup["run_count"], 1)
            self.assertEqual(rollup["runs"][0]["name"], "synthetic 2")
            perception = rollup["runs"][0]["inputs"]["perception_rgb"]
            self.assertAlmostEqual(perception["delta_precision@0.50_mean"], 0.1)
            self.assertAlmostEqual(perception["delta_recall@0.50_mean"], 0.05)
            self.assertAlmostEqual(perception["delta_fp@0.50_mean"], -0.2)
            self.assertAlmostEqual(perception["delta_det_count_mean"], 0.1)
            calibrated = rollup["runs"][0]["inputs"]["perception_calibrated_fusion_rgb_aux"]
            self.assertAlmostEqual(calibrated["delta_precision@0.50_mean"], 0.2)
            self.assertAlmostEqual(calibrated["delta_recall@0.50_mean"], -0.01)
            html_path = write_rollup_report(rollup, root / "rollup")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Comparison Rollup", html_path.read_text())
            self.assertIn("human_rgb", html_path.read_text())
            self.assertIn("dFP@0.50", html_path.read_text())
            self.assertTrue((html_path.parent / "rollup_summary.json").exists())

    def test_rollup_can_use_custom_baseline_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "run_a"
            report_dir.mkdir()
            (report_dir / "comparison_summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 2,
                        "run_config": {"source": "synthetic", "count": 2},
                        "aggregate": {
                            "human_rgb": {"precision@0.50_mean": 0.5, "recall@0.50_mean": 0.4, "small_recall@0.50_mean": 0.2},
                            "perception_fusion_rgb_aux": {
                                "precision@0.50_mean": 0.6,
                                "recall@0.50_mean": 0.45,
                                "small_recall@0.50_mean": 0.25,
                                "fp@0.50_mean": 1.2,
                                "det_count_mean": 3.0,
                            },
                            "perception_calibrated_fusion_rgb_aux": {
                                "precision@0.50_mean": 0.7,
                                "recall@0.50_mean": 0.43,
                                "small_recall@0.50_mean": 0.24,
                                "fp@0.50_mean": 0.8,
                                "det_count_mean": 2.5,
                            },
                        },
                    }
                )
                + "\n"
            )
            rollup = build_rollup([report_dir], baseline_input="perception_fusion_rgb_aux")
            self.assertEqual(rollup["baseline_input"], "perception_fusion_rgb_aux")
            baseline = rollup["runs"][0]["inputs"]["perception_fusion_rgb_aux"]
            self.assertNotIn("delta_precision@0.50_mean", baseline)
            calibrated = rollup["runs"][0]["inputs"]["perception_calibrated_fusion_rgb_aux"]
            self.assertAlmostEqual(calibrated["delta_precision@0.50_mean"], 0.1)
            self.assertAlmostEqual(calibrated["delta_recall@0.50_mean"], -0.02)
            self.assertAlmostEqual(calibrated["delta_fp@0.50_mean"], -0.4)
            self.assertAlmostEqual(calibrated["delta_det_count_mean"], -0.5)

    def test_rollup_names_proposal_calibration_feature_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "score_label_aux"
            report_dir.mkdir()
            (report_dir / "comparison_summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 2,
                        "run_config": {
                            "source": "synthetic",
                            "count": 2,
                            "proposal_calibration": {
                                "feature_set": "score_label_aux",
                                "output_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                            },
                        },
                        "aggregate": {
                            "human_rgb": {"precision@0.50_mean": 0.5},
                            "perception_calibrated_score_label_aux_fusion_rgb_aux": {"precision@0.50_mean": 0.7},
                        },
                    }
                )
                + "\n"
            )
            rollup = build_rollup([report_dir])
            self.assertEqual(rollup["runs"][0]["name"], "synthetic 2 - calibrated score_label_aux")

    def test_rollup_disambiguates_duplicate_run_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for dirname in ("source_harness", "score_label_aux"):
                report_dir = root / dirname
                report_dir.mkdir()
                paths.append(report_dir)
                (report_dir / "comparison_summary.json").write_text(
                    json.dumps(
                        {
                            "sample_count": 2,
                            "run_config": {
                                "source": "synthetic",
                                "count": 2,
                                "proposal_calibration": {"feature_set": "score_label_aux"},
                            },
                            "aggregate": {"human_rgb": {"precision@0.50_mean": 0.5}},
                        }
                    )
                    + "\n"
                )
            rollup = build_rollup(paths)
            names = [run["name"] for run in rollup["runs"]]
            self.assertEqual(
                names,
                [
                    "synthetic 2 - calibrated score_label_aux (source_harness)",
                    "synthetic 2 - calibrated score_label_aux (score_label_aux)",
                ],
            )


if __name__ == "__main__":
    unittest.main()
