from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.report_rollup import build_rollup, write_rollup_report


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
            perception = rollup["runs"][0]["inputs"]["perception_rgb"]
            self.assertAlmostEqual(perception["delta_precision@0.50_mean"], 0.1)
            self.assertAlmostEqual(perception["delta_recall@0.50_mean"], 0.05)
            calibrated = rollup["runs"][0]["inputs"]["perception_calibrated_fusion_rgb_aux"]
            self.assertAlmostEqual(calibrated["delta_precision@0.50_mean"], 0.2)
            self.assertAlmostEqual(calibrated["delta_recall@0.50_mean"], -0.01)
            html_path = write_rollup_report(rollup, root / "rollup")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Comparison Rollup", html_path.read_text())
            self.assertTrue((html_path.parent / "rollup_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
