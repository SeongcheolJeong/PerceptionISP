from __future__ import annotations

import unittest
from pathlib import Path

from perception_isp.resolution_sweep import parse_resolutions, summarize_run


class ResolutionSweepTest(unittest.TestCase):
    def test_parse_resolutions(self) -> None:
        self.assertEqual(parse_resolutions("640x480, 1280*960"), ((640, 480), (1280, 960)))

    def test_summarize_run_includes_raw_provenance_counts(self) -> None:
        result = {
            "run_config": {"width": 640, "height": 480},
            "aggregate": {
                "reference_rgb": {"recall@0.50_mean": 0.6, "precision@0.50_mean": 0.8},
                "perception_rgb": {"recall@0.50_mean": 0.5, "precision@0.50_mean": 0.7},
                "perception_fusion_rgb_aux": {"recall@0.50_mean": 0.55, "precision@0.50_mean": 0.75},
            },
            "samples": [
                {
                    "metadata": {
                        "raw_provenance": {
                            "true_sensor_cfa_mosaic": True,
                            "pattern_remapped": False,
                            "native_resolution_matches_target": True,
                            "native_resolution_at_least_target": True,
                            "requested_cfa_pattern": "auto",
                            "source_shape": [480, 640],
                            "source_cfa_pattern": "GRBG",
                            "target_cfa_pattern": "GRBG",
                        }
                    }
                }
            ],
        }
        summary = summarize_run(result, Path("640x480/index.html"))
        self.assertEqual(summary["resolution"], "640x480")
        self.assertEqual(summary["raw_provenance"]["true_sensor_cfa_mosaic_count"], 1)
        self.assertEqual(summary["raw_provenance"]["pattern_remapped_count"], 0)
        self.assertEqual(summary["raw_provenance"]["target_patterns"], ["GRBG"])
        self.assertEqual(summary["metrics"]["perception_rgb"]["recall@0.50"], 0.5)
        self.assertEqual(summary["metrics"]["perception_fusion_rgb_aux"]["recall@0.50"], 0.55)


if __name__ == "__main__":
    unittest.main()
