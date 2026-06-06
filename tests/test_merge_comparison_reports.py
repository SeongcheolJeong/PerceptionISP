from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.merge_comparison_reports import merge_comparison_reports


class MergeComparisonReportsTest(unittest.TestCase):
    def test_merges_samples_and_recomputes_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = _write_shard(root / "a", sample_id="a", recall=0.25, precision=0.50, offset=0)
            shard_b = _write_shard(root / "b", sample_id="b", recall=0.75, precision=1.00, offset=1)

            merged = merge_comparison_reports([shard_a, shard_b], name="unit")

            self.assertEqual(merged["sample_count"], 2)
            self.assertEqual(merged["run_config"]["merged"], True)
            self.assertEqual(merged["run_config"]["merged_shard_count"], 2)
            self.assertEqual(merged["run_config"]["merged_sample_count"], 2)
            metrics = merged["aggregate"]["perception_rgb"]
            self.assertAlmostEqual(metrics["recall@0.50_mean"], 0.50)
            self.assertAlmostEqual(metrics["precision@0.50_mean"], 0.75)


def _write_shard(root: Path, *, sample_id: str, recall: float, precision: float, offset: int) -> Path:
    root.mkdir()
    summary = {
        "sample_count": 1,
        "run_config": {"source": "synthetic", "count": 1, "offset": offset},
        "aggregate": {},
        "breakdown": {},
        "samples": [
            {
                "sample_id": sample_id,
                "source": "synthetic",
                "ground_truth": [],
                "metadata": {},
                "isp_metadata": {},
                "detectors": [],
                "metrics": {
                    "perception_rgb": {
                        "gt_count": 1,
                        "det_count": 1,
                        "precision@0.50": precision,
                        "recall@0.50": recall,
                        "precision@0.75": 0.0,
                        "recall@0.75": 0.0,
                        "small_gt_count": 0,
                        "small_recall@0.50": 0.0,
                    }
                },
                "breakdown": {"perception_rgb": {"labels": {}, "areas": {}}},
            }
        ],
    }
    path = root / "comparison_summary.json"
    path.write_text(json.dumps(summary) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
