from __future__ import annotations

import unittest
import tempfile

import numpy as np

from perception_isp.comparison import compare_dataset, write_comparison_report
from perception_isp.detectors import fuse_rgb_aux_results
from perception_isp.eval_types import BoundingBox, Detection, DetectorResult
from perception_isp.synthetic_eval import make_synthetic_evaluation_samples


class ComparisonHarnessTest(unittest.TestCase):
    def test_synthetic_comparison_produces_default_input_metrics(self) -> None:
        samples = make_synthetic_evaluation_samples(count=1, width=96, height=64)
        result = compare_dataset(samples, fusion_options={"low_score_threshold": 0.50, "low_support_threshold": 0.10})
        self.assertEqual(result["sample_count"], 1)
        self.assertIn("human_rgb", result["aggregate"])
        self.assertIn("perception_rgb", result["aggregate"])
        self.assertIn("perception_fusion_rgb_aux", result["aggregate"])
        self.assertIn("perception_aux_rgb", result["aggregate"])
        self.assertIn("human_rgb", result["breakdown"])
        self.assertIn("perception_fusion_rgb_aux", result["breakdown"])
        self.assertIn("labels", result["breakdown"]["human_rgb"])
        self.assertIn("areas", result["breakdown"]["human_rgb"])
        self.assertIn("car", result["breakdown"]["human_rgb"]["labels"])
        self.assertEqual(len(result["samples"][0]["ground_truth"]), 3)

    def test_rgb_aux_fusion_keeps_rgb_labels_and_adds_support(self) -> None:
        aux = np.zeros((48, 64, 3), dtype=np.float64)
        aux[:, :, 2] = 0.1
        aux[10:30, 10:30, 0] = 1.0
        aux[10:30, 10:30, 2] = 1.0
        rgb_result = DetectorResult(
            "rgb",
            "perception_rgb",
            (
                Detection(BoundingBox((10, 10, 30, 30), label="car"), score=0.40),
                Detection(BoundingBox((40, 35, 50, 45), label="person"), score=0.30),
            ),
            1.0,
        )
        aux_result = DetectorResult("aux", "perception_aux_rgb", (), 1.0)
        fused = fuse_rgb_aux_results(rgb_result, aux_result, aux)
        self.assertEqual(fused.input_name, "perception_fusion_rgb_aux")
        self.assertEqual(len(fused.detections), 1)
        self.assertEqual(fused.detections[0].box.label, "car")
        self.assertIn("fusion", fused.detections[0].metadata)
        self.assertGreater(fused.detections[0].metadata["fusion"]["aux_support"], 0.3)

    def test_comparison_includes_reference_rgb_when_available(self) -> None:
        samples = list(make_synthetic_evaluation_samples(count=1, width=96, height=64))
        samples[0].reference_rgb = np.zeros((64, 96, 3), dtype=np.float64)
        result = compare_dataset(samples)
        self.assertIn("reference_rgb", result["aggregate"])
        self.assertIn("reference_rgb", result["samples"][0]["metrics"])

    def test_report_writes_visual_overlay_assets(self) -> None:
        samples = make_synthetic_evaluation_samples(count=1, width=96, height=64)
        result = compare_dataset(samples, include_images=True)
        result["run_config"] = {
            "source": "synthetic",
            "rgb_detector": "numpy_risk_object_detector",
            "fusion": True,
            "fusion_options": {"low_score_threshold": 0.50},
        }
        with tempfile.TemporaryDirectory() as tmp:
            html_path = write_comparison_report(result, tmp)
            self.assertTrue(html_path.exists())
            html_text = html_path.read_text()
            self.assertIn("Run Config", html_text)
            self.assertIn("HumanISP Delta", html_text)
            self.assertIn("rgb_detector", html_text)
            self.assertIn("fusion_options", html_text)
            assets = sorted((html_path.parent / "assets").glob("*.png"))
            self.assertGreaterEqual(len(assets), 3)
            json_text = (html_path.parent / "comparison_summary.json").read_text()
            self.assertNotIn("_visuals", json_text)
            self.assertIn("visuals", json_text)


if __name__ == "__main__":
    unittest.main()
