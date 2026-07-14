from __future__ import annotations

import unittest
import tempfile
from contextlib import redirect_stderr
from io import StringIO

import numpy as np

from perception_isp.evaluation.comparison import build_pipeline_images, compare_dataset, write_comparison_report
from perception_isp.core.detectors import DetectorAdapter, LabelMapDetector, fuse_rgb_aux_results
from perception_isp.core.task_types import BoundingBox, Detection, DetectorResult
from perception_isp.evaluation.synthetic_eval import make_synthetic_evaluation_samples
from perception_isp.core.types import PerceptionISPConfig


class ComparisonHarnessTest(unittest.TestCase):
    def test_label_map_detector_remaps_output_labels(self) -> None:
        detector = LabelMapDetector(_StaticDetector("pedestrian"), {"pedestrian": "person"})

        result = detector.detect(np.zeros((4, 4, 3), dtype=np.float64), input_name="perception_rgb_aux_dnn")

        self.assertEqual(result.input_name, "perception_rgb_aux_dnn")
        self.assertEqual(result.detections[0].box.label, "person")
        self.assertEqual(result.detections[0].metadata["original_label"], "pedestrian")
        self.assertTrue(result.detections[0].metadata["label_mapped"])

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

    def test_rgb_aux_fusion_can_refine_boxes_to_aux_edges(self) -> None:
        aux = np.zeros((48, 64, 3), dtype=np.float64)
        aux[:, :, 2] = 1.0
        aux[10:30, 10, 0] = 1.0
        aux[10:30, 30, 0] = 1.0
        aux[10, 10:31, 0] = 1.0
        aux[30, 10:31, 0] = 1.0
        rgb_result = DetectorResult(
            "rgb",
            "perception_rgb",
            (Detection(BoundingBox((12, 12, 28, 28), label="car"), score=0.80),),
            1.0,
        )
        aux_result = DetectorResult("aux", "perception_aux_rgb", (), 1.0)

        fused = fuse_rgb_aux_results(
            rgb_result,
            aux_result,
            aux,
            refine_boxes=True,
            refine_max_shift=0.25,
            refine_max_shift_px=4.0,
            refine_min_edge=0.20,
            refine_min_gain=0.05,
        )

        self.assertEqual(len(fused.detections), 1)
        refined = fused.detections[0].box.xyxy
        self.assertLess(refined[0], 12.0)
        self.assertLess(refined[1], 12.0)
        self.assertGreater(refined[2], 28.0)
        self.assertGreater(refined[3], 28.0)
        metadata = fused.detections[0].metadata["fusion"]["box_refinement"]
        self.assertTrue(metadata["enabled"])
        self.assertTrue(metadata["changed"])

    def test_comparison_can_apply_proposal_calibration_artifact(self) -> None:
        artifact = {
            "model_type": "proposal_calibration_v1",
            "input": "perception_fusion_rgb_aux",
            "output_input": "perception_calibrated_fusion_rgb_aux",
            "threshold": 0.0,
            "feature_set": "score",
            "feature_names": ["score"],
            "train_gt_labels": ["car", "person", "traffic_light"],
            "model": {
                "feature_set": "score",
                "feature_names": ["score"],
                "weights": [0.0],
                "bias": 0.0,
                "mean": [0.0],
                "std": [1.0],
            },
        }
        samples = make_synthetic_evaluation_samples(count=1, width=96, height=64)
        result = compare_dataset(samples, proposal_calibration_artifact=artifact)
        self.assertIn("perception_calibrated_fusion_rgb_aux", result["aggregate"])
        self.assertIn("perception_calibrated_fusion_rgb_aux", result["samples"][0]["metrics"])
        detector_names = {item["input_name"] for item in result["samples"][0]["detectors"]}
        self.assertIn("perception_calibrated_fusion_rgb_aux", detector_names)

    def test_proposal_calibration_requires_fusion(self) -> None:
        artifact = {
            "model_type": "proposal_calibration_v1",
            "input": "perception_fusion_rgb_aux",
            "output_input": "perception_calibrated_fusion_rgb_aux",
            "threshold": 0.0,
            "model": {"feature_names": [], "weights": [], "bias": 0.0, "mean": [], "std": []},
        }
        samples = make_synthetic_evaluation_samples(count=1, width=96, height=64)
        with self.assertRaises(ValueError):
            compare_dataset(samples, include_fusion=False, proposal_calibration_artifact=artifact)

    def test_comparison_includes_reference_rgb_when_available(self) -> None:
        samples = list(make_synthetic_evaluation_samples(count=1, width=96, height=64))
        samples[0].reference_rgb = np.zeros((64, 96, 3), dtype=np.float64)
        result = compare_dataset(samples)
        self.assertIn("reference_rgb", result["aggregate"])
        self.assertIn("reference_rgb", result["samples"][0]["metrics"])

    def test_build_pipeline_images_can_keep_human_baseline_config_separate(self) -> None:
        sample = make_synthetic_evaluation_samples(count=1, width=96, height=64)[0]
        images = build_pipeline_images(
            sample,
            config=PerceptionISPConfig(tone_mapping="linear", denoise_strength=0.0),
            human_config=PerceptionISPConfig(tone_mapping="log", denoise_strength=0.18),
        )
        self.assertEqual(images.metadata["processing"]["tone_mapping"], "linear")
        self.assertEqual(images.human_metadata["processing"]["tone_mapping"], "log")

    def test_progress_interval_writes_status_to_stderr(self) -> None:
        samples = make_synthetic_evaluation_samples(count=2, width=48, height=32)
        stream = StringIO()
        with redirect_stderr(stream):
            compare_dataset(samples, progress_interval=1, progress_label="unit")
        text = stream.getvalue()
        self.assertIn("[unit] 1/2 samples", text)
        self.assertIn("[unit] 2/2 samples", text)

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


class _StaticDetector(DetectorAdapter):
    name = "static"

    def __init__(self, label: str) -> None:
        self.label = label

    def detect(self, image, *, input_name: str = "image") -> DetectorResult:
        return DetectorResult(
            self.name,
            input_name,
            (Detection(BoundingBox((0.0, 0.0, 2.0, 2.0), label=self.label), score=0.5),),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
