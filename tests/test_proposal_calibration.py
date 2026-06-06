from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perception_isp.proposal_calibration import (
    apply_proposal_calibration_to_report,
    build_proposal_calibration,
    proposal_calibration_model_artifact,
    split_sample_indices,
    write_proposal_calibration,
)


class ProposalCalibrationTest(unittest.TestCase):
    def test_split_sample_indices_keeps_train_and_eval(self) -> None:
        samples = [{"sample_id": f"sample_{index}"} for index in range(10)]
        train, eval_ = split_sample_indices(samples, train_fraction=0.7, strategy="sequential", seed="unit")
        self.assertEqual(train, tuple(range(7)))
        self.assertEqual(eval_, tuple(range(7, 10)))

    def test_build_proposal_calibration_reorders_scores_from_aux_and_label_evidence(self) -> None:
        report = {
            "sample_count": 4,
            "run_config": {"label_agnostic": False},
            "samples": [_sample(index) for index in range(4)],
        }

        summary = build_proposal_calibration(
            report,
            input_name="perception_fusion_rgb_aux",
            feature_sets=("score_label_aux",),
            thresholds=(0.50,),
            baseline_input="human_rgb",
            train_fraction=0.5,
            split_strategy="sequential",
            epochs=250,
            learning_rate=0.05,
            l2=0.0,
        )

        self.assertEqual(summary["train_sample_count"], 2)
        self.assertEqual(summary["eval_sample_count"], 2)
        self.assertEqual(summary["models"][0]["train_positive_count"], 2)
        self.assertEqual(summary["models"][0]["train_negative_count"], 2)
        row = summary["rows"][0]
        self.assertAlmostEqual(row["metrics"]["precision@0.50_mean"], 1.0)
        self.assertAlmostEqual(row["metrics"]["recall@0.50_mean"], 1.0)
        self.assertAlmostEqual(row["metrics"]["fp@0.50_mean"], 0.0)
        artifact = proposal_calibration_model_artifact(summary)
        applied = apply_proposal_calibration_to_report(report, artifact)
        self.assertIn("perception_calibrated_fusion_rgb_aux", applied["aggregate"])
        self.assertAlmostEqual(applied["aggregate"]["perception_calibrated_fusion_rgb_aux"]["precision@0.50_mean"], 1.0)
        self.assertIn("proposal_calibration", applied["run_config"])
        self.assertIn(
            "perception_calibrated_fusion_rgb_aux",
            {item["input_name"] for item in applied["samples"][0]["detectors"]},
        )
        eval_applied = apply_proposal_calibration_to_report(report, artifact, indices=artifact["eval_indices"])
        self.assertEqual(eval_applied["sample_count"], 2)
        self.assertEqual(eval_applied["run_config"]["proposal_calibration"]["applied_sample_count"], 2)

    def test_write_proposal_calibration_outputs_json_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            html_path = write_proposal_calibration(
                {
                    "source_report": "unit",
                    "input": "perception_fusion_rgb_aux",
                    "baseline_input": "human_rgb",
                    "train_sample_count": 0,
                    "eval_sample_count": 0,
                    "label_agnostic": False,
                    "baseline_metrics": {},
                    "original_input_metrics": {},
                    "models": [],
                    "best": {},
                    "rows": [],
                },
                Path(tmp),
            )
            self.assertTrue(html_path.exists())
            self.assertTrue((Path(tmp) / "proposal_calibration_summary.json").exists())
            self.assertFalse((Path(tmp) / "proposal_calibration_model.json").exists())

    def test_write_proposal_calibration_outputs_model_artifact_when_best_exists(self) -> None:
        report = {
            "sample_count": 4,
            "run_config": {"label_agnostic": False},
            "samples": [_sample(index) for index in range(4)],
        }
        summary = build_proposal_calibration(
            report,
            input_name="perception_fusion_rgb_aux",
            feature_sets=("score_label_aux",),
            thresholds=(0.50,),
            baseline_input="human_rgb",
            train_fraction=0.5,
            split_strategy="sequential",
            epochs=250,
            learning_rate=0.05,
            l2=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_proposal_calibration(summary, Path(tmp))
            model_path = Path(tmp) / "proposal_calibration_model.json"
            self.assertTrue(model_path.exists())
            self.assertIn("proposal_calibration_v1", model_path.read_text())


def _sample(index: int) -> dict:
    true_score = 0.25 + 0.02 * index
    false_score = 0.85 - 0.02 * index
    return {
        "sample_id": f"sample_{index}",
        "metadata": {"width": 100, "height": 100},
        "ground_truth": [{"xyxy": [10, 10, 30, 30], "label": "car"}],
        "detectors": [
            {
                "input_name": "human_rgb",
                "detections": [
                    {"box": {"xyxy": [10, 10, 30, 30], "label": "car"}, "score": 0.90, "metadata": {}},
                ],
            },
            {
                "input_name": "perception_fusion_rgb_aux",
                "detections": [
                    {
                        "box": {"xyxy": [10, 10, 30, 30], "label": "car"},
                        "score": true_score,
                        "metadata": {
                            "fusion": {
                                "rgb_score": true_score,
                                "aux_support": 0.90,
                                "edge_support": 0.80,
                                "saturation_support": 0.05,
                                "reliability_support": 0.95,
                                "aux_box_iou": 0.80,
                            }
                        },
                    },
                    {
                        "box": {"xyxy": [60, 60, 90, 90], "label": "bench"},
                        "score": false_score,
                        "metadata": {
                            "fusion": {
                                "rgb_score": false_score,
                                "aux_support": 0.05,
                                "edge_support": 0.02,
                                "saturation_support": 0.00,
                                "reliability_support": 0.20,
                                "aux_box_iou": 0.00,
                            }
                        },
                    },
                ],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
