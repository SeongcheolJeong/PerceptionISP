from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.proposal_calibration import (
    apply_proposal_calibration_to_report,
    build_proposal_calibration,
    main as proposal_calibration_main,
    proposal_calibration_model_artifact,
    split_sample_indices,
    write_proposal_calibration,
)
from perception_isp.proposal_calibration_apply import main as proposal_calibration_apply_main


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

    def test_write_proposal_calibration_outputs_feature_artifacts(self) -> None:
        summary = {
            "source_report": "unit",
            "input": "perception_fusion_rgb_aux",
            "train_gt_labels": ["car"],
            "train_indices": [0],
            "eval_indices": [1],
            "recall_delta_floor": -0.001,
            "models": [
                {
                    "feature_set": "score_label",
                    "feature_names": ["score"],
                    "weights": [1.0],
                    "bias": 0.0,
                    "mean": [0.0],
                    "std": [1.0],
                },
                {
                    "feature_set": "score_label_aux",
                    "feature_names": ["score", "aux_support"],
                    "weights": [1.0, 0.5],
                    "bias": 0.1,
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                },
            ],
            "rows": [
                {
                    "feature_set": "score_label",
                    "threshold": 0.02,
                    "metrics": {"precision@0.50_mean": 0.6},
                    "delta_vs_baseline": {},
                    "delta_vs_original": {"recall@0.50_mean": 0.0},
                },
                {
                    "feature_set": "score_label_aux",
                    "threshold": 0.03,
                    "metrics": {"precision@0.50_mean": 0.7},
                    "delta_vs_baseline": {},
                    "delta_vs_original": {"recall@0.50_mean": 0.0},
                },
            ],
            "best": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_proposal_calibration(
                summary,
                Path(tmp),
                artifact_feature_set="score_label",
                artifact_threshold=0.02,
                artifact_output_input="label_calibrated",
                feature_artifact_sets=("score_label", "score_label_aux"),
            )
            default_model = (Path(tmp) / "proposal_calibration_model.json").read_text()
            label_model = (Path(tmp) / "proposal_calibration_model_score_label.json").read_text()
            label_aux_model = (Path(tmp) / "proposal_calibration_model_score_label_aux.json").read_text()
            self.assertIn('"output_input": "label_calibrated"', default_model)
            self.assertIn('"output_input": "perception_calibrated_score_label_fusion_rgb_aux"', label_model)
            self.assertIn('"output_input": "perception_calibrated_score_label_aux_fusion_rgb_aux"', label_aux_model)

    def test_cli_writes_feature_artifacts(self) -> None:
        report = {
            "sample_count": 4,
            "run_config": {"label_agnostic": False},
            "samples": [_sample(index) for index in range(4)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "comparison_summary.json"
            output_dir = Path(tmp) / "calibration"
            report_path.write_text(json.dumps(report) + "\n")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = proposal_calibration_main(
                    [
                        str(report_path),
                        "--feature-sets",
                        "score_label,score_label_aux",
                        "--thresholds",
                        "0.50",
                        "--train-fraction",
                        "0.50",
                        "--split-strategy",
                        "sequential",
                        "--epochs",
                        "20",
                        "--write-feature-artifacts",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "proposal_calibration_model_score_label.json").exists())
            self.assertTrue((output_dir / "proposal_calibration_model_score_label_aux.json").exists())
            self.assertEqual(len(printed["feature_model_json"]), 2)

    def test_apply_cli_handles_multiple_models_and_rollup(self) -> None:
        report = {
            "sample_count": 4,
            "run_config": {"label_agnostic": False, "source": "unit", "count": 4},
            "samples": [_sample(index) for index in range(4)],
        }
        summary = build_proposal_calibration(
            report,
            input_name="perception_fusion_rgb_aux",
            feature_sets=("score_label", "score_label_aux"),
            thresholds=(0.50,),
            baseline_input="human_rgb",
            train_fraction=0.5,
            split_strategy="sequential",
            epochs=20,
            learning_rate=0.05,
            l2=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "comparison_summary.json"
            calibration_dir = root / "calibration"
            apply_dir = root / "applied"
            rollup_dir = root / "rollup"
            report_path.write_text(json.dumps(report) + "\n")
            write_proposal_calibration(
                summary,
                calibration_dir,
                feature_artifact_sets=("score_label", "score_label_aux"),
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = proposal_calibration_apply_main(
                    [
                        str(report_path),
                        "--model",
                        str(calibration_dir / "proposal_calibration_model_score_label.json"),
                        "--model",
                        str(calibration_dir / "proposal_calibration_model_score_label_aux.json"),
                        "--output-dir",
                        str(apply_dir),
                        "--rollup-output-dir",
                        str(rollup_dir),
                        "--include-source-report-in-rollup",
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["report_count"], 2)
            first_aggregate = printed["reports"][0]["aggregate"]
            self.assertIn("perception_calibrated_score_label_fusion_rgb_aux", first_aggregate)
            calibrated_metrics = first_aggregate["perception_calibrated_score_label_fusion_rgb_aux"]
            self.assertIn("precision@0.50_mean", calibrated_metrics)
            self.assertNotIn("precision@0.50_min", calibrated_metrics)
            self.assertTrue((apply_dir / "score_label" / "comparison_summary.json").exists())
            self.assertTrue((apply_dir / "score_label_aux" / "comparison_summary.json").exists())
            rollup = json.loads((rollup_dir / "rollup_summary.json").read_text())
            self.assertEqual(rollup["run_count"], 3)
            self.assertEqual(printed["rollup"], str(rollup_dir / "index.html"))

    def test_model_artifact_can_select_feature_set_and_threshold(self) -> None:
        summary = {
            "source_report": "unit",
            "input": "perception_fusion_rgb_aux",
            "train_gt_labels": ["car"],
            "train_indices": [0],
            "eval_indices": [1],
            "recall_delta_floor": -0.001,
            "models": [
                {
                    "feature_set": "score_label",
                    "feature_names": ["score"],
                    "weights": [1.0],
                    "bias": 0.0,
                    "mean": [0.0],
                    "std": [1.0],
                },
                {
                    "feature_set": "score_label_aux",
                    "feature_names": ["score", "aux_support"],
                    "weights": [1.0, 0.5],
                    "bias": 0.1,
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                },
            ],
            "rows": [
                {
                    "feature_set": "score_label",
                    "threshold": 0.02,
                    "metrics": {"precision@0.50_mean": 0.6},
                    "delta_vs_baseline": {},
                    "delta_vs_original": {"recall@0.50_mean": 0.0},
                },
                {
                    "feature_set": "score_label_aux",
                    "threshold": 0.03,
                    "metrics": {"precision@0.50_mean": 0.7},
                    "delta_vs_baseline": {},
                    "delta_vs_original": {"recall@0.50_mean": 0.0},
                },
            ],
            "best": {},
        }
        label_artifact = proposal_calibration_model_artifact(
            summary,
            feature_set="score_label",
            threshold=0.02,
            output_input_name="label_calibrated",
        )
        self.assertEqual(label_artifact["feature_set"], "score_label")
        self.assertEqual(label_artifact["threshold"], 0.02)
        self.assertEqual(label_artifact["output_input"], "label_calibrated")

        aux_artifact = proposal_calibration_model_artifact(summary, feature_set="score_label_aux")
        self.assertEqual(aux_artifact["feature_set"], "score_label_aux")
        self.assertEqual(aux_artifact["threshold"], 0.03)


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
