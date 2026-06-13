from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from perception_isp.adverse_native_slice import (
    SUMMARY_FILENAME,
    apply_adverse_condition,
    build_adverse_summary,
    condition_id,
    condition_samples,
    parse_conditions,
    summarize_condition_run,
)
from perception_isp.camerae2e_bridge import raw_from_rgb_direct
from perception_isp.eval_types import BoundingBox, EvaluationSample
from perception_isp.types import PerceptionISPConfig


class AdverseNativeSliceTest(unittest.TestCase):
    def test_parse_conditions_normalizes_and_rejects_unknown(self) -> None:
        self.assertEqual(parse_conditions(("Nominal", "low-light", "low_light", "fog")), ("nominal", "low_light", "fog"))
        self.assertEqual(condition_id("low_mtf"), "condition-low-mtf")
        with self.assertRaises(ValueError):
            parse_conditions(("nominal", "unsupported"))

    def test_adverse_transforms_move_scene_statistics(self) -> None:
        rgb = _gradient_rgb()

        night = apply_adverse_condition(rgb, "night", seed=1)
        fog = apply_adverse_condition(rgb, "fog", seed=1)
        glare = apply_adverse_condition(rgb, "glare", seed=1)
        low_mtf = apply_adverse_condition(rgb, "low_mtf", seed=1)

        self.assertLess(float(np.mean(night)), float(np.mean(rgb)))
        self.assertGreater(float(np.mean(fog)), float(np.mean(rgb)) - 0.02)
        self.assertGreater(float(np.max(glare)), float(np.max(rgb)) - 1.0e-6)
        self.assertEqual(low_mtf.shape, rgb.shape)

    def test_condition_samples_preserve_labels_and_record_metadata(self) -> None:
        sample = _sample()

        conditioned = condition_samples(
            (sample,),
            condition="fog",
            cfa_pattern="RGGB",
            width=32,
            height=18,
            use_camerae2e=False,
        )

        self.assertEqual(len(conditioned), 1)
        self.assertEqual(conditioned[0].sample_id, "sample_000_fog")
        self.assertEqual(conditioned[0].ground_truth[0].label, "car")
        self.assertEqual(conditioned[0].metadata["adverse_condition"], "fog")
        self.assertEqual(conditioned[0].raw.metadata.cfa_pattern, "RGGB")

    def test_build_summary_records_adverse_tradeoff(self) -> None:
        args = _args()
        runs = [
            summarize_condition_run(
                _comparison_result(condition="nominal", human_fp=1.0, primary_fp=0.9, primary_recall=0.40),
                Path("001_condition-nominal/index.html"),
            ),
            summarize_condition_run(
                _comparison_result(condition="fog", human_fp=1.0, primary_fp=0.7, primary_recall=0.39),
                Path("002_condition-fog/index.html"),
            ),
            summarize_condition_run(
                _comparison_result(condition="glare", human_fp=1.0, primary_fp=1.2, primary_recall=0.42),
                Path("003_condition-glare/index.html"),
            ),
        ]

        summary = build_adverse_summary(
            args=args,
            runs=runs,
            conditions=("nominal", "fog", "glare"),
            label_map={"pedestrian": "person"},
            config=PerceptionISPConfig(),
            human_config=PerceptionISPConfig(),
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["claim_status"], "adverse_fp_recall_tradeoff")
        self.assertEqual(summary["aggregate"]["adverse_fp_win_count"], 1)
        self.assertEqual(summary["aggregate"]["adverse_condition_count"], 2)
        self.assertEqual(summary["checks"][1]["status"], "pass")
        self.assertEqual(summary["ground_truth_label_map"]["pedestrian"], "person")

    def test_build_summary_uses_explicit_dnn_primary_input(self) -> None:
        args = _args()
        runs = [
            summarize_condition_run(
                _comparison_result(
                    condition="nominal",
                    human_fp=1.0,
                    primary_fp=0.9,
                    primary_recall=0.40,
                    dnn_fp=0.8,
                    dnn_recall=0.41,
                ),
                Path("001_condition-nominal/index.html"),
            ),
            summarize_condition_run(
                _comparison_result(
                    condition="fog",
                    human_fp=1.0,
                    primary_fp=1.3,
                    primary_recall=0.42,
                    dnn_fp=0.7,
                    dnn_recall=0.40,
                ),
                Path("002_condition-fog/index.html"),
            ),
        ]

        summary = build_adverse_summary(
            args=args,
            runs=runs,
            conditions=("nominal", "fog"),
            label_map={},
            config=PerceptionISPConfig(),
            human_config=PerceptionISPConfig(),
            primary_input="perception_rgb_aux_dnn",
        )

        self.assertEqual(summary["primary_input"], "perception_rgb_aux_dnn")
        self.assertEqual(summary["aggregate"]["primary_rows"][0]["input"], "perception_rgb_aux_dnn")
        self.assertEqual(summary["aggregate"]["adverse_fp_win_count"], 1)
        self.assertEqual(summary["claim_status"], "adverse_fp_reducer_supported")
        self.assertEqual(
            [row for row in summary["checks"] if row["id"] == "primary_input_metrics_available"][0]["status"],
            "pass",
        )

    def test_explicit_primary_input_must_exist(self) -> None:
        args = _args()
        runs = [
            summarize_condition_run(
                _comparison_result(condition="nominal", human_fp=1.0, primary_fp=0.9, primary_recall=0.40),
                Path("001_condition-nominal/index.html"),
            ),
        ]

        with self.assertRaisesRegex(ValueError, "primary input"):
            build_adverse_summary(
                args=args,
                runs=runs,
                conditions=("nominal",),
                label_map={},
                config=PerceptionISPConfig(),
                human_config=PerceptionISPConfig(),
                primary_input="perception_rgb_aux_dnn",
            )

    def test_summary_filename_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / SUMMARY_FILENAME
            path.write_text("{}\n")
            self.assertTrue(path.exists())


def _gradient_rgb() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 18, dtype=np.float64)[:, None]
    red = np.broadcast_to(x[None, :], (18, 32))
    green = np.broadcast_to(y, (18, 32))
    blue = 1.0 - red * 0.5
    return np.stack([red, green, blue], axis=2)


def _sample() -> EvaluationSample:
    rgb = _gradient_rgb()
    raw = raw_from_rgb_direct(rgb, width=32, height=18, cfa_pattern="RGGB")
    return EvaluationSample(
        sample_id="sample_000",
        raw=raw,
        ground_truth=(BoundingBox((4.0, 3.0, 18.0, 12.0), label="car"),),
        source="unit",
        metadata={"image_path": "sample.png", "label_path": "sample.txt", "dataset_index": 0},
        reference_rgb=rgb,
    )


def _args() -> Namespace:
    return Namespace(
        source="yolo-dataset",
        dataset="data/kitti/data.yaml",
        split="val",
        count=2,
        offset=0,
        width=32,
        height=18,
        cfa="GRBG",
        severity=1.0,
        psf_sigma=0.0,
        no_camerae2e=False,
        raw_cache_dir=None,
        condition_raw_cache_dir=None,
        label_aware=True,
        proposal_calibration_model=None,
    )


def _comparison_result(
    *,
    condition: str,
    human_fp: float,
    primary_fp: float,
    primary_recall: float,
    dnn_fp: float | None = None,
    dnn_recall: float | None = None,
) -> dict:
    aggregate = {
        "human_rgb": {
            "precision@0.50_mean": 0.50,
            "recall@0.50_mean": 0.40,
            "small_recall@0.50_mean": 0.20,
            "fp@0.50_mean": human_fp,
            "det_count_mean": 2.0,
        },
        "perception_calibrated_score_label_aux_fusion_rgb_aux": {
            "precision@0.50_mean": 0.55,
            "recall@0.50_mean": primary_recall,
            "small_recall@0.50_mean": 0.20,
            "fp@0.50_mean": primary_fp,
            "det_count_mean": 1.7,
        },
    }
    if dnn_fp is not None and dnn_recall is not None:
        aggregate["perception_rgb_aux_dnn"] = {
            "precision@0.50_mean": 0.57,
            "recall@0.50_mean": float(dnn_recall),
            "small_recall@0.50_mean": 0.21,
            "fp@0.50_mean": float(dnn_fp),
            "det_count_mean": 1.6,
        }
    return {
        "sample_count": 2,
        "run_config": {
            "condition_index": 1,
            "run_id": condition_id(condition),
            "adverse_condition": condition,
            "count": 2,
        },
        "aggregate": aggregate,
        "samples": [
            _sample_summary(condition),
            _sample_summary(condition),
        ],
    }


def _sample_summary(condition: str) -> dict:
    return {
        "metadata": {"adverse_condition": condition, "psf_sigma": 0.0},
        "isp_metadata": {
            "raw_provenance": {
                "source_cfa_pattern": "GRBG",
                "target_cfa_pattern": "GRBG",
                "requested_cfa_pattern": "GRBG",
                "pattern_remapped": False,
                "true_sensor_cfa_mosaic": True,
                "camerae2e_camera_type": "bayer-grbg",
                "camerae2e_native_cfa_bridge_version": "native_bayer_v1",
                "eval_psf_sigma": 0.0,
                "adverse_condition": condition,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
