from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from perception_isp.cfa_lenspsf_detector_sweep import (
    SUMMARY_FILENAME,
    build_sweep_summary,
    condition_id,
    parse_cfa_patterns,
    parse_psf_sigmas,
    raw_condition_summary,
    summarize_condition_run,
)
from perception_isp.types import PerceptionISPConfig


class CfaLensPsfDetectorSweepTest(unittest.TestCase):
    def test_parse_conditions_normalizes_and_deduplicates_values(self) -> None:
        self.assertEqual(parse_cfa_patterns(("auto", "rggb", "RGGB", "gb-rg")), ("auto", "RGGB", "GBRG"))
        self.assertEqual(parse_psf_sigmas((0.0, 1.2, -1.0, 1.2)), (0.0, 1.2))
        self.assertEqual(condition_id("GRBG", 1.2), "cfa-grbg_psf-1p20")

    def test_raw_condition_summary_counts_psf_and_cfa_remap_provenance(self) -> None:
        samples = [
            {
                "metadata": {"psf_sigma": 0.8},
                "isp_metadata": {
                    "raw_provenance": {
                        "source_cfa_pattern": "GRBG",
                        "target_cfa_pattern": "RGGB",
                        "requested_cfa_pattern": "RGGB",
                        "pattern_remapped": True,
                        "true_sensor_cfa_mosaic": True,
                        "camerae2e_camera_type": "bayer-rggb",
                        "camerae2e_native_cfa_bridge_version": "native_bayer_v1",
                        "eval_psf_sigma": 0.8,
                    }
                },
            },
            {
                "metadata": {"psf_sigma": 0.8},
                "isp_metadata": {
                    "raw_provenance": {
                        "source_cfa_pattern": "GRBG",
                        "target_cfa_pattern": "RGGB",
                        "requested_cfa_pattern": "RGGB",
                        "pattern_remapped": True,
                        "true_sensor_cfa_mosaic": True,
                        "camerae2e_camera_type": "bayer-rggb",
                        "camerae2e_native_cfa_bridge_version": "native_bayer_v1",
                        "eval_psf_sigma": 0.8,
                    }
                },
            },
        ]

        summary = raw_condition_summary(samples)

        self.assertEqual(summary["source_cfa_patterns"], {"GRBG": 2})
        self.assertEqual(summary["target_cfa_patterns"], {"RGGB": 2})
        self.assertEqual(summary["pattern_remapped_count"], 2)
        self.assertEqual(summary["true_sensor_cfa_mosaic_count"], 2)
        self.assertEqual(summary["camerae2e_camera_types"], {"bayer-rggb": 2})
        self.assertEqual(summary["camerae2e_native_cfa_bridge_versions"], {"native_bayer_v1": 2})
        self.assertEqual(summary["psf_recorded_count"], 2)
        self.assertAlmostEqual(summary["pattern_remapped_fraction"], 1.0)
        self.assertAlmostEqual(summary["true_sensor_cfa_mosaic_fraction"], 1.0)

    def test_summarize_condition_run_computes_deltas_and_raw_summary(self) -> None:
        result = _comparison_result(cfa="GRBG", psf=0.8, human_fp=1.0, perception_fp=0.7)

        run = summarize_condition_run(result, Path("001_cfa-grbg_psf-0p80/index.html"))

        self.assertEqual(run["cfa_pattern"], "GRBG")
        self.assertAlmostEqual(run["delta_vs_human"]["perception_rgb"]["fp@0.50_mean"], -0.3)
        self.assertEqual(run["raw_condition_summary"]["psf_recorded_count"], 2)

    def test_build_sweep_summary_checks_grid_and_selects_best_fp_delta(self) -> None:
        args = _args()
        runs = [
            summarize_condition_run(
                _comparison_result(cfa="GRBG", psf=0.0, human_fp=1.0, perception_fp=0.8),
                Path("001_cfa-grbg_psf-0p00/index.html"),
            ),
            summarize_condition_run(
                _comparison_result(cfa="GRBG", psf=0.8, human_fp=1.0, perception_fp=0.6),
                Path("002_cfa-grbg_psf-0p80/index.html"),
            ),
        ]

        summary = build_sweep_summary(
            args=args,
            runs=runs,
            cfa_patterns=("GRBG",),
            psf_sigmas=(0.0, 0.8),
            label_map={"pedestrian": "person"},
            config=PerceptionISPConfig(),
            human_config=PerceptionISPConfig(),
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["checks"][0]["status"], "pass")
        best = summary["best"]["perception_rgb"]["best_delta_fp@0.50"]
        self.assertEqual(best["run_id"], "cfa-grbg_psf-0p80")
        self.assertAlmostEqual(best["delta"], -0.4)
        self.assertEqual(summary["ground_truth_label_map"]["pedestrian"], "person")

    def test_write_summary_name_is_stable_for_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / SUMMARY_FILENAME
            path.write_text("{}\n")
            self.assertTrue(path.exists())


def _args() -> Namespace:
    return Namespace(
        source="yolo-dataset",
        dataset="data/kitti/data.yaml",
        split="val",
        count=2,
        offset=0,
        width=640,
        height=192,
        no_camerae2e=False,
        raw_cache_dir=None,
        label_aware=True,
        proposal_calibration_model=None,
    )


def _comparison_result(*, cfa: str, psf: float, human_fp: float, perception_fp: float) -> dict:
    return {
        "sample_count": 2,
        "run_config": {
            "condition_index": 1,
            "run_id": condition_id(cfa, psf),
            "cfa": cfa,
            "psf_sigma": psf,
            "count": 2,
        },
        "aggregate": {
            "human_rgb": {
                "precision@0.50_mean": 0.50,
                "recall@0.50_mean": 0.40,
                "small_recall@0.50_mean": 0.20,
                "fp@0.50_mean": human_fp,
                "det_count_mean": 2.0,
            },
            "perception_rgb": {
                "precision@0.50_mean": 0.55,
                "recall@0.50_mean": 0.39,
                "small_recall@0.50_mean": 0.21,
                "fp@0.50_mean": perception_fp,
                "det_count_mean": 1.7,
            },
        },
        "samples": [
            {
                "metadata": {"psf_sigma": psf},
                "isp_metadata": {
                    "raw_provenance": {
                        "source_cfa_pattern": "GRBG",
                        "target_cfa_pattern": cfa,
                        "requested_cfa_pattern": cfa,
                        "pattern_remapped": cfa != "GRBG",
                        "true_sensor_cfa_mosaic": True,
                        "camerae2e_camera_type": f"bayer-{cfa.lower()}",
                        "camerae2e_native_cfa_bridge_version": "native_bayer_v1",
                        "eval_psf_sigma": psf,
                    }
                },
            },
            {
                "metadata": {"psf_sigma": psf},
                "isp_metadata": {
                    "raw_provenance": {
                        "source_cfa_pattern": "GRBG",
                        "target_cfa_pattern": cfa,
                        "requested_cfa_pattern": cfa,
                        "pattern_remapped": cfa != "GRBG",
                        "true_sensor_cfa_mosaic": True,
                        "camerae2e_camera_type": f"bayer-{cfa.lower()}",
                        "camerae2e_native_cfa_bridge_version": "native_bayer_v1",
                        "eval_psf_sigma": psf,
                    }
                },
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
