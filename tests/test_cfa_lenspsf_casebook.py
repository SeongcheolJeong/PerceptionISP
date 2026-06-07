from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.cfa_lenspsf_casebook import build_cfa_lenspsf_casebook_from_path, write_cfa_lenspsf_casebook


class CfaLensPsfCasebookTest(unittest.TestCase):
    def test_builds_visual_casebook_rollup_across_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = _write_sweep_fixture(root / "sweep")

            summary = build_cfa_lenspsf_casebook_from_path(
                sweep,
                max_cases_per_category=1,
                max_showcase_cases=4,
            )

            self.assertEqual(summary["condition_count"], 2)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["category_totals"]["fp_reduction_success"]["selected_case_count"], 1)
            self.assertEqual(summary["category_totals"]["recall_tradeoff"]["selected_case_count"], 1)
            self.assertEqual(summary["selected_case_count"], 2)
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["casebook_uses_native_cfa_rows"], "pass")
            self.assertEqual(checks["casebook_separates_simulated_native_rows"], "pass")
            self.assertEqual(checks["casebook_includes_counterexamples"], "pass")

            html_path = write_cfa_lenspsf_casebook(summary, root / "casebook")
            written = json.loads((html_path.parent / "cfa_lenspsf_casebook_summary.json").read_text())
            self.assertEqual(written["status"], "pass")
            self.assertEqual(len(written["showcase_cases"]), 2)
            self.assertTrue(Path(written["showcase_cases"][0]["visual_path"]).exists())
            self.assertIn("CFA/LensPSF Visual Casebook", html_path.read_text())

    def test_simulated_native_casebook_is_warning_not_true_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = _write_sweep_fixture(root / "sweep", raw_derived=True)

            summary = build_cfa_lenspsf_casebook_from_path(
                sweep,
                max_cases_per_category=1,
                max_showcase_cases=4,
            )

            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(summary["status"], "warning")
            self.assertEqual(checks["casebook_uses_native_cfa_rows"], "fail")
            self.assertEqual(checks["casebook_separates_simulated_native_rows"], "warning")


def _write_sweep_fixture(path: Path, *, raw_derived: bool = False) -> Path:
    path.mkdir()
    image_path = path / "scene.png"
    image = np.zeros((40, 48, 3), dtype=np.uint8)
    image[8:30, 8:30, :] = 180
    Image.fromarray(image).save(image_path)
    runs = []
    for index, (run_id, cfa, psf, mode) in enumerate(
        (
            ("cfa-grbg_psf-0p00", "GRBG", 0.0, "success"),
            ("cfa-rggb_psf-0p80", "RGGB", 0.8, "tradeoff"),
        ),
        start=1,
    ):
        condition_dir = path / f"{index:03d}_{run_id}"
        condition_dir.mkdir()
        (condition_dir / "index.html").write_text("<html></html>")
        (condition_dir / "comparison_summary.json").write_text(json.dumps(_comparison_payload(image_path, mode=mode)) + "\n")
        runs.append(
            {
                "run_id": run_id,
                "report": f"{index:03d}_{run_id}/index.html",
                "cfa_pattern": cfa,
                "psf_sigma": psf,
                "raw_condition_summary": {
                    "pattern_remapped_fraction": 0.0,
                    "true_sensor_cfa_mosaic_fraction": 1.0,
                    "native_raw_input_fraction": 0.0 if raw_derived else 1.0,
                    "raw_derived_png_input_fraction": 1.0 if raw_derived else 0.0,
                    "camerae2e_used_fraction": 1.0 if raw_derived else 0.0,
                },
                "metrics": {
                    "human_rgb": {},
                    "perception_fusion_rgb_aux": {},
                    "perception_calibrated_score_label_aux_fusion_rgb_aux_t001": {},
                },
            }
        )
    sweep_summary = {
        "run_count": 2,
        "cfa_patterns": ["GRBG", "RGGB"],
        "psf_sigmas": [0.0, 0.8],
        "runs": runs,
    }
    (path / "cfa_lenspsf_detector_sweep_summary.json").write_text(json.dumps(sweep_summary) + "\n")
    return path


def _comparison_payload(image_path: Path, *, mode: str) -> dict:
    baseline = "perception_fusion_rgb_aux"
    target = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001"
    tp = _detection((8, 8, 28, 28), score=0.9)
    fp = _detection((32, 6, 44, 18), score=0.4)
    target_detections = [tp] if mode == "success" else []
    target_metrics = {"tp@0.50": 1, "fp@0.50": 0, "fn@0.50": 0} if mode == "success" else {"tp@0.50": 0, "fp@0.50": 0, "fn@0.50": 1}
    return {
        "run_config": {"label_agnostic": False},
        "sample_count": 1,
        "samples": [
            {
                "sample_id": mode,
                "source": "unit",
                "metadata": {"image_path": str(image_path), "width": 48, "height": 40, "cfa_pattern": "GRBG", "raw_provenance": {"pattern_remapped": False, "true_sensor_cfa_mosaic": True}},
                "ground_truth": [{"xyxy": [8, 8, 28, 28], "label": "object"}],
                "detectors": [
                    {"input_name": baseline, "detections": [tp, fp]},
                    {"input_name": target, "detections": target_detections},
                ],
                "metrics": {
                    baseline: {"tp@0.50": 1, "fp@0.50": 1, "fn@0.50": 0},
                    target: target_metrics,
                },
            }
        ],
    }


def _detection(xyxy: tuple[int, int, int, int], *, score: float) -> dict:
    return {
        "box": {"xyxy": list(xyxy), "label": "object"},
        "score": score,
        "metadata": {"fusion": {"edge_support": 0.2, "aux_support": 0.3, "reliability_support": 0.8}},
    }


if __name__ == "__main__":
    unittest.main()
