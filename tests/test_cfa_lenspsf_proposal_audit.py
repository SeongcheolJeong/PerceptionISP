from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.evaluation.cfa_lenspsf_proposal_audit import build_cfa_lenspsf_proposal_audit_from_path, write_cfa_lenspsf_proposal_audit


class CfaLensPsfProposalAuditTest(unittest.TestCase):
    def test_build_proposal_audit_from_sweep_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = _write_sweep_fixture(root / "sweep")

            summary = build_cfa_lenspsf_proposal_audit_from_path(sweep)

            self.assertEqual(summary["condition_count"], 1)
            self.assertEqual(summary["checks"][0]["status"], "pass")
            condition = summary["conditions"][0]
            self.assertEqual(condition["removed_fp_count"], 1)
            self.assertEqual(condition["removed_tp_count"], 0)
            self.assertLess(condition["edge_support_delta_removed_fp_minus_kept_tp"], 0.0)
            self.assertLess(condition["scene_edge_support_delta_removed_fp_minus_kept_tp"], 0.0)
            self.assertGreater(condition["scene_edge_auc_low_predicts_removed_fp"], 0.5)
            aggregate = summary["aggregate"]
            self.assertEqual(aggregate["sample_count"], 1)
            self.assertEqual(aggregate["removed_fp_count"], 1)
            self.assertEqual(aggregate["edge_delta_negative_condition_count"], 1)
            self.assertEqual(aggregate["scene_edge_delta_negative_condition_count"], 1)
            self.assertAlmostEqual(
                aggregate["edge_support_delta_condition_mean"],
                condition["edge_support_delta_removed_fp_minus_kept_tp"],
            )
            self.assertAlmostEqual(
                aggregate["scene_edge_auc_condition_mean"],
                condition["scene_edge_auc_low_predicts_removed_fp"],
            )
            html_path = write_cfa_lenspsf_proposal_audit(summary, root / "audit")
            self.assertTrue(html_path.exists())
            self.assertTrue((html_path.parent / "cfa_lenspsf_proposal_audit_summary.json").exists())
            self.assertIn("CFA/LensPSF Proposal Audit", html_path.read_text())


def _write_sweep_fixture(path: Path) -> Path:
    path.mkdir()
    condition_dir = path / "001_cfa-grbg_psf-0p00"
    condition_dir.mkdir()
    (condition_dir / "index.html").write_text("<html></html>")
    image_path = path / "scene.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 7:9, :] = 255
    Image.fromarray(image).save(image_path)
    (condition_dir / "comparison_summary.json").write_text(json.dumps(_comparison_payload(image_path)) + "\n")
    sweep_summary = {
        "run_count": 1,
        "cfa_patterns": ["GRBG"],
        "psf_sigmas": [0.0],
        "runs": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "report": "001_cfa-grbg_psf-0p00/index.html",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "metrics": {
                    "human_rgb": {},
                    "perception_fusion_rgb_aux": {},
                    "perception_calibrated_score_label_aux_fusion_rgb_aux_t001": {},
                },
            }
        ],
    }
    (path / "cfa_lenspsf_detector_sweep_summary.json").write_text(json.dumps(sweep_summary) + "\n")
    return path


def _comparison_payload(image_path: Path) -> dict:
    baseline = "perception_fusion_rgb_aux"
    target = "perception_calibrated_score_label_aux_fusion_rgb_aux_t001"
    kept_tp = _detection((2, 2, 12, 12), score=0.9, edge_support=0.8)
    removed_fp = _detection((20, 20, 28, 28), score=0.4, edge_support=0.1)
    return {
        "run_config": {"label_agnostic": False},
        "aggregate": {
            baseline: {},
            target: {},
        },
        "samples": [
            {
                "sample_id": "sample",
                "metadata": {"image_path": str(image_path), "width": 32, "height": 32},
                "ground_truth": [{"box": {"xyxy": [2, 2, 12, 12], "label": "object"}}],
                "detectors": [
                    {"input_name": baseline, "detections": [kept_tp, removed_fp]},
                    {"input_name": target, "detections": [kept_tp]},
                ],
            }
        ],
    }


def _detection(xyxy: tuple[int, int, int, int], *, score: float, edge_support: float) -> dict:
    return {
        "box": {"xyxy": list(xyxy), "label": "object"},
        "score": score,
        "metadata": {
            "fusion": {
                "aux_support": edge_support,
                "edge_support": edge_support,
                "saturation_support": 0.0,
                "reliability_support": 0.9,
                "aux_box_iou": edge_support,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
