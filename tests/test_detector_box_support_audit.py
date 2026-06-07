from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.detector_box_support_audit import (
    build_detector_box_support_audit,
    main as detector_box_support_main,
    write_detector_box_support_audit,
)


class DetectorBoxSupportAuditTest(unittest.TestCase):
    def test_build_audit_records_global_fp_counterexample_and_removed_fp_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = _write_scene(root / "scene.png")
            report = _comparison_report(image_path)

            summary = build_detector_box_support_audit(
                report,
                audit_inputs=("human_rgb", "perception_fusion_rgb_aux", "perception_target"),
                target_input="perception_target",
                transition_baseline_input="perception_fusion_rgb_aux",
                source_report_path=_write_report(root / "report", report),
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], "detector_box_support_diagnostic")
            target = summary["target_summary"]
            self.assertEqual(target["fp_count"], 1)
            self.assertEqual(target["tp_count"], 1)
            target_rows = {
                row["feature"]: row
                for row in target["correlations"]["rows"]
            }
            self.assertGreater(target_rows["edge_support"]["delta_fp_minus_tp"], 0.0)
            self.assertFalse(target_rows["edge_support"]["lower_feature_predicts_fp"])
            self.assertLess(target_rows["score"]["delta_fp_minus_tp"], 0.0)
            self.assertTrue(target_rows["score"]["lower_feature_predicts_fp"])

            bridge = summary["transition_bridge"]
            self.assertEqual(bridge["removed_fp_count"], 1)
            self.assertEqual(bridge["removed_tp_count"], 0)
            bridge_rows = {
                (row["comparison"], row["feature"]): row
                for row in bridge["proposal_correlation"]["rows"]
            }
            removed_edge = bridge_rows[("removed_fp_vs_kept_tp", "edge_support")]
            self.assertLess(removed_edge["delta"], 0.0)
            self.assertTrue(removed_edge["lower_feature_predicts_positive"])

            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["target_aux_edge_global_result_recorded"], "pass")
            self.assertEqual(checks["removed_fp_has_lower_aux_edge_support"], "pass")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = _write_scene(root / "scene.png")
            report = _comparison_report(image_path)
            report_dir = root / "comparison"
            report_path = _write_report(report_dir, report)

            summary = build_detector_box_support_audit(
                report,
                audit_inputs=("human_rgb", "perception_fusion_rgb_aux", "perception_target"),
                target_input="perception_target",
                transition_baseline_input="perception_fusion_rgb_aux",
                source_report_path=report_path,
            )
            html_path = write_detector_box_support_audit(summary, root / "audit")
            self.assertTrue((html_path.parent / "detector_box_support_audit_summary.json").exists())
            self.assertIn("Detector-Box Support Audit", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = detector_box_support_main(
                    [
                        str(report_dir),
                        "--audit-input",
                        "human_rgb",
                        "--audit-input",
                        "perception_fusion_rgb_aux",
                        "--audit-input",
                        "perception_target",
                        "--target-input",
                        "perception_target",
                        "--transition-baseline-input",
                        "perception_fusion_rgb_aux",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["target_input"], "perception_target")


def _write_scene(path: Path) -> Path:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 8:10, :] = 255
    Image.fromarray(image).save(path)
    return path


def _write_report(path: Path, payload: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / "comparison_summary.json"
    report_path.write_text(json.dumps(payload) + "\n")
    (path / "index.html").write_text("<html></html>")
    return report_path


def _comparison_report(image_path: Path) -> dict:
    kept_tp = _det("car", (5, 5, 20, 20), 0.90, edge=0.60, aux=0.60)
    global_fp_kept = _det("truck", (40, 5, 55, 20), 0.30, edge=0.90, aux=0.90)
    removed_fp = _det("truck", (40, 40, 55, 55), 0.25, edge=0.10, aux=0.20)
    return {
        "sample_count": 1,
        "run_config": {"label_agnostic": False},
        "samples": [
            {
                "sample_id": "sample",
                "metadata": {"image_path": str(image_path), "width": 64, "height": 64},
                "ground_truth": [{"box": {"xyxy": [5, 5, 20, 20], "label": "car"}}],
                "detectors": [
                    {"input_name": "human_rgb", "detections": [kept_tp]},
                    {"input_name": "perception_fusion_rgb_aux", "detections": [kept_tp, global_fp_kept, removed_fp]},
                    {"input_name": "perception_target", "detections": [kept_tp, global_fp_kept]},
                ],
            }
        ],
    }


def _det(label: str, xyxy: tuple[int, int, int, int], score: float, *, edge: float, aux: float) -> dict:
    return {
        "box": {"xyxy": list(xyxy), "label": label},
        "score": score,
        "metadata": {
            "fusion": {
                "aux_support": aux,
                "edge_support": edge,
                "saturation_support": 0.0,
                "reliability_support": 0.8,
                "aux_box_iou": aux,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
