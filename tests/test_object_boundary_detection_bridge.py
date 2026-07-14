from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.object_boundary_detection_bridge import (
    build_object_boundary_detection_bridge,
    main as bridge_main,
    write_object_boundary_detection_bridge,
)


class ObjectBoundaryDetectionBridgeTest(unittest.TestCase):
    def test_build_bridge_joins_edge_rows_to_detection_outcomes(self) -> None:
        object_boundary = _object_boundary_summary()
        comparison = _comparison_summary()

        summary = build_object_boundary_detection_bridge(
            object_boundary,
            comparison,
            baseline_input="human_rgb",
            target_input="perception_score_aux",
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["claim_status"], "object_boundary_detection_bridge_diagnostic")
        self.assertEqual(summary["object_count"], 3)
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["baseline_detected_count"], 2)
        self.assertEqual(aggregate["target_detected_count"], 2)
        self.assertEqual(aggregate["target_only_detected_count"], 1)
        self.assertEqual(aggregate["baseline_only_detected_count"], 1)
        self.assertIn("target_detected_vs_missed_aux_edge_confidence_boundary_f1_auc_high_feature_predicts_positive", summary["correlations"]["key_results"])
        self.assertEqual({row["id"]: row["status"] for row in summary["checks"]}["target_detection_correlation_computable"], "pass")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_dir = root / "object"
            comparison_dir = root / "comparison"
            object_dir.mkdir()
            comparison_dir.mkdir()
            (object_dir / "object_boundary_edge_summary.json").write_text(json.dumps(_object_boundary_summary()) + "\n")
            (comparison_dir / "comparison_summary.json").write_text(json.dumps(_comparison_summary()) + "\n")

            summary = build_object_boundary_detection_bridge(
                _object_boundary_summary(),
                _comparison_summary(),
                baseline_input="human_rgb",
                target_input="perception_score_aux",
            )
            html_path = write_object_boundary_detection_bridge(summary, root / "report")
            self.assertTrue((html_path.parent / "object_boundary_detection_bridge_summary.json").exists())
            self.assertIn("Object Boundary Detection Bridge", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = bridge_main(
                    [
                        "--object-boundary-edge",
                        str(object_dir),
                        "--comparison-report",
                        str(comparison_dir),
                        "--baseline-input",
                        "human_rgb",
                        "--target-input",
                        "perception_score_aux",
                        "--output-dir",
                        str(root / "cli_report"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["object_count"], 3)


def _object_boundary_summary() -> dict:
    return {
        "status": "pass",
        "pass": True,
        "cases": [
            {
                "id": "sample-1",
                "box_rows": [
                    _box("sample-1", "car", [10, 10, 30, 30], human=0.20, perception=0.22, aux_strength=0.24, aux_conf=0.25),
                    _box("sample-1", "person", [50, 10, 70, 40], human=0.06, perception=0.07, aux_strength=0.08, aux_conf=0.18),
                ],
            },
            {
                "id": "sample-2",
                "box_rows": [
                    _box("sample-2", "car", [12, 12, 32, 32], human=0.05, perception=0.05, aux_strength=0.04, aux_conf=0.03),
                ],
            },
        ],
    }


def _box(sample_id: str, label: str, xyxy: list[float], *, human: float, perception: float, aux_strength: float, aux_conf: float) -> dict:
    return {
        "sample_id": sample_id,
        "label": label,
        "area": float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])),
        "area_bucket": "small",
        "xyxy": xyxy,
        "human_rgb_edge_boundary_f1": human,
        "perception_rgb_edge_boundary_f1": perception,
        "aux_edge_strength_boundary_f1": aux_strength,
        "aux_edge_confidence_boundary_f1": aux_conf,
        "perception_rgb_minus_human_boundary_f1": perception - human,
        "aux_strength_minus_human_boundary_f1": aux_strength - human,
        "aux_confidence_minus_human_boundary_f1": aux_conf - human,
        "human_rgb_edge_boundary_separation": human + 0.01,
        "perception_rgb_edge_boundary_separation": perception + 0.01,
        "aux_edge_strength_boundary_separation": aux_strength + 0.01,
        "aux_edge_confidence_boundary_separation": aux_conf + 0.01,
    }


def _comparison_summary() -> dict:
    return {
        "sample_count": 2,
        "run_config": {"label_agnostic": False},
        "samples": [
            {
                "sample_id": "sample-1",
                "detectors": [
                    {"input_name": "human_rgb", "detections": [_det("car", [10, 10, 30, 30], 0.80), _det("person", [50, 10, 70, 40], 0.60)]},
                    {"input_name": "perception_score_aux", "detections": [_det("car", [10, 10, 30, 30], 0.90)]},
                ],
            },
            {
                "sample_id": "sample-2",
                "detectors": [
                    {"input_name": "human_rgb", "detections": []},
                    {"input_name": "perception_score_aux", "detections": [_det("car", [12, 12, 32, 32], 0.55)]},
                ],
            },
        ],
    }


def _det(label: str, xyxy: list[float], score: float) -> dict:
    return {"box": {"xyxy": xyxy, "label": label}, "score": score, "metadata": {}}


if __name__ == "__main__":
    unittest.main()
