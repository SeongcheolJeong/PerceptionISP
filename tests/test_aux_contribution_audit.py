from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.aux_contribution_audit import (
    build_aux_contribution_audit,
    build_aux_contribution_audit_from_paths,
    main as aux_audit_main,
    write_aux_contribution_audit,
)


class AuxContributionAuditTest(unittest.TestCase):
    def test_build_aux_contribution_audit_passes_incremental_aux_checks(self) -> None:
        summary = build_aux_contribution_audit(_rollup(), calibration_summary=_calibration_summary())

        self.assertEqual(summary["status"], "pass")
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["score_aux_uses_aux_for_fp_reduction"]["status"], "pass")
        self.assertEqual(checks["aux_adds_incremental_value_over_score_label"]["status"], "pass")
        self.assertEqual(checks["score_label_aux_model_contains_aux_features"]["status"], "pass")

        comparisons = {row["id"]: row for row in summary["comparisons"]}
        incremental = comparisons["score_label_aux_vs_score_label"]["deltas"]
        self.assertGreater(incremental["precision@0.50_mean"], 0.0)
        self.assertLess(incremental["fp@0.50_mean"], -0.02)
        self.assertGreaterEqual(incremental["recall@0.50_mean"], -0.005)
        self.assertIn("aux_support", summary["feature_audit"]["aux_features"])

    def test_write_and_cli_output_json_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollup_dir = root / "rollup"
            calibration_dir = root / "calibration"
            rollup_dir.mkdir()
            calibration_dir.mkdir()
            (rollup_dir / "rollup_summary.json").write_text(json.dumps(_rollup()) + "\n")
            (calibration_dir / "proposal_calibration_summary.json").write_text(json.dumps(_calibration_summary()) + "\n")

            summary = build_aux_contribution_audit_from_paths(rollup_dir, calibration_summary=calibration_dir)
            html_path = write_aux_contribution_audit(summary, root / "audit")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Aux Contribution Audit", html_path.read_text())
            persisted = json.loads((html_path.parent / "aux_contribution_audit_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aux_audit_main(
                    [
                        str(rollup_dir),
                        "--calibration-summary",
                        str(calibration_dir),
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["failed_checks"], [])
            self.assertTrue((root / "cli" / "aux_contribution_audit_summary.json").exists())

    def test_sample_bridge_audits_incremental_aux_removed_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score_label = root / "score_label"
            score_label_aux = root / "score_label_aux"
            score_label.mkdir()
            score_label_aux.mkdir()
            _write_comparison_summary(
                score_label / "comparison_summary.json",
                input_name="perception_calibrated_score_label_fusion_rgb_aux",
                samples=[
                    _sample("a", [_det("person", 10, 10, 30, 30, 0.90), _det("car", 50, 50, 70, 70, 0.45, aux=0.10, edge=0.05)], [_gt("person", 10, 10, 30, 30)]),
                    _sample("b", [_det("car", 4, 4, 20, 20, 0.80), _det("truck", 60, 60, 80, 80, 0.40, aux=0.15, edge=0.08)], [_gt("car", 4, 4, 20, 20)]),
                ],
            )
            _write_comparison_summary(
                score_label_aux / "comparison_summary.json",
                input_name="perception_calibrated_score_label_aux_fusion_rgb_aux",
                samples=[
                    _sample("a", [_det("person", 10, 10, 30, 30, 0.91)], [_gt("person", 10, 10, 30, 30)]),
                    _sample("b", [_det("car", 4, 4, 20, 20, 0.82)], [_gt("car", 4, 4, 20, 20)]),
                ],
            )
            rollup = _rollup(
                score_label_summary=score_label / "comparison_summary.json",
                score_label_aux_summary=score_label_aux / "comparison_summary.json",
            )

            summary = build_aux_contribution_audit(rollup, calibration_summary=_calibration_summary())

            bridge = summary["sample_bridge"]
            self.assertEqual(bridge["compared_sample_count"], 2)
            self.assertEqual(bridge["removed_fp_count"], 2)
            self.assertEqual(bridge["removed_tp_count"], 0)
            self.assertEqual(bridge["fp_delta_count"], -2)
            checks = {row["id"]: row for row in summary["checks"]}
            self.assertEqual(checks["same_sample_aux_bridge_available"]["status"], "pass")
            self.assertEqual(checks["incremental_aux_removes_more_fp_than_tp"]["status"], "pass")
            self.assertEqual(checks["incremental_aux_net_fp_reduction_same_sample"]["status"], "pass")

            html_path = write_aux_contribution_audit(summary, root / "audit")
            self.assertIn("Same-Sample Bridge", html_path.read_text())


def _rollup(score_label_summary: Path | None = None, score_label_aux_summary: Path | None = None) -> dict:
    baseline = _metrics(0.60, 0.500, 0.300, 0.200, 1.000, 3.000)
    score_aux = _metrics(0.61, 0.497, 0.298, 0.198, 0.950, 2.950)
    score_label = _metrics(0.65, 0.496, 0.297, 0.199, 0.800, 2.800)
    score_label_aux = _metrics(0.66, 0.494, 0.296, 0.198, 0.750, 2.750)
    return {
        "run_count": 4,
        "baseline_input": "perception_fusion_rgb_aux",
        "runs": [
            _run("source", {"perception_fusion_rgb_aux": baseline}),
            _run("score_aux", {"perception_fusion_rgb_aux": baseline, "perception_calibrated_score_aux_fusion_rgb_aux": score_aux}),
            _run(
                "score_label",
                {"perception_fusion_rgb_aux": baseline, "perception_calibrated_score_label_fusion_rgb_aux": score_label},
                summary_path=score_label_summary,
            ),
            _run(
                "score_label_aux",
                {
                    "perception_fusion_rgb_aux": baseline,
                    "perception_calibrated_score_label_aux_fusion_rgb_aux": score_label_aux,
                },
                summary_path=score_label_aux_summary,
            ),
        ],
    }


def _run(name: str, inputs: dict, *, summary_path: Path | None = None) -> dict:
    return {
        "name": name,
        "sample_count": 10,
        "summary_path": str(summary_path or f"/tmp/{name}/comparison_summary.json"),
        "html_path": f"/tmp/{name}/index.html",
        "inputs": inputs,
    }


def _metrics(precision: float, recall: float, recall75: float, small_recall: float, fp: float, det_count: float) -> dict:
    return {
        "precision@0.50_mean": precision,
        "recall@0.50_mean": recall,
        "recall@0.75_mean": recall75,
        "small_recall@0.50_mean": small_recall,
        "fp@0.50_mean": fp,
        "det_count_mean": det_count,
    }


def _calibration_summary() -> dict:
    return {
        "models": [
            {
                "feature_set": "score_label_aux",
                "feature_names": ["score", "aux_support", "edge_support", "label_seen_gt"],
                "weights": [0.4, -0.2, 0.1, 0.5],
            }
        ]
    }


def _write_comparison_summary(path: Path, *, input_name: str, samples: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "sample_count": len(samples),
                "run_config": {"label_agnostic": False},
                "samples": [
                    {
                        **sample,
                        "detectors": [{"input_name": input_name, "detector_name": "unit", "detections": sample["detections"]}],
                    }
                    for sample in samples
                ],
            }
        )
        + "\n"
    )


def _sample(sample_id: str, detections: list[dict], ground_truth: list[dict]) -> dict:
    return {"sample_id": sample_id, "ground_truth": ground_truth, "detections": detections}


def _gt(label: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"xyxy": [x1, y1, x2, y2], "label": label}


def _det(label: str, x1: float, y1: float, x2: float, y2: float, score: float, *, aux: float = 0.6, edge: float = 0.4) -> dict:
    return {
        "box": {"xyxy": [x1, y1, x2, y2], "label": label},
        "score": score,
        "metadata": {
            "fusion": {
                "rgb_score": score,
                "aux_support": aux,
                "edge_support": edge,
                "saturation_support": 0.0,
                "reliability_support": 0.7,
                "aux_box_iou": 0.1,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
