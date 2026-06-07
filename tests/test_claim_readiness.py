from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.claim_readiness import main as readiness_main, run_claim_readiness


class ClaimReadinessTest(unittest.TestCase):
    def test_run_claim_readiness_builds_gates_training_rollup_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            train_dir = _write_train_summary(root / "train")
            eval_dir = _write_eval_summary(root / "eval")
            rollup_dir = _write_comparison_rollup(root / "rollup")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")

            summary = run_claim_readiness(
                comparison_report=report_dir,
                min_samples=3,
                bootstrap_samples=16,
                bootstrap_seed="unit",
                training_summaries=[train_dir, eval_dir],
                comparison_rollups=[f"Calibration={rollup_dir}"],
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                output_dir=root / "readiness",
            )

            self.assertFalse(summary["broad_superiority"]["pass"])
            self.assertTrue(summary["fp_reducer"]["pass"])
            self.assertTrue((root / "readiness" / "broad_superiority_vs_human" / "claim_gate_summary.json").exists())
            self.assertTrue((root / "readiness" / "fp_reducer_vs_fusion" / "claim_gate_summary.json").exists())
            self.assertTrue((root / "readiness" / "task_metrics" / "task_metrics_summary.json").exists())
            self.assertTrue((root / "readiness" / "task_gate" / "task_gate_summary.json").exists())
            self.assertTrue((root / "readiness" / "condition_metrics" / "condition_metrics_summary.json").exists())
            self.assertTrue((root / "readiness" / "condition_gate" / "condition_gate_summary.json").exists())
            self.assertTrue((root / "readiness" / "benchmark_protocol" / "protocol_coverage_summary.json").exists())
            self.assertTrue((root / "readiness" / "rgb_aux_training_rollup" / "training_rollup_summary.json").exists())
            self.assertTrue((root / "readiness" / "dashboard" / "claim_dashboard_summary.json").exists())
            self.assertIn("task_metrics", summary)
            self.assertIn("task_gate", summary)
            self.assertIn("condition_metrics", summary)
            self.assertIn("condition_gate", summary)
            self.assertIn("mechanism_validation", summary)
            self.assertIn("cfa_stress_sweep", summary)
            self.assertIn("edge_confidence_suite", summary)
            self.assertIn("benchmark_protocol", summary)
            self.assertEqual(summary["benchmark_protocol"]["status"], "not_claim_ready")
            self.assertEqual(summary["benchmark_protocol"]["coverage_status"], "coverage_incomplete")
            self.assertEqual(summary["benchmark_protocol"]["metric_claim_status"], "fp_reducer_only")
            decisions = {item["claim"]: item["status"] for item in summary["dashboard"]["decisions"]}
            self.assertEqual(decisions["Broad HumanISP superiority is not supported by the current gate evidence."], "not_supported")
            self.assertEqual(decisions["Recall-budgeted FP reduction versus the RGB+Aux fusion baseline is supported."], "supported")
            self.assertEqual(
                decisions["Task-level `recall_improvement` gate passed for the evaluated groups."],
                "supported",
            )
            self.assertTrue(
                any(
                    claim.startswith("Benchmark protocol coverage is incomplete")
                    and status == "not_supported"
                    for claim, status in decisions.items()
                )
            )
            dashboard_summary = json.loads((root / "readiness" / "dashboard" / "claim_dashboard_summary.json").read_text())
            self.assertEqual(dashboard_summary["task_metrics"]["status"], "candidate_needs_gate")
            self.assertEqual(dashboard_summary["task_gate"]["verdict"], "task_gate_pass")
            self.assertTrue(dashboard_summary["mechanism_validation"]["pass"])
            self.assertTrue(dashboard_summary["cfa_stress_sweep"]["pass"])
            self.assertTrue(dashboard_summary["edge_confidence_suite"]["pass"])
            self.assertEqual(dashboard_summary["protocol_coverage"]["status"], "not_claim_ready")
            self.assertEqual(dashboard_summary["protocol_coverage"]["coverage_status"], "coverage_incomplete")
            self.assertEqual(dashboard_summary["protocol_coverage"]["metric_claim_status"], "fp_reducer_only")

    def test_claim_readiness_cli_outputs_compact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = readiness_main(
                    [
                        str(report_dir),
                        "--min-samples",
                        "3",
                        "--bootstrap-samples",
                        "16",
                        "--bootstrap-seed",
                        "unit",
                        "--output-dir",
                        str(root / "readiness"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(printed["broad_superiority"]["pass"])
            self.assertTrue(printed["fp_reducer"]["pass"])
            self.assertIn("task_metrics", printed)
            self.assertIn("task_gate", printed)
            self.assertIn("condition_metrics", printed)
            self.assertIn("condition_gate", printed)
            self.assertIn("benchmark_protocol", printed)
            self.assertTrue((root / "readiness" / "claim_readiness_summary.json").exists())

    def test_claim_readiness_accepts_protocol_only_comparison_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            naive_dir = _write_comparison_report(root / "naive", tone_mapping="linear", demosaic_method="bilinear", denoise_strength=0.0)
            train_dir = _write_train_summary(root / "train")
            eval_dir = _write_eval_summary(root / "eval")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")

            summary = run_claim_readiness(
                comparison_report=report_dir,
                min_samples=3,
                bootstrap_samples=16,
                bootstrap_seed="unit",
                training_summaries=[train_dir, eval_dir],
                protocol_comparison_reports=[naive_dir],
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                output_dir=root / "readiness",
            )

            self.assertEqual(summary["benchmark_protocol"]["status"], "claim_ready")
            self.assertEqual(summary["benchmark_protocol"]["coverage_status"], "coverage_complete")
            self.assertEqual(summary["benchmark_protocol"]["metric_claim_status"], "fp_reducer_only")
            self.assertIn(str(naive_dir), summary["protocol_comparison_reports"])
            dashboard_summary = json.loads((root / "readiness" / "dashboard" / "claim_dashboard_summary.json").read_text())
            self.assertEqual(dashboard_summary["protocol_coverage"]["status"], "claim_ready")
            self.assertEqual(dashboard_summary["protocol_coverage"]["coverage_status"], "coverage_complete")
            self.assertEqual(dashboard_summary["protocol_coverage"]["metric_claim_status"], "fp_reducer_only")
            self.assertTrue(dashboard_summary["mechanism_validation"]["pass"])
            self.assertTrue(dashboard_summary["cfa_stress_sweep"]["pass"])
            self.assertTrue(dashboard_summary["edge_confidence_suite"]["pass"])


def _write_comparison_report(
    path: Path,
    *,
    tone_mapping: str = "log",
    demosaic_method: str = "edge_aware",
    denoise_strength: float = 0.18,
) -> Path:
    path.mkdir()
    samples = []
    for index in range(3):
        target_detections = [_match_detection()]
        samples.append(
            {
                "sample_id": str(index),
                "ground_truth": [{"xyxy": [0.0, 0.0, 20.0, 20.0], "label": "person"}],
                "detectors": [
                    {"input_name": "human_rgb", "detections": [_match_detection(), _fp_detection()]},
                    {"input_name": "perception_fusion_rgb_aux", "detections": [_match_detection(), _fp_detection()]},
                    {"input_name": "perception_calibrated_score_label_aux_fusion_rgb_aux", "detections": target_detections},
                ],
                "metrics": {
                    "human_rgb": {
                        "precision@0.50": 0.60,
                        "recall@0.50": 0.470,
                        "recall@0.75": 0.300,
                        "small_recall@0.50": 0.280,
                        "fp@0.50": 1.30,
                    },
                    "perception_fusion_rgb_aux": {
                        "precision@0.50": 0.600,
                        "recall@0.50": 0.464,
                        "recall@0.75": 0.303,
                        "small_recall@0.50": 0.279,
                        "fp@0.50": 1.30,
                    },
                    "perception_calibrated_score_label_aux_fusion_rgb_aux": {
                        "precision@0.50": 0.630,
                        "recall@0.50": 0.459,
                        "recall@0.75": 0.299,
                        "small_recall@0.50": 0.276,
                        "fp@0.50": 1.00,
                    },
                },
            }
        )
    aggregate = {
        "human_rgb": _aggregate([sample["metrics"]["human_rgb"] for sample in samples]),
        "perception_fusion_rgb_aux": _aggregate([sample["metrics"]["perception_fusion_rgb_aux"] for sample in samples]),
        "perception_calibrated_score_label_aux_fusion_rgb_aux": _aggregate(
            [sample["metrics"]["perception_calibrated_score_label_aux_fusion_rgb_aux"] for sample in samples]
        ),
    }
    payload = {
        "sample_count": len(samples),
        "samples": samples,
        "aggregate": aggregate,
        "run_config": {
            "source": "yolo-dataset",
            "dataset": "KITTI",
            "split": "val",
            "count": len(samples),
            "rgb_detector": "ultralytics_yolo",
            "rgb_detector_model": "yolo11n.pt",
            "rgb_detector_confidence": 0.25,
            "label_agnostic": False,
            "tone_mapping": tone_mapping,
            "demosaic_method": demosaic_method,
            "denoise_strength": denoise_strength,
        },
    }
    (path / "comparison_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _match_detection() -> dict:
    return {"box": {"xyxy": [0.0, 0.0, 20.0, 20.0], "label": "person"}, "score": 0.9}


def _fp_detection() -> dict:
    return {"box": {"xyxy": [40.0, 40.0, 60.0, 60.0], "label": "person"}, "score": 0.4}


def _aggregate(rows: list[dict]) -> dict:
    mapping = {
        "precision@0.50_mean": "precision@0.50",
        "recall@0.50_mean": "recall@0.50",
        "recall@0.75_mean": "recall@0.75",
        "small_recall@0.50_mean": "small_recall@0.50",
        "fp@0.50_mean": "fp@0.50",
    }
    return {out_key: sum(float(row[in_key]) for row in rows) / len(rows) for out_key, in_key in mapping.items()}


def _write_train_summary(path: Path) -> Path:
    path.mkdir()
    (path / "train_dense_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 3,
                "train_sample_count": 2,
                "eval_sample_count": 1,
                "epochs": 1,
                "device": "cpu",
                "channel_mode": "rgb_aux",
                "elapsed_seconds": 0.1,
                "sample_epochs_per_second": 30.0,
            }
        )
        + "\n"
    )
    return path


def _write_eval_summary(path: Path) -> Path:
    path.mkdir()
    (path / "dense_eval_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 1,
                "checkpoint_summary": {"channel_mode": "rgb_aux"},
                "aggregate": {
                    "precision@0.50_mean": 0.01,
                    "recall@0.50_mean": 0.09,
                    "fp@0.50_mean": 40.0,
                    "det_count_mean": 41.0,
                },
            }
        )
        + "\n"
    )
    return path


def _write_comparison_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "rollup_summary.json").write_text(json.dumps({"run_count": 1, "baseline_input": "human_rgb", "runs": []}) + "\n")
    return path


def _write_mechanism_validation(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "cfa_patterns": ["RGGB", "GRBG", "RCCB", "RGBIR"],
        "mechanisms": [
            {"id": "low_light_noise_response", "status": "pass"},
            {"id": "glare_saturation_response", "status": "pass"},
            {"id": "low_mtf_edge_confidence_response", "status": "pass"},
            {"id": "cfa_variant_support", "status": "pass"},
        ],
    }
    (path / "mechanism_validation_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_stress_sweep(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "cfa_patterns": ["RGGB", "RCCB", "RGBIR", "MONO"],
        "support": {"case_count": 8, "all_finite": True, "all_supported": True, "failed_cases": []},
        "condition_rankings": [
            {
                "condition": "low_light",
                "score_definition": "unit low light",
                "ranked_cfas": [{"rank": 1, "cfa_pattern": "MONO", "condition_score": 0.65}],
            },
            {
                "condition": "glare",
                "score_definition": "unit glare",
                "ranked_cfas": [{"rank": 1, "cfa_pattern": "RGBIR", "condition_score": 0.60}],
            },
        ],
    }
    (path / "cfa_stress_sweep_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_edge_confidence_suite(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "cases": [{"id": "nominal_sharp"}, {"id": "low_light"}, {"id": "glare_saturated"}, {"id": "low_mtf"}],
        "checks": [
            {
                "id": "low_light_edge_confidence_drop",
                "status": "pass",
                "criteria": [{"metric": "edge_confidence_mean", "delta": -0.15, "threshold": -0.10, "pass": True}],
            },
            {
                "id": "glare_edge_confidence_drop",
                "status": "pass",
                "criteria": [{"metric": "demosaic_confidence_mean", "delta": -0.12, "threshold": -0.08, "pass": True}],
            },
        ],
    }
    (path / "edge_confidence_suite_summary.json").write_text(json.dumps(payload) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
