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
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_edge_sweep = _write_scene_edge_confidence(root / "scene_edge_sweep", cfa_pattern="RGGB", psf_sigmas=(0.0, 1.0))
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")
            cfa_lenspsf_proposal = _write_cfa_lenspsf_proposal_audit(root / "cfa_lenspsf_proposal")
            cfa_lenspsf_native = _write_cfa_lenspsf_native_audit(root / "cfa_lenspsf_native")
            cfa_lenspsf_casebook = _write_cfa_lenspsf_casebook(root / "cfa_lenspsf_casebook")
            casebook = _write_casebook(root / "casebook")

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
                edge_fidelity_suite=edge_fidelity,
                scene_edge_confidence=[scene_edge, scene_edge_sweep],
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                cfa_lenspsf_proposal_audit=cfa_lenspsf_proposal,
                cfa_lenspsf_native_audit=cfa_lenspsf_native,
                cfa_lenspsf_casebook=cfa_lenspsf_casebook,
                casebook=casebook,
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
            self.assertIn("edge_fidelity_suite", summary)
            self.assertIn("scene_edge_confidence", summary)
            self.assertIn("scene_information_stress", summary)
            self.assertIn("aux_contribution_audit", summary)
            self.assertIn("cfa_lenspsf_proposal_audit", summary)
            self.assertIn("cfa_lenspsf_native_audit", summary)
            self.assertIn("cfa_lenspsf_casebook", summary)
            self.assertIn("casebook", summary)
            self.assertIn("benchmark_protocol", summary)
            self.assertEqual(summary["benchmark_protocol"]["status"], "not_claim_ready")
            self.assertEqual(summary["benchmark_protocol"]["coverage_status"], "coverage_incomplete")
            self.assertEqual(summary["benchmark_protocol"]["metric_claim_status"], "fp_reducer_only")
            self.assertTrue(summary["scene_information_stress"]["pass"])
            self.assertTrue(summary["scene_edge_confidence"]["pass"])
            self.assertEqual(summary["scene_edge_confidence"]["report_count"], 2)
            self.assertTrue(summary["cfa_lenspsf_proposal_audit"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_proposal_audit"]["removed_fp_count"], 5)
            self.assertTrue(summary["cfa_lenspsf_native_audit"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_native_audit"]["native_run_count"], 1)
            self.assertTrue(summary["cfa_lenspsf_casebook"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_casebook"]["selected_case_count"], 5)
            self.assertTrue(summary["casebook"]["pass"])
            self.assertEqual(summary["casebook"]["selected_case_count"], 4)
            self.assertEqual(summary["scene_edge_confidence"]["cfa_patterns"], ["GRBG", "RGGB"])
            self.assertAlmostEqual(summary["scene_edge_confidence"]["perception_rgb_minus_human_source_edge_f1_mean"], 0.01)
            self.assertAlmostEqual(summary["scene_edge_confidence"]["perception_aux_strength_source_edge_f1_win_rate"], 1.0)
            self.assertTrue(summary["edge_fidelity_suite"]["pass"])
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
            self.assertTrue(dashboard_summary["edge_fidelity_suite"]["pass"])
            self.assertTrue(dashboard_summary["scene_edge_confidence"]["pass"])
            self.assertEqual(dashboard_summary["scene_edge_confidence"]["report_count"], 2)
            self.assertTrue(dashboard_summary["scene_information_stress"]["pass"])
            self.assertTrue(dashboard_summary["aux_contribution_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_proposal_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_native_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_casebook"]["pass"])
            self.assertTrue(dashboard_summary["casebook"]["pass"])
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
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")
            cfa_lenspsf_detector = _write_cfa_lenspsf_detector_sweep(root / "cfa_lenspsf_detector")
            cfa_lenspsf_native = _write_cfa_lenspsf_native_audit(root / "cfa_lenspsf_native", all_native=True)
            cfa_lenspsf_casebook = _write_cfa_lenspsf_casebook(root / "cfa_lenspsf_casebook")

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
                edge_fidelity_suite=edge_fidelity,
                scene_edge_confidence=scene_edge,
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                cfa_lenspsf_detector_sweep=cfa_lenspsf_detector,
                cfa_lenspsf_native_audit=cfa_lenspsf_native,
                cfa_lenspsf_casebook=cfa_lenspsf_casebook,
                output_dir=root / "readiness",
            )

            self.assertEqual(summary["benchmark_protocol"]["status"], "claim_ready")
            self.assertEqual(summary["benchmark_protocol"]["coverage_status"], "coverage_complete")
            self.assertEqual(summary["benchmark_protocol"]["metric_claim_status"], "fp_reducer_only")
            self.assertTrue(summary["scene_information_stress"]["pass"])
            self.assertTrue(summary["scene_edge_confidence"]["pass"])
            self.assertTrue(summary["edge_fidelity_suite"]["pass"])
            self.assertIn(str(naive_dir), summary["protocol_comparison_reports"])
            dashboard_summary = json.loads((root / "readiness" / "dashboard" / "claim_dashboard_summary.json").read_text())
            self.assertEqual(dashboard_summary["protocol_coverage"]["status"], "claim_ready")
            self.assertEqual(dashboard_summary["protocol_coverage"]["coverage_status"], "coverage_complete")
            self.assertEqual(dashboard_summary["protocol_coverage"]["metric_claim_status"], "fp_reducer_only")
            self.assertTrue(dashboard_summary["mechanism_validation"]["pass"])
            self.assertTrue(dashboard_summary["cfa_stress_sweep"]["pass"])
            self.assertTrue(dashboard_summary["edge_confidence_suite"]["pass"])
            self.assertTrue(dashboard_summary["edge_fidelity_suite"]["pass"])
            self.assertTrue(dashboard_summary["scene_edge_confidence"]["pass"])
            self.assertTrue(dashboard_summary["scene_information_stress"]["pass"])
            self.assertTrue(dashboard_summary["aux_contribution_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_detector_sweep"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_native_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_casebook"]["pass"])


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


def _write_edge_fidelity_suite(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "cfa_patterns": ["RGGB", "GRBG", "RCCB"],
        "psf_sigmas": [0.0, 1.2],
        "cases": [{"id": "psf_0.00_RGGB", "psf_sigma": 0.0, "cfa_pattern": "RGGB", "metrics": {"aux_object_edge_f1": 0.70}}],
        "checks": [
            {"id": "finite_edge_fidelity_outputs", "status": "pass"},
            {"id": "object_and_sensor_edge_oracles_present", "status": "pass"},
            {"id": "edge_fidelity_metrics_bounded", "status": "pass"},
            {"id": "lens_psf_reduces_sensor_edge_contrast", "status": "pass"},
        ],
        "rankings": [
            {
                "psf_sigma": 0.0,
                "ranked_cfas": [
                    {
                        "rank": 1,
                        "cfa_pattern": "RGGB",
                        "aux_object_edge_f1": 0.70,
                        "perception_object_edge_f1": 0.67,
                        "edge_confidence_separation": 0.12,
                    }
                ],
            }
        ],
    }
    (path / "edge_fidelity_suite_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_scene_edge_confidence(path: Path, *, cfa_pattern: str = "GRBG", psf_sigmas: tuple[float, ...] = (0.0,)) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "cases": [
            {
                "id": "bus",
                "source": "sample_image_camerae2e",
                "cfa_pattern": cfa_pattern,
                "metrics": {
                    "human_rgb_proxy_source_edge_f1": 0.66,
                    "perception_rgb_proxy_source_edge_f1": 0.67,
                    "perception_rgb_minus_human_source_edge_f1": 0.01,
                    "perception_aux_strength_source_edge_f1": 0.75,
                    "perception_aux_strength_minus_human_source_edge_f1": 0.09,
                    "perception_aux_confidence_source_edge_f1": 0.37,
                    "perception_aux_confidence_minus_human_source_edge_f1": -0.29,
                },
            }
        ],
        "checks": [
            {"id": "finite_scene_edge_outputs", "status": "pass"},
            {"id": "reference_scene_edges_present", "status": "pass"},
            {"id": "scene_edge_metrics_bounded", "status": "pass"},
            {"id": "human_and_perception_edges_track_scene_edges", "status": "pass"},
            {"id": "scene_edge_f1_delta_metrics_computable", "status": "pass"},
            {"id": "camerae2e_cfa_pattern_preserved", "status": "pass"},
        ],
        "aggregate": {
            "human_rgb_proxy_source_edge_f1_mean": 0.66,
            "perception_rgb_proxy_source_edge_f1_mean": 0.67,
            "perception_rgb_minus_human_source_edge_f1_mean": 0.01,
            "perception_rgb_source_edge_f1_win_rate": 1.0,
            "perception_aux_strength_source_edge_f1_mean": 0.75,
            "perception_aux_strength_minus_human_source_edge_f1_mean": 0.09,
            "perception_aux_strength_source_edge_f1_win_rate": 1.0,
            "perception_aux_confidence_source_edge_f1_mean": 0.37,
            "perception_aux_confidence_minus_human_source_edge_f1_mean": -0.29,
            "perception_aux_confidence_source_edge_f1_win_rate": 0.0,
        },
        "cfa_patterns": [cfa_pattern],
        "psf_sigmas": list(psf_sigmas),
    }
    (path / "scene_edge_confidence_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_scene_information_stress(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "sensor_width": 160,
        "sensor_height": 96,
        "scene_width": 1280,
        "scene_height": 768,
        "cfa_pattern": "RGGB",
        "cases": [
            {
                "id": "supersampled_thin_detail",
                "sample_mode": "box",
                "metrics": {
                    "scene_luma_gradient_p90": 0.44,
                    "sensor_luma_gradient_p90": 0.0,
                    "luma_detail_retention_p90": 0.0,
                    "scene_chroma_gradient_p90": 0.0,
                    "color_confidence_mean": 0.85,
                    "signal_contrast_retention": 0.0,
                },
            },
            {
                "id": "cfa_chroma_alias",
                "sample_mode": "point",
                "metrics": {
                    "scene_luma_gradient_p90": 0.0,
                    "sensor_luma_gradient_p90": 0.0,
                    "luma_detail_retention_p90": 0.0,
                    "scene_chroma_gradient_p90": 0.86,
                    "color_confidence_mean": 0.0,
                    "signal_contrast_retention": 0.0,
                },
            },
        ],
        "checks": [
            {"id": "latent_high_frequency_detail_loss", "status": "pass"},
            {"id": "cfa_chroma_alias_color_confidence_drop", "status": "pass"},
            {"id": "subpixel_signal_fill_factor_loss", "status": "pass"},
        ],
    }
    (path / "scene_information_stress_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_aux_contribution_audit(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "checks": [
            {"id": "score_aux_uses_aux_for_fp_reduction", "status": "pass"},
            {"id": "aux_adds_incremental_value_over_score_label", "status": "pass"},
        ],
        "comparisons": [
            {
                "id": "score_label_aux_vs_score_label",
                "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                "baseline_input": "perception_calibrated_score_label_fusion_rgb_aux",
                "deltas": {"precision@0.50_mean": 0.005, "recall@0.50_mean": -0.002, "fp@0.50_mean": -0.060},
            }
        ],
        "feature_audit": {"aux_feature_count": 3, "aux_features": ["aux_support", "edge_support", "reliability_support"]},
    }
    (path / "aux_contribution_audit_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_lenspsf_detector_sweep(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "run_count": 2,
        "expected_run_count": 2,
        "count": 6,
        "width": 640,
        "height": 192,
        "cfa_patterns": ["GRBG", "RGGB"],
        "psf_sigmas": [0.0],
        "checks": [
            {"id": "condition_grid_complete", "status": "pass", "evidence": "runs=2 expected=2"},
            {"id": "psf_sigma_recorded_in_raw_provenance", "status": "pass", "evidence": "recorded=6 samples=6"},
            {"id": "detector_metrics_available", "status": "pass", "evidence": "metric_runs=2/2"},
        ],
        "runs": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "raw_condition_summary": {
                    "pattern_remapped_fraction": 0.0,
                    "true_sensor_cfa_mosaic_fraction": 1.0,
                    "camerae2e_camera_types": {"bayer-grbg": 3},
                    "camerae2e_native_cfa_bridge_versions": {"native_bayer_v1": 3},
                },
            },
            {
                "run_id": "cfa-rggb_psf-0p00",
                "raw_condition_summary": {
                    "pattern_remapped_fraction": 0.0,
                    "true_sensor_cfa_mosaic_fraction": 1.0,
                    "camerae2e_camera_types": {"bayer-rggb": 3},
                    "camerae2e_native_cfa_bridge_versions": {"native_bayer_v1": 3},
                },
            },
        ],
    }
    (path / "cfa_lenspsf_detector_sweep_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_lenspsf_proposal_audit(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "condition_count": 1,
        "expected_condition_count": 1,
        "cfa_patterns": ["GRBG"],
        "psf_sigmas": [0.0],
        "checks": [
            {"id": "condition_bridges_available", "status": "pass", "evidence": "bridges=1 expected=1"},
            {"id": "removed_fp_observed_across_conditions", "status": "pass", "evidence": "removed_fp=5 removed_tp=0"},
            {"id": "source_scene_edge_predicts_removed_fp_in_some_conditions", "status": "pass", "evidence": "positive_conditions=1/1"},
            {"id": "aux_edge_predicts_removed_fp_in_some_conditions", "status": "pass", "evidence": "positive_conditions=1/1"},
        ],
        "aggregate": {
            "condition_count": 1,
            "removed_fp_count": 5,
            "removed_tp_count": 0,
            "fp_delta_count": -5,
            "tp_delta_count": 0,
            "scene_edge_positive_condition_count": 1,
            "edge_positive_condition_count": 1,
            "best_scene_edge_auc_condition": {
                "run_id": "cfa-grbg_psf-0p00",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "scene_edge_auc_low_predicts_removed_fp": 0.62,
            },
            "best_edge_auc_condition": {
                "run_id": "cfa-grbg_psf-0p00",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "edge_auc_low_predicts_removed_fp": 0.53,
            },
        },
        "conditions": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "report": str(path / "001_cfa-grbg_psf-0p00" / "comparison_summary.json"),
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "sample_count": 3,
                "removed_fp_count": 5,
                "removed_tp_count": 0,
                "fp_delta_count": -5,
                "edge_support_delta_removed_fp_minus_kept_tp": -0.05,
                "edge_auc_low_predicts_removed_fp": 0.53,
                "scene_edge_support_delta_removed_fp_minus_kept_tp": -0.02,
                "scene_edge_auc_low_predicts_removed_fp": 0.62,
            }
        ],
        "interpretation": "unit CFA/LensPSF proposal-edge audit",
        "claim_boundary": "unit proposal boundary",
    }
    (path / "cfa_lenspsf_proposal_audit_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_lenspsf_native_audit(path: Path, *, all_native: bool = False) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    native_run_count = 2 if all_native else 1
    native_sample_count = 6 if all_native else 3
    remapped_run_count = 0 if all_native else 1
    remapped_sample_count = 0 if all_native else 3
    native_cfas = ["GRBG", "RGGB"] if all_native else ["GRBG"]
    remapped_cfas = [] if all_native else ["RGGB"]
    payload = {
        "status": "pass",
        "run_count": 2,
        "expected_run_count": 2,
        "cfa_patterns": ["GRBG", "RGGB"],
        "psf_sigmas": [0.0],
        "checks": [
            {"id": "sweep_rows_available", "status": "pass", "evidence": "runs=2"},
            {"id": "native_rows_identified", "status": "pass", "evidence": f"native_runs={native_run_count}"},
            {"id": "remapped_rows_separated", "status": "pass", "evidence": f"remapped_runs={remapped_run_count} partial_runs=0"},
        ],
        "groups": {
            "native": {"run_count": native_run_count, "sample_count": native_sample_count, "cfa_patterns": native_cfas, "psf_sigmas": [0.0]},
            "partial_remap": {"run_count": 0, "sample_count": 0, "cfa_patterns": [], "psf_sigmas": []},
            "remapped": {"run_count": remapped_run_count, "sample_count": remapped_sample_count, "cfa_patterns": remapped_cfas, "psf_sigmas": [0.0]},
        },
        "runs": [],
        "interpretation": "unit native audit",
        "claim_boundary": "unit native boundary",
    }
    (path / "cfa_lenspsf_native_audit_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_lenspsf_casebook(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "condition_count": 2,
        "expected_condition_count": 2,
        "selected_case_count": 5,
        "cfa_patterns": ["GRBG", "RGGB"],
        "psf_sigmas": [0.0],
        "checks": [
            {"id": "condition_casebooks_available", "status": "pass", "evidence": "conditions=2 expected=2"},
            {"id": "casebook_covers_cfa_psf_grid", "status": "pass", "evidence": "cfa=GRBG,RGGB psf=0.0000"},
            {"id": "casebook_uses_native_cfa_rows", "status": "pass", "evidence": "native=2/2"},
            {"id": "casebook_has_selected_cases", "status": "pass", "evidence": "selected_cases=5"},
            {"id": "casebook_includes_fp_reduction_successes", "status": "pass", "evidence": "selected_successes=3"},
            {"id": "casebook_includes_counterexamples", "status": "pass", "evidence": "selected_counterexamples=2"},
        ],
        "category_totals": {
            "fp_reduction_success": {"case_count": 8, "selected_case_count": 3},
            "recall_tradeoff": {"case_count": 1, "selected_case_count": 1},
            "recall_loss_failure": {"case_count": 1, "selected_case_count": 1},
            "fp_regression_failure": {"case_count": 0, "selected_case_count": 0},
        },
        "conditions": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "selected_case_count": 3,
                "pattern_remapped_fraction": 0.0,
                "true_sensor_cfa_mosaic_fraction": 1.0,
            },
            {
                "run_id": "cfa-rggb_psf-0p00",
                "cfa_pattern": "RGGB",
                "psf_sigma": 0.0,
                "selected_case_count": 2,
                "pattern_remapped_fraction": 0.0,
                "true_sensor_cfa_mosaic_fraction": 1.0,
            },
        ],
    }
    (path / "cfa_lenspsf_casebook_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_casebook(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "sample_count": 3,
        "selected_case_count": 4,
        "baseline_input": "human_rgb",
        "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
        "aggregate": {"tp_delta_count": 0, "fp_delta_count": -3},
        "checks": [
            {"id": "casebook_has_selected_cases", "status": "pass", "evidence": "selected_cases=4"},
            {"id": "casebook_includes_fp_reduction_successes", "status": "pass", "evidence": "selected_successes=1"},
            {"id": "casebook_includes_counterexamples", "status": "pass", "evidence": "selected_counterexamples=3"},
        ],
        "categories": {
            "fp_reduction_success": {"case_count": 1, "selected_case_count": 1, "cases": [{"sample_id": "success"}]},
            "recall_tradeoff": {"case_count": 1, "selected_case_count": 1, "cases": [{"sample_id": "tradeoff"}]},
            "recall_loss_failure": {"case_count": 1, "selected_case_count": 1, "cases": [{"sample_id": "recall_loss"}]},
            "fp_regression_failure": {"case_count": 1, "selected_case_count": 1, "cases": [{"sample_id": "fp_regression"}]},
        },
        "interpretation": "unit casebook",
        "claim_boundary": "unit casebook boundary",
    }
    (path / "casebook_summary.json").write_text(json.dumps(payload) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
