from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.benchmark_protocol import build_protocol_coverage, main as protocol_main, write_protocol_coverage


class BenchmarkProtocolTest(unittest.TestCase):
    def test_protocol_marks_missing_raw_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = _write_comparison_report(root / "comparison", tone_mapping="log", demosaic_method="edge_aware")
            training = _write_training_rollup(root / "training")
            gate = _write_claim_gate(root / "gate")
            task = _write_task_metrics(root / "task")
            task_gate = _write_task_gate(root / "task_gate")
            condition = _write_condition_metrics(root / "condition")
            condition_gate = _write_condition_gate(root / "condition_gate")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_edge_sweep = _write_scene_edge_confidence(root / "scene_edge_sweep", cfa_pattern="RGGB", psf_sigmas=(0.0, 1.0))
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")

            summary = build_protocol_coverage(
                comparison_reports=[report],
                training_rollup=training,
                claim_gates=[gate],
                task_metrics=task,
                task_gate=task_gate,
                condition_metrics=condition,
                condition_gate=condition_gate,
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                edge_fidelity_suite=edge_fidelity,
                scene_edge_confidence=[scene_edge, scene_edge_sweep],
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                min_samples=3,
            )

            rows = {row["id"]: row for row in summary["requirements"]}
            self.assertEqual(summary["status"], "not_claim_ready")
            self.assertEqual(summary["coverage_status"], "coverage_incomplete")
            self.assertEqual(summary["metric_claim_status"], "broad_superiority_not_supported")
            self.assertEqual(rows["paired_human_baseline"]["status"], "covered")
            self.assertEqual(rows["classical_lightweight_transform"]["status"], "covered")
            self.assertEqual(rows["front_end_mechanism_validation"]["status"], "covered")
            self.assertEqual(rows["cfa_stress_sweep"]["status"], "covered")
            self.assertEqual(rows["edge_confidence_suite"]["status"], "covered")
            self.assertEqual(rows["edge_fidelity_suite"]["status"], "covered")
            self.assertEqual(rows["scene_edge_confidence"]["status"], "covered")
            self.assertEqual(rows["scene_information_stress"]["status"], "covered")
            self.assertEqual(rows["aux_contribution_audit"]["status"], "covered")
            self.assertEqual(rows["naive_raw_baseline"]["status"], "missing")
            self.assertIn("naive_raw_baseline", summary["missing_raw_claim"])

            html_path = write_protocol_coverage(summary, root / "protocol")
            self.assertTrue((html_path.parent / "protocol_coverage_summary.json").exists())
            self.assertIn("Naive RAW", html_path.read_text())

    def test_protocol_passes_when_minimum_matrix_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classical = _write_comparison_report(root / "classical", tone_mapping="log", demosaic_method="edge_aware")
            naive = _write_comparison_report(root / "naive", tone_mapping="linear", demosaic_method="bilinear", denoise_strength=0.0)
            rollup = _write_comparison_rollup(root / "rollup", reports=[classical, naive])
            training = _write_training_rollup(root / "training")
            gate = _write_claim_gate(root / "gate")
            task = _write_task_metrics(root / "task")
            task_gate = _write_task_gate(root / "task_gate")
            condition = _write_condition_metrics(root / "condition")
            condition_gate = _write_condition_gate(root / "condition_gate")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_edge_sweep = _write_scene_edge_confidence(root / "scene_edge_sweep", cfa_pattern="RGGB", psf_sigmas=(0.0, 1.0))
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")

            summary = build_protocol_coverage(
                comparison_reports=[classical, naive],
                comparison_rollups=[rollup],
                training_rollup=training,
                claim_gates=[gate],
                task_metrics=task,
                task_gate=task_gate,
                condition_metrics=condition,
                condition_gate=condition_gate,
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                edge_fidelity_suite=edge_fidelity,
                scene_edge_confidence=[scene_edge, scene_edge_sweep],
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                min_samples=3,
            )

            self.assertEqual(summary["status"], "claim_ready")
            self.assertEqual(summary["coverage_status"], "coverage_complete")
            self.assertEqual(summary["metric_claim_status"], "broad_superiority_not_supported")
            self.assertEqual(summary["evidence"]["scene_edge_confidence"]["report_count"], 2)
            self.assertEqual(summary["evidence"]["scene_edge_confidence"]["case_count"], 2)
            self.assertEqual(summary["evidence"]["scene_edge_confidence"]["cfa_patterns"], ["GRBG", "RGGB"])
            self.assertAlmostEqual(summary["evidence"]["scene_edge_confidence"]["perception_rgb_minus_human_source_edge_f1_mean"], 0.01)
            self.assertAlmostEqual(summary["evidence"]["scene_edge_confidence"]["perception_aux_strength_source_edge_f1_win_rate"], 1.0)
            self.assertFalse(summary["claim_gate_outcomes"]["broad_superiority_pass"])
            self.assertEqual(summary["missing_required"], [])
            self.assertEqual(summary["missing_raw_claim"], [])

    def test_protocol_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = _write_comparison_report(root / "comparison", tone_mapping="log", demosaic_method="edge_aware")
            task = _write_task_metrics(root / "task")
            task_gate = _write_task_gate(root / "task_gate")
            condition = _write_condition_metrics(root / "condition")
            condition_gate = _write_condition_gate(root / "condition_gate")
            gate = _write_claim_gate(root / "gate")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = protocol_main(
                    [
                        "--comparison-report",
                        str(report),
                        "--claim-gate",
                        str(gate),
                        "--task-metrics",
                        str(task),
                        "--task-gate",
                        str(task_gate),
                        "--condition-metrics",
                        str(condition),
                        "--condition-gate",
                        str(condition_gate),
                        "--mechanism-validation",
                        str(mechanism),
                        "--cfa-stress-sweep",
                        str(cfa_stress),
                        "--edge-confidence-suite",
                        str(edge_confidence),
                        "--edge-fidelity-suite",
                        str(edge_fidelity),
                        "--scene-edge-confidence",
                        str(scene_edge),
                        "--scene-information-stress",
                        str(scene_information),
                        "--aux-contribution-audit",
                        str(aux_contribution),
                        "--min-samples",
                        "3",
                        "--output-dir",
                        str(root / "protocol"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "not_claim_ready")
            self.assertEqual(printed["coverage_status"], "coverage_incomplete")
            self.assertEqual(printed["metric_claim_status"], "broad_superiority_not_supported")
            self.assertIn("naive_raw_baseline", printed["missing_raw_claim"])
            self.assertTrue((root / "protocol" / "protocol_coverage_summary.json").exists())


def _write_comparison_report(path: Path, *, tone_mapping: str, demosaic_method: str, denoise_strength: float = 0.18) -> Path:
    path.mkdir()
    payload = {
        "sample_count": 3,
        "aggregate": {
            "human_rgb": {"precision@0.50_mean": 0.6, "recall@0.50_mean": 0.4, "sample_count": 3},
            "perception_rgb": {"precision@0.50_mean": 0.6, "recall@0.50_mean": 0.4, "sample_count": 3},
            "perception_fusion_rgb_aux": {"precision@0.50_mean": 0.7, "recall@0.50_mean": 0.4, "sample_count": 3},
        },
        "run_config": {
            "source": "kitti-dataset",
            "dataset": "KITTI",
            "split": "val",
            "count": 3,
            "rgb_detector": "yolo",
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


def _write_comparison_rollup(path: Path, *, reports: list[Path]) -> Path:
    path.mkdir()
    runs = []
    for report in reports:
        payload = json.loads((report / "comparison_summary.json").read_text())
        runs.append(
            {
                "name": report.name,
                "sample_count": payload["sample_count"],
                "run_config": payload["run_config"],
                "inputs": {name: {} for name in payload["aggregate"]},
            }
        )
    (path / "rollup_summary.json").write_text(json.dumps({"run_count": len(runs), "baseline_input": "human_rgb", "runs": runs}) + "\n")
    return path


def _write_training_rollup(path: Path) -> Path:
    path.mkdir()
    runs = [
        {"kind": "train_dense", "channel_mode": "rgb_aux", "tensor_key": "rgb_aux_extended_chw"},
        {"kind": "train_dense", "channel_mode": "rgb_only", "tensor_key": "rgb_aux_chw"},
        {"kind": "train_dense", "channel_mode": "aux_only", "tensor_key": "rgb_aux_chw"},
        {"kind": "dense_eval", "channel_mode": "rgb_aux", "tensor_key": "rgb_aux_extended_chw"},
    ]
    (path / "training_rollup_summary.json").write_text(json.dumps({"run_count": len(runs), "runs": runs}) + "\n")
    return path


def _write_claim_gate(path: Path) -> Path:
    path.mkdir()
    (path / "claim_gate_summary.json").write_text(
        json.dumps({"profile": "broad_superiority", "pass": False, "sample_count": 3, "thresholds": {"require_ci": True}}) + "\n"
    )
    return path


def _write_task_metrics(path: Path) -> Path:
    path.mkdir()
    payload = {
        "inputs": ["human_rgb", "perception_fusion_rgb_aux"],
        "groups": [{"name": "person"}],
        "label_agnostic": False,
        "metrics": {},
    }
    (path / "task_metrics_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_task_gate(path: Path) -> Path:
    path.mkdir()
    payload = {
        "profile": "recall_improvement",
        "pass": False,
        "verdict": "task_gate_fail",
        "evaluated_group_count": 1,
        "failed_group_count": 1,
        "skipped_group_count": 0,
    }
    (path / "task_gate_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_condition_metrics(path: Path) -> Path:
    path.mkdir()
    payload = {
        "inputs": ["human_rgb", "perception_fusion_rgb_aux"],
        "conditions": [{"name": "all", "sample_count": 3}, {"name": "low_light_proxy", "sample_count": 1}],
        "label_agnostic": False,
        "metrics": {},
    }
    (path / "condition_metrics_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_condition_gate(path: Path) -> Path:
    path.mkdir()
    payload = {
        "profile": "fp_reducer",
        "pass": True,
        "verdict": "condition_gate_pass",
        "evaluated_condition_count": 1,
        "failed_condition_count": 0,
        "skipped_condition_count": 1,
    }
    (path / "condition_gate_summary.json").write_text(json.dumps(payload) + "\n")
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


if __name__ == "__main__":
    unittest.main()
