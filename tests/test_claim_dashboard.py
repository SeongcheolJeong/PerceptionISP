from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.claim_dashboard import build_claim_dashboard, main as dashboard_main, write_claim_dashboard


class ClaimDashboardTest(unittest.TestCase):
    def test_dashboard_separates_supported_and_blocked_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = _write_claim_gate(root / "broad", profile="broad_superiority", passed=False)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            training = _write_training_rollup(root / "training")
            task_metrics = _write_task_metrics(root / "task_metrics")
            protocol = _write_protocol_coverage(root / "protocol")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_edge_sweep = _write_scene_edge_confidence(root / "scene_edge_sweep", cfa_pattern="RGGB", psf_sigmas=(0.0, 1.0))
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")
            comparison = _write_comparison_rollup(root / "rollup")

            dashboard = build_claim_dashboard(
                claim_gate_specs=[f"Human superiority={broad}", f"FP reducer={fp}"],
                training_rollup=training,
                task_metrics=task_metrics,
                protocol_coverage=protocol,
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                edge_fidelity_suite=edge_fidelity,
                scene_edge_confidence=[scene_edge, scene_edge_sweep],
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                comparison_rollup_specs=[f"Calibration={comparison}"],
            )

            self.assertEqual(len(dashboard["claims"]), 2)
            statuses = [item["status"] for item in dashboard["decisions"]]
            self.assertIn("supported", statuses)
            self.assertIn("not_supported", statuses)
            self.assertEqual(dashboard["training"]["status"], "diagnostic_only")
            self.assertEqual(dashboard["task_metrics"]["status"], "recall_tradeoff")
            self.assertEqual(dashboard["protocol_coverage"]["status"], "not_claim_ready")
            self.assertTrue(dashboard["mechanism_validation"]["pass"])
            self.assertTrue(dashboard["cfa_stress_sweep"]["pass"])
            self.assertTrue(dashboard["edge_confidence_suite"]["pass"])
            self.assertTrue(dashboard["edge_fidelity_suite"]["pass"])
            self.assertTrue(dashboard["scene_edge_confidence"]["pass"])
            self.assertEqual(dashboard["scene_edge_confidence"]["report_count"], 2)
            self.assertEqual(dashboard["scene_edge_confidence"]["cfa_patterns"], ["GRBG", "RGGB"])
            self.assertAlmostEqual(dashboard["scene_edge_confidence"]["perception_rgb_minus_human_source_edge_f1_mean"], 0.01)
            self.assertAlmostEqual(dashboard["scene_edge_confidence"]["perception_aux_strength_source_edge_f1_win_rate"], 1.0)
            self.assertTrue(dashboard["scene_information_stress"]["pass"])
            self.assertTrue(dashboard["aux_contribution_audit"]["pass"])
            self.assertEqual(dashboard["comparison_rollups"][0]["name"], "Calibration")
            self.assertIn(
                "Task-level VRU/person recall improvement versus HumanISP is not supported; the current evidence supports only the narrower FP-reduction claim.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "PerceptionISP front-end mechanism validation passed; aux/confidence maps respond to controlled sensor stressors.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "CFA stress sweep is available as diagnostic evidence for condition-dependent front-end signals; it is not detector-performance evidence.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Edge-confidence suite passed; PerceptionISP confidence maps respond to difficult-edge stressors, but this is not detector-performance evidence.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Object edge-fidelity suite passed; HumanISP, PerceptionISP, and aux edge maps are compared against object/sensor edge oracles across CFA and LensPSF, but this is not detector-performance evidence.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Scene edge-confidence suite passed; HumanISP RGB, PerceptionISP RGB, aux edge-strength, and aux edge-confidence are compared against a high-information scene-edge proxy, but this is not detector-performance evidence.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Scene-information stress suite passed; high-information scene loss and CFA/color uncertainty are covered as diagnostic evidence, not detector-performance evidence.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Aux contribution audit passed; aux features add proposal-scoring FP reduction within the recall budget, but this is calibration evidence rather than DNN performance.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertTrue(
                any(
                    item["claim"].startswith("Benchmark protocol coverage is incomplete")
                    and item["status"] == "not_supported"
                    for item in dashboard["decisions"]
                )
            )

            html_path = write_claim_dashboard(dashboard, root / "dashboard")
            html = html_path.read_text()
            self.assertIn("PerceptionISP Claim Readiness Dashboard", html)
            self.assertIn("Broad HumanISP superiority gate failed", html)
            self.assertIn("Recall-budgeted FP-reduction gate passed", html)
            self.assertIn("Task Metrics", html)
            self.assertIn("Aux Contribution Audit", html)
            self.assertIn("Mechanism Validation", html)
            self.assertIn("CFA Stress Sweep", html)
            self.assertIn("Edge Confidence Suite", html)
            self.assertIn("Object Edge Fidelity", html)
            self.assertIn("Scene Edge Confidence", html)
            self.assertIn("Evidence Report", html)
            self.assertIn("RGB Delta", html)
            self.assertIn("Scene Information Stress", html)
            self.assertIn("Benchmark Protocol Coverage", html)
            self.assertIn("recall_tradeoff", html)
            self.assertTrue((html_path.parent / "claim_dashboard_summary.json").exists())

    def test_dashboard_cli_outputs_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            task_metrics = _write_task_metrics(root / "task_metrics")
            protocol = _write_protocol_coverage(root / "protocol")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_edge_sweep = _write_scene_edge_confidence(root / "scene_edge_sweep", cfa_pattern="RGGB", psf_sigmas=(0.0, 1.0))
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = dashboard_main(
                    [
                        "--claim-gate",
                        str(fp),
                        "--task-metrics",
                        str(task_metrics),
                        "--protocol-coverage",
                        str(protocol),
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
                        "--scene-edge-confidence",
                        str(scene_edge_sweep),
                        "--scene-information-stress",
                        str(scene_information),
                        "--aux-contribution-audit",
                        str(aux_contribution),
                        "--output-dir",
                        str(root / "dashboard"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["claim_count"], 1)
            summary = json.loads((root / "dashboard" / "claim_dashboard_summary.json").read_text())
            self.assertEqual(summary["task_metrics"]["status"], "recall_tradeoff")
            self.assertEqual(summary["protocol_coverage"]["status"], "not_claim_ready")
            self.assertTrue(summary["mechanism_validation"]["pass"])
            self.assertTrue(summary["cfa_stress_sweep"]["pass"])
            self.assertTrue(summary["edge_confidence_suite"]["pass"])
            self.assertTrue(summary["edge_fidelity_suite"]["pass"])
            self.assertTrue(summary["scene_edge_confidence"]["pass"])
            self.assertEqual(summary["scene_edge_confidence"]["report_count"], 2)
            self.assertTrue(summary["scene_information_stress"]["pass"])
            self.assertTrue(summary["aux_contribution_audit"]["pass"])
            self.assertTrue((root / "dashboard" / "claim_dashboard_summary.json").exists())

    def test_dashboard_uses_task_gate_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True, baseline_input="human_rgb")
            task_metrics = _write_task_metrics(root / "task_metrics")
            task_gate = _write_task_gate(root / "task_gate")

            dashboard = build_claim_dashboard(
                claim_gate_specs=[str(fp)],
                task_metrics=task_metrics,
                task_gate=task_gate,
            )

            self.assertEqual(dashboard["task_gate"]["verdict"], "task_gate_fail")
            decisions = {item["claim"]: item["status"] for item in dashboard["decisions"]}
            self.assertEqual(
                decisions["Task-level `recall_improvement` gate failed for vru, person; do not promote that task-level claim."],
                "not_supported",
            )
            html_path = write_claim_dashboard(dashboard, root / "dashboard")
            self.assertIn("Task Gate", html_path.read_text())

    def test_fp_reducer_decision_names_human_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = _write_claim_gate(root / "fp_human", profile="fp_reducer", passed=True, baseline_input="human_rgb")
            dashboard = build_claim_dashboard(claim_gate_specs=[str(fp)])

            decisions = {item["claim"]: item["status"] for item in dashboard["decisions"]}
            self.assertEqual(decisions["Recall-budgeted FP reduction versus HumanISP is supported."], "supported")


def _write_claim_gate(path: Path, *, profile: str, passed: bool, baseline_input: str | None = None) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    criteria = [
        _criterion("precision@0.50_mean", 0.03, 0.0, passed=True),
        _criterion("recall@0.50_mean", -0.01 if not passed else -0.005, 0.0 if profile == "broad_superiority" else -0.01, passed=passed),
        _criterion("small_recall@0.50_mean", -0.002, 0.0 if profile == "broad_superiority" else -0.01, passed=passed),
        _criterion("fp@0.50_mean", -0.3, 0.0 if profile == "broad_superiority" else -0.10, passed=True),
        {"metric": "sample_count", "pass": True, "target": 1000, "threshold": 1000},
    ]
    (path / "claim_gate_summary.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "verdict": "metric_gate_pass" if passed else "metric_gate_fail",
                "pass": passed,
                "sample_count": 1000,
                "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                "baseline_input": baseline_input or ("human_rgb" if profile == "broad_superiority" else "perception_fusion_rgb_aux"),
                "criteria": criteria,
                "interpretation": "unit",
            }
        )
        + "\n"
    )
    return path


def _criterion(metric: str, delta: float, threshold: float, *, passed: bool) -> dict:
    return {
        "metric": metric,
        "delta": delta,
        "threshold": threshold,
        "pass": passed,
        "paired_delta": {"ci_low": delta - 0.001, "ci_high": delta + 0.001},
    }


def _write_training_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "training_rollup_summary.json").write_text(
        json.dumps(
            {
                "run_count": 2,
                "runs": [
                    {
                        "name": "train",
                        "kind": "train_dense",
                        "sample_count": 128,
                        "epochs": 5,
                        "channel_mode": "rgb_aux",
                        "elapsed_seconds": 8.0,
                        "throughput": 60.0,
                    },
                    {
                        "name": "eval",
                        "kind": "dense_eval",
                        "sample_count": 32,
                        "channel_mode": "rgb_aux",
                        "metrics": {
                            "precision@0.50_mean": 0.01,
                            "recall@0.50_mean": 0.09,
                            "fp@0.50_mean": 40.0,
                            "det_count_mean": 41.0,
                        },
                    },
                ],
            }
        )
        + "\n"
    )
    return path


def _write_task_metrics(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "task_metrics_summary.json").write_text(
        json.dumps(
            {
                "baseline_input": "human_rgb",
                "inputs": [
                    "human_rgb",
                    "perception_fusion_rgb_aux",
                    "perception_calibrated_score_label_aux_fusion_rgb_aux",
                ],
                "groups": [
                    {"name": "vru", "kind": "label"},
                    {"name": "person", "kind": "label"},
                    {"name": "vehicle", "kind": "label"},
                    {"name": "small_all", "kind": "area"},
                ],
                "label_agnostic": True,
                "metrics": {
                    "human_rgb": {
                        "vru": {"recall@0.50": 0.30, "fp@0.50_per_sample": 0.24},
                        "person": {"recall@0.50": 0.40, "fp@0.50_per_sample": 0.18},
                    },
                    "perception_calibrated_score_label_aux_fusion_rgb_aux": {
                        "vru": {
                            "gt_count": 100,
                            "det_count": 70,
                            "precision@0.50": 0.59,
                            "recall@0.50": 0.29,
                            "recall@0.75": 0.08,
                            "fp@0.50_per_sample": 0.16,
                            "delta_precision@0.50": 0.09,
                            "delta_recall@0.50": -0.01,
                            "delta_recall@0.75": -0.004,
                            "delta_fp@0.50_per_sample": -0.08,
                        },
                        "person": {
                            "gt_count": 80,
                            "det_count": 60,
                            "precision@0.50": 0.60,
                            "recall@0.50": 0.38,
                            "recall@0.75": 0.10,
                            "fp@0.50_per_sample": 0.15,
                            "delta_precision@0.50": 0.02,
                            "delta_recall@0.50": -0.02,
                            "delta_recall@0.75": -0.006,
                            "delta_fp@0.50_per_sample": -0.03,
                        },
                        "vehicle": {
                            "gt_count": 120,
                            "det_count": 100,
                            "precision@0.50": 0.72,
                            "recall@0.50": 0.50,
                            "recall@0.75": 0.33,
                            "fp@0.50_per_sample": 0.85,
                            "delta_precision@0.50": 0.03,
                            "delta_recall@0.50": -0.01,
                            "delta_recall@0.75": -0.003,
                            "delta_fp@0.50_per_sample": -0.14,
                        },
                        "small_all": {
                            "gt_count": 90,
                            "det_count": 80,
                            "precision@0.50": 0.50,
                            "recall@0.50": 0.30,
                            "recall@0.75": 0.08,
                            "fp@0.50_per_sample": 0.80,
                            "delta_precision@0.50": 0.05,
                            "delta_recall@0.50": -0.002,
                            "delta_recall@0.75": -0.001,
                            "delta_fp@0.50_per_sample": -0.07,
                        },
                    },
                },
            }
        )
        + "\n"
    )
    return path


def _write_task_gate(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "task_gate_summary.json").write_text(
        json.dumps(
            {
                "profile": "recall_improvement",
                "verdict": "task_gate_fail",
                "pass": False,
                "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                "baseline_input": "human_rgb",
                "evaluated_group_count": 4,
                "failed_group_count": 2,
                "skipped_group_count": 0,
                "groups": [
                    {"group": "vru", "status": "fail"},
                    {"group": "person", "status": "fail"},
                    {"group": "vehicle", "status": "pass"},
                    {"group": "small_all", "status": "pass"},
                ],
                "interpretation": "unit",
            }
        )
        + "\n"
    )
    return path


def _write_protocol_coverage(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "protocol_coverage_summary.json").write_text(
        json.dumps(
            {
                "status": "not_claim_ready",
                "missing_required": ["held_out_scale"],
                "missing_raw_claim": ["naive_raw_baseline"],
                "requirements": [
                    {
                        "id": "held_out_scale",
                        "label": "Held-out sample scale",
                        "scope": "claim_required",
                        "status": "missing",
                        "evidence": "max report sample_count: 3",
                        "missing_reason": "The largest available held-out report is too small for a broad claim.",
                    }
                ],
                "interpretation": "Protocol evidence is incomplete.",
            }
        )
        + "\n"
    )
    return path


def _write_mechanism_validation(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "mechanism_validation_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "cfa_patterns": ["RGGB", "GRBG", "RCCB", "RGBIR"],
                "mechanisms": [
                    {"id": "low_light_noise_response", "status": "pass"},
                    {"id": "glare_saturation_response", "status": "pass"},
                    {"id": "low_mtf_edge_confidence_response", "status": "pass"},
                    {"id": "cfa_variant_support", "status": "pass"},
                ],
                "interpretation": "unit mechanism validation",
            }
        )
        + "\n"
    )
    return path


def _write_cfa_stress_sweep(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "cfa_stress_sweep_summary.json").write_text(
        json.dumps(
            {
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
                "interpretation": "unit cfa stress sweep",
            }
        )
        + "\n"
    )
    return path


def _write_edge_confidence_suite(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "edge_confidence_suite_summary.json").write_text(
        json.dumps(
            {
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
                "interpretation": "unit edge confidence suite",
            }
        )
        + "\n"
    )
    return path


def _write_edge_fidelity_suite(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "edge_fidelity_suite_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "cfa_patterns": ["RGGB", "GRBG", "RCCB"],
                "psf_sigmas": [0.0, 1.2],
                "cases": [
                    {
                        "id": "psf_0.00_RGGB",
                        "psf_sigma": 0.0,
                        "cfa_pattern": "RGGB",
                        "metrics": {
                            "human_object_edge_f1": 0.65,
                            "perception_object_edge_f1": 0.67,
                            "aux_object_edge_f1": 0.70,
                            "edge_confidence_separation": 0.12,
                        },
                    }
                ],
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
                "interpretation": "unit edge fidelity",
                "claim_boundary": "unit diagnostic boundary",
            }
        )
        + "\n"
    )
    return path


def _write_scene_edge_confidence(path: Path, *, cfa_pattern: str = "GRBG", psf_sigmas: tuple[float, ...] = (0.0,)) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "scene_edge_confidence_summary.json").write_text(
        json.dumps(
            {
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
                "interpretation": "unit scene edge confidence",
                "claim_boundary": "unit diagnostic boundary",
            }
        )
        + "\n"
    )
    return path


def _write_scene_information_stress(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "scene_information_stress_summary.json").write_text(
        json.dumps(
            {
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
                    {
                        "id": "subpixel_signal",
                        "sample_mode": "box",
                        "metrics": {
                            "scene_luma_gradient_p90": 0.0,
                            "sensor_luma_gradient_p90": 0.0,
                            "luma_detail_retention_p90": 0.0,
                            "scene_chroma_gradient_p90": 0.0,
                            "color_confidence_mean": 0.52,
                            "signal_contrast_retention": 0.09,
                        },
                    },
                ],
                "checks": [
                    {"id": "latent_high_frequency_detail_loss", "status": "pass"},
                    {"id": "cfa_chroma_alias_color_confidence_drop", "status": "pass"},
                    {"id": "subpixel_signal_fill_factor_loss", "status": "pass"},
                ],
                "interpretation": "unit scene-information stress",
                "claim_boundary": "unit diagnostic boundary",
            }
        )
        + "\n"
    )
    return path


def _write_aux_contribution_audit(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "aux_contribution_audit_summary.json").write_text(
        json.dumps(
            {
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
                "interpretation": "unit aux contribution audit",
            }
        )
        + "\n"
    )
    return path


def _write_comparison_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "rollup_summary.json").write_text(json.dumps({"run_count": 1, "baseline_input": "human_rgb", "runs": []}) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
