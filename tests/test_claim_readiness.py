from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.claim_readiness import main as readiness_main, run_claim_readiness


class ClaimReadinessTest(unittest.TestCase):
    def test_run_claim_readiness_builds_gates_training_rollup_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            train_dir = _write_train_summary(root / "train")
            eval_dir = _write_eval_summary(root / "eval")
            rgb_aux_dnn_gate = _write_rgb_aux_dnn_gate(root / "rgb_aux_dnn_gate", passed=False)
            rgb_aux_dnn_sweep = _write_rgb_aux_dnn_sweep(root / "rgb_aux_dnn_sweep", passed=False)
            dense_input_ablation = _write_dense_input_ablation_gate(root / "dense_input_ablation", passed=True)
            rollup_dir = _write_comparison_rollup(root / "rollup")
            mechanism = _write_mechanism_validation(root / "mechanism")
            cfa_stress = _write_cfa_stress_sweep(root / "cfa_stress")
            edge_confidence = _write_edge_confidence_suite(root / "edge_confidence")
            edge_fidelity = _write_edge_fidelity_suite(root / "edge_fidelity")
            object_boundary = _write_object_boundary_edge(root / "object_boundary")
            object_boundary_bridge = _write_object_boundary_detection_bridge(root / "object_boundary_bridge")
            detector_box_support = _write_detector_box_support_audit(root / "detector_box_support")
            scene_edge = _write_scene_edge_confidence(root / "scene_edge")
            scene_edge_sweep = _write_scene_edge_confidence(root / "scene_edge_sweep", cfa_pattern="RGGB", psf_sigmas=(0.0, 1.0))
            scene_information = _write_scene_information_stress(root / "scene_information")
            aux_contribution = _write_aux_contribution_audit(root / "aux_contribution")
            adverse_native = _write_adverse_native_slice(root / "adverse_native")
            adverse_task = _write_adverse_task_slice(root / "adverse_task")
            cfa_lenspsf_proposal = _write_cfa_lenspsf_proposal_audit(root / "cfa_lenspsf_proposal")
            cfa_lenspsf_native = _write_cfa_lenspsf_native_audit(root / "cfa_lenspsf_native")
            cfa_lenspsf_casebook = _write_cfa_lenspsf_casebook(root / "cfa_lenspsf_casebook")
            cfa_lenspsf_aux_ablation = _write_cfa_lenspsf_aux_ablation(root / "cfa_lenspsf_aux_ablation")
            casebook = _write_casebook(root / "casebook")

            summary = run_claim_readiness(
                comparison_report=report_dir,
                min_samples=3,
                bootstrap_samples=16,
                bootstrap_seed="unit",
                training_summaries=[train_dir, eval_dir],
                rgb_aux_dnn_gate=rgb_aux_dnn_gate,
                rgb_aux_dnn_sweep=rgb_aux_dnn_sweep,
                dense_input_ablation_gate=dense_input_ablation,
                comparison_rollups=[f"Calibration={rollup_dir}"],
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                edge_fidelity_suite=edge_fidelity,
                object_boundary_edge=object_boundary,
                object_boundary_detection_bridge=object_boundary_bridge,
                detector_box_support_audit=detector_box_support,
                scene_edge_confidence=[scene_edge, scene_edge_sweep],
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                adverse_native_slice=adverse_native,
                adverse_task_slice=adverse_task,
                cfa_lenspsf_proposal_audit=cfa_lenspsf_proposal,
                cfa_lenspsf_native_audit=cfa_lenspsf_native,
                cfa_lenspsf_casebook=cfa_lenspsf_casebook,
                cfa_lenspsf_aux_ablation=cfa_lenspsf_aux_ablation,
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
            self.assertIn("object_boundary_edge", summary)
            self.assertIn("object_boundary_detection_bridge", summary)
            self.assertIn("detector_box_support_audit", summary)
            self.assertIn("scene_edge_confidence", summary)
            self.assertIn("scene_information_stress", summary)
            self.assertIn("aux_contribution_audit", summary)
            self.assertIn("rgb_aux_dnn_gate", summary)
            self.assertIn("rgb_aux_dnn_sweep", summary)
            self.assertIn("dense_input_ablation_gate", summary)
            self.assertIn("adverse_native_slice", summary)
            self.assertIn("adverse_task_slice", summary)
            self.assertIn("cfa_lenspsf_proposal_audit", summary)
            self.assertIn("cfa_lenspsf_native_audit", summary)
            self.assertIn("cfa_lenspsf_casebook", summary)
            self.assertIn("cfa_lenspsf_aux_ablation", summary)
            self.assertIn("casebook", summary)
            self.assertIn("benchmark_protocol", summary)
            self.assertEqual(summary["benchmark_protocol"]["status"], "not_claim_ready")
            self.assertEqual(summary["benchmark_protocol"]["coverage_status"], "coverage_incomplete")
            self.assertEqual(summary["benchmark_protocol"]["metric_claim_status"], "fp_reducer_only")
            self.assertTrue(summary["scene_information_stress"]["pass"])
            self.assertTrue(summary["scene_edge_confidence"]["pass"])
            self.assertEqual(summary["scene_edge_confidence"]["report_count"], 2)
            self.assertFalse(summary["rgb_aux_dnn_gate"]["pass"])
            self.assertEqual(summary["rgb_aux_dnn_gate"]["claim_status"], "rgb_aux_dnn_not_claim_ready")
            self.assertFalse(summary["rgb_aux_dnn_sweep"]["pass"])
            self.assertEqual(summary["rgb_aux_dnn_sweep"]["claim_status"], "rgb_aux_dnn_sweep_no_claim_operating_point")
            self.assertTrue(summary["dense_input_ablation_gate"]["pass"])
            self.assertEqual(summary["dense_input_ablation_gate"]["claim_status"], "aux_input_used_by_dense_dnn")
            self.assertTrue(summary["adverse_native_slice"]["pass"])
            self.assertEqual(summary["adverse_native_slice"]["claim_status"], "adverse_fp_reducer_supported")
            self.assertTrue(summary["adverse_task_slice"]["pass"])
            self.assertEqual(summary["adverse_task_slice"]["claim_status"], "adverse_task_gate_partially_supported")
            self.assertEqual(summary["adverse_task_slice"]["adverse_passed_condition_count"], 4)
            self.assertTrue(summary["cfa_lenspsf_proposal_audit"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_proposal_audit"]["removed_fp_count"], 5)
            self.assertTrue(summary["cfa_lenspsf_native_audit"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_native_audit"]["native_run_count"], 1)
            self.assertTrue(summary["cfa_lenspsf_casebook"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_casebook"]["selected_case_count"], 5)
            self.assertTrue(summary["cfa_lenspsf_aux_ablation"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_aux_ablation"]["claim_status"], "aux_recall_fp_tradeoff")
            self.assertTrue(summary["casebook"]["pass"])
            self.assertEqual(summary["casebook"]["selected_case_count"], 4)
            self.assertEqual(summary["scene_edge_confidence"]["cfa_patterns"], ["GRBG", "RGGB"])
            self.assertAlmostEqual(summary["scene_edge_confidence"]["perception_rgb_minus_human_source_edge_f1_mean"], 0.01)
            self.assertAlmostEqual(summary["scene_edge_confidence"]["perception_aux_strength_source_edge_f1_win_rate"], 1.0)
            self.assertAlmostEqual(summary["scene_edge_confidence"]["perception_aux_confidence_minus_human_source_edge_f1_mean"], -0.29)
            self.assertAlmostEqual(summary["scene_edge_confidence"]["perception_aux_confidence_source_edge_f1_win_rate"], 0.0)
            self.assertTrue(summary["edge_fidelity_suite"]["pass"])
            self.assertTrue(summary["object_boundary_edge"]["pass"])
            self.assertEqual(summary["object_boundary_edge"]["claim_status"], "object_boundary_edge_diagnostic")
            self.assertTrue(summary["object_boundary_detection_bridge"]["pass"])
            self.assertEqual(summary["object_boundary_detection_bridge"]["claim_status"], "object_boundary_detection_bridge_diagnostic")
            self.assertTrue(summary["detector_box_support_audit"]["pass"])
            self.assertEqual(summary["detector_box_support_audit"]["claim_status"], "detector_box_support_diagnostic")
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
            self.assertTrue(dashboard_summary["object_boundary_edge"]["pass"])
            self.assertTrue(dashboard_summary["object_boundary_detection_bridge"]["pass"])
            self.assertTrue(dashboard_summary["detector_box_support_audit"]["pass"])
            self.assertTrue(dashboard_summary["scene_edge_confidence"]["pass"])
            self.assertEqual(dashboard_summary["scene_edge_confidence"]["report_count"], 2)
            self.assertTrue(dashboard_summary["scene_information_stress"]["pass"])
            self.assertTrue(dashboard_summary["aux_contribution_audit"]["pass"])
            self.assertFalse(dashboard_summary["rgb_aux_dnn_gate"]["pass"])
            self.assertEqual(dashboard_summary["rgb_aux_dnn_gate"]["claim_status"], "rgb_aux_dnn_not_claim_ready")
            self.assertFalse(dashboard_summary["rgb_aux_dnn_sweep"]["pass"])
            self.assertEqual(dashboard_summary["rgb_aux_dnn_sweep"]["claim_status"], "rgb_aux_dnn_sweep_no_claim_operating_point")
            self.assertTrue(dashboard_summary["dense_input_ablation_gate"]["pass"])
            self.assertEqual(dashboard_summary["dense_input_ablation_gate"]["claim_status"], "aux_input_used_by_dense_dnn")
            self.assertTrue(dashboard_summary["adverse_native_slice"]["pass"])
            self.assertTrue(dashboard_summary["adverse_task_slice"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_proposal_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_native_audit"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_casebook"]["pass"])
            self.assertTrue(dashboard_summary["cfa_lenspsf_aux_ablation"]["pass"])
            self.assertTrue(dashboard_summary["casebook"]["pass"])
            self.assertEqual(dashboard_summary["protocol_coverage"]["status"], "not_claim_ready")
            self.assertEqual(dashboard_summary["protocol_coverage"]["coverage_status"], "coverage_incomplete")
            self.assertEqual(dashboard_summary["protocol_coverage"]["metric_claim_status"], "fp_reducer_only")

    def test_claim_readiness_cli_outputs_compact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = _write_comparison_report(root / "comparison")
            adverse_native = _write_adverse_native_slice(root / "adverse_native")
            adverse_task = _write_adverse_task_slice(root / "adverse_task")
            cfa_lenspsf_aux_ablation = _write_cfa_lenspsf_aux_ablation(root / "cfa_lenspsf_aux_ablation")
            rgb_aux_dnn_gate = _write_rgb_aux_dnn_gate(root / "rgb_aux_dnn_gate", passed=False)
            rgb_aux_dnn_sweep = _write_rgb_aux_dnn_sweep(root / "rgb_aux_dnn_sweep", passed=False)
            dense_select_test = _write_dense_select_test_gate(root / "dense_select_test", passed=True)
            dense_input_ablation = _write_dense_input_ablation_gate(root / "dense_input_ablation", passed=True)
            object_boundary = _write_object_boundary_edge(root / "object_boundary")
            object_boundary_bridge = _write_object_boundary_detection_bridge(root / "object_boundary_bridge")
            detector_box_support = _write_detector_box_support_audit(root / "detector_box_support")
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
                        "--adverse-native-slice",
                        str(adverse_native),
                        "--adverse-task-slice",
                        str(adverse_task),
                        "--rgb-aux-dnn-gate",
                        str(rgb_aux_dnn_gate),
                        "--rgb-aux-dnn-sweep",
                        str(rgb_aux_dnn_sweep),
                        "--dense-select-test-gate",
                        str(dense_select_test),
                        "--dense-input-ablation-gate",
                        str(dense_input_ablation),
                        "--object-boundary-edge",
                        str(object_boundary),
                        "--object-boundary-detection-bridge",
                        str(object_boundary_bridge),
                        "--detector-box-support-audit",
                        str(detector_box_support),
                        "--cfa-lenspsf-aux-ablation",
                        str(cfa_lenspsf_aux_ablation),
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
            self.assertFalse(printed["rgb_aux_dnn_gate"]["pass"])
            self.assertFalse(printed["rgb_aux_dnn_sweep"]["pass"])
            self.assertTrue(printed["dense_select_test_gate"]["pass"])
            self.assertEqual(printed["dense_select_test_gate"]["claim_status"], "strict_fair_tuned_rgb_baseline_heldout_pass")
            self.assertTrue(printed["dense_input_ablation_gate"]["pass"])
            self.assertEqual(printed["dense_input_ablation_gate"]["claim_status"], "aux_input_used_by_dense_dnn")
            self.assertTrue(printed["object_boundary_edge"]["pass"])
            self.assertEqual(printed["object_boundary_edge"]["claim_status"], "object_boundary_edge_diagnostic")
            self.assertTrue(printed["object_boundary_detection_bridge"]["pass"])
            self.assertEqual(printed["object_boundary_detection_bridge"]["claim_status"], "object_boundary_detection_bridge_diagnostic")
            self.assertTrue(printed["detector_box_support_audit"]["pass"])
            self.assertEqual(printed["detector_box_support_audit"]["claim_status"], "detector_box_support_diagnostic")
            self.assertTrue(printed["adverse_native_slice"]["pass"])
            self.assertTrue(printed["adverse_task_slice"]["pass"])
            self.assertEqual(printed["adverse_task_slice"]["claim_status"], "adverse_task_gate_partially_supported")
            self.assertEqual(printed["cfa_lenspsf_aux_ablation"]["claim_status"], "aux_recall_fp_tradeoff")
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


def _write_rgb_aux_dnn_gate(path: Path, *, passed: bool) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    criteria = [
        _dnn_gate_criterion("sample_count", "fail" if not passed else "pass", target=32 if not passed else 1200, threshold=1000),
        _dnn_gate_criterion("absolute_precision", "pass", target=0.06, threshold=0.05),
        _dnn_gate_criterion("absolute_recall", "fail" if not passed else "pass", target=0.09 if not passed else 0.20, threshold=0.10),
        _dnn_gate_criterion("absolute_fp_per_sample", "fail" if not passed else "pass", target=48.0 if not passed else 3.0, threshold=5.0),
        _dnn_gate_criterion("precision_vs_rgb_only", "pass", target=0.06, baseline=0.05, delta=0.01, threshold=0.0),
        _dnn_gate_criterion("recall_vs_rgb_only", "pass", target=0.09 if not passed else 0.20, baseline=0.08, delta=0.01 if not passed else 0.12, threshold=0.0),
        _dnn_gate_criterion("small_recall_vs_rgb_only", "fail" if not passed else "pass", target=0.02, baseline=0.03 if not passed else 0.01, delta=-0.01 if not passed else 0.01, threshold=0.0),
        _dnn_gate_criterion("fp_vs_rgb_only", "fail" if not passed else "pass", target=48.0 if not passed else 3.0, baseline=40.0 if not passed else 4.0, delta=8.0 if not passed else -1.0, threshold=0.0),
    ]
    (path / "rgb_aux_dnn_gate_summary.json").write_text(
        json.dumps(
            {
                "status": "pass" if passed else "fail",
                "pass": passed,
                "claim_status": "rgb_aux_dnn_claim_ready" if passed else "rgb_aux_dnn_not_claim_ready",
                "profile": "claim_quality",
                "primary_run": "rgb_aux",
                "baseline_run": "rgb_only",
                "runs": [
                    {
                        "name": "rgb_aux",
                        "sample_count": 32 if not passed else 1200,
                        "channel_mode": "rgb_aux",
                        "tensor_key": "rgb_aux_chw",
                        "input_channels": 6,
                        "precision@0.50_mean": 0.06,
                        "recall@0.50_mean": 0.09 if not passed else 0.20,
                        "small_recall@0.50_mean": 0.02,
                        "fp@0.50_mean": 48.0 if not passed else 3.0,
                    },
                    {
                        "name": "rgb_only",
                        "sample_count": 32 if not passed else 1200,
                        "channel_mode": "rgb_only",
                        "tensor_key": "rgb_chw",
                        "input_channels": 3,
                        "precision@0.50_mean": 0.05,
                        "recall@0.50_mean": 0.08,
                        "small_recall@0.50_mean": 0.03 if not passed else 0.01,
                        "fp@0.50_mean": 40.0 if not passed else 4.0,
                    },
                ],
                "deltas": {
                    "precision@0.50_mean": 0.01,
                    "recall@0.50_mean": 0.01 if not passed else 0.12,
                    "small_recall@0.50_mean": -0.01 if not passed else 0.01,
                    "fp@0.50_mean": 8.0 if not passed else -1.0,
                },
                "criteria": criteria,
                "interpretation": "unit RGB+Aux DNN gate",
                "claim_boundary": "unit compact-DNN boundary",
            }
        )
        + "\n"
    )
    return path


def _dnn_gate_criterion(
    identifier: str,
    status: str,
    *,
    target: float,
    threshold: float,
    baseline: float | None = None,
    delta: float | None = None,
) -> dict:
    row = {
        "id": identifier,
        "status": status,
        "pass": status == "pass",
        "target": target,
        "threshold": threshold,
    }
    if baseline is not None:
        row["baseline"] = baseline
    if delta is not None:
        row["delta"] = delta
    return row


def _write_rgb_aux_dnn_sweep(path: Path, *, passed: bool) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    rows = [
        _dnn_sweep_row(
            confidence=0.50,
            metric_pass=False,
            aux=(0.02, 0.16, 0.05, 40.0),
            rgb=(0.01, 0.10, 0.02, 80.0),
            failed=("absolute_precision", "absolute_fp_per_sample"),
        ),
        _dnn_sweep_row(
            confidence=0.93,
            metric_pass=passed,
            aux=(0.06, 0.04 if not passed else 0.12, 0.02, 4.0),
            rgb=(0.01, 0.02, 0.01, 20.0),
            failed=() if passed else ("absolute_recall", "small_recall_vs_rgb_only"),
        ),
    ]
    (path / "rgb_aux_dnn_sweep_summary.json").write_text(
        json.dumps(
            {
                "status": "pass" if passed else "fail",
                "pass": passed,
                "metric_pass": passed,
                "claim_status": "rgb_aux_dnn_sweep_claim_ready" if passed else "rgb_aux_dnn_sweep_no_claim_operating_point",
                "profile": "claim_quality",
                "row_count": len(rows),
                "rows": rows,
                "best_passing_row": rows[1] if passed else None,
                "best_metric_row": rows[1] if passed else None,
                "best_recall_positive_delta_row": rows[0],
                "lowest_fp_positive_recall_delta_row": rows[1],
                "interpretation": "unit RGB+Aux DNN sweep",
                "claim_boundary": "unit confidence-sweep boundary",
            }
        )
        + "\n"
    )
    return path


def _write_dense_select_test_gate(path: Path, *, passed: bool) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass" if passed else "mixed",
                "claim_status": "strict_fair_tuned_rgb_baseline_heldout_pass" if passed else "strict_fair_tuned_rgb_baseline_mixed",
                "pass_test_seed_count": 3 if passed else 2,
                "seed_count": 3,
                "source_eval_count": 395,
                "selection_sample_count": 197,
                "test_sample_count": 198,
                "candidate_epochs": [0, 1, 2, 3],
                "candidate_thresholds": [0.12, 0.20, 0.28],
                "candidate_max_detections": [20, 30],
                "tune_rgb_baseline": True,
                "cache_detections": True,
                "aux_fp_budget_source": "selected_rgb",
                "mean_selection_deltas": {"precision": 0.01, "recall": 0.03, "fp": -4.0, "det_count": -4.0, "small_recall": 0.0},
                "mean_test_deltas": {"precision": 0.011, "recall": 0.045, "fp": -7.2, "det_count": -7.1, "small_recall": 0.0},
                "rows": [
                    {
                        "seed": 101,
                        "selection_status": "pass",
                        "test_status": "pass" if passed else "mixed",
                        "selected_epoch": 3,
                        "selected_confidence": 0.5,
                        "selected_max_detections": 30,
                        "selected_rgb_confidence": 0.28,
                        "selected_rgb_max_detections": 30,
                        "test_deltas": {"precision": 0.01, "recall": 0.04, "fp": -7.0, "det_count": -7.0, "small_recall": 0.0},
                    }
                ],
            }
        )
        + "\n"
    )
    return path


def _write_dense_input_ablation_gate(path: Path, *, passed: bool) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    checks = [
        {
            "id": "zero_aux_reduces_dense_dnn_performance",
            "required": True,
            "status": "pass" if passed else "fail",
            "description": "Zeroing Aux channels should degrade held-out dense-detector performance.",
            "criteria": [
                {"metric": "recall_drop", "value": 0.50 if passed else 0.01, "threshold": 0.05, "pass": passed},
                {"metric": "precision_drop", "value": 0.04 if passed else -0.01, "threshold": 0.0, "pass": passed},
            ],
        },
        {
            "id": "zero_rgb_is_not_sufficient_for_selected_detector",
            "required": True,
            "status": "pass",
            "description": "Zeroing RGB should not outperform the full RGB+Aux input.",
            "criteria": [{"metric": "recall_drop", "value": 0.40, "threshold": 0.05, "pass": True}],
        },
    ]
    payload = {
        "status": "pass" if passed else "fail",
        "claim_status": "aux_input_used_by_dense_dnn" if passed else "aux_input_dependence_not_supported",
        "seed_count": 3,
        "test_sample_count": 125,
        "modes": ["none", "zero_aux", "shuffle_aux", "zero_rgb"],
        "mean_by_mode": {
            "none": {"precision": 0.05, "recall": 0.60, "small_recall": 0.02, "fp": 18.0, "det_count": 19.0},
            "zero_aux": {"precision": 0.01, "recall": 0.10, "small_recall": 0.00, "fp": 26.0, "det_count": 26.5},
            "shuffle_aux": {"precision": 0.04, "recall": 0.55, "small_recall": 0.02, "fp": 18.5, "det_count": 19.0},
            "zero_rgb": {"precision": 0.02, "recall": 0.20, "small_recall": 0.00, "fp": 24.0, "det_count": 24.5},
        },
        "deltas_vs_none": {
            "zero_aux": {"precision": -0.04, "recall": -0.50, "small_recall": -0.02, "fp": 8.0, "det_count": 7.5},
            "shuffle_aux": {"precision": -0.01, "recall": -0.05, "small_recall": 0.0, "fp": 0.5, "det_count": 0.0},
            "zero_rgb": {"precision": -0.03, "recall": -0.40, "small_recall": -0.02, "fp": 6.0, "det_count": 5.5},
        },
        "checks": checks,
        "interpretation": "unit dense input ablation",
        "claim_boundary": "unit compact dense-DNN boundary",
    }
    (path / "dense_input_ablation_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _dnn_sweep_row(
    *,
    confidence: float,
    metric_pass: bool,
    aux: tuple[float, float, float, float],
    rgb: tuple[float, float, float, float],
    failed: tuple[str, ...],
) -> dict:
    return {
        "confidence": confidence,
        "rgb_aux": {
            "sample_count": 32,
            "channel_mode": "rgb_aux",
            "precision@0.50_mean": aux[0],
            "recall@0.50_mean": aux[1],
            "small_recall@0.50_mean": aux[2],
            "fp@0.50_mean": aux[3],
        },
        "rgb_only": {
            "sample_count": 32,
            "channel_mode": "rgb_only",
            "precision@0.50_mean": rgb[0],
            "recall@0.50_mean": rgb[1],
            "small_recall@0.50_mean": rgb[2],
            "fp@0.50_mean": rgb[3],
        },
        "deltas": {
            "precision@0.50_mean": aux[0] - rgb[0],
            "recall@0.50_mean": aux[1] - rgb[1],
            "small_recall@0.50_mean": aux[2] - rgb[2],
            "fp@0.50_mean": aux[3] - rgb[3],
        },
        "criteria": [],
        "pass": False,
        "metric_pass": metric_pass,
        "failed_criteria": ("sample_count", *failed) if failed else ("sample_count",),
        "failed_metric_criteria": list(failed),
    }


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


def _write_object_boundary_edge(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "pass": True,
        "claim_status": "object_boundary_edge_diagnostic",
        "sample_count": 4,
        "box_count": 6,
        "included_labels": ["car", "pedestrian"],
        "checks": [
            {"id": "finite_object_boundary_outputs", "status": "pass"},
            {"id": "object_box_boundaries_present", "status": "pass"},
            {"id": "object_boundary_metrics_bounded", "status": "pass"},
            {"id": "camerae2e_cfa_pattern_preserved", "status": "pass"},
        ],
        "aggregate": {
            "human_rgb_edge_boundary_f1_mean": 0.030,
            "perception_rgb_edge_boundary_f1_mean": 0.031,
            "aux_edge_strength_boundary_f1_mean": 0.032,
            "aux_edge_confidence_boundary_f1_mean": 0.035,
            "aux_confidence_minus_human_boundary_f1_mean": 0.005,
            "aux_confidence_minus_human_boundary_f1_win_rate": 0.67,
        },
        "label_breakdown": [],
        "area_breakdown": [],
        "interpretation": "unit object-boundary edge proxy",
        "claim_boundary": "unit box-boundary proxy; not segmentation-contour or detector-performance evidence",
    }
    (path / "object_boundary_edge_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_object_boundary_detection_bridge(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "pass": True,
        "claim_status": "object_boundary_detection_bridge_diagnostic",
        "sample_count": 4,
        "object_count": 6,
        "baseline_input": "human_rgb",
        "target_input": "perception_score_aux",
        "checks": [
            {"id": "object_boundary_rows_matched_to_comparison_samples", "status": "pass"},
            {"id": "baseline_and_target_inputs_present", "status": "pass"},
            {"id": "object_boundary_bridge_features_finite", "status": "pass"},
            {"id": "target_detection_correlation_computable", "status": "pass"},
            {"id": "baseline_detection_correlation_computable", "status": "pass"},
        ],
        "aggregate": {
            "baseline_recall_proxy": 0.48,
            "target_recall_proxy": 0.50,
            "target_minus_baseline_recall_proxy": 0.02,
            "target_only_detected_count": 2,
            "baseline_only_detected_count": 1,
        },
        "group_breakdown": [],
        "label_breakdown": [],
        "correlations": {"rows": []},
        "interpretation": "unit object-boundary detection bridge",
        "claim_boundary": "unit TP/miss bridge only",
    }
    (path / "object_boundary_detection_bridge_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_detector_box_support_audit(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "pass": True,
        "claim_status": "detector_box_support_diagnostic",
        "sample_count": 8,
        "target_input": "perception_score_aux",
        "transition_baseline_input": "perception_fusion_rgb_aux",
        "checks": [
            {"id": "audit_inputs_have_detector_rows", "status": "pass"},
            {"id": "target_fp_tp_rows_available", "status": "pass"},
            {"id": "target_score_separates_fp_from_tp", "status": "pass"},
            {"id": "target_aux_edge_global_result_recorded", "status": "pass"},
            {"id": "transition_bridge_available", "status": "pass"},
            {"id": "transition_removes_more_fp_than_tp", "status": "pass"},
            {"id": "removed_fp_has_lower_aux_edge_support", "status": "pass"},
            {"id": "removed_fp_has_lower_source_scene_edge_support", "status": "pass"},
        ],
        "target_summary": {
            "input_name": "perception_score_aux",
            "detection_count": 20,
            "tp_count": 12,
            "fp_count": 8,
            "precision_proxy": 0.60,
            "correlations": {
                "rows": [
                    {"feature": "score", "delta_fp_minus_tp": -0.40, "auc_low_feature_predicts_fp": 0.85, "lower_feature_predicts_fp": True},
                    {"feature": "edge_support", "delta_fp_minus_tp": 0.02, "auc_low_feature_predicts_fp": 0.44, "lower_feature_predicts_fp": False},
                ]
            },
        },
        "input_summaries": [],
        "transition_bridge": {
            "compared_sample_count": 8,
            "baseline_detection_count": 22,
            "target_detection_count": 20,
            "removed_fp_count": 4,
            "removed_tp_count": 0,
            "fp_delta_count": -4,
            "tp_delta_count": 0,
            "proposal_correlation": {
                "rows": [
                    {
                        "comparison": "removed_fp_vs_kept_tp",
                        "feature": "edge_support",
                        "delta": -0.05,
                        "auc_low_feature_predicts_positive": 0.62,
                        "lower_feature_predicts_positive": True,
                    },
                    {
                        "comparison": "removed_fp_vs_kept_tp",
                        "feature": "scene_edge_support",
                        "delta": -0.03,
                        "auc_low_feature_predicts_positive": 0.66,
                        "lower_feature_predicts_positive": True,
                    },
                ]
            },
        },
        "interpretation": "unit detector-box support audit",
        "claim_boundary": "unit support metadata boundary",
    }
    (path / "detector_box_support_audit_summary.json").write_text(json.dumps(payload) + "\n")
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


def _write_adverse_native_slice(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    conditions = ["nominal", "night", "fog", "glare", "low_mtf", "hdr"]
    payload = {
        "status": "pass",
        "claim_status": "adverse_fp_reducer_supported",
        "run_count": 6,
        "expected_run_count": 6,
        "count": 32,
        "conditions": conditions,
        "cfa_pattern": "GRBG",
        "psf_sigma": 0.0,
        "use_camerae2e": True,
        "runs": [
            {
                "run_id": f"adverse-{condition}",
                "raw_condition_summary": {
                    "true_sensor_cfa_mosaic_count": 32,
                    "pattern_remapped_count": 0,
                },
            }
            for condition in conditions
        ],
        "checks": [
            {"id": "condition_grid_complete", "status": "pass", "evidence": "runs=6 expected=6"},
            {"id": "native_camerae2e_raw_used", "status": "pass", "evidence": "true_native=192 remapped=0"},
            {"id": "adverse_fp_reduction_observed", "status": "pass", "evidence": "fp_wins=5/5"},
        ],
        "aggregate": {
            "sample_count": 192,
            "adverse_condition_count": 5,
            "adverse_fp_win_count": 5,
            "adverse_recall_preserved_count": 4,
            "adverse_joint_fp_recall_win_count": 4,
            "mean_adverse_delta_precision@0.50": 0.0477,
            "mean_adverse_delta_recall@0.50": -0.003,
            "mean_adverse_delta_small_recall@0.50": -0.0016,
            "mean_adverse_delta_fp@0.50": -0.35,
            "primary_rows": [
                {
                    "condition": "night",
                    "run_id": "adverse-night",
                    "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
                    "delta_precision@0.50": 0.053,
                    "delta_recall@0.50": 0.0,
                    "delta_small_recall@0.50": 0.0,
                    "delta_fp@0.50": -0.469,
                }
            ],
        },
    }
    (path / "adverse_native_slice_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_adverse_task_slice(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "claim_status": "adverse_task_gate_partially_supported",
        "profile": "fp_reducer",
        "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
        "baseline_input": "human_rgb",
        "condition_count": 6,
        "expected_condition_count": 6,
        "cfa_pattern": "GRBG",
        "psf_sigma": 0.0,
        "aggregate": {
            "adverse_condition_count": 5,
            "adverse_passed_condition_count": 4,
            "adverse_failed_condition_count": 1,
            "failed_group_count": 3,
            "skipped_group_count": 6,
        },
        "checks": [
            {"id": "condition_reports_available", "status": "pass", "evidence": "conditions=6 expected=6"},
            {"id": "task_groups_evaluated", "status": "pass", "evidence": "evaluated_groups=30"},
            {"id": "adverse_task_gate_majority", "status": "pass", "evidence": "adverse_passed=4/5"},
        ],
        "group_summary": [
            {
                "group": "vru",
                "evaluated_condition_count": 6,
                "pass_condition_count": 6,
                "fail_condition_count": 0,
                "skipped_condition_count": 0,
                "gt_count_total": 320,
                "mean_delta_precision@0.50": 0.045,
                "mean_delta_recall@0.50": 0.0,
                "mean_delta_fp@0.50_per_sample": -0.0625,
                "worst_delta_recall@0.50": 0.0,
                "worst_delta_fp@0.50_per_sample": -0.03125,
            }
        ],
        "conditions": [
            {
                "condition": "nominal",
                "run_id": "condition-nominal",
                "report": "001_condition-nominal/comparison_summary.json",
                "sample_count": 32,
                "verdict": "task_gate_fail",
                "pass": False,
                "evaluated_group_count": 5,
                "failed_group_count": 1,
                "skipped_group_count": 1,
                "failed_groups": ["small_all"],
                "skipped_groups": ["traffic_light"],
            },
            {
                "condition": "night",
                "run_id": "condition-night",
                "report": "002_condition-night/comparison_summary.json",
                "sample_count": 32,
                "verdict": "task_gate_pass",
                "pass": True,
                "evaluated_group_count": 5,
                "failed_group_count": 0,
                "skipped_group_count": 1,
                "failed_groups": [],
                "skipped_groups": ["traffic_light"],
            },
            {
                "condition": "hdr",
                "run_id": "condition-hdr",
                "report": "006_condition-hdr/comparison_summary.json",
                "sample_count": 32,
                "verdict": "task_gate_fail",
                "pass": False,
                "evaluated_group_count": 5,
                "failed_group_count": 2,
                "skipped_group_count": 1,
                "failed_groups": ["vehicle", "small_all"],
                "skipped_groups": ["traffic_light"],
            },
        ],
        "interpretation": "unit adverse task slice",
        "claim_boundary": "unit simulated task boundary",
    }
    (path / "adverse_task_slice_summary.json").write_text(json.dumps(payload) + "\n")
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


def _write_cfa_lenspsf_aux_ablation(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "claim_status": "aux_recall_fp_tradeoff",
        "condition_count": 2,
        "expected_condition_count": 2,
        "aggregate": {
            "sample_count": 32,
            "aux_recall_win_count": 2,
            "aux_fp_win_count": 0,
            "mean_aux_minus_no_aux_recall@0.50": 0.003,
            "mean_aux_minus_no_aux_fp@0.50": 0.05,
        },
        "checks": [
            {"id": "matched_conditions_available", "status": "pass", "evidence": "matched=2 expected=2"},
            {"id": "aux_recall_tradeoff_measured", "status": "pass", "evidence": "aux_recall_wins=2/2"},
            {"id": "aux_fp_incremental_gain_majority", "status": "warning", "evidence": "aux_fp_wins=0/2"},
        ],
    }
    (path / "cfa_lenspsf_aux_ablation_summary.json").write_text(json.dumps(payload) + "\n")
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
