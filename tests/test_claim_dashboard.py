from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.claim_dashboard import build_claim_dashboard, main as dashboard_main, write_claim_dashboard


class ClaimDashboardTest(unittest.TestCase):
    def test_dnn_sweep_evidence_handles_missing_best_recall_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = root / "sweep"
            sweep.mkdir()
            (sweep / "index.html").write_text("<html></html>")
            row = _dnn_sweep_row(
                confidence=0.98,
                metric_pass=False,
                aux=(0.06, 0.05, 0.04, 4.8),
                rgb=(0.03, 0.01, 0.01, 3.1),
                failed=("absolute_recall", "fp_vs_rgb_only"),
            )
            (sweep / "rgb_aux_dnn_sweep_summary.json").write_text(
                json.dumps(
                    {
                        "status": "fail",
                        "pass": False,
                        "metric_pass": False,
                        "claim_status": "rgb_aux_dnn_sweep_no_claim_operating_point",
                        "profile": "claim_quality",
                        "row_count": 1,
                        "rows": [row],
                        "best_passing_row": None,
                        "best_metric_row": None,
                        "best_recall_positive_delta_row": None,
                        "lowest_fp_positive_recall_delta_row": row,
                        "interpretation": "unit no operating point",
                        "claim_boundary": "unit boundary",
                    }
                )
                + "\n"
            )

            dashboard = build_claim_dashboard(claim_gate_specs=[], rgb_aux_dnn_sweep=sweep)

            evidence = next(
                item
                for item in dashboard["evidence_map"]["current_evidence"]
                if item["area"] == "RGB+Aux DNN operating-point sweep"
            )
            self.assertIn("bestRecall none", evidence["evidence"])
            self.assertIn("lowestFPPositive conf=0.9800", evidence["evidence"])

    def test_dashboard_separates_supported_and_blocked_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = _write_claim_gate(root / "broad", profile="broad_superiority", passed=False)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            training = _write_training_rollup(root / "training")
            rgb_aux_dnn_gate = _write_rgb_aux_dnn_gate(root / "rgb_aux_dnn_gate", passed=False)
            rgb_aux_dnn_sweep = _write_rgb_aux_dnn_sweep(root / "rgb_aux_dnn_sweep", passed=False)
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
            adverse_native = _write_adverse_native_slice(root / "adverse_native")
            adverse_task = _write_adverse_task_slice(root / "adverse_task")
            cfa_lenspsf_detector = _write_cfa_lenspsf_detector_sweep(root / "cfa_lenspsf_detector")
            cfa_lenspsf_proposal = _write_cfa_lenspsf_proposal_audit(root / "cfa_lenspsf_proposal")
            cfa_lenspsf_native = _write_cfa_lenspsf_native_audit(root / "cfa_lenspsf_native")
            cfa_lenspsf_casebook = _write_cfa_lenspsf_casebook(root / "cfa_lenspsf_casebook")
            cfa_lenspsf_aux_ablation = _write_cfa_lenspsf_aux_ablation(root / "cfa_lenspsf_aux_ablation")
            casebook = _write_casebook(root / "casebook")
            comparison = _write_comparison_rollup(root / "rollup")

            dashboard = build_claim_dashboard(
                claim_gate_specs=[f"Human superiority={broad}", f"FP reducer={fp}"],
                training_rollup=training,
                rgb_aux_dnn_gate=rgb_aux_dnn_gate,
                rgb_aux_dnn_sweep=rgb_aux_dnn_sweep,
                task_metrics=task_metrics,
                protocol_coverage=protocol,
                mechanism_validation=mechanism,
                cfa_stress_sweep=cfa_stress,
                edge_confidence_suite=edge_confidence,
                edge_fidelity_suite=edge_fidelity,
                scene_edge_confidence=[scene_edge, scene_edge_sweep],
                scene_information_stress=scene_information,
                aux_contribution_audit=aux_contribution,
                adverse_native_slice=adverse_native,
                adverse_task_slice=adverse_task,
                cfa_lenspsf_detector_sweep=cfa_lenspsf_detector,
                cfa_lenspsf_proposal_audit=cfa_lenspsf_proposal,
                cfa_lenspsf_native_audit=cfa_lenspsf_native,
                cfa_lenspsf_casebook=cfa_lenspsf_casebook,
                cfa_lenspsf_aux_ablation=cfa_lenspsf_aux_ablation,
                casebook=casebook,
                comparison_rollup_specs=[f"Calibration={comparison}"],
            )

            self.assertEqual(len(dashboard["claims"]), 2)
            statuses = [item["status"] for item in dashboard["decisions"]]
            self.assertIn("supported", statuses)
            self.assertIn("not_supported", statuses)
            self.assertEqual(dashboard["training"]["status"], "diagnostic_only")
            self.assertFalse(dashboard["rgb_aux_dnn_gate"]["pass"])
            self.assertEqual(dashboard["rgb_aux_dnn_gate"]["claim_status"], "rgb_aux_dnn_not_claim_ready")
            self.assertFalse(dashboard["rgb_aux_dnn_sweep"]["pass"])
            self.assertEqual(dashboard["rgb_aux_dnn_sweep"]["claim_status"], "rgb_aux_dnn_sweep_no_claim_operating_point")
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
            self.assertTrue(dashboard["adverse_native_slice"]["pass"])
            self.assertEqual(dashboard["adverse_native_slice"]["claim_status"], "adverse_fp_reducer_supported")
            self.assertEqual(dashboard["adverse_native_slice"]["adverse_fp_win_count"], 5)
            self.assertEqual(dashboard["adverse_native_slice"]["native_count"], 192)
            self.assertEqual(dashboard["adverse_native_slice"]["remapped_count"], 0)
            self.assertTrue(dashboard["adverse_task_slice"]["pass"])
            self.assertEqual(dashboard["adverse_task_slice"]["claim_status"], "adverse_task_gate_partially_supported")
            self.assertEqual(dashboard["adverse_task_slice"]["adverse_passed_condition_count"], 4)
            self.assertTrue(dashboard["cfa_lenspsf_detector_sweep"]["pass"])
            detector_run = dashboard["cfa_lenspsf_detector_sweep"]["runs"][0]
            self.assertEqual(detector_run["true_sensor_cfa_mosaic_fraction"], 1.0)
            self.assertEqual(detector_run["camerae2e_camera_types"], ["bayer-grbg"])
            self.assertEqual(detector_run["camerae2e_native_cfa_bridge_versions"], ["native_bayer_v1"])
            detector_evidence = next(row for row in dashboard["evidence_map"]["current_evidence"] if row["area"] == "CFA/LensPSF detector condition sweep")
            self.assertIn("min_true_cfa=1.0000", detector_evidence["evidence"])
            self.assertIn("native_bayer_v1", detector_evidence["evidence"])
            self.assertTrue(dashboard["cfa_lenspsf_proposal_audit"]["pass"])
            self.assertTrue(dashboard["cfa_lenspsf_native_audit"]["pass"])
            self.assertTrue(dashboard["cfa_lenspsf_casebook"]["pass"])
            self.assertEqual(dashboard["cfa_lenspsf_casebook"]["selected_counterexample_count"], 2)
            self.assertTrue(dashboard["cfa_lenspsf_aux_ablation"]["pass"])
            self.assertEqual(dashboard["cfa_lenspsf_aux_ablation"]["claim_status"], "aux_recall_fp_tradeoff")
            self.assertEqual(dashboard["cfa_lenspsf_aux_ablation"]["aggregate"]["aux_recall_win_count"], 2)
            self.assertEqual(dashboard["cfa_lenspsf_aux_ablation"]["aggregate"]["aux_fp_win_count"], 0)
            self.assertTrue(dashboard["casebook"]["pass"])
            self.assertEqual(
                dashboard["evidence_map"]["claim_posture"]["recommended_claim"],
                "Use a narrow recall-budgeted FP-reduction claim, with front-end/aux evidence as feasibility support.",
            )
            self.assertEqual(dashboard["evidence_map"]["claim_posture"]["blocked_claim"], "Do not claim broad HumanISP superiority.")
            evidence_areas = [row["area"] for row in dashboard["evidence_map"]["current_evidence"]]
            self.assertIn("Recall-budgeted FP reduction", evidence_areas)
            self.assertIn("High-information scene edge similarity", evidence_areas)
            self.assertIn("Aux evidence used downstream", evidence_areas)
            self.assertIn("Adverse native RAW slice", evidence_areas)
            self.assertIn("Adverse task-specific slice", evidence_areas)
            self.assertIn("CFA/LensPSF detector condition sweep", evidence_areas)
            self.assertIn("CFA/LensPSF proposal-edge bridge", evidence_areas)
            self.assertIn("CFA/LensPSF native-CFA separation", evidence_areas)
            self.assertIn("CFA/LensPSF visual casebook", evidence_areas)
            self.assertIn("CFA/LensPSF score-label aux ablation", evidence_areas)
            self.assertIn("RGB+Aux DNN fine-tune gate", evidence_areas)
            self.assertIn("RGB+Aux DNN operating-point sweep", evidence_areas)
            self.assertIn("Visual success/failure casebook", evidence_areas)
            self.assertTrue(
                any(
                    row["area"] == "Broad HumanISP superiority" and row["status"] == "not_supported"
                    for row in dashboard["evidence_map"]["current_evidence"]
                )
            )
            future_evidence = [row["evidence"] for row in dashboard["evidence_map"]["future_evidence"]]
            self.assertIn("Adverse-condition/native RAW slice", future_evidence)
            self.assertIn("Task-specific adverse gate", future_evidence)
            self.assertIn("Scene-edge proposal correlation across CFA/LensPSF", future_evidence)
            self.assertIn("CFA/LensPSF detector sweep", future_evidence)
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
            self.assertIn(
                "Adverse native RAW slice supports simulated-condition FP reduction: adverse FP wins 5/5, recall preserved 4/5, mean dR -0.0030, mean dFP -0.3500. Treat this as simulated native RAW evidence, not proof on real adverse datasets.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Adverse task-specific slice supports simulated task FP-reducer behavior in 4/5 adverse conditions under profile fp_reducer. Failed conditions: nominal, hdr. Use as simulated task-slice evidence, not real adverse task proof.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "CFA/LensPSF detector sweep is available as condition-level detector evidence; use it for sensitivity analysis, not broad superiority.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "CFA/LensPSF proposal-edge audit passed as condition-level bridge evidence: removed FP 7, removed TP 1, source scene-edge positive conditions 2, aux-edge positive conditions 1, mean scene-edge AUC 0.6200, mean aux-edge AUC 0.5300.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "CFA/LensPSF native-CFA audit passed: native rows 1, remapped rows 1. Use remapped rows only as bridge sensitivity evidence.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "CFA/LensPSF visual casebook is available for condition review: conditions 2/2, selected cases 5, selected FP-reduction successes 3, selected counterexamples 2.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "CFA/LensPSF aux ablation shows a recall/FP tradeoff rather than incremental aux FP superiority: aux recall wins 2/2, aux FP wins 0/2, mean dR +0.0030, mean dFP +0.0500. Do not claim aux improves FP beyond score/label calibration from this sweep.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "Visual success/failure casebook is available for qualitative review: selected FP-reduction successes 2, selected counterexamples 3.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "RGB+Aux DNN gate failed for sample_count, absolute_recall, absolute_fp_per_sample, small_recall_vs_rgb_only, fp_vs_rgb_only; do not claim the aux tensor improves a learned detector yet.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertIn(
                "RGB+Aux DNN confidence sweep found no claim-ready operating point; threshold tuning alone does not support learned RGB+Aux detector improvement.",
                [item["claim"] for item in dashboard["decisions"]],
            )
            self.assertTrue(
                any(
                    item["claim"].startswith("Front-end/downstream bridge is directionally positive")
                    and item["status"] == "diagnostic"
                    and "same-sample causal correlation" in item["claim"]
                    for item in dashboard["decisions"]
                )
            )
            self.assertTrue(
                any(
                    item["claim"].startswith("Same-sample aux bridge passed")
                    and item["status"] == "diagnostic"
                    and "edge support delta" in item["claim"]
                    and "Low-edge AUC" in item["claim"]
                    and "Source scene-edge AUC" in item["claim"]
                    for item in dashboard["decisions"]
                )
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
            self.assertIn("Performance Evidence Map", html)
            self.assertIn("What More Evidence To Build", html)
            self.assertIn("Do not claim broad HumanISP superiority", html)
            self.assertIn("Scene-edge proposal correlation across CFA/LensPSF", html)
            self.assertIn("Task Metrics", html)
            self.assertIn("RGB+Aux DNN Gate", html)
            self.assertIn("rgb_aux_dnn_not_claim_ready", html)
            self.assertIn("RGB+Aux DNN Confidence Sweep", html)
            self.assertIn("rgb_aux_dnn_sweep_no_claim_operating_point", html)
            self.assertIn("Aux Contribution Audit", html)
            self.assertIn("Same-Sample Aux Bridge", html)
            self.assertIn("Adverse Native RAW Slice", html)
            self.assertIn("Adverse Condition Rows", html)
            self.assertIn("adverse_fp_reducer_supported", html)
            self.assertIn("Adverse Task Slice", html)
            self.assertIn("adverse_task_gate_partially_supported", html)
            self.assertIn("Task Groups", html)
            self.assertIn("Success/Failure Casebook", html)
            self.assertIn("fp_reduction_success", html)
            self.assertIn("Removed FP Edge Delta vs Kept TP", html)
            self.assertIn("Low-Edge AUC", html)
            self.assertIn("Source Scene Edge AUC", html)
            self.assertIn("Mechanism Validation", html)
            self.assertIn("CFA Stress Sweep", html)
            self.assertIn("Edge Confidence Suite", html)
            self.assertIn("Object Edge Fidelity", html)
            self.assertIn("Scene Edge Confidence", html)
            self.assertIn("CFA/LensPSF Proposal Edge Bridge", html)
            self.assertIn("CFA/LensPSF Native-CFA Separation", html)
            self.assertIn("CFA/LensPSF Visual Casebook", html)
            self.assertIn("CFA/LensPSF Score-Label Aux Ablation", html)
            self.assertIn("Aux Ablation By CFA", html)
            self.assertIn("Proposal Edge Bridge By Condition", html)
            self.assertIn("CFA/LensPSF Casebook Category Totals", html)
            self.assertIn("Evidence Report", html)
            self.assertIn("RGB Delta", html)
            self.assertIn("Front-end/downstream bridge is directionally positive", html)
            self.assertIn("Scene Information Stress", html)
            self.assertIn("Benchmark Protocol Coverage", html)
            self.assertIn("recall_tradeoff", html)
            self.assertTrue((html_path.parent / "claim_dashboard_summary.json").exists())

    def test_large_native_cfa_lenspsf_evidence_changes_next_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector = _write_cfa_lenspsf_detector_sweep(root / "cfa_lenspsf_detector")
            proposal = _write_cfa_lenspsf_proposal_audit(root / "cfa_lenspsf_proposal")
            native = _write_cfa_lenspsf_native_audit(root / "cfa_lenspsf_native")
            _scale_cfa_lenspsf_detector(detector, count=128, run_count=12)
            _scale_cfa_lenspsf_proposal(proposal, sample_count=1536, condition_count=12)
            _scale_cfa_lenspsf_native(native, sample_count=1536, run_count=12)

            dashboard = build_claim_dashboard(
                claim_gate_specs=[],
                cfa_lenspsf_detector_sweep=detector,
                cfa_lenspsf_proposal_audit=proposal,
                cfa_lenspsf_native_audit=native,
            )

            future = {row["evidence"]: row["current_gap"] for row in dashboard["evidence_map"]["future_evidence"]}
            self.assertIn(
                "Large native CFA/LensPSF proposal bridge exists",
                future["Scene-edge proposal correlation across CFA/LensPSF"],
            )
            self.assertIn(
                "Detector sweep exists at larger native scale",
                future["CFA/LensPSF detector sweep"],
            )
            current = {row["area"]: row["next_evidence"] for row in dashboard["evidence_map"]["current_evidence"]}
            self.assertIn("adverse-condition RAW/native scene slices", current["CFA/LensPSF detector condition sweep"])
            self.assertIn("incremental aux ablation", current["CFA/LensPSF proposal-edge bridge"])
            self.assertIn("Keep the native/remap guardrail", current["CFA/LensPSF native-CFA separation"])

    def test_dashboard_cli_outputs_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = _write_claim_gate(root / "fp", profile="fp_reducer", passed=True)
            rgb_aux_dnn_gate = _write_rgb_aux_dnn_gate(root / "rgb_aux_dnn_gate", passed=False)
            rgb_aux_dnn_sweep = _write_rgb_aux_dnn_sweep(root / "rgb_aux_dnn_sweep", passed=False)
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
            adverse_native = _write_adverse_native_slice(root / "adverse_native")
            adverse_task = _write_adverse_task_slice(root / "adverse_task")
            cfa_lenspsf_detector = _write_cfa_lenspsf_detector_sweep(root / "cfa_lenspsf_detector")
            cfa_lenspsf_proposal = _write_cfa_lenspsf_proposal_audit(root / "cfa_lenspsf_proposal")
            cfa_lenspsf_native = _write_cfa_lenspsf_native_audit(root / "cfa_lenspsf_native")
            cfa_lenspsf_casebook = _write_cfa_lenspsf_casebook(root / "cfa_lenspsf_casebook")
            cfa_lenspsf_aux_ablation = _write_cfa_lenspsf_aux_ablation(root / "cfa_lenspsf_aux_ablation")
            casebook = _write_casebook(root / "casebook")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = dashboard_main(
                    [
                        "--claim-gate",
                        str(fp),
                        "--rgb-aux-dnn-gate",
                        str(rgb_aux_dnn_gate),
                        "--rgb-aux-dnn-sweep",
                        str(rgb_aux_dnn_sweep),
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
                        "--adverse-native-slice",
                        str(adverse_native),
                        "--adverse-task-slice",
                        str(adverse_task),
                        "--cfa-lenspsf-detector-sweep",
                        str(cfa_lenspsf_detector),
                        "--cfa-lenspsf-proposal-audit",
                        str(cfa_lenspsf_proposal),
                        "--cfa-lenspsf-native-audit",
                        str(cfa_lenspsf_native),
                        "--cfa-lenspsf-casebook",
                        str(cfa_lenspsf_casebook),
                        "--cfa-lenspsf-aux-ablation",
                        str(cfa_lenspsf_aux_ablation),
                        "--casebook",
                        str(casebook),
                        "--output-dir",
                        str(root / "dashboard"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["claim_count"], 1)
            summary = json.loads((root / "dashboard" / "claim_dashboard_summary.json").read_text())
            self.assertFalse(summary["rgb_aux_dnn_gate"]["pass"])
            self.assertEqual(summary["rgb_aux_dnn_gate"]["claim_status"], "rgb_aux_dnn_not_claim_ready")
            self.assertFalse(summary["rgb_aux_dnn_sweep"]["pass"])
            self.assertEqual(summary["rgb_aux_dnn_sweep"]["claim_status"], "rgb_aux_dnn_sweep_no_claim_operating_point")
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
            self.assertTrue(summary["adverse_native_slice"]["pass"])
            self.assertEqual(summary["adverse_native_slice"]["claim_status"], "adverse_fp_reducer_supported")
            self.assertTrue(summary["adverse_task_slice"]["pass"])
            self.assertEqual(summary["adverse_task_slice"]["claim_status"], "adverse_task_gate_partially_supported")
            self.assertTrue(summary["cfa_lenspsf_detector_sweep"]["pass"])
            self.assertTrue(summary["cfa_lenspsf_proposal_audit"]["pass"])
            self.assertTrue(summary["cfa_lenspsf_native_audit"]["pass"])
            self.assertTrue(summary["cfa_lenspsf_casebook"]["pass"])
            self.assertTrue(summary["cfa_lenspsf_aux_ablation"]["pass"])
            self.assertEqual(summary["cfa_lenspsf_aux_ablation"]["claim_status"], "aux_recall_fp_tradeoff")
            self.assertTrue(summary["casebook"]["pass"])
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
        {
            "id": "eval_classes_present",
            "status": "pass",
            "pass": True,
            "target": 0,
            "threshold": 0,
            "direction": "empty",
            "evidence": "none",
        },
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


def _dnn_sweep_row(
    *,
    confidence: float,
    metric_pass: bool,
    aux: tuple[float, float, float, float],
    rgb: tuple[float, float, float, float],
    failed: tuple[str, ...],
) -> dict:
    deltas = {
        "precision@0.50_mean": aux[0] - rgb[0],
        "recall@0.50_mean": aux[1] - rgb[1],
        "small_recall@0.50_mean": aux[2] - rgb[2],
        "fp@0.50_mean": aux[3] - rgb[3],
    }
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
        "deltas": deltas,
        "criteria": [],
        "pass": False,
        "metric_pass": metric_pass,
        "failed_criteria": ("sample_count", *failed) if failed else ("sample_count",),
        "failed_metric_criteria": list(failed),
    }


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
                "sample_bridge": {
                    "status": "pass",
                    "baseline_input": "perception_calibrated_score_label_fusion_rgb_aux",
                    "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux",
                    "compared_sample_count": 10,
                    "baseline_detection_count": 30,
                    "target_detection_count": 28,
                    "baseline_only_detection_count": 2,
                    "target_only_detection_count": 0,
                    "removed_fp_count": 2,
                    "removed_tp_count": 0,
                    "added_fp_count": 0,
                    "added_tp_count": 0,
                    "fp_delta_count": -2,
                    "tp_delta_count": 0,
                    "removed_fp_fraction": 1.0,
                    "removed_fp_to_tp_ratio": 2.0,
                    "support_means": {},
                    "support_deltas": {"removed_fp_minus_kept_tp_edge_support_mean": -0.2},
                    "proposal_correlation": {
                        "status": "pass",
                        "baseline_proposal_count": 10,
                        "rows": [
                            {
                                "comparison": "removed_fp_vs_kept_tp",
                                "feature": "edge_support",
                                "positive_status": "removed_fp",
                                "negative_status": "kept_tp",
                                "positive_count": 2,
                                "negative_count": 6,
                                "positive_mean": 0.1,
                                "negative_mean": 0.3,
                                "delta": -0.2,
                                "point_biserial": -0.4,
                                "auc_low_feature_predicts_positive": 0.8,
                                "lower_feature_predicts_positive": True,
                            },
                            {
                                "comparison": "removed_fp_vs_kept_tp",
                                "feature": "scene_edge_support",
                                "positive_status": "removed_fp",
                                "negative_status": "kept_tp",
                                "positive_count": 2,
                                "negative_count": 6,
                                "positive_mean": 0.12,
                                "negative_mean": 0.28,
                                "delta": -0.16,
                                "point_biserial": -0.35,
                                "auc_low_feature_predicts_positive": 0.75,
                                "lower_feature_predicts_positive": True,
                            }
                        ],
                    },
                    "interpretation": "unit same-sample bridge",
                },
                "feature_audit": {"aux_feature_count": 3, "aux_features": ["aux_support", "edge_support", "reliability_support"]},
                "interpretation": "unit aux contribution audit",
            }
        )
        + "\n"
    )
    return path


def _write_adverse_native_slice(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    conditions = ["nominal", "night", "fog", "glare", "low_mtf", "hdr"]
    runs = [
        {
            "run_id": f"adverse-{condition}",
            "condition": condition,
            "raw_condition_summary": {
                "true_sensor_cfa_mosaic_count": 32,
                "pattern_remapped_count": 0,
            },
        }
        for condition in conditions
    ]
    primary_rows = [
        {
            "condition": "nominal",
            "run_id": "adverse-nominal",
            "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
            "delta_precision@0.50": 0.035,
            "delta_recall@0.50": -0.016,
            "delta_small_recall@0.50": -0.031,
            "delta_fp@0.50": -0.250,
        },
        {
            "condition": "night",
            "run_id": "adverse-night",
            "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
            "delta_precision@0.50": 0.053,
            "delta_recall@0.50": 0.0,
            "delta_small_recall@0.50": 0.0,
            "delta_fp@0.50": -0.469,
        },
        {
            "condition": "fog",
            "run_id": "adverse-fog",
            "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
            "delta_precision@0.50": 0.082,
            "delta_recall@0.50": 0.005,
            "delta_small_recall@0.50": 0.005,
            "delta_fp@0.50": -0.438,
        },
        {
            "condition": "glare",
            "run_id": "adverse-glare",
            "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
            "delta_precision@0.50": 0.076,
            "delta_recall@0.50": -0.003,
            "delta_small_recall@0.50": 0.005,
            "delta_fp@0.50": -0.406,
        },
        {
            "condition": "low_mtf",
            "run_id": "adverse-low_mtf",
            "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
            "delta_precision@0.50": 0.039,
            "delta_recall@0.50": 0.0,
            "delta_small_recall@0.50": 0.0,
            "delta_fp@0.50": -0.250,
        },
        {
            "condition": "hdr",
            "run_id": "adverse-hdr",
            "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
            "delta_precision@0.50": -0.011,
            "delta_recall@0.50": -0.018,
            "delta_small_recall@0.50": -0.018,
            "delta_fp@0.50": -0.188,
        },
    ]
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
        "runs": runs,
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
            "primary_rows": primary_rows,
        },
        "interpretation": "unit adverse native RAW slice",
        "claim_boundary": "unit simulated adverse boundary",
    }
    (path / "adverse_native_slice_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_adverse_task_slice(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    condition_dir = path / "006_condition-hdr"
    condition_dir.mkdir()
    (condition_dir / "index.html").write_text("<html></html>")
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
            },
            {
                "group": "person",
                "evaluated_condition_count": 6,
                "pass_condition_count": 6,
                "fail_condition_count": 0,
                "skipped_condition_count": 0,
                "gt_count_total": 220,
                "mean_delta_precision@0.50": 0.010,
                "mean_delta_recall@0.50": 0.0,
                "mean_delta_fp@0.50_per_sample": -0.0052,
                "worst_delta_recall@0.50": 0.0,
                "worst_delta_fp@0.50_per_sample": 0.0,
            },
            {
                "group": "small_all",
                "evaluated_condition_count": 6,
                "pass_condition_count": 4,
                "fail_condition_count": 2,
                "skipped_condition_count": 0,
                "gt_count_total": 180,
                "mean_delta_precision@0.50": 0.015,
                "mean_delta_recall@0.50": -0.0021,
                "mean_delta_fp@0.50_per_sample": -0.0260,
                "worst_delta_recall@0.50": -0.0250,
                "worst_delta_fp@0.50_per_sample": 0.0625,
            },
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
                "report": str(condition_dir / "comparison_summary.json"),
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


def _write_comparison_rollup(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "rollup_summary.json").write_text(json.dumps({"run_count": 1, "baseline_input": "human_rgb", "runs": []}) + "\n")
    return path


def _write_cfa_lenspsf_detector_sweep(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "run_count": 2,
        "expected_run_count": 2,
        "count": 16,
        "cfa_patterns": ["GRBG"],
        "psf_sigmas": [0.0, 1.2],
        "use_camerae2e": True,
        "checks": [
            {"id": "condition_grid_complete", "status": "pass", "evidence": "runs=2 expected=2"},
            {"id": "psf_sigma_recorded_in_raw_provenance", "status": "pass", "evidence": "recorded=32 samples=32"},
        ],
        "rankings": {
            "calibrated_or_fusion_by_delta_fp@0.50": [
                {
                    "run_id": "cfa-grbg_psf-1p20",
                    "report": "002_cfa-grbg_psf-1p20/index.html",
                    "input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
                    "cfa_pattern": "GRBG",
                    "psf_sigma": 1.2,
                    "delta": -0.4,
                }
            ]
        },
        "runs": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "report": "001_cfa-grbg_psf-0p00/index.html",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "sample_count": 16,
                "raw_condition_summary": {
                    "pattern_remapped_fraction": 0.0,
                    "true_sensor_cfa_mosaic_fraction": 1.0,
                    "psf_recorded_fraction": 1.0,
                    "camerae2e_camera_types": {"bayer-grbg": 16},
                    "camerae2e_native_cfa_bridge_versions": {"native_bayer_v1": 16},
                },
                "metrics": {
                    "perception_calibrated_score_label_aux_fusion_rgb_aux_t001": {
                        "precision@0.50_mean": 0.65,
                        "recall@0.50_mean": 0.45,
                        "fp@0.50_mean": 1.0,
                    }
                },
                "delta_vs_human": {
                    "perception_calibrated_score_label_aux_fusion_rgb_aux_t001": {
                        "precision@0.50_mean": 0.03,
                        "recall@0.50_mean": -0.004,
                        "fp@0.50_mean": -0.3,
                    }
                },
            }
        ],
        "interpretation": "unit CFA/LensPSF detector sweep",
        "claim_boundary": "unit boundary",
    }
    (path / "cfa_lenspsf_detector_sweep_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_lenspsf_proposal_audit(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    condition_dir = path / "001_cfa-grbg_psf-0p00"
    condition_dir.mkdir()
    (condition_dir / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "condition_count": 2,
        "expected_condition_count": 2,
        "cfa_patterns": ["GRBG"],
        "psf_sigmas": [0.0, 1.2],
        "checks": [
            {"id": "condition_bridges_available", "status": "pass", "evidence": "bridges=2 expected=2"},
            {"id": "removed_fp_observed_across_conditions", "status": "pass", "evidence": "removed_fp=7 removed_tp=1"},
            {"id": "source_scene_edge_predicts_removed_fp_in_some_conditions", "status": "pass", "evidence": "positive_conditions=2/2"},
            {"id": "aux_edge_predicts_removed_fp_in_some_conditions", "status": "pass", "evidence": "positive_conditions=1/2"},
        ],
        "aggregate": {
            "condition_count": 2,
            "removed_fp_count": 7,
            "removed_tp_count": 1,
            "fp_delta_count": -7,
            "tp_delta_count": -1,
            "scene_edge_positive_condition_count": 2,
            "edge_positive_condition_count": 1,
            "scene_edge_delta_negative_condition_count": 2,
            "edge_delta_negative_condition_count": 1,
            "scene_edge_auc_condition_mean": 0.62,
            "edge_auc_condition_mean": 0.53,
            "scene_edge_support_delta_condition_mean": -0.02,
            "edge_support_delta_condition_mean": -0.05,
            "scene_edge_auc_removed_fp_weighted_mean": 0.62,
            "edge_auc_removed_fp_weighted_mean": 0.53,
            "scene_edge_support_delta_removed_fp_weighted_mean": -0.02,
            "edge_support_delta_removed_fp_weighted_mean": -0.05,
            "best_scene_edge_auc_condition": {
                "run_id": "cfa-grbg_psf-0p00",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "scene_edge_auc_low_predicts_removed_fp": 0.66,
            },
            "best_edge_auc_condition": {
                "run_id": "cfa-grbg_psf-1p20",
                "cfa_pattern": "GRBG",
                "psf_sigma": 1.2,
                "edge_auc_low_predicts_removed_fp": 0.53,
            },
        },
        "conditions": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "report": str(condition_dir / "comparison_summary.json"),
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "sample_count": 16,
                "removed_fp_count": 4,
                "removed_tp_count": 0,
                "fp_delta_count": -4,
                "edge_support_delta_removed_fp_minus_kept_tp": -0.04,
                "edge_auc_low_predicts_removed_fp": 0.52,
                "scene_edge_support_delta_removed_fp_minus_kept_tp": -0.03,
                "scene_edge_auc_low_predicts_removed_fp": 0.66,
            }
        ],
        "interpretation": "unit CFA/LensPSF proposal-edge audit",
        "claim_boundary": "unit proposal boundary",
    }
    (path / "cfa_lenspsf_proposal_audit_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_cfa_lenspsf_native_audit(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload = {
        "status": "pass",
        "run_count": 2,
        "expected_run_count": 2,
        "cfa_patterns": ["GRBG", "RGGB"],
        "psf_sigmas": [0.0],
        "checks": [
            {"id": "sweep_rows_available", "status": "pass", "evidence": "runs=2"},
            {"id": "native_rows_identified", "status": "pass", "evidence": "native_runs=1"},
            {"id": "remapped_rows_separated", "status": "pass", "evidence": "remapped_runs=1 partial_runs=0"},
        ],
        "groups": {
            "native": {
                "run_count": 1,
                "sample_count": 16,
                "cfa_patterns": ["GRBG"],
                "psf_sigmas": [0.0],
                "mean_delta_precision@0.50": 0.05,
                "mean_delta_recall@0.50": 0.0,
                "mean_delta_small_recall@0.50": 0.0,
                "mean_delta_fp@0.50": -0.4,
                "best_delta_fp@0.50": {"run_id": "cfa-grbg_psf-0p00", "delta": -0.4},
                "best_delta_recall@0.50": {"run_id": "cfa-grbg_psf-0p00", "delta": 0.0},
            },
            "partial_remap": {"run_count": 0, "sample_count": 0, "cfa_patterns": [], "psf_sigmas": []},
            "remapped": {
                "run_count": 1,
                "sample_count": 16,
                "cfa_patterns": ["RGGB"],
                "psf_sigmas": [0.0],
                "mean_delta_precision@0.50": 0.04,
                "mean_delta_recall@0.50": 0.0,
                "mean_delta_small_recall@0.50": 0.0,
                "mean_delta_fp@0.50": -0.2,
                "best_delta_fp@0.50": {"run_id": "cfa-rggb_psf-0p00", "delta": -0.2},
                "best_delta_recall@0.50": {"run_id": "cfa-rggb_psf-0p00", "delta": 0.0},
            },
        },
        "runs": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "native_status": "native",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "sample_count": 16,
                "pattern_remapped_count": 0,
                "pattern_remapped_fraction": 0.0,
                "delta_precision@0.50_mean": 0.05,
                "delta_recall@0.50_mean": 0.0,
                "delta_fp@0.50_mean": -0.4,
            },
            {
                "run_id": "cfa-rggb_psf-0p00",
                "native_status": "remapped",
                "cfa_pattern": "RGGB",
                "psf_sigma": 0.0,
                "sample_count": 16,
                "pattern_remapped_count": 16,
                "pattern_remapped_fraction": 1.0,
                "delta_precision@0.50_mean": 0.04,
                "delta_recall@0.50_mean": 0.0,
                "delta_fp@0.50_mean": -0.2,
            },
        ],
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
        "baseline_input": "perception_fusion_rgb_aux",
        "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
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
                "status": "warning",
                "sample_count": 16,
                "selected_case_count": 3,
                "tp_delta_count": 0,
                "fp_delta_count": -3,
                "pattern_remapped_fraction": 0.0,
                "true_sensor_cfa_mosaic_fraction": 1.0,
            },
            {
                "run_id": "cfa-rggb_psf-0p00",
                "cfa_pattern": "RGGB",
                "psf_sigma": 0.0,
                "status": "pass",
                "sample_count": 16,
                "selected_case_count": 2,
                "tp_delta_count": -1,
                "fp_delta_count": -2,
                "pattern_remapped_fraction": 0.0,
                "true_sensor_cfa_mosaic_fraction": 1.0,
            },
        ],
        "showcase_cases": [
            {
                "run_id": "cfa-grbg_psf-0p00",
                "cfa_pattern": "GRBG",
                "psf_sigma": 0.0,
                "category": "fp_reduction_success",
                "sample_id": "success",
                "fp_delta@0.50": -1,
                "tp_delta@0.50": 0,
            },
            {
                "run_id": "cfa-rggb_psf-0p00",
                "cfa_pattern": "RGGB",
                "psf_sigma": 0.0,
                "category": "recall_loss_failure",
                "sample_id": "counter",
                "fp_delta@0.50": 0,
                "tp_delta@0.50": -1,
            },
        ],
        "interpretation": "unit cfa lenspsf casebook",
        "claim_boundary": "unit cfa lenspsf casebook boundary",
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
        "no_aux_input": "perception_calibrated_score_label_fusion_rgb_aux",
        "aux_input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
        "aggregate": {
            "condition_count": 2,
            "sample_count": 32,
            "aux_precision_win_count": 1,
            "aux_recall_win_count": 2,
            "aux_recall_loss_count": 0,
            "aux_small_recall_win_count": 0,
            "aux_fp_win_count": 0,
            "mean_aux_minus_no_aux_precision@0.50": -0.002,
            "mean_aux_minus_no_aux_recall@0.50": 0.003,
            "mean_aux_minus_no_aux_small_recall@0.50": 0.0,
            "mean_aux_minus_no_aux_fp@0.50": 0.05,
        },
        "cfa_groups": [
            {
                "group": "GRBG",
                "condition_count": 1,
                "sample_count": 16,
                "aux_precision_win_count": 0,
                "aux_recall_win_count": 1,
                "aux_fp_win_count": 0,
                "mean_aux_minus_no_aux_precision@0.50": -0.004,
                "mean_aux_minus_no_aux_recall@0.50": 0.002,
                "mean_aux_minus_no_aux_small_recall@0.50": 0.0,
                "mean_aux_minus_no_aux_fp@0.50": 0.04,
            },
            {
                "group": "RGGB",
                "condition_count": 1,
                "sample_count": 16,
                "aux_precision_win_count": 1,
                "aux_recall_win_count": 1,
                "aux_fp_win_count": 0,
                "mean_aux_minus_no_aux_precision@0.50": 0.002,
                "mean_aux_minus_no_aux_recall@0.50": 0.004,
                "mean_aux_minus_no_aux_small_recall@0.50": 0.0,
                "mean_aux_minus_no_aux_fp@0.50": 0.06,
            },
        ],
        "psf_groups": [
            {
                "group": 0.0,
                "condition_count": 1,
                "sample_count": 16,
                "aux_precision_win_count": 0,
                "aux_recall_win_count": 1,
                "aux_fp_win_count": 0,
                "mean_aux_minus_no_aux_precision@0.50": -0.004,
                "mean_aux_minus_no_aux_recall@0.50": 0.002,
                "mean_aux_minus_no_aux_small_recall@0.50": 0.0,
                "mean_aux_minus_no_aux_fp@0.50": 0.04,
            },
            {
                "group": 1.6,
                "condition_count": 1,
                "sample_count": 16,
                "aux_precision_win_count": 1,
                "aux_recall_win_count": 1,
                "aux_fp_win_count": 0,
                "mean_aux_minus_no_aux_precision@0.50": 0.002,
                "mean_aux_minus_no_aux_recall@0.50": 0.004,
                "mean_aux_minus_no_aux_small_recall@0.50": 0.0,
                "mean_aux_minus_no_aux_fp@0.50": 0.06,
            },
        ],
        "checks": [
            {"id": "matched_conditions_available", "status": "pass", "evidence": "matched=2 expected=2"},
            {"id": "aux_recall_tradeoff_measured", "status": "pass", "evidence": "aux_recall_wins=2/2"},
            {"id": "aux_fp_incremental_gain_majority", "status": "warning", "evidence": "aux_fp_wins=0/2"},
        ],
        "interpretation": "unit aux ablation",
        "claim_boundary": "unit aux ablation boundary",
    }
    (path / "cfa_lenspsf_aux_ablation_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_casebook(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "visual.png").write_text("placeholder")
    payload = {
        "status": "pass",
        "sample_count": 16,
        "selected_case_count": 5,
        "baseline_input": "human_rgb",
        "target_input": "perception_calibrated_score_label_aux_fusion_rgb_aux_t001",
        "aggregate": {"tp_delta_count": -1, "fp_delta_count": -6},
        "checks": [
            {"id": "casebook_has_selected_cases", "status": "pass", "evidence": "selected_cases=5"},
            {"id": "casebook_includes_fp_reduction_successes", "status": "pass", "evidence": "selected_successes=2"},
            {"id": "casebook_includes_counterexamples", "status": "pass", "evidence": "selected_counterexamples=3"},
        ],
        "categories": {
            "fp_reduction_success": {
                "case_count": 9,
                "selected_case_count": 2,
                "cases": [{"sample_id": "success", "visual_path": str(path / "visual.png"), "category": "fp_reduction_success"}],
            },
            "recall_tradeoff": {"case_count": 3, "selected_case_count": 1, "cases": [{"sample_id": "tradeoff"}]},
            "recall_loss_failure": {"case_count": 2, "selected_case_count": 1, "cases": [{"sample_id": "recall_loss"}]},
            "fp_regression_failure": {"case_count": 1, "selected_case_count": 1, "cases": [{"sample_id": "fp_regression"}]},
        },
        "interpretation": "unit casebook",
        "claim_boundary": "unit casebook boundary",
    }
    (path / "casebook_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _scale_cfa_lenspsf_detector(path: Path, *, count: int, run_count: int) -> None:
    summary = path / "cfa_lenspsf_detector_sweep_summary.json"
    payload = json.loads(summary.read_text())
    payload["count"] = count
    payload["run_count"] = run_count
    payload["expected_run_count"] = run_count
    payload["checks"][0]["evidence"] = f"runs={run_count} expected={run_count}"
    payload["checks"][1]["evidence"] = f"recorded={count * run_count} samples={count * run_count}"
    summary.write_text(json.dumps(payload) + "\n")


def _scale_cfa_lenspsf_proposal(path: Path, *, sample_count: int, condition_count: int) -> None:
    summary = path / "cfa_lenspsf_proposal_audit_summary.json"
    payload = json.loads(summary.read_text())
    payload["condition_count"] = condition_count
    payload["expected_condition_count"] = condition_count
    payload["aggregate"]["condition_count"] = condition_count
    payload["aggregate"]["sample_count"] = sample_count
    payload["aggregate"]["scene_edge_positive_condition_count"] = condition_count
    payload["aggregate"]["edge_positive_condition_count"] = condition_count
    payload["checks"][0]["evidence"] = f"bridges={condition_count} expected={condition_count}"
    payload["checks"][2]["evidence"] = f"positive_conditions={condition_count}/{condition_count}"
    payload["checks"][3]["evidence"] = f"positive_conditions={condition_count}/{condition_count}"
    summary.write_text(json.dumps(payload) + "\n")


def _scale_cfa_lenspsf_native(path: Path, *, sample_count: int, run_count: int) -> None:
    summary = path / "cfa_lenspsf_native_audit_summary.json"
    payload = json.loads(summary.read_text())
    payload["run_count"] = run_count
    payload["expected_run_count"] = run_count
    payload["groups"]["native"]["sample_count"] = sample_count
    payload["groups"]["native"]["run_count"] = run_count
    payload["checks"][0]["evidence"] = f"runs={run_count}"
    payload["checks"][1]["evidence"] = f"native_runs={run_count}"
    summary.write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    unittest.main()
