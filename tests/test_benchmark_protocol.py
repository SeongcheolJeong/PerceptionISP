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
                min_samples=3,
            )

            self.assertEqual(summary["status"], "claim_ready")
            self.assertEqual(summary["coverage_status"], "coverage_complete")
            self.assertEqual(summary["metric_claim_status"], "broad_superiority_not_supported")
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


if __name__ == "__main__":
    unittest.main()
