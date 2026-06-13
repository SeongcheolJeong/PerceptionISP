from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perception_isp import dense_input_ablation_gate as gate
from perception_isp.dense_input_ablation_gate import (
    build_dense_input_ablation_gate,
    main as dense_input_ablation_main,
    parse_modes,
    write_dense_input_ablation_gate,
)


class DenseInputAblationGateTest(unittest.TestCase):
    def test_parse_modes_normalizes_and_deduplicates(self) -> None:
        self.assertEqual(parse_modes("none,zero-aux,none,shuffle_aux"), ("none", "zero_aux", "shuffle_aux"))

    def test_build_gate_detects_aux_input_dependence(self) -> None:
        with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=_fake_eval):
            summary = build_dense_input_ablation_gate(
                gate_summary=_gate_summary(),
                modes=("none", "zero_aux", "shuffle_aux", "zero_rgb"),
                include_labels=("car", "person"),
                device="cpu",
            )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["claim_status"], "aux_input_used_by_dense_dnn")
        self.assertEqual(summary["test_sample_count"], 4)
        self.assertEqual(summary["seed_count"], 2)
        self.assertAlmostEqual(summary["mean_by_mode"]["none"]["recall"], 0.70)
        self.assertAlmostEqual(summary["mean_by_mode"]["zero_aux"]["recall"], 0.10)
        self.assertAlmostEqual(summary["deltas_vs_none"]["zero_aux"]["recall"], -0.60)
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["zero_aux_reduces_dense_dnn_performance"]["status"], "pass")
        self.assertTrue(checks["zero_aux_reduces_dense_dnn_performance"]["required"])
        self.assertEqual(checks["shuffle_aux_spatial_sensitivity_diagnostic"]["status"], "pass")
        self.assertFalse(checks["shuffle_aux_spatial_sensitivity_diagnostic"]["required"])

    def test_zero_aux_precision_gain_can_still_pass_when_recall_drops(self) -> None:
        def fake_eval(**kwargs):
            mode = str(kwargs["input_ablation"])
            values = {
                "none": (0.40, 0.60, 18.0, 19.0),
                "zero_aux": (0.45, 0.25, 8.0, 8.4),
                "zero_rgb": (0.10, 0.15, 20.0, 20.5),
            }[mode]
            precision, recall, fp, det_count = values
            return {
                "sample_count": 4,
                "aggregate": {
                    "precision@0.50_mean": precision,
                    "recall@0.50_mean": recall,
                    "fp@0.50_mean": fp,
                    "det_count_mean": det_count,
                    "small_recall@0.50_mean": 0.0,
                },
            }

        with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=fake_eval):
            summary = build_dense_input_ablation_gate(
                gate_summary=_gate_summary(),
                modes=("none", "zero_aux", "zero_rgb"),
            )

        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(checks["zero_aux_reduces_dense_dnn_performance"]["status"], "pass")
        precision_criterion = checks["zero_aux_reduces_dense_dnn_performance"]["criteria"][1]
        self.assertFalse(precision_criterion["required"])
        self.assertLess(precision_criterion["value"], 0.0)

    def test_write_and_cli_emit_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_dir = root / "gate"
            gate_dir.mkdir()
            (gate_dir / "summary.json").write_text(json.dumps(_gate_summary()) + "\n")

            with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=_fake_eval):
                summary = build_dense_input_ablation_gate(gate_summary=gate_dir)
            html_path = write_dense_input_ablation_gate(summary, root / "report")
            self.assertTrue(html_path.exists())
            self.assertTrue((html_path.parent / "dense_input_ablation_summary.json").exists())
            self.assertIn("PerceptionISP Dense Input Ablation Gate", html_path.read_text())

            stdout = io.StringIO()
            with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=_fake_eval):
                with contextlib.redirect_stdout(stdout):
                    rc = dense_input_ablation_main(
                        [
                            "--gate-summary",
                            str(gate_dir),
                            "--include-labels",
                            "car,person",
                            "--device",
                            "cpu",
                            "--output-dir",
                            str(root / "cli"),
                        ]
                    )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["failed_required_checks"], [])
            self.assertTrue((root / "cli" / "dense_input_ablation_summary.json").exists())


def _gate_summary() -> dict:
    return {
        "manifest": "manifest.jsonl",
        "test_indices": [0, 1, 2, 3],
        "rows": [
            {
                "seed": 101,
                "aux_checkpoint": "aux_101.pt",
                "selected_confidence": 0.2,
                "selected_nms_iou": None,
                "selected_max_detections": 30,
            },
            {
                "seed": 202,
                "aux_checkpoint": "aux_202.pt",
                "selected_confidence": 0.24,
                "selected_nms_iou": 0.5,
                "selected_max_detections": 20,
            },
        ],
    }


def _fake_eval(**kwargs) -> dict:
    mode = str(kwargs["input_ablation"])
    values = {
        "none": (0.50, 0.70, 10.0, 11.0),
        "zero_aux": (0.05, 0.10, 18.0, 18.2),
        "shuffle_aux": (0.48, 0.62, 10.5, 11.2),
        "zero_rgb": (0.10, 0.20, 17.0, 17.4),
    }[mode]
    precision, recall, fp, det_count = values
    return {
        "sample_count": 4,
        "aggregate": {
            "precision@0.50_mean": precision,
            "recall@0.50_mean": recall,
            "fp@0.50_mean": fp,
            "det_count_mean": det_count,
            "small_recall@0.50_mean": 0.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
