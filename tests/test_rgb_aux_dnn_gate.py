from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.rgb_aux_dnn_gate import build_rgb_aux_dnn_gate, main as gate_main, write_rgb_aux_dnn_gate


class RgbAuxDnnGateTest(unittest.TestCase):
    def test_gate_fails_claim_quality_when_sample_scale_and_fp_are_weak(self) -> None:
        rgb_aux = _dense_summary(
            mode="rgb_aux",
            channels=6,
            sample_count=32,
            precision=0.012,
            recall=0.09,
            small_recall=0.02,
            fp=48.0,
        )
        rgb_only = _dense_summary(
            mode="rgb_only",
            channels=3,
            sample_count=32,
            precision=0.013,
            recall=0.08,
            small_recall=0.03,
            fp=40.0,
        )

        summary = build_rgb_aux_dnn_gate(rgb_aux, rgb_only)

        self.assertFalse(summary["pass"])
        self.assertEqual(summary["claim_status"], "rgb_aux_dnn_not_claim_ready")
        failed = {row["id"] for row in summary["criteria"] if row["status"] == "fail"}
        self.assertIn("sample_count", failed)
        self.assertIn("absolute_fp_per_sample", failed)
        self.assertIn("small_recall_vs_rgb_only", failed)

    def test_diagnostic_profile_pass_is_not_claim_ready(self) -> None:
        rgb_aux = _dense_summary(
            mode="rgb_aux",
            channels=16,
            sample_count=8,
            precision=0.0,
            recall=0.0,
            small_recall=0.0,
            fp=22.0,
        )
        rgb_only = _dense_summary(
            mode="rgb_only",
            channels=16,
            sample_count=8,
            precision=0.008,
            recall=0.18,
            small_recall=0.0,
            fp=30.0,
        )

        summary = build_rgb_aux_dnn_gate(rgb_aux, rgb_only, thresholds={"profile": "diagnostic", "min_samples": 8})

        self.assertTrue(summary["pass"])
        self.assertEqual(summary["claim_status"], "rgb_aux_dnn_diagnostic_pass")
        self.assertIn("diagnostic smoke gate", summary["interpretation"])

    def test_gate_writes_report_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_aux_dir = _write_dense_eval(
                root / "rgb_aux",
                _dense_summary(
                    mode="rgb_aux",
                    channels=6,
                    sample_count=1200,
                    precision=0.10,
                    recall=0.22,
                    small_recall=0.16,
                    fp=2.5,
                ),
            )
            rgb_only_dir = _write_dense_eval(
                root / "rgb_only",
                _dense_summary(
                    mode="rgb_only",
                    channels=3,
                    sample_count=1200,
                    precision=0.08,
                    recall=0.20,
                    small_recall=0.12,
                    fp=3.0,
                ),
            )

            summary = build_rgb_aux_dnn_gate(
                json.loads((rgb_aux_dir / "dense_eval_summary.json").read_text()),
                json.loads((rgb_only_dir / "dense_eval_summary.json").read_text()),
            )
            html_path = write_rgb_aux_dnn_gate(summary, root / "gate")
            self.assertTrue(summary["pass"])
            self.assertTrue((html_path.parent / "rgb_aux_dnn_gate_summary.json").exists())
            self.assertIn("RGB+Aux DNN Gate", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = gate_main(
                    [
                        "--rgb-aux",
                        str(rgb_aux_dir),
                        "--rgb-only",
                        str(rgb_only_dir),
                        "--output-dir",
                        str(root / "gate_cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(printed["pass"])
            self.assertEqual(printed["status"], "pass")


def _dense_summary(
    *,
    mode: str,
    channels: int,
    sample_count: int,
    precision: float,
    recall: float,
    small_recall: float,
    fp: float,
) -> dict:
    return {
        "split": "eval",
        "sample_count": sample_count,
        "aggregate": {
            "sample_count": sample_count,
            "precision@0.50_mean": precision,
            "recall@0.50_mean": recall,
            "recall@0.75_mean": recall * 0.5,
            "small_recall@0.50_mean": small_recall,
            "fp@0.50_mean": fp,
            "det_count_mean": fp + 1.0,
        },
        "checkpoint_summary": {
            "channel_mode": mode,
            "tensor_key": "rgb_aux_chw" if mode != "rgb_only" else "rgb_chw",
            "input_channels": channels,
            "missing_eval_class_names": [],
        },
    }


def _write_dense_eval(path: Path, payload: dict) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "dense_eval_summary.json").write_text(json.dumps(payload) + "\n")
    return path
