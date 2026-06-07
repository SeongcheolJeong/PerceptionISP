from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.rgb_aux_dnn_sweep import build_rgb_aux_dnn_sweep, main as sweep_main, write_rgb_aux_dnn_sweep


class RgbAuxDnnSweepTest(unittest.TestCase):
    def test_sweep_distinguishes_metric_pass_from_sample_scale(self) -> None:
        pairs = [
            _pair(0.50, aux=(32, 0.02, 0.16, 0.05, 40.0), rgb=(32, 0.01, 0.10, 0.02, 80.0)),
            _pair(0.90, aux=(32, 0.06, 0.11, 0.04, 4.0), rgb=(32, 0.02, 0.07, 0.02, 20.0)),
        ]

        summary = build_rgb_aux_dnn_sweep(pairs)

        self.assertFalse(summary["pass"])
        self.assertTrue(summary["metric_pass"])
        self.assertEqual(summary["claim_status"], "rgb_aux_dnn_sweep_needs_scale")
        self.assertEqual(summary["best_metric_row"]["confidence"], 0.90)
        self.assertIn("sample_count", summary["best_metric_row"]["failed_criteria"])

    def test_sweep_reports_no_claim_operating_point(self) -> None:
        pairs = [
            _pair(0.80, aux=(1200, 0.04, 0.12, 0.04, 20.0), rgb=(1200, 0.01, 0.06, 0.02, 50.0)),
            _pair(0.93, aux=(1200, 0.06, 0.04, 0.02, 4.0), rgb=(1200, 0.01, 0.02, 0.01, 20.0)),
        ]

        summary = build_rgb_aux_dnn_sweep(pairs)

        self.assertFalse(summary["pass"])
        self.assertFalse(summary["metric_pass"])
        self.assertEqual(summary["claim_status"], "rgb_aux_dnn_sweep_no_claim_operating_point")
        failed_by_conf = {row["confidence"]: row["failed_metric_criteria"] for row in summary["rows"]}
        self.assertIn("absolute_precision", failed_by_conf[0.80])
        self.assertIn("absolute_recall", failed_by_conf[0.93])

    def test_sweep_writes_report_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aux50 = _write_dense_eval(root / "aux50", _summary(sample_count=32, p=0.02, r=0.16, small=0.05, fp=40.0, mode="rgb_aux"))
            rgb50 = _write_dense_eval(root / "rgb50", _summary(sample_count=32, p=0.01, r=0.10, small=0.02, fp=80.0, mode="rgb_only"))
            aux90 = _write_dense_eval(root / "aux90", _summary(sample_count=32, p=0.06, r=0.11, small=0.04, fp=4.0, mode="rgb_aux"))
            rgb90 = _write_dense_eval(root / "rgb90", _summary(sample_count=32, p=0.02, r=0.07, small=0.02, fp=20.0, mode="rgb_only"))

            summary = build_rgb_aux_dnn_sweep(
                [
                    {"confidence": 0.50, "rgb_aux": json.loads((aux50 / "dense_eval_summary.json").read_text()), "rgb_only": json.loads((rgb50 / "dense_eval_summary.json").read_text())},
                    {"confidence": 0.90, "rgb_aux": json.loads((aux90 / "dense_eval_summary.json").read_text()), "rgb_only": json.loads((rgb90 / "dense_eval_summary.json").read_text())},
                ]
            )
            html_path = write_rgb_aux_dnn_sweep(summary, root / "sweep")
            self.assertTrue((html_path.parent / "rgb_aux_dnn_sweep_summary.json").exists())
            self.assertIn("RGB+Aux DNN Confidence Sweep", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = sweep_main(
                    [
                        "--confidence",
                        "0.50",
                        "--rgb-aux",
                        str(aux50),
                        "--rgb-only",
                        str(rgb50),
                        "--confidence",
                        "0.90",
                        "--rgb-aux",
                        str(aux90),
                        "--rgb-only",
                        str(rgb90),
                        "--output-dir",
                        str(root / "sweep_cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(printed["pass"])
            self.assertTrue(printed["metric_pass"])
            self.assertEqual(printed["best_metric_confidence"], 0.90)


def _pair(confidence: float, *, aux: tuple[int, float, float, float, float], rgb: tuple[int, float, float, float, float]) -> dict:
    return {
        "confidence": confidence,
        "rgb_aux": _summary(sample_count=aux[0], p=aux[1], r=aux[2], small=aux[3], fp=aux[4], mode="rgb_aux"),
        "rgb_only": _summary(sample_count=rgb[0], p=rgb[1], r=rgb[2], small=rgb[3], fp=rgb[4], mode="rgb_only"),
    }


def _summary(*, sample_count: int, p: float, r: float, small: float, fp: float, mode: str) -> dict:
    return {
        "split": "eval",
        "sample_count": sample_count,
        "aggregate": {
            "sample_count": sample_count,
            "precision@0.50_mean": p,
            "recall@0.50_mean": r,
            "small_recall@0.50_mean": small,
            "fp@0.50_mean": fp,
            "det_count_mean": fp + 1.0,
        },
        "checkpoint_summary": {
            "channel_mode": mode,
            "missing_eval_class_names": [],
        },
    }


def _write_dense_eval(path: Path, payload: dict) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    (path / "dense_eval_summary.json").write_text(json.dumps(payload) + "\n")
    return path
