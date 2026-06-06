from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.claim_gate import build_claim_gate, main as claim_gate_main, write_claim_gate


class ClaimGateTest(unittest.TestCase):
    def test_claim_gate_fails_when_recall_drops_despite_precision_and_fp_gain(self) -> None:
        summary = build_claim_gate(
            _report(
                human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "recall@0.75_mean": 0.30, "small_recall@0.50_mean": 0.28, "fp@0.50_mean": 1.30},
                target={"precision@0.50_mean": 0.64, "recall@0.50_mean": 0.46, "recall@0.75_mean": 0.29, "small_recall@0.50_mean": 0.27, "fp@0.50_mean": 1.00},
            ),
            target_input="perception_calibrated_fusion_rgb_aux",
        )
        self.assertFalse(summary["pass"])
        failed = {item["metric"] for item in summary["criteria"] if not item["pass"]}
        self.assertIn("recall@0.50_mean", failed)
        self.assertIn("recall@0.75_mean", failed)
        self.assertIn("small_recall@0.50_mean", failed)
        self.assertNotIn("precision@0.50_mean", failed)
        self.assertNotIn("fp@0.50_mean", failed)

    def test_claim_gate_passes_conservative_metric_superiority(self) -> None:
        summary = build_claim_gate(
            _report(
                human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "recall@0.75_mean": 0.30, "small_recall@0.50_mean": 0.28, "fp@0.50_mean": 1.30},
                target={"precision@0.50_mean": 0.61, "recall@0.50_mean": 0.48, "recall@0.75_mean": 0.31, "small_recall@0.50_mean": 0.29, "fp@0.50_mean": 1.20},
            ),
            target_input="perception_calibrated_fusion_rgb_aux",
        )
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["verdict"], "metric_gate_pass")

    def test_claim_gate_fails_missing_metrics(self) -> None:
        report = _report(
            human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "fp@0.50_mean": 1.30},
            target={"precision@0.50_mean": 0.61, "recall@0.50_mean": 0.48, "fp@0.50_mean": 1.20},
        )
        summary = build_claim_gate(report, target_input="perception_calibrated_fusion_rgb_aux")
        self.assertFalse(summary["pass"])
        missing = [item for item in summary["criteria"] if item["metric"] == "recall@0.75_mean"][0]
        self.assertFalse(missing["available"])
        self.assertFalse(missing["pass"])

    def test_claim_gate_records_paired_bootstrap_intervals(self) -> None:
        report = _report_with_samples(
            human_rows=[
                {"precision@0.50": 0.5, "recall@0.50": 0.5, "recall@0.75": 0.4, "small_recall@0.50": 0.3, "fp@0.50": 2},
                {"precision@0.50": 0.6, "recall@0.50": 0.6, "recall@0.75": 0.5, "small_recall@0.50": 0.4, "fp@0.50": 1},
                {"precision@0.50": 0.7, "recall@0.50": 0.7, "recall@0.75": 0.6, "small_recall@0.50": 0.5, "fp@0.50": 1},
            ],
            target_rows=[
                {"precision@0.50": 0.6, "recall@0.50": 0.6, "recall@0.75": 0.5, "small_recall@0.50": 0.4, "fp@0.50": 1},
                {"precision@0.50": 0.7, "recall@0.50": 0.7, "recall@0.75": 0.6, "small_recall@0.50": 0.5, "fp@0.50": 0},
                {"precision@0.50": 0.8, "recall@0.50": 0.8, "recall@0.75": 0.7, "small_recall@0.50": 0.6, "fp@0.50": 0},
            ],
        )
        summary = build_claim_gate(
            report,
            target_input="perception_calibrated_fusion_rgb_aux",
            thresholds={"bootstrap_samples": 64, "bootstrap_seed": "unit", "require_ci": True},
        )
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["thresholds"]["bootstrap_seed"], "unit")
        precision = [item for item in summary["criteria"] if item["metric"] == "precision@0.50_mean"][0]
        self.assertEqual(precision["paired_delta"]["sample_count"], 3)
        self.assertGreaterEqual(precision["paired_delta"]["ci_low"], 0.0)
        fp = [item for item in summary["criteria"] if item["metric"] == "fp@0.50_mean"][0]
        self.assertLessEqual(fp["paired_delta"]["ci_high"], 0.0)

    def test_claim_gate_require_ci_fails_without_sample_metrics(self) -> None:
        summary = build_claim_gate(
            _report(
                human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "recall@0.75_mean": 0.30, "small_recall@0.50_mean": 0.28, "fp@0.50_mean": 1.30},
                target={"precision@0.50_mean": 0.61, "recall@0.50_mean": 0.48, "recall@0.75_mean": 0.31, "small_recall@0.50_mean": 0.29, "fp@0.50_mean": 1.20},
            ),
            target_input="perception_calibrated_fusion_rgb_aux",
            thresholds={"require_ci": True},
        )
        self.assertFalse(summary["pass"])
        precision = [item for item in summary["criteria"] if item["metric"] == "precision@0.50_mean"][0]
        self.assertIsNone(precision["paired_delta"])
        self.assertFalse(precision["ci_pass"])

    def test_claim_gate_writes_json_and_html(self) -> None:
        summary = build_claim_gate(
            _report(
                human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "recall@0.75_mean": 0.30, "small_recall@0.50_mean": 0.28, "fp@0.50_mean": 1.30},
                target={"precision@0.50_mean": 0.64, "recall@0.50_mean": 0.46, "recall@0.75_mean": 0.29, "small_recall@0.50_mean": 0.27, "fp@0.50_mean": 1.00},
            ),
            target_input="perception_calibrated_fusion_rgb_aux",
        )
        with tempfile.TemporaryDirectory() as tmp:
            html_path = write_claim_gate(summary, Path(tmp))
            self.assertTrue(html_path.exists())
            self.assertTrue((Path(tmp) / "claim_gate_summary.json").exists())
            self.assertIn("metric_gate_fail", html_path.read_text())
            self.assertIn("CI Low", html_path.read_text())

    def test_claim_gate_cli_outputs_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "comparison_summary.json"
            output_dir = Path(tmp) / "claim"
            report_path.write_text(
                json.dumps(
                    _report(
                        human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "recall@0.75_mean": 0.30, "small_recall@0.50_mean": 0.28, "fp@0.50_mean": 1.30},
                        target={"precision@0.50_mean": 0.64, "recall@0.50_mean": 0.46, "recall@0.75_mean": 0.29, "small_recall@0.50_mean": 0.27, "fp@0.50_mean": 1.00},
                    )
                )
                + "\n"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = claim_gate_main([str(report_path), "--output-dir", str(output_dir)])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(printed["pass"])
            self.assertIn("recall@0.50_mean", printed["failed"])
            self.assertTrue((output_dir / "claim_gate_summary.json").exists())

    def test_claim_gate_cli_can_return_nonzero_on_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "comparison_summary.json"
            output_dir = Path(tmp) / "claim"
            report_path.write_text(
                json.dumps(
                    _report(
                        human={"precision@0.50_mean": 0.60, "recall@0.50_mean": 0.47, "recall@0.75_mean": 0.30, "small_recall@0.50_mean": 0.28, "fp@0.50_mean": 1.30},
                        target={"precision@0.50_mean": 0.64, "recall@0.50_mean": 0.46, "recall@0.75_mean": 0.29, "small_recall@0.50_mean": 0.27, "fp@0.50_mean": 1.00},
                    )
                )
                + "\n"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = claim_gate_main([str(report_path), "--fail-on-fail", "--output-dir", str(output_dir)])
            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(stdout.getvalue())["pass"])
            self.assertTrue((output_dir / "claim_gate_summary.json").exists())


def _report(*, human: dict, target: dict) -> dict:
    return {
        "sample_count": 10,
        "aggregate": {
            "human_rgb": dict(human, sample_count=10),
            "perception_calibrated_fusion_rgb_aux": dict(target, sample_count=10),
        },
    }


def _report_with_samples(*, human_rows: list[dict], target_rows: list[dict]) -> dict:
    samples = []
    for index, (human, target) in enumerate(zip(human_rows, target_rows)):
        samples.append(
            {
                "sample_id": str(index),
                "metrics": {
                    "human_rgb": human,
                    "perception_calibrated_fusion_rgb_aux": target,
                },
            }
        )
    return {
        "sample_count": len(samples),
        "samples": samples,
        "aggregate": {
            "human_rgb": _aggregate(human_rows),
            "perception_calibrated_fusion_rgb_aux": _aggregate(target_rows),
        },
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = {
        "precision@0.50_mean": "precision@0.50",
        "recall@0.50_mean": "recall@0.50",
        "recall@0.75_mean": "recall@0.75",
        "small_recall@0.50_mean": "small_recall@0.50",
        "fp@0.50_mean": "fp@0.50",
    }
    return {out_key: sum(float(row[in_key]) for row in rows) / len(rows) for out_key, in_key in keys.items()}


if __name__ == "__main__":
    unittest.main()
