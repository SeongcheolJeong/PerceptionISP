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


def _report(*, human: dict, target: dict) -> dict:
    return {
        "sample_count": 10,
        "aggregate": {
            "human_rgb": dict(human, sample_count=10),
            "perception_calibrated_fusion_rgb_aux": dict(target, sample_count=10),
        },
    }


if __name__ == "__main__":
    unittest.main()
