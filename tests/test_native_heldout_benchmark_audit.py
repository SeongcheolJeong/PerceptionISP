from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.native_heldout_benchmark_audit import (
    build_native_heldout_benchmark_audit,
    main as native_heldout_main,
    write_native_heldout_benchmark_audit,
)


class NativeHeldoutBenchmarkAuditTest(unittest.TestCase):
    def test_build_audit_accepts_large_native_camerae2e_cfa_report(self) -> None:
        report = _comparison_report(sample_count=4, remapped=False)

        summary = build_native_heldout_benchmark_audit(
            report,
            baseline_input="human_rgb",
            target_input="perception_target",
            min_samples=4,
            source_report_path="/tmp/report/comparison_summary.json",
        )

        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["claim_status"], "large_native_fp_reducer_with_recall_tradeoff")
        self.assertEqual(summary["sample_count"], 4)
        provenance = summary["provenance"]
        self.assertEqual(provenance["camerae2e_used_fraction"], 1.0)
        self.assertEqual(provenance["native_raw_source_accepted_fraction"], 1.0)
        self.assertEqual(provenance["true_sensor_cfa_mosaic_fraction"], 1.0)
        self.assertEqual(provenance["pattern_remapped_count"], 0)
        self.assertEqual(provenance["source_cfa_patterns"], {"GRBG": 4})
        deltas = summary["metric_summary"]["deltas"]
        self.assertGreater(deltas["precision@0.50_mean"], 0.0)
        self.assertLess(deltas["recall@0.50_mean"], 0.0)
        self.assertLess(deltas["fp@0.50_mean"], 0.0)

    def test_build_audit_accepts_real_nef_native_raw_report(self) -> None:
        report = _comparison_report(sample_count=4, remapped=False, camerae2e_used=False, bridge="pascalraw_native_nef", raw_source_key="PASCALRAW/original/raw/*.nef")

        summary = build_native_heldout_benchmark_audit(
            report,
            baseline_input="human_rgb",
            target_input="perception_target",
            min_samples=4,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["pass"])
        provenance = summary["provenance"]
        self.assertEqual(provenance["camerae2e_used_fraction"], 0.0)
        self.assertEqual(provenance["native_raw_source_accepted_fraction"], 1.0)
        self.assertEqual(provenance["bridges"], {"pascalraw_native_nef": 4})
        checks = {row["id"]: row["status"] for row in summary["checks"]}
        self.assertEqual(checks["native_raw_source_for_all_samples"], "pass")

    def test_build_audit_rejects_remapped_or_too_small_reports(self) -> None:
        report = _comparison_report(sample_count=3, remapped=True)

        summary = build_native_heldout_benchmark_audit(
            report,
            baseline_input="human_rgb",
            target_input="perception_target",
            min_samples=4,
        )

        self.assertEqual(summary["status"], "warning")
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["claim_status"], "native_heldout_benchmark_not_supported")
        checks = {row["id"]: row["status"] for row in summary["checks"]}
        self.assertEqual(checks["large_heldout_sample_count"], "fail")
        self.assertEqual(checks["no_pattern_remap"], "fail")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "comparison"
            report_dir.mkdir()
            (report_dir / "comparison_summary.json").write_text(json.dumps(_comparison_report(sample_count=4, remapped=False)) + "\n")
            (report_dir / "index.html").write_text("<html></html>")

            summary = build_native_heldout_benchmark_audit(
                json.loads((report_dir / "comparison_summary.json").read_text()),
                baseline_input="human_rgb",
                target_input="perception_target",
                min_samples=4,
                source_report_path=report_dir / "comparison_summary.json",
            )
            html_path = write_native_heldout_benchmark_audit(summary, root / "audit")
            self.assertTrue((html_path.parent / "native_heldout_benchmark_audit_summary.json").exists())
            self.assertIn("PerceptionISP Native Held-Out Benchmark Audit", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = native_heldout_main(
                    [
                        str(report_dir),
                        "--baseline-input",
                        "human_rgb",
                        "--target-input",
                        "perception_target",
                        "--min-samples",
                        "4",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["claim_status"], "large_native_fp_reducer_with_recall_tradeoff")


def _comparison_report(
    *,
    sample_count: int,
    remapped: bool,
    camerae2e_used: bool = True,
    bridge: str = "native_bayer_v1",
    raw_source_key: str = "camerae2e_scene",
) -> dict:
    return {
        "aggregate": {
            "human_rgb": {
                "precision@0.50_mean": 0.50,
                "recall@0.50_mean": 0.80,
                "recall@0.75_mean": 0.60,
                "small_recall@0.50_mean": 0.20,
                "fp@0.50_mean": 2.0,
                "fp@0.75_mean": 2.4,
                "det_count_mean": 3.0,
                "tp@0.50_mean": 1.0,
                "fn@0.50_mean": 0.2,
            },
            "perception_target": {
                "precision@0.50_mean": 0.56,
                "recall@0.50_mean": 0.79,
                "recall@0.75_mean": 0.61,
                "small_recall@0.50_mean": 0.205,
                "fp@0.50_mean": 1.4,
                "fp@0.75_mean": 1.9,
                "det_count_mean": 2.4,
                "tp@0.50_mean": 0.98,
                "fn@0.50_mean": 0.22,
            },
        },
        "samples": [
            {
                "sample_id": f"sample_{index}",
                "metadata": {
                    "raw_provenance": {
                        "camerae2e_used": camerae2e_used,
                        "true_sensor_cfa_mosaic": True,
                        "pattern_remapped": remapped,
                        "source_cfa_pattern": "GRBG",
                        "target_cfa_pattern": "RGGB" if remapped else "GRBG",
                        "requested_cfa_pattern": "RGGB" if remapped else "GRBG",
                        "bridge": "remap_bridge" if remapped else bridge,
                        "raw_source_key": raw_source_key,
                        "source_shape": [375, 1242],
                        "target_shape": [375, 1242],
                        "native_resolution_matches_target": True,
                        "native_resolution_at_least_target": True,
                    }
                },
            }
            for index in range(sample_count)
        ],
    }


if __name__ == "__main__":
    unittest.main()
