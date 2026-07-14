from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.object_boundary_edge_suite import (
    build_object_boundary_edge_suite,
    main as object_boundary_main,
    write_object_boundary_edge_suite,
)
from perception_isp.evaluation.synthetic_eval import make_synthetic_evaluation_samples


class ObjectBoundaryEdgeSuiteTest(unittest.TestCase):
    def test_build_object_boundary_edge_suite_computes_box_boundary_metrics(self) -> None:
        samples = make_synthetic_evaluation_samples(count=2, width=96, height=64, cfa_pattern="RGGB")

        summary = build_object_boundary_edge_suite(samples, include_labels=("car", "person"))

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["claim_status"], "object_boundary_edge_diagnostic")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["box_count"], 4)
        aggregate = summary["aggregate"]
        self.assertIn("human_rgb_edge_boundary_f1_mean", aggregate)
        self.assertIn("perception_rgb_minus_human_boundary_f1_mean", aggregate)
        self.assertIn("aux_strength_minus_human_boundary_f1_win_rate", aggregate)
        self.assertTrue(0.0 <= aggregate["human_rgb_edge_boundary_f1_mean"] <= 1.0)
        self.assertTrue(0.0 <= aggregate["aux_edge_strength_boundary_f1_mean"] <= 1.0)
        self.assertEqual({row["id"]: row["status"] for row in summary["checks"]}["object_box_boundaries_present"], "pass")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = make_synthetic_evaluation_samples(count=1, width=80, height=48, cfa_pattern="RGGB")
            summary = build_object_boundary_edge_suite(samples)
            html_path = write_object_boundary_edge_suite(summary, root / "report")
            persisted = json.loads((html_path.parent / "object_boundary_edge_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertIn("Object Boundary Edge Evidence", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = object_boundary_main(
                    [
                        "--source",
                        "camerae2e-synthetic",
                        "--no-camerae2e",
                        "--count",
                        "1",
                        "--width",
                        "80",
                        "--height",
                        "48",
                        "--cfa",
                        "RGGB",
                        "--output-dir",
                        str(root / "cli_report"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["claim_status"], "object_boundary_edge_diagnostic")


if __name__ == "__main__":
    unittest.main()
