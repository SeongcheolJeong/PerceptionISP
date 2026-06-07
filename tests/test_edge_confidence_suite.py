from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.edge_confidence_suite import build_edge_confidence_suite, main as edge_suite_main, write_edge_confidence_suite


class EdgeConfidenceSuiteTest(unittest.TestCase):
    def test_build_edge_confidence_suite_passes_expected_confidence_checks(self) -> None:
        summary = build_edge_confidence_suite(width=96, height=56, cfa_pattern="RGGB")

        self.assertEqual(summary["status"], "pass")
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["low_light_edge_confidence_drop"]["status"], "pass")
        self.assertEqual(checks["glare_edge_confidence_drop"]["status"], "pass")
        self.assertEqual(checks["low_mtf_strong_edge_confidence_drop"]["status"], "pass")

        cases = {row["id"]: row for row in summary["cases"]}
        nominal = cases["nominal_sharp"]["metrics"]
        self.assertLess(cases["low_light"]["metrics"]["edge_confidence_mean"], nominal["edge_confidence_mean"])
        self.assertLess(cases["glare_saturated"]["metrics"]["demosaic_confidence_mean"], nominal["demosaic_confidence_mean"])
        self.assertLess(cases["low_mtf"]["metrics"]["strong_edge_confidence_mean"], nominal["strong_edge_confidence_mean"])

    def test_write_edge_confidence_suite_outputs_json_html_assets_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_edge_confidence_suite(width=80, height=48)
            html_path = write_edge_confidence_suite(summary, root / "report")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Edge Confidence Suite", html_path.read_text())
            persisted = json.loads((html_path.parent / "edge_confidence_suite_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertNotIn("_assets_source", persisted["cases"][0])
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = edge_suite_main(
                    [
                        "--width",
                        "80",
                        "--height",
                        "48",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["failed_checks"], [])
            self.assertTrue((root / "cli" / "edge_confidence_suite_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
