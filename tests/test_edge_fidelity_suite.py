from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.edge_fidelity_suite import build_edge_fidelity_suite, main as edge_fidelity_main, write_edge_fidelity_suite


class EdgeFidelitySuiteTest(unittest.TestCase):
    def test_build_edge_fidelity_suite_passes_object_edge_checks(self) -> None:
        summary = build_edge_fidelity_suite(
            sensor_width=96,
            sensor_height=56,
            oversample=4,
            cfa_patterns=("RGGB", "GRBG", "RCCB"),
            psf_sigmas=(0.0, 1.2),
        )

        self.assertEqual(summary["status"], "pass")
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["finite_edge_fidelity_outputs"]["status"], "pass")
        self.assertEqual(checks["object_and_sensor_edge_oracles_present"]["status"], "pass")
        self.assertEqual(checks["edge_fidelity_metrics_bounded"]["status"], "pass")
        self.assertEqual(checks["lens_psf_reduces_sensor_edge_contrast"]["status"], "pass")
        self.assertGreaterEqual(len(summary["rankings"]), 2)
        visibility = summary["psf_visibility"]
        self.assertEqual([row["psf_sigma"] for row in visibility], [0.0, 1.2])
        self.assertEqual(visibility[0]["visibility_status"], "nominal")
        self.assertEqual(visibility[-1]["visibility_status"], "visible")
        self.assertLessEqual(visibility[-1]["sensor_edge_strength_p95_ratio_vs_nominal"], 0.85)
        self.assertLess(visibility[-1]["sensor_edge_strength_p95_delta_vs_previous"], 0.0)

        cases = summary["cases"]
        self.assertEqual(len(cases), 6)
        for case in cases:
            metrics = case["metrics"]
            self.assertGreater(metrics["object_edge_fraction"], 0.005)
            self.assertGreater(metrics["sensor_edge_fraction"], 0.005)
            self.assertGreaterEqual(metrics["aux_object_edge_f1"], 0.0)
            self.assertLessEqual(metrics["aux_object_edge_f1"], 1.0)

    def test_write_edge_fidelity_suite_outputs_json_html_assets_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_edge_fidelity_suite(
                sensor_width=80,
                sensor_height=48,
                oversample=3,
                cfa_patterns=("RGGB", "MONO"),
                psf_sigmas=(0.0, 1.0),
            )
            html_path = write_edge_fidelity_suite(summary, root / "report")
            self.assertTrue(html_path.exists())
            html = html_path.read_text()
            self.assertIn("PerceptionISP Object Edge Fidelity", html)
            self.assertIn("LensPSF Visibility", html)
            persisted = json.loads((html_path.parent / "edge_fidelity_suite_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertIn("psf_visibility", persisted)
            self.assertNotIn("_assets_source", persisted["cases"][0])
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = edge_fidelity_main(
                    [
                        "--sensor-width",
                        "80",
                        "--sensor-height",
                        "48",
                        "--oversample",
                        "3",
                        "--cfa",
                        "RGGB",
                        "--cfa",
                        "MONO",
                        "--psf-sigma",
                        "0.0",
                        "--psf-sigma",
                        "1.0",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["failed_checks"], [])
            self.assertTrue((root / "cli" / "edge_fidelity_suite_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
