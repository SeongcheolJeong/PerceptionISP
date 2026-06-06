from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.mechanism_validation import build_mechanism_validation, main as mechanism_main, write_mechanism_validation


class MechanismValidationTest(unittest.TestCase):
    def test_build_mechanism_validation_passes_expected_synthetic_checks(self) -> None:
        summary = build_mechanism_validation(width=96, height=56, cfa_patterns=("RGGB", "GRBG", "RCCB", "RGBIR"))

        self.assertEqual(summary["status"], "pass")
        mechanisms = {row["id"]: row for row in summary["mechanisms"]}
        self.assertEqual(mechanisms["low_light_noise_response"]["status"], "pass")
        self.assertEqual(mechanisms["glare_saturation_response"]["status"], "pass")
        self.assertEqual(mechanisms["low_mtf_edge_confidence_response"]["status"], "pass")
        self.assertEqual(mechanisms["cfa_variant_support"]["status"], "pass")

        cases = {row["id"]: row for row in summary["cases"]}
        self.assertLess(cases["low_light_rggb"]["metrics"]["snr_map_mean"], cases["nominal_rggb"]["metrics"]["snr_map_mean"])
        self.assertGreater(cases["glare_rggb"]["metrics"]["saturation_mean"], cases["nominal_rggb"]["metrics"]["saturation_mean"])
        self.assertLess(cases["low_mtf_rggb"]["metrics"]["edge_confidence_mean"], cases["nominal_rggb"]["metrics"]["edge_confidence_mean"])

    def test_write_mechanism_validation_outputs_json_html_assets_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_mechanism_validation(width=80, height=48, cfa_patterns=("RGGB", "GRBG", "RCCB", "RGBIR"))
            html_path = write_mechanism_validation(summary, root / "report")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Mechanism Validation", html_path.read_text())
            self.assertTrue((html_path.parent / "mechanism_validation_summary.json").exists())
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

            persisted = json.loads((html_path.parent / "mechanism_validation_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertNotIn("_assets_source", persisted["cases"][0])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = mechanism_main(
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
            self.assertEqual(printed["failed_mechanisms"], [])
            self.assertTrue((root / "cli" / "mechanism_validation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
