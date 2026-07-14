from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.evaluation.scene_information_stress import (
    build_scene_information_stress,
    main as scene_information_main,
    write_scene_information_stress,
)


class SceneInformationStressTest(unittest.TestCase):
    def test_build_scene_information_stress_passes_scene_oracle_checks(self) -> None:
        summary = build_scene_information_stress(sensor_width=96, sensor_height=56, oversample=6)

        self.assertEqual(summary["status"], "pass")
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["latent_high_frequency_detail_loss"]["status"], "pass")
        self.assertEqual(checks["cfa_chroma_alias_color_confidence_drop"]["status"], "pass")
        self.assertEqual(checks["subpixel_signal_fill_factor_loss"]["status"], "pass")

        cases = {row["id"]: row for row in summary["cases"]}
        thin = cases["supersampled_thin_detail"]["metrics"]
        self.assertGreater(thin["scene_luma_gradient_p90"], 0.20)
        self.assertLess(thin["luma_detail_retention_p90"], 0.20)
        chroma = cases["cfa_chroma_alias"]["metrics"]
        self.assertGreater(chroma["scene_chroma_gradient_p90"], 0.50)
        self.assertLess(chroma["color_confidence_mean"], 0.10)
        signal = cases["subpixel_signal"]["metrics"]
        self.assertGreater(signal["scene_signal_contrast"], 1.0)
        self.assertLess(signal["signal_contrast_retention"], 0.25)

    def test_write_scene_information_stress_outputs_json_html_assets_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_scene_information_stress(sensor_width=80, sensor_height=48, oversample=4)
            html_path = write_scene_information_stress(summary, root / "report")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Scene Information Stress", html_path.read_text())
            persisted = json.loads((html_path.parent / "scene_information_stress_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertNotIn("_assets_source", persisted["cases"][0])
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = scene_information_main(
                    [
                        "--sensor-width",
                        "80",
                        "--sensor-height",
                        "48",
                        "--oversample",
                        "4",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["failed_checks"], [])
            self.assertTrue((root / "cli" / "scene_information_stress_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
