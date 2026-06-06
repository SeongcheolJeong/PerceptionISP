from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.cfa_stress_sweep import build_cfa_stress_sweep, main as cfa_sweep_main, write_cfa_stress_sweep


class CFAStressSweepTest(unittest.TestCase):
    def test_build_cfa_stress_sweep_ranks_each_condition(self) -> None:
        summary = build_cfa_stress_sweep(
            width=80,
            height=48,
            cfa_patterns=("RGGB", "RCCB", "RGBIR", "MONO"),
            conditions=("nominal_hdr", "low_light", "glare", "low_mtf"),
        )

        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["support"]["all_finite"])
        self.assertTrue(summary["support"]["all_supported"])
        self.assertEqual(len(summary["cases"]), 16)
        rankings = {row["condition"]: row for row in summary["condition_rankings"]}
        self.assertEqual(set(rankings), {"nominal_hdr", "low_light", "glare", "low_mtf"})
        for ranking in rankings.values():
            scores = [row["condition_score"] for row in ranking["ranked_cfas"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

        self.assertEqual(rankings["low_light"]["ranked_cfas"][0]["cfa_pattern"], "MONO")
        self.assertEqual(rankings["glare"]["ranked_cfas"][0]["cfa_pattern"], "RGBIR")
        self.assertEqual(rankings["low_mtf"]["ranked_cfas"][0]["cfa_pattern"], "MONO")

    def test_write_cfa_stress_sweep_outputs_json_html_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_cfa_stress_sweep(width=64, height=40, cfa_patterns=("RGGB", "RCCB"), conditions=("low_light", "glare"))
            html_path = write_cfa_stress_sweep(summary, root / "report")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP CFA Stress Sweep", html_path.read_text())
            persisted = json.loads((html_path.parent / "cfa_stress_sweep_summary.json").read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertEqual(len(persisted["cases"]), 4)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cfa_sweep_main(
                    [
                        "--width",
                        "64",
                        "--height",
                        "40",
                        "--cfa",
                        "RGGB",
                        "--cfa",
                        "RCCB",
                        "--condition",
                        "low_light",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["case_count"], 2)
            self.assertTrue((root / "cli" / "cfa_stress_sweep_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
