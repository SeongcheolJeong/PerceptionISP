from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from perception_isp.core.aux_dnn import RGB_AUX_CHANNELS, RGB_AUX_EXTENDED_CHANNELS
from perception_isp.core.synthetic import make_synthetic_raw
from perception_isp.evaluation.example_suite import build_example_suite, main, write_example_suite


class ExampleSuiteTest(unittest.TestCase):
    def test_default_suite_passes_and_skips_unrequested_camerae2e(self) -> None:
        summary = build_example_suite(width=96, height=64, seed=7)

        self.assertEqual(summary["status"], "pass")
        sections = {row["id"]: row for row in summary["sections"]}
        for section_id in ("hdr", "metadata", "calibration", "cfa-optics", "temporal", "dnn-contract"):
            self.assertEqual(sections[section_id]["status"], "pass")
        self.assertEqual(sections["camerae2e"]["status"], "skip")

        metadata_statuses = {row["status"] for row in sections["metadata"]["support_matrix"]}
        self.assertEqual(metadata_statuses, {"active processing", "propagated only", "declared but unused"})
        support_by_field = {row["field"]: row["status"] for row in sections["metadata"]["support_matrix"]}
        self.assertEqual(support_by_field["timestamp_us"], "active processing")
        self.assertEqual(support_by_field["frame_counter"], "propagated only")
        self.assertEqual(support_by_field["camera_id"], "propagated only")
        hdr_checks = {row["id"]: row["status"] for row in sections["hdr"]["checks"]}
        self.assertEqual(hdr_checks["bracket_recovers_highlights"], "pass")
        self.assertEqual(hdr_checks["motion_raises_ghost_risk"], "pass")

    def test_dnn_contract_reports_exact_channel_sets(self) -> None:
        summary = build_example_suite(width=80, height=48, seed=11, case_groups=("dnn-contract",))
        section = next(row for row in summary["sections"] if row["id"] == "dnn-contract")
        metrics = section["cases"][0]["metrics"]

        self.assertEqual(metrics["stable_channels"], list(RGB_AUX_CHANNELS))
        self.assertEqual(metrics["extended_channels"], list(RGB_AUX_EXTENDED_CHANNELS))
        self.assertFalse(metrics["scalar_metadata_in_tensor"])

    def test_seeded_hdr_metrics_are_deterministic(self) -> None:
        first = build_example_suite(width=80, height=48, seed=17, case_groups=("hdr",))
        second = build_example_suite(width=80, height=48, seed=17, case_groups=("hdr",))
        first_hdr = next(row for row in first["sections"] if row["id"] == "hdr")
        second_hdr = next(row for row in second["sections"] if row["id"] == "hdr")

        self.assertEqual(first_hdr["checks"], second_hdr["checks"])
        self.assertEqual(first_hdr["cases"][2]["metrics"], second_hdr["cases"][2]["metrics"])

    def test_write_suite_outputs_tabbed_html_json_and_png_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_example_suite(width=80, height=48, seed=23, case_groups=("metadata", "dnn-contract"))
            html_path = write_example_suite(summary, Path(tmp) / "report")
            persisted = json.loads((html_path.parent / "summary.json").read_text())
            html = html_path.read_text()

            self.assertEqual(persisted["status"], "pass")
            self.assertNotIn("_assets_source", json.dumps(persisted))
            self.assertIn("data-tab=\"overview\"", html)
            self.assertIn("data-tab=\"calibration-optics\"", html)
            self.assertNotIn("data-tab=\"cfa-optics\"", html)
            self.assertIn("Sensor Metadata", html)
            self.assertIn("DNN Contract", html)
            self.assertEqual(html.count('class="tab-button'), 7)
            self.assertIn('loading="lazy"', html)
            self.assertIn('data-src="assets/', html)
            self.assertIn("image.src = image.dataset.src", html)
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

    def test_cli_lists_cases_and_runs_selected_group(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--list-cases"]), 0)
        self.assertIn("hdr", stdout.getvalue())
        self.assertIn("dnn-contract", stdout.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--case",
                        "dnn-contract",
                        "--width",
                        "80",
                        "--height",
                        "48",
                        "--output-dir",
                        str(Path(tmp) / "cli"),
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "pass")
            self.assertTrue(Path(payload["report"]).exists())

    def test_required_camerae2e_failure_is_not_replaced_by_synthetic_fallback(self) -> None:
        with mock.patch(
            "perception_isp.evaluation.example_suite.raw_from_camerae2e",
            side_effect=RuntimeError("CameraE2E unavailable"),
        ):
            summary = build_example_suite(
                width=80,
                height=48,
                seed=5,
                case_groups=("dnn-contract",),
                with_camerae2e=True,
            )

        camera = next(row for row in summary["sections"] if row["id"] == "camerae2e")
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(camera["status"], "fail")
        self.assertIn("CameraE2E unavailable", camera["checks"][0]["evidence"])

    def test_required_camerae2e_success_records_native_provenance(self) -> None:
        raw = make_synthetic_raw(width=80, height=48, cfa_pattern="GRBG")
        raw.provenance = {
            "bridge": "camerae2e",
            "camerae2e_used": True,
            "raw_source_key": "sensor.volts",
            "true_sensor_cfa_mosaic": True,
            "source_cfa_pattern": "GRBG",
            "target_cfa_pattern": "GRBG",
            "pattern_remapped": False,
            "source_native_hw": [96, 96],
            "target_shape": [48, 80],
            "native_resolution_at_least_target": True,
        }
        with mock.patch("perception_isp.evaluation.example_suite.raw_from_camerae2e", return_value=raw) as bridge:
            summary = build_example_suite(
                width=80,
                height=48,
                seed=5,
                case_groups=("dnn-contract",),
                with_camerae2e=True,
                scene_name="uniform ee",
            )

        bridge.assert_called_once_with(scene_name="uniform ee", width=80, height=48, cfa_pattern="auto")
        camera = next(row for row in summary["sections"] if row["id"] == "camerae2e")
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(camera["status"], "pass")
        self.assertTrue(all(row["status"] == "pass" for row in camera["checks"]))


if __name__ == "__main__":
    unittest.main()
