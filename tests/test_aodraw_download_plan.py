from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.aodraw_download_plan import (
    build_aodraw_download_plan,
    main as aodraw_download_plan_main,
    write_aodraw_download_plan,
)


class AODRawDownloadPlanTest(unittest.TestCase):
    def test_build_plan_prioritizes_downsampled_srgb_and_subset_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_aodraw_download_plan(
                manifest=_manifest(),
                availability_summary=_availability(),
                dataset_root=root / "aodraw",
            )

            self.assertEqual(summary["status"], "ready_for_manual_download")
            self.assertEqual(summary["recommended_first"], "AODRaw images_downsampled_srgb 4.3GB via Baidu/TeraBox")
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["subset_raw_file_count"], 2)
            self.assertEqual(summary["subset_srgb_file_count"], 2)
            self.assertEqual(summary["missing_file_count"], 4)
            steps = {row["name"]: row for row in summary["download_steps"]}
            self.assertEqual(steps["Download downsampled sRGB directory"]["expected_size_gb"], 4.3)
            self.assertEqual(steps["Download selected downsampled RAW files if the file browser supports partial selection"]["status"], "recommended_if_partial_selection_available")
            self.assertIn("images_downsampled_raw/00000001.npy", summary["required_subset_files"])
            self.assertIn("aodraw_image_availability", summary["post_download_commands"][0])

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            availability_path = root / "availability.json"
            manifest_path.write_text(json.dumps(_manifest()))
            availability_path.write_text(json.dumps(_availability()))

            summary = build_aodraw_download_plan(
                manifest=manifest_path,
                availability_summary=availability_path,
                dataset_root=root / "aodraw",
            )
            html_path = write_aodraw_download_plan(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_download_plan_summary.json").exists())
            self.assertTrue((html_path.parent / "aodraw_manual_download_checklist.md").exists())
            self.assertTrue((html_path.parent / "aodraw_download_targets.txt").exists())
            self.assertIn("AODRaw Partial Download Plan", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_download_plan_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--availability-summary",
                        str(availability_path),
                        "--dataset-root",
                        str(root / "aodraw"),
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "ready_for_manual_download")
            self.assertTrue((root / "cli" / "aodraw_manual_download_checklist.md").exists())


def _manifest() -> list[dict]:
    return [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
        },
        {
            "image_id": 2,
            "file_name": "00000002.JPG",
            "expected_raw_relative_path": "images_downsampled_raw/00000002.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000002.JPG",
        },
    ]


def _availability() -> dict:
    return {
        "evaluation_ready": False,
        "missing_file_count": 4,
        "missing_files": [
            {"relative_path": "images_downsampled_raw/00000001.npy"},
            {"relative_path": "images_downsampled_srgb/00000001.JPG"},
            {"relative_path": "images_downsampled_raw/00000002.npy"},
            {"relative_path": "images_downsampled_srgb/00000002.JPG"},
        ],
    }


if __name__ == "__main__":
    unittest.main()
