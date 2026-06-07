from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perception_isp.aodraw_download_plan import (
    build_aodraw_download_plan,
    main as aodraw_download_plan_main,
    write_aodraw_download_plan,
)


class AODRawDownloadPlanTest(unittest.TestCase):
    def test_build_plan_prioritizes_downsampled_srgb_and_subset_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("perception_isp.aodraw_download_plan._disk_summary", return_value=_disk(fits_raw_headroom=True)):
                summary = build_aodraw_download_plan(
                    manifest=_manifest(),
                    availability_summary=_availability(),
                    dataset_root=root / "aodraw",
                )

            self.assertEqual(summary["status"], "ready_for_manual_download")
            self.assertEqual(summary["recommended_first"], "AODRaw_test_downsampled_raw.zip 58.94GB via Baidu")
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["subset_raw_file_count"], 2)
            self.assertEqual(summary["subset_srgb_file_count"], 2)
            self.assertEqual(summary["missing_file_count"], 4)
            steps = {row["name"]: row for row in summary["download_steps"]}
            self.assertEqual(steps["Download AODRaw test downsampled RAW zip"]["expected_size_gb"], 58.94)
            self.assertEqual(steps["Download AODRaw test downsampled RAW zip"]["archive_filename"], "AODRaw_test_downsampled_raw.zip")
            self.assertEqual(steps["Download AODRaw test downsampled RAW zip"]["status"], "recommended_now")
            self.assertEqual(steps["Download downsampled sRGB zip or directory"]["expected_size_gb"], 4.3)
            self.assertEqual(steps["Download selected downsampled RAW files if the file browser supports partial selection"]["status"], "recommended_if_partial_selection_available")
            self.assertEqual(summary["downsampled_raw_archives"]["train"]["filename"], "AODRaw_train_downsampled_raw.zip")
            self.assertIn("images_downsampled_raw/00000001.npy", summary["required_subset_files"])
            self.assertIn("--kind raw", summary["post_download_commands"][0])
            self.assertIn("aodraw_image_availability", summary["post_download_commands"][0])

    def test_build_plan_blocks_when_raw_zip_has_no_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("perception_isp.aodraw_download_plan._disk_summary", return_value=_disk(fits_raw_headroom=False)):
                summary = build_aodraw_download_plan(
                    manifest=_manifest(),
                    availability_summary=_availability(),
                    dataset_root=root / "aodraw",
                )

            self.assertEqual(summary["status"], "blocked")
            self.assertIn("Free at least 10 GiB more headroom", summary["recommended_first"])
            steps = {row["name"]: row for row in summary["download_steps"]}
            self.assertEqual(steps["Download AODRaw test downsampled RAW zip"]["status"], "disk_blocked")
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["sufficient_disk_for_test_raw_zip"], "pass")
            self.assertEqual(checks["sufficient_disk_for_test_raw_zip_with_headroom"], "fail")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            availability_path = root / "availability.json"
            manifest_path.write_text(json.dumps(_manifest()))
            availability_path.write_text(json.dumps(_availability()))

            with mock.patch("perception_isp.aodraw_download_plan._disk_summary", return_value=_disk(fits_raw_headroom=True)):
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
            with mock.patch("perception_isp.aodraw_download_plan._disk_summary", return_value=_disk(fits_raw_headroom=True)):
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


def _disk(*, fits_raw_headroom: bool) -> dict:
    return {
        "path": "/tmp",
        "total_gib": 256.0,
        "used_gib": 128.0,
        "available_gib": 72.0 if fits_raw_headroom else 55.0,
        "fits_downsampled_srgb_4p3gb": True,
        "fits_downsampled_raw_test_zip_58p9gb": True,
        "fits_downsampled_raw_test_zip_with_headroom": bool(fits_raw_headroom),
        "fits_downsampled_raw_train_zip_137p2gb": False,
        "fits_downsampled_raw_all_zips_196p1gb": False,
        "fits_downsampled_raw_223gb": False,
        "fits_downsampled_srgb_plus_raw_test_zip": bool(fits_raw_headroom),
        "raw_test_zip_download_headroom_gib": 10.0,
    }


if __name__ == "__main__":
    unittest.main()
