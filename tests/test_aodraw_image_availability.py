from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.datasets.aodraw_image_availability import (
    build_aodraw_image_availability,
    main as aodraw_image_availability_main,
    write_aodraw_image_availability,
)


class AODRawImageAvailabilityTest(unittest.TestCase):
    def test_build_audit_passes_when_required_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest()
            _write_required_files(root, manifest)

            summary = build_aodraw_image_availability(manifest, dataset_root=root)

            self.assertEqual(summary["status"], "pass")
            self.assertTrue(summary["evaluation_ready"])
            self.assertEqual(summary["required_file_count"], 4)
            self.assertEqual(summary["missing_file_count"], 0)
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["raw_files_available"], "pass")
            self.assertEqual(checks["srgb_files_available"], "pass")

    def test_build_audit_blocks_when_images_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest()
            _touch(root / "images_downsampled_srgb" / "00000001.JPG")

            summary = build_aodraw_image_availability(manifest, dataset_root=root)

            self.assertEqual(summary["status"], "blocked")
            self.assertFalse(summary["evaluation_ready"])
            self.assertEqual(summary["required_file_count"], 4)
            self.assertEqual(summary["available_file_count"], 1)
            self.assertEqual(summary["missing_raw_count"], 2)
            self.assertEqual(summary["missing_srgb_count"], 1)
            missing = {row["relative_path"] for row in summary["missing_files"]}
            self.assertIn("images_downsampled_raw/00000001.npy", missing)
            self.assertIn("images_downsampled_raw/00000002.npy", missing)
            self.assertIn("images_downsampled_srgb/00000002.JPG", missing)

    def test_raw_only_audit_passes_without_srgb_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest()
            for row in manifest:
                _touch(root / row["expected_raw_relative_path"])

            summary = build_aodraw_image_availability(manifest, dataset_root=root, kind="raw")

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["kind"], "raw")
            self.assertTrue(summary["evaluation_ready"])
            self.assertEqual(summary["required_file_count"], 2)
            self.assertEqual(summary["missing_srgb_count"], 0)
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["raw_files_available"], "pass")
            self.assertEqual(checks["srgb_files_available"], "pass")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()))
            summary = build_aodraw_image_availability(manifest_path, dataset_root=root / "dataset")

            html_path = write_aodraw_image_availability(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_image_availability_summary.json").exists())
            self.assertTrue((html_path.parent / "aodraw_required_files.txt").exists())
            self.assertTrue((html_path.parent / "aodraw_missing_files.txt").exists())
            self.assertIn("AODRaw Image Availability Audit", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_image_availability_main(
                    [
                        str(manifest_path),
                        "--dataset-root",
                        str(root / "dataset"),
                        "--kind",
                        "raw",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(printed["evaluation_ready"])
            self.assertEqual(printed["missing_file_count"], 2)
            self.assertTrue((root / "cli" / "aodraw_missing_files.json").exists())


def _manifest() -> list[dict]:
    return [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "selection_condition": "low_light",
            "box_count": 2,
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
        },
        {
            "image_id": 2,
            "file_name": "00000002.JPG",
            "selection_condition": "rain",
            "box_count": 1,
            "expected_raw_relative_path": "images_downsampled_raw/00000002.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000002.JPG",
        },
    ]


def _write_required_files(root: Path, manifest: list[dict]) -> None:
    for row in manifest:
        _touch(root / row["expected_raw_relative_path"])
        _touch(root / row["expected_srgb_relative_path"])


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unit-test")


if __name__ == "__main__":
    unittest.main()
