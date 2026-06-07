from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.aodraw_download_watch import (
    main as aodraw_download_watch_main,
    watch_aodraw_downloads,
    write_aodraw_download_watch,
)


class AODRawDownloadWatchTest(unittest.TestCase):
    def test_watch_reports_waiting_when_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = watch_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(root / "Downloads",),
                dataset_root=root / "aodraw",
                max_iterations=1,
                interval_seconds=0.0,
            )

            self.assertEqual(summary["status"], "waiting_for_files")
            self.assertFalse(summary["evaluation_ready"])
            self.assertEqual(summary["missing_file_count"], 2)
            self.assertEqual(summary["iteration_count"], 1)

    def test_watch_imports_files_and_reaches_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            _write(downloads / "images_downsampled_srgb" / "00000001.JPG", b"srgb")
            _write(downloads / "images_downsampled_raw" / "00000001.npy", b"raw")

            summary = watch_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(downloads,),
                dataset_root=root / "aodraw",
                max_iterations=3,
                interval_seconds=0.0,
            )

            self.assertEqual(summary["status"], "ready")
            self.assertTrue(summary["evaluation_ready"])
            self.assertEqual(summary["missing_file_count"], 0)
            self.assertEqual(summary["iteration_count"], 1)
            self.assertTrue((root / "aodraw" / "images_downsampled_srgb" / "00000001.JPG").exists())

    def test_dry_run_does_not_import_even_when_sources_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            _write(downloads / "images_downsampled_srgb" / "00000001.JPG", b"srgb")
            _write(downloads / "images_downsampled_raw" / "00000001.npy", b"raw")

            summary = watch_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(downloads,),
                dataset_root=root / "aodraw",
                max_iterations=1,
                interval_seconds=0.0,
                dry_run=True,
            )

            self.assertEqual(summary["status"], "dry_run")
            self.assertFalse(summary["evaluation_ready"])
            self.assertEqual(summary["missing_file_count"], 2)
            self.assertFalse((root / "aodraw" / "images_downsampled_srgb" / "00000001.JPG").exists())

    def test_raw_only_watch_reaches_ready_without_srgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            _write(downloads / "images_downsampled_raw" / "00000001.npy", b"raw")

            summary = watch_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(downloads,),
                dataset_root=root / "aodraw",
                kind="raw",
                max_iterations=2,
                interval_seconds=0.0,
            )

            self.assertEqual(summary["status"], "ready")
            self.assertTrue(summary["evaluation_ready"])
            self.assertEqual(summary["missing_file_count"], 0)
            self.assertTrue((root / "aodraw" / "images_downsampled_raw" / "00000001.npy").exists())
            self.assertFalse((root / "aodraw" / "images_downsampled_srgb" / "00000001.JPG").exists())

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()))
            summary = watch_aodraw_downloads(
                manifest=manifest_path,
                download_roots=(root / "Downloads",),
                dataset_root=root / "aodraw",
                max_iterations=1,
                interval_seconds=0.0,
            )
            html_path = write_aodraw_download_watch(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_download_watch_summary.json").exists())
            self.assertIn("AODRaw Download Watch", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_download_watch_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--download-root",
                        str(root / "Downloads"),
                        "--dataset-root",
                        str(root / "aodraw"),
                        "--interval",
                        "0",
                        "--max-iterations",
                        "1",
                        "--output-dir",
                        str(root / "cli"),
                        "--availability-output-dir",
                        str(root / "availability"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "waiting_for_files")
            self.assertTrue((root / "availability" / "aodraw_image_availability_summary.json").exists())


def _manifest() -> list[dict]:
    return [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "selection_condition": "low_light",
            "box_count": 1,
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
        }
    ]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
