from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from perception_isp.datasets.aodraw_download_importer import (
    import_aodraw_downloads,
    main as aodraw_download_importer_main,
    write_aodraw_download_import,
)


class AODRawDownloadImporterTest(unittest.TestCase):
    def test_import_from_directory_copies_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            dataset = root / "aodraw"
            _write(downloads / "bundle" / "images_downsampled_srgb" / "00000001.JPG", b"srgb")
            _write(downloads / "bundle" / "images_downsampled_raw" / "00000001.npy", b"raw")

            summary = import_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(downloads,),
                dataset_root=dataset,
                dry_run=False,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["imported_file_count"], 2)
            self.assertEqual(summary["missing_file_count"], 0)
            self.assertEqual((dataset / "images_downsampled_srgb" / "00000001.JPG").read_bytes(), b"srgb")
            self.assertEqual((dataset / "images_downsampled_raw" / "00000001.npy").read_bytes(), b"raw")

    def test_dry_run_resolves_zip_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "aodraw.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("AODRaw/images_downsampled_srgb/00000001.JPG", b"srgb")
                z.writestr("AODRaw/images_downsampled_raw/00000001.npy", b"raw")

            summary = import_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(archive,),
                dataset_root=root / "aodraw",
                dry_run=True,
            )

            self.assertEqual(summary["status"], "ready_to_import")
            self.assertEqual(summary["would_import_file_count"], 2)
            self.assertFalse((root / "aodraw" / "images_downsampled_srgb" / "00000001.JPG").exists())

    def test_import_from_zip_extracts_required_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "aodraw.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("AODRaw/images_downsampled_srgb/00000001.JPG", b"srgb")
                z.writestr("AODRaw/images_downsampled_raw/00000001.npy", b"raw")
                z.writestr("AODRaw/images_downsampled_srgb/unused.JPG", b"unused")

            summary = import_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(archive,),
                dataset_root=root / "aodraw",
                dry_run=False,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["imported_file_count"], 2)
            self.assertFalse((root / "aodraw" / "images_downsampled_srgb" / "unused.JPG").exists())
            self.assertEqual((root / "aodraw" / "images_downsampled_raw" / "00000001.npy").read_bytes(), b"raw")

    def test_missing_files_block_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = import_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(Path(tmp) / "empty",),
                dataset_root=Path(tmp) / "aodraw",
                dry_run=False,
            )

            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(summary["missing_file_count"], 2)

    def test_raw_only_import_command_uses_raw_availability_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            _write(downloads / "images_downsampled_raw" / "00000001.npy", b"raw")

            summary = import_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(downloads,),
                dataset_root=root / "aodraw",
                kind="raw",
                dry_run=True,
            )

            self.assertEqual(summary["status"], "ready_to_import")
            self.assertEqual(summary["required_file_count"], 1)
            self.assertIn("--kind raw", summary["post_import_command"])
            self.assertIn("raw_only", summary["post_import_command"])

    def test_bad_zip_is_reported_as_source_scan_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "AODRaw_downsampled_srgb.zip"
            archive.write_bytes(b"not a complete zip")

            summary = import_aodraw_downloads(
                manifest=_manifest(),
                download_roots=(archive,),
                dataset_root=root / "aodraw",
                dry_run=True,
            )

            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(summary["missing_file_count"], 2)
            self.assertEqual(summary["source_scan_error_count"], 1)
            self.assertIn("BadZipFile", summary["source_scan_errors"][0]["error"])

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()))
            downloads = root / "Downloads"
            _write(downloads / "images_downsampled_srgb" / "00000001.JPG", b"srgb")
            _write(downloads / "images_downsampled_raw" / "00000001.npy", b"raw")

            summary = import_aodraw_downloads(
                manifest=manifest_path,
                download_roots=(downloads,),
                dataset_root=root / "aodraw",
                dry_run=True,
            )
            html_path = write_aodraw_download_import(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_download_import_summary.json").exists())
            self.assertIn("AODRaw Download Import", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_download_importer_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--download-root",
                        str(downloads),
                        "--dataset-root",
                        str(root / "cli_aodraw"),
                        "--dry-run",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "ready_to_import")
            self.assertEqual(printed["resolved_file_count"], 2)


def _manifest() -> list[dict]:
    return [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
        }
    ]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
