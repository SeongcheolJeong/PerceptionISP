from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perception_isp.aodraw_acquisition import main as aodraw_acquisition_main
from perception_isp.aodraw_acquisition import run_aodraw_acquisition, write_aodraw_acquisition
from perception_isp.aodraw_download_plan import CLEANUP_CONFIRM_TOKEN


class AODRawAcquisitionTest(unittest.TestCase):
    def test_plan_mode_does_not_delete_or_open_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture(Path(tmp))

            with _patch_space(root):
                summary = run_aodraw_acquisition(
                    project_root=root,
                    manifest=root / "manifest.json",
                    availability_summary=root / "availability.json",
                    dataset_root=root / "data/raw_datasets/aodraw",
                    download_roots=(root / "Downloads",),
                    output_dir=root / "report",
                )

            self.assertEqual(summary["status"], "planned")
            self.assertEqual(summary["cleanup"]["status"], "dry_run")
            self.assertEqual(summary["download_open"]["status"], "skipped")
            self.assertEqual(summary["watch"]["status"], "skipped")
            self.assertTrue((root / "data/raw_datasets/pascalraw_full_archive/PASCALRAW.tar.gzaa").exists())

    def test_cleanup_without_token_blocks_before_open_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture(Path(tmp))
            calls = []

            with _patch_space(root):
                summary = run_aodraw_acquisition(
                    project_root=root,
                    manifest=root / "manifest.json",
                    availability_summary=root / "availability.json",
                    dataset_root=root / "data/raw_datasets/aodraw",
                    download_roots=(root / "Downloads",),
                    execute_cleanup=True,
                    confirm_token="wrong",
                    open_download=True,
                    output_dir=root / "report",
                    opener=lambda command: calls.append(command) or {"returncode": 0, "stdout": "", "stderr": ""},
                )

            self.assertEqual(summary["status"], "blocked_cleanup_confirmation")
            self.assertEqual(summary["cleanup"]["status"], "blocked_confirmation_required")
            self.assertEqual(summary["download_open"]["status"], "blocked_disk")
            self.assertFalse(calls)

    def test_confirmed_cleanup_can_open_download_with_injected_opener(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture(Path(tmp))
            calls = []

            with _patch_space(root):
                summary = run_aodraw_acquisition(
                    project_root=root,
                    manifest=root / "manifest.json",
                    availability_summary=root / "availability.json",
                    dataset_root=root / "data/raw_datasets/aodraw",
                    download_roots=(root / "Downloads",),
                    execute_cleanup=True,
                    confirm_token=CLEANUP_CONFIRM_TOKEN,
                    open_download=True,
                    output_dir=root / "report",
                    opener=lambda command: calls.append(tuple(command)) or {"returncode": 0, "stdout": "", "stderr": ""},
                )

            self.assertEqual(summary["cleanup"]["status"], "executed")
            self.assertEqual(summary["download_open"]["status"], "opened")
            self.assertEqual(calls[0][0:3], ("open", "-a", "Safari"))
            self.assertFalse((root / "data/raw_datasets/pascalraw_full_archive").exists())

    def test_watch_dry_run_keeps_missing_raw_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture(Path(tmp))

            with _patch_space(root):
                summary = run_aodraw_acquisition(
                    project_root=root,
                    manifest=root / "manifest.json",
                    availability_summary=root / "availability.json",
                    dataset_root=root / "data/raw_datasets/aodraw",
                    download_roots=(root / "Downloads",),
                    watch=True,
                    dry_run=True,
                    watch_iterations=1,
                    output_dir=root / "report",
                )

            self.assertEqual(summary["status"], "waiting_for_download")
            self.assertEqual(summary["watch"]["status"], "dry_run")
            self.assertEqual(summary["watch"]["missing_file_count"], 1)

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture(Path(tmp))
            with _patch_space(root):
                summary = run_aodraw_acquisition(
                    project_root=root,
                    manifest=root / "manifest.json",
                    availability_summary=root / "availability.json",
                    dataset_root=root / "data/raw_datasets/aodraw",
                    download_roots=(root / "Downloads",),
                    output_dir=root / "report",
                )
            html_path = write_aodraw_acquisition(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_acquisition_summary.json").exists())
            self.assertIn("AODRaw Acquisition Runner", html_path.read_text())

            stdout = io.StringIO()
            with _patch_space(root):
                with contextlib.redirect_stdout(stdout):
                    exit_code = aodraw_acquisition_main(
                        [
                            "--project-root",
                            str(root),
                            "--manifest",
                            str(root / "manifest.json"),
                            "--availability-summary",
                            str(root / "availability.json"),
                            "--dataset-root",
                            str(root / "data/raw_datasets/aodraw"),
                            "--download-root",
                            str(root / "Downloads"),
                            "--output-dir",
                            str(root / "cli"),
                            "--dry-run",
                        ]
                    )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "planned")
            self.assertTrue((root / "cli" / "aodraw_acquisition_summary.json").exists())


def _fixture(root: Path) -> Path:
    _write(root / "data/raw_datasets/pascalraw_full_archive/PASCALRAW.tar.gzaa", b"x" * 2 * 1024 * 1024)
    _write(root / "data/raw_datasets/pascalraw_full_extract/PASCALRAW/original/raw/2014_000001.NEF", b"raw")
    _write(root / "data/raw_datasets/pascalraw_full_extract/PASCALRAW/original/jpg/2014_000001.jpg", b"jpg")
    _write(root / "data/raw_datasets/aodraw/annotations/.keep", b"")
    _write(root / "Downloads/.keep", b"")
    manifest = [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
        }
    ]
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "availability.json").write_text(json.dumps({"evaluation_ready": False, "missing_file_count": 1, "missing_files": []}))
    return root


def _patch_space(root: Path):
    def disk_summary(_path: Path) -> dict:
        archive = root / "data/raw_datasets/pascalraw_full_archive"
        available = 55.0 if archive.exists() else 72.0
        return {
            "path": str(root),
            "total_gib": 256.0,
            "used_gib": 256.0 - available,
            "available_gib": available,
            "fits_downsampled_srgb_4p3gb": True,
            "fits_downsampled_raw_test_zip_58p9gb": True,
            "fits_downsampled_raw_test_zip_with_headroom": available >= 64.89,
            "fits_downsampled_raw_train_zip_137p2gb": False,
            "fits_downsampled_raw_all_zips_196p1gb": False,
            "fits_downsampled_raw_223gb": False,
            "fits_downsampled_srgb_plus_raw_test_zip": available >= 63.24,
            "raw_test_zip_download_headroom_gib": 10.0,
        }

    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch("perception_isp.aodraw_download_plan._disk_summary", side_effect=disk_summary))
    stack.enter_context(mock.patch("perception_isp.aodraw_storage_cleanup._disk_summary", side_effect=disk_summary))
    stack.enter_context(mock.patch("perception_isp.aodraw_download_plan._disk_usage_bytes", return_value=int(110 * 1024**3)))
    return stack


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
